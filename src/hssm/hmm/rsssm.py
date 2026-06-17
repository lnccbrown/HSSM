"""RSSSM: Regime-Switching Sequential Sampling Model.

``RSSSM`` fits regime-switching SSMs through the same user-facing pattern as
``HSSM`` / ``RLSSM``.  Each trial belongs to one of ``K`` hidden regimes that
evolve as a Markov chain; within a regime the ``(rt, response)`` emission is a
standard SSM with regime-specific *switching* parameters and shared values for
the rest.  The discrete regimes are marginalised out by the forward algorithm
and contributed as a single scalar ``pm.Potential`` (design doc §3.4).

Unlike ``HSSM`` / ``RLSSM``, ``RSSSM`` builds the PyMC model **directly** rather
than through bambi: the HMM's defining latents (the transition matrix, the
regime-indexed parameter vectors) are not row-indexed quantities bambi's formula
system can declare (decision 10.1.8).  ``RSSSM`` therefore subclasses
``HSSMBase`` only for its non-bambi surface (save/load arg capture, post-fit
helpers) and overrides ``__init__``, ``sample``, the ``pymc_model`` property,
``graph``, ``vi``, and ``log_likelihood``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, cast

import bambi as bmb
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.distributions.transforms import ordered as ordered_transform

from ..base import HSSMBase
from ..config import BaseModelConfig
from ..defaults import default_model_config
from .config import RSSSMConfig
from .likelihoods.builder import make_hmm_logp_op
from .likelihoods.emissions import resolve_emission_dist
from .ordering import resolve_anchor
from .specs import (
    DirichletInitialDistribution,
    FixedInitialDistribution,
    NoPooling,
    StickyDirichlet,
    UniformInitialDistribution,
    resolve_initial_distribution,
    resolve_ordering,
    resolve_pooling,
    resolve_transition_prior,
)
from .utils import pad_and_align_to_T_max

if TYPE_CHECKING:
    import arviz as az
    import pandas as pd

    from .._types import LoglikKind, SupportedModels

_logger = logging.getLogger("hssm")


def _bounds_based_prior(bounds: tuple[float, float] | None) -> dict[str, Any]:
    """Return a sensible default prior dict given parameter bounds."""
    if bounds is None:
        return {"name": "Normal", "mu": 0.0, "sigma": 2.0}
    lo, hi = bounds
    lo_inf, hi_inf = np.isinf(lo), np.isinf(hi)
    if lo_inf and hi_inf:
        return {"name": "Normal", "mu": 0.0, "sigma": 2.0}
    if not lo_inf and hi_inf:  # [lo, inf)
        return {"name": "HalfNormal", "sigma": 2.0}
    if lo_inf and not hi_inf:  # (-inf, hi]
        return {"name": "Normal", "mu": hi - 1.0, "sigma": 2.0}
    return {"name": "Uniform", "lower": float(lo), "upper": float(hi)}


def _ascending_initval(K: int, bounds: tuple[float, float] | None) -> np.ndarray:
    """Return a strictly-ascending in-support init vector for the anchor."""
    if bounds is None:
        return np.linspace(-2.0, 2.0, K)
    lo, hi = bounds
    lo_inf, hi_inf = np.isinf(lo), np.isinf(hi)
    if lo_inf and hi_inf:
        return np.linspace(-2.0, 2.0, K)
    if not lo_inf and hi_inf:
        return lo + 0.5 + np.arange(K) * 0.5
    if lo_inf and not hi_inf:
        return hi - 0.5 - np.arange(K)[::-1] * 0.5
    return np.linspace(lo, hi, K + 2)[1:-1]


class RSSSM(HSSMBase):
    """Regime-Switching Sequential Sampling Model.

    Parameters
    ----------
    data
        Long-format trial data, sorted by ``(participant, trial)``.  Panels may
        be unbalanced.  Must contain the emission columns (``rt``, ``response``)
        and, for multi-participant data, ``participant_col``.
    model
        SSM identifier (e.g. ``"ddm"``) or a pre-built ``BaseModelConfig``.
    K
        Number of hidden regimes (``>= 2``).
    switching_params
        SSM parameters inferred per regime.
    model_config
        Pre-built ``RSSSMConfig`` (advanced path).  Mutually exclusive with the
        granular args above.
    transition_prior, initial_distribution, ordering, pooling
        Spec inputs (dataclass, HSSM-style dict, or shorthand) for the Markov
        chain structure and label-switching.
    loglik_kind
        ``"analytical"`` (Phase 2) or ``"approx_differentiable"`` (Phase 3).
    participant_col
        Column identifying participants.  Synthesised as a constant when absent.
    **kwargs
        Per-parameter input specs (the three-mode rule): ``v=0.5`` shares a
        scalar, ``v=[0.5, 1.5]`` fixes per regime, ``v={"name": "Normal", ...}``
        infers with that prior.
    """

    # Narrow the inherited attribute type: RSSSM always stores an RSSSMConfig.
    model_config: RSSSMConfig

    def __init__(
        self,
        data: pd.DataFrame,
        model: str | BaseModelConfig | None = None,
        K: int | None = None,
        switching_params: list[str] | None = None,
        *,
        model_config: RSSSMConfig | None = None,
        transition_prior: Any = None,
        initial_distribution: Any = None,
        loglik_kind: "LoglikKind" | None = None,
        backend: Literal["jax", "pytensor"] | None = None,
        participant_col: str = "participant_id",
        ordering: Any = None,
        pooling: Any = None,
        missing_data: bool | float = False,
        deadline: bool | str = False,
        p_outlier: float | dict | bmb.Prior | None = None,
        lapse: float | dict | bmb.Prior | None = None,
        **kwargs: Any,
    ) -> None:
        # ===== save/load serialisation =====
        self._init_args = self._store_init_args(locals(), kwargs)

        # ===== minimal HSSMBase state (we bypass HSSMBase.__init__) =====
        self.data = data.copy()
        self._inference_obj: az.InferenceData | None = None
        self._inference_obj_vi: pm.Approximation | None = None
        self._vi_approx = None
        self._map_dict = None
        self._initvals: dict[str, Any] = {}

        # ===== reject incompatible inherited kwargs (decision 10.1.9) =====
        self._reject_unsupported_kwargs(missing_data, deadline, p_outlier, lapse)

        # ===== resolve the RSSSMConfig =====
        if model_config is not None:
            if any(x is not None for x in (model, K, switching_params)):
                raise ValueError(
                    "Pass either `model_config` or the granular args "
                    "(`model`, `K`, `switching_params`, ...), not both."
                )
            self.model_config = model_config
        else:
            self.model_config = self._build_config(
                model=model,
                K=K,
                switching_params=switching_params,
                transition_prior=transition_prior,
                initial_distribution=initial_distribution,
                loglik_kind=loglik_kind,
                backend=backend,
                ordering=ordering,
                pooling=pooling,
                param_specs=kwargs,
            )

        self.model_config.validate()

        cfg = self.model_config
        self.K = cfg.K
        self.switching_params = list(cfg.switching_params)
        self.list_params = list(cfg.list_params)  # type: ignore[arg-type]
        self.bounds = dict(cfg.bounds)
        self.response = (
            list(cfg.response) if cfg.response is not None else ["rt", "response"]
        )
        self.loglik_kind = cfg.loglik_kind
        self.model_name = cfg.model_name

        # ===== resolve participant column =====
        if participant_col not in self.data.columns:
            self.data[participant_col] = 0
            _logger.info(
                "No participant column found; treating all rows as a single "
                "participant."
            )
        self.participant_col = participant_col

        # ===== pad / align the panel =====
        (
            self._data_padded,
            self._mask,
            self.n_participants,
            self.n_trials,
        ) = pad_and_align_to_T_max(self.data, participant_col, self.response)

        # ===== resolve the emission distribution (L2) =====
        self._emission_dist = resolve_emission_dist(
            model=cfg.model if isinstance(cfg.model, str) else cfg.model_name,
            loglik_kind=cfg.loglik_kind,  # type: ignore[arg-type]
            backend=cfg.backend,
        )

        # ===== build the PyMC model directly =====
        self._pymc_model_obj = self._build_pymc_model()
        _logger.info("RSSSM model initialized successfully.")

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reject_unsupported_kwargs(missing_data, deadline, p_outlier, lapse) -> None:
        """Raise on inherited kwargs unsupported in v1 (decision 10.1.9)."""
        if missing_data is not False:
            raise NotImplementedError(
                "RSSSM does not support `missing_data`: re-ordering rows for "
                "missing RTs would corrupt the trial-axis Markov structure. "
                "Remove missing trials before passing data to RSSSM."
            )
        if deadline is not False:
            raise NotImplementedError(
                "RSSSM does not support `deadline`: re-ordering rows would "
                "corrupt the trial-axis Markov structure."
            )
        if p_outlier is not None:
            raise NotImplementedError(
                "RSSSM rejects the top-level `p_outlier` kwarg: a single global "
                "iid lapse mixture double-models the lapse regime. Model lapses "
                "structurally via a regime, or pass `p_outlier` as a per-regime "
                "parameter (in `switching_params` or a length-K list)."
            )
        if lapse is not None:
            raise NotImplementedError(
                "RSSSM rejects the top-level `lapse` kwarg (decision 10.1.9)."
            )

    def _build_config(
        self,
        *,
        model,
        K,
        switching_params,
        transition_prior,
        initial_distribution,
        loglik_kind,
        backend,
        ordering,
        pooling,
        param_specs,
    ) -> RSSSMConfig:
        """Assemble an ``RSSSMConfig`` from the granular constructor args."""
        if model is None:
            raise ValueError('`model` (e.g. "ddm") must be provided.')
        if K is None:
            raise ValueError("`K` (number of regimes) must be provided.")

        model_name = (
            model if isinstance(model, str) else getattr(model, "model_name", "rsssm")
        )

        # Resolve SSM parameter metadata (list_params, bounds) from defaults.
        list_params, bounds = self._resolve_ssm_param_meta(model, loglik_kind)

        resolved_loglik_kind = cast("LoglikKind", loglik_kind or "analytical")
        resolved_backend = backend
        if resolved_loglik_kind == "approx_differentiable" and resolved_backend is None:
            resolved_backend = "jax"

        return RSSSMConfig(
            model_name=f"rsssm_{model_name}",
            model=model,
            K=K,
            switching_params=list(switching_params or []),
            list_params=list_params,
            bounds=bounds,
            loglik_kind=resolved_loglik_kind,
            backend=resolved_backend,
            transition_prior=resolve_transition_prior(transition_prior),
            initial_distribution=resolve_initial_distribution(initial_distribution),
            ordering=resolve_ordering(ordering),
            pooling=resolve_pooling(pooling),
            param_specs=dict(param_specs),
        )

    @staticmethod
    def _resolve_ssm_param_meta(
        model, loglik_kind
    ) -> tuple[list[str], dict[str, tuple[float, float]]]:
        """Return (list_params, bounds) for the SSM emission model."""
        if isinstance(model, BaseModelConfig):
            if model.list_params is None:
                raise ValueError("Provided model config has no `list_params`.")
            return list(model.list_params), dict(model.bounds)

        if model not in default_model_config:
            raise ValueError(
                f"Unknown model {model!r}; provide a model config instead, or use "
                "a supported SSM."
            )
        cfg = default_model_config[cast("SupportedModels", model)]
        list_params = list(cfg["list_params"])
        # Pull bounds from the analytical likelihood config when available.
        kind = cast("LoglikKind", loglik_kind or "analytical")
        likelihoods = cfg["likelihoods"]
        if kind not in likelihoods:
            kind = next(iter(likelihoods))
        bounds = dict(likelihoods[kind].get("bounds", {}))
        return list_params, bounds

    # ------------------------------------------------------------------
    # Model graph
    # ------------------------------------------------------------------

    def _build_pymc_model(self) -> pm.Model:
        """Open one ``pm.Model`` and declare every node in dependency order."""
        cfg = self.model_config
        K = self.K
        N = self.n_participants
        is_no_pooling = isinstance(cfg.pooling, NoPooling)

        anchor = resolve_anchor(cfg.ordering, self.switching_params)

        with pm.Model() as model:
            # --- transition matrix P (K, K) ---
            tprior = cfg.transition_prior or StickyDirichlet()
            alpha = (
                tprior.concentration(K)
                if hasattr(tprior, "concentration")
                else StickyDirichlet().concentration(K)
            )
            P = pm.Dirichlet("P", a=alpha, shape=(K, K))
            log_P = pt.log(P)

            # --- initial distribution pi0 (K,) ---
            log_pi0 = self._make_log_pi0(cfg.initial_distribution, K)

            # --- SSM parameters (switching / shared / fixed) ---
            param_values: dict[str, pt.TensorVariable] = {}
            regime_params: set[str] = set()
            for name in self.list_params or []:
                val, has_regime = self._make_param(name, anchor, is_no_pooling, N, K)
                param_values[name] = val
                if has_regime:
                    regime_params.add(name)

            # --- L3 builder: emission + forward + Potential ---
            pooling_mode = "none" if is_no_pooling else "full"
            builder = make_hmm_logp_op(
                dist_class=self._emission_dist,
                data_padded=self._data_padded,
                mask=self._mask,
                K=K,
                n_participants=N,
                n_trials=self.n_trials,
                regime_params=regime_params,
                pooling=pooling_mode,
            )
            builder(param_values, log_P, log_pi0)

        return model

    def _make_log_pi0(self, spec, K: int) -> pt.TensorVariable:
        """Create the log initial-state distribution node."""
        if isinstance(spec, UniformInitialDistribution):
            return pt.as_tensor_variable(np.log(spec.pi0_value(K)))
        if isinstance(spec, FixedInitialDistribution):
            return pt.as_tensor_variable(np.log(spec.pi0_value(K)))
        if isinstance(spec, DirichletInitialDistribution):
            pi0 = pm.Dirichlet("pi0", a=spec.concentration(K), shape=(K,))
            return pt.log(pi0)
        raise TypeError(f"Unsupported initial_distribution spec {type(spec)!r}.")

    def _make_param(
        self, name: str, anchor, is_no_pooling: bool, N: int, K: int
    ) -> tuple[pt.TensorVariable, bool]:
        """Create the RV/constant for one SSM parameter; return (value, has_regime).

        Implements the three-mode rule: scalar = shared, length-K list = fixed
        per regime, prior dict / in switching_params = inferred.
        """
        spec = self.model_config.param_specs.get(name)
        bounds = self.bounds.get(name)
        in_switching = name in self.switching_params
        is_anchor = anchor is not None and anchor.name == name

        # Fixed value(s) supplied directly.
        if isinstance(spec, (int, float)) and not isinstance(spec, bool):
            return pt.as_tensor_variable(float(spec)), False
        if isinstance(spec, (list, tuple, np.ndarray)):
            arr = np.asarray(spec, dtype=float)
            if arr.shape != (K,):
                raise ValueError(
                    f"Fixed-per-regime value for {name!r} must have shape ({K},), "
                    f"got {arr.shape}."
                )
            return pt.as_tensor_variable(arr), True

        # Inferred: resolve the prior (explicit dict/Prior, else default).
        prior = spec if isinstance(spec, (dict, bmb.Prior)) else None
        if prior is None:
            default_priors = self._ssm_default_priors()
            prior = default_priors.get(name) or _bounds_based_prior(bounds)

        if in_switching:
            rv = self._make_switching_rv(
                name, prior, is_anchor, anchor, is_no_pooling, N, K
            )
            return rv, True
        # Shared inferred parameter.
        shape = (N,) if is_no_pooling else None
        return self._make_dist(prior, name, shape=shape), False

    def _make_switching_rv(
        self, name, prior, is_anchor, anchor, is_no_pooling, N, K
    ) -> pt.TensorVariable:
        """Create a per-regime switching RV, applying the anchor transform."""
        shape = (N, K) if is_no_pooling else (K,)
        if not is_anchor:
            return self._make_dist(prior, name, shape=shape)

        # Anchor: apply the `ordered` transform (ascending).
        bounds = self.bounds.get(name)
        asc = _ascending_initval(K, bounds)
        initval = np.broadcast_to(asc, shape).copy() if is_no_pooling else asc

        if anchor.direction == "desc":
            return self._make_descending_anchor(name, prior, shape, initval, K)
        return self._make_dist(
            prior, name, shape=shape, transform=ordered_transform, initval=initval
        )

    def _make_descending_anchor(
        self, name, prior, shape, asc_initval, K
    ) -> pt.TensorVariable:
        """Realise a descending anchor via the negated ordered parameter.

        Only supported for real-line symmetric priors (e.g. ``Normal``); the
        underlying ordered RV is on ``-anchor`` and the anchor is exposed as a
        deterministic negation.
        """
        dist_name = prior["name"] if isinstance(prior, dict) else prior.name
        if dist_name != "Normal":
            raise NotImplementedError(
                "Descending ordering is currently supported only for a Normal "
                f"anchor prior, not {dist_name!r}. Use direction='asc' or "
                "NoOrdering."
            )
        prior_args = dict(prior) if isinstance(prior, dict) else dict(prior.args)
        prior_args.pop("name", None)
        # u = -anchor ~ Normal(-mu, sigma), ordered ascending.
        neg_args = dict(prior_args)
        neg_args["mu"] = -float(prior_args.get("mu", 0.0))
        u = pm.Normal(
            f"{name}_ordered",
            **neg_args,
            shape=shape,
            transform=ordered_transform,
            initval=asc_initval,
        )
        return pm.Deterministic(name, -u)

    def _make_dist(
        self, prior, var_name, shape=None, transform=None, initval=None
    ) -> pt.TensorVariable:
        """Create a PyMC RV from a prior dict / ``bmb.Prior``."""
        if isinstance(prior, bmb.Prior):
            dist_name, args = prior.name, dict(prior.args)
        else:
            args = dict(prior)
            dist_name = args.pop("name")
        dist_cls = getattr(pm, dist_name)
        extra: dict[str, Any] = {}
        if shape is not None:
            extra["shape"] = shape
        if transform is not None:
            extra["transform"] = transform
        if initval is not None:
            extra["initval"] = initval
        return dist_cls(var_name, **args, **extra)

    def _ssm_default_priors(self) -> dict[str, Any]:
        """Return the SSM model's analytical default priors, if any."""
        model = self.model_config.model
        if not isinstance(model, str) or model not in default_model_config:
            return {}
        model_cfg = default_model_config[cast("SupportedModels", model)]
        likelihoods = model_cfg["likelihoods"]
        kind = (
            self.loglik_kind
            if self.loglik_kind in likelihoods
            else next(iter(likelihoods))
        )
        return dict(likelihoods[kind].get("default_priors", {}))

    def _make_model_distribution(self):
        """Satisfy the abstract method; unused on the direct-build path."""
        raise NotImplementedError(
            "RSSSM builds its PyMC model directly and does not use the bambi "
            "distribution path."
        )

    # ------------------------------------------------------------------
    # Overridden HSSMBase surface (direct-build path)
    # ------------------------------------------------------------------

    @property
    def pymc_model(self) -> pm.Model:
        """The directly-built PyMC model (no bambi)."""
        return self._pymc_model_obj

    def sample(  # type: ignore[override]
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        nuts_sampler: str = "numpyro",
        **kwargs: Any,
    ) -> az.InferenceData:
        """Sample the model via ``pm.sample`` on the directly-built graph.

        Defaults to ``nuts_sampler="numpyro"`` (decision 10.1.10): the forward
        ``pytensor.scan`` JIT-compiles to ``jax.lax.scan`` under numpyro, which
        is dramatically faster than the PyMC NUTS default on the batched
        recursion.  All other ``pm.sample`` kwargs pass through.
        """
        if self._inference_obj is not None:
            _logger.warning(
                "The model has already been sampled. Overwriting the previous "
                "inference object."
            )
        with self._pymc_model_obj:
            self._inference_obj = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                nuts_sampler=nuts_sampler,
                **kwargs,
            )
        return self._inference_obj

    def graph(self, formatting="plain", **kwargs):
        """Render the PyMC model graph via graphviz."""
        return pm.model_to_graphviz(self._pymc_model_obj, formatting=formatting)

    def log_likelihood(self, *args: Any, **kwargs: Any):
        """Unavailable on the scalar-marginal graph; use ``compute_log_likelihood``."""
        raise NotImplementedError(
            "RSSSM contributes the likelihood as a scalar marginal, so the "
            "per-trial `log_likelihood` group is not produced at sampling time. "
            "Use `compute_log_likelihood(idata)` (Phase 4) to reconstruct it "
            "post-hoc for arviz.loo / waic."
        )

    def vi(self, *args: Any, **kwargs: Any):
        """Variational inference on the directly-built model."""
        raise NotImplementedError(
            "Variational inference for RSSSM is not implemented in v1."
        )
