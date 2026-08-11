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
from ..defaults import INITVAL_SETTINGS
from ..modelconfig import get_default_model_config
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
    from collections.abc import Sequence

    import pandas as pd
    from xarray import DataTree

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


def _is_fixed_scalar(spec: Any) -> bool:
    """Return whether ``spec`` is a scalar *fixed* value for a parameter.

    ``np.number`` is included so numpy scalars (e.g. ``np.float32(0.8)``, which
    unlike ``np.float64`` is not a ``float`` subclass) fix the parameter rather
    than silently falling through to an inferred RV.  Booleans are excluded even
    though ``bool`` subclasses ``int``.
    """
    return isinstance(spec, (int, float, np.number)) and not isinstance(
        spec, (bool, np.bool_)
    )


def _ascending_initval(
    K: int, bounds: tuple[float, float] | None, center: float | None = None
) -> np.ndarray:
    """Return a strictly-ascending in-support init vector for the anchor.

    When ``center`` (a known-safe seed from ``INITVAL_SETTINGS``) is supplied,
    the grid is built *around* that value with a half-width that never reaches
    the nearest finite support boundary.  This keeps an anchor with a
    data-coupled validity bound — notably the non-decision time ``t``, which is
    invalid wherever ``t >= rt`` — out of its invalid region, mirroring the safe
    seed that non-anchor parameters already receive via ``_param_initval``.  For
    an unbounded anchor such as ``v`` (center 0) this reproduces the previous
    ``linspace(-2, 2, K)`` exactly.  Without a safe seed the grid spans the
    parameter's bounds as before.

    The grid is returned in ``pytensor.config.floatX`` (a float64 grid cannot be
    stored in a float32 random variable, which is what ``hssm.set_floatX
    ("float32")`` makes every RV).
    """
    grid: np.ndarray
    if center is not None:
        half_width = 2.0  # default for sides open to infinity (e.g. v)
        if bounds is not None:
            lo, hi = bounds
            if not np.isinf(lo):
                half_width = min(half_width, 0.5 * (center - lo))
            if not np.isinf(hi):
                half_width = min(half_width, 0.5 * (hi - center))
        # Floor guards the degenerate seed-on-boundary case; with the real
        # seeds (t=0.025, a=1.5, v=0) the boundary-derived width dominates.
        half_width = max(half_width, 1e-3)
        grid = center + np.linspace(-half_width, half_width, K)
    elif bounds is None:
        grid = np.linspace(-2.0, 2.0, K)
    else:
        lo, hi = bounds
        lo_inf, hi_inf = np.isinf(lo), np.isinf(hi)
        if lo_inf and hi_inf:
            grid = np.linspace(-2.0, 2.0, K)
        elif not lo_inf and hi_inf:
            grid = lo + 0.5 + np.arange(K) * 0.5
        elif lo_inf and not hi_inf:
            grid = hi - 0.5 - np.arange(K)[::-1] * 0.5
        else:
            grid = np.linspace(lo, hi, K + 2)[1:-1]
    return pm.pytensorf.floatX(grid)


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
        Required.  The emission is **always** resolved from the registry entry
        named by the identifier (``model.model_name`` for a config), so a config
        must name a *registered* SSM; it may override ``list_params`` /
        ``bounds`` / ``loglik_kind``, but a custom ``loglik`` is rejected rather
        than silently discarded.
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
        self._inference_obj: DataTree | None = None
        self._inference_obj_vi: pm.Approximation | None = None
        self._vi_approx = None
        self._map_dict = None
        self._initvals: dict[str, Any] = {}

        # ===== reject incompatible inherited kwargs (decision 10.1.9) =====
        # `p_outlier` is *not* rejected here: a per-regime lapse (in
        # switching_params, or a length-K list) is supported and resolved in
        # `_build_config`; only the global iid form is rejected there.
        self._reject_unsupported_kwargs(missing_data, deadline, lapse)

        # ===== resolve the RSSSMConfig =====
        if model_config is not None:
            # Every granular arg is already carried by the config, so silently
            # ignoring one would build a model that disagrees with the call.
            granular = {
                "model": model,
                "K": K,
                "switching_params": switching_params,
                "transition_prior": transition_prior,
                "initial_distribution": initial_distribution,
                "loglik_kind": loglik_kind,
                "backend": backend,
                "ordering": ordering,
                "pooling": pooling,
                "p_outlier": p_outlier,
            }
            supplied = [n for n, v in granular.items() if v is not None]
            supplied += sorted(kwargs)
            if supplied:
                raise ValueError(
                    "Pass either `model_config` or the granular args "
                    "(`model`, `K`, `switching_params`, ...), not both. Got "
                    f"`model_config` together with: {', '.join(supplied)}."
                )
            self.model_config = model_config
            self._fill_config_ssm_param_meta(model_config)
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
                p_outlier=p_outlier,
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

        # ===== data sanity checks =====
        # RSSSM bypasses `HSSMBase.__init__`, so `DataValidatorMixin`'s checks
        # have to be wired up explicitly here.  Without them an invalid response
        # coding (e.g. `{1, 2}` for a model whose choices are `{-1, 1}`) or a
        # negative / NaN RT silently yields a *wrong* likelihood instead of the
        # error `HSSM` raises on the same data.
        self.choices = self._resolve_choices(cfg)  # type: ignore[assignment]
        self.n_choices = len(self.choices)
        self.extra_fields = cfg.extra_fields
        # Choice-only emissions, `missing_data` and `deadline` are all rejected
        # in v1, so the corresponding branches of the mixin are inactive.
        self.is_choice_only = False
        self.missing_data = False
        self.deadline = False
        self.missing_data_value = -999.0
        self._pre_check_data_sanity()
        self._post_check_data_sanity()

        # ===== pad / align the panel =====
        (
            self._data_padded,
            self._mask,
            self.n_participants,
            self.n_trials,
        ) = pad_and_align_to_T_max(self.data, participant_col, self.response)

        # ===== resolve the emission distribution (L2) =====
        # The LAN backend="jax" path needs each per-row parameter broadcast to a
        # vector (it drives the JAX vmap); analytical / pytensor pass scalars.
        self._broadcast_params = (
            cfg.loglik_kind == "approx_differentiable" and cfg.backend == "jax"
        )
        # Per-regime lapse mixture: a trailing `p_outlier` parameter means the
        # emission is `(1 - p_outlier_k) * SSM_k + p_outlier_k * lapse` with a
        # fixed Uniform(0, 20) lapse over RT (§1.2; v1 does not expose `lapse`).
        self._has_p_outlier = "p_outlier" in (self.list_params or [])
        self._lapse = (
            bmb.Prior("Uniform", lower=0.0, upper=20.0) if self._has_p_outlier else None
        )
        self._emission_dist = resolve_emission_dist(
            model=self._emission_model_name(cfg),
            loglik_kind=cfg.loglik_kind,  # type: ignore[arg-type]
            backend=cfg.backend,
            list_params=self.list_params,
            lapse=self._lapse,
        )

        # ===== build the PyMC model directly =====
        self._pymc_model_obj = self._build_pymc_model()
        _logger.info("RSSSM model initialized successfully.")

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _emission_model_name(cfg: RSSSMConfig) -> str:
        """Return the SSM identifier the emission distribution is built from.

        ``cfg.model_name`` is the *prefixed* RSSSM name (``"rsssm_ddm"``), which
        is not a supported SSM, so the emission must be resolved from
        ``cfg.model``: the identifier itself on the string path, or the wrapped
        config's own ``model_name`` when a pre-built ``BaseModelConfig`` is
        passed as ``model``.

        The emission is **always** rebuilt from the registry entry for that
        identifier, so a wrapped ``BaseModelConfig`` must name a *registered*
        SSM.  Its ``list_params`` / ``bounds`` / ``loglik_kind`` are honoured
        (that is what the path is for), but a *custom* ``loglik`` would be
        silently discarded, so it is rejected here rather than producing a model
        that looks custom and computes the stock likelihood.
        """
        model = cfg.model
        if isinstance(model, str):
            return model
        if isinstance(model, BaseModelConfig):
            if model.loglik is not None and not RSSSM._is_registry_loglik(
                model, cfg.loglik_kind
            ):
                raise NotImplementedError(
                    "RSSSM cannot use a custom `loglik` on a `model=<config>`: the "
                    "emission is always rebuilt from the registry entry for "
                    f"{model.model_name!r}, so the custom likelihood would be "
                    "silently discarded (the model would compute the stock "
                    f"{model.model_name!r} likelihood). Register the custom model "
                    "first, then pass its name; `list_params` / `bounds` / "
                    "`loglik_kind` overrides are supported."
                )
            return model.model_name
        raise ValueError(
            "Cannot resolve the emission distribution: `model_config.model` must "
            "be the SSM identifier (a string such as 'ddm', or a "
            f"`BaseModelConfig`), got {model!r}."
        )

    @classmethod
    def _fill_config_ssm_param_meta(cls, cfg: RSSSMConfig) -> None:
        """Derive missing SSM parameter metadata on the ``model_config=`` path.

        The granular path fills ``list_params`` / ``bounds`` / ``loglik_kind``
        from the SSM registry in :meth:`_build_config`; a hand-built
        ``RSSSMConfig`` used to have to repeat all of it by hand or fail
        validation with "list_params must be populated ...".  Anything the user
        did set is left untouched — only the gaps are filled.

        Parameters
        ----------
        cfg
            The user-supplied config, mutated in place.
        """
        if cfg.list_params is not None or cfg.model is None:
            return
        list_params, bounds, kind = cls._resolve_ssm_param_meta(
            cfg.model, cfg.loglik_kind
        )
        cfg.list_params = list_params
        cfg.bounds = {**bounds, **dict(cfg.bounds)}
        cfg.loglik_kind = kind
        if kind == "approx_differentiable" and cfg.backend is None:
            cfg.backend = "jax"

    @staticmethod
    def _is_registry_loglik(model: BaseModelConfig, loglik_kind: str | None) -> bool:
        """Return whether ``model.loglik`` is the registry's own log-likelihood.

        ``Config.from_defaults("ddm", ...)`` copies the registered log-likelihood
        onto the config, which is harmless (the emission rebuilds the very same
        thing).  Only a *different* one signals a custom likelihood the RSSSM
        emission would discard.

        Compared with ``==``, not ``is``:
        :func:`~hssm.modelconfig.get_default_model_config` re-imports the config
        module on every call, so the LAN entries — whose ``loglik`` is an
        ``.onnx`` *path string* — are a fresh object each time and would never be
        identical.  Callables and classes still compare by identity under ``==``,
        so a genuinely custom likelihood is still rejected.

        The comparison is scoped to ``loglik_kind`` so that borrowing another
        kind's log-likelihood (the analytical one under
        ``loglik_kind="approx_differentiable"``, say) is *not* mistaken for the
        registry's own — that too would be silently rebuilt as the stock LAN
        emission.
        """
        try:
            registered = get_default_model_config(
                cast("SupportedModels", model.model_name)
            )
        except Exception:  # unregistered model name; rejected downstream
            return False
        likelihoods = registered["likelihoods"]
        kinds = (
            [likelihoods[loglik_kind]]
            if loglik_kind in likelihoods
            else list(likelihoods.values())
        )
        return any(model.loglik == kind_cfg.get("loglik") for kind_cfg in kinds)

    @staticmethod
    def _resolve_choices(cfg: RSSSMConfig) -> list[int]:
        """Return the valid response codings of the emission model.

        Taken from the *registry entry* of the SSM the emission is built from,
        which is where the emission itself comes from (see
        :meth:`_emission_model_name`).  This matters on both advanced paths,
        whose ``choices`` field defaults to the generic ``(0, 1)``: a wrapped
        ``BaseModelConfig`` that never touched ``choices`` would otherwise
        reject standard ``{-1, 1}`` DDM data.  It also mirrors ``hssm.HSSM``,
        which deliberately ignores a user-supplied ``choices`` for a registered
        model string.  Falls back to the config's own ``choices``.
        """
        model = cfg.model
        model_name = model.model_name if isinstance(model, BaseModelConfig) else model
        choices: Sequence[int] | None = None
        if isinstance(model_name, str):
            try:
                choices = get_default_model_config(cast("SupportedModels", model_name))[
                    "choices"
                ]
            except Exception:  # custom / unregistered model name
                choices = None
        if choices is None:
            choices = (
                model.choices if isinstance(model, BaseModelConfig) else cfg.choices
            )
        return list(choices) if choices is not None else []

    @staticmethod
    def _reject_unsupported_kwargs(missing_data, deadline, lapse) -> None:
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
        if lapse is not None:
            raise NotImplementedError(
                "RSSSM rejects the top-level `lapse` kwarg (decision 10.1.9); the "
                "per-regime lapse uses a fixed Uniform(0, 20) lapse distribution."
            )

    @staticmethod
    def _resolve_p_outlier_spec(p_outlier, K, switching_params):
        """Return ``(active, spec)`` for per-regime ``p_outlier``.

        Per-regime ``p_outlier`` is allowed (design §1.2): listed in
        ``switching_params`` (inferred per regime) or supplied as a length-K
        list (fixed per regime).  The global iid form — a scalar or a single
        prior not tied to a regime — is rejected (decision 10.1.9): it
        double-models the lapse regime.  ``spec`` is the value to store in
        ``param_specs`` (a prior dict / ``bmb.Prior`` / length-K list), or
        ``None`` to fall back to the default inferred prior.
        """
        in_switching = "p_outlier" in (switching_params or [])
        active = (p_outlier is not None) or in_switching
        if not active:
            return False, None

        is_list = isinstance(p_outlier, (list, tuple, np.ndarray))
        if is_list and len(p_outlier) != K:
            raise ValueError(
                f"Fixed-per-regime `p_outlier` must have length K={K}, got "
                f"{len(p_outlier)}."
            )
        per_regime = in_switching or is_list
        if not per_regime:
            raise NotImplementedError(
                "RSSSM rejects a global iid `p_outlier`: a single lapse "
                "probability shared across regimes double-models the lapse "
                "regime (decision 10.1.9). Pass `p_outlier` per regime — list it "
                "in `switching_params` (inferred) or give a length-K list "
                "(fixed per regime)."
            )
        return True, p_outlier

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
        p_outlier=None,
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
        # `resolved_loglik_kind` is the kind the bounds were actually read from,
        # which is what the emission must be built with as well.
        list_params, bounds, resolved_loglik_kind = self._resolve_ssm_param_meta(
            model, loglik_kind
        )

        # Per-regime p_outlier: add a trailing `p_outlier` SSM parameter (the
        # emission gains the lapse mixture) plumbed through the three-mode rule.
        param_specs = dict(param_specs)
        bounds = dict(bounds)
        active, spec = self._resolve_p_outlier_spec(p_outlier, K, switching_params)
        if active:
            if "p_outlier" not in list_params:
                list_params = list(list_params) + ["p_outlier"]
            bounds.setdefault("p_outlier", (0.0, 1.0))
            if spec is not None:
                param_specs["p_outlier"] = spec

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
            param_specs=param_specs,
        )

    @staticmethod
    def _resolve_ssm_param_meta(
        model, loglik_kind
    ) -> tuple[list[str], dict[str, tuple[float, float]], "LoglikKind"]:
        """Return (list_params, bounds, loglik_kind) for the SSM emission model.

        The returned kind is the one the bounds were actually read from: a model
        with no analytical likelihood (e.g. ``angle``) falls back to its first
        available kind, and the caller must build the emission with that same
        kind — otherwise the bounds / default priors / anchor grid and the
        emission itself come from different likelihoods.
        """
        if isinstance(model, BaseModelConfig):
            if model.list_params is None:
                raise ValueError("Provided model config has no `list_params`.")
            if model.is_choice_only:
                raise NotImplementedError(
                    "RSSSM does not support choice-only emission models in v1: the "
                    "emission is fed `(rt, response)` and the panel is keyed on the "
                    "`rt` column."
                )
            kind = cast("LoglikKind", loglik_kind or model.loglik_kind or "analytical")
            return list(model.list_params), dict(model.bounds), kind

        try:
            cfg = get_default_model_config(cast("SupportedModels", model))
        except Exception as exc:  # unknown / unsupported model name
            raise ValueError(
                f"Unknown model {model!r}; provide a model config instead, or use "
                "a supported SSM."
            ) from exc
        if len(cfg["response"]) == 1:  # choice-only SSM (no RT dimension)
            raise NotImplementedError(
                f"RSSSM does not support the choice-only model {model!r} in v1: the "
                "emission is fed `(rt, response)` and the panel is keyed on the "
                "`rt` column."
            )
        list_params = list(cfg["list_params"])
        # Pull bounds from the requested likelihood kind when available, else the
        # first available kind (e.g. LAN-only models such as `angle`).
        kind = cast("LoglikKind", loglik_kind or "analytical")
        likelihoods = cfg["likelihoods"]
        if kind not in likelihoods:
            kind = next(iter(likelihoods))
        bounds = dict(likelihoods[kind].get("bounds", {}))
        return list_params, bounds, kind

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

        # pyrefly: ignore[bad-context-manager]
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

            # Stash the regime-param set + pooling mode so the post-hoc FFBS /
            # per-trial-logp path (Section 5.5/5.6) can recompile the *same*
            # emission to a NumPy callable.
            self._regime_params = set(regime_params)
            self._anchor = anchor

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
                broadcast_params=self._broadcast_params,
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

        is_fixed_scalar = _is_fixed_scalar(spec)
        is_fixed_vector = isinstance(spec, (list, tuple, np.ndarray))

        # Anything that is not a fixed value or a prior would silently fall
        # through to "inferred with the default prior" — a numpy scalar meant to
        # fix the parameter, or a string, would quietly become an RV.
        if spec is not None and not (
            is_fixed_scalar or is_fixed_vector or isinstance(spec, (dict, bmb.Prior))
        ):
            raise TypeError(
                f"Unsupported input for {name!r} of type {type(spec).__name__}: "
                f"{spec!r}. Pass a number (shared value), a length-K list, tuple "
                "or ndarray (fixed per regime), or a prior dict / `bmb.Prior` "
                "(inferred)."
            )

        # The three modes are mutually exclusive: a parameter listed in
        # switching_params (inferred per regime) must not also be handed a fixed
        # value — that silently collapses the regime structure.
        if in_switching and (is_fixed_scalar or is_fixed_vector):
            raise ValueError(
                f"{name!r} is in switching_params (inferred per regime) but was "
                f"also given a fixed value {spec!r}. Pass a prior dict / bmb.Prior "
                f"to infer it per regime, or drop it from switching_params to fix "
                f"it."
            )

        # Fixed value(s) supplied directly.  Under no pooling the emission
        # builder expects per-participant shapes (shared -> (N,), per-regime ->
        # (N, K)), so broadcast the global fixed value across participants.
        if is_fixed_scalar:
            val = float(spec)  # type: ignore[arg-type]
            self._check_fixed_in_bounds(name, val, bounds)
            if is_no_pooling:
                return pt.as_tensor_variable(np.full(N, val)), False
            return pt.as_tensor_variable(val), False
        if is_fixed_vector:
            arr = np.asarray(spec, dtype=float)
            if arr.shape != (K,):
                raise ValueError(
                    f"Fixed-per-regime value for {name!r} must have shape ({K},), "
                    f"got {arr.shape}."
                )
            self._check_fixed_in_bounds(name, arr, bounds)
            if is_no_pooling:
                return pt.as_tensor_variable(np.broadcast_to(arr, (N, K)).copy()), True
            return pt.as_tensor_variable(arr), True

        # Inferred: resolve the prior (explicit dict/Prior, else default).
        prior = spec if isinstance(spec, (dict, bmb.Prior)) else None
        if prior is None:
            if name == "p_outlier":
                # Weakly-informative small-lapse default (mean ~0.06).
                prior = {"name": "Beta", "alpha": 1.0, "beta": 15.0}
            else:
                default_priors = self._ssm_default_priors()
                prior = default_priors.get(name) or _bounds_based_prior(bounds)

        if in_switching:
            rv = self._make_switching_rv(
                name, prior, is_anchor, anchor, is_no_pooling, N, K
            )
            return rv, True
        # Shared inferred parameter.
        shape = (N,) if is_no_pooling else None
        return (
            self._make_dist(
                prior, name, shape=shape, initval=self._param_initval(name, shape)
            ),
            False,
        )

    @staticmethod
    def _check_fixed_in_bounds(
        name: str, value: float | np.ndarray, bounds: tuple[float, float] | None
    ) -> None:
        """Raise when a user-supplied *fixed* value falls outside the SSM bounds.

        The direct-build path never goes through ``Param.validate``, so without
        this an out-of-support value (``a=-1.0``, ``z=2.0``) silently produces a
        *finite but wrong* likelihood instead of the error ``hssm.HSSM`` raises
        on the same input.  Applies to both the shared scalar and the length-K
        fixed-per-regime vector.

        Parameters
        ----------
        name
            SSM parameter name (used in the error message).
        value
            The fixed scalar or the length-K fixed-per-regime array.
        bounds
            ``(lower, upper)`` support of the parameter, or ``None`` when the
            SSM declares no bounds for it.
        """
        if bounds is None:
            return
        lower, upper = bounds
        if np.any(np.asarray(value) < lower) or np.any(np.asarray(value) > upper):
            raise ValueError(
                f"Fixed Value {value} not in bounds {bounds} for parameter {name}"
            )

    def _param_initval(self, name: str, shape: tuple[int, ...] | None):
        """Return a safe starting value for ``name`` from HSSM's INITVAL_SETTINGS.

        This mirrors HSSM's initval post-processing (which the direct-build path
        bypasses).  It matters most for the non-decision time ``t``: its prior
        mode sits above typical RTs, so the default start lands in the SSM's
        invalid region (``rt < t``) where the gradient is NaN — harmless under
        numpyro's re-init but fatal for the PyMC NUTS sampler.

        Vector starts are cast to ``pytensor.config.floatX``: a float64 array
        cannot be stored in the float32 random variables that
        ``hssm.set_floatX("float32")`` creates.
        """
        val = INITVAL_SETTINGS.get(None, {}).get(name)
        if val is None:
            return None
        if shape is None:
            return float(val)
        return pm.pytensorf.floatX(np.full(shape, float(val)))

    def _make_switching_rv(
        self, name, prior, is_anchor, anchor, is_no_pooling, N, K
    ) -> pt.TensorVariable:
        """Create a per-regime switching RV, applying the anchor transform."""
        shape = (N, K) if is_no_pooling else (K,)
        if not is_anchor:
            return self._make_dist(
                prior, name, shape=shape, initval=self._param_initval(name, shape)
            )

        # Anchor: apply the `ordered` transform (ascending).  Seed the grid on
        # the param's known-safe value (when one exists) so an anchor with a
        # data-coupled validity bound (e.g. `t`, invalid where `t >= rt`) is not
        # placed in its invalid region — the bug `_param_initval` already fixes
        # for non-anchor params.
        bounds = self.bounds.get(name)
        asc = _ascending_initval(K, bounds, center=self._param_initval(name, None))
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

        The ordered RV is ``u = -anchor``, so its start must be the *reversed*
        negation ``-asc[::-1]`` — which puts the anchor itself at ``asc[::-1]``,
        i.e. inside the same in-support grid the ascending path uses.  Passing
        the ascending grid ``asc`` straight through (what this method used to
        do) started ``u`` there and hence the anchor at ``-asc``, outside the
        support of any one-sided parameter (e.g. ``a > 0``): with
        ``asc = [0.75, 2.25]`` the anchor started at ``[-0.75, -2.25]``, which
        is both out of support and descending-ordered the wrong way round.
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
            initval=-np.asarray(asc_initval)[..., ::-1].copy(),
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
        """Return the SSM model's default priors for the chosen kind, if any."""
        model = self.model_config.model
        if not isinstance(model, str):
            return {}
        try:
            model_cfg = get_default_model_config(cast("SupportedModels", model))
        except Exception:
            return {}
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

    def __repr__(self) -> str:
        """Create a representation of the model.

        ``HSSMBase.__repr__`` walks ``self.params`` (the bambi parameter
        objects), which the direct-build path never creates, so it is replaced
        here by a summary of the regime-switching structure.
        """
        anchor = getattr(self, "_anchor", None)
        shared = [p for p in self.list_params or [] if p not in self.switching_params]
        pooling = "none" if isinstance(self.model_config.pooling, NoPooling) else "full"
        output = [
            "Regime-Switching Sequential Sampling Model",
            f"Model: {self.model_name}\n",
            f"Response variable: {', '.join(self.response)}",
            f"Likelihood: {self.loglik_kind}",
            f"Observations: {len(self.data)}",
            f"Participants: {self.n_participants} (T_max: {self.n_trials})\n",
            f"Regimes (K): {self.K}",
            f"Switching parameters: {', '.join(self.switching_params) or 'none'}",
            f"Shared parameters: {', '.join(shared) or 'none'}",
            f"Pooling: {pooling}",
            (
                f"Ordering anchor: {anchor.name} ({anchor.direction})"
                if anchor is not None
                else "Ordering anchor: none"
            ),
        ]
        return "\n".join(output)

    def __str__(self) -> str:
        """Create a string representation of the model."""
        return self.__repr__()

    def add_likelihood_parameters_to_datatree(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        """Unavailable: the trial-wise likelihood parameters come from bambi."""
        raise NotImplementedError(
            "RSSSM does not support `add_likelihood_parameters_to_datatree`: the "
            "trial-wise likelihood parameters are computed by the bambi model, "
            "which the directly-built RSSSM graph does not have. The regime "
            "parameters are already in the posterior; use `infer_regimes` for "
            "the per-trial regime probabilities."
        )

    def sample(  # type: ignore[override]
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        nuts_sampler: Literal["blackjax", "numpyro", "nutpie", "pymc"] = "numpyro",
        include_log_likelihood: bool = False,
        *,
        sampler: Any = None,
        **kwargs: Any,
    ) -> DataTree:
        """Sample the model via ``pm.sample`` on the directly-built graph.

        Defaults to ``nuts_sampler="numpyro"`` (decision 10.1.10): the forward
        ``pytensor.scan`` JIT-compiles to ``jax.lax.scan`` under numpyro, which
        is dramatically faster than the PyMC NUTS default on the batched
        recursion.  All other ``pm.sample`` kwargs pass through.

        Note
        ----
        This signature deliberately differs from :meth:`HSSM.sample`: RSSSM
        builds the PyMC model directly and calls ``pm.sample`` rather than
        ``bambi.Model.fit``, so the sampler is selected with ``nuts_sampler=``
        (not ``sampler=``) and the bambi-specific ``init`` / ``initvals`` /
        ``include_response_params`` arguments are not accepted.  ``sampler=``
        is accepted only to raise a clear error.  The *positional* order also
        differs — :meth:`HSSM.sample` takes ``sampler`` first while this method
        takes ``draws`` first — so always pass these arguments by keyword.

        Parameters
        ----------
        sampler
            Not supported; present only so that ``sampler="numpyro"`` raises a
            clear ``TypeError`` pointing at ``nuts_sampler=`` instead of
            leaking an internal PyMC error.
        include_log_likelihood
            When ``True``, attach the per-trial ``log_likelihood`` group via
            :meth:`compute_log_likelihood` (needed for ``arviz.loo``).
            Defaults to ``False``: unlike a standard HSSM model whose per-trial
            logp is vectorised, RSSSM reconstructs it post-hoc with a pure-NumPy
            forward filter over every draw (``O(chains·draws·N·T)``), which is
            costly on large posteriors — so it is opt-in here (call
            ``compute_log_likelihood`` later instead if preferred).

        Notes
        -----
        ``HSSMBase._clean_posterior_group`` is intentionally *not* applied: it
        prunes bambi trial-wise deterministics, of which the directly-built RSSSM
        graph has none, and it would risk dropping the descending-anchor
        ``Deterministic`` (``OrderByParam(direction="desc")``).
        """
        if sampler is not None:
            raise TypeError(
                "RSSSM.sample() does not accept `sampler=`; it calls `pm.sample` "
                "directly rather than `bambi.Model.fit`. Select the NUTS backend "
                f"with `nuts_sampler={sampler!r}` instead."
            )
        if self._inference_obj is not None:
            _logger.warning(
                "The model has already been sampled. Overwriting the previous "
                "inference object."
            )
        # pyrefly: ignore[bad-context-manager]
        with self._pymc_model_obj:
            self._inference_obj = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                nuts_sampler=nuts_sampler,
                **kwargs,
            )
        if include_log_likelihood:
            self.compute_log_likelihood(self._inference_obj)
        return self._inference_obj

    def graph(self, formatting="plain", name=None, figsize=None, dpi=300, fmt="png"):
        """Render the PyMC model graph via graphviz.

        The signature matches :meth:`HSSM.graph`, but only ``formatting`` is
        honoured: the direct-build path renders the PyMC model itself rather
        than a bambi model, so the figure-saving arguments (``name`` /
        ``figsize`` / ``dpi`` / ``fmt``) are ignored with a warning.  Save the
        returned graphviz object yourself if you need a file.

        Parameters
        ----------
        formatting
            One of ``"plain"`` or ``"plain_with_params"``. Defaults to
            ``"plain"``.
        name, figsize, dpi, fmt
            Accepted for signature compatibility with :meth:`HSSM.graph` and
            ignored with a warning when set to anything other than the default.

        Returns
        -------
        graphviz.Graph
            The graph.
        """
        ignored = [
            arg_name
            for arg_name, value, default in (
                ("name", name, None),
                ("figsize", figsize, None),
                ("dpi", dpi, 300),
                ("fmt", fmt, "png"),
            )
            if value != default
        ]
        if ignored:
            _logger.warning(
                "RSSSM.graph() ignores %s; only `formatting` is supported. Render "
                "and save the returned graphviz object directly if needed.",
                ", ".join(ignored),
            )
        return pm.model_to_graphviz(self._pymc_model_obj, formatting=formatting)

    def log_likelihood(self, *args: Any, **kwargs: Any):
        """Unavailable on the scalar-marginal graph; use ``compute_log_likelihood``."""
        raise NotImplementedError(
            "RSSSM contributes the likelihood as a scalar marginal, so the "
            "per-trial `log_likelihood` group is not produced at sampling time. "
            "Use `compute_log_likelihood(idata)` (Phase 4) to reconstruct it "
            "post-hoc for arviz.loo."
        )

    def vi(self, *args: Any, **kwargs: Any):
        """Variational inference on the directly-built model."""
        raise NotImplementedError(
            "Variational inference for RSSSM is not implemented in v1."
        )

    # The predictive family is out of v1 scope (design §6.3): the HMM contributes
    # its likelihood as a scalar `pm.Potential`, so the directly-built model has
    # *no observed response RV* for PyMC's predictive samplers to draw from, and
    # the inherited implementations reach through `self.model` (the bambi model
    # RSSSM never builds).  Override them to raise an informative error rather
    # than leak the bare `AttributeError: no attribute 'model'`.
    _PREDICTIVE_MSG = (
        "RSSSM does not support {name} in v1 (design §6.3): the regime-switching "
        "likelihood is contributed as a scalar marginal, so the model has no "
        "observed response random variable for PyMC's predictive samplers. "
        "Predictive simulation for a regime-switching model is bespoke (draw a "
        "regime path from P/pi0, then RTs from each trial's per-regime SSM) and "
        "is a deferred helper. Use `infer_regimes` for posterior regime recovery."
    )

    def sample_posterior_predictive(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        """Unavailable in v1: no observed response RV (design §6.3)."""
        raise NotImplementedError(
            self._PREDICTIVE_MSG.format(name="`sample_posterior_predictive`")
        )

    def predict(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        """Unavailable in v1: no observed response RV (design §6.3)."""
        raise NotImplementedError(self._PREDICTIVE_MSG.format(name="`predict`"))

    def sample_prior_predictive(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        """Unavailable in v1: no observed response RV (design §6.3)."""
        raise NotImplementedError(
            self._PREDICTIVE_MSG.format(name="`sample_prior_predictive`")
        )

    def plot_predictive(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        """Unavailable in v1: depends on the predictive samplers (design §6.3)."""
        raise NotImplementedError(self._PREDICTIVE_MSG.format(name="`plot_predictive`"))

    def sample_do(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        """Unavailable in v1: no observed response RV (design §6.3).

        The inherited implementation calls ``pm.sample_prior_predictive`` on the
        do-model.  RSSSM's entire likelihood is a ``pm.Potential``, which prior
        predictive sampling ignores, so it would return draws that do not depend
        on the data at all — a silently wrong result rather than a loud failure.
        """
        raise NotImplementedError(self._PREDICTIVE_MSG.format(name="`sample_do`"))

    def set_alias(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        """Unavailable: RSSSM builds the PyMC model directly (no bambi aliases)."""
        raise NotImplementedError(
            "RSSSM builds its PyMC model directly rather than through bambi, so "
            "bambi parameter aliases are not supported. Name variables via the "
            "constructor's per-parameter specs instead."
        )

    # ------------------------------------------------------------------
    # Post-hoc regime recovery / per-trial logp (Phase 4, §5.5/§5.6)
    # ------------------------------------------------------------------

    def infer_regimes(
        self,
        idata: DataTree | None = None,
        n_draws: int = 200,
        seed: int | None = None,
    ) -> DataTree:
        """Recover the latent regime sequences from the posterior via FFBS.

        NUTS marginalises the discrete regimes out at sampling time, so the
        posterior holds only ``theta``, ``P``, ``pi0``.  ``infer_regimes`` runs
        Forward-Filter Backward-Sample for ``n_draws`` posterior draws, drawing
        one regime sequence per participant per draw (§5.5).

        Parameters
        ----------
        idata
            Posterior to use; defaults to the model's own ``traces``.
        n_draws
            Number of posterior draws to run FFBS on.
        seed
            Seed for draw selection and the backward sampling.

        Returns
        -------
        DataTree
            A ``posterior_regimes`` group with ``regimes``
            ``(draw, participant, trial)`` and ``regime_sample_frequency``
            ``(participant, trial, regime)``.
        """
        from .ffbs import infer_regimes as _infer_regimes

        idata = cast("DataTree", idata if idata is not None else self.traces)
        return _infer_regimes(self, idata, n_draws=n_draws, seed=seed)

    def compute_log_likelihood(self, idata: DataTree | None = None) -> DataTree:
        """Attach the post-hoc per-trial log-likelihood group (§5.6).

        The sampler graph contributes only the scalar marginal (§3.4), so the
        ``log_likelihood`` group needed by ``arviz.loo`` is
        reconstructed here from the saved posterior: per draw, the forward
        filter's running log-evidence yields ``delta_t = logZ_t - logZ_{t-1}``,
        whose per-participant sum equals the marginal the sampler used.

        The per-trial terms are one-step-ahead predictives and are serially
        dependent within a participant, so the ``arviz.loo`` estimate
        is an *approximate* leave-one-out (use it as a relative comparison score,
        not an exact LOO).  This recomputes the forward filter for every
        posterior draw, so it can be slow on large posteriors.

        Parameters
        ----------
        idata
            Posterior to use; defaults to the model's own ``traces``.  The
            ``log_likelihood`` group is added to it in place and returned.
        """
        from .ffbs import compute_log_likelihood as _compute_ll

        idata = cast("DataTree", idata if idata is not None else self.traces)
        return _compute_ll(self, idata)

    def plot_regime_recovery(
        self,
        regimes_idata: DataTree | None = None,
        participant: int = 0,
        true_regimes: Any = None,
        ax: Any = None,
        n_draws: int = 200,
        seed: int | None = None,
    ):
        """Stacked-area plot of the posterior regime probabilities over trials.

        Parameters
        ----------
        regimes_idata
            Output of :meth:`infer_regimes`; computed on the fly (via
            ``infer_regimes``) when ``None``.
        participant
            Index of the participant to plot.
        true_regimes
            Optional ground-truth regime sequence to overlay as a step line.
        ax
            Optional matplotlib axis; created when ``None``.
        n_draws, seed
            Forwarded to :meth:`infer_regimes` when ``regimes_idata is None``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if regimes_idata is None:
            regimes_idata = self.infer_regimes(n_draws=n_draws, seed=seed)

        freq = regimes_idata.posterior_regimes[  # type: ignore[attr-defined]
            "regime_sample_frequency"
        ].values[participant]  # (T, K)
        n_real = int(np.sum(~np.isnan(freq[:, 0])))
        freq = freq[:n_real]
        trials = np.arange(n_real)

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 3))
        ax.stackplot(
            trials,
            *(freq[:, k] for k in range(self.K)),
            labels=[f"regime {k}" for k in range(self.K)],
            alpha=0.8,
        )
        if true_regimes is not None:
            ax.step(
                trials,
                np.asarray(true_regimes)[:n_real],
                where="mid",
                color="black",
                lw=1.0,
                label="true regime",
            )
        ax.set_xlabel("trial")
        ax.set_ylabel("P(regime | data)")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, max(n_real - 1, 1))
        ax.legend(loc="upper right", fontsize=8)
        return ax
