"""The attentional Drift Diffusion Model (aDDM) as an ``HSSMBase`` subclass.

``aDDM`` mirrors :class:`hssm.rl.rlssm.RLSSM`: it builds a differentiable
PyTensor ``Op`` from the vendored JAX kernel in ``__init__``, injects it into a
copy of the config via ``dataclasses.replace(loglik=op, backend="jax")``, and
overrides :meth:`_make_model_distribution` to consume that Op directly.

Unlike ``RLSSM`` there is **no** balanced-panel reshape (each trial's likelihood
depends only on that trial's own covariates) and ``missing_data``/``deadline``
are **allowed** (forwarded to the base class). Two aDDM-specific concerns live on
the subclass: column validation (:meth:`_validate_addm_columns`) and
materializing the 2-D ``sacc_array`` covariate (:meth:`_stack_sacc_array`).
"""

from dataclasses import replace
from typing import Any, Literal, cast

import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm

from ..base import HSSMBase
from ..distribution_utils import make_distribution
from .attention_process import resolve_attention_process
from .config import aDDMConfig
from .likelihoods.builder import make_addm_logp_op


class aDDM(HSSMBase):
    """Attentional Drift Diffusion Model.

    Parameters
    ----------
    data
        A pandas ``DataFrame`` with the response columns (``rt``, ``response``)
        and the per-trial covariate columns ``r1, r2, flag, sacc_array, d,
        sigma``. ``sacc_array`` is per-row a sequence of fixation onset times.
    model_config
        An :class:`aDDMConfig`. Defaults to ``aDDMConfig()`` (the complete default
        config) when omitted.
    include
        Optional list of regression/parameter specifications (HSSM's standard
        ``include=[...]`` machinery), e.g. a hierarchical regression on ``eta``.
    p_outlier, lapse, prior_settings, missing_data, deadline, **kwargs
        Forwarded to :class:`hssm.base.HSSMBase`.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        model_config: aDDMConfig | None = None,
        include: list[dict[str, Any] | Any] | None = None,
        p_outlier: float | dict | bmb.Prior | None = 0.05,
        lapse: dict | bmb.Prior | None = bmb.Prior("Uniform", lower=0.0, upper=20.0),
        prior_settings: Literal["safe"] | None = "safe",
        missing_data: bool | float = False,
        deadline: bool | str = False,
        **kwargs: Any,
    ) -> None:
        # Snapshot constructor args for save/load (before any local reassignment).
        self._init_args = self._store_init_args(locals(), kwargs)

        # Default to the complete default config (every aDDMConfig field defaults).
        # NOTE: this is the factory, NOT aDDMConfig.get_defaults() (which is the
        # abstract per-param method).
        model_config = model_config or aDDMConfig()
        model_config.validate()

        # aDDM-specific column/shape validation (lives on the subclass; no
        # model-name-gated hook in the generic validator).
        self._validate_addm_columns(data, model_config)

        # Materialize `sacc_array` as bambi-safe hashable tuples. HSSM passes
        # `self.data` to `bmb.Model`, and bambi converts every object column to
        # categorical (`with_categorical_cols`); a column of numpy arrays is
        # unhashable and crashes factorization, while a column of tuples survives
        # (bambi's categorical lands on its own internal copy, leaving self.data's
        # tuples intact for our extra-field extraction). The column is padded to a
        # uniform width so it stacks cleanly back to (n_trials, max_d).
        data = self._prepare_addm_data(data)

        # Resolve the attention process to a concrete callable so the Op closure
        # captures it.
        attention_process = resolve_attention_process(model_config.attention_process)

        # Build the differentiable Op. Pass fresh list() copies so HSSMBase later
        # appending "p_outlier" to self.list_params is invisible to the Op.
        loglik_op = make_addm_logp_op(
            attention_process=attention_process,
            list_params=list(model_config.list_params or []),
            extra_fields=list(model_config.extra_fields or []),
        )

        # Inject the Op + backend, leaving the caller's config object unmodified.
        model_config = replace(model_config, loglik=loglik_op, backend="jax")

        super().__init__(
            data=data,
            model_config=model_config,
            include=include,
            p_outlier=p_outlier,
            lapse=lapse,
            prior_settings=prior_settings,
            missing_data=missing_data,
            deadline=deadline,
            **kwargs,
        )

    def _make_model_distribution(self) -> type[pm.Distribution]:
        """Build a ``pm.Distribution`` from the pre-built aDDM log-likelihood Op.

        Like :meth:`RLSSM._make_model_distribution`, this bypasses
        ``make_likelihood_callable`` and uses the Op directly. Unlike RLSSM,
        ``params_is_trialwise`` is driven by which params the user put under a
        regression (``param.is_trialwise``) — aDDM params are scalar by default.
        """
        list_params = self.list_params
        assert list_params is not None, "list_params must be set"

        # Standard trial-wise pattern (identical to hssm.py / RLSSM): one flag per
        # model param, True iff the user put it under a regression. dist.py then
        # broadcasts those to (n_obs,) so the builder's logp maps them per-trial
        # (see likelihoods/builder.py). Core params default to scalar; a regressed
        # one (eta, kappa, a, b, x0) flows through per-trial. sigma stays scalar.
        params_is_trialwise = [
            param.is_trialwise
            for param_name, param in self.params.items()
            if param_name != "p_outlier"
        ]

        extra_fields_data = self._addm_extra_fields_data(self.data)

        assert self.model_config.loglik is not None, "model_config.loglik must be set"
        loglik_op = cast("Any", self.model_config.loglik)

        dist = make_distribution(
            rv=self.model_config.model_name,
            loglik=loglik_op,
            list_params=list_params,
            bounds=self.bounds,
            lapse=self.lapse,
            extra_fields=extra_fields_data,
            params_is_trialwise=params_is_trialwise,
        )
        # Expose the observed fixations to the RV's generative path so
        # posterior-predictive draws condition on them (see _push_rv_extra_fields).
        self._push_rv_extra_fields(dist, extra_fields_data)
        return dist

    def _push_rv_extra_fields(
        self, dist: type[pm.Distribution], extra_fields_data: list[np.ndarray] | None
    ) -> None:
        """Set the aDDM covariates on the RV class so ``rng_fn`` conditions on them.

        The likelihood already receives ``extra_fields`` (dist.py); the *generative*
        rng_fn does not. We key the observed fixations by name and stash them on the
        RV class (``rv_op`` is an instance, so its class carries the attr consumed by
        the ``rng_fn`` classmethod) — used for covariate-conditioned PPC. Each aDDM
        builds its own RV class, so this does not leak across models.
        """
        if extra_fields_data is None or self.extra_fields is None:
            return
        mapping = dict(zip(self.extra_fields, extra_fields_data))
        type(dist.rv_op)._extra_fields = mapping

    def _update_extra_fields(self, new_data: pd.DataFrame | None = None) -> None:
        """Refresh the distribution's extra fields, stacking ``sacc_array`` to 2-D.

        Overrides ``DataValidatorMixin._update_extra_fields`` (called from
        ``sample()`` and the data-update path) so the 2-D ``sacc_array`` is
        materialized rather than passed as a ragged object/tuple column. Refreshes
        both the likelihood extra fields and the RV's covariate-conditioned PPC
        fixations from the (possibly new) data.
        """
        if new_data is None:
            new_data = self.data
        if self.extra_fields is not None:
            extra_fields_data = self._addm_extra_fields_data(new_data)
            self.model_distribution.extra_fields = extra_fields_data
            self._push_rv_extra_fields(self.model_distribution, extra_fields_data)

    # ------------------------------------------------------------------ #
    # aDDM-specific data handling
    # ------------------------------------------------------------------ #
    def _addm_extra_fields_data(self, data: pd.DataFrame) -> list[np.ndarray] | None:
        """Materialize the extra-field columns as numeric arrays for the Op.

        ``sacc_array`` is per-row a length-``max_d`` vector → stacked into a
        ``(n_trials, max_d)`` float array; every other covariate is a
        scalar-per-row column.
        """
        extra_fields = self.extra_fields or []
        if not extra_fields:
            return None
        out: list[np.ndarray] = []
        for field in extra_fields:
            if field == "sacc_array":
                out.append(self._stack_sacc_array(data[field]))
            else:
                out.append(data[field].to_numpy(copy=True))
        return out

    @staticmethod
    def _prepare_addm_data(data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``data`` with ``sacc_array`` as padded hashable tuples."""
        data = data.copy()
        sacc_2d = aDDM._stack_sacc_array(data["sacc_array"])
        data["sacc_array"] = pd.Series(
            [tuple(row) for row in sacc_2d], index=data.index, dtype=object
        )
        return data

    @staticmethod
    def _stack_sacc_array(col) -> np.ndarray:
        """Stack a column of per-row onset sequences into ``(n_trials, max_d)``."""
        rows = [np.asarray(x, dtype=float).ravel() for x in col]
        max_d = max(r.size for r in rows)
        out = np.zeros((len(rows), max_d), dtype=float)
        for i, r in enumerate(rows):
            out[i, : r.size] = r  # zero-pad beyond observed onsets (kernel uses `d`)
        return out

    @staticmethod
    def _validate_addm_columns(data: pd.DataFrame, model_config: aDDMConfig) -> None:
        """Validate that the DataFrame carries well-formed aDDM covariates."""
        for col in ("r1", "r2", "d", "flag", "sigma", "sacc_array"):
            if col not in data.columns:
                raise ValueError(f"aDDM data is missing required column {col!r}.")

        flag = np.asarray(data["flag"])
        if not np.all(np.isin(flag, (0, 1))):
            raise ValueError("`flag` must be binary (0/1).")

        # sigma is the model-level diffusion-noise constant; the kernel treats it
        # as a scalar, so it must be constant across trials.
        sigma = np.asarray(data["sigma"], dtype=float)
        if not np.allclose(sigma, sigma.flat[0]):
            raise ValueError(
                "`sigma` (diffusion noise) must be constant across trials; "
                "per-trial sigma is not supported by the kernel."
            )

        # Reuse the single sacc-parsing path; its width is the max stage count.
        max_d = aDDM._stack_sacc_array(data["sacc_array"]).shape[1]
        d = np.asarray(data["d"])
        if np.any(d < 1) or np.any(d > max_d):
            raise ValueError(
                f"each `d` must satisfy 1 <= d <= sacc_array width (max width {max_d})."
            )
