"""RLSSM: Reinforcement Learning Sequential Sampling Model.

This module defines the :class:`RLSSM` class, a subclass of :class:`HSSMBase`
for models that couple a reinforcement learning (RL) learning process with a
sequential sampling decision model (SSM).

The key difference from :class:`HSSM` is the likelihood:
  - ``HSSM`` wraps an analytical / ONNX / blackbox callable via
    :func:`~hssm.distribution_utils.make_likelihood_callable`.
  - ``RLSSM`` builds a differentiable pytensor ``Op`` directly from an
    :class:`~hssm.rl.likelihoods.builder.AnnotatedFunction` via
    :func:`~hssm.rl.likelihoods.builder.make_rl_logp_op`, which internally
    handles the RL learning rule and per-participant trial structure.
    This Op is then passed straight to
    :func:`~hssm.distribution_utils.make_distribution`, bypassing the
    standard ``loglik`` / ``loglik_kind`` wrapping pipeline.
"""

from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import bambi as bmb
import pandas as pd
import pymc as pm

if TYPE_CHECKING:
    from pytensor.graph import Op

from hssm.config import RLSSMConfig
from hssm.defaults import (
    INITVAL_JITTER_SETTINGS,
)
from hssm.distribution_utils import make_distribution
from hssm.rl.likelihoods.builder import make_rl_logp_op
from hssm.rl.utils import validate_balanced_panel

from ..base import HSSMBase


class RLSSM(HSSMBase):
    """Reinforcement Learning Sequential Sampling Model.

    Combines a reinforcement learning (RL) process with a sequential sampling
    model (SSM) inside a single differentiable likelihood.  The RL component
    computes trial-wise intermediate parameters (e.g., drift rates) from the
    learning history, which are then fed into the SSM log-likelihood.

    The likelihood is built via
    :func:`~hssm.rl.likelihoods.builder.make_rl_logp_op` from the annotated
    SSM function stored in *rlssm_config.ssm_logp_func*.  This produces a
    differentiable pytensor ``Op`` that is passed directly to
    :func:`~hssm.distribution_utils.make_distribution`, superseding the
    ``loglik`` / ``loglik_kind`` dispatching used by :class:`~hssm.hssm.HSSM`.

    Parameters
    ----------
    data : pd.DataFrame
        Trial-level data.  Must contain at least the response columns
        specified in *rlssm_config* (typically ``"rt"`` and ``"response"``),
        a participant identifier column (default ``"participant_id"``), and
        any extra fields listed in *rlssm_config.extra_fields*.
        The data **must** form a balanced panel: every participant must have
        the same number of trials.
    rlssm_config : RLSSMConfig
        Full configuration for the RLSSM model.  Must have ``ssm_logp_func``
        set to the annotated JAX SSM log-likelihood function.
    participant_col : str, optional
        Name of the column that uniquely identifies participants.
        Used to infer ``n_participants`` and ``n_trials`` from *data*.
        Defaults to ``"participant_id"``.
    include : list, optional
        Parameter specifications forwarded to :class:`~hssm.base.HSSMBase`.
    p_outlier : float | dict | bmb.Prior | None, optional
        Lapse probability specification. Defaults to ``0.05``.
    lapse : dict | bmb.Prior | None, optional
        Lapse distribution. Defaults to ``Uniform(0, 20)``.
    link_settings : Literal["log_logit"] | None, optional
        Link-function preset. Defaults to ``None``.
    prior_settings : Literal["safe"] | None, optional
        Prior preset. Defaults to ``"safe"``.
    extra_namespace : dict | None, optional
        Extra variables for formula evaluation. Defaults to ``None``.
    missing_data : bool | float, optional
        Whether to handle missing RT data coded as ``-999.0``.
        Defaults to ``False``.
    deadline : bool | str, optional
        Whether to handle deadline data. Defaults to ``False``.
    loglik_missing_data : Callable | None, optional
        Custom likelihood for missing observations. Defaults to ``None``.
    process_initvals : bool, optional
        Whether to post-process initial values. Defaults to ``True``.
    initval_jitter : float, optional
        Jitter magnitude for initial values.
        Defaults to :data:`~hssm.defaults.INITVAL_JITTER_SETTINGS` epsilon.
    **kwargs
        Additional keyword arguments forwarded to :class:`bmb.Model`.

    Attributes
    ----------
    _rlssm_config : RLSSMConfig
        The RLSSM configuration object.
    _n_participants : int
        Number of participants inferred from *data*.
    _n_trials : int
        Number of trials per participant inferred from *data*.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        rlssm_config: RLSSMConfig,
        participant_col: str = "participant_id",
        include: list[dict[str, Any] | Any] | None = None,
        p_outlier: float | dict | bmb.Prior | None = 0.05,
        lapse: dict | bmb.Prior | None = bmb.Prior("Uniform", lower=0.0, upper=20.0),
        link_settings: Literal["log_logit"] | None = None,
        prior_settings: Literal["safe"] | None = "safe",
        extra_namespace: dict[str, Any] | None = None,
        missing_data: bool | float = False,
        deadline: bool | str = False,
        loglik_missing_data: Callable | None = None,
        process_initvals: bool = True,
        initval_jitter: float = INITVAL_JITTER_SETTINGS["jitter_epsilon"],
        **kwargs: Any,
    ) -> None:
        # Validate config (ensures ssm_logp_func is present, etc.)
        rlssm_config.validate()

        # RLSSM reshapes rows into (n_participants, n_trials, ...) by position,
        # so _rearrange_data (which moves missing/deadline rows to the front)
        # would scramble per-participant trial sequences and corrupt RL dynamics.
        # Raise early so the user gets a clear message before model construction.
        if missing_data is not False:
            raise ValueError(
                "RLSSM does not support `missing_data` handling. "
                "The RL log-likelihood Op relies on strict row order to recover "
                "per-participant trial sequences; rearranging rows for missing RT "
                "values would corrupt the RL learning dynamics. "
                "Please remove missing trials from the data before passing it to RLSSM."
            )
        if deadline is not False:
            raise ValueError(
                "RLSSM does not support `deadline` handling. "
                "The RL log-likelihood Op relies on strict row order to recover "
                "per-participant trial sequences; rearranging rows for deadline "
                "trials would corrupt the RL learning dynamics. Please remove "
                "deadline trials from the data before passing it to RLSSM."
            )

        # Infer panel structure and validate balance BEFORE calling super so any
        # error surfaces before the expensive model-build steps.
        n_participants, n_trials = validate_balanced_panel(data, participant_col)

        # Store RL-specific state on self BEFORE super().__init__() so that
        # _make_model_distribution() (called from super) can access them.
        self._rlssm_config = rlssm_config
        self._n_participants = n_participants
        self._n_trials = n_trials

        # Build the differentiable pytensor Op from the annotated SSM function.
        # This Op supersedes the loglik/loglik_kind workflow: it is passed as
        # `loglik` to HSSMBase so Config.validate() is satisfied, and
        # _make_model_distribution() uses it directly without any further wrapping.
        #
        # Fresh list() copies are passed to make_rl_logp_op so the closure inside
        # captures its own isolated list objects.  HSSMBase will later append
        # "p_outlier" to self.list_params, and that mutation must NOT be visible
        # to the Op's _validate_args_length check at sampling time.
        loglik_op = make_rl_logp_op(
            ssm_logp_func=rlssm_config.ssm_logp_func,
            n_participants=n_participants,
            n_trials=n_trials,
            data_cols=list(rlssm_config.response),  # type: ignore[arg-type]
            list_params=list(rlssm_config.list_params),  # type: ignore[arg-type]
            extra_fields=list(rlssm_config.extra_fields or []),
        )

        # Delegate ModelConfig construction to RLSSMConfig, which already owns
        # all the required fields (response, list_params, choices, bounds, …).
        mc = rlssm_config.to_model_config()

        super().__init__(
            data=data,
            model=rlssm_config.model_name,
            include=include,
            model_config=mc,
            # Pass the Op as loglik so Config.validate() is satisfied.
            # loglik_kind="approx_differentiable" reflects that the Op is
            # differentiable (gradients flow through its VJP).
            loglik=loglik_op,
            loglik_kind="approx_differentiable",
            p_outlier=p_outlier,
            lapse=lapse,
            link_settings=link_settings,
            prior_settings=prior_settings,
            extra_namespace=extra_namespace,
            missing_data=missing_data,
            deadline=deadline,
            loglik_missing_data=loglik_missing_data,
            process_initvals=process_initvals,
            initval_jitter=initval_jitter,
            **kwargs,
        )

    def _make_model_distribution(self) -> type[pm.Distribution]:
        """Build a pm.Distribution using the pre-built RL log-likelihood Op.

        Unlike :meth:`HSSM._make_model_distribution`, this method does not go
        through :func:`~hssm.distribution_utils.make_likelihood_callable`.
        Instead it uses ``self.loglik`` directly — the differentiable pytensor
        ``Op`` built in :meth:`__init__` from
        ``self._rlssm_config.ssm_logp_func``.

        The Op already handles:
        - The RL learning rule (computing trial-wise intermediate parameters).
        - The per-participant / per-trial data reshaping.
        - Gradient computation via its embedded VJP.

        Missing-data network assembly (OPN / CPN) is not yet supported for
        RLSSM and ``missing_data`` / ``deadline`` are rejected in ``__init__``
        before this method is ever reached.
        """
        # Build params_is_trialwise in the same order as self.list_params so the
        # length always matches the list_params= argument passed to make_distribution.
        # p_outlier is a scalar mixture weight (not trialwise); every other RLSSM
        # parameter is trialwise (the Op receives one value per trial).
        assert self.list_params is not None, "list_params should be set by HSSMBase"
        params_is_trialwise = [name != "p_outlier" for name in self.list_params]

        extra_fields_data = (
            None
            if not self.extra_fields
            else [self.data[field].to_numpy(copy=True) for field in self.extra_fields]
        )

        # self.loglik was set to the pytensor Op built in __init__; cast to
        # narrow the inherited union type so make_distribution's type-checker
        # accepts it without a runtime penalty.
        loglik_op = cast("Callable[..., Any] | Op", self.loglik)
        return make_distribution(
            rv=self.model_name,
            loglik=loglik_op,
            list_params=self.list_params,
            bounds=self.bounds,
            lapse=self.lapse,
            extra_fields=extra_fields_data,
            params_is_trialwise=params_is_trialwise,
        )

    def _get_prefix(self, name_str: str) -> str:
        """Resolve parameter prefix, handling underscore-containing RL param names.

        The base-class implementation splits ``name_str`` on the first ``_`` and
        returns that single token (e.g. ``"rl_alpha_Intercept" → "rl"``), which
        breaks for RL parameters whose names contain underscores.  It also uses a
        substring check (``"p_outlier" in name_str``) for the lapse parameter,
        which would misfire for any parameter whose name merely *contains* that
        substring.

        This override replaces both heuristics with a single longest-prefix-first
        token search: split on ``_``, then try joining 1…N tokens (longest first)
        until a candidate is found in ``self.params``.  This is both correct for
        multi-token RL param names and collision-free for ``p_outlier``.
        """
        if "_" in name_str:
            parts = name_str.split("_")
            for i in range(len(parts), 0, -1):
                candidate = "_".join(parts[:i])
                if candidate in self.params:
                    return candidate
        return super()._get_prefix(name_str)
