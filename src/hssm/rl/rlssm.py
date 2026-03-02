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

import logging
from copy import deepcopy
from typing import Any, Callable, Literal

import bambi as bmb
import pandas as pd
import pymc as pm

from hssm.config import ModelConfig, RLSSMConfig
from hssm.defaults import (
    INITVAL_JITTER_SETTINGS,
    MissingDataNetwork,
)
from hssm.distribution_utils import make_distribution
from hssm.rl.likelihoods.builder import make_rl_logp_op
from hssm.rl.utils import validate_balanced_panel
from hssm.utils import _rearrange_data

from ..base import HSSMBase

_logger = logging.getLogger("hssm")


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

        # Infer panel structure and validate balance BEFORE calling super so any
        # error surfaces before the expensive model-build steps.
        n_participants, n_trials = validate_balanced_panel(data, participant_col)

        # Store RL-specific state on self BEFORE super().__init__() so that
        # _make_model_distribution() (called from super) can access them.
        self._rlssm_config = rlssm_config
        self._n_participants = n_participants
        self._n_trials = n_trials

        # Determine data / param column names for the Op
        data_cols: list[str] = (
            list(rlssm_config.response) if rlssm_config.response else ["rt", "response"]
        )
        list_params: list[str] = rlssm_config.list_params or []
        extra_fields: list[str] = rlssm_config.extra_fields or []

        # Build the differentiable pytensor Op from the annotated SSM function.
        # This Op supersedes the loglik/loglik_kind workflow: it is passed as
        # `loglik` to HSSMBase so Config.validate() is satisfied, and
        # _make_model_distribution() uses it directly without any further wrapping.
        loglik_op = make_rl_logp_op(
            ssm_logp_func=rlssm_config.ssm_logp_func,
            n_participants=n_participants,
            n_trials=n_trials,
            data_cols=data_cols,
            list_params=list_params,
            extra_fields=extra_fields,
        )

        # Build default_priors from params_default for HSSMBase
        default_priors: dict[str, Any] = (
            {
                param: default
                for param, default in zip(
                    rlssm_config.list_params, rlssm_config.params_default
                )
            }
            if rlssm_config.list_params and rlssm_config.params_default
            else {}
        )

        # Build a ModelConfig so HSSMBase._build_model_config can apply the
        # RLSSM-specific fields (response, list_params, choices, bounds, …).
        mc = ModelConfig(
            response=(tuple(rlssm_config.response) if rlssm_config.response else None),
            list_params=list_params,
            choices=(tuple(rlssm_config.choices) if rlssm_config.choices else None),
            default_priors=default_priors,
            bounds=rlssm_config.bounds or {},
            extra_fields=extra_fields if extra_fields else None,
            backend="jax",  # RLSSM always uses the JAX backend
        )

        super().__init__(
            data=data,
            model=rlssm_config.model_name,
            choices=list(rlssm_config.choices) if rlssm_config.choices else None,
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
        RLSSM and logs a warning if requested.
        """
        # Warn if a missing-data network was requested; not supported yet.
        if self.missing_data_network != MissingDataNetwork.NONE:
            _logger.warning(
                "Missing-data network assembly (OPN/CPN) is not yet supported "
                "for RLSSM. The missing_data_network setting will be ignored."
            )

        if self.missing_data:
            _logger.info(
                "Re-arranging data to separate missing and observed datapoints. "
                "Missing data (rt == %s) will be on top, "
                "observed datapoints follow.",
                self.missing_data_value,
            )

        # Rearrange data so missing rows come first (no-op when missing_data=False).
        self.data = _rearrange_data(self.data)

        # All RLSSM parameters are treated as trialwise: the Op expects arrays of
        # length n_total_trials for every parameter, and make_distribution.logp
        # broadcasts scalar / (1,)-shaped tensors up to (n_obs,) accordingly.
        params_is_trialwise = [
            True for param_name in self.params if param_name != "p_outlier"
        ]

        extra_fields_data = (
            None
            if not self.extra_fields
            else [deepcopy(self.data[field].values) for field in self.extra_fields]
        )

        assert self.list_params is not None, "list_params should be set"
        return make_distribution(
            rv=self.model_name,
            loglik=self.loglik,
            list_params=self.list_params,
            bounds=self.bounds,
            lapse=self.lapse,
            extra_fields=extra_fields_data,
            params_is_trialwise=params_is_trialwise,
        )
