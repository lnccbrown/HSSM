"""RLSSM: Reinforcement Learning Sequential Sampling Model.

This module defines:

- :class:`_RLSSM` — the internal base class (previously ``RLSSM``) that requires
  a fully populated :class:`~hssm.rl.config.RLSSMConfig` to be passed directly.
- :class:`RLSSM` — the public-facing subclass with a simplified constructor that
  accepts a *model* name string, optional *learning_process* / *decision_process*
  overrides, and an optional *model_config* escape hatch.  Config construction is
  delegated to :func:`~hssm.rl.registry.get_rlssm_model_config`.

The key difference from :class:`~hssm.hssm.HSSM` is the likelihood:
  - ``HSSM`` wraps an analytical / ONNX / blackbox callable via
    :func:`~hssm.distribution_utils.make_likelihood_callable`.
  - ``_RLSSM`` / ``RLSSM`` build a differentiable pytensor ``Op`` directly from
    an :class:`~hssm.rl.likelihoods.builder.AnnotatedFunction` via
    :func:`~hssm.rl.likelihoods.builder.make_rl_logp_op`, which internally
    handles the RL learning rule and per-participant trial structure.
    This Op is then passed straight to
    :func:`~hssm.distribution_utils.make_distribution`, bypassing the
    standard ``loglik`` / ``loglik_kind`` wrapping pipeline.
"""

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import bambi as bmb
import pandas as pd
import pymc as pm

if TYPE_CHECKING:
    from pytensor.graph import Op


from hssm.defaults import (
    INITVAL_JITTER_SETTINGS,
)
from hssm.distribution_utils import make_distribution
from hssm.rl.likelihoods.builder import make_rl_logp_op
from hssm.rl.utils import validate_balanced_panel

from ..base import HSSMBase
from .config import RLSSMConfig
from .registry import get_rlssm_model_config

_logger = logging.getLogger("hssm")


class _RLSSM(HSSMBase):
    """Internal Reinforcement Learning Sequential Sampling Model.

    Requires a fully populated :class:`RLSSMConfig` (with ``ssm_logp_func`` set)
    to be passed directly.  End users should use :class:`RLSSM` instead, which
    provides a simplified interface backed by the named-model registry.

    Combines a reinforcement learning (RL) process with a sequential sampling
    model (SSM) inside a single differentiable likelihood.  The RL component
    computes trial-wise intermediate parameters (e.g., drift rates) from the
    learning history, which are then fed into the SSM log-likelihood.

    The likelihood is built via
    :func:`~hssm.rl.likelihoods.builder.make_rl_logp_op` from the annotated
    SSM function stored in *model_config.ssm_logp_func*.  This produces a
    differentiable pytensor ``Op`` that is passed directly to
    :func:`~hssm.distribution_utils.make_distribution`, superseding the
    ``loglik`` / ``loglik_kind`` dispatching used by :class:`~hssm.hssm.HSSM`.

    Parameters
    ----------
    data : pd.DataFrame
        Trial-level data.  Must contain at least the response columns
        specified in *model_config* (typically ``"rt"`` and ``"response"``),
        a participant identifier column (default ``"participant_id"``), and
        any extra fields listed in *model_config.extra_fields*.
        The data **must** form a balanced panel: every participant must have
        the same number of trials.
    model_config : RLSSMConfig
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
    model_config : RLSSMConfig
        The RLSSM configuration object (stored as ``self.model_config`` on
        :class:`~hssm.base.HSSMBase` with the built ``loglik`` Op injected).
    n_participants : int
        Number of participants inferred from *data*.
    n_trials : int
        Number of trials per participant inferred from *data*.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        model_config: RLSSMConfig,
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
        # ===== save/load serialisation =====
        self._init_args = self._store_init_args(locals(), kwargs)

        # Validate config (ensures ssm_logp_func is present, etc.)
        model_config.validate()

        # RLSSM reshapes rows into (n_participants, n_trials, ...) by position,
        # so _rearrange_data (which moves missing/deadline rows to the front)
        # would scramble per-participant trial sequences and corrupt RL dynamics.
        # Raise early so the user gets a clear message before model construction.
        if missing_data is not False:
            raise NotImplementedError(
                "RLSSM currently does not support `missing_data` handling. "
                "The RL log-likelihood Op relies on strict row order to recover "
                "per-participant trial sequences; rearranging rows for missing RT "
                "values would corrupt the RL learning dynamics. "
                "Please remove missing trials from the data before passing it to RLSSM."
            )
        if deadline is not False:
            raise NotImplementedError(
                "RLSSM currently does not support `deadline` handling. "
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
        self.config = model_config
        self.n_participants = n_participants
        self.n_trials = n_trials

        # Build the differentiable pytensor Op from the annotated SSM function.
        # This Op supersedes the loglik/loglik_kind workflow: it is stored on
        # rlssm_config.loglik so that HSSMBase can access it uniformly via
        # self.model_config.loglik, without any Config conversion.
        #
        # Fresh list() copies are passed to make_rl_logp_op so the closure inside
        # captures its own isolated list objects.  HSSMBase will later append
        # "p_outlier" to self.list_params, and that mutation must NOT be visible
        # to the Op's _validate_args_length check at sampling time.
        loglik_op = make_rl_logp_op(
            ssm_logp_func=model_config.ssm_logp_func,
            n_participants=n_participants,
            n_trials=n_trials,
            data_cols=list(model_config.response),  # type: ignore[arg-type]
            list_params=list(model_config.list_params),  # type: ignore[arg-type]
            extra_fields=list(model_config.extra_fields or []),
        )

        # Build a new RLSSMConfig with the Op and backend injected, leaving
        # the caller's object unmodified (dataclasses.replace creates a shallow
        # copy with only the specified fields overridden).
        #
        # backend is hardcoded to "jax" because the entire RLSSM likelihood
        # stack is JAX-only. See ssm_logp_func, make_rl_logp_op, and
        #  _make_model_distribution for details.
        model_config = replace(model_config, loglik=loglik_op, backend="jax")

        super().__init__(
            data=data,
            model_config=model_config,
            include=include,
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
        ``self.model_config.ssm_logp_func``.

        The Op already handles:
        - The RL learning rule (computing trial-wise intermediate parameters).
        - The per-participant / per-trial data reshaping.
        - Gradient computation via its embedded VJP.

        Missing-data network assembly (OPN / CPN) is not yet supported for
        RLSSM and ``missing_data`` / ``deadline`` are rejected in ``__init__``
        before this method is ever reached.
        """
        # Use self.list_params (managed by HSSMBase, includes p_outlier when
        # has_lapse=True) rather than self.model_config.list_params (the original
        # config list, never mutated by HSSMBase).
        list_params = self.list_params
        assert list_params is not None, "list_params must be set"
        assert isinstance(list_params, list), (
            "list_params must be a list"
        )  # for type checker

        # Every RLSSM distribution parameter is trialwise (the Op receives one
        # value per trial). p_outlier is excluded to match the contract of
        # make_distribution, which strips p_outlier before indexing this list.
        params_is_trialwise = [True for name in list_params if name != "p_outlier"]

        extra_fields = self.model_config.extra_fields or []
        extra_fields_data = (
            None
            if not extra_fields
            else [self.data[field].to_numpy(copy=True) for field in extra_fields]
        )

        # The differentiable pytensor Op was stored on model_config.loglik during
        # __init__; ensure it's present and cast for typing.
        assert self.model_config.loglik is not None, "model_config.loglik must be set"
        loglik_op = cast("Callable[..., Any] | Op", self.model_config.loglik)

        # RLSSMConfig carries no `rv` field; use model_name as the rv identifier.
        rv_name = self.model_config.model_name

        return make_distribution(
            rv=rv_name,
            loglik=loglik_op,
            list_params=list_params,
            bounds=self.bounds,
            lapse=self.lapse,
            extra_fields=extra_fields_data,
            params_is_trialwise=params_is_trialwise,
        )


# ---------------------------------------------------------------------------
# Blocked-attribute descriptor
# ---------------------------------------------------------------------------


class _BlockedAttribute:
    """Data descriptor that blocks read access with NotImplementedError.

    During initialisation, writes are stored and reads return the stored
    value so that :meth:`MissingDataMixin._process_missing_data_and_deadline`
    (which both writes and reads ``self.missing_data`` / ``self.deadline`` /
    ``self.loglik_missing_data``) can complete without error.

    Once ``instance.__dict__['_rlssm_fully_initialized']`` is set to ``True``
    at the end of :meth:`RLSSM.__init__`, any read raises
    :exc:`NotImplementedError`.

    Using a data descriptor (one with both ``__get__`` and ``__set__``) is
    necessary because data descriptors take priority over instance ``__dict__``
    entries, so the descriptor's ``__get__`` fires even after a write.
    """

    def __init__(self, name: str, message: str) -> None:
        self._name = name
        self._message = message
        self._storage_key = f"_ba_{name}"

    def __set_name__(self, owner: type, name: str) -> None:  # noqa: D105
        self._name = name
        self._storage_key = f"_ba_{name}"

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:  # noqa: D105
        if obj is None:
            # Class-level access — return the descriptor itself.
            return self
        if not obj.__dict__.get("_rlssm_fully_initialized", False):
            # During __init__: return the stored value so internal code works.
            return obj.__dict__.get(self._storage_key, False)
        raise NotImplementedError(self._message)

    def __set__(self, obj: Any, value: Any) -> None:  # noqa: D105
        # Store the value so internal reads during __init__ work correctly.
        obj.__dict__[self._storage_key] = value


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------


class RLSSM(_RLSSM):
    """Reinforcement Learning Sequential Sampling Model — simplified public API.

    This class wraps :class:`_RLSSM` with a user-friendly constructor that
    accepts a *model* name string (looked up in the named-model registry) and
    optional overrides for *learning_process*, *decision_process*, and
    *choices*.  Advanced users can bypass the registry entirely by supplying a
    pre-built *model_config*.

    ``missing_data``, ``deadline``, and ``loglik_missing_data`` are not
    supported for RLSSM models and raise :exc:`NotImplementedError` if accessed.

    Parameters
    ----------
    data : pd.DataFrame
        Trial-level data (balanced panel required).
    model : str, optional
        Name of a registered RLSSM model. Defaults to ``"rldm"``.
    choices : list[int] | None, optional
        Override the choice values in the registry. ``None`` uses the registry
        default.
    include : list | None, optional
        Parameter specifications forwarded to :class:`~hssm.base.HSSMBase`.
    model_config : RLSSMConfig | None, optional
        Fully built config (escape hatch). When provided, *model*,
        *learning_process*, *decision_process*, and *choices* are ignored
        (a warning is emitted if they are non-default).
    learning_process : dict | None, optional
        Override the learning-process dict in the registry. ``None`` uses the
        registry default.
    decision_process : str | None, optional
        Override the SSM name in the registry. ``None`` uses the registry
        default.
    participant_col : str, optional
        Column identifying participants. Defaults to ``"participant_id"``.
    p_outlier : float | dict | bmb.Prior | None, optional
        Lapse probability. Defaults to ``0.05``.
    lapse : dict | bmb.Prior | None, optional
        Lapse distribution. Defaults to ``Uniform(0, 20)``.
    link_settings : Literal["log_logit"] | None, optional
        Link-function preset. Defaults to ``None``.
    prior_settings : Literal["safe"] | None, optional
        Prior preset. Defaults to ``"safe"``.
    extra_namespace : dict | None, optional
        Extra variables for formula evaluation. Defaults to ``None``.
    process_initvals : bool, optional
        Whether to post-process initial values. Defaults to ``True``.
    initval_jitter : float, optional
        Jitter magnitude for initial values.
    **kwargs
        Additional keyword arguments forwarded to :class:`bmb.Model`.
    """

    # Block read access to the three missing-data attributes while silently
    # accepting any writes made by the base-class initialisation path.
    missing_data = _BlockedAttribute(  # type: ignore[assignment]
        "missing_data",
        "RLSSM does not support 'missing_data'. "
        "The RL log-likelihood Op relies on strict row order; rearranging rows "
        "for missing RT values would corrupt the RL learning dynamics. "
        "Please remove missing trials from the data before passing it to RLSSM.",
    )
    deadline = _BlockedAttribute(  # type: ignore[assignment]
        "deadline",
        "RLSSM does not support 'deadline'. "
        "The RL log-likelihood Op relies on strict row order; rearranging rows "
        "for deadline trials would corrupt the RL learning dynamics. "
        "Please remove deadline trials from the data before passing it to RLSSM.",
    )
    loglik_missing_data = _BlockedAttribute(  # type: ignore[assignment]
        "loglik_missing_data",
        "RLSSM does not support 'loglik_missing_data'. "
        "Missing-data network assembly (OPN / CPN) is not implemented for RLSSM.",
    )

    def __init__(
        self,
        data: pd.DataFrame,
        model: str = "rldm",
        choices: list[int] | None = None,
        include: list[dict[str, Any] | Any] | None = None,
        model_config: RLSSMConfig | None = None,
        learning_process: dict[str, Any] | None = None,
        decision_process: str | None = None,
        participant_col: str = "participant_id",
        p_outlier: float | dict | bmb.Prior | None = 0.05,
        lapse: dict | bmb.Prior | None = bmb.Prior("Uniform", lower=0.0, upper=20.0),
        link_settings: Literal["log_logit"] | None = None,
        prior_settings: Literal["safe"] | None = "safe",
        extra_namespace: dict[str, Any] | None = None,
        process_initvals: bool = True,
        initval_jitter: float = INITVAL_JITTER_SETTINGS["jitter_epsilon"],
        **kwargs: Any,
    ) -> None:
        # Capture simplified args BEFORE calling super so they can be
        # restored afterwards for save/load serialisation.
        _my_init_args = self._store_init_args(locals(), kwargs)

        if model_config is not None:
            # Escape-hatch path: caller supplied a fully built config.
            if model != "rldm" or any(
                x is not None for x in [learning_process, decision_process, choices]
            ):
                _logger.warning(
                    "model_config was provided; ignoring model, learning_process, "
                    "decision_process, and choices arguments."
                )
        else:
            model_config = get_rlssm_model_config(
                model=model,
                choices=choices,
                learning_process=learning_process,
                decision_process=decision_process,
            )

        super().__init__(
            data=data,
            model_config=model_config,
            participant_col=participant_col,
            include=include,
            p_outlier=p_outlier,
            lapse=lapse,
            link_settings=link_settings,
            prior_settings=prior_settings,
            extra_namespace=extra_namespace,
            process_initvals=process_initvals,
            initval_jitter=initval_jitter,
            **kwargs,
        )

        # Restore the simplified constructor args so that save/load round-trips
        # reconstruct the model via RLSSM(model=...) rather than
        # _RLSSM(model_config=...).
        self._init_args = _my_init_args

        # Mark initialisation complete — after this point _BlockedAttribute
        # raises NotImplementedError on any read of missing_data / deadline /
        # loglik_missing_data.
        self.__dict__["_rlssm_fully_initialized"] = True
