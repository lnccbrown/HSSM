"""RL-specific configuration classes.

This module houses `RLSSMConfig` which was previously defined in
`hssm.config`. It is intentionally lightweight and re-uses
`BaseModelConfig` from :mod:`hssm.config` to avoid duplicating core
behaviour.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import MISSING, dataclass, field, fields
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .._types import LoglikKind, SupportedModels
    from ..config import ModelConfig

from ..config import DEFAULT_SSM_CHOICES, DEFAULT_SSM_OBSERVED_DATA, BaseModelConfig
from ..utils import annotate_function

_logger = logging.getLogger("hssm")


@dataclass
class RLSSMConfig(BaseModelConfig):
    """Config for reinforcement learning + sequential sampling models.

    Extends `BaseModelConfig` with the fields required by the RLSSM
    likelihood pipeline.  The key extra fields are:

    - ``ssm_logp_func``: the annotated JAX SSM log-likelihood function (see
      below) whose ``computed`` dict drives per-parameter RL computations.
    - ``learning_process``: a mapping that declares *how* each computed
      parameter is specified (see below).
    - ``decision_process``: the name (string) or `ModelConfig` instance
      that identifies the SSM decision process (e.g. ``"ddm"``, ``"angle"``).
    - ``decision_process_loglik_kind`` / ``learning_process_kind``: string
      tags that record which kind of likelihood and which kind of learning rule
      are used (e.g. ``"approx_differentiable"`` / ``"blackbox"``).

    ssm_logp_func:
        A JAX function decorated with ``@annotate_function``.  It must carry:

        - ``.inputs`` — ordered list of all parameter names the function
          expects (e.g. ``["v", "a", "z", "t", "theta", "rt", "response"]``).
        - ``.outputs`` — list of output names (e.g. ``["logp"]``).
        - ``.computed`` — dict mapping each *computed* parameter name to the
          annotated function that produces it.  For example::

              {"v": compute_v_annotated}

          where ``compute_v_annotated`` is itself decorated with
          ``@annotate_function`` and carries ``.inputs`` / ``.outputs``.

        ``make_rl_logp_op`` inspects ``ssm_logp_func.computed`` to resolve
        which parameters come from data / sampled posteriors and which must
        be computed by the RL learning rule at each gradient step.

    learning_process:
        A dict keyed by the name of each computed parameter (matching the keys
        in ``ssm_logp_func.computed``).  Values record how that parameter is
        specified.  The dict is intentionally permissive — current supported
        value forms are:

        - **callable** — an annotated function (or plain function) used to
          compute the parameter.  The actual computation at runtime is driven
          by ``ssm_logp_func.computed``; this entry serves as declarative
          documentation and for config serialisation / round-trip::

              learning_process = {"v": compute_v_annotated}

        - **string** — a symbolic identifier for declarative / YAML-based
          configs that can be resolved to a callable by the caller::

              learning_process = {"v": "subject_wise_function"}

        An empty dict ``{}`` is valid when the SSM has no computed parameters
        (i.e. ``ssm_logp_func.computed`` is also empty).

        .. note::
            The dict is *not* directly consumed by ``make_rl_logp_op``.
            The actual compute functions used at runtime come from
            ``ssm_logp_func.computed``.  ``learning_process`` therefore acts
            as a config-level record of intent and is useful for inspection,
            serialisation, and future higher-level tooling.
    """

    decision_process_loglik_kind: str = field(kw_only=True)
    learning_process_kind: str = field(kw_only=True)
    params_default: list[float] = field(kw_only=True)
    decision_process: str | "ModelConfig" = field(kw_only=True)
    learning_process: dict[str, Any] = field(kw_only=True)
    ssm_logp_func: Any = field(default=None, kw_only=True)

    # Private metadata attached by from_ssms_model(); not constructor arguments.
    _ssms_model_config: Any = field(default=None, init=False, repr=False)
    _ssms_assembled_model: Any = field(default=None, init=False, repr=False)
    _ssms_response_to_choice: dict[int, int] | None = field(
        default=None, init=False, repr=False
    )
    _ssms_participant_contract: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):  # noqa: D105
        if self.loglik_kind is None:
            self.loglik_kind = "approx_differentiable"
            _logger.debug(
                "RLSSMConfig: loglik_kind not specified; "
                "defaulting to 'approx_differentiable'."
            )

    @classmethod
    def from_defaults(  # noqa: D102
        cls, model_name: "SupportedModels" | str, loglik_kind: "LoglikKind" | None
    ) -> "RLSSMConfig":
        raise NotImplementedError(
            "RLSSMConfig does not support from_defaults(). "
            "Use RLSSMConfig.from_rlssm_dict() or the constructor directly."
        )

    @classmethod
    def from_rlssm_dict(cls, config_dict: dict[str, Any]) -> "RLSSMConfig":  # noqa: D102
        # Derive required fields from the dataclass itself: a field is required
        # iff it has no default and no default_factory. This keeps the dataclass
        # as the single source of truth — no separate required-key list needed.
        field_exceptions = ("loglik", "loglik_kind", "backend")
        required_fields = [
            f.name
            for f in fields(cls)
            if f.name not in field_exceptions
            and f.default is MISSING
            and f.default_factory is MISSING  # type: ignore[misc]
        ]
        for field_name in required_fields:
            if field_name not in config_dict or config_dict[field_name] is None:
                raise ValueError(f"{field_name} must be provided in config_dict")

        # ssm_logp_func has a dataclass default of None but is required in practice.
        if config_dict.get("ssm_logp_func") is None:
            raise ValueError("ssm_logp_func must be provided in config_dict")

        init_kwargs = dict(
            model_name=config_dict["model_name"],
            description=config_dict.get("description"),
            list_params=config_dict.get("list_params"),
            extra_fields=config_dict.get("extra_fields"),
            params_default=config_dict["params_default"],
            decision_process=config_dict["decision_process"],
            learning_process=config_dict["learning_process"],
            ssm_logp_func=config_dict.get("ssm_logp_func"),
            bounds=config_dict.get("bounds", {}),
            decision_process_loglik_kind=config_dict["decision_process_loglik_kind"],
            learning_process_kind=config_dict["learning_process_kind"],
        )

        def _get_or_warn(key: str, default: Any) -> None:
            if key not in config_dict:
                _logger.warning(
                    "'%s' not specified in the RLSSM config; using default value: %r.",
                    key,
                    default,
                )
            init_kwargs[key] = config_dict.get(key, default)

        _get_or_warn("response", DEFAULT_SSM_OBSERVED_DATA)
        _get_or_warn("choices", DEFAULT_SSM_CHOICES)

        return cls(**init_kwargs)

    @classmethod
    def from_ssms_model(
        cls,
        model: str | Any,
        *,
        backend: Literal["auto", "jax"] = "jax",
        decision_process_loglik_kind: str = "approx_differentiable",
    ) -> "RLSSMConfig":
        """Build an HSSM RLSSM config from a canonical ``ssms.rl`` model.

        Thin bridge between ``ssm-simulators``' ``ssms.rl`` package — which owns
        the RLSSM model registry and the learning kernel — and HSSM, which owns
        the decision-process SSM log-likelihood and inference. The learning
        recursion (and the ``response_to_choice`` mapping) is taken from the ssms
        assembled participant function; the decision-process SSM logp is built
        HSSM-side from ``hssm.modelconfig`` + ONNX via :mod:`hssm.rl.registry`.

        Parameters
        ----------
        model
            A registered ``ssms.rl`` preset name (e.g. ``"2AB_RW_Angle"``) or an
            ``ssms.rl.ModelConfig`` instance.
        backend
            Learning backend requested from ssms; must yield gradient support.
            Defaults to ``"jax"``.
        decision_process_loglik_kind
            Loglik kind for the HSSM decision process. Defaults to
            ``"approx_differentiable"``.

        Returns
        -------
        RLSSMConfig
            An HSSM-ready config, usable as
            ``hssm.RLSSM(data, model_config=config)``.
        """
        try:
            ssms_rl = importlib.import_module("ssms.rl")
        except ImportError as exc:
            raise ImportError(
                "RLSSMConfig.from_ssms_model() requires an ssm-simulators "
                "version that exposes `ssms.rl` (>=0.12.4). Install or upgrade "
                "`ssm-simulators` with RLSSM/JAX support."
            ) from exc

        structural_config = ssms_rl.resolve_model(model)
        assembled = structural_config.assemble(backend=backend)
        if assembled.gradient != "available":
            raise ValueError(
                "HSSM's ssms-backed RLSSM bridge requires gradient support. "
                f"Assembled model {assembled.model_name!r} resolved "
                f"learning_backend={assembled.learning_backend!r} and "
                f"gradient={assembled.gradient!r}."
            )

        # Learning kernel is owned by ssms: wrap the assembled participant
        # function as HSSM annotated computed-parameter functions.
        computed_functions = _make_ssms_computed_functions(assembled)

        # Decision-process SSM logp is owned by HSSM: reuse the registry's
        # modelconfig+ONNX builder, then attach the ssms-derived computed funcs.
        # Local import avoids a config <-> registry import cycle.
        from . import registry  # noqa: PLC0415

        ssm_base_logp_func = registry._get_ssm_logp(assembled.decision_process)
        ssm_logp_func = registry._build_ssm_logp_func(
            ssm_base_logp_func, computed_functions
        )

        config = cls.from_rlssm_dict(
            {
                "model_name": assembled.model_name,
                "description": getattr(structural_config, "description", None),
                "list_params": list(assembled.list_params),
                "bounds": dict(assembled.bounds),
                "params_default": list(assembled.params_default),
                "decision_process": assembled.decision_process,
                "learning_process": computed_functions,
                "ssm_logp_func": ssm_logp_func,
                "response": list(assembled.response),
                "choices": tuple(assembled.choices),
                # ssms exposes task/design columns (feedback, condition, block,
                # ...) as `context_fields`; HSSM carries them through its
                # internal `extra_fields` data-column plumbing.
                "extra_fields": list(assembled.context_fields),
                "decision_process_loglik_kind": decision_process_loglik_kind,
                "learning_process_kind": "approx_differentiable",
            }
        )
        config._ssms_model_config = structural_config
        config._ssms_assembled_model = assembled
        config._ssms_response_to_choice = dict(assembled.response_to_choice)
        if hasattr(structural_config, "participant_contract"):
            config._ssms_participant_contract = structural_config.participant_contract()
        return config

    def validate(self) -> None:  # noqa: D102
        if self.response is None:
            raise ValueError("Please provide `response` columns in the configuration.")
        if self.list_params is None:
            raise ValueError("Please provide `list_params` in the configuration.")
        if self.choices is None:
            raise ValueError("Please provide `choices` in the configuration.")
        if self.decision_process is None:
            raise ValueError("Please specify a `decision_process`.")

        logpfunc = self.ssm_logp_func
        if logpfunc is None:
            raise ValueError(
                "Please provide `ssm_logp_func`: the fully annotated JAX SSM "
                "log-likelihood function required by `make_rl_logp_op`."
            )
        if not callable(logpfunc):
            raise ValueError(
                f"`ssm_logp_func` must be a callable, but got {type(logpfunc)!r}."
            )
        missing_attrs = [
            attr
            for attr in ("inputs", "outputs", "computed")
            if not hasattr(logpfunc, attr)
        ]
        if missing_attrs:
            raise ValueError(
                "`ssm_logp_func` must be decorated with `@annotate_function` "
                "so that it carries the attributes required by `make_rl_logp_op`. "
                f"Missing attribute(s): {missing_attrs}. "
            )

        if not isinstance(logpfunc.computed, dict) or not all(
            callable(v) for v in logpfunc.computed.values()
        ):
            raise ValueError(
                "`ssm_logp_func.computed` must be a dictionary with callable values."
            )

        if self.params_default and self.list_params:
            if len(self.params_default) != len(self.list_params):
                raise ValueError(
                    f"params_default length ({len(self.params_default)}) doesn't "
                    f"match list_params length ({len(self.list_params)})"
                )

        if self.list_params:
            missing_bounds = [p for p in self.list_params if p not in self.bounds]
            if missing_bounds:
                raise ValueError(
                    f"Missing bounds for parameter(s): {missing_bounds}. "
                    "Every parameter in `list_params` must have a corresponding "
                    "entry in `bounds`."
                )

    def get_defaults(  # noqa: D102
        self, param: str
    ) -> tuple[float | None, tuple[float, float] | None]:
        return None, self.bounds.get(param)


def _make_ssms_computed_functions(assembled: Any) -> dict[str, Any]:
    """Wrap an ssms ``AssembledModel`` participant fn as HSSM computed funcs.

    Each ssms computed decision parameter becomes an ``@annotate_function``-
    decorated callable that HSSM's ``make_rl_logp_func`` can resolve. The
    learning recursion — and the ``response_to_choice`` mapping that converts raw
    SSM response labels (e.g. ``-1``/``1``) to zero-based learning choices — lives
    inside the ssms assembled participant function.
    """
    input_fields = list(assembled.participant_input_fields())
    participant_fn = assembled.assemble_participant_fn(output="dict")
    computed_params = list(assembled.computed_params)

    if len(computed_params) == 1:
        param_name = computed_params[0]

        @annotate_function(inputs=input_fields, outputs=[param_name])
        def compute(subject_trials):
            return participant_fn(subject_trials)[param_name]

        return {param_name: compute}

    @annotate_function(inputs=input_fields, outputs=computed_params)
    def compute(subject_trials):
        values = participant_fn(subject_trials)
        return {param_name: values[param_name] for param_name in computed_params}

    return {param_name: compute for param_name in computed_params}
