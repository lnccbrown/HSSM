"""Registry for named RLSSM models and SSM base log-likelihood functions.

This module provides:

- :data:`_SSM_REGISTRY` — holds *custom* SSM entries added via
  :func:`register_ssm`.  Built-in HSSM models (``"ddm"``, ``"angle"``,
  ``"weibull"``, and any other model in :mod:`hssm.modelconfig` that exposes an
  ``approx_differentiable`` likelihood) are resolved automatically from
  :func:`hssm.modelconfig.get_default_model_config` and do **not** need to be
  pre-registered here.
- :data:`_RLSSM_REGISTRY` — maps named RLSSM model strings (e.g.
  ``"2AB_RescorlaWagner_ddm"``) to their default decision process,
  learning process, and parameter info.
- :func:`get_rlssm_model_config` — builds a :class:`~hssm.rl.config.RLSSMConfig`
  from a named model string with optional overrides.
- :func:`register_rlssm_model` — register a custom named RLSSM model.
- :func:`register_ssm` — register a custom SSM base logp function.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from hssm.distribution_utils.onnx import make_jax_matrix_logp_funcs_from_onnx
from hssm.rl.likelihoods.two_armed_bandit import compute_v_subject_wise
from hssm.utils import annotate_function

from .config import RLSSMConfig

_logger = logging.getLogger(__name__)


@dataclass
class LearningProcessMetadata:
    """Metadata describing the sampled-parameter side of a learning process."""

    learning_process: dict[str, Any]
    sampled_params: list[str] = field(default_factory=list)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    defaults: list[float] = field(default_factory=list)
    extra_fields: list[str] = field(default_factory=list)
    kind: str = "blackbox"

    def copy(self) -> LearningProcessMetadata:
        """Return a defensive copy for safe registry reuse."""
        return LearningProcessMetadata(
            learning_process=dict(self.learning_process),
            sampled_params=list(self.sampled_params),
            bounds=dict(self.bounds),
            defaults=list(self.defaults),
            extra_fields=list(self.extra_fields),
            kind=self.kind,
        )

    def validate(self) -> None:
        """Ensure defaults and bounds align with sampled params."""
        if len(self.defaults) != len(self.sampled_params):
            raise ValueError(
                "learning_process defaults length "
                f"({len(self.defaults)}) must match sampled params length "
                f"({len(self.sampled_params)})"
            )

        missing_bounds = [p for p in self.sampled_params if p not in self.bounds]
        if missing_bounds:
            raise ValueError(
                f"Missing bounds for learning-process parameter(s): {missing_bounds}."
            )

    def with_learning_process_override(
        self,
        learning_process: dict[str, Any],
        response: list[str],
    ) -> LearningProcessMetadata:
        """Return metadata updated for a new learning-process mapping.

        Sampled parameter names are always re-derived from the override. Bounds and
        defaults are preserved only for parameters that still exist under the new
        learning process; newly introduced parameters must be supplied through a
        custom model registration or model_config, otherwise we fail early.
        """
        new_learning_process = dict(learning_process)
        new_sampled_params = _derive_lp_params(
            new_learning_process, response, self.extra_fields
        )

        defaults_by_param = dict(zip(self.sampled_params, self.defaults))
        missing_bounds = [p for p in new_sampled_params if p not in self.bounds]
        missing_defaults = [p for p in new_sampled_params if p not in defaults_by_param]

        if missing_bounds or missing_defaults:
            missing_parts: list[str] = []
            if missing_bounds:
                missing_parts.append(f"bounds for {missing_bounds}")
            if missing_defaults:
                missing_parts.append(f"defaults for {missing_defaults}")
            raise ValueError(
                "learning_process override introduced sampled parameter metadata "
                f"not available in the registry entry: {', '.join(missing_parts)}. "
                "Register a custom RLSSM model with matching metadata or pass "
                "model_config= directly to RLSSM()."
            )

        return LearningProcessMetadata(
            learning_process=new_learning_process,
            sampled_params=new_sampled_params,
            bounds={p: self.bounds[p] for p in new_sampled_params},
            defaults=[defaults_by_param[p] for p in new_sampled_params],
            extra_fields=list(self.extra_fields),
            kind=self.kind,
        )


@dataclass
class RLSSMRegistryEntry:
    """Structured registry entry for a named RLSSM template."""

    decision_process: str | dict[str, Any]
    learning_process_metadata: LearningProcessMetadata
    choices: list[int] = field(default_factory=lambda: [0, 1])
    description: str | None = None
    decision_process_loglik_kind: str = "approx_differentiable"
    model_name: str | None = None

    def copy(self) -> RLSSMRegistryEntry:
        """Return a defensive copy for safe local overrides."""
        return RLSSMRegistryEntry(
            decision_process=deepcopy(self.decision_process),
            learning_process_metadata=self.learning_process_metadata.copy(),
            choices=list(self.choices),
            description=self.description,
            decision_process_loglik_kind=self.decision_process_loglik_kind,
            model_name=self.model_name,
        )


# ---------------------------------------------------------------------------
# Default annotated Rescorla-Wagner learning function
# ---------------------------------------------------------------------------

_compute_v_annotated = annotate_function(
    inputs=["rl_alpha", "scaler", "response", "feedback"],
    outputs=["v"],
)(compute_v_subject_wise)

# ---------------------------------------------------------------------------
# SSM base log-likelihood registry
# ---------------------------------------------------------------------------
# This dict holds only *custom* SSM entries added at runtime via register_ssm().
# Built-in HSSM models are resolved on demand from hssm.modelconfig — see
# _build_ssm_spec_from_modelconfig() and _get_decision_process_spec().
#
# Each entry (custom or derived) provides:
#   ssm_base_logp_func  - annotated JAX function (inputs + outputs, no computed)
#   list_params_ssm     - ordered SSM parameter names (including computed ones)
#   bounds_ssm          - bounds for all SSM params
#   params_default_ssm  - default values aligned with list_params_ssm
#   response            - data column names

_SSM_REGISTRY: dict[str, dict[str, Any]] = {}

# Cache for resolved SSM base logp functions (populated on first use by
# _get_ssm_logp, or immediately by register_ssm when a pre-built func is
# supplied by the caller).
_SSM_LOGP_CACHE: dict[str, Any] = {}


def _make_ssm_base_logp_from_onnx(
    onnx_file: str,
    list_params_ssm: list[str],
    response: list[str],
) -> Any:
    """Build and annotate a JAX log-likelihood function from an ONNX model file."""
    _raw = make_jax_matrix_logp_funcs_from_onnx(model=onnx_file)
    return annotate_function(
        inputs=list_params_ssm + response,
        outputs=["logp"],
    )(_raw)


def _build_ssm_spec_from_modelconfig(name: str) -> dict[str, Any]:
    """Build an SSM registry-compatible spec from HSSM's modelconfig system.

    This allows any built-in HSSM model with an ``approx_differentiable``
    likelihood to be used as an RLSSM decision process without re-registering
    it in ``_SSM_REGISTRY``.  Parameter defaults are computed as midpoints of
    the model's parameter bounds.

    Raises
    ------
    ValueError
        If *name* is not a supported HSSM model or it has no
        ``approx_differentiable`` likelihood.
    """
    # Local import to avoid circular dependencies at module level.
    from hssm.modelconfig import get_default_model_config  # noqa: PLC0415

    try:
        mc = get_default_model_config(name)  # type: ignore[arg-type]
    except ValueError as exc:
        raise ValueError(
            f"Decision process '{name}' is not a registered custom SSM and is not a "
            "supported HSSM model. "
            f"Custom SSMs in registry: {list(_SSM_REGISTRY.keys())}. "
            "Use register_ssm() to add a custom decision process."
        ) from exc

    ad = mc["likelihoods"].get("approx_differentiable")
    if ad is None:
        raise ValueError(
            f"Model '{name}' has no approx_differentiable likelihood and cannot be "
            "used as an RLSSM decision process."
        )

    list_params_ssm: list[str] = list(mc["list_params"])
    bounds_ssm: dict[str, tuple[float, float]] = dict(ad["bounds"])
    response: list[str] = list(mc["response"])
    onnx_file: str = str(ad["loglik"])

    # Derive parameter defaults as midpoints of their respective bounds.
    params_default_ssm = [
        sum(bounds_ssm[p]) / 2 if p in bounds_ssm else 0.0 for p in list_params_ssm
    ]

    # Capture loop variables explicitly to avoid closure-over-variable issues.
    def _factory(
        _onnx_file: str = onnx_file,
        _params: list[str] = list_params_ssm,
        _response: list[str] = response,
    ) -> Any:
        return _make_ssm_base_logp_from_onnx(_onnx_file, _params, _response)

    return {
        "ssm_base_logp_func_factory": _factory,
        "list_params_ssm": list_params_ssm,
        "bounds_ssm": bounds_ssm,
        "params_default_ssm": params_default_ssm,
        "response": response,
        "name": name,
    }


def _get_decision_process_spec(
    decision_process: str | dict[str, Any],
) -> dict[str, Any]:
    """Return a defensive copy of a decision-process specification.

    Custom SSMs (registered via :func:`register_ssm`) take precedence.  For
    everything else the spec is derived on the fly from HSSM's modelconfig
    system, meaning any built-in model with an ``approx_differentiable``
    likelihood (e.g. ``"ddm"``, ``"angle"``, ``"weibull"``) works out of the
    box without explicit registration.
    """
    if isinstance(decision_process, dict):
        return deepcopy(decision_process)

    # Custom registry takes precedence over modelconfig.
    if decision_process in _SSM_REGISTRY:
        spec = deepcopy(_SSM_REGISTRY[decision_process])
        spec["name"] = decision_process
        return spec

    # Fall back to HSSM's modelconfig for built-in SSMs.
    return _build_ssm_spec_from_modelconfig(decision_process)


def _get_ssm_logp(name: str) -> Any:
    """Return the annotated SSM base logp function, building it on first use.

    ONNX models are downloaded / loaded only when first called (lazy
    initialisation).  Subsequent calls return the cached object.
    """
    if name in _SSM_LOGP_CACHE:
        return _SSM_LOGP_CACHE[name]

    if name not in _SSM_REGISTRY:
        # Build from HSSM's modelconfig for built-in SSMs.
        spec = _build_ssm_spec_from_modelconfig(name)
        _SSM_LOGP_CACHE[name] = spec["ssm_base_logp_func_factory"]()
        return _SSM_LOGP_CACHE[name]

    entry = _SSM_REGISTRY[name]
    if "ssm_base_logp_func_factory" in entry:
        _SSM_LOGP_CACHE[name] = entry["ssm_base_logp_func_factory"]()
    else:
        # Pre-built function registered via register_ssm().
        _SSM_LOGP_CACHE[name] = entry["ssm_base_logp_func"]
    return _SSM_LOGP_CACHE[name]


# ---------------------------------------------------------------------------
# RLSSM named model registry
# ---------------------------------------------------------------------------
# Each entry provides:
#   decision_process            - key into _SSM_REGISTRY
#   learning_process_metadata   - LearningProcessMetadata for RL params and LP inputs
#   choices                     - response choice values
#   description                 - human-readable description
#   decision_process_loglik_kind


def _make_rescorla_wagner_learning_metadata() -> LearningProcessMetadata:
    """Return the shared learning metadata for built-in Rescorla-Wagner RLSSMs."""
    return LearningProcessMetadata(
        learning_process={"v": _compute_v_annotated},
        sampled_params=["rl_alpha", "scaler"],
        bounds={"rl_alpha": (0.0, 1.0), "scaler": (0.0, 10.0)},
        defaults=[0.1, 1.0],
        extra_fields=["feedback"],
        kind="blackbox",
    )


_RLSSM_REGISTRY: dict[str, RLSSMRegistryEntry] = {
    "2AB_RescorlaWagner_DDM": RLSSMRegistryEntry(
        decision_process="ddm",
        learning_process_metadata=_make_rescorla_wagner_learning_metadata(),
        choices=[0, 1],
        description=(
            "RLSSM model with Rescorla-Wagner Q-learning and the "
            "standard DDM as decision process."
        ),
        decision_process_loglik_kind="approx_differentiable",
    ),
    "2AB_RescorlaWagner_Angle": RLSSMRegistryEntry(
        decision_process="angle",
        learning_process_metadata=_make_rescorla_wagner_learning_metadata(),
        choices=[0, 1],
        description=(
            "RLSSM model with Rescorla-Wagner Q-learning and a "
            "collapsing-bound DDM (angle model) as decision process."
        ),
        decision_process_loglik_kind="approx_differentiable",
    ),
    "2AB_RescorlaWagner_Weibull": RLSSMRegistryEntry(
        decision_process="weibull",
        learning_process_metadata=_make_rescorla_wagner_learning_metadata(),
        choices=[0, 1],
        description=(
            "RLSSM model with Rescorla-Wagner Q-learning and a "
            "Weibull-bound DDM as decision process."
        ),
        decision_process_loglik_kind="approx_differentiable",
    ),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_ssm_logp_func(ssm_base_logp_func: Any, learning_process: dict) -> Any:
    """Re-annotate *ssm_base_logp_func* adding ``computed=learning_process``.

    Creates a new wrapper that carries the same ``.inputs`` and ``.outputs`` as
    the base function but adds the ``computed`` dict so that
    :func:`~hssm.rl.likelihoods.builder.make_rl_logp_op` can resolve which
    parameters are produced by the RL learning rule at runtime.
    """
    existing = getattr(ssm_base_logp_func, "computed", None)
    if existing:
        raise ValueError(
            "ssm_base_logp_func already has a non-empty .computed attribute. "
            "Pass the raw (base) annotated function without .computed instead."
        )
    return annotate_function(
        inputs=ssm_base_logp_func.inputs,
        outputs=ssm_base_logp_func.outputs,
        computed=learning_process,
    )(ssm_base_logp_func)


def _derive_lp_params(
    learning_process: dict[str, Any],
    response: list[str],
    extra_fields: list[str],
) -> list[str]:
    """Return sampled RL parameter names inferred from *learning_process*.

    Iterates over each LP function's ``.inputs`` and collects names that are
    neither response columns nor extra fields.
    """
    exclude = set(response) | set(extra_fields)
    learning_process_params: list[str] = []
    seen: set[str] = set()
    for param_name, lp_func in learning_process.items():
        if not hasattr(lp_func, "inputs"):
            _logger.warning(
                "Learning process function for '%s' has no .inputs attribute; "
                "cannot derive RL parameters from it. "
                "Ensure it is decorated with @annotate_function.",
                param_name,
            )
            continue
        for inp in lp_func.inputs:
            if inp not in exclude and inp not in seen:
                learning_process_params.append(inp)
                seen.add(inp)
    return learning_process_params


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def get_rlssm_model_config(
    model: str = "2AB_RescorlaWagner_DDM",
    choices: list[int] | None = None,
    learning_process: dict[str, Any] | None = None,
    decision_process: str | None = None,
) -> RLSSMConfig:
    """Build an :class:`~hssm.rl.config.RLSSMConfig` from a named model.

    Parameters
    ----------
    model:
        Name of a registered RLSSM model (e.g. ``"2AB_RescorlaWagner_DDM"``).
    choices:
        Override the response choice values stored in the registry.
    learning_process:
        Override the learning process dict stored in the registry.
    decision_process:
        Override the SSM name stored in the registry.

    Returns
    -------
    RLSSMConfig
        Fully populated configuration ready to be passed to :class:`_RLSSM`.

    Raises
    ------
    ValueError
        If *model* or the resolved *decision_process* is not registered.
    """
    if model not in _RLSSM_REGISTRY:
        available = list(_RLSSM_REGISTRY.keys())
        raise ValueError(
            f"Model '{model}' not found in the RLSSM registry. "
            f"Available models: {available}. "
            "To add a custom model, use register_rlssm_model() "
            "(and register_ssm() for custom decision processes), "
            "or pass 'model_config=' directly to RLSSM()."
        )

    entry = _RLSSM_REGISTRY[model].copy()
    learning_metadata = entry.learning_process_metadata.copy()

    if decision_process is not None:
        entry.decision_process = _get_decision_process_spec(decision_process)
    if choices is not None:
        entry.choices = list(choices)

    ssm_entry = _get_decision_process_spec(entry.decision_process)
    response = list(ssm_entry["response"])
    if learning_process is not None:
        learning_metadata = learning_metadata.with_learning_process_override(
            learning_process, response
        )

    dp: str = ssm_entry["name"]
    ssm_base = _get_ssm_logp(dp)
    lp: dict[str, Any] = dict(learning_metadata.learning_process)

    # Compose the full ssm_logp_func with .computed = learning_process.
    ssm_logp_func = _build_ssm_logp_func(ssm_base, lp)

    # list_params = [sampled RL params] + [sampled SSM params (non-computed)]
    ssm_sampled = [p for p in ssm_entry["list_params_ssm"] if p not in lp]
    learning_process_params = list(learning_metadata.sampled_params)
    list_params = learning_process_params + ssm_sampled

    # bounds: RL bounds ∪ SSM sampled bounds
    missing_bounds = [p for p in ssm_sampled if p not in ssm_entry["bounds_ssm"]]
    if missing_bounds:
        raise ValueError(
            f"SSM parameter(s) {missing_bounds} are included in list_params but have "
            f"no entry in bounds_ssm for decision process '{dp}'. "
            "Provide bounds for all sampled parameters via register_ssm() or ensure "
            "the built-in modelconfig includes them."
        )
    bounds: dict[str, tuple[float, float]] = dict(learning_metadata.bounds)
    for p in ssm_sampled:
        bounds[p] = ssm_entry["bounds_ssm"][p]

    # params_default aligned with list_params
    rl_defaults: list[float] = list(learning_metadata.defaults)
    ssm_all_defaults: list[float] = list(ssm_entry["params_default_ssm"])
    ssm_sampled_defaults = [
        ssm_all_defaults[i]
        for i, p in enumerate(ssm_entry["list_params_ssm"])
        if p not in lp
    ]
    params_default = rl_defaults + ssm_sampled_defaults

    return RLSSMConfig(
        model_name=entry.model_name or model,
        description=entry.description,
        decision_process=dp,
        decision_process_loglik_kind=entry.decision_process_loglik_kind,
        learning_process_kind=learning_metadata.kind,
        learning_process=lp,
        ssm_logp_func=ssm_logp_func,
        list_params=list_params,
        bounds=bounds,
        params_default=params_default,
        response=response,
        choices=tuple(entry.choices),
        extra_fields=list(learning_metadata.extra_fields)
        if learning_metadata.extra_fields is not None
        else None,
    )


# ---------------------------------------------------------------------------
# Public query helpers
# ---------------------------------------------------------------------------


def list_models() -> dict[str, str | None]:
    """Return the names and descriptions of all registered RLSSM models.

    This is the recommended starting point for new users who want to discover
    which models are available out of the box.

    Returns
    -------
    dict[str, str | None]
        Mapping of model name → description string (or ``None`` if no
        description was provided at registration time).

    Examples
    --------
    >>> import hssm
    >>> hssm.rl.list_models()
    {'2AB_RescorlaWagner_DDM': 'RLSSM model with Rescorla-Wagner ...', ...}
    """
    return {name: entry.description for name, entry in _RLSSM_REGISTRY.items()}


# ---------------------------------------------------------------------------
# Public registration helpers
# ---------------------------------------------------------------------------


def register_rlssm_model(
    name: str,
    decision_process: str,
    learning_process: dict[str, Any],
    learning_process_params: list[str],
    learning_process_bounds: dict[str, tuple[float, float]],
    learning_process_params_default: list[float],
    extra_fields: list[str] | None = None,
    choices: list[int] | None = None,
    description: str | None = None,
    decision_process_loglik_kind: str = "approx_differentiable",
    learning_process_kind: str = "blackbox",
) -> None:
    """Register a named RLSSM model in the global registry.

    Parameters
    ----------
    name:
        Registry key (e.g. ``"my_rldm"``).
    decision_process:
        Name of the SSM to use. This may be either a custom SSM already registered
        in the SSM registry via :func:`register_ssm`, or a built-in HSSM modelconfig SSM
        name such as ``"ddm"``, ``"angle"``, or ``"weibull"``.
    learning_process:
        Dict mapping computed parameter name → annotated learning function.
    learning_process_params:
        Ordered list of sampled RL parameter names.
    learning_process_bounds:
        Parameter bounds for the RL parameters.
    learning_process_params_default:
        Default values aligned with *learning_process_params*.
    extra_fields:
        Data column names required by the learning process (e.g. ``["feedback"]``).
    choices:
        Response choice values. Defaults to ``[0, 1]``.
    description:
        Optional human-readable description.
    decision_process_loglik_kind:
        Loglik kind tag. Defaults to ``"approx_differentiable"``.
    learning_process_kind:
        Learning process kind tag. Defaults to ``"blackbox"``.
    """
    if name in _RLSSM_REGISTRY:
        _logger.warning(
            "Model '%s' is already in the RLSSM registry and will be overwritten.",
            name,
        )
    learning_process_metadata = LearningProcessMetadata(
        learning_process=dict(learning_process),
        sampled_params=list(learning_process_params),
        bounds=dict(learning_process_bounds),
        defaults=list(learning_process_params_default),
        extra_fields=list(extra_fields) if extra_fields is not None else [],
        kind=learning_process_kind,
    )
    learning_process_metadata.validate()

    _RLSSM_REGISTRY[name] = RLSSMRegistryEntry(
        decision_process=decision_process,
        learning_process_metadata=learning_process_metadata,
        choices=list(choices) if choices is not None else [0, 1],
        description=description,
        decision_process_loglik_kind=decision_process_loglik_kind,
    )


def register_ssm(
    name: str,
    ssm_base_logp_func: Any,
    list_params_ssm: list[str],
    bounds_ssm: dict[str, tuple[float, float]],
    params_default_ssm: list[float],
    response: list[str] | None = None,
) -> None:
    """Register an SSM base log-likelihood function in the SSM registry.

    Parameters
    ----------
    name:
        Registry key (e.g. ``"ddm"``).
    ssm_base_logp_func:
        An annotated JAX function (created with ``@annotate_function``) that
        computes the SSM log-likelihood from a parameter matrix.  Must carry
        ``.inputs`` and ``.outputs`` attributes but should **not** have a
        ``.computed`` key — that is injected by the factory at config-build time.
    list_params_ssm:
        Ordered list of all SSM parameter names (including any that will be
        computed by the learning process).
    bounds_ssm:
        Bounds for the non-computed SSM parameters.
    params_default_ssm:
        Default values aligned with *list_params_ssm*.
    response:
        Data column names. Defaults to ``["rt", "response"]``.
    """
    if not callable(ssm_base_logp_func):
        raise ValueError(
            f"ssm_base_logp_func must be callable, got {type(ssm_base_logp_func)!r}."
        )
    if not hasattr(ssm_base_logp_func, "inputs") or not hasattr(
        ssm_base_logp_func, "outputs"
    ):
        raise ValueError(
            "ssm_base_logp_func must be decorated with @annotate_function "
            "(missing .inputs or .outputs attribute)."
        )
    existing_computed = getattr(ssm_base_logp_func, "computed", None)
    if existing_computed:
        raise ValueError(
            "ssm_base_logp_func should not have a non-empty .computed attribute "
            "at registration time. The .computed dict is injected later by "
            "get_rlssm_model_config() when composing the learning process. "
            "Pass the raw base function instead."
        )
    if name in _SSM_REGISTRY:
        _logger.warning(
            "SSM '%s' is already in the SSM registry and will be overwritten.", name
        )
    _SSM_REGISTRY[name] = {
        "ssm_base_logp_func": ssm_base_logp_func,
        "list_params_ssm": list(list_params_ssm),
        "bounds_ssm": dict(bounds_ssm),
        "params_default_ssm": list(params_default_ssm),
        "response": list(response) if response is not None else ["rt", "response"],
    }
    # Pre-built: cache immediately so _get_ssm_logp never calls a factory.
    _SSM_LOGP_CACHE[name] = ssm_base_logp_func
