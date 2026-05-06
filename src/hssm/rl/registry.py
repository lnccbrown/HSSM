"""Registry for named RLSSM models and SSM base log-likelihood functions.

This module provides:

- :data:`_SSM_REGISTRY` — maps SSM names (e.g. ``"angle"``) to their base
  annotated JAX log-likelihood functions and parameter metadata.
- :data:`_RLSSM_REGISTRY` — maps named RLSSM model strings (e.g. ``"rldm"``)
  to their default decision process, learning process, and parameter info.
- :func:`get_rlssm_model_config` — builds a :class:`~hssm.rl.config.RLSSMConfig`
  from a named model string with optional overrides.
- :func:`register_rlssm_model` — register a custom named RLSSM model.
- :func:`register_ssm` — register a custom SSM base logp function.
"""

from __future__ import annotations

import logging
from typing import Any

from hssm.distribution_utils.onnx import make_jax_matrix_logp_funcs_from_onnx
from hssm.rl.likelihoods.two_armed_bandit import compute_v_subject_wise
from hssm.utils import annotate_function

from .config import RLSSMConfig

_logger = logging.getLogger("hssm")

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
# Each entry provides:
#   ssm_base_logp_func  - annotated JAX function (inputs + outputs, no computed)
#   list_params_ssm     - ordered SSM parameter names (including computed ones)
#   bounds_ssm          - bounds for non-computed SSM params
#   params_default_ssm  - default values aligned with list_params_ssm
#   response            - data column names


def _make_angle_base_logp() -> Any:
    """Build the annotated angle SSM base logp function from its ONNX file."""
    _raw = make_jax_matrix_logp_funcs_from_onnx(model="angle.onnx")
    return annotate_function(
        inputs=["v", "a", "z", "t", "theta", "rt", "response"],
        outputs=["logp"],
    )(_raw)


_SSM_REGISTRY: dict[str, dict[str, Any]] = {
    "angle": {
        # Factory callable — invoked lazily on first use via _get_ssm_logp().
        # Storing a callable (not the result) avoids loading angle.onnx at
        # import time (which would trigger hf_hub_download in offline envs).
        "ssm_base_logp_func_factory": _make_angle_base_logp,
        # All SSM params in the order the SSM expects them (includes computed).
        "list_params_ssm": ["v", "a", "z", "t", "theta"],
        # Bounds only for params that will be *sampled* (not RL-computed).
        "bounds_ssm": {
            "a": (0.3, 3.0),
            "z": (0.1, 0.9),
            "t": (0.001, 2.0),
            "theta": (-0.1, 1.3),
        },
        # Defaults aligned with list_params_ssm: v, a, z, t, theta
        "params_default_ssm": [0.0, 1.5, 0.5, 0.5, 0.0],
        "response": ["rt", "response"],
    },
}

# Cache for resolved SSM base logp functions (populated on first use by
# _get_ssm_logp, or immediately by register_ssm when a pre-built func is
# supplied by the caller).
_SSM_LOGP_CACHE: dict[str, Any] = {}


def _get_ssm_logp(name: str) -> Any:
    """Return the annotated SSM base logp function, building it on first use.

    For built-in SSMs the ONNX model is downloaded / loaded only when this
    function is first called (lazy initialisation).  Subsequent calls return
    the cached object without any I/O.
    """
    if name not in _SSM_LOGP_CACHE:
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
#   learning_process            - {param: annotated_func}
#   rl_params                   - ordered list of sampled RL parameter names
#   rl_bounds                   - {param: (lo, hi)} for RL params
#   rl_params_default           - default values aligned with rl_params
#   extra_fields                - extra data column names required by LP
#   choices                     - response choice values
#   description                 - human-readable description
#   decision_process_loglik_kind
#   learning_process_kind

_RLSSM_REGISTRY: dict[str, dict[str, Any]] = {
    "rldm": {
        "decision_process": "angle",
        "learning_process": {"v": _compute_v_annotated},
        "rl_params": ["rl_alpha", "scaler"],
        "rl_bounds": {
            "rl_alpha": (0.0, 1.0),
            "scaler": (0.0, 10.0),
        },
        "rl_params_default": [0.1, 1.0],
        "extra_fields": ["feedback"],
        "choices": [0, 1],
        "description": (
            "RLSSM model with Rescorla-Wagner Q-learning and a "
            "collapsing-bound DDM (angle model) as decision process."
        ),
        "decision_process_loglik_kind": "approx_differentiable",
        "learning_process_kind": "blackbox",
    },
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
    return annotate_function(
        inputs=ssm_base_logp_func.inputs,
        outputs=ssm_base_logp_func.outputs,
        computed=learning_process,
    )(ssm_base_logp_func)


def _derive_rl_params(
    learning_process: dict[str, Any],
    response: list[str],
    extra_fields: list[str],
) -> list[str]:
    """Return sampled RL parameter names inferred from *learning_process*.

    Iterates over each LP function's ``.inputs`` and collects names that are
    neither response columns nor extra fields.
    """
    exclude = set(response) | set(extra_fields)
    rl_params: list[str] = []
    seen: set[str] = set()
    for lp_func in learning_process.values():
        if not hasattr(lp_func, "inputs"):
            continue
        for inp in lp_func.inputs:
            if inp not in exclude and inp not in seen:
                rl_params.append(inp)
                seen.add(inp)
    return rl_params


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def get_rlssm_model_config(
    model: str = "rldm",
    choices: list[int] | None = None,
    learning_process: dict[str, Any] | None = None,
    decision_process: str | None = None,
) -> RLSSMConfig:
    """Build an :class:`~hssm.rl.config.RLSSMConfig` from a named model.

    Parameters
    ----------
    model:
        Name of a registered RLSSM model (e.g. ``"rldm"``).
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
            "To use a custom model, pass 'model_config=' directly."
        )

    # Shallow-copy so overrides don't mutate the registry entry.
    entry = dict(_RLSSM_REGISTRY[model])

    if learning_process is not None:
        entry["learning_process"] = learning_process
    if decision_process is not None:
        entry["decision_process"] = decision_process
    if choices is not None:
        entry["choices"] = choices

    dp: str = entry["decision_process"]
    if dp not in _SSM_REGISTRY:
        available_ssms = list(_SSM_REGISTRY.keys())
        raise ValueError(
            f"Decision process '{dp}' not found in the SSM registry. "
            f"Available: {available_ssms}. Use register_ssm() to add it."
        )

    ssm_entry = _SSM_REGISTRY[dp]
    ssm_base = _get_ssm_logp(dp)
    lp: dict[str, Any] = entry["learning_process"]

    # Compose the full ssm_logp_func with .computed = learning_process.
    ssm_logp_func = _build_ssm_logp_func(ssm_base, lp)

    # list_params = [sampled RL params] + [sampled SSM params (non-computed)]
    computed_set = set(lp.keys())
    ssm_sampled = [p for p in ssm_entry["list_params_ssm"] if p not in computed_set]

    rl_params: list[str] = entry.get("rl_params") or _derive_rl_params(
        lp, ssm_entry["response"], entry.get("extra_fields") or []
    )
    list_params = rl_params + ssm_sampled

    # bounds: RL bounds ∪ SSM sampled bounds
    bounds: dict[str, tuple[float, float]] = dict(entry.get("rl_bounds") or {})
    for p in ssm_sampled:
        if p in ssm_entry["bounds_ssm"]:
            bounds[p] = ssm_entry["bounds_ssm"][p]

    # params_default aligned with list_params
    rl_defaults: list[float] = list(entry.get("rl_params_default") or [])
    ssm_all_defaults: list[float] = list(ssm_entry["params_default_ssm"])
    ssm_sampled_defaults = [
        ssm_all_defaults[i]
        for i, p in enumerate(ssm_entry["list_params_ssm"])
        if p not in computed_set
    ]
    params_default = rl_defaults + ssm_sampled_defaults

    return RLSSMConfig(
        model_name=entry.get("model_name", model),
        description=entry.get("description"),
        decision_process=dp,
        decision_process_loglik_kind=entry["decision_process_loglik_kind"],
        learning_process_kind=entry["learning_process_kind"],
        learning_process=lp,
        ssm_logp_func=ssm_logp_func,
        list_params=list_params,
        bounds=bounds,
        params_default=params_default,
        response=ssm_entry["response"],
        choices=entry["choices"],
        extra_fields=entry.get("extra_fields"),
    )


# ---------------------------------------------------------------------------
# Public registration helpers
# ---------------------------------------------------------------------------


def register_rlssm_model(
    name: str,
    decision_process: str,
    learning_process: dict[str, Any],
    rl_params: list[str],
    rl_bounds: dict[str, tuple[float, float]],
    rl_params_default: list[float],
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
        Name of the SSM to use (must already be in the SSM registry).
    learning_process:
        Dict mapping computed parameter name → annotated learning function.
    rl_params:
        Ordered list of sampled RL parameter names.
    rl_bounds:
        Parameter bounds for the RL parameters.
    rl_params_default:
        Default values aligned with *rl_params*.
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
    _RLSSM_REGISTRY[name] = {
        "decision_process": decision_process,
        "learning_process": learning_process,
        "rl_params": rl_params,
        "rl_bounds": rl_bounds,
        "rl_params_default": rl_params_default,
        "extra_fields": extra_fields or [],
        "choices": choices if choices is not None else [0, 1],
        "description": description,
        "decision_process_loglik_kind": decision_process_loglik_kind,
        "learning_process_kind": learning_process_kind,
    }


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
    if name in _SSM_REGISTRY:
        _logger.warning(
            "SSM '%s' is already in the SSM registry and will be overwritten.", name
        )
    _SSM_REGISTRY[name] = {
        "ssm_base_logp_func": ssm_base_logp_func,
        "list_params_ssm": list_params_ssm,
        "bounds_ssm": bounds_ssm,
        "params_default_ssm": params_default_ssm,
        "response": response or ["rt", "response"],
    }
    # Pre-built: cache immediately so _get_ssm_logp never calls a factory.
    _SSM_LOGP_CACHE[name] = ssm_base_logp_func
