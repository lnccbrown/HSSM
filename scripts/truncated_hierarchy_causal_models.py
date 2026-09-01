"""Same-model parameterizations for the TruncatedNormal causal experiment.

The builders in this module all define the same natural-scale hierarchy::

    group_location ~ TruncatedNormal(base_mean, location_sigma, bounds)
    group_scale ~ Weibull(1.5, 0.3)
    group_effect[g] ~ TruncatedNormal(group_location, group_scale, bounds)
    y[i] ~ Normal(group_effect[group_index[i]], observation_sigma)

Only the coordinates used to represent the two truncated variables change.  This
lets the #1282 experiment distinguish a density implementation problem from a
centered-coordinate geometry problem without changing the statistical model.

``manual_centered`` codes the truncated density and bounded-coordinate Jacobian
explicitly.  The two inverse-CDF parameterizations map standard-Normal offsets
through ``Phi`` to Uniform quantiles, then use an independent Acklam
normal-quantile implementation composed only of PyTensor primitives that lower
to JAX.  No builder samples.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from scripts.truncated_hierarchy_models import (
    SCALE_PRIOR_ALPHA,
    SCALE_PRIOR_BETA,
    Bounds,
    GeometryContractError,
    NativeTruncatedPrior,
    SyntheticHierarchyData,
)

Parameterization = Literal[
    "native_centered",
    "manual_centered",
    "group_icdf_noncentered",
    "full_icdf_noncentered",
]

_LOG_2PI = math.log(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)

# Peter J. Acklam's inverse-normal rational approximation.  Keeping the
# coefficients here, rather than calling either PyMC's Normal or
# TruncatedNormal ``icdf``, gives the causal experiment an independent path.
_ACKLAM_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_ACKLAM_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_ACKLAM_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_ACKLAM_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_ACKLAM_LOW = 0.02425


class CausalModelContractError(GeometryContractError):
    """Raised when an input would make the four models non-isomorphic."""


def _constant(value: float) -> Any:
    """Return a scalar in the active PyTensor floating-point dtype."""
    return pt.as_tensor_variable(np.asarray(value, dtype=pytensor.config.floatX))


def _polyval(coefficients: tuple[float, ...], value: Any) -> Any:
    result = _constant(coefficients[0])
    for coefficient in coefficients[1:]:
        result = result * value + _constant(coefficient)
    return result


def _standard_normal_lower_icdf(
    probability: Any | None = None,
    *,
    log_probability: Any | None = None,
) -> Any:
    """Approximate ``Phi^-1(p)`` from ``p`` or ``log(p)`` for ``p <= 0.5``.

    Acklam's tail and central rational approximations avoid ``erfinv`` and
    ``erfcinv``.  The latter currently lacks a native PyTensor-to-JAX lowering.
    The log-probability path remains finite when an ordinary probability would
    underflow.  A Newton step on ``log(Phi(x))`` retains that stability.
    """
    if (probability is None) == (log_probability is None):
        raise CausalModelContractError(
            "provide exactly one of probability or log_probability"
        )
    if log_probability is None:
        if probability is None:  # pragma: no cover - guarded above
            raise AssertionError("unreachable missing probability")
        p = pt.as_tensor_variable(probability)
        log_p = pt.log(p)
    else:
        log_p = pt.as_tensor_variable(log_probability)
        p = pt.exp(log_p)
    log_p = pt.minimum(log_p, _constant(math.log(0.5)))

    tail_argument = pt.sqrt(-_constant(2.0) * log_p)
    tail = _polyval(_ACKLAM_C, tail_argument) / (
        _polyval(_ACKLAM_D, tail_argument) * tail_argument + _constant(1.0)
    )

    centered = p - _constant(0.5)
    squared = centered * centered
    central = (
        _polyval(_ACKLAM_A, squared)
        * centered
        / (_polyval(_ACKLAM_B, squared) * squared + _constant(1.0))
    )
    approximation = pt.switch(
        pt.lt(log_p, _constant(math.log(_ACKLAM_LOW))), tail, central
    )
    # One Newton step brings Acklam's ~1e-9 approximation to floating-point
    # precision.  Solving in log-CDF space avoids underflow in the far tail.
    approximation_logcdf = _normal_logcdf(approximation)
    approximation_logpdf = -_constant(0.5) * approximation * approximation - _constant(
        0.5 * _LOG_2PI
    )
    logcdf_derivative = pt.exp(approximation_logpdf - approximation_logcdf)
    return approximation - (approximation_logcdf - log_p) / logcdf_derivative


def _standard_normal_logcdf_and_logsurvival(value: Any) -> tuple[Any, Any]:
    """Return log-CDF and log-survival without ordinary-probability underflow."""
    value = pt.as_tensor_variable(value)
    return _normal_logcdf(value), _normal_logcdf(-value)


def truncated_normal_icdf(
    quantile: Any,
    *,
    mu: Any,
    sigma: Any,
    bounds: Bounds,
    quantile_survival: Any | None = None,
    log_quantile: Any | None = None,
    log_quantile_survival: Any | None = None,
) -> Any:
    """Map Uniform quantiles to an exact truncated Normal natural variable.

    Both the target CDF and target survival probability are interpolated
    directly.  Selecting the smaller one before applying the symmetric
    inverse-normal approximation avoids cancellation in either tail.  ``upper``
    may be infinite; the causal study deliberately requires a finite lower bound.
    """
    if bounds.lower is None:
        raise CausalModelContractError(
            "causal TruncatedNormal models require a finite lower bound"
        )
    quantile = pt.as_tensor_variable(quantile)
    mu = pt.as_tensor_variable(mu)
    sigma = pt.as_tensor_variable(sigma)
    lower_standard = (_constant(bounds.lower) - mu) / sigma
    lower_logcdf, lower_logsurvival = _standard_normal_logcdf_and_logsurvival(
        lower_standard
    )

    if bounds.upper is None:
        upper_logcdf = _constant(0.0)
        upper_logsurvival = _constant(-np.inf)
    else:
        upper_standard = (_constant(bounds.upper) - mu) / sigma
        upper_logcdf, upper_logsurvival = _standard_normal_logcdf_and_logsurvival(
            upper_standard
        )

    complement = (
        _constant(1.0) - quantile
        if quantile_survival is None
        else pt.as_tensor_variable(quantile_survival)
    )
    log_q = pt.log(quantile) if log_quantile is None else log_quantile
    log_complement = (
        pt.log(complement) if log_quantile_survival is None else log_quantile_survival
    )
    target_logcdf = pt.logaddexp(
        log_complement + lower_logcdf,
        log_q + upper_logcdf,
    )
    target_logsurvival = pt.logaddexp(
        log_complement + lower_logsurvival,
        log_q + upper_logsurvival,
    )
    from_lower = _standard_normal_lower_icdf(log_probability=target_logcdf)
    from_upper = -_standard_normal_lower_icdf(log_probability=target_logsurvival)
    standard_value = pt.switch(
        pt.le(target_logcdf, target_logsurvival), from_lower, from_upper
    )
    return mu + sigma * standard_value


def _truncated_normal_from_standard_normal(
    offset: Any,
    *,
    mu: Any,
    sigma: Any,
    bounds: Bounds,
) -> Any:
    """Map a standard-Normal offset through its Uniform CDF to the TN inverse."""
    log_quantile, log_quantile_survival = _standard_normal_logcdf_and_logsurvival(
        offset
    )
    quantile = pt.exp(log_quantile)
    quantile_survival = pt.exp(log_quantile_survival)
    return truncated_normal_icdf(
        quantile,
        mu=mu,
        sigma=sigma,
        bounds=bounds,
        quantile_survival=quantile_survival,
        log_quantile=log_quantile,
        log_quantile_survival=log_quantile_survival,
    )


def bounded_from_unconstrained(coordinate: Any, bounds: Bounds) -> Any:
    """Apply the same lower/interval mapping used by PyMC's Interval transform."""
    if bounds.lower is None:
        raise CausalModelContractError(
            "causal TruncatedNormal models require a finite lower bound"
        )
    coordinate = pt.as_tensor_variable(coordinate)
    if bounds.upper is None:
        return _constant(bounds.lower) + pt.exp(coordinate)
    width = bounds.width
    if width is None:  # pragma: no cover - narrowed by the finite upper bound
        raise CausalModelContractError("finite bounds must have a finite width")
    return _constant(bounds.lower) + _constant(width) * pt.sigmoid(coordinate)


def bounded_log_jacobian(coordinate: Any, bounds: Bounds) -> Any:
    """Return ``log(abs(dx/dz))`` for :func:`bounded_from_unconstrained`."""
    if bounds.lower is None:
        raise CausalModelContractError(
            "causal TruncatedNormal models require a finite lower bound"
        )
    coordinate = pt.as_tensor_variable(coordinate)
    if bounds.upper is None:
        return coordinate
    width = bounds.width
    if width is None:  # pragma: no cover - narrowed by the finite upper bound
        raise CausalModelContractError("finite bounds must have a finite width")
    return (
        _constant(math.log(width)) - pt.softplus(-coordinate) - pt.softplus(coordinate)
    )


def _normal_logcdf(value: Any) -> Any:
    """Compute a standard-normal log-CDF without tail underflow."""
    value = pt.as_tensor_variable(value)
    scaled = value / _constant(_SQRT_2)
    # Keep each inactive branch in its numerically safe half-plane.  This also
    # prevents an overflowing erfcx value from contaminating autodiff through
    # an elementwise switch.  Explicit switches preserve the selected branch's
    # one-sided derivative at zero; ``maximum(x, 0)`` would give JAX's 1/2
    # subgradient there and corrupt the non-centered transform.
    negative = pt.lt(value, _constant(0.0))
    tail_argument = pt.switch(negative, -scaled, _constant(0.0))
    central_argument = pt.switch(negative, _constant(0.0), scaled)
    tail = (
        _constant(math.log(0.5))
        - _constant(0.5) * value * value
        + pt.log(pt.erfcx(tail_argument))
    )
    central = pt.log1p(-_constant(0.5) * pt.erfc(central_argument))
    return pt.switch(negative, tail, central)


def _logdiffexp(log_big: Any, log_small: Any) -> Any:
    """Compute ``log(exp(big) - exp(small))`` without cancellation."""
    return log_big + pt.log1mexp(log_small - log_big)


def manual_truncated_normal_logp(
    value: Any,
    *,
    mu: Any,
    sigma: Any,
    bounds: Bounds,
) -> Any:
    """Independently code the normalized truncated-Normal natural log-density."""
    if bounds.lower is None:
        raise CausalModelContractError(
            "causal TruncatedNormal models require a finite lower bound"
        )
    value = pt.as_tensor_variable(value)
    mu = pt.as_tensor_variable(mu)
    sigma = pt.as_tensor_variable(sigma)
    lower_standard = (_constant(bounds.lower) - mu) / sigma
    log_survival_lower = _normal_logcdf(-lower_standard)
    in_support = pt.gt(value, _constant(bounds.lower))

    if bounds.upper is None:
        log_normalizer = log_survival_lower
    else:
        upper_standard = (_constant(bounds.upper) - mu) / sigma
        log_cdf_lower = _normal_logcdf(lower_standard)
        log_cdf_upper = _normal_logcdf(upper_standard)
        log_survival_upper = _normal_logcdf(-upper_standard)
        from_cdf = _logdiffexp(log_cdf_upper, log_cdf_lower)
        from_survival = _logdiffexp(log_survival_lower, log_survival_upper)
        log_normalizer = pt.switch(
            pt.ge(lower_standard, _constant(0.0)), from_survival, from_cdf
        )
        in_support = in_support & pt.lt(value, _constant(bounds.upper))

    standardized = (value - mu) / sigma
    logp = (
        -_constant(0.5) * standardized * standardized
        - pt.log(sigma)
        - _constant(0.5 * _LOG_2PI)
        - log_normalizer
    )
    valid = in_support & pt.gt(sigma, _constant(0.0))
    return pt.switch(valid, logp, _constant(-np.inf))


def _validate_inputs(prior: NativeTruncatedPrior, data: SyntheticHierarchyData) -> None:
    if prior.bounds != data.spec.bounds:
        raise CausalModelContractError("prior and data bounds must be identical")
    if prior.bounds.lower is None:
        raise CausalModelContractError(
            "causal TruncatedNormal models require a finite lower bound"
        )
    if not math.isclose(prior.scale_prior_alpha, SCALE_PRIOR_ALPHA):
        raise CausalModelContractError(
            f"group_scale alpha must remain frozen at {SCALE_PRIOR_ALPHA}"
        )
    if not math.isclose(prior.scale_prior_beta, SCALE_PRIOR_BETA):
        raise CausalModelContractError(
            f"group_scale beta must remain frozen at {SCALE_PRIOR_BETA}"
        )


def _add_group_scale(floatx: str) -> Any:
    scale_rv = pm.Weibull(
        "group_scale_rv",
        alpha=np.asarray(SCALE_PRIOR_ALPHA, dtype=floatx),
        beta=np.asarray(SCALE_PRIOR_BETA, dtype=floatx),
    )
    return pm.Deterministic("group_scale", scale_rv)


def _add_likelihood(group_effect: Any, data: SyntheticHierarchyData) -> None:
    if not data.y.size:
        return
    pm.Normal(
        "y",
        mu=group_effect[data.group_index],
        sigma=np.asarray(data.spec.observation_sigma, dtype=data.spec.floatx),
        observed=data.y,
        dims="observation",
    )


def _model_coords(data: SyntheticHierarchyData) -> dict[str, Any]:
    coords: dict[str, Any] = {"group": data.group_labels}
    if data.y.size:
        coords["observation"] = np.arange(data.y.size)
    return coords


def build_native_centered(
    prior: NativeTruncatedPrior, data: SyntheticHierarchyData
) -> pm.Model:
    """Build the native centered PyMC reference model."""
    _validate_inputs(prior, data)
    lower, upper = prior.bounds.pymc_limits()
    with pytensor.config.change_flags(floatX=data.spec.floatx):
        with pm.Model(coords=_model_coords(data)) as model:
            location_rv = pm.TruncatedNormal(
                "group_location_rv",
                mu=np.asarray(prior.location_base_mean, dtype=data.spec.floatx),
                sigma=np.asarray(prior.location_prior_sigma, dtype=data.spec.floatx),
                lower=lower,
                upper=upper,
            )
            location = pm.Deterministic("group_location", location_rv)
            scale = _add_group_scale(data.spec.floatx)
            effect_rv = pm.TruncatedNormal(
                "group_effect_rv",
                mu=location,
                sigma=scale,
                lower=lower,
                upper=upper,
                dims="group",
            )
            effect = pm.Deterministic("group_effect", effect_rv, dims="group")
            _add_likelihood(effect, data)
    return model


def build_manual_centered(
    prior: NativeTruncatedPrior, data: SyntheticHierarchyData
) -> pm.Model:
    """Build the centered model from Flat coordinates and explicit log-density."""
    _validate_inputs(prior, data)
    with pytensor.config.change_flags(floatX=data.spec.floatx):
        with pm.Model(coords=_model_coords(data)) as model:
            location_coordinate = pm.Flat("group_location_coordinate")
            location = pm.Deterministic(
                "group_location",
                bounded_from_unconstrained(location_coordinate, prior.bounds),
            )
            pm.Potential(
                "group_location_density",
                manual_truncated_normal_logp(
                    location,
                    mu=np.asarray(prior.location_base_mean, dtype=data.spec.floatx),
                    sigma=np.asarray(
                        prior.location_prior_sigma, dtype=data.spec.floatx
                    ),
                    bounds=prior.bounds,
                )
                + bounded_log_jacobian(location_coordinate, prior.bounds),
            )
            scale = _add_group_scale(data.spec.floatx)
            effect_coordinate = pm.Flat("group_effect_coordinate", dims="group")
            effect = pm.Deterministic(
                "group_effect",
                bounded_from_unconstrained(effect_coordinate, prior.bounds),
                dims="group",
            )
            pm.Potential(
                "group_effect_density",
                pt.sum(
                    manual_truncated_normal_logp(
                        effect,
                        mu=location,
                        sigma=scale,
                        bounds=prior.bounds,
                    )
                    + bounded_log_jacobian(effect_coordinate, prior.bounds)
                ),
            )
            _add_likelihood(effect, data)
    return model


def build_group_icdf_noncentered(
    prior: NativeTruncatedPrior, data: SyntheticHierarchyData
) -> pm.Model:
    """Keep the location centered and inverse-CDF non-center group effects."""
    _validate_inputs(prior, data)
    lower, upper = prior.bounds.pymc_limits()
    with pytensor.config.change_flags(floatX=data.spec.floatx):
        with pm.Model(coords=_model_coords(data)) as model:
            location_rv = pm.TruncatedNormal(
                "group_location_rv",
                mu=np.asarray(prior.location_base_mean, dtype=data.spec.floatx),
                sigma=np.asarray(prior.location_prior_sigma, dtype=data.spec.floatx),
                lower=lower,
                upper=upper,
            )
            location = pm.Deterministic("group_location", location_rv)
            scale = _add_group_scale(data.spec.floatx)
            effect_offset = pm.Normal(
                "group_effect_offset", mu=0.0, sigma=1.0, dims="group"
            )
            effect = pm.Deterministic(
                "group_effect",
                _truncated_normal_from_standard_normal(
                    effect_offset,
                    mu=location,
                    sigma=scale,
                    bounds=prior.bounds,
                ),
                dims="group",
            )
            _add_likelihood(effect, data)
    return model


def build_full_icdf_noncentered(
    prior: NativeTruncatedPrior, data: SyntheticHierarchyData
) -> pm.Model:
    """Inverse-CDF non-center both the location and the group effects."""
    _validate_inputs(prior, data)
    with pytensor.config.change_flags(floatX=data.spec.floatx):
        with pm.Model(coords=_model_coords(data)) as model:
            location_offset = pm.Normal("group_location_offset", mu=0.0, sigma=1.0)
            location = pm.Deterministic(
                "group_location",
                _truncated_normal_from_standard_normal(
                    location_offset,
                    mu=np.asarray(prior.location_base_mean, dtype=data.spec.floatx),
                    sigma=np.asarray(
                        prior.location_prior_sigma, dtype=data.spec.floatx
                    ),
                    bounds=prior.bounds,
                ),
            )
            scale = _add_group_scale(data.spec.floatx)
            effect_offset = pm.Normal(
                "group_effect_offset", mu=0.0, sigma=1.0, dims="group"
            )
            effect = pm.Deterministic(
                "group_effect",
                _truncated_normal_from_standard_normal(
                    effect_offset,
                    mu=location,
                    sigma=scale,
                    bounds=prior.bounds,
                ),
                dims="group",
            )
            _add_likelihood(effect, data)
    return model


def build_causal_model(
    parameterization: Parameterization,
    prior: NativeTruncatedPrior,
    data: SyntheticHierarchyData,
) -> pm.Model:
    """Dispatch to one of the four frozen same-natural-model builders."""
    builders = {
        "native_centered": build_native_centered,
        "manual_centered": build_manual_centered,
        "group_icdf_noncentered": build_group_icdf_noncentered,
        "full_icdf_noncentered": build_full_icdf_noncentered,
    }
    return builders[parameterization](prior, data)
