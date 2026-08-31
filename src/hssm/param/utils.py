"""Utility functions for the parameter classes."""

import logging

import bambi as bmb
import numpy as np

_logger = logging.getLogger("hssm")


def validate_bounds(bounds: tuple[float, float]) -> None:
    """Validate the bounds."""
    if len(bounds) != 2:
        raise ValueError(f"Invalid bounds: {bounds}")
    lower, upper = bounds
    if lower >= upper:
        raise ValueError(f"Invalid bounds: {bounds}")


def _make_default_prior(bounds: tuple[float, float] | None) -> bmb.Prior:
    """Make a default prior from bounds.

    Parameters
    ----------
    bounds
        The (lower, upper) bounds for the default prior.

    Returns
    -------
        A bmb.Prior object representing the default prior for the provided bounds.
    """
    if bounds is None:
        raise ValueError("Bounds parameter unspecified.")
    lower, upper = bounds
    if np.isinf(lower) and np.isinf(upper):
        prior = bmb.Prior("Normal", mu=0.0, sigma=2.0)
    elif np.isinf(lower) and not np.isinf(upper):
        prior = bmb.Prior("TruncatedNormal", mu=upper, upper=upper, sigma=2.0)
    elif not np.isinf(lower) and np.isinf(upper):
        if lower == 0:
            prior = bmb.Prior("HalfNormal", sigma=2.0)
        else:
            prior = bmb.Prior("TruncatedNormal", mu=lower, lower=lower, sigma=2.0)
    else:
        prior = bmb.Prior(name="Uniform", lower=lower, upper=upper)

    return prior


def _clamp_default_initval_to_bounds(
    value: float, name: str, bounds: tuple[float, float] | None
) -> float:
    """Clamp a default initial value into a parameter's declared bounds.

    The value is returned unchanged when it lies strictly inside ``bounds``;
    otherwise it is moved to the point 5% of the bound width inside the violated
    endpoint. The defaults in ``INITVAL_SETTINGS`` are shared across models, so a
    model's declared bounds may exclude them, and a start outside the bounds has
    -inf log-probability from which sampling cannot move. This applies only to
    defaults on the natural scale (the ``None``-link branch); user-supplied
    initial values are never touched.

    Parameters
    ----------
    value
        The default initial value, on the natural scale of the parameter.
    name
        The parameter name as it appears in the model's initial point. Used only
        in the warning emitted when the value is moved.
    bounds
        The parameter's ``(lower, upper)`` bounds, or ``None`` if the parameter
        declares none.

    Returns
    -------
        ``value`` itself when it lies strictly inside ``bounds``, or when
        ``bounds`` is ``None``; otherwise a finite value strictly inside
        ``bounds``, a distance of 5% of the bound width from the violated
        endpoint.
    """
    if bounds is None:
        return value
    lower, upper = bounds
    if lower < value < upper:
        return value
    # A one-sided bound - a: (0, inf), st: (0, inf) - makes the width infinite,
    # and a margin proportional to it would place the result at +/-inf. Scale
    # the margin off whichever endpoint is finite instead, so the clamped value
    # is always finite and strictly inside the bounds.
    width = upper - lower
    if np.isfinite(width):
        margin = 0.05 * width
    else:
        # The 1.0 floor keeps the margin strictly positive when the only finite
        # endpoint is 0.0, as in (0, inf); a margin of 0 there would clamp onto
        # the excluded boundary itself.
        finite_endpoints = [abs(b) for b in (lower, upper) if np.isfinite(b)]
        margin = 0.05 * max(finite_endpoints + [1.0])
    clamped = float(np.clip(value, lower + margin, upper - margin))
    _logger.warning(
        "Default initial value %s for %s lies outside the declared bounds "
        "(%s, %s); using %s instead. Pass an explicit initval to override.",
        value,
        name,
        lower,
        upper,
        clamped,
    )
    return clamped
