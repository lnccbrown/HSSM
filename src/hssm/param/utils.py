"""Utility functions for the parameter classes."""

import bambi as bmb
import numpy as np


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
        raise ValueError("Parameter unspecified.")
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
