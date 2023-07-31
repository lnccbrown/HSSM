"""A subclass of bmb.Prior that can handle bounds.

This class is a subclass of bmb.Prior, which is used to temporarily represent
a prior distribution until model build time. This class retains all functionalities
bmb.Prior but adds the following:

1. The ability to represent a truncated prior.
2. The ability to still print out the prior before the truncation.
3. The ability to shorten the output of bmb.Prior.
"""

from __future__ import annotations

from typing import Callable

import bambi as bmb
import numpy as np
import pymc as pm
from bambi.backend.utils import get_distribution
from bambi.priors.prior import format_arg

pymc_dist_args = ["rng", "initval", "dims", "observed", "total_size", "transform"]


# mypy: disable-error-code="has-type"
class Prior(bmb.Prior):
    """Abstract specification of a prior.

    Parameters
    ----------
    name
        Name of prior distribution. Must be the name of a PyMC distribution
        (e.g., ``"Normal"``, ``"Bernoulli"``, etc.)
    auto_scale : optional
        Whether to adjust the parameters of the prior or use them as passed.
        Default to ``True``.
    kwargs
        Optional keywords specifying the parameters of the named distribution.
    dist : optional
        A callable that returns a valid PyMC distribution. The signature must contain
        ``name``, ``dims``, and ``shape``, as well as its own keyworded arguments.
    bounds : optional
        A tuple of two floats indicating the lower and upper bounds of the prior.
    """

    def __init__(
        self,
        name: str,
        auto_scale: bool = True,
        dist: Callable | None = None,
        bounds: tuple[float, float] | None = None,
        **kwargs,
    ):
        bmb.Prior.__init__(self, name, auto_scale, dist, **kwargs)
        self.is_truncated = False
        self.bounds = bounds

        if self.bounds is not None:
            assert self.dist is None, (
                "We cannot bound a prior defined with the `dist` argument. The "
                + "`dist` and `bounds` arguments cannot both be supplied."
            )
            lower, upper = self.bounds
            if np.isinf(lower) and np.isinf(upper):
                return

            self.is_truncated = True
            self.dist = _make_truncated_dist(self.name, lower, upper, **self.args)
            self._args = self.args.copy()
            self.args: dict = {}

    def __str__(self) -> str:
        """Create the printout of the object."""
        args = self._args if self.is_truncated else self.args
        args_str = ", ".join(
            [
                f"{k}: {format_arg(v, 4)}"
                if not isinstance(v, type(self))
                else f"{k}: {v}"
                for k, v in args.items()
            ]
        )
        return f"{self.name}({args_str})"

    def __repr__(self) -> str:
        """Create the string representation of the object."""
        return self.__str__()

    def __eq__(self, other) -> bool:
        """Test equality."""
        if isinstance(other, Prior):
            if self.is_truncated and other.is_truncated:
                return (
                    self.name == other.name
                    and self._args == other._args
                    and self.bounds == other.bounds
                )
            if not self.is_truncated and not other.is_truncated:
                return self.name == other.name and self.args == other.args
            return False

        if isinstance(other, bmb.Prior):
            if self.is_truncated:
                return False
            if self.dist is not None:
                return self.name == other.name and self.dist == other.dist
            return self.name == other.name and self.args == other.args

        return False


def _make_truncated_dist(
    dist_name: str, lower_bound: float, upper_bound: float, **kwargs
) -> Callable:
    """Create custom functions with truncated priors.

    Helper function that creates a custom function with truncated priors.

    Parameters
    ----------
    lower_bound
        The lower bound for the distribution.
    upper_bound
        The upper bound for the distribution.
    kwargs
        Typically a dictionary with a name for the name of the Prior distribution
        and other arguments passed to bmb.Prior object.

    Returns
    -------
    Callable
        A distribution (TensorVariable) created with pm.Truncated().
    """
    truncated_kwargs = {k: kwargs.pop(k) for k in pymc_dist_args if k in kwargs}

    def TruncatedDist(name):
        dist = get_distribution(dist_name).dist(**kwargs)
        return pm.Truncated(
            name=name,
            dist=dist,
            lower=lower_bound if np.isfinite(lower_bound) else None,
            upper=upper_bound if np.isfinite(upper_bound) else None,
            **truncated_kwargs,
        )

    return TruncatedDist
