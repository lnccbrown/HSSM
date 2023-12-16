"""A subclass of bmb.Prior that can handle bounds.

This class is a subclass of bmb.Prior, which is used to temporarily represent
a prior distribution until model build time. This class retains all functionalities
bmb.Prior but adds the following:

1. The ability to represent a truncated prior.
2. The ability to still print out the prior before the truncation.
3. The ability to shorten the output of bmb.Prior.
"""
from copy import deepcopy
from statistics import mean
from typing import Any, Callable

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


def generate_prior(
    dist: str | dict | int | float | Prior,
    bounds: tuple[float, float] | None = None,
    **kwargs,
):
    """Generate a Prior distribution.

    The parameter ``kwargs`` is used to pass hyperpriors that are assigned to the
    parameters of the prior to be built.

    This function is taken from bambi.priors.prior.py and modified to handle bounds.

    Parameters
    ----------
    dist:
        If a string, it is the name of the prior distribution with default values taken
        from ``SETTINGS_DISTRIBUTIONS``. If a number, it is a factor used to scale the
        standard deviation of the priors generated automatically by Bambi. If a `dict`,
        it must contain a ``"dist"`` key with the name of the distribution and other
        keys.
    bounds: optional
        A tuple of two floats indicating the lower and upper bounds of the prior.

    Raises
    ------
    ValueError
        If ``dist`` is not a string, number, or dict.

    Returns
    -------
    Prior
        The Prior instance.
    """
    if isinstance(dist, str):
        default_settings = deepcopy(HSSM_SETTINGS_DISTRIBUTIONS[dist])
        if kwargs:
            for k, v in kwargs.items():
                default_settings[k] = generate_prior(v)
        prior: Prior | int | float = Prior(dist, bounds=bounds, **default_settings)
    elif isinstance(dist, dict):
        prior_settings = deepcopy(dist)
        dist_name: str = prior_settings.pop("dist")
        for k, v in prior_settings.items():
            prior_settings[k] = generate_prior(v)
        prior = Prior(dist_name, bounds=bounds, **prior_settings)
    elif isinstance(dist, Prior):
        prior = dist
    elif isinstance(dist, (int, float)):
        if bounds is not None:
            lower, upper = bounds
            if dist < lower or dist > upper:
                raise ValueError(
                    f"The prior value {dist} is outside the bounds {bounds}."
                )
        prior = dist
    else:
        raise ValueError(
            "'dist' must be the name of a distribution or a numeric value."
        )
    return prior


def get_default_prior(term_type: str, bounds: tuple[float, float] | None):
    """Generate a Prior based on the default settings.

    The following summarizes default priors for each type of term:

    * common_intercept: Bounded Normal prior (N(mean(bounds), 0.25)).
    * common: Normal prior (N(0, 0.25)).
    * group_intercept: Normal prior N(N(0, 0.25), Weibull(1.5, 0.3). It's supposed to
    be bounded but Bambi does not fully support it yet.
    * group_specific: Normal prior N(N(0, 0.25), Weibull(1.5, 0.3).

    This function is taken from bambi.priors.prior.py and modified to handle hssm-
    specific situations.

    Parameters
    ----------
    term_type : str
        The type of the term for which the default prior is wanted.
    bounds : tuple[float, float] | None
        A tuple of two floats indicating the lower and upper bounds of the prior.

    Raises
    ------
    ValueError
        If ``term_type`` is not within the values listed above.

    Returns
    -------
    prior: Prior
        The instance of Prior according to the ``term_type``.
    """
    if term_type == "common":
        prior = generate_prior("Normal", bounds=None)
    elif term_type == "common_intercept":
        if bounds is not None:
            if any(np.isinf(b) for b in bounds):
                # TODO: Make it more specific.
                prior = generate_prior("Normal", bounds=bounds)
            else:
                prior = generate_prior(
                    "Normal", mu=mean(bounds), sigma=0.25, bounds=bounds
                )
        else:
            prior = generate_prior("Normal")
    elif term_type == "group_intercept":
        prior = generate_prior("Normal", mu="Normal", sigma="Weibull")
    elif term_type == "group_specific":
        prior = generate_prior("Normal", mu="Normal", sigma="Weibull")
    elif term_type == "group_intercept_with_common":
        prior = generate_prior("Normal", mu=0.0, sigma="Weibull")
    else:
        raise ValueError("Unrecognized term type.")
    return prior


def get_hddm_default_prior(
    term_type: str, param: str, bounds: tuple[float, float] | None
):
    """Generate a Prior based on the default settings - the HDDM case."""
    if term_type == "common":
        prior = generate_prior("Normal", bounds=None)
    elif term_type == "common_intercept":
        prior = generate_prior(HDDM_MU[param], bounds=bounds)
    elif term_type == "group_intercept":
        prior = generate_prior(HDDM_SETTINGS_GROUP[param], bounds=None)
    elif term_type == "group_specific":
        prior = generate_prior("Normal", mu="Normal", sigma="Weibull", bounds=None)
    else:
        raise ValueError("Unrecognized term type.")
    return prior


HSSM_SETTINGS_DISTRIBUTIONS: dict[Any, Any] = {
    "Normal": {"mu": 0.0, "sigma": 0.25},
    "Weibull": {"alpha": 1.5, "beta": 0.3},
    "HalfNormal": {"sigma": 0.25},
    "Beta": {"alpha": 1.0, "beta": 1.0},
    "Gamma": {"mu": 1.0, "sigma": 1.0},
}

HDDM_MU: dict[Any, Any] = {
    "v": {"dist": "Normal", "mu": 2.0, "sigma": 3.0},
    "a": {"dist": "Gamma", "mu": 1.5, "sigma": 0.75},
    "z": {"dist": "Gamma", "mu": 10, "sigma": 10},
    "t": {"dist": "Gamma", "mu": 0.4, "sigma": 0.2},
    "sv": {"dist": "HalfNormal", "sigma": 2.0},
    "st": {"dist": "HalfNormal", "sigma": 0.3},
    "sz": {"dist": "HalfNormal", "sigma": 0.5},
}

HDDM_SIGMA: dict[Any, Any] = {
    "v": {"dist": "HalfNormal", "sigma": 2.0},
    "a": {"dist": "HalfNormal", "sigma": 0.1},
    "z": {"dist": "Gamma", "mu": 10, "sigma": 10},
    "t": {"dist": "HalfNormal", "sigma": 1.0},
    "sv": {"dist": "Weibull", "alpha": 1.5, "beta": "0.3"},
    "sz": {"dist": "Weibull", "alpha": 1.5, "beta": "0.3"},
    "st": {"dist": "Weibull", "alpha": 1.5, "beta": "0.3"},
}

HDDM_SETTINGS_GROUP: dict[Any, Any] = {
    "v": {"dist": "Normal", "mu": HDDM_MU["v"], "sigma": HDDM_SIGMA["v"]},
    "a": {"dist": "Gamma", "mu": HDDM_MU["a"], "sigma": HDDM_SIGMA["a"]},
    "z": {"dist": "Beta", "alpha": HDDM_MU["z"], "beta": HDDM_SIGMA["z"]},
    "t": {"dist": "Normal", "mu": HDDM_MU["t"], "sigma": HDDM_SIGMA["t"]},
    "sv": {"dist": "Gamma", "mu": HDDM_MU["sv"], "sigma": HDDM_SIGMA["sv"]},
    "sz": {"dist": "Gamma", "mu": HDDM_MU["sz"], "sigma": HDDM_SIGMA["sz"]},
    "st": {"dist": "Gamma", "mu": HDDM_MU["st"], "sigma": HDDM_SIGMA["st"]},
}
