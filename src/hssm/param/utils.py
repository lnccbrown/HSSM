"""Utility functions for the parameter classes."""

from typing import Any, Literal, TypedDict, Union, cast

import bambi as bmb
import numpy as np

from ..prior import deserialize_prior_obj, serialize_prior_obj


class SerializedPrior(TypedDict):
    """A dictionary that represents a serialized prior."""

    type: Literal["constant", "array", "prior", "simple", "regression"]
    value: Union[
        int, float, str, dict[str, Any], dict[str, "SerializedPrior"], "SerializedPrior"
    ]


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


def _make_prior_dict(
    prior: dict[str, float | dict[dict, Any] | bmb.Prior],
) -> dict[str, float | bmb.Prior]:
    """Make bambi priors from a ``dict`` of priors for the regression case.

    Parameters
    ----------
    prior
        A dictionary where each key is the name of a parameter in a regression
        and each value is the prior specification for that parameter.

    Returns
    -------
    dict[str, float | bmb.Prior]
        A dictionary where each key is the name of a parameter in a regression and each
        value is either a float or a bmb.Prior object.
    """
    priors = {
        # Convert dict to bmb.Prior if a dict is passed
        param: _make_priors_recursive(cast(dict[str, Any], prior))
        if isinstance(prior, dict)
        else prior
        for param, prior in prior.items()
    }

    return priors


def _make_priors_recursive(
    prior: dict[str, float | dict[str, Any] | bmb.Prior],
) -> bmb.Prior:
    """Make `bmb.Prior` objects from ``dict``s.

    Helper function that recursively converts a dict that might have some fields that
    have a parameter definitions as dicts to bmb.Prior objects.

    Parameters
    ----------
    prior
        A dictionary that contains parameter specifications.

    Returns
    -------
    bmb.Prior
        A bmb.Prior object with fields that can be converted to bmb.Prior objects also
        converted.
    """
    for k, v in prior.items():
        if isinstance(v, dict) and "name" in v:
            prior[k] = _make_priors_recursive(v)

    return bmb.Prior(**prior)


def serialize_prior(
    prior: float | np.ndarray | dict[str, Any] | bmb.Prior,
) -> SerializedPrior:
    """Serialize a dictionary of priors to a dictionary.

    Parameters
    ----------
    prior
        A dictionary of priors.

    Returns
    -------
    dict
        A dictionary of serialized priors.
    """
    if isinstance(prior, (int, float)):
        return {"type": "constant", "value": prior}
    if isinstance(prior, np.ndarray):
        return {"type": "array", "value": array_to_str(prior)}
    if isinstance(prior, bmb.Prior):
        return {"type": "prior", "value": serialize_prior_obj(prior)}
    if "name" in prior:
        return {"type": "simple", "value": prior}

    regression_prior: dict[str, SerializedPrior] = {}
    for key, value in prior.items():
        regression_prior[key] = serialize_prior(value)

    return {"type": "regression", "value": regression_prior}


def deserialize_prior(
    serialized_prior: SerializedPrior,
) -> float | np.ndarray | dict[str, Any] | bmb.Prior:
    """Deserialize a serialized prior.

    Parameters
    ----------
    serialized_prior
        A dictionary of serialized priors.

    Returns
    -------
    float | np.ndarray | dict[str, Any] | bmb.Prior
        The deserialized prior.
    """
    prior_type = serialized_prior["type"]
    prior_value = serialized_prior["value"]

    match prior_type:
        case "constant":
            return prior_value
        case "array":
            prior_value = cast(str, prior_value)
            return str_to_array(prior_value)
        case "prior":
            prior_value = cast(dict[str, Any], prior_value)
            return deserialize_prior_obj(prior_value)
        case "simple":
            return prior_value
        case "regression":
            prior_value = cast(dict[str, SerializedPrior], prior_value)
            regression_prior: dict[str, Any] = {}
            for key, value in prior_value.items():
                regression_prior[key] = deserialize_prior(value)

    return regression_prior


def array_to_str(array: np.ndarray) -> str:
    """Convert a numpy array to a string.

    Parameters
    ----------
    array
        The numpy array to convert.

    Returns
    -------
    str
        The string representation of the numpy array.
    """
    # generate an array with strings
    array_str = np.char.mod("%f", array)
    # combine to a string
    return ",".join(array_str)


def str_to_array(array_str: str) -> np.ndarray:
    """Convert a string to a numpy array.

    Parameters
    ----------
    array_str
        The string representation of the numpy array.

    Returns
    -------
    np.ndarray
        The numpy array.
    """
    if array_str == "":
        return np.array([])
    return np.array(array_str.split(",")).astype(float)
