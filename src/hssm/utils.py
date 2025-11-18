"""Utility functions and classes.

HSSM has to reconcile with two representations: it's own representation as an HSSM and
the representation acceptable to Bambi. The two are not equivalent. This file contains
the Param class that reconcile these differences.

The Param class is an abstraction that stores the parameter specifications and turns
these representations in Bambi-compatible formats through convenience function
_parse_bambi().
"""

import contextlib
import itertools
import logging
import os
from copy import deepcopy
from typing import Any, Literal, cast

import arviz as az
import bambi as bmb
import jax
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr
from bambi.terms import CommonTerm, GroupSpecificTerm, HSGPTerm, OffsetTerm
from bambi.utils import get_aliased_name, response_evaluate_new_data
from tqdm import tqdm

from .param.param import Param

_logger = logging.getLogger("hssm")


def make_alias_dict_from_parent(parent: Param) -> dict[str, str]:
    """Make aliases from the parent parameter.

    From a Param object that represents a parent parameter in Bambi,
    returns a dict that represents how Bambi should alias its parameters to
    make it more HSSM-friendly.

    Parameters
    ----------
    parent: A Param object that represents a parent parameter.

    Returns
    -------
        A dict that indicates how Bambi should alias its parameters.
    """
    assert parent.is_parent, "This Param object should be a parent!"
    assert parent.name is not None

    result_dict = {"c(rt, response)": "rt,response"}

    # The easy case. We will just alias "Intercept" as the actual name of the
    # parameter
    if not parent.is_regression:
        result_dict |= {"Intercept": parent.name}

        return result_dict

    # The regression case:
    # In this case, the name of the response variable should actually be
    # the name of the parent parameter
    result_dict["c(rt, response)"] = parent.name

    return result_dict


def _get_alias_dict(
    model: bmb.Model, parent: Param, response_c: str, response_str: str
) -> dict[str, str | dict]:
    """Make a list of aliases.

    Iterates through a list of Param objects, and aliases a Bambi model's parameters
    to make it more HSSM-friendly.

    Parameters
    ----------
    model
        A Bambi model.
    parent
        The Param representation of the parent parameter.
    response_c
        The name of the response parameters in the c() format.
    response_str
        The name of the response parameters in the comma-separated format.

    Returns
    -------
    dict[str, str | dict]
        A dict that indicates how Bambi should alias its parameters.
    """
    parent_name = cast("str", parent.name)
    alias_dict: dict[str, Any] = {response_c: response_str}

    if len(model.distributional_components) == 1:
        if not parent.is_regression or (
            parent.is_regression and parent.formula is None
        ):
            alias_dict[parent_name] = f"{parent_name}_mean"
            alias_dict["Intercept"] = parent_name
        else:
            for name, term in model.components[parent_name].terms.items():
                if isinstance(
                    term, (CommonTerm, GroupSpecificTerm, HSGPTerm, OffsetTerm)
                ):
                    alias_dict[name] = f"{parent_name}_{name}"

        return alias_dict

    for component_name, component in model.distributional_components.items():
        if component_name == parent_name:
            alias_dict[component_name] = {}
            if not parent.is_regression:
                # Most likely this branch will not be reached
                alias_dict[component_name]["Intercept"] = f"{parent_name}_Intercept"
            else:
                for name, term in component.terms.items():
                    if isinstance(
                        term, (CommonTerm, GroupSpecificTerm, HSGPTerm, OffsetTerm)
                    ):
                        alias_dict[component_name] |= {name: f"{parent_name}_{name}"}
            break

    return alias_dict


def _compute_log_likelihood(
    model: bmb.Model,
    idata: az.InferenceData,
    data: pd.DataFrame | None,
    inplace: bool = True,
) -> az.InferenceData | None:
    """Compute the model's log-likelihood.

    Parameters
    ----------
    idata : InferenceData
        The `InferenceData` instance returned by `.fit()`.
    data : pandas.DataFrame or None
        An optional data frame with values for the predictors and the response on which
        the model's log-likelihood function is evaluated.
        If omitted, the original dataset is used.
    inplace : bool
        If True` it will modify `idata` in-place. Otherwise, it will return a copy of
        `idata` with the `log_likelihood` group added.

    Returns
    -------
    InferenceData or None
    """
    # These are not formal parameters because it does not make sense to...
    #   1. compute the log-likelihood omitting
    #      the group-specific components of the model.
    #   2. compute the log-likelihood on unseen groups.
    include_group_specific = True
    sample_new_groups = False

    # Get the aliased response name
    response_aliased_name = get_aliased_name(model.response_component.term)

    if not inplace:
        idata = deepcopy(idata)

    # # Populate the posterior in the InferenceData object
    # with the likelihood parameters
    idata = model._compute_likelihood_params(  # pylint: disable=protected-access
        idata, data, include_group_specific, sample_new_groups
    )

    required_kwargs = {"model": model, "posterior": idata["posterior"], "data": data}
    log_likelihood_out = log_likelihood(model.family, **required_kwargs).to_dataset(
        name=response_aliased_name
    )

    # Drop the existing log_likelihood group if it exists
    if "log_likelihood" in idata:
        _logger.info("Replacing existing log_likelihood group in idata.")
        del idata["log_likelihood"]

    # Assign the log-likelihood group to the InferenceData object
    idata.add_groups({"log_likelihood": log_likelihood_out})
    setattr(
        idata,
        "log_likelihood",
        idata["log_likelihood"].assign_attrs(
            modeling_interface="bambi", modeling_interface_version=bmb.__version__
        ),
    )
    return idata


def log_likelihood(
    family: bmb.Family,
    model: bmb.Model,
    posterior: xr.DataArray,
    data: pd.DataFrame | None = None,
    **kwargs,
) -> xr.DataArray:
    """Evaluate the model log-likelihood.

    This is a variation on the `bambi.utils.log_likelihood` function that
    loops over the chains and draws to evaluate the log-likelihood for each
    instead of attempting to batch the computation as is done in the orignal.

    Parameters
    ----------
    model : bambi.Model
        The model
    posterior : xr.Dataset
        The xarray dataset that contains the draws for
        all the parameters in the posterior.
        It must contain the parameters that are needed
        in the distribution of the response, or
        the parameters that allow to derive them.
    kwargs :
        Parameters that are used to get draws but do
        not appear in the posterior object or
        other configuration parameters.
        For instance, the 'n' in binomial models and multinomial models.

    Returns
    -------
    xr.DataArray
        A data array with the value of the log-likelihood
        for each chain, draw, and value of the response variable.
    """
    # Child classes pass "y_values" through the "y" kwarg
    y_values = kwargs.pop("y", None)

    # Get the values of the outcome variable
    if y_values is None:  # when it's not handled by the specific family
        if data is None:
            y_values = np.squeeze(model.response_component.term.data)
        else:
            y_values = response_evaluate_new_data(model, data)

    response_dist = get_response_dist(model.family)
    response_term = model.response_component.term
    kwargs, coords = family._make_dist_kwargs_and_coords(model, posterior, **kwargs)

    # If it's multivariate, it's going to have a fourth coord,
    # but we actually don't need it. We just need "chain", "draw", "__obs__"
    coords = dict(list(coords.items())[:3])

    n_chains = len(coords["chain"])
    n_draws = len(coords["draw"])
    output_array = np.zeros((n_chains, n_draws, len(y_values)))
    kwargs_prep = {key_: val[0][0] for key_, val in kwargs.items()}
    shape_dict = {key_: val.shape for key_, val in kwargs_prep.items()}
    pt_dict = {
        key_: (pt.vector(key_, shape=((1,) if val[0] == 1 else (None,))))
        for key_, val in shape_dict.items()
    }

    # Compile likelihood function
    if not response_term.is_constrained:
        rv_logp = pm.logp(response_dist.dist(**pt_dict), y_values)
        logp_compiled = pm.compile(
            [val for key_, val in pt_dict.items()],
            rv_logp,
            allow_input_downcast=True,
        )
    else:
        # Bounds are scalars, we can safely pick them from the first row
        lower, upper = response_term.data[0, 1:]
        lower = lower if lower != -np.inf else None
        upper = upper if upper != np.inf else None

        # Finally evaluate logp
        rv_logp = pm.logp(
            pm.Truncated.dist(
                response_dist.dist(**kwargs_prep), lower=lower, upper=upper
            ),
            y_values,
        )
        logp_compiled = pm.compile(
            [val for key_, val in pt_dict.items()], rv_logp, allow_input_downcast=True
        )

    # Loop through chain and draws
    for ids in tqdm(
        list(itertools.product(coords["chain"].values, coords["draw"].values))
    ):
        kwargs_tmp = {
            key_: (
                val[ids[0], ids[1], ...]
                if (val.shape[0] == n_chains and val.shape[1] == n_draws)
                else val[0, 0, ...]
            )
            for key_, val in kwargs.items()
        }

        output_array[ids[0], ids[1], :] = logp_compiled(**kwargs_tmp)

    # output_array
    return xr.DataArray(output_array, coords=coords)


def get_response_dist(family: bmb.Family) -> pm.Distribution:
    """Get the PyMC distribution for the response.

    Parameters
    ----------
    family : bambi.Family
        The family for which the response distribution is wanted

    Returns
    -------
    pm.Distribution
        The response distribution
    """
    mapping = {"Cumulative": pm.Categorical, "StoppingRatio": pm.Categorical}

    if family.likelihood.dist:
        dist = family.likelihood.dist
    elif family.likelihood.name in mapping:
        dist = mapping[family.likelihood.name]
    else:
        dist = getattr(pm, family.likelihood.name)
    return dist


def set_floatX(dtype: Literal["float32", "float64"], update_jax: bool = True):
    """Set float types for pytensor and Jax.

    Often we wish to work with a specific type of float in both PyTensor and JAX.
    This function helps set float types in both packages.

    Parameters
    ----------
    dtype
        Either `float32` or `float64`. Float type for pytensor (and jax if `jax=True`).
    update_jax : optional
        Whether this function also sets float type for JAX by changing the
        `jax_enable_x64` setting in JAX config. Defaults to True.
    """
    if dtype not in ["float32", "float64"]:
        raise ValueError('`dtype` must be either "float32" or "float64".')

    pytensor.config.floatX = dtype
    _logger.info("Setting PyTensor floatX type to %s.", dtype)

    if update_jax:
        jax_enable_x64 = dtype == "float64"
        jax.config.update("jax_enable_x64", jax_enable_x64)

        _logger.info(
            'Setting "jax_enable_x64" to %s. '
            + "If this is not intended, please set `jax` to False.",
            jax_enable_x64,
        )


def _print_prior(term: CommonTerm | GroupSpecificTerm) -> str:
    """Make the output string of a term.

    If prior is a float, print x: prior. Otherwise, print x ~ prior.

    Parameters
    ----------
    term
        A BaseTerm in Bambi

    Returns
    -------
        A string representing the term_name ~ prior pair
    """
    term_name = term.alias or term.name
    prior = term._prior

    if isinstance(prior, float):
        return f"        {term_name}: {prior}"

    return f"        {term_name} ~ {prior}"


def _generate_random_indices(
    n_samples: int | float | None, n_draws: int
) -> np.ndarray | None:
    """Generate random indices for sampling an InferenceData object.

    Parameters
    ----------
    n_samples
        When an interger >= 1, the number of samples to be extracted from the draw
        dimension. If this integer is larger than n_draws, returns None, which means
        all samples are extracted. When a float between 0 and 1, the proportion of
        samples to be extracted from the draw dimension. If this proportion is very
        small, at least one sample will be drawn. When None, returns None.
    n_draws
        The number of total draws in the InferenceData object.

    Returns
    -------
    np.ndarray
        A 2D array of shape (n_chains, n_draws) with random indices or None, which means
        using the entire dataset without random sampling.
    """
    if n_draws <= 0:
        raise ValueError("n_draws must be >= 1.")

    if n_samples is None:
        return None

    if n_samples > n_draws:
        _logger.warning("n_samples > n_draws. Using the entire dataset.")
        return None

    if isinstance(n_samples, float):
        if n_samples <= 0 or n_samples > 1:
            raise ValueError("When a float, n_samples must be between 0 and 1.")
        n_samples = max(int(n_samples * n_draws), 1)

    if n_samples < 1:
        raise ValueError("When an int, n_samples must be >= 1.")

    sampling_indices = np.random.choice(n_draws, size=n_samples, replace=False)

    return sampling_indices


def _random_sample(
    data: xr.DataArray | xr.Dataset, n_samples: int | float | None
) -> xr.DataArray | xr.Dataset:
    """Randomly sample a DataArray or Dataset.

    Parameters
    ----------
    data
        A DataArray or Dataset to be sampled.
    n_samples
        When an interger >= 1, the number of samples to be extracted from the draw
        dimension. If this integer is larger than n_draws, returns None, which means
        all samples are extracted. When a float between 0 and 1, the proportion of
        samples to be extracted from the draw dimension. If this proportion is very
        small, at least one sample will be drawn. When None, returns None.

    Returns
    -------
    xr.DataArray | xr.Dataset
        The sampled InferenceData object.
    """
    n_draws = data.draw.size
    sampling_indices = _generate_random_indices(n_samples, n_draws)

    if sampling_indices is None:
        return data
    return data.isel(draw=sampling_indices)


def _rearrange_data(data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    """Rearrange a dataframe so that missing values are on top.

    We assume the dataframe's first column can contain missing values coded as -999.0.

    Parameters
    ----------
    df
        The dataframe or numpy array to be rearranged.

    Returns
    -------
    pd.DataFrame | np.ndarray
        The rearranged dataframe.
    """
    if isinstance(data, pd.DataFrame):
        missing_indices = data.iloc[:, 0] == -999.0
        split_missing = data.loc[missing_indices, :]
        split_not_missing = data.loc[~missing_indices, :]

        return pd.concat([split_missing, split_not_missing])

    missing_indices = data[:, 0] == -999.0
    split_missing = data[missing_indices, :]
    split_not_missing = data[~missing_indices, :]

    return np.concatenate([split_missing, split_not_missing])


def _split_array(data: np.ndarray | list[int], divisor: int) -> list[np.ndarray]:
    num_splits = len(data) // divisor + (1 if len(data) % divisor != 0 else 0)
    return [tmp.astype(int) for tmp in np.array_split(data, num_splits)]


def check_data_for_rl(
    data: pd.DataFrame,
    participant_id_col: str = "participant_id",
    trial_id_col: str = "trial_id",
) -> tuple[pd.DataFrame, int, int]:
    """Check if the data is suitable for Reinforcement Learning (RL) models.

    Parameters
    ----------
    data : pd.DataFrame
        The data to check.
    participant_id_col : str
        The name of the column containing participant IDs.
    trial_id_col : str
        The name of the column containing trial IDs.

    Returns
    -------
    tuple[pd.DataFrame, int, int]
        A tuple containing the cleaned data, number of participants,
        and number of trials.
    """
    if participant_id_col not in data.columns:
        raise ValueError(f"Column '{participant_id_col}' not found in data.")
    if trial_id_col not in data.columns:
        raise ValueError(f"Column '{trial_id_col}' not found in data.")

    sorted_data = data.sort_values(
        by=[participant_id_col, trial_id_col], ignore_index=True
    )

    n_participants = data[participant_id_col].nunique()
    trials_by_participant = sorted_data.groupby(participant_id_col)[trial_id_col].size()

    if not np.all(trials_by_participant == trials_by_participant.iloc[0]):
        raise ValueError("All participants must have the same number of trials.")

    n_trials = trials_by_participant.iloc[0]

    return sorted_data, n_participants, n_trials


def predictive_idata_to_dataframe(
    idata: az.InferenceData,
    predictive_group: Literal[
        "posterior_predictive", "prior_predictive"
    ] = "posterior_predictive",
    response_str: str = "rt,response",
    response_dim: str = "rt,response_dim",
) -> pd.DataFrame:
    """Convert a predictive InferenceData object to a dataframe.

    Parameters
    ----------
    idata : az.InferenceData
        The InferenceData object to convert.
    predictive_group : Literal["posterior_predictive", "prior_predictive"]
        The predictive group to convert.

    Returns
    -------
    pd.DataFrame:
        A dataframe with the predictive samples.
    """
    df = idata[predictive_group].to_dataframe().reset_index(drop=False)
    df_wide = df.pivot_table(
        index=["chain", "draw", "__obs__"], columns=response_dim, values=response_str
    ).reset_index()

    df_wide.columns.name = None
    df_wide = df_wide.rename(columns={0: "rt", 1: "response"})
    return df_wide


class SuppressOutput:
    """Context manager for suppressing output.

    This context manager redirects both stdout and stderr to `os.devnull`,
    effectively silencing all output during the execution of the block.
    It also disables logging by setting the logging level to `CRITICAL`.

    Examples
    --------
    >>> with SuppressOutput():
    ...     grad_func = pytensor.function(
    ...         [v, a, z, t],
    ...         grad,
    ...         mode=nan_guard_mode,
    ...     )

    Methods
    -------
    __enter__()
        Redirects stdout and stderr, and disables logging.

    __exit__(exc_type, exc_value, traceback)
        Restores stdout, stderr, and logging upon exit.
    """

    def __enter__(self):  # noqa: D105
        self._null_file = open(os.devnull, "w")
        self._stdout_context = contextlib.redirect_stdout(self._null_file)
        self._stderr_context = contextlib.redirect_stderr(self._null_file)
        self._stdout_context.__enter__()
        self._stderr_context.__enter__()
        logging.disable(logging.CRITICAL)  # Disable logging

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: D105
        self._stdout_context.__exit__(exc_type, exc_value, traceback)
        self._stderr_context.__exit__(exc_type, exc_value, traceback)
        self._null_file.close()
        logging.disable(logging.NOTSET)  # Re-enable logging
