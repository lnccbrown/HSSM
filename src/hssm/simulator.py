"""Simulates data with basic ssm_simulators."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from ssms.basic_simulators.simulator import simulator
from ssms.config import model_config


def simulate_data(
    model: str,
    theta: dict[str, ArrayLike] | list[float] | ArrayLike,
    size: int,
    random_state: int | None = None,
    output_df: bool = True,
    **kwargs,
) -> np.ndarray | pd.DataFrame:
    """Sample simulated data from specified distributions.

    Parameters
    ----------
    model
        A model name that must be supported in `ssm_simulators`. For a detailed list of
        supported models, please see all fields in the `model_config` dict
        [here](https://github.com/AlexanderFengler/ssm-simulators/blob
        /e09eb2528d885c7b3340516597849fff4d9a5bf8/ssms/config/config.py#L6)
    theta
        Parameters of the process. Can be supplied as dictionary with parameter names as
        key and np.array or float as values. Can also be supplied as a list or 1D-array,
        however in this case the order of parameters is important and must match
        specifications [here](https://github.com/AlexanderFengler/
        ssm-simulators/blob/e09eb2528d885c7b3340516597849fff4d9a5bf8/ssms/config/config.py#L6).
        Parameters can be specificed 'trial-wise', by supplying 1D arrays of shape
        `size` to the dictionary, or by supplying a 2D array
        of shape `(size, n_parameters)` dicrectly.
    size
        The size of the data to be simulated. If `theta` is a 2D ArrayLike, this
        parameter indicates the size of data to be simulated for each trial.
    random_state : optional
        A random seed for reproducibility.
    output_df : optional
        If True, outputs a DataFrame with column names "rt", "response". Otherwise a
        2-column numpy array, by default True.
    kwargs : optional
        Other arguments passed to ssms.basic_simulators.simulator.

    Returns
    -------
    np.ndarray | pd.DataFrame
        An array or DataFrame with simulated data.
    """
    if model not in model_config:
        raise ValueError(f"model must be one of {list(model_config.keys())}.")

    sims = simulator(
        theta,
        model=model,
        n_samples=size,
        random_state=random_state,
        **kwargs,
    )

    rts = np.squeeze(sims["rts"]).reshape(-1)
    responses = np.squeeze(sims["choices"]).reshape(-1)

    sims_array = np.column_stack([rts, responses])

    if output_df:
        return pd.DataFrame(sims_array, columns=["rt", "response"])

    return sims_array
