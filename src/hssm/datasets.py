"""
Base IO code for datasets.

Heavily influenced by Arviz's(scikit-learn's, and Bambi's) implementation.
"""

import os
from typing import NamedTuple, Optional, Union

import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class FileMetadata(NamedTuple):
    """Typing for dataset metadata."""

    filename: str
    path: str
    description: str


DATASETS = {
    "cavanagh_theta": FileMetadata(
        filename="cavanagh_theta",
        path=os.path.join(base_dir, "hssm/datasets/cavanagh_theta_nn.csv"),
        description="Description for cavanagh_theta dataset",
    ),
    "cavanagh_theta_old": FileMetadata(
        filename="cavanagh_theta",
        path=os.path.join(base_dir, "hssm/datasets/cavanagh_theta_nn_old.csv"),
        description="Description for the original cavanagh_theta dataset",
    ),
}


def load_data(dataset: Optional[str] = None) -> Union[pd.DataFrame, str]:
    """
    Load a dataset as a pandas DataFrame.

    If a valid dataset name is provided, this function will return the
    corresponding DataFrame. Otherwise, it lists the available datasets.

    Parameters
    ----------
    dataset : str, optional
        Name of the dataset to load. If not provided, a list
        of available datasets is returned.

    Raises
    ------
    ValueError
        If the provided dataset name does not match any of the available datasets.

    Returns
    -------
    pd.DataFrame or str
        Loaded dataset as a DataFrame if a valid dataset name was provided,
        otherwise a string listing the available datasets.
    """
    if dataset in DATASETS:
        datafile = DATASETS[dataset]
        file_path = datafile.path

        if not os.path.exists(file_path):
            raise ValueError(f"File {file_path} does not exist.")

        return pd.read_csv(file_path)

    if dataset is None:
        return _list_datasets()

    raise ValueError(
        f"Dataset {dataset} not found! The following are available:\n{_list_datasets()}"
    )


def _list_datasets() -> str:
    """
    Create a string listing all the available datasets.

    The string includes the datasets' names, their paths and descriptions.

    Returns
    -------
    str
        String listing all the available datasets.
    """
    lines = []
    for filename, resource in DATASETS.items():
        file_path = resource.path
        location = (
            "location: file does not exist"
            if not os.path.exists(file_path)
            else f"location: {file_path}"
        )
        lines.append(
            f"{filename}\n{'=' * len(filename)}\n{resource.description}\n{location}"
        )

    return f"\n\n{10 * '-'}\n\n".join(lines)
