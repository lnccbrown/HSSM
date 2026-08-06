"""
Base IO code for datasets.

Heavily influenced by Arviz's(scikit-learn's, and Bambi's) implementation.
"""

import os
from typing import NamedTuple

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


def load_data(dataset: str) -> pd.DataFrame:
    """
    Load a built-in dataset as a pandas DataFrame.

    Use `hssm.list_data` to see the names of the available datasets.

    Parameters
    ----------
    dataset : str
        Name of the dataset to load.

    Raises
    ------
    ValueError
        If the provided dataset name does not match any of the available datasets.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    if dataset not in DATASETS:
        raise ValueError(
            f"Dataset {dataset} not found! The following are available:\n"
            f"{_list_datasets()}"
        )

    file_path = DATASETS[dataset].path

    if not os.path.exists(file_path):
        raise ValueError(f"File {file_path} does not exist.")

    return pd.read_csv(file_path)


def list_data() -> tuple[str, ...]:
    """Return the names of the built-in HSSM datasets.

    Use `hssm.load_data` to load any of them by name.

    Returns
    -------
    tuple[str, ...]
        A tuple containing all built-in HSSM dataset names.
    """
    return tuple(DATASETS)


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
