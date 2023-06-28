"""Base IO code for datasets. Heavily influenced by Arviz's (scikit-learn's, and Bambi's) implementation."""

import os
import pandas as pd
from collections import namedtuple
from typing import Optional, Union

# Define your base directory
base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Tuple to store metadata for each file
FileMetadata = namedtuple("FileMetadata", ["filename", "path", "description"])

# Dictionary of datasets
DATASETS = {
    "cavanagh_theta": FileMetadata(
        filename="cavanagh_theta",
        path=os.path.join(base_dir, "hssm/datasets/cavanagh_theta_nn.csv"),
        description="Description for cavanagh_theta dataset",
    )
}


def load_data(dataset: Optional[str] = None) -> Union[pd.DataFrame, str]:
    """
    Loads a dataset as a pandas DataFrame if a valid dataset name is provided,
    otherwise lists the available datasets.

    Parameters:
    dataset (str, optional): Name of the dataset to load. If not provided, a list
                             of available datasets is returned.

    Raises:
    ValueError: If the provided dataset name does not match any of the available datasets.

    Returns:
    pd.DataFrame/str: Loaded dataset as a DataFrame if a valid dataset name was provided,
                      otherwise a string listing the available datasets.
    """
    if dataset in DATASETS:
        datafile = DATASETS[dataset]
        file_path = datafile.path

        if not os.path.exists(file_path):
            raise ValueError(f"File {file_path} does not exist.")

        return pd.read_csv(file_path)
    else:
        if dataset is None:
            return _list_datasets()
        else:
            raise ValueError(
                f"Dataset {dataset} not found! "
                f"The following are available:\n{_list_datasets()}"
            )


def _list_datasets() -> str:
    """
    Creates a string listing all the available datasets, their paths and descriptions.

    Returns:
    str: String listing all the available datasets.
    """
    lines = []
    for filename, resource in DATASETS.items():
        file_path = resource.path
        if not os.path.exists(file_path):
            location = f"location: file does not exist"
        else:
            location = f"location: {file_path}"
        lines.append(
            f"{filename}\n{'=' * len(filename)}\n{resource.description}\n{location}"
        )

    return f"\n\n{10 * '-'}\n\n".join(lines)
