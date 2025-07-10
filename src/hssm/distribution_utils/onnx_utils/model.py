"""Utilities for working with models in ONNX format."""

from os import PathLike
from pathlib import Path

import onnx
from huggingface_hub import hf_hub_download

REPO_ID = "franklab/HSSM"


def download_hf(path: str):
    """
    Download a file from a HuggingFace repository.

    Parameters
    ----------
    path : str
        The path of the file to download in the repository.

    Returns
    -------
    str
        The local path where the file is downloaded.

    Notes
    -----
    The repository is specified by the REPO_ID constant,
    which should be a valid HuggingFace.co repository ID.
    The file is downloaded using the HuggingFace Hub's
     hf_hub_download function.
    """
    return hf_hub_download(repo_id=REPO_ID, filename=path)


def load_onnx_model(
    model: str | PathLike | onnx.ModelProto,
) -> onnx.ModelProto:
    """Load an ONNX model from a local file or from HuggingFace.

    Parameters
    ----------
    model : str | PathLike | onnx.ModelProto
        The model can be a path to a local ONNX file, a HuggingFace file,
        or a loaded ONNX ModelProto object.

    Returns
    -------
    onnx.ModelProto
        The loaded ONNX model.
    """
    if isinstance(model, onnx.ModelProto):
        return model

    if not isinstance(model, (str, PathLike)):
        raise ValueError(
            f"The model must be a path to a local ONNX file, a HuggingFace file, "
            f"or an ONNX ModelProto object, but got {model}."
        )

    model_path = Path(model)
    if model_path.exists():
        return onnx.load(model_path)

    if isinstance(model, PathLike):
        raise FileNotFoundError(
            f"The specified ONNX model file does not exist: {model}"
        )

    if isinstance(model, str):
        return onnx.load(hf_hub_download(repo_id=REPO_ID, filename=model))

    raise ValueError(
        f"The model must be a path to a local ONNX file, a HuggingFace file, "
        f"or an ONNX ModelProto object, but got {model}."
    )
