"""Utilities for working with models in ONNX format."""

from os import PathLike
from pathlib import Path

import onnx
from huggingface_hub import hf_hub_download

from hssm import _tracking_hooks as tracking_hooks

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
    local_path = hf_hub_download(
        repo_id=REPO_ID, filename=path, revision=tracking_hooks.pinned_revision(path)
    )
    tracking_hooks.record_network(path, local_path)
    return local_path


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

    # A pin means `load_run` is rebuilding a fit whose network came from the
    # hub; a same-named file in the cwd must not stand in for it.
    revision = tracking_hooks.pinned_revision(model) if isinstance(model, str) else None
    model_path = Path(model)
    if model_path.exists() and revision is None:
        return onnx.load(model_path)

    if isinstance(model, PathLike):
        raise FileNotFoundError(
            f"The specified ONNX model file does not exist: {model}"
        )

    if isinstance(model, str):
        local_path = hf_hub_download(repo_id=REPO_ID, filename=model, revision=revision)
        tracking_hooks.record_network(model, local_path)
        return onnx.load(local_path)

    raise ValueError(
        f"The model must be a path to a local ONNX file, a HuggingFace file, "
        f"or an ONNX ModelProto object, but got {model}."
    )
