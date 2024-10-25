"""Model loading and saving functions."""

import json
import logging
import pickle
from io import BytesIO
from os import PathLike
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Literal
from zipfile import ZipFile

import arviz as az
import pandas as pd

from hssm.config import ModelConfig
from hssm.param.user_param import UserParam
from hssm.param.utils import deserialize_prior, serialize_prior

if TYPE_CHECKING:
    from . import HSSM

_logger = logging.getLogger("hssm")

SERIALIZABLE_ATTRS = [
    "model_name",
    "choices",
    "loglik_kind",
    "global_formula",
    "link_settings",
    "prior_settings",
    "missing_data",
    "deadline",
]


def _save_model(
    model: "HSSM",
    path: str | PathLike,
    save_data: bool = True,
    save_data_format: Literal["csv", "parquet"] = "csv",
) -> None:
    """Save the model to a file.

    Parameters
    ----------
    model
        The model to save.
    path
        The path to save the model to.
    save_data
        Whether to save the data along with model specifications. Set to false if you
        do not want the data to be saved. Defaults to True.
    save_data_format
        The format to save the data in. Either "csv" or "parquet". The "csv" format
        provides more compatibility, but the "parquet" format provides faster loading
        and smaller file size. When "parquet", `pyarrow` is required. Defaults to "csv".
    """
    if save_data:
        if save_data_format not in ["csv", "parquet"]:
            raise ValueError("save_data_format must be either 'csv' or 'parquet'.")

        if save_data_format == "parquet":
            _check_pyarrow()

    # First, save all the attributes that are serializable (str, int, float, etc.)
    attrs_dict = {
        attr: value
        for attr in SERIALIZABLE_ATTRS
        if (value := getattr(model, attr)) is not None
    }

    # Save all user-specified parameters
    attrs_dict["include"] = model.params.serialize_user_params()

    if model.has_lapse:
        attrs_dict["lapse"] = serialize_prior(model.lapse)

    # Save the user-specified attributes
    user_attrs = model.user_spec.copy()
    if (model_config := model.user_spec.get("model_config")) is not None:
        user_attrs["model_config"] = model_config.serialize()

    with ZipFile(path, "w") as zipf:
        if "loglik" in user_attrs and not isinstance(user_attrs["loglik"], str):
            _write_pickle(zipf, "loglik", user_attrs.pop("loglik"), check_none=False)

        if "loglik_missing_data" in user_attrs and not isinstance(
            user_attrs["loglik_missing_data"], str
        ):
            _write_pickle(
                zipf,
                "loglik_missing_data",
                user_attrs.pop("loglik_missing_data"),
                check_none=False,
            )

        attrs_dict |= user_attrs

        # Save the model into JSON
        zipf.writestr("model.json", json.dumps(attrs_dict))

        # Save data
        if save_data:
            dataIO = BytesIO()

            if save_data_format == "csv":
                model.data.to_csv(dataIO)
                zipf.writestr("data.csv", dataIO.getvalue())
            else:
                model.data.to_parquet(dataIO)
                zipf.writestr("data.parquet", dataIO.getvalue())

        # Save the traces to a netcdf file
        if model._inference_obj is not None:
            with NamedTemporaryFile(delete=True) as tmpfile:
                model._inference_obj.to_netcdf(tmpfile.name)
                zipf.write(tmpfile.name, "traces.nc")

        _write_pickle(zipf, "traces_vi", model._inference_obj_vi)
        _write_pickle(zipf, "vi_approx", model._vi_approx)
        _write_pickle(zipf, "map_dict", model._map_dict)

    _logger.info("Model saved.")


def _load_model(path: str | PathLike, data: pd.DataFrame | None = None) -> "HSSM":
    """Load a model from a .hssm file.

    Parameters
    ----------
    path
        The path to the saved model file.
    data
        The data to be used. If None, the data will be loaded from the saved model file.
        An error will be raised if the data is not found in the saved model file.
    """
    with ZipFile(path, "r") as zipf:
        if data is None:
            if "data.csv" in zipf.namelist():
                with zipf.open("data.csv") as f:
                    data = pd.read_csv(f, index_col=0)
            elif "data.parquet" in zipf.namelist():
                _check_pyarrow()
                with zipf.open("data.parquet") as f:
                    data = pd.read_parquet(f)
            else:
                raise ValueError(
                    "Data not found in the saved model file. "
                    "Please provide the data as an argument."
                )

        with zipf.open("model.json") as f:
            model_dict = json.load(f)

        model_dict["data"] = data
        # Because "model" is the keyword argument for HSSM, we change `model_name` to
        # `model` here.
        model_dict["model"] = model_dict.pop("model_name")

        # If "p_outlier" is in the dict saved in `include`, that means a p_outlier
        # parameter was included in the model. We need to deal with it separately.
        include_dict = model_dict.pop("include", None)

        if include_dict is not None:
            if "p_outlier" in include_dict:
                p_outlier = UserParam.deserialize(include_dict.pop("p_outlier"))
                model_dict["p_outlier"] = p_outlier.prior

        if "lapse" in model_dict:
            model_dict["lapse"] = deserialize_prior(model_dict["lapse"])

        if "model_config" in model_dict:
            model_config = model_dict.pop("model_config")
            model_config = ModelConfig.deserialize(model_config)

            # If the model has a p_outlier parameter, we need to remove it from the
            # list of parameters.
            if hasattr(model_config, "list_params"):
                list_params = model_config.list_params
                if "p_outlier" in list_params:
                    list_params.remove("p_outlier")
            model_dict["model_config"] = model_config

        model_dict["include"] = [
            UserParam.deserialize(param) for param in include_dict.values()
        ]

        model_dict["loglik"] = _load_pickle(zipf, "loglik")
        model_dict["loglik_missing_data"] = _load_pickle(zipf, "loglik_missing_data")

        kwargs = model_dict.pop("kwargs", {})
        model_dict.update(kwargs)

        # Importing in function to avoid circular imports
        from . import HSSM

        model = HSSM(**model_dict)

        # Load the traces if exist
        if "traces.nc" in zipf.namelist():
            with NamedTemporaryFile(delete=True) as tmpfile:
                with zipf.open("traces.nc") as f:
                    tmpfile.write(f.read())
                model._inference_obj = az.from_netcdf(tmpfile.name)

        # Load additional objects
        for attr in ["traces_vi", "vi_approx", "map_dict"]:
            setattr(model, f"_{attr}", _load_pickle(zipf, attr))

    return model


def _write_pickle(zipf: ZipFile, name: str, obj: Any, check_none: bool = True) -> None:
    """Write an object to a ZipFile.

    Parameters
    ----------
    zipf
        The open ZipFile object to write to.
    name
        The name of the object in the ZipFile.
    obj
        The object to pickle.
    check_none
        Whether to check if the object is None before pickling.
    """
    if check_none and obj is None:
        return

    zipf.writestr(name, pickle.dumps(obj))


def _load_pickle(zipf: ZipFile, obj_name: str) -> Any | None:
    """Load a picked object from a ZipFile.

    Parameters
    ----------
    zipf
        The open ZipFile object to read from.
    obj_name
        The name of the object in the ZipFile.

    Returns
    -------
        The unpickled object.
    """
    if obj_name in zipf.namelist():
        with zipf.open(obj_name, "r") as f:
            return pickle.load(f)

    return None


def _check_pyarrow():
    """Check if pyarrow is installed."""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise ImportError(
            "The pyarrow package is required to save models. "
            "Please install it using `pip install pyarrow`."
        )
