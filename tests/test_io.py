import json
import pickle
from tempfile import NamedTemporaryFile, TemporaryDirectory
from zipfile import ZipFile

import arviz as az
import numpy as np
import pandas as pd
import pytest

import hssm
from hssm.io import (
    SERIALIZABLE_ATTRS,
    _load_model,
    _save_model,
)
from hssm.likelihoods.analytical import logp_ddm


def _check_seriaizable_attrs(obj, d):
    for attr in SERIALIZABLE_ATTRS:
        if attr in d:
            assert getattr(obj, attr) == d[attr]


def test__save_model(data_ddm, cav_idata):
    basic_model = hssm.HSSM(data=data_ddm)

    with NamedTemporaryFile() as tmp:
        _save_model(basic_model, tmp.name)

        with ZipFile(tmp.name, "r") as zip_ref:
            assert "model.json" in zip_ref.namelist()

            with zip_ref.open("model.json") as f:
                model_dict = json.load(f)
                _check_seriaizable_attrs(basic_model, model_dict)

                assert len(model_dict["include"]) == 1
                assert "p_outlier" in model_dict["include"]

            assert "loglik" not in zip_ref.namelist()
            assert "loglik_missing_data" not in zip_ref.namelist()
            assert "data.csv" in zip_ref.namelist()

            with zip_ref.open("data.csv") as f:
                data = pd.read_csv(f)
                assert np.all(pd.Series(["rt", "response"]).isin(data.columns))

            assert "traces.nc" not in zip_ref.namelist()

    basic_model.restore_traces(cav_idata)

    with NamedTemporaryFile() as tmp:
        _save_model(basic_model, tmp.name)

        with ZipFile(tmp.name, "r") as zip_ref:
            assert "model.json" in zip_ref.namelist()
            assert "traces.nc" in zip_ref.namelist()

            with TemporaryDirectory() as tmp_dir:
                zip_ref.extract("traces.nc", path=tmp_dir)
                idata = az.from_netcdf(f"{tmp_dir}/traces.nc")

                assert np.all(idata.posterior == cav_idata.posterior)

    model_with_loglik = hssm.HSSM(
        data=data_ddm,
        loglik=logp_ddm,
        include=[
            {
                "name": "v",
                "prior": {
                    "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0}
                },
                "formula": "v ~ 1",
                "link": "identity",
            }
        ],
    )

    with NamedTemporaryFile() as tmp:
        _save_model(model_with_loglik, tmp.name, save_data_format="parquet")

        with ZipFile(tmp.name, "r") as zip_ref:
            assert "loglik" in zip_ref.namelist()

            with zip_ref.open("model.json") as f:
                model_dict = json.load(f)
                _check_seriaizable_attrs(basic_model, model_dict)

                assert len(model_dict["include"]) == 2
                assert "v" in model_dict["include"]

            assert "data.parquet" in zip_ref.namelist()

            with zip_ref.open("data.parquet") as f:
                data = pd.read_parquet(f)
                assert np.all(pd.Series(["rt", "response"]).isin(data.columns))

            with zip_ref.open("loglik") as f:
                loglik = pickle.load(f)
                assert loglik == logp_ddm


user_attrs = [
    "model_config",
    "process_initvals",
    "initval_jitter",
    "extra_namespace",
]


def check_attr(attr, obj_a, obj_b):
    attr_value_a = getattr(obj_a, attr)
    attr_value_b = getattr(obj_b, attr)

    if attr_value_a is None:
        assert attr_value_b is None
    else:
        assert attr_value_a == attr_value_b


def check_user_attr(attr, obj_a, obj_b):
    attr_a = obj_a.user_spec.get(attr, None)

    if not attr_a:
        assert not obj_b.user_spec.get(attr, None)
    else:
        attr_b = obj_b.user_spec[attr]
        assert attr_a == attr_b


def test__load_model(data_ddm, cav_idata):
    basic_model = hssm.HSSM(data=data_ddm)

    with NamedTemporaryFile() as tmp:
        _save_model(basic_model, tmp.name)

        loaded_model = _load_model(tmp.name)

    for attr in SERIALIZABLE_ATTRS:
        check_attr(attr, basic_model, loaded_model)

    assert "p_outlier" in loaded_model.params
    check_attr("lapse", basic_model, loaded_model)

    for attr in user_attrs:
        check_user_attr(attr, basic_model, loaded_model)

    assert np.allclose(basic_model.data.values, loaded_model.data.values)

    assert loaded_model._inference_obj is None

    basic_model.restore_traces(cav_idata)

    with NamedTemporaryFile() as tmp:
        _save_model(basic_model, tmp.name)

        loaded_model = _load_model(tmp.name)

    assert loaded_model._inference_obj is not None

    basic_model = hssm.HSSM(data=data_ddm)

    with NamedTemporaryFile() as tmp:
        _save_model(basic_model, tmp.name, save_data_format="parquet")

        loaded_model = _load_model(tmp.name)

    assert np.allclose(basic_model.data.values, loaded_model.data.values)

    model_with_loglik = hssm.HSSM(
        data=data_ddm,
        loglik=logp_ddm,
        include=[
            {
                "name": "v",
                "prior": {
                    "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0}
                },
                "formula": "v ~ 1",
                "link": "identity",
            }
        ],
    )

    with NamedTemporaryFile() as tmp:
        _save_model(model_with_loglik, tmp.name)

        loaded_model = _load_model(tmp.name)

    assert loaded_model.user_spec["loglik"] is not None
    assert "v" in loaded_model.params

    with pytest.raises(
        ValueError,
        match="Data not found in the saved model file. "
        + "Please provide the data as an argument.",
    ):
        with NamedTemporaryFile() as tmp:
            _save_model(basic_model, tmp.name, save_data=False)

            _load_model(tmp.name)
