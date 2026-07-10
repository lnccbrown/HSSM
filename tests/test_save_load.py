import numpy as np
import pytest
import xarray as xr

import hssm


# Check some attributes are the same
def compare_hssm_class_attributes(model_a, model_b):
    a = np.array([type(v) for k, v in model_a._init_args.items()])
    b = np.array([type(v) for k, v in model_b._init_args.items()])
    assert (a == b).all(), "Init arg types not the same"
    assert (model_a.data).equals(model_b.data), "Data not the same"
    assert model_a.model_config.model_name == model_b.model_config.model_name, (
        "Model name not the same"
    )
    assert model_a.pymc_model._repr_latex_() == model_b.pymc_model._repr_latex_(), (
        "Latex representation of model not the same"
    )
    assert [tmp_.name for tmp_ in model_a.pymc_model.basic_RVs] == [
        tmp_.name for tmp_ in model_b.pymc_model.basic_RVs
    ], "Basic RVs not the same"


@pytest.mark.slow
def test_save_load_model_only(basic_hssm_model, tmp_path):
    tmp_model_name = "hssm_model_pytest"
    basic_hssm_model.save_model(
        model_name=tmp_model_name, base_path=tmp_path, allow_absolute_base_path=True
    )
    loaded_model = hssm.HSSM.load_model(path=tmp_path / tmp_model_name)
    compare_hssm_class_attributes(basic_hssm_model, loaded_model)


@pytest.mark.slow
def test_save_load_vi_mcmc(basic_hssm_model, tmp_path):
    # Sample to attach vi and mcmc traces to model
    # Using minimal parameters since we only need traces to exist, not be accurate
    basic_hssm_model.sample(
        sampler="numpyro", tune=10, draws=10, chains=1, mp_ctx="spawn"
    )

    # 1
    # Save model and mcmc traces, no vi
    tmp_model_name_1 = "hssm_model_pytest_1"
    basic_hssm_model.save_model(
        model_name=tmp_model_name_1, base_path=tmp_path, allow_absolute_base_path=True
    )
    loaded_model = hssm.HSSM.load_model(path=tmp_path / tmp_model_name_1)
    assert isinstance(loaded_model, hssm.HSSM)

    # Check that idata is attached to loaded model
    compare_hssm_class_attributes(basic_hssm_model, loaded_model)
    assert loaded_model._inference_obj is not None
    assert loaded_model._inference_obj_vi is None

    # 2
    # Save whole model after running vi as well
    basic_hssm_model.vi(method="advi", niter=10)
    basic_hssm_model.vi(method="advi", niter=10)
    tmp_model_name_2 = "hssm_model_pytest_2"
    basic_hssm_model.save_model(
        model_name=tmp_model_name_2, base_path=tmp_path, allow_absolute_base_path=True
    )

    loaded_model = hssm.HSSM.load_model(path=tmp_path / tmp_model_name_2)
    assert isinstance(loaded_model, hssm.HSSM)

    # Check that idata is attached to loaded model
    assert loaded_model._inference_obj is not None
    assert loaded_model._inference_obj_vi is not None
    compare_hssm_class_attributes(basic_hssm_model, loaded_model)

    # 3
    # Save and load idata only
    tmp_model_name_3 = "hssm_model_pytest_3"
    basic_hssm_model.save_model(
        model_name=tmp_model_name_3,
        save_traces_only=True,
        base_path=tmp_path,
        allow_absolute_base_path=True,
    )

    loaded_idata = hssm.HSSM.load_model(path=tmp_path / tmp_model_name_3)

    assert isinstance(loaded_idata, xr.DataTree)
    assert loaded_idata["idata_mcmc"] is not None
    assert loaded_idata["idata_vi"] is not None

    # 4
    # Save model with vi traces, no mcmc traces
    # Just need to delete the vi traces and save/load again here
    basic_hssm_model._inference_obj_vi = None
    tmp_model_name_4 = "hssm_model_pytest_4"
    basic_hssm_model.save_model(
        model_name=tmp_model_name_4, base_path=tmp_path, allow_absolute_base_path=True
    )
    loaded_model = hssm.HSSM.load_model(path=tmp_path / tmp_model_name_4)

    # Check that vi traces are not attached to loaded model
    assert isinstance(loaded_model, hssm.HSSM)
    assert loaded_model._inference_obj_vi is None
    assert loaded_model._inference_obj is not None
    compare_hssm_class_attributes(basic_hssm_model, loaded_model)


@pytest.mark.parametrize(
    ("existing_filename", "loaded_group", "missing_filename"),
    [
        ("traces.nc", "idata_mcmc", "vi_traces.nc"),
        ("vi_traces.nc", "idata_vi", "traces.nc"),
    ],
)
def test_load_model_traces_tolerates_each_missing_file(
    caplog,
    tmp_path,
    existing_filename,
    loaded_group,
    missing_filename,
):
    """Either trace file can be loaded when its counterpart is absent."""
    traces = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {"theta": (("chain", "draw"), np.array([[0.25, 0.75]]))},
                coords={"chain": [0], "draw": [0, 1]},
            )
        }
    )
    traces.to_netcdf(tmp_path / existing_filename)

    loaded = hssm.HSSM.load_model_traces(tmp_path)

    assert set(loaded.children) == {loaded_group}
    xr.testing.assert_identical(
        loaded[f"{loaded_group}/posterior"].ds,
        traces["posterior"].ds,
    )
    assert f"{missing_filename} file does not exist" in caplog.text
