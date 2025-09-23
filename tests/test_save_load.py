import numpy as np
import shutil
from pathlib import Path

import pytest

import hssm


# Check some attributes are the same
def compare_hssm_class_attributes(model_a, model_b):
    a = np.array([type(v) for k, v in model_a._init_args.items()])
    b = np.array([type(v) for k, v in model_b._init_args.items()])
    assert (a == b).all(), "Init arg types not the same"
    assert (model_a.data).equals(model_b.data), "Data not the same"
    assert model_a.model_name == model_b.model_name, "Model name not the same"
    assert model_a.pymc_model._repr_latex_() == model_b.pymc_model._repr_latex_(), (
        "Latex representation of model not the same"
    )
    assert [tmp_.name for tmp_ in model_a.pymc_model.basic_RVs] == [
        tmp_.name for tmp_ in model_b.pymc_model.basic_RVs
    ], "Basic RVs not the same"


@pytest.mark.slow
def test_save_load_model_only(basic_hssm_model):
    tmp_folder_name = "models_pytest"
    tmp_model_name = "hssm_model_pytest"
    basic_hssm_model.save_model(model_name=tmp_model_name, base_path=tmp_folder_name)
    loaded_model = hssm.HSSM.load_model(
        path=Path(tmp_folder_name).joinpath(tmp_model_name)
    )
    compare_hssm_class_attributes(basic_hssm_model, loaded_model)

    # Clean up generated files
    shutil.rmtree(tmp_folder_name)


@pytest.mark.slow
def test_save_load_vi_mcmc(basic_hssm_model):
    tmp_folder_name = "models_pytest"
    tmp_model_name = "hssm_model_pytest"
    # Sample to attach vi and mcmc traces to model
    basic_hssm_model.sample(sampler="nuts_numpyro", tune=100, draws=100, chains=2)

    # 1
    # Save model and mcmc traces, no vi
    basic_hssm_model.save_model(model_name=tmp_model_name, base_path=tmp_folder_name)
    loaded_model = hssm.HSSM.load_model(
        path=Path(tmp_folder_name).joinpath(tmp_model_name)
    )

    # Check that idata is attached to loaded model
    compare_hssm_class_attributes(basic_hssm_model, loaded_model)
    assert loaded_model._inference_obj is not None
    assert loaded_model._inference_obj_vi is None

    # Clean up generated files
    shutil.rmtree(tmp_folder_name)

    # 2
    # Save whole model after running vi as well
    basic_hssm_model.vi(method="advi", niter=1000)
    basic_hssm_model.vi(method="advi", niter=1000)
    basic_hssm_model.save_model(model_name=tmp_model_name, base_path=tmp_folder_name)

    loaded_model = hssm.HSSM.load_model(
        path=Path(tmp_folder_name).joinpath(tmp_model_name)
    )

    # Check that idata is attached to loaded model
    assert loaded_model._inference_obj is not None
    assert loaded_model._inference_obj_vi is not None
    compare_hssm_class_attributes(basic_hssm_model, loaded_model)

    # Clean up generated files
    shutil.rmtree(tmp_folder_name)

    # 3
    # Save and load idata only
    basic_hssm_model.save_model(
        model_name=tmp_model_name, save_idata_only=True, base_path=tmp_folder_name
    )

    loaded_idata = hssm.HSSM.load_model(
        path=Path(tmp_folder_name).joinpath(tmp_model_name)
    )

    # Check that idata is attached to loaded model
    assert isinstance(loaded_idata, dict)
    assert loaded_idata["idata_mcmc"] is not None
    assert loaded_idata["idata_vi"] is not None

    # Clean up generated files
    shutil.rmtree(tmp_folder_name)

    # 4
    # Save model with vi traces, no mcmc traces
    # Just need to delete the vi traces and save/load again here
    basic_hssm_model._inference_obj_vi = None
    basic_hssm_model.save_model(model_name=tmp_model_name, base_path=tmp_folder_name)
    loaded_model = hssm.HSSM.load_model(
        path=Path(tmp_folder_name).joinpath(tmp_model_name)
    )

    # Check that vi traces are not attached to loaded model
    assert loaded_model._inference_obj_vi is None
    assert loaded_model._inference_obj is not None
    compare_hssm_class_attributes(basic_hssm_model, loaded_model)

    # Clean up generated files
    shutil.rmtree(tmp_folder_name)
