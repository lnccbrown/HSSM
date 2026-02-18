import numpy as np


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


def test_save_load_model_only(basic_hssm_model, tmp_path):
    tmp_model_name = "hssm_model_pytest"
    basic_hssm_model.save_model(
        model_name=tmp_model_name, base_path=tmp_path, allow_absolute_base_path=True
    )
    loaded_model = hssm.HSSM.load_model(path=tmp_path / tmp_model_name)
    compare_hssm_class_attributes(basic_hssm_model, loaded_model)


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

    # Check that idata is attached to loaded model
    assert loaded_model._inference_obj is not None
    assert loaded_model._inference_obj_vi is not None
    compare_hssm_class_attributes(basic_hssm_model, loaded_model)

    # 3
    # Save and load idata only
    tmp_model_name_3 = "hssm_model_pytest_3"
    basic_hssm_model.save_model(
        model_name=tmp_model_name_3,
        save_idata_only=True,
        base_path=tmp_path,
        allow_absolute_base_path=True,
    )

    loaded_idata = hssm.HSSM.load_model(path=tmp_path / tmp_model_name_3)

    # Check that idata is attached to loaded model
    assert isinstance(loaded_idata, dict)
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
    assert loaded_model._inference_obj_vi is None
    assert loaded_model._inference_obj is not None
    compare_hssm_class_attributes(basic_hssm_model, loaded_model)
