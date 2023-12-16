from pathlib import Path

import bambi as bmb
import numpy as np
import pytest

import hssm
from hssm import HSSM
from hssm.utils import download_hf
from hssm.likelihoods import DDM, logp_ddm

hssm.set_floatX("float32")

param_v = {
    "name": "v",
    "prior": {
        "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
        "x": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
        "y": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
    },
    "formula": "v ~ 1 + x + y",
}

param_a = param_v | dict(name="a", formula="a ~ 1 + x + y")


@pytest.mark.parametrize(
    "include, should_raise_exception",
    [
        (
            [param_v],
            False,
        ),
        (
            [
                param_v,
                param_a,
            ],
            False,
        ),
        (
            [{"name": "invalid_param", "prior": "invalid_param"}],
            True,
        ),
        (
            [
                {
                    "name": "v",
                    "prior": {
                        "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0}
                    },
                    "formula": "v ~ 1",
                    "invalid_key": "identity",
                }
            ],
            True,
        ),
        (
            [
                {
                    "name": "v",
                    "prior": {
                        "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0}
                    },
                    "formula": "invalid_formula",
                }
            ],
            True,
        ),
    ],
)
def test_transform_params_general(data_ddm_reg, include, should_raise_exception):
    if should_raise_exception:
        with pytest.raises(Exception):
            HSSM(data=data_ddm_reg, include=include)
    else:
        model = HSSM(data=data_ddm_reg, include=include)
        # Check model properties using a loop
        param_names = ["v", "a", "z", "t", "p_outlier"]
        model_param_names = list(model.params.keys())
        assert model_param_names == param_names
        assert len(model.params) == 5


def test_custom_model(data_ddm):
    with pytest.raises(
        ValueError, match="When using a custom model, please provide a `loglik_kind.`"
    ):
        model = HSSM(data=data_ddm, model="custom")

    with pytest.raises(
        ValueError, match="Please provide `list_params` via `model_config`."
    ):
        model = HSSM(data=data_ddm, model="custom", loglik_kind="analytical")

    with pytest.raises(
        ValueError, match="Please provide `list_params` via `model_config`."
    ):
        model = HSSM(
            data=data_ddm, model="custom", loglik=DDM, loglik_kind="analytical"
        )

    with pytest.raises(
        ValueError,
        match="Please provide `list_params` via `model_config`.",
    ):
        model = HSSM(
            data=data_ddm,
            model="custom",
            loglik=DDM,
            loglik_kind="analytical",
            model_config={},
        )

    model = HSSM(
        data=data_ddm,
        model="custom",
        model_config={
            "list_params": ["v", "a", "z", "t"],
            "bounds": {
                "v": (-3.0, 3.0),
                "a": (0.3, 2.5),
                "z": (0.1, 0.9),
                "t": (0.0, 2.0),
            },
        },
        loglik=logp_ddm,
        loglik_kind="analytical",
    )

    assert model.model_name == "custom"
    assert model.loglik_kind == "analytical"
    assert model.list_params == ["v", "a", "z", "t", "p_outlier"]


def test_model_definition_outside_include(data_ddm):
    model_with_one_param_fixed = HSSM(data_ddm, a=0.5)

    assert "a" in model_with_one_param_fixed.priors
    assert model_with_one_param_fixed.priors["a"] == 0.5

    model_with_one_param = HSSM(
        data_ddm, a={"prior": {"name": "Normal", "mu": 0.5, "sigma": 0.1}}
    )

    assert "a" in model_with_one_param.priors
    assert model_with_one_param.priors["a"].name == "Normal"

    with pytest.raises(
        ValueError, match='Parameter "a" is already specified in `include`'
    ):
        HSSM(data_ddm, include=[{"name": "a", "prior": 0.5}], a=0.5)


def test_model_with_approx_differentiable_likelihood_type(data_angle):
    loglik_kind = "approx_differentiable"
    loglik = "angle.onnx"
    model = HSSM(data=data_angle, model="angle", loglik_kind=loglik_kind, loglik=loglik)
    assert model.loglik == download_hf(loglik)


def test_sample_prior_predictive(data_ddm_reg):
    data_ddm_reg = data_ddm_reg.iloc[:10, :]

    model_no_regression = HSSM(data=data_ddm_reg)
    rng = np.random.default_rng()

    prior_predictive_1 = model_no_regression.sample_prior_predictive(draws=10)
    prior_predictive_2 = model_no_regression.sample_prior_predictive(
        draws=10, random_seed=rng
    )

    model_regression = HSSM(
        data=data_ddm_reg, include=[dict(name="v", formula="v ~ 1 + x")]
    )
    prior_predictive_3 = model_regression.sample_prior_predictive(draws=10)

    model_regression_a = HSSM(
        data=data_ddm_reg, include=[dict(name="a", formula="a ~ 1 + x")]
    )
    prior_predictive_4 = model_regression_a.sample_prior_predictive(draws=10)

    model_regression_multi = HSSM(
        data=data_ddm_reg,
        include=[
            dict(name="v", formula="v ~ 1 + x"),
            dict(name="a", formula="a ~ 1 + y"),
        ],
    )
    prior_predictive_5 = model_regression_multi.sample_prior_predictive(draws=10)

    data_ddm_reg.loc[:, "subject_id"] = np.arange(10)

    model_regression_random_effect = HSSM(
        data=data_ddm_reg,
        include=[
            dict(name="v", formula="v ~ (1|subject_id) + x"),
            dict(name="a", formula="a ~ (1|subject_id) + y"),
        ],
    )
    prior_predictive_6 = model_regression_random_effect.sample_prior_predictive(
        draws=10
    )


def test_hierarchical(data_ddm):
    data_ddm = data_ddm.iloc[:10, :].copy()
    data_ddm["participant_id"] = np.arange(10)

    model = HSSM(data=data_ddm, hierarchical=True)
    assert all(
        param.is_regression
        for name, param in model.params.items()
        if name != "p_outlier"
    )
    assert model.prior_settings == "safe"

    model = HSSM(
        data=data_ddm,
        v=bmb.Prior("Uniform", lower=-10.0, upper=10.0),
        hierarchical=True,
    )
    assert all(
        param.is_regression
        for name, param in model.params.items()
        if name not in ["v", "p_outlier"]
    )

    model = HSSM(
        data=data_ddm, a=bmb.Prior("Uniform", lower=0.0, upper=10.0), hierarchical=True
    )
    assert all(
        param.is_regression
        for name, param in model.params.items()
        if name not in ["a", "p_outlier"]
    )

    model = HSSM(
        data=data_ddm,
        v=bmb.Prior("Uniform", lower=-10.0, upper=10.0),
        a=bmb.Prior("Uniform", lower=0.0, upper=10.0),
        hierarchical=True,
    )
    assert all(
        param.is_regression
        for name, param in model.params.items()
        if name not in ["v", "a", "p_outlier"]
    )


def test_override_default_link(caplog, data_ddm_reg):
    param_v = {
        "name": "v",
        "prior": {
            "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
            "x": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
            "y": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
        },
        "formula": "v ~ 1 + x + y",
    }
    param_v = param_v | dict(bounds=(-np.inf, np.inf))
    param_a = param_v | dict(name="a", formula="a ~ 1 + x + y", bounds=(0, np.inf))
    param_z = param_v | dict(name="z", formula="z ~ 1 + x + y", bounds=(0, 1))
    param_t = param_v | dict(name="t", formula="t ~ 1 + x + y", bounds=(0.1, np.inf))

    model = HSSM(
        data=data_ddm_reg,
        include=[param_v, param_a, param_z, param_t],
        link_settings="log_logit",
    )

    assert model.params["v"].link == "identity"
    assert model.params["a"].link == "log"
    assert model.params["z"].link.name == "gen_logit"
    assert model.params["t"].link == "identity"

    assert "t" in caplog.records[0].message
    assert "strange" in caplog.records[0].message
