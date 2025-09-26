import pytest

import arviz as az
import hssm
import numpy as np

from hssm.utils import _rearrange_data

hssm.set_floatX("float32", update_jax=True)


@pytest.fixture
def data_ddm_missing(data_ddm):
    data = data_ddm.copy()
    missing_indices = np.random.choice(data_ddm.shape[0], 50, replace=False)
    data.iloc[missing_indices, 0] = -999.0

    return _rearrange_data(data)


# @pytest.fixture
# def data_ddm_deadline(data_ddm):
#     data = data_ddm.copy()
#     data["deadline"] = data["rt"] + np.random.normal(0, 0.1, data.shape[0])
#     missing_indices = data["rt"] > data["deadline"]
#     data.iloc[missing_indices, 0] = -999.0

#     return _rearrange_data(data)


@pytest.fixture
def data_ddm_reg_missing(data_ddm_reg):
    data = data_ddm_reg.copy()
    missing_indices = np.random.choice(data_ddm_reg.shape[0], 50, replace=False)
    data.iloc[missing_indices, 0] = -999.0

    return _rearrange_data(data)


# @pytest.fixture
# def data_ddm_reg_deadline(data_ddm_reg):
#     data = data_ddm_reg.copy()
#     data["deadline"] = data["rt"] + np.random.normal(0, 0.1, data.shape[0])
#     missing_indices = data["rt"] > data["deadline"]
#     data.iloc[missing_indices, 0] = -999.0

#     return _rearrange_data(data)


@pytest.fixture
def fixture_path():
    from pathlib import Path

    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def cpn(fixture_path):
    return fixture_path / "ddm_cpn.onnx"


# @pytest.fixture
# def opn(fixture_path):
#     return fixture_path / "ddm_opn.onnx"


PARAMETER_NAMES = "loglik_kind, backend, method, expected"
PARAMETER_GRID = [
    ("blackbox", "pytensor", "advi", ValueError),  # Defaults should work
    ("blackbox", "pytensor", "fullrank_advi", ValueError),
    ("approx_differentiable", "pytensor", "advi", True),  # Defaults should work
    ("approx_differentiable", "pytensor", "fullrank_advi", True),
    ("approx_differentiable", "jax", "advi", True),  # Defaults should work
    ("approx_differentiable", "jax", "fullrank_advi", True),
]


def run_vi(model, method, expected):
    import pymc as pm

    if expected == True:
        model.vi(method, niter=1000)
        assert isinstance(model.vi_idata, az.InferenceData)
        assert isinstance(model.vi_approx, pm.Approximation)
    else:
        with pytest.raises(expected):
            model.vi(method, niter=1000)


@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, PARAMETER_GRID)
# @pytest.mark.xfail(reason="Needs to be reactivated, CPN logic needs to be revised")
def test_simple_models_missing_data(
    data_ddm_missing, loglik_kind, backend, method, expected, cpn
):
    model = hssm.HSSM(
        data_ddm_missing,
        loglik_kind=loglik_kind,
        model_config={"backend": backend},
        missing_data=True,
        loglik_missing_data=cpn,
        deadline=False,
    )
    run_vi(model, method, expected)


@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, PARAMETER_GRID)
# @pytest.mark.xfail(reason="Needs to be reactivated, CPN logic needs to be revised")
def test_reg_models_missing_data(
    data_ddm_reg_missing, loglik_kind, backend, method, expected, cpn
):
    param_reg = dict(
        formula="v ~ 1 + x + y",
        prior={
            "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
            "x": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
            "y": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
        },
    )
    model = hssm.HSSM(
        data_ddm_reg_missing,
        loglik_kind=loglik_kind,
        model_config={"backend": backend},
        v=param_reg,
        missing_data=True,
        loglik_missing_data=cpn,
        deadline=False,
    )
    run_vi(model, method, expected)


# @pytest.mark.slow
# @pytest.mark.parametrize(PARAMETER_NAMES, PARAMETER_GRID)
# def test_simple_models_deadline(
#     data_ddm_deadline, loglik_kind, backend, method, expected, opn
# ):
#     model = hssm.HSSM(
#         data_ddm_deadline,
#         loglik_kind=loglik_kind,
#         model_config={"backend": backend},
#         deadline=True,
#         missing_data=True,
#         loglik_missing_data=opn,
#     )
#     run_vi(model, method, expected)


# @pytest.mark.slow
# @pytest.mark.parametrize(PARAMETER_NAMES, PARAMETER_GRID)
# def test_reg_models_deadline(
#     data_ddm_reg_deadline, loglik_kind, backend, method, expected, opn
# ):
#     param_reg = dict(
#         formula="v ~ 1 + x + y",
#         prior={
#             "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
#             "x": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
#             "y": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
#         },
#     )
#     model = hssm.HSSM(
#         data_ddm_reg_deadline,
#         loglik_kind=loglik_kind,
#         model_config={"backend": backend},
#         v=param_reg,
#         deadline=True,
#         missing_data=True,
#         loglik_missing_data=opn,
#     )
#     run_vi(model, method, expected)
