import bambi as bmb
import numpy as np

import hssm
from hssm.param import Param


def test_param_creation_non_regression():
    # Test different param creation strategies
    v = {
        "name": "v",
        "prior": {
            "name": "Normal",
            "mu": 0.0,
            "sigma": 2.0,
        },
    }

    param_v = Param(**v, is_parent=False)

    assert param_v.name == "v"
    assert isinstance(param_v.prior, bmb.Prior)
    assert param_v.prior.args["mu"] == 0.0
    assert not param_v.is_regression
    assert not param_v.is_truncated
    assert not param_v.is_fixed

    a = {
        "name": "a",
        "prior": dict(
            name="Uniform",
            lower=0.01,
            upper=5,
        ),
        "bounds": (0, 1e5),
    }

    param_a = Param(**a, is_parent=False)

    assert param_a.is_truncated
    assert not param_a.is_fixed
    assert param_a._prior is not None
    param_a_output = param_a.__str__().split("\r\n")[1].split("Prior: ")[1]
    assert param_a_output == param_a._prior.__str__()

    # A Uniform prior for `z` over (0, 1) set using bmb.Prior.
    # bounds are not set, existing default bounds will be used
    z = {"name": "z", "prior": bmb.Prior("Uniform", lower=0.0, upper=1.0)}

    param_z = Param(**z, is_parent=False)
    assert not param_z.is_truncated
    assert not param_z.is_regression
    assert not param_z.is_fixed

    z1 = {
        "name": "z1",
        "prior": bmb.Prior("Uniform", lower=0.0, upper=1.0),
        "bounds": (0, 1),
    }

    param_z1 = Param(**z1)
    assert param_z1.is_truncated
    assert not param_z1.is_fixed
    assert param_z1._prior is not None
    param_z1_output = param_z1.__str__().split("\r\n")[1].split("Prior: ")[1]
    assert param_z1_output == param_z1._prior.__str__()

    # A fixed value for t
    t = {"name": "t", "prior": 0.5, "bounds": (0, 1)}
    param_t = Param(**t)
    assert not param_t.is_truncated
    assert param_t.is_fixed
    param_t_output = param_t.__str__().split("\r\n")[1].split("Value: ")[1]
    assert param_t_output == str(param_t.prior)

    model = hssm.HSSM(
        model="angle",
        data=hssm.simulate_data(
            model="angle", theta=[0.5, 1.5, 0.5, 0.5, 0.3], size=10
        ),
        include=[v, a, z, t],
    )

    pv, pa, pz, pt, ptheta, _ = model.params.values()
    assert pv.is_truncated
    assert pa.is_truncated
    assert pz.is_truncated
    assert not pt.is_truncated
    assert pt.is_fixed
    assert not ptheta.is_truncated


def test_param_creation_regression():
    v_reg = {
        "name": "v",
        "formula": "v ~ 1 + x + y",
        "prior": {
            "Intercept": {"name": "Uniform", "lower": 0.0, "upper": 0.5},
            "x": dict(name="Uniform", lower=0.0, upper=1.0),
            "y": bmb.Prior("Uniform", lower=0.0, upper=1.0),
            "z": 0.1,
        },
        "link": "identity",
    }

    v_reg_param = Param(**v_reg)

    assert v_reg_param.is_regression
    assert not v_reg_param.is_fixed
    assert not v_reg_param.is_truncated
    assert v_reg_param.formula == v_reg["formula"]

    # Generate some fake simulation data
    intercept = 0.3
    x = np.random.uniform(0.5, 0.2, size=10)
    y = np.random.uniform(0.4, 0.1, size=10)
    z = np.random.uniform(0.2, 0.5, size=10)

    v = intercept + 0.8 * x + 0.3 * y + 0.1 * z

    true_values = np.column_stack([v, np.repeat([[1.5, 0.5, 0.5]], axis=0, repeats=10)])

    dataset_reg_v = hssm.simulate_data(
        model="ddm",
        theta=true_values,
        size=1,  # Generate one data point for each of the 1000 set of true values
    )

    dataset_reg_v["x"] = x
    dataset_reg_v["y"] = y

    model_reg_v = hssm.HSSM(
        data=dataset_reg_v,
        model="ddm",
        include=[v_reg],
    )

    v_reg_param = model_reg_v.params["v"]

    assert v_reg_param.is_regression
    assert not v_reg_param.is_fixed
    assert not v_reg_param.is_truncated
    assert v_reg_param.formula == v_reg["formula"]
