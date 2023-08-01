import pytest
import bambi as bmb
import numpy as np
import hssm

hssm.set_floatX("float32")


@pytest.fixture
def non_reg_data():
    data = hssm.simulate_data(model="ddm", theta=[0.5, 1.5, 0.5, 0.1], size=1000)
    return data


@pytest.fixture
def reg_data():
    # Generate some fake simulation data
    intercept = 0.3
    x = np.random.uniform(0.5, 0.2, size=1000)
    y = np.random.uniform(0.4, 0.1, size=1000)

    v = intercept + 0.8 * x + 0.3 * y
    true_values = np.column_stack(
        [v, np.repeat([[1.5, 0.5, 0.5]], axis=0, repeats=1000)]
    )

    dataset_reg_v = hssm.simulate_data(
        model="ddm",
        theta=true_values,
        size=1,  # Generate one data point for each of the 1000 set of true values
    )

    dataset_reg_v["x"] = x
    dataset_reg_v["y"] = y

    return dataset_reg_v


def test_non_reg_models(non_reg_data):
    model1 = hssm.HSSM(non_reg_data)
    model1.sample(cores=1, chains=1, tune=10, draws=10)
    model1.sample(sampler="nuts_numpyro", cores=1, chains=1, tune=10, draws=10)

    model2 = hssm.HSSM(non_reg_data, loglik_kind="approx_differentiable")
    model2.sample(cores=1, chains=1, tune=10, draws=10)
    model2.sample(sampler="nuts_numpyro", cores=1, chains=1, tune=10, draws=10)


def test_reg_models(reg_data):
    param_reg = dict(
        formula="v ~ 1 + x + y",
        prior={
            "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
            "x": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
            "y": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
        },
    )

    model1 = hssm.HSSM(reg_data, v=param_reg)
    model1.sample(cores=1, chains=1, tune=10, draws=10)
    model1.sample(sampler="nuts_numpyro", cores=1, chains=1, tune=10, draws=10)

    model2 = hssm.HSSM(reg_data, loglik_kind="approx_differentiable", v=param_reg)
    model2.sample(cores=1, chains=1, tune=10, draws=10)
    model2.sample(sampler="nuts_numpyro", cores=1, chains=1, tune=10, draws=10)

    model3 = hssm.HSSM(reg_data, a=param_reg)
    model3.sample(cores=1, chains=1, tune=10, draws=10)
    model3.sample(sampler="nuts_numpyro", cores=1, chains=1, tune=10, draws=10)

    model4 = hssm.HSSM(reg_data, loglik_kind="approx_differentiable", a=param_reg)
    model4.sample(cores=1, chains=1, tune=10, draws=10)
    model4.sample(sampler="nuts_numpyro", cores=1, chains=1, tune=10, draws=10)
