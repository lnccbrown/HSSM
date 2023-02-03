from typing import Sequence

import pandas as pd
import pymc as pm
import pytensor

from hssm import wfpt
from hssm.wfpt.utils import data_check


def create_hssm(
    data: pd.DataFrame,
    model: str = "base",
    cores: int = 2,
    draws: int = 500,
    tune: int = 500,
    mp_ctx: str = "forkserver",
    response_rates: str = None,
    response: str = None,
    additional_args: Sequence[str] = None,
):
    pytensor.config.floatX = "float32"
    parameters = {
        "base": {
            "model": "base",
            "list_params": ["v", "sv", "a", "z", "t"],
            "backend": "pytensor",
        },
        "lan": {
            "model": "test.onnx",
            "list_params": ["v", "sv", "a", "z", "theta"],
            "backend": "jax",
        },
    }
    params = parameters[model]["list_params"]
    model = wfpt.make_ssm_distribution(
        model=parameters[model]["model"],
        list_params=parameters[model]["list_params"],
        backend=parameters[model]["backend"],
    )
    data = data_check(data, response_rates, response, additional_args)
    with pm.Model() as m_angle:
        param0 = pm.Uniform(params[0], -3.0, 3.0)
        param1 = pm.Uniform(params[1], 0.0, 1.2)
        param2 = pm.Uniform(params[2], 2.0, 3.5)
        param3 = pm.Uniform(params[3], 0.1, 0.9)
        param4 = pm.Uniform(params[4], 0.0, 1.0)
        if "t" in params:
            rt = model(  # type: ignore
                name="rt",
                v=param0,
                sv=param1,
                a=param2,
                z=param3,
                t=param4,
                observed=data,
            )
            samples = pm.sample(cores=cores, draws=draws, tune=tune, mp_ctx=mp_ctx)
        else:
            rt = model(  # type: ignore
                name="rt",
                v=param0,
                sv=param1,
                a=param2,
                z=param3,
                theta=param4,
                observed=data,
            )
            samples = pm.sample(cores=cores, draws=draws, tune=tune, mp_ctx=mp_ctx)

    return samples
