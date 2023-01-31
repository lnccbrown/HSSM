from typing import List

import pandas as pd
import pymc as pm
import pytensor

from hssm import wfpt

from .utils import data_check


def create_model(
    data: pd.DataFrame,
    model: str = "base",
    list_params: List[str] = ["v", "sv", "a", "z", "t"],
    cores: int = 2,
    draws: int = 500,
    tune: int = 500,
    mp_ctx: str = "forkserver",
    response_rates: str = None,
    response: str = None,
    additional_args: List[str] = None,
):
    pytensor.config.floatX = "float32"
    model = wfpt.make_ssm_distribution(
        model=model,
        list_params=list_params,
    )
    data = data_check(data, response_rates, response, additional_args)
    with pm.Model() as m_angle:
        v = pm.Uniform("v", -3.0, 3.0)
        sv = pm.Uniform("sv", 0.0, 1.2)
        a = pm.Uniform("a", 2.0, 3.5)
        z = pm.Uniform("z", 0.1, 0.9)
        t = pm.Uniform("t", 0.0, 1.0)

        rt = model(
            name="rt",
            v=v,
            sv=sv,
            a=a,
            z=z,
            t=t,
            observed=data,
        )
        trace_ddm_nuts_jax_wrapped = pm.sample(
            cores=cores, draws=draws, tune=tune, mp_ctx=mp_ctx
        )
    return trace_ddm_nuts_jax_wrapped
