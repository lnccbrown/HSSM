import pandas as pd
import pymc as pm
import pytensor

from hssm import wfpt
from hssm.wfpt.utils import data_check


# pylint: disable=unused-variable
class HSSM:
    def __init__(self):
        pytensor.config.floatX = "float32"
        self.parameters = {
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

    def hssm(
        self,
        data: pd.DataFrame,
        response_rates: str = None,
        response: str = None,
        additional_args: list = None,
        model: str = "base",
        cores: int = 2,
        draws: int = 500,
        tune: int = 500,
        mp_ctx: str = "forkserver",
    ):
        ssm_model = wfpt.make_ssm_distribution(
            model=self.parameters[model]["model"],
            list_params=self.parameters[model]["list_params"],
            backend=self.parameters[model]["backend"],
        )
        data = data_check(data, response_rates, response, additional_args)
        params = self.parameters[model]["list_params"]

        with pm.Model():
            param0 = pm.Uniform(params[0], -3.0, 3.0)
            param1 = pm.Uniform(params[1], 0.0, 1.2)
            param2 = pm.Uniform(params[2], 2.0, 3.5)
            param3 = pm.Uniform(params[3], 0.1, 0.9)
            param4 = pm.Uniform(params[4], 0.0, 1.0)
            apply_ssm = ssm_model(
                name="rt",
                v=param0,
                sv=param1,
                a=param2,
                z=param3,
                **{params[4]: param4} if params[4] == "t" else {"theta": param4},
                observed=data,
            )
            samples = pm.sample(cores=cores, draws=draws, tune=tune, mp_ctx=mp_ctx)

        return samples
