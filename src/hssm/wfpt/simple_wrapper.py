# pylint: disable=unused-variable
import bambi as bmb
import pandas as pd
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
                "formula": "c(rt,response)  ~ 1",
            },
            "lan": {
                "model": "test.onnx",
                "list_params": ["v", "sv", "a", "z", "theta"],
                "backend": "jax",
                "formula": "c(rt,response)  ~ 1",
            },
        }

    def hssm(
        self,
        data: pd.DataFrame,
        response_rates: str = None,
        response: str = None,
        additional_args: list = None,
        model: str = "base",
        formula: str = None,
    ):
        data = data_check(data, response_rates, response, additional_args)
        params = self.parameters[model]["list_params"]
        ssm_model = wfpt.make_ssm_distribution(
            model=self.parameters[model]["model"],
            list_params=params,
            backend=self.parameters[model]["backend"],
        )
        likelihood = bmb.Likelihood(
            self.parameters[model]["model"],
            params=params,
            parent=params[0],
            dist=ssm_model,
        )
        family = bmb.Family(
            self.parameters[model]["model"],
            likelihood=likelihood,
            link={param: "identity" for param in params},
        )
        priors = {
            "Intercept": bmb.Prior("Uniform", lower=-3, upper=3),
            params[1]: bmb.Prior("Uniform", lower=0.0, upper=1.2),
            params[2]: bmb.Prior("Uniform", lower=0.5, upper=2.0),
            params[3]: bmb.Prior("Uniform", lower=0.1, upper=0.9),
            params[4]: bmb.Prior("Uniform", lower=0.0, upper=2.0),
        }
        bmb_ddm = bmb.Model(
            self.parameters[model]["formula"] if formula is None else formula,
            data,
            family=family,
            priors=priors,
        )
        return bmb_ddm
