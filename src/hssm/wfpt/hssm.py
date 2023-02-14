# pylint: disable=unused-variable
import bambi as bmb
import pandas as pd

from hssm import wfpt
from hssm.wfpt.config import model_config
from hssm.wfpt.utils import formula_replacer


class HSSM:
    def __init__(
        self, data: pd.DataFrame, model_name: str = "analytical", include: dict = None
    ):
        self.model_config = model_config

        self.list_params = self.model_config[model_name]["list_params"]
        self.ssm_model = wfpt.make_ssm_distribution(
            model=self.model_config[model_name]["model"],  # type: ignore
            list_params=self.list_params,  # type: ignore
            backend=self.model_config[model_name]["backend"],  # type: ignore
        )
        self.likelihood = bmb.Likelihood(
            self.model_config[model_name]["model"],
            params=self.list_params,
            parent=self.list_params[0],  # type: ignore
            dist=self.ssm_model,
        )
        self.family = bmb.Family(
            self.model_config[model_name]["model"],
            likelihood=self.likelihood,
            link={param: "identity" for param in self.list_params},
        )

        self.formula = self.model_config[model_name]["formula"]
        if include and include.get("depends_on"):
            self.formula = formula_replacer(
                self.formula, include["depends_on"]  # type: ignore
            )

        self.priors = {}
        for i, param in enumerate(self.list_params):
            self.priors[param] = bmb.Prior(
                "Uniform",
                lower=self.model_config[model_name]["priors"][param][0],  # type: ignore
                upper=self.model_config[model_name]["priors"][param][1],  # type: ignore
            )

        if include and include.get("prior"):
            self.priors[include["param"]] = bmb.Prior(
                "Uniform", lower=include["priors"][0], upper=include["priors"][1]
            )

        self.model = bmb.Model(
            self.formula, data, family=self.family, priors=self.priors
        )

    def sample(
        self,
        cores: int = 2,
        draws: int = 500,
        tune: int = 500,
        mp_ctx: str = "forkserver",
    ):
        return self.model.fit(cores=cores, draws=draws, tune=tune, mp_ctx=mp_ctx)
