from __future__ import annotations

from typing import List

import bambi as bmb
import pandas as pd

from hssm import wfpt
from hssm.utils import Param
from hssm.wfpt.config import default_model_config


# add custom link function
class HSSM:
    """
    Initialize the HSSM class.

    Args:
        data: A pandas DataFrame containing the data to be analyzed.
        model_name: The name of the model to use. Default is "analytical".
        include: A list of dictionaries specifying additional parameters
         to include in the model.
        model_config: A dictionary containing the model configuration information.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        model_name: str = "analytical",
        include: List[dict] = None,
        model_config: dict = None,
    ):
        self.model_config = (
            model_config if model_config is not None else default_model_config
        )
        self.list_params = self.model_config[model_name]["list_params"]
        if model_name == "analytical":
            self.ssm_model = wfpt.WFPT
        else:
            self.ssm_model = wfpt.make_ssm_distribution(
                model=self.model_config[model_name]["model"],  # type: ignore
                list_params=self.list_params,  # type: ignore
                # type: ignore
                backend=self.model_config[model_name]["backend"],
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
            link=self.model_config[model_name]["link"],
        )
        self.priors = {}
        for param in self.list_params:
            self.priors[param] = bmb.Prior(  # type: ignore
                self.model_config[model_name]["prior"][param]["name"],  # type: ignore
                lower=self.model_config[model_name]["prior"][param]["lower"],
                upper=self.model_config[model_name]["prior"][param]["upper"],
            )

        self.formula = self.model_config[model_name]["formula"]

        if include:
            self._transform_include(include)

        self.model = bmb.Model(
            self.formula, data, family=self.family, priors=self.priors
        )

    def _transform_include(self, params: List[dict]) -> None:
        formulas = [p["formula"] for p in params if p.get("formula")]
        first_item = formulas[0].split(" ~ ")[0]
        formulas[0] = formulas[0].replace(first_item, "c(rt,response)")
        self.formula = bmb.Formula(*formulas)
        self.params = []
        for dictionary in params:
            self.params.append(Param(**dictionary))
            coefs = dictionary["formula"].split(" ~ ")[1]
            coefs = coefs.split(" + ")
            coefs[coefs.index("1")] = "Intercept"
            self.priors[dictionary["name"]] = {}
            for coef in coefs:
                try:
                    new_prior = bmb.Prior(
                        dictionary["prior"][coef]["name"],
                        lower=dictionary["prior"][coef]["lower"],
                        upper=dictionary["prior"][coef]["upper"],
                        initval=dictionary["prior"][coef]["initval"],
                    )
                except KeyError:
                    new_prior = bmb.Prior(
                        dictionary["prior"][coef]["name"],
                        lower=dictionary["prior"][coef]["lower"],
                        upper=dictionary["prior"][coef]["upper"],
                    )
                self.priors[dictionary["name"]][coef] = new_prior

    def sample(
        self,
        cores: int = 2,
        draws: int = 500,
        tune: int = 500,
        mp_ctx: str = "fork",
        sampler: str = "pytensor",
    ):
        if sampler == "jax":
            return self.model.fit(
                cores=cores,
                draws=draws,
                tune=tune,
                mp_ctx=mp_ctx,
                inference_method="nuts_numpyro",
            )
        else:
            return self.model.fit(cores=cores, draws=draws, tune=tune, mp_ctx=mp_ctx)
