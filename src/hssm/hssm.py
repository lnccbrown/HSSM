from __future__ import annotations

from typing import List

import bambi as bmb
import pandas as pd

from hssm import wfpt
from hssm.utils import Param, formula_replacer
from hssm.wfpt.config import default_model_config


# add custom link function
class HSSM:
    def __init__(
        self,
        data: pd.DataFrame,
        model_name: str = "analytical",
        include: dict | List[dict] = None,
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
            self.priors[param] = bmb.Prior(
                # type: ignore
                self.model_config[model_name]["prior"][param]["name"],
                # type: ignore
                lower=self.model_config[model_name]["prior"][param]["lower"],
                upper=self.model_config[model_name]["prior"][param]["upper"],
            )

        self.formula = self.model_config[model_name]["formula"]

        if isinstance(include, dict) and include.get("formula"):  # type: ignore
            self.params = Param(**include)
            self.formula = formula_replacer(self.formula, include)
            coefs = self.formula.split(" ~ ")[1].split(" + ")
            coefs[coefs.index("1")] = "Intercept"
            self.priors[include["name"]] = {}
            for coef in coefs:
                if include["prior"][coef].get("initval") is not None:
                    new_prior = bmb.Prior(
                        include["prior"][coef]["name"],
                        lower=include["prior"][coef]["lower"],
                        upper=include["prior"][coef]["upper"],
                        initval=include["prior"][coef]["initval"],
                    )
                    self.priors[include["name"]][coef] = new_prior
                else:
                    new_prior = bmb.Prior(
                        include["prior"][coef]["name"],
                        lower=include["prior"][coef]["lower"],
                        upper=include["prior"][coef]["upper"],
                    )
                    self.priors[include["name"]][coef] = new_prior
        elif isinstance(include, list):
            formulas = [item["formula"] for item in include if item.get("formula")]
            first_item = formulas[0].split(" ~ ")[0]
            formulas[0] = formulas[0].replace(first_item, "c(rt,response)")
            self.formula = bmb.Formula(*formulas)
            for dictionary in include:
                self.params = Param(**dictionary)
                coefs = dictionary["formula"].split(" ~ ")[1]
                coefs = coefs.split(" + ")
                coefs[coefs.index("1")] = "Intercept"
                self.priors[dictionary["name"]] = {}
                for coef in coefs:
                    if dictionary["prior"].get("initval") is not None:
                        new_prior = bmb.Prior(
                            dictionary["prior"][coef]["name"],
                            lower=dictionary["prior"][coef]["lower"],
                            upper=dictionary["prior"][coef]["upper"],
                            initval=dictionary["prior"][coef]["initval"],
                        )
                        self.priors[dictionary["name"]][coef] = new_prior
                    else:
                        new_prior = bmb.Prior(
                            dictionary["prior"][coef]["name"],
                            lower=dictionary["prior"][coef]["lower"],
                            upper=dictionary["prior"][coef]["upper"],
                        )
                        self.priors[dictionary["name"]][coef] = new_prior

        self.model = bmb.Model(
            self.formula, data, family=self.family, priors=self.priors
        )

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
