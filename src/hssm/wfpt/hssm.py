from __future__ import annotations

from typing import List

import bambi as bmb
import pandas as pd

from hssm import wfpt
from hssm.wfpt.config import default_model_config
from hssm.wfpt.utils import formula_replacer


class HSSM:
    def __init__(
        self,
        data: pd.DataFrame,
        model_name: str = "analytical",
        include: dict | List[dict] = None,  # type: ignore
        model_config: dict = None,
    ):
        self.model_config = (
            model_config if model_config is not None else default_model_config
        )
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
        self.priors = {}
        for param in self.list_params:
            self.priors[param] = bmb.Prior(
                "Uniform",
                lower=self.model_config[model_name]["priors"][param][0],  # type: ignore
                upper=self.model_config[model_name]["priors"][param][1],  # type: ignore
            )

        self.formula = self.model_config[model_name]["formula"]
        if isinstance(include, dict) and include.get("formula"):  # type: ignore
            self.formula = formula_replacer(self.formula, include)
            self.priors[include["param"]] = bmb.Prior(
                "Uniform", lower=include["priors"][0], upper=include["priors"][1]
            )
        elif isinstance(include, list):
            formulas = [item["formula"] for item in include if item.get("formula")]
            self.formula = bmb.Formula(*formulas)
            self.priors[include["param"]] = bmb.Prior(  # type: ignore
                "Uniform",
                lower=include["priors"][0],  # type: ignore
                upper=include["priors"][1],  # type: ignore
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
        method: str = None,  # nuts_numpyro if an user want to use jax
    ):
        return self.model.fit(
            cores=cores, draws=draws, tune=tune, mp_ctx=mp_ctx, method=method
        )
