import re
from typing import List

import bambi as bmb
import pandas as pd

from hssm import wfpt
from hssm.utils import Param
from hssm.wfpt.config import default_model_config


# add custom link function
class HSSM:
    """
    The Hierarchical Sequential Sampling Model (HSSM) class.

    Args:
    data (pandas.DataFrame): A pandas DataFrame containing the data to be
     analyzed.
    model_name (str): The name of the model to use. Default is "analytical".
    include (List[dict], optional): A list of dictionaries specifying additional
     parameters to include in the model. Defaults to None.
    model_config (dict, optional): A dictionary containing the model
     configuration information. Defaults to None.

    Attributes:
        list_params (list): The list of parameter names.
        model_name (str): The name of the model.
        ssm_model: A SSM model object.
        likelihood: A Bambi likelihood object.
        family: A Bambi family object.
        priors (dict): A dictionary containing the prior distribution
         of parameters.
        formula (str): A string representing the model formula.
        params (list): A list of Param objects representing model parameters.
        model: A Bambi model object.

    Methods:
        _transform_include: A method to transform include list
         into Bambi's format.
        sample: A method to sample posterior distributions.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        model_name: str = "analytical",
        include: List[dict] = None,
        model_config: dict = None,
        link: dict = None,
    ):
        if model_config is not None:
            model_name = None

        self.model_config = (
            model_config if model_config else default_model_config[model_name]
        )

        self.list_params = self.model_config["list_params"]
        self.parent = self.model_config["list_params"][0]
        if model_name == "analytical":
            self.ssm_model = wfpt.WFPT
        else:
            self.ssm_model = wfpt.make_ssm_distribution(
                model=self.model_config["model"],  # type: ignore
                list_params=self.list_params,  # type: ignore
                backend=self.model_config["backend"],
            )
        self.likelihood = bmb.Likelihood(
            self.model_config["model"],
            params=self.list_params,
            parent=self.parent,
            dist=self.ssm_model,
        )
        self.family = bmb.Family(
            self.model_config["model"],
            likelihood=self.likelihood,
            link=link if link else self.model_config["link"],
        )
        self.priors = {}
        for param in self.list_params:
            self.priors[param] = (
                bmb.Prior(
                    self.model_config["prior"][param]["name"],
                    lower=self.model_config["prior"][param]["lower"],
                    upper=self.model_config["prior"][param]["upper"],
                )
                if param != self.parent
                else {
                    "Intercept": bmb.Prior(
                        self.model_config["prior"][param]["Intercept"]["name"],
                        lower=self.model_config["prior"][param]["Intercept"]["lower"],
                        upper=self.model_config["prior"][param]["Intercept"]["upper"],
                    )
                }
            )

        self.formula = self.model_config["formula"]

        if include:
            self._transform_include(include)

        self.model = bmb.Model(
            self.formula, data, family=self.family, priors=self.priors
        )

    def _transform_include(self, include: List[dict]) -> None:
        if all("formula" in d for d in include):
            formulas = [p.get("formula") for p in include if p.get("formula")]
            v_dict = next((p for p in include if p.get("name") == "v"), None)
            if not v_dict:
                formulas.insert(0, self.model_config["formula"])
            else:
                first_item = formulas[0].split("~")[0]  # type: ignore
                formulas[0] = formulas[0].replace(  # type: ignore
                    first_item, "c(rt,response)"
                )

            self.formula = bmb.Formula(*formulas)
        self.params = []
        for dictionary in include:
            self.params.append(Param(**dictionary))
            if "formula" in dictionary:
                dictionary["formula"] = re.sub(r"\s+", "", dictionary["formula"])
                coefs = dictionary["formula"].split("~")[1]
                coefs = coefs.split("+")
                coefs[coefs.index("1")] = "Intercept"
                self.priors[dictionary["name"]] = {}
                for coef in coefs:
                    lower_key = list((dictionary["prior"][coef].keys()))[1]
                    upper_key = list((dictionary["prior"][coef].keys()))[2]

                    if "initval" not in dictionary["prior"][coef]:
                        new_prior = bmb.Prior(
                            dictionary["prior"][coef]["name"],
                            lower=dictionary["prior"][coef][lower_key],
                            upper=dictionary["prior"][coef][upper_key],
                        )
                    else:
                        new_prior = bmb.Prior(
                            dictionary["prior"][coef]["name"],
                            lower=dictionary["prior"][coef][lower_key],
                            upper=dictionary["prior"][coef][upper_key],
                            initval=dictionary["prior"][coef]["initval"],
                        )
                    self.priors[dictionary["name"]][coef] = new_prior
            elif isinstance(dictionary["prior"], (int, float)):
                self.priors[dictionary["name"]] = dictionary["prior"]

    def sample(
        self,
        cores: int = 2,
        draws: int = 500,
        tune: int = 500,
        mp_ctx: str = "fork",
        sampler: str = "pytensor",
    ):
        """
        Perform posterior sampling using the `fit` function of Bambi.

        Args:
            cores (int): The number of cores to use for parallel sampling.
             Default is 2.
            draws (int): The number of posterior samples to draw. Default is 500.
            tune (int): The number of tuning steps to perform before starting
             to draw posterior samples. Default is 500.
            mp_ctx (str): The multiprocessing context to use. Default is "fork".
            sampler (str): The sampler to use. Can be either "jax" or "pytensor".
             Default is "pytensor".

        Returns:
            The posterior samples, which is an instance of the
             `arviz.InferenceData` class.
        """
        if sampler == "jax":
            return self.model.fit(
                cores=cores,
                draws=draws,
                tune=tune,
                inference_method="nuts_numpyro",
                chain_method="parallel",
            )
        else:
            return self.model.fit(cores=cores, draws=draws, tune=tune, mp_ctx=mp_ctx)