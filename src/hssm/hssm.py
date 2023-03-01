from __future__ import annotations

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
    data (pandas.DataFrame): A pandas DataFrame with the minimum requirements of
        containing the data with the columns 'rt' and 'response'.
    model_name (str): The name of the model to use. Default is "ddm".
    Current default implementations are "ddm" | "lan" | "custom".
        ddm - Computes the log-likelihood of the drift diffusion model f(t|v,a,z) using
        the method and implementation of Navarro & Fuss, 2009.
        lan - Likelihood Approximation Network (LAN) extension for the Wiener
         First-Passage Time (WFPT) distribution.
    include (List[dict], optional): A list of dictionaries specifying additional
     parameters to include in the model. Defaults to None. Priors specification range:
        v: Mean drift rate. (-inf, inf).
        sv: Standard deviation of the drift rate [0, inf).
        a: Value of decision upper bound. (0, inf).
        z: Normalized decision starting point. (0, 1).
        t: Non-decision time [0, inf).
        theta: [0, inf).
    model_config (dict, optional): A dictionary containing the model
     configuration information. Defaults to None.

    Attributes:
        list_params (list): The list of parameter names.
        model_name (str): The name of the model.
        model_distribution: A SSM model object.
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
        model_name: str | None = "ddm",
        include: List[dict] | None = None,
        model_config: dict | None = None,
    ):
        if model_name not in ["lan", "custom", "ddm"]:
            raise Exception("Please provide a correct model_name")

        self.model_config = (
            model_config if model_config else default_model_config[model_name]
        )

        self.list_params = self.model_config["list_params"]
        if model_name == "ddm":
            self.model_distribution = wfpt.WFPT
        elif model_name == "lan":
            self.model_distribution = wfpt.make_lan_distribution(
                model=self.model_config["model"],  # type: ignore
                list_params=self.list_params,
                backend=self.model_config["backend"],
            )
        self.likelihood = bmb.Likelihood(
            self.model_config["model"],
            params=self.list_params,
            parent=self.model_config["list_params"][0],
            dist=self.model_distribution,
        )

        self.priors = {}
        for param in self.list_params:
            if param == self.model_config["list_params"][0]:
                self.priors["Intercept"] = (
                    self.model_config["prior"][param]["Intercept"]
                    if isinstance(
                        self.model_config["prior"][param]["Intercept"], bmb.Prior
                    )
                    else bmb.Prior(**self.model_config["prior"][param]["Intercept"])
                )
            else:
                self.priors[param] = (
                    self.model_config["prior"][param]
                    if isinstance(self.model_config["prior"][param], bmb.Prior)
                    else bmb.Prior(**self.model_config["prior"][param])
                )

        self.formula = self.model_config["formula"]
        self.link = self.model_config["link"]

        if include:
            self._transform_include(include)

        self.family = bmb.Family(
            self.model_config["model"],
            likelihood=self.likelihood,
            link=self.link,
        )

        self.model = bmb.Model(
            self.formula, data, family=self.family, priors=self.priors
        )

    def _transform_include(self, include: List[dict]) -> None:
        if all("formula" in d for d in include):
            formulas = [p.get("formula") for p in include if p.get("formula")]
            v_dict = next(
                (
                    p
                    for p in include
                    if p.get("name") == self.model_config["list_params"][0]
                ),
                None,
            )
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
                    new_prior = (
                        dictionary["prior"][coef]
                        if isinstance(dictionary["prior"][coef], bmb.Prior)
                        else bmb.Prior(**dictionary["prior"][coef])
                    )
                    self.priors[dictionary["name"]][coef] = new_prior
            elif isinstance(dictionary["prior"], (int, float)):
                self.priors[dictionary["name"]] = dictionary["prior"]
            if "link" in dictionary:
                self.link[dictionary["name"]] = dictionary["link"]

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
