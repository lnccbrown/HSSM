from typing import Any, Callable, Dict, Literal, Optional, Union

import arviz as az
import bambi as bmb
import pandas as pd
import pymc as pm
import pytensor
from numpy.typing import ArrayLike

from hssm import wfpt
from hssm.utils import HSSMModelGraph, Param, _parse_bambi, get_alias_dict, merge_dicts
from hssm.wfpt.config import Config, SupportedModels, default_model_config, download_hf

LogLikeFunc = Callable[..., ArrayLike]


class HSSM:
    """
    The Hierarchical Sequential Sampling Model (HSSM) class which utilizes specified models for computations.

    Args:
        data (pd.DataFrame): A DataFrame with at least two columns - "rt" and "response", specifying response time and responses, respectively.
        model (str, optional): Model to be utilized. Currently supports "ddm", "angle", "levy", "ornstein", "weibull", "race_no_bias_angle_4", "ddm_seq2_no_bias". "custom" for custom models. Defaults to "ddm".
        include (list[dict] | None, optional): A list of dictionaries specifying parameter specifications to include in the model. Defaults to None if left unspecified.
        likelihood_type (str, optional): Specifies the likelihood type. Defaults to None.
        model_config (Config | None, optional): Model configuration. Defaults to None.
        loglik (LogLikeFunc | pytensor.graph.Op | None, optional): Log-likelihood function. Defaults to None.

    Attributes:
        list_params (list[str]): List of parameter names.
        model_name (str): Name of the model.
        model_config (dict): Model configuration.
        model_distribution (Likelihood distribution): Likelihood function of the model in the form of a pm.Distribution subclass.
        family (bmb.Family): A Bambi family object.
        priors (dict): Prior distribution of parameters.
        formula (str): Model formula.
        link (str or dict): Link functions for all parameters.
        params (list[Param]): List of Param objects representing model parameters.

    Raises:
        ValueError: If unsupported model name is passed or the likelihood type and log-likelihood function are not provided for a custom model, or if the log-likelihood kind is invalid.
        TypeError: If parameter specifications are not in the correct format.

    Returns:
        HSSM: A class object with the specified attributes and methods.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        model: SupportedModels = "ddm",
        include: Optional[list[dict]] = None,
        likelihood_kind: str = None,
        model_config: Optional[Config] = None,
        loglik: Optional[Union[LogLikeFunc, pytensor.graph.Op]] = None,
        **kwargs,
    ):
        self.data = data
        self._inference_obj = None

        if model == "custom":
            if model_config:
                self.model_config = model_config
            else:
                if likelihood_kind is None and loglik is None:
                    raise ValueError(
                        "For custom models, both `likelihood_kind` and `loglik` must be provided."
                    )
                if likelihood_kind == "analytical":
                    model = "ddm"
                elif likelihood_kind == "approx_differentiable":
                    model = "angle"
                self.model_config = default_model_config[model]
        else:
            if model not in default_model_config:
                supported_models = list(default_model_config.keys())
                raise ValueError(
                    f"`model` must be one of {supported_models} or 'custom'."
                )
            self.model_config = (
                default_model_config[model]
                if model_config is None
                else merge_dicts(default_model_config[model], model_config)
            )

            if not self.model_config:
                raise ValueError("Invalid custom model configuration.")
        self.model_name = model
        if self.model_config["loglik_kind"] == "approx_differentiable":
            if loglik:
                try:
                    self.model_config["loglik"] = download_hf(loglik)  # type: ignore
                except Exception as e:
                    print(
                        f"Failed to download the model with name '{loglik}'. Error: {e}. Using the model name as is."
                    )
                    self.model_config["loglik"] = loglik
            else:
                self.model_config["loglik"] = download_hf(self.model_config["loglik"])  # type: ignore
        elif loglik and self.model_config["loglik_kind"] == "analytical":
            self.model_config["loglik"] = loglik

        self.list_params = self.model_config["list_params"]
        self._parent = self.list_params[0]

        if include is None:
            include = []
        params_in_include = [param["name"] for param in include]

        other_kwargs: dict[Any, Any] = {}
        for k, v in kwargs.items():
            if k in self.list_params:
                if k in params_in_include:
                    raise ValueError(
                        f'Parameter "{k}" is already specified in `include`.'
                    )

                if isinstance(v, (int, float, bmb.Prior)):
                    include.append({"name": k, "prior": v})
                elif isinstance(v, dict):
                    include.append(v | {"name": k})
                else:
                    raise ValueError(
                        f"Parameter {k} must be a float, a dict, or a bmb.Prior object."
                    )
            else:
                other_kwargs |= {k: v}

        self.params, self.formula, self.priors, self.link = self._transform_params(
            include, self.model_name, self.model_config
        )

        for param in self.params:
            if param.name == self._parent:
                self._parent_param = param
                break

        assert self._parent_param is not None

        params_is_reg = [param.is_regression for param in self.params]

        if "loglik_kind" not in self.model_config or self.model_config[
            "loglik_kind"
        ] not in [
            "analytical",
            "approx_differentiable",
            "blackbox",
        ]:
            raise ValueError(
                "'loglike_kind' field of model_config must be one of "
                + '"analytical", "approx_differentiable", "blackbox".'
            )

        if (
            "loglik" in self.model_config
            and self.model_config["loglik_kind"] != "approx_differentiable"
        ):
            # If a user has already provided a log-likelihood function
            if issubclass(self.model_config["loglik"], pm.Distribution):
                # Test if the it is a distribution
                self.model_distribution = self.model_config["loglik"]
            else:
                # If not, create a distribution
                self.model_distribution = wfpt.make_distribution(
                    loglik=loglik, list_params=self.list_params  # type: ignore
                )
        else:
            # If not, in the case of "approx_differentiable"
            if self.model_config["loglik_kind"] == "approx_differentiable":
                # Check if a loglik is provided.
                if (
                    "loglik" not in self.model_config
                    or self.model_config["loglik"] is None
                ):
                    raise ValueError(
                        "Please provide either a path to an onnx file for the log "
                        + "likelihood or a log-likelihood function."
                    )
                self.model_distribution = wfpt.make_lan_distribution(
                    model=self.model_config["loglik"],
                    list_params=self.list_params,
                    backend=self.model_config["backend"],
                    params_is_reg=params_is_reg,
                )
            else:
                raise ValueError(
                    "Please provide a likelihood function or a pm.Distribution "
                    + "in the `loglik` field of model_config!"
                )

        assert self.model_distribution is not None

        self.likelihood = bmb.Likelihood(
            self.model_config["loglik_kind"],
            params=self.list_params,
            parent=self.model_config["list_params"][0],
            dist=self.model_distribution,
        )

        self.family = bmb.Family(
            self.model_config["loglik_kind"], likelihood=self.likelihood, link=self.link
        )

        self.model = bmb.Model(
            self.formula, data, family=self.family, priors=self.priors, **other_kwargs
        )

        self._aliases = get_alias_dict(self.model, self._parent_param)
        self.set_alias(self._aliases)

    def _transform_params(
        self, include: Optional[list[dict]], model: str, model_config: Config
    ) -> tuple[
        list[Param],
        bmb.Formula,
        Optional[Dict],
        Union[Dict[str, Union[str, bmb.Link]], str],
    ]:
        """
        Function Name:
            _transform_params

        Purpose:
            Transforms a list of dictionaries containing parameter information into a list of Param objects.
            Also, it generates a formula, priors, and a link for the Bambi package based on the parameters.

        Args:
            include (list[dict] | None): A list of dictionaries containing information about the parameters.
            model (str): A string indicating the model type.
            model_config (Config): A dict containing the configuration for the model.

        Returns:
            tuple: A tuple of four items, with the last three for creating the Bambi model:
            - A list (of the same length as self.list_params) containing Param objects.
            - A bmb.formula object.
            - An optional dict containing prior information for Bambi.
            - An optional dict of link functions for Bambi.

        Raises:
            TypeError: If the input parameters are not in the correct format.
            ValueError: If an unsupported model type is passed.
        """
        processed = []
        params = []
        if include:
            for param_dict in include:
                processed.append(param_dict["name"])
                is_parent = param_dict["name"] == self._parent
                param = Param(
                    bounds=model_config["bounds"][param_dict["name"]],
                    is_parent=is_parent,
                    **param_dict,
                )
                params.append(param)

        for param_str in self.list_params:
            if param_str not in processed:
                is_parent = param_str == self._parent
                bounds = model_config["bounds"][param_str]
                prior = 0.0 if model == "ddm" and param_str == "sv" else None
                param = Param(
                    name=param_str,
                    prior=prior,
                    bounds=bounds,
                    is_parent=is_parent,
                )
                params.append(param)

        if len(params) != len(self.list_params):
            raise ValueError("Please provide a correct set of priors")

        return params, *_parse_bambi(params)

    def sample(
        self,
        sampler: Literal[
            "mcmc", "nuts_numpyro", "nuts_blackjax", "laplace", "vi"
        ] = "mcmc",
        **kwargs: Any,
    ) -> Union[az.InferenceData, pm.Approximation]:
        """
        Function Name:
            sample

        Purpose:
            Executes the sampling process using the 'fit' method through bambi.Model.

        Args:
            sampler (Literal): The sampler to use. Options include "mcmc" (default), "nuts_numpyro", "nuts_blackjax", "laplace", or "vi".
            kwargs: Additional arguments passed to bmb.Model.fit().

        Returns:
            az.InferenceData | pm.Approximation: An ArviZ `InferenceData` instance if the inference_method is "mcmc" (default),
            "nuts_numpyro", "nuts_blackjax", or "laplace". An `Approximation` object if "vi".

        Raises:
            TypeError: If the 'sampler' argument is not a recognized string.
            ValueError: If the 'kwargs' do not match the expected arguments for bmb.Model.fit().
        """

        supported_samplers = ["mcmc", "nuts_numpyro", "nuts_blackjax", "laplace", "vi"]

        if sampler not in supported_samplers:
            raise ValueError(
                f"Unsupported sampler '{sampler}', must be one of {supported_samplers}"
            )

        self._inference_obj = self.model.fit(inference_method=sampler, **kwargs)

        return self.traces

    @property
    def pymc_model(self) -> pm.Model:
        """
        Function Name:
            pymc_model

        Purpose:
            Provides a convenience function that fetches the PyMC model constructed by bambi, primarily to bypass verbose calls like self.model.backend.model...

        Returns:
            pm.Model: The PyMC model established by bambi.

        Raises:
            AttributeError: If the bambi model or the PyMC model doesn't exist.
        """

        return self.model.backend.model

    def set_alias(self, aliases: Dict[str, Union[str, Dict]]):
        """
        Function Name:
            set_alias

        Purpose:
            Assigns aliases based on the provided dictionary and reconstructs the model.

        Args:
            alias (dict): A dictionary specifying the parameter names being aliased and the respective aliases.

        Raises:
            TypeError: If the provided argument is not a dictionary.
            KeyError: If a parameter to be aliased doesn't exist in the model.
        """

        self.model.set_alias(aliases)
        self.model.build()

    # NOTE: can't annotate return type because the graphviz dependency is optional
    def graph(self, formatting="plain", name=None, figsize=None, dpi=300, fmt="png"):
        """
        Function Name:
            graph

        Purpose:
            Generates a graphviz Digraph from a constructed HSSM model. This function relies on the Graphviz package,
            which you can install using ``conda install -c conda-forge python-graphviz`` or manually installing the
            ``graphviz`` binaries and using ``pip install graphviz`` for the Python bindings.

        Args:
            formatting (str): Defines the type of formatting to be applied. Can either be ``"plain"`` or
                ``"plain_with_params"``. Defaults to ``"plain"``.
            name (str): The name of the saved figure. If set to ``None`` (default), no figure is saved.
            figsize (tuple): Maximum width and height of the figure in inches. If specified, and if the generated
                drawing exceeds this size, it's uniformly scaled down to fit. Only applicable if ``name`` is specified.
                Defaults to ``None``.
            dpi (int): Defines the resolution of the saved figure in dots per inch. Only applicable if ``name`` is specified.
                Defaults to 300.
            fmt (str): Specifies the format of the saved figure. Only applicable if ``name`` is specified.
                Defaults to ``"png"``.

        Returns:
            A graphviz Digraph object.

        Raises:
            ImportError: If the graphviz package is not installed.
        """
        self.model._check_built()

        graphviz = HSSMModelGraph(
            model=self.pymc_model, parent=self._parent_param
        ).make_graph(formatting=formatting)

        width, height = (None, None) if figsize is None else figsize

        if name is not None:
            graphviz_ = graphviz.copy()
            graphviz_.graph_attr.update(size=f"{width},{height}!")
            graphviz_.graph_attr.update(dpi=str(dpi))
            graphviz_.render(filename=name, format=fmt, cleanup=True)

            return graphviz_

        return graphviz

    def __repr__(self) -> str:
        """
        Function Name:
            __repr__

        Purpose:
            Creates a representation of the model.

        Returns:
            str: A string representation of the model.
        """

        output = []

        output.append("Hierarchical Sequential Sampling Model")
        output.append(f"Model: {self.model_name}")
        output.append("")

        output.append("Response variable: rt,response")
        output.append(f"Observations: {len(self.data)}")
        output.append("")

        output.append("Parameters:")
        output.append("")

        for param in self.params:
            output.append(str(param))

        return "\r\n".join(output)

    def __str__(self) -> str:
        """
        Function Name:
            __str__

        Purpose:
            Creates a string representation of the model.

        Returns:
            str: A string representation of the model.
        """

        return self.__repr__()

    @property
    def traces(self) -> Union[az.InferenceData, pm.Approximation]:
        """
        Function Name:
            traces

        Purpose:
            Provides the trace of the model after sampling has been conducted.

        Returns:
            az.InferenceData | pm.Approximation: A trace of the model post-sampling.

        Raises:
            ValueError: If the model hasn't undergone sampling yet.
        """

        if not self._inference_obj:
            raise ValueError("Please sample the model first.")

        return self._inference_obj
