"""Inspect the likelihood approximation network (LAN) behind an HSSM model.

The plots come from LANfactory's ``network_inspectors``; this module feeds them
a predictor built from the ONNX network the model itself loads, so the network
inspected is the one the model is fitted with. LANfactory depends on torch and
is an optional extra (``hssm[inspect]``), imported only when a plot is made.

LANfactory's predictor contract: it receives a float32 array of shape
``(N, n_params + 2)`` whose rows are ``[params..., rt, choice]`` and returns an
array whose first column holds the log-likelihood of each row. HSSM's ONNX
networks take one trial at a time, so the rows are batched with ``jax.vmap``,
as HSSM's own likelihoods are.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

import jax
import numpy as np
import onnx
import pandas as pd
from ssms.config import ModelConfigBuilder

from .. import tracking
from ..distribution_utils.onnx_utils.model import load_onnx_model
from ..distribution_utils.onnx_utils.onnx2jax import make_jax_func

if TYPE_CHECKING:
    from types import ModuleType

    import plotly.graph_objects as go
    from lanfactory.network_inspectors import GridSpec, PlotConfig

    from ..base import HSSMBase

INSPECT_IMPORT_ERROR = (
    "LANfactory is required for network inspection. Install it with "
    '`uv add "hssm[inspect]"` (or `pip install "hssm[inspect]"`).'
)
LAN_LOGLIK_KIND = "approx_differentiable"
# Each network input row is the parameters followed by these data columns.
DATA_COLUMNS = ("rt", "choice")
# Parameters HSSM adds to a model that are not inputs of its network.
NON_NETWORK_PARAMS = frozenset({"p_outlier"})
# LANfactory builds the (rt, choice) grid of a two-choice network with these
# choices, whatever choices the model declares.
TWO_CHOICE_GRID_CHOICES = (-1, 1)

Predictor = Callable[[np.ndarray], np.ndarray]


def _import_network_inspectors() -> ModuleType:
    """Import LANfactory's network inspectors, naming the extra if missing."""
    try:
        from lanfactory import network_inspectors
    except ImportError as exc:
        raise ImportError(INSPECT_IMPORT_ERROR) from exc
    return network_inspectors


def network_source(model: HSSMBase) -> str | os.PathLike | onnx.ModelProto:
    """Return the ONNX network a model's likelihood was built from.

    Parameters
    ----------
    model
        An HSSM model.

    Returns
    -------
    str | os.PathLike | onnx.ModelProto
        The network file (a local path or a file in HSSM's HuggingFace
        repository), or the loaded ONNX model it was given.

    Raises
    ------
    ValueError
        If the model's likelihood is not an ONNX network: an analytical or
        blackbox likelihood, or an approximate likelihood given as an already
        built callable or Op.
    """
    config = model.model_config
    if config.loglik_kind != LAN_LOGLIK_KIND:
        raise ValueError(
            "Network inspection needs a model whose likelihood is a network "
            f"(loglik_kind='{LAN_LOGLIK_KIND}'); this model's loglik_kind is "
            f"'{config.loglik_kind}'."
        )
    loglik = config.loglik
    if not isinstance(loglik, (str, os.PathLike, onnx.ModelProto)):
        raise ValueError(
            "Network inspection needs the model's ONNX network, but its "
            f"likelihood was given as a {type(loglik).__name__}. Build the model "
            "with `loglik` set to an ONNX file path or a HuggingFace file name."
        )
    # mypy keeps `type[Any]` (from LogLik's `type[Distribution]`) through the
    # isinstance check above, because ModelProto's metaclass makes a class
    # object a possible instance.
    return cast("str | os.PathLike | onnx.ModelProto", loglik)


def network_params(model: HSSMBase) -> list[str]:
    """Return the network's parameter names, checked against ssm-simulators.

    LANfactory labels the network inputs, and simulates the data it compares
    the network with, from the ssm-simulators model of the same name. The
    model's parameters and choices must therefore match that model's, in the
    same order, or the plots would show a different network than they claim.

    Parameters
    ----------
    model
        An HSSM model.

    Returns
    -------
    list[str]
        The parameter names in network input order.

    Raises
    ------
    ValueError
        If ssm-simulators has no model named like ``model``, if its
        parameters or choices differ from the model's, or if a two-choice
        model's choices are not the -1 and 1 LANfactory evaluates.
    """
    name = model.model_name
    try:
        ssms_config = ModelConfigBuilder.from_model(name)
    except ValueError as exc:
        raise ValueError(
            f"Network inspection needs an ssm-simulators model named '{name}', "
            "which LANfactory simulates to compare the network with, but "
            "ssm-simulators has none."
        ) from exc

    ssms_params = list(ssms_config["params"])
    hssm_params = [
        param
        for param in model.model_config.list_params or []
        if param not in NON_NETWORK_PARAMS
    ]
    if hssm_params != ssms_params:
        raise ValueError(
            f"The parameters of model '{name}' ({hssm_params}) do not match the "
            f"ssm-simulators model of that name ({ssms_params}), which LANfactory "
            "uses to label the network inputs."
        )

    ssms_choices = [int(choice) for choice in ssms_config["choices"]]
    hssm_choices = [int(choice) for choice in model.choices or []]
    if hssm_choices != ssms_choices:
        raise ValueError(
            f"The choices of model '{name}' ({hssm_choices}) do not match the "
            f"ssm-simulators model of that name ({ssms_choices}), which LANfactory "
            "simulates to compare the network with."
        )
    if len(hssm_choices) == 2 and hssm_choices != list(TWO_CHOICE_GRID_CHOICES):
        raise ValueError(
            f"Model '{name}' has choices {hssm_choices}, but LANfactory evaluates "
            "a two-choice network at choices "
            f"{list(TWO_CHOICE_GRID_CHOICES)}, so it cannot inspect this network."
        )
    return ssms_params


def load_model_network(model: HSSMBase) -> onnx.ModelProto:
    """Load the ONNX network of a model, at the revision the model used.

    A network fetched from HuggingFace is fetched again at the commit recorded
    when the model was built, so a network updated on the hub since then does
    not stand in for the model's own.

    Parameters
    ----------
    model
        An HSSM model with an ONNX likelihood.

    Returns
    -------
    onnx.ModelProto
        The loaded network.
    """
    source = network_source(model)
    record = getattr(model, "_tracking_network", None) or {}
    revision = record.get("hf_revision")
    pinned = (
        tracking.pinned_revisions({source: revision})
        if isinstance(source, str) and revision and record.get("network_file") == source
        else contextlib.nullcontext()
    )
    with pinned:
        return load_onnx_model(source)


def network_input_size(network: onnx.ModelProto) -> int:
    """Return the number of values in one network input row.

    HSSM's networks take a single trial, with an input shape such as
    ``(1, n_params + 2)`` or ``(n_params + 2,)``.
    """
    dims = network.graph.input[0].type.tensor_type.shape.dim
    return int(np.prod([dim.dim_value for dim in dims]))


def make_network_predictor(network: onnx.ModelProto) -> Predictor:
    """Make a LANfactory predictor from a single-trial ONNX network.

    Parameters
    ----------
    network
        A network that takes one ``[params..., rt, choice]`` row and returns
        its log-likelihood.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        A function from an ``(N, n_params + 2)`` array of rows to an
        ``(N, 1)`` array of their log-likelihoods.
    """
    n_inputs = network_input_size(network)
    # The graph is traced at its single-trial shape; feeding it a matrix would
    # silently give wrong values for some networks, so batch over rows.
    batched = jax.jit(jax.vmap(make_jax_func(network)))

    def predict(inputs: np.ndarray) -> np.ndarray:
        rows = np.asarray(inputs)
        if rows.ndim != 2 or rows.shape[1] != n_inputs:
            raise ValueError(
                f"Expected an array of shape (N, {n_inputs}), got {rows.shape}."
            )
        return np.asarray(batched(rows)).reshape(rows.shape[0], -1)

    return predict


def make_model_predictor(model: HSSMBase) -> Predictor:
    """Make a LANfactory predictor from the network of an HSSM model.

    Parameters
    ----------
    model
        An HSSM model with an ONNX likelihood.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        See :func:`make_network_predictor`.

    Raises
    ------
    ValueError
        If the model has no ONNX network, does not match its ssm-simulators
        model, or its network does not take its parameters plus rt and choice.
    """
    params = network_params(model)
    network = load_model_network(model)
    n_inputs = network_input_size(network)
    expected = len(params) + len(DATA_COLUMNS)
    if n_inputs != expected:
        raise ValueError(
            f"The network of model '{model.model_name}' takes {n_inputs} inputs, "
            f"but its {len(params)} parameters plus rt and choice make {expected}."
        )
    return make_network_predictor(network)


def as_parameter_df(
    parameters: pd.DataFrame | Mapping[str, float],
) -> pd.DataFrame:
    """Return parameter sets as a DataFrame with one set per row.

    A mapping of parameter names to values is a single set.
    """
    if isinstance(parameters, pd.DataFrame):
        return parameters
    return pd.DataFrame([dict(parameters)])


def manifold_parameters(
    model_name: str,
    parameters: pd.DataFrame | Mapping[str, float] | None,
) -> pd.DataFrame:
    """Return the parameter set a manifold sweep starts from.

    Parameters
    ----------
    model_name
        The ssm-simulators model name.
    parameters
        A DataFrame (its first row is used) is returned as given. A mapping
        sets some or all parameters; the others, and all of them when
        ``parameters`` is None, take the ssm-simulators default values.

    Returns
    -------
    pd.DataFrame
        A one-row DataFrame for a mapping or None.

    Raises
    ------
    ValueError
        If the mapping names a parameter the model does not have.
    """
    if isinstance(parameters, pd.DataFrame):
        return parameters
    ssms_config = ModelConfigBuilder.from_model(model_name)
    defaults = dict(
        zip(ssms_config["params"], np.asarray(ssms_config["default_params"]).tolist())
    )
    given = dict(parameters or {})
    unknown = sorted(set(given) - set(defaults))
    if unknown:
        raise ValueError(
            f"Model '{model_name}' has no parameter(s) {unknown}; its parameters "
            f"are {list(defaults)}."
        )
    return pd.DataFrame([defaults | given])


def plot_kde_vs_lan(
    model: HSSMBase,
    parameter_df: pd.DataFrame | Mapping[str, float],
    n_samples: int = 10,
    n_reps: int = 10,
    grid: GridSpec | None = None,
    plot: PlotConfig | None = None,
) -> None:
    """Compare a model's network likelihood with KDEs of simulated data.

    For each parameter set, plots the likelihood the model's network gives over
    a grid of response times and choices against kernel density estimates of
    data simulated with ssm-simulators. Wraps
    ``lanfactory.network_inspectors.kde_vs_lan_likelihoods`` and needs the
    ``inspect`` extra (``hssm[inspect]``).

    Parameters
    ----------
    model
        An HSSM model with ``loglik_kind="approx_differentiable"`` whose
        likelihood is an ONNX network.
    parameter_df
        One parameter set per row, with a column per model parameter, or a
        mapping of parameter names to values for a single set.
    n_samples
        Number of simulated trials per KDE.
    n_reps
        Number of KDEs per parameter set.
    grid
        A ``lanfactory.network_inspectors.GridSpec`` for the response-time
        grid; LANfactory's default if None.
    plot
        A ``lanfactory.network_inspectors.PlotConfig``. Its ``show`` displays
        the figure and ``save`` writes it to ``save_dir``; LANfactory closes the
        figure afterwards, so it is not returned.

    Raises
    ------
    ImportError
        If LANfactory is not installed.
    ValueError
        If the model has no ONNX network or does not match its ssm-simulators
        model.
    """
    inspectors = _import_network_inspectors()
    inspectors.kde_vs_lan_likelihoods(
        as_parameter_df(parameter_df),
        model.model_name,
        make_model_predictor(model),
        n_samples=n_samples,
        n_reps=n_reps,
        grid=grid,
        plot=plot,
    )


def plot_lan_manifold(
    model: HSSMBase,
    vary: Mapping[str, Any] | None = None,
    parameters: pd.DataFrame | Mapping[str, float] | None = None,
    grid: GridSpec | None = None,
    plot: PlotConfig | None = None,
) -> go.Figure:
    """Plot a model's network likelihood as a surface while one parameter varies.

    Draws the likelihood over response time and choice for each value of one
    parameter, as a 3D plotly surface. Wraps
    ``lanfactory.network_inspectors.lan_manifold`` and needs the ``inspect``
    extra (``hssm[inspect]``). Only two-choice models are supported.

    Parameters
    ----------
    model
        An HSSM model with ``loglik_kind="approx_differentiable"`` whose
        likelihood is an ONNX network.
    vary
        A one-entry mapping of the parameter to vary to its values, e.g.
        ``{"v": [-1.0, 0.0, 1.0]}``. LANfactory's default sweep of ``v`` if
        None.
    parameters
        The values of the other parameters. A mapping may set only some of
        them; the rest take the ssm-simulators defaults, as all do if None. A
        DataFrame must have every parameter, and its first row is used.
    grid
        A ``lanfactory.network_inspectors.GridSpec`` for the response-time
        grid; LANfactory's default if None.
    plot
        A ``lanfactory.network_inspectors.PlotConfig``. Its ``show`` displays
        the figure and ``save`` writes it to ``save_dir``.

    Returns
    -------
    plotly.graph_objects.Figure
        The manifold figure.

    Raises
    ------
    ImportError
        If LANfactory is not installed.
    ValueError
        If the model has no ONNX network, does not match its ssm-simulators
        model, or ``parameters`` names a parameter the model does not have.
    """
    inspectors = _import_network_inspectors()
    predictor = make_model_predictor(model)
    return inspectors.lan_manifold(
        manifold_parameters(model.model_name, parameters),
        vary_dict=None if vary is None else dict(vary),
        model=model.model_name,
        torch_mlp_predict=predictor,
        grid=grid,
        plot=plot,
    )
