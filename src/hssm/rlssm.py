"""RLSSM integration for reinforcement-learning-driven sequential sampling models.

Defines the RLSSM model that pairs learning-process callables with a decision
log-likelihood, builds RL-aware logp functions/ops, and wires optional
missing-data composition for HSSM.
"""

import logging
from copy import deepcopy
from inspect import isclass
from typing import TYPE_CHECKING, Any, Callable, Literal
from typing import cast as typing_cast

import numpy as np
import pymc as pm

from hssm.config import Config, RLSSMConfig
from hssm.defaults import MissingDataNetwork, missing_data_networks_suffix
from hssm.distribution_utils import (
    assemble_callables,
    make_distribution,
    make_likelihood_callable,
    make_missing_data_callable,
)
from hssm.rl.likelihoods.builder import (
    AnnotatedFunction,
    make_rl_logp_func,
    make_rl_logp_op,
)
from hssm.utils import _rearrange_data, annotate_function

from .base import HSSMBase

if TYPE_CHECKING:
    from os import PathLike

    from pytensor.graph.op import Op

_logger = logging.getLogger("hssm")


class RLSSM(HSSMBase):
    """Reinforcement Learning Sequential Sampling Model.

    Tailors HSSM to RL settings where a learning process produces trialwise
    computed variables that feed a decision-process log-likelihood.

    Key expectations
    ----------------
    - data must include at least ``rt``, ``response``, ``participant_id`` and a
      uniform number of ``trial_id`` per participant; any additional columns used
      by learning/decision callables should appear in ``extra_fields``.
    - model_config must be an ``RLSSMConfig`` describing ``list_params`` (RL +
      decision parameters), bounds/defaults, ``learning_process`` mapping
      computed names to annotated callables, and ``extra_fields`` including
      ``participant_id``.
    - loglik is the decision-process logp (callable/Op/pm.Distribution or ONNX
      path). For callables, annotate expected inputs with ``annotate_function``;
      computed outputs from ``learning_process`` must also be listed in
      ``inputs``.
    - loglik_kind typically ``approx_differentiable``; backend defaults to JAX.

    Other arguments (e.g., ``p_outlier``, missing-data settings) follow
    ``HSSMBase`` semantics.
    """

    @classmethod
    def _build_model_config(
        cls,
        model: Any,
        loglik_kind: Any,
        model_config: Any | None,
        choices: list[int] | None,
    ) -> Config:
        """Construct an RLSSMConfig from user input.

        RLSSM requires an explicit RLSSMConfig (or dict) because defaults are
        model-specific and not enumerated like standard SSM models.
        """
        if model_config is None:
            raise ValueError("RLSSM requires an explicit `model_config`.")

        config = (
            model_config
            if isinstance(model_config, RLSSMConfig)
            else RLSSMConfig(**model_config)
        )

        if loglik_kind is not None:
            config.loglik_kind = loglik_kind

        if choices is not None and config.choices is None:
            config.choices = tuple(choices)

        config.validate()
        return typing_cast("Config", config)

    def _make_model_distribution(self) -> type[pm.Distribution]:
        """Build the pm.Distribution backing the RLSSM likelihood."""
        if isclass(self.loglik) and issubclass(self.loglik, pm.Distribution):
            return self.loglik

        assert self.loglik is not None, "loglik should be set"
        assert self.list_params is not None, "list_params should be set"
        rl_config = typing_cast("RLSSMConfig", self.model_config)

        data_cols = list(rl_config.response or ["rt", "response"])
        extra_fields = rl_config.extra_fields or []

        # Trialwise flags: parameters (excluding p_outlier) plus extra_fields
        # (always trialwise)
        params_is_trialwise_base = [
            param.is_trialwise
            for name, param in self.params.items()
            if name != "p_outlier"
        ]
        params_is_trialwise = params_is_trialwise_base + [True for _ in extra_fields]
        rl_list_params = [name for name in self.list_params if name != "p_outlier"]

        # Resolve participant/trial counts for the RL builder
        if "participant_id" not in extra_fields:
            raise ValueError(
                "RLSSM requires `participant_id` in extra_fields for RL builder."
            )
        participant_id_col = "participant_id"
        n_participants = int(self.data[participant_id_col].nunique())
        # Try to infer uniform n_trials
        trials_per_participant = self.data.groupby(participant_id_col).size()
        if trials_per_participant.nunique() != 1:
            raise ValueError(
                "All participants must have the same number of trials for RLSSM."
            )
        n_trials = int(trials_per_participant.iloc[0])

        # Prepare SSM likelihood callable; require inputs metadata for RL builder
        loglik_callable = typing_cast(
            "Op | Callable[..., Any] | PathLike | str", self.loglik
        )
        loglik_inputs = getattr(loglik_callable, "inputs", None)
        loglik_computed = getattr(loglik_callable, "computed", None)

        # Build/extend computed mapping from learning_process first, so we can
        # decide whether the raw callable or a wrapped Op is needed below.
        computed_map: dict = loglik_computed or {}
        if rl_config.learning_process:
            for param_name, compute_func in rl_config.learning_process.items():
                # Fill in outputs default if absent
                func = compute_func
                if not hasattr(func, "outputs") or not getattr(func, "outputs"):
                    func = annotate_function(outputs=[param_name])(func)
                # Require inputs metadata
                if not hasattr(func, "inputs"):
                    raise ValueError(
                        f"Compute function for '{param_name}' must be annotated with"
                        " `inputs`."
                    )
                computed_map[param_name] = func

        has_computed_params = bool(computed_map)

        # When learning_process has compute functions the inner SSM logp is
        # called with concrete numpy/JAX arrays inside Op.perform.  In that
        # context a wrapped JAX Op (as produced by make_likelihood_callable for
        # the "jax" backend) cannot accept JAX arrays as inputs.  Use the raw
        # callable directly; make_rl_logp_op will handle all necessary wrapping.
        # For the no-learning-process case keep the full make_likelihood_callable
        # pipeline unchanged.
        if has_computed_params:
            ssm_loglik = loglik_callable
        else:
            ssm_loglik = make_likelihood_callable(
                loglik=loglik_callable,
                loglik_kind=self.loglik_kind or "approx_differentiable",
                backend=rl_config.backend or "jax",
                params_is_reg=params_is_trialwise,
            )
            # Propagate metadata lost by wrapping (e.g., jax vmapping)
            if loglik_inputs is not None and not hasattr(ssm_loglik, "inputs"):
                setattr(ssm_loglik, "inputs", loglik_inputs)
            if loglik_computed is not None and not hasattr(ssm_loglik, "computed"):
                setattr(ssm_loglik, "computed", loglik_computed)

        # Ensure the decision-process logp exposes required metadata
        if not hasattr(ssm_loglik, "inputs"):
            raise ValueError(
                "RLSSM requires the decision-process log-likelihood to declare `inputs`"
                "(e.g., annotate_function). Please annotate the logp function with "
                "an ordered list of expected columns including any computed parameters."
            )

        setattr(ssm_loglik, "computed", computed_map)

        if not getattr(ssm_loglik, "inputs", None):
            raise ValueError(
                "Decision-process log-likelihood must define a non-empty `inputs` list."
            )

        # Ensure computed params appear in the decision logp inputs
        missing_in_inputs = [
            name
            for name in computed_map
            if name not in ssm_loglik.inputs  # type: ignore[attr-defined]
        ]
        if missing_in_inputs:
            raise ValueError(
                "Computed parameters must be included in the decision log-likelihood"
                f" `inputs`: missing {', '.join(missing_in_inputs)}"
            )

        # Build RL-aware logp.  Wrap in a PyTensor Op when using the explicit
        # pytensor backend OR when learning_process has compute functions.
        # Op wrapping ensures that the JAX compute functions receive concrete
        # numpy arrays (via Op.perform) rather than symbolic PyTensor tensors
        # that would cause jnp.stack/jnp.concatenate to fail.
        annotated_loglik = typing_cast("AnnotatedFunction", ssm_loglik)

        if rl_config.backend == "pytensor" or has_computed_params:
            self.loglik = typing_cast(
                "Op",
                make_rl_logp_op(
                    annotated_loglik,
                    n_participants,
                    n_trials,
                    data_cols=data_cols,
                    list_params=rl_list_params,
                    extra_fields=extra_fields,
                    backend=rl_config.backend,
                ),
            )
        else:
            self.loglik = typing_cast(
                "AnnotatedFunction",
                make_rl_logp_func(
                    annotated_loglik,
                    n_participants,
                    n_trials,
                    data_cols=data_cols,
                    list_params=rl_list_params,
                    extra_fields=extra_fields,
                ),
            )

        # Optional missing-data / deadline composition (reuses HSSM logic)
        if self.missing_data_network != MissingDataNetwork.NONE:
            params_only: bool | None
            if self.missing_data_network == MissingDataNetwork.OPN:
                params_only = False
            elif self.missing_data_network == MissingDataNetwork.CPN:
                params_only = True
            else:
                params_only = None

            if self.loglik_missing_data is None:
                self.loglik_missing_data = (
                    self.model_name
                    + missing_data_networks_suffix[self.missing_data_network]
                    + ".onnx"
                )

            backend_tmp: Literal["pytensor", "jax", "other"] | None = (
                "jax" if rl_config.backend != "pytensor" else rl_config.backend
            )
            missing_data_callable = make_missing_data_callable(
                self.loglik_missing_data, backend_tmp, params_is_trialwise, params_only
            )

            self.loglik_missing_data = missing_data_callable

            self.loglik = assemble_callables(
                self.loglik,
                self.loglik_missing_data,
                params_only,
                has_deadline=self.deadline,
                params_is_trialwise=params_is_trialwise,
            )

        if self.missing_data:
            _logger.info(
                "Re-arranging data to separate missing and observed datapoints. "
                "Missing data (rt == %s) will be on top, observed datapoints follow.",
                self.missing_data_value,
            )

        self.data = _rearrange_data(self.data)

        fixed_vector_params = {
            name: param.prior
            for name, param in self.params.items()
            if isinstance(param.prior, np.ndarray)
        }

        extra_fields_data = (
            None
            if not extra_fields
            else [deepcopy(self.data[field].values) for field in extra_fields]
        )

        rv_spec = getattr(rl_config, "rv", None)
        return make_distribution(
            rv=rv_spec or self.model_name,
            loglik=self.loglik,
            list_params=self.list_params,
            bounds=self.bounds,
            lapse=self.lapse,
            extra_fields=extra_fields_data,
            fixed_vector_params=fixed_vector_params or None,
            params_is_trialwise=params_is_trialwise_base,
        )
