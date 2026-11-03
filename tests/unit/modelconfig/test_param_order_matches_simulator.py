"""`list_params` order must be the order every positional consumer expects.

HSSM passes model parameters positionally, in `list_params` order, to two
consumers that do not check names:

* the log-likelihood, called as `loglik(data, *dist_params)`;
* the ssm-simulators random variable behind `sample_prior_predictive` and
  `sample_posterior_predictive`, which stacks the same values into `theta` and
  reads its columns in the order of its own registry entry.

A permutation therefore raises nothing. If `list_params` matches the likelihood
but not the registry, inference is right while every predictive draw comes from
permuted parameters, so a well-fitting model fails its predictive checks.
ssm-simulators is the authority for the generative order; analytical
likelihoods must follow it in their signatures.
"""

import inspect
from typing import get_args

import numpy as np
import pymc as pm
import ssms

import hssm
from hssm.defaults import SupportedModels
from hssm.likelihoods.analytical import RDM3
from hssm.modelconfig import get_default_model_config


def _registry_mismatches():
    """Models whose declared params or choices differ from the simulator's."""
    found = []
    for model in get_args(SupportedModels):
        config = get_default_model_config(model)
        for kind, likelihood in config["likelihoods"].items():
            # The simulator is `rv` when the config declares one, else the
            # model name (hssm.py resolves it the same way).
            rv = likelihood.get("rv")
            simulator = rv if isinstance(rv, str) else model
            if simulator not in ssms.config.model_config:
                # No ssm-simulators model to compare against, for example
                # softmax_inv_temperature_2 and softmax_inv_temperature_3.
                continue
            registry = ssms.config.model_config[simulator]
            for field, declared, expected in (
                ("params", config["list_params"], registry["params"]),
                ("choices", config["choices"], registry["choices"]),
            ):
                if list(declared) != list(expected):
                    found.append((model, kind, field, list(declared), list(expected)))
    return found


def test_list_params_match_the_simulator_registry():
    """Declared params and choices follow the ssm-simulators registry."""
    mismatched = _registry_mismatches()
    assert not mismatched, (
        "declared order differs from the ssm-simulators registry "
        "(model, likelihood kind, field, hssm, ssms):\n"
        + "\n".join(f"  {row}" for row in mismatched)
    )


def test_python_loglik_signatures_follow_list_params():
    """Named Python likelihoods declare their parameters in list_params order."""
    # The likelihood receives `*dist_params` in `list_params` order, so a
    # named Python likelihood must declare its parameters in that order too.
    # Callables taking `*args` (blackbox wrappers, softmax logits) and
    # non-callable likelihoods (ONNX files) carry no names to compare.
    mismatched = []
    for model in get_args(SupportedModels):
        config = get_default_model_config(model)
        list_params = list(config["list_params"])
        for kind, likelihood in config["likelihoods"].items():
            loglik = likelihood.get("loglik")
            if not inspect.isfunction(loglik):
                continue
            parameters = list(inspect.signature(loglik).parameters.values())[1:]
            if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters):
                continue
            names = [p.name for p in parameters[: len(list_params)]]
            if names != list_params:
                mismatched.append((model, kind, names, list_params))
    assert not mismatched, "\n".join(f"  {row}" for row in mismatched)


def test_rdm3_predictive_draws_match_the_simulator():
    """racing_diffusion_3 predictive draws match the simulator called by name."""
    # End to end on the model that drifted: draws from the distribution's RV
    # (the predictive sampling path) must match the simulator called by name
    # at the same named parameters. While the simulator read the values sent
    # as (A, b, v0, v1, v2) as (v0, v1, v2, A, b), the choice-0 share fell
    # from 0.71 to 0.27 and the mean RT from 0.88 s to 0.34 s for these values.
    seed = 20261010
    n_samples = 5000
    theta = dict(v0=2.0, v1=0.5, v2=0.5, A=0.2, b=1.5, t=0.3)

    reference = hssm.simulate_data(
        model="racing_diffusion_3", theta=theta, size=n_samples, random_state=seed
    )
    draws = np.asarray(pm.draw(RDM3.dist(**theta, size=n_samples), random_seed=seed))

    # Tolerances are at least four Monte Carlo standard errors of a difference
    # between two independent samples of this size (SE about 0.009 for a share
    # near 0.7, about 0.006 s for the mean RT); the permutation moves both
    # statistics by more than 0.4.
    for choice in (0, 1, 2):
        share_rv = np.mean(draws[:, 1] == choice)
        share_ref = np.mean(reference["response"].to_numpy() == choice)
        assert abs(share_rv - share_ref) < 0.04, (choice, share_rv, share_ref)
    assert abs(draws[:, 0].mean() - reference["rt"].mean()) < 0.03
