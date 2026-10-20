"""Choice-only (softmax inverse temperature) lapse (#925)."""

import numpy as np
import pandas as pd
import pytest

import hssm

hssm.set_floatX("float32", update_jax=True)


def _model(n_choices, **kwargs):
    choices = [-1, 1] if n_choices == 2 else list(range(n_choices))
    data = pd.DataFrame({"response": np.random.default_rng(0).choice(choices, size=50)})
    return hssm.HSSM(data=data, model=f"softmax_inv_temperature_{n_choices}", **kwargs)


def test_lapse_logp_is_a_mixture_of_probabilities():
    """logp = log((1 - p) * softmax + p / n_choices), never above 0."""
    model = _model(2, p_outlier=0.5)
    point = model.pymc_model.initial_point()

    logp = model.pymc_model.compile_logp(sum=False)(point)[-1]

    # The initial point has logit1 = 0, so the softmax is 1/2 for each choice.
    np.testing.assert_allclose(logp, np.log(0.5), rtol=1e-5)
