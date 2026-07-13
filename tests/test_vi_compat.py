"""Tests for the scoped PyMC VI compatibility shims (hssm._vi_compat)."""

import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytensor
from pymc.variational import approximations

from hssm._vi_compat import coerce_approx_params_to_numpy, static_shape_vi_params


def _tiny_model():
    rng = np.random.default_rng(0)
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("y", mu, sigma, observed=rng.normal(size=20))
    return model


def test_static_shape_vi_params_makes_params_static():
    model = _tiny_model()
    with static_shape_vi_params():
        with model:
            approx = pm.MeanField()
        shapes = [p.type.shape for p in approx.params]
    assert all(None not in s for s in shapes), shapes


def test_static_shape_vi_params_restores_originals():
    originals = {
        cls: cls.create_shared_params
        for cls in [
            approximations.MeanFieldGroup,
            approximations.FullRankGroup,
            approximations.EmpiricalGroup,
        ]
    }
    with static_shape_vi_params():
        assert all(
            cls.create_shared_params is not orig for cls, orig in originals.items()
        )
    assert all(cls.create_shared_params is orig for cls, orig in originals.items())


def test_static_shape_vi_params_restores_on_error():
    orig = approximations.MeanFieldGroup.create_shared_params
    try:
        with static_shape_vi_params():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert approximations.MeanFieldGroup.create_shared_params is orig


def test_coerce_approx_params_to_numpy():
    shared = pytensor.shared(np.zeros(3), "p")
    shared.container.storage[0] = jnp.ones(3)

    class FakeApprox:
        params = [shared]

    coerce_approx_params_to_numpy(FakeApprox())
    assert isinstance(shared.container.storage[0], np.ndarray)
    np.testing.assert_array_equal(shared.container.storage[0], np.ones(3))

    # Idempotent no-op on numpy storage.
    coerce_approx_params_to_numpy(FakeApprox())
    assert isinstance(shared.container.storage[0], np.ndarray)
