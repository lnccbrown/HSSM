"""Tests for the scoped PyMC VI compatibility shims (hssm._vi_compat)."""

import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from pymc.variational import approximations
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Constant

from hssm._vi_compat import (
    coerce_approx_params_to_numpy,
    freeze_shared_data,
    static_shape_vi_params,
)


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


def _data_model(n=20):
    """A model whose graph derives a shape from shared data and `__obs__`.

    Mirrors bambi 0.20 (data as ``pm.Data``, a shared ``__obs__`` length) and
    HSSM's missing-data logp (a slice length computed from the data values).
    """
    rng = np.random.default_rng(0)
    y_obs = rng.normal(size=n)
    with pm.Model(coords={"__obs__": range(n)}) as model:
        y = pm.Data("y", y_obs, dims="__obs__")
        mu = pm.Normal("mu", 0, 1)
        n_pos = pt.sum(y > 0).astype("int64")
        mu_obs = pt.broadcast_to(mu, (model.dim_lengths["__obs__"],))
        pm.Normal("y_pos", mu_obs[n_pos:], 1, observed=y[n_pos:])
    return model


def test_freeze_shared_data_covers_data_and_dim_lengths():
    model = _data_model()
    replacements = freeze_shared_data(model)
    frozen = {var.name for var in replacements}
    assert frozen == {"y", "__obs__"}
    assert all(isinstance(var, SharedVariable) for var in replacements)
    assert all(isinstance(const, Constant) for const in replacements.values())
    np.testing.assert_array_equal(replacements[model["y"]].data, model["y"].get_value())
    assert replacements[model.dim_lengths["__obs__"]].data == 20


def test_freeze_shared_data_makes_jax_fit_traceable():
    """The frozen graph compiles through the JAX linker; the live one does not.

    If the second half starts passing, ``pm.fit(backend="jax")`` freezes
    shared data itself and ``freeze_shared_data`` can be dropped.
    """
    model = _data_model()
    with model, static_shape_vi_params():
        pm.fit(
            n=5,
            method="advi",
            backend="jax",
            progressbar=False,
            more_replacements=freeze_shared_data(model),
        )
        # JAX reports the traced shape as a TypeError (Alloc) or IndexError
        # (slice) depending on which node it reaches first.
        with pytest.raises((TypeError, IndexError), match="depends on the value"):
            pm.fit(n=5, method="advi", backend="jax", progressbar=False)


def test_freeze_shared_data_reflects_set_data():
    model = _data_model()
    with model:
        pm.set_data({"y": np.zeros(7)}, coords={"__obs__": range(7)})
    replacements = freeze_shared_data(model)
    assert replacements[model["y"]].data.shape == (7,)
    assert replacements[model.dim_lengths["__obs__"]].data == 7
