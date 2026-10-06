"""Tests for HSSM distribution utility helpers."""

from unittest.mock import patch

import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest

import hssm
from hssm import distribution_utils
from hssm.config import Config
from hssm.defaults import default_model_config
from hssm.distribution_utils import dist as dist_module
from hssm.distribution_utils.dist import (
    LOGP_LB,
    _apply_lapse_model,
    _create_arg_arrays,
    _extract_size,
    _get_p_outlier,
    _ndt_lower_edge,
    apply_param_bounds_to_loglik,
    ensure_positive_ndt,
    make_distribution,
    make_distribution_for_supported_model,
)
from hssm.likelihoods.analytical import DDM, logp_ddm
from hssm.likelihoods.blackbox import logp_full_ddm
from hssm.register import register_model

hssm.set_floatX("float32")

# Response times for the ndt_edge_shift wiring tests. With t = 0.5 and st = 0.1
# the candidate edges are 0.5 (no declaration), 0.45 (scale 0.5), 0.4 (scale 1.0)
# and 0.2 (scale 3.0); every grid point keeps 0.01 clear of all of them, so
# float32 rounding of the edge cannot move a point across it.
NDT_RT_GRID = np.array([0.1, 0.19, 0.21, 0.3, 0.39, 0.41, 0.49, 0.51, 0.6, 1.0])
NDT_BOUNDS = {
    "v": (-3.0, 3.0),
    "a": (0.3, 2.5),
    "z": (0.1, 0.9),
    "t": (0.0, 2.0),
    "st": (0.0, 1.0),
}


def _flat_logp(data, v, a, z, t, st):
    """Score every response 1.0, so only the guard's floor can change a value."""
    return pt.ones_like(data[:, 0])


def _ndt_grid_data():
    """Return the grid as a (rt, response) array and as an HSSM data frame."""
    response = np.where(np.arange(NDT_RT_GRID.size) % 2 == 0, 1.0, -1.0)
    data = np.column_stack([NDT_RT_GRID, response])
    return data, pd.DataFrame(data, columns=["rt", "response"])


def _floored_at(edge):
    """Return the expected logp on the grid for a floor at ``edge``."""
    return np.where(NDT_RT_GRID <= edge, LOGP_LB, 1.0)


def test_make_hssm_rv():
    """Check that generated HSSM random variables sample deterministically."""
    params = ["v", "a", "z", "t"]
    seed = 42

    # The order of true values, however, is
    # v, a, z, t
    true_values = [0.5, 0.5, 0.5, 0.3]

    wfpt_rv = distribution_utils.make_hssm_rv("ddm", params)
    rng = np.random.default_rng()

    random_sample = wfpt_rv.rng_fn(rng, *true_values, size=500)

    assert random_sample.shape == (500, 2)

    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)

    sequential_sample_1 = np.array(
        [wfpt_rv.rng_fn(rng1, *true_values, size=500) for _ in range(5)]
    )

    sequential_sample_2 = np.array(
        [wfpt_rv.rng_fn(rng2, *true_values, size=500) for _ in range(5)]
    )

    np.testing.assert_array_equal(sequential_sample_1, sequential_sample_2)

    true_values[0] = np.ones(100) * 0.5

    random_sample = wfpt_rv.rng_fn(rng, *true_values, size=100)

    assert random_sample.shape == (100, 2)


def test_lapse_distribution():
    """Check lapse sampling shape, support, and reproducibility."""
    lapse_dist = bmb.Prior("Uniform", lower=0.0, upper=1.0)
    rv = distribution_utils.make_hssm_rv("ddm", ["v", "a", "z", "t"], lapse=lapse_dist)
    random_sample = rv.rng_fn(np.random.default_rng(), *[0.5, 0.5, 0.5, 0.3], 0.05, 10)
    assert random_sample.shape == (10, 2)

    random_sample_1 = rv.rng_fn(
        np.random.default_rng(), np.random.uniform(size=10), *[0.5, 0.5, 0.3], 0.05, 10
    )

    assert random_sample_1.shape == (10, 2)
    assert -1.0 in random_sample_1[:, 1]
    assert 1.0 in random_sample_1[:, 1]
    assert 0 not in random_sample_1[:, 1]

    rng1 = np.random.default_rng(10)
    rng2 = np.random.default_rng(10)

    random_sample_a = rv.rng_fn(rng1, *[0.5, 0.5, 0.5, 0.3], 0.05, 10)
    random_sample_b = rv.rng_fn(rng2, *[0.5, 0.5, 0.5, 0.3], 0.05, 10)

    np.testing.assert_array_equal(random_sample_a, random_sample_b)

    # Test reproducibility
    random_sample_a = rv.rng_fn(rng1, *[0.5, 0.5, 0.5, 0.3], 0.05, 10)
    random_sample_b = rv.rng_fn(rng2, *[0.5, 0.5, 0.5, 0.3], 0.05, 10)

    np.testing.assert_array_equal(random_sample_a, random_sample_b)


def test_apply_lapse_model_rejects_numeric_lapse_distribution():
    """Numeric choice-only lapse values cannot simulate RT lapse samples."""
    sims_out = np.asarray([[0.2, 0.0], [0.3, 1.0]], dtype=float)
    rng = np.random.default_rng(42)

    with pytest.raises(TypeError, match="numeric lapse"):
        _apply_lapse_model(
            sims_out=sims_out,
            p_outlier=0.5,
            rng=rng,
            lapse_dist=0.5,
            choices=[0, 1],
        )


@pytest.mark.parametrize("rv", ["choice_only_model", lambda *args, **kwargs: None])
def test_make_distribution_forwards_choice_only_to_generated_rv(monkeypatch, rv):
    """Generated RVs must keep the choice-only support-shape contract."""
    captured = {}

    def fake_make_hssm_rv(simulator_fun, list_params, lapse=None, is_choice_only=False):
        captured["is_choice_only"] = is_choice_only

        class FakeRV:
            def __call__(self):
                return object()

        return FakeRV

    monkeypatch.setattr(dist_module, "make_hssm_rv", fake_make_hssm_rv)

    make_distribution(
        rv=rv,
        loglik=lambda data, beta: data,
        list_params=["beta"],
        is_choice_only=True,
    )

    assert captured["is_choice_only"] is True


@pytest.mark.slow
def test_apply_param_bounds_to_loglik():
    """Tests the function in separation."""
    logp = np.random.normal(size=1000)

    list_params = ["param1", "param2"]
    bounds = {"param1": [-1.0, 1.0], "param2": [-1.0, 1.0]}

    scalar_in_bound = -0.5
    scalar_out_of_bound = 2.0

    random_vector = np.random.uniform(-3, 3, size=1000)
    out_of_bound_indices = np.logical_or(random_vector <= -1.0, random_vector >= 1.0)

    np.testing.assert_array_equal(
        apply_param_bounds_to_loglik(
            logp, list_params, scalar_in_bound, scalar_in_bound, bounds=bounds
        ).eval(),
        logp,
    )

    np.testing.assert_array_equal(
        apply_param_bounds_to_loglik(
            logp, list_params, scalar_in_bound, scalar_out_of_bound, bounds=bounds
        ).eval(),
        -66.1,
    )

    results_vector = np.asarray(
        apply_param_bounds_to_loglik(
            logp, list_params, scalar_in_bound, random_vector, bounds=bounds
        ).eval(),
    )

    np.testing.assert_array_equal(results_vector[out_of_bound_indices], -66.1)

    np.testing.assert_array_equal(
        results_vector[~out_of_bound_indices], logp[~out_of_bound_indices]
    )


@pytest.mark.slow
def test_make_distribution():
    """Check custom distribution logp values and parameter-bound masking."""

    def fake_logp_function(data, param1, param2):
        """Make up a fake log likelihood function for this test only."""
        return data[:, 0] * param1 * param2

    data = np.zeros((1000, 2))
    data[:, 0] = np.random.normal(size=1000)
    bounds = {"param1": [-1.0, 1.0], "param2": [-1.0, 1.0]}

    Dist = make_distribution(
        rv="fake",
        loglik=fake_logp_function,
        list_params=["param1", "param2"],
        bounds=bounds,
    )

    scalar_in_bound = -0.5
    scalar_out_of_bound = 2.0

    random_vector = np.random.uniform(-3, 3, size=1000)
    out_of_bound_indices = np.logical_or(random_vector <= -1.0, random_vector >= 1.0)

    np.testing.assert_array_equal(
        Dist.logp(data, scalar_in_bound, scalar_in_bound).eval(),
        data[:, 0] * scalar_in_bound * scalar_in_bound,
    )

    np.testing.assert_array_equal(
        Dist.logp(data, scalar_in_bound, scalar_out_of_bound).eval(),
        -66.1,
    )

    results_vector = np.asarray(Dist.logp(data, scalar_in_bound, random_vector).eval())

    np.testing.assert_array_equal(results_vector[out_of_bound_indices], -66.1)

    np.testing.assert_array_equal(
        results_vector[~out_of_bound_indices],
        data[:, 0][~out_of_bound_indices]
        * scalar_in_bound
        * random_vector[~out_of_bound_indices],
    )


def test_make_distribution_floors_at_t_without_declaration():
    """A parameter named st does not move the edge unless a config declares it.

    Pins that the guard infers nothing from parameter names: with no
    ``ndt_edge_shift`` the floor sits at t, exactly as for a model without st.
    """

    def fake_logp_function(data, v, a, z, t, st):
        """Make up a fake log likelihood function for this test only."""
        return np.ones(data.shape[0])

    rt = np.array([0.2, 0.3, 0.31, 0.4, 0.5, 0.6])
    data = np.column_stack([rt, np.ones(rt.size)])

    Dist = make_distribution(
        rv="fake",
        loglik=fake_logp_function,
        list_params=["v", "a", "z", "t", "st"],
    )

    result = np.asarray(Dist.logp(data, 0.5, 0.5, 0.5, 0.5, 0.2).eval())

    np.testing.assert_array_equal(result, [LOGP_LB] * 5 + [1.0])


@pytest.mark.parametrize(
    ("scale", "edge"),
    [(1.0, 0.4), (3.0, 0.2)],
    ids=["scale_1", "scale_3"],
)
def test_make_distribution_applies_declared_ndt_edge_shift(scale, edge):
    """A declared shift moves the floor to t - scale * param, at both call sites.

    The lapse path wraps the floored logp in the outlier mixture, so it is
    compared against the mixture of the expected floor rather than to the floor
    itself. It also evaluates the data to score the lapse density, so it is
    handed a tensor, as pm.logp would.
    """
    data, _ = _ndt_grid_data()
    declaration = {"param": "st", "scale": scale}

    Dist = make_distribution(
        rv="fake",
        loglik=_flat_logp,
        list_params=["v", "a", "z", "t", "st"],
        ndt_edge_shift=declaration,
    )
    result = np.asarray(Dist.logp(data, 0.5, 0.5, 0.5, 0.5, 0.1).eval())
    np.testing.assert_array_equal(result, _floored_at(edge))

    lapse = bmb.Prior("Uniform", lower=0.0, upper=10.0)
    DistLapse = make_distribution(
        rv="fake",
        loglik=_flat_logp,
        list_params=["v", "a", "z", "t", "st"],
        lapse=lapse,
        ndt_edge_shift=declaration,
    )
    result_lapse = np.asarray(
        DistLapse.logp(
            pt.as_tensor_variable(data), 0.5, 0.5, 0.5, 0.5, 0.1, 0.05
        ).eval()
    )
    expected_lapse = np.log(
        0.95 * np.exp(_floored_at(edge)) + 0.05 * np.exp(-np.log(10.0)) + 1e-29
    )
    np.testing.assert_allclose(result_lapse, expected_lapse, rtol=1e-6)


@pytest.mark.slow
def test_make_distribution_for_supported_model():
    """Check supported-model distribution creation and unsupported-model errors."""
    data = np.zeros((10, 2))
    data[:, 0] = np.random.normal(size=10)

    Dist = make_distribution_for_supported_model("ddm")

    np.testing.assert_array_equal(
        Dist.logp(data, 0.5, 1.0, 0.5, 0.3).eval(),
        DDM.logp(data, 0.5, 1.0, 0.5, 0.3).eval(),
    )

    with pytest.raises(ValueError, match="`model` must be one of"):
        make_distribution_for_supported_model("unsupported_model")


@pytest.mark.slow
def test_extra_fields(data_ddm):
    """Check extra likelihood fields are forwarded through generated distributions."""
    ones = np.ones(data_ddm.shape[0])
    x = ones * 0.5
    y = ones * 4.0

    def logp_ddm_extra_fields(data, v, a, z, t, x, y):
        return logp_ddm(data, v, a, z, t) * x * y

    DDM_WITH_XY = make_distribution(
        rv="ddm",
        loglik=logp_ddm_extra_fields,
        list_params=["v", "a", "z", "t"],
        extra_fields=[x, y],
    )

    true_values = dict(v=0.5, a=1.5, z=0.5, t=0.5)

    np.testing.assert_almost_equal(
        pm.logp(DDM.dist(**true_values), data_ddm).eval(),
        pm.logp(DDM_WITH_XY.dist(**true_values), data_ddm).eval() / 2.0,
    )

    data_ddm_copy = data_ddm.copy()
    data_ddm_copy["x"] = x
    data_ddm_copy["y"] = y

    ddm_model_xy = hssm.HSSM(
        data=data_ddm_copy,
        model_config=dict(extra_fields=["x", "y"]),
        loglik=logp_ddm_extra_fields,
        p_outlier=None,
        lapse=None,
    )

    np.testing.assert_almost_equal(
        pm.logp(DDM.dist(**true_values), data_ddm).eval(),
        pm.logp(ddm_model_xy.model_distribution.dist(**true_values), data_ddm).eval()
        / 2.0,
    )

    ddm_model = hssm.HSSM(data=data_ddm)
    ddm_model_p = hssm.HSSM(
        data=data_ddm_copy,
        model_config=dict(extra_fields=["x", "y"]),
        loglik=logp_ddm_extra_fields,
    )
    ddm_model_p_logp_without_lapse = (
        pm.logp(
            ddm_model_p.model_distribution.dist(**true_values, p_outlier=0),
            data_ddm,
        )
        / 2
    )
    ddm_model_p_logp_lapse = pt.log(
        0.95 * pt.exp(ddm_model_p_logp_without_lapse)
        + 0.05
        * pt.exp(pm.logp(pm.Uniform.dist(lower=0.0, upper=20.0), data_ddm["rt"].values))
    )
    np.testing.assert_almost_equal(
        pm.logp(
            ddm_model.model_distribution.dist(**true_values, p_outlier=0.05), data_ddm
        ).eval(),
        ddm_model_p_logp_lapse.eval(),
    )


@pytest.mark.parametrize("edge", [0.5, 0.3])
@pytest.mark.parametrize("trialwise", [False, True], ids=["scalar", "trialwise"])
def test_ensure_positive_ndt(edge, trialwise):
    """Response times at or below the given edge receive the sentinel logp.

    The guard takes the edge ready-made, so the only things to pin are the
    inclusive comparison (one response time sits exactly on the edge), the
    exemption for missing responses (-999.0 is below any edge but keeps its
    logp), and that a trial-wise edge tensor is accepted.
    """
    rt = np.array([0.1, 0.25, 0.29, 0.31, 0.4, 0.49, 0.51, 0.6, 1.0, -999.0, edge])
    data = np.column_stack([rt, np.ones(rt.size)])
    logp = np.arange(1.0, rt.size + 1.0)

    lower_edge = pt.as_tensor_variable(
        pm.pytensorf.floatX(np.full(rt.size, edge) if trialwise else edge)
    )

    after = ensure_positive_ndt(data, logp, lower_edge).eval()
    mask = (rt - edge <= 1e-15) & (rt != -999.0)

    assert np.all(after[mask] == LOGP_LB)
    assert np.all(after[~mask] == logp[~mask])


@pytest.mark.parametrize(
    ("list_params", "values", "ndt_edge_shift", "expected"),
    [
        (["v", "a", "z"], [0.5, 1.0, 0.5], None, None),
        (["v", "a", "z"], [0.5, 1.0, 0.5], {"param": "a", "scale": 1.0}, None),
        (["v", "a", "z", "t"], [0.5, 1.0, 0.5, 0.5], None, 0.5),
        (["v", "a", "z", "t", "st"], [0.5, 1.0, 0.5, 0.5, 0.1], None, 0.5),
        (
            ["v", "a", "z", "t", "st"],
            [0.5, 1.0, 0.5, 0.5, 0.1],
            {"param": "st", "scale": 0.5},
            0.45,
        ),
        (
            ["v", "a", "z", "tau", "t"],
            [0.5, 1.0, 0.5, 0.1, 0.5],
            {"param": "tau", "scale": 2.0},
            0.3,
        ),
        (
            ["v", "a", "z", "t", "st"],
            [0.5, 1.0, 0.5, 0.5, np.array([0.1, 0.2])],
            {"param": "st", "scale": 1.0},
            np.array([0.4, 0.3]),
        ),
    ],
    ids=[
        "no_t_is_no_floor",
        "no_t_ignores_declaration",
        "t_only",
        "st_without_declaration",
        "declared_st",
        "declared_param_not_named_st",
        "trialwise_param",
    ],
)
def test_ndt_lower_edge(list_params, values, ndt_edge_shift, expected):
    """The resolver reads only ``t`` and the declaration, nothing else by name."""
    dist_params = [pt.as_tensor_variable(pm.pytensorf.floatX(v)) for v in values]

    edge = _ndt_lower_edge(list_params, dist_params, ndt_edge_shift)

    if expected is None:
        assert edge is None
    elif ndt_edge_shift is None:
        # With nothing declared the edge is t itself, not a copy of it.
        assert edge is dist_params[list_params.index("t")]
    else:
        np.testing.assert_allclose(edge.eval(), expected, rtol=1e-6)


def test_make_likelihood_callable():
    """Test the make_likelihood_callable function."""

    def mock_func(*args, **kwargs):
        return 1, 2, 3

    with patch(
        "hssm.distribution_utils.dist.make_jax_logp_funcs_from_callable",
        return_value=("mocked_function", "mocked_grad", "mocked_nojit"),
    ) as mock_make_jax_logp_funcs:
        distribution_utils.make_likelihood_callable(
            loglik=mock_func,
            loglik_kind="analytical",
            backend="jax",
            params_only=False,
        )

        mock_make_jax_logp_funcs.assert_called_once_with(
            mock_func, vmap=False, params_only=False
        )

    with patch(
        "hssm.distribution_utils.dist.make_jax_logp_funcs_from_callable",
        return_value=("mocked_function", "mocked_grad", "mocked_nojit"),
    ) as mock_make_jax_logp_funcs:
        distribution_utils.make_likelihood_callable(
            loglik=mock_func,
            loglik_kind="approx_differentiable",
            params_is_reg=[False, False, False, False],
            backend="jax",
            params_only=False,
        )

        mock_make_jax_logp_funcs.assert_called_once_with(
            mock_func,
            vmap=True,
            params_only=False,
            params_is_reg=[False, False, False, False],
        )


class MockHasListParams:
    """Mock class that implements _HasListParams protocol."""

    def __init__(self, list_params):
        self._list_params = list_params


class TestCreateArgArrays:
    """Tests for _create_arg_arrays function."""

    def test_create_arg_arrays_basic(self):
        """Test basic functionality of _create_arg_arrays."""
        cls = MockHasListParams(["a", "b", "c"])
        args = (1, 2, 3, 4)
        result = _create_arg_arrays(cls, args)
        assert result == [np.array(i) for i in (1, 2, 3)]

    def test_create_arg_arrays_fewer_args_than_params(self):
        """Test _create_arg_arrays when args has fewer elements than params."""
        cls = MockHasListParams(["a", "b", "c", "d"])
        args = (1, 2)
        result = _create_arg_arrays(cls, args)
        assert result == [np.array(i) for i in (1, 2)]

    def test_create_arg_arrays_array_inputs(self):
        """Test _create_arg_arrays with array inputs."""
        cls = MockHasListParams(["a", "b"])
        args = ([1, 2, 3], np.array([4, 5, 6]))
        result = _create_arg_arrays(cls, args)

        assert len(result) == 2
        np.testing.assert_array_equal(result[0], np.array([1, 2, 3]))
        np.testing.assert_array_equal(result[1], np.array([4, 5, 6]))


class TestExtractSize:
    """Tests for _extract_size function."""

    def test_extract_size_from_kwargs(self):
        """Test _extract_size when size is in kwargs."""
        args = (1, 2, 3)
        kwargs = {"size": 10, "other": "value"}

        size, new_args, new_kwargs = _extract_size(args, kwargs)

        assert size == 10
        assert new_args == (1, 2, 3)
        assert new_kwargs == {"other": "value"}

    def test_extract_size_from_args(self):
        """Test _extract_size when size is in args."""
        args = (1, 2, 3, 15)
        kwargs = {"other": "value"}

        size, new_args, new_kwargs = _extract_size(args, kwargs)

        assert size == 15
        assert new_args == (1, 2, 3)
        assert new_kwargs == {"other": "value"}

    def test_extract_size_none_default(self):
        """Test _extract_size when size is None, should default to 1."""
        args = (1, 2, 3, None)
        kwargs = {}

        size, new_args, new_kwargs = _extract_size(args, kwargs)

        assert size == 1
        assert new_args == (1, 2, 3)
        assert new_kwargs == {}


class TestGetPOutlier:
    """Tests for _get_p_outlier function."""

    def test_get_p_outlier_present(self):
        """Test _get_p_outlier when p_outlier is present."""
        cls = MockHasListParams(["a", "b", "p_outlier"])
        arg_arrays = [np.array([1, 2]), np.array([3, 4]), np.array([0.1, 0.2])]

        p_outlier, new_arg_arrays = _get_p_outlier(cls, arg_arrays)

        np.testing.assert_array_equal(p_outlier, np.array([0.1, 0.2]))
        assert len(new_arg_arrays) == 2
        np.testing.assert_array_equal(new_arg_arrays[0], np.array([1, 2]))
        np.testing.assert_array_equal(new_arg_arrays[1], np.array([3, 4]))

    def test_get_p_outlier_not_present(self):
        """Test _get_p_outlier when p_outlier is not present."""
        cls = MockHasListParams(["a", "b", "c"])
        arg_arrays = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]

        p_outlier, new_arg_arrays = _get_p_outlier(cls, arg_arrays)

        assert p_outlier is None
        assert len(new_arg_arrays) == 3
        assert new_arg_arrays is arg_arrays  # Should be the same object

    def test_get_p_outlier_empty_params(self):
        """Test _get_p_outlier when _list_params is empty."""
        cls = MockHasListParams([])
        arg_arrays = [np.array([1, 2])]

        p_outlier, new_arg_arrays = _get_p_outlier(cls, arg_arrays)

        assert p_outlier is None
        assert new_arg_arrays is arg_arrays


@pytest.mark.parametrize(
    ("scale", "edge"),
    [(1.0, 0.4), (3.0, 0.2)],
    ids=["scale_1", "scale_3"],
)
def test_registered_ndt_edge_shift_reaches_logp(scale, edge):
    """A registry entry's declaration reaches the compiled logp through HSSM.

    Route: register_model -> Config.from_defaults -> HSSM -> make_distribution.
    """
    name = "ndt_edge_shift_registry_model"
    data, df = _ndt_grid_data()

    register_model(
        name,
        response=["rt", "response"],
        list_params=["v", "a", "z", "t", "st"],
        choices=[-1, 1],
        description=None,
        likelihoods={
            "analytical": {
                "loglik": _flat_logp,
                "backend": None,
                "default_priors": {},
                "bounds": NDT_BOUNDS,
                "extra_fields": None,
                "ndt_edge_shift": {"param": "st", "scale": scale},
            }
        },
    )
    try:
        model = hssm.HSSM(model=name, data=df, p_outlier=None, lapse=None)
    finally:
        default_model_config.pop(name, None)

    result = np.asarray(
        model.model_distribution.logp(data, 0.5, 0.5, 0.5, 0.5, 0.1).eval()
    )
    np.testing.assert_array_equal(result, _floored_at(edge))


@pytest.mark.parametrize(
    ("scale", "edge"),
    [(1.0, 0.4), (3.0, 0.2)],
    ids=["scale_1", "scale_3"],
)
def test_model_config_ndt_edge_shift_reaches_logp(scale, edge):
    """A user's model_config declaration reaches the compiled logp through HSSM.

    Route: model_config dict -> ModelConfig -> Config.update_config -> HSSM ->
    make_distribution.
    """
    data, df = _ndt_grid_data()

    model = hssm.HSSM(
        model="ndt_edge_shift_custom_model",
        data=df,
        loglik=_flat_logp,
        loglik_kind="analytical",
        model_config={
            "list_params": ["v", "a", "z", "t", "st"],
            "choices": [-1, 1],
            "bounds": NDT_BOUNDS,
            "ndt_edge_shift": {"param": "st", "scale": scale},
        },
        p_outlier=None,
        lapse=None,
    )

    result = np.asarray(
        model.model_distribution.logp(data, 0.5, 0.5, 0.5, 0.5, 0.1).eval()
    )
    np.testing.assert_array_equal(result, _floored_at(edge))


@pytest.mark.parametrize(
    ("scale", "edge"),
    [(0.5, 0.45), (3.0, 0.2)],
    ids=["scale_half", "scale_3"],
)
def test_make_distribution_for_supported_model_forwards_ndt_edge_shift(
    monkeypatch, scale, edge
):
    """make_distribution_for_supported_model forwards the registry declaration.

    full_ddm is the one bundled model with a declaration; its blackbox entry is
    patched to the flat stand-in likelihood and to the scale under test so the
    floored set reads off the grid.
    """
    data, _ = _ndt_grid_data()

    def flat_logp_full_ddm(data, v, a, z, t, sz, sv, st):
        return np.ones(data.shape[0])

    # Registers full_ddm on first use, so the entry below exists.
    Config.from_defaults("full_ddm", "blackbox")
    entry = default_model_config["full_ddm"]["likelihoods"]["blackbox"]
    monkeypatch.setitem(entry, "loglik", flat_logp_full_ddm)
    monkeypatch.setitem(entry, "ndt_edge_shift", {"param": "st", "scale": scale})

    Dist = make_distribution_for_supported_model("full_ddm", loglik_kind="blackbox")

    # v, a, z, t, sz, sv, st
    result = np.asarray(Dist.logp(data, 1.0, 1.0, 0.5, 0.5, 0.1, 0.3, 0.1).eval())
    np.testing.assert_array_equal(result, _floored_at(edge))


def test_full_ddm_keeps_likelihood_for_rt_below_t():
    """full_ddm returns hddm_wfpt's own log-likelihood on both sides of t.

    hddm_wfpt reads st as the full width, so it assigns real density to response
    times in (t - st / 2, t]. A guard that floors every rt <= t replaces that
    density with LOGP_LB; the t - st edge must leave it untouched. The response
    times run from below t - st to above t, so they cover the band the guard
    floors, the band hddm_wfpt itself scores as zero density, and the band it
    does not.
    """
    # v, a, z, t, sz, sv, st. Cast once so that both sides see the same values
    # whichever floatX is in effect.
    params = pm.pytensorf.floatX([1.0, 1.0, 0.5, 0.4, 0.1, 0.3, 0.3])
    rt = np.linspace(0.06, 0.46, 21)
    data = pm.pytensorf.floatX(np.column_stack([rt, np.ones_like(rt)]))

    # lapse defaults to None, so there is no p_outlier mixture on the HSSM side.
    Dist = make_distribution_for_supported_model("full_ddm", loglik_kind="blackbox")

    # Equal up to the float32 rounding of the output.
    np.testing.assert_allclose(
        Dist.logp(data, *params).eval(),
        logp_full_ddm(data, *params),
        rtol=1e-6,
    )
