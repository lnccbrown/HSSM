"""Tests for the network-inspection plots and their predictor adapter (#1505).

The adapter helpers need no LANfactory and always run; the plot smoke tests skip
without it (it comes with the ``inspect`` extra or the ``notebook`` group).
"""

from types import SimpleNamespace

import numpy as np
import onnx
import pandas as pd
import pytest
from ssms.config import ModelConfigBuilder

import hssm
from hssm import tracking
from hssm.distribution_utils.onnx import make_jax_logp_funcs_from_onnx
from hssm.distribution_utils.onnx_utils.onnx2jax import make_jax_func
from hssm.plotting import network_inspection
from hssm.plotting.network_inspection import (
    as_parameter_df,
    load_model_network,
    make_model_predictor,
    make_network_predictor,
    manifold_parameters,
    network_input_size,
    network_params,
    network_source,
)

pytestmark = pytest.mark.usefixtures("no_hf_hub_download")

ANGLE_PARAMS = {"v": 0.5, "a": 1.5, "z": 0.5, "t": 0.3, "theta": 0.2}
DDM_PARAMS = {"v": 0.5, "a": 1.5, "z": 0.5, "t": 0.3}


def _stand_in(
    model_name="ddm",
    loglik="ddm.onnx",
    loglik_kind="approx_differentiable",
    list_params=("v", "a", "z", "t", "p_outlier"),
    choices=(-1, 1),
    tracking_network=None,
):
    """Build a stand-in for an HSSM model with the attributes the adapter reads."""
    return SimpleNamespace(
        model_name=model_name,
        choices=choices,
        model_config=SimpleNamespace(
            loglik=loglik, loglik_kind=loglik_kind, list_params=list(list_params)
        ),
        _tracking_network=tracking_network or {},
    )


def _rows(params: dict, n: int = 50, seed: int = 0) -> np.ndarray:
    """Network input rows ``[params..., rt, choice]`` over random rt and choice."""
    rng = np.random.default_rng(seed)
    rt = rng.uniform(0.35, 3.0, n)
    choice = rng.choice([-1.0, 1.0], n)
    return np.column_stack([np.tile(list(params.values()), (n, 1)), rt, choice]).astype(
        np.float32
    )


@pytest.fixture
def ddm_lan_model(fixture_path):
    """A DDM built on the local copy of its network."""
    data = hssm.simulate_data(model="ddm", theta=DDM_PARAMS, size=50, random_state=1)
    return hssm.HSSM(
        data=data,
        model="ddm",
        loglik=fixture_path / "ddm.onnx",
        loglik_kind="approx_differentiable",
    )


# --------------------------------------------------------------------------- #
# network_source
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("loglik_kind", ["analytical", "blackbox"])
def test_network_source_rejects_non_network_likelihoods(loglik_kind):
    model = _stand_in(loglik_kind=loglik_kind, loglik=lambda *args: 0.0)
    with pytest.raises(ValueError, match="loglik_kind='approx_differentiable'"):
        network_source(model)


def test_network_source_rejects_a_built_callable():
    model = _stand_in(loglik=lambda *args: 0.0)
    with pytest.raises(ValueError, match="given as a function"):
        network_source(model)


def test_network_source_returns_the_network_file(fixture_path):
    assert network_source(_stand_in()) == "ddm.onnx"
    path = fixture_path / "ddm.onnx"
    assert network_source(_stand_in(loglik=path)) == path


def test_network_source_of_a_built_model_is_its_onnx_file(ddm_lan_model, fixture_path):
    """The model keeps the network path in its config after building its Op."""
    assert callable(ddm_lan_model.loglik)
    assert network_source(ddm_lan_model) == fixture_path / "ddm.onnx"


def test_network_source_of_an_analytical_model_raises():
    data = hssm.simulate_data(model="ddm", theta=DDM_PARAMS, size=20, random_state=1)
    model = hssm.HSSM(data=data, model="ddm", loglik_kind="analytical")
    with pytest.raises(ValueError, match="loglik_kind is 'analytical'"):
        network_source(model)


# --------------------------------------------------------------------------- #
# network_params
# --------------------------------------------------------------------------- #


def test_network_params_drops_p_outlier():
    assert network_params(_stand_in()) == ["v", "a", "z", "t"]


def test_network_params_of_a_built_model(ddm_lan_model):
    assert "p_outlier" in ddm_lan_model.model_config.list_params
    assert network_params(ddm_lan_model) == ["v", "a", "z", "t"]


def test_network_params_rejects_a_different_order():
    model = _stand_in(list_params=("a", "v", "z", "t"))
    with pytest.raises(ValueError, match="do not match the ssm-simulators model"):
        network_params(model)


def test_network_params_rejects_different_choices():
    model = _stand_in(choices=(0, 1))
    with pytest.raises(ValueError, match=r"choices of model 'ddm' \(\[0, 1\]\)"):
        network_params(model)


def test_network_params_rejects_two_choices_other_than_minus_one_and_one():
    """LANfactory evaluates two-choice networks at choices -1 and 1 only."""
    model = _stand_in(
        model_name="lba2", list_params=("A", "b", "v0", "v1"), choices=(0, 1)
    )
    with pytest.raises(ValueError, match=r"choices \[0, 1\], but LANfactory"):
        network_params(model)


def test_network_params_rejects_a_model_unknown_to_ssms():
    model = _stand_in(model_name="not_an_ssms_model")
    with pytest.raises(ValueError, match="ssm-simulators has none"):
        network_params(model)


@pytest.mark.parametrize(
    "model_name",
    [
        name
        for name in hssm.modelconfig.list_models()
        if "approx_differentiable"
        in hssm.modelconfig.get_default_model_config(name)["likelihoods"]
    ],
)
def test_shipped_networks_match_their_ssms_models(model_name):
    """Every built-in LAN model passes the check LANfactory relies on."""
    config = hssm.modelconfig.get_default_model_config(model_name)
    model = _stand_in(
        model_name=model_name,
        list_params=config["list_params"],
        choices=config["choices"],
    )
    assert network_params(model) == list(
        ModelConfigBuilder.from_model(model_name)["params"]
    )


# --------------------------------------------------------------------------- #
# load_model_network
# --------------------------------------------------------------------------- #


def _record_pinned_revision(monkeypatch):
    seen = {}

    def fake_load(source):
        seen["revision"] = (
            tracking.pinned_revision(source) if isinstance(source, str) else None
        )
        return "network"

    monkeypatch.setattr(network_inspection, "load_onnx_model", fake_load)
    return seen


def test_load_model_network_fetches_the_recorded_revision(monkeypatch):
    seen = _record_pinned_revision(monkeypatch)
    model = _stand_in(
        tracking_network={"network_file": "ddm.onnx", "hf_revision": "abc123"}
    )
    assert load_model_network(model) == "network"
    assert seen["revision"] == "abc123"


@pytest.mark.parametrize(
    "tracking_network",
    [{}, {"network_file": "angle.onnx", "hf_revision": "abc123"}],
)
def test_load_model_network_without_a_matching_record_does_not_pin(
    monkeypatch, tracking_network
):
    seen = _record_pinned_revision(monkeypatch)
    load_model_network(_stand_in(tracking_network=tracking_network))
    assert seen["revision"] is None


def test_load_model_network_loads_a_local_file(ddm_lan_model):
    assert isinstance(load_model_network(ddm_lan_model), onnx.ModelProto)


# --------------------------------------------------------------------------- #
# Predictors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("network_file", "params"),
    [("ddm.onnx", DDM_PARAMS), ("angle.onnx", ANGLE_PARAMS)],
)
def test_predictor_matches_the_single_trial_network(fixture_path, network_file, params):
    """Batching over rows gives what the network gives one row at a time."""
    network = onnx.load(fixture_path / network_file)
    rows = _rows(params)

    out = make_network_predictor(network)(rows)

    single_trial = make_jax_func(network)
    expected = np.array([np.asarray(single_trial(row)) for row in rows])
    assert out.shape == (len(rows), 1)
    np.testing.assert_allclose(out[:, 0], expected, rtol=1e-6, atol=1e-6)


def test_predictor_matches_the_model_likelihood(fixture_path):
    """The predictor gives the log-likelihood HSSM's own likelihood computes."""
    network = onnx.load(fixture_path / "angle.onnx")
    rows = _rows(ANGLE_PARAMS)
    logp, _, _ = make_jax_logp_funcs_from_onnx(
        network, params_is_reg=[False] * len(ANGLE_PARAMS)
    )

    expected = np.asarray(logp(rows[:, -2:], *rows[0, :-2]))

    out = make_network_predictor(network)(rows)
    np.testing.assert_allclose(out[:, 0], expected, rtol=1e-5, atol=1e-5)


def test_predictor_rejects_rows_of_the_wrong_width(fixture_path):
    predict = make_network_predictor(onnx.load(fixture_path / "ddm.onnx"))
    with pytest.raises(ValueError, match=r"shape \(N, 6\)"):
        predict(np.zeros((3, 7), dtype=np.float32))
    with pytest.raises(ValueError, match=r"shape \(N, 6\)"):
        predict(np.zeros(6, dtype=np.float32))


def test_network_input_size(fixture_path):
    assert network_input_size(onnx.load(fixture_path / "ddm.onnx")) == 6
    assert network_input_size(onnx.load(fixture_path / "angle.onnx")) == 7


def test_make_model_predictor(ddm_lan_model):
    rows = _rows(DDM_PARAMS, n=5)
    assert make_model_predictor(ddm_lan_model)(rows).shape == (5, 1)


def test_make_model_predictor_rejects_a_network_of_another_size(fixture_path):
    """An angle model on the DDM's network would misread its inputs."""
    model = _stand_in(
        model_name="angle",
        loglik=fixture_path / "ddm.onnx",
        list_params=("v", "a", "z", "t", "theta"),
    )
    with pytest.raises(ValueError, match="takes 6 inputs"):
        make_model_predictor(model)


# --------------------------------------------------------------------------- #
# Parameter handling
# --------------------------------------------------------------------------- #


def test_as_parameter_df():
    df = pd.DataFrame([DDM_PARAMS, DDM_PARAMS])
    assert as_parameter_df(df) is df
    pd.testing.assert_frame_equal(
        as_parameter_df(DDM_PARAMS), pd.DataFrame([DDM_PARAMS])
    )


def test_manifold_parameters_default_to_ssms_defaults():
    config = ModelConfigBuilder.from_model("angle")
    expected = dict(zip(config["params"], config["default_params"]))
    out = manifold_parameters("angle", None)
    assert out.shape == (1, 5)
    assert out.iloc[0].to_dict() == pytest.approx(expected)


def test_manifold_parameters_override_some_defaults():
    out = manifold_parameters("angle", {"a": 1.7, "theta": 0.3})
    assert list(out.columns) == ["v", "a", "z", "t", "theta"]
    assert out.loc[0, "a"] == 1.7
    assert out.loc[0, "theta"] == 0.3


def test_manifold_parameters_reject_unknown_names():
    with pytest.raises(ValueError, match=r"no parameter\(s\) \['b'\]"):
        manifold_parameters("ddm", {"b": 1.0})


def test_manifold_parameters_pass_a_dataframe_through():
    df = pd.DataFrame([DDM_PARAMS])
    assert manifold_parameters("ddm", df) is df


# --------------------------------------------------------------------------- #
# Plots (need LANfactory)
# --------------------------------------------------------------------------- #


@pytest.fixture
def inspectors():
    return pytest.importorskip("lanfactory.network_inspectors")


def test_plot_kde_vs_lan(ddm_lan_model, inspectors, tmp_path):
    """The comparison plot draws, and saves when asked to."""
    out = hssm.plotting.plot_kde_vs_lan(
        ddm_lan_model,
        pd.DataFrame([DDM_PARAMS, DDM_PARAMS | {"v": -0.5}]),
        n_samples=200,
        n_reps=2,
        grid=inspectors.GridSpec(n_points_2c=100, rt_step_2c=0.05),
        plot=inspectors.PlotConfig(show=False, save=True, save_dir=f"{tmp_path}/"),
    )
    assert out is None
    assert (tmp_path / "kde_vs_mlp_plot.png").exists()


def test_plot_kde_vs_lan_accepts_a_mapping(ddm_lan_model, inspectors):
    hssm.plotting.plot_kde_vs_lan(
        ddm_lan_model,
        DDM_PARAMS,
        n_samples=200,
        n_reps=1,
        grid=inspectors.GridSpec(n_points_2c=50, rt_step_2c=0.05),
        plot=inspectors.PlotConfig(show=False),
    )


def test_plot_lan_manifold(ddm_lan_model, inspectors):
    fig = hssm.plotting.plot_lan_manifold(
        ddm_lan_model,
        vary={"a": [1.0, 1.5, 2.0]},
        parameters={"v": 0.5},
        grid=inspectors.GridSpec(n_rt_steps=20),
        plot=inspectors.PlotConfig(show=False),
    )
    assert type(fig).__name__ == "Figure"
    assert len(fig.data) == 1


def test_plot_lan_manifold_rejects_an_analytical_model(inspectors):
    data = hssm.simulate_data(model="ddm", theta=DDM_PARAMS, size=20, random_state=1)
    model = hssm.HSSM(data=data, model="ddm", loglik_kind="analytical")
    with pytest.raises(ValueError, match="loglik_kind is 'analytical'"):
        hssm.plotting.plot_lan_manifold(model)
