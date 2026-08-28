"""Fast contracts for the fixed-PSDM recovery runner."""

import numpy as np
import pytest

import scripts.benchmark_jeam_psdm_recovery as runner
from scripts.benchmark_jeam_psdm_recovery import (
    ARCHIVED_RESULT_PATH,
    DEFAULT_SPEC_PATH,
    REPO_ROOT,
    _assert_model_contract,
    _configure_precision,
    _load_json,
    _make_model,
    _make_objectives,
    _optimization_bounds,
    _polar_unit_vector_summary,
    _scenario,
    _simulate_dataset,
    _worker_settings,
)


@pytest.fixture(scope="module")
def spec():
    """Load the frozen study definition once."""
    return _load_json(DEFAULT_SPEC_PATH)


@pytest.fixture(scope="module")
def tiny_data(spec):
    """Generate a small deterministic prefix outside the canonical budget."""
    pytest.importorskip("jeam")
    scenario = dict(_scenario(spec, "baseline_asymmetric"))
    scenario["trials"] = 16
    return _simulate_dataset(scenario)


@pytest.fixture
def _precision_state():
    """Restore PyTensor and JAX precision independently after each mutation."""
    import jax
    import pytensor

    import hssm

    original_floatx = pytensor.config.floatX
    original_jax_x64 = bool(jax.config.jax_enable_x64)
    try:
        yield
    finally:
        hssm.set_floatX(original_floatx)
        jax.config.update("jax_enable_x64", original_jax_x64)


def test_load_json_requires_an_object(tmp_path):
    """A malformed top-level protocol should fail before any computation."""
    path = tmp_path / "not-an-object.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(TypeError, match="Expected a JSON object"):
        _load_json(path)


def test_unknown_scenario_fails_explicitly(spec):
    """Typos must not silently select a different recovery case."""
    with pytest.raises(ValueError, match="Unknown scenario 'missing'"):
        _scenario(spec, "missing")


def test_public_model_resolves_the_frozen_priors_bounds_and_initvals(spec, tiny_data):
    """The runner should test the registered HSSM defaults without overrides."""
    contract = _assert_model_contract(_make_model(tiny_data), spec)

    assert contract == {
        "priors": spec["priors_and_initialization"]["priors"],
        "initvals": spec["priors_and_initialization"][
            "resolved_untransformed_initvals"
        ],
        "bounds": spec["priors_and_initialization"]["configured_bounds"],
    }


@pytest.mark.parametrize("float_dtype", ["float32", "float64"])
def test_public_model_contract_is_stable_across_pytensor_precision(
    spec, tiny_data, float_dtype, _precision_state
):
    """Runtime scalar representation must not change the frozen contract."""
    import hssm

    hssm.set_floatX(float_dtype)
    contract = _assert_model_contract(_make_model(tiny_data), spec)

    assert contract == {
        "priors": spec["priors_and_initialization"]["priors"],
        "initvals": spec["priors_and_initialization"][
            "resolved_untransformed_initvals"
        ],
        "bounds": spec["priors_and_initialization"]["configured_bounds"],
    }


def test_direct_and_compiled_objectives_match_on_asymmetric_data(
    spec, tiny_data, _precision_state
):
    """The canonical optimizer comparison should preserve actual pointwise math."""
    _configure_precision()
    direct, compiled = _make_objectives(tiny_data)
    candidates = (
        np.array([0.6, 1.0, 1.1, 0.2]),
        np.array([0.48, 1.1, 1.02, 0.17]),
    )

    for candidate in candidates:
        assert compiled(candidate) == pytest.approx(
            direct(candidate),
            abs=spec["scientific_acceptance"]["maximum_objective_absolute_error"],
        )


def test_optimizer_bounds_respect_the_observed_ndt_support(spec, tiny_data):
    """Differential evolution should never evaluate t at or above minimum RT."""
    bounds = _optimization_bounds(tiny_data, spec)

    assert bounds[:3] == ((-2.0, 2.0), (0.02, 2.0), (0.4, 2.0))
    assert bounds[3][0] == 0.02
    assert bounds[3][1] < np.min(tiny_data[:, 0])
    assert np.nextafter(bounds[3][1], np.inf) == np.min(tiny_data[:, 0])


def test_polar_summary_uses_unit_vectors_instead_of_raw_angles():
    """Polar directions should receive a geometric unit-vector summary."""
    angles = np.array([0.0, np.pi / 2.0, np.pi])

    mean_sin, mean_cos = _polar_unit_vector_summary(angles)

    assert mean_sin == pytest.approx(1.0 / 3.0)
    assert mean_cos == pytest.approx(0.0, abs=1e-15)


def test_smoke_budget_cannot_be_confused_with_canonical_execution(spec):
    """Local wiring checks should be visibly smaller in every expensive dimension."""
    canonical = _worker_settings(spec, False)
    smoke = _worker_settings(spec, True)

    assert canonical == {
        "chains": 4,
        "tune": 1000,
        "draws": 1000,
        "prior_predictive_draws": 100,
        "posterior_predictive_draws": 40,
        "optimizer_maxiter": 20,
        "optimizer_popsize": 15,
    }
    assert all(smoke[key] < canonical[key] for key in canonical)


def test_archived_canonical_run_fails_before_any_side_effect(
    spec, tmp_path, monkeypatch
):
    """A replayed checkout must never create a new artifact labeled canonical v1."""
    output_path = tmp_path / "result.json"
    work_dir = tmp_path / "work"

    monkeypatch.setattr(
        runner,
        "_configure_precision",
        lambda: pytest.fail("canonical guard ran after precision mutation"),
    )

    with pytest.raises(RuntimeError, match="archived and must not be rerun"):
        runner.run_study(
            spec_path=DEFAULT_SPEC_PATH,
            output_path=output_path,
            work_dir=work_dir,
        )

    assert not output_path.exists()
    assert not work_dir.exists()


def test_smoke_cannot_overwrite_the_archived_result_path(spec, tmp_path, monkeypatch):
    """The noncanonical escape hatch must not target the historical result."""
    work_dir = tmp_path / "work"
    monkeypatch.setattr(
        runner,
        "_configure_precision",
        lambda: pytest.fail("archived-path guard ran after precision mutation"),
    )

    with pytest.raises(RuntimeError, match="result path is immutable"):
        runner.run_study(
            spec_path=DEFAULT_SPEC_PATH,
            output_path=ARCHIVED_RESULT_PATH,
            work_dir=work_dir,
            scenario_names=["baseline_asymmetric"],
            smoke=True,
        )

    assert not work_dir.exists()


def test_default_spec_path_is_repository_relative():
    """The default should remain relocatable with the HSSM checkout."""
    assert DEFAULT_SPEC_PATH.is_file()
    assert DEFAULT_SPEC_PATH == (
        REPO_ROOT / "benchmarks" / "specs" / "jeam_fixed_psdm_recovery_v1.json"
    )
