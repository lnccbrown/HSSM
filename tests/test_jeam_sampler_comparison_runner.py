"""Fast tests for the fixed-CDM sampler-comparison orchestration."""

import copy
import json
from pathlib import Path

import pytest

from scripts.benchmark_jeam_sampler_comparison import (
    DEFAULT_SPEC_PATH,
    _fit_failures,
    _scientific_gate,
    _worker_command,
    _worker_settings,
)


@pytest.fixture(scope="module")
def spec():
    """Load the frozen protocol without importing JEAM."""
    return json.loads(DEFAULT_SPEC_PATH.read_text(encoding="utf-8"))


def _passing_fit(sampler: str = "pymc_nuts") -> dict:
    """Return one compact canonical fit exactly inside every gate."""
    return {
        "scenario": "baseline_asymmetric",
        "sampler": sampler,
        "role": "canonical",
        "status": "completed",
        "sampler_diagnostics": {"divergences": 0 if sampler != "slice" else None},
        "parameters": [
            {
                "name": name,
                "rhat": 1.005,
                "ess_bulk": 800.0,
                "ess_tail": 700.0,
                "mcse_over_posterior_sd": 0.02,
                "truth_in_hdi": True,
            }
            for name in ("a", "t", "v_x", "v_y")
        ],
        "predictive": {
            "rt_quantile_absolute_errors": [0.02, 0.03, 0.04],
            "mean_angle_distance": 0.03,
            "mean_resultant_length_absolute_error": 0.02,
        },
    }


def test_worker_settings_never_confuse_smoke_with_canonical(spec):
    """The smoke reduction must leave all canonical counts untouched."""
    canonical = _worker_settings(spec, smoke=False)
    smoke = _worker_settings(spec, smoke=True)

    assert canonical == {
        "chains": 4,
        "tune": 1000,
        "draws": 1000,
        "posterior_predictive_draws": 40,
    }
    assert smoke == {
        "chains": 1,
        "tune": 5,
        "draws": 5,
        "posterior_predictive_draws": 2,
    }


def test_worker_command_carries_one_explicit_route(tmp_path):
    """Fresh workers should receive exact data, route, and evidence destinations."""
    command = _worker_command(
        spec_path=DEFAULT_SPEC_PATH,
        scenario_name="baseline_asymmetric",
        sampler_id="numpyro_nuts",
        data_path=tmp_path / "data.npy",
        work_dir=tmp_path,
        result_path=tmp_path / "worker.json",
        smoke=True,
    )

    assert command.count("--scenario") == command.count("--sampler") == 1
    assert command[command.index("--scenario") + 1] == "baseline_asymmetric"
    assert command[command.index("--sampler") + 1] == "numpyro_nuts"
    assert Path(command[command.index("--data-file") + 1]) == tmp_path / "data.npy"
    assert command[-1] == "--smoke"


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (("sampler_diagnostics", "divergences", 1), "divergences are not zero"),
        (("parameters", 0, "rhat", 1.01), "R-hat gate failed"),
        (("parameters", 0, "ess_bulk", 400.0), "bulk ESS gate failed"),
        (("parameters", 0, "ess_tail", 400.0), "tail ESS gate failed"),
        (("parameters", 0, "mcse_over_posterior_sd", 0.05), "MCSE/SD gate failed"),
        (("parameters", 0, "truth_in_hdi", False), "truth is outside"),
        (("predictive", "rt_quantile_absolute_errors", [0.121]), "RT gate failed"),
        (("predictive", "mean_angle_distance", 0.101), "mean-angle gate failed"),
        (
            ("predictive", "mean_resultant_length_absolute_error", 0.081),
            "resultant-length gate failed",
        ),
    ],
)
def test_fit_gates_are_strict_at_the_predeclared_boundaries(spec, mutation, expected):
    """Each stored raw metric should independently trigger its declared failure."""
    fit = _passing_fit()
    if mutation[0] == "parameters":
        _, index, field, value = mutation
        fit["parameters"][index][field] = value
    else:
        section, field, value = mutation
        fit[section][field] = value

    failures = _fit_failures(fit, spec)
    assert any(expected in failure for failure in failures)


def test_scientific_gate_rejects_missing_fits_and_shared_failures(spec):
    """A subset cannot pass by omitting hard scenarios or failed shared checks."""
    fits = [_passing_fit()]
    preflight = [
        {
            "scenario": "baseline_asymmetric",
            "prior_predictive": {"passed": False},
            "maximum_objective_absolute_error": 2e-8,
        }
    ]

    gate = _scientific_gate(copy.deepcopy(fits), preflight, spec)

    assert gate["passed"] is False
    assert all(result["passed"] is False for result in gate["by_sampler"].values())
    assert any("prior-predictive" in failure for failure in gate["failures"])
    assert any("objective parity" in failure for failure in gate["failures"])
    assert any("exactly one fit" in failure for failure in gate["failures"])
