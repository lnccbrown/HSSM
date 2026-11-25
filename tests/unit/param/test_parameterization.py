"""Tests for shared group-specific parameterization helpers."""

import pytest

import hssm
from hssm.param.parameterization import _resolve_noncentered
from hssm.param.regression_param import RegressionParam


@pytest.mark.parametrize(
    ("noncentered", "component_name", "expected"),
    [
        pytest.param(True, "v", True, id="scalar-true"),
        pytest.param(False, "v", False, id="scalar-false"),
        pytest.param(None, "v", False, id="explicit-none-centers"),
        pytest.param({"v": True}, "v", True, id="component-true"),
        pytest.param({"v": False}, "v", False, id="component-false"),
        pytest.param({"a": False}, "v", True, id="missing-component-default"),
    ],
)
def test_resolve_noncentered_model_setting(noncentered, component_name, expected):
    """Resolve scalar, default, and component-dictionary settings."""
    assert _resolve_noncentered(noncentered, component_name) is expected


@pytest.mark.parametrize(
    ("noncentered", "prior_noncentered", "expected"),
    [
        pytest.param(False, True, True, id="prior-enables"),
        pytest.param(True, False, False, id="prior-disables"),
        pytest.param({"v": False}, True, True, id="prior-beats-component"),
        pytest.param(None, False, False, id="prior-beats-default"),
    ],
)
def test_resolve_noncentered_prior_override(noncentered, prior_noncentered, expected):
    """Give a per-prior setting precedence over every model-level form."""
    assert _resolve_noncentered(noncentered, "v", prior_noncentered) is expected


def test_hssm_threads_noncentered_to_safe_priors(cavanagh_test, monkeypatch):
    """Pass one captured model setting unchanged through parameter construction."""
    received = []
    original = RegressionParam._make_safe_priors

    def capture_noncentered(self, design_matrices, is_ddm, noncentered=True):
        received.append(noncentered)
        return original(self, design_matrices, is_ddm, noncentered)

    monkeypatch.setattr(RegressionParam, "_make_safe_priors", capture_noncentered)
    model_setting = {"v": False}

    hssm.HSSM(
        data=cavanagh_test,
        model="ddm",
        include=[{"name": "v", "formula": "v ~ 1 + (1|participant_id)"}],
        p_outlier=0.0,
        noncentered=model_setting,
        process_initvals=False,
    )

    assert received == [model_setting]
    assert received[0] is model_setting
