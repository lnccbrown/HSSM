"""Delegation tests for the HSSM plotting convenience methods.

docs/api/plotting.md promises every plotting function an equivalent method on
`hssm.HSSM`; these tests enforce that claim for all three.
"""

import pytest

import hssm


@pytest.mark.parametrize(
    "method",
    ["plot_predictive", "plot_quantile_probability", "plot_model_cartoon"],
)
def test_plot_method_delegates(monkeypatch, method):
    captured = {}

    def fake(model, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return "sentinel"

    monkeypatch.setattr(hssm.plotting, method, fake)
    model = object.__new__(hssm.HSSM)  # delegation needs no model state
    result = getattr(hssm.HSSM, method)(model, foo=1)

    assert result == "sentinel"
    assert captured["model"] is model  # self passed positionally
    assert captured["kwargs"] == {"foo": 1}
