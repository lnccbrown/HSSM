"""Check which predictive plots can be interpreted for circular responses."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import hssm


def test_circular_plots_require_choices_only_for_choice_diagnostics():
    """Reject choice diagnostics before sampling; pooled RT prediction stays valid."""
    data = pd.DataFrame({"rt": [0.4, 0.6, 1.0], "response": [-1.3, 0.0, 2.1]})
    model = hssm.HSSM(
        model="circular_diffusion", data=data, p_outlier=None, a=1.0, t=0.1
    )
    with pytest.raises(ValueError, match="require categorical responses"):
        model.plot_quantile_probability(cond="condition")
    with pytest.raises(ValueError, match="require categorical responses"):
        model.plot_model_cartoon()

    draws = model.sample_prior_predictive(draws=3, random_seed=17)
    ax = model.plot_predictive(
        dt=draws, predictive_group="prior_predictive", n_samples=None, uncertainty=None
    )
    try:
        assert ax.get_xlabel() == "Response Time"
    finally:
        plt.close(ax.figure)
