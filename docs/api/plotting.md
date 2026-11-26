The `hssm.plotting` module provide functionalities to create HSSM-specific plots such
as the posterior predictive plots. Please checkout
[the plotting tutorial](../tutorials/plotting.ipynb) for more examples on how to use
these functions. Note that each of the posterior predictive, quantile probability and
model cartoon plotting functions has an equivalent in the
[`hssm.HSSM`](hssm.md) class, so you can call these functions from a built model without
having to import these functions.

`plot_kde_vs_lan` and `plot_lan_manifold` inspect the network behind a model with
`loglik_kind="approx_differentiable"`. They use
[LANfactory](https://github.com/lnccbrown/LANfactory), which depends on torch, so they
need the `inspect` extra: `uv add "hssm[inspect]"` (or `pip install "hssm[inspect]"`).
Their `grid` and `plot` arguments take LANfactory's `GridSpec` and `PlotConfig` from
`lanfactory.network_inspectors`.

:::hssm.plotting
