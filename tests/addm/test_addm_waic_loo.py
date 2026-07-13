"""CH: pointwise log-likelihood for aDDM model comparison (WAIC / LOO).

The post-hoc pointwise log-likelihood re-evaluates the log-likelihood Op with a
draw dimension on the parameters. The trial-wise builder from commit CC (which
picks its vmap ``in_axes`` from the runtime parameter shapes) handles that
re-evaluation for free, so ``idata_kwargs={"log_likelihood": True}`` yields a
finite ``(chain, draw, n_obs)`` log-likelihood and ``az.loo`` runs —
no dedicated draw-dimension kernel was needed.

This needs only the vendored JAX likelihood (no ssm-simulators), so it runs in the
normal suite. It samples, so it is marked slow.
"""

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_addm_subclass import make_addm_dataframe  # noqa: E402

import hssm  # noqa: E402


@pytest.mark.slow
def test_addm_log_likelihood_enables_waic_loo():
    """Pointwise log-likelihood is emitted, correctly shaped, and drives ``az.loo``."""
    hssm.set_floatX("float64")
    df = make_addm_dataframe(30, seed=1)
    model = hssm.aDDM(data=df)

    # >=26 total draws so arviz 1.x PSIS-LOO has a >=5-sample Pareto tail.
    idata = model.sample(
        draws=30,
        tune=15,
        chains=2,
        cores=1,
        idata_kwargs={"log_likelihood": True},
    )

    # Pointwise log-likelihood is present, correctly shaped, and finite.
    assert "log_likelihood" in idata
    ll = idata.log_likelihood
    var = next(iter(ll.data_vars))
    arr = np.asarray(ll[var])
    assert arr.shape[:2] == (2, 30)  # (chain, draw, ...)
    assert arr.shape[-1] == len(df)  # one column per observed trial
    assert np.isfinite(arr).all()

    # LOO runs and returns a finite estimate (arviz 1.x dropped `az.waic`).
    loo = az.loo(idata)
    assert np.isfinite(loo.elpd)  # arviz 1.x: ELPDData.elpd (was `elpd_loo`)
