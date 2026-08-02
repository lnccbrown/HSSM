"""End-to-end sampling with an hsgp() regression term (gh-624)."""

import bambi as bmb
import numpy as np
import pytest

import hssm

hssm.set_floatX("float32", update_jax=True)

HSGP_TERM = "hsgp(x, m=8, c=2)"


@pytest.mark.slow
def test_hsgp_regression_samples():
    rng = np.random.default_rng(42)
    data = hssm.simulate_data(
        model="ddm", theta=dict(v=0.5, a=1.5, z=0.5, t=0.3), size=300
    )
    data["x"] = rng.uniform(0.0, 6.0, len(data))

    model = hssm.HSSM(
        data=data,
        model="ddm",
        include=[
            {
                "name": "v",
                "formula": f"v ~ 0 + {HSGP_TERM}",
                "prior": {
                    HSGP_TERM: {
                        "sigma": bmb.Prior("Exponential", lam=3.0),
                        "ell": bmb.Prior("InverseGamma", mu=2.0, sigma=0.2),
                    },
                },
            }
        ],
        loglik_kind="analytical",
    )

    idata = model.sample(
        draws=25,
        tune=25,
        chains=1,
        cores=1,
        progressbar=False,
        compute_convergence_checks=False,
    )

    # The full HSGP posterior must be present: covariance parameters, basis
    # weights, and the per-observation GP contribution.
    posterior_vars = set(idata.posterior.data_vars)
    for suffix in ("_sigma", "_ell", "_weights"):
        assert f"v_{HSGP_TERM}{suffix}" in posterior_vars
    assert f"v_{HSGP_TERM}" in posterior_vars
