import hssm

hssm.set_floatX("float32")


def test_sample_posterior_predictive(cav_idata, cavanagh_test):
    model = hssm.HSSM(
        data=cavanagh_test,
        include=[
            {
                "name": "v",
                "prior": {
                    "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                    "theta": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                },
                "formula": "v ~ (1|participant_id) + theta",
                "link": "identity",
            },
        ],
    )  # Doesn't matter what model or data we use here
    delattr(cav_idata, "posterior_predictive")
    cav_idata_copy = cav_idata.copy()

    posterior_predictive = model.sample_posterior_predictive(
        idata=cav_idata_copy, n_samples=1, inplace=False
    )
    assert posterior_predictive.posterior_predictive.draw.size == 1
    assert "posterior_predictive" not in cav_idata_copy

    model.sample_posterior_predictive(idata=cav_idata_copy, n_samples=1, inplace=True)
    assert cav_idata_copy.posterior_predictive.draw.size == 1
    assert cav_idata_copy.posterior.draw.size == 500

    model._inference_obj = cav_idata
    posterior_predictive = model.sample_posterior_predictive(n_samples=1, inplace=False)

    assert posterior_predictive.posterior_predictive.draw.size == 1
    assert "posterior_predictive" not in cav_idata

    model.sample_posterior_predictive(n_samples=1, inplace=True)
    assert cav_idata.posterior_predictive.draw.size == 1
    assert cav_idata.posterior.draw.size == 500
