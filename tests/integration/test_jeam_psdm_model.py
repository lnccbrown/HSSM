"""Construction and compiled-likelihood tests for the registered fixed PSDM."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jeam")

from jeam.Models.Spherical import ProjectedSphericalDiffusionModel

import hssm
from hssm.integrations.jeam import (
    logp_projected_spherical_diffusion,
    simulate_projected_spherical_diffusion,
)

COMPILED_RTOL = 5e-7
COMPILED_ATOL = 5e-7


def _build_projected_spherical_model() -> hssm.HSSM:
    """Build an ordinary HSSM model from deterministic JEAM observations."""
    observations = simulate_projected_spherical_diffusion(
        theta=np.array([0.70, 0.35, 0.40, 0.08]),
        random_state=1947,
        n_replicas=12,
    )
    return hssm.HSSM(
        data=pd.DataFrame(observations, columns=["rt", "response"]),
        model="projected_spherical_diffusion",
        p_outlier=None,
    )


def test_simulated_projected_data_builds_an_ordinary_hssm_model():
    """The fixed PSDM should use HSSM's standard config-driven lifecycle."""
    model = _build_projected_spherical_model()

    assert type(model) is hssm.HSSM
    assert model.model_name == "projected_spherical_diffusion"
    assert model.response_kind == "continuous"
    assert model.response_bounds == {"response": (0.0, np.pi)}
    assert model.choices is None
    assert model.n_choices is None
    assert model.list_params == ["v_x", "v_y", "a", "t"]
    assert model.model_config.loglik is logp_projected_spherical_diffusion
    assert model.model_config.rv is simulate_projected_spherical_diffusion
    assert model.params["v_y"].bounds == (0.0, 3.0)


def test_compiled_distribution_matches_adapter_and_direct_jeam():
    """The registered black-box path should preserve pointwise JEAM values."""
    model = _build_projected_spherical_model()
    data = np.array([[0.28, 0.2], [0.53, 1.4], [0.91, 2.7]])
    v_x = 0.45
    v_y = 0.70
    threshold = 1.10
    ndt = 0.08
    producer = ProjectedSphericalDiffusionModel(threshold_dynamic="fixed")
    direct = producer.joint_lpdf(
        rt=data[:, 0],
        theta=data[:, 1],
        drift_vec=np.column_stack(
            (np.full(data.shape[0], v_x), np.full(data.shape[0], v_y))
        ),
        ndt=np.full(data.shape[0], ndt),
        threshold=threshold,
        decay=0.0,
        threshold_function=None,
        dt_threshold_function=None,
        s_v=0.0,
        s_t=0.0,
        sigma=1.0,
    )
    adapted = logp_projected_spherical_diffusion(
        data,
        v_x,
        v_y,
        threshold,
        ndt,
    )
    compiled = np.asarray(
        model.model_distribution.logp(
            data,
            v_x,
            v_y,
            threshold,
            ndt,
        ).eval()
    )

    np.testing.assert_allclose(adapted, direct, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        compiled,
        direct,
        rtol=COMPILED_RTOL,
        atol=COMPILED_ATOL,
    )


def test_compiled_distribution_accepts_trialwise_radial_drift():
    """Positive trial-wise radial-drift vectors should reach JEAM row by row."""
    model = _build_projected_spherical_model()
    data = np.array([[0.28, 0.2], [0.53, 1.4], [0.91, 2.7]])
    radial_drift = np.array([0.70, 0.55, 0.40])
    direct = ProjectedSphericalDiffusionModel(threshold_dynamic="fixed").joint_lpdf(
        rt=data[:, 0],
        theta=data[:, 1],
        drift_vec=np.column_stack((np.full(data.shape[0], 0.45), radial_drift)),
        ndt=np.full(data.shape[0], 0.08),
        threshold=1.10,
        decay=0.0,
        threshold_function=None,
        dt_threshold_function=None,
        s_v=0.0,
        s_t=0.0,
        sigma=1.0,
    )

    compiled = np.asarray(
        model.model_distribution.logp(
            data,
            0.45,
            radial_drift,
            1.10,
            0.08,
        ).eval()
    )

    np.testing.assert_allclose(
        compiled,
        direct,
        rtol=COMPILED_RTOL,
        atol=COMPILED_ATOL,
    )
