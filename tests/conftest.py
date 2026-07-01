"""Configuration and shared fixtures for the HSSM test suite."""

import gc
import logging
import os

import matplotlib as mpl

mpl.use("Agg")

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ssms.basic_simulators.simulator import simulator

import hssm

_memory_logger = logging.getLogger("hssm.tests.memory")

MIB = 1024 * 1024
_PSUTIL_AVAILABLE = True


def _clear_jax_caches() -> None:
    """Best-effort clear of JAX compilation caches.

    ``jax.clear_caches`` may be absent on older JAX versions and could raise
    depending on the backend state. Since this runs in fixture teardown, any
    failure here must not fail the test run or mask the original result — so we
    guard the attribute and swallow errors, logging at debug level.
    """
    try:
        import jax
    except ImportError:
        return

    clear_caches = getattr(jax, "clear_caches", None)
    if clear_caches is None:
        return

    try:
        clear_caches()
    except Exception:  # noqa: BLE001 - teardown must never raise
        _memory_logger.debug("jax.clear_caches() failed; ignoring.", exc_info=True)


@pytest.fixture(autouse=True)
def _slow_test_memory(request):
    """Reclaim backend memory after each slow test and optionally log RSS.

    Slow tests repeatedly build PyMC models and sample with the JAX, numpyro and
    pytensor backends, which allocate native memory (compiled functions, device
    buffers) that Python allocators never see. Without releasing it between tests
    a serial run grows RSS until the process is OOM-killed. Gated on the ``slow``
    marker so the fast suite is unaffected, and applies wherever the slow test
    lives (not just ``tests/slow/``).

    Reclamation and RSS logging live in one fixture on purpose: two separate
    autouse fixtures have no guaranteed teardown order, so the logged RSS could
    be sampled either before or after cleanup. Here the sequence is explicit:
    end-of-test RSS (the leak-hunting signal) is sampled first, then cleanup
    runs, then post-cleanup RSS is sampled to show how much was freed. RSS
    logging is opt-in via ``HSSM_TEST_RSS_LOG`` because JAX and pytensor allocate
    outside the Python heap, which allocation-only profilers miss.
    """
    global _PSUTIL_AVAILABLE

    is_slow = request.node.get_closest_marker("slow") is not None
    log_rss = is_slow and os.environ.get("HSSM_TEST_RSS_LOG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    process = None
    rss_before = 0
    if log_rss and _PSUTIL_AVAILABLE:
        try:
            import psutil

            process = psutil.Process()
            rss_before = process.memory_info().rss
        except ImportError:
            _PSUTIL_AVAILABLE = False
            _memory_logger.warning("psutil not installed; skipping RSS logging.")

    yield

    if not is_slow:
        return

    # Sample RSS at end of test (before reclaiming). This is a point-in-time
    # reading, not a peak — the process may have used more RSS earlier.
    rss_end = process.memory_info().rss if process is not None else None

    plt.close("all")
    _clear_jax_caches()
    gc.collect()

    if process is None:
        return

    rss_post_cleanup = process.memory_info().rss
    _memory_logger.info(
        "RSS %s: start=%.1f MiB end=%.1f MiB post_cleanup=%.1f MiB "
        "(test delta=%+.1f MiB, freed=%+.1f MiB)",
        request.node.nodeid,
        rss_before / MIB,
        rss_end / MIB,
        rss_post_cleanup / MIB,
        (rss_end - rss_before) / MIB,
        (rss_end - rss_post_cleanup) / MIB,
    )


@pytest.fixture(scope="module")
def data_ddm():
    """Return DDM simulation data."""
    v_true, a_true, z_true, t_true = [0.5, 1.5, 0.5, 0.5]
    obs_ddm = simulator([v_true, a_true, z_true, t_true], model="ddm", n_samples=100)
    obs_ddm = np.column_stack([obs_ddm["rts"][:, 0], obs_ddm["choices"][:, 0]])
    data = pd.DataFrame(obs_ddm, columns=["rt", "response"])

    return data


@pytest.fixture(scope="module")
def data_angle():
    """Return Angle simulation data."""
    v_true, a_true, z_true, t_true, theta_true = [0.5, 1.5, 0.5, 0.5, 0.3]
    obs_angle = simulator(
        [v_true, a_true, z_true, t_true, theta_true], model="angle", n_samples=100
    )
    obs_angle = np.column_stack([obs_angle["rts"][:, 0], obs_angle["choices"][:, 0]])
    data = pd.DataFrame(obs_angle, columns=["rt", "response"])
    return data


@pytest.fixture(scope="module")
def data_ddm_reg():
    """Return DDM simulation data with regression."""
    # Generate some fake simulation data
    intercept = 1.5
    x = np.random.uniform(-0.5, 0.5, size=250)
    y = np.random.uniform(-0.5, 0.5, size=250)

    v = intercept + 0.8 * x + 0.3 * y
    true_values = np.column_stack(
        [v, np.repeat([[1.5, 0.5, 0.5]], axis=0, repeats=250)]
    )

    dataset_reg_v = hssm.simulate_data(
        model="ddm",
        theta=true_values,
        size=1,
    )

    dataset_reg_v["x"] = x
    dataset_reg_v["y"] = y

    return dataset_reg_v


@pytest.fixture(scope="module")
def data_ddm_reg_va():
    """Return DDM simulation data with regression on v and a."""
    # Generate some fake simulation data
    intercept = 1.5
    intercept_a = 1.0
    x = np.random.uniform(-0.5, 0.5, size=100)
    y = np.random.uniform(-0.5, 0.5, size=100)

    m = np.random.uniform(-0.5, 0.5, size=100)
    n = np.random.uniform(-0.5, 0.5, size=100)

    v = intercept + 0.8 * x + 0.3 * y
    a = intercept_a + 0.1 * m + 0.1 * n
    true_values = np.column_stack([v, a, np.repeat([[0.5, 0.5]], axis=0, repeats=100)])

    dataset_reg_va = hssm.simulate_data(
        model="ddm",
        theta=true_values,
        size=1,  # Generate one data point for each of the 1000 set of true values
    )

    dataset_reg_va["x"] = x
    dataset_reg_va["y"] = y
    dataset_reg_va["m"] = m
    dataset_reg_va["n"] = n

    return dataset_reg_va


@pytest.fixture
def cav_dt():
    """Return Cavanagh idata."""
    return az.from_netcdf("tests/fixtures/cavanagh_idata.nc")


@pytest.fixture
def posterior():
    """Return posterior predictive."""
    return xr.open_dataarray("tests/fixtures/cavanagh_idata_pps.nc")


@pytest.fixture
def cavanagh_test():
    """Return Cavanagh test data."""
    return pd.read_csv("tests/fixtures/cavanagh_theta_test.csv", index_col=None)


# @pytest.fixture
# def cavanagh_data():
#     return hssm.load_data("cavanagh_theta")


@pytest.fixture
def basic_hssm_model():
    """Return a basic HSSM model."""
    cav_data = hssm.load_data("cavanagh_theta")
    basic_hssm_model = hssm.HSSM(
        data=cav_data,
        process_initvals=True,
        link_settings="log_logit",
        model="angle",
        include=[
            {
                "name": "v",
                "formula": "v ~ 1 + C(stim)",
            }
        ],
    )
    return basic_hssm_model


# Cartoon plot fixtures
@pytest.fixture
def cav_model_cartoon(cavanagh_test):
    """Return a Cavanagh model for cartoon plots."""
    cav_model = hssm.HSSM(
        model="ddm",
        data=cavanagh_test,
        include=[
            {
                "name": "v",
                "prior": {
                    "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1.5},
                },
                "formula": "v ~ 1 + stim",
                "link": "identity",
            },
            {
                "name": "a",
                "prior": {
                    "Intercept": {"name": "Normal", "mu": 1.5, "sigma": 0.5},
                },
                "formula": "a ~ 1 + (1|participant_id)",
                "link": "identity",
            },
        ],
        p_outlier=0.00,
    )

    # Attach trace
    idata_cav = az.from_netcdf("tests/fixtures/idata_cavanagh_cartoon.nc")
    cav_model._inference_obj = idata_cav
    return cav_model


@pytest.fixture
def intercept_only_ddm_cartoon(cavanagh_test):
    """Intercept-only DDM model (no regression covariates).

    This covers the case where Bambi >= 0.17 returns scalar deterministics
    with shape (1,) instead of (n_obs,), which previously caused a
    dimensionality error in attach_trialwise_params_to_df.
    """
    model = hssm.HSSM(
        model="ddm",
        data=cavanagh_test,
        p_outlier=0.00,
    )

    # Attach a minimal mock posterior so plot_model_cartoon can proceed
    # without requiring actual MCMC sampling.
    posterior = xr.Dataset(
        {
            "v_Intercept": (["chain", "draw"], np.array([[0.5, 0.6]])),
            "a_Intercept": (["chain", "draw"], np.array([[1.4, 1.5]])),
            "z_Intercept": (["chain", "draw"], np.array([[0.5, 0.5]])),
            "t_Intercept": (["chain", "draw"], np.array([[0.3, 0.3]])),
        },
        coords={"chain": [0], "draw": [0, 1]},
    )
    model._inference_obj = xr.DataTree.from_dict({"posterior": posterior})
    return model


@pytest.fixture
def race_model_cartoon():
    """Return a race model for cartoon plots."""
    my_race_data = pd.read_csv("tests/fixtures/data_race.csv")
    race_model = hssm.HSSM(
        model="race_no_bias_angle_4",
        data=my_race_data,
        include=[
            {
                "name": "v0",
                "prior": {
                    "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1.5},
                },
                "formula": "v0 ~ 1 + stim",
                "link": "identity",
            },
            {
                "name": "a",
                "prior": {
                    "Intercept": {"name": "Normal", "mu": 1.5, "sigma": 0.5},
                },
                "formula": "a ~ 1 + (1|participant_id)",
                "link": "identity",
            },
        ],
        p_outlier=0.00,
    )
    # Attach trace
    idata_race = az.from_netcdf("tests/fixtures/test_idata_race.nc")
    race_model._inference_obj = idata_race
    return race_model


# Only useful if running tests serially
def pytest_collection_modifyitems(config, items):
    """Reorder tests so fast tests run before slow tests."""
    slow_tests = [item for item in items if "slow" in item.keywords]
    fast_tests = [item for item in items if "slow" not in item.keywords]

    # Reorder items so fast tests run first, then slow tests
    items[:] = fast_tests + slow_tests
