"""aDDM model-cartoon plot: collapsing boundary + drift path + sample trajectories.

`hssm.plotting.plot_model_cartoon` is generic — it reads the boundary, a
representative evidence path, and a relative start marker from *simulator*
metadata (`metadata['boundary'|'trajectory'|'z']`). Two pieces make aDDM work:

* the CE-1+ ssm-simulators build emits those keys from `cssm.addm` (this file
  skips otherwise, like the aDDM PPC tests), and
* `plot_model_cartoon` resolves `list_params`/`choices` from `model.model_config`
  rather than the `default_model_config` registry (aDDM is an HSSMBase subclass,
  absent from that registry). See lnccbrown/HSSM#1031.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from matplotlib.axes import Axes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_addm_subclass import make_addm_dataframe  # noqa: E402

try:
    from ssms.basic_simulators.simulator import simulator as _sim
    from ssms.config import model_config as _ssms_model_config

    _md = _sim(
        model="addm",
        theta=np.array([[0.4, 1.0, 1.5, 0.2, 0.0, 0.0]]),
        n_samples=1,
        no_noise=True,
        delta_t=0.05,
        max_t=5.0,
    )["metadata"]
    _HAS_ADDM_CARTOON = (
        "addm" in _ssms_model_config
        and {"boundary", "trajectory", "z"} <= set(_md)
        and bool((np.asarray(_md["trajectory"]) > -999).any())
    )
except Exception:  # pragma: no cover - old ssm-sim build
    _HAS_ADDM_CARTOON = False

needs_addm_cartoon = pytest.mark.skipif(
    not _HAS_ADDM_CARTOON,
    reason="ssm-simulators build without aDDM cartoon metadata (boundary/trajectory/z)",
)

import hssm  # noqa: E402


@needs_addm_cartoon
@pytest.mark.slow
def test_plot_model_cartoon_addm_posterior():
    """Render an aDDM cartoon from a short-sampled model.

    Exercises the collapsing boundary, the deterministic drift path, and the
    stochastic sample trajectories on a single Axes.
    """
    model = hssm.aDDM(data=make_addm_dataframe(40, seed=3))
    idata = model.sample(
        draws=10,
        tune=10,
        chains=1,
        cores=1,
        idata_kwargs={"log_likelihood": False},
    )

    ax = hssm.plotting.plot_model_cartoon(
        model,
        dt=idata,
        n_samples=5,
        bins=20,
        plot_predictive_mean=True,
        plot_predictive_samples=False,
        n_trajectories=3,  # exercise the stochastic example paths (engine traj-out)
    )

    assert isinstance(ax, Axes)
    drawn = [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 0]
    assert len(drawn) >= 2  # boundary + at least one drift/sample path
    # The collapsing boundary spans a positive y-range (the wedge).
    assert any(np.ptp(np.asarray(ln.get_ydata())) > 0 for ln in drawn)
