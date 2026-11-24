"""aDDM fixation-continuation: config validation + per-call PPC policy override.

The continuation policy for posterior predictive is a per-call knob on
aDDM.sample_posterior_predictive (continuation_mode / continuation_params). It
rewrites entries on the RV class attr the generative rng_fn reads at draw time, so
one fitted model can be swept across policies with no rebuild/re-fit.

Requires the ssm-simulators build with ``cssm.addm`` AND the ``fixation_continuation``
module; skips otherwise (like the aDDM PPC tests).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_addm_subclass import make_addm_dataframe  # noqa: E402

try:
    from ssms.basic_simulators.fixation_continuation import (  # noqa: F401
        FIXATION_CONTINUATION_MODES,
    )
    from ssms.config import model_config as _ssms_model_config

    _HAS_ADDM_CONT = "addm" in _ssms_model_config
except Exception:  # pragma: no cover - old ssm-sim build
    _HAS_ADDM_CONT = False

needs_addm_continuation = pytest.mark.skipif(
    not _HAS_ADDM_CONT,
    reason="ssm-simulators build without aDDM fixation-continuation support",
)

import hssm  # noqa: E402
from hssm.addm.config import aDDMConfig  # noqa: E402

_GAMMA = {"dist": "gamma", "dist_params": {"a": 2.0, "scale": 0.3}}


@needs_addm_continuation
def test_addm_config_validates_continuation():
    """aDDMConfig.validate accepts good continuation settings and rejects bad ones."""
    aDDMConfig(
        continuation_mode="sample_continuation", continuation_params=_GAMMA
    ).validate()
    aDDMConfig(
        continuation_mode="resample_all_fixations", continuation_params=_GAMMA
    ).validate()
    aDDMConfig().validate()  # default (prolong_last_fixation, None) is valid

    for bad in (
        dict(continuation_mode="nope"),
        dict(continuation_mode="sample_continuation"),  # missing params
        dict(
            continuation_mode="sample_continuation",
            continuation_params={"dist": "not_a_dist"},
        ),
        dict(continuation_mode="prolong_last_fixation", continuation_params=_GAMMA),
    ):
        with pytest.raises(ValueError):
            aDDMConfig(**bad).validate()


@needs_addm_continuation
@pytest.mark.slow
def test_addm_ppc_per_call_continuation_override():
    """One fitted model: the per-call policy threads to the RV conduit and reverts.

    Each sample_posterior_predictive call also runs the generative simulator, so this
    exercises all three continuation modes end-to-end.
    """
    model = hssm.aDDM(data=make_addm_dataframe(30, seed=3))
    idata = model.sample(
        draws=10, tune=10, chains=1, cores=1, idata_kwargs={"log_likelihood": False}
    )
    rv_cls = type(model.model_distribution.rv_op)

    # Config default -> prolong_last_fixation on the conduit.
    model.sample_posterior_predictive(idata, inplace=True)
    assert rv_cls._extra_fields["continuation_mode"] == "prolong_last_fixation"
    assert rv_cls._extra_fields["continuation_params"] is None

    # Per-call override reaches the RV conduit (and runs the simulator without error).
    for mode in ("sample_continuation", "resample_all_fixations"):
        model.sample_posterior_predictive(
            idata, inplace=False, continuation_mode=mode, continuation_params=_GAMMA
        )
        assert rv_cls._extra_fields["continuation_mode"] == mode
        assert rv_cls._extra_fields["continuation_params"]["dist"] == "gamma"

    # The finally-block cleared the per-call override — the actual leak guard;
    # without it the next default call could not revert.
    assert model._continuation_override is None

    # A later call with no override reverts to the config default (no leakage).
    model.sample_posterior_predictive(idata, inplace=False)
    assert rv_cls._extra_fields["continuation_mode"] == "prolong_last_fixation"
    assert rv_cls._extra_fields["continuation_params"] is None
