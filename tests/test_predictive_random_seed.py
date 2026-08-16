"""Tests for deterministic HSSM predictive-stream forwarding."""

import numpy as np
import xarray as xr

import hssm


def test_posterior_predictive_reuses_one_generator_across_safe_chunks(
    data_ddm,
    minimal_posterior_datatree,
    monkeypatch,
):
    """Safe-mode chunks should advance one stream rather than restart a seed."""
    model = hssm.HSSM(data=data_ddm, p_outlier=None)
    posterior = (
        minimal_posterior_datatree()["posterior"].isel(draw=np.zeros(12, dtype=int)).ds
    )
    posterior = posterior.assign_coords(draw=np.arange(12))
    traces = xr.DataTree.from_dict({"posterior": posterior})
    received_seeds = []

    def fake_predict(
        dt,
        kind,
        data,
        inplace,
        include_group_specific,
        random_seed=None,
    ):
        del kind, data, inplace, include_group_specific
        received_seeds.append(random_seed)
        draws = dt["posterior"].coords["draw"].values
        dt["posterior_predictive"] = xr.Dataset(
            {"response": (("chain", "draw"), np.zeros((1, len(draws))))},
            coords={"chain": [0], "draw": draws},
        )

    monkeypatch.setattr(model.model, "predict", fake_predict)

    result = model.sample_posterior_predictive(
        dt=traces,
        draws=12,
        safe_mode=True,
        inplace=False,
        random_seed=519,
    )

    assert result is not None
    assert result["posterior_predictive"].sizes["draw"] == 12
    assert len(received_seeds) == 2
    assert isinstance(received_seeds[0], np.random.Generator)
    assert received_seeds[0] is received_seeds[1]
