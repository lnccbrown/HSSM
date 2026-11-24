"""Graph regressions for structurally matched HSSM safe group priors."""

import numpy as np
import pandas as pd
import pytest

import hssm
from hssm.param.parameterization_check import find_disconnected_free_rvs

GROUP_SLOPES = {
    "blocktype2_num|participant_id",
    "task_num|participant_id",
    "blocktype2_num:task_num|participant_id",
}


def _discussion_948_data() -> pd.DataFrame:
    """Return the balanced synthetic 2x2 design used by the reproducer notebook."""
    rows = []
    for participant in range(2):
        for cell, (block, task) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
            rows.append(
                {
                    "rt": 0.45 + 0.03 * cell + 0.01 * participant,
                    "response": (-1, 1, 1, -1)[cell],
                    "blocktype2_num": block,
                    "task_num": task,
                    "participant_id": f"p{participant}",
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("noncentered", [True, False], ids=["noncentered", "centered"])
def test_discussion_948_safe_prior_graph_has_no_group_slope_means(noncentered):
    """Build both parameterizations without redundant or orphan group means."""
    formula = (
        "v ~ 1 + blocktype2_num*task_num + (1 + blocktype2_num*task_num|participant_id)"
    )

    model = hssm.HSSM(
        data=_discussion_948_data(),
        model="ddm",
        include=[{"name": "v", "formula": formula, "link": "identity"}],
        prior_settings="safe",
        z=0.5,
        p_outlier=0.0,
        noncentered=noncentered,
        process_initvals=False,
    )

    for term_name in GROUP_SLOPES:
        assert model.params["v"].prior[term_name].args["mu"] == 0.0

    group_mean_names = {
        name for name in model.pymc_model.named_vars if "|participant_id_mu" in name
    }
    assert group_mean_names == set()
    if noncentered:
        assert find_disconnected_free_rvs(model.pymc_model) == []


def test_discussion_948_common_design_is_full_rank():
    """Guard the reproducer against accidental fixed-effect rank deficiency."""
    from formulae import design_matrices

    data = _discussion_948_data()
    matrices = design_matrices("response ~ 1 + blocktype2_num*task_num", data=data)

    assert matrices.common is not None
    assert np.linalg.matrix_rank(matrices.common.design_matrix) == 4
