import pytest

import hssm


@pytest.mark.xfail(
    reason="bambi 0.20 migration (#1305): R1 bambi 0.20 gives the 2-D `c(rt, response)` response only a 1-D `__obs__` dim",
    strict=False,
)
@pytest.mark.slow
def test_simple_graphing(data_ddm):
    model = hssm.HSSM(data=data_ddm, model="ddm")
    graph = model.graph()

    assert graph is not None
    # TODO: Test below is not crucial but should be reinstantiated
    # later when this gets addressed
    # assert all(f"{model._parent}_mean" not in node for node in graph.body)
