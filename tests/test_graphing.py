import hssm


def test_simple_graphing(data_ddm):
    model = hssm.HSSM(data=data_ddm, model="ddm")
    graph = model.graph()

    assert graph is not None
    assert all(f"{model._parent}_mean" not in node for node in graph.body)
