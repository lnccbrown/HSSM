import pytest

from hssm.modelconfig import get_default_model_config
from hssm.defaults import default_model_config


@pytest.mark.flaky(reruns=2, reruns_delay=1)
def test_get_model_meta():
    for model, data in default_model_config.items():
        assert data == get_default_model_config(model)

    with pytest.raises(ValueError):
        get_default_model_config("non_existent_model")
