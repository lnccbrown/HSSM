import pytest

from hssm.defaults import get_default_model_meta, default_model_config

def test_get_model_meta():
    for model, data in default_model_config.items():
        assert data == get_default_model_meta(model)

    with pytest.raises(ValueError):
        get_default_model_meta("non_existent_model")