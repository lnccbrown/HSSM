"""Test the import behavior of default_model_config."""

from hssm.register import register_model
from hssm.defaults import default_model_config as config1

# Register a new model
register_model(
    name="test_model",
    list_params=["param1"],
    loglik_kind="analytical",
    description="Test model for import behavior"
)

# Import again and verify it's the same object
from hssm.defaults import default_model_config as config2

print("Are the configs the same object?", config1 is config2)
print("Does config2 have our new model?", "test_model" in config2)

# Import in a different way and verify again
import hssm.defaults
print("Third import has our model?", "test_model" in hssm.defaults.default_model_config)
