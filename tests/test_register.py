"""Test the model registration functionality."""

import numpy as np
import pandas as pd
from hssm.register import register_model, list_registered_models, get_model_info
import hssm
import pdb


# Create a simple likelihood function
def my_loglik(rt, response, param1, param2):
    return -np.sum((rt - param1) ** 2 + (response - param2) ** 2)


# Create sample data
n_trials = 100
data = pd.DataFrame(
    {
        "rt": np.abs(np.random.normal(0.8, 0.2, n_trials)),
        "response": np.where(np.random.binomial(1, 0.7, n_trials) == 1, 1, -1),
    }
)

# Register a new model
pdb.set_trace()
register_model(
    name="my_custom_model",
    description="A custom model for testing",
    loglik_kind="approx_differentiable",
    include=[{"name": "a", "formula": "a ~ 1 + theta + (1|participant_id)"}],
    process_initvals=False,
    initval_jitter=0.5,
)


# List all registered models
print("\nRegistered models:")
for name, desc in list_registered_models().items():
    print(f"- {name}: {desc}")

# Get detailed info about our model
print("\nModel details:")
print(get_model_info("my_custom_model"))

# Create an HSSM instance with the new model
pdb.set_trace()
new_model = hssm.HSSM(
    model="my_custom_model",
    data=data,
    choices=[-1, 1],
)


register_model(
    name="my_custom_model_other",
    list_params=["param1", "param2"],
    loglik=my_loglik,
    loglik_kind="analytical",
    description="Another custom model for testing",
    default_priors={
        "param1": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
        "param2": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
    },
    bounds={
        "param1": (0, 10),
        "param2": (-2, 2),
    },
)
