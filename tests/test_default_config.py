"""Test updating default_model_config at runtime."""

import numpy as np
import pandas as pd
from copy import deepcopy
from hssm.defaults import default_model_config
import hssm
from random import seed, choice

seed(42)


# Create some sample data
n_trials = 100
# Generate responses as -1 or 1
responses = np.where(np.random.binomial(1, 0.7, n_trials) == 1, 1, -1)

data = pd.DataFrame({
    'rt': np.abs(np.random.normal(0.8, 0.2, n_trials)),  # RTs must be positive
    'response': responses,
})

# Get an existing model config as template (using DDM)
_choice = choice(list(default_model_config.keys()))
md_config = default_model_config[_choice]
default_model_config['new_model1'] = deepcopy(md_config)

_choice = choice(list(default_model_config.keys()))
md_config = default_model_config[_choice]
default_model_config['new_model2'] = deepcopy(md_config)

# Now create an HSSM instance with the new model
model1 = hssm.HSSM(
    model='new_model1',   # our new model name
    data=data,           # DataFrame with rt and response columns
    choices=[-1, 1],     # binary choice model (-1, 1)
)

model2 = hssm.HSSM(
    model='new_model2',   # our new model name
    data=data,           # DataFrame with rt and response columns
    choices=[-1, 1],     # binary choice model (-1, 1)
)
