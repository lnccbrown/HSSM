"""Regression specifications shared by the integration test modules.

`V_REG` and `A_REG` were duplicated verbatim across ``test_mcmc.py``,
``test_vi.py``, ``test_missing_data_mcmc.py`` and ``test_missing_data_vi.py``.
They are plain module-level constants rather than fixtures because each test
module builds its ``MODEL_SHAPES`` table from them at import time, before any
fixture could run.
"""

V_REG = dict(
    formula="v ~ 1 + x + y",
    prior={
        "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
        "x": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
        "y": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
    },
)

A_REG = dict(
    formula="a ~ 1 + m + n",
    prior={
        "Intercept": {
            "name": "Normal",
            "mu": 1.0,
            "sigma": 0.5,
        },
        "m": {"name": "Uniform", "lower": 0.0, "upper": 0.2},
        "n": {"name": "Uniform", "lower": 0.0, "upper": 0.2},
    },
    link="identity",
)
