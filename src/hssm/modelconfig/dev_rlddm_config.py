"""Registration for `dev_rlddm`, contributed by HSSMCortex.

Blocked on the `ssm-simulators` release carrying the simulator; opened as a draft.
Spec: papers/ssm-theory/fontanesi-2019-rlddm.spec.md
"""

from .._types import DefaultConfig


def get_dev_rlddm_config() -> DefaultConfig:
    """Get the default configuration for the `dev_rlddm` model.

    Returns
    -------
    DefaultConfig
        Response variables, parameters, choices and likelihood specification.
    """
    return {
        "response": ["rt", "response"],
        "list_params": ['eta_pos', 'eta_neg', 'v_mod', 'v_max', 'a_fix', 'a_mod', 't_er', 'q_cor', 'q_inc', 'q_pres'],
        "choices": [-1, 1],
        "description": 'fontanesi-2019-rlddm',
        "likelihoods": {
            "approx_differentiable": {
                # NOT TRAINED. This registration is a draft: the likelihood approximator this
                # model needs does not exist yet, and training one is a modelling decision with a
                # cost, made by a person. Naming the file it would be is how the reviewer sees
                # what is outstanding; setting it to something that exists would be worse.
                "loglik": None,
                "backend": "jax",
                "default_priors": {},
                "bounds": {
                    'eta_pos': (0.01, 0.3),
                    'eta_neg': (0.01, 0.3),
                    'v_mod': (0.1, 2.0),
                    'v_max': (1.5, 6.0),
                    'a_fix': (0.3, 1.8),
                    'a_mod': (-0.05, 0.01),
                    't_er': (0.4, 1.1),
                    'q_cor': (27.5, 55.0),
                    'q_inc': (27.5, 55.0),
                    'q_pres': (27.5, 55.0),
                },
                "extra_fields": None,
            },
        },
    }
