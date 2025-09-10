# ruff: noqa: D100
from functools import partial

import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm
import ssms

from hssm.distribution_utils import (
    make_distribution,
    make_hssm_rv,
    make_likelihood_callable,
)
from hssm.utils import decorate_atomic_simulator, ssms_sim_wrapper

network_path_extended_r_10 = (
    "shrink_spot_simple_extended_lan_64708af6803c11f0931d4b53c8b8bed9_model.onnx"
)
shrinkspot_2_logp_jax_op_simple = make_likelihood_callable(
    loglik=network_path_extended_r_10,
    # Because we are using a network trained to approximate a likelihood
    loglik_kind="approx_differentiable",
    backend="jax",  
    params_is_reg=[
        False,
        False,
        False,
        False,
        True,
        False,
        False,
    ],
    
    params_only=False,
)

simulator_with_predefined_settings = partial(
    ssms_sim_wrapper,
    simulator_fun=ssms.basic_simulators.simulator.simulator,
    model="shrink_spot_simple_extended",
)


decorated_simulator = decorate_atomic_simulator(
    model_name="shrink_spot_simple_extended", choices=[-1, 1], obs_dim=2
)(simulator_with_predefined_settings)


ShrinkSpotRV = make_hssm_rv(
    simulator_fun=decorated_simulator,
    list_params=["a", "z", "t", "ptarget", "pouter", "r", "sda"],
    lapse=bmb.Prior("Uniform", lower=0, upper=20),
)  # specify lapse distribution to add p_outlier

bounds = {
    ssms.config.model_config["shrink_spot_simple_extended"]["params"][i]: [
        ssms.config.model_config["shrink_spot_simple_extended"]["param_bounds"][0][i],
        ssms.config.model_config["shrink_spot_simple_extended"]["param_bounds"][1][i],
    ]
    for i in range(
        len(ssms.config.model_config["shrink_spot_simple_extended"]["params"])
    )
}

list_params = ssms.config.model_config["shrink_spot_simple_extended"]["params"]
list_params.append("p_outlier")

SimpleShrinkSpot2 = make_distribution(
    rv=ShrinkSpotRV,
    loglik=shrinkspot_2_logp_jax_op_simple,
    list_params=list_params,
    bounds=bounds,
    lapse=bmb.Prior("Uniform", lower=0, upper=20),
)


############################
p_trial_properties_sim = np.random.choice([-1, 1], size=500)
p_global = 6.0
pouter_trialwise = p_trial_properties_sim * p_global
print(len(p_trial_properties_sim))


data_hier_a = decorated_simulator(
    theta=dict(
        a=0.5, z=0.3, t=0.22, ptarget=p_global, pouter=pouter_trialwise, r=2.7, sda=0.8
    ),
    n_replicas=1,
    random_state=42,
)

data_pd_sim = pd.DataFrame(data_hier_a, columns=["rt", "response"])


with pm.Model() as model_1:
    # data
    y = pm.Data("y", data_pd_sim[["rt", "response"]])
    congruency_trialwise = pm.Data("congruency_trialwise", p_trial_properties_sim)

    # priors
    a = pm.Uniform("a", lower=0.3, upper=3.0)
    t = pm.Weibull("t", alpha=3.0, beta=0.5)
    z = pm.Beta("z", alpha=10.0, beta=10.0)
    r = pm.Uniform("r", lower=0.01, upper=0.05)
    sda = pm.Uniform("sda", lower=1, upper=3)
    ptarget = pm.Uniform("ptarget", lower=2.0, upper=5.5)
    pouter = pm.Deterministic("pouter", ptarget * congruency_trialwise)
    p_outlier = 0.05
    # likelihood term
    obs = SimpleShrinkSpot2(
        "obs",
        a=a,
        z=z,
        t=t,
        ptarget=ptarget,
        pouter=pouter,
        r=r,
        sda=sda,
        p_outlier=p_outlier,
        observed=y,
    )
pm.model_to_graphviz(model_1)