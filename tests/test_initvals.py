import numpy as np
import pytest
import hssm
import logging

hssm.set_floatX("float32", update_jax=True)
logger = logging.getLogger("hssm")

parameter_names = "loglik_kind, model, sampler, initvals"
parameter_grid = [
    ("approx_differentiable", "ddm", "nuts_numpyro", "map"),
    ("analytical", "ddm", "nuts_numpyro", "map"),
    ("approx_differentiable", "angle", "nuts_numpyro", "map"),
    (
        "approx_differentiable",
        "ddm",
        "mcmc",
        "map",
    ),
    ("analytical", "ddm", "mcmc", "map"),
    (
        "approx_differentiable",
        "angle",
        "mcmc",
        "map",
    ),
    ("approx_differentiable", "ddm", "nuts_numpyro", "initial_point"),
    ("analytical", "ddm", "nuts_numpyro", "initial_point"),
    ("approx_differentiable", "angle", "nuts_numpyro", "initial_point"),
    ("approx_differentiable", "ddm", "mcmc", "initial_point"),
    ("analytical", "ddm", "mcmc", "initial_point"),
    ("approx_differentiable", "angle", "mcmc", "initial_point"),
]


@pytest.mark.slow
@pytest.mark.parametrize(parameter_names, parameter_grid)
def test_sample_map(caplog, loglik_kind, model, sampler, initvals):
    """Test sampling from MAP starting point."""
    logger.info(
        "\nTesting starting point setting at sampler level, \n"
        "for model=%s, loglik_kind=%s, sampler=%s, initvals=%s",
        model,
        loglik_kind,
        sampler,
        initvals,
    )
    cav_data = hssm.load_data("cavanagh_theta")
    caplog.set_level(logging.INFO)
    model_on = hssm.HSSM(
        data=cav_data,
        model=model,
        loglik_kind=loglik_kind,
        process_initvals=True,
    )

    initial_point = model_on.initial_point(transformed=True)

    if initvals == "initial_point":
        model_on.sample(
            sampler=sampler,
            initvals=initial_point,
            chains=1,
            cores=1,
            draws=10,
            tune=10,
        )
    if initvals == "map":
        model_on.sample(
            sampler=sampler, initvals=initvals, chains=1, cores=1, draws=10, tune=10
        )


def _check_initval_defaults_correctness(model) -> None:
    """Check if initial values from default dictionary are correctly applied."""
    # Consider case where link functions are set to 'log_logit'
    # or 'None'
    if model.link_settings not in ["log_logit", None]:
        return None

    # Set initial values for particular parameters
    for name_, starting_value in model._initvals.items():
        # If the user actively supplies a link function, the user
        # should also have supplied an initial value insofar it matters.
        if model.params[model._get_prefix(name_)].is_regression:
            param_link_setting = model.link_settings
        else:
            param_link_setting = None

        # Go through parameters that are specified in the initial value defaults
        # If not specified in there, we won't touch the parameter during post-processing
        # anyways
        if name_ in hssm.defaults.INITVAL_SETTINGS[param_link_setting]:
            # Figure out if user specified a custom initial value for the parameter

            # If yes, we need to check it this custom value successfully overrode our
            # global defaults
            # If not, we want to check if our defaults where successfully applied
            user_initval = model._check_if_initval_user_supplied(
                name_, return_value=True
            )

            if user_initval is not None:
                # If the user specified custom initial values for anything
                # in our INITVAL_DEFAULTS dictionary, we need to check if
                # the user's initial value was successfully applied
                model_initial_point = model._initvals[name_]
                assert np.allclose(
                    model_initial_point, user_initval, atol=1e-3
                ), f"""User supplied initial value for {name_} is {user_initval},
                    which does not match the initial point set by model,
                    which is {model_initial_point}"""
            else:
                # If the user did not specify custom initial values,
                # we need to check that our INITVAL_DEFAULTS
                # were successfully applied
                model_initial_point = model._initvals[name_]
                default_initial_point = hssm.defaults.INITVAL_SETTINGS[
                    param_link_setting
                ][name_]

                assert np.allclose(
                    model_initial_point,
                    default_initial_point,
                    atol=1e-3,
                ), f"""Initial value for {name_} is supposed to be {default_initial_point},
                       and does not match the initial point set by model,
                       which is {model_initial_point}."""
        else:
            pass


@pytest.mark.slow
def test_basic_model(caplog):
    """Test basic model with p_outlier distribution defined."""
    caplog.set_level(logging.INFO)
    logger.info("\nTesting most basic model.")
    cav_data = hssm.load_data("cavanagh_theta")
    model = hssm.HSSM(
        data=cav_data,
        model="ddm",
        process_initvals=True,
    )
    _check_initval_defaults_correctness(model)


@pytest.mark.slow
def test_basic_model_p_outlier(caplog):
    """Test basic model with p_outlier distribution defined."""
    caplog.set_level(logging.INFO)
    logger.info("\nTesting basic model with p_outlier distribution defined.")
    cav_data = hssm.load_data("cavanagh_theta")
    model = hssm.HSSM(
        data=cav_data,
        model="ddm",
        process_initvals=True,
        p_outlier={"name": "Uniform", "lower": 0.0001, "upper": 0.5},
    )
    _check_initval_defaults_correctness(model)


@pytest.mark.slow
def test_basic_model_p_outlier_initval(caplog):
    """Test basic model with p_outlier distribution defined."""
    caplog.set_level(logging.INFO)
    logger.info(
        """\nTesting basic model with p_outlier distribution
                and initval defined."""
    )
    cav_data = hssm.load_data("cavanagh_theta")
    model = hssm.HSSM(
        data=cav_data,
        model="ddm",
        process_initvals=True,
        p_outlier={"name": "Uniform", "lower": 0.0001, "upper": 0.5, "initval": 0.5},
    )
    _check_initval_defaults_correctness(model)


@pytest.mark.slow
def test_reg_model(caplog):
    """Test regression model, with regression on all parameters."""
    caplog.set_level(logging.INFO)
    logger.info("\nTesting regression model.")
    cav_data = hssm.load_data("cavanagh_theta")
    model = hssm.HSSM(
        data=cav_data,
        model="ddm",
        process_initvals=True,
        include=[
            {"name": "v", "formula": "v ~ 1 + (1|participant_id)"},
            {"name": "a", "formula": "a ~ 1 + (1|participant_id)"},
            {"name": "z", "formula": "z ~ 1 + (1|participant_id)"},
            {"name": "t", "formula": "t ~ 1 + (1|participant_id)"},
        ],
    )
    _check_initval_defaults_correctness(model)


@pytest.mark.slow
def test_reg_model_subset(caplog):
    """Test regression model, with subset of parameters being regressions."""
    caplog.set_level(logging.INFO)
    logger.info(
        "\nTesting regression model with subset of parameters being regressions."
    )
    cav_data = hssm.load_data("cavanagh_theta")
    model = hssm.HSSM(
        data=cav_data,
        model="ddm",
        process_initvals=True,
        include=[
            {"name": "v", "formula": "v ~ 1 + (1|participant_id)"},
            {"name": "a", "formula": "a ~ 1 + (1|participant_id)"},
        ],
    )


@pytest.mark.slow
def test_angle_model_reg(caplog):
    """Test with angle model regression."""
    caplog.set_level(logging.INFO)
    logger.info(
        """\nTesting regression model with subset of parameters being regressions,
        for angle model."""
    )
    cav_data = hssm.load_data("cavanagh_theta")
    model = hssm.HSSM(
        data=cav_data,
        model="angle",
        process_initvals=True,
        include=[
            {"name": "v", "formula": "v ~ 1 + (1|participant_id)"},
            {"name": "a", "formula": "a ~ 1 + (1|participant_id)"},
        ],
    )
    _check_initval_defaults_correctness(model)


@pytest.mark.slow
def test_angle_model(caplog):
    """Test with angle model basic."""
    caplog.set_level(logging.INFO)
    logger.info("\nTesting basic angle model.")
    cav_data = hssm.load_data("cavanagh_theta")
    model = hssm.HSSM(
        data=cav_data,
        model="angle",
        process_initvals=True,
    )
    _check_initval_defaults_correctness(model)


@pytest.mark.slow
def test_process_no_process(caplog):
    """Test mismatch with and without preprocessing."""
    caplog.set_level(logging.INFO)
    logger.info(
        """\nTesting that turning initval-processing off,
                doesn't change initial values."""
    )

    cav_data = hssm.load_data("cavanagh_theta")
    model_on = hssm.HSSM(
        data=cav_data,
        model="angle",
        process_initvals=True,
    )

    model_off = hssm.HSSM(
        data=cav_data,
        model="angle",
        process_initvals=False,
    )

    assert (
        model_on.initvals != model_off.initvals
    ), """Initial values should not be the same when
    initval processing is turned off vs. turned on."""
