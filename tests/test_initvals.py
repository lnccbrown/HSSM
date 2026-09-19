import numpy as np
import pytest
import hssm
import logging

from hssm.param.utils import _clamp_default_initval_to_bounds

hssm.set_floatX("float32", update_jax=True)
logger = logging.getLogger("hssm")

parameter_names = "loglik_kind, model, sampler, initvals"
parameter_grid = [
    ("approx_differentiable", "ddm", "numpyro", "map"),
    ("analytical", "ddm", "numpyro", "map"),
    ("approx_differentiable", "angle", "numpyro", "map"),
    (
        "approx_differentiable",
        "ddm",
        "pymc",
        "map",
    ),
    ("analytical", "ddm", "pymc", "map"),
    (
        "approx_differentiable",
        "angle",
        "pymc",
        "map",
    ),
    ("approx_differentiable", "ddm", "numpyro", "initial_point"),
    ("analytical", "ddm", "numpyro", "initial_point"),
    ("approx_differentiable", "angle", "numpyro", "initial_point"),
    ("approx_differentiable", "ddm", "pymc", "initial_point"),
    ("analytical", "ddm", "pymc", "initial_point"),
    ("approx_differentiable", "angle", "pymc", "initial_point"),
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
            progressbar=False,
        )
    if initvals == "map":
        model_on.sample(
            sampler=sampler,
            initvals=initvals,
            chains=1,
            cores=1,
            draws=10,
            tune=10,
            progressbar=False,
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


@pytest.mark.parametrize(
    ("link_settings", "expected_link", "expected_initval"),
    [(None, "identity", 1.5), ("log_logit", "log", 0.0)],
)
def test_valid_link_settings_preserve_regression_initvals(
    cavanagh_test, link_settings, expected_link, expected_initval
):
    """Both valid presets retain their documented deterministic starts."""
    model = hssm.HSSM(
        data=cavanagh_test.iloc[:12],
        include=[{"name": "a", "formula": "a ~ 1 + theta"}],
        v=0.0,
        z=0.5,
        t=0.2,
        p_outlier=0.0,
        prior_settings=None,
        link_settings=link_settings,
        process_initvals=True,
        initval_jitter=0.0,
    )

    assert model.params["a"].link == expected_link
    np.testing.assert_allclose(model._initvals["a_Intercept"], expected_initval)


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


def test_default_initval_clamped_into_bounds():
    """A default initval outside the declared bounds is moved inside them."""
    # Below the lower bound -> moved inside by 5% of the bound width.
    clamped_low = _clamp_default_initval_to_bounds(0.025, "t", (0.25, 2.25))
    assert clamped_low == 0.25 + 0.05 * 2.0

    # Above the upper bound -> moved inside.
    clamped_high = _clamp_default_initval_to_bounds(5.0, "a", (0.3, 2.5))
    assert 0.3 < clamped_high < 2.5

    # Inside the bounds -> returned unchanged.
    assert _clamp_default_initval_to_bounds(0.4, "t", (0.25, 2.25)) == 0.4

    # No bounds declared -> returned unchanged.
    assert _clamp_default_initval_to_bounds(0.025, "t", None) == 0.025


@pytest.mark.parametrize(
    ("bounds", "value"),
    [
        ((0.0, np.inf), -1.0),
        ((0.3, np.inf), 0.025),
        ((-np.inf, 1.0), 5.0),
    ],
)
def test_default_initval_clamped_into_one_sided_bounds(bounds, value):
    """Bounds with an infinite endpoint still yield a finite interior value.

    Shipped configs declare one-sided bounds - a: (0, inf) and t: (0, inf) in
    the analytical likelihoods, sz/st: (0, inf) in full_ddm - and a user may
    merge their own. A margin proportional to an infinite width would return
    +/-inf, which is a worse starting value than the unclamped default.
    """
    lower, upper = bounds
    result = _clamp_default_initval_to_bounds(value, "t", bounds)
    assert np.isfinite(result)
    assert lower < result < upper


def test_doubly_infinite_bounds_leave_finite_default_untouched():
    """A doubly-infinite bound already contains every finite default."""
    assert _clamp_default_initval_to_bounds(0.0, "v", (-np.inf, np.inf)) == 0.0


def test_clamp_warns_naming_the_parameter_and_replacement(caplog):
    """Moving a default is announced, so a surprising start is traceable."""
    caplog.set_level(logging.WARNING, logger="hssm")

    _clamp_default_initval_to_bounds(0.025, "t", (0.25, 2.25))

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "Default initial value 0.025 for t" in message
    assert "outside the declared bounds (0.25, 2.25)" in message
    assert "using 0.35 instead" in message
    assert "Pass an explicit initval to override." in message


@pytest.mark.parametrize(
    ("model", "name", "bounds"),
    [
        # A finite two-sided bound excluding the shared default t = 0.025.
        ("ddm", "t", (0.25, 2.0)),
        ("ddm_sdv", "t", (0.25, 2.0)),
        ("angle", "t", (0.25, 2.0)),
        # A one-sided bound, whose infinite width still yields a finite start.
        ("ddm", "t", (0.3, np.inf)),
        # A bound that already contains the default, which stays untouched.
        ("ddm", "t", (0.0, 2.0)),
        # An _Intercept name resolves to its base parameter's bounds.
        ("ddm", "t_Intercept", (0.25, 2.0)),
    ],
)
def test_declared_bounds_reach_the_initial_value(cavanagh_test, model, name, bounds):
    """Bounds passed through ``include=`` land on the resulting initial value.

    ``include=[{"name": ..., "bounds": ...}]`` stores bounds on the ``Param``,
    not on ``model_config``, and it is the only route by which a user of a
    shipped model can declare a bound that excludes a default initval.
    """
    lower, upper = bounds
    spec: dict = {"name": "t", "bounds": bounds}
    if name.endswith("_Intercept"):
        spec["formula"] = "t ~ 1"

    fitted = hssm.HSSM(
        data=cavanagh_test.iloc[:12],
        model=model,
        include=[spec],
        p_outlier=0.0,
        prior_settings=None,
        link_settings=None,
        process_initvals=True,
        initval_jitter=0.0,
    )

    assert fitted.params["t"].bounds == bounds
    initval = fitted._initvals[name]
    assert np.isfinite(initval)
    assert lower < initval < upper

    # A default that is already inside its bounds is passed through as-is.
    default = hssm.defaults.INITVAL_SETTINGS[None][name]
    if lower < default < upper:
        assert initval == np.array(default).astype(initval.dtype)


def test_identity_link_override_clamps_into_bounds(caplog, cavanagh_test):
    """An explicit identity link puts the default on the natural scale.

    ``link_settings="log_logit"`` is model-wide, but a regression may override
    the link for one parameter. Under identity the default is natural-scale, so
    the declared bounds apply to it and it is clamped into them.
    """
    model = hssm.HSSM(
        data=cavanagh_test,
        model="ddm",
        link_settings="log_logit",
        initval_jitter=0,
        include=[
            {
                "name": "t",
                "formula": "t ~ 1 + stim",
                "link": "identity",
                "bounds": (0.25, 2.0),
            }
        ],
    )
    assert getattr(model.params["t"].link, "name", model.params["t"].link) == "identity"
    initval = float(np.asarray(model._initvals["t_Intercept"]))
    # the link-space default of -4.0 would be far outside these bounds
    assert 0.25 < initval < 2.0
    assert initval == pytest.approx(0.25 + 0.05 * (2.0 - 0.25))


def test_user_log_link_gets_link_space_default(cavanagh_test):
    """A regression's own log link selects the link-space default.

    The model-wide ``link_settings`` is ``None`` here, so the natural-scale
    table would have applied before; the parameter's effective link decides.
    """
    model = hssm.HSSM(
        data=cavanagh_test,
        model="ddm",
        link_settings=None,
        initval_jitter=0,
        include=[{"name": "a", "formula": "a ~ 1 + stim", "link": "log"}],
    )
    assert getattr(model.params["a"].link, "name", model.params["a"].link) == "log"
    # the log-space default, i.e. a = exp(0) = 1, not the natural-scale 1.5
    assert float(np.asarray(model._initvals["a_Intercept"])) == 0.0
