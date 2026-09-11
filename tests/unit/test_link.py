"""Tests for HSSM link functions."""

import bambi as bmb
import numpy as np
import pandas as pd
import pytensor.tensor as pt
import pytest
from pytensor.tensor.variable import TensorVariable

from hssm import HSSM, Link


@pytest.mark.parametrize(
    ("name", "response_values", "predictor_values"),
    [
        ("identity", np.array([-1.0, 0.0, 1.0]), np.array([-1.0, 0.0, 1.0])),
        ("log", np.array([0.25, 1.0, 4.0]), np.array([-1.0, 0.0, 1.0])),
        ("logit", np.array([0.2, 0.5, 0.8]), np.array([-1.0, 0.0, 1.0])),
    ],
)
def test_builtin_link_matches_bambi(name, response_values, predictor_values):
    """Delegate Bambi's built-in link behavior through the HSSM subclass."""
    expected = bmb.Link(name)
    link = Link(name)

    assert link.name == expected.name
    np.testing.assert_allclose(
        link.link(response_values), expected.link(response_values)
    )
    np.testing.assert_allclose(
        link.inverse_link(predictor_values), expected.inverse_link(predictor_values)
    )
    assert link.bounds is None


def test_builtin_link_retains_bounds_metadata():
    """Retain optional HSSM bounds after delegating construction to Bambi."""
    link = Link("identity", bounds=(-2.0, 3.0))

    assert link.bounds == (-2.0, 3.0)


def test_custom_link_matches_bambi_semantics():
    """Retain the forward and inverse functions supplied for a custom link."""
    link = Link(
        "log1p",
        link=np.log1p,
        inverse_link=np.expm1,
        bounds=(-1.0, np.inf),
    )

    assert link.name == "log1p"
    assert link.link is np.log1p
    assert link.inverse_link is np.expm1
    assert link.bounds == (-1.0, np.inf)
    response_values = np.array([0.0, 0.5, 2.0])
    np.testing.assert_allclose(
        link.inverse_link(link.link(response_values)), response_values
    )


def test_custom_link_forward_function_is_optional():
    """Accept a custom link that supplies only the inverse, as Bambi now does."""
    link = Link("log1p", inverse_link=np.expm1)

    assert link.name == "log1p"
    assert link.link is None
    assert link.inverse_link is np.expm1


def test_incomplete_custom_link_uses_bambi_validation():
    """Raise Bambi's validation error when the inverse link is missing."""
    with pytest.raises(
        ValueError,
        match=("Link name 'log1p' is not supported and 'inverse_link' is unspecified"),
    ):
        Link("log1p", link=np.log1p)


def test_generalized_logit_requires_bounds():
    """Keep the HSSM-specific generalized-logit bounds requirement."""
    with pytest.raises(
        ValueError,
        match="Bounds must be specified for generalized log link function",
    ):
        Link("gen_logit")


def test_generalized_logit_round_trip():
    """Keep generalized-logit forward and inverse transformations unchanged."""
    bounds = (-2.0, 3.0)
    response_values = np.array([-1.5, 0.0, 2.5])
    link = Link("gen_logit", bounds=bounds)

    transformed = link.link(response_values)

    assert link.bounds == bounds
    np.testing.assert_allclose(link.inverse_link(transformed), response_values)
    assert str(link) == "Generalized logit link function with bounds (-2.0, 3.0)"


def test_generalized_logit_inverse_is_pytensor_compatible():
    """Use the single generalized-logit inverse to build a symbolic graph."""
    link = Link("gen_logit", bounds=(-2.0, 3.0))
    symbolic_eta = pt.vector("eta")

    symbolic_parameter = link.inverse_link(symbolic_eta)

    assert isinstance(symbolic_parameter, TensorVariable)
    # `hssm.set_floatX` mutates `pytensor.config.floatX` process-wide, so match
    # the ambient dtype rather than assuming float64.
    predictor_values = np.array([-1.0, 0.0, 1.0], dtype=symbolic_eta.dtype)
    np.testing.assert_allclose(
        symbolic_parameter.eval({symbolic_eta: predictor_values}),
        link.inverse_link(predictor_values),
        rtol=1e-6,
    )


def test_custom_link_builds_symbolic_hssm_regression():
    """Use the custom inverse link with a symbolic HSSM predictor."""
    data = pd.DataFrame(
        {
            "rt": [0.4, 0.5, 0.6, 0.7],
            "response": [-1, 1, -1, 1],
            "x": [-1.0, -0.5, 0.5, 1.0],
        }
    )
    backend_inputs = []

    def inverse_link(value):
        backend_inputs.append(value)
        return pt.exp(value)

    link = Link("custom_log", link=np.log, inverse_link=inverse_link)

    model = HSSM(
        data=data,
        model="ddm",
        include=[{"name": "v", "formula": "v ~ 1 + x", "link": link}],
        prior_settings=None,
        z=0.5,
        p_outlier=0.0,
        process_initvals=False,
        initval_jitter=0,
    )

    assert model.params["v"].link is link
    assert model.model.family.link["v"] is link
    assert backend_inputs
    assert all(isinstance(value, TensorVariable) for value in backend_inputs)
