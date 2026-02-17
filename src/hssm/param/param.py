"""The parent class for all parameters in the model."""

from typing import Any

import bambi as bmb
import numpy as np

from .user_param import UserParam
from .utils import validate_bounds


class Param:
    """The parent class for all parameters.

    Parameters
    ----------
    name
        The name of the parameter.
    prior
        The prior specification for the parameter.
    formula
        The formula for the parameter.
    link
        The link function for the parameter.
    bounds
        The bounds for the parameter.
    user_param
        The parameter as specified by the user.

    Attributes
    ----------
    name
    prior
    formula
    link
    bounds
    user_param
    is_regression
        Whether the parameter is a regression.
    is_fixed
        Whether the parameter is fixed.
    is_trialwise
        Whether the parameter varies across observations.
    is_parent
        Whether the parameter is a parent parameter.

    Methods
    -------
    from_user_param
        Create a Param object from a UserParam object.
    fill_defaults
        Fill in the default values for the parameter.
    validate
        Validate the parameter.
    process_prior
        Process the prior specification
    parse_bambi
        Parse the parameter for Bambi.
    """

    name: str
    prior: float | np.ndarray | dict[str, Any] | bmb.Prior | None
    formula: str | None
    link: str | bmb.Link | None
    bounds: tuple[float, float] | None
    user_param: UserParam | None

    def __init__(
        self,
        name: str,
        prior: float | np.ndarray | dict[str, Any] | bmb.Prior | None = None,
        formula: str | None = None,
        link: str | bmb.Link | None = None,
        bounds: tuple[float, float] | None = None,
        user_param: UserParam | None = None,
    ) -> None:
        self.name = name
        self.prior = prior
        self.formula = formula
        self.link = link
        self.bounds = bounds
        self.user_param = user_param
        self._parent = False

        if self.bounds is not None:
            validate_bounds(self.bounds)

    @classmethod
    def from_user_param(cls, user_param: UserParam) -> "Param":
        """Create a Param object from a UserParam object.

        Parameters
        ----------
        user_param
            A UserParam object.

        Returns
        -------
        Param
            A Param object with the specified parameters.
        """
        return cls(
            **user_param.to_dict(),
            user_param=user_param,
        )

    @property
    def is_regression(self) -> bool:
        """Whether the parameter is a regression."""
        return self.formula is not None

    @property
    def is_fixed(self) -> bool:
        """Whether the parameter is fixed as a scalar or a vector."""
        return isinstance(self.prior, (int, float, np.ndarray))

    @property
    def is_parent(self) -> bool:
        """Whether the parameter is a parent parameter."""
        return self._parent

    @is_parent.setter
    def is_parent(self, value: bool) -> None:
        self._parent = value

    @property
    def is_trialwise(self) -> bool:
        """Whether the parameter varies across observations.

        Returns ``True`` when the parameter is the parent, a regression,
        or a fixed vector (``np.ndarray`` prior).

        NOTE on ``is_parent``:
        Bambi's ``model.predict()`` always returns ``(n_obs,)`` values for the
        parent parameter, even for intercept-only models --- it evaluates the
        full linear predictor for every observation.  The JAX/ONNX vmap layer
        (``LANLogpOp.perform`` / ``LANLogpVJPOp.perform``) and the post-hoc
        log-likelihood computation (``utils._compute_log_likelihood``) use this
        flag to set ``in_axes=0``, telling vmap that the input varies along the
        observation axis.

        Treating intercept-only parents as scalars (``in_axes=None``) would be
        incorrect because ``model.predict()`` is the canonical entry point for
        extracting parameter values and it *always* broadcasts to ``(n_obs,)``.
        Every downstream consumer --- plotting (``attach_trialwise_params_to_df``),
        log-likelihood evaluation, posterior predictive checks --- receives the
        parent through ``model.predict()``.  If ``is_trialwise`` returned
        ``False`` for the parent, the vmap specification would declare a scalar
        input while actually receiving an ``(n_obs,)`` array, causing a shape
        mismatch in JAX.  Fixing that would require either patching Bambi's
        ``predict()`` (upstream, not under our control) or adding a reduction
        step at every consumption site, which is fragile and error-prone.
        """
        return (
            self.is_parent or self.is_regression or isinstance(self.prior, np.ndarray)
        )

    def fill_defaults(
        self,
        prior: dict[str, Any] | None = None,
        bounds: tuple[float, float] | None = None,
        **kwargs,
    ) -> None:
        """Fill in the default values for the parameter.

        Parameters
        ----------
        prior
            The prior specification for the parameter.
        bounds
            The bounds for the parameter.
        """
        if self.bounds is None:
            self.bounds = bounds
        if self.prior is None:
            self.prior = prior
        for key, value in kwargs.items():
            if getattr(self, key) is None:
                setattr(self, key, value)

    def validate(self) -> None:
        """Validate the parameter."""
        raise NotImplementedError("This method is to be implemented in subclasses.")

    def process_prior(self) -> None:
        """Process the prior specification."""
        raise NotImplementedError("This method is to be implemented in subclasses.")

    def parse_bambi(
        self,
    ) -> tuple[str | None, dict[str, Any] | bmb.Prior | None, str | bmb.Link | None]:
        """Parse the parameter for Bambi.

        Returns
        -------
        tuple
            1. A bmb.Formula object, or None if the parameter is simple.
            2. A bmb.Prior object, in the simple case, or a dictionary of priors in
            the regression case, if any is specified, or None if the priors
            should be determined by Bambi.
            3. A string or bmb.Link object representing the link function, or None
            in the simple case.
        """
        return self.formula, self.prior, self.link
