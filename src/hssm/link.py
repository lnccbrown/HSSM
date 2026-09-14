"""A class that extends bmb.Link to allow for more generalized links with bounds."""

import bambi as bmb
import numpy as np

HSSM_LINKS = {"gen_logit"}


class Link(bmb.Link):
    """Representation of a generalized link function.

    This object contains two main functions. One is the link function itself, the
    function that maps values in the response scale to the linear predictor, and the
    other is the inverse of the link function, that maps values of the linear predictor
    to the response scale.

    The great majority of users will never interact with this class unless they want to
    create a custom ``Family`` with a custom ``Link``. This is automatically handled for
    all the built-in families.

    Parameters
    ----------
    name
        The name of the link function. If it is a known name, it's not necessary to pass
        any other arguments because functions are already defined internally. If not
        known, ``inverse_link`` must be specified.
    link : optional
        A function that maps the response to the linear predictor. Known as the
        :math:`g` function in GLM jargon. It is optional for custom links because Bambi
        does not currently use it. It does not need to be specified when ``name`` is a
        known name.
    inverse_link : optional
        A function that maps the linear predictor to the response. Known as the
        :math:`g^{-1}` function in GLM jargon. For custom links it must be compatible
        with the active backend, which is currently PyMC. NumPy ufuncs such as
        ``np.exp`` qualify because they dispatch to symbolic operations on PyTensor
        tensors. It does not need to be specified when ``name`` is a known name.
    bounds : optional
        Bounds of the response scale. Only needed when ``name`` is ``gen_logit``.

    Examples
    --------
    Use any link name supported by Bambi:

    >>> import hssm
    >>> identity_link = hssm.Link("identity")

    A custom link requires a backend-compatible inverse; the forward function is
    optional:

    >>> import numpy as np
    >>> custom_log = hssm.Link(
    ...     "custom_log",
    ...     link=np.log,  # Optional: response -> linear predictor
    ...     inverse_link=np.exp,  # Required: linear predictor -> response
    ... )

    HSSM also provides a generalized logit for bounded response scales:

    >>> bounded_link = hssm.Link("gen_logit", bounds=(0.1, 0.9))
    """

    def __init__(
        self,
        name,
        link=None,
        inverse_link=None,
        bounds: tuple[float, float] | None = None,
    ):
        if name in HSSM_LINKS:
            self.name = name
            if name == "gen_logit":
                if bounds is None:
                    raise ValueError(
                        "Bounds must be specified for generalized log link function."
                    )
                self.link = self._make_generalized_logit_simple(*bounds)
                self.inverse_link = self._make_generalized_sigmoid_simple(*bounds)
        else:
            super().__init__(
                name=name,
                link=link,
                inverse_link=inverse_link,
            )

        self.bounds = bounds

    def _make_generalized_sigmoid_simple(self, a, b):
        """Make a generalized sigmoid inverse link with bounds a and b.

        ``np.exp`` dispatches to a symbolic operation on PyTensor tensors, so the
        returned function serves both numerical and backend use.
        """

        def invlink_(x):
            return a + ((b - a) / (1 + np.exp(-x)))

        return invlink_

    def _make_generalized_logit_simple(self, a, b):
        """Make a generalized logit link function with bounds a and b."""

        def link_(x):
            return np.log((x - a) / (b - x))

        return link_

    def __str__(self):
        """Return a string representation of the link function."""
        if self.name == "gen_logit":
            lower, upper = self.bounds
            return f"Generalized logit link function with bounds ({lower}, {upper})"
        return super().__str__()
