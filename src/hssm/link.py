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
        known, all of `link``, ``linkinv`` and ``linkinv_backend`` must be specified.
    link : optional
        A function that maps the response to the linear predictor. Known as the
        :math:`g` function in GLM jargon. Does not need to be specified when ``name``
        is a known name.
    linkinv : optional
        A function that maps the linear predictor to the response. Known as the
        :math:`g^{-1}` function in GLM jargon. Does not need to be specified when
        ``name`` is a known name.
    linkinv_backend : optional
        Same than ``linkinv`` but must be something that works with PyMC backend
        (i.e. it must work with PyTensor tensors). Does not need to be specified when
        ``name`` is a known name.
    bounds : optional
        Bounds of the response scale. Only needed when ``name`` is ``gen_logit``.
    """

    def __init__(
        self,
        name,
        link=None,
        linkinv=None,
        linkinv_backend=None,
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
                self.linkinv = self._make_generalized_sigmoid_simple(*bounds)
                self.linkinv_backend = self._make_generalized_sigmoid_simple(*bounds)
        else:
            bmb.Link.__init__(name, link, linkinv, linkinv_backend)

        self.bounds = bounds

    def _make_generalized_sigmoid_simple(self, a, b):
        """Make a generalized sigmoid link function with bounds a and b."""

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
