"""Reusable utilities for RL likelihood builders."""

import functools
from typing import Callable


def annotate_function(**kwargs):
    """Attach arbitrary metadata as attributes to a function.

    Parameters
    ----------
    **kwargs
        Arbitrary keyword arguments to attach as attributes.

    Returns
    -------
    Callable
        Decorator that adds metadata attributes to the wrapped function.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **inner_kwargs):
            return func(*args, **inner_kwargs)

        for key, value in kwargs.items():
            setattr(wrapper, key, value)
        return wrapper

    return decorator
