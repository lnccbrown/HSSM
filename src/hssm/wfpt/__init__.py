"""
This module provides all functionalities related to the Wiener Firt-Passage Time
distribution.
"""

from .wfpt import (
    WFPT,
    make_distribution,
    make_family,
    make_lan_distribution,
    make_model_rv,
)

__all__ = [
    "make_model_rv",
    "make_distribution",
    "make_lan_distribution",
    "make_family",
    "WFPT",
]
