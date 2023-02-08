"""
This module provides all functionalities related to the Wiener Firt-Passage Time
distribution.
"""

from .wfpt import (
    WFPT,
    make_distribution,
    make_family,
    make_ssm_distribution,
    make_wfpt_rv,
)

__all__ = [
    "make_wfpt_rv",
    "make_distribution",
    "make_ssm_distribution",
    "make_family",
    "WFPT",
]
