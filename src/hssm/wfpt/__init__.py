"""
This module provides all functionalities related to the Wiener Firt-Passage Time
distribution.
"""

from .classic import WFPTClassic
from .lan import LAN
from .wfpt import WFPT

__all__ = ["LAN", "WFPT", "WFPTClassic"]
