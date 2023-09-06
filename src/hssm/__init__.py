"""HSSM - Hierarchical Sequential Sampling Models."""

import logging
import sys

from .config import ModelConfig
from .datasets import load_data
from .defaults import show_defaults
from .hssm import HSSM
from .param import Param
from .prior import Prior
from .simulator import simulate_data
from .utils import set_floatX

_logger = logging.getLogger("hssm")
_logger.setLevel(logging.INFO)
handler = logging.StreamHandler(stream=sys.stdout)
_logger.addHandler(handler)

__version__ = "0.1.4"

__all__ = [
    "HSSM",
    "load_data",
    "ModelConfig",
    "Param",
    "Prior",
    "simulate_data",
    "set_floatX",
    "show_defaults",
]
