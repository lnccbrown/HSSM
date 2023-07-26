"""HSSM - Hierarchical Sequential Sampling Models."""

import logging
import sys

from .config import show_defaults
from .datasets import load_data
from .hssm import HSSM
from .simulator import simulate_data
from .utils import set_floatX

_logger = logging.getLogger("hssm")
_logger.setLevel(logging.INFO)
handler = logging.StreamHandler(stream=sys.stdout)
_logger.addHandler(handler)

__all__ = ["HSSM", "load_data", "simulate_data", "set_floatX", "show_defaults"]
