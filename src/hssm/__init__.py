"""Top-level entry to the HSSM package.

The `hssm` module is the top-level entry to the HSSM package. It exports some of the
most important classes, including the `HSSM` class that handles model creation and
sampling. You will also find utility classes such as `hssm.Prior`, `hssm.ModelConfig`,
and `hssm.Param`, with which users will often interface. Additionally, most frequently
used utility functions can also be found.
"""

import importlib.metadata
import logging
import sys

from .config import ModelConfig
from .datasets import load_data
from .defaults import show_defaults
from .hssm import HSSM
from .link import Link
from .param import Param
from .prior import Prior
from .simulator import simulate_data
from .utils import set_floatX

_logger = logging.getLogger("hssm")
_logger.setLevel(logging.INFO)
handler = logging.StreamHandler(stream=sys.stdout)
_logger.addHandler(handler)

__version__ = importlib.metadata.version(__package__ or __name__)

__all__ = [
    "HSSM",
    "Link",
    "load_data",
    "ModelConfig",
    "Param",
    "Prior",
    "simulate_data",
    "set_floatX",
    "show_defaults",
]
