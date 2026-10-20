"""Governed, reproducible experiment templates for public HSSM evidence.

The runner accepts only registered Tier-S templates and exact input schemas. It is
not a general execution surface and never evaluates caller-provided code.
"""

from .experiments import (
    ExperimentRequestError,
    available_templates,
    run_experiment,
    validate_request,
)

__all__ = [
    "ExperimentRequestError",
    "available_templates",
    "run_experiment",
    "validate_request",
]
