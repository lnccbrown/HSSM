"""Utility functions for the parameter classes."""


def validate_bounds(bounds: tuple[float, float]) -> None:
    """Validate the bounds."""
    if len(bounds) != 2:
        raise ValueError(f"Invalid bounds: {bounds}")
    lower, upper = bounds
    if lower >= upper:
        raise ValueError(f"Invalid bounds: {bounds}")
