"""Tests for model cartoon trajectory colors."""

import matplotlib.pyplot as plt
import numpy as np

from hssm.plotting.model_cartoon import _add_trajectories


def test_add_trajectories_accepts_color_lists():
    """Allow callers to provide per-choice trajectory colors as a list."""
    fig, ax = plt.subplots()
    sample = {
        0: {
            "metadata": {
                "possible_choices": [1.0, -1.0],
                "boundary": np.array([1.0, 1.0, 1.0, 1.0]),
                "t": np.array([0.0]),
                "trajectory": np.array([0.1, 0.2, -1000.0]),
            },
            "choices": np.array([[1.0]]),
        }
    }

    _add_trajectories(
        ax,
        sample,
        np.array([0.0, 0.1, 0.2]),
        n=1,
        colors=["green", "black"],
    )

    try:
        assert ax.lines[0].get_color() == "green"
    finally:
        plt.close(fig)
