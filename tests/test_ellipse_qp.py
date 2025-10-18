"""Quick test of the ellipse plotting functionality for quantile probability plots."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the helper functions
import sys

sys.path.insert(0, "src")

from hssm.plotting.quantile_probability import (
    _confidence_to_n_std,
    _compute_ellipse_params,
)


def test_confidence_to_n_std():
    """Test the confidence to n_std conversion."""
    print("Testing confidence_to_n_std...")

    # Test known values
    assert abs(_confidence_to_n_std(0.393) - 1.0) < 0.01, (
        "1-sigma should be ~39.3% in 2D"
    )
    assert abs(_confidence_to_n_std(0.865) - 2.0) < 0.01, (
        "2-sigma should be ~86.5% in 2D"
    )
    assert abs(_confidence_to_n_std(0.989) - 3.0) < 0.01, (
        "3-sigma should be ~98.9% in 2D"
    )

    print("  ✓ All confidence conversions correct!")


def test_compute_ellipse_params():
    """Test ellipse parameter computation."""
    print("\nTesting _compute_ellipse_params...")

    # Create synthetic data with known covariance
    np.random.seed(42)
    mean = np.array([0.5, 1.0])
    cov = np.array([[0.01, 0.005], [0.005, 0.02]])

    # Generate points
    points = np.random.multivariate_normal(mean, cov, size=100)
    df = pd.DataFrame(points, columns=["x", "y"])

    # Compute ellipse params
    params = _compute_ellipse_params(df, "x", "y", confidence=0.95)

    assert params is not None, "Should compute params with sufficient data"
    assert len(params["center"]) == 2, "Center should be 2D"
    assert params["width"] > 0, "Width should be positive"
    assert params["height"] > 0, "Height should be positive"
    assert params["n_points"] == 100, "Should report correct number of points"

    print(f"  ✓ Ellipse parameters computed successfully")
    print(f"    Center: {params['center']}")
    print(f"    Width: {params['width']:.4f}, Height: {params['height']:.4f}")
    print(f"    Angle: {params['angle']:.2f}°")

    # Test with insufficient data
    df_small = df.head(2)
    params_small = _compute_ellipse_params(df_small, "x", "y")
    assert params_small is None, "Should return None with insufficient data"
    print("  ✓ Correctly handles insufficient data")


def test_ellipse_visual():
    """Create a visual test of the ellipse plotting."""
    print("\nCreating visual test...")

    np.random.seed(42)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (conf, ax) in enumerate(zip([0.50, 0.68, 0.95], axes)):
        # Generate data
        mean = np.array([0.5, 1.0])
        cov = np.array([[0.01, 0.005], [0.005, 0.02]])
        points = np.random.multivariate_normal(mean, cov, size=200)

        # Plot points
        ax.scatter(points[:, 0], points[:, 1], alpha=0.3, s=10, label="Data")

        # Compute and plot ellipse
        df = pd.DataFrame(points, columns=["x", "y"])
        params = _compute_ellipse_params(df, "x", "y", confidence=conf)

        if params:
            from matplotlib.patches import Ellipse

            ellipse = Ellipse(
                xy=params["center"],
                width=params["width"],
                height=params["height"],
                angle=params["angle"],
                facecolor="none",
                edgecolor="red",
                linewidth=2,
                alpha=0.8,
                label=f"{conf * 100:.0f}% confidence",
            )
            ax.add_patch(ellipse)
            ax.plot(
                params["center"][0],
                params["center"][1],
                "ro",
                markersize=8,
                label="Mean",
            )

        ax.set_xlim(0.3, 0.7)
        ax.set_ylim(0.6, 1.4)
        ax.set_aspect("equal")
        ax.set_title(f"{conf * 100:.0f}% Confidence Ellipse")
        ax.set_xlabel("Proportion")
        ax.set_ylabel("RT")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("test_ellipse_output.png", dpi=100, bbox_inches="tight")
    print("  ✓ Visual test saved to test_ellipse_output.png")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Ellipse Plotting Functionality")
    print("=" * 60)

    test_confidence_to_n_std()
    test_compute_ellipse_params()
    test_ellipse_visual()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
