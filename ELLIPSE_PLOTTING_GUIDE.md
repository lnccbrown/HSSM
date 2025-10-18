# Ellipse Plotting for Quantile Probability Plots

## Overview

The quantile probability plotting functionality in HSSM now supports plotting bivariate confidence ellipses instead of (or in addition to) point clouds for posterior predictive samples. This provides a cleaner, more compact visualization of uncertainty in the quantile-probability space.

## New Features

### 1. Three Plotting Modes

You can now choose how to visualize posterior predictive samples:

- **`predictive_style="points"`** (default): Traditional scatter plot of all samples
- **`predictive_style="ellipse"`**: Confidence ellipses only for each quantile+condition group
- **`predictive_style="both"`**: Both points and ellipses together

### 2. Configurable Confidence Levels

- **`ellipse_confidence`**: Set the confidence level (0 to 1) for the ellipses
  - `0.50`: 50% of probability mass (median)
  - `0.68`: ~68% (roughly 1-sigma in 2D)
  - `0.95`: 95% confidence (default)
  - `0.99`: 99% confidence

### 3. Automatic Fallback

- Ellipses require sufficient data points to compute covariance
- **`ellipse_min_points`**: Minimum points needed per group (default: 5)
- Groups with fewer points are automatically skipped with a debug warning

### 4. Customizable Appearance

- **`ellipse_kwargs`**: Dictionary for customizing ellipse appearance
  - `facecolor`: Fill color (default: "none")
  - `edgecolor`: Border color (automatically matched to quantile)
  - `linewidth`: Border thickness (default: 1.5)
  - `alpha`: Transparency (default: 0.6)

## Usage Examples

### Basic Usage: Just Ellipses

```python
import hssm

# Load model and data
model = hssm.HSSM(...)
model.sample()

# Plot with ellipses instead of points
model.plot_quantile_probability(
    cond="stimulus",
    predictive_style="ellipse"
)
```

### Custom Confidence Level

```python
# Plot 68% confidence ellipses (roughly 1-sigma in 2D)
model.plot_quantile_probability(
    cond="stimulus",
    predictive_style="ellipse",
    ellipse_confidence=0.68
)
```

### Both Points and Ellipses

```python
# Show both for comparison
model.plot_quantile_probability(
    cond="stimulus",
    predictive_style="both",
    ellipse_confidence=0.95,
    pps_kwargs={"alpha": 0.1, "s": 5},  # Make points more transparent/smaller
    ellipse_kwargs={"linewidth": 2, "alpha": 0.8}  # Make ellipses more prominent
)
```

### Custom Ellipse Styling

```python
# Red ellipses with semi-transparent fill
model.plot_quantile_probability(
    cond="stimulus",
    predictive_style="ellipse",
    ellipse_confidence=0.95,
    ellipse_kwargs={
        "facecolor": "red",
        "alpha": 0.2,
        "edgecolor": "darkred",
        "linewidth": 2
    }
)
```

### Adjust Minimum Points Threshold

```python
# Require at least 10 points per group before drawing ellipse
model.plot_quantile_probability(
    cond="stimulus",
    predictive_style="ellipse",
    ellipse_min_points=10
)
```

### With Prior Predictive

```python
# Works with prior predictive too
model.plot_quantile_probability(
    cond="stimulus",
    predictive_group="prior_predictive",
    predictive_style="ellipse",
    ellipse_confidence=0.90
)
```

## Understanding Confidence Levels in 2D

**Important**: Confidence levels in 2D differ from 1D!

| Confidence | n_std | 1D Interpretation | 2D Reality |
|------------|-------|-------------------|------------|
| 39.3%      | 1.0σ  | 68% in 1D         | ~39% in 2D |
| 86.5%      | 2.0σ  | 95% in 1D         | ~86% in 2D |
| 98.9%      | 3.0σ  | 99.7% in 1D       | ~99% in 2D |

A 95% confidence ellipse in 2D contains 95% of the probability mass in the (proportion, rt) space.

## When to Use Ellipses vs Points

### Use Ellipses When:
- ✅ You have many posterior samples (cluttered point clouds)
- ✅ You want to show correlation structure between proportion and RT
- ✅ You need a cleaner, more publication-ready visualization
- ✅ You want to compare uncertainty across different conditions/quantiles

### Use Points When:
- ✅ You have few posterior samples
- ✅ You want to show individual sample variation
- ✅ The distribution is clearly non-Gaussian (ellipses assume Gaussian)
- ✅ You want to see outliers or multi-modal structure

### Use Both When:
- ✅ You want to validate the ellipse approximation
- ✅ For exploratory data analysis
- ✅ To show both individual samples and overall uncertainty

## Technical Details

### Ellipse Computation

For each quantile+condition group:
1. Extract all (proportion, rt) pairs
2. Compute mean and covariance matrix
3. Find eigenvalues and eigenvectors of covariance
4. Ellipse axes align with eigenvectors
5. Ellipse size scales with √(eigenvalues) × n_std
6. n_std determined from chi-square distribution with 2 DOF

### Mathematical Foundation

For a bivariate Gaussian with mean μ and covariance Σ:
- The confidence ellipse at level α contains α fraction of probability mass
- The scaling factor is: n_std = √(χ²(α, df=2))
- Where χ²(α, df=2) is the chi-square quantile function

### Robustness

The implementation handles edge cases:
- **Insufficient data**: Requires at least 3 points (configurable via `ellipse_min_points`)
- **Singular covariance**: Skips ellipse if covariance matrix is not positive definite
- **Automatic logging**: Debug/warning messages for skipped groups

## API Reference

### New Parameters in `plot_quantile_probability()`

```python
def plot_quantile_probability(
    model,
    cond: str,
    ...,
    predictive_style: Literal["points", "ellipse", "both"] = "points",
    ellipse_confidence: float = 0.95,
    ellipse_min_points: int = 5,
    ...,
    ellipse_kwargs: dict[str, Any] | None = None,
    ...,
)
```

#### Parameters

- **predictive_style** : `{'points', 'ellipse', 'both'}`, default='points'
  - How to visualize posterior predictive samples

- **ellipse_confidence** : `float`, default=0.95
  - Confidence level for ellipses (must be between 0 and 1)
  - Higher values = larger ellipses

- **ellipse_min_points** : `int`, default=5
  - Minimum number of points required per group to draw an ellipse
  - Groups with fewer points are skipped

- **ellipse_kwargs** : `dict`, optional
  - Additional keyword arguments for `matplotlib.patches.Ellipse`
  - Common options: `facecolor`, `edgecolor`, `linewidth`, `alpha`, `linestyle`

### Helper Functions (Internal)

```python
def _confidence_to_n_std(confidence: float, n_dim: int = 2) -> float:
    """Convert confidence level to number of standard deviations."""

def _compute_ellipse_params(
    df_group: pd.DataFrame,
    x_col: str,
    y_col: str,
    confidence: float = 0.95,
) -> dict | None:
    """Compute ellipse parameters from a group of points."""
```

## Examples Gallery

### Example 1: Compare Different Confidence Levels

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, conf in zip(axes, [0.50, 0.68, 0.95]):
    model.plot_quantile_probability(
        cond="stimulus",
        predictive_style="ellipse",
        ellipse_confidence=conf,
        title=f"{conf*100:.0f}% Confidence",
        ax=ax
    )
```

### Example 2: Multiple Conditions with Ellipses

```python
# Works seamlessly with faceting
model.plot_quantile_probability(
    cond="difficulty",
    col="block",
    row="subject_group",
    predictive_style="ellipse",
    ellipse_confidence=0.95
)
```

### Example 3: Publication-Ready Plot

```python
model.plot_quantile_probability(
    cond="stimulus",
    predictive_style="ellipse",
    ellipse_confidence=0.95,
    title="Quantile-Probability Plot with 95% Confidence Ellipses",
    xlabel="Response Proportion",
    ylabel="Response Time (s)",
    data_kwargs={"linewidth": 2.5, "marker": "D", "markersize": 8},
    ellipse_kwargs={
        "facecolor": "none",
        "linewidth": 2.0,
        "alpha": 0.7,
        "linestyle": "--"
    }
)
```

## Troubleshooting

### Ellipses Not Appearing

**Check:**
1. Ensure `predictive_style="ellipse"` or `"both"`
2. Check you have enough samples: `n_samples >= ellipse_min_points`
3. Look for warning messages in logs
4. Verify your data has variation in both dimensions

### Ellipses Look Wrong

**Possible causes:**
- Non-Gaussian distribution (ellipses assume Gaussian)
- Outliers affecting covariance estimate
- Very skewed data

**Solutions:**
- Use `predictive_style="both"` to see points and ellipse together
- Try transforming your data
- Consider using points instead

### Performance Issues with Many Groups

**Tip:** Ellipses are actually faster than plotting thousands of points!
- Ellipses: O(n_groups)
- Points: O(n_samples × n_groups)

## Implementation Notes

- Ellipse colors automatically match the quantile colors from seaborn palette
- Ellipses are drawn after points (if both) for proper layering
- Each ellipse is computed independently per quantile+condition group
- The implementation uses scipy.stats.chi2 for confidence level conversion
- Compatible with all existing quantile probability plot options

## Future Enhancements (Potential)

- [ ] Support for alternative ellipse fitting methods (robust estimators)
- [ ] Confidence bands along the quantile line
- [ ] Customizable ellipse fill gradients
- [ ] Animation support for exploring different confidence levels

---

**Version**: Added in HSSM v0.2.11
**Author**: Implementation based on bivariate Gaussian confidence regions
**Dependencies**: scipy.stats.chi2, matplotlib.patches.Ellipse
