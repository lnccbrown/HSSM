# Model configuration and response domains

`hssm.ModelConfig.response` is the ordered sequence of **physical DataFrame
columns** that make up one observation. The names are never semantic aliases:
each name must be present as its own scalar column in `data`. Built-in models
with one non-RT response keep the physical column name `response`; declaring a
different domain does not rename that column.

`response_domains` annotates every non-RT response column by that exact physical
name. Do not include `rt` in this mapping: response-time positivity, deadline,
and missing-data handling continue to use the existing RT contract.

## Domain specifications

| Kind | Required metadata | Accepted values |
|------|-------------------|-----------------|
| `categorical` | `values` | A non-empty list or tuple of distinct integer labels. Observations must match one of them exactly. |
| `continuous` | Optional `bounds` | Without bounds, any finite value. With finite, increasing bounds, both endpoints are included: `[lower, upper]`. |
| `circular` | `bounds` | Finite, increasing bounds defining a half-open interval: `[lower, upper)`. |

The mapping keys must match the non-RT entries in `response` exactly. HSSM
normalizes mapping insertion order to the order declared by `response`.

## Custom RT and two-coordinate example

The domain mapping below deliberately lists `azimuth` before `polar`. The
resolved order is still `polar`, then `azimuth`, because `response` is the sole
source of truth for observation order.

```python
import numpy as np
import pandas as pd

from hssm import ModelConfig

model_config = ModelConfig(
    response=("rt", "polar", "azimuth"),
    list_params=["v"],
    response_domains={
        "azimuth": {
            "kind": "circular",
            "bounds": (-np.pi, np.pi),
        },
        "polar": {
            "kind": "continuous",
            "bounds": (0.0, np.pi),
        },
    },
)

data = pd.DataFrame(
    {
        "rt": [0.42, 0.73],
        "polar": [0.80, 1.20],
        "azimuth": [-2.40, 0.90],
    }
)
```

HSSM uses the normalized `("rt", "polar", "azimuth")` order for likelihood
observations and predictive output labels. Custom predictive generators must
return exactly `len(response)` scalars in that declared order. HSSM checks a
callable generator's declared width against `len(response)`, but it cannot infer
the semantic meaning of each position. Authors who pass a prebuilt random
variable or distribution own both its width and order contract.

## Migrating from `choices`

`choices` remains a legacy shorthand only for a configuration with exactly one
categorical non-RT response. Canonical `ModelConfig` callers declare
`response_domains` and omit `choices`; never provide both. The current
`register_model` signature retains the compatibility argument, so canonical
registrations pass `choices=None` alongside `response_domains`.

After resolution, a derived `choices` view exists only when there is exactly one
response domain and it is categorical. Continuous, circular, mixed, and
multiple-domain configurations have no derived `choices` view.

## Current limits

- In a multicolumn RT layout, `rt` must appear exactly once and first.
- A choice-only model supports exactly one scalar non-RT response column.
- RT-less multicolumn responses are not supported.
- Each coordinate must occupy its own scalar DataFrame column; packed array- or
  object-valued response cells are not supported.
- No built-in model currently combines different response-domain kinds.
- `p_outlier` must be `None` or `0` outside the established one-categorical
  response layouts.
- Plotting, KDE utilities, and the legacy `hssm.simulate_data` interface are not
  generalized to wider mixed-response layouts.

## API reference

::: hssm.ModelConfig
