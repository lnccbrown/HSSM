# Predict for new or unidentified participants

A hierarchical model fits one coefficient per level of a grouping factor, for
example one drift-rate intercept per `participant_id`. When you pass new data to
`sample_posterior_predictive(data=...)` or `log_likelihood(data=...)`, the
grouping value of each new observation decides how its coefficients are
obtained. There is no keyword to choose the strategy; HSSM and Bambi read it off
the data.

| Grouping value in `data`                 | Interpretation             | Coefficients used                                                                                     |
| ---------------------------------------- | -------------------------- | ----------------------------------------------------------------------------------------------------- |
| A level seen during fitting              | Known group                | That group's posterior draws                                                                          |
| Missing (`None`, `np.nan`, `pd.NA`)      | Unknown identity           | Those of a fitted group, drawn uniformly at random per observation and per posterior draw            |
| A non-missing level not seen in fitting  | New group                  | Fresh draws from the population-level prior with the fitted hyperparameters, shared by the group     |

## Choose the strategy through the data

Use a **missing value** when the observation comes from one of the fitted
participants but you do not know which. Every observation borrows a fitted
group's coefficients independently, so the predictive distribution is a mixture
over the fitted participants, and two unidentified observations need not agree.

Use an **unseen level** when the observation comes from a genuinely new
participant. Bambi generates that participant's coefficients from the fitted
population distribution (for a term like `1|participant_id` with a
`Normal(mu=0, sigma=...)` prior, from `Normal(0, sigma)` at each draw's
`sigma`), once per new level. All observations sharing the level share the
coefficients, and distinct unseen levels receive independent draws.

```python
import hssm
import numpy as np
import pandas as pd

model = hssm.HSSM(
    data=data,
    include=[{"name": "v", "formula": "v ~ 1 + (1|participant_id)"}],
)
model.sample()

new_data = pd.DataFrame(
    {
        "rt": [0.8, 0.9, 1.1, 1.0],
        "response": [1, -1, 1, 1],
        # 3 is a fitted participant, NaN is an unidentified one and
        # 101 is a participant that was not in the fitted data.
        "participant_id": pd.Series([3, np.nan, 101, 101], dtype="object"),
    }
)

model.sample_posterior_predictive(data=new_data)
model.log_likelihood(data=new_data)
```

Mixing the three kinds of observation in one frame is fine. Each row picks its
strategy from its own grouping value, but rows that share the same unseen level
still share one generated coefficient, as described above.

## Practical notes

- Any pandas representation of a missing value works: `np.nan` in a float or
  object column, `None` in an object column, or `pd.NA` in a nullable `Int64`
  column. Fitted levels are matched by value, so an integer id still resolves
  after the column has been cast to float by the presence of `np.nan`.
- Because both strategies involve random draws (of a donor group, or of new
  coefficients), repeated out-of-sample calls on the same posterior do not give
  identical trial-wise parameters for unknown or new groups. Fitted groups are
  deterministic given the posterior.
- The retired Bambi keyword `sample_new_groups` is not exposed by HSSM. Passing
  it to the underlying `bambi.Model.predict` raises a `FutureWarning` and has no
  effect.
