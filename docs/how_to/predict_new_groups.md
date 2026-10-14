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
  deterministic given the posterior. Pass `random_seed` to `log_likelihood` to
  fix the generated coefficients and get a reproducible result.
- The retired Bambi keyword `sample_new_groups` is not exposed by HSSM. Passing
  it to the underlying `bambi.Model.predict` raises a `FutureWarning` and has no
  effect.

## Log likelihood for unknown and new participants

Bambi's own `compute_log_likelihood` rejects grouping values it has not seen.
HSSM does not: `log_likelihood(data=...)` first obtains every observation's
parameters from Bambi's prediction machinery, as described above, and then
evaluates the HSSM likelihood on them. Each value in the `log_likelihood` group
is therefore **conditional on the coefficients generated for that posterior
draw**:

- **New participant** (unseen, non-missing id): within a draw, all of the
  participant's observations are scored with the same generated coefficient. It
  is re-generated from draw to draw.
- **Unknown identity** (missing id): each observation borrows a donor group
  independently, per observation and per draw. Several rows with a missing id
  are not treated as one participant, even within a draw.

The generated coefficients are random, so the log likelihood of these rows
changes between calls. Pass `random_seed` to make it reproducible:

```python
dt = model.log_likelihood(data=new_data, inplace=False, random_seed=42)
```

### Averaging draws into a predictive score

The raw values have shape `(chain, draw, __obs__)`. A predictive score averages
the *likelihood* (not the log likelihood) over draws, and the result depends on
what you average:

- **Trial level.** Average each observation's likelihood over draws on its own.
  This scores every trial as if it were the only one observed from its
  participant, and it is the appropriate score for rows with a missing id.
- **Participant level.** Sum the log likelihood of a participant's observations
  within each draw first, then average the joint likelihood over draws. Because
  a new participant's coefficient is shared within a draw, this integrates over
  that coefficient once for all of the participant's trials, which gives the
  predictive score of the participant as a whole. Do not use it for rows with a
  missing id: their coefficients are not shared, so the rows do not form one
  participant.

The two scores differ in general, because the average of a product is not the
product of averages.

```python
import numpy as np
import xarray as xr
from scipy.special import logsumexp

ll = dt["log_likelihood"]["rt,response"].stack(sample=("chain", "draw"))
log_n = np.log(ll.sizes["sample"])

# Trial level: one score per observation.
trial_score = logsumexp(ll, axis=ll.get_axis_num("sample")) - log_n

# Participant level: one score per participant.
participant = xr.DataArray(new_data["participant_id"].to_numpy(), dims="__obs__")
joint = ll.groupby(participant).sum()
participant_score = logsumexp(joint, axis=joint.get_axis_num("sample")) - log_n
```

`groupby` drops rows with a missing id, so they do not enter the participant
level score; score them at the trial level instead.

Both are Monte Carlo estimates with one generated coefficient per posterior
draw, so they are noisy when the posterior has few draws. HSSM has no dedicated
participant-level scoring API; compute it as shown.
