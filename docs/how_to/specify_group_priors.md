# Specify hierarchical group priors

Group-specific regression terms describe how coefficients vary across
participants, items, or another grouping factor. Their priors are hierarchical:
the distribution for the group coefficients must contain at least one prior-valued
argument, usually a prior on its scale.

For most models, start with `prior_settings="safe"`. HSSM then chooses compatible
defaults from the formula structure. Supply an explicit group prior when the
scientific model requires a different family, scale, or population location.

## Identify the population-location owner

When a common and a group-specific term have the same Formulae expression, the
common term owns the population effect and the group term is a zero-mean deviation:

```python
import hssm

participant_deviation = hssm.Prior(
    "Normal",
    mu=0.0,
    sigma=hssm.Prior("HalfNormal", sigma=0.5),
    noncentered=True,
)

model = hssm.HSSM(
    data=data,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + x + (0 + x | participant_id)",
            "prior": {"x|participant_id": participant_deviation},
        }
    ],
)
```

Here `x` is the population slope and `x|participant_id` describes participant
deviations around it. Estimating a second free `mu` for the group term would make
only the sum of the common and group locations identifiable in the likelihood.

Without a matching common `x`, one group distribution can own the population
location. Keep that prior effectively centered so its `mu` remains part of the
model:

```python
participant_location = hssm.Prior(
    "Normal",
    mu=hssm.Prior("Normal", mu=0.0, sigma=0.5),
    sigma=hssm.Prior("HalfNormal", sigma=0.5),
    noncentered=False,
)

model = hssm.HSSM(
    data=data,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + (0 + x | participant_id)",
            "prior": {"x|participant_id": participant_location},
        }
    ],
)
```

The per-prior `noncentered=False` override takes precedence over a component or
model-level `noncentered=True` setting. Generated safe priors apply this centered
fallback automatically for a unique group-only location.

When that generated owner is an **intercept on an identity link**, HSSM also
uses the parameter's configured bounds when it can do so without changing an
HDDM-calibrated prior family. A generic parameter with at least one finite bound
receives a native hierarchical `TruncatedNormal`: both its generated population
location and its group coefficients stay inside the configured interval, and
the group term is centered so Bambi retains that hierarchy. A pure formula such
as `b ~ 0 + (1 | participant_id)` therefore keeps the complete predictor inside
the bounds.

This generated-safe behavior is deliberately narrower than general constraint
propagation. It does not apply to slopes, matching zero-mean deviations,
transformed links, or explicit priors. Analytical and black-box DDM families
keep their calibrated response-scale Gamma, Beta, or Normal hierarchies when
those families already match the built-in parameter support.

If the same unmatched expression occurs under several grouping factors, do not
give every group distribution a free location. Add the exact common expression
and use zero-mean group deviations, or deliberately choose exactly one group term
as the location owner. HSSM rejects ambiguous generated defaults and warns when
several explicit centered priors leave a location ridge.

## Know the current compatibility boundary

HSSM validates explicit group priors before asking Bambi to build the PyMC model.
The relevant rules are:

| Specification | Current behavior |
| --- | --- |
| Prior with no prior-valued top-level argument | Rejected; a group prior must be hierarchical |
| Numeric regression-term value | Rejected; it does not fix a group coefficient |
| Effectively non-centered plain `Normal` | Supported only with hierarchical `sigma`, absent or all-zero `mu`, no truncation or custom distribution, and no extra arguments |
| Free or nonzero group `mu` | Use `noncentered=False` so the requested location is retained |
| Hierarchical non-Normal or custom outer family | Use `noncentered=False` |
| Explicit `hssm.Prior(..., bounds=...)` on a group term | Rejected under either parameterization; HSSM's custom truncated wrapper cannot satisfy Bambi's group-hyperprior contract. This is distinct from the native named `TruncatedNormal` hierarchy HSSM generates for the bounded safe case above. |

These checks do not rewrite explicit priors. HSSM raises when continuing would
either fail in Bambi or silently construct a different prior tree. The same
effective-parameterization check applies recursively when a hyperprior itself has
prior-valued arguments.

## Treat links and coefficient bounds separately

All common and group coefficients first combine on the linear-predictor scale,
and the inverse link is applied afterward. Bounding one identity-linked group
intercept therefore does **not** guarantee that the full predictor remains inside
the parameter's support once slopes and other effects are added. HSSM warns when
it generates a bounded group-location coefficient in such a mixed predictor.

Use a support-respecting transformed link when it matches the model:

- `log` maps the full predictor to a positive parameter;
- `gen_logit` maps it between finite lower and upper bounds;
- transformed-link group priors live on the unconstrained predictor scale and
  should not receive response-scale bounds.

If an identity link is scientifically required, choose a centered hierarchical
family with appropriate natural support when possible, and remember that this
constrains that coefficient rather than the entire predictor. For values outside
configured likelihood bounds, HSSM substitutes a finite per-trial log-likelihood
floor (`-66.1`); this is a penalty with a flat region, not a hard-support prior.
A transformed link is the mechanism that constrains the *complete* additive
predictor when its inverse image matches the parameter support.

For the underlying scale and location logic, continue with [Link functions and
safe priors](../tutorials/link_functions.ipynb). For the general prior interface,
see [Specify priors and fix parameters](specify_priors.ipynb).
