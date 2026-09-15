# Canonical response domains

**Status:** Draft architectural contract<br>
**Target:** HSSM `main`
**Issue:** [#1265](https://github.com/lnccbrown/HSSM/issues/1265)

## Goal

HSSM should describe each non-RT observation column in one canonical place. That
contract must support mixed categorical, continuous, and circular observations and
arbitrary fixed observation widths without model-specific branches.

The implementation should replace fixed-width assumptions, not surround them with a
second abstraction layer.

## Canonical representation

`Config.response` is the ordered list of physical observation-column names in the input
DataFrame. HSSM validates those names directly; it is not a semantic alias map.

`response_domains` annotates every declared response column other than `rt`, keyed by
those exact physical names. Its order is normalized to the model configuration's
`response` order rather than the caller's mapping insertion order. A domain kind never
implies a required column name: existing single-response models keep `response`, while a
custom model may use other distinct configured names.

For an established single-response model:

```python
import math

response = ["rt", "response"]
response_domains = {
    "response": {"kind": "circular", "bounds": (-math.pi, math.pi)},
}
```

For a response with two scalar coordinates:

```python
response = ["rt", "response1", "response2"]
response_domains = {
    "response1": {"kind": "continuous", "bounds": (0.0, math.pi)},
    "response2": {"kind": "circular", "bounds": (-math.pi, math.pi)},
}
```

Names such as `polar` and `azimuth` are also valid when they are the actual DataFrame
columns and appear in `Config.response`; they are never required merely because those
coordinate domains are used. The likelihood interprets coordinates through their
declared order and model contract, not by inspecting their names.

Each column has exactly one atomic kind:

- `categorical` declares distinct integer labels and compares observations without
  coercion;
- `continuous` may omit bounds or declare finite closed bounds `[lower, upper]`;
- `circular` declares finite half-open bounds `[lower, upper)`.

`mixed` is neither an atomic kind nor a new main-side global response kind. Multi-domain
consumers inspect `response_domains` directly.

RT retains HSSM's existing positivity, deadline, missing-data, and omission semantics.
It is deliberately not duplicated in `response_domains`.

The supported response layouts are deliberately narrow:

- an RT-based model declares `rt` exactly once at index zero, followed by one or more
  scalar response columns, one for each response coordinate;
- a choice-only model declares exactly one scalar response column and no `rt`;
- multi-column models without RT are deferred.

Packing multiple coordinates into one array- or object-valued `response` column is not
part of this contract.

## Invariants

1. At least one non-RT response column exists, and each appears exactly once. Missing and
   extra keys fail during configuration resolution.
2. Domain definitions are validated before model construction. Empty categorical value
   sets, duplicate values, invalid bounds, nonfinite bounds, and unknown fields fail
   closed.
3. HSSM has one normalization boundary. Legacy `choices` can create a categorical domain
   only when there is exactly one non-RT response column. Multiple response columns
   require canonical metadata.
4. Canonical metadata combined with explicitly supplied `choices` at a raw input
   boundary is ambiguous and fails. A resolved config may retain an exactly matching
   derived `choices` view so ordinary dataclass copying and validation remain
   idempotent; conflicting values fail. Resolved configs have no implicit choice
   default: built-in factories supply their existing labels explicitly. Resolved
   configs are construction snapshots, not mutable domain registries. Directly
   mutating nested canonical metadata is unsupported and a later validation fails if
   the derived view no longer matches; construct a replacement with canonical metadata
   and `choices=None` instead.
5. On `main`, `choices` is derived only for exactly one categorical domain. It is `None`
   for continuous, circular, or multiple domains, including multiple categorical
   domains.
6. The provisional `response_kind` and `response_bounds` fields remain dev-only. Final
   integration adapts them as input and retains atomic compatibility views only for
   existing single-domain models; multi-domain consumers use the canonical mapping.
7. Registered defaults, user configurations, constructor snapshots, and serialized
   models own detached copies of nested domain values.
8. Data validation, observation width, likelihood input order, predictive coordinates,
   and predictive DataFrame names derive from the resolved physical response declaration.
   There is no implicit rename layer, and they do not dispatch on model names.
9. Lapse/outlier behavior remains limited to the two established categorical layouts:
   RT plus one response, or one-column choice-only. Other layouts require
   `p_outlier=None` or zero until separately specified and tested.

## Compatibility promises

- Existing one- and two-column models retain their public response order, choices,
  predictive dimension names, coordinate values, seeded behavior, and serialized state.
- Existing single-response data retain the physical column name `response`; its
  categorical, continuous, or circular meaning comes from model configuration.
- Existing categorical model configurations need not be rewritten merely to adopt the
  internal canonical representation, and keep their exact integer `choices`.
- A new multi-response model must provide canonical per-column metadata; HSSM will not
  invent mixed domains from a global choice count or another homogeneous shortcut. Each
  scalar coordinate has a distinct configured DataFrame column, whose name may be
  generic (`response1`) or semantic (`polar`).
- Choice-only RL remains one-column in this track; choice-only RL and RT-based RL
  configurations retain their established public behavior.
- The final `main -> dev` integration normalizes the provisional global dev metadata and
  removes it as parallel internal state.

## Delivery boundary

The cumulative track covers:

1. canonical types, resolution, copying, and compatibility projections;
2. per-column data validation;
3. fixed observation widths one through four in likelihood, predictive, and save/load
   plumbing;
4. dependency-free synthetic end-to-end proofs and user documentation.

The following are explicitly separate work:

- JEAM models, dependencies, evidence, and scientific claims;
- ssm-simulators observation-metadata projection or dependency-floor changes;
- real SDM, HSDM, or PHSDM registration;
- plotting, KDE, LAN-training, and legacy `simulate_data` generalization;
- PSDM successor execution and response-variability work.

## Complexity guard

- Keep one canonical mapping, one resolver, one per-column validation path, and one
  observation-width propagation path.
- Prefer plain typed mappings and pure functions over new stateful objects or class
  hierarchies.
- Do not add model-name conditionals.
- Introduce a helper only when at least two real call sites share the rule.
- Prefer deleting or replacing a fixed-width branch over wrapping it.
- Preserve compatibility through derived views rather than duplicated mutable fields.
- Tests may be broad; production abstractions must remain narrow and independently
  justified.

If a slice requires a second source of truth, a model-specific exception, or a public
promise that its tests cannot exercise, the slice stops for redesign.
