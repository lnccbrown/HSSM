# Hierarchical TruncatedNormal qualification

This directory owns the numerical release gate for the bounded group-location
defaults proposed in HSSM #1269. Construction tests prove that the intended graph
exists; this qualification asks the separate question of whether the graph has
correct gradients, usable NUTS geometry, and calibrated recovery.

This is the experiment stacked in HSSM #1282 on the implementation branch for #1269;
it is evidence for that proposed default, not part of the default implementation
itself.

The generated default must not ship until the qualification tier passes. A failure
must lead to a root fix and complete rerun, a narrower supported policy, or removal of
the automatic hierarchical `TruncatedNormal` default.

## Reproducibility contract

`benchmarks/specs/truncated_hierarchy_v1.json` is the immutable, reviewable manifest.
It uses strict JSON: `NaN`, positive infinity, and negative infinity are invalid (an
absent bound is `null`). Its semantic SHA-256 is embedded in every plan, environment
sidecar, and result. Scenario identifiers are never reused for different models.
Dataset and chain seeds are derived with BLAKE2b from the master seed, scenario
identifier, replicate number, purpose, and (for sampling) chain index. Each candidate
and its exact-dimension control deliberately share a dataset seed; their chain seeds
remain independent. The derivation does not use Python's process-randomized `hash()`.

The three tiers have distinct meanings:

- `smoke`: cheap construction, initialization, log-density, gradient, and short-NUTS
  screening. A pass is necessary but never qualifies the default.
- `qualification`: the predeclared release gate, including five fixed datasets for
  canonical HSSM cells, four chains, and 1,000 warmup plus 1,000 retained draws.
- `stress`: deliberately adverse geometry used to localize limitations. Stress
  failures are reported but do not silently redefine the primary gate.

Every `gate=primary` scenario participates in the release decision. This includes the
direct PyMC toy hierarchy, the same graph built through both Bambi 0.19 and the
resolved dependency stack, and complete HSSM models under PyMC and NumPyro.
`canonical=true` has a narrower
meaning: it marks the predeclared LBA2 and approximate-DDM HSSM sentinels for which
every one of the five fits must have exactly zero divergences. It does **not** exempt
non-canonical primary cells from the other per-fit and aggregate gates.

The direct PyMC lower-bound/outside-support and two-sided/near-boundary candidates
are qualified under both PyMC and NumPyro, with same-backend controls. The primary
matrix is fixed at `target_accept=0.90`. The `0.95` and `0.99` cells live only in the
stress tier and may explain a failure but cannot rescue it.

The approximate-DDM primary cells use the committed test network and its actual
training interval, `z in (0.1, 0.9)`. The broader configured `(0, 1)` interval is a
separate diagnostic cell because excursions outside the network's training box would
confound prior geometry with likelihood extrapolation. Likewise, the LBA2 cells must
first pass a PyTensor/JAX likelihood-value and gradient parity screen. A backend
likelihood disagreement is reported at that layer and is never relabelled as a
`TruncatedNormal` failure.

Every fixed-truth primary candidate has a one-to-one control matching its tier, gate,
layer, model, bound, truth regime, group dimensions, precision, sampler, budget,
recovery setting, and initialization policy. Only the prior construction differs.
The dedicated SBC cells instead draw truths and data from the candidate prior itself,
so they are not compared to a control dataset. Primary HSSM cells use the real
`default` initialization policy, including the
support-aware default jitter supplied by the stacked implementation. There are no
hand-tuned primary starts. A dedicated `no-jitter-gradient-screen` policy is reserved
for a future deterministic gradient-only screen and is not used by the primary gate.

Cells with the same `data_id` use exactly the same dataset seed. Candidate/control
pairs and PyMC/NumPyro pairs therefore target the same data-generating problem while
retaining independent chain seeds. A `posterior_pair_id` joins exactly one PyMC and
one NumPyro scenario with otherwise identical scientific settings; posterior
agreement is assessed from those paired results rather than inferred from different
simulated datasets.

Fixed-truth recovery and simulation-based calibration answer different questions and
are never pooled. Five-replicate fixed-truth cells diagnose boundary bias and
recovery in each named regime. Two direct-PyMC candidate geometries additionally run
275 prior-predictive SBC replicates. That count is prospectively powered: after
Bonferroni correction across ten parameter units and two interval levels, it is the
smallest count with at least 90% power to detect ten-percentage-point undercoverage
for both nominal 90% and 95% intervals. Coverage and rank-uniformity are assessed for
each scenario and monitored parameter separately; controls, other models, and easy
interior cases cannot dilute a failing near-boundary candidate.

Generate the exact run plan without sampling:

```bash
uv run python scripts/truncated_hierarchy_qualification.py validate

uv run python scripts/truncated_hierarchy_qualification.py plan \
  --tier smoke \
  --dependency-profile current-resolved \
  --output-dir /tmp/hssm-1282-smoke
```

Before primary execution, capture one clean-checkout environment sidecar from each
locked profile. Results name the semantic digest of the sidecar that produced them;
evidence cannot be reassigned to a different profile after the fact.

The later sampling runner writes one atomic JSON object per planned cell into a
dedicated `cells/` directory. Aggregate them in canonical plan order; cells without a
published result become explicit `missing` rows and failed cells retain their stage,
error type, and message:

Use `benchmark-runs/` for local raw traces, logs, transformed starts, and cell files;
that directory is intentionally ignored. Only the frozen specification and later
reviewed aggregate evidence belong in Git.

```bash
uv run python scripts/truncated_hierarchy_qualification.py aggregate \
  --tier qualification \
  --results-dir /path/to/cells \
  --environment /path/to/current-resolved/environment.json \
  --environment /path/to/bambi-0.19/environment.json \
  --output-dir /path/to/aggregate
```

Assess the aggregate JSONL against the frozen manifest:

```bash
uv run python scripts/truncated_hierarchy_qualification.py assess \
  --tier qualification \
  --results /path/to/aggregate/results.jsonl \
  --environment /path/to/current-resolved/environment.json \
  --environment /path/to/bambi-0.19/environment.json \
  --output /path/to/aggregate/assessment.json
```

`plan` writes byte-stable JSONL and CSV plus an `environment.json` sidecar. The
sidecar records the manifest digest, runner schema, Git commit/branch/dirty state,
the `pyproject.toml` SHA-256, Python/platform, and exact versions (or explicit
absence) of HSSM, Bambi, PyMC, PyTensor, JAX, NumPy, and NumPyro. A result repeats the
planned identity and seeds, references its environment digest, declares `completed`
or `failed`, and contains only finite Boolean/numeric metrics. Its reserved per-cell
provenance records sampler, device, floatX, and a digest of the actual transformed
sampler start (not merely `model.initial_point()`). Divergence count, retained draw
count, and rate must agree exactly. Raw posterior traces are workflow artifacts and
are not committed to the repository.

## Primary decision thresholds

At `target_accept=0.90`, canonical LBA2 and approximate-DDM cells must pass all five
datasets with zero divergences on mandatory backends. Hyperparameters require
`R-hat < 1.01`, bulk and tail ESS of at least 400, BFMI of at least 0.30,
tree-depth saturation below 0.1%, and `MCSE / posterior SD <= 0.05`. Group
coefficients require maximum `R-hat < 1.01`, with at least 95% attaining both ESS
thresholds.

Other primary cells require an aggregate divergence rate below 0.1%, no individual
fit at or above 1%, at least 95% passing fits, and no scenario failing twice. Median
hyperparameter ESS/second and leapfrog cost must remain within fivefold of the paired
scientifically admissible control. The assessor derives these ratios from raw
candidate and control metrics; a cell cannot self-report that it passed. Recovery and
SBC checks have their separately predeclared interval/rank requirements in the
manifest. Comparators are data, not
implicit code conventions: for example, `R-hat < 1.01` rejects exactly `1.01`, and
divergence rate `< 0.001` rejects exactly `0.001`.

Supported-backend compilation failure, non-finite gradients, repeated
`R-hat > 1.05`, ESS below 100, divergence rate at or above 1%, greater-than-tenfold
efficiency collapse, or reproducible recovery bias is an immediate no-go. Raising
`target_accept`, selecting favorable seeds, or hand-tuning starts cannot convert a
failure into a pass.

Gradient evidence is recorded as raw maximum absolute and relative errors in the
transformed coordinates seen by NUTS. The assessor applies the predeclared float32 or
float64 finite-difference, PyTensor/JAX, and Bambi-isomorphism tolerances. A worker may
not replace those measurements with a self-reported pass Boolean.

Missing cells or required metrics produce `incomplete`, never `pass`. An observed
execution failure or failed scientific check produces `fail`. Smoke can only produce
`screening-pass`; stress can only produce a diagnostic outcome. Only a complete,
passing qualification assessment sets `qualifies_default=true`.
