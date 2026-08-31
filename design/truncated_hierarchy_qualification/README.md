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

Even a numerical pass is conditional on the dependency floor exercised by the
JAX/NumPyro path. HSSM therefore requires JAX 0.11 or newer, where `erfcx` is native,
and PyTensor 3.2.4 or newer, where that native operation is used during JAXification.
Older combinations fall back to TensorFlow Probability; its stable release is not
compatible with the frozen JAX stack. TensorFlow Probability is deliberately not an
experiment dependency.

## Reproducibility contract

`benchmarks/specs/truncated_hierarchy_v2.json` is the executable, immutable,
reviewable manifest. The earlier v1 manifest is retained as the historical
pre-execution design, but its overloaded anchor labels are not admissible evidence.
V2 gives the prior hyper-location, fixed-truth location and scale, group indices,
data-generating contract, and initialization policy explicitly for every scenario.

The manifest uses strict JSON: `NaN`, positive infinity, and negative infinity are
invalid (an absent bound is `null`). Its semantic SHA-256 is embedded in every plan,
environment sidecar, and result. Scenario identifiers are never reused for different
models. A shared `data_seed` is derived with BLAKE2b from the master seed, `data_id`,
and replicate; domain-separated truth, group-value, and observation seeds are then
derived from that parent. Initialization, backend-default per-chain start, SBC-draw,
SBC-tie, and sampler seeds use their own frozen domains. HSSM-default cells consume
their initialization seed once, transform HSSM's processed `_initvals` once, and
replicate that identical start across chains. PyMC receives one explicit chain seed
per chain. NumPyro receives one explicit JAX master seed; the result records the exact
`PRNGKey` (one chain) or `jax.random.split` keys (multiple chains) used by the pinned
backend. Each candidate and its exact control deliberately share the truth and
observations while retaining independent initialization and sampler seeds. The
derivation does not use Python's process-randomized `hash()`.

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
layer, model, bounds, numeric truth, group dimensions, precision, sampler, budget,
and initialization policy. Both are fitted to the exact same natural-scale group
values and observations. Only the prior construction differs. Candidate recovery is
assessed against the generating family; linked controls are geometry and efficiency
references and therefore set `recovery=false` rather than making a false
cross-parameterization recovery claim.

Each non-SBC qualification candidate/control replicate executes as one atomic pair
on the same physical worker. Even replicates run candidate then control; odd
replicates reverse that order. Both members are attempted even if the first fails,
and each receives fresh phase-local compilation caches. Opaque pair, worker, and
per-cell attempt identities are recorded and cross-checked by the assessor, so
separately scheduled or order-confounded fits cannot qualify.

The candidate is the exact proposed response-scale hierarchy: a centered native
`TruncatedNormal` group distribution with a `TruncatedNormal` location hyperprior and
`Weibull(1.5, 0.3)` scale. The control is a centered Normal hierarchy on predictor
scale, mapped to response scale with `lower + exp(eta)`, `upper - exp(eta)`, or the
finite-interval generalized-logit inverse. These are deliberately different prior
families applied to the same likelihood and dataset.

The dedicated SBC cells instead draw truths and data from the candidate prior itself,
so they are candidate-family calibration cells and are not compared to a control
dataset. `purpose` records candidate/control membership; the orthogonal
`calibration_kind` field records that these cells run SBC. Primary HSSM cells use
the real `hssm-default` initialization policy, including the
support-aware default jitter supplied by the stacked implementation. There are no
hand-tuned primary starts. Direct PyMC and Bambi cells use `backend-default` starts.

Cells with the same `data_id` use exactly the same dataset seed. Candidate/control
pairs and PyMC/NumPyro pairs therefore target the same data-generating problem while
retaining independent chain seeds. A `posterior_pair_id` joins exactly one PyMC and
one NumPyro scenario with otherwise identical scientific settings; posterior
agreement is assessed from those paired results rather than inferred from different
simulated datasets.

For fixed-truth toy cells, the frozen group values are drawn once from
SciPy's standardized `truncnorm` using `group_seed`, then indexed by the explicit
`group_indices`; observations use the independent `observation_seed`. HSSM cells use
the corresponding frozen LBA2, DDM, or softmax data-generating equations. Candidate
and control never redraw the truth or observations independently. SBC uses
`truth_seed` first for the candidate location and then its Weibull scale, `group_seed`
for coefficients, and `observation_seed` for data. Cells with zero trials omit the
observed likelihood while retaining the declared latent group vector.

Fixed-truth recovery and simulation-based calibration answer different questions and
are never pooled. Five-replicate fixed-truth cells diagnose boundary bias and
recovery in each named regime. At five replicates, exact sign tests and Holm decisions
are descriptive only—the smallest possible two-sided sign-test p-value is 0.0625—so
the predeclared fixed-truth bias gate is magnitude-based and cannot claim
"reproducibility" from an unreachable significance threshold. Two direct-PyMC
candidate geometries additionally run
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

Canonical hosted execution is defined by
`.github/workflows/qualify_truncated_hierarchy.yml`. On a same-repository pull
request, adding the `run-truncated-hierarchy-qualification` label requests the full
study; the mandatory smoke assessment must pass before qualification or stress jobs
can start. Manual dispatch supports four explicit modes: `smoke`; `qualification`
(smoke then qualification); `stress` (smoke then diagnostic stress only); and `full`
(smoke then both later tiers). Cell artifacts use `tn-cells-<tier>-*`, profile
attestations use `tn-env-*`, and reviewed aggregate decisions use
`tn-summary-<tier>`. All jobs check out the exact requested SHA with read-only
permissions, and aggregation runs even when a shard fails so missing evidence stays
visible.

Before primary execution, capture one clean-checkout reference attestation from each
locked profile. Results name the attestation's semantic digest; evidence cannot be
reassigned to a different profile after the fact. Qualification
runs require CPU, one thread for each listed numerical runtime, and JAX x64 exactly
when `floatx=float64`. Every fresh worker recollects its environment and must match
the attested source commit, clean-tree state, project and lock hashes, exact packages,
Python minor version, and interpreter. The attestation job's Python patch version and
kernel image describe that reference environment but are not cross-job identity keys;
the effective per-cell PyTensor and JAX precision remains mandatory.

The sampling runner executes each cell in a fresh process with cell-local PyTensor and
JAX caches. Data and model construction are untimed. The timer covers one exact start
generation plus the sampler's compile/JIT, warmup, and retained draws; artifact I/O
and diagnostic probes are excluded. Gradient and backend-parity probes run afterward
in a separate fresh process/cache, so they cannot warm the timed sampler.

The runner writes a canonical shared dataset/truth artifact to
`data/<data_id>-r<replicate>.json`, the exact transformed starts to
`starts/<cell>.json`,
the standardized monitored posterior to `chains/<cell>.nc`, and one atomic result
object to `cells/<cell>.json`. Group locations and the complete group-effect vector
are on response scale; first/middle/last group coefficients are retained as named
recovery summaries. `group_scale` remains on its hierarchy's native scale: response
scale for the candidate and predictor scale for the linked control. The result records
SHA-256 digests of the exact data, start, and chain bytes. The chain also retains the
standardized divergence, energy, tree-depth, leapfrog-step, step-size, and acceptance
statistics needed to recompute every gate. A partial write can therefore never
masquerade as complete evidence. On failure, only artifacts completed before the
failing stage remain, and the final cell result records absent artifacts as `null`.

Aggregate cell results in canonical plan order; cells without a published result
become explicit `missing` rows and failed cells retain their stage, error type, and
message. Use `benchmark-runs/` for local raw traces, logs, starts, chains, and cell
files; that directory is intentionally ignored. Only the frozen specification and
later reviewed aggregate evidence belong in Git.

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
  --artifact-root /path/to/run \
  --output /path/to/aggregate/assessment.json
```

`plan` writes byte-stable JSONL and CSV plus an `environment.json` sidecar. The
sidecar records the manifest digest, runner schema, Git commit/branch/dirty state,
the `pyproject.toml` SHA-256, Python/platform, and exact versions (or explicit
absence) of HSSM, Bambi, PyMC, PyTensor, JAX, NumPy, and NumPyro. A result repeats the
planned identity and seeds, references its environment digest, declares `completed`
or `failed`, and contains only finite Boolean/numeric metrics. Its reserved per-cell
provenance records sampler, device, planned floatX, the observed PyTensor floatX and
JAX x64 state, execution time, and digests of the actual transformed sampler starts
and raw chain artifact (not merely
`model.initial_point()`). The chain artifact exposes the natural-scale
`group_location`, `group_scale`, the complete `group_effect` vector, and predeclared
first/middle/last recovery summaries with `chain` and `draw` dimensions. The assessor
uses the complete vector for group-wide R-hat and ESS gates, verifies that the named
summaries are exact slices of it, and recomputes paired-backend rank R-hat from the
hash-verified chain artifacts. Divergence count, retained draw count, and rate must
agree exactly. SBC selects exactly 100 retained draws by frozen SHA-256 scoring
without replacement; selection is independent of file order. Raw posterior traces
are workflow artifacts and are not committed to the repository.

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
efficiency collapse, or failure of a predeclared recovery or calibration gate is an
immediate no-go. Raising `target_accept`, selecting favorable seeds, or hand-tuning
starts cannot convert a failure into a pass.

Gradient evidence is recorded as raw maximum absolute and relative errors in the
transformed coordinates seen by NUTS. Those maxima are descriptive: the gate uses the
standard combined tolerance, evaluated for every coordinate and then maximized,
`abs(observed - reference) / (atol + rtol * max(abs(reference), abs(observed))) <= 1`.
The assessor applies the predeclared float32 or float64 finite-difference,
PyTensor/JAX, and Bambi-isomorphism tolerances to that normalized error. This avoids
falsely rejecting a numerically negligible absolute error on a near-zero gradient. A
worker may not replace those measurements with a self-reported pass Boolean. The
frozen finite-difference check uses a five-point central stencil at the first chain's
exact transformed start, including the log-Jacobian. Posterior summaries use every
retained draw with NumPy's linear quantiles and ArviZ's rank-R-hat, bulk/tail ESS, and
mean MCSE; only SBC rank computation subsamples the frozen 100 draws.

Missing cells or required metrics produce `incomplete`, never `pass`. An observed
execution failure or failed scientific check produces `fail`. Smoke can only produce
`screening-pass`; stress can only produce a diagnostic outcome. Only a complete,
passing qualification assessment sets `qualifies_default=true`.
