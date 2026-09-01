# Hierarchical TruncatedNormal causal experiment

This directory defines the follow-up experiment for the hierarchical
`TruncatedNormal` sampling failures observed by the frozen v2 qualification. It is a
research contract, not an HSSM default-policy change. The general prior-construction
work and this causal investigation deliberately remain separate; a later policy
change may use the evidence produced here, but this experiment does not modify a
default by itself.

The frozen v2 result establishes three useful facts:

1. the failure is not confined to HSSM or Bambi, because a direct PyMC Gaussian
   hierarchy failed the smoke divergence screen;
2. the tested PyTensor and JAX gradients agreed locally, so the historical failure
   is not already explained by a demonstrated wrong gradient at those points; and
3. a linked-Normal hierarchy can be a practical alternative, but it changes the
   prior and therefore cannot identify the cause of a same-model failure.

V3 answers the narrower causal question: when the natural probability model is held
fixed, which density implementation or centering choice changes sampling health?

## Frozen inputs and independence from v2

[`truncated_hierarchy_causal_v3.json`](../../benchmarks/specs/truncated_hierarchy_causal_v3.json)
is the only executable design. It has a new schema, study identifier, seed domain,
artifact namespace, runner, and assessor. It does not import or extend the v2
validator. The v2 manifest remains immutable and is protected by an exact-byte
SHA-256 assertion. V3 additionally verifies that both copied regimes still match
their named v2 source scenarios field by field.

Replicate zero reuses the exact v2 data, truth, group, and observation seeds. Later
replicates use a new BLAKE2b domain with explicit purpose separation. Python's
process-randomized `hash()` is never used.

The two regimes are the exact direct-PyMC smoke failures:

| Regime | Bounds | Prior anchor | Truth | Data | Precision |
| --- | --- | --- | --- | --- | --- |
| `lower-outside-weak` | `[0.2, +inf)` | `0.0` | location `0.23`, scale `0.3` | 4 groups × 2 observations, observation SD `0.5` | float64 |
| `two-sided-near` | `[0.1, 0.9]` | `0.5` | location `0.13`, scale `0.3` | 4 groups × 25 observations, observation SD `0.5` | float32 |

They are never pooled. Support shape, anchor conflict, information, and precision all
differ, so a pooled label would make the causal conclusion ambiguous.

## Same natural model, five representations

Every fit defines exactly this natural-scale hierarchy:

```text
group_location ~ TruncatedNormal(prior_anchor, 0.25, bounds)
group_scale ~ Weibull(alpha=1.5, beta=0.3)
group_effect[g] ~ TruncatedNormal(group_location, group_scale, bounds)
y[i] ~ Normal(group_effect[group_index[i]], 0.5)
```

Only computational coordinates and density implementation change:

| Representation | Location | Groups | Role |
| --- | --- | --- | --- |
| `native-centered` | centered | centered | native PyMC reference |
| `manual-centered` | centered | centered | independently normalized density and Jacobians |
| `group-icdf-noncentered` | centered | inverse-CDF non-centered | C/NC factorial cell |
| `location-icdf-noncentered` | inverse-CDF non-centered | centered | NC/C factorial cell |
| `full-icdf-noncentered` | inverse-CDF non-centered | inverse-CDF non-centered | NC/NC factorial cell |

The fifth, NC/C representation is essential. Four forms would omit one cell of the
2×2 centering intervention, making a location main effect indistinguishable from an
interaction. Native and manual C/C occupy the same factorial cell; their difference
isolates native density/graph behavior from centering geometry.

The inverse-CDF forms are exact reparameterizations, not linked-Normal controls. A
standard-Normal offset is mapped through its CDF to a uniform quantile and then
through the independently implemented truncated-Normal inverse CDF. The induced
natural prior remains the same.

## Matrix and execution blocks

Both sampler paths are mandatory:

- PyMC NUTS through PyTensor;
- NumPyro NUTS through PyTensor-to-JAX.

The smoke tier has one replicate, two chains, 250 warmup draws, and 250 retained
draws. Its 20 fits validate construction, replay of the same frozen failing dataset
and regime, initialization, artifact flow, and short-chain execution. Natural-start
and sampler seeds are newly derived in the v3 domain, so this is not a literal replay
of the original v2 chains or their exact failure. Smoke cannot support a causal
conclusion: the assessor reports `screening-only` for each regime and no causal label,
even if a one-replicate health pattern happens to match a confirmation rule.

The confirmation tier has eight replicates, four chains, 1,000 warmup draws, and
1,000 retained draws. It contains 160 fits. Eight paired datasets are the smallest
grid on which an all-directional two-sided exact sign test reaches `0.0078125`, below
the predeclared Bonferroni threshold `0.05 / 6 = 0.008333...`. The six stochastic
comparisons are five conceptual within-backend contrasts—native versus manual and
the four factorial simple effects—each treated as an intersection-union test that
must pass in the frozen direction on both backend strata, plus one block-level
backend omnibus. Representation-specific backend contrasts are descriptive only.
The deterministic density/derivative oracle is a correctness gate, not a seventh
stochastic hypothesis. This is an inferentially
defensible minimum for near-deterministic effects, not a high-power design for small
effects.

Both tiers fix `target_accept=0.90` and maximum tree depth 10. Raising
`target_accept`, extending warmup after seeing a result, or selecting favorable seeds
cannot rescue a primary classification.

Each logical regime × backend × replicate block still contains all five
representations, but the scheduler's indivisible worker unit is the pair of PyMC and
NumPyro blocks for one regime × replicate: ten cells total. The paired placement is
necessary for the backend omnibus; otherwise a backend label would be confounded with
different hosted workers. The two backend blocks share a worker identity and one
auditable pair-execution identity. Backend order rotates by
`(replicate + regime index) modulo 2`, so the two smoke workers use opposite orders
and each confirmation regime runs four pairs in each order.

Within a backend block, representation order rotates by
`(4 × replicate + 2 × regime index + backend index) modulo 5`. The four logical smoke
blocks therefore start at four distinct positions, and confirmation traverses all
five positions. A cell failure does not suppress the other nine attempts. Every cell
still runs in a fresh child process with fresh PyTensor and JAX caches.

## Shared data and starts

All forms and both backends consume byte-identical immutable data for a given tier,
regime, and replicate. Data generation is completed and hash-bound before model
construction.

Starts are also an intervention control. For every chain, the runner builds the
native centered model, calls PyMC's per-chain initial-point generator with all free
variables jittered uniformly from −1 to 1 in transformed coordinates, and uses the
predeclared chain seed. It maps that point to natural `group_location`,
`group_scale`, and every `group_effect` exactly once. Each representation then maps
the shared natural point into its own coordinates and must round-trip it back to
natural scale within the frozen float32 or float64 tolerance.

PyMC and NumPyro receive the same natural starts. Sampling uses `adapt_diag` with
those exact mapped coordinates and no second jitter. This removes backend-default
initialization as a hidden intervention while retaining ordinary diagonal mass-matrix
adaptation. Chains have distinct start points.

Sampler randomness remains backend-specific and explicit. For PyMC 6.3.1, the
planned integer sequence is `SeedSequence` entropy, not one directly consumed seed
per chain. PyMC spawns one `numpy.random.Generator(PCG64)` per chain, draws one
integer below `2**30` for initial-point/NUTS-step construction, and then samples with
the advanced generators. Provenance records the entropy input and, for every chain,
the spawn key, pool size, initialization/step integer, and exact post-draw generator
state hash. NumPyro receives one scalar seed, and provenance records the exact uint32
JAX `PRNGKey` or `jax.random.split` key for every chain.

## Deterministic correctness gate

Sampling geometry is interpreted only after each graph passes the independent B1
oracle. At fixed grid points, every shared start, and hash-selected posterior
trajectory points, the runner compares transformed log density, gradient, and
Hessian. The gate uses the componentwise combined error

```text
abs(observed - reference)
-----------------------------------------------
atol + rtol * max(abs(reference), abs(observed))
```

and requires the maximum to be at most one. Value, gradient, and Hessian tolerances
are frozen separately for float32 and float64 in the manifest. The gate also checks
the natural-coordinate round trip, tail finiteness, and inverse-CDF branch
continuity.

The evidence count is executable rather than a claimed nonzero scalar. Before
sampling it is `chains + int(replicate == 0) + 5 × non-centered ICDF layers`: the
shared starts, the replicate-zero fixed point, and five branch/tail points for each
non-centered layer. A completed cell adds one hash-selected posterior-trajectory
point per chain. Failures after the pre-sampling oracle preserve that evidence:
`compile` and `sample` failures have no chain, while a `diagnose` failure may retain
the sampled chain and the complete pre-sampling oracle. A `summarize` failure occurs
after the full trajectory oracle and preserves that full phase. Whenever raw oracle
diagnostics are retained, the complete registered oracle metrics are retained too;
raw-oracle/summary mismatches are contract errors. This permits correctness evidence
to survive any later scientific failure. Sampling-based labels still require a
completed fit and full posterior-trajectory evidence.

Aggregation does not trust the runner's oracle summaries. It recomputes the scaled
value, gradient, and Hessian errors from the retained observed/reference arrays and
re-derives the natural-coordinate round-trip error, tail finiteness, and branch
continuity. Every oracle record retains its coordinate vector and natural values.
Aggregation maps that vector back to natural scale and binds fixed truth to the data
artifact, every start to both shared-natural and representation-coordinate start
artifacts, each ICDF probe to the exact chain-zero construction, and each trajectory
probe to the independently selected lowest-SHA draw in the natural chain artifact.
It also requires the exact frozen fixed/start/trajectory point IDs and, for each
non-centered layer, exactly one left/zero/right branch triplet and low/high tail pair
at the expected coordinate. Missing, relabelled, duplicated, self-consistent but
non-minimal trajectory selections, forged round-trip summaries, or summary-only
oracle evidence fail the contract.

A native-only value, gradient, or Hessian mismatch on at least two distinct dataset
replicates, with corresponding replicate/backend manual C/C evidence agreeing, is
sufficient evidence of a native PyMC correctness defect. Counting the two backend
paths on one dataset twice is not reproducibility. Sampling success is unnecessary;
sampling failure, round-trip failure, or divergences alone are never treated as a
density/derivative mismatch. Conversely, oracle agreement does not guarantee usable
NUTS geometry; it only rules out the measured deterministic mismatch.

## Health gates

Confirmation health applies the existing v2 thresholds to each regime × backend ×
representation family, never across easy and difficult cells:

- compilation, initialization, finite log density, finite gradient, and sampling
  must succeed;
- per-fit divergence rate must be below 1%, and the eight-fit aggregate rate below
  0.1%;
- rank-normalized split R-hat must be below 1.01;
- bulk and tail ESS must be at least 400 for both hyperparameters and for at least
  95% of group effects;
- BFMI must be at least 0.30 for every chain;
- tree-depth saturation must be below 0.1%; and
- hyperparameter MCSE divided by posterior SD must be at most 0.05.

The family fit-pass fraction is at least 95%. With eight replicates that means 8/8;
the manifest therefore explicitly permits zero failed replicates. This exact integer
boundary is tested. A sampling-based classifier branch must satisfy both its paired
per-fit sign pattern and the corresponding family-level health pattern; a form that
passes the 1% per-fit divergence bound but fails the 0.1% aggregate bound cannot be
called healthy by a causal label. Every sampling-based label also requires the full
oracle gate for all five forms on both backends.

V3 does not add a second posterior-agreement decision gate. Exact same-model identity
is checked deterministically by the independent value/gradient/Hessian oracle, while
the classifier compares sampling health. The raw natural-scale chains are retained,
so a later review may describe posterior agreement among healthy fits, but such an
unimplemented statistic is not allowed to affect the frozen v3 conclusion.

## Predeclared classification

The assessor classifies each regime separately in this precedence order:

1. `native-pymc-correctness-defect`: native transformed derivatives or density
   disagree on at least two distinct dataset replicates while the corresponding
   manual C/C replicate/backend evidence agrees.
2. `native-graph-or-adaptation`: native C/C is unhealthy, manual C/C is healthy, and
   both agree with the oracle.
3. `group-conditional-centering`: the two group-centered factorial cells are
   unhealthy while both group-non-centered cells are healthy.
4. `location-centering`: the two location-centered factorial cells are unhealthy
   while both location-non-centered cells are healthy.
5. `joint-centering-interaction`: only full NC/NC is healthy. This does not establish
   two independent main effects.
6. `backend-path-specific`: within every paired replicate, subtract NumPyro's number
   of healthy forms from PyMC's number of healthy forms. All eight differences must
   have the same nonzero sign, pass the exact sign test, and agree with the aggregate
   family-count direction. This localizes a compiler/lowering or adaptation-path
   interaction; it does not by itself prove which component is wrong.
7. `residual-tn-or-scale-geometry`: every exact representation is unhealthy after
   all oracle and inverse-CDF checks pass.
8. `initialization-or-budget-sensitive`: reserved for a separately frozen replay;
   the primary matrix cannot assign this label post hoc.
9. `all-representations-healthy`: all five forms pass on both backends.
10. `mixed-inconclusive`: complete evidence matches no stronger rule.

Missing cells always produce `incomplete`. Infrastructure, environment, artifact,
contract, assertion, and programming failures are not published as scientific cell
failures. A scientific failure has one closed stage—`data`, `build`, `initialize`,
`compile`, `sample`, `summarize`, or `diagnose`—and may reference only artifacts
completed before that stage.

## Artifacts and provenance

Every backend pair receives a parent-minted context binding its two ordered logical
blocks and exact ordered ten cells to the manifest digest, source commit, worker
identity, pair execution identity, and ten distinct attempt identities. The context
embeds the complete worker-local
environment attestation as well as its digest; aggregation recomputes that digest and
validates the embedded packages, lock, source, runtime, and clean-tree state against
the manifest. A sidecar collected by a different job cannot stand in for the sampling
worker. Cell processes cannot mint or replace that identity.

The run tree is:

```text
contexts/<pair_id>.json
data/<data_id>.json
starts/natural/<start_id>.json
starts/coordinates/<cell_id>.json
chains/<cell_id>.nc
diagnostics/<cell_id>.json
cells/<cell_id>.json
aggregate/<tier>/results.jsonl
aggregate/<tier>/assessment.json
```

JSON is strict, canonical, and finite. Every reference records path, exact-byte
SHA-256, and byte count. Writes are atomic and cannot replace an existing path. The
cell result is written last as the completion marker. Aggregation checks safe paths,
sizes, hashes, context bindings, cell identity, effective precision, and sampler
RNG provenance. A missing marker remains missing evidence even if partial files
exist.

Raw chains retain the natural parameters and sampler statistics needed to recompute
health, adaptation behavior, divergence locations, costs, and optional descriptive
cross-fit comparisons. Those comparisons are explicitly non-gating in v3. The
compact result record contains only registered scalar metrics; a runner cannot
self-report that a gate passed.

## No-sampling commands

Validate the manifest and its frozen v2 byte anchor:

```bash
uv run python scripts/truncated_hierarchy_causal_contract.py validate
```

Write deterministic JSONL and CSV plans:

```bash
uv run python scripts/truncated_hierarchy_causal_contract.py plan \
  --tier smoke \
  --output-dir /tmp/hssm-1282-causal-smoke
```

Emit the backend-pair hosted-job matrix (2 smoke workers or 16 confirmation workers):

```bash
uv run python scripts/truncated_hierarchy_causal_contract.py matrix \
  --tier confirmation
```

Collect the exact dependency and source attestation:

```bash
uv run python scripts/truncated_hierarchy_causal_contract.py environment \
  --output /path/to/run/environment/environment.json
```

Aggregate hash-verified final markers in canonical plan order:

```bash
uv run python scripts/truncated_hierarchy_causal_contract.py aggregate \
  --tier confirmation \
  --run-dir /path/to/run \
  --output /path/to/run/aggregate/confirmation/results.jsonl
```

Apply the frozen gates and classifier:

```bash
uv run python scripts/truncated_hierarchy_causal_contract.py assess \
  --tier confirmation \
  --results /path/to/run/aggregate/confirmation/results.jsonl \
  --output /path/to/run/aggregate/confirmation/assessment.json
```

The contract commands do not construct models or sample. Sampling belongs to the
separate causal runner and opt-in workflow. Raw run directories are not committed;
only the frozen design and a later reviewed aggregate evidence bundle belong in Git.
Before this workflow lands on the default branch, label-triggered pull-request runs
can exercise the workflow from the PR merge ref, but manual `workflow_dispatch` is
unavailable because GitHub requires the workflow file on the default branch for
manual dispatch. No separate scaffold is required for the label-triggered path.

## Limits and follow-up policy

The two regimes are targeted reproductions, not a survey of every HSSM model. Eight
replicates can support a strong, consistent paired contrast but do not imply high
power for subtle effects. A calibrated prior anchor changes the prior, truth-based
starts change initialization, a linked Normal changes the family, and a higher
`target_accept` changes the sampler policy. Those can be useful diagnostics or
practical alternatives, but none can be introduced after seeing v3 outcomes and
called part of this primary causal experiment.

If evidence selects a safe representation or establishes that no tested exact form
is reliable, the resulting default-policy proposal belongs in a separate production
PR. That bridge must state exactly which finding it uses, retain the broader prior
robustification work independently, and rerun any production-level qualification
required by the chosen policy.
