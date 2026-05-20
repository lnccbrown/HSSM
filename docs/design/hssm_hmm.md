# HSSM-HMM Class: Design Document

**Issue:** [#957](https://github.com/lnccbrown/HSSM/issues/957)
**Status:** Draft — design phase (v1 feasibility-validated 2026-05-19)
**Companion:** [hssm_hmm_overview.md](./hssm_hmm_overview.md) (high-level summary for applied users)

A throwaway prototype mirrored the v1 layered architecture (`HMMConfig` typed
unions, the `HSSM_HMM` facade, the L1/L2/L3 likelihood, FFBS) and was exercised
against the tutorial on single- and multi-participant balanced panels.
Posteriors matched the hand-written tutorial bit-for-bit and FFBS matched
trial-by-trial. Four substantive changes to the original spec are folded in
below — see Section 10.1 items 4–7.

---

## 1. Scope and goals

### 1.1 What this design covers

A new top-level class `HSSM_HMM` that fits **regime-switching sequential sampling models** through the same user-facing pattern as `HSSM(...)` and `RLSSM(...)`. The class wraps the manual construction shown in
[docs/tutorials/hmm_ddm_regime_switching.ipynb](../tutorials/hmm_ddm_regime_switching.ipynb) — Markov chain over hidden regimes, SSM emission per regime, forward-algorithm marginalization, post-hoc FFBS state recovery — so that users do not need to assemble the PyMC model by hand.

### 1.2 v1 scope (this design)

The first version targets the **simplest useful case** so that we can:
- Ship something usable to applied users quickly.
- Validate the architectural choices against a concrete implementation.
- Match the tutorial's numerical results as a regression test.

**v1 includes:**
- Arbitrary number of regimes `K >= 2`.
- A subset of SSM parameters that switch by regime (declared as a flat list).
- All non-switching SSM parameters shared across regimes; standard HSSM priors / regressions for them.
- Balanced panel of participants (every participant has the same number of trials), with no pooling across participants (each gets independent parameter draws).
- Configurable Dirichlet prior on each row of the transition matrix (sticky concentration by default).
- Fixed uniform initial-state distribution `pi0` by default; user-overridable to a fixed vector.
- Automatic soft ordering constraint to break label-switching symmetry.
- Both `analytical` and `approx_differentiable` emission backends for any SSM model supported by HSSM.
- Post-hoc Forward-Filter Backward-Sample (FFBS) regime recovery as a method on the class.

### 1.3 Explicit out-of-scope for v1

These are deferred but **must be accommodated by the v1 architecture** so they can be added later without breaking changes. Each appears in the design with an explicit extension hook (see Section 8).

- Hierarchical pooling of regime-specific or transition-matrix parameters across participants.
- Covariate-driven transitions (`P` as a function of trial-level covariates).
- Per-regime priors or regressions for switching parameters (e.g. `v_0 ~ 1 + difficulty`, `v_1 ~ 1 + difficulty`).
- Estimable initial-state distribution `pi0`.
- Duration-dependent (semi-Markov) regimes.
- Cross-emission models (e.g. one regime is a DDM, another is a uniform "guess" distribution).
- Model comparison across `K` (LOO/WAIC is delicate for HMMs because observations are not conditionally iid; flagged but not solved).

### 1.4 PR boundary (open)

The implementation phases (Section 9) are independent commits that can land in one PR or be split. The architecture is designed so that **adding any of the deferred features is local** — a new field on the config, a new branch in the Op factory, or a new method on the class — never a refactor of the public API.

---

## 2. Background: where this sits in HSSM

### 2.1 The extension surface

`HSSMBase` (in [src/hssm/base.py](../../src/hssm/base.py)) exposes exactly one abstract method:

```python
@abstractmethod
def _make_model_distribution(self) -> type[pm.Distribution]: ...
```

Everything else — data validation, parameter parsing, formula construction, prior assembly, bambi family, sampler entry points, save/load, plotting — is concrete and inherited. A subclass exists to produce a `pm.Distribution` for the model and (optionally) to add domain-specific helper methods.

### 2.2 The per-trial likelihood contract

`make_distribution` in [src/hssm/distribution_utils/dist.py](../../src/hssm/distribution_utils/dist.py) expects the loglik (Op or callable) to return **per-trial log-likelihood contributions**: one logp per row of `data`. The total log-likelihood is the sum. Standard HSSM models satisfy this trivially because their trials are conditionally independent.

For RLSSM and HMM, conditional independence holds only after handling the cross-trial coupling internally inside the Op:
- **RLSSM** computes per-trial parameters deterministically from the RL rule, then evaluates the emission per trial. Per-trial contribution is recovered directly.
- **HMM** uses the forward-algorithm one-step-ahead decomposition to produce per-trial contributions that sum to the marginal likelihood (see Section 3.3).

### 2.3 RLSSM as structural template

[src/hssm/rl/rlssm.py](../../src/hssm/rl/rlssm.py) demonstrates the across-trial-dynamics pattern HSSM-HMM follows:

1. Subclass `HSSMBase`.
2. Define a config dataclass extending `BaseModelConfig` ([src/hssm/rl/config.py](../../src/hssm/rl/config.py)).
3. Validate panel structure; store `n_participants` / `n_trials` on `self`.
4. Reject `missing_data` and `deadline` (they would re-order rows and corrupt cross-trial structure).
5. Build the differentiable Op **before** `super().__init__()` (because base `__init__` calls `_make_model_distribution`).
6. Inject the built Op into the config via `dataclasses.replace(config, loglik=op, backend="jax")`.
7. Override `_make_model_distribution` minimally: bypass `make_likelihood_callable` and hand the Op to `make_distribution` directly.

HSSM-HMM follows the same skeleton with HMM-specific contents.

---

## 3. Model formulation

### 3.1 Notation

- `K` — number of regimes (an integer `>= 2`).
- `T` — number of trials per participant.
- `N` — number of participants. Total trials = `N * T` (balanced panel).
- `s_{n,t} in {0, ..., K-1}` — hidden regime label for participant `n` at trial `t`.
- `y_{n,t} = (rt_{n,t}, response_{n,t})` — observed emission.
- `theta_switching` — SSM parameters that vary by regime. Each is a length-K vector.
- `theta_shared` — SSM parameters shared across regimes.
- `P` — `K x K` transition matrix, row `k` is `P(s_{t+1} = . | s_t = k)`.
- `pi0` — length-K initial-state distribution.

### 3.2 Generative model (per participant; participants are independent in v1)

```
s_1            ~ Categorical(pi0)
s_t | s_{t-1}  ~ Categorical(P[s_{t-1}, :])         for t = 2, ..., T
y_t | s_t = k  ~ SSM(theta_switching[k], theta_shared)
```

Priors in v1:
- `P[k, :] ~ Dirichlet(alpha_k)` independently for each row `k`. The default `alpha_k` is sticky-diagonal: large value on `alpha_k[k]`, small uniform value off-diagonal.
- `pi0` is a fixed length-K vector (uniform by default).
- `theta_switching[m]` for each switching parameter `m`: K independent draws from the standard HSSM prior for that parameter (Normal for unbounded, HalfNormal/Beta for bounded, etc.).
- `theta_shared` follows standard HSSM priors / regressions / hierarchical specs.
- A soft ordering Potential is added automatically to break label-switching symmetry (Section 5.3).

### 3.3 Likelihood via the forward algorithm

The discrete state sequence is marginalized analytically so that only continuous parameters remain — required for gradient-based sampling (NUTS). Define the log-forward variables:

```
log_alpha_t(k) = log p(y_1, ..., y_t, s_t = k | theta, P, pi0)
```

Recursion:

```
log_alpha_1(k) = log pi0(k) + log p(y_1 | s_1 = k, theta)

log_alpha_t(k) = logsumexp_j [ log_alpha_{t-1}(j) + log P[j, k] ]
                 + log p(y_t | s_t = k, theta)        for t = 2, ..., T
```

Marginal log-likelihood of the trial sequence:

```
L = log p(y_1..T | theta, P, pi0) = logsumexp_k log_alpha_T(k)
```

### 3.4 v1 ships the scalar marginal; per-trial logp is post-hoc

The forward algorithm above produces a *scalar* marginal log-likelihood `L = logsumexp_k log_alpha_T(k)`. There are two ways to wire that into the PyMC model graph:

- **Option B (v1 default — what the tutorial does):** contribute `L` to the model via a single `pm.Potential("hmm_loglik", logsumexp_k log_alpha_T(k))`. The sampler sees a scalar log-density per evaluation. Simple, matches the tutorial exactly.

- **Option A (deferred):** return a length-T vector of one-step-ahead conditionals
  ```
  log_Z_t = logsumexp_k log_alpha_t(k)       (running log-evidence)
  delta_1 = log_Z_1
  delta_t = log_Z_t - log_Z_{t-1}            for t = 2, ..., T
  ```
  so that `sum_t delta_t = L` and each `delta_t = log p(y_t | y_1..t-1)` is the per-trial predictive log-density. This *would* match the `make_distribution` per-trial contract (Section 2.2) and feed `arviz.loo` / `log_likelihood` / posterior-predictive plotting without extra work.

**Why v1 ships Option B, not Option A.** The two formulations are mathematically identical (verified: logp agrees to 6 decimal places at every point). But the per-trial-deltas autodiff graph (pytensor.scan + per-trial subtraction) produces a different gradient trajectory under NUTS than the scalar marginal. In the feasibility prototype, identical-seed runs against the analytical DDM backend showed:

- **Option B:** chains converged to the same mode; posteriors matched the tutorial bit-for-bit; R-hat ≈ 1.0.
- **Option A:** one chain converged to the correct mode, the other to the label-switched mode (despite the same soft-ordering Potential); R-hat ≈ 1.83, ESS ≈ 3 for the switching parameters.

The soft-ordering Potential alone is insufficient to keep both chains in the same mode when the gradient flows through the per-trial decomposition.

**Implication for the `make_distribution` per-trial contract.** Because v1 contributes a single scalar via `pm.Potential`, it bypasses `make_distribution`'s per-trial assumption rather than satisfying it. Downstream tooling that wants per-trial logp (`log_likelihood` group, `arviz.loo`) gets it *post-hoc*: the same compiled forward function used by FFBS (Section 5.5) computes per-trial deltas from any saved posterior draw in pure NumPy, and the result is attached to the returned `InferenceData` if requested.

This trades the design's original "trivial superset, no flags needed" property for sampler robustness. Re-evaluate Option A in a future revision if a JAX-only LAN backend with a custom VJP makes the autodiff path stable.

### 3.5 Per-participant batching — one scan, not N

For balanced panels of `N` participants and `T` trials each, the builder reshapes the input `(N*T, ...)` arrays to `(N, T, ...)` internally and runs a **single** forward recursion whose hidden-state tensor carries a leading participant axis. The recursion proceeds along the *trial* axis, and at every step its scan step processes all N participants in parallel.

Concretely, the per-step `log_alpha` has shape `(N, K)` rather than `(K,)`, and the transition update broadcasts the `(K, K)` log-transition matrix against the `(N, K, 1)` previous-state tensor:

```
log_alpha_new[n, k] = logsumexp_j ( log_alpha_prev[n, j] + log_P[j, k] )
                     + log_emission[n, t, k]
```

This produces:
- **pytensor backend:** one `pytensor.scan` over `T` (not N), one JAX-JIT compile unit overall.
- **JAX backend:** the same recursion expressed with `jax.lax.scan` + `jax.vmap` over the participant axis.

The joint marginal `sum_n L_n` is then `pt.sum(pt.logsumexp(log_alpha_T, axis=1))` — one scalar contributed to the model as a single `pm.Potential`.

**Why not a Python for-loop over participants.** The feasibility prototype tried the obvious "loop over n in Python, build one `pytensor.scan` per subject" pattern. At `N = 1` it works fine. At `N = 5` it samples in ~2.5 min. At `N = 20` it never finished a 400-draw run in 40+ minutes — because each per-subject scan becomes its own JIT-compile unit and per-subject compilation dominates. The batched single-scan pattern above scales linearly in N for the leaf compute while keeping compilation roughly constant in N.

Row-order assumption (inherited from RLSSM): rows are grouped by participant, and within each participant they appear in trial order. The class validates this via `validate_balanced_panel` before constructing the builder.

---

## 4. Architecture

### 4.1 Layered design

The implementation separates three concerns so that future extensions touch the smallest possible surface:

| Layer | Responsibility | Future extension points |
|---|---|---|
| **L1: Chain dynamics** | Initial distribution, transition matrix, forward recursion | Covariate-driven `P`; estimable `pi0`; semi-Markov durations |
| **L2: Emission** | Per-regime SSM log-density evaluation | Cross-emission models; new SSMs |
| **L3: Composition** | Builds the differentiable Op by composing L1 + L2 | None expected — the seam is the contract between layers |

L3 is what's stored in `model_config.loglik`. The class itself is mostly L3-glue + lifecycle.

### 4.2 Module layout

A new subpackage `src/hssm/hmm/` mirroring `src/hssm/rl/`:

```
src/hssm/hmm/
  __init__.py
  hmm.py              # HSSM_HMM(HSSMBase)
  config.py           # HMMConfig(BaseModelConfig)
  likelihoods/
    __init__.py
    builder.py        # make_hmm_logp_op(...) — composes L1 + L2; analytical → pm.Potential closure, LAN → JAX Op
    forward.py        # L1: forward recursion (pytensor for analytical, JAX for LAN)
    emissions.py      # L2: helper to resolve a per-regime SSM logp
  ordering.py         # Label-switching heuristic
  ffbs.py             # Post-hoc FFBS regime recovery + per-trial logp (Section 5.7)
  utils.py            # validate_balanced_panel re-export, K checks, etc.
```

User-facing API:

```python
from hssm import HSSM_HMM        # re-exported from hssm.__init__
from hssm.hmm import HMMConfig   # advanced / custom-config path
```

### 4.3 Public exports

- `HSSM_HMM` — main class.
- `HMMConfig` — config dataclass (mirrors `RLSSMConfig`'s role).
- Nothing else publicly exported from `hssm.hmm` in v1. The Op factory, FFBS, and helpers are internal.

---

## 5. Component design

### 5.1 `HMMConfig(BaseModelConfig)`

```python
@dataclass
class HMMConfig(BaseModelConfig):
    # --- Markov chain structure ---
    K: int = field(kw_only=True)
    switching_params: list[str] = field(kw_only=True)
    transition_prior: TransitionPriorSpec = field(kw_only=True)
    initial_distribution: InitialDistributionSpec = field(
        default_factory=UniformInitialDistribution, kw_only=True,
    )

    # --- Emission ---
    decision_process: str | BaseModelConfig = field(kw_only=True)
    decision_process_loglik_kind: LoglikKind = field(kw_only=True)
    emission_logp_func: Any = field(default=None, kw_only=True)

    # --- Label-switching ---
    ordering: OrderingSpec = field(default_factory=AutoOrdering, kw_only=True)

    # --- Hierarchical / pooling (v1: no-op; reserved for v2) ---
    pooling: PoolingSpec = field(default_factory=NoPooling, kw_only=True)
```

`TransitionPriorSpec`, `InitialDistributionSpec`, `OrderingSpec`, `PoolingSpec` are typed unions of small dataclasses. Each starts with one "default" concrete case in v1 and the design leaves room for additional cases without API changes:

- `TransitionPriorSpec = StickyDirichlet | DirichletConcentration | CovariateDrivenTransition` — v1 ships the first two; the third is the documented extension hook.
- `InitialDistributionSpec = UniformInitialDistribution | FixedInitialDistribution | DirichletInitialDistribution` — v1 ships the first two.
- `OrderingSpec = AutoOrdering | OrderByParam | NoOrdering` — v1 ships `AutoOrdering` (default) and `NoOrdering` (escape hatch).
- `PoolingSpec = NoPooling | PartialPooling(...) | FullPooling(...)` — v1 ships only `NoPooling`.

This pattern means the constructor signature **never changes** when a new variant is added; only the union grows.

`HMMConfig.validate()` enforces:
- `K >= 2`.
- Every element of `switching_params` appears in the decision-process's `list_params`.
- `emission_logp_func` is set (or resolvable from `decision_process` + `decision_process_loglik_kind`).
- `transition_prior` is consistent with `K` (e.g. concentration matrix has shape `(K, K)`).

### 5.2 The forward-algorithm builder (`make_hmm_logp_op`)

The builder closes over panel shape, K, switching-param identity, and the resolved emission logp callable. **The output type differs by emission backend** — see "Two backend paths" below.

```python
def make_hmm_logp_op(
    emission_logp_func: Callable[..., Array],
    n_participants: int,
    n_trials: int,
    K: int,
    list_params: list[str],          # full param order seen by Op
    switching_params: list[str],     # subset that has K values
    data_cols: list[str],            # e.g. ["rt", "response"]
    backend: Literal["pytensor", "jax"],
    extra_fields: list[str] | None = None,
) -> _LoglikContribution: ...
```

Common contract:
- Inputs: panel data `(N*T, len(data_cols))`, switching params shaped `(K,)`, shared params shaped `()` (broadcast), transition matrix shaped `(K, K)`, initial distribution shaped `(K,)`. Exact wiring is internal.
- Output: contributes the joint marginal `sum_n L_n` to the model. v1 contributes a single scalar (Section 3.4 Option B). Per-trial logp for downstream tooling is computed post-hoc — see Section 5.7.

**Two backend paths.** The Op-factory abstraction's value depends on what the emission likelihood is written in:

1. **Analytical backend (`pytensor`).** `logp_ddm` is already pure pytensor and `pytensor.scan` differentiates correctly; numpyro compiles the full graph to JAX as a final step. Wrapping this in a custom JAX Op with a hand-rolled VJP adds complexity with no benefit. For this path, `make_hmm_logp_op` returns a **model-builder closure** that adds `pm.Potential("hmm_loglik", joint_marginal)` to the active `pm.Model`. The feasibility prototype validates this path against the tutorial.

2. **LAN backend (`jax`).** The LAN forward pass is a JAX function with no pytensor counterpart, and routing it through pytensor's scan would lose the ONNX-backed JIT compilation. For this path, `make_hmm_logp_op` returns a `pytensor.graph.Op` whose forward and VJP are JAX functions (same pattern as `make_rl_logp_op`), and the class wraps it in a `pm.CustomDist` / `pm.Potential` whose logp is the scalar joint marginal.

The class hides this split: callers pass `decision_process_loglik_kind` and `HMMConfig.backend` is set accordingly. The L1 / L2 / L3 *layering* is identical between backends; only the L3 wrapping changes.

Layer 1 (`hmm/likelihoods/forward.py`) implements the recursion in whichever framework the chosen backend uses (pytensor or JAX):

```python
def forward_log_marginal(
    log_emission_lik,         # shape (T, K) — per-regime per-trial logp
    log_P,                    # shape (K, K)
    log_pi0,                  # shape (K,)
):                            # returns a scalar: log p(y_1..T | theta, P, pi0)
    ...
```

Layer 2 (`hmm/likelihoods/emissions.py`) computes `log_emission_lik` by evaluating the SSM logp once per regime under that regime's parameter values:
- `analytical` — calls into the existing analytical likelihoods in `hssm.likelihoods` (e.g. `logp_ddm`).
- `approx_differentiable` — calls into the JAX/ONNX LAN via `make_distribution_for_supported_model(..., backend="jax")`.

The composition (L3 in `builder.py`) runs a **single** batched forward recursion whose hidden state carries the participant axis (see Section 3.5). One `pytensor.scan` over `T` for the pytensor path; one `jax.lax.scan` over `T` (often combined with `jax.vmap` over `N`) for the JAX path. A Python loop over participants was tried in the feasibility prototype and rejected — it does not scale (40+ min at N=20).

### 5.3 Label-switching: `AutoOrdering`

`AutoOrdering` is the default `OrderingSpec`. It adds a soft ordering Potential on **one** switching parameter so the posterior is uniquely identified up to label-switching.

Heuristic to pick the constrained parameter (in order of preference):
1. If `v` is among `switching_params`, constrain on `v`.
2. Else if there is exactly one switching parameter, constrain on it.
3. Else use the first parameter in `switching_params` and emit a `_logger.warning` naming the choice and pointing to `OrderByParam` / `NoOrdering` for overrides.

Constraint form: a strict chain of `>` Potentials,

```python
pm.Potential("hmm_order", pt.switch(
    pt.all(param[:-1] > param[1:]), 0.0, -1e10,
))
```

(equivalent to `param[0] > param[1] > ... > param[K-1]`).

**Why strict `>`, not `>=`.** The default initial value for unbounded switching parameters is `0` for every regime, so `param[:-1] >= param[1:]` is *True* at init (every pair ties). The Potential then contributes `0` to the logp at init and the sampler has no signal to break the symmetry; two chains can drift to opposite sides of the constraint surface during tuning and each gets locked into its own label-switching mode. Strict `>` evaluates *False* at the tie, applies the `-1e10` penalty immediately, and forces all chains to break symmetry the same way. The feasibility prototype confirmed this: `>=` produced R-hat 1.83 on the switching parameter; switching the single character to `>` fixed it.

> **Note for future hierarchical extensions.** The soft Potential is sufficient for v1 (a single length-K switching vector). It is *not* expected to hold once switching parameters are pooled across participants — the larger energy landscape lets NUTS cross the sharp `-1e10` barrier. A hierarchical extension would replace it with PyMC's `ordered` transform on the group-level anchor. This is out of scope for v1; flagged here so the v1 ordering code is written in a way that is easy to swap.

`OrderByParam(name=..., direction="asc"|"desc")` is the user-supplied override; `NoOrdering()` is the escape hatch (with a `_logger.warning` reminding the user that posteriors may be multi-modal).

### 5.4 The class: `HSSM_HMM(HSSMBase)`

Constructor signature mirrors `HSSM`/`RLSSM`:

```python
class HSSM_HMM(HSSMBase):
    def __init__(
        self,
        data: pd.DataFrame,
        decision_process: str | BaseModelConfig,
        K: int,
        switching_params: list[str],
        *,
        transition_prior: TransitionPriorSpec | dict | None = None,
        initial_distribution: InitialDistributionSpec | None = None,
        decision_process_loglik_kind: LoglikKind | None = None,
        participant_col: str = "participant_id",
        ordering: OrderingSpec | None = None,
        # ---- inherited HSSM kwargs ----
        include: list[...] | None = None,
        p_outlier: float | bmb.Prior | None = 0.05,
        lapse: bmb.Prior | None = ...,
        link_settings: ... = None,
        prior_settings: ... = "safe",
        extra_namespace: dict | None = None,
        process_initvals: bool = True,
        initval_jitter: float = ...,
        **kwargs,
    ) -> None: ...
```

Order of operations in `__init__`:

1. `self._init_args = self._store_init_args(locals(), kwargs)` — save/load.
2. Build the `HMMConfig` from the user args (resolves defaults, normalizes dict-shorthand for `transition_prior` etc.).
3. `model_config.validate()`.
4. **Reject** `missing_data` / `deadline` like RLSSM — strict row order needed.
5. Resolve `participant_col`:
   - If `participant_col` is in `data.columns`, use it as-is.
   - Otherwise synthesize a column of constant `0` and emit `_logger.info("No participant column found; treating all rows as a single participant.")`. This handles the common single-participant case without requiring the user to add a dummy column.
6. `validate_balanced_panel(data, participant_col)` → stash `self.n_participants`, `self.n_trials`.
7. Resolve `emission_logp_func` from `decision_process` + `decision_process_loglik_kind` if not user-supplied.
8. Build `loglik_op = make_hmm_logp_op(...)`.
9. `model_config = replace(model_config, loglik=loglik_op, backend="jax")`.
10. `super().__init__(data=..., model_config=model_config, ...)` — `HSSMBase` calls `_make_model_distribution` during init.
11. After `super().__init__`, register the transition-matrix and initial-distribution priors as PyMC random variables inside the underlying `pymc_model` (they cannot be declared via bambi's formula system; they live as ordinary PyMC nodes that the Op consumes alongside the bambi-managed params). Add the ordering Potential.

Step 11 is the **only place** the HMM diverges from RLSSM's lifecycle. The transition matrix and initial distribution are not row-aligned trial parameters — they're "global" latents, sized `(K, K)` and `(K,)`. They need to be declared on the PyMC model that bambi has just built. The alternative (declaring them as bambi formula params) is awkward because bambi assumes scalar-per-row priors; we accept the step-11 seam as the standing approach.

**v1 caveat — bambi vs. direct `pm.Model` for the scalar-only case.** For v1, every SSM parameter is scalar (no regressions, no hierarchical pooling) and bambi is doing only formula-management busywork: declaring scalar priors that PyMC could declare directly. The step-11 hook (build via bambi, then mutate the resulting `pm.Model` to add P / pi0 / Potential) is therefore complexity in service of a future feature, not v1. The feasibility prototype bypassed bambi entirely and built `pm.Model` directly with no functional loss — same RV names, same posteriors as the tutorial.

Two paths are both viable for Phase 2:

- **(a) Keep step 11 as written.** Honors the design's principle that `HSSM_HMM` inherits the bambi lifecycle so that per-regime regressions (Section 1.3 deferred) drop in later as a `switching_params: dict[str, RegimePriorSpec]` extension without restructuring the class.
- **(b) Build `pm.Model` directly for v1 and adopt the bambi flow when the first regression-driven feature lands.** Simpler v1 `__init__`; the transition from (b) to (a) is local (it changes how priors are declared inside `_build_pymc_model`, not the external API).

To be decided when Phase 2 starts; the rest of this document is written assuming (a) so the lifecycle is forward-compatible. The decision is logged in Section 10.2 as an open question.

`_make_model_distribution` differs from RLSSM. Because v1 contributes the joint marginal as a scalar `pm.Potential` (Section 3.4), it does not satisfy `make_distribution`'s per-trial contract. RLSSM-style routing through `make_distribution(..., params_is_trialwise=[True, ...])` is therefore not used.

Instead, `_make_model_distribution` returns a `pm.CustomDist` whose logp is the scalar joint marginal `sum_n L_n` (analytical backend: `pm.Potential` on the active model; LAN backend: the pytensor `Op` from Section 5.2's path 2). The `pm.Distribution` wrapper preserves bambi/HSSMBase compatibility — the wrapped logp is one number per evaluation rather than one per row, which is well-formed for `pm.CustomDist` as long as the observed data is the entire panel rather than a per-row response. The class registers a single dummy observed value to satisfy bambi's "observed must be present" check; the actual data is captured by the closure inside the L3 builder.

### 5.5 FFBS post-hoc method

```python
def infer_regimes(
    self,
    idata: az.InferenceData | None = None,
    n_draws: int = 200,
    seed: int | None = None,
) -> az.InferenceData: ...
```

Behavior:
- Uses `self.traces` if `idata is None`.
- Randomly selects `n_draws` posterior draws (across chains).
- For each selected draw, runs FFBS:
  1. Compute the per-regime emission logp matrix `(T, K)` for each participant (via the same emission callable used at sampling time).
  2. Forward pass to get `log_alpha_t(k)`.
  3. Backward sample: `s_T` from the normalized `log_alpha_T`; for `t = T-1, ..., 1`, sample `s_t` from `p(s_t = k | s_{t+1}, y_1..T)` proportional to `alpha_t(k) * P[k, s_{t+1}]`.
- Returns an `arviz.InferenceData` with:
  - Group `posterior_regimes` containing the sampled state sequences, shape `(n_draws, n_participants, n_trials)` with dim names `("draw", "participant", "trial")`.
  - Optionally a derived `regime_probability` variable of shape `(n_participants, n_trials, K)` giving the marginal posterior probability of each regime per trial.

Implementation lives in `hssm/hmm/ffbs.py`. It does not require the Op — only a NumPy callable for the emission logp (compiled once from the same `emission_logp_func` used at sampling time, or directly via `pytensor.function` for analytical likelihoods). This means FFBS works even if the Op was JAX-only, by routing through the analytical likelihood when available (with a documented fallback to invoking the JAX function via `jax2numpy` if not).

### 5.6 Helper: `validate_balanced_panel`

Re-used from RLSSM ([src/hssm/rl/utils.py](../../src/hssm/rl/utils.py)). **v1 decision:** import from `hssm.rl.utils` directly — no module churn for a single shared helper. Hoist to a shared location (`src/hssm/utils.py`) when a third subclass needs it. The import path inside `hssm/hmm/utils.py` is the single seam to update on hoisting; nothing else in the package depends on the current location.

### 5.7 Post-hoc per-trial log-likelihood

Because v1 contributes the joint marginal as a scalar (Section 3.4), the `log_likelihood` group of the returned `InferenceData` is empty by default and tools like `arviz.loo` will not work out of the box.

```python
def compute_log_likelihood(
    self,
    idata: az.InferenceData | None = None,
) -> az.InferenceData: ...
```

Behavior:
- Uses `self.traces` if `idata is None`.
- For each posterior draw, evaluates the forward algorithm in pure NumPy (using the same compiled emission callable as `infer_regimes`) and computes the per-trial deltas `delta_t = log_Z_t - log_Z_{t-1}` (Section 3.4 Option A, evaluated post-hoc).
- Attaches the result to `idata.log_likelihood` with shape `(chain, draw, n_participants, n_trials)`, ready for `arviz.loo` / `arviz.waic`.

This separates the "what the sampler needs" graph (Option B — robust gradients) from the "what arviz wants" array (Option A — per-trial, computed once after sampling). The two are mathematically consistent: their sum over trials equals the scalar marginal the sampler saw.

Implementation lives alongside FFBS in `hssm/hmm/ffbs.py` (or a sibling module) since both consume the same per-regime emission callable.

---

## 6. User-facing API

### 6.1 Minimal example (v1)

```python
import hssm
import pandas as pd

df = pd.read_csv("...")  # columns: rt, response, participant_id

model = hssm.HSSM_HMM(
    data=df,
    decision_process="ddm",
    K=2,
    switching_params=["v"],
    transition_prior={"sticky_diag": 20.0, "sticky_offdiag": 2.0},
    participant_col="participant_id",
)

idata    = model.sample(draws=1000, tune=1000, chains=4)
summary  = model.summary()
regimes  = model.infer_regimes(idata, n_draws=200)
```

### 6.2 Custom config path (advanced)

```python
from hssm.hmm import HMMConfig

cfg = HMMConfig(
    model_name="hmm_ddm",
    K=3,
    switching_params=["v", "a"],
    decision_process="ddm",
    decision_process_loglik_kind="analytical",
    transition_prior=DirichletConcentration(
        alpha=np.array([[30, 2, 2], [2, 30, 2], [2, 2, 30]]),
    ),
    initial_distribution=FixedInitialDistribution(pi0=[0.5, 0.3, 0.2]),
    ordering=OrderByParam(name="v", direction="desc"),
    # ... standard BaseModelConfig fields ...
)

model = hssm.HSSM_HMM(data=df, model_config=cfg, ...)
```

### 6.3 Inherited behavior

The class inherits unchanged from `HSSMBase`:
`sample`, `vi`, `find_MAP`, `sample_posterior_predictive`, `sample_prior_predictive`, `predict`, `log_likelihood`, `summary`, `plot_trace`, `plot_predictive`, `graph`, `save_model`, `load_model`.

### 6.4 New methods on `HSSM_HMM` only

- `infer_regimes(idata=None, n_draws=200, seed=None)` — FFBS-based regime recovery.
- `compute_log_likelihood(idata=None)` — post-hoc per-trial logp (Section 5.7), required for `arviz.loo` / `arviz.waic` because the sampler graph contributes only the scalar marginal.
- `plot_regime_recovery(regimes_idata, ax=None, ...)` — wraps the tutorial's stacked-area plot.

---

## 7. Validation strategy

### 7.1 Tutorial regression

The first non-trivial test reproduces [docs/tutorials/hmm_ddm_regime_switching.ipynb](../tutorials/hmm_ddm_regime_switching.ipynb) using `HSSM_HMM`:
- Same data-generating process, same priors, same sampling settings, same seed.
- Posterior means for `v`, `a`, `z`, `t`, `P` must match the tutorial's results **bit-for-bit** (to 4 decimal places) — the feasibility prototype confirmed this is achievable when the graph is structurally identical (Option B + strict `>` constraint).
- Recovered FFBS regime probabilities must agree trial-by-trial with the tutorial's FFBS output (per-trial probabilities differing by less than 0.05 on average; the prototype matched to 4 decimal places).

### 7.2 Synthetic recovery suite

- Recover ground-truth parameters across a grid of `(K, n_trials, n_participants)`.
- Verify both analytical and LAN emission backends produce overlapping posteriors.
- Sanity-check label-switching: posteriors are unimodal in every chain when `AutoOrdering` is on.

### 7.3 Edge cases

- `K = 2` with only one switching param (canonical case).
- `K = 3` with multiple switching params (stresses the ordering heuristic).
- Single-participant data.
- Multi-participant balanced panel.
- Empty regime in simulation (one regime never occupied) — must not crash.
- Highly imbalanced regime mixtures.

---

## 8. Extensibility: where v2/v3 features plug in

This is the most load-bearing section of the design. Every deferred feature listed in Section 1.3 corresponds to a named hook below.

| Future feature | v1 hook | Surface change |
|---|---|---|
| Hierarchical pooling | `PoolingSpec` union on `HMMConfig.pooling` | New variant `PartialPooling(...)`; constructor signature unchanged. |
| Covariate-driven `P` | `TransitionPriorSpec` union on `HMMConfig.transition_prior` | New variant `CovariateDrivenTransition(formula=..., link=...)`; Op factory branches on variant type. |
| Per-regime priors / regressions for switching params | `switching_params` is `list[str]` in v1; v2 widens to `list[str] | dict[str, RegimePriorSpec]` | Backwards-compatible: a list is treated as a dict with default specs. |
| Estimable `pi0` | `InitialDistributionSpec` union | New variant `DirichletInitialDistribution(alpha=...)`; class registers an extra PyMC node in step 11 of `__init__`. |
| Semi-Markov durations | Layer 1 (`forward.py`) gains a new recursion implementation; `TransitionPriorSpec` union grows | The L1 contract (in → scalar log-marginal) is unchanged. |
| Cross-emission models | Layer 2 (`emissions.py`) gains a path for "one emission spec per regime" | `decision_process` becomes `str | list[str]` (uniform vs per-regime); validated by `HMMConfig.validate`. |
| New SSMs | None — L2 already resolves via `make_distribution_for_supported_model` | Free. |

Design principles enforced by this table:

1. **Every user-facing config field is a typed union of small dataclasses.** Growing a union never breaks existing call sites.
2. **The L1 / L2 / L3 contracts are fixed.** Future features extend existing layers but never change the shape of the data that crosses a layer boundary.
3. **The Op factory dispatches on config-spec types.** Adding a new variant means adding a new `match` arm.
4. **The class lifecycle (Section 5.4) is unchanged across versions.** Step 10 may register additional PyMC nodes for new latents, but the order of operations is preserved.

---

## 9. Implementation phases

Tracking issue should list these as separate items. Whether they ship in one PR or several is a separate scoping call.

### Phase 1 — Design doc *(this document)*
- Approval of API + architecture.
- Sign-off on the v1 / v2 boundary.

### Phase 2 — Prototype: K=2, DDM, single participant
- `hssm/hmm/` skeleton in place.
- `HMMConfig` with the smallest possible field set (single-participant case, no `participant_col` machinery yet).
- L1 forward recursion (Option B: scalar marginal — see Section 3.4) + L2 analytical DDM emission + L3 model-builder closure.
- `HSSM_HMM.__init__` happy path. Open question: bambi vs. direct `pm.Model` (Section 5.4 step 11 caveat); resolve at the start of this phase before writing code.
- Strict `>` ordering Potential (Section 5.3).
- Tutorial regression test (Section 7.1) — must pass before this phase ends. The feasibility prototype already showed bit-for-bit equality is achievable; the regression test should target the same standard.

### Phase 3 — Generalize: arbitrary K, multiple switching params, balanced panel
- Panel validation; a single batched forward scan over `(subject, trial)` in the L3 builder — not a Python loop over participants (Section 3.5, decision 10.1.7).
- LAN emission backend — this is the path that earns the `pytensor.graph.Op` + JAX VJP wrapping (Section 5.2 path 2); the feasibility prototype did not exercise it, so allocate time for a focused validation against the tutorial's LAN run.
- `AutoOrdering` heuristic; `OrderByParam` / `NoOrdering` variants.
- Synthetic recovery suite (Section 7.2).

### Phase 4 — FFBS regime recovery and per-trial logp
- `infer_regimes` method.
- `compute_log_likelihood` method (Section 5.7) — share the compiled forward emission callable with FFBS.
- `plot_regime_recovery` helper.
- Test against tutorial FFBS outputs.

### Phase 5 — Documentation
- User-facing tutorial notebook in `docs/tutorials/` mirroring `hmm_ddm_regime_switching.ipynb` but using `HSSM_HMM`.
- API docs entry under `docs/api/`.
- Changelog entry.

### Phase 6+ — Deferred features (separate PRs, in priority order)
1. Hierarchical pooling of switching params across participants.
2. Estimable `pi0`.
3. Covariate-driven transitions.
4. Per-regime priors / regressions for switching params.
5. Cross-emission and semi-Markov support.

These are out of scope for v1 and not required by issue #957. Each has an
extension hook in Section 8; the deferred work begins only after Phases 2–5
land a working v1 class.

---

## 10. Decisions and open questions

### 10.1 Decisions (1–3 resolved 2026-05-12, 4–7 resolved 2026-05-19 from the v1 feasibility prototype)

1. **`P` and `pi0` placement in the PyMC graph.** Register them as PyMC random variables in the step-11 lifecycle hook (after bambi builds the model). The alternative — extending `Params.from_user_specs` to understand non-trial-shaped priors — is cleaner architecturally but not justified for a single use case. Revisit if a second feature needs the same shape of node.

2. **`participant_col` default and missing-column behavior.** If the column is absent from `data`, synthesize a constant column and emit a `_logger.info` message identifying the choice. This makes single-participant usage friction-free without surprising multi-participant users. Spelled out in Section 5.4 step 5.

3. **`validate_balanced_panel` location.** Import from `hssm.rl.utils` for v1 — no module churn for a single shared helper. Hoist to `hssm.utils` when a third subclass needs it. Single import seam inside `hssm/hmm/utils.py`. Spelled out in Section 5.6.

4. **Scalar marginal (Option B), not per-trial deltas (Option A), in the sampler graph.** The two are mathematically identical, but Option A's autodiff path under NUTS produces unstable gradients on the analytical backend (R-hat 1.83 vs 1.0 in the feasibility prototype with identical seeds). v1 contributes a single scalar `pm.Potential` per evaluation; per-trial logp is reconstructed post-hoc via `compute_log_likelihood` (Section 5.7) for `arviz.loo` / `log_likelihood` integration. Re-evaluate Option A for the LAN backend if a custom JAX VJP stabilizes the gradient.

5. **Strict `>` for the ordering Potential.** Non-strict `>=` is satisfied at the default init (all switching params start at 0), gives the sampler no signal to break label symmetry at tuning time, and produced R-hat 1.83 in the feasibility prototype. Strict `>` evaluates False at the tie, applies the `-1e10` penalty immediately, and forces all chains to break symmetry the same way. Spelled out in Section 5.3.

6. **Op-factory contract is per-backend.** The analytical and LAN backends share the L1 / L2 / L3 *layering*, but `make_hmm_logp_op`'s output type differs: a model-builder closure for the pytensor (analytical) backend, a `pytensor.graph.Op` with hand-rolled JAX VJP for the JAX (LAN) backend. Trying to wrap pytensor's analytical logp in a JAX Op adds complexity without benefit because pytensor.scan already differentiates correctly and numpyro JIT-compiles the whole graph. Spelled out in Section 5.2.

7. **One batched scan over participants, not a Python loop.** An earlier draft of the design said "a Python for-loop suffices for the pytensor path with realistic N." The feasibility prototype tried it: ~2.5 min at N=5, failed to finish in 40+ min at N=20. The cause is that each iteration of the Python loop creates an independent `pytensor.scan` operation, and each scan becomes its own JIT-compile unit — so compilation time scales linearly in N with a large constant, and the JIT graph blows up. The fix is to run a *single* forward recursion whose hidden-state tensor carries the participant axis, so all N subjects are advanced in lockstep at every trial step. Spelled out in Section 3.5 and Section 5.2's L3 description.

### 10.2 Open questions

These remain to be resolved before Phase 2 begins (or, where noted, during implementation).

1. **Sampler default for HSSM-HMM.** Tutorial uses `nuts_sampler="numpyro"` because the LAN + JAX backend makes the PyMC default very slow. **Proposed:** make `nuts_sampler="numpyro"` the documented default in the tutorial but do not override `HSSMBase.sample`'s default in code (avoid surprising behavior divergence between subclasses).

2. **`infer_regimes` return shape: per-draw sequences vs marginal probabilities.** Section 5.5 returns both. **Proposed:** keep both — they serve different downstream uses (per-draw for plotting credible bands, marginals for trial-level summaries).

3. **PR scoping for v1.** Phases 2–5 together = a fully usable v1. They can also be split across multiple PRs (e.g. Phase 2 alone for the initial commit, then a second PR with Phases 3–5). To be decided when the first phase nears completion; the architecture supports either path.

4. **Bambi lifecycle (Section 5.4 step 11) vs. direct `pm.Model` for v1.** With all-scalar SSM params, bambi adds no functional value in v1 and the step-11 mutate-after-build pattern is complexity in service of future hierarchical / regression features. The feasibility prototype bypassed bambi entirely and matched the tutorial exactly. Two options: (a) keep the bambi lifecycle as written so future regression features (`switching_params: dict[str, RegimePriorSpec]`) drop in locally; (b) build `pm.Model` directly in v1 and migrate to (a) when the first regression feature lands. Decide before Phase 2 begins. The rest of this document assumes (a) so the lifecycle stays forward-compatible.

---

## Appendix A — Notation cross-reference with the tutorial

| Doc symbol | Tutorial code |
|---|---|
| `K` | `K = 2` |
| `T` | `N_TRIALS = 500` |
| `P` | `pm.Dirichlet("P", a=sticky_alpha, shape=(K, K))` |
| `pi0` | `pt.ones(K) / K` |
| `theta_switching` (`v`) | `pm.Normal("v", mu=0.0, sigma=3.0, shape=(K,))` |
| `theta_shared` (`a, z, t`) | `pm.HalfNormal("a", ...)`, `pm.Beta("z", ...)`, `pm.HalfNormal("t", ...)` |
| `log_alpha_t(k)` | the `log_alphas` output of `pytensor.scan(...)` |
| `L = logsumexp_k log_alpha_T(k)` (v1 marginal) | `pt.logsumexp(log_alphas[-1])` |
| `delta_t` (Section 5.7 post-hoc per-trial logp) | *(not exposed in tutorial; computed post-hoc in `compute_log_likelihood`)* |
| FFBS | `ffbs_single_ddm(...)` |

## Appendix B — Files touched / created

**Created:**
- `src/hssm/hmm/__init__.py`
- `src/hssm/hmm/hmm.py`
- `src/hssm/hmm/config.py`
- `src/hssm/hmm/likelihoods/__init__.py`
- `src/hssm/hmm/likelihoods/builder.py`
- `src/hssm/hmm/likelihoods/forward.py`
- `src/hssm/hmm/likelihoods/emissions.py`
- `src/hssm/hmm/ordering.py`
- `src/hssm/hmm/ffbs.py`
- `src/hssm/hmm/utils.py`
- `tests/hmm/...` (mirroring the tree)
- `docs/tutorials/hmm_ddm_class.ipynb` (Phase 5)
- `docs/api/hmm.md` (Phase 5)

**Modified:**
- `src/hssm/__init__.py` — re-export `HSSM_HMM` and `HMMConfig`.
- `docs/changelog.md` — entry under the next release.
- Possibly `src/hssm/utils.py` — if `validate_balanced_panel` is hoisted.
