# Bring your own likelihood: external trainers

You trained a likelihood surrogate outside the HSSM ecosystem's LAN pipeline —
in [sbi](https://github.com/sbi-dev/sbi) or
[BayesFlow](https://github.com/bayesflow-org/bayesflow). This page routes you
to the right integration path.

## The routes

| Route | Source | Mechanism | When to use | Tutorial |
|-------|--------|-----------|-------------|----------|
| **NRE → ONNX** | sbi | `lanfactory.onnx.transform_sbi_to_onnx` → `loglik="file.onnx"` | Portable, shareable artifact; ratio estimators | [sbi NRE integration](../tutorials/sbi_nre_integration.ipynb) |
| **NLE → ONNX** | BayesFlow | `lanfactory.onnx.transform_bayesflow_to_onnx` → `loglik="file.onnx"` | Portable artifact; flow-based density estimators | [BayesFlow NLE integration](../tutorials/bayesflow_nle_onnx_integration.ipynb) |
| **LRE → JAX callable** | BayesFlow | in-memory JAX function → `loglik=<callable>` | Fast iteration during model development; no export step | [BayesFlow LRE integration](../tutorials/bayesflow_lre_integration.ipynb) |

Not yet supported: MNLE-style mixed discrete/continuous observations and
NSF-based flows (blocked on `SearchSorted` ONNX support).

## The shared workflow

All three tutorials follow the same skeleton, so you can read any one of them
and transfer the pattern:

1. **Simulate** training data from a known ground truth (`ssm-simulators` or `hssm.simulate_data`).
2. **Train** the surrogate in its home library.
3. **Export** — for the ONNX routes, via LANfactory's exporters, which enforce
   [the ONNX likelihood contract](custom_onnx_likelihoods.ipynb) by construction
   (see LANfactory's [sbi](https://lnccbrown.github.io/LANfactory/exporting_sbi_models/)
   and [BayesFlow](https://lnccbrown.github.io/LANfactory/exporting_bayesflow_models/)
   export guides for the framework-specific constraints).
4. **Load into HSSM** — `hssm.HSSM(loglik=..., loglik_kind="approx_differentiable")`;
   HSSM does not need to know which framework trained the surrogate.
5. **Validate** — for the DDM you can compare against HSSM's analytical
   likelihood as a gold standard; in general, check recovery on simulated data
   before trusting the surrogate on real data.

## See also

- [The ONNX likelihood contract](custom_onnx_likelihoods.ipynb) — the rules an ONNX artifact must satisfy, runnable
- [Custom models from JAX callables](../tutorials/jax_callable_contribution_onnx_example.ipynb) — the same callable gesture without an external trainer
- [Understanding likelihood functions in HSSM](../tutorials/likelihoods.ipynb) — where `approx_differentiable` fits among the likelihood kinds
