# The HSSM ecosystem

HSSM is the user-facing package of a four-part toolchain. Most users never
leave it: install HSSM, fit models, publish. You end up here when a question
crosses a package boundary — where does a likelihood network come from, which
version pairs with which, why does an ONNX file have to look a certain way.

This page is the map. It is maintained in
[HSSMSpine](https://github.com/lnccbrown/HSSMSpine), the ecosystem's
coordination repository, and published here so there is one description of the
whole rather than three partial ones.

## The packages you install

| Package | Owns | Start here if you want to... |
|---|---|---|
| [HSSM](https://lnccbrown.github.io/HSSM/) | Bayesian inference on sequential sampling models — model specification, priors, sampling, diagnostics, model comparison | ...fit a model to behavioral data. This is the default answer. |
| [ssm-simulators](https://lnccbrown.github.io/ssm-simulators/) (`ssms`) | The generative models: simulators for the SSM family, task environments and learning rules for RLSSMs, and the training-data generators | ...simulate from a model, add a new model to the family, or generate training data. |
| [LANfactory](https://lnccbrown.github.io/LANfactory/) | Training likelihood approximation networks on simulated data, and exporting them to ONNX | ...train your own likelihood network, or export one trained elsewhere. |
| [LAN_pipeline_minimal](https://github.com/lnccbrown/LAN_pipeline_minimal) | Orchestration on a cluster: data generation and network training as scheduled jobs | ...produce networks at scale rather than one at a time. |

## How the pieces connect

The chain runs in one direction, and the handoffs are files rather than
imports:

```text
ssm-simulators ──simulated training data──> LANfactory ──ONNX network──> HuggingFace
                                                                             │
                                                                             ▼
                                                                           HSSM
                                                          (downloads networks at run time)
```

Two things follow from this shape.

**Networks are artifacts, not code.** Many SSMs have no analytical likelihood.
For those, a neural network is trained once — offline, on simulated data — to
approximate the likelihood, and HSSM calls that network during sampling. The
trained networks live on
[HuggingFace](https://huggingface.co/franklab/HSSM); HSSM downloads what a
model needs on first use. You do not need LANfactory installed to use a
network someone else trained.

**The boundary is a contract, not a convention.** Any ONNX file HSSM loads
must expose one per-trial forward pass with every input dimension concrete;
HSSM batches across trials itself. That rule is stated once, with runnable
checks, in [The ONNX likelihood
contract](https://lnccbrown.github.io/HSSM/how_to/custom_onnx_likelihoods/).
It is also what makes the ecosystem open at the edges: a network trained in
[sbi](https://github.com/sbi-dev/sbi) or
[BayesFlow](https://github.com/bayesflow-org/bayesflow) becomes usable in HSSM
by exporting it to ONNX with LANfactory's exporters, after which HSSM loads it
with the same `loglik="model.onnx"` gesture it uses for its own networks — no
library-specific glue on the HSSM side.

## Which package answers your question

| Your question | Where it is answered |
|---|---|
| How do I fit this model to my data? | [HSSM — Learn](https://lnccbrown.github.io/HSSM/) |
| What models exist, and what are their parameters? | [ssm-simulators — Reference](https://lnccbrown.github.io/ssm-simulators/) |
| How do I simulate data from a model? | [ssm-simulators — Learn](https://lnccbrown.github.io/ssm-simulators/) |
| How do I add a model that does not exist yet? | [ssm-simulators — Contributing](https://lnccbrown.github.io/ssm-simulators/) |
| How do I train a likelihood network for it? | [LANfactory — Learn](https://lnccbrown.github.io/LANfactory/) |
| I trained a network in sbi or BayesFlow — now what? | [HSSM — Bring your own likelihood](https://lnccbrown.github.io/HSSM/how_to/external_trainers/) |
| How do I track training and data-generation runs? | [Tracking runs with MLflow](#tracking-runs-with-mlflow), below |
| Which versions work together? | [Version compatibility](#version-compatibility), below |
| What is this package in my traceback or lockfile? | [Supporting components](#supporting-components), below |
| How do I contribute across several packages? | [Development and coordination](#development-and-coordination), below |

Each site is organised the same way — **Learn**, **How-to guides**,
**Explanations**, **Reference** — so the tab you want sits in the same place
on all three.

## Supporting components

These are not packages you choose — they arrive with HSSM, or hold artifacts
it fetches. They are listed here because their names show up in tracebacks,
lockfiles, and download logs.

| Component | What it is |
|---|---|
| [`hddm-wfpt`](https://github.com/lnccbrown/hddm-wfpt) | The Cython implementation of the Wiener first-passage-time likelihood, inherited from HDDM. Installed with HSSM, and used for the analytical DDM likelihoods. |
| [`franklab/HSSM`](https://huggingface.co/franklab/HSSM) | The HuggingFace repository holding trained likelihood networks. HSSM downloads from it on first use of a model without an analytical likelihood. |
| [`franklab/ssms_gui`](https://huggingface.co/spaces/franklab/ssms_gui) | A HuggingFace Space for exploring SSM behaviour interactively, built on `ssm-simulators`. Useful for building intuition about what a parameter does. |
| conda-forge feedstocks | `hssm` and `ssm-simulators` are also published on conda-forge; the feedstock repositories carry the recipes. |

Third-party libraries that do real work under the hood — PyMC, Bambi, ArviZ,
JAX, PyTensor, ONNX Runtime — are dependencies rather than ecosystem
components, and their own documentation is the right reference for them.

## Development and coordination

Two repositories exist for people working *on* the ecosystem rather than with
it. You never need them to fit models, and neither is a Python package you
install alongside HSSM.

| Repository | Role |
|---|---|
| [HSSMSpine](https://github.com/lnccbrown/HSSMSpine) | The coordination repository. It holds no library code — it carries cross-repo context, shared development workflows, the release playbook, the shared documentation brand, and this page. Contributors working across two or more packages start here. |
| [HSSMCortex](https://github.com/lnccbrown/HSSMCortex) | The capability layer: a knowledge base of papers, modeling taxonomies, and curated guides, plus tooling that makes that knowledge queryable during development. |

If you are contributing to a single package, its own contributing guide is the
place to start; the spine matters when a change spans packages, such as adding
a model that needs a simulator, a trained network, and an HSSM configuration.

## Tracking runs with MLflow

Data generation (`ssms`) and network training (LANfactory) both log to
[MLflow](https://mlflow.org/). Point them at the same tracking store and the
two halves of a network's history end up in one place.

Two environment variables are the interface:

```bash
export MLFLOW_TRACKING_URI="sqlite:////absolute/path/to/tracking.db"
export MLFLOW_ARTIFACT_LOCATION="/absolute/path/to/artifacts"
```

`ssms` records each generation run — the generator and model configuration,
the `data_output_folder`, the number of files produced and their total size,
and tags for the run phase and any SLURM job it ran under. LANfactory records
each training run's configuration and metrics.

The two are linked explicitly rather than by convention: pass the data
generation run's experiment id to the trainer with
`--data-generation-experiment-id`, and LANfactory records the lineage — it can
also discover the training-data folder from MLflow instead of being told where
it is.

Per-package details — CLI flags, what exactly is logged, how to query it —
live with the packages:
[ssm-simulators](https://lnccbrown.github.io/ssm-simulators/core_tutorials/using_mlflow/)
and [LANfactory](https://lnccbrown.github.io/LANfactory/using_mlflow/).

## Version compatibility

The packages are released independently, in dependency order:
`ssm-simulators` → `LANfactory` → HSSM. Install floors, rather than pins, are
what the ecosystem guarantees:

| Consumer | Requires |
|---|---|
| HSSM | `ssm-simulators>=0.13.1` |
| LANfactory | `ssm-simulators>=0.13.1` |
| LAN_pipeline_minimal | `ssm-simulators>=0.13.2`, `lanfactory>=0.8` |

All released packages require Python 3.12 or newer. `pip install hssm` pulls a
compatible `ssm-simulators` automatically; you only need to think about this
when you are pinning an environment or building networks yourself.

## Where to ask

- **Usage questions and modeling advice** —
  [HSSM Discussions](https://github.com/lnccbrown/HSSM/discussions).
- **Bugs** — the issue tracker of the package the bug is in. If you are not
  sure which, HSSM's tracker is the right default and we will move it.
- **New models** — [ssm-simulators](https://github.com/lnccbrown/ssm-simulators/issues),
  which is where the model family lives.
