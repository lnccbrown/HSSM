# The HSSM ecosystem

HSSM is the user-facing package of a four-part toolchain. Most users never
leave it: install HSSM, fit models, publish. You end up here when a question
crosses a package boundary — where does a likelihood network come from, which
version pairs with which, why does an ONNX file have to look a certain way.

This page is the map. It is maintained in
[HSSMSpine](https://github.com/lnccbrown/HSSMSpine), the ecosystem's
coordination repository, and published here so there is one description of the
whole rather than three partial ones.

## The packages

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
It is what lets networks trained in
[sbi](https://github.com/sbi-dev/sbi) or
[BayesFlow](https://github.com/bayesflow-org/bayesflow) drop into HSSM
unchanged.

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

Each site is organised the same way — **Learn**, **How-to guides**,
**Explanations**, **Reference** — so the tab you want sits in the same place
on all three.

## Tracking runs with MLflow

Data generation (`ssms`) and network training (LANfactory) both log to
[MLflow](https://mlflow.org/), and they log to the *same* tracking store when
you point them at one. That is what makes a trained network traceable back to
the data that produced it.

The environment variable is the whole interface:

```bash
export MLFLOW_TRACKING_URI="file:///absolute/path/to/mlruns"   # or an http:// server
```

With that set, `ssms` records each generation run (model, output folder, file
count, config hash) and LANfactory records each training run (model, network
type, backend, training-data folder, the run UUID that appears in the
artifact filenames). Joining the two is what lets you answer "which data
trained this network" months later.

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
