# The HSSM ecosystem

The HSSM ecosystem separates simulation, network training, validated artifact
production, inference, capability support, and cross-repository coordination.
Most users begin with HSSM; contributors use this map when a question crosses
one of those ownership boundaries.

This is the canonical public map, hosted by
[HSSM](https://lnccbrown.github.io/HSSM/). HSSMSpine maintains the shared source
and synchronization contract while its repository remains private, so the
established [HSSM ecosystem URL](https://lnccbrown.github.io/HSSM/ecosystem/)
continues to be the durable entry point.

<!-- Preserve fragments from the established HSSM ecosystem URL. -->
<span id="the-packages-you-install"></span>
<span id="development-and-coordination"></span>

## The six native sites

| Documentation site | Owns | Start here when you want to... |
|---|---|---|
| [HSSM](https://lnccbrown.github.io/HSSM/) | User-facing Bayesian inference for sequential sampling models | fit and diagnose a model with the supported HSSM interfaces |
| [ssm-simulators](https://lnccbrown.github.io/ssm-simulators/) | Simulator behavior, model configuration, and simulation-result contracts | simulate data or extend the simulator family |
| [LANfactory](https://lnccbrown.github.io/LANfactory/) | Likelihood-network training and export | train or export a likelihood approximation |
| [LAN_pipeline_minimal](https://lnccbrown.github.io/LAN_pipeline_minimal/) | Validated generation, training, staging, and promotion workflows | produce and validate network artifacts through the supported operator path |
| [HSSMCortex](https://lnccbrown.github.io/HSSMCortex/) | Optional knowledge and capability support for ecosystem development | query curated knowledge or develop shared capability content |
| [HSSMSpine](https://lnccbrown.github.io/HSSMSpine/) | Cross-repository context, contracts, launchers, and maintenance; the source workspace remains private | coordinate a change spanning repositories |

## How the pieces connect

```text
ssm-simulators ──> LANfactory ──> trained networks ──> artifact store
       │                                                   │
       │                                                   v
       └───────────────────────────────────────────────> HSSM

LAN_pipeline_minimal orchestrates generation, training, validation,
staging, and promotion. HSSMSpine coordinates repository work.
HSSMCortex provides optional knowledge and capability support.
```

Python dependency edges and artifact handoffs are different. HSSM and
LANfactory consume ssm-simulators as a package. Trained networks cross the
boundary as artifacts plus metadata, and HSSM owns consumer-side loading.
LAN_pipeline_minimal owns the validated promotion path; the artifact store is
a delivery surface, not the source of training policy.

The ONNX consumer contract is owned by HSSM. Producers and third-party
exporters should follow the
[rendered HSSM contract](https://lnccbrown.github.io/HSSM/how_to/custom_onnx_likelihoods/)
rather than a duplicate summary here.

<span id="which-package-answers-your-question"></span>
<span id="tracking-runs-with-mlflow"></span>

## Which site answers the question?

| Question | Owning documentation |
|---|---|
| How do I fit or diagnose a model? | [HSSM](https://lnccbrown.github.io/HSSM/) |
| How do I simulate data or change simulator output? | [ssm-simulators](https://lnccbrown.github.io/ssm-simulators/) |
| How do I train or export a likelihood network? | [LANfactory](https://lnccbrown.github.io/LANfactory/) |
| How do I validate and promote a trained artifact? | [LAN pipeline](https://lnccbrown.github.io/LAN_pipeline_minimal/) |
| How do I track generation and training runs? | [LAN pipeline — MLflow](https://lnccbrown.github.io/LAN_pipeline_minimal/how-to/track-with-mlflow/) |
| How do I query or extend the capability layer? | [HSSMCortex](https://lnccbrown.github.io/HSSMCortex/) |
| How do I coordinate a cross-repository change? | [HSSMSpine](https://lnccbrown.github.io/HSSMSpine/) |

All native sites use the same top-level path—**Home**, **Learn**,
**How-to guides**, **Explanations**, and **Reference**—while keeping package
behavior in the repository that implements and tests it.

<span id="supporting-components"></span>

## Artifact and integration surfaces

- [`franklab/HSSM`](https://huggingface.co/franklab/HSSM) delivers trained
  network artifacts consumed by HSSM.
- [`franklab/ssms_gui`](https://huggingface.co/spaces/franklab/ssms_gui)
  provides an interactive simulator-based visualization surface.
- conda-forge feedstocks package HSSM and ssm-simulators after their PyPI
  releases.

Third-party inference, array, and visualization libraries retain their own
documentation and release policies.

<span id="version-compatibility"></span>

## Versions and releases

The packages release independently. Exact compatibility floors belong in each
consumer's project metadata and rendered installation reference. Coordinated
releases follow the dependency and artifact flow: ssm-simulators first,
LANfactory when required, validated networks when changed, and HSSM last.

Contributors should use the public
[HSSMSpine release procedure](https://lnccbrown.github.io/HSSMSpine/release-playbook/)
rather than copying a version table that can drift.
LANfactory tags are bare; HSSM and ssm-simulators tags use a `v` prefix.

## Where to ask

- Usage and modeling questions: [HSSM Discussions](https://github.com/lnccbrown/HSSM/discussions).
- A confirmed bug: the issue tracker of the repository that owns the behavior.
- An uncertain cross-repository problem: begin with HSSM, then use
  [HSSMSpine's source-ownership reference](https://lnccbrown.github.io/HSSMSpine/reference/source-ownership/)
  to route it.
