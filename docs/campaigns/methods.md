# Method and model stories

The useful HSSM story is a scientific contrast, not a list of model names. Start
with a question and retain the assumptions, diagnostics and limitations needed to
interpret the answer.

| Question | Public route | What to check |
|---|---|---|
| Can the model reproduce the response-time and choice patterns that matter? | [Posterior predictive gallery](../tutorials/ppc_gallery.ipynb) | Mismatch, uncertainty, omitted trials and relevant axes |
| What does posterior uncertainty imply for latent decision geometry? | [Model cartoon gallery](../tutorials/cartoon_gallery.ipynb) | Whether uncertainty and observed/predicted data are both visible |
| How do hierarchical assumptions change an estimate? | [Hierarchical modeling](../getting_started/hierarchical_modeling.ipynb) | Prior predictive behavior, shrinkage and group structure |
| What can attention add to a choice-and-response-time account? | [Attentional DDM walkthrough](../tutorials/attentional_ddm.ipynb) | Identification, fixation assumptions and causal limits |
| When is a likelihood artifact trustworthy? | [ONNX likelihood contract](../how_to/custom_onnx_likelihoods.md) | Shape contract, validation evidence and source artifact |

For reusable public figures, use the
[governed experiment templates](../reference/marketing-experiments.md). Their
outputs are synthetic; a simulation is never captioned as an observed effect.
