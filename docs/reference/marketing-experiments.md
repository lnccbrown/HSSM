# Governed public-evidence experiment templates

`hssm.marketing` exposes a small Tier-S runner for reproducible, synthetic teaching
and communication artifacts. It is deliberately not a general notebook, Python, shell,
fit, public-data, GPU or network execution surface.

## Registered templates

| Template id | Question | Output |
|---|---|---|
| `same-accuracy-different-mechanism-v1` | Can similar choice rates coexist with different response-time mechanisms? | Seeded `hssm.simulate_data()` DDM comparison |
| `partial-pooling-shrinkage-v1` | How can shrinkage change noisy group estimates? | Known-variance synthetic group comparison |
| `addm-attention-contrast-v1` | How does attended versus unattended value enter instantaneous evidence? | Conceptual standard-alternating aDDM drift contrast |
| `parameter-vs-model-recovery-v1` | Why are parameter and model recovery different checks? | Paired synthetic recovery summaries |
| `prior-sensitivity-threshold-v1` | How can a predeclared prior affect a threshold probability? | Conjugate-normal sensitivity calculation |

List the exact input schemas with:

```bash
python -m hssm.marketing --list-templates
```

## Request boundary

The runner accepts one `marketing-experiment-request-v1` JSON object with an exact
top-level schema, one registered template id, one uint32 seed, synthetic data, Tier S,
the template's diagnostic policy, a pinned HSSM ref and environment digest, and the
fixed output list. Inputs accept only the named numeric fields and bounded ranges.

The following fail before execution: arbitrary code or commands, URLs or dataset
paths, unregistered fields, dependency changes, non-synthetic data, fit/extended tiers,
more than one seed, quota escalation, or a non-empty output directory.

```json
{
  "schema": "marketing-experiment-request-v1",
  "experiment_id": "experiment-B01-001",
  "brief_id": "brief-B01-001",
  "template_id": "same-accuracy-different-mechanism-v1",
  "scientific_question": "Can matched choice rates hide different timing?",
  "expected_contrast": "Choice rates remain close while median response times differ.",
  "data_class": "synthetic",
  "dataset_ref": "synthetic:generated",
  "hssm_ref": "v0.5.0",
  "environment_digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
  "inputs": {"n_trials": 2000},
  "random_seeds": [1729],
  "resource_tier": "S",
  "diagnostic_policy": "tier-s-scientific-v1",
  "expected_outputs": ["summary", "data", "png", "svg", "asset"],
  "visibility": "internal",
  "approval": {"mode": "validated_intake"}
}
```

Run it only in an isolated, network-disabled worker with a fresh output directory:

```bash
python -m hssm.marketing request.json artifact/
```

## Artifact convention

Each run writes the exact request, machine-readable summary and CSV, deterministic
PNG/SVG master figures, proposed asset metadata with caption and alt text, and a
`marketing-experiment-run-v1` manifest. Every file carries a SHA-256 digest. The run
records the template and environment digests, bounded logs, diagnostics, result digest
and `passed`/`failed` status. The result digest also covers runtime provenance and the
bounded log records, so either must be re-attested if an isolation wrapper adds its own
verified launcher/profile identity. Generated SVG masters omit external doctypes so the
governing boundary can enforce a passive, local-reference-only representation. The
standalone runner records the requested ref and environment digest; the governing
worker is responsible for matching them to its content-addressed runtime image,
canonical runtime manifest, signed isolation receipt, and launcher/profile identities
before claim use.

All five templates label data as synthetic. A `failed` run remains inspectable but every
output is marked `claim_use=blocked` and its asset is rejected. Even a passed asset stays
`proposed`: scientific, accessibility and destination review happen with the final text
and derivative. The runner never turns a simulation into an observed effect or grants
publication approval.
