"""Bounded scientific templates for evidence-backed HSSM communications.

Only deterministic, synthetic Tier-S work belongs here. Fit, public-data, GPU and
HPC work require separately approved research runners. A request cannot provide
Python, a shell command, a URL, a dataset path, a dependency, or a resource override.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hssm.simulator import simulate_data

DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
ARTIFACT_ID = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?$")
REQUEST_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "brief_id",
        "template_id",
        "scientific_question",
        "expected_contrast",
        "data_class",
        "dataset_ref",
        "hssm_ref",
        "environment_digest",
        "inputs",
        "random_seeds",
        "resource_tier",
        "diagnostic_policy",
        "expected_outputs",
        "visibility",
        "approval",
    }
)
VISIBILITIES = frozenset(("internal", "gated_external", "public"))
MAX_ROWS = 50_000
FIGURE_SIZE = (10.0, 5.625)
FIGURE_DPI = 160


class ExperimentRequestError(ValueError):
    """A request is outside the registered Tier-S experiment boundary."""


@dataclass(frozen=True)
class FieldSpec:
    """One exact input field in a registered template."""

    kind: type
    default: int | float
    minimum: int | float
    maximum: int | float

    def validate(self, name: str, value: object) -> int | float:
        """Return a normalized numeric value or reject it."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ExperimentRequestError(f"input `{name}` must be numeric")
        try:
            finite = math.isfinite(float(value))
        except OverflowError:
            finite = False
        if not finite:
            raise ExperimentRequestError(f"input `{name}` must be finite")
        if self.kind is int and not isinstance(value, int):
            raise ExperimentRequestError(f"input `{name}` must be an integer")
        if not self.minimum <= value <= self.maximum:
            raise ExperimentRequestError(
                f"input `{name}` must be between {self.minimum} and {self.maximum}"
            )
        return self.kind(value)


@dataclass(frozen=True)
class ExperimentResult:
    """Pure template output before artifact serialization."""

    rows: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
    diagnostics: tuple[Mapping[str, Any], ...]
    caption: str
    alt_text: str
    draw: Callable[[Any], None]


@dataclass(frozen=True)
class ExperimentTemplate:
    """A named, versioned and bounded Tier-S experiment."""

    template_id: str
    version: str
    title: str
    description: str
    fields: Mapping[str, FieldSpec]
    runner: Callable[[Mapping[str, int | float], int], ExperimentResult]
    diagnostic_policy: str = "tier-s-scientific-v1"

    def definition(self) -> dict[str, Any]:
        """Return the immutable definition used to identify this template."""
        return {
            "template_id": self.template_id,
            "version": self.version,
            "title": self.title,
            "description": self.description,
            "resource_tier": "S",
            "diagnostic_policy": self.diagnostic_policy,
            "inputs": {
                name: {
                    "type": spec.kind.__name__,
                    "default": spec.default,
                    "minimum": spec.minimum,
                    "maximum": spec.maximum,
                }
                for name, spec in sorted(self.fields.items())
            },
        }


def canonical_json(value: Any) -> bytes:
    """Encode an artifact deterministically for hashing and storage."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _digest(value: Any) -> str:
    return _digest_bytes(canonical_json(value))


def _run_id(experiment_id: str, request_digest: str) -> str:
    """Derive a portable run id without rejecting a valid 128-character id."""
    suffix = request_digest[7:19]
    prefix = experiment_id[: 128 - len(suffix) - 1]
    return f"{prefix}-{suffix}"


def _asset_id(request_digest: str) -> str:
    """Derive a short content-bound asset identifier."""
    return f"asset-{request_digest[7:31]}"


def _diagnostic(
    name: str,
    passed: bool,
    observed: float,
    criterion: str,
    message: str,
) -> dict[str, Any]:
    return {
        "id": name,
        "status": "passed" if passed else "failed",
        "observed": float(observed),
        "criterion": criterion,
        "message": message,
    }


def _same_accuracy(inputs: Mapping[str, int | float], seed: int) -> ExperimentResult:
    n_trials = int(inputs["n_trials"])
    product = float(inputs["drift_boundary_product"])
    fast_boundary = float(inputs["fast_boundary"])
    cautious_boundary = float(inputs["cautious_boundary"])
    nondecision_time = float(inputs["nondecision_time"])
    params = (
        ("lower boundary", product / fast_boundary, fast_boundary, seed),
        ("higher boundary", product / cautious_boundary, cautious_boundary, seed + 1),
    )
    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, float]] = {}
    for label, drift, boundary, group_seed in params:
        data = simulate_data(
            model="ddm",
            theta={"v": drift, "a": boundary, "z": 0.5, "t": nondecision_time},
            size=n_trials,
            random_state=group_seed,
            output_df=False,
        )
        rt = np.asarray(data[:, 0], dtype=float)
        response = np.asarray(data[:, 1], dtype=float)
        accuracy = response == 1
        summaries[label] = {
            "drift": drift,
            "boundary": boundary,
            "accuracy": float(np.mean(accuracy)),
            "median_rt": float(np.median(rt)),
        }
        rows.extend(
            {
                "group": label,
                "trial": index,
                "rt": float(trial_rt),
                "response": int(trial_response),
                "is_upper_response": bool(trial_response == 1),
            }
            for index, (trial_rt, trial_response) in enumerate(
                zip(rt, response, strict=True)
            )
        )
    low = summaries["lower boundary"]
    high = summaries["higher boundary"]
    accuracy_gap = abs(low["accuracy"] - high["accuracy"])
    rt_gap = abs(low["median_rt"] - high["median_rt"])
    diagnostics = (
        _diagnostic(
            "matched_accuracy",
            accuracy_gap <= float(inputs["max_accuracy_gap"]),
            accuracy_gap,
            f"<= {inputs['max_accuracy_gap']}",
            "The two synthetic groups should have similar response accuracy.",
        ),
        _diagnostic(
            "different_timing",
            rt_gap >= float(inputs["min_median_rt_gap"]),
            rt_gap,
            f">= {inputs['min_median_rt_gap']}",
            "The matched-accuracy construction should retain a timing contrast.",
        ),
    )

    def draw(axes: Any) -> None:
        labels = list(summaries)
        axes[0].bar(labels, [summaries[label]["accuracy"] for label in labels])
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Upper-boundary response rate")
        axes[0].set_title("Similar choice rate")
        for label in labels:
            values = sorted(row["rt"] for row in rows if row["group"] == label)
            probability = np.arange(1, len(values) + 1) / len(values)
            axes[1].plot(values, probability, label=label)
        axes[1].set_xlabel("Response time (seconds)")
        axes[1].set_ylabel("Empirical cumulative probability")
        axes[1].set_title("Different timing distribution")
        axes[1].legend(frameon=False)

    return ExperimentResult(
        rows=tuple(rows),
        summary={
            "data_class": "synthetic",
            "model": "ddm",
            "groups": summaries,
            "accuracy_gap": accuracy_gap,
            "median_rt_gap": rt_gap,
        },
        diagnostics=diagnostics,
        caption=(
            "Synthetic DDM groups with a matched drift×boundary product show "
            "similar upper-boundary response rates while their response-time "
            "distributions differ."
        ),
        alt_text=(
            "Two panels compare synthetic DDM groups. Their response-rate bars are "
            "similar, while the higher-boundary group's response-time cumulative curve "
            "is shifted toward slower responses."
        ),
        draw=draw,
    )


def _partial_pooling(inputs: Mapping[str, int | float], seed: int) -> ExperimentResult:
    rng = np.random.default_rng(seed)
    n_groups = int(inputs["n_groups"])
    observations = int(inputs["observations_per_group"])
    population_sd = float(inputs["population_sd"])
    observation_sd = float(inputs["observation_sd"])
    truth = rng.normal(0, population_sd, n_groups)
    observed = truth + rng.normal(0, observation_sd / math.sqrt(observations), n_groups)
    weight = population_sd**2 / (population_sd**2 + observation_sd**2 / observations)
    pooled = weight * observed
    separate_rmse = float(np.sqrt(np.mean((observed - truth) ** 2)))
    pooled_rmse = float(np.sqrt(np.mean((pooled - truth) ** 2)))
    rows = tuple(
        {
            "group": index,
            "true_effect": float(truth[index]),
            "separate_estimate": float(observed[index]),
            "partial_pooling_estimate": float(pooled[index]),
        }
        for index in range(n_groups)
    )
    diagnostics = (
        _diagnostic(
            "finite_estimates",
            bool(np.isfinite(np.concatenate((truth, observed, pooled))).all()),
            float(np.isfinite(pooled).mean()),
            "all finite",
            "Every synthetic group estimate must be finite.",
        ),
        _diagnostic(
            "predeclared_rmse_comparison",
            pooled_rmse <= separate_rmse,
            pooled_rmse - separate_rmse,
            "partial-pooling RMSE <= separate-estimate RMSE",
            "This seed and parameterization should demonstrate shrinkage, "
            "not promise it universally.",
        ),
    )

    def draw(axes: Any) -> None:
        limit = max(
            1.0, float(np.max(np.abs(np.concatenate((truth, observed, pooled)))))
        )
        axes[0].scatter(truth, observed, alpha=0.75, label="separate")
        axes[0].scatter(truth, pooled, alpha=0.75, label="partial pooling")
        axes[0].plot([-limit, limit], [-limit, limit], color="black", linewidth=1)
        axes[0].set(xlabel="True synthetic effect", ylabel="Estimate")
        axes[0].set_title("Group estimates")
        axes[0].legend(frameon=False)
        axes[1].bar(["separate", "partial pooling"], [separate_rmse, pooled_rmse])
        axes[1].set_ylabel("RMSE for this simulation")
        axes[1].set_title("A predeclared comparison")

    return ExperimentResult(
        rows=rows,
        summary={
            "data_class": "synthetic",
            "n_groups": n_groups,
            "observations_per_group": observations,
            "shrinkage_weight": weight,
            "separate_rmse": separate_rmse,
            "partial_pooling_rmse": pooled_rmse,
        },
        diagnostics=diagnostics,
        caption=(
            "Synthetic group effects compare separate estimates with a "
            "known-variance partial-pooling estimator; the result is specific "
            "to this predeclared setup."
        ),
        alt_text=(
            "A scatter plot shows separate estimates farther from the identity "
            "line than partially pooled estimates for several small synthetic "
            "groups. A bar chart reports RMSE for this simulation only."
        ),
        draw=draw,
    )


def _attention_contrast(
    inputs: Mapping[str, int | float], seed: int
) -> ExperimentResult:
    rng = np.random.default_rng(seed)
    n_trials = int(inputs["n_trials"])
    eta = float(inputs["eta"])
    kappa = float(inputs["kappa"])
    item_1 = rng.normal(0.5, 0.8, n_trials)
    item_2 = rng.normal(0.5, 0.8, n_trials)
    attend_1 = kappa * (item_1 - eta * item_2)
    attend_2 = kappa * (eta * item_1 - item_2)
    contrast = attend_1 - attend_2
    rows = tuple(
        {
            "trial": index,
            "item_1_value": float(item_1[index]),
            "item_2_value": float(item_2[index]),
            "drift_attend_item_1": float(attend_1[index]),
            "drift_attend_item_2": float(attend_2[index]),
            "attention_contrast": float(contrast[index]),
        }
        for index in range(n_trials)
    )
    mean_abs_contrast = float(np.mean(np.abs(contrast)))
    diagnostics = (
        _diagnostic(
            "attention_changes_instantaneous_drift",
            mean_abs_contrast >= float(inputs["min_mean_abs_contrast"]),
            mean_abs_contrast,
            f">= {inputs['min_mean_abs_contrast']}",
            "The conceptual aDDM contrast should visibly change instantaneous drift.",
        ),
    )

    def draw(axes: Any) -> None:
        axes[0].scatter(item_1 - item_2, attend_1, alpha=0.35, label="attend item 1")
        axes[0].scatter(item_1 - item_2, attend_2, alpha=0.35, label="attend item 2")
        axes[0].axhline(0, color="black", linewidth=1)
        axes[0].set(
            xlabel="Synthetic item-value difference", ylabel="Instantaneous drift"
        )
        axes[0].set_title("Attention changes the evidence input")
        axes[0].legend(frameon=False)
        axes[1].hist(contrast, bins=30, color="#5B6CFF", alpha=0.8)
        axes[1].axvline(0, color="black", linewidth=1)
        axes[1].set(xlabel="Drift contrast", ylabel="Synthetic trials")
        axes[1].set_title("Not an observed purchasing effect")

    return ExperimentResult(
        rows=rows,
        summary={
            "data_class": "synthetic",
            "model": "conceptual standard-alternating aDDM drift",
            "eta": eta,
            "kappa": kappa,
            "mean_absolute_attention_contrast": mean_abs_contrast,
        },
        diagnostics=diagnostics,
        caption=(
            "A synthetic conceptual aDDM calculation shows how attending to "
            "different items changes instantaneous evidence; it is not an "
            "observed purchasing effect."
        ),
        alt_text=(
            "Synthetic item values map to different instantaneous drift values "
            "depending on which item is attended. A second panel shows the "
            "distribution of that within-trial conceptual contrast around zero."
        ),
        draw=draw,
    )


def _recovery_distinction(
    inputs: Mapping[str, int | float], seed: int
) -> ExperimentResult:
    rng = np.random.default_rng(seed)
    n_datasets = int(inputs["n_datasets"])
    noise = float(inputs["parameter_noise"])
    separation = float(inputs["model_separation"])
    truth = rng.uniform(-1.5, 1.5, n_datasets)
    estimate = truth + rng.normal(0, noise, n_datasets)
    parameter_correlation = float(np.corrcoef(truth, estimate)[0, 1])
    true_model = rng.integers(0, 2, n_datasets)
    score = separation * (2 * true_model - 1) + rng.normal(0, 1, n_datasets)
    selected_model = (score > 0).astype(int)
    model_accuracy = float(np.mean(selected_model == true_model))
    rows = tuple(
        {
            "dataset": index,
            "true_parameter": float(truth[index]),
            "estimated_parameter": float(estimate[index]),
            "true_model": int(true_model[index]),
            "selected_model": int(selected_model[index]),
            "selection_score": float(score[index]),
        }
        for index in range(n_datasets)
    )
    diagnostics = (
        _diagnostic(
            "parameter_recovery_identifiable",
            parameter_correlation >= float(inputs["min_parameter_correlation"]),
            parameter_correlation,
            f">= {inputs['min_parameter_correlation']}",
            "Parameter recovery is assessed only within the generating model.",
        ),
        _diagnostic(
            "model_recovery_nontrivial",
            0.5 < model_accuracy < 1.0,
            model_accuracy,
            "between chance and perfect",
            "Model recovery is a separate classification question.",
        ),
    )

    def draw(axes: Any) -> None:
        axes[0].scatter(truth, estimate, alpha=0.55)
        axes[0].plot([-1.5, 1.5], [-1.5, 1.5], color="black", linewidth=1)
        axes[0].set(xlabel="True parameter", ylabel="Estimated parameter")
        axes[0].set_title(f"Parameter recovery: r={parameter_correlation:.2f}")
        axes[1].bar(
            ["correct model", "wrong model"],
            [model_accuracy, 1 - model_accuracy],
            color=["#2A9D8F", "#E76F51"],
        )
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("Proportion of synthetic datasets")
        axes[1].set_title("Model recovery asks another question")

    return ExperimentResult(
        rows=rows,
        summary={
            "data_class": "synthetic",
            "parameter_recovery_correlation": parameter_correlation,
            "model_recovery_accuracy": model_accuracy,
        },
        diagnostics=diagnostics,
        caption=(
            "Synthetic parameter recovery and model recovery are shown side by "
            "side: estimating a parameter within one model is not selecting the "
            "generating model."
        ),
        alt_text=(
            "A scatter plot shows true versus estimated synthetic parameters with "
            "their correlation. A separate bar chart shows correct and incorrect "
            "model selection, emphasizing that the two recovery checks answer "
            "different questions."
        ),
        draw=draw,
    )


def _normal_probability_above(mean: float, sd: float, threshold: float) -> float:
    z = (threshold - mean) / (sd * math.sqrt(2))
    return 0.5 * math.erfc(z)


def _prior_sensitivity(
    inputs: Mapping[str, int | float], seed: int
) -> ExperimentResult:
    del seed  # The analytic conjugate calculation is deterministic by construction.
    observed_mean = float(inputs["observed_mean"])
    standard_error = float(inputs["standard_error"])
    threshold = float(inputs["decision_threshold"])
    configurations = (
        ("weak prior", float(inputs["weak_prior_sd"])),
        ("skeptical prior", float(inputs["skeptical_prior_sd"])),
    )
    rows_list: list[dict[str, Any]] = []
    for label, prior_sd in configurations:
        posterior_variance = 1 / (1 / prior_sd**2 + 1 / standard_error**2)
        posterior_mean = posterior_variance * observed_mean / standard_error**2
        posterior_sd = math.sqrt(posterior_variance)
        probability = _normal_probability_above(posterior_mean, posterior_sd, threshold)
        rows_list.append(
            {
                "prior": label,
                "prior_sd": prior_sd,
                "posterior_mean": posterior_mean,
                "posterior_sd": posterior_sd,
                "probability_above_threshold": probability,
            }
        )
    rows = tuple(rows_list)
    probability_gap = abs(
        rows[0]["probability_above_threshold"] - rows[1]["probability_above_threshold"]
    )
    diagnostics = (
        _diagnostic(
            "probabilities_valid",
            all(0 <= row["probability_above_threshold"] <= 1 for row in rows),
            probability_gap,
            "both probabilities in [0, 1]",
            "Posterior threshold probabilities must be numerically valid.",
        ),
        _diagnostic(
            "sensitivity_visible",
            probability_gap >= float(inputs["min_probability_gap"]),
            probability_gap,
            f">= {inputs['min_probability_gap']}",
            "The predeclared example should make prior sensitivity visible.",
        ),
    )

    def draw(axes: Any) -> None:
        x = np.linspace(
            min(row["posterior_mean"] - 4 * row["posterior_sd"] for row in rows),
            max(row["posterior_mean"] + 4 * row["posterior_sd"] for row in rows),
            500,
        )
        for row in rows:
            density = np.exp(
                -0.5 * ((x - row["posterior_mean"]) / row["posterior_sd"]) ** 2
            ) / (row["posterior_sd"] * math.sqrt(2 * math.pi))
            axes[0].plot(x, density, label=row["prior"])
        axes[0].axvline(threshold, color="black", linestyle="--", label="threshold")
        axes[0].set(xlabel="Synthetic effect", ylabel="Posterior density")
        axes[0].set_title("Same summary, different prior")
        axes[0].legend(frameon=False)
        axes[1].bar(
            [row["prior"] for row in rows],
            [row["probability_above_threshold"] for row in rows],
        )
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("Posterior probability above threshold")
        axes[1].set_title("A posterior is not the decision rule")

    return ExperimentResult(
        rows=rows,
        summary={
            "data_class": "synthetic",
            "observed_mean": observed_mean,
            "standard_error": standard_error,
            "decision_threshold": threshold,
            "probability_gap": probability_gap,
        },
        diagnostics=diagnostics,
        caption=(
            "A conjugate synthetic example holds the observed summary fixed and "
            "changes the prior, showing why a posterior threshold needs "
            "sensitivity analysis."
        ),
        alt_text=(
            "Two posterior density curves use the same synthetic observed summary but "
            "different prior scales. A bar chart shows different probabilities of "
            "exceeding the predeclared threshold."
        ),
        draw=draw,
    )


TEMPLATES: dict[str, ExperimentTemplate] = {
    template.template_id: template
    for template in (
        ExperimentTemplate(
            template_id="same-accuracy-different-mechanism-v1",
            version="1.0.0",
            title="Same accuracy, different mechanism",
            description=(
                "Seeded HSSM DDM simulation with matched drift×boundary product."
            ),
            fields={
                "n_trials": FieldSpec(int, 2000, 100, 10_000),
                "drift_boundary_product": FieldSpec(float, 1.2, 0.2, 3.0),
                "fast_boundary": FieldSpec(float, 0.8, 0.3, 1.5),
                "cautious_boundary": FieldSpec(float, 1.6, 1.0, 2.5),
                "nondecision_time": FieldSpec(float, 0.2, 0.0, 1.0),
                "max_accuracy_gap": FieldSpec(float, 0.05, 0.0, 0.2),
                "min_median_rt_gap": FieldSpec(float, 0.08, 0.0, 1.0),
            },
            runner=_same_accuracy,
        ),
        ExperimentTemplate(
            template_id="partial-pooling-shrinkage-v1",
            version="1.0.0",
            title="Partial pooling changes the question",
            description="Known-variance synthetic group-estimation comparison.",
            fields={
                "n_groups": FieldSpec(int, 24, 4, 100),
                "observations_per_group": FieldSpec(int, 8, 2, 100),
                "population_sd": FieldSpec(float, 0.5, 0.05, 3.0),
                "observation_sd": FieldSpec(float, 1.5, 0.05, 5.0),
            },
            runner=_partial_pooling,
        ),
        ExperimentTemplate(
            template_id="addm-attention-contrast-v1",
            version="1.0.0",
            title="Attention changes instantaneous evidence",
            description="Conceptual standard-alternating aDDM drift calculation.",
            fields={
                "n_trials": FieldSpec(int, 1000, 100, 10_000),
                "eta": FieldSpec(float, 0.3, 0.0, 1.0),
                "kappa": FieldSpec(float, 1.0, 0.1, 5.0),
                "min_mean_abs_contrast": FieldSpec(float, 0.1, 0.0, 5.0),
            },
            runner=_attention_contrast,
        ),
        ExperimentTemplate(
            template_id="parameter-vs-model-recovery-v1",
            version="1.0.0",
            title="Parameter and model recovery answer different questions",
            description=(
                "Paired synthetic recovery summaries with separate diagnostics."
            ),
            fields={
                "n_datasets": FieldSpec(int, 400, 100, 5000),
                "parameter_noise": FieldSpec(float, 0.35, 0.05, 2.0),
                "model_separation": FieldSpec(float, 0.45, 0.05, 2.0),
                "min_parameter_correlation": FieldSpec(float, 0.7, 0.0, 0.99),
            },
            runner=_recovery_distinction,
        ),
        ExperimentTemplate(
            template_id="prior-sensitivity-threshold-v1",
            version="1.0.0",
            title="Prior sensitivity before a decision threshold",
            description="Analytic conjugate-normal sensitivity example.",
            fields={
                "observed_mean": FieldSpec(float, 0.25, -3.0, 3.0),
                "standard_error": FieldSpec(float, 0.15, 0.01, 2.0),
                "decision_threshold": FieldSpec(float, 0.1, -3.0, 3.0),
                "weak_prior_sd": FieldSpec(float, 1.0, 0.05, 5.0),
                "skeptical_prior_sd": FieldSpec(float, 0.12, 0.01, 2.0),
                "min_probability_gap": FieldSpec(float, 0.05, 0.0, 0.9),
            },
            runner=_prior_sensitivity,
        ),
    )
}


def available_templates() -> list[dict[str, Any]]:
    """Return public-safe definitions for every registered template."""
    return [TEMPLATES[key].definition() for key in sorted(TEMPLATES)]


def validate_request(
    request: Mapping[str, Any],
) -> tuple[ExperimentTemplate, dict[str, int | float]]:
    """Validate a request and return its template plus normalized inputs."""
    if not isinstance(request, Mapping) or any(
        not isinstance(field, str) for field in request
    ):
        raise ExperimentRequestError("request must be an object with text field names")
    unknown = sorted(set(request) - REQUEST_FIELDS)
    missing = sorted(REQUEST_FIELDS - set(request))
    if unknown:
        raise ExperimentRequestError(f"unknown request field(s): {', '.join(unknown)}")
    if missing:
        raise ExperimentRequestError(f"missing request field(s): {', '.join(missing)}")
    if request["schema"] != "marketing-experiment-request-v1":
        raise ExperimentRequestError("schema must be marketing-experiment-request-v1")
    template_id = request["template_id"]
    if not isinstance(template_id, str) or template_id not in TEMPLATES:
        raise ExperimentRequestError(f"unknown template_id {template_id!r}")
    template = TEMPLATES[template_id]
    for field in ("experiment_id", "brief_id"):
        if (
            not isinstance(request[field], str)
            or ARTIFACT_ID.fullmatch(request[field]) is None
        ):
            raise ExperimentRequestError(
                f"`{field}` must be a portable 1-128 character identifier"
            )
    for field in (
        "scientific_question",
        "expected_contrast",
        "dataset_ref",
        "hssm_ref",
    ):
        if not isinstance(request[field], str) or not request[field].strip():
            raise ExperimentRequestError(f"`{field}` must be non-empty text")
    if request["data_class"] != "synthetic":
        raise ExperimentRequestError("Tier-S templates accept synthetic data only")
    if request["dataset_ref"] != "synthetic:generated":
        raise ExperimentRequestError(
            "Tier-S templates accept the generated synthetic dataset only"
        )
    if request["resource_tier"] != "S":
        raise ExperimentRequestError("this runner accepts resource_tier `S` only")
    if request["diagnostic_policy"] != template.diagnostic_policy:
        raise ExperimentRequestError(
            f"diagnostic_policy must be {template.diagnostic_policy!r}"
        )
    if (
        not isinstance(request["visibility"], str)
        or request["visibility"] not in VISIBILITIES
    ):
        raise ExperimentRequestError(
            "visibility must be internal, gated_external or public"
        )
    if not isinstance(request["environment_digest"], str) or not DIGEST.fullmatch(
        request["environment_digest"]
    ):
        raise ExperimentRequestError("environment_digest must be a sha256 digest")
    seeds = request["random_seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) != 1
        or isinstance(seeds[0], bool)
        or not isinstance(seeds[0], int)
        or not 0 <= seeds[0] < 2**32
    ):
        raise ExperimentRequestError("Tier-S requests require exactly one uint32 seed")
    if request["expected_outputs"] != ["summary", "data", "png", "svg", "asset"]:
        raise ExperimentRequestError(
            "expected_outputs must be ['summary', 'data', 'png', 'svg', 'asset']"
        )
    approval = request["approval"]
    if not isinstance(approval, Mapping) or approval != {"mode": "validated_intake"}:
        raise ExperimentRequestError(
            "Tier-S approval must be exactly {'mode': 'validated_intake'}"
        )
    supplied = request["inputs"]
    if not isinstance(supplied, Mapping) or any(
        not isinstance(field, str) for field in supplied
    ):
        raise ExperimentRequestError("inputs must be an object")
    unknown_inputs = sorted(set(supplied) - set(template.fields))
    if unknown_inputs:
        raise ExperimentRequestError(
            f"unknown input field(s): {', '.join(unknown_inputs)}"
        )
    normalized = {
        name: spec.validate(name, supplied.get(name, spec.default))
        for name, spec in template.fields.items()
    }
    row_estimate = max(
        int(normalized.get("n_trials", 0)) * 2,
        int(normalized.get("n_groups", 0)),
        int(normalized.get("n_datasets", 0)),
        2,
    )
    if row_estimate > MAX_ROWS:
        raise ExperimentRequestError(f"request exceeds the {MAX_ROWS}-row Tier-S quota")
    return template, normalized


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json(value) + b"\n")


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ExperimentRequestError("template returned no machine-readable rows")
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise ExperimentRequestError("template rows do not share one exact schema")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(result: ExperimentResult, png: Path, svg: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    settings: dict[Any, Any] = {
        "figure.dpi": FIGURE_DPI,
        "font.family": "DejaVu Sans",
        "svg.hashsalt": "hssm-marketing-v1",
    }
    with mpl.rc_context(settings):
        figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, constrained_layout=True)
        result.draw(axes)
        figure.suptitle("Synthetic demonstration", fontsize=14, fontweight="bold")
        figure.savefig(
            png,
            dpi=FIGURE_DPI,
            facecolor="white",
            metadata={"Software": "HSSM governed experiment runner"},
        )
        figure.savefig(
            svg,
            facecolor="white",
            metadata={"Creator": "HSSM governed experiment runner", "Date": None},
        )
        plt.close(figure)
    svg_payload = svg.read_bytes()
    svg_payload = re.sub(
        rb"<!DOCTYPE\s+svg\b.*?>\s*",
        b"",
        svg_payload,
        count=1,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if b"<!DOCTYPE" in svg_payload.upper():
        raise ExperimentRequestError("generated SVG contains an unsupported doctype")
    svg.write_bytes(svg_payload)


def _package_version() -> str:
    try:
        return importlib.metadata.version("HSSM")
    except importlib.metadata.PackageNotFoundError:
        return "uninstalled-source"


def run_experiment(
    request: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Run one validated template and write an immutable evidence package.

    The caller must provide a missing or empty output directory. Existing artifacts
    are never overwritten, and the runner performs no network access.
    """
    template, inputs = validate_request(request)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ExperimentRequestError("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)

    request_digest = _digest(request)
    run_id = _run_id(request["experiment_id"], request_digest)
    result = template.runner(inputs, request["random_seeds"][0])
    if len(result.rows) > MAX_ROWS:
        raise ExperimentRequestError("template output exceeded the Tier-S row quota")
    status = (
        "passed"
        if result.diagnostics
        and all(entry["status"] == "passed" for entry in result.diagnostics)
        else "failed"
    )

    request_path = output / "request.json"
    summary_path = output / "summary.json"
    data_path = output / "data.csv"
    png_path = output / "figure.png"
    svg_path = output / "figure.svg"
    asset_path = output / "asset.json"
    _write_json(request_path, request)
    _write_json(summary_path, result.summary)
    _write_rows(data_path, result.rows)
    _save_figure(result, png_path, svg_path)

    output_entries = []
    for path, media_type, role in (
        (request_path, "application/json", "request"),
        (summary_path, "application/json", "summary"),
        (data_path, "text/csv", "data"),
        (png_path, "image/png", "master_visual"),
        (svg_path, "image/svg+xml", "master_visual"),
    ):
        payload = path.read_bytes()
        output_entries.append(
            {
                "path": path.name,
                "media_type": media_type,
                "role": role,
                "digest": _digest_bytes(payload),
                "bytes": len(payload),
                "claim_use": "eligible" if status == "passed" else "blocked",
            }
        )

    masters = {
        entry["media_type"]: {
            "path": entry["path"],
            "digest": entry["digest"],
            "dimensions": [1600, 900] if entry["media_type"] == "image/png" else None,
        }
        for entry in output_entries
        if entry["role"] == "master_visual"
    }
    asset = {
        "schema": "marketing-asset-v1",
        "asset_id": _asset_id(request_digest),
        "brief_id": request["brief_id"],
        "source_refs": [request["dataset_ref"]],
        "run_ids": [run_id],
        "creator": {
            "tool": "hssm.marketing",
            "template_id": template.template_id,
            "template_version": template.version,
        },
        "caption": result.caption,
        "alt_text": result.alt_text,
        "licence": "HSSM repository LICENSE; separate publication review required",
        "consent": {"required": False, "status": "not_applicable_synthetic_data"},
        "master": masters,
        "derivatives": {},
        "transformations": [],
        "review_state": "proposed" if status == "passed" else "rejected",
        "data_class": "synthetic",
    }
    _write_json(asset_path, asset)
    asset_payload = asset_path.read_bytes()
    output_entries.append(
        {
            "path": asset_path.name,
            "media_type": "application/json",
            "role": "asset_metadata",
            "digest": _digest_bytes(asset_payload),
            "bytes": len(asset_payload),
            "claim_use": "eligible" if status == "passed" else "blocked",
        }
    )

    runtime_provenance = {
        "hssm_requested_ref": request["hssm_ref"],
        "hssm_version": _package_version(),
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "template_id": template.template_id,
        "template_version": template.version,
        "template_digest": _digest(template.definition()),
        "environment_digest": request["environment_digest"],
        "network_access": "requires-external-isolation",
    }
    logs = [
        "validated exact Tier-S request",
        f"executed template {template.template_id}@{template.version}",
        f"wrote {len(result.rows)} machine-readable rows",
    ]
    result_digest = _digest(
        {
            "summary": result.summary,
            "diagnostics": result.diagnostics,
            "outputs": {entry["path"]: entry["digest"] for entry in output_entries},
            "runtime_provenance": runtime_provenance,
            "logs": logs,
        }
    )
    manifest = {
        "schema": "marketing-experiment-run-v1",
        "run_id": run_id,
        "request_digest": request_digest,
        "runtime_provenance": runtime_provenance,
        "logs": logs,
        "outputs": output_entries,
        "diagnostics": list(result.diagnostics),
        "result_digest": result_digest,
        "status": status,
    }
    _write_json(output / "run.json", manifest)
    return manifest
