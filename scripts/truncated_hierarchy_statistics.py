"""Pure statistical contracts for the truncated-hierarchy qualification.

The helpers in this module consume compact, validated posterior summaries. They do
not build models, sample, read files, or infer candidate/control membership. Every
family-level API takes one declared family and rejects mixed-family input so control
results cannot dilute candidate evidence. Family evaluators also require the full
predeclared scenario/parameter unit set, preventing omitted evidence from shrinking
the multiplicity correction. Randomized SBC tie ranks are verified against a seed
supplied by the frozen experiment plan.

Combined rank-normalized R-hat is intentionally left to the sampling integration,
where the raw per-chain draws are available. Mean agreement can be checked here from
posterior means and their Monte Carlo standard errors.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any, Literal, TypeVar

import numpy as np
from scipy.stats import beta, binom, binomtest

Family = Literal["candidate", "control"]
AnalysisUnit = tuple[str, str]
FAMILIES: frozenset[str] = frozenset({"candidate", "control"})
COVERAGE_LEVELS: tuple[tuple[float, str, str], ...] = (
    (0.90, "q05", "q95"),
    (0.95, "q025", "q975"),
)
SUMMARY_FIELDS = {
    "family",
    "scenario_id",
    "parameter_id",
    "replicate",
    "truth",
    "posterior_mean",
    "posterior_sd",
    "posterior_mcse",
    "q025",
    "q05",
    "q50",
    "q95",
    "q975",
}
SBC_FIELDS = {
    "rank_less",
    "rank_equal",
    "rank_tie_index",
    "rank",
    "rank_draw_count",
}


class QualificationStatisticsError(ValueError):
    """Raised when a statistical input violates the predeclared contract."""


@dataclass(frozen=True, slots=True)
class ParameterSummary:
    """Validated natural-scale posterior summary for one parameter and replicate."""

    family: Family
    scenario_id: str
    parameter_id: str
    replicate: int
    truth: float
    posterior_mean: float
    posterior_sd: float
    posterior_mcse: float
    q025: float
    q05: float
    q50: float
    q95: float
    q975: float
    rank_less: int | None = None
    rank_equal: int | None = None
    rank_tie_index: int | None = None
    rank: int | None = None
    rank_draw_count: int | None = None

    @property
    def analysis_unit(self) -> AnalysisUnit:
        """Return the scenario/parameter unit that may be aggregated."""
        return (self.scenario_id, self.parameter_id)

    @property
    def has_sbc_rank(self) -> bool:
        """Whether the complete optional SBC rank payload is present."""
        return self.rank is not None


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """Closed confidence interval."""

    lower: float
    upper: float

    def contains(self, value: float) -> bool:
        """Return whether ``value`` lies in the closed interval."""
        return self.lower <= value <= self.upper


@dataclass(frozen=True, slots=True)
class CoverageCheck:
    """Multiplicity-adjusted exact coverage result for one analysis unit."""

    family: Family
    scenario_id: str
    parameter_id: str
    nominal: float
    successes: int
    replicates: int
    family_comparisons: int
    alpha_per_comparison: float
    interval: ConfidenceInterval
    passed: bool


@dataclass(frozen=True, slots=True)
class SbcRankCheck:
    """Simultaneous DKW-Bonferroni rank-ECDF result for one analysis unit."""

    family: Family
    scenario_id: str
    parameter_id: str
    replicates: int
    rank_draw_count: int
    family_curves: int
    epsilon: float
    max_abs_deviation: float
    passed: bool


@dataclass(frozen=True, slots=True)
class BiasCheck:
    """Magnitude-gated bias result with descriptive sign-test diagnostics."""

    family: Family
    scenario_id: str
    parameter_id: str
    replicates: int
    mean_standardized_error: float
    abs_mean_standardized_error: float
    median_standardized_error: float
    standardized_rmse: float
    sign_test_pvalue: float
    holm_rejected: bool
    magnitude_passed: bool


@dataclass(frozen=True, slots=True)
class BackendMeanCheck:
    """MCSE-standardized posterior-mean agreement for one backend pair."""

    family: Family
    parameter_id: str
    replicate: int
    mcse_z: float
    limit: float
    passed: bool


@dataclass(frozen=True, slots=True)
class CoveragePower:
    """Power and acceptance range for one nominal coverage target."""

    nominal: float
    alternative: float
    acceptance_min: int
    acceptance_max: int
    power: float


@dataclass(frozen=True, slots=True)
class CoveragePowerDesign:
    """Smallest replicate design satisfying every requested power target."""

    replicates: int
    family_comparisons: int
    familywise_alpha: float
    alpha_per_comparison: float
    target_power: float
    targets: tuple[CoveragePower, ...]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise QualificationStatisticsError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise QualificationStatisticsError(f"{field} must be finite")
    return result


def _validate_probability(value: float, field: str, *, allow_one: bool = False) -> None:
    upper_valid = value <= 1 if allow_one else value < 1
    if value <= 0 or not upper_valid:
        interval = "(0, 1]" if allow_one else "(0, 1)"
        raise QualificationStatisticsError(f"{field} must lie in {interval}")


def validate_parameter_summary(
    value: Mapping[str, Any],
    *,
    require_sbc: bool = False,
    expected_sbc_tie_seed: int | None = None,
) -> ParameterSummary:
    """Validate one strict per-parameter summary and return an immutable value.

    When SBC primitives are present, ``expected_sbc_tie_seed`` must be supplied
    from trusted plan metadata. The reported randomized tie index is recomputed
    from that seed and rejected if it was caller-selected.
    """
    if not isinstance(value, Mapping):
        raise QualificationStatisticsError("summary must be a mapping")
    keys = set(value)
    rank_keys = keys & SBC_FIELDS
    if rank_keys and rank_keys != SBC_FIELDS:
        missing_rank_fields = sorted(SBC_FIELDS - rank_keys)
        raise QualificationStatisticsError(
            "SBC rank primitives must be all present or all absent; "
            f"missing {missing_rank_fields}"
        )
    expected = SUMMARY_FIELDS | (SBC_FIELDS if rank_keys else set())
    missing = SUMMARY_FIELDS - keys
    unknown = keys - expected
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unknown {sorted(unknown)}")
        raise QualificationStatisticsError(
            f"summary has invalid fields: {', '.join(details)}"
        )
    if require_sbc and not rank_keys:
        raise QualificationStatisticsError("summary requires SBC rank primitives")
    if rank_keys and expected_sbc_tie_seed is None:
        raise QualificationStatisticsError(
            "SBC rank primitives require expected_sbc_tie_seed"
        )
    if not rank_keys and expected_sbc_tie_seed is not None:
        raise QualificationStatisticsError(
            "expected_sbc_tie_seed requires SBC rank primitives"
        )

    family = value["family"]
    if family not in FAMILIES:
        raise QualificationStatisticsError("family must be candidate or control")
    for field in ("scenario_id", "parameter_id"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise QualificationStatisticsError(f"{field} must be a non-empty string")
    replicate = value["replicate"]
    if not _is_int(replicate) or replicate < 0:
        raise QualificationStatisticsError("replicate must be a non-negative integer")

    numeric = {
        field: _finite_number(value[field], field)
        for field in (
            "truth",
            "posterior_mean",
            "posterior_sd",
            "posterior_mcse",
            "q025",
            "q05",
            "q50",
            "q95",
            "q975",
        )
    }
    if numeric["posterior_sd"] <= 0:
        raise QualificationStatisticsError("posterior_sd must be positive")
    if numeric["posterior_mcse"] <= 0:
        raise QualificationStatisticsError("posterior_mcse must be positive")
    quantiles = [
        numeric["q025"],
        numeric["q05"],
        numeric["q50"],
        numeric["q95"],
        numeric["q975"],
    ]
    if quantiles != sorted(quantiles):
        raise QualificationStatisticsError("posterior quantiles must be ordered")

    ranks: dict[str, int | None] = dict.fromkeys(SBC_FIELDS)
    if rank_keys:
        for field in SBC_FIELDS:
            rank_value = value[field]
            if not _is_int(rank_value) or rank_value < 0:
                raise QualificationStatisticsError(
                    f"{field} must be a non-negative integer"
                )
            ranks[field] = rank_value
        draw_count = ranks["rank_draw_count"]
        rank_less = ranks["rank_less"]
        rank_equal = ranks["rank_equal"]
        tie_index = ranks["rank_tie_index"]
        rank = ranks["rank"]
        assert draw_count is not None
        assert rank_less is not None
        assert rank_equal is not None
        assert tie_index is not None
        assert rank is not None
        if draw_count <= 0:
            raise QualificationStatisticsError("rank_draw_count must be positive")
        if rank_less + rank_equal > draw_count:
            raise QualificationStatisticsError(
                "rank_less + rank_equal cannot exceed rank_draw_count"
            )
        if tie_index > rank_equal:
            raise QualificationStatisticsError(
                "rank_tie_index cannot exceed rank_equal"
            )
        assert expected_sbc_tie_seed is not None
        expected_tie_index = derive_sbc_rank_tie_index(
            tie_seed=expected_sbc_tie_seed,
            family=family,
            scenario_id=value["scenario_id"],
            parameter_id=value["parameter_id"],
            replicate=replicate,
            rank_less=rank_less,
            rank_equal=rank_equal,
            rank_draw_count=draw_count,
        )
        if tie_index != expected_tie_index:
            raise QualificationStatisticsError(
                "rank_tie_index does not match the deterministic tie index "
                "derived from expected_sbc_tie_seed"
            )
        if rank != rank_less + tie_index or rank > draw_count:
            raise QualificationStatisticsError(
                "rank must equal rank_less + rank_tie_index and be in range"
            )

    return ParameterSummary(
        family=family,
        scenario_id=value["scenario_id"],
        parameter_id=value["parameter_id"],
        replicate=replicate,
        truth=numeric["truth"],
        posterior_mean=numeric["posterior_mean"],
        posterior_sd=numeric["posterior_sd"],
        posterior_mcse=numeric["posterior_mcse"],
        q025=numeric["q025"],
        q05=numeric["q05"],
        q50=numeric["q50"],
        q95=numeric["q95"],
        q975=numeric["q975"],
        rank_less=ranks["rank_less"],
        rank_equal=ranks["rank_equal"],
        rank_tie_index=ranks["rank_tie_index"],
        rank=ranks["rank"],
        rank_draw_count=ranks["rank_draw_count"],
    )


def validate_parameter_summaries(
    values: Sequence[Mapping[str, Any]],
    *,
    require_sbc: bool = False,
    expected_sbc_tie_seed: int | None = None,
) -> tuple[ParameterSummary, ...]:
    """Validate a sequence of per-parameter summaries."""
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise QualificationStatisticsError("summaries must be a sequence")
    return tuple(
        validate_parameter_summary(
            value,
            require_sbc=require_sbc,
            expected_sbc_tie_seed=expected_sbc_tie_seed,
        )
        for value in values
    )


def derive_sbc_rank_tie_index(
    *,
    tie_seed: int,
    family: Family,
    scenario_id: str,
    parameter_id: str,
    replicate: int,
    rank_less: int,
    rank_equal: int,
    rank_draw_count: int,
) -> int:
    """Derive an auditable, unbiased randomized-rank tie index.

    ``tie_seed`` must come from the frozen experiment plan, not from a result
    producer. The identity and auditable rank counts are domain-separated into a
    SHAKE-256 stream. Rejection sampling avoids modulo bias for every tie count.
    """
    if not _is_int(tie_seed) or tie_seed < 0:
        raise QualificationStatisticsError("tie_seed must be a non-negative integer")
    if family not in FAMILIES:
        raise QualificationStatisticsError("family must be candidate or control")
    for field, identity_value in (
        ("scenario_id", scenario_id),
        ("parameter_id", parameter_id),
    ):
        if not isinstance(identity_value, str) or not identity_value.strip():
            raise QualificationStatisticsError(f"{field} must be a non-empty string")
    for field, count_value in (
        ("replicate", replicate),
        ("rank_less", rank_less),
        ("rank_equal", rank_equal),
        ("rank_draw_count", rank_draw_count),
    ):
        if not _is_int(count_value) or count_value < 0:
            raise QualificationStatisticsError(
                f"{field} must be a non-negative integer"
            )
    if rank_draw_count <= 0:
        raise QualificationStatisticsError("rank_draw_count must be positive")
    if rank_less + rank_equal > rank_draw_count:
        raise QualificationStatisticsError(
            "rank_less + rank_equal cannot exceed rank_draw_count"
        )

    payload = json.dumps(
        {
            "contract": "hssm-truncated-hierarchy-sbc-tie-v1",
            "family": family,
            "parameter_id": parameter_id,
            "rank_draw_count": rank_draw_count,
            "rank_equal": rank_equal,
            "rank_less": rank_less,
            "replicate": replicate,
            "scenario_id": scenario_id,
            "tie_seed": tie_seed,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    bucket_count = rank_equal + 1
    byte_count = max(8, (bucket_count.bit_length() + 7) // 8)
    sample_space = 1 << (8 * byte_count)
    acceptance_limit = sample_space - sample_space % bucket_count
    counter = 0
    while True:
        digest = hashlib.shake_256(
            payload + counter.to_bytes(8, byteorder="big")
        ).digest(byte_count)
        draw = int.from_bytes(digest, byteorder="big")
        if draw < acceptance_limit:
            return draw % bucket_count
        counter += 1


def coverage_indicators(summary: ParameterSummary) -> dict[float, bool]:
    """Derive inclusive equal-tailed 90% and 95% coverage indicators."""
    return {
        nominal: getattr(summary, lower) <= summary.truth <= getattr(summary, upper)
        for nominal, lower, upper in COVERAGE_LEVELS
    }


def _complete_family_groups(
    summaries: Sequence[ParameterSummary],
    *,
    family: Family,
    expected_replicates: int,
    expected_units: Sequence[AnalysisUnit],
) -> dict[AnalysisUnit, tuple[ParameterSummary, ...]]:
    if family not in FAMILIES:
        raise QualificationStatisticsError("family must be candidate or control")
    if not _is_int(expected_replicates) or expected_replicates <= 0:
        raise QualificationStatisticsError("expected_replicates must be positive")
    declared_units = _validate_expected_units(expected_units)

    grouped: defaultdict[AnalysisUnit, list[ParameterSummary]] = defaultdict(list)
    for summary in summaries:
        if not isinstance(summary, ParameterSummary):
            raise QualificationStatisticsError(
                "family evaluators require validated ParameterSummary values"
            )
        if summary.family != family:
            raise QualificationStatisticsError(
                f"mixed families are forbidden: expected {family}, got {summary.family}"
            )
        grouped[summary.analysis_unit].append(summary)

    actual_units = set(grouped)
    declared_unit_set = set(declared_units)
    if actual_units != declared_unit_set:
        raise QualificationStatisticsError(
            "analysis unit set mismatch: "
            f"missing {sorted(declared_unit_set - actual_units)}, "
            f"unexpected {sorted(actual_units - declared_unit_set)}"
        )

    expected = set(range(expected_replicates))
    complete: dict[AnalysisUnit, tuple[ParameterSummary, ...]] = {}
    for unit in sorted(grouped):
        records = grouped[unit]
        replicate_ids = [record.replicate for record in records]
        duplicates = sorted(
            replicate
            for replicate in set(replicate_ids)
            if replicate_ids.count(replicate) > 1
        )
        if duplicates:
            raise QualificationStatisticsError(
                f"{unit} has duplicate replicates {duplicates}"
            )
        actual = set(replicate_ids)
        if actual != expected:
            raise QualificationStatisticsError(
                f"{unit} replicate set mismatch: missing {sorted(expected - actual)}, "
                f"unexpected {sorted(actual - expected)}"
            )
        complete[unit] = tuple(sorted(records, key=lambda record: record.replicate))
    return complete


def _validate_expected_units(
    expected_units: Sequence[AnalysisUnit],
) -> tuple[AnalysisUnit, ...]:
    if isinstance(expected_units, str | bytes) or not isinstance(
        expected_units, Sequence
    ):
        raise QualificationStatisticsError("expected_units must be a sequence")
    if not expected_units:
        raise QualificationStatisticsError("expected_units must not be empty")
    normalized: list[AnalysisUnit] = []
    for index, unit in enumerate(expected_units):
        if (
            isinstance(unit, str | bytes)
            or not isinstance(unit, Sequence)
            or len(unit) != 2
        ):
            raise QualificationStatisticsError(
                f"expected_units[{index}] must be a scenario/parameter pair"
            )
        scenario_id, parameter_id = unit
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            raise QualificationStatisticsError(
                f"expected_units[{index}].scenario_id must be a non-empty string"
            )
        if not isinstance(parameter_id, str) or not parameter_id.strip():
            raise QualificationStatisticsError(
                f"expected_units[{index}].parameter_id must be a non-empty string"
            )
        normalized.append((scenario_id, parameter_id))
    duplicates = sorted(unit for unit in set(normalized) if normalized.count(unit) > 1)
    if duplicates:
        raise QualificationStatisticsError(
            f"expected_units contains duplicates {duplicates}"
        )
    return tuple(sorted(normalized))


def clopper_pearson_interval(
    successes: int, trials: int, *, alpha: float
) -> ConfidenceInterval:
    """Return the exact two-sided Clopper-Pearson binomial interval."""
    if not _is_int(trials) or trials <= 0:
        raise QualificationStatisticsError("trials must be a positive integer")
    if not _is_int(successes) or not 0 <= successes <= trials:
        raise QualificationStatisticsError("successes must lie in [0, trials]")
    alpha = _finite_number(alpha, "alpha")
    _validate_probability(alpha, "alpha")
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    )
    return ConfidenceInterval(lower=lower, upper=upper)


def _coverage_power(
    trials: int, *, nominal: float, alternative: float, alpha: float
) -> CoveragePower:
    counts = np.arange(trials + 1)
    # Inverting the equal-tailed exact binomial test is equivalent to asking
    # whether the matching Clopper-Pearson interval contains ``nominal``.
    accepted = counts[
        (binom.cdf(counts, trials, nominal) >= alpha / 2)
        & (binom.sf(counts - 1, trials, nominal) >= alpha / 2)
    ]
    if accepted.size == 0:
        raise QualificationStatisticsError(
            "coverage gate has no accepted count at the requested design"
        )
    acceptance_min = int(accepted[0])
    acceptance_max = int(accepted[-1])
    lower_rejection = (
        0.0
        if acceptance_min == 0
        else float(binom.cdf(acceptance_min - 1, trials, alternative))
    )
    upper_rejection = (
        0.0
        if acceptance_max == trials
        else float(binom.sf(acceptance_max, trials, alternative))
    )
    return CoveragePower(
        nominal=nominal,
        alternative=alternative,
        acceptance_min=acceptance_min,
        acceptance_max=acceptance_max,
        power=lower_rejection + upper_rejection,
    )


def minimum_coverage_replicates(
    *,
    family_comparisons: int,
    nominals: Sequence[float] = (0.90, 0.95),
    undercoverage_delta: float = 0.10,
    familywise_alpha: float = 0.01,
    target_power: float = 0.90,
    max_replicates: int = 10_000,
) -> CoveragePowerDesign:
    """Find the prospective exact-binomial replicate count for undercoverage.

    A count is accepted exactly when its multiplicity-adjusted Clopper-Pearson
    interval contains the nominal coverage. Power is the binomial probability of
    falling outside that acceptance range under ``nominal - undercoverage_delta``.
    """
    if not _is_int(family_comparisons) or family_comparisons <= 0:
        raise QualificationStatisticsError("family_comparisons must be positive")
    if not _is_int(max_replicates) or max_replicates <= 0:
        raise QualificationStatisticsError("max_replicates must be positive")
    if isinstance(nominals, str | bytes) or not isinstance(nominals, Sequence):
        raise QualificationStatisticsError("nominals must be a sequence")
    checked_nominals = tuple(_finite_number(value, "nominal") for value in nominals)
    if not checked_nominals:
        raise QualificationStatisticsError("nominals must not be empty")
    for nominal in checked_nominals:
        _validate_probability(nominal, "nominal")
    undercoverage_delta = _finite_number(undercoverage_delta, "undercoverage_delta")
    if undercoverage_delta <= 0 or any(
        nominal - undercoverage_delta <= 0 for nominal in checked_nominals
    ):
        raise QualificationStatisticsError(
            "undercoverage_delta must be positive and below every nominal"
        )
    familywise_alpha = _finite_number(familywise_alpha, "familywise_alpha")
    _validate_probability(familywise_alpha, "familywise_alpha")
    target_power = _finite_number(target_power, "target_power")
    _validate_probability(target_power, "target_power", allow_one=True)
    adjusted_alpha = familywise_alpha / family_comparisons

    for trials in range(1, max_replicates + 1):
        targets = tuple(
            _coverage_power(
                trials,
                nominal=nominal,
                alternative=nominal - undercoverage_delta,
                alpha=adjusted_alpha,
            )
            for nominal in checked_nominals
        )
        if all(target.power >= target_power for target in targets):
            return CoveragePowerDesign(
                replicates=trials,
                family_comparisons=family_comparisons,
                familywise_alpha=familywise_alpha,
                alpha_per_comparison=adjusted_alpha,
                target_power=target_power,
                targets=targets,
            )
    raise QualificationStatisticsError(
        f"target power is not reached by {max_replicates} replicates"
    )


def evaluate_coverage_family(
    summaries: Sequence[ParameterSummary],
    *,
    family: Family,
    expected_replicates: int,
    expected_units: Sequence[AnalysisUnit],
    familywise_alpha: float = 0.01,
) -> tuple[CoverageCheck, ...]:
    """Evaluate exact coverage over one predeclared multiplicity family."""
    familywise_alpha = _finite_number(familywise_alpha, "familywise_alpha")
    _validate_probability(familywise_alpha, "familywise_alpha")
    groups = _complete_family_groups(
        summaries,
        family=family,
        expected_replicates=expected_replicates,
        expected_units=expected_units,
    )
    comparisons = len(groups) * len(COVERAGE_LEVELS)
    adjusted_alpha = familywise_alpha / comparisons
    checks = []
    for (scenario_id, parameter_id), records in groups.items():
        indicators = [coverage_indicators(record) for record in records]
        for nominal, _, _ in COVERAGE_LEVELS:
            successes = sum(item[nominal] for item in indicators)
            interval = clopper_pearson_interval(
                successes, expected_replicates, alpha=adjusted_alpha
            )
            checks.append(
                CoverageCheck(
                    family=family,
                    scenario_id=scenario_id,
                    parameter_id=parameter_id,
                    nominal=nominal,
                    successes=successes,
                    replicates=expected_replicates,
                    family_comparisons=comparisons,
                    alpha_per_comparison=adjusted_alpha,
                    interval=interval,
                    passed=interval.contains(nominal),
                )
            )
    return tuple(checks)


def evaluate_sbc_rank_family(
    summaries: Sequence[ParameterSummary],
    *,
    family: Family,
    expected_replicates: int,
    expected_units: Sequence[AnalysisUnit],
    familywise_alpha: float = 0.01,
) -> tuple[SbcRankCheck, ...]:
    """Evaluate every predeclared rank ECDF with a simultaneous DKW envelope."""
    familywise_alpha = _finite_number(familywise_alpha, "familywise_alpha")
    _validate_probability(familywise_alpha, "familywise_alpha")
    groups = _complete_family_groups(
        summaries,
        family=family,
        expected_replicates=expected_replicates,
        expected_units=expected_units,
    )
    curve_count = len(groups)
    epsilon = math.sqrt(
        math.log(2 * curve_count / familywise_alpha) / (2 * expected_replicates)
    )
    checks = []
    for (scenario_id, parameter_id), records in groups.items():
        if any(not record.has_sbc_rank for record in records):
            raise QualificationStatisticsError(
                f"{(scenario_id, parameter_id)} lacks SBC rank primitives"
            )
        draw_counts = {record.rank_draw_count for record in records}
        if len(draw_counts) != 1:
            raise QualificationStatisticsError(
                f"{(scenario_id, parameter_id)} changes rank_draw_count"
            )
        draw_count = draw_counts.pop()
        assert draw_count is not None
        ranks = [record.rank for record in records]
        assert all(rank is not None for rank in ranks)
        max_deviation = max(
            abs(
                sum(rank <= boundary for rank in ranks if rank is not None)
                / expected_replicates
                - (boundary + 1) / (draw_count + 1)
            )
            for boundary in range(draw_count + 1)
        )
        checks.append(
            SbcRankCheck(
                family=family,
                scenario_id=scenario_id,
                parameter_id=parameter_id,
                replicates=expected_replicates,
                rank_draw_count=draw_count,
                family_curves=curve_count,
                epsilon=epsilon,
                max_abs_deviation=max_deviation,
                passed=max_deviation <= epsilon,
            )
        )
    return tuple(checks)


def exact_sign_test_pvalue(values: Sequence[float]) -> float:
    """Return the exact two-sided sign-test p-value, excluding exact zeros."""
    finite_values = [_finite_number(value, "sign-test value") for value in values]
    positive = sum(value > 0 for value in finite_values)
    negative = sum(value < 0 for value in finite_values)
    nonzero = positive + negative
    if nonzero == 0:
        return 1.0
    return float(binomtest(positive, nonzero, p=0.5, alternative="two-sided").pvalue)


KeyT = TypeVar("KeyT", bound=Hashable)


def holm_rejections(
    pvalues: Mapping[KeyT, float], *, familywise_alpha: float = 0.01
) -> dict[KeyT, bool]:
    """Apply Holm's step-down familywise correction to named p-values."""
    familywise_alpha = _finite_number(familywise_alpha, "familywise_alpha")
    _validate_probability(familywise_alpha, "familywise_alpha")
    checked = {}
    for key, value in pvalues.items():
        probability = _finite_number(value, f"pvalue[{key!r}]")
        if not 0 <= probability <= 1:
            raise QualificationStatisticsError("p-values must lie in [0, 1]")
        checked[key] = probability
    if not checked:
        raise QualificationStatisticsError("Holm family must not be empty")

    ordered = sorted(checked.items(), key=lambda item: (item[1], repr(item[0])))
    rejected = dict.fromkeys(checked, False)
    for index, (key, probability) in enumerate(ordered):
        threshold = familywise_alpha / (len(ordered) - index)
        if probability > threshold:
            break
        rejected[key] = True
    return rejected


def evaluate_bias_family(
    summaries: Sequence[ParameterSummary],
    *,
    family: Family,
    expected_replicates: int,
    expected_units: Sequence[AnalysisUnit],
    bias_limit: float = 0.5,
    familywise_alpha: float = 0.01,
) -> tuple[BiasCheck, ...]:
    """Evaluate fixed-truth bias using a magnitude-only release gate.

    The exact sign-test p-values and Holm decisions are retained as descriptive
    diagnostics. They do not participate in the release decision: with the frozen
    five-replicate design, the smallest attainable two-sided sign-test p-value is
    0.0625 and therefore cannot reject at familywise alpha 0.01.
    """
    bias_limit = _finite_number(bias_limit, "bias_limit")
    if bias_limit < 0:
        raise QualificationStatisticsError("bias_limit must be non-negative")
    groups = _complete_family_groups(
        summaries,
        family=family,
        expected_replicates=expected_replicates,
        expected_units=expected_units,
    )
    errors = {
        unit: tuple(
            (record.posterior_mean - record.truth) / record.posterior_sd
            for record in records
        )
        for unit, records in groups.items()
    }
    pvalues = {unit: exact_sign_test_pvalue(values) for unit, values in errors.items()}
    rejected = holm_rejections(pvalues, familywise_alpha=familywise_alpha)

    checks = []
    for (scenario_id, parameter_id), values in errors.items():
        mean_error = math.fsum(values) / expected_replicates
        abs_mean = abs(mean_error)
        magnitude_passed = abs_mean <= bias_limit
        checks.append(
            BiasCheck(
                family=family,
                scenario_id=scenario_id,
                parameter_id=parameter_id,
                replicates=expected_replicates,
                mean_standardized_error=mean_error,
                abs_mean_standardized_error=abs_mean,
                median_standardized_error=float(median(values)),
                standardized_rmse=math.sqrt(
                    math.fsum(value**2 for value in values) / expected_replicates
                ),
                sign_test_pvalue=pvalues[(scenario_id, parameter_id)],
                holm_rejected=rejected[(scenario_id, parameter_id)],
                magnitude_passed=magnitude_passed,
            )
        )
    return tuple(checks)


def paired_backend_mean_check(
    left: ParameterSummary,
    right: ParameterSummary,
    *,
    limit: float = 3.0,
) -> BackendMeanCheck:
    """Compare backend posterior means using their independent MCSEs."""
    if not isinstance(left, ParameterSummary) or not isinstance(
        right, ParameterSummary
    ):
        raise QualificationStatisticsError("backend checks require validated summaries")
    if (
        left.family != right.family
        or left.parameter_id != right.parameter_id
        or left.replicate != right.replicate
        or left.truth != right.truth
    ):
        raise QualificationStatisticsError(
            "backend summaries must match family, parameter, replicate, and truth"
        )
    limit = _finite_number(limit, "limit")
    if limit < 0:
        raise QualificationStatisticsError("limit must be non-negative")
    mcse_z = abs(left.posterior_mean - right.posterior_mean) / math.hypot(
        left.posterior_mcse, right.posterior_mcse
    )
    return BackendMeanCheck(
        family=left.family,
        parameter_id=left.parameter_id,
        replicate=left.replicate,
        mcse_z=mcse_z,
        limit=limit,
        passed=mcse_z <= limit,
    )
