"""Tests for the pure truncated-hierarchy statistical contracts."""

from __future__ import annotations

import copy
import math

import pytest

from scripts.truncated_hierarchy_statistics import (
    ParameterSummary,
    QualificationStatisticsError,
    clopper_pearson_interval,
    coverage_indicators,
    derive_sbc_rank_tie_index,
    evaluate_bias_family,
    evaluate_coverage_family,
    evaluate_sbc_rank_family,
    exact_sign_test_pvalue,
    holm_rejections,
    minimum_coverage_replicates,
    paired_backend_mean_check,
    validate_parameter_summary,
)

SBC_TIE_SEED = 1_282_001
DEFAULT_EXPECTED_UNITS = (("scenario-a", "mu-group"),)


def _summary_record(
    replicate: int = 0,
    *,
    family: str = "candidate",
    scenario_id: str = "scenario-a",
    parameter_id: str = "mu-group",
    standardized_error: float = 0.0,
    coverage: str = "both",
    rank: int | None = None,
    rank_draw_count: int = 100,
) -> dict[str, object]:
    truth = 0.0
    if coverage == "both":
        quantiles = (-2.0, -1.5, 0.0, 1.5, 2.0)
    elif coverage == "95-only":
        quantiles = (-2.0, 0.5, 1.0, 1.5, 2.0)
    elif coverage == "neither":
        quantiles = (0.5, 0.75, 1.0, 1.5, 2.0)
    else:
        raise AssertionError(f"unknown test coverage case: {coverage}")
    record: dict[str, object] = {
        "family": family,
        "scenario_id": scenario_id,
        "parameter_id": parameter_id,
        "replicate": replicate,
        "truth": truth,
        "posterior_mean": truth + standardized_error,
        "posterior_sd": 1.0,
        "posterior_mcse": 0.1,
        "q025": quantiles[0],
        "q05": quantiles[1],
        "q50": quantiles[2],
        "q95": quantiles[3],
        "q975": quantiles[4],
    }
    if rank is not None:
        record.update(
            {
                "rank_less": rank,
                "rank_equal": 0,
                "rank_tie_index": 0,
                "rank": rank,
                "rank_draw_count": rank_draw_count,
            }
        )
    return record


def _validated_records(
    count: int,
    *,
    family: str = "candidate",
    scenario_id: str = "scenario-a",
    parameter_id: str = "mu-group",
    errors: list[float] | None = None,
    coverage: str = "both",
) -> list[ParameterSummary]:
    errors = [0.0] * count if errors is None else errors
    assert len(errors) == count
    return [
        validate_parameter_summary(
            _summary_record(
                replicate,
                family=family,
                scenario_id=scenario_id,
                parameter_id=parameter_id,
                standardized_error=errors[replicate],
                coverage=coverage,
            )
        )
        for replicate in range(count)
    ]


def _validated_sbc_record(
    replicate: int,
    *,
    scenario_id: str = "scenario-a",
    parameter_id: str = "mu-group",
    rank: int,
    rank_draw_count: int = 100,
) -> ParameterSummary:
    return validate_parameter_summary(
        _summary_record(
            replicate,
            scenario_id=scenario_id,
            parameter_id=parameter_id,
            rank=rank,
            rank_draw_count=rank_draw_count,
        ),
        require_sbc=True,
        expected_sbc_tie_seed=SBC_TIE_SEED,
    )


def test_summary_validation_and_coverage_are_natural_scale_and_inclusive() -> None:
    """Validate complete summaries and derive both nested coverage indicators."""
    record = _summary_record(coverage="95-only")
    record["truth"] = record["q05"]
    summary = validate_parameter_summary(record)

    assert isinstance(summary, ParameterSummary)
    assert coverage_indicators(summary) == {0.90: True, 0.95: True}

    only_95 = validate_parameter_summary(_summary_record(coverage="95-only"))
    assert coverage_indicators(only_95) == {0.90: False, 0.95: True}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda item: item.update(posterior_sd=0.0), "posterior_sd"),
        (lambda item: item.update(posterior_mcse=math.inf), "finite"),
        (lambda item: item.update(q05=0.5, q50=0.25), "quantiles"),
        (lambda item: item.update(replicate=True), "replicate"),
        (lambda item: item.update(family="diagnostic"), "family"),
        (lambda item: item.update(extra=1), "unknown"),
        (lambda item: item.update(rank=1), "all present or all absent"),
    ],
)
def test_summary_validation_rejects_invalid_values(mutation, message) -> None:
    """Reject malformed values before they can enter scientific decisions."""
    record = _summary_record()
    mutation(record)

    with pytest.raises(QualificationStatisticsError, match=message):
        validate_parameter_summary(record)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {
                "rank_less": 90,
                "rank_equal": 11,
                "rank_tie_index": 0,
                "rank": 90,
            },
            "cannot exceed",
        ),
        (
            {
                "rank_less": 20,
                "rank_equal": 2,
                "rank_tie_index": 3,
                "rank": 23,
            },
            "tie_index",
        ),
        (
            {
                "rank_less": 20,
                "rank_equal": 2,
                "rank_tie_index": 1,
                "rank": 22,
            },
            "must equal",
        ),
    ],
)
def test_summary_validation_rejects_inconsistent_rank_primitives(
    updates, message
) -> None:
    """Do not trust a reported rank that disagrees with its auditable primitives."""
    record = _summary_record(rank=20)
    record.update(updates)

    with pytest.raises(QualificationStatisticsError, match=message):
        validate_parameter_summary(
            record,
            require_sbc=True,
            expected_sbc_tie_seed=SBC_TIE_SEED,
        )


def test_sbc_tie_index_is_deterministically_derived_from_trusted_seed() -> None:
    """Reject a self-consistent rank when its reported tie draw was forged."""
    record = _summary_record(rank=0)
    record.update(rank_less=20, rank_equal=7, rank_draw_count=100)
    expected_tie_index = derive_sbc_rank_tie_index(
        tie_seed=SBC_TIE_SEED,
        family="candidate",
        scenario_id="scenario-a",
        parameter_id="mu-group",
        replicate=0,
        rank_less=20,
        rank_equal=7,
        rank_draw_count=100,
    )
    record["rank_tie_index"] = expected_tie_index
    record["rank"] = 20 + expected_tie_index

    validated = validate_parameter_summary(
        record,
        require_sbc=True,
        expected_sbc_tie_seed=SBC_TIE_SEED,
    )

    assert expected_tie_index == 1
    assert validated.rank_tie_index == expected_tie_index
    assert validated.rank == 20 + expected_tie_index
    assert (
        derive_sbc_rank_tie_index(
            tie_seed=SBC_TIE_SEED,
            family="candidate",
            scenario_id="scenario-a",
            parameter_id="mu-group",
            replicate=0,
            rank_less=20,
            rank_equal=7,
            rank_draw_count=100,
        )
        == expected_tie_index
    )

    forged = copy.deepcopy(record)
    forged_tie_index = (expected_tie_index + 1) % 8
    forged["rank_tie_index"] = forged_tie_index
    forged["rank"] = 20 + forged_tie_index
    with pytest.raises(QualificationStatisticsError, match="deterministic tie index"):
        validate_parameter_summary(
            forged,
            require_sbc=True,
            expected_sbc_tie_seed=SBC_TIE_SEED,
        )
    with pytest.raises(QualificationStatisticsError, match="deterministic tie index"):
        validate_parameter_summary(
            record,
            require_sbc=True,
            expected_sbc_tie_seed=SBC_TIE_SEED + 1,
        )


def test_sbc_rank_payload_requires_a_trusted_tie_seed() -> None:
    """Never accept a caller-selected randomized rank without frozen entropy."""
    with pytest.raises(QualificationStatisticsError, match="expected_sbc_tie_seed"):
        validate_parameter_summary(_summary_record(rank=20), require_sbc=True)


def test_clopper_pearson_exact_boundaries_match_frozen_family_rule() -> None:
    """Pin the exact acceptance ranges for R=100 and six comparisons."""
    adjusted_alpha = 0.01 / 6
    for successes in (80, 98):
        assert clopper_pearson_interval(successes, 100, alpha=adjusted_alpha).contains(
            0.90
        )
    for successes in (79, 99):
        assert not clopper_pearson_interval(
            successes, 100, alpha=adjusted_alpha
        ).contains(0.90)
    for successes in (87, 100):
        assert clopper_pearson_interval(successes, 100, alpha=adjusted_alpha).contains(
            0.95
        )
    assert not clopper_pearson_interval(86, 100, alpha=adjusted_alpha).contains(0.95)


def test_prospective_exact_binomial_design_is_digestible_and_minimal() -> None:
    """Freeze the 20-comparison, ten-point-undercoverage power calculation."""
    design = minimum_coverage_replicates(family_comparisons=20)

    assert design.replicates == 275
    assert design.alpha_per_comparison == pytest.approx(0.0005)
    assert [target.power for target in design.targets] == pytest.approx(
        [0.9019, 0.9874], abs=5e-5
    )
    assert all(target.power >= 0.90 for target in design.targets)
    with pytest.raises(QualificationStatisticsError, match="not reached by 274"):
        minimum_coverage_replicates(
            family_comparisons=20, max_replicates=design.replicates - 1
        )


def test_coverage_is_per_scenario_parameter_and_cannot_be_diluted() -> None:
    """A failed unit remains visible beside any number of perfect units."""
    records = _validated_records(10, scenario_id="bad", coverage="neither")
    for scenario_id in ("good-a", "good-b", "good-c"):
        records.extend(_validated_records(10, scenario_id=scenario_id))

    checks = evaluate_coverage_family(
        records,
        family="candidate",
        expected_replicates=10,
        expected_units=(
            ("bad", "mu-group"),
            ("good-a", "mu-group"),
            ("good-b", "mu-group"),
            ("good-c", "mu-group"),
        ),
    )
    bad = [check for check in checks if check.scenario_id == "bad"]
    good = [check for check in checks if check.scenario_id != "bad"]

    assert {check.nominal for check in bad} == {0.90, 0.95}
    assert not any(check.passed for check in bad)
    assert all(check.passed for check in good)
    assert not all(check.passed for check in checks)


def test_missing_or_duplicate_replicates_are_errors() -> None:
    """Never reduce the denominator after missing or duplicate evidence."""
    missing = _validated_records(5)[:-1]
    with pytest.raises(QualificationStatisticsError, match=r"missing \[4\]"):
        evaluate_coverage_family(
            missing,
            family="candidate",
            expected_replicates=5,
            expected_units=DEFAULT_EXPECTED_UNITS,
        )

    duplicate = _validated_records(5)
    duplicate[-1] = duplicate[0]
    with pytest.raises(QualificationStatisticsError, match="duplicate replicates"):
        evaluate_coverage_family(
            duplicate,
            family="candidate",
            expected_replicates=5,
            expected_units=DEFAULT_EXPECTED_UNITS,
        )


@pytest.mark.parametrize(
    ("records", "expected_units", "message"),
    [
        (
            _validated_records(5),
            (("scenario-a", "mu-group"), ("scenario-b", "mu-group")),
            r"missing \[\('scenario-b', 'mu-group'\)\]",
        ),
        (
            [
                *_validated_records(5),
                *_validated_records(5, scenario_id="undeclared"),
            ],
            DEFAULT_EXPECTED_UNITS,
            r"unexpected \[\('undeclared', 'mu-group'\)\]",
        ),
    ],
)
def test_coverage_rejects_missing_or_extra_analysis_units(
    records, expected_units, message
) -> None:
    """A whole omitted unit cannot shrink either evidence or multiplicity."""
    with pytest.raises(QualificationStatisticsError, match=message):
        evaluate_coverage_family(
            records,
            family="candidate",
            expected_replicates=5,
            expected_units=expected_units,
        )


def test_candidate_family_size_is_stable_when_controls_are_added() -> None:
    """Keep control multiplicity separate from the candidate family."""
    candidates = _validated_records(5)
    before = evaluate_coverage_family(
        candidates,
        family="candidate",
        expected_replicates=5,
        expected_units=DEFAULT_EXPECTED_UNITS,
    )
    controls = _validated_records(5, family="control", scenario_id="control-a")
    controls.extend(_validated_records(5, family="control", scenario_id="control-b"))
    after = evaluate_coverage_family(
        candidates,
        family="candidate",
        expected_replicates=5,
        expected_units=DEFAULT_EXPECTED_UNITS,
    )
    control_checks = evaluate_coverage_family(
        controls,
        family="control",
        expected_replicates=5,
        expected_units=(
            ("control-a", "mu-group"),
            ("control-b", "mu-group"),
        ),
    )

    assert before == after
    assert {check.family_comparisons for check in before} == {2}
    assert {check.family_comparisons for check in control_checks} == {4}
    with pytest.raises(QualificationStatisticsError, match="mixed families"):
        evaluate_coverage_family(
            [*candidates, *controls],
            family="candidate",
            expected_replicates=5,
            expected_units=DEFAULT_EXPECTED_UNITS,
        )


def test_uniform_sbc_ranks_pass_and_all_zero_ranks_fail() -> None:
    """Check the complete rank ECDF, not a pooled or mean-rank surrogate."""
    uniform = [
        _validated_sbc_record(replicate, rank=replicate) for replicate in range(101)
    ]
    all_zero = [_validated_sbc_record(replicate, rank=0) for replicate in range(101)]

    uniform_check = evaluate_sbc_rank_family(
        uniform,
        family="candidate",
        expected_replicates=101,
        expected_units=DEFAULT_EXPECTED_UNITS,
    )[0]
    zero_check = evaluate_sbc_rank_family(
        all_zero,
        family="candidate",
        expected_replicates=101,
        expected_units=DEFAULT_EXPECTED_UNITS,
    )[0]

    assert uniform_check.max_abs_deviation == pytest.approx(0.0)
    assert uniform_check.passed is True
    assert zero_check.max_abs_deviation > zero_check.epsilon
    assert zero_check.passed is False


@pytest.mark.parametrize(
    ("include_extra", "expected_units", "message"),
    [
        (
            False,
            (("scenario-a", "mu-group"), ("scenario-b", "mu-group")),
            r"missing \[\('scenario-b', 'mu-group'\)\]",
        ),
        (
            True,
            DEFAULT_EXPECTED_UNITS,
            r"unexpected \[\('undeclared', 'mu-group'\)\]",
        ),
    ],
)
def test_sbc_rejects_missing_or_extra_analysis_units(
    include_extra, expected_units, message
) -> None:
    """Freeze every rank curve before applying its Bonferroni envelope."""
    records = [
        _validated_sbc_record(replicate, rank=replicate) for replicate in range(5)
    ]
    if include_extra:
        records.extend(
            _validated_sbc_record(
                replicate,
                scenario_id="undeclared",
                rank=replicate,
            )
            for replicate in range(5)
        )

    with pytest.raises(QualificationStatisticsError, match=message):
        evaluate_sbc_rank_family(
            records,
            family="candidate",
            expected_replicates=5,
            expected_units=expected_units,
        )


def test_bias_gate_and_reproducible_no_go_have_distinct_thresholds() -> None:
    """Separate the 0.5 magnitude gate from the significant 1.0 no-go."""
    symmetric_errors = [-0.6, 0.6] * 10
    symmetric = evaluate_bias_family(
        _validated_records(20, errors=symmetric_errors),
        family="candidate",
        expected_replicates=20,
        expected_units=DEFAULT_EXPECTED_UNITS,
    )[0]
    systematic = evaluate_bias_family(
        _validated_records(20, scenario_id="systematic", errors=[0.6] * 20),
        family="candidate",
        expected_replicates=20,
        expected_units=(("systematic", "mu-group"),),
    )[0]
    immediate = evaluate_bias_family(
        _validated_records(20, scenario_id="immediate", errors=[1.0] * 20),
        family="candidate",
        expected_replicates=20,
        expected_units=(("immediate", "mu-group"),),
    )[0]

    assert symmetric.mean_standardized_error == pytest.approx(0.0)
    assert symmetric.magnitude_passed is True
    assert symmetric.holm_rejected is False
    assert symmetric.reproducible_bias is False
    assert systematic.abs_mean_standardized_error == pytest.approx(0.6)
    assert systematic.magnitude_passed is False
    assert systematic.holm_rejected is True
    assert systematic.reproducible_bias is False
    assert immediate.abs_mean_standardized_error == pytest.approx(1.0)
    assert immediate.holm_rejected is True
    assert immediate.reproducible_bias is True


@pytest.mark.parametrize(
    ("records", "expected_units", "message"),
    [
        (
            _validated_records(5),
            (("scenario-a", "mu-group"), ("scenario-b", "mu-group")),
            r"missing \[\('scenario-b', 'mu-group'\)\]",
        ),
        (
            [
                *_validated_records(5),
                *_validated_records(5, scenario_id="undeclared"),
            ],
            DEFAULT_EXPECTED_UNITS,
            r"unexpected \[\('undeclared', 'mu-group'\)\]",
        ),
    ],
)
def test_bias_rejects_missing_or_extra_analysis_units(
    records, expected_units, message
) -> None:
    """Keep the Holm family fixed if a whole unit is omitted or injected."""
    with pytest.raises(
        QualificationStatisticsError,
        match=message,
    ):
        evaluate_bias_family(
            records,
            family="candidate",
            expected_replicates=5,
            expected_units=expected_units,
        )


def test_expected_analysis_units_must_be_unique() -> None:
    """Do not let duplicate declarations distort a multiplicity family."""
    with pytest.raises(QualificationStatisticsError, match="contains duplicates"):
        evaluate_coverage_family(
            _validated_records(5),
            family="candidate",
            expected_replicates=5,
            expected_units=(*DEFAULT_EXPECTED_UNITS, *DEFAULT_EXPECTED_UNITS),
        )


def test_exact_sign_test_and_holm_have_explicit_step_down_semantics() -> None:
    """Pin zero handling and the stop-after-first-failure Holm rule."""
    assert exact_sign_test_pvalue([0.0, 0.0]) == 1.0
    assert exact_sign_test_pvalue([1.0] * 10) == pytest.approx(2 / 2**10)
    assert holm_rejections(
        {"first": 0.001, "second": 0.006, "third": 0.02},
        familywise_alpha=0.01,
    ) == {"first": True, "second": False, "third": False}


def test_backend_mean_check_uses_both_mcse_values_and_matching_identity() -> None:
    """Derive backend mean agreement while leaving rank-Rhat to raw-chain code."""
    left = validate_parameter_summary(_summary_record())
    right_record = copy.deepcopy(_summary_record())
    right_record["scenario_id"] = "scenario-a-numpyro"
    right_record["posterior_mean"] = 0.2
    right = validate_parameter_summary(right_record)

    check = paired_backend_mean_check(left, right)

    assert check.mcse_z == pytest.approx(math.sqrt(2))
    assert check.passed is True
    mismatched = validate_parameter_summary(
        _summary_record(family="control", scenario_id="control")
    )
    with pytest.raises(QualificationStatisticsError, match="must match family"):
        paired_backend_mean_check(left, mismatched)
