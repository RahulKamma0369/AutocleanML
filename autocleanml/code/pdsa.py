from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

from pyspark.sql import DataFrame

from .evaluation import DataQualityEvaluator
from .opex import build_opex_metrics
from .profiler import DataProfiler
from .repair import RepairEngine, RepairPolicy


@dataclass
class PDSAConfig:
    """
    Quality thresholds that govern when the PDSA loop considers a cycle
    complete. Later repair passes are restricted to iterative_issue_types so
    deterministic single-pass repairs are not repeated unnecessarily.
    """
    max_missingness_ratio: float = 0.01
    max_outlier_ratio: float = 0.01
    max_duplicate_ratio: float = 0.0
    max_skew_ratio: float | None = None
    max_iterations: int = 3
    iterative_issue_types: tuple[str, ...] = ("outliers", "skew")


@dataclass
class PDSAIterationRecord:
    iteration: int
    repair_actions: list[dict[str, Any]]
    residual_missingness_ratio: float | None
    residual_outlier_ratio: float | None
    residual_duplicate_ratio: float | None
    residual_skew_ratio: float | None
    thresholds_met: bool
    repair_issue_types: tuple[str, ...]
    policy_adjustments: list[str]


@dataclass
class PDSAResult:
    """
    Mirrors AutoCleanMLResult but includes the full iteration history so
    each PDSA cycle (Plan-Do-Study-Act) can be reported in the thesis.
    """
    raw_profile: dict[str, Any]
    cleaned_profile: dict[str, Any]
    evaluation: dict[str, Any]
    opex_metrics: dict[str, Any]
    repair_actions: list[dict[str, Any]]
    cleaned_df: DataFrame
    pdsa_iterations: list[PDSAIterationRecord]
    converged: bool
    total_iterations: int


class PDSALoop:
    """
    Wraps AutoCleanML in a Plan-Do-Study-Act feedback loop.

    Each iteration:
      Plan   — profile the current (dirty or partially cleaned) DataFrame
      Do     — apply repairs using the current policy
      Study  — re-profile and evaluate; compute residual issue rates
      Act    — if quality thresholds are not met, log residual issues and
               feed the cleaned DataFrame back into a selective next cycle.
               Later cycles only rerun configured iterative issue types such
               as outliers/skew, avoiding expensive repeated missingness and
               deduplication work by default.

    The loop terminates when either all thresholds are met (converged) or
    max_iterations is reached.
    """

    def __init__(
        self,
        repair_policy: RepairPolicy | None = None,
        pdsa_config: PDSAConfig | None = None,
        profiler: DataProfiler | None = None,
        evaluator: DataQualityEvaluator | None = None,
    ):
        self.initial_policy = repair_policy or RepairPolicy()
        self.config = pdsa_config or PDSAConfig()
        if self.config.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        unsupported_iterative_types = (
            set(self.config.iterative_issue_types) - set(_all_issue_types())
        )
        if unsupported_iterative_types:
            raise ValueError(
                "Unsupported iterative_issue_types. Expected values from "
                f"{_all_issue_types()}. Got: {sorted(unsupported_iterative_types)}"
            )
        self.profiler = profiler or DataProfiler()
        self.evaluator = evaluator or DataQualityEvaluator()

    def run(
        self,
        df: DataFrame,
        key_columns: list[str] | None = None,
        reference_schema: dict | None = None,
        label_col: str | None = None,
    ) -> PDSAResult:
        total_start = perf_counter()

        raw_profile = self.profiler.profile(
            df=df,
            key_columns=key_columns,
            reference_schema=reference_schema,
            label_col=label_col,
        )

        policy = self.initial_policy
        current_df = df
        iteration_records: list[PDSAIterationRecord] = []
        all_repair_actions: list[dict[str, Any]] = []
        converged = False
        repair_issue_types = _all_issue_types()

        for i in range(1, self.config.max_iterations + 1):
            current_repair_issue_types = repair_issue_types
            current_profile = self.profiler.profile(
                df=current_df,
                key_columns=key_columns,
                reference_schema=reference_schema,
                label_col=label_col,
            ) if i > 1 else raw_profile
            profile_to_repair = _profile_for_issue_types(
                current_profile,
                current_repair_issue_types,
            )
            iteration_policy = _policy_for_issue_types(
                policy,
                current_repair_issue_types,
            )

            repair_result = RepairEngine(iteration_policy).repair(
                df=current_df,
                profile_report=profile_to_repair,
                reference_schema=reference_schema,
                key_columns=key_columns,
                label_col=label_col,
            )
            all_repair_actions.extend(repair_result.actions)

            cleaned_profile = self.profiler.profile(
                df=repair_result.cleaned_df,
                key_columns=key_columns,
                reference_schema=reference_schema,
                label_col=label_col,
            )

            residuals = _compute_residuals(cleaned_profile)
            thresholds_met = _check_thresholds(residuals, self.config)

            adjustments: list[str] = []
            next_repair_issue_types: tuple[str, ...] | None = None
            stop_after_record = False
            if not thresholds_met and i < self.config.max_iterations:
                failed_issue_types = _failed_issue_types(residuals, self.config)
                actionable_issue_types = tuple(
                    issue
                    for issue in self.config.iterative_issue_types
                    if issue in failed_issue_types
                )
                adjustments = _act(
                    residuals=residuals,
                    config=self.config,
                    failed_issue_types=failed_issue_types,
                    actionable_issue_types=actionable_issue_types,
                )
                if not actionable_issue_types:
                    stop_after_record = True
                else:
                    next_repair_issue_types = actionable_issue_types

            iteration_records.append(PDSAIterationRecord(
                iteration=i,
                repair_actions=repair_result.actions,
                residual_missingness_ratio=residuals["missingness_ratio"],
                residual_outlier_ratio=residuals["outlier_ratio"],
                residual_duplicate_ratio=residuals["duplicate_ratio"],
                residual_skew_ratio=residuals["skew_ratio"],
                thresholds_met=thresholds_met,
                repair_issue_types=current_repair_issue_types,
                policy_adjustments=adjustments,
            ))

            current_df = repair_result.cleaned_df

            if thresholds_met:
                converged = True
                break
            if stop_after_record:
                break
            if next_repair_issue_types is not None:
                repair_issue_types = next_repair_issue_types

        final_evaluation = self.evaluator.evaluate(
            raw_profile=raw_profile,
            cleaned_profile=cleaned_profile,
            repair_actions=all_repair_actions,
        )
        total_time = perf_counter() - total_start
        opex_metrics = build_opex_metrics(
            raw_profile=raw_profile,
            cleaned_profile=cleaned_profile,
            evaluation=final_evaluation.metrics,
            raw_profile_time_seconds=0.0,
            repair_time_seconds=0.0,
            cleaned_profile_time_seconds=0.0,
            evaluation_time_seconds=0.0,
            total_time_seconds=total_time,
        )

        return PDSAResult(
            raw_profile=raw_profile,
            cleaned_profile=cleaned_profile,
            evaluation=final_evaluation.metrics,
            opex_metrics=opex_metrics,
            repair_actions=all_repair_actions,
            cleaned_df=current_df,
            pdsa_iterations=iteration_records,
            converged=converged,
            total_iterations=len(iteration_records),
        )


def _compute_residuals(profile: dict[str, Any]) -> dict[str, float | None]:
    row_count = profile.get("row_count", 0) or 0

    missingness = profile.get("missingness", {})
    total_missing = sum(
        v.get("missing_count", 0) for v in missingness.values()
        if isinstance(v, dict)
    )
    total_cells = row_count * max(len(missingness), 1)
    missingness_ratio = total_missing / total_cells if total_cells > 0 else 0.0

    outliers = profile.get("outliers", {})
    total_outliers = sum(
        v.get("outlier_count", 0) for v in outliers.values()
        if isinstance(v, dict)
    )
    outlier_ratio = total_outliers / row_count if row_count > 0 else 0.0

    duplicates = profile.get("duplicates", {})
    dup_count = duplicates.get("exact", {}).get(
        "duplicate_count", duplicates.get("duplicate_count", 0)
    )
    duplicate_ratio = dup_count / row_count if row_count > 0 else 0.0

    skew = profile.get("skew", {})
    skew_ratios = [
        stats.get("skew_ratio")
        for stats in skew.values()
        if isinstance(stats, dict) and stats.get("skew_ratio") is not None
    ]
    skew_ratio = max(skew_ratios) if skew_ratios else None

    return {
        "missingness_ratio": round(missingness_ratio, 6),
        "outlier_ratio": round(outlier_ratio, 6),
        "duplicate_ratio": round(duplicate_ratio, 6),
        "skew_ratio": round(skew_ratio, 6) if skew_ratio is not None else None,
    }


def _check_thresholds(
    residuals: dict[str, float | None],
    config: PDSAConfig,
) -> bool:
    return (
        (residuals["missingness_ratio"] or 0.0) <= config.max_missingness_ratio
        and (residuals["outlier_ratio"] or 0.0) <= config.max_outlier_ratio
        and (residuals["duplicate_ratio"] or 0.0) <= config.max_duplicate_ratio
        and (
            config.max_skew_ratio is None
            or residuals["skew_ratio"] is None
            or residuals["skew_ratio"] <= config.max_skew_ratio
        )
    )


def _act(
    residuals: dict[str, float | None],
    config: PDSAConfig,
    failed_issue_types: tuple[str, ...],
    actionable_issue_types: tuple[str, ...],
) -> list[str]:
    """
    Log which thresholds were violated and which issue types will be retried.
    Deterministic single-pass repairs such as missingness and exact duplicates
    are skipped on later cycles unless explicitly configured as iterative.
    """
    adjustments: list[str] = []

    for issue in failed_issue_types:
        metric_name, threshold = _issue_threshold(issue, config)
        residual_value = residuals.get(metric_name)
        if issue in actionable_issue_types:
            adjustments.append(
                f"re-attempting {issue} repair: residual {metric_name} "
                f"{residual_value:.4f} > threshold {threshold}"
            )
        else:
            adjustments.append(
                f"not re-running {issue}: residual {metric_name} "
                f"{residual_value:.4f} > threshold {threshold}; "
                "issue type is not configured for iterative repair"
            )

    return adjustments


def _all_issue_types() -> tuple[str, ...]:
    return (
        "schema_drift",
        "missingness",
        "outliers",
        "label_noise",
        "duplicates",
        "skew",
    )


def _profile_for_issue_types(
    profile: dict[str, Any],
    issue_types: tuple[str, ...],
) -> dict[str, Any]:
    selected = set(issue_types)
    return {
        **profile,
        "schema_drift": (
            profile.get("schema_drift", {}) if "schema_drift" in selected else {}
        ),
        "missingness": (
            profile.get("missingness", {}) if "missingness" in selected else {}
        ),
        "outliers": profile.get("outliers", {}) if "outliers" in selected else {},
        "label_noise": (
            profile.get("label_noise", {}) if "label_noise" in selected else {}
        ),
        "duplicates": (
            profile.get("duplicates", {}) if "duplicates" in selected else {}
        ),
        "skew": profile.get("skew", {}) if "skew" in selected else {},
    }


def _policy_for_issue_types(
    policy: RepairPolicy,
    issue_types: tuple[str, ...],
) -> RepairPolicy:
    selected = set(issue_types)
    return replace(
        policy,
        align_schema=policy.align_schema and "schema_drift" in selected,
        missingness_threshold=(
            policy.missingness_threshold if "missingness" in selected else 1.0
        ),
        outlier_strategy=(
            policy.outlier_strategy if "outliers" in selected else "none"
        ),
        label_imputation=(
            policy.label_imputation if "label_noise" in selected else "none"
        ),
        drop_duplicates=policy.drop_duplicates and "duplicates" in selected,
        skew_strategy=policy.skew_strategy if "skew" in selected else "none",
    )


def _failed_issue_types(
    residuals: dict[str, float | None],
    config: PDSAConfig,
) -> tuple[str, ...]:
    failed = []
    if (residuals["missingness_ratio"] or 0.0) > config.max_missingness_ratio:
        failed.append("missingness")
    if (residuals["outlier_ratio"] or 0.0) > config.max_outlier_ratio:
        failed.append("outliers")
    if (residuals["duplicate_ratio"] or 0.0) > config.max_duplicate_ratio:
        failed.append("duplicates")
    if (
        config.max_skew_ratio is not None
        and residuals["skew_ratio"] is not None
        and residuals["skew_ratio"] > config.max_skew_ratio
    ):
        failed.append("skew")
    return tuple(failed)


def _issue_threshold(issue: str, config: PDSAConfig) -> tuple[str, float]:
    if issue == "missingness":
        return "missingness_ratio", config.max_missingness_ratio
    if issue == "outliers":
        return "outlier_ratio", config.max_outlier_ratio
    if issue == "duplicates":
        return "duplicate_ratio", config.max_duplicate_ratio
    if issue == "skew":
        return "skew_ratio", config.max_skew_ratio or 0.0
    raise ValueError(f"Unsupported PDSA issue type: {issue}")
