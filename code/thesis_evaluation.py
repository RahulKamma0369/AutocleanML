from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from .ml_evaluation import MLEvaluationResult
from .pipeline import AutoCleanMLResult
from .repair import RepairPolicy


@dataclass
class ThesisEvaluationReport:
    ml_metrics: dict[str, Any]
    data_quality_metrics: dict[str, Any]
    process_efficiency_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ml_metrics": self.ml_metrics,
            "data_quality_metrics": self.data_quality_metrics,
            "process_efficiency_metrics": self.process_efficiency_metrics,
        }


class ThesisEvaluationReportBuilder:
    """
    Builds thesis-facing metrics grouped by the proposal evaluation categories.
    """

    def build(
        self,
        result: AutoCleanMLResult,
        ml_metrics: MLEvaluationResult | None = None,
        manual_cleaning_steps_baseline: int | None = None,
        dataset_specific_cleaning_code_lines_baseline: int | None = None,
        manual_cycle_time_seconds_baseline: float | None = None,
    ) -> ThesisEvaluationReport:
        report = ThesisEvaluationReport(
            ml_metrics=self._ml_metrics(ml_metrics),
            data_quality_metrics=self._data_quality_metrics(result),
            process_efficiency_metrics=self._process_efficiency_metrics(
                result=result,
                manual_cleaning_steps_baseline=manual_cleaning_steps_baseline,
                dataset_specific_cleaning_code_lines_baseline=(
                    dataset_specific_cleaning_code_lines_baseline
                ),
                manual_cycle_time_seconds_baseline=manual_cycle_time_seconds_baseline,
            ),
        )
        return report

    def _ml_metrics(
        self,
        ml_metrics: MLEvaluationResult | None,
    ) -> dict[str, Any]:
        if ml_metrics is None:
            return {
                "captured": False,
                "message": "ML evaluation was not run for this experiment.",
            }

        raw = ml_metrics.raw_metrics
        cleaned = ml_metrics.cleaned_metrics
        delta = ml_metrics.delta

        if "accuracy" in raw:
            return {
                "captured": True,
                "task_type": "classification",
                "accuracy": self._raw_cleaned_delta(raw, cleaned, delta, "accuracy"),
                "f1": self._raw_cleaned_delta(raw, cleaned, delta, "f1"),
                "auc": self._raw_cleaned_delta(raw, cleaned, delta, "auc"),
                "weighted_precision": self._raw_cleaned_delta(
                    raw,
                    cleaned,
                    delta,
                    "weighted_precision",
                ),
                "weighted_recall": self._raw_cleaned_delta(
                    raw,
                    cleaned,
                    delta,
                    "weighted_recall",
                ),
                "stability_across_validation_folds": {
                    "raw": raw.get("fold_stability"),
                    "cleaned": cleaned.get("fold_stability"),
                },
                "model_sensitivity_to_noisy_vs_clean_data": delta,
            }

        if "rmse" in raw:
            return {
                "captured": True,
                "task_type": "regression",
                "rmse": self._raw_cleaned_delta(raw, cleaned, delta, "rmse"),
                "mae": self._raw_cleaned_delta(raw, cleaned, delta, "mae"),
                "r2": self._raw_cleaned_delta(raw, cleaned, delta, "r2"),
                "stability_across_validation_folds": {
                    "raw": raw.get("fold_stability"),
                    "cleaned": cleaned.get("fold_stability"),
                },
                "model_sensitivity_to_noisy_vs_clean_data": delta,
            }

        return {
            "captured": True,
            "message": "ML metrics were supplied but task type was not recognized.",
            "raw": raw,
            "cleaned": cleaned,
            "delta": delta,
        }

    def _data_quality_metrics(
        self,
        result: AutoCleanMLResult,
    ) -> dict[str, Any]:
        evaluation = result.evaluation
        return {
            "change_in_missingness": evaluation.get("missingness"),
            "reduction_in_duplicates": evaluation.get("duplicates"),
            "improvement_in_skew_balance": self._skew_balance_change(result),
            "outlier_reduction": evaluation.get("outliers"),
            "schema_consistency_measures": evaluation.get("schema_drift"),
        }

    def _process_efficiency_metrics(
        self,
        result: AutoCleanMLResult,
        manual_cleaning_steps_baseline: int | None,
        dataset_specific_cleaning_code_lines_baseline: int | None,
        manual_cycle_time_seconds_baseline: float | None,
    ) -> dict[str, Any]:
        opex = result.opex_metrics
        automated_steps = int(opex.get("automated_repair_actions", 0))
        cycle_seconds = opex.get("total_time_seconds")
        time_saved = (
            round(manual_cycle_time_seconds_baseline - cycle_seconds, 6)
            if manual_cycle_time_seconds_baseline is not None
            and cycle_seconds is not None
            else None
        )

        return {
            "reduction_in_manual_cleaning_steps": {
                "baseline_steps": manual_cleaning_steps_baseline,
                "automated_repair_actions": automated_steps,
                "estimated_steps_avoided": opex.get("manual_steps_avoided_estimate"),
                "remaining_manual_steps_estimate": (
                    max(manual_cleaning_steps_baseline - automated_steps, 0)
                    if manual_cleaning_steps_baseline is not None
                    else None
                ),
            },
            "reduction_in_dataset_specific_cleaning_code": {
                "baseline_lines": dataset_specific_cleaning_code_lines_baseline,
                "autocleanml_policy_lines_estimate": self._policy_line_estimate(result),
                "estimated_lines_reduced": (
                    dataset_specific_cleaning_code_lines_baseline
                    - self._policy_line_estimate(result)
                    if dataset_specific_cleaning_code_lines_baseline is not None
                    else None
                ),
                "message": (
                    "Provide a manual baseline line count to compute this metric."
                    if dataset_specific_cleaning_code_lines_baseline is None
                    else None
                ),
            },
            "time_saved_per_cleaning_cycle": {
                "manual_baseline_seconds": manual_cycle_time_seconds_baseline,
                "autocleanml_cycle_seconds": cycle_seconds,
                "estimated_seconds_saved": time_saved,
                "message": (
                    "Provide a manual baseline runtime to compute time saved."
                    if manual_cycle_time_seconds_baseline is None
                    else None
                ),
            },
            "runtime_cost": opex,
        }

    def _raw_cleaned_delta(
        self,
        raw: dict[str, Any],
        cleaned: dict[str, Any],
        delta: dict[str, Any],
        metric_name: str,
    ) -> dict[str, Any]:
        return {
            "raw": raw.get(metric_name),
            "cleaned": cleaned.get(metric_name),
            "delta": delta.get(metric_name),
        }

    def _skew_balance_change(self, result: AutoCleanMLResult) -> dict[str, Any]:
        raw_skew = result.raw_profile.get("skew", {})
        cleaned_skew = result.cleaned_profile.get("skew", {})
        columns = sorted(set(raw_skew) | set(cleaned_skew))

        by_column = {}
        raw_total = 0.0
        cleaned_total = 0.0
        comparable_columns = 0

        for col in columns:
            raw_ratio = raw_skew.get(col, {}).get("skew_ratio")
            cleaned_ratio = cleaned_skew.get(col, {}).get("skew_ratio")
            if raw_ratio is not None and cleaned_ratio is not None:
                raw_total += raw_ratio
                cleaned_total += cleaned_ratio
                comparable_columns += 1
            by_column[col] = {
                "raw_skew_ratio": raw_ratio,
                "cleaned_skew_ratio": cleaned_ratio,
                "reduction": (
                    round(raw_ratio - cleaned_ratio, 4)
                    if raw_ratio is not None and cleaned_ratio is not None
                    else None
                ),
                "raw_severity": raw_skew.get(col, {}).get("severity"),
                "cleaned_severity": cleaned_skew.get(col, {}).get("severity"),
            }

        return {
            "by_column": by_column,
            "average_raw_skew_ratio": (
                round(raw_total / comparable_columns, 4)
                if comparable_columns
                else None
            ),
            "average_cleaned_skew_ratio": (
                round(cleaned_total / comparable_columns, 4)
                if comparable_columns
                else None
            ),
            "average_reduction": (
                round((raw_total - cleaned_total) / comparable_columns, 4)
                if comparable_columns
                else None
            ),
            "note": (
                "Value-skew ratios may stay unchanged when repair uses Spark "
                "repartitioning, because repartitioning changes physical "
                "execution layout rather than key frequency distribution."
            ),
        }

    def _policy_line_estimate(self, result: AutoCleanMLResult) -> int:
        policy = result.repair_policy
        if policy is None:
            return 5
        defaults = RepairPolicy()
        non_default_count = sum(
            1 for f in fields(policy)
            if getattr(policy, f.name) != getattr(defaults, f.name)
        )
        # 5 lines of fixed overhead: RepairPolicy(...), AutoCleanML(...).run(),
        # key_columns, reference_schema, label_col arguments
        return non_default_count + 5
