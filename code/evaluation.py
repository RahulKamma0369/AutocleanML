from dataclasses import dataclass
from typing import Any


@dataclass
class DataQualityEvaluation:
    metrics: dict[str, Any]


class DataQualityEvaluator:
    """
    Compares raw and cleaned profiling reports.

    This is the data-quality portion of the proposal's evaluation layer. ML
    performance evaluation can be added separately once experiment datasets and
    target columns are finalized.
    """

    def evaluate(
        self,
        raw_profile: dict[str, Any],
        cleaned_profile: dict[str, Any],
        repair_actions: list[dict[str, Any]] | None = None,
    ) -> DataQualityEvaluation:
        repair_actions = repair_actions or []

        metrics = {
            "row_count": self._row_count_change(raw_profile, cleaned_profile),
            "missingness": self._missingness_change(raw_profile, cleaned_profile),
            "duplicates": self._duplicate_change(raw_profile, cleaned_profile),
            "outliers": self._outlier_change(raw_profile, cleaned_profile),
            "skew": self._skew_change(raw_profile, cleaned_profile),
            "schema_drift": self._schema_drift_change(raw_profile, cleaned_profile),
            "repair_action_count": len(repair_actions),
            "repair_actions_by_issue": self._actions_by_issue(repair_actions),
        }

        return DataQualityEvaluation(metrics=metrics)

    def _row_count_change(
        self,
        raw_profile: dict[str, Any],
        cleaned_profile: dict[str, Any],
    ) -> dict[str, Any]:
        raw_count = raw_profile.get("row_count", 0)
        cleaned_count = cleaned_profile.get("row_count", 0)

        return {
            "raw": raw_count,
            "cleaned": cleaned_count,
            "change": cleaned_count - raw_count,
            "change_ratio": self._safe_ratio(cleaned_count - raw_count, raw_count),
        }

    def _missingness_change(
        self,
        raw_profile: dict[str, Any],
        cleaned_profile: dict[str, Any],
    ) -> dict[str, Any]:
        raw_missing = raw_profile.get("missingness", {})
        cleaned_missing = cleaned_profile.get("missingness", {})
        shared_columns = sorted(set(raw_missing) & set(cleaned_missing))
        added_columns = sorted(set(cleaned_missing) - set(raw_missing))
        removed_columns = sorted(set(raw_missing) - set(cleaned_missing))

        by_column = {}
        raw_total = 0
        cleaned_total = 0

        for col in shared_columns:
            raw_count = raw_missing.get(col, {}).get("missing_count", 0)
            cleaned_count = cleaned_missing.get(col, {}).get("missing_count", 0)
            raw_total += raw_count
            cleaned_total += cleaned_count
            by_column[col] = {
                "raw": raw_count,
                "cleaned": cleaned_count,
                "reduction": raw_count - cleaned_count,
                "reduction_ratio": self._safe_ratio(raw_count - cleaned_count, raw_count),
            }

        return {
            "raw_total": raw_total,
            "cleaned_total": cleaned_total,
            "reduction": raw_total - cleaned_total,
            "reduction_ratio": self._safe_ratio(raw_total - cleaned_total, raw_total),
            "by_column": by_column,
            "scope": "shared_columns",
            "added_columns": {
                col: cleaned_missing.get(col, {}).get("missing_count", 0)
                for col in added_columns
            },
            "removed_columns": {
                col: raw_missing.get(col, {}).get("missing_count", 0)
                for col in removed_columns
            },
        }

    def _duplicate_change(
        self,
        raw_profile: dict[str, Any],
        cleaned_profile: dict[str, Any],
    ) -> dict[str, Any]:
        raw_count = raw_profile.get("duplicates", {}).get("duplicate_count", 0)
        cleaned_count = cleaned_profile.get("duplicates", {}).get("duplicate_count", 0)

        return {
            "raw": raw_count,
            "cleaned": cleaned_count,
            "reduction": raw_count - cleaned_count,
            "reduction_ratio": self._safe_ratio(raw_count - cleaned_count, raw_count),
        }

    def _outlier_change(
        self,
        raw_profile: dict[str, Any],
        cleaned_profile: dict[str, Any],
    ) -> dict[str, Any]:
        raw_outliers = raw_profile.get("outliers", {})
        cleaned_outliers = cleaned_profile.get("outliers", {})
        columns = sorted(set(raw_outliers) | set(cleaned_outliers))

        by_column = {}
        raw_total = 0
        cleaned_total = 0

        for col in columns:
            raw_count = raw_outliers.get(col, {}).get("outlier_count", 0)
            cleaned_count = cleaned_outliers.get(col, {}).get("outlier_count", 0)
            raw_total += raw_count
            cleaned_total += cleaned_count
            by_column[col] = {
                "raw": raw_count,
                "cleaned": cleaned_count,
                "reduction": raw_count - cleaned_count,
                "reduction_ratio": self._safe_ratio(raw_count - cleaned_count, raw_count),
            }

        return {
            "raw_total": raw_total,
            "cleaned_total": cleaned_total,
            "reduction": raw_total - cleaned_total,
            "reduction_ratio": self._safe_ratio(raw_total - cleaned_total, raw_total),
            "by_column": by_column,
        }

    def _skew_change(
        self,
        raw_profile: dict[str, Any],
        cleaned_profile: dict[str, Any],
    ) -> dict[str, Any]:
        raw_skew = raw_profile.get("skew", {})
        cleaned_skew = cleaned_profile.get("skew", {})
        columns = sorted(set(raw_skew) | set(cleaned_skew))

        by_column = {}
        for col in columns:
            raw_ratio = raw_skew.get(col, {}).get("skew_ratio")
            cleaned_ratio = cleaned_skew.get(col, {}).get("skew_ratio")
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

        return {"by_column": by_column}

    def _schema_drift_change(
        self,
        raw_profile: dict[str, Any],
        cleaned_profile: dict[str, Any],
    ) -> dict[str, Any]:
        raw_drift = raw_profile.get("schema_drift", {})
        cleaned_drift = cleaned_profile.get("schema_drift", {})

        return {
            "raw_drift_detected": raw_drift.get("drift_detected", False),
            "cleaned_drift_detected": cleaned_drift.get("drift_detected", False),
            "raw_issue_count": self._schema_issue_count(raw_drift),
            "cleaned_issue_count": self._schema_issue_count(cleaned_drift),
        }

    def _actions_by_issue(
        self,
        repair_actions: list[dict[str, Any]],
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for action in repair_actions:
            issue = action.get("issue", "unknown")
            counts[issue] = counts.get(issue, 0) + 1
        return counts

    def _schema_issue_count(self, schema_drift: dict[str, Any]) -> int:
        return (
            len(schema_drift.get("added_columns", []))
            + len(schema_drift.get("removed_columns", []))
            + len(schema_drift.get("type_changes", []))
        )

    def _safe_ratio(self, numerator: int | float, denominator: int | float) -> float:
        if denominator == 0:
            return 0.0
        return round(numerator / denominator, 4)
