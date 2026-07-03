from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from autocleanml import (
    AutoCleanML,
    DataProfiler,
    DataQualityEvaluator,
    RepairPolicy,
    SparkMLClassificationEvaluator,
    SyntheticDataGenerator,
    SyntheticIssueConfig,
)


NUMERIC_FEATURES = ["feature_num1", "feature_num2"]
CATEGORICAL_FEATURES = ["category", "join_key"]
KEY_COLUMNS = ["join_key"]
LABEL_COL = "label"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a full synthetic data-quality experiment: dirty ML baseline, "
            "AutoCleanML repair, manual repair baseline, ML re-training, and OPEX."
        )
    )
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing-rate", type=float, default=0.08)
    parser.add_argument("--duplicate-rate", type=float, default=0.03)
    parser.add_argument("--outlier-rate", type=float, default=0.05)
    parser.add_argument("--skew-rate", type=float, default=0.50)
    parser.add_argument("--label-noise-rate", type=float, default=0.03)
    parser.add_argument("--missing-label-rate", type=float, default=0.15)
    parser.add_argument("--validation-folds", type=int, default=3)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Optional directory where JSON experiment artifacts should be written.",
    )
    args = parser.parse_args()

    spark = build_spark()
    try:
        config = SyntheticIssueConfig(
            row_count=args.rows,
            seed=args.seed,
            missing_rate=args.missing_rate,
            duplicate_rate=args.duplicate_rate,
            outlier_rate=args.outlier_rate,
            skew_rate=args.skew_rate,
            schema_drift=True,
            label_noise_rate=args.label_noise_rate,
            missing_label_rate=args.missing_label_rate,
        )
        synthetic = SyntheticDataGenerator(spark, config).generate_classification_dataset()

        auto_policy = RepairPolicy(
            drop_added_columns=True,
            label_imputation="model",
            label_confidence_threshold=args.confidence_threshold,
            skew_strategy="repartition",
            skew_target_partitions=4,
        )
        autoclean_result = AutoCleanML(repair_policy=auto_policy).run(
            synthetic.dirty_df,
            key_columns=KEY_COLUMNS,
            reference_schema=synthetic.reference_schema,
            label_col=LABEL_COL,
        )

        validation_result = run_validation_only_baseline(
            dirty_df=synthetic.dirty_df,
            reference_schema=synthetic.reference_schema,
        )

        manual_result = run_manual_baseline(
            dirty_df=synthetic.dirty_df,
            reference_schema=synthetic.reference_schema,
        )

        ml_evaluator = SparkMLClassificationEvaluator(
            validation_folds=args.validation_folds,
        )
        # Validation-only: cleaned_df == dirty_df (no repair applied)
        validation_ml = ml_evaluator.evaluate_logistic_regression(
            raw_df=synthetic.dirty_df,
            cleaned_df=synthetic.dirty_df,
            label_col=LABEL_COL,
            numeric_cols=NUMERIC_FEATURES,
            categorical_cols=CATEGORICAL_FEATURES,
        )
        autoclean_ml = ml_evaluator.evaluate_logistic_regression(
            raw_df=synthetic.dirty_df,
            cleaned_df=autoclean_result.cleaned_df,
            label_col=LABEL_COL,
            numeric_cols=NUMERIC_FEATURES,
            categorical_cols=CATEGORICAL_FEATURES,
        )
        manual_ml = ml_evaluator.evaluate_logistic_regression(
            raw_df=synthetic.dirty_df,
            cleaned_df=manual_result["cleaned_df"],
            label_col=LABEL_COL,
            numeric_cols=NUMERIC_FEATURES,
            categorical_cols=CATEGORICAL_FEATURES,
        )

        report = build_report(
            metadata=synthetic.metadata,
            autoclean_result=autoclean_result,
            validation_result=validation_result,
            manual_result=manual_result,
            autoclean_ml=autoclean_ml,
            validation_ml=validation_ml,
            manual_ml=manual_ml,
        )
        print_summary(report)

        if args.log_dir:
            run_dir = log_run(
                output_dir=Path(args.log_dir),
                run_name="synthetic_full_quality",
                artifacts={
                    "metadata": {
                        "dataset": "synthetic_classification",
                        "experiment": "full_quality_issues",
                        "synthetic_metadata": synthetic.metadata,
                        "validation_folds": args.validation_folds,
                        "confidence_threshold": args.confidence_threshold,
                    },
                    "autoclean_policy": auto_policy,
                    "autoclean_evaluation": autoclean_result.evaluation,
                    "autoclean_opex": autoclean_result.opex_metrics,
                    "autoclean_repair_actions": autoclean_result.repair_actions,
                    "autoclean_ml": autoclean_ml,
                    "validation_profile": validation_result["profile"],
                    "validation_ml": validation_ml,
                    "manual_evaluation": manual_result["evaluation"],
                    "manual_opex": manual_result["opex_metrics"],
                    "manual_repair_actions": manual_result["actions"],
                    "manual_ml": manual_ml,
                    "comparison_report": report,
                },
            )
            print(f"\nExperiment artifacts written to: {run_dir}")
    finally:
        spark.stop()


def run_validation_only_baseline(
    dirty_df: DataFrame,
    reference_schema: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Profiling only — no repair applied.

    Represents tools like Deequ or Great Expectations that detect issues but
    leave repair to the engineer. ML trains on the same dirty data, so ML
    performance is identical to the raw baseline. This isolates the value of
    the repair step specifically.
    """
    profiler = DataProfiler()
    start = perf_counter()
    profile = profiler.profile(
        dirty_df,
        key_columns=KEY_COLUMNS,
        reference_schema=reference_schema,
        label_col=LABEL_COL,
    )
    profile_time = perf_counter() - start

    issues_detected = {
        "missingness_cols_flagged": sum(
            1 for v in profile.get("missingness", {}).values()
            if v.get("severity", "none") != "none"
        ),
        "duplicates_detected": profile.get("duplicates", {}).get("duplicate_count", 0),
        "outlier_cols_flagged": sum(
            1 for v in profile.get("outliers", {}).values()
            if v.get("severity", "none") != "none"
        ),
        "schema_drift_detected": profile.get("schema_drift", {}).get("drift_detected", False),
        "skew_cols_flagged": sum(
            1 for v in profile.get("skew", {}).values()
            if v.get("severity", "none") not in {"none", "low", "unknown"}
        ),
    }

    return {
        "profile": profile,
        "issues_detected": issues_detected,
        "profile_time_seconds": round(profile_time, 6),
        "note": "Detection only — no repair. ML trains on dirty data.",
    }


def run_manual_baseline(
    dirty_df: DataFrame,
    reference_schema: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    profiler = DataProfiler()
    evaluator = DataQualityEvaluator()

    total_start = perf_counter()

    raw_start = perf_counter()
    raw_profile = profiler.profile(
        dirty_df,
        key_columns=KEY_COLUMNS,
        reference_schema=reference_schema,
        label_col=LABEL_COL,
    )
    raw_profile_time = perf_counter() - raw_start

    repair_start = perf_counter()
    cleaned_df, actions = manual_clean_synthetic_dataframe(
        dirty_df,
        reference_schema,
    )
    repair_time = perf_counter() - repair_start

    cleaned_start = perf_counter()
    cleaned_profile = profiler.profile(
        cleaned_df,
        key_columns=KEY_COLUMNS,
        reference_schema=reference_schema,
        label_col=LABEL_COL,
    )
    cleaned_profile_time = perf_counter() - cleaned_start

    evaluation_start = perf_counter()
    evaluation = evaluator.evaluate(
        raw_profile=raw_profile,
        cleaned_profile=cleaned_profile,
        repair_actions=actions,
    ).metrics
    evaluation_time = perf_counter() - evaluation_start
    total_time = perf_counter() - total_start

    return {
        "cleaned_df": cleaned_df,
        "raw_profile": raw_profile,
        "cleaned_profile": cleaned_profile,
        "evaluation": evaluation,
        "actions": actions,
        "opex_metrics": build_manual_opex(
            raw_profile=raw_profile,
            cleaned_profile=cleaned_profile,
            evaluation=evaluation,
            actions=actions,
            raw_profile_time=raw_profile_time,
            repair_time=repair_time,
            cleaned_profile_time=cleaned_profile_time,
            evaluation_time=evaluation_time,
            total_time=total_time,
        ),
    }


def manual_clean_synthetic_dataframe(
    df: DataFrame,
    reference_schema: dict[str, dict[str, Any]],
) -> tuple[DataFrame, list[dict[str, Any]]]:
    """
    Dataset-specific manual Spark cleaning for the synthetic classification data.
    """

    cleaned_df = df
    actions: list[dict[str, Any]] = []
    row_count = cleaned_df.count()

    if "feature_num1" in cleaned_df.columns:
        cleaned_df = cleaned_df.withColumn("feature_num1", F.col("feature_num1").cast("double"))
        actions.append({
            "issue": "schema_drift",
            "column": "feature_num1",
            "strategy": "manual_cast_to_double",
        })

    if "new_source_col" in cleaned_df.columns:
        cleaned_df = cleaned_df.drop("new_source_col")
        actions.append({
            "issue": "schema_drift",
            "column": "new_source_col",
            "strategy": "manual_drop_added_column",
        })

    feature_num1_missing = cleaned_df.filter(F.col("feature_num1").isNull()).count()
    if feature_num1_missing > 0:
        median = cleaned_df.approxQuantile("feature_num1", [0.5], 0.01)[0]
        cleaned_df = cleaned_df.fillna({"feature_num1": median})
        actions.append({
            "issue": "missingness",
            "column": "feature_num1",
            "strategy": "manual_median_fill",
            "fill_value": median,
            "missing_count": feature_num1_missing,
        })

    category_missing = cleaned_df.filter(F.col("category").isNull()).count()
    if category_missing > 0:
        cleaned_df = cleaned_df.fillna({"category": "unknown"})
        actions.append({
            "issue": "missingness",
            "column": "category",
            "strategy": "manual_constant_fill_unknown",
            "missing_count": category_missing,
        })

    outlier_stats = manual_iqr_outlier_stats(cleaned_df, "feature_num2", row_count)
    lower_bound = outlier_stats["lower_bound"]
    upper_bound = outlier_stats["upper_bound"]
    if lower_bound is not None and upper_bound is not None:
        cleaned_df = cleaned_df.withColumn(
            "feature_num2",
            F.when(F.col("feature_num2") < lower_bound, F.lit(lower_bound))
            .when(F.col("feature_num2") > upper_bound, F.lit(upper_bound))
            .otherwise(F.col("feature_num2")),
        )
        actions.append({
            "issue": "outliers",
            "column": "feature_num2",
            "strategy": "manual_iqr_cap",
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": outlier_stats["outlier_count"],
        })

    duplicate_count = row_count - cleaned_df.distinct().count()
    if duplicate_count > 0:
        cleaned_df = cleaned_df.dropDuplicates()
        actions.append({
            "issue": "duplicates",
            "strategy": "manual_drop_exact_duplicates",
            "duplicate_count": duplicate_count,
        })

    skew_stats = manual_skew_stats(cleaned_df, "join_key")
    if skew_stats["severity"] == "high":
        cleaned_df = cleaned_df.repartition(4, F.col("join_key"))
        actions.append({
            "issue": "skew",
            "column": "join_key",
            "strategy": "manual_repartition",
            "target_partitions": 4,
            "skew_ratio": skew_stats["skew_ratio"],
        })

    return cleaned_df, actions


def manual_iqr_outlier_stats(
    df: DataFrame,
    column: str,
    row_count: int,
) -> dict[str, Any]:
    quantiles = df.approxQuantile(column, [0.25, 0.75], 0.01)
    if len(quantiles) < 2:
        return {"lower_bound": None, "upper_bound": None, "outlier_count": 0}

    q1, q3 = quantiles
    iqr = q3 - q1
    if iqr == 0:
        return {"lower_bound": None, "upper_bound": None, "outlier_count": 0}

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = df.filter(
        (F.col(column) < lower_bound) | (F.col(column) > upper_bound)
    ).count()
    return {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": outlier_count,
        "outlier_ratio": round(outlier_count / row_count, 4) if row_count else 0.0,
    }


def manual_skew_stats(df: DataFrame, column: str) -> dict[str, Any]:
    key_counts = df.groupBy(column).count()
    max_count = key_counts.agg(F.max("count")).collect()[0][0]
    median_values = key_counts.approxQuantile("count", [0.5], 0.01)
    median_count = median_values[0] if median_values else None
    skew_ratio = max_count / median_count if median_count else None
    if skew_ratio is None:
        severity = "unknown"
    elif skew_ratio <= 5:
        severity = "low"
    elif skew_ratio <= 10:
        severity = "medium"
    else:
        severity = "high"
    return {
        "max_count": max_count,
        "median_count": median_count,
        "skew_ratio": round(skew_ratio, 4) if skew_ratio is not None else None,
        "severity": severity,
    }


def build_manual_opex(
    *,
    raw_profile: dict[str, Any],
    cleaned_profile: dict[str, Any],
    evaluation: dict[str, Any],
    actions: list[dict[str, Any]],
    raw_profile_time: float,
    repair_time: float,
    cleaned_profile_time: float,
    evaluation_time: float,
    total_time: float,
) -> dict[str, Any]:
    source_lines = (
        inspect.getsourcelines(manual_clean_synthetic_dataframe)[0]
        + inspect.getsourcelines(manual_iqr_outlier_stats)[0]
        + inspect.getsourcelines(manual_skew_stats)[0]
    )
    code_lines = len([
        line for line in source_lines
        if line.strip() and not line.strip().startswith("#")
    ])
    input_rows = int(raw_profile.get("row_count", 0))
    output_rows = int(cleaned_profile.get("row_count", 0))

    return {
        "manual_cleaning_steps": len(actions),
        "manual_actions_by_issue": evaluation.get("repair_actions_by_issue", {}),
        "dataset_specific_cleaning_code_lines": code_lines,
        "raw_profile_time_seconds": round(raw_profile_time, 6),
        "manual_repair_time_seconds": round(repair_time, 6),
        "cleaned_profile_time_seconds": round(cleaned_profile_time, 6),
        "evaluation_time_seconds": round(evaluation_time, 6),
        "total_cycle_time_seconds": round(total_time, 6),
        "seconds_per_1000_input_rows": (
            round(total_time / input_rows * 1000, 6)
            if input_rows
            else None
        ),
        "input_row_count": input_rows,
        "output_row_count": output_rows,
        "row_count_delta": output_rows - input_rows,
    }


def compute_detection_accuracy(
    issues: dict[str, Any],
    raw_profile: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    numeric_accuracies: list[float] = []

    def _pct_accuracy(detected: float | None, injected: float) -> float | None:
        if detected is None or injected == 0:
            return None
        return round(max(0.0, 1 - abs(detected - injected) / injected) * 100, 1)

    if "missingness" in issues:
        injected_rate = issues["missingness"]["rate"]
        by_col: dict[str, Any] = {}
        for col in issues["missingness"]["columns"]:
            detected = raw_profile.get("missingness", {}).get(col, {}).get("missing_ratio")
            acc = _pct_accuracy(detected, injected_rate)
            by_col[col] = {"injected": injected_rate, "detected": detected, "accuracy": acc}
            if acc is not None:
                numeric_accuracies.append(acc)
        scored = [a["accuracy"] for a in by_col.values() if a["accuracy"] is not None]
        mean_acc = round(sum(scored) / len(scored), 1) if scored else None
        result["missingness"] = {"by_column": by_col, "mean_accuracy": mean_acc}

    if "missing_labels" in issues:
        injected_rate = issues["missing_labels"]["rate"]
        col = issues["missing_labels"]["column"]
        detected = raw_profile.get("missingness", {}).get(col, {}).get("missing_ratio")
        acc = _pct_accuracy(detected, injected_rate)
        result["missing_labels"] = {"column": col, "injected": injected_rate, "detected": detected, "accuracy": acc}
        if acc is not None:
            numeric_accuracies.append(acc)

    if "outliers" in issues:
        injected_rate = issues["outliers"]["rate"]
        col = issues["outliers"]["column"]
        detected = raw_profile.get("outliers", {}).get(col, {}).get("outlier_ratio")
        acc = _pct_accuracy(detected, injected_rate)
        result["outliers"] = {"column": col, "injected": injected_rate, "detected": detected, "accuracy": acc}
        if acc is not None:
            numeric_accuracies.append(acc)

    if "duplicates" in issues:
        injected_rate = issues["duplicates"]["rate"]
        detected = raw_profile.get("duplicates", {}).get("duplicate_ratio")
        acc = _pct_accuracy(detected, injected_rate)
        result["duplicates"] = {"injected": injected_rate, "detected": detected, "accuracy": acc}
        if acc is not None:
            numeric_accuracies.append(acc)

    if "key_skew" in issues:
        col = issues["key_skew"]["column"]
        result["key_skew"] = {
            "column": col,
            "injected_rate": issues["key_skew"]["rate"],
            "detected_ratio": raw_profile.get("skew", {}).get(col, {}).get("skew_ratio"),
            "detected_severity": raw_profile.get("skew", {}).get(col, {}).get("severity"),
        }

    if "schema_drift" in issues:
        detected = raw_profile.get("schema_drift", {}).get("drift_detected", False)
        result["schema_drift"] = {"injected": True, "detected": detected, "accurate": detected}

    if "label_noise" in issues:
        injected_rate = issues["label_noise"]["rate"]
        noise_report = raw_profile.get("label_noise", {}).get("confidence_noise", {})
        if noise_report.get("evaluated"):
            detected = noise_report.get("suspected_noise_ratio")
            acc = _pct_accuracy(detected, injected_rate)
            result["label_noise"] = {"injected": injected_rate, "suspected": detected, "accuracy": acc}
            if acc is not None:
                numeric_accuracies.append(acc)
        else:
            result["label_noise"] = {
                "injected": injected_rate,
                "evaluated": False,
                "message": noise_report.get("message"),
            }

    result["mean_accuracy"] = (
        round(sum(numeric_accuracies) / len(numeric_accuracies), 1)
        if numeric_accuracies else None
    )
    return result


def build_report(
    *,
    metadata: dict[str, Any],
    autoclean_result: Any,
    validation_result: dict[str, Any],
    manual_result: dict[str, Any],
    autoclean_ml: Any,
    validation_ml: Any,
    manual_ml: Any,
) -> dict[str, Any]:
    return {
        "experiment": "synthetic_full_quality_issues",
        "synthetic_issues": metadata.get("issues", {}),
        "detection_accuracy": compute_detection_accuracy(
            metadata.get("issues", {}),
            autoclean_result.raw_profile,
        ),
        "data_quality": {
            "autocleanml": summarize_quality(autoclean_result.evaluation),
            "manual_baseline": summarize_quality(manual_result["evaluation"]),
            "validation_only": {
                "repair_applied": False,
                "issues_detected": validation_result["issues_detected"],
                "profile_time_seconds": validation_result["profile_time_seconds"],
            },
        },
        "ml_metrics": {
            "raw_baseline": {
                "accuracy": autoclean_ml.raw_metrics.get("accuracy"),
                "f1": autoclean_ml.raw_metrics.get("f1"),
                "auc": autoclean_ml.raw_metrics.get("auc"),
                "ml_row_count": autoclean_ml.raw_metrics.get("ml_row_count"),
            },
            "validation_only": {
                "accuracy": validation_ml.cleaned_metrics.get("accuracy"),
                "f1": validation_ml.cleaned_metrics.get("f1"),
                "auc": validation_ml.cleaned_metrics.get("auc"),
                "ml_row_count": validation_ml.cleaned_metrics.get("ml_row_count"),
                "delta_vs_raw": validation_ml.delta,
                "note": "No repair applied — ML result expected to match raw baseline.",
            },
            "autocleanml": {
                "accuracy": autoclean_ml.cleaned_metrics.get("accuracy"),
                "f1": autoclean_ml.cleaned_metrics.get("f1"),
                "auc": autoclean_ml.cleaned_metrics.get("auc"),
                "ml_row_count": autoclean_ml.cleaned_metrics.get("ml_row_count"),
                "delta_vs_raw": autoclean_ml.delta,
            },
            "manual_baseline": {
                "accuracy": manual_ml.cleaned_metrics.get("accuracy"),
                "f1": manual_ml.cleaned_metrics.get("f1"),
                "auc": manual_ml.cleaned_metrics.get("auc"),
                "ml_row_count": manual_ml.cleaned_metrics.get("ml_row_count"),
                "delta_vs_raw": manual_ml.delta,
            },
        },
        "opex": {
            "autocleanml": autoclean_result.opex_metrics,
            "manual_baseline": manual_result["opex_metrics"],
            "validation_only_profile_seconds": validation_result["profile_time_seconds"],
            "dataset_specific_code_lines_reduced": (
                manual_result["opex_metrics"]["dataset_specific_cleaning_code_lines"] - 1
            ),
        },
        "interpretation": (
            "Four conditions: raw (dirty data, no intervention), validation-only "
            "(profiling only, no repair — ML identical to raw), manual baseline "
            "(dataset-specific Spark cleaning), AutoCleanML (automated rule-driven "
            "repair). Validation-only isolates the repair step: if ML improves only "
            "with repair conditions, detection alone is insufficient."
        ),
    }


def summarize_quality(evaluation: dict[str, Any]) -> dict[str, Any]:
    skew_by_col = evaluation.get("skew", {}).get("by_column", {})
    skew_cols_reduced = sum(
        1 for v in skew_by_col.values()
        if v.get("reduction") is not None and v["reduction"] > 0
    )
    return {
        "missingness_reduction": evaluation.get("missingness", {}).get("reduction"),
        "duplicate_reduction": evaluation.get("duplicates", {}).get("reduction"),
        "outlier_reduction": evaluation.get("outliers", {}).get("reduction"),
        "skew_columns_reduced": skew_cols_reduced if skew_by_col else None,
        "schema_raw_issue_count": evaluation.get("schema_drift", {}).get(
            "raw_issue_count"
        ),
        "schema_cleaned_issue_count": evaluation.get("schema_drift", {}).get(
            "cleaned_issue_count"
        ),
        "repair_actions_by_issue": evaluation.get("repair_actions_by_issue"),
    }


def print_summary(report: dict[str, Any]) -> None:
    ml = report["ml_metrics"]
    quality = report["data_quality"]
    opex = report["opex"]

    print("\n=== Synthetic Full Data-Quality Experiment ===")
    print("Injected issues:")
    for issue, detail in report["synthetic_issues"].items():
        print(f"  {issue}: {detail}")

    print("\n--- ML Performance: Four-Condition Comparison ---")
    print(f"{'Condition':<22} {'Accuracy':>10} {'F1':>8} {'AUC':>8} {'Rows':>8}")
    print("-" * 60)
    for label, key in [
        ("Raw (dirty)",       "raw_baseline"),
        ("Validation-only",   "validation_only"),
        ("AutoCleanML",       "autocleanml"),
        ("Manual baseline",   "manual_baseline"),
    ]:
        m = ml[key]
        print(
            f"{label:<22} {str(m.get('accuracy')):>10} "
            f"{str(m.get('f1')):>8} {str(m.get('auc')):>8} "
            f"{str(m.get('ml_row_count')):>8}"
        )

    print("\n--- Delta vs Raw ---")
    for label, key in [
        ("Validation-only",   "validation_only"),
        ("AutoCleanML",       "autocleanml"),
        ("Manual baseline",   "manual_baseline"),
    ]:
        d = ml[key].get("delta_vs_raw", {})
        print(
            f"  {label:<20} accuracy={d.get('accuracy'):+.4f}  "
            f"f1={d.get('f1'):+.4f}  auc={str(d.get('auc')):>8}  "
            f"rows={d.get('ml_row_count'):+d}"
        )

    print("\n--- Data-Quality Improvements (AutoCleanML) ---")
    aq = quality["autocleanml"]
    print(f"  missingness reduction: {aq['missingness_reduction']}")
    print(f"  duplicate reduction:   {aq['duplicate_reduction']}")
    print(f"  outlier reduction:     {aq['outlier_reduction']}")
    print(
        f"  schema issues: "
        f"{aq['schema_raw_issue_count']} -> {aq['schema_cleaned_issue_count']}"
    )
    vd = quality["validation_only"]["issues_detected"]
    print(f"\n--- Validation-only detection (no repair) ---")
    for k, v in vd.items():
        print(f"  {k}: {v}")

    print("\n--- OPEX ---")
    print(f"  AutoCleanML cycle seconds:    {opex['autocleanml']['total_time_seconds']}")
    print(f"  Manual baseline cycle seconds: {opex['manual_baseline']['total_cycle_time_seconds']}")
    print(f"  Validation-only profile secs:  {opex['validation_only_profile_seconds']}")
    print(f"  Dataset-specific code lines reduced: {opex['dataset_specific_code_lines_reduced']}")


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("autocleanml-synthetic-full-quality")
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def log_run(
    output_dir: Path,
    run_name: str,
    artifacts: dict[str, Any],
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_dir / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "run_name": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_files": {},
    }
    for name, payload in artifacts.items():
        filename = f"{name}.json"
        write_json(run_dir / filename, payload)
        manifest["artifact_files"][name] = filename

    write_json(run_dir / "manifest.json", manifest)
    return run_dir


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


if __name__ == "__main__":
    main()
