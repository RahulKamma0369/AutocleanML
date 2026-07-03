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
    SparkMLRegressionEvaluator,
    SyntheticDataGenerator,
    SyntheticIssueConfig,
)


NUMERIC_FEATURES = ["sqft_living", "sqft_lot", "bedrooms", "bathrooms", "age_years", "distance_to_center", "school_rating"]
CATEGORICAL_FEATURES = ["neighborhood", "house_type", "condition"]
KEY_COLUMNS = ["neighborhood", "house_type"]
LABEL_COL = "price"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run synthetic regression cleaning and OPEX experiment."
    )
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing-rate", type=float, default=0.08)
    parser.add_argument("--duplicate-rate", type=float, default=0.03)
    parser.add_argument("--outlier-rate", type=float, default=0.05)
    parser.add_argument("--skew-rate", type=float, default=0.50)
    parser.add_argument("--target-noise-rate", type=float, default=0.03)
    parser.add_argument("--validation-folds", type=int, default=3)
    parser.add_argument("--log-dir", default=None)
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
            label_noise_rate=args.target_noise_rate,
            missing_label_rate=0.0,
        )
        synthetic = SyntheticDataGenerator(spark, config).generate_house_price_dataset()

        policy = RepairPolicy(
            drop_added_columns=True,
            skew_strategy="repartition",
            skew_target_partitions=4,
        )
        autoclean_result = AutoCleanML(repair_policy=policy).run(
            synthetic.dirty_df,
            key_columns=KEY_COLUMNS,
            reference_schema=synthetic.reference_schema,
            label_col=LABEL_COL,
        )
        manual_result = run_manual_baseline(
            synthetic.dirty_df,
            synthetic.reference_schema,
        )
        validation_result = run_validation_only_baseline(
            synthetic.dirty_df,
            synthetic.reference_schema,
        )

        ml_evaluator = SparkMLRegressionEvaluator(validation_folds=args.validation_folds)
        autoclean_ml = ml_evaluator.evaluate_linear_regression(
            raw_df=synthetic.dirty_df,
            cleaned_df=autoclean_result.cleaned_df,
            label_col=LABEL_COL,
            numeric_cols=NUMERIC_FEATURES,
            categorical_cols=CATEGORICAL_FEATURES,
        )
        manual_ml = ml_evaluator.evaluate_linear_regression(
            raw_df=synthetic.dirty_df,
            cleaned_df=manual_result["cleaned_df"],
            label_col=LABEL_COL,
            numeric_cols=NUMERIC_FEATURES,
            categorical_cols=CATEGORICAL_FEATURES,
        )
        validation_ml = ml_evaluator.evaluate_linear_regression(
            raw_df=synthetic.dirty_df,
            cleaned_df=synthetic.dirty_df,
            label_col=LABEL_COL,
            numeric_cols=NUMERIC_FEATURES,
            categorical_cols=CATEGORICAL_FEATURES,
        )

        report = build_report(
            metadata=synthetic.metadata,
            autoclean_result=autoclean_result,
            manual_result=manual_result,
            autoclean_ml=autoclean_ml,
            manual_ml=manual_ml,
            validation_result=validation_result,
            validation_ml=validation_ml,
        )
        print_summary(report)

        if args.log_dir:
            run_dir = log_run(
                Path(args.log_dir),
                "synthetic_house_price",
                {
                    "metadata": {
                        "dataset": "synthetic_house_price",
                        "synthetic_metadata": synthetic.metadata,
                        "validation_folds": args.validation_folds,
                    },
                    "autoclean_policy": policy,
                    "autoclean_evaluation": autoclean_result.evaluation,
                    "autoclean_opex": autoclean_result.opex_metrics,
                    "autoclean_repair_actions": autoclean_result.repair_actions,
                    "autoclean_ml": autoclean_ml,
                    "manual_evaluation": manual_result["evaluation"],
                    "manual_opex": manual_result["opex_metrics"],
                    "manual_repair_actions": manual_result["actions"],
                    "manual_ml": manual_ml,
                    "validation_only": validation_result,
                    "validation_only_ml": validation_ml,
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
    """Profile only — no repair. Isolates the value of the repair step."""
    from autocleanml import DataProfiler
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
    cleaned_df, actions = manual_clean_regression(dirty_df)
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
    evaluation = evaluator.evaluate(raw_profile, cleaned_profile, actions).metrics
    evaluation_time = perf_counter() - evaluation_start
    total_time = perf_counter() - total_start

    return {
        "cleaned_df": cleaned_df,
        "evaluation": evaluation,
        "actions": actions,
        "opex_metrics": build_manual_opex(
            raw_profile,
            cleaned_profile,
            evaluation,
            actions,
            raw_profile_time,
            repair_time,
            cleaned_profile_time,
            evaluation_time,
            total_time,
        ),
    }


def manual_clean_regression(
    df: DataFrame,
) -> tuple[DataFrame, list[dict[str, Any]]]:
    cleaned_df = df
    actions: list[dict[str, Any]] = []
    row_count = cleaned_df.count()

    cleaned_df = cleaned_df.withColumn("sqft_living", F.col("sqft_living").cast("double"))
    actions.append({
        "issue": "schema_drift",
        "column": "sqft_living",
        "strategy": "manual_cast_to_double",
    })

    if "data_source" in cleaned_df.columns:
        cleaned_df = cleaned_df.drop("data_source")
        actions.append({
            "issue": "schema_drift",
            "column": "data_source",
            "strategy": "manual_drop_added_column",
        })

    school_missing = cleaned_df.filter(F.col("school_rating").isNull()).count()
    if school_missing > 0:
        median = cleaned_df.approxQuantile("school_rating", [0.5], 0.01)[0]
        cleaned_df = cleaned_df.fillna({"school_rating": median})
        actions.append({
            "issue": "missingness",
            "column": "school_rating",
            "strategy": "manual_median_fill",
            "fill_value": median,
            "missing_count": school_missing,
        })

    hood_missing = cleaned_df.filter(F.col("neighborhood").isNull()).count()
    if hood_missing > 0:
        cleaned_df = cleaned_df.fillna({"neighborhood": "unknown"})
        actions.append({
            "issue": "missingness",
            "column": "neighborhood",
            "strategy": "manual_constant_fill_unknown",
            "missing_count": hood_missing,
        })

    outlier_stats = manual_iqr_outlier_stats(cleaned_df, "sqft_lot", row_count)
    lower_bound = outlier_stats["lower_bound"]
    upper_bound = outlier_stats["upper_bound"]
    if lower_bound is not None and upper_bound is not None:
        cleaned_df = cleaned_df.withColumn(
            "sqft_lot",
            F.when(F.col("sqft_lot") < lower_bound, F.lit(lower_bound))
            .when(F.col("sqft_lot") > upper_bound, F.lit(upper_bound))
            .otherwise(F.col("sqft_lot")),
        )
        actions.append({
            "issue": "outliers",
            "column": "sqft_lot",
            "strategy": "manual_iqr_cap",
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

    skew_stats = manual_skew_stats(cleaned_df, "neighborhood")
    if skew_stats["severity"] == "high":
        cleaned_df = cleaned_df.repartition(4, F.col("neighborhood"))
        actions.append({
            "issue": "skew",
            "column": "neighborhood",
            "strategy": "manual_repartition",
            "target_partitions": 4,
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

    if "target_noise" in issues:
        result["target_noise"] = {
            "injected": issues["target_noise"]["rate"],
            "evaluated": False,
            "message": "Continuous target noise is not detectable by the profiler's label noise scorer.",
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
    manual_result: dict[str, Any],
    autoclean_ml: Any,
    manual_ml: Any,
    validation_result: dict[str, Any],
    validation_ml: Any,
) -> dict[str, Any]:
    return {
        "experiment": "synthetic_house_price",
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
            "raw": {
                "rmse": autoclean_ml.raw_metrics.get("rmse"),
                "mae": autoclean_ml.raw_metrics.get("mae"),
                "r2": autoclean_ml.raw_metrics.get("r2"),
                "ml_row_count": autoclean_ml.raw_metrics.get("ml_row_count"),
            },
            "validation_only": {
                "rmse": validation_ml.cleaned_metrics.get("rmse"),
                "mae": validation_ml.cleaned_metrics.get("mae"),
                "r2": validation_ml.cleaned_metrics.get("r2"),
                "ml_row_count": validation_ml.cleaned_metrics.get("ml_row_count"),
                "delta_vs_raw": validation_ml.delta,
                "note": "No repair applied — ML result expected to match raw baseline.",
            },
            "autocleanml": {
                "rmse": autoclean_ml.cleaned_metrics.get("rmse"),
                "mae": autoclean_ml.cleaned_metrics.get("mae"),
                "r2": autoclean_ml.cleaned_metrics.get("r2"),
                "ml_row_count": autoclean_ml.cleaned_metrics.get("ml_row_count"),
                "delta_vs_raw": autoclean_ml.delta,
                "note": "Full AutoCleanML pipeline including median/mode imputation.",
            },
            "manual_baseline": {
                "rmse": manual_ml.cleaned_metrics.get("rmse"),
                "mae": manual_ml.cleaned_metrics.get("mae"),
                "r2": manual_ml.cleaned_metrics.get("r2"),
                "ml_row_count": manual_ml.cleaned_metrics.get("ml_row_count"),
                "delta_vs_raw": manual_ml.delta,
                "note": "Hardcoded dataset-specific cleaning. ML parity and OPEX reference.",
            },
        },
        "interpretation": (
            "Four conditions: raw (dirty data, null rows dropped for ML), "
            "validation-only (profiling only — ML matches raw), "
            "AutoCleanML (full pipeline with imputation), "
            "manual baseline (hardcoded cleaning — OPEX and ML parity reference)."
        ),
        "opex": {
            "autocleanml": autoclean_result.opex_metrics,
            "manual_baseline": manual_result["opex_metrics"],
            "dataset_specific_code_lines_reduced": (
                manual_result["opex_metrics"]["dataset_specific_cleaning_code_lines"]
            ),
        },
    }


def summarize_quality(evaluation: dict[str, Any]) -> dict[str, Any]:
    skew_by_col = evaluation.get("skew", {}).get("by_column", {})
    skew_cols_reduced = sum(
        1 for v in skew_by_col.values()
        if v.get("reduction") is not None and v["reduction"] > 0
    ) if skew_by_col else None
    return {
        "missingness_reduction": evaluation.get("missingness", {}).get("reduction"),
        "duplicate_reduction": evaluation.get("duplicates", {}).get("reduction"),
        "outlier_reduction": evaluation.get("outliers", {}).get("reduction"),
        "skew_columns_reduced": skew_cols_reduced,
        "schema_raw_issue_count": evaluation.get("schema_drift", {}).get("raw_issue_count"),
        "schema_cleaned_issue_count": evaluation.get("schema_drift", {}).get("cleaned_issue_count"),
        "repair_actions_by_issue": evaluation.get("repair_actions_by_issue"),
    }


def build_manual_opex(
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
        inspect.getsourcelines(manual_clean_regression)[0]
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


def print_summary(report: dict[str, Any]) -> None:
    auto_quality = report["data_quality"]["autocleanml"]
    raw_ml = report["ml_metrics"]["raw"]
    val_ml = report["ml_metrics"]["validation_only"]
    auto_ml = report["ml_metrics"]["autocleanml"]
    manual_ml = report["ml_metrics"]["manual_baseline"]
    print("\n=== Synthetic House Price Regression Experiment ===")
    print(f"Missingness reduction: {auto_quality['missingness_reduction']}")
    print(f"Duplicate reduction: {auto_quality['duplicate_reduction']}")
    print(f"Outlier reduction: {auto_quality['outlier_reduction']}")
    print(
        "Schema issues: "
        f"{auto_quality['schema_raw_issue_count']} -> "
        f"{auto_quality['schema_cleaned_issue_count']}"
    )
    print("\n--- ML Performance: Four-Condition Comparison ---")
    print(f"{'Condition':<30} {'RMSE':>10} {'MAE':>10} {'R2':>10} {'Rows':>8}")
    print("-" * 72)
    for label, m in [
        ("Raw (dirty)", raw_ml),
        ("Validation-only", val_ml),
        ("AutoCleanML", auto_ml),
        ("Manual baseline", manual_ml),
    ]:
        print(
            f"{label:<30} {str(m.get('rmse')):>10} "
            f"{str(m.get('mae')):>10} {str(m.get('r2')):>10} "
            f"{str(m.get('ml_row_count')):>8}"
        )
    print("\nDeltas vs raw:")
    for label, m in [
        ("AutoCleanML", auto_ml),
        ("Manual baseline", manual_ml),
    ]:
        delta = m.get("delta_vs_raw", {})
        print(f"  {label}: rmse={delta.get('rmse')}  mae={delta.get('mae')}  r2={delta.get('r2')}")


def build_spark() -> SparkSession:
    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    return (
        SparkSession.builder
        .appName("autocleanml-synthetic-regression")
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def log_run(output_dir: Path, run_name: str, artifacts: dict[str, Any]) -> Path:
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
