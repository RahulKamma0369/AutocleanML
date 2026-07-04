from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from autocleanml import (
    AutoCleanML,
    DataProfiler,
    ExperimentLogger,
    RepairPolicy,
    SparkMLClassificationEvaluator,
    ThesisEvaluationReportBuilder,
)


ADULT_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
ADULT_TEST_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
)

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

TYPE_CASTS = {
    "age": "int",
    "fnlwgt": "double",
    "education_num": "int",
    "capital_gain": "double",
    "capital_loss": "double",
    "hours_per_week": "int",
}

NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AutoCleanML on the UCI Adult/Census Income dataset."
    )
    parser.add_argument(
        "--data-dir",
        default="autocleanml/data/adult",
        help="Directory where Adult dataset files are stored.",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Include adult.test in addition to adult.data.",
    )
    parser.add_argument(
        "--ml-eval",
        action="store_true",
        help="Train raw-vs-cleaned logistic regression models and print ML metrics.",
    )
    parser.add_argument(
        "--validation-folds",
        type=int,
        default=3,
        help="Repeated validation folds for ML stability metrics when --ml-eval is set.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Optional directory where JSON experiment artifacts should be written.",
    )
    parser.add_argument(
        "--driver-memory",
        default="4g",
        help="Spark driver memory for local ML evaluation.",
    )
    parser.add_argument(
        "--executor-memory",
        default="4g",
        help="Spark executor memory for local ML evaluation.",
    )
    parser.add_argument(
        "--manual-baseline-dir",
        default=None,
        help=(
            "Path to a run_adult_manual_baseline experiment directory. "
            "If provided, manual_process_metrics.json is read and its values "
            "are passed to ThesisEvaluationReportBuilder so that "
            "process_efficiency_metrics includes baseline comparisons."
        ),
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / "adult.data"
    test_path = data_dir / "adult.test"
    download_adult_files(train_path, test_path)

    spark = build_spark(
        driver_memory=args.driver_memory,
        executor_memory=args.executor_memory,
    )
    try:
        paths = [str(train_path)]
        if args.include_test:
            paths.append(str(test_path))

        df = load_adult_dataframe(spark, paths)
        policy = RepairPolicy()
        autoclean = AutoCleanML(repair_policy=policy)
        result = autoclean.run(
            df,
            key_columns=["education", "occupation", "native_country"],
            label_col="income",
        )

        print_summary(result)
        ml_result = None
        ml_error = None
        validation_result = run_validation_only_baseline(df)
        validation_ml = None
        if args.ml_eval:
            try:
                evaluator = SparkMLClassificationEvaluator(
                    validation_folds=args.validation_folds,
                )
                ml_result = evaluator.evaluate_logistic_regression(
                    raw_df=df,
                    cleaned_df=result.cleaned_df,
                    label_col="income",
                    numeric_cols=NUMERIC_FEATURES,
                    categorical_cols=CATEGORICAL_FEATURES,
                )
                validation_ml = evaluator.evaluate_logistic_regression(
                    raw_df=df,
                    cleaned_df=df,
                    label_col="income",
                    numeric_cols=NUMERIC_FEATURES,
                    categorical_cols=CATEGORICAL_FEATURES,
                )
                print_ml_summary(ml_result)
            except Exception as exc:
                ml_error = {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                }
                print("\nML evaluation failed; cleaning artifacts will still be logged.")
                print(f"  {ml_error['type']}: {ml_error['message']}")

        manual_baselines = load_manual_baselines(args.manual_baseline_dir)
        thesis_report = ThesisEvaluationReportBuilder().build(
            result,
            ml_metrics=ml_result,
            **manual_baselines,
        )
        if args.log_dir:
            run_dir = ExperimentLogger(args.log_dir).log_run(
                run_name="adult",
                result=result,
                policy=policy,
                metadata={
                    "dataset": "uci_adult",
                    "data_paths": paths,
                    "include_test": args.include_test,
                    "ml_eval_enabled": args.ml_eval,
                    "validation_folds": args.validation_folds,
                    "driver_memory": args.driver_memory,
                    "executor_memory": args.executor_memory,
                    "ml_error": ml_error,
                    "numeric_features": NUMERIC_FEATURES,
                    "categorical_features": CATEGORICAL_FEATURES,
                    "label_col": "income",
                    "key_columns": ["education", "occupation", "native_country"],
                    "validation_only": validation_result,
                    "validation_only_ml": validation_ml,
                },
                ml_metrics=ml_result,
                thesis_report=thesis_report,
            )
            print(f"\nExperiment artifacts written to: {run_dir}")
    finally:
        import os
        if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
            spark.stop()


def run_validation_only_baseline(df) -> dict:
    """Profile only — no repair. Isolates the value of the repair step."""
    from time import perf_counter
    profiler = DataProfiler()
    start = perf_counter()
    profile = profiler.profile(
        df,
        key_columns=["education", "occupation", "native_country"],
        label_col="income",
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


def load_manual_baselines(manual_baseline_dir: str | None) -> dict:
    if manual_baseline_dir is None:
        return {}
    metrics_path = Path(manual_baseline_dir) / "manual_process_metrics.json"
    if not metrics_path.exists():
        print(f"Warning: manual_process_metrics.json not found in {manual_baseline_dir}; skipping baselines.")
        return {}
    with metrics_path.open() as f:
        m = json.load(f)
    return {
        "manual_cleaning_steps_baseline": m.get("manual_cleaning_steps"),
        "dataset_specific_cleaning_code_lines_baseline": m.get("dataset_specific_cleaning_code_lines"),
        "manual_cycle_time_seconds_baseline": m.get("total_cycle_time_seconds"),
    }


def download_adult_files(train_path: Path, test_path: Path) -> None:
    train_path.parent.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        print(f"Downloading {ADULT_DATA_URL}")
        urlretrieve(ADULT_DATA_URL, train_path)

    if not test_path.exists():
        print(f"Downloading {ADULT_TEST_URL}")
        urlretrieve(ADULT_TEST_URL, test_path)


def build_spark(
    driver_memory: str = "4g",
    executor_memory: str = "4g",
) -> SparkSession:
    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    return (
        SparkSession.builder
        .appName("autocleanml-adult")
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", executor_memory)
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def load_adult_dataframe(spark: SparkSession, paths: list[str]):
    raw_df = (
        spark.read
        .option("header", "false")
        .option("comment", "|")
        .option("ignoreLeadingWhiteSpace", "true")
        .option("ignoreTrailingWhiteSpace", "true")
        .csv(paths)
        .toDF(*COLUMNS)
    )

    cleaned_strings_df = raw_df
    for col in COLUMNS:
        cleaned_strings_df = cleaned_strings_df.withColumn(
            col,
            F.trim(F.regexp_replace(F.col(col), r"\.$", "")),
        )
        cleaned_strings_df = cleaned_strings_df.withColumn(
            col,
            F.when(F.col(col) == "?", None).otherwise(F.col(col)),
        )

    typed_df = cleaned_strings_df
    for col, data_type in TYPE_CASTS.items():
        typed_df = typed_df.withColumn(col, F.col(col).cast(data_type))

    return typed_df


def print_summary(result) -> None:
    raw_profile = result.raw_profile
    cleaned_profile = result.cleaned_profile
    evaluation = result.evaluation

    print("\n=== Adult Dataset AutoCleanML Summary ===")
    print(f"Raw rows: {raw_profile['row_count']}")
    print(f"Cleaned rows: {cleaned_profile['row_count']}")
    print(f"Raw duplicate rows: {raw_profile['duplicates']['duplicate_count']}")
    print(f"Cleaned duplicate rows: {cleaned_profile['duplicates']['duplicate_count']}")
    print(f"Repair actions: {evaluation['repair_actions_by_issue']}")
    print_opex_summary(result)

    print("\nKey duplicate diagnostics:")
    for col, stats in raw_profile.get("duplicates", {}).get("by_key", {}).items():
        if "error" in stats:
            print(f"  {col}: {stats['error']}")
            continue
        print(
            f"  {col}: duplicate_groups={stats['duplicate_group_count']}, "
            f"duplicate_rows={stats['duplicate_row_count']}, "
            f"severity={stats['severity']}"
        )
    composite_key = raw_profile.get("duplicates", {}).get("composite_key", {})
    if composite_key.get("duplicate_group_count") is not None:
        print(
            "  composite_key "
            f"{composite_key.get('valid_key_columns')}: "
            f"duplicate_groups={composite_key['duplicate_group_count']}, "
            f"duplicate_rows={composite_key['duplicate_row_count']}, "
            f"severity={composite_key['severity']}"
        )

    print("\nMissingness reduction by column:")
    for col, stats in evaluation["missingness"]["by_column"].items():
        if stats["raw"] > 0 or stats["cleaned"] > 0:
            print(
                f"  {col}: {stats['raw']} -> {stats['cleaned']} "
                f"(reduction={stats['reduction']})"
            )

    print("\nHighest skew ratios:")
    skew_items = []
    for col, stats in raw_profile.get("skew", {}).items():
        skew_ratio = stats.get("skew_ratio")
        if skew_ratio is not None:
            skew_items.append((skew_ratio, col, stats))

    for _, col, stats in sorted(skew_items, reverse=True)[:5]:
        print(
            f"  {col}: ratio={stats['skew_ratio']}, "
            f"severity={stats['severity']}, max={stats['max_count']}, "
            f"median={stats['median_count']}"
        )

    print("\nOutlier reduction:")
    print(
        f"  total: {evaluation['outliers']['raw_total']} -> "
        f"{evaluation['outliers']['cleaned_total']} "
        f"(reduction={evaluation['outliers']['reduction']})"
    )
    print("\nOutlier details:")
    for col, stats in raw_profile.get("outliers", {}).items():
        print(
            f"  {col}: method={stats.get('method')}, "
            f"count={stats.get('outlier_count')}, "
            f"repairable={stats.get('repairable')}, "
            f"severity={stats.get('severity')}"
        )


def print_opex_summary(result) -> None:
    metrics = result.opex_metrics

    print("\nOPEX metrics:")
    print(f"  total cleaning cycle seconds: {metrics['total_time_seconds']}")
    print(f"  seconds per 1000 input rows: {metrics['seconds_per_1000_input_rows']}")
    print(f"  automated repair actions: {metrics['automated_repair_actions']}")
    print(
        "  stage seconds: "
        f"raw_profile={metrics['raw_profile_time_seconds']}, "
        f"repair={metrics['repair_time_seconds']}, "
        f"cleaned_profile={metrics['cleaned_profile_time_seconds']}, "
        f"evaluation={metrics['evaluation_time_seconds']}"
    )


def print_ml_summary(ml_result) -> None:
    raw = ml_result.raw_metrics
    cleaned = ml_result.cleaned_metrics
    delta = ml_result.delta

    print("\n=== Adult Raw vs Cleaned ML Evaluation ===")
    print("Model: Spark ML logistic regression")
    print("Raw rows with complete ML features:")
    print(
        f"  total={raw['ml_row_count']}, "
        f"train={raw['train_row_count']}, test={raw['test_row_count']}"
    )
    print("Cleaned rows with complete ML features:")
    print(
        f"  total={cleaned['ml_row_count']}, "
        f"train={cleaned['train_row_count']}, test={cleaned['test_row_count']}"
    )
    print("\nMetrics:")
    print(
        f"  accuracy: {raw['accuracy']} -> {cleaned['accuracy']} "
        f"(delta={delta['accuracy']})"
    )
    print(f"  f1: {raw['f1']} -> {cleaned['f1']} (delta={delta['f1']})")
    print(f"  auc: {raw['auc']} -> {cleaned['auc']} (delta={delta['auc']})")
    print(
        f"  weighted_precision: {raw['weighted_precision']} -> "
        f"{cleaned['weighted_precision']} "
        f"(delta={delta['weighted_precision']})"
    )
    print(
        f"  weighted_recall: {raw['weighted_recall']} -> "
        f"{cleaned['weighted_recall']} "
        f"(delta={delta['weighted_recall']})"
    )
    print("\nFold stability:")
    print(f"  raw: {raw['fold_stability']}")
    print(f"  cleaned: {cleaned['fold_stability']}")


if __name__ == "__main__":
    main()
