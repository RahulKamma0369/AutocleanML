from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve
from typing import Any

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from autocleanml import (
    AutoCleanML,
    DataProfiler,
    ExperimentLogger,
    RepairPolicy,
    SparkMLRegressionEvaluator,
    ThesisEvaluationReportBuilder,
)


TAXI_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
)

NUMERIC_FEATURES = [
    "passenger_count",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "hour_of_day",
    "day_of_week",
]
CATEGORICAL_FEATURES = ["payment_type", "RatecodeID"]
KEY_COLUMNS = ["payment_type", "RatecodeID"]
LABEL_COL = "fare_amount"
SAMPLE_SIZE = 100_000


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AutoCleanML on the NYC Yellow Taxi Fare dataset."
    )
    parser.add_argument(
        "--data-dir",
        default="autocleanml/data/nyc_taxi",
        help="Directory where the parquet file is stored or will be downloaded.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=SAMPLE_SIZE,
        help="Number of rows to sample from the full dataset.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
    )
    parser.add_argument("--ml-eval", action="store_true")
    parser.add_argument("--validation-folds", type=int, default=3)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--driver-memory", default="6g")
    parser.add_argument("--executor-memory", default="6g")
    parser.add_argument(
        "--manual-baseline-dir",
        default=None,
        help=(
            "Path to a run_nyc_taxi_manual_baseline experiment directory. "
            "If provided, manual_process_metrics.json is read and its values "
            "are included in the thesis report for OPEX comparison."
        ),
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    parquet_path = data_dir / "yellow_tripdata_2023-01.parquet"
    download_taxi_file(parquet_path)

    spark = build_spark(args.driver_memory, args.executor_memory)
    try:
        raw_df = load_taxi_dataframe(spark, parquet_path, args.sample_size, args.sample_seed)
        print(f"Loaded {raw_df.count()} rows after sampling and preprocessing.")

        policy = RepairPolicy()
        result = AutoCleanML(repair_policy=policy).run(
            raw_df,
            key_columns=KEY_COLUMNS,
            label_col=LABEL_COL,
        )

        validation_result = run_validation_only_baseline(raw_df)
        print_quality_summary(result)

        ml_result = None
        validation_ml = None

        if args.ml_eval:
            evaluator = SparkMLRegressionEvaluator(validation_folds=args.validation_folds)
            ml_result = evaluator.evaluate_linear_regression(
                raw_df=raw_df,
                cleaned_df=result.cleaned_df,
                label_col=LABEL_COL,
                numeric_cols=NUMERIC_FEATURES,
                categorical_cols=CATEGORICAL_FEATURES,
            )
            validation_ml = evaluator.evaluate_linear_regression(
                raw_df=raw_df,
                cleaned_df=raw_df,
                label_col=LABEL_COL,
                numeric_cols=NUMERIC_FEATURES,
                categorical_cols=CATEGORICAL_FEATURES,
            )
            print_ml_summary(ml_result)

        manual_baselines = load_manual_baselines(args.manual_baseline_dir)
        thesis_report = ThesisEvaluationReportBuilder().build(
            result,
            ml_metrics=ml_result,
            **manual_baselines,
        )

        if args.log_dir:
            run_dir = ExperimentLogger(args.log_dir).log_run(
                run_name="nyc_taxi",
                result=result,
                policy=policy,
                metadata={
                    "dataset": "nyc_taxi_yellow_2023_01",
                    "sample_size": args.sample_size,
                    "sample_seed": args.sample_seed,
                    "task_type": "regression",
                    "label_col": LABEL_COL,
                    "numeric_features": NUMERIC_FEATURES,
                    "categorical_features": CATEGORICAL_FEATURES,
                    "key_columns": KEY_COLUMNS,
                    "validation_folds": args.validation_folds,
                    "validation_only": validation_result,
                    "validation_only_ml": validation_ml,
                    "natural_issues_note": (
                        "No issues injected. Natural quality problems include: "
                        "zero/negative passenger counts, zero trip distances, "
                        "negative or extreme fare amounts, and duplicate records."
                    ),
                },
                ml_metrics=ml_result,
                thesis_report=thesis_report,
            )
            print(f"\nExperiment artifacts written to: {run_dir}")
    finally:
        import os
        if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
            spark.stop()


def load_manual_baselines(manual_baseline_dir: str | None) -> dict[str, Any]:
    if manual_baseline_dir is None:
        return {}
    metrics_path = Path(manual_baseline_dir) / "manual_process_metrics.json"
    if not metrics_path.exists():
        print(
            f"Warning: manual_process_metrics.json not found in {manual_baseline_dir}; "
            "skipping manual baseline."
        )
        return {}
    with metrics_path.open() as f:
        m = json.load(f)
    return {
        "manual_cleaning_steps_baseline": m.get("manual_cleaning_steps"),
        "dataset_specific_cleaning_code_lines_baseline": m.get(
            "dataset_specific_cleaning_code_lines"
        ),
        "manual_cycle_time_seconds_baseline": m.get("total_cycle_time_seconds"),
    }


def load_taxi_dataframe(
    spark: SparkSession,
    path: Path,
    sample_size: int,
    seed: int,
) -> DataFrame:
    df = spark.read.parquet(str(path))

    df = df.select(
        F.col("fare_amount").cast("double"),
        F.col("passenger_count").cast("double"),
        F.col("trip_distance").cast("double"),
        F.col("PULocationID").cast("double"),
        F.col("DOLocationID").cast("double"),
        F.col("payment_type").cast("int").cast("string").alias("payment_type"),
        F.col("RatecodeID").cast("int").cast("string").alias("RatecodeID"),
        F.hour("tpep_pickup_datetime").cast("double").alias("hour_of_day"),
        F.dayofweek("tpep_pickup_datetime").cast("double").alias("day_of_week"),
    )

    df = df.filter(F.col("fare_amount").isNotNull())

    total = df.count()
    fraction = min(1.0, sample_size / total) if total > 0 else 1.0
    df = df.sample(withReplacement=False, fraction=fraction * 1.2, seed=seed).limit(sample_size)

    return df


def run_validation_only_baseline(df: DataFrame) -> dict[str, Any]:
    from time import perf_counter
    profiler = DataProfiler()
    start = perf_counter()
    profile = profiler.profile(
        df,
        key_columns=KEY_COLUMNS,
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


def download_taxi_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading NYC Yellow Taxi data from {TAXI_URL} ...")
        urlretrieve(TAXI_URL, path)
        print(f"Saved to {path}")
    else:
        print(f"Using cached file: {path}")


def print_quality_summary(result) -> None:
    raw = result.raw_profile
    evaluation = result.evaluation
    print("\n=== NYC Taxi AutoCleanML Quality Summary ===")
    print(f"Raw rows: {raw['row_count']}")
    print(f"Cleaned rows: {result.cleaned_profile['row_count']}")
    print(f"Repair actions: {evaluation['repair_actions_by_issue']}")
    print(f"Missingness reduction: {evaluation['missingness']['reduction']}")
    print(f"Duplicate reduction: {evaluation['duplicates']['reduction']}")
    print(f"Outlier reduction: {evaluation['outliers']['reduction']}")

    print("\nNatural issues detected:")
    for col, stats in raw.get("outliers", {}).items():
        if stats.get("outlier_count", 0) > 0:
            print(
                f"  {col}: {stats['outlier_count']} outliers "
                f"({stats.get('outlier_ratio', 0):.1%}), "
                f"severity={stats.get('severity')}"
            )
    for col, stats in raw.get("missingness", {}).items():
        if stats.get("missing_count", 0) > 0:
            print(
                f"  {col}: {stats['missing_count']} missing "
                f"({stats.get('missing_ratio', 0):.1%}), "
                f"severity={stats.get('severity')}"
            )
    dup = raw.get("duplicates", {})
    if dup.get("duplicate_count", 0) > 0:
        print(f"  duplicates: {dup['duplicate_count']} ({dup.get('duplicate_ratio', 0):.1%})")

    print("\nKey skew:")
    for col, stats in raw.get("skew", {}).items():
        print(
            f"  {col}: ratio={stats.get('skew_ratio')}, "
            f"severity={stats.get('severity')}"
        )

    opex = result.opex_metrics
    print(f"\nOPEX: {opex.get('automated_repair_actions')} automated actions, "
          f"{opex.get('total_time_seconds'):.2f}s total")


def print_ml_summary(ml_result) -> None:
    raw = ml_result.raw_metrics
    cleaned = ml_result.cleaned_metrics
    delta = ml_result.delta
    print("\n=== NYC Taxi ML Evaluation (Linear Regression) ===")
    print(f"Raw rows used for ML:     {raw['ml_row_count']}")
    print(f"Cleaned rows used for ML: {cleaned['ml_row_count']}")
    print(f"  rmse: {raw['rmse']} -> {cleaned['rmse']} (delta={delta['rmse']})")
    print(f"  mae:  {raw['mae']} -> {cleaned['mae']} (delta={delta['mae']})")
    print(f"  r2:   {raw['r2']} -> {cleaned['r2']} (delta={delta['r2']})")


def build_spark(driver_memory: str = "6g", executor_memory: str = "6g") -> SparkSession:
    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    import os
    master = os.environ.get("SPARK_MASTER_URL", "local[*]")
    builder = (
        SparkSession.builder
        .appName("autocleanml-nyc-taxi")
        .master(master)
        .config("spark.ui.enabled", "true")
        .config("spark.sql.shuffle.partitions", "16")
    )
    if master.startswith("local"):
        builder = (
            builder
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.memory", driver_memory)
            .config("spark.executor.memory", executor_memory)
            .config("spark.driver.maxResultSize", "2g")
        )
    return builder.getOrCreate()


if __name__ == "__main__":
    main()
