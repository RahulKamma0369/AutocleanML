from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.request import urlretrieve

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from autocleanml import (
    DataProfiler,
    DataQualityEvaluator,
    SparkMLRegressionEvaluator,
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

# Domain-knowledge bounds for NYC yellow taxi fares (TLC rules, 2023)
FARE_MIN = 2.50
FARE_MAX = 500.0
TRIP_DISTANCE_MIN = 0.01
PASSENGER_COUNT_MIN = 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manual cleaning baseline for NYC Yellow Taxi dataset."
    )
    parser.add_argument("--data-dir", default="autocleanml/data/nyc_taxi")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--ml-eval", action="store_true")
    parser.add_argument("--validation-folds", type=int, default=3)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--driver-memory", default="6g")
    parser.add_argument("--executor-memory", default="6g")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    parquet_path = data_dir / "yellow_tripdata_2023-01.parquet"
    download_taxi_file(parquet_path)

    spark = build_spark(args.driver_memory, args.executor_memory)
    try:
        df = load_taxi_dataframe(spark, parquet_path, args.sample_size, args.sample_seed)
        print(f"Loaded {df.count()} rows after sampling and preprocessing.")

        profiler = DataProfiler()
        evaluator = DataQualityEvaluator()

        total_start = perf_counter()
        raw_profile_start = perf_counter()
        raw_profile = profiler.profile(df, key_columns=KEY_COLUMNS, label_col=LABEL_COL)
        raw_profile_time = perf_counter() - raw_profile_start

        repair_start = perf_counter()
        cleaned_df, manual_actions = manual_clean_taxi_dataframe(df)
        repair_time = perf_counter() - repair_start

        cleaned_profile_start = perf_counter()
        cleaned_profile = profiler.profile(
            cleaned_df, key_columns=KEY_COLUMNS, label_col=LABEL_COL
        )
        cleaned_profile_time = perf_counter() - cleaned_profile_start

        evaluation_start = perf_counter()
        evaluation = evaluator.evaluate(
            raw_profile=raw_profile,
            cleaned_profile=cleaned_profile,
            repair_actions=manual_actions,
        ).metrics
        evaluation_time = perf_counter() - evaluation_start
        total_time = perf_counter() - total_start

        ml_result = None
        ml_error = None
        if args.ml_eval:
            try:
                ml_result = SparkMLRegressionEvaluator(
                    validation_folds=args.validation_folds,
                ).evaluate_linear_regression(
                    raw_df=df,
                    cleaned_df=cleaned_df,
                    label_col=LABEL_COL,
                    numeric_cols=NUMERIC_FEATURES,
                    categorical_cols=CATEGORICAL_FEATURES,
                )
            except Exception as exc:
                ml_error = {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                }

        process_metrics = build_manual_process_metrics(
            raw_profile=raw_profile,
            cleaned_profile=cleaned_profile,
            evaluation=evaluation,
            manual_actions=manual_actions,
            raw_profile_time_seconds=raw_profile_time,
            repair_time_seconds=repair_time,
            cleaned_profile_time_seconds=cleaned_profile_time,
            evaluation_time_seconds=evaluation_time,
            total_time_seconds=total_time,
        )

        print_summary(raw_profile, cleaned_profile, evaluation, process_metrics, ml_result)

        if args.log_dir:
            run_dir = _log_run(
                log_dir=args.log_dir,
                raw_profile=raw_profile,
                cleaned_profile=cleaned_profile,
                evaluation=evaluation,
                manual_actions=manual_actions,
                process_metrics=process_metrics,
                ml_result=ml_result,
                ml_error=ml_error,
                args=args,
            )
            print(f"\nManual baseline artifacts written to: {run_dir}")
    finally:
        import os
        if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
            spark.stop()


def manual_clean_taxi_dataframe(
    df: DataFrame,
) -> tuple[DataFrame, list[dict[str, Any]]]:
    """
    NYC taxi-specific manual cleaning baseline.

    Applies domain-knowledge rules and IQR-based outlier capping using only
    hardcoded column names. Code size and runtime are measured and compared
    against AutoCleanML's reusable pipeline.
    """
    cleaned_df = df
    actions: list[dict[str, Any]] = []
    row_count = cleaned_df.count()

    # 1. Drop rows with invalid fare amounts (domain rule: TLC minimum $2.50)
    invalid_fares = cleaned_df.filter(
        F.col("fare_amount").isNull()
        | (F.col("fare_amount") < FARE_MIN)
    ).count()
    if invalid_fares > 0:
        cleaned_df = cleaned_df.filter(
            F.col("fare_amount").isNotNull() & (F.col("fare_amount") >= FARE_MIN)
        )
        actions.append({
            "issue": "outliers",
            "column": "fare_amount",
            "strategy": "manual_domain_filter_below_minimum",
            "threshold": FARE_MIN,
            "rows_removed": invalid_fares,
        })

    # 2. Cap extreme fare_amount outliers using IQR
    fare_stats = manual_iqr_outlier_stats(cleaned_df, "fare_amount", cleaned_df.count())
    if fare_stats.get("repairable") and fare_stats.get("outlier_count", 0) > 0:
        lower = fare_stats["lower_bound"]
        upper = min(fare_stats["upper_bound"], FARE_MAX)
        cleaned_df = cleaned_df.withColumn(
            "fare_amount",
            F.when(F.col("fare_amount") < lower, F.lit(lower))
            .when(F.col("fare_amount") > upper, F.lit(upper))
            .otherwise(F.col("fare_amount")),
        )
        actions.append({
            "issue": "outliers",
            "column": "fare_amount",
            "strategy": "manual_iqr_cap",
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": fare_stats["outlier_count"],
        })

    # 3. Drop rows with zero or missing trip_distance (invalid trips)
    invalid_dist = cleaned_df.filter(
        F.col("trip_distance").isNull() | (F.col("trip_distance") < TRIP_DISTANCE_MIN)
    ).count()
    if invalid_dist > 0:
        cleaned_df = cleaned_df.filter(
            F.col("trip_distance").isNotNull()
            & (F.col("trip_distance") >= TRIP_DISTANCE_MIN)
        )
        actions.append({
            "issue": "outliers",
            "column": "trip_distance",
            "strategy": "manual_domain_filter_zero_distance",
            "threshold": TRIP_DISTANCE_MIN,
            "rows_removed": invalid_dist,
        })

    # 4. Impute missing passenger_count with mode (most common non-null value)
    null_passengers = cleaned_df.filter(F.col("passenger_count").isNull()).count()
    if null_passengers > 0:
        mode_row = (
            cleaned_df.filter(F.col("passenger_count").isNotNull())
            .groupBy("passenger_count")
            .count()
            .orderBy(F.col("count").desc())
            .first()
        )
        mode_val = float(mode_row["passenger_count"]) if mode_row else 1.0
        cleaned_df = cleaned_df.fillna({"passenger_count": mode_val})
        actions.append({
            "issue": "missingness",
            "column": "passenger_count",
            "strategy": "manual_fill_mode",
            "fill_value": mode_val,
            "missing_count": null_passengers,
        })

    # 5. Fill missing payment_type and RatecodeID with "unknown"
    for cat_col in ["payment_type", "RatecodeID"]:
        null_count = cleaned_df.filter(F.col(cat_col).isNull()).count()
        if null_count > 0:
            cleaned_df = cleaned_df.fillna({cat_col: "unknown"})
            actions.append({
                "issue": "missingness",
                "column": cat_col,
                "strategy": "manual_fill_constant_unknown",
                "missing_count": null_count,
            })

    # 6. Drop exact duplicate rows
    count_before = cleaned_df.count()
    cleaned_df = cleaned_df.dropDuplicates()
    duplicate_count = count_before - cleaned_df.count()
    if duplicate_count > 0:
        actions.append({
            "issue": "duplicates",
            "strategy": "manual_drop_exact_duplicates",
            "duplicate_count": duplicate_count,
        })

    return cleaned_df, actions


def manual_iqr_outlier_stats(
    df: DataFrame,
    column: str,
    row_count: int,
) -> dict[str, Any]:
    quantiles = df.approxQuantile(column, [0.25, 0.75], 0.01)
    if len(quantiles) < 2:
        return {"repairable": False, "method": "iqr", "outlier_count": 0}
    q1, q3 = quantiles
    iqr = q3 - q1
    if iqr == 0:
        return {"repairable": False, "method": "iqr", "outlier_count": 0}
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_count = df.filter(
        (F.col(column) < lower) | (F.col(column) > upper)
    ).count()
    return {
        "repairable": True,
        "method": "iqr",
        "q1": q1,
        "q3": q3,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": outlier_count,
        "outlier_ratio": round(outlier_count / row_count, 4) if row_count else 0.0,
    }


def build_manual_process_metrics(
    *,
    raw_profile: dict[str, Any],
    cleaned_profile: dict[str, Any],
    evaluation: dict[str, Any],
    manual_actions: list[dict[str, Any]],
    raw_profile_time_seconds: float,
    repair_time_seconds: float,
    cleaned_profile_time_seconds: float,
    evaluation_time_seconds: float,
    total_time_seconds: float,
) -> dict[str, Any]:
    source_lines = (
        inspect.getsourcelines(manual_clean_taxi_dataframe)[0]
        + inspect.getsourcelines(manual_iqr_outlier_stats)[0]
    )
    cleaning_code_lines = len([
        line for line in source_lines
        if line.strip() and not line.strip().startswith("#")
    ])
    input_rows = int(raw_profile.get("row_count", 0))
    output_rows = int(cleaned_profile.get("row_count", 0))

    return {
        "manual_cleaning_steps": len(manual_actions),
        "manual_cleaning_actions_by_issue": evaluation.get("repair_actions_by_issue", {}),
        "dataset_specific_cleaning_code_lines": cleaning_code_lines,
        "raw_profile_time_seconds": round(raw_profile_time_seconds, 6),
        "manual_repair_time_seconds": round(repair_time_seconds, 6),
        "cleaned_profile_time_seconds": round(cleaned_profile_time_seconds, 6),
        "evaluation_time_seconds": round(evaluation_time_seconds, 6),
        "total_cycle_time_seconds": round(total_time_seconds, 6),
        "seconds_per_1000_input_rows": (
            round(total_time_seconds / input_rows * 1000, 6) if input_rows > 0 else None
        ),
        "input_row_count": input_rows,
        "output_row_count": output_rows,
        "row_count_delta": output_rows - input_rows,
        "input_column_count": int(raw_profile.get("column_count", 0)),
        "output_column_count": int(cleaned_profile.get("column_count", 0)),
    }


def print_summary(raw_profile, cleaned_profile, evaluation, process_metrics, ml_result) -> None:
    print("\n=== NYC Taxi Manual Baseline Summary ===")
    print(f"Input rows:  {process_metrics['input_row_count']}")
    print(f"Output rows: {process_metrics['output_row_count']} "
          f"(delta={process_metrics['row_count_delta']})")
    print(f"Manual cleaning steps: {process_metrics['manual_cleaning_steps']}")
    print(f"Dataset-specific code lines: {process_metrics['dataset_specific_cleaning_code_lines']}")
    print(f"Total cycle time: {process_metrics['total_cycle_time_seconds']:.3f}s")
    print(
        f"  raw_profile={process_metrics['raw_profile_time_seconds']:.3f}s  "
        f"repair={process_metrics['manual_repair_time_seconds']:.3f}s  "
        f"cleaned_profile={process_metrics['cleaned_profile_time_seconds']:.3f}s  "
        f"evaluation={process_metrics['evaluation_time_seconds']:.3f}s"
    )
    print(f"Missingness reduction: {evaluation['missingness']['reduction']}")
    print(f"Duplicate reduction:   {evaluation['duplicates']['reduction']}")
    print(f"Outlier reduction:     {evaluation['outliers']['reduction']}")
    if ml_result is not None:
        print("\n--- ML metrics (manual baseline vs raw) ---")
        print(f"  rmse: {ml_result.raw_metrics['rmse']} -> {ml_result.cleaned_metrics['rmse']} "
              f"(delta={ml_result.delta['rmse']})")
        print(f"  mae:  {ml_result.raw_metrics['mae']} -> {ml_result.cleaned_metrics['mae']} "
              f"(delta={ml_result.delta['mae']})")
        print(f"  r2:   {ml_result.raw_metrics['r2']} -> {ml_result.cleaned_metrics['r2']} "
              f"(delta={ml_result.delta['r2']})")


def _serialize(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def _log_run(
    *,
    log_dir: str,
    raw_profile: dict,
    cleaned_profile: dict,
    evaluation: dict,
    manual_actions: list,
    process_metrics: dict,
    ml_result: Any,
    ml_error: dict | None,
    args: Any,
) -> Path:
    run_dir = Path(log_dir) / f"nyc_taxi_manual_baseline_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_name": "nyc_taxi_manual_baseline",
        "dataset": "nyc_taxi_yellow_2023_01",
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed,
        "manual_actions": manual_actions,
        "raw_profile": raw_profile,
        "cleaned_profile": cleaned_profile,
        "evaluation": evaluation,
        "manual_process_metrics": process_metrics,
        "ml_metrics": _serialize(ml_result) if ml_result else None,
        "ml_error": ml_error,
    }
    (run_dir / "manual_process_metrics.json").write_text(
        json.dumps(process_metrics, indent=2)
    )
    (run_dir / "run_summary.json").write_text(
        json.dumps(payload, indent=2, default=str)
    )
    return run_dir


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
    return df.sample(withReplacement=False, fraction=fraction * 1.2, seed=seed).limit(sample_size)


def download_taxi_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading NYC Yellow Taxi data from {TAXI_URL} ...")
        urlretrieve(TAXI_URL, path)
        print(f"Saved to {path}")
    else:
        print(f"Using cached file: {path}")


def build_spark(driver_memory: str = "6g", executor_memory: str = "6g") -> SparkSession:
    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    import os
    master = os.environ.get("SPARK_MASTER_URL", "local[*]")
    builder = (
        SparkSession.builder
        .appName("autocleanml-nyc-taxi-manual-baseline")
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
