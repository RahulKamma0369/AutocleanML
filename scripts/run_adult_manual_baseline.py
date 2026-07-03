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
    SparkMLClassificationEvaluator,
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

MISSING_CATEGORICAL_COLUMNS = [
    "workclass",
    "occupation",
    "native_country",
]

OUTLIER_COLUMNS = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a dataset-specific manual Adult cleaning baseline."
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
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for fast smoke tests. Omit for thesis runs.",
    )
    parser.add_argument(
        "--ml-eval",
        action="store_true",
        help="Train raw-vs-cleaned logistic regression models and print ML metrics.",
    )
    parser.add_argument(
        "--validation-folds",
        type=int,
        default=1,
        help="Repeated validation folds for ML stability metrics when --ml-eval is set.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Optional directory where JSON baseline artifacts should be written.",
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
        if args.limit is not None:
            df = df.limit(args.limit)

        profiler = DataProfiler()
        evaluator = DataQualityEvaluator()

        total_start = perf_counter()
        raw_profile_start = perf_counter()
        raw_profile = profiler.profile(
            df,
            key_columns=["education", "occupation", "native_country"],
            label_col="income",
        )
        raw_profile_time = perf_counter() - raw_profile_start

        repair_start = perf_counter()
        cleaned_df, manual_actions = manual_clean_adult_dataframe(df)
        repair_time = perf_counter() - repair_start

        cleaned_profile_start = perf_counter()
        cleaned_profile = profiler.profile(
            cleaned_df,
            key_columns=["education", "occupation", "native_country"],
            label_col="income",
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
                ml_result = SparkMLClassificationEvaluator(
                    validation_folds=args.validation_folds,
                ).evaluate_logistic_regression(
                    raw_df=df,
                    cleaned_df=cleaned_df,
                    label_col="income",
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

        print_summary(
            raw_profile=raw_profile,
            cleaned_profile=cleaned_profile,
            evaluation=evaluation,
            process_metrics=process_metrics,
            ml_result=ml_result,
            ml_error=ml_error,
        )

        if args.log_dir:
            run_dir = log_manual_run(
                output_dir=Path(args.log_dir),
                run_name="adult_manual_baseline",
                artifacts={
                    "metadata": {
                        "dataset": "uci_adult",
                        "baseline_type": "manual_dataset_specific_spark_script",
                        "data_paths": paths,
                        "include_test": args.include_test,
                        "limit": args.limit,
                        "ml_eval_enabled": args.ml_eval,
                        "validation_folds": args.validation_folds,
                        "driver_memory": args.driver_memory,
                        "executor_memory": args.executor_memory,
                        "ml_error": ml_error,
                        "numeric_features": NUMERIC_FEATURES,
                        "categorical_features": CATEGORICAL_FEATURES,
                        "label_col": "income",
                    },
                    "raw_profile": raw_profile,
                    "manual_cleaning_actions": manual_actions,
                    "cleaned_profile": cleaned_profile,
                    "evaluation": evaluation,
                    "manual_process_metrics": process_metrics,
                    "ml_metrics": ml_result,
                    "manual_baseline_report": {
                        "ml_metrics": ml_result,
                        "data_quality_metrics": {
                            "change_in_missingness": evaluation.get("missingness"),
                            "reduction_in_duplicates": evaluation.get("duplicates"),
                            "outlier_reduction": evaluation.get("outliers"),
                            "improvement_in_skew_balance": evaluation.get("skew"),
                            "schema_consistency_measures": evaluation.get(
                                "schema_drift"
                            ),
                        },
                        "process_efficiency_metrics": process_metrics,
                    },
                },
            )
            print(f"\nManual baseline artifacts written to: {run_dir}")
    finally:
        spark.stop()


def manual_clean_adult_dataframe(
    df: DataFrame,
) -> tuple[DataFrame, list[dict[str, Any]]]:
    """
    Adult-specific manual baseline.

    This function intentionally hard-codes columns and cleaning choices so its
    code size and runtime can be compared against AutoCleanML's reusable policy.
    """

    cleaned_df = df
    actions: list[dict[str, Any]] = []
    row_count = cleaned_df.count()

    for col in MISSING_CATEGORICAL_COLUMNS:
        missing_count = cleaned_df.filter(F.col(col).isNull()).count()
        if missing_count > 0:
            cleaned_df = cleaned_df.fillna({col: "unknown"})
            actions.append({
                "issue": "missingness",
                "column": col,
                "strategy": "manual_fill_constant_unknown",
                "missing_count": missing_count,
            })

    for col in OUTLIER_COLUMNS:
        stats = manual_iqr_outlier_stats(cleaned_df, col, row_count)

        outlier_count = stats.get("outlier_count", 0)
        if not stats.get("repairable", True):
            actions.append({
                "issue": "outliers",
                "column": col,
                "strategy": "manual_skip_zero_iqr",
                "method": stats.get("method"),
                "outlier_count": outlier_count,
            })
            continue

        lower_bound = stats.get("lower_bound")
        upper_bound = stats.get("upper_bound")
        if lower_bound is None or upper_bound is None:
            continue
        if outlier_count <= 0:
            continue

        cleaned_df = cleaned_df.withColumn(
            col,
            F.when(F.col(col) < lower_bound, F.lit(lower_bound))
            .when(F.col(col) > upper_bound, F.lit(upper_bound))
            .otherwise(F.col(col)),
        )
        actions.append({
            "issue": "outliers",
            "column": col,
            "strategy": "manual_iqr_cap",
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": outlier_count,
            "outlier_ratio": stats.get(
                "outlier_ratio",
                round(outlier_count / row_count, 4) if row_count else 0.0,
            ),
            })

    duplicate_count = row_count - cleaned_df.distinct().count()
    if duplicate_count > 0:
        cleaned_df = cleaned_df.dropDuplicates()
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
        return {
            "method": "insufficient_quantiles",
            "outlier_count": 0,
            "repairable": False,
        }

    q1, q3 = quantiles
    iqr = q3 - q1
    if iqr == 0:
        outlier_count = df.filter(
            F.col(column).isNotNull() & (F.col(column) != q1)
        ).count()
        return {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": None,
            "upper_bound": None,
            "outlier_count": outlier_count,
            "outlier_ratio": round(outlier_count / row_count, 4) if row_count else 0.0,
            "method": "zero_iqr_deviation",
            "repairable": False,
        }

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = df.filter(
        (F.col(column) < lower_bound) | (F.col(column) > upper_bound)
    ).count()
    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": outlier_count,
        "outlier_ratio": round(outlier_count / row_count, 4) if row_count else 0.0,
        "method": "iqr",
        "repairable": True,
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
        inspect.getsourcelines(manual_clean_adult_dataframe)[0]
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
        "manual_cleaning_actions_by_issue": evaluation.get(
            "repair_actions_by_issue",
            {},
        ),
        "dataset_specific_cleaning_code_lines": cleaning_code_lines,
        "raw_profile_time_seconds": round(raw_profile_time_seconds, 6),
        "manual_repair_time_seconds": round(repair_time_seconds, 6),
        "cleaned_profile_time_seconds": round(cleaned_profile_time_seconds, 6),
        "evaluation_time_seconds": round(evaluation_time_seconds, 6),
        "total_cycle_time_seconds": round(total_time_seconds, 6),
        "seconds_per_1000_input_rows": (
            round(total_time_seconds / input_rows * 1000, 6)
            if input_rows > 0
            else None
        ),
        "input_row_count": input_rows,
        "output_row_count": output_rows,
        "row_count_delta": output_rows - input_rows,
        "input_column_count": int(raw_profile.get("column_count", 0)),
        "output_column_count": int(cleaned_profile.get("column_count", 0)),
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
    return (
        SparkSession.builder
        .appName("autocleanml-adult-manual-baseline")
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


def load_adult_dataframe(spark: SparkSession, paths: list[str]) -> DataFrame:
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


def log_manual_run(
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
        if payload is None:
            continue
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


def print_summary(
    *,
    raw_profile: dict[str, Any],
    cleaned_profile: dict[str, Any],
    evaluation: dict[str, Any],
    process_metrics: dict[str, Any],
    ml_result: Any | None,
    ml_error: dict[str, str] | None,
) -> None:
    print("\n=== Adult Manual Baseline Summary ===")
    print(f"Raw rows: {raw_profile['row_count']}")
    print(f"Cleaned rows: {cleaned_profile['row_count']}")
    print(f"Manual cleaning steps: {process_metrics['manual_cleaning_steps']}")
    print(
        "Dataset-specific cleaning code lines: "
        f"{process_metrics['dataset_specific_cleaning_code_lines']}"
    )
    print(
        "Total manual baseline cycle seconds: "
        f"{process_metrics['total_cycle_time_seconds']}"
    )
    print(f"Repair actions: {evaluation['repair_actions_by_issue']}")
    print(
        "Missingness reduction: "
        f"{evaluation['missingness']['raw_total']} -> "
        f"{evaluation['missingness']['cleaned_total']}"
    )
    print(
        "Duplicate reduction: "
        f"{evaluation['duplicates']['raw']} -> {evaluation['duplicates']['cleaned']}"
    )
    print(
        "Outlier reduction: "
        f"{evaluation['outliers']['raw_total']} -> "
        f"{evaluation['outliers']['cleaned_total']}"
    )

    if ml_result is not None:
        print("\nML metrics:")
        print(
            "  accuracy: "
            f"{ml_result.raw_metrics['accuracy']} -> "
            f"{ml_result.cleaned_metrics['accuracy']} "
            f"(delta={ml_result.delta['accuracy']})"
        )
        print(
            "  f1: "
            f"{ml_result.raw_metrics['f1']} -> "
            f"{ml_result.cleaned_metrics['f1']} "
            f"(delta={ml_result.delta['f1']})"
        )
        print(
            "  auc: "
            f"{ml_result.raw_metrics['auc']} -> "
            f"{ml_result.cleaned_metrics['auc']} "
            f"(delta={ml_result.delta['auc']})"
        )
    elif ml_error is not None:
        print("\nML evaluation failed; manual baseline artifacts were still logged.")
        print(f"  {ml_error['type']}: {ml_error['message']}")


if __name__ == "__main__":
    main()
