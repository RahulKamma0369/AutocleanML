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


RED_WINE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
    "winequality-red.csv"
)
WHITE_WINE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
    "winequality-white.csv"
)

NUMERIC_FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
CATEGORICAL_FEATURES = ["wine_type"]
LABEL_COL = "quality"
KEY_COLUMNS = ["wine_type"]
OUTLIER_COLUMNS = ["residual_sugar"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a dataset-specific manual Wine Quality cleaning baseline."
    )
    parser.add_argument("--data-dir", default="autocleanml/data/wine_quality")
    parser.add_argument("--include-white", action="store_true")
    parser.add_argument("--missing-rate", type=float, default=0.05)
    parser.add_argument("--duplicate-rate", type=float, default=0.02)
    parser.add_argument("--outlier-rate", type=float, default=0.03)
    parser.add_argument("--schema-drift", action="store_true")
    parser.add_argument("--ml-eval", action="store_true")
    parser.add_argument("--validation-folds", type=int, default=3)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for fast smoke tests. Omit for thesis runs.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    red_path = data_dir / "winequality-red.csv"
    white_path = data_dir / "winequality-white.csv"
    download_wine_files(red_path, white_path)

    spark = build_spark()
    try:
        clean_df = load_wine_dataframe(
            spark,
            red_path=red_path,
            white_path=white_path if args.include_white else None,
        )
        if args.limit is not None:
            clean_df = clean_df.limit(args.limit)

        reference_schema = schema_profile(clean_df)
        dirty_df, injected_issues = inject_wine_issues(
            clean_df,
            missing_rate=args.missing_rate,
            duplicate_rate=args.duplicate_rate,
            outlier_rate=args.outlier_rate,
            schema_drift=args.schema_drift,
        )

        profiler = DataProfiler()
        evaluator = DataQualityEvaluator()

        total_start = perf_counter()
        raw_profile_start = perf_counter()
        raw_profile = profiler.profile(
            dirty_df,
            key_columns=KEY_COLUMNS,
            reference_schema=reference_schema,
            label_col=LABEL_COL,
        )
        raw_profile_time = perf_counter() - raw_profile_start

        repair_start = perf_counter()
        cleaned_df, manual_actions = manual_clean_wine_dataframe(dirty_df)
        repair_time = perf_counter() - repair_start

        cleaned_profile_start = perf_counter()
        cleaned_profile = profiler.profile(
            cleaned_df,
            key_columns=KEY_COLUMNS,
            reference_schema=reference_schema,
            label_col=LABEL_COL,
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
                    raw_df=dirty_df,
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

        print_summary(
            raw_profile=raw_profile,
            cleaned_profile=cleaned_profile,
            evaluation=evaluation,
            process_metrics=process_metrics,
            ml_result=ml_result,
            ml_error=ml_error,
            injected_issues=injected_issues,
        )

        if args.log_dir:
            ml_metrics_serializable = None
            if ml_result is not None:
                ml_metrics_serializable = to_jsonable(ml_result)
            run_dir = log_manual_run(
                output_dir=Path(args.log_dir),
                run_name="wine_quality_manual_baseline",
                artifacts={
                    "metadata": {
                        "dataset": "uci_wine_quality",
                        "baseline_type": "manual_dataset_specific_spark_script",
                        "include_white": args.include_white,
                        "injected_issues": injected_issues,
                        "missing_rate": args.missing_rate,
                        "duplicate_rate": args.duplicate_rate,
                        "outlier_rate": args.outlier_rate,
                        "schema_drift": args.schema_drift,
                        "limit": args.limit,
                        "ml_eval_enabled": args.ml_eval,
                        "validation_folds": args.validation_folds,
                        "ml_error": ml_error,
                        "numeric_features": NUMERIC_FEATURES,
                        "categorical_features": CATEGORICAL_FEATURES,
                        "label_col": LABEL_COL,
                        "key_columns": KEY_COLUMNS,
                    },
                    "raw_profile": raw_profile,
                    "manual_cleaning_actions": manual_actions,
                    "cleaned_profile": cleaned_profile,
                    "evaluation": evaluation,
                    "manual_process_metrics": process_metrics,
                    "ml_metrics": ml_metrics_serializable,
                    "manual_baseline_report": {
                        "ml_metrics": ml_metrics_serializable,
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


def manual_clean_wine_dataframe(
    df: DataFrame,
) -> tuple[DataFrame, list[dict[str, Any]]]:
    """
    Wine Quality-specific manual baseline cleaning.

    Handles the four injected issues in order: schema drift type restoration,
    missingness (median for alcohol, constant for wine_type), IQR-capped
    outliers on residual_sugar, and exact duplicate removal.
    Code size and runtime can be compared against AutoCleanML's reusable policy.
    """
    cleaned_df = df
    actions: list[dict[str, Any]] = []
    row_count = cleaned_df.count()
    alcohol_type = dict(cleaned_df.dtypes).get("alcohol")

    if alcohol_type is not None and alcohol_type != "double":
        cleaned_df = cleaned_df.withColumn("alcohol", F.col("alcohol").cast("double"))
        actions.append({
            "issue": "schema_drift",
            "column": "alcohol",
            "strategy": "manual_cast_to_double",
            "from_type": alcohol_type,
            "to_type": "double",
        })

    if "source_system" in cleaned_df.columns:
        cleaned_df = cleaned_df.drop("source_system")
        actions.append({
            "issue": "schema_drift",
            "column": "source_system",
            "strategy": "manual_drop_added_column",
        })

    alcohol_missing = cleaned_df.filter(F.col("alcohol").isNull()).count()
    if alcohol_missing > 0:
        quantiles = cleaned_df.approxQuantile("alcohol", [0.5], 0.01)
        median_val = quantiles[0] if quantiles else 10.0
        cleaned_df = cleaned_df.withColumn(
            "alcohol",
            F.when(F.col("alcohol").isNull(), F.lit(float(median_val))).otherwise(
                F.col("alcohol")
            ),
        )
        actions.append({
            "issue": "missingness",
            "column": "alcohol",
            "strategy": "manual_fill_median",
            "fill_value": median_val,
            "missing_count": alcohol_missing,
        })

    wine_type_missing = cleaned_df.filter(F.col("wine_type").isNull()).count()
    if wine_type_missing > 0:
        cleaned_df = cleaned_df.fillna({"wine_type": "unknown"})
        actions.append({
            "issue": "missingness",
            "column": "wine_type",
            "strategy": "manual_fill_constant_unknown",
            "missing_count": wine_type_missing,
        })

    for col in OUTLIER_COLUMNS:
        stats = manual_iqr_outlier_stats(cleaned_df, col, row_count)
        outlier_count = stats.get("outlier_count", 0)
        if not stats.get("repairable", True):
            actions.append({
                "issue": "outliers",
                "column": col,
                "strategy": "manual_skip_zero_iqr",
                "outlier_count": outlier_count,
            })
            continue
        lower_bound = stats.get("lower_bound")
        upper_bound = stats.get("upper_bound")
        if lower_bound is None or upper_bound is None or outlier_count <= 0:
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
        inspect.getsourcelines(manual_clean_wine_dataframe)[0]
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


def download_wine_files(red_path: Path, white_path: Path) -> None:
    red_path.parent.mkdir(parents=True, exist_ok=True)
    if not red_path.exists():
        print(f"Downloading {RED_WINE_URL}")
        urlretrieve(RED_WINE_URL, red_path)
    if not white_path.exists():
        print(f"Downloading {WHITE_WINE_URL}")
        urlretrieve(WHITE_WINE_URL, white_path)


def load_wine_dataframe(
    spark: SparkSession,
    red_path: Path,
    white_path: Path | None,
) -> DataFrame:
    red_df = read_wine_csv(spark, red_path).withColumn("wine_type", F.lit("red"))
    if white_path is None:
        return red_df
    white_df = read_wine_csv(spark, white_path).withColumn("wine_type", F.lit("white"))
    return red_df.unionByName(white_df)


def read_wine_csv(spark: SparkSession, path: Path) -> DataFrame:
    raw_df = (
        spark.read
        .option("header", "true")
        .option("sep", ";")
        .csv(str(path))
    )
    renamed_df = raw_df
    for col in raw_df.columns:
        renamed_df = renamed_df.withColumnRenamed(col, normalize_column(col))
    typed_df = renamed_df
    for col in NUMERIC_FEATURES + [LABEL_COL]:
        typed_df = typed_df.withColumn(col, F.col(col).cast("double"))
    return typed_df


def inject_wine_issues(
    df: DataFrame,
    missing_rate: float,
    duplicate_rate: float,
    outlier_rate: float,
    schema_drift: bool,
) -> tuple[DataFrame, dict[str, Any]]:
    dirty_df = df
    issues: dict[str, Any] = {}

    if missing_rate > 0:
        dirty_df = dirty_df.withColumn(
            "alcohol",
            F.when(F.rand(101) < missing_rate, None).otherwise(F.col("alcohol")),
        )
        dirty_df = dirty_df.withColumn(
            "wine_type",
            F.when(F.rand(102) < missing_rate, None).otherwise(F.col("wine_type")),
        )
        issues["missingness"] = {
            "columns": ["alcohol", "wine_type"],
            "rate": missing_rate,
        }

    if outlier_rate > 0:
        dirty_df = dirty_df.withColumn(
            "residual_sugar",
            F.when(F.rand(103) < outlier_rate, F.lit(100.0)).otherwise(
                F.col("residual_sugar")
            ),
        )
        issues["outliers"] = {
            "column": "residual_sugar",
            "rate": outlier_rate,
            "outlier_value": 100.0,
        }

    if schema_drift:
        dirty_df = (
            dirty_df
            .withColumn("alcohol", F.col("alcohol").cast("string"))
            .withColumn("source_system", F.lit("wine_quality_v2"))
        )
        issues["schema_drift"] = {
            "type_changes": [{"column": "alcohol", "to_type": "string"}],
            "added_columns": ["source_system"],
        }

    if duplicate_rate > 0:
        duplicate_df = dirty_df.filter(F.rand(104) < duplicate_rate)
        dirty_df = dirty_df.unionByName(duplicate_df)
        issues["duplicates"] = {"rate": duplicate_rate}

    return dirty_df, issues


def schema_profile(df: DataFrame) -> dict[str, Any]:
    return {
        field.name: {
            "data_type": field.dataType.simpleString(),
            "nullable": field.nullable,
        }
        for field in df.schema.fields
    }


def normalize_column(col: str) -> str:
    return col.strip().replace(" ", "_")


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("autocleanml-wine-quality-manual-baseline")
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


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
    injected_issues: dict[str, Any],
) -> None:
    print("\n=== Wine Quality Manual Baseline Summary ===")
    print(f"Injected issues: {injected_issues}")
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
        print("\nRegression ML metrics:")
        print(
            f"  rmse: {ml_result.raw_metrics['rmse']} -> "
            f"{ml_result.cleaned_metrics['rmse']} "
            f"(delta={ml_result.delta['rmse']})"
        )
        print(
            f"  mae: {ml_result.raw_metrics['mae']} -> "
            f"{ml_result.cleaned_metrics['mae']} "
            f"(delta={ml_result.delta['mae']})"
        )
        print(
            f"  r2: {ml_result.raw_metrics['r2']} -> "
            f"{ml_result.cleaned_metrics['r2']} "
            f"(delta={ml_result.delta['r2']})"
        )
    elif ml_error is not None:
        print("\nML evaluation failed; manual baseline artifacts were still logged.")
        print(f"  {ml_error['type']}: {ml_error['message']}")


if __name__ == "__main__":
    main()
