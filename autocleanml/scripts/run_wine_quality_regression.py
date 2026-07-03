from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from autocleanml import (
    AutoCleanML,
    DataProfiler,
    ExperimentLogger,
    RepairPolicy,
    SparkMLRegressionEvaluator,
    ThesisEvaluationReportBuilder,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AutoCleanML on the UCI Wine Quality regression dataset."
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
        "--manual-baseline-dir",
        default=None,
        help=(
            "Path to a run_wine_quality_manual_baseline experiment directory. "
            "If provided, manual_process_metrics.json is read and its values "
            "are passed to ThesisEvaluationReportBuilder so that "
            "process_efficiency_metrics includes baseline comparisons."
        ),
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
        reference_schema = schema_profile(clean_df)
        dirty_df, injected_issues = inject_wine_issues(
            clean_df,
            missing_rate=args.missing_rate,
            duplicate_rate=args.duplicate_rate,
            outlier_rate=args.outlier_rate,
            schema_drift=args.schema_drift,
        )

        policy = RepairPolicy(drop_added_columns=True)
        result = AutoCleanML(repair_policy=policy).run(
            dirty_df,
            key_columns=KEY_COLUMNS,
            reference_schema=reference_schema,
            label_col=LABEL_COL,
        )

        ml_result = None
        validation_result = run_validation_only_baseline(dirty_df, reference_schema)
        validation_ml = None
        if args.ml_eval:
            evaluator = SparkMLRegressionEvaluator(
                validation_folds=args.validation_folds,
            )
            ml_result = evaluator.evaluate_linear_regression(
                raw_df=dirty_df,
                cleaned_df=result.cleaned_df,
                label_col=LABEL_COL,
                numeric_cols=NUMERIC_FEATURES,
                categorical_cols=CATEGORICAL_FEATURES,
            )
            validation_ml = evaluator.evaluate_linear_regression(
                raw_df=dirty_df,
                cleaned_df=dirty_df,
                label_col=LABEL_COL,
                numeric_cols=NUMERIC_FEATURES,
                categorical_cols=CATEGORICAL_FEATURES,
            )

        manual_baselines = load_manual_baselines(args.manual_baseline_dir)
        thesis_report = ThesisEvaluationReportBuilder().build(
            result,
            ml_metrics=ml_result,
            **manual_baselines,
        )
        print_summary(result, ml_result, injected_issues)

        if args.log_dir:
            run_dir = ExperimentLogger(args.log_dir).log_run(
                run_name="wine_quality_regression",
                result=result,
                policy=policy,
                metadata={
                    "dataset": "uci_wine_quality",
                    "task_type": "regression",
                    "include_white": args.include_white,
                    "injected_issues": injected_issues,
                    "label_col": LABEL_COL,
                    "numeric_features": NUMERIC_FEATURES,
                    "categorical_features": CATEGORICAL_FEATURES,
                    "validation_folds": args.validation_folds,
                    "validation_only": validation_result,
                    "validation_only_ml": validation_ml,
                },
                ml_metrics=ml_result,
                thesis_report=thesis_report,
            )
            print(f"\nExperiment artifacts written to: {run_dir}")
    finally:
        spark.stop()


def run_validation_only_baseline(dirty_df, reference_schema) -> dict:
    """Profile only — no repair. Isolates the value of the repair step."""
    from time import perf_counter
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


def load_manual_baselines(manual_baseline_dir: str | None) -> dict[str, Any]:
    if manual_baseline_dir is None:
        return {}
    metrics_path = Path(manual_baseline_dir) / "manual_process_metrics.json"
    if not metrics_path.exists():
        print(
            f"Warning: manual_process_metrics.json not found in {manual_baseline_dir}; "
            "skipping baselines."
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
):
    red_df = read_wine_csv(spark, red_path).withColumn("wine_type", F.lit("red"))
    if white_path is None:
        return red_df

    white_df = read_wine_csv(spark, white_path).withColumn("wine_type", F.lit("white"))
    return red_df.unionByName(white_df)


def read_wine_csv(spark: SparkSession, path: Path):
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
    df,
    missing_rate: float,
    duplicate_rate: float,
    outlier_rate: float,
    schema_drift: bool,
):
    dirty_df = df
    issues = {}

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


def schema_profile(df) -> dict:
    return {
        field.name: {
            "data_type": field.dataType.simpleString(),
            "nullable": field.nullable,
        }
        for field in df.schema.fields
    }


def normalize_column(col: str) -> str:
    return col.strip().replace(" ", "_")


def print_summary(result, ml_result, injected_issues) -> None:
    print("\n=== Wine Quality Regression AutoCleanML Summary ===")
    print(f"Injected issues: {injected_issues}")
    print(f"Raw rows: {result.raw_profile['row_count']}")
    print(f"Cleaned rows: {result.cleaned_profile['row_count']}")
    print(f"Missingness reduction: {result.evaluation['missingness']['reduction']}")
    print(f"Duplicate reduction: {result.evaluation['duplicates']['reduction']}")
    print(f"Outlier reduction: {result.evaluation['outliers']['reduction']}")
    print(f"Schema drift: {result.evaluation['schema_drift']}")
    print(f"OPEX: {result.opex_metrics}")
    if ml_result is not None:
        print("\nRegression metrics:")
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


def build_spark() -> SparkSession:
    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    return (
        SparkSession.builder
        .appName("autocleanml-wine-quality")
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


if __name__ == "__main__":
    main()
