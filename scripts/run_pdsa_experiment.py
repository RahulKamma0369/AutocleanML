from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from pyspark.sql import SparkSession

from autocleanml import (
    AutoCleanML,
    PDSAConfig,
    PDSALoop,
    RepairPolicy,
    SparkMLRegressionEvaluator,
    SyntheticDataGenerator,
    SyntheticIssueConfig,
)


NUMERIC_FEATURES = ["feature_num1", "feature_num2"]
CATEGORICAL_FEATURES = ["category", "join_key"]
KEY_COLUMNS = ["join_key"]
LABEL_COL = "target"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PDSA feedback loop experiment on synthetic regression dataset."
    )
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing-rate", type=float, default=0.08)
    parser.add_argument("--duplicate-rate", type=float, default=0.03)
    parser.add_argument("--outlier-rate", type=float, default=0.05)
    parser.add_argument("--skew-rate", type=float, default=0.50)
    parser.add_argument("--target-noise-rate", type=float, default=0.03)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--max-missingness-ratio", type=float, default=0.005)
    parser.add_argument("--max-outlier-ratio", type=float, default=0.005)
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
        synthetic = SyntheticDataGenerator(spark, config).generate_regression_dataset()

        repair_policy = RepairPolicy(
            drop_added_columns=True,
            skew_strategy="repartition",
            skew_target_partitions=4,
        )
        pdsa_config = PDSAConfig(
            max_missingness_ratio=args.max_missingness_ratio,
            max_outlier_ratio=args.max_outlier_ratio,
            max_duplicate_ratio=0.0,
            max_iterations=args.max_iterations,
            iterative_issue_types=("outliers", "skew"),
        )

        pdsa_result = PDSALoop(
            repair_policy=repair_policy,
            pdsa_config=pdsa_config,
        ).run(
            synthetic.dirty_df,
            key_columns=KEY_COLUMNS,
            reference_schema=synthetic.reference_schema,
            label_col=LABEL_COL,
        )

        single_pass_result = AutoCleanML(repair_policy=repair_policy).run(
            synthetic.dirty_df,
            key_columns=KEY_COLUMNS,
            reference_schema=synthetic.reference_schema,
            label_col=LABEL_COL,
        )

        ml_evaluator = SparkMLRegressionEvaluator(validation_folds=args.validation_folds)
        pdsa_ml = ml_evaluator.evaluate_linear_regression(
            raw_df=synthetic.dirty_df,
            cleaned_df=pdsa_result.cleaned_df,
            label_col=LABEL_COL,
            numeric_cols=NUMERIC_FEATURES,
            categorical_cols=CATEGORICAL_FEATURES,
        )
        single_pass_ml = ml_evaluator.evaluate_linear_regression(
            raw_df=synthetic.dirty_df,
            cleaned_df=single_pass_result.cleaned_df,
            label_col=LABEL_COL,
            numeric_cols=NUMERIC_FEATURES,
            categorical_cols=CATEGORICAL_FEATURES,
        )

        print_summary(
            pdsa_result=pdsa_result,
            pdsa_ml=pdsa_ml,
            single_pass_ml=single_pass_ml,
            pdsa_config=pdsa_config,
        )

        if args.log_dir:
            run_dir = log_run(
                Path(args.log_dir),
                "pdsa_experiment",
                {
                    "metadata": {
                        "dataset": "synthetic_regression",
                        "synthetic_metadata": synthetic.metadata,
                        "validation_folds": args.validation_folds,
                        "pdsa_config": asdict(pdsa_config),
                        "repair_policy": asdict(repair_policy),
                    },
                    "pdsa_converged": pdsa_result.converged,
                    "pdsa_total_iterations": pdsa_result.total_iterations,
                    "pdsa_iteration_records": [
                        asdict(r) for r in pdsa_result.pdsa_iterations
                    ],
                    "pdsa_evaluation": pdsa_result.evaluation,
                    "pdsa_opex": pdsa_result.opex_metrics,
                    "pdsa_ml": pdsa_ml,
                    "single_pass_evaluation": single_pass_result.evaluation,
                    "single_pass_opex": single_pass_result.opex_metrics,
                    "single_pass_ml": single_pass_ml,
                },
            )
            print(f"\nExperiment artifacts written to: {run_dir}")
    finally:
        spark.stop()


def print_summary(
    pdsa_result: Any,
    pdsa_ml: Any,
    single_pass_ml: Any,
    pdsa_config: PDSAConfig,
) -> None:
    print("\n=== PDSA Feedback Loop Experiment ===")
    print(f"Converged: {pdsa_result.converged}")
    print(f"Total iterations: {pdsa_result.total_iterations}")
    print(f"Thresholds — missingness: {pdsa_config.max_missingness_ratio}  "
          f"outlier: {pdsa_config.max_outlier_ratio}  "
          f"duplicate: {pdsa_config.max_duplicate_ratio}")

    print("\n--- PDSA Iteration History ---")
    print(
        f"{'Iter':<6} {'Issue Types':<30} {'Missingness':>12} "
        f"{'Outliers':>10} {'Duplicates':>12} {'Converged':>10}"
    )
    print("-" * 82)
    for rec in pdsa_result.pdsa_iterations:
        issue_types = ", ".join(rec.repair_issue_types)
        print(
            f"{rec.iteration:<6} {issue_types:<30} "
            f"{str(rec.residual_missingness_ratio):>12} "
            f"{str(rec.residual_outlier_ratio):>10} "
            f"{str(rec.residual_duplicate_ratio):>12} "
            f"{str(rec.thresholds_met):>10}"
        )
        if rec.policy_adjustments:
            for adj in rec.policy_adjustments:
                print(f"       Act: {adj}")

    print("\n--- ML Performance: PDSA vs Single-Pass ---")
    print(f"{'Condition':<22} {'RMSE':>10} {'MAE':>10} {'R2':>10}")
    print("-" * 54)
    for label, ml in [
        ("Raw (dirty)", pdsa_ml),
        ("Single-pass", single_pass_ml),
        ("PDSA", pdsa_ml),
    ]:
        metrics = ml.raw_metrics if label == "Raw (dirty)" else ml.cleaned_metrics
        print(
            f"{label:<22} {str(metrics.get('rmse')):>10} "
            f"{str(metrics.get('mae')):>10} {str(metrics.get('r2')):>10}"
        )


def log_run(output_dir: Path, run_name: str, artifacts: dict[str, Any]) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = output_dir / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for name, artifact in artifacts.items():
        path = run_dir / f"{name}.json"
        with path.open("w") as f:
            json.dump(to_jsonable(artifact), f, indent=2)

    return run_dir


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float) and (obj != obj):
        return None
    return obj


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("autocleanml-pdsa")
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


if __name__ == "__main__":
    main()
