from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from autocleanml import (
    AutoCleanML,
    RepairPolicy,
    SparkMLClassificationEvaluator,
    SyntheticDataGenerator,
    SyntheticIssueConfig,
)


NUMERIC_FEATURES = ["feature_num1", "feature_num2"]
CATEGORICAL_FEATURES = ["category", "join_key"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare synthetic classification cleaning with and without "
            "model-based missing-label imputation."
        )
    )
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing-rate", type=float, default=0.05)
    parser.add_argument("--duplicate-rate", type=float, default=0.02)
    parser.add_argument("--outlier-rate", type=float, default=0.03)
    parser.add_argument("--skew-rate", type=float, default=0.30)
    parser.add_argument("--label-noise-rate", type=float, default=0.0)
    parser.add_argument("--missing-label-rate", type=float, default=0.20)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--validation-folds", type=int, default=1)
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
        truth_df = synthetic.clean_df.select(
            F.col("row_id").alias("_truth_row_id"),
            F.col("label").alias("true_label"),
        )

        no_label_policy = RepairPolicy(label_imputation="none")
        impute_label_policy = RepairPolicy(
            label_imputation="model",
            label_confidence_threshold=args.confidence_threshold,
        )

        no_label_result = AutoCleanML(repair_policy=no_label_policy).run(
            synthetic.dirty_df,
            key_columns=["join_key"],
            reference_schema=synthetic.reference_schema,
            label_col="label",
        )
        imputed_label_result = AutoCleanML(repair_policy=impute_label_policy).run(
            synthetic.dirty_df,
            key_columns=["join_key"],
            reference_schema=synthetic.reference_schema,
            label_col="label",
        )

        evaluator = SparkMLClassificationEvaluator(
            validation_folds=args.validation_folds,
        )
        policy_ml_comparison = evaluator.evaluate_logistic_regression(
            raw_df=no_label_result.cleaned_df,
            cleaned_df=imputed_label_result.cleaned_df,
            label_col="label",
            numeric_cols=NUMERIC_FEATURES,
            categorical_cols=CATEGORICAL_FEATURES,
        )

        label_accuracy = {
            "without_label_imputation": label_recovery_metrics(
                no_label_result.cleaned_df,
                truth_df,
            ),
            "with_label_imputation": label_recovery_metrics(
                imputed_label_result.cleaned_df,
                truth_df,
            ),
        }
        comparison = build_comparison(
            metadata=synthetic.metadata,
            no_label_result=no_label_result,
            imputed_label_result=imputed_label_result,
            policy_ml_comparison=policy_ml_comparison,
            label_accuracy=label_accuracy,
            confidence_threshold=args.confidence_threshold,
        )

        print_summary(comparison)

        if args.log_dir:
            run_dir = log_run(
                output_dir=Path(args.log_dir),
                run_name="synthetic_label_imputation",
                artifacts={
                    "metadata": {
                        "dataset": "synthetic_classification",
                        "experiment": "missing_label_imputation",
                        "synthetic_metadata": synthetic.metadata,
                        "confidence_threshold": args.confidence_threshold,
                        "validation_folds": args.validation_folds,
                    },
                    "policy_without_label_imputation": no_label_policy,
                    "policy_with_label_imputation": impute_label_policy,
                    "without_label_imputation_evaluation": no_label_result.evaluation,
                    "with_label_imputation_evaluation": imputed_label_result.evaluation,
                    "without_label_imputation_opex": no_label_result.opex_metrics,
                    "with_label_imputation_opex": imputed_label_result.opex_metrics,
                    "policy_ml_comparison": policy_ml_comparison,
                    "label_recovery_metrics": label_accuracy,
                    "comparison": comparison,
                },
            )
            print(f"\nExperiment artifacts written to: {run_dir}")
    finally:
        spark.stop()


def label_recovery_metrics(cleaned_df: DataFrame, truth_df: DataFrame) -> dict[str, Any]:
    compared_df = (
        cleaned_df
        .join(truth_df, cleaned_df.row_id == truth_df._truth_row_id, "inner")
        .select("row_id", "label", "true_label")
    )
    total_rows = compared_df.count()
    labeled_rows = compared_df.filter(F.col("label").isNotNull()).count()
    correct_rows = compared_df.filter(F.col("label") == F.col("true_label")).count()
    remaining_missing = total_rows - labeled_rows

    return {
        "total_rows_compared_to_clean_truth": total_rows,
        "labeled_rows": labeled_rows,
        "remaining_missing_labels": remaining_missing,
        "label_coverage_ratio": round(labeled_rows / total_rows, 4)
        if total_rows
        else 0.0,
        "correct_labels_against_clean_truth": correct_rows,
        "label_accuracy_against_clean_truth": round(correct_rows / labeled_rows, 4)
        if labeled_rows
        else None,
    }


def build_comparison(
    *,
    metadata: dict[str, Any],
    no_label_result: Any,
    imputed_label_result: Any,
    policy_ml_comparison: Any,
    label_accuracy: dict[str, Any],
    confidence_threshold: float,
) -> dict[str, Any]:
    label_action = next(
        (
            action
            for action in imputed_label_result.repair_actions
            if action.get("strategy") == "model_label_imputation"
        ),
        {},
    )
    return {
        "experiment": "synthetic_missing_label_imputation",
        "confidence_threshold": confidence_threshold,
        "synthetic_issues": metadata.get("issues", {}),
        "label_imputation_action": label_action,
        "missing_label_profile": {
            "without_label_imputation": no_label_result.cleaned_profile.get(
                "label_noise",
                {},
            ),
            "with_label_imputation": imputed_label_result.cleaned_profile.get(
                "label_noise",
                {},
            ),
        },
        "label_recovery_against_clean_truth": label_accuracy,
        "ml_metrics": {
            "without_label_imputation": {
                "metrics": policy_ml_comparison.raw_metrics,
            },
            "with_label_imputation": {
                "metrics": policy_ml_comparison.cleaned_metrics,
            },
            "improvement_from_label_imputation": {
                "accuracy": policy_ml_comparison.delta["accuracy"],
                "f1": policy_ml_comparison.delta["f1"],
                "auc": policy_ml_comparison.delta["auc"],
                "ml_row_count": policy_ml_comparison.delta["ml_row_count"],
            },
        },
        "opex_metrics": {
            "without_label_imputation": no_label_result.opex_metrics,
            "with_label_imputation": imputed_label_result.opex_metrics,
        },
        "interpretation": (
            "This experiment evaluates model-based missing-label imputation as "
            "pseudo-labeling. Improvements should be reported as downstream "
            "model and coverage changes, not as guaranteed ground-truth label "
            "recovery unless compared against the synthetic clean labels."
        ),
    }


def print_summary(comparison: dict[str, Any]) -> None:
    improvement = comparison["ml_metrics"]["improvement_from_label_imputation"]
    recovery = comparison["label_recovery_against_clean_truth"]
    action = comparison.get("label_imputation_action", {})

    print("\n=== Synthetic Missing-Label Imputation Experiment ===")
    print(f"Imputed labels: {action.get('imputed_count')}")
    print(f"Candidate missing labels: {action.get('candidate_count')}")
    print("\nML improvement from label imputation:")
    print(f"  accuracy: {improvement['accuracy']}")
    print(f"  f1: {improvement['f1']}")
    print(f"  auc: {improvement['auc']}")
    print(f"  ML row count: {improvement['ml_row_count']}")
    print("\nLabel recovery against synthetic clean labels:")
    print(f"  without: {recovery['without_label_imputation']}")
    print(f"  with: {recovery['with_label_imputation']}")


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("autocleanml-synthetic-label-imputation")
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
