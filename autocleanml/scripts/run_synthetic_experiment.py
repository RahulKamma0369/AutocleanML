from __future__ import annotations

import argparse

from pyspark.sql import SparkSession

from autocleanml import (
    AutoCleanML,
    ExperimentLogger,
    RepairPolicy,
    SyntheticDataGenerator,
    SyntheticIssueConfig,
    ThesisEvaluationReportBuilder,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AutoCleanML on a controlled synthetic dataset."
    )
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing-rate", type=float, default=0.05)
    parser.add_argument("--duplicate-rate", type=float, default=0.02)
    parser.add_argument("--outlier-rate", type=float, default=0.03)
    parser.add_argument("--skew-rate", type=float, default=0.60)
    parser.add_argument("--label-noise-rate", type=float, default=0.05)
    parser.add_argument("--missing-label-rate", type=float, default=0.02)
    parser.add_argument("--no-schema-drift", action="store_true")
    parser.add_argument(
        "--repair-skew",
        action="store_true",
        help="Enable opt-in repartition-based skew repair.",
    )
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
            schema_drift=not args.no_schema_drift,
            label_noise_rate=args.label_noise_rate,
            missing_label_rate=args.missing_label_rate,
        )
        synthetic = SyntheticDataGenerator(spark, config).generate_classification_dataset()

        policy = RepairPolicy(
            skew_strategy="repartition" if args.repair_skew else "none",
            skew_target_partitions=4 if args.repair_skew else None,
        )
        result = AutoCleanML(repair_policy=policy).run(
            synthetic.dirty_df,
            key_columns=["join_key"],
            reference_schema=synthetic.reference_schema,
            label_col="label",
        )

        print_summary(synthetic.metadata, result)
        thesis_report = ThesisEvaluationReportBuilder().build(result)
        if args.log_dir:
            run_dir = ExperimentLogger(args.log_dir).log_run(
                run_name="synthetic",
                result=result,
                policy=policy,
                metadata={
                    "dataset": "synthetic_classification",
                    "synthetic_metadata": synthetic.metadata,
                    "label_col": "label",
                    "key_columns": ["join_key"],
                    "repair_skew": args.repair_skew,
                },
                thesis_report=thesis_report,
            )
            print(f"\nExperiment artifacts written to: {run_dir}")
    finally:
        spark.stop()


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("autocleanml-synthetic")
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )


def print_summary(metadata, result) -> None:
    print("\n=== Synthetic AutoCleanML Experiment ===")
    print("Injected issues:")
    for issue, details in metadata["issues"].items():
        print(f"  {issue}: {details}")

    print("\nDetected and repaired:")
    print(f"  raw rows: {result.raw_profile['row_count']}")
    print(f"  cleaned rows: {result.cleaned_profile['row_count']}")
    print(
        "  missingness reduction "
        f"({result.evaluation['missingness']['scope']}): "
        f"{result.evaluation['missingness']['reduction']}"
    )
    print(
        "  missingness in schema-added columns: "
        f"{result.evaluation['missingness']['added_columns']}"
    )
    print(f"  duplicate reduction: {result.evaluation['duplicates']['reduction']}")
    print(f"  outlier reduction: {result.evaluation['outliers']['reduction']}")
    print(f"  schema drift: {result.evaluation['schema_drift']}")
    print(f"  repair actions: {result.evaluation['repair_actions_by_issue']}")
    print_opex_summary(result)

    print("\nSkew profile:")
    print(f"  raw: {result.raw_profile['skew']}")
    print(f"  cleaned: {result.cleaned_profile['skew']}")

    print("\nLabel profile:")
    print(f"  raw: {result.raw_profile['label_noise']}")
    print(f"  cleaned: {result.cleaned_profile['label_noise']}")

    print_detection_accuracy(metadata, result)


def print_detection_accuracy(metadata, result) -> None:
    print("\n=== Detection Accuracy (injected vs profiler) ===")
    issues = metadata.get("issues", {})
    raw = result.raw_profile

    # Missingness
    if "missingness" in issues:
        injected_rate = issues["missingness"]["rate"]
        for col in issues["missingness"]["columns"]:
            detected = raw["missingness"].get(col, {}).get("missing_ratio", None)
            if detected is not None:
                error = abs(detected - injected_rate)
                accuracy = round((1 - error / injected_rate) * 100, 1) if injected_rate > 0 else None
                print(f"  missingness [{col}]: injected={injected_rate:.3f}  detected={detected:.3f}  accuracy={accuracy}%")

    # Outliers
    if "outliers" in issues:
        injected_rate = issues["outliers"]["rate"]
        col = issues["outliers"]["column"]
        detected = raw["outliers"].get(col, {}).get("outlier_ratio", None)
        if detected is not None:
            error = abs(detected - injected_rate)
            accuracy = round((1 - error / injected_rate) * 100, 1) if injected_rate > 0 else None
            print(f"  outliers    [{col}]: injected={injected_rate:.3f}  detected={detected:.3f}  accuracy={accuracy}%")

    # Duplicates
    if "duplicates" in issues:
        injected_rate = issues["duplicates"]["rate"]
        detected = raw["duplicates"].get("duplicate_ratio", None)
        if detected is not None:
            error = abs(detected - injected_rate)
            accuracy = round((1 - error / injected_rate) * 100, 1) if injected_rate > 0 else None
            print(f"  duplicates:          injected={injected_rate:.3f}  detected={detected:.3f}  accuracy={accuracy}%")

    # Key skew
    if "key_skew" in issues:
        col = issues["key_skew"]["column"]
        detected_severity = raw["skew"].get(col, {}).get("severity", None)
        detected_ratio = raw["skew"].get(col, {}).get("skew_ratio", None)
        print(f"  skew        [{col}]: injected_rate={issues['key_skew']['rate']:.2f}  detected_skew_ratio={detected_ratio}  severity={detected_severity}")

    # Schema drift
    if "schema_drift" in issues:
        drift_report = raw["schema_drift"]
        detected_drift = drift_report.get("drift_detected", False)
        added = drift_report.get("added_columns", [])
        type_changes = drift_report.get("type_changes", [])
        print(f"  schema_drift:        injected=True  detected={detected_drift}  added_cols={added}  type_changes={[c['column'] for c in type_changes]}")

    # Label noise
    if "label_noise" in issues:
        injected_rate = issues["label_noise"]["rate"]
        noise_report = raw["label_noise"].get("confidence_noise", {})
        if noise_report.get("evaluated"):
            detected = noise_report.get("suspected_noise_ratio", None)
            error = abs(detected - injected_rate)
            accuracy = round((1 - error / injected_rate) * 100, 1) if injected_rate > 0 else None
            print(f"  label_noise [label]: injected={injected_rate:.3f}  suspected={detected:.3f}  accuracy={accuracy}%")
        else:
            print(f"  label_noise [label]: injected={injected_rate:.3f}  confidence_scoring={noise_report.get('message')}")


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


if __name__ == "__main__":
    main()
