from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create thesis-ready summary tables across AutoCleanML experiments."
    )
    parser.add_argument("--experiments-dir", default="autocleanml/experiments")
    parser.add_argument(
        "--output-dir",
        default="autocleanml/experiments/thesis_summaries",
    )
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    runs = find_experiment_runs(experiments_dir)
    summary = build_summary(runs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    json_path = output_dir / f"{timestamp}_thesis_experiment_summary.json"
    markdown_path = output_dir / f"{timestamp}_thesis_experiment_summary.md"

    write_json(json_path, summary)
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")

    print("Experiment runs:")
    for key, path in runs.items():
        print(f"  {key}: {path if path is not None else '(not found)'}")
    print(f"Summary JSON: {json_path}")
    print(f"Summary Markdown: {markdown_path}")


def find_experiment_runs(experiments_dir: Path) -> dict[str, Path]:
    return {
        "synthetic_classification": newest_run(
            experiments_dir,
            "_synthetic_full_quality",
        ),
        "synthetic_regression": newest_run(
            experiments_dir,
            "_synthetic_regression",
        ),
        "adult_classification": newest_run(
            experiments_dir,
            "_adult",
        ),
        "adult_manual_baseline": newest_run(
            experiments_dir,
            "_adult_manual_baseline",
        ),
        "wine_quality_regression": newest_run(
            experiments_dir,
            "_wine_quality_regression",
        ),
        "wine_quality_manual_baseline": newest_run_optional(
            experiments_dir,
            "_wine_quality_manual_baseline",
        ),
    }


def newest_run(experiments_dir: Path, suffix: str) -> Path:
    candidates = [
        path
        for path in experiments_dir.iterdir()
        if path.is_dir() and path.name.endswith(suffix)
    ]
    if not candidates:
        raise FileNotFoundError(f"No run ending with {suffix!r} in {experiments_dir}.")
    return sorted(candidates)[-1]


def newest_run_optional(experiments_dir: Path, suffix: str) -> Path | None:
    try:
        return newest_run(experiments_dir, suffix)
    except FileNotFoundError:
        return None


def build_summary(runs: dict[str, Path]) -> dict[str, Any]:
    synthetic_classification = read_json(
        runs["synthetic_classification"] / "comparison_report.json"
    )
    synthetic_regression = read_json(
        runs["synthetic_regression"] / "comparison_report.json"
    )
    synthetic_classification_ml = read_json(
        runs["synthetic_classification"] / "autoclean_ml.json"
    )
    synthetic_regression_ml = read_json(
        runs["synthetic_regression"] / "autoclean_ml.json"
    )
    adult_report = read_json(runs["adult_classification"] / "thesis_report.json")
    adult_ml = read_json(runs["adult_classification"] / "ml_metrics.json")
    adult_manual_process = read_json(
        runs["adult_manual_baseline"] / "manual_process_metrics.json"
    )
    wine_report = read_json(runs["wine_quality_regression"] / "thesis_report.json")
    wine_ml = read_json(runs["wine_quality_regression"] / "ml_metrics.json")
    adult_code_lines_reduced = dataset_specific_code_lines_reduced(
        adult_manual_process
    )
    wine_manual_process = (
        read_json(runs["wine_quality_manual_baseline"] / "manual_process_metrics.json")
        if runs.get("wine_quality_manual_baseline") is not None
        else None
    )
    wine_code_lines_reduced = (
        dataset_specific_code_lines_reduced(wine_manual_process)
        if wine_manual_process is not None
        else None
    )

    def _runtime_delta_vs_manual(ac_opex: dict, manual_opex: dict) -> float | None:
        t_manual = manual_opex.get("total_cycle_time_seconds")
        t_auto = ac_opex.get("total_time_seconds")
        if t_manual is not None and t_auto is not None:
            return round(t_manual - t_auto, 3)
        return None

    def _manual_baseline_steps(manual_opex: dict) -> int | None:
        return manual_opex.get("manual_cleaning_steps")

    detection_rows = [
        detection_accuracy_row(
            "E1",
            "Synthetic Classification",
            synthetic_classification.get("detection_accuracy", {}),
        ),
        detection_accuracy_row(
            "E2",
            "Synthetic Regression",
            synthetic_regression.get("detection_accuracy", {}),
        ),
    ]
    detection_rows = [
        row for row in detection_rows
        if any(
            row.get(key) is not None
            for key in (
                "missingness_accuracy",
                "outlier_accuracy",
                "duplicate_accuracy",
                "schema_drift_detected",
                "label_noise_accuracy",
                "mean_accuracy",
            )
        )
    ]

    return {
        "runs": {
            key: str(path) for key, path in runs.items() if path is not None
        },
        "experiment_overview": [
            {
                "experiment": "E1",
                "name": "Synthetic Full-Quality Classification",
                "dataset_type": "synthetic",
                "task": "classification",
                "purpose": "Controlled all-issue detection, repair, ML impact, and OPEX.",
            },
            {
                "experiment": "E2",
                "name": "Synthetic Regression",
                "dataset_type": "synthetic",
                "task": "regression",
                "purpose": "Controlled regression RMSE/MAE impact under quality issues.",
            },
            {
                "experiment": "E3",
                "name": "Adult/Census Income",
                "dataset_type": "popular real",
                "task": "classification",
                "purpose": "Real classification dataset plus manual baseline/OPEX comparison.",
            },
            {
                "experiment": "E4",
                "name": "UCI Wine Quality",
                "dataset_type": "popular real",
                "task": "regression",
                "purpose": "Real regression dataset for RMSE/MAE evidence.",
            },
        ],
        "data_quality": [
            quality_row(
                "E1",
                "Synthetic Classification",
                synthetic_classification["data_quality"]["autocleanml"],
            ),
            quality_row(
                "E2",
                "Synthetic Regression",
                synthetic_regression["data_quality"]["autocleanml"],
            ),
            quality_row_from_thesis(
                "E3",
                "Adult",
                adult_report["data_quality_metrics"],
            ),
            quality_row_from_thesis(
                "E4",
                "Wine Quality",
                wine_report["data_quality_metrics"],
            ),
        ],
        "ml_metrics": [
            classification_ml_row(
                "E1",
                "Synthetic Classification",
                {
                    "raw": synthetic_classification_ml["raw_metrics"],
                    "cleaned": synthetic_classification_ml["cleaned_metrics"],
                    "delta": synthetic_classification_ml["delta"],
                },
            ),
            regression_ml_row(
                "E2",
                "Synthetic Regression",
                {
                    "raw": synthetic_regression_ml["raw_metrics"],
                    "cleaned": synthetic_regression_ml["cleaned_metrics"],
                    "delta": synthetic_regression_ml["delta"],
                },
            ),
            classification_ml_row(
                "E3",
                "Adult",
                {
                    "raw": adult_ml["raw_metrics"],
                    "cleaned": adult_ml["cleaned_metrics"],
                    "delta": adult_ml["delta"],
                },
            ),
            regression_ml_row_from_thesis(
                "E4",
                "Wine Quality",
                wine_report["ml_metrics"],
                ml_artifact=wine_ml,
            ),
        ],
        "detection_accuracy": detection_rows,
        "opex": [
            opex_row(
                "E1",
                "Synthetic Classification",
                synthetic_classification["opex"]["autocleanml"],
                synthetic_classification["opex"].get("dataset_specific_code_lines_reduced"),
                runtime_delta_vs_manual=_runtime_delta_vs_manual(
                    synthetic_classification["opex"]["autocleanml"],
                    synthetic_classification["opex"]["manual_baseline"],
                ),
                manual_baseline_steps=_manual_baseline_steps(
                    synthetic_classification["opex"]["manual_baseline"],
                ),
            ),
            opex_row(
                "E2",
                "Synthetic Regression",
                synthetic_regression["opex"]["autocleanml"],
                synthetic_regression["opex"].get("dataset_specific_code_lines_reduced"),
                runtime_delta_vs_manual=_runtime_delta_vs_manual(
                    synthetic_regression["opex"]["autocleanml"],
                    synthetic_regression["opex"]["manual_baseline"],
                ),
                manual_baseline_steps=_manual_baseline_steps(
                    synthetic_regression["opex"]["manual_baseline"],
                ),
            ),
            opex_row_from_thesis(
                "E3",
                "Adult",
                adult_report["process_efficiency_metrics"],
                code_lines_reduced=adult_code_lines_reduced,
                manual_baseline_steps=adult_manual_process.get("manual_cleaning_steps"),
            ),
            opex_row_from_thesis(
                "E4",
                "Wine Quality",
                wine_report["process_efficiency_metrics"],
                code_lines_reduced=wine_code_lines_reduced,
                manual_baseline_steps=(
                    wine_manual_process.get("manual_cleaning_steps")
                    if wine_manual_process is not None
                    else None
                ),
            ),
        ],
        "interpretation": {
            "data_quality": (
                "Across the four experiments, AutoCleanML reduced measured "
                "missingness, duplicates, outliers, and/or schema inconsistency "
                "depending on the issues present in each dataset."
            ),
            "ml": (
                "Downstream ML effects were dataset- and metric-dependent: "
                "synthetic classification and synthetic regression improved, "
                "Adult showed mixed classification deltas, and Wine improved R2 "
                "while RMSE/MAE worsened slightly."
            ),
            "opex": (
                "OPEX evidence is strongest for reduced dataset-specific "
                "implementation effort and standardized reproducible artifacts, "
                "not universal runtime speedup."
            ),
        },
    }


def quality_row(experiment: str, name: str, quality: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "dataset": name,
        "missingness_reduction": quality.get("missingness_reduction"),
        "duplicate_reduction": quality.get("duplicate_reduction"),
        "outlier_reduction": quality.get("outlier_reduction"),
        "skew_columns_reduced": quality.get("skew_columns_reduced"),
        "schema_raw_issues": quality.get("schema_raw_issue_count"),
        "schema_cleaned_issues": quality.get("schema_cleaned_issue_count"),
    }


def quality_row_from_thesis(
    experiment: str,
    name: str,
    quality: dict[str, Any],
) -> dict[str, Any]:
    schema = quality.get("schema_consistency_measures", {})
    skew = quality.get("improvement_in_skew_balance", {})
    skew_by_col = skew.get("by_column", {})
    skew_cols_reduced = sum(
        1 for v in skew_by_col.values()
        if v.get("reduction") is not None and v["reduction"] > 0
    ) if skew_by_col else None
    return {
        "experiment": experiment,
        "dataset": name,
        "missingness_reduction": quality.get("change_in_missingness", {}).get(
            "reduction"
        ),
        "duplicate_reduction": quality.get("reduction_in_duplicates", {}).get(
            "reduction"
        ),
        "outlier_reduction": quality.get("outlier_reduction", {}).get("reduction"),
        "skew_columns_reduced": skew_cols_reduced,
        "schema_raw_issues": schema.get("raw_issue_count"),
        "schema_cleaned_issues": schema.get("cleaned_issue_count"),
    }


def classification_ml_row(
    experiment: str,
    name: str,
    ml: dict[str, Any],
) -> dict[str, Any]:
    raw = ml["raw"]
    cleaned = ml["cleaned"]
    delta = ml["delta"]
    return {
        "experiment": experiment,
        "dataset": name,
        "task": "classification",
        "accuracy_raw": raw.get("accuracy"),
        "accuracy_cleaned": cleaned.get("accuracy"),
        "accuracy_delta": delta.get("accuracy"),
        "f1_raw": raw.get("f1"),
        "f1_cleaned": cleaned.get("f1"),
        "f1_delta": delta.get("f1"),
        "auc_raw": raw.get("auc"),
        "auc_cleaned": cleaned.get("auc"),
        "auc_delta": delta.get("auc"),
        "ml_row_delta": delta.get("ml_row_count"),
        "fold_stability_raw": raw.get("fold_stability"),
        "fold_stability_cleaned": cleaned.get("fold_stability"),
    }


def regression_ml_row(
    experiment: str,
    name: str,
    ml: dict[str, Any],
) -> dict[str, Any]:
    raw = ml["raw"]
    cleaned = ml["cleaned"]
    delta = ml["delta"]
    return {
        "experiment": experiment,
        "dataset": name,
        "task": "regression",
        "rmse_raw": raw.get("rmse"),
        "rmse_cleaned": cleaned.get("rmse"),
        "rmse_delta": delta.get("rmse"),
        "mae_raw": raw.get("mae"),
        "mae_cleaned": cleaned.get("mae"),
        "mae_delta": delta.get("mae"),
        "r2_raw": raw.get("r2"),
        "r2_cleaned": cleaned.get("r2"),
        "r2_delta": delta.get("r2"),
        "ml_row_delta": delta.get("ml_row_count"),
        "fold_stability_raw": raw.get("fold_stability"),
        "fold_stability_cleaned": cleaned.get("fold_stability"),
    }


def regression_ml_row_from_thesis(
    experiment: str,
    name: str,
    ml: dict[str, Any],
    ml_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_stability = ml.get("stability_across_validation_folds", {}).get("raw")
    cleaned_stability = ml.get("stability_across_validation_folds", {}).get("cleaned")
    if ml_artifact is not None:
        raw_stability = ml_artifact.get("raw_metrics", {}).get("fold_stability")
        cleaned_stability = ml_artifact.get("cleaned_metrics", {}).get(
            "fold_stability"
        )
    return {
        "experiment": experiment,
        "dataset": name,
        "task": "regression",
        "rmse_raw": ml.get("rmse", {}).get("raw"),
        "rmse_cleaned": ml.get("rmse", {}).get("cleaned"),
        "rmse_delta": ml.get("rmse", {}).get("delta"),
        "mae_raw": ml.get("mae", {}).get("raw"),
        "mae_cleaned": ml.get("mae", {}).get("cleaned"),
        "mae_delta": ml.get("mae", {}).get("delta"),
        "r2_raw": ml.get("r2", {}).get("raw"),
        "r2_cleaned": ml.get("r2", {}).get("cleaned"),
        "r2_delta": ml.get("r2", {}).get("delta"),
        "ml_row_delta": ml.get("model_sensitivity_to_noisy_vs_clean_data", {}).get(
            "ml_row_count"
        ),
        "fold_stability_raw": raw_stability,
        "fold_stability_cleaned": cleaned_stability,
    }


def detection_accuracy_row(
    experiment: str,
    name: str,
    da: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "dataset": name,
        "missingness_accuracy": da.get("missingness", {}).get("mean_accuracy"),
        "outlier_accuracy": da.get("outliers", {}).get("accuracy"),
        "duplicate_accuracy": da.get("duplicates", {}).get("accuracy"),
        "schema_drift_detected": da.get("schema_drift", {}).get("accurate"),
        "label_noise_accuracy": da.get("label_noise", {}).get("accuracy"),
        "mean_accuracy": da.get("mean_accuracy"),
    }


def opex_row(
    experiment: str,
    name: str,
    opex: dict[str, Any],
    code_lines_reduced: int | None,
    runtime_delta_vs_manual: float | None = None,
    manual_baseline_steps: int | None = None,
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "dataset": name,
        "cycle_seconds": opex.get("total_time_seconds"),
        "seconds_per_1000_rows": opex.get("seconds_per_1000_input_rows"),
        "repair_actions": opex.get("repair_action_count"),
        "actions_by_issue": opex.get("repair_actions_by_issue"),
        "dataset_specific_code_lines_reduced": code_lines_reduced,
        "runtime_delta_vs_manual_seconds": runtime_delta_vs_manual,
        "manual_baseline_steps": manual_baseline_steps,
    }


def opex_row_from_thesis(
    experiment: str,
    name: str,
    process: dict[str, Any],
    code_lines_reduced: int | None = None,
    manual_baseline_steps: int | None = None,
) -> dict[str, Any]:
    runtime = process.get("runtime_cost", {})
    code = process.get("reduction_in_dataset_specific_cleaning_code", {})
    time_cycle = process.get("time_saved_per_cleaning_cycle", {})
    steps = process.get("reduction_in_manual_cleaning_steps", {})
    line_reduction = (
        code_lines_reduced
        if code_lines_reduced is not None
        else code.get("estimated_lines_reduced")
    )
    baseline_steps = (
        manual_baseline_steps
        if manual_baseline_steps is not None
        else steps.get("baseline_steps")
    )
    return {
        "experiment": experiment,
        "dataset": name,
        "cycle_seconds": runtime.get("total_time_seconds"),
        "seconds_per_1000_rows": runtime.get("seconds_per_1000_input_rows"),
        "repair_actions": runtime.get("repair_action_count"),
        "actions_by_issue": runtime.get("repair_actions_by_issue"),
        "dataset_specific_code_lines_reduced": line_reduction,
        "runtime_delta_vs_manual_seconds": time_cycle.get("estimated_seconds_saved"),
        "manual_baseline_steps": baseline_steps,
    }


def dataset_specific_code_lines_reduced(
    manual_process: dict[str, Any],
    autocleanml_policy_lines: int = 1,
) -> int | None:
    manual_lines = manual_process.get("dataset_specific_cleaning_code_lines")
    if manual_lines is None:
        return None
    return manual_lines - autocleanml_policy_lines


def render_markdown(summary: dict[str, Any]) -> str:
    sections = [
        "# AutoCleanML Thesis Experiment Summary",
        "",
        "## Experiment Overview",
        "",
        table(
            ["Exp", "Dataset", "Type", "Task", "Purpose"],
            [
                [
                    row["experiment"],
                    row["name"],
                    row["dataset_type"],
                    row["task"],
                    row["purpose"],
                ]
                for row in summary["experiment_overview"]
            ],
        ),
        "",
        "## Data Quality Metrics",
        "",
        table(
            [
                "Exp",
                "Dataset",
                "Missingness Reduced",
                "Duplicates Reduced",
                "Outliers Reduced",
                "Skew Cols Reduced",
                "Schema Issues",
            ],
            [
                [
                    row["experiment"],
                    row["dataset"],
                    row["missingness_reduction"],
                    row["duplicate_reduction"],
                    row["outlier_reduction"],
                    row.get("skew_columns_reduced"),
                    f"{row['schema_raw_issues']} -> {row['schema_cleaned_issues']}",
                ]
                for row in summary["data_quality"]
            ],
        ),
        "",
        *render_detection_accuracy_section(summary),
        "## ML Metrics",
        "",
        render_ml_tables(summary["ml_metrics"]),
        "",
        "## OPEX Metrics",
        "",
        table(
            [
                "Exp",
                "Dataset",
                "Cycle Seconds",
                "Seconds / 1000 Rows",
                "Repair Actions",
                "Code Lines Reduced",
                "Runtime Delta vs Manual (s)",
                "Manual Baseline Steps",
            ],
            [
                [
                    row["experiment"],
                    row["dataset"],
                    row["cycle_seconds"],
                    row["seconds_per_1000_rows"],
                    row["repair_actions"],
                    row["dataset_specific_code_lines_reduced"],
                    row.get("runtime_delta_vs_manual_seconds"),
                    row.get("manual_baseline_steps"),
                ]
                for row in summary["opex"]
            ],
        ),
        "",
        "## Interpretation",
        "",
        f"- Data quality: {summary['interpretation']['data_quality']}",
        f"- ML impact: {summary['interpretation']['ml']}",
        f"- OPEX: {summary['interpretation']['opex']}",
        "",
    ]
    return "\n".join(sections)


def render_detection_accuracy_section(summary: dict[str, Any]) -> list[str]:
    rows = summary.get("detection_accuracy", [])
    if not rows:
        return []
    return [
        "",
        "## Detection Accuracy (Synthetic Experiments Only)",
        "",
        table(
            [
                "Exp",
                "Dataset",
                "Missingness %",
                "Outlier %",
                "Duplicate %",
                "Schema Drift",
                "Label Noise %",
                "Mean %",
            ],
            [
                [
                    row["experiment"],
                    row["dataset"],
                    row.get("missingness_accuracy"),
                    row.get("outlier_accuracy"),
                    row.get("duplicate_accuracy"),
                    row.get("schema_drift_detected"),
                    row.get("label_noise_accuracy"),
                    row.get("mean_accuracy"),
                ]
                for row in rows
            ],
        ),
        "",
    ]


def render_ml_tables(rows: list[dict[str, Any]]) -> str:
    classification_rows = [row for row in rows if row["task"] == "classification"]
    regression_rows = [row for row in rows if row["task"] == "regression"]
    return "\n".join([
        "### Classification",
        "",
        table(
            [
                "Exp",
                "Dataset",
                "Accuracy",
                "F1",
                "AUC",
                "Cleaned Fold Stability",
                "ML Row Delta",
            ],
            [
                [
                    row["experiment"],
                    row["dataset"],
                    delta_cell(row["accuracy_raw"], row["accuracy_cleaned"], row["accuracy_delta"]),
                    delta_cell(row["f1_raw"], row["f1_cleaned"], row["f1_delta"]),
                    delta_cell(row["auc_raw"], row["auc_cleaned"], row["auc_delta"]),
                    stability_cell(row["fold_stability_cleaned"], "accuracy"),
                    row["ml_row_delta"],
                ]
                for row in classification_rows
            ],
        ),
        "",
        "### Regression",
        "",
        table(
            [
                "Exp",
                "Dataset",
                "RMSE",
                "MAE",
                "R2",
                "Cleaned Fold Stability",
                "ML Row Delta",
            ],
            [
                [
                    row["experiment"],
                    row["dataset"],
                    delta_cell(row["rmse_raw"], row["rmse_cleaned"], row["rmse_delta"]),
                    delta_cell(row["mae_raw"], row["mae_cleaned"], row["mae_delta"]),
                    delta_cell(row["r2_raw"], row["r2_cleaned"], row["r2_delta"]),
                    stability_cell(row["fold_stability_cleaned"], "rmse"),
                    row["ml_row_delta"],
                ]
                for row in regression_rows
            ],
        ),
    ])


def delta_cell(raw: Any, cleaned: Any, delta: Any) -> str:
    return f"{fmt(raw)} -> {fmt(cleaned)} ({fmt(delta)})"


def stability_cell(stability: dict[str, Any] | None, metric: str) -> str:
    if not stability:
        return "-"
    metric_summaries = stability.get("summary", stability)
    metric_summary = metric_summaries.get(metric)
    if not metric_summary:
        return "-"
    mean = fmt(metric_summary.get("mean"))
    stddev = fmt(metric_summary.get("stddev"))
    return f"{mean} +/- {stddev}"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(fmt(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
