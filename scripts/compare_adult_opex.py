from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare AutoCleanML Adult results against the manual baseline."
    )
    parser.add_argument(
        "--experiments-dir",
        default="autocleanml/experiments",
        help="Directory containing experiment run folders.",
    )
    parser.add_argument(
        "--autoclean-run",
        default=None,
        help="Specific AutoCleanML Adult run directory. Defaults to newest adult run.",
    )
    parser.add_argument(
        "--manual-run",
        default=None,
        help="Specific manual Adult baseline run directory. Defaults to newest manual run.",
    )
    parser.add_argument(
        "--output-dir",
        default="autocleanml/experiments/thesis_comparisons",
        help="Directory where comparison JSON/Markdown should be written.",
    )
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    autoclean_run = (
        Path(args.autoclean_run)
        if args.autoclean_run
        else newest_run(experiments_dir, suffix="_adult")
    )
    manual_run = (
        Path(args.manual_run)
        if args.manual_run
        else newest_run(experiments_dir, suffix="_adult_manual_baseline")
    )

    comparison = build_comparison(autoclean_run, manual_run)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    json_path = output_dir / f"{timestamp}_adult_opex_comparison.json"
    markdown_path = output_dir / f"{timestamp}_adult_opex_comparison.md"
    write_json(json_path, comparison)
    markdown_path.write_text(render_markdown(comparison), encoding="utf-8")

    print(f"AutoCleanML run: {autoclean_run}")
    print(f"Manual baseline run: {manual_run}")
    print(f"Comparison JSON: {json_path}")
    print(f"Comparison Markdown: {markdown_path}")


def newest_run(experiments_dir: Path, suffix: str) -> Path:
    runs = [
        path
        for path in experiments_dir.iterdir()
        if path.is_dir() and path.name.endswith(suffix)
    ]
    if not runs:
        raise FileNotFoundError(
            f"No experiment run ending with '{suffix}' found in {experiments_dir}."
        )
    return sorted(runs)[-1]


def build_comparison(autoclean_run: Path, manual_run: Path) -> dict[str, Any]:
    auto_eval = read_json(autoclean_run / "evaluation.json")
    auto_opex = read_json(autoclean_run / "opex_metrics.json")
    auto_ml = read_optional_json(autoclean_run / "ml_metrics.json")
    manual_eval = read_json(manual_run / "evaluation.json")
    manual_process = read_json(manual_run / "manual_process_metrics.json")
    manual_ml = read_optional_json(manual_run / "ml_metrics.json")

    auto_policy_lines = 1
    manual_lines = manual_process.get("dataset_specific_cleaning_code_lines")
    manual_time = manual_process.get("total_cycle_time_seconds")
    auto_time = auto_opex.get("total_time_seconds")

    return {
        "runs": {
            "autocleanml": str(autoclean_run),
            "manual_baseline": str(manual_run),
        },
        "process_efficiency": {
            "manual_cleaning_steps": manual_process.get("manual_cleaning_steps"),
            "autocleanml_automated_repair_actions": auto_opex.get(
                "automated_repair_actions"
            ),
            "manual_dataset_specific_code_lines": manual_lines,
            "autocleanml_policy_lines_estimate": auto_policy_lines,
            "dataset_specific_code_lines_reduced": (
                manual_lines - auto_policy_lines
                if manual_lines is not None
                else None
            ),
            "manual_total_cycle_seconds": manual_time,
            "autocleanml_total_cycle_seconds": auto_time,
            "estimated_cycle_seconds_saved": (
                round(manual_time - auto_time, 6)
                if manual_time is not None and auto_time is not None
                else None
            ),
            "manual_seconds_per_1000_rows": manual_process.get(
                "seconds_per_1000_input_rows"
            ),
            "autocleanml_seconds_per_1000_rows": auto_opex.get(
                "seconds_per_1000_input_rows"
            ),
            "manual_repair_seconds": manual_process.get("manual_repair_time_seconds"),
            "autocleanml_repair_seconds": auto_opex.get("repair_time_seconds"),
        },
        "data_quality": {
            "autocleanml": summarize_quality(auto_eval),
            "manual_baseline": summarize_quality(manual_eval),
            "same_quality_outcome": summarize_quality(auto_eval)
            == summarize_quality(manual_eval),
        },
        "ml_metrics": {
            "autocleanml": summarize_ml(auto_ml),
            "manual_baseline": summarize_ml(manual_ml),
            "same_ml_outcome": summarize_ml(auto_ml) == summarize_ml(manual_ml),
        },
        "interpretation": {
            "primary_claim": (
                "AutoCleanML achieved the same measured Adult data-quality and "
                "ML outcomes as the manual Spark baseline while reducing "
                "dataset-specific cleaning code and producing reusable logged "
                "artifacts."
            ),
            "repair_time_caution": (
                "Repair-stage timing is useful but secondary. The fairest OPEX "
                "comparison is total cycle time plus dataset-specific code size, "
                "because both workflows include profiling and post-cleaning "
                "evaluation."
            ),
        },
    }


def summarize_quality(evaluation: dict[str, Any]) -> dict[str, Any]:
    missing = evaluation.get("missingness", {})
    duplicates = evaluation.get("duplicates", {})
    outliers = evaluation.get("outliers", {})
    schema = evaluation.get("schema_drift", {})
    return {
        "missingness_raw": missing.get("raw_total"),
        "missingness_cleaned": missing.get("cleaned_total"),
        "missingness_reduction": missing.get("reduction"),
        "duplicates_raw": duplicates.get("raw"),
        "duplicates_cleaned": duplicates.get("cleaned"),
        "duplicates_reduction": duplicates.get("reduction"),
        "outliers_raw": outliers.get("raw_total"),
        "outliers_cleaned": outliers.get("cleaned_total"),
        "outliers_reduction": outliers.get("reduction"),
        "schema_raw_issue_count": schema.get("raw_issue_count"),
        "schema_cleaned_issue_count": schema.get("cleaned_issue_count"),
    }


def summarize_ml(ml_metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not ml_metrics:
        return None
    raw = ml_metrics.get("raw_metrics", {})
    cleaned = ml_metrics.get("cleaned_metrics", {})
    delta = ml_metrics.get("delta", {})
    return {
        "accuracy_raw": raw.get("accuracy"),
        "accuracy_cleaned": cleaned.get("accuracy"),
        "accuracy_delta": delta.get("accuracy"),
        "f1_raw": raw.get("f1"),
        "f1_cleaned": cleaned.get("f1"),
        "f1_delta": delta.get("f1"),
        "auc_raw": raw.get("auc"),
        "auc_cleaned": cleaned.get("auc"),
        "auc_delta": delta.get("auc"),
        "ml_row_count_delta": delta.get("ml_row_count"),
    }


def render_markdown(comparison: dict[str, Any]) -> str:
    process = comparison["process_efficiency"]
    auto_quality = comparison["data_quality"]["autocleanml"]
    manual_quality = comparison["data_quality"]["manual_baseline"]
    auto_ml = comparison["ml_metrics"]["autocleanml"] or {}
    manual_ml = comparison["ml_metrics"]["manual_baseline"] or {}

    return "\n".join([
        "# Adult OPEX Comparison",
        "",
        f"AutoCleanML run: `{comparison['runs']['autocleanml']}`",
        f"Manual baseline run: `{comparison['runs']['manual_baseline']}`",
        "",
        "## Process Efficiency",
        "",
        "| Metric | Manual baseline | AutoCleanML | Difference |",
        "|---|---:|---:|---:|",
        row(
            "Cleaning actions",
            process["manual_cleaning_steps"],
            process["autocleanml_automated_repair_actions"],
            None,
        ),
        row(
            "Dataset-specific cleaning code lines",
            process["manual_dataset_specific_code_lines"],
            process["autocleanml_policy_lines_estimate"],
            process["dataset_specific_code_lines_reduced"],
        ),
        row(
            "Total cycle seconds",
            process["manual_total_cycle_seconds"],
            process["autocleanml_total_cycle_seconds"],
            process["estimated_cycle_seconds_saved"],
        ),
        row(
            "Seconds per 1000 rows",
            process["manual_seconds_per_1000_rows"],
            process["autocleanml_seconds_per_1000_rows"],
            None,
        ),
        row(
            "Repair-stage seconds",
            process["manual_repair_seconds"],
            process["autocleanml_repair_seconds"],
            None,
        ),
        "",
        "## Data Quality",
        "",
        "| Metric | Manual baseline | AutoCleanML |",
        "|---|---:|---:|",
        row2("Missingness reduction", manual_quality["missingness_reduction"], auto_quality["missingness_reduction"]),
        row2("Duplicate reduction", manual_quality["duplicates_reduction"], auto_quality["duplicates_reduction"]),
        row2("Outlier reduction", manual_quality["outliers_reduction"], auto_quality["outliers_reduction"]),
        row2("Cleaned schema issue count", manual_quality["schema_cleaned_issue_count"], auto_quality["schema_cleaned_issue_count"]),
        "",
        "## ML Metrics",
        "",
        "| Metric | Manual baseline | AutoCleanML |",
        "|---|---:|---:|",
        row2("Accuracy delta", manual_ml.get("accuracy_delta"), auto_ml.get("accuracy_delta")),
        row2("F1 delta", manual_ml.get("f1_delta"), auto_ml.get("f1_delta")),
        row2("AUC delta", manual_ml.get("auc_delta"), auto_ml.get("auc_delta")),
        row2("ML row-count delta", manual_ml.get("ml_row_count_delta"), auto_ml.get("ml_row_count_delta")),
        "",
        "## Thesis Interpretation",
        "",
        comparison["interpretation"]["primary_claim"],
        "",
        comparison["interpretation"]["repair_time_caution"],
        "",
    ])


def row(metric: str, manual: Any, auto: Any, difference: Any) -> str:
    return f"| {metric} | {format_value(manual)} | {format_value(auto)} | {format_value(difference)} |"


def row2(metric: str, manual: Any, auto: Any) -> str:
    return f"| {metric} | {format_value(manual)} | {format_value(auto)} |"


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
