# AutoCleanML

**AutoCleanML** is a Spark-native prototype for automated data-quality management in big-data machine-learning workflows, developed as part of a Purdue CIT Master's thesis.

The framework profiles six data-quality dimensions, applies configurable rule-driven repair, re-profiles to measure improvement, and produces structured artifacts for thesis evaluation — all without dataset-specific cleaning code.

## Features

- **Distributed profiling** across six dimensions: missingness, duplicates, outliers, key skew, schema drift, and label noise
- **Rule-driven repair** controlled by a single `RepairPolicy` dataclass — the same policy works across all datasets without modification
- **Data-quality evaluation** comparing raw and cleaned profiles to measure reduction in each issue type
- **ML performance comparison** across four conditions: raw (C1), validation-only (C2), AutoCleanML (C3), and manual baseline (C4)
- **OPEX metrics** measuring engineering effort reduction: dataset-specific code lines, manual steps avoided, and issue types handled without custom code
- **Structured artifact logging** producing per-run JSON files for thesis reporting

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.10 and PySpark (see `requirements.txt`).

## Quick Start

```python
from autocleanml import AutoCleanML

result = AutoCleanML().run(
    df,
    key_columns=["customer_id"],
    label_col="label",
)

result.raw_profile       # pre-repair profiler output
result.repair_actions    # list of repair operations applied
result.cleaned_profile   # post-repair profiler output
result.evaluation        # quality reduction metrics
result.opex_metrics      # timing and automation metrics
result.cleaned_df        # repaired Spark DataFrame
```

### Custom repair policy

```python
from autocleanml import AutoCleanML, RepairPolicy

policy = RepairPolicy(
    categorical_imputation="mode",   # fill nulls with column mode instead of "unknown"
    outlier_strategy="cap",          # IQR-cap outliers instead of leaving them
)
result = AutoCleanML(repair_policy=policy).run(df, key_columns=["id"], label_col="label")
```

Column-level outlier overrides:

```python
policy = RepairPolicy(
    outlier_strategy="cap",
    outlier_column_strategies={"age": "cap", "income": "none"},
)
```

Skew mitigation via repartitioning:

```python
policy = RepairPolicy(
    skew_strategy="repartition",
    repair_skew_severities=("high",),
    skew_target_partitions=8,
)
```

## Experiments

Four experiments are included in the thesis evaluation. All scripts support `--ml-eval` to run the ML comparison pipeline and `--log-dir` to save JSON artifacts.

### E1 — Synthetic Employee Attrition (classification)

```bash
python autocleanml/scripts/run_synthetic_classification_experiment.py \
    --row-count 50000 --ml-eval --log-dir experiments/e1
```

### E2 — Synthetic House Price (regression)

```bash
python autocleanml/scripts/run_synthetic_regression_experiment.py \
    --row-count 50000 --ml-eval --log-dir experiments/e2
```

### E3 — UCI Adult Census Income (classification, real-world)

Download data from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/2/adult) and place `adult.data` and `adult.test` in `data/adult/`.

Run the manual baseline first, then the main experiment:

```bash
python autocleanml/scripts/run_adult_manual_baseline.py \
    --data-dir data/adult --include-test --ml-eval --log-dir experiments/e3_manual

python autocleanml/scripts/run_adult_dataset.py \
    --data-dir data/adult --include-test --ml-eval \
    --manual-baseline-dir experiments/e3_manual \
    --log-dir experiments/e3
```

### E4 — NYC Yellow Taxi Jan 2023 (regression, real-world)

The parquet file is downloaded automatically on first run from the NYC TLC public archive.

```bash
python autocleanml/scripts/run_nyc_taxi_manual_baseline.py \
    --data-dir data/nyc_taxi --sample-size 100000 --ml-eval --log-dir experiments/e4_manual

python autocleanml/scripts/run_nyc_taxi_experiment.py \
    --data-dir data/nyc_taxi --sample-size 100000 --ml-eval \
    --manual-baseline-dir experiments/e4_manual \
    --log-dir experiments/e4
```

## Databricks

To run on a Databricks General Purpose cluster:

1. Build the wheel: `python -m build`
2. Upload `dist/autocleanml-0.1.0-py3-none-any.whl` to DBFS at `/FileStore/autocleanml/`
3. Attach `autocleanml/databricks/cluster_init.sh` as a cluster init script
4. Upload the UCI Adult data to `/FileStore/autocleanml/data/adult/`
5. Upload the NYC Taxi parquet to `/FileStore/autocleanml/data/nyc_taxi/`
6. Run `autocleanml/databricks/run_all_experiments.py` as a notebook

All experiment scripts detect an active Databricks `SparkSession` automatically — no local config changes are needed.

## Artifact Output

Each logged run writes a timestamped directory containing:

| File | Contents |
|------|----------|
| `metadata.json` | Dataset parameters and run flags |
| `policy.json` | RepairPolicy configuration |
| `raw_profile.json` | Pre-repair profiler output |
| `repair_actions.json` | List of repair operations applied |
| `cleaned_profile.json` | Post-repair profiler output |
| `evaluation.json` | Quality reduction metrics |
| `opex_metrics.json` | Timing and effort metrics |
| `ml_metrics.json` | Raw/cleaned model metrics with deltas |
| `thesis_report.json` | All metrics grouped by thesis proposal category |
| `manifest.json` | Run ID, timestamp, paths to all artifacts |

## Package Structure

```
autocleanml/
  code/
    pipeline.py          # AutoCleanML entry point
    profiler.py          # DataProfiler (6 dimensions)
    repair.py            # RepairPolicy + RepairEngine
    evaluation.py        # DataQualityEvaluator
    ml_evaluation.py     # SparkML classification + regression evaluators
    opex.py              # OPEX metrics builder
    thesis_evaluation.py # ThesisEvaluationReportBuilder
    synthetic.py         # SyntheticDataGenerator (E1 + E2 datasets)
    pdsa.py              # PDSALoop (supplementary)
  scripts/
    run_synthetic_classification_experiment.py  # E1
    run_synthetic_regression_experiment.py      # E2
    run_adult_dataset.py                        # E3
    run_adult_manual_baseline.py                # E3 manual baseline (C4)
    run_nyc_taxi_experiment.py                  # E4
    run_nyc_taxi_manual_baseline.py             # E4 manual baseline (C4)
  databricks/
    cluster_init.sh          # Cluster init script
    run_all_experiments.py   # Orchestration notebook
```

## Reproducibility

All randomness is controlled by fixed seeds (default: 42). Given the same input data and seed, every experiment produces identical JSON artifact outputs.

## License

MIT
