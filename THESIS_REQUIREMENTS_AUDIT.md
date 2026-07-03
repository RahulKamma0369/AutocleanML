# AutoCleanML Thesis Requirements Audit

Generated from the current codebase and experiment artifacts.

## Verdict

AutoCleanML is aligned with the proposal as a Spark-native, data-centric
framework for profiling, rule-driven repair, and evaluation of data quality in
ML preprocessing. The strongest thesis claim is operational: reusable
dataset-agnostic cleaning policies reduce dataset-specific manual cleaning code
and produce repeatable quality, ML, and OPEX artifacts.

The project should not claim universal runtime speedup or guaranteed ML
improvement. Runtime and downstream ML effects are dataset-dependent.

## Proposal Requirement Mapping

| Proposal Requirement | Status | Evidence |
| --- | --- | --- |
| RQ1: Automatically profile and quantify diverse Spark data-quality issues | Aligned | `code/profiler.py` profiles missingness, exact duplicates, per-column and composite-key duplicates, IQR outliers, key skew, schema drift, and label-quality signals. |
| RQ2: Rule-driven, dataset-agnostic repair without dataset-specific code | Aligned | `code/repair.py` implements `RepairPolicy` and generic repairs for missing values, duplicates, outliers, schema alignment, optional missing-label imputation, and opt-in repartition-based skew mitigation. |
| RQ3: Evaluate ML, data quality, and pipeline/OPEX behavior | Aligned | `code/evaluation.py`, `code/ml_evaluation.py`, `code/thesis_evaluation.py`, and `code/opex.py` generate raw-vs-cleaned data-quality, ML, fold-stability, and runtime/action metrics. |
| Classification metrics: Accuracy, F1, AUC | Aligned | Adult and synthetic classification artifacts report Accuracy, F1, and AUC. |
| Regression metrics: RMSE, MAE | Aligned | Synthetic regression and Wine Quality artifacts report RMSE and MAE; R2 is also included. |
| Stability across validation folds | Aligned | Final summary includes cleaned fold stability for all four experiments. |
| Model sensitivity to noisy vs clean data | Aligned | ML artifacts report raw-vs-cleaned deltas and ML row-count changes. |
| Data-quality metrics: missingness, duplicates, skew, outliers, schema | Aligned | Final summary reports missingness reduction, duplicate reduction, outlier reduction, skew columns reduced, and schema issue changes. |
| Process-efficiency metrics: manual steps, code reduction, time | Mostly aligned | Synthetic and Adult have manual baseline step/code evidence. Wine has runtime/action metrics but no manual baseline, so code-line reduction is intentionally blank. |
| PDSA framing | Aligned | `code/pdsa.py` implements a Plan/Profile, Do/Repair, Study/Evaluate, Act/Selective Retry loop. Later cycles rerun only configured iterative issue types, defaulting to outliers and skew to avoid unnecessary repeated missingness/deduplication work. |

## Final Experiment Set

| Exp | Dataset | Type | Task | Current Role |
| --- | --- | --- | --- | --- |
| E1 | Synthetic Full-Quality Classification | Synthetic | Classification | Controlled all-issue detection, repair, ML, and OPEX comparison. |
| E2 | Synthetic Regression | Synthetic | Regression | Controlled regression RMSE/MAE/R2 evidence plus OPEX comparison. |
| E3 | Adult/Census Income | Real popular | Classification | Real classification evidence plus manual baseline/OPEX comparison. |
| E4 | UCI Wine Quality | Real popular | Regression | Real regression and data-quality evidence. |

Latest summary artifact:

`experiments/thesis_summaries/20260624T002453763507Z_thesis_experiment_summary.md`

## Current Limitations To State Clearly

- Wine Quality does not currently have a manual-cleaning baseline, so its
  dataset-specific code-line reduction is not computed.
- AutoCleanML is not consistently faster than manual scripts in the current
  small/local experiments. OPEX should be framed around reduced manual coding,
  reusable policies, standardized artifacts, and reproducibility.
- Label support includes missing-label imputation and confidence-based label
  profiling, but it is not a full noisy-label correction system.
- Skew repair is physical Spark repartitioning; it does not change semantic
  value distribution. PDSA can include skew only when `max_skew_ratio` is set.
- Experiments run in local Spark, consistent with the proposal limitations.

## Verification Commands

```bash
autocleanml/.venv/bin/python -m py_compile autocleanml/__init__.py autocleanml/code/*.py autocleanml/scripts/*.py
autocleanml/.venv/bin/python autocleanml/scripts/summarize_thesis_experiments.py
```

Both commands passed during the audit.
