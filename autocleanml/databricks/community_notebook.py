# Databricks notebook source
# AutoCleanML — Community Edition Experiment Runner
# Run each cell in order. Wait for each cell to complete before running the next.
# The pip install cell (Cell 2) triggers an automatic kernel restart — this is normal.

# COMMAND ----------
# MAGIC %md
# MAGIC # AutoCleanML Thesis Experiments
# MAGIC Runs all four experiments (E1–E4) on Databricks Community Edition.
# MAGIC
# MAGIC **Run order:** execute cells top to bottom, one at a time. Do not skip cells.

# COMMAND ----------
# Cell 1 — Clone the repo from GitHub
# MAGIC %sh
# MAGIC rm -rf /tmp/autocleanml
# MAGIC git clone https://github.com/RahulKamma0369/autocleanml.git /tmp/autocleanml
# MAGIC echo "Clone complete: $(ls /tmp/autocleanml)"

# COMMAND ----------
# Cell 2 — Install the package (triggers automatic kernel restart — normal)
# MAGIC %pip install -e /tmp/autocleanml

# COMMAND ----------
# Cell 3 — Download UCI Adult data
# MAGIC %sh
# MAGIC mkdir -p /tmp/autocleanml/data/adult
# MAGIC if [ ! -f /tmp/autocleanml/data/adult/adult.data ]; then
# MAGIC   wget -q "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" \
# MAGIC        -O /tmp/autocleanml/data/adult/adult.data
# MAGIC   wget -q "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test" \
# MAGIC        -O /tmp/autocleanml/data/adult/adult.test
# MAGIC   echo "Adult data downloaded"
# MAGIC else
# MAGIC   echo "Adult data already present"
# MAGIC fi

# COMMAND ----------
# Cell 4 — Setup paths and experiment runner
import sys
import runpy

REPO_DIR    = "/tmp/autocleanml"
SCRIPTS_DIR = f"{REPO_DIR}/autocleanml/scripts"
DATA_DIR    = f"{REPO_DIR}/data"
RESULTS_DIR = f"{REPO_DIR}/results"


def run_experiment(script_name: str, *args: str) -> None:
    script_path = f"{SCRIPTS_DIR}/{script_name}"
    sys.argv = [script_name] + list(args)
    print(f"\n{'='*60}")
    print(f"Starting: {script_name}")
    print(f"Args: {list(args)}")
    print("=" * 60)
    runpy.run_path(script_path, run_name="__main__")
    print(f"\nDone: {script_name}")

# COMMAND ----------
# Cell 5 — E1: Synthetic Employee Attrition (classification)
run_experiment(
    "run_synthetic_classification_experiment.py",
    "--row-count", "50000",
    "--ml-eval",
    "--log-dir", f"{RESULTS_DIR}/e1",
)

# COMMAND ----------
# Cell 6 — E2: Synthetic House Price (regression)
run_experiment(
    "run_synthetic_regression_experiment.py",
    "--row-count", "50000",
    "--ml-eval",
    "--log-dir", f"{RESULTS_DIR}/e2",
)

# COMMAND ----------
# Cell 7 — E3 Manual Baseline (UCI Adult)
run_experiment(
    "run_adult_manual_baseline.py",
    "--data-dir", f"{DATA_DIR}/adult",
    "--include-test",
    "--ml-eval",
    "--log-dir", f"{RESULTS_DIR}/e3_manual",
)

# COMMAND ----------
# Cell 8 — E3: UCI Adult Census Income (classification)
run_experiment(
    "run_adult_dataset.py",
    "--data-dir", f"{DATA_DIR}/adult",
    "--include-test",
    "--ml-eval",
    "--manual-baseline-dir", f"{RESULTS_DIR}/e3_manual",
    "--log-dir", f"{RESULTS_DIR}/e3",
)

# COMMAND ----------
# Cell 9 — E4 Manual Baseline (NYC Taxi — auto-downloads parquet on first run)
run_experiment(
    "run_nyc_taxi_manual_baseline.py",
    "--data-dir", f"{DATA_DIR}/nyc_taxi",
    "--sample-size", "100000",
    "--ml-eval",
    "--log-dir", f"{RESULTS_DIR}/e4_manual",
)

# COMMAND ----------
# Cell 10 — E4: NYC Yellow Taxi Jan 2023 (regression)
run_experiment(
    "run_nyc_taxi_experiment.py",
    "--data-dir", f"{DATA_DIR}/nyc_taxi",
    "--sample-size", "100000",
    "--ml-eval",
    "--manual-baseline-dir", f"{RESULTS_DIR}/e4_manual",
    "--log-dir", f"{RESULTS_DIR}/e4",
)

# COMMAND ----------
# Cell 11 — Copy results to DBFS for persistence across sessions
# MAGIC %sh
# MAGIC mkdir -p /dbfs/FileStore/autocleanml_results
# MAGIC cp -r /tmp/autocleanml/results/* /dbfs/FileStore/autocleanml_results/
# MAGIC echo "Results saved to DBFS:"
# MAGIC ls /dbfs/FileStore/autocleanml_results/

# COMMAND ----------
# Cell 12 — Summary
import os, json

print("=" * 60)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 60)

for exp in ["e1", "e2", "e3", "e4"]:
    exp_dir = f"{RESULTS_DIR}/{exp}"
    if not os.path.exists(exp_dir):
        print(f"{exp}: not found")
        continue
    runs = sorted(os.listdir(exp_dir))
    if not runs:
        print(f"{exp}: no runs")
        continue
    latest = runs[-1]
    report_path = f"{exp_dir}/{latest}/thesis_report.json"
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
        print(f"\n{exp.upper()} ({latest}):")
        dq = report.get("data_quality", {})
        print(f"  Missingness reduction : {dq.get('missingness_reduction')}")
        print(f"  Duplicate reduction   : {dq.get('duplicate_reduction')}")
        print(f"  Outlier reduction     : {dq.get('outlier_reduction')}")
        ml = report.get("ml_performance", {})
        if ml:
            print(f"  ML metrics            : {ml}")
    else:
        print(f"{exp}: thesis_report.json not found in {latest}")
