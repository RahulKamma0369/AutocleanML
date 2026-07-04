# Databricks notebook source

# COMMAND ----------
# MAGIC %md
# MAGIC # AutoCleanML Thesis Experiments
# MAGIC Run cells top to bottom, one at a time. Wait for each cell to finish before running the next.
# MAGIC The pip install cell triggers an automatic kernel restart — this is normal.

# COMMAND ----------
# MAGIC %sh
# MAGIC rm -rf /tmp/autocleanml
# MAGIC git clone https://github.com/RahulKamma0369/AutocleanML.git /tmp/autocleanml
# MAGIC echo "Clone complete: $(ls /tmp/autocleanml)"

# COMMAND ----------
# MAGIC %pip install -e /tmp/autocleanml

# COMMAND ----------
# MAGIC %sh
# MAGIC mkdir -p /tmp/autocleanml/data/adult
# MAGIC if [ ! -f /tmp/autocleanml/data/adult/adult.data ]; then
# MAGIC   wget -q "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" -O /tmp/autocleanml/data/adult/adult.data
# MAGIC   wget -q "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test" -O /tmp/autocleanml/data/adult/adult.test
# MAGIC   echo "Adult data downloaded"
# MAGIC else
# MAGIC   echo "Adult data already present"
# MAGIC fi

# COMMAND ----------
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
# E1 — Synthetic Employee Attrition (classification)
run_experiment(
    "run_synthetic_classification_experiment.py",
    "--rows", "50000",
    "--log-dir", f"{RESULTS_DIR}/e1",
)

# COMMAND ----------
# E2 — Synthetic House Price (regression)
run_experiment(
    "run_synthetic_regression_experiment.py",
    "--rows", "50000",
    "--log-dir", f"{RESULTS_DIR}/e2",
)

# COMMAND ----------
# E3 — Manual Baseline (UCI Adult)
run_experiment(
    "run_adult_manual_baseline.py",
    "--data-dir", f"{DATA_DIR}/adult",
    "--include-test",
    "--ml-eval",
    "--log-dir", f"{RESULTS_DIR}/e3_manual",
)

# COMMAND ----------
# E3 — UCI Adult Census Income (classification)
run_experiment(
    "run_adult_dataset.py",
    "--data-dir", f"{DATA_DIR}/adult",
    "--include-test",
    "--ml-eval",
    "--manual-baseline-dir", f"{RESULTS_DIR}/e3_manual",
    "--log-dir", f"{RESULTS_DIR}/e3",
)

# COMMAND ----------
# E4 — Manual Baseline (NYC Taxi)
run_experiment(
    "run_nyc_taxi_manual_baseline.py",
    "--data-dir", f"{DATA_DIR}/nyc_taxi",
    "--sample-size", "100000",
    "--ml-eval",
    "--log-dir", f"{RESULTS_DIR}/e4_manual",
)

# COMMAND ----------
# E4 — NYC Yellow Taxi Jan 2023 (regression)
run_experiment(
    "run_nyc_taxi_experiment.py",
    "--data-dir", f"{DATA_DIR}/nyc_taxi",
    "--sample-size", "100000",
    "--ml-eval",
    "--manual-baseline-dir", f"{RESULTS_DIR}/e4_manual",
    "--log-dir", f"{RESULTS_DIR}/e4",
)

# COMMAND ----------
# MAGIC %sh
# MAGIC mkdir -p /dbfs/FileStore/autocleanml_results
# MAGIC cp -r /tmp/autocleanml/results/* /dbfs/FileStore/autocleanml_results/
# MAGIC echo "Results saved to DBFS:"
# MAGIC ls /dbfs/FileStore/autocleanml_results/

# COMMAND ----------
import os
import json

print("=" * 60)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 60)

for exp in ["e1", "e2", "e3", "e4"]:
    exp_dir = f"{RESULTS_DIR}/{exp}"
    if not os.path.exists(exp_dir):
        print(f"\n{exp.upper()}: not found")
        continue
    runs = sorted(os.listdir(exp_dir))
    if not runs:
        print(f"\n{exp.upper()}: no runs found")
        continue
    latest = runs[-1]
    report_path = f"{exp_dir}/{latest}/thesis_report.json"
    if not os.path.exists(report_path):
        print(f"\n{exp.upper()}: thesis_report.json missing in {latest}")
        continue
    with open(report_path) as f:
        report = json.load(f)
    print(f"\n{exp.upper()} — {latest}")
    dq = report.get("data_quality", {})
    print(f"  Missingness reduction : {dq.get('missingness_reduction')}")
    print(f"  Duplicate reduction   : {dq.get('duplicate_reduction')}")
    print(f"  Outlier reduction     : {dq.get('outlier_reduction')}")
    ml = report.get("ml_performance", {})
    if ml:
        for k, v in ml.items():
            print(f"  {k}: {v}")
    opex = report.get("process_efficiency", {})
    if opex:
        print(f"  Code lines (C4)      : {opex.get('dataset_specific_code_lines_baseline')}")
        print(f"  Manual steps avoided : {opex.get('manual_steps_avoided_estimate')}")
