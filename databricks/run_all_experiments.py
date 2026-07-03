# Databricks notebook source
# Run all four AutoCleanML thesis experiments on a Databricks general-purpose cluster.
# Attach this notebook to the cluster that has the init script configured.
# Results are written to DBFS under /dbfs/FileStore/autocleanml/results/.

# COMMAND ----------

import sys
import subprocess

RESULTS_DIR = "/dbfs/FileStore/autocleanml/results"
DATA_DIR    = "/dbfs/FileStore/autocleanml/data"
SCRIPTS_DIR = "/Workspace/autocleanml/scripts"   # adjust if repo is in a different Workspace path

def run(script: str, *extra_args: str) -> None:
    cmd = [sys.executable, f"{SCRIPTS_DIR}/{script}", "--log-dir", RESULTS_DIR] + list(extra_args)
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{script} exited with code {result.returncode}")

# COMMAND ----------
# E1 — Synthetic Employee Attrition (classification)

run(
    "run_synthetic_classification_experiment.py",
    "--row-count", "50000",
    "--ml-eval",
)

# COMMAND ----------
# E2 — Synthetic House Price (regression)

run(
    "run_synthetic_regression_experiment.py",
    "--row-count", "50000",
    "--ml-eval",
)

# COMMAND ----------
# E3 — UCI Adult Census Income (classification, real-world)

run(
    "run_adult_dataset.py",
    "--data-dir", f"{DATA_DIR}/adult",
    "--include-test",
    "--ml-eval",
)

# COMMAND ----------
# E4 — NYC Yellow Taxi Trips (regression, real-world)

run(
    "run_nyc_taxi_experiment.py",
    "--data-dir", f"{DATA_DIR}/nyc_taxi",
    "--sample-size", "100000",
    "--ml-eval",
)

# COMMAND ----------
print("\nAll experiments complete. Results written to:", RESULTS_DIR)
