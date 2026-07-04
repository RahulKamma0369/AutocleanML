#!/bin/bash
set -e

MASTER="spark://autocleanml-master:7077"
SCRIPTS="/opt/autocleanml/autocleanml/scripts"
DATA="/opt/autocleanml/data"
RESULTS="/opt/autocleanml/experiments/cluster_run"

run() {
    local script=$1
    shift
    echo ""
    echo "========================================"
    echo "Running: $script"
    echo "========================================"
    docker exec \
        -e SPARK_MASTER_URL="$MASTER" \
        autocleanml-master \
        python "$SCRIPTS/$script" "$@"
}

echo "Starting AutoCleanML experiments on cluster..."
echo "Master: $MASTER"

# E1 — Synthetic Employee Attrition (classification)
run run_synthetic_classification_experiment.py \
    --rows 50000 \
    --log-dir "$RESULTS/e1"

# E2 — Synthetic House Price (regression)
run run_synthetic_regression_experiment.py \
    --rows 50000 \
    --log-dir "$RESULTS/e2"

# E3 — Manual Baseline (UCI Adult)
run run_adult_manual_baseline.py \
    --data-dir "$DATA/adult" \
    --include-test \
    --ml-eval \
    --log-dir "$RESULTS/e3_manual"

# E3 — UCI Adult Census Income (classification)
run run_adult_dataset.py \
    --data-dir "$DATA/adult" \
    --include-test \
    --ml-eval \
    --manual-baseline-dir "$RESULTS/e3_manual" \
    --log-dir "$RESULTS/e3"

# E4 — Manual Baseline (NYC Taxi)
run run_nyc_taxi_manual_baseline.py \
    --data-dir "$DATA/nyc_taxi" \
    --sample-size 100000 \
    --ml-eval \
    --log-dir "$RESULTS/e4_manual"

# E4 — NYC Yellow Taxi (regression)
run run_nyc_taxi_experiment.py \
    --data-dir "$DATA/nyc_taxi" \
    --sample-size 100000 \
    --ml-eval \
    --manual-baseline-dir "$RESULTS/e4_manual" \
    --log-dir "$RESULTS/e4"

echo ""
echo "All experiments complete. Results in: $RESULTS"
