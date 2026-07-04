#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Starting AutoCleanML Spark cluster..."
docker compose up -d

echo "Waiting for cluster to initialise (15s)..."
sleep 15

echo "Installing AutoCleanML on all nodes..."
docker exec autocleanml-master  pip install -e /opt/autocleanml --quiet
docker exec autocleanml-worker-1 pip install -e /opt/autocleanml --quiet
docker exec autocleanml-worker-2 pip install -e /opt/autocleanml --quiet

echo ""
echo "Cluster ready."
echo "  Spark Master UI : http://localhost:8080"
echo "  Worker 1 UI     : http://localhost:8081"
echo "  Worker 2 UI     : http://localhost:8082"
echo "  App UI (live)   : http://localhost:4040"
echo ""
echo "Run experiments with:"
echo "  ./docker/run_experiments.sh"
