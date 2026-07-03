#!/bin/bash
# Cluster init script — runs on every node at cluster startup.
# Install the autocleanml package from the wheel uploaded to DBFS.
# Upload the wheel first:
#   databricks fs cp dist/autocleanml-0.1.0-py3-none-any.whl dbfs:/FileStore/autocleanml/

set -e

WHEEL_PATH="/dbfs/FileStore/autocleanml/autocleanml-0.1.0-py3-none-any.whl"

if [ -f "$WHEEL_PATH" ]; then
    pip install --quiet "$WHEEL_PATH"
    echo "autocleanml installed from $WHEEL_PATH"
else
    echo "ERROR: wheel not found at $WHEEL_PATH — upload it first." >&2
    exit 1
fi
