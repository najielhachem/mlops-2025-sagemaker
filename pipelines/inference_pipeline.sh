#!/bin/bash
set -e

SCRIPTS=$(dirname "$0")/../scripts
DATA=$(dirname "$0")/../data
ARTIFACTS=$(dirname "$0")/../artifacts

# Create necessary directories
mkdir -p "$DATA/preprocessed"
mkdir -p "$DATA/features"
mkdir -p "$DATA/predictions"
mkdir -p "$ARTIFACTS/models"

# NOT WORKING
uv run python "$SCRIPTS/preprocess.py" \
  --test_path "$DATA/input/test.csv" \
  --output_test "$DATA/preprocessed/test.csv"

uv run python "$SCRIPTS/featurize.py" \
  --test_path "$DATA/preprocessed/test.csv" \
  --output_test "$DATA/features/test.csv" \
  --output_transformer "$ARTIFACTS/models/transformer.pkl"

uv run python "$SCRIPTS/predict.py" \
  --model "$ARTIFACTS/models/log_reg.pkl" \
  --input "$DATA/features/test.csv" \
  --output "$DATA/predictions/test.csv"
