#!/bin/bash
set -e

SCRIPTS=$(dirname "$0")/../scripts
DATA=$(dirname "$0")/../data
ARTIFACTS=$(dirname "$0")/../artifacts

# Create necessary directories
mkdir -p "$DATA/preprocessed"
mkdir -p "$DATA/features"
mkdir -p "$ARTIFACTS/models"
mkdir -p "$ARTIFACTS/metrics"

echo "Starting training pipeline..."
echo "Preprocessing data..."

uv run python "$SCRIPTS/preprocess.py" \
  --train_path "$DATA/input/train.csv" \
  --test_path "$DATA/input/test.csv" \
  --output_train "$DATA/preprocessed/train.csv" \
  --output_test "$DATA/preprocessed/test.csv"

echo "Featurizing data..."
uv run python "$SCRIPTS/featurize.py" \
  --train_path "$DATA/preprocessed/train.csv" \
  --test_path "$DATA/preprocessed/test.csv" \
  --output_train "$DATA/features/train.csv" \
  --output_test "$DATA/features/test.csv" \
  --output_transformer "$ARTIFACTS/models/transformer.pkl"

echo "Training model..."
uv run python "$SCRIPTS/train.py" \
  --featurized_train "$DATA/features/train.csv" \
  --model_output "$ARTIFACTS/models/log_reg.pkl"
