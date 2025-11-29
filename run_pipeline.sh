#!/bin/bash
set -e

uv run python scripts/preprocess.py \
  --train_path data/titanic/train.csv \
  --test_path data/titanic/test.csv \
  --output_train data/preprocessed_train.csv \
  --output_test data/preprocessed_test.csv

uv run python scripts/featurize.py \
  --train_path data/preprocessed_train.csv \
  --test_path data/preprocessed_test.csv \
  --output_train data/featurized_train.csv \
  --output_test data/featurized_test.csv \
  --output_transformer models/transformer.pkl

uv run python scripts/train.py \
  --featurized_train data/featurized_train.csv \
  --model_output models/log_reg.pkl

uv run python scripts/evaluate.py \
  --featurized_test data/featurized_train.csv \
  --model_path models/log_reg.pkl \
  --metrics_output metrics/train_eval.json

uv run python scripts/predict.py \
  --model models/log_reg.pkl \
  --input data/featurized_test.csv \
  --output predictions/test_predictions.csv