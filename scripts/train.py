"""
Train a baseline model for Titanic survival prediction.
Loads featurized train/test data, trains a LogisticRegression,
evaluates it, and saves the model for later use.
"""

import argparse
import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

TARGET_COLUMN = "Survived"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train baseline Titanic survival model"
    )

    # Data directories: default to SageMaker channels
    parser.add_argument(
        "--train-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN"),
        help="Directory containing featurized training data (train.csv)",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TEST"),
        help="Directory containing featurized test data (test.csv)",
    )

    # Model output directory: default to SageMaker model dir
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR"),
        help="Directory where the trained model will be saved",
    )

    # Optional hyperparameters
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load featurized train data ---
    train_path = os.path.join(args.train_dir, "train.csv")
    test_path = os.path.join(args.test_dir, "test.csv")

    print(f"Reading train data from: {train_path}")
    print(f"Reading test data from: {test_path}")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train.drop(columns=[TARGET_COLUMN])
    y_train = df_train[TARGET_COLUMN]

    X_test = df_test.drop(columns=[TARGET_COLUMN])
    y_test = df_test[TARGET_COLUMN]

    # --- Train baseline Logistic Regression ---
    model = LogisticRegression(
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    model.fit(X_train, y_train)

    # --- Evaluate on test set ---
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

    # --- Save trained model ---
    model_dir = args.model_dir
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Trained model saved to: {model_path}")


if __name__ == "__main__":
    main()
