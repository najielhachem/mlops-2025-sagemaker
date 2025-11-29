"""
Train a baseline model for Titanic survival prediction.
Loads featurized train data, trains a LogisticRegression, and saves the model for later use.
"""

import argparse
from pathlib import Path
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Train baseline Titanic survival model")
    parser.add_argument("--featurized_train", type=str, required=True, help="Path to featurized train CSV")
    parser.add_argument("--model_output", type=str, required=True, help="Path to save trained model (pickle)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data to use for validation")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load featurized train data
    df = pd.read_csv(args.featurized_train)
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Train baseline Logistic Regression
    model = LogisticRegression(max_iter=1000, random_state=args.random_state)
    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")

    # Save trained model
    Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_output, "wb") as f:
        pickle.dump(model, f)
    print(f"Trained model saved to: {args.model_output}")


if __name__ == "__main__":
    main()

