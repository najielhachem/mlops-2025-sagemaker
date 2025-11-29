"""
Evaluate a trained Titanic survival model on featurized test data.
Computes accuracy and optionally saves metrics to JSON.
"""

import argparse
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Evaluate Titanic model")
    parser.add_argument("--featurized_test", type=str, required=True, help="Path to featurized test CSV")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model pickle file")
    parser.add_argument("--metrics_output", type=str, required=False, help="Optional path to save metrics JSON")

    args = parser.parse_args()

    # Load featurized test data
    df_test = pd.read_csv(args.featurized_test)

    if "Survived" not in df_test.columns:
        raise ValueError("featurized_test.csv must include the 'Survived' column for evaluation")

    X_test = df_test.drop(columns=["Survived"])
    y_test = df_test["Survived"]

    # Load trained model
    with open(args.model_path, "rb") as f:
        model = pickle.load(f)

    # Predict
    y_pred = model.predict(X_test)

    # Compute accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

    # Save metrics to JSON if requested
    if args.metrics_output:
        metrics = {"accuracy": acc}
        Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_output, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to: {args.metrics_output}")


if __name__ == "__main__":
    main()

