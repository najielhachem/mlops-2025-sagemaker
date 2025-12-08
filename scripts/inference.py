"""
Batch inference script for Titanic model.

Expected by the BatchInference ProcessingStep:

Inputs:
  --features-dir  : directory with featurized CSV files (from featurize.py)
  --model-dir     : directory containing model.tar.gz
  --output-dir    : directory where predictions.csv will be written

Behavior:
  - finds and extracts model.tar.gz in model_dir
  - loads model.pkl
  - reads all *.csv files in features_dir
  - drops "Survived" if present (to avoid using labels as features)
  - runs model.predict(X)
  - writes predictions.csv to output_dir (original columns + "prediction")
"""

import argparse
import os
import pickle
import tarfile
from pathlib import Path

import pandas as pd

TARGET_COLUMN = "Survived"  # if present, will be dropped before inference


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run batch inference for Titanic model"
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        required=True,
        help="Directory containing featurized input CSV files",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing model.tar.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where predictions.csv will be saved",
    )
    return parser.parse_args()


def find_single_file_with_suffix(directory: str, suffixes):
    """
    Find the first file in `directory` whose name ends with one of `suffixes`.
    Raises a ValueError if none found.
    """
    if isinstance(suffixes, str):
        suffixes = (suffixes,)

    for entry in os.listdir(directory):
        if any(entry.endswith(suf) for suf in suffixes):
            return os.path.join(directory, entry)
    raise ValueError(f"No file with suffix {suffixes} found in directory: {directory}")


def load_model(model_dir: str):
    """
    Locate model.tar.gz in model_dir, extract it, and load model.pkl.
    """
    # 1. Find tarball
    tar_path = find_single_file_with_suffix(model_dir, (".tar.gz", ".tar"))

    print(f"Found model archive: {tar_path}")

    # 2. Extract archive
    extract_dir = os.path.join(model_dir, "extracted")
    Path(extract_dir).mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(path=extract_dir)

    # 3. Find model.pkl (or any *.pkl if name differs)
    try:
        model_path = os.path.join(extract_dir, "model.pkl")
        if not os.path.exists(model_path):
            # fallback: first .pkl we find
            model_path = find_single_file_with_suffix(extract_dir, ".pkl")
    except ValueError as e:
        raise ValueError(
            f"Could not find model.pkl (or any .pkl file) in extracted model dir: {extract_dir}"
        ) from e

    print(f"Loading model from: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


def load_features(features_dir: str) -> pd.DataFrame:
    """
    Read and concatenate all CSV files in features_dir.
    """
    csv_files = [f for f in os.listdir(features_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        raise ValueError(f"No CSV files found in features directory: {features_dir}")

    dfs = []
    for fname in csv_files:
        path = os.path.join(features_dir, fname)
        print(f"Reading features from: {path}")
        df = pd.read_csv(path)
        dfs.append(df)

    if len(dfs) == 1:
        return dfs[0]
    return pd.concat(dfs, ignore_index=True)


def main():
    args = parse_args()

    features_dir = args.features_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load model
    model = load_model(model_dir)

    # 2. Load features
    df = load_features(features_dir)

    # Keep a copy with all columns for output
    df_out = df.copy()

    # 3. Build feature matrix X (drop label if present)
    if TARGET_COLUMN in df.columns:
        print(f"Dropping target column '{TARGET_COLUMN}' from features for inference")
        X = df.drop(columns=[TARGET_COLUMN])
    else:
        X = df

    # 4. Run predictions
    print("Running model.predict on input features...")
    y_pred = model.predict(X)

    # 5. Attach predictions and save
    df_out["prediction"] = y_pred
    output_path = os.path.join(output_dir, "predictions.csv")
    df_out.to_csv(output_path, index=False)

    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
