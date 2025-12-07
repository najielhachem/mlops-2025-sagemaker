"""
Data preprocessing script for Titanic survival prediction.
Handles data loading, cleaning, and basic preprocessing steps.
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

# Ignore all warnings
warnings.filterwarnings("ignore")


def load_data(train_path, test_path):
    """Load training and test datasets."""
    print(f"Loading data from {train_path}")
    train = pd.read_csv(train_path)
    print(f"Loading data from {test_path}")
    test = pd.read_csv(test_path)
    return train, test


def clean_data(train, test):
    """Clean the data by handling missing values and dropping unnecessary columns."""
    # Drop Cabin column due to numerous null values
    train.drop(columns=["Cabin"], inplace=True)
    test.drop(columns=["Cabin"], inplace=True)

    # Fill missing values
    train["Embarked"].fillna("S", inplace=True)
    test["Fare"].fillna(test["Fare"].mean(), inplace=True)

    # Create unified dataframe for easier manipulation
    df = pd.concat([train, test], sort=True).reset_index(drop=True)
    df.corr(numeric_only=True)["Age"].abs()
    # Fill missing Age values using group median
    df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )

    return df


def split_data(df):
    """Split the unified dataframe back into train and test sets."""

    # Ensure Survived column is int in train set
    if "Survived" not in df.columns:
        raise ValueError("Survived column missing from training data")

    df["Survived"] = df["Survived"].astype("int64")

    train = df.loc[:890].copy()
    test = df.loc[891:].copy()

    return train, test


def main():
    parser = argparse.ArgumentParser(description="Preprocess Titanic dataset")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing `train.csv` and `test.csv`",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to write preprocessed `train.csv` and `test.csv`",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Build expected file paths
    train_path = input_dir / "train.csv"
    test_path = input_dir / "test.csv"

    # Ensure input files exist
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    output_train = output_dir / "train.csv"
    output_test = output_dir / "test.csv"

    print("Loading data...")
    train, test = load_data(train_path, test_path)
    print(f"Loaded train: {train.shape}, test: {test.shape}")

    print("Cleaning data...")
    df = clean_data(train, test)

    print("Splitting data...")
    train_processed, test_processed = split_data(df)

    print("Saving preprocessed data...")
    train_processed.to_csv(output_train, index=False)
    test_processed.to_csv(output_test, index=False)

    print(f"Preprocessed train saved to: {output_train}")
    print(f"Preprocessed test saved to: {output_test}")
    print(f"Final train shape: {train_processed.shape}")
    print(f"Final test shape: {test_processed.shape}")


if __name__ == "__main__":
    main()
