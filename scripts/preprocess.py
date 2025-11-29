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
    train = pd.read_csv(train_path)
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
    train = df.loc[:890].copy()
    test = df.loc[891:].copy()

    # Remove Survived column from test set
    if "Survived" in test.columns:
        test.drop(columns=["Survived"], inplace=True)

    # Ensure Survived column is int in train set
    if "Survived" in train.columns:
        train["Survived"] = train["Survived"].astype("int64")

    return train, test


def main():
    parser = argparse.ArgumentParser(description="Preprocess Titanic dataset")
    parser.add_argument(
        "--train_path", type=str, required=True, help="Path to training CSV file"
    )
    parser.add_argument(
        "--test_path", type=str, required=True, help="Path to test CSV file"
    )
    parser.add_argument(
        "--output_train",
        type=str,
        required=True,
        help="Output path for preprocessed training data",
    )
    parser.add_argument(
        "--output_test",
        type=str,
        required=True,
        help="Output path for preprocessed test data",
    )

    args = parser.parse_args()

    # Create output directories if they don't exist
    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_test).parent.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train, test = load_data(args.train_path, args.test_path)
    print(f"Loaded train: {train.shape}, test: {test.shape}")

    print("Cleaning data...")
    df = clean_data(train, test)

    print("Splitting data...")
    train_processed, test_processed = split_data(df)

    print("Saving preprocessed data...")
    train_processed.to_csv(args.output_train, index=False)
    test_processed.to_csv(args.output_test, index=False)

    print(f"Preprocessed train saved to: {args.output_train}")
    print(f"Preprocessed test saved to: {args.output_test}")
    print(f"Final train shape: {train_processed.shape}")
    print(f"Final test shape: {test_processed.shape}")


if __name__ == "__main__":
    main()
