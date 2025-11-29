"""
Featurize Titanic dataset for survival prediction.
Applies notebook transformations:
- MinMax scaling + KBinsDiscretizer for Age and Fare
- OneHotEncoding for Sex, Embarked, Parch, Pclass
- OrdinalEncoding for SibSp
Saves transformed train/test and the fitted transformer for later use.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer, OneHotEncoder, OrdinalEncoder
import joblib
import warnings

warnings.filterwarnings("ignore")


def build_feature_pipeline():
    """
    Build the feature transformation pipeline using notebook logic.
    """
    pipeline = ColumnTransformer([
        # Scale + bin Age and Fare
        ("scaled_age_fare", Pipeline([
            ("scaler", MinMaxScaler()),
            ("kbins", KBinsDiscretizer(n_bins=15, encode="ordinal", strategy="quantile"))
        ]), ["Age", "Fare"]),

        # OneHot encode Sex and Embarked
        ("onehot_sex_embarked", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Sex", "Embarked"]),

        # Ordinal encode SibSp
        ("ordinal_sibsp", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["SibSp"]),

        # OneHot encode Parch and Pclass
        ("onehot_parch_pclass", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Parch", "Pclass"])
    ])

    return pipeline


def featurize(train_df, test_df, pipeline_save_path):
    """
    Fit the feature pipeline on train_df and transform train + test.
    Returns transformed X_train, y_train, X_test, and the fitted pipeline.
    """
    # Drop unused columns
    drop_cols = ["PassengerId", "Name", "Ticket"]
    train_df = train_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=drop_cols)

    # Separate target
    y_train = train_df["Survived"].astype(int)
    X_train = train_df.drop(columns=["Survived"])
    X_test = test_df.copy()

    if "Survived" in test_df.columns:
        y_test = X_test["Survived"].astype(int)
        X_test = X_test.drop(columns=["Survived"])
    else:
        y_test = None
        X_test = test_df.copy()

    # Build pipeline
    pipeline = build_feature_pipeline()

    # Fit on train
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    # Save pipeline for later evaluation/inference
    joblib.dump(pipeline, pipeline_save_path)
    print(f"Saved feature pipeline to: {pipeline_save_path}")

    return X_train_transformed, y_train, X_test_transformed,y_test


def main():
    parser = argparse.ArgumentParser(description="Featurize Titanic dataset")
    parser.add_argument("--train_path", type=str, required=True, help="Path to preprocessed train CSV")
    parser.add_argument("--test_path", type=str, required=True, help="Path to preprocessed test CSV")
    parser.add_argument("--output_train", type=str, required=True, help="Path to save featurized train CSV")
    parser.add_argument("--output_test", type=str, required=True, help="Path to save featurized test CSV")
    parser.add_argument("--output_transformer", type=str, required=True, help="Path to save fitted transformer")

    args = parser.parse_args()

    # Load preprocessed CSVs
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    # Featurize
    X_train_transformed, y_train, X_test_transformed, y_test = featurize(
        train_df, test_df, args.output_transformer
    )

    # Save transformed train (with target)
    train_out = pd.DataFrame(X_train_transformed)
    train_out["Survived"] = y_train.values
    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(args.output_train, index=False)
    print(f"Saved featurized train to: {args.output_train}")

    # Save transformed test
    test_out = pd.DataFrame(X_test_transformed)
    if y_test is not None:
        test_out["Survived"] = y_test.values  # Only add Survived if it exists

    Path(args.output_test).parent.mkdir(parents=True, exist_ok=True)
    test_out.to_csv(args.output_test, index=False)
    print(f"Saved featurized test to: {args.output_test}")

    print("Featurization complete.")
    print("Train shape (with target):", train_out.shape)
    print("Test shape:", test_out.shape)


if __name__ == "__main__":
    main()
