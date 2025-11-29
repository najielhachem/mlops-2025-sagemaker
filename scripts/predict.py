# scripts/predict.py
import argparse
import pandas as pd
import pickle
from pathlib import Path

def build_parser():
    p = argparse.ArgumentParser(description="Predict Titanic survival")
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p

def main():
    args = build_parser().parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(args.input)
    preds = model.predict(df)
    output = pd.DataFrame({"Prediction": preds})
    output.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()
