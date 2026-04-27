from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "reports"


def find_latest_model() -> Path:
    model_files = sorted(MODELS_DIR.glob("*.joblib"))
    if not model_files:
        raise FileNotFoundError("No saved model found in models/ directory.")
    return model_files[-1]


def load_model_bundle(model_path: Path) -> dict:
    bundle = joblib.load(model_path)
    required_keys = {"model", "feature_names"}
    if not required_keys.issubset(bundle.keys()):
        raise ValueError(f"Invalid model bundle. Missing keys: {required_keys - set(bundle.keys())}")
    return bundle


def align_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    missing_cols = [col for col in feature_names if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in feature_names]

    if missing_cols:
        raise ValueError(f"Input data is missing required columns: {missing_cols}")

    if extra_cols:
        df = df.drop(columns=extra_cols)

    return df[feature_names].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run intrusion detection predictions on new data.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--model", default=None, help="Optional path to a specific model file.")
    parser.add_argument("--output", default=None, help="Optional output CSV path.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    model_path = Path(args.model) if args.model else find_latest_model()
    bundle = load_model_bundle(model_path)

    model = bundle["model"]
    feature_names = bundle["feature_names"]

    df = pd.read_csv(input_path)
    X = align_features(df, feature_names)

    predictions = model.predict(X)

    result_df = df.copy()
    result_df["prediction"] = predictions
    result_df["prediction_label"] = result_df["prediction"].map({0: "BENIGN", 1: "ATTACK"})

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
        result_df["attack_probability"] = probabilities

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "predictions.csv"
    result_df.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()