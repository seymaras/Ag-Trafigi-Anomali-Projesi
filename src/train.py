from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cicids2017_clean.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "is_attack"
DROP_COLS = ["Label", "is_attack"]


def ensure_directories() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = [col for col in df.columns if col not in DROP_COLS]

    if not feature_cols:
        raise ValueError("No feature columns found.")

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    if X.isnull().sum().sum() > 0:
        raise ValueError("Input features contain NaN values.")

    return X, y


def build_models() -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }


def run_cross_validation(model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    f1_scores = cross_val_score(model, X, y, cv=5, scoring="f1", n_jobs=-1)
    recall_scores = cross_val_score(model, X, y, cv=5, scoring="recall", n_jobs=-1)
    roc_auc_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc", n_jobs=-1)

    return {
        "cv_f1_mean": float(f1_scores.mean()),
        "cv_f1_std": float(f1_scores.std()),
        "cv_recall_mean": float(recall_scores.mean()),
        "cv_recall_std": float(recall_scores.std()),
        "cv_roc_auc_mean": float(roc_auc_scores.mean()),
        "cv_roc_auc_std": float(roc_auc_scores.std()),
    }


def save_json(data: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_class_distribution(y: pd.Series, timestamp: str) -> None:
    class_counts = y.value_counts().sort_index()

    class_counts.to_csv(REPORTS_DIR / f"class_distribution_{timestamp}.csv")

    plt.figure()
    class_counts.plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"class_distribution_{timestamp}.png")
    plt.close()


def save_plots(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    timestamp: str,
) -> None:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"{model_name}_confusion_matrix_{timestamp}.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"{model_name}_roc_curve_{timestamp}.png")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure()
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} - Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"{model_name}_pr_curve_{timestamp}.png")
    plt.close()


def save_threshold_analysis(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    timestamp: str,
) -> pd.DataFrame:
    y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    results = []

    for threshold in thresholds:
        y_pred_threshold = (y_prob >= threshold).astype(int)

        results.append(
            {
                "model": model_name,
                "threshold": threshold,
                "precision": precision_score(y_test, y_pred_threshold, zero_division=0),
                "recall": recall_score(y_test, y_pred_threshold, zero_division=0),
                "f1": f1_score(y_test, y_pred_threshold, zero_division=0),
            }
        )

    threshold_df = pd.DataFrame(results)
    threshold_df.to_csv(
        REPORTS_DIR / f"{model_name}_threshold_analysis_{timestamp}.csv",
        index=False,
    )

    plt.figure()
    plt.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall")
    plt.plot(threshold_df["threshold"], threshold_df["f1"], label="F1-score")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"{model_name} - Threshold Analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"{model_name}_threshold_analysis_{timestamp}.png")
    plt.close()

    return threshold_df


def save_feature_importance(
    model: Any,
    feature_names: list[str],
    model_name: str,
    timestamp: str,
) -> None:
    if not hasattr(model, "feature_importances_"):
        return

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    importance_df.to_csv(
        REPORTS_DIR / f"{model_name}_feature_importance_{timestamp}.csv",
        index=False,
    )

    top_features = importance_df.head(20)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 20 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"{model_name}_feature_importance_{timestamp}.png")
    plt.close()


def main() -> None:
    ensure_directories()

    print("Loading processed dataset...")
    df = load_data(DATA_PATH)

    print("Preparing features...")
    X, y = prepare_features(df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Saving class distribution...")
    save_class_distribution(y, timestamp)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = build_models()

    summary: dict[str, Any] = {
        "run_timestamp": timestamp,
        "dataset_path": str(DATA_PATH),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "n_features": int(X.shape[1]),
        "target_column": TARGET_COL,
        "models": {},
    }

    comparison_rows = []

    best_model_name = None
    best_model = None
    best_f1 = -1.0

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)

        print(f"Evaluating {model_name}...")
        metrics = evaluate_model(model, X_test, y_test)

        print(f"Running cross validation for {model_name}...")
        cv_metrics = run_cross_validation(model, X, y)

        metrics.update(cv_metrics)

        print(f"Saving plots for {model_name}...")
        save_plots(model, X_test, y_test, model_name, timestamp)

        print(f"Running threshold analysis for {model_name}...")
        save_threshold_analysis(model, X_test, y_test, model_name, timestamp)

        if model_name == "random_forest":
            print("Saving feature importance...")
            save_feature_importance(
                model=model,
                feature_names=list(X.columns),
                model_name=model_name,
                timestamp=timestamp,
            )

        model_report_path = REPORTS_DIR / f"{model_name}_metrics_{timestamp}.json"
        save_json(metrics, model_report_path)

        summary["models"][model_name] = metrics

        comparison_rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "cv_f1_mean": metrics["cv_f1_mean"],
                "cv_recall_mean": metrics["cv_recall_mean"],
                "cv_roc_auc_mean": metrics["cv_roc_auc_mean"],
            }
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model_name = model_name
            best_model = model

    if best_model is None or best_model_name is None:
        raise RuntimeError("No best model selected.")

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(REPORTS_DIR / f"model_comparison_{timestamp}.csv", index=False)

    model_output_path = MODELS_DIR / f"{best_model_name}_{timestamp}.joblib"
    joblib.dump(
        {
            "model": best_model,
            "feature_names": list(X.columns),
            "target_column": TARGET_COL,
            "best_model_name": best_model_name,
            "timestamp": timestamp,
        },
        model_output_path,
    )

    summary["best_model_name"] = best_model_name
    summary["best_model_path"] = str(model_output_path)

    summary_path = REPORTS_DIR / f"training_summary_{timestamp}.json"
    save_json(summary, summary_path)

    print("\nTraining complete.")
    print(f"Best model: {best_model_name}")
    print(f"Saved model to: {model_output_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved model comparison to: {REPORTS_DIR / f'model_comparison_{timestamp}.csv'}")


if __name__ == "__main__":
    main()