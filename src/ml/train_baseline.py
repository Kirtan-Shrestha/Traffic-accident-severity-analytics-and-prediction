"""
Baseline severity classifier: Logistic Regression with balanced class
weights, establishing the evaluation harness (recall, macro-F1, confusion
matrix) that Phase 3's model comparison will reuse. Everything is logged
to MLflow.
"""

import logging
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, recall_score, f1_score, confusion_matrix,
)

from src.ml.data_prep import load_features, prepare_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("accident_severity_classification")


def evaluate(model, X, y, split_name: str) -> dict:
    y_pred = model.predict(X)
    macro_f1 = f1_score(y, y_pred, average="macro")
    macro_recall = recall_score(y, y_pred, average="macro")
    fatal_recall = recall_score(y, y_pred, labels=["Fatal"], average="macro")

    logger.info(f"\n--- {split_name} ---")
    logger.info(f"\n{classification_report(y, y_pred)}")
    logger.info(f"Confusion matrix:\n{confusion_matrix(y, y_pred, labels=['Fatal', 'Serious', 'Slight'])}")

    return {
        f"{split_name}_macro_f1": macro_f1,
        f"{split_name}_macro_recall": macro_recall,
        f"{split_name}_fatal_recall": fatal_recall,
    }


def main():
    df = load_features()
    splits, feature_cols = prepare_splits(df)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["validation"]
    X_test, y_test = splits["test"]

    with mlflow.start_run(run_name="baseline_logistic_regression"):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_size", len(X_train))

        model = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42,
        )
        model.fit(X_train, y_train)

        train_metrics = evaluate(model, X_train, y_train, "train")
        val_metrics = evaluate(model, X_val, y_val, "validation")

        all_metrics = {**train_metrics, **val_metrics}
        for name, value in all_metrics.items():
            mlflow.log_metric(name, value)

        mlflow.sklearn.log_model(model, "model")

        logger.info(f"\n=== Summary ===")
        for name, value in all_metrics.items():
            logger.info(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()