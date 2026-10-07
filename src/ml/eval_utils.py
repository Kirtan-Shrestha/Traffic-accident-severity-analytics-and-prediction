"""Shared model evaluation utilities, reused across all training scripts."""

import logging
from sklearn.metrics import classification_report, recall_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)

SEVERITY_LABELS = ["Fatal", "Serious", "Slight"]


def evaluate(model, X, y, split_name: str) -> dict:
    y_pred = model.predict(X)
    macro_f1 = f1_score(y, y_pred, average="macro")
    macro_recall = recall_score(y, y_pred, average="macro")
    fatal_recall = recall_score(y, y_pred, labels=["Fatal"], average="macro")

    logger.info(f"\n--- {split_name} ---")
    logger.info(f"\n{classification_report(y, y_pred)}")
    logger.info(f"Confusion matrix (rows=actual, cols=predicted, order={SEVERITY_LABELS}):\n"
                f"{confusion_matrix(y, y_pred, labels=SEVERITY_LABELS)}")

    return {
        f"{split_name}_macro_f1": macro_f1,
        f"{split_name}_macro_recall": macro_recall,
        f"{split_name}_fatal_recall": fatal_recall,
    }