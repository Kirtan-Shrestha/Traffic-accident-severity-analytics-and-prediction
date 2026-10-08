"""
Drift and performance monitoring report.

Compares the "current" window (the test split: Nov-Dec, the most recent
data) against the "reference" window (the train split: Jan-Aug) that the
production model was trained on. This mimics how a production monitoring
job would compare live traffic against the training distribution.

Computes:
  - Population Stability Index (PSI) for the target class distribution
    (class drift) and for key categorical/location features (feature
    drift), each against a standard interpretation threshold.
  - Per-class recall and false-negative rate for the production model on
    both windows, with particular attention to the Fatal class (the rare,
    safety-critical class).

Logs everything to MLflow as a "drift_monitoring" run so results are
tracked over time, and prints a human-readable report.
"""
import logging

import lightgbm as lgb  # noqa: F401 - import before pandas/mlflow, see train_tree_models.py
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score

from src.ml.data_prep import load_features, get_raw_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("accident_severity_production")

MODEL_NAME = "accident_severity_classifier"
MODEL_ALIAS = "production"

# PSI interpretation thresholds (standard industry convention)
PSI_STABLE = 0.1
PSI_MODERATE = 0.25


def population_stability_index(reference: pd.Series, current: pd.Series, bins=None) -> float:
    """
    PSI = sum((current% - reference%) * ln(current% / reference%)) over bins.
    For a categorical series, each unique category is a bin. For a
    numeric series, `bins` (a list of bin edges) must be supplied.
    """
    if bins is not None:
        ref_binned = pd.cut(reference, bins=bins, include_lowest=True)
        cur_binned = pd.cut(current, bins=bins, include_lowest=True)
    else:
        ref_binned = reference
        cur_binned = current

    ref_dist = ref_binned.value_counts(normalize=True, sort=False)
    cur_dist = cur_binned.value_counts(normalize=True, sort=False)

    # Align on the same set of bins, filling missing bins with a small
    # epsilon to avoid divide-by-zero / log(0).
    all_bins = ref_dist.index.union(cur_dist.index)
    eps = 1e-4
    ref_dist = ref_dist.reindex(all_bins, fill_value=0) + eps
    cur_dist = cur_dist.reindex(all_bins, fill_value=0) + eps

    psi = float(((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)).sum())
    return psi


def interpret_psi(psi: float) -> str:
    if psi < PSI_STABLE:
        return "stable"
    elif psi < PSI_MODERATE:
        return "moderate shift"
    else:
        return "significant shift - review/retrain"


def main():
    df = load_features()
    raw_splits = get_raw_splits(df)
    X_train_raw, y_train_raw = raw_splits["train"]
    X_test_raw, y_test_raw = raw_splits["test"]

    with mlflow.start_run(run_name="drift_monitoring"):
        report_lines = []
        report_lines.append("=== Drift Monitoring Report ===")
        report_lines.append(f"Reference window: train split ({len(X_train_raw)} rows, Jan-Aug)")
        report_lines.append(f"Current window:   test split ({len(X_test_raw)} rows, Nov-Dec)")
        report_lines.append("")

        # --- Class drift: target distribution ---
        class_psi = population_stability_index(y_train_raw, y_test_raw)
        report_lines.append(f"Class drift (target distribution) PSI: {class_psi:.4f} -> {interpret_psi(class_psi)}")
        mlflow.log_metric("psi_class_distribution", class_psi)

        train_class_dist = y_train_raw.value_counts(normalize=True)
        test_class_dist = y_test_raw.value_counts(normalize=True)
        for label in ["Fatal", "Serious", "Slight"]:
            report_lines.append(
                f"  {label}: train={train_class_dist.get(label, 0):.4f}  test={test_class_dist.get(label, 0):.4f}"
            )

        # --- Feature drift: categorical features ---
        report_lines.append("")
        report_lines.append("Feature drift (categorical):")
        for col in ["road_type_label", "weather_conditions_label", "urban_or_rural_label"]:
            psi = population_stability_index(X_train_raw[col], X_test_raw[col])
            report_lines.append(f"  {col}: PSI={psi:.4f} -> {interpret_psi(psi)}")
            mlflow.log_metric(f"psi_{col}", psi)

        # --- Location drift: lat/lon binned into a coarse grid ---
        report_lines.append("")
        report_lines.append("Location drift (lat/lon, 1-degree grid):")
        lat_bins = np.arange(49, 62, 1)
        lon_bins = np.arange(-8, 3, 1)
        lat_psi = population_stability_index(X_train_raw["latitude"], X_test_raw["latitude"], bins=lat_bins)
        lon_psi = population_stability_index(X_train_raw["longitude"], X_test_raw["longitude"], bins=lon_bins)
        report_lines.append(f"  latitude:  PSI={lat_psi:.4f} -> {interpret_psi(lat_psi)}")
        report_lines.append(f"  longitude: PSI={lon_psi:.4f} -> {interpret_psi(lon_psi)}")
        mlflow.log_metric("psi_latitude", lat_psi)
        mlflow.log_metric("psi_longitude", lon_psi)

        # --- Model performance / false-negative rate on each window ---
        report_lines.append("")
        report_lines.append("Model performance (production model):")
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

        for window_name, X_raw, y_true in [("train", X_train_raw, y_train_raw), ("test", X_test_raw, y_test_raw)]:
            y_pred = model.predict(X_raw)
            fatal_recall = recall_score(y_true, y_pred, labels=["Fatal"], average="macro")
            fatal_fnr = 1 - fatal_recall
            cm = confusion_matrix(y_true, y_pred, labels=["Fatal", "Serious", "Slight"])
            report_lines.append(
                f"  {window_name}: Fatal recall={fatal_recall:.4f}  Fatal false-negative rate={fatal_fnr:.4f}"
            )
            report_lines.append(f"    confusion matrix (rows=actual, cols=predicted, order=[Fatal,Serious,Slight]):\n{cm}")
            mlflow.log_metric(f"{window_name}_fatal_recall", fatal_recall)
            mlflow.log_metric(f"{window_name}_fatal_false_negative_rate", fatal_fnr)

        report_text = "\n".join(report_lines)
        logger.info("\n" + report_text)
        mlflow.log_text(report_text, "drift_report.txt")


if __name__ == "__main__":
    main()