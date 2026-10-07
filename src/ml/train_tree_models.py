"""
Train and compare Random Forest, XGBoost, and LightGBM for accident
severity classification. All three handle class imbalance (via class
weighting or scale_pos_weight equivalents) and log to MLflow using the
same evaluation harness as the baseline for direct comparison.
"""
import lightgbm as lgb  # must import before pandas/sklearn to avoid Windows native DLL conflict
import logging
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import numpy as np

from src.ml.data_prep import load_features, prepare_splits
from src.ml.eval_utils import evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("accident_severity_classification")


def train_random_forest(X_train, y_train, X_val, y_val):
    with mlflow.start_run(run_name="random_forest"):
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 12)

        model = RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train)

        metrics = {**evaluate(model, X_train, y_train, "train"),
                   **evaluate(model, X_val, y_val, "validation")}
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(model, "model")
        return model, metrics


def train_xgboost(X_train, y_train, X_val, y_val):
    label_map = {"Fatal": 0, "Serious": 1, "Slight": 2}
    inv_map = {v: k for k, v in label_map.items()}
    y_train_enc = y_train.map(label_map)
    y_val_enc = y_val.map(label_map)

    # Class weighting for multiclass XGBoost via sample_weight
    classes = np.array(list(label_map.values()))
    weights = compute_class_weight("balanced", classes=classes, y=y_train_enc)
    weight_map = dict(zip(classes, weights))
    sample_weight = y_train_enc.map(weight_map)

    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_param("model_type", "XGBClassifier")
        mlflow.log_param("class_weight", "balanced (sample_weight)")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 6)

        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            objective="multi:softprob", num_class=3,
            random_state=42, n_jobs=-1, eval_metric="mlogloss",
        )
        model.fit(X_train, y_train_enc, sample_weight=sample_weight)

        class Wrapped:
            """Wraps XGBoost to predict original string labels for shared evaluate()."""
            def predict(self, X):
                preds = model.predict(X)
                return np.array([inv_map[p] for p in preds])

        wrapped = Wrapped()
        metrics = {**evaluate(wrapped, X_train, y_train, "train"),
                   **evaluate(wrapped, X_val, y_val, "validation")}
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(model, "model")
        return model, metrics


def train_lightgbm(X_train, y_train, X_val, y_val):
    label_map = {"Fatal": 0, "Serious": 1, "Slight": 2}
    inv_map = {v: k for k, v in label_map.items()}
    y_train_enc = y_train.map(label_map)
    y_val_enc = y_val.map(label_map)

    with mlflow.start_run(run_name="lightgbm"):
        mlflow.log_param("model_type", "LGBMClassifier")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", -1)

        model = lgb.LGBMClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1,
        )
        X_train_np = np.ascontiguousarray(X_train.to_numpy(dtype=np.float64))
        y_train_np = np.ascontiguousarray(y_train_enc.to_numpy(dtype=np.float64))
        model.fit(X_train_np, y_train_np)

        class Wrapped:
            def predict(self, X):
                X_np = X.to_numpy(dtype=np.float64) if hasattr(X, "to_numpy") else X
                preds = model.predict(X_np)
                return np.array([inv_map[p] for p in preds])
        wrapped = Wrapped()
        metrics = {**evaluate(wrapped, X_train, y_train, "train"),
                   **evaluate(wrapped, X_val, y_val, "validation")}
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(model, "model")
        return model, metrics


def main():
    df = load_features()
    splits, feature_cols = prepare_splits(df)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["validation"]

    logger.info("=== Training Random Forest ===")
    _, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)

    logger.info("=== Training XGBoost ===")
    _, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val)

    logger.info("=== Training LightGBM ===")
    _, lgb_metrics = train_lightgbm(X_train, y_train, X_val, y_val)

    logger.info("\n=== Comparison (validation set) ===")
    for name, metrics in [("Random Forest", rf_metrics), ("XGBoost", xgb_metrics), ("LightGBM", lgb_metrics)]:
        logger.info(f"{name}: macro_f1={metrics['validation_macro_f1']:.4f}, "
                     f"macro_recall={metrics['validation_macro_recall']:.4f}, "
                     f"fatal_recall={metrics['validation_fatal_recall']:.4f}")


if __name__ == "__main__":
    main()