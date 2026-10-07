"""
Retrains the chosen production model (LightGBM) on the training split,
packages it with preprocessing into a single MLflow pyfunc model, logs
it, and registers it in the MLflow Model Registry under the
"production" alias, so Phase 5 (FastAPI serving) can load it by a
stable reference rather than a specific run ID.
"""
import logging

import lightgbm as lgb  # must import before pandas-dependent modules - see train_tree_models.py
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score

from src.ml.data_prep import load_features, prepare_splits, get_raw_splits
from src.ml.inference_pipeline import AccidentSeverityPipeline, LABEL_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("accident_severity_classification")

MODEL_NAME = "accident_severity_classifier"


def evaluate_pipeline(pipeline, X_raw, y_true, split_name):
    y_pred = pipeline.predict(None, X_raw)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    macro_recall = recall_score(y_true, y_pred, average="macro")
    fatal_recall = recall_score(y_true, y_pred, labels=["Fatal"], average="macro")

    logger.info(f"\n--- {split_name} (via packaged pipeline) ---")
    logger.info(f"\n{classification_report(y_true, y_pred)}")
    logger.info(
        f"Confusion matrix:\n"
        f"{confusion_matrix(y_true, y_pred, labels=['Fatal', 'Serious', 'Slight'])}"
    )
    mlflow.log_metric(f"{split_name}_macro_f1", macro_f1)
    mlflow.log_metric(f"{split_name}_macro_recall", macro_recall)
    mlflow.log_metric(f"{split_name}_fatal_recall", fatal_recall)


def main():
    df = load_features()
    splits, feature_cols = prepare_splits(df)
    X_train, y_train = splits["train"]
    y_train_enc = y_train.map(LABEL_MAP)

    raw_splits = get_raw_splits(df)
    X_train_raw, y_train_raw = raw_splits["train"]
    X_val_raw, y_val_raw = raw_splits["validation"]

    with mlflow.start_run(run_name="production_lightgbm_pipeline") as run:
        mlflow.log_param("model_type", "LightGBM")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("n_features", len(feature_cols))

        model = lgb.LGBMClassifier(
            n_estimators=300, class_weight="balanced", random_state=42,
            n_jobs=1, verbose=-1,
        )
        model.fit(X_train, y_train_enc)

        pipeline = AccidentSeverityPipeline(model=model, feature_cols=feature_cols)

        # Prove the packaged, raw-input pipeline matches training-time
        # performance -- this is what Phase 5's FastAPI service will call.
        evaluate_pipeline(pipeline, X_train_raw, y_train_raw, "train")
        evaluate_pipeline(pipeline, X_val_raw, y_val_raw, "validation")

        mlflow.pyfunc.log_model(
            artifact_path="pipeline",
            python_model=pipeline,
            input_example=X_train_raw.head(3),
            registered_model_name=MODEL_NAME,
            pip_requirements=["lightgbm", "pandas", "scikit-learn", "mlflow==2.17.2", "numpy"],
        )

        run_id = run.info.run_id

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest_version = max(int(v.version) for v in versions if v.run_id == run_id)
    client.set_registered_model_alias(MODEL_NAME, "production", str(latest_version))
    logger.info(f"Registered {MODEL_NAME} version {latest_version} with alias 'production'")


if __name__ == "__main__":
    main()