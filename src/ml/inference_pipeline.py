"""
Packages categorical preprocessing and the trained LightGBM classifier
into a single MLflow pyfunc model, so a raw, un-encoded feature record
(the kind FastAPI will receive from a client) can be scored directly
without re-deriving one-hot encoding logic at serving time.
"""
import mlflow.pyfunc
import pandas as pd

from src.ml.data_prep import CATEGORICAL_COLS

LABEL_MAP = {"Fatal": 0, "Serious": 1, "Slight": 2}
INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


class AccidentSeverityPipeline(mlflow.pyfunc.PythonModel):
    """
    Wraps a trained LightGBM classifier together with the exact
    one-hot encoding vocabulary (feature_cols) it was trained on, so
    inference can run on raw feature records straight from a client
    request rather than requiring the caller to replicate the encoding.
    """

    def __init__(self, model, feature_cols):
        self.model = model
        self.feature_cols = feature_cols

    def _encode(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()
        df["is_weekend"] = df["is_weekend"].astype(int)
        df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
        # Reindex to the exact training-time columns: a category not seen
        # at training time is dropped; a training-time category missing
        # from this input is filled with 0 (one-hot "off").
        df_encoded = df_encoded.reindex(columns=self.feature_cols, fill_value=0)
        bool_cols = df_encoded.select_dtypes(include="bool").columns
        df_encoded[bool_cols] = df_encoded[bool_cols].astype(float)
        return df_encoded.astype(float)

    def predict(self, context, model_input, params=None):
        X = self._encode(model_input)
        preds_numeric = self.model.predict(X)
        return pd.Series(preds_numeric).map(INVERSE_LABEL_MAP).to_numpy()