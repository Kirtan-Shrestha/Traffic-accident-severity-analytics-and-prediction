# Monitoring and Retraining Strategy

**Project:** Traffic Accident Severity Analytics and Prediction (Part 2 — MLOps)
**Author:** Kirtan Shrestha (USN 1CR23AI047)

## 1. Model Lifecycle

1. **Train** — `src/ml/train_tree_models.py` trains Random Forest, XGBoost, and
   LightGBM on the chronological training window (Jan-Aug 2023), with class-weighted
   loss to address severity class imbalance. All runs and metrics are logged to MLflow.
2. **Validate** — Each model is evaluated on the chronological validation window
   (Sep-Oct 2023) using recall, macro-F1, and confusion matrices per class, with
   particular attention to Fatal-class recall given the safety cost of missing a
   fatal-severity prediction.
3. **Package** — The selected model (LightGBM) is wrapped together with its
   preprocessing (categorical encoding) into a single `mlflow.pyfunc.PythonModel`
   (`src/ml/inference_pipeline.py`), so raw feature records can be scored directly
   without re-implementing encoding logic at serving time.
4. **Register** — `src/ml/register_model.py` logs the packaged pipeline to the
   MLflow Model Registry under `accident_severity_classifier` and promotes it with
   the `production` alias.
5. **Serve** — The FastAPI service (`src/api/main.py`) loads
   `models:/accident_severity_classifier@production` at startup and serves
   predictions over `/predict`, containerized via Docker (`Dockerfile.api`,
   `docker-compose.yml`).
6. **Monitor** — `src/monitoring/drift_report.py` runs periodically to compute
   drift and performance-degradation metrics (Section 2), logged to MLflow as a
   `drift_monitoring` run.
7. **Retrain** — When monitoring metrics cross the thresholds in Section 3, the
   pipeline returns to step 1 using an updated training window, and the newly
   registered model version is promoted to the `production` alias once it passes
   validation, replacing the previous version without any serving-code change
   (the API always resolves the alias at startup).

## 2. What Is Monitored

### 2.1 Input / Target Drift (Population Stability Index)

`drift_report.py` computes PSI between the training window (Jan-Aug) and the
test window (Nov-Dec) for:

- **Target distribution** — severity class proportions (Fatal / Serious / Slight)
- **Categorical features** — `road_type_label`, `weather_conditions_label`,
  `urban_or_rural_label`
- **Geographic distribution** — latitude/longitude binned into a 1-degree grid

PSI interpretation (standard thresholds):
| PSI | Interpretation |
|---|---|
| < 0.10 | Stable, no meaningful drift |
| 0.10 - 0.25 | Moderate shift, worth investigating |
| > 0.25 | Significant shift, action required |

**Observed finding:** `weather_conditions_label` PSI between train and test
windows was 0.1378 (moderate shift) — consistent with seasonal weather
differences between the Jan-Aug training period and the Nov-Dec test period
(more rain/fog in winter). Class distribution and location PSI were both
stable (< 0.01).

### 2.2 Model Performance Degradation

For the production model, `drift_report.py` recomputes Fatal-class recall and
false-negative rate on both the training window and the test window.

**Observed finding:** Fatal-class recall was 0.9949 on the training window but
only 0.1107 on the test window — a large gap indicating the production
LightGBM model is likely overfitting on the rare Fatal class and does not
generalize well to later time periods. This is flagged as a priority issue
for the next retraining cycle (e.g., stronger regularization, a higher class
weight ratio, or a return to the better-generalizing class-weighted Logistic
Regression baseline for the Fatal class specifically).

### 2.3 Serving-Time Monitoring

The FastAPI service logs, for every request, via `@app.middleware("http")` in
`src/api/main.py`:
- request path
- response status code (so failures / validation errors are visible)
- latency in milliseconds

These logs (visible via `docker logs traffic_accident_api`) give a baseline for
request volume, error rate, and latency that can be fed into a log-aggregation
or alerting tool in a production deployment.

## 3. Retraining Criteria

Retraining is triggered when any of the following hold, evaluated on a
periodic run of `drift_report.py` against newly collected data:

1. **Drift trigger** — Any monitored feature's PSI exceeds 0.25 (significant
   shift) relative to the current production model's training window.
2. **Performance trigger** — Fatal-class recall on recent data (the latest
   available window) drops more than 15 percentage points below its
   validation-time value, or false-negative rate on the Fatal class exceeds
   0.50.
3. **Schedule trigger** — At minimum, every 6 months, to account for gradual
   seasonal and infrastructure changes even if no single metric has crossed
   its threshold yet.
4. **Operational trigger** — A sustained increase in `/predict` error rate
   (422/500 responses) in the serving logs, which may indicate upstream data
   schema drift rather than model drift.

When a retraining trigger fires, the new model is trained on an updated
window, evaluated against the same validation metrics as the original model,
and only promoted to the `production` alias if it matches or improves on the
current production model's validation macro-F1 and Fatal-class recall. This
keeps the promotion decision tied to measured validation performance rather
than retraining cadence alone.

## 4. Known Limitation

The large train-vs-test Fatal-recall gap documented in Section 2.2 should be
treated as an immediate retraining candidate rather than waiting for the
6-month schedule trigger, since it indicates the current production model
already underperforms on the rare, highest-cost class in later time periods.

## 5. Qualitative Scenario Testing (Sanity Check)

As a sanity check beyond the aggregate metrics in Section 2, five
hand-constructed scenarios were sent to the live `/predict` endpoint,
ranging from low-risk to high-risk conditions:

| Scenario | Conditions | Predicted Severity |
|---|---|---|
| A | Urban, 20 mph, daylight, fine weather, 1 vehicle | Serious |
| B | Rural, 60 mph, rain, evening, 2 vehicles | Slight |
| C | Rural, 70 mph, fog, night, 3 vehicles | Slight |
| D | Rural, snow + high winds, night, 2 vehicles | Slight |
| E | Urban rush hour, rain + high winds, 4 vehicles | Slight |

The model predicted its most severe outcome for the lowest-risk scenario
(A) and predicted the mildest outcome (Slight) for every higher-risk
scenario (B-E), including severe weather, high speed, night-time, and
multi-vehicle conditions. A model with genuine risk sensitivity would be
expected to show the opposite pattern.

This is consistent with the overfitting finding in Section 2.2: the
production model's predictions do not appear to track real-world risk
factors in a generalizable way, likely driven by the severe class
imbalance toward "Slight" severity in the training data causing the
model to default to the majority class except in narrow regions of the
feature space it memorized during training. This reinforces that the
current production model should not be treated as reliable for
risk-based decision-making, and is a priority candidate for retraining
with stronger regularization and/or a different modeling approach (e.g.
the better-generalizing class-weighted Logistic Regression baseline
noted in Section 2.2) before any real-world use.
