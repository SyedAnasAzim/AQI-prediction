"""
inference.py

Hourly job. For each forecast horizon (24h/48h/72h):
  1. Pulls a 200h raw window from `karachi_aqi` and computes features
     (shared logic with training — see feature_engineering.py).
  2. Picks the best-performing registered model for that horizon by
     comparing stored test_mae across the latest Ridge/RF/NN versions.
  3. Predicts AQI for that horizon.
  4. Computes SHAP values for the single prediction against the
     daily-refreshed background sample (shap_background_karachi_aqi).
  5. Logs the prediction to `aqi_predictions` and the SHAP values to
     `aqi_shap_values`.
  6. Backfills `actual_aqi` on any past prediction rows whose
     target_timestamp is the hour that just became "actual" data.

Assumptions flagged for verification on first real run (can't be tested
without live Hopsworks access):
  - `aqi_predictions` / `aqi_shap_values` are online-enabled feature groups,
    so re-inserting a row with an existing primary key upserts it rather
    than appending a duplicate. This is what the actual_aqi backfill and
    the "overwrite prediction if this hour already ran" case rely on.
  - `model.training_metrics` is populated on Model objects returned by
    `mr.get_models(name=...)`, containing the same dict passed as `metrics=`
    during `create_model()` in model_training.py.
"""

import os
import shutil
import joblib
import numpy as np
import pandas as pd
import shap
import hopsworks

from datetime import datetime, timedelta
from dotenv import load_dotenv

import tensorflow as tf

from feature_engineering import (
    clean_raw,
    compute_features,
    get_feature_sets,
    latest_valid_row,
    HORIZONS,
    HORIZON_TO_LEAD_DAYS,
    HORIZON_TO_HOURS,
    MIN_HISTORY_HOURS,
)

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
RAW_WINDOW_HOURS = 200  # buffer well above MIN_HISTORY_HOURS (72)
ALGOS = ["ridge", "rf", "nn"]

PREDICTIONS_FG_NAME = "aqi_predictions"
PREDICTIONS_FG_VERSION = 1
SHAP_VALUES_FG_NAME = "aqi_shap_values"
SHAP_VALUES_FG_VERSION = 1
SHAP_BACKGROUND_FG_NAME = "shap_background_karachi_aqi"
SHAP_BACKGROUND_FG_VERSION = 1

MODEL_DOWNLOAD_ROOT = "downloaded_models"


# ----------------------------------------------------------------------
# 1. Hopsworks connection helpers
# ----------------------------------------------------------------------
def connect():
    load_dotenv()
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project = hopsworks.login(
        project="PearlsAQI_Project",
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=api_key,
    )
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    return project, fs, mr


def fetch_raw_window(fs, window_hours=RAW_WINDOW_HOURS):
    """Pulls only the trailing `window_hours` of raw data — enough to
    compute a valid current-hour feature row, without re-reading the
    full multi-year history the training job uses."""
    fg = fs.get_feature_group(name="karachi_aqi", version=1)
    cutoff = datetime.utcnow() - timedelta(hours=window_hours)
    # Query API: filter by timestamp then read. If your Hopsworks client
    # version doesn't support fg.filter(...).read() this way, fall back to
    # fg.read() and slice with df[df.timestamp >= cutoff] instead.
    query = fg.select_all().filter(fg.timestamp >= cutoff)
    raw_df = query.read()
    return raw_df, fg


# ----------------------------------------------------------------------
# 2. Model registry: pick the best algo per horizon
# ----------------------------------------------------------------------
def get_latest_model(mr, name):
    """Returns the highest-version registered Model object for `name`."""
    candidates = mr.get_models(name=name)
    if not candidates:
        raise ValueError(f"No registered models found for name='{name}'")
    return max(candidates, key=lambda m: m.version)


def pick_best_model_for_horizon(mr, horizon):
    """
    Compares the latest registered version of aqi_ridge_*, aqi_rf_*,
    aqi_nn_* for this horizon using their stored test_mae metric, and
    returns (best_algo, model_object).
    """
    suffix = horizon.replace("+", "_")
    candidates = {}
    for algo in ALGOS:
        name = f"aqi_{algo}_{suffix}"
        model_obj = get_latest_model(mr, name)
        mae = model_obj.training_metrics.get("test_mae")
        if mae is None:
            raise ValueError(f"Model '{name}' v{model_obj.version} has no test_mae metric.")
        candidates[algo] = (model_obj, mae)

    best_algo = min(candidates, key=lambda a: candidates[a][1])
    best_model_obj, best_mae = candidates[best_algo]
    print(f"[{horizon}] best model: {best_algo} v{best_model_obj.version} (test_mae={best_mae:.3f})")
    return best_algo, best_model_obj


def download_model(model_obj, algo, horizon):
    """Downloads a registered model's artifact dir and loads model (+scaler if present)."""
    suffix = horizon.replace("+", "_")
    local_dir = os.path.join(MODEL_DOWNLOAD_ROOT, f"{algo}_{suffix}_v{model_obj.version}")
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    downloaded_path = model_obj.download()  # returns local path to artifact dir

    scaler = None
    if algo == "nn":
        model = tf.keras.models.load_model(os.path.join(downloaded_path, "model.keras"))
        scaler = joblib.load(os.path.join(downloaded_path, "scaler.pkl"))
    elif algo == "ridge":
        model = joblib.load(os.path.join(downloaded_path, "model.pkl"))
        scaler = joblib.load(os.path.join(downloaded_path, "scaler.pkl"))
    else:  # rf — no scaler
        model = joblib.load(os.path.join(downloaded_path, "model.pkl"))

    return model, scaler


# ----------------------------------------------------------------------
# 3. Build the feature vector for "now"
# ----------------------------------------------------------------------
def build_current_feature_row(raw_df):
    df, _missing = clean_raw(raw_df)  # inference doesn't write missing rows back
    df = compute_features(df)
    tree_cols, linear_cols = get_feature_sets(df)

    all_needed = sorted(set(tree_cols) | set(linear_cols))
    row = latest_valid_row(df, all_needed)
    return row, tree_cols, linear_cols


def prep_model_input(row, algo, tree_cols, linear_cols, scaler):
    """Slices + (for ridge/nn) scales the current row to match training-time input."""
    if algo == "rf":
        return row[tree_cols]

    x = row[linear_cols].copy()
    cyclic_cols = [c for c in linear_cols if c.endswith(("_sin", "_cos"))]
    cols_to_scale = [c for c in linear_cols if c not in cyclic_cols]
    x[cols_to_scale] = scaler.transform(x[cols_to_scale])
    return x


def predict_one(model, algo, X):
    if algo == "nn":
        return float(model.predict(X.values, verbose=0).flatten()[0])
    return float(model.predict(X)[0])


# ----------------------------------------------------------------------
# 4. SHAP for a single prediction
# ----------------------------------------------------------------------
def load_shap_background(fs):
    fg = fs.get_feature_group(name=SHAP_BACKGROUND_FG_NAME, version=SHAP_BACKGROUND_FG_VERSION)
    return fg.read()


def compute_shap_for_prediction(algo, model, X, background_df, tree_cols, linear_cols, scaler):
    """
    Returns a dict {feature_name: shap_value} for the single row X.
    - RF: TreeExplainer (exact, fast, no background needed beyond defaults).
    - Ridge: LinearExplainer with the (scaled) background sample as masker.
    - NN: KernelExplainer with the (scaled) background sample — slower, so
      background is capped small (~150 rows) deliberately.
    """
    if algo == "rf":
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X[tree_cols])
        values = np.array(sv).flatten()
        return dict(zip(tree_cols, values))

    cyclic_cols = [c for c in linear_cols if c.endswith(("_sin", "_cos"))]
    cols_to_scale = [c for c in linear_cols if c not in cyclic_cols]

    bg = background_df[linear_cols].copy()
    bg[cols_to_scale] = scaler.transform(bg[cols_to_scale])

    if algo == "ridge":
        explainer = shap.LinearExplainer(model, bg)
        sv = explainer.shap_values(X[linear_cols])
        values = np.array(sv).flatten()
        return dict(zip(linear_cols, values))

    # nn
    predict_fn = lambda data: model.predict(data, verbose=0).flatten()
    explainer = shap.KernelExplainer(predict_fn, bg.sample(n=min(50, len(bg)), random_state=42))
    sv = explainer.shap_values(X[linear_cols], nsamples=100)
    values = np.array(sv).flatten()
    return dict(zip(linear_cols, values))


# ----------------------------------------------------------------------
# 5. Logging to Hopsworks
# ----------------------------------------------------------------------
def log_prediction(fs, prediction_made_at, target_timestamp, horizon,
                    predicted_aqi, model_used, model_version):
    fg = fs.get_or_create_feature_group(
        name=PREDICTIONS_FG_NAME,
        version=PREDICTIONS_FG_VERSION,
        description="Hourly logged AQI predictions per horizon, with actuals backfilled once available.",
        primary_key=["target_timestamp", "horizon"],
        event_time="target_timestamp",
        online_enabled=True,
    )
    row = pd.DataFrame([{
        "prediction_made_at": prediction_made_at,
        "target_timestamp": target_timestamp,
        "horizon": horizon,
        "lead_time_days": HORIZON_TO_LEAD_DAYS[horizon],
        "predicted_aqi": predicted_aqi,
        "model_used": model_used,
        "model_version": model_version,
        "actual_aqi": np.nan,
    }])
    fg.insert(row, write_options={"wait_for_job": True})
    return fg


def log_shap_values(fs, target_timestamp, horizon, shap_dict):
    fg = fs.get_or_create_feature_group(
        name=SHAP_VALUES_FG_NAME,
        version=SHAP_VALUES_FG_VERSION,
        description="Per-feature SHAP values for each hourly prediction (long format: one row per feature).",
        primary_key=["target_timestamp", "horizon", "feature_name"],
        event_time="target_timestamp",
        online_enabled=True,
    )
    rows = pd.DataFrame([
        {
            "target_timestamp": target_timestamp,
            "horizon": horizon,
            "feature_name": feat,
            "shap_value": float(val),
        }
        for feat, val in shap_dict.items()
    ])
    fg.insert(rows, write_options={"wait_for_job": True})


def backfill_actuals(fs, predictions_fg, raw_df):
    """
    Finds the most recent hour with a real us_aqi reading in the raw window
    and, for any prediction rows whose target_timestamp equals that hour
    and whose actual_aqi is still null, re-inserts them with actual_aqi
    filled in. Relies on upsert-by-primary-key (see module docstring).
    """
    df, _ = clean_raw(raw_df)
    if df.empty:
        return
    latest_actual_ts = df.index.max()
    latest_actual_value = df.loc[latest_actual_ts, "us_aqi"]

    query = predictions_fg.select_all().filter(
        predictions_fg.target_timestamp == latest_actual_ts
    )
    existing = query.read()
    if existing.empty:
        return

    to_update = existing[existing["actual_aqi"].isna()].copy()
    if to_update.empty:
        print(f"No pending predictions to backfill for {latest_actual_ts}.")
        return

    to_update["actual_aqi"] = latest_actual_value
    predictions_fg.insert(to_update, write_options={"wait_for_job": True})
    print(f"Backfilled actual_aqi={latest_actual_value} for {len(to_update)} row(s) at {latest_actual_ts}.")


# ----------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------
def main():
    project, fs, mr = connect()

    print("Fetching raw window...")
    raw_df, raw_fg = fetch_raw_window(fs)

    print("Building current feature row...")
    row, tree_cols, linear_cols = build_current_feature_row(raw_df)
    current_ts = row.index[0]
    prediction_made_at = datetime.utcnow()
    print(f"Feature row timestamp (local Asia/Karachi): {current_ts}")

    background_df = load_shap_background(fs)

    predictions_fg = None
    for horizon in HORIZONS:
        print(f"\n=== {horizon} ===")
        best_algo, model_obj = pick_best_model_for_horizon(mr, horizon)
        model, scaler = download_model(model_obj, best_algo, horizon)

        X = prep_model_input(row, best_algo, tree_cols, linear_cols, scaler)
        predicted_aqi = predict_one(model, best_algo, X)

        target_timestamp = current_ts + pd.Timedelta(hours=HORIZON_TO_HOURS[horizon])
        print(f"Predicted AQI for {target_timestamp}: {predicted_aqi:.1f} (model: {best_algo} v{model_obj.version})")

        predictions_fg = log_prediction(
            fs, prediction_made_at, target_timestamp, horizon,
            predicted_aqi, best_algo, model_obj.version,
        )

        print("Computing SHAP values...")
        shap_dict = compute_shap_for_prediction(
            best_algo, model, X, background_df, tree_cols, linear_cols, scaler,
        )
        log_shap_values(fs, target_timestamp, horizon, shap_dict)

    print("\nBackfilling actuals for the most recently completed hour...")
    backfill_actuals(fs, predictions_fg, raw_df)

    print("\nHourly inference run complete.")


if __name__ == "__main__":
    main()
