import os
import time
import shutil
import joblib
import numpy as np
import pandas as pd
import hopsworks

from datetime import datetime
from dotenv import load_dotenv

import tensorflow as tf
import shap

from feature_engineering import (
    clean_raw,
    compute_features,
    get_feature_sets,
    latest_valid_row,
    HORIZONS,
    HORIZON_TO_LEAD_DAYS,
    HORIZON_TO_HOURS,
)


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

RAW_WINDOW_HOURS = 200
ALGOS = ["ridge", "rf", "nn"]

SHAP_VALUES_FG_NAME = "aqi_shap_values"
SHAP_VALUES_FG_VERSION = 1

SHAP_BACKGROUND_FG_NAME = "shap_background_karachi_aqi"
SHAP_BACKGROUND_FG_VERSION = 1

MODEL_DOWNLOAD_ROOT = "downloaded_models"
PREDICTIONS_CSV_PATH = "predictions_log.csv"


# ----------------------------------------------------------------------
# Retry helpers
# ----------------------------------------------------------------------

def retry(fn, *, retries=3, delay=10, backoff=2, label="operation"):
    """
    Retry fn() on ANY exception. Hopsworks' Query Service raises its own
    exception types (not just connection/timeout errors), so we don't try
    to enumerate every possible failure class — just retry broadly, with
    exponential backoff.
    """
    last_exc = None
    current_delay = delay
    for attempt in range(1, retries + 1):
        try:
            print(f"[{label}] attempt {attempt}/{retries}...")
            return fn()
        except Exception as e:
            last_exc = e
            print(f"[{label}] attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                print(f"[{label}] retrying in {current_delay}s...")
                time.sleep(current_delay)
                current_delay *= backoff
    raise last_exc


def connect():
    load_dotenv()
    api_key = os.getenv("HOPSWORKS_API_KEY")

    def _login():
        project = hopsworks.login(
            project="PearlsAQI_Project",
            host="eu-west.cloud.hopsworks.ai",
            port=443,
            api_key_value=api_key,
        )
        fs = project.get_feature_store()
        mr = project.get_model_registry()
        return project, fs, mr

    return retry(_login, retries=3, delay=10, backoff=2, label="Hopsworks login")


def read_fg_with_full_retry(get_fg_fn, max_full_retries=3, read_retries=2, label="feature group read"):
    """
    Fully reconnects (fresh login + fresh feature group handle) on each
    outer attempt, not just re-reading on a possibly-stale connection.
    get_fg_fn: function taking `fs` and returning the feature group object.
    """
    for full_attempt in range(1, max_full_retries + 1):
        try:
            project, fs, mr = retry(connect, retries=2, delay=10, backoff=2, label="login (full retry)")
            fg = get_fg_fn(fs)

            df = retry(
                lambda: fg.read(),
                retries=read_retries, delay=15, backoff=2,
                label=f"{label} (fg.read)",
            )

            print(f"[{label}] succeeded on full attempt {full_attempt}/{max_full_retries}")
            return df, fg, fs, mr, project

        except Exception as e:
            print(f"[{label}] full attempt {full_attempt}/{max_full_retries} failed end-to-end: {e}")
            if full_attempt < max_full_retries:
                time.sleep(20)

    raise RuntimeError(f"[{label}] all retries exhausted — login+read failed repeatedly")


# ----------------------------------------------------------------------
# 1. Raw data
# ----------------------------------------------------------------------

def fetch_raw_window(window_hours=RAW_WINDOW_HOURS):
    raw_df, fg, fs, mr, project = read_fg_with_full_retry(
        get_fg_fn=lambda fs: fs.get_feature_group(name="karachi_aqi", version=1),
        max_full_retries=3,
        read_retries=2,
        label="Reading karachi_aqi",
    )

    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    cutoff = raw_df["timestamp"].max() - pd.Timedelta(hours=window_hours)
    raw_df = raw_df[raw_df["timestamp"] >= cutoff].reset_index(drop=True)

    return raw_df, fg, fs, mr, project


# ----------------------------------------------------------------------
# 2. Model registry
# ----------------------------------------------------------------------

def get_latest_model(mr, name):
    candidates = retry(
        lambda: mr.get_models(name=name),
        retries=3, delay=10, backoff=2,
        label=f"Getting models: {name}",
    )

    if not candidates:
        raise ValueError(f"No registered models found for name='{name}'")

    return max(candidates, key=lambda m: m.version)


def pick_best_model_for_horizon(mr, horizon):
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
    suffix = horizon.replace("+", "_")
    local_dir = os.path.join(MODEL_DOWNLOAD_ROOT, f"{algo}_{suffix}_v{model_obj.version}")

    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)

    downloaded_path = retry(
        lambda: model_obj.download(),
        retries=3, delay=10, backoff=2,
        label=f"Downloading {algo} model for {horizon}",
    )

    scaler = None

    if algo == "nn":
        model = tf.keras.models.load_model(os.path.join(downloaded_path, "model.keras"))
        scaler = joblib.load(os.path.join(downloaded_path, "scaler.pkl"))
    elif algo == "ridge":
        model = joblib.load(os.path.join(downloaded_path, "model.pkl"))
        scaler = joblib.load(os.path.join(downloaded_path, "scaler.pkl"))
    else:
        model = joblib.load(os.path.join(downloaded_path, "model.pkl"))

    return model, scaler


# ----------------------------------------------------------------------
# 3. Feature vector
# ----------------------------------------------------------------------

def build_current_feature_row(raw_df):
    df, _missing = clean_raw(raw_df)
    df = compute_features(df)

    tree_cols, linear_cols = get_feature_sets(df)
    all_needed = sorted(set(tree_cols) | set(linear_cols))

    row = latest_valid_row(df, all_needed)

    return row, tree_cols, linear_cols


def prep_model_input(row, algo, tree_cols, linear_cols, scaler):
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
# 4. SHAP
# ----------------------------------------------------------------------

def load_shap_background(fs):
    background_df, fg, fs2, mr, project = read_fg_with_full_retry(
        get_fg_fn=lambda fs: fs.get_feature_group(
            name=SHAP_BACKGROUND_FG_NAME, version=SHAP_BACKGROUND_FG_VERSION
        ),
        max_full_retries=3,
        read_retries=2,
        label="Reading SHAP background",
    )
    return background_df


def compute_shap_for_prediction(algo, model, X, background_df, tree_cols, linear_cols, scaler):
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

    # Neural Network
    predict_fn = lambda data: model.predict(data, verbose=0).flatten()
    explainer = shap.KernelExplainer(predict_fn, bg.sample(n=min(50, len(bg)), random_state=42))
    sv = explainer.shap_values(X[linear_cols], nsamples=100)
    values = np.array(sv).flatten()
    return dict(zip(linear_cols, values))


# ----------------------------------------------------------------------
# 5a. Predictions log — CSV (no longer a Hopsworks feature group)
# ----------------------------------------------------------------------

def log_prediction_csv(prediction_made_at, target_timestamp, horizon,
                        predicted_aqi, model_used, model_version):
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

    if os.path.exists(PREDICTIONS_CSV_PATH):
        row.to_csv(PREDICTIONS_CSV_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(PREDICTIONS_CSV_PATH, mode="w", header=True, index=False)

    print(f"Logged prediction for {horizon} @ {target_timestamp} -> {PREDICTIONS_CSV_PATH}")


def backfill_actuals_csv(raw_df):
    """
    Fills in actual_aqi for any past prediction whose target hour has now
    actually occurred — reads/writes the CSV directly, no Hopsworks call
    needed for this step at all.
    """
    if not os.path.exists(PREDICTIONS_CSV_PATH):
        print("No predictions_log.csv yet — nothing to backfill.")
        return

    df, _ = clean_raw(raw_df)
    if df.empty:
        return

    latest_actual_ts = df.index.max()
    latest_actual_value = df.loc[latest_actual_ts, "us_aqi"]

    log = pd.read_csv(PREDICTIONS_CSV_PATH, parse_dates=["target_timestamp", "prediction_made_at"])

    mask = (log["target_timestamp"] == latest_actual_ts) & (log["actual_aqi"].isna())

    if not mask.any():
        print(f"No pending predictions to backfill for {latest_actual_ts}.")
        return

    log.loc[mask, "actual_aqi"] = latest_actual_value
    log.to_csv(PREDICTIONS_CSV_PATH, index=False)

    print(f"Backfilled actual_aqi={latest_actual_value} for {mask.sum()} row(s) at {latest_actual_ts}.")


# ----------------------------------------------------------------------
# 5b. SHAP values log — stays in Hopsworks (not requested to move)
# ----------------------------------------------------------------------

def log_shap_values(fs, target_timestamp, horizon, shap_dict):
    fg = fs.get_or_create_feature_group(
        name=SHAP_VALUES_FG_NAME,
        version=SHAP_VALUES_FG_VERSION,
        description="Per-feature SHAP values for each hourly prediction.",
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

    retry(
        lambda: fg.insert(rows, write_options={"wait_for_job": True}),
        retries=3, delay=10, backoff=2,
        label=f"Inserting SHAP values for {horizon}",
    )


# ----------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------

def main():
    print("Fetching raw window...")
    raw_df, raw_fg, fs, mr, project = fetch_raw_window()

    print("Building current feature row...")
    row, tree_cols, linear_cols = build_current_feature_row(raw_df)

    current_ts = row.index[0]
    prediction_made_at = datetime.utcnow()

    print(f"Feature row timestamp (local Asia/Karachi): {current_ts}")

    print("Loading SHAP background...")
    background_df = load_shap_background(fs)

    for horizon in HORIZONS:
        print(f"\n=== {horizon} ===")

        best_algo, model_obj = pick_best_model_for_horizon(mr, horizon)
        model, scaler = download_model(model_obj, best_algo, horizon)

        X = prep_model_input(row, best_algo, tree_cols, linear_cols, scaler)
        predicted_aqi = predict_one(model, best_algo, X)

        target_timestamp = current_ts + pd.Timedelta(hours=HORIZON_TO_HOURS[horizon])

        print(f"Predicted AQI for {target_timestamp}: {predicted_aqi:.1f} "
              f"(model: {best_algo} v{model_obj.version})")

        log_prediction_csv(
            prediction_made_at, target_timestamp, horizon,
            predicted_aqi, best_algo, model_obj.version,
        )

        print("Computing SHAP values...")
        shap_dict = compute_shap_for_prediction(
            best_algo, model, X, background_df, tree_cols, linear_cols, scaler,
        )
        log_shap_values(fs, target_timestamp, horizon, shap_dict)

    print("\nBackfilling actuals for the most recently completed hour...")
    backfill_actuals_csv(raw_df)

    print("\nHourly inference run complete.")


if __name__ == "__main__":
    main()
