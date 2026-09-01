"""
inference.py

Hourly job. For each forecast horizon (24h/48h/72h):
  1. Pulls a 200h raw window from `karachi_aqi` and computes features.
  2. Picks the best-performing registered model for that horizon.
  3. Predicts AQI for that horizon.
  4. Computes SHAP values for the single prediction.
  5. Logs the prediction and SHAP values.
  6. Backfills actual AQI values for completed predictions.
"""

import os
import shutil
import joblib
import numpy as np
import pandas as pd
import shap
import hopsworks
import time
import requests

from datetime import datetime
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
)


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

RAW_WINDOW_HOURS = 200
ALGOS = ["ridge", "rf", "nn"]

PREDICTIONS_FG_NAME = "aqi_predictions"
PREDICTIONS_FG_VERSION = 1

SHAP_VALUES_FG_NAME = "aqi_shap_values"
SHAP_VALUES_FG_VERSION = 1

SHAP_BACKGROUND_FG_NAME = "shap_background_karachi_aqi"
SHAP_BACKGROUND_FG_VERSION = 1

MODEL_DOWNLOAD_ROOT = "downloaded_models"


# ----------------------------------------------------------------------
# Retry helper
# ----------------------------------------------------------------------

def retry_operation(
    operation,
    name="operation",
    retries=3,
    delay=10,
):
    """
    Retries temporary network-related failures.

    Wait times:
        Attempt 1 fails -> wait 10 seconds
        Attempt 2 fails -> wait 20 seconds
        Attempt 3 fails -> raise error
    """

    for attempt in range(1, retries + 1):

        try:
            print(f"{name} (attempt {attempt}/{retries})...")
            return operation()

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as e:

            print(
                f"{name} failed on attempt "
                f"{attempt}/{retries}: {e}"
            )

            if attempt == retries:
                print(
                    f"{name} failed after "
                    f"{retries} attempts."
                )
                raise

            wait_time = delay * attempt

            print(
                f"Retrying {name} in "
                f"{wait_time} seconds..."
            )

            time.sleep(wait_time)


# ----------------------------------------------------------------------
# 1. Hopsworks connection helpers
# ----------------------------------------------------------------------

def connect():

    load_dotenv()

    api_key = os.getenv("HOPSWORKS_API_KEY")

    project = retry_operation(
        lambda: hopsworks.login(
            project="PearlsAQI_Project",
            host="eu-west.cloud.hopsworks.ai",
            port=443,
            api_key_value=api_key,
        ),
        name="Hopsworks login",
        retries=3,
        delay=10,
    )

    fs = project.get_feature_store()
    mr = project.get_model_registry()

    return project, fs, mr


def fetch_raw_window(
    fs,
    window_hours=RAW_WINDOW_HOURS,
):
    """
    Pulls the raw karachi_aqi feature group and slices
    to the trailing `window_hours` locally in pandas.
    """

    fg = fs.get_feature_group(
        name="karachi_aqi",
        version=1,
    )

    raw_df = retry_operation(
        lambda: fg.read(),
        name="Reading karachi_aqi",
        retries=3,
        delay=10,
    )

    raw_df["timestamp"] = pd.to_datetime(
        raw_df["timestamp"]
    )

    cutoff = (
        raw_df["timestamp"].max()
        - pd.Timedelta(hours=window_hours)
    )

    raw_df = (
        raw_df[
            raw_df["timestamp"] >= cutoff
        ]
        .reset_index(drop=True)
    )

    return raw_df, fg


# ----------------------------------------------------------------------
# 2. Model registry
# ----------------------------------------------------------------------

def get_latest_model(mr, name):
    """
    Returns the highest-version registered Model object.
    """

    candidates = retry_operation(
        lambda: mr.get_models(name=name),
        name=f"Getting models: {name}",
        retries=3,
        delay=10,
    )

    if not candidates:
        raise ValueError(
            f"No registered models found "
            f"for name='{name}'"
        )

    return max(
        candidates,
        key=lambda m: m.version,
    )


def pick_best_model_for_horizon(
    mr,
    horizon,
):
    """
    Compares latest Ridge/RF/NN models using test_mae.
    """

    suffix = horizon.replace("+", "_")

    candidates = {}

    for algo in ALGOS:

        name = f"aqi_{algo}_{suffix}"

        model_obj = get_latest_model(
            mr,
            name,
        )

        mae = model_obj.training_metrics.get(
            "test_mae"
        )

        if mae is None:
            raise ValueError(
                f"Model '{name}' "
                f"v{model_obj.version} "
                f"has no test_mae metric."
            )

        candidates[algo] = (
            model_obj,
            mae,
        )

    best_algo = min(
        candidates,
        key=lambda a: candidates[a][1],
    )

    best_model_obj, best_mae = (
        candidates[best_algo]
    )

    print(
        f"[{horizon}] best model: "
        f"{best_algo} "
        f"v{best_model_obj.version} "
        f"(test_mae={best_mae:.3f})"
    )

    return best_algo, best_model_obj


def download_model(
    model_obj,
    algo,
    horizon,
):
    """
    Downloads a registered model and loads
    the model and scaler if required.
    """

    suffix = horizon.replace("+", "_")

    local_dir = os.path.join(
        MODEL_DOWNLOAD_ROOT,
        f"{algo}_{suffix}_v{model_obj.version}",
    )

    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)

    downloaded_path = retry_operation(
        lambda: model_obj.download(),
        name=(
            f"Downloading {algo} model "
            f"for {horizon}"
        ),
        retries=3,
        delay=10,
    )

    scaler = None

    if algo == "nn":

        model = tf.keras.models.load_model(
            os.path.join(
                downloaded_path,
                "model.keras",
            )
        )

        scaler = joblib.load(
            os.path.join(
                downloaded_path,
                "scaler.pkl",
            )
        )

    elif algo == "ridge":

        model = joblib.load(
            os.path.join(
                downloaded_path,
                "model.pkl",
            )
        )

        scaler = joblib.load(
            os.path.join(
                downloaded_path,
                "scaler.pkl",
            )
        )

    else:

        model = joblib.load(
            os.path.join(
                downloaded_path,
                "model.pkl",
            )
        )

    return model, scaler


# ----------------------------------------------------------------------
# 3. Build feature vector
# ----------------------------------------------------------------------

def build_current_feature_row(raw_df):

    df, _missing = clean_raw(raw_df)

    df = compute_features(df)

    tree_cols, linear_cols = get_feature_sets(
        df
    )

    all_needed = sorted(
        set(tree_cols)
        | set(linear_cols)
    )

    row = latest_valid_row(
        df,
        all_needed,
    )

    return (
        row,
        tree_cols,
        linear_cols,
    )


def prep_model_input(
    row,
    algo,
    tree_cols,
    linear_cols,
    scaler,
):

    if algo == "rf":

        return row[tree_cols]

    x = row[linear_cols].copy()

    cyclic_cols = [
        c
        for c in linear_cols
        if c.endswith(("_sin", "_cos"))
    ]

    cols_to_scale = [
        c
        for c in linear_cols
        if c not in cyclic_cols
    ]

    x[cols_to_scale] = scaler.transform(
        x[cols_to_scale]
    )

    return x


def predict_one(
    model,
    algo,
    X,
):

    if algo == "nn":

        return float(
            model.predict(
                X.values,
                verbose=0,
            )
            .flatten()[0]
        )

    return float(
        model.predict(X)[0]
    )


# ----------------------------------------------------------------------
# 4. SHAP
# ----------------------------------------------------------------------

def load_shap_background(fs):

    fg = fs.get_feature_group(
        name=SHAP_BACKGROUND_FG_NAME,
        version=SHAP_BACKGROUND_FG_VERSION,
    )

    background_df = retry_operation(
        lambda: fg.read(),
        name="Reading SHAP background",
        retries=3,
        delay=10,
    )

    return background_df


def compute_shap_for_prediction(
    algo,
    model,
    X,
    background_df,
    tree_cols,
    linear_cols,
    scaler,
):

    if algo == "rf":

        explainer = shap.TreeExplainer(
            model
        )

        sv = explainer.shap_values(
            X[tree_cols]
        )

        values = np.array(sv).flatten()

        return dict(
            zip(
                tree_cols,
                values,
            )
        )

    cyclic_cols = [
        c
        for c in linear_cols
        if c.endswith(("_sin", "_cos"))
    ]

    cols_to_scale = [
        c
        for c in linear_cols
        if c not in cyclic_cols
    ]

    bg = background_df[
        linear_cols
    ].copy()

    bg[cols_to_scale] = scaler.transform(
        bg[cols_to_scale]
    )

    if algo == "ridge":

        explainer = shap.LinearExplainer(
            model,
            bg,
        )

        sv = explainer.shap_values(
            X[linear_cols]
        )

        values = np.array(sv).flatten()

        return dict(
            zip(
                linear_cols,
                values,
            )
        )

    # Neural Network

    predict_fn = (
        lambda data:
        model.predict(
            data,
            verbose=0,
        ).flatten()
    )

    explainer = shap.KernelExplainer(
        predict_fn,
        bg.sample(
            n=min(50, len(bg)),
            random_state=42,
        ),
    )

    sv = explainer.shap_values(
        X[linear_cols],
        nsamples=100,
    )

    values = np.array(sv).flatten()

    return dict(
        zip(
            linear_cols,
            values,
        )
    )


# ----------------------------------------------------------------------
# 5. Logging to Hopsworks
# ----------------------------------------------------------------------

def log_prediction(
    fs,
    prediction_made_at,
    target_timestamp,
    horizon,
    predicted_aqi,
    model_used,
    model_version,
):

    fg = fs.get_or_create_feature_group(
        name=PREDICTIONS_FG_NAME,
        version=PREDICTIONS_FG_VERSION,
        description=(
            "Hourly logged AQI predictions "
            "per horizon, with actuals "
            "backfilled once available."
        ),
        primary_key=[
            "target_timestamp",
            "horizon",
        ],
        event_time="target_timestamp",
        online_enabled=True,
    )

    row = pd.DataFrame([
        {
            "prediction_made_at":
                prediction_made_at,

            "target_timestamp":
                target_timestamp,

            "horizon":
                horizon,

            "lead_time_days":
                HORIZON_TO_LEAD_DAYS[horizon],

            "predicted_aqi":
                predicted_aqi,

            "model_used":
                model_used,

            "model_version":
                model_version,

            "actual_aqi":
                np.nan,
        }
    ])

    retry_operation(
        lambda: fg.insert(
            row,
            write_options={
                "wait_for_job": True
            },
        ),
        name=(
            f"Inserting prediction "
            f"for {horizon}"
        ),
        retries=3,
        delay=10,
    )

    return fg


def log_shap_values(
    fs,
    target_timestamp,
    horizon,
    shap_dict,
):

    fg = fs.get_or_create_feature_group(
        name=SHAP_VALUES_FG_NAME,
        version=SHAP_VALUES_FG_VERSION,
        description=(
            "Per-feature SHAP values "
            "for each hourly prediction."
        ),
        primary_key=[
            "target_timestamp",
            "horizon",
            "feature_name",
        ],
        event_time="target_timestamp",
        online_enabled=True,
    )

    rows = pd.DataFrame([
        {
            "target_timestamp":
                target_timestamp,

            "horizon":
                horizon,

            "feature_name":
                feat,

            "shap_value":
                float(val),
        }

        for feat, val
        in shap_dict.items()
    ])

    retry_operation(
        lambda: fg.insert(
            rows,
            write_options={
                "wait_for_job": True
            },
        ),
        name=(
            f"Inserting SHAP values "
            f"for {horizon}"
        ),
        retries=3,
        delay=10,
    )


def backfill_actuals(
    fs,
    predictions_fg,
    raw_df,
):

    df, _ = clean_raw(raw_df)

    if df.empty:
        return

    latest_actual_ts = df.index.max()

    latest_actual_value = df.loc[
        latest_actual_ts,
        "us_aqi",
    ]

    existing = retry_operation(
        lambda: predictions_fg.read(),
        name="Reading predictions for backfill",
        retries=3,
        delay=10,
    )

    if existing.empty:
        return

    existing["target_timestamp"] = (
        pd.to_datetime(
            existing["target_timestamp"]
        )
    )

    existing = existing[
        existing["target_timestamp"]
        == latest_actual_ts
    ]

    if existing.empty:
        return

    to_update = existing[
        existing["actual_aqi"].isna()
    ].copy()

    if to_update.empty:

        print(
            f"No pending predictions "
            f"to backfill for "
            f"{latest_actual_ts}."
        )

        return

    to_update["actual_aqi"] = (
        latest_actual_value
    )

    retry_operation(
        lambda: predictions_fg.insert(
            to_update,
            write_options={
                "wait_for_job": True
            },
        ),
        name="Backfilling actual AQI",
        retries=3,
        delay=10,
    )

    print(
        f"Backfilled "
        f"actual_aqi={latest_actual_value} "
        f"for {len(to_update)} row(s) "
        f"at {latest_actual_ts}."
    )


# ----------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------

def main():

    project, fs, mr = connect()

    print("Fetching raw window...")

    raw_df, raw_fg = fetch_raw_window(
        fs
    )

    print(
        "Building current feature row..."
    )

    (
        row,
        tree_cols,
        linear_cols,
    ) = build_current_feature_row(
        raw_df
    )

    current_ts = row.index[0]

    prediction_made_at = (
        datetime.utcnow()
    )

    print(
        f"Feature row timestamp "
        f"(local Asia/Karachi): "
        f"{current_ts}"
    )

    background_df = (
        load_shap_background(fs)
    )

    predictions_fg = None

    for horizon in HORIZONS:

        print(
            f"\n=== {horizon} ==="
        )

        best_algo, model_obj = (
            pick_best_model_for_horizon(
                mr,
                horizon,
            )
        )

        model, scaler = (
            download_model(
                model_obj,
                best_algo,
                horizon,
            )
        )

        X = prep_model_input(
            row,
            best_algo,
            tree_cols,
            linear_cols,
            scaler,
        )

        predicted_aqi = predict_one(
            model,
            best_algo,
            X,
        )

        target_timestamp = (
            current_ts
            + pd.Timedelta(
                hours=
                HORIZON_TO_HOURS[horizon]
            )
        )

        print(
            f"Predicted AQI for "
            f"{target_timestamp}: "
            f"{predicted_aqi:.1f} "
            f"(model: {best_algo} "
            f"v{model_obj.version})"
        )

        predictions_fg = (
            log_prediction(
                fs,
                prediction_made_at,
                target_timestamp,
                horizon,
                predicted_aqi,
                best_algo,
                model_obj.version,
            )
        )

        print(
            "Computing SHAP values..."
        )

        shap_dict = (
            compute_shap_for_prediction(
                best_algo,
                model,
                X,
                background_df,
                tree_cols,
                linear_cols,
                scaler,
            )
        )

        log_shap_values(
            fs,
            target_timestamp,
            horizon,
            shap_dict,
        )

    print(
        "\nBackfilling actuals "
        "for the most recently "
        "completed hour..."
    )

    backfill_actuals(
        fs,
        predictions_fg,
        raw_df,
    )

    print(
        "\nHourly inference run complete."
    )


if __name__ == "__main__":
    main()
