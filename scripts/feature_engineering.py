"""
feature_engineering.py

Shared feature engineering logic used by BOTH the training pipeline
(model_training.py) and the hourly inference pipeline (inference.py).

This module exists to solve one specific problem: the original preprocess()
in model_training.py dropped every row with a NaN in the target columns
(aqi_t+24h/48h/72h). That's correct for training, but the most recent ~72
hours ALWAYS have NaN targets (the future hasn't happened yet) — so running
that same function at inference time would strip out exactly the row we
need to predict from.

The fix: split feature computation (compute_features) from target creation
(add_targets). Inference calls only compute_features(); training calls both.
Both pipelines now share one implementation of lags/rolling/cyclic-encoding,
so there's no risk of the two silently drifting apart over time.
"""

import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 24, 48, 72]
HORIZONS = ["aqi_t+24h", "aqi_t+48h", "aqi_t+72h"]
HORIZON_TO_LEAD_DAYS = {"aqi_t+24h": 1, "aqi_t+48h": 2, "aqi_t+72h": 3}
HORIZON_TO_HOURS = {"aqi_t+24h": 24, "aqi_t+48h": 48, "aqi_t+72h": 72}

# Deepest dependency for the most recent row's features = lag_72 /
# rolling_24h / aqi_change_rate_24h, all needing up to 72h of prior data.
MIN_HISTORY_HOURS = 72

INTERPOLATE_INT_COLS = (
    "us_aqi", "relative_humidity_2m", "wind_direction_10m", "cloud_cover",
)


def clean_raw(df, int_cols=INTERPOLATE_INT_COLS):
    """
    Sorts by timestamp, reindexes to a full hourly range, and interpolates
    short gaps only (limit=2), leaving longer genuine outages as NaN rather
    than fabricating them.

    Returns (clean_df, missing_df). missing_df holds the rows that were
    absent from the input and have now been filled in via interpolation —
    callers that own a feature-group handle (training) can write these back
    to Hopsworks; inference does not need to.
    """
    df = df.copy()
    for col in ["Unnamed: 0"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    if len(df) == 0:
        raise ValueError("clean_raw() received an empty dataframe.")

    expected = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
    missing_rows = expected.difference(df.index)

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    full_index.name = "timestamp"
    df = df.reindex(full_index)
    df = df.interpolate(method="time", limit=2)

    missing_df = df.loc[missing_rows].copy()
    missing_df.index.name = "timestamp"
    present_int_cols = [c for c in int_cols if c in missing_df.columns]
    if present_int_cols and len(missing_df) > 0:
        missing_df[present_int_cols] = (
            missing_df[present_int_cols].round().astype("Int64")
        )

    return df, missing_df


def compute_features(df):
    """
    Adds every engineered FEATURE column: time-based, derived (AQI change
    rate), rolling stats, lags, and cyclic encodings.

    Deliberately does NOT create targets and does NOT drop any rows.
    Rows at the very start of df will still have NaNs in lag/rolling columns
    (no history before them) — callers should select the row(s) they
    actually need (see latest_valid_row) rather than blanket-dropna here.
    """
    df = df.copy()

    df["month"] = df.index.month - 1
    df["day_of_week"] = df.index.day_of_week
    df["is_weekday"] = (df["day_of_week"] < 5).astype(int)
    df["hour"] = df.index.hour

    df["aqi_change_rate_1h"] = df["us_aqi"].diff(1)
    df["aqi_change_rate_24h"] = df["us_aqi"].diff(24)

    df["rolling_mean_24h"] = df["us_aqi"].rolling(window=24).mean()
    df["rolling_std_24h"] = df["us_aqi"].rolling(window=24).std()

    for lag in LAGS:
        df[f"us_aqi_lag{lag}"] = df["us_aqi"].shift(lag)

    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["wind_direction_10m_sin"] = np.sin(2 * np.pi * df["wind_direction_10m"] / 360)
    df["wind_direction_10m_cos"] = np.cos(2 * np.pi * df["wind_direction_10m"] / 360)

    if "ammonia" in df.columns:
        df = df.drop(columns=["ammonia"])

    return df


def add_targets(df):
    """Training-only: appends the 3 forecast-horizon target columns."""
    df = df.copy()
    df["aqi_t+24h"] = df["us_aqi"].shift(-24)
    df["aqi_t+48h"] = df["us_aqi"].shift(-48)
    df["aqi_t+72h"] = df["us_aqi"].shift(-72)
    return df


def get_feature_sets(df):
    """Same split used throughout: tree_cols (raw) vs linear_cols (scaled subset)."""
    tree_drop = ["city", "day"] + HORIZONS
    tree_cols = [c for c in df.columns if c not in tree_drop]

    linear_drop = [
        "city", "hour", "day_of_week", "month",
        "wind_direction_10m", "dust",
    ] + HORIZONS
    linear_cols = [c for c in df.columns if c not in linear_drop]

    return tree_cols, linear_cols


def latest_valid_row(df, feature_cols):
    """
    Returns the most recent single-row DataFrame where every column in
    feature_cols is non-NaN. Used by inference to grab "now"'s feature
    vector after compute_features() has been applied to a raw window.

    Raises if no such row exists — this means the raw window pulled for
    inference was too short, or had a gap wider than clean_raw()'s
    interpolation limit right before the most recent hour.
    """
    valid = df.dropna(subset=feature_cols)
    if valid.empty:
        raise ValueError(
            "No row with complete features found in the supplied window — "
            f"it may be shorter than the required {MIN_HISTORY_HOURS}h of "
            "trailing history, or has an unfillable gap near the most "
            "recent hour."
        )
    return valid.iloc[[-1]]
