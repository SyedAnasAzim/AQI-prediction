import os
import joblib
import numpy as np
import pandas as pd
import hopsworks

from dotenv import load_dotenv
from datetime import timedelta
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from feature_engineering import (
    clean_raw,
    compute_features,
    add_targets,
    get_feature_sets as shared_get_feature_sets,
    HORIZONS,
    LAGS,
)


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
ROLLING_WINDOW_DAYS = 730  # ~2 years — keeps full seasonal coverage; currently a no-op
                            # since the dataset doesn't yet exceed this span, but will
                            # begin trimming the oldest data once it does
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]

# HORIZONS and LAGS now live in feature_engineering.py (imported below) so
# training and inference can never define them differently.

SAVE_ROOT = "saved_models"

# --- SHAP background sample config ---
SHAP_BACKGROUND_FG_NAME = "shap_background_karachi_aqi"
SHAP_BACKGROUND_FG_VERSION = 1
SHAP_BACKGROUND_N_SAMPLES = 150  # target size of the reference sample
SHAP_STRAT_COLS = ["month", "hour"]  # stratify so all seasons/times of day are represented
SHAP_RANDOM_STATE = 42


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


# ----------------------------------------------------------------------
# 1. Fetch raw data from Hopsworks
# ----------------------------------------------------------------------
def fetch_raw_data():
    load_dotenv()
    api_key = os.getenv("HOPSWORKS_API_KEY")

    project = hopsworks.login(
        project="PearlsAQI_Project",
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=api_key,
    )

    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="karachi_aqi", version=1)
    df = fg.read()

    return project, df, fg


# ----------------------------------------------------------------------
# 2. Preprocessing — now delegates to feature_engineering.py, the module
#    shared with inference.py, so training and serving can never compute
#    features differently by accident.
# ----------------------------------------------------------------------
def preprocess(raw_df, fg):
    # Clean + fill short gaps; write back any interpolated rows that were
    # missing from the raw feature group (unchanged behavior from before).
    df, missing_df = clean_raw(raw_df)
    if len(missing_df) > 0:
        fg.insert(missing_df.reset_index())

    # Shared feature computation (time-based, derived, rolling, lag, cyclic).
    df = compute_features(df)

    # Training-only: create the 3 forecast-horizon targets.
    df = add_targets(df)

    # Drop rows with NaNs introduced by lagging/rolling/target-shifting
    # (expected: ~72 rows at the start, ~72 at the end).
    df = df.dropna(axis=0)
    return df


# ----------------------------------------------------------------------
# 3. Apply rolling window (2-year cap)
# ----------------------------------------------------------------------
def apply_rolling_window(df, window_days=ROLLING_WINDOW_DAYS):
    cutoff = df.index.max() - timedelta(days=window_days)
    print(cutoff)
    return df[df.index >= cutoff]


# ----------------------------------------------------------------------
# 4. Time-based train/val/test split (matches the EDA notebook's approach)
# ----------------------------------------------------------------------
def time_based_split(df):
    min_date, max_date = df.index.min(), df.index.max()
    train_split = ((max_date - min_date) * 0.7 + min_date).floor("h")
    val_split = ((max_date - min_date) * 0.85 + min_date).floor("h")

    train = df.loc[:train_split]
    val = df.loc[train_split + pd.Timedelta(hours=1):val_split]
    test = df.loc[val_split + pd.Timedelta(hours=1):]

    return train, val, test


# ----------------------------------------------------------------------
# 5. Feature set definitions (Set A: tree-based, Set B: linear/NN scaled)
#    Delegates to feature_engineering.py so training and inference always
#    slice features identically.
# ----------------------------------------------------------------------
def get_feature_sets(df):
    return shared_get_feature_sets(df)


def build_nn(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mae",
        metrics=["mae"]
    )
    return model


# ----------------------------------------------------------------------
# 5b. SHAP background/reference sample
# ----------------------------------------------------------------------
def build_shap_background_sample(
    df,
    feature_cols,
    strat_cols=SHAP_STRAT_COLS,
    n_samples=SHAP_BACKGROUND_N_SAMPLES,
    random_state=SHAP_RANDOM_STATE,
):
    """
    Draws a small, seasonally-representative sample of engineered feature rows
    to serve as the SHAP background/reference distribution at inference time.

    Why not just use a rolling window of recent hours at inference time:
    - Recent hours are highly autocorrelated (not independent samples), which
      biases LinearExplainer's covariance estimate.
    - A short recent window has no seasonal coverage (e.g. dust storms vs.
      winter smog vs. monsoon humidity effects), so "typical" ends up meaning
      "whatever conditions looked like this week."
    - The SHAP base value (expected model output over the background set)
      would drift hour to hour instead of representing a stable reference,
      making SHAP values hard to compare across different prediction runs.

    Strategy: take one row per (month, hour) bucket so the full seasonal /
    diurnal cycle is represented, then subsample down to n_samples if that
    produced more rows than needed. This is refreshed once a day (each
    training run), matching the daily retrain cadence.
    """
    sampled = (
        df.groupby(strat_cols, group_keys=False)
        .apply(lambda g: g.sample(n=1, random_state=random_state))
    )

    if len(sampled) > n_samples:
        sampled = sampled.sample(n=n_samples, random_state=random_state)

    sampled = sampled[feature_cols].copy()
    sampled = sampled.reset_index()  # brings "timestamp" back as a column
    return sampled


def save_shap_background_sample(fs, df, tree_cols, linear_cols):
    """
    Builds and saves/overwrites the SHAP background sample feature group.
    """
    feature_cols = sorted(set(tree_cols) | set(linear_cols))
    background_df = build_shap_background_sample(df, feature_cols)

    background_fg = fs.get_or_create_feature_group(
        name=SHAP_BACKGROUND_FG_NAME,
        version=SHAP_BACKGROUND_FG_VERSION,
        description=(
            "Stratified (month, hour) sample of engineered feature rows, "
            "used as the SHAP background/reference distribution for "
            "LinearExplainer/KernelExplainer at inference time. "
            "Refreshed daily by the training pipeline."
        ),
        primary_key=["timestamp"],
        event_time="timestamp",
    )

    # insert() handles both first-time registration (creates the group's
    # schema from this dataframe) and subsequent writes — no separate
    # "save" step needed or exists for data in hsfs.
    background_fg.insert(
        background_df,
        write_options={"wait_for_job": True},
    )

    print(
        f"SHAP background sample saved: {len(background_df)} rows "
        f"to '{SHAP_BACKGROUND_FG_NAME}' v{SHAP_BACKGROUND_FG_VERSION}"
    )

# ----------------------------------------------------------------------
# 6. Main training routine
# ----------------------------------------------------------------------
def main():
    print("Fetching raw data from Hopsworks...")
    project, raw_df, fg = fetch_raw_data()

    print("Preprocessing / feature engineering...")
    df = preprocess(raw_df, fg)

    print("Applying rolling window...")
    df = apply_rolling_window(df)
    print(f"Training window: {df.index.min()} to {df.index.max()} ({len(df)} rows)")

    train, val, test = time_based_split(df)
    tree_cols, linear_cols = get_feature_sets(df)

    print("Building SHAP background sample...")
    fs = project.get_feature_store()
    save_shap_background_sample(fs, df, tree_cols, linear_cols)

    X_train_rf, X_val_rf, X_test_rf = train[tree_cols], val[tree_cols], test[tree_cols]
    X_train_lin = train[linear_cols].copy()
    X_val_lin = val[linear_cols].copy()
    X_test_lin = test[linear_cols].copy()

    cyclic_cols = [c for c in linear_cols if c.endswith(("_sin", "_cos"))]
    cols_to_scale = [c for c in linear_cols if c not in cyclic_cols]

    scaler = StandardScaler()
    X_train_lin[cols_to_scale] = scaler.fit_transform(X_train_lin[cols_to_scale])
    X_val_lin[cols_to_scale] = scaler.transform(X_val_lin[cols_to_scale])
    X_test_lin[cols_to_scale] = scaler.transform(X_test_lin[cols_to_scale])

    mr = project.get_model_registry()
    all_metrics = {"ridge": {}, "rf": {}, "nn": {}}

    for horizon in HORIZONS:
        y_train, y_val, y_test = train[horizon], val[horizon], test[horizon]
        print(f"\n=== {horizon} ===")

        # --- Persistence baseline (logged for comparison, not registered) ---
        persist_pred = df.loc[test.index, "us_aqi"]
        persist_r2 = r2_score(y_test, persist_pred)
        persist_rmse = rmse(y_test, persist_pred)
        persist_mae = mean_absolute_error(y_test, persist_pred)
        print(f"Persistence — R²: {persist_r2:.3f}, MAE: {persist_mae:.3f}, RMSE: {persist_rmse:.3f}")

        # --- Ridge ---
        best_alpha, best_val_mae = None, float("inf")
        for alpha in RIDGE_ALPHAS:
            m = Ridge(alpha=alpha).fit(X_train_lin, y_train)
            val_mae = mean_absolute_error(y_val, m.predict(X_val_lin))
            if val_mae < best_val_mae:
                best_val_mae, best_alpha = val_mae, alpha

        ridge_model = Ridge(alpha=best_alpha).fit(X_train_lin, y_train)
        ridge_pred = ridge_model.predict(X_test_lin)
        all_metrics["ridge"][horizon] = {
            "test_r2": r2_score(y_test, ridge_pred),
            "test_mae": mean_absolute_error(y_test, ridge_pred),
            "test_rmse": rmse(y_test, ridge_pred),
        }
        print(f"Ridge (alpha={best_alpha}) — {all_metrics['ridge'][horizon]}")
        print(f"Compared to persistence — MAE: {(all_metrics['ridge'][horizon]['test_mae']-persist_mae)*100/persist_mae:.3f}, RMSE: {(all_metrics['ridge'][horizon]['test_rmse']-persist_rmse)*100/persist_rmse:.3f}")
        # --- Random Forest ---
        rf_model = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=20,
            min_samples_split=40, max_features="sqrt",
            n_jobs=-1, random_state=42
        ).fit(X_train_rf, y_train)

        rf_pred = rf_model.predict(X_test_rf)
        all_metrics["rf"][horizon] = {
            "test_r2": r2_score(y_test, rf_pred),
            "test_mae": mean_absolute_error(y_test, rf_pred),
            "test_rmse": rmse(y_test, rf_pred),
        }
        print(f"Random Forest — {all_metrics['rf'][horizon]}")
        print(f"Compared to persistence — MAE: {(all_metrics['rf'][horizon]['test_mae']-persist_mae)*100/persist_mae:.3f}, RMSE: {(all_metrics['rf'][horizon]['test_rmse']-persist_rmse)*100/persist_rmse:.3f}")
  
        # --- Neural Network (full retrain, same as Ridge/RF) ---
        tf.random.set_seed(42)
        nn_model = build_nn(X_train_lin.shape[1])
        early_stop = callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        )
        nn_model.fit(
            X_train_lin, y_train,
            validation_data=(X_val_lin, y_val),
            epochs=200, batch_size=64,
            callbacks=[early_stop], verbose=0
        )
        nn_pred = nn_model.predict(X_test_lin, verbose=0).flatten()
        all_metrics["nn"][horizon] = {
            "test_r2": r2_score(y_test, nn_pred),
            "test_mae": mean_absolute_error(y_test, nn_pred),
            "test_rmse": rmse(y_test, nn_pred),
        }
        print(f"Neural Network — {all_metrics['nn'][horizon]}")
        print(f"Compared to persistence — MAE: {(all_metrics['nn'][horizon]['test_mae']-persist_mae)*100/persist_mae:.3f}, RMSE: {(all_metrics['nn'][horizon]['test_rmse']-persist_rmse)*100/persist_rmse:.3f}")

        # --- Identify today's best model for this horizon ---
        scores = {algo: all_metrics[algo][horizon]["test_mae"] for algo in ["ridge", "rf", "nn"]}
        best_algo = min(scores, key=scores.get)
        print(f"Best model for {horizon}: {best_algo} (R²={scores[best_algo]:.3f})")

        # --- Register all three models ---
        model_objs = {"ridge": ridge_model, "rf": rf_model, "nn": nn_model}

        for algo in ["ridge", "rf", "nn"]:
            model_dir = f"{SAVE_ROOT}/{algo}_{horizon.replace('+', '_')}"
            os.makedirs(model_dir, exist_ok=True)
            tag = " [BEST TODAY]" if algo == best_algo else ""

            if algo == "nn":
                model_objs[algo].save(f"{model_dir}/model.keras")
                joblib.dump(scaler, f"{model_dir}/scaler.pkl")
                registered = mr.tensorflow.create_model(
                    name=f"aqi_{algo}_{horizon.replace('+', '_')}",
                    metrics=all_metrics[algo][horizon],
                    description=f"{algo} — {horizon}. Daily full retrain, {ROLLING_WINDOW_DAYS}-day window.{tag}"
                )
            else:
                joblib.dump(model_objs[algo], f"{model_dir}/model.pkl")
                if algo == "ridge":
                    joblib.dump(scaler, f"{model_dir}/scaler.pkl")
                registered = mr.python.create_model(
                    name=f"aqi_{algo}_{horizon.replace('+', '_')}",
                    metrics=all_metrics[algo][horizon],
                    description=(
                        f"{algo} — {horizon}. Daily full retrain, {ROLLING_WINDOW_DAYS}-day window. "
                        + ("Includes bundled scaler.pkl for preprocessing." if algo == "ridge"
                           else "No scaling required (tree-based).")
                        + tag
                    )
                )

            registered.save(model_dir)
            print(f"Registered: aqi_{algo}_{horizon.replace('+', '_')}{tag}")

    print("\nDaily training complete.")
    print(all_metrics)


if __name__ == "__main__":
    main()
