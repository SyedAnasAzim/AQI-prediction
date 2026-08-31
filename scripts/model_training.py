import os
import joblib
import numpy as np
import pandas as pd
import hopsworks

from datetime import timedelta
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
ROLLING_WINDOW_DAYS = 730  # ~2 years — keeps full seasonal coverage; currently a no-op
                            # since the dataset doesn't yet exceed this span, but will
                            # begin trimming the oldest data once it does
HORIZONS = ["aqi_t+24h", "aqi_t+48h", "aqi_t+72h"]
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]
LAGS = [1, 2, 3, 24, 48, 72]

SAVE_ROOT = "saved_models"


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


# ----------------------------------------------------------------------
# 1. Fetch raw data from Hopsworks
# ----------------------------------------------------------------------
def fetch_raw_data():
    api_key = os.environ["HOPSWORKS_API_KEY"]

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
# 2. Preprocessing — replicates every step from the EDA/feature-engineering notebook
# ----------------------------------------------------------------------
def preprocess(raw_df, fg):
    df = raw_df.copy()

    # Drop any Hopsworks-internal index/id columns if present
    for col in ["Unnamed: 0"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    expected = pd.date_range(
    start=df.index.min(),
    end=df.index.max(),
    freq="h"
    )

    missing_rows = expected.difference(df.index)


    # Reindex to a full hourly range and interpolate short gaps only (limit=2),
    # so longer genuine outages are left as NaN rather than fabricated
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_index)
    df = df.interpolate(method="time", limit=2)


    missing_df = df.loc[missing_rows].copy()
    missing_df.index.name = "timestamp"
    fg.insert(missing_df.reset_index()) 

    # Time-based features
    df["month"] = df.index.month - 1
    df["day_of_week"] = df.index.day_of_week
    df["is_weekday"] = (df["day_of_week"] < 5).astype(int)

    df["hour"] = df.index.hour


    # AQI change rate — derived feature named explicitly in the project spec
    df["aqi_change_rate_1h"] = df["us_aqi"].diff(1)
    df["aqi_change_rate_24h"] = df["us_aqi"].diff(24)

    # Rolling statistics
    df["rolling_mean_24h"] = df["us_aqi"].rolling(window=24).mean()
    df["rolling_std_24h"] = df["us_aqi"].rolling(window=24).std()

    # Lag features — chosen from PACF analysis in the EDA notebook (lags 1-3
    # capture short-term AR structure; 24/48/72 capture daily seasonality and
    # align with the forecast horizons)
    for lag in LAGS:
        df[f"us_aqi_lag{lag}"] = df["us_aqi"].shift(lag)

    # Cyclic encoding for time-based and directional features
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["wind_direction_10m_sin"] = np.sin(2 * np.pi * df["wind_direction_10m"] / 360)
    df["wind_direction_10m_cos"] = np.cos(2 * np.pi * df["wind_direction_10m"] / 360)

    # Targets — shifted forward for each forecast horizon
    df["aqi_t+24h"] = df["us_aqi"].shift(-24)
    df["aqi_t+48h"] = df["us_aqi"].shift(-48)
    df["aqi_t+72h"] = df["us_aqi"].shift(-72)

    # Drop rows with NaNs introduced by lagging/rolling/target-shifting
    # (expected: ~72 rows at the start, ~72 at the end, per the EDA notebook's
    # confirmed-contiguous check)
    df = df.dropna(axis=0)

    return df


# ----------------------------------------------------------------------
# 3. Apply rolling window (2-year cap)
# ----------------------------------------------------------------------
def apply_rolling_window(df, window_days=ROLLING_WINDOW_DAYS):
    cutoff = df.index.max() - timedelta(days=window_days)
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
# ----------------------------------------------------------------------
def get_feature_sets(df):
    tree_drop = ["city", "day"] + HORIZONS
    tree_cols = [c for c in df.columns if c not in tree_drop]

    linear_drop = [
        "city", "hour", "day_of_week", "month", "ammonia",
        "wind_direction_10m", "dust"
    ] + HORIZONS
    linear_cols = [c for c in df.columns if c not in linear_drop]

    return tree_cols, linear_cols


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
