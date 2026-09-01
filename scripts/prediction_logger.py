"""
Prediction logging.

Called once per day from train_pipeline.py, right after the best model per
horizon has been selected. Computes today's forecast (t+24h, t+48h, t+72h)
using the just-trained winning models and appends them to a dedicated
Hopsworks feature group, so the web dashboard can later look back and show
"what did we predict for this hour, 1/2/3 days before it happened".

This is the only source of historical prediction data — nothing before this
script starts running will have logged predictions, so the accuracy chart
will only start accumulating from whenever this is first deployed.
"""

import pandas as pd


PREDICTIONS_FG_NAME = "aqi_predictions_log"
PREDICTIONS_FG_VERSION = 1

HORIZON_HOURS = {"aqi_t+24h": 24, "aqi_t+48h": 48, "aqi_t+72h": 72}


def get_or_create_predictions_fg(fs):
    return fs.get_or_create_feature_group(
        name=PREDICTIONS_FG_NAME,
        version=PREDICTIONS_FG_VERSION,
        description="Logged AQI forecasts, one row per horizon per day, so "
                     "predicted-vs-actual accuracy can be tracked over time.",
        primary_key=["target_timestamp", "horizon_hours"],
        event_time="predicted_at",
    )


def log_todays_predictions(fs, predicted_at, best_models, X_latest_by_algo):
    """
    predicted_at: pd.Timestamp — the timestamp of the latest available data row
    best_models: dict like {"aqi_t+24h": {"algo": "rf", "model": <obj>, "version": 3}, ...}
    X_latest_by_algo: dict like {"rf": X_row_df, "ridge": X_row_df, "nn": X_row_df}
                       (the correctly-shaped/scaled feature row for each algo type)
    """
    fg = get_or_create_predictions_fg(fs)

    rows = []
    for horizon, info in best_models.items():
        hours = HORIZON_HOURS[horizon]
        algo = info["algo"]
        model = info["model"]
        X_row = X_latest_by_algo[algo]

        pred = float(model.predict(X_row).flatten()[0]) if algo == "nn" else float(model.predict(X_row)[0])

        rows.append({
            "predicted_at": predicted_at,
            "target_timestamp": predicted_at + pd.Timedelta(hours=hours),
            "horizon_hours": hours,
            "predicted_aqi": round(pred, 2),
            "model_used": algo,
            "model_version": info["version"],
        })

    log_df = pd.DataFrame(rows)
    fg.insert(log_df)
    print(f"Logged {len(log_df)} predictions for {predicted_at}")
    return log_df
