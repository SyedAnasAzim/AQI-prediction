import os
import time
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import hopsworks
from dotenv import load_dotenv

# Set page configuration
st.set_page_config(
    page_title="Karachi AQI Forecast Dashboard",
    page_icon="🌫️",
    layout="wide",
)

load_dotenv()


# --- RECOVERY & RETRY LOGIC ---
def retry(fn, *, retries=3, delay=10, backoff=2, label="operation"):
    """Retry fn() on exception with exponential backoff."""
    last_exc = None
    current_delay = delay
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            st.warning(f"[{label}] Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(current_delay)
                current_delay *= backoff
    raise last_exc


# --- HOPSWORKS CONNECTION (Cached for speed with Retries) ---
@st.cache_resource(show_spinner="Connecting to Hopsworks Feature Store...")
def get_hopsworks_project():
    api_key = st.secrets["HOPSWORKS_API_KEY"]
    if not api_key:
        st.error("`HOPSWORKS_API_KEY` environment variable not found!")
        st.stop()

    def _login():
        return hopsworks.login(
            project="PearlsAQI_Project",
            host="eu-west.cloud.hopsworks.ai",
            port=443,
            api_key_value=api_key,
        )

    return retry(_login, retries=3, delay=5, label="Hopsworks Login")


@st.cache_data(ttl=300, show_spinner="Fetching latest predictions & actuals...")
def load_data():
    project = get_hopsworks_project()
    fs = retry(lambda: project.get_feature_store(), retries=2, label="Get Feature Store")

    # 1. Fetch Predictions from Local Repository CSV
    csv_file = "predictions_log.csv"
    if not os.path.exists(csv_file):
        csv_file = "predictions.csv"

    if os.path.exists(csv_file):
        df_preds = pd.read_csv(csv_file)
    else:
        df_preds = pd.DataFrame(columns=[
            "prediction_made_at", "target_timestamp", "horizon", 
            "predicted_aqi", "model_used", "model_version"
        ])

    # 2. Fetch Latest SHAP Values from Hopsworks with Retry
    def _read_shap():
        fg = fs.get_feature_group(name="aqi_shap_values", version=1)
        return fg.read()

    df_shap = retry(_read_shap, retries=3, delay=10, label="Read SHAP Feature Group")

    # 3. Fetch Raw/Actual AQI Data for Karachi from Hopsworks with Retry
    def _read_actuals():
        fg = fs.get_feature_group(name="karachi_aqi", version=1)
        return fg.read()

    df_actuals = retry(_read_actuals, retries=3, delay=10, label="Read Actuals Feature Group")

    return df_preds, df_shap, df_actuals


# --- AQI CATEGORY UTILS ---
def get_aqi_status(aqi_val):
    if aqi_val <= 50:
        return "Good", "#00e400", "🟢"
    elif aqi_val <= 100:
        return "Moderate", "#ffff00", "🟡"
    elif aqi_val <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00", "🟠"
    elif aqi_val <= 200:
        return "Unhealthy", "#ff0000", "🔴"
    elif aqi_val <= 300:
        return "Very Unhealthy", "#8f3f97", "🟣"
    else:
        return "Hazardous", "#7e0023", "🟤"


# --- MAIN APP LAYOUT ---
def main():
    st.title("🌫️ Karachi Air Quality Index (AQI) Forecast")
    st.caption("Live hourly machine learning forecasts powered by Hopsworks, Open-Meteo API, Ridge, Random Forest, and Neural Networks.")

    try:
        df_preds, df_shap, df_actuals = load_data()
    except Exception as e:
        st.error(f"Failed to load data from Hopsworks or CSV log: {e}")
        return

    if df_preds.empty:
        st.warning("No predictions found in local CSV log. Please ensure `inference.py` has run.")
        return

    # Clean & sort dataframes
    df_preds["target_timestamp"] = pd.to_datetime(df_preds["target_timestamp"])
    df_preds["prediction_made_at"] = pd.to_datetime(df_preds["prediction_made_at"])
    df_actuals["timestamp"] = pd.to_datetime(df_actuals["timestamp"])

    # Extract latest predictions
    latest_run_time = df_preds["prediction_made_at"].max()
    latest_preds = df_preds[df_preds["prediction_made_at"] == latest_run_time]

    # ------------------------------------------------------------------
    # AQI DYNAMIC ALERT BANNER
    # ------------------------------------------------------------------
    max_predicted_aqi = latest_preds["predicted_aqi"].max()
    peak_row = latest_preds[latest_preds["predicted_aqi"] == max_predicted_aqi].iloc[0]
    peak_horizon = peak_row["horizon"].upper()
    peak_time = peak_row["target_timestamp"].strftime("%Y-%m-%d %H:%M")

    if max_predicted_aqi > 200:
        st.error(
            f"🚨 **SEVERE AQI ALERT:** Peak predicted AQI reaches **{round(max_predicted_aqi, 1)}** "
            f"(Very Unhealthy/Hazardous) for the **{peak_horizon}** forecast ({peak_time}). "
            f"Avoid outdoor activities and wear an N95 mask!"
        )
    elif max_predicted_aqi > 150:
        st.warning(
            f"⚠️ **UNHEALTHY AQI WARNING:** Peak predicted AQI is **{round(max_predicted_aqi, 1)}** "
            f"for the **{peak_horizon}** forecast ({peak_time}). "
            f"Sensitive groups (children, elderly, asthmatics) should stay indoors."
        )
    elif max_predicted_aqi > 100:
        st.info(
            f"🟡 **MODERATE TO SENSITIVE AQI ADVISORY:** Forecasted AQI peaks at **{round(max_predicted_aqi, 1)}** "
            f"for the **{peak_horizon}** forecast. Air quality is acceptable, but sensitive individuals may feel slight discomfort."
        )
    else:
        st.success(
            f"🟢 **GOOD AIR QUALITY:** Forecasted AQI remains safe at or below **{round(max_predicted_aqi, 1)}** "
            f"across all upcoming horizons."
        )

    st.markdown("---")

    # ------------------------------------------------------------------
    # 1. Top Section: Forecast Cards
    # ------------------------------------------------------------------
    st.subheader("📍 Current Air Quality Forecasts for Karachi")

    horizons = sorted(latest_preds["horizon"].unique())
    cols = st.columns(len(horizons) if len(horizons) > 0 else 1)

    for i, horizon in enumerate(horizons):
        row = latest_preds[latest_preds["horizon"] == horizon].iloc[0]
        aqi_val = round(row["predicted_aqi"], 1)
        model_used = row["model_used"]
        model_ver = row["model_version"]
        target_ts = row["target_timestamp"].strftime("%Y-%m-%d %H:%M")

        status, color, emoji = get_aqi_status(aqi_val)

        with cols[i]:
            st.metric(
                label=f"Horizon: {horizon.upper()} ({target_ts})",
                value=f"{aqi_val} AQI {emoji}",
                delta=f"Model: {model_used.upper()} (v{model_ver})",
                delta_color="off"
            )
            st.markdown(f"**Health Category:** <span style='color:{color}; font-weight:bold;'>{status}</span>", unsafe_allow_html=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # 2. Main Visuals: Historical Actuals vs Forecast Trend
    # ------------------------------------------------------------------
    col_chart, col_shap = st.columns([6, 4])

    with col_chart:
        st.subheader("📈 AQI Trend: Actual vs Forecast")
        
        fig = go.Figure()

        # Actual AQI trace (last 7 days)
        recent_actuals = df_actuals.sort_values("timestamp").tail(168)
        fig.add_trace(go.Scatter(
            x=recent_actuals["timestamp"],
            y=recent_actuals["us_aqi"],
            mode="lines",
            name="Actual Karachi US AQI",
            line=dict(color="#1f77b4", width=2)
        ))

        # Predictions trace
        fig.add_trace(go.Scatter(
            x=latest_preds["target_timestamp"],
            y=latest_preds["predicted_aqi"],
            mode="markers+lines",
            name="Model Predictions",
            marker=dict(size=10, color="#ff7f0e"),
            line=dict(dash="dash")
        ))

        fig.update_layout(
            xaxis_title="Timestamp (Karachi Time)",
            yaxis_title="US AQI Level",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 3. SHAP Feature Drivers Section
    # ------------------------------------------------------------------
   with col_shap:
    st.subheader("🔍 Prediction Drivers (SHAP Values)")

    selected_horizon = st.selectbox(
        "Select Horizon to view drivers:",
        horizons
    )

    target_ts_horizon = latest_preds[
        latest_preds["horizon"] == selected_horizon
    ]["target_timestamp"].iloc[0]

    # Make timestamps comparable
    target_ts_horizon = pd.to_datetime(target_ts_horizon)

    # Filter SHAP data
    shap_row = df_shap[
        (df_shap["horizon"] == selected_horizon) &
        (
            pd.to_datetime(df_shap["target_timestamp"]) ==
            target_ts_horizon
        )
    ].copy()

    if not shap_row.empty:

        # --------------------------------
        # SHAP data is already long format
        # --------------------------------

        df_shap_plot = shap_row[
            ["feature_name", "shap_value"]
        ].copy()

        # Make sure SHAP values are numeric
        df_shap_plot["shap_value"] = pd.to_numeric(
            df_shap_plot["shap_value"],
            errors="coerce"
        )

        # Remove invalid values
        df_shap_plot = df_shap_plot.dropna(
            subset=["shap_value"]
        )

        # Calculate absolute impact for ranking
        df_shap_plot["abs_impact"] = (
            df_shap_plot["shap_value"].abs()
        )

        # Get top 8 features
        df_shap_plot = (
            df_shap_plot
            .sort_values("abs_impact", ascending=False)
            .head(8)
        )

        # --------------------------------
        # Plot
        # --------------------------------

        fig_shap = px.bar(
            df_shap_plot.sort_values("shap_value"),
            x="shap_value",
            y="feature_name",
            orientation="h",
            color="shap_value",
            color_continuous_scale="RdBu_r",
            title=f"Top Features Influencing {selected_horizon.upper()} Forecast",
            labels={
                "feature_name": "Feature",
                "shap_value": "SHAP Impact"
            }
        )

        fig_shap.update_layout(
            yaxis=dict(autorange="reversed"),
            margin=dict(l=0, r=0, t=50, b=0)
        )

        st.plotly_chart(
            fig_shap,
            use_container_width=True
        )

    else:
        st.info(
            "No SHAP values logged for the selected horizon yet."
        )
    # ------------------------------------------------------------------
    # 4. Footer & Model Information
    # ------------------------------------------------------------------
    st.markdown("---")
    with st.expander("ℹ️ About the Models & Pipeline Architecture"):
        st.write("""
        - **Location:** Karachi, Pakistan (Lat: 24.8607, Lon: 67.0011)
        - **Data Sources:** Open-Meteo Air Quality & Weather APIs stored in **Hopsworks Feature Store**.
        - **Models in Registry:**
            - **Ridge Regression** (L2 Regularized Linear Model)
            - **Random Forest Regressor** (Tree-based Ensemble)
            - **Neural Network** (Multi-Layer Perceptron trained in TensorFlow/Keras)
        - **Daily Retraining:** The highest performing model per forecast horizon is automatically tagged `[BEST TODAY]` and selected by the `inference.py` script.
        """)


if __name__ == "__main__":
    main()
