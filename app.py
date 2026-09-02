import os
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

# --- HOPSWORKS CONNECTION (Cached for speed) ---
@st.cache_resource(show_spinner="Connecting to Hopsworks Feature Store...")
def get_hopsworks_project():
    api_key = st.secrets["HOPSWORKS_API_KEY"]
    if not api_key:
        st.error("`HOPSWORKS_API_KEY` environment variable not found!")
        st.stop()

    project = hopsworks.login(
        project="PearlsAQI_Project",
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=api_key,
    )
    return project

@st.cache_data(ttl=300, show_spinner="Fetching latest predictions & actuals...")
def load_data():
    project = get_hopsworks_project()
    fs = project.get_feature_store()

    # 1. Fetch Latest Forecasts & Models
    
    df_preds = pd.read_csv("predictions_log.csv")

    # 2. Fetch Latest SHAP Values
    shap_fg = fs.get_feature_group(name="aqi_shap_values", version=1)
    df_shap = shap_fg.read()

    # 3. Fetch Raw/Actual AQI Data for Karachi
    raw_fg = fs.get_feature_group(name="karachi_aqi", version=1)
    df_actuals = raw_fg.read()

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
    # Header
    st.title("🌫️ Karachi Air Quality Index (AQI) Forecast")
    st.caption("Live hourly machine learning forecasts powered by Hopsworks, Open-Meteo API, Ridge, Random Forest, and Neural Networks.")
    st.markdown("---")
    st.write("Hopsworks version:", hopsworks.__version__)
    try:
        df_preds, df_shap, df_actuals = load_data()
    except Exception as e:
        st.error(f"Failed to load data from Hopsworks: {e}")
        return

    # Clean & sort dataframes
    df_preds["target_timestamp"] = pd.to_datetime(df_preds["target_timestamp"])
    df_preds["prediction_made_at"] = pd.to_datetime(df_preds["prediction_made_at"])
    df_actuals["timestamp"] = pd.to_datetime(df_actuals["timestamp"])

    # ------------------------------------------------------------------
    # 1. Top Section: Forecast Cards (Latest predictions per horizon)
    # ------------------------------------------------------------------
    st.subheader("📍 Current Air Quality Forecasts for Karachi")
    
    # Get the most recent inference run predictions
    latest_run_time = df_preds["prediction_made_at"].max()
    latest_preds = df_preds[df_preds["prediction_made_at"] == latest_run_time]

    horizons = sorted(latest_preds["horizon"].unique())
    cols = st.columns(len(horizons) if len(horizons) > 0 else 1)

    for i, horizon in enumerate(horizons):
        row = latest_preds[latest_preds["horizon"] == horizon].iloc[0]
        aqi_val = round(row["predicted_aqi"], 1)
        model_used = row["model_used"]  # e.g., 'ridge', 'rf', 'nn'
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
        
        # Plot actuals + predictions
        fig = go.Figure()

        # Actual AQI trace
        recent_actuals = df_actuals.sort_values("timestamp").tail(168)  # last 7 days
        fig.add_trace(go.Scatter(
            x=recent_actuals["timestamp"],
            y=recent_actuals["us_aqi"],
            mode="lines",
            name="Actual Karachi US AQI",
            line=dict(color="#1f77b4", width=2)
        ))

        # Latest predictions trace
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
        
        selected_horizon = st.selectbox("Select Horizon to view drivers:", horizons)
        
        # Get matching SHAP row
        shap_row = df_shap[
            (df_shap["horizon"] == selected_horizon) & 
            (pd.to_datetime(df_shap["target_timestamp"]) == latest_preds[latest_preds["horizon"] == selected_horizon]["target_timestamp"].iloc[0])
        ]

        if not shap_row.empty:
            # Parse top SHAP features (excluding non-feature metadata columns)
            ignore_cols = ["timestamp", "target_timestamp", "horizon", "prediction_made_at"]
            feature_impacts = {
                col: shap_row[col].values[0] 
                for col in shap_row.columns 
                if col not in ignore_cols and pd.notna(shap_row[col].values[0])
            }
            
            # Sort top 8 highest absolute impacts
            df_shap_plot = (
                pd.DataFrame(list(feature_impacts.items()), columns=["Feature", "SHAP Impact"])
                .reindex(df_shap_plot["SHAP Impact"].abs().sort_values(ascending=False).index)
                .head(8)
            )

            fig_shap = px.bar(
                df_shap_plot,
                x="SHAP Impact",
                y="Feature",
                orientation="h",
                color="SHAP Impact",
                color_continuous_scale="RdBu_r",
                title=f"Top Features Influencing {selected_horizon.upper()} Forecast"
            )
            fig_shap.update_layout(yaxis=dict(autorange="reverse"), margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("No SHAP values logged for the selected horizon yet.")

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