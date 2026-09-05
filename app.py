import os
import time
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from datetime import datetime, timedelta

DATA_FOLDER = "data"
DATA_6D_FILE_NAME = "hourly_data_6d"


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

# --- AQI utility func ---
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



# --- gauge/speedometer func ---

def darken_color(hex_color, factor=0.65):
    """
    Makes the active AQI bar slightly darker
    than the background AQI section.
    """
    hex_color = hex_color.lstrip("#")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)

    return f"#{r:02x}{g:02x}{b:02x}"

def us_aqi_gauge(aqi):

    aqi = float(aqi)

    category, color, emoji = get_aqi_status(aqi)

    # Slightly darker version for active bar
    active_color = darken_color(color, 0.8)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=aqi,

            number={
                "font": {
                    "size": 85,
                    "color": color
                }
            },

            title={
                "text": "US AQI",
                "font": {
                    "size": 18
                }
            },

            gauge={
                "shape": "angular",

                "axis": {
                    "range": [0, 500],
                    "tickvals": [
                        0, 50, 100, 150,
                        200, 300, 500
                    ],
                    "tickfont": {
                        "size": 10
                    }
                },

                # Active portion
                "bar": {
                    "color": active_color,
                    "thickness": 0.20
                },

                # AQI ranges
                "steps": [
                    {
                        "range": [0, 50],
                        "color": "#00e400"
                    },
                    {
                        "range": [50, 100],
                        "color": "#ffff00"
                    },
                    {
                        "range": [100, 150],
                        "color": "#ff7e00"
                    },
                    {
                        "range": [150, 200],
                        "color": "#ff0000"
                    },
                    {
                        "range": [200, 300],
                        "color": "#8f3f97"
                    },
                    {
                        "range": [300, 500],
                        "color": "#7e0023"
                    }
                ],

                "borderwidth": 0
            },

            domain={
                "x": [0, 1],
                "y": [0.20, 1]
            }
        )
    )

    fig.update_layout(
        height=320,

        margin=dict(
            l=10,
            r=10,
            t=20,
            b=0
        ),

        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig, category, color, emoji


@st.cache_data(ttl=300, show_spinner="Fetching latest predictions & actuals...")
def load_data():

    # 1. Fetch Predictions from Local Repository CSV
    csv_file = "predictions_log.csv"

    if os.path.exists(csv_file):
        df_preds = pd.read_csv(csv_file)
    else:
        df_preds = pd.DataFrame(columns=[
            "prediction_made_at", "target_timestamp", "horizon", 
            "predicted_aqi", "model_used", "model_version"
        ])

    # 2. Fetch Latest SHAP Values from Hopsworks with Retry
    df_shap = pd.read_csv(f"{DATA_FOLDER}/aqi_shap_values.csv")

    # 3. Fetch Raw/Actual AQI Data for Karachi from Hopsworks with Retry
    df_actuals = pd.read_csv(f"{DATA_FOLDER}/{DATA_6D_FILE_NAME}.csv")

    return df_preds, df_shap, df_actuals


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
    df_preds["target_timestamp"] = pd.to_datetime(df_preds["target_timestamp"],format="mixed",errors="coerce")
    df_preds["prediction_made_at"] = pd.to_datetime(df_preds["prediction_made_at"],format="mixed",errors="coerce")
    df_actuals["timestamp"] = pd.to_datetime(df_actuals["timestamp"],format="mixed",errors="coerce")

    # Extract latest predictions
    latest_run_time = df_preds["prediction_made_at"].max()
    latest_preds = df_preds[df_preds["prediction_made_at"] == latest_run_time]
    # ------------------------------------------------------------------
    # Indicator for AQI
    # ------------------------------------------------------------------
    st.markdown("---")

    latest_actual_row = df_actuals.iloc[df_actuals["timestamp"].idxmax()]
    latest_actual_aqi = latest_actual_row["us_aqi"]
    left, right = st.columns([1, 2])
    with left:
        fig, category, color, emoji = us_aqi_gauge(latest_actual_aqi)

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )

        st.markdown(
            f"""
            <div style="
                text-align: center;
                color: {color};
                font-size: 24px;
                font-weight: 700;
                margin-top: -55px;
            ">
                {category}
            </div>
            """,
            unsafe_allow_html=True
        )


    # -------------------------
    # POLLUTANTS
    # -------------------------

    with right:

        st.subheader("Pollutants")

        with st.container(border=True):

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "PM₂.₅",
                    f"{latest_actual_row['pm2_5']} µg/m³"
                )

            with col2:
                st.metric(
                    "PM₁₀",
                    f"{latest_actual_row['pm10']} µg/m³"
                )

            with col3:
                st.metric(
                    "CO",
                    f"{latest_actual_row['carbon_monoxide']} µg/m³"
                )


            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "NO₂",
                    f"{latest_actual_row['nitrogen_dioxide']} µg/m³"
                )

            with col2:
                st.metric(
                    "SO₂",
                    f"{latest_actual_row['sulphur_dioxide']} µg/m³"
                )

            with col3:
                st.metric(
                    "O₃",
                    f"{latest_actual_row['ozone']} µg/m³"
                )


            st.metric(
                "Dust",
                f"{latest_actual_row['dust']} µg/m³"
            )

    st.markdown("---")

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

        # Actual AQI trace (last 6 days)
        recent_actuals = df_actuals.sort_values("timestamp").tail(144)
        fig.add_trace(go.Scatter(
            x=recent_actuals["timestamp"],
            y=recent_actuals["us_aqi"],
            mode="lines",
            name="Actual Karachi US AQI",
            line=dict(color="#1f77b4", width=2)
        ))

        # Predictions trace
        hori_color = {"aqi_t+24h":"#ff7f0e","aqi_t+48h":"#ad0707","aqi_t+72h":"#5805e8"}
        for hori, group in df_preds.groupby("horizon"):
            fig.add_trace(go.Scatter(
                x=group["target_timestamp"],
                y=group["predicted_aqi"].round(2),
                mode="lines+markers",
                name=f"Model predictionds {hori}",  
                marker=dict(size=4, color=hori_color[hori]),
                line=dict(color=hori_color[hori])
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
                use_container_width=True,
                config = {
                    "displayModeBar": False
                }
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