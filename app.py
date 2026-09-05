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

HORIZON_LABELS = {
    "aqi_t+24h": "Next 24 Hours",
    "aqi_t+48h": "Next 48 Hours",
    "aqi_t+72h": "Next 72 Hours",
}
HORIZON_COLORS = {
    "aqi_t+24h": "#C50A0A",
    "aqi_t+48h": "#E6D600",
    "aqi_t+72h": "#00c3ff",
}

st.set_page_config(
    page_title="Karachi Air Quality Forecast",
    page_icon="〰",
    layout="wide",
)

load_dotenv()


# ----------------------------------------------------------------------
# Styling — blue accent system, dark-theme friendly
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --blue-1: #1f3d99;
    --blue-2: #3459b8;
    --blue-3: #5b82d6;
    --blue-4: #82a4e8;
    --blue-5: #aecafd;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Source Serif 4', serif;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: #eaf0ff;
}

h1 {
    background: linear-gradient(90deg, var(--blue-4), var(--blue-5));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

[data-testid="stMetricValue"] {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    color: #f0f4ff;
}

[data-testid="stMetricLabel"] {
    color: var(--blue-4);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}

[data-testid="stMetricDelta"] {
    color: var(--blue-5) !important;
}

hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--blue-2), transparent);
    margin: 1.5rem 0;
}

[data-testid="stVerticalBlockBorderWrapper"] > div {
    border: 1px solid rgba(91, 130, 214, 0.25) !important;
    background: rgba(52, 89, 184, 0.06);
    border-radius: 10px;
}

.reading-time {
    color: var(--blue-4);
    font-size: 0.8rem;
    font-style: italic;
    margin-top: -8px;
}

.location-line {
    color: var(--blue-4);
    font-size: 0.88rem;
    margin-top: -8px;
    margin-bottom: 1.2rem;
    font-weight: 500;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 1.1rem;
    font-weight: 600;
}

.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}

.alert-card {
    border-left: 4px solid var(--blue-2);
    padding: 0.9rem 1.2rem;
    border-radius: 6px;
    margin-bottom: 1rem;
    font-size: 0.95rem;
    line-height: 1.5;
    background: rgba(52, 89, 184, 0.10);
}

.stSelectbox label {
    color: var(--blue-4) !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Retry helper
# ----------------------------------------------------------------------
def retry(fn, *, retries=3, delay=10, backoff=2, label="operation"):
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


# ----------------------------------------------------------------------
# AQI category
# ----------------------------------------------------------------------
def get_aqi_status(aqi_val):
    if aqi_val <= 50:
        return "Good", "#2e7d32"
    elif aqi_val <= 100:
        return "Moderate", "#c9a227"
    elif aqi_val <= 150:
        return "Unhealthy for Sensitive Groups", "#d97b29"
    elif aqi_val <= 200:
        return "Unhealthy", "#c0392b"
    elif aqi_val <= 300:
        return "Very Unhealthy", "#8e44ad"
    else:
        return "Hazardous", "#7b1e2e"


def darken_color(hex_color, factor=0.8):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"


def us_aqi_gauge(aqi):
    aqi = float(aqi)
    category, color = get_aqi_status(aqi)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi,
        number={"font": {"size": 76, "color": color, "family": "Inter"}},
        title={"text": "US AQI", "font": {"size": 15, "family": "Inter", "color": "#a8a8a8"}},
        gauge={
            "shape": "angular",
            "axis": {
                "range": [0, 500],
                "tickvals": [0, 50, 100, 150, 200, 300, 500],
                "tickfont": {"size": 10, "color": "#a8a8a8"},
            },
            "bar": {"color": darken_color(color), "thickness": 0.22},
            "threshold": {"line": {"color": "#ffffff", "width": 3}, "thickness": 0.88, "value": aqi},
            "steps": [
                {"range": [0, 50], "color": "#2e7d32"},
                {"range": [50, 100], "color": "#c9a227"},
                {"range": [100, 150], "color": "#d97b29"},
                {"range": [150, 200], "color": "#c0392b"},
                {"range": [200, 300], "color": "#8e44ad"},
                {"range": [300, 500], "color": "#7b1e2e"},
            ],
            "borderwidth": 0,
        },
        domain={"x": [0, 1], "y": [0.15, 1]},
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig, category, color


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner="Fetching latest forecast data...")
def load_data():
    csv_file = "predictions_log.csv"
    if os.path.exists(csv_file):
        df_preds = pd.read_csv(csv_file)
    else:
        df_preds = pd.DataFrame(columns=[
            "prediction_made_at", "target_timestamp", "horizon",
            "predicted_aqi", "model_used", "model_version"
        ])

    shap_path = f"{DATA_FOLDER}/aqi_shap_values.csv"
    if os.path.exists(shap_path):
        df_shap = pd.read_csv(shap_path)
    else:
        df_shap = pd.DataFrame(columns=["target_timestamp", "horizon", "feature_name", "shap_value"])

    actuals_path = f"{DATA_FOLDER}/{DATA_6D_FILE_NAME}.csv"
    if os.path.exists(actuals_path):
        df_actuals = pd.read_csv(actuals_path)
    else:
        df_actuals = pd.DataFrame(columns=["timestamp", "us_aqi"])

    return df_preds, df_shap, df_actuals


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    st.markdown("# Karachi Air Quality")
    st.caption("Hourly forecasts from an ensemble of Ridge, Random Forest, and Neural Network models.")
    st.markdown(
        '<div class="location-line">📍 Karachi, Pakistan · 24.8607°N, 67.0011°E</div>',
        unsafe_allow_html=True
    )

    try:
        df_preds, df_shap, df_actuals = load_data()
    except Exception as e:
        st.error(f"Couldn't load forecast data: {e}")
        return

    if df_preds.empty:
        st.info("No forecasts logged yet — check back once the hourly pipeline has run.")
        return

    df_preds["target_timestamp"] = pd.to_datetime(df_preds["target_timestamp"], format="mixed", errors="coerce")
    df_preds["prediction_made_at"] = pd.to_datetime(df_preds["prediction_made_at"], format="mixed", errors="coerce")
    df_actuals["timestamp"] = pd.to_datetime(df_actuals["timestamp"], format="mixed", errors="coerce")

    latest_run_time = df_preds["prediction_made_at"].max()
    latest_preds = df_preds[df_preds["prediction_made_at"] == latest_run_time]

    latest_actual_row = df_actuals.iloc[df_actuals["timestamp"].idxmax()]
    latest_actual_aqi = latest_actual_row["us_aqi"]
    latest_actual_time = latest_actual_row["timestamp"].strftime("%b %d, %H:%M")

    # ------------------------------------------------------------------
    # Alert — only surfaced with weight when conditions warrant it
    # ------------------------------------------------------------------
    max_predicted_aqi = latest_preds["predicted_aqi"].max()
    peak_row = latest_preds[latest_preds["predicted_aqi"] == max_predicted_aqi].iloc[0]
    peak_label = HORIZON_LABELS.get(peak_row["horizon"], peak_row["horizon"])
    peak_time = peak_row["target_timestamp"].strftime("%b %d, %H:%M")

    if max_predicted_aqi > 200:
        st.markdown(f"""
        <div class="alert-card" style="border-color:#7b1e2e; background:rgba(123,30,46,0.12);">
            <strong>Severe air quality expected.</strong> The forecast peaks at
            {round(max_predicted_aqi, 1)} AQI ({peak_label}, around {peak_time}).
            Avoid outdoor activity where possible and consider a mask outdoors.
        </div>
        """, unsafe_allow_html=True)
    elif max_predicted_aqi > 150:
        st.markdown(f"""
        <div class="alert-card" style="border-color:#d97b29; background:rgba(217,123,41,0.12);">
            <strong>Unhealthy air quality expected.</strong> Peak forecast is
            {round(max_predicted_aqi, 1)} AQI ({peak_label}, around {peak_time}).
            Sensitive groups should limit time outdoors.
        </div>
        """, unsafe_allow_html=True)
    elif max_predicted_aqi > 100:
        st.markdown(f"""
        <div class="alert-card" style="border-color:#c9a227; background:rgba(201,162,39,0.12);">
            Air quality trends moderate over the coming days — peak forecast of
            {round(max_predicted_aqi, 1)} AQI ({peak_label}).
        </div>
        """, unsafe_allow_html=True)
    # No banner when the forecast stays good.

    st.markdown("---")

    # ------------------------------------------------------------------
    # Current reading
    # ------------------------------------------------------------------
    left, right = st.columns([1, 2])

    with left:
        fig, category, color = us_aqi_gauge(latest_actual_aqi)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"""
        <div style="text-align:center; margin-top:-50px;">
            <span class="status-badge" style="color:{color};">
                <span class="status-dot" style="background:{color};"></span>{category}
            </span>
        </div>
        <div style="text-align:center;" class="reading-time">as of {latest_actual_time}</div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("### Pollutants")
        with st.container(border=True):
            c1, c2 = st.columns(2)
            c1.metric("PM2.5", f"{latest_actual_row['pm2_5']} µg/m³")
            c2.metric("PM10", f"{latest_actual_row['pm10']} µg/m³")

            c1, c2 = st.columns(2)
            c1.metric("Carbon Monoxide", f"{latest_actual_row['carbon_monoxide']} µg/m³")
            c2.metric("Nitrogen Dioxide", f"{latest_actual_row['nitrogen_dioxide']} µg/m³")

            c1, c2 = st.columns(2)
            c1.metric("Sulphur Dioxide", f"{latest_actual_row['sulphur_dioxide']} µg/m³")
            c2.metric("Ozone", f"{latest_actual_row['ozone']} µg/m³")

            st.metric("Dust", f"{latest_actual_row['dust']} µg/m³")

    st.markdown("---")

    st.markdown("### Weather")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Temperature", f"{latest_actual_row['temperature_2m']} °C")
        c2.metric("Humidity", f"{latest_actual_row['relative_humidity_2m']}%")
        c3.metric("Wind Speed", f"{latest_actual_row['wind_speed_10m']} km/h")
        c4.metric("Wind Direction", f"{latest_actual_row['wind_direction_10m']}°")

        c1, c2, c3 = st.columns(3)
        c1.metric("Pressure", f"{latest_actual_row['surface_pressure']} hPa")
        c2.metric("Precipitation", f"{latest_actual_row['precipitation']} mm")
        c3.metric("Cloud Cover", f"{latest_actual_row['cloud_cover']}%")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Forecast cards
    # ------------------------------------------------------------------
    st.markdown("### Next 3 Days Forecast")
    with st.container(border=True):
        horizons = sorted(latest_preds["horizon"].unique())
        cols = st.columns(len(horizons) if horizons else 1)

        for i, horizon in enumerate(horizons):
            row = latest_preds[latest_preds["horizon"] == horizon].iloc[0]
            aqi_val = round(row["predicted_aqi"], 1)
            model_used = row["model_used"]
            model_ver = row["model_version"]
            target_ts = row["target_timestamp"].strftime("%b %d, %H:%M")
            label = HORIZON_LABELS.get(horizon, horizon)
            status, color = get_aqi_status(aqi_val)

            with cols[i]:
                st.metric(label=f"{label} · {target_ts}", value=f"{aqi_val} AQI",
                        delta=f"{model_used.upper()} v{model_ver}", delta_color="off")
                st.markdown(
                    f'<span class="status-badge" style="color:{color}; font-size:0.9rem;">'
                    f'<span class="status-dot" style="background:{color}; width:8px; height:8px;"></span>{status}</span>',
                    unsafe_allow_html=True
                )

    st.markdown("---")

    # ------------------------------------------------------------------
    # Trend + SHAP
    # ------------------------------------------------------------------
    col_chart, col_shap = st.columns([6, 4])

    with col_chart:
        st.markdown("### Actual vs. Forecast")
        fig = go.Figure()
        recent_actuals = df_actuals.sort_values("timestamp").tail(144)
        fig.add_trace(go.Scatter(
            x=recent_actuals["timestamp"], y=recent_actuals["us_aqi"],
            mode="lines", name="Actual AQI", line=dict(color="#2a4699", width=2.5)
        ))
        for hori, group in df_preds.groupby("horizon"):
            fig.add_trace(go.Scatter(
                x=group["target_timestamp"], y=group["predicted_aqi"].round(2),
                mode="lines+markers", name=HORIZON_LABELS.get(hori, hori),
                marker=dict(size=4, color=HORIZON_COLORS.get(hori, "#888")),
                line=dict(color=HORIZON_COLORS.get(hori, "#888"), dash="dot")
            ))
        fig.update_layout(
            xaxis_title=None, yaxis_title="US AQI",
            hovermode="x unified", margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(color="#a8bce8")),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(91,130,214,0.15)", color="#a8bce8"),
            yaxis=dict(gridcolor="rgba(91,130,214,0.15)", color="#a8bce8"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_shap:
        st.markdown("### What's Driving This (SHAP)")
        selected_horizon = st.selectbox(
            "Horizon", horizons, format_func=lambda h: HORIZON_LABELS.get(h, h)
        )
        target_ts_horizon = pd.to_datetime(
            latest_preds[latest_preds["horizon"] == selected_horizon]["target_timestamp"].iloc[0]
        )
        shap_row = df_shap[
            (df_shap["horizon"] == selected_horizon) &
            (pd.to_datetime(df_shap["target_timestamp"]) == target_ts_horizon)
        ].copy()

        if not shap_row.empty:
            df_shap_plot = shap_row[["feature_name", "shap_value"]].copy()
            df_shap_plot["shap_value"] = pd.to_numeric(df_shap_plot["shap_value"], errors="coerce")
            df_shap_plot = df_shap_plot.dropna(subset=["shap_value"])
            df_shap_plot["abs_impact"] = df_shap_plot["shap_value"].abs()
            df_shap_plot = df_shap_plot.sort_values("abs_impact", ascending=False).head(8)

            fig_shap = px.bar(
                df_shap_plot.sort_values("shap_value"),
                x="shap_value", y="feature_name", orientation="h",
                color="shap_value", color_continuous_scale=["#c0392b","#da6363" , "#c8cdd6", "#5b82d6", "#3065d6"],
                labels={"feature_name": "", "shap_value": "Impact on prediction"},
            )
            fig_shap.update_layout(
                yaxis=dict(autorange="reversed", color="#a8bce8"),
                xaxis=dict(color="#a8bce8"),
                margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("No driver data logged for this horizon yet.")
        st.caption("SHAP values show how much each factor pushed this prediction up or down.")

    st.markdown("---")
    with st.expander("About this forecast"):
        st.write("""
        Location: Karachi, Pakistan (24.86°N, 67.00°E)

        Air quality and weather data come from Open-Meteo. Three models — Ridge
        Regression, Random Forest, and a small neural network — are retrained daily,
        and the best-performing model for each horizon is used automatically.
        """)

    st.markdown("---")
    with st.expander("Model performance"):
        latest_metrics = latest_preds[["horizon", "model_used", "model_version", "model_r2", "model_mae", "model_rmse"]].copy()
        latest_metrics["horizon"] = latest_metrics["horizon"].map(lambda h: HORIZON_LABELS.get(h, h))
        latest_metrics.columns = ["Horizon", "Model", "Version", "R²", "MAE", "RMSE"]
        st.dataframe(latest_metrics.set_index("Horizon"), use_container_width=True)

if __name__ == "__main__":
    main()