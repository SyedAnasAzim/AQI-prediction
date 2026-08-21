"""
Backfill historical weather + air quality data from Open-Meteo (free, no API key).

Fetches:
  - Air quality: PM2.5, PM10, CO, NO2, SO2, O3, US AQI
  - Weather: temperature, humidity, wind speed/direction, pressure, precipitation

Merges both on timestamp and saves one CSV per city, ready to feed into your
feature pipeline / Feature Store ingestion step.

Usage:
    python fetch_openmeteo_backfill.py
"""

import time
import requests
import pandas as pd
from datetime import date, timedelta, datetime

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

# Add/edit cities here. (lat, lon) — pooled model uses "city" as a feature.
CITIES = {
    "Karachi": (24.8607, 67.0011),
    # "Lahore": (31.5497, 74.3436),
    # "Islamabad": (33.6844, 73.0479),
}

AIR_QUALITY_VARS = "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,dust,ammonia,us_aqi"
WEATHER_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure,precipitation,cloud_cover"

# Open-Meteo global CAMS data starts ~Aug 2022; adjust if you want a shorter window
START_DATE = "2024-08-01"
END_DATE = date.today().isoformat()

# Air-quality archive is queried in chunks to stay well within response-size limits
CHUNK_DAYS = 90


def daterange_chunks(start_str, end_str, chunk_days):
    start = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        yield current.isoformat(), chunk_end.isoformat()
        current = chunk_end + timedelta(days=1)


def fetch_json(url, params, retries=3, backoff=5):
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"  Attempt {attempt}/{retries} failed: {e}")
            if attempt == retries:
                raise
            time.sleep(backoff)


def fetch_air_quality(lat, lon, start_date, end_date):
    frames = []
    for chunk_start, chunk_end in daterange_chunks(start_date, end_date, CHUNK_DAYS):
        print(f"  Air quality: {chunk_start} to {chunk_end}")
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": chunk_start,
            "end_date": chunk_end,
            "hourly": AIR_QUALITY_VARS,
            "timezone": "auto",
        }
        data = fetch_json(AIR_QUALITY_URL, params)
        df = pd.DataFrame(data["hourly"])
        frames.append(df)
        time.sleep(1)  # be polite to the free API
    return pd.concat(frames, ignore_index=True)


def fetch_weather(lat, lon, start_date, end_date):
    frames = []
    for chunk_start, chunk_end in daterange_chunks(start_date, end_date, CHUNK_DAYS):
        print(f"  Weather: {chunk_start} to {chunk_end}")
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": chunk_start,
            "end_date": chunk_end,
            "hourly": WEATHER_VARS,
            "timezone": "auto",
        }
        data = fetch_json(WEATHER_URL, params)
        df = pd.DataFrame(data["hourly"])
        frames.append(df)
        time.sleep(1)
    return pd.concat(frames, ignore_index=True)


def build_city_dataset(city, lat, lon):
    print(f"\nFetching data for {city} ({lat}, {lon})")
    aq_df = fetch_air_quality(lat, lon, START_DATE, END_DATE)
    weather_df = fetch_weather(lat, lon, START_DATE, END_DATE)
    now = pd.Timestamp.now().floor("h")

    merged = pd.merge(aq_df, weather_df, on="time", how="inner")
    merged["city"] = city
    merged = merged.rename(columns={"time": "timestamp"})
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    merged = merged[merged["timestamp"] <= now]
    # Basic sanity check: drop rows where all pollutant readings are null
    pollutant_cols = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "dust", "ammonia"]
    merged = merged.dropna(subset=pollutant_cols, how="all")

    return merged


def main():
    all_cities = []
    for city, (lat, lon) in CITIES.items():
        df = build_city_dataset(city, lat, lon)
        out_path = f"{city.lower()}_openmeteo_backfill.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} rows -> {out_path}")
        all_cities.append(df)

    if len(all_cities) > 1:
        combined = pd.concat(all_cities, ignore_index=True)
        combined.to_csv("all_cities_openmeteo_backfill.csv", index=False)
        print(f"\nSaved combined dataset: {len(combined)} rows -> all_cities_openmeteo_backfill.csv")


if __name__ == "__main__":
    main()


