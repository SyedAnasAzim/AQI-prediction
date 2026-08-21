import os
import time
import requests
import numpy as np
import pandas as pd
import json
from datetime import date, timedelta, datetime
import hopsworks
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

AIR_QUALITY_VARS = "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,dust,ammonia,us_aqi"
WEATHER_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure,precipitation,cloud_cover"

CITIES = {
    "Karachi": (24.8607, 67.0011),
    # "Lahore": (31.5497, 74.3436),
    # "Islamabad": (33.6844, 73.0479),
}

params_aqi = {
            "latitude": CITIES["Karachi"][0],
            "longitude": CITIES["Karachi"][1],
            "current": AIR_QUALITY_VARS,
            "timezone": "auto",
        }

params_weather = {
            "latitude": CITIES["Karachi"][0],
            "longitude": CITIES["Karachi"][1],
            "current": WEATHER_VARS,
            "timezone": "auto",
        }


resp_aqi = requests.get(AIR_QUALITY_URL,params=params_aqi)
resp_weather = requests.get(WEATHER_URL,params=params_weather)

resp_aqi.raise_for_status()
resp_weather.raise_for_status()

df_aqi = pd.DataFrame([resp_aqi.json()["current"]])
df_aqi = df_aqi.rename(columns={"time": "timestamp"})
df_aqi["timestamp"] = pd.to_datetime(df_aqi["timestamp"])

df_weather = pd.DataFrame([resp_weather.json()["current"]])
df_weather = df_weather.rename(columns={"time": "timestamp"})
df_weather["timestamp"] = pd.to_datetime(df_weather["timestamp"]).dt.floor('h')

df_merged = df_aqi.merge(df_weather,on="timestamp")
assert len(df_merged) > 0, "Merge produced no rows — timestamp mismatch between AQI and weather calls"

df_merged.drop(columns=["interval_x","interval_y"],inplace=True)
df_merged["city"] = "Karachi"
df_merged["ammonia"] = df_merged["ammonia"].replace({None: np.nan})
print(df_merged)


print(json.dumps(resp_aqi.json()["current"],indent=4))
print(json.dumps(resp_weather.json()["current"],indent=4))

project = hopsworks.login(
    project = 'PearlsAQI_Project',  # Replace with your project name
    host = "eu-west.cloud.hopsworks.ai",
    port = 443,
    api_key_value = api_key # Get from Hopsworks UI > Account Settings > API Keys
)

# Access the feature store
fs = project.get_feature_store()

# Example: Read an existing feature group
fg = fs.get_feature_group(
    name="karachi_aqi",
    version=1
)

# Read data from the feature group


fg.insert(df_merged)

print("New row has been added")
