# 🌫️ Karachi AQI Prediction & Forecasting

An end-to-end machine learning system for **Air Quality Index (AQI) prediction and forecasting in Karachi, Pakistan**.

The project combines historical air-quality and weather data, feature engineering, machine learning models, explainable AI, cloud-based feature storage, and automated inference to provide an interactive AQI forecasting dashboard.

---

## 📌 Overview

Air pollution is a major environmental concern in densely populated cities such as Karachi. This project aims to predict future AQI levels using historical air-quality measurements and meteorological conditions.

The system is designed as a complete ML pipeline:

**Data Collection → Data Processing → Feature Engineering → Model Training → Prediction → Explainability → Logging → Dashboard**

The project also uses **GitHub Actions** to automate recurring data updates and prediction workflows.

---

## ✨ Features

* 📊 Historical AQI and air-quality data processing
* 🌦️ Weather and meteorological feature integration
* 🧠 Machine learning based AQI forecasting
* 🔮 Multi-horizon future AQI predictions
* 🔍 SHAP-based model explainability
* 📈 Interactive Streamlit dashboard
* 🗄️ Hopsworks Feature Store integration
* ⚙️ Automated prediction pipeline
* 🤖 GitHub Actions workflow automation
* 📝 Prediction logging
* 🔄 Automated data updates
* 📊 Interactive charts and visualizations
* 🔁 Retry and recovery logic for external service connections

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │   Historical Data    │
                    │ AQI + Weather Data   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Data Preprocessing  │
                    │ Cleaning & Resampling │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Engineering  │
                    │ Temporal + Weather   │
                    │ Air Quality Features │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Model Training    │
                    │  ML Regression Model │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Inference       │
                    │ Future AQI Forecasts │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    ▼                      ▼
          ┌─────────────────┐    ┌─────────────────┐
          │ SHAP Explainability│ │ Prediction Log  │
          └────────┬────────┘    └────────┬────────┘
                   │                      │
                   └──────────┬───────────┘
                              ▼
                    ┌──────────────────────┐
                    │ Streamlit Dashboard  │
                    │ AQI Forecast + SHAP  │
                    └──────────────────────┘
```

---

## 📂 Project Structure

```text
AQI-prediction/
│
├── .github/
│   └── workflows/
│       └── GitHub Actions workflows
│
├── Notebook/
│   └── Jupyter notebooks for analysis and experimentation
│
├── scripts/
│   ├── feature_engineering.py
│   ├── historical_data_script.py
│   ├── inference.py
│   ├── model_training.py
│   ├── prediction_logger.py
│   ├── update_data.py
│   ├── requirements.txt
│   ├── requirements_inference.txt
│   └── requirements_update_data.txt
│
├── app.py
├── predictions_log.csv
├── requirements.txt
└── README.md
```

---

## 🧩 Main Components

### `app.py`

The main **Streamlit dashboard**.

It provides the user interface for viewing AQI forecasts, historical information, visualizations, and prediction drivers. The application connects to the Hopsworks Feature Store to retrieve project data and includes retry logic for connection failures.

### `scripts/historical_data_script.py`

Handles retrieval and preparation of historical air-quality and weather data.

### `scripts/update_data.py`

Responsible for updating the dataset with newly available data.

### `scripts/feature_engineering.py`

Creates the features required by the forecasting models, including temporal and environmental features.

### `scripts/model_training.py`

Contains the model training pipeline used to train the AQI prediction model.

### `scripts/inference.py`

Runs the trained model to generate future AQI predictions.

### `scripts/prediction_logger.py`

Stores generated predictions in the prediction log for tracking and analysis.

### `predictions_log.csv`

Contains recorded model predictions produced by the inference pipeline.

---

## 📊 Data

The project uses air-quality and meteorological information for **Karachi**.

The dataset includes variables such as:

### Air Quality

* PM2.5
* PM10
* Carbon Monoxide (CO)
* Nitrogen Dioxide (NO₂)
* Sulphur Dioxide (SO₂)
* Ozone (O₃)
* Dust
* US AQI

### Weather

* Temperature
* Relative Humidity
* Wind Speed
* Wind Direction
* Surface Pressure
* Precipitation
* Cloud Cover

Temporal information is also used to capture patterns related to:

* Hour of day
* Day of week
* Month
* Weekend/weekday
* Seasonal patterns

---

## 🤖 Machine Learning Pipeline

The machine learning pipeline follows these major stages:

### 1. Data Collection

Historical AQI and weather data are collected and maintained for Karachi.

### 2. Data Preprocessing

The data is cleaned, sorted chronologically, resampled to the required time frequency, and missing observations are handled.

### 3. Feature Engineering

Relevant temporal, weather, and air-quality features are generated.

### 4. Model Training

The processed dataset is used to train regression models for AQI forecasting.

### 5. Inference

The trained model generates AQI predictions for future timestamps across multiple forecasting horizons.

### 6. Explainability

SHAP values are used to identify which features contribute most to individual predictions.

---

## 🔍 Explainable AI with SHAP

The project incorporates **SHAP (SHapley Additive exPlanations)** to make the model predictions easier to understand.

Instead of only showing the predicted AQI, the dashboard can show the major factors influencing a prediction.

For example:

```text
Prediction
   │
   ├── PM2.5             ─────────►
   ├── PM10              ───────►
   ├── Temperature       ───►
   ├── Humidity          ─────►
   └── Wind Speed        ──►
```

This allows users to understand **why the model produced a particular forecast** rather than treating the prediction as a black box.

---

## 🖥️ Dashboard

The project includes an interactive **Streamlit dashboard**.

The dashboard is designed to provide:

* Current AQI information
* Future AQI forecasts
* Multiple prediction horizons
* Interactive plots
* Prediction history
* Model prediction drivers
* SHAP explanations
* Air-quality trends

The dashboard uses Plotly for interactive visualizations.

---

## 🗄️ Hopsworks Feature Store

The project uses **Hopsworks** as part of its data and feature management pipeline.

The Streamlit application connects to the Hopsworks Feature Store to retrieve the required data for the dashboard.

Sensitive credentials such as API keys should be stored as environment variables or Streamlit secrets rather than being committed to the repository.

Example:

```text
HOPSWORKS_API_KEY=your_api_key
```

**Never commit your actual API key to GitHub.**

---

## ⚙️ Automated ML Pipeline

GitHub Actions is used to automate recurring parts of the pipeline.

The automated workflow can perform tasks such as:

```text
Scheduled Trigger
       │
       ▼
Update Data
       │
       ▼
Feature Engineering
       │
       ▼
Run Inference
       │
       ▼
Generate Predictions
       │
       ▼
Update Prediction Log
```

This allows the forecasting system to operate with minimal manual intervention.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/SyedAnasAzim/AQI-prediction.git
cd AQI-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Dashboard

Start the Streamlit application with:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🔐 Environment Variables

Some components require credentials for external services.

Create a `.env` file for local development or configure secrets through your deployment platform.

Example:

```env
HOPSWORKS_API_KEY=your_api_key
```

Do not commit `.env` files or API keys to the repository.

---

## 📈 Prediction Horizons

The forecasting system generates predictions for multiple future time horizons.

Users can select a forecast horizon from the dashboard and inspect:

* Predicted AQI
* Target timestamp
* Prediction history
* Prediction drivers
* SHAP feature contributions

---

## 🛠️ Technologies Used

| Technology       | Purpose                         |
| ---------------- | ------------------------------- |
| Python           | Core programming language       |
| Pandas           | Data processing                 |
| NumPy            | Numerical computing             |
| Scikit-learn     | Machine learning                |
| SHAP             | Model explainability            |
| Streamlit        | Web dashboard                   |
| Plotly           | Interactive visualization       |
| Hopsworks        | Feature Store & Model Registery |
| GitHub Actions   | Workflow automation             |
| Jupyter Notebook | Data analysis & experimentation |

---

## 📚 Project Goals

The main goals of this project are to:

1. Build a reliable AQI forecasting pipeline for Karachi.
2. Incorporate both air-quality and meteorological information.
3. Produce forecasts across multiple future horizons.
4. Make predictions interpretable using SHAP.
5. Automate recurring data and inference workflows.
6. Provide an accessible dashboard for exploring AQI forecasts.

---

## 🔮 Future Improvements

Potential future improvements include:

* Improving long-term forecasting accuracy
* Testing additional ML and deep learning architectures
* Adding uncertainty/confidence intervals
* Incorporating additional environmental variables
* Adding real-time alerts for unhealthy AQI levels
* Expanding predictions to additional locations
* Improving model monitoring
* Adding automated model retraining
* Improving dashboard performance and design

---

## ⚠️ Disclaimer

This project is intended for **educational, research, and demonstration purposes**.

AQI predictions are model estimates and should not be treated as official environmental or medical guidance.

---

## 👨‍💻 Author

**Syed Anas Azim**

Computer & Information Systems Engineering Student

GitHub: [@SyedAnasAzim](https://github.com/SyedAnasAzim)

---

## 📄 License

This project does not currently specify a license.

If you intend to allow others to freely use, modify, and distribute the project, consider adding an appropriate open-source license.
