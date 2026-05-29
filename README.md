# TimeSeriesAI Forecasting Platform

A Flask-based end-to-end time series forecasting platform with a White.js-inspired frontend dashboard.

## Features

- Dataset upload and automatic date/target detection
- Multi-model time series training: Linear Regression, Moving Average, ARIMA, SARIMA, Prophet, LSTM
- Automatic best-model selection using RMSE and comparison metrics
- Pickle model persistence and metadata storage
- Dashboard visualizations and forecasting analytics
- Explainability and observability pages
- API health and prediction logs

## Setup

1. Create a Python virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

2. Run the Flask app:
   ```powershell
   python app.py
   ```

3. Open `http://localhost:5000` in your browser.

## Usage

- Upload your CSV dataset on the `Upload Data` page.
- Train all forecasting models using the training panel.
- Explore performance comparisons, forecasts, explainability, and observability.

## Notes

- Ensure your CSV contains a date/time column and at least one numeric target column.
- Forecast results are saved in `models/best_model.pkl` and metadata in `models/model_meta.json`.
- Logs are stored in `logs/engine.log`.
