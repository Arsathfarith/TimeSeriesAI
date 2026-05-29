import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .utils import score_metrics


def _split_series(df, target_col, test_frac=0.2):
    split_index = int(len(df) * (1 - test_frac))
    return df.iloc[:split_index], df.iloc[split_index:]


def _future_dates(last_date, periods, freq):
    offset = pd.tseries.frequencies.to_offset(freq)
    start = last_date + offset
    return pd.date_range(start=start, periods=periods, freq=freq)


def train_linear_regression(df, target_col, feature_cols, freq):
    train_df, val_df = _split_series(df, target_col)
    X_train = train_df[['time_index'] + feature_cols] if feature_cols else train_df[['time_index']]
    X_val = val_df[['time_index'] + feature_cols] if feature_cols else val_df[['time_index']]
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    metrics = score_metrics(y_val, predictions)
    return {
        'name': 'Linear Regression',
        'type': 'linear_regression',
        'model': model,
        'metrics': metrics,
        'validation_index': val_df.index.tolist(),
        'validation_target': y_val.tolist(),
        'validation_prediction': predictions.tolist(),
        'feature_cols': feature_cols,
        'freq': freq,
    }


def forecast_linear_regression(model, df, target_col, feature_cols, periods, freq):
    last_index = int(df['time_index'].iloc[-1])
    future_time = np.arange(last_index + 1, last_index + periods + 1)
    future_data = pd.DataFrame({'time_index': future_time})
    for col in feature_cols:
        future_data[col] = df[col].iloc[-1]
    forecast = model.predict(future_data[['time_index'] + feature_cols] if feature_cols else future_data[['time_index']])
    return _future_dates(df.index[-1], periods, freq), forecast.tolist()


def train_moving_average(df, target_col, feature_cols, freq):
    window = min(12, len(df) // 4 or 1)
    train_df, val_df = _split_series(df, target_col)
    history = train_df[target_col].rolling(window=window, min_periods=1).mean()
    predictions = np.full(len(val_df), history.iloc[-1])
    metrics = score_metrics(val_df[target_col].values, predictions)
    return {
        'name': 'Moving Average',
        'type': 'moving_average',
        'model': {'window': window, 'last_value': float(history.iloc[-1])},
        'metrics': metrics,
        'validation_index': val_df.index.tolist(),
        'validation_target': val_df[target_col].tolist(),
        'validation_prediction': predictions.tolist(),
        'feature_cols': feature_cols,
        'freq': freq,
    }


def forecast_moving_average(model, df, target_col, feature_cols, periods, freq):
    forecast = np.full(periods, model['last_value'])
    return _future_dates(df.index[-1], periods, freq), forecast.tolist()


def train_arima(df, target_col, feature_cols, freq):
    train_df, val_df = _split_series(df, target_col)
    order = (1, 1, 1)
    series = train_df[target_col].astype(float)
    try:
        model = ARIMA(series, order=order).fit()
        predictions = model.forecast(steps=len(val_df))
        metrics = score_metrics(val_df[target_col].values, predictions.values)
        return {
            'name': 'ARIMA',
            'type': 'arima',
            'model': model,
            'metrics': metrics,
            'validation_index': val_df.index.tolist(),
            'validation_target': val_df[target_col].tolist(),
            'validation_prediction': predictions.tolist(),
            'feature_cols': feature_cols,
            'freq': freq,
        }
    except Exception:
        return train_moving_average(df, target_col, feature_cols, freq)


def forecast_arima(model, df, target_col, feature_cols, periods, freq):
    forecast = model.forecast(steps=periods)
    return _future_dates(df.index[-1], periods, freq), forecast.tolist()


def train_sarima(df, target_col, feature_cols, freq):
    train_df, val_df = _split_series(df, target_col)
    seasonal = 12 if 'M' in freq or 'Y' in freq else 7
    try:
        model = SARIMAX(train_df[target_col].astype(float), order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        predictions = model.forecast(steps=len(val_df))
        metrics = score_metrics(val_df[target_col].values, predictions.values)
        return {
            'name': 'SARIMA',
            'type': 'sarima',
            'model': model,
            'metrics': metrics,
            'validation_index': val_df.index.tolist(),
            'validation_target': val_df[target_col].tolist(),
            'validation_prediction': predictions.tolist(),
            'feature_cols': feature_cols,
            'freq': freq,
        }
    except Exception:
        return train_arima(df, target_col, feature_cols, freq)


def forecast_sarima(model, df, target_col, feature_cols, periods, freq):
    forecast = model.forecast(steps=periods)
    return _future_dates(df.index[-1], periods, freq), forecast.tolist()


