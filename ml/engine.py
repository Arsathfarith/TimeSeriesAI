import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from .preprocessing import preprocess_dataset, summarize_dataset
from .utils import save_pickle, save_json, load_json, load_pickle
from .models import (
    train_linear_regression,
    train_moving_average,
    train_arima,
    train_sarima,
    forecast_linear_regression,
    forecast_moving_average,
    forecast_arima,
    forecast_sarima,
)


class ForecastEngine:
    def __init__(self, data_path='data/input.csv', models_dir='models', logs_dir='logs'):
        self.data_path = data_path
        self.models_dir = models_dir
        self.logs_dir = logs_dir
        self.summary_path = os.path.join(self.models_dir, 'summary.json')
        self.model_path = os.path.join(self.models_dir, 'best_model.pkl')
        self.model_meta_path = os.path.join(self.models_dir, 'model_meta.json')
        self.status = {
            'api_requests': 0,
            'predictions': 0,
            'train_runs': 0,
            'dataset_loaded': False,
            'model_selected': False,
            'last_train': None,
        }
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        self.df = None
        self.best_model = None
        self.best_meta = None
        self.dataset_summary = {}
        self.summary = {}
        self.load_dataset()
        self.load_summary()

    def find_dataset_file(self):
        if os.path.exists(self.data_path):
            return self.data_path
        candidate_dir = os.path.dirname(self.data_path)
        if os.path.isdir(candidate_dir):
            candidates = [
                os.path.join(candidate_dir, fname)
                for fname in os.listdir(candidate_dir)
                if fname.lower().endswith('.csv')
            ]
            if candidates:
                return sorted(candidates)[0]
        return None

    def load_dataset(self):
        data_file = self.find_dataset_file()
        if data_file is None:
            self.create_sample_dataset()
            data_file = self.data_path

        if data_file and os.path.exists(data_file):
            self.df, self.target_col, self.feature_cols, self.freq = preprocess_dataset(data_file)
            self.dataset_summary = summarize_dataset(self.df, self.target_col)
            self.status['dataset_loaded'] = True
            self.data_path = data_file
        else:
            self.df = None
            self.status['dataset_loaded'] = False

    def create_sample_dataset(self):
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        dates = pd.date_range(end=pd.Timestamp.today(), periods=90, freq='D')
        values = 100 + np.arange(len(dates)) * 0.4 + np.sin(np.arange(len(dates)) / 6.0) * 10
        sample_df = pd.DataFrame({'date': dates, 'value': values})
        sample_df.to_csv(self.data_path, index=False)
        self.append_log(f'Created sample dataset at {self.data_path}')
        return self.data_path

    def load_summary(self):
        if os.path.exists(self.summary_path):
            try:
                self.summary = load_json(self.summary_path)
                self.best_meta = load_json(self.model_meta_path) if os.path.exists(self.model_meta_path) else None
                if os.path.exists(self.model_path):
                    self.best_model = load_pickle(self.model_path)
                    self.status['model_selected'] = self.best_meta is not None
            except Exception:
                self.summary = {}
                self.best_model = None
                self.best_meta = None
                self.status['model_selected'] = False

    def append_log(self, message):
        timestamp = datetime.utcnow().isoformat() + 'Z'
        path = os.path.join(self.logs_dir, 'engine.log')
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write(f'[{timestamp}] {message}\n')

    def train_all(self):
        if self.df is None:
            raise RuntimeError('No dataset loaded. Upload a CSV dataset first.')

        self.append_log('Starting full training pipeline.')
        models = [
            train_linear_regression,
            train_moving_average,
            train_arima,
            train_sarima,
        ]
        results = []
        for constructor in models:
            try:
                result = constructor(self.df, self.target_col, self.feature_cols, self.freq)
                results.append(result)
                self.append_log(f"Trained {result['name']} model with RMSE={result['metrics']['rmse']:.4f}.")
            except Exception as exc:
                self.append_log(f"Skipped {constructor.__name__}: {exc}")

        if not results:
            raise RuntimeError('Training failed for all forecasting models.')

        self.summary['models'] = [
            {
                'name': r['name'],
                'type': r['type'],
                'metrics': r['metrics'],
                'feature_cols': r['feature_cols'],
            }
            for r in results
        ]
        self.best_model = min(results, key=lambda r: r['metrics']['rmse'] if r['metrics'] else float('inf'))
        self.best_meta = {
            'type': self.best_model['type'],
            'name': self.best_model['name'],
            'target_col': self.target_col,
            'feature_cols': self.feature_cols,
            'freq': self.freq,
        }
        self.save_best_model()
        self.summary['best_model'] = self.best_meta
        self.summary['dataset_summary'] = self.dataset_summary
        self.summary['trained_at'] = datetime.utcnow().isoformat() + 'Z'
        self.save_summary()
        self.status['train_runs'] += 1
        self.status['last_train'] = self.summary['trained_at']
        self.status['model_selected'] = True
        self.append_log(f"Selected best model {self.best_model['name']}.")
        return self.summary

    def save_best_model(self):
        save_pickle(self.best_model, self.model_path)
        save_json(self.best_meta, self.model_meta_path)

    def save_summary(self):
        save_json(self.summary, self.summary_path)

    def status_report(self):
        return {
            'dataset_loaded': self.status['dataset_loaded'],
            'model_selected': self.status['model_selected'],
            'api_requests': self.status['api_requests'],
            'predictions': self.status['predictions'],
            'train_runs': self.status['train_runs'],
            'last_train': self.status['last_train'],
            'dataset_summary': self.dataset_summary,
        }

    def summary_report(self):
        return self.summary or {
            'dataset_summary': self.dataset_summary,
            'models': [],
            'best_model': self.best_meta,
        }

    def historical_report(self, limit=40):
        if self.df is None:
            return {'history': []}
        series = self.df[[self.target_col]].iloc[-limit:]
        return {
            'history': [
                {'date': str(idx), 'value': float(value)}
                for idx, value in zip(series.index, series[self.target_col])
            ],
            'validation': {
                'dates': self.best_model.get('validation_index', []) if self.best_model else [],
                'actual': self.best_model.get('validation_target', []),
                'predicted': self.best_model.get('validation_prediction', []),
            },
        }

    def _resolve_horizon_steps(self, periods, horizon, freq):
        horizon = str(horizon).lower().strip()
        periods = max(int(periods), 1)
        if horizon not in {'days', 'weeks', 'months', 'years'}:
            horizon = 'days'

        if freq.startswith('D'):
            mapping = {'days': 1, 'weeks': 7, 'months': 30, 'years': 365}
        elif freq.startswith('W'):
            mapping = {'days': 1, 'weeks': 1, 'months': 4, 'years': 52}
        elif freq.startswith('M'):
            mapping = {'days': 1, 'weeks': 1, 'months': 1, 'years': 12}
        else:
            mapping = {'days': 1, 'weeks': 7, 'months': 30, 'years': 365}

        step = mapping.get(horizon, 1)
        result = int(max(1, round(periods * step)))
        self.append_log(f'Resolved forecast horizon {periods} {horizon} to {result} periods at frequency {freq}.')
        return result

    def predict(self, periods=30, horizon='days'):
        if self.best_model is None or self.best_meta is None:
            raise RuntimeError('No trained model available. Train forecasting models first.')

        self.status['predictions'] += 1
        forecast_steps = self._resolve_horizon_steps(periods, horizon, self.freq)
        model_type = self.best_meta['type']
        raw_model = self.best_model['model'] if isinstance(self.best_model, dict) and 'model' in self.best_model else self.best_model
        if model_type == 'linear_regression':
            index, forecast = forecast_linear_regression(raw_model, self.df, self.target_col, self.feature_cols, forecast_steps, self.freq)
        elif model_type == 'moving_average':
            index, forecast = forecast_moving_average(raw_model, self.df, self.target_col, self.feature_cols, forecast_steps, self.freq)
        elif model_type == 'arima':
            index, forecast = forecast_arima(raw_model, self.df, self.target_col, self.feature_cols, forecast_steps, self.freq)
        elif model_type == 'sarima':
            index, forecast = forecast_sarima(raw_model, self.df, self.target_col, self.feature_cols, forecast_steps, self.freq)
        else:
            raise RuntimeError(f'Unsupported model type: {model_type}')

        results = [
            {'date': str(d), 'prediction': float(p)}
            for d, p in zip(index, forecast)
        ]
        self.append_log(f"Generated {forecast_steps} period forecast with {self.best_meta['name']}.")
        metrics = self.best_model['metrics'] if isinstance(self.best_model, dict) and 'metrics' in self.best_model else {}
        return {
            'model': self.best_meta,
            'forecast': results,
            'accuracy': metrics,
            'requested_horizon': {'count': periods, 'unit': horizon, 'steps': forecast_steps},
        }
