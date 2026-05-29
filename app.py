import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for
from ml.engine import ForecastEngine

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'input.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

app = Flask(__name__)
engine = ForecastEngine(data_path=DATA_PATH, models_dir=MODELS_DIR, logs_dir=LOGS_DIR)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    return render_template('index.html')


@app.route('/forecast')
def forecast_page():
    return render_template('forecast.html')


@app.route('/compare')
def compare_page():
    return render_template('compare.html')


@app.route('/explain')
def explain_page():
    return render_template('explain.html')


@app.route('/observe')
def observe_page():
    return render_template('observe.html')


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/api/status')
def api_status():
    engine.status['api_requests'] += 1
    return jsonify(engine.status_report())


@app.route('/api/model-summary')
def api_model_summary():
    engine.status['api_requests'] += 1
    return jsonify(engine.summary_report())


@app.route('/api/train', methods=['POST'])
def api_train():
    engine.status['api_requests'] += 1
    engine.load_dataset()
    try:
        result = engine.train_all()
        return jsonify({'status': 'success', 'result': result})
    except RuntimeError as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 400
    except Exception as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    engine.status['api_requests'] += 1
    engine.load_dataset()
    data = request.json or {}
    periods = int(data.get('periods', 30))
    horizon = data.get('horizon', 'days')
    try:
        if engine.best_model is None:
            if not engine.status['dataset_loaded']:
                raise RuntimeError('No dataset loaded. Upload a CSV dataset before predicting.')
            engine.append_log('Best model missing; auto-training before prediction.')
            engine.train_all()
        forecast = engine.predict(periods=periods, horizon=horizon)
        return jsonify({'status': 'success', 'forecast': forecast})
    except RuntimeError as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 400
    except Exception as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/logs')
def api_logs():
    engine.status['api_requests'] += 1
    log_path = os.path.join(LOGS_DIR, 'engine.log')
    if not os.path.exists(log_path):
        return jsonify({'status': 'success', 'logs': []})
    with open(log_path, 'r', encoding='utf-8') as handle:
        lines = handle.readlines()[-20:]
    return jsonify({'status': 'success', 'logs': [line.strip() for line in lines]})


@app.route('/api/historical')
def api_historical():
    engine.status['api_requests'] += 1
    return jsonify(engine.historical_report())


@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'dataset' not in request.files:
        return jsonify({'status': 'error', 'message': 'Dataset file missing'}), 400
    dataset = request.files['dataset']
    if dataset.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    dataset.save(DATA_PATH)
    engine.load_dataset()
    return jsonify({'status': 'success', 'message': 'Dataset uploaded successfully', 'summary': engine.dataset_summary})


@app.route('/api/health')
def api_health():
    return jsonify({
        'status': 'running',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'services': {
            'dataset': bool(engine.df is not None),
            'model': bool(engine.best_model is not None),
            'forecast_engine': True,
        },
        'metrics': engine.status_report(),
    })


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
