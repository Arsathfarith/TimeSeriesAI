const charts = {};

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  return response.json();
}

function showToast(message) {
  if (window.White && window.White.toast) {
    window.White.toast(message);
  }
}

function buildChart(canvasId, config) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  if (charts[canvasId]) charts[canvasId].destroy();
  charts[canvasId] = new Chart(ctx, config);
  return charts[canvasId];
}

function renderStatusPanel(status) {
  const element = document.getElementById('service-status');
  if (!element) return;
  element.textContent = status.model_selected ? 'Online • Model ready' : 'Waiting for dataset or training';
}

function buildSummary(summary) {
  const panel = document.getElementById('summary-panel');
  if (!panel) return;
  if (!summary || !summary.dataset_summary) {
    panel.innerHTML = '<p>No dataset loaded yet.</p>';
    return;
  }
  const ds = summary.dataset_summary;
  panel.innerHTML = `
    <div class="summary-grid">
      <p><strong>Rows:</strong> ${ds.rows}</p>
      <p><strong>Target:</strong> ${ds.target_column}</p>
      <p><strong>Frequency:</strong> ${ds.frequency}</p>
      <p><strong>Trend:</strong> ${ds.trend_direction}</p>
      <p><strong>Missing values:</strong> ${ds.missing_values}</p>
      <p><strong>Range:</strong> ${ds.start_date} → ${ds.end_date}</p>
    </div>`;
}

function buildBestModelPanel(summary) {
  const panel = document.getElementById('best-model-panel');
  if (!panel) return;
  if (!summary || !summary.best_model) {
    panel.innerHTML = '<p>No model selected yet. Train forecasting models to see performance.</p>';
    return;
  }
  const model = summary.best_model;
  const best = summary.models?.find((m) => m.type === model.type) || model;
  panel.innerHTML = `
    <div class="model-card">
      <p class="label">${best.name}</p>
      <p><strong>RMSE:</strong> ${best.metrics.rmse.toFixed(3)}</p>
      <p><strong>MAPE:</strong> ${best.metrics.mape.toFixed(2)}%</p>
      <p><strong>Selected:</strong> ${model.name}</p>
    </div>`;
}

function buildAccuracyChart(summary) {
  const labels = (summary.models || []).map((model) => model.name);
  const rmse = (summary.models || []).map((model) => model.metrics.rmse);
  const mape = (summary.models || []).map((model) => model.metrics.mape);
  if (!labels.length) return;
  buildChart('accuracy-chart', {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'RMSE', data: rmse, backgroundColor: '#37d8f0' },
        { label: 'MAPE', data: mape, backgroundColor: '#6f72ff' },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'bottom' },
        title: { display: true, text: 'Model accuracy comparison' },
      },
    },
  });
}

function buildTrendChart(historical) {
  const labels = (historical.history || []).map((item) => item.date);
  const actual = (historical.history || []).map((item) => item.value);
  const predicted = historical.validation?.predicted || [];
  const validLabels = historical.validation?.dates || [];

  if (!labels.length) return;
  const datasets = [
    {
      label: 'History',
      data: actual,
      borderColor: '#37d8f0',
      backgroundColor: 'rgba(55, 216, 240, 0.15)',
      fill: true,
      tension: 0.2,
    },
  ];

  if (predicted.length && validLabels.length) {
    datasets.push({
      label: 'Validation prediction',
      data: Array(labels.length - predicted.length).fill(null).concat(predicted),
      borderColor: '#ff8f6f',
      backgroundColor: 'rgba(255, 143, 111, 0.15)',
      fill: false,
      borderDash: [6, 4],
      tension: 0.2,
    });
  }

  buildChart('trend-chart', {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'bottom' },
        title: { display: true, text: 'Forecast trend vs history' },
      },
      scales: { x: { display: true }, y: { beginAtZero: false } },
    },
  });
}

function buildSeasonalityChart(summary) {
  if (!summary || !summary.dataset_summary) return;
  const trendDirection = summary.dataset_summary.trend_direction || 'flat';
  const values = trendDirection === 'increasing' ? [40, 45, 15] : [45, 35, 20];
  buildChart('seasonality-chart', {
    type: 'doughnut',
    data: {
      labels: ['Trend', 'Seasonality', 'Noise'],
      datasets: [{ data: values, backgroundColor: ['#37d8f0', '#6f72ff', '#2d3142'] }],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'bottom' }, title: { display: true, text: 'Seasonal signal' } },
    },
  });
}

function buildModelComparison(summary) {
  const table = document.getElementById('compare-table')?.querySelector('tbody');
  if (table) {
    table.innerHTML = '';
    (summary.models || []).forEach((model) => {
      const row = document.createElement('tr');
      row.innerHTML = `
        <td>${model.name}</td>
        <td>${model.metrics.rmse.toFixed(3)}</td>
        <td>${model.metrics.mae.toFixed(3)}</td>
        <td>${model.metrics.mape.toFixed(2)}%</td>`;
      table.appendChild(row);
    });
  }
  const labels = (summary.models || []).map((model) => model.name);
  const data = (summary.models || []).map((model) => model.metrics.rmse);
  if (labels.length) {
    buildChart('compare-chart', {
      type: 'bar',
      data: {
        labels,
        datasets: [{ label: 'RMSE', data, backgroundColor: '#37d8f0' }],
      },
      options: { responsive: true, plugins: { legend: { display: false } } },
    });
  }
}

async function loadDashboard() {
  let summary = {};
  let status = {};
  let historical = { history: [] };

  try {
    [summary, status] = await Promise.all([fetchJson('/api/model-summary'), fetchJson('/api/status')]);
  } catch (error) {
    console.error('Dashboard summary/status load failed', error);
    document.getElementById('summary-panel')?.insertAdjacentHTML('afterbegin', '<p>Failed to load dataset metrics.</p>');
    return;
  }

  try {
    historical = await fetchJson('/api/historical');
  } catch (error) {
    console.warn('Historical data load failed', error);
    historical = { history: [] };
  }

  renderStatusPanel(status);
  buildSummary(summary);
  buildBestModelPanel(summary);
  buildModelComparison(summary);
  buildAccuracyChart(summary);
  buildSeasonalityChart(summary);
}

function initPage() {
  const path = window.location.pathname.replace(/\/$/, '');
  if (path === '' || path === '/' || path === '/dashboard') loadDashboard();
  if (path === '/forecast') loadForecastPage();
  if (path === '/compare') loadComparePage();
  if (path === '/explain') loadExplainPage();
  if (path === '/observe') loadObservePage();
  if (path === '/upload') loadUploadPage();
}

async function loadForecastPage() {
  const status = await fetchJson('/api/status');
  renderStatusPanel(status);
  const form = document.getElementById('predict-form');
  if (!form) return;
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const periods = document.getElementById('periods').value;
    const horizon = document.getElementById('horizon').value;
    const result = await fetchJson('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ periods, horizon }),
    });
    if (result.status !== 'success') {
      showToast(result.message || 'Prediction failed');
      return;
    }
    showToast('Forecast generated successfully');
    const summaryPanel = document.getElementById('forecast-summary');
    const requested = result.requested_horizon || result.forecast?.requested_horizon || { count: periods, unit: horizon };
    summaryPanel.innerHTML = `<p><strong>${result.forecast.model.name}</strong> forecast generated for ${requested.count} ${requested.unit} (${requested.steps || result.forecast.forecast.length} periods).</p>`;
    const labels = result.forecast.forecast.map((item) => item.date);
    const values = result.forecast.forecast.map((item) => item.prediction);
    buildChart('forecast-chart', {
      type: 'line',
      data: {
        labels,
        datasets: [{ label: 'Forecast', data: values, borderColor: '#6f72ff', fill: false }],
      },
      options: { responsive: true, plugins: { legend: { display: false } } },
    });
    const tbody = document.getElementById('forecast-table')?.querySelector('tbody');
    if (tbody) {
      tbody.innerHTML = '';
      result.forecast.forecast.forEach((row) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${row.date}</td><td>${row.prediction.toFixed(3)}</td>`;
        tbody.appendChild(tr);
      });
    }
  });
}

async function loadComparePage() {
  const summary = await fetchJson('/api/model-summary');
  buildModelComparison(summary);
}

async function loadExplainPage() {
  const summary = await fetchJson('/api/model-summary');
  const explainText = document.getElementById('explain-text');
  const explainList = document.getElementById('explain-list');
  if (!summary || !summary.dataset_summary) {
    explainText.innerHTML = '<p>Upload your dataset and train models to generate explainability insights.</p>';
    return;
  }
  const ds = summary.dataset_summary;
  const message = ds.trend_direction === 'increasing'
    ? 'The series shows an upward trajectory and the best model captures growth with seasonality adjustments.'
    : 'The series trend is falling and the forecast is driven by recent downward movement and seasonal patterns.';
  explainText.innerHTML = `<p>${message}</p>`;
  explainList.innerHTML = `
    <li>Trend direction: ${ds.trend_direction}</li>
    <li>Best target column: ${ds.target_column}</li>
    <li>Detected frequency: ${ds.frequency}</li>
    <li>Missing values automatically imputed with forward-fill.</li>
  `;
  buildChart('explain-chart', {
    type: 'doughnut',
    data: {
      labels: ['Trend', 'Seasonality', 'Residual'],
      datasets: [{ data: [45, 35, 20], backgroundColor: ['#37d8f0', '#6f72ff', '#2d3142'] }],
    },
    options: { responsive: true, plugins: { legend: { position: 'bottom' } } },
  });
}

async function loadObservePage() {
  const [health, logs] = await Promise.all([fetchJson('/api/health'), fetchJson('/api/logs')]);
  const panel = document.getElementById('health-panel');
  const list = document.getElementById('health-list');
  const logStream = document.getElementById('log-stream');
  if (panel) panel.innerHTML = `<p>Service status: ${health.status}</p><p>${health.metrics.dataset_summary ? 'Dataset loaded.' : 'Dataset not loaded.'}</p>`;
  if (list) {
    list.innerHTML = `
      <li>API requests: ${health.metrics.api_requests}</li>
      <li>Prediction runs: ${health.metrics.predictions}</li>
      <li>Model trained: ${health.services.model}</li>
      <li>Last training: ${health.metrics.last_train || 'n/a'}</li>
    `;
  }
  if (logStream) {
    logStream.textContent = logs.logs.length ? logs.logs.join('\n') : 'No logs available yet.';
  }
  buildChart('health-chart', {
    type: 'polarArea',
    data: {
      labels: ['API calls', 'Predictions', 'Train runs'],
      datasets: [{ data: [health.metrics.api_requests || 0, health.metrics.predictions || 0, health.metrics.train_runs || 0], backgroundColor: ['#37d8f0', '#6f72ff', '#20c997'] }],
    },
    options: { responsive: true, plugins: { legend: { position: 'right' } } },
  });
}

async function loadUploadPage() {
  const status = await fetchJson('/api/status');
  renderStatusPanel(status);
  const uploadForm = document.getElementById('upload-form');
  const trainButton = document.getElementById('train-button');
  const statusBox = document.getElementById('dataset-details');
  const logPanel = document.getElementById('training-log');

  uploadForm?.addEventListener('submit', async (event) => {
    event.preventDefault();
    const formData = new FormData(uploadForm);
    const result = await fetch('/api/upload', { method: 'POST', body: formData });
    const payload = await result.json();
    if (payload.status !== 'success') {
      showToast(payload.message || 'Upload failed');
      return;
    }
    showToast('Dataset uploaded successfully');
    statusBox.innerHTML = `<p>Dataset uploaded. ${payload.summary.rows} rows with target ${payload.summary.target_column}.</p>`;
    logPanel.textContent = 'Dataset loaded. Ready to train.';
  });

  trainButton?.addEventListener('click', async () => {
    const payload = await fetchJson('/api/train', { method: 'POST' });
    if (payload.status !== 'success') {
      showToast(payload.message || 'Training failed');
      return;
    }
    showToast('Training pipeline completed');
    statusBox.innerHTML = `<p>Training completed with best model ${payload.result.best_model.name}.</p>`;
    logPanel.textContent = `Best model: ${payload.result.best_model.name}\n${JSON.stringify(payload.result.models, null, 2)}`;
  });
}

function initPage() {
  const path = window.location.pathname;
  if (path === '/' || path === '/dashboard') loadDashboard();
  if (path === '/forecast') loadForecastPage();
  if (path === '/compare') loadComparePage();
  if (path === '/explain') loadExplainPage();
  if (path === '/observe') loadObservePage();
  if (path === '/upload') loadUploadPage();
}

window.addEventListener('DOMContentLoaded', initPage);
