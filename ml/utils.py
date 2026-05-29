import os
import json
import pickle
import numpy as np
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle)


def load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


def mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    if not mask.any():
        return float('nan')
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def score_metrics(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    return {
        'mae': float(mean_absolute_error(actual, predicted)),
        'rmse': float(rmse(actual, predicted)),
        'mape': float(mape(actual, predicted)),
    }
