import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from dateutil.parser import parse


def infer_date_column(df: pd.DataFrame):
    date_candidates = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    for col in date_candidates:
        parsed = pd.to_datetime(df[col], errors='coerce')
        if parsed.notna().sum() >= len(parsed) * 0.8:
            return col

    for col in df.columns:
        if df[col].dtype == object:
            parsed = pd.to_datetime(df[col], errors='coerce')
            if parsed.notna().sum() >= len(parsed) * 0.8:
                return col
    return None


def is_id_column(col_name: str, values: pd.Series):
    lower = col_name.lower()
    if any(key in lower for key in ['id', 'code', 'postal', 'zip', 'order']):
        return True
    if values.nunique(dropna=True) > len(values) * 0.9:
        return True
    return False


def infer_target_column(df: pd.DataFrame, date_col: str):
    numeric = df.select_dtypes(include=[np.number]).copy()
    if date_col in numeric.columns:
        numeric = numeric.drop(columns=[date_col])
    if numeric.empty:
        raise ValueError('Could not identify a numeric target column for forecasting.')

    preferred = ['sales', 'revenue', 'profit', 'amount', 'quantity', 'demand', 'units']
    for candidate in preferred:
        for col in numeric.columns:
            if candidate in col.lower():
                return col

    valid_columns = [col for col in numeric.columns if not is_id_column(col, numeric[col])]
    if valid_columns:
        scores = numeric[valid_columns].var().sort_values(ascending=False)
        return scores.index[0]

    scores = numeric.var().sort_values(ascending=False)
    return scores.index[0]


def build_time_index(df: pd.DataFrame, date_col: str):
    ordered = df.sort_values(date_col).reset_index(drop=True)
    ordered['time_index'] = np.arange(len(ordered))
    return ordered


def _read_csv_with_fallback(path: str):
    encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin1']
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception:
            continue
    raise UnicodeDecodeError('utf-8', b'', 0, 1, 'Unable to decode file with standard encodings.')


def preprocess_dataset(path: str):
    df = _read_csv_with_fallback(path)
    date_col = infer_date_column(df)
    if date_col is None:
        raise ValueError('No date or time column found in the dataset.')

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    if df[date_col].duplicated().any():
        agg = {}
        for col in df.columns:
            if col == date_col:
                continue
            agg[col] = 'sum' if is_numeric_dtype(df[col]) else 'first'
        df = df.groupby(date_col, as_index=False).agg(agg)
    target_col = infer_target_column(df, date_col)
    df = build_time_index(df, date_col)
    df = df.set_index(date_col)

    freq = pd.infer_freq(df.index)
    if freq is None:
        freq = 'D'
    df = df.asfreq(freq, method='pad')
    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in {target_col, 'time_index'} and not is_id_column(c, df[c])
    ]

    return df, target_col, feature_cols, freq


def summarize_dataset(df: pd.DataFrame, target_col: str):
    summary = {
        'rows': int(df.shape[0]),
        'columns': int(df.shape[1]),
        'target_column': target_col,
        'feature_columns': [c for c in df.columns if c != target_col and c != 'time_index'],
        'missing_values': int(df.isna().sum().sum()),
        'start_date': str(df.index.min()),
        'end_date': str(df.index.max()),
        'frequency': str(pd.infer_freq(df.index) or 'D'),
        'mean': float(df[target_col].mean()),
        'trend_direction': 'increasing' if df[target_col].iloc[-1] > df[target_col].iloc[0] else 'decreasing',
    }
    return summary
