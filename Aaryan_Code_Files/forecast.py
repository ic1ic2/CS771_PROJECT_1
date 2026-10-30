"""Part 5 submission: next-day (24 h) PM2.5 forecast for all 468 Bihar stations.

The folder must contain: forecast.py, features.py, model.npz.
model.npz holds the fitted model as plain numpy arrays (feature-builder statistics, PCA and k-means results,
ridge coefficients, and every boosted tree's splits and leaf values). It is read with numpy alone, so it works
under any scikit-learn version. forecast() only computes features from `history` and applies the frozen
models (mean of ridge and boosted trees, both predicting the 24 h change); it never refits anything.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import features as F  # noqa: E402

MODEL = F.PortableModel(HERE / "model.npz")      # loaded once, at import time
SENSORS = MODEL.sensor_ids
LOOKBACK = MODEL.lookback


def forecast(history: pd.DataFrame, issue_time: pd.Timestamp) -> pd.Series:
    """history: wide hourly table (DatetimeIndex 'timestamp', one string column per sensor_id),
    every hour from 1 Jan 2024 00:00 up to and including issue_time.
    issue_time: 12:00 on some day (IST).
    Returns predicted PM2.5 for issue_time + 24 h, indexed by sensor_id (all 468 stations)."""
    issue_time = pd.Timestamp(issue_time)
    hist = history.copy()
    hist.columns = hist.columns.astype(str)
    hist = hist.loc[:issue_time]                                   # never look past the issue time

    # The last LOOKBACK + 1 hours on a regular hourly grid, in the training column order
    hours = pd.date_range(issue_time - pd.Timedelta(hours=LOOKBACK), issue_time, freq="h")
    win = hist.reindex(columns=SENSORS).reindex(hours)
    if win.isna().to_numpy().any():                                # defensive: gaps -> carry last value forward
        win = win.ffill().bfill()
        win = win.fillna(pd.Series(MODEL.builder.station_mean_, index=SENSORS))
    Vw = win.to_numpy(dtype=float)
    idx = np.array([LOOKBACK])                                      # issue time is the last row

    X, _ = MODEL.builder.transform(Vw, idx, win.index)
    pred = MODEL.predict_change(X) + Vw[LOOKBACK]
    pred = np.clip(np.nan_to_num(pred, nan=0.0), 0, None)
    return pd.Series(pred, index=SENSORS, name="pm25_forecast")
