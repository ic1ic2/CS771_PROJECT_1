"""Part 5 submission: next-day (24 h) PM2.5 forecast for all 468 Bihar stations.

The folder must contain: forecast.py, features.py, model.pkl.
model.pkl holds the fitted feature builder (station statistics, PCA, k-means, neighbour weights) and the two
fitted models (ridge + shallow boosted trees, both predicting the 24 h change). forecast() only computes
features from `history` and applies the frozen models; it never refits anything.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import features as F  # noqa: E402  (needed to unpickle the feature builder)

BUNDLE = joblib.load(HERE / "model.pkl")          # loaded once, at import time
BUILDER = BUNDLE["builder"]
RIDGE, HGB = BUNDLE["ridge"], BUNDLE["hgb"]
SENSORS = [str(s) for s in BUNDLE["sensor_ids"]]
LOOKBACK = int(BUNDLE["lookback_h"])


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
        win = win.fillna(pd.Series(BUILDER.station_mean_, index=SENSORS))
    Vw = win.to_numpy(dtype=float)
    idx = np.array([LOOKBACK])                                      # issue time is the last row

    X, _ = BUILDER.transform(Vw, idx, win.index)
    now = Vw[LOOKBACK]
    pred = 0.5 * (RIDGE.predict(X) + HGB.predict(X)) + now
    pred = np.clip(np.nan_to_num(pred, nan=0.0), 0, None)
    return pd.Series(pred, index=SENSORS, name="pm25_forecast")
