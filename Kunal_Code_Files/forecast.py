"""Part 5 submission: Next-day (24h lead) PM2.5 forecasting for 468 Bihar stations."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# Load fitted objects once, at import time (not inside forecast()).
MODEL_PATH = HERE / "model.pkl"
ARTIFACT = joblib.load(MODEL_PATH)
STATIONS = pd.read_csv(HERE / "bihar_stations.csv", dtype={"sensor_id": str}).set_index("sensor_id")

SCALER = ARTIFACT["scaler"]
MODEL = ARTIFACT["model"]
SENSOR_IDS = ARTIFACT["sensor_ids"]
NN5_IDX = ARTIFACT["nn5_idx"]
NN10_IDX = ARTIFACT["nn10_idx"]
NN25_IDX = ARTIFACT["nn25_idx"]
NN50_IDX = ARTIFACT["nn50_idx"]
NN100_IDX = ARTIFACT["nn100_idx"]
W10 = ARTIFACT["w10"]
W25 = ARTIFACT["w25"]
W50 = ARTIFACT["w50"]
K15 = ARTIFACT["K15"]
K35 = ARTIFACT["K35"]
K75 = ARTIFACT["K75"]
W_WEST = ARTIFACT["W_west"]
W_EAST = ARTIFACT["W_east"]
W_NORTH = ARTIFACT["W_north"]
W_SOUTH = ARTIFACT["W_south"]
EXP_W6 = ARTIFACT["EXP_W6"]
EXP_W12 = ARTIFACT["EXP_W12"]


def _build_features(arr: np.ndarray, idx: int) -> np.ndarray:
    """Compute the 5 spatio-temporal feature groups (101 features per station) at issue index idx."""
    n_st = arr.shape[1]

    # Group 1: Station 24-hour trajectory (35 features)
    w24 = arr[idx - 23 : idx + 1, :][::-1, :].T  # (468, 24)
    s_mean24 = w24.mean(axis=1, keepdims=True)
    s_std24 = w24.std(axis=1, keepdims=True)
    s_min24 = w24.min(axis=1, keepdims=True)
    s_max24 = w24.max(axis=1, keepdims=True)
    s_med24 = np.median(w24, axis=1, keepdims=True)
    s_diff3 = (w24[:, 0] - w24[:, 3])[:, None]
    s_diff6 = (w24[:, 0] - w24[:, 6])[:, None]
    s_diff12 = (w24[:, 0] - w24[:, 12])[:, None]
    s_diff24 = (w24[:, 0] - w24[:, 23])[:, None]
    ema6 = (w24[:, :6] @ EXP_W6)[:, None]
    ema12 = (w24[:, :12] @ EXP_W12)[:, None]
    g1 = np.hstack([
        w24, s_mean24, s_std24, s_min24, s_max24, s_med24,
        s_diff3, s_diff6, s_diff12, s_diff24, ema6, ema12
    ])

    # Group 2: Multi-day history & temporal trends (15 features)
    d2 = arr[max(0, idx - 48), :][:, None]
    d3 = arr[max(0, idx - 72), :][:, None]
    d4 = arr[max(0, idx - 96), :][:, None]
    d5 = arr[max(0, idx - 120), :][:, None]
    d6 = arr[max(0, idx - 144), :][:, None]
    d7 = arr[max(0, idx - 168), :][:, None]
    w72 = arr[max(0, idx - 71) : idx + 1, :]
    w168 = arr[max(0, idx - 167) : idx + 1, :]
    w336 = arr[max(0, idx - 335) : idx + 1, :]
    s_mean72 = w72.mean(axis=0)[:, None]
    s_std72 = w72.std(axis=0)[:, None]
    s_mean168 = w168.mean(axis=0)[:, None]
    s_std168 = w168.std(axis=0)[:, None]
    s_mean336 = w336.mean(axis=0)[:, None]
    s_trend_1_3 = s_mean24 - s_mean72
    s_trend_1_7 = s_mean24 - s_mean168
    noon_3d_mean = (w24[:, 0:1] + w24[:, 23:24] + d2) / 3.0
    noon_7d_mean = (w24[:, 0:1] + w24[:, 23:24] + d2 + d3 + d4 + d5 + d6 + d7) / 8.0
    g2 = np.hstack([
        d2, d3, d4, d5, d6, d7,
        s_mean72, s_std72, s_mean168, s_std168, s_mean336,
        s_trend_1_3, s_trend_1_7, noon_3d_mean, noon_7d_mean
    ])

    vals_t = arr[idx, :]
    vals_tm6 = arr[max(0, idx - 6), :]
    vals_tm12 = arr[max(0, idx - 12), :]
    vals_tm24 = arr[max(0, idx - 23), :]
    mean24_s = s_mean24[:, 0]
    mean72_s = s_mean72[:, 0]
    mean168_s = s_mean168[:, 0]

    # Group 3: Local Neighbourhood (22 features)
    nn5_t = vals_t[NN5_IDX].mean(axis=1, keepdims=True)
    nn10_t = vals_t[NN10_IDX].mean(axis=1, keepdims=True)
    nn25_t = vals_t[NN25_IDX].mean(axis=1, keepdims=True)
    idw10_t = np.sum(vals_t[NN10_IDX] * W10, axis=1, keepdims=True)
    idw25_t = np.sum(vals_t[NN25_IDX] * W25, axis=1, keepdims=True)
    k15_t = (K15 @ vals_t)[:, None]
    k35_t = (K35 @ vals_t)[:, None]
    nn10_med_t = np.median(vals_t[NN10_IDX], axis=1, keepdims=True)
    nn25_med_t = np.median(vals_t[NN25_IDX], axis=1, keepdims=True)
    nn10_std_t = vals_t[NN10_IDX].std(axis=1, keepdims=True)
    nn25_std_t = vals_t[NN25_IDX].std(axis=1, keepdims=True)
    nn10_tm6 = vals_tm6[NN10_IDX].mean(axis=1, keepdims=True)
    nn10_tm12 = vals_tm12[NN10_IDX].mean(axis=1, keepdims=True)
    nn10_tm24 = vals_tm24[NN10_IDX].mean(axis=1, keepdims=True)
    nn5_24 = mean24_s[NN5_IDX].mean(axis=1, keepdims=True)
    nn10_24 = mean24_s[NN10_IDX].mean(axis=1, keepdims=True)
    nn25_24 = mean24_s[NN25_IDX].mean(axis=1, keepdims=True)
    idw10_24 = np.sum(mean24_s[NN10_IDX] * W10, axis=1, keepdims=True)
    idw25_24 = np.sum(mean24_s[NN25_IDX] * W25, axis=1, keepdims=True)
    diff_self_idw10 = vals_t[:, None] - idw10_t
    diff_self_nn25 = vals_t[:, None] - nn25_t
    diff_nn10_t_24 = nn10_t - nn10_24
    g3 = np.hstack([
        nn5_t, nn10_t, nn25_t, idw10_t, idw25_t, k15_t, k35_t,
        nn10_med_t, nn25_med_t, nn10_std_t, nn25_std_t,
        nn10_tm6, nn10_tm12, nn10_tm24,
        nn5_24, nn10_24, nn25_24, idw10_24, idw25_24,
        diff_self_idw10, diff_self_nn25, diff_nn10_t_24
    ])

    # Group 4: Regional & Directional Plume Advection (15 features)
    nn50_t = vals_t[NN50_IDX].mean(axis=1, keepdims=True)
    nn100_t = vals_t[NN100_IDX].mean(axis=1, keepdims=True)
    idw50_t = np.sum(vals_t[NN50_IDX] * W50, axis=1, keepdims=True)
    k75_t = (K75 @ vals_t)[:, None]
    west_t = (W_WEST @ vals_t)[:, None]
    east_t = (W_EAST @ vals_t)[:, None]
    north_t = (W_NORTH @ vals_t)[:, None]
    south_t = (W_SOUTH @ vals_t)[:, None]
    west_24 = (W_WEST @ mean24_s)[:, None]
    east_24 = (W_EAST @ mean24_s)[:, None]
    nn50_24 = mean24_s[NN50_IDX].mean(axis=1, keepdims=True)
    k35_24 = (K35 @ mean24_s)[:, None]
    k75_24 = (K75 @ mean24_s)[:, None]
    nn25_72 = mean72_s[NN25_IDX].mean(axis=1, keepdims=True)
    nn25_168 = mean168_s[NN25_IDX].mean(axis=1, keepdims=True)
    g4 = np.hstack([
        nn50_t, nn100_t, idw50_t, k75_t,
        west_t, east_t, north_t, south_t, west_24, east_24,
        nn50_24, k35_24, k75_24, nn25_72, nn25_168
    ])

    # Group 5: Network-wide state & regime dynamics (14 features)
    net_mean_t = np.full((n_st, 1), vals_t.mean())
    net_std_t = np.full((n_st, 1), vals_t.std())
    net_med_t = np.full((n_st, 1), np.median(vals_t))
    net_q25_t = np.full((n_st, 1), np.percentile(vals_t, 25))
    net_q75_t = np.full((n_st, 1), np.percentile(vals_t, 75))
    net_mean_tm6 = np.full((n_st, 1), vals_tm6.mean())
    net_mean_tm12 = np.full((n_st, 1), vals_tm12.mean())
    net_mean_tm24 = np.full((n_st, 1), vals_tm24.mean())
    net_mean_24 = np.full((n_st, 1), mean24_s.mean())
    net_mean_72 = np.full((n_st, 1), mean72_s.mean())
    net_mean_168 = np.full((n_st, 1), mean168_s.mean())
    net_diff_24 = net_mean_t - net_mean_tm24
    net_trend_1_3 = net_mean_24 - net_mean_72
    net_trend_1_7 = net_mean_24 - net_mean_168
    g5 = np.hstack([
        net_mean_t, net_std_t, net_med_t, net_q25_t, net_q75_t,
        net_mean_tm6, net_mean_tm12, net_mean_tm24,
        net_mean_24, net_mean_72, net_mean_168,
        net_diff_24, net_trend_1_3, net_trend_1_7
    ])

    return np.hstack([g1, g2, g3, g4, g5])


def forecast(history: pd.DataFrame, issue_time: pd.Timestamp) -> pd.Series:
    """history: wide hourly table (DatetimeIndex 'timestamp', one string column per sensor_id),
    every hour from 1 Jan 2024 00:00 up to and including issue_time.
    issue_time: 12:00 on some day (IST).
    Returns predicted PM2.5 for issue_time + 24 h, indexed by sensor_id (all 468 stations)."""
    # Ensure column order matches training sensor_ids andslice up to issue_time (last 336 hours are sufficient)
    sub = history.loc[:issue_time, SENSOR_IDS].iloc[-336:]
    arr = sub.to_numpy(dtype=np.float64)
    idx = len(arr) - 1

    X = _build_features(arr, idx)
    X_scaled = SCALER.transform(X)
    prediction = MODEL.predict(X_scaled)

    out = pd.Series(np.clip(prediction, 0.0, None), index=SENSOR_IDS, name="pm25_forecast")
    return out.reindex(history.columns.astype(str))
