"""Feature construction for next-day PM2.5 forecasting (shared by the notebook and forecast.py).

Every feature for an issue time t uses only rows of V at or before t. Statistics that need fitting
(station long-run means, PCA of the network state, k-means station clusters) are fitted in
FeatureBuilder.fit() on training rows only and then frozen.

Notation:
    V      : (T, S) array of hourly PM2.5, rows = hours, columns = stations (same order as the stations table)
    times  : DatetimeIndex of length T for the rows of V
    idx    : int array of issue-time row positions in V; one example per (issue time, station)
Rows of every feature matrix are ordered issue-major: all S stations for idx[0], then all for idx[1], ...
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

DAY0 = pd.Timestamp("2024-01-01")


# ----------------------------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------------------------
def to_km(lat, lon):
    """Approximate km coordinates required by the assignment."""
    return (np.asarray(lon) - 85.5) * 100.3, (np.asarray(lat) - 25.7) * 110.6


def distance_matrix(stations):
    x, y = to_km(stations["latitude"], stations["longitude"])
    return np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])


def knn_weights(D, k=10):
    """(S, S) row-stochastic matrix averaging each station's k nearest OTHER stations."""
    S = len(D)
    D = D.copy()
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, axis=1, kind="stable")[:, :k]
    M = np.zeros((S, S))
    M[np.repeat(np.arange(S), k), nn.ravel()] = 1.0 / k
    return M


def radius_weights(D, r_km):
    """(S, S) row-stochastic matrix averaging all OTHER stations within r_km (falls back to the 3 nearest)."""
    S = len(D)
    A = (D <= r_km).astype(float)
    np.fill_diagonal(A, 0.0)
    empty = A.sum(1) == 0
    if empty.any():
        A[empty] = knn_weights(D, 3)[empty] * 3
    return A / A.sum(1, keepdims=True)


# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------
def _lags(V, idx, lags):
    """(n, S, len(lags)) array of V at idx - lag."""
    return np.stack([V[idx - l] for l in lags], axis=-1)


def _window_mean(V, idx, start, length):
    """Mean of V over rows idx-start-length+1 ... idx-start, shape (n, S)."""
    c = np.vstack([np.zeros((1, V.shape[1])), np.cumsum(V, axis=0)])
    hi = idx - start + 1
    lo = hi - length
    return (c[hi] - c[lo]) / length


def _tile(a, S):
    """Repeat a per-issue array (n,) or (n, k) across S stations -> (n, S, k)."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    return np.repeat(a[:, None, :], S, axis=1)


def _per_station(a, n):
    """Repeat a per-station array (S,) or (S, k) across n issues -> (n, S, k)."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    return np.repeat(a[None, :, :], n, axis=0)


def issue_days(times, idx):
    return np.asarray((times[idx].normalize() - DAY0).days, dtype=float)


def targets(V, idx, lead=24):
    """Target vector aligned with the feature rows: V at idx + lead, flattened issue-major."""
    return V[idx + lead].reshape(-1)


# ----------------------------------------------------------------------------------------------
# Part 1: the fixed 29-feature ridge baseline
# ----------------------------------------------------------------------------------------------
BASE29_NAMES = ([f"lag_{l}h" for l in range(24)] + ["nn10_now", "nn10_mean24", "net_now", "doy_sin", "doy_cos"])


def base29(V, idx, times, M10):
    """The 29 Part-1 features: 24 own lags, 10-NN mean now and over 24 h, network mean now, sin/cos(2πd/365)."""
    n, S = len(idx), V.shape[1]
    own = _lags(V, idx, range(24))                                   # t, t-1h, ..., t-23h
    nn_now = V[idx] @ M10.T
    nn_24 = _window_mean(V, idx, 0, 24) @ M10.T                      # mean over t-23h ... t of the 10-NN mean
    net_now = V[idx].mean(1)
    d = issue_days(times, idx)
    X = np.concatenate([own, nn_now[..., None], nn_24[..., None], _tile(net_now, S),
                        _tile(np.sin(2 * np.pi * d / 365), S), _tile(np.cos(2 * np.pi * d / 365), S)], axis=-1)
    return X.reshape(n * S, -1)


# ----------------------------------------------------------------------------------------------
# Part 2: grouped features
# ----------------------------------------------------------------------------------------------
GROUPS = ["recent", "neighbours", "network", "history", "trend", "calendar", "station"]
MAX_LOOKBACK_H = 7 * 24 + 23          # oldest row any feature touches, relative to the issue time


class FeatureBuilder:
    """Builds the Part 2 feature groups. Call fit() on training data only, then transform()."""

    def __init__(self, stations, groups=GROUPS, n_pca=5, n_clusters=6, seed=0, lead=24):
        self.stations = stations
        self.lead = lead                      # hours from issue to target (used for the target's day of week)
        self.groups = list(groups)
        self.n_pca, self.n_clusters, self.seed = n_pca, n_clusters, seed
        D = distance_matrix(stations)
        self.M10 = knn_weights(D, 10)
        self.M25 = radius_weights(D, 25.0)
        self.M50 = radius_weights(D, 50.0)
        self.x_km, self.y_km = to_km(stations["latitude"], stations["longitude"])

    # -- fitting: training rows only ------------------------------------------------------------
    def fit(self, V_train, train_idx):
        """V_train: rows of the hourly table up to the end of the training period (no later rows!).
        train_idx: issue-time rows used for training (for PCA of the network state)."""
        self.station_mean_ = V_train.mean(0)                          # long-run station climatology
        self.station_std_ = V_train.std(0)
        snap = np.log1p(V_train[train_idx])                           # network snapshots at issue times
        self.pca_ = PCA(n_components=self.n_pca, random_state=self.seed).fit(snap)
        geo = np.c_[self.x_km, self.y_km]
        geo = (geo - geo.mean(0)) / geo.std(0)
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.seed).fit(geo)
        self.cluster_ = self.kmeans_.labels_
        return self

    # -- the groups ------------------------------------------------------------------------------
    def _recent(self, V, idx, times):
        own = _lags(V, idx, range(24))
        return own, [f"lag_{l}h" for l in range(24)]

    def _neighbours(self, V, idx, times):
        now = V[idx]
        m24 = _window_mean(V, idx, 0, 24)
        f = [now @ self.M10.T, m24 @ self.M10.T, now @ self.M25.T, now @ self.M50.T, m24 @ self.M50.T]
        return np.stack(f, -1), ["nn10_now", "nn10_mean24", "r25_now", "r50_now", "r50_mean24"]

    def _network(self, V, idx, times):
        S = V.shape[1]
        net_now = V[idx].mean(1)
        net_24 = _window_mean(V, idx, 0, 24).mean(1)
        net_72 = _window_mean(V, idx, 0, 72).mean(1)
        pcs = self.pca_.transform(np.log1p(V[idx]))
        f = np.concatenate([_tile(net_now, S), _tile(net_24, S), _tile(net_72, S), _tile(pcs, S)], -1)
        return f, ["net_now", "net_mean24", "net_mean72"] + [f"net_pc{i + 1}" for i in range(self.n_pca)]

    def _history(self, V, idx, times):
        same_hour = _lags(V, idx, [24 * k for k in range(1, 8)])                       # same hour, 1..7 days ago
        day_means = np.stack([_window_mean(V, idx, 24 * k, 24) for k in range(1, 7)], -1)  # previous 6 daily means
        m7 = _window_mean(V, idx, 0, 168)[..., None]
        names = ([f"same_hour_d-{k}" for k in range(1, 8)] + [f"day_mean_d-{k}" for k in range(1, 7)] + ["mean_7d"])
        return np.concatenate([same_hour, day_means, m7], -1), names

    def _trend(self, V, idx, times):
        today = _window_mean(V, idx, 0, 24)
        yday = _window_mean(V, idx, 24, 24)
        f = [today - yday, V[idx] - V[idx - 24], V[idx] - V[idx - 3], V[idx] - today]
        return np.stack(f, -1), ["trend_day", "trend_24h", "trend_3h", "now_minus_mean24"]

    def _calendar(self, V, idx, times):
        S = V.shape[1]
        d = issue_days(times, idx)
        tgt_dow = np.asarray((times[idx] + pd.Timedelta(hours=self.lead)).dayofweek, dtype=float)
        f = [np.sin(2 * np.pi * d / 365), np.cos(2 * np.pi * d / 365),
             np.sin(2 * np.pi * tgt_dow / 7), np.cos(2 * np.pi * tgt_dow / 7)]
        return np.concatenate([_tile(a, S) for a in f], -1), ["doy_sin", "doy_cos", "dow_sin", "dow_cos"]

    def _station(self, V, idx, times):
        n = len(idx)
        m30 = _window_mean(V, idx, 0, 7 * 24)                          # last 7 days (within lookback)
        onehot = np.eye(self.n_clusters)[self.cluster_]
        f = np.concatenate([_per_station(self.station_mean_, n), _per_station(self.station_std_, n),
                            (m30 / np.maximum(m30.mean(1, keepdims=True), 1e-6))[..., None],
                            _per_station(np.c_[self.x_km, self.y_km], n), _per_station(onehot, n)], -1)
        names = (["station_mean_train", "station_std_train", "rel_level_7d", "x_km", "y_km"]
                 + [f"cluster_{c}" for c in range(self.n_clusters)])
        return f, names

    # -- public ----------------------------------------------------------------------------------
    def transform(self, V, idx, times, groups=None):
        groups = self.groups if groups is None else groups
        idx = np.asarray(idx)
        if (idx - MAX_LOOKBACK_H < 0).any():
            raise ValueError("some issue times have less than 8 days of history")
        blocks, names = [], []
        for g in groups:
            f, nm = getattr(self, f"_{g}")(V, idx, times)
            blocks.append(f)
            names += nm
        X = np.concatenate(blocks, -1)
        return X.reshape(len(idx) * V.shape[1], -1), names

    def group_slices(self, V, idx, times):
        """Column ranges of each group in the full feature matrix (for ablation)."""
        out, start = {}, 0
        for g in self.groups:
            _, nm = getattr(self, f"_{g}")(V, idx[:1], times)
            out[g] = (start, start + len(nm))
            start += len(nm)
        return out
