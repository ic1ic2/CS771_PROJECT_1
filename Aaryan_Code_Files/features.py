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
    """Mean of V over rows idx-start-length+1 ... idx-start, shape (n, S).
    Computed slice by slice (not with running sums), so the result is bit-for-bit identical whether V holds the
    whole year (notebook) or only the last few days (forecast.py)."""
    return np.stack([V[i - start - length + 1: i - start + 1].mean(0) for i in np.asarray(idx)])


def _spatial(A, M):
    """Row-wise A @ M.T for a row-stochastic neighbour matrix M, as an explicit weighted sum over each station's
    neighbours. Unlike a BLAS matrix product, the arithmetic does not depend on how many rows A has."""
    nz = M > 0
    k = int(nz.sum(1).max())
    order = np.argsort(~nz, axis=1, kind="stable")[:, :k]            # neighbour indices first, then padding
    w = np.take_along_axis(M, order, axis=1)                         # padding gets weight 0
    out = np.zeros((A.shape[0], A.shape[1]))
    for j in range(k):                                               # fixed summation order, element by element
        out += A[:, order[:, j]] * w[:, j]
    return out


def _rowmean(A):
    """Mean across stations (axis 1) in a fixed order, independent of memory layout and batch size."""
    out = np.zeros(A.shape[0])
    for j in range(A.shape[1]):
        out += A[:, j]
    return out / A.shape[1]


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
    nn_now = _spatial(V[idx], M10)
    nn_24 = _spatial(_window_mean(V, idx, 0, 24), M10)               # mean over t-23h ... t of the 10-NN mean
    net_now = _rowmean(V[idx])
    d = issue_days(times, idx)
    X = np.concatenate([own, nn_now[..., None], nn_24[..., None], _tile(net_now, S),
                        _tile(np.sin(2 * np.pi * d / 365), S), _tile(np.cos(2 * np.pi * d / 365), S)], axis=-1)
    return X.reshape(n * S, -1)


# ----------------------------------------------------------------------------------------------
# Part 2: grouped features
# ----------------------------------------------------------------------------------------------
GROUPS = ["recent", "neighbours", "network", "history", "trend", "calendar", "station", "anomaly"]
ANOMALY_NAMES = ["own_minus_nn10", "own_minus_r25", "own_minus_r50", "own_minus_net",
                 "mean24_minus_nn10_mean24", "now_anom_minus_mean24_anom"]
MAX_LOOKBACK_H = 7 * 24 + 23          # oldest row any feature touches, relative to the issue time


class FeatureBuilder:
    """Builds the Part 2 feature groups. Call fit() on training data only, then transform()."""

    def __init__(self, stations, groups=GROUPS, n_pca=5, n_clusters=6, seed=0, lead=24, n_anom_pca=2):
        self.stations = stations
        self.lead = lead                      # hours from issue to target (used for the target's day of week)
        self.groups = list(groups)
        self.n_pca, self.n_clusters, self.seed = n_pca, n_clusters, seed
        self.n_anom_pca = n_anom_pca
        D = distance_matrix(stations)
        self.M10 = knn_weights(D, 10)
        self.M25 = radius_weights(D, 25.0)
        self.M50 = radius_weights(D, 50.0)
        self.x_km, self.y_km = to_km(stations["latitude"], stations["longitude"])

    # -- fitting: training rows only ------------------------------------------------------------
    def fit(self, V_train, train_idx):
        """V_train: rows of the hourly table up to the end of the training period (no later rows!).
        train_idx: issue-time rows used for training (for PCA of the network state)."""
        from sklearn.cluster import KMeans            # imported here so forecast.py never needs scikit-learn
        from sklearn.decomposition import PCA
        self.station_mean_ = V_train.mean(0)                          # long-run station climatology
        self.station_std_ = V_train.std(0)
        snap = np.log1p(V_train[train_idx])                           # network snapshots at issue times
        p = PCA(n_components=self.n_pca, random_state=self.seed).fit(snap)
        self.pca_ = _LinearMap(p.mean_, p.components_)               # our own projection: same arithmetic everywhere
        geo = np.c_[self.x_km, self.y_km]
        geo = (geo - geo.mean(0)) / geo.std(0)
        self.cluster_ = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.seed).fit(geo).labels_
        # spatial-anomaly group: standardise its 6 raw features, then PCA (both fitted on training rows only)
        A = self._anomaly_raw(V_train, np.asarray(train_idx)).reshape(-1, len(ANOMALY_NAMES))
        self.anom_mean_, self.anom_std_ = A.mean(0), A.std(0)
        pa = PCA(n_components=self.n_anom_pca, random_state=self.seed).fit((A - self.anom_mean_) / self.anom_std_)
        self.anom_pca_ = _LinearMap(pa.mean_, pa.components_)
        self.anom_explained_ = float(pa.explained_variance_ratio_.sum())   # share of variance the 2 components keep
        return self

    # -- the groups ------------------------------------------------------------------------------
    def _recent(self, V, idx, times):
        own = _lags(V, idx, range(24))
        return own, [f"lag_{l}h" for l in range(24)]

    def _neighbours(self, V, idx, times):
        now = V[idx]
        m24 = _window_mean(V, idx, 0, 24)
        f = [_spatial(now, self.M10), _spatial(m24, self.M10), _spatial(now, self.M25), _spatial(now, self.M50),
             _spatial(m24, self.M50)]
        return np.stack(f, -1), ["nn10_now", "nn10_mean24", "r25_now", "r50_now", "r50_mean24"]

    def _network(self, V, idx, times):
        S = V.shape[1]
        net_now = _rowmean(V[idx])
        net_24 = _rowmean(_window_mean(V, idx, 0, 24))
        net_72 = _rowmean(_window_mean(V, idx, 0, 72))
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
                            (m30 / np.maximum(_rowmean(m30)[:, None], 1e-6))[..., None],
                            _per_station(np.c_[self.x_km, self.y_km], n), _per_station(onehot, n)], -1)
        names = (["station_mean_train", "station_std_train", "rel_level_7d", "x_km", "y_km"]
                 + [f"cluster_{c}" for c in range(self.n_clusters)])
        return f, names

    def _anomaly_raw(self, V, idx):
        """(n, S, 6): how far each station sits above or below its surroundings, now and over the last 24 h."""
        now = V[idx]
        m24 = _window_mean(V, idx, 0, 24)
        net = np.repeat(_rowmean(now)[:, None], V.shape[1], axis=1)
        a_now = now - _spatial(now, self.M10)
        a_24 = m24 - _spatial(m24, self.M10)
        return np.stack([a_now, now - _spatial(now, self.M25), now - _spatial(now, self.M50), now - net, a_24,
                         a_now - a_24], -1)

    def _anomaly(self, V, idx, times):
        n, S = len(idx), V.shape[1]
        A = (self._anomaly_raw(V, idx).reshape(n * S, -1) - self.anom_mean_) / self.anom_std_
        pcs = self.anom_pca_.transform(A).reshape(n, S, -1)
        return pcs, [f"anom_pc{i + 1}" for i in range(self.n_anom_pca)]

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


# ----------------------------------------------------------------------------------------------
# Part 5: a version-independent model file (plain numpy arrays, no pickled scikit-learn objects)
# ----------------------------------------------------------------------------------------------
class _LinearMap:
    """Stand-in for a fitted PCA: transform(X) = (X - mean) @ components.T (no whitening)."""

    def __init__(self, mean, components):
        self.mean_, self.components_ = np.asarray(mean, float), np.asarray(components, float)

    def transform(self, X):
        # explicit elementwise products + sums (no BLAS), so each row's result does not depend on the batch size
        Xc = np.asarray(X, float) - self.mean_
        out = np.zeros((Xc.shape[0], self.components_.shape[0]))
        for j in range(Xc.shape[1]):                                 # fixed summation order
            out += Xc[:, j, None] * self.components_[None, :, j]
        return out


def export_model(path, builder, ridge_pipe, hgb, sensor_ids, lookback):
    """Save the fitted feature builder, ridge pipeline and boosted trees as plain arrays (.npz).
    The file loads with numpy alone, so it does not depend on the scikit-learn version."""
    scaler, ridge = ridge_pipe.steps[0][1], ridge_pipe.steps[-1][1]
    trees = [preds[0].nodes for preds in hgb._predictors]          # one regression tree per boosting iteration
    sizes = np.array([len(t) for t in trees])
    nodes = np.concatenate(trees)
    if nodes["is_categorical"].any():
        raise ValueError("categorical splits are not supported by the portable export")
    np.savez_compressed(
        path,
        sensor_ids=np.array([str(s) for s in sensor_ids]), groups=np.array(builder.groups),
        lookback=lookback, lead=builder.lead, n_pca=builder.n_pca, n_clusters=builder.n_clusters,
        n_anom_pca=builder.n_anom_pca, seed=builder.seed,
        M10=builder.M10, M25=builder.M25, M50=builder.M50, x_km=builder.x_km, y_km=builder.y_km,
        station_mean=builder.station_mean_, station_std=builder.station_std_, cluster=builder.cluster_,
        pca_mean=builder.pca_.mean_, pca_comp=builder.pca_.components_,
        anom_mean=builder.anom_mean_, anom_std=builder.anom_std_,
        anom_pca_mean=builder.anom_pca_.mean_, anom_pca_comp=builder.anom_pca_.components_,
        sc_mean=scaler.mean_, sc_scale=scaler.scale_, ridge_coef=ridge.coef_, ridge_intercept=ridge.intercept_,
        hgb_baseline=np.ravel(hgb._baseline_prediction)[0], tree_sizes=sizes,
        t_feature=nodes["feature_idx"], t_threshold=nodes["num_threshold"], t_value=nodes["value"],
        t_left=nodes["left"], t_right=nodes["right"], t_leaf=nodes["is_leaf"],
        t_missing_left=nodes["missing_go_to_left"])


class PortableModel:
    """Loads model.npz and predicts with numpy only: features -> ridge + boosted trees (24 h change)."""

    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        b = FeatureBuilder.__new__(FeatureBuilder)                   # rebuild the builder from arrays
        b.groups = [str(g) for g in d["groups"]]
        b.lead, b.n_pca, b.n_clusters = int(d["lead"]), int(d["n_pca"]), int(d["n_clusters"])
        b.n_anom_pca, b.seed = int(d["n_anom_pca"]), int(d["seed"])
        b.M10, b.M25, b.M50 = d["M10"], d["M25"], d["M50"]
        b.x_km, b.y_km = d["x_km"], d["y_km"]
        b.station_mean_, b.station_std_, b.cluster_ = d["station_mean"], d["station_std"], d["cluster"]
        b.pca_ = _LinearMap(d["pca_mean"], d["pca_comp"])
        b.anom_mean_, b.anom_std_ = d["anom_mean"], d["anom_std"]
        b.anom_pca_ = _LinearMap(d["anom_pca_mean"], d["anom_pca_comp"])
        self.builder = b
        self.sensor_ids = [str(s) for s in d["sensor_ids"]]
        self.lookback = int(d["lookback"])
        self.sc_mean, self.sc_scale = d["sc_mean"], d["sc_scale"]
        self.coef, self.intercept = d["ridge_coef"], float(d["ridge_intercept"])
        self.baseline = float(d["hgb_baseline"])
        starts = np.r_[0, np.cumsum(d["tree_sizes"])[:-1]]
        self.trees = [(s, int(n)) for s, n in zip(starts, d["tree_sizes"])]
        self.f, self.thr, self.val = d["t_feature"], d["t_threshold"], d["t_value"]
        self.left, self.right = d["t_left"].astype(np.int64), d["t_right"].astype(np.int64)
        self.leaf, self.miss_left = d["t_leaf"].astype(bool), d["t_missing_left"].astype(bool)

    def ridge_predict(self, X):
        return ((X - self.sc_mean) / self.sc_scale) @ self.coef + self.intercept

    def hgb_predict(self, X):
        out = np.full(len(X), self.baseline)
        rows = np.arange(len(X))
        for start, _ in self.trees:
            node = np.full(len(X), start)
            active = ~self.leaf[node]
            while active.any():
                n = node[active]
                x = X[rows[active], self.f[n]]
                go_left = np.where(np.isnan(x), self.miss_left[n], x <= self.thr[n])
                node[active] = start + np.where(go_left, self.left[n], self.right[n])
                active = ~self.leaf[node]
            out += self.val[node]
        return out

    def predict_change(self, X):
        return 0.5 * (self.ridge_predict(X) + self.hgb_predict(X))
