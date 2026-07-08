
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler

SENSOR_COLS = ["arnd", "bfo2", "cso1"]


# ─────────────────────────────────────────────────────────────────────────────
# Channel typing
# ─────────────────────────────────────────────────────────────────────────────

def classify_channel(values: np.ndarray) -> str:
    """Classify a channel by its statistical character so each gets the right
    detector logic. Returns one of: binary, constant, drift, oscillatory."""
    v = values[~np.isnan(values)]
    uniq = np.unique(v)
    if set(uniq).issubset({0.0, 1.0}) or len(uniq) <= 3:
        return "binary"
    std = v.std()
    if std < 0.5:                      # cso1: tight noise around a constant
        return "constant"
    centered = v - v.mean()
    fft = np.abs(np.fft.rfft(centered))
    exclude = max(3, int(0.01 * len(fft)))
    fft[:exclude] = 0.0
    peak_ratio = fft.max() / (fft.mean() + 1e-9)
    if peak_ratio > 90.0:              # sharp spectral peak → periodic
        return "oscillatory"           # arnd: sine
    return "drift"                     # bfo2: random walk, wanders far


# ─────────────────────────────────────────────────────────────────────────────
# Stationary features for ML models
# ─────────────────────────────────────────────────────────────────────────────

def make_stationary(data: np.ndarray, sensors: list[str], ch_types: dict) -> np.ndarray:
    """Differencing per channel type so the ML models see chunk-invariant input."""
    out = np.empty_like(data, dtype=np.float64)
    df = pd.DataFrame(data, columns=sensors)
    for i, col in enumerate(sensors):
        t = ch_types[col]
        s = df[col].astype(float)
        if t == "binary":
            out[:, i] = s.values
        elif t == "constant":
            out[:, i] = (s - s.rolling(30, min_periods=1).median()).values
        else:  # drift or oscillatory → first difference
            out[:, i] = s.diff().fillna(0).values
    return out.astype(np.float32)


def rolling_features(data: np.ndarray, sensors: list[str],
                     windows: tuple = (10, 30, 60)) -> np.ndarray:
    """Rolling stats per channel/window. Used by XGBoost and IsolationForest."""
    T, C = data.shape
    df = pd.DataFrame(data, columns=sensors)
    blocks = []
    for i, col in enumerate(sensors):
        s = df[col]
        vals = data[:, i]
        is_binary = set(np.unique(vals[~np.isnan(vals)])).issubset({0.0, 1.0, -1.0})
        for w in windows:
            roll = s.rolling(w, min_periods=max(1, w // 4))
            mn  = roll.mean().values
            sd  = roll.std().fillna(0).values
            cur = vals
            z   = np.where(sd > 1e-8, (cur - mn) / sd, 0.0)
            amx = s.abs().rolling(w, min_periods=1).max().values
            if is_binary:
                blocks.append(np.column_stack([mn, sd]))
            else:
                blocks.append(np.column_stack([mn, sd, z, amx]))
    return np.hstack(blocks).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical detector — purpose-built for these anomaly archetypes
# ─────────────────────────────────────────────────────────────────────────────

def _robust_floor(computed_std: float, reference_value: float,
                  rel_frac: float = 0.02, abs_floor: float = 1e-3) -> float:
    computed_std = float(computed_std) if np.isfinite(computed_std) else 0.0
    rel = rel_frac * abs(float(reference_value)) if np.isfinite(reference_value) else 0.0
    return max(computed_std, rel, abs_floor)

STAT_SCORE_CEILING = 200.0


class StatDetector:
    def __init__(self, window: int = 30):
        self.window = window
        self.sensors: list[str] = []
        self.ch_types: dict = {}
        self.stats: dict = {}      # per-channel calibrated baselines

    def _roll_std(self, s: pd.Series) -> np.ndarray:
        return s.rolling(self.window, min_periods=max(2, self.window//3)).std().bfill().fillna(0).values

    def _roll_mean(self, s: pd.Series) -> np.ndarray:
        return s.rolling(self.window, min_periods=1).mean().values

    def _roll_slope(self, s: pd.Series) -> np.ndarray:

        w = self.window
        net = (s - s.shift(w)).abs()
        return net.fillna(0).values

    def fit(self, data: np.ndarray, sensors: list[str], ch_types: dict | None = None):

        self.sensors = sensors
        df = pd.DataFrame(data, columns=sensors)
        for i, col in enumerate(sensors):
            t = ch_types[col] if ch_types and col in ch_types else classify_channel(data[:, i])
            self.ch_types[col] = t
            s = df[col].astype(float)
            st = {}
            if t == "binary":
                rate = pd.Series(s).rolling(self.window, min_periods=1).mean().values
                st["rate_mean"] = float(np.mean(rate))
                st["rate_std"]  = _robust_floor(np.std(rate), np.mean(rate))
            elif t == "constant":
                rs = self._roll_std(s)
                st["std_mean"] = float(np.mean(rs))
                st["std_std"]  = _robust_floor(np.std(rs), np.mean(rs))
                st["level"]    = float(s.median())
                st["level_scale"] = _robust_floor(s.std(), s.abs().median())
            elif t == "drift":
                sl = self._roll_slope(s)
                st["slope_p95"]  = float(np.percentile(sl, 95))
                st["slope_std"]  = _robust_floor(np.std(sl), np.mean(sl))
                st["slope_mean"] = float(np.mean(sl))
            else:  # oscillatory
                rs = self._roll_std(s)
                st["std_mean"] = float(np.mean(rs))
                st["std_std"]  = _robust_floor(np.std(rs), np.mean(rs))
            self.stats[col] = st

    def score(self, data: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(data, columns=self.sensors)
        T = len(df)
        per_ch = np.zeros((T, len(self.sensors)), dtype=np.float32)
        for i, col in enumerate(self.sensors):
            t = self.ch_types[col]
            st = self.stats[col]
            s = df[col].astype(float)
            if t == "binary":
                rate = pd.Series(s).rolling(self.window, min_periods=1).mean().values
                # stuck ON = activation rate well above normal
                per_ch[:, i] = np.maximum(0, (rate - st["rate_mean"]) / st["rate_std"])
            elif t == "constant":
                rs = self._roll_std(s)
                # freeze = std well BELOW normal (z is negative → take -z)
                freeze = np.maximum(0, (st["std_mean"] - rs) / st["std_std"])
                # level shift = value far from normal level
                level = np.abs(self._roll_mean(s) - st["level"]) / st["level_scale"]
                per_ch[:, i] = np.maximum(freeze, level)
            elif t == "drift":
                sl = self._roll_slope(s)
                # trend break = slope magnitude well above the NORMAL 95th-pctile
                # excursion (random walks naturally wander, so baseline off p95)
                base = st.get("slope_p95", st["slope_mean"])
                per_ch[:, i] = np.maximum(0, (sl - base) / st["slope_std"])
            else:  # oscillatory
                rs = self._roll_std(s)
                # burst = std well above normal
                per_ch[:, i] = np.maximum(0, (rs - st["std_mean"]) / st["std_std"])
        per_ch = np.minimum(per_ch, STAT_SCORE_CEILING)
        return per_ch.max(axis=1)

    def score_per_channel(self, data: np.ndarray) -> np.ndarray:
        """Same as score() but returns the (T, C) matrix for attribution."""
        df = pd.DataFrame(data, columns=self.sensors)
        T = len(df)
        per_ch = np.zeros((T, len(self.sensors)), dtype=np.float32)
        for i, col in enumerate(self.sensors):
            t = self.ch_types[col]; st = self.stats[col]
            s = df[col].astype(float)
            if t == "binary":
                rate = pd.Series(s).rolling(self.window, min_periods=1).mean().values
                per_ch[:, i] = np.maximum(0, (rate - st["rate_mean"]) / st["rate_std"])
            elif t == "constant":
                rs = self._roll_std(s)
                freeze = np.maximum(0, (st["std_mean"] - rs) / st["std_std"])
                level = np.abs(self._roll_mean(s) - st["level"]) / st["level_scale"]
                per_ch[:, i] = np.maximum(freeze, level)
            elif t == "drift":
                sl = self._roll_slope(s)
                base = st.get("slope_p95", st["slope_mean"])
                per_ch[:, i] = np.maximum(0, (sl - base) / st["slope_std"])
            else:
                rs = self._roll_std(s)
                per_ch[:, i] = np.maximum(0, (rs - st["std_mean"]) / st["std_std"])
        # Same ceiling as score() — see STAT_SCORE_CEILING docstring.
        return np.minimum(per_ch, STAT_SCORE_CEILING)


# ─────────────────────────────────────────────────────────────────────────────
# Forecasting models
# ─────────────────────────────────────────────────────────────────────────────

class LSTMForecaster(nn.Module):
    def __init__(self, n_channels: int, hidden: int = 64, n_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, n_layers, batch_first=True)
        self.head = nn.Linear(hidden, n_channels)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class PatchTST(nn.Module):
    def __init__(self, n_channels: int, window: int = 60, patch_len: int = 12,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.n_patches = window // patch_len
        self.patch_embed = nn.Linear(patch_len * n_channels, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                         dim_feedforward=d_model*4, dropout=dropout,
                                         batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_channels)
    def forward(self, x):
        B, W, C = x.shape
        W2 = self.n_patches * self.patch_len
        x = x[:, :W2, :].reshape(B, self.n_patches, self.patch_len * C)
        x = self.patch_embed(x) + self.pos_enc
        x = self.transformer(x)
        return self.head(x[:, -1, :])


class IsolationForestDetector:
    def __init__(self, n_estimators: int = 150, contamination: float = 0.02):
        from sklearn.ensemble import IsolationForest
        self.clf = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                                   random_state=42, n_jobs=-1)
        self.scaler = RobustScaler()
        self.sensors: list[str] = []
    def fit(self, stat_data: np.ndarray, sensors: list[str]):
        self.sensors = sensors
        feats = np.nan_to_num(rolling_features(stat_data, sensors), nan=0.0)
        self.clf.fit(self.scaler.fit_transform(feats))
    def score(self, stat_data: np.ndarray) -> np.ndarray:
        feats = np.nan_to_num(rolling_features(stat_data, self.sensors), nan=0.0)
        raw = self.clf.decision_function(self.scaler.transform(feats))
        return np.maximum(0.0, -raw).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Windowing + NN scoring utilities
# ─────────────────────────────────────────────────────────────────────────────

def make_forecast_windows(data: np.ndarray, window: int, stride: int = 1):
    T, C = data.shape
    n = (T - window - 1) // stride + 1
    if n <= 0:
        return (np.empty((0, window, C), data.dtype),
                np.empty((0, C), data.dtype), np.empty((0,), np.int64))
    s0, s1 = data.strides
    X = np.lib.stride_tricks.as_strided(data, (n, window, C), (stride*s0, s0, s1))
    end_idx = np.arange(n) * stride + window
    return X, data[end_idx], end_idx


@torch.no_grad()
def nn_forecast_errors(model, X, y, channel_norm, device="cpu", batch_size=512, agg="max"):
    model.eval()
    N, C = len(X), y.shape[1]
    errors = np.empty((N, C), np.float32)
    for s in range(0, N, batch_size):
        e = min(s+batch_size, N)
        xb = torch.from_numpy(np.ascontiguousarray(X[s:e])).float().to(device)
        errors[s:e] = (model(xb).cpu().numpy() - y[s:e])**2
    normed = errors / (channel_norm[None,:] + 1e-12)
    if agg == "max":  return normed.max(axis=1)
    if agg == "top2":
        return np.partition(normed, -2, axis=1)[:, -2:].mean(axis=1)
    return normed.mean(axis=1)


def errors_to_timestep(win_scores, end_idx, n_timesteps):
    out = np.full(n_timesteps, np.nan, np.float32)
    valid = end_idx < n_timesteps
    out[end_idx[valid]] = win_scores[valid]
    first = end_idx[0] if len(end_idx) else 0
    if first < n_timesteps and not np.isnan(out[first]):
        out[:first] = out[first]
    mask = np.isnan(out)
    if mask.any():
        idx = np.where(~mask, np.arange(n_timesteps), 0)
        np.maximum.accumulate(idx, out=idx)
        out = out[idx]
    return out


def calibrate_channel_norm(model, stat_scaled, window, device="cpu", max_rows=30000):
    d = stat_scaled[:max_rows]
    X, y, _ = make_forecast_windows(d, window, stride=1)
    X = np.ascontiguousarray(X); y = np.ascontiguousarray(y)
    C = y.shape[1]; errors = np.empty((len(X), C), np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(X), 512):
            e = min(s+512, len(X))
            xb = torch.from_numpy(X[s:e]).float().to(device)
            errors[s:e] = (model(xb).cpu().numpy() - y[s:e])**2
    norm = errors.mean(axis=0)
    return np.maximum(norm, max(1e-8, float(np.median(norm))*1e-3))
