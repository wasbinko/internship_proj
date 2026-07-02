"""
NHITS forecaster, via Darts — a "fancier engine" alternative to the
from-scratch LSTMForecaster/PatchTST in models.py.
=====================================================================
Same role as LSTM/PatchTST: predict each channel's next value from recent
history, score by how wrong the prediction was. NHITS (Neural Hierarchical
Interpolation for Time Series) is a more sophisticated, purpose-built
forecasting architecture — hierarchical multi-rate decomposition of the
signal — that often outperforms a plain LSTM, at the cost of being a
heavier, third-party dependency instead of a few hand-written lines.

This module deliberately does NOT touch models.py's existing classes. It's
an additive, optional detector: everything here produces the same
per-timestep score array shape and the same bundle dict fields
(sensors, ch_types, scaler, channel_norm, threshold_p99/p995/p999, agg)
that LSTM/PatchTST already use, so infer.py's score_all() and app.py's UI
can treat "nhits" as just one more model name in the dispatch table.

A real environment issue, handled here rather than ignored
--------------------------------------------------------------
PyTorch 2.6+ changed torch.load()'s default to weights_only=True. Darts'
own checkpoint loading (via PyTorch Lightning) does not yet pass
weights_only=False through that call, so NHiTSModel.load() raises an
UnpicklingError on a stock recent PyTorch install. load_nhits() below works
around this with a narrowly-scoped monkeypatch of torch.load, active only
for the duration of the load call — safe here because the checkpoint being
loaded was written by this same training pipeline, not an untrusted file.
"""

from __future__ import annotations
import os
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)

import numpy as np
import pandas as pd

try:
    import torch
    from darts import TimeSeries
    from darts.models import NHiTSModel
    NHITS_AVAILABLE = True
except ImportError:
    NHITS_AVAILABLE = False


def _quiet_trainer_kwargs() -> dict:
    return {
        "accelerator": "cpu",
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
    }


def build_series(stat_scaled: np.ndarray, sensors: list[str]):
    """Wrap a (T, C) stationary/scaled array into a Darts multivariate
    TimeSeries. Uses a plain integer index — no real datetime needed, Darts
    supports non-time-indexed series directly."""
    df = pd.DataFrame(stat_scaled.astype(np.float32), columns=sensors)
    return TimeSeries.from_dataframe(df)


def train_nhits(stat_scaled: np.ndarray, sensors: list[str], window: int,
                epochs: int = 20, num_stacks: int = 3, layer_widths: int = 128,
                random_state: int = 42):
    """
    Train an NHITS model on the full (clean) stationary/scaled training
    array. Unlike the from-scratch LSTM/PatchTST (trained on many sampled
    windows), Darts models train directly on the whole series — internally
    they still learn from sliding windows of length `window`, just without
    us needing to construct them by hand.
    """
    if not NHITS_AVAILABLE:
        raise ImportError(
            "darts / pytorch-lightning not installed. Run: "
            "pip install darts pytorch-lightning"
        )
    series = build_series(stat_scaled, sensors)
    model = NHiTSModel(
        input_chunk_length=window, output_chunk_length=1,
        num_stacks=num_stacks, layer_widths=layer_widths,
        n_epochs=epochs, random_state=random_state,
        pl_trainer_kwargs=_quiet_trainer_kwargs(),
    )
    model.fit(series, verbose=False)
    return model


def score_series(model, stat_scaled: np.ndarray, sensors: list[str], window: int,
                 channel_norm: np.ndarray, agg: str = "max") -> np.ndarray:
    """
    Score every timestep in stat_scaled using walk-forward one-step-ahead
    forecasts, then normalise per channel and aggregate — same scoring
    logic as nn_forecast_errors() in models.py, just fed by Darts'
    historical_forecasts() instead of a hand-rolled windowing loop (Darts
    already does this efficiently internally).

    Returns a (T,) score array, NaN-padded for the first `window` timesteps
    where no prediction exists yet (not enough history) — same convention
    as errors_to_timestep() elsewhere, callers should forward-fill these.
    """
    series = build_series(stat_scaled, sensors)
    T = len(stat_scaled)
    if T <= window + 1:
        return np.full(T, np.nan, dtype=np.float32)

    # historical_forecasts walks forward one step at a time from `start`,
    # reusing the already-trained model (retrain=False) -- this is the
    # efficient, Darts-native equivalent of scoring every window by hand.
    start_idx = window
    hist = model.historical_forecasts(
        series, start=start_idx, forecast_horizon=1, stride=1,
        retrain=False, last_points_only=True, verbose=False,
        show_warnings=False,
    )
    preds = hist.values()                          # (T - window, C)
    actual = stat_scaled[start_idx: start_idx + len(preds)]

    sq_err = (preds - actual) ** 2
    normed = sq_err / (channel_norm[None, :] + 1e-12)
    if agg == "max":
        win_scores = normed.max(axis=1)
    elif agg == "top2" and normed.shape[1] >= 2:
        win_scores = np.partition(normed, -2, axis=1)[:, -2:].mean(axis=1)
    else:
        win_scores = normed.mean(axis=1)

    out = np.full(T, np.nan, dtype=np.float32)
    out[start_idx: start_idx + len(win_scores)] = win_scores
    # Forward/back-fill the warm-up region the same way errors_to_timestep() does
    if np.isnan(out[0]):
        first_valid = start_idx
        out[:first_valid] = out[first_valid] if not np.isnan(out[first_valid]) else 0.0
    mask = np.isnan(out)
    if mask.any():
        idx = np.where(~mask, np.arange(T), 0)
        np.maximum.accumulate(idx, out=idx)
        out = out[idx]
    return out.astype(np.float32)


def calibrate_channel_norm(model, stat_scaled: np.ndarray, sensors: list[str],
                           window: int, max_rows: int = 20_000) -> np.ndarray:
    """Per-channel nominal MSE, same role as calibrate_channel_norm() in
    models.py, computed via a walk-forward pass over clean nominal data."""
    d = stat_scaled[:max_rows]
    series = build_series(d, sensors)
    T = len(d)
    if T <= window + 1:
        return np.full(len(sensors), 1e-3, dtype=np.float32)
    hist = model.historical_forecasts(
        series, start=window, forecast_horizon=1, stride=1,
        retrain=False, last_points_only=True, verbose=False,
        show_warnings=False,
    )
    preds = hist.values()
    actual = d[window: window + len(preds)]
    sq_err = (preds - actual) ** 2
    norm = sq_err.mean(axis=0)
    floor = max(1e-8, float(np.median(norm)) * 1e-3)
    return np.maximum(norm, floor).astype(np.float32)


def save_nhits(model, path: str):
    model.save(path)


def load_nhits(path: str):
    """
    Load a saved NHITS model, working around PyTorch 2.6+'s weights_only=True
    default (see module docstring). The monkeypatch is scoped tightly to just
    this call and always restored in a finally block, even on error.
    """
    if not NHITS_AVAILABLE:
        raise ImportError("darts / pytorch-lightning not installed.")
    original_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)
    torch.load = _patched_load
    try:
        return NHiTSModel.load(path)
    finally:
        torch.load = original_load
