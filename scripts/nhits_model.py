
from __future__ import annotations
import os
import warnings
warnings.filterwarnings("ignore")
import logging
for _name in ("pytorch_lightning", "pytorch_lightning.utilities.rank_zero",
             "lightning", "lightning_fabric", "lightning_fabric.utilities.rank_zero",
             "darts.models.forecasting.torch_forecasting_model"):
    logging.getLogger(_name).setLevel(logging.ERROR)

import numpy as np
import pandas as pd
NHITS_IMPORT_ERROR = None
try:
    import torch
    from darts import TimeSeries
    from darts.models import NHiTSModel
    NHITS_AVAILABLE = True
except Exception as e:
    NHITS_AVAILABLE = False
    NHITS_IMPORT_ERROR = f"{type(e).__name__}: {e}"


def _quiet_trainer_kwargs() -> dict:
    return {
        "accelerator": "cpu",
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
    }


def build_series(stat_scaled: np.ndarray, sensors: list[str]):
    df = pd.DataFrame(stat_scaled.astype(np.float32), columns=sensors)
    return TimeSeries.from_dataframe(df)


def train_nhits(stat_scaled: np.ndarray, sensors: list[str], window: int,
                epochs: int = 20, num_stacks: int = 3, layer_widths: int = 128,
                random_state: int = 42):
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
