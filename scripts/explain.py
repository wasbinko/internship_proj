"""
SHAP explainability for the XGBoost detector.
================================================
Answers "why did the model flag this specific moment?" for any timestep,
using SHAP's TreeExplainer — a fast, exact attribution method purpose-built
for tree ensembles like XGBoost (no approximation needed, unlike LIME).

Why XGBoost specifically
--------------------------
Of the five detectors in this project, XGBoost is the natural fit for SHAP:
TreeExplainer computes EXACT feature attributions for tree models in
milliseconds. The forecasters (LSTM/PatchTST) would need a slower,
approximate explainer (DeepExplainer/GradientExplainer) for meaningfully
noisier results, and StatDetector's logic is already fully transparent by
construction (each channel's z-score IS its own explanation — there's
nothing hidden to attribute).

What "explaining" means here
-------------------------------
XGBoost doesn't predict "anomalous" directly — it predicts what each
channel's NEXT value should be, and the anomaly score comes from how wrong
that prediction was. So "explaining an anomaly" means: explaining the
regressor's PREDICTION for the channel that drove the highest anomaly
score at that moment. SHAP tells you which input features (e.g. "bfo2's
rolling standard deviation over the last 30 seconds") pushed that
prediction up or down the most. A large gap between what was predicted and
what actually happened, for a feature-driven reason you can name, is a much
more concrete story than a bare anomaly score.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from models import make_stationary, rolling_features


def rolling_feature_names(sensors: list[str], ch_types: dict,
                          windows: tuple = (10, 30, 60)) -> list[str]:
    """
    Human-readable names matching rolling_features()'s exact column order.
    Must stay in lockstep with that function — see its docstring for the
    per-channel, per-window block layout this mirrors.
    """
    names = []
    for col in sensors:
        is_binary = ch_types.get(col) == "binary"
        for w in windows:
            if is_binary:
                names += [f"{col}_mean_{w}s", f"{col}_std_{w}s"]
            else:
                names += [f"{col}_mean_{w}s", f"{col}_std_{w}s",
                          f"{col}_zscore_{w}s", f"{col}_absmax_{w}s"]
    return names


def explain_forecaster_row(bundle: dict, raw_data: np.ndarray, row_idx: int,
                           top_k: int = 8) -> dict:
    """
    Explain why row `row_idx` scored the way it did under an LSTM or
    PatchTST bundle, using Captum's Integrated Gradients.

    Unlike XGBoost's flat feature vector, a forecaster's input is a whole
    WINDOW of the past (window_seconds x n_channels) — so "explaining" a
    prediction means attributing importance across BOTH time and channel at
    once. Integrated Gradients handles this the same way it handles any
    input shape: it walks from a neutral baseline (here, all-zeros in the
    scaled/stationary space, which represents "no signal / typical value"
    since RobustScaler centers on the median) up to the real input, and
    accumulates how much each individual (timestep, channel) position
    pushed the model's prediction for the worst-scoring channel.

    Returns:
        {
          "channel": str, "predicted": float, "actual": float, "error": float,
          "attribution": np.ndarray,   # shape (window, n_channels), signed
          "sensors": list[str],        # column labels for the attribution grid
          "window": int,               # row labels count (most recent = last row)
          "convergence_delta": float,  # Captum's own approximation-quality check;
                                        # should be small relative to the prediction
                                        # magnitude — large values mean don't trust
                                        # this particular attribution closely.
        }
    """
    import torch
    from captum.attr import IntegratedGradients

    required = ("sensors", "scaler", "model", "channel_norm", "window")
    missing = [k for k in required if k not in bundle]
    if missing:
        raise KeyError(
            f"Forecaster bundle is missing expected key(s): {missing}. "
            f"Available keys: {list(bundle.keys())}. Retrain to refresh the model file."
        )

    sensors  = bundle["sensors"]
    ch_types = bundle.get("ch_types")
    if ch_types is None:
        from models import classify_channel
        ch_types = {s: classify_channel(raw_data[:, i]) for i, s in enumerate(sensors)}
    scaler   = bundle["scaler"]
    model    = bundle["model"]
    channel_norm = bundle["channel_norm"]
    window   = bundle["window"]

    stat = make_stationary(raw_data, sensors, ch_types)
    ds = scaler.transform(stat).astype(np.float32)

    if row_idx < window or row_idx >= len(ds):
        raise ValueError(
            f"row_idx {row_idx} needs at least {window} rows of history before it "
            f"(data has {len(ds)} rows total) — pick a later timestep."
        )

    # The window that predicts row_idx is the `window` rows immediately
    # before it — matches make_forecast_windows()'s end_idx = start + window.
    x_window = ds[row_idx - window: row_idx]              # (window, C)
    y_actual = ds[row_idx]                                 # (C,)

    model.eval()
    x_tensor = torch.from_numpy(x_window).float().unsqueeze(0)  # (1, window, C)
    x_tensor.requires_grad_(True)

    with torch.no_grad():
        pred = model(x_tensor).numpy()[0]                   # (C,)

    errors = {s: (pred[i] - y_actual[i])**2 / (channel_norm[i] + 1e-12)
             for i, s in enumerate(sensors)}
    worst_channel = max(errors, key=errors.get)
    worst_idx = sensors.index(worst_channel)

    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(x_tensor)
    attribution, convergence_delta = ig.attribute(
        x_tensor, baselines=baseline, target=worst_idx,
        n_steps=50, return_convergence_delta=True,
    )
    attribution = attribution.squeeze(0).detach().numpy()   # (window, C)

    return {
        "channel": worst_channel,
        "predicted": float(pred[worst_idx]),
        "actual": float(y_actual[worst_idx]),
        "error": float(errors[worst_channel]),
        "attribution": attribution,
        "sensors": sensors,
        "window": window,
        "convergence_delta": float(np.atleast_1d(convergence_delta.detach().numpy())[0]),
    }


def top_attribution_cells(result: dict, top_k: int = 8) -> list[tuple[str, int, float]]:
    """
    Flatten a Captum attribution grid down to the top_k individual
    (channel, seconds-ago) cells by absolute impact — the time-series
    equivalent of SHAP's top_features list, for a compact summary alongside
    the full heatmap.
    """
    attr = result["attribution"]          # (window, C)
    sensors = result["sensors"]
    window = result["window"]
    flat_idx = np.argsort(-np.abs(attr), axis=None)[:top_k]
    out = []
    for idx in flat_idx:
        t, c = np.unravel_index(idx, attr.shape)
        seconds_ago = window - t
        out.append((sensors[c], int(seconds_ago), float(attr[t, c])))
    return out


def humanize_feature_name(name: str) -> str:
    """
    Turn a raw feature column name like "bfo2_std_30s" into a plain-English
    fragment like "how much bfo2 was jumping around in the last 30 seconds".
    Used to make SHAP/Captum output readable to someone who doesn't know
    what a rolling standard deviation is.
    """
    parts = name.rsplit("_", 2)   # [channel, stat, "30s"]
    if len(parts) != 3:
        return name
    channel, stat, window = parts
    window_txt = window.rstrip("s") + " seconds"
    stat_phrases = {
        "mean":   f"{channel}'s typical value over the last {window_txt}",
        "std":    f"how much {channel} was jumping around in the last {window_txt}",
        "zscore": f"how unusual {channel}'s value was right then, compared to the last {window_txt}",
        "absmax": f"the biggest single swing {channel} made in the last {window_txt}",
    }
    return stat_phrases.get(stat, name)


def plain_summary(result: dict) -> str:
    """One-sentence plain-language summary of what the model expected vs.
    what actually happened, for either the SHAP or Captum result dict."""
    ch = result["channel"]
    pred, actual = result["predicted"], result["actual"]
    direction = "much higher than" if actual > pred else "much lower than"
    return (f"The model expected **{ch}** to read around **{pred:.2f}**, "
            f"but it actually read **{actual:.2f}** — {direction} expected.")


def plain_feature_reasons(top_features: list[tuple[str, float]], max_items: int = 3) -> list[dict]:
    """
    Convert SHAP's top_features list into plain-language reason cards:
    [{"text": "...", "direction": "up"/"down", "strength": float 0-1}, ...]
    strength is normalised against the strongest item so the UI can size/
    color things consistently regardless of the raw SHAP value's scale.
    """
    items = top_features[:max_items]
    if not items:
        return []
    max_abs = max(abs(v) for _, v in items) or 1.0
    out = []
    for name, val in items:
        out.append({
            "text": humanize_feature_name(name),
            "direction": "up" if val > 0 else "down",
            "strength": abs(val) / max_abs,
        })
    return out


def plain_attribution_reasons(top_cells: list[tuple[str, int, float]], max_items: int = 3) -> list[dict]:
    """Same idea as plain_feature_reasons but for Captum's (channel, seconds_ago,
    value) cells — used for the forecaster explanations."""
    items = top_cells[:max_items]
    if not items:
        return []
    max_abs = max(abs(v) for _, _, v in items) or 1.0
    out = []
    for sensor, seconds_ago, val in items:
        when = "just before this moment" if seconds_ago <= 2 else f"{seconds_ago} seconds before this moment"
        out.append({
            "text": f"{sensor}'s value {when}",
            "direction": "up" if val > 0 else "down",
            "strength": abs(val) / max_abs,
        })
    return out


def explain_xgboost_row(bundle: dict, raw_data: np.ndarray, row_idx: int,
                        top_k: int = 6) -> dict:
    """
    Explain why row `row_idx` scored the way it did under the XGBoost bundle.

    Returns:
        {
          "channel": str,            # which channel drove the highest score
          "predicted": float,        # what XGBoost expected for that channel
          "actual": float,           # what actually happened (stationary/scaled space)
          "error": float,            # squared error driving the anomaly score
          "top_features": [(name, shap_value), ...],  # sorted by |impact|
          "base_value": float,       # the model's average/baseline prediction
        }

    top_features' shap_value is signed: positive means that feature pushed
    the prediction UP, negative means it pushed it DOWN. The magnitude is
    how much it mattered, in the same units as the (scaled, stationary)
    prediction itself.
    """
    import shap

    required = ("sensors", "scaler", "models", "channel_norm")
    missing = [k for k in required if k not in bundle]
    if missing:
        raise KeyError(
            f"XGBoost bundle is missing expected key(s): {missing}. "
            f"Available keys: {list(bundle.keys())}. This usually means the "
            f"model was trained with a different version of train.py — retrain "
            f"to refresh models/xgboost.pkl."
        )

    sensors  = bundle["sensors"]
    # ch_types may be absent in older bundles; fall back to re-classifying
    # from the raw data rather than crashing, so this degrades gracefully.
    ch_types = bundle.get("ch_types")
    if ch_types is None:
        from models import classify_channel
        ch_types = {s: classify_channel(raw_data[:, i]) for i, s in enumerate(sensors)}
    scaler   = bundle["scaler"]
    models   = bundle["models"]
    channel_norm = bundle["channel_norm"]
    feat_windows = bundle.get("feat_windows", (10, 30, 60))

    missing_channels = [s for s in sensors if s not in models]
    if missing_channels:
        raise KeyError(
            f"XGBoost bundle's 'models' dict is missing regressor(s) for: "
            f"{missing_channels}. Available: {list(models.keys())}."
        )

    stat = make_stationary(raw_data, sensors, ch_types)
    ds = scaler.transform(stat).astype(np.float32)
    feats = np.nan_to_num(rolling_features(ds, sensors, feat_windows), nan=0.0)

    if row_idx < 0 or row_idx >= len(feats):
        raise ValueError(f"row_idx {row_idx} out of range for {len(feats)} rows")

    # Find which channel drove the highest anomaly contribution at this row
    preds = {s: models[s].predict(feats[row_idx:row_idx+1])[0] for s in sensors}
    errors = {s: (preds[s] - ds[row_idx, i])**2 / (channel_norm[i] + 1e-12)
             for i, s in enumerate(sensors)}
    worst_channel = max(errors, key=errors.get)

    # SHAP-explain that channel's regressor for this specific row
    explainer = shap.TreeExplainer(models[worst_channel])
    shap_values = explainer.shap_values(feats[row_idx:row_idx+1])[0]
    base_value = float(np.atleast_1d(explainer.expected_value)[0])

    names = rolling_feature_names(sensors, ch_types, feat_windows)
    order = np.argsort(-np.abs(shap_values))[:top_k]
    top_features = [(names[i], float(shap_values[i])) for i in order]

    ch_idx = sensors.index(worst_channel)
    return {
        "channel": worst_channel,
        "predicted": float(preds[worst_channel]),
        "actual": float(ds[row_idx, ch_idx]),
        "error": float(errors[worst_channel]),
        "top_features": top_features,
        "base_value": base_value,
    }
