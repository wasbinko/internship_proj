
from __future__ import annotations
import numpy as np
import pandas as pd

from models import make_stationary, rolling_features


def rolling_feature_names(sensors: list[str], ch_types: dict,
                          windows: tuple = (10, 30, 60)) -> list[str]:
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
