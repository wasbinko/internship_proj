

from __future__ import annotations
import os, pickle, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import (
    LSTMForecaster, PatchTST, StatDetector,
    make_stationary, make_forecast_windows, nn_forecast_errors,
    errors_to_timestep, rolling_features,
)


def _load_nn(model_dir, name, ctor):
    pt, meta = f"{model_dir}/{name}.pt", f"{model_dir}/{name}_meta.pkl"
    if not (os.path.exists(pt) and os.path.exists(meta)):
        return None
    b = pickle.load(open(meta, "rb"))
    C = b["channel_norm"].shape[0]
    b["model"] = ctor(b, C)
    b["model"].load_state_dict(torch.load(pt, map_location="cpu"))
    b["model"].eval()
    return b


def load_all(model_dir):
    bundles = {}
    nn_specs = {
        "lstm": lambda b, C: LSTMForecaster(C),
        "patchtst": lambda b, C: PatchTST(C, window=b["window"], patch_len=b["patch_len"],
                                          d_model=b.get("d_model",64), n_heads=b.get("n_heads",4),
                                          n_layers=b.get("n_layers",2)),
    }
    for name, ctor in nn_specs.items():
        try:
            b = _load_nn(model_dir, name, ctor)
            if b: bundles[name] = b; print(f"[LOAD] {name:10} ✓")
        except Exception as e:
            print(f"[LOAD] {name:10} ✗ ({e})")
    for name in ("xgboost", "iforest", "stat"):
        path = f"{model_dir}/{name}.pkl"
        if os.path.exists(path):
            try:
                bundles[name] = pickle.load(open(path, "rb")); print(f"[LOAD] {name:10} ✓")
            except Exception as e:
                print(f"[LOAD] {name:10} ✗ ({e})")
    return bundles


def _stationary(bundle, raw):
    return make_stationary(raw, bundle["sensors"], bundle["ch_types"])


def score_nn(bundle, raw, device="cpu"):
    stat = _stationary(bundle, raw)
    ds = bundle["scaler"].transform(stat).astype(np.float32)
    X, y, eidx = make_forecast_windows(ds, bundle["window"], stride=1)
    if len(X) == 0: return np.zeros(len(raw), np.float32)
    X = np.ascontiguousarray(X); y = np.ascontiguousarray(y)
    ws = nn_forecast_errors(bundle["model"], X, y, bundle["channel_norm"],
                            device, agg=bundle.get("agg","max"))
    return errors_to_timestep(ws, eidx, len(raw))


def score_xgboost(bundle, raw):
    stat = _stationary(bundle, raw)
    ds = bundle["scaler"].transform(stat).astype(np.float32)
    feats = np.nan_to_num(rolling_features(ds, bundle["sensors"], bundle["feat_windows"]), nan=0.0)
    preds = np.column_stack([bundle["models"][s].predict(feats) for s in bundle["sensors"]])
    normed = ((preds - ds)**2) / (bundle["channel_norm"][None,:] + 1e-12)
    agg = bundle.get("agg","max")
    if agg == "max":  return normed.max(axis=1).astype(np.float32)
    if agg == "top2": return np.partition(normed,-2,axis=1)[:,-2:].mean(axis=1).astype(np.float32)
    return normed.mean(axis=1).astype(np.float32)


def score_iforest(bundle, raw):
    stat = _stationary(bundle, raw)
    ds = bundle["scaler"].transform(stat).astype(np.float32)
    return bundle["detector"].score(ds)


def score_stat(bundle, raw):
    return bundle["detector"].score(raw)   # StatDetector works on RAW data


def score_all(bundles, raw, sensors, device="cpu", weights=None):
    scores = {}
    for name, b in bundles.items():
        try:
            if name in ("lstm","patchtst"): s = score_nn(b, raw, device)
            elif name == "xgboost":         s = score_xgboost(b, raw)
            elif name == "iforest":         s = score_iforest(b, raw)
            elif name == "stat":            s = score_stat(b, raw)
            else: continue
            scores[name] = np.nan_to_num(s, nan=0.0, posinf=1e9, neginf=0.0)
        except Exception as e:
            print(f"[SCORE] {name} failed: {e}")
    if len(scores) > 1:
        wts = weights or {}
        log = [wts.get(n,1.0)*np.log10(s+1e-9) for n,s in scores.items()]
        scores["ensemble"] = np.power(10.0, np.mean(log, axis=0)).astype(np.float32)
    return scores


def derive_threshold(scores, method="mad", k=6.0, pct=99.5, linear=False):
    
    s = scores[~np.isnan(scores)]
    if len(s) == 0:
        return 1.0

    if linear:
        med = np.median(s)
        mad = np.median(np.abs(s - med)) * 1.4826
        if mad < 1e-6:
            return float(np.percentile(s, 99.5))
        return float(med + k * mad)

    nonzero = s[s > 1e-6]
    if len(nonzero) < max(10, int(0.05*len(s))):
        return float(np.percentile(s, 99.5))
    log_s = np.log10(nonzero + 1e-9)
    if method == "mad":
        med = np.median(log_s); mad = np.median(np.abs(log_s-med))*1.4826
        if mad < 1e-6: return float(np.percentile(s, 99.5))
        return float(10**(med + k*mad))
    if method == "iqr":
        q1,q3 = np.percentile(log_s,25), np.percentile(log_s,75)
        if q3-q1 < 1e-6: return float(np.percentile(s,99.5))
        return float(10**(np.median(log_s) + k*(q3-q1)))
    return float(np.percentile(s, pct))


# Models whose scores are linear z-scores (use linear thresholding)
LINEAR_SCORE_MODELS = {"stat"}
