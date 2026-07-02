

from __future__ import annotations
import numpy as np
import pandas as pd


PSI_STABLE_THRESHOLD   = 0.10
PSI_MODERATE_THRESHOLD = 0.25


def compute_psi(baseline: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    baseline = baseline[~np.isnan(baseline)]
    current  = current[~np.isnan(current)]
    if len(baseline) < n_bins or len(current) == 0:
        return 0.0

    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(baseline, quantiles))
    if len(edges) < 3:
        # Degenerate channel (near-constant) — bucket edges collapse.
        # Fall back to a coarser binning so PSI is still computable.
        edges = np.unique(np.percentile(baseline, np.linspace(0, 100, 4)))
        if len(edges) < 3:
            return 0.0

    edges[0], edges[-1] = -np.inf, np.inf   # catch any current-data outliers

    base_counts, _ = np.histogram(baseline, bins=edges)
    curr_counts, _ = np.histogram(current, bins=edges)

    base_pct = base_counts / max(1, base_counts.sum())
    curr_pct = curr_counts / max(1, curr_counts.sum())

    # Floor both at a small epsilon so an empty bucket doesn't produce
    # log(0) or divide-by-zero — standard PSI implementation detail.
    eps = 1e-4
    base_pct = np.maximum(base_pct, eps)
    curr_pct = np.maximum(curr_pct, eps)

    psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
    return float(psi)


def psi_severity(psi: float) -> str:
    if psi < PSI_STABLE_THRESHOLD:
        return "stable"
    if psi < PSI_MODERATE_THRESHOLD:
        return "moderate"
    return "significant"


def _drift_comparable_values(values: np.ndarray, ch_type: str) -> np.ndarray:
    if ch_type in ("drift", "oscillatory"):
        return np.diff(values)
    return values


def build_baseline_snapshot(raw_data: np.ndarray, sensors: list[str],
                            ch_types: dict, max_samples: int = 20_000) -> dict:
    rng = np.random.default_rng(42)
    snapshot = {}
    for i, s in enumerate(sensors):
        col = _drift_comparable_values(raw_data[:, i], ch_types.get(s, "constant"))
        if len(col) > max_samples:
            idx = rng.choice(len(col), max_samples, replace=False)
            col = col[idx]
        snapshot[s] = col.astype(np.float32)
    return {"sensors": sensors, "ch_types": ch_types, "samples": snapshot}


def check_drift(baseline_snapshot: dict, current_raw: np.ndarray,
                sensors: list[str], n_bins: int = 10) -> list[dict]:

    ch_types = baseline_snapshot.get("ch_types", {})
    reports = []
    for i, s in enumerate(sensors):
        if s not in baseline_snapshot["samples"]:
            continue
        baseline_vals = baseline_snapshot["samples"][s]
        current_vals  = _drift_comparable_values(current_raw[:, i], ch_types.get(s, "constant"))
        psi = compute_psi(baseline_vals, current_vals, n_bins=n_bins)
        reports.append({
            "channel": s,
            "psi": psi,
            "severity": psi_severity(psi),
            "baseline_mean": float(np.nanmean(baseline_vals)),
            "baseline_std":  float(np.nanstd(baseline_vals)),
            "current_mean":  float(np.nanmean(current_vals)),
            "current_std":   float(np.nanstd(current_vals)),
        })
    reports.sort(key=lambda r: r["psi"], reverse=True)
    return reports


def overall_drift_status(reports: list[dict]) -> str:
    """Roll per-channel reports up into one overall status for a quick glance."""
    if not reports:
        return "unknown"
    severities = [r["severity"] for r in reports]
    if "significant" in severities:
        return "significant"
    if "moderate" in severities:
        return "moderate"
    return "stable"
