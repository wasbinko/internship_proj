
from __future__ import annotations
import argparse, os, pickle, sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import (
    LSTMForecaster, PatchTST, IsolationForestDetector, StatDetector,
    classify_channel, make_stationary, make_forecast_windows,
    calibrate_channel_norm, nn_forecast_errors, errors_to_timestep,
    rolling_features,
)
from drift import build_baseline_snapshot
# See infer.py for why this tries both import styles.
try:
    from nhits_model import (train_nhits, score_series as nhits_score_series,
                             calibrate_channel_norm as nhits_calibrate_channel_norm,
                             save_nhits, NHITS_AVAILABLE, NHITS_IMPORT_ERROR)
except ImportError:
    try:
        from scripts.nhits_model import (train_nhits, score_series as nhits_score_series,
                                         calibrate_channel_norm as nhits_calibrate_channel_norm,
                                         save_nhits, NHITS_AVAILABLE, NHITS_IMPORT_ERROR)
    except ImportError as e:
        NHITS_AVAILABLE = False
        NHITS_IMPORT_ERROR = f"{type(e).__name__}: {e}"

try:
    from tide_model import (train_tide, score_series as tide_score_series,
                            calibrate_channel_norm as tide_calibrate_channel_norm,
                            save_tide, TIDE_AVAILABLE, TIDE_IMPORT_ERROR)
except ImportError:
    try:
        from scripts.tide_model import (train_tide, score_series as tide_score_series,
                                        calibrate_channel_norm as tide_calibrate_channel_norm,
                                        save_tide, TIDE_AVAILABLE, TIDE_IMPORT_ERROR)
    except ImportError as e:
        TIDE_AVAILABLE = False
        TIDE_IMPORT_ERROR = f"{type(e).__name__}: {e}"

NON_SENSOR = {"timestamp","year","month","day","hour","minute","second",
              "y","label","anomaly","is_anomaly"}


# ─────────────────────────────────────────────────────────────────────────────
# MLflow helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_mlflow(enabled: bool, tracking_dir: str, experiment: str):
    if not enabled:
        return None
    try:
        import mlflow
    except ImportError:
        print("[MLFLOW] mlflow not installed — skipping tracking. `pip install mlflow` to enable.")
        return None
    os.makedirs(tracking_dir, exist_ok=True)
    db_path = os.path.join(os.path.abspath(tracking_dir), "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment(experiment)
    return mlflow


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation against a labeled test set (optional — for real P/R/F1 in MLflow)
# ─────────────────────────────────────────────────────────────────────────────

def load_labeled_eval(eval_dir: str, eval_labels: str):
    """Load a labeled test set built by make_labeled_data.py."""
    files = sorted(f for f in os.listdir(eval_dir) if f.endswith(".csv"))
    dfs = [pd.read_csv(os.path.join(eval_dir, f)) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    labels = np.load(eval_labels)
    return df, labels


def evaluate_predictions(pred: np.ndarray, labels: np.ndarray) -> dict:
    """Precision/recall/F1 plus segment-level recall (did we catch each
    contiguous anomaly at all, not just what fraction of its timesteps)."""
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    d = np.diff(np.concatenate([[0], labels, [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    n_segments = len(starts)
    segments_caught = sum(1 for s, e in zip(starts, ends) if pred[s:e].any())

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "n_segments": n_segments, "segments_caught": segments_caught,
        "segment_recall": segments_caught / n_segments if n_segments else 0.0,
    }


def smart_consensus_predictions(stat_scores, forecaster_predictions: dict,
                                stat_k: float, min_forecaster_agree: int = 2) -> np.ndarray:
    def smooth(sc, w=5):
        return pd.Series(sc).rolling(w, center=True, min_periods=1).median().values
    def linear_thr(sc, k):
        med = np.median(sc); mad = np.median(np.abs(sc - med)) * 1.4826
        return float(np.percentile(sc, 99.5)) if mad < 1e-6 else float(med + k * mad)

    fc_vote = np.sum(list(forecaster_predictions.values()), axis=0) if forecaster_predictions else None
    fc_strong = (fc_vote >= min_forecaster_agree).astype(int) if fc_vote is not None else None
    fc_any    = (fc_vote >= 1).astype(int) if fc_vote is not None else None

    if stat_scores is None:
        if fc_strong is not None:
            return fc_strong
        return np.zeros(0, dtype=int)

    stat_sc = smooth(stat_scores)
    strong = (stat_sc > linear_thr(stat_sc, stat_k * 1.5)).astype(int)
    weak   = (stat_sc > linear_thr(stat_sc, max(1.0, stat_k * 0.6))).astype(int)
    confirmed = np.maximum(strong, weak & (fc_any if fc_any is not None else 0))
    if fc_strong is not None:
        confirmed = np.maximum(confirmed, fc_strong)
    return confirmed


def load_telemetry(data_dir, max_rows=None):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")
    dfs, total = [], 0
    for f in files:
        d = pd.read_csv(os.path.join(data_dir, f), low_memory=False)
        dfs.append(d); total += len(d)
        if max_rows and total >= max_rows:
            break
    df = pd.concat(dfs, ignore_index=True)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df.head(max_rows) if max_rows else df


def load_telemetry_kafka(bootstrap_servers: str, topic: str, max_rows: int | None):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from kafka_io import TelemetryConsumer, chunks_to_dataframe

    rows_per_chunk = 300
    n_chunks = min(5000, (max_rows // rows_per_chunk + 5) if max_rows else 500)

    consumer = TelemetryConsumer(bootstrap_servers=bootstrap_servers, topic=topic, group_id=None)
    chunks = consumer.read_last_n_chunks(n_chunks)
    consumer.close()

    df = chunks_to_dataframe(chunks)
    if df is None or len(df) == 0:
        raise RuntimeError(
            f"No messages found on Kafka topic '{topic}' @ {bootstrap_servers}. "
            f"Is the producer running (generate_telemetry.py --sink kafka)?"
        )
    return df.head(max_rows) if max_rows else df


def detect_sensors(df, override=None):
    if override:
        return [c for c in override if c in df.columns]
    num = df.select_dtypes(include=[np.number]).columns
    return [c for c in num if c not in NON_SENSOR and df[c].std() > 0]


def get_device(force_cpu=False):
    return "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"


def train_nn(model, X, y, epochs, bs, lr, val_frac, device, label):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    n = len(X); n_val = int(n*val_frac)
    perm = np.random.permutation(n); vi, ti = perm[:n_val], perm[n_val:]
    Xtr, ytr = torch.from_numpy(X[ti]), torch.from_numpy(y[ti])
    Xv, yv = torch.from_numpy(X[vi]).to(device), torch.from_numpy(y[vi]).to(device)
    best, best_state = float("inf"), None
    for ep in range(epochs):
        model.train(); order = torch.randperm(len(Xtr)); run=0; nb=0
        for s in range(0, len(Xtr), bs):
            idx = order[s:s+bs]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            run += loss.item(); nb += 1
        model.eval()
        with torch.no_grad(): vl = crit(model(Xv), yv).item()
        if vl < best:
            best = vl; best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
        print(f"  [{label}] ep {ep+1:>2}/{epochs} train={run/max(1,nb):.5f} val={vl:.5f}")
    if best_state: model.load_state_dict(best_state)
    return model.cpu()


def trim_contamination(scores_per_row, trim_frac, smooth_window=15, pad=5):
    import pandas as pd
    n = len(scores_per_row)
    if trim_frac <= 0:
        return np.ones(n, dtype=bool)

    smoothed = pd.Series(scores_per_row).rolling(
        smooth_window, center=True, min_periods=1).max().values
    cutoff = np.percentile(smoothed, (1 - trim_frac) * 100)
    flagged = smoothed > cutoff

    # Pad each flagged segment slightly so we don't leave a sliver of
    # transition rows right at a real anomaly's edge in the "clean" set.
    keep_mask = ~flagged
    d = np.diff(np.concatenate([[0], flagged.astype(int), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    for s, e in zip(starts, ends):
        keep_mask[max(0, s - pad): min(n, e + pad)] = False

    return keep_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["csv", "kafka"], default="csv",
                   help="Where to load training data from. csv = read files from "
                        "--data_dir (original behaviour). kafka = read historical "
                        "chunks directly from a topic — use this if you're running "
                        "generate_telemetry.py with --sink kafka and no CSV files "
                        "land on disk anymore.")
    p.add_argument("--data_dir", default="live_telemetry_stream",
                   help="Only used with --source csv.")
    p.add_argument("--kafka_bootstrap", default="localhost:9092")
    p.add_argument("--kafka_topic", default="telemetry.raw")
    p.add_argument("--model_dir", default="models")
    p.add_argument("--max_rows", type=int, default=200_000)
    p.add_argument("--sensors", nargs="+", default=None)
    p.add_argument("--models", nargs="+",
                   default=["lstm","patchtst","xgboost","iforest","stat"],
                   choices=["lstm","patchtst","xgboost","iforest","stat","nhits","tide"],
                   help="'nhits' and 'tide' are opt-in (not trained by default) -- "
                        "both are heavier third-party dependencies (Darts + PyTorch "
                        "Lightning) offering more sophisticated forecasting "
                        "architectures as alternatives to the from-scratch "
                        "LSTM/PatchTST. Same role, same interface, fancier engines. "
                        "TiDE is the lighter-weight of the two -- an MLP-based "
                        "architecture that trains faster than NHITS with comparable "
                        "accuracy on standard benchmarks.")
    p.add_argument("--trim", type=float, default=0.25,
                   help="Fraction of worst-scoring training rows to drop as "
                        "likely-contaminated before final training. 0 = data is clean.")
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--max_windows", type=int, default=50_000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--agg", default="max", choices=["mean","max","top2"])
    p.add_argument("--stat_window", type=int, default=30)
    p.add_argument("--patch_len", type=int, default=12)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--xgb_trees", type=int, default=200)
    p.add_argument("--if_trees", type=int, default=150)
    p.add_argument("--nhits_epochs", type=int, default=25,
                   help="NHITS trains noticeably slower than the from-scratch "
                        "models (heavier architecture, Lightning overhead), so "
                        "it gets its own epoch count rather than sharing --epochs.")
    p.add_argument("--nhits_stacks", type=int, default=3)
    p.add_argument("--nhits_width", type=int, default=128)
    p.add_argument("--tide_epochs", type=int, default=25)
    p.add_argument("--tide_hidden", type=int, default=128)
    p.add_argument("--tide_encoder_layers", type=int, default=1)
    p.add_argument("--tide_decoder_layers", type=int, default=1)
    p.add_argument("--force_cpu", action="store_true")
    # MLflow tracking
    p.add_argument("--mlflow", dest="use_mlflow", action="store_true", default=True,
                   help="Enable MLflow tracking (default on — local file store, no server needed).")
    p.add_argument("--no_mlflow", dest="use_mlflow", action="store_false",
                   help="Disable MLflow tracking entirely.")
    p.add_argument("--mlflow_dir", default="mlruns",
                   help="Local directory for the MLflow file-store tracking data.")
    p.add_argument("--mlflow_experiment", default="telemetry-anomaly-detection")
    p.add_argument("--run_name", default=None,
                   help="Optional name for this training run (shown in the MLflow UI).")
    # Optional evaluation against a labeled test set (see make_labeled_data.py)
    p.add_argument("--eval_dir", default=None,
                   help="Directory of labeled test CSVs to evaluate against after training.")
    p.add_argument("--eval_labels", default=None,
                   help="Path to the corresponding labels .npy file.")
    args = p.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    device = get_device(args.force_cpu)
    np.random.seed(42); torch.manual_seed(42)

    mlflow = get_mlflow(args.use_mlflow, args.mlflow_dir, args.mlflow_experiment)
    from contextlib import nullcontext

    if args.source == "csv":
        df = load_telemetry(args.data_dir, args.max_rows)
    else:
        print(f"[INFO] Loading training data from Kafka topic '{args.kafka_topic}' "
              f"@ {args.kafka_bootstrap} ...")
        df = load_telemetry_kafka(args.kafka_bootstrap, args.kafka_topic, args.max_rows)
    sensors = detect_sensors(df, args.sensors)
    ch_types = {s: classify_channel(df[s].values) for s in sensors}
    print(f"[INFO] {len(df):,} rows | sensors: {sensors} | device: {device}")
    print(f"[INFO] Channel types: {ch_types}")
    print(f"[INFO] Contamination trim: {args.trim:.0%}")

    eval_df, eval_labels = None, None
    if args.eval_dir and args.eval_labels:
        eval_df, eval_labels = load_labeled_eval(args.eval_dir, args.eval_labels)
        print(f"[INFO] Evaluation set: {len(eval_df):,} rows, "
              f"{eval_labels.sum():,} anomalous ({eval_labels.mean()*100:.1f}%)")

    raw = df[sensors].values.astype(np.float32)

    drift_baseline = build_baseline_snapshot(raw, sensors, ch_types)
    with open(os.path.join(args.model_dir, "drift_baseline.pkl"), "wb") as f:
        pickle.dump(drift_baseline, f)
    print(f"[INFO] Saved drift-monitoring baseline "
          f"({sum(len(v) for v in drift_baseline['samples'].values()):,} sample points)")

    stat_raw = make_stationary(raw, sensors, ch_types)
    scaler = RobustScaler().fit(stat_raw)
    stat_scaled = scaler.transform(stat_raw).astype(np.float32)

    # ── Contamination trimming: quick LSTM to score rows, drop worst ──
    keep_mask = np.ones(len(raw), dtype=bool)
    if args.trim > 0 and any(m in args.models for m in ["lstm","patchtst","xgboost"]):
        print(f"\n[TRIM] Training quick scout model to identify clean rows ...")
        Xs, ys, eidx = make_forecast_windows(stat_scaled, args.window, stride=2)
        Xs = np.ascontiguousarray(Xs).astype(np.float32); ys = np.ascontiguousarray(ys).astype(np.float32)
        if len(Xs) > 20000:
            sel = np.random.choice(len(Xs), 20000, replace=False); sel.sort()
            Xs, ys, eidx = Xs[sel], ys[sel], eidx[sel]
        scout = LSTMForecaster(len(sensors), args.hidden, 1)
        scout = train_nn(scout, Xs, ys, max(5, args.epochs//2), args.batch_size,
                         args.lr, args.val_frac, device, "scout")
        cn = calibrate_channel_norm(scout, stat_scaled, args.window, device)
        Xa, ya, ea = make_forecast_windows(stat_scaled, args.window, stride=1)
        Xa = np.ascontiguousarray(Xa); ya = np.ascontiguousarray(ya)
        ws = nn_forecast_errors(scout, Xa, ya, cn, device, agg="max")
        row_scores = errors_to_timestep(ws, ea, len(raw))
        keep_mask = trim_contamination(row_scores, args.trim)
        print(f"[TRIM] Keeping {keep_mask.sum():,}/{len(raw):,} rows "
              f"({keep_mask.mean()*100:.0f}%) as clean baseline")

    # Clean (trimmed) views
    raw_clean = raw[keep_mask]
    stat_clean = make_stationary(raw_clean, sensors, ch_types)
    scaler = RobustScaler().fit(stat_clean)
    stat_clean_scaled = scaler.transform(stat_clean).astype(np.float32)

    saved = {}
    t0 = time.time()
    common = dict(sensors=sensors, ch_types=ch_types, scaler=scaler, agg=args.agg)

    # Predictions on the eval set, collected per model, for the final Smart
    # Consensus evaluation logged on the parent run.
    eval_predictions = {}
    eval_stat_scores = None

    parent_ctx = (mlflow.start_run(run_name=args.run_name) if mlflow
                  else nullcontext())
    with parent_ctx:
        if mlflow:
            mlflow.log_params({
                "data_dir": args.data_dir if args.source == "csv" else f"kafka:{args.kafka_topic}",
                "source": args.source, "n_rows": len(df), "sensors": ",".join(sensors),
                "trim": args.trim, "window": args.window, "stride": args.stride,
                "agg": args.agg, "models": ",".join(args.models),
                "max_windows": args.max_windows,
            })
            for s, t in ch_types.items():
                mlflow.log_param(f"channel_type_{s}", t)

        # ── LSTM ──
        if "lstm" in args.models:
            print(f"\n{'='*50}\n[LSTM]\n{'='*50}")
            child_ctx = mlflow.start_run(run_name="lstm", nested=True) if mlflow else nullcontext()
            with child_ctx:
                X,y,_ = make_forecast_windows(stat_clean_scaled, args.window, stride=args.stride)
                X=np.ascontiguousarray(X).astype(np.float32); y=np.ascontiguousarray(y).astype(np.float32)
                if len(X)>args.max_windows:
                    sel=np.random.choice(len(X),args.max_windows,replace=False); sel.sort(); X,y=X[sel],y[sel]
                m = train_nn(LSTMForecaster(len(sensors),args.hidden,1), X,y, args.epochs,
                             args.batch_size,args.lr,args.val_frac,device,"LSTM")
                cn = calibrate_channel_norm(m, stat_clean_scaled, args.window, device)
                nom_scores = nn_forecast_errors(
                    m, *make_forecast_windows(stat_clean_scaled, args.window, stride=1)[:2],
                    cn, device, agg=args.agg)
                nom_smooth = pd.Series(nom_scores).rolling(5, center=True, min_periods=1).median().values
                thr_p99, thr_p995, thr_p999 = (float(np.percentile(nom_smooth, p)) for p in (99, 99.5, 99.9))
                print(f"  Nominal score thresholds: p99={thr_p99:.2f} p99.5={thr_p995:.2f} p99.9={thr_p999:.2f}")
                lstm_path = f"{args.model_dir}/lstm.pt"
                lstm_meta_path = f"{args.model_dir}/lstm_meta.pkl"
                torch.save(m.state_dict(), lstm_path)
                pickle.dump({**common,"window":args.window,"channel_norm":cn,
                             "threshold_p99":thr_p99,"threshold_p995":thr_p995,"threshold_p999":thr_p999,
                             "model_type":"LSTMForecaster"}, open(lstm_meta_path,"wb"))
                saved["lstm"]=1; print("  → saved")

                if mlflow:
                    mlflow.log_params({"epochs":args.epochs,"hidden":args.hidden,
                                       "lr":args.lr,"batch_size":args.batch_size})
                    mlflow.log_metrics({"threshold_p99":thr_p99,"threshold_p995":thr_p995,
                                        "threshold_p999":thr_p999,"nominal_score_median":float(np.median(nom_smooth))})
                    mlflow.log_artifact(lstm_path); mlflow.log_artifact(lstm_meta_path)

                if eval_df is not None:
                    ds = scaler.transform(make_stationary(eval_df[sensors].values.astype(np.float32), sensors, ch_types)).astype(np.float32)
                    Xe, ye, ee = make_forecast_windows(ds, args.window, stride=1)
                    we = nn_forecast_errors(m, np.ascontiguousarray(Xe), np.ascontiguousarray(ye), cn, device, agg=args.agg)
                    sc_e = errors_to_timestep(we, ee, len(eval_df))
                    sc_e_smooth = pd.Series(sc_e).rolling(5, center=True, min_periods=1).median().values
                    pred_e = (sc_e_smooth > thr_p995).astype(int)
                    eval_predictions["lstm"] = pred_e
                    metrics = evaluate_predictions(pred_e, eval_labels)
                    print(f"  Eval: P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                          f"F1={metrics['f1']:.3f} segments={metrics['segments_caught']}/{metrics['n_segments']}")
                    if mlflow:
                        mlflow.log_metrics({f"eval_{k}": v for k,v in metrics.items()})

        # ── NHITS (via Darts) — opt-in, not trained by default ──
        if "nhits" in args.models:
            print(f"\n{'='*50}\n[NHITS]\n{'='*50}")
            if not NHITS_AVAILABLE:
                print(f"  SKIPPED: darts/pytorch-lightning import failed. "
                      f"Real error: {NHITS_IMPORT_ERROR}. If you've already run "
                      f"`pip install darts pytorch-lightning`, this is likely a version "
                      f"mismatch between them and your installed torch.")
            else:
                child_ctx = mlflow.start_run(run_name="nhits", nested=True) if mlflow else nullcontext()
                with child_ctx:
                    
                    nhits_model = train_nhits(
                        stat_clean_scaled, sensors, window=args.window,
                        epochs=args.nhits_epochs, num_stacks=args.nhits_stacks,
                        layer_widths=args.nhits_width,
                    )
                    full_stat_n = make_stationary(raw, sensors, ch_types)
                    full_scaled_n = scaler.transform(full_stat_n).astype(np.float32)
                    cn = nhits_calibrate_channel_norm(nhits_model, full_scaled_n, sensors, args.window)
                    nom_scores = nhits_score_series(nhits_model, full_scaled_n, sensors,
                                                    args.window, cn, agg=args.agg)
                    nom_smooth = pd.Series(nom_scores).rolling(5, center=True, min_periods=1).median().values

                    # Robust median+MAD-in-log-space instead of a raw
                    # percentile -- same reasoning as XGBoost's fix above.
                    log_sc = np.log10(np.maximum(nom_smooth, 1e-9))
                    log_med, log_mad = float(np.median(log_sc)), float(np.median(np.abs(log_sc - np.median(log_sc))) * 1.4826)
                    def _robust_log_thr_n(k):
                        return float(10 ** (log_med + k * log_mad)) if log_mad > 1e-6 else float(np.percentile(nom_smooth, 99.5))
                    thr_p99, thr_p995, thr_p999 = _robust_log_thr_n(3.0), _robust_log_thr_n(4.5), _robust_log_thr_n(7.0)
                    print(f"  Nominal score thresholds (full-data, robust): "
                          f"p99={thr_p99:.2f} p99.5={thr_p995:.2f} p99.9={thr_p999:.2f}")
                    nhits_path = f"{args.model_dir}/nhits.pt"
                    nhits_meta_path = f"{args.model_dir}/nhits_meta.pkl"
                    save_nhits(nhits_model, nhits_path)
                    pickle.dump({**common, "window": args.window, "channel_norm": cn,
                                "threshold_p99": thr_p99, "threshold_p995": thr_p995,
                                "threshold_p999": thr_p999, "model_type": "NHITS"},
                               open(nhits_meta_path, "wb"))
                    saved["nhits"] = 1; print("  → saved")

                    if mlflow:
                        mlflow.log_params({"epochs": args.nhits_epochs, "num_stacks": args.nhits_stacks,
                                           "layer_widths": args.nhits_width, "window": args.window,
                                           "calibration": "full_untrimmed_robust"})
                        mlflow.log_metrics({"threshold_p99": thr_p99, "threshold_p995": thr_p995,
                                            "threshold_p999": thr_p999,
                                            "nominal_score_median": float(np.median(nom_smooth))})
                        mlflow.log_artifact(nhits_path); mlflow.log_artifact(nhits_meta_path)

                    if eval_df is not None:
                        ds = scaler.transform(make_stationary(
                            eval_df[sensors].values.astype(np.float32), sensors, ch_types)).astype(np.float32)
                        sc_e = nhits_score_series(nhits_model, ds, sensors, args.window, cn, agg=args.agg)
                        sc_e_smooth = pd.Series(sc_e).rolling(5, center=True, min_periods=1).median().values
                        pred_e = (sc_e_smooth > thr_p995).astype(int)
                        eval_predictions["nhits"] = pred_e
                        metrics = evaluate_predictions(pred_e, eval_labels)
                        print(f"  Eval: P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                              f"F1={metrics['f1']:.3f} segments={metrics['segments_caught']}/{metrics['n_segments']}")
                        if mlflow:
                            mlflow.log_metrics({f"eval_{k}": v for k, v in metrics.items()})

        # ── TiDE (via Darts) — opt-in, not trained by default ──
        if "tide" in args.models:
            print(f"\n{'='*50}\n[TiDE]\n{'='*50}")
            if not TIDE_AVAILABLE:
                print(f"  SKIPPED: darts/pytorch-lightning import failed. "
                      f"Real error: {TIDE_IMPORT_ERROR}. If you've already run "
                      f"`pip install darts pytorch-lightning`, this is likely a version "
                      f"mismatch between them and your installed torch.")
            else:
                child_ctx = mlflow.start_run(run_name="tide", nested=True) if mlflow else nullcontext()
                with child_ctx:
                    tide_model = train_tide(
                        stat_clean_scaled, sensors, window=args.window,
                        epochs=args.tide_epochs, hidden_size=args.tide_hidden,
                        num_encoder_layers=args.tide_encoder_layers,
                        num_decoder_layers=args.tide_decoder_layers,
                    )
                    full_stat_t = make_stationary(raw, sensors, ch_types)
                    full_scaled_t = scaler.transform(full_stat_t).astype(np.float32)
                    cn = tide_calibrate_channel_norm(tide_model, full_scaled_t, sensors, args.window)
                    nom_scores = tide_score_series(tide_model, full_scaled_t, sensors,
                                                   args.window, cn, agg=args.agg)
                    nom_smooth = pd.Series(nom_scores).rolling(5, center=True, min_periods=1).median().values

                    log_sc = np.log10(np.maximum(nom_smooth, 1e-9))
                    log_med, log_mad = float(np.median(log_sc)), float(np.median(np.abs(log_sc - np.median(log_sc))) * 1.4826)
                    def _robust_log_thr_t(k):
                        return float(10 ** (log_med + k * log_mad)) if log_mad > 1e-6 else float(np.percentile(nom_smooth, 99.5))
                    thr_p99, thr_p995, thr_p999 = _robust_log_thr_t(3.0), _robust_log_thr_t(4.5), _robust_log_thr_t(7.0)
                    print(f"  Nominal score thresholds (full-data, robust): "
                          f"p99={thr_p99:.2f} p99.5={thr_p995:.2f} p99.9={thr_p999:.2f}")
                    tide_path = f"{args.model_dir}/tide.pt"
                    tide_meta_path = f"{args.model_dir}/tide_meta.pkl"
                    save_tide(tide_model, tide_path)
                    pickle.dump({**common, "window": args.window, "channel_norm": cn,
                                "threshold_p99": thr_p99, "threshold_p995": thr_p995,
                                "threshold_p999": thr_p999, "model_type": "TiDE"},
                               open(tide_meta_path, "wb"))
                    saved["tide"] = 1; print("  → saved")

                    if mlflow:
                        mlflow.log_params({"epochs": args.tide_epochs, "hidden_size": args.tide_hidden,
                                           "num_encoder_layers": args.tide_encoder_layers,
                                           "num_decoder_layers": args.tide_decoder_layers,
                                           "window": args.window, "calibration": "full_untrimmed_robust"})
                        mlflow.log_metrics({"threshold_p99": thr_p99, "threshold_p995": thr_p995,
                                            "threshold_p999": thr_p999,
                                            "nominal_score_median": float(np.median(nom_smooth))})
                        mlflow.log_artifact(tide_path); mlflow.log_artifact(tide_meta_path)

                    if eval_df is not None:
                        ds = scaler.transform(make_stationary(
                            eval_df[sensors].values.astype(np.float32), sensors, ch_types)).astype(np.float32)
                        sc_e = tide_score_series(tide_model, ds, sensors, args.window, cn, agg=args.agg)
                        sc_e_smooth = pd.Series(sc_e).rolling(5, center=True, min_periods=1).median().values
                        pred_e = (sc_e_smooth > thr_p995).astype(int)
                        eval_predictions["tide"] = pred_e
                        metrics = evaluate_predictions(pred_e, eval_labels)
                        print(f"  Eval: P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                              f"F1={metrics['f1']:.3f} segments={metrics['segments_caught']}/{metrics['n_segments']}")
                        if mlflow:
                            mlflow.log_metrics({f"eval_{k}": v for k, v in metrics.items()})

        # ── PatchTST ──
        if "patchtst" in args.models:
            print(f"\n{'='*50}\n[PatchTST]\n{'='*50}")
            child_ctx = mlflow.start_run(run_name="patchtst", nested=True) if mlflow else nullcontext()
            with child_ctx:
                pl=args.patch_len; w=(args.window//pl)*pl or pl*4
                X,y,_ = make_forecast_windows(stat_clean_scaled, w, stride=args.stride)
                X=np.ascontiguousarray(X).astype(np.float32); y=np.ascontiguousarray(y).astype(np.float32)
                if len(X)>args.max_windows:
                    sel=np.random.choice(len(X),args.max_windows,replace=False); sel.sort(); X,y=X[sel],y[sel]
                m = train_nn(PatchTST(len(sensors),w,pl,args.d_model,args.n_heads,args.n_layers),
                             X,y,args.epochs,args.batch_size,args.lr,args.val_frac,device,"PatchTST")
                cn = calibrate_channel_norm(m, stat_clean_scaled, w, device)
                nom_scores = nn_forecast_errors(
                    m, *make_forecast_windows(stat_clean_scaled, w, stride=1)[:2],
                    cn, device, agg=args.agg)
                nom_smooth = pd.Series(nom_scores).rolling(5, center=True, min_periods=1).median().values
                thr_p99, thr_p995, thr_p999 = (float(np.percentile(nom_smooth, p)) for p in (99, 99.5, 99.9))
                print(f"  Nominal score thresholds: p99={thr_p99:.2f} p99.5={thr_p995:.2f} p99.9={thr_p999:.2f}")
                pt_path = f"{args.model_dir}/patchtst.pt"
                pt_meta_path = f"{args.model_dir}/patchtst_meta.pkl"
                torch.save(m.state_dict(), pt_path)
                pickle.dump({**common,"window":w,"patch_len":pl,"d_model":args.d_model,
                             "n_heads":args.n_heads,"n_layers":args.n_layers,
                             "channel_norm":cn,
                             "threshold_p99":thr_p99,"threshold_p995":thr_p995,"threshold_p999":thr_p999,
                             "model_type":"PatchTST"}, open(pt_meta_path,"wb"))
                saved["patchtst"]=1; print("  → saved")

                if mlflow:
                    mlflow.log_params({"epochs":args.epochs,"d_model":args.d_model,
                                       "n_heads":args.n_heads,"n_layers":args.n_layers,
                                       "patch_len":pl,"window":w})
                    mlflow.log_metrics({"threshold_p99":thr_p99,"threshold_p995":thr_p995,
                                        "threshold_p999":thr_p999,"nominal_score_median":float(np.median(nom_smooth))})
                    mlflow.log_artifact(pt_path); mlflow.log_artifact(pt_meta_path)

                if eval_df is not None:
                    ds = scaler.transform(make_stationary(eval_df[sensors].values.astype(np.float32), sensors, ch_types)).astype(np.float32)
                    Xe, ye, ee = make_forecast_windows(ds, w, stride=1)
                    we = nn_forecast_errors(m, np.ascontiguousarray(Xe), np.ascontiguousarray(ye), cn, device, agg=args.agg)
                    sc_e = errors_to_timestep(we, ee, len(eval_df))
                    sc_e_smooth = pd.Series(sc_e).rolling(5, center=True, min_periods=1).median().values
                    pred_e = (sc_e_smooth > thr_p995).astype(int)
                    eval_predictions["patchtst"] = pred_e
                    metrics = evaluate_predictions(pred_e, eval_labels)
                    print(f"  Eval: P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                          f"F1={metrics['f1']:.3f} segments={metrics['segments_caught']}/{metrics['n_segments']}")
                    if mlflow:
                        mlflow.log_metrics({f"eval_{k}": v for k,v in metrics.items()})

        # ── XGBoost ──
        if "xgboost" in args.models:
            import xgboost as xgb
            print(f"\n{'='*50}\n[XGBoost]\n{'='*50}")
            child_ctx = mlflow.start_run(run_name="xgboost", nested=True) if mlflow else nullcontext()
            with child_ctx:
                W = max(args.window, 30)
                feats = np.nan_to_num(rolling_features(stat_clean_scaled, sensors, (10, W//2, W)), nan=0.0)
                targets = np.roll(stat_clean_scaled, -1, axis=0); targets[-1]=targets[-2]
                if len(feats)>args.max_windows:
                    idx=np.random.choice(len(feats),args.max_windows,replace=False); idx.sort()
                    ftr,ttr=feats[idx],targets[idx]
                else: ftr,ttr=feats,targets
                mdls={}
                for i,s in enumerate(sensors):
                    reg=xgb.XGBRegressor(n_estimators=args.xgb_trees,max_depth=5,learning_rate=0.05,
                                         subsample=0.8,colsample_bytree=0.8,tree_method="hist",
                                         random_state=42,verbosity=0)
                    reg.fit(ftr,ttr[:,i]); mdls[s]=reg
                preds=np.column_stack([mdls[s].predict(feats) for s in sensors])

                full_stat = make_stationary(raw, sensors, ch_types)
                full_scaled = scaler.transform(full_stat).astype(np.float32)
                full_feats = np.nan_to_num(rolling_features(full_scaled, sensors, (10, W//2, W)), nan=0.0)
                full_preds = np.column_stack([mdls[s].predict(full_feats) for s in sensors])
                cn=((full_preds-full_scaled)**2).mean(axis=0)
                cn=np.maximum(cn,max(1e-8,float(np.median(cn))*1e-3))
                agg_fn = {"max": lambda a: a.max(axis=1),
                          "top2": lambda a: np.partition(a,-2,axis=1)[:,-2:].mean(axis=1),
                          "mean": lambda a: a.mean(axis=1)}[args.agg]
                full_normed = ((full_preds - full_scaled)**2) / (cn[None,:] + 1e-12)
                nom_scores = agg_fn(full_normed)
                nom_smooth = pd.Series(nom_scores).rolling(5, center=True, min_periods=1).median().values

                log_sc = np.log10(np.maximum(nom_smooth, 1e-9))
                log_med, log_mad = float(np.median(log_sc)), float(np.median(np.abs(log_sc - np.median(log_sc))) * 1.4826)
                def _robust_log_thr(k):
                    return float(10 ** (log_med + k * log_mad)) if log_mad > 1e-6 else float(np.percentile(nom_smooth, 99.5))
                thr_p99, thr_p995, thr_p999 = _robust_log_thr(3.0), _robust_log_thr(4.5), _robust_log_thr(7.0)
                print(f"  Nominal score thresholds (full-data, robust): "
                      f"p99={thr_p99:.2f} p99.5={thr_p995:.2f} p99.9={thr_p999:.2f}")
                xgb_path = f"{args.model_dir}/xgboost.pkl"
                pickle.dump({**common,"window":W,"channel_norm":cn,"feat_windows":(10,W//2,W),
                             "models":mdls,
                             "threshold_p99":thr_p99,"threshold_p995":thr_p995,"threshold_p999":thr_p999,
                             "model_type":"XGBoost"}, open(xgb_path,"wb"))
                saved["xgboost"]=1; print("  → saved")

                if mlflow:
                    mlflow.log_params({"xgb_trees":args.xgb_trees,"window":W,"max_depth":5,
                                       "learning_rate":0.05})
                    mlflow.log_metrics({"threshold_p99":thr_p99,"threshold_p995":thr_p995,
                                        "threshold_p999":thr_p999,"nominal_score_median":float(np.median(nom_smooth))})
                    mlflow.log_artifact(xgb_path)

                if eval_df is not None:
                    ds = scaler.transform(make_stationary(eval_df[sensors].values.astype(np.float32), sensors, ch_types)).astype(np.float32)
                    feats_e = np.nan_to_num(rolling_features(ds, sensors, (10, W//2, W)), nan=0.0)
                    preds_e = np.column_stack([mdls[s].predict(feats_e) for s in sensors])
                    normed_e = ((preds_e - ds)**2) / (cn[None,:] + 1e-12)
                    sc_e = agg_fn(normed_e)
                    sc_e_smooth = pd.Series(sc_e).rolling(5, center=True, min_periods=1).median().values
                    pred_e = (sc_e_smooth > thr_p995).astype(int)
                    eval_predictions["xgboost"] = pred_e
                    metrics = evaluate_predictions(pred_e, eval_labels)
                    print(f"  Eval: P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                          f"F1={metrics['f1']:.3f} segments={metrics['segments_caught']}/{metrics['n_segments']}")
                    if mlflow:
                        mlflow.log_metrics({f"eval_{k}": v for k,v in metrics.items()})

        # ── IsolationForest ──
        if "iforest" in args.models:
            print(f"\n{'='*50}\n[IsolationForest]\n{'='*50}")
            child_ctx = mlflow.start_run(run_name="iforest", nested=True) if mlflow else nullcontext()
            with child_ctx:
                det = IsolationForestDetector(args.if_trees, contamination=0.02)
                det.fit(stat_clean_scaled, sensors)

                full_stat_if = make_stationary(raw, sensors, ch_types)
                full_scaled_if = scaler.transform(full_stat_if).astype(np.float32)
                ns = det.score(full_scaled_if)
                nom_smooth_if = pd.Series(ns).rolling(5, center=True, min_periods=1).median().values
                thr99, thr995, thr999 = (float(np.percentile(nom_smooth_if, p)) for p in (99, 99.5, 99.9))
                print(f"  Nominal score thresholds (full-data, percentile): "
                      f"p99={thr99:.4f} p99.5={thr995:.4f} p99.9={thr999:.4f}")

                if_path = f"{args.model_dir}/iforest.pkl"
                pickle.dump({**common,"detector":det,
                            "threshold_p99":thr99,"threshold_p995":thr995,"threshold_p999":thr999,
                            "model_type":"IsolationForest"}, open(if_path,"wb"))
                saved["iforest"]=1; print("  → saved")

                if mlflow:
                    mlflow.log_params({"if_trees":args.if_trees,"contamination":0.02,
                                       "calibration":"full_untrimmed_robust"})
                    mlflow.log_metrics({"threshold_p99":thr99,"threshold_p995":thr995,"threshold_p999":thr999})
                    mlflow.log_artifact(if_path)

                if eval_df is not None:
                    ds = scaler.transform(make_stationary(eval_df[sensors].values.astype(np.float32), sensors, ch_types)).astype(np.float32)
                    sc_e = det.score(ds)
                    pred_e = (sc_e > thr995).astype(int)
                    eval_predictions["iforest"] = pred_e
                    metrics = evaluate_predictions(pred_e, eval_labels)
                    print(f"  Eval: P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                          f"F1={metrics['f1']:.3f} segments={metrics['segments_caught']}/{metrics['n_segments']}")
                    if mlflow:
                        mlflow.log_metrics({f"eval_{k}": v for k,v in metrics.items()})

        # ── StatDetector (calibrated on RAW clean data, not stationary) ──
        if "stat" in args.models:
            print(f"\n{'='*50}\n[StatDetector]\n{'='*50}")
            child_ctx = mlflow.start_run(run_name="stat", nested=True) if mlflow else nullcontext()
            with child_ctx:
                sd = StatDetector(window=args.stat_window)
                sd.fit(raw, sensors, ch_types=ch_types)
                print(f"  Calibrated per-channel baselines: {sd.ch_types}")
                nom_scores = sd.score(raw)
                nom_med = float(np.median(nom_scores))
                nom_mad = float(np.median(np.abs(nom_scores - nom_med)) * 1.4826)
                def _lin_thr(k):
                    return nom_med + k * nom_mad if nom_mad > 1e-6 else float(np.percentile(nom_scores, 99.5))
                thr_strong = _lin_thr(9.0)   # matches k_val(default 6) * 1.5 used by consensus
                thr_weak   = _lin_thr(3.6)   # matches k_val(default 6) * 0.6
                thr_p995   = _lin_thr(6.0)
                print(f"  Nominal linear thresholds: weak={thr_weak:.2f} p995={thr_p995:.2f} strong={thr_strong:.2f}")

                stat_path = f"{args.model_dir}/stat.pkl"
                pickle.dump({"sensors":sensors,"detector":sd,"model_type":"StatDetector",
                             "threshold_weak":thr_weak,"threshold_p995":thr_p995,
                             "threshold_strong":thr_strong,
                             "nominal_median":nom_med,"nominal_mad":nom_mad},
                            open(stat_path,"wb"))
                saved["stat"]=1; print("  → saved")

                if mlflow:
                    mlflow.log_param("stat_window", args.stat_window)
                    for s, t in sd.ch_types.items():
                        mlflow.log_param(f"stat_channel_type_{s}", t)
                    mlflow.log_metrics({"threshold_weak":thr_weak,"threshold_p995":thr_p995,
                                        "threshold_strong":thr_strong,"nominal_median":nom_med,
                                        "nominal_mad":nom_mad})
                    mlflow.log_artifact(stat_path)

                if eval_df is not None:
                    sc_e = sd.score(eval_df[sensors].values.astype(np.float32))
                    eval_stat_scores = sc_e
                    sc_e_smooth = pd.Series(sc_e).rolling(5, center=True, min_periods=1).median().values
                    pred_e = (sc_e_smooth > thr_p995).astype(int)
                    eval_predictions["stat"] = pred_e
                    metrics = evaluate_predictions(pred_e, eval_labels)
                    print(f"  Eval: P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                          f"F1={metrics['f1']:.3f} segments={metrics['segments_caught']}/{metrics['n_segments']}")
                    if mlflow:
                        mlflow.log_metrics({f"eval_{k}": v for k,v in metrics.items()})

        # ── Smart Consensus evaluation (parent run) ──
        if eval_df is not None and eval_predictions:
            forecaster_preds = {k: v for k, v in eval_predictions.items() if k != "stat"}
            consensus_pred = smart_consensus_predictions(
                eval_stat_scores, forecaster_preds, stat_k=6.0)
            if len(consensus_pred):
                metrics = evaluate_predictions(consensus_pred, eval_labels)
                print(f"\n[SMART CONSENSUS] P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                      f"F1={metrics['f1']:.3f} segments={metrics['segments_caught']}/{metrics['n_segments']}")
                if mlflow:
                    mlflow.log_metrics({f"consensus_{k}": v for k, v in metrics.items()})

        print(f"\n[DONE] Trained {list(saved.keys())} in {time.time()-t0:.0f}s")
        if mlflow:
            print(f"[MLFLOW] Run logged. View with: mlflow ui --backend-store-uri sqlite:///{args.mlflow_dir}/mlflow.db")


if __name__ == "__main__":
    main()
