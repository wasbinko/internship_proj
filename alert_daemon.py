import os
import sys
import time
import glob
import smtplib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from collections import deque
from email.message import EmailMessage

sys.path.insert(0, "scripts")
from infer import load_all, score_all


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Telemetry watchdog daemon")
    p.add_argument("--source", choices=["csv", "kafka"], default="csv",
                   help="Where to read new chunks from. csv = poll DATA_DIR for new "
                        "files (original behaviour). kafka = consume from a Kafka "
                        "topic using a persistent consumer group (offsets survive "
                        "restarts — no processed_files bookkeeping needed).")
    p.add_argument("--kafka_bootstrap", default=KAFKA_BOOTSTRAP)
    p.add_argument("--kafka_topic", default=KAFKA_TOPIC)
    p.add_argument("--kafka_group", default=KAFKA_GROUP_ID)
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "live_telemetry_stream"
MODEL_DIR = "models"
CHECK_INTERVAL_SECONDS = 30      # how often to poll the folder for new files (csv source only)

# Kafka source settings (only used with --source kafka)
KAFKA_BOOTSTRAP = "localhost:9092"
KAFKA_TOPIC     = "telemetry.raw"
KAFKA_GROUP_ID  = "alert-daemon"   # persistent consumer group — offsets survive restarts

# Email settings
EMAIL_SENDER   = "wasbfifa228@gmail.com"
EMAIL_PASSWORD = "ifut trhq sfjn mnad"     # use an App Password, never a real password
EMAIL_RECEIVER = "mammadovn228@gmail.com"

# Models to run
MODELS_TO_USE = ["stat", "lstm", "patchtst", "xgboost"]

# Detection settings
THRESHOLD_METHOD   = "mad"   # log-space median + k*MAD for forecaster/tree models
THRESHOLD_K        = 4.0     # higher = stricter = fewer false alarms (forecasters)
STAT_K             = 4.0     # base k for StatDetector's linear z-score threshold
SCORE_SMOOTHING_S  = 5       # rolling-median smoothing window (seconds)
MIN_DURATION_S     = 10      # discard flagged runs shorter than this

# Rolling context buffer: how many recent chunks to keep for scoring.
BUFFER_CHUNKS = 10

# Cooldown: once an alert fires, don't fire again for this many seconds even
# if the same ongoing anomaly is still being flagged in subsequent chunks.
ALERT_COOLDOWN_SECONDS = 0   # had this for anyone that might need to use this.


# ─────────────────────────────────────────────────────────────────────────────
# Detection helpers
# ─────────────────────────────────────────────────────────────────────────────

LINEAR_SCORE_MODELS = {"stat"}   # StatDetector outputs z-scores, not log-scale errors


def smooth_scores(sc: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return sc
    return pd.Series(sc).rolling(window, center=True, min_periods=1).median().values.astype(np.float32)


def linear_threshold(sc: np.ndarray, k: float) -> float:
    """StatDetector's score is already a z-score; threshold directly in linear space."""
    med = np.median(sc)
    mad = np.median(np.abs(sc - med)) * 1.4826
    if mad < 1e-6:
        return float(np.percentile(sc, 99.5))
    return float(med + k * mad)


def log_threshold(sc: np.ndarray, method: str, k: float) -> float:

    nonzero = sc[sc > 1e-6]
    if len(nonzero) < max(10, int(0.05 * len(sc))):
        return float(np.percentile(sc, 99.5))

    log_sc = np.log10(nonzero + 1e-9)
    if method == "mad":
        med = np.median(log_sc)
        mad = np.median(np.abs(log_sc - med)) * 1.4826
        if mad < 1e-6:
            return float(np.percentile(sc, 99.5))
        return float(10 ** (med + k * mad))
    if method == "iqr":
        q1, q3 = np.percentile(log_sc, 25), np.percentile(log_sc, 75)
        spread = q3 - q1
        if spread < 1e-6:
            return float(np.percentile(sc, 99.5))
        return float(10 ** (np.median(log_sc) + k * spread))
    return float(np.percentile(sc, min(99.9, k * 10)))


def apply_min_duration(pred: np.ndarray, min_dur: int) -> np.ndarray:
    if min_dur <= 1:
        return pred
    out = pred.copy()
    d = np.diff(np.concatenate([[0], pred.astype(int), [0]]))
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if (e - s) < min_dur:
            out[s:e] = 0
    return out


def regions(pred: np.ndarray):
    d = np.diff(np.concatenate([[0], pred.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def smart_consensus(predictions: dict, scores: dict, stat_thresholds: tuple,
                    min_forecaster_agree: int = 2) -> np.ndarray:

    fc_names = [n for n in ("lstm", "patchtst", "nhits", "xgboost") if n in predictions]
    fc_vote = sum(predictions[n] for n in fc_names) if fc_names else None
    fc_strong_agree = (fc_vote >= min_forecaster_agree).astype(int) if fc_vote is not None else None

    if "stat" not in scores:
        if fc_strong_agree is not None:
            return fc_strong_agree
        return next(iter(predictions.values())).astype(int) if predictions else np.array([], dtype=int)

    strong_thr, weak_thr = stat_thresholds
    stat_sc = scores["stat"]
    stat_strong = (stat_sc > strong_thr).astype(int)
    stat_weak   = (stat_sc > weak_thr).astype(int)
    fc_any_agree = (fc_vote >= 1).astype(int) if fc_vote is not None else np.zeros(len(stat_sc), dtype=int)

    confirmed = np.maximum(stat_strong, stat_weak & fc_any_agree)
    if fc_strong_agree is not None:
        confirmed = np.maximum(confirmed, fc_strong_agree)
    return confirmed


# ─────────────────────────────────────────────────────────────────────────────
# Email
# ─────────────────────────────────────────────────────────────────────────────

def send_anomaly_email(filename: str, model_counts: dict, n_confirmed_regions: int,
                       confirmed_duration_s: int):
    msg = EmailMessage()
    msg['Subject'] = f"🚨 CONFIRMED Telemetry Anomaly ({filename})"
    msg['From'] = EMAIL_SENDER
    msg['To']   = EMAIL_RECEIVER

    body  = f"A CONFIRMED anomaly was detected by the StatDetector + Forecaster consensus.\n"
    body += f"Latest chunk: {filename}\n\n"
    body += f"Confirmed anomaly regions: {n_confirmed_regions}\n"
    body += f"Total confirmed duration: {confirmed_duration_s}s\n\n"
    body += "Per-model flagged timesteps (in current scoring buffer):\n"
    for model_name, count in model_counts.items():
        body += f"  - {model_name.upper()}: {count} timesteps flagged\n"
    body += "\nOpen the Telemetry Dashboard → Deep Dive tab to investigate.\n"
    body += f"\n(Detection: StatDetector strong-signal OR weak-signal+forecaster-backed, "
    body += f"smoothing={SCORE_SMOOTHING_S}s, min_duration={MIN_DURATION_S}s, stat_k={STAT_K})"
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f"   📧 Alert email sent to {EMAIL_RECEIVER}")
    except Exception as e:
        print(f"   ❌ Failed to send email: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print("🛡️  Starting Telemetry Watchdog Daemon (v3 — StatDetector + smart consensus)...")

    bundles = load_all(MODEL_DIR)
    active_bundles = {k: v for k, v in bundles.items() if k in MODELS_TO_USE}
    if not active_bundles:
        print("No trained models found. Exiting.")
        return
    if "stat" not in active_bundles:
        print("⚠️  WARNING: StatDetector ('stat') not found in models/. Frozen-sensor "
              "anomalies will NOT be detected — forecasting models are structurally "
              "blind to them. Run train.py with 'stat' included to fix this.")
    print(f"✅ Models loaded: {list(active_bundles.keys())}")

    kafka_consumer = None
    if args.source == "csv":
        print(f"👀 Watching folder: ./{DATA_DIR}/")
    else:
        from kafka_io import TelemetryConsumer
        try:
            kafka_consumer = TelemetryConsumer(
                bootstrap_servers=args.kafka_bootstrap, topic=args.kafka_topic,
                group_id=args.kafka_group, auto_offset_reset="earliest",
            )
        except Exception as e:
            print(f"ERROR: could not connect to Kafka broker at {args.kafka_bootstrap}: {e}")
            print("Check that the broker is running (see docker-compose.yml) and reachable.")
            return
        print(f"👀 Consuming Kafka topic '{args.kafka_topic}' @ {args.kafka_bootstrap} "
              f"(consumer group '{args.kafka_group}')")

    print(f"⚙️  Detection: smart consensus (StatDetector strong OR weak+forecaster-backed) | "
          f"stat_k={STAT_K} | forecaster {THRESHOLD_METHOD} k={THRESHOLD_K} | "
          f"smoothing={SCORE_SMOOTHING_S}s | min_duration={MIN_DURATION_S}s | "
          f"buffer={BUFFER_CHUNKS} chunks")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CSV-source-only bookkeeping. For the Kafka source, the broker's
    # per-consumer-group committed offset replaces this entirely — a
    # restarted daemon resumes exactly where it left off with no local state.
    processed_files = set(glob.glob(os.path.join(DATA_DIR, "*.csv"))) if args.source == "csv" else None

    chunk_buffer = deque(maxlen=BUFFER_CHUNKS)   # rolling context of recent DataFrames
    last_alert_time = 0.0

    try:
        while True:
            # ── Fetch new chunks, csv or kafka, into a uniform (id, df) list ──
            if args.source == "csv":
                current_files = set(glob.glob(os.path.join(DATA_DIR, "*.csv")))
                new_paths = sorted(current_files - processed_files)
                new_chunks = []
                for file_path in new_paths:
                    try:
                        new_chunks.append((os.path.basename(file_path), pd.read_csv(file_path)))
                    except Exception as e:
                        print(f"   ⚠️ Could not read {os.path.basename(file_path)}: {e}")
                    processed_files.add(file_path)
            else:
                polled = kafka_consumer.poll_new_chunks(max_records=50)
                new_chunks = [(key, df) for key, df in polled]

            for chunk_id, df_new in new_chunks:
                print(f"[{time.strftime('%H:%M:%S')}] New data detected: {chunk_id}. Analyzing...")

                chunk_buffer.append(df_new)

               #also maybe useful, if needed change to 900 or more.
                buffered_df = pd.concat(list(chunk_buffer), ignore_index=True)
                if len(buffered_df) < 300:
                    print(f"   Building context buffer ({len(buffered_df)} rows so far)... skipping scoring.")
                    continue

                skip = {"timestamp", "y", "label", "anomaly"}
                sensors = [c for c in buffered_df.select_dtypes(include=[np.number]).columns
                          if c not in skip]
                data_arr = buffered_df[sensors].values.astype(np.float32)

                # Score the buffered context
                raw_scores = score_all(active_bundles, data_arr, sensors, device=device)
                raw_scores.pop("ensemble", None)
                predictions = {}
                smoothed = {}
                model_counts = {}
                stat_thresholds = None
                for name, sc in raw_scores.items():
                    sc_smooth = smooth_scores(sc, SCORE_SMOOTHING_S)
                    if name in LINEAR_SCORE_MODELS:

                        saved_stat = active_bundles.get(name, {})
                        if "threshold_strong" in saved_stat and "threshold_weak" in saved_stat:
                            thr = saved_stat.get("threshold_p995", saved_stat["threshold_weak"])
                            stat_thresholds = (saved_stat["threshold_strong"], saved_stat["threshold_weak"])
                        else:
                            thr = linear_threshold(sc_smooth, STAT_K)
                            stat_thresholds = (
                                linear_threshold(sc_smooth, STAT_K * 1.5),   # strong
                                linear_threshold(sc_smooth, max(1.0, STAT_K * 0.6)),  # weak
                            )
                            print(f"   ⚠️ {name}: no saved nominal threshold found, using live "
                                  f"median+MAD (retrain with current train.py to fix)")
                    else:
 
                        saved_bundle = active_bundles.get(name, {})
                        saved_thr = saved_bundle.get("threshold_p995")
                        if saved_thr is not None:
                            thr = saved_thr
                        else:
                            thr = log_threshold(sc_smooth, THRESHOLD_METHOD, THRESHOLD_K)
                            print(f"   ⚠️ {name}: no saved nominal threshold found, using live "
                                  f"log-MAD (retrain with current train.py to fix)")
                    pred = (sc_smooth > thr).astype(int)

                    win = active_bundles.get(name, {}).get("window")
                    if win:
                        pred[:min(win, len(pred))] = 0
                    pred = apply_min_duration(pred, MIN_DURATION_S)
                    predictions[name] = pred
                    smoothed[name]    = sc_smooth
                    model_counts[name] = int(pred.sum())

                # Only look at the TAIL of the buffer corresponding to the new chunk —
                # earlier rows have already been evaluated in prior iterations.
                n_new = len(df_new)
                tail_predictions = {k: v[-n_new:] for k, v in predictions.items()}
                tail_smoothed     = {k: v[-n_new:] for k, v in smoothed.items()}

                if stat_thresholds is not None:
                    consensus = smart_consensus(tail_predictions, tail_smoothed, stat_thresholds)
                else:
                    consensus = smart_consensus(tail_predictions, tail_smoothed, (0.0, 0.0))
                consensus = apply_min_duration(consensus, MIN_DURATION_S)
                confirmed = regions(consensus)

                now = time.time()
                cooldown_active = (now - last_alert_time) < ALERT_COOLDOWN_SECONDS

                if confirmed and not cooldown_active:
                    total_dur = sum(e - s for s, e in confirmed)
                    print(f"   🚨 CONFIRMED anomaly: {len(confirmed)} region(s), {total_dur}s total.")
                    send_anomaly_email(chunk_id, model_counts, len(confirmed), total_dur)
                    last_alert_time = now
                elif confirmed and cooldown_active:
                    remaining = int(ALERT_COOLDOWN_SECONDS - (now - last_alert_time))
                    print(f"   🔇 Anomaly still ongoing but within cooldown "
                          f"({remaining}s remaining) — not re-alerting.")
                else:
                    flagged_models = [k for k, v in model_counts.items() if v > 0]
                    if flagged_models:
                        print(f"   ✅ Nominal — {flagged_models} flagged isolated points "
                              f"but consensus rejected them as unconfirmed noise.")
                    else:
                        print("   ✅ Nominal. No anomalies detected.")

            time.sleep(CHECK_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n🛑 Watchdog stopped.")
    finally:
        if kafka_consumer is not None:
            kafka_consumer.close()


if __name__ == "__main__":
    main()
