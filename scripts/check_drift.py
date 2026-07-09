
import argparse, os, pickle, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from drift import check_drift, overall_drift_status, PSI_STABLE_THRESHOLD, PSI_MODERATE_THRESHOLD


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="models")
    p.add_argument("--source", choices=["csv", "kafka"], default="csv",
                   help="Where to read recent telemetry from. csv = read files "
                        "from --data_dir (original behaviour). kafka = read the "
                        "most recent chunks directly from a topic — use this if "
                        "you're running generate_telemetry.py with --sink kafka "
                        "and no longer have CSV files landing in a folder.")
    p.add_argument("--data_dir", default="live_telemetry_stream",
                   help="Only used with --source csv.")
    p.add_argument("--kafka_bootstrap", default="localhost:9092")
    p.add_argument("--kafka_topic", default="telemetry.raw")
    p.add_argument("--n_files", type=int, default=20,
                   help="How many of the most recent chunks to check drift against "
                        "(CSV files or Kafka messages, depending on --source).")
    return p.parse_args()


def load_recent_data(args) -> pd.DataFrame | None:
    """Fetch the most recent telemetry, from whichever source was requested."""
    if args.source == "csv":
        files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))[-args.n_files:]
        if not files:
            return None
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    else:
        from kafka_io import TelemetryConsumer, chunks_to_dataframe
        consumer = TelemetryConsumer(
            bootstrap_servers=args.kafka_bootstrap, topic=args.kafka_topic,
            group_id=None,   # one-shot read, same as the app's Kafka mode
        )
        chunks = consumer.read_last_n_chunks(args.n_files)
        consumer.close()
        return chunks_to_dataframe(chunks)


def main():
    args = parse_args()

    baseline_path = os.path.join(args.model_dir, "drift_baseline.pkl")
    if not os.path.exists(baseline_path):
        print(f"ERROR: {baseline_path} not found.")
        print("This file is created automatically by train.py — retrain with the "
              "current version of train.py to generate it.")
        sys.exit(3)
    baseline = pickle.load(open(baseline_path, "rb"))
    sensors = baseline["sensors"]

    df = load_recent_data(args)
    if df is None or len(df) == 0:
        if args.source == "csv":
            print(f"ERROR: no CSV files found in {args.data_dir}")
        else:
            print(f"ERROR: no messages found on Kafka topic '{args.kafka_topic}' "
                  f"@ {args.kafka_bootstrap}. Is the producer running "
                  f"(generate_telemetry.py --sink kafka)?")
        sys.exit(3)

    missing = [s for s in sensors if s not in df.columns]
    if missing:
        print(f"ERROR: live data is missing expected sensor columns: {missing}")
        sys.exit(3)

    raw = df[sensors].values.astype(np.float32)
    reports = check_drift(baseline, raw, sensors)
    status = overall_drift_status(reports)

    print("=" * 66)
    source_desc = f"{args.n_files} recent files" if args.source == "csv" else \
                  f"{args.n_files} recent Kafka chunks (topic '{args.kafka_topic}')"
    print(f"DRIFT CHECK — {len(df):,} rows from {source_desc}")
    print("=" * 66)
    print(f"\n{'Channel':10} {'PSI':>8}  {'Status':12} {'Baseline mean/std':>20} "
          f"{'Current mean/std':>20}")
    for r in reports:
        marker = {"stable": "  ", "moderate": "! ", "significant": "!!"}[r["severity"]]
        print(f"{marker}{r['channel']:8} {r['psi']:8.4f}  {r['severity']:12} "
              f"{r['baseline_mean']:9.3f} / {r['baseline_std']:7.3f}   "
              f"{r['current_mean']:9.3f} / {r['current_std']:7.3f}")

    print(f"\nOverall status: {status.upper()}")
    if status == "stable":
        print("No meaningful drift detected. Current models should still be reliable.")
    elif status == "moderate":
        print("Some channels have drifted moderately. Worth keeping an eye on — not")
        print("urgent, but consider retraining if this persists over several checks.")
    else:
        print("At least one channel has drifted significantly. The trained models'")
        print("baseline no longer reflects current normal behaviour — detection")
        print("quality may be degraded. Consider retraining:")
        if args.source == "csv":
            print(f"    python scripts/train.py --data_dir {args.data_dir} "
                  f"--model_dir {args.model_dir} --trim 0.25")
        else:
            print(f"    python scripts/train.py --source kafka "
                  f"--kafka_bootstrap {args.kafka_bootstrap} --kafka_topic {args.kafka_topic} "
                  f"--model_dir {args.model_dir} --trim 0.25")

    print(f"\n(Thresholds: stable < {PSI_STABLE_THRESHOLD}, "
          f"moderate < {PSI_MODERATE_THRESHOLD}, significant >= {PSI_MODERATE_THRESHOLD})")

    sys.exit({"stable": 0, "moderate": 1, "significant": 2}[status])


if __name__ == "__main__":
    main()
