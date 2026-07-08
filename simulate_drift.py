
import argparse, os, pickle, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

PRESETS = {
    "cso1": {"kind": "level_shift", "stable": 0.0,  "moderate": 0.035, "significant": 0.15},
    "bfo2": {"kind": "step_widen",  "stable": 1.0,  "moderate": 1.5,   "significant": 2.5},
    "arnd": {"kind": "amp_widen",   "stable": 1.0,  "moderate": 1.35,  "significant": 1.8},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="models", help="Where drift_baseline.pkl lives")
    p.add_argument("--output_dir", default="drift_demo_data",
                   help="Where to write the drifted CSV chunks -- point the app's "
                        "Data Source at this folder")
    p.add_argument("--channel", required=True, choices=["cso1", "bfo2", "arnd"])
    p.add_argument("--severity", required=True, choices=["stable", "moderate", "significant"])
    p.add_argument("--n_chunks", type=int, default=10, help="5 min of data each, so 10 = ~50 min")
    p.add_argument("--seed", type=int, default=None, help="Omit for a fresh random run each time")
    return p.parse_args()


def gen_drifted_chunk(t0, n_rows, bfo2_start, arnd_phase, channel, magnitude, kind):
    time_seconds = np.arange(n_rows)
    bfo2_step_mult = magnitude if (channel == "bfo2" and kind == "step_widen") else 1.0

    arnd = np.sin((time_seconds + arnd_phase) * 0.1) * 5 + np.random.normal(0, 0.5, n_rows)
    if channel == "arnd" and kind == "amp_widen":
        arnd = arnd.mean() + (arnd - arnd.mean()) * magnitude
    bfo2 = bfo2_start + np.cumsum(np.random.normal(0, 0.2 * bfo2_step_mult, n_rows))
    cso1 = np.random.normal(12.0, 0.1, n_rows)
    if channel == "cso1" and kind == "level_shift":
        cso1 = cso1 + magnitude

    next_bfo2_start = float(bfo2[-1])
    next_arnd_phase = float((arnd_phase + n_rows) % (2 * np.pi / 0.1))
    timestamps = [t0 + timedelta(seconds=i) for i in range(n_rows)]
    df = pd.DataFrame({"timestamp": timestamps, "arnd": arnd, "bfo2": bfo2, "cso1": cso1})
    return df, next_bfo2_start, next_arnd_phase


def main():
    args = parse_args()
    baseline_path = os.path.join(args.model_dir, "drift_baseline.pkl")
    if not os.path.exists(baseline_path):
        print(f"ERROR: {baseline_path} not found. Train a model first "
              f"(train.py saves this automatically).")
        sys.exit(1)
    baseline = pickle.load(open(baseline_path, "rb"))
    print(f"Loaded baseline for channels: {list(baseline['samples'].keys())}")

    preset = PRESETS[args.channel]
    magnitude = preset[args.severity]
    kind = preset["kind"]

    os.makedirs(args.output_dir, exist_ok=True)
    for f in glob.glob(os.path.join(args.output_dir, "*.csv")):
        os.remove(f)

    if args.seed is not None:
        np.random.seed(args.seed)

    # Recent, current timestamps -- so if the app is pointed at this folder,
    # its "most recent N files" loading picks these up immediately.
    now = datetime.now()
    bfo2_start, arnd_phase = 50.0, 0.0
    for i in range(args.n_chunks):
        t0 = now - timedelta(seconds=(args.n_chunks - i) * 300)
        df, bfo2_start, arnd_phase = gen_drifted_chunk(
            t0, 300, bfo2_start, arnd_phase, args.channel, magnitude, kind)
        fname = f"telemetry_buffer_{t0.strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(os.path.join(args.output_dir, fname), index=False)

    print(f"Wrote {args.n_chunks} chunks ({args.n_chunks * 300} rows, "
          f"~{args.n_chunks * 5} min) to {args.output_dir}/")

    # Run the actual check right now too, so you know what to expect before
    # even opening the app.
    from drift import check_drift, overall_drift_status
    all_files = sorted(glob.glob(os.path.join(args.output_dir, "*.csv")))
    full_df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    sensors = baseline["sensors"]
    reports = check_drift(baseline, full_df[sensors].values.astype(np.float32), sensors)

    print(f"\n{'='*60}\nExpected Data Health result: {overall_drift_status(reports).upper()}\n{'='*60}")
    for r in reports:
        flag = "  <-- the channel this demo drifted" if r["channel"] == args.channel else ""
        print(f"  {r['channel']:6}: PSI={r['psi']:.4f}  severity={r['severity']:12}{flag}")

    print(f"\nPoint the app's Data Source at '{args.output_dir}' (Local CSV files) "
          f"and open the Data Health tab to see this live.")


if __name__ == "__main__":
    main()
