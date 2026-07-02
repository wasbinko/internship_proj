
import argparse, os, pickle, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="models")
    p.add_argument("--data_dir", default="live_telemetry_stream")
    p.add_argument("--n_files", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()

    stat_path = os.path.join(args.model_dir, "stat.pkl")
    if not os.path.exists(stat_path):
        print(f"ERROR: {stat_path} not found. Train first."); return
    bundle = pickle.load(open(stat_path, "rb"))
    sd = bundle["detector"]
    sensors = bundle["sensors"]

    print("=" * 70)
    print("STATDETECTOR CALIBRATION (what was learned at training time)")
    print("=" * 70)
    for s in sensors:
        print(f"\n  {s}  ->  classified as: {sd.ch_types[s]}")
        for k, v in sd.stats[s].items():
            print(f"      {k:15} = {v:.6f}")

    print(f"\n  Saved threshold fields on stat.pkl: "
          f"{[k for k in bundle if 'threshold' in k or 'nominal' in k]}")
    for k in ("threshold_weak", "threshold_p995", "threshold_strong",
              "nominal_median", "nominal_mad"):
        if k in bundle:
            print(f"    {k:16} = {bundle[k]:.6f}")
        else:
            print(f"    {k:16} = MISSING (this model was trained before the "
                  f"threshold-calibration fix — retrain to get this)")

    # ── Load real live data ──
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))[-args.n_files:]
    if not files:
        print(f"\nERROR: no CSV files found in {args.data_dir}"); return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"\n{'=' * 70}")
    print(f"LIVE DATA: {len(df):,} rows from {len(files)} files")
    print("=" * 70)

    missing = [s for s in sensors if s not in df.columns]
    if missing:
        print(f"  WARNING: live data is missing expected sensor columns: {missing}")
        print(f"  Live data columns: {list(df.columns)}")
        return

    raw = df[sensors].values.astype(np.float32)

    # Basic sanity check per channel: does the live data's own scale even
    # roughly match what was seen at training time?
    print("\nPer-channel LIVE data stats vs what was likely seen at training:")
    for i, s in enumerate(sensors):
        v = raw[:, i]
        print(f"  {s:6}: live min={v.min():10.4f}  max={v.max():10.4f}  "
              f"mean={v.mean():10.4f}  std={v.std():10.4f}")

    # ── Score and break down per channel ──
    per_ch = sd.score_per_channel(raw)
    total = sd.score(raw)

    print(f"\n{'=' * 70}")
    print("PER-CHANNEL SCORE BREAKDOWN ON LIVE DATA")
    print("=" * 70)
    for i, s in enumerate(sensors):
        col = per_ch[:, i]
        pct_high = (col > 10).mean() * 100
        print(f"  {s:6} ({sd.ch_types[s]:11}): median={np.median(col):8.3f}  "
              f"max={col.max():8.3f}  %rows>10={pct_high:5.1f}%")

    print(f"\n  Aggregate score (max across channels): "
          f"median={np.median(total):.3f}  max={total.max():.3f}")

    # Identify the worst offender
    worst_ch_idx = np.argmax(per_ch.max(axis=0))
    worst_ch = sensors[worst_ch_idx]
    print(f"\n  >>> Channel driving the highest scores: '{worst_ch}' "
          f"(classified: {sd.ch_types[worst_ch]})")

    # Show the actual arithmetic for the worst channel at its worst moment
    worst_row = np.argmax(per_ch[:, worst_ch_idx])
    print(f"  >>> Worst single row: index {worst_row}, "
          f"raw value = {raw[worst_row, worst_ch_idx]:.4f}")
    st = sd.stats[worst_ch]
    print(f"  >>> Calibrated baseline for '{worst_ch}': {st}")

    print(f"\n{'=' * 70}")
    print("If one channel's %rows>10 is much higher than the others, that's")
    print("the channel to focus on. Compare its 'live' stats above to its")
    print("calibrated baseline — a large mismatch in scale or classification")
    print("mismatch (e.g. expected 'oscillatory' but classified 'drift')")
    print("points at the exact mechanism.")
    print("=" * 70)


if __name__ == "__main__":
    main()
