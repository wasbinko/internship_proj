
import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

# --- Configuration ---
OUTPUT_FOLDER = "live_telemetry_stream"
ROWS_PER_FILE = 300     # Exactly 5 minutes at 1 Hz
INTERVAL_SECONDS = 300  # 5 minutes
ANOMALY_PROBABILITY = 0.25  # 25% chance to inject an anomaly per file


def generate_telemetry_chunk(start_time, num_rows, inject_anomaly=False):
    timestamps = [start_time + timedelta(seconds=i) for i in range(num_rows)]

    # Nominal baseline generation
    amud = np.random.choice([0, 1], size=num_rows, p=[0.95, 0.05])
    time_seconds = np.arange(num_rows)
    arnd = np.sin(time_seconds * 0.1) * 5 + np.random.normal(0, 0.5, num_rows)
    bfo2 = np.cumsum(np.random.normal(0, 0.2, num_rows)) + 50.0
    cso1 = np.random.normal(12.0, 0.1, num_rows)

    anomaly_log = None

    if inject_anomaly:
        anom_duration = np.random.randint(30, 90)
        anom_start = np.random.randint(10, num_rows - anom_duration - 10)
        anom_end = anom_start + anom_duration

        anomaly_type = np.random.choice(["frozen_sensor", "contextual_break", "massive_spike"])

        if anomaly_type == "frozen_sensor":
            freeze_value = cso1[anom_start]
            cso1[anom_start:anom_end] = freeze_value
            anomaly_log = f"Frozen Sensor (cso1 stuck at {freeze_value:.2f} for {anom_duration}s)"

        elif anomaly_type == "contextual_break":
            amud[anom_start:anom_end] = 1
            bfo2[anom_start:anom_end] -= np.linspace(0, 10, anom_duration)
            anomaly_log = f"Contextual Break (amud stuck ON, bfo2 dropping for {anom_duration}s)"

        elif anomaly_type == "massive_spike":
            arnd[anom_start:anom_end] += np.random.normal(20, 5, anom_duration)
            anomaly_log = f"Massive Spike (arnd went erratic for {anom_duration}s)"

    df = pd.DataFrame({
        "timestamp": timestamps,
        "amud": amud,
        "arnd": arnd,
        "bfo2": bfo2,
        "cso1": cso1
    })

    return df, anomaly_log


def parse_args():
    p = argparse.ArgumentParser(description="Mock telemetry generator")
    p.add_argument("--sink", choices=["csv", "kafka", "both"], default="csv",
                   help="Where to send generated chunks. Default: csv (original behaviour).")
    p.add_argument("--kafka_bootstrap", default="localhost:9092",
                   help="Kafka bootstrap servers (only used with --sink kafka/both).")
    p.add_argument("--kafka_topic", default="telemetry.raw",
                   help="Kafka topic to publish chunks to.")
    p.add_argument("--anomaly_probability", type=float, default=ANOMALY_PROBABILITY)
    p.add_argument("--interval", type=int, default=INTERVAL_SECONDS,
                   help="Seconds between chunks (default 300 = 5 min). Use a small "
                        "value like 5 for fast local testing.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    use_csv   = args.sink in ("csv", "both")
    use_kafka = args.sink in ("kafka", "both")

    producer = None
    if use_kafka:
        from scripts.kafka_io import TelemetryProducer
        try:
            producer = TelemetryProducer(bootstrap_servers=args.kafka_bootstrap,
                                         topic=args.kafka_topic)
            print(f"Connected to Kafka broker at {args.kafka_bootstrap}, topic '{args.kafka_topic}'")
        except Exception as e:
            print(f"ERROR: could not connect to Kafka broker at {args.kafka_bootstrap}: {e}")
            print("Check that the broker is running (see docker-compose.yml) and reachable.")
            sys.exit(1)

    print("Starting mock telemetry array with ANOMALY INJECTOR...")
    print(f"Sink: {args.sink}" + (f" (topic '{args.kafka_topic}' @ {args.kafka_bootstrap})" if use_kafka else ""))
    if use_csv:
        print(f"Saving to: ./{OUTPUT_FOLDER}/")
    print(f"Interval: {args.interval / 60:.1f} minutes")
    print(f"Anomaly Probability: {args.anomaly_probability*100:.0f}% per chunk\n")

    try:
        while True:
            current_time = datetime.now()
            start_time = current_time - timedelta(seconds=args.interval)

            is_anomalous = np.random.rand() < args.anomaly_probability
            df, anomaly_info = generate_telemetry_chunk(start_time, ROWS_PER_FILE, inject_anomaly=is_anomalous)

            timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")

            if use_csv:
                filename = f"telemetry_buffer_{timestamp_str}.csv"
                filepath = os.path.join(OUTPUT_FOLDER, filename)
                df.to_csv(filepath, index=False)

            if use_kafka:
                try:
                    producer.send_chunk(df, timestamp_str, sync=True)
                except Exception as e:
                    print(f"   WARNING: failed to send chunk to Kafka: {e}")

            window_start = start_time.strftime('%H:%M:%S')
            window_end = (start_time + timedelta(seconds=ROWS_PER_FILE-1)).strftime('%H:%M:%S')

            dest = []
            if use_csv:   dest.append(f"file: telemetry_buffer_{timestamp_str}.csv")
            if use_kafka: dest.append(f"kafka: {args.kafka_topic}")
            print(f"[{current_time.strftime('%H:%M:%S')}] Saved {ROWS_PER_FILE} rows | "
                  f"Data window: {window_start} -> {window_end} | {' | '.join(dest)}")

            if is_anomalous:
                print(f"!!!!!!!!!!!! ANOMALY INJECTED: {anomaly_info}")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nTelemetry generator stopped by user.")
    finally:
        if producer:
            producer.close()


if __name__ == "__main__":
    main()
