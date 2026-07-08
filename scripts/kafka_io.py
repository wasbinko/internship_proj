from __future__ import annotations
import json
import time
from typing import Iterator, Optional

import pandas as pd

DEFAULT_TOPIC = "telemetry.raw"
DEFAULT_BOOTSTRAP = "localhost:9092"


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation
# ─────────────────────────────────────────────────────────────────────────────

def chunk_to_message(df: pd.DataFrame, chunk_timestamp: str) -> dict:
    """Serialise a chunk DataFrame into the wire message shape."""
    cols = list(df.columns)
    # Convert to plain python types so json.dumps never chokes on numpy/pandas
    # scalar types (np.float32, pd.Timestamp, etc).
    rows = []
    for row in df.itertuples(index=False):
        out = []
        for v in row:
            if isinstance(v, pd.Timestamp):
                out.append(v.isoformat())
            elif hasattr(v, "item"):   # numpy scalar
                out.append(v.item())
            else:
                out.append(v)
        rows.append(out)
    return {"chunk_timestamp": chunk_timestamp, "columns": cols, "rows": rows}


def message_to_chunk(msg: dict) -> pd.DataFrame:
    """Deserialise a wire message back into a chunk DataFrame."""
    df = pd.DataFrame(msg["rows"], columns=msg["columns"])
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Producer
# ─────────────────────────────────────────────────────────────────────────────

class TelemetryProducer:
    def __init__(self, bootstrap_servers: str = DEFAULT_BOOTSTRAP,
                 topic: str = DEFAULT_TOPIC):
        from kafka import KafkaProducer
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8"),
            # Small acks + linger for low-latency single-message sends at
            # this data rate (one chunk every 5 min); no need for batching.
            acks=1,
            linger_ms=10,
        )

    def send_chunk(self, df: pd.DataFrame, chunk_timestamp: str, sync: bool = True):
        msg = chunk_to_message(df, chunk_timestamp)
        future = self.producer.send(self.topic, key=chunk_timestamp, value=msg)
        if sync:
            future.get(timeout=10)   # raises on failure
        return future

    def close(self):
        self.producer.flush()
        self.producer.close()


# ─────────────────────────────────────────────────────────────────────────────
# Consumer
# ─────────────────────────────────────────────────────────────────────────────

class TelemetryConsumer:
    def __init__(self, bootstrap_servers: str = DEFAULT_BOOTSTRAP,
                 topic: str = DEFAULT_TOPIC, group_id: Optional[str] = None,
                 auto_offset_reset: str = "latest",
                 consumer_timeout_ms: int = 5000):
        from kafka import KafkaConsumer
        self.topic = topic
        self._subscribed = False
        self.consumer = KafkaConsumer(
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,                      # None = no offset persistence (app one-shot reads)
            auto_offset_reset=auto_offset_reset,     # "latest" or "earliest"
            enable_auto_commit=(group_id is not None),
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
            consumer_timeout_ms=consumer_timeout_ms,  # stop iterating after this much idle time
        )

    def poll_new_chunks(self, max_records: int = 100) -> list[tuple[str, pd.DataFrame]]:
        if not self._subscribed:
            self.consumer.subscribe([self.topic])
            self._subscribed = True
        results = []
        records = self.consumer.poll(timeout_ms=1000, max_records=max_records)
        for tp, msgs in records.items():
            for m in msgs:
                results.append((m.key, message_to_chunk(m.value)))
        return results

    def read_last_n_chunks(self, n: int) -> list[tuple[str, pd.DataFrame]]:
        if self._subscribed:
            raise RuntimeError(
                "This TelemetryConsumer already called poll_new_chunks() (which "
                "subscribes to the topic). read_last_n_chunks() uses manual "
                "partition assignment and cannot be mixed with subscribe() on "
                "the same instance. Create a separate TelemetryConsumer for "
                "one-shot reads."
            )
        from kafka import TopicPartition
        partitions = self.consumer.partitions_for_topic(self.topic)
        if not partitions:
            return []
        tps = [TopicPartition(self.topic, p) for p in partitions]
        self.consumer.assign(tps)
        end_offsets = self.consumer.end_offsets(tps)

        per_partition = max(1, n // len(tps))
        for tp in tps:
            end = end_offsets[tp]
            start = max(0, end - per_partition)
            self.consumer.seek(tp, start)

        results = []
        for msg in self.consumer:
            results.append((msg.key, message_to_chunk(msg.value)))
            if len(results) >= n:
                break
        results.sort(key=lambda r: r[0])   # chunk_timestamp string sorts chronologically
        return results

    def close(self):
        self.consumer.close()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: assemble consumed chunks into one DataFrame (matches the shape
# that load_telemetry()/load_data() produce from CSV files)
# ─────────────────────────────────────────────────────────────────────────────

def chunks_to_dataframe(chunks: list[tuple[str, pd.DataFrame]]) -> Optional[pd.DataFrame]:
    if not chunks:
        return None
    chunks_sorted = sorted(chunks, key=lambda c: c[0])   # sort by chunk_timestamp key
    dfs = [df for _, df in chunks_sorted]
    out = pd.concat(dfs, ignore_index=True)
    
    if "timestamp" in out.columns and len(out) > 1:
        ts = pd.to_datetime(out["timestamp"])
        backwards = int((ts.diff().dt.total_seconds() < 0).sum())
        if backwards > 0:
            print(f"[kafka_io] WARNING: {backwards} row(s) have an embedded timestamp "
                  f"earlier than the row before them, even after ordering by chunk "
                  f"send-time. This usually means the producer's system clock changed "
                  f"between restarts. Consider clearing old messages "
                  f"(docker compose down -v) before a fresh training run.")
    return out
