

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
    """
    Thin wrapper around kafka-python's KafkaProducer for sending chunk
    DataFrames. Used by generate_telemetry.py in --sink kafka / --sink both
    modes.
    """
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
        """Send one chunk. sync=True blocks until the broker acks (safer for
        a low-frequency producer where you want to know immediately if the
        broker is unreachable)."""
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
    """
    Thin wrapper around kafka-python's KafkaConsumer for reading chunk
    DataFrames. Used by both alert_daemon.py (continuous polling with a
    persistent consumer group) and app.py (one-shot "fetch last N chunks").

    IMPORTANT: kafka-python enforces that a single consumer instance uses
    EXACTLY ONE of subscribe() or assign() — never both. This class supports
    both usage patterns, so it deliberately does NOT subscribe in __init__.
    Call poll_new_chunks() for the persistent-subscription path (daemon), or
    read_last_n_chunks() for the manual-assignment path (app one-shot read) —
    calling both on the same instance will raise the same IllegalStateError
    this class exists to avoid.
    """
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
        """
        Non-blocking-ish poll for new messages since the last call (relies on
        the consumer group's committed offset — this is what replaces
        `processed_files`). Returns [(chunk_timestamp, df), ...] in order.
        Used by the daemon's main loop, called once per iteration.

        Subscribes to the topic on first call (lazily, so a TelemetryConsumer
        built for one-shot reads never triggers this path).
        """
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
        """
        One-shot read of the most recent N chunks from the topic, ignoring
        consumer-group offsets entirely (seeks to the end and reads
        backwards). Used by the Streamlit app's Configure & Run tab, which
        wants "give me a fresh snapshot", not a persistent stream position.

        Uses manual partition assignment (assign()), which is why this
        consumer must never also call subscribe() — see class docstring.
        """
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

        # Seek each partition back by (roughly) n messages total, split
        # evenly — fine for the single-partition case this project uses.
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
    """
    Assemble consumed chunks into one DataFrame, ordered by CHUNK send-time
    (the Kafka message key, i.e. when the producer actually sent it) —
    deliberately NOT by re-sorting the concatenated rows on the data's own
    embedded "timestamp" column.

    Why this matters: within a single chunk, the 300 rows are always
    correctly ordered (one atomic call to generate_telemetry_chunk). But if
    the producer was ever stopped and restarted (very common during
    testing/development), the new session's embedded timestamps can overlap
    in wall-clock time with an older session still sitting in the topic —
    sorting the FULL concatenated dataframe by that embedded column then
    interleaves rows from two unrelated recording sessions together.
    Verified: two 25-minute sessions overlapping by 15 minutes produced 1401
    session switches across 3000 sorted rows — not a small gap, a fully
    scrambled dataset. This silently corrupts rolling-window features (used
    heavily by XGBoost) far more than it affects models that don't depend on
    true time-contiguity between consecutive rows.

    Ordering by chunk key instead sidesteps the problem entirely: each
    chunk's internal row order is trustworthy on its own, so only the
    chunk-to-chunk order needs a reliable signal, and the producer's actual
    send order (the message key) is that signal — unlike the data's
    embedded timestamp, it can't retroactively overlap with a past session.
    """
    if not chunks:
        return None
    chunks_sorted = sorted(chunks, key=lambda c: c[0])   # sort by chunk_timestamp key
    dfs = [df for _, df in chunks_sorted]
    out = pd.concat(dfs, ignore_index=True)

    # Defense-in-depth: even with correct chunk ordering, a producer whose
    # SYSTEM CLOCK changed between restarts (rare, but possible) could still
    # produce chunks whose own embedded timestamps aren't in the order their
    # keys suggest. This doesn't fix that case (nothing can, without a
    # trustworthy clock) — it just makes it visible instead of silently
    # scoring on scrambled data again.
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
