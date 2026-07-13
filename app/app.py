

from __future__ import annotations
import os, sys, glob, time, warnings
from datetime import datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
sys.path.insert(0, "scripts")
from infer import load_all, score_all, derive_threshold

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telemetry Anomaly Platform",
    page_icon=":material/satellite_alt:", layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_COLORS = {
    "stat":     "#C2410C",
    "lstm":     "#1D4ED8",
    "patchtst": "#047857",
    "nhits":    "#6D28D9",
    "xgboost":  "#B45309",
    "iforest":  "#B91C1C",
    "ensemble": "#7E22CE",
    "consensus":"#DC2626",
}
MODEL_LABELS = {
    "stat":     "StatDetector",
    "lstm":     "LSTM Forecaster",
    "patchtst": "PatchTST",
    "nhits":    "NHITS (Darts)",
    "xgboost":  "XGBoost",
    "iforest":  "Isolation Forest",
    "ensemble": "Ensemble",
    "consensus":"Confirmed (StatDetector + Forecaster)",
}

ANOMALY_TYPE_INFO = {
    "frozen_sensor":    ("Frozen Sensor",   "Caught mostly by StatDetector forecasters usually miss it entirely."),
    "contextual_break": ("Contextual Break", "arnd goes quiet while bfo2 keeps drifting. Caught by forecaster cross-channel error."),
    "massive_spike":    ("Massive Spike",    "arnd shoots up 20. Caught by all models usually."),
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _model_dir_signature(model_dir: str) -> str:
    if not os.path.isdir(model_dir):
        return "missing"
    parts = []
    for name in sorted(os.listdir(model_dir)):
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            st_ = os.stat(p)
            parts.append(f"{name}:{st_.st_size}:{int(st_.st_mtime)}")
    return "|".join(parts)


@st.cache_resource
def load_models_cached(model_dir: str, _signature: str):
    return load_all(model_dir)


@st.cache_data(show_spinner=False)
def load_data(data_dir: str, n_files: int) -> pd.DataFrame | None:
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        return None
    dfs = [pd.read_csv(f) for f in files[-n_files:]]
    df = pd.concat(dfs, ignore_index=True)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_data_kafka(bootstrap_servers: str, topic: str, n_chunks: int) -> tuple[pd.DataFrame | None, str]:
    try:
        from kafka_io import TelemetryConsumer, chunks_to_dataframe
    except ImportError:
        return None, "kafka-python is not installed. Run: pip install kafka-python"

    try:
        consumer = TelemetryConsumer(
            bootstrap_servers=bootstrap_servers, topic=topic,
            group_id=None, 
        )
        chunks = consumer.read_last_n_chunks(n_chunks)
        consumer.close()
    except Exception as e:
        return None, f"Could not read from Kafka ({bootstrap_servers}, topic '{topic}'): {e}"

    df = chunks_to_dataframe(chunks)
    if df is None:
        return None, f"No messages found on topic '{topic}'. Is the generator running?"
    return df, ""


def get_latest_data_timestamp(source_mode: str, data_dir: str | None = None,
                              kafka_bootstrap: str | None = None,
                              kafka_topic: str | None = None) -> "datetime | None":
    if source_mode == "Local CSV files":
        if not data_dir:
            return None
        files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        if not files:
            return None
        fname = os.path.basename(files[-1])
        try:
            ts_str = fname.replace("telemetry_buffer_", "").replace(".csv", "")
            return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
        except ValueError:
            return None
    else:
        if not kafka_bootstrap or not kafka_topic:
            return None
        try:
            from kafka_io import TelemetryConsumer, chunks_to_dataframe
            consumer = TelemetryConsumer(bootstrap_servers=kafka_bootstrap, topic=kafka_topic, group_id=None)
            chunks = consumer.read_last_n_chunks(1)
            consumer.close()
            df = chunks_to_dataframe(chunks)
            if df is None or len(df) == 0 or "timestamp" not in df.columns:
                return None
            return pd.to_datetime(df["timestamp"]).max().to_pydatetime()
        except Exception:
            return None


def detect_sensors(df: pd.DataFrame, bundles: dict) -> list[str]:
    for b in bundles.values():
        if "sensors" in b:
            return [c for c in b["sensors"] if c in df.columns]
    skip = {"timestamp","y","label","anomaly"}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip]


def get_threshold(scores: np.ndarray, method: str, k: float) -> float:
    return derive_threshold(scores, method=method, k=k)


def apply_thr(scores: np.ndarray, thr: float) -> np.ndarray:
    return (scores > thr).astype(int)


def _regions(pred: np.ndarray, merge_gap: int = 5, max_regions: int = 200):
    d = np.diff(np.concatenate([[0], pred.astype(int), [0]]))
    starts = np.where(d == 1)[0]
    ends   = np.where(d == -1)[0]
    if len(starts) == 0:
        return []
    # Merge regions separated by a small gap
    merged = [[starts[0], ends[0]]]
    for s, e in zip(starts[1:], ends[1:]):
        if s - merged[-1][1] <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    # Cap: keep the longest regions if still too many
    if len(merged) > max_regions:
        merged.sort(key=lambda r: r[1] - r[0], reverse=True)
        merged = merged[:max_regions]
        merged.sort(key=lambda r: r[0])
    return merged


def build_event_options(pred: np.ndarray, sc: np.ndarray, x, min_row: int = 0,
                        min_duration: int = 2, max_events: int = 15) -> list[dict]:
    candidates = []
    for s, e in _regions(pred):
        if (e - s) < min_duration:
            continue
        peak_offset = int(np.argmax(sc[s:e])) if e > s else 0
        peak_idx = s + peak_offset
        if peak_idx < min_row:
            continue
        candidates.append((s, e, peak_idx, float(sc[peak_idx])))


    if not candidates:
        for s, e in _regions(pred):
            peak_offset = int(np.argmax(sc[s:e])) if e > s else 0
            peak_idx = s + peak_offset
            if peak_idx < min_row:
                continue
            candidates.append((s, e, peak_idx, float(sc[peak_idx])))

    if not candidates:
        return []

    if len(candidates) > max_events:
        candidates.sort(key=lambda c: c[3], reverse=True)
        candidates = candidates[:max_events]
    candidates.sort(key=lambda c: c[0])   # back to chronological order for display

    x_arr = np.asarray(x)
    is_datetime = np.issubdtype(x_arr.dtype, np.datetime64)

    events = []
    for i, (s, e, peak_idx, peak_score) in enumerate(candidates, start=1):
        t_start, t_end = x[s], x[min(e, len(x)-1)]
        if is_datetime:
            time_label = f"{pd.Timestamp(t_start):%H:%M:%S} – {pd.Timestamp(t_end):%H:%M:%S}"
        else:
            time_label = f"row {s} – {e}"
        events.append({
            "label": f"Event {i}  ·  {time_label}  ·  {e - s}s",
            "peak_idx": peak_idx,
            "start": s, "end": e,
        })
    return events


def render_event_navigator(events: list[dict], key: str) -> dict:
    state_key = f"event_nav_{key}"
    idx = st.session_state.get(state_key, 0)
    idx = max(0, min(idx, len(events) - 1))
    st.session_state[state_key] = idx

    nav_prev, nav_label, nav_next = st.columns([1, 3, 1])
    with nav_prev:
        if st.button("◀ Previous", key=f"{state_key}_prev", disabled=(idx == 0)):
            st.session_state[state_key] = max(0, idx - 1)
            st.rerun()
    with nav_next:
        if st.button("Next ▶", key=f"{state_key}_next", disabled=(idx == len(events) - 1)):
            st.session_state[state_key] = min(len(events) - 1, idx + 1)
            st.rerun()
    with nav_label:
        st.markdown(
            f"<div style='text-align:center; padding-top:0.45rem; color:#4A5A72'>"
            f"Event <b>{idx + 1}</b> of <b>{len(events)}</b></div>",
            unsafe_allow_html=True,
        )

    chosen_event = events[idx]
    st.caption(chosen_event["label"])
    return chosen_event


def shade(fig, pred: np.ndarray, color: str, row=None, col=None, x=None):
    
    border_color = color
    if color.startswith("rgba("):
        parts = color[color.index("(")+1:color.index(")")].split(",")
        border_color = f"rgba({parts[0]},{parts[1]},{parts[2]},0.9)"
    kw = dict(fillcolor=color, line=dict(color=border_color, width=1), layer="above")
    if row is not None:
        kw["row"] = row; kw["col"] = col
    for s, e in _regions(pred):
        if x is not None:
            x0 = x[min(s, len(x)-1)]
            x1 = x[min(e, len(x)-1)]
            if "datetime" in str(type(x0)):
                x0 = pd.Timestamp(x0).isoformat()
                x1 = pd.Timestamp(x1).isoformat()
        else:
            x0, x1 = int(s), int(e)
        fig.add_vrect(x0=x0, x1=x1, **kw)


def hex_to_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"


# ─────────────────────────────────────────────────────────────────────────────
# Session state bootstrap
# ─────────────────────────────────────────────────────────────────────────────

if "results" not in st.session_state:
    st.session_state["results"] = None      # None = not yet run
if "run_config" not in st.session_state:
    st.session_state["run_config"] = {}


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_cfg, tab_results, tab_deep, tab_compare, tab_health = st.tabs([
    "Configure & Run",
    "Results",
    "Deep Dive",
    "Model Comparison",
    "Data Health",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 - Configure & Run
# ═══════════════════════════════════════════════════════════════════════════════
with tab_cfg:
    st.title("Telemetry Anomaly Intelligence Platform")
    st.caption("Configure and click **Run Analysis**.")
    st.divider()

    # ── AUTO-RUN ──
    st.subheader("Automatic Analysis")
    
    if not AUTOREFRESH_AVAILABLE:
        st.warning("This needs one more package: run `pip install streamlit-autorefresh` "
                  "and restart the app to enable it.")
        auto_run_enabled = False
        auto_run_minutes = 30
    else:
        ac1, ac2, ac3 = st.columns([1, 1, 2])
        with ac1:
            auto_run_enabled = st.checkbox("Enable auto-run", value=False, key="auto_run_enabled")
        with ac2:
            auto_run_minutes = st.number_input(
                "Every N minutes", min_value=5, max_value=240, value=30, step=5,
                key="auto_run_minutes",
            )

        n_files_for_interval = max(1, int(auto_run_minutes) // 5)

        with ac3:
            if auto_run_enabled:
                file_note = f"loading the {n_files_for_interval} most recent file(s)/chunk(s)"
                st.info(f"Auto-run is ON, synced to incoming data - {file_note}. "
                       f"Status detail appears above the Run button below.")

        if auto_run_enabled:
            st_autorefresh(interval=5_000, key="auto_run_ticker")
    n_files_for_interval = max(1, int(auto_run_minutes) // 5)

    st.divider()

    # ── Row 1: Paths ──
    col_data, col_model = st.columns(2)
    with col_data:
        st.subheader("Data Source")
        source_mode = st.radio(
            "Source", ["Kafka topic", "Local CSV files"], horizontal=True,
            help="Kafka: consume the most recent chunks directly from a topic "
                 "(--sink kafka/both). Requires a running broker. Local files: "
                 "read from a folder written by generate_telemetry.py (--sink csv/both).",
        )
        if source_mode == "Local CSV files":
            data_dir = st.text_input("Telemetry stream folder", value="live_telemetry_stream",
                                      help="Folder where generate_telemetry.py writes CSV files.")
            if auto_run_enabled:
                n_files = n_files_for_interval
                st.slider("Files to load (most recent)", 1, 50, n_files, disabled=True,
                         help="Locked to the auto-run interval above - each file covers "
                              "5 minutes, so this always matches the schedule.")
            else:
                n_files  = st.slider("Files to load (most recent)", 1, 50, 10,
                                      help="Each file = 300 rows / 5-minute chunk.")
            available = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
            if available:
                st.success(f"{len(available)} files found in `{data_dir}`")
                st.caption(f"Oldest: {os.path.basename(available[0])}  →  "
                           f"Latest: {os.path.basename(available[-1])}")
            else:
                st.warning(f"No CSV files in `{data_dir}`. Run `python generate_telemetry.py` first.")
            kafka_bootstrap = kafka_topic = None
        else:
            kafka_bootstrap = st.text_input("Kafka bootstrap servers", value="localhost:9092")
            kafka_topic     = st.text_input("Kafka topic", value="telemetry.raw")
            if auto_run_enabled:
                n_files = n_files_for_interval
                st.slider("Chunks to load (most recent)", 1, 50, n_files, disabled=True,
                         help="Locked to the auto-run interval above - each chunk covers "
                              "5 minutes, so this always matches the schedule.")
            else:
                n_files = st.slider("Chunks to load (most recent)", 1, 50, 10,
                                     help="Each Kafka message = one 300-row chunk, same "
                                          "type as a CSV file.")
            data_dir = None
            st.caption("Fetched fresh from the topic on each run.")

    with col_model:
        st.subheader("Model Directory")
        model_dir = st.text_input("Model directory", value="models",
                                   help="Where train.py saved the model files.")
        available_models = {}
        if os.path.isdir(model_dir):
            if os.path.exists(os.path.join(model_dir, "stat.pkl")):          available_models["stat"] = True
            if os.path.exists(os.path.join(model_dir, "lstm.pt")):           available_models["lstm"] = True
            if os.path.exists(os.path.join(model_dir, "patchtst.pt")):       available_models["patchtst"] = True
            if os.path.exists(os.path.join(model_dir, "nhits.pt")):          available_models["nhits"] = True
            if os.path.exists(os.path.join(model_dir, "xgboost.pkl")):       available_models["xgboost"] = True
            if os.path.exists(os.path.join(model_dir, "iforest.pkl")):       available_models["iforest"] = True
        if available_models:
            st.success(f"{len(available_models)} trained model(s) found: {', '.join(available_models)}")
            if "stat" not in available_models:
                st.warning("StatDetector not trained - this is the primary detector for "
                          "frozen-sensor anomalies. Run `train.py` with `stat` included.")
        else:
            st.warning(f"No trained models in `{model_dir}`. Run `python scripts/train.py` first.")

        if st.button("Force-reload models from disk",
                     help="Retrained recently and results still look stale, you should click this."):
            st.cache_resource.clear()
            st.success("Model cache cleared - next Run will load fresh from disk.")

    st.divider()

    # ── Row 2: Model Selection ──
    st.subheader("Select Models to Run")
    

    mcols = st.columns(6)
    model_names = ["stat", "lstm", "patchtst", "nhits", "xgboost", "iforest"]
    selected = {}
    for i, name in enumerate(model_names):
        with mcols[i]:
            trained = name in available_models
            label   = MODEL_LABELS[name]
            color   = MODEL_COLORS[name]
            # Styled card
            st.markdown(
                f"""<div style="border:2px solid {'#CCCCCC' if not trained else color};
                border-radius:10px; padding:12px; margin-bottom:4px;
                background:{'#F5F5F5' if not trained else '#FAFAFA'};
                opacity:{'0.55' if not trained else '1'}">
                <b style="color:{color}">{label}</b>
                </div>""",
                unsafe_allow_html=True,
            )
            selected[name] = st.checkbox(
                f"Use {label}", value=trained, disabled=not trained, key=f"sel_{name}"
            )

    enabled = [n for n, v in selected.items() if v]

    st.divider()


    # ── Row 3: Detection Settings ──
    st.subheader("Detection Settings")
    dcol1, dcol2, dcol3, dcol4 = st.columns(4)

    with dcol1:
        agg_mode = st.selectbox(
            "Channel aggregation",
            ["max", "mean", "top2"],
            help="How per-channel errors are combined into a single score.\n"
                 "max = most sensitive to a single broken channel.\n"
                 "mean = averaged across all channels.",
        )
    with dcol2:
        thr_method = st.selectbox(
            "Threshold method",
            ["mad", "iqr", "pct"],
            help=(
                "mad: median + k·MAD  - most robust, recommended\n"
                "iqr: median + k·IQR  - slightly less robust\n"
                "pct: top (100-k)% percentile of scores"
            ),
        )
    with dcol3:
        k_val = st.slider(
            "Threshold k", 1.0, 15.0, 4.0, 0.5,
            help=(
                "Higher k = stricter = fewer but higher-confidence detections.\n"
                "Lower k = more sensitive = more detections including weaker ones.\n"
                "Default is 4, based on validated real-world testing on this project's "
                "live data. Raise toward 6–8 if you're seeing too many false alarms."
            ),
        )
    with dcol4:
        min_duration = st.slider(
            "Min anomaly duration (s)", 1, 60, 10, 1,
            help=(
                "Discard any flagged run shorter than this many seconds.\n"
                "Your generator injects anomalies of 30–90s, so 10–20s is safe.\n"
                "This alone eliminates most isolated-second false alarms."
            ),
        )

    dcol5, dcol6 = st.columns(2)
    with dcol5:
        include_ensemble = st.checkbox(
            "Ensemble score",
            value=len(enabled) > 1,
            help="Weighted log-mean of all selected models.",
            disabled=len(enabled) < 2,
        )
    with dcol6:
        score_smoothing = st.slider(
            "Score smoothing (s)", 1, 30, 5, 1,
            help=(
                "Rolling median smoothing applied to raw scores before thresholding.\n"
                "Suppresses single-second noise spikes in the score without affecting "
                "sustained anomaly regions. 5–10s is a good default."
            ),
        )

    use_smart_consensus = False
    n_forecasters_enabled = len([n for n in ("lstm","patchtst","xgboost") if n in enabled])
    if "stat" in enabled or n_forecasters_enabled >= 2:
        st.markdown("**Smart Consensus** (recommended - combines complementary detector strengths)")
        ccol1, ccol2 = st.columns([1, 2])
        with ccol1:
            use_smart_consensus = st.checkbox(
                "Enable consensus detection", value=True,
                help="Adds a 'Confirmed' detector: fires when StatDetector is "
                     "strongly elevated alone, OR weakly elevated and backed by "
                     "at least one forecaster, OR when at least 2 forecasters "
                     "agree regardless of StatDetector.",
            )
        with ccol2:
            if use_smart_consensus:
                fc_present = [n for n in ("lstm","patchtst","xgboost") if n in enabled]
                if "stat" in enabled and len(fc_present) >= 2:
                    st.caption(f"Full coverage: StatDetector + {len(fc_present)} forecasters.")
                elif "stat" in enabled:
                    st.caption("Select 2+ forecasters for full coverage.")
                else:
                    st.caption(f"No StatDetector - frozen-sensor detection unavailable.")

    if len(enabled) >= 2:
        st.markdown("**Ensemble weights** (drag to adjust model influence)")
        wcols = st.columns(len(enabled))
        weights = {}
        for i, name in enumerate(enabled):
            with wcols[i]:
                weights[name] = st.slider(
                    MODEL_LABELS[name], 0.1, 3.0, 1.0, 0.1,
                    key=f"w_{name}",
                )
    else:
        weights = {n: 1.0 for n in enabled}

    st.divider()

    # ── RUN BUTTON ──
    if source_mode == "Local CSV files":
        data_ready = bool(available)
        data_warning = "Select at least one trained model and ensure data files exist."
    else:
        data_ready = True   # can't cheaply check Kafka connectivity without blocking the UI;
                             # a bad connection surfaces as a clear error when Run is clicked
        data_warning = "Select at least one trained model to enable Run."
    run_disabled = not enabled or not data_ready
    if run_disabled:
        st.warning(data_warning)

    run_clicked = st.button(
        "Run Analysis",
        type="primary",
        disabled=run_disabled,
        width='stretch',
    )
    auto_run_triggered = False
    if AUTOREFRESH_AVAILABLE and auto_run_enabled and not run_disabled:
        latest_data_ts = get_latest_data_timestamp(
            source_mode,
            data_dir=data_dir if source_mode == "Local CSV files" else None,
            kafka_bootstrap=kafka_bootstrap if source_mode != "Local CSV files" else None,
            kafka_topic=kafka_topic if source_mode != "Local CSV files" else None,
        )
        last_seen_data_ts = st.session_state.get("last_auto_run_data_ts")
        file_note = f"loading the {max(1, int(auto_run_minutes)//5)} most recent file(s)/chunk(s)"

        if latest_data_ts is None:
            st.info(f"Auto-run is ON, waiting for data to become available - {file_note}.")
        elif last_seen_data_ts is None:
            auto_run_triggered = True
            st.session_state["last_auto_run_data_ts"] = latest_data_ts
            st.session_state["last_auto_run_ts"] = time.time()
        else:
            data_elapsed = (latest_data_ts - last_seen_data_ts).total_seconds()
            if data_elapsed >= auto_run_minutes * 60:
                auto_run_triggered = True
                st.session_state["last_auto_run_data_ts"] = latest_data_ts
                st.session_state["last_auto_run_ts"] = time.time()
            else:
                have_min = max(0, data_elapsed / 60)
                st.info(f"Auto-run is ON, synced to incoming data - last run used data through "
                       f"{last_seen_data_ts:%H:%M:%S}. Next run once {auto_run_minutes} minutes "
                       f"of new data has arrived, {file_note}.")

    if run_clicked or auto_run_triggered:
        if auto_run_triggered:
            st.caption("Running automatically, synced to newly arrived data.")
        with st.status("Running analysis...", expanded=True) as status:
            # Load data
            st.write("Loading telemetry data...")
            fetch_n_files = n_files + 1
            if source_mode == "Local CSV files":
                df = load_data(data_dir, fetch_n_files)
                if df is None:
                    st.error("No data found."); st.stop()
                st.write(f"   → {len(df):,} rows from {n_files} files "
                         f"(+1 extra chunk fetched for lookback context)")
            else:
                df, kafka_err = load_data_kafka(kafka_bootstrap, kafka_topic, fetch_n_files)
                if df is None:
                    st.error(kafka_err); st.stop()
                st.write(f"   → {len(df):,} rows from {n_files} Kafka chunks "
                         f"(topic '{kafka_topic}', +1 extra chunk fetched for lookback context)")

            # Load models
            st.write("Loading model bundles...")
            bundles = load_models_cached(model_dir, _model_dir_signature(model_dir))
            active_bundles = {k: v for k, v in bundles.items() if k in enabled}
            st.write(f"   → {list(active_bundles.keys())}")

            sensors = detect_sensors(df, active_bundles)
            data_arr = df[sensors].values.astype(np.float32)

            # Apply aggregation override
            for name in active_bundles:
                active_bundles[name] = dict(active_bundles[name])
                active_bundles[name]["agg"] = agg_mode

            # Score
            st.write("Scoring with selected models...")
            t0 = time.time()
            all_scores = score_all(
                active_bundles, data_arr, sensors,
                weights=weights if include_ensemble else None,
            )
            if not include_ensemble and "ensemble" in all_scores:
                del all_scores["ensemble"]
            elapsed = time.time() - t0
            st.write(f"   → Done in {elapsed:.2f}s")

            ROWS_PER_CHUNK = 300
            trim_rows = 0
            if len(df) > n_files * ROWS_PER_CHUNK:
                trim_rows = len(df) - n_files * ROWS_PER_CHUNK
                df = df.iloc[trim_rows:].reset_index(drop=True)
                data_arr = data_arr[trim_rows:]
                all_scores = {k: v[trim_rows:] for k, v in all_scores.items()}

            LINEAR_SCORE_MODELS = {"stat"}

            def _smooth(sc: np.ndarray, w: int) -> np.ndarray:
                if w <= 1:
                    return sc
                return pd.Series(sc).rolling(w, center=True, min_periods=1).median().values.astype(np.float32)

            def _linear_threshold(sc: np.ndarray, k: float) -> float:
                med = np.median(sc)
                mad = np.median(np.abs(sc - med)) * 1.4826
                if mad < 1e-6:
                    return float(np.percentile(sc, 99.5))
                return float(med + k * mad)

            def _log_threshold(sc: np.ndarray, method: str, k: float) -> float:
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
                    if q3 - q1 < 1e-6:
                        return float(np.percentile(sc, 99.5))
                    return float(10 ** (np.median(log_sc) + k * (q3 - q1)))
                return float(np.percentile(sc, min(99.9, k * 10)))

            def _apply_min_duration(pred: np.ndarray, min_dur: int) -> np.ndarray:
                if min_dur <= 1:
                    return pred
                out = pred.copy()
                d = np.diff(np.concatenate([[0], pred.astype(int), [0]]))
                for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
                    if (e - s) < min_dur:
                        out[s:e] = 0
                return out

            def _warmup_length(model_name: str) -> int:
                b = active_bundles.get(model_name, {})
                if "window" in b:
                    return int(b["window"])
                det = b.get("detector")
                if det is not None and hasattr(det, "window"):
                    return int(det.window)
                return 0

            thresholds = {}
            predictions = {}
            smoothed_scores = {}
            threshold_basis = {}   # tracks whether each model used a saved or live threshold
            for name, sc in all_scores.items():
                sc_smooth = _smooth(sc, score_smoothing)
                if name in LINEAR_SCORE_MODELS:
                    saved_bundle = active_bundles.get(name, {})
                    saved_thr = None
                    if "nominal_median" in saved_bundle and "nominal_mad" in saved_bundle:
                        nm, nmad = saved_bundle["nominal_median"], saved_bundle["nominal_mad"]
                        saved_thr = nm + k_val * nmad if nmad > 1e-6 else saved_bundle.get("threshold_p995")
                    if saved_thr is not None:
                        thr = saved_thr
                        threshold_basis[name] = "saved (nominal-calibrated, k-scaled)"
                    else:
                        thr = _linear_threshold(sc_smooth, k_val)
                        threshold_basis[name] = "live (linear, no saved threshold - retrain to fix)"
                else:
                    saved_bundle = active_bundles.get(name, {})
                    saved_thr = None
                    if "threshold_p995" in saved_bundle:
                        if k_val <= 4.0:
                            saved_thr = saved_bundle.get("threshold_p99")
                        elif k_val <= 8.0:
                            saved_thr = saved_bundle.get("threshold_p995")
                        else:
                            saved_thr = saved_bundle.get("threshold_p999")
                    if saved_thr is not None:
                        thr = saved_thr
                        threshold_basis[name] = "saved (nominal-calibrated)"
                    else:
                        thr = _log_threshold(sc_smooth, thr_method, k_val)
                        threshold_basis[name] = "live (log-MAD, no saved threshold - retrain to fix)"
                pred = (sc_smooth > thr).astype(int)

                win = _warmup_length(name)
                if win:
                    pred[:min(win, len(pred))] = 0

                pred = _apply_min_duration(pred, min_duration)
                thresholds[name]      = thr
                predictions[name]     = pred
                smoothed_scores[name] = sc_smooth


            forecaster_names = [n for n in ("lstm", "patchtst", "nhits", "xgboost") if n in predictions]
            min_forecaster_agree = 2

            if (("stat" in smoothed_scores and use_smart_consensus) or len(forecaster_names) >= min_forecaster_agree):
                if forecaster_names:
                    fc_vote = np.sum([predictions[n] for n in forecaster_names], axis=0)
                    fc_any_agree    = (fc_vote >= 1).astype(int)
                    fc_strong_agree = (fc_vote >= min_forecaster_agree).astype(int)
                else:
                    n_rows = len(df)
                    fc_any_agree    = np.zeros(n_rows, dtype=int)
                    fc_strong_agree = np.zeros(n_rows, dtype=int)

                if "stat" in smoothed_scores and use_smart_consensus:
                    stat_sc = smoothed_scores["stat"]
                    saved_stat = active_bundles.get("stat", {})
                    if "nominal_median" in saved_stat and "nominal_mad" in saved_stat:
                        nm, nmad = saved_stat["nominal_median"], saved_stat["nominal_mad"]
                        if nmad > 1e-6:
                            stat_strong_thr = nm + (k_val * 1.5) * nmad
                            stat_weak_thr   = nm + max(1.0, k_val * 0.6) * nmad
                        else:
                            stat_strong_thr = saved_stat.get("threshold_strong", _linear_threshold(stat_sc, k_val * 1.5))
                            stat_weak_thr   = saved_stat.get("threshold_weak", _linear_threshold(stat_sc, max(1.0, k_val * 0.6)))
                    else:
                        stat_strong_thr = _linear_threshold(stat_sc, k_val * 1.5)
                        stat_weak_thr   = _linear_threshold(stat_sc, max(1.0, k_val * 0.6))
                    stat_strong = (stat_sc > stat_strong_thr).astype(int)
                    stat_weak   = (stat_sc > stat_weak_thr).astype(int)
                    stat_win = _warmup_length("stat")
                    if stat_win:
                        stat_strong[:min(stat_win, len(stat_strong))] = 0
                        stat_weak[:min(stat_win, len(stat_weak))] = 0

                    display_basis = stat_sc
                    display_thr   = stat_strong_thr
                else:
                    stat_strong = np.zeros(len(df), dtype=int)
                    stat_weak   = np.zeros(len(df), dtype=int)
                    display_basis = (fc_vote.astype(np.float32) if forecaster_names
                                     else np.zeros(len(df), dtype=np.float32))
                    display_thr   = float(min_forecaster_agree)
                consensus_pred = np.maximum(stat_strong, stat_weak & fc_any_agree)
                consensus_pred = np.maximum(consensus_pred, fc_strong_agree)
                consensus_pred = _apply_min_duration(consensus_pred, min_duration)
                consensus_reason = np.zeros(len(consensus_pred), dtype=np.int8)
                consensus_reason[(stat_weak.astype(bool)) & (fc_any_agree.astype(bool))] = 2
                consensus_reason[fc_strong_agree.astype(bool)] = 3
                consensus_reason[stat_strong.astype(bool)] = 1   # strongest, evaluated last so it wins ties
                consensus_reason = consensus_reason * (consensus_pred > 0)  # only meaningful where actually confirmed

                predictions["consensus"]     = consensus_pred
                smoothed_scores["consensus"] = display_basis
                thresholds["consensus"]      = display_thr
                all_scores["consensus"]      = display_basis

            # Store everything in session state
            st.session_state["results"] = {
                "df": df, "sensors": sensors, "data_arr": data_arr,
                "all_scores": smoothed_scores,   # display smoothed scores
                "raw_scores": all_scores,         # keep raw for reference
                "thresholds": thresholds,
                "predictions": predictions, "elapsed": elapsed,
                "n_files": n_files, "enabled": enabled,
                "bundles": active_bundles,
                "threshold_basis": threshold_basis,
                "consensus_weak_thr": stat_weak_thr if "stat" in smoothed_scores and use_smart_consensus else None,
                "consensus_reason": consensus_reason if "consensus" in predictions else None,
                "consensus_forecaster_names": forecaster_names if "consensus" in predictions else [],
                "warmup_already_trimmed": trim_rows > 0,
            }
            st.session_state["run_config"] = {
                "data_dir": data_dir, "model_dir": model_dir,
                "agg_mode": agg_mode, "thr_method": thr_method,
                "k_val": k_val, "weights": weights,
                "source_mode": source_mode,
                "kafka_topic": kafka_topic if source_mode != "Local CSV files" else None,
            }
            status.update(label="Analysis complete! Go to Results.", state="complete")

        st.info("Switch to the **Results** tab to see detections.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 - Results
# ═══════════════════════════════════════════════════════════════════════════════
with tab_results:
    r = st.session_state.get("results")
    if r is None:
        st.info("No results yet. Go to **Configure & Run** and click **Run Analysis**.")
    else:
        df          = r["df"]
        sensors     = r["sensors"]
        all_scores  = r["all_scores"]
        thresholds  = r["thresholds"]
        predictions = r["predictions"]
        x = df["timestamp"].values if "timestamp" in df.columns else np.arange(len(df))

        # Summary metrics
        st.subheader("Detection Summary")
        cols = st.columns(len(all_scores) + 1)
        cols[0].metric("Rows analysed", f"{len(df):,}",
                       delta=f"{r['n_files']} files | {r['elapsed']:.1f}s")
        for i, (name, pred) in enumerate(predictions.items(), 1):
            flagged = int(pred.sum())
            cols[i].metric(
                MODEL_LABELS.get(name, name),
                f"{flagged:,} anomalies",
                delta=f"{flagged/len(df)*100:.1f}% of rows",
                delta_color="inverse",
            )

        st.divider()

        # ── Combined score chart ──
        st.subheader("Anomaly Scores - All Models")
        fig = go.Figure()

        for name, sc in all_scores.items():
            if name in ("ensemble", "consensus"):
                continue
            fig.add_trace(go.Scatter(
                x=x, y=sc,
                name=MODEL_LABELS.get(name, name),
                line=dict(color=MODEL_COLORS[name], width=1.4),
                opacity=0.85,
            ))

        if "ensemble" in all_scores:
            fig.add_trace(go.Scatter(
                x=x, y=all_scores["ensemble"],
                name="Ensemble",
                line=dict(color=MODEL_COLORS["ensemble"], width=2.5, dash="solid"),
            ))
            ens_pred = predictions["ensemble"]
            shade(fig, ens_pred, "rgba(192,132,252,0.18)", x=x)
            fig.add_hline(
                y=thresholds["ensemble"], line_dash="dash", line_color="#7E22CE",
                annotation_text="Ensemble threshold", annotation_position="top right",
            )

        if "consensus" in all_scores:
            cons_pred = predictions["consensus"]
            shade(fig, cons_pred, "rgba(255,59,59,0.22)", x=x)
            fig.add_annotation(
                text="Red bands = Confirmed (StatDetector + Forecaster consensus) - "
                     "the recommended primary signal",
                xref="paper", yref="paper", x=0, y=1.08, showarrow=False,
                font=dict(color="#DC2626", size=11), align="left",
            )

        # Bound the y-axis using the threshold range across all plotted models,
        # not the raw maximum (which can be a rare extreme single-point spike).
        all_thrs = [thresholds[n] for n in all_scores if n in thresholds]
        all_p999 = [float(np.percentile(sc, 99.9)) for n, sc in all_scores.items()]
        combined_visible_max = max(all_thrs + all_p999) * 3 if (all_thrs or all_p999) else 100
        nonzero_vals = np.concatenate([sc[sc > 0] for sc in all_scores.values() if (sc > 0).any()])
        combined_visible_min = max(1e-3, float(np.percentile(nonzero_vals, 1)) * 0.3) if len(nonzero_vals) else 1e-3

        fig.update_layout(
            xaxis_title="Timestamp", yaxis_title="Anomaly Score (log scale)",
            yaxis_type="log", height=420, template="plotly_white",
            yaxis_range=[np.log10(combined_visible_min), np.log10(combined_visible_max)],
            legend=dict(orientation="h", y=-0.28),
        )
        st.plotly_chart(fig, width='stretch')

        # ── Per-model individual charts ──
        st.subheader("Per-Model Detections")
        named_models = [k for k in all_scores if k != "ensemble"]
        threshold_basis = r.get("threshold_basis", {})
        for name in named_models:
            sc    = all_scores[name]
            pred  = predictions[name]
            thr   = thresholds[name]
            color = MODEL_COLORS[name]
            basis = threshold_basis.get(name, "")

            with st.expander(f"{MODEL_LABELS[name]}  -  {int(pred.sum()):,} anomalies flagged  ({pred.mean()*100:.1f}%)", expanded=True):
                if basis.startswith("live (log-MAD"):
                    st.warning(f"{name} is using a live-computed threshold (no nominal "
                              "calibration found in the saved model). A handful of rare, "
                              "harmless single-point forecast noise spikes can inflate this "
                              "threshold and make real anomalies harder to cross. **Retrain "
                              "with the current train.py to fix this** - it now saves a "
                              "threshold calibrated on clean nominal data.")

                fig2 = go.Figure()

                sc_display = sc.copy()
                if not r.get("warmup_already_trimmed", False):
                    if name == "consensus":
                        warm_lens = []
                        for bname, b in r["bundles"].items():
                            wl = b.get("window")
                            if wl is None:
                                wdet = b.get("detector")
                                if wdet is not None and hasattr(wdet, "window"):
                                    wl = wdet.window
                            if wl:
                                warm_lens.append(int(wl))
                        warm_len = max(warm_lens) if warm_lens else None
                    else:
                        warm_bundle = r["bundles"].get(name, {})
                        warm_len = warm_bundle.get("window")
                        if warm_len is None:
                            warm_det = warm_bundle.get("detector")
                            if warm_det is not None and hasattr(warm_det, "window"):
                                warm_len = warm_det.window
                    if warm_len:
                        sc_display[:min(int(warm_len), len(sc_display))] = np.nan
                fig2.add_trace(go.Scatter(
                    x=x, y=sc_display, name="Score",
                    fill="tozeroy",
                    fillcolor=hex_to_rgba(color, 0.15),
                    line=dict(color=color, width=1.2),
                ))
                shade(fig2, pred, hex_to_rgba(color, 0.25), x=x)
                fig2.add_hline(y=thr, line_dash="dash", line_color=color,
                               annotation_text=f"Threshold ({thr:.4f})",
                               annotation_position="top right")

                if name == "consensus":
                    weak_thr_display = r.get("consensus_weak_thr")
                    if weak_thr_display is not None:
                        fig2.add_hline(y=weak_thr_display, line_dash="dot", line_color=color,
                                       annotation_text=f"Weak threshold ({weak_thr_display:.4f})",
                                       annotation_position="bottom right")

                visible_max = max(thr, float(np.percentile(sc, 99.9))) * 3
                visible_min = max(1e-3, float(np.percentile(sc[sc > 0], 1)) * 0.3) if (sc > 0).any() else 1e-3
                fig2.update_layout(
                    yaxis_type="log", height=280, template="plotly_white",
                    yaxis_range=[np.log10(visible_min), np.log10(visible_max)],
                    xaxis_title="Timestamp", yaxis_title="Score",
                    showlegend=False, margin=dict(t=20, b=30),
                )
                st.plotly_chart(fig2, width='stretch')

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Max score",    f"{sc.max():.4f}")
                c2.metric("Median score", f"{np.median(sc):.6f}")
                c3.metric("Threshold",    f"{thr:.4f}")

                if name == "consensus" and pred.sum() > 0:
                    st.divider()
                    st.markdown("**Why was this confirmed?**")
                    reason_arr = r.get("consensus_reason")
                    fc_names_used = r.get("consensus_forecaster_names", [])
                    if reason_arr is None:
                        st.caption("Reason tracking unavailable for this run - rerun "
                                  "Configure & Run to enable it.")
                    else:
                        events_c = build_event_options(pred, sc, x)
                        if events_c:
                            chosen_c = render_event_navigator(events_c, key=f"confirmed_{name}")
                            idx_c = chosen_c["peak_idx"]
                            reason_code = int(reason_arr[idx_c])
                            reason_text = {
                                1: "**StatDetector alone was strongly elevated** - confident "
                                   "enough on its own, no other forecaster backup needed.",
                                2: "**StatDetector was only mildly elevated** - not enough to "
                                   "confirm alone - **but at least one forecaster independently "
                                   "agreed**, which was enough to set it to confirmed.",
                                3: "**Two or more forecasters agreed independently**, regardless "
                                   "of what StatDetector showed.",
                                0: "No specific reason recorded for this exact point (it may sit "
                                   "at the edge of a confirmed region due to smoothing).",
                            }.get(reason_code, "Unknown.")
                            st.markdown(reason_text)

                            if fc_names_used:
                                agreeing = [n for n in fc_names_used
                                           if n in predictions and predictions[n][idx_c]]
                                st.caption(
                                    f"Forecasters that agreed at this exact point: "
                                    f"{', '.join(MODEL_LABELS.get(n, n) for n in agreeing) or 'none'}."
                                )
                        else:
                            st.caption("No distinct confirmed events to explain here.")

                
                if name == "xgboost" and pred.sum() > 0:
                    st.divider()
                    st.markdown("**Why was this flagged?**")

                    try:
                        import shap  # noqa: F401  (import check only, used inside explain.py)
                        shap_available = True
                    except ImportError:
                        shap_available = False
                        st.warning("This feature needs one more setup step. Run "
                                  "`pip install shap` and restart the app to enable it.")

                    if shap_available:
                        events = build_event_options(pred, sc, x)
                        if not events:
                            st.info("No distinct anomaly events to explain here.")
                            chosen_idx = None
                        else:
                            chosen_event = render_event_navigator(events, key=f"shap_{name}")
                            chosen_idx = chosen_event["peak_idx"]
                            if chosen_event["end"] - chosen_event["start"] > 1:
                                st.caption("Showing the most extreme moment within this event.")
                        try:
                            from explain import (explain_xgboost_row, plain_summary,
                                                 plain_feature_reasons)
                            result = (explain_xgboost_row(r["bundles"]["xgboost"], r["data_arr"], chosen_idx)
                                     if chosen_idx is not None else None)
                        except Exception as e:
                            import traceback
                            st.error("Something went wrong computing this explanation.")
                            with st.expander("Technical error details"):
                                st.code(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
                            result = None

                        if result is not None:
                            st.markdown(plain_summary(result))

                            reasons = plain_feature_reasons(result["top_features"], max_items=3)
                            ordinals = ["The biggest reason:", "Another contributing reason:",
                                       "A smaller contributing reason:"]
                            for i, rsn in enumerate(reasons):
                                nudge = "which nudged the model's guess higher" if rsn["direction"] == "up" \
                                    else "which nudged the model's guess lower"
                                st.markdown(f"**{ordinals[i]}** {rsn['text']}, {nudge}.")

                            with st.expander("Technical details"):
                                feat_names = [f for f, _ in result["top_features"]]
                                feat_vals  = [v for _, v in result["top_features"]]
                                colors_bar = ["#15803D" if v > 0 else "#B91C1C" for v in feat_vals]
                                fig_shap = go.Figure(go.Bar(
                                    x=feat_vals, y=feat_names, orientation="h",
                                    marker_color=colors_bar,
                                ))
                                fig_shap.update_layout(
                                    xaxis_title="SHAP value", template="plotly_white",
                                    height=220, margin=dict(t=10, b=20),
                                )
                                st.plotly_chart(fig_shap, width='stretch')
                                dc1, dc2, dc3 = st.columns(3)
                                dc1.metric("Channel", result["channel"])
                                dc2.metric("Predicted → Actual",
                                          f"{result['predicted']:.2f} → {result['actual']:.2f}")
                                dc3.metric("Normalised error", f"{result['error']:.2f}")

                if name in ("lstm", "patchtst") and pred.sum() > 0:
                    st.divider()
                    st.markdown("**Why was this flagged?**")

                    try:
                        import captum
                        captum_available = True
                    except ImportError:
                        captum_available = False
                        st.warning("This feature needs one more setup step. Run "
                                  "`pip install captum` and restart the app to enable it.")

                    if captum_available:
                        window_needed = r["bundles"][name]["window"]
                        events = build_event_options(pred, sc, x, min_row=window_needed)
                        if not events:
                            st.info("Not enough history yet before any flagged event to explain it.")
                        else:
                            chosen_event = render_event_navigator(events, key=f"captum_{name}")
                            chosen_idx = chosen_event["peak_idx"]
                            if chosen_event["end"] - chosen_event["start"] > 1:
                                st.caption("Showing the most extreme moment within this event.")
                            try:
                                from explain import (explain_forecaster_row, top_attribution_cells,
                                                     plain_summary, plain_attribution_reasons)
                                cresult = explain_forecaster_row(r["bundles"][name], r["data_arr"], chosen_idx)
                            except Exception as e:
                                import traceback
                                st.error("Something went wrong computing this explanation.")
                                with st.expander("Technical error details"):
                                    st.code(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
                                cresult = None

                            if cresult is not None:
                                st.markdown(plain_summary(cresult))

                                top_cells = top_attribution_cells(cresult, top_k=6)
                                reasons = plain_attribution_reasons(top_cells, max_items=3)
                                ordinals = ["The biggest reason:", "Another contributing reason:",
                                           "A smaller contributing reason:"]
                                for i, rsn in enumerate(reasons):
                                    nudge = "which nudged the model's guess higher" if rsn["direction"] == "up" \
                                        else "which nudged the model's guess lower"
                                    st.markdown(f"**{ordinals[i]}** {rsn['text']}, {nudge}.")

                                confidence_ok = abs(cresult["convergence_delta"]) < 0.5
                                if not confidence_ok:
                                    st.caption("Note: this explanation is a rougher estimate than "
                                              "usual for this particular moment - consider it a"
                                              "general guide rather than an exact explanation.")

                                with st.expander("Technical details"):
                                    attr = cresult["attribution"]
                                    seconds_ago = list(range(cresult["window"], 0, -1))
                                    fig_captum = go.Figure(go.Heatmap(
                                        z=attr.T, x=seconds_ago, y=cresult["sensors"],
                                        colorscale="RdBu", zmid=0,
                                        colorbar=dict(title="Attribution"),
                                    ))
                                    fig_captum.update_layout(
                                        xaxis_title="Seconds before the prediction",
                                        xaxis_autorange="reversed",
                                        template="plotly_white", height=220,
                                        margin=dict(t=10, b=20),
                                    )
                                    st.plotly_chart(fig_captum, width='stretch')
                                    dc1, dc2, dc3, dc4 = st.columns(4)
                                    dc1.metric("Channel", cresult["channel"])
                                    dc2.metric("Predicted → Actual",
                                              f"{cresult['predicted']:.2f} → {cresult['actual']:.2f}")
                                    dc3.metric("Normalised error", f"{cresult['error']:.2f}")
                                    dc4.metric("Convergence delta",
                                              f"{cresult['convergence_delta']:+.4f}",
                                              help="Near zero means this explanation is reliable; "
                                                   "large means treat it as a rough guide only.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 - Deep Dive
# ═══════════════════════════════════════════════════════════════════════════════
with tab_deep:
    r = st.session_state.get("results")
    if r is None:
        st.info("Run analysis first (Configure & Run tab).")
    else:
        df_d = r["df"]; sensors_d = r["sensors"]
        predictions_d = r["predictions"]
        deep_x = df_d["timestamp"].values if "timestamp" in df_d.columns else np.arange(len(df_d))

        st.subheader("Channel Signals with Anomaly Overlays")
        with st.expander("Anomaly type quick reference"):
            for atype, (label, desc) in ANOMALY_TYPE_INFO.items():
                st.markdown(f"**{label}** - {desc}")


        non_ens_preds = [np.asarray(p).ravel() for k, p in predictions_d.items() if k != "ensemble"]
        if non_ens_preds:
            any_flag = np.clip(np.sum(non_ens_preds, axis=0), 0, 1).astype(int)
        else:
            any_flag = np.zeros(len(df_d), dtype=int)

        n_ch = len(sensors_d)
        fig3 = make_subplots(rows=n_ch, cols=1, shared_xaxes=True,
                             subplot_titles=sensors_d, vertical_spacing=0.05)
        for i, col in enumerate(sensors_d, 1):
            fig3.add_trace(
                go.Scatter(x=deep_x, y=df_d[col].values, name=col,
                           line=dict(color="#2563AA", width=1)),
                row=i, col=1,
            )

        shapes = []
        for i in range(1, n_ch + 1):
            xref = "x" if i == 1 else f"x{i}"
            yref = "y domain" if i == 1 else f"y{i} domain"
            for s, e in _regions(any_flag):
                x0 = deep_x[min(s, len(deep_x)-1)]
                x1 = deep_x[min(e, len(deep_x)-1)]
                if "datetime" in str(type(x0)):
                    x0, x1 = pd.Timestamp(x0).isoformat(), pd.Timestamp(x1).isoformat()
                shapes.append(dict(
                    type="rect", xref=xref, yref=yref,
                    x0=x0, x1=x1, y0=0, y1=1,
                    fillcolor="rgba(241,107,107,0.20)",
                    line=dict(color="rgba(241,107,107,0.9)", width=1),
                    layer="below",
                ))

        fig3.update_layout(
            height=200*n_ch, template="plotly_white", showlegend=False,
            shapes=shapes,
            title="Sensor Signals (shaded = flagged by at least one active model)",
        )
        st.plotly_chart(fig3, width='stretch')

        


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 - Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    r = st.session_state.get("results")
    _named = [k for k in r.get("all_scores", {}) if k != "ensemble"] if r else []
    if r is None:
        st.info("Run analysis first (Configure & Run tab).")
    elif len(_named) < 2:
        st.info("Select at least two models to use the comparison tab.")
    else:
        all_scores  = r["all_scores"]
        predictions = r["predictions"]
        df_c        = r["df"]
        ts_x = df_c["timestamp"].values if "timestamp" in df_c.columns else np.arange(len(df_c))
        named = _named

        # ── Stacked per-model view ──
        st.subheader("Side-by-Side Score Chart (log scale)")
        fig5 = make_subplots(
            rows=len(named), cols=1, shared_xaxes=True, vertical_spacing=0.06,
            subplot_titles=[MODEL_LABELS.get(n, n) for n in named],
        )
        cmp_shapes = []
        for i, name in enumerate(named, 1):
            sc    = all_scores[name]
            pred  = predictions[name]
            thr   = r["thresholds"][name]
            color = MODEL_COLORS[name]
            fig5.add_trace(
                go.Scatter(x=ts_x, y=sc, name=MODEL_LABELS[name],
                           line=dict(color=color, width=1.1)),
                row=i, col=1,
            )
            fig5.add_hline(y=thr, line_dash="dot", line_color=color, row=i, col=1)
            # Batch shapes instead of calling shade() with row/col
            xref = "x" if i == 1 else f"x{i}"
            yref = "y domain" if i == 1 else f"y{i} domain"
            rgba = hex_to_rgba(color, 0.2)
            for s, e in _regions(pred):
                x0 = ts_x[min(s, len(ts_x)-1)]
                x1 = ts_x[min(e, len(ts_x)-1)]
                if "datetime" in str(type(x0)):
                    x0, x1 = pd.Timestamp(x0).isoformat(), pd.Timestamp(x1).isoformat()
                cmp_shapes.append(dict(
                    type="rect", xref=xref, yref=yref,
                    x0=x0, x1=x1, y0=0, y1=1,
                    fillcolor=rgba, line=dict(color=hex_to_rgba(color, 0.9), width=1),
                    layer="below",
                ))
        fig5.update_yaxes(type="log")
        fig5.update_layout(
            height=240*len(named), template="plotly_white",
            showlegend=False, shapes=cmp_shapes,
        )
        st.plotly_chart(fig5, width='stretch')

        st.divider()

        # ── Consensus ──
        st.subheader("Consensus Detections")
        vote = sum(predictions[n] for n in named)
        vcols = st.columns(3)
        for i, min_votes in enumerate([1, 2, len(named)]):
            if min_votes > len(named):
                continue
            cons  = (vote >= min_votes).astype(int)
            label = {1: "Any model", 2: "≥2 models", len(named): "All models"}[min_votes]
            vcols[i].metric(label, f"{cons.sum():,}",
                            delta=f"{cons.mean()*100:.1f}% of rows", delta_color="inverse")

        unanimous = (vote == len(named)).astype(int)
        if unanimous.sum() > 0:
            fig7 = go.Figure()
            fig7.add_trace(go.Scatter(
                x=ts_x, y=vote.astype(float), fill="tozeroy",
                fillcolor="rgba(192,132,252,0.2)",
                line=dict(color="#7E22CE", width=1.5), name="Vote count",
            ))
            shade(fig7, unanimous, "rgba(255,80,80,0.25)", x=ts_x)
            fig7.add_hline(y=len(named)-0.05, line_dash="dash", line_color="red",
                           annotation_text="All models agree")
            fig7.update_layout(
                title="Model Vote Count (red = unanimous anomaly)",
                xaxis_title="Timestamp", yaxis_title="# models flagging",
                height=280, template="plotly_white", showlegend=False,
            )
            st.plotly_chart(fig7, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# TAB - Data Health (drift monitoring)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_health:
    from drift import (check_drift, overall_drift_status, _drift_comparable_values,
                   PSI_STABLE_THRESHOLD, PSI_MODERATE_THRESHOLD)

    st.subheader("Has \"normal\" quietly changed?")
    
    st.divider()

    saved_cfg = st.session_state.get("run_config", {})
    hc1, hc2 = st.columns(2)
    with hc1:
        health_model_dir = st.text_input(
            "Model directory", value=saved_cfg.get("model_dir", "models"), key="health_model_dir",
        )
        health_source = st.radio(
            "Data source", ["Kafka topic", "Local CSV files"], horizontal=True,
            index=0 if saved_cfg.get("source_mode", "Kafka topic") != "Local CSV files" else 1,
            key="health_source",
        )
    with hc2:
        health_n = st.slider("Chunks to check against (most recent)", 5, 50, 20, key="health_n")
        if health_source == "Local CSV files":
            health_data_dir = st.text_input(
                "Telemetry stream folder", value=saved_cfg.get("data_dir", "live_telemetry_stream"),
                key="health_data_dir",
            )
            health_kafka_bootstrap = health_kafka_topic = None
        else:
            health_kafka_bootstrap = st.text_input("Kafka bootstrap servers", value="localhost:9092",
                                                    key="health_kafka_bootstrap")
            health_kafka_topic = st.text_input(
                "Kafka topic", value=saved_cfg.get("kafka_topic") or "telemetry.raw",
                key="health_kafka_topic",
            )
            health_data_dir = None

    baseline_path = os.path.join(health_model_dir, "drift_baseline.pkl")
    baseline_exists = os.path.exists(baseline_path)
    if not baseline_exists:
        st.warning(
            f"No drift baseline found at `{baseline_path}`. This file is created "
            "automatically the next time you train - retrain with the current "
            "train.py to enable this check."
        )

    if st.button("Check Drift", type="primary", disabled=not baseline_exists):
        with st.spinner("Loading recent data and comparing against the training baseline..."):
            if health_source == "Local CSV files":
                health_df = load_data(health_data_dir, health_n)
                source_err = None if health_df is not None else \
                    f"No CSV files found in `{health_data_dir}`."
            else:
                health_df, source_err = load_data_kafka(health_kafka_bootstrap, health_kafka_topic, health_n)

            if health_df is None:
                st.error(source_err)
            else:
                import pickle
                baseline = pickle.load(open(baseline_path, "rb"))
                sensors_h = baseline["sensors"]
                missing = [s for s in sensors_h if s not in health_df.columns]
                if missing:
                    st.error(f"Live data is missing expected columns: {missing}")
                else:
                    raw_h = health_df[sensors_h].values.astype(np.float32)
                    reports = check_drift(baseline, raw_h, sensors_h)
                    status = overall_drift_status(reports)
                    st.session_state["drift_results"] = {
                        "reports": reports, "status": status,
                        "n_rows": len(health_df), "n_chunks": health_n,
                        "source": health_source,
                        "df": health_df, "sensors": sensors_h,
                        "ch_types": baseline.get("ch_types", {}),
                    }

    dr = st.session_state.get("drift_results")
    if dr:
        st.divider()
        status = dr["status"]
        banner = {"stable": st.success, "moderate": st.warning, "significant": st.error}[status]
        banner_text = {
            "stable": f"Stable - no meaningful drift across {dr['n_rows']:,} recent rows. "
                     "Current models should still be reliable.",
            "moderate": f"Moderate drift detected across {dr['n_rows']:,} recent rows. "
                       "Not urgent, but worth watching - consider retraining if this "
                       "persists across repeated checks.",
            "significant": f"Significant drift detected across {dr['n_rows']:,} recent rows. "
                          "Consider retraining.",
        }[status]
        banner(banner_text)

        st.markdown("**Per-channel breakdown**")
        for rep in dr["reports"]:
            with st.expander(f"{rep['channel']}  -  {rep['severity']}  (PSI = {rep['psi']:.3f})",
                             expanded=(rep["severity"] != "stable")):

                pc1, pc2, pc3 = st.columns([1, 1, 1])
                pc1.metric("Baseline (training time)",
                          f"mean {rep['baseline_mean']:.3f}, spread {rep['baseline_std']:.3f}")
                pc2.metric("Current (just now)",
                          f"mean {rep['current_mean']:.3f}, spread {rep['current_std']:.3f}")
                severity_color = {"stable": "#15803D", "moderate": "#B45309", "significant": "#DC2626"}[rep["severity"]]
                with pc3:
                    st.markdown("PSI (drift score)")
                    st.markdown(
                        f"<span style='font-size:1.9rem; font-weight:600; color:{severity_color}'>"
                        f"{rep['psi']:.3f}</span> "
                        f"<span style='color:{severity_color}; font-weight:600'>{rep['severity']}</span>",
                        unsafe_allow_html=True,
                    )
                bvals = rep["baseline_samples"]
                cvals = rep["current_samples"]
                fig_h = go.Figure()
                fig_h.add_trace(go.Histogram(
                    x=bvals, name="Baseline (training time)", opacity=0.55,
                    marker_color="#1D4ED8", histnorm="probability density", nbinsx=40,
                ))
                fig_h.add_trace(go.Histogram(
                    x=cvals, name="Current (just now)", opacity=0.55,
                    marker_color="#DC2626", histnorm="probability density", nbinsx=40,
                ))
                fig_h.add_vline(x=float(np.nanmean(bvals)), line_dash="dash", line_color="#1D4ED8",
                               annotation_text="baseline mean", annotation_position="top left")
                fig_h.add_vline(x=float(np.nanmean(cvals)), line_dash="dash", line_color="#DC2626",
                               annotation_text="current mean", annotation_position="top right")
                fig_h.update_layout(
                    barmode="overlay", template="plotly_white", height=280,
                    margin=dict(t=30, b=10, l=10, r=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis_title=f"{rep['channel']} value (differenced)" if rep["channel"] != "cso1"
                                else f"{rep['channel']} value",
                    yaxis_title="density",
                )
                st.plotly_chart(fig_h, width='stretch')
 
        st.caption(f"Thresholds: stable < {PSI_STABLE_THRESHOLD}, "
                  f"moderate < {PSI_MODERATE_THRESHOLD}, significant \u2265 {PSI_MODERATE_THRESHOLD}.")

        st.caption(f"Thresholds: stable < {PSI_STABLE_THRESHOLD}, "
                  f"moderate < {PSI_MODERATE_THRESHOLD}, significant \u2265 {PSI_MODERATE_THRESHOLD}.")


