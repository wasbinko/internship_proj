# Internship Telemetry ML Pipeline

An end-to-end Machine Learning and monitoring pipeline for processing, modeling, and alerting on telemetry data. This application utilizes Docker for containerization (Kafka), MLflow for model tracking, Streamlit for an interactive dashboard, and a dedicated Python daemon for proactive email alerts. Desktop GUI wrappers are also available for training and running the alert daemon without needing a terminal.

## Table of Contents
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Monitoring and Diagnostics](#monitoring-and-diagnostics)

---

## Project Structure

```text
Internship_proj/
├── alert_daemon.py          # Background service for triggering email alerts
├── daemon_gui.py            # Desktop GUI wrapper for starting/stopping the alert daemon
├── train_gui.py             # Desktop GUI wrapper for training models
├── simulate_drift.py        # Generates deliberately drifted data to demo the Data Health tab
├── app/
│   └── app.py               # Streamlit application for real-time dashboard and deep dives
├── docker-compose.yml       # Docker orchestration for Kafka
├── generate_telemetry.py    # Script to simulate or ingest raw telemetry data with anomalies
├── requirements.txt         # Python dependencies
└── scripts/                 # ML and data processing scripts
    ├── check_drift.py       # Standalone data drift check against the training baseline
    ├── drift.py              # Core Population Stability Index (PSI) drift calculation logic
    ├── explain.py             # Model explainability (SHAP for XGBoost, Captum for forecasters)
    ├── infer.py                # Runs model inference and scoring on incoming data
    ├── kafka_io.py              # Handles Kafka producers/consumers for data streams
    ├── models.py                 # ML model definitions (LSTM, PatchTST, XGBoost, IF, StatDetector)
    ├── nhits_model.py             # NHITS forecasting model (via Darts) — optional detector
    └── train.py                    # Model training, MLflow logging, and drift baseline snapshotting

```

---

## Core Components

### 1. Telemetry Data Generation (`generate_telemetry.py`)

This script acts as the entry point for your data stream. It generates simulated real-time telemetry data across three sensor channels and pushes it into the pipeline via local CSV files, Kafka, or both (`--sink csv|kafka|both`). It features a built-in anomaly injector (`frozen_sensor`, `contextual_break`, `massive_spike`) for testing.

### 2. Machine Learning & MLflow (`scripts/train.py`)

Trains up to seven detectors — StatDetector, LSTM, PatchTST, XGBoost, Isolation Forest, and two optional forecasters via the Darts library, NHITS and TiDE — with contamination-robust training. It leverages **MLflow** using a local SQLite backend (`mlruns/mlflow.db`) to track experiments, log metrics, and manage model versions: every training run is recorded automatically, with zero manual note-taking, so any regression has an exact record to compare against and revert to. Training data can be pulled from local CSV files or read directly from a Kafka topic (`--source csv|kafka`), and every run automatically snapshots a drift baseline used later by the drift-monitoring tools.

TiDE is the lighter-weight of the two optional Darts forecasters — a simpler, MLP-based architecture that trains noticeably faster than NHITS with comparable accuracy, useful if training speed matters more than squeezing out the last bit of accuracy.

Prefer not to use the terminal for this? See **`train_gui.py`** below.

### 3. Interactive Web Dashboard (`app/app.py`)

A comprehensive Streamlit interface that provides:

* Real-time scoring and detection overlays across all trained models, with a Smart Consensus "Confirmed" signal and a transparent breakdown of exactly why any given detection fired.
* A "Deep Dive" tab for investigating specific sensor anomalies.
* A "Model Comparison" tab for side-by-side score comparison across models.
* Explainability integration — SHAP for XGBoost, Captum for the forecasters (LSTM, PatchTST, NHITS, TiDE) — with plain-language explanations of why a specific moment was flagged.
* A "Data Health" tab for drift monitoring (Population Stability Index) against the training baseline. See `simulate_drift.py` below for a quick way to demo this tab reacting to real drift.
* Optional scheduled auto-run, synced to the data's own timestamps rather than wall-clock time — a run fires once genuinely new data has arrived, not on a fixed clock offset from whenever auto-run happened to be enabled.
* Supports both local CSV files and a live Kafka topic as the data source.

### 4. Email Alerting Daemon (`alert_daemon.py`)

A persistent background script that monitors the data stream (CSV or Kafka, via `--source csv|kafka`). It uses a "Smart Consensus" mechanism (combining StatDetector and the available forecasters) to prevent false positives. If a confirmed anomaly occurs, it dispatches an email alert to configured stakeholders.

Data folder, model folder, and email sender/receiver are configured as constants inside the script itself — see the note under Prerequisites below.

Prefer not to use the terminal for this? See **`daemon_gui.py`** below.

### 5. Desktop GUI Utilities (`train_gui.py`, `daemon_gui.py`)

Two small Tkinter desktop windows that wrap the terminal workflows above, useful for demos or day-to-day convenience without needing to type commands:

* **`train_gui.py`** — checkboxes for which models to train, a data source picker, and a Train Models button. Streams the real `train.py` output live into the window, including the MLflow confirmation lines, and shows a completion popup once done.
* **`daemon_gui.py`** — Start/Stop buttons for the alert daemon, with the same live log streaming. Closing the window while the daemon is still running prompts for confirmation first, so it doesn't get left running invisibly in the background.

Both are thin wrappers around the exact same underlying scripts used from the terminal — they don't reimplement or change any training, scoring, or alerting logic, just the way you interact with it.

### 6. Docker Containerization (`docker-compose.yml`)

Spins up an Apache Kafka broker in KRaft mode (no Zookeeper required) to handle high-throughput telemetry streams natively.

---

## Prerequisites

Before running the application, ensure you have the following installed:

* **Docker & Docker Compose** (for running Kafka)
* **Python 3.10+**
* **Required Python Packages:** `pip install -r requirements.txt` (Note: install `shap`, `captum`, `darts`, `pytorch-lightning`, and `streamlit-autorefresh` if you want to use the explainability, NHITS/TiDE, and scheduled auto-run features).
* **Tkinter**, if you want to use `train_gui.py` or `daemon_gui.py` — this ships by default with the standard python.org Windows installer (make sure "tcl/tk and IDLE" is checked during setup if it's missing).

> **Note on Email Alerts:** Before running `alert_daemon.py` (or `daemon_gui.py`, which runs it underneath), ensure you update the `EMAIL_SENDER`, `EMAIL_PASSWORD` (use an App Password), and `EMAIL_RECEIVER` variables in the script.
>
>

---

## Getting Started

### Step 1: Start Kafka Infrastructure

Start the Kafka broker using Docker Compose:

```bash
docker-compose up -d

```

### Step 2: Generate Telemetry Data

Start feeding data into the system. You can sink data to `csv`, `kafka`, or `both`.

```bash
# Keep this running in the background or a separate terminal
python generate_telemetry.py --sink kafka

```

### Step 3: Train the Models

Wait for a few chunks of data to be generated, then train the anomaly detection models. By default, this uses MLflow for tracking.

```bash
# Train on the generated Kafka stream and trim 25% of anomalous data out during training
python scripts/train.py --source kafka --kafka_topic telemetry.raw --model_dir models --trim 0.25 --models stat lstm patchtst xgboost iforest nhits

```

*Add `tide` to `--models` too if you want the lighter-weight Darts forecaster alongside NHITS.*

*Prefer a GUI? Run `python train_gui.py` instead — check the models you want, pick your data source, and click Train Models.*

*To view MLflow training runs, run: `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db` and open `localhost:5000`*

### Step 4: Run the Dashboard

Open the interactive Streamlit dashboard to view incoming data and anomalies:

```bash
streamlit run app/app.py

```

*Once open, go to "Configure & Run", select "Kafka topic" as your source, and click "Run Analysis". Enable "Auto-run" on the same tab if you want the dashboard to re-score itself automatically, synced to newly arriving data, instead of running it manually each time.*

### Step 5: Start the Alert Daemon

To enable automated email alerting for confirmed anomalies:

```bash
python alert_daemon.py --source kafka

```

*Prefer a GUI? Run `python daemon_gui.py` instead — pick your data source and click Start Daemon.*

---

## Monitoring and Diagnostics

To manually check for data drift or diagnose statistical baseline behavior, utilize the dedicated scripts (the same checks are also available directly in the dashboard's "Data Health" tab):

**Data Drift Check:**
Checks whether the current telemetry data has statistically drifted (using Population Stability Index) from the clean baseline established during training.

```bash
python scripts/check_drift.py --source kafka --kafka_topic telemetry.raw --model_dir models

```

**Drift Demo Generator:**
Writes deliberately drifted CSV data (choose the channel and severity) so you can see the Data Health tab react to a real drift event without waiting for one to happen naturally — useful for demos.

```bash
python simulate_drift.py --model_dir models --channel cso1 --severity significant

```

**StatDetector Diagnostics:**
Prints StatDetector's saved calibration and a per-channel score breakdown against live data — including which channel is actually driving any flagged points — useful for tracing exactly why a channel is or isn't being flagged.

```bash
python diagnose_stat_v2.py --model_dir models --source kafka --kafka_topic telemetry.raw --n_files 20

```

**Sensor Count Check:**
Quickly confirms which sensor channels a saved model was trained on — useful after any change to the data generator, to confirm your trained models match the current data shape before scoring against it.

```bash
python check_model_sensors.py models

```
