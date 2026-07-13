# Internship Telemetry ML Pipeline

An end-to-end Machine Learning and monitoring pipeline for processing, modeling, and alerting on telemetry data. This application utilizes Docker for containerization (Kafka), MLflow for model tracking, Streamlit for an interactive dashboard, and a dedicated Python daemon for proactive email alerts. A single desktop control panel, under `guis/`, is the recommended way to generate data, train models, and run the alert daemon without needing a terminal.

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
├── guis/
│   ├── control_panel.py     # Combined desktop GUI: generation, training, and the alert daemon in one window
│   ├── telemetry_gui.py     # Standalone GUI wrapper for generating telemetry data
│   ├── train_gui.py         # Standalone GUI wrapper for training models
│   └── daemon_gui.py        # Standalone GUI wrapper for starting/stopping the alert daemon
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
    ├── nhits_model.py             # NHITS forecasting model (via Darts) - optional detector
    └── train.py                    # Model training, MLflow logging, and drift baseline snapshotting

```

---

## Core Components

### 1. Telemetry Data Generation (`generate_telemetry.py`)

This script acts as the entry point for your data stream. It generates simulated real-time telemetry data across three sensor channels and pushes it into the pipeline via local CSV files, Kafka, or both (`--sink csv|kafka|both`). It features a built-in anomaly injector (`frozen_sensor`, `contextual_break`, `massive_spike`) for testing.

The recommended way to run this is the **Generate Telemetry** tab in `guis/control_panel.py` (see below) - the terminal command still works exactly the same underneath if you prefer it.

### 2. Machine Learning & MLflow (`scripts/train.py`)

Trains up to seven detectors - StatDetector, LSTM, PatchTST, XGBoost, Isolation Forest, and two optional forecasters via the Darts library, NHITS - with contamination-robust training. It leverages **MLflow** using a local SQLite backend (`mlruns/mlflow.db`) to track experiments, log metrics, and manage model versions: every training run is recorded automatically, with zero manual note-taking, so any regression has an exact record to compare against and revert to. Training data can be pulled from local CSV files or read directly from a Kafka topic (`--source csv|kafka`), and every run automatically snapshots a drift baseline used later by the drift-monitoring tools.

The recommended way to run this is the **Train Models** tab in `guis/control_panel.py` (see below) - the terminal command still works exactly the same underneath if you prefer it.

### 3. Interactive Web Dashboard (`app/app.py`)

A comprehensive Streamlit interface that provides:

* Real-time scoring and detection overlays across all trained models, with a Smart Consensus "Confirmed" signal and a transparent breakdown of exactly why any given detection fired.
* A "Deep Dive" tab for investigating specific sensor anomalies.
* A "Model Comparison" tab for side-by-side score comparison across models.
* Explainability integration - SHAP for XGBoost, Captum for the forecasters (LSTM, PatchTST, NHITS) - with plain-language explanations of why a specific moment was flagged.
* A "Data Health" tab for drift monitoring (Population Stability Index) against the training baseline. See `simulate_drift.py` below for a quick way to demo this tab reacting to real drift.
* Optional scheduled auto-run, synced to the data's own timestamps rather than wall-clock time - a run fires once genuinely new data has arrived, not on a fixed clock offset from whenever auto-run happened to be enabled.
* Supports both local CSV files and a live Kafka topic as the data source.

### 4. Email Alerting Daemon (`alert_daemon.py`)

A persistent background script that monitors the data stream (CSV or Kafka, via `--source csv|kafka`). It uses a "Smart Consensus" mechanism (combining StatDetector and the available forecasters) to prevent false positives. If a confirmed anomaly occurs, it dispatches an email alert to configured stakeholders.

Data folder and model folder are configured as constants inside the script itself - see the note under Prerequisites below. The email recipient can be set directly in the GUI instead, without editing the script.

The recommended way to run this is the **Alert Daemon** tab in `guis/control_panel.py` (see below) - the terminal command still works exactly the same underneath if you prefer it.

### 5. Desktop GUI Utilities (`guis/`)

**`control_panel.py`** is the recommended way to run this project day to day, or for a demo: one window with three tabs, so generating data, training, and running the alert daemon no longer means juggling separate windows.

* **Generate Telemetry** - output sink (CSV / Kafka / Both), anomaly probability, and interval, plus a Start/Stop button. Streams the real `generate_telemetry.py` output live, including each injected anomaly as it happens.
* **Train Models** - checkboxes for which models to train, a data source picker, and a Train Models button. Streams the real `train.py` output live into the window, including the MLflow confirmation lines, and shows a completion popup once done.
* **Alert Daemon** - Start/Stop buttons for the alert daemon, a field to set the alert recipient without editing the script, and the same live log streaming.

Closing the window while the generator or the daemon is still running prompts for confirmation first, so nothing gets left running invisibly in the background - and since both can genuinely run at once, each tab tracks its own process independently, so stopping one never affects the other.

The three tabs are thin wrappers around the exact same underlying scripts used from the terminal - they don't reimplement or change any generation, training, scoring, or alerting logic, just the way you interact with it. The terminal commands are still documented in full under Getting Started, for anyone who prefers them.

`telemetry_gui.py`, `train_gui.py`, and `daemon_gui.py` are still available individually in the same folder, for anyone who'd rather keep these as separate windows instead of tabs in one.

### 6. Docker Containerization (`docker-compose.yml`)

Spins up an Apache Kafka broker in KRaft mode (no Zookeeper required) to handle high-throughput telemetry streams natively.

---

## Prerequisites

Before running the application, ensure you have the following installed:

* **Docker & Docker Compose** (for running Kafka)
* **Python 3.10+**
* **Required Python Packages:** `pip install -r requirements.txt` (Note: install `shap`, `captum`, `darts`, `pytorch-lightning`, and `streamlit-autorefresh` if you want to use the explainability, NHITS, and scheduled auto-run features).
* **Tkinter**, for anything in `guis/` - this ships by default with the standard python.org Windows installer (make sure "tcl/tk and IDLE" is checked during setup if it's missing).

> **Note on Email Alerts:** Before running `alert_daemon.py` (or the Alert Daemon tab in `guis/control_panel.py`, which runs it underneath), ensure you update the `EMAIL_SENDER` and `EMAIL_PASSWORD` (use an App Password) variables in the script. The recipient (`EMAIL_RECEIVER`) can either be set the same way, or overridden directly from the GUI without touching the script.
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

Start feeding data into the system.

```bash
python guis/control_panel.py

```

On the **Generate Telemetry** tab, choose your output sink (CSV, Kafka, or Both), set the anomaly probability and interval, and click **Start Generating**. Leave the window open - it runs continuously, the same way the terminal version would in the background.

*If you'd rather run this in the terminal:*

```bash
# Keep this running in the background or a separate terminal
python generate_telemetry.py --sink kafka

```

### Step 3: Train the Models

Wait for a few chunks of data to be generated, then train the anomaly detection models.

In the same control panel window, switch to the **Train Models** tab. Check the models you want, pick your data source, and click **Train Models**. The window streams the real training output live, including the MLflow confirmation lines, and shows a completion popup once training finishes.

*If you'd rather run this in the terminal:*

```bash
# Train on the generated Kafka stream and trim 25% of anomalous data out during training
python scripts/train.py --source kafka --kafka_topic telemetry.raw --model_dir models --trim 0.25 --models stat lstm patchtst xgboost iforest nhits

```

*To view MLflow training runs either way, run: `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db` and open `localhost:5000`*

### Step 4: Run the Dashboard

Open the interactive Streamlit dashboard to view incoming data and anomalies:

```bash
streamlit run app/app.py

```

*Once open, go to "Configure & Run", select "Kafka topic" as your source, and click "Run Analysis". Enable "Auto-run" on the same tab if you want the dashboard to re-score itself automatically, synced to newly arriving data, instead of running it manually each time.*

### Step 5: Start the Alert Daemon

To enable automated email alerting for confirmed anomalies, switch to the **Alert Daemon** tab in the same control panel window.

Pick your data source, optionally set an alert recipient, and click **Start Daemon**. Closing the window while it's still running will ask for confirmation first, so it doesn't keep running invisibly in the background.

*If you'd rather run this in the terminal:*

```bash
python alert_daemon.py --source kafka

```

---

## Monitoring and Diagnostics

To manually check for data drift or diagnose statistical baseline behavior, utilize the dedicated scripts (the same checks are also available directly in the dashboard's "Data Health" tab):

**Drift Demo Generator:**
Writes deliberately drifted CSV data (choose the channel and severity) so you can see the Data Health tab react to a real drift event without waiting for one to happen naturally - useful for demos.

```bash
python simulate_drift.py --model_dir models --channel cso1 --severity significant

```
