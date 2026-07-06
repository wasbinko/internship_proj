# Internship Telemetry ML Pipeline

An end-to-end Machine Learning and monitoring pipeline for processing, modeling, and alerting on telemetry data. This application utilizes Docker for containerization (Kafka), MLflow for model tracking, Streamlit for an interactive dashboard, and a dedicated Python daemon for proactive email alerts.

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
├── app/
│   └── app.py               # Streamlit application for real-time dashboard and deep dives
├── docker-compose.yml       # Docker orchestration for Kafka
├── generate_telemetry.py    # Script to simulate or ingest raw telemetry data with anomalies
├── requirements.txt         # Python dependencies
└── scripts/                 # ML and data processing scripts
    ├── check_drift.py       # Standalone data drift check against the training baseline
    ├── diagnose_stat.py     # Statistical diagnostics for the StatDetector
    ├── drift.py             # Core Population Stability Index (PSI) drift calculation logic
    ├── explain.py           # Model explainability (SHAP for XGBoost, Captum for forecasters)
    ├── infer.py              # Runs model inference and scoring on incoming data
    ├── kafka_io.py            # Handles Kafka producers/consumers for data streams
    ├── kafka_peek.py          # Utility to inspect recent messages on a Kafka topic
    ├── make_labeled_data.py   # Builds a labeled test set for measuring detection quality
    ├── models.py              # ML model definitions (LSTM, PatchTST, XGBoost, IF, StatDetector)
    ├── nhits_model.py         # NHITS forecasting model (via Darts) — optional sixth detector
    └── train.py               # Model training, MLflow logging, and drift baseline snapshotting

```

---

## Core Components

### 1. Telemetry Data Generation (`generate_telemetry.py`)

This script acts as the entry point for your data stream. It generates simulated real-time telemetry data (e.g., system metrics, sensor readings) and pushes it into the pipeline via local CSV files, Kafka, or both (`--sink csv|kafka|both`). It features a built-in anomaly injector (`frozen_sensor`, `contextual_break`, `massive_spike`) for testing.

### 2. Machine Learning & MLflow (`scripts/train.py`)

Trains up to six detectors — StatDetector, LSTM, PatchTST, XGBoost, Isolation Forest, and an optional NHITS model (via Darts) — with contamination-robust training. It leverages **MLflow** using a local SQLite backend (`mlruns/mlflow.db`) to track experiments, log metrics, and manage model versions. Training data can be pulled from local CSV files or read directly from a Kafka topic (`--source csv|kafka`), and every run automatically snapshots a drift baseline used later by the drift-monitoring tools.

### 3. Interactive Web Dashboard (`app/app.py`)

A comprehensive Streamlit interface that provides:

* Real-time scoring and detection overlays across all trained models, with a Smart Consensus "Confirmed" signal and a transparent breakdown of exactly why any given detection fired.
* A "Deep Dive" tab for investigating specific sensor anomalies.
* A "Model Comparison" tab for side-by-side score comparison across models.
* Explainability integration — SHAP for XGBoost, Captum for the forecasters (LSTM, PatchTST, NHITS) — with plain-language explanations of why a specific moment was flagged.
* A "Data Health" tab for drift monitoring (Population Stability Index) against the training baseline.
* Optional scheduled auto-run, re-scoring automatically on an interval that scales with your data generation rate.
* Supports both local CSV files and a live Kafka topic as the data source.

### 4. Email Alerting Daemon (`alert_daemon.py`)

A persistent background script that monitors the data stream (CSV or Kafka, via `--source csv|kafka`). It uses a "Smart Consensus" mechanism (combining StatDetector and the available forecasters) to prevent false positives. If a confirmed anomaly occurs, it dispatches an email alert to configured stakeholders.

### 5. Docker Containerization (`docker-compose.yml`)

Spins up an Apache Kafka broker in KRaft mode (no Zookeeper required) to handle high-throughput telemetry streams natively.

---

## Prerequisites

Before running the application, ensure you have the following installed:

* **Docker & Docker Compose** (for running Kafka)
* **Python 3.10+**
* **Required Python Packages:** `pip install -r requirements.txt` (Note: install `shap`, `captum`, `darts`, `pytorch-lightning`, and `streamlit-autorefresh` if you want to use the explainability, NHITS, and scheduled auto-run features).

> **Note on Email Alerts:** Before running `alert_daemon.py`, ensure you update the `EMAIL_SENDER`, `EMAIL_PASSWORD` (use an App Password), and `EMAIL_RECEIVER` variables in the script.
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

*To view MLflow training runs, run: `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db` and open `localhost:5000`*

### Step 4: Run the Dashboard

Open the interactive Streamlit dashboard to view incoming data and anomalies:

```bash
streamlit run app/app.py

```

*Once open, go to "Configure & Run", select "Kafka topic" as your source, and click "Run Analysis". Enable "Auto-run" on the same tab if you want the dashboard to re-score itself automatically on a schedule instead of running it manually each time.*

### Step 5: Start the Alert Daemon

To enable automated email alerting for confirmed anomalies:

```bash
python alert_daemon.py --source kafka

```

---

## Monitoring and Diagnostics

To manually check for data drift or diagnose statistical baseline behavior, utilize the dedicated scripts (the same checks are also available directly in the dashboard's "Data Health" tab):

**Data Drift Check:**
Checks whether the current telemetry data has statistically drifted (using Population Stability Index) from the clean baseline established during training.

```bash
python scripts/check_drift.py --source kafka --kafka_topic telemetry.raw --model_dir models

```

**StatDetector Diagnostics:**
Prints StatDetector's calibrated per-channel baselines against a sample of live data, useful for tracing exactly why a channel is or isn't being flagged.

```bash
python scripts/diagnose_stat.py --model_dir models --data_dir live_telemetry_stream

```
