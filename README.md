
# Azercosmos Telemetry ML Pipeline

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
azercosmos_proj/
├── alert_daemon.py          # Background service for triggering email alerts
├── app/
│   └── app.py               # Streamlit application for real-time dashboard and deep dives
├── docker-compose.yml       # Docker orchestration for Kafka (KRaft mode)
├── generate_telemetry.py    # Script to simulate or ingest raw telemetry data with anomalies
├── requirements.txt         # Python dependencies
└── scripts/                 # ML and Data processing scripts
    ├── check_drift.py       # Detects data drift in telemetry against training baseline
    ├── diagnose_stat.py     # Statistical diagnostics for the StatDetector
    ├── drift.py             # Core Population Stability Index (PSI) drift calculation logic
    ├── explain.py           # Model explainability (SHAP & Captum) for XGBoost/Forecasters
    ├── infer.py             # Runs model inference and scoring on incoming data
    ├── kafka_io.py          # Handles Kafka producers/consumers for data streams
    ├── models.py            # ML model definitions (LSTM, PatchTST, XGBoost, IF, StatDetector)
    ├── nhits_model.py       # N-HiTS time-series forecasting model (via Darts)
    └── train.py             # Model training and MLflow logging pipeline

```

---

## Core Components

### 1. Telemetry Data Generation (`generate_telemetry.py`)

This script acts as the entry point for your data stream. It generates simulated real-time telemetry data (e.g., system metrics, sensor readings) and pushes it into the pipeline via local CSV files or Kafka. It features a built-in anomaly injector (`frozen_sensor`, `contextual_break`, `massive_spike`) for testing.

### 2. Machine Learning & MLflow (`scripts/train.py`)

Trains five distinct detectors (StatDetector, LSTM, PatchTST, XGBoost, Isolation Forest) with contamination-robust training. It leverages **MLflow** using a local SQLite backend (`mlruns/mlflow.db`) to track experiments, log metrics, and manage model versions.

### 3. Interactive Web Dashboard (`app/app.py`)

A comprehensive Streamlit interface that provides:

* Real-time scoring and detection overlays.


* A "Deep Dive" tab for investigating specific sensor anomalies.


* Explainability integration (SHAP for XGBoost, Captum for Forecasters) to see *why* models fired.


* Data drift monitoring visualizations.



### 4. Email Alerting Daemon (`alert_daemon.py`)

A persistent background script that monitors the data stream (CSV or Kafka). It uses a "Smart Consensus" mechanism (combining StatDetector and Forecasters) to prevent false positives. If a confirmed anomaly occurs, it dispatches an email alert to configured stakeholders.

### 5. Docker Containerization (`docker-compose.yml`)

Spins up an Apache Kafka broker in KRaft mode (no Zookeeper required) to handle high-throughput telemetry streams natively.

---

## Prerequisites

Before running the application, ensure you have the following installed:

* **Docker & Docker Compose** (for running Kafka)


* **Python 3.8+**

* **Required Python Packages:** `pip install -r requirements.txt` (Note: Install `shap`, `captum`, and `darts` if you want to use the explainability and N-HiTS features).



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
python scripts/train.py --source kafka --trim 0.25

```

*To view MLflow training runs, run: `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db*`

### Step 4: Run the Dashboard

Open the interactive Streamlit dashboard to view incoming data and anomalies:

```bash
streamlit run app/app.py

```

*Once open, go to "Configure & Run", select "Kafka topic" as your source, and click "Run Analysis".*

### Step 5: Start the Alert Daemon

To enable automated email alerting for confirmed anomalies:

```bash
python alert_daemon.py --source kafka

```

---

## Monitoring and Diagnostics

To manually check for data drift or diagnose statistical baseline behavior, utilize the dedicated scripts:

**Data Drift Check:**
Checks if the current telemetry data has statistically drifted (using Population Stability Index) from the clean baseline established during training.

```bash
python scripts/check_drift.py --source kafka

```

**StatDetector Diagnostics:**
Inspects the calibrated baselines (variance, rates, slopes) learned by the statistical detector.

```bash
python scripts/diagnose_stat.py --data_dir live_telemetry_stream

```
