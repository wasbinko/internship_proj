import sys, os, subprocess, threading, queue

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
except ImportError:
    print("ERROR: tkinter is not available in this Python installation.")
    print("On Windows, tkinter ships with the standard python.org installer --")
    print("if this is missing, reinstall Python and make sure 'tcl/tk and IDLE' "
          "is checked during setup.")
    sys.exit(1)


MAX_LOG_LINES = 2000

MODEL_OPTIONS = [
    ("stat",     "StatDetector",           False),
    ("lstm",     "LSTM",                   False),
    ("patchtst", "PatchTST",               False),
    ("xgboost",  "XGBoost",                False),
    ("iforest",  "Isolation Forest",       False),
    ("nhits",    "NHITS (Darts, slower)",  False),
]


def build_train_command(python_exe: str, script_path: str, selected_models: list[str],
                        source: str, csv_dir: str, kafka_topic: str,
                        model_dir: str) -> list[str] | None:
    if not selected_models:
        return None
    cmd = [python_exe, "-u", script_path,
            "--model_dir", model_dir,
            "--models"] + selected_models
    cmd += ["--source", source]
    if source == "csv":
        cmd += ["--data_dir", csv_dir]
    else:
        cmd += ["--kafka_topic", kafka_topic]
    return cmd


def build_daemon_command(python_exe: str, script_path: str, source: str,
                         kafka_bootstrap: str, kafka_topic: str, kafka_group: str,
                         email_receiver: str = "") -> list[str]:
    cmd = [python_exe, "-u", script_path, "--source", source]
    if source == "kafka":
        cmd += ["--kafka_bootstrap", kafka_bootstrap,
                "--kafka_topic", kafka_topic,
                "--kafka_group", kafka_group]
    if email_receiver.strip():
        cmd += ["--email_receiver", email_receiver.strip()]
    return cmd


def build_generator_command(python_exe: str, script_path: str, sink: str,
                            kafka_bootstrap: str, kafka_topic: str,
                            anomaly_probability: float, interval: int) -> list[str]:
    cmd = [python_exe, "-u", script_path, "--sink", sink,
           "--anomaly_probability", str(anomaly_probability),
           "--interval", str(interval)]
    if sink in ("kafka", "both"):
        cmd += ["--kafka_bootstrap", kafka_bootstrap, "--kafka_topic", kafka_topic]
    return cmd


class TrainingTab:
    def __init__(self, notebook, project_root):
        self.frame = ttk.Frame(notebook)
        self.project_root = project_root
        self.model_vars = {}
        self.log_queue = queue.Queue()
        self.process = None
        self._build_ui()
        self._poll_queue()

    def _build_ui(self):
        model_frame = ttk.LabelFrame(self.frame, text="Models to Train", padding=10)
        model_frame.pack(fill="x", padx=10, pady=(10, 5))
        for i, (key, label, default) in enumerate(MODEL_OPTIONS):
            var = tk.BooleanVar(value=default)
            self.model_vars[key] = var
            ttk.Checkbutton(model_frame, text=label, variable=var).grid(
                row=i // 2, column=i % 2, sticky="w", padx=8, pady=3)

        source_frame = ttk.LabelFrame(self.frame, text="Data Source", padding=10)
        source_frame.pack(fill="x", padx=10, pady=5)

        self.source_var = tk.StringVar(value="kafka")
        ttk.Radiobutton(source_frame, text="Kafka", variable=self.source_var,
                       value="kafka", command=self._toggle_source).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(source_frame, text="Local CSV files", variable=self.source_var,
                       value="csv", command=self._toggle_source).grid(row=0, column=1, sticky="w")

        ttk.Label(source_frame, text="CSV folder:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.csv_dir_var = tk.StringVar(value="live_telemetry_stream")
        self.csv_entry = ttk.Entry(source_frame, textvariable=self.csv_dir_var, width=42, state="disabled")
        self.csv_entry.grid(row=1, column=1, sticky="w", pady=(8, 0))

        ttk.Label(source_frame, text="Kafka topic:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.kafka_topic_var = tk.StringVar(value="telemetry.raw")
        self.kafka_entry = ttk.Entry(source_frame, textvariable=self.kafka_topic_var, width=42)
        self.kafka_entry.grid(row=2, column=1, sticky="w", pady=(4, 0))

        opts_frame = ttk.LabelFrame(self.frame, text="Options", padding=10)
        opts_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(opts_frame, text="Model output folder:").grid(row=0, column=0, sticky="w")
        self.model_dir_var = tk.StringVar(value="models")
        ttk.Entry(opts_frame, textvariable=self.model_dir_var, width=30).grid(row=0, column=1, sticky="w")

        self.train_button = ttk.Button(self.frame, text="Train Models", command=self._start_training)
        self.train_button.pack(pady=8)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.frame, textvariable=self.status_var, font=("Segoe UI", 10, "bold")).pack()

        log_frame = ttk.LabelFrame(self.frame, text="Training Log", padding=5)
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=15, state="disabled", font=("Consolas", 9), wrap="word")
        self.log_text.pack(fill="both", expand=True)

    def _toggle_source(self):
        if self.source_var.get() == "csv":
            self.csv_entry.config(state="normal")
            self.kafka_entry.config(state="disabled")
        else:
            self.csv_entry.config(state="disabled")
            self.kafka_entry.config(state="normal")

    def _start_training(self):
        selected = [k for k, v in self.model_vars.items() if v.get()]
        cmd = build_train_command(
            python_exe=sys.executable,
            script_path=os.path.join(self.project_root, "scripts", "train.py"),
            selected_models=selected,
            source=self.source_var.get(),
            csv_dir=self.csv_dir_var.get(),
            kafka_topic=self.kafka_topic_var.get(),
            model_dir=self.model_dir_var.get(),
        )
        if cmd is None:
            messagebox.showwarning("No models selected", "Select at least one model to train.")
            return

        self.train_button.config(state="disabled")
        self.status_var.set("Training in progress...")
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, f"Running: {' '.join(cmd)}\n\n")
        self.log_text.config(state="disabled")

        threading.Thread(target=self._run_training, args=(cmd,), daemon=True).start()

    def _run_training(self, cmd):
        try:
            child_env = os.environ.copy()
            child_env["PYTHONIOENCODING"] = "utf-8"
            n_cores = str(os.cpu_count() or 4)
            child_env["OMP_NUM_THREADS"] = n_cores
            child_env["MKL_NUM_THREADS"] = n_cores
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=self.project_root,
                encoding="utf-8", errors="replace", env=child_env,
            )
            for line in self.process.stdout:
                self.log_queue.put(("line", line))
            self.process.wait()
            self.log_queue.put(("done", self.process.returncode == 0))
        except Exception as e:
            self.log_queue.put(("error", str(e)))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "line":
                    self.log_text.config(state="normal")
                    self.log_text.insert(tk.END, payload)
                    self.log_text.see(tk.END)
                    self.log_text.config(state="disabled")
                elif kind == "done":
                    self.train_button.config(state="normal")
                    if payload:
                        self.status_var.set("\u2713 Training complete - results logged to MLflow")
                        messagebox.showinfo(
                            "Training Complete",
                            "All selected models finished training successfully.\n\n"
                            "Results have been logged to MLflow.\n"
                            "Run this to view them:\n"
                            "mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db\n\n"
                        )
                    else:
                        self.status_var.set("\u2717 Training failed - see log above")
                        messagebox.showerror(
                            "Training Failed",
                            "Training exited with an error. Check the log panel for details."
                        )
                elif kind == "error":
                    self.train_button.config(state="normal")
                    self.status_var.set("\u2717 Error")
                    messagebox.showerror("Error", f"Could not start training:\n{payload}")
        except queue.Empty:
            pass
        self.frame.after(100, self._poll_queue)


class DaemonTab:
    def __init__(self, notebook, project_root):
        self.frame = ttk.Frame(notebook)
        self.project_root = project_root
        self.log_queue = queue.Queue()
        self.process = None
        self._log_line_count = 0
        self._build_ui()
        self._poll_queue()

    def _build_ui(self):
        source_frame = ttk.LabelFrame(self.frame, text="Data Source", padding=10)
        source_frame.pack(fill="x", padx=10, pady=(10, 5))

        self.source_var = tk.StringVar(value="kafka")
        ttk.Radiobutton(source_frame, text="Kafka", variable=self.source_var,
                       value="kafka", command=self._toggle_source).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(source_frame, text="Local CSV files", variable=self.source_var,
                       value="csv", command=self._toggle_source).grid(row=0, column=1, sticky="w")

        ttk.Label(source_frame, text="Kafka bootstrap:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.kafka_bootstrap_var = tk.StringVar(value="localhost:9092")
        self.kafka_bootstrap_entry = ttk.Entry(source_frame, textvariable=self.kafka_bootstrap_var, width=30)
        self.kafka_bootstrap_entry.grid(row=1, column=1, sticky="w", pady=(8, 0))

        ttk.Label(source_frame, text="Kafka topic:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.kafka_topic_var = tk.StringVar(value="telemetry.raw")
        self.kafka_topic_entry = ttk.Entry(source_frame, textvariable=self.kafka_topic_var, width=30)
        self.kafka_topic_entry.grid(row=2, column=1, sticky="w", pady=(4, 0))

        ttk.Label(source_frame, text="Kafka group:").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.kafka_group_var = tk.StringVar(value="alert-daemon")
        self.kafka_group_entry = ttk.Entry(source_frame, textvariable=self.kafka_group_var, width=30)
        self.kafka_group_entry.grid(row=3, column=1, sticky="w", pady=(4, 0))

        ttk.Label(source_frame, text="Send alerts to:").grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.email_receiver_var = tk.StringVar(value="")
        ttk.Entry(source_frame, textvariable=self.email_receiver_var, width=30).grid(
            row=4, column=1, sticky="w", pady=(8, 0))

        note = "Leave \"Send alerts to\" blank to use the default receiver set inside alert_daemon.py."
        ttk.Label(source_frame, text=note, wraplength=650, foreground="#555").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(10, 0))

        button_frame = ttk.Frame(self.frame)
        button_frame.pack(pady=8)
        self.start_button = ttk.Button(button_frame, text="Start Daemon", command=self._start_daemon)
        self.start_button.grid(row=0, column=0, padx=5)
        self.stop_button = ttk.Button(button_frame, text="Stop Daemon",
                                      command=self._stop_daemon, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=5)

        self.status_var = tk.StringVar(value="\u25cf Stopped")
        self.status_label = ttk.Label(self.frame, textvariable=self.status_var,
                                      font=("Segoe UI", 11, "bold"), foreground="#888888")
        self.status_label.pack()

        log_frame = ttk.LabelFrame(self.frame, text="Daemon Log", padding=5)
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=18, state="disabled", font=("Consolas", 9), wrap="word")
        self.log_text.pack(fill="both", expand=True)

    def _toggle_source(self):
        state = "normal" if self.source_var.get() == "kafka" else "disabled"
        self.kafka_bootstrap_entry.config(state=state)
        self.kafka_topic_entry.config(state=state)
        self.kafka_group_entry.config(state=state)

    def _start_daemon(self):
        cmd = build_daemon_command(
            python_exe=sys.executable,
            script_path=os.path.join(self.project_root, "alert_daemon.py"),
            source=self.source_var.get(),
            kafka_bootstrap=self.kafka_bootstrap_var.get(),
            kafka_topic=self.kafka_topic_var.get(),
            kafka_group=self.kafka_group_var.get(),
            email_receiver=self.email_receiver_var.get(),
        )
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_var.set("\u25cf Running")
        self.status_label.config(foreground="#1a7a3c")
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, f"Running: {' '.join(cmd)}\n\n")
        self.log_text.config(state="disabled")
        self._log_line_count = 0

        threading.Thread(target=self._run_daemon, args=(cmd,), daemon=True).start()

    def _run_daemon(self, cmd):
        try:
            child_env = os.environ.copy()
            child_env["PYTHONIOENCODING"] = "utf-8"
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=self.project_root,
                encoding="utf-8", errors="replace", env=child_env,
            )
            for line in self.process.stdout:
                self.log_queue.put(("line", line))
            self.process.wait()
            self.log_queue.put(("stopped", self.process.returncode))
        except Exception as e:
            self.log_queue.put(("error", str(e)))

    def _stop_daemon(self):
        self.stop_button.config(state="disabled")
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "line":
                    self.log_text.config(state="normal")
                    self.log_text.insert(tk.END, payload)
                    self._log_line_count += 1
                    if self._log_line_count > MAX_LOG_LINES:
                        self.log_text.delete("1.0", "500.0")
                        self._log_line_count -= 500
                    self.log_text.see(tk.END)
                    self.log_text.config(state="disabled")
                elif kind == "stopped":
                    self.start_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.status_var.set("\u25cf Stopped" if payload in (0, None)
                                       else f"\u25cf Stopped (exit code {payload})")
                    self.status_label.config(foreground="#888888")
                elif kind == "error":
                    self.start_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.status_var.set("\u25cf Error")
                    self.status_label.config(foreground="#c0392b")
                    messagebox.showerror("Error", f"Could not start daemon:\n{payload}")
        except queue.Empty:
            pass
        self.frame.after(100, self._poll_queue)


class GeneratorTab:
    def __init__(self, notebook, project_root):
        self.frame = ttk.Frame(notebook)
        self.project_root = project_root
        self.log_queue = queue.Queue()
        self.process = None
        self._log_line_count = 0
        self._build_ui()
        self._poll_queue()

    def _build_ui(self):
        sink_frame = ttk.LabelFrame(self.frame, text="Output", padding=10)
        sink_frame.pack(fill="x", padx=10, pady=(10, 5))

        self.sink_var = tk.StringVar(value="kafka")
        for i, val in enumerate(["csv", "kafka", "both"]):
            ttk.Radiobutton(sink_frame, text=val.upper(), variable=self.sink_var,
                           value=val, command=self._toggle_kafka).grid(row=0, column=i, sticky="w", padx=5)

        ttk.Label(sink_frame, text="Kafka bootstrap:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.kafka_bootstrap_var = tk.StringVar(value="localhost:9092")
        self.kafka_bootstrap_entry = ttk.Entry(sink_frame, textvariable=self.kafka_bootstrap_var, width=30)
        self.kafka_bootstrap_entry.grid(row=1, column=1, columnspan=2, sticky="w", pady=(8, 0))

        ttk.Label(sink_frame, text="Kafka topic:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.kafka_topic_var = tk.StringVar(value="telemetry.raw")
        self.kafka_topic_entry = ttk.Entry(sink_frame, textvariable=self.kafka_topic_var, width=30)
        self.kafka_topic_entry.grid(row=2, column=1, columnspan=2, sticky="w", pady=(4, 0))

        opts_frame = ttk.LabelFrame(self.frame, text="Options", padding=10)
        opts_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(opts_frame, text="Anomaly probability (0.0 - 1.0):").grid(row=0, column=0, sticky="w")
        self.anomaly_var = tk.DoubleVar(value=0.25)
        ttk.Spinbox(opts_frame, from_=0.0, to=1.0, increment=0.05,
                   textvariable=self.anomaly_var, width=10).grid(row=0, column=1, sticky="w")

        ttk.Label(opts_frame, text="Interval in seconds (300 = 5 min; try 5 for fast testing):").grid(
            row=1, column=0, sticky="w", pady=(8, 0))
        self.interval_var = tk.IntVar(value=300)
        ttk.Spinbox(opts_frame, from_=1, to=3600, increment=5,
                   textvariable=self.interval_var, width=10).grid(row=1, column=1, sticky="w", pady=(8, 0))

        button_frame = ttk.Frame(self.frame)
        button_frame.pack(pady=8)
        self.start_button = ttk.Button(button_frame, text="Start Generating", command=self._start_generator)
        self.start_button.grid(row=0, column=0, padx=5)
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self._stop_generator, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=5)

        self.status_var = tk.StringVar(value="\u25cf Stopped")
        self.status_label = ttk.Label(self.frame, textvariable=self.status_var,
                                      font=("Segoe UI", 11, "bold"), foreground="#888888")
        self.status_label.pack()

        log_frame = ttk.LabelFrame(self.frame, text="Generator Log", padding=5)
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=16, state="disabled", font=("Consolas", 9), wrap="word")
        self.log_text.pack(fill="both", expand=True)

    def _toggle_kafka(self):
        state = "normal" if self.sink_var.get() in ("kafka", "both") else "disabled"
        self.kafka_bootstrap_entry.config(state=state)
        self.kafka_topic_entry.config(state=state)

    def _start_generator(self):
        cmd = build_generator_command(
            python_exe=sys.executable,
            script_path=os.path.join(self.project_root, "generate_telemetry.py"),
            sink=self.sink_var.get(),
            kafka_bootstrap=self.kafka_bootstrap_var.get(),
            kafka_topic=self.kafka_topic_var.get(),
            anomaly_probability=self.anomaly_var.get(),
            interval=self.interval_var.get(),
        )
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_var.set("\u25cf Running")
        self.status_label.config(foreground="#1a7a3c")
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, f"Running: {' '.join(cmd)}\n\n")
        self.log_text.config(state="disabled")
        self._log_line_count = 0

        threading.Thread(target=self._run_generator, args=(cmd,), daemon=True).start()

    def _run_generator(self, cmd):
        try:
            child_env = os.environ.copy()
            child_env["PYTHONIOENCODING"] = "utf-8"
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=self.project_root,
                encoding="utf-8", errors="replace", env=child_env,
            )
            for line in self.process.stdout:
                self.log_queue.put(("line", line))
            self.process.wait()
            self.log_queue.put(("stopped", self.process.returncode))
        except Exception as e:
            self.log_queue.put(("error", str(e)))

    def _stop_generator(self):
        self.stop_button.config(state="disabled")
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "line":
                    self.log_text.config(state="normal")
                    self.log_text.insert(tk.END, payload)
                    self._log_line_count += 1
                    if self._log_line_count > MAX_LOG_LINES:
                        self.log_text.delete("1.0", "500.0")
                        self._log_line_count -= 500
                    self.log_text.see(tk.END)
                    self.log_text.config(state="disabled")
                elif kind == "stopped":
                    self.start_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.status_var.set("\u25cf Stopped" if payload in (0, None)
                                       else f"\u25cf Stopped (exit code {payload})")
                    self.status_label.config(foreground="#888888")
                elif kind == "error":
                    self.start_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.status_var.set("\u25cf Error")
                    self.status_label.config(foreground="#c0392b")
                    messagebox.showerror("Error", f"Could not start generator:\n{payload}")
        except queue.Empty:
            pass
        self.frame.after(100, self._poll_queue)


class CombinedGUI:
    def __init__(self, root):
        self.root = root
        root.title("Project Control Panel")
        root.geometry("760x680")
        root.minsize(640, 540)

        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)

        self.generator_tab = GeneratorTab(notebook, self.project_root)
        self.training_tab = TrainingTab(notebook, self.project_root)
        self.daemon_tab = DaemonTab(notebook, self.project_root)
        notebook.add(self.generator_tab.frame, text="Generate Telemetry")
        notebook.add(self.training_tab.frame, text="Train Models")
        notebook.add(self.daemon_tab.frame, text="Alert Daemon")

        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        running = []
        if self.generator_tab.is_running():
            running.append("telemetry generator")
        if self.daemon_tab.is_running():
            running.append("alert daemon")
        if running:
            if not messagebox.askyesno(
                "Still running",
                f"The {' and '.join(running)} {'is' if len(running) == 1 else 'are'} "
                f"still running. Stop and quit?"):
                return
            if self.generator_tab.is_running():
                self.generator_tab._stop_generator()
            if self.daemon_tab.is_running():
                self.daemon_tab._stop_daemon()
        if self.training_tab.process and self.training_tab.process.poll() is None:
            self.training_tab.process.terminate()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CombinedGUI(root)
    root.mainloop()
