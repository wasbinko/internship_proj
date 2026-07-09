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
                        model_dir: str, trim: float) -> list[str] | None:
    if not selected_models:
        return None
    cmd = [python_exe, "-u", script_path,
           "--model_dir", model_dir,
           "--trim", str(trim),
           "--models"] + selected_models
    cmd += ["--source", source]
    if source == "csv":
        cmd += ["--data_dir", csv_dir]
    else:
        cmd += ["--kafka_topic", kafka_topic]
    return cmd


class TrainingGUI:
    def __init__(self, root):
        self.root = root
        root.title("Train Anomaly Detection Models")
        root.geometry("720x640")
        root.minsize(600, 500)

        self.model_vars = {}
        self.log_queue = queue.Queue()
        self.process = None
        self.project_root = os.path.dirname(os.path.abspath(__file__))

        self._build_ui()
        self._poll_queue()

    def _build_ui(self):
        model_frame = ttk.LabelFrame(self.root, text="Models to Train", padding=10)
        model_frame.pack(fill="x", padx=10, pady=(10, 5))
        for i, (key, label, default) in enumerate(MODEL_OPTIONS):
            var = tk.BooleanVar(value=default)
            self.model_vars[key] = var
            ttk.Checkbutton(model_frame, text=label, variable=var).grid(
                row=i // 2, column=i % 2, sticky="w", padx=8, pady=3)

        source_frame = ttk.LabelFrame(self.root, text="Data Source", padding=10)
        source_frame.pack(fill="x", padx=10, pady=5)

        
        self.source_var = tk.StringVar(value="kafka")
        ttk.Radiobutton(source_frame, text="Kafka", variable=self.source_var,
                       value="kafka", command=self._toggle_source).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(source_frame, text="Local CSV files", variable=self.source_var,
                       value="csv", command=self._toggle_source).grid(row=0, column=1, sticky="w")

        ttk.Label(source_frame, text="CSV folder:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.csv_dir_var = tk.StringVar(value="live_telemetry_stream")
        self.csv_entry = ttk.Entry(source_frame, textvariable=self.csv_dir_var, width=42)
        self.csv_entry.grid(row=1, column=1, sticky="w", pady=(8, 0))

        ttk.Label(source_frame, text="Kafka topic:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.kafka_topic_var = tk.StringVar(value="telemetry.raw")
        self.kafka_entry = ttk.Entry(source_frame, textvariable=self.kafka_topic_var,
                                    width=42, state="disabled")
        self.kafka_entry.grid(row=2, column=1, sticky="w", pady=(4, 0))

        opts_frame = ttk.LabelFrame(self.root, text="Options", padding=10)
        opts_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(opts_frame, text="Model output folder:").grid(row=0, column=0, sticky="w")
        self.model_dir_var = tk.StringVar(value="models")
        ttk.Entry(opts_frame, textvariable=self.model_dir_var, width=30).grid(
            row=0, column=1, sticky="w")

        ttk.Label(opts_frame, text="Contamination trim (0.0 - 0.5):").grid(
            row=1, column=0, sticky="w", pady=(8, 0))
        self.trim_var = tk.DoubleVar(value=0.25)
        ttk.Spinbox(opts_frame, from_=0.0, to=0.5, increment=0.05,
                   textvariable=self.trim_var, width=10).grid(row=1, column=1, sticky="w", pady=(8, 0))

        self.train_button = ttk.Button(self.root, text="Train Models", command=self._start_training)
        self.train_button.pack(pady=8)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var,
                 font=("Segoe UI", 10, "bold")).pack()

        log_frame = ttk.LabelFrame(self.root, text="Training Log", padding=5)
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
            trim=self.trim_var.get(),
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
        self.root.after(100, self._poll_queue)


if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingGUI(root)
    root.mainloop()
