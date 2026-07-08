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


MAX_LOG_LINES = 2000   # trim old lines past this so a long-running daemon
                       # session doesn't grow the log panel unbounded


def build_daemon_command(python_exe: str, script_path: str, source: str,
                         kafka_bootstrap: str, kafka_topic: str, kafka_group: str) -> list[str]:
    cmd = [python_exe, "-u", script_path, "--source", source]
    if source == "kafka":
        cmd += ["--kafka_bootstrap", kafka_bootstrap,
                "--kafka_topic", kafka_topic,
                "--kafka_group", kafka_group]
    return cmd


class DaemonGUI:
    def __init__(self, root):
        self.root = root
        root.title("Alert Daemon Control")
        root.geometry("720x580")
        root.minsize(600, 460)

        self.log_queue = queue.Queue()
        self.process = None
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self._log_line_count = 0

        self._build_ui()
        self._poll_queue()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        source_frame = ttk.LabelFrame(self.root, text="Data Source", padding=10)
        source_frame.pack(fill="x", padx=10, pady=(10, 5))

        self.source_var = tk.StringVar(value="csv")
        ttk.Radiobutton(source_frame, text="Local CSV files", variable=self.source_var,
                       value="csv", command=self._toggle_source).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(source_frame, text="Kafka", variable=self.source_var,
                       value="kafka", command=self._toggle_source).grid(row=0, column=1, sticky="w")

        ttk.Label(source_frame, text="Kafka bootstrap:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.kafka_bootstrap_var = tk.StringVar(value="localhost:9092")
        self.kafka_bootstrap_entry = ttk.Entry(source_frame, textvariable=self.kafka_bootstrap_var,
                                               width=30, state="disabled")
        self.kafka_bootstrap_entry.grid(row=1, column=1, sticky="w", pady=(8, 0))

        ttk.Label(source_frame, text="Kafka topic:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.kafka_topic_var = tk.StringVar(value="telemetry.raw")
        self.kafka_topic_entry = ttk.Entry(source_frame, textvariable=self.kafka_topic_var,
                                          width=30, state="disabled")
        self.kafka_topic_entry.grid(row=2, column=1, sticky="w", pady=(4, 0))

        ttk.Label(source_frame, text="Kafka group:").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.kafka_group_var = tk.StringVar(value="alert-daemon")
        self.kafka_group_entry = ttk.Entry(source_frame, textvariable=self.kafka_group_var,
                                          width=30, state="disabled")
        self.kafka_group_entry.grid(row=3, column=1, sticky="w", pady=(4, 0))

        note = ("Data folder, model folder, and email sender are set as constants inside "
                "alert_daemon.py itself, not here — edit that file directly to change them.")
        ttk.Label(source_frame, text=note, wraplength=650, foreground="#555").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(10, 0))

        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=8)
        self.start_button = ttk.Button(button_frame, text="Start Daemon", command=self._start_daemon)
        self.start_button.grid(row=0, column=0, padx=5)
        self.stop_button = ttk.Button(button_frame, text="Stop Daemon",
                                      command=self._stop_daemon, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=5)

        self.status_var = tk.StringVar(value="\u25cf Stopped")
        self.status_label = ttk.Label(self.root, textvariable=self.status_var,
                                      font=("Segoe UI", 11, "bold"), foreground="#888888")
        self.status_label.pack()

        log_frame = ttk.LabelFrame(self.root, text="Daemon Log", padding=5)
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

    def _on_close(self):
        if self.process and self.process.poll() is None:
            if messagebox.askyesno("Daemon still running",
                                   "The alert daemon is still running. Stop it and quit?"):
                self._stop_daemon()
                self.root.destroy()
            # else: cancel the close, leave the window open
        else:
            self.root.destroy()

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
                    if payload in (0, None):
                        self.status_var.set("\u25cf Stopped")
                    else:
                        self.status_var.set(f"\u25cf Stopped (exit code {payload})")
                    self.status_label.config(foreground="#888888")
                elif kind == "error":
                    self.start_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.status_var.set("\u25cf Error")
                    self.status_label.config(foreground="#c0392b")
                    messagebox.showerror("Error", f"Could not start daemon:\n{payload}")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)


if __name__ == "__main__":
    root = tk.Tk()
    app = DaemonGUI(root)
    root.mainloop()
