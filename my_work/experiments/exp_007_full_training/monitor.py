"""Logging-only instrumentation around the unchanged community training entry point."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def process_memory_mib(pid):
    """Sample this process only, including CUDA allocations outside PyTorch (e.g. Warp)."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    if result.returncode:
        return None
    amounts = []
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid) and parts[1].isdigit():
            amounts.append(int(parts[1]))
    return sum(amounts) if amounts else None


class TrainingMonitor:
    """Change event flushing only; do not wrap losses, optimizers or model forward calls."""

    def __init__(self, run_dir, sample_interval=30, sample_memory=True):
        self.run_dir = Path(run_dir)
        self.sample_interval = sample_interval
        self.sample_memory = sample_memory
        self.writers = []
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = None
        self.started = time.monotonic()
        self.state = {
            "status": "running",
            "pid": os.getpid(),
            "started_at_utc": datetime.now(UTC).isoformat(),
            "event_flush_seconds": 15,
            "gpu_sample_interval_seconds": sample_interval,
            "gpu_sampled_process_peak_mib": None,
            "gpu_torch_peak_allocated_mib": None,
            "gpu_torch_peak_reserved_mib": None,
            "nonfinite_scalar_count": 0,
            "last_logged_iteration": None,
            "last_logged_env_steps": None,
            "last_logged_policy_updates": None,
        }

    def persist(self):
        with self.lock:
            value = dict(self.state)
            value["heartbeat_at_utc"] = datetime.now(UTC).isoformat()
            value["process_wall_seconds"] = time.monotonic() - self.started
            atomic_json(self.run_dir / "monitor_status.json", value)

    def sample(self):
        try:
            amount = process_memory_mib(os.getpid())
            if amount is not None:
                with self.lock:
                    previous = self.state["gpu_sampled_process_peak_mib"]
                    self.state["gpu_sampled_process_peak_mib"] = max(previous or 0, amount)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            with self.lock:
                self.state["gpu_sampling_error"] = type(exc).__name__

    def sampling_loop(self):
        while not self.stop_event.is_set():
            self.sample()
            self.persist()
            self.stop_event.wait(self.sample_interval)

    def observe(self, tag, value, step):
        value = float(value)
        if not math.isfinite(value):
            with self.lock:
                self.state["nonfinite_scalar_count"] += 1
            folder = self.run_dir / "logs"
            folder.mkdir(exist_ok=True)
            # Only anomalies go here. Normal scalar samples stay in native TensorBoard files.
            with (folder / "anomalies.jsonl").open("a") as log:
                log.write(
                    json.dumps(
                        {
                            "tag": tag,
                            "step": step,
                            "value": str(value),
                            "time_utc": datetime.now(UTC).isoformat(),
                        }
                    )
                    + "\n"
                )
                log.flush()
                os.fsync(log.fileno())
        if tag == "time/env_steps":
            self.state["last_logged_env_steps"] = int(value) if math.isfinite(value) else None
        elif tag == "time/policy_updates":
            self.state["last_logged_policy_updates"] = int(value) if math.isfinite(value) else None
            self.state["last_logged_iteration"] = step
            self.persist()

    def __enter__(self):
        import torch
        from torch.utils import tensorboard

        self.original_writer = tensorboard.SummaryWriter
        self.existing_loop = sys.modules.get("supermimic.training.loop")
        self.existing_loop_writer = getattr(self.existing_loop, "SummaryWriter", None)
        monitor = self

        class ObservedWriter(monitor.original_writer):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("flush_secs", 15)
                super().__init__(*args, **kwargs)
                monitor.writers.append(self)

            def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, **kwargs):
                super().add_scalar(tag, scalar_value, global_step, walltime, **kwargs)
                monitor.observe(tag, scalar_value, global_step)
                if tag == "time/hours" and torch.cuda.is_initialized():
                    allocated = torch.cuda.max_memory_allocated() / 1024**2
                    reserved = torch.cuda.max_memory_reserved() / 1024**2
                    monitor.state["gpu_torch_peak_allocated_mib"] = allocated
                    monitor.state["gpu_torch_peak_reserved_mib"] = reserved
                    # These are genuinely new metrics, not copies of existing training scalars.
                    super().add_scalar("gpu/torch_peak_allocated_mib", allocated, global_step)
                    super().add_scalar("gpu/torch_peak_reserved_mib", reserved, global_step)

        tensorboard.SummaryWriter = ObservedWriter
        if self.existing_loop is not None:
            self.existing_loop.SummaryWriter = ObservedWriter
        self.persist()
        if self.sample_memory:
            self.thread = threading.Thread(target=self.sampling_loop, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, kind, value, traceback):
        import torch
        from torch.utils import tensorboard

        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=6)
            self.sample()
        for writer in self.writers:
            writer.flush()
            writer.close()
        if torch.cuda.is_initialized():
            self.state["gpu_torch_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
            self.state["gpu_torch_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
        self.state["status"] = (
            "completed"
            if kind is None
            else ("interrupted" if issubclass(kind, (KeyboardInterrupt, SystemExit)) else "failed")
        )
        self.state["finished_at_utc"] = datetime.now(UTC).isoformat()
        if kind is not None:
            self.state["exception_type"] = kind.__name__
        self.persist()
        tensorboard.SummaryWriter = self.original_writer
        loop = sys.modules.get("supermimic.training.loop")
        if loop is not None:
            loop.SummaryWriter = self.existing_loop_writer or self.original_writer
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("community_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    community_args = args.community_args
    if community_args[:1] == ["--"]:
        community_args = community_args[1:]
    if not community_args or community_args[0] != "release":
        parser.error("Expected the unchanged release CLI after --")
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        with TrainingMonitor(args.run_dir):
            # Logging adapter installed before Trainer binds its SummaryWriter import.
            from supermimic.scripts.train import main as train

            sys.argv = ["sm-train", *community_args]
            train()
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
