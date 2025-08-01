#!/usr/bin/env python3
"""Meta-Agent Background Launcher
================================

Utility script to spawn multiple UnityMetaAgent processes in the
background. Each process executes the demonstration function from
``ml_framework.meta_reinforcement.unity_meta_agent``. Processes log their
output to the ``logs/`` directory and can be run for a limited
``--duration``.

This is a lightweight example and not intended for perpetual execution.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "meta_agent"
LOG_DIR.mkdir(parents=True, exist_ok=True)

META_AGENT_CMD = [
    sys.executable,
    "-m",
    "ml_framework.meta_reinforcement.unity_meta_agent",
]


def start_process(index: int) -> subprocess.Popen:
    """Start a single meta-agent subprocess."""
    stdout_path = LOG_DIR / f"meta_agent_{index}_stdout.log"
    stderr_path = LOG_DIR / f"meta_agent_{index}_stderr.log"
    stdout_file = open(stdout_path, "a")
    stderr_file = open(stderr_path, "a")
    process = subprocess.Popen(
        META_AGENT_CMD,
        cwd=PROJECT_ROOT,
        stdout=stdout_file,
        stderr=stderr_file,
        text=True,
        preexec_fn=os.setsid if os.name != "nt" else None,
    )
    return process


def stop_process(process: subprocess.Popen) -> None:
    """Terminate a subprocess gracefully."""
    if process.poll() is None:
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=10)
        except Exception:
            if os.name != "nt":
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()


def run_background(num_processes: int, duration: int) -> None:
    """Launch multiple meta-agent processes for a given duration."""
    processes: List[subprocess.Popen] = []
    for i in range(num_processes):
        proc = start_process(i)
        processes.append(proc)
        print(f"Started meta-agent #{i} (PID {proc.pid})")

    start_time = time.time()
    try:
        while True:
            time.sleep(1)
            if duration and (time.time() - start_time) > duration:
                print("Duration reached, stopping meta agents.")
                break
    except KeyboardInterrupt:
        print("Interrupted by user, stopping meta agents.")
    finally:
        for proc in processes:
            stop_process(proc)
        print("All meta agents stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch background UnityMetaAgent processes")
    parser.add_argument("--processes", type=int, default=2, help="Number of processes to launch")
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration to run in seconds (0 for no limit)",
    )
    args = parser.parse_args()
    run_background(args.processes, args.duration)


if __name__ == "__main__":
    main()
