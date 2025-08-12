#!/usr/bin/env python3
"""Recursive Unity Convergence Spawner
===================================

Spawn subprocesses recursively to demonstrate unity convergence using the
core UnityMathematics engine. Each process computes `1 + 1 = 1` and
optionally spawns child processes until the specified depth is reached.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = Path(__file__).resolve()


def run_unity_operation() -> None:
    """Perform a unity addition using the core engine."""
    from src.core.unity_mathematics import UnityMathematics

    unity = UnityMathematics()
    result = unity.unity_add(1.0, 1.0)
    print(f"[PID {os.getpid()}] unity_add result: {result.value:.6f}")


def spawn_children(depth: int, branching: int) -> List[subprocess.Popen]:
    """Spawn child processes to continue the recursion."""
    children: List[subprocess.Popen] = []
    for _ in range(branching):
        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "--depth",
            str(depth - 1),
            "--branching",
            str(branching),
        ]
        proc = subprocess.Popen(
            cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        children.append(proc)
    return children


def run(depth: int, branching: int) -> None:
    """Run the recursive convergence routine."""
    if depth <= 0:
        run_unity_operation()
        return

    children = spawn_children(depth, branching)
    for proc in children:
        out, err = proc.communicate()
        if out:
            print(out.strip())
        if err:
            print(err.strip(), file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recursively spawn unity convergence processes.")
    parser.add_argument("--depth", type=int, default=1, help="Recursion depth")
    parser.add_argument("--branching", type=int, default=1, help="Processes to spawn per level")
    args = parser.parse_args()
    run(args.depth, args.branching)


if __name__ == "__main__":
    main()
