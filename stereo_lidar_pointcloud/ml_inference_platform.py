"""Shared conda-run helpers for optional GPU inference envs (Depth Anything V2, etc.)."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys


def is_macos() -> bool:
    return platform.system() == "Darwin"


def is_windows() -> bool:
    return platform.system() == "Windows"


def conda_env_exists(env_name: str) -> bool:
    if shutil.which("conda") is None:
        return False
    result = subprocess.run(
        ["conda", "env", "list"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    for line in result.stdout.splitlines():
        if line.split()[:1] == [env_name]:
            return True
    return False


def conda_run_python(env_name: str, code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["conda", "run", "-n", env_name, "--no-capture-output", "python", "-c", code],
        capture_output=True,
        text=True,
    )


def dav2_env_ready(env_name: str = "depth_anything_v2") -> bool:
    if not conda_env_exists(env_name):
        return False
    probe = conda_run_python(env_name, "import torch; raise SystemExit(0)")
    return probe.returncode == 0


def dav2_setup_command() -> str:
    if is_windows():
        return r"powershell -ExecutionPolicy Bypass -File scripts\setup_depth_anything_v2.ps1"
    return "bash scripts/setup_depth_anything_v2.sh"


def require_dav2_or_exit(env_name: str = "depth_anything_v2") -> None:
    if shutil.which("conda") is None:
        print("conda not found. Install Miniconda, then run:")
        print(f"  {dav2_setup_command()}")
        sys.exit(2)
    if not dav2_env_ready(env_name):
        print(f"Depth Anything V2 env '{env_name}' is not ready.")
        print(f"Setup:\n  {dav2_setup_command()}")
        sys.exit(2)
