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


def current_python_has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def dav2_env_ready(env_name: str = "depth_anything_v2") -> bool:
    if conda_env_exists(env_name):
        probe = conda_run_python(env_name, "import torch; raise SystemExit(0)")
        if probe.returncode == 0:
            return True
    return current_python_has_torch()


def dav2_python_command(env_name: str = "depth_anything_v2") -> list[str]:
    """Python executable argv prefix for DA-V2 inference (conda env or current venv)."""
    if conda_env_exists(env_name):
        probe = conda_run_python(env_name, "import torch; raise SystemExit(0)")
        if probe.returncode == 0:
            return ["conda", "run", "-n", env_name, "--no-capture-output", "python"]
    return [sys.executable]


def dav2_setup_command() -> str:
    if is_windows():
        return r"powershell -ExecutionPolicy Bypass -File scripts\setup_depth_anything_v2.ps1"
    return "bash scripts/setup_depth_anything_v2.sh"


def require_dav2_or_exit(env_name: str = "depth_anything_v2") -> None:
    if dav2_env_ready(env_name):
        return
    print(f"Depth Anything V2 is not ready (no torch in conda env '{env_name}' or current Python).")
    if shutil.which("conda") is not None:
        print(f"Setup:\n  {dav2_setup_command()}")
    else:
        print("Install PyTorch in the active venv, or install Miniconda and run:")
        print(f"  {dav2_setup_command()}")
    sys.exit(2)
