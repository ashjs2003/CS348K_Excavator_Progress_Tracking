"""Platform helpers so FoundationStereo is optional and never required on macOS."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path


def is_macos() -> bool:
    return platform.system() == "Darwin"


def is_windows() -> bool:
    return platform.system() == "Windows"


def foundation_stereo_supported_here() -> bool:
    """True only when conda env exists and reports CUDA (typical: Windows/Linux + NVIDIA)."""
    if is_macos():
        return False
    if shutil.which("conda") is None:
        return False
    probe = subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "foundation_stereo",
            "python",
            "-c",
            "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)",
        ],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def setup_command() -> str:
    if is_windows():
        return r"powershell -ExecutionPolicy Bypass -File scripts\setup_foundationstereo.ps1"
    if is_macos():
        return (
            "On Mac: skip FoundationStereo setup. Use OpenCV only:\n"
            "  python 02_make_stereo_pointcloud.py\n"
            "On Windows (NVIDIA GPU): scripts\\setup_foundationstereo.ps1"
        )
    return "bash scripts/setup_foundationstereo.sh"


def require_foundation_stereo_or_exit() -> None:
    """Called at start of 02_make_stereo_pointcloud_foundation.py on unsupported hosts."""
    if is_macos():
        print(
            "FoundationStereo inference needs an NVIDIA GPU and does not run on macOS.\n"
            "Your Mac workflow is unchanged — keep using:\n"
            "  python 02_make_stereo_pointcloud.py\n"
            "\n"
            "On your Windows machine (same repo clone):\n"
            f"  {setup_command()}\n"
            "  python 02_make_stereo_pointcloud_foundation.py --run <run_id>\n"
            "\n"
            "Copy outputs/runs/<run>/ back to Mac to compare with compare_stereo_methods.py"
        )
        sys.exit(2)
    if not foundation_stereo_supported_here():
        print(
            "FoundationStereo is not ready on this machine (missing conda env or CUDA).\n"
            f"Setup:\n  {setup_command()}\n"
            "OpenCV stereo still works:\n  python 02_make_stereo_pointcloud.py"
        )
        sys.exit(2)
