"""
Timestamped run directories under outputs/runs/ so captures never overwrite.

Layout per run:
    outputs/runs/<YYYYMMDD_HHMMSS>_<label>/
        run_info.json
        capture/          rgb1.png, rgb2.png, lidar_scan.csv, metadata.json
        stereo/             disparity, point clouds, rectification images
        validation/         LiDAR vs stereo metrics
        overlays/           LiDAR projected on RGB images
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = REPO_ROOT / "outputs"
RUNS_ROOT = OUTPUTS_ROOT / "runs"
LATEST_POINTER = RUNS_ROOT / "latest.txt"

# Legacy flat paths (used when no runs exist yet)
LEGACY_CAPTURE_DIR = REPO_ROOT / "capture"
LEGACY_OUTPUTS_DIR = OUTPUTS_ROOT


def slugify(label: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower())
    return slug.strip("_") or "run"


def create_run_dir(label: str = "capture") -> Path:
    """Create a new timestamped run folder and mark it as latest."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{stamp}_{slugify(label)}"
    run_dir = RUNS_ROOT / run_name
    for sub in ("capture", "stereo", "validation", "overlays"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    LATEST_POINTER.write_text(run_name + "\n")

    latest_link = RUNS_ROOT / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.resolve())

    return run_dir


def list_runs() -> list[str]:
    if not RUNS_ROOT.exists():
        return []
    return sorted(
        path.name
        for path in RUNS_ROOT.iterdir()
        if path.is_dir() and path.name not in {"latest"}
    )


def capture_is_ready(capture_dir: Path) -> bool:
    return (capture_dir / "rgb1.png").is_file() and (capture_dir / "rgb2.png").is_file()


def runs_with_capture() -> list[str]:
    ready = []
    for name in list_runs():
        if capture_is_ready(RUNS_ROOT / name / "capture"):
            ready.append(name)
    return ready


def resolve_run_dir(run: str | None = None) -> Path:
    """
    Resolve a run directory.

    run:
      - None or "latest" -> newest run that has rgb1.png + rgb2.png in capture/
      - otherwise exact folder name under outputs/runs/
    """
    if run is None or run == "latest":
        for name in reversed(runs_with_capture()):
            return RUNS_ROOT / name
        if LATEST_POINTER.exists():
            empty_latest = RUNS_ROOT / LATEST_POINTER.read_text().strip()
            if empty_latest.exists() and not capture_is_ready(empty_latest / "capture"):
                raise FileNotFoundError(
                    f"Latest run '{empty_latest.name}' has no capture images.\n"
                    "Run: python 01_capture_one_set.py --label carpet\n"
                    f"Or remove the empty run and use legacy data in {LEGACY_CAPTURE_DIR}"
                )
        latest_link = RUNS_ROOT / "latest"
        if latest_link.is_symlink():
            linked = latest_link.resolve()
            if linked.exists() and capture_is_ready(linked / "capture"):
                return linked

    run_dir = RUNS_ROOT / run
    if run_dir.exists():
        if not capture_is_ready(run_dir / "capture"):
            raise FileNotFoundError(
                f"Run '{run}' exists but capture/ is missing rgb1.png and rgb2.png.\n"
                "Re-capture with: python 01_capture_one_set.py --label <name>"
            )
        return run_dir

    raise FileNotFoundError(
        f"Run not found: {run_dir}\n"
        f"Runs with capture: {', '.join(runs_with_capture()) or '(none)'}\n"
        "Create one with: python 01_capture_one_set.py --label <name>"
    )


def write_run_info(run_dir: Path, **fields) -> None:
    path = run_dir / "run_info.json"
    payload = {}
    if path.exists():
        payload = json.loads(path.read_text())
    payload.update(fields)
    payload["run_dir"] = str(run_dir.resolve())
    path.write_text(json.dumps(payload, indent=2) + "\n")


def add_run_cli_arguments(parser) -> None:
    parser.add_argument(
        "--run",
        default="latest",
        help="Run folder under outputs/runs/ (default: latest)",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List available runs and exit",
    )


def handle_list_runs(args) -> bool:
    if not getattr(args, "list_runs", False):
        return False
    runs = list_runs()
    if not runs:
        print("No runs yet. Capture with: python 01_capture_one_set.py --label <name>")
        return True
    latest = LATEST_POINTER.read_text().strip() if LATEST_POINTER.exists() else None
    ready = set(runs_with_capture())
    for name in runs:
        tags = []
        if name == latest:
            tags.append("latest")
        if name in ready:
            tags.append("has capture")
        else:
            tags.append("empty capture")
        suffix = f" ({', '.join(tags)})" if tags else ""
        print(f"  {name}{suffix}")
    return True


class RunPaths:
    """Resolved paths for one run (or legacy flat layout)."""

    def __init__(self, run_dir: Path | None):
        self.run_dir = run_dir
        if run_dir is not None:
            self.capture = run_dir / "capture"
            self.stereo = run_dir / "stereo"
            self.validation = run_dir / "validation"
            self.overlays = run_dir / "overlays"
            for d in (self.stereo, self.validation, self.overlays):
                d.mkdir(parents=True, exist_ok=True)
        else:
            self.capture = LEGACY_CAPTURE_DIR
            self.stereo = LEGACY_OUTPUTS_DIR
            self.validation = LEGACY_OUTPUTS_DIR
            self.overlays = LEGACY_OUTPUTS_DIR

    @property
    def rgb1_image(self) -> Path:
        return self.capture / "rgb1.png"

    @property
    def rgb2_image(self) -> Path:
        return self.capture / "rgb2.png"

    @property
    def lidar_csv(self) -> Path:
        return self.capture / "lidar_scan.csv"


def resolve_run_paths(run: str | None = "latest") -> RunPaths:
    try:
        paths = RunPaths(resolve_run_dir(run))
        if run in (None, "latest") and paths.run_dir and not capture_is_ready(paths.capture):
            raise FileNotFoundError(f"Run '{paths.run_dir.name}' has no capture images.")
        return paths
    except FileNotFoundError as exc:
        if run not in (None, "latest"):
            raise
        if capture_is_ready(LEGACY_CAPTURE_DIR):
            print(f"{exc}")
            print(f"Using legacy {LEGACY_CAPTURE_DIR}/ and {LEGACY_OUTPUTS_DIR}/")
            return RunPaths(None)
        raise
