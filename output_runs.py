"""
Timestamped run directories under outputs/runs/ so captures never overwrite.

Layout per run:
    outputs/runs/<YYYYMMDD_HHMMSS>_<label>/
        run_info.json
        capture/          rgb1.png, rgb2.png, lidar_scan.csv, metadata.json
        depth/              stereo + monocular depth, disparity, point clouds, rectified RGB
        validation/         LiDAR vs stereo metrics
        overlays/           LiDAR projected on RGB images

    Batch imports from data/ also use nested runs:

        outputs/runs/<data_folder>/pair_<id>/
            capture/  depth/  validation/  overlays/  run_info.json
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
DEPTH_SUBDIR = "depth"
LEGACY_DEPTH_SUBDIR = "stereo"  # pre-rename runs

# Legacy flat paths (used when no runs exist yet)
LEGACY_CAPTURE_DIR = REPO_ROOT / "capture"
LEGACY_OUTPUTS_DIR = OUTPUTS_ROOT


def path_for_manifest(path: Path, base: Path = REPO_ROOT) -> str:
    """Repo-relative POSIX path (portable in JSON)."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def run_dir_from_id(run: str) -> Path:
    """Map --run id to a directory under outputs/runs (supports nested scene/pair_000)."""
    return RUNS_ROOT.joinpath(*run.replace("\\", "/").split("/"))


def data_pair_run_id(scene_folder: str, pair_id: str) -> str:
    return f"{scene_folder}/pair_{pair_id}"


def capture_is_ready(capture_dir: Path) -> bool:
    return (capture_dir / "rgb1.png").is_file() and (capture_dir / "rgb2.png").is_file()


def iter_run_dirs() -> list[tuple[str, Path]]:
    """All run directories that contain capture/rgb1.png + rgb2.png."""
    if not RUNS_ROOT.exists():
        return []
    found: list[tuple[str, Path]] = []
    for capture_dir in RUNS_ROOT.rglob("capture"):
        if capture_dir.name != "capture" or not capture_dir.is_dir():
            continue
        run_dir = capture_dir.parent
        if not capture_is_ready(capture_dir):
            continue
        run_id = run_dir.relative_to(RUNS_ROOT).as_posix()
        found.append((run_id, run_dir))
    return sorted(found, key=lambda item: item[0])


def slugify(label: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower())
    return slug.strip("_") or "run"


def create_run_dir(label: str = "capture") -> Path:
    """Create a new timestamped run folder and mark it as latest."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{stamp}_{slugify(label)}"
    run_dir = RUNS_ROOT / run_name
    for sub in ("capture", DEPTH_SUBDIR, "validation", "overlays"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    LATEST_POINTER.write_text(run_name + "\n")

    latest_link = RUNS_ROOT / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.resolve())

    return run_dir


def list_runs() -> list[str]:
    return [run_id for run_id, _ in iter_run_dirs()]


def runs_with_capture() -> list[str]:
    return list_runs()


def top_level_runs_with_capture() -> list[str]:
    """Runs directly under outputs/runs/ (live captures), excluding nested data imports."""
    ready = []
    if not RUNS_ROOT.exists():
        return ready
    for path in RUNS_ROOT.iterdir():
        if not path.is_dir() or path.name == "latest":
            continue
        if capture_is_ready(path / "capture"):
            ready.append(path.name)
    return sorted(ready)


def resolve_run_dir(run: str | None = None) -> Path:
    """
    Resolve a run directory.

    run:
      - None or "latest" -> newest top-level run with capture/ (01_capture_one_set)
      - otherwise path under outputs/runs/, e.g. 20260521_143022_carpet or checkerboard_data/pair_000
    """
    if run is None or run == "latest":
        for name in reversed(top_level_runs_with_capture()):
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

    run_dir = run_dir_from_id(run)
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
        "Create one with: python 01_capture_one_set.py --label <name>\n"
        "Or import from data/: python batch_dav2_data_folders.py"
    )


def resolve_depth_dir(run_dir: Path, create: bool = True) -> Path:
    """Depth products folder (stereo + monocular); falls back to legacy ``stereo/``."""
    depth = run_dir / DEPTH_SUBDIR
    legacy = run_dir / LEGACY_DEPTH_SUBDIR
    if depth.is_dir() or not legacy.is_dir():
        if create:
            depth.mkdir(parents=True, exist_ok=True)
        return depth
    return legacy


def write_run_info(run_dir: Path, **fields) -> None:
    path = run_dir / "run_info.json"
    payload = {}
    if path.exists():
        payload = json.loads(path.read_text())
    payload.update(fields)
    payload["run_dir"] = path_for_manifest(run_dir)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def add_run_cli_arguments(parser) -> None:
    parser.add_argument(
        "--run",
        default="latest",
        help="Run under outputs/runs/ (default: latest). Nested ok: checkerboard_data/pair_000",
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
            self.depth = resolve_depth_dir(run_dir)
            self.validation = run_dir / "validation"
            self.overlays = run_dir / "overlays"
            for d in (self.validation, self.overlays):
                d.mkdir(parents=True, exist_ok=True)
        else:
            self.capture = LEGACY_CAPTURE_DIR
            self.depth = LEGACY_OUTPUTS_DIR
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
