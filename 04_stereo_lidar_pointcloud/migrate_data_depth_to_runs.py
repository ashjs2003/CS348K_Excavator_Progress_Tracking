"""
Move legacy batch outputs from data/<scene>/depth/pair_<id>/ into outputs/runs/<scene>/pair_<id>/.

Raw captures stay in data/. Generated depth products belong under outputs/runs/.

    python migrate_data_depth_to_runs.py
    python migrate_data_depth_to_runs.py --folders checkerboard_data --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from output_runs import RUNS_ROOT, data_pair_run_id, path_for_manifest, write_run_info

DEFAULT_DATA_ROOT = _REPO_ROOT / "data"
LEGACY_DEPTH = "depth"


def parse_args():
    parser = argparse.ArgumentParser(description="Migrate data/<scene>/depth/ → outputs/runs/<scene>/pair_<id>/")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--folders", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--remove-legacy", action="store_true", help="Delete data/<scene>/depth/ after successful migrate")
    return parser.parse_args()


def discover_legacy_scenes(data_root: Path, only: list[str] | None) -> list[Path]:
    scenes = []
    for folder in sorted(data_root.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        if only and folder.name not in only:
            continue
        legacy = folder / LEGACY_DEPTH
        if legacy.is_dir() and any(legacy.glob("pair_*")):
            scenes.append(folder)
    return scenes


def migrate_pair(scene_dir: Path, pair_dir: Path, dry_run: bool) -> Path:
    pair_id = pair_dir.name.replace("pair_", "", 1) if pair_dir.name.startswith("pair_") else pair_dir.name
    run_id = data_pair_run_id(scene_dir.name, pair_id)
    run_dir = RUNS_ROOT / scene_dir.name / f"pair_{pair_id}"
    capture = run_dir / "capture"
    depth_out = run_dir / "depth"

    left = scene_dir / f"pair_{pair_id}_rgb_L.png"
    right = scene_dir / f"pair_{pair_id}_rgb_R.png"
    lidar = scene_dir / f"pair_{pair_id}_lidar.csv"

    print(f"  {path_for_manifest(pair_dir)} -> {path_for_manifest(run_dir)}")

    if dry_run:
        return run_dir

    if run_dir.exists():
        shutil.rmtree(run_dir)
    capture.mkdir(parents=True)
    depth_out.mkdir(parents=True)
    (run_dir / "validation").mkdir(parents=True, exist_ok=True)
    (run_dir / "overlays").mkdir(parents=True, exist_ok=True)

    shutil.copy2(left, capture / "rgb1.png")
    shutil.copy2(right, capture / "rgb2.png")
    if lidar.is_file():
        shutil.copy2(lidar, capture / "lidar_scan.csv")

    for item in pair_dir.iterdir():
        dest = depth_out / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    write_run_info(
        run_dir,
        data_source=path_for_manifest(scene_dir),
        data_pair_id=pair_id,
        migrated_from=path_for_manifest(pair_dir),
    )
    return run_dir


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    scenes = discover_legacy_scenes(data_root, args.folders)
    if not scenes:
        print(f"No legacy {LEGACY_DEPTH}/pair_* folders under {data_root}")
        return

    migrated = []
    for scene_dir in scenes:
        legacy_root = scene_dir / LEGACY_DEPTH
        print(f"\n# {scene_dir.name}")
        for pair_dir in sorted(legacy_root.glob("pair_*")):
            if not pair_dir.is_dir():
                continue
            run_dir = migrate_pair(scene_dir, pair_dir, args.dry_run)
            migrated.append((scene_dir, pair_dir, run_dir))

        if args.remove_legacy and not args.dry_run:
            shutil.rmtree(legacy_root)
            print(f"  removed {path_for_manifest(legacy_root)}")

    stale_manifest = data_root / "batch_dav2_summary.json"
    if args.remove_legacy and not args.dry_run and stale_manifest.is_file():
        stale_manifest.unlink()
        print(f"  removed stale {path_for_manifest(stale_manifest)}")

    print(f"\nMigrated {len(migrated)} pair(s).")
    if not args.dry_run:
        print("Evaluate with: python 06_evaluate_run.py --run <scene>/pair_<id>")


if __name__ == "__main__":
    main()
