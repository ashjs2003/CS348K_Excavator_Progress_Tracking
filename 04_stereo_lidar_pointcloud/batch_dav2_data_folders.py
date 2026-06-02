"""
Batch Depth Anything V2 (and OpenCV stereo for metric scaling) on data/ capture folders.

Each subfolder of data/ with pair_*_rgb_L.png + pair_*_rgb_R.png becomes one run:

    outputs/runs/<data_folder>/pair_<id>/
        capture/   rgb1.png, rgb2.png, lidar_scan.csv (if present)
        depth/     stereo + DA-V2 products
        validation/  (use 06_evaluate_run.py --run <folder>/pair_<id>)
        overlays/
        run_info.json

Manifest: outputs/runs/batch_from_data_summary.json (repo-relative paths).

Run from 04_stereo_lidar_pointcloud:

    python batch_dav2_data_folders.py
    python batch_dav2_data_folders.py --folders checkerboard_data excavator_S
    python batch_dav2_data_folders.py --skip-existing --continue-on-error
    python 06_evaluate_run.py --run checkerboard_data/pair_000
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from output_runs import (
    RUNS_ROOT,
    data_pair_run_id,
    path_for_manifest,
    run_dir_from_id,
    write_run_info,
)
from ml_inference_platform import require_dav2_or_exit
from stereo_methods import DEFAULT_STEREO_METHOD, normalize_stereo_method

DEFAULT_DATA_ROOT = _REPO_ROOT / "data"
STEREO_SCRIPT = _SCRIPT_DIR / "02_make_stereo_pointcloud.py"
DAV2_SCRIPT = _SCRIPT_DIR / "02_make_depth_anything_pointcloud.py"
MANIFEST_PATH = RUNS_ROOT / "batch_from_data_summary.json"
LEGACY_DATA_DEPTH = "depth"  # old layout: data/<scene>/depth/pair_<id>/


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OpenCV stereo + DA-V2 for each pair; outputs under outputs/runs/<scene>/pair_<id>/."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--folders", nargs="*", default=None)
    parser.add_argument(
        "--stereo-method",
        type=normalize_stereo_method,
        default=DEFAULT_STEREO_METHOD,
        help="stereobm (default), sgbm, flow, blend; aliases: carpet, bm",
    )
    parser.add_argument("--depth-min", type=float, default=0.45)
    parser.add_argument("--depth-max", type=float, default=2.0)
    parser.add_argument("--conda-env", default="depth_anything_v2")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if outputs/runs/<scene>/pair_<id>/depth/depth_metric_dav2.npy exists",
    )
    parser.add_argument(
        "--mirror-to-data",
        action="store_true",
        help="Also copy depth/ into data/<scene>/depth/pair_<id>/ (legacy layout)",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover_scene_folders(data_root: Path, only: list[str] | None) -> list[Path]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    folders = sorted(
        p for p in data_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if only:
        wanted = set(only)
        folders = [p for p in folders if p.name in wanted]
        missing = wanted - {p.name for p in folders}
        if missing:
            raise FileNotFoundError(f"Unknown or empty folders under {data_root}: {sorted(missing)}")
    return [p for p in folders if discover_pairs(p)]


def discover_pairs(folder: Path) -> list[str]:
    pair_ids = []
    for left_path in sorted(folder.glob("pair_*_rgb_L.png")):
        pair_id = left_path.stem.replace("pair_", "").replace("_rgb_L", "")
        if (folder / f"pair_{pair_id}_rgb_R.png").is_file():
            pair_ids.append(pair_id)
    return pair_ids


def pair_run_dir(scene: str, pair_id: str) -> Path:
    return run_dir_from_id(data_pair_run_id(scene, pair_id))


def legacy_data_depth_dir(folder: Path, pair_id: str) -> Path:
    return folder / LEGACY_DATA_DEPTH / f"pair_{pair_id}"


def stage_capture_run(run_dir: Path, left_path: Path, right_path: Path, lidar_path: Path | None) -> None:
    capture = run_dir / "capture"
    capture.mkdir(parents=True, exist_ok=True)
    shutil.copy2(left_path, capture / "rgb1.png")
    shutil.copy2(right_path, capture / "rgb2.png")
    if lidar_path is not None and lidar_path.is_file():
        shutil.copy2(lidar_path, capture / "lidar_scan.csv")
    for sub in ("depth", "validation", "overlays"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)


def mirror_depth_products(src_depth: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for item in src_depth.iterdir():
        target = dest / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def run_subprocess(script: Path, run_id: str, extra: list[str], dry_run: bool) -> None:
    cmd = [sys.executable, str(script), "--run", run_id, *extra]
    print("  $", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=str(_SCRIPT_DIR), check=True)


def process_pair(folder: Path, pair_id: str, args, summary_rows: list[dict]) -> None:
    left_path = folder / f"pair_{pair_id}_rgb_L.png"
    right_path = folder / f"pair_{pair_id}_rgb_R.png"
    lidar_path = folder / f"pair_{pair_id}_lidar.csv"
    run_id = data_pair_run_id(folder.name, pair_id)
    run_dir = pair_run_dir(folder.name, pair_id)
    marker = run_dir / "depth" / "depth_metric_dav2.npy"

    if args.skip_existing and marker.is_file():
        print(f"  skip (exists): {marker}")
        summary_rows.append(
            {
                "folder": folder.name,
                "pair_id": pair_id,
                "run_id": run_id,
                "status": "skipped",
                "output": path_for_manifest(run_dir),
            }
        )
        return

    print(f"\n=== {run_id} ===")

    if args.dry_run:
        print(f"  -> {path_for_manifest(run_dir)}")
        summary_rows.append(
            {"folder": folder.name, "pair_id": pair_id, "run_id": run_id, "status": "dry_run"}
        )
        return

    if run_dir.exists():
        shutil.rmtree(run_dir)
    stage_capture_run(run_dir, left_path, right_path, lidar_path if lidar_path.is_file() else None)

    stereo_args = [
        "--method",
        args.stereo_method,
        "--depth-min",
        str(args.depth_min),
        "--depth-max",
        str(args.depth_max),
    ]
    dav2_args = [
        "--reuse-rectified",
        "--depth-min",
        str(args.depth_min),
        "--depth-max",
        str(args.depth_max),
        "--conda-env",
        args.conda_env,
    ]

    run_subprocess(STEREO_SCRIPT, run_id, stereo_args, dry_run=False)
    run_subprocess(DAV2_SCRIPT, run_id, dav2_args, dry_run=False)

    stats = {}
    scale_path = run_dir / "depth" / "depth_scaling_dav2.json"
    if scale_path.is_file():
        stats = json.loads(scale_path.read_text())

    write_run_info(
        run_dir,
        data_source=path_for_manifest(folder),
        data_pair_id=pair_id,
        batch_stereo_method=args.stereo_method,
        dav2_scale_info=stats,
    )

    if args.mirror_to_data:
        mirror_depth_products(run_dir / "depth", legacy_data_depth_dir(folder, pair_id))

    summary_rows.append(
        {
            "folder": folder.name,
            "pair_id": pair_id,
            "run_id": run_id,
            "status": "ok",
            "output": path_for_manifest(run_dir),
            "fit_correlation": stats.get("fit_correlation"),
            "scale_reference": stats.get("scale_reference"),
        }
    )
    print(f"  saved: {path_for_manifest(run_dir)}")


def main():
    args = parse_args()
    if not args.dry_run:
        require_dav2_or_exit(args.conda_env)

    data_root = args.data_root.resolve()
    folders = discover_scene_folders(data_root, args.folders)
    if not folders:
        print(f"No scene folders with stereo pairs under {data_root}")
        return

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Data root: {path_for_manifest(data_root)}")
    print(f"Output root: {path_for_manifest(RUNS_ROOT)}")
    print(f"Folders ({len(folders)}): {', '.join(p.name for p in folders)}")

    summary_rows: list[dict] = []
    failures: list[str] = []

    for folder in folders:
        (RUNS_ROOT / folder.name).mkdir(parents=True, exist_ok=True)
        pair_ids = discover_pairs(folder)
        print(f"\n# {folder.name} — {len(pair_ids)} pair(s)")
        for pair_id in pair_ids:
            try:
                process_pair(folder, pair_id, args, summary_rows)
            except Exception as exc:
                msg = f"{folder.name}/pair_{pair_id}: {exc}"
                failures.append(msg)
                summary_rows.append(
                    {
                        "folder": folder.name,
                        "pair_id": pair_id,
                        "run_id": data_pair_run_id(folder.name, pair_id),
                        "status": "error",
                        "error": str(exc),
                    }
                )
                print(f"  ERROR: {exc}")
                if not args.continue_on_error:
                    raise

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "path_base": "repo_root",
        "data_root": path_for_manifest(data_root),
        "runs_root": path_for_manifest(RUNS_ROOT),
        "layout": "outputs/runs/<data_folder>/pair_<id>/",
        "stereo_method": args.stereo_method,
        "depth_band_m": [args.depth_min, args.depth_max],
        "pairs": summary_rows,
        "failures": failures,
    }
    if not args.dry_run:
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\nWrote {path_for_manifest(MANIFEST_PATH)}")

    ok = sum(1 for r in summary_rows if r.get("status") == "ok")
    skipped = sum(1 for r in summary_rows if r.get("status") == "skipped")
    print(f"\nDone: {ok} processed, {skipped} skipped, {len(failures)} failed.")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
