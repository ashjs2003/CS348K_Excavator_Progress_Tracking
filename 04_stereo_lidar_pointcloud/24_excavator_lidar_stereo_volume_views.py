"""
Excavator all-views table: V = LiDAR width × catalog height × stereo ROI depth.

Same views as excavator_MS_all_views (reads that JSON for pair list when present).

Outputs:
  outputs/runs/_combined/excavator_MS_lidar_stereo_volume_all_views.png
  outputs/runs/_combined/excavator_MS_lidar_stereo_volume_all_views.csv
  outputs/runs/_combined/excavator_MS_lidar_stereo_volume_all_views.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.excavator_lidar_stereo_volume import (
    EXCAVATOR_GT_VOLUME_CM3,
    build_excavator_lidar_depth_volume_rows,
    build_excavator_lidar_stereo_volume_rows,
    pair_list_from_combined_json,
)
from output_runs import RUNS_ROOT
from evaluation.roi_gt_compare import METHOD_ORDER

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "combine_volume_scenes",
    _SCRIPT_DIR / "12_combine_volume_scenes.py",
)
_combine = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_combine)


def parse_args():
    p = argparse.ArgumentParser(description="Excavator LiDAR×height×stereo-depth volume table")
    p.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    p.add_argument("--data-root", type=Path, default=_REPO_ROOT / "data")
    p.add_argument("--out-dir", type=Path, default=RUNS_ROOT / "_combined")
    p.add_argument(
        "--reference-json",
        type=Path,
        default=RUNS_ROOT / "_combined" / "excavator_MS_all_views.json",
        help="Use same scene/pair rows as this file",
    )
    p.add_argument("--table-columns", type=int, default=3)
    p.add_argument(
        "--also-lidar-depth",
        action="store_true",
        help="Also write excavator_MS_lidar_depth_volume_all_views (LiDAR depth instead of stereo)",
    )
    return p.parse_args()


def _write_combined_table(
    rows: list[dict],
    method_cols: list[str],
    *,
    stem: str,
    subtitle: str,
    out_dir: Path,
    runs_root: Path,
    scene_gt: dict[str, float],
    table_columns: int,
) -> None:
    title = f"{stem}: volume estimates across views"
    _combine.write_csv(rows, method_cols, out_dir / f"{stem}.csv", None, scene_gt)
    _combine.write_json(rows, list(EXCAVATOR_GT_VOLUME_CM3.keys()), out_dir / f"{stem}.json", None, scene_gt)
    _combine.write_png_with_embedded_images(
        rows,
        method_cols,
        out_dir / f"{stem}.png",
        title,
        None,
        scene_gt,
        runs_root,
        table_columns,
    )
    print(f"Wrote {out_dir / f'{stem}.png'}  ({subtitle})")
    print(f"Rows: {len(rows)}")


def main() -> int:
    args = parse_args()
    pair_filter = pair_list_from_combined_json(Path(args.reference_json))
    pf = pair_filter if pair_filter else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_gt = dict(EXCAVATOR_GT_VOLUME_CM3)

    rows = build_excavator_lidar_stereo_volume_rows(args.runs_root, args.data_root, pair_filter=pf)
    if not rows:
        print(
            "No rows built. Need excavator roi_bbox_volume_estimates.csv/json, "
            "LiDAR CSVs, and stereo depth columns."
        )
        return 1

    method_cols = [
        f"{m}_volume_cm3"
        for m in METHOD_ORDER
        if any(str(r.get(f"{m}_volume_cm3", "")).strip() for r in rows)
    ]
    _write_combined_table(
        rows,
        method_cols,
        stem="excavator_MS_lidar_stereo_volume_all_views",
        subtitle="V = LiDAR width × catalog height × stereo ROI depth",
        out_dir=out_dir,
        runs_root=args.runs_root,
        scene_gt=scene_gt,
        table_columns=args.table_columns,
    )

    if args.also_lidar_depth:
        rows_d = build_excavator_lidar_depth_volume_rows(args.runs_root, args.data_root, pair_filter=pf)
        if rows_d:
            _write_combined_table(
                rows_d,
                ["lidar_volume_cm3"],
                stem="excavator_MS_lidar_depth_volume_all_views",
                subtitle="V = LiDAR width × catalog height × LiDAR edge→back depth",
                out_dir=out_dir,
                runs_root=args.runs_root,
                scene_gt=scene_gt,
                table_columns=args.table_columns,
            )
        else:
            print("Skipped LiDAR-depth table: no rows built.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
