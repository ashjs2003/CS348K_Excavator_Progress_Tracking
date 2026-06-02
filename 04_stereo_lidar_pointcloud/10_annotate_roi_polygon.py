"""
Draw a polygon ROI on each capture image; saved under data/<scene>/pair_XXX_roi.json.

Use the rectified preview from the run when available (matches depth maps).

    python 10_annotate_roi_polygon.py --scene checkerboard_data --pair 001
    python 10_annotate_roi_polygon.py --scene checkerboard_data --all

Controls:
  Left click: add vertex
  Right click: close polygon (need >= 3 points)
  u: undo last vertex
  c: clear polygon
  s: save and continue
  n: skip (no save)
  q: quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.roi_gt_compare import (
    DEFAULT_WALL_PERCENTILE,
    annotation_image_paths,
    discover_data_pairs,
    roi_json_path,
    save_roi_polygon,
)

DEFAULT_DATA_ROOT = _REPO_ROOT / "data"


class PolygonAnnotator:
    def __init__(self, image_bgr: np.ndarray, window_name: str):
        self.base = image_bgr.copy()
        self.window = window_name
        self.points: list[tuple[int, int]] = []
        self.closed = False

    def _redraw(self) -> np.ndarray:
        vis = self.base.copy()
        if self.points:
            pts = np.array(self.points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], self.closed, (0, 255, 255), 2)
            for p in self.points:
                cv2.circle(vis, p, 4, (0, 200, 255), -1)
        cv2.putText(
            vis,
            "LMB=vertex  RMB=close  u=undo  c=clear  s=save  n=skip  q=quit",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        return vis

    def on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((int(x), int(y)))
            self.closed = False
        elif event == cv2.EVENT_RBUTTONDOWN and len(self.points) >= 3:
            self.closed = True

    def run(self) -> list[tuple[int, int]] | None:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.on_mouse)
        while True:
            cv2.imshow(self.window, self._redraw())
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                return None
            if key == ord("n"):
                return []
            if key == ord("u") and self.points:
                self.points.pop()
                self.closed = False
            if key == ord("c"):
                self.points.clear()
                self.closed = False
            if key == ord("s"):
                if len(self.points) >= 3:
                    return self.points
        cv2.destroyWindow(self.window)


def parse_args():
    p = argparse.ArgumentParser(description="Annotate polygon ROI per data pair")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--scene", required=True, help="data/<scene> folder name")
    p.add_argument("--pair", type=str, default=None, help="e.g. 001 or pair_001")
    p.add_argument("--all", action="store_true", help="All pairs in scene missing ROI or --force")
    p.add_argument("--force", action="store_true", help="Re-annotate even if ROI exists")
    p.add_argument("--raw-left", action="store_true", help="Use data/*_rgb_L.png instead of rectified")
    p.add_argument("--wall-percentile", type=float, default=DEFAULT_WALL_PERCENTILE)
    return p.parse_args()


def normalize_pair_id(pair: str) -> str:
    return pair.replace("pair_", "", 1) if pair.startswith("pair_") else pair


def annotate_one(
    scene: str,
    pair_id: str,
    data_root: Path,
    repo_root: Path,
    *,
    force: bool,
    use_raw: bool,
    wall_percentile: float,
) -> bool:
    scene_dir = data_root / scene
    out_path = roi_json_path(scene_dir, pair_id)
    if out_path.is_file() and not force:
        print(f"  pair_{pair_id}: {out_path.name} exists, skip (use --force to redo)")
        return True

    rect_p, raw_p = annotation_image_paths(repo_root, scene, pair_id)
    img_path = raw_p if use_raw else (rect_p or raw_p)
    if img_path is None:
        print(f"  pair_{pair_id}: no image found")
        return False
    if rect_p is None and not use_raw:
        print(f"  pair_{pair_id}: using raw left (no rectified run yet)")

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"  pair_{pair_id}: could not read {img_path}")
        return False

    title = f"{scene} pair_{pair_id}"
    annotator = PolygonAnnotator(bgr, title)
    pts = annotator.run()
    cv2.destroyAllWindows()
    if pts is None:
        print("Quit.")
        return False
    if len(pts) < 3:
        print(f"  pair_{pair_id}: skipped")
        return True

    save_roi_polygon(
        out_path,
        scene=scene,
        pair_id=pair_id,
        image_path=img_path,
        polygon_xy=[[int(x), int(y)] for x, y in pts],
        wall_percentile=wall_percentile,
    )
    print(f"  pair_{pair_id}: saved {out_path} ({len(pts)} vertices)")
    return True


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    scene_dir = data_root / args.scene
    if not scene_dir.is_dir():
        print(f"Missing {scene_dir}")
        return 1

    if args.pair:
        pair_ids = [normalize_pair_id(args.pair)]
    elif args.all:
        pair_ids = discover_data_pairs(scene_dir)
    else:
        print("Specify --pair ID or --all")
        return 1

    for pair_id in pair_ids:
        if not annotate_one(
            args.scene,
            pair_id,
            data_root,
            _REPO_ROOT,
            force=args.force,
            use_raw=args.raw_left,
            wall_percentile=args.wall_percentile,
        ):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
