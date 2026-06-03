"""
Debug grid: ROI − flap depth heuristic overlaid on RGB (cardboard S / M / L).

Each cell = 2×2 panels (OpenCV, DA-V2, DA-V2 GT, Foundation).

Output:
  outputs/runs/_combined/box_depth_debug_overlay_grid.png
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

from evaluation.depth_debug_overlay_grid import render_depth_debug_grid
from output_runs import RUNS_ROOT


def parse_args():
    p = argparse.ArgumentParser(description="Depth heuristic debug overlay grid")
    p.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    p.add_argument("--data-root", type=Path, default=_REPO_ROOT / "data")
    p.add_argument("--out-dir", type=Path, default=RUNS_ROOT / "_combined")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_png = Path(args.out_dir) / "box_depth_debug_overlay_grid.png"
    render_depth_debug_grid(Path(args.data_root), Path(args.runs_root), out_png)
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
