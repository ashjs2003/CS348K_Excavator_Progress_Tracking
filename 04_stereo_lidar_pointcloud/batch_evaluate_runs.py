"""
Run 06_evaluate_run.py on every run under outputs/runs/ with depth + LiDAR.

    python batch_evaluate_runs.py
    python batch_evaluate_runs.py --continue-on-error
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from output_runs import RUNS_ROOT, iter_run_dirs

EVAL_SCRIPT = _SCRIPT_DIR / "06_evaluate_run.py"
LOG_PATH = RUNS_ROOT / "batch_evaluate_all.log"


def runs_ready_for_eval() -> list[str]:
    ready = []
    for run_id, run_dir in iter_run_dirs():
        depth = run_dir / "depth"
        if not depth.is_dir():
            continue
        if not (run_dir / "capture" / "lidar_scan.csv").is_file():
            continue
        geo = depth / "shared" / "stereo_geometry.npz"
        if not geo.is_file() and not (depth / "stereo_geometry.npz").is_file():
            continue
        ready.append(run_id)
    return ready


def parse_args():
    p = argparse.ArgumentParser(description="Batch structured evaluation for all data runs")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--log", type=Path, default=LOG_PATH)
    return p.parse_args()


def main():
    args = parse_args()
    runs = runs_ready_for_eval()
    print(f"Evaluating {len(runs)} runs -> {args.log}")
    ok = fail = 0
    failures: list[str] = []
    log_lines: list[str] = []

    for i, run_id in enumerate(runs, 1):
        header = f"\n===== [{i}/{len(runs)}] {run_id} ====="
        print(header, flush=True)
        log_lines.append(header)
        proc = subprocess.run(
            [sys.executable, str(EVAL_SCRIPT), "--run", run_id],
            cwd=str(_SCRIPT_DIR),
        )
        if proc.returncode == 0:
            ok += 1
        else:
            fail += 1
            failures.append(run_id)
            log_lines.append(f"FAILED: {run_id}")
            if not args.continue_on_error:
                break

    summary = f"\nDone: {ok} ok, {fail} failed"
    print(summary)
    log_lines.append(summary)
    if failures:
        log_lines.append("Failures:")
        for f in failures:
            log_lines.append(f"  {f}")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
