"""
Write a plain-language comparison report from evaluation_summary.json.

Run after 06_evaluate_run.py:
    python 07_generate_eval_report.py --run latest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from output_runs import add_run_cli_arguments, handle_list_runs, resolve_run_paths

METHOD_LABELS = {
    "opencv": "Classic stereo (OpenCV)",
    "dav2": "AI single-camera depth (Depth Anything V2)",
    "foundation": "AI stereo (FoundationStereo)",
}

# Minimum share of LiDAR hits that got a depth reading before we trust ray accuracy
MIN_ASSOCIATION_RATE = 0.10
# Ray error below this (meters) is "good" on toy scale
GOOD_RAY_MEDIAN_M = 0.15
# Free-space violation above this (%) is a red flag
MAX_FREE_SPACE_PCT = 15.0


def fmt_m(value, none="not enough data"):
    if value is None:
        return none
    return f"{value * 100:.1f} cm" if value < 1.0 else f"{value:.2f} m"


def fmt_pct(value):
    if value is None:
        return "n/a"
    return f"{value:.1f}%"


def grade_ray(median_m, association_rate, free_space_pct):
    if association_rate < MIN_ASSOCIATION_RATE:
        return "Unreliable for this check", "Too few LiDAR points could be compared (low overlap with depth map)."
    if free_space_pct > MAX_FREE_SPACE_PCT:
        return "Poor", "Depth often appears closer than the laser measurement (scale or calibration problem)."
    if median_m is None:
        return "Unknown", "No comparison data."
    if median_m <= GOOD_RAY_MEDIAN_M:
        return "Good", "Typical error is within about 15 cm of the laser line."
    if median_m <= 0.35:
        return "Fair", "Usable for rough shape, not for precise volume."
    return "Poor", "Large mismatch with the laser; do not use for precise measurements."


def grade_coverage(pct):
    if pct >= 70:
        return "High", "Most of the image has depth."
    if pct >= 25:
        return "Medium", "Depth in part of the scene."
    return "Low", "Depth only in small patches (common on plain carpet/ sand)."


def pick_winners(methods: dict, photometric: dict) -> dict:
    """Simple rules for non-technical winner lines."""
    ranked_accuracy = []
    ranked_coverage = []
    for key, row in methods.items():
        med = row.get("ray_median_error_m")
        assoc = row.get("association_rate") or 0.0
        cov = row.get("depth_coverage_pct") or 0.0
        fs = row.get("free_space_violation_pct") or 0.0
        if assoc >= MIN_ASSOCIATION_RATE and fs <= MAX_FREE_SPACE_PCT and med is not None:
            ranked_accuracy.append((med, key))
        ranked_coverage.append((-cov, key))  # negate for ascending sort

    ranked_accuracy.sort()
    ranked_coverage.sort()

    best_accuracy = ranked_accuracy[0][1] if ranked_accuracy else None
    best_coverage = ranked_coverage[0][1] if ranked_coverage else None

    overall = best_accuracy or best_coverage
    if best_accuracy and best_coverage and best_accuracy != best_coverage:
        note = (
            f"For **accuracy vs. laser**, prefer **{METHOD_LABELS[best_accuracy]}**. "
            f"For **filling the whole image**, prefer **{METHOD_LABELS[best_coverage]}**. "
            "They are not the same on this scene."
        )
    elif overall:
        note = f"Overall pick for this capture: **{METHOD_LABELS[overall]}**."
    else:
        note = "No method passed our minimum data-quality checks. Recapture or recalibrate."

    return {
        "best_accuracy": best_accuracy,
        "best_coverage": best_coverage,
        "overall_note": note,
    }


def build_report(summary: dict, run_label: str, validation_dir: Path) -> str:
    methods = summary.get("methods", {})
    photometric = summary.get("photometric", {})
    cross = summary.get("cross_method", {})
    consensus = summary.get("consensus_map", {})
    winners = pick_winners(methods, photometric)

    lines = [
        "# Depth method comparison (plain-language report)",
        "",
        f"**Scene / run:** {run_label}",
        "",
        "This report compares three ways to estimate 3D shape from your cameras. "
        "We use your **2D laser** as an independent ruler: where the laser hits a surface, "
        "we check whether each method’s depth agrees.",
        "",
        "---",
        "",
        "## Bottom line (read this first)",
        "",
        winners["overall_note"],
        "",
    ]

    if winners["best_accuracy"]:
        m = methods[winners["best_accuracy"]]
        lines.append(
            f"- **Closest to laser:** {METHOD_LABELS[winners['best_accuracy']]} "
            f"(typical gap {fmt_m(m.get('ray_median_error_m'))})"
        )
    if winners["best_coverage"] and winners["best_coverage"] != winners.get("best_accuracy"):
        m = methods[winners["best_coverage"]]
        lines.append(
            f"- **Most complete picture:** {METHOD_LABELS[winners['best_coverage']]} "
            f"({fmt_pct(m.get('depth_coverage_pct'))} of image has depth)"
        )

    lines.extend([
        "",
        "---",
        "",
        "## What each method is",
        "",
        "| Short name | What it is | Needs |",
        "|------------|------------|-------|",
        "| Classic stereo (OpenCV) | Two cameras, geometry math | Both cameras, calibration |",
        "| Depth Anything V2 | AI depth from left image only | Left camera; scaled using stereo |",
        "| FoundationStereo | AI two-camera depth | Both cameras, Windows + GPU |",
        "",
        "---",
        "",
        "## Comparison table",
        "",
        "| Method | Surface coverage | Match to laser (typical error) | Trust this laser check? | Free-space warnings | Stereo photo match |",
        "|--------|------------------|-------------------------------|-------------------------|---------------------|-------------------|",
    ])

    for key in ("opencv", "dav2", "foundation"):
        if key not in methods:
            continue
        row = methods[key]
        g, why = grade_ray(
            row.get("ray_median_error_m"),
            row.get("association_rate") or 0,
            row.get("free_space_violation_pct") or 0,
        )
        cov_grade, _ = grade_coverage(row.get("depth_coverage_pct"))
        photo = photometric.get(key, {})
        photo_err = photo.get("mean_photometric_error")
        photo_s = f"{photo_err:.1f}" if photo_err is not None else "n/a (not run)"
        trust = "Yes" if (row.get("association_rate") or 0) >= MIN_ASSOCIATION_RATE else "**No** — too little overlap"
        lines.append(
            f"| {METHOD_LABELS[key]} | {cov_grade} ({fmt_pct(row.get('depth_coverage_pct'))}) | "
            f"{fmt_m(row.get('ray_median_error_m'))} — **{g}** | {trust} | "
            f"{fmt_pct(row.get('free_space_violation_pct'))} | {photo_s} |"
        )

    lines.extend([
        "",
        "### How to read the columns",
        "",
        "- **Surface coverage:** How much of the image gets a depth value. Low on plain carpet is normal for classic stereo.",
        "- **Match to laser:** Smaller is better (we report typical error in cm). Needs enough overlap to trust.",
        "- **Free-space warnings:** High % means depth looks *in front of* the laser hit (often a scale bug, especially for AI single-camera).",
        "- **Stereo photo match:** Lower is better (left vs right image consistency). Only for two-camera methods.",
        "",
        "---",
        "",
        "## Method details",
        "",
    ])

    for key in ("opencv", "dav2", "foundation"):
        if key not in methods:
            lines.append(f"### {METHOD_LABELS[key]}\n\n*Not run for this capture.*\n")
            continue
        row = methods[key]
        g, why = grade_ray(
            row.get("ray_median_error_m"),
            row.get("association_rate") or 0,
            row.get("free_space_violation_pct") or 0,
        )
        cov_g, cov_why = grade_coverage(row.get("depth_coverage_pct"))
        lines.extend([
            f"### {METHOD_LABELS[key]}",
            "",
            f"- **Coverage:** {cov_g} — {cov_why}",
            f"- **Laser agreement:** {g} — {why}",
            f"- **LiDAR points compared:** {row.get('associated_pixels', 0)} of "
            f"{row.get('projected_in_image', 0)} visible to the camera "
            f"({fmt_pct(100 * (row.get('association_rate') or 0))} association rate).",
            f"- **Within 5 cm of laser:** {fmt_pct(100 * (row.get('inlier_ratio') or 0))} of compared points.",
            "",
        ])

    if cross.get("pairwise_depth"):
        lines.extend(["---", "", "## Do the methods agree with each other?", ""])
        for pair, stats in cross["pairwise_depth"].items():
            label = pair.replace("_", " ").replace("opencv", "OpenCV").replace("dav2", "DA-V2").replace("foundation", "Foundation")
            med = stats.get("median_abs_diff_m")
            lines.append(
                f"- **{label}:** typical depth difference {fmt_m(med)} where both have data "
                f"({stats.get('overlap_pixels', 0):,} pixels)."
            )
        c = cross.get("consensus", {})
        if c.get("median_std_m") is not None:
            lines.append(
                f"- **All methods together:** typical disagreement {fmt_m(c['median_std_m'])} "
                f"(see heatmap below)."
            )

    lines.extend([
        "",
        "---",
        "",
        "## Figures for your report",
        "",
        "**Charts (run `python 08_generate_eval_charts.py`):**",
        f"- `{validation_dir / 'chart_scorecard.png'}` — green/yellow/red at-a-glance table",
        f"- `{validation_dir / 'chart_coverage_and_accuracy.png'}` — side-by-side bar charts",
        f"- `{validation_dir / 'chart_ray_error_histogram.png'}` — laser error distribution",
        f"- `{validation_dir / 'chart_error_vs_range.png'}` — |ΔZ| vs LiDAR range (median/p90 per bin)",
        f"- `{validation_dir / 'chart_photometric.png'}` — stereo consistency (if Foundation/OpenCV run)",
        "",
        "**From evaluation (`06`):**",
        f"- `{validation_dir / 'consensus_depth_std.png'}` — σ(Z) heatmap; caption in `evaluation_summary.json` → `consensus_map.caption`",
        f"- `{validation_dir / 'consensus_depth_std_on_rgb.png'}` — same on `depth/rgb1_rectified.png`",
    ])
    if consensus.get("summary"):
        lines.extend([
            "",
            f"**Cross-method consensus:** {consensus['summary']}",
        ])
    lines.extend([
        "",
        "**Scene previews:**",
        f"- `../depth/disparity_preview.png`, `depth_preview_dav2.png`",
        f"- `../overlays/lidar_overlay_rgb1.png`",
        "",
        "---",
        "",
        "## Choosing a method for your scenario",
        "",
        "| Your scene looks like… | Start with |",
        "|------------------------|------------|",
        "| Checkerboard / box / lots of texture | Classic stereo (OpenCV) |",
        "| Plain carpet, sand, uniform color | Depth Anything V2 or FoundationStereo |",
        "| Need full image filled with depth | Depth Anything V2 (if laser check is acceptable) |",
        "| Need best laser agreement on textured targets | Whichever wins **Closest to laser** above |",
        "| No Windows GPU | OpenCV + Depth Anything V2 (skip Foundation) |",
        "",
        "*Generated by `07_generate_eval_report.py`. Technical JSON: `evaluation_summary.json`.*",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Plain-language eval report")
    add_run_cli_arguments(parser)
    args = parser.parse_args()
    if handle_list_runs(args):
        return

    paths = resolve_run_paths(args.run)
    summary_path = paths.validation / "evaluation_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run: python 06_evaluate_run.py --run {args.run or 'latest'}"
        )

    summary = json.loads(summary_path.read_text())
    run_label = summary.get("run") or (paths.run_dir.name if paths.run_dir else "unknown")
    report = build_report(summary, run_label, paths.validation)

    out_md = paths.validation / "EVAL_REPORT.md"
    out_md.write_text(report, encoding="utf-8")
    print(f"Saved {out_md}")
    print()
    print(report[:2500])
    if len(report) > 2500:
        print("\n... (see full file)")


if __name__ == "__main__":
    main()
