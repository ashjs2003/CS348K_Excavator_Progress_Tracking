"""
Generate easy-to-read charts from evaluation outputs (for slides / reports).

Run after 06_evaluate_run.py:
    python 08_generate_eval_charts.py --run latest
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.error_vs_range import (
    binned_error_vs_range,
    load_ray_pairs_csv,
    method_ray_csv_suffix,
    shared_bin_edges,
)
from output_runs import add_run_cli_arguments, handle_list_runs, resolve_run_paths

METHOD_ORDER = ("opencv", "dav2", "foundation")
METHOD_LABELS = {
    "opencv": "OpenCV\n(classic stereo)",
    "dav2": "Depth Anything V2\n(AI mono)",
    "foundation": "FoundationStereo\n(AI stereo)",
}
METHOD_COLORS = {
    "opencv": "#4C78A8",
    "dav2": "#F58518",
    "foundation": "#54A24B",
}
MIN_ASSOCIATION_RATE = 0.10


def load_summary(validation_dir: Path) -> dict:
    path = validation_dir / "evaluation_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run: python 06_evaluate_run.py --run <id>")
    return json.loads(path.read_text())


def methods_in_summary(summary: dict) -> list[str]:
    return [k for k in METHOD_ORDER if k in summary.get("methods", {})]


def load_ray_errors_csv(path: Path) -> np.ndarray:
    if not path.is_file():
        return np.array([])
    errors = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("ray_error_m", "")
            if val == "" or val is None:
                continue
            try:
                e = float(val)
                if np.isfinite(e):
                    errors.append(e)
            except ValueError:
                pass
    return np.asarray(errors, dtype=float)


def chart_coverage_and_accuracy(summary: dict, out_path: Path, scene_label: str):
    """Two bar charts: coverage % and laser error (cm)."""
    methods = methods_in_summary(summary)
    if not methods:
        return

    labels = [METHOD_LABELS[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]
    coverage = [summary["methods"][m].get("depth_coverage_pct") or 0 for m in methods]
    ray_cm = []
    trusted = []
    for m in methods:
        row = summary["methods"][m]
        med = row.get("ray_median_error_m")
        assoc = row.get("association_rate") or 0
        trusted.append(assoc >= MIN_ASSOCIATION_RATE)
        ray_cm.append((med * 100) if med is not None else np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f"Depth methods compared — {scene_label}", fontsize=14, fontweight="bold", y=1.02)

    ax = axes[0]
    bars = ax.bar(labels, coverage, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Percent of image with depth")
    ax.set_title("Surface coverage (higher = more complete)")
    ax.set_ylim(0, 105)
    ax.axhline(50, color="#888", linestyle="--", linewidth=0.8, alpha=0.6)
    for bar, val in zip(bars, coverage):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{val:.0f}%", ha="center", fontsize=10)

    ax = axes[1]
    x = np.arange(len(methods))
    bars = ax.bar(x, ray_cm, color=colors, edgecolor="white", linewidth=1.2)
    for i, (bar, ok) in enumerate(zip(bars, trusted)):
        if not ok:
            bar.set_hatch("//")
            bar.set_alpha(0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Typical gap vs laser (cm)")
    ax.set_title("Match to laser (lower = better)")
    ax.axhline(15, color="#2CA02C", linestyle="--", linewidth=1, label="~15 cm target")
    ax.axhline(35, color="#D62728", linestyle="--", linewidth=1, alpha=0.7)
    for i, val in enumerate(ray_cm):
        if np.isfinite(val):
            note = f"{val:.0f}" if trusted[i] else f"{val:.0f}*"
            ax.text(i, val + 1.5, note, ha="center", fontsize=10)
    ax.text(0.02, 0.02, "* too few points to trust this bar", transform=ax.transAxes, fontsize=8, color="#555")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_error_vs_range(
    validation_dir: Path,
    summary: dict,
    out_path: Path,
    scene_label: str,
    n_bins: int = 8,
):
    """Median |Z_est - Z_lidar| vs Z_lidar range; one line per method (shared bin edges)."""
    methods = methods_in_summary(summary)
    if not methods:
        return False

    series = []
    all_ranges = []
    for m in methods:
        csv_path = validation_dir / f"lidar_ray_per_point{method_ray_csv_suffix(m)}.csv"
        r_m, e_m = load_ray_pairs_csv(csv_path)
        if len(r_m):
            all_ranges.append(r_m)
            series.append((m, r_m, e_m))

    if not series:
        return False

    edges = summary.get("error_vs_range_bin_edges_m")
    if edges:
        bin_edges = np.asarray(edges, dtype=float)
    else:
        bin_edges = shared_bin_edges(all_ranges, n_bins=n_bins)
        if bin_edges is None:
            return False

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        f"Depth error vs LiDAR range — {scene_label}",
        fontsize=13,
        fontweight="bold",
    )

    for m, r_m, e_m in series:
        evr = binned_error_vs_range(r_m, e_m, bin_edges)
        centers, med_cm, p90_cm, counts = [], [], [], []
        for b in evr["bins"]:
            if b["count"] == 0 or b["median_error_m"] is None:
                continue
            centers.append(b["range_center_m"])
            med_cm.append(100.0 * b["median_error_m"])
            p90_cm.append(100.0 * b["p90_error_m"])
            counts.append(b["count"])
        if not centers:
            continue
        centers = np.asarray(centers)
        med_cm = np.asarray(med_cm)
        p90_cm = np.asarray(p90_cm)
        counts = np.asarray(counts)
        color = METHOD_COLORS[m]
        label = METHOD_LABELS[m].replace("\n", " ")
        ax.plot(centers, med_cm, "o-", color=color, linewidth=2, markersize=7, label=f"{label} · median")
        ax.plot(
            centers,
            p90_cm,
            "--",
            color=color,
            linewidth=1.2,
            alpha=0.75,
            label=f"{label} · p90",
        )
        for x, y, n in zip(centers, med_cm, counts):
            ax.annotate(
                str(n),
                (x, y),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=7,
                color=color,
            )

    ax.set_xlabel("LiDAR depth Z in rectified RGB1 frame (m)")
    ax.set_ylabel("|Z_est − Z_lidar| (cm)")
    ax.set_title("n above each point = laser checks in that distance band")

    from matplotlib.lines import Line2D

    style_handles = [
        Line2D([0], [0], color="#555", linestyle="-", marker="o", markersize=6, label="Solid = median"),
        Line2D(
            [0],
            [0],
            color="#555",
            linestyle="--",
            linewidth=1.5,
            label="Dashed = p90 (90% of errors below)",
        ),
    ]
    method_handles, method_labels = ax.get_legend_handles_labels()
    ax.axhline(5, color="#888", linestyle=":", linewidth=0.9, alpha=0.7)
    ax.axhline(15, color="#888", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.legend(
        style_handles + method_handles,
        [h.get_label() for h in style_handles] + list(method_labels),
        loc="best",
        fontsize=8,
        ncol=1,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def chart_ray_histograms(validation_dir: Path, summary: dict, out_path: Path, scene_label: str):
    methods = methods_in_summary(summary)
    suffix_map = {"opencv": "", "dav2": "_dav2", "foundation": "_foundation"}
    data = []
    for m in methods:
        csv_path = validation_dir / f"lidar_ray_per_point{suffix_map[m]}.csv"
        errs = load_ray_errors_csv(csv_path)
        if len(errs):
            data.append((m, errs * 100))

    if not data:
        return

    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Laser vs depth error distribution — {scene_label}", fontsize=13, fontweight="bold")

    all_cm = np.concatenate([d[1] for d in data])
    hi = min(120, float(np.percentile(all_cm, 98)) + 5) if len(all_cm) else 80
    bins = np.linspace(0, hi, 25)

    for ax, (m, errs) in zip(axes, data):
        ax.hist(errs, bins=bins, color=METHOD_COLORS[m], alpha=0.85, edgecolor="white")
        ax.axvline(np.median(errs), color="black", linestyle="-", linewidth=2, label=f"median {np.median(errs):.0f} cm")
        ax.set_xlabel("Error (cm)")
        ax.set_title(METHOD_LABELS[m].replace("\n", " "))
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Number of laser points")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_scorecard(summary: dict, out_path: Path, scene_label: str):
    """Color-coded table image: Good / Fair / Poor per method."""
    methods = methods_in_summary(summary)

    def grade_coverage(pct):
        if pct >= 70:
            return "High", "#C6EFCE"
        if pct >= 25:
            return "Medium", "#FFEB9C"
        return "Low", "#FFC7CE"

    def grade_ray(med, assoc, fs):
        if assoc < MIN_ASSOCIATION_RATE:
            return "Low trust", "#E0E0E0"
        if fs > 15:
            return "Poor", "#FFC7CE"
        if med is None:
            return "—", "#FFFFFF"
        cm = med * 100
        if cm <= 15:
            return "Good", "#C6EFCE"
        if cm <= 35:
            return "Fair", "#FFEB9C"
        return "Poor", "#FFC7CE"

    rows = ["Surface coverage", "Laser match", "Trust laser check?"]
    col_labels = [METHOD_LABELS[m].replace("\n", " ") for m in methods]
    cell_text = []
    cell_colors = []

    for row_name in rows:
        texts, colors = [], []
        for m in methods:
            r = summary["methods"][m]
            if row_name == "Surface coverage":
                g, c = grade_coverage(r.get("depth_coverage_pct") or 0)
                texts.append(f"{g}\n({r.get('depth_coverage_pct', 0):.0f}%)")
            elif row_name == "Laser match":
                g, c = grade_ray(r.get("ray_median_error_m"), r.get("association_rate") or 0, r.get("free_space_violation_pct") or 0)
                med = r.get("ray_median_error_m")
                extra = f"{med*100:.0f} cm" if med is not None else "n/a"
                texts.append(f"{g}\n({extra})")
            else:
                ok = (r.get("association_rate") or 0) >= MIN_ASSOCIATION_RATE
                texts.append("Yes" if ok else "No")
                colors.append("#C6EFCE" if ok else "#FFC7CE")
                continue
            colors.append(c)
        cell_text.append(texts)
        cell_colors.append(colors)

    fig, ax = plt.subplots(figsize=(2 + 2.2 * len(methods), 2.8))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=rows,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    ax.set_title(f"At-a-glance scorecard — {scene_label}", fontsize=13, fontweight="bold", pad=20)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_photometric(validation_dir: Path, out_path: Path, scene_label: str) -> bool:
    path = validation_dir / "photometric_reprojection.json"
    if not path.is_file():
        return False
    phot = json.loads(path.read_text())
    methods = [m for m in ("opencv", "foundation") if m in phot]
    if not methods:
        return False

    labels = [METHOD_LABELS[m].replace("\n", " ") for m in methods]
    vals = [phot[m].get("mean_photometric_error") for m in methods]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals, color=[METHOD_COLORS[m] for m in methods], edgecolor="white")
    ax.set_ylabel("Mean left–right gray difference (lower = better)")
    ax.set_title(f"Stereo image consistency — {scene_label}")
    for i, v in enumerate(vals):
        if v is not None:
            ax.text(i, v + 0.3, f"{v:.1f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation charts")
    add_run_cli_arguments(parser)
    args = parser.parse_args()
    if handle_list_runs(args):
        return

    paths = resolve_run_paths(args.run)
    summary = load_summary(paths.validation)
    scene = summary.get("run", paths.run_dir.name if paths.run_dir else "run")
    if "carpet" in scene or paths.run_dir:
        label = paths.run_dir.name.split("_")[-1] if paths.run_dir else scene
    else:
        label = scene

    out_dir = paths.validation
    charts = []

    p1 = out_dir / "chart_coverage_and_accuracy.png"
    chart_coverage_and_accuracy(summary, p1, label)
    charts.append(p1)

    p2 = out_dir / "chart_ray_error_histogram.png"
    chart_ray_histograms(out_dir, summary, p2, label)
    if p2.exists():
        charts.append(p2)

    p_range = out_dir / "chart_error_vs_range.png"
    if chart_error_vs_range(out_dir, summary, p_range, label):
        charts.append(p_range)

    p3 = out_dir / "chart_scorecard.png"
    chart_scorecard(summary, p3, label)
    charts.append(p3)

    p4 = out_dir / "chart_photometric.png"
    if chart_photometric(out_dir, p4, label):
        charts.append(p4)

    print(f"Scene: {label}")
    print("Saved charts:")
    for p in charts:
        print(f"  {p}")
    print("\nAlso use (from 06):")
    print(f"  {out_dir / 'consensus_depth_std.png'}  — disagreement heatmap")
    print(f"  {out_dir / 'consensus_depth_std_on_rgb.png'}  — same, on rectified RGB1")
    print(f"  {out_dir / 'EVAL_REPORT.md'}  — plain-language write-up")


if __name__ == "__main__":
    main()
