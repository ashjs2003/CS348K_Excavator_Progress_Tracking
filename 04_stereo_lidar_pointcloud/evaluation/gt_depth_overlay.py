"""Overlay pixels whose metric depth matches ruler GT (.txt) or back-wall reference."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from evaluation.consensus_labels import method_label
from evaluation.depth_maps import discover_methods, load_metric_depth

# Project defaults (no CLI required during evaluation).
DEFAULT_GT_TOLERANCE_M = 0.05  # ±5 cm match band
WALL_GT_CM = 100.0  # back wall distance for all captures

# BGR for cv2 quick saves; RGB tuples for matplotlib overlays
METHOD_OVERLAY_RGB = {
    "opencv": (0.20, 0.95, 0.25),
    "dav2": (1.0, 0.55, 0.05),
    "dav2_gt": (0.95, 0.35, 0.15),
    "foundation": (0.85, 0.25, 0.95),
}

# Union / “any method” panel — distinct from per-method greens in METHOD_OVERLAY_RGB
RULER_MATCH_RGB = (0.10, 0.78, 0.55)  # teal-green: depth ≈ ruler distance
WALL_UNION_RGB = (0.15, 0.55, 1.0)
METHOD_PANEL_ORDER = ("opencv", "dav2", "dav2_gt", "foundation")
PANEL_SHORT_NAME = {
    "opencv": "OpenCV",
    "dav2": "DA-V2",
    "dav2_gt": "DA-V2 GT",
    "foundation": "Foundation",
}


def parse_pair_distance_txt(path: Path) -> float | None:
    """Parse pair_###.txt: value in cm when > 2, else meters. Returns None if missing/invalid."""
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text or text.upper() == "TEST":
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value / 100.0 if value > 2.0 else value


def load_gt_reference_for_run(
    run_dir: Path | None,
    repo_root: Path,
    *,
    tolerance_m: float | None = None,
    wall_gt_cm: float | None = None,
) -> dict:
    """
    Resolve manual GT for evaluation overlays.

    - Target distance: data/<scene>/pair_<id>.txt (cm when value > 2).
    - Wall distance: fixed WALL_GT_CM (100 cm) for all data.
    - Tolerance: fixed DEFAULT_GT_TOLERANCE_M (5 cm).
    """
    tol = DEFAULT_GT_TOLERANCE_M if tolerance_m is None else float(tolerance_m)
    wall_cm = WALL_GT_CM if wall_gt_cm is None else float(wall_gt_cm)
    txt_path = resolve_pair_txt_path(run_dir, Path(repo_root))
    target_gt_m = parse_pair_distance_txt(txt_path) if txt_path else None
    return {
        "txt_path": txt_path,
        "target_gt_m": target_gt_m,
        "target_gt_cm": None if target_gt_m is None else float(target_gt_m * 100.0),
        "wall_gt_m": wall_cm / 100.0,
        "wall_gt_cm": wall_cm,
        "tolerance_m": tol,
        "tolerance_cm": tol * 100.0,
    }


def resolve_pair_txt_path(run_dir: Path | None, repo_root: Path) -> Path | None:
    """Find data/<scene>/pair_<id>.txt from run_info or run folder layout."""
    if run_dir is None:
        return None
    info_path = run_dir / "run_info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        src = info.get("data_source")
        pid = info.get("data_pair_id")
        if src and pid:
            candidate = repo_root / str(src) / f"pair_{pid}.txt"
            if candidate.is_file():
                return candidate
    name = run_dir.name
    if name.startswith("pair_") and run_dir.parent.name not in ("runs", "outputs"):
        candidate = repo_root / "data" / run_dir.parent.name / f"{name}.txt"
        if candidate.is_file():
            return candidate
    return None


def match_mask(depth_m: np.ndarray, gt_m: float, tolerance_m: float) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    return valid & (np.abs(depth_m - gt_m) <= tolerance_m)


def _pct(mask: np.ndarray) -> float:
    return 100.0 * float(np.count_nonzero(mask)) / mask.size


def _blend_rgb(
    c1: tuple[float, float, float],
    c2: tuple[float, float, float],
    t: float = 0.5,
) -> tuple[float, float, float]:
    a = np.array(c1, dtype=np.float32)
    b = np.array(c2, dtype=np.float32)
    return tuple(np.clip((1.0 - t) * a + t * b, 0.0, 1.0))


def _coverage_stats_line(ruler_mask: np.ndarray, wall_mask: np.ndarray) -> str:
    return f"{_pct(ruler_mask):.1f}% ruler  |  {_pct(wall_mask):.1f}% wall"


def _paint_ruler_wall_overlay(
    rgb: np.ndarray,
    ruler_mask: np.ndarray,
    wall_mask: np.ndarray,
    ruler_color: tuple[float, float, float],
    *,
    alpha: float = 0.58,
) -> np.ndarray:
    """Highlight pixels whose depth matches ruler distance and/or back wall (no yellow)."""
    base = rgb.astype(np.float32) / 255.0
    display = 0.45 * base.copy()
    both = ruler_mask & wall_mask
    ruler_only = ruler_mask & ~both
    wall_only = wall_mask & ~both
    blend = _blend_rgb(ruler_color, WALL_UNION_RGB, 0.5)

    def paint(mask: np.ndarray, color: tuple[float, float, float]) -> None:
        if not np.any(mask):
            return
        c = np.array(color, dtype=np.float32)
        display[mask] = (1.0 - alpha) * display[mask] + alpha * c

    paint(wall_only, WALL_UNION_RGB)
    paint(ruler_only, ruler_color)
    paint(both, blend)
    return np.clip(display, 0, 1)


def _load_rectified_rgb(stereo_dir: Path, shape: tuple[int, int]) -> np.ndarray | None:
    from depth_layout import resolve_path

    found = resolve_path(Path(stereo_dir), None, "rgb1_rectified.png")
    path = found if found is not None else Path(stereo_dir) / "rgb1_rectified.png"
    if not path.is_file():
        return None
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    h, w = shape
    if bgr.shape[0] != h or bgr.shape[1] != w:
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _blend_overlay(
    base_rgb: np.ndarray,
    mask: np.ndarray,
    color_rgb: tuple[float, float, float],
    alpha: float = 0.55,
) -> np.ndarray:
    out = base_rgb.astype(np.float32).copy()
    if not np.any(mask):
        return out
    color = np.array(color_rgb, dtype=np.float32)
    m = mask
    out[m] = (1.0 - alpha) * out[m] + alpha * color * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _save_rgb_png(path: Path, rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _paint_union_overlay(
    rgb: np.ndarray,
    ruler_union: np.ndarray,
    wall_union: np.ndarray,
) -> np.ndarray:
    """All-methods panel: teal = any method at ruler distance, blue = wall."""
    return _paint_ruler_wall_overlay(rgb, ruler_union, wall_union, RULER_MATCH_RGB)


def _render_gt_reference_on_rgb_with_key(
    rgb: np.ndarray,
    target_union: np.ndarray,
    wall_union: np.ndarray,
    both_mask: np.ndarray,
    target_gt_m: float,
    wall_gt_m: float,
    tolerance_m: float,
    scene_label: str | None,
    out_path: Path,
) -> None:
    """Save gt_depth_reference_on_rgb.png with an on-image color key."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    display = _paint_union_overlay(rgb, target_union, wall_union)
    ruler_cm = target_gt_m * 100.0
    wall_cm = wall_gt_m * 100.0
    both_mask = target_union & wall_union
    blend_legend = _blend_rgb(RULER_MATCH_RGB, WALL_UNION_RGB, 0.5)
    h, w = display.shape[:2]
    fig_w = max(10.0, w / 100.0)
    fig_h = max(5.6, h / 100.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(display, interpolation="nearest")
    ax.set_axis_off()

    tol_cm = tolerance_m * 100.0
    title = f"{scene_label}  |  {ruler_cm:.0f} cm  |  wall {wall_cm:.0f} cm  |  +/-{tol_cm:.0f} cm"
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)

    handles = [
        Patch(facecolor=RULER_MATCH_RGB, edgecolor="#333", label=f"Ruler {ruler_cm:.0f} cm"),
        Patch(facecolor=WALL_UNION_RGB, edgecolor="#333", label=f"Wall {wall_cm:.0f} cm"),
    ]
    if np.any(both_mask):
        handles.append(Patch(facecolor=blend_legend, edgecolor="#333", label="Both"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)


def _render_gt_depth_method_grid_figure(
    rgb: np.ndarray,
    target_union: np.ndarray,
    wall_union: np.ndarray,
    both_mask: np.ndarray,
    per_method_masks: dict[str, dict[str, np.ndarray]],
    target_gt_m: float,
    wall_gt_m: float,
    tolerance_m: float,
    txt_path: Path | None,
    scene_label: str | None,
    out_path: Path,
) -> None:
    """
    gt_depth_reference_labeled.png — 2×2 grid: all methods + up to three depth sources.

    Highlights pixels whose metric depth is within ±tolerance of the ruler distance
    (pair_*.txt) and/or the fixed back-wall distance (100 cm).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ruler_cm = target_gt_m * 100.0
    wall_cm = wall_gt_m * 100.0
    tol_cm = tolerance_m * 100.0

    panels: list[tuple[str, np.ndarray, str]] = [
        (
            "All",
            _paint_union_overlay(rgb, target_union, wall_union),
            _coverage_stats_line(target_union, wall_union),
        ),
    ]
    for mid in METHOD_PANEL_ORDER:
        if mid not in per_method_masks:
            continue
        tm = per_method_masks[mid]["target"]
        wm = per_method_masks[mid]["wall"]
        color = METHOD_OVERLAY_RGB.get(mid, (0.75, 0.75, 0.75))
        name = PANEL_SHORT_NAME.get(mid, mid)
        panels.append(
            (
                name,
                _paint_ruler_wall_overlay(rgb, tm, wm, color),
                _coverage_stats_line(tm, wm),
            )
        )

    n_panels = len(panels)
    ncols = 2
    nrows = max(2, (n_panels + 1) // 2)  # 4 panels → 2×2; extra methods add rows
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.75 * ncols, 4.6 * nrows), squeeze=False)

    for idx in range(nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        if idx < n_panels:
            panel_title, display, stats_line = panels[idx]
            ax.imshow(display, interpolation="nearest")
            ax.set_title(f"{panel_title}: {stats_line}", fontsize=9, pad=4)
        ax.set_axis_off()

    run_line = scene_label or "run"
    fig.suptitle(
        f"{run_line}  |  ruler {ruler_cm:.0f} cm  |  wall {wall_cm:.0f} cm  |  +/-{tol_cm:.0f} cm",
        fontsize=10,
        fontweight="bold",
        y=0.98,
    )

    blend_legend = _blend_rgb(RULER_MATCH_RGB, WALL_UNION_RGB, 0.5)
    handles = [
        Patch(facecolor=RULER_MATCH_RGB, edgecolor="#333", label=f"Ruler {ruler_cm:.0f} cm"),
        Patch(facecolor=WALL_UNION_RGB, edgecolor="#333", label=f"Wall {wall_cm:.0f} cm"),
    ]
    if np.any(both_mask):
        handles.append(Patch(facecolor=blend_legend, edgecolor="#333", label="Both"))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.12, wspace=0.05)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.06)
    plt.close(fig)


def save_gt_depth_reference_overlays(
    stereo_dir: Path,
    geometry: dict,
    validation_dir: Path,
    *,
    run_dir: Path | None,
    repo_root: Path,
    tolerance_m: float | None = None,
    wall_gt_cm: float | None = None,
    scene_label: str | None = None,
) -> dict:
    """
    Save GT reference overlays on rgb1_rectified.

    - Combined figure: union of all methods vs target (.txt) and wall (100 cm).
    - Per-method PNGs: green = target, blue = wall.
    """
    stereo_dir = Path(stereo_dir)
    validation_dir = Path(validation_dir)
    gt_ref = load_gt_reference_for_run(
        run_dir, repo_root, tolerance_m=tolerance_m, wall_gt_cm=wall_gt_cm
    )
    txt_path = gt_ref["txt_path"]
    target_gt_m = gt_ref["target_gt_m"]
    wall_gt_m = gt_ref["wall_gt_m"]
    wall_gt_cm = gt_ref["wall_gt_cm"]
    tolerance_m = gt_ref["tolerance_m"]

    methods = discover_methods(stereo_dir)
    if not methods:
        return {"saved": False, "reason": "no depth methods found"}

    depths: dict[str, np.ndarray] = {}
    for name in methods:
        z = load_metric_depth(stereo_dir, name, geometry)
        if z is not None:
            depths[name] = z

    if not depths:
        return {"saved": False, "reason": "no loadable depth maps"}

    h, w = next(iter(depths.values())).shape
    rgb = _load_rectified_rgb(stereo_dir, (h, w))
    if rgb is None:
        return {"saved": False, "reason": "rgb1_rectified.png missing in depth/"}

    if target_gt_m is None:
        return {
            "saved": False,
            "reason": "no valid pair_*.txt target distance (missing or non-numeric)",
            "txt_path": str(txt_path) if txt_path else None,
            "wall_gt_m": wall_gt_m,
            "wall_gt_cm": wall_gt_cm,
            "tolerance_m": tolerance_m,
            "tolerance_cm": gt_ref["tolerance_cm"],
            "settings_source": {
                "target": "data/<scene>/pair_<id>.txt",
                "wall_cm": WALL_GT_CM,
                "tolerance_cm": DEFAULT_GT_TOLERANCE_M * 100.0,
            },
        }

    per_method: dict[str, dict] = {}
    per_method_masks: dict[str, dict[str, np.ndarray]] = {}
    target_union = np.zeros((h, w), dtype=bool)
    wall_union = np.zeros((h, w), dtype=bool)

    for name, depth_m in depths.items():
        tm = match_mask(depth_m, target_gt_m, tolerance_m)
        wm = match_mask(depth_m, wall_gt_m, tolerance_m)
        per_method_masks[name] = {"target": tm, "wall": wm}
        target_union |= tm
        wall_union |= wm
        per_method[name] = {
            "frac_target_pct": _pct(tm),
            "frac_wall_pct": _pct(wm),
            "n_target_pixels": int(np.count_nonzero(tm)),
            "n_wall_pixels": int(np.count_nonzero(wm)),
        }

    both_mask = target_union & wall_union

    combined_path = validation_dir / "gt_depth_reference_on_rgb.png"
    _render_gt_reference_on_rgb_with_key(
        rgb,
        target_union,
        wall_union,
        both_mask,
        target_gt_m,
        wall_gt_m,
        tolerance_m,
        scene_label,
        combined_path,
    )

    all_methods_rgb = rgb.copy()
    for name in per_method_masks:
        all_methods_rgb = _blend_overlay(
            all_methods_rgb,
            per_method_masks[name]["target"],
            METHOD_OVERLAY_RGB.get(name, (0.75, 0.75, 0.75)),
            alpha=0.48,
        )
    all_methods_rgb = _blend_overlay(all_methods_rgb, wall_union, WALL_UNION_RGB, alpha=0.35)
    all_methods_path = validation_dir / "gt_depth_all_methods_on_rgb.png"
    _save_rgb_png(all_methods_path, all_methods_rgb)

    for name, masks in per_method_masks.items():
        method_rgb = rgb.copy()
        method_rgb = _blend_overlay(method_rgb, masks["wall"], WALL_UNION_RGB)
        method_rgb = _blend_overlay(
            method_rgb, masks["target"], METHOD_OVERLAY_RGB.get(name, (0.75, 0.75, 0.75))
        )
        _save_rgb_png(validation_dir / f"gt_depth_match_{name}_on_rgb.png", method_rgb)

    figure_path = validation_dir / "gt_depth_reference_labeled.png"
    _render_gt_depth_method_grid_figure(
        rgb,
        target_union,
        wall_union,
        both_mask,
        per_method_masks,
        target_gt_m,
        wall_gt_m,
        tolerance_m,
        txt_path,
        scene_label,
        figure_path,
    )

    return {
        "saved": True,
        "target_gt_m": float(target_gt_m),
        "target_gt_cm": float(target_gt_m * 100.0),
        "wall_gt_m": float(wall_gt_m),
        "wall_gt_cm": float(wall_gt_cm),
        "tolerance_m": float(tolerance_m),
        "tolerance_cm": float(gt_ref["tolerance_cm"]),
        "txt_path": str(txt_path) if txt_path else None,
        "settings_source": {
            "target": "data/<scene>/pair_<id>.txt",
            "wall_cm": WALL_GT_CM,
            "tolerance_cm": DEFAULT_GT_TOLERANCE_M * 100.0,
        },
        "combined_on_rgb": str(combined_path),
        "all_methods_on_rgb": str(all_methods_path),
        "labeled_figure": str(figure_path),
        "per_method_on_rgb": {
            name: str(validation_dir / f"gt_depth_match_{name}_on_rgb.png") for name in per_method_masks
        },
        "frac_target_any_method_pct": _pct(target_union),
        "frac_wall_any_method_pct": _pct(wall_union),
        "frac_both_any_method_pct": _pct(both_mask),
        "per_method": per_method,
        "methods": list(depths.keys()),
    }
