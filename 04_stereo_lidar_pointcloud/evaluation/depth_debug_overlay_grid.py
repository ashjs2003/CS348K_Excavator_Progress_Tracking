"""RGB overlays for ROI − flap depth heuristic (cardboard S / M / L debug grid)."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from evaluation.box_volume_heuristic import (
    DEFAULT_OUTSIDE_CLOSER_MARGIN_M,
    DEFAULT_OUTSIDE_CONSISTENCY_M,
    box_size_class,
    gt_nominal_depth_cm,
    outside_flap_depth,
)
from evaluation.depth_maps import discover_methods, load_metric_depth, load_or_compute_stereo_geometry
from evaluation.lidar_polar_setup_grid import (
    CARDBOARD_DISTANCE_COLUMNS_CM,
    CARDBOARD_SML_ROWS,
    CARDBOARD_SML_SCENES,
    index_cardboard_sml_cells,
)
from evaluation.roi_error_vs_distance import METHOD_LABELS, METHOD_ORDER
from evaluation.roi_gt_compare import load_roi_polygon, polygon_mask

# Region outline colors (matplotlib RGB)
_ROI_COLOR = "#2563eb"
_FLAP_COLOR = "#dc2626"

DEFAULT_RING_PAD_PX = 24
DEFAULT_RING_INNER_GAP_PX = 4
DEFAULT_CLOSER_MARGIN_M = DEFAULT_OUTSIDE_CLOSER_MARGIN_M
DEFAULT_CONSISTENCY_M = DEFAULT_OUTSIDE_CONSISTENCY_M


def bbox_from_polygon(poly: np.ndarray) -> tuple[int, int, int, int]:
    xs = poly[:, 0]
    ys = poly[:, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def rectangle_mask(h: int, w: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    if x1 >= x0 and y1 >= y0:
        m[y0 : y1 + 1, x0 : x1 + 1] = True
    return m


def outside_ring_mask(
    h: int,
    w: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    pad_px: int,
    inner_gap_px: int,
) -> np.ndarray:
    ox0 = max(0, x0 - pad_px)
    oy0 = max(0, y0 - pad_px)
    ox1 = min(w - 1, x1 + pad_px)
    oy1 = min(h - 1, y1 + pad_px)
    outer = rectangle_mask(h, w, ox0, oy0, ox1, oy1)
    ix0 = max(0, x0 - inner_gap_px)
    iy0 = max(0, y0 - inner_gap_px)
    ix1 = min(w - 1, x1 + inner_gap_px)
    iy1 = min(h - 1, y1 + inner_gap_px)
    inner = rectangle_mask(h, w, ix0, iy0, ix1, iy1)
    return outer & ~inner


def _load_rgb_bgr(run_dir: Path) -> np.ndarray | None:
    """Rectified left RGB (same pixels as ROI annotation and depth maps)."""
    depth_dir = run_dir / "depth"
    for rel in ("shared/rgb1_rectified.png", "rgb1_rectified.png"):
        p = depth_dir / rel
        if p.is_file():
            img = cv2.imread(str(p))
            if img is not None:
                return img
    cap = run_dir / "capture" / "rgb1.png"
    if cap.is_file():
        return cv2.imread(str(cap))
    return None


def _masks_for_pair(
    depth_shape: tuple[int, int],
    poly: np.ndarray,
    img_size: tuple[int, int],
    *,
    ring_pad_px: int,
    ring_inner_gap_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """roi_mask, ring_mask, polygon_xy int32 at depth resolution."""
    h_img, w_img = img_size[1], img_size[0]
    dh, dw = depth_shape
    x0, y0, x1, y1 = bbox_from_polygon(poly)
    if (dh, dw) != (h_img, w_img):
        sx, sy = dw / float(w_img), dh / float(h_img)
        poly_s = poly.astype(np.float64).copy()
        poly_s[:, 0] *= sx
        poly_s[:, 1] *= sy
        poly_d = poly_s.astype(np.int32)
        roi_m = polygon_mask((dh, dw), poly_d)
        ring_m = outside_ring_mask(
            dh,
            dw,
            int(x0 * sx),
            int(y0 * sy),
            int(x1 * sx),
            int(y1 * sy),
            pad_px=ring_pad_px,
            inner_gap_px=ring_inner_gap_px,
        )
    else:
        poly_d = poly.astype(np.int32)
        roi_m = polygon_mask((h_img, w_img), poly_d)
        ring_m = outside_ring_mask(
            h_img, w_img, x0, y0, x1, y1, pad_px=ring_pad_px, inner_gap_px=ring_inner_gap_px
        )
    return roi_m, ring_m, poly_d


def render_depth_debug_panel(
    rgb_bgr: np.ndarray,
    depth_m: np.ndarray,
    roi_mask: np.ndarray,
    ring_mask: np.ndarray,
    depth_part: dict,
    *,
    method_label: str,
    gt_depth_cm: float,
    diff_vmax_cm: float = 20.0,
    target_h: int = 300,
) -> np.ndarray:
    """
    ROI (blue outline) and flap pixels used for z_out (red outline).
    Heatmap: depth(cm) − z_flap(cm) inside those regions (turbo).
    """
    flap_mask = depth_part.get("consistent_ring_mask")
    if flap_mask is None:
        flap_mask = np.zeros_like(ring_mask, dtype=bool)
    else:
        flap_mask = flap_mask.astype(bool)

    valid = np.isfinite(depth_m) & (depth_m > 0)
    depth_cm = np.where(valid, depth_m.astype(np.float64) * 100.0, np.nan)
    z_out_cm = float(depth_part["z_outside_m"] * 100.0)
    z_in_cm = float(depth_part["z_inside_m"] * 100.0)
    box_depth_cm = float(depth_part["depth_m"] * 100.0)

    diff_cm = depth_cm - z_out_cm
    region = (roi_mask | flap_mask) & valid

    rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    h, w = rgb_rgb.shape[:2]
    dpi = 100
    fig_w, fig_h = w / dpi, h / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(np.clip(rgb_rgb * 0.35, 0, 1))
    if np.any(region):
        vmax = float(diff_vmax_cm)
        masked = np.ma.masked_where(~region, diff_cm)
        ax.imshow(masked, cmap="turbo", vmin=0.0, vmax=vmax, alpha=0.92, interpolation="nearest")

    if np.any(roi_mask):
        ax.contour(roi_mask.astype(float), levels=[0.5], colors=[_ROI_COLOR], linewidths=2.2)
    if np.any(flap_mask):
        ax.contour(flap_mask.astype(float), levels=[0.5], colors=[_FLAP_COLOR], linewidths=2.2)

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.axis("off")
    ax.text(
        0.01,
        0.99,
        f"{method_label}\n"
        f"z_roi={z_in_cm:.1f}  z_flap={z_out_cm:.1f}  Δ={box_depth_cm:.1f} cm\n"
        f"GT Δ={gt_depth_cm:.0f} cm",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        ha="left",
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55, edgecolor="none"),
    )

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    plt.close(fig)
    out_rgb = buf[:, :, :3]
    new_w = max(1, int(w * target_h / max(h, 1)))
    out_rgb = cv2.resize(out_rgb, (new_w, target_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def build_capture_panels(
    cap: dict,
    runs_root: Path,
    data_root: Path,
    *,
    ring_pad_px: int = DEFAULT_RING_PAD_PX,
    ring_inner_gap_px: int = DEFAULT_RING_INNER_GAP_PX,
    outside_closer_margin_m: float = DEFAULT_CLOSER_MARGIN_M,
    consistency_tol_m: float = DEFAULT_CONSISTENCY_M,
    diff_vmax_cm: float = 20.0,
) -> list[np.ndarray] | None:
    scene = cap["scene"]
    pair_id = cap["pair_id"]
    run_dir = runs_root / scene / f"pair_{pair_id}"
    roi_path = data_root / scene / f"pair_{pair_id}_roi.json"
    if not run_dir.is_dir() or not roi_path.is_file():
        return None

    rgb = _load_rgb_bgr(run_dir)
    if rgb is None:
        return None

    roi = load_roi_polygon(roi_path)
    poly = np.asarray(roi["polygon_xy"], dtype=np.int32)
    depth_dir = run_dir / "depth"
    geometry = load_or_compute_stereo_geometry(
        depth_dir,
        run_dir / "capture" / "rgb1.png",
        run_dir / "capture" / "rgb2.png",
    )
    # stereo_geometry image_size is (width, height), same frame as rgb1_rectified / ROI JSON
    w_img, h_img = geometry["image_size"]
    size = box_size_class(scene)
    gt_d = gt_nominal_depth_cm(size) if size else 0.0
    methods = [m for m in METHOD_ORDER if m in discover_methods(depth_dir)]
    panels: list[np.ndarray] = []

    for method in METHOD_ORDER:
        if method not in methods:
            panels.append(
                np.full((280, 320, 3), 220, dtype=np.uint8)
            )
            continue
        try:
            depth = load_metric_depth(depth_dir, method, geometry)
        except Exception:
            panels.append(np.full((280, 320, 3), 200, dtype=np.uint8))
            continue
        if depth is None:
            panels.append(np.full((280, 320, 3), 200, dtype=np.uint8))
            continue

        dh, dw = depth.shape[:2]
        roi_m, ring_m, poly_d = _masks_for_pair(
            (dh, dw),
            poly,
            (w_img, h_img),
            ring_pad_px=ring_pad_px,
            ring_inner_gap_px=ring_inner_gap_px,
        )

        part = outside_flap_depth(
            depth,
            roi_m,
            ring_m,
            outside_closer_margin_m=outside_closer_margin_m,
            consistency_tol_m=consistency_tol_m,
            return_masks=True,
        )
        if part is None:
            blank = np.full((280, 320, 3), 180, dtype=np.uint8)
            cv2.putText(blank, f"{METHOD_LABELS[method]}: fail", (12, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            panels.append(blank)
            continue

        part["polygon_xy"] = poly_d
        rgb_use = rgb
        if (dh, dw) != rgb.shape[:2]:
            rgb_use = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_AREA)

        panels.append(
            render_depth_debug_panel(
                rgb_use,
                depth,
                roi_m,
                ring_m,
                part,
                method_label=METHOD_LABELS[method],
                gt_depth_cm=gt_d,
                diff_vmax_cm=diff_vmax_cm,
            )
        )

    return panels


def _tile_2x2(panels: list[np.ndarray]) -> np.ndarray:
    pw = max(p.shape[1] for p in panels)
    ph = max(p.shape[0] for p in panels)
    norm = []
    for p in panels:
        canvas = np.full((ph, pw, 3), 240, dtype=np.uint8)
        canvas[: p.shape[0], : p.shape[1]] = p
        norm.append(canvas)
    top = np.hstack([norm[0], norm[1]])
    bot = np.hstack([norm[2], norm[3]])
    return np.vstack([top, bot])


def render_depth_debug_grid(
    data_root: Path,
    runs_root: Path,
    out_path: Path,
    *,
    distance_columns_cm: list[float] | None = None,
) -> dict:
    distance_columns_cm = distance_columns_cm or list(CARDBOARD_DISTANCE_COLUMNS_CM)
    cells = index_cardboard_sml_cells(data_root, list(CARDBOARD_SML_SCENES), distance_columns_cm)
    diff_vmax = 20.0

    n_rows = len(CARDBOARD_SML_ROWS)
    n_cols = len(distance_columns_cm)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.8 * n_cols, 2.35 * n_rows), squeeze=False
    )

    for ri, (size, ang) in enumerate(CARDBOARD_SML_ROWS):
        for ci, d_cm in enumerate(distance_columns_cm):
            ax = axes[ri, ci]
            ax.axis("off")
            cap = cells.get((size, ang, float(d_cm)))
            if cap is None:
                ax.text(0.5, 0.5, "no capture", ha="center", va="center", fontsize=9, color="#888")
                continue
            panels = build_capture_panels(cap, runs_root, data_root, diff_vmax_cm=diff_vmax)
            if panels is None or len(panels) < 4:
                ax.text(0.5, 0.5, "missing data", ha="center", va="center", fontsize=9, color="#888")
                continue
            tile = _tile_2x2(panels[:4])
            ax.imshow(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
            if ci == 0:
                ang_lbl = f"{int(ang)}°" if ang == int(ang) else f"{ang:.0f}°"
                ax.text(
                    -0.06,
                    0.5,
                    f"{size} · {ang_lbl}",
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight="bold",
                    va="center",
                    ha="right",
                )

    fig.suptitle("Depth debug: ROI vs flap (heatmap = depth − z_flap)", fontsize=12, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.94,
        "Blue = ROI (z_in)   Red = outside flap (z_out)   Color = stereo depth above flap reference  "
        f"· 2×2: OpenCV | DA-V2 | DA-V2 GT | Foundation",
        ha="center",
        fontsize=8,
        color="#444",
    )
    fig.subplots_adjust(left=0.06, right=0.88, top=0.90, bottom=0.05, wspace=0.06, hspace=0.12)

    sm = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(vmin=0, vmax=diff_vmax))
    sm.set_array([])
    cax = fig.add_axes([0.90, 0.12, 0.018, 0.76])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Depth − z_flap (cm)", fontsize=9)
    cbar.ax.text(
        0.5,
        1.02,
        f"box Δ = median(ROI)−median(flap)",
        transform=cbar.ax.transAxes,
        ha="center",
        fontsize=7,
        color="#444",
    )

    for ci, d_cm in enumerate(distance_columns_cm):
        pos = axes[0, ci].get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2,
            pos.y1 + 0.012,
            f"{int(d_cm)} cm",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="medium",
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    return {"cells": cells, "distance_columns_cm": distance_columns_cm}
