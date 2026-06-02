"""Save per-pixel depth disagreement across methods as heatmap PNGs (standalone + on RGB)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from evaluation.consensus_labels import (
    ONLY_METHOD_RGBA,
    build_consensus_caption,
)
from evaluation.depth_maps import METHODS, load_metric_depth


def _compute_consensus(stereo_dir: Path, geometry: dict) -> dict | None:
    stereo_dir = Path(stereo_dir)
    depths = []
    labels = []
    for name in METHODS:
        z = load_metric_depth(stereo_dir, name, geometry)
        if z is not None:
            depths.append(z.astype(np.float64))
            labels.append(name)

    if len(depths) < 2:
        return {
            "ok": False,
            "reason": "need at least two method depth maps",
            "methods": labels,
        }

    stack = np.stack(depths, axis=0)
    valid = np.isfinite(stack) & (stack > 0)
    count = np.sum(valid, axis=0).astype(np.int32)
    std_map = np.full(stack.shape[1:], np.nan, dtype=np.float64)
    mask2 = count >= 2
    if np.any(mask2):
        std_map[mask2] = np.nanstd(stack[:, mask2], axis=0)

    if not np.any(mask2):
        return {
            "ok": False,
            "reason": "no pixels with >=2 valid methods",
            "methods": labels,
        }

    alone_masks = {}
    for i, name in enumerate(labels):
        alone_masks[name] = valid[i] & (count == 1)

    vals_cm = std_map[mask2] * 100.0
    p50 = float(np.percentile(vals_cm, 50))
    p95 = float(np.percentile(vals_cm, 95))
    vmax = max(p95, 5.0)

    n_pix = count.size
    frac_compared = 100.0 * float(np.count_nonzero(mask2)) / n_pix
    frac_single = 100.0 * float(np.count_nonzero(count == 1)) / n_pix
    frac_none = 100.0 * float(np.count_nonzero(count == 0)) / n_pix
    frac_under_5cm = 100.0 * float(np.count_nonzero(vals_cm < 5.0)) / len(vals_cm)
    frac_under_15cm = 100.0 * float(np.count_nonzero(vals_cm < 15.0)) / len(vals_cm)

    alone_pct = {
        name: 100.0 * float(np.count_nonzero(alone_masks[name])) / n_pix for name in labels
    }

    return {
        "ok": True,
        "methods": labels,
        "std_map": std_map,
        "count": count,
        "mask2": mask2,
        "alone_masks": alone_masks,
        "alone_pct": alone_pct,
        "display_cm": std_map * 100.0,
        "p50": p50,
        "p95": p95,
        "vmax": vmax,
        "frac_compared": frac_compared,
        "frac_single": frac_single,
        "frac_none": frac_none,
        "frac_under_5cm": frac_under_5cm,
        "frac_under_15cm": frac_under_15cm,
    }


def _load_rectified_rgb(stereo_dir: Path, target_shape: tuple[int, int]) -> np.ndarray | None:
    from depth_layout import resolve_path

    found = resolve_path(Path(stereo_dir), None, "rgb1_rectified.png")
    path = found if found is not None else Path(stereo_dir) / "rgb1_rectified.png"
    if not path.is_file():
        return None
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    h, w = target_shape
    if bgr.shape[0] != h or bgr.shape[1] != w:
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _save_figure(fig, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)


def _legend_handles_standalone(caption, norm, cmap_module):
    from matplotlib.patches import Patch

    handles = [Patch(facecolor="#3a3a3a", label=caption.legend_standalone[0])]
    if len(caption.legend_standalone) > 1:
        handles.append(Patch(facecolor=cmap_module(norm(5)), label=caption.legend_standalone[1]))
    return handles


def _legend_handles_overlay(caption, norm, cmap_module):
    from matplotlib.patches import Patch

    handles = []
    for kind, text in caption.legend_overlay:
        if kind.startswith("alone_"):
            mid = kind.replace("alone_", "")
            rgba = ONLY_METHOD_RGBA.get(mid, (0.5, 0.5, 0.5, 0.4))
            handles.append(Patch(facecolor=rgba[:3], alpha=rgba[3], label=text))
        elif kind == "overlap":
            handles.append(Patch(facecolor=cmap_module(norm(5)), label=text))
        elif kind == "none":
            handles.append(Patch(facecolor="white", edgecolor="#333", label=text))
    return handles


def _render_figure(
    caption,
    out_path: Path,
    draw_fn,
    legend_handles,
) -> None:
    """draw_fn(ax_img) -> matplotlib AxesImage for colorbar."""
    import matplotlib.pyplot as plt

    n_key = max(len(legend_handles), 1)
    n_footer = len(caption.footer_lines)
    fig_h = 8.0 + 0.12 * n_key + 0.55 * n_footer
    fig = plt.figure(figsize=(10, fig_h))
    gs = fig.add_gridspec(
        5,
        1,
        height_ratios=[
            14,
            max(1.0, 0.35 * n_key),
            1.35,
            0.45,
            max(1.35, 0.5 * n_footer),
        ],
        hspace=0.55,
    )

    ax_img = fig.add_subplot(gs[0])
    ax_key = fig.add_subplot(gs[1])
    ax_cbar = fig.add_subplot(gs[2])
    ax_spacer = fig.add_subplot(gs[3])
    ax_footer = fig.add_subplot(gs[4])
    ax_spacer.axis("off")

    mappable = draw_fn(ax_img)
    ax_img.set_axis_off()

    fig.suptitle(caption.title, fontsize=12, fontweight="bold", y=0.99)
    ax_img.set_title(caption.subtitle, fontsize=10, pad=6)

    vmax = caption.colorbar_max_cm
    tick_max = int(np.ceil(vmax / 5.0) * 5)
    ticks = np.linspace(0, tick_max, min(6, max(2, tick_max // 5 + 1)))
    cbar = fig.colorbar(mappable, cax=ax_cbar, orientation="horizontal")
    cbar.set_label(
        f"{caption.colorbar_label}  ({caption.colorbar_note})",
        fontsize=9,
        labelpad=14,
    )
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(pad=4)

    ax_key.axis("off")
    ncol = min(len(legend_handles), 3) or 1
    ax_key.legend(
        handles=legend_handles,
        loc="center",
        ncol=ncol,
        fontsize=9,
        frameon=False,
        title="Key",
        title_fontsize=10,
    )

    ax_footer.axis("off")
    ax_footer.text(
        0.5,
        0.55,
        "\n".join(caption.footer_lines),
        ha="center",
        va="center",
        fontsize=9,
        transform=ax_footer.transAxes,
        bbox=dict(boxstyle="round", facecolor="#f8f8f8", alpha=0.95, edgecolor="#ccc", pad=0.8),
    )
    fig.subplots_adjust(top=0.94, bottom=0.06)

    _save_figure(fig, out_path)


def _render_standalone(data: dict, caption, out_path: Path) -> None:
    from matplotlib import cm
    from matplotlib.colors import Normalize

    display = np.ma.masked_where(~data["mask2"], data["display_cm"])
    vmin, vmax = 0.0, data["vmax"]
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("turbo").copy()
    cmap.set_bad(color="#3a3a3a")

    def draw(ax):
        return ax.imshow(display, cmap=cmap, norm=norm, interpolation="nearest")

    handles = _legend_handles_standalone(caption, norm, cm.get_cmap("turbo"))
    _render_figure(caption, out_path, draw, handles)


def _render_on_rgb(data: dict, rgb: np.ndarray, caption, out_path: Path) -> None:
    from matplotlib import cm
    from matplotlib.colors import Normalize

    count = data["count"]
    mask2 = data["mask2"]
    vmin, vmax = 0.0, data["vmax"]
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("turbo").copy()
    cmap.set_bad(color=(0, 0, 0, 0))
    display = np.ma.masked_where(~mask2, data["display_cm"])

    def draw(ax):
        ax.imshow(rgb, interpolation="nearest")
        for name in data["methods"]:
            mask = data["alone_masks"].get(name)
            if mask is None or not np.any(mask):
                continue
            rgba = np.zeros((*count.shape, 4), dtype=np.float32)
            rgba[mask] = ONLY_METHOD_RGBA.get(name, (0.5, 0.5, 0.5, 0.4))
            ax.imshow(rgba, interpolation="nearest")
        return ax.imshow(display, cmap=cmap, norm=norm, alpha=0.62, interpolation="nearest")

    handles = _legend_handles_overlay(caption, norm, cm.get_cmap("turbo"))
    _render_figure(caption, out_path, draw, handles)


def save_consensus_depth_std_png(
    stereo_dir: Path,
    geometry: dict,
    out_path: Path,
    scene_label: str | None = None,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")

    stereo_dir = Path(stereo_dir)
    out_path = Path(out_path)
    data = _compute_consensus(stereo_dir, geometry)
    if not data.get("ok"):
        return {"saved": False, "reason": data.get("reason"), "methods": data.get("methods", [])}

    caption = build_consensus_caption(data, run_id=scene_label)
    labels = caption.to_dict()

    _render_standalone(data, caption, out_path)

    overlay_path = out_path.with_name(out_path.stem + "_on_rgb" + out_path.suffix)
    overlay_saved = False
    overlay_reason = None
    h, w = data["std_map"].shape
    rgb = _load_rectified_rgb(stereo_dir, (h, w))
    if rgb is not None:
        _render_on_rgb(data, rgb, caption, overlay_path)
        overlay_saved = True
    else:
        overlay_reason = "rgb1_rectified.png not found in depth/ (run 02_make_stereo_pointcloud.py first)"

    return {
        "saved": True,
        "path": str(out_path),
        "overlay_path": str(overlay_path) if overlay_saved else None,
        "overlay_saved": overlay_saved,
        "overlay_reason": overlay_reason,
        "caption": labels,
        "summary": caption.summary,
        "methods": data["methods"],
        "median_std_m": float(data["p50"] / 100.0),
        "p95_std_m": float(data["p95"] / 100.0),
        "median_std_cm": data["p50"],
        "p95_std_cm": data["p95"],
        "frac_pixels_compared_pct": data["frac_compared"],
        "frac_single_method_pct": data["frac_single"],
        "frac_no_depth_pct": data["frac_none"],
        "frac_within_5cm_pct": data["frac_under_5cm"],
        "frac_within_15cm_pct": data["frac_under_15cm"],
        "alone_pct_by_method": data["alone_pct"],
        "colorbar_max_cm": data["vmax"],
    }
