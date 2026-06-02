"""Numeric table + error heatmap grids for ROI GT evaluation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from evaluation.roi_gt_compare import METHOD_CHART_LABELS, METHOD_ORDER


def _cell_color(median_cm: float | None) -> str:
    if median_cm is None:
        return "#f0f0f0"
    if median_cm <= 5:
        return "#c6efce"
    if median_cm <= 15:
        return "#ffeb9c"
    return "#ffc7ce"


def capture_label(summary: dict) -> str:
    scene = summary.get("scene", "")
    pair_id = summary.get("pair_id", "")
    return f"{scene}  pair_{pair_id}"


def active_gt_columns(summary: dict) -> list[tuple[str, str, float]]:
    """(role_key, column_label, gt_cm) for roles with compared pixels."""
    cols: list[tuple[str, str, float]] = []
    ruler_cm = summary.get("target_gt_cm")
    wall_cm = summary.get("wall_gt_cm")
    if summary.get("ruler_pixels", 0) > 0 and ruler_cm is not None:
        cols.append(("ruler", f"GT {ruler_cm:.0f} cm", float(ruler_cm)))
    if summary.get("wall_pixels", 0) > 0 and wall_cm is not None:
        cols.append(("wall", f"GT {wall_cm:.0f} cm", float(wall_cm)))
    return cols


def gt_subtitle(summary: dict) -> str:
    parts = [label for _, label, _ in active_gt_columns(summary)]
    if not parts:
        return ""
    if len(parts) == 1:
        return f"Ground truth: {parts[0].replace('GT ', '')}"
    return "Ground truth:  " + "  ·  ".join(p.replace("GT ", "") for p in parts)


def _method_panel_title(summary: dict, method: str, role: str | None = None) -> str:
    """Subplot title: method name + median depth (+ error)."""
    label = METHOD_CHART_LABELS.get(method, method)
    mdata = summary.get("methods", {}).get(method, {})
    if role:
        stats = mdata.get(role, {})
    else:
        stats = mdata.get("all_roi_compared", {})
    z_cm = stats.get("median_depth_cm")
    err_cm = stats.get("median_error_cm")
    if z_cm is None:
        return f"{label}\n—"
    lines = [label, f"{z_cm:.1f} cm"]
    if err_cm is not None:
        lines.append(f"Δ {err_cm:.1f} cm")
    return "\n".join(lines)


def render_numeric_grid(summary: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    methods = [m for m in METHOD_ORDER if m in summary.get("methods", {})]
    if not methods:
        return

    gt_cols = active_gt_columns(summary)
    if not gt_cols:
        return

    nrows = len(methods)
    ncol = len(gt_cols)
    fig_w = max(4.0, 3.2 * ncol)
    fig_h = 0.55 * nrows + 1.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    fig.suptitle(capture_label(summary), fontsize=13, fontweight="bold", y=0.98)
    sub = gt_subtitle(summary)
    if sub:
        fig.text(0.5, 0.93, sub, ha="center", fontsize=10, color="#333")

    col_headers = [hdr for _, hdr, _ in gt_cols]
    table = []
    cell_colors = []
    for method in methods:
        row_cells = []
        row_colors = []
        mdata = summary["methods"][method]
        for role_key, _, _ in gt_cols:
            stats = mdata.get(role_key, {})
            z_cm = stats.get("median_depth_cm")
            err_cm = stats.get("median_error_cm")
            n = stats.get("n_pixels", 0)
            if n == 0 or z_cm is None:
                row_cells.append("—")
                row_colors.append("#f0f0f0")
            else:
                row_cells.append(f"{z_cm:.1f} cm\nΔ {err_cm:.1f} cm")
                row_colors.append(_cell_color(err_cm))
        table.append(row_cells)
        cell_colors.append(row_colors)

    tbl = ax.table(
        cellText=table,
        rowLabels=[METHOD_CHART_LABELS.get(m, m) for m in methods],
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.15, 2.4)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0 or col == -1:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(fontweight="bold")
        elif row > 0 and col >= 0:
            cell.set_facecolor(cell_colors[row - 1][col])

    ax.text(
        0.5,
        0.02,
        "Median depth in ROI band  ·  cell color = |Z − GT|",
        ha="center",
        transform=ax.transAxes,
        fontsize=8,
        color="#666",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.90])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_heatmap_grid_combined(
    rgb_bgr: np.ndarray,
    polygon_xy: np.ndarray,
    error_maps_cm: dict[str, np.ndarray],
    summary: dict,
    out_path: Path,
    *,
    vmax_cm: float = 30.0,
) -> None:
    """One row: per-method error vs per-pixel GT (ruler + wall bands combined)."""
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    methods = [m for m in METHOD_ORDER if m in error_maps_cm]
    if not methods:
        return

    ncols = len(methods)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.2), squeeze=False)
    axes = axes[0]

    fig.suptitle(capture_label(summary), fontsize=12, fontweight="bold", y=1.02)
    sub = gt_subtitle(summary)
    if sub:
        fig.text(0.5, 0.96, sub, ha="center", fontsize=10, color="#333", transform=fig.transFigure)

    norm = Normalize(vmin=0, vmax=vmax_cm)
    cmap = cm.get_cmap("turbo").copy()
    cmap.set_bad(color=(0.2, 0.2, 0.2, 0.25))

    rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    pts = polygon_xy.reshape(-1, 1, 2)

    im = None
    for col, method in enumerate(methods):
        ax = axes[col]
        err = error_maps_cm[method]
        err_show = np.ma.masked_where(~np.isfinite(err), err)
        ax.imshow(rgb_rgb, interpolation="nearest")
        im = ax.imshow(err_show, cmap=cmap, norm=norm, alpha=0.7, interpolation="nearest")
        ax.plot(pts[:, 0, 0], pts[:, 0, 1], color="yellow", linewidth=1.8)
        ax.set_title(_method_panel_title(summary, method), fontsize=10)
        ax.set_axis_off()

    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.022, pad=0.02, label="|Z − GT| (cm)")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_heatmap_grid_split_roles(
    rgb_bgr: np.ndarray,
    polygon_xy: np.ndarray,
    error_maps_cm: dict[str, np.ndarray],
    wall_mask: np.ndarray,
    ruler_mask: np.ndarray,
    summary: dict,
    out_path: Path,
    *,
    vmax_cm: float = 30.0,
) -> None:
    """Separate rows per GT band when both ruler and wall are compared."""
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    methods = [m for m in METHOD_ORDER if m in error_maps_cm]
    if not methods:
        return

    rows: list[tuple[str, np.ndarray, str]] = []
    ruler_cm = summary.get("target_gt_cm")
    wall_cm = summary.get("wall_gt_cm")
    if summary.get("ruler_pixels", 0) > 0 and ruler_cm is not None:
        rows.append(("ruler", ruler_mask, f"GT {ruler_cm:.0f} cm"))
    if summary.get("wall_pixels", 0) > 0 and wall_cm is not None:
        rows.append(("wall", wall_mask, f"GT {wall_cm:.0f} cm"))
    if not rows:
        return

    ncols = len(methods)
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.8 * nrows), squeeze=False)
    if nrows == 1:
        axes = axes.reshape(1, -1)

    norm = Normalize(vmin=0, vmax=vmax_cm)
    cmap = cm.get_cmap("turbo").copy()
    cmap.set_bad(color=(0.2, 0.2, 0.2, 0.2))

    rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    pts = polygon_xy.reshape(-1, 1, 2)

    im = None
    for row_i, (role_key, mask, row_label) in enumerate(rows):
        for col, method in enumerate(methods):
            ax = axes[row_i][col]
            err_all = error_maps_cm[method]
            ax.imshow(rgb_rgb, interpolation="nearest")
            err_show = np.ma.masked_where(~(mask & np.isfinite(err_all)), err_all)
            im = ax.imshow(err_show, cmap=cmap, norm=norm, alpha=0.72, interpolation="nearest")
            ax.plot(pts[:, 0, 0], pts[:, 0, 1], color="yellow", linewidth=1.5)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=10, fontweight="bold")
            ax.set_title(_method_panel_title(summary, method, role_key), fontsize=9)
            ax.set_axis_off()

    fig.suptitle(capture_label(summary), fontsize=12, fontweight="bold", y=1.01)
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="|Z − GT| (cm)")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
