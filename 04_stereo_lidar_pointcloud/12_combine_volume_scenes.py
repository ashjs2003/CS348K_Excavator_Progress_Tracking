"""
Combine volume-estimate outputs from multiple scenes into one table/chart.

Example:
    python 12_combine_volume_scenes.py \
      --name L_box_all_views \
      --scenes L_carboard_box L_cardboard_box_30
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

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from output_runs import RUNS_ROOT


def parse_args():
    p = argparse.ArgumentParser(description="Combine per-scene ROI bbox volume tables")
    p.add_argument("--name", required=True, help="Combined output prefix, e.g. L_box_all_views")
    p.add_argument("--scenes", nargs="+", required=True, help="Scene folders under outputs/runs/")
    p.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    p.add_argument("--gt-volume-cm3", type=float, default=None, help="Ground-truth volume for cell coloring")
    p.add_argument(
        "--gt-scene-volume",
        nargs="*",
        default=[],
        help="Per-scene GT volume entries like scene=cm3 (overrides --gt-volume-cm3 for matched scenes)",
    )
    p.add_argument(
        "--include-images",
        action="store_true",
        help="Also render a scene/pair thumbnail sheet from capture/rgb1.png",
    )
    p.add_argument(
        "--embed-images-in-table",
        action="store_true",
        help="Render combined PNG as image+metrics grid (drops scene/pair text columns)",
    )
    p.add_argument(
        "--table-columns",
        type=int,
        default=1,
        help="Side-by-side blocks for embedded table (use 2 for slides)",
    )
    p.add_argument(
        "--exclude-pair",
        nargs="*",
        default=[],
        metavar="SCENE/PAIR",
        help="Omit rows, e.g. excavator_M/007 or excavator_S=021 (pair may be 7 or 007)",
    )
    return p.parse_args()


def read_scene_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float_or_inf(v: str) -> float:
    if v is None or v == "":
        return float("inf")
    try:
        return float(v)
    except ValueError:
        return float("inf")


def combine_rows(scenes: list[str], runs_root: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    method_cols: list[str] = []
    for scene in scenes:
        csv_path = runs_root / scene / "roi_bbox_volume_estimates.csv"
        for row in read_scene_csv(csv_path):
            row2 = dict(row)
            row2["scene"] = scene
            rows.append(row2)
            for k in row2:
                if k.endswith("_volume_cm3"):
                    method_cols.append(k)
    rows.sort(key=lambda r: (to_float_or_inf(r.get("ruler_distance_cm", "")), r.get("scene", ""), r.get("pair_id", "")))
    method_cols = sorted(set(method_cols))
    return rows, method_cols


def _norm_pair_id(pair: str) -> str:
    p = str(pair).strip().removeprefix("pair_")
    if p.isdigit():
        return p.zfill(3)
    return p


def parse_exclude_pairs(entries: list[str]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for e in entries:
        if "/" in e:
            scene, pair = e.split("/", 1)
        elif "=" in e:
            scene, pair = e.split("=", 1)
        else:
            continue
        out.add((scene.strip(), _norm_pair_id(pair)))
    return out


def filter_excluded_rows(rows: list[dict], exclude: set[tuple[str, str]]) -> list[dict]:
    if not exclude:
        return rows
    return [
        r
        for r in rows
        if (r.get("scene", ""), _norm_pair_id(r.get("pair_id", ""))) not in exclude
    ]


def parse_scene_gt_entries(entries: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for e in entries:
        if "=" not in e:
            continue
        k, v = e.split("=", 1)
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            continue
    return out


def gt_for_row(row: dict, gt_volume_cm3: float | None, scene_gt: dict[str, float]) -> float | None:
    scene = row.get("scene", "")
    if scene in scene_gt:
        return scene_gt[scene]
    return gt_volume_cm3


def write_csv(
    rows: list[dict],
    method_cols: list[str],
    out_csv: Path,
    gt_volume_cm3: float | None,
    scene_gt: dict[str, float],
) -> None:
    fields = ["scene", "pair_id"]
    if gt_volume_cm3 is not None or scene_gt:
        fields.append("gt_volume_cm3")
    fields += method_cols
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            payload = {k: r.get(k, "") for k in fields}
            gt_row = gt_for_row(r, gt_volume_cm3, scene_gt)
            if gt_row is not None:
                payload["gt_volume_cm3"] = f"{gt_row:.1f}"
            w.writerow(payload)


def write_json(
    rows: list[dict],
    scenes: list[str],
    out_json: Path,
    gt_volume_cm3: float | None,
    scene_gt: dict[str, float],
) -> None:
    payload = {
        "combined_scenes": scenes,
        "gt_volume_cm3": gt_volume_cm3,
        "gt_volume_by_scene_cm3": scene_gt,
        "n_rows": len(rows),
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")


def _method_label_from_col(col: str) -> str:
    return col.replace("_volume_cm3", "").replace("_", " ").upper()

def _method_header_short(col: str) -> str:
    name = col.replace("_volume_cm3", "")
    mapping = {
        "opencv": "OpenCV",
        "dav2": "DA-V2",
        "dav2_gt": "DA-V2 GT",
        "foundation": "Foundation",
        "lidar_volume": "LiDAR",
    }
    return mapping.get(name, name.replace("_", " ").title())


# Five bands: (max_pct_inclusive, color, legend_label)
ERROR_BANDS: list[tuple[float, str, str]] = [
    (10.0, "#d8f3dc", "≤10%"),
    (25.0, "#b7e4c7", "10–25%"),
    (50.0, "#ffeaa7", "25–50%"),
    (100.0, "#fab1a0", "50–100%"),
    (float("inf"), "#e17055", ">100%"),
]


def _error_cell_color(pct: float | None) -> str:
    if pct is None:
        return "#f5f5f5"
    for max_pct, color, _label in ERROR_BANDS:
        if pct <= max_pct:
            return color
    return ERROR_BANDS[-1][1]


def _error_text_color(pct: float) -> str:
    """Light backgrounds get dark text; strong orange/red get white text."""
    if pct <= 50.0:
        return "#111111"
    return "#ffffff"


def _draw_percent_error_legend(fig) -> None:
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=color, edgecolor="#666666", linewidth=0.8, label=label)
        for _max_pct, color, label in ERROR_BANDS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(ERROR_BANDS),
        frameon=True,
        fontsize=14,
        title="Percent error vs GT volume",
        title_fontsize=16,
        handlelength=2.4,
        handleheight=1.6,
        labelspacing=0.7,
        borderpad=1.0,
        bbox_to_anchor=(0.5, 0.0),
        borderaxespad=0.0,
    )


def write_png(
    rows: list[dict],
    method_cols: list[str],
    out_png: Path,
    title: str,
    gt_volume_cm3: float | None,
    scene_gt: dict[str, float],
) -> None:
    if not rows:
        return
    shown_methods = [c for c in method_cols if any(r.get(c, "") != "" for r in rows)]
    col_labels = ["Scene", "Pair"]
    if gt_volume_cm3 is not None or scene_gt:
        col_labels.append("GT (cm³)")
    col_labels += [_method_label_from_col(c) + " (cm³)" for c in shown_methods]
    cell_text = []
    cell_colors: list[list[str]] = []
    for r in rows:
        row = [r.get("scene", ""), r.get("pair_id", "")]
        row_colors = ["white", "white"]
        gt_row = gt_for_row(r, gt_volume_cm3, scene_gt)
        if gt_row is not None:
            row.append(f"{gt_row:,.0f}")
            row_colors.append("#e8e8e8")
        for c in shown_methods:
            val = r.get(c, "")
            if val == "":
                row.append("—")
                row_colors.append("#f6f6f6")
            else:
                try:
                    v = float(val)
                    row.append(f"{v:,.0f}")
                    if gt_row is not None and gt_row > 0:
                        pct = abs(v - gt_row) * 100.0 / gt_row
                        row_colors.append(_error_cell_color(pct))
                    else:
                        row_colors.append("white")
                except ValueError:
                    row.append(val)
                    row_colors.append("white")
        cell_text.append(row)
        cell_colors.append(row_colors)

    fig_h = 1.5 + 0.34 * len(cell_text)
    fig_w = max(10.0, 2.8 + 1.8 * len(col_labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.35)
    for (row_idx, col_idx), cell in tbl.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(fontweight="bold")
        elif row_idx > 0:
            cell.set_facecolor(cell_colors[row_idx - 1][col_idx])
    if gt_volume_cm3 is not None or scene_gt:
        if scene_gt:
            key_txt = "GT key: " + ", ".join(f"{k}={v:,.0f} cm³" for k, v in sorted(scene_gt.items()))
            ax.text(
                0.5,
                0.01,
                key_txt,
                ha="center",
                transform=ax.transAxes,
                fontsize=8,
                color="#555",
            )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_image_gallery(rows: list[dict], runs_root: Path, out_png: Path, title: str) -> None:
    if not rows:
        return
    cols = 4
    n = len(rows)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.6 * cols, 2.9 * rows_n), squeeze=False)
    fig.suptitle(f"{title}: scene/pair thumbnails", fontsize=12, fontweight="bold", y=0.99)
    axes_flat = axes.ravel()

    for idx, ax in enumerate(axes_flat):
        ax.axis("off")
        if idx >= n:
            continue
        r = rows[idx]
        scene = r.get("scene", "")
        pair = r.get("pair_id", "")
        img_path = Path(runs_root) / scene / f"pair_{pair}" / "capture" / "rgb1.png"
        if img_path.is_file():
            try:
                im = plt.imread(str(img_path))
                ax.imshow(im)
            except Exception:
                ax.text(0.5, 0.5, "image read failed", ha="center", va="center", fontsize=8)
                ax.set_facecolor("#f3f3f3")
        else:
            ax.text(0.5, 0.5, "missing rgb1.png", ha="center", va="center", fontsize=8)
            ax.set_facecolor("#f3f3f3")
        ax.set_title(f"{scene} / pair_{pair}", fontsize=8, pad=2)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_png_with_embedded_images(
    rows: list[dict],
    method_cols: list[str],
    out_png: Path,
    title: str,
    gt_volume_cm3: float | None,
    scene_gt: dict[str, float],
    runs_root: Path,
    table_columns: int = 1,
) -> None:
    if not rows:
        return
    table_columns = max(1, int(table_columns))
    shown_methods = [c for c in method_cols if any(r.get(c, "") != "" for r in rows)]
    headers = ["Image"]
    if gt_volume_cm3 is not None or scene_gt:
        headers.append("GT\n(cm³)")
    headers += [f"{_method_header_short(c)}\n(cm³)" for c in shown_methods]

    n_cols_block = len(headers)
    n_items = len(rows)
    rows_per_block = (n_items + table_columns - 1) // table_columns
    n_rows = rows_per_block + 1  # include header
    n_cols = n_cols_block * table_columns
    width_ratios = []
    for _ in range(table_columns):
        width_ratios += [2.8] + [1.25] * (n_cols_block - 1)
    fig_w = max(13.5, 2.0 * n_cols)
    fig_h = max(6.5, 1.8 * n_rows + 0.9)  # extra space for legend
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": width_ratios},
        squeeze=False,
    )
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)

    # Header row in each block.
    for b in range(table_columns):
        base = b * n_cols_block
        for j, h in enumerate(headers):
            ax = axes[0, base + j]
            ax.set_facecolor("#e8e8e8")
            ax.text(0.5, 0.5, h, ha="center", va="center", fontsize=10, fontweight="bold", wrap=True)
            ax.set_xticks([])
            ax.set_yticks([])

    # Data rows distributed across blocks.
    for idx, r in enumerate(rows):
        b = idx // rows_per_block
        r_in_block = idx % rows_per_block
        i = r_in_block + 1
        base = b * n_cols_block
        gt_row = gt_for_row(r, gt_volume_cm3, scene_gt)

        # Image cell with scene/pair caption.
        ax_img = axes[i, base + 0]
        scene = r.get("scene", "")
        pair = r.get("pair_id", "")
        img_path = Path(runs_root) / scene / f"pair_{pair}" / "capture" / "rgb1.png"
        if img_path.is_file():
            try:
                im = plt.imread(str(img_path))
                ax_img.imshow(im)
            except Exception:
                ax_img.set_facecolor("#f4f4f4")
                ax_img.text(0.5, 0.5, "image read failed", ha="center", va="center", fontsize=8)
        else:
            ax_img.set_facecolor("#f4f4f4")
            ax_img.text(0.5, 0.5, "missing image", ha="center", va="center", fontsize=8)
        ax_img.set_title(f"{scene} / pair_{pair}", fontsize=10, pad=2)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        col_offset = 1
        if gt_row is not None:
            ax_gt = axes[i, base + 1]
            ax_gt.set_facecolor("#e8e8e8")
            ax_gt.text(0.5, 0.5, f"{gt_row:,.0f}", ha="center", va="center", fontsize=11, fontweight="bold")
            ax_gt.set_xticks([])
            ax_gt.set_yticks([])
            col_offset = 2

        for k, c in enumerate(shown_methods):
            ax = axes[i, base + k + col_offset]
            val = r.get(c, "")
            if val == "":
                ax.set_facecolor("#f6f6f6")
                ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=11)
            else:
                try:
                    v = float(val)
                    if gt_row is not None and gt_row > 0:
                        pct = abs(v - gt_row) * 100.0 / gt_row
                        ax.set_facecolor(_error_cell_color(pct))
                        tcolor = _error_text_color(pct)
                    else:
                        ax.set_facecolor("white")
                        tcolor = "#111111"
                    ax.text(
                        0.5,
                        0.5,
                        f"{v:,.0f}",
                        ha="center",
                        va="center",
                        fontsize=11,
                        fontweight="bold",
                        color=tcolor,
                    )
                except ValueError:
                    ax.set_facecolor("white")
                    ax.text(0.5, 0.5, val, ha="center", va="center", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])

    # Hide empty slots in last block if needed.
    for idx in range(n_items, rows_per_block * table_columns):
        b = idx // rows_per_block
        r_in_block = idx % rows_per_block
        i = r_in_block + 1
        base = b * n_cols_block
        for j in range(n_cols_block):
            axes[i, base + j].axis("off")

    # Grid lines
    for i in range(n_rows):
        for j in range(n_cols):
            for spine in axes[i, j].spines.values():
                spine.set_edgecolor("#b0b0b0")
                spine.set_linewidth(0.8)

    _draw_percent_error_legend(fig)
    fig.tight_layout(rect=[0, 0.13, 1, 0.985])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=145, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    scene_gt = parse_scene_gt_entries(args.gt_scene_volume)
    exclude = parse_exclude_pairs(args.exclude_pair)
    rows, method_cols = combine_rows(args.scenes, args.runs_root)
    rows = filter_excluded_rows(rows, exclude)
    if not rows:
        print("No rows found to combine.")
        return 1
    out_dir = args.runs_root / "_combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name
    out_csv = out_dir / f"{stem}.csv"
    out_json = out_dir / f"{stem}.json"
    out_png = out_dir / f"{stem}.png"
    out_gallery = out_dir / f"{stem}_gallery.png"
    write_csv(rows, method_cols, out_csv, args.gt_volume_cm3, scene_gt)
    write_json(rows, args.scenes, out_json, args.gt_volume_cm3, scene_gt)
    if args.embed_images_in_table:
        write_png_with_embedded_images(
            rows,
            method_cols,
            out_png,
            f"{stem}: volume estimates across views",
            args.gt_volume_cm3,
            scene_gt,
            args.runs_root,
            args.table_columns,
        )
    else:
        write_png(rows, method_cols, out_png, f"{stem}: volume estimates across views", args.gt_volume_cm3, scene_gt)
    if args.include_images:
        write_image_gallery(rows, args.runs_root, out_gallery, stem)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_png}")
    if args.include_images:
        print(f"Wrote {out_gallery}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

