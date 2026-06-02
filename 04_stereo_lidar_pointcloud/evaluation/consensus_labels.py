"""
Programmatic captions for cross-method consensus figures.

All user-visible strings are built from run metrics + method metadata so new runs
and additional depth sources stay consistent without hand-editing PNG text.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evaluation.depth_maps import METHODS

# Per-method display + depth product path (under run depth/)
METHOD_META: dict[str, dict[str, str]] = {
    "opencv": {
        "label": "OpenCV stereo",
        "depth_product": "depth/depth_metric_opencv.npy",
        "description": "rectified pair → disparity.npy → metric Z (Q, P1)",
    },
    "dav2": {
        "label": "Depth Anything V2",
        "depth_product": "depth/depth_metric_dav2.npy",
        "description": "rgb1_rectified → relative depth → scale fit vs stereo",
    },
    "foundation": {
        "label": "FoundationStereo",
        "depth_product": "depth/disparity_foundation.npy → metric Z",
        "description": "rectified pair → learned disparity → metric Z (Q, P1)",
    },
}

# Neutral grey overlay when only one source has valid Z (avoids clash with turbo heatmap)
ONLY_METHOD_RGBA = {
    "opencv": (0.42, 0.42, 0.42, 0.48),
    "dav2": (0.42, 0.42, 0.42, 0.48),
    "foundation": (0.42, 0.42, 0.42, 0.48),
}


def method_label(method_id: str) -> str:
    return METHOD_META.get(method_id, {}).get("label", method_id)


def method_depth_product(method_id: str) -> str:
    meta = METHOD_META.get(method_id, {})
    if meta.get("depth_product"):
        return meta["depth_product"]
    cfg = METHODS.get(method_id, {})
    if cfg.get("depth_name"):
        return f"depth/{cfg['depth_name']}"
    if cfg.get("disparity_name"):
        return f"depth/{cfg['disparity_name']}"
    return f"depth/<{method_id}>"


def compared_methods_line(method_ids: list[str]) -> str:
    return " vs ".join(method_label(m) for m in method_ids)


@dataclass
class ConsensusCaption:
    run_id: str | None
    method_ids: list[str]
    image_shape: tuple[int, int]
    n_pixels: int
    n_overlap: int
    frac_overlap_pct: float
    frac_no_depth_pct: float
    alone_pct: dict[str, float]
    frac_within_5cm_pct: float
    frac_within_15cm_pct: float
    median_std_cm: float
    p95_std_cm: float
    colorbar_max_cm: float

    title: str = field(init=False)
    subtitle: str = field(init=False)
    colorbar_label: str = field(init=False)
    colorbar_note: str = field(init=False)
    footer_lines: list[str] = field(default_factory=list)
    summary: str = field(init=False)
    legend_standalone: list[str] = field(default_factory=list)
    legend_overlay: list[tuple[str, str]] = field(default_factory=list)  # (kind, text)

    def __post_init__(self) -> None:
        run = self.run_id or "unknown"
        h, w = self.image_shape
        self.title = f"Cross-method depth disagreement — {run}"
        self.subtitle = (
            f"Metric Z, rectified RGB1 ({w}×{h}) — {compared_methods_line(self.method_ids)}"
        )
        n = len(self.method_ids)
        self.colorbar_label = f"σ(Z) across {n} depth sources (cm), overlap pixels only"
        self.colorbar_note = f"scale max = {self.colorbar_max_cm:.0f} cm (p95 of σ)"
        self.footer_lines = self._build_footer_lines()
        self.summary = self._build_summary()
        self.legend_standalone = self._build_standalone_legend()
        self.legend_overlay = self._build_overlay_legend()

    def _build_footer_lines(self) -> list[str]:
        lines = [
            (
                f"Coverage: overlap (≥2 valid Z) = {self.frac_overlap_pct:.1f}% "
                f"({self.n_overlap:,} / {self.n_pixels:,} px)"
            ),
        ]
        for mid in self.method_ids:
            pct = self.alone_pct.get(mid, 0.0)
            if pct > 0.05:
                lines.append(f"  {method_label(mid)} only: {pct:.1f}% of px")
        if self.frac_no_depth_pct > 0.05:
            lines.append(f"  no valid depth (any source): {self.frac_no_depth_pct:.1f}% of px")
        if self.n_overlap > 0:
            lines.append(
                f"Overlap σ(Z): <5 cm on {self.frac_within_5cm_pct:.1f}% of px | "
                f"<15 cm on {self.frac_within_15cm_pct:.1f}% | "
                f"median {self.median_std_cm:.1f} cm, p95 {self.p95_std_cm:.1f} cm"
            )
        return lines

    def _build_summary(self) -> str:
        parts = [
            f"Overlap {self.frac_overlap_pct:.1f}% of frame",
        ]
        dominant = [(m, self.alone_pct.get(m, 0)) for m in self.method_ids]
        dominant.sort(key=lambda x: -x[1])
        if dominant and dominant[0][1] > 1.0:
            m, p = dominant[0]
            parts.append(f"{method_label(m)}-only {p:.1f}%")
        if self.n_overlap > 0:
            parts.append(
                f"on overlap: median σ(Z)={self.median_std_cm:.1f} cm, "
                f"{self.frac_within_15cm_pct:.1f}% of px with σ<15 cm"
            )
        return "; ".join(parts) + "."

    def _build_standalone_legend(self) -> list[str]:
        return [
            "Dark gray: exactly one source has valid Z (no σ computed)",
            "Colored: ≥2 sources valid; color = σ(Z) in cm (see colorbar)",
        ]

    def _build_overlay_legend(self) -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        for mid in self.method_ids:
            pct = self.alone_pct.get(mid, 0.0)
            if pct < 0.05:
                continue
            entries.append(
                (
                    f"alone_{mid}",
                    f"Grey tint: {method_label(mid)} only ({pct:.1f}% of image)",
                )
            )
        entries.append(
            ("overlap", "Heatmap: ≥2 sources valid; color = σ(Z) (cm)"),
        )
        if self.frac_no_depth_pct > 0.05:
            entries.append(
                ("none", f"No tint: no valid Z — {self.frac_no_depth_pct:.1f}% px"),
            )
        return entries

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "method_ids": self.method_ids,
            "title": self.title,
            "subtitle": self.subtitle,
            "colorbar_label": self.colorbar_label,
            "colorbar_note": self.colorbar_note,
            "footer_lines": self.footer_lines,
            "summary": self.summary,
            "legend_standalone": self.legend_standalone,
            "legend_overlay": [{"kind": k, "text": t} for k, t in self.legend_overlay],
            "sources": {
                m: {
                    "label": method_label(m),
                    "depth_product": method_depth_product(m),
                    "description": METHOD_META.get(m, {}).get("description", ""),
                }
                for m in self.method_ids
            },
        }


def build_consensus_caption(data: dict, run_id: str | None = None) -> ConsensusCaption:
    shape = data["std_map"].shape
    n_pix = int(data["count"].size)
    n_overlap = int(data["mask2"].sum())
    return ConsensusCaption(
        run_id=run_id,
        method_ids=data["methods"],
        image_shape=(shape[0], shape[1]),
        n_pixels=n_pix,
        n_overlap=n_overlap,
        frac_overlap_pct=data["frac_compared"],
        frac_no_depth_pct=data["frac_none"],
        alone_pct=data["alone_pct"],
        frac_within_5cm_pct=data["frac_under_5cm"],
        frac_within_15cm_pct=data["frac_under_15cm"],
        median_std_cm=data["p50"],
        p95_std_cm=data["p95"],
        colorbar_max_cm=data["vmax"],
    )
