# Depth method comparison (plain-language report)

**Scene / run:** pair_004

This report compares three ways to estimate 3D shape from your cameras. We use your **2D laser** as an independent ruler: where the laser hits a surface, we check whether each method’s depth agrees.

---

## Bottom line (read this first)

Overall pick for this capture: **DA-V2 scaled to all valid OpenCV depth**.

- **Closest to laser:** DA-V2 scaled to all valid OpenCV depth (typical gap 47.3 cm)

---

## What each method is

| Short name | What it is | Needs |
|------------|------------|-------|
| Classic stereo (OpenCV) | Two cameras, geometry math | Both cameras, calibration |
| Depth Anything V2 | AI depth from left image only | Left camera; scaled using stereo |
| FoundationStereo | AI two-camera depth | Both cameras, Windows + GPU |

---

## Comparison table

| Method | Surface coverage | Match to laser (typical error) | Trust this laser check? | Free-space warnings | Stereo photo match |
|--------|------------------|-------------------------------|-------------------------|---------------------|-------------------|
| Classic stereo (OpenCV) | Low (2.5%) | 48.3 cm — **Unreliable for this check** | **No** — too little overlap | 0.0% | 25.9 |
| DA-V2 scaled to all valid OpenCV depth | High (100.0%) | 47.3 cm — **Poor** | Yes | 0.0% | n/a (not run) |
| DA-V2 scaled using OpenCV pixels at manual GT distances | High (100.0%) | 75.2 cm — **Poor** | Yes | 0.0% | n/a (not run) |

### How to read the columns

- **Surface coverage:** How much of the image gets a depth value. Low on plain carpet is normal for classic stereo.
- **Match to laser:** Smaller is better (we report typical error in cm). Needs enough overlap to trust.
- **Free-space warnings:** High % means depth looks *in front of* the laser hit (often a scale bug, especially for AI single-camera).
- **Stereo photo match:** Lower is better (left vs right image consistency). Only for two-camera methods.

---

## Method details

### Classic stereo (OpenCV)

- **Coverage:** Low — Depth only in small patches (common on plain carpet/ sand).
- **Laser agreement:** Unreliable for this check — Too few LiDAR points could be compared (low overlap with depth map).
- **LiDAR points compared:** 1 of 100 visible to the camera (1.0% association rate).
- **Within 5 cm of laser:** 0.0% of compared points.

### DA-V2 scaled to all valid OpenCV depth

- **Coverage:** High — Most of the image has depth.
- **Laser agreement:** Poor — Large mismatch with the laser; do not use for precise measurements.
- **LiDAR points compared:** 66 of 100 visible to the camera (66.0% association rate).
- **Within 5 cm of laser:** 0.0% of compared points.

### DA-V2 scaled using OpenCV pixels at manual GT distances

- **Coverage:** High — Most of the image has depth.
- **Laser agreement:** Poor — Large mismatch with the laser; do not use for precise measurements.
- **LiDAR points compared:** 66 of 100 visible to the camera (66.0% association rate).
- **Within 5 cm of laser:** 0.0% of compared points.

### AI stereo (FoundationStereo)

*Not run for this capture.*

---

## Do the methods agree with each other?

- **OpenCV vs DA-V2:** typical depth difference 14.6 cm where both have data (22,641 pixels).
- **OpenCV vs DA-V2 gt:** typical depth difference 29.1 cm where both have data (22,641 pixels).
- **DA-V2 vs DA-V2 gt:** typical depth difference 17.9 cm where both have data (921,600 pixels).
- **All methods together:** typical disagreement 12.1 cm (see heatmap below).

---

## Figures for your report

**Charts (run `python 08_generate_eval_charts.py`):**
- `/Users/yashasvinigopalan/Documents/College/Spring 2026/Visual Computing Systems/Final Project/CS348K_Excavator_Progress_Tracking/outputs/runs/checkerboard_data/pair_004/validation/chart_scorecard.png` — green/yellow/red at-a-glance table
- `/Users/yashasvinigopalan/Documents/College/Spring 2026/Visual Computing Systems/Final Project/CS348K_Excavator_Progress_Tracking/outputs/runs/checkerboard_data/pair_004/validation/chart_coverage_and_accuracy.png` — side-by-side bar charts
- `/Users/yashasvinigopalan/Documents/College/Spring 2026/Visual Computing Systems/Final Project/CS348K_Excavator_Progress_Tracking/outputs/runs/checkerboard_data/pair_004/validation/chart_ray_error_histogram.png` — laser error distribution
- `/Users/yashasvinigopalan/Documents/College/Spring 2026/Visual Computing Systems/Final Project/CS348K_Excavator_Progress_Tracking/outputs/runs/checkerboard_data/pair_004/validation/chart_error_vs_range.png` — |ΔZ| vs LiDAR range (median/p90 per bin)
- `/Users/yashasvinigopalan/Documents/College/Spring 2026/Visual Computing Systems/Final Project/CS348K_Excavator_Progress_Tracking/outputs/runs/checkerboard_data/pair_004/validation/chart_photometric.png` — stereo consistency (if Foundation/OpenCV run)

**From evaluation (`06`):**
- `/Users/yashasvinigopalan/Documents/College/Spring 2026/Visual Computing Systems/Final Project/CS348K_Excavator_Progress_Tracking/outputs/runs/checkerboard_data/pair_004/validation/consensus_depth_std.png` — σ(Z) heatmap; caption in `evaluation_summary.json` → `consensus_map.caption`
- `/Users/yashasvinigopalan/Documents/College/Spring 2026/Visual Computing Systems/Final Project/CS348K_Excavator_Progress_Tracking/outputs/runs/checkerboard_data/pair_004/validation/consensus_depth_std_on_rgb.png` — same on `depth/rgb1_rectified.png`

**Cross-method consensus:** Overlap 100.0% of frame; on overlap: median σ(Z)=9.0 cm, 67.9% of px with σ<15 cm.

**Scene previews:**
- `../depth/disparity_preview.png`, `depth_preview_dav2.png`
- `../overlays/lidar_overlay_rgb1.png`

---

## Choosing a method for your scenario

| Your scene looks like… | Start with |
|------------------------|------------|
| Checkerboard / box / lots of texture | Classic stereo (OpenCV) |
| Plain carpet, sand, uniform color | Depth Anything V2 or FoundationStereo |
| Need full image filled with depth | Depth Anything V2 (if laser check is acceptable) |
| Need best laser agreement on textured targets | Whichever wins **Closest to laser** above |
| No Windows GPU | OpenCV + Depth Anything V2 (skip Foundation) |

*Generated by `07_generate_eval_report.py`. Technical JSON: `evaluation_summary.json`.*
