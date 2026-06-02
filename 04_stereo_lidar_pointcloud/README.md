# Stereo RGB + 2D LiDAR Point Cloud Workflow

Capture RGB1/RGB2/LiDAR, build a **metric point cloud**, validate against LiDAR, and compare **three depth methods**.

Run all commands from this folder:

```bash
cd stereo_lidar_pointcloud
```

Runs are stored under `../outputs/runs/<timestamp>_<label>/` (see [../outputs/README.md](../outputs/README.md)). Use `--run latest` (default) or `--list-runs` on any step-02 script.

---

## Three depth methods

All methods share the **same calibration** (OpenCV checkerboard + stereo extrinsics) and the **same rectified RGB1/RGB2 pair**. They differ only in how disparity/depth is computed.

| Method | Script | Input | Best for | Mac | Windows (NVIDIA GPU) |
|--------|--------|-------|----------|-----|----------------------|
| **1. OpenCV** | `02_make_stereo_pointcloud.py` | RGB1 + RGB2 | Default pipeline, no extra deps, works everywhere | Yes | Yes |
| **2. Depth Anything V2** | `02_make_depth_anything_pointcloud.py` | RGB1 only (monocular) | Dense depth on low texture (carpet); scaled to meters via OpenCV | Yes (MPS) | Yes (CUDA) |
| **3. FoundationStereo** | `02_make_stereo_pointcloud_foundation.py` | RGB1 + RGB2 | Learned stereo matching, often strong zero-shot | No | Yes (CUDA) |

**Recommendation**

- **Always run OpenCV first** — it is required for LiDAR validation defaults and for scaling Depth Anything V2 to metric meters.
- Add **Depth Anything V2** on Mac or Windows when OpenCV coverage is poor (e.g. carpet).
- Add **FoundationStereo** on a Windows CUDA machine when you want a learned stereo baseline vs OpenCV.

```text
capture/rgb1.png + rgb2.png
        │
        ├─► 02_make_stereo_pointcloud.py              → disparity.npy, stereo_pointcloud_*.ply
        │
        ├─► 02_make_depth_anything_pointcloud.py      → depth_metric_dav2.npy, *_dav2.ply
        │
        └─► 02_make_stereo_pointcloud_foundation.py   → disparity_foundation.npy, *_foundation.ply
```

---

## Platform summary

| Task | macOS | Windows |
|------|-------|---------|
| Capture (`01_capture_one_set.py`) | Yes | Yes |
| OpenCV stereo (`02_make_stereo_pointcloud.py`) | Yes | Yes |
| Depth Anything V2 | Yes — conda env `depth_anything_v2`, MPS | Yes — same env, CUDA |
| FoundationStereo | **Skip** (no NVIDIA CUDA) | Yes — conda env `foundation_stereo` |
| Validate / view / LiDAR overlay (`03`–`05`) | Yes | Yes |

Your **main project Python** (OpenCV, capture, LiDAR) stays separate from optional **conda envs** used only for deep models.

---

## Prerequisites

Calibration files are selected by the repo-level `../config.yaml`:

- `left_intrinsics` (RGB1 / left camera)
- `right_intrinsics` (RGB2 / right camera)
- `stereo_rgb1_rgb2_extrinsics`
- `lidar_to_rgb1_extrinsics` (for LiDAR validation / overlays)

RGB2 should be calibrated separately when possible; copying RGB1 intrinsics is a quick prototype only.

---

## One-time setup (optional methods)

### Depth Anything V2 — Mac or Windows

```bash
# Mac / Linux
bash scripts/setup_depth_anything_v2.sh

# Windows (PowerShell, repo root)
powershell -ExecutionPolicy Bypass -File scripts\setup_depth_anything_v2.ps1
```

Creates conda env **`depth_anything_v2`** and downloads the Small (**vits**) checkpoint to  
`third_party/Depth-Anything-V2/checkpoints/` (gitignored).

Verify:

```bash
conda run -n depth_anything_v2 python -c "import torch; print('cuda', torch.cuda.is_available(), 'mps', torch.backends.mps.is_available())"
```

### FoundationStereo — Windows only

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_foundationstereo.ps1
```

Download weights per script output into  
`third_party/FoundationStereo/pretrained_models/23-51-11/`.

On Mac, `bash scripts/setup_foundationstereo.sh` **exits without installing** (by design).

---

## Standard pipeline (every capture)

Replace `latest` with a run id from `--list-runs` if needed.

### 1. Capture

```bash
python 01_capture_one_set.py --label carpet
```

### 2. Depth / point cloud

**OpenCV (required baseline):**

```bash
python 02_make_stereo_pointcloud.py --run latest --method stereobm
```

`--method` choices: `stereobm` (default, OpenCV StereoBM), `sgbm`, `flow`, `blend` (aliases: `carpet`, `bm`).

**Depth Anything V2** (after OpenCV):

```bash
python 02_make_depth_anything_pointcloud.py --run latest --reuse-rectified

# Batch all scene folders under ../data/ → outputs/runs/<scene>/pair_<id>/
python batch_dav2_data_folders.py
python batch_dav2_data_folders.py --folders checkerboard_data excavator_S --skip-existing
python 06_evaluate_run.py --run checkerboard_data/pair_000
```

**FoundationStereo** (Windows + CUDA, after OpenCV):

```powershell
python 02_make_stereo_pointcloud_foundation.py --run latest --reuse-rectified
```

### 3. Structured evaluation (recommended)

Ray depth, free-space violations, error-vs-range bins, photometric (M1/M3), cross-method, consensus PNG:

```bash
python 06_evaluate_run.py --run latest
python 07_generate_eval_report.py --run latest
python 08_generate_eval_charts.py --run latest
python compare_stereo_methods.py --run latest
```

Writes `evaluation_summary.json`, `EVAL_REPORT.md`, charts (`chart_*.png`), `consensus_depth_std.png`, and **manual GT overlays** (`gt_depth_reference_on_rgb.png`, etc.).

**ROI polygon + per-pixel GT check** (ruler from `pair_*.txt`, wall = 100 cm):

```bash
# Draw ROI on rectified image (needs display); saves data/<scene>/pair_XXX_roi.json
python 10_annotate_roi_polygon.py --scene checkerboard_data --pair 001
python 10_annotate_roi_polygon.py --scene checkerboard_data --all

# Compare each depth method at ROI pixels (wall = farthest depth band inside ROI)
python 10_evaluate_roi_gt_depth.py --run checkerboard_data/pair_001
python 10_evaluate_roi_gt_depth.py --scene checkerboard_data --all
```

Outputs per capture (`validation/`):

- `roi_gt_eval_grid.png` numeric table (median error cm per method)
- `roi_gt_eval_heatmap.png` combined error map on RGB
- `roi_gt_eval_heatmap_roles.png` ruler row + wall row (wall omitted on excavator)
- `roi_gt_depth_compare.json`, `roi_gt_overlay.png`, `roi_gt_per_point_<method>.csv`

Batch all annotated pairs (no 100 cm wall on `excavator_M` / `excavator_S`):

```bash
python 10_batch_roi_gt_eval_grid.py
```

Per-scene ROI error vs ground-truth distance (pooled annotated captures):

```bash
python 11_roi_scene_error_vs_distance.py
```

Writes `outputs/runs/<scene>/roi_error_vs_gt_distance.png` and `.json`.

**Manual GT overlays** (automatic with every `06_evaluate_run.py`):

- **Target distance:** `data/<scene>/pair_<id>.txt` (value in **cm** when &gt; 2, e.g. `25` → 25 cm).
- **Back wall:** fixed **100 cm** for all data.
- **Match band:** fixed **±5 cm**.

```bash
python 06_evaluate_run.py --run checkerboard_data/pair_001
```

Outputs: `validation/gt_depth_reference_on_rgb.png` (union overlay + on-image key), `gt_depth_reference_labeled.png` (2×2 grid: all methods + per depth source), `gt_depth_all_methods_on_rgb.png`, and per-method `gt_depth_match_<method>_on_rgb.png`. Teal ≈ ruler distance (`pair_*.txt`); blue ≈ back wall (100 cm); overlap uses a teal/blue blend (no yellow).

Single method only (subset of 06):

```bash
python 03_validate_with_lidar.py --run latest
python 03_validate_with_lidar.py --run latest --stereo-suffix _dav2 --metrics-suffix _dav2
```

### 4. View

```bash
# OpenCV + LiDAR
python 04_view_pointcloud.py --run latest --stereo-backend opencv --mode both

# Depth Anything V2 only
python 04_view_pointcloud.py --run latest --stereo-backend dav2 --mode stereo

# FoundationStereo only (if generated on Windows)
python 04_view_pointcloud.py --run latest --stereo-backend foundation --mode stereo

# Side-by-side (2 or 3 panels, stereo only)
python 04_view_pointcloud.py --run latest --stereo-backend compare --mode stereo
python 04_view_pointcloud.py --run latest --stereo-backend compare-all --mode stereo
```

### 5. LiDAR on RGB images

```bash
python 05_project_lidar_overlay.py --run latest
```

---

## Comparing methods

### Quick stats table

```bash
python 06_evaluate_run.py --run latest
python compare_stereo_methods.py --run latest
```

Prints coverage, **ray median**, **free-space violation %**, photometric error, and cross-method stats from `evaluation_summary.json`.

### Output files (same run, `depth/` folder)

| OpenCV | Depth Anything V2 | FoundationStereo |
|--------|-------------------|------------------|
| `disparity.npy` | — | — |
| `disparity_preview.png` | — | — |
| `stereo_pointcloud.ply` | `stereo_pointcloud_dav2.ply` | `stereo_pointcloud_foundation.ply` |
| `stereo_pointcloud_downsampled.ply` | `stereo_pointcloud_downsampled_dav2.ply` | `stereo_pointcloud_downsampled_foundation.ply` |
| — | `depth_metric_dav2.npy`, `depth_metric_dav2_gt.npy` | `disparity_foundation.npy` |
| — | `depth_preview_dav2.png`, `depth_preview_dav2_gt.png` | `disparity_preview_foundation.png` |
| — | `depth_scaling_dav2.json`, `depth_scaling_dav2_gt.json` | — |

Shared for all methods:

- `rgb1_rectified.png`, `rgb2_rectified.png`, `rectification_check.png`

### LiDAR validation metrics (`validation/`)

| Method | Metrics file |
|--------|----------------|
| OpenCV | `lidar_stereo_error_metrics.json` |
| DA-V2 (OpenCV scale) | `lidar_ray_depth_metrics_dav2.json` |
| DA-V2 (GT anchors) | `lidar_ray_depth_metrics_dav2_gt.json` |
| FoundationStereo | `lidar_stereo_error_metrics_foundation.json` |

Lower **median_error** (meters) = better agreement with the 2D LiDAR scan in the rectified RGB1 frame.

### Moving runs Mac ↔ Windows

1. Capture and OpenCV on either machine → `outputs/runs/<run_id>/`.
2. Copy that folder to Windows for FoundationStereo if needed.
3. Copy back `depth/*_foundation*` or run DA-V2 on either platform.
4. Compare on either machine with `compare_stereo_methods.py` and `04_view_pointcloud.py`.

---

## Method details

### 1. OpenCV (`02_make_stereo_pointcloud.py`)

- Classical stereo: **StereoBM** (`--method stereobm`) or SGBM / optical flow.
- Metric depth from `cv2.reprojectImageTo3D(disparity, Q)`.
- Works on **Python 3.12+** in your normal env (no conda GPU stack).

### 2. Depth Anything V2 (`02_make_depth_anything_pointcloud.py`)

- Monocular depth on **rectified RGB1**; conda env **`depth_anything_v2`**.
- Relative depth → **metric meters** by linear fit to OpenCV metric depth (`disparity.npy` + `Q`). Default **`--scale-modes both`** writes two evaluation methods from one inference:
  - **`dav2`** — `depth_metric_dav2.npy` — fit on all valid OpenCV Z.
  - **`dav2_gt`** — `depth_metric_dav2_gt.npy` — fit only where OpenCV Z is within ±5 cm of **pair_*.txt** target or **100 cm** wall (GT overlay anchors). Skipped if `pair_*.txt` is missing when using `both`.
- Use `--scale-modes opencv` or `opencv-gt` to save only one variant.
- Options: `--encoder vits` (default, fast), `vitb`, `vitl`; `--input-size 518` (try larger for detail).
- `xFormers not available` on Mac is normal and not fatal.

### 3. FoundationStereo (`02_make_stereo_pointcloud_foundation.py`)

- Learned stereo on rectified pair; conda env **`foundation_stereo`**; **CUDA required**.
- Same `Q` matrix as OpenCV for fair point-cloud comparison.
- Options: `--scale 0.5` (faster), `--valid-iters 16`.

---

## Useful flags

| Flag | Used on |
|------|---------|
| `--run latest` | All steps (or specific run folder name) |
| `--list-runs` | Any script with run CLI |
| `--reuse-rectified` | DA-V2, FoundationStereo (skip re-rectify if `rgb1_rectified.png` exists) |
| `--scale-modes both` / `opencv` / `opencv-gt` | DA-V2 metric scaling variants (see step 2 above) |
| `--depth-min` / `--depth-max` | OpenCV, DA-V2 (scene depth band, default 0.45–2.0 m) |

---

## Troubleshooting

| Issue | What to do |
|-------|------------|
| Low OpenCV coverage on carpet | Try DA-V2; improve lighting/texture; calibrate RGB2 |
| `02_make_depth_anything` needs `disparity.npy` | Run `02_make_stereo_pointcloud.py` first |
| FoundationStereo on Mac | Use Windows + NVIDIA GPU, or use DA-V2 on Mac |
| DA-V2 / FS conda errors | Remove env and re-run setup script; install Miniconda |
| LiDAR vs stereo look misaligned in 3D | Re-run `03_validate_with_lidar.py` after calibration fixes; check `rectification_check.png` |
| `FileNotFoundError` for validation PLY | Run `03_validate_with_lidar.py` once (creates shared LiDAR PLY) |

---

## Script index

| Step | Script |
|------|--------|
| Capture | `01_capture_one_set.py` |
| OpenCV depth | `02_make_stereo_pointcloud.py` |
| Depth Anything V2 | `02_make_depth_anything_pointcloud.py` |
| FoundationStereo | `02_make_stereo_pointcloud_foundation.py` |
| Compare methods | `compare_stereo_methods.py` |
| LiDAR validation | `03_validate_with_lidar.py` |
| 3D viewer | `04_view_pointcloud.py` |
| RGB overlays | `05_project_lidar_overlay.py` |

Setup scripts (repo root `scripts/`): `setup_depth_anything_v2.sh`, `setup_depth_anything_v2.ps1`, `setup_foundationstereo.ps1`, `setup_foundationstereo.sh` (Mac skip).
