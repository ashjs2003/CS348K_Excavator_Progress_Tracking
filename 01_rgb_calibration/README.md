# RGB Camera Calibration Workflow

## Summary

- **Inputs:** 23 checkerboard images in `calibration_images/`, 9 x 6 inner corners, 0.025 m square size, 1280 x 720 RGB frames.
- **Outputs:** calibration files in `config/`, dataset QA plots/metrics in `outputs/`, and final L/R intrinsics selected by `../config.yaml`.
- **Result:** 23/23 checkerboards detected, 76% image-grid coverage, good size/pose diversity, and acceptable sharpness.
- **Final L calibration:** `config/camera_calibration_L_normal_no_outliers.npz`, made from 19 kept images after removing `calib_011.png`, `calib_012.png`, `calib_017.png`, and `calib_022.png`.
- **R calibration:** capture R images into `calibration_images_R/`, then run the same calibration commands and choose `R`.
- **Hardware:** L/R camera indices, frame size, LiDAR port, and LiDAR baudrate are set in `../config.yaml`.

## Normal vs Fisheye

| Model | File | Images | RMS reprojection error | Notes |
| --- | --- | ---: | ---: | --- |
| L normal pinhole | `config/camera_calibration_L_normal.npz` | 23 | 1.221 px | Baseline L calibration using all detected images. |
| L normal pinhole, no outliers | `config/camera_calibration_L_normal_no_outliers.npz` | 19 | 1.076 px | Current L default selected by `../config.yaml`. |
| L fisheye | `config/camera_calibration_L_fisheye.npz` | 23 | 1.225 px | Similar numeric fit, but only use if normal undistortion looks wrong near wide-angle edges. |
| L fisheye, no outliers | `config/camera_calibration_L_fisheye_no_outliers.npz` | 19 | 1.040 px | Lowest L RMS, but validate visually before switching the main pipeline to fisheye. |

For this dataset, the cleaned fisheye calibration has the lowest RMS error, while
the cleaned normal pinhole calibration remains the configured default because the
downstream pipeline expects standard OpenCV pinhole intrinsics. Switch to fisheye
only after visual undistortion/projection checks look better with the fisheye model.

This folder calibrates the InnoMaker U20CAM-1080P RGB camera with OpenCV and a printed checkerboard.

Checkerboard settings:

- Printed board: 10 squares across, 7 squares down
- OpenCV inner corners: `(9, 6)`
- Square size: `0.025` meters
- Target resolution: `1280x720`

## How To Use

1. Print the checkerboard at 100% scale.
2. Measure the square size after printing. These scripts use `0.025` meters.
3. Keep the camera fixed during image capture.
4. Move the checkerboard around the frame, not the camera.
5. Capture 20-30 good images.
6. Include center, corners, edges, tilted, near, and far poses.
7. Run normal calibration first.
8. Check RMS and mean reprojection error. Lower is better; bad images usually increase the error.
9. Test undistortion with the normal model.
10. If normal undistortion looks too warped for the wide-angle lens, run fisheye calibration.
11. Use the same resolution for calibration and later LiDAR overlay.

Calibration `.npz` files for this workflow live in `01_rgb_calibration/config/`.
The repo-level `config.yaml` chooses which L/R version downstream workflows should use.

## Commands

Run these from this folder:

```powershell
python 01_capture_checkerboard_images.py
python 02_calibrate_camera_normal.py
python 02c_calibrate_camera_normal_without_outliers.py
python 02d_calibrate_camera_fisheye_without_outliers.py
python 03_undistort_live_normal.py
python 02b_calibrate_camera_fisheye.py
python 03b_undistort_live_fisheye.py
```

To identify camera indices before capture:

```powershell
python 00_live_camera_index_viewer.py
```

Each live stream is labeled with its camera index and whether it is currently
configured as `L` or `R`.

When `--camera` is omitted, capture and calibration scripts ask whether the run
is for `L` or `R`. You can also pass it explicitly:

```powershell
python 01_capture_checkerboard_images.py --camera R
python 02_calibrate_camera_normal.py --camera R
python 02c_calibrate_camera_normal_without_outliers.py --camera R
```

## Step 1: Capture Images

```powershell
python 01_capture_checkerboard_images.py
```

- Press `s` to save only when checkerboard corners are detected.
- Press `q` to quit.
- Images are saved in `calibration_images/` as `calib_000.png`, `calib_001.png`, etc.

## Step 2: Normal Calibration

```powershell
python 02_calibrate_camera_normal.py
```

This creates `config/camera_calibration_L_normal.npz` or
`config/camera_calibration_R_normal.npz` containing:

- `camera_matrix`
- `dist_coeffs`
- `image_size`
- `checkerboard_size`
- `square_size`
- `rms_error`
- `mean_reprojection_error`

This is the baseline calibration. Downstream workflows use the final L/R path
named in `../config.yaml`.

## Step 2c: Normal Calibration Without Outliers

```powershell
python 02c_calibrate_camera_normal_without_outliers.py
```

This first calibrates with all detected checkerboards, removes images whose
per-image reprojection error is above `1.5 px`, then recalibrates with the kept
images. Use `--max-error` to change the cutoff.

It saves:

- `config/camera_calibration_L_normal_no_outliers.npz` or `config/camera_calibration_R_normal_no_outliers.npz`
- `outputs/calibration_L_outlier_filter_report.csv` or `outputs/calibration_R_outlier_filter_report.csv`

## Step 2d: Fisheye Calibration Without Outliers

```powershell
python 02d_calibrate_camera_fisheye_without_outliers.py
```

This reuses the L/R outlier report from Step 2c, keeps the non-outlier images,
and saves:

- `config/camera_calibration_L_fisheye_no_outliers.npz` or `config/camera_calibration_R_fisheye_no_outliers.npz`

## Step 3: Normal Live Undistortion

```powershell
python 03_undistort_live_normal.py
```

- Press `0` for `alpha=0`, which crops invalid pixels.
- Press `1` for `alpha=1`, which keeps more field of view.
- Press `q` to quit.

## Step 4: Fisheye Calibration

```powershell
python 02b_calibrate_camera_fisheye.py
```

This creates `config/camera_calibration_L_fisheye.npz` or
`config/camera_calibration_R_fisheye.npz`. Use this if the normal model does not
look good for the wide-angle lens.

## Step 5: Fisheye Live Undistortion

```powershell
python 03b_undistort_live_fisheye.py
```

- Press `[` to decrease balance for more crop.
- Press `]` to increase balance for more field of view.
- Press `0` for `balance=0.0`.
- Press `1` for `balance=1.0`.
- Press `q` to quit.
