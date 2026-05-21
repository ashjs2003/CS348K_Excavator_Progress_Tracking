# RGB Camera Calibration Workflow

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

## Commands

Run these from this folder:

```powershell
python 01_capture_checkerboard_images.py
python 02_calibrate_camera_normal.py
python 03_undistort_live_normal.py
python 02b_calibrate_camera_fisheye.py
python 03b_undistort_live_fisheye.py
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

This creates `camera_calibration_normal.npz` containing:

- `camera_matrix`
- `dist_coeffs`
- `image_size`
- `checkerboard_size`
- `square_size`
- `rms_error`
- `mean_reprojection_error`

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

This creates `camera_calibration_fisheye.npz`. Use this if the normal model does not look good for the wide-angle lens.

## Step 5: Fisheye Live Undistortion

```powershell
python 03b_undistort_live_fisheye.py
```

- Press `[` to decrease balance for more crop.
- Press `]` to increase balance for more field of view.
- Press `0` for `balance=0.0`.
- Press `1` for `balance=1.0`.
- Press `q` to quit.
