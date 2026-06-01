# LiDAR-Camera Calibration

This folder calibrates the 2D LiDAR into the RGB1 reference frame using the
fisheye RGB calibration and fixed fisheye stereo calibration.

## Current Inputs

- RGB1 intrinsics: `01_rgb_calibration/config/camera_calibration_L_fisheye_no_outliers.npz`
- RGB2 intrinsics: `01_rgb_calibration/config/camera_calibration_R_fisheye_no_outliers.npz`
- Stereo extrinsics: `02_stereo_calibration/config/stereo_rgb1_rgb2_fisheye_extrinsics.npz`
- Checkerboard/LiDAR captures: `data/checkerboard_data*`

The script also searches `00_data_capture/checkerboard_*` first, so newer
captures can be placed there without changing code.

## Run

```powershell
python 01_calibrate_lidar_to_rgb1_fisheye.py
```

To use captures where the full internal checkerboard is not detected:

```powershell
python 01_calibrate_lidar_to_rgb1_fisheye.py --annotate-missing
```

The default fallback is the simple line mode. For each skipped RGB1 frame, click
the two endpoints of the visible LiDAR-hit line on the checkerboard. This works
for this dataset because only distance changes and the LiDAR scan line is
parallel to the board direction.

For a stronger partial checkerboard pose when multiple internal corners are
visible, run:

```powershell
python 01_calibrate_lidar_to_rgb1_fisheye.py --annotate-missing --annotation-mode partial-grid
```

Then click visible **internal checkerboard intersections** and enter each
point's inner-corner `col,row` index. Valid columns are `0-8`; valid rows are
`0-5`.

The points must not all lie on one line. A single visible row or column gives a
direction, but not a full board plane. Use points from at least two rows or two
columns.

The annotations are saved in `outputs/manual_board_corners.json`, so future runs
reuse them automatically.

If the whole outer board rectangle is visible and you prefer clicking its four
outer corners instead, run:

```powershell
python 01_calibrate_lidar_to_rgb1_fisheye.py --annotate-missing --annotation-mode outer
```

## Method

1. Detect checkerboard corners in RGB1.
2. Undistort RGB1 fisheye points.
3. Run `solvePnP` to estimate each checkerboard pose in RGB1.
4. Convert each checkerboard pose to an RGB1 plane `n^T X + d = 0`.
5. If the full internal checkerboard is not detected, optionally use manually
   clicked outer board corners and the known `0.250 m x 0.175 m` board size.
6. Load LiDAR scans and select returns near the expected distance from each
   `pair_###.txt` file. The selected LiDAR segment is also biased toward the
   known board dimension crossed by the LiDAR scan. For this vertical-board
   dataset the default is the shorter side:
   `7 * 0.025 = 0.175 m`. Use `--segment-axis width` if the scan crosses the
   full `0.250 m` board width instead.
7. Optimize `R_lidar_to_rgb1` and `t_lidar_to_rgb1` by minimizing LiDAR
   point-to-plane error.
8. Project selected LiDAR board points into RGB1 and RGB2 using fisheye
   projection and the fixed stereo transform.

## Outputs

- `config/lidar_rgb1_extrinsics.npz`
- `outputs/lidar_rgb1_calibration_eval.json`
- `outputs/lidar_rgb1_calibration_eval_per_capture.csv`
- `outputs/lidar_rgb1_leave_one_distance_out.csv`
- `outputs/rgb1_overlays/*.png`
- `outputs/rgb2_overlays/*.png`

## Latest Result

Usable observations: 6

Skipped observations: 13

Point-to-plane error:

- Mean: `0.0229 m`
- Median: `0.0172 m`
- P90: `0.0529 m`
- Max: `0.0622 m`

Most skipped captures failed RGB1 checkerboard pose detection, and two had
non-numeric distance metadata.
