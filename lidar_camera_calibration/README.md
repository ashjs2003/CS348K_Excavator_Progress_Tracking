# LiDAR-Camera Calibration Workflow

This is a prototype workflow for calibrating a 2D LiDAR to the already calibrated RGB camera.

Run all commands from this `lidar_camera_calibration` folder.

## Assumptions

- RGB calibration exists at `../rgb_calibration/camera_calibration_normal.npz`.
- LiDAR is an RPLidar-compatible device on `COM5` at baudrate `460800`.
- Checkerboard is mounted flat on a rigid board.
- The 2D LiDAR scan plane intersects the checkerboard board edge-to-edge.
- The board is kept within `1.0 m` during capture.
- Units are meters in saved calibration files.

## Fixed-Length Segment Setup

The shared settings live in `calibration_settings.py`.

For a 10 x 7 square checkerboard with 25 mm squares:

- Full board width: `10 * 0.025 = 0.250 m`
- Full board height: `7 * 0.025 = 0.175 m`
- One square only: `0.025 m`

The workflow currently assumes the LiDAR segment crosses the full board width:

```python
EXPECTED_SEGMENT_AXIS = "width"
EXPECTED_SEGMENT_LENGTH_M = 0.250
```

If your LiDAR scan crosses the full board height instead, change this in `calibration_settings.py`:

```python
EXPECTED_SEGMENT_AXIS = "height"
```

Only use `0.025 m` if the LiDAR segment is intentionally one square long, not the full board.

## Steps

1. Capture RGB + LiDAR pairs:

```powershell
python 01_capture_rgb_lidar_pairs.py
```

Press `s` to save pairs into `pairs/`.

Press `p` to save the pair and immediately select the LiDAR board segment. The selection helper prints:

- selected point count
- fitted LiDAR segment length
- length error compared to the expected board length
- RMS line residual

A good capture has checkerboard corners detected in RGB and a clean, straight LiDAR segment with length near the configured expected length.

2. Detect checkerboard pose in each RGB image:

```powershell
python 02_detect_checkerboard_pose.py
```

This saves checkerboard plane estimates into `checkerboard_poses/`.

3. Select the LiDAR points that hit the board:

```powershell
python 03_select_lidar_board_points.py
```

For each pair, drag a box around the LiDAR returns from the board and press Enter.

This also saves fitted LiDAR segment endpoints in `selected_lidar_points/`. The optimizer uses these endpoints as the known edge-to-edge board constraint.

4. Tune a manual overlay first:

```powershell
python 04_manual_lidar_camera_overlay.py
```

Use the trackbars to adjust `tx`, `ty`, `tz`, `roll`, `pitch`, and `yaw`. Press `s` to save `lidar_to_camera_extrinsics_manual.npz`.

5. Optimize extrinsics using the selected LiDAR board points:

```powershell
python 05_optimize_lidar_to_camera_extrinsics.py
```

The objective is point-to-plane distance: transformed LiDAR board points should lie on the checkerboard plane estimated from the RGB image.

With the fixed-length edge-to-edge setup, the optimizer also constrains the transformed LiDAR segment endpoints to the known opposite board edges.

6. Validate projection:

```powershell
python 06_validate_lidar_projection.py
```

Press `u` to toggle raw vs undistorted image, `s` to save the overlay, and `q` to quit.

## Capture Tips

- Keep the camera and LiDAR rigidly fixed together.
- Move the checkerboard board to several locations and angles.
- Make sure the LiDAR scan plane crosses the whole intended board dimension edge-to-edge.
- Keep the board within 1 m so the capture preview filters away most background points.
- Capture poses at different distances and horizontal positions.
- Start with manual overlay. The automatic optimization has limited constraints because a single 2D LiDAR scan gives points on a line, not a full 3D surface.
