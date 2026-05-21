# Stereo RGB + 2D LiDAR Point Cloud Workflow

This folder captures one RGB1/RGB2/LiDAR set, builds a stereo point cloud,
validates it against the 2D LiDAR, and opens viewers/overlays.

The workflow avoids Open3D so it can run more easily on Python 3.12. It writes
standard ASCII `.ply` files directly and uses Matplotlib for the lightweight 3D
viewer. If SciPy is installed, validation uses `scipy.spatial.cKDTree`; otherwise
it falls back to a slower chunked nearest-neighbor search.

RGB2 should ideally be calibrated separately. Reusing same-model camera
intrinsics is useful for a quick prototype, but real cameras can still have
different focal lengths, principal points, and lens distortion.

## Calibration Files

The scripts look in this folder, the repo root, `../config`, and
`../stereo_calibration`.

Expected files:

- `camera_calibration_rgb1.npz`
- `camera_calibration_rgb2.npz`
- `stereo_rgb1_rgb2_extrinsics.npz`
- `lidar_to_rgb1_extrinsics.npz`

Fallbacks supported by the code:

- `camera_calibration_rgb2_approx.npz`
- `lidar_to_camera_extrinsics.npz`

## Run Order

Run commands from this folder:

```powershell
cd "C:\Ashmitha\Stanford\Quarter 3\CS 348K\CS348K_Excavator_Progress_Tracking\stereo_lidar_pointcloud"
```

1. Capture one synchronized-ish set:

```powershell
python 01_capture_one_set.py
```

2. Build the stereo point cloud:

```powershell
python 02_make_stereo_pointcloud.py
```

3. Validate stereo geometry against LiDAR:

```powershell
python 03_validate_with_lidar.py
```

4. View the point cloud:

```powershell
python 04_view_pointcloud.py
```

Optional modes:

```powershell
python 04_view_pointcloud.py --mode stereo
python 04_view_pointcloud.py --mode lidar
python 04_view_pointcloud.py --mode both
```

5. Project LiDAR onto both RGB images:

```powershell
python 05_project_lidar_overlay.py
```

## Outputs

- `../capture/rgb1.png`
- `../capture/rgb2.png`
- `../capture/lidar_scan.csv`
- `../capture/metadata.json`
- `../outputs/rgb1_rectified.png`
- `../outputs/rgb2_rectified.png`
- `../outputs/rectification_check.png`
- `../outputs/disparity.npy`
- `../outputs/disparity_preview.png`
- `../outputs/stereo_pointcloud.ply`
- `../outputs/stereo_pointcloud_downsampled.ply`
- `../outputs/lidar_points_in_rgb1_frame.ply`
- `../outputs/lidar_stereo_error_metrics.json`
- `../outputs/lidar_stereo_error_per_point.csv`
- `../outputs/lidar_overlay_rgb1.png`
- `../outputs/lidar_overlay_rgb2.png`
