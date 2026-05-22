# Outputs layout

Each capture + processing session gets its own folder under `runs/` (nothing is overwritten).

```text
outputs/
  runs/
    latest.txt              # name of the most recent run
    latest -> <run_dir>/     # symlink to latest run
    20260521_143022_carpet/
      run_info.json         # label, timestamps, disparity method, coverage, etc.
      capture/
        rgb1.png
        rgb2.png
        lidar_scan.csv
        metadata.json
      stereo/
        rgb1_rectified.png
        rgb2_rectified.png
        rectification_check.png
        disparity.npy                    # OpenCV
        disparity_preview.png
        stereo_pointcloud.ply
        stereo_pointcloud_downsampled.ply
        depth_metric_dav2.npy          # Depth Anything V2 (optional)
        stereo_pointcloud_downsampled_dav2.ply
        disparity_foundation.npy       # FoundationStereo (optional, Windows GPU)
        stereo_pointcloud_downsampled_foundation.ply
      validation/
        lidar_points_in_rgb1_frame.ply
        lidar_stereo_error_metrics.json
        lidar_stereo_error_metrics_dav2.json
        lidar_stereo_error_metrics_foundation.json
      overlays/
        lidar_overlay_rgb1.png
        lidar_overlay_rgb2.png
```

## Commands

```bash
cd stereo_lidar_pointcloud

# New capture run (creates outputs/runs/<timestamp>_<label>/)
python 01_capture_one_set.py --label carpet

# Process that run (default: latest) — see stereo_lidar_pointcloud/README.md for all three depth methods
python 02_make_stereo_pointcloud.py --method carpet
python 02_make_depth_anything_pointcloud.py --reuse-rectified   # optional
python compare_stereo_methods.py
python 03_validate_with_lidar.py
python 04_view_pointcloud.py --mode both
python 05_project_lidar_overlay.py

# Use a specific run
python 02_make_stereo_pointcloud.py --run 20260521_143022_carpet --method carpet

# List runs
python 02_make_stereo_pointcloud.py --list-runs
```

Legacy flat folders (`../capture/`, top-level files in `outputs/`) are still supported as fallback if no runs exist.
