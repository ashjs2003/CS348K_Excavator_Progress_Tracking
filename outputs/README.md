# Outputs layout

Each capture + processing session gets its own folder under `runs/` (nothing is overwritten).

**Do not** store generated depth under `data/<scene>/depth/` — keep `data/` for raw captures only.
Batch outputs go to `outputs/runs/<scene>/pair_<id>/` (migrate old layout with `migrate_data_depth_to_runs.py`).

```text
outputs/
  runs/
    latest.txt              # name of the most recent live capture run
    latest -> <run_dir>/     # symlink to latest live capture
    batch_from_data_summary.json   # index after batch_dav2_data_folders.py
    20260521_143022_carpet/          # live capture (--run latest)
      ...
    checkerboard_data/               # batch import from data/ (one run per pair)
      pair_000/
        run_info.json
        capture/  depth/  validation/  overlays/
      pair_001/
      ...
    excavator_S/
      pair_014/
      ...
    20260521_143022_carpet/          # same layout for every run id:
      run_info.json         # label, timestamps, disparity method, coverage, etc.
      capture/
        rgb1.png
        rgb2.png
        lidar_scan.csv
        metadata.json
      depth/
        stereo_geometry.npz            # Q, P1, R1 for evaluation
        rgb1_rectified.png
        rgb2_rectified.png
        rectification_check.png
        disparity.npy                    # OpenCV
        depth_metric_opencv.npy
        disparity_preview.png
        stereo_pointcloud.ply
        stereo_pointcloud_downsampled.ply
        depth_metric_dav2.npy          # Depth Anything V2 (optional)
        stereo_pointcloud_downsampled_dav2.ply
        disparity_foundation.npy       # FoundationStereo (optional, Windows GPU)
        stereo_pointcloud_downsampled_foundation.ply
      validation/
        evaluation_summary.json        # scorecard (run 06_evaluate_run.py)
        EVAL_REPORT.md                 # plain-language summary (07)
        chart_scorecard.png            # green/yellow/red table (08)
        chart_coverage_and_accuracy.png
        chart_ray_error_histogram.png
        chart_error_vs_range.png
        chart_photometric.png
        lidar_ray_depth_metrics.json   # ray + free-space per method
        lidar_ray_depth_metrics_dav2.json
        lidar_ray_depth_metrics_foundation.json
        photometric_reprojection.json  # OpenCV + Foundation
        cross_method_metrics.json
        consensus_depth_std.png
        consensus_depth_std_on_rgb.png
        lidar_points_in_rgb1_frame.ply
        lidar_stereo_error_metrics.json   # legacy NN cloud (optional)
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
python 02_make_stereo_pointcloud.py --method stereobm
python 02_make_depth_anything_pointcloud.py --reuse-rectified   # optional
python 06_evaluate_run.py --run latest

# Batch all data/ scenes into outputs/runs/<scene>/pair_<id>/
python batch_dav2_data_folders.py --skip-existing --continue-on-error
python 06_evaluate_run.py --run checkerboard_data/pair_000
python compare_stereo_methods.py
python 03_validate_with_lidar.py   # single method only; 06 preferred
python 04_view_pointcloud.py --mode both
python 05_project_lidar_overlay.py

# Use a specific run
python 02_make_stereo_pointcloud.py --run 20260521_143022_carpet --method stereobm

# List runs
python 02_make_stereo_pointcloud.py --list-runs
```

Legacy flat folders (`../capture/`, top-level files in `outputs/`) are still supported as fallback if no runs exist.
