# Stereo Calibration

## Summary

| Metric | Value |
| --- | ---: |
| Stereo RMS calibration error | 0.712 px |
| Rectification vertical error, mean / p90 / max | 0.543 px / 1.124 px / 5.095 px |
| Epipolar error before rectification, mean / p90 / max | 0.600 px / 1.236 px / 5.601 px |
| Baseline | 0.0732 m |
| Evaluated pairs | 37 |

Lower is better for all pixel-error metrics. The rectification vertical error is
the remaining y-mismatch between corresponding checkerboard corners after
`cv2.fisheye.stereoRectify`; good rectification should make this close to zero.

## Commands

Run from this folder:

```powershell
python 01_evaluate_stereo_image_set.py
python 02b_stereo_calibrate_fisheye_fixed_intrinsics.py
python 04_evaluate_stereo_calibration.py
```

Optional visual check:

```powershell
python 03b_save_numbered_stereo_corners.py
```

## Outputs

- `outputs/fisheye_no_outliers/stereo_calibration_eval.json`
- `outputs/fisheye_no_outliers/stereo_calibration_eval_per_pair.csv`
- `outputs/fisheye_no_outliers/stereo_calibration_epipolar_error_plot.png`
- `outputs/fisheye_no_outliers/stereo_calibration_rectification_vertical_error_plot.png`
- `outputs/fisheye_no_outliers/stereo_calibration_error_histograms.png`
- `outputs/fisheye_no_outliers/stereo_rectified_alignment_example.png`
- `config/stereo_rgb1_rgb2_fisheye_extrinsics.npz`
- `stereo_pairs/numbered_stereo_corners_###.png`

## Inputs

- Stereo pairs: `C:\Ashmitha\Stanford\Quarter 3\CS 348K\CS348K_Excavator_Progress_Tracking\00_data_capture\int_ext_calib_rgb`
- RGB1 intrinsics: `C:\Ashmitha\Stanford\Quarter 3\CS 348K\CS348K_Excavator_Progress_Tracking\01_rgb_calibration\config\camera_calibration_L_fisheye_no_outliers.npz`
- RGB2 intrinsics: `C:\Ashmitha\Stanford\Quarter 3\CS 348K\CS348K_Excavator_Progress_Tracking\01_rgb_calibration\config\camera_calibration_R_fisheye_no_outliers.npz`
- Stereo extrinsics: `C:\Ashmitha\Stanford\Quarter 3\CS 348K\CS348K_Excavator_Progress_Tracking\02_stereo_calibration\config\stereo_rgb1_rgb2_fisheye_extrinsics.npz`
- Model: `fisheye`
