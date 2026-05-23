# Stereo Calibration

## Summary

| Metric | Value |
| --- | ---: |
| Stereo RMS calibration error | 21.703 px |
| Rectification vertical error, mean / p90 / max | 5.660 px / 10.832 px / 32.380 px |
| Epipolar error before rectification, mean / p90 / max | 7.278 px / 15.663 px / 43.359 px |
| Baseline | 0.0540 m |
| Evaluated pairs | 21 |

Lower is better for all pixel-error metrics. The rectification vertical error is
the remaining y-mismatch between corresponding checkerboard corners after
`cv2.stereoRectify`; good rectification should make this close to zero.

Along the way we removed the outliers, so we can better the callibration and improve the scores. But ti did not imporve further. We will have to try generating the pointcloud and validating the lidar depth with it to know for sure.

## Commands

Run from this folder:

```powershell
python 01_evaluate_stereo_image_set.py
python 02_stereo_calibrate_rgb1_rgb2_fixed_intrinsics.py
python 03_save_stereo_calibration_pair_image.py
python 04_evaluate_stereo_calibration.py
```

## Outputs

- `outputs/stereo_calibration_pair_example.png`
- `outputs/stereo_calibration_eval.json`
- `outputs/stereo_calibration_eval_per_pair.csv`
- `outputs/stereo_calibration_epipolar_error_plot.png`
- `outputs/stereo_calibration_rectification_vertical_error_plot.png`
- `outputs/stereo_calibration_error_histograms.png`
- `outputs/stereo_rectified_alignment_example.png`
- `config/stereo_rgb1_rgb2_extrinsics.npz`

## Inputs

- Stereo pairs: `C:\Ashmitha\Stanford\Quarter 3\CS 348K\CS348K_Excavator_Progress_Tracking\02_stereo_calibration\stereo_pairs`
- RGB1 intrinsics: `C:\Ashmitha\Stanford\Quarter 3\CS 348K\CS348K_Excavator_Progress_Tracking\01_rgb_calibration\config\camera_calibration_L_normal_no_outliers.npz`
- RGB2 intrinsics: `C:\Ashmitha\Stanford\Quarter 3\CS 348K\CS348K_Excavator_Progress_Tracking\01_rgb_calibration\config\camera_calibration_R_normal_no_outliers.npz`
- Stereo extrinsics: `C:\Ashmitha\Stanford\Quarter 3\CS 348K\CS348K_Excavator_Progress_Tracking\02_stereo_calibration\config\stereo_rgb1_rgb2_extrinsics.npz`
