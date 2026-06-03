## Depth Estimation Pipeline

Once the RGB and stereo calibration steps produced rectified image pairs, we tested three ways to estimate depth. Each method takes either the rectified stereo pair or the rectified RGB1 image and produces depth or disparity that can be converted into a point cloud.

### 1. OpenCV Stereo

OpenCV stereo was the baseline method. It uses the rectified RGB1 and RGB2 images to estimate disparity, then converts disparity into 3D points using the stereo `Q` matrix. We used this because it is transparent, lightweight, and does not require a GPU or learned model.

Parameters considered:

- Stereo method: block matching, semi-global block matching, optical flow, and blended variants.
- Scene-specific method choice, including a low-texture setting for carpet-like surfaces.
- Depth range limits such as minimum and maximum valid depth.

Parameters not fully tuned:+

- We did not exhaustively tune `numDisparities`, `blockSize`, `minDisparity`, `uniquenessRatio`, `speckleWindowSize`, `speckleRange`, `disp12MaxDiff`, or the SGBM smoothness penalties `P1` and `P2`.
- We did not jointly optimize OpenCV stereo settings with camera placement, baseline distance, and calibration quality. The calibration fixes the camera geometry.
- We also did not fully test every OpenCV stereo variant. We did not exhaustively compare StereoBM against all StereoSGBM path aggregation modes, including `MODE_SGBM`, `MODE_HH`, `MODE_SGBM_3WAY`, and `MODE_HH4`. 

### 2. Depth Anything V2

[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) was used as a **monocular** dense-depth method. It takes rectified RGB-L as input and predicts a dense relative depth map. Because monocular depth is not naturally metric, we scaled its output into meters using the OpenCV stereo depth from the same frame as the reference.

### 3. Depth Anything V2 with Ground Truth Anchoring
TODO

### 3. FoundationStereo

[NVIDIA FoundationStereo](https://github.com/NVlabs/FoundationStereo) was included as a learned **stereo method**. Like OpenCV stereo, it takes the rectified RGB1/RGB2 pair as input and predicts disparity. This made it a useful learned alternative to the classical stereo baseline.
