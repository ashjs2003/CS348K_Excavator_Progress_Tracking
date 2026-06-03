# Conclusion

The 1:18 toy excavator setup was useful as a low-cost platform for learning about calibration, sensor placement, and model failure modes, but it is **not accurate enough to validate production excavation-volume algorithms**.

Calibration was the strongest result. RGB intrinsics reached subpixel reprojection error: **0.265 px** for RGB-L and **0.262 px** for RGB-R. Stereo calibration was also usable, with **0.712 px** RMS error, **0.543 px** mean rectification vertical error, and a **0.0732 m** baseline. This shows that careful checkerboard visibility, corner ordering checks, and outlier filtering can produce reasonable camera geometry.

The limiting factor was metric depth. Best cases reached a few centimeters of error, especially DA-V2 with ground-truth anchoring at mid/far distances, but typical errors were often **20-40 cm**. That is too large relative to the box depths and toy-scale excavation geometry, so volume estimates become unreliable. LiDAR-derived width was useful diagnostically, but the 2D LiDAR often underestimated width and struggled with small boxes.

**SWOT Summary**

- **Strengths:** Identified calibration/data requirements; compared OpenCV, DA-V2, FoundationStereo, and GT-anchored depth; showed how distance, angle, texture, object size, and sensor resolution interact.
- **Weaknesses:** Depth error is too high for volume estimation; LiDAR-to-RGB calibration lacked reliable correspondences; volume estimation is diagnostic, not deployable.
- **Opportunities:** Improve DA-V2 anchoring with better stereo/scale constraints; use larger baseline, larger workspace, or better depth sensors; design stronger LiDAR-camera calibration targets.
- **Threats:** Toy-scale artifacts may dominate algorithm behavior; low texture and occlusion can break stereo; GT-anchored methods may not transfer to deployment.

Overall, the toy excavator is best understood as a **failure-finding and design-learning platform**. The next testbed should increase the geometric signal with a larger workspace, larger stereo baseline, and better depth or cross-sensor calibration.
