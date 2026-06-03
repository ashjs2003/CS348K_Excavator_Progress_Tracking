# Volume Estimation Evaluation

![LiDAR width estimates compared with measured box width](image-16.png)

**Figure 1.** LiDAR-derived width estimates for the S, M, and L cardboard boxes at 0 degree and -30 degree placements. The x-axis is the measured ruler distance from the setup, the y-axis is the width estimated from the raw LiDAR scan, the blue line is the LiDAR-measured width, and the dotted gray line is the measured/catalog box-face width.

The main takeaway is that the raw 2D LiDAR scan can recover a rough width trend, but it generally **underestimates the true box width**. This is expected because the box face is represented by a small number of scan points, and the detected span depends on where the scan line intersects the object and how cleanly the face edges appear in the range profile.

The -30 degree placement performs better than the 0 degree placement for the medium and large boxes. For the M box, the LiDAR width at -30 degrees rises close to the measured width and stays more stable across distance. For the L box, the -30 degree placement also gets much closer to the true width than the 0 degree placement. This suggests that the angled view can expose a more usable range profile or stronger face edges for larger boxes.

The S box remains difficult in both placements. Its true width is small, so the LiDAR has fewer points across the face and the estimate becomes resolution-limited. The measured LiDAR width stays below the catalog width and changes only slightly with distance, showing that small targets are near the limit of what this LiDAR setup can resolve.

Overall, this supports the diagnostic purpose of the volume experiment: LiDAR-derived width is useful for understanding sensing resolution, but it is not yet reliable enough to serve as a general width estimator. When coupled with RGB-derived depth, the volume estimate inherits both sources of error.
