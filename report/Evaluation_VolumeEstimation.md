# Volume Estimation Evaluation 

This document evaluates a diagnostic pipeline on controlled cardboard boxes and excavator trench captures. It is not a deployable volumetry product.

## Method

| Component | Cardboard (Figures 1–2) | Excavator (Figure 3) |
| --- | --- | --- |
| Width | Raw 2D LiDAR face span | LiDAR edge profile, with gate-span fallback if the profile fails |
| Height | Ground truth prior (S 7 cm, M 19 cm, L 24 cm) | Ground truth M/S priors (19 cm / 7 cm) |
| Depth | Stereo flap-to-back: median Z in ROI minus median Z in outside ring | Legacy ROI inside/outside Z from `roi_bbox_volume_estimates` |
| Volume | LiDAR width × ground truth height × depth | Same product |

Ground truth for cardboard uses measured box volumes (S 294, M 4940, L 15360 cm³). 

## Intended use

- Stress-test whether toy-scale LiDAR plus RGB depth carry enough metric signal for known boxes.
- Compare depth methods and placements under manual ROIs.
- Isolate LiDAR width versus bbox width on excavator while holding stereo depth fixed.

## Out of scope

- Production excavation volumetry or progress tracking without trench ground truth.
- Treating near-proxy excavator cells or occasional near-GT cardboard cells as validated volume.
- Using DA-V2 GT as a deployable volume method. 

## Evaluation data

- S/M/L cardboard at 0° and 30°, with multiple ruler distances per scene.
- `excavator_M` and `excavator_S`, six captures each, with range gating via stereo `z_inside` because excavator scenes have no GT ruler.

---

## Results

### LiDAR width

![LiDAR width estimates compared with measured box width](image-16.png)

**Figure 1.** LiDAR width versus ruler distance for S/M/L at 0° and −30°. The dotted line is ground truth face width.

| Observation | Detail |
| --- | --- |
| Bias | LiDAR generally underestimates ground truth width because the face has sparse hits and the span depends on edge detection. |
| Angle | −30° helps M and L. Width tracks ground truth more closely than at 0°. |
| Small targets | The S box stays resolution-limited in both angles. |

**Summary:** Raw 2D LiDAR scan can recover a rough width trend, but it generally underestimates the true box width. 
This is expected because the box face is represented by a small number of scan points, and the detected span depends on where the scan 
line intersects the object and how cleanly the face edges appear in the range profile.

The -30 degree placement performs better than the 0 degree placement for the medium and large boxes. For the M box, the LiDAR width at -30 
degrees rises close to the measured width and stays more stable across distance. For the L box, the -30 degree placement also gets much 
closer to the true width than the 0 degree placement. This suggests that the angled view can expose a more usable range profile or 
stronger face edges for larger boxes.

The S box remains difficult in both placements. Its true width is small, so the LiDAR has fewer points across the face and the estimate 
becomes resolution-limited. The measured LiDAR width stays below the catalog width and changes only slightly with distance, showing that 
small targets are near the limit of what this LiDAR setup can resolve.

---

### Cardboard volume

![Volume percent error for cardboard boxes](../outputs/runs/_combined/volume_error_dashboard.png)

**Figure 2.** Signed percent error `(V_est − V_GT) / V_GT` by size, angle, distance, and method (OpenCV, DA-V2, DA-V2 GT, Foundation).

| Metric | Value |
| --- | --- |
| Cells within ±25% of GT | 3 total: M at 30° and 20 cm (OpenCV), L at 30° and 75 cm (DA-V2), S at 30° and 100 cm (DA-V2) |
| Typical error sign | Strong negative, from width underestimate multiplied by thin flap depth |
| Learned methods | They cluster in a narrow underestimation band. DA-V2 GT is often flat after global scaling, so its depth term collapses. |
| OpenCV | It is bimodal. A few cells approach GT, but others overshoot by hundreds of percent when flap depth spikes. A median signed error near +20% reflects those spikes. |

**Summary:** LiDAR width and RGB flap depth do not recover ground truth-accurate volume at toy scale. Near-GT cells are rare exceptions.

**Recommendation:** DA-V2 performs better than current implementation of DA-V2 GT as DA-V2 has more points of references. Anchor DA-V2 with several measured depths at close spacing instead of a single global scale. That may more sensitive to small depth change and recover a more usable box-depth term.

---

### Excavator volume

![Excavator M and S volume table](../outputs/runs/_combined/excavator_MS_lidar_stereo_volume_all_views.png)

**Figure 3.** LiDAR width × ground truth height × stereo depth per capture. Cell color shows percent error versus cardboard proxy GT.

| Observation | Detail |
| --- | --- |
| LiDAR width | It is present on M rows but does not stabilize volume. |
| Depth driver | Most scatter comes from stereo depth using the legacy heuristic, not box flap-to-back depth. |
| OpenCV | It is inconsistent across M versus S. It sometimes lands near the proxy on M and can overshoot badly on S (for example pair 016). |
| DA-V2 / Foundation | They keep depth thin and underestimate versus the proxy. |
| vs `excavator_MS_all_views` | The stereo depths are the same. This table swaps bbox width for LiDAR width. |

**Summary:** On excavator captures, LiDAR width does not make volume reliable. Stereo depth still drives most of the error. OpenCV is unstable across trench size, DA-V2 and Foundation stay low.

---

## Limitations

- LiDAR width underestimates ground truth face width, especially for S boxes and 0° placements.
- Flap-to-back depth is often only a few centimeters for learned maps, so volume error grows multiplicatively.
- DA-V2 GT global scaling can flatten ROIs and zero out depth.
- OpenCV disparity noise can produce rare but very large volume overshoots.
- ROIs are currently marked manually.

## Future work

- Add multi-point depth anchoring for DA-V2 and re-evaluate flap depth.
- Fix LiDAR–RGB calibration if projecting width into the image is required later.
