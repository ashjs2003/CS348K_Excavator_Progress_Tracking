# Edge-Deployable Perception Stack for Earthwork Progress Tracking

**Authors:** Ashmitha Jaysi Sivakumar, Yashasvini Gopalan  
**Contact:** ashjs@stanford.edu, ygopalan@stanford.edu

## Summary

Construction sites change continuously, creating a reality capture problem. Our goal is to build a modular computing system that can be retrofitted on heavy equipment to provide structured information about the site. The application of the system is centred around safety, progress tracking and autonomy workflows. For this project, we focus on an excavator-mounted system that (1) captures data in real time, (2) packages data for different levels of analysis, and (3) analyses its localization, the excavated volume, and its composition. 

## Project Goal

At a conceptual level, the system takes in whatever the excavator-mounted reality capture device can observe and outputs information about how much excavation has been completed.

A key goal of the project is to determine:

- What level of output detail is useful
- What input data formats are required
- How data packaging affects computational requirements
- How the system can generalize to future construction reality-capture tasks

## Inputs and Outputs

### Inputs

#### Excavator-Mounted Sensor Data

- Time-stamped LiDAR data
- Time-stamped RGB image data
- GPS data
- IMU data

#### Ground Truth Data

- Manual volume measurements
- Material in scoop
- Terrain topography

### Outputs

- Volume excavated over time
- Spatial progress map of completed excavation
- Material composition estimate *(stretch goal)*

## Design Constraints

- Robustness vs. Perfection: Construction sites are dusty, cluttered, occluded, and constantly changing. The system prioritizes robust, best-effort outputs over perfect reconstruction. When the system is inaccurate, it should explicitly communicate the reason, such as sensor failure, occlusion or localization noise.

- Remote Connectivity: Construction sites may have limited connectivity, making centralized processing difficult. However, due to the size and weight limitations of the toy excavator prototype, the project may initially assume centralized processing. The long-term goal is to run urgent perception tasks in real time on equipment, while non-urgent analytics can remain batch-processed or cloud-based.

## System Tasks

### 1. Hardware Choice and Installation

- Design the hardware stack based on cost, weight, and dimension constraints
- Select and integrate components such as:
  - Camera
  - LiDAR
  - GPU or edge compute device
  - IMU
  - Raspberry Pi or similar controller
- Decide between cloud and edge processing based on installation feasibility

### 2. Sensor Placement

- Identify three potential sensor mounting locations on the excavator
- Mount the camera and LiDAR on the toy excavator
- Calibrate and validate the sensor setup using known geometry

### 3. Algorithm and Data Packaging Design

#### Localization Algorithm

Possible localization approaches include:

- Positional encoders
- GPS
- SLAM
- A combination of the above

#### Excavation Progress Tracking Algorithm

Candidate approaches include:

- **RGB-only:** Estimate the volume of the filled scoop and calculate excavated volume over time
- **LiDAR-only:** Estimate volume change after each surface reconstruction
- **LiDAR + RGB:** Combine geometric reconstruction with visual context to estimate progress and material composition

#### Computational Efficiency

The project will couple:

- Data formats
- Localization stack
- Vision algorithms

This coupling is intended to make the perception pipeline computationally efficient and adaptable.

### 4. Building the Outputs

The system should produce:

- Volume excavated vs. time logs
- Excavation completion heatmaps as per-cell volume change estimates
- Reconstructed Surface view history

The heatmap output should ingest the SLAM map and sensor data to show spatial progress across the excavation area.

### 5. Testing

The testing plan includes:

- Capturing at least 20 digging sequences
- Testing across rock, soil, and gravel
- Testing three sensor placements
- Recording sensor data against ground truth
- Running the designed algorithms
- Evaluating accuracy against ground truth
- Analyzing sensitivity to:
  - Localization noise
  - Grid resolution
- Identifying the best configuration

### 6. Feedback

The team will obtain feedback from construction professionals to evaluate whether the system fits the intended construction-site application.

## Team Responsibilities

### Ashmitha

- Hardware configuration
- Localization algorithms

### Yashasvini

- Data collection
- Vision algorithms

### Shared Responsibilities

- System testing
- Evaluation
- Feedback from construction professionals

## Expected Deliverables

The primary deliverable is an end-to-end modular excavation progress tracking system that runs on logged LiDAR, RGB, GPS, and IMU data from a toy excavator.

The system should produce outputs at different levels of detail:

1. Excavated volume-over-time log
2. Spatial progress heatmap
3. Material composition estimate *(stretch goal)*

The system should operate within an on-machine compute budget or be structured to eventually support on-machine deployment.

## Evaluation Plan

The project will evaluate the system on one realistic multi-cycle digging session and measure:
- Runtime Performance: The system should keep up with the sensor stream and process each new frame fast enough to be usable during operation.
- Accuracy and Stability: The estimated excavated volume and progress heatmaps should remain within approximately **10–15% error** and should not fluctuate wildly under slight localization noise.

## Long-Term Vision

Although the prototype is focused on excavation progress monitoring, when we design this pipeline, we hope to structure it in a way that it can be adapted to any reality capture task for a construction site in the future by defining the data packaging and the pros and cons from the outputs based on the level of detail and computational requirement. In other words, theoretically we will define the core layer of the system robust to any task but end-to-end test it for excavation progress monitoring within the scope of the class. 

## Biggest Risks

- Physical Testbed Risk: The project depends on having a stable experimental environment for the 1:14-scale toy excavator. Without a fixed space for the excavator, sand, and sensing equipment, it will be difficult to collect consistent data. High-resolution ground truth after every excavation cycle is also difficult to obtain, making the testbed a foundational dependency.

- Sensing and SLAM Risk: LiDAR- or camera-based localization on a small, noisy platform introduces significant uncertainty. If SLAM drifts, fails, or cannot maintain a stable map, downstream perception tasks may fail.

- Dynamic Earthwork Risk: Earthwork is difficult because the terrain is actively changing during observation. Dust, occlusions, moving machinery, and changing geometry make it difficult to recover clean, temporally consistent signals.
