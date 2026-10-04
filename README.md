# FR-SLAM

**FR-SLAM** is a ROS 2 LiDAR SLAM system for real-time LiDAR odometry, LiDAR-inertial odometry, structural constraints, loop closure, pose-graph optimization, and global mapping.

The current system integrates:

- LiDAR / IMU preprocessing and motion compensation
- LiDAR-Inertial Odometry (LIO) with an Iterated Error-State Kalman Filter (IESKF)
- Ground constraints for roll / pitch / vertical stabilization
- Wall constraints for lateral motion and yaw stabilization
- Scan Context based loop-candidate retrieval
- Geometric loop verification
- g2o pose-graph optimization
- Asynchronous post-PGO map refinement
- CUDA-accelerated exact dense-grid KNN and plane fitting
- Global / optimized / refined map export

---

## System Overview

![FR-SLAM System Overview](docs/FR_SLAM_Overview.png)

A PDF version of the system diagram is also available:

[**FR-SLAM System Overview (PDF)**](docs/FR_SLAM_Overview.pdf)

---

## Pipeline

```text
LiDAR + IMU
    │
    ▼
Preprocessing / Deskew
    │
    ▼
IMU Propagation
    │
    ▼
Scan-to-Local-Map LiDAR Update
    │
    ├── Ground Constraint
    └── Wall Constraint
    │
    ▼
IESKF / LIO Pose
    │
    ▼
Local Map + Keyframes
    │
    ▼
Loop Candidate Retrieval
    │
    ▼
Geometric Verification
    │
    ▼
Pose Graph Optimization
    │
    ▼
Asynchronous Post-PGO Refinement
    │
    ▼
Optimized Trajectory + Global Map
```

---

## Key Features

### LiDAR-Inertial Frontend

FR-SLAM uses IMU propagation together with scan-to-local-map point-to-plane constraints inside an IESKF frontend.

The frontend includes IMU initialization and propagation, LiDAR motion compensation, range / ROI filtering, voxel downsampling, outlier rejection, scan-to-local-map registration, and keyframe / local-map management.

### Structural Constraints

- **Ground constraint:** primarily stabilizes roll, pitch, and vertical motion.
- **Wall constraint:** primarily constrains lateral motion and yaw.

These constraints can be enabled or disabled independently from the launch configuration.

### Loop Closure and Backend

The backend includes Scan Context loop-candidate retrieval, geometric candidate verification, pose-graph optimization, structural constraint support, asynchronous post-PGO refinement, and final refinement before map export.

### CUDA Loop Verification

The CUDA loop verifier uses two production optimizations:

1. **Exact dense-grid KNN** — replaces brute-force global KNN search while preserving exact nearest-neighbor semantics.
2. **Fast analytic 3×3 smallest-eigenvector solver** — accelerates local plane fitting and falls back to the validated Jacobi eigensolver for numerically degenerate cases.

---

## Accuracy

The following numbers are representative results from the **Strawberry_4** evaluation sequence.

### Global Trajectory Accuracy

| Metric | FR-SLAM | FAST-LIO2 |
|---|---:|---:|
| Translation RMSE | **0.1439 m** | 0.1803 m |
| Rotation RMSE | 0.8004° | **0.6508°** |
| Z RMSE | 0.0464 m | **0.0286 m** |

On this sequence, FR-SLAM reduced global translation RMSE by approximately **20.2%** relative to FAST-LIO2.

> These results are dataset- and configuration-dependent and are provided as representative experimental results rather than universal performance guarantees.

### Preserve-Z PGO Validation

A validated Preserve-Z PGO experiment produced:

| Metric | Result |
|---|---:|
| Global ATE | **0.0994 m** |
| XY ATE | **0.0943 m** |
| Z RMSE | **0.0357 m** |
| Yaw RMSE | **0.4089°** |
| Rotation APE RMSE | **0.6387°** |

---

## Runtime Optimization

Representative CUDA profiling on the current loop-verification pipeline:

| Stage | Earlier Baseline | Current Production |
|---|---:|---:|
| KNN stage | 96.60 s | **41.49 s** |
| Geometry stage | 13.03 s | **11.23 s** |
| Total CUDA loop-verification time | 114.91 s | **58.22 s** |
| Backend queue backlog | observed previously | **0** |

The full CUDA loop-verification pipeline is approximately **1.97× faster** than the earlier brute-force baseline in this benchmark.

The production fast eigensolver reduced the sampled smallest-eigenvector cost from approximately **101k cycles/query** to **27k cycles/query**, corresponding to approximately **3.7×** speedup for that operation.

> Absolute timings depend on GPU, dataset size, map density, compiler options, and runtime configuration.

---

## Build

Tested with ROS 2 Humble.

```bash
cd ~/ros2_ws

colcon build \
  --packages-select fr_slam \
  --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  '-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -march=native -mtune=native'

source ~/ros2_ws/install/setup.bash
```

---

## Run

### HortiMulti / Outdoor LIO

```bash
ros2 launch fr_slam lio.launch.py \
  sensor:=hortimulti \
  profile:=outdoor \
  backend_loop_closure_enable:=true \
  ground_constraint_enable:=true \
  wall_constraint_enable:=true \
  planar_motion_mode:=false
```

### Livox

```bash
ros2 launch fr_slam lio.launch.py \
  sensor:=livox \
  profile:=indoor \
  backend_loop_closure_enable:=true \
  ground_constraint_enable:=true \
  planar_motion_mode:=false
```

### Hesai

```bash
ros2 launch fr_slam lio.launch.py \
  sensor:=hesai \
  profile:=outdoor \
  backend_loop_closure_enable:=true \
  ground_constraint_enable:=true \
  planar_motion_mode:=false
```

---

## Post-PGO Refinement

The refinement scheduler supports three modes:

```yaml
post_pgo_refinement_mode: "off"    # online refinement disabled
post_pgo_refinement_mode: "sync"   # synchronous refinement
post_pgo_refinement_mode: "async"  # latest-only background refinement
```

The HortiMulti configuration currently uses:

```yaml
post_pgo_refinement_mode: "async"
post_pgo_refinement_loop_stride: 8
```

---

## Save Map and Trajectory

```bash
ros2 service call /save_slam_maps std_srvs/srv/Trigger "{}"
```

Each save creates a separate timestamped output directory to avoid overwriting previous results.

Before saving, FR-SLAM can force a final post-PGO refinement so that the exported refined map corresponds to the latest optimized pose graph.

---

## Demo

Full demo:

https://github.com/user-attachments/assets/b335c305-ebd7-4d81-80b5-f00bc73c70aa

Release video:

https://github.com/Fanxu2002/FR_SLAM/releases/download/fr_slam_pre_esikf_20260912/fr_slam_demo.mp4

---

## Repository Status

The production `main` branch currently includes:

- Exact dense-grid CUDA KNN for loop verification
- Fast analytic plane eigensolver with Jacobi fallback
- Ground / wall structural constraints
- Pose-graph optimization
- Asynchronous post-PGO map refinement
- Final-save refinement and timestamped map export

---

## License

This project is released under the **Apache-2.0 License**.
