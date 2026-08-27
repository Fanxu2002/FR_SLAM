# FR-SLAM

FR-SLAM is a lightweight 3D LiDAR SLAM system implemented in **C++17** and **ROS 2**.

Current release: **v1.0.0**

The current version provides a complete basic SLAM pipeline including LiDAR preprocessing, IMU-assisted motion compensation, Scan-to-LocalMap registration, keyframe/submap management, loop closure detection, pose graph optimization, and optimized global map reconstruction.

> Note: the current IMU module is used for initialization, rotation-only deskew, rotation prediction, and gravity reference. It is **not yet a fully tightly-coupled LiDAR-Inertial Odometry (LIO)** system.

---

## 1. Tested Environment

Recommended environment:

- Ubuntu 22.04
- ROS 2 Humble
- GCC / G++ with C++17 support
- CMake
- colcon
- Eigen3
- PCL
- fmt
- g2o

---

## 2. Main Features

FR-SLAM v1.0.0 currently includes:

- LiDAR point cloud preprocessing
- Voxel downsampling
- SOR / ROR filtering
- IMU initialization
- Rotation-only LiDAR deskew
- IMU relative-rotation prediction
- LiDAR constant-motion translation prediction
- Point-to-plane Scan-to-LocalMap registration
- Hessian degeneracy diagnostics
- Tracking quality gate
- Keyframe detection and management
- LocalMap / Submap management
- Scan Context loop candidate detection
- Historical Submap ICP loop verification
- Temporal / sequence loop consistency checking
- Keyframe-level Pose Graph
- g2o global pose graph optimization
- Gravity Guard
- XY trajectory Shape Guard
- Map-to-Odom correction
- Corrected real-time odometry
- Incremental optimized global map reconstruction

---

## 3. ROS 2 Dependencies

Install ROS 2 Humble first.

Then install common ROS 2 dependencies:

```bash
sudo apt update

sudo apt install -y \
    ros-humble-rclcpp \
    ros-humble-sensor-msgs \
    ros-humble-nav-msgs \
    ros-humble-geometry-msgs \
    ros-humble-visualization-msgs \
    ros-humble-tf2 \
    ros-humble-tf2-ros \
    ros-humble-pcl-conversions
```

If `ros-humble-desktop` is already installed, many of these packages may already exist.

---

## 4. System Dependencies

Install the basic build tools and libraries:

```bash
sudo apt update

sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3-colcon-common-extensions \
    python3-rosdep \
    libeigen3-dev \
    libpcl-dev \
    libfmt-dev
```

Initialize rosdep if needed:

```bash
sudo rosdep init
rosdep update
```

If `rosdep` has already been initialized, skip `sudo rosdep init`.

---

## 5. Install g2o

FR-SLAM uses g2o for pose graph optimization.

The current `CMakeLists.txt` expects the g2o shared libraries to be installed under:

```text
/usr/local/lib
```

Therefore, building and installing g2o from source is currently recommended.

```bash
mkdir -p ~/third_party
cd ~/third_party

git clone https://github.com/RainerKuemmerle/g2o.git
cd g2o

mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON

make -j$(nproc)

sudo make install
sudo ldconfig
```

After installation, check that the libraries exist:

```bash
ls /usr/local/lib/libg2o_core.so
ls /usr/local/lib/libg2o_stuff.so
ls /usr/local/lib/libg2o_solver_eigen.so
ls /usr/local/lib/libg2o_types_slam3d.so
```

If these files do not exist, check your g2o installation path before building FR-SLAM.

For best reproducibility, use the same g2o commit on all computers.

You can check the current g2o commit on a working computer with:

```bash
cd /path/to/g2o
git rev-parse HEAD
```

Then on another computer:

```bash
git checkout <same_commit_hash>
```

---

## 6. Clone FR-SLAM

Create a ROS 2 workspace:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

Clone the repository:

```bash
git clone https://github.com/Fanxu2002/FR_SLAM.git
```

Enter the repository:

```bash
cd FR_SLAM
```

To reproduce the current stable baseline:

```bash
git checkout v1.0.0
```

---

## 7. Build

Build FR-SLAM in Release mode:

```bash
cd ~/ros2_ws

source /opt/ros/humble/setup.bash

colcon build \
    --packages-select fr_slam \
    --symlink-install \
    --cmake-args -DCMAKE_BUILD_TYPE=Release
```

After a successful build:

```bash
source ~/ros2_ws/install/setup.bash
```

If the package was previously built and you want a clean rebuild:

```bash
cd ~/ros2_ws

rm -rf build/fr_slam install/fr_slam

source /opt/ros/humble/setup.bash

colcon build \
    --packages-select fr_slam \
    --symlink-install \
    --cmake-args -DCMAKE_BUILD_TYPE=Release

source install/setup.bash
```

---

## 8. Run

The main Scan-to-LocalMap SLAM node can be started with:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 run fr_slam lidar_registration_scan2localmap
```

A Scan-to-Scan registration node is also available:

```bash
ros2 run fr_slam lidar_registration_scan2scan
```

---

## 9. Default Input Topics

The current main node expects:

```text
/livox/lidar
/livox/imu
```

Before running FR-SLAM, confirm the topics are available:

```bash
ros2 topic list
```

You can also inspect their frequencies:

```bash
ros2 topic hz /livox/lidar
ros2 topic hz /livox/imu
```

---

## 10. Main Output Topics

The current system publishes topics including:

```text
/lidar_odometry
/corrected_odometry
/lidar_path
/pose_graph_before_path
/optimized_path
/local_map
/raw_keyframe_map
/optimized_map
/refined_map
/pose_graph_markers
/tf
/tf_static
```

Refinement debug topics may also be available:

```text
/refinement_historical_target
/refinement_current_before
/refinement_current_after
```

---

## 11. SLAM Architecture

The current pipeline is approximately:

```text
LiDAR + IMU
    |
    v
IMU Initialization
    |
    v
Rotation-only Deskew
    |
    v
Point Cloud Preprocessing
    |
    v
IMU Rotation Prediction
    |
    v
Scan-to-LocalMap Registration
    |
    v
Tracking / Quality Gate
    |
    v
Keyframe Detection
    |
    +------------------------------+
    |                              |
    v                              v
LocalMap / Submap            Scan Context
    |                              |
    |                              v
    |                       Loop Candidate
    |                              |
    |                              v
    |                    Historical Submap ICP
    |                              |
    |                              v
    |                     Loop Consistency Check
    |                              |
    +------------------------------+
                                   |
                                   v
                              Pose Graph
                                   |
                                   v
                                  g2o
                                   |
                                   v
                    Gravity Guard + Shape Guard
                                   |
                                   v
                         Optimized Keyframe Poses
                                   |
                      +------------+------------+
                      |                         |
                      v                         v
               Map-to-Odom Bridge      Incremental Global Map
                      |                         |
                      v                         v
            Corrected Real-time Pose     Optimized Map
```

---

## 12. Current Limitations

FR-SLAM v1.0.0 is an initial working SLAM prototype.

Current limitations include:

- IMU deskew currently uses rotation only
- Translation deskew is not yet implemented
- IMU is not yet tightly coupled into the backend state
- Pose graph information matrices are still relatively simple
- Long-term multi-loop robustness still needs more testing
- False-loop rejection can be further improved
- Ground-truth trajectory evaluation is not yet integrated
- Incremental graph optimization such as iSAM2 is not yet implemented
- Current g2o linking still depends on `/usr/local/lib`

---

## 13. Recommended Next Development Steps

Planned improvements may include:

1. Full translational LiDAR deskew
2. Better loop information / covariance modeling
3. Stronger false-loop rejection
4. Multiple-loop and long-duration experiments
5. Ground-truth trajectory evaluation
6. Tightly-coupled LiDAR-IMU estimation
7. Incremental graph optimization
8. More portable CMake dependency discovery for g2o

---

## 14. Version

Current stable baseline:

```text
v1.0.0
```

To check the current repository version:

```bash
git log -1 --oneline
git describe --tags --always
```

---

## 15. Repository

```text
https://github.com/Fanxu2002/FR_SLAM
```

---

## 16. Quick Setup Summary

For a new Ubuntu 22.04 + ROS 2 Humble computer:

```bash
# 1. Install dependencies
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3-colcon-common-extensions \
    python3-rosdep \
    libeigen3-dev \
    libpcl-dev \
    libfmt-dev

# 2. Install g2o from source
mkdir -p ~/third_party
cd ~/third_party
git clone https://github.com/RainerKuemmerle/g2o.git
cd g2o
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
make -j$(nproc)
sudo make install
sudo ldconfig

# 3. Clone FR-SLAM
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/Fanxu2002/FR_SLAM.git
cd FR_SLAM
git checkout v1.0.0

# 4. Build
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build \
    --packages-select fr_slam \
    --symlink-install \
    --cmake-args -DCMAKE_BUILD_TYPE=Release

# 5. Run
source ~/ros2_ws/install/setup.bash
ros2 run fr_slam lidar_registration_scan2localmap
```

---

FR-SLAM v1.0.0 is intended as a clean baseline for continued development and reproducible testing on multiple computers.
