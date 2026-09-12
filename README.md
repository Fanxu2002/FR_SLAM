# FR-SLAM

FR-SLAM is a ROS 2 LiDAR SLAM system for indoor and outdoor robotic mapping.

The current system includes:

- Livox / Hesai LiDAR support
- IMU integration and LiDAR deskew
- Scan-to-local-map LiDAR odometry
- Ground plane detection
- Persistent wall association
- BTC loop closure
- g2o pose graph optimization
- Incremental global mapping

A tightly coupled ESIKF frontend is currently under development.

---

## Platform

Tested on:

- Ubuntu 22.04
- ROS 2 Humble
- PCL 1.12
- Eigen3
- OpenCV
- Ceres Solver

---

## Clone

```bash
git clone --recursive git@github.com:Fanxu2002/FR_SLAM.git
cd FR_SLAM
```

If the repository has already been cloned:

```bash
git submodule update --init --recursive
```

---

## Build

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

### Livox

```bash
ros2 launch fr_slam lo.launch.py \
  sensor:=livox \
  profile:=indoor \
  planar_motion_mode:=false \
  ground_constraint_enable:=true \
  backend_loop_closure_enable:=true
```

### Hesai

```bash
ros2 launch fr_slam lo.launch.py \
  sensor:=hesai \
  profile:=outdoor \
  planar_motion_mode:=false \
  ground_constraint_enable:=true \
  backend_loop_closure_enable:=true
```

---

## BTC Loop Closure

BTC configuration files are stored in:

```text
config/btc/
├── config_indoor.yaml
└── config_outdoor.yaml
```

The official BTC implementation is included as a Git submodule:

```text
third_party/upstream/btc_descriptor
```

The current outdoor BTC profile uses:

```yaml
voxel_size: 1.5
```

---

## 保存地图与轨迹

运行 SLAM 后，可通过 ROS 2 service 保存当前地图与轨迹：

```bash
ros2 service call /save_slam_maps std_srvs/srv/Trigger "{}"
```

每次保存都会生成独立的时间戳目录，避免覆盖之前的实验结果。

---

## Demo

[Watch the FR-SLAM running demo](https://github.com/Fanxu2002/FR_SLAM/releases/download/fr_slam_pre_esikf_20260912/fr_slam_demo.mp4)

---

## Version

Stable pre-ESIKF baseline:

```text
fr_slam_pre_esikf_20260912
```

Current ESIKF development branch:

```text
feature/esikf
```

---

## License

For research use.
