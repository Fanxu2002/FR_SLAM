

# FR-SLAM

FR-SLAM is a ROS 2 LiDAR SLAM system supporting LiDAR odometry, ground/wall structural constraints, BTC loop closure, g2o pose graph optimization, and global mapping.

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

## Save Map and Trajectory

```bash
ros2 service call /save_slam_maps std_srvs/srv/Trigger "{}"
```

Each save creates a separate timestamped directory to avoid overwriting previous results.

## Demo

Full demo:


https://github.com/user-attachments/assets/b335c305-ebd7-4d81-80b5-f00bc73c70aa


https://github.com/Fanxu2002/FR_SLAM/releases/download/fr_slam_pre_esikf_20260912/fr_slam_demo.mp4

<!--
After uploading fr_slam_demo_readme.mp4 to GitHub as a user attachment,
replace the line below with the generated user-attachments URL to show
the video directly in the README.

https://github.com/user-attachments/assets/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
-->

## Version

Stable baseline:

```text
fr_slam_pre_esikf_20260912
```

Current development branch:

```text
feature/esikf
```
