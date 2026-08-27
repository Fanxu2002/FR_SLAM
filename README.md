# FR-SLAM 使用说明

## 1. 环境

推荐：

- Ubuntu 22.04
- ROS 2 Humble
- C++17

先确保 ROS 2 Humble 已经安装。

---

## 2. 安装依赖

```bash
sudo apt update

sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3-colcon-common-extensions \
    libeigen3-dev \
    libpcl-dev \
    libfmt-dev \
    ros-humble-rclcpp \
    ros-humble-sensor-msgs \
    ros-humble-nav-msgs \
    ros-humble-geometry-msgs \
    ros-humble-visualization-msgs \
    ros-humble-tf2-ros \
    ros-humble-pcl-conversions
```

### 安装 g2o

FR-SLAM 的后端 Pose Graph 使用 g2o。

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

安装后可以检查：

```bash
ls /usr/local/lib/libg2o_core.so
ls /usr/local/lib/libg2o_stuff.so
ls /usr/local/lib/libg2o_solver_eigen.so
ls /usr/local/lib/libg2o_types_slam3d.so
```

---

## 3. 下载 FR-SLAM

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

git clone https://github.com/Fanxu2002/FR_SLAM.git
```

如果想使用 v1.0.0：

```bash
cd FR_SLAM
git checkout v1.0.0
```

---

## 4. 编译

```bash
cd ~/ros2_ws

source /opt/ros/humble/setup.bash

colcon build \
    --packages-select fr_slam \
    --symlink-install \
    --cmake-args -DCMAKE_BUILD_TYPE=Release
```

编译完成后：

```bash
source ~/ros2_ws/install/setup.bash
```

---

## 5. 运行

当前默认输入：

```text
/livox/lidar
/livox/imu
```

先确认话题存在：

```bash
ros2 topic list
```

运行 Scan-to-LocalMap SLAM：

```bash
ros2 run fr_slam lidar_registration_scan2localmap
```

如果要运行 Scan-to-Scan：

```bash
ros2 run fr_slam lidar_registration_scan2scan
```

---

## 6. 常用输出

```text
/lidar_odometry
/corrected_odometry
/lidar_path
/local_map
/raw_keyframe_map
/optimized_map
/optimized_path
/pose_graph_markers
```
