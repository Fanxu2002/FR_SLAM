# FR-SLAM

FR-SLAM 是一个基于 **ROS 2 Humble + C++17** 开发的 3D LiDAR SLAM 系统，主要面向室外与农业机器人场景。

目前系统已经完成从 LiDAR 前端里程计、局部地图、回环检测到后端位姿图优化的完整 SLAM 流程，并加入 IMU 辅助 Deskew 与旋转预测。

## 目前实现

- Scan-to-LocalMap point-to-plane ICP
- IMU rotation-only Deskew
- IMU 相对旋转作为 ICP 初始值
- Keyframe + LocalMap
- Huber 鲁棒核
- Hessian 退化检测与归一化
- Local Ground Constraint
- Scan Context 回环检测
- Loop Geometry Verification
- g2o Pose Graph Optimization
- Post-PGO Map Refinement
- 异步 Backend
- Mid-360 / Hesai LiDAR Adapter
- YAML 配置 LiDAR / IMU Topic 与预处理参数
- 地图与轨迹 Snapshot 保存

## 主要解决的问题

目前 FR-SLAM 主要针对以下问题进行了处理：

- LiDAR 扫描过程中由运动造成的点云畸变
- Scan-to-LocalMap 配准中的错误对应点与异常值
- 退化环境下 Hessian 弱方向导致的位姿不稳定
- 斜坡与非水平地面环境下 roll / pitch / z 漂移
- 回环误匹配与错误 Loop Edge
- Pose Graph 优化后地图整体错位与局部不一致
- 长时间运行中的前端 / 后端计算负载问题
- 不同 LiDAR 数据格式之间的兼容问题
- 多次实验地图和轨迹相互覆盖的问题

当前系统仍属于 **LiDAR SLAM + IMU-assisted frontend**，IMU 主要用于 Deskew 和旋转预测，尚未实现紧耦合 LIO。

## 启动

```bash
cd ~/ros2_ws

colcon build \
    --packages-select fr_slam \
    --cmake-args -DCMAKE_BUILD_TYPE=Release

source install/setup.bash

ros2 launch fr_slam fr_slam.launch.py
```

## 保存地图与轨迹

```bash
ros2 service call /save_slam_maps std_srvs/srv/Trigger "{}"
```

每次保存都会生成独立的时间戳目录，避免覆盖之前的实验结果。
