# FR-SLAM 稳定基线启动命令

当前稳定基线：

```text
Commit: c4bae79
Tag:    fr-slam-deterministic-baseline-20260921
Branch: refactor/lio-cleanup-v1
```

> 离线 rosbag 实验必须使用 `RELIABLE` 模式。
> 实机传感器运行使用 `BEST_EFFORT` 模式。
> 当前离线确定性基线已经验证：相同 bag 的 callback / queue / process / LIO state / Loop Samples / SC / BTC / PGO 可重复。

---

## 1. 编译 FR-SLAM

只有源码修改后才需要重新编译。

```bash
cd ~/ros2_ws

source /opt/ros/humble/setup.bash

colcon build \
  --packages-select fr_slam \
  --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  '-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -march=native -mtune=native'

source ~/ros2_ws/install/setup.bash
```

---

## 2. 离线 rosbag / Benchmark 模式

### Terminal 1：切换 FR-SLAM 到 RELIABLE

```bash
cd ~/ros2_ws/src/fr_slam

./scripts/set_qos_mode.sh replay
```

确认：

```bash
grep -nE \
'imu_qos_reliable|lidar_qos_reliable' \
config/fr_slam_livox.yaml
```

应该看到：

```text
imu_qos_reliable: true
lidar_qos_reliable: true
```

然后：

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

启动 FR-SLAM：

```bash
ros2 launch fr_slam lio.launch.py sensor:=livox
```

如果当前实验还有其它 launch 参数，请继续沿用当前已经验证过的参数，不要为了重复性测试同时修改 SLAM 参数。

启动后确认日志中出现：

```text
IMU QoS   | ... reliability=RELIABLE | durability=VOLATILE
LiDAR QoS | ... reliability=RELIABLE | durability=VOLATILE
```

---

### Terminal 2：播放 `20260629` bag

```bash
source /opt/ros/humble/setup.bash

ros2 bag play ~/20260629/ \
  --qos-profile-overrides-path \
  ~/ros2_ws/src/fr_slam/config/rosbag_reliable_qos.yaml \
  --wait-for-all-acked 5
```

**以后论文实验、SC/BTC 对比、消融实验、回归测试、重复性实验都固定使用这条回放命令。**

不要使用下面这种普通回放方式做正式对比实验：

```bash
ros2 bag play ~/20260629/
```

因为此前已经确认 `BEST_EFFORT` 回放会导致不同 run 的 LiDAR callback 输入序列发生变化。

---

## 3. 实机 Livox 模式

先切回 `BEST_EFFORT`：

```bash
cd ~/ros2_ws/src/fr_slam

./scripts/set_qos_mode.sh live
```

确认：

```bash
grep -nE \
'imu_qos_reliable|lidar_qos_reliable' \
config/fr_slam_livox.yaml
```

应该看到：

```text
imu_qos_reliable: false
lidar_qos_reliable: false
```

然后：

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 launch fr_slam lio.launch.py sensor:=livox
```

---

## 4. 离线实验结束后再次确认当前模式

如果接下来继续跑 bag：

```bash
cd ~/ros2_ws/src/fr_slam
./scripts/set_qos_mode.sh replay
```

如果接下来连接真实 Livox：

```bash
cd ~/ros2_ws/src/fr_slam
./scripts/set_qos_mode.sh live
```

---

## 5. 查看当前 Git 稳定基线

```bash
cd ~/ros2_ws/src/fr_slam

git status

git show --no-patch --oneline \
  fr-slam-deterministic-baseline-20260921
```

当前恢复点：

```text
c4bae79 FR-SLAM deterministic replay baseline
```

查看当前代码相对稳定基线的改动：

```bash
git diff fr-slam-deterministic-baseline-20260921
```

如果需要在不破坏当前开发分支的情况下回到稳定基线：

```bash
git switch -c recovery-deterministic-baseline \
  fr-slam-deterministic-baseline-20260921
```

---

## 6. 当前固定运行规则

```text
================ OFFLINE / ROSBAG =================

FR-SLAM subscriber:
    IMU   = RELIABLE
    LiDAR = RELIABLE

rosbag publisher:
    IMU   = RELIABLE
    LiDAR = RELIABLE

Command:
    ./scripts/set_qos_mode.sh replay

===================================================


==================== LIVE SENSOR ==================

FR-SLAM subscriber:
    IMU   = BEST_EFFORT
    LiDAR = BEST_EFFORT

Command:
    ./scripts/set_qos_mode.sh live

===================================================
```

## 7. 当前已经验证的确定性结果

同一个 `~/20260629/` bag，在 RELIABLE replay 下：

```text
callback hash A = B
queue hash    A = B
process hash  A = B

LIO diagnostics:
    3541 frames
    46 columns
    only lio_ms differs

remove lio_ms:
    deterministic_state_cmp=0

Loop Retrieval Samples = identical
SC-SINGLE              = identical
SC-WINDOW              = identical
BTC                     = identical
PGO loop edges          = identical
PGO diagnostics         = identical
```

因此当前离线 benchmark 基线定义为：

```text
FR-SLAM Deterministic Replay Baseline
```
