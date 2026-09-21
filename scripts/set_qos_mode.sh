#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/ros2_ws/src/fr_slam"
CFG="${ROOT}/config/fr_slam_livox.yaml"

MODE="${1:-}"

case "${MODE}" in
    replay|benchmark)
        VALUE="true"
        ;;
    live|sensor)
        VALUE="false"
        ;;
    *)
        echo "Usage:"
        echo "  $0 replay"
        echo "  $0 live"
        exit 1
        ;;
esac

sed -i \
    "s/^[[:space:]]*imu_qos_reliable:.*/    imu_qos_reliable: ${VALUE}/" \
    "${CFG}"

sed -i \
    "s/^[[:space:]]*lidar_qos_reliable:.*/    lidar_qos_reliable: ${VALUE}/" \
    "${CFG}"

echo
echo "===== FR-SLAM QoS MODE ====="

if [ "${VALUE}" = "true" ]; then
    echo "Mode  : OFFLINE REPLAY / BENCHMARK"
    echo "IMU   : RELIABLE"
    echo "LiDAR : RELIABLE"
else
    echo "Mode  : LIVE SENSOR"
    echo "IMU   : BEST_EFFORT"
    echo "LiDAR : BEST_EFFORT"
fi

echo
grep -nE \
'imu_qos_reliable|lidar_qos_reliable' \
"${CFG}"
