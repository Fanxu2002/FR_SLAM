#!/usr/bin/env bash
set -euo pipefail

# FR-SLAM header organization migration.
# Run from: ~/ros2_ws/src/fr_slam

if [[ ! -f "package.xml" || ! -d "include/fr_slam" || ! -d "src" ]]; then
  echo "[ERROR] Please run this script from the fr_slam package root, e.g.:"
  echo "        cd ~/ros2_ws/src/fr_slam"
  exit 1
fi

mkdir -p \
  include/fr_slam/common \
  include/fr_slam/sensor \
  include/fr_slam/imu \
  include/fr_slam/lidar \
  include/fr_slam/frontend \
  include/fr_slam/mapping \
  include/fr_slam/loop \
  include/fr_slam/backend \
  include/fr_slam/ros

move_header() {
  local src="$1"
  local dst="$2"

  if [[ -f "$dst" && ! -f "$src" ]]; then
    echo "[SKIP] already moved: $dst"
    return 0
  fi

  if [[ ! -f "$src" ]]; then
    echo "[ERROR] missing header: $src"
    exit 1
  fi

  mkdir -p "$(dirname "$dst")"

  if git rev-parse --is-inside-work-tree >/dev/null 2>&1 && \
     git ls-files --error-unmatch "$src" >/dev/null 2>&1; then
    git mv "$src" "$dst"
  else
    mv "$src" "$dst"
  fi

  echo "[MOVE] $src -> $dst"
}

# -----------------------------------------------------------------------------
# common
# -----------------------------------------------------------------------------
move_header include/fr_slam/fr_point_types.hpp \
            include/fr_slam/common/fr_point_types.hpp
move_header include/fr_slam/fr_lidar_frame.hpp \
            include/fr_slam/common/fr_lidar_frame.hpp

# -----------------------------------------------------------------------------
# sensor adapters
# -----------------------------------------------------------------------------
move_header include/fr_slam/fr_lidar_adapter.hpp \
            include/fr_slam/sensor/fr_lidar_adapter.hpp
move_header include/fr_slam/fr_hesai_adapter.hpp \
            include/fr_slam/sensor/fr_hesai_adapter.hpp
move_header include/fr_slam/fr_mid360s_adapter.hpp \
            include/fr_slam/sensor/fr_mid360s_adapter.hpp
move_header include/fr_slam/fr_imu_adapter.hpp \
            include/fr_slam/sensor/fr_imu_adapter.hpp

# -----------------------------------------------------------------------------
# imu
# -----------------------------------------------------------------------------
move_header include/fr_slam/fr_imu_types.hpp \
            include/fr_slam/imu/fr_imu_types.hpp
move_header include/fr_slam/fr_imu_buffer.hpp \
            include/fr_slam/imu/fr_imu_buffer.hpp
move_header include/fr_slam/fr_imu_initializer.hpp \
            include/fr_slam/imu/fr_imu_initializer.hpp
move_header include/fr_slam/fr_imu_integrator.hpp \
            include/fr_slam/imu/fr_imu_integrator.hpp

# -----------------------------------------------------------------------------
# lidar
# -----------------------------------------------------------------------------
move_header include/fr_slam/fr_lidar_deskew.hpp \
            include/fr_slam/lidar/fr_lidar_deskew.hpp
move_header include/fr_slam/fr_lidar_preprocessor.hpp \
            include/fr_slam/lidar/fr_lidar_preprocessor.hpp
move_header include/fr_slam/fr_PreprocessorConfig.hpp \
            include/fr_slam/lidar/fr_preprocessor_config.hpp
move_header include/fr_slam/fr_lidar_registration.hpp \
            include/fr_slam/lidar/fr_lidar_registration.hpp
move_header include/fr_slam/fr_lidar_registration_config.hpp \
            include/fr_slam/lidar/fr_lidar_registration_config.hpp

# -----------------------------------------------------------------------------
# frontend
# -----------------------------------------------------------------------------
move_header include/fr_slam/fr_registration_scan2localmap.hpp \
            include/fr_slam/frontend/fr_lidar_frontend.hpp
move_header include/fr_slam/fr_ground_segmenter.hpp \
            include/fr_slam/frontend/fr_ground_segmenter.hpp
move_header include/fr_slam/fr_ground_input_bridge.hpp \
            include/fr_slam/frontend/fr_ground_input_bridge.hpp

# -----------------------------------------------------------------------------
# mapping
# -----------------------------------------------------------------------------
move_header include/fr_slam/fr_keyframe.hpp \
            include/fr_slam/mapping/fr_keyframe.hpp
move_header include/fr_slam/fr_keyframe_detector.hpp \
            include/fr_slam/mapping/fr_keyframe_detector.hpp
move_header include/fr_slam/fr_keyframe_manager.hpp \
            include/fr_slam/mapping/fr_keyframe_manager.hpp
move_header include/fr_slam/fr_local_map.hpp \
            include/fr_slam/mapping/fr_local_map.hpp
move_header include/fr_slam/fr_submap.hpp \
            include/fr_slam/mapping/fr_submap.hpp
move_header include/fr_slam/fr_submap_manager.hpp \
            include/fr_slam/mapping/fr_submap_manager.hpp
move_header include/fr_slam/fr_incremental_global_map.hpp \
            include/fr_slam/mapping/fr_incremental_global_map.hpp

# -----------------------------------------------------------------------------
# loop closure
# -----------------------------------------------------------------------------
move_header include/fr_slam/fr_scan_context.hpp \
            include/fr_slam/loop/fr_scan_context.hpp
move_header include/fr_slam/fr_loop_detector.hpp \
            include/fr_slam/loop/fr_loop_detector.hpp
move_header include/fr_slam/fr_loop_verifier.hpp \
            include/fr_slam/loop/fr_loop_verifier.hpp
move_header include/fr_slam/fr_loop_consistency.hpp \
            include/fr_slam/loop/fr_loop_consistency.hpp

# -----------------------------------------------------------------------------
# backend
# -----------------------------------------------------------------------------
move_header include/fr_slam/fr_pose_graph.hpp \
            include/fr_slam/backend/fr_pose_graph.hpp
move_header include/fr_slam/fr_pose_graph_optimizer.hpp \
            include/fr_slam/backend/fr_pose_graph_optimizer.hpp

# -----------------------------------------------------------------------------
# ros
# -----------------------------------------------------------------------------
move_header include/fr_slam/fr_state_publisher.hpp \
            include/fr_slam/ros/fr_state_publisher.hpp

replace_include() {
  local old="$1"
  local new="$2"
  local changed=0

  while IFS= read -r -d '' file; do
    if grep -qF "$old" "$file"; then
      sed -i "s|${old}|${new}|g" "$file"
      changed=1
    fi
  done < <(find app src include test -type f \
             \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.cc' \) \
             -print0 2>/dev/null)

  if [[ "$changed" -eq 1 ]]; then
    echo "[INCLUDE] $old -> $new"
  fi
}

# -----------------------------------------------------------------------------
# Update project-local include paths everywhere.
# -----------------------------------------------------------------------------
replace_include 'fr_slam/fr_point_types.hpp' \
                'fr_slam/common/fr_point_types.hpp'
replace_include 'fr_slam/fr_lidar_frame.hpp' \
                'fr_slam/common/fr_lidar_frame.hpp'

replace_include 'fr_slam/fr_lidar_adapter.hpp' \
                'fr_slam/sensor/fr_lidar_adapter.hpp'
replace_include 'fr_slam/fr_hesai_adapter.hpp' \
                'fr_slam/sensor/fr_hesai_adapter.hpp'
replace_include 'fr_slam/fr_mid360s_adapter.hpp' \
                'fr_slam/sensor/fr_mid360s_adapter.hpp'
replace_include 'fr_slam/fr_imu_adapter.hpp' \
                'fr_slam/sensor/fr_imu_adapter.hpp'

replace_include 'fr_slam/fr_imu_types.hpp' \
                'fr_slam/imu/fr_imu_types.hpp'
replace_include 'fr_slam/fr_imu_buffer.hpp' \
                'fr_slam/imu/fr_imu_buffer.hpp'
replace_include 'fr_slam/fr_imu_initializer.hpp' \
                'fr_slam/imu/fr_imu_initializer.hpp'
replace_include 'fr_slam/fr_imu_integrator.hpp' \
                'fr_slam/imu/fr_imu_integrator.hpp'

replace_include 'fr_slam/fr_lidar_deskew.hpp' \
                'fr_slam/lidar/fr_lidar_deskew.hpp'
replace_include 'fr_slam/fr_lidar_preprocessor.hpp' \
                'fr_slam/lidar/fr_lidar_preprocessor.hpp'
replace_include 'fr_slam/fr_PreprocessorConfig.hpp' \
                'fr_slam/lidar/fr_preprocessor_config.hpp'
replace_include 'fr_slam/fr_lidar_registration.hpp' \
                'fr_slam/lidar/fr_lidar_registration.hpp'
replace_include 'fr_slam/fr_lidar_registration_config.hpp' \
                'fr_slam/lidar/fr_lidar_registration_config.hpp'

replace_include 'fr_slam/fr_registration_scan2localmap.hpp' \
                'fr_slam/frontend/fr_lidar_frontend.hpp'
replace_include 'fr_slam/fr_ground_segmenter.hpp' \
                'fr_slam/frontend/fr_ground_segmenter.hpp'
replace_include 'fr_slam/fr_ground_input_bridge.hpp' \
                'fr_slam/frontend/fr_ground_input_bridge.hpp'

replace_include 'fr_slam/fr_keyframe.hpp' \
                'fr_slam/mapping/fr_keyframe.hpp'
replace_include 'fr_slam/fr_keyframe_detector.hpp' \
                'fr_slam/mapping/fr_keyframe_detector.hpp'
replace_include 'fr_slam/fr_keyframe_manager.hpp' \
                'fr_slam/mapping/fr_keyframe_manager.hpp'
replace_include 'fr_slam/fr_local_map.hpp' \
                'fr_slam/mapping/fr_local_map.hpp'
replace_include 'fr_slam/fr_submap.hpp' \
                'fr_slam/mapping/fr_submap.hpp'
replace_include 'fr_slam/fr_submap_manager.hpp' \
                'fr_slam/mapping/fr_submap_manager.hpp'
replace_include 'fr_slam/fr_incremental_global_map.hpp' \
                'fr_slam/mapping/fr_incremental_global_map.hpp'

replace_include 'fr_slam/fr_scan_context.hpp' \
                'fr_slam/loop/fr_scan_context.hpp'
replace_include 'fr_slam/fr_loop_detector.hpp' \
                'fr_slam/loop/fr_loop_detector.hpp'
replace_include 'fr_slam/fr_loop_verifier.hpp' \
                'fr_slam/loop/fr_loop_verifier.hpp'
replace_include 'fr_slam/fr_loop_consistency.hpp' \
                'fr_slam/loop/fr_loop_consistency.hpp'

replace_include 'fr_slam/fr_pose_graph.hpp' \
                'fr_slam/backend/fr_pose_graph.hpp'
replace_include 'fr_slam/fr_pose_graph_optimizer.hpp' \
                'fr_slam/backend/fr_pose_graph_optimizer.hpp'

replace_include 'fr_slam/fr_state_publisher.hpp' \
                'fr_slam/ros/fr_state_publisher.hpp'

# -----------------------------------------------------------------------------
# Verify all quoted project-local includes resolve under include/.
# -----------------------------------------------------------------------------
echo
echo "[CHECK] Validating #include \"fr_slam/...\" paths..."
missing=0
while IFS= read -r inc; do
  [[ -z "$inc" ]] && continue
  if [[ ! -f "include/$inc" ]]; then
    echo "[MISSING] include/$inc"
    missing=1
  fi
done < <(
  grep -RhoE '#include[[:space:]]+"fr_slam/[^\"]+"' app src include test 2>/dev/null \
    | sed -E 's/^#include[[:space:]]+"([^"]+)"$/\1/' \
    | sort -u
)

if [[ "$missing" -ne 0 ]]; then
  echo "[ERROR] Some project-local includes do not resolve. Fix the entries above before building."
  exit 2
fi

echo "[OK] All project-local fr_slam includes resolve."

echo
echo "[TREE] include/fr_slam"
if command -v tree >/dev/null 2>&1; then
  tree include/fr_slam
else
  find include/fr_slam -type f | sort
fi

echo
echo "[GIT] Current status"
git status --short 2>/dev/null || true

echo
echo "Done. No algorithm code or class names were changed."
echo "Next build command:"
echo "  cd ~/ros2_ws && rm -rf build/fr_slam install/fr_slam && colcon build --packages-select fr_slam --cmake-args -DCMAKE_BUILD_TYPE=Release"
