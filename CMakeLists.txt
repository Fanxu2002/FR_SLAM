cmake_minimum_required(VERSION 3.8)
project(fr_slam)

# ------------------------------------------------------------
# Optional CUDA acceleration for LoopVerifier.
#
# ON:
#   use CUDA when a CUDA compiler is available.
#
# OFF or CUDA compiler unavailable:
#   build the CPU stub.  The normal CPU verifier remains usable.
# ------------------------------------------------------------
option(
    FR_SLAM_ENABLE_CUDA_LOOP_VERIFIER
    "Build CUDA accelerated LoopVerifier"
    ON
)

set(
    FR_SLAM_LOOP_VERIFIER_ACCEL_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/loop/loop_verifier_cuda_stub.cpp
)

set(FR_SLAM_CUDA_LOOP_VERIFIER_AVAILABLE OFF)

if(FR_SLAM_ENABLE_CUDA_LOOP_VERIFIER)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)

        set(
            FR_SLAM_LOOP_VERIFIER_ACCEL_SOURCE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/loop/loop_verifier_cuda.cu
        )

        set(FR_SLAM_CUDA_LOOP_VERIFIER_AVAILABLE ON)

        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(
                CMAKE_CUDA_ARCHITECTURES
                86
                CACHE STRING
                "CUDA architectures for FR-SLAM"
            )
        endif()

        message(
            STATUS
            "FR-SLAM CUDA LoopVerifier: ENABLED "
            "(${CMAKE_CUDA_COMPILER})"
        )
    else()
        message(
            WARNING
            "FR-SLAM CUDA LoopVerifier requested but CUDA compiler "
            "was not found; building CPU-only verifier"
        )
    endif()
else()
    message(
        STATUS
        "FR-SLAM CUDA LoopVerifier: DISABLED (CPU-only)"
    )
endif()



if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(std_srvs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(TBB REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(fmt REQUIRED)
find_package(visualization_msgs REQUIRED)

# PCL/FLANN on this system uses LZ4. Link it explicitly for the
# scan2localmap executable so the linker does not report
# "DSO missing from command line" for LZ4_decompress_safe_continue.
find_library(LZ4_LIBRARY NAMES lz4 REQUIRED)


# ============================================================================
# Bundled g2o for FR-SLAM
#
# Keep g2o inside this project so that FR-SLAM, Eigen and g2o are compiled
# with the same compiler/ISA settings. This avoids mixing the native
# FR-SLAM code with the generic /usr/local g2o build.
# ============================================================================
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

set(BUILD_WITH_MARCH_NATIVE ON CACHE BOOL "" FORCE)

set(G2O_BUILD_APPS OFF CACHE BOOL "" FORCE)
set(G2O_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(G2O_USE_OPENGL OFF CACHE BOOL "" FORCE)

set(G2O_BUILD_SLAM2D_TYPES OFF CACHE BOOL "" FORCE)
set(G2O_BUILD_SLAM2D_ADDON_TYPES OFF CACHE BOOL "" FORCE)
set(G2O_BUILD_DATA_TYPES OFF CACHE BOOL "" FORCE)
set(G2O_BUILD_SCLAM2D_TYPES OFF CACHE BOOL "" FORCE)

set(G2O_BUILD_SLAM3D_TYPES ON CACHE BOOL "" FORCE)
set(G2O_BUILD_SLAM3D_ADDON_TYPES OFF CACHE BOOL "" FORCE)

set(G2O_BUILD_SBA_TYPES OFF CACHE BOOL "" FORCE)
set(G2O_BUILD_ICP_TYPES OFF CACHE BOOL "" FORCE)
set(G2O_BUILD_SIM3_TYPES OFF CACHE BOOL "" FORCE)

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/g2o
    ${CMAKE_CURRENT_BINARY_DIR}/third_party/g2o
)



# # ============================================
# #                 TEST
# # ============================================
# add_executable(test_mid360s_adapter)
# target_sources(test_mid360s_adapter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/test_mid360s_adapter.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor/mid360s_adapter.cpp)
# target_include_directories(test_mid360s_adapter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# ament_target_dependencies(test_mid360s_adapter rclcpp sensor_msgs pcl_conversions)
# target_link_libraries(test_mid360s_adapter ${PCL_LIBRARIES})
# target_compile_definitions(test_mid360s_adapter PRIVATE ${PCL_NO_PRECOMPILE})

# add_executable(test_hesai_adapter)
# target_sources(test_hesai_adapter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/test_hesai_adapter.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor/hesai_adapter.cpp)
# target_include_directories(test_hesai_adapter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# ament_target_dependencies(test_hesai_adapter rclcpp sensor_msgs pcl_conversions)
# target_link_libraries(test_hesai_adapter ${PCL_LIBRARIES})
# target_compile_definitions(test_hesai_adapter PRIVATE ${PCL_NO_PRECOMPILE})

# add_executable(test_imu_adapter)
# target_sources(test_imu_adapter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/test_imu_adapter.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor/imu_adapter.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/imu/imu_buffer.cpp
#                                             )
# target_include_directories(test_imu_adapter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# ament_target_dependencies(test_imu_adapter rclcpp sensor_msgs pcl_conversions)
# target_link_libraries(test_imu_adapter ${PCL_LIBRARIES})
# target_compile_definitions(test_imu_adapter PRIVATE ${PCL_NO_PRECOMPILE})

# add_executable(test_imu_buffer)
# target_sources(test_imu_buffer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/test_imu_buffer.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor/imu_adapter.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/imu/imu_buffer.cpp
#                                             )
# target_include_directories(test_imu_buffer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# ament_target_dependencies(test_imu_buffer rclcpp sensor_msgs pcl_conversions)
# target_link_libraries(test_imu_buffer ${PCL_LIBRARIES})
# target_compile_definitions(test_imu_buffer PRIVATE ${PCL_NO_PRECOMPILE})

# add_executable(test_imu_initializer)
# target_sources(test_imu_initializer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/test_imu_initializer.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor/imu_adapter.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/imu/imu_buffer.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/imu/imu_initializer.cpp
#                                             )
# target_include_directories(test_imu_initializer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# ament_target_dependencies(test_imu_initializer rclcpp sensor_msgs pcl_conversions)
# target_link_libraries(test_imu_initializer ${PCL_LIBRARIES})
# target_compile_definitions(test_imu_initializer PRIVATE ${PCL_NO_PRECOMPILE})

# add_executable(test_imu_integrator)
# target_sources(test_imu_integrator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/test_imu_integrator.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor/imu_adapter.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/imu/imu_buffer.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/imu/imu_initializer.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/imu/imu_integrator.cpp
#                                             )
# target_include_directories(test_imu_integrator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# ament_target_dependencies(test_imu_integrator rclcpp sensor_msgs pcl_conversions nav_msgs geometry_msgs)
# target_link_libraries(test_imu_integrator ${PCL_LIBRARIES})
# target_compile_definitions(test_imu_integrator PRIVATE ${PCL_NO_PRECOMPILE})

# install(
#       DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
#       DESTINATION include/${PROJECT_NAME}
# )
# install(
#   TARGETS test_mid360s_adapter
#           test_hesai_adapter
#           test_imu_adapter
#           test_imu_buffer
#           test_imu_initializer
#           test_imu_integrator
#   RUNTIME DESTINATION lib/${PROJECT_NAME}/test
# )


# ============================================
#                 APP
# ============================================

# --------------------------------------------------------------------------
# Sensor adapters
# --------------------------------------------------------------------------
set(FR_SLAM_SENSOR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor/imu_adapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor/mid360s_adapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor/hesai_adapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor/velodyne_adapter.cpp
)

# --------------------------------------------------------------------------
# IMU
# --------------------------------------------------------------------------
set(FR_SLAM_IMU_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/imu/imu_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/imu/imu_initializer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/imu/imu_integrator.cpp
)

# --------------------------------------------------------------------------
# LiDAR
# --------------------------------------------------------------------------
set(FR_SLAM_LIDAR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lidar/lidar_deskew.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lidar/lidar_preprocessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lidar/lidar_registration.cpp
)

# --------------------------------------------------------------------------
# LO frontend
# --------------------------------------------------------------------------
set(FR_SLAM_LO_FRONTEND_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend/lo/lo_frontend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend/lo/lo_frontend_ground.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend/constraints/plane/multi_plane_extractor.cpp
    src/frontend/constraints/plane/wall_association.cpp
)

# --------------------------------------------------------------------------
# LO frontend - long-memory 60/30 diagnostic
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# LIO frontend
# --------------------------------------------------------------------------
set(FR_SLAM_LIO_FRONTEND_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend/lio/ieskf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend/lio/lio_measurement.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend/lio/lio_frontend.cpp
)


# --------------------------------------------------------------------------
# LIO frontend - rotation update diagnostic
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# LIO frontend - D5 query-centered plane experiment
# --------------------------------------------------------------------------
set(FR_SLAM_LIO_QUERY_PLANE_FRONTEND_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend/lio/ieskf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend/lio/lio_measurement_query_plane.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend/lio/lio_frontend.cpp
)


# --------------------------------------------------------------------------
# Ground constraint
# --------------------------------------------------------------------------
set(FR_SLAM_GROUND_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend/constraints/ground/ground_segmenter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend/constraints/ground/ground_input_bridge.cpp
)

# --------------------------------------------------------------------------
# Mapping
# --------------------------------------------------------------------------
set(FR_SLAM_MAPPING_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mapping/local_map.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mapping/keyframe_detector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mapping/keyframe_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mapping/submap_manager.cpp
)


# --------------------------------------------------------------------------
# LIO map-reference-age diagnostic
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Mapping - 60/30 long-span same-count diagnostic
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Loop closure
# --------------------------------------------------------------------------
set(FR_SLAM_LOOP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/loop/loop_detector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/loop/scan_context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/loop/scan_context_shadow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/loop/scan_context_window_shadow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/loop/loop_verifier.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/loop/loop_consistency.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/loop/btc_adapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/loop/loop_closure.cpp
)

# --------------------------------------------------------------------------
# Loop closure - SC exact Verify reuse SHADOW diagnostic
#
# Formal FR_SLAM_LOOP_SOURCES remains unchanged.
# --------------------------------------------------------------------------
set(FR_SLAM_LOOP_SC_REUSE_DIAG_SOURCES
    ${FR_SLAM_LOOP_SOURCES}
)

list(
    REMOVE_ITEM
    FR_SLAM_LOOP_SC_REUSE_DIAG_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/loop/loop_closure.cpp
)

list(
    APPEND
    FR_SLAM_LOOP_SC_REUSE_DIAG_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/loop/loop_closure_sc_reuse_diag.cpp
)



# --------------------------------------------------------------------------
# Loop closure - SC exact Verify reuse ACTIVE V2 diagnostic
# --------------------------------------------------------------------------
set(FR_SLAM_LOOP_SC_REUSE_V2_DIAG_SOURCES
    ${FR_SLAM_LOOP_SOURCES}
)

list(
    REMOVE_ITEM
    FR_SLAM_LOOP_SC_REUSE_V2_DIAG_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/loop/loop_closure.cpp
)

list(
    APPEND
    FR_SLAM_LOOP_SC_REUSE_V2_DIAG_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/loop/loop_closure_sc_reuse_v2_diag.cpp
)

set(FR_SLAM_LOOP_SC_REUSE_PARALLEL_V3_DIAG_SOURCES
    ${FR_SLAM_LOOP_SOURCES}
)

list(
    REMOVE_ITEM
    FR_SLAM_LOOP_SC_REUSE_PARALLEL_V3_DIAG_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/loop/loop_closure.cpp
)

list(
    APPEND
    FR_SLAM_LOOP_SC_REUSE_PARALLEL_V3_DIAG_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/loop/loop_closure_sc_reuse_parallel_v3_diag.cpp
)


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------
set(FR_SLAM_BACKEND_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/pose_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/pose_graph_optimizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/backend_worker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/pose_graph_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/global_map_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/post_pgo_refinement.cpp
)


# --------------------------------------------------------------------------
# Third party
# --------------------------------------------------------------------------
set(FR_SLAM_THIRD_PARTY_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/btc_core/src/btc_core.cpp
)


add_executable(lo)

target_sources(
    lo
    PRIVATE

    ${CMAKE_CURRENT_SOURCE_DIR}/app/lo.cpp

    ${FR_SLAM_SENSOR_SOURCES}
    ${FR_SLAM_IMU_SOURCES}
    ${FR_SLAM_LIDAR_SOURCES}

    ${FR_SLAM_LO_FRONTEND_SOURCES}
    ${FR_SLAM_GROUND_SOURCES}

    ${FR_SLAM_MAPPING_SOURCES}
    ${FR_SLAM_LOOP_SOURCES}
    ${FR_SLAM_LOOP_VERIFIER_ACCEL_SOURCE}

    ${FR_SLAM_BACKEND_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/pgo_ground_confidence_bridge.cpp

    ${FR_SLAM_THIRD_PARTY_SOURCES}
)

target_include_directories(
    lo
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/btc_core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/g2o
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/include
)

ament_target_dependencies(
    lo
    rclcpp
    sensor_msgs
    std_srvs
    pcl_conversions
    nav_msgs
    geometry_msgs
    visualization_msgs
    tf2_ros
)

target_link_libraries(
    lo

    ${PCL_LIBRARIES}

    types_slam3d
    solver_eigen
    core
    stuff

    ${OpenCV_LIBS}
    Ceres::ceres
    TBB::tbb

    Threads::Threads
    fmt::fmt
    ${LZ4_LIBRARY}
)

target_compile_definitions(
    lo
    PRIVATE
    PCL_NO_PRECOMPILE
)

target_compile_definitions(
    lo
    PRIVATE
    FR_SLAM_BTC_CONFIG_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/config/btc\"
)


# ============================================

# ============================================
#   LIO
# ============================================

add_executable(lio)

target_sources(
    lio
    PRIVATE

    ${CMAKE_CURRENT_SOURCE_DIR}/app/lio.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor/livox_custom_adapter.cpp

    ${FR_SLAM_SENSOR_SOURCES}
    ${FR_SLAM_IMU_SOURCES}
    ${FR_SLAM_LIDAR_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lidar/lio_lidar_deskew.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lidar/lio_preprocessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend/lio/lio_map_odom_bridge.cpp

    ${FR_SLAM_LO_FRONTEND_SOURCES}
    ${FR_SLAM_LIO_FRONTEND_SOURCES}
    ${FR_SLAM_GROUND_SOURCES}

    ${FR_SLAM_MAPPING_SOURCES}
    ${FR_SLAM_LOOP_SOURCES}
    ${FR_SLAM_LOOP_VERIFIER_ACCEL_SOURCE}
    ${FR_SLAM_BACKEND_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/pgo_ground_confidence_bridge.cpp

    ${FR_SLAM_THIRD_PARTY_SOURCES}
)

set_target_properties(
    lio
    PROPERTIES
    CUDA_ARCHITECTURES 86
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

target_include_directories(
    lio
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/btc_core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/g2o
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/include
)

ament_target_dependencies(
    lio
    rclcpp
    sensor_msgs
    livox_ros_driver2
    std_srvs
    pcl_conversions
    nav_msgs
    geometry_msgs
    visualization_msgs
    tf2_ros
)

target_link_libraries(
    lio

    ${PCL_LIBRARIES}

    types_slam3d
    solver_eigen
    core
    stuff

    ${OpenCV_LIBS}
    Ceres::ceres
    TBB::tbb

    Threads::Threads
    fmt::fmt
    ${LZ4_LIBRARY}
)

target_compile_definitions(
    lio
    PRIVATE
    PCL_NO_PRECOMPILE
)

target_compile_definitions(
    lio
    PRIVATE
    FR_SLAM_BTC_CONFIG_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/config/btc\"
)






# ============================================
#   LIO GRAVITY PRIOR 0.5 DEG ABLATION
# ============================================
#
# Single-variable ablation:
# initial gravity uncertainty
#   2.0 deg -> 0.5 deg
#
# All normal LIO frontend / IESKF / mapping /
# backend source files remain identical.
# ============================================


# ============================================
#   LIO MAP REFERENCE AGE DIAGNOSTIC
# ============================================
#
# Baseline 30/15 architecture.
#
# Changes only:
#   lio_measurement_ref_age_diag.cpp
#   submap_manager_ref_age_diag.cpp
#   map_ref_age_registry.cpp
#
# ============================================

# ============================================================
# Historical diagnostic / ablation executables
#
# Legacy experiment target definitions were archived under:
#   experiments/legacy_cmake/
#
# Mature functionality has been promoted into the production
# lio target. Runtime diagnostics remain available in the
# production implementation.
# ============================================================

add_executable(lidar_imu_rotation_calibration)

target_sources(
    lidar_imu_rotation_calibration
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/lidar_imu_rotation_calibration.cpp
)

target_link_libraries(
    lidar_imu_rotation_calibration
    Eigen3::Eigen
)


# ============================================
#        GROUND V4.0 VALIDATION NODE
# ============================================
#
# This node is deliberately separate from the realtime SLAM worker.
# It subscribes to /livox/lidar and publishes diagnostic-only ground results:
#
#     /ground_analysis_cloud
#     /ground_cloud
#     /nonground_cloud
#     /ground_local_normal
#     /support_ground_cloud
#     /support_ground_normal
#     /support_constraint_cloud
#     /support_constraint_normal
#
# Ground V4.0 can therefore be validated in RViz before it is allowed to change
# Scan-to-LocalMap, T_WL, Hessian or PoseGraph behavior.
# ============================================

add_executable(ground_segmentation_debug)

target_sources(
    ground_segmentation_debug
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/ground_segmentation_debug.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor/mid360s_adapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend/constraints/ground/ground_segmenter.cpp
)

target_include_directories(
    ground_segmentation_debug
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

ament_target_dependencies(
    ground_segmentation_debug
    rclcpp
    sensor_msgs
    pcl_conversions
    geometry_msgs
    visualization_msgs
)

target_link_libraries(
    ground_segmentation_debug
    ${PCL_LIBRARIES}
)

target_compile_definitions(
    ground_segmentation_debug
    PRIVATE
    PCL_NO_PRECOMPILE
)


install(
      DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
      DESTINATION include/${PROJECT_NAME}
)
install(
      DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/config
      DESTINATION share/${PROJECT_NAME}
)
install(
      DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/launch
      DESTINATION share/${PROJECT_NAME}
)

  # ---- Offline LIO vs GT evaluator ----
add_executable(lio_gt_evaluator
  tools/evaluation/lio_gt_evaluator.cpp
)

target_compile_features(lio_gt_evaluator PRIVATE cxx_std_17)

target_link_libraries(lio_gt_evaluator PRIVATE
  Eigen3::Eigen
)

install(TARGETS lio_gt_evaluator
  DESTINATION lib/${PROJECT_NAME}
)

# ---- Offline trajectory frame converter ----
add_executable(trajectory_frame_converter
  tools/evaluation/trajectory_frame_converter.cpp
)

target_compile_features(
  trajectory_frame_converter
  PRIVATE cxx_std_17
)

target_link_libraries(
  trajectory_frame_converter
  PRIVATE Eigen3::Eigen
)

install(
  TARGETS trajectory_frame_converter
  DESTINATION lib/${PROJECT_NAME}
)

# ---- Generic Odometry CSV recorder ----
add_executable(odometry_csv_recorder
  tools/evaluation/odometry_csv_recorder.cpp
)

target_compile_features(odometry_csv_recorder PRIVATE cxx_std_17)

ament_target_dependencies(odometry_csv_recorder
  rclcpp
  nav_msgs
)

install(TARGETS odometry_csv_recorder
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()



# ============================================================
# FR-SLAM PRODUCTION RUNTIME INSTALL
# ============================================================

install(
    TARGETS
        lo
        lio
        lidar_imu_rotation_calibration
        ground_segmentation_debug
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)

ament_package()
