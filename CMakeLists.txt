cmake_minimum_required(VERSION 3.8)
project(fr_slam)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(fmt REQUIRED)
find_package(visualization_msgs REQUIRED)

# PCL/FLANN on this system uses LZ4.  Link it explicitly for the
# scan2localmap executable so the linker does not report
# "DSO missing from command line" for LZ4_decompress_safe_continue.
find_library(LZ4_LIBRARY NAMES lz4 REQUIRED)


# # ============================================
# #                 TEST
# # ============================================
# add_executable(test_mid360s_adapter)
# target_sources(test_mid360s_adapter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/test_mid360s_adapter.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor/fr_mid360s_adapter.cpp)
# target_include_directories(test_mid360s_adapter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# ament_target_dependencies(test_mid360s_adapter rclcpp sensor_msgs pcl_conversions)
# target_link_libraries(test_mid360s_adapter ${PCL_LIBRARIES})
# target_compile_definitions(test_mid360s_adapter PRIVATE ${PCL_NO_PRECOMPILE})

# add_executable(test_hesai_adapter)
# target_sources(test_hesai_adapter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/test_hesai_adapter.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor/fr_hesai_adapter.cpp)
# target_include_directories(test_hesai_adapter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# ament_target_dependencies(test_hesai_adapter rclcpp sensor_msgs pcl_conversions)
# target_link_libraries(test_hesai_adapter ${PCL_LIBRARIES})
# target_compile_definitions(test_hesai_adapter PRIVATE ${PCL_NO_PRECOMPILE})

# add_executable(test_imu_adapter)
# target_sources(test_imu_adapter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/test_imu_adapter.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor/fr_imu_adapter.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/imu/fr_imu_buffer.cpp
#                                             )
# target_include_directories(test_imu_adapter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# ament_target_dependencies(test_imu_adapter rclcpp sensor_msgs pcl_conversions)
# target_link_libraries(test_imu_adapter ${PCL_LIBRARIES})
# target_compile_definitions(test_imu_adapter PRIVATE ${PCL_NO_PRECOMPILE})

# add_executable(test_imu_buffer)
# target_sources(test_imu_buffer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/test_imu_buffer.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor/fr_imu_adapter.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/imu/fr_imu_buffer.cpp
#                                             )
# target_include_directories(test_imu_buffer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# ament_target_dependencies(test_imu_buffer rclcpp sensor_msgs pcl_conversions)
# target_link_libraries(test_imu_buffer ${PCL_LIBRARIES})
# target_compile_definitions(test_imu_buffer PRIVATE ${PCL_NO_PRECOMPILE})

# add_executable(test_imu_initializer)
# target_sources(test_imu_initializer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/test_imu_initializer.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor/fr_imu_adapter.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/imu/fr_imu_buffer.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/imu/fr_imu_initializer.cpp
#                                             )
# target_include_directories(test_imu_initializer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# ament_target_dependencies(test_imu_initializer rclcpp sensor_msgs pcl_conversions)
# target_link_libraries(test_imu_initializer ${PCL_LIBRARIES})
# target_compile_definitions(test_imu_initializer PRIVATE ${PCL_NO_PRECOMPILE})

# add_executable(test_imu_integrator)
# target_sources(test_imu_integrator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/test_imu_integrator.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor/fr_imu_adapter.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/imu/fr_imu_buffer.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/imu/fr_imu_initializer.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/imu/fr_imu_integrator.cpp
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

add_executable(fr_slam_node)
target_sources(fr_slam_node PRIVATE
                                            ${CMAKE_CURRENT_SOURCE_DIR}/app/fr_slam_node.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor/fr_imu_adapter.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/imu/fr_imu_buffer.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/imu/fr_imu_initializer.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/imu/fr_imu_integrator.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/lidar/fr_lidar_deskew.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor/fr_imu_adapter.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor/fr_mid360s_adapter.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor/fr_hesai_adapter.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor/fr_hesai_adapter.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/lidar/fr_lidar_preprocessor.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/lidar/fr_lidar_registration.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/mapping/fr_local_map.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend/fr_lidar_frontend.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/mapping/fr_keyframe_detector.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/mapping/fr_keyframe_manager.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/mapping/fr_submap_manager.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/fr_pose_graph.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/loop/fr_loop_detector.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/loop/fr_scan_context.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/loop/fr_loop_verifier.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/loop/fr_loop_consistency.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/fr_pose_graph_optimizer.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend/fr_ground_segmenter.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend/fr_ground_input_bridge.cpp
                                            )

target_include_directories(fr_slam_node PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(
    fr_slam_node
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /usr/local/include
)

ament_target_dependencies(
    fr_slam_node
    rclcpp
    sensor_msgs
    std_srvs
    pcl_conversions
    nav_msgs
    geometry_msgs
    visualization_msgs
    tf2_ros
)

target_link_libraries(fr_slam_node ${PCL_LIBRARIES} ${G2O_LIBRARIES})
target_link_libraries(
    fr_slam_node

    ${PCL_LIBRARIES}

    /usr/local/lib/libg2o_types_slam3d.so
    /usr/local/lib/libg2o_solver_eigen.so
    /usr/local/lib/libg2o_core.so
    /usr/local/lib/libg2o_stuff.so

    Threads::Threads
    fmt::fmt
    ${LZ4_LIBRARY}
)

target_compile_definitions(fr_slam_node PRIVATE PCL_NO_PRECOMPILE)


# ============================================
#   OFFLINE LIDAR-IMU ROTATION CALIBRATION
# ============================================
#
# This is a normal C++ executable installed in the ROS 2 package so it can be
# launched with `ros2 run`.  It does not create an rclcpp::Node and does not
# subscribe to topics.  It reads the synchronized CSV exported by fr_slam_node.
# ============================================

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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor/fr_mid360s_adapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend/fr_ground_segmenter.cpp
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
install(
  TARGETS
          fr_slam_node
          lidar_imu_rotation_calibration
          ground_segmentation_debug
  RUNTIME DESTINATION lib/${PROJECT_NAME})


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
