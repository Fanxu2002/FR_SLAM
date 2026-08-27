cmake_minimum_required(VERSION 3.8)
project(fr_slam)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(fmt REQUIRED)
find_package(visualization_msgs REQUIRED)


# ============================================
#                 TEST
# ============================================
add_executable(test_mid360s_adapter)
target_sources(test_mid360s_adapter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/test_mid360s_adapter.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/mid360s_adapter.cpp)
target_include_directories(test_mid360s_adapter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
ament_target_dependencies(test_mid360s_adapter rclcpp sensor_msgs pcl_conversions)
target_link_libraries(test_mid360s_adapter ${PCL_LIBRARIES})
target_compile_definitions(test_mid360s_adapter PRIVATE ${PCL_NO_PRECOMPILE})

add_executable(test_hesai_adapter)
target_sources(test_hesai_adapter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/test_hesai_adapter.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/hesai_adapter.cpp)
target_include_directories(test_hesai_adapter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
ament_target_dependencies(test_hesai_adapter rclcpp sensor_msgs pcl_conversions)
target_link_libraries(test_hesai_adapter ${PCL_LIBRARIES})
target_compile_definitions(test_hesai_adapter PRIVATE ${PCL_NO_PRECOMPILE})

add_executable(test_imu_adapter)
target_sources(test_imu_adapter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/test_imu_adapter.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/imu_adapter.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_buffer.cpp
                                            )
target_include_directories(test_imu_adapter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
ament_target_dependencies(test_imu_adapter rclcpp sensor_msgs pcl_conversions)
target_link_libraries(test_imu_adapter ${PCL_LIBRARIES})
target_compile_definitions(test_imu_adapter PRIVATE ${PCL_NO_PRECOMPILE})

add_executable(test_imu_buffer)
target_sources(test_imu_buffer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/test_imu_buffer.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/imu_adapter.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_buffer.cpp
                                            )
target_include_directories(test_imu_buffer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
ament_target_dependencies(test_imu_buffer rclcpp sensor_msgs pcl_conversions)
target_link_libraries(test_imu_buffer ${PCL_LIBRARIES})
target_compile_definitions(test_imu_buffer PRIVATE ${PCL_NO_PRECOMPILE})

add_executable(test_imu_initializer)
target_sources(test_imu_initializer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/test_imu_initializer.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/imu_adapter.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_buffer.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_initializer.cpp

                                            )
target_include_directories(test_imu_initializer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
ament_target_dependencies(test_imu_initializer rclcpp sensor_msgs pcl_conversions)
target_link_libraries(test_imu_initializer ${PCL_LIBRARIES})
target_compile_definitions(test_imu_initializer PRIVATE ${PCL_NO_PRECOMPILE})

add_executable(test_imu_integrator)
target_sources(test_imu_integrator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/test_imu_integrator.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/imu_adapter.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_buffer.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_initializer.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_integrator.cpp


                                            )
target_include_directories(test_imu_integrator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
ament_target_dependencies(test_imu_integrator rclcpp sensor_msgs pcl_conversions nav_msgs geometry_msgs)
target_link_libraries(test_imu_integrator ${PCL_LIBRARIES})
target_compile_definitions(test_imu_integrator PRIVATE ${PCL_NO_PRECOMPILE})

# add_executable(test_lidar_deskewer)
# target_sources(test_lidar_deskewer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/test_lidar_deskewer.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/imu_adapter.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_buffer.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_initializer.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_integrator.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_lidar_deskewer.cpp



#                                             )
# target_include_directories(test_lidar_deskewer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# ament_target_dependencies(test_lidar_deskewer rclcpp sensor_msgs pcl_conversions)
# target_link_libraries(test_lidar_deskewer ${PCL_LIBRARIES})
# target_compile_definitions(test_lidar_deskewer PRIVATE ${PCL_NO_PRECOMPILE})


# add_executable(test_lidar_registration)
# target_sources(test_lidar_registration PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test/test_lidar_registration.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/imu_adapter.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_buffer.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_initializer.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_integrator.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_lidar_deskewer.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_lidar_registration.cpp




#                                             )
# target_include_directories(test_lidar_registration PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# ament_target_dependencies(test_lidar_registration rclcpp sensor_msgs pcl_conversions)
# target_link_libraries(test_lidar_registration ${PCL_LIBRARIES})
# target_compile_definitions(test_lidar_registration PRIVATE ${PCL_NO_PRECOMPILE})

install(
      DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include 
      DESTINATION include/${PROJECT_NAME}
)
install(
  TARGETS test_mid360s_adapter 
          test_hesai_adapter 
          test_imu_adapter
          test_imu_buffer
          test_imu_initializer
          test_imu_integrator
          # test_lidar_deskewer
          # test_lidar_registration
  RUNTIME DESTINATION lib/${PROJECT_NAME}/test
)


# ============================================
#                 APP
# ============================================

# add_executable(lidar_deskewer)
# target_sources(lidar_deskewer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/app/lidar_deskewer.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/imu_adapter.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_buffer.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_initializer.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_integrator.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_lidar_deskewer.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/mid360s_adapter.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_lidar_registration.cpp
#                                             )
# target_include_directories(lidar_deskewer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# ament_target_dependencies(lidar_deskewer rclcpp sensor_msgs pcl_conversions)
# target_link_libraries(lidar_deskewer ${PCL_LIBRARIES})
# target_compile_definitions(lidar_deskewer PRIVATE ${PCL_NO_PRECOMPILE})

add_executable(lidar_registration_scan2scan)
target_sources(lidar_registration_scan2scan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/app/lidar_registration_scan2scan.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/imu_adapter.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_buffer.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_initializer.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_integrator.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_lidar_deskewer.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/imu_adapter.cpp                                           
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/mid360s_adapter.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_lidar_preprocessor.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_lidar_registration.cpp
                                            )
target_include_directories(lidar_registration_scan2scan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
ament_target_dependencies(lidar_registration_scan2scan rclcpp sensor_msgs pcl_conversions nav_msgs geometry_msgs)
target_link_libraries(lidar_registration_scan2scan ${PCL_LIBRARIES})
target_link_libraries(
lidar_registration_scan2scan    Threads::Threads
)
target_compile_definitions(lidar_registration_scan2scan PRIVATE PCL_NO_PRECOMPILE)



add_executable(lidar_registration_scan2localmap)
target_sources(lidar_registration_scan2localmap PRIVATE 
                                            ${CMAKE_CURRENT_SOURCE_DIR}/app/lidar_registration_scan2localmap.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/imu_adapter.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_buffer.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_initializer.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_integrator.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_lidar_deskewer.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/imu_adapter.cpp                                           
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/mid360s_adapter.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_lidar_preprocessor.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_lidar_registration.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_local_map.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_registration_scan2localmap.cpp    
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_keyframe_detector.cpp   
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_keyframe_manager.cpp   
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_submap_manager.cpp  
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_pose_graph.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_loop_detector.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_scan_context.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_loop_verifier.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_loop_consistency_checker.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_pose_graph_optimizer.cpp

                                            )
target_include_directories(lidar_registration_scan2localmap PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(
    lidar_registration_scan2localmap
    PRIVATE

    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /usr/local/include
)
ament_target_dependencies(lidar_registration_scan2localmap rclcpp sensor_msgs pcl_conversions nav_msgs geometry_msgs visualization_msgs )
target_link_libraries(lidar_registration_scan2localmap ${PCL_LIBRARIES} ${G2O_LIBRARIES})
target_link_libraries(
    lidar_registration_scan2localmap

    ${PCL_LIBRARIES}

    /usr/local/lib/libg2o_types_slam3d.so
    /usr/local/lib/libg2o_solver_eigen.so
    /usr/local/lib/libg2o_core.so
    /usr/local/lib/libg2o_stuff.so

    Threads::Threads
    fmt::fmt
)

target_compile_definitions(lidar_registration_scan2localmap PRIVATE PCL_NO_PRECOMPILE)

# add_executable(path)
# target_sources(path PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/app/path.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/imu_adapter.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_local_map.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_buffer.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_initializer.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_imu_integrator.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_lidar_deskewer.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/mid360s_adapter.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_lidar_preprocessor.cpp
#                                             ${CMAKE_CURRENT_SOURCE_DIR}/src/fr_lidar_registration.cpp
#                                             )
# target_include_directories(path PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# ament_target_dependencies(path rclcpp sensor_msgs pcl_conversions nav_msgs geometry_msgs  tf2_ros)
# target_link_libraries(path ${PCL_LIBRARIES})
# target_link_libraries(
#     path
#     Threads::Threads
# )
# target_compile_definitions(path PRIVATE ${PCL_NO_PRECOMPILE})

install(
      DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include 
      DESTINATION include/${PROJECT_NAME}
)
install(
  TARGETS 
          # lidar_deskewer
          lidar_registration_scan2scan
          lidar_registration_scan2localmap
          # path
  RUNTIME DESTINATION lib/${PROJECT_NAME}/test)







if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
