from launch import LaunchDescription

from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    # ============================================================
    # FR-SLAM package share directory
    #
    # After colcon build:
    #
    #   ~/ros2_ws/install/fr_slam/share/fr_slam
    #
    # ============================================================
    package_share_directory = get_package_share_directory(
        'fr_slam'
    )

    # ============================================================
    # RViz config
    #
    # Source:
    #
    #   ~/ros2_ws/src/fr_slam/config/FR_SLAM.rviz
    #
    # Installed:
    #
    #   ~/ros2_ws/install/fr_slam/share/fr_slam/config/FR_SLAM.rviz
    #
    # ============================================================
    rviz_config_path = os.path.join(
        package_share_directory,
        'config',
        'FR_SLAM.rviz'
    )

    # ============================================================
    # FR-SLAM log directory
    # ============================================================
    log_directory = os.path.expanduser(
        '~/ros2_ws/log/fr_slam'
    )
    # ============================================================
    # FR-SLAM output directory
    # ============================================================
    output_directory = os.path.expanduser(
        '~/ros2_ws/src/fr_slam/output'
    )

    maps_directory = os.path.join(
        output_directory,
        'maps'
    )

    trajectory_directory = os.path.join(
        output_directory,
        'trajectory'
    )

    loop_directory = os.path.join(
        output_directory,
        'loop'
    )

    diagnostics_directory = os.path.join(
        output_directory,
        'diagnostics'
    )

    os.makedirs(
        maps_directory,
        exist_ok=True
    )

    os.makedirs(
        trajectory_directory,
        exist_ok=True
    )

    os.makedirs(
        loop_directory,
        exist_ok=True
    )

    os.makedirs(
        diagnostics_directory,
        exist_ok=True
    )

    os.makedirs(
        log_directory,
        exist_ok=True
    )


    # ============================================================
    # PCD map output directory
    #
    #   ~/ros2_ws/src/fr_slam/Map
    #
    # This path is passed to the C++ node through the ROS parameter
    # "pcd_save_directory".
    # ============================================================
    map_directory = os.path.expanduser(
        '~/ros2_ws/src/fr_slam/Map'
    )

    os.makedirs(
        map_directory,
        exist_ok=True
    )

    # ============================================================
    # FR-SLAM node
    # ============================================================
    slam_node = Node(
        package='fr_slam',
        executable='lidar_registration_scan2localmap',
        name='fr_slam',

        output='both',

        parameters=[
            {
                'preprocessor_enable_sor': True,
                'preprocessor_enable_ror': True,
                'max_lidar_queue_size': 3,
                'pcd_save_directory': map_directory
            }
        ],

        emulate_tty=True,

    additional_env={
        'ROS_LOG_DIR': log_directory,
        'FR_SLAM_OUTPUT_DIR': output_directory
    }
    )

    # ============================================================
    # RViz2
    # ============================================================
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='fr_slam_rviz',

        output='screen',

        arguments=[
            '-d',
            rviz_config_path
        ]
    )

    # ============================================================
    # Launch
    # ============================================================
    return LaunchDescription([
        slam_node,
        rviz_node
    ])
