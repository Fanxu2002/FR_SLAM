from launch import LaunchDescription

from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    package_share_directory = get_package_share_directory(
        'fr_slam'
    )

    # ============================================================
    # RViz config
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

    os.makedirs(
        log_directory,
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
                'max_lidar_queue_size': 3
            }
        ],

        emulate_tty=True,

        additional_env={
            'ROS_LOG_DIR': log_directory
        }
    )

    # ============================================================
    # RViz2
    # ============================================================
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
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