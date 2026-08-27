from launch import LaunchDescription

from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    package_share_directory = get_package_share_directory(
        'fr_slam'
    )

    rviz_config_path = os.path.join(
        package_share_directory,
        'config',
        'FR_SLAM.rviz'
    )

    slam_node = Node(
        package='fr_slam',
        executable='lidar_registration_scan2localmap',
        name='fr_slam',
        output='screen'
    )

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

    return LaunchDescription([
        slam_node,
        rviz_node
    ])
