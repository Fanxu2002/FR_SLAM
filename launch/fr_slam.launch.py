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
    #   <workspace>/install/fr_slam/share/fr_slam
    #
    # ============================================================
    package_share_directory = get_package_share_directory(
        'fr_slam'
    )

    # ============================================================
    # Workspace directory
    #
    # package_share_directory:
    #   <workspace>/install/fr_slam/share/fr_slam
    #
    # workspace_directory:
    #   <workspace>
    #
    # No user-specific absolute path is hard-coded here.
    # ============================================================
    workspace_directory = os.path.abspath(
        os.path.join(
            package_share_directory,
            '..',
            '..',
            '..',
            '..'
        )
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
    # FR-SLAM YAML configuration
    # ============================================================
    slam_config_path = os.path.join(
        package_share_directory,
        'config',
        'fr_slam.yaml'
    )

    # ============================================================
    # FR-SLAM output directory
    #
    #   <workspace>/src/fr_slam/output
    #
    # ============================================================
    output_directory = os.path.join(
        workspace_directory,
        'src',
        'fr_slam',
        'output'
    )

    saves_directory = os.path.join(
        output_directory,
        'saves'
    )

    loop_directory = os.path.join(
        output_directory,
        'loop'
    )

    diagnostics_directory = os.path.join(
        output_directory,
        'diagnostics'
    )

    frontend_diagnostics_directory = os.path.join(
        diagnostics_directory,
        'frontend'
    )

    backend_diagnostics_directory = os.path.join(
        diagnostics_directory,
        'backend'
    )

    # ============================================================
    # FR-SLAM log directory
    #
    #   <workspace>/log/fr_slam
    #
    # ============================================================
    log_directory = os.path.join(
        workspace_directory,
        'log',
        'fr_slam'
    )

    # ============================================================
    # Create runtime directories
    # ============================================================
    os.makedirs(
        saves_directory,
        exist_ok=True
    )

    os.makedirs(
        loop_directory,
        exist_ok=True
    )

    os.makedirs(
        frontend_diagnostics_directory,
        exist_ok=True
    )

    os.makedirs(
        backend_diagnostics_directory,
        exist_ok=True
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
        executable='fr_slam_node',
        name='fr_slam',

        output='both',

    parameters=[
        slam_config_path,
        {
            'save_root_directory': saves_directory
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
