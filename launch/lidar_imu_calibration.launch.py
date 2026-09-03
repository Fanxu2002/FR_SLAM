from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

from datetime import datetime
from pathlib import Path


SUPPORTED_SENSORS = ('livox', 'hesai')


def _workspace_directory(package_share_directory):
    # Keep the installed path structure intact under --symlink-install.
    return Path(package_share_directory).absolute().parents[3]


def _sensor_name(context):
    sensor = LaunchConfiguration('sensor').perform(context).strip().lower()
    if sensor not in SUPPORTED_SENSORS:
        raise RuntimeError(
            'Unsupported sensor profile: '
            + sensor
            + '. Valid values: '
            + ', '.join(SUPPORTED_SENSORS)
        )
    return sensor


def _launch_setup(context):
    sensor = _sensor_name(context)
    package_share_directory = Path(
        get_package_share_directory('fr_slam')
    )
    workspace_directory = _workspace_directory(
        package_share_directory
    )

    rviz_config_path = (
        package_share_directory /
        'config' /
        'FR_SLAM.rviz'
    )
    sensor_config_path = (
        package_share_directory /
        'config' /
        ('fr_slam_' + sensor + '.yaml')
    )
    if not sensor_config_path.is_file():
        raise RuntimeError(
            'Sensor profile YAML does not exist: '
            + str(sensor_config_path)
        )

    # Calibration sessions are isolated by both sensor and timestamp.
    session_id = datetime.now().strftime(
        '%Y%m%d_%H%M%S_%f'
    )
    output_directory = (
        workspace_directory /
        'src' /
        'fr_slam' /
        'output'
    )
    session_directory = (
        output_directory /
        'calibration' /
        sensor /
        'sessions' /
        session_id
    )
    saves_directory = session_directory / 'saves'
    loop_directory = session_directory / 'loop'
    frontend_diagnostics_directory = (
        session_directory /
        'diagnostics' /
        'frontend'
    )
    backend_diagnostics_directory = (
        session_directory /
        'diagnostics' /
        'backend'
    )
    rotation_pairs_path = (
        session_directory /
        'lidar_imu_rotation_pairs.csv'
    )
    calibration_result_path = (
        session_directory /
        'lidar_imu_rotation_calibration.yaml'
    )
    log_directory = (
        workspace_directory /
        'log' /
        'fr_slam' /
        'calibration' /
        sensor /
        session_id
    )

    for directory in (
        saves_directory,
        loop_directory,
        frontend_diagnostics_directory,
        backend_diagnostics_directory
    ):
        directory.mkdir(parents=True, exist_ok=False)
    log_directory.mkdir(parents=True, exist_ok=False)

    collector_node = Node(
        package='fr_slam',
        executable='fr_slam_node',
        name='fr_slam',
        output='both',
        parameters=[
            str(sensor_config_path),
            {
                'save_root_directory': str(saves_directory),
                'lidar_imu_rotation_pairs_path':
                    str(rotation_pairs_path),
                'enable_lidar_imu_rotation_pair_export': True,
                'calibration_use_imu_initial_guess': False,
                'imu_extrinsic_q_il_x': 0.0,
                'imu_extrinsic_q_il_y': 0.0,
                'imu_extrinsic_q_il_z': 0.0,
                'imu_extrinsic_q_il_w': 1.0,
                'imu_extrinsic_p_il_x': 0.0,
                'imu_extrinsic_p_il_y': 0.0,
                'imu_extrinsic_p_il_z': 0.0
            }
        ],
        emulate_tty=True,
        additional_env={
            'ROS_LOG_DIR': str(log_directory),
            'FR_SLAM_OUTPUT_DIR': str(session_directory)
        }
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='fr_slam_calibration_rviz',
        output='screen',
        arguments=[
            '-d',
            str(rviz_config_path)
        ]
    )

    solver_command = (
        'ros2 run fr_slam lidar_imu_rotation_calibration '
        '--input ' + str(rotation_pairs_path) + ' '
        '--output ' + str(calibration_result_path) + ' '
        '--min-angle-deg 0.2'
    )

    return [
        LogInfo(
            msg='FR-SLAM LIDAR-IMU CALIBRATION MODE | sensor='
                + sensor
        ),
        LogInfo(
            msg='Sensor profile: ' + str(sensor_config_path)
        ),
        LogInfo(
            msg='This session cannot overwrite another sensor/session: '
                + str(session_directory)
        ),
        LogInfo(
            msg='After collecting data and stopping this launch, run: '
                + solver_command
        ),
        collector_node,
        rviz_node
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'sensor',
            default_value='livox',
            description='Sensor profile: livox or hesai'
        ),
        OpaqueFunction(function=_launch_setup)
    ])
