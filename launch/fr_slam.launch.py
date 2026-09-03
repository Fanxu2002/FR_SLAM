from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

from datetime import datetime
from pathlib import Path
import math
import os
import yaml


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


def _candidate_calibration_files(calibration_directory):
    candidates = []

    sessions_directory = calibration_directory / 'sessions'
    if sessions_directory.is_dir():
        candidates.extend(
            sessions_directory.glob(
                '*/lidar_imu_rotation_calibration*.yaml'
            )
        )

    candidates.extend(
        calibration_directory.glob(
            'lidar_imu_rotation_calibration*.yaml'
        )
    )

    return [path for path in candidates if path.is_file()]


def _select_calibration_file(
    calibration_directory,
    sensor,
    legacy_calibration_directory
):
    environment_name = (
        'FR_SLAM_'
        + sensor.upper()
        + '_CALIBRATION_FILE'
    )
    explicit_path = os.environ.get(
        environment_name,
        ''
    ).strip()

    if explicit_path:
        selected_path = Path(explicit_path).expanduser().resolve()
        if not selected_path.is_file():
            raise RuntimeError(
                environment_name
                + ' does not exist: '
                + str(selected_path)
            )
        return selected_path

    candidates = _candidate_calibration_files(
        calibration_directory
    )

    # The old output/calibration/*.yaml layout was created with Livox.
    # It is deliberately never considered for Hesai.
    if sensor == 'livox':
        candidates.extend(
            _candidate_calibration_files(
                legacy_calibration_directory
            )
        )

    if not candidates:
        raise RuntimeError(
            'No solved '
            + sensor
            + ' LiDAR-IMU calibration YAML was found under: '
            + str(calibration_directory)
            + '\nRun lidar_imu_calibration.launch.py sensor:='
            + sensor
            + ' and the offline solver before starting normal FR-SLAM.'
        )

    return max(
        set(candidates),
        key=lambda path: path.stat().st_mtime_ns
    )


def _load_q_il(calibration_file):
    with calibration_file.open(
        mode='r',
        encoding='utf-8'
    ) as input_stream:
        document = yaml.safe_load(input_stream)

    try:
        calibration = document[
            'lidar_imu_rotation_calibration'
        ]
        quaternion = calibration[
            'inverse_lidar_to_imu'
        ][
            'quaternion_xyzw'
        ]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            'Calibration YAML does not contain '
            'inverse_lidar_to_imu.quaternion_xyzw: '
            + str(calibration_file)
        ) from error

    diagnostics = calibration.get('diagnostics', {})
    if diagnostics.get('excitation_sufficient') is not True:
        raise RuntimeError(
            'Refusing to start normal FR-SLAM with a calibration whose '
            'excitation_sufficient value is not true: '
            + str(calibration_file)
        )

    if not isinstance(quaternion, list) or len(quaternion) != 4:
        raise RuntimeError(
            'Calibration quaternion must be [x, y, z, w]: '
            + str(calibration_file)
        )

    quaternion = [float(value) for value in quaternion]
    if not all(math.isfinite(value) for value in quaternion):
        raise RuntimeError(
            'Calibration quaternion contains a non-finite value: '
            + str(calibration_file)
        )

    norm = math.sqrt(
        sum(value * value for value in quaternion)
    )
    if norm < 1.0e-12:
        raise RuntimeError(
            'Calibration quaternion has zero norm: '
            + str(calibration_file)
        )

    return [value / norm for value in quaternion]


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
    slam_config_path = (
        package_share_directory /
        'config' /
        ('fr_slam_' + sensor + '.yaml')
    )
    if not slam_config_path.is_file():
        raise RuntimeError(
            'Sensor profile YAML does not exist: '
            + str(slam_config_path)
        )

    output_directory = (
        workspace_directory /
        'src' /
        'fr_slam' /
        'output'
    )
    legacy_calibration_directory = (
        output_directory /
        'calibration'
    )
    calibration_directory = (
        legacy_calibration_directory /
        sensor
    )

    calibration_file = _select_calibration_file(
        calibration_directory,
        sensor,
        legacy_calibration_directory
    )
    q_il = _load_q_il(calibration_file)

    # Each sensor keeps an independent run history.
    run_id = datetime.now().strftime(
        '%Y%m%d_%H%M%S_%f'
    )
    run_directory = (
        output_directory /
        'runs' /
        sensor /
        run_id
    )
    saves_directory = run_directory / 'saves'
    loop_directory = run_directory / 'loop'
    frontend_diagnostics_directory = (
        run_directory /
        'diagnostics' /
        'frontend'
    )
    backend_diagnostics_directory = (
        run_directory /
        'diagnostics' /
        'backend'
    )
    log_directory = (
        workspace_directory /
        'log' /
        'fr_slam' /
        'runs' /
        sensor /
        run_id
    )

    for directory in (
        saves_directory,
        loop_directory,
        frontend_diagnostics_directory,
        backend_diagnostics_directory
    ):
        directory.mkdir(parents=True, exist_ok=False)
    log_directory.mkdir(parents=True, exist_ok=False)

    slam_node = Node(
        package='fr_slam',
        executable='fr_slam_node',
        name='fr_slam',
        output='both',
        parameters=[
            str(slam_config_path),
            {
                'save_root_directory': str(saves_directory),
                'enable_lidar_imu_rotation_pair_export': False,
                'calibration_use_imu_initial_guess': True,
                'imu_extrinsic_q_il_x': q_il[0],
                'imu_extrinsic_q_il_y': q_il[1],
                'imu_extrinsic_q_il_z': q_il[2],
                'imu_extrinsic_q_il_w': q_il[3]
            }
        ],
        emulate_tty=True,
        additional_env={
            'ROS_LOG_DIR': str(log_directory),
            'FR_SLAM_OUTPUT_DIR': str(run_directory)
        }
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='fr_slam_rviz',
        output='screen',
        arguments=[
            '-d',
            str(rviz_config_path)
        ]
    )

    return [
        LogInfo(
            msg='FR-SLAM NORMAL MODE | sensor=' + sensor
        ),
        LogInfo(
            msg='Sensor profile: ' + str(slam_config_path)
        ),
        LogInfo(
            msg='Calibration loaded from: '
                + str(calibration_file)
        ),
        LogInfo(
            msg='This run will be preserved under: '
                + str(run_directory)
        ),
        slam_node,
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
