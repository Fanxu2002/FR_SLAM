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
SUPPORTED_PROFILES = ('outdoor', 'indoor')


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

def _profile_name(context):
    profile = (
        LaunchConfiguration('profile')
        .perform(context)
        .strip()
        .lower()
    )

    if profile not in SUPPORTED_PROFILES:
        raise RuntimeError(
            'Unsupported environment profile: '
            + profile
            + '. Valid values: '
            + ', '.join(SUPPORTED_PROFILES)
        )

    return profile

def _optional_boolean(context, argument_name):
    value = LaunchConfiguration(argument_name).perform(
        context
    ).strip().lower()

    if not value:
        return None

    if value in ('true', '1', 'on', 'yes'):
        return True

    if value in ('false', '0', 'off', 'no'):
        return False

    raise RuntimeError(
        'Invalid boolean launch argument '
        + argument_name
        + '='
        + value
        + '. Use true or false.'
    )



def _optional_float(context, argument_name):
    value = LaunchConfiguration(argument_name).perform(
        context
    ).strip()

    if not value:
        return None

    try:
        parsed = float(value)
    except ValueError as error:
        raise RuntimeError(
            'Invalid numeric launch argument '
            + argument_name
            + '='
            + value
        ) from error

    if not math.isfinite(parsed) or parsed <= 0.0:
        raise RuntimeError(
            'Launch argument '
            + argument_name
            + ' must be finite and > 0.'
        )

    return parsed

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
    profile = _profile_name(context)
    backend_loop_closure_override = _optional_boolean(
        context,
        'backend_loop_closure_enable'
    )
    ground_constraint_override = _optional_boolean(
        context,
        'ground_constraint_enable'
    )
    planar_motion_override = _optional_boolean(
        context,
        'planar_motion_mode'
    )
    wall_constraint_minimum_radius_override = _optional_float(
        context,
        'wall_constraint_minimum_radius_m'
    )
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
    # Prefer the source-tree YAML during development so parameter-only
    # changes do not require rebuilding/installing the ROS package.
    source_slam_config_path = (
        workspace_directory /
        'src' /
        'fr_slam' /
        'config' /
        ('fr_slam_' + sensor + '.yaml')
    )

    installed_slam_config_path = (
        package_share_directory /
        'config' /
        ('fr_slam_' + sensor + '.yaml')
    )

    if source_slam_config_path.is_file():
        slam_config_path = source_slam_config_path
    else:
        slam_config_path = installed_slam_config_path

    if not slam_config_path.is_file():
        raise RuntimeError(
            'Sensor profile YAML does not exist: '
            + str(slam_config_path)
        )

    # Environment-specific Wall profile.
    #
    # Keep sensor calibration / LiDAR / Ground configuration in the normal
    # sensor YAML, and override only Wall-specific parameters here.
    source_wall_profile_config_path = (
        workspace_directory /
        'src' /
        'fr_slam' /
        'config' /
        ('fr_slam_wall_' + profile + '.yaml')
    )

    installed_wall_profile_config_path = (
        package_share_directory /
        'config' /
        ('fr_slam_wall_' + profile + '.yaml')
    )

    if source_wall_profile_config_path.is_file():
        wall_profile_config_path = (
            source_wall_profile_config_path
        )
    else:
        wall_profile_config_path = (
            installed_wall_profile_config_path
        )

    if not wall_profile_config_path.is_file():
        raise RuntimeError(
            'Wall profile YAML does not exist: '
            + str(wall_profile_config_path)
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

    parameter_overrides = {
        'save_root_directory': str(saves_directory),
        'enable_lidar_imu_rotation_pair_export': False,
        'calibration_use_imu_initial_guess': True,
        'imu_extrinsic_q_il_x': q_il[0],
        'imu_extrinsic_q_il_y': q_il[1],
        'imu_extrinsic_q_il_z': q_il[2],
        'imu_extrinsic_q_il_w': q_il[3]
    }

    if backend_loop_closure_override is not None:
        parameter_overrides['backend_loop_closure_enable'] = (
            backend_loop_closure_override
        )

    if ground_constraint_override is not None:
        parameter_overrides['ground_constraint_enable'] = (
            ground_constraint_override
        )

    if planar_motion_override is not None:
        parameter_overrides['planar_motion_mode'] = (
            planar_motion_override
        )

    if wall_constraint_minimum_radius_override is not None:
        parameter_overrides[
            'wall_constraint_minimum_radius_m'
        ] = wall_constraint_minimum_radius_override

    slam_node = Node(
        package='fr_slam',
        executable='lo_node',
        name='fr_slam',
        output='both',
        parameters=[
            str(slam_config_path),
            str(wall_profile_config_path),
            parameter_overrides
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
            msg='Environment profile: ' + profile
        ),
        LogInfo(
            msg='Wall profile: '
                + str(wall_profile_config_path)
        ),
        LogInfo(
            msg='Calibration loaded from: '
                + str(calibration_file)
        ),
        LogInfo(
            msg='This run will be preserved under: '
                + str(run_directory)
        ),
        LogInfo(
            msg=(
                'Ablation overrides | backend_loop_closure='
                + (
                    'YAML default'
                    if backend_loop_closure_override is None
                    else (
                        'ON'
                        if backend_loop_closure_override
                        else 'OFF'
                    )
                )
                + ' | ground_constraint='
                + (
                    'YAML default'
                    if ground_constraint_override is None
                    else (
                        'ON'
                        if ground_constraint_override
                        else 'OFF'
                    )
                )
                + ' | planar_motion='
                + (
                    'YAML default'
                    if planar_motion_override is None
                    else (
                        'ON'
                        if planar_motion_override
                        else 'OFF'
                    )
                )
            )
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
        DeclareLaunchArgument(
            'profile',
            default_value='outdoor',
            description='Environment profile: outdoor or indoor'
        ),
        DeclareLaunchArgument(
            'backend_loop_closure_enable',
            default_value='',
            description=(
                'Optional loop-closure override: true/false; '
                'empty uses the sensor YAML value'
            )
        ),
        DeclareLaunchArgument(
            'ground_constraint_enable',
            default_value='',
            description=(
                'Optional Ground-constraint override: true/false; '
                'empty uses the sensor YAML value'
            )
        ),
        DeclareLaunchArgument(
            'planar_motion_mode',
            default_value='',
            description=(
                'Optional planar-motion debug override: true/false; '
                'empty uses the sensor YAML value'
            )
        ),
        DeclareLaunchArgument(
            'wall_constraint_minimum_radius_m',
            default_value='',
            description=(
                'Optional MultiPlane Wall-only minimum radius in meters; '
                'empty uses the selected Wall profile YAML. '
                'This is only a temporary command-line override.'
            )
        ),
        OpaqueFunction(function=_launch_setup)
    ])
