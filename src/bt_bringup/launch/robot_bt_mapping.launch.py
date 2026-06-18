#!/usr/bin/env python3
"""Dedicated saved-map construction runtime.

Starts only simulation, robot sensors/odometry, SLAM Toolbox and optional RViz.
Navigation, perception, manipulation and inference services are intentionally
excluded so Gazebo can spend more resources on stable scan acquisition.
"""

import json
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetLaunchConfiguration,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from nav2_common.launch import RewrittenYaml
from launch_ros.actions import Node


def resolve_environment_profile(context, profiles_file):
    environment_id = LaunchConfiguration('environment').perform(context)
    with open(profiles_file, encoding='utf-8') as stream:
        profiles = json.load(stream)

    profile = profiles.get(environment_id)
    if profile is None:
        supported = ', '.join(sorted(profiles))
        raise RuntimeError(
            f'Unknown environment "{environment_id}". Supported: {supported}'
        )

    spawn = profile['spawn']
    return [
        SetLaunchConfiguration('resolved_world', profile['world_file']),
        SetLaunchConfiguration('resolved_spawn_x', str(spawn['x'])),
        SetLaunchConfiguration('resolved_spawn_y', str(spawn['y'])),
        SetLaunchConfiguration('resolved_spawn_yaw', str(spawn['yaw'])),
        SetLaunchConfiguration(
            'resolved_slam_max_laser_range',
            str(profile['slam_max_laser_range']),
        ),
    ]


def generate_launch_description():
    pkg_description = get_package_share_directory(
        'turtlebot3_manipulation_description'
    )
    pkg_bringup = get_package_share_directory('bt_bringup')

    use_sim_time = LaunchConfiguration('use_sim_time')
    environment = LaunchConfiguration('environment')
    resolved_world = LaunchConfiguration('resolved_world')
    resolved_spawn_x = LaunchConfiguration('resolved_spawn_x')
    resolved_spawn_y = LaunchConfiguration('resolved_spawn_y')
    resolved_spawn_yaw = LaunchConfiguration('resolved_spawn_yaw')
    use_rviz = LaunchConfiguration('use_rviz')
    headless = LaunchConfiguration('headless')
    slam_params_file = LaunchConfiguration('slam_params_file')
    resolved_slam_max_laser_range = LaunchConfiguration(
        'resolved_slam_max_laser_range'
    )
    profiles_file = os.path.join(pkg_bringup, 'config', 'environments.json')

    configured_slam_params = RewrittenYaml(
        source_file=slam_params_file,
        root_key='',
        param_rewrites={
            'max_laser_range': resolved_slam_max_laser_range,
        },
        convert_types=True,
    )

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_description, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'use_rviz': use_rviz,
            'world': resolved_world,
            'headless': headless,
            'x_pose': resolved_spawn_x,
            'y_pose': resolved_spawn_y,
            'yaw_pose': resolved_spawn_yaw,
        }.items(),
    )

    environment_publisher = Node(
        package='bt_bringup',
        executable='environment_publisher.py',
        name='environment_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'environment_id': environment,
        }],
        output='screen',
    )

    slam_launch = TimerAction(
        period=8.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_bringup, 'launch', 'slam.launch.py')
                ),
                launch_arguments={
                    'use_sim_time': use_sim_time,
                    'slam_params_file': configured_slam_params,
                }.items(),
            )
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use the Gazebo simulation clock',
        ),
        DeclareLaunchArgument(
            'environment',
            default_value='aws_small_house',
            description='Environment profile: aws_small_house | aws_hospital',
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Start RViz for continuous map and scan inspection',
        ),
        DeclareLaunchArgument(
            'headless',
            default_value='false',
            description='Run Gazebo without its GUI',
        ),
        DeclareLaunchArgument(
            'slam_params_file',
            default_value=os.path.join(
                pkg_bringup,
                'config',
                'slam_toolbox_config.yaml',
            ),
            description='SLAM Toolbox configuration',
        ),
        OpaqueFunction(
            function=resolve_environment_profile,
            args=[profiles_file],
        ),
        environment_publisher,
        gazebo_launch,
        slam_launch,
    ])
