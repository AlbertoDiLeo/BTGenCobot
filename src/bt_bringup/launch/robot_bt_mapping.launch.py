#!/usr/bin/env python3
"""Dedicated saved-map construction runtime.

Starts only simulation, robot sensors/odometry, SLAM Toolbox and optional RViz.
Navigation, perception, manipulation and inference services are intentionally
excluded so Gazebo can spend more resources on stable scan acquisition.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_description = get_package_share_directory(
        'turtlebot3_manipulation_description'
    )
    pkg_bringup = get_package_share_directory('bt_bringup')

    use_sim_time = LaunchConfiguration('use_sim_time')
    world = LaunchConfiguration('world')
    use_rviz = LaunchConfiguration('use_rviz')
    headless = LaunchConfiguration('headless')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    slam_params_file = LaunchConfiguration('slam_params_file')

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_description, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'use_rviz': use_rviz,
            'world': world,
            'headless': headless,
            'x_pose': x_pose,
            'y_pose': y_pose,
        }.items(),
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
                    'slam_params_file': slam_params_file,
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
            'world',
            default_value='aws_small_house',
            description='World selector passed to the Gazebo launch',
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
            'x_pose',
            default_value='0.35',
            description='Initial Gazebo world x position',
        ),
        DeclareLaunchArgument(
            'y_pose',
            default_value='0.05',
            description='Initial Gazebo world y position',
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
        gazebo_launch,
        slam_launch,
    ])
