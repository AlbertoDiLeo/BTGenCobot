#!/usr/bin/env python3
"""
Frontend launch for the AWS small house world.
Starts the simulation, mapping, navigation, BT text interface, perception,
manipulation services, and Foxglove bridge needed by the frontend.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_tb3_manipulation = get_package_share_directory('turtlebot3_manipulation_description')
    pkg_bt_bringup = get_package_share_directory('bt_bringup')

    use_sim_time = LaunchConfiguration('use_sim_time')
    world = LaunchConfiguration('world')
    use_rviz = LaunchConfiguration('use_rviz')
    headless = LaunchConfiguration('headless')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    inference_server_url = LaunchConfiguration('inference_server_url')
    bt_output_dir = LaunchConfiguration('bt_output_dir')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_world_cmd = DeclareLaunchArgument(
        'world',
        default_value='aws_small_house',
        description='World selector: default | structured_house | aws_small_house | absolute path to .sdf/.world'
    )

    declare_use_rviz_cmd = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to start RViz'
    )

    declare_headless_cmd = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo in headless mode (no GUI)'
    )

    declare_x_pose_cmd = DeclareLaunchArgument(
        'x_pose',
        default_value='0.35',
        description='Initial x position of the robot'
    )

    declare_y_pose_cmd = DeclareLaunchArgument(
        'y_pose',
        default_value='0.05',
        description='Initial y position of the robot'
    )

    declare_inference_server_url_cmd = DeclareLaunchArgument(
        'inference_server_url',
        default_value='http://host.docker.internal:8080',
        description='URL of the BT generation inference server'
    )

    declare_bt_output_dir_cmd = DeclareLaunchArgument(
        'bt_output_dir',
        default_value='/workspace/generated_bts',
        description='Directory to save generated BehaviorTrees'
    )

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tb3_manipulation, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'use_rviz': use_rviz,
            'world': world,
            'headless': headless,
            'x_pose': x_pose,
            'y_pose': y_pose,
        }.items()
    )

    # Gazebo.launch.py spawns the robot after 3 seconds. Start SLAM only after
    # the sensor bridges and odom/tf chain have had time to appear.
    slam_launch = TimerAction(
        period=8.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_bt_bringup, 'launch', 'slam.launch.py')
                ),
                launch_arguments={
                    'use_sim_time': use_sim_time
                }.items()
            )
        ]
    )

    nav2_launch = TimerAction(
        period=14.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_bt_bringup, 'launch', 'nav2_bringup.launch.py')
                ),
                launch_arguments={
                    'use_sim_time': use_sim_time
                }.items()
            )
        ]
    )

    # Frontend NL execution relies on bt_text_interface subscribing to
    # /btgen_nl_command and forwarding generation/execution through Nav2.
    bt_interface_node = TimerAction(
        period=16.0,
        actions=[
            Node(
                package='bt_text_interface',
                executable='bt_interface_node',
                name='bt_interface_node',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'inference_server_url': inference_server_url,
                    'bt_output_dir': bt_output_dir,
                    'generation_timeout': 30.0,
                    'execution_timeout': 300.0,
                }],
                output='screen',
                emulate_tty=True,
            )
        ]
    )

    florence2_service = TimerAction(
        period=18.0,
        actions=[
            Node(
                package='vision_services',
                executable='florence2_service',
                name='florence2_service',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'use_mock': False,
                    'florence2_model': 'microsoft/Florence-2-base',
                    'device': 'auto',
                    'publish_debug_images': True,
                }],
                output='screen',
            )
        ]
    )

    manipulator_service = TimerAction(
        period=18.0,
        actions=[
            Node(
                package='manipulator_control',
                executable='manipulator_service',
                name='manipulator_service',
                parameters=[{
                    'use_sim_time': use_sim_time,
                }],
                output='screen',
            )
        ]
    )

    foxglove_bridge = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        parameters=[{
            'port': 8765,
            'address': '0.0.0.0',
            'tls': False,
            'certfile': '',
            'keyfile': '',
            'topic_whitelist': ['.*'],
            'service_whitelist': ['.*'],
            'param_whitelist': ['.*'],
            'client_topic_whitelist': ['.*'],
            'use_sim_time': use_sim_time,
            'capabilities': ['clientPublish', 'services', 'parameters', 'connectionGraph'],
        }],
        output='screen'
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_world_cmd)
    ld.add_action(declare_use_rviz_cmd)
    ld.add_action(declare_headless_cmd)
    ld.add_action(declare_x_pose_cmd)
    ld.add_action(declare_y_pose_cmd)
    ld.add_action(declare_inference_server_url_cmd)
    ld.add_action(declare_bt_output_dir_cmd)
    ld.add_action(gazebo_launch)
    ld.add_action(slam_launch)
    ld.add_action(nav2_launch)
    ld.add_action(bt_interface_node)
    ld.add_action(florence2_service)
    ld.add_action(manipulator_service)
    ld.add_action(foxglove_bridge)
    return ld
