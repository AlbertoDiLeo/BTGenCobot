#!/usr/bin/env python3
"""
Complete Launch File for BT Generation System
Launches Gazebo, SLAM/Nav2, BT Interface Node, and Foxglove Bridge
Based on full_nav2_bringup.launch.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directories
    pkg_tb3_manipulation = get_package_share_directory('turtlebot3_manipulation_description')
    pkg_bt_bringup = get_package_share_directory('bt_bringup')

    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    world = LaunchConfiguration('world')
    slam_mode = LaunchConfiguration('slam_mode')
    inference_server_url = LaunchConfiguration('inference_server_url')
    bt_output_dir = LaunchConfiguration('bt_output_dir')

    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_world_cmd = DeclareLaunchArgument(
        'world',
        default_value='default',
        description='World selector: default | structured_house | aws_small_house | absolute path to .sdf/.world'
    )

    declare_slam_mode_cmd = DeclareLaunchArgument(
        'slam_mode',
        default_value='true',
        description='Whether to run in SLAM mode (true) or use pre-built map (false)'
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

    # Launch Gazebo with robot (use_rviz=true to start RViz, headless=false for GUI)
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tb3_manipulation, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'use_rviz': 'true',
            'world': world,
            'headless': 'false'
        }.items()
    )

    # Launch SLAM Toolbox
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bt_bringup, 'launch', 'slam.launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # Launch Nav2
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bt_bringup, 'launch', 'nav2_bringup.launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # Launch BT Text Interface Node (Action Server for BT generation)
    bt_interface_node = Node(
        package='bt_text_interface',
        executable='bt_interface_node',
        name='bt_interface_node',
        parameters=[{
            'use_sim_time': use_sim_time,
            'inference_server_url': inference_server_url,
            'bt_output_dir': bt_output_dir,
            'generation_timeout': 30.0,
            'execution_timeout': 120.0
        }],
        output='screen',
        emulate_tty=True
    )

    # Launch Florence-2 Object Detection Service
    # Uses Florence-2 for text-prompted object detection
    florence2_service = Node(
        package='vision_services',
        executable='florence2_service',
        name='florence2_service',
        parameters=[{
            'use_sim_time': use_sim_time,
            'use_mock': False,  # Use real models
            'florence2_model': 'microsoft/Florence-2-base',
            'device': 'auto',
            'publish_debug_images': True,
        }],
        output='screen',
    )

    # Launch Manipulator Control Service
    # Uses ikpy for inverse kinematics and ros2_control for trajectory execution
    manipulator_service = Node(
        package='manipulator_control',
        executable='manipulator_service',
        name='manipulator_service',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
        output='screen',
    )


    # Launch Foxglove Bridge with client publish capability
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
            'client_topic_whitelist': ['.*'],  # Enable client publishing on all topics
            'use_sim_time': use_sim_time,
            'capabilities': ['clientPublish', 'services', 'parameters', 'connectionGraph'],
        }],
        output='screen'
    )

    # Create launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_world_cmd)
    ld.add_action(declare_slam_mode_cmd)
    ld.add_action(declare_inference_server_url_cmd)
    ld.add_action(declare_bt_output_dir_cmd)

    # Add launch files
    ld.add_action(gazebo_launch)
    ld.add_action(slam_launch)
    ld.add_action(nav2_launch)

    # Add BT Interface Node (main action server)
    ld.add_action(bt_interface_node)

    # Add Florence-2 Service (object detection)
    ld.add_action(florence2_service)

    # Add Manipulator Control Service (pick/place operations)
    ld.add_action(manipulator_service)

    # Add Foxglove Bridge
    ld.add_action(foxglove_bridge)

    return ld
