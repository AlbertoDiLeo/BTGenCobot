#!/usr/bin/env python3
"""
Complete Launch File for BT Generation System with Localization (No SLAM)
Launches Gazebo, Nav2 with pre-built map, BT Interface Node, and Foxglove Bridge
Uses AMCL for localization instead of SLAM
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
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
    map_file = LaunchConfiguration('map_file')
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
        default_value='/workspace/worlds/indoor_world.sdf',
        description='Full path to world file to load'
    )

    declare_map_file_cmd = DeclareLaunchArgument(
        'map_file',
        default_value='/workspace/maps/my_map.yaml',
        description='Full path to map yaml file to use for localization'
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

    # Launch Map Server (instead of SLAM)
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'yaml_filename': map_file
        }]
    )

    # Launch Lifecycle Manager for Map Server
    map_lifecycle_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='map_lifecycle_manager',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': ['map_server']
        }]
    )

    # Launch AMCL for localization with delay to allow TF to stabilize
    # Wrapped in TimerAction to ensure Gazebo TF is publishing before AMCL starts
    amcl_node = TimerAction(
        period=8.0,  # Wait for Gazebo and TF bridges to stabilize
        actions=[
            Node(
                package='nav2_amcl',
                executable='amcl',
                name='amcl',
                output='screen',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'global_frame_id': 'map',
                    'odom_frame_id': 'odom',
                    'base_frame_id': 'base_footprint',
                    'scan_topic': 'scan',
                    'robot_model_type': 'nav2_amcl::DifferentialMotionModel',
                    'set_initial_pose': True,
                    'initial_pose.x': 0.0,
                    'initial_pose.y': 0.0,
                    'initial_pose.z': 0.0,
                    'initial_pose.yaw': 0.0,
                    # AMCL parameters
                    'min_particles': 500,
                    'max_particles': 2000,
                    'update_min_d': 0.1,  # Update after 10cm movement
                    'update_min_a': 0.1,  # Update after ~6° rotation
                    'resample_interval': 1,
                    'transform_tolerance': 1.0,  # Increased for Gazebo timing jitter
                    'recovery_alpha_slow': 0.0,
                    'recovery_alpha_fast': 0.0,
                    'tf_broadcast': True,
                }]
            )
        ]
    )

    # Lifecycle manager for AMCL (delayed to match AMCL startup)
    amcl_lifecycle_node = TimerAction(
        period=9.0,  # Start after AMCL has had time to initialize
        actions=[
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='amcl_lifecycle_manager',
                output='screen',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'autostart': True,
                    'node_names': ['amcl']
                }]
            )
        ]
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
            'generation_timeout': 180.0,
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
        output='screen'
    )

    # Launch Manipulator Control Service (pick/place using ikpy)
    manipulator_service = Node(
        package='manipulator_control',
        executable='manipulator_service',
        name='manipulator_service',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
        output='screen'
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
    ld.add_action(declare_map_file_cmd)
    ld.add_action(declare_inference_server_url_cmd)
    ld.add_action(declare_bt_output_dir_cmd)

    # Add launch files
    ld.add_action(gazebo_launch)

    # Add Map Server and AMCL (instead of SLAM)
    ld.add_action(map_server_node)
    ld.add_action(map_lifecycle_node)
    ld.add_action(amcl_node)
    ld.add_action(amcl_lifecycle_node)

    # Add Nav2
    ld.add_action(nav2_launch)

    # Add BT Interface Node (main action server)
    ld.add_action(bt_interface_node)

    # Add Florence-2 Service (object detection)
    ld.add_action(florence2_service)

    # Add Manipulator Control Service (pick/place)
    ld.add_action(manipulator_service)

    # Add Foxglove Bridge
    ld.add_action(foxglove_bridge)

    return ld
