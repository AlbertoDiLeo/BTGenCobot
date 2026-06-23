#!/usr/bin/env python3
"""
Complete Launch File for BT Generation System with Localization (No SLAM)
Launches Gazebo, Nav2 with pre-built map, BT Interface Node, and Foxglove Bridge
Uses AMCL for localization instead of SLAM
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
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


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
    initial_pose = profile['initial_pose']
    return [
        SetLaunchConfiguration('resolved_world', profile['world_file']),
        SetLaunchConfiguration('resolved_map_file', profile['map_file']),
        SetLaunchConfiguration('resolved_spawn_x', str(spawn['x'])),
        SetLaunchConfiguration('resolved_spawn_y', str(spawn['y'])),
        SetLaunchConfiguration('resolved_spawn_yaw', str(spawn['yaw'])),
        SetLaunchConfiguration('resolved_initial_x', str(initial_pose['x'])),
        SetLaunchConfiguration('resolved_initial_y', str(initial_pose['y'])),
        SetLaunchConfiguration('resolved_initial_yaw', str(initial_pose['yaw'])),
    ]


def generate_launch_description():
    # Get package directories
    pkg_tb3_manipulation = get_package_share_directory('turtlebot3_manipulation_description')
    pkg_bt_bringup = get_package_share_directory('bt_bringup')

    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    environment = LaunchConfiguration('environment')
    resolved_world = LaunchConfiguration('resolved_world')
    resolved_map_file = LaunchConfiguration('resolved_map_file')
    resolved_spawn_x = LaunchConfiguration('resolved_spawn_x')
    resolved_spawn_y = LaunchConfiguration('resolved_spawn_y')
    resolved_spawn_yaw = LaunchConfiguration('resolved_spawn_yaw')
    resolved_initial_x = LaunchConfiguration('resolved_initial_x')
    resolved_initial_y = LaunchConfiguration('resolved_initial_y')
    resolved_initial_yaw = LaunchConfiguration('resolved_initial_yaw')
    inference_server_url = LaunchConfiguration('inference_server_url')
    bt_output_dir = LaunchConfiguration('bt_output_dir')

    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_environment_cmd = DeclareLaunchArgument(
        'environment',
        default_value='aws_small_house',
        description='Environment profile: aws_small_house | aws_hospital'
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
            'world': resolved_world,
            'headless': 'false',
            'x_pose': resolved_spawn_x,
            'y_pose': resolved_spawn_y,
            'yaw_pose': resolved_spawn_yaw,
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
            'yaml_filename': resolved_map_file
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

    # Launch AMCL only after the robot has been spawned and Gazebo TF/scan are stable.
    amcl_node = TimerAction(
        period=18.0,
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
                    'initial_pose.x': ParameterValue(
                        resolved_initial_x,
                        value_type=float,
                    ),
                    'initial_pose.y': ParameterValue(
                        resolved_initial_y,
                        value_type=float,
                    ),
                    'initial_pose.z': 0.0,
                    'initial_pose.yaw': ParameterValue(
                        resolved_initial_yaw,
                        value_type=float,
                    ),
                    # AMCL parameters
                    'min_particles': 500,
                    'max_particles': 2000,
                    'update_min_d': 0.1,  # Update after 10cm movement
                    'update_min_a': 0.1,  # Update after ~6° rotation
                    'resample_interval': 1,
                    'transform_tolerance': 2.0,  # Increased for slower hospital sim
                    'recovery_alpha_slow': 0.0,
                    'recovery_alpha_fast': 0.0,
                    'tf_broadcast': True,
                }]
            )
        ]
    )

    # Lifecycle manager for AMCL (delayed to match AMCL startup)
    amcl_lifecycle_node = TimerAction(
        period=23.0,
        actions=[
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='amcl_lifecycle_manager',
                output='screen',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'autostart': True,
                    'node_names': ['amcl'],
                    'bond_timeout': 20.0,
                }]
            )
        ]
    )

    # Configure Nav2 only after map_server and AMCL have established the
    # map -> odom -> base_footprint transform chain.
    nav2_launch = TimerAction(
        period=24.0,
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
            'execution_timeout': 300.0
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

    # Create launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_environment_cmd)
    ld.add_action(declare_inference_server_url_cmd)
    ld.add_action(declare_bt_output_dir_cmd)

    # Add launch files
    profiles_file = os.path.join(
        pkg_bt_bringup,
        'config',
        'environments.json',
    )
    ld.add_action(OpaqueFunction(
        function=resolve_environment_profile,
        args=[profiles_file],
    ))
    ld.add_action(environment_publisher)
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
