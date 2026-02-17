#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, Command, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Get package directories
    pkg_description = get_package_share_directory('turtlebot3_manipulation_description')
    
    # Build resource paths for meshes
    resource_paths = [
        pkg_description,
        os.path.dirname(pkg_description),
        '/workspace/worlds/models',
        '/workspace/models/turtlebot3_gazebo/models',
    ]
    gz_resource_path = ':'.join(resource_paths)

    # Paths
    urdf_file = os.path.join(pkg_description, 'urdf', 'turtlebot3_manipulation.urdf.xacro')
    controller_config_file = os.path.join(pkg_description, 'config', 'controller_manager.yaml')

    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')
    world_file_arg = LaunchConfiguration('world')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    headless = LaunchConfiguration('headless')

    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_use_rviz_cmd = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Whether to start RViz'
    )

    declare_world_cmd = DeclareLaunchArgument(
        'world',
        default_value='/workspace/worlds/indoor_world.sdf',
        description='Full path to world file to load'
    )

    declare_x_pose_cmd = DeclareLaunchArgument(
        'x_pose',
        default_value='0.0',
        description='Initial x position of the robot'
    )

    declare_y_pose_cmd = DeclareLaunchArgument(
        'y_pose',
        default_value='0.0',
        description='Initial y position of the robot'
    )

    declare_headless_cmd = DeclareLaunchArgument(
        'headless',
        default_value='true',
        description='Run Gazebo in headless mode (no GUI)'
    )

    # Start Gazebo Harmonic - headless or with GUI based on parameter
    env_vars = {
        'GZ_SIM_RESOURCE_PATH': gz_resource_path,
        'IGN_GAZEBO_RESOURCE_PATH': gz_resource_path,
        'GZ_SIM_SYSTEM_PLUGIN_PATH': ':'.join([
            '/opt/ros/jazzy/lib',
            '/opt/ros/jazzy/opt/gz_sim_vendor/lib'
        ])
    }
    
    # Run headless (no GUI) with -s (server only) flag
    start_gazebo_headless_cmd = ExecuteProcess(
        condition=IfCondition(headless),
        cmd=['gz', 'sim', '-r', '-s', '-v', '4', world_file_arg],
        output='screen',
        additional_env=env_vars
    )

    # Run with GUI (no -s flag)
    start_gazebo_gui_cmd = ExecuteProcess(
        condition=UnlessCondition(headless),
        cmd=['gz', 'sim', '-r', '-v', '4', world_file_arg],
        output='screen',
        additional_env=env_vars
    )

    # Robot State Publisher - use_sim argument set to true for Gazebo Harmonic
    robot_description_content = Command([
        'xacro', ' ', urdf_file,
        ' ', 'prefix:=',
        ' ', 'use_sim:=true'
    ])
    
    robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': ParameterValue(robot_description_content, value_type=str)
        }],
        remappings=[('/joint_states', '/joint_states_combined')]
    )

    # Spawn robot in Gazebo
    spawn_robot_cmd = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='ros_gz_sim',
                executable='create',
                arguments=[
                    '-name', 'turtlebot3_manipulation',
                    '-topic', 'robot_description',
                    '-x', x_pose,
                    '-y', y_pose,
                    '-z', '0.1'
                ],
                output='screen'
            )
        ]
    )

    # Bridge for clock
    bridge_clock_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen'
    )

    # Bridge for sensor topics
    bridge_sensors_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/model/turtlebot3_manipulation/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
            '/camera@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/camera/depth@sensor_msgs/msg/Image[gz.msgs.Image'
        ],
        output='screen',
        remappings=[
            ('/model/turtlebot3_manipulation/odometry', '/odom')
        ],
        parameters=[{
            'qos_overrides./scan.publisher.reliability': 'best_effort',
            'qos_overrides./scan.publisher.durability': 'volatile',
            'qos_overrides./scan.publisher.history': 'keep_last',
            'qos_overrides./scan.publisher.depth': 5
        }]
    )

    # Bridge for cmd_vel - bidirectional bridge from ROS to Gazebo model
    bridge_cmd_vel = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/model/turtlebot3_manipulation/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist'],
        output='screen',
        remappings=[
            ('/model/turtlebot3_manipulation/cmd_vel', '/cmd_vel')
        ]
    )

    # Bridge for TF transforms
    bridge_tf_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='odom_base_tf_bridge',
        arguments=['/model/turtlebot3_manipulation/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V'],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[('/model/turtlebot3_manipulation/tf', '/tf')]
    )

    # Bridge for Gazebo wheel joint states - publish to separate topic
    bridge_wheel_joint_states = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='wheel_joint_states_bridge',
        arguments=['/gazebo_joint_states@sensor_msgs/msg/JointState[gz.msgs.Model'],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[('/gazebo_joint_states', '/wheel_states')]
    )

    # Joint state publisher to merge wheel and arm joint states
    joint_state_publisher_cmd = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_merger',
        parameters=[{
            'use_sim_time': use_sim_time,
            'source_list': ['/joint_states', '/wheel_states'],
            'rate': 20
        }],
        remappings=[('joint_states', '/joint_states_combined')],
        output='screen'
    )

    # RViz
    rviz_config_file = os.path.join(pkg_description, 'rviz', 'model.rviz')
    rviz_cmd = TimerAction(
        period=5.0,
        actions=[
            Node(
                condition=IfCondition(use_rviz),
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', rviz_config_file],
                parameters=[{'use_sim_time': use_sim_time}],
                output='screen'
            )
        ]
    )

    # Spawn controllers - delays increased to ensure Gazebo and ros2_control are fully initialized
    spawn_joint_state_broadcaster_cmd = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
                output='screen'
            )
        ]
    )

    spawn_imu_broadcaster_cmd = TimerAction(
        period=10.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['imu_broadcaster', '--controller-manager', '/controller_manager'],
                output='screen'
            )
        ]
    )

    spawn_arm_controller_cmd = TimerAction(
        period=12.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['arm_controller', '--controller-manager', '/controller_manager'],
                output='screen'
            )
        ]
    )

    spawn_gripper_controller_cmd = TimerAction(
        period=14.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['gripper_controller', '--controller-manager', '/controller_manager'],
                output='screen'
            )
        ]
    )

    # Create launch description
    ld = LaunchDescription()

    # Declare launch options
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_use_rviz_cmd)
    ld.add_action(declare_world_cmd)
    ld.add_action(declare_x_pose_cmd)
    ld.add_action(declare_y_pose_cmd)
    ld.add_action(declare_headless_cmd)

    # Add nodes
    ld.add_action(start_gazebo_headless_cmd)
    ld.add_action(start_gazebo_gui_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(bridge_clock_cmd)
    ld.add_action(bridge_sensors_cmd)
    ld.add_action(bridge_cmd_vel)
    ld.add_action(bridge_tf_cmd)
    ld.add_action(bridge_wheel_joint_states)
    ld.add_action(joint_state_publisher_cmd)
    ld.add_action(spawn_robot_cmd)
    ld.add_action(rviz_cmd)
    ld.add_action(spawn_joint_state_broadcaster_cmd)
    ld.add_action(spawn_imu_broadcaster_cmd)
    ld.add_action(spawn_arm_controller_cmd)
    ld.add_action(spawn_gripper_controller_cmd)

    return ld
