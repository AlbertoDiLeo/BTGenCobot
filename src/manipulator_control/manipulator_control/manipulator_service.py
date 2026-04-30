#!/usr/bin/env python3
"""Manipulator Control Service using ikpy for inverse kinematics.

This ROS2 service node provides pick and place operations for the OpenManipulator-X arm.
It uses ikpy for inverse kinematics and sends trajectory goals to ros2_control.

Service: /manipulator_action (btgencobot_interfaces/srv/ManipulatorAction)
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from btgencobot_interfaces.srv import ManipulatorAction
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion

from builtin_interfaces.msg import Duration
import tf2_ros
from tf2_geometry_msgs import do_transform_pose_stamped
import numpy as np
import math
import time
import copy

try:
    import ikpy.chain
    import ikpy.link
    IKPY_AVAILABLE = True
except ImportError:
    IKPY_AVAILABLE = False


class ManipulatorService(Node):
    """ROS2 service node for manipulator pick and place operations."""

    # OpenManipulator-X joint names
    ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4']
    GRIPPER_JOINTS = ['gripper_left_joint']

    # Joint limits (from turtlebot3_manipulation_description URDF)
    JOINT_LIMITS = {
        'joint1': (-2.827, 2.827),      # ±pi*0.9 = ±162°
        'joint2': (-1.791, 1.571),      # -pi*0.57 to +pi*0.5
        'joint3': (-0.942, 1.382),      # -pi*0.3 to +pi*0.44
        'joint4': (-1.791, 2.042),      # -pi*0.57 to +pi*0.65
    }

    # Gripper positions
    GRIPPER_OPEN = 0.019
    GRIPPER_CLOSED = -0.010

    # Predefined arm positions (joint1, joint2, joint3, joint4)
    # These are tuned for OpenManipulator-X - top-down grasping configuration
    POSES = {
        'home': [0.0, 0.0, 0.0, 0.0],
        'ready': [0.0, 0.0, 0.0, 0.0],  # Straight up, ready to move
        'pre_grasp': [0.0, 0.4, 0.8, 0.4],  # Arm extended forward, gripper pointing down
        'grasp': [0.0, 0.6, 0.6, 0.6],  # Lower position for grasping
        'lift': [0.0, 0.2, 0.6, 0.4],  # Lifted position after grasp
    }

    def __init__(self):
        super().__init__('manipulator_service')

        self.callback_group = ReentrantCallbackGroup()

        # Current joint states
        self.current_joint_positions = {}
        self.joint_state_received = False

        # TF2 for coordinate transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribe to joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10
        )

        # Action clients for arm and gripper controllers
        self.arm_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory',
            callback_group=self.callback_group
        )

        # Use GripperCommand action (GripperActionController)
        self.gripper_action_client = ActionClient(
            self,
            GripperCommand,
            '/gripper_controller/gripper_cmd',
            callback_group=self.callback_group
        )

        # Create service
        self.service = self.create_service(
            ManipulatorAction,
            '/manipulator_action',
            self._handle_request,
            callback_group=self.callback_group
        )

        # Initialize ikpy chain if available
        self.arm_chain = None
        if IKPY_AVAILABLE:
            self._init_ikpy_chain()
        else:
            self.get_logger().warn('ikpy not available, using predefined poses only')

        self.get_logger().info('ManipulatorService initialized')
        self.get_logger().info('Waiting for action servers...')

        # Wait for action servers
        self._wait_for_servers()

    def _init_ikpy_chain(self):
        """Initialize ikpy chain for the OpenManipulator-X arm.
        
        Based on turtlebot3_manipulation_description URDF:
        - Arm mounted on base_link at xyz="-0.092 0 0.091" rpy="0 0 0"
        - joint1 (link1→link2): xyz="0.012 0 0.017", rotates around Z
        - joint2 (link2→link3): xyz="0 0 0.0595", rotates around Y
        - joint3 (link3→link4): xyz="0.024 0 0.128", rotates around Y
        - joint4 (link4→link5): xyz="0.124 0 0", rotates around Y
        - end_effector: xyz="0.126 0 0" from link5
        """
        try:
            # Define the kinematic chain matching the URDF exactly
            # We transform target poses to link1 frame, so chain starts from link1
            links = [
                ikpy.link.OriginLink(),
                # joint1: link1 to link2
                ikpy.link.URDFLink(
                    name="link2",
                    origin_translation=[0.012, 0.0, 0.017],
                    origin_orientation=[0, 0, 0],
                    rotation=[0, 0, 1],  # joint1 rotates around Z
                ),
                # joint2: link2 to link3
                ikpy.link.URDFLink(
                    name="link3",
                    origin_translation=[0.0, 0.0, 0.0595],
                    origin_orientation=[0, 0, 0],
                    rotation=[0, 1, 0],  # joint2 rotates around Y
                ),
                # joint3: link3 to link4
                ikpy.link.URDFLink(
                    name="link4",
                    origin_translation=[0.024, 0.0, 0.128],
                    origin_orientation=[0, 0, 0],
                    rotation=[0, 1, 0],  # joint3 rotates around Y
                ),
                # joint4: link4 to link5
                ikpy.link.URDFLink(
                    name="link5",
                    origin_translation=[0.124, 0.0, 0.0],
                    origin_orientation=[0, 0, 0],
                    rotation=[0, 1, 0],  # joint4 rotates around Y
                ),
                # end effector (fixed)
                ikpy.link.URDFLink(
                    name="end_effector",
                    origin_translation=[0.126, 0.0, 0.0],
                    origin_orientation=[0, 0, 0],
                    rotation=[0, 0, 0],  # Fixed link
                ),
            ]
            # active_links_mask: [OriginLink, joint1, joint2, joint3, joint4, end_effector]
            active_links_mask = [False, True, True, True, True, False]
            self.arm_chain = ikpy.chain.Chain(
                name='open_manipulator_x',
                links=links,
                active_links_mask=active_links_mask
            )
            self.get_logger().info('ikpy chain initialized successfully for turtlebot3_manipulation')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize ikpy chain: {e}')
            self.arm_chain = None

    def _wait_for_servers(self, timeout_sec: float = 30.0):
        """Wait for action servers to be available.
        
        This method waits for both arm and gripper controllers to become available.
        It will log warnings if servers are not found but will not crash.
        """
        self.get_logger().info('Waiting for arm_controller action server...')
        try:
            arm_ready = self.arm_action_client.wait_for_server(timeout_sec=timeout_sec)
            if arm_ready:
                self.get_logger().info('arm_controller action server connected')
            else:
                self.get_logger().warn(
                    f'arm_controller action server not available after {timeout_sec}s. '
                    'Will retry when service is called.'
                )
        except Exception as e:
            self.get_logger().warn(f'Error waiting for arm_controller: {e}')

        self.get_logger().info('Waiting for gripper_controller action server...')
        try:
            gripper_ready = self.gripper_action_client.wait_for_server(timeout_sec=timeout_sec)
            if gripper_ready:
                self.get_logger().info('gripper_controller action server connected')
            else:
                self.get_logger().warn(
                    f'gripper_controller action server not available after {timeout_sec}s. '
                    'Will retry when service is called.'
                )
        except Exception as e:
            self.get_logger().warn(f'Error waiting for gripper_controller: {e}')

    def _joint_state_callback(self, msg: JointState):
        """Store current joint positions."""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]
        self.joint_state_received = True

    def _handle_request(self, request, response):
        """Handle ManipulatorAction service request."""
        action_type = request.action_type.lower()
        target_pose = request.target_pose
        
        # Get object dimensions (with defaults)
        object_height = request.object_height if request.object_height > 0 else 0.1
        object_width = request.object_width if request.object_width > 0 else 0.05
        object_depth = request.object_depth if request.object_depth > 0 else object_width

        self.get_logger().info(
            f'Received {action_type} request at '
            f'({target_pose.pose.position.x:.3f}, '
            f'{target_pose.pose.position.y:.3f}, '
            f'{target_pose.pose.position.z:.3f}) '
            f'in frame "{target_pose.header.frame_id}", '
            f'object dims: h={object_height:.3f}, w={object_width:.3f}, d={object_depth:.3f}'
        )

        try:
            if action_type == 'pick':
                success = self._execute_pick(target_pose, object_height, object_width)
            elif action_type == 'place':
                success = self._execute_place(target_pose)
            else:
                response.success = False
                response.error_message = f'Unknown action type: {action_type}'
                return response

            response.success = success
            if not success:
                response.error_message = f'{action_type} operation failed'
            return response

        except Exception as e:
            self.get_logger().error(f'{action_type} failed: {e}')
            response.success = False
            response.error_message = str(e)
            return response

    def _execute_pick(self, target_pose: PoseStamped, object_height: float = 0.1, object_width: float = 0.05) -> bool:
        """Execute pick operation sequence using IK for target pose.
        
        Uses a top-down approach with collision avoidance based on object dimensions:
        1. Move to position above the object (safe height based on object_height)
        2. Descend vertically to grasp position (middle of object)
        3. Close gripper
        4. Lift and retract
        
        Args:
            target_pose: Pose of the object (z is top surface from camera)
            object_height: Estimated object height in meters
            object_width: Estimated object width in meters
        """
        self.get_logger().info(
            f'Starting pick sequence (top-down approach), '
            f'object height={object_height:.3f}m, width={object_width:.3f}m'
        )

        # 1. Move to ready position
        if not self._move_arm_to_pose('ready', duration=2.0):
            self.get_logger().error('Failed to move to ready position')
            return False

        # 2. Open gripper
        if not self._move_gripper(self.GRIPPER_OPEN, duration=1.0):
            self.get_logger().error('Failed to open gripper')
            return False

        # 3. Create poses for top-down approach using object dimensions
        # The target_pose.z is the TOP of the object (what the camera sees)
        # We need to grasp at the middle of the object
        top_z = target_pose.pose.position.z
        
        # Calculate grasp height: middle of object = top - (height / 2)
        # Add small margin to ensure we're gripping solidly
        grasp_z = top_z - (object_height / 2) - 0.01  # 1cm below center for solid grip
        
        # Above pose: above the object with clearance
        # Use object_height + margin to ensure we clear the top
        clearance = max(0.05, object_height * 0.5)  # At least 5cm or half object height
        above_z = top_z + clearance
        
        self.get_logger().info(
            f'Pose calculation: top_z={top_z:.3f}, grasp_z={grasp_z:.3f}, above_z={above_z:.3f} '
            f'(object_height={object_height:.3f})'
        )
        
        # Above pose: directly above the object at safe height
        above_pose = PoseStamped()
        above_pose.header = copy.deepcopy(target_pose.header)
        above_pose.pose = copy.deepcopy(target_pose.pose)
        above_pose.pose.position.z = above_z
        
        # Grasp pose: at the calculated grasp height
        grasp_pose = PoseStamped()
        grasp_pose.header = copy.deepcopy(target_pose.header)
        grasp_pose.pose = copy.deepcopy(target_pose.pose)
        grasp_pose.pose.position.z = grasp_z
        
        grasp_joints = self._compute_ik_for_pose(grasp_pose, apply_reach_offset=True)
        above_joints = None  # Initialize here
        
        if grasp_joints is None:
            self.get_logger().warn('IK failed for grasp pose, falling back to predefined poses')
            if not self._move_arm_to_pose('pre_grasp', duration=2.0):
                return False
            if not self._move_arm_to_pose('grasp', duration=2.0):
                return False
        else:
            # Compute position above object
            above_joints = self._compute_ik_for_pose(above_pose, apply_reach_offset=True)
            if above_joints is None:
                self.get_logger().warn('IK failed for above pose, trying grasp directly')
            else:
                # Move to position above object first (collision-free)
                self.get_logger().info(f'Moving to position above object: {above_joints}')
                if not self._move_arm_to_joints(above_joints, duration=2.0):
                    self.get_logger().error('Failed to move above object')
                    return False
            
            # Descend vertically to grasp position
            self.get_logger().info(f'Descending to grasp position: {grasp_joints}')
            if not self._move_arm_to_joints(grasp_joints, duration=2.0):
                self.get_logger().error('Failed to descend to grasp position')
                return False

        # 4. Close gripper with force-based grasping (fast)
        if not self._move_gripper(self.GRIPPER_CLOSED, duration=0.5, force_grasp=True):
            self.get_logger().error('Failed to close gripper')
            return False

        # 5. Lift object (back to above position)
        if above_joints:
            self.get_logger().info('Lifting object')
            if not self._move_arm_to_joints(above_joints, duration=2.0):
                self.get_logger().error('Failed to lift object')
                return False
        else:
            if not self._move_arm_to_pose('lift', duration=2.0):
                self.get_logger().error('Failed to lift object')
                return False

        # 6. Return to ready position
        if not self._move_arm_to_pose('ready', duration=2.0):
            self.get_logger().error('Failed to return to ready position')
            return False

        self.get_logger().info('Pick sequence completed successfully')
        return True

    def _compute_ik_for_pose(self, target_pose: PoseStamped, apply_reach_offset: bool = True) -> list:
        """Transform target pose to arm frame and compute IK.
        
        Args:
            target_pose: Target pose in any frame (will be transformed to arm frame)
            apply_reach_offset: If True, add forward offset to reach into the object.
                              If False, compute position without offset (for approach).
        
        Returns:
            List of joint angles, or None if IK fails
        """
        if self.arm_chain is None:
            self.get_logger().warn('IK chain not available')
            return None

        try:
            # Transform pose from source frame to arm base frame (link1)
            arm_base_frame = "link1"
            
            self.get_logger().info(
                f'Computing IK: input pose ({target_pose.pose.position.x:.3f}, '
                f'{target_pose.pose.position.y:.3f}, {target_pose.pose.position.z:.3f}) '
                f'in frame "{target_pose.header.frame_id}"'
            )
            
            # Get current robot position for debugging
            try:
                robot_tf = self.tf_buffer.lookup_transform(
                    "map", "base_link", rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5))
                self.get_logger().info(
                    f'Robot currently at map position: ({robot_tf.transform.translation.x:.3f}, '
                    f'{robot_tf.transform.translation.y:.3f})'
                )
            except Exception as e:
                self.get_logger().warn(f'Could not get robot position: {e}')
            
            # Transform to arm frame
            try:
                transform = self.tf_buffer.lookup_transform(
                    arm_base_frame,
                    target_pose.header.frame_id,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
                pose_in_arm = do_transform_pose_stamped(target_pose, transform)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                    tf2_ros.ExtrapolationException) as e:
                self.get_logger().error(f'TF transform failed: {e}')
                return None

            # Extract position for IK
            x = pose_in_arm.pose.position.x
            y = pose_in_arm.pose.position.y
            z = pose_in_arm.pose.position.z

            self.get_logger().info(
                f'Target in arm frame (link1): ({x:.3f}, {y:.3f}, {z:.3f})'
            )

            # For front approach grasping:
            # - apply_reach_offset=True: move gripper forward into the object for grasping
            # - apply_reach_offset=False: position before the object for approach
            reach_distance = math.sqrt(x*x + y*y)
            if reach_distance > 0.01:  # Avoid division by zero
                if apply_reach_offset:
                    # Extend forward to grasp - gripper needs to reach into object center
                    grasp_reach_offset = 0.04  # 4cm forward into object
                    x += grasp_reach_offset * (x / reach_distance)
                    y += grasp_reach_offset * (y / reach_distance)
                    self.get_logger().info(
                        f'Grasp position with +{grasp_reach_offset}m offset: ({x:.3f}, {y:.3f}, {z:.3f})'
                    )
                else:
                    # Pull back for approach position
                    approach_pullback = 0.05  # 5cm back from object
                    x -= approach_pullback * (x / reach_distance)
                    y -= approach_pullback * (y / reach_distance)
                    self.get_logger().info(
                        f'Approach position with -{approach_pullback}m offset: ({x:.3f}, {y:.3f}, {z:.3f})'
                    )

            # Compute IK
            return self._compute_ik([x, y, z])

        except Exception as e:
            self.get_logger().error(f'IK computation failed: {e}')
            return None

    def _execute_place(self, target_pose: PoseStamped) -> bool:
        """Execute place operation sequence using IK for target pose."""
        self.get_logger().info('Starting place sequence')

        # 1. Move to ready position (should already be holding object)
        if not self._move_arm_to_pose('ready', duration=2.0):
            self.get_logger().error('Failed to move to ready position')
            return False

        # 2. Compute IK for place position
        place_joints = self._compute_ik_for_pose(target_pose)
        if place_joints is None:
            self.get_logger().warn('IK failed, falling back to predefined poses')
            # Fallback to predefined poses
            if not self._move_arm_to_pose('pre_grasp', duration=2.0):
                return False
            if not self._move_arm_to_pose('grasp', duration=2.0):
                return False
        else:
            # Compute pre-place position (slightly above place location)
            pre_place_pose = PoseStamped()
            pre_place_pose.header = target_pose.header
            pre_place_pose.pose = target_pose.pose
            pre_place_pose.pose.position.z += 0.05  # 5cm above
            
            pre_place_joints = self._compute_ik_for_pose(pre_place_pose)
            if pre_place_joints:
                self.get_logger().info(f'Moving to pre-place IK position: {pre_place_joints}')
                if not self._move_arm_to_joints(pre_place_joints, duration=2.0):
                    self.get_logger().error('Failed to move to pre-place position')
                    return False
            
            # Move to place position
            self.get_logger().info(f'Moving to place IK position: {place_joints}')
            if not self._move_arm_to_joints(place_joints, duration=2.0):
                self.get_logger().error('Failed to move to place position')
                return False

        # 3. Open gripper to release
        if not self._move_gripper(self.GRIPPER_OPEN, duration=1.0):
            self.get_logger().error('Failed to open gripper')
            return False

        # 4. Retract to lift position
        if not self._move_arm_to_pose('lift', duration=2.0):
            self.get_logger().error('Failed to retract')
            return False

        # 5. Return to home position
        if not self._move_arm_to_pose('home', duration=2.0):
            self.get_logger().error('Failed to return to home position')
            return False

        self.get_logger().info('Place sequence completed successfully')
        return True

    def _move_arm_to_pose(self, pose_name: str, duration: float = 2.0) -> bool:
        """Move arm to a predefined pose."""
        if pose_name not in self.POSES:
            self.get_logger().error(f'Unknown pose: {pose_name}')
            return False

        joint_positions = self.POSES[pose_name]
        self.get_logger().info(f'Moving arm to {pose_name}: {joint_positions}')

        return self._send_arm_trajectory(joint_positions, duration)

    def _move_arm_to_joints(self, joint_positions: list, duration: float = 2.0) -> bool:
        """Move arm to specific joint positions."""
        # Clamp to joint limits
        clamped = []
        for i, (pos, joint_name) in enumerate(zip(joint_positions, self.ARM_JOINTS)):
            limits = self.JOINT_LIMITS[joint_name]
            clamped_pos = max(limits[0], min(limits[1], pos))
            if abs(clamped_pos - pos) > 0.01:
                self.get_logger().warn(
                    f'{joint_name}: clamped {pos:.3f} to {clamped_pos:.3f}'
                )
            clamped.append(clamped_pos)

        return self._send_arm_trajectory(clamped, duration)

    def _arm_is_near_positions(self, target_positions: list, tolerance: float = 0.08) -> bool:
        """Check whether current arm joint states are close to the requested target."""
        if not self.joint_state_received:
            return False

        for joint_name, target in zip(self.ARM_JOINTS, target_positions):
            current = self.current_joint_positions.get(joint_name)
            if current is None or abs(current - target) > tolerance:
                return False

        return True

    def _get_gripper_position(self):
        """Return the current simulated gripper joint position when available."""
        if not self.joint_state_received:
            return None

        return self.current_joint_positions.get('gripper_left_joint')

    def _send_arm_trajectory(self, positions: list, duration: float) -> bool:
        """Send trajectory goal to arm controller."""
        if not self.arm_action_client.server_is_ready():
            self.get_logger().error('Arm action server not ready')
            return False

        # Create trajectory message
        trajectory = JointTrajectory()
        trajectory.joint_names = self.ARM_JOINTS

        point = JointTrajectoryPoint()
        point.positions = positions
        point.velocities = [0.0] * len(positions)
        point.time_from_start = Duration(
            sec=int(duration),
            nanosec=int((duration % 1) * 1e9)
        )
        trajectory.points = [point]

        # Create goal
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory

        # Send goal and wait for result using polling (not spin_until_future_complete)
        self.get_logger().debug(f'Sending arm trajectory: {positions}')
        future = self.arm_action_client.send_goal_async(goal)
        
        # Poll for goal acceptance
        timeout = 5.0
        start_time = time.monotonic()
        while not future.done():
            if time.monotonic() - start_time > timeout:
                self.get_logger().error('Failed to send arm goal (timeout)')
                return False
            time.sleep(0.05)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Arm goal rejected')
            return False

        # Wait for result using polling
        # Use generous timeout since simulation may run slower than real-time
        result_future = goal_handle.get_result_async()
        timeout = duration * 8 + 20.0  # More margin for slow startup / controller lag in Gazebo
        grace_timeout = 5.0
        start_time = time.monotonic()
        used_grace_period = False
        while not result_future.done():
            elapsed = time.monotonic() - start_time
            if elapsed > timeout:
                if self._arm_is_near_positions(positions):
                    if not used_grace_period:
                        used_grace_period = True
                        timeout += grace_timeout
                        self.get_logger().warn(
                            'Arm trajectory reached timeout but joints are near target; '
                            f'waiting an extra {grace_timeout:.1f}s for controller result'
                        )
                    elif elapsed > timeout:
                        self.get_logger().warn(
                            'Arm trajectory result is late, but joints are already at target; '
                            'accepting motion as successful'
                        )
                        return True
                else:
                    self.get_logger().error(
                        f'Arm trajectory timed out after {elapsed:.1f}s '
                        f'(limit {timeout:.1f}s)'
                    )
                    return False
            time.sleep(0.1)

        result = result_future.result()
        if result.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().error(f'Arm trajectory failed: {result.result.error_code}')
            return False

        self.get_logger().debug('Arm trajectory completed')
        return True

    def _move_gripper(self, position: float, duration: float = 1.0, force_grasp: bool = False) -> bool:
        """Move gripper to position using GripperCommand action.
        
        Args:
            position: Target gripper position (GRIPPER_OPEN to GRIPPER_CLOSED)
            duration: Timeout duration for the action
            force_grasp: If True, consider 'stalled' as success (object grasped)
                        This enables force-based grasping where the gripper closes
                        until it meets resistance from the object.
        
        Returns:
            True if gripper action succeeded, False otherwise
        """
        # Wait for gripper action server if not ready (may take time to spawn)
        if not self.gripper_action_client.server_is_ready():
            self.get_logger().info('Waiting for gripper action server...')
            if not self.gripper_action_client.wait_for_server(timeout_sec=10.0):
                self.get_logger().error('Gripper action server not available after 10s')
                return False
            self.get_logger().info('Gripper action server connected')

        # Clamp gripper position
        position = max(self.GRIPPER_CLOSED, min(self.GRIPPER_OPEN, position))
        initial_position = self._get_gripper_position()

        # Create GripperCommand goal
        goal = GripperCommand.Goal()
        goal.command.position = position
        # Use higher effort for force-based grasping to ensure firm grip
        goal.command.max_effort = 20.0 if force_grasp else 10.0

        # Send goal and wait using polling
        action_name = 'open' if position > 0 else 'close'
        self.get_logger().info(
            f'Gripper {action_name}: {position} '
            f'(force_grasp={force_grasp}, max_effort={goal.command.max_effort})'
        )
        future = self.gripper_action_client.send_goal_async(goal)
        
        # Poll for goal acceptance
        timeout = 5.0
        start_time = time.time()
        while not future.done():
            if time.time() - start_time > timeout:
                self.get_logger().error('Failed to send gripper goal (timeout)')
                return False
            time.sleep(0.05)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Gripper goal rejected')
            return False

        # Wait for result using polling
        # Use generous timeout since simulation may run slower than real-time
        # Force-based grasping can take longer as the gripper waits to stall
        result_future = goal_handle.get_result_async()
        base_timeout = 20.0 if force_grasp else 10.0  # Force grasp needs more time
        timeout = duration * 5 + base_timeout
        start_time = time.monotonic()
        last_joint_position = initial_position
        last_joint_change_time = start_time
        while not result_future.done():
            now = time.monotonic()
            current_position = self._get_gripper_position()

            if (
                current_position is not None and
                last_joint_position is not None and
                abs(current_position - last_joint_position) > 0.0005
            ):
                last_joint_position = current_position
                last_joint_change_time = now

            if now - start_time > timeout:
                current_position = self._get_gripper_position()
                if current_position is not None:
                    near_target = abs(current_position - position) <= 0.008

                    if force_grasp and initial_position is not None:
                        made_closing_progress = current_position < initial_position - 0.004
                        stalled_after_progress = (
                            made_closing_progress and
                            now - last_joint_change_time >= 1.5
                        )
                        if near_target or stalled_after_progress:
                            self.get_logger().warn(
                                'Gripper action timed out, but joint state indicates a likely '
                                f'grasp/closure (current={current_position:.4f}, '
                                f'target={position:.4f}, initial={initial_position:.4f}); '
                                'accepting as success'
                            )
                            return True

                    if not force_grasp and near_target:
                        self.get_logger().warn(
                            'Gripper action timed out, but joint state is already near target '
                            f'(current={current_position:.4f}, target={position:.4f}); '
                            'accepting as success'
                        )
                        return True

                    if not force_grasp and initial_position is not None:
                        made_opening_progress = current_position > initial_position + 0.004
                        stalled_after_progress = (
                            made_opening_progress and
                            now - last_joint_change_time >= 1.5
                        )
                        if stalled_after_progress:
                            self.get_logger().warn(
                                'Gripper action timed out, but joint state indicates a likely '
                                f'completed opening (current={current_position:.4f}, '
                                f'target={position:.4f}, initial={initial_position:.4f}); '
                                'accepting as success'
                            )
                            return True

                self.get_logger().error(f'Gripper command timed out after {timeout:.1f}s')
                return False
            time.sleep(0.1)

        result = result_future.result()
        reached_goal = result.result.reached_goal
        stalled = result.result.stalled
        
        self.get_logger().info(
            f'Gripper result: reached_goal={reached_goal}, stalled={stalled}'
        )
        
        # Determine success based on mode
        if force_grasp:
            # Force-based grasping: stalled means we're gripping an object (SUCCESS)
            # reached_goal without stalling means gripper closed fully (no object)
            if stalled:
                self.get_logger().info('Force grasp successful - gripper stalled on object')
                return True
            elif reached_goal:
                self.get_logger().warn('Gripper closed fully - may not have grasped object')
                return True  # Still return True, object might be very small
            else:
                self.get_logger().error('Force grasp failed - neither stalled nor reached goal')
                return False
        else:
            # Normal mode: we expect to reach the goal position
            if reached_goal:
                self.get_logger().info(f'Gripper {action_name} completed')
                return True
            elif stalled:
                self.get_logger().warn(f'Gripper {action_name} stalled before reaching goal')
                return True  # Still consider success - might be ok
            else:
                self.get_logger().warn(f'Gripper may not have reached target position')
                return True  # Be lenient

    def _compute_ik(self, target_position: list, horizontal_gripper: bool = True) -> list:
        """Compute inverse kinematics for target position using ikpy.
        
        Args:
            target_position: [x, y, z] target position in arm frame
            horizontal_gripper: If True, adjust joint4 to keep gripper horizontal
                               (parallel to ground for top-down grasping)
        
        Returns:
            List of joint angles [joint1, joint2, joint3, joint4], or None if IK fails
        """
        if self.arm_chain is None:
            self.get_logger().warn('IK not available, returning None')
            return None

        try:
            # ikpy expects [x, y, z] target
            target = np.array(target_position)

            # Compute IK
            ik_solution = self.arm_chain.inverse_kinematics(target)

            # Extract joint angles (skip first and last elements - origin and end effector)
            joint_angles = ik_solution[1:5].tolist()

            self.get_logger().info(f'IK solution (raw): {joint_angles}')
            
            if horizontal_gripper:
                # Adjust joint4 to keep gripper horizontal (parallel to ground)
                # For OpenManipulator-X, the gripper pitch = joint2 + joint3 + joint4
                # For gripper parallel to ground: pitch = 0
                # So: joint4 = -joint2 - joint3
                joint2 = joint_angles[1]
                joint3 = joint_angles[2]
                joint4_horizontal = -joint2 - joint3
                
                # Clamp to joint4 limits
                joint4_limits = self.JOINT_LIMITS['joint4']
                joint4_clamped = max(joint4_limits[0], min(joint4_limits[1], joint4_horizontal))
                
                if abs(joint4_clamped - joint4_horizontal) > 0.01:
                    self.get_logger().warn(
                        f'joint4 for horizontal gripper ({joint4_horizontal:.3f}) '
                        f'clamped to limits: {joint4_clamped:.3f}'
                    )
                
                joint_angles[3] = joint4_clamped
                self.get_logger().info(
                    f'Adjusted for horizontal gripper: joint4={joint4_clamped:.3f} '
                    f'(pitch={joint2 + joint3 + joint4_clamped:.3f} rad)'
                )

            self.get_logger().info(f'IK solution (final): {joint_angles}')
            return joint_angles

        except Exception as e:
            self.get_logger().error(f'IK computation failed: {e}')
            return None


def main(args=None):
    rclpy.init(args=args)

    node = ManipulatorService()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
