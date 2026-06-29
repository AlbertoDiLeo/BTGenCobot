"""ROS2 Action Server for BehaviorTree Generation and Execution"""
import json
import math
import re
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional


CURATED_ROOM_ROUTES = {
    "living room": [
        {"x": 0.35, "y": 0.05, "frame_id": "map"},
    ],
    "kitchen": [
        {"x": 1.7, "y": 0.05, "frame_id": "map"},
        {"x": 3.35, "y": 0.0, "frame_id": "map"},
        {"x": 4.85, "y": -0.05, "frame_id": "map"},
    ],
    "bedroom": [
        {"x": -1.95, "y": 0.16, "frame_id": "map"},
        {"x": -3.35, "y": 0.48, "frame_id": "map"},
        {"x": -4.55, "y": 0.85, "frame_id": "map"},
    ],
    "charging station": [
        {"x": 0.35, "y": 0.05, "frame_id": "map"},
    ],
}

ROOM_ALIASES = {
    "living room": "living room",
    "living-room": "living room",
    "lounge": "living room",
    "kitchen": "kitchen",
    "bedroom": "bedroom",
    "charging station": "charging station",
    "charging-station": "charging station",
    "dock": "charging station",
    "docking area": "charging station",
}

import requests
import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy

from btgencobot_interfaces.action import GenerateAndExecuteBT
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from std_srvs.srv import Trigger


NAV_STATUS_NAMES = {
    1: 'UNKNOWN', 2: 'ACCEPTED', 3: 'EXECUTING',
    4: 'SUCCEEDED', 5: 'CANCELED', 6: 'ABORTED'
}

def _escape_xml_attr(value) -> str:
    return (
        str(value)
        .replace('&', '&amp;')
        .replace('"', '&quot;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
    )


class BTInterfaceNode(Node):
    """ROS2 Action Server for generating BehaviorTrees from natural language and executing them via Nav2"""

    def __init__(self):
        super().__init__('bt_interface_node')
        self._declare_parameters()
        self._initialize_state()
        self._setup_interfaces()
        self._log_configuration()

    def _declare_parameters(self):
        """Declare and load ROS parameters"""
        self.declare_parameter('inference_server_url', 'http://host.docker.internal:8080')
        self.declare_parameter('bt_output_dir', '/workspace/generated_bts')
        self.declare_parameter('generation_timeout', 30.0)
        self.declare_parameter('execution_timeout', 300.0)
        self.declare_parameter('feedback_rate', 2.0)
        self.declare_parameter('nav_action', '/navigate_to_pose')
        self.declare_parameter('nav_server_wait_timeout', 90.0)

        self.inference_url = self.get_parameter('inference_server_url').value
        self.bt_output_dir = Path(self.get_parameter('bt_output_dir').value)
        self.generation_timeout = self.get_parameter('generation_timeout').value
        self.execution_timeout = self.get_parameter('execution_timeout').value
        self.feedback_rate = self.get_parameter('feedback_rate').value
        self.nav_action_name = self.get_parameter('nav_action').value
        self.nav_server_wait_timeout = self.get_parameter('nav_server_wait_timeout').value

        self.bt_output_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_state(self):
        """Initialize state variables"""
        self.current_goal_handle = None
        self.current_nav_goal_handle = None
        self.active_client_goal_handle = None
        self.is_executing = False
        self.last_bt_xml = None

    def _setup_interfaces(self):
        """Setup ROS interfaces: publishers, subscribers, action servers/clients, services"""
        self.action_callback_group = ReentrantCallbackGroup()

        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self._bt_xml_publisher = self.create_publisher(String, '/generated_behavior_tree', qos_latched)
        self._bt_execution_feedback_publisher = self.create_publisher(
            String,
            '/bt_execution_feedback',
            10,
        )
        self._cmd_vel_nav_publisher = self.create_publisher(Twist, '/cmd_vel_nav', 10)
        self._cmd_vel_smoothed_publisher = self.create_publisher(Twist, '/cmd_vel_smoothed', 10)
        self._cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.bt_republish_timer = self.create_timer(2.0, self._republish_last_bt)

        self._action_server = ActionServer(
            self, GenerateAndExecuteBT, '/generate_and_execute_bt',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.action_callback_group
        )

        self._nav_client = ActionClient(
            self, NavigateToPose, self.nav_action_name,
            callback_group=self.action_callback_group
        )

        self._self_client = ActionClient(
            self, GenerateAndExecuteBT, '/generate_and_execute_bt',
            callback_group=self.action_callback_group
        )

        self._emergency_stop_srv = self.create_service(
            Trigger, '/emergency_stop_bt', self.emergency_stop_callback
        )

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self._command_subscriber = self.create_subscription(
            String, '/btgen_nl_command', self.command_topic_callback, qos_profile
        )

    def _log_configuration(self):
        self.get_logger().info(f'BT output directory: {self.bt_output_dir}')
        self.get_logger().info(f'Inference server URL: {self.inference_url}')
        self.get_logger().info(f'Navigation action target: {self.nav_action_name}')
        self.get_logger().info('BT Interface Node initialized')

    def goal_callback(self, goal_request):
        self.get_logger().info(f'Received goal request: {goal_request.command}')
        if self.is_executing:
            self.get_logger().warn('Already executing — rejecting concurrent goal')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.current_goal_handle = goal_handle
        self.is_executing = True

        command = goal_handle.request.command
        self.get_logger().info(f'Command: {command}')

        result = GenerateAndExecuteBT.Result()
        result.success = False
        result.bt_xml_path = ''
        result.error_message = ''

        try:
            self.publish_feedback(goal_handle, 'generating', 0.1, 'Calling inference server...')
            bt_xml, error = await self.generate_bt_from_command(command)
            
            # Uscita pulita se cancellato durante la generazione
            if goal_handle.is_cancel_requested:
                self.get_logger().info("Esecuzione interrotta per nuovo comando (preemption).")
                goal_handle.canceled()
                result.success = True
                result.error_message = 'Task preempted gracefully'
                return result

            if bt_xml is None:
                result.error_message = f'BT generation failed: {error}'
                self.get_logger().error(result.error_message)
                return result

            self.publish_feedback(goal_handle, 'validating', 0.3, 'Validating generated BT...')
            is_valid, val_error = self.validate_bt_xml(bt_xml)
            if not is_valid:
                result.error_message = f'BT validation failed: {val_error}'
                self.get_logger().error(result.error_message)
                return result

            self.publish_feedback(goal_handle, 'validating', 0.4, 'Writing BT to file...')
            bt_file_path = self.write_bt_file(bt_xml)
            result.bt_xml_path = str(bt_file_path)

            self.last_bt_xml = bt_xml
            bt_msg = String()
            bt_msg.data = self.add_uids_for_foxglove(bt_xml)
            self._bt_xml_publisher.publish(bt_msg)

            self.publish_feedback(goal_handle, 'executing', 0.5, 'Executing BehaviorTree...')
            execution_success, exec_error = await self.execute_bt(bt_file_path, goal_handle)
            
            # --- MODIFICA CHIAVE PER IL FRONTEND ---
            # Se è stato cancellato dall'utente (preemption), chiudiamo con successo per non far crashare la UI
            if goal_handle.is_cancel_requested or exec_error == 'PREEMPTED':
                self.get_logger().info('Task sostituito con successo dal nuovo comando.')
                result.success = True
                result.error_message = 'Task preempted by new command'
                return result

            if not execution_success:
                result.error_message = f'BT execution failed: {exec_error}'
                self.get_logger().error(result.error_message)
                return result

            self.get_logger().info('BT execution completed successfully')
            self.publish_feedback(goal_handle, 'completed', 1.0, 'BT execution completed')

            result.success = True
            goal_handle.succeed()

        except Exception as e:
            result.error_message = f'Unexpected error: {str(e)}'
            self.get_logger().error(result.error_message)
            goal_handle.abort()

        finally:
            if self.current_goal_handle == goal_handle:
                self.is_executing = False
                self.current_goal_handle = None

        return result

    def _normalize_room_label(self, room_label: str) -> str:
        normalized = re.sub(r'\s+', ' ', room_label.strip().lower())
        return ROOM_ALIASES.get(normalized, normalized)

    def _extract_curated_room(self, command: str) -> Optional[str]:
        normalized_command = re.sub(r'\s+', ' ', command.strip().lower())
        match = re.search(r'\bgo to(?: the)? ([a-z\- ]+?)(?: and | then |$)', normalized_command)
        if not match:
            return None
        return self._normalize_room_label(match.group(1))

    def _build_curated_room_bt(self, room_name: str, command: str) -> str:
        route = CURATED_ROOM_ROUTES[room_name]
        include_wait = 'wait for further instructions' in command.lower()

        sequence_lines = ['      <Sequence name="RoomNavigation">']
        for index, pose in enumerate(route, start=1):
            pose_value = self._pose_to_bt_string(pose)
            path_key = f'{{path_{index}}}'
            sequence_lines.append(
                f'        <ComputePathToPose goal="{pose_value}" path="{path_key}" planner_id="GridBased"/>'
            )
            sequence_lines.append(
                f'        <FollowPath path="{path_key}" controller_id="FollowPath"/>'
            )

        if include_wait:
            sequence_lines.append('        <Wait wait_duration="3.0"/>')

        sequence_lines.append('      </Sequence>')

        xml_lines = [
            '<root BTCPP_format="4" main_tree_to_execute="MainTree">',
            '  <BehaviorTree ID="MainTree">',
            *sequence_lines,
            '  </BehaviorTree>',
            '</root>',
        ]
        return '\n'.join(xml_lines)

    def _parse_semantic_navigation_command(self, command: str) -> Optional[dict]:
        try:
            envelope = json.loads(command)
        except json.JSONDecodeError:
            return None

        if not isinstance(envelope, dict):
            return None
        if envelope.get('type') != 'semantic_navigation':
            return None

        semantic = envelope.get('semantic_navigation')
        if not isinstance(semantic, dict):
            raise ValueError('semantic_navigation envelope missing semantic_navigation object')

        waypoints = semantic.get('waypoints')
        if not isinstance(waypoints, list) or len(waypoints) == 0:
            raise ValueError('semantic_navigation request must include at least one waypoint')

        for index, waypoint in enumerate(waypoints, start=1):
            if not isinstance(waypoint, dict):
                raise ValueError(f'Waypoint {index} is not an object')
            if 'x' not in waypoint or 'y' not in waypoint:
                raise ValueError(f'Waypoint {index} must include x and y')

        return {
            'command': envelope.get('command') or f"go to {semantic.get('destination_label', 'destination')}",
            'semantic_navigation': semantic,
        }

    def _semantic_waypoint_to_pose(self, waypoint: dict) -> dict:
        return {
            'x': float(waypoint['x']),
            'y': float(waypoint['y']),
            'frame_id': waypoint.get('frame_id') or waypoint.get('frameId') or 'map',
            'yaw_rad': waypoint.get('yaw_rad', waypoint.get('yawRad')),
        }

    def _pose_to_bt_string(self, pose: dict) -> str:
        yaw_rad = float(pose.get('yaw_rad') or 0.0)
        # Nav2 pose strings use quaternion x;y;z;w after position. For planar
        # room navigation only yaw is relevant.
        half_yaw = yaw_rad / 2.0
        z = math.sin(half_yaw)
        w = math.cos(half_yaw)
        return f'0;{pose["frame_id"]};{pose["x"]};{pose["y"]};0;0;0;{z};{w}'

    def _build_semantic_navigation_rewritten_input(self, semantic: dict) -> str:
        destination_label = semantic.get('destination_label') or semantic.get('destination_id') or 'destination'
        start_node_id = semantic.get('start_node_id', 'unknown')
        target_node_id = semantic.get('target_node_id', 'unknown')
        topological_path = semantic.get('topological_path') or []
        waypoints = [
            self._semantic_waypoint_to_pose(waypoint)
            for waypoint in semantic.get('waypoints', [])
        ]

        waypoint_lines = []
        for index, waypoint in enumerate(waypoints, start=1):
            pose_value = self._pose_to_bt_string(waypoint)
            waypoint_lines.append(
                f'- waypoint_{index}: goal="{pose_value}" path="{{path_{index}}}"'
            )

        return '\n'.join([
            'Structure: Sequence',
            'Actions: ComputePathToPose, FollowPath',
            f'Task: Navigate to the resolved semantic destination "{destination_label}".',
            f'Route source: {semantic.get("planner", "external-route-planner")}.',
            f'Topological route: {" -> ".join(topological_path)}.',
            f'Start node: {start_node_id}.',
            f'Target node: {target_node_id}.',
            'Use only the metric waypoints below. Do not invent rooms, goals, coordinates, object actions, recovery actions, or additional navigation targets.',
            'For each waypoint, generate a ComputePathToPose action immediately followed by a FollowPath action using the same path key.',
            *waypoint_lines,
        ])

    def _build_semantic_navigation_instruction(self) -> str:
        return (
            'Generate a BehaviorTree XML for a resolved room-navigation task. '
            'The route has already been computed by an external topological route planner. '
            'Use only ComputePathToPose and FollowPath actions. '
            'For every provided waypoint, first compute a path to the exact goal string, '
            'then follow that path. Copy each goal attribute exactly as provided, '
            'including the leading timestamp/frame fields such as 0;map;x;y;0;0;0;z;w. '
            'Do not shorten, normalize, or reinterpret goal strings. '
            'Do not add DetectObject, PickObject, PlaceObject, '
            'Spin, BackUp, Wait, conditions, extra goals, or recovery branches. '
            'Output only valid XML with BTCPP_format="4".'
        )

    def _semantic_navigation_expected_goals(self, semantic: dict) -> list[str]:
        return [
            self._pose_to_bt_string(self._semantic_waypoint_to_pose(waypoint))
            for waypoint in semantic.get('waypoints', [])
        ]

    def _get_bt_node_id(self, element: ET.Element) -> str:
        return element.tag if element.tag != 'Action' else element.get('ID', element.tag)

    def _validate_semantic_navigation_bt(self, bt_xml: str, semantic: dict) -> tuple[bool, Optional[str]]:
        try:
            root = ET.fromstring(bt_xml)
        except ET.ParseError as e:
            return False, f'Invalid XML: {e}'

        expected_goals = self._semantic_navigation_expected_goals(semantic)
        compute_nodes = [
            element
            for element in root.iter()
            if self._get_bt_node_id(element) == 'ComputePathToPose'
        ]
        follow_nodes = [
            element
            for element in root.iter()
            if self._get_bt_node_id(element) == 'FollowPath'
        ]

        if len(compute_nodes) != len(expected_goals):
            return False, (
                f'Expected {len(expected_goals)} ComputePathToPose node(s), '
                f'got {len(compute_nodes)}'
            )
        if len(follow_nodes) < len(expected_goals):
            return False, (
                f'Expected at least {len(expected_goals)} FollowPath node(s), '
                f'got {len(follow_nodes)}'
            )

        actual_goals = [node.get('goal') for node in compute_nodes]
        for index, (actual_goal, expected_goal) in enumerate(zip(actual_goals, expected_goals), start=1):
            if actual_goal != expected_goal:
                return False, (
                    f'Waypoint {index} goal mismatch. '
                    f'Expected "{expected_goal}", got "{actual_goal}"'
                )

        return True, None

    def _build_resolved_route_navigation_bt(self, semantic: dict) -> str:
        destination_label = _escape_xml_attr(
            semantic.get('destination_label') or semantic.get('destination_id') or 'Destination'
        )
        waypoints = [
            self._semantic_waypoint_to_pose(waypoint)
            for waypoint in semantic.get('waypoints', [])
        ]

        sequence_lines = [f'      <Sequence name="Navigate to {destination_label}">']
        multiple_waypoints = len(waypoints) > 1
        for index, pose in enumerate(waypoints, start=1):
            pose_value = _escape_xml_attr(self._pose_to_bt_string(pose))
            path_key = f'{{path_{index}}}'
            plan_name = (
                f'Plan path to {destination_label} waypoint {index}'
                if multiple_waypoints
                else f'Plan path to {destination_label}'
            )
            move_name = (
                f'Move to {destination_label} waypoint {index}'
                if multiple_waypoints
                else f'Move to {destination_label}'
            )
            sequence_lines.append(
                f'        <ComputePathToPose name="{plan_name}" goal="{pose_value}" path="{path_key}" planner_id="GridBased"/>'
            )
            sequence_lines.append(
                f'        <FollowPath name="{move_name}" path="{path_key}" controller_id="FollowPath"/>'
            )
        sequence_lines.append('      </Sequence>')

        xml_lines = [
            '<root BTCPP_format="4" main_tree_to_execute="MainTree">',
            '  <BehaviorTree ID="MainTree">',
            *sequence_lines,
            '  </BehaviorTree>',
            '</root>',
        ]
        return '\n'.join(xml_lines)

    async def generate_bt_from_semantic_navigation(self, payload: dict) -> tuple[Optional[str], Optional[str]]:
        semantic = payload['semantic_navigation']
        command = payload['command']
        rewritten_input = self._build_semantic_navigation_rewritten_input(semantic)
        custom_instruction = self._build_semantic_navigation_instruction()

        self.get_logger().info(
            'Generating semantic navigation BT through inference server: '
            f'{semantic.get("start_node_id")} -> {semantic.get("target_node_id")} '
            f'({len(semantic.get("waypoints", []))} waypoint(s))'
        )

        try:
            response = requests.post(
                f'{self.inference_url}/generate_bt',
                json={
                    'command': command,
                    'max_tokens': 1024,
                    'temperature': 0.1,
                    'prompt_format': 'alpaca',
                    'use_query_rewriting': False,
                    'rewritten_input': rewritten_input,
                    'custom_instruction': custom_instruction,
                },
                timeout=self.generation_timeout
            )
            if response.status_code == 200:
                data = response.json()
                bt_xml = data.get('bt_xml')
                if data.get('success', False) and bt_xml:
                    is_valid_semantic_bt, validation_error = self._validate_semantic_navigation_bt(bt_xml, semantic)
                    if not is_valid_semantic_bt:
                        self.get_logger().warn(
                            'Semantic navigation LLM output is not executable; '
                            'compiling resolved route in backend: '
                            f'{validation_error}'
                        )
                        return self._build_resolved_route_navigation_bt(semantic), None

                    self.get_logger().info('Semantic navigation BT generated by inference server')
                    return bt_xml, None

                self.get_logger().warn(
                    f'Semantic navigation LLM generation failed; '
                    f'compiling resolved route in backend: '
                    f'{data.get("error", "Unknown error")}'
                )
            else:
                self.get_logger().warn(
                    f'Semantic navigation inference server HTTP {response.status_code}, '
                    'compiling resolved route in backend'
                )
        except requests.Timeout:
            self.get_logger().warn(
                'Semantic navigation inference server timeout; '
                'compiling resolved route in backend'
            )
        except requests.ConnectionError:
            self.get_logger().warn(
                'Semantic navigation inference server unavailable; '
                'compiling resolved route in backend'
            )
        except Exception as e:
            self.get_logger().warn(
                f'Semantic navigation generation error; '
                f'compiling resolved route in backend: {e}'
            )

        return self._build_resolved_route_navigation_bt(semantic), None

    async def generate_bt_from_command(self, command: str) -> tuple[Optional[str], Optional[str]]:
        try:
            semantic_payload = self._parse_semantic_navigation_command(command)
        except ValueError as e:
            return None, str(e)

        if semantic_payload:
            return await self.generate_bt_from_semantic_navigation(semantic_payload)

        curated_room = self._extract_curated_room(command)
        if curated_room in CURATED_ROOM_ROUTES:
            self.get_logger().info(f'Using curated room navigation BT for: {curated_room}')
            return self._build_curated_room_bt(curated_room, command), None

        try:
            response = requests.post(
                f'{self.inference_url}/generate_bt',
                json={
                    'command': command,
                    'max_tokens': 1024,
                    'temperature': 0.1,
                    'prompt_format': 'alpaca',
                    'use_query_rewriting': True
                },
                timeout=self.generation_timeout
            )
            if response.status_code != 200:
                return None, f'HTTP {response.status_code}: {response.text}'
            data = response.json()
            if not data.get('success', False):
                return None, data.get('error', 'Unknown error')
            return data.get('bt_xml'), None
        except requests.Timeout:
            return None, 'Inference server timeout'
        except requests.ConnectionError:
            return None, 'Could not connect to inference server'
        except Exception as e:
            return None, f'Request failed: {str(e)}'

    def validate_bt_xml(self, xml_string: str) -> tuple[bool, Optional[str]]:
        try:
            root = ET.fromstring(xml_string)
            if root.tag != 'root': return False, f"Root element must be 'root', got '{root.tag}'"
            if root.get('BTCPP_format') != '4': return False, "BTCPP_format must be '4'"
            if len(root.findall('BehaviorTree')) == 0: return False, 'No BehaviorTree element found'
            return True, None
        except Exception as e:
            return False, f'Validation error: {str(e)}'

    def add_uids_for_foxglove(self, xml_string: str) -> str:
        try:
            root = ET.fromstring(xml_string)
            node_counts = {}

            def add_uid_recursive(element):
                tag = element.tag
                if tag == 'root': uid = 'root'
                elif tag == 'BehaviorTree': uid = f"BehaviorTree_{element.get('ID', 'Tree')}"
                else:
                    name = element.get('name') or element.get('ID') or tag
                    base = f'{tag}_{name}'.replace(' ', '_').replace('"', '').replace("'", '')
                    node_counts[base] = node_counts.get(base, 0) + 1
                    uid = f'{base}_{node_counts[base]}' if node_counts[base] > 1 else base
                element.set('_uid', uid)
                for child in element: add_uid_recursive(child)

            add_uid_recursive(root)
            return ET.tostring(root, encoding='unicode', xml_declaration=True)
        except ET.ParseError:
            return xml_string

    def write_bt_file(self, xml_content: str) -> Path:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = self.bt_output_dir / f'bt_{timestamp}_{uuid.uuid4().hex[:8]}.xml'
        with open(file_path, 'w') as f: f.write(xml_content)
        file_path.chmod(0o644)
        return file_path

    def _extract_final_nav_pose_from_bt(self, bt_file_path: Path) -> Optional[dict]:
        try:
            root = ET.parse(bt_file_path).getroot()
        except Exception as e:
            self.get_logger().warn(f'Unable to parse BT goal for action pose: {e}')
            return None

        compute_goals = [
            element.get('goal')
            for element in root.iter()
            if self._get_bt_node_id(element) == 'ComputePathToPose'
        ]
        compute_goals = [goal for goal in compute_goals if goal]
        if not compute_goals:
            return None

        final_goal = compute_goals[-1]
        parts = final_goal.split(';')
        if len(parts) != 9:
            self.get_logger().warn(f'Unexpected BT goal format: {final_goal}')
            return None

        try:
            return {
                'frame_id': parts[1] or 'map',
                'x': float(parts[2]),
                'y': float(parts[3]),
                'z': float(parts[4]),
                'qx': float(parts[5]),
                'qy': float(parts[6]),
                'qz': float(parts[7]),
                'qw': float(parts[8]),
            }
        except ValueError as e:
            self.get_logger().warn(f'Invalid numeric BT goal value: {e}')
            return None

    async def execute_bt(
        self,
        bt_file_path: Path,
        goal_handle,
    ) -> tuple[bool, Optional[str]]:
        try:
            if not self._nav_client.wait_for_server(timeout_sec=self.nav_server_wait_timeout):
                return False, f'Nav2 action server not available: {self.nav_action_name}'

            nav_goal = NavigateToPose.Goal()
            nav_goal.behavior_tree = str(bt_file_path.absolute())
            nav_goal.pose.header.frame_id = 'map'
            # Dynamic object tasks compute their real goal inside the BT. Keep
            # the mandatory action-level placeholder a valid planar pose.
            nav_goal.pose.pose.orientation.w = 1.0
            final_pose = self._extract_final_nav_pose_from_bt(bt_file_path)
            if final_pose:
                nav_goal.pose.header.frame_id = final_pose['frame_id']
                nav_goal.pose.pose.position.x = final_pose['x']
                nav_goal.pose.pose.position.y = final_pose['y']
                nav_goal.pose.pose.position.z = final_pose['z']
                nav_goal.pose.pose.orientation.x = final_pose['qx']
                nav_goal.pose.pose.orientation.y = final_pose['qy']
                nav_goal.pose.pose.orientation.z = final_pose['qz']
                nav_goal.pose.pose.orientation.w = final_pose['qw']
                self.get_logger().info(
                    'Using final BT waypoint as Nav2 action goal: '
                    f'{final_pose["frame_id"]} '
                    f'({final_pose["x"]:.2f}, {final_pose["y"]:.2f})'
                )
            
            send_goal_future = self._nav_client.send_goal_async(nav_goal)
            start_wait = time.time()
            while not send_goal_future.done() and (time.time() - start_wait) < 10.0:
                time.sleep(0.05)

            self.current_nav_goal_handle = send_goal_future.result()
            if not self.current_nav_goal_handle.accepted:
                return False, 'Navigation goal rejected by Nav2'

            get_result_future = self.current_nav_goal_handle.get_result_async()
            start_time = self.get_clock().now()
            last_feedback_time = start_time

            while not get_result_future.done():
                # Uscita pulita senza loggare come errore critico
                if goal_handle.is_cancel_requested:
                    self.get_logger().info('Ricevuto stop/preemption, fermo Nav2...')
                    self.current_nav_goal_handle.cancel_goal_async()
                    goal_handle.canceled()
                    return False, 'PREEMPTED'

                elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
                if elapsed > self.execution_timeout:
                    self.current_nav_goal_handle.cancel_goal_async()
                    return False, f'Execution timeout after {self.execution_timeout}s'

                current_time = self.get_clock().now()
                if (current_time - last_feedback_time).nanoseconds / 1e9 >= (1.0 / self.feedback_rate):
                    progress = min(0.5 + (elapsed / self.execution_timeout) * 0.5, 0.99)
                    self.publish_feedback(goal_handle, 'executing', progress, f'Executing BT... ({int(elapsed)}s elapsed)')
                    last_feedback_time = current_time

                time.sleep(0.1)

            result = get_result_future.result()
            if result.status == 4:
                return True, None
            else:
                return False, f'Navigation failed with status: {NAV_STATUS_NAMES.get(result.status, result.status)}'

        except Exception as e:
            return False, f'Execution error: {str(e)}'

    def publish_feedback(self, goal_handle, status: str, progress: float, step: str):
        feedback = GenerateAndExecuteBT.Feedback()
        feedback.status, feedback.progress, feedback.current_step = status, progress, step
        goal_handle.publish_feedback(feedback)

        msg = String()
        msg.data = json.dumps({
            'status': status,
            'progress': progress,
            'current_step': step,
            'timestamp_ms': int(time.time() * 1000),
        })
        self._bt_execution_feedback_publisher.publish(msg)

    def command_topic_callback(self, msg: String):
        command = msg.data.strip()
        if not command: return

        if command.upper() in ["STOP", "STOP_EXECUTION"]:
            self.get_logger().warn("Ricevuto STOP dal frontend. Interrompo il task.")
            self._cancel_active_execution()
            if self.current_goal_handle:
                self.current_goal_handle.abort()
                self.current_goal_handle = None
            self._publish_zero_velocity()
            self.is_executing = False
            return

        if self.is_executing:
            self.get_logger().warn('Un task è già in esecuzione. Lo cancello per applicare la correzione...')
            self._cancel_active_execution()
            if self.current_goal_handle:
                self.current_goal_handle.abort()
                self.current_goal_handle = None
            self._publish_zero_velocity()
            self.is_executing = False
            time.sleep(0.2)

        goal_msg = GenerateAndExecuteBT.Goal(command=command)
        if self._self_client.wait_for_server(timeout_sec=2.0):
            send_goal_future = self._self_client.send_goal_async(goal_msg)
            send_goal_future.add_done_callback(self._goal_sent_callback)

    def _goal_sent_callback(self, future):
        try:
            goal_handle = future.result()
            if goal_handle.accepted:
                self.active_client_goal_handle = goal_handle
        except Exception as e:
            self.get_logger().error(f'Failed to send goal: {e}')

    def emergency_stop_callback(self, request, response):
        if not self.is_executing:
            response.success, response.message = True, 'No active BT execution to stop'
            return response
        try:
            self._cancel_active_execution()
            if self.current_goal_handle:
                self.current_goal_handle.abort()
                self.current_goal_handle = None
            self._publish_zero_velocity()
            self.is_executing = False
            response.success, response.message = True, 'BT execution aborted successfully'
        except Exception as e:
            response.success, response.message = False, f'Emergency stop failed: {str(e)}'
        return response

    def _publish_zero_velocity(self):
        stop_msg = Twist()
        for _ in range(10):
            self._cmd_vel_nav_publisher.publish(stop_msg)
            self._cmd_vel_smoothed_publisher.publish(stop_msg)
            self._cmd_vel_publisher.publish(stop_msg)
            time.sleep(0.05)

    def _cancel_active_execution(self):
        cancel_futures = []

        if self.current_nav_goal_handle:
            cancel_futures.append(self.current_nav_goal_handle.cancel_goal_async())
            self.current_nav_goal_handle = None

        if self.active_client_goal_handle:
            cancel_futures.append(self.active_client_goal_handle.cancel_goal_async())
            self.active_client_goal_handle = None

        for future in cancel_futures:
            start_wait = time.time()
            while not future.done() and (time.time() - start_wait) < 0.5:
                time.sleep(0.05)

    def _republish_last_bt(self):
        if self.last_bt_xml:
            bt_msg = String(data=self.add_uids_for_foxglove(self.last_bt_xml))
            self._bt_xml_publisher.publish(bt_msg)

def main(args=None):
    rclpy.init(args=args)
    node = BTInterfaceNode()
    try: rclpy.spin(node, executor=MultiThreadedExecutor())
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
