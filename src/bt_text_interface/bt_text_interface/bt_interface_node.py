"""ROS2 Action Server for BehaviorTree Generation and Execution"""
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy

from btgencobot_interfaces.action import GenerateAndExecuteBT
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String
from std_srvs.srv import Trigger


NAV_STATUS_NAMES = {
    1: 'UNKNOWN', 2: 'ACCEPTED', 3: 'EXECUTING',
    4: 'SUCCEEDED', 5: 'CANCELED', 6: 'ABORTED'
}


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

        self.inference_url = self.get_parameter('inference_server_url').value
        self.bt_output_dir = Path(self.get_parameter('bt_output_dir').value)
        self.generation_timeout = self.get_parameter('generation_timeout').value
        self.execution_timeout = self.get_parameter('execution_timeout').value
        self.feedback_rate = self.get_parameter('feedback_rate').value

        self.bt_output_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_state(self):
        """Initialize state variables"""
        self.current_goal_handle = None
        self.current_nav_goal_handle = None
        self.is_executing = False
        self.last_bt_xml = None  # Store last executed BT for republishing

    def _setup_interfaces(self):
        """Setup ROS interfaces: publishers, subscribers, action servers/clients, services"""
        self.action_callback_group = ReentrantCallbackGroup()

        # Use TRANSIENT_LOCAL durability so late-joining subscribers get the last message
        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self._bt_xml_publisher = self.create_publisher(String, '/generated_behavior_tree', qos_latched)

        # Republish last BT every 2 seconds for visibility
        self.bt_republish_timer = self.create_timer(60.0, self._republish_last_bt)

        self._action_server = ActionServer(
            self, GenerateAndExecuteBT, '/generate_and_execute_bt',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.action_callback_group
        )

        self._nav_client = ActionClient(
            self, NavigateToPose, '/navigate_to_pose',
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
        """Log node configuration"""
        self.get_logger().info(f'BT output directory: {self.bt_output_dir}')
        self.get_logger().info(f'Inference server URL: {self.inference_url}')
        self.get_logger().info('BT Interface Node initialized')
        self.get_logger().info('Action server: /generate_and_execute_bt')
        self.get_logger().info('Topic subscriber: /btgen_nl_command')
        self.get_logger().info('Emergency stop service: /emergency_stop_bt')

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action"""
        self.get_logger().info(f'Received goal request: {goal_request.command}')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject a client request to cancel an action"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute the action: generate BT from command and execute it"""
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
            if bt_xml is None:
                result.error_message = f'BT generation failed: {error}'
                self.get_logger().error(result.error_message)
                self.is_executing = False
                return result
            self.get_logger().info('BT generation successful')

            self.publish_feedback(goal_handle, 'validating', 0.3, 'Validating generated BT...')
            is_valid, val_error = self.validate_bt_xml(bt_xml)
            if not is_valid:
                result.error_message = f'BT validation failed: {val_error}'
                self.get_logger().error(result.error_message)
                self.is_executing = False
                return result
            self.get_logger().info('BT validation successful')

            self.publish_feedback(goal_handle, 'validating', 0.4, 'Writing BT to file...')
            bt_file_path = self.write_bt_file(bt_xml)
            result.bt_xml_path = str(bt_file_path)
            self.get_logger().info(f'BT written to: {bt_file_path}')

            # Store the original BT and publish with UIDs for Foxglove visualization
            self.last_bt_xml = bt_xml
            bt_msg = String()
            bt_msg.data = self.add_uids_for_foxglove(bt_xml)
            self._bt_xml_publisher.publish(bt_msg)
            self.get_logger().info('BT published to /generated_behavior_tree topic')

            self.publish_feedback(goal_handle, 'executing', 0.5, 'Executing BehaviorTree...')
            execution_success, exec_error = await self.execute_bt(bt_file_path, goal_handle)
            if not execution_success:
                result.error_message = f'BT execution failed: {exec_error}'
                self.get_logger().error(result.error_message)
                self.is_executing = False
                return result

            self.get_logger().info('BT execution completed successfully')
            self.publish_feedback(goal_handle, 'completed', 1.0, 'BT execution completed')

            result.success = True
            goal_handle.succeed()

        except Exception as e:
            result.error_message = f'Unexpected error: {str(e)}'
            self.get_logger().error(result.error_message)
            self.get_logger().error(f'Exception: {e}', exc_info=True)
            goal_handle.abort()

        finally:
            self.is_executing = False
            self.current_goal_handle = None

        return result

    async def generate_bt_from_command(self, command: str) -> tuple[Optional[str], Optional[str]]:
        """Call the inference server to generate BT XML from natural language command.
        Returns (bt_xml, error_message) - bt_xml is None if generation failed."""
        try:
            self.get_logger().info(f'Calling inference server: {self.inference_url}/generate_bt')

            response = requests.post(
                f'{self.inference_url}/generate_bt',
                json={
                    'command': command,
                    'max_tokens': 1024,
                    'temperature': 0.1,
                    'prompt_format': 'alpaca',
                    'use_query_rewriting': False
                },
                timeout=self.generation_timeout
            )

            if response.status_code != 200:
                return None, f'HTTP {response.status_code}: {response.text}'

            data = response.json()
            if not data.get('success', False):
                return None, data.get('error', 'Unknown error')

            bt_xml = data.get('bt_xml')
            method = data.get('method_used')
            gen_time_ms = data.get('generation_time_ms')

            self.get_logger().info(f'Generation method: {method}, time: {gen_time_ms}ms')
            return bt_xml, None

        except requests.Timeout:
            return None, 'Inference server timeout'
        except requests.ConnectionError:
            return None, 'Could not connect to inference server'
        except Exception as e:
            return None, f'Request failed: {str(e)}'

    def validate_bt_xml(self, xml_string: str) -> tuple[bool, Optional[str]]:
        """Validate BehaviorTree XML structure. Returns (is_valid, error_message)."""
        try:
            root = ET.fromstring(xml_string)

            if root.tag != 'root':
                return False, f"Root element must be 'root', got '{root.tag}'"

            btcpp_format = root.get('BTCPP_format')
            if btcpp_format != '4':
                return False, f"BTCPP_format must be '4', got '{btcpp_format}'"

            behavior_trees = root.findall('BehaviorTree')
            if len(behavior_trees) == 0:
                return False, 'No BehaviorTree element found'

            for bt in behavior_trees:
                tree_id = bt.get('ID')
                if not tree_id:
                    return False, 'BehaviorTree missing ID attribute'
                if len(list(bt)) == 0:
                    return False, f"BehaviorTree '{tree_id}' is empty"

            return True, None

        except ET.ParseError as e:
            return False, f'XML parse error: {str(e)}'
        except Exception as e:
            return False, f'Validation error: {str(e)}'

    def add_uids_for_foxglove(self, xml_string: str) -> str:
        """Add unique _uid attributes to all nodes for Foxglove Polymath BT panel visualization.
        This is only used for publishing to the topic, not for the file written to disk."""
        try:
            root = ET.fromstring(xml_string)
            node_counts = {}  # Track counts per node type for uniqueness

            def add_uid_recursive(element):
                """Recursively add _uid to element and all children."""
                tag = element.tag

                # Build a meaningful UID based on node type and attributes
                if tag == 'root':
                    uid = 'root'
                elif tag == 'BehaviorTree':
                    tree_id = element.get('ID', 'Tree')
                    uid = f'BehaviorTree_{tree_id}'
                else:
                    # For action/condition/control nodes, use tag + name or key attribute
                    name = element.get('name')
                    node_id = element.get('ID')  # For explicit syntax <Action ID="..."/>

                    if name:
                        base = f'{tag}_{name}'
                    elif node_id:
                        base = f'{tag}_{node_id}'
                    elif tag in ('Action', 'Condition'):
                        # Explicit syntax without name
                        base = tag
                    else:
                        # Control/decorator nodes or leaf nodes with compact syntax
                        base = tag

                    # Add distinguishing attribute for certain nodes
                    if tag == 'DetectObject':
                        obj = element.get('object_description', '')
                        if obj:
                            base = f'{tag}_{obj}'
                    elif tag == 'PickObject':
                        obj = element.get('object_description', '')
                        if obj:
                            base = f'{tag}_{obj}'
                    elif tag == 'PlaceObject':
                        desc = element.get('place_description', '')
                        if desc:
                            base = f'{tag}_{desc}'
                    elif tag in ('SpinLeft', 'SpinRight'):
                        dist = element.get('spin_dist', '')
                        if dist:
                            base = f'{tag}_{dist}rad'
                    elif tag == 'Wait':
                        dur = element.get('wait_duration', '')
                        if dur:
                            base = f'{tag}_{dur}s'
                    elif tag == 'Repeat':
                        cycles = element.get('num_cycles', '')
                        if cycles:
                            base = f'{tag}_{cycles}x'

                    # Sanitize and ensure uniqueness
                    base = base.replace(' ', '_').replace('"', '').replace("'", '')
                    node_counts[base] = node_counts.get(base, 0) + 1
                    count = node_counts[base]
                    uid = f'{base}_{count}' if count > 1 else base

                element.set('_uid', uid)
                for child in element:
                    add_uid_recursive(child)

            add_uid_recursive(root)

            return ET.tostring(root, encoding='unicode', xml_declaration=True)

        except ET.ParseError:
            return xml_string

    def write_bt_file(self, xml_content: str) -> Path:
        """Write BT XML to file with UUID naming. Returns path to written file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:8]
        filename = f'bt_{timestamp}_{unique_id}.xml'
        file_path = self.bt_output_dir / filename

        with open(file_path, 'w') as f:
            f.write(xml_content)

        file_path.chmod(0o644)
        return file_path

    async def execute_bt(self, bt_file_path: Path, goal_handle) -> tuple[bool, Optional[str]]:
        """Execute BT by calling Nav2's navigate_to_pose action with behavior_tree parameter.
        Returns (success, error_message)."""
        try:
            self.get_logger().info('Waiting for Nav2 action server...')
            if not self._nav_client.wait_for_server(timeout_sec=5.0):
                return False, 'Nav2 action server not available'

            nav_goal = NavigateToPose.Goal()
            nav_goal.behavior_tree = str(bt_file_path.absolute())
            nav_goal.pose.header.frame_id = 'map'
            nav_goal.pose.header.stamp = self.get_clock().now().to_msg()

            self.get_logger().info(f'Sending NavigateToPose goal with BT: {bt_file_path}')

            send_goal_future = self._nav_client.send_goal_async(nav_goal)

            start_wait = time.time()
            while not send_goal_future.done() and (time.time() - start_wait) < 10.0:
                time.sleep(0.05)

            if not send_goal_future.done():
                return False, 'Failed to send navigation goal (timeout)'

            self.current_nav_goal_handle = send_goal_future.result()
            if not self.current_nav_goal_handle.accepted:
                return False, 'Navigation goal rejected by Nav2'

            self.get_logger().info('Navigation goal accepted, waiting for result...')

            get_result_future = self.current_nav_goal_handle.get_result_async()
            start_time = self.get_clock().now()
            last_feedback_time = start_time
            feedback_interval = 1.0 / self.feedback_rate

            while not get_result_future.done():
                if goal_handle.is_cancel_requested:
                    self.get_logger().info('Goal cancelled, aborting navigation...')
                    self.current_nav_goal_handle.cancel_goal_async()
                    goal_handle.canceled()
                    return False, 'Cancelled by user'

                elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
                if elapsed > self.execution_timeout:
                    self.get_logger().warn(f'Execution timeout ({self.execution_timeout}s)')
                    self.current_nav_goal_handle.cancel_goal_async()
                    return False, f'Execution timeout after {self.execution_timeout}s'

                current_time = self.get_clock().now()
                time_since_last_feedback = (current_time - last_feedback_time).nanoseconds / 1e9

                if time_since_last_feedback >= feedback_interval:
                    progress = min(0.5 + (elapsed / self.execution_timeout) * 0.5, 0.99)
                    self.publish_feedback(goal_handle, 'executing', progress, f'Executing BT... ({int(elapsed)}s elapsed)')
                    last_feedback_time = current_time

                time.sleep(0.1)

            result = get_result_future.result()
            if result.status == 4:
                self.get_logger().info('Navigation completed successfully')
                return True, None
            else:
                status_name = NAV_STATUS_NAMES.get(result.status, f'UNKNOWN({result.status})')
                return False, f'Navigation failed with status: {status_name}'

        except Exception as e:
            return False, f'Execution error: {str(e)}'

    def publish_feedback(self, goal_handle, status: str, progress: float, step: str):
        """Publish feedback to action client"""
        feedback = GenerateAndExecuteBT.Feedback()
        feedback.status = status
        feedback.progress = progress
        feedback.current_step = step
        goal_handle.publish_feedback(feedback)
        self.get_logger().info(f'[{status}] {progress*100:.0f}% - {step}')

    def command_topic_callback(self, msg: String):
        """Callback for /btgen_nl_command topic.
        Receives NL command from topic (e.g., Foxglove) and triggers the action server internally."""
        command = msg.data.strip()

        if not command:
            self.get_logger().warn('Received empty command on /btgen_nl_command')
            return

        self.get_logger().info(f'Received command from topic: {command}')
        if self.is_executing:
            self.get_logger().warn('Note: A command is currently executing. This command will be queued.')

        goal_msg = GenerateAndExecuteBT.Goal()
        goal_msg.command = command

        if not self._self_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('Action server not available for self-invocation!')
            return

        self.get_logger().info('Sending goal to action server...')
        send_goal_future = self._self_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self._goal_sent_callback)

    def _goal_sent_callback(self, future):
        """Callback when goal is sent to action server"""
        try:
            goal_handle = future.result()
            if goal_handle.accepted:
                self.get_logger().info('Goal accepted by action server')
            else:
                self.get_logger().error('Goal rejected by action server!')
        except Exception as e:
            self.get_logger().error(f'Failed to send goal: {e}')

    def emergency_stop_callback(self, request, response):
        """Emergency stop service callback. Aborts current BT execution."""
        self.get_logger().warn('EMERGENCY STOP REQUESTED')

        if not self.is_executing:
            response.success = True
            response.message = 'No active BT execution to stop'
            self.get_logger().info(response.message)
            return response

        try:
            if self.current_nav_goal_handle is not None:
                self.current_nav_goal_handle.cancel_goal_async()
                self.get_logger().info('Navigation goal cancelled')

            if self.current_goal_handle is not None:
                self.current_goal_handle.abort()
                self.get_logger().info('Action goal aborted')

            self.is_executing = False
            response.success = True
            response.message = 'BT execution aborted successfully'

        except Exception as e:
            response.success = False
            response.message = f'Emergency stop failed: {str(e)}'
            self.get_logger().error(response.message)

        return response

    def _republish_last_bt(self):
        """Periodically republish the last executed BT for late-joining subscribers"""
        if self.last_bt_xml is not None:
            bt_msg = String()
            bt_msg.data = self.add_uids_for_foxglove(self.last_bt_xml)
            self._bt_xml_publisher.publish(bt_msg)


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)

    node = BTInterfaceNode()

    # Use multi-threaded executor for concurrent action handling
    executor = MultiThreadedExecutor()

    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
