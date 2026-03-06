"""ROS2 Action Server for BehaviorTree Generation and Execution"""
import hashlib
import json
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
from nav2_msgs.msg import BehaviorTreeLog
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
        self.active_tree_id = None
        self.active_xml_hash = None
        self.uid_lookup = {}
        self.node_status = {}

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
        self._active_tree_publisher = self.create_publisher(String, '/bt_active_tree', qos_latched)

        self._bt_execution_event_publisher = self.create_publisher(String, '/bt_execution_event', 10)
        self._bt_execution_state_publisher = self.create_publisher(String, '/bt_execution_state', 10)

        self._bt_log_subscriber = self.create_subscription(
            BehaviorTreeLog,
            '/behavior_tree_log',
            self.behavior_tree_log_callback,
            10
        )

        # Republish last BT every 2 seconds for visibility
        self.bt_republish_timer = self.create_timer(2.0, self._republish_last_bt)

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
            bt_xml_with_uids, context = self.prepare_bt_for_execution(bt_xml)
            self.active_tree_id = context['tree_id']
            self.active_xml_hash = context['xml_hash']
            self.uid_lookup = context['uid_lookup']
            self.node_status = {'root': 'IDLE'}

            bt_file_path = self.write_bt_file(bt_xml_with_uids)
            result.bt_xml_path = str(bt_file_path)
            self.get_logger().info(f'BT written to: {bt_file_path}')

            # Store and publish the active BT with stable UIDs for frontend alignment
            self.last_bt_xml = bt_xml_with_uids
            bt_msg = String()
            bt_msg.data = bt_xml_with_uids
            self._bt_xml_publisher.publish(bt_msg)

            active_tree_msg = String()
            active_tree_msg.data = json.dumps({
                'treeId': self.active_tree_id,
                'xmlHash': self.active_xml_hash,
                'ts': self.current_time_ms()
            })
            self._active_tree_publisher.publish(active_tree_msg)
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

    def prepare_bt_for_execution(self, xml_string: str) -> tuple[str, dict]:
        """Inject stable uid attributes into BT nodes and build lookup metadata."""
        root = ET.fromstring(xml_string)
        xml_hash = hashlib.sha256(xml_string.encode('utf-8')).hexdigest()[:16]
        tree_id = f"bt_{datetime.now().strftime('%Y%m%dT%H%M%S')}_{xml_hash}"

        uid_lookup = {
            'by_name': {},
            'by_id': {},
            'by_tag': {},
        }

        counter = 0

        def sanitize(value: str) -> str:
            return ''.join(ch if ch.isalnum() else '_' for ch in value).strip('_').lower()[:40]

        def register_lookup(element, uid: str):
            name_attr = element.get('name')
            id_attr = element.get('ID')
            tag = element.tag

            if name_attr and name_attr not in uid_lookup['by_name']:
                uid_lookup['by_name'][name_attr] = uid
            if id_attr and id_attr not in uid_lookup['by_id']:
                uid_lookup['by_id'][id_attr] = uid
            uid_lookup['by_tag'].setdefault(tag, []).append(uid)

        def walk(element):
            nonlocal counter
            counter += 1
            base = sanitize(f"{element.tag}_{element.get('name') or element.get('ID') or counter}")
            uid = f"n_{counter:04d}_{base}"
            element.set('uid', uid)
            element.set('_uid', uid)
            register_lookup(element, uid)

            for child in element:
                walk(child)

        walk(root)

        context = {
            'tree_id': tree_id,
            'xml_hash': xml_hash,
            'uid_lookup': uid_lookup,
        }
        return ET.tostring(root, encoding='unicode', xml_declaration=True), context

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
                    'use_query_rewriting': True
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

    def behavior_tree_log_callback(self, msg: BehaviorTreeLog):
        """Convert Nav2 BehaviorTreeLog transitions into frontend-friendly JSON events."""
        if not self.active_tree_id:
            return

        for change in msg.event_log:
            node_name = getattr(change, 'node_name', '')
            from_state = self.normalize_status(getattr(change, 'previous_status', 'UNKNOWN'))
            to_state = self.normalize_status(getattr(change, 'current_status', 'UNKNOWN'))
            uid = self.resolve_uid(node_name)

            event = {
                'treeId': self.active_tree_id,
                'xmlHash': self.active_xml_hash,
                'uid': uid,
                'nodeName': node_name,
                'from': from_state,
                'to': to_state,
                'ts': self.current_time_ms(),
            }
            self.node_status[uid] = to_state
            self.publish_execution_event(event)

        self.publish_execution_snapshot()

    def resolve_uid(self, node_name: str) -> str:
        """Resolve a Nav2 node name to generated uid for frontend alignment."""
        if node_name in self.uid_lookup.get('by_name', {}):
            return self.uid_lookup['by_name'][node_name]
        if node_name in self.uid_lookup.get('by_id', {}):
            return self.uid_lookup['by_id'][node_name]

        by_tag = self.uid_lookup.get('by_tag', {}).get(node_name, [])
        if len(by_tag) == 1:
            return by_tag[0]
        if by_tag:
            return by_tag[0]

        return f'unknown_{node_name or "node"}'

    def publish_execution_event(self, payload: dict):
        msg = String()
        msg.data = json.dumps(payload)
        self._bt_execution_event_publisher.publish(msg)

    def publish_execution_snapshot(self):
        msg = String()
        msg.data = json.dumps({
            'treeId': self.active_tree_id,
            'xmlHash': self.active_xml_hash,
            'ts': self.current_time_ms(),
            'nodes': [{'uid': uid, 'status': status} for uid, status in sorted(self.node_status.items())],
        })
        self._bt_execution_state_publisher.publish(msg)

    @staticmethod
    def normalize_status(status) -> str:
        status_map = {
            0: 'IDLE',
            1: 'RUNNING',
            2: 'SUCCESS',
            3: 'FAILURE',
            4: 'SKIPPED',
            5: 'HALTED',
        }
        if isinstance(status, int):
            return status_map.get(status, 'UNKNOWN')

        normalized = str(status or 'UNKNOWN').upper()
        known = {'IDLE', 'RUNNING', 'SUCCESS', 'FAILURE', 'SKIPPED', 'HALTED', 'UNKNOWN'}
        return normalized if normalized in known else 'UNKNOWN'

    @staticmethod
    def current_time_ms() -> int:
        return int(time.time() * 1000)

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
            bt_msg.data = self.last_bt_xml
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
