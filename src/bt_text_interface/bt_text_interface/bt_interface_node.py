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
        self.get_logger().info(f'BT output directory: {self.bt_output_dir}')
        self.get_logger().info(f'Inference server URL: {self.inference_url}')
        self.get_logger().info('BT Interface Node initialized')

    def goal_callback(self, goal_request):
        self.get_logger().info(f'Received goal request: {goal_request.command}')
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

    async def generate_bt_from_command(self, command: str) -> tuple[Optional[str], Optional[str]]:
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

    async def execute_bt(self, bt_file_path: Path, goal_handle) -> tuple[bool, Optional[str]]:
        try:
            if not self._nav_client.wait_for_server(timeout_sec=5.0):
                return False, 'Nav2 action server not available'

            nav_goal = NavigateToPose.Goal()
            nav_goal.behavior_tree = str(bt_file_path.absolute())
            nav_goal.pose.header.frame_id = 'map'
            
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

    def command_topic_callback(self, msg: String):
        command = msg.data.strip()
        if not command: return

        if command.upper() in ["STOP", "STOP_EXECUTION"]:
            self.get_logger().warn("Ricevuto STOP dal frontend. Interrompo il task.")
            if self.is_executing and self.active_client_goal_handle:
                self.active_client_goal_handle.cancel_goal_async()
                self.active_client_goal_handle = None
            return

        if self.is_executing:
            self.get_logger().warn('Un task è già in esecuzione. Lo cancello per applicare la correzione...')
            if self.active_client_goal_handle:
                self.active_client_goal_handle.cancel_goal_async()
                self.active_client_goal_handle = None

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
            if self.current_nav_goal_handle: self.current_nav_goal_handle.cancel_goal_async()
            if self.current_goal_handle: self.current_goal_handle.abort()
            self.is_executing = False
            response.success, response.message = True, 'BT execution aborted successfully'
        except Exception as e:
            response.success, response.message = False, f'Emergency stop failed: {str(e)}'
        return response

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