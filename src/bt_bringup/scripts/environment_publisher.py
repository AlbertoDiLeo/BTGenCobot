#!/usr/bin/env python3

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class EnvironmentPublisher(Node):
    def __init__(self) -> None:
        super().__init__('environment_publisher')
        self.declare_parameter('environment_id', 'aws_small_house')

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.publisher = self.create_publisher(String, '/btgen/environment', qos)
        self.environment_id = (
            self.get_parameter('environment_id').get_parameter_value().string_value
        )
        self.timer = self.create_timer(1.0, self.publish_environment)
        self.publish_environment()

    def publish_environment(self) -> None:
        message = String()
        message.data = self.environment_id
        self.publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EnvironmentPublisher()
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
