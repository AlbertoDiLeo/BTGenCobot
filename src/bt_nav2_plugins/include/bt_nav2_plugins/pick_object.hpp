#ifndef BT_NAV2_PLUGINS__PICK_OBJECT_HPP_
#define BT_NAV2_PLUGINS__PICK_OBJECT_HPP_

#include <string>
#include <memory>
#include <atomic>

#include "behaviortree_cpp/action_node.h"
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "btgencobot_interfaces/srv/manipulator_action.hpp"
#include "btgencobot_interfaces/srv/detect_object.hpp"
#include "geometry_msgs/msg/twist.hpp"

namespace bt_nav2_plugins
{

/**
 * @brief BT node to pick up an object with the manipulator
 *
 * This node performs close-range object detection before picking for accurate pose estimation.
 * The robot should already be positioned near the object (via prior navigation).
 *
 * Input Ports:
 *   object_description - Natural language description of object to pick (e.g., "red cup")
 *   box_threshold - Detection confidence threshold (default: 0.35)
 *
 * The node:
 * 1. Captures current camera image
 * 2. Calls /detect_object service for accurate close-range detection
 * 3. Computes object pose from detection result
 * 4. Calls /manipulator_action service to execute pick
 */
class PickObject : public BT::StatefulActionNode
{
public:
  PickObject(
    const std::string & name,
    const BT::NodeConfiguration & config);

  ~PickObject() override = default;

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<std::string>("object_description", "Natural language description of object to pick"),
      BT::InputPort<double>("box_threshold", 0.35, "Detection confidence threshold (0-1)")
    };
  }

private:
  BT::NodeStatus onStart() override;
  BT::NodeStatus onRunning() override;
  void onHalted() override;

  // Callbacks for camera data
  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
  void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg);
  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

  // Convert detection result to 3D pose
  geometry_msgs::msg::PoseStamped computeObjectPose(
    float center_x,
    float center_y,
    float depth_value,
    const std::string & frame_id);

  // Nav2's node for logging
  rclcpp::Node::SharedPtr node_;
  
  // Separate node for service calls and subscriptions
  rclcpp::Node::SharedPtr service_node_;
  
  // TF2 for coordinate transforms
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // Service clients
  rclcpp::Client<btgencobot_interfaces::srv::DetectObject>::SharedPtr detect_client_;
  rclcpp::Client<btgencobot_interfaces::srv::ManipulatorAction>::SharedPtr manipulator_client_;
  
  // Subscriptions for camera data
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

  // Latest camera data
  sensor_msgs::msg::Image::SharedPtr latest_image_;
  sensor_msgs::msg::Image::SharedPtr latest_depth_;
  
  // Camera calibration
  bool has_camera_info_;
  double fx_, fy_, cx_, cy_;

  // Input parameters
  std::string object_description_;
  double box_threshold_;

  // State machine for the pick operation
  enum class PickState {
    WAITING_FOR_IMAGE,
    DETECTING,
    APPROACHING,  // Final approach using direct cmd_vel
    PICKING,
    RETREATING,
    DONE
  };
  PickState state_;
  
  // Detection state
  btgencobot_interfaces::srv::DetectObject::Response::SharedPtr detection_response_;
  std::atomic<bool> detection_sent_;
  std::atomic<bool> detection_received_;
  
  // Pick state
  btgencobot_interfaces::srv::ManipulatorAction::Response::SharedPtr pick_response_;
  std::atomic<bool> pick_sent_;
  std::atomic<bool> pick_received_;
  
  // Computed object pose and dimensions
  geometry_msgs::msg::PoseStamped object_pose_;
  double object_height_;
  double object_width_;

  // Timing
  rclcpp::Time operation_start_time_;
  rclcpp::Time retreat_start_time_;

  // Final approach - direct motion control bypassing Nav2 costmap
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  float detected_depth_;  // Object depth from detection (for approach distance)
  rclcpp::Time approach_start_time_;
  bool approach_done_;  // Flag to prevent infinite approach loops
  bool retreat_started_;
  double retreat_start_x_;
  double retreat_start_y_;
  static constexpr double APPROACH_VELOCITY = 0.06;  // m/s - controlled final approach
  static constexpr double APPROACH_ANGULAR_GAIN = 1.4;
  static constexpr double MAX_APPROACH_ANGULAR_VELOCITY = 0.35;  // rad/s
  static constexpr double APPROACH_HEADING_TOLERANCE = 0.07;  // rad, about 4 degrees
  static constexpr double RETREAT_VELOCITY = -0.08;  // m/s - match approach speed in reverse
  static constexpr double RETREAT_DISTANCE = 0.30;  // Back away from table before reporting success
  static constexpr double MAX_RETREAT_TIME = 6.0;  // seconds
  // Distance from robot base_link to object for arm to reach
  // Arm reaches ~0.286m from link1, which is at -0.092m from base_link
  // So arm can reach ~0.19m in front of base_link
  // Keep the mobile base clear of the support while remaining within arm reach.
  static constexpr double MIN_APPROACH_DISTANCE = 0.28;  // Keep the base clear of the table
};

}  // namespace bt_nav2_plugins

#endif  // BT_NAV2_PLUGINS__PICK_OBJECT_HPP_
