#include "bt_nav2_plugins/place_object.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include "cv_bridge/cv_bridge.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

using namespace std::chrono_literals;

namespace bt_nav2_plugins
{

PlaceObject::PlaceObject(
  const std::string & name,
  const BT::NodeConfiguration & config)
: BT::StatefulActionNode(name, config),
  has_camera_info_(false),
  state_(PlaceState::WAITING_FOR_IMAGE),
  detection_sent_(false),
  detection_received_(false),
  place_sent_(false),
  place_received_(false),
  detected_depth_(0.0f),
  approach_done_(false),
  retreat_started_(false),
  retreat_start_x_(0.0),
  retreat_start_y_(0.0)
{
  // Get ROS node from config (Nav2's node - used for logging)
  if (!config.blackboard->get("node", node_) || !node_) {
    throw BT::RuntimeError("PlaceObject: 'node' not found in blackboard");
  }

  // Create a separate node for service calls and subscriptions
  service_node_ = std::make_shared<rclcpp::Node>("place_object_service_node");

  // Initialize TF2
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Create service clients
  detect_client_ = service_node_->create_client<btgencobot_interfaces::srv::DetectObject>(
    "/detect_object");
  manipulator_client_ = service_node_->create_client<btgencobot_interfaces::srv::ManipulatorAction>(
    "/manipulator_action");

  // Use sensor data QoS for camera topics
  auto sensor_qos = rclcpp::SensorDataQoS();

  // Subscribe to camera topics
  image_sub_ = service_node_->create_subscription<sensor_msgs::msg::Image>(
    "/camera",
    sensor_qos,
    std::bind(&PlaceObject::imageCallback, this, std::placeholders::_1));

  depth_sub_ = service_node_->create_subscription<sensor_msgs::msg::Image>(
    "/camera/depth",
    sensor_qos,
    std::bind(&PlaceObject::depthCallback, this, std::placeholders::_1));

  camera_info_sub_ = service_node_->create_subscription<sensor_msgs::msg::CameraInfo>(
    "/camera/camera_info",
    sensor_qos,
    std::bind(&PlaceObject::cameraInfoCallback, this, std::placeholders::_1));

  // Publisher for direct motion control (final approach bypassing Nav2 costmap)
  cmd_vel_pub_ = service_node_->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

  RCLCPP_INFO(node_->get_logger(), "PlaceObject BT node initialized with integrated detection and final approach");
}

BT::NodeStatus PlaceObject::onStart()
{
  RCLCPP_INFO(node_->get_logger(), "PlaceObject: Starting place operation with close-range detection");

  // Get input ports
  if (!getInput<std::string>("place_description", place_description_)) {
    RCLCPP_ERROR(node_->get_logger(), "PlaceObject: Missing required input 'place_description'");
    return BT::NodeStatus::FAILURE;
  }

  if (!getInput<double>("box_threshold", box_threshold_)) {
    box_threshold_ = 0.35;
  }

  RCLCPP_INFO(
    node_->get_logger(),
    "PlaceObject: Will detect '%s' (threshold: %.2f) then place",
    place_description_.c_str(),
    box_threshold_);

  // Reset state
  state_ = PlaceState::WAITING_FOR_IMAGE;
  detection_sent_ = false;
  detection_received_ = false;
  detection_response_.reset();
  place_sent_ = false;
  place_received_ = false;
  place_response_.reset();
  latest_image_.reset();
  latest_depth_.reset();
  has_camera_info_ = false;
  detected_depth_ = 0.0f;
  approach_done_ = false;
  retreat_started_ = false;
  retreat_start_x_ = 0.0;
  retreat_start_y_ = 0.0;

  operation_start_time_ = node_->now();

  return BT::NodeStatus::RUNNING;
}

BT::NodeStatus PlaceObject::onRunning()
{
  // Spin our node to process callbacks
  rclcpp::spin_some(service_node_);

  switch (state_) {
    case PlaceState::WAITING_FOR_IMAGE:
    {
      // Wait for fresh image
      if (!latest_image_) {
        RCLCPP_WARN_THROTTLE(
          node_->get_logger(),
          *node_->get_clock(),
          1000,
          "PlaceObject: Waiting for camera image...");
        return BT::NodeStatus::RUNNING;
      }

      // Check if image is fresh
      rclcpp::Time image_time(latest_image_->header.stamp);
      if (image_time < operation_start_time_) {
        RCLCPP_INFO_THROTTLE(
          node_->get_logger(),
          *node_->get_clock(),
          500,
          "PlaceObject: Discarding stale image, waiting for fresh one...");
        latest_image_.reset();
        latest_depth_.reset();
        return BT::NodeStatus::RUNNING;
      }

      // Use default camera info if not received
      if (!has_camera_info_) {
        RCLCPP_WARN_ONCE(
          node_->get_logger(),
          "PlaceObject: No camera_info received, using defaults");
        fx_ = 554.3;
        fy_ = 554.3;
        cx_ = 320.5;
        cy_ = 240.5;
      }

      state_ = PlaceState::DETECTING;
      RCLCPP_INFO(node_->get_logger(), "PlaceObject: Got fresh image, starting detection...");
      return BT::NodeStatus::RUNNING;
    }

    case PlaceState::DETECTING:
    {
      // Send detection request if not sent
      if (!detection_sent_) {
        if (!detect_client_->wait_for_service(0s)) {
          RCLCPP_WARN_THROTTLE(
            node_->get_logger(),
            *node_->get_clock(),
            2000,
            "PlaceObject: Waiting for /detect_object service...");
          return BT::NodeStatus::RUNNING;
        }

        auto request = std::make_shared<btgencobot_interfaces::srv::DetectObject::Request>();
        request->image = *latest_image_;
        request->object_description = place_description_;
        request->box_threshold = static_cast<float>(box_threshold_);

        RCLCPP_INFO(node_->get_logger(), "PlaceObject: Sending close-range detection request for place location...");

        detect_client_->async_send_request(
          request,
          [this](rclcpp::Client<btgencobot_interfaces::srv::DetectObject>::SharedFuture future) {
            try {
              detection_response_ = future.get();
              detection_received_ = true;
            } catch (const std::exception & e) {
              RCLCPP_ERROR(node_->get_logger(), "PlaceObject: Detection failed: %s", e.what());
              detection_received_ = true;
            }
          });
        detection_sent_ = true;
        return BT::NodeStatus::RUNNING;
      }

      // Wait for detection response
      if (!detection_received_) {
        RCLCPP_INFO_THROTTLE(
          node_->get_logger(),
          *node_->get_clock(),
          2000,
          "PlaceObject: Waiting for detection response...");
        return BT::NodeStatus::RUNNING;
      }

      // Process detection result
      if (!detection_response_ || !detection_response_->detected) {
        RCLCPP_ERROR(
          node_->get_logger(),
          "PlaceObject: Failed to detect place location '%s' at close range",
          place_description_.c_str());
        return BT::NodeStatus::FAILURE;
      }

      RCLCPP_INFO(
        node_->get_logger(),
        "PlaceObject: Detected place location '%s' with confidence %.2f",
        detection_response_->phrase.c_str(),
        detection_response_->confidence);

      // Compute depth for place location
      // For placing, we want to find the TOP surface of the detected object (table, box, etc.)
      float depth = 0.3f;  // Default close-range depth
      float place_center_x = detection_response_->center_x;
      float place_center_y = detection_response_->center_y;

      if (latest_depth_) {
        try {
          cv_bridge::CvImagePtr depth_ptr = cv_bridge::toCvCopy(
            latest_depth_,
            sensor_msgs::image_encodings::TYPE_32FC1);

          struct DepthSample {
            float depth;
            int x;
            int y;
          };
          std::vector<DepthSample> depth_samples;

          if (detection_response_->bbox.size() >= 4) {
            int x1 = static_cast<int>(detection_response_->bbox[0]);
            int y1 = static_cast<int>(detection_response_->bbox[1]);
            int x2 = static_cast<int>(detection_response_->bbox[2]);
            int y2 = static_cast<int>(detection_response_->bbox[3]);

            x1 = std::max(0, std::min(x1, depth_ptr->image.cols - 1));
            y1 = std::max(0, std::min(y1, depth_ptr->image.rows - 1));
            x2 = std::max(0, std::min(x2, depth_ptr->image.cols - 1));
            y2 = std::max(0, std::min(y2, depth_ptr->image.rows - 1));

            // For place location (surface), sample from the top portion of bbox
            // This helps find the top surface for placing
            int height = y2 - y1;
            int top_region_y2 = y1 + height / 3;  // Top third of bbox
            
            int margin_x = (x2 - x1) * 0.15;
            int inner_x1 = x1 + margin_x;
            int inner_x2 = x2 - margin_x;

            RCLCPP_INFO(
              node_->get_logger(),
              "PlaceObject: Sampling depth from top region (%d,%d)-(%d,%d)",
              inner_x1, y1, inner_x2, top_region_y2);

            int sample_stride = 2;
            for (int y = y1; y <= top_region_y2; y += sample_stride) {
              for (int x = inner_x1; x <= inner_x2; x += sample_stride) {
                float d = depth_ptr->image.at<float>(y, x);
                if (!std::isnan(d) && d > 0.1 && d < 10.0) {
                  depth_samples.push_back({d, x, y});
                }
              }
            }
          }

          if (!depth_samples.empty()) {
            // For placing, use the closest (surface) depth
            std::sort(depth_samples.begin(), depth_samples.end(),
              [](const DepthSample& a, const DepthSample& b) { return a.depth < b.depth; });

            // Use median of front samples for robust surface depth
            size_t front_count = std::min(depth_samples.size(), static_cast<size_t>(depth_samples.size() / 3 + 1));
            depth = depth_samples[front_count / 2].depth;

            // Use center of top region for place position
            double sum_x = 0.0, sum_y = 0.0;
            for (size_t i = 0; i < front_count; ++i) {
              sum_x += depth_samples[i].x;
              sum_y += depth_samples[i].y;
            }
            place_center_x = static_cast<float>(sum_x / front_count);
            place_center_y = static_cast<float>(sum_y / front_count);

            RCLCPP_INFO(
              node_->get_logger(),
              "PlaceObject: Place surface at depth %.3fm, center (%.1f, %.1f)",
              depth, place_center_x, place_center_y);
          }
        } catch (const std::exception & e) {
          RCLCPP_WARN(node_->get_logger(), "PlaceObject: Depth processing failed: %s", e.what());
        }
      }

      // Compute place pose - slightly above the detected surface
      place_pose_ = computePlacePose(place_center_x, place_center_y, depth, "camera_rgb_optical_frame");

      // Save depth for approach calculation
      detected_depth_ = depth;

      // Check if we need to approach closer (only if we haven't already approached)
      if (!approach_done_ && detected_depth_ > MIN_APPROACH_DISTANCE) {
        state_ = PlaceState::APPROACHING;
        approach_start_time_ = node_->now();
        RCLCPP_INFO(
          node_->get_logger(),
          "PlaceObject: Place location at %.2fm, starting final approach to get within %.2fm",
          detected_depth_, MIN_APPROACH_DISTANCE);
      } else {
        state_ = PlaceState::PLACING;
        if (approach_done_) {
          RCLCPP_INFO(node_->get_logger(), "PlaceObject: Approach already done, proceeding with place (depth: %.2fm)", detected_depth_);
        } else {
          RCLCPP_INFO(node_->get_logger(), "PlaceObject: Already close enough (%.2fm), starting place...", detected_depth_);
        }
      }
      return BT::NodeStatus::RUNNING;
    }

    case PlaceState::APPROACHING:
    {
      // Final approach: drive forward until robot is close enough to the place location
      // Use TF to get current robot position and calculate distance to target

      double elapsed = (node_->now() - approach_start_time_).seconds();

      // Safety timeout
      const double MAX_APPROACH_TIME = 15.0;
      if (elapsed > MAX_APPROACH_TIME) {
        geometry_msgs::msg::Twist stop_msg;
        cmd_vel_pub_->publish(stop_msg);

        RCLCPP_WARN(
          node_->get_logger(),
          "PlaceObject: Approach timeout after %.1fs, proceeding with place anyway",
          elapsed);
        state_ = PlaceState::PLACING;
        return BT::NodeStatus::RUNNING;
      }

      // Get current robot position in map frame
      geometry_msgs::msg::TransformStamped robot_transform;
      try {
        robot_transform = tf_buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);
      } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN_THROTTLE(
          node_->get_logger(),
          *node_->get_clock(),
          1000,
          "PlaceObject: Could not get robot position: %s", ex.what());
        // Keep moving if we can't get position
        geometry_msgs::msg::Twist cmd_vel;
        cmd_vel.linear.x = APPROACH_VELOCITY;
        cmd_vel_pub_->publish(cmd_vel);
        return BT::NodeStatus::RUNNING;
      }

      // Calculate distance from robot to place location (in 2D)
      double dx = place_pose_.pose.position.x - robot_transform.transform.translation.x;
      double dy = place_pose_.pose.position.y - robot_transform.transform.translation.y;
      double distance_to_target = std::sqrt(dx * dx + dy * dy);

      RCLCPP_INFO_THROTTLE(
        node_->get_logger(),
        *node_->get_clock(),
        500,
        "PlaceObject: Approaching... distance to target: %.3fm, target: %.2fm",
        distance_to_target, MIN_APPROACH_DISTANCE);

      // Check if we're close enough
      if (distance_to_target <= MIN_APPROACH_DISTANCE) {
        geometry_msgs::msg::Twist stop_msg;
        cmd_vel_pub_->publish(stop_msg);

        RCLCPP_INFO(
          node_->get_logger(),
          "PlaceObject: Approach complete, distance to target: %.3fm",
          distance_to_target);

        state_ = PlaceState::PLACING;
        return BT::NodeStatus::RUNNING;
      }

      // Still need to move closer
      geometry_msgs::msg::Twist cmd_vel;
      cmd_vel.linear.x = APPROACH_VELOCITY;
      cmd_vel_pub_->publish(cmd_vel);

      return BT::NodeStatus::RUNNING;
    }

    case PlaceState::PLACING:
    {
      // Send place request if not sent
      if (!place_sent_) {
        if (!manipulator_client_->wait_for_service(0s)) {
          RCLCPP_WARN_THROTTLE(
            node_->get_logger(),
            *node_->get_clock(),
            2000,
            "PlaceObject: Waiting for /manipulator_action service...");
          return BT::NodeStatus::RUNNING;
        }

        auto request = std::make_shared<btgencobot_interfaces::srv::ManipulatorAction::Request>();
        request->action_type = "place";
        request->target_pose = place_pose_;

        RCLCPP_INFO(
          node_->get_logger(),
          "PlaceObject: Sending place request for pose [%.2f, %.2f, %.2f]",
          place_pose_.pose.position.x,
          place_pose_.pose.position.y,
          place_pose_.pose.position.z);

        manipulator_client_->async_send_request(
          request,
          [this](rclcpp::Client<btgencobot_interfaces::srv::ManipulatorAction>::SharedFuture future) {
            try {
              place_response_ = future.get();
              place_received_ = true;
            } catch (const std::exception & e) {
              RCLCPP_ERROR(node_->get_logger(), "PlaceObject: Place service failed: %s", e.what());
              place_received_ = true;
            }
          });
        place_sent_ = true;
        return BT::NodeStatus::RUNNING;
      }

      // Wait for place response
      if (!place_received_) {
        RCLCPP_INFO_THROTTLE(
          node_->get_logger(),
          *node_->get_clock(),
          2000,
          "PlaceObject: Waiting for place operation to complete...");
        return BT::NodeStatus::RUNNING;
      }

      // Check place result
      if (!place_response_) {
        RCLCPP_ERROR(node_->get_logger(), "PlaceObject: Place service returned null response");
        return BT::NodeStatus::FAILURE;
      }

      if (place_response_->success) {
        RCLCPP_INFO(
          node_->get_logger(),
          "PlaceObject: Place operation completed successfully, starting safety retreat");
        state_ = PlaceState::RETREATING;
        retreat_start_time_ = node_->now();
        retreat_started_ = false;
        return BT::NodeStatus::RUNNING;
      } else {
        RCLCPP_ERROR(
          node_->get_logger(),
          "PlaceObject: Place operation failed: %s",
          place_response_->error_message.c_str());
        return BT::NodeStatus::FAILURE;
      }
    }

    case PlaceState::RETREATING:
    {
      geometry_msgs::msg::TransformStamped robot_transform;
      bool have_pose = false;

      try {
        robot_transform = tf_buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);
        have_pose = true;
      } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN_THROTTLE(
          node_->get_logger(),
          *node_->get_clock(),
          1000,
          "PlaceObject: Could not get robot pose during retreat: %s",
          ex.what());
      }

      if (have_pose && !retreat_started_) {
        retreat_start_x_ = robot_transform.transform.translation.x;
        retreat_start_y_ = robot_transform.transform.translation.y;
        retreat_started_ = true;
      }

      const double elapsed = (node_->now() - retreat_start_time_).seconds();
      double retreated_distance = 0.0;

      if (have_pose && retreat_started_) {
        const double dx = robot_transform.transform.translation.x - retreat_start_x_;
        const double dy = robot_transform.transform.translation.y - retreat_start_y_;
        retreated_distance = std::sqrt(dx * dx + dy * dy);
      }

      if ((retreat_started_ && retreated_distance >= RETREAT_DISTANCE) ||
        elapsed >= MAX_RETREAT_TIME)
      {
        geometry_msgs::msg::Twist stop_msg;
        cmd_vel_pub_->publish(stop_msg);

        if (retreat_started_ && retreated_distance >= RETREAT_DISTANCE) {
          RCLCPP_INFO(
            node_->get_logger(),
            "PlaceObject: Safety retreat complete after %.2fm",
            retreated_distance);
        } else {
          RCLCPP_WARN(
            node_->get_logger(),
            "PlaceObject: Safety retreat timed out after %.1fs, continuing",
            elapsed);
        }

        state_ = PlaceState::DONE;
        return BT::NodeStatus::SUCCESS;
      }

      geometry_msgs::msg::Twist cmd_vel;
      cmd_vel.linear.x = RETREAT_VELOCITY;
      cmd_vel_pub_->publish(cmd_vel);

      RCLCPP_INFO_THROTTLE(
        node_->get_logger(),
        *node_->get_clock(),
        1000,
        "PlaceObject: Retreating from place target... %.2fm / %.2fm",
        retreated_distance,
        RETREAT_DISTANCE);

      return BT::NodeStatus::RUNNING;
    }

    case PlaceState::DONE:
      return BT::NodeStatus::SUCCESS;
  }

  return BT::NodeStatus::FAILURE;
}

void PlaceObject::onHalted()
{
  RCLCPP_INFO(node_->get_logger(), "PlaceObject: Halted");

  // Stop the robot if we were approaching
  if (state_ == PlaceState::APPROACHING || state_ == PlaceState::RETREATING) {
    geometry_msgs::msg::Twist stop_msg;
    cmd_vel_pub_->publish(stop_msg);
  }

  state_ = PlaceState::WAITING_FOR_IMAGE;
  detection_sent_ = false;
  detection_received_ = false;
  detection_response_.reset();
  place_sent_ = false;
  place_received_ = false;
  place_response_.reset();
  retreat_started_ = false;
}

void PlaceObject::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  latest_image_ = msg;
}

void PlaceObject::depthCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  latest_depth_ = msg;
}

void PlaceObject::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
  if (!has_camera_info_) {
    fx_ = msg->k[0];
    fy_ = msg->k[4];
    cx_ = msg->k[2];
    cy_ = msg->k[5];
    has_camera_info_ = true;

    RCLCPP_INFO(
      node_->get_logger(),
      "PlaceObject: Camera calibration received: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f",
      fx_, fy_, cx_, cy_);
  }
}

geometry_msgs::msg::PoseStamped PlaceObject::computePlacePose(
  float center_x,
  float center_y,
  float depth_value,
  const std::string & frame_id)
{
  // Pinhole camera model: pixel to 3D point
  double x = (center_x - cx_) * depth_value / fx_;
  double y = (center_y - cy_) * depth_value / fy_;
  double z = depth_value;

  // Create pose in camera optical frame
  geometry_msgs::msg::PoseStamped pose_camera;
  pose_camera.header.frame_id = frame_id;
  pose_camera.header.stamp = node_->now();
  pose_camera.pose.position.x = x;
  pose_camera.pose.position.y = y;
  pose_camera.pose.position.z = z;
  pose_camera.pose.orientation.w = 1.0;

  RCLCPP_INFO(
    node_->get_logger(),
    "PlaceObject: Place location in camera frame: (%.3f, %.3f, %.3f)",
    x, y, z);

  // Transform to map frame
  try {
    geometry_msgs::msg::PoseStamped pose_map = tf_buffer_->transform(
      pose_camera,
      "map",
      tf2::durationFromSec(1.0));

    // For placing, add a small height offset above the surface
    const double place_height_offset = 0.05;  // 5cm above detected surface
    pose_map.pose.position.z += place_height_offset;

    RCLCPP_INFO(
      node_->get_logger(),
      "PlaceObject: Place location in map frame: (%.3f, %.3f, %.3f) (with %.2fm height offset)",
      pose_map.pose.position.x,
      pose_map.pose.position.y,
      pose_map.pose.position.z,
      place_height_offset);

    // Set orientation to face robot (for placing)
    geometry_msgs::msg::TransformStamped robot_transform;
    try {
      robot_transform = tf_buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);
      double dx = pose_map.pose.position.x - robot_transform.transform.translation.x;
      double dy = pose_map.pose.position.y - robot_transform.transform.translation.y;
      double yaw = std::atan2(dy, dx);

      const double distance_to_target = std::sqrt(dx * dx + dy * dy);
      if (distance_to_target > 1e-3) {
        pose_map.pose.position.x += (dx / distance_to_target) * PLACE_FORWARD_BIAS_METERS;
        pose_map.pose.position.y += (dy / distance_to_target) * PLACE_FORWARD_BIAS_METERS;
      }
      
      tf2::Quaternion q;
      q.setRPY(0, 0, yaw);
      pose_map.pose.orientation.x = q.x();
      pose_map.pose.orientation.y = q.y();
      pose_map.pose.orientation.z = q.z();
      pose_map.pose.orientation.w = q.w();

      RCLCPP_INFO(
        node_->get_logger(),
        "PlaceObject: Applied %.2fm forward place bias, adjusted map pose: (%.3f, %.3f, %.3f)",
        PLACE_FORWARD_BIAS_METERS,
        pose_map.pose.position.x,
        pose_map.pose.position.y,
        pose_map.pose.position.z);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(node_->get_logger(), "Could not get robot pose for orientation: %s", ex.what());
    }

    return pose_map;

  } catch (const tf2::TransformException & ex) {
    RCLCPP_ERROR(node_->get_logger(), "TF transform failed: %s", ex.what());
    return pose_camera;
  }
}

}  // namespace bt_nav2_plugins
