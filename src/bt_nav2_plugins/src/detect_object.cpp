#include "bt_nav2_plugins/detect_object.hpp"

#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <cmath>
#include "cv_bridge/cv_bridge.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

using namespace std::chrono_literals;

namespace bt_nav2_plugins
{

DetectObject::DetectObject(
  const std::string & name,
  const BT::NodeConfiguration & config)
: BT::StatefulActionNode(name, config),
  has_camera_info_(false),
  service_call_sent_(false),
  response_received_(false),
  box_threshold_(0.35)
{
  // Get ROS node from blackboard (shared with Nav2) for logging
  if (!config.blackboard->get("node", node_) || !node_) {
    throw BT::RuntimeError("DetectObject: 'node' not found in blackboard");
  }

  // Create a separate node for subscriptions and service calls
  // We spin this ourselves to ensure callbacks are processed
  sub_node_ = std::make_shared<rclcpp::Node>("detect_object_sub_node");
  
  // Reuse Nav2's long-lived TF buffer. A buffer created when this BT node is
  // instantiated has no transform history yet, while Florence-2 processes an
  // image captured before that short-lived buffer received its first TF data.
  if (!config.blackboard->get("tf_buffer", tf_buffer_) || !tf_buffer_) {
    throw BT::RuntimeError("DetectObject: 'tf_buffer' not found in blackboard");
  }

  RCLCPP_INFO(node_->get_logger(), "DetectObject BT node initialized");

  // Create service client on our own node so we can spin it ourselves
  detect_client_ = sub_node_->create_client<btgencobot_interfaces::srv::DetectObject>("/detect_object");

  // Use sensor data QoS for camera topics (best effort, volatile)
  auto sensor_qos = rclcpp::SensorDataQoS();

  // Subscribe to camera topics on our own node
  image_sub_ = sub_node_->create_subscription<sensor_msgs::msg::Image>(
    "/camera",
    sensor_qos,
    std::bind(&DetectObject::imageCallback, this, std::placeholders::_1));

  depth_sub_ = sub_node_->create_subscription<sensor_msgs::msg::Image>(
    "/camera/depth",
    sensor_qos,
    std::bind(&DetectObject::depthCallback, this, std::placeholders::_1));

  camera_info_sub_ = sub_node_->create_subscription<sensor_msgs::msg::CameraInfo>(
    "/camera/camera_info",
    sensor_qos,
    std::bind(&DetectObject::cameraInfoCallback, this, std::placeholders::_1));

  RCLCPP_INFO(node_->get_logger(), "Subscribed to /camera, /camera/depth, /camera/camera_info");
}

BT::NodeStatus DetectObject::onStart()
{
  // Get input ports
  if (!getInput<std::string>("object_description", object_description_)) {
    RCLCPP_ERROR(node_->get_logger(), "Missing required input 'object_description'");
    return BT::NodeStatus::FAILURE;
  }

  // Get detection threshold (with default)
  if (!getInput<double>("box_threshold", box_threshold_)) {
    box_threshold_ = 0.35;  // default
  }

  RCLCPP_INFO(
    node_->get_logger(),
    "Starting object detection for: '%s' (threshold: %.2f)",
    object_description_.c_str(),
    box_threshold_);

  // Reset state for fresh detection
  service_call_sent_ = false;
  response_received_ = false;
  detection_response_.reset();
  latest_image_.reset();
  latest_depth_.reset();
  has_camera_info_ = false;
  
  // Record the time when detection started - we'll only use images captured after this
  detection_start_time_ = node_->now();
  
  RCLCPP_INFO(
    node_->get_logger(),
    "Detection start time: %.3f - will wait for fresh image",
    detection_start_time_.seconds());
  
  // Log subscription status
  RCLCPP_INFO(
    node_->get_logger(),
    "Image subscription active: %s, count: %zu",
    image_sub_ ? "yes" : "no",
    image_sub_ ? image_sub_->get_publisher_count() : 0);

  return BT::NodeStatus::RUNNING;
}

BT::NodeStatus DetectObject::onRunning()
{
  // Spin our subscription node to process camera callbacks
  rclcpp::spin_some(sub_node_);

  // Check if we have image data
  if (!latest_image_) {
    RCLCPP_WARN_THROTTLE(
      node_->get_logger(),
      *node_->get_clock(),
      1000,
      "Waiting for camera image...");
    return BT::NodeStatus::RUNNING;
  }
  
  // Check if the image is fresh (captured after detection started)
  // This ensures we don't use stale images from before a spin action completed
  rclcpp::Time image_time(latest_image_->header.stamp);
  if (image_time < detection_start_time_) {
    RCLCPP_INFO_THROTTLE(
      node_->get_logger(),
      *node_->get_clock(),
      500,
      "Discarding stale image (%.3f < %.3f), waiting for fresh image...",
      image_time.seconds(),
      detection_start_time_.seconds());
    latest_image_.reset();
    // Don't reset depth here - let it accumulate independently
    return BT::NodeStatus::RUNNING;
  }

  // Warn if no camera info (but continue with defaults)
  if (!has_camera_info_) {
    RCLCPP_WARN_ONCE(
      node_->get_logger(),
      "No camera_info received. Using default calibration values. "
      "3D pose estimation may be inaccurate!");
    // Set default values (typical for RGB cameras)
    fx_ = 554.3;
    fy_ = 554.3;
    cx_ = 320.5;
    cy_ = 240.5;
  }

  // If service call not sent yet, send it
  if (!service_call_sent_) {
    // Check if service is available
    if (!detect_client_->wait_for_service(0s)) {
      RCLCPP_WARN_THROTTLE(
        node_->get_logger(),
        *node_->get_clock(),
        2000,
        "Waiting for /detect_object service...");
      return BT::NodeStatus::RUNNING;
    }

    // Create service request
    auto request = std::make_shared<btgencobot_interfaces::srv::DetectObject::Request>();
    request->image = *latest_image_;
    request->object_description = object_description_;
    request->box_threshold = static_cast<float>(box_threshold_);

    RCLCPP_INFO(node_->get_logger(), "Sending detection request to service...");

    // Send async request with callback
    detect_client_->async_send_request(
      request,
      [this](rclcpp::Client<btgencobot_interfaces::srv::DetectObject>::SharedFuture future) {
        try {
          detection_response_ = future.get();
          response_received_ = true;
          RCLCPP_INFO(node_->get_logger(), "Detection response received via callback");
        } catch (const std::exception & e) {
          RCLCPP_ERROR(node_->get_logger(), "Service call failed: %s", e.what());
          response_received_ = true;  // Mark as received so we can handle the error
        }
      });
    service_call_sent_ = true;

    return BT::NodeStatus::RUNNING;
  }

  // Check if service response is ready
  if (!response_received_) {
    RCLCPP_INFO_THROTTLE(
      node_->get_logger(),
      *node_->get_clock(),
      2000,
      "Waiting for detection service response...");
    return BT::NodeStatus::RUNNING;
  }

  // Get service response
  auto response = detection_response_;
  
  // Check if we got a valid response
  if (!response) {
    RCLCPP_ERROR(node_->get_logger(), "Detection service returned null response");
    setOutput("detected", false);
    setOutput("confidence", 0.0);
    return BT::NodeStatus::FAILURE;
  }

  // Check if detection was successful
  if (!response->detected) {
    RCLCPP_WARN(
      node_->get_logger(),
      "Object '%s' not detected: %s",
      object_description_.c_str(),
      response->error_message.c_str());

    setOutput("detected", false);
    setOutput("confidence", 0.0);
    return BT::NodeStatus::FAILURE;
  }

  // Improved depth sampling: find the object (closest surface) and compute depth-weighted centroid
  float depth = 1.5;  // Default if no depth available
  float refined_center_x = response->center_x;
  float refined_center_y = response->center_y;

  if (latest_depth_) {
    try {
      cv_bridge::CvImagePtr depth_ptr = cv_bridge::toCvCopy(
        latest_depth_,
        sensor_msgs::image_encodings::TYPE_32FC1);

      // Structure to hold depth samples with their pixel coordinates
      struct DepthSample {
        float depth;
        int x;
        int y;
      };
      std::vector<DepthSample> depth_samples;

      // Sample depth from bounding box
      if (response->bbox.size() >= 4) {
        int x1 = static_cast<int>(response->bbox[0]);
        int y1 = static_cast<int>(response->bbox[1]);
        int x2 = static_cast<int>(response->bbox[2]);
        int y2 = static_cast<int>(response->bbox[3]);

        // Clamp to image bounds
        x1 = std::max(0, std::min(x1, depth_ptr->image.cols - 1));
        y1 = std::max(0, std::min(y1, depth_ptr->image.rows - 1));
        x2 = std::max(0, std::min(x2, depth_ptr->image.cols - 1));
        y2 = std::max(0, std::min(y2, depth_ptr->image.rows - 1));

        // Sample from inner region of bounding box (80% center area to avoid edges)
        int margin_x = (x2 - x1) * 0.10;
        int margin_y = (y2 - y1) * 0.10;
        int inner_x1 = x1 + margin_x;
        int inner_y1 = y1 + margin_y;
        int inner_x2 = x2 - margin_x;
        int inner_y2 = y2 - margin_y;

        RCLCPP_INFO(
          node_->get_logger(),
          "Sampling depth from bounding box (%d,%d)-(%d,%d), inner region (%d,%d)-(%d,%d)",
          x1, y1, x2, y2, inner_x1, inner_y1, inner_x2, inner_y2);

        // Sample every few pixels for efficiency
        int sample_stride = 2;
        for (int y = inner_y1; y <= inner_y2; y += sample_stride) {
          for (int x = inner_x1; x <= inner_x2; x += sample_stride) {
            float d = depth_ptr->image.at<float>(y, x);
            if (!std::isnan(d) && d > 0.1 && d < 10.0) {
              depth_samples.push_back({d, x, y});
            }
          }
        }
      }

      if (!depth_samples.empty()) {
        // Sort by depth to find the closest cluster (the actual object, not background)
        std::sort(depth_samples.begin(), depth_samples.end(),
          [](const DepthSample& a, const DepthSample& b) { return a.depth < b.depth; });

        float min_depth = depth_samples.front().depth;
        float max_depth = depth_samples.back().depth;
        float depth_range = max_depth - min_depth;

        RCLCPP_INFO(
          node_->get_logger(),
          "Sampled %zu depth points, range: %.2f - %.2fm (span: %.2fm)",
          depth_samples.size(), min_depth, max_depth, depth_range);

        // Use depth clustering: object is the closest surface
        // Accept points within a tolerance of the minimum depth
        // Use adaptive threshold based on depth (closer objects need tighter threshold)
        float depth_tolerance = std::max(0.05f, min_depth * 0.15f);  // 15% of min depth, at least 5cm
        
        // If depth range is small, object fills the bbox - use all samples
        if (depth_range < 0.3) {
          depth_tolerance = depth_range + 0.05f;
        }

        // Collect object samples (closest cluster) and compute geometric centroid
        std::vector<DepthSample> object_samples;
        double sum_x = 0.0, sum_y = 0.0;
        
        for (const auto& sample : depth_samples) {
          if (sample.depth <= min_depth + depth_tolerance) {
            object_samples.push_back(sample);
            // Use unweighted sum for geometric center (symmetric grasp)
            sum_x += sample.x;
            sum_y += sample.y;
          }
        }

        if (!object_samples.empty()) {
          // Compute geometric centroid of object cluster
          refined_center_x = static_cast<float>(sum_x / object_samples.size());
          refined_center_y = static_cast<float>(sum_y / object_samples.size());

          // Use median of object samples for robust depth
          std::sort(object_samples.begin(), object_samples.end(),
            [](const DepthSample& a, const DepthSample& b) { return a.depth < b.depth; });
          depth = object_samples[object_samples.size() / 2].depth;

          RCLCPP_INFO(
            node_->get_logger(),
            "Object cluster: %zu/%zu samples within %.2fm of min depth",
            object_samples.size(), depth_samples.size(), depth_tolerance);
          RCLCPP_INFO(
            node_->get_logger(),
            "Refined center: (%.1f, %.1f) -> (%.1f, %.1f), depth: %.2fm",
            response->center_x, response->center_y, refined_center_x, refined_center_y, depth);
        } else {
          // Fallback to 25th percentile (front quarter of samples)
          size_t idx = depth_samples.size() / 4;
          depth = depth_samples[idx].depth;
          RCLCPP_WARN(
            node_->get_logger(),
            "No object cluster found, using 25th percentile depth: %.2fm", depth);
        }
      } else {
        RCLCPP_WARN(
          node_->get_logger(),
          "No valid depth samples from bounding box, using default 1.5m");
      }

    } catch (const std::exception & e) {
      RCLCPP_WARN(
        node_->get_logger(),
        "Failed to sample depth: %s. Using default 1.5m", e.what());
    }
  } else {
    RCLCPP_WARN_ONCE(
      node_->get_logger(),
      "No depth data available. Using default 1.5m");
  }

  // Use camera optical frame for pose estimation
  std::string camera_frame = "camera_rgb_optical_frame";

  geometry_msgs::msg::PoseStamped target_pose;
  try {
    target_pose = pixelToPose(
      refined_center_x,
      refined_center_y,
      depth,
      camera_frame);
  } catch (const tf2::TransformException & ex) {
    RCLCPP_ERROR(
      node_->get_logger(),
      "Cannot compute navigation approach pose in map frame: %s",
      ex.what());
    return BT::NodeStatus::FAILURE;
  }

  RCLCPP_INFO(
    node_->get_logger(),
    "Detected '%s' at bbox center (%.1f, %.1f) -> refined (%.1f, %.1f), depth %.2fm",
    response->phrase.c_str(),
    response->center_x, response->center_y,
    refined_center_x, refined_center_y,
    depth);
  RCLCPP_INFO(
    node_->get_logger(),
    "Approach pose in map: (%.2f, %.2f) (conf: %.2f)",
    target_pose.pose.position.x,
    target_pose.pose.position.y,
    response->confidence);

  // Set output ports (only approach pose for navigation - object pose computed by PickObject/PlaceObject)
  setOutput("target_pose", target_pose);
  setOutput("detected", true);
  setOutput("confidence", static_cast<double>(response->confidence));

  return BT::NodeStatus::SUCCESS;
}

void DetectObject::onHalted()
{
  RCLCPP_INFO(node_->get_logger(), "DetectObject node halted");
  service_call_sent_ = false;
  response_received_ = false;
  detection_response_.reset();
}

void DetectObject::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  latest_image_ = msg;
}

void DetectObject::depthCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  latest_depth_ = msg;
}

void DetectObject::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
  if (!has_camera_info_) {
    // Extract camera intrinsics from K matrix
    // K = [fx  0  cx]
    //     [ 0 fy  cy]
    //     [ 0  0   1]
    fx_ = msg->k[0];
    fy_ = msg->k[4];
    cx_ = msg->k[2];
    cy_ = msg->k[5];

    has_camera_info_ = true;

    RCLCPP_INFO(
      node_->get_logger(),
      "Camera calibration received: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f",
      fx_, fy_, cx_, cy_);
  }
}

geometry_msgs::msg::PoseStamped DetectObject::pixelToPose(
  float center_x,
  float center_y,
  float depth_value,
  const std::string & frame_id)
{
  // Pinhole camera model: pixel to 3D point in optical frame
  // Optical frame: +X=right, +Y=down, +Z=forward
  double x = (center_x - cx_) * depth_value / fx_;
  double y = (center_y - cy_) * depth_value / fy_;
  double z = depth_value;

  // Create pose in camera optical frame
  geometry_msgs::msg::PoseStamped pose_camera;
  pose_camera.header.frame_id = frame_id;
  // Use the latest available transform to avoid extrapolation into the past
  // when Gazebo runs below real-time and the TF buffer lags node_->now().
  pose_camera.header.stamp = rclcpp::Time(0);
  pose_camera.pose.position.x = x;
  pose_camera.pose.position.y = y;
  pose_camera.pose.position.z = z;
  pose_camera.pose.orientation.w = 1.0;

  RCLCPP_INFO(
    node_->get_logger(),
    "Object in camera frame: (%.2f, %.2f, %.2f) at depth %.2fm",
    x, y, z, depth_value);

  // Transform to map frame
  try {
    geometry_msgs::msg::PoseStamped pose_map = tf_buffer_->transform(
      pose_camera,
      "map",
      tf2::durationFromSec(1.0));

    // Store original object position
    double obj_x = pose_map.pose.position.x;
    double obj_y = pose_map.pose.position.y;
    
    // Get robot's current position in map frame
    const auto robot_transform =
      tf_buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);
    const double robot_x = robot_transform.transform.translation.x;
    const double robot_y = robot_transform.transform.translation.y;
    RCLCPP_INFO(
      node_->get_logger(),
      "Robot position in map: (%.2f, %.2f)", robot_x, robot_y);
    
    // Calculate direction vector from robot to object
    double dx = obj_x - robot_x;
    double dy = obj_y - robot_y;
    double distance_to_object = std::sqrt(dx * dx + dy * dy);
    
    RCLCPP_INFO(
      node_->get_logger(),
      "Object at (%.2f, %.2f), distance from robot: %.2fm",
      obj_x, obj_y, distance_to_object);
    
    // Calculate orientation: face toward object from robot position
    double yaw = std::atan2(dy, dx);
    tf2::Quaternion q;
    q.setRPY(0, 0, yaw);

    // Calculate approach pose (for navigation)
    // Project to ground plane (z=0 for 2D navigation)
    pose_map.pose.position.z = 0.0;
    
    // Set navigation goal at a distance where the camera can still see the object clearly.
    // The pick/place nodes will do fine adjustments using direct cmd_vel to get within arm reach.
    // This offset should be large enough that:
    // - The object remains visible and recognizable in the camera frame
    // - The robot has room to re-detect and adjust during pick/place
    const double approach_offset = 0.45;  // Nav2 goal 45cm from object
    
    if (distance_to_object > approach_offset) {
      // Normalize direction vector and place goal close to object
      double unit_dx = dx / distance_to_object;
      double unit_dy = dy / distance_to_object;
      pose_map.pose.position.x = obj_x - unit_dx * approach_offset;
      pose_map.pose.position.y = obj_y - unit_dy * approach_offset;
    } else {
      // Already very close, just use object position
      pose_map.pose.position.x = obj_x;
      pose_map.pose.position.y = obj_y;
    }

    // Face toward object
    pose_map.pose.orientation.x = q.x();
    pose_map.pose.orientation.y = q.y();
    pose_map.pose.orientation.z = q.z();
    pose_map.pose.orientation.w = q.w();

    RCLCPP_INFO(
      node_->get_logger(),
      "Navigation goal: (%.2f, %.2f) in map, object at (%.2f, %.2f)",
      pose_map.pose.position.x, pose_map.pose.position.y, obj_x, obj_y);

    return pose_map;

  } catch (const tf2::TransformException & ex) {
    RCLCPP_ERROR(
      node_->get_logger(),
      "TF transform failed: %s", ex.what());
    // Never return camera-frame coordinates as if they were already in map.
    throw;
  }
}

}  // namespace bt_nav2_plugins
