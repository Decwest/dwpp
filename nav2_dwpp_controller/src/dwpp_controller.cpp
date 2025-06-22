#include <algorithm>
#include <string>
#include <limits>
#include <memory>
#include <vector>
#include <utility>

#include "nav2_dwpp_controller/dwpp_controller.hpp"
#include "nav2_core/exceptions.hpp"
#include "nav2_util/node_utils.hpp"
#include "nav2_util/geometry_utils.hpp"

using std::hypot;
using std::max;
using std::min;
using std::abs;
using nav2_util::declare_parameter_if_not_declared;
using nav2_util::geometry_utils::euclidean_distance;

namespace nav2_dwpp_controller
{

void DWPPController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  // Call parent configure first
  nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController::configure(
    parent, name, tf, costmap_ros);

  parent_ = parent;
  plugin_name_ = name;
  
  auto node = parent_.lock();

  // Declare DWPP-specific parameters
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".max_linear_accel", rclcpp::ParameterValue(0.5));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".max_angular_accel", rclcpp::ParameterValue(1.0));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".velocity_sample_resolution", rclcpp::ParameterValue(0.1));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".use_dynamic_window", rclcpp::ParameterValue(true));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".deceleration_distance_factor", rclcpp::ParameterValue(2.0));

  // Get DWPP parameters
  node->get_parameter(plugin_name_ + ".max_linear_accel", max_linear_accel_);
  node->get_parameter(plugin_name_ + ".max_angular_accel", max_angular_accel_);
  node->get_parameter(plugin_name_ + ".velocity_sample_resolution", velocity_sample_resolution_);
  node->get_parameter(plugin_name_ + ".use_dynamic_window", use_dynamic_window_);
  node->get_parameter(plugin_name_ + ".deceleration_distance_factor", deceleration_distance_factor_);

  RCLCPP_INFO(
    logger_, "DWPP Controller configured with max_linear_accel: %.2f, max_angular_accel: %.2f",
    max_linear_accel_, max_angular_accel_);
}

void DWPPController::cleanup()
{
  nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController::cleanup();
}

void DWPPController::activate()
{
  nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController::activate();
}

void DWPPController::deactivate()
{
  nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController::deactivate();
}

void DWPPController::setPlan(const nav_msgs::msg::Path & path)
{
  nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController::setPlan(path);
}

void DWPPController::setSpeedLimit(const double & speed_limit, const bool & percentage)
{
  nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController::setSpeedLimit(
    speed_limit, percentage);
}

geometry_msgs::msg::TwistStamped DWPPController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & velocity,
  nav2_core::GoalChecker * goal_checker)
{
  std::lock_guard<std::mutex> lock_reinit(mutex_);

  // If dynamic window is disabled, use parent RPP controller
  if (!use_dynamic_window_) {
    return nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController::computeVelocityCommands(
      pose, velocity, goal_checker);
  }

  // Get basic RPP computation (curvature, look-ahead point, etc.)
  geometry_msgs::msg::TwistStamped parent_cmd = 
    nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController::computeVelocityCommands(
      pose, velocity, goal_checker);

  // If parent returns zero velocity (goal reached, obstacle, etc.), return as is
  if (abs(parent_cmd.twist.linear.x) < 0.01 && abs(parent_cmd.twist.angular.z) < 0.01) {
    return parent_cmd;
  }

  // Calculate dynamic window
  DynamicWindow dw = calculateDynamicWindow(velocity);

  // Get current curvature from parent computation
  double curvature = 0.0;
  if (abs(parent_cmd.twist.linear.x) > 0.01) {
    curvature = parent_cmd.twist.angular.z / parent_cmd.twist.linear.x;
  }

  // Get regulated linear velocity from parent
  double regulated_linear_vel = parent_cmd.twist.linear.x;

  // Generate velocity candidates within dynamic window
  std::vector<VelocityCandidate> candidates = generateVelocityCandidates(
    dw, curvature, regulated_linear_vel);

  // Determine if we should accelerate or decelerate
  bool should_accelerate = shouldAccelerate(pose);

  // Select optimal velocity
  VelocityCandidate optimal_vel = selectOptimalVelocity(candidates, should_accelerate);

  // Create output command
  geometry_msgs::msg::TwistStamped cmd_vel = parent_cmd;
  cmd_vel.twist.linear.x = optimal_vel.vel_x;
  cmd_vel.twist.angular.z = optimal_vel.vel_theta;

  return cmd_vel;
}

DWPPController::DynamicWindow DWPPController::calculateDynamicWindow(
  const geometry_msgs::msg::Twist & current_velocity)
{
  DynamicWindow dw;
  
  // Linear velocity bounds
  dw.min_vel_x = std::max(
    current_velocity.linear.x - max_linear_accel_ * control_duration_,
    min_vel_x_);
  dw.max_vel_x = std::min(
    current_velocity.linear.x + max_linear_accel_ * control_duration_,
    max_vel_x_);

  // Angular velocity bounds
  dw.min_vel_theta = std::max(
    current_velocity.angular.z - max_angular_accel_ * control_duration_,
    -max_vel_theta_);
  dw.max_vel_theta = std::min(
    current_velocity.angular.z + max_angular_accel_ * control_duration_,
    max_vel_theta_);

  return dw;
}

std::vector<DWPPController::VelocityCandidate> DWPPController::generateVelocityCandidates(
  const DynamicWindow & dw,
  const double & curvature,
  const double & regulated_linear_vel)
{
  std::vector<VelocityCandidate> candidates;

  // Intersection points with curvature line
  auto intersection = calculateIntersectionWithDynamicWindow(dw, curvature);
  
  // Add intersection points as candidates
  if (intersection.first >= dw.min_vel_x && intersection.first <= dw.max_vel_x) {
    double vel_theta = curvature * intersection.first;
    if (vel_theta >= dw.min_vel_theta && vel_theta <= dw.max_vel_theta) {
      candidates.push_back({intersection.first, vel_theta, 0.0});
    }
  }
  
  if (intersection.second >= dw.min_vel_x && intersection.second <= dw.max_vel_x) {
    double vel_theta = curvature * intersection.second;
    if (vel_theta >= dw.min_vel_theta && vel_theta <= dw.max_vel_theta) {
      candidates.push_back({intersection.second, vel_theta, 0.0});
    }
  }

  // If curvature is non-zero, add angular velocity boundary intersections
  if (abs(curvature) > 1e-6) {
    double vel_x_min = dw.min_vel_theta / curvature;
    double vel_x_max = dw.max_vel_theta / curvature;
    
    if (vel_x_min >= dw.min_vel_x && vel_x_min <= dw.max_vel_x) {
      candidates.push_back({vel_x_min, dw.min_vel_theta, 0.0});
    }
    if (vel_x_max >= dw.min_vel_x && vel_x_max <= dw.max_vel_x) {
      candidates.push_back({vel_x_max, dw.max_vel_theta, 0.0});
    }
  }

  // If no valid intersection points, find closest points on DW boundary
  if (candidates.empty()) {
    std::vector<std::pair<double, double>> dw_corners = {
      {dw.min_vel_x, dw.min_vel_theta},
      {dw.min_vel_x, dw.max_vel_theta},
      {dw.max_vel_x, dw.min_vel_theta},
      {dw.max_vel_x, dw.max_vel_theta}
    };

    double min_distance = std::numeric_limits<double>::max();
    for (const auto & corner : dw_corners) {
      double distance = calculateDistanceFromCurvatureLine(corner.first, corner.second, curvature);
      if (distance < min_distance) {
        min_distance = distance;
      }
    }

    // Add all corners with minimum distance
    for (const auto & corner : dw_corners) {
      double distance = calculateDistanceFromCurvatureLine(corner.first, corner.second, curvature);
      if (abs(distance - min_distance) < 1e-6) {
        candidates.push_back({corner.first, corner.second, distance});
      }
    }
  }

  // Apply regulated velocity constraint
  for (auto & candidate : candidates) {
    if (candidate.vel_x > regulated_linear_vel) {
      candidate.vel_x = std::max(dw.min_vel_x, regulated_linear_vel);
      candidate.vel_theta = curvature * candidate.vel_x;
    }
  }

  return candidates;
}

DWPPController::VelocityCandidate DWPPController::selectOptimalVelocity(
  const std::vector<VelocityCandidate> & candidates,
  const bool & should_accelerate)
{
  if (candidates.empty()) {
    return {0.0, 0.0, 0.0};
  }

  // Sort candidates by linear velocity
  auto sorted_candidates = candidates;
  std::sort(sorted_candidates.begin(), sorted_candidates.end(),
    [](const VelocityCandidate & a, const VelocityCandidate & b) {
      return a.vel_x < b.vel_x;
    });

  // Select based on acceleration/deceleration preference
  if (should_accelerate) {
    return sorted_candidates.back();  // Highest velocity
  } else {
    return sorted_candidates.front(); // Lowest velocity
  }
}

bool DWPPController::shouldAccelerate(const geometry_msgs::msg::PoseStamped & pose)
{
  double goal_distance = calculateGoalDistance(pose);
  double deceleration_distance = (max_vel_x_ * max_vel_x_) / (2.0 * max_linear_accel_);
  
  return goal_distance > deceleration_distance * deceleration_distance_factor_;
}

double DWPPController::calculateGoalDistance(const geometry_msgs::msg::PoseStamped & pose)
{
  if (global_plan_.poses.empty()) {
    return 0.0;
  }

  const auto & goal_pose = global_plan_.poses.back();
  return euclidean_distance(pose.pose.position, goal_pose.pose.position);
}

std::pair<double, double> DWPPController::calculateIntersectionWithDynamicWindow(
  const DynamicWindow & dw,
  const double & curvature)
{
  // Calculate intersection of curvature line with DW linear velocity bounds
  double vel_x_at_min_theta = (abs(curvature) > 1e-6) ? dw.min_vel_theta / curvature : dw.min_vel_x;
  double vel_x_at_max_theta = (abs(curvature) > 1e-6) ? dw.max_vel_theta / curvature : dw.max_vel_x;

  return {std::min(vel_x_at_min_theta, vel_x_at_max_theta),
          std::max(vel_x_at_min_theta, vel_x_at_max_theta)};
}

double DWPPController::calculateDistanceFromCurvatureLine(
  const double & vel_x,
  const double & vel_theta,
  const double & curvature)
{
  if (abs(curvature) < 1e-6) {
    return abs(vel_theta);
  }
  
  // Distance from point to line: |ax + by + c| / sqrt(a^2 + b^2)
  // Line equation: curvature * vel_x - vel_theta = 0
  return abs(curvature * vel_x - vel_theta) / sqrt(curvature * curvature + 1.0);
}

}  // namespace nav2_dwpp_controller

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(nav2_dwpp_controller::DWPPController, nav2_core::Controller)