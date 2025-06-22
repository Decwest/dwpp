#ifndef NAV2_DWPP_CONTROLLER__DWPP_CONTROLLER_HPP_
#define NAV2_DWPP_CONTROLLER__DWPP_CONTROLLER_HPP_

#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <mutex>

#include "nav2_regulated_pure_pursuit_controller/regulated_pure_pursuit_controller.hpp"
#include "nav2_core/controller.hpp"
#include "rclcpp/rclcpp.hpp"
#include "pluginlib/class_loader.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "nav2_util/odometry_utils.hpp"
#include "nav2_util/geometry_utils.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"

namespace nav2_dwpp_controller
{

class DWPPController : public nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController
{
public:
  DWPPController() = default;
  ~DWPPController() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * /*goal_checker*/) override;

  void setPlan(const nav_msgs::msg::Path & path) override;

  void setSpeedLimit(const double & speed_limit, const bool & percentage) override;

protected:
  struct DynamicWindow {
    double min_vel_x;
    double max_vel_x;
    double min_vel_theta;
    double max_vel_theta;
  };

  struct VelocityCandidate {
    double vel_x;
    double vel_theta;
    double cost;
  };

  DynamicWindow calculateDynamicWindow(
    const geometry_msgs::msg::Twist & current_velocity);

  std::vector<VelocityCandidate> generateVelocityCandidates(
    const DynamicWindow & dw,
    const double & curvature,
    const double & regulated_linear_vel);

  VelocityCandidate selectOptimalVelocity(
    const std::vector<VelocityCandidate> & candidates,
    const bool & should_accelerate);

  bool shouldAccelerate(
    const geometry_msgs::msg::PoseStamped & pose);

  double calculateGoalDistance(
    const geometry_msgs::msg::PoseStamped & pose);

  std::pair<double, double> calculateIntersectionWithDynamicWindow(
    const DynamicWindow & dw,
    const double & curvature);

  double calculateDistanceFromCurvatureLine(
    const double & vel_x,
    const double & vel_theta,
    const double & curvature);

  // DWPP-specific parameters
  double max_linear_accel_;
  double max_angular_accel_;
  double velocity_sample_resolution_;
  bool use_dynamic_window_;
  double deceleration_distance_factor_;

  rclcpp_lifecycle::LifecycleNode::WeakPtr parent_;
  std::string plugin_name_;
};

}  // namespace nav2_dwpp_controller

#endif  // NAV2_DWPP_CONTROLLER__DWPP_CONTROLLER_HPP_