#!/usr/bin/env python3

import rospy
import numpy as np
import math
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, Odometry
from tf.transformations import euler_from_quaternion
from typing import Tuple, List, Optional

class PurePursuitController:
    def __init__(self):
        rospy.init_node('pure_pursuit_controller', anonymous=True)
        
        # Parameters
        self.MIN_LOOK_AHEAD_DISTANCE = rospy.get_param('~min_look_ahead_distance', 0.4)
        self.MAX_LOOK_AHEAD_DISTANCE = rospy.get_param('~max_look_ahead_distance', 0.5)
        self.LOOK_AHEAD_TIME = rospy.get_param('~look_ahead_time', 1.0)
        self.V_MAX = rospy.get_param('~max_linear_velocity', 0.5)
        self.V_MIN = rospy.get_param('~min_linear_velocity', 0.0)
        self.W_MAX = rospy.get_param('~max_angular_velocity', 1.0)
        self.W_MIN = rospy.get_param('~min_angular_velocity', -1.0)
        self.A_MAX = rospy.get_param('~max_linear_acceleration', 0.5)
        self.AW_MAX = rospy.get_param('~max_angular_acceleration', 1.0)
        self.GOAL_TOLERANCE_DIST = rospy.get_param('~goal_tolerance_distance', 0.05)
        self.APPROACH_VELOCITY_SCALING_DIST = rospy.get_param('~approach_velocity_scaling_distance', 0.3)
        self.MIN_APPROACH_LINEAR_VELOCITY = rospy.get_param('~min_approach_linear_velocity', 0.05)
        self.REGULATED_LINEAR_SCALING_MIN_RADIUS = rospy.get_param('~regulated_linear_scaling_min_radius', 0.9)
        self.REGULATED_LINEAR_SCALING_MIN_SPEED = rospy.get_param('~regulated_linear_scaling_min_speed', 0.0)
        self.CONTROL_FREQUENCY = rospy.get_param('~control_frequency', 20.0)
        self.METHOD_NAME = rospy.get_param('~method_name', 'dwpp')  # pp, app, rpp, dwpp
        
        # State variables
        self.current_pose: Optional[np.ndarray] = None
        self.current_velocity: np.ndarray = np.array([0.0, 0.0])
        self.path: Optional[np.ndarray] = None
        self.goal_reached: bool = False
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('~cmd_vel', Twist, queue_size=1)
        
        # Subscribers
        self.odom_sub = rospy.Subscriber('~odom', Odometry, self.odom_callback)
        self.path_sub = rospy.Subscriber('~path', Path, self.path_callback)
        
        # Validate method name
        valid_methods = ['pp', 'app', 'rpp', 'dwpp']
        if self.METHOD_NAME not in valid_methods:
            rospy.logerr(f"Invalid method name: {self.METHOD_NAME}. Valid options: {valid_methods}")
            rospy.signal_shutdown("Invalid method name")
            return
        
        # Control timer
        self.dt = 1.0 / self.CONTROL_FREQUENCY
        self.control_timer = rospy.Timer(rospy.Duration(self.dt), self.control_callback)
        
        rospy.loginfo(f"Pure Pursuit Controller initialized with method: {self.METHOD_NAME}")

    def odom_callback(self, msg: Odometry) -> None:
        """Callback for odometry messages"""
        # Extract position
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        
        # Convert quaternion to euler angles
        _, _, yaw = euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])
        
        self.current_pose = np.array([position.x, position.y, yaw])
        
        # Extract velocity
        linear_vel = msg.twist.twist.linear.x
        angular_vel = msg.twist.twist.angular.z
        self.current_velocity = np.array([linear_vel, angular_vel])

    def path_callback(self, msg: Path) -> None:
        """Callback for path messages"""
        if len(msg.poses) == 0:
            rospy.logwarn("Received empty path")
            return
            
        path_points = []
        for pose_stamped in msg.poses:
            position = pose_stamped.pose.position
            path_points.append([position.x, position.y])
        
        self.path = np.array(path_points)
        self.goal_reached = False
        rospy.loginfo(f"Received path with {len(path_points)} points")

    def control_callback(self, timer_event) -> None:
        """Main control loop callback"""
        if self.current_pose is None or self.path is None:
            return
            
        if self.goal_reached:
            self.publish_zero_velocity()
            return
            
        # Check if goal is reached
        distance_to_goal = np.linalg.norm(self.path[-1] - self.current_pose[:2])
        if distance_to_goal < self.GOAL_TOLERANCE_DIST:
            self.goal_reached = True
            self.publish_zero_velocity()
            rospy.loginfo("Goal reached!")
            return
            
        # Compute Pure Pursuit control
        try:
            next_velocity_ref = self.pure_pursuit_control(
                self.current_pose, self.current_velocity, self.path, self.METHOD_NAME
            )
            
            # Apply acceleration constraints
            next_velocity = self.calc_accel_constrained_velocity(
                self.current_velocity, next_velocity_ref
            )
            
            # Publish velocity command
            self.publish_velocity(next_velocity)
            
        except Exception as e:
            rospy.logerr(f"Error in control loop: {str(e)}")
            self.publish_zero_velocity()

    def pure_pursuit_control(self, current_pose: np.ndarray, current_velocity: np.ndarray, 
                           path: np.ndarray, method_name: str) -> np.ndarray:
        """Pure Pursuit control algorithm with multiple method support"""
        # Calculate current path index
        current_idx = self.calc_index(current_pose, path)
        
        # Calculate path distances
        path_distances = self.calc_path_distances(path)
        
        # Calculate look ahead distance
        look_ahead_distance = self.calc_look_ahead_distance(current_velocity, method_name)
        
        # Calculate curvature to look ahead position
        curvature, _ = self.calc_curvature_to_look_ahead_position(
            current_pose, current_idx, path, path_distances, look_ahead_distance
        )
        
        if method_name in ["rpp", "dwpp"]:
            # Calculate regulated translational velocity (Regulated Pure Pursuit)
            regulated_v = self.calc_regulated_translational_velocity(curvature)
        else:
            regulated_v = self.V_MAX
        
        if method_name in ["pp", "app", "rpp"]:
            # Calculate translational velocity
            v_ref = self.calc_reference_translational_velocity(current_pose, path[-1])
            
            # Regulate translational velocity for RPP
            if method_name == "rpp":
                v_ref = min(v_ref, regulated_v)
            
            # Calculate angular velocity
            w_ref = curvature * v_ref
            next_velocity_ref = np.array([v_ref, w_ref])
            
        else:  # dwpp
            # Decide acceleration or deceleration
            is_accel = self.decide_accel_or_decel(current_idx, path_distances)
            
            # Calculate optimal velocity considering dynamic window
            next_velocity_ref = self.calc_optimal_velocity_considering_dynamic_window(
                current_velocity, regulated_v, curvature, is_accel
            )
        
        return next_velocity_ref

    def calc_index(self, current_pose: np.ndarray, path: np.ndarray) -> int:
        """Find the closest point index on the path"""
        distances = np.linalg.norm(path - current_pose[:2], axis=1)
        return int(np.argmin(distances))

    def calc_path_distances(self, path: np.ndarray) -> np.ndarray:
        """Calculate cumulative distances along the path"""
        differences = np.diff(path, axis=0)
        distances = np.linalg.norm(differences, axis=1)
        return np.concatenate(([0.0], np.cumsum(distances)))

    def calc_look_ahead_distance(self, current_velocity: np.ndarray, method_name: str) -> float:
        """Calculate look ahead distance based on method"""
        if method_name in ["app", "rpp", "dwpp"]:
            # Adaptive look ahead distance
            look_ahead_distance = self.LOOK_AHEAD_TIME * current_velocity[0]
            return np.clip(look_ahead_distance, 
                          self.MIN_LOOK_AHEAD_DISTANCE, 
                          self.MAX_LOOK_AHEAD_DISTANCE)
        else:  # pp
            # Fixed look ahead distance
            return self.MIN_LOOK_AHEAD_DISTANCE

    def calc_curvature_to_look_ahead_position(self, current_pose: np.ndarray, 
                                            current_idx: int, path: np.ndarray, 
                                            path_distances: np.ndarray, 
                                            look_ahead_distance: float) -> Tuple[float, np.ndarray]:
        """Calculate curvature to the look ahead position"""
        # Find look ahead position
        current_distance = path_distances[current_idx]
        look_ahead_pos_distance = current_distance + look_ahead_distance
        look_ahead_idx = min(np.searchsorted(path_distances, look_ahead_pos_distance), 
                            len(path) - 1)
        look_ahead_pos = path[look_ahead_idx]
        
        # Calculate curvature
        look_ahead_angle = (math.atan2(look_ahead_pos[1] - current_pose[1], 
                                     look_ahead_pos[0] - current_pose[0]) - current_pose[2])
        L = np.linalg.norm(look_ahead_pos - current_pose[:2])
        
        if L == 0:
            curvature = 0.0
        else:
            curvature = 2.0 * math.sin(look_ahead_angle) / L
        
        return curvature, look_ahead_pos

    def calc_reference_translational_velocity(self, current_pose: np.ndarray, goal_pose: np.ndarray) -> float:
        """Calculate reference translational velocity considering approach to goal"""
        v_ref = self.V_MAX
        
        # Reduce velocity when approaching goal
        distance_to_goal = np.linalg.norm(goal_pose - current_pose[:2])
        if distance_to_goal < self.APPROACH_VELOCITY_SCALING_DIST:
            v_ref = max(v_ref * distance_to_goal / self.APPROACH_VELOCITY_SCALING_DIST, 
                       self.MIN_APPROACH_LINEAR_VELOCITY)
        if distance_to_goal < self.GOAL_TOLERANCE_DIST:
            v_ref = 0.0
        
        return v_ref

    def calc_regulated_translational_velocity(self, curvature: float) -> float:
        """Calculate regulated translational velocity based on curvature"""
        if curvature == 0.0:
            return self.V_MAX
        
        curvature_radius = 1.0 / abs(curvature)
        if curvature_radius <= self.REGULATED_LINEAR_SCALING_MIN_RADIUS:
            regulated_v = self.V_MAX * curvature_radius / self.REGULATED_LINEAR_SCALING_MIN_RADIUS
        else:
            regulated_v = self.V_MAX
        
        return max(regulated_v, self.REGULATED_LINEAR_SCALING_MIN_SPEED)

    def decide_accel_or_decel(self, current_idx: int, path_distances: np.ndarray) -> bool:
        """Decide whether to accelerate or decelerate based on remaining distance"""
        goal_distance = path_distances[-1] - path_distances[current_idx]
        decel_distance = (self.V_MAX ** 2) / (2 * self.A_MAX)
        return goal_distance > decel_distance

    def calc_optimal_velocity_considering_dynamic_window(self, current_velocity: np.ndarray, 
                                                       regulated_v: float, curvature: float, 
                                                       is_accel: bool) -> np.ndarray:
        """Calculate optimal velocity considering dynamic window constraints"""
        # Create dynamic window
        dw_vmax = min(current_velocity[0] + self.A_MAX * self.dt, self.V_MAX)
        dw_vmin = max(current_velocity[0] - self.A_MAX * self.dt, self.V_MIN)
        dw_wmax = min(current_velocity[1] + self.AW_MAX * self.dt, self.W_MAX)
        dw_wmin = max(current_velocity[1] - self.AW_MAX * self.dt, self.W_MIN)
        
        # Consider regulated velocity
        if dw_vmax > regulated_v:
            dw_vmax = max(dw_vmin, regulated_v)
        
        # Find intersection points between dynamic window and curvature line
        velocity_candidates = []
        p1 = (dw_vmin, curvature * dw_vmin)
        p2 = (dw_vmax, curvature * dw_vmax)
        velocity_candidates.extend([p1, p2])
        
        if curvature != 0.0:
            p3 = (dw_wmin / curvature, dw_wmin)
            p4 = (dw_wmax / curvature, dw_wmax)
            velocity_candidates.extend([p3, p4])
        
        # Filter valid candidates
        valid_velocity_candidates = []
        for v in velocity_candidates:
            if dw_vmin <= v[0] <= dw_vmax and dw_wmin <= v[1] <= dw_wmax:
                valid_velocity_candidates.append(v)
        
        # Select optimal velocity
        if len(valid_velocity_candidates) > 0:
            valid_velocity_candidates.sort(key=lambda x: x[0])
            if is_accel:
                next_velocity = valid_velocity_candidates[-1]
            else:
                next_velocity = valid_velocity_candidates[0]
        else:
            # If no intersection, find closest point on dynamic window boundary
            dw_coords = [
                (dw_vmin, dw_wmin), (dw_vmin, dw_wmax),
                (dw_vmax, dw_wmin), (dw_vmax, dw_wmax)
            ]
            
            distances = []
            for p in dw_coords:
                if curvature == 0:
                    dist = abs(p[1])
                else:
                    dist = abs(curvature * p[0] - p[1]) / math.sqrt(curvature**2 + 1)
                distances.append(dist)
            
            min_dist = min(distances)
            min_dist_coords = [p for p, dist in zip(dw_coords, distances) if dist == min_dist]
            min_dist_coords.sort(key=lambda x: x[0])
            
            if is_accel:
                next_velocity = min_dist_coords[-1]
            else:
                next_velocity = min_dist_coords[0]
        
        return np.array(next_velocity)

    def calc_accel_constrained_velocity(self, current_velocity: np.ndarray, 
                                      next_velocity_ref: np.ndarray) -> np.ndarray:
        """Apply acceleration constraints to reference velocity"""
        next_velocity = next_velocity_ref.copy()
        
        # Linear acceleration constraint
        max_linear_vel = min(current_velocity[0] + self.A_MAX * self.dt, self.V_MAX)
        min_linear_vel = max(current_velocity[0] - self.A_MAX * self.dt, self.V_MIN)
        next_velocity[0] = np.clip(next_velocity[0], min_linear_vel, max_linear_vel)
        
        # Angular acceleration constraint
        max_angular_vel = min(current_velocity[1] + self.AW_MAX * self.dt, self.W_MAX)
        min_angular_vel = max(current_velocity[1] - self.AW_MAX * self.dt, self.W_MIN)
        next_velocity[1] = np.clip(next_velocity[1], min_angular_vel, max_angular_vel)
        
        return next_velocity

    def publish_velocity(self, velocity: np.ndarray) -> None:
        """Publish velocity command"""
        cmd_msg = Twist()
        cmd_msg.linear.x = float(velocity[0])
        cmd_msg.angular.z = float(velocity[1])
        self.cmd_vel_pub.publish(cmd_msg)

    def publish_zero_velocity(self) -> None:
        """Publish zero velocity command"""
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0
        cmd_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_msg)

if __name__ == '__main__':
    try:
        controller = PurePursuitController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass