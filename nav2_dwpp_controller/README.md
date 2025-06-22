# Nav2 DWPP Controller

Dynamic Window Pure Pursuit Controller plugin for Nav2, extending the Regulated Pure Pursuit Controller with dynamic window constraints.

## Features

- **Dynamic Window Constraints**: Applies velocity and acceleration limits in real-time
- **Regulated Pure Pursuit**: Inherits all RPP features (curvature regulation, goal approach, etc.)
- **Acceleration/Deceleration Logic**: Intelligent speed selection based on remaining path distance
- **Nav2 Integration**: Full compatibility with Nav2 navigation stack

## Installation

1. Clone this package into your ROS2 workspace:
```bash
cd ~/ros2_ws/src
cp -r /path/to/nav2_dwpp_controller .
```

2. Build the workspace:
```bash
cd ~/ros2_ws
colcon build --packages-select nav2_dwpp_controller
source install/setup.bash
```

## Usage

### Configuration

Add the DWPP controller to your Nav2 configuration file:

```yaml
controller_server:
  ros__parameters:
    controller_plugins: ["DWPPController"]
    
    DWPPController:
      plugin: "nav2_dwpp_controller::DWPPController"
      
      # Standard RPP parameters
      desired_linear_vel: 0.5
      lookahead_dist: 0.6
      # ... (other RPP parameters)
      
      # DWPP-specific parameters
      max_linear_accel: 0.5
      max_angular_accel: 1.0
      use_dynamic_window: true
      deceleration_distance_factor: 2.0
```

### Launch

Use with any Nav2 launch file by updating the controller configuration:

```bash
ros2 launch nav2_bringup navigation_launch.py params_file:=path/to/dwpp_controller_params.yaml
```

## Parameters

### DWPP-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_linear_accel` | double | 0.5 | Maximum linear acceleration (m/s²) |
| `max_angular_accel` | double | 1.0 | Maximum angular acceleration (rad/s²) |
| `velocity_sample_resolution` | double | 0.1 | Resolution for velocity sampling |
| `use_dynamic_window` | bool | true | Enable/disable dynamic window constraints |
| `deceleration_distance_factor` | double | 2.0 | Factor for deceleration distance calculation |

### Inherited RPP Parameters

All parameters from `nav2_regulated_pure_pursuit_controller` are supported. See [Nav2 RPP documentation](https://navigation.ros.org/configuration/packages/configuring-regulated-pp.html) for details.

## Algorithm Overview

1. **Base Computation**: Uses RPP to calculate curvature and look-ahead point
2. **Dynamic Window**: Calculates velocity bounds based on current velocity and acceleration limits
3. **Velocity Candidates**: Generates possible velocities within the dynamic window
4. **Optimal Selection**: Selects velocity based on acceleration/deceleration strategy
5. **Output**: Publishes velocity command respecting all constraints

## Comparison with Other Controllers

| Controller | Look-ahead | Velocity Regulation | Dynamic Constraints |
|------------|------------|-------------------|-------------------|
| Pure Pursuit (PP) | Fixed | No | No |
| Adaptive PP (APP) | Adaptive | No | No |
| Regulated PP (RPP) | Adaptive | Yes | No |
| **DWPP** | Adaptive | Yes | **Yes** |

## Dependencies

- ROS2 (Humble/Iron/Rolling)
- Nav2 navigation stack
- nav2_regulated_pure_pursuit_controller

## License

Apache-2.0

## Citation

```
Fumiya Ohnishi, Masaki Takahashi, "Dynamic Window Pure Pursuit for Robot Path Tracking Considering Velocity and Acceleration Constraints", Proceedings of the 19th International Conference on Intelligent Autonomous Systems, Genoa, Italy, 2025.
```