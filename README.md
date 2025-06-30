# DWPP: Dynamic Window Pure Pursuit for Robot Path Tracking Considering Velocity and Acceleration Constraints

## Folder Structure

- `scripts/`  
  Contains simulation scripts for various pure pursuit methods. You can refer to these to understand how DWPP can be implemented.

- `ros1_node_example/`  
  Includes example ROS 1 implementations of DWPP and conventional pure pursuit methods.  
  The file `dwpp_ros_node.py` was used in real-world robot experiments, but it depends on our private configurations and cannot be used directly. Please use it as a reference only.

- `nav2_dwpp_controller/`  
  A work-in-progress (WIP) implementation of the DWPP controller as a Nav2 plugin.  
  The final version will be released in a separate GitHub repository.

- `results/`  
  Contains result figures used in the paper.

## Citation

If you use this code, please cite the following paper:

> Fumiya Ohnishi, Masaki Takahashi, “Dynamic Window Pure Pursuit for Robot Path Tracking Considering Velocity and Acceleration Constraints”, Proceedings of the 19th International Conference on Intelligent Autonomous Systems, Genoa, Italy, 2025.
