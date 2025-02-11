from pure_pursuit import pure_pursuit
from path import *
from robot import forward_simulation_differential
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from config import method_name_dict, DT, method_name_list
from draw import draw_animation, draw_paths, draw_velocity_profile
from stats import calc_rmse, calc_break_constraints_rate
from collections import defaultdict

## 経路
# path_name = 'sin_curve'
path = sin_curve()
path_name = 'step_curve'
path = step_curve()


def simulation(path: np.ndarray, draw: bool)-> tuple[dict, dict]:
    # 一つの曲線に対するシミュレーションと描画

    robot_poses_dict = defaultdict(list)
    robot_velocities_dict = defaultdict(list)
    robot_ref_velocities_dict = defaultdict(list)
    look_ahead_positions_dict = defaultdict(list)
    break_constraints_flags_dict = defaultdict(list)
    time_stamp_dict = defaultdict(list)

    for method_name in method_name_list:
        print(f"Method: {method_name}")
        
        # 初期設定
        ## ロボットの状態
        current_pose = np.array([0.0, 0.0, 0.0])
        current_velocity = np.array([0.0, 0.0])
        # データの保存
        robot_poses = [current_pose]
        robot_velocities = [current_velocity]
        robot_ref_velocities = [current_velocity]
        look_ahead_positions = []
        break_constraints_flags = [[False, False]]
        time_stamps = [0.0]
        
        sim_start_time = perf_counter()
        
        while True:
            # 終了条件
            if np.linalg.norm(current_pose[:2] - path[-1]) < 0.01:
                break
            
            next_velocity, next_velocity_ref, look_ahead_pos, break_constraints_flag = pure_pursuit(current_pose, current_velocity, path, method_name)
            next_pose = forward_simulation_differential(current_pose, next_velocity)
            
            current_pose = next_pose
            current_velocity = next_velocity
            
            robot_poses.append(current_pose)
            robot_velocities.append(current_velocity)
            robot_ref_velocities.append(next_velocity_ref)
            look_ahead_positions.append(look_ahead_pos)
            break_constraints_flags.append(break_constraints_flag)
            time_stamps.append(time_stamps[-1] + DT)
            
            if current_velocity[0] == 0.0 and current_velocity[1] == 0.0:
                break
            
        sim_end_time = perf_counter()
        print(f"Simulation time: {sim_end_time - sim_start_time}s")
        
        robot_poses_dict[method_name] = robot_poses
        robot_velocities_dict[method_name] = robot_velocities
        robot_ref_velocities_dict[method_name] = robot_ref_velocities
        look_ahead_positions_dict[method_name] = look_ahead_positions
        break_constraints_flags_dict[method_name] = break_constraints_flags
        time_stamp_dict[method_name] = time_stamps

    # 統計量の算出
    # RMSEの算出
    rmse_dict = calc_rmse(robot_poses_dict, path)
    
    # 制約違反した割合の算出
    break_constraints_rate_dict = calc_break_constraints_rate(break_constraints_flags_dict)
    
    if not draw:
        return rmse_dict, break_constraints_rate_dict
    
    for method_name in method_name_list:
        robot_poses = np.array(robot_poses_dict[method_name])
        look_ahead_positions = np.array(look_ahead_positions_dict[method_name])
        robot_velocities = np.array(robot_velocities_dict[method_name])
        robot_ref_velocities = np.array(robot_ref_velocities_dict[method_name])
        break_constraints_flags = np.array(break_constraints_flags_dict[method_name])
        time_stamps = np.array(time_stamp_dict[method_name])
        
        # 速度プロファイルの描画
        draw_velocity_profile(time_stamps, robot_velocities, robot_ref_velocities, break_constraints_flags, method_name, path_name)
        
        # 各手法ごとのアニメーションの描画
        draw_animation(path, robot_poses, look_ahead_positions, method_name, path_name)

    # 軌跡の描画
    draw_paths(path, robot_poses_dict, path_name)
    
    return rmse_dict, break_constraints_rate_dict

simulation(path, draw=False)
