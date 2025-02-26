import numpy as np
from config import DT, A_MAX, AW_MAX, V_MAX, V_MIN, W_MAX, W_MIN

def forward_simulation_differential(current_pose: np.ndarray, current_velocity: np.ndarray, next_velocity_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    next_velocity = calc_accel_constrained_velocity(current_velocity, next_velocity_ref)
    
    if next_velocity[1] == 0.0:
        # 直進運動
        next_pose = current_pose + np.array([
            next_velocity[0] * np.cos(current_pose[2]),
            next_velocity[0] * np.sin(current_pose[2]),
            next_velocity[1]
        ]) * DT
    else:
        # 旋回運動
        R = next_velocity[0] / next_velocity[1]
        next_pose = current_pose + np.array([
            R * (np.sin(current_pose[2] + next_velocity[1] * DT) - np.sin(current_pose[2])),
            R * (-np.cos(current_pose[2] + next_velocity[1] * DT) + np.cos(current_pose[2])),
            next_velocity[1] * DT
        ])
    
    return next_pose, next_velocity
    

def calc_accel_constrained_velocity(current_velocity: np.ndarray, next_velocity_ref: np.ndarray) -> np.ndarray:
    # 加速度制約によるクリッピング
    next_velocity = next_velocity_ref.copy()
    ## 並進加速度の制約
    if next_velocity_ref[0] > min(current_velocity[0] + A_MAX * DT, V_MAX):
        next_velocity[0] = min(current_velocity[0] + A_MAX * DT, V_MAX)
    elif next_velocity_ref[0] < max(current_velocity[0] - A_MAX * DT, V_MIN):
        next_velocity[0] = max(current_velocity[0] - A_MAX * DT, V_MIN)
    ## 角速度の制約
    if next_velocity_ref[1] > min(current_velocity[1] + AW_MAX * DT, W_MAX):
        next_velocity[1] = min(current_velocity[1] + AW_MAX * DT, W_MAX)
    elif next_velocity_ref[1] < max(current_velocity[1] - AW_MAX * DT, W_MIN):
        next_velocity[1] = max(current_velocity[1] - AW_MAX * DT, W_MIN)
    
    return next_velocity

