import numpy as np
from config import DT

def forward_simulation_differential(current_pose: np.ndarray, next_velocity: np.ndarray) -> np.ndarray:
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
    
    return next_pose
    