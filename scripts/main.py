from dwp import dwp
from path import *
from robot import forward_simulation_differential
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch

# 初期設定
## ロボットの状態
current_pose = np.array([0.0, 0.0, np.pi / 2])
current_velocity = np.array([0.0, 0.0])

## 経路
# path_name = 'sin_curve'
path = sin_curve()
path_name = 'step_curve'
path = step_curve()

# データの保存
robot_poses = [current_pose]
robot_velocities = [current_velocity]
look_ahead_positions = []

# シミュレーション
method_name = "dwp"
sim_start_time = perf_counter()
while True:
    # 終了条件
    if np.linalg.norm(current_pose[:2] - path[-1]) < 0.01:
        break
    
    next_velocity, look_ahead_pos = dwp(current_pose, current_velocity, path)
    next_pose = forward_simulation_differential(current_pose, next_velocity)
    
    current_pose = next_pose
    current_velocity = next_velocity
    
    robot_poses.append(current_pose)
    robot_velocities.append(current_velocity)
    look_ahead_positions.append(look_ahead_pos)
sim_end_time = perf_counter()
print(f"Simulation time: {sim_end_time - sim_start_time}s")

# アニメーションの描画
# データの変換
robot_poses = np.array(robot_poses)
look_ahead_positions = np.array(look_ahead_positions)
path = np.array(path)

fig, ax = plt.subplots()
ax.set_xlim(np.min(path[:, 0]) - 1, np.max(path[:, 0]) + 1)
ax.set_ylim(np.min(path[:, 1]) - 1, np.max(path[:, 1]) + 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Dynamic Window Pure Pursuit Simulation')
ax.set_aspect('equal')

# プロット要素の初期化
path_line, = ax.plot(path[:, 0], path[:, 1], 'k--', label='Path')  # パス
robot_point, = ax.plot([], [], 'bo', label='Robot')  # ロボット位置
robot_path, = ax.plot([], [], linewidth=1, color="cyan")  # ロボット位置
look_ahead_point, = ax.plot([], [], 'ro', label='Look Ahead')  # ルックアヘッド位置

robot_xs = []
robot_ys = []

# ロボットの向きを示す矢印
robot_arrow = FancyArrowPatch((0, 0), (0, 0), mutation_scale=20, color='blue')
ax.add_patch(robot_arrow)

def init():
    robot_point.set_data([], [])
    robot_path.set_data([], [])
    look_ahead_point.set_data([], [])
    robot_arrow.set_visible(False)
    return robot_point, robot_path, look_ahead_point, robot_arrow

def update(frame):
    # ロボットの位置を更新
    robot_x = robot_poses[frame, 0]
    robot_y = robot_poses[frame, 1]
    robot_theta = robot_poses[frame, 2]
    
    robot_xs.append(robot_x)
    robot_ys.append(robot_y)

    robot_point.set_data([robot_x], [robot_y])
    robot_path.set_data(robot_xs, robot_ys)

    # ロボットの向きを示す矢印を更新
    arrow_length = 0.5  # 矢印の長さ
    dx = arrow_length * np.cos(robot_theta)
    dy = arrow_length * np.sin(robot_theta)

    # 矢印の位置を更新
    robot_arrow.set_positions((robot_x, robot_y), (robot_x + dx, robot_y + dy))
    robot_arrow.set_visible(True)

    # ルックアヘッドの位置を更新
    if frame < len(robot_poses) - 1:
        look_ahead_x = look_ahead_positions[frame, 0]
        look_ahead_y = look_ahead_positions[frame, 1]
        look_ahead_point.set_data([look_ahead_x], [look_ahead_y])
    else:
        look_ahead_point.set_data([], [])

    return path_line, robot_point, look_ahead_point, robot_arrow

ani = FuncAnimation(
    fig, update, frames=len(robot_poses), init_func=init, blit=False, interval=100, repeat=False
)

# 凡例を表示
ax.legend()

# アニメーションを表示または保存
# plt.show()
ani.save(f'../videos/{method_name}_{path_name}.mp4', writer='ffmpeg', fps=10)

