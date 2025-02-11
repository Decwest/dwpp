from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from config import method_name_dict
from collections import defaultdict
import os

# 軌跡の描画
def draw_paths(path: np.ndarray, robot_poses_dist: defaultdict, path_name: str):
    print("drawing paths...")
    
    fig, ax = plt.subplots()
    ax.set_xlim(np.min(path[:, 0]) - 1, np.max(path[:, 0]) + 1)
    ax.set_ylim(np.min(path[:, 1]) - 1, np.max(path[:, 1]) + 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{path_name} Simulation')
    ax.set_aspect('equal')

    # パス
    ax.plot(path[:, 0], path[:, 1], 'k--', label='Path')

    # ロボットの位置
    for method_name, robot_poses in robot_poses_dist.items():
        robot_poses = np.array(robot_poses)
        ax.plot(robot_poses[:, 0], robot_poses[:, 1], label=method_name_dict[method_name])

    # 凡例を表示
    ax.legend()
    
    plt.tight_layout()

    # グラフを表示
    plt.savefig(f'../results/{path_name}/paths.png')

# 速度プロファイルの描画
def draw_velocity_profile(time_stamps: np.ndarray, robot_velocities: np.ndarray, robot_ref_velocities: np.ndarray, method_name: str, path_name: str):
    print("drawing velocity profiles...")
    
    # translation
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel('t')
    ax1.set_ylabel('v')
    ax1.set_title(f'Translational Velocity Profile, Path: {path_name}, Method: {method_name_dict[method_name]}')
    
    ax1.plot(time_stamps, robot_velocities[:, 0], label='Command Velocity')
    ax1.scatter(time_stamps, robot_ref_velocities[:, 0], label='Reference Velocity', s=5)
    
    ax1.legend()
    plt.tight_layout()
    os.makedirs(f'../results/{path_name}', exist_ok=True)
    plt.savefig(f'../results/{path_name}/{method_name}_translational_velocity_profile.png')
    
    # rotation
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel('t')
    ax2.set_ylabel('w')
    ax2.set_title(f'Rotational Velocity Profile, Path: {path_name}, Method: {method_name_dict[method_name]}')
    
    ax2.plot(time_stamps, robot_velocities[:, 1], label='Command Velocity')
    ax2.scatter(time_stamps, robot_ref_velocities[:, 1], label='Reference Velocity', s=5)
    
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f'../results/{path_name}/{method_name}_rotational_velocity_profile.png')
    

# アニメーションの描画
def draw_animation(path: np.ndarray, robot_poses: np.ndarray, look_ahead_positions: np.ndarray, method_name:str, path_name: str):
    print("drawing animations...")

    fig, ax = plt.subplots()
    ax.set_xlim(np.min(path[:, 0]) - 1, np.max(path[:, 0]) + 1)
    ax.set_ylim(np.min(path[:, 1]) - 1, np.max(path[:, 1]) + 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{method_name_dict[method_name]} Simulation')
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
    ani.save(f'../results/{path_name}/{method_name}.mp4', writer='ffmpeg', fps=10)
