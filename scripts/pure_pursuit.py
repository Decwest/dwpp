import numpy as np
import math
from config import MIN_LOOK_AHEAD_DISTANCE, MAX_LOOK_AHEAD_DISTANCE, LOOK_AHEAD_TIME, V_MAX, V_MIN, W_MAX, W_MIN, \
    A_MAX, AW_MAX, DT, REGULATED_LINEAR_SCALING_MIN_RADIUS, REGULATED_LINEAR_SCALING_MIN_SPEED

def pure_pursuit(current_pose: np.ndarray, current_velocity: np.ndarray, path: np.ndarray, method_name: str)\
    -> tuple[np.ndarray, np.ndarray, np.ndarray, list[bool], float, float]:
    # calc index of current position
    current_idx = calc_index(current_pose, path)
    
    # calc distances between initial position and each position
    path_distances = calc_path_distances(path)
    
    # calc look ahead distance (Adaptive Pure Pursuit)
    look_ahead_distance = calc_look_ahead_distance(current_velocity, method_name)
    
    # calc curvature to the look ahead position
    curvature, look_ahead_pos = calc_curvature_to_look_ahead_position(current_pose, current_idx, path, path_distances, look_ahead_distance)
    
    # decide accel or decel
    is_accel = decide_accel_or_decel(current_idx, path_distances)
    
    if method_name in ["rpp", "dwpp"]:
        # calc regulated translational velocity (Regulated Pure Pursuit)
        regulated_v = calc_regulated_translational_velocity(curvature)
    else:
        regulated_v = V_MAX
    
    if method_name in ["pp", "app", "rpp"]:
        # calc translational velocity
        if is_accel:
            v_ref = min(current_velocity[0] + A_MAX * DT, V_MAX)
        else:
            v_ref = max(current_velocity[0] - A_MAX * DT, V_MIN)
        
        # regulate translational velocity
        if method_name == "rpp":
            v_ref = min(v_ref, regulated_v)
        
        # calc angular velocity
        w_ref = curvature * v_ref
        next_velocity_ref = np.array([v_ref, w_ref])
        
        # 加速度制約によるクリッピング
        next_velocity = calc_accel_constrained_velocity(current_velocity, next_velocity_ref)
        
    else:
        # calc dynamic window and optimal next velocity
        next_velocity = calc_optimal_velocity_considering_dynamic_window(current_velocity, regulated_v, curvature, is_accel)
        next_velocity_ref = next_velocity

    break_constraints_flag = [False, False]
    if next_velocity[0] != next_velocity_ref[0]:
        break_constraints_flag[0] = True
    if next_velocity[1] != next_velocity_ref[1]:
        break_constraints_flag[1] = True
    
    # debug用に、next_velocity_refと前方注視点の位置も返す
    return next_velocity, next_velocity_ref, look_ahead_pos, break_constraints_flag, curvature, regulated_v

def calc_index(current_pose: np.ndarray, path: np.ndarray) -> np.intp:
    # current_pose: [x, y, theta]
    # ロボットの位置と、経路上の各位置間の距離を計算し、最も近い位置のインデックスを返す
    distances = np.linalg.norm(path[:, :2] - current_pose[:2], axis=1)
    idx = np.argmin(distances)
    
    return idx

def calc_path_distances(path: np.ndarray) -> np.ndarray:
    # 経路の距離の累積和を計算
    ## 点間の差を計算
    differences = np.diff(path, axis=0)
    ## 各差のノルム（距離）を計算
    distances = np.linalg.norm(differences, axis=1)
    ## 累積距離を計算
    path_distances = np.concatenate(([0.0], np.cumsum(distances)))
    
    return path_distances

def calc_look_ahead_distance(current_velocity: np.ndarray, method_name: str) -> float:
    # calc look ahead distance
    if method_name in ["app", "rpp", "dwpp", "dwpp_wo_rpp"]:
        look_ahead_distance = LOOK_AHEAD_TIME * current_velocity[0]
        look_ahead_distance = min(max(look_ahead_distance, MIN_LOOK_AHEAD_DISTANCE), MAX_LOOK_AHEAD_DISTANCE)
        # look_ahead_distance = MIN_LOOK_AHEAD_DISTANCE + (STATIC_LOOK_AHEAD_DISTANCE - MIN_LOOK_AHEAD_DISTANCE) / V_MAX * current_velocity[0]
    else:
        look_ahead_distance = MIN_LOOK_AHEAD_DISTANCE
        
    return look_ahead_distance

def calc_curvature_to_look_ahead_position(current_pose: np.ndarray, current_idx: np.intp, path: np.ndarray, path_distances: np.ndarray, look_ahead_distance: float ) -> tuple[float, np.ndarray]:
    # 前方注視点の位置を計算
    ## 現在位置の距離を取得
    current_distance = path_distances[current_idx]
    ## 前方注視点の距離を計算
    look_ahead_pos_distance = current_distance + look_ahead_distance
    ## 前方注視点のインデックスを取得
    look_ahead_idx = min(np.searchsorted(path_distances, look_ahead_pos_distance), len(path) - 1)
    ## 前方注視点の位置を取得
    look_ahead_pos = path[look_ahead_idx]
    
    # 曲率を計算
    ## 前方注視点に向けた角度を計算
    look_ahead_angle = (math.atan2(look_ahead_pos[1] - current_pose[1], look_ahead_pos[0] - current_pose[0]) - current_pose[2])
    ## 前方注視点までの距離を計算
    L = float(np.linalg.norm(look_ahead_pos - current_pose[:2]))
    ## 曲率を計算
    curvature = 2.0 * math.sin(look_ahead_angle) / L
    
    return curvature, look_ahead_pos

def calc_regulated_translational_velocity(curvature: float) -> float:
    # Curvature heuristics
    if curvature == 0.0:
        return V_MAX
    
    curvature_radius = 1.0 / abs(curvature)
    if curvature_radius <= REGULATED_LINEAR_SCALING_MIN_RADIUS:
        regulated_v = V_MAX * curvature_radius / REGULATED_LINEAR_SCALING_MIN_RADIUS
    else:
        regulated_v = V_MAX
    
    regulated_v = max(regulated_v, REGULATED_LINEAR_SCALING_MIN_SPEED)
    
    # Proximity heuristics is ommitted, this simulation does not include obstacles
    
    return regulated_v

def decide_accel_or_decel(current_idx: np.intp, path_distances: np.ndarray) -> bool:
    # 経路のゴールまでの距離を計算
    goal_distance = path_distances[-1] - path_distances[current_idx]
    
    # 制動距離を計算
    decel_distance = (V_MAX ** 2) / (2 * A_MAX)
    
    # ゴール距離が制動距離よりも長い場合、加速
    if goal_distance > decel_distance:
        return True
    else:
        return False

def calc_optimal_velocity_considering_dynamic_window(current_velocity: np.ndarray, regulated_v: float, curvature: float, is_accel: bool) -> np.ndarray:
    # dynamic windowを作る
    dw_vmax = min(current_velocity[0] + A_MAX * DT, V_MAX)
    dw_vmin = max(current_velocity[0] - A_MAX * DT, V_MIN)
    dw_wmax = min(current_velocity[1] + AW_MAX * DT, W_MAX)
    dw_wmin = max(current_velocity[1] - AW_MAX * DT, W_MIN) 
    
    # regulated_vの考慮
    if dw_vmax > regulated_v:
        dw_vmax = max(dw_vmin, regulated_v)
        # print("Regulated v is considered.")
        # print(f"dw_vmax: {dw_vmax}")
    
    # Dynamic windowと曲率直線の交点を計算
    ## DWの4辺との交点を算出（解析解を代入）
    velocity_candidates = []
    p1 = (dw_vmin, curvature * dw_vmin)
    p2 = (dw_vmax, curvature * dw_vmax)
    velocity_candidates.append(p1)
    velocity_candidates.append(p2)
    if curvature != 0.0:
        p3 = (dw_wmin / curvature, dw_wmin)
        p4 = (dw_wmax / curvature, dw_wmax)
        velocity_candidates.append(p3)
        velocity_candidates.append(p4)
    ## 交点がDWの範囲内にあるか確認
    valid_velocity_candidates = []
    for v in velocity_candidates:
        if dw_vmin <= v[0] <= dw_vmax and dw_wmin <= v[1] <= dw_wmax:
            valid_velocity_candidates.append(v)
    
    # 最適な速度を計算
    ## 交点がある場合
    if len(valid_velocity_candidates) > 0:
        # 加速減速に基づいて目標速度を決める
        # 並進速度でソートする
        valid_velocity_candidates.sort(key=lambda x: x[0])
        # 加速する場合は、最も早い速度を選択
        if is_accel:
            next_velocity = valid_velocity_candidates[-1]
        # 減速する場合は、最も遅い速度を選択
        else:
            next_velocity = valid_velocity_candidates[0]
    
    ## 交点がない場合
    else:
        # 4頂点との距離を計算（距離関数は凸関数で、長方形は凸領域であることから、極値は4頂点上でとる）
        distance_from_coords = []
        dw_coords = [
            (dw_vmin, dw_wmin),
            (dw_vmin, dw_wmax),
            (dw_vmax, dw_wmin),
            (dw_vmax, dw_wmax),
        ]
        for p in dw_coords:
            # 曲率直線から頂点までの距離を算出
            dist = abs(curvature * p[0] - p[1]) / math.sqrt(curvature**2 + 1)
            distance_from_coords.append(dist)
        
        # 最短距離を見つける
        min_dist = min(distance_from_coords)
        # 最短距離となる速度候補のリストを作る
        min_dist_dw_coords = []
        for p, dist in zip(dw_coords, distance_from_coords):
            if dist == min_dist:
                min_dist_dw_coords.append(p)
        # 最短距離となる速度候補のリストを並進速度でソート
        min_dist_dw_coords.sort(key=lambda x: x[0])
        # 加速する場合は、最も早い速度を選択
        if is_accel:
            next_velocity = min_dist_dw_coords[-1]
        # 減速する場合は、最も遅い速度を選択
        else:
            next_velocity = min_dist_dw_coords[0]
    
    return np.array(next_velocity)

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
