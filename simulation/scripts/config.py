STATIC_LOOK_AHEAD_DISTANCE = 0.5
V_MAX = 0.8
V_MIN = 0.0
W_MAX = 1.0
W_MIN = -1.0
A_MAX = 0.8
AW_MAX = 1.0
DT = 0.02
REGULATED_LINEAR_SCALING_MIN_RADIUS = 2.0
REGULATED_LINEAR_SCALING_MIN_SPEED = 0.3

method_name_dict = {
    "pp": "Pure Pursuit",
    "app": "Adaptive Pure Pursuit",
    "rpp": "Regulated Pure Pursuit",
    "dwpp": "Dynamic Window Pure Pursuit"
}
method_name_list = ["pp", "app", "rpp", "dwpp"]
