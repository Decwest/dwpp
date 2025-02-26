MIN_LOOK_AHEAD_DISTANCE = 0.4
MAX_LOOK_AHEAD_DISTANCE = 0.5
LOOK_AHEAD_TIME = 1.0
V_MAX = 0.5
V_MIN = 0.0
W_MAX = 1.0
W_MIN = -1.0
A_MAX = 0.5
AW_MAX = 1.0
DT = 0.05
REGULATED_LINEAR_SCALING_MIN_RADIUS = 0.9
REGULATED_LINEAR_SCALING_MIN_SPEED = 0.0

method_name_dict = {
    "pp": "Pure Pursuit",
    "app": "Adaptive Pure Pursuit",
    "rpp": "Regulated Pure Pursuit",
    # "dwpp_wo_rpp": "Dynamic Window Pure Pursuit without RPP",
    "dwpp": "Dynamic Window Pure Pursuit"
}
# method_name_list = ["pp", "app", "rpp", "dwpp_wo_rpp", "dwpp"]
method_name_list = ["pp", "app", "rpp", "dwpp"]
