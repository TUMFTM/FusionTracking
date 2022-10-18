## Parameter description
All parameters can be set in [main-config_tracking](main_config_tracking.ini).

### Section : `SENSOR_SETUP`

| Parameter | Type | Default | Description
| ------------- | ------------- | ------ | ----- |
active_sensors | list(str) | ["lidar", "lidar_cluster", "radar"] | set active sensors to subscribe |
### Section : `SENSOR_MODELS`
Each sensor model that should be subscribed in `active_sensors` has to be specified in this section with the following parameters.

| Parameter | Type | Default | Description
| ------------- | ------------- | ------ | ----- |
meas_vars | list(str) | - | Measured kinematic object features |
std_dev | list(float) | - | Standard deviation of measured kinematic object features |
bool_local | boolean | false | If true measured object features are received in local vehicle coordinates |
max_delay_s | float | - | Maximal allowed delay in s from sensor input to tracking subscription |
n_init_counter | int | 3 | Individual value to initialize the matching counter for a new object |
n_add_counter | int | 1 | Individual value to increase counter after successful match |
get_yaw | str | - | Specify source of yaw measurement, either map-based ("from_track") or from detection algorithm ("measured") |
is_fallback | boolean | - | If true detection has to be subscribed and stay within the maximal allowed delay for valid tracking module state|
ros2_topic | str| - | topic name of detection message |
individual_header | boolean | false | If true each object in the receivec object list has an individual message header |



### Section : `MISCS`
| Parameter | Type | Default | Description
| ------------- | ------------- | ------ | ----- |
w_vehicle_m | float | 1.886 | Width of vehicle in m |
l_vehicle_m | float | 4.921 | Length of vehicle in m|
rear_ax_geoc_m | float | 1.74 | Distance from rear axle to geometric center of the vehicle in m |
attacker_modes | list(int) | [0, 2, 3, 6, 7, 8] | Track flag to switch to attacker mode |
track_file_IMS | str | "traj_ltpl_cl_IMS_GPS.csv" | File name of track map for Indianapolis Motor Speedway |
pit_file_IMS | str | "traj_ltpl_cl_IMS_pitlane_GPS.csv" | File name of pit map of Indianapolis Motor Speedway |
track_file_LVMS | str | "traj_ltpl_cl_LVMS_GPS.csv" | File name of track map of Las Vegas Motor Speedway |
track_file_LVMS_center | str | "traj_ltpl_cl_LVMS_center_GPS.csv" | File name of track map of Las Vegas Motor Speedway, center raceline |
track_file_LVMS_inner | str | "traj_ltpl_cl_LVMS_inner_GPS.csv" | File name of track map of Las Vegas Motor Speedway, inner raceline |
track_file_LVMS_outer | str | "traj_ltpl_cl_LVMS_outer_GPS.csv" | File name of track map of Las Vegas Motor Speedway, outer raceline |
pit_file_LVMS | str | "traj_ltpl_cl_LVMS_pitlane_GPS.csv" | File name of pit map of Las Vegas Motor Speedway |


### Section : `TRACKING`
| Parameter | Type | Default | Description
| ------------- | ------------- | ------ | ----- |
state_vars | list(str) |  ["x", "y", "yaw", "v", "yawrate", "a"] | Kinematic variables of state estimation model |
n_max_obsv | int | 350 | Number of observation stored per object |
add_ego_variables | boolean | true | If true ego variables are added during object initialization for non-measured features |
ego_speed_factor | float | 0.8 | Factor of ego speed to add to object initialization |
filter_frequency | float | 100.0 | Frequency of Extended Kalman Filter (EKF) |
publish_downsample_factor | int | 10 | Factor to downsample object storage entries to publish object lists |

#### `filter`
| Parameter | Type | Default | Description
| ------------- | ------------- | ------ | ----- |
P_init | list(float) | [0.01, 0.01, 0.3, 4.0, 0.3, 1.0] | Diagonal entries to initialize P-matrix of EKF for new object|
dx_std | float | 0.2 | Derivate of process uncertainty, longitudinal in m/s |
dy_std | float | 0.2 | Derivate of process uncertainty, lateral in m/s |
dyaw_std | float | 0.3 | Derivate of process uncertainty, heading in rad/s |
dv_std | float | 1.0 | Derivate of process uncertainty, speed in m/s² |
dyawrate_std | float | 0.5 | Derivate of process uncertainty, yaw rate in rad/s² |
da_std | float | 3.0 | Derivate of process uncertainty, acceleration in m/s³ |


#### `ego_lowpass`
| Parameter | Type | Default | Description
| ------------- | ------------- | ------ | ----- |
bool_lowpass | boolean | true | If true a low-pass filter is applied to subscribed ego state |
tc_yaw | float | 0.9 | Time constant of low-pass filter, heading |
tc_yawrate | float | 0.4| Time constane of low-pass filter, yaw rate |

#### `overlap_check`
| Parameter | Type | Default | Description
| ------------- | ------------- | ------ | ----- |
bool_overlap_check | boolean | true | If true detection input is checked for overlapping objects |
overlap_dist_m | float | 5.1 | Euclidean distance threshold for overlapping objects |

#### `collision_check`
| Parameter | Type | Default | Description
| ------------- | ------------- | ------ | ----- |
collision_dist_m | list(str) | - | Measured kinematic object features |
max_time_gap_s | list(float) | - | Standard deviation of measured kinematic object features |
time_res_s | list(float) | - | Standard deviation of measured kinematic object features |



#### `prediction`
| Parameter | Type | Default | Description
| ------------- | ------------- | ------ | ----- |
lap_wo_pred | float | -1.0 | Number of driven laps without prediction of opposing objects |
cone_angle_deg | float | 0.0 | Cone angle to ignore vehicle within behind ego vehicle in deg |
cone_offset_m | float | 1.0 | Offset of cone form rear of ego vehicle in m |
n_min_meas | int | 3 | Number of minimal detection of an object to consider in prediction |
step_size_s | float | 0.1 | Step size of predicted trajectory in s |
n_steps_phys_pred | int | 5 | Number of steps to predict physics-based |
rail_pred_switch_idx | int | 2 | Number of step to switch from physics-based to rail-based prediction |
n_steps_rail_pred | int | 50 | Number of steps to predict rail-based |
bool_rail_accl | boolean | false | If true acceleration is considered in speed profile of rail-based prediction |
bool_use_raceline | boolean | false | If true raceline is considered in rail-based prediction |
obj_raceline | str | "default" | Specify type of raceline to consider in rail-based prediction ("default", "inner", "center", "outer") |


#### `bounds_check`
| Parameter | Type | Default | Description
| ------------- | ------------- | ------ | ----- |
bool_bounds_check | boolean | true | If true objects outside the track boundaries are filtered out from detection input |
bounds_buffer_m | float | -1.5 | Distance from outer track bound to additionally filter objects out |
bounds_buffer_inner_m | float | -0.5 | Distance from inner track bound to additionally filter objects out |

#### `matching`
| Parameter | Type | Default | Description
| ------------- | ------------- | ------ | ----- |
max_match_dist_m | float | 5.0 | Euclidean distance threshold between objects for a successful match in m |
n_max_counter | int | 25 | Maximal value of matching counter of an object |
last_time_seen_s | float | 2.0 | Maximal duration sind last successful match, if exceeded object is discarded |
sr_threshhold_m | float | 60.0 | Euclidean distance threshold between short range and long range area in m |
sr_sensors | list(str) | ["lidar_cluster"] | Sensors for valid match and counter increase in short range |
lr_sensors | list(str) | ["radar"] | Sensors for valid match and counter increase in long range  |
