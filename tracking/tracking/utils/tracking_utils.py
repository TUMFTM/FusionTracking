"""Tracking utils."""
import math
from collections import deque
import numpy as np
from scipy.spatial import cKDTree

from .geometry import pi_range, rotate_loc_glob, rotate_glob_loc, angle_between_loc


def reduce_list(is_out_of_track, is_out_of_pit, new_object_list, logger, sensor_str):
    """Reduce list by filterings objects out of bounds."""
    bounds_list = [min(k) for k in zip(is_out_of_track, is_out_of_pit)]
    if any(bounds_list):
        new_object_list[:] = [
            obj_el for j, obj_el in enumerate(new_object_list) if not bounds_list[j]
        ]
        logger.info(
            "OUT OF TRACK: sensor = {}, remove {} of {}".format(
                sensor_str, sum(bounds_list), len(is_out_of_track)
            )
        )


def check_estimated_dict(est_dict, t_track, logger):
    """Check if estimation dict is updaed correctly."""
    for obj_id, obj_val in est_dict.items():
        if obj_val["t"] != t_track:
            logger.error(
                "INVALID STORAGE HANDLING: dt = {:.02f} ms, obj_id = {}".format(
                    (obj_val["t"] - t_track) / 1e6, obj_id
                )
            )


def check_observation_storage(obsv_storage, old_obj_dict, logger):
    """Check entry and time stamp size of objects."""
    for obj_id, obj_val in obsv_storage.items():
        if obj_id == "ego":
            continue
        if len(set(np.diff(obj_val["t"]))) > 1:
            raise IndexError("{}".format(np.diff(obj_val["t"])))

    if obsv_storage and set(obsv_storage.keys()) - set(old_obj_dict.keys()) != {"ego"}:
        logger.warning(
            "INVALID STORAGE HANDLING: self.observation_storage.keys() = {}".format(
                obsv_storage.keys()
            )
        )
        logger.warning(
            "INVALID STORAGE HANDLING: self.old_object_dict.keys() = {}".format(
                obsv_storage.keys()
            )
        )


def check_input_dimension(new_object: dict, state_keys: list, sensor_str: str, logger):
    """Check input dimensions."""
    if len(new_object["state"]) != len(state_keys):
        logger.error(
            "DIMENSION ERROR: object = "
            "{}, sensor_str = {}, state_keys = {}".format(
                new_object,
                sensor_str,
                state_keys,
            )
        )
        return False

    return True


def check_none_realistic_object(
    measure_tuple: tuple, overlap_dist_m: float, logger=None
):
    """Check for non-realistic objects.

    Non-realistic is referred to objects with the overlap distance.
    If so, the objects are clustered and the merged object is returned.

    Args:
        measure_tuple (tuple): Tuple of detection input including name, time stamp and new object list
        overlap_dist_m (float): Radius to check for overlapping objects in m
        logger (_type_, optional): Message logging. Defaults to None.

    Returns:
        tuple: Tuple of detection input with merged objects
    """
    sensor_str, (new_object_list, detection_timestamp_ns) = measure_tuple

    n_0 = len(new_object_list)

    # extract coordinates
    coords = [point["state"][:2] for point in new_object_list]
    if coords == []:
        return measure_tuple

    kd_tree1 = cKDTree(coords)

    # index List of "matches" within the overlap radius
    indexes = kd_tree1.query_ball_tree(kd_tree1, r=overlap_dist_m)

    # take only the longest list in list, otherwise its redundant
    longest = max(len(elem) for elem in indexes)
    if longest == 1:
        return measure_tuple

    # Remove single entries
    object_idx = [list(x) for x in indexes if len(x) >= 2]

    # only keep lists which have diffrent entries
    object_idx = [list(tupl) for tupl in {tuple(item) for item in object_idx}]

    # Using the object_idx list, merge the respective objects into one.new_object_list
    new_new_object_list = []
    idx_list = []
    for indices in object_idx:
        idx_list.extend(indices)
        objects_states = [new_object_list[idx]["state"] for idx in indices]
        # take the average of the respective objects.
        avg_state = [sum(x) / len(x) for x in zip(*objects_states)]
        new_new_object_list.append(
            {
                "state": avg_state,
                "t": detection_timestamp_ns,
                "keys": new_object_list[0]["keys"],
            }
        )

    for old_idx, new_obj in enumerate(new_object_list):
        if old_idx in idx_list:
            continue
        new_new_object_list.append(new_obj)

    # logging
    n_1 = len(new_new_object_list)
    if logger is not None:
        logger.info(
            "MERGE OBJECTS: sensor = {}, "
            "merge {} to {} objects".format(sensor_str, n_0, n_1)
        )
    return (sensor_str, (new_new_object_list, detection_timestamp_ns))


def first_obj_logger(
    detection_obj_dict: dict,
    is_connecting: list,
    is_not_empty: list,
    logger,
):
    """Log of initialized interface status.

    Args:
        detection_obj_dict (dict): (sensor_str, (new_object_list, detection_timestamp_ns))
        is_connecting (list): list of not connected sensors (strings)
        is_not_empty (list): list of sensors which haven"t detected an object yet (strings)
        logger (Logger): message logger
    """
    removal_list = []
    for sensor_str in is_not_empty:
        if sensor_str in detection_obj_dict:
            # get object_list and timestamp
            (new_object_list, detection_timestamp_ns) = detection_obj_dict[sensor_str]

            if new_object_list:
                logger.info(
                    "CONNECTED: sensor = {}, time_stamp = {}, n_obj = {}".format(
                        sensor_str, detection_timestamp_ns, len(new_object_list)
                    )
                )
                for idx, obj in enumerate(new_object_list):
                    logger.info(
                        "CONNECTED: sensor = {}, object {}: {}".format(
                            sensor_str,
                            idx,
                            obj,
                        )
                    )
                # add sensor_str to removal_list
                removal_list.append(sensor_str)
            else:
                logger.info(
                    "CONNECTED: sensor = {}, time_stamp = {}, empty list".format(
                        sensor_str, detection_timestamp_ns
                    )
                )

            # connection established
            if sensor_str in is_connecting:
                is_connecting.remove(sensor_str)
                if not is_connecting:
                    logger.info("FULLY CONNECTED: connected to all sensors")

        elif sensor_str in is_connecting:
            logger.warning("NOT CONNECTED: sensor = {}".format(sensor_str))

    for sensor_str in removal_list:
        is_not_empty.remove(sensor_str)

    if not is_not_empty:
        logger.info("FULLY CONNECTED: all sensors detected an object")


def ignore_behind_cone(
    obj_pose, ego_pose, veh_length, cone_angle, cone_offset_m, logger
):
    """Ignore vehicles behind ego for prediction if inside specific cone.

    Cone is spaned from center of rear-axle (cog - length / 2.0)

    cone_angle = Totel Angle of Cone. 0.5 per side (right, left)

    return bool: True if vehicle is ignored, i.e. inside cone

    Test:
        ego_pose = np.array([-300, -1800, -np.pi])
        cone_angle = 60 / 180 * np.pi
        veh_length = 4.921
        obj_pose = np.array(
            [-300 - 20 * np.tan(cone_angle/2.0),
            -1780 + veh_length / 2.0, -np.pi]
        )
    """
    obj_pose[:2] += rotate_loc_glob(
        np.array([veh_length / 2.0, 0.0]), obj_pose[2], matrix=False
    )
    loc_obj_pos = rotate_glob_loc(
        obj_pose[:2] - ego_pose[:2], ego_pose[2], matrix=False
    )
    loc_obj_pos[0] += veh_length / 2.0

    if loc_obj_pos[0] > -cone_offset_m:
        return False

    obj_angle = pi_range(math.atan2(loc_obj_pos[1], loc_obj_pos[0]) - np.pi)

    if abs(obj_angle) > cone_angle / 2.0:
        return False

    logger.info(
        "CONE - Ignoring object, obj-pose = {}, "
        "angle behind ego-rear-axle = {:.03f} rad, ego_pose = {}".format(
            obj_pose, obj_angle, ego_pose
        )
    )

    return True


def get_new_storage_item(deq_length: int, x_init: list, P_init: list, t_init: list):
    """Get template for object storage entry."""
    return {
        "state": deque(x_init, maxlen=deq_length),
        "t": deque(t_init, maxlen=deq_length),
        "P": deque(P_init, maxlen=deq_length),
        "num_measured": 1,
        "tracking_id": -1,
        "last_time_seen_ns": max(t_init),
    }


def ctrv_single(x_in, dt_s):
    """Conduct Single CTRV-Step.

    State Vector x = [x-pos, y-pos, yaw, velocity, yawrate, acceleration]
    """
    xout = np.zeros(x_in.shape)
    xout[0] = x_in[0] - x_in[3] * dt_s * np.sin(x_in[2])
    xout[1] = x_in[1] + x_in[3] * dt_s * np.cos(x_in[2])
    xout[2] = x_in[2] + x_in[4] * dt_s
    xout[3] = x_in[3]  # Constant Speed
    xout[4] = x_in[4]  # Constant Turn Rate
    xout[5] = x_in[5]  # Constant Acceleration

    return xout


def get_initialized_state(state_vars, meas_obj: dict) -> list:
    """Get fully initialized state of an object according to reference state variables.

    Args:
        meas_obj (dict): measured object, contains keys: ("keys", "state", "t")

    Returns:
        list: Unified object state, i.e. a list with values accord to reference state variables
    """
    meas_state_indices = [state_vars.index(key) for key in meas_obj["keys"]]
    new_state_list = [0.0] * len(state_vars)

    for idx, val in zip(meas_state_indices, meas_obj["state"]):
        new_state_list[idx] = val

    return new_state_list


def get_H_mat(index_tuples, n_dof=6):
    """Determine output matrix of measured values of EKF model."""
    meas_shape = (len(index_tuples[0]), n_dof)
    H_mat = np.zeros(meas_shape)
    H_mat[index_tuples[0], index_tuples[1]] = 1.0
    return H_mat


def get_R_mat(ego_pos, obj_pos, _measure_covs_sens):
    """Determine measurement noise of EKF model."""
    # get R-Matrix
    R_mat = np.copy(_measure_covs_sens)
    # rotate
    rel_yaw = pi_range(angle_between_loc(obj_pos - ego_pos))
    R_mat[:2, :2] = rotate_loc_glob(R_mat[:2, :2], rel_yaw)
    return R_mat


def log_filter_vals(
    obj_filter: np.ndarray,
    obj_filter_log: list,
    time_stamp: int,
    n_est: int,
    is_measured: bool,
    sensor: str,
):
    """Pass filter values to logging dict."""
    # measured: log priors
    if is_measured:
        obj_filter_log.append(
            [
                obj_filter.x_prior,
                obj_filter.P_prior,
                obj_filter.x_post,
                obj_filter.P_post,
                obj_filter.x,
                obj_filter.P,
                obj_filter.y,
                [obj_filter.z_meas, sensor],
                time_stamp,
                n_est,
                obj_filter.num_updates,
            ]
        )
    # not measured: no priors available
    else:
        obj_filter_log.append(
            [
                None,
                None,
                None,
                None,
                obj_filter.x,
                obj_filter.P,
                None,
                None,
                time_stamp,
                n_est,
                None,
            ]
        )
    return obj_filter_log
