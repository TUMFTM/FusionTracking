"""Class to check collision of object list."""
import numpy as np
from utils.tracking_utils import ctrv_single


class CheckCollisions:
    """Class to check collision of object list."""

    __slots__ = (
        "__max_time_gap_s",
        "__time_res_s",
        "__collision_dist_m",
    )

    def __init__(self, collisions_params: dict):
        """Initialize class."""
        self.__max_time_gap_s = collisions_params["max_time_gap_s"]
        self.__time_res_s = collisions_params["time_res_s"]
        self.__collision_dist_m = collisions_params["collision_dist_m"]

    def __call__(self, ego_state: dict, object_dict: dict, logger) -> int:
        """Check collision between two objects.

        Determine the euclidean distance
        between ego and opponent object for collision check.

        Args:
            ego_state (dict): Dict of ego state including kinematic state and timestamp.
            object_dict (dict): Object dict including kinematic state and timestamp.
            logger (logger): Log information about collisions.

        Returns:
            int: Module state (30: valid, 50: soft emergency)
        """
        if self.__collision_dist_m <= 0.0 or not object_dict:
            return 30

        for obj_id, object_el in object_dict.items():
            dt_s = (ego_state["t"] - object_el["t"]) / 1e9

            # clip maximal valid time gap
            if np.abs(dt_s) > self.__max_time_gap_s:
                logger.warning(
                    "COLLISION: clipping, dt = {:.02f} ms, "
                    "dt_max = {:.02f} ms, obj_id = {}".format(
                        abs(dt_s) * 1e3,
                        self.__max_time_gap_s * 1e3,
                        obj_id,
                    )
                )
                dt_s = np.clip(
                    dt_s,
                    a_min=-self.__max_time_gap_s,
                    a_max=self.__max_time_gap_s,
                )

            # either directly determine distance or predict "older" object ahead
            if np.abs(dt_s) > self.__time_res_s:
                if dt_s > 0.0:  # Ego is the "newer" object, so predict opponent ahead
                    xy_ego = np.array(ego_state["state"][:2])
                    xy_obj = ctrv_single(np.array(object_el["state"]), dt_s)[:2]
                else:  # Opponent is the "newer" object, so predict ego ahead
                    xy_obj = np.array(object_el["state"][:2])
                    xy_ego = ctrv_single(np.array(ego_state["state"]), -dt_s)[:2]
            else:
                xy_obj = np.array(object_el["state"][:2])
                xy_ego = np.array(ego_state["state"][:2])

            # critical distance
            obj_dist = np.linalg.norm(xy_ego - xy_obj)

            # trigger collision
            if obj_dist < self.__collision_dist_m:
                log_dist = np.linalg.norm(
                    np.array(object_el["state"][:2]) - np.array(ego_state["state"][:2])
                )

                logger.error(
                    "COLLISION: d = {:.02f} m, d_crit = {:.02f} m, obj_id = {}".format(
                        obj_dist, self.__collision_dist_m, obj_id
                    )
                )
                logger.warning(
                    "COLLISION: dt = "
                    "{:.02f} ms, xy_ego = {:.02f}, {:.02f}, "
                    "xy_obj = {:.02f}, {:.02f}, d_log = {:.02f} m".format(
                        dt_s * 1e3,
                        ego_state["state"][0],
                        ego_state["state"][1],
                        object_el["state"][0],
                        object_el["state"][1],
                        log_dist,
                    )
                )
                return 50

            # trigger warning
            if obj_dist < 2.0 * self.__collision_dist_m:
                log_dist = np.linalg.norm(
                    np.array(object_el["state"][:2]) - np.array(ego_state["state"][:2])
                )

                logger.warning(
                    "COLLISION: warning, d = {:.02f} m, d_crit = {:.02f} m, obj_id = {}".format(
                        obj_dist, self.__collision_dist_m, obj_id
                    )
                )
                logger.warning(
                    "COLLISION: warning, dt = "
                    "{:.0f} ms, xy_ego = {:.02f}, {:.02f}, "
                    "xy_obj = {:.02f}, {:.02f}, d_log = {:.02f} m".format(
                        dt_s * 1000,
                        ego_state["state"][0],
                        ego_state["state"][1],
                        object_el["state"][0],
                        object_el["state"][1],
                        log_dist,
                    )
                )

        return 30
