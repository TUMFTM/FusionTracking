"""Class of kinematic checks."""
import logging
import numpy as np

from utils.geometry import pi_range


class KinematicCheck:
    """Class of kinematic checks."""

    __slots__ = (
        "__logger",
        "__max_err_path",
        "__max_err_yaw",
        "__max_err_speed",
        "__modulo_logger",
        "__path_violation_ctr",
        "__yaw_violation_ctr",
        "__speed_violation_ctr",
    )

    def __init__(self):
        """Initiliaze class."""
        self.__logger = logging.getLogger("msg_logger")
        self.__max_err_path = 1.0
        self.__max_err_yaw = 0.2
        self.__max_err_speed = 3.0
        self.__modulo_logger = 50
        self.__path_violation_ctr = 0
        self.__yaw_violation_ctr = 0
        self.__speed_violation_ctr = 0

    def __call__(
        self, old_obj_dict: dict, obj_filters: dict, num_est: int, obj_id: int
    ):
        """Check kinematic plausibility of object"s motion."""
        # get step size in seconds
        dt_step_s = obj_filters[obj_id].dt_s * num_est

        # Check path consistency
        # path by filter
        pos_hist = np.array(old_obj_dict[obj_id]["state"][:2])
        pos_vector = obj_filters[obj_id].x[:2] - pos_hist
        ds_target = np.linalg.norm(pos_vector)

        # path by integration of speed
        ds_integration = dt_step_s * old_obj_dict[obj_id]["state"][3]

        # deviation
        delta_ds_int = abs(ds_target - ds_integration)
        if delta_ds_int > self.__max_err_path:
            if not self.__path_violation_ctr % self.__modulo_logger:
                self.__logger.warning(
                    "KINEMATIC VIOLATION - PATH: deviation = {:.03f} m, num = {:d}".format(
                        delta_ds_int,
                        self.__path_violation_ctr,
                    )
                )
                self.__logger.warning(
                    "KINEMATIC VIOLATION - PATH: prev = "
                    "{} m, vector (x, y) = {} m, norm = {:.05f} m, integration = {:.05f} m, "
                    "v_prev = {:.05f} m/s, residual = {}, K[0, :] = {} K[1, :] = {}".format(
                        pos_hist,
                        pos_vector,
                        old_obj_dict[obj_id]["state"][3],
                        ds_target,
                        ds_integration,
                        obj_filters[obj_id].y,
                        obj_filters[obj_id].K[0, :],
                        obj_filters[obj_id].K[1, :],
                    )
                )
            self.__path_violation_ctr += 1

        # Check yaw consistency
        # # # Yaw by kinematics
        # # quite noisy for high frequent detection (e.g. ansys)
        # yaw_kinematics = math.atan2(pos_vector[1], pos_vector[0]) - np.pi / 2.0
        # delta_yaw_kin = pi_range(obj_filters[obj_id].x[2] - yaw_kinematics)

        # Yaw by integration
        yaw_integration = (dt_step_s * old_obj_dict[obj_id]["state"][4]) + old_obj_dict[
            obj_id
        ]["state"][2]

        # deviation
        delta_yaw_int = pi_range(obj_filters[obj_id].x[2] - yaw_integration)
        if abs(delta_yaw_int) > self.__max_err_yaw:
            if not self.__yaw_violation_ctr % self.__modulo_logger:
                self.__logger.warning(
                    "KINEMATIC VIOLATION - YAW: deviation = {:.03f} rad, num = {:d}".format(
                        delta_yaw_int,
                        self.__yaw_violation_ctr,
                    )
                )
                self.__logger.warning(
                    "KINEMATIC VIOLATION - YAW: prev = "
                    "{:.05f} rad, filter = {:.05f} rad, intergration = {:.05f} rad, "
                    "residual = {}, K[2, :] = {}".format(
                        old_obj_dict[obj_id]["state"][2],
                        obj_filters[obj_id].x[2],
                        yaw_integration,
                        obj_filters[obj_id].y,
                        obj_filters[obj_id].K[2, :],
                    )
                )
            self.__yaw_violation_ctr += 1

        # Check speed consistency
        # speed by integration
        vel_integration = (dt_step_s * old_obj_dict[obj_id]["state"][5]) + old_obj_dict[
            obj_id
        ]["state"][3]

        # deviation
        delta_vel_int = abs(obj_filters[obj_id].x[3] - vel_integration)
        if delta_vel_int > self.__max_err_speed:
            if not self.__speed_violation_ctr % self.__modulo_logger:
                self.__logger.warning(
                    "KINEMATIC VIOLATION - SPEED: deviation = {:.03f} m/s, num = {:d}".format(
                        delta_vel_int,
                        self.__speed_violation_ctr,
                    )
                )
                self.__logger.warning(
                    "KINEMATIC VIOLATION - SPEED: prev = "
                    "{:.05f} m/s, filter = {:.05f} m/s, integration = {:.05f} m/s, "
                    "residual = {}, K[3, :] = {}".format(
                        old_obj_dict[obj_id]["state"][3],
                        obj_filters[obj_id].x[3],
                        vel_integration,
                        obj_filters[obj_id].y,
                        obj_filters[obj_id].K[3, :],
                    )
                )
        self.__speed_violation_ctr += 1
