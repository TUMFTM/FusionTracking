"""Subscriber class for TUM detection messages."""
import logging

import numpy as np
from pyquaternion import Quaternion
from utils.setup_helpers import stamp2time


class DetectionSubscriber:
    """Class to subscribe ROS2-Message TUMDetectedObjects."""

    def __init__(self, sensor_str: str, sensor_specif: dict, get_clock):
        """Initialize subscribe with sensor model."""
        self.__logger = logging.getLogger("msg_logger")
        self.get_clock = get_clock

        self.__sensor_str = sensor_str
        self.__state_keys = sensor_specif["meas_vars"]
        self.__bool_global_coord = not sensor_specif.get("bool_local", False)
        self.__individual_header = sensor_specif.get("individual_header", False)
        self.__last_detection_timestamp_ns = None
        self.__detection_input = (None, -1)
        self.__max_delay_detection_ns = int(sensor_specif["max_delay_s"] * 1e9)
        self.__float_type_check = True
        self.__delay_violation_counter = 0
        self.__n_max_delay_violations = sensor_specif.get("n_max_delay_violations", 5)
        self.__is_yaw_measured = bool("meas" in sensor_specif["get_yaw"])
        self.__yaw_from_track = bool(sensor_specif["get_yaw"] == "from_track")
        self.__add_speed = bool("v" in self.__state_keys)
        self.__add_yawrate = bool("yawrate" in self.__state_keys)
        if sensor_specif["get_yaw"] and "yaw" not in sensor_specif["meas_vars"]:
            raise KeyError("'yaw' not in sensor_specif")

        # Set escalation level
        if sensor_specif.get("is_fallback", False):
            self.__log_level = "error"
        else:
            self.__log_level = "warning"

        self.__latency_log_list = [-1, -1, -1, -1]

    def __get_detection_input(self):
        """Getter method."""
        return self.__detection_input

    detection_input = property(__get_detection_input)

    def __get_latency_log_list(self):
        """Getter method."""
        return self.__latency_log_list

    latency_log_list = property(__get_latency_log_list)

    def __get_state_keys(self):
        """Getter method."""
        return self.__state_keys

    state_keys = property(__get_state_keys)

    def __get_yaw_from_track(self):
        """Getter method."""
        return self.__yaw_from_track

    yaw_from_track = property(__get_yaw_from_track)

    def check_delayed(self, time_stamp: int):
        """Check if detection is above maximal delay."""
        if self.__last_detection_timestamp_ns is None:
            return 30

        # Check delay
        dt_ns = time_stamp - self.__last_detection_timestamp_ns
        if dt_ns > self.__max_delay_detection_ns:
            self.__delay_violation_counter += 1
            tmp_logger = self.__logger.__getattribute__(self.__log_level)
            tmp_logger(
                "TIME VIOLATION: "
                "t_track - t_detect = {:.03f} s, sensor: {}, num: {}".format(
                    dt_ns / 1e9,
                    self.__sensor_str,
                    self.__delay_violation_counter,
                )
            )
        else:
            self.__delay_violation_counter = 0

        # return state
        if self.__delay_violation_counter > self.__n_max_delay_violations:
            return 50

        return 30

    def reset(self):
        """Reset detection input after each run."""
        self.__detection_input = (None, -1)

    def reset_latency_log_list(self):
        """Reset latency_log_list."""
        self.__latency_log_list = [-1, -1, -1, -1]

    def add_num_valid_objects(self, num_not_filtered: int):
        """Add number of valid object to latency logs."""
        self.__latency_log_list[3] = num_not_filtered

    def receive_msg(self, msg):
        """Receive and convert ros2 message."""
        # log latency
        self.__latency_log_list[1] = stamp2time(
            *self.get_clock().now().seconds_nanoseconds()
        )
        self.__latency_log_list[0] = msg.latency_id

        # check input types
        self.__float_type_check = True

        # get timestamp
        if self.__individual_header and len(msg.objects):
            t_msg_ns = self.get_timestamp_from_individual_header(msg=msg)
        else:
            t_msg_ns = int(msg.header.stamp.sec * 1e9) + msg.header.stamp.nanosec

        # if message is redundand, do no further msg process
        if t_msg_ns == self.__last_detection_timestamp_ns:
            self.__logger.error("REDUNDANT: {}".format(self.__sensor_str))
            self.__detection_input = (None, -1)
            return

        # Check delay
        dt_ns = stamp2time(*self.get_clock().now().seconds_nanoseconds()) - t_msg_ns
        if dt_ns > self.__max_delay_detection_ns:
            self.__logger.warning(
                "TIME VIOLATION: sensor: {}, ignore detection input, "
                "t_now - t_detect = {:.02f} ms".format(
                    self.__sensor_str,
                    dt_ns / 1e6,
                )
            )
            self.__latency_log_list[2] = len(msg.objects)
            return

        # get detected objects
        list_objects = []
        for object_el in msg.objects:
            # check for zero-only object:
            if (
                abs(object_el.kinematics.pose.pose.position.x)
                + abs(object_el.kinematics.pose.pose.position.y)
                == 0.0
            ):
                self.__logger.error(
                    "ZERO OBJECT: x-, y-positions = (0.0, 0.0), "
                    "object ignored, sensor: {}".format(self.__sensor_str)
                )
                self.__logger.warning("objects msg: {}".format(msg.objects))
                continue

            # create state list
            state_tmp = [
                object_el.kinematics.pose.pose.position.x,
                object_el.kinematics.pose.pose.position.y,
            ]

            # add yaw
            if self.__is_yaw_measured:
                state_tmp.append(self.get_yaw_from_quat(object_el=object_el))

            # add speed
            if self.__add_speed:
                state_tmp.append(
                    np.linalg.norm(
                        [
                            object_el.kinematics.twist.twist.linear.x,
                            object_el.kinematics.twist.twist.linear.y,
                        ]
                    )
                )

            # add yaw rate: yawrate
            if self.__add_yawrate:
                state_tmp.append(object_el.kinematics.twist.twist.angular.z)

            # add object to detection input list
            # Check for valid format
            if self.check_valid_state(state_tmp, t_msg_ns):
                list_objects.append(
                    {"state": state_tmp, "t": t_msg_ns, "keys": self.__state_keys}
                )

        # all objects are invalid, detection is treated as not received
        self.write_to_detection_input(list_objects=list_objects, t_msg_ns=t_msg_ns)

        # log latency
        if self.__detection_input[0] is not None:
            self.__latency_log_list[2] = len(list_objects)

    def write_to_detection_input(self, list_objects, t_msg_ns):
        """Write object list to detection input for tracking."""
        # Write tuple to detection input for tracking
        if self.__float_type_check or list_objects:
            self.__last_detection_timestamp_ns = t_msg_ns
            self.__detection_input = (list_objects, t_msg_ns)
        else:
            self.__logger.error(
                "TYPE ERROR: All objects invalid, sensor: {}, time_stamp: {}".format(
                    self.__sensor_str,
                    t_msg_ns,
                )
            )
            self.__detection_input = (None, -1)

    def check_valid_state(self, state_tmp: list, t_msg_ns: int):
        """Check if the state is valid.

        Args:
            state_tmp (list): Temporary kinematic state of object
            t_msg_ns (int): Timestamp of message in ns

        Returns:
            bool: True if all state entries are valid, otherwise false
        """
        if (
            any([np.isnan(val) or np.isinf(val) for val in state_tmp])
            or np.isnan(t_msg_ns)
            or np.isinf(t_msg_ns)
        ):
            self.__float_type_check = False
            self.__logger.error(
                "TYPE ERROR: Invalid numeric value, sensor: {}, time_stamp: {}".format(
                    self.__sensor_str,
                    t_msg_ns,
                )
            )
            return False
        return True

    def get_yaw_from_quat(self, object_el):
        """Get yaw in radian from quaternions.

        Args:
            object_el (namespace): Object element in TUMDetectedObject format.

        Returns:
            float: Yaw in global cartesian coordinates in rad.
        """
        yaw, _, _ = Quaternion(
            object_el.kinematics.pose.pose.orientation.w,
            object_el.kinematics.pose.pose.orientation.x,
            object_el.kinematics.pose.pose.orientation.y,
            object_el.kinematics.pose.pose.orientation.z,
        ).yaw_pitch_roll
        if self.__bool_global_coord:
            yaw -= np.pi / 2.0
            # Roborace (ours): yaw is 0.0 pointing towards y-axis (global "north")
            # Euler (default): yaw is 0.0 pointing towards x-axis (global "east")

        return yaw

    def get_timestamp_from_individual_header(self, msg):
        """Get unified timestamp for detection input with individual header timestamps.

        Args:
            msg (TUMDetectedObjects): TUMDetectedObjects message

        Returns:
            int: Unified message timestamp in ns
        """
        individual_timestamp_list = [
            int(object_el.header.stamp.sec * 1e9) + object_el.header.stamp.nanosec
            for object_el in msg.objects
        ]
        t_msg_ns = int(np.mean(individual_timestamp_list))

        # check timegap
        if len(individual_timestamp_list) > 1:
            diff_timestamps_ms = np.diff(individual_timestamp_list) / 1e6
            max_timegap_ms = (
                max(individual_timestamp_list) - min(individual_timestamp_list)
            ) / 1e6

            if max_timegap_ms > 50.0:
                self.__logger.warning(
                    "OBJECT LIFETIME INCONSISTENCY - OBJECTS: dt_max = {:.02f} ms".format(
                        max_timegap_ms
                    )
                )
                self.__logger.info(
                    "INDIVIDUAL TIMESTAMP: "
                    "np.diff() = {} ms".format(np.round(diff_timestamps_ms, 1))
                )

                self.__logger.info(
                    "INDIVIDUAL TIMESTAMP: "
                    "{} objects, t_avg: = {} ns".format(
                        len(individual_timestamp_list), individual_timestamp_list
                    )
                )

                self.__logger.info(msg)

        # compare to header message
        t_head = int(msg.header.stamp.sec * 1e9) + msg.header.stamp.nanosec
        max_timegap_header_ms = (t_head - min(individual_timestamp_list)) / 1e6
        if max_timegap_header_ms > 100.0:
            self.__logger.warning(
                "OBJECT LIFETIME INCONSISTENCY - HEADER: dt_max = {:.02f} ms".format(
                    max_timegap_header_ms
                )
            )
            self.__logger.info(
                "INDIVIDUAL TIMESTAMP: sensor = {}, t_avg = {} ns, "
                "header = {} ns, dt = {:.02f} ms".format(
                    self.__sensor_str,
                    t_msg_ns,
                    t_head,
                    (t_head - t_msg_ns) / 1e6,
                )
            )
            self.__logger.info(msg)
        return t_msg_ns
