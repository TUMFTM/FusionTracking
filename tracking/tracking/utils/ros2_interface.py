"""Interface handle for ROS2 interfaces."""
import sys
import time
import configparser
import json

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult, ParameterDescriptor

# Custom imports
from std_msgs.msg import Bool
from tum_msgs.msg import (
    TUMModuleStatus,
    TUMStateEstimate,
    TUMPredictedObjects,
    TUMPredictedObject,
    TUMPredState,
    TUMTrackedObjects,
    TUMTrackedObject,
    TUMTrackedState,
    TUMDetectedObjects,
    TUMControlParam,
)

from .setup_helpers import (
    check_sensor_specification,
    setup_logging,
    log_params,
    overwrite_params,
)
from .map_utils import get_map_data
from .geometry import pi_range
from .detection_subscriber import DetectionSubscriber


class ROS2Handler(Node):
    """Class to handle ros2 node."""

    def __init__(self, path_dict):
        """Intialize ROS 2 node."""
        super().__init__("tracking_node")

        # Set health stats
        self.watchdog = 0
        self.module_state = 10
        self.control_module_state = -1
        self.control_is_connected = False

        # Set history depth for qos
        history_depth = 1

        # Declare parameters
        self.declare_parameter(
            name="frequency",
            descriptor=ParameterDescriptor(dynamic_typing=True),
            value=20.0,
        )
        self.declare_parameter(
            name="max_delay_ego_s",
            value=0.15,
        )
        self.declare_parameter(
            name="checks_enabled",
            value=False,
        )
        self.declare_parameter(
            name="track",
            value="LVMS",
        )
        self.declare_parameter(name="ego_raceline", value="default")
        self.declare_parameter(
            name="send_prediction",
            value=True,
        )
        self.__is_dynamic_typing = ["frequency"]

        # Set paramters callback function
        self.add_on_set_parameters_callback(self._set_parameters_callback)

        # Initialize parameters
        self.params = {
            "frequency": float(self.get_parameter("frequency").value),
            "max_delay_ego_s": self.get_parameter("max_delay_ego_s").value,
            "track": self.get_parameter("track").value,
            "use_sim_time": self.get_parameter("use_sim_time").value,
            "checks_enabled": self.get_parameter("checks_enabled").value,
            "ego_raceline": self.get_parameter("ego_raceline").value,
            "send_prediction": self.get_parameter("send_prediction").value,
        }

        # Setup logging
        self.msg_logger, self.data_logger, self.latency_logger = setup_logging(
            path_dict=path_dict,
        )
        self.latency_dict = {
            "id": 0,
            "t_detect": -1,
            "t_track": -1,
            "t_pred": -1,
            "t_send": -1,
        }
        self.send_watchdog_time = time.time()

        # Add static parameters
        self.get_params_from_config(path_dict=path_dict)

        # Check sensor specification
        check_sensor_specification(main_class=self)

        # SUBSCRIBER NODES #
        # ROS2 Control-WatchDog to Prediction
        self.watchdog_controller_state = self.create_subscription(
            msg_type=TUMModuleStatus,
            topic="/mod_control/status_control",
            callback=self.subscribe_software_state,
            qos_profile=history_depth,
        )
        self.watchdog_controller_state_msg = None

        # ROS2 receive_ego_state from Controller
        self.state_estimation_from_controller_subscriber = self.create_subscription(
            msg_type=TUMStateEstimate,
            topic="/mod_control/vehicle_state_estimate",
            callback=self.subscribe_state_estimation_from_controller,
            qos_profile=history_depth,
        )

        # ROS2 receive number of lap from ltpl
        self.lap_counter_from_ltpl_subscriber = self.create_subscription(
            msg_type=TUMControlParam,
            topic="/mod_local_planner/lap_counter",
            callback=self.subscribe_lap_number,
            qos_profile=history_depth,
        )
        self.__lap_num = -1.0

        # ROS2 receive number of lap from ltpl
        self.pit_bool_from_ltpl_subscriber = self.create_subscription(
            msg_type=Bool,
            topic="/mod_local_planner/on_pitlane",
            callback=self.subscribe_pit_bool,
            qos_profile=history_depth,
        )

        self.detection_subscriber_dict = {}
        self.subscription_dict = {}
        for sensor_str, sensor_specif in self.params["ACTIVE_SENSORS"].items():
            self.detection_subscriber_dict[sensor_str] = DetectionSubscriber(
                sensor_str=sensor_str,
                sensor_specif=sensor_specif,
                get_clock=self.get_clock,
            )

            self.subscription_dict[sensor_str] = self.create_subscription(
                msg_type=TUMDetectedObjects,
                topic=sensor_specif["ros2_topic"],
                callback=self.detection_subscriber_dict[sensor_str].receive_msg,
                qos_profile=history_depth,
            )

        # PUBLISHER NODES #
        # ROS2 Predicted Output to Local Planner
        self.tracked_objects_publisher = self.create_publisher(
            msg_type=TUMTrackedObjects,
            topic="/mod_tracking/TrackedObjects",
            qos_profile=history_depth,
        )

        self.predicted_objects_publisher = self.create_publisher(
            msg_type=TUMPredictedObjects,
            topic="/mod_tracking/PredictedObjects",
            qos_profile=history_depth,
        )

        # ROS2 Watchdog to Control
        self.watchdog_state_to_controller = self.create_publisher(
            msg_type=TUMModuleStatus,
            topic="/mod_tracking/status_tracking",
            qos_profile=history_depth,
        )
        self.watchdog_state_to_controller_msg = TUMModuleStatus()

        # Setup safety checks
        self.__delay_violation_counter_ego = 0
        self.__n_max_delay_violations_ego = 3
        self.__max_delay_ego_ns = int(self.params["max_delay_ego_s"] * 1e9)

        if self.params["checks_enabled"]:
            self.__ego_error_state = 50
        else:
            self.__ego_error_state = 30

        # Set callback timer
        time_period = 1.0 / self.params["frequency"]
        self.timer = self.create_timer(time_period, self.timer_callback)

        # Log all params
        log_params(
            all_params=self.params,
            msg_logger=self.msg_logger,
            module_state=self.module_state,
        )

    def _set_parameters_callback(self, params):
        """Set parameters via callback function.

        Arguments:
            params: Parameters to set, <list>.

        Returns:
            Parameter set result (success), <SetParametersResult>.
        """
        for param in params:
            if param.name in self.params:

                if param.name == rclpy.time_source.USE_SIM_TIME_NAME:
                    self._time_source._on_parameter_event(
                        [Parameter("use_sim_time", Parameter.Type.BOOL, param.value)]
                    )
                    self.params[param.name] = param.value
                elif param.name in self.__is_dynamic_typing:
                    self.params[param.name] = param.value
                elif isinstance(self.params[param.name], float):
                    self.params[param.name] = float(param.value)
                elif isinstance(self.params[param.name], int):
                    self.params[param.name] = int(param.value)
                elif isinstance(self.params[param.name], bool):
                    self.params[param.name] = bool(param.value)
                elif isinstance(self.params[param.name], str):
                    self.params[param.name] = str(param.value)

                self.msg_logger.info(
                    "PARAMETER SET: {} = {}, type = {}".format(
                        param.name,
                        self.params[param.name],
                        type(self.params[param.name]),
                    )
                )
            else:
                return SetParametersResult(successful=False)

        return SetParametersResult(successful=True)

    def get_params_from_config(self, path_dict: dict):
        """Get all params from .ini file."""
        # read in all params
        main_config_parser = configparser.ConfigParser()
        main_config_parser.optionxform = str
        if not main_config_parser.read(path_dict["tracking_config"]):
            raise ValueError(
                "main_config_tracking.ini not found, "
                "path_dict['tracking_config'] = {}".format(
                    path_dict["tracking_config"],
                )
            )

        self.params.update(get_dict_from_tracking_config(main_config_parser))

        # overwrite params if overwrite file present
        main_config_overwrite_parser = configparser.ConfigParser()
        main_config_overwrite_parser.optionxform = str
        if main_config_overwrite_parser.read(path_dict["tracking_config_overwrite"]):
            self.msg_logger.info(
                "OVERWRITE: config from docker_iac, path = {}".format(
                    path_dict["tracking_config_overwrite"]
                )
            )
            params_overwrite = get_dict_from_tracking_config(
                main_config_overwrite_parser
            )
            overwrite_params(self, params_overwrite, switch_only=False)
            overwrite_exists = True
        else:
            self.msg_logger.warning(
                "tracking_config_overwrite.ini not found, "
                "path_dict['tracking_config_overwrite'] = {}".format(
                    path_dict["tracking_config_overwrite"],
                )
            )
            overwrite_exists = False

        # Specify map data paths
        _ = get_map_data(
            main_class=self,
            path_dict=path_dict,
        )

        # Create copy of params for atk-def-switch:
        if overwrite_exists:
            overwrite_params(self, params_overwrite, switch_only=True)

    def subscribe_software_state(self, controller_state):
        """Receive overall software state from mod_control.

        udp interface to mod_control
        message: [watchdog:none:uint8, status:none:uint8]

        data_size = 2 * 4 bytes (uint)
        """
        # Set connection status to true
        if not self.control_is_connected:
            self.msg_logger.info("CONNECTED: connected to mod_control")
            self.control_is_connected = True

        # Set overall software state
        if controller_state.status != self.control_module_state:
            self.msg_logger.info(
                "STATE: mod_control state = {}".format(controller_state.status)
            )
            self.control_module_state = controller_state.status

    def subscribe_state_estimation_from_controller(self, control_msg):
        """Receive ego state and status from the controller."""
        if control_msg.status == 2:
            if (
                "t" in self.object_handler.ego_state
                and control_msg.time_ns == self.object_handler.ego_state["t"]
            ):
                self.msg_logger.warning("REDUNDANT: ego-state")
                return

            # reject ego state if position is nan
            if np.isnan(control_msg.x_cg_m) or np.isnan(control_msg.y_cg_m):
                self.msg_logger.warning(
                    "INVALID TYPE: Rejected ego-state, "
                    "control_msg.x_cg_m = {}, control_msg.y_cg_m = {}".format(
                        control_msg.x_cg_m,
                        control_msg.y_cg_m,
                    )
                )
                return

            # debug
            if not isinstance(control_msg.time_ns, int):
                self.msg_logger.error(
                    "INVALID TYPE: type(control_msg.time_ns) != int, but {}".format(
                        type(control_msg.time_ns)
                    )
                )
                control_msg.time_ns = int(control_msg.time_ns)

            t_ego = control_msg.time_ns

            self.object_handler.ego_state["t"] = t_ego
            self.object_handler.ego_state["state"] = [
                control_msg.x_cg_m,
                control_msg.y_cg_m,
                pi_range(control_msg.psi_cg_rad),
                np.linalg.norm(
                    [
                        control_msg.vx_cg_mps,
                        control_msg.vy_cg_mps,
                    ],
                    ord=2,
                ),
                control_msg.dpsi_cg_radps,
                control_msg.ax_cg_mps2,
            ]
            self.object_handler.ego_state["is_updated"] = True
            self.object_handler.ego_state[
                "tracking_id"
            ] = self.object_handler.tracking_id
            self.object_handler.tracking_id += 1
        else:
            self.msg_logger.warning(
                "ERROR STATUS: control_msg.status = {}".format(control_msg.status)
            )

    def subscribe_lap_number(self, lap_msg):
        """Receive number of current lap from ltpl.

        Message: TUMControlParam
        """
        if int(lap_msg.value) != int(self.__lap_num):
            self.msg_logger.info("LAP NUMBER: new lap = {:.02f}".format(lap_msg.value))
        self.__lap_num = lap_msg.value

    def subscribe_pit_bool(self, pit_bool):
        """Receive boolean for pit status of ego."""
        if "in_pit" not in self.object_handler.ego_state:
            self.msg_logger.info("PIT BOOL: init pit bool = {}".format(pit_bool.data))
            self.object_handler.ego_state["in_pit"] = pit_bool.data
        elif pit_bool.data != self.object_handler.ego_state["in_pit"]:
            self.msg_logger.info("PIT BOOL: new pit bool = {}".format(pit_bool.data))
            self.object_handler.ego_state["in_pit"] = pit_bool.data

    def check_module_state(self):
        """Check module state and exit running module in case of emergency state.

        30: valid running state
        50: emergency state
        """
        if self.module_state != 30:

            if self.module_state >= 50:
                msg_log = self.msg_logger.error
                shutdown_node = True
            else:
                msg_log = self.msg_logger.warning
                shutdown_node = False

            msg_log("STATE: tracking_node state = {:d}".format(self.module_state))

            if shutdown_node:
                for _ in range(0, 5):
                    self.publish_watchdog()
                    time.sleep(0.1)
                msg_log("STATE: tracking_node state = {:d}".format(self.module_state))
                self.shutdown_ros()

    def check_ego_status(self):
        """Check if all values are valid and delay is accepable."""
        if self.module_state != 30:
            return

        if "state" not in self.object_handler.ego_state:
            self.msg_logger.warning(
                "MISSING EGO STATE: no check possible, "
                "ego_state = {}, control_module_state = {}".format(
                    self.object_handler.ego_state,
                    self.control_module_state,
                )
            )
            return

        if self.object_handler.ego_state["is_updated"] and (
            any(
                [
                    np.isnan(val) or np.isinf(val)
                    for val in self.object_handler.ego_state["state"]
                ]
            )
            or np.isnan(self.object_handler.ego_state["t"])
            or np.isinf(self.object_handler.ego_state["t"])
        ):
            self.msg_logger.error(
                "TYPE ERROR: Invalid numeric value, ego, time_stamp: {}".format(
                    self.object_handler.ego_state["t"],
                )
            )
            self.module_state = self.__ego_error_state

        if (
            self.object_handler.tracking_timestamp_ns
            - self.object_handler.ego_state["t"]
            > self.__max_delay_ego_ns
        ):
            self.__delay_violation_counter_ego += 1
            self.msg_logger.error(
                "EGO TIME VIOLATION: "
                "t_track - t_ego = {:.0f} ms, num = {}".format(
                    (
                        self.object_handler.tracking_timestamp_ns
                        - self.object_handler.ego_state["t"]
                    )
                    / 1e6,
                    self.__delay_violation_counter_ego,
                )
            )
            self.msg_logger.warning(
                "UPDATED EGO: is_updated = {}, t_track =  {:d} ns,"
                " t_ego = {:d} ns, t_track - t_ego = {:.0f} ms".format(
                    self.object_handler.ego_state["is_updated"],
                    self.object_handler.tracking_timestamp_ns,
                    self.object_handler.ego_state["t"],
                    (
                        self.object_handler.tracking_timestamp_ns
                        - self.object_handler.ego_state["t"]
                    )
                    / 1e6,
                )
            )
        else:
            self.__delay_violation_counter_ego = 0

        if self.__delay_violation_counter_ego > self.__n_max_delay_violations_ego:
            self.module_state = self.__ego_error_state

    def check_send_empty(self):
        """Check if empty prediction should be sent."""
        if self.object_handler.ego_state["in_pit"]:
            return False
        if self.__lap_num <= self.params["TRACKING"]["prediction"]["lap_wo_pred"]:
            return True

        return False

    def publish(self):
        """Publish all topics."""
        # Send object list
        self.publish_object_list()

        # Send prediction
        if self.params["send_prediction"]:
            self.publish_prediction()

        # Send watchdog
        self.publish_watchdog()

    def publish_object_list(self):
        """Publish object list of tracked objects."""
        tracked_objects = TUMTrackedObjects()

        # check if there are tracked objects
        tracked_objects.empty = not bool(self.object_handler.observation_storage)

        # iterate through objects
        for object_id, obj_el in self.object_handler.observation_storage.items():
            tracked_object = TUMTrackedObject()

            tracked_object.tracking_id = obj_el["tracking_id"]
            tracked_object.object_id = str(object_id)
            tracked_object.t_abs_perception = obj_el["t"][0]

            # iterate through state per object
            for n_track in range(
                0,
                min(len(obj_el["t"]), self.params["TRACKING"]["n_max_obsv"]),
                self.params["TRACKING"]["publish_downsample_factor"],
            ):
                trackstate = TUMTrackedState()
                trackstate.t = obj_el["t"][n_track]
                trackstate.x = obj_el["state"][n_track][0]
                trackstate.y = obj_el["state"][n_track][1]
                trackstate.yaw = obj_el["state"][n_track][2]
                trackstate.v = obj_el["state"][n_track][3]
                trackstate.dyaw = obj_el["state"][n_track][4]
                trackstate.a = 0.0

                tracked_object.track.append(trackstate)

            tracked_objects.objects.append(tracked_object)

        # publish topic
        self.tracked_objects_publisher.publish(tracked_objects)

    def publish_prediction(self):
        """ROS sender to send prediction-dict to mod_local_planner."""
        predicted_objects = TUMPredictedObjects()

        # send prediction
        if self.check_send_empty():
            predicted_objects.empty = True
        elif self.physics_prediction.pred_dict is not None:
            predicted_objects.empty = not bool(self.physics_prediction.pred_dict)

            id_list = [*self.physics_prediction.pred_dict.keys()]
            if "ego" in id_list:
                id_list.remove("ego")
            for num_ids in id_list:
                predicted_object = TUMPredictedObject()
                predicted_object.prediction_id = self.physics_prediction.pred_dict[
                    num_ids
                ]["prediction_id"]
                predicted_object.valid = bool(
                    self.physics_prediction.pred_dict[num_ids]["valid"]
                )
                predicted_object.vehicle_id = str(num_ids)
                predicted_object.prediction_type = str(
                    self.physics_prediction.pred_dict[num_ids]["prediction_type"]
                )
                predicted_object.t_abs_perception = int(
                    self.physics_prediction.pred_dict[num_ids]["t_abs_perception"]
                )

                if self.physics_prediction.pred_dict[num_ids]["valid"]:
                    for num_preds in range(
                        0, len(self.physics_prediction.pred_dict[num_ids]["t"])
                    ):
                        predstate = TUMPredState()
                        predstate.t = self.physics_prediction.pred_dict[num_ids]["t"][
                            num_preds
                        ]
                        predstate.x = self.physics_prediction.pred_dict[num_ids]["x"][
                            num_preds
                        ]
                        predstate.y = self.physics_prediction.pred_dict[num_ids]["y"][
                            num_preds
                        ]
                        predstate.heading = self.physics_prediction.pred_dict[num_ids][
                            "heading"
                        ][num_preds]
                        predicted_object.pred.append(predstate)
                else:
                    predstate = TUMPredState()
                    predstate.t = self.physics_prediction.pred_dict[num_ids]["t"]
                    predstate.x = self.physics_prediction.pred_dict[num_ids]["x"]
                    predstate.y = self.physics_prediction.pred_dict[num_ids]["y"]
                    predstate.heading = self.physics_prediction.pred_dict[num_ids][
                        "heading"
                    ]
                    predicted_object.pred.append(predstate)

                predicted_objects.objects.append(predicted_object)

        # publish topic
        self.predicted_objects_publisher.publish(predicted_objects)

    def publish_watchdog(self):
        """Ros sender to send watchdog 0 .. 255 and state to mod_control."""
        self.watchdog = (self.watchdog + 1) % 255
        self.watchdog_state_to_controller_msg.watchdog = self.watchdog
        self.watchdog_state_to_controller_msg.status = self.module_state
        self.watchdog_state_to_controller.publish(self.watchdog_state_to_controller_msg)
        self.send_watchdog_time = time.time()

    def shutdown_ros(self, sig=None, frame=None):
        """Close all ros nodes."""
        self.main_logger.info("Closing ROS Nodes...")
        self.destroy_node()
        rclpy.shutdown()
        time.sleep(0.5)
        self.main_logger.info("Shutdown complete!")

        sys.exit(0)


def get_dict_from_tracking_config(config_parser):
    """Get dictionary from tracking config."""
    param_dict = config_parser._sections
    for key, val in param_dict.items():
        for sub_key, sub_val in val.items():
            param_dict[key][sub_key] = json.loads(sub_val)
    return param_dict
