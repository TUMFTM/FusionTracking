"""Main functionalities for multi-sensor object tracking."""
# pylint: disable=C0413,W1202,W0703
import logging
import copy
import time

import numpy as np

from utils.tracking_utils import (
    first_obj_logger,
    log_filter_vals,
    check_input_dimension,
    check_none_realistic_object,
    get_initialized_state,
    get_new_storage_item,
    reduce_list,
    check_estimated_dict,
    check_observation_storage,
)
from utils.setup_helpers import stamp2time
from .object_matching import ObjectMatching
from .lowpass_filter import LowpassFilter
from .check_collisions import CheckCollisions
from .transform_objects import TransformObjects
from .ekf import EKF


class ObjectHandler:
    """Class object central for multi-sensor object tracking."""

    def __init__(
        self, all_params: dict, track_boundary_cls, pit_boundary_cls, get_clock
    ):
        """Initialize object tracking with arguments.

        Args:
            all_params (dict): Parameter dict from main_tracking_config.ini
            track_boundary_cls (class): Track boundary class, see src-folder
            pit_boundary_cls (class): Pit boundary class, see src-folder

        Raises:
            IndexError: Raised if  no fallback sensor is specified
        """
        # Create logger
        self.logger = logging.getLogger("msg_logger")
        self.get_clock = get_clock

        # Timing
        self.tracking_timestamp_ns = int(0)

        # Observation storage
        self.old_object_dict = {}
        self.observation_storage = {}
        self.state_vars = all_params["TRACKING"]["state_vars"]
        self.tracking_id = 0
        self.__n_max_obsv = all_params["TRACKING"]["n_max_obsv"]
        self.__add_ego_variables = all_params["TRACKING"]["add_ego_variables"]
        self.__ego_speed_factor = all_params["TRACKING"]["ego_speed_factor"]

        self.__bool_bounds_check = all_params["TRACKING"]["bounds_check"][
            "bool_bounds_check"
        ]

        # Shared bound classes
        self.track_boundary = track_boundary_cls
        self.pit_boundary = pit_boundary_cls

        # Detection processing
        self.detection_input = None
        self.bool_detection_input = False
        self.__is_local_detection = {
            sensor_str: val["bool_local"]
            for sensor_str, val in all_params["ACTIVE_SENSORS"].items()
        }
        self.detection_obj_dict = {
            sensor_str: (None, -1) for sensor_str in all_params["ACTIVE_SENSORS"]
        }
        self.__detection_ctr = {
            sensor_str: [0, 0, 0] for sensor_str in all_params["ACTIVE_SENSORS"]
        }
        self.__detection_ctr["calls"] = 0
        self.tracking_input_list = []
        self.preprocessed_tracking_input_list = []

        # Ego state
        self.ego_state = {
            "is_updated": False,
            "in_pit": False,
            "tracking_id": -1,
        }
        self.__bool_ego_lowpass = all_params["TRACKING"]["ego_lowpass"]["bool_lowpass"]
        self.object_lowpass = LowpassFilter(
            params=all_params["TRACKING"]["ego_lowpass"]
        )

        # Filter
        self.obj_filters = {}
        self.filter_log = {}
        self.num_est = 1
        self.estimated_object_dict = {}
        self.skipped_object_dict = {}
        self.filter_timestep_ns = int(1e9 / all_params["TRACKING"]["filter_frequency"])
        self.filter_params = all_params["TRACKING"]["filter"]
        self.filter_params["P_init"] = np.diag(self.filter_params["P_init"])
        self.filter_params["timestep_s"] = (
            1.0 / all_params["TRACKING"]["filter_frequency"]
        )
        self.filter_params["measure_covs_dict"] = {
            sensor_name: np.diag([std_dev**2 for std_dev in specif["std_dev"]])
            for sensor_name, specif in all_params["ACTIVE_SENSORS"].items()
        }

        # Matching
        self.matching = ObjectMatching(
            matching_params=all_params["TRACKING"]["matching"],
            sensor_models=all_params["ACTIVE_SENSORS"],
        )

        # Collision check ego-object
        self.check_collisions = CheckCollisions(
            collisions_params=all_params["TRACKING"]["collision_check"]
        )
        # Collison check object-object
        self.__check_overlap = all_params["TRACKING"]["overlap_check"][
            "bool_overlap_check"
        ]
        self.overlap_dist_m = all_params["TRACKING"]["overlap_check"]["overlap_dist_m"]

        # Safety checks
        self.__fallback_sensors = [
            sensor_str
            for sensor_str, sensor_specif in all_params["ACTIVE_SENSORS"].items()
            if sensor_specif.get("is_fallback", False)
        ]
        if all_params["ACTIVE_SENSORS"] and not self.__fallback_sensors:
            raise IndexError(
                "No fallback sensor specified, "
                "set 'is_fallback': true in SENSOR_MODELS"
            )
        self.is_connecting_to_detection = list(all_params["ACTIVE_SENSORS"])
        self.is_not_empty = list(all_params["ACTIVE_SENSORS"])
        self.ros_dt_s = (
            1.0 / all_params["frequency"]
            + 1.0 / all_params["TRACKING"]["filter_frequency"]
        )
        self.__max_estimation_iterations = int(
            self.ros_dt_s * all_params["TRACKING"]["filter_frequency"]
        )
        self.is_time_violation = False
        self.__checks_enabled = all_params["checks_enabled"]

        # init transform_objects
        self.transform_objects = TransformObjects(all_params["MISCS"]["rear_ax_geoc_m"])

    def sync_time(self):
        """Update tracking timestamp."""
        if self.tracking_timestamp_ns:
            # Check filter frequency
            tic_ns = (
                stamp2time(*self.get_clock().now().seconds_nanoseconds())
                - self.tracking_timestamp_ns
            )

            # Logging
            if tic_ns < 0:
                self.logger.error(
                    "Negative iteration duration: {:d} ms".format(int(tic_ns / 1e6))
                )

            # Determine filter iterations
            if tic_ns < self.filter_timestep_ns:
                # Determine residual and take a nap
                dt_ns = self.filter_timestep_ns - tic_ns
                time.sleep(dt_ns / 1e9)

                # Update tracking timestamp
                self.num_est = 1

                # debug
                self.logger.warning(
                    "ABOVE FREQUENCY: too fast, dt_sleep = {:.02f} ms".format(
                        dt_ns / 1e6
                    )
                )

            else:
                # Estimate multiple times
                # keep tracking frequency equidistant
                add_one = 1
                if tic_ns % self.filter_timestep_ns == 0:
                    add_one = 0

                # Update tracking timestamp
                self.num_est = tic_ns // self.filter_timestep_ns + add_one

            self.tracking_timestamp_ns += self.num_est * self.filter_timestep_ns

            # Logging
            if self.num_est > self.__max_estimation_iterations:
                self.logger.warning(
                    "TIME VIOLATION: dt_cycle = "
                    "{:.02f} ms, dt_max: {:.02f} ms, n_iter = {:d}".format(
                        tic_ns / 1e6,
                        self.ros_dt_s * 1e3,
                        self.num_est,
                    )
                )
                self.is_time_violation = True

        else:
            # Initialize tracking timestamp
            self.tracking_timestamp_ns = stamp2time(
                *self.get_clock().now().seconds_nanoseconds()
            )
            self.logger.info(
                "INITIALIZATION: t_track = {:d} ns".format(self.tracking_timestamp_ns)
            )

    def estimate(self):
        """Estimate object states."""
        # Reset tracking variables
        self.detection_input = None
        self.matching.reset_match_dict()

        # Estimate object states
        self.estimated_object_dict = {}

        for obj_id in self.old_object_dict:

            x_list = [None] * self.num_est
            t_list = [None] * self.num_est
            p_list = [None] * self.num_est

            for n_est in range(self.num_est):
                # Iterative estimation step
                self.obj_filters[obj_id].predict()

                x_list[n_est] = self.obj_filters[obj_id].x.tolist()
                t_list[n_est] = int(
                    self.tracking_timestamp_ns
                    - (self.num_est - n_est - 1) * self.filter_timestep_ns
                )
                p_list[n_est] = self.obj_filters[obj_id].P.tolist()

            self.observation_storage[obj_id]["state"].extendleft(x_list)
            self.observation_storage[obj_id]["t"].extendleft(t_list)
            self.observation_storage[obj_id]["P"].extendleft(p_list)

            # Store predicted states
            self.estimated_object_dict[obj_id] = {
                "state": x_list[-1],
                "t": t_list[-1],
            }

            # Reset filter values for delay compensation
            self.obj_filters[obj_id].reset()

    def compensate(self):
        """Compensate object states."""
        tic = stamp2time(*self.get_clock().now().seconds_nanoseconds())

        for obj_id, obj_filt in self.obj_filters.items():
            t_obj = int(self.tracking_timestamp_ns)

            for n_est in range(obj_filt.idx_hist + 1):
                # n_updates = num_updates_list[0]
                # idx_hist = idx_hist_list[0]

                # predict back to present
                if n_est > 0:
                    obj_filt.predict()

                t_obj = int(
                    self.tracking_timestamp_ns
                    - (obj_filt.idx_hist - n_est) * self.filter_timestep_ns
                )

                self.observation_storage[obj_id]["state"][
                    obj_filt.idx_hist - n_est
                ] = obj_filt.x.tolist()
                self.observation_storage[obj_id]["P"][
                    obj_filt.idx_hist - n_est
                ] = obj_filt.P.tolist()

            # Reset filter values for delay compensation
            obj_filt.reset()

            # Store predicted states
            self.estimated_object_dict[obj_id] = {
                "state": obj_filt.x.tolist(),
                "t": t_obj,
            }

        toc = (stamp2time(*self.get_clock().now().seconds_nanoseconds()) - tic) / 1e6
        if toc > 6.0:
            self.logger.warning(
                "COMPENSATE STORAGE: t_calc = {:.02f} ms, num_obj = {}".format(
                    toc,
                    len(self.obj_filters),
                )
            )

    def process_detections(self, detections_subscriber_dict: dict) -> int:
        """Get detections from ros2 subscriber."""
        self.tracking_input_list = []
        self.preprocessed_tracking_input_list = []
        self.detection_obj_dict = {
            sensor_str: subscriber.detection_input
            for sensor_str, subscriber in detections_subscriber_dict.items()
            if subscriber.detection_input[0] is not None
        }

        # reset detection input
        _ = {subscriber.reset() for subscriber in detections_subscriber_dict.values()}

        # Logging
        if self.is_connecting_to_detection or self.is_not_empty:
            first_obj_logger(
                detection_obj_dict=self.detection_obj_dict,
                is_connecting=self.is_connecting_to_detection,
                is_not_empty=self.is_not_empty,
                logger=self.logger,
            )

        # Check for vaild detections
        self.bool_detection_input = bool(self.detection_obj_dict)

        # return module state
        if set(self.is_connecting_to_detection).intersection(self.__fallback_sensors):
            self.logger.warning(
                "NOT CONNECTED: Waiting for fallback sensor: {}".format(
                    self.__fallback_sensors
                )
            )
            if self.__checks_enabled:
                return 20

        return 30

    def __call__(
        self,
        detections_subscriber_dict: dict,
        module_state,
    ) -> int:
        """Process detected objects: match, compensate delay, update state, check for collisions."""
        if module_state != 30:
            return

        # Estimation Step
        self.estimate()

        # Reset for logging
        self.filter_log = {}

        # Check for ego
        if "state" not in self.ego_state:
            self.logger.warning("MISSING EGO STATE: no ego state received")
            return

        # Safety checks: delay of detection pipelines
        conn_status_list = [
            detections_subscriber_dict[sensor_str].check_delayed(
                self.tracking_timestamp_ns
            )
            == 50
            for sensor_str in self.__fallback_sensors
        ]
        if conn_status_list and all(conn_status_list) and self.__checks_enabled:
            module_state = 50

        if self.ego_state["is_updated"]:
            self.store_ego()

        self.__detection_ctr["calls"] += 1

        # Process detection inputs
        if self.bool_detection_input:
            # check if ego state is up to date
            if not self.ego_state["is_updated"]:
                self.logger.warning(
                    "OUTDATED EGO STATE: outdated ego state, "
                    "t_track - t_ego = {:.02f} ms".format(
                        (self.tracking_timestamp_ns - self.ego_state["t"]) / 1e6
                    )
                )

            # Logging
            self.detection_input = copy.deepcopy(self.detection_obj_dict)

            # Sorted by timestamp (oldest detection first)
            self.tracking_input_list = list(
                sorted(self.detection_obj_dict.items(), key=lambda item: item[1][1])
            )

            # Sort detection input
            detection_timestamp_iter_ns = iter(
                (det_list[1][1] for det_list in self.tracking_input_list[1:])
            )

            for sensor_str, (
                new_object_list,
                detection_timestamp_ns,
            ) in self.tracking_input_list:
                self.skipped_object_dict = {}
                detection_timestamp_next_ns = next(detection_timestamp_iter_ns, None)

                # log detection input: unfiltered
                self.__detection_ctr[sensor_str][1] += bool(new_object_list)

                if new_object_list:

                    # get delayed states
                    ego_t, ego_state = self.get_delayed_object_states(
                        detection_timestamp_ns=detection_timestamp_ns,
                        detection_timestamp_next_ns=detection_timestamp_next_ns,
                        sensor_str=sensor_str,
                    )

                    # transform detection input from local to global coordinates
                    if self.__is_local_detection[sensor_str]:
                        self.transform_objects(
                            new_object_list=new_object_list,
                            yaw_from_track=detections_subscriber_dict[
                                sensor_str
                            ].yaw_from_track,
                            detection_timestamp_ns=detection_timestamp_ns,
                            ego_t=ego_t,
                            ego_state=ego_state,
                        )

                    # check out of bound and add yaw from track position
                    is_out_of_track = self.track_boundary.yaw_and_bounds(
                        new_object_list,
                        add_yaw=detections_subscriber_dict[sensor_str].yaw_from_track,
                        sensor_str=sensor_str,
                        logger_fn=self.logger.info,
                    )

                    # check input dimensions
                    new_object_list = [
                        new_object
                        for new_object in new_object_list
                        if check_input_dimension(
                            new_object=new_object,
                            state_keys=detections_subscriber_dict[
                                sensor_str
                            ].state_keys,
                            sensor_str=sensor_str,
                            logger=self.logger,
                        )
                    ]

                    # debug
                    dimension_wrong = False
                    for obj in new_object_list:
                        if sensor_str == "radar" and len(obj["state"]) > 4:
                            dimension_wrong = True
                        if sensor_str == "lidar_cluster" and len(obj["state"]) > 3:
                            dimension_wrong = True

                    # debug
                    if dimension_wrong:
                        self.logger.error(
                            "DIMENSION ERROR: sensor = {}".format(sensor_str)
                        )
                        self.logger.info("new_object_list = {}".format(new_object_list))
                        self.logger.info(
                            "self.preprocessed_tracking_input_list = {}".format(
                                self.preprocessed_tracking_input_list
                            )
                        )
                        self.logger.info(
                            "self.detection_input = {}".format(self.detection_input)
                        )
                        new_object_list = []
                    elif not new_object_list:
                        self.logger.warning(
                            "DIMENSION ERROR: all new objects are invalid"
                        )

                    # Check if objects from detection input are within the track bounds
                    if new_object_list and self.__bool_bounds_check:
                        is_out_of_pit = self.pit_boundary.yaw_and_bounds(
                            new_object_list,
                            add_yaw=False,
                            sensor_str=sensor_str,
                            logger_fn=self.logger.info,
                        )

                        reduce_list(
                            is_out_of_track=is_out_of_track,
                            is_out_of_pit=is_out_of_pit,
                            new_object_list=new_object_list,
                            logger=self.logger,
                            sensor_str=sensor_str,
                        )

                    # check for none realistic objects
                    if self.__check_overlap:
                        ((sensor_str, (new_object_list, detection_timestamp_ns)),) = [
                            check_none_realistic_object(
                                measure_tuple=(
                                    sensor_str,
                                    (new_object_list, detection_timestamp_ns),
                                ),
                                overlap_dist_m=self.overlap_dist_m,
                                logger=self.logger,
                            )
                        ]

                        self.preprocessed_tracking_input_list.append(
                            (sensor_str, (new_object_list, detection_timestamp_ns)),
                        )

                # log latency stats
                detections_subscriber_dict[sensor_str].add_num_valid_objects(
                    num_not_filtered=len(new_object_list)
                )

                # log detection input stats
                self.log_input_stats(
                    sensor_str=sensor_str, is_non_empty=bool(new_object_list)
                )

                # Filter and matching, discard old objects
                self.process_objects(
                    new_object_list=new_object_list,
                    detection_timestamp_ns=detection_timestamp_ns,
                    sensor_str=sensor_str,
                )

            check_estimated_dict(
                est_dict=self.estimated_object_dict,
                t_track=self.tracking_timestamp_ns,
                logger=self.logger,
            )

        # Update tracking id
        self.update_tracking_id()

        # Store all active objects
        self.old_object_dict.update(self.estimated_object_dict)

        check_observation_storage(
            obsv_storage=self.observation_storage,
            old_obj_dict=self.old_object_dict,
            logger=self.logger,
        )

        # Check for collisions
        if module_state == 30:
            module_state = self.check_collisions(
                ego_state=self.ego_state,
                object_dict=self.estimated_object_dict,
                logger=self.logger,
            )

    def get_delayed_object_states(
        self,
        detection_timestamp_ns: int,
        detection_timestamp_next_ns: int,
        sensor_str: str,
    ):
        """Check if detection is delayed and update prior object states."""
        # Get historic data if detection is outdated
        dt_ns = self.tracking_timestamp_ns - detection_timestamp_ns

        if dt_ns > self.filter_timestep_ns / 2.0 and self.observation_storage:
            # Logging: Debug
            self.logger.info(
                "DELAY: sensor = {}, t_track = "
                "{:d} ns, t_ego = {:d} ns, t_detect = {:d} ns".format(
                    sensor_str,
                    self.tracking_timestamp_ns,
                    self.ego_state["t"],
                    detection_timestamp_ns,
                )
            )
            self.logger.info(
                "DELAY: sensor = {}, t_track - t_now = {:.02f} ms, "
                "t_track - t_ego = {:.02f} ms, "
                "t_track - t_detect = {:.02f} ms, "
                "dt_max = {:.02f} ms".format(
                    sensor_str,
                    (
                        self.tracking_timestamp_ns
                        - stamp2time(*self.get_clock().now().seconds_nanoseconds())
                    )
                    / 1e6,
                    (self.tracking_timestamp_ns - self.ego_state["t"]) / 1e6,
                    dt_ns / 1e6,
                    self.filter_timestep_ns / 2e6,
                )
            )

            # Get closest historic ego value
            index_min_ego = np.abs(
                np.array(self.observation_storage["ego"]["t"]) - detection_timestamp_ns
            ).argmin()
            ego_state_hist = np.array(
                self.observation_storage["ego"]["state"][index_min_ego]
            )
            ego_time_hist = self.observation_storage["ego"]["t"][index_min_ego]

            for obj_id, obj in self.observation_storage.items():
                if obj_id == "ego":
                    continue

                # Skip object if oldest timestamp is newer than detection timestamp
                if (
                    obj["t"][-1] - detection_timestamp_ns
                    > self.filter_timestep_ns / 2.0
                ):
                    try:
                        self.skipped_object_dict[
                            obj_id
                        ] = self.estimated_object_dict.pop(obj_id)
                        self.logger.info(
                            "DELAY COMPENSATION: pop for skipping, obj_id = {}".format(
                                obj_id
                            )
                        )
                    except Exception as exc:
                        self.logger.error(
                            "DELAY COMPENSATION: invalid skipping, error: {}".format(
                                exc
                            )
                        )
                    continue

                # Get closest historic value
                index_min = np.abs(np.array(obj["t"]) - detection_timestamp_ns).argmin()

                # Store historic states
                self.estimated_object_dict[obj_id] = {
                    "state": obj["state"][index_min],
                    "t": obj["t"][index_min],
                }

                # Grab x_hist only for the oldest detection input
                if not self.obj_filters[obj_id].bool_oldest_detect:
                    self.obj_filters[obj_id].bool_oldest_detect = True
                    self.obj_filters[obj_id].x_hist = np.array(obj["state"][index_min])
                    self.obj_filters[obj_id].P_hist = np.array(obj["P"][index_min])
                    self.obj_filters[obj_id].egopos_hist = ego_state_hist[:2]

                self.obj_filters[obj_id].idx_hist = index_min

                if detection_timestamp_next_ns:
                    index_min_next = np.abs(
                        np.array(obj["t"]) - detection_timestamp_next_ns
                    ).argmin()
                    self.obj_filters[obj_id].num_updates = index_min - index_min_next
                else:
                    # Add additional step to update to current time stamp
                    self.obj_filters[obj_id].num_updates = index_min + 1

            return ego_time_hist, np.array(ego_state_hist)

        return self.ego_state["t"], np.array(self.ego_state["state"])

    def process_objects(
        self, new_object_list: list, detection_timestamp_ns: int, sensor_str: str
    ):
        """Execute object matching and update object states."""
        if self.old_object_dict:
            # Match
            matched_objects, unmatched_new_objects = self.matching(
                est_object_dict=self.estimated_object_dict,
                new_object_list=new_object_list,
                sensor_str=sensor_str,
            )

            unmatched_old_objects = self.matching.check_old_objects(
                obs_storage=self.observation_storage,
                old_obj_dict=self.old_object_dict,
                obj_filters=self.obj_filters,
                est_obj_dict=self.estimated_object_dict,
                skipped_obj_dict=self.skipped_object_dict,
                ego_pos=self.ego_state["state"][:2],
                sensor_str=sensor_str,
                time_ns=self.tracking_timestamp_ns,
            )

            # Matched old objects
            for obj_tuple in matched_objects.items():
                self.update(
                    obj_tuple,
                    detection_timestamp_ns,
                    sensor_str,
                    meas_old_tuple=(True, True),
                )

            # Unmatched old objects
            for obj_tuple in unmatched_old_objects.items():
                self.update(
                    obj_tuple,
                    detection_timestamp_ns,
                    sensor_str,
                    meas_old_tuple=(False, True),
                )

        else:
            # Initialize new objects, as no old objects are stored
            unmatched_new_objects = self.matching.init_blank(
                new_object_list=new_object_list,
                sensor_str=sensor_str,
            )

        # Unmatched new objects
        for obj_tuple in unmatched_new_objects.items():
            self.update(
                obj_tuple,
                detection_timestamp_ns,
                sensor_str,
                meas_old_tuple=(True, False),
            )

        # compensate delayed detections
        self.compensate()

        # add skipped object back to estimated objects
        self.estimated_object_dict.update(self.skipped_object_dict)

        # update old object dict with new estimations
        self.old_object_dict.update(self.estimated_object_dict)

    def update(
        self,
        obj_tuple: tuple,
        detection_timestamp_ns: int,
        sensor_type: str,
        meas_old_tuple: tuple,
    ):
        """Execute measurement update after successful object matching."""
        obj_id, object_el = obj_tuple
        is_measured, is_old = meas_old_tuple
        if is_old:
            if is_measured:
                self.observation_storage[obj_id]["num_measured"] += 1
                self.observation_storage[obj_id][
                    "last_time_seen_ns"
                ] = detection_timestamp_ns

                # Get measurement variables and associated indices
                z_meas = np.array(object_el["state"])

                self.obj_filters[obj_id].update(
                    z_meas=z_meas,
                    sensor_type=sensor_type,
                    ego_pos=self.ego_state["state"][:2],
                    measure_keys=object_el["keys"],
                )
            else:
                self.obj_filters[obj_id].reset()

            # Logging
            self.filter_log[obj_id] = log_filter_vals(
                obj_filter=self.obj_filters[obj_id],
                obj_filter_log=self.filter_log.get(obj_id, []),
                time_stamp=self.tracking_timestamp_ns,
                n_est=self.num_est,
                is_measured=is_measured,
                sensor=sensor_type,
            )
        else:
            # Initialize state of new object
            self.estimated_object_dict[obj_id] = self.init_filter(
                object_el=object_el,
                obj_id=obj_id,
                detection_timestamp_ns=detection_timestamp_ns,
                sensor_type=sensor_type,
            )

    def init_filter(
        self,
        object_el: dict,
        obj_id: int,
        detection_timestamp_ns: int,
        sensor_type: str,
    ):
        """Initialize filter for new object."""
        object_el["state"] = get_initialized_state(
            state_vars=self.state_vars,
            meas_obj=object_el,
        )

        # Add yaw based on track position, if not measured
        if "yaw" in object_el["keys"]:
            unkown_curvature = True
        else:
            unkown_curvature = False
            curvature = self.track_boundary.add_yaw(
                new_object=object_el,
            )
            self.logger.info(
                "FILTER INIT: add yaw-estimate = {:.02f} rad, obj_id = {}".format(
                    object_el["state"][2], obj_id
                )
            )

        if self.__add_ego_variables:
            # Add speed from ego-vehicle
            if "v" not in object_el["keys"]:
                if self.ego_state["in_pit"]:
                    object_el["state"][3] = self.ego_state["state"][3] * 0.0
                else:
                    object_el["state"][3] = (
                        self.ego_state["state"][3] * self.__ego_speed_factor
                    )
                self.logger.info(
                    "FILTER INIT: add speed est = {:.02f} m/s, obj_id = {}".format(
                        object_el["state"][3], obj_id
                    )
                )

            # Add yaw rate
            if "yawrate" not in object_el["keys"]:
                if unkown_curvature:
                    curvature = self.track_boundary.get_curvature(
                        new_object=object_el,
                    )

                object_el["state"][4] = object_el["state"][3] * curvature
                self.logger.info(
                    "FILTER INIT: add yawrate est = "
                    "{:.02f} rad/s, curvature = {:.03f} 1/m, obj_id = {}".format(
                        object_el["state"][4], curvature, obj_id
                    )
                )

            # Add acceleration from ego-vehicle
            if "a" not in object_el["keys"] and self.ego_state["state"][3] < 10.0:
                object_el["state"][5] = self.ego_state["state"][5]
                self.logger.info(
                    "FILTER INIT: add acceleration estimate = {:.02f} m/s2, obj_id = {}".format(
                        object_el["state"][5], obj_id
                    )
                )

        # Intialize filter for new object
        self.obj_filters[obj_id] = EKF(
            x_init=object_el["state"],
            params=self.filter_params,
            sensor_init=sensor_type,
            ego_init=self.ego_state,
        )

        # Sync detection to current timestamp
        dt_ns = self.tracking_timestamp_ns - detection_timestamp_ns

        # calculate number of iterations for filter init
        n_estimates = max(int(np.round(dt_ns / self.filter_timestep_ns)), 0)
        t_0 = self.tracking_timestamp_ns - n_estimates * self.filter_timestep_ns

        self.logger.info(
            "FILTER INIT: sensor = {}, state = {}, "
            "t_obj = {} ns, obj_id = {}".format(
                sensor_type,
                self.obj_filters[obj_id].x,
                t_0,
                obj_id,
            )
        )

        x_list = [None] * (n_estimates + 1)
        t_list = [None] * (n_estimates + 1)
        p_list = [None] * (n_estimates + 1)

        x_list[-1] = self.obj_filters[obj_id].x.tolist()
        t_list[-1] = t_0
        p_list[-1] = self.obj_filters[obj_id].P

        # compensate delay for object initialization
        for n_it in range(n_estimates):
            self.obj_filters[obj_id].predict()

            x_list[n_estimates - n_it - 1] = self.obj_filters[obj_id].x.tolist()
            t_list[n_estimates - n_it - 1] = (
                self.tracking_timestamp_ns
                - (n_estimates - n_it - 1) * self.filter_timestep_ns
            )
            p_list[n_estimates - n_it - 1] = self.obj_filters[obj_id].P.tolist()

        # init observation storage
        self.observation_storage[obj_id] = get_new_storage_item(
            deq_length=self.__n_max_obsv,
            x_init=x_list,
            P_init=p_list,
            t_init=t_list,
        )

        # Logging
        self.logger.info(
            "FILTER INIT: compensated delay = {:.02f} ms, num_est = {:d}".format(
                (self.filter_timestep_ns * n_estimates) / 1e6, n_estimates
            )
        )

        # return delay compensated object
        return {
            "state": x_list[0],
            "t": self.tracking_timestamp_ns,
        }

    def store_ego(self):
        """Store ego to deque."""
        if "ego" not in self.observation_storage:
            self.observation_storage["ego"] = get_new_storage_item(
                deq_length=self.__n_max_obsv,
                x_init=[self.ego_state["state"]],
                P_init=[],
                t_init=[self.ego_state["t"]],
            )

            # Logging
            self.logger.info(
                "STORAGE INIT: new object, obj_id = {}, "
                "state = {}, time_stamp = {}".format(
                    "ego", self.ego_state["state"], self.ego_state["t"]
                )
            )
            return

        storage_overwrite = (
            self.ego_state["t"] == self.observation_storage["ego"]["t"][0]
        )

        if self.__bool_ego_lowpass:
            _ = self.object_lowpass(
                dyn_object=self.ego_state,
                obs_storage=self.observation_storage,
                obj_id="ego",
                storage_overwrite=storage_overwrite,
            )
        else:
            self.observation_storage["ego"]["state"].appendleft(self.ego_state["state"])
            self.observation_storage["ego"]["t"].appendleft(self.ego_state["t"])

        self.observation_storage["ego"]["num_measured"] += 1
        self.observation_storage["ego"]["last_time_seen_ns"] = self.ego_state["t"]

    def update_tracking_id(self):
        """Update tracking id of objects."""
        for _, object_el in self.estimated_object_dict.items():
            object_el["tracking_id"] = self.tracking_id
            self.tracking_id += 1

    def log_input_stats(self, sensor_str: str, is_non_empty: bool):
        """Log detection input stats."""
        self.__detection_ctr[sensor_str][0] += 1
        self.__detection_ctr[sensor_str][2] += is_non_empty
        if self.__detection_ctr[sensor_str][0] % 100 == 0:
            for key, val in self.__detection_ctr.items():
                if key == "calls":
                    continue
                self.logger.info(
                    "DETECTION STATS: calls = {}, sensor = "
                    "{}, inputs = {}, raw = {}, filtered = {}".format(
                        self.__detection_ctr["calls"],
                        key,
                        val[0],
                        val[1],
                        val[2],
                    )
                )
