"""Helper function for logging of tracking module."""
import sys
import os
import logging
import json
from shutil import copyfile
from pathlib import Path
import numpy as np
from tqdm import tqdm


def default(obj):
    """Handle numpy arrays when converting to json."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError("Not serializable (type: " + str(type(obj)) + ")")


def get_msg_logger(path_dict: dict):
    """Create path."""
    if not os.path.exists(path_dict["abs_log_path"]):
        os.makedirs(path_dict["abs_log_path"])

    # Setup log file path
    logfile_path = os.path.join(path_dict["abs_log_path"], "msg_logs.log")

    # logger config
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    msg_logger = logging.getLogger("msg_logger")
    msg_logger_format = logging.Formatter(
        "%(levelname)s [%(asctime)s]: %(message)s", "%H:%M:%S"
    )

    # console loggers
    hdlr = logging.StreamHandler(sys.stdout)
    hdlr.setFormatter(msg_logger_format)
    hdlr.setLevel(logging.ERROR)

    # file logger
    fhdlr = logging.FileHandler(logfile_path)
    fhdlr.setFormatter(msg_logger_format)
    fhdlr.setLevel(logging.DEBUG)

    # add loggers to main logger
    msg_logger.addHandler(hdlr)
    msg_logger.addHandler(fhdlr)
    msg_logger.setLevel(logging.DEBUG)

    # Start logging
    msg_logger.info("{:*^80}".format(" Message Logging "))

    # Copy version.txt file to log folder
    version_path = os.path.join(path_dict["src_path"], "version.txt")
    if os.path.exists(version_path):
        logdir = os.path.join(path_dict["abs_log_path"], "version.txt")
        copyfile(version_path, logdir)
    else:
        msg_logger.warning("No version.txt-file found.")

    return msg_logger


class DataLogging:
    """Logging class that handles the setup and data-flow."""

    def __init__(
        self, path_dict: dict, header_only: bool = False, latency_log: bool = False
    ) -> None:
        """Initialize class."""
        if latency_log:
            self.header = (
                "id;latency_id_lidar;t_receive_lidar;num_objects_lidar;num_filtered_lidar;latency_id_cluster;"
                "t_receive_cluster;num_objects_cluster;num_filtered_cluster;latency_id_radar;t_receive_radar;"
                "num_objects_radar;num_filtered_radar;latency_id_camera;t_receive_camera;num_objects_camera;"
                "num_filtered_camera;t_detect;t_track;t_pred;t_send;len_observation_storage;len_prediction_dict"
            )
            file_name = "latency_logs.csv"
            self.latency_sensors = ["lidar", "lidar_cluster", "radar", "camera"]
        else:
            self.header = (
                "time_ns;"
                "detection_input;"
                "ego_state;"
                "tracking_input;"
                "match_dict;"
                "filter_log;"
                "object_dict;"
                "pred_dict;"
                "cycle_time_ns"
            )
            file_name = "data_logs.csv"
        if header_only:
            return

        # Create directories
        if not os.path.exists(path_dict["abs_log_path"]):
            os.makedirs(path_dict["abs_log_path"])
        self.__log_path = os.path.join(path_dict["abs_log_path"], file_name)
        Path(os.path.dirname(self.__log_path)).mkdir(parents=True, exist_ok=True)

        # write header to logging file
        with open(self.__log_path, "w+") as fh:
            fh.write(self.header)

    # ----------------------------------------------------------------------------------------------------------
    # CLASS METHODS --------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------

    def get_headers(self):
        """Get headers."""
        return self.header

    def log_data(
        self,
        time_ns: int,
        detection_input: dict,
        ego_state: dict,
        tracking_input: dict,
        match_dict: dict,
        filter_log: dict,
        object_dict: dict,
        pred_dict: dict,
        cycle_time_ns: int,
    ) -> None:
        """Log tracking data.

        Args:
            time_ns (int): timestamp in ns
            detection_input (dict): Dict of all received detection inputs (raw)
            ego_state (dict): Ego state including time stamp and kinematic state
            tracking_input (dict): Dict of pre-processed detection inputs which are input to tracking
            match_dict (dict): Dict of matching old and new objects per sensor (key entry in dict)
            filter_log (dict): Logs of EKF steps
            object_dict (dict): Dict of updated object list (matched and updated by EKF)
            pred_dict (dict): Dict of predictions, keys are the obj_ids, values are the predicted trajectories
            cycle_time_ns (int): Cycle time of current call in ns
        """
        with open(self.__log_path, "a") as fh:
            fh.write(
                "\n"
                + str(time_ns)
                + ";"
                + json.dumps(detection_input, default=default)
                + ";"
                + json.dumps(ego_state, default=default)
                + ";"
                + json.dumps(tracking_input, default=default)
                + ";"
                + json.dumps(match_dict, default=default)
                + ";"
                + json.dumps(filter_log, default=default)
                + ";"
                + json.dumps(object_dict, default=default)
                + ";"
                + json.dumps(pred_dict, default=default)
                + ";"
                + json.dumps(cycle_time_ns, default=default)
            )

    def log_latency(
        self,
        latency_dict: dict,
        detection_subscriber_dict: dict,
        len_obsv_storage: int,
        pred_dict: dict,
    ) -> None:
        """Log latency time stamps and IDs.

        Args:
            pred_dict (dict): Dict of predictions, keys are the obj_ids, values are the predicted trajectories
            len_obsv_storage (int): Length of observation storage, i.e. number of tracked objects
            detection_subscriber_dict (dict): Dict of subscribed detections
            latency_dict (dict): Dict of time stamp triggered during callback.
        """
        # prepare latency logging
        if pred_dict is None:
            len_pred_dict = -1
        else:
            len_pred_dict = len(pred_dict)

        latency_logs = [[-1] * 4] * len(self.latency_sensors)
        for j, latency_sensor in enumerate(self.latency_sensors):
            for (
                active_sensor,
                detection_subscriber,
            ) in detection_subscriber_dict.items():
                if latency_sensor == active_sensor.strip("_"):
                    latency_logs[j] = detection_subscriber.latency_log_list
                    continue

        _ = [
            detection_subscriber.reset_latency_log_list()
            for detection_subscriber in detection_subscriber_dict.values()
        ]

        # log latency to csv
        self.log_latency_data(
            latency_id=latency_dict["id"],
            latency_log_list_lidar=latency_logs[0],
            latency_log_list_cluster=latency_logs[1],
            latency_log_list_radar=latency_logs[2],
            latency_log_list_camera=latency_logs[3],
            t_detect=latency_dict["t_detect"],
            t_track=latency_dict["t_track"],
            t_pred=latency_dict["t_pred"],
            t_send=latency_dict["t_send"],
            len_obsv_storage=len_obsv_storage,
            len_pred_dict=len_pred_dict,
        )

        # reset latency variables
        latency_dict["id"] += 1
        latency_dict["t_detect"] = -1
        latency_dict["t_track"] = -1
        latency_dict["t_pred"] = -1
        latency_dict["t_send"] = -1

    def log_latency_data(
        self,
        latency_id: int,
        latency_log_list_lidar: list,
        latency_log_list_cluster: list,
        latency_log_list_radar: list,
        latency_log_list_camera: list,
        t_detect: int,
        t_track: int,
        t_pred: int,
        t_send: int,
        len_obsv_storage: int,
        len_pred_dict: int,
    ) -> None:
        """
        Write one line to the log file.

            latency_id_lidar (int): running counter int(str(<sensor_id>) + str(<running_int>)),
                                    otherwise -1 (sensor disabled)
            t_receive_lidar (int): ros2-timestamp in ns of receiving the message,
                                   otherwise -1 (sensor disabled)
            num_objects_lidar (int): number of detected objects of the sensor,
                                     otherwise -1 (sensor disabled)
            num_filtered_lidar (int): number of objects, which are filtered out of track, of the sensor,
                                      otherwise -1 (sensor disabled)

            latency_id_cluster (int): running counter int(str(<sensor_id>) + str(<running_int>)),
                                      otherwise -1 (sensor disabled)
            t_receive_cluster (int): ros2-timestamp in ns of receiving the message,
                                     otherwise -1 (sensor disabled)
            num_objects_cluster (int): number of detected objects of the sensor,
                                       otherwise -1 (sensor disabled)
            num_filtered_cluster (int): number of objects, which are filtered out of track, of the sensor,
                                        otherwise -1 (sensor disabled)

            latency_id_radar (int): running counter int(str(<sensor_id>) + str(<running_int>)),
                                    otherwise -1 (sensor disabled)
            t_receive_radar (int): ros2-timestamp in ns of receiving the message,
                                   otherwise -1 (sensor disabled)
            num_objects_radar (int): number of detected objects of the sensor,
                                     otherwise -1 (sensor disabled)
            num_filtered_radar (int): number of objects, which are filtered out of track, of the sensor,
                                      otherwise -1 (sensor disabled)

            latency_id_camera (int): running counter int(str(<sensor_id>) + str(<running_int>)),
                                     otherwise -1 (sensor disabled)
            t_receive_camera (int): ros2-timestamp in ns of receiving the message,
                                    otherwise -1 (sensor disabled)
            num_objects_camera (int): number of detected objects of the sensor,
                                      otherwise -1 (sensor disabled)
            num_filtered_camera (int): number of objects, which are filtered out of track, of the sensor,
                                       otherwise -1 (sensor disabled)

            t_detect (int): ros2-timestamp in ns after all detection are received and before tracking step starts
            t_track (int): ros2-timestamp in ns after tracking timestep suceeded,
                           otherwise -1 (no object input = no tracking timstep)
            t_pred (int): ros2-timestamp in ns after prediction (physics, data and collision check rules) suceeded
            t_send (int): ros2-timestamp in ns after publishing predicted objects
                          (note: watchdog to mod_control is send aftwards!!)
            len_observation_storage (int): number of active objects
            len_prediction_dict (int): number of predicted objects

        """
        (
            latency_id_lidar,
            t_receive_lidar,
            num_objects_lidar,
            num_filtered_lidar,
        ) = latency_log_list_lidar
        (
            latency_id_cluster,
            t_receive_cluster,
            num_objects_cluster,
            num_filtered_cluster,
        ) = latency_log_list_cluster
        (
            latency_id_radar,
            t_receive_radar,
            num_objects_radar,
            num_filtered_radar,
        ) = latency_log_list_radar
        (
            latency_id_camera,
            t_receive_camera,
            num_objects_camera,
            num_filtered_camera,
        ) = latency_log_list_camera

        with open(self.__log_path, "a") as fh:
            fh.write(
                "\n"
                + json.dumps(latency_id)
                + ";"
                + json.dumps(latency_id_lidar)
                + ";"
                + json.dumps(t_receive_lidar)
                + ";"
                + json.dumps(num_objects_lidar)
                + ";"
                + json.dumps(num_filtered_lidar)
                + ";"
                + json.dumps(latency_id_cluster)
                + ";"
                + json.dumps(t_receive_cluster)
                + ";"
                + json.dumps(num_objects_cluster)
                + ";"
                + json.dumps(num_filtered_cluster)
                + ";"
                + json.dumps(latency_id_radar)
                + ";"
                + json.dumps(t_receive_radar)
                + ";"
                + json.dumps(num_objects_radar)
                + ";"
                + json.dumps(num_filtered_radar)
                + ";"
                + json.dumps(latency_id_camera)
                + ";"
                + json.dumps(t_receive_camera)
                + ";"
                + json.dumps(num_objects_camera)
                + ";"
                + json.dumps(num_filtered_camera)
                + ";"
                + json.dumps(t_detect)
                + ";"
                + json.dumps(t_track)
                + ";"
                + json.dumps(t_pred)
                + ";"
                + json.dumps(t_send)
                + ";"
                + json.dumps(len_obsv_storage)
                + ";"
                + json.dumps(len_pred_dict)
            )

    def output_latency_data(
        self,
        msg_logger,
    ) -> None:
        """Output latency data in case of time violation."""
        with open(self.__log_path) as fh:
            for k, line in enumerate(reversed(list(fh))):
                try:
                    (
                        latency_id,
                        _,
                        t_receive_lidar,
                        num_objects_lidar,
                        _,
                        _,
                        t_receive_lidar_cluster,
                        num_objects_lidar_cluster,
                        _,
                        _,
                        t_receive_radar,
                        num_objects_radar,
                        _,
                        _,
                        t_receive_camera,
                        num_objects_camera,
                        _,
                        t_detect,
                        t_track,
                        t_pred,
                        t_send,
                        len_obsv_storage,
                        len_pred_dict,
                    ) = tuple(json.loads(ll) for ll in line.split(";"))

                    invalid_detection = (
                        max(
                            num_objects_lidar,
                            num_objects_lidar_cluster,
                            num_objects_radar,
                            num_objects_camera,
                        )
                        == -1
                    )
                    if invalid_detection:
                        t_init_string = "TIME REPORT: WITHOUT DETECTION"
                        t_sub_string = ""
                        t_tot_string = ", dt_tot = {:.0f} ms".format(
                            (t_send - t_detect) / 1e6
                        )
                    else:
                        t_min_receive = min(
                            [
                                t_recv
                                for t_recv in [
                                    t_receive_lidar,
                                    t_receive_lidar_cluster,
                                    t_receive_radar,
                                    t_receive_camera,
                                ]
                                if t_recv > 0
                            ]
                        )
                        list_detected_obj = [
                            num_obj
                            for num_obj in [
                                num_objects_lidar,
                                num_objects_lidar_cluster,
                                num_objects_radar,
                                num_objects_camera,
                            ]
                            if num_obj > -1
                        ]

                        t_init_string = "TIME REPORT: WITH DETECTION, n_sens = {}, n_obj = {}".format(
                            len(list_detected_obj), sum(list_detected_obj)
                        )
                        t_sub_string = ", dt_subscribe = {:.0f} ms".format(
                            (t_detect - t_min_receive) / 1e6
                        )
                        t_tot_string = ", dt_tot = {:.0f} ms".format(
                            (t_send - t_min_receive) / 1e6
                        )

                    msg_logger(
                        t_init_string
                        + ", ID = {}".format(latency_id)
                        + t_sub_string
                        + ", dt_track = {:.0f} ms".format((t_track - t_detect) / 1e6)
                        + ", dt_pred = {:.0f} ms".format((t_pred - t_track) / 1e6)
                        + t_tot_string
                        + ", num_tracks = {}, num_preds = {}".format(
                            len_obsv_storage, len_pred_dict
                        )
                    )
                except Exception as exc:
                    msg_logger("error: {}".format(exc))
                    break

                if k == 2:
                    break


def read_all_data(file_path_in, keys=None, zip_horz=False):
    """Read all date of the log file and return zipped."""
    with open(file_path_in) as f:
        total_lines = sum(1 for _ in f)

    total_lines = max(1, total_lines)

    all_data = None

    assert (
        total_lines > 1
    ), "Invalid logs: No tracking files, most likely short simulation time"

    # extract a certain line number (based on time_stamp)

    with open(file_path_in) as file:
        # get to top of file (1st line)
        file.seek(0)
        # get header (":-1" in order to remove tailing newline character)
        header = file.readline()[:-1]
        # extract line
        line = ""
        for j in tqdm(range(total_lines - 1)):
            line = file.readline()

            if zip_horz:
                if all_data is None:
                    all_data = []
                    all_data = [header.split(";"), [None] * (total_lines - 1)]

                all_data[1][j] = tuple(json.loads(ll) for ll in line.split(";"))
            else:
                # parse the data objects we want to retrieve from that line
                data = dict(zip(header.split(";"), line.split(";")))
                if all_data is None:
                    if keys is None:
                        keys = data.keys()
                    all_data = {key: [0.0] * (total_lines - 1) for key in keys}
                for key in keys:
                    all_data[key][j] = json.loads(data[key])

    return all_data


def read_info_data(info_file_path):
    """Extract the infos about the observations.

    Associate the infos with each ID and stores them in
    a dictionary, which can be queried by the IDs.
    """
    info_dict = {}
    with open(info_file_path) as f:
        for line in f:
            if "Prediction-ID" in line:
                i0 = line.index("Prediction-ID") + len("Prediction-ID") + 1
                i1 = line.index(":", i0)
                ID = line[i0:i1]
                info_dict[ID] = []

                if "static" in line:
                    info_dict[ID].append("static")

                elif "physics-prediction" in line:
                    info_dict[ID].append("physics-based")

                    i0 = line.index("reason:") + len("reason:") + 1
                    i1 = line.index(",", i0)
                    info_dict[ID].append(line[i0:i1])

                elif "data-prediction" in line:
                    info_dict[ID].append("data-based")

                    if "mixers" in line:
                        i0 = line.index("mixers")
                        info_dict[ID].append("\n")
                        info_dict[ID].append(line[i0:])

                elif "data-physics-override-prediction" in line:
                    info_dict[ID].append("data-physics-override")

                    if "mixers" in line:
                        i0 = line.index("mixers")
                        info_dict[ID].append("\n")
                        info_dict[ID].append(line[i0:])

                elif "rail-prediction" in line:
                    info_dict[ID].append("rail-based")

                    i0 = line.index("reason:") + len("reason:") + 1
                    i1 = line.index(",", i0)
                    info_dict[ID].append(line[i0:i1])

                elif "potential-field" in line:
                    info_dict[ID].append("potential-field")

                    i0 = (
                        line.index("potential-field prediction")
                        + len("potential-field prediction")
                        + 2
                    )
                    i1 = line.index("id:")
                    info_dict[ID].append(line[i0:i1])

                elif "Invalid" in line:
                    info_dict[ID].append("invalid")

            elif "Collision" in line:
                i0 = line.index("IDs") + len("IDs") + 1
                i1 = line.index("(", i0) - 1
                ID1 = line[i0:i1]

                i0 = line.index("and") + len("and") + 1
                i1 = line.index("(", i0) - 1
                ID2 = line[i0:i1]

                i0 = line.index("timestep") + len("timestep") + 1
                ts = line[i0:-2]

                info_dict[ID1].append("collision with ID " + ID2 + " at " + ts)
                info_dict[ID2].append("collision with ID " + ID1 + " at " + ts)

            elif "not adjusted" in line:
                pass

            elif "adjusted" in line:
                if "ID" in line:
                    i0 = line.index("ID") + len("ID") + 1
                    i1 = line.index("adjusted", i0) - 1
                    ID = line[i0:i1]
                    if "right" in line:
                        direction = "right"
                    else:
                        direction = "left"

                    i0 = line.index("distance of") + len("distance of") + 1
                    dist = line[i0:-2]

                    info_dict[ID].append("adjusted to the " + direction + " by " + dist)

    return info_dict


def get_number_of_lines(file_path_in: str):
    """Get number of lines in file."""
    with open(file_path_in) as file:
        row_count = sum(1 for row in file)

    return row_count
