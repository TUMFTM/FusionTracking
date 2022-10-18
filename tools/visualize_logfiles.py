"""Show Tracking Logs."""
import os
import sys
import argparse
import pickle
from ast import literal_eval
import copy
from statistics import mean, stdev
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from tqdm import tqdm

REPO_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
PKG_PATH = os.path.join(
    REPO_PATH,
    "tracking",
)
NODE_PATH = os.path.join(
    PKG_PATH,
    "tracking",
)
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(REPO_PATH)
sys.path.append(PKG_PATH)
sys.path.append(NODE_PATH)

from src.transform_objects import TransformObjects
from src.track_boundary import TrackBoundary
from utils.tracking_utils import get_H_mat, get_R_mat
from utils.setup_helpers import create_path_dict
from utils.map_utils import get_track_paths, get_map_data
from utils.logging_helper import read_all_data, DataLogging
from utils.geometry import rotate_loc_glob, rotate_glob_loc

from visualiziation_helper import confidence_ellipse

PATH_DICT = create_path_dict()


class ViszTracking:
    """Visualize Tracking Logs."""

    def __init__(self, args):
        """Init all neccessary params."""
        self.axis_range = 10
        self.n_std = 3
        self.textprops = dict(boxstyle="round", facecolor="silver")
        self.append_list = ["Prior", "Post", "cov", "delta", "p", "t_rel", "dt"]
        self.obj_center_points_frame = None
        self.markers = []
        self.labels = []
        if "markersize" not in self.__dict__:
            self.markersize = 15
            self.linewidth = 4

        if "args" not in self.__dict__:
            self.args = args

        if not self.load_extracted_data():
            log_file, main_log_file = self.__get_log_data()
            self.__read_logs(main_log_file)

            print("reading data from logs ..")
            _, self.log_rows = read_all_data(log_file, zip_horz=True)
            self.split_data_params()
            self.split_data_obj_id()

        # get track bounds and pit bounds
        self.get_track_data()

        self.log_start_time = None
        self.dt_hist = int(3 * 1e9)  # ns
        self.dt_fut = int(5 * 1e9)  # ns
        self.ns_to_s = int(1e9)  # ns
        self.t_last_time_receiv = {}
        self.filtered_ids = ["ego"]
        self.textstr = ""

        # colors
        self.ego_col = "red"
        self.legend_color = "black"
        self.raw_data_color = "c"
        self.color_list = self.set_colors()
        self.track_color = {"track_bounds": "dimgray", "pit_bounds": "dimgray"}
        self.button_color_dict = {"On": "limegreen", "Off": "gainsboro"}
        self._sensor_marker = dict(
            zip(
                self.params["SENSOR_SETUP"]["active_sensors"],
                ["d", "*", "p", "s", "8", "*"],
            )
        )
        self.ylabel_list = [
            "x in m",
            "y in m",
            "yaw in rad",
            "v in m/s",
            "yawrate in rad/s",
            "a in m/s^2",
        ]

        self.rectangle = np.stack(
            [
                [
                    self.params["MISCS"]["l_vehicle_m"] / 2.0,
                    -self.params["MISCS"]["w_vehicle_m"] / 2.0,
                ],
                [
                    self.params["MISCS"]["l_vehicle_m"] / 2.0,
                    self.params["MISCS"]["w_vehicle_m"] / 2.0,
                ],
                [
                    -self.params["MISCS"]["l_vehicle_m"] / 2.0,
                    self.params["MISCS"]["w_vehicle_m"] / 2.0,
                ],
                [
                    -self.params["MISCS"]["l_vehicle_m"] / 2.0,
                    -self.params["MISCS"]["w_vehicle_m"] / 2.0,
                ],
            ],
            axis=1,
        )

        self.measure_variables = ["x", "y"]
        self.num_sensors_all = []
        for kk in self.log_params["detection_input"]:
            if kk is None:
                continue
            for key in kk:
                if key not in self.num_sensors_all:
                    self.num_sensors_all.append(key)
            if len(self.num_sensors_all) == len(self.params["ACTIVE_SENSORS"]):
                break
        self.n_sensors = {}

        if self.args.save_file_name:
            self.save_extracted_data()

        if self.args.states or self.args.filter or self.args.mis_num_obj:
            return

        # from inheritance
        if (
            not self.args.is_main
            and not self.args.show_filtered
            and not self.args.show_sample
        ):
            return

        self.__init_axes(hide_annots=self.args.show_sample)
        self.__set_ax_scale()

    def save_extracted_data(self):
        """Save extracted log data to .pkl."""
        if not os.path.exists(os.path.dirname(self.abs_save_path)):
            os.makedirs(os.path.dirname(self.abs_save_path))

        if os.path.exists(self.abs_save_path):
            print("Data not saved, already exists")
            return

        with open(self.abs_save_path, "wb") as f:
            pickle.dump(self.__dict__, f)

        print("Saved data to {}".format(self.abs_save_path))

    def load_extracted_data(self):
        """Load extracted data.

        Data is processed before and stored as .pkl.
        """
        pref = str(self.args.save_file_name)
        if self.args.timestamp:
            pref += "_" + self.args.timestamp

        self.abs_save_path = os.path.join(
            REPO_PATH, "evaluation", "processed_logs", pref + ".pkl"
        )
        # overwrite only these params, others are defined in __init__
        overwrite_keys = [
            "params",
            "transform_objects",
            "track_boundary",
            "log_rows",
            "log_params",
            "header_str",
            "obj_ids",
            "count_obj_ids",
            "obj_items",
        ]
        if os.path.exists(self.abs_save_path):
            with open(self.abs_save_path, "rb") as f:
                self.__dict__.update(
                    {
                        key: val
                        for key, val in pickle.load(f).items()
                        if key in overwrite_keys
                    }
                )

            print("Loaded data from {}".format(self.abs_save_path))
            return True

        return False

    def __get_log_data(
        self,
        tracking_log_name: str = "data_logs.csv",
        main_log_name: str = "msg_logs.log",
    ):
        # get all logs in logs-directory
        log_path = os.path.join(NODE_PATH, "logs")
        all_logs = []
        for day_dirs in os.listdir(log_path):
            if day_dirs[0] != "2":
                continue
            for time_dirs in os.listdir(os.path.join(log_path, day_dirs)):
                all_logs.append((day_dirs, time_dirs))

        # choose log-data
        if self.args.timestamp is None:
            log_file_list = ["-".join(log) for log in all_logs]
            log_file_list.sort()
            if len(log_file_list) == 1:
                num_file = 0
            elif len(log_file_list) == 0:
                print("No logs available")
                sys.exit(0)
            else:
                _ = [
                    print("{:d}: {:s}".format(j, log_file))
                    for j, log_file in enumerate(log_file_list)
                ]
                try:
                    num_file = int(input("Enter a number: ".format()))
                except Exception:
                    num_file = len(log_file_list) - 1
            try:
                day_stamp, time_stamp = [tuple(ll.split("-")) for ll in log_file_list][
                    num_file
                ]
            except Exception:
                day_stamp, time_stamp = [tuple(ll.split("-")) for ll in log_file_list][
                    -1
                ]
                print("Invalid input, running with " + log_file_list[-1])

            self.args.timestamp = day_stamp + "-" + time_stamp
        else:
            # derive path from logging timestamp, e.g. SIL_stamp = "2021_02_01-21_46_23"
            try:
                day_stamp, time_stamp = self.args.timestamp.split("-")
            except Exception:
                day_stamp, time_stamp = all_logs[-1]
                print("Invalid input, running latest logs")

        # store selection
        log_file = os.path.join(log_path, day_stamp, time_stamp, tracking_log_name)
        main_log_file = os.path.join(log_path, day_stamp, time_stamp, main_log_name)

        return log_file, main_log_file

    def __read_logs(self, main_log_file):
        """Read selected log-file."""
        with open(main_log_file) as _f:
            for line in _f:
                if "SIL - params: " in line:
                    _a = line.split("SIL - params: ")
                    self.params = literal_eval(_a[1])
                    break

        self.msg_logger = None

        _ = get_map_data(
            main_class=self,
            path_dict=PATH_DICT,
        )

        self.transform_objects = TransformObjects(
            rear_ax_geoc_m=self.params["MISCS"]["rear_ax_geoc_m"]
        )

        self.track_boundary = TrackBoundary(
            all_params=self.params,
        )

    def split_data_params(self):
        """Split data into smaller variables."""
        print("processing data per object ..")

        self.header_str = (
            DataLogging(path_dict=PATH_DICT, header_only=True).get_headers().split(";")
        )
        self.log_params = dict(zip(self.header_str, zip(*self.log_rows)))

        for key, val in self.log_params.items():
            self.log_params[key] = list(val)

    def split_data_obj_id(self):
        """Split logged data by object IDs."""
        all_ids = sum(
            [list(obj_d) for obj_d in self.log_params["object_dict"] if obj_d], []
        )
        self.obj_ids = set(all_ids)

        self.count_obj_ids = list(
            sorted(
                [[x, all_ids.count(x)] for x in set(self.obj_ids)],
                key=lambda item: -int(item[1]),
            )
        )
        self.args.n_obs = min([len(self.count_obj_ids), self.args.n_obs])
        self.count_obj_ids = self.count_obj_ids[: self.args.n_obs]

        # split down to id-level
        self.obj_ids.add("ego")
        self.obj_items = {}
        for obj_id in tqdm(self.obj_ids):
            self.obj_items[obj_id] = {}

            if obj_id == "ego":
                # get object list of element by n-th logging (replaces observation storage)
                self.obj_items[obj_id]["object_states"] = {
                    idx: objs["state"]
                    for idx, objs in enumerate(self.log_params["ego_state"])
                    if bool(objs) and "state" in objs
                }
                # get object list of element by n-th logging (replaces observation storage)
                self.obj_items[obj_id]["t_dict"] = {
                    idx: self.log_params["ego_state"][idx]["t"]
                    for idx in self.obj_items[obj_id]["object_states"].keys()
                }

            else:
                # get object list of element by n-th logging (replaces observation storage)
                self.obj_items[obj_id]["object_states"] = {
                    idx: val["state"]
                    for idx, obj_dict in enumerate(self.log_params["object_dict"])
                    if obj_dict
                    for key, val in obj_dict.items()
                    if key == obj_id
                }
                # get object list of element by time stamps (replaces observation storage)
                self.obj_items[obj_id]["t_dict"] = {
                    idx: self.log_params["object_dict"][idx][obj_id]["t"]
                    for idx in self.obj_items[obj_id]["object_states"].keys()
                }

                # get filter values of element by n-th logging
                self.obj_items[obj_id]["filter_vals"] = {
                    idx: values
                    for idx, obj_dict in enumerate(self.log_params["filter_log"])
                    if obj_dict
                    for obj_id_j, values in obj_dict.items()
                    if obj_id_j == obj_id
                }

            assert len(self.obj_items[obj_id]["t_dict"]) == len(
                self.obj_items[obj_id]["object_states"]
            )

    def get_track_data(self):
        """Get track data form map."""
        _, _, self.right_bound, self.left_bound = get_track_paths(
            track_path=self.params["track_path"]
        )
        _, _, self.right_bound_pit, self.left_bound_pit = get_track_paths(
            track_path=self.params["pit_path"]
        )

    @staticmethod
    def set_colors():
        """Sort colors by hue, saturation, value and name."""
        sorted_colors = sorted(
            (tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), name)
            for name, color in mcolors.CSS4_COLORS.items()
        )
        names = [name for _, name in sorted_colors]
        delete_index = [
            30,
            34,
            36,
            37,
            40,
            42,
            43,
            44,
            45,
            48,
            49,
            52,
            54,
            58,
            59,
            60,
            61,
            70,
            77,
            82,
            85,
            89,
            90,
            91,
            92,
            96,
            97,
            100,
            101,
            103,
            104,
            106,
            108,
            109,
            110,
            111,
            115,
            116,
            132,
            133,
            143,
            145,
            146,
        ]
        for idx in sorted(delete_index, reverse=True):
            del names[idx]
        del names[0:28]

        unsorted_colors = list(mcolors.CSS4_COLORS)
        color_list = [col for col in unsorted_colors if col in names]

        return color_list

    def __set_visible(self, _bool):
        if _bool:
            switch = "on"
        else:
            switch = "off"

        self.__filter_prior_button.ax.patch.set_visible(_bool)
        self.__filter_prior_button.label.set_visible(_bool)
        self.__filter_prior_button.ax.axis(switch)

        self.__filter_delay_button.ax.patch.set_visible(_bool)
        self.__filter_delay_button.label.set_visible(_bool)
        self.__filter_delay_button.ax.axis(switch)

        self.__filter_post_button.ax.patch.set_visible(_bool)
        self.__filter_post_button.label.set_visible(_bool)
        self.__filter_post_button.ax.axis(switch)

    def _set_label(self, msg, mark, dot=True, markeredge=None):
        if msg in self.labels:
            return
        if dot:
            label_obs = mlines.Line2D(
                [], [], color=self.legend_color, marker=mark, markeredgecolor=markeredge
            )
        else:
            label_obs = mlines.Line2D(
                [],
                [],
                color=self.legend_color,
                linestyle=mark,
                markeredgecolor=markeredge,
            )
        self.markers.append(label_obs)
        self.labels.append(msg)

    def __init_axes(self, hide_annots=False):

        self.annot = None
        self.scatter = None
        self.scatter1 = None

        if hide_annots:
            self.fig, (self._ax) = plt.subplots(
                nrows=1,
                figsize=(16, 9),
            )
            return

        self.fig, (self._ax, ax_dummy, self.ax_slide, ax_dummy2) = plt.subplots(
            nrows=4,
            figsize=(16, 9),
            gridspec_kw=dict(height_ratios=[7, 0.01, 0.2, 0.1]),
        )
        ax_dummy.axis("off")
        ax_dummy2.axis("off")
        self.sfreq = Slider(
            self.ax_slide,
            "Timestep",
            0,
            len(self.log_rows) - 1,
            valinit=0,
            valstep=1,
        )

        # previous trajectory
        self.__bool_prev_traj = False
        axprev = plt.axes([0.02, 0.95, 0.063, 0.04])
        self.__prev_button = Button(axprev, "Prev Traj")
        self.__prev_button.color = self.button_color_dict["Off"]

        # future trajectory
        self.__bool_fut_traj = False
        axfut = plt.axes([0.09, 0.95, 0.063, 0.04])
        self.__fut_button = Button(axfut, "Fut Traj")
        self.__fut_button.color = self.button_color_dict["Off"]

        # Display raw measurement data
        self.__bool_raw_data = False
        ax_check_object_col = plt.axes([0.16, 0.95, 0.063, 0.04])
        self.__raw_data_button = Button(ax_check_object_col, "Detection\ninput")
        self.__raw_data_button.color = self.button_color_dict["Off"]

        # matched Detections
        self.__bool_sensor_detection = False
        axdet = plt.axes([0.23, 0.95, 0.063, 0.04])
        self.__det_button = Button(axdet, "Detection\nmatched")
        self.__det_button.color = self.button_color_dict["Off"]

        # Filter possibilities
        self.__bool_filter = False
        axfilter = plt.axes([0.30, 0.95, 0.063, 0.04])
        self.__filter_button = Button(axfilter, "Filter")
        self.__filter_button.color = self.button_color_dict["Off"]

        # Filter-Prior
        self.__bool_filter_prior = False
        axprior = plt.axes([0.2705, 0.90, 0.04, 0.02])
        self.__filter_prior_button = Button(axprior, "Prior")
        self.__filter_prior_button.color = self.button_color_dict["Off"]

        # Filter-Delay
        self.__bool_filter_delay = False
        axdelay = plt.axes([0.3115, 0.90, 0.04, 0.02])
        self.__filter_delay_button = Button(axdelay, "Delay")
        self.__filter_delay_button.color = self.button_color_dict["Off"]

        # Filter-Posterior
        self.__bool_filter_post = False
        axpost = plt.axes([0.3525, 0.90, 0.04, 0.02])
        self.__filter_post_button = Button(axpost, "Post")
        self.__filter_post_button.color = self.button_color_dict["Off"]

        self.__set_visible(False)

        # Prediction
        self.__bool_predict = False
        axpred = plt.axes([0.37, 0.95, 0.063, 0.04])
        self.__predict_button = Button(axpred, "Prediction")
        self.__predict_button.color = self.button_color_dict["Off"]

        # unmatched Detections
        self.__bool_unmatched = False
        axunmatch = plt.axes([0.44, 0.95, 0.063, 0.04])
        self.__unmatch_button = Button(axunmatch, "Detection\nunmatched")
        self.__unmatch_button.color = self.button_color_dict["Off"]

        # check buttons
        self.labels_check = [
            "Cov: $\\sigma_x$, $\\sigma_y$, $\\rho_{xy}$",
            "$\\Delta x$, $\\Delta y$, $\\Delta \\Psi$, $\\Delta v$",
            "$|P|_2$",
            "t_rel / ms",
            "dt / ms",
        ]
        self.__bool_check_button = [False] * len(self.labels_check)
        axcheck = plt.axes([0.51, 0.90, 0.08, 0.09])
        self.__check_button = CheckButtons(axcheck, self.labels_check)

        # show customized states of objects
        axbox1 = plt.axes([0.68, 0.95, 0.2, 0.04])
        inital_text_states = "[]"
        self.__custom_states = TextBox(
            axbox1,
            "Visualize Obj-States\nEnter object-IDs:",
            initial=inital_text_states,
        )

        # show customized object-filter states
        axbox2 = plt.axes([0.68, 0.9, 0.2, 0.04])
        inital_text_filter = "[]"
        self.__custom_filter = TextBox(
            axbox2,
            "Plot filter propagation.\nEnter object-IDs:",
            initial=inital_text_filter,
        )

        # check button mis_stats
        label_mis_states = ["Mis-Matches &\n Num-Objects:"]
        axcheck_mis_stats = plt.axes([0.9, 0.90, 0.08, 0.04])
        self.__check_mis_stats_button = CheckButtons(
            axcheck_mis_stats, label_mis_states
        )

    def __set_ax_scale(self):
        self._ax.axis("equal")
        self._ax.grid(True)
        self._ax.autoscale(False)

    def __press(self, event):
        if event.key == "right":
            _vv = self.sfreq.val + 1
            _vv = np.clip(_vv, 0, len(self.log_rows) - 1)
            self.sfreq.set_val(_vv)
        elif event.key == "left":
            _vv = self.sfreq.val - 1
            _vv = np.clip(_vv, 0, len(self.log_rows) - 1)
            self.sfreq.set_val(_vv)

    def __hover(self, event):
        if self.annot is None:
            return
        vis = self.annot.get_visible()
        if event.inaxes == self._ax:
            cont, ind = self.scatter.contains(event)
            if cont:
                self.__update_annot(ind)
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.fig.canvas.draw_idle()
            if self.scatter1:
                cont1, ind1 = self.scatter1.contains(event)
                if cont1 and self.transformed_detection_dict is not None:
                    self.__update_annot1(ind1)
                    self.annot.set_visible(True)
                    self.fig.canvas.draw_idle()

    def __update_annot(self, ind):
        pos = self.scatter.get_offsets()[ind["ind"][0]]
        self.annot.xy = pos
        # show information of vehicle
        _id = self.obj_center_points_frame["id"][ind["ind"][0]]
        text = "Obj-ID: " + str(_id)
        text += "\nX-Pos: {:.2f}m".format(
            self.obj_center_points_frame["x"][ind["ind"][0]]
        )
        text += "\nY-Pos: {:.2f}m".format(
            self.obj_center_points_frame["y"][ind["ind"][0]]
        )
        yaw = self.obj_center_points_frame["yaw"][ind["ind"][0]] / np.pi * 180.0
        text += "\nYaw: {:.2f}°".format(yaw)
        text += "\nv: {:.2f}m/s".format(
            self.obj_center_points_frame["v"][ind["ind"][0]]
        )
        text += "\nyawrate: {:.2f}rad/s".format(
            self.obj_center_points_frame["yawrate"][ind["ind"][0]]
        )
        text += "\na: {:.2f}m/s^2".format(
            self.obj_center_points_frame["a"][ind["ind"][0]]
        )

        if self.__bool_filter_prior and _id != "ego":
            text += "\nPrior: {}".format(
                self.obj_center_points_frame["Prior"][ind["ind"][0]]
            )
        if self.__bool_filter_post and _id != "ego":
            post_list = self.obj_center_points_frame["Post"][ind["ind"][0]]
            if (post_list is None) or (post_list == []):
                text += "\nPosterior (Sensor, |P|): {}".format(post_list)
            else:
                text += "\nPosterior (Sensor, |P|): "
                for _, sen_val in enumerate(post_list):
                    text += "\n   {}, {:.3f}".format(sen_val[0], sen_val[1])
        if any(self.__bool_check_button) and _id != "ego":
            for idx, _check in enumerate(self.__bool_check_button):
                if _check:
                    key = self.append_list[idx + 2]
                    txt_check = self.obj_center_points_frame[key][ind["ind"][0]]
                    text += "\n" + self.labels_check[idx] + ":  "
                    if isinstance(txt_check, list):
                        for value in txt_check:
                            text += value + ", "
                    else:
                        text += str(txt_check)

        self.annot.set_text(text)
        # set color
        self.annot.get_bbox_patch().set_facecolor(
            self.obj_center_points_frame["color"][ind["ind"][0]]
        )
        self.annot.get_bbox_patch().set_alpha(0.2)

    def __update_annot1(self, ind):
        pos = self.scatter1.get_offsets()[ind["ind"][0]]
        self.annot.xy = pos
        index = ind["ind"][0]
        text = "Data-ID: " + str(index)

        # show information of measurement
        for sensor_str, (
            detection_list,
            detection_timestamp_ns,
        ) in self.transformed_detection_dict.items():
            if index >= len(detection_list):
                index -= len(detection_list)
                continue
            text += "\nSensor: " + str(sensor_str)
            text += "\nX-Pos: {:.2f}m".format(detection_list[index]["state"][0])
            text += "\nY-Pos: {:.2f}m".format(detection_list[index]["state"][1])
            text += "\nYaw: {:.2f}°".format(detection_list[index]["state"][2])
            if len(detection_list[index]["state"]) > 3:
                text += "\nv: {:.2f}m/s".format(detection_list[index]["state"][3])

            text += "\nTimestamp: {:.3f}s".format(detection_timestamp_ns / 1e9)
            break

        self.annot.set_text(text)
        # set color
        self.annot.get_bbox_patch().set_facecolor(self.raw_data_color)
        self.annot.get_bbox_patch().set_alpha(0.2)

    def __call__(self, frame_value=0, is_iterative=False):
        """Call required functions."""
        if self.args.states:
            self.__plt_states(show_ego=self.args.ego_state, n_obs=5)
            return
        if self.args.filter:
            self.__plt_filter()
            return
        if self.args.mis_num_obj:
            self.__plt_mis_match()
            return
        self.__plt_log(frame_value=frame_value)

        if is_iterative:
            return

        self.sfreq.on_changed(self.__plt_log)
        self.fig.canvas.mpl_connect("key_press_event", self.__press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.__hover)

        self.__prev_button.on_clicked(self.__prev_traj)
        self.__fut_button.on_clicked(self.__fut_traj)
        self.__det_button.on_clicked(self.__det_)
        self.__filter_button.on_clicked(self.__filter_)
        self.__predict_button.on_clicked(self.__predict_)
        self.__unmatch_button.on_clicked(self.__unmatch_)
        self.__check_button.on_clicked(self.__check_)
        self.__custom_states.on_submit(self.__custom_states_plt)
        self.__custom_filter.on_submit(self.__custom_filter_plt)
        self.__check_mis_stats_button.on_clicked(self.__check_mis_stats)
        self.__raw_data_button.on_clicked(self.__raw_data_)
        plt.show()

    def __plt_log(self, frame_value=0):
        """Plot logs in temporal evoluation with Slider."""
        iter_idx = int(frame_value)

        (
            time_stamp_ns,
            detection_input,
            ego_state,
            tracking_input,
            match_dict,
            filter_log,
            object_dict,
            pred_dict,
            cycle_time_ns,
        ) = self.log_rows[iter_idx]

        cylce_time_ms = cycle_time_ns / 1e6
        # reset obj storage
        self.obj_center_points_frame = {
            "id": [],
            "x": [],
            "y": [],
            "yaw": [],
            "v": [],
            "yawrate": [],
            "a": [],
            "color": [],
            "Prior": [],
            "Post": [],
            "cov": [],
            "delta": [],
            "p": [],
            "t_rel": [],
            "dt": [],
        }
        self.markers = []
        self.labels = []

        unmatched_old_dict = {}
        matched_old_dict = {}
        init_object_dict = {}

        # General Information: Text Box
        self.textstr = "t_cycle = {:d} ms".format(
            int(cylce_time_ms)
        ) + "\nt_abs = {:f} s".format(time_stamp_ns / 1e9)

        # Set start time
        if ego_state is not None and "t" in ego_state.keys():
            if self.log_start_time is None:
                self.log_start_time = ego_state["t"]
            else:
                self.textstr += "\nt_rel = {:.02f} s".format(
                    (ego_state["t"] - self.log_start_time) / self.ns_to_s
                )

        # Add ego to iterated objects
        iter_keys = ["ego"] + list(object_dict.keys())
        object_dict["ego"] = copy.deepcopy(ego_state)

        sorted(object_dict.items(), key=lambda x: x[0])

        # Empty Plot
        self._ax.cla()
        for _t in self.fig.texts:
            self.fig.texts.remove(_t)
        self._xx = []
        self._yy = []

        # Translate objects (only if ego_state is True, i.e. detection was in local coordinates)
        self.transformed_detection_dict = self.get_transformed_detection_input(
            detection_input=detection_input
        )

        # Plot detection input
        if self.transformed_detection_dict is not None:

            transformed_det_input_list = tracking_input
            if self.__bool_raw_data:
                self.plot_raw_detections(
                    transformed_detection_dict=self.transformed_detection_dict,
                )

            # Get matched object
            new_indices = {
                sensorname: {
                    obj_id: get_new_match_index(
                        obj=object_dict[obj_id],
                        obj_id=obj_id,
                        detected_objects=new_object_list,
                        match_dict_sens=match_dict[sensorname],
                    )
                    for obj_id in iter_keys
                    if obj_id != "ego" and new_object_list
                }
                for sensorname, (new_object_list, _) in transformed_det_input_list
            }

        # Go through all objects in observation storage
        _u = 0
        for obj_id in iter_keys:

            if self.args.show_filtered:
                if obj_id not in self.filtered_ids:
                    continue

            obj = object_dict[obj_id]
            if "t" not in obj.keys():
                continue

            # get colors
            if obj_id == "ego":
                obj_col = self.ego_col
            else:
                obj_col = self.color_list[int(int(obj_id) % len(self.color_list))]
                _u += 1

            if _u >= self.args.n_obs:
                break

            hist_t = self.plot_hist_fut(obj, obj_id, obj_col)

            # Plot detection input: iterate over sensors
            _dt = None
            _sensorname_list = None

            # Plot merged detections
            if self.transformed_detection_dict is not None:
                _sensorname_list = []
                for sensorname, (
                    new_object_list,
                    t_detect,
                ) in transformed_det_input_list:
                    self.t_last_time_receiv[sensorname] = t_detect
                    # No input from this sensor in this timestep
                    if not bool(new_object_list) or obj_id == "ego":
                        continue

                    # Get matched object
                    index_new = new_indices[sensorname].get(obj_id, None)

                    _dt = None
                    if index_new is not None:
                        # matched
                        if len(hist_t) > 1:
                            _dt = (t_detect - hist_t[1]) / self.ns_to_s

                        # store matched sensor info for object-ID
                        if obj_id not in matched_old_dict:
                            matched_old_dict[obj_id] = []
                        matched_old_dict[obj_id].append(sensorname)

                        if self.__bool_sensor_detection:
                            self.plot_matched_object(
                                sensorname=sensorname,
                                matched_obj=new_object_list[index_new],
                                obj_col=obj_col,
                            )

                        _sensorname_list.append(sensorname)
                    else:
                        if len(hist_t) == 1 and obj_id not in init_object_dict:
                            # init
                            init_object_dict[obj_id] = "-"

                        # not matched
                        if obj_id not in unmatched_old_dict:
                            unmatched_old_dict[obj_id] = []
                        unmatched_old_dict[obj_id].append(sensorname)

            # Check for filter values
            if obj_id in filter_log and filter_log[obj_id] is not None:
                self.plot_filter_log(
                    filter_log,
                    obj_id,
                    _sensorname_list,
                    obj_col,
                    object_dict,
                    _u,
                    ego_state,
                    obj,
                    _dt,
                )

            # Plot Prediction
            if self.__bool_predict:
                self.plot_pred(pred_dict, obj_id, obj_col)

        self.scatter = self._ax.scatter(
            self.obj_center_points_frame["x"],
            self.obj_center_points_frame["y"],
            marker="x",
            s=self.markersize,
            linewidths=self.linewidth,
            color=self.obj_center_points_frame["color"],
        )

        # Plot unmatched detections
        if self.transformed_detection_dict is not None:
            self.textstr += "\nUnmatched new objects: "
            self.plot_unmachted(transformed_det_input_list, new_indices)
        else:
            str_temp = ", ".join(
                [
                    str(sensorname)
                    + ": "
                    + str(int((time_stamp_ns - t_last) / 1e6))
                    + " ms"
                    for sensorname, t_last in self.t_last_time_receiv.items()
                ]
            )
            self.textstr += "\ntime wo detection: " + str_temp

        self.plt_bounds()

        if self.args.show_sample:
            return

        self.arange_plot(
            object_dict=object_dict,
            init_object_dict=init_object_dict,
            matched_old_dict=matched_old_dict,
            unmatched_old_dict=unmatched_old_dict,
        )

    def plt_bounds(self):
        """Plot track boundaries."""
        self._ax.plot(
            self.right_bound[:, 0],
            self.right_bound[:, 1],
            color=self.track_color["track_bounds"],
            linewidth=self.linewidth,
        )
        self._ax.plot(
            self.left_bound[:, 0],
            self.left_bound[:, 1],
            color=self.track_color["track_bounds"],
            label=None,
            linewidth=self.linewidth,
        )

    def __update_plot(self):
        self.sfreq.set_val(self.sfreq.val)

    def __prev_traj(self, _):
        if not self.__bool_prev_traj:
            self.__bool_prev_traj = True
            self.__prev_button.color = self.button_color_dict["On"]
        else:
            self.__bool_prev_traj = False
            self.__prev_button.color = self.button_color_dict["Off"]
        self.__update_plot()

    def __fut_traj(self, _):
        if not self.__bool_fut_traj:
            self.__bool_fut_traj = True
            self.__fut_button.color = self.button_color_dict["On"]
        else:
            self.__bool_fut_traj = False
            self.__fut_button.color = self.button_color_dict["Off"]
        self.__update_plot()

    def __det_(self, _):
        if not self.__bool_sensor_detection:
            self.__bool_sensor_detection = True
            self.__det_button.color = self.button_color_dict["On"]
        else:
            self.__bool_sensor_detection = False
            self.__det_button.color = self.button_color_dict["Off"]
        self.__update_plot()

    def __filter_(self, _):
        if not self.__bool_filter:
            self.__bool_filter = True
            self.__filter_button.color = self.button_color_dict["On"]
        else:
            self.__bool_filter = False
            self.__filter_button.color = self.button_color_dict["Off"]

        if self.__bool_filter:
            self.__set_visible(True)
            self.__filter_prior_button.on_clicked(self.__filter_prior)
            self.__filter_delay_button.on_clicked(self.__filter_delay)
            self.__filter_post_button.on_clicked(self.__filter_post)
        else:
            self.__set_visible(False)
            self.__filter_prior_button.color = self.button_color_dict["Off"]
            self.__filter_delay_button.color = self.button_color_dict["Off"]
            self.__filter_post_button.color = self.button_color_dict["Off"]
            self.__bool_filter_prior = False
            self.__bool_filter_delay = False
            self.__bool_filter_post = False
        self.__update_plot()

    def __filter_prior(self, _):
        if not self.__bool_filter_prior:
            self.__bool_filter_prior = True
            self.__filter_prior_button.color = self.button_color_dict["On"]
        else:
            self.__bool_filter_prior = False
            self.__filter_prior_button.color = self.button_color_dict["Off"]
        self.__update_plot()

    def __filter_delay(self, _):
        if not self.__bool_filter_delay:
            self.__bool_filter_delay = True
            self.__filter_delay_button.color = self.button_color_dict["On"]
        else:
            self.__bool_filter_delay = False
            self.__filter_delay_button.color = self.button_color_dict["Off"]
        self.__update_plot()

    def __filter_post(self, _):
        if not self.__bool_filter_post:
            self.__bool_filter_post = True
            self.__filter_post_button.color = self.button_color_dict["On"]
        else:
            self.__bool_filter_post = False
            self.__filter_post_button.color = self.button_color_dict["Off"]
        self.__update_plot()

    def __predict_(self, _):
        if not self.__bool_predict:
            self.__bool_predict = True
            self.__predict_button.color = self.button_color_dict["On"]
        else:
            self.__bool_predict = False
            self.__predict_button.color = self.button_color_dict["Off"]
        self.__update_plot()

    def __unmatch_(self, _):
        if not self.__bool_unmatched:
            self.__bool_unmatched = True
            self.__unmatch_button.color = self.button_color_dict["On"]
        else:
            self.__bool_unmatched = False
            self.__unmatch_button.color = self.button_color_dict["Off"]
        self.__update_plot()

    def __raw_data_(self, _):
        if not self.__bool_raw_data:
            self.__bool_raw_data = True
            self.__raw_data_button.color = self.button_color_dict["On"]
        else:
            self.__bool_raw_data = False
            self.__raw_data_button.color = self.button_color_dict["Off"]
        self.__update_plot()

    def __check_(self, event):
        index = self.labels_check.index(event)
        if not self.__bool_check_button[index]:
            self.__bool_check_button[index] = True
        else:
            self.__bool_check_button[index] = False
        self.__update_plot()

    def __custom_states_plt(self, event):
        recieve_list = literal_eval(event)
        self.__plt_states(obj_id_list=recieve_list)

    def __custom_filter_plt(self, event):
        recieve_list = literal_eval(event)
        self.__plt_filter(obj_id_list=recieve_list)

    def __check_mis_stats(self, _):
        self.__plt_mis_match()

    def __plt_states(self, obj_id_list=False, show_ego=False, n_obs=None):
        """Plot filter prior-posterior propagation."""
        n_states = 6  # x, y, yaw, v, yawrate, a
        # Plot all six states (x, y, yaw, v, yawrate, a)
        n_figs = 6

        stor_dots = "k--"
        stor_lab = "Storages"
        prior_dots = "g+"
        prior_lab = "Prior"
        meas_dots = "*"
        meas_lab = "Measure"
        post_dots = "mx"
        post_lab = "Posterior"

        # Plot the "self.args.n_obs" most occured objects
        if obj_id_list:
            if "ego" in obj_id_list:
                show_ego = True
                obj_id_list.remove("ego")
            _pltid_list = obj_id_list
        else:
            _pltid_list = [vals[0] for vals in self.count_obj_ids]

        if n_obs is not None and n_obs + int(show_ego) < self.args.n_obs:
            _pltid_list = _pltid_list[: n_obs + int(show_ego)]
            print("Reduce number of object to {}".format(n_obs))

        for _pltid in _pltid_list:
            _, axs = plt.subplots(n_figs, figsize=(16, 9), sharex=True)
            plt_vals = self.obj_items[str(_pltid)]

            # Init Lists
            tabs_list = [0.0] * len(plt_vals["object_states"])
            states = np.zeros(
                [len(plt_vals["object_states"]), n_states]
            )  # Indices [x, y, yaw, v, yawrate, a]
            priors = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * len(plt_vals["object_states"])
            measures = [None] * len(plt_vals["object_states"])
            posteriors = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * len(
                plt_vals["object_states"]
            )

            time_list_iter = iter(self.obj_items[str(_pltid)]["t_dict"])
            for j, (idx, obj) in enumerate(plt_vals["object_states"].items()):
                tabs_list[j] = copy.deepcopy(
                    self.obj_items[str(_pltid)]["t_dict"][next(time_list_iter)] / 1e9
                )

                # State Variable form storage
                states[j, :] = obj

                # Filter propagations: Only for succesful matching and measurment update
                if idx in plt_vals["filter_vals"].keys():
                    # mean over filter_vals
                    _priors = []
                    _posteriors = []
                    _measur = []
                    for _val in plt_vals["filter_vals"][idx]:
                        if _val[0] is None:
                            continue
                        _priors.append(np.array(_val[0]))
                        _posteriors.append(np.array(_val[4]))
                        _measur.append(_val[7])

                    if _priors != []:
                        priors[j] = list([np.mean(k) for k in zip(*_priors)])
                    if _posteriors != []:
                        posteriors[j] = list([np.mean(k) for k in zip(*_posteriors)])
                    measures[j] = _measur

            # Plot all States of measures
            _meas_x = [None] * len(measures)
            _meas_y = [None] * len(measures)
            _meas_yaw = [None] * len(measures)
            _meas_v = [None] * len(measures)
            _meas_yawrate = [None] * len(measures)
            _meas_a = [None] * len(measures)
            for _k, _mm in enumerate(measures):
                _avg_x = []
                _avg_y = []
                _avg_yaw = []
                _avg_v = []
                _avg_yawrate = []
                _avg_a = []
                if _mm is not None:
                    for _mval in _mm:
                        meas_key = self.params["ACTIVE_SENSORS"][_mval[1]]["meas_vars"]
                        if "x" in meas_key:
                            _avg_x.extend([_mval[0][meas_key.index("x")]])
                        if "y" in meas_key:
                            _avg_y.extend([_mval[0][meas_key.index("y")]])
                        if "yaw" in meas_key:
                            _avg_yaw.extend([_mval[0][meas_key.index("yaw")]])
                        if "v" in meas_key:
                            _avg_v.extend([_mval[0][meas_key.index("v")]])
                        if "yawrate" in meas_key:
                            _avg_yawrate.extend([_mval[0][meas_key.index("yawrate")]])
                        if "a" in meas_key:
                            _avg_a.extend([_mval[0][meas_key.index("a")]])
                if _avg_x != []:
                    _meas_x[_k] = np.mean(_avg_x)
                if _avg_y != []:
                    _meas_y[_k] = np.mean(_avg_y)
                if _avg_yaw != []:
                    _meas_yaw[_k] = np.mean(_avg_yaw)
                if _avg_v != []:
                    _meas_v[_k] = np.mean(_avg_v)
                if _avg_yawrate != []:
                    _meas_yawrate[_k] = np.mean(_avg_yawrate)
                if _avg_a != []:
                    _meas_a[_k] = np.mean(_avg_a)

            # Plot all States
            idx_priors = [j for j, ff in enumerate(priors) if ff[0] != 0.0]
            idx_posts = [j for j, ff in enumerate(posteriors) if ff[0] != 0.0]
            plt_t_list = list(map(tabs_list.__getitem__, idx_priors))
            plt_t_list = list(map(tabs_list.__getitem__, idx_posts))

            for k in range(n_states):
                axs[k].plot(
                    plt_t_list,
                    [priors[idx][k] for idx in idx_priors],
                    prior_dots,
                    label=prior_lab,
                    markersize=self.markersize,
                    linewidth=self.linewidth,
                )
                if k == 0:
                    axs[k].scatter(
                        tabs_list, _meas_x, c="r", marker=meas_dots, label=meas_lab
                    )
                if k == 1:
                    axs[k].scatter(
                        tabs_list, _meas_y, c="r", marker=meas_dots, label=meas_lab
                    )
                if k == 2:
                    axs[k].scatter(
                        tabs_list, _meas_yaw, c="r", marker=meas_dots, label=meas_lab
                    )
                if k == 3:
                    axs[k].scatter(
                        tabs_list, _meas_v, c="r", marker=meas_dots, label=meas_lab
                    )
                if k == 4:
                    axs[k].scatter(
                        tabs_list,
                        _meas_yawrate,
                        c="r",
                        marker=meas_dots,
                        label=meas_lab,
                    )
                if k == 5:
                    axs[k].scatter(
                        tabs_list, _meas_a, c="r", marker=meas_dots, label=meas_lab
                    )
                axs[k].plot(
                    plt_t_list,
                    [posteriors[idx][k] for idx in idx_posts],
                    post_dots,
                    label=post_lab,
                    markersize=self.markersize,
                    linewidth=self.linewidth,
                )
                axs[k].plot(tabs_list, states[:, k], stor_dots, label=stor_lab)
                axs[k].set_ylabel(self.ylabel_list[k])
                if k == n_states - 1:
                    axs[k].set_xlabel("Time in s")
            axs[0].legend(loc="upper center", bbox_to_anchor=(0.8, 1.35), ncol=4)
            _ = [axx.grid() for axx in axs]
            axs[0].set_title("Obj-ID: " + str(_pltid))
        if show_ego:
            self.__plt_ego_states()
        plt.show()

    def __plt_ego_states(self):
        """Plot EGO-States for comparison."""
        n_states = 6  # x, y, yaw, v, yawrate, a
        # Plot all six states (x, y, yaw, v, yawrate, a)
        n_figs = 6

        dots = "m-"
        lab = "Ego-States"

        # Plot ego states
        t_list = (
            np.array(list(self.obj_items["ego"]["t_dict"].values()))
            - list(self.obj_items["ego"]["t_dict"].values())[0]
        ) / 1e9
        _, axs = plt.subplots(n_figs, figsize=(16, 9), sharex=True)

        # Plot States
        for k in range(n_states):
            axs[k].plot(
                t_list,
                [state[k] for state in self.obj_items["ego"]["object_states"].values()],
                dots,
                linewidth=self.linewidth,
                label=lab,
            )
            axs[k].set_ylabel(self.ylabel_list[k])
            if k == n_states - 1:
                axs[k].set_xlabel("Zeit in s")

        axs[0].legend(loc="upper center", bbox_to_anchor=(0.8, 1.35), ncol=4)
        _ = [axx.grid() for axx in axs]
        axs[0].set_title("Obj-ID: EGO")
        plt.show()

    def get_filter_step_w_measurement(self, obj_id):
        """Get all filter steps, which contain a valid detection input."""
        filter_list = []
        self.n_sensors[obj_id] = []

        for key, values in self.obj_items[str(obj_id)]["filter_vals"].items():
            # iterate in case of multiple perception inputs
            kk = False
            for val in values:
                # use only if measurment is matched (idx 6 is measurment)
                # val[6] is [obj_filter.z_meas, sensor]
                if val[6] is not None:
                    # val[7][1] is sensor name
                    if val[7][1] not in self.n_sensors[obj_id]:
                        self.n_sensors[obj_id].append(val[7][1])
                    # check if yaw is mesured
                    if "yaw" in self.params["ACTIVE_SENSORS"][val[7][1]]["meas_vars"]:
                        if "yaw" not in self.measure_variables:
                            self.measure_variables.append("yaw")

                    # check if v is mesured
                    if "v" in self.params["ACTIVE_SENSORS"][val[7][1]]["meas_vars"]:
                        if "v" not in self.measure_variables:
                            self.measure_variables.append("v")

                    if not kk:
                        filter_list.append((key, values))
                        kk = True

        if not filter_list:
            return None, None, None, None, None

        counter, objfilter = zip(*filter_list)

        p_post_dict = {
            sensor_str: [np.nan for _ in range(len(objfilter) - 1)]
            for sensor_str in self.n_sensors[obj_id]
        }
        nis_dict = {
            sensor_str: [np.nan for _ in range(len(objfilter) - 1)]
            for sensor_str in self.n_sensors[obj_id]
        }
        measure_res_dict = {
            sensor_str: [np.nan for _ in range(len(objfilter) - 1)]
            for sensor_str in self.n_sensors[obj_id]
        }
        sensor_measure_dict = {
            sensor_str: [np.nan for _ in range(len(objfilter) - 1)]
            for sensor_str in self.n_sensors[obj_id]
        }

        # get covariance and residuum per sensor
        for j, obj_fil in enumerate(objfilter):
            for _fil_vals in obj_fil:
                # _fil_vals[1] is prior P matrix
                # _fil_vals[7][1] is sensor name
                # _fil_vals[7][0] is sensor measure
                # _fil_vals[5] is post covariance
                # _fil_vals[6] is residuum
                if j > 1 and _fil_vals[7] is not None:
                    sensor_str = _fil_vals[7][1]
                    h_mat = get_H_mat(
                        index_tuples=self.params["TRACKING"]["filter"]["H_indices"][
                            sensor_str
                        ]
                    )
                    r_mat = get_R_mat(
                        ego_pos=np.array(self.obj_items["ego"]["object_states"][j][:2]),
                        obj_pos=np.array(_fil_vals[0][:2]),
                        _measure_covs_sens=self.measure_covs_dict[sensor_str],
                    )
                    meas_res = _fil_vals[6]
                    s_mat = np.dot(h_mat, np.dot(_fil_vals[1], h_mat.T)) + r_mat

                    meas_res_arr = np.array(meas_res)
                    nis_dict[sensor_str][j - 1] = np.dot(
                        np.dot(meas_res_arr.T, np.linalg.inv(s_mat)), meas_res_arr
                    )

                    p_post_dict[sensor_str][j - 1] = _fil_vals[5]
                    measure_res_dict[sensor_str][j - 1] = _fil_vals[6]
                    sensor_measure_dict[sensor_str][j - 1] = _fil_vals[7][0]

        return counter, p_post_dict, measure_res_dict, sensor_measure_dict, nis_dict

    def get_filter_errors(self, measure_res, inpt_yaws, str_sens, p_post=None):
        """Get errors of filter steps."""
        xy_loc = [np.array([np.nan, np.nan]) for _ in range(len(measure_res))]
        v_delta = [np.nan for _ in range(len(measure_res))]
        yaw_delta = [np.nan for _ in range(len(measure_res))]
        p_loc = [np.full([6, 6], np.nan) for _ in range(len(measure_res))]

        # bool checks
        is_yaw_meas = bool(
            "yaw" in self.params["ACTIVE_SENSORS"][str_sens]["meas_vars"]
        )
        is_v_meas = bool("v" in self.params["ACTIVE_SENSORS"][str_sens]["meas_vars"])
        add_idx = int(len(self.params["ACTIVE_SENSORS"][str_sens]["meas_vars"]) > 3)

        for _kk, meas_res in enumerate(measure_res):
            if meas_res is None or np.isnan(meas_res).any():
                continue

            # get x, y residuum
            rot_angle = inpt_yaws[_kk]
            xy_loc[_kk] = rotate_glob_loc(
                global_matrix=meas_res[:2],
                rot_angle=rot_angle,
                matrix=False,
            )

            # get yaw residuum
            if is_yaw_meas:
                yaw_delta[_kk] = meas_res[2] * 180.0 / np.pi

            # get v residuum
            if is_v_meas:
                v_delta[_kk] = meas_res[2 + add_idx]

            # get local covariances
            if p_post is None or np.isnan(p_post[_kk]).any():
                continue

            p_loc[_kk] = np.array(p_post[_kk])
            p_loc[_kk][:2, :2] = rotate_glob_loc(
                global_matrix=p_loc[_kk][:2, :2],
                rot_angle=rot_angle,
                matrix=True,
            )

        # local x, local y
        x_delta, y_delta = zip(*xy_loc)

        if p_post is None:
            return x_delta, y_delta, yaw_delta, v_delta

        return x_delta, y_delta, yaw_delta, v_delta, p_loc

    def eval_filter_performance(
        self, obj_id, j=None, rows=None, cell_cols=None, cell_text=None, axs=None
    ):
        """Get all steps with matched measurement."""
        (
            counter,
            p_post_dict,
            measure_res_dict,
            sensor_measure_dict,
            s_mat_dict,
        ) = self.get_filter_step_w_measurement(obj_id)

        if counter is None:
            return (
                None,
                None,
                None,
                None,
                None,
                None,
            )

        # get yaw for rotation back to local coordinates
        obj_yaws = [
            (key, val[2])
            for key, val in self.obj_items[str(obj_id)]["object_states"].items()
        ]
        yaw_counter, yaws = zip(*obj_yaws)
        inpt_yaws = np.interp(counter, yaw_counter, yaws)

        if axs is None:
            return (
                counter,
                p_post_dict,
                measure_res_dict,
                sensor_measure_dict,
                inpt_yaws,
                s_mat_dict,
            )

        # check if yaw and v should also plotted
        set_yaw = 0
        if "yaw" in self.measure_variables and "v" in self.measure_variables:
            set_yaw = 1

        # table infos
        markersize = self.markersize
        cell_text["x"].append([])
        cell_text["y"].append([])
        cell_text["yaw"].append([])
        cell_text["v"].append([])
        cell_text["P"].append([])
        cell_text["res"].append([])
        rows.append(str(obj_id))
        col = self.color_list[int(int(obj_id) % len(self.color_list))]
        cell_cols.append(col)

        for str_sens in self.num_sensors_all:
            # object is never seen by object
            if str_sens not in self.n_sensors[obj_id]:
                cell_text["x"][j].append("-, -")
                cell_text["y"][j].append("-, -")

                if "yaw" in self.params["ACTIVE_SENSORS"][str_sens]["meas_vars"]:
                    cell_text["yaw"][j].append("-, -")

                if "v" in self.params["ACTIVE_SENSORS"][str_sens]["meas_vars"]:
                    cell_text["v"][j].append("-, -")

                cell_text["P"][j].append("-")
                cell_text["res"][j].append("-")

                continue

            _p_post = p_post_dict[str_sens]
            measure_res = measure_res_dict[str_sens]

            x_delta, y_delta, yaw_delta, v_delta = self.get_filter_errors(
                measure_res,
                inpt_yaws,
                str_sens,
            )

            # x residuum
            axs[0, 0].set_title("Residual is measurement - prediction")
            axs[0, 0].plot(
                counter[1:],
                x_delta,
                marker=self._sensor_marker[str_sens],
                markersize=markersize,
                linewidth=self.linewidth,
                linestyle="None",
                color=col,
            )
            cell_text["x"][j].append(
                "{:.2f}m, {:.2f}m".format(
                    np.nanmean(x_delta),
                    np.nanstd(x_delta),
                )
            )

            # y residuum
            axs[1, 0].plot(
                counter[1:],
                y_delta,
                marker=self._sensor_marker[str_sens],
                markersize=markersize,
                linewidth=self.linewidth,
                linestyle="None",
                color=col,
            )
            cell_text["y"][j].append(
                "{:.2f}m, {:.2f}m".format(
                    np.nanmean(y_delta),
                    np.nanstd(y_delta),
                )
            )

            if "yaw" in self.params["ACTIVE_SENSORS"][str_sens]["meas_vars"]:
                # yaw residuum
                axs[2, 0].plot(
                    counter[1:],
                    yaw_delta,
                    marker=self._sensor_marker[str_sens],
                    markersize=markersize,
                    linewidth=self.linewidth,
                    linestyle="None",
                    color=col,
                )
                cell_text["yaw"][j].append(
                    "{:.2f}m, {:.2f}m".format(
                        np.nanmean(yaw_delta),
                        np.nanstd(yaw_delta),
                    )
                )

            if "v" in self.params["ACTIVE_SENSORS"][str_sens]["meas_vars"]:
                # v residuum
                axs[2 + set_yaw, 0].plot(
                    counter[1:],
                    v_delta,
                    marker=self._sensor_marker[str_sens],
                    markersize=markersize,
                    linewidth=self.linewidth,
                    linestyle="None",
                    color=col,
                )
                cell_text["v"][j].append(
                    "{:.2f}m, {:.2f}m".format(
                        np.nanmean(v_delta),
                        np.nanstd(v_delta),
                    )
                )

            # Plot |P|
            jj, cov_x, cov_y = zip(
                *[
                    (j, PP[0][0], PP[1][1])
                    for j, PP in enumerate(_p_post)
                    if not np.isnan(PP).any()
                ]
            )
            idx_list = [counter[jk] for jk in jj]
            axs[-2, 0].plot(
                idx_list,
                cov_x,
                marker=self._sensor_marker[str_sens],
                markersize=markersize,
                linewidth=self.linewidth,
                linestyle="None",
                color=col,
            )
            axs[-2, 0].plot(
                idx_list,
                cov_y,
                marker=self._sensor_marker[str_sens],
                markersize=markersize,
                linewidth=self.linewidth,
                linestyle="None",
                color=col,
            )
            cell_text["P"][j].append(
                "{:.2f}, {:.2f}".format(np.nanmean(cov_x), np.nanstd(cov_y))
            )

            # Plot |residuum of position|
            jj, y_res_abs = zip(
                *[
                    (j, np.linalg.norm(meas_re[:2]))
                    for j, meas_re in enumerate(measure_res)
                    if not np.isnan(meas_re).any()
                ]
            )
            idx_list = [counter[jk] for jk in jj]
            axs[-1, 0].plot(
                idx_list,
                y_res_abs,
                marker=self._sensor_marker[str_sens],
                markersize=markersize,
                linewidth=self.linewidth,
                linestyle="None",
                color=col,
            )
            cell_text["res"][j].append(
                "{:.2f}m, {:.2f}m".format(
                    mean(y_res_abs),
                    stdev(y_res_abs),
                )
            )

        return set_yaw

    def __plt_filter(self, obj_id_list=False, n_obs=5):
        rows = []
        cell_text = {"x": [], "y": [], "yaw": [], "v": [], "P": [], "res": []}
        cell_cols = []

        if obj_id_list:
            if "ego" in obj_id_list:
                print("No _plt_filter for EGO available!!")
                obj_id_list.remove("ego")
            _pltid_list = obj_id_list
        else:
            _pltid_list = [vals[0] for vals in self.count_obj_ids[:n_obs]]
            if len(self.count_obj_ids) > n_obs:
                print("Reduce number of object to {}".format(n_obs))

        # plot x, y and yaw / v (if measured) and norm(P) and norm(y)
        n_subplots = len(self.measure_variables) + 2
        _, axs = plt.subplots(
            n_subplots,
            2,
            figsize=(16, 9),
            gridspec_kw={"width_ratios": [5, 1]},
        )
        for _j in range(n_subplots):
            axs[_j, 1].axis("off")

        for j, obj_id in enumerate(_pltid_list):
            set_yaw = self.eval_filter_performance(
                obj_id=obj_id,
                j=j,
                rows=rows,
                cell_cols=cell_cols,
                cell_text=cell_text,
                axs=axs,
            )

            # header-columns for tables
            columns_xy = []
            columns_yaw = []
            columns_v = []
            columns_p_res = []
            for str_sens in self.num_sensors_all:
                if str_sens not in self.n_sensors[obj_id]:
                    columns_xy.append(str(str_sens))
                    if "yaw" in self.params["ACTIVE_SENSORS"][str_sens]["meas_vars"]:
                        columns_yaw.append(str(str_sens))
                    if "v" in self.params["ACTIVE_SENSORS"][str_sens]["meas_vars"]:
                        columns_v.append(str(str_sens))
                    columns_p_res.append(str(str_sens))
                    continue
                columns_xy.append(str(str_sens) + "\n$\\mu$ m, $\\sigma$ m")
                if "yaw" in self.params["ACTIVE_SENSORS"][str_sens]["meas_vars"]:
                    columns_yaw.append(str(str_sens) + "\n$\\mu$ °, $\\sigma$ °")
                if "v" in self.params["ACTIVE_SENSORS"][str_sens]["meas_vars"]:
                    columns_v.append(str(str_sens) + "\n$\\mu$ m/s, $\\sigma$ m/s")
                columns_p_res.append(str(str_sens))

        # Plot Delta x
        axs[0, 0].set_ylabel("$\\Delta x$")
        axs[0, 0].grid(True)

        if cell_text["x"][0] != []:
            table_x = axs[0, 1].table(
                cellText=cell_text["x"],
                rowLabels=rows,
                rowColours=cell_cols,
                colLabels=columns_xy,
                loc="center",
                zorder=3,
            )
            table_x.auto_set_font_size(False)
            table_x.set_fontsize(6)

        # Plot Delta y
        axs[1, 0].set_ylabel("$\\Delta y$")
        axs[1, 0].grid(True)

        if cell_text["y"][0] != []:
            table_y = axs[1, 1].table(
                cellText=cell_text["y"],
                rowLabels=rows,
                rowColours=cell_cols,
                colLabels=columns_xy,
                loc="center",
                zorder=3,
            )
            table_y.auto_set_font_size(False)
            table_y.set_fontsize(6)

        if "yaw" in self.measure_variables:
            # Plot Delta yaw
            axs[2, 0].set_ylabel("$\\Delta \\Psi$")
            axs[2, 0].grid(True)
            if cell_text["yaw"][0] != []:
                table_yaw = axs[2, 1].table(
                    cellText=cell_text["yaw"],
                    rowLabels=rows,
                    rowColours=cell_cols,
                    colLabels=columns_yaw,
                    loc="center",
                    zorder=3,
                )
                table_yaw.auto_set_font_size(False)
                table_yaw.set_fontsize(6)

        if "v" in self.measure_variables:
            # Plot Delta v
            axs[2 + set_yaw, 0].set_ylabel("$\\Delta v$")
            axs[2 + set_yaw, 0].grid(True)
            if cell_text["v"][0] != []:
                table_v = axs[2 + set_yaw, 1].table(
                    cellText=cell_text["v"],
                    rowLabels=rows,
                    rowColours=cell_cols,
                    colLabels=columns_v,
                    loc="center",
                    zorder=3,
                )
                table_v.auto_set_font_size(False)
                table_v.set_fontsize(6)

        # Plot |P|
        axs[-2, 0].set_ylabel("|P|")
        axs[-2, 0].grid(True)

        if cell_text["P"][0] != []:
            table_p = axs[-2, 1].table(
                cellText=cell_text["P"],
                rowLabels=rows,
                rowColours=cell_cols,
                colLabels=columns_p_res,
                loc="center",
                zorder=3,
            )
            table_p.auto_set_font_size(False)
            table_p.set_fontsize(6)

        # Plot |residuum|
        axs[-1, 0].set_xlabel("steps")
        axs[-1, 0].set_ylabel("|residuum|")
        axs[-1, 0].grid(True)

        if cell_text["res"][0] != []:
            table_res = axs[-1, 1].table(
                cellText=cell_text["res"],
                rowLabels=rows,
                rowColours=cell_cols,
                colLabels=columns_p_res,
                loc="center",
                zorder=3,
            )
            table_res.auto_set_font_size(False)
            table_res.set_fontsize(6)

        # Debug:
        if cell_text["x"][0] == []:
            print("Probably too short observation time.")
        plt.setp(axs, xlim=axs[0, 0].get_xlim())
        plt.show()

    def __plt_mis_match(self):
        permutations = {}
        mis_matches = {}
        # get overall filter logs
        match_vals = [(j, log_vals[4]) for j, log_vals in enumerate(self.log_rows)]

        for k, matchings in enumerate(match_vals):
            for sensor_name, sens_match in matchings[1].items():
                if sensor_name not in permutations.keys():
                    permutations[sensor_name] = {"counts": 0, "mises": 0}
                    mis_matches[sensor_name] = {ij: 0 for ij in range(len(match_vals))}
                if bool(sens_match):
                    permutations[sensor_name]["counts"] += 1
                    for _tt in sens_match:
                        if _tt[0] != _tt[1]:
                            permutations[sensor_name]["mises"] += 1
                            mis_matches[sensor_name][k] += 1

        # Plot mis-match stats
        obj_dicts = [(j, log_vals[6]) for j, log_vals in enumerate(self.log_rows)]
        _, axs = plt.subplots(2, figsize=(16, 9))
        _cc, objs = zip(*obj_dicts)
        n_objs = [len(obj) if obj is not None else -1 for obj in objs]
        axs[0].plot(_cc, n_objs, "x", markersize=self.markersize)
        min_n, max_n = (0, 0)
        for key, val in mis_matches.items():
            counts, mises = zip(*val.items())
            min_n = min(min_n, min(mises))
            max_n = max(max_n, max(mises))
            axs[1].plot(
                counts,
                mises,
                label="{:s}, Total mises: {:d} / {:d}".format(
                    key, permutations[key]["mises"], permutations[sensor_name]["counts"]
                ),
            )

        axs[0].set_ylabel("No. of Objects in Storage")
        axs[0].grid(True)
        axs[0].set_yticks(range(min(n_objs) - 1, max(n_objs) + 2))

        axs[1].set_ylabel("Mis-Matches")
        axs[1].set_xlabel("steps")
        axs[1].grid(True)
        axs[1].set_yticks(range(min_n - 1, max_n + 2))
        axs[1].legend()
        plt.show()

    def get_transformed_detection_input(self, detection_input: dict):
        """Transform detection input (local to global)."""
        transformed_detection_dict = copy.deepcopy(detection_input)
        if transformed_detection_dict is not None:
            for sensor_str, (
                detection_list,
                detection_timestamp_ns,
            ) in transformed_detection_dict.items():
                if (
                    self.params["ACTIVE_SENSORS"][sensor_str]["bool_local"]
                    and detection_list is not None
                ):
                    yaw_from_track = bool(
                        self.params["ACTIVE_SENSORS"][sensor_str].get("get_yaw", None)
                        == "from_track"
                    )
                    ego_t_idx_list = list(self.obj_items["ego"]["t_dict"].values())
                    ego_idx = np.abs(
                        np.array(ego_t_idx_list) - detection_timestamp_ns
                    ).argmin()
                    ego_t = ego_t_idx_list[ego_idx]
                    ego_state = list(self.obj_items["ego"]["object_states"].values())[
                        ego_idx
                    ]
                    self.transform_objects(
                        detection_list,
                        yaw_from_track,
                        detection_timestamp_ns,
                        ego_t=ego_t,
                        ego_state=np.array(ego_state),
                    )
                else:
                    yaw_from_track = bool(
                        self.params["ACTIVE_SENSORS"][sensor_str].get("get_yaw", None)
                        == "from_track"
                    )

                if yaw_from_track and detection_list:
                    self.track_boundary.add_yaw_batch(detection_list)

        return transformed_detection_dict

    def arange_plot(
        self,
        object_dict,
        init_object_dict,
        matched_old_dict,
        unmatched_old_dict,
    ):
        """Arange plot, i.e. set limits, add textbox, add legend, add grid."""
        if len(self._xx) > 0:
            self._ax.set_xlim(
                [
                    np.min(self._xx) - self.axis_range,
                    np.max(self._xx) + self.axis_range,
                ]
            )
            self._ax.set_ylim(
                [
                    np.min(self._yy) - self.axis_range,
                    np.max(self._yy) + self.axis_range,
                ]
            )

        # Add text box
        self.textstr += "\nNew objects: "
        self.textstr += ", ".join(
            [
                str(obj_id) + " (" + sensor_str + ")"
                for obj_id, sensor_str in init_object_dict.items()
            ]
        )
        self.textstr += "\nMatched old objects: "
        self.textstr += ", ".join(
            [
                str(obj_id) + " (" + ", ".join(sensor_list) + ")"
                for obj_id, sensor_list in matched_old_dict.items()
            ]
        )
        self.textstr += "\nUnmatched old objects: "
        self.textstr += ", ".join(
            [
                str(obj_id) + " (" + ", ".join(sensor_list) + ")"
                for obj_id, sensor_list in unmatched_old_dict.items()
            ]
        )
        self.textstr += "\nn_obj = {:d}".format(len(object_dict) - 1)

        # plot text box
        self.text_box = plt.gcf().text(
            0.02, 0.02, self.textstr, fontsize=8, bbox=self.textprops
        )
        self._ax.plot(
            self.right_bound_pit[:, 0],
            self.right_bound_pit[:, 1],
            color=self.track_color["pit_bounds"],
            linewidth=self.linewidth,
        )
        self._ax.plot(
            self.left_bound_pit[:, 0],
            self.left_bound_pit[:, 1],
            color=self.track_color["pit_bounds"],
            label=None,
            linewidth=self.linewidth,
        )

        self._ax.legend(self.markers, self.labels, loc=4)

        self._ax.set_ylabel("North in m")
        self._ax.set_xlabel("East in m")
        self._ax.grid(True)

    def plot_unmachted(self, transformed_det_input_list, new_indices):
        """Plot unmatched objects in detection input."""
        any_unmatches = False
        for sensorname, (
            new_object_list,
            _,
        ) in transformed_det_input_list:
            if not new_object_list:
                continue

            num_unmatches = 0
            for _kk, new_obj in enumerate(new_object_list):
                if _kk in new_indices[sensorname].values():
                    continue
                if self.__bool_unmatched:
                    self._set_label(
                        "Unmatched: {}".format(sensorname),
                        self._sensor_marker[sensorname],
                        markeredge="b",
                    )
                    if "yaw" in new_obj["keys"]:
                        edges_glob = rotate_loc_glob(
                            self.rectangle, new_obj["state"][2], matrix=False
                        ) + np.expand_dims(np.array(new_obj["state"][:2]), axis=1)
                        poly_1 = Polygon(edges_glob.T)
                        _x, _y = poly_1.exterior.xy
                        self._ax.plot(
                            _x,
                            _y,
                            linewidth=self.linewidth,
                            linestyle="dashed",
                            color="silver",
                        )

                    self._ax.plot(
                        new_obj["state"][0],
                        new_obj["state"][1],
                        marker=self._sensor_marker[sensorname],
                        mfc="none",
                        markersize=self.markersize,
                        linewidth=self.linewidth,
                        color="silver",
                    )
                    self._xx.append(new_obj["state"][0])
                    self._yy.append(new_obj["state"][1])
                num_unmatches += 1

            if num_unmatches > 0:
                if any_unmatches:
                    self.textstr += ", "
                any_unmatches = True
                self.textstr += "{:s} = {:d}".format(sensorname, num_unmatches)

    def plot_raw_detections(self, transformed_detection_dict: dict):
        """Plot raw detections."""
        self.raw_data = {
            "id": [],
            "x": [],
            "y": [],
            "sens": [],
        }
        for _sens_, (det_input, _) in transformed_detection_dict.items():
            for det_check_obj in det_input:
                self._set_label(
                    "Raw_detection: " + str(_sens_),
                    self._sensor_marker[_sens_],
                )
                if "yaw" in det_check_obj["keys"]:
                    edges_glob = rotate_loc_glob(
                        self.rectangle,
                        det_check_obj["state"][2],
                        matrix=False,
                    ) + np.expand_dims(
                        np.array(det_check_obj["state"][:2]),
                        axis=1,
                    )
                    poly_1 = Polygon(edges_glob.T)
                    _x, _y = poly_1.exterior.xy
                    self._ax.plot(
                        _x,
                        _y,
                        linewidth=self.linewidth,
                        linestyle="dashdot",
                        color=self.raw_data_color,
                    )
                    __x, __y = poly_1.centroid.xy
                    self._ax.plot(
                        __x,
                        __y,
                        marker=self._sensor_marker[_sens_],
                        mfc="none",
                        markersize=self.markersize,
                        linewidth=self.linewidth,
                        color=self.raw_data_color,
                    )
                    self.raw_data["x"].append(__x)
                    self.raw_data["y"].append(__y)
                self._xx.append(det_check_obj["state"][0])
                self._yy.append(det_check_obj["state"][1])
                self.scatter1 = self._ax.scatter(
                    self.raw_data["x"],
                    self.raw_data["y"],
                    s=self.markersize,
                    linewidths=self.linewidth,
                    color=self.raw_data_color,
                )

    def plot_hist_fut(self, obj, obj_id, obj_col):
        """Plot history and future of object."""
        # get full object history
        obj_items = self.obj_items[obj_id]

        # get object history in time-region of interest
        t_hist = max(0, obj["t"] - self.dt_hist)
        max_t = max(list(obj_items["t_dict"].values()))
        t_fut = min(max_t, obj["t"] + self.dt_fut)
        hist_t, hist_x, hist_y, hist_yaw = zip(
            *[
                [obj_items["t_dict"][idx]] + values[:3]
                for idx, values in obj_items["object_states"].items()
                if t_hist <= obj_items["t_dict"][idx] <= obj["t"]
            ]
        )
        fut_t, fut_x, fut_y = zip(
            *[
                [obj_items["t_dict"][idx]] + values[:2]
                for idx, values in obj_items["object_states"].items()
                if obj["t"] <= obj_items["t_dict"][idx] <= t_fut
            ]
        )
        t_intp = np.arange(fut_t[0], fut_t[-1], 0.1 * 1e9)
        fut_x = np.interp(t_intp, fut_t, fut_x)
        fut_y = np.interp(t_intp, fut_t, fut_y)

        # newest values first (at index 0)
        hist_t = list(hist_t)
        hist_x = list(hist_x)
        hist_y = list(hist_y)
        hist_yaw = list(hist_yaw)
        fut_t = list(fut_t)
        fut_x = list(fut_x)
        fut_y = list(fut_y)
        hist_x.reverse()
        hist_y.reverse()
        hist_t.reverse()
        hist_yaw.reverse()

        # Add latest objects position to adjust legend
        if obj_id == "ego" or len(hist_x) > 2:
            self._xx.append(obj["state"][0])
            self._yy.append(obj["state"][1])

        # Plot observation storage: latest position
        if self.__bool_prev_traj:
            edges_glob = rotate_loc_glob(
                self.rectangle, hist_yaw[-1], matrix=False
            ) + np.expand_dims(np.array([hist_x[-1], hist_y[-1]]), axis=1)
            poly_1 = Polygon(edges_glob.T)
            _x, _y = poly_1.exterior.xy
            self._ax.plot(
                _x,
                _y,
                linewidth=self.linewidth,
                color=obj_col,
                linestyle="dashed",
            )
            self._ax.plot(
                hist_x,
                hist_y,
                "-",
                color=obj_col,
            )
            self._xx.append(hist_x[-1])
            self._yy.append(hist_y[-1])

        # store obj information
        self.obj_center_points_frame["id"].append(obj_id)
        self.obj_center_points_frame["x"].append(obj["state"][0])
        self.obj_center_points_frame["y"].append(obj["state"][1])
        self.obj_center_points_frame["yaw"].append(obj["state"][2])
        self.obj_center_points_frame["v"].append(obj["state"][3])
        self.obj_center_points_frame["yawrate"].append(obj["state"][4])
        self.obj_center_points_frame["a"].append(obj["state"][5])
        self.obj_center_points_frame["color"].append(obj_col)
        for key in self.append_list:
            self.obj_center_points_frame[key].append(None)

        # init annotate
        self.annot = self._ax.annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
        )
        self.annot.set_visible(False)

        edges_glob = rotate_loc_glob(
            self.rectangle, obj["state"][2], matrix=False
        ) + np.expand_dims(np.array(obj["state"][:2]), axis=1)
        poly_1 = Polygon(edges_glob.T)
        _x, _y = poly_1.exterior.xy
        self._ax.plot(
            _x,
            _y,
            linewidth=self.linewidth,
            color=obj_col,
        )
        self._xx.append(obj["state"][0])
        self._yy.append(obj["state"][1])

        # plot future traj
        if self.__bool_fut_traj and len(fut_x) > 0:
            self._ax.plot(
                fut_x,
                fut_y,
                linestyle="-",
                color=obj_col,
                linewidth=self.linewidth,
            )
            self._ax.plot(
                fut_x[-1],
                fut_y[-1],
                linestyle="-",
                marker="x",
                markersize=self.markersize,
                linewidth=self.linewidth,
                color=obj_col,
            )
            self._xx.append(fut_x[-1])
            self._yy.append(fut_y[-1])

        return hist_t

    def plot_matched_object(self, sensorname, matched_obj, obj_col):
        """Plot matched objects."""
        self._set_label(
            "Matched: " + str(sensorname),
            self._sensor_marker[sensorname],
        )

        if "yaw" in matched_obj["keys"]:
            edges_glob = rotate_loc_glob(
                self.rectangle,
                matched_obj["state"][2],
                matrix=False,
            ) + np.expand_dims(
                np.array(matched_obj["state"][:2]),
                axis=1,
            )
            poly_1 = Polygon(edges_glob.T)
            _x, _y = poly_1.exterior.xy
            self._ax.plot(
                _x,
                _y,
                linewidth=self.linewidth,
                linestyle="dashed",
                color=obj_col,
            )

        # Plot matched object with object color
        self._ax.plot(
            matched_obj["state"][0],
            matched_obj["state"][1],
            marker=self._sensor_marker[sensorname],
            mfc="none",
            color=obj_col,
            markersize=self.markersize,
            linewidth=self.linewidth,
        )
        self._xx.append(matched_obj["state"][0])
        self._yy.append(matched_obj["state"][1])

    def plot_pred(self, pred_dict, obj_id, obj_col):
        """Plot prediction logs."""
        if pred_dict is not None and obj_id in pred_dict:
            self._set_label("Prediction", ".")
            if "x_rail" in pred_dict[obj_id]:
                self._ax.plot(
                    pred_dict[obj_id]["x_rail"],
                    pred_dict[obj_id]["y_rail"],
                    linestyle="None",
                    marker=".",
                    markersize=self.markersize,
                    linewidth=self.linewidth,
                    color=obj_col,
                )
                self._xx.append(pred_dict[obj_id]["x_rail"][-1])
                self._yy.append(pred_dict[obj_id]["y_rail"][-1])
            else:
                self._ax.plot(
                    pred_dict[obj_id]["x"],
                    pred_dict[obj_id]["y"],
                    linestyle="None",
                    marker=".",
                    markersize=self.markersize,
                    linewidth=self.linewidth,
                    color=obj_col,
                )
                if isinstance(pred_dict[obj_id]["x"], float):
                    self._xx.append(pred_dict[obj_id]["x"])
                    self._yy.append(pred_dict[obj_id]["y"])
                else:
                    self._xx.append(pred_dict[obj_id]["x"][-1])
                    self._yy.append(pred_dict[obj_id]["y"][-1])

    def plot_filter_log(
        self,
        filter_log,
        obj_id,
        _sensorname_list,
        obj_col,
        object_dict,
        _u,
        ego_state,
        obj,
        _dt,
    ):
        """Plot logs of filter."""
        _prior_list = []
        _posterior_list = []
        if filter_log[obj_id][0][0] is not None:
            num_filter_steps = len(filter_log[obj_id])
            last_meas_idx = [
                nn
                for nn in range(num_filter_steps)
                if filter_log[obj_id][nn][6] is not None
            ][-1]
            for _n in range(num_filter_steps):
                # Index 0: Prior is available, i.e. Measurement suceeded
                if filter_log[obj_id][_n][0] is not None:
                    # Plot prior, only possible, if not a new object
                    x_prior = filter_log[obj_id][_n][0]
                    if num_filter_steps > 1:
                        if _sensorname_list is not None and _n < len(_sensorname_list):
                            _prior_list.append(_sensorname_list[_n])

                        if self.__bool_filter_prior:
                            self._set_label("Prior", "s", markeredge="b")
                            self._ax.plot(
                                x_prior[0],
                                x_prior[1],
                                "s",
                                color=obj_col,
                                markeredgecolor="b",
                            )
                            self._xx.append(x_prior[0])
                            self._yy.append(x_prior[1])

                            self._set_label("P, Prior", "--", dot=False)
                            _p_prior = np.array(filter_log[obj_id][_n][1])
                            confidence_ellipse(
                                x_prior,
                                _p_prior,
                                ax=self._ax,
                                n_std=self.n_std,
                                linestyle="--",
                                edgecolor=obj_col,
                            )

                # Index 2: Delay compensated position
                if self.__bool_filter_delay and filter_log[obj_id][_n][2] is not None:
                    # Delayed State Update
                    self._set_label("Posterior (Delayed)", "+", markeredge="b")
                    x_hist_post = filter_log[obj_id][_n][2]
                    self._ax.plot(
                        x_hist_post[0],
                        x_hist_post[1],
                        "+",
                        color=obj_col,
                        markeredgecolor="b",
                    )
                    self._xx.append(x_hist_post[0])
                    self._yy.append(x_hist_post[1])

                    # Delayed Covariance Update
                    self._set_label("P, Delayed", "-.", dot=False)
                    _p_hist_post = np.array(filter_log[obj_id][_n][3])
                    confidence_ellipse(
                        x_hist_post,
                        _p_hist_post,
                        ax=self._ax,
                        n_std=self.n_std,
                        linestyle="-.",
                        edgecolor=obj_col,
                    )

                # Plot processed object list after filter.update-step
                _sigx = -1.0
                _sigy = -1.0
                rho = 0.0
                _p_norm = -1.0
                resx, resy, resyaw, resv = (0.0, 0.0, 0.0, 0.0)
                if obj_id in object_dict:
                    # Plot only final updated position (in case of multiple sensor inputs)
                    if _n == num_filter_steps - 1 and self.__bool_filter_post:
                        self._set_label("Posterior", "x", markeredge="b")
                        self._ax.plot(
                            object_dict[str(obj_id)]["state"][0],
                            object_dict[str(obj_id)]["state"][1],
                            "x",
                            color=obj_col,
                        )
                        self._xx.append(object_dict[str(obj_id)]["state"][0])
                        self._yy.append(object_dict[str(obj_id)]["state"][1])

                    # Plot only final updated position (in case of multiple sensor inputs)
                    _p_post = np.array(filter_log[obj_id][_n][5])
                    if num_filter_steps > 1:
                        if _sensorname_list is not None and _n < len(_sensorname_list):
                            _posterior_list.append(
                                [_sensorname_list[_n], np.linalg.norm(_p_post)]
                            )

                    if self.__bool_filter_post:
                        self._set_label("P, Posterior", "-", dot=False)
                        confidence_ellipse(
                            object_dict[str(obj_id)]["state"],
                            _p_post,
                            ax=self._ax,
                            n_std=self.n_std,
                            linestyle="-",
                            edgecolor=obj_col,
                        )
                    if _n == last_meas_idx:
                        _sigx = np.sqrt(_p_post[0, 0])
                        _sigy = np.sqrt(_p_post[1, 1])
                        if min(_sigx, _sigy) == 0:
                            rho = 0.0
                        else:
                            rho = _p_post[1, 0] / _sigx / _sigy
                        _p_norm = np.linalg.norm(_p_post)
                        n_meas = min(len(filter_log[obj_id][_n][6]), 3)
                        if n_meas == 3 and (
                            "yaw"
                            in self.params["ACTIVE_SENSORS"][
                                filter_log[obj_id][_n][7][1]
                            ]["meas_vars"]
                        ):
                            resx, resy, resyaw = filter_log[obj_id][_n][6][:n_meas]
                        elif n_meas == 3 and (
                            "v"
                            in self.params["ACTIVE_SENSORS"][
                                filter_log[obj_id][_n][7][1]
                            ]["meas_vars"]
                        ):
                            resx, resy, resv = filter_log[obj_id][_n][6][:n_meas]
                        else:
                            resx, resy = filter_log[obj_id][_n][6][:n_meas]

            self.obj_center_points_frame["Prior"][_u] = _prior_list
            self.obj_center_points_frame["Post"][_u] = _posterior_list
            self.obj_center_points_frame["cov"][_u] = [
                "{:.02f}m".format(_sigx),
                "{:.02f}m".format(_sigy),
                "{:.02f}m".format(rho),
            ]
            self.obj_center_points_frame["delta"][_u] = [
                "{:.02f}m".format(resx),
                "{:.02f}m".format(resy),
                "{:.02f}°".format(resyaw / np.pi * 180.0),
                "{:.02f}m".format(resv),
            ]
            self.obj_center_points_frame["p"][_u] = _p_norm
            if obj_id != "ego" and bool(ego_state) and "t" in ego_state:
                self.obj_center_points_frame["t_rel"][_u] = int(
                    (obj["t"] - ego_state["t"]) / 1e6
                )
            if _dt is not None:
                self.obj_center_points_frame["dt"][_u] = int(_dt * 1000)


def get_new_match_index(
    obj, obj_id: str, detected_objects: list, match_dict_sens: dict
):
    """Get index of net matched object.

    Find in new_object_list the right index for this detection
    index_new doesn"t fit, due to changes of the list during run time
    like check out of bounds...

    Args:
        obj (str): ID of object to match
        detected_objects (list): New detected objects
        match_dict_sens (dict): Dict of matches per sensor modality

    Returns:
        int: Index of match
    """
    index_new = next(
        (mms[0] for mms in match_dict_sens if mms[1] == int(obj_id)),
        None,
    )
    if index_new is not None:
        eucl_dist = []
        for find_obj in detected_objects:
            eucl_dist.append(
                np.linalg.norm(
                    np.array(obj["state"][:2]) - np.array(find_obj["state"][:2])
                )
            )
        return eucl_dist.index(min(eucl_dist))

    return None


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--n_obs", type=int, default=np.inf)
    parser.add_argument("--states", default=False, action="store_true")
    parser.add_argument("--ego_state", default=False, action="store_true")
    parser.add_argument("--filter", default=False, action="store_true")
    parser.add_argument("--mis_num_obj", default=False, action="store_true")
    parser.add_argument("--eval", default=False, action="store_true")
    parser.add_argument("--save_file_name", type=str, default=None)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--show_sample", default=False, action="store_true")
    parser.add_argument("--show_filtered", default=False, action="store_true")
    args = parser.parse_args()
    args.is_main = True

    LogViszTracking = ViszTracking(args=args)
    _ = LogViszTracking()
