"""Show Tracking Logs."""
import os
import sys
import argparse
import pickle
import copy

import matplotlib.pyplot as plt

TUM_BLUE = (0 / 255, 101 / 255, 189 / 255)
MIX_NET_COL = TUM_BLUE
TUM_ORAN = (227 / 255, 114 / 255, 34 / 255)
INDY_NET_COL = TUM_ORAN
HIST_COL = "black"
HIST_LS = "solid"
GT_COL = "black"
GT_LS = "dashed"
BOUND_COL = (204 / 255, 204 / 255, 204 / 255)

COLUMNWIDTH = 3.5
PAGEWIDTH = COLUMNWIDTH * 2.0
FONTSIZE = 8

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import numpy as np

REPO_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(REPO_PATH)

from tracking.tracking.utils.geometry import rotate_glob_loc, pi_range
from tools.visualize_logfiles import ViszTracking


def update_matplotlib():
    """Update matplotlib font style."""
    plt.rcParams.update(
        {
            "font.size": FONTSIZE,
            "font.family": "Times New Roman",
            "text.usetex": True,
            "figure.autolayout": True,
            "xtick.labelsize": FONTSIZE * 1.0,
            "ytick.labelsize": FONTSIZE * 1.0,
        }
    )


class Evaluation(ViszTracking):
    """Evaluation class of the tracking module.

    Args:
        ViszTracking (class): Inheritance
    """

    def __init__(self, args):
        """Initialize class.

        Args:
            args (namespace): Arguments to set up function call.
        """
        # add args that are not used
        args.n_obs = np.inf
        args.states = False
        args.ego_state = False
        args.filter = False
        args.mis_num_obj = False
        args.show_sample = False
        args.is_main = False
        if not args.results_save_path:
            if args.load_path:
                args.results_save_path = args.load_path
            else:
                args.results_save_path = "evaluation/results"
            print("set results_save_path to default: {}".format(args.results_save_path))

        self.args = args

        if not args.load_path:
            super().__init__(args=args)

        self.show_plt = args.show_plt
        self.save_file_name = args.save_file_name
        self.load_path = args.load_path
        self.results_save_path = args.results_save_path
        self.count_dir = None

        # evaluation parameters
        self.min_ego_vel = 28.0  # pit speed is 25 mps
        self.min_obj_vel = 10
        self.min_obs_time_s = 0.01
        self.min_valid_measures = {
            "lidar_cluster": 1,
            "radar": 1,
            "total": 3,
        }
        self.velocity_bounds = [40.0, 50.0, np.inf]
        self.long_pos_bounds = [-0, 50, np.inf]
        self.t_bounds = [1.0, 2.0, 3.0]
        self.t_obs = 10.0

        self.num_true_positives = 17
        self.precision_baseline = 0.5

        self.add_yaw = True
        self.add_vel = True

        self.eval_residuum = {}
        self.timer_dict = {}
        self.meas_cov_dict = {}

        # properties to visualize sensor data:
        self.sensor_visz_dict = {
            "lidar_cluster": {"col": TUM_BLUE, "name": "LiDAR"},
            "radar": {"col": TUM_ORAN, "name": "RADAR"},
            "lidar": {"col": "k", "name": "PRCNN"},
        }

        # entry: key, y_label, title, bool(y_lim starts at zero)
        self.violin_items = [
            # ["ego_vel", "$v$ in m/s", "$v_{\mathrm{ego}}$", True],
            ["obj_vel", "$v_{\mathrm{}}$ in m/s", "$v_{\mathrm{obj}}$", True],
            ["rel_vel", "$v$ in m/s", "$v_{\mathrm{rel}}$", False],
            ["dist", "$d$ in m", "$d_{\mathrm{obj}}$", True],
            ["x_meas", "$x$ in m", "$x_{\mathrm{loc}}$", False],
            [
                "num_measures",
                "$n$",
                "$n_{\mathrm{det}}$",
                True,
            ],
        ]

        if not os.path.exists(args.results_save_path):
            os.makedirs(args.results_save_path)
            print("Created dir {}".format(args.results_save_path))

        if not args.print_terminal:
            self.set_stdout_to_file()

    def set_stdout_to_file(self):
        """Set stdout to file, i.e. print to file."""
        self.orig_stdout = sys.stdout
        pref = self.get_save_name_w_timestamp()
        self.f = open(
            os.path.join(self.args.results_save_path, pref + "_stats.txt"),
            "w",
        )
        sys.stdout = self.f

    def set_stdout_to_sys_out(self):
        """Set stdout back to sys out."""
        sys.stdout = self.orig_stdout
        self.f.close()

    def is_ego_in_pit(self, obj_idx_list):
        """Determine if ego is in pit for all indices that a given object is stored."""
        return max(
            [self.log_params["ego_state"][idx]["in_pit"] for idx in obj_idx_list]
        )

    def get_relevant_objects(self):
        """Get all relevant objects to consider for tracking evaluation."""
        t_log_start_ns = self.log_params["time_ns"][0]

        for obj_id, _ in self.count_obj_ids:
            obj_idx = list(self.obj_items[obj_id]["object_states"].keys())
            t_0_rel_ns = self.obj_items[obj_id]["t_dict"][obj_idx[0]] - t_log_start_ns
            t_end_rel_ns = (
                self.obj_items[obj_id]["t_dict"][obj_idx[-1]] - t_log_start_ns
            )
            t_obs = (t_end_rel_ns - t_0_rel_ns) / 1e9

            ego_vel_obj = [
                self.obj_items["ego"]["object_states"][idx][3] for idx in obj_idx
            ]
            if np.mean(ego_vel_obj) < self.min_ego_vel:
                continue

            obj_vel_list = [
                state[3] for state in self.obj_items[obj_id]["object_states"].values()
            ]
            if np.mean(obj_vel_list) < self.min_obj_vel:
                continue

            if t_obs < self.min_obs_time_s:
                continue

            if self.is_ego_in_pit(obj_idx_list=obj_idx):
                continue

            # get covariance and residuums
            (
                counter,
                p_post_dict,
                measure_res_dict,
                sensor_meas_dict,
                inpt_yaws,
                nis_dict,
            ) = self.eval_filter_performance(obj_id=obj_id)
            if counter is None:
                continue

            log_idx = counter[1:]

            # initialize evaluation_dict
            eval_dict = self.get_eval_dict_template(obj_id, log_idx)

            # iterate through measures per sensor
            any_valid = False
            num_valids = []
            for str_sens, measure_res in measure_res_dict.items():
                if measure_res == []:
                    continue
                # get local residuen
                p_post_glob = p_post_dict[str_sens]
                (
                    x_delta,
                    y_delta,
                    yaw_delta,
                    v_delta,
                    p_post_loc,
                ) = self.get_filter_errors(
                    measure_res,
                    inpt_yaws,
                    str_sens,
                    p_post_glob,
                )

                # get valid indices
                eval_dict["valid_idx"][str_sens] = [
                    idx for idx in range(len(x_delta)) if not np.isnan(x_delta[idx])
                ]
                eval_dict["valid_t"][str_sens] = [
                    eval_dict["obj_t"][kk] for kk in eval_dict["valid_idx"][str_sens]
                ]

                # check if there are enough valid values
                num_valids.append(len(eval_dict["valid_idx"][str_sens]))
                str_list = []
                if num_valids[-1] < self.min_valid_measures[str_sens]:
                    str_list.append(
                        "low observations: sensor = {}, n = {}".format(
                            str_sens, num_valids[-1]
                        )
                    )
                    continue
                any_valid = True
                eval_dict["sensors"].append(str_sens)

                # get local measurement
                # check if yaw is also measured
                eval_dict["is_yaw_measured"] = (
                    sum(~np.isnan(yaw_delta)) >= self.min_valid_measures[str_sens]
                )
                eval_dict["is_v_measured"] = (
                    sum(~np.isnan(v_delta)) >= self.min_valid_measures[str_sens]
                )

                local_states = self.get_local_positions(
                    global_vals=sensor_meas_dict[str_sens],
                    log_idx=log_idx,
                    eval_dict=eval_dict,
                )

                # write measurements to eval dict
                self.unzip_local_states(local_states, eval_dict, str_sens)

                eval_dict["dist"][str_sens] = [
                    np.linalg.norm(loc_p[:2]) for loc_p in local_states
                ]

                self.get_locations_seen(eval_dict, str_sens)

                # get residuen
                eval_dict["x_res"][str_sens] = x_delta
                eval_dict["y_res"][str_sens] = y_delta
                eval_dict["yaw_res"][str_sens] = yaw_delta
                eval_dict["v_res"][str_sens] = v_delta

                # get local coovariances
                (
                    eval_dict["x_cov"][str_sens],
                    eval_dict["y_cov"][str_sens],
                    eval_dict["yaw_cov"][str_sens],
                    eval_dict["v_cov"][str_sens],
                    eval_dict["dyaw_cov"][str_sens],
                    eval_dict["a_cov"][str_sens],
                ) = list(
                    zip(
                        *[
                            np.diag(p_p)
                            if not np.isnan(p_p).any()
                            else np.full([6], np.nan)
                            for p_p in p_post_loc
                        ]
                    )
                )

            if not any_valid:
                continue

            if sum(num_valids) < self.min_valid_measures["total"]:
                continue

            eval_dict["nis"] = nis_dict

            self.get_sensors_seen(eval_dict)

            self.eval_residuum[obj_id] = eval_dict

            self.print_info(
                obj_id,
                t_obs,
                t_0_rel_ns,
                str_list,
            )

    def process_logs(self):
        """Process log files from tracking module."""
        # recover filter params
        self.measure_covs_dict = {
            sensor_name: np.diag([std_dev**2 for std_dev in specif["std_dev"]])
            for sensor_name, specif in self.params["ACTIVE_SENSORS"].items()
        }

        # get tracked objects
        self.get_relevant_objects()

        # check number of evaluated objects
        self.update_eval_length()

        # evaluate timing: sensor delay and calculation times
        (
            self.timer_dict["t_delay_ns"],
            self.timer_dict["t_cycle_ns"],
        ) = self.get_timing_stats()

        if self.results_save_path:
            self.save_eval_data()

    def get_timing_stats(self):
        """Determine sensor delay and node calculation time."""
        delay_dict = {}
        cycle_time_list = []
        iter_zip = zip(
            self.log_params["tracking_input"],
            self.log_params["object_dict"],
            self.log_params["cycle_time_ns"],
        )

        for tracking_input, obj_dict, cycle_time_ns in iter_zip:
            if tracking_input is None:
                continue
            for sensor_str, (obj_list, t_det_ns) in tracking_input:
                # consider only delay compensation if really conducted, i.e. objects detected
                if not bool(obj_list):
                    continue

                if not bool(obj_dict):
                    raise LookupError("tracking input received but not processed")

                # log sensor delay
                if sensor_str not in delay_dict:
                    delay_dict[sensor_str] = []
                t_track_ns = list(obj_dict.values())[0]["t"]
                delay_dict[sensor_str].append(t_track_ns - t_det_ns)

                # log calculation time
                cycle_time_list.append(cycle_time_ns)

        return delay_dict, cycle_time_list

    def eval_delay_stats(self):
        """Evaluate perception delay from all tracked objects."""
        delay_dict = self.merge_dicts(key_str="delay")
        if not bool(delay_dict):
            return

        # print some stats
        print(
            "Sensor Delay: \n"
            + "\n".join(
                [
                    "{}: n = {:d}, mean = {:.02f} ms, med = {:.02f} ms, q10 = {:.02f} ms, q90 = {:.02f} ms".format(
                        sensor_str,
                        len(v_in),
                        np.mean(v_in) / 1e6,
                        np.median(v_in) / 1e6,
                        np.quantile(v_in, 0.1) / 1e6,
                        np.quantile(v_in, 0.9) / 1e6,
                    )
                    for sensor_str, v_in in delay_dict.items()
                ]
            )
        )

        if len(delay_dict) == 2:
            vi = ["dummy", "$t_{\mathrm{}}$ in ms", "$t_{\mathrm{delay}}$", True]
            # visualize delay dict
            _, ax = plt.subplots(nrows=1, ncols=1, figsize=(COLUMNWIDTH / 2, 2))
            n_in = []
            sensor_str_list = list(delay_dict.keys())
            for idx, (sensor_str, v_in) in enumerate(delay_dict.items()):
                col = self.sensor_visz_dict[sensor_str]["col"]
                if idx == 0:
                    side = "left"
                else:
                    side = "right"
                n_in.append(len(v_in))
                self.get_pretty_violin(
                    ax,
                    v_in=np.array(v_in) / 1e6,
                    vi=vi,
                    col=col,
                    plot_pairs=side,
                    n_in=n_in,
                )

                title = "Delay stats ms {}".format(
                    self.sensor_visz_dict[sensor_str_list[0]]["name"]
                    + " and "
                    + self.sensor_visz_dict[sensor_str_list[1]]["name"]
                )

            ax.set_ylim(0.0)

            plt.tight_layout()
            # fig.suptitle(title)
            if self.results_save_path:
                file_name = title.lower().replace(" ", "_") + ".pdf"
                self.save_plt(file_name=file_name)

            if self.show_plt:
                plt.show()

        else:
            pass

    def merge_dicts(self, key_str: str, is_dict=True):
        """Merge multiple evaluation dicts from single rosbag recordings.

        _extended_summary_

        Args:
            key_str (str): key string to be merged, valid options are 'cycle' and 'delay'
            is_dict (bool, optional): If true dicts are merged.

        Returns:
            Instance: Returns merged items.
        """
        if is_dict:
            merge_instance = {}
        else:
            merge_instance = []
        for key, val in self.timer_dict.items():
            if key_str not in key:
                continue
            if is_dict:
                for sensor_str, delay_list in val.items():
                    if sensor_str not in merge_instance:
                        merge_instance[sensor_str] = []
                    merge_instance[sensor_str] += delay_list
            else:
                merge_instance += val
        return merge_instance

    def eval_nis_dist(self, nis_ctr, quantile=0.95):
        """Evaluate overall NIS-value (Normalized Innovation Error) form all tracked objects within a given quantile."""
        if len(nis_ctr) == 2:
            vi = ["dummy", "$\mathrm{NIS}$ ", "$\mathrm{NIS}$", True]
            # visualize delay dict
            _, ax = plt.subplots(nrows=1, ncols=1, figsize=(COLUMNWIDTH / 2, 2))
            n_in = []
            sensor_str_list = list(nis_ctr.keys())
            for idx, (sensor_str, v_in) in enumerate(nis_ctr.items()):
                col = self.sensor_visz_dict[sensor_str]["col"]
                if idx == 0:
                    side = "left"
                else:
                    side = "right"
                v_in_filt = np.array(v_in)
                v_in_filt = v_in_filt[~np.isnan(v_in_filt)]
                v_in_filt = v_in_filt[v_in_filt < np.quantile(v_in_filt, quantile)]
                n_in.append(len(v_in_filt))

                # frequency, bins = np.histogram(x, bins=10, range=[0, 100])
                # plt.hist(v_in_filt, bins=20)
                # plt.show()

                self.get_pretty_violin(
                    ax,
                    v_in=v_in_filt,
                    vi=vi,
                    col=col,
                    plot_pairs=side,
                    n_in=n_in,
                )

                title = "NIS stats {}".format(
                    self.sensor_visz_dict[sensor_str_list[0]]["name"]
                    + " and "
                    + self.sensor_visz_dict[sensor_str_list[1]]["name"]
                )

            ax.set_ylim(0.0)

            plt.tight_layout()
            # fig.suptitle(title)
            if self.results_save_path:
                file_name = title.lower().replace(" ", "_") + ".pdf"
                self.save_plt(file_name=file_name)

            if self.show_plt:
                plt.show()

        else:
            pass

    def eval_calc_time(self):
        """Evaluate calculation time of module call from logs."""
        cycle_time_dict = self.merge_dicts(key_str="cycle", is_dict=False)
        if not bool(cycle_time_dict):
            return

        t_cycle_ms = np.array(cycle_time_dict) / 1e6

        # print some stats
        print(
            "Average calculation time: n = {:d}, mean = {:.02f} ms, min = {:.02f} ms, "
            "max = {:.02f} ms, med = {:.02f} ms, q10 = {:.02f} ms, q90 = {:.02f} ms".format(
                len(t_cycle_ms),
                np.mean(t_cycle_ms),
                np.min(t_cycle_ms),
                np.max(t_cycle_ms),
                np.median(t_cycle_ms),
                np.quantile(t_cycle_ms, 0.1),
                np.quantile(t_cycle_ms, 0.9),
            )
        )

        vi = ["dummy", "$t_{\mathrm{}}$ in ms", "$t_{\mathrm{calc}}$", True]
        # visualize delay dict
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(COLUMNWIDTH / 2, 2))
        col = TUM_BLUE
        self.get_pretty_violin(
            ax,
            v_in=t_cycle_ms,
            vi=vi,
            col=col,
        )

        title = "Calc stats ms"

        ax.set_ylim(0.0)

        plt.tight_layout()
        # fig.suptitle(title)
        if self.results_save_path:
            file_name = title.lower().replace(" ", "_") + ".pdf"
            self.save_plt(file_name=file_name)

        if self.show_plt:
            plt.show()

    def print_info(
        self,
        obj_id: str,
        t_obs: int,
        t_0_rel_ns: int,
        str_list: list,
    ):
        """Print information (tracking stats) about a tracked object."""
        print(
            "\n\n+++++ EVALUATION OBJECT, No. {} +++++".format(len(self.eval_residuum))
        )
        print(
            "obj_id = {}, t_obs = {} s, t_0 = {:.02f} s".format(
                obj_id,
                t_obs,
                t_0_rel_ns / 1e9,
            )
        )

        # print short observations if any
        if str_list:
            _ = [print(ss) for ss in str_list]

        # print number of measures
        print(
            "Num valid measures: "
            + ", ".join(
                [
                    "{} = {}".format(str_sens, len(valid_idx))
                    for str_sens, valid_idx in self.eval_residuum[obj_id][
                        "valid_idx"
                    ].items()
                ]
            )
        )

        # prind speed stats
        def print_vel_stats(pref_str, obj_id):
            vel_key = pref_str + "_vel"
            print(
                vel_key
                + ": mean = {:.02f} m/s, std = {:.02f} m/s".format(
                    np.mean(self.eval_residuum[obj_id][vel_key]),
                    np.std(self.eval_residuum[obj_id][vel_key]),
                )
            )

        # print speed stats for ego, object and relativ speed
        print_vel_stats(pref_str="ego", obj_id=obj_id)
        print_vel_stats(pref_str="obj", obj_id=obj_id)
        print_vel_stats(pref_str="rel", obj_id=obj_id)

        # print first time detected
        def print_first_last(pref_str, obj_id):
            seen_sens_str = pref_str + "_seen_sensor"
            seen_loc_str = pref_str + "_seen_location"
            seen_dist_m_str = pref_str + "_seen_dist_m"
            if pref_str == "first":
                idx_ = 0
            else:
                idx_ = -1

            print(
                pref_str
                + " detection overall: "
                + ", ".join(
                    [str(fs) for fs in self.eval_residuum[obj_id][seen_sens_str]]
                )
            )
            for sensor_str, loc in self.eval_residuum[obj_id][seen_loc_str].items():
                print(
                    "First detection: {}, location: {}, distance = {:.02f} m, t_valid_{} = {} s".format(
                        sensor_str,
                        loc,
                        self.eval_residuum[obj_id][seen_dist_m_str][sensor_str],
                        pref_str,
                        self.eval_residuum[obj_id]["valid_t"][sensor_str][idx_],
                    )
                )

        print_first_last(pref_str="first", obj_id=obj_id)
        print_first_last(pref_str="last", obj_id=obj_id)

        # detection ranges
        for sensor_str, dist_list in self.eval_residuum[obj_id]["dist"].items():
            print(
                "Detection range in m: sensor : "
                "{}, min = {:.02f}, mean = {:.02f}, med = {:.02f}, max = {:.02f}".format(
                    sensor_str,
                    np.nanmin(dist_list),
                    np.nanmean(dist_list),
                    np.nanmedian(dist_list),
                    np.nanmax(dist_list),
                )
            )

    def get_local_positions(self, global_vals, log_idx, eval_dict):
        """Get local positions from global coordinate system."""
        local_states = [np.full([4], np.nan) for _ in range(len(log_idx))]
        for (
            j,
            idx,
        ) in enumerate(log_idx):
            sensor_meas = np.array(global_vals[j])
            if np.isnan(sensor_meas).all():
                continue

            ego_state = np.array(self.obj_items["ego"]["object_states"][idx][:4])
            local_states[j][:2] = rotate_glob_loc(
                global_matrix=sensor_meas[:2] - ego_state[:2],
                rot_angle=ego_state[2],
                matrix=False,
            )
            local_states[j][0] += eval_dict["rear_ax_geoc_m"]

            if eval_dict["is_yaw_measured"]:
                local_states[j][2] = pi_range(sensor_meas[2] - ego_state[2])

            if eval_dict["is_v_measured"]:
                local_states[j][3] = sensor_meas[3] - ego_state[3]

        return local_states

    @staticmethod
    def get_sensors_seen(eval_dict):
        """Determine all sensors modalities which detected the object including location."""
        idx_list_min = [
            eval_dict["valid_idx"][str_sens][0] for str_sens in eval_dict["sensors"]
        ]
        min_sort_idx = np.argsort(idx_list_min)
        min_idx = np.min(idx_list_min)

        for s_idx in min_sort_idx:
            if idx_list_min[s_idx] > min_idx:
                break
            eval_dict["first_seen_sensor"].append(eval_dict["sensors"][s_idx])

        idx_list_max = [
            eval_dict["valid_idx"][str_sens][-1] for str_sens in eval_dict["sensors"]
        ]
        max_sort_idx = np.argsort(idx_list_max)[::-1]
        max_idx = np.max(idx_list_max)

        for s_idx in max_sort_idx:
            if idx_list_max[s_idx] < max_idx:
                break
            eval_dict["last_seen_sensor"].append(eval_dict["sensors"][s_idx])

    @staticmethod
    def get_locations_seen(eval_dict, str_sens):
        """Determine location where object was seen the first time for a specific sensor."""
        # check location of first time seen
        first_idx = eval_dict["valid_idx"][str_sens][0]
        if eval_dict["x_meas"][str_sens][first_idx] > 0.0:
            str_loc = "front"
        else:
            str_loc = "rear"
        eval_dict["first_seen_location"][str_sens] = str_loc
        eval_dict["first_seen_dist_m"][str_sens] = eval_dict["dist"][str_sens][
            first_idx
        ]

        # check location of last time seen
        last_idx = eval_dict["valid_idx"][str_sens][-1]
        if eval_dict["x_meas"][str_sens][last_idx] > 0.0:
            str_loc = "front"
        else:
            str_loc = "rear"
        eval_dict["last_seen_location"][str_sens] = str_loc
        eval_dict["last_seen_dist_m"][str_sens] = eval_dict["dist"][str_sens][last_idx]

    @staticmethod
    def unzip_local_states(local_states, eval_dict, str_sens):
        """Unzip local states of an object."""
        if len(local_states[0]) == 4:
            (
                eval_dict["x_meas"][str_sens],
                eval_dict["y_meas"][str_sens],
                eval_dict["yaw_meas"][str_sens],
                eval_dict["v_meas"][str_sens],
            ) = zip(*local_states)

        elif len(local_states[0]) == 3:
            (
                eval_dict["x_meas"][str_sens],
                eval_dict["y_meas"][str_sens],
                eval_dict["yaw_meas"][str_sens],
            ) = zip(*local_states)

        elif len(local_states[0]) == 2:
            (eval_dict["x_meas"][str_sens], eval_dict["y_meas"][str_sens]) = zip(
                *local_states
            )

    def get_eval_dict_template(self, obj_id, log_idx):
        """Get template of evaluation dict for a new tracked object."""
        return {
            "rear_ax_geoc_m": self.params["MISCS"]["rear_ax_geoc_m"],
            "is_yaw_measured": False,
            "is_v_measured": False,
            "sensors": [],
            "nis": {},
            "x_res": {},
            "y_res": {},
            "yaw_res": {},
            "v_res": {},
            "dist": {},
            "x_meas": {},
            "y_meas": {},
            "yaw_meas": {},
            "v_meas": {},
            "valid_idx": {},
            "valid_t": {},
            "x_cov": {},
            "y_cov": {},
            "yaw_cov": {},
            "v_cov": {},
            "dyaw_cov": {},
            "a_cov": {},
            "first_seen_sensor": [],
            "last_seen_sensor": [],
            "first_seen_location": {},
            "last_seen_location": {},
            "first_seen_dist_m": {},
            "last_seen_dist_m": {},
            "obj_idx": list(self.obj_items[obj_id]["object_states"].keys()),
            "obj_states": [
                self.obj_items[obj_id]["object_states"][ll] for ll in log_idx
            ],
            "obj_t": [
                (
                    self.obj_items[obj_id]["t_dict"][ll]
                    - self.obj_items[obj_id]["t_dict"][log_idx[0]]
                )
                / 1e9
                for ll in log_idx
            ],
            "obj_vel": [
                self.obj_items[obj_id]["object_states"][ll][3] for ll in log_idx
            ],
            "ego_vel": [
                self.obj_items["ego"]["object_states"][ll][3] for ll in log_idx
            ],
            "rel_vel": [
                (
                    self.obj_items[obj_id]["object_states"][ll][3]
                    - self.obj_items["ego"]["object_states"][ll][3]
                )
                for ll in log_idx
            ],
        }

    def get_save_name_w_timestamp(self):
        """Get save name of including timestamp."""
        if self.args.save_file_name:
            if self.args.timestamp:
                return self.save_file_name + "_" + self.args.timestamp
            return self.save_file_name
        if self.args.timestamp:
            return self.args.timestamp
        if self.args.load_path:
            return os.path.basename(self.args.load_path)
        if self.count_dir is None:
            self.count_dir = str(len(os.listdir(self.results_save_path)) + 1)
        return self.count_dir.zfill(4)

    def save_eval_data(self):
        """Save processed evaluation data to .pkl."""
        pref = self.get_save_name_w_timestamp()
        save_file_path = os.path.join(
            self.results_save_path,
            pref + ".pkl",
        )
        with open(save_file_path, "wb") as f:
            pickle.dump(
                (self.eval_residuum, self.timer_dict, self.measure_covs_dict), f
            )

        print("Save eval dict to {}".format(save_file_path))

    def load_eval_data(self):
        """Load evaluation data from former pre-processing.

        Raises:
            ValueError: Load path does not exist.
            ValueError: Load path is empty.
        """
        if not os.path.exists(self.load_path):
            raise ValueError("load path does not exist: {}".format(self.load_path))

        if len(os.listdir(self.load_path)) == 0:
            raise ValueError("load path is empty: {}".format(self.load_path))

        for j, files in enumerate(os.listdir(self.load_path)):
            pkl_file = os.path.join(
                self.load_path,
                files,
            )

            if not pkl_file.endswith(".pkl"):
                continue

            with open(pkl_file, "rb") as f:
                loaded_dict, timer_dict, meas_cov_dict = pickle.load(f)
                self.eval_residuum.update(
                    dict(
                        zip(
                            ["{}_".format(j) + ss for ss in loaded_dict.keys()],
                            loaded_dict.values(),
                        )
                    )
                )
                self.timer_dict.update(
                    dict(
                        zip(
                            ["{}_".format(j) + ss for ss in timer_dict.keys()],
                            timer_dict.values(),
                        )
                    )
                )
                self.meas_cov_dict.update(
                    dict(
                        zip(
                            ["{}_".format(j) + ss for ss in meas_cov_dict.keys()],
                            meas_cov_dict.values(),
                        )
                    )
                )

        self.update_eval_length()

    def update_eval_length(self):
        """Update length of evaluation dict."""
        print("\nNumber of evaluation objects = {}".format(len(self.eval_residuum)))

    @staticmethod
    def get_non_nan_list(in_array):
        """Filter NaN entries out of array."""
        return [val for val in in_array if not np.isnan(val)]

    @staticmethod
    def get_labels(vel_bounds, no_int=False):
        """Get label for bin plots.

        3 bins: smaller than, between, greater than.
        """
        x_tick_labels = []
        for j in range(len(vel_bounds)):

            if j == 0:
                if no_int:
                    x_tick_labels.append("$<{}$".format(vel_bounds[j]))
                else:
                    x_tick_labels.append("$<{}$".format(int(vel_bounds[j])))

                continue

            if j == len(vel_bounds) - 1 and vel_bounds[j] == np.inf:
                if no_int:
                    x_tick_labels.append("${}<$".format(vel_bounds[j - 1]))
                else:
                    x_tick_labels.append("${}<$".format(int(vel_bounds[j - 1])))
                continue

            if no_int:
                x_tick_labels.append("${}-{}$".format(vel_bounds[j - 1], vel_bounds[j]))
            else:
                x_tick_labels.append(
                    "${}-{}$".format(int(vel_bounds[j - 1]), int(vel_bounds[j]))
                )

        return x_tick_labels

    def prepare_data(self):
        """Prepare evaluation.

        Helper function to run only pre-processing to generate evaluation dict.
        """
        self.process_logs()
        return

    def get_eval_dict(self):
        """Get evaluation dict, which contains all log data and processed metrics.

        Either loaded from .pkl file or by processing log files.
        """
        if self.load_path:
            self.load_eval_data()
        else:
            self.process_logs()

    def main_call(self):
        """Call main function."""
        self.get_eval_dict()

        if not self.eval_residuum:
            print("No object to evaluate, run with other logs")
            return

        self.run_eval()

        if not args.print_terminal and args.save_file_name:
            self.set_stdout_to_sys_out()

        print(
            "All data can be found in {}/{}".format(
                self.args.results_save_path, self.args.save_file_name
            )
        )

    def check_for_valid_sensors(self):
        """Check for all valid sensors in given data logs.

        Returns:
            list: List of valid sensors.
        """
        return list(set(*zip(*[val["sensors"] for val in self.eval_residuum.values()])))

    def run_eval(self):
        """Run whole evaluation pipeline."""
        if args.show_filtered:
            # run visualization
            self.visualize_filtered()
            return

        update_matplotlib()

        # sensor_str_list = self.check_for_valid_sensors()
        sensor_str_list = ["lidar_cluster", "radar"]

        self.eval_delay_stats()
        self.eval_calc_time()

        # evaluate residuals for different timeslots
        self.eval_metrics(sensor_str_list)
        self.eval_metrics(sensor_str_list, t_max=2.0)
        self.eval_metrics(sensor_str_list, t_max=1.0)
        self.eval_metrics(sensor_str_list, t_max=0.5)

        self.eval_res_over_time(sensor_str_list)

        # violin plots of data stats
        plt.rcParams.update({"ytick.labelsize": FONTSIZE * 0.75})
        self.eval_track_stats_paired(copy.deepcopy(sensor_str_list))
        # self.eval_track_stats(sensor_str_list)
        plt.rcParams.update({"ytick.labelsize": FONTSIZE * 1.0})

        # boxplots separated various criteria
        plt.rcParams.update({"ytick.labelsize": FONTSIZE * 0.75})
        plt.rcParams.update({"xtick.labelsize": FONTSIZE * 0.75})
        self.eval_per_vel(sensor_str_list)
        self.eval_per_obs_time(sensor_str_list)
        self.eval_per_long_pos(sensor_str_list)

        # visualization of transition behavior
        self.eval_transitions(t_obs=self.t_obs)

    def visualize_filtered(self):
        """Visualize filtered values."""
        pref = self.get_save_name_w_timestamp()
        self.filtered_ids += list(self.eval_residuum.keys())

        id_file_txt = os.path.join(self.args.results_save_path, pref + "_IDs.txt")
        with open(id_file_txt, "w") as f:
            f.write(
                "Filtered IDs by Occurence (n = {})\n".format(len(self.eval_residuum))
            )
            f.write("ID\t\tstart\t\tstop\n")
            f.write(
                "\n".join(
                    [
                        f"{key}\t\t"
                        + "\t\t".join(
                            [
                                str(eval_vals["obj_idx"][0]),
                                str(eval_vals["obj_idx"][-1]),
                            ]
                        )
                        for key, eval_vals in self.eval_residuum.items()
                    ]
                )
            )

        sorted_idx = np.argsort(
            [ss["obj_idx"][0] for ss in self.eval_residuum.values()]
        )
        iter_list = list(self.eval_residuum.items())

        with open(id_file_txt, "a") as f:
            f.write("")
            f.write(
                "\n\nFiltered IDs chronologically (n = {})\n".format(
                    len(self.eval_residuum)
                )
            )
            f.write("ID\t\tstart\t\tstop\n")
            f.write(
                "\n".join(
                    [
                        f"{iter_list[idx][0]}\t\t"
                        + "\t\t".join(
                            [
                                str(iter_list[idx][1]["obj_idx"][0]),
                                str(iter_list[idx][1]["obj_idx"][-1]),
                            ]
                        )
                        for idx in sorted_idx
                    ]
                )
            )

        # visualize the logs
        self.__call__()

    def eval_res_over_time(self, sensor_str_list, min_vals=0):
        """Evaluate residuals on observation time bins."""
        x_label = "$t$ in s"

        step_size_s = 0.1
        time_range_s = 2.1
        self.time_bound = np.arange(0, time_range_s, step_size_s) + step_size_s / 2.0

        _, y_label_list, in_containers, _ = self.fill_containers(
            eval_key="valid_t",
            bounds_list=self.time_bound,
            sensor_str_list=sensor_str_list,
            idx_res=1,
        )

        in_containers = self.reduce_container(in_containers)

        analyze_key_dict = [
            ("nanmean", None, TUM_BLUE, "solid", "Mean"),
            ("nanquantile", 0.25, TUM_BLUE, "dotted", "Q=0.25"),
            ("nanquantile", 0.75, TUM_BLUE, "dashed", "Q=0.75"),
        ]

        in_containers_zipped = []
        for ic in in_containers:
            for j in range(len(ic)):

                if len(in_containers_zipped) < j + 1:
                    in_containers_zipped.append([[np.nan] for _ in range(len(ic[j]))])
                for kk in range(len(ic[j])):
                    in_containers_zipped[j][kk] += ic[j][kk]

        _, ax = plt.subplots(
            nrows=len(in_containers_zipped), figsize=(len(in_containers[0]) * 3, 9)
        )

        for j, ic in enumerate(in_containers_zipped):
            for key, arg_add, col, ls, lab in analyze_key_dict:
                x_analyze_val = [
                    np.__dict__[key](ic_t, arg_add)
                    if len(ic_t) > min_vals and ~np.isnan(ic_t).all()
                    else np.nan
                    for ic_t in ic
                ]
                ax[j].plot(
                    self.time_bound, x_analyze_val, color=col, linestyle=ls, label=lab
                )
                ax[j].set_ylabel(y_label_list[j])
                ax[j].grid(True)
                ax[j].set_xlim([0, time_range_s - step_size_s])

            ax[0].legend(
                bbox_to_anchor=(0.5, 0.72, 0.48, 0.102),
                loc="lower left",
                ncol=len(analyze_key_dict),
                mode="expand",
                borderaxespad=0.0,
            )
            ax[-1].x_label = x_label

        if self.results_save_path:
            postf = self.get_post_fix_str(sensor_str_list)
            file_name = "res_over_time_" + postf + ".pdf"
            self.save_plt(file_name=file_name)

        if self.show_plt:
            plt.show()

    def save_plt(self, file_name: str, add_svg=True):
        """Save plot to .pdf, and additionally to .svg format."""
        plt_path = os.path.join(self.results_save_path, "plots")
        if not os.path.exists(plt_path):
            os.makedirs(plt_path)

        if file_name.endswith(".pdf"):
            plt.savefig(os.path.join(plt_path, file_name), format="pdf")

        if not add_svg:
            return

        file_name = file_name.replace(".pdf", ".svg")
        svg_path = os.path.join(plt_path, "svg")
        if not os.path.exists(svg_path):
            os.makedirs(svg_path)
        plt.savefig(os.path.join(svg_path, file_name), format="svg")

    def get_violin_data(self, str_sensor: str, violin_total_data: list):
        """Determine data for violine plots."""
        violin_data = {tup[0]: [] for tup in self.violin_items}
        for eval_dict in iter(self.eval_residuum.values()):
            if str_sensor not in eval_dict["sensors"]:
                continue

            for vi in self.violin_items:
                if vi[0] == "num_measures":
                    violin_data[vi[0]].append(len(eval_dict["valid_idx"][str_sensor]))
                else:
                    if isinstance(eval_dict[vi[0]], dict):
                        violin_data[vi[0]] += [
                            eval_dict[vi[0]][str_sensor][idx]
                            for idx in eval_dict["valid_idx"][str_sensor]
                        ]
                    else:
                        violin_data[vi[0]] += [
                            eval_dict[vi[0]][idx]
                            for idx in eval_dict["valid_idx"][str_sensor]
                        ]

        for vi in self.violin_items:
            violin_total_data[vi[0]] += violin_data[vi[0]]

        return violin_data

    def eval_track_stats_paired(self, sensor_str_list):
        """Eval data stats in violin plots, paired version."""
        if len(sensor_str_list) == 1:
            print("Duplicating single entry: {}".format(sensor_str_list))
            sensor_str_list.append(sensor_str_list[0])

        if len(sensor_str_list) != 2:
            print("paired evaluation not possible, need a pair of two sensors")
            return

        violin_total_data = {tup[0]: [] for tup in self.violin_items}

        _, ax = plt.subplots(
            nrows=1, ncols=len(self.violin_items), figsize=(PAGEWIDTH, 3)
        )
        n_p = []
        n_d = []
        for j in range(len(sensor_str_list)):
            if j == 0:
                side = "left"
            else:
                side = "right"
            str_sensor = sensor_str_list[j]
            violin_data = self.get_violin_data(
                str_sensor=str_sensor, violin_total_data=violin_total_data
            )
            # plot violins
            for idx, vi in enumerate(self.violin_items):
                v_in = violin_data[vi[0]]
                col = self.sensor_visz_dict[str_sensor]["col"]

                plt_det = False
                if idx == 0:
                    n_p.append(len(v_in))
                elif "$n" in vi[2]:
                    n_d.append(len(v_in))
                    plt_det = True

                if plt_det:
                    n_in = n_d
                else:
                    n_in = n_p
                self.get_pretty_violin(
                    ax[idx],
                    v_in=v_in,
                    vi=vi,
                    col=col,
                    plot_pairs=side,
                    n_in=n_in,
                )

                if vi[-1]:
                    ax[idx].set_ylim(0.0)

        title = "Data stats {}".format(
            self.sensor_visz_dict[sensor_str_list[0]]["name"]
            + " and "
            + self.sensor_visz_dict[sensor_str_list[1]]["name"]
        )

        plt.tight_layout()
        # fig.suptitle(title)
        if self.results_save_path:
            file_name = title.lower().replace(" ", "_") + ".pdf"
            self.save_plt(file_name=file_name)

        if self.show_plt:
            plt.show()

    def eval_track_stats(self, sensor_str_list):
        """Eval data stats in violin plots, single version."""
        violin_total_data = {tup[0]: [] for tup in self.violin_items}
        num_plots = len(sensor_str_list) + 1

        # get data stats visualized with violins
        for j in range(num_plots):
            fig, ax = plt.subplots(
                nrows=1, ncols=len(self.violin_items), figsize=(16, 9)
            )
            # get violion data from each sensor
            if j < num_plots - 1:
                str_sensor = sensor_str_list[j]
                violin_data = self.get_violin_data(
                    str_sensor=str_sensor, violin_total_data=violin_total_data
                )

                title = "Data stats {}".format(
                    self.sensor_visz_dict[str_sensor]["name"]
                )
            else:
                title = "Data stats total"

            # plot violins
            for idx, vi in enumerate(self.violin_items):
                if j < num_plots - 1:
                    v_in = violin_data[vi[0]]
                    col = self.sensor_visz_dict[str_sensor]["col"]
                else:
                    v_in = violin_total_data[vi[0]]
                    col = "g"

                self.get_pretty_violin(
                    ax[idx],
                    v_in=v_in,
                    vi=vi,
                    col=col,
                )

            # fig.suptitle(title)
            if self.results_save_path:
                file_name = title.lower().replace(" ", "_") + ".pdf"
                self.save_plt(file_name=file_name)

            if self.show_plt:
                plt.show()

    def get_pretty_violin(
        self,
        ax,
        v_in,
        vi,
        col,
        plot_pairs=False,
        showmedians=False,
        quantiles=[],
        n_in=[],
    ):
        """Create violin plot either unified or splitted."""
        violin_parts = ax.violinplot(
            v_in,
            showextrema=False,
            showmedians=showmedians,
            quantiles=quantiles,
        )
        ax.set_title(None)
        ax.set_ylabel(vi[1])
        ax.grid(True)
        ax.set_xticks([1])

        ax.set_xticklabels([vi[2]])
        if plot_pairs:
            if len(n_in) > 1:
                print(
                    vi[2]
                    + ",\n$n_{\mathrm{points}}$ = "
                    + ", ".join([str(nn) for nn in n_in])
                )
        else:
            print(vi[2] + "\n($n_{\mathrm{points}}$ = " + "{})".format(len(v_in)))

        for key, b in violin_parts.items():
            if isinstance(b, list):
                b = b[0]

            if plot_pairs:
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                if plot_pairs == "right":
                    # modify the paths to not go further right than the center
                    b.get_paths()[0].vertices[:, 0] = np.clip(
                        b.get_paths()[0].vertices[:, 0], m, np.inf
                    )
                elif plot_pairs == "left":
                    b.get_paths()[0].vertices[:, 0] = np.clip(
                        b.get_paths()[0].vertices[:, 0], -np.inf, m
                    )

            if key == "bodies":
                b.set_facecolor(col)
                b.set_edgecolor(col)
                b.set_alpha(1.0)
            else:
                b.set_edgecolor("k")
                b.set_linewidth(3)

    def plt_obj_states(self, eval_data, marker="x", t_obs=10.0, y_limits=None):
        """Plot object states."""
        plt_keys = ["x_res", "y_res", "yaw_res"]
        y_label_list = [
            "$x_{\mathrm{res}}$ in m",
            "$y_{\mathrm{res}}$ in m",
            "$\psi_{\mathrm{res}}$ in °",
            "$v_{\mathrm{res}}$ in m/s",
        ]
        sensor_list = list(eval_data["x_res"].keys())

        if (
            "radar" in sensor_list
            and eval_data["obj_t"][eval_data["valid_idx"]["radar"][0]] < t_obs
        ):
            plt_keys.append("v_res")

        _, axs = plt.subplots(nrows=len(plt_keys), ncols=1, figsize=(16, 9))

        plt_data = [eval_data[plt_k] for plt_k in plt_keys]
        for ss in sensor_list:
            for j in range(len(plt_data)):
                axs[j].plot(
                    eval_data["obj_t"],
                    plt_data[j][ss],
                    linestyle="None",
                    marker=marker,
                    color=self.sensor_visz_dict[ss]["col"],
                )
                axs[j].set_title(None)
                axs[j].set_ylabel(y_label_list[j])

                axs[j].grid(True)
                if j == len(axs) - 1:
                    axs[j].set_xlabel("$t$ in s")

                axs[j].set_xlim([0.0, t_obs])

                if y_limits and plt_keys[j] in y_limits:
                    axs[j].set_ylim(y_limits[plt_keys[j]])

        if self.results_save_path:
            file_name = "res_over_t_transition_vel_{}_mps.pdf".format(
                int(np.mean(eval_data["ego_vel"]))
            )
            self.save_plt(file_name=file_name)

        if self.show_plt:
            plt.show()

    @staticmethod
    def get_front_detection(idx_list, eval_res_list, t_obs):
        """Get information about front detection for a given object."""
        for vel_idx in idx_list:
            d_temp = eval_res_list[vel_idx]
            if "front" == d_temp["first_seen_location"][d_temp["first_seen_sensor"][0]]:

                if d_temp["obj_t"][-1] < t_obs:
                    continue

                return d_temp

        print("no front detection found")
        return eval_res_list[idx_list[0]]

    @staticmethod
    def get_alt_indices(vel_idx_sort):
        """Get alternating indices of a list."""
        mid_idx = int(len(vel_idx_sort) / 2)
        idx_list_alt = []
        idx_list_alt.append(vel_idx_sort[mid_idx])
        for j in range(mid_idx - 1):
            idx_list_alt.append(vel_idx_sort[mid_idx + (j + 1)])
            idx_list_alt.append(vel_idx_sort[mid_idx - (j + 1)])
        return idx_list_alt

    def eval_transitions(self, t_obs=10.0):
        """Evaluate transient behavior of tracking module.

        Args:
            t_obs (float, optional): Time span to investigate transitions. Defaults to 10.0.
        """
        eval_res_list = list(self.eval_residuum.values())
        vel_idx_sort = np.argsort(
            [np.mean(eval_res["ego_vel"]) for eval_res in eval_res_list]
        )

        vel_min_data = self.get_front_detection(
            idx_list=vel_idx_sort,
            eval_res_list=eval_res_list,
            t_obs=t_obs,
        )
        vel_max_data = self.get_front_detection(
            idx_list=vel_idx_sort[::-1],
            eval_res_list=eval_res_list,
            t_obs=t_obs,
        )
        idx_list_alt = self.get_alt_indices(vel_idx_sort)
        vel_mid_data = self.get_front_detection(
            idx_list=idx_list_alt,
            eval_res_list=eval_res_list,
            t_obs=t_obs,
        )

        y_limits = self.get_limits(
            (vel_min_data, vel_max_data, vel_mid_data), t_obs=t_obs
        )
        self.plt_obj_states(vel_min_data, t_obs=t_obs, y_limits=y_limits)
        self.plt_obj_states(vel_max_data, t_obs=t_obs, y_limits=y_limits)
        self.plt_obj_states(vel_mid_data, t_obs=t_obs, y_limits=y_limits)

    @staticmethod
    def get_limits(
        data_tuple,
        t_obs: int,
        keys=("x_res", "y_res", "yaw_res", "v_res"),
        sensor_str=("lidar_cluster", "radar"),
        v_s=1.2,
    ):
        """Get ax limits for given plots config.

        Args:
            data_tuple (tuple): Tuple of evaluation values of one object, keys are sensor modalities
            t_obs (int): Observation time in ns
            keys (tuple, optional): Evaluation keys to get limits of.
                                    Defaults to ("x_res", "y_res", "yaw_res", "v_res").
            sensor_str (tuple, optional): Tuple of sensor string to investigate. Defaults to ("lidar_cluster", "radar").
            v_s (float, optional): Scale factor of detemined limits. Defaults to 1.2.

        Returns:
            dict: Dict with limits (min, max) of given keys
        """
        idx_val = [
            {
                str_sens: [
                    val_i
                    for val_i in date["valid_idx"][str_sens]
                    if date["obj_t"][val_i] < t_obs
                ]
                for str_sens in sensor_str
                if str_sens in date["valid_idx"]
            }
            for date in data_tuple
        ]
        a = [
            [
                (
                    np.nanmin(
                        np.concatenate(
                            [np.array([0.0])]
                            + [
                                np.array(date[key][str_sens])[idx_val[j][str_sens]]
                                for str_sens in sensor_str
                                if str_sens in date[key]
                            ]
                        )
                    ),
                    np.nanmax(
                        np.concatenate(
                            [np.array([0.0])]
                            + [
                                np.array(date[key][str_sens])[idx_val[j][str_sens]]
                                for str_sens in sensor_str
                                if str_sens in date[key]
                            ]
                        )
                    ),
                )
                for j, date in enumerate(data_tuple)
            ]
            for key in keys
        ]

        return dict(zip(keys, [(v_s * np.nanmin(ai), v_s * np.nanmax(ai)) for ai in a]))

    def eval_per_long_pos(self, sensor_str_list):
        """Evaluate residuals per longitudinal position.

        Args:
            sensor_str_list (list): list of sensor modalities
        """
        (
            x_tick_labels,
            y_label_list,
            in_containers,
            nis_container,
        ) = self.fill_containers(
            eval_key="x_meas",
            bounds_list=self.long_pos_bounds,
            sensor_str_list=sensor_str_list,
        )

        in_containers = self.reduce_container(in_containers)

        postf = self.get_post_fix_str(sensor_str_list)

        self._create_boxplots(
            in_containers,
            title=None,
            ax_titles=[None for _ in range(len(in_containers[0]))],
            x_label="$x_{\mathrm{loc}}$ in m",
            y_label_list=y_label_list,
            x_tick_labels=x_tick_labels,
            sensor_str_list=sensor_str_list,
            # y_lims=[(0.0, 22.0), (0.0, 11.0), (0.0, 22.0)],
            file_save_name="res_vs_long_pos_" + postf,
        )

        self._create_boxplots(
            nis_container,
            title=None,
            ax_titles=[None],
            x_label="$x_{\mathrm{loc}}$ in m",
            y_label_list=["$\mathrm{NIS}$"],
            x_tick_labels=x_tick_labels,
            sensor_str_list=sensor_str_list,
            # y_lims=[(0.0, 22.0), (0.0, 11.0), (0.0, 22.0)],
            file_save_name="nis_vs_long_pos_" + postf,
        )

    def eval_metrics(self, sensor_str_list, t_max=np.inf):
        """Evaluate tracking performance by residuals (res_cts) and NIS values."""
        nis_cts = {sensor_str: [] for sensor_str in sensor_str_list}
        x_res_cts = {sensor_str: [] for sensor_str in sensor_str_list}
        y_res_cts = {sensor_str: [] for sensor_str in sensor_str_list}
        yaw_res_cts = {sensor_str: [] for sensor_str in sensor_str_list}
        vel_res_cts = {sensor_str: [] for sensor_str in sensor_str_list}
        max_match_front_cts = {sensor_str: [] for sensor_str in sensor_str_list}
        max_match_rear_cts = {sensor_str: [] for sensor_str in sensor_str_list}
        max_lost_front_cts = {sensor_str: [] for sensor_str in sensor_str_list}
        max_lost_rear_cts = {sensor_str: [] for sensor_str in sensor_str_list}
        iter_res_list = iter(self.eval_residuum.values())

        for eval_res in iter_res_list:
            for sensor_str in sensor_str_list:
                if sensor_str not in eval_res["sensors"]:
                    continue

                if eval_res["first_seen_location"][sensor_str] == "front":
                    max_match_front_cts[sensor_str].append(
                        eval_res["first_seen_dist_m"][sensor_str]
                    )
                    max_lost_front_cts[sensor_str].append(
                        eval_res["last_seen_dist_m"][sensor_str]
                    )
                else:
                    max_match_rear_cts[sensor_str].append(
                        eval_res["first_seen_dist_m"][sensor_str]
                    )
                    max_lost_rear_cts[sensor_str].append(
                        eval_res["last_seen_dist_m"][sensor_str]
                    )

                last_idx = sum(np.array(eval_res["obj_t"]) <= t_max)
                nis_cts[sensor_str] += eval_res["nis"][sensor_str][:last_idx]
                x_res_cts[sensor_str] += eval_res["x_res"][sensor_str][:last_idx]
                y_res_cts[sensor_str] += eval_res["y_res"][sensor_str][:last_idx]
                yaw_res_cts[sensor_str] += eval_res["yaw_res"][sensor_str][:last_idx]
                vel_res_cts[sensor_str] += eval_res["v_res"][sensor_str][:last_idx]

        self.eval_nis_dist(nis_cts)

        if t_max == np.inf:
            print("\n\n+++++ OVERALL sensor range (max, mean) +++++")
            match_cts = max_match_front_cts
            lost_cts = max_lost_front_cts
            pref = "FRONT"
            _ = [
                self.print_sensor_range(pref, ss, match_cts, lost_cts)
                for ss in sensor_str_list
            ]

            match_cts = max_match_rear_cts
            lost_cts = max_lost_rear_cts
            pref = "REAR"
            _ = [
                self.print_sensor_range(pref, ss, match_cts, lost_cts)
                for ss in sensor_str_list
            ]

        print(
            "\n\n+++++ OVERALL residuum: (mean, std_dev, med, q95), t_obs = {} +++++".format(
                t_max
            )
        )
        one_line_str = ""
        one_line_str += self.print_stats("nis", nis_cts, "-")
        one_line_str += ", " + self.print_stats("x_res", x_res_cts)
        one_line_str += ", " + self.print_stats("y_res", y_res_cts)
        one_line_str += ", " + self.print_stats("yaw_res", yaw_res_cts, unit_in="°")
        one_line_str += ", " + self.print_stats("vel_res", vel_res_cts, unit_in="mps")
        key_str = "nis, x_res, y_res, yaw_res, vel_res"
        if t_max == np.inf:
            one_line_str += ", " + str(
                np.round(self.num_true_positives / len(self.eval_residuum), 2)
            )
            key_str += (
                ", "
                + ", ".join([str(ss) for ss in sensor_str_list])
                + ", Precision abs"
            )
            one_line_str += ", " + str(
                np.round(
                    self.num_true_positives
                    / len(self.eval_residuum)
                    / self.precision_baseline
                    - 1.0,
                    2,
                )
            )
            key_str += (
                ", "
                + ", ".join([str(ss) for ss in sensor_str_list])
                + ", Precision rel"
            )
        print(key_str)
        print(one_line_str.replace(",", " $ & $").replace("nan", "-") + " $")
        if t_max == np.inf:
            one_line_str += ", " + self.get_num_measures_str(
                sensor_str_list=sensor_str_list
            )
            one_line_str += ", " + str(len(self.eval_residuum))
            key_str += (
                ", " + ", ".join([str(ss) for ss in sensor_str_list]) + ", n_objects"
            )

        print(key_str)
        print(one_line_str)

    def get_num_measures_str(self, sensor_str_list):
        """Get number of measured values as strings."""
        return ", ".join(
            [
                str(
                    sum(
                        [
                            len(eval_res["valid_idx"].get(sensor_str, []))
                            for eval_res in self.eval_residuum.values()
                        ]
                    )
                )
                for sensor_str in sensor_str_list
            ]
        )

    @staticmethod
    def print_sensor_range(pref: str, sensor_str: str, match_cts: dict, lost_cts):
        """Print stats of sensor range.

        Args:
            pref (str): prefix to print.
            sensor_str (str): String of sensor modality
            match_cts (dict): Container of match stats (distance, sensor modality, etc.)
            lost_cts (dict): Container of lost stats (distance, sensor modality etc.)
        """
        str_out = pref + ", {}".format(sensor_str)
        if not match_cts[sensor_str]:
            str_out += ": No detections"
            print(str_out)
            return

        str_out += " (n = {}): ".format(len(match_cts[sensor_str]))
        str_out += "match: {:.02f} m, {:.02f} m, ".format(
            np.max(match_cts[sensor_str]),
            np.mean(match_cts[sensor_str]),
        )

        str_out += "lost: {:.02f} m, {:.02f} m".format(
            np.max(lost_cts[sensor_str]),
            np.mean(lost_cts[sensor_str]),
        )

        print(str_out)

    @staticmethod
    def print_stats(str_val, container, unit_in="m"):
        """Print stat for given container."""
        str_out = str_val + ": "
        for key, val in container.items():
            if np.isnan(val).all() or not bool(val):
                continue
            str_out += "{} in {} = ({:.02f}, {:.02f}, {:.02f}, {:.02f}); ".format(
                key,
                unit_in,
                np.nanmean(val),
                np.nanstd(val),
                np.nanmedian(val),
                np.nanquantile(val, 0.95),
            )
        flat_list = [item for sublist in list(container.values()) for item in sublist]
        if np.isnan(flat_list).all():
            str_out += "total in {} = nan, nan, nan, nan".format(unit_in)
            tot_val = "{}, {}".format(np.nan, np.nan)
        else:
            str_out += "total in {} = {:.02f}, {:.02f}, {:.02f}, {:.02f}".format(
                unit_in,
                np.nanmean(flat_list),
                np.nanstd(flat_list),
                np.nanmedian(flat_list),
                np.nanquantile(val, 0.95),
            )
            tot_val = "{:.02f}, {:.02f}".format(
                np.nanmean(flat_list), np.nanstd(flat_list)
            )
        print(str_out)

        return tot_val

    def fill_containers(
        self,
        bounds_list: list,
        eval_key: str,
        sensor_str_list: list,
        idx_res=0,
        no_int=False,
    ):
        """Fill containers with evaluation values from all objects.

        Args:
            bounds_list (list): List of limits to separate values. (bins)
            eval_key (str): Key to evaluate, i.e. to split into the bins.
            sensor_str_list (list): List of sensor strings.
            idx_res (int, optional): Index residual to shift bins. Defaults to 0.
            no_int (bool, optional): If true label keys are integer. Defaults to False.

        Returns:
            tuple: Labels to plot, filled containers and NIS-values
        """
        x_tick_labels = self.get_labels(bounds_list, no_int=no_int)
        print("\n\n+++++ Fill containers, eval_key = {} +++++".format(eval_key))
        print("Bounds: " + ", ".join(["{:.02f}".format(vel) for vel in bounds_list]))

        # creating the containers for the boxplots
        bound_array = np.array(bounds_list)

        nis_cts, x_res_cts, y_res_cts, yaw_res_cts, vel_res_cts = (
            {
                sensor_str: [[] for _ in range(len(bounds_list))]
                for sensor_str in sensor_str_list
            }
            for _ in range(5)
        )
        iter_res_list = iter(self.eval_residuum.values())
        for eval_res in iter_res_list:
            for sensor_str in sensor_str_list:
                if sensor_str not in eval_res["sensors"]:
                    continue

                if isinstance(eval_res[eval_key], dict):
                    iter_bound_val = iter(eval_res[eval_key][sensor_str])
                else:
                    iter_bound_val = iter(eval_res[eval_key])

                for idx, bounds_val in enumerate(iter_bound_val):
                    group_idx = np.sum(bound_array < bounds_val) - idx_res

                    if (
                        bound_array[-1] != np.inf
                        and group_idx > len(x_res_cts[sensor_str]) - 1
                    ):
                        continue

                    nis_cts[sensor_str][group_idx].append(
                        eval_res["nis"][sensor_str][idx]
                    )
                    x_res_cts[sensor_str][group_idx].append(
                        eval_res["x_res"][sensor_str][idx]
                    )
                    y_res_cts[sensor_str][group_idx].append(
                        eval_res["y_res"][sensor_str][idx]
                    )
                    yaw_res_cts[sensor_str][group_idx].append(
                        eval_res["yaw_res"][sensor_str][idx]
                    )
                    vel_res_cts[sensor_str][group_idx].append(
                        eval_res["v_res"][sensor_str][idx]
                    )

        in_containers = [
            [
                [self.get_non_nan_list(x_ct) for x_ct in x_res_cts[sensor_str]],
                [self.get_non_nan_list(x_ct) for x_ct in y_res_cts[sensor_str]],
                [self.get_non_nan_list(x_ct) for x_ct in yaw_res_cts[sensor_str]],
                [self.get_non_nan_list(x_ct) for x_ct in vel_res_cts[sensor_str]],
            ]
            for sensor_str in sensor_str_list
        ]

        non_nan_nis = [
            [[self.get_non_nan_list(x_ct) for x_ct in nis_cts[sensor_str]]]
            for sensor_str in sensor_str_list
        ]

        if len(sensor_str_list) == 2:
            in_containers.append([])
            non_nan_nis.append([])
            for s1, s2 in zip(in_containers[0], in_containers[1]):
                in_containers[-1].append([s1[j] + s2[j] for j in range(len(s1))])

            for s1, s2 in zip(non_nan_nis[0], non_nan_nis[1]):
                non_nan_nis[-1].append([s1[j] + s2[j] for j in range(len(s1))])

        for j, sensor_str in enumerate(sensor_str_list):
            print("Number of measures per bin, sensor = {}:".format(sensor_str))
            print(", ".join([str(len(vel)) for vel in in_containers[j][0]]))

        y_label_list = [
            "$x_{\mathrm{res}}$ in m",
            "$y_{\mathrm{res}}$ in m",
            "$\psi_{\mathrm{res}}$ in °",
            "$v_{\mathrm{res}}$ in m/s",
        ]

        return x_tick_labels, y_label_list, in_containers, non_nan_nis

    def reduce_container(self, in_containers):
        """Reduce containers.

        Depends if vaw and velocity should be also visualized.

        Args:
            in_containers (list): list of evaluation cotainers.

        Returns:
            list: Reduced containers.
        """
        if not self.add_yaw and not self.add_vel:
            idx_list = [0, 1]
        elif not self.add_yaw and self.add_vel:
            idx_list = [0, 1, 3]
        elif self.add_yaw and not self.add_vel:
            idx_list = [0, 1, 2]
        else:
            idx_list = [0, 1, 2, 3]
        return [[ic[idx] for idx in idx_list] for ic in in_containers]

    def get_post_fix_str(self, sensor_str_list):
        """Get posterior string to save figure.

        Args:
            sensor_str_list (list): list of sensor modalities

        Returns:
            str: posterior string
        """
        postf = "_".join(
            [
                self.sensor_visz_dict[str_sensor]["name"]
                for str_sensor in sensor_str_list
            ]
        )
        if self.add_yaw:
            postf += "_yaw"
        if self.add_vel:
            postf += "_vel"

        return postf

    def eval_per_obs_time(self, sensor_str_list):
        """Evaluate residuals per observation time bins.

        Args:
            sensor_str_list (list): list of sensor modalities
        """
        x_label = "$t_{\mathrm{obs}}$ in s"
        (
            x_tick_labels,
            y_label_list,
            in_containers,
            nis_container,
        ) = self.fill_containers(
            eval_key="obj_t",
            bounds_list=self.t_bounds,
            sensor_str_list=sensor_str_list,
            no_int=False,
        )
        in_containers = self.reduce_container(in_containers)

        postf = self.get_post_fix_str(sensor_str_list)

        self._create_boxplots(
            in_containers,
            title=None,
            ax_titles=[None for _ in range(len(in_containers[0]))],
            x_label=x_label,
            y_label_list=y_label_list,
            x_tick_labels=x_tick_labels,
            sensor_str_list=sensor_str_list,
            # y_lims=[(0.0, 22.0), (0.0, 11.0), (0.0, 22.0)],
            file_save_name="res_vs_t_observation_" + postf,
        )

        self._create_boxplots(
            nis_container,
            title=None,
            ax_titles=[None],
            x_label=x_label,
            y_label_list=["$\mathrm{NIS}$"],
            x_tick_labels=x_tick_labels,
            sensor_str_list=sensor_str_list,
            # y_lims=[(0.0, 22.0), (0.0, 11.0), (0.0, 22.0)],
            file_save_name="nis_vs_t_observation_" + postf,
        )

    def eval_per_vel(self, sensor_str_list):
        """Evaluate residuals per velocity bins.

        Args:
            sensor_str_list (list): list of sensor modalities
        """
        x_label = "$v$ in m/s"
        (
            x_tick_labels,
            y_label_list,
            in_containers,
            nis_container,
        ) = self.fill_containers(
            eval_key="ego_vel",
            bounds_list=self.velocity_bounds,
            sensor_str_list=sensor_str_list,
        )

        in_containers = self.reduce_container(in_containers)

        postf = self.get_post_fix_str(sensor_str_list)

        self._create_boxplots(
            in_containers,
            title=None,
            ax_titles=[None for _ in range(len(in_containers[0]))],
            x_label=x_label,
            y_label_list=y_label_list,
            x_tick_labels=x_tick_labels,
            sensor_str_list=sensor_str_list,
            # y_lims=[(0.0, 22.0), (0.0, 11.0), (0.0, 22.0)],
            file_save_name="res_vs_velocity_" + postf,
        )

        self._create_boxplots(
            nis_container,
            title=None,
            ax_titles=[None],
            x_label=x_label,
            y_label_list=["$\mathrm{NIS}$"],
            x_tick_labels=x_tick_labels,
            sensor_str_list=sensor_str_list,
            # y_lims=[(0.0, 22.0), (0.0, 11.0), (0.0, 22.0)],
            file_save_name="nis_vs_velocity_" + postf,
        )

    @staticmethod
    def set_box_color(bp, color):
        """Set box colors.

        Args:
            bp (Boxplot): Matplotlib boxplot.
            color (string): color string.
        """
        plt.setp(bp["boxes"], color=color)
        plt.setp(bp["whiskers"], color=color)
        plt.setp(bp["caps"], color=color)
        plt.setp(bp["medians"], color=color)
        plt.setp(bp["means"], color=color)

    def get_legend_elements(self, num_tuples, sensor_str_list):
        """Get element for legend in plot.

        Args:
            num_tuples (int): Number of tuples
            sensor_str_list (list): list of sensor modalities.

        Returns:
            _type_: List of Line2D elements.
        """
        if num_tuples > 1:
            return [
                Line2D(
                    [0],
                    [0],
                    color=self.sensor_visz_dict[sensor_str]["col"],
                    linestyle="solid",
                    linewidth=3,
                    label=self.sensor_visz_dict[sensor_str]["name"],
                )
                for sensor_str in sensor_str_list
            ]

        return [
            Line2D(
                [0],
                [0],
                color=self.sensor_visz_dict[sensor_str_list[0]]["col"],
                linestyle="dashed",
                linewidth=3,
                label="Mean",
            ),
            Line2D(
                [0],
                [0],
                color=self.sensor_visz_dict[sensor_str_list[0]]["col"],
                linestyle="solid",
                linewidth=3,
                label="Median",
            ),
        ]

    def _create_boxplots(
        self,
        data_container_list: list,
        title: str,
        ax_titles: list,
        x_label: str,
        y_label_list: list,
        x_tick_labels: list = None,
        y_lims: list = None,
        sensor_str_list: list = ["lidar_cluster"],
        file_save_name: str = None,
        w_tot: float = 0.6,
    ):
        """Create a plot with boxplots.

        args:
            data_containers: [list], contains in each item the data for one of the subplots, which are boxplots.
            title: [str], the title of the whole figure.
            ax_titles: [list of strings], the titles of the small boxplots.
            x_label: [str], the label of the x axes. (The same for every boxplot)
            y_label: [str], the label of the y axes. (The same for every boxplot)
            x_tick_labels: [list], the ticks to set on the x axis for the boxplots, if provided.
            y_lims: [list of tuples] The y limits of the plots if provided. Each tuple is in (bottom, top) format.
            file_save_name: [str], set file_save_name if desired
        """
        num_tuples = len(data_container_list)
        num_boxplots = len(data_container_list[0])
        num_splits = len(data_container_list[0][0])

        fig, ax_list = plt.subplots(
            1, num_boxplots, figsize=(PAGEWIDTH, PAGEWIDTH / (num_boxplots * 6) * 10)
        )

        if num_boxplots == 1:
            ax_list = [ax_list]

        # fig.suptitle(title)
        fig.canvas.manager.set_window_title(title)

        # legend_elements = self.get_legend_elements(
        #     num_tuples,
        #     sensor_str_list,
        # )

        props = dict(linewidth=1.5, color="k")
        colcols = [
            self.sensor_visz_dict[sensor_str]["col"] for sensor_str in sensor_str_list
        ]

        def get_lims(idx=0, scale=1.1, sym=True):
            _max = -np.inf
            _min = np.inf
            for k in range(num_tuples):
                for j in range(num_splits):
                    if not data_container_list[k][idx][j]:
                        continue

                    q25 = np.quantile(data_container_list[k][idx][j], 0.25)
                    q75 = np.quantile(data_container_list[k][idx][j], 0.75)
                    w_low = q25 - 1.5 * (q75 - q25)
                    w_up = q75 + 1.5 * (q75 - q25)

                    _min = np.min([_min, w_low])
                    _max = np.max([_max, w_up])

            _min *= scale
            _max *= scale

            if sym:
                abs_max = np.max([np.abs(_min), np.abs(_max)])
                return (-abs_max, abs_max)
            return (_min, _max)

        def get_same_lims(x_lims, y_lims):
            if x_lims[1] > y_lims[1]:
                return x_lims

            return y_lims

        x_res_lims = get_lims(idx=0)
        if num_boxplots > 1:
            y_res_lims = get_lims(idx=1)
            x_res_lims = get_same_lims(x_res_lims, y_res_lims)

        for i, ax in enumerate(ax_list):
            idx_list = []
            bp_pair = [
                [dd[i][k] for dd in data_container_list] for k in range(num_splits)
            ]

            # plot per split
            split_idx = []
            for jx in range(num_splits):
                bp_active_idx = [j for j, bb in enumerate(bp_pair[jx]) if bool(bb)]
                active_tuples = len(bp_active_idx)
                if active_tuples > 1:
                    pp = [
                        jx + 1 - w_tot * 0.8 * (jj / (num_tuples - 1) - 0.5)
                        for jj in range(active_tuples)
                    ]
                else:
                    pp = [jx + 1]

                idx_list += bp_active_idx
                if bp_active_idx:
                    split_idx.append(jx + 1)

                for j, idx in enumerate(bp_active_idx):
                    bp = bp_pair[jx][idx]
                    if idx < len(colcols):
                        _col = colcols[idx]
                    else:
                        _col = "k"
                    ppx = pp[j]
                    boxplt = ax.boxplot(
                        bp,
                        showfliers=False,
                        meanline=True,
                        showmeans=False,
                        boxprops=props,
                        whiskerprops=props,
                        capprops=props,
                        medianprops=props,
                        meanprops=props,
                        positions=[ppx],
                        widths=w_tot / num_tuples,
                    )
                    self.set_box_color(boxplt, _col)
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if ax_titles[i]:
                ax.set(title=ax_titles[i])

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label_list[i])

            if not x_label:
                plt.xticks(color="w")
            if not y_label_list[i]:
                plt.yticks(color="w")

            ax.grid(True)

            # if len(data_container_list) > 1:
            #     lg_in = [
            #         leg_el for j, leg_el in enumerate(legend_elements) if j in idx_list
            #     ]
            # else:
            #     lg_in = legend_elements
            # ax.legend(handles=lg_in, prop={"size": FONTSIZE})

            if x_tick_labels is not None:
                ax.set_xticks(split_idx)
                ax.set_xticklabels([x_tick_labels[xt - 1] for xt in split_idx])

            if y_lims is not None:
                ax.set_ylim(bottom=y_lims[i][0], top=y_lims[i][1])
            elif i < 2:
                ax.set_ylim(bottom=x_res_lims[0], top=x_res_lims[1])
            else:
                _lims = get_lims(idx=i, scale=1.2, sym=True)
                ax.set_ylim(bottom=_lims[0], top=_lims[1])

        if self.results_save_path:
            if file_save_name is None:
                file_name = title.replace(" ", "_") + ".pdf"
            else:
                file_name = file_save_name + ".pdf"
            try:
                self.save_plt(file_name=file_name, bbox_inches="tight", pad_inches=0)
            except Exception:
                self.save_plt(file_name=file_name)
        if self.show_plt:
            plt.show()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--save_file_name", type=str, default=None)
    parser.add_argument("--results_save_path", type=str, default=None)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--show_plt", default=False, action="store_true")
    parser.add_argument("--show_filtered", default=False, action="store_true")
    parser.add_argument("--print_terminal", default=False, action="store_true")

    args = parser.parse_args()

    eval = Evaluation(args)
    eval.main_call()
