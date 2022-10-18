"""Shoe Delay Compensation."""
import os
import sys
import argparse
import matplotlib.pyplot as plt

TUM_BLUE = (0 / 255, 101 / 255, 189 / 255)
EGO_COL = TUM_BLUE
TUM_ORAN = (227 / 255, 114 / 255, 34 / 255)
OBJ_COL = TUM_ORAN
HIST_COL = "black"
HIST_LS = "solid"
GT_COL = "black"
GT_LS = "dashed"
BOUND_COL = (204 / 255, 204 / 255, 204 / 255)
OFF_COL = (233 / 255, 233 / 255, 233 / 255)

FONTSIZE = 8  # IEEE
COLUMNWIDTH = 3.5
PAGEWIDTH = COLUMNWIDTH * 2.0
FIGWIDTH = PAGEWIDTH

if FIGWIDTH == PAGEWIDTH:
    MARKERSIZE = FONTSIZE
    LINEWIDTH = MARKERSIZE / 8
else:
    MARKERSIZE = FONTSIZE / 1.5
    LINEWIDTH = MARKERSIZE / 8


plt.rcParams.update({"font.size": FONTSIZE})
plt.rcParams.update({"font.family": "Times New Roman"})
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"figure.autolayout": True})

plt.rcParams.update(
    {
        "legend.fontsize": FONTSIZE,
        "axes.labelsize": FONTSIZE,
        "axes.titlesize": FONTSIZE,
        "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE,
    }
)


from matplotlib.lines import Line2D
from shapely.geometry import Polygon

import numpy as np

REPO_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(REPO_PATH)


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

from tools.visualize_logfiles import ViszTracking
from utils.geometry import rotate_loc_glob
from utils.tracking_utils import ctrv_single

# default setting
RUN3_FINAL = "RUN3_final_2022_10_01-00_00_01"
RESULTS_PATH = "evaluation/results/__default"
OBJ_ID = str(345)  # fastest overtake
EVAL_IDX = 45858


class ViszDelayComp(ViszTracking):
    """Class to visualize delay compensation mechanism.

    Args:
        ViszTracking (class): Inheritance.
    """

    def __init__(self, args):
        """Initialize class.

        Args:
            args (namespace): Arguments to set up the class.
        """
        # add args that are not used
        args.n_obs = np.inf
        args.states = False
        args.ego_state = False
        args.filter = False
        args.mis_num_obj = False
        args.eval = False
        args.show_sample = True
        args.show_filtered = False
        args.is_main = False
        self.args = args

        # manipulate plot
        self.markersize = MARKERSIZE
        self.linewidth = LINEWIDTH

        self.results_save_path = args.results_save_path

        super().__init__(args=args)

        self.obj_id = args.obj_id
        self.eval_idx = args.eval_idx

        self.obj_idx = list(self.obj_items[self.obj_id]["object_states"].keys())

        # previous trajectory
        self._ViszTracking__bool_prev_traj = False

        # future trajectory
        self._ViszTracking__bool_fut_traj = False

        # matched Detections
        self._ViszTracking__bool_sensor_detection = True

        # Filter possibilities
        self._ViszTracking__bool_filter = False

        # Filter-Prior
        self._ViszTracking__bool_filter_prior = False

        # Filter-Delay
        self._ViszTracking__bool_filter_delay = False

        # Filter-Posterior
        self._ViszTracking__bool_filter_post = False

        # Prediction
        self._ViszTracking__bool_predict = False

        # unmatched Detections
        self._ViszTracking__bool_unmatched = True

        # raw detection data
        self._ViszTracking__bool_raw_data = True

        self.file_name = "delay_compensation.pdf"
        self.txt_name = "delay_compensation_stats.txt"

        self.manipulate_plt()

    def manipulate_plt(self):
        """Manipulate plot items.

        Set markers, linewidths, fonts, object colors.
        """

        # set colors
        self.ego_col = EGO_COL
        self.color_list = [BOUND_COL for _ in range(len(self.color_list))]
        self.color_list[int(int(self.obj_id) % len(self.color_list))] = OBJ_COL
        self.track_color = {"track_bounds": BOUND_COL, "pit_bounds": BOUND_COL}
        self.raw_data_color = "k"

    def visualize_(self, is_iterative=False):
        """Visualize one sample, i.e. a specific time stamp of one object.

        Visualization contains comprehensive information about delay compensation.

        Args:
            is_iterative (bool, optional): If True plot is called iteratively. Defaults to False.
        """
        # plot the log frame
        self._ViszTracking__plt_log(frame_value=self.eval_idx)

        fig = plt.gcf()
        fig.set_size_inches(FIGWIDTH, FIGWIDTH * 1.5 / 3.5)

        # set limits
        xlims, ylims = self.get_ax_lims()

        # show off road part in gray
        self.shadow_off_track(xlims, ylims)
        (
            detected_pose,
            obj_hist_comp,
            obj_hist_non_comp,
            match_state,
        ) = self.analyze_log_item(is_iterative=is_iterative)

        # plot object history
        if obj_hist_comp is not None:
            self.plt_obj_hist(self._ax, obj_hist_comp, obj_hist_non_comp, match_state)

            plt_all_steps = True

            if not is_iterative:
                loc = [0.03, 0.65, 0.16, 0.26]
                insax_2 = self.plt_ins_ax(
                    obj_hist_comp,
                    obj_hist_non_comp,
                    match_state,
                    detected_pose,
                    loc,
                    ax_lims="tight",
                )
                self._ax.indicate_inset_zoom(insax_2, edgecolor="black")

            # arange plot
            self.arange_plt(xlims, ylims)
        else:
            plt_all_steps = False
            self.arange_plt(
                xlims,
                ylims,
                add_detections=bool(detected_pose),
                add_prior=False,
                add_post=False,
            )

        plt.tight_layout()
        plt.show(block=False)

        if is_iterative:
            plt.pause(1)
            self.save_fig(is_iterative=True)
            return

        self.save_fig()
        plt.pause(1.0)

        if plt_all_steps:
            self.plt_all_steps(fig=fig, xlims=xlims, ylims=ylims)

    def plt_all_steps(self, fig, xlims, ylims):
        """Plot all steps of the tracking module.

        Plots are created by removing all features step by step.
        """
        fig_children = fig.axes[0].get_children()

        # indices of fig children
        # 0 - rectangle of lidar detection
        # 2 - 13 - rectangles of radar detection
        # 14 center points of raw detecions
        # 16 - rectangle ego
        # 18 - rectangle tracked object
        # 19 - rectangle historic object
        # 20 - center point of lidar cluster in object color
        # 21 - center points of ego
        # 22 - right bound
        # 23 - left bound
        # 24 - gray area south
        # 25 - gray area north
        # 26 - correct storage (posterior)
        # 27 - old storage (prior)
        # 28 - old object state from storage to match
        # 29 - rectangle to show zoom area
        # 30 - 33 - lines from zoom area to insert axes
        # 43 - inserted axes (whole box)
        # 44 - legend

        # remove inserted axes
        if len(fig_children) < 44:
            return

        fig_children[43].remove()
        [fig_children[j].remove() for j in range(29, 34)]
        self.save_fig("_1")

        plt.show(block=False)
        plt.pause(1.0)

        # remove corrected storage
        fig_children[26].remove()
        self.arange_plt(
            xlims, ylims, add_detections=True, add_prior=True, add_post=False
        )
        self.save_fig("_2")
        plt.show(block=False)
        plt.pause(1.0)

        # remove match
        fig_children[28].remove()
        fig_children[20].remove()
        fig_children[19].remove()
        self.save_fig("_3")
        plt.show(block=False)
        plt.pause(1.0)

        # remove old storage
        fig_children[27].remove()
        self.arange_plt(
            xlims, ylims, add_detections=True, add_prior=False, add_post=False
        )
        self.save_fig("_4")
        plt.show(block=False)
        plt.pause(1.0)

        # remove gray bounds (out of bound filter)
        fig_children[24].remove()
        fig_children[25].remove()
        self.save_fig("_5")
        plt.show(block=False)
        plt.pause(1.0)

        # remove detections
        [fig_children[j].remove() for j in range(15)]
        self.arange_plt(
            xlims, ylims, add_detections=False, add_prior=False, add_post=False
        )
        self.save_fig("_6")
        plt.show(block=False)
        plt.pause(1.0)

    def save_fig(self, post_str="", add_svg=True, is_iterative=False):
        """Save figure as .pdf, and additionally to .svg.

        Args:
            post_str (str, optional): Postfix string to add to save name. Defaults to "".
            add_svg (bool, optional): If true figure is also saved as svg. Defaults to True.
            is_iterative (bool, optional): If true save name is modified. Defaults to False.
        """
        save_name = self.file_name.replace(".", post_str + ".")
        # save fig
        plt_path = os.path.join(self.results_save_path, "plots")

        if is_iterative:
            plt_path = os.path.join(plt_path, "iterative")
            save_name = save_name.replace(".", "_{}.".format(self.eval_idx))

        if os.path.exists(plt_path):
            _ = [
                os.remove(os.path.join(plt_path, kk))
                for kk in os.listdir(plt_path)
                if kk.endswith(".png")
            ]
        else:
            os.makedirs(plt_path)

        plt.savefig(os.path.join(plt_path, save_name), format="pdf")

        if not add_svg:
            return

        save_name = save_name.replace(".pdf", ".svg")
        svg_path = os.path.join(plt_path, "svg")
        if not os.path.exists(svg_path):
            os.makedirs(svg_path)
        plt.savefig(os.path.join(svg_path, save_name), format="svg")

    def get_ax_lims(self):
        """Get adaptive ax limits."""
        x_fig, y_fig = self.fig.get_size_inches()
        obj_x, obj_y, _, _, _, _ = self.obj_items[self.obj_id]["object_states"][
            self.eval_idx
        ]
        ego_x, ego_y, _, _, _, _ = self.obj_items["ego"]["object_states"][self.eval_idx]

        x_min = np.min([obj_x, ego_x])
        x_max = np.max([obj_x, ego_x])

        dx = x_max - x_min

        xmin = int(np.round(x_min - dx * 0.25, -1))
        xmax = int(np.round(x_max + dx * 0.25, -1))

        yspan = (xmax - xmin) * y_fig / x_fig
        ymin = int(np.mean([obj_y, ego_y]) - yspan / 3)
        ymax = int(np.mean([obj_y, ego_y]) + 2 * yspan / 3)

        return ([xmin, xmax], [ymin, ymax])

    def shadow_off_track(self, xlims: list, ylims: list):
        """Shadow non-driveable area in gray within x- and ylims.

        Args:
            xlims (list): List with [min, max] x limits.
            ylims (list): List with [min, max] y limits.
        """
        xx = np.linspace(xlims[0] * 0.8, xlims[1] * 1.2)
        rb_idx = [
            idx
            for idx, val in enumerate(self.right_bound)
            if self.bool_inside_area(xlims, ylims, val)
        ]
        lb_idx = [
            idx
            for idx, val in enumerate(self.left_bound)
            if self.bool_inside_area(xlims, ylims, val)
        ]
        rb_b = self.right_bound[rb_idx[::-1]]
        lb_b = self.left_bound[lb_idx[::-1]]

        rb_intp = np.interp(xx, rb_b[:, 0], rb_b[:, 1])
        lb_intp = np.interp(xx, lb_b[:, 0], lb_b[:, 1])

        self._ax.fill_between(xx, 0.0, lb_intp, color=OFF_COL)
        self._ax.fill_between(xx, rb_intp, 2 * np.max(rb_intp), color=OFF_COL)

    def analyze_log_item(self, is_iterative=False):
        """Analyze given log entry on specific time step.

        Comprises analyses of perception delay and print further information.

        Args:
            is_iterative (bool, optional): If true iterative call of figure is expected. Defaults to False.

        Returns:
            tuple: Detected post, compensated object history, non-compensated obj history, x prior of EKF.
        """
        # log row entries
        (
            _,
            detection_input,
            _,  # ego_state,
            tracking_input,  # tracking_input,
            _,  # match_dict,
            filter_log,
            object_dict,
            _,  # pred_dict,
            cycle_time_ns,
        ) = self.log_rows[self.eval_idx]

        # get match idx
        dt_s = 1 / self.params["TRACKING"]["filter_frequency"]

        def upsample_states(inp_state, n_comps):
            upsampled_state = [np.array(inp_state)]
            for _ in range(n_comps):
                xx = np.array(upsampled_state[-1])
                upsampled_state.append(ctrv_single(xx, dt_s))
            return upsampled_state

        def print_stuff(str_sens, dt_min, dt_delay_ms):
            print(
                "Sample of {} is {:.02f} s after first detection of the object".format(
                    str_sens,
                    (
                        self.obj_items[self.obj_id]["t_dict"][self.eval_idx]
                        - list(self.obj_items[self.obj_id]["t_dict"].values())[0]
                    )
                    / 1e9,
                )
            )
            print(
                "Object Speed: {:.02f} km/h".format(
                    self.obj_items[self.obj_id]["object_states"][self.eval_idx][3] * 3.6
                )
            )
            print(
                "Ego Speed: {:.02f} km/h".format(
                    self.obj_items["ego"]["object_states"][self.eval_idx][3] * 3.6
                )
            )
            print("Cycle Time: {:.02f} ms".format(cycle_time_ns / 1e6))
            print(
                "Compensated {:.02f} ms delay, i.e. {:.01f} m".format(
                    dt_delay_ms,
                    object_dict[self.obj_id]["state"][3] * dt_delay_ms / 1e3,
                )
            )
            print(
                "dt from perception to historic state = {:.02f} ms".format(dt_min / 1e6)
            )

        for j, (str_sens, (obj_list, t_det)) in enumerate(tracking_input):
            # no valid objects
            if not obj_list:
                continue

            # no filter step conducted
            if filter_log[self.obj_id][j][0] is None:
                continue

            dt_delay_ms = (object_dict[self.obj_id]["t"] - t_det) / 1e6
            t_vals = list(self.obj_items[self.obj_id]["t_dict"].values())
            up_times = np.arange(t_vals[0], t_vals[-1], int(dt_s * 1e9))

            abs_diff = np.abs(up_times - t_det)
            index_start = abs_diff.argmin()
            dt_min = abs_diff.min()

            print_stuff(str_sens, dt_min, dt_delay_ms)

            # take only one match to visualize:
            if not is_iterative:
                filter_vals = filter_log[self.obj_id][j]
                x_prior = filter_vals[0]
                x_post = filter_vals[2]
                detected_pose = np.array(filter_vals[7][0][:3])

                # get compensated and non-compensated hist
                index_end = int(np.where(up_times == object_dict[self.obj_id]["t"])[0])
                n_comps = index_end - index_start

                inp_state = x_prior
                obj_hist_non_comp = upsample_states(
                    inp_state=inp_state, n_comps=n_comps
                )

                inp_state = x_post
                obj_hist_comp = upsample_states(inp_state=inp_state, n_comps=n_comps)

                return detected_pose, obj_hist_comp, obj_hist_non_comp, x_prior

        return detection_input, None, None, None

    def plt_obj_hist(self, ax, obj_hist_comp, obj_hist_non_comp, match_state):
        """Plot history of the track object.

        Args:
            ax (ax): Axis to plot
            obj_hist_comp (np.ndarray): Compensated object history, x-y-position (n x 2 shape)
            obj_hist_non_comp (np.ndarray): Non-compensated object history, x-y-position (n x 2 shape)
            match_state (list): List of [x, y] positions of matched state.
        """
        obj_hist_comp = np.array(obj_hist_comp)
        obj_hist_non_comp = np.array(obj_hist_non_comp)
        ax.plot(
            obj_hist_comp[:, 0],
            obj_hist_comp[:, 1],
            ".",
            markersize=0.2 * MARKERSIZE,
            linewidth=self.linewidth,
            color=OBJ_COL,
        )
        ax.plot(
            obj_hist_non_comp[:, 0],
            obj_hist_non_comp[:, 1],
            ".",
            markersize=0.2 * MARKERSIZE,
            linewidth=self.linewidth,
            color=BOUND_COL,
        )
        ax.plot(
            match_state[0],
            match_state[1],
            "*",
            markersize=MARKERSIZE,
            linewidth=self.linewidth,
            mfc="none",
            color=OBJ_COL,
        )

    def plt_ins_ax(
        self,
        obj_hist_comp,
        obj_hist_non_comp,
        match_state,
        detected_pose,
        loc,
        ax_lims=None,
    ):
        """Plot inserted axes to given figure.

        Args:
            obj_hist_comp (np.ndarray): Compensated object history, x-y-position (n x 2 shape)
            obj_hist_non_comp (np.ndarray): Non-compensated object history, x-y-position (n x 2 shape)
            match_state (list): List of [x, y] positions of matched state.
            detected_pose (list): List of detected pose, containing [x, y, yaw]
            loc (list): List of entries to locate inserted axes.
            ax_lims (str, optional): Define ax limits 'tight' or normal. Defaults to None.

        Returns:
            axis: Inserted axes.
        """
        if ax_lims == "tight":
            ins_xlims = [
                int(np.round(detected_pose[0] - 1.5)),
                int(np.round(detected_pose[0] + 1.5)),
            ]
            ins_ylims = [
                int(np.round(detected_pose[1] - 1.5)),
                int(np.round(detected_pose[1] + 1.5)),
            ]
        else:
            obj_x, obj_y, _, _, _, _ = self.obj_items[self.obj_id]["object_states"][
                self.eval_idx
            ]
            xmin = int(np.round(obj_x - 4, 0))
            ymin = int(np.round(obj_y - 2, 0))
            ins_xlims = [xmin, xmin + 20]
            ins_ylims = [ymin, ymin + 8]

        insax_1 = self._ax.inset_axes(loc)
        insax_1.axis("equal")
        self.plt_obj_hist(insax_1, obj_hist_comp, obj_hist_non_comp, match_state)

        insax_1.set_xlim(ins_xlims)
        insax_1.set_ylim(ins_ylims)
        insax_1.set_xticklabels([])
        insax_1.set_yticklabels([])

        insax_1.plot(
            detected_pose[0],
            detected_pose[1],
            self._sensor_marker["lidar_cluster"],
            markersize=1.0 * MARKERSIZE,
            linewidth=self.linewidth,
            color=self.raw_data_color,
        )
        insax_1.plot(
            detected_pose[0],
            detected_pose[1],
            self._sensor_marker["lidar_cluster"],
            markersize=1.0 * MARKERSIZE,
            linewidth=self.linewidth,
            mfc="none",
            color=OBJ_COL,
        )
        edges_glob = rotate_loc_glob(
            self.rectangle,
            detected_pose[2],
            matrix=False,
        ) + np.expand_dims(
            np.array(detected_pose[:2]),
            axis=1,
        )
        poly_1 = Polygon(edges_glob.T)
        _x, _y = poly_1.exterior.xy
        insax_1.plot(
            _x,
            _y,
            linewidth=LINEWIDTH,
            linestyle="solid",
            color=self.raw_data_color,
        )
        insax_1.plot(
            _x,
            _y,
            linewidth=LINEWIDTH,
            linestyle="--",
            color=OBJ_COL,
        )
        obj_pose = self.obj_items[self.obj_id]["object_states"][self.eval_idx][:3]
        edges_glob = rotate_loc_glob(
            self.rectangle,
            obj_pose[2],
            matrix=False,
        ) + np.expand_dims(
            np.array(obj_pose[:2]),
            axis=1,
        )
        poly_1 = Polygon(edges_glob.T)
        _x, _y = poly_1.exterior.xy
        insax_1.plot(
            _x,
            _y,
            linewidth=LINEWIDTH,
            linestyle="solid",
            color=OBJ_COL,
        )

        insax_1.plot(
            obj_pose[0],
            obj_pose[1],
            "x",
            markersize=0.5 * MARKERSIZE,
            linewidth=self.linewidth,
            color=OBJ_COL,
        )

        insax_1.tick_params(
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,
            left=False,
            right=False,  # ticks along the top edge are off
            labelbottom=False,
        )  # labels along the bottom edge are off

        return insax_1

    def arange_plt(
        self, xlims, ylims, add_detections=True, add_prior=True, add_post=True
    ):
        """Arange plot, i.e. set label, get legend handles, activate grid.

        Args:
            xlims (list): x limits, list of [min, max]
            ylims (list): y limits, list of [min, max]
            add_detections (bool, optional): If true detection label is added to legend. Defaults to True.
            add_prior (bool, optional): If true prior label is added to legend. Defaults to True.
            add_post (bool, optional): If true post label is added to legend. Defaults to True.
        """
        self._ax.set_xlabel("$x_{\mathrm{glob}}$ in m")
        self._ax.set_ylabel("$y_{\mathrm{glob}}$ in m")
        self._ax.grid(True)

        _ = [txt.set_visible(False) for txt in self.fig.texts]
        lg_handles = [
            Line2D(
                [0],
                [0],
                color=EGO_COL,
                linestyle="solid",
                linewidth=LINEWIDTH,
                label="Ego",
            ),
            Line2D(
                [0],
                [0],
                color=OBJ_COL,
                linestyle="solid",
                linewidth=LINEWIDTH,
                label="Object",
            ),
        ]

        if add_detections:
            lg_handles += [
                Line2D(
                    [0],
                    [0],
                    color=self.raw_data_color,
                    marker=self._sensor_marker["lidar_cluster"],
                    markersize=0.5 * MARKERSIZE,
                    linewidth=self.linewidth,
                    mfc="none",
                    linestyle="none",
                    label="LiDAR",
                ),
                Line2D(
                    [0],
                    [0],
                    color=self.raw_data_color,
                    marker=self._sensor_marker["radar"],
                    markersize=0.5 * MARKERSIZE,
                    linewidth=self.linewidth,
                    linestyle="none",
                    mfc="none",
                    label="RADAR",
                ),
            ]

        if add_prior:
            lg_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=OFF_COL,
                    linestyle="dotted",
                    linewidth=LINEWIDTH,
                    label="prior storage",
                )
            )

        if add_post:
            lg_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=OBJ_COL,
                    linestyle="dotted",
                    linewidth=LINEWIDTH,
                    label="posterior storage",
                )
            )

        if len(lg_handles) == 4:
            n_col = 2
        else:
            n_col = 1

        if FIGWIDTH == PAGEWIDTH:
            k = 1.0
        else:
            k = 0.6
        self._ax.legend(
            handles=lg_handles,
            prop={"size": k * FONTSIZE},
            loc="lower right",
            ncol=n_col,
        )
        self._ax.axis("equal")
        self._ax.set_xlim(xlims)
        self._ax.set_ylim(ylims)

    @staticmethod
    def bool_inside_area(xlims, ylims, in_arr):
        """Check if array is inside give area.

        Args:
            xlims (list): x limits, list of [min, max]
            ylims (list): y limits, list of [min, max]
            in_arr (np.ndarray): Array of x,y position shape: (,2)

        Returns:
            bool: If true position of in_arr is within limits.
        """
        x_low, x_high = xlims
        y_low, y_high = ylims

        return (
            1.1 * x_high > in_arr[0] > x_low * 0.95
            and 1.1 * y_high > in_arr[1] > 0.95 * y_low
        )

    def set_stdout_to_file(self):
        """Set stdout to file, i.e. print to file."""
        self.orig_stdout = sys.stdout
        self.f = open(
            os.path.join(self.args.results_save_path, self.txt_name),
            "w",
        )
        sys.stdout = self.f

    def set_stdout_to_sys_out(self):
        """Set std out back to sys out."""
        sys.stdout = self.orig_stdout
        self.f.close()

    def iter_visualize_(self, n_iters=100):
        """Iterate through all samples of one tracked object."""
        self.set_stdout_to_file()

        iter_idx = range(self.eval_idx - n_iters, self.eval_idx)
        for eval_idx in iter_idx:
            self.eval_idx = eval_idx
            self.visualize_(is_iterative=True)

        self.set_stdout_to_file()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--save_file_name", type=str, default=RUN3_FINAL)
    parser.add_argument("--results_save_path", type=str, default=RESULTS_PATH)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--obj_id", type=str, default=OBJ_ID)
    parser.add_argument("--eval_idx", type=int, default=EVAL_IDX)
    parser.add_argument("--iterative_visz", default=False, action="store_true")

    args = parser.parse_args()

    eval = ViszDelayComp(args)
    if args.iterative_visz:
        eval.iter_visualize_()
    else:
        eval.visualize_()
