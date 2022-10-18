"""Class of track boundary checks."""
import os
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

from utils.map_utils import get_track_kinematics, get_track_paths


class TrackBoundary:
    """Class to handle track related features."""

    def __init__(self, all_params, track_path=None):
        """Initaliaze class.

        params: params from main_object.ini -> Tracking
        track_file: .csv with ltpl specification
        """
        bounds_buffer_m = all_params["TRACKING"]["bounds_check"]["bounds_buffer_m"]
        self.bounds_buffer_outer_m = bounds_buffer_m
        if track_path is None:
            track_path = all_params["track_path"]

        if "pitlane_" in track_path:
            bounds_buffer_inner_m = None
        else:
            bounds_buffer_inner_m = (
                all_params["TRACKING"]["bounds_check"]["bounds_buffer_inner_m"],
            )

        if bounds_buffer_inner_m is None:
            self.bounds_buffer_inner_m = bounds_buffer_m
        else:
            self.bounds_buffer_inner_m = bounds_buffer_inner_m

        # get right, left and centerline
        (
            self.arc_center,
            self.center_line,
            self.bound_right,
            self.bound_left,
            track_width,
        ) = get_track_paths(track_path, bool_track_width=True)

        # get raceline
        (
            _,
            _,
            _,
            _,
            self.raceline,
        ) = get_track_paths(track_path, bool_track_width=False, bool_raceline=True)

        # for bool_outofbounds to provide yaw_angle
        center_yaw, center_curvature = get_track_kinematics(
            path_name="centerline",
            track_path=track_path,
            bool_get_yaw_curv=True,
        )

        # interpolate boundaries function
        track_kinematics = np.hstack(
            [
                self.raceline,
                self.bound_right,
                self.bound_left,
                np.expand_dims(track_width, 1),
                np.expand_dims(center_yaw, 1),
                np.expand_dims(center_curvature, 1),
            ]
        )
        self.fn_interpol_tracks = interpolate.interp1d(
            self.arc_center, track_kinematics, axis=0
        )

        self.max_arc = float(self.arc_center[-1])
        self.num_center = len(track_kinematics)
        self.vec_center = np.array(
            [
                self.center_line[(i + 1) % self.num_center, :] - self.center_line[i, :]
                for i in range(self.num_center)
            ]
        )

        # variables to get_boundaries
        self.center_line_batch = self.center_line.reshape(1, -1, 2)

    def get_arc_start(self, translation):
        """Get arc length at translation point."""
        indi_min = (
            np.argmin(np.linalg.norm(self.center_line - translation, axis=1))
            % self.num_center
        )
        vec_center = self.vec_center[indi_min]
        vec_veh = translation - self.center_line[indi_min]

        delta_arc = np.linalg.norm(
            np.dot(vec_center, vec_veh) / np.dot(vec_center, vec_center) * vec_center
        )
        arc_start = (self.arc_center[indi_min] + delta_arc) % self.max_arc

        return arc_start

    def get_kinematics(
        self, translation, arc_intp, bool_batch=False, bool_intp_values=False
    ):
        """Get kinematics starting from translation with arc_intp."""
        track_bounds_intp = self.fn_interpol_tracks(arc_intp)
        if bool_batch:
            dist_right = np.linalg.norm(translation - track_bounds_intp[:, 2:4], axis=1)
            dist_left = np.linalg.norm(translation - track_bounds_intp[:, 4:6], axis=1)
            track_width = track_bounds_intp[:, 6]
            yaw_angle = track_bounds_intp[:, 7]
            curvature = track_bounds_intp[:, 8]
        else:
            dist_right = np.linalg.norm(translation - track_bounds_intp[2:4])
            dist_left = np.linalg.norm(translation - track_bounds_intp[4:6])
            track_width = track_bounds_intp[6]
            yaw_angle = track_bounds_intp[7]
            curvature = track_bounds_intp[8]

        if bool_intp_values:
            return (
                dist_right,
                dist_left,
                track_width,
                yaw_angle,
                track_bounds_intp,
            )

        return dist_right, dist_left, track_width, yaw_angle, curvature

    def track_fn_single(self, translation, yaw_and_curvature: bool = False):
        """Calculate track kinematics - single."""
        arc_start = self.get_arc_start(translation)
        rel_position = arc_start / self.max_arc

        (
            dist_right,
            dist_left,
            track_width,
            yaw_angle,
            curvature,
        ) = self.get_kinematics(translation, arc_start, bool_intp_values=False)
        if yaw_and_curvature:
            return yaw_angle, curvature

        bool_outofbound = bool(
            dist_right - track_width > self.bounds_buffer_inner_m
            or dist_left - track_width > self.bounds_buffer_outer_m
        )
        return bool_outofbound, rel_position, yaw_angle

    def path_along_track(self, translation, pred_points, use_raceline=False):
        """Get path along track."""
        arc_start = self.get_arc_start(translation)

        pred_points += arc_start
        pred_points = pred_points % self.max_arc

        (
            dist_right,
            dist_left,
            track_width,
            pred_yaw,
            track_bounds_intp,
        ) = self.get_kinematics(
            translation, pred_points, bool_batch=True, bool_intp_values=True
        )

        is_inside = dist_left[0] <= track_width[0] and dist_right[0] <= track_width[0]
        if is_inside:
            weight_right = dist_left[0] / (dist_right[0] + dist_left[0])
        else:
            if dist_left[0] > dist_right[0]:
                weight_right = 1.0
            else:
                weight_right = 0.0

        pred_path = (
            weight_right * track_bounds_intp[:, 2:4]
            + (1.0 - weight_right) * track_bounds_intp[:, 4:6]
        )

        if use_raceline and is_inside:
            add_rl_factor = np.expand_dims(np.linspace(0, 1, len(track_bounds_intp)), 1)
            pred_path = (
                add_rl_factor * track_bounds_intp[:, :2]
                + np.flip(add_rl_factor) * pred_path
            )

        return pred_path, pred_yaw

    def track_fn_batch(self, translation, yaw_only: bool = False):
        """Check if the detected objects are on track.

        translation: np.array nx2 (#n detected objects)
            x and y position of the n objects in global coordinates
        Returns
        _ _ _ _
        bool_outofbounds: bool
            true: yes object is out of defined bounds
        rel_positions: array
            relative position on track: value between 0.0 and 1.0
        yaw_angle: array
            yaw_angle calculated from relative position along track
        """
        trla = translation.reshape(-1, 1, 2)
        deltas = self.center_line_batch - trla
        indis_min = np.argmin(np.einsum("ijk,ijk->ij", deltas, deltas), axis=1).tolist()

        v_centers = self.vec_center[indis_min, :].T
        v_vehs = translation - self.center_line_batch[0, indis_min, :]

        delta_arcs = np.linalg.norm(
            np.einsum("ji,ij->i", v_centers, v_vehs) / sum(v_centers**2) * v_centers,
            axis=0,
        )
        arc_starts = (self.arc_center[indis_min] + delta_arcs) % self.max_arc
        rel_positions = arc_starts / self.max_arc

        dists_right, dists_left, track_widths, yaw_angles, _ = self.get_kinematics(
            translation, arc_starts, bool_batch=True
        )

        if yaw_only:
            return yaw_angles

        bool_outofbounds = np.max(
            np.stack(
                [
                    dists_right - track_widths > self.bounds_buffer_inner_m,
                    dists_left - track_widths > self.bounds_buffer_outer_m,
                ]
            ),
            axis=0,
        )
        return bool_outofbounds, rel_positions, yaw_angles

    def add_yaw_batch(self, new_object_list: list):
        """Add yaw for filter intialization if not in measured variables."""
        input_arr = np.array(
            [np.array(new_object["state"][:2]) for new_object in new_object_list]
        )
        yaw_angles = self.track_fn_batch(input_arr, yaw_only=True)
        for idx, new_object in enumerate(new_object_list):
            new_object["state"].insert(2, yaw_angles[idx])

    def add_yaw(self, new_object: dict):
        """Add yaw for filter intialization if not in measured variables."""
        yaw_angle, curvature = self.track_fn_single(
            np.array(new_object["state"][:2]), yaw_and_curvature=True
        )
        new_object["state"][2] = yaw_angle
        return curvature

    def get_curvature(self, new_object: dict):
        """Add yaw for filter intialization if not in measured variables."""
        _, curvature = self.track_fn_single(
            np.array(new_object["state"][:2]), yaw_and_curvature=True
        )
        return curvature

    def yaw_and_bounds(
        self, new_object_list: dict, add_yaw: bool, sensor_str, logger_fn
    ):
        """Calculate yaw and bounds."""
        input_arr = np.array(
            [np.array(new_object["state"][:2]) for new_object in new_object_list]
        )

        bool_outofbounds, _, yaw_angles = self.track_fn_batch(input_arr)

        if add_yaw:
            logger_fn(
                "YAW ADD: PRIOR, sensor = {}, n_obj = {}, obj_list = {}".format(
                    sensor_str,
                    len(new_object_list),
                    new_object_list,
                )
            )
            for i, new_object in enumerate(new_object_list):
                new_object["state"].insert(2, yaw_angles[i])
            logger_fn(
                "YAW ADD: POST, sensor = {}, n_obj = {}, obj_list = {}".format(
                    sensor_str,
                    len(new_object_list),
                    new_object_list,
                )
            )

        return bool_outofbounds


if __name__ == "__main__":
    repo_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    track_path = os.path.join(repo_path, "data", "map_data", "traj_ltpl_cl_LO_GPS.csv")
    pit_path = os.path.join(
        repo_path, "data", "map_data", "traj_ltpl_cl_IMS_pitlane_GPS.csv"
    )

    bounds_buffer_m = -1.0
    bounds_buffer_inner_m = 0.0

    input_params = {
        "track_path": track_path,
        "pit_path": pit_path,
        "TRACKING": {
            "bounds_check": {
                "bounds_buffer_m": bounds_buffer_m,
                "bounds_buffer_inner_m": bounds_buffer_inner_m,
            }
        },
    }

    track_boundary = TrackBoundary(all_params=input_params)
    pit_boundary = TrackBoundary(
        all_params=input_params,
        track_path=pit_path,
    )
    bool_outofbound_single = []

    # add a shift of test_params["bounds_buffer_m"] to test the TrackBoundary
    n_steps = np.arange(
        0,
        track_boundary.bound_right.shape[0],
        int(track_boundary.bound_right.shape[0] / 100),
    )

    translations = track_boundary.bound_left[n_steps, :2]
    # translations[:, 0] -= 2.0
    for trans in list(translations):
        bool_outofbound_single.append(track_boundary.track_fn_single(trans)[0])
    bool_outofbound_batch, _, _ = track_boundary.track_fn_batch(translations)

    try:
        for n, bool_single in enumerate(bool_outofbound_single):
            assert bool_single == bool_outofbound_batch[n]
    except ValueError:
        print("invalid boundaries")

    plt.figure()
    plt.plot(track_boundary.bound_right[:, 0], track_boundary.bound_right[:, 1], "k")
    plt.plot(track_boundary.bound_left[:, 0], track_boundary.bound_left[:, 1], "k")
    for j, bool_val in enumerate(bool_outofbound_single):
        if bool_val:
            col_tuple = ".r"
        else:
            col_tuple = "xg"
        plt.plot(translations[j, 0], translations[j, 1], col_tuple)
    _ = plt.axis("equal")
    plt.grid(True)
    plt.show()

    # plt.figure()
    # plt.plot(track_boundary.bound_right[:, 0], track_boundary.bound_right[:, 1], "k")
    # plt.plot(track_boundary.bound_left[:, 0], track_boundary.bound_left[:, 1], "k")
    # plt.plot(pit_boundary.bound_right[:, 0], pit_boundary.bound_right[:, 1], "r")
    # plt.plot(pit_boundary.bound_left[:, 0], pit_boundary.bound_left[:, 1], "r")

    # for obj in egos:
    #     bool_track, _, _ = track_boundary.track_fn_single(obj["state"][:2])
    #     bool_pit, _, _ = pit_boundary.track_fn_single(obj["state"][:2])
    #     plt.plot(obj["state"][0], obj["state"][1], "xm")
    #     print(bool_track)
    #     print(bool_pit)
    #     print("\n\n")
    # _ = plt.axis("equal")
    # plt.grid(True)
    # plt.show()
    # sys.exit(0)
