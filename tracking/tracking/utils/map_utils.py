"""Functions to process map information."""
import os
import logging
import numpy as np
from scipy.signal import savgol_filter


def get_raceline_csv_name(
    map_params: dict,
    raceline: str,
    track_key: str,
) -> str:
    """Get csv-file name for chosen raceline.

    Possible raceline: default, inner, outer, center (LVMS only)

    Args:
        map_params (dict): Contains tracking parameter
        raceline (str): String specifying raceline
        track_key (str): Abbreviation for track ("LVMS", "IMS", "LOR")

    Returns:
        str: csv-file name for chosen raceline.
    """
    track_file_str = "track_file_" + track_key
    if raceline == "default":
        return map_params[track_file_str]
    if raceline == "center":
        return map_params[track_file_str + "_center"]
    if raceline == "inner":
        return map_params[track_file_str + "_inner"]
    if raceline == "outer":
        return map_params[track_file_str + "_outer"]

    return map_params[track_file_str]


def write_path(
    path_dict: dict,
    csv_file: str,
    logger: logging.Logger,
    is_ego: bool = False,
) -> str:
    """Concatenate path and csv file name.

    Either uses local path or overwrite path from ltpl-configs in docker_iac

    Args:
        path_dict ([path_dict]): Contains all paths
        csv_file (str): csv-file name of map data
        logger (logging.Logger): Message logger
        is_ego (bool): Check if ego raceline is evaluated

    Returns:
        str: Absolute path of map data stored in csv-file.
    """
    if os.path.exists(
        os.path.join(
            path_dict["map_path_overwrite"],
            csv_file,
        )
    ):
        logger.info(
            "OVERWRITE: new map csv (ego={}) {} at path {}".format(
                is_ego, csv_file, path_dict["map_path_overwrite"]
            )
        )
        return os.path.join(
            path_dict["map_path_overwrite"],
            csv_file,
        )

    return os.path.join(
        path_dict["map_path"],
        csv_file,
    )


def get_map_data(main_class, path_dict) -> None:
    """Get map paths for pit and track.

    Args:
        main_class ([ObjectTracking]): Main tracking class
        path_dict ([path_dict]): Contains all paths
    """
    assert main_class.params["track"] in (
        "LOR",
        "LVMS",
        "IMS",
    ), "No valid track specified, choose 'LVMS', 'IMS', 'LOR'"
    pit_file_key = "pit_file_" + main_class.params["track"]

    if main_class.params["TRACKING"]["prediction"]["obj_raceline"] == "ego":
        main_class.params["TRACKING"]["prediction"]["obj_raceline"] = main_class.params[
            "ego_raceline"
        ]
        main_class.msg_logger(
            "RACELINE: set ego raceline = {} for rail-based prediction, enabled = {}".format(
                main_class.params["ego_raceline"],
                main_class.params["TRACKING"]["prediction"]["bool_use_raceline"],
            )
        )

    obj_raceline_csv = get_raceline_csv_name(
        map_params=main_class.params["MISCS"],
        raceline=main_class.params["TRACKING"]["prediction"]["obj_raceline"],
        track_key=main_class.params["track"],
    )

    main_class.params["track_path"] = write_path(
        path_dict=path_dict,
        csv_file=obj_raceline_csv,
        logger=main_class.msg_logger,
    )
    check_path_exist(main_class=main_class, key_str="track_path")

    main_class.params["pit_path"] = write_path(
        path_dict=path_dict,
        csv_file=main_class.params["MISCS"][pit_file_key],
        logger=main_class.msg_logger,
    )
    check_path_exist(main_class=main_class, key_str="pit_path")

    ego_raceline_csv = get_raceline_csv_name(
        map_params=main_class.params["MISCS"],
        raceline=main_class.params["ego_raceline"],
        track_key=main_class.params["track"],
    )

    main_class.params["TRACKING"]["prediction"]["ego_raceline_path"] = write_path(
        path_dict=path_dict,
        csv_file=ego_raceline_csv,
        logger=main_class.msg_logger,
        is_ego=True,
    )


def check_path_exist(main_class, key_str: str):
    """Check if path exits.

    Args:
        main_class (class): Tracking class
        key_str (str): key str to check path for.
    """
    assert os.path.exists(main_class.params[key_str]), "{} does not exist: {}".format(
        key_str, main_class.params[key_str]
    )


def get_track_paths(
    track_path: str,
    bool_track_width: bool = False,
    bool_raceline: bool = False,
    bool_all_kinematics: bool = False,
) -> tuple:
    """Read the map data from the unified map file."""
    (
        refline,
        t_width_right,
        t_width_left,
        normvec_normalized,
        alpha_dist,
        s_rl,
        _,
        _,
        _,
        _,
        _,
        s_refline,
        psi_refline,
        kappa_refline,
        _,
    ) = import_global_trajectory_csv(import_path=track_path)

    x_intp = 0.999
    close_bound = x_intp * (refline[0, :] - refline[-1, :]) + refline[-1, :]
    refline = np.vstack([refline, close_bound])

    close_bound = (
        x_intp * (normvec_normalized[0, :] - normvec_normalized[-1, :])
        + normvec_normalized[-1, :]
    )
    normvec_normalized = np.vstack([normvec_normalized, close_bound])

    close_bound = x_intp * (t_width_right[0] - t_width_right[-1]) + t_width_right[-1]
    t_width_right = np.append(t_width_right, close_bound)

    close_bound = x_intp * (t_width_left[0] - t_width_left[-1]) + t_width_left[-1]
    t_width_left = np.append(t_width_left, close_bound)

    bound_right = refline + normvec_normalized * np.expand_dims(t_width_right, 1)
    bound_left = refline - normvec_normalized * np.expand_dims(t_width_left, 1)

    track_width = t_width_right + t_width_left
    if bool_track_width:
        return (s_rl, refline, bound_right, bound_left, track_width)
    if bool_raceline:
        close_bound = x_intp * (alpha_dist[0] - alpha_dist[-1]) + alpha_dist[-1]
        alpha_dist = np.append(alpha_dist, close_bound)
        raceline = refline + normvec_normalized * alpha_dist[:, np.newaxis]
        return (s_refline, refline, bound_right, bound_left, raceline)
    if bool_all_kinematics:
        close_bound = (
            x_intp * (kappa_refline[0] - kappa_refline[-1]) + kappa_refline[-1]
        )
        kappa_refline = np.append(kappa_refline, close_bound)
        close_bound = x_intp * (psi_refline[0] - psi_refline[-1]) + psi_refline[-1]
        psi_refline = np.append(psi_refline, close_bound)
        return (
            s_refline,
            refline,
            bound_right,
            bound_left,
            track_width,
            psi_refline,
            kappa_refline,
        )

    return (s_refline, refline, bound_right, bound_left)


def import_global_trajectory_csv(import_path: str) -> tuple:
    """Import global trajectory.

    :param import_path: path to the csv file containing the optimal global trajectory
    :type import_path: str
    :return: - xy_refline: x and y coordinate of reference-line

             - t_width_right: width to right track bound at given reference-line coordinates in meters

             - t_width_left: width to left track bound at given reference-line coordinates in meters

             - normvec_normalized: x and y components of normalized normal vector at given reference-line coordinates

             - alpha_dist: distance from optimal racing-line to reference-line at given reference-line coordinates

             - s_refline: s-coordinate of reference-line at given reference-line coordinates

             - psi_refline: heading at given reference-line coordinates

             - kappa_refline: curvature at given reference-line coordinates

             - dkappa_refline: derivative of curvature at given reference-line coordinates

             - s_rl: s-coordinate of racing-line at given racing-line coordinates

             - vel_rl: velocity at given racing-line coordinates

             - acc_rl: acceleration at given racing-line coordinates

             - psi_rl: heading at given racing-line coordinates

             - kappa_rl: curvature at given racing-line coordinates

             - banking: banking at given racling-line coordinates
    :rtype: tuple
    """
    # ------------------------------------------------------------------------------------------------------------------
    # IMPORT DATA ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # load data from csv file (closed; assumed order listed below)
    # x_ref_m, y_ref_m, width_right_m, width_left_m, x_normvec_m, y_normvec_m, alpha_m, s_racetraj_m,
    # psi_racetraj_rad, kappa_racetraj_radpm, vx_racetraj_mps, ax_racetraj_mps2
    csv_data_temp = np.loadtxt(import_path, delimiter=";")

    # get reference-line
    xy_refline = csv_data_temp[:-1, 0:2]

    # get distances to right and left bound from reference-line
    t_width_right = csv_data_temp[:-1, 2]
    t_width_left = csv_data_temp[:-1, 3]

    # get normalized normal vectors
    normvec_normalized = csv_data_temp[:-1, 4:6]

    # get racing-line alpha
    alpha_dist = csv_data_temp[:-1, 6]

    # get racing-line s-coordinate
    s_rl = csv_data_temp[:, 7]

    # get heading at racing-line points
    psi_rl = csv_data_temp[:-1, 8]

    # get kappa at racing-line points
    kappa_rl = csv_data_temp[:-1, 9]

    # get velocity at racing-line points
    vel_rl = csv_data_temp[:-1, 10]

    # get acceleration at racing-line points
    acc_rl = csv_data_temp[:-1, 11]

    # get banking
    banking = csv_data_temp[:-1, 12]

    # get reference-line s-coordinate
    s_refline = csv_data_temp[:, 13]

    # get heading at reference-line points
    psi_refline = csv_data_temp[:-1, 14]

    # get curvature at reference-line points
    kappa_refline = csv_data_temp[:-1, 15]

    # get derivative of curvature at reference-line points
    dkappa_refline = csv_data_temp[:-1, 16]

    return (
        xy_refline,
        t_width_right,
        t_width_left,
        normvec_normalized,
        alpha_dist,
        s_rl,
        psi_rl,
        kappa_rl,
        vel_rl,
        acc_rl,
        banking,
        s_refline,
        psi_refline,
        kappa_refline,
        dkappa_refline,
    )


def get_glob_raceline(
    loctraj_param_path, bool_vel_const=True, velocity=100.0, vel_scale=1.0
):
    """Load data from csv files."""
    (
        refline,
        _,
        _,
        normvec_normalized,
        alpha_mincurv,
        s_rl,
        psi_rl,
        _,
        vel_rl,
        acc_rl,
        _,
        _,
        _,
        _,
        _,
    ) = import_global_trajectory_csv(import_path=loctraj_param_path)

    # get race line
    raceline = refline + normvec_normalized * alpha_mincurv[:, np.newaxis]

    x_intp = 0.999
    close_bound = x_intp * (raceline[0, :] - raceline[-1, :]) + raceline[-1, :]
    raceline = np.vstack([raceline, close_bound])

    close_bound = x_intp * (psi_rl[0] - psi_rl[-1]) + psi_rl[-1]
    psi_rl = np.append(psi_rl, close_bound)

    close_bound = x_intp * (vel_rl[0] - vel_rl[-1]) + vel_rl[-1]
    vel_rl = np.append(vel_rl, close_bound)

    close_bound = x_intp * (acc_rl[0] - acc_rl[-1]) + acc_rl[-1]
    acc_rl = np.append(acc_rl, close_bound)

    if bool_vel_const:
        vel_rl = np.ones(vel_rl.shape) * velocity
        acc_rl = np.zeros(vel_rl.shape)
    else:
        vel_rl *= vel_scale
        acc_rl *= vel_scale

    psi_rl = remove_psi_step(psi_rl)

    dpsi_rl = get_dpsi(psi_rl, s_rl, vel_rl)

    return list((s_rl, vel_rl, raceline, psi_rl, dpsi_rl, acc_rl))


def remove_psi_step(psi_rl):
    """Remove jump in heading values (-pi to +pi)."""
    step = np.diff(psi_rl)
    max_step = np.argmax(abs(step))
    if step[max_step] > 0:
        psi_rl[max_step + 1 :] -= 2 * np.pi
    else:
        psi_rl[max_step + 1 :] += 2 * np.pi
    if np.min(psi_rl) < -6.5:
        psi_rl += 2 * np.pi
    return psi_rl


def get_dpsi(psi_rl, s_rl, vel_rl):
    """Get derivate of yaw angle psi, i.e. dpsi."""
    if np.max(vel_rl) == 0.0:
        dts = np.diff(s_rl) / 0.01
    else:
        dts = np.diff(s_rl) / vel_rl[:-1]
    delta_psi = np.diff(psi_rl)
    delta_psi = np.append(delta_psi, delta_psi[-1])
    dts = np.append(dts, dts[-1])
    for j in range(len(delta_psi)):
        if delta_psi[j] > np.pi / 2.0:
            delta_psi[j] -= 2 * np.pi
        elif delta_psi[j] < -np.pi / 2.0:
            delta_psi[j] += 2 * np.pi

    delta_psi /= dts
    dpsi_rl = savgol_filter(delta_psi, 5, 2, 0)

    return dpsi_rl


def get_track_kinematics(
    path_name, velocity=100, vel_scale=1.0, track_path=None, bool_get_yaw_curv=False
):
    """Get kinematics for a single path of the track.

    Kinematics is the list: [arc_length, vel_rl, raceline, psi_rl, dpsi_rl, acc_rl]
    Possible Inputs:
        track_name = "trackboundary_right", "trackboundary_left", "centerline", "glob_optimal_raceline", "raceline"
    """
    if "raceline" in path_name and "optimal" in path_name:
        return get_glob_raceline(track_path, bool_vel_const=False, vel_scale=vel_scale)
    if path_name == "raceline":
        return get_glob_raceline(track_path, bool_vel_const=True, velocity=velocity)

    tracks_arrays = get_track_paths(track_path, bool_all_kinematics=True)

    if bool_get_yaw_curv:
        return tracks_arrays[5], tracks_arrays[6]

    if "left" in path_name:
        idx = 3
    elif "right" in path_name:
        idx = 2
    else:  # centerline
        idx = 1

    # use heading from refline
    psi = tracks_arrays[5]

    arc_length = tracks_arrays[0]
    line_path = tracks_arrays[idx]

    vel_rl = np.ones(len(line_path)) * velocity
    acc_rl = np.zeros(len(line_path))

    psi = remove_psi_step(psi)

    dpsi_rl = get_dpsi(psi, arc_length, vel_rl)

    return [arc_length, vel_rl, line_path, psi, dpsi_rl, acc_rl]
