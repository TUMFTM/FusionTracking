"""Helper functions to setup Objects class."""
import os
import copy
import datetime

from .logging_helper import get_msg_logger, DataLogging


def stamp2time(seconds: int, nanoseconds: int) -> int:
    """Convert ros2-timestamp to integer value in ns."""
    return seconds * 1000000000 + nanoseconds


def get_base_key(key: str):
    """Get base key of attacker or defender mode.

    Args:
        key (str): Key str of switching mode.

    Raises:
        ValueError: Raised if key is not specified for attacker and defender

    Returns:
        str: Tuple of key in defender and attacker mode
    """
    if key.endswith("_def"):
        return key.replace("_def", ""), key, key.replace("_def", "_atk")

    if key.endswith("_atk"):
        return key.replace("_atk", ""), key.replace("_atk", "_def"), key

    raise ValueError("Invalid key for switch mode entered: {}".format(key))


def setup_logging(path_dict: dict):
    """Set logger for data and message logging.

    Args:
        path_dict (dict): Dict containing absolute paths of param files
        all_params (dict): Dict with all params from main_config_tracking.ini
        module_state (int): Software module state, 30 is valid, 50 is soft emergency
    Returns:
        tuple: Tuple of message, data and latency logger
    """
    return (
        get_msg_logger(path_dict=path_dict),
        DataLogging(path_dict=path_dict),
        DataLogging(path_dict=path_dict, latency_log=True),
    )


def switch_params(main_class, switch_to_atk: bool) -> None:
    """Switch paramaters from defender to attacker and vice versa.

    Args:
        main_class ([ObjectTracking]): Main tracking class
        switch_to_atk (bool): Boolean to define is switch to attacker (True) or defender (False)
    """
    for section, sec_items in main_class.params.items():
        if not isinstance(sec_items, dict):
            continue

        for key in sec_items:
            if switch_to_atk and key.endswith("_atk"):
                base_key, _, ovr_key = get_base_key(key)
                strstr = "attacker"
            elif not switch_to_atk and key.endswith("_def"):
                base_key, ovr_key, _ = get_base_key(key)
                strstr = "defender"
            else:
                continue

            main_class.params[section][base_key] = copy.deepcopy(
                main_class.params[section][ovr_key]
            )

            main_class.msg_logger.info(
                "SWTICH MODE: {}, {}, {}, overwrite from {}".format(
                    strstr, section, base_key, ovr_key
                )
            )

            break


def overwrite_params(main_class, params_overwrite: dict, switch_only: bool):
    """Overwrite tracking parameters from docker_iac overwrite config.

    Args:
        main_class ([ObjectTracking]): Main tracking class
        params_overwrite ([dict]): Overwrite dict loaded from docker_iac overwrite config
    """
    for section, sec_items in params_overwrite.items():
        if section not in main_class.params.keys():
            main_class.msg_logger.warning("NOT FOUND: {}".format(section))
            continue

        for key, val in sec_items.items():
            if key.endswith(("_def", "_atk")):
                if switch_only:
                    base_key, def_key, atk_key = get_base_key(key)
                    if def_key not in main_class.params[section]:
                        main_class.params[section][def_key] = copy.deepcopy(
                            main_class.params[section][base_key]
                        )
                    if atk_key not in main_class.params[section]:
                        main_class.params[section][atk_key] = copy.deepcopy(
                            main_class.params[section][base_key]
                        )

                    main_class.msg_logger.info(
                        "SWTICH MODE: {}, {}, {}".format(section, key, val)
                    )
                else:
                    continue

            if isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    if sub_key in main_class.params[section][key].keys():
                        if main_class.params[section][key][sub_key] != sub_val:
                            old_val = main_class.params[section][key][sub_key]
                            new_val = sub_val
                            main_class.params[section][key][sub_key] = sub_val
                            main_class.msg_logger.info(
                                "OVERWRITE: {}, {}, {}, (old, new) = ({}, {})".format(
                                    section, key, sub_key, old_val, new_val
                                )
                            )
                    else:
                        main_class.msg_logger.warning(
                            "NOT FOUND: {}, {}, {}".format(section, key, sub_key)
                        )
            else:
                if key in main_class.params[section].keys():
                    if main_class.params[section][key] != val:
                        old_val = main_class.params[section][key]
                        new_val = val
                        main_class.params[section][key] = val
                        main_class.msg_logger.info(
                            "OVERWRITE: {}, {}, (old, new) = ({}, {})".format(
                                section, key, old_val, new_val
                            )
                        )
                else:
                    main_class.msg_logger.warning(
                        "NOT FOUND: {}, {}".format(section, key)
                    )


def check_sensor_specification(main_class):
    """Check specification of given sensor modality.

    Args:
        main_class (class): Tracking class

    Raises:
        IndexError: Raised if sensor spec is invalid.
        ValueError: Raised if naming of sensor modality is inconsistent.
    """
    active_sensors = main_class.params["SENSOR_SETUP"]["active_sensors"]

    if len(active_sensors) == 0:
        strstr = "### WARNING: No active sensors specified ###"
        print(strstr)
        main_class.msg_logger.warning(strstr)
    elif len(active_sensors) == 1 and not main_class.params["SENSOR_MODELS"][
        active_sensors[0]
    ].get("is_fallback", False):
        main_class.params["SENSOR_MODELS"][active_sensors[0]]["is_fallback"] = True
        main_class.msg_logger.warning(
            "single sensor mode ({}), setting remaining sensor as fallback sensor".format(
                active_sensors
            )
        )

    for sensor in active_sensors:
        # Remove yaw if from key and std_dev if not explicity desired ("yaw_from_track")
        if (
            not main_class.params["SENSOR_MODELS"][sensor]["get_yaw"]
            and "yaw" in main_class.params["SENSOR_MODELS"][sensor]["meas_vars"]
        ):
            yaw_idx = main_class.params["SENSOR_MODELS"][sensor]["meas_vars"].index(
                "yaw"
            )
            main_class.params["SENSOR_MODELS"][sensor]["meas_vars"].remove("yaw")
            del main_class.params["SENSOR_MODELS"][sensor]["std_dev"][yaw_idx]

        # Check for valid state variables
        for s_sens in main_class.params["SENSOR_MODELS"][sensor]["meas_vars"]:
            if s_sens not in main_class.params["TRACKING"]["state_vars"]:
                raise IndexError(
                    "invalid specification of sensor states, "
                    "{} not in tracking states {}, sensor: {}".format(
                        s_sens,
                        main_class.params["TRACKING"]["state_vars"],
                        sensor,
                    )
                )

        # Create H-matrix for Kalman Filter (y = H * x)
        try:
            if "H_indices" not in main_class.params["TRACKING"]["filter"]:
                main_class.params["TRACKING"]["filter"]["H_indices"] = {}
            main_class.params["TRACKING"]["filter"]["H_indices"][sensor] = list(
                zip(
                    *[
                        (
                            row_idx,
                            main_class.params["TRACKING"]["state_vars"].index(var_key),
                        )
                        for row_idx, var_key in enumerate(
                            main_class.params["SENSOR_MODELS"][sensor]["meas_vars"]
                        )
                    ]
                )
            )
        except ValueError:
            print(
                "VARIABLE NAMING VIOLATION: "
                "sensor keys = {} do not conform naming = {}, sensor: {}".format(
                    main_class.params["SENSOR_MODELS"][sensor]["meas_vars"],
                    main_class.params["TRACKING"]["state_vars"],
                    sensor,
                )
            )

    main_class.params["ACTIVE_SENSORS"] = {
        sensor: main_class.params["SENSOR_MODELS"][sensor] for sensor in active_sensors
    }

    assert ["x", "y", "yaw", "v", "yawrate", "a"] == main_class.params["TRACKING"][
        "state_vars"
    ], "invalid state variables"


def log_params(all_params: dict, msg_logger, module_state: int):
    """Log all params."""
    for main_key, main_vals in all_params.items():
        if isinstance(main_vals, dict):
            msg_logger.info("{:*^80}".format(" {:s} ".format(main_key)))
            for key, vals in main_vals.items():
                if isinstance(vals, dict):
                    msg_logger.info(f"{str(key): <30}:\t")
                    for sub_key, sub_vals in vals.items():
                        msg_logger.info(f"\t{str(sub_key): <26}:\t{sub_vals}")
                else:
                    msg_logger.info(f"{str(key): <30}:\t{vals}")

            msg_logger.info("{:*^80}".format(""))
        else:
            msg_logger.info(f"{str(main_key): <30}:\t{main_vals}")
    msg_logger.info("SIL - params: {}".format(all_params))

    # Log module state
    msg_logger.info(f"STATE: tracking state = {module_state}")


def create_path_dict():
    """Create a dict of all required paths.

    Args:
        node_path (str): absolute path to module

    Returns:
        dict: dict with paths
    """
    node_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    path_dict = {
        "node_path": node_path,
        "src_path": os.path.abspath(os.path.join("src", "mod_tracking")),
        "abs_log_path": os.path.join(
            node_path,
            "logs",
            datetime.date.today().strftime("%Y_%m_%d"),
            datetime.datetime.now().strftime("%H_%M_%S"),
        ),
        "tracking_config": os.path.join(
            node_path, "config", "main_config_tracking.ini"
        ),
        "tracking_config_overwrite": os.path.join(
            node_path,
            "config",
            "from_docker_iac",
            "mod_tracking",
            "main_config_tracking_overwrite.ini",
        ),
        "map_path": os.path.join(node_path, "data", "map_data"),
        "map_path_overwrite": os.path.join(
            node_path,
            "config",
            "from_docker_iac",
            "mod_local_planner",
            "inputs",
            "traj_ltpl_cl",
        ),
    }

    return path_dict
