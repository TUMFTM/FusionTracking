"""Class to handle object storage."""
import logging
import math

import numpy as np

from utils.geometry import pi_range, rotate_glob_loc
from utils.tracking_utils import ctrv_single

from .track_boundary import TrackBoundary


class PhysicsPrediction:
    """Class of physics prediction with ctrv model and track rails."""

    __slots__ = (
        "__logger",
        "__physics_based_input",
        "__in_pit_dict",
        "__pred_dict",
        "__pred_params",
        "__ego_raceline",
        "track_boundary",
        "pit_boundary",
        "__prediction_id",
        "__l_half_vehicle_m",
    )

    def __init__(
        self,
        all_params: dict,
        track_boundary_cls,
        pit_boundary_cls,
    ):
        """Initaliaze class.

        all_params: config from main_config.ini
        track_boundary_cls: class to check track boundaries
        pit_boundary_cls: class to check pit boundaries
        """
        self.__logger = logging.getLogger("msg_logger")
        self.__physics_based_input = {}
        self.__in_pit_dict = {}
        self.__pred_dict = None
        self.__prediction_id = 0

        self.__pred_params = all_params["TRACKING"]["prediction"]
        self.__pred_params["use_raceline"] = all_params["TRACKING"]["prediction"][
            "bool_use_raceline"
        ]
        if self.__pred_params["use_raceline"]:
            self.__logger.info(
                "PHYS PRED: use_raceline = {}, str = {}".format(
                    self.__pred_params["use_raceline"],
                    all_params["TRACKING"]["prediction"]["obj_raceline"],
                )
            )

        self.__l_half_vehicle_m = all_params["MISCS"]["l_vehicle_m"] / 2.0

        self.__ego_raceline = TrackBoundary(
            all_params=all_params,
            track_path=self.__pred_params["ego_raceline_path"],
        )

        self.create_changeable_params()

        self.track_boundary = track_boundary_cls
        self.pit_boundary = pit_boundary_cls

    @property
    def pred_dict(self):
        """Get private class member."""
        return self.__pred_dict

    @property
    def prediction_id(self):
        """Get private class member."""
        return self.__prediction_id

    @property
    def pred_params(self):
        """Get private class member."""
        return self.__pred_params

    @pred_params.setter
    def pred_params(self, new_dict: dict):
        """Setter method."""
        self.__pred_params.update(new_dict)

        self.create_changeable_params()

        self.__logger.info(
            "SWITCH MODE: new cone_angle_deg = {}, half_rad = {}".format(
                self.__pred_params["cone_angle_deg"],
                self.__pred_params["cone_angle_half_rad"],
            )
        )

    def check_params(self):
        """Check if params are consistent."""
        if (
            self.__pred_params["n_steps_phys_pred"]
            < self.__pred_params["rail_pred_switch_idx"]
        ):
            self.__logger.error(
                "n_steps_phys_pred < rail_pred_switch_idx,"
                "adjusting n_steps_phys_pred to {:d}".format(
                    self.__pred_params["rail_pred_switch_idx"]
                )
            )
            self.__pred_params["n_steps_phys_pred"] = self.__pred_params[
                "rail_pred_switch_idx"
            ]

    def create_changeable_params(self) -> None:
        """Create further keys for physics prediction."""
        self.check_params()

        self.__pred_params["physpred_range"] = range(
            self.__pred_params["n_steps_phys_pred"]
        )

        # Rail-based prediction
        self.__pred_params["railpred_steps_s_add"] = np.arange(
            0.0,
            (
                self.__pred_params["n_steps_rail_pred"]
                - self.__pred_params["rail_pred_switch_idx"]
            )
            * self.__pred_params["step_size_s"]
            + 1e-3,
            self.__pred_params["step_size_s"],
        )
        self.__pred_params["railpred_steps_s"] = np.arange(
            0.0,
            self.__pred_params["n_steps_rail_pred"] * self.__pred_params["step_size_s"]
            + 1e-3,
            self.__pred_params["step_size_s"],
        )

        self.__pred_params["cone_angle_half_rad"] = (
            self.__pred_params["cone_angle_deg"] * np.pi / 180.0 / 2.0
        )

    def __call__(
        self,
        obs_storage: dict,
        obj_filters: dict,
        ego_state: dict,
    ):
        """Predict objects ahead.

        Physics-Based Short-Term Prediction
        Args:
            _ _ _ _
            object_list: list of dicts, each dict describing an object
            with keys ["t", "x", "y", "yaw"]

            Returns:
            _ _ _ _
            prediction_dict: dict {id: [dict((), ..), t_abs_perception]}
            each dict entry contains the prediction values as dict [0], perception time [1]
        """
        # Only predict & prepare data-based prediction if some new
        # information was received (detection or ego-state)
        self.__pred_dict = None
        self.__in_pit_dict = {}
        self.__physics_based_input = {}
        if "state" not in ego_state:
            self.__logger.warning("MISSING EGO STATE: no prediction")
            return

        if obs_storage:
            self.prepare_prediction(
                obs_storage=obs_storage,
                ego_state=ego_state,
            )

            if self.__physics_based_input:
                self.__pred_dict = {}
                self.calculate_prediction(
                    obj_filters=obj_filters,
                )

        else:
            self.__logger.warning(
                "NO PREDICTION: actives objects = {}".format(list(obs_storage.keys()))
            )

    def prepare_prediction(self, obs_storage: dict, ego_state: dict):
        """Create inputs list for physics- and data-based prediction.

        _extended_summary_

        Args:
            obs_storage (dict): Storage of object states, key is obj id, values are timestamps and kinematic states.
            ego_state (dict): Dict of ego state including kinematic state and time stamp
        """
        # get predictable objects
        for obj_id, obs in obs_storage.items():
            if obs["num_measured"] < self.__pred_params["n_min_meas"]:
                continue

            if obj_id != "ego":
                is_obj_valid = not self.ignore_behind_cone(
                    obj_pose=np.array(obs["state"][0][:3]),
                    ego_pose=np.array(ego_state["state"][:3]),
                )
            else:
                is_obj_valid = True

            # Check if vehicle is on pit lane but not on track (overlapping maps)
            bool_out_of_pit, _, _ = self.pit_boundary.track_fn_single(
                translation=obs["state"][0][:2]
            )
            bool_out_of_track, _, _ = self.track_boundary.track_fn_single(
                translation=obs["state"][0][:2]
            )
            if bool_out_of_track and bool_out_of_pit:
                is_obj_valid = False

            # get input to physics-based prediction
            self.__physics_based_input[obj_id] = {
                "t_abs_perception": obs["t"][0],
                "state": obs["state"][0],
                "valid": is_obj_valid,
                "out_of_track": bool_out_of_track,
                "out_of_pit": bool_out_of_pit,
                "prediction_id": self.__prediction_id,
            }

            self.__prediction_id += 1

    def calculate_prediction(self, obj_filters: dict):
        """Predict a batch of multiple objects and timesteps at once.

        Returns:
        _ _ _ _ _
        self.__pred_dict: dict
            dict with entries for each object within his predicted state values
        """
        for obj_id, obj_vals in self.__physics_based_input.items():
            if obj_id in obj_filters:
                x_obj = obj_filters[obj_id].x
            else:
                x_obj = obj_vals["state"]

            if obj_vals["valid"]:
                pos_preds = self.ctrv_steps(x_obj=x_obj)
                pos0_rail = pos_preds[self.__pred_params["rail_pred_switch_idx"], :2]
                # get rail speed
                v0_rail = max(
                    0.0, pos_preds[self.__pred_params["rail_pred_switch_idx"], 3]
                )

                if self.__pred_params["bool_rail_accl"]:
                    pred_points = self.rail_accl_based_w_clipping(
                        pos_preds=pos_preds, v0_rail=v0_rail, obj_id=obj_id, x_obj=x_obj
                    )
                else:
                    pred_points = v0_rail * self.__pred_params["railpred_steps_s_add"]

                if not obj_vals["out_of_pit"] and obj_vals["out_of_track"]:
                    self.__in_pit_dict[obj_id] = True
                    railpred, railyaw = self.pit_boundary.path_along_track(
                        translation=pos0_rail, pred_points=pred_points
                    )
                    self.__logger.info(
                        "PIT LANE: x, y = {:.02f}, {:.02f}, obj_id = {}".format(
                            x_obj[0],
                            x_obj[1],
                            obj_id,
                        )
                    )
                else:
                    self.__in_pit_dict[obj_id] = False
                    if obj_id == "ego":
                        railpred, railyaw = self.__ego_raceline.path_along_track(
                            translation=pos0_rail,
                            pred_points=pred_points,
                            use_raceline=True,
                        )
                    else:
                        railpred, railyaw = self.track_boundary.path_along_track(
                            translation=pos0_rail,
                            pred_points=pred_points,
                            use_raceline=self.__pred_params["use_raceline"],
                        )

                railpred = np.vstack(
                    [
                        pos_preds[: self.__pred_params["rail_pred_switch_idx"], :2],
                        railpred,
                    ]
                )
                railyaw = np.concatenate(
                    [
                        pos_preds[: self.__pred_params["rail_pred_switch_idx"], 2],
                        railyaw,
                    ]
                )

                self.__pred_dict[obj_id] = {
                    "t_abs_perception": obj_vals["t_abs_perception"],
                    "t": self.__pred_params["railpred_steps_s"],
                    "x": railpred[:, 0],
                    "y": railpred[:, 1],
                    "heading": railyaw,
                    "valid": obj_vals["valid"],
                    "prediction_id": obj_vals["prediction_id"],
                    "prediction_type": "rail",
                }
            else:
                # invalid object, i.e. inside ignore cone
                self.__pred_dict[obj_id] = {
                    "t_abs_perception": obj_vals["t_abs_perception"],
                    "t": 0.0,
                    "x": x_obj[0],
                    "y": x_obj[1],
                    "heading": x_obj[2],
                    "valid": obj_vals["valid"],
                    "prediction_id": obj_vals["prediction_id"],
                    "prediction_type": "invalid",
                }

    def ignore_behind_cone(self, obj_pose: np.ndarray, ego_pose: np.ndarray):
        """Ignore vehicles behind ego for prediction if inside specific cone.

        Cone is spaned from center of rear-axle (cog - length / 2.0)

        cone_angle = Total Angle of Cone. 0.5 per side (right, left)

        return bool: True if vehicle is ignored, i.e. inside cone

        Test:
            ego_pose = np.array([-300, -1800, -np.pi])
            cone_angle = 60 / 180 * np.pi
            veh_length = 4.921
            obj_pose = np.array(
                [-300 - 20 * np.tan(cone_angle/2.0),
                -1780 + veh_length / 2.0, -np.pi]
            )
        """
        # rotate obj pose in to ego-coordinate system
        loc_obj_pos = rotate_glob_loc(
            obj_pose[:2] - ego_pose[:2], ego_pose[2], matrix=False
        )

        # add half vehicle length
        loc_obj_pos[0] += self.__l_half_vehicle_m

        # obect inside cone -> do ignore
        strstr = "INSIDE"
        bool_return = True

        # object in front of object / within safety distance -> do no ignore
        if loc_obj_pos[0] > -self.__pred_params["cone_offset_m"]:
            strstr = "DISTANCE"
            bool_return = False

        # check if inside cone
        obj_angle = pi_range(math.atan2(loc_obj_pos[1], loc_obj_pos[0]) - np.pi)
        if abs(obj_angle) > self.__pred_params["cone_angle_half_rad"]:
            strstr = "ANGLE"
            bool_return = False

        if bool_return:
            self.__logger.info(
                "CONE: do_ignore = {}, reason = {}, "
                "cone_angle_half_rad = {:.02f} rad, cone_offset_m = {:.02f} m, "
                "obj_pose = {}, ego_pose = {}".format(
                    bool_return,
                    strstr,
                    self.__pred_params["cone_angle_half_rad"],
                    self.__pred_params["cone_offset_m"],
                    obj_pose,
                    ego_pose,
                )
            )
        return bool_return

    def ctrv_steps(
        self,
        x_obj: np.ndarray,
    ):
        """Conduct ctrv steps."""
        pos_preds = np.zeros([self.__pred_params["n_steps_phys_pred"] + 1, len(x_obj)])
        pos_preds[0, :] = x_obj
        for j in self.__pred_params["physpred_range"]:
            pos_preds[j + 1, :] = self.ctrv_step_j(
                x_in=pos_preds[j, :],
                dt_s=self.__pred_params["step_size_s"],
            )
            pos_preds[j + 1, 2] = pi_range(pos_preds[j + 1, 2])

        return pos_preds

    def ctrv_step_j(self, x_in: np.ndarray, dt_s: float):
        """Calculate Physics-Based Prediction Step.

        Args:
            x_in (np.ndarry): Kinematic state at t.
            dt_s (float): Time steps size in seconds.

        Returns:
            np.ndarry: Kinematic state at t+dt_s.
        """
        x_out = ctrv_single(x_in, dt_s)
        x_out[2] = pi_range(x_out[2])

        return x_out

    def rail_accl_based_w_clipping(
        self, pos_preds: np.ndarray, v0_rail: float, obj_id, x_obj
    ):
        """Calculate rail based path profile by v * t + a * t ** 2 / 2."""
        a0_rail = max(
            -v0_rail / self.__pred_params["railpred_steps_s_add"][-1],
            pos_preds[self.__pred_params["rail_pred_switch_idx"], 5],
        )

        if abs(a0_rail) > 10.0:
            self.__logger.warning(
                "Accleration for rail-based prediction is"
                " {:.03f}, obj_id = {:s}".format(x_obj[5], str(obj_id))
            )
        return (
            v0_rail * self.__pred_params["railpred_steps_s_add"]
            + a0_rail * self.__pred_params["railpred_steps_s_add"] ** 2 / 2.0
        )
