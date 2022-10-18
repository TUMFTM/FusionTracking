"""Class to transform objects into global coordinates."""
import numpy as np

from utils.geometry import rotate_loc_glob, pi_range
from utils.tracking_utils import ctrv_single


class TransformObjects:
    """Returns every object in global coordinates."""

    __slots__ = ("__rear_ax_geoc_m",)

    def __init__(self, rear_ax_geoc_m: float):
        """Initialize class."""
        self.__rear_ax_geoc_m = rear_ax_geoc_m

    def __call__(
        self,
        new_object_list: list,
        yaw_from_track: bool,
        detection_timestamp_ns: int,
        ego_t: int,
        ego_state: np.ndarray,
    ):
        """Convert list of detected objects in global coordinates.

        Detection determines object position in reference to ego rear axle.
        All objects in prediction, planning, control are referenced to their geometrical center.
        So it is necessary to consider the distance between rear axle
        and geo-center in local coordinates.

        Args:
            new_object_list (list): List of detected objects in local coordinates
            detection_timestamp_ns (int): Timestamp of detection
            yaw_from_track (bool): If true yaw estimation from track map is added
            detection_timestamp_ns (int): Time stamp from sensor detection in ns
            ego_t (int): Time stamp of ego state in ns
            ego_state (np.ndarray): Kinematic ego state
        """
        dt_s = (detection_timestamp_ns - ego_t) / 1e9
        pred_ego_state = ctrv_single(ego_state, dt_s)

        for objs in new_object_list:
            local_pos = np.array(objs["state"][:2])
            local_pos[0] -= self.__rear_ax_geoc_m
            global_pos = (
                rotate_loc_glob(local_pos, pred_ego_state[2], matrix=False)
                + pred_ego_state[:2]
            )
            objs["state"][:2] = global_pos
            if not yaw_from_track and "yaw" in objs["keys"]:
                objs["state"][2] = pi_range(objs["state"][2] + pred_ego_state[2])
