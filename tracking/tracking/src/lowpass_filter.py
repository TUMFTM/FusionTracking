"""Class of lowpass object yaw and yaw Rate."""
from utils.geometry import pi_range


class LowpassFilter:
    """Class to lowpass orientation, i.e yaw and yaw Rate."""

    __slots__ = (
        "__tc_yaw",
        "__tc_yawrate",
        "__is_lowpass",
    )

    def __init__(self, params: dict):
        """Initialize class.

        params: Contains time constants for yaw and yaw Rate
        """
        self.__tc_yaw: float = params["tc_yaw"]
        self.__tc_yawrate: float = params["tc_yawrate"]
        self.__is_lowpass: bool = params["bool_lowpass"]

    def __call__(
        self, dyn_object: dict, obs_storage: dict, obj_id, storage_overwrite: bool
    ):
        """Apply low pass to yaw and yaw-rate to smooth physics-prediction."""
        if not self.__is_lowpass:
            return False

        if storage_overwrite:
            idx_storage = 1
        else:
            idx_storage = 0

        if len(obs_storage[obj_id]["state"]) >= idx_storage + 1:
            yaw_k = obs_storage[obj_id]["state"][idx_storage][2]
            yawrate_k = obs_storage[obj_id]["state"][idx_storage][4]

            yaw_k1 = pi_range(
                yaw_k + self.__tc_yaw * pi_range(dyn_object["state"][2] - yaw_k)
            )
            yawrate_k1 = (
                1.0 - self.__tc_yawrate
            ) * yawrate_k + self.__tc_yawrate * dyn_object["state"][4]

            if storage_overwrite:
                obs_storage[obj_id]["state"][0][2] = yaw_k1
                obs_storage[obj_id]["state"][0][4] = yawrate_k1
            else:
                dyn_object["state"][2] = yaw_k1
                dyn_object["state"][4] = yawrate_k1
                obs_storage[obj_id]["state"].appendleft(dyn_object["state"])
                obs_storage[obj_id]["t"].appendleft(dyn_object["t"])

        return True
