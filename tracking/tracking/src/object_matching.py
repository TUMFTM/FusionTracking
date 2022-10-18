"""Main functionalities for multi-sensor object tracking."""
# pylint: disable=C0413,W0703,W1202
import logging
import copy

import numpy as np
from scipy.optimize import linear_sum_assignment


class ObjectMatching:
    """Class of object machting."""

    __slots__ = (
        "__logger",
        "__object_id",
        "__matched_obj",
        "__unmatched_new_obj",
        "__unmatched_old_obj",
        "__death_candidates",
        "__n_init_counter",
        "__n_add_counter",
        "__matching_params",
        "__active_sensors",
        "__match_dict",
    )

    def __init__(self, matching_params: dict, sensor_models: dict):
        """Initialize class.

        matching_params:
            max_match_dict in m
            bool_all_sensors_seen: defines if single sensor removal possible
        sensor_models:
            sensor specifications
        """
        self.__logger = logging.getLogger("msg_logger")
        self.__object_id = 0
        self.__matched_obj = {}
        self.__unmatched_new_obj = {}
        self.__unmatched_old_obj = {}
        self.__death_candidates = {k: 0 for k in range(20)}
        self.__n_init_counter = {
            key: val.get("n_init_counter", 3) for key, val in sensor_models.items()
        }
        self.__n_add_counter = {
            key: val.get("n_add_counter", 1) for key, val in sensor_models.items()
        }
        self.__matching_params = matching_params
        self.__matching_params["last_time_seen_ns"] = int(
            self.__matching_params["last_time_seen_s"] * 1e9
        )
        self.__active_sensors = list(sensor_models)

        self.check_sensors()

        self.__match_dict = {sensor_str: [] for sensor_str in self.__active_sensors}

    @property
    def death_candidates(self):
        """Getter method of death_candidates."""
        return self.__death_candidates

    @property
    def match_dict(self):
        """Getter method of match_dict."""
        return self.__match_dict

    def reset_match_dict(self):
        """Reset log match dict."""
        self.__match_dict = {sensor_str: [] for sensor_str in self.__active_sensors}

    def check_sensors(self):
        """Check if active sensors are allowed to remove."""
        if set(self.__active_sensors).intersection(
            self.__matching_params["sr_sensors"]
        ) or set(self.__active_sensors).intersection(
            self.__matching_params["lr_sensors"]
        ):
            return

        self.__matching_params["sr_sensors"] += self.__active_sensors
        self.__matching_params["lr_sensors"] += self.__active_sensors

        self.__logger.warning(
            "REMOVE: No active sensors permitted to remove, all added"
        )
        self.__logger.info(
            "self.__matching_params['sr_sensors'] = {}".format(
                self.__matching_params["sr_sensors"]
            )
        )
        self.__logger.info(
            "self.__matching_params['lr_sensors'] = {}".format(
                self.__matching_params["lr_sensors"]
            )
        )

    def reset(self):
        """Reset class member."""
        self.__matched_obj = {}
        self.__unmatched_new_obj = {}

    def match_by_id(self, old_ids: list, new_object_list: list, sensor_str: str):
        """Match objects by id, i.e. detection send id for each object."""
        old_obj_indices = []
        for new_obj in new_object_list:
            # match with old object
            if new_obj["id"] in old_ids:
                self.__matched_obj[new_obj["id"]] = new_obj

                # increase death counter
                if (
                    self.__death_candidates[new_obj["id"]]
                    < self.__matching_params["n_max_counter"]
                ):
                    self.__death_candidates[new_obj["id"]] += self.__n_add_counter[
                        sensor_str
                    ]

                old_obj_indices.append(old_ids.index(new_obj["id"]))
            # create new object
            else:
                self.init_new_object(new_obj, sensor_str=sensor_str, v2v=True)

        return old_obj_indices

    def init_new_object(self, new_object: dict, sensor_str: str, v2v: bool = False):
        """Initialize new object."""
        if v2v:
            self.__unmatched_new_obj[new_object["id"]] = new_object
        else:
            self.__unmatched_new_obj[self.__object_id] = new_object

        # add to death counter
        self.__death_candidates[self.__object_id] = self.__n_init_counter[sensor_str]

        self.__logger.info(
            "COUNTER INIT:  sensor = {}, counter = {}, obj_id = {}".format(
                sensor_str,
                self.__death_candidates[self.__object_id],
                self.__object_id,
            )
        )

        # increase running object id
        if v2v:
            self.__object_id = max(new_object["id"], self.__object_id) + 1
        else:
            self.__object_id += 1

    @staticmethod
    def get_dist_mat(est_object_val_list: list, new_object_list: list):
        """Get distance matrix by linear sum assignment."""
        dist_mat = np.zeros([len(new_object_list), len(est_object_val_list)])

        for i, obj_vals in enumerate(est_object_val_list):
            x_obj = np.array(obj_vals["state"][:2])
            for j, obj_j in enumerate(new_object_list):
                est_x_obj = np.array(obj_j["state"][:2])
                err_eucl = np.linalg.norm(x_obj[:2] - est_x_obj[:2])
                dist_mat[j, i] = err_eucl

        return dist_mat

    def match_by_dist(
        self,
        est_object_dict: dict,
        new_object_list: list,
        sensor_str: str,
    ):
        """Match object by euclidean distance."""
        # get distance matrix, pairwise
        dist_mat = self.get_dist_mat(list(est_object_dict.values()), new_object_list)
        old_ids = list(est_object_dict)

        # conduct linear sum assigment
        new_obj_indices, old_obj_matches = linear_sum_assignment(dist_mat)

        # back to list to handle object matching
        old_obj_matches = old_obj_matches.tolist()
        new_obj_indices = new_obj_indices.tolist()

        # create new objects
        _ = [
            self.init_new_object(new_object, sensor_str=sensor_str)
            for idx, new_object in enumerate(new_object_list)
            if idx not in new_obj_indices
        ]

        # match
        rm_idxes = []
        for idx_new, idx_old in zip(new_obj_indices, old_obj_matches):

            is_out_of_reach = (
                dist_mat[idx_new, idx_old] > self.__matching_params["max_match_dist_m"]
            )

            if is_out_of_reach:
                # no valid match
                self.init_new_object(new_object_list[idx_new], sensor_str=sensor_str)
                rm_idxes.append(idx_old)
            else:
                # valid match
                self.__matched_obj[old_ids[idx_old]] = new_object_list[idx_new]

                if (
                    self.__death_candidates[old_ids[idx_old]]
                    < self.__matching_params["n_max_counter"]
                ):
                    self.__death_candidates[old_ids[idx_old]] += self.__n_add_counter[
                        sensor_str
                    ]

                # Logging
                self.__match_dict[sensor_str].append((idx_new, old_ids[idx_old]))

        _ = [old_obj_matches.remove(idx_old) for idx_old in rm_idxes]
        return old_obj_matches

    def __call__(
        self,
        est_object_dict: dict,
        new_object_list: list,
        sensor_str: str,
    ):
        """Match objects based on hungarian method.

        try to match the objects from the new and old lists
        if an old object is not detected it will removed after predifined timesteps
        if an new object could not match with an old object it is added
        to the list and counted how often it appears

        Returns:
            _ _ _ _
            self.__matched_obj: list
                objects which are with the same id detected in the old_object_list
                as well in the new_object_list self.__unmatched_new_obj: list
                objects which are not in the old_objects_list but were detected
            self.__unmatched_old_obj: list
                objects which are in the old_objects_list but were not detected in this time step
        """
        if not new_object_list:
            self.__unmatched_old_obj = copy.deepcopy(est_object_dict)
            return {}, {}

        self.reset()

        # match by id
        if "id" in new_object_list[0].keys():
            old_obj_matches = self.match_by_id(
                old_ids=list(est_object_dict),
                new_object_list=new_object_list,
                sensor_str=sensor_str,
            )

        # match by distance
        else:
            old_obj_matches = self.match_by_dist(
                est_object_dict=est_object_dict,
                new_object_list=new_object_list,
                sensor_str=sensor_str,
            )

        # handle umatched objects
        self.__unmatched_old_obj = {
            old_obj_id: old_object
            for idx, (old_obj_id, old_object) in enumerate(est_object_dict.items())
            if idx not in old_obj_matches
        }

        return self.__matched_obj, self.__unmatched_new_obj

    def init_blank(self, new_object_list: list, sensor_str: str):
        """Initialize without any previous seen object.

        No matching required.
        """
        if self.__unmatched_new_obj:
            self.__logger.info(
                "reseting __unmatched_new_obj, value was: {}".format(
                    self.__unmatched_new_obj
                )
            )
            self.reset()
        for new_object in new_object_list:
            self.init_new_object(
                new_object=new_object,
                sensor_str=sensor_str,
                v2v=bool("id" in new_object),
            )

        return self.__unmatched_new_obj

    def check_old_objects(
        self,
        obs_storage: dict,
        old_obj_dict: dict,
        obj_filters: dict,
        est_obj_dict: dict,
        skipped_obj_dict: dict,
        ego_pos: list,
        sensor_str: str,
        time_ns: int,
    ):
        """Remove objects from tracking if not detected anymore."""
        # remove unmatched objects
        for obj_id in list(self.__unmatched_old_obj):
            if obj_id in self.__death_candidates:
                dist_to_ego = np.sqrt(
                    (ego_pos[0] - old_obj_dict[obj_id]["state"][0]) ** 2
                    + (ego_pos[1] - old_obj_dict[obj_id]["state"][1]) ** 2
                )

                if (
                    dist_to_ego <= self.__matching_params["sr_threshhold_m"]
                    and sensor_str in self.__matching_params["sr_sensors"]
                ):
                    self.__death_candidates[obj_id] -= 1
                elif (
                    dist_to_ego > self.__matching_params["sr_threshhold_m"]
                    and sensor_str in self.__matching_params["lr_sensors"]
                ):
                    self.__death_candidates[obj_id] -= 1

                # check if maximal time since last detection is passed
                dt_ns = time_ns - obs_storage[obj_id]["last_time_seen_ns"]
                if dt_ns > self.__matching_params["last_time_seen_ns"]:
                    self.__logger.warning(
                        "DEATH: last time seen, "
                        "dt_obj = {:.02f} ms, dt_max = {:.02f} ms, obj_id = {}".format(
                            dt_ns / 1e6,
                            self.__matching_params["last_time_seen_ns"] / 1e6,
                            obj_id,
                        )
                    )
                    self.__death_candidates[obj_id] = -1

                if self.__death_candidates[obj_id] <= 0:
                    # remove all entries
                    del self.__death_candidates[obj_id]
                    del self.__unmatched_old_obj[obj_id]

                    try:
                        del self.__unmatched_new_obj[obj_id]
                    except Exception as exc:
                        self.__logger.warning(
                            "DEATH FAIL: __unmatched_new_obj '{}'".format(exc)
                        )

                    # variables in object tracking
                    del obs_storage[obj_id]
                    del old_obj_dict[obj_id]
                    del obj_filters[obj_id]
                    try:
                        del est_obj_dict[obj_id]
                    except Exception as exc:
                        self.__logger.warning(
                            "DEATH FAIL: est_obj_dict '{}'".format(exc)
                        )
                    if obj_id in skipped_obj_dict:
                        del skipped_obj_dict[obj_id]
                        self.__logger.warning(
                            "DEATH: remove from skipped_obj_dict, obj_id = {}".format(
                                obj_id
                            )
                        )

                    # Logging
                    self.__logger.info(
                        "DEATH: object removed, "
                        "obj_id = {}, t_track = {:d} ns".format(obj_id, time_ns)
                    )

        return self.__unmatched_old_obj
