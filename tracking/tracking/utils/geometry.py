"""Geometric helper functions."""
import numpy as np


def pi_range(yaw: float) -> float:
    """Clip yaw to (-pi, +pi].

    Args:
        yaw (float): yaw angle in rad

    Returns:
        float: clipped yaw angle in rad
    """
    if yaw <= -np.pi:
        yaw += 2 * np.pi
        return yaw
    if yaw > np.pi:
        yaw -= 2 * np.pi
        return yaw
    return yaw


def rotate_loc_glob(
    local_matrix: np.ndarray, rot_angle: float, matrix=True
) -> np.ndarray:
    """Rotates matrices from local (vehicle coordinates) to global coordinates.

    Angle Convention:
    yaw = 0: local x-axis parallel to global y-axis
    yaw = -np.pi / 2: local x-axis parallel to global x-axis --> should result in np.eye(2)
    2D only

    Args:
        local_matrix (np.ndarray): local matrix or vector
        rot_angle (float): rotation angle in rad
        matrix (bool, optional): If false only vector rotation. Defaults to True.

    Returns:
        np.ndarray: Global matrix or vector.

    """
    rot_mat = np.array(
        [
            [-np.sin(rot_angle), -np.cos(rot_angle)],
            [np.cos(rot_angle), -np.sin(rot_angle)],
        ]
    )

    mat_temp = np.matmul(rot_mat, local_matrix)

    if matrix:
        return np.matmul(mat_temp, rot_mat.T)

    return mat_temp


def rotate_glob_loc(
    global_matrix: np.ndarray, rot_angle: float, matrix=True
) -> np.ndarray:
    """Rotate matrices from global to local coordinates (vehicle coordinates).

    Angle Convention:
    yaw = 0: local x-axis parallel to global y-axis
    yaw = -np.pi / 2: local x-axis parallel to global x-axis --> should result in np.eye(2)

    Args:
        global_matrix (np.ndarray): global matrix or vector
        rot_angle (float): rotation angle in rad
        matrix (bool, optional): If false vector rotation only. Defaults to True.

    Returns:
        np.ndarray: local matrix
    """
    rot_mat = np.array(
        [
            [-np.sin(rot_angle), np.cos(rot_angle)],
            [-np.cos(rot_angle), -np.sin(rot_angle)],
        ]
    )

    mat_temp = np.matmul(rot_mat, global_matrix)

    if matrix:
        return np.matmul(mat_temp, rot_mat.T)

    return mat_temp


def angle_between_loc(v1: np.ndarray) -> float:
    """Return the angle in radians between vectors "v1" and "v_ref" in local coordinates.

    v_ref = np.array([0, 1]) pointing to the left of the vehicle

    Args:
        v1 np.ndarray: vector1 in 2DDefaults to np.array([0, 1]).

    Returns:
        float: angle in rad.
    """
    if np.linalg.norm(v1) == 0:
        return np.arccos(0)

    v_ref = np.array([0, 1])

    if v1[0] > 0:
        return -np.arccos(
            np.clip(
                np.dot(v1, v_ref) / np.linalg.norm(v1) / np.linalg.norm(v_ref),
                -1.0,
                1.0,
            )
        )

    return np.arccos(
        np.clip(
            np.dot(v1, v_ref) / np.linalg.norm(v1) / np.linalg.norm(v_ref),
            -1.0,
            1.0,
        )
    )
