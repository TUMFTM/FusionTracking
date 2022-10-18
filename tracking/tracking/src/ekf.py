"""Implementation of the Extended-Kalman Filter."""
import numpy as np

from utils.numba_functions import predict_numba
from utils.geometry import pi_range, rotate_loc_glob, angle_between_loc


class EKF:
    """Extendend Kalman Filter Class with CTRV-State Equations."""

    def __init__(self, x_init: list, params: dict, sensor_init: str, ego_init: dict):
        """Class Initialization."""
        # State Variables
        self.dt_s = params["timestep_s"]  # Filter Timestep in s
        self._n_dof = params["P_init"].shape[0]
        self.x = np.array(x_init)
        self.x[2] = pi_range(self.x[2])
        self.F = np.eye(self._n_dof)  # State Jacobian Matrix Variables
        self.F[2, 4] = self.dt_s
        self.ctrv_deriv()  # Update State Jacobian Matrix

        # Update Variables
        self.K = np.zeros(self.x.shape)  # Kalman gain
        self.S = np.zeros((self._n_dof, self._n_dof))  # System Uncertainty
        self.SI = np.zeros((self._n_dof, self._n_dof))  # Inverse System Uncertainty
        self._I = np.eye(self._n_dof)  # Identiy Matrix

        # Measurement Variables
        self._measure_covs_dict = params["measure_covs_dict"]
        self.y = np.zeros(
            (len(self._measure_covs_dict[sensor_init]), 1)
        )  # Measurement Residual
        self.H = np.eye(self._n_dof)
        self.H_indices = params["H_indices"]
        self.mat_out(sensor_init)  # Intialize Output Matrix
        self.z_meas = None
        ego_pos = np.array(ego_init["state"][:2])
        self.update_measurementnoise(ego_pose=ego_pos, x=self.x, sensor=sensor_init)

        # Initialization of the Q-matrix
        self.Q_glob = self.init_processnoise(params)
        self.Q = np.zeros((self._n_dof, self._n_dof))
        self.update_processnoise()
        self.P = params["P_init"]  # State Uncertainty
        self.P[:2, :2] = rotate_loc_glob(self.P[:2, :2], self.x[2])

        # Storage Variables
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)
        self.x_post = None
        self.P_post = None
        self.S = None

        # Delay Compensation
        self.num_updates = 0
        self.bool_oldest_detect = False
        self.idx_hist = -1
        self.x_hist = None
        self.P_hist = None
        self.egopos_hist = None

    def predict(self):
        """Conduct prediction step for state and covariance."""
        self.F, self.x, self.Q, self.P = predict_numba(
            self.F, self.x, self.dt_s, self.Q, self.Q_glob[:2, :2], self.P
        )

    def update(self, z_meas, sensor_type, ego_pos, measure_keys):
        """Conduct update step for state and covariance after receiving a measurement."""
        if "yaw" in measure_keys:
            z_meas[2] = pi_range(z_meas[2])
        self.z_meas = np.copy(z_meas)

        # Create output jacobian (rectangle matrix): Depending on measured variables
        self.mat_out(sensor_type)

        if self.bool_oldest_detect:
            x_arr = self.x_hist
            P_arr = self.P_hist
            self.bool_oldest_detect = False
            log_post = True
            ego_pos = self.egopos_hist
        else:
            x_arr = self.x
            P_arr = self.P
            log_post = False

        self.update_measurementnoise(ego_pose=ego_pos, x=x_arr, sensor=sensor_type)

        # Save Prior
        self.x_prior = np.copy(x_arr)
        self.P_prior = np.copy(P_arr)

        # Residuum
        self.y = z_meas - x_arr[list(self.H_indices[sensor_type][1])]
        if "yaw" in measure_keys:
            self.y[2] = pi_range(self.y[2])

        # Kalman Gain
        PHT = np.dot(P_arr, self.H.T)  # temporary variable, used twice
        self.S = np.dot(self.H, PHT) + self.R
        self.SI = np.linalg.inv(self.S)
        self.K = np.dot(PHT, self.SI)

        # State Update
        self.x = x_arr + np.dot(self.K, self.y)
        self.x[2] = pi_range(self.x[2])

        # Covariance Update
        I_KH = self._I - np.dot(self.K, self.H)  # temporary variable, used once
        self.P = np.dot(I_KH, P_arr)

        if log_post:
            # Logging: Store historic state update
            self.x_post = np.copy(self.x)
            self.P_post = np.copy(self.P)
        else:
            # Do not store post values as they are equal to x and P
            self.x_post = None
            self.P_post = None

    def ctrv_deriv(self):
        """Determine Jacobian-Matrix for Current Object State.

        State Vector x = [x-pos, y-pos, yaw, velocity, yawrate, acceleration]
        Note: yaw is 0.0 in global north direction (y-axis), -3.141 for global east (x-axis)
        """
        self.F[0, 2] = -self.x[3] * self.dt_s * np.cos(self.x[2])
        self.F[0, 3] = -self.dt_s * np.sin(self.x[2])
        self.F[1, 2] = -self.x[3] * self.dt_s * np.sin(self.x[2])
        self.F[1, 3] = self.dt_s * np.cos(self.x[2])

    def mat_out(self, sensor_str):
        """Get output of kinematic model, i.e. measurement values."""
        meas_shape = (len(self.H_indices[sensor_str][0]), self._n_dof)
        if meas_shape == self.H.shape:
            return

        self.H = np.zeros(meas_shape)
        self.H[self.H_indices[sensor_str][0], self.H_indices[sensor_str][1]] = 1.0

    def update_processnoise(self):
        """Update Q-Matrix based of objects yaw-angle."""
        self.Q = np.copy(self.Q_glob)
        self.Q[:2, :2] = rotate_loc_glob(self.Q[:2, :2], self.x[2])

    def update_measurementnoise(self, ego_pose, x, sensor):
        """Update R-Matrix."""
        # get R-Matrix
        self.R = np.copy(self._measure_covs_dict[sensor])
        # rotate
        rel_yaw = pi_range(angle_between_loc(x[:2] - ego_pose))
        self.R[:2, :2] = rotate_loc_glob(self.R[:2, :2], rel_yaw)

    def init_processnoise(self, params):
        """Initialize the Q-Matrix with filter time_step size.

        self.dt_s - filter time-step in s
        dx_std - Derivative of Covariance in x in (m/s), float
        dy_std - Derivative of Covariance in y in (m/s), float
        dyaw_std - Derivative of Covariance in yaw in (rad/s), float
        dv_std - Derivative of Covariance in v in (m/s**2), float
        dyaw_std - Derivative of Covariance in yaw in (rad/s**2), float
        da_std - Derivative of Covariance in yawrate in (m/s**3), float
        """
        return (
            np.diag(
                [
                    params["dx_std"] ** 2,
                    params["dy_std"] ** 2,
                    params["dyaw_std"] ** 2,
                    params["dv_std"] ** 2,
                    params["dyawrate_std"] ** 2,
                    params["da_std"] ** 2,
                ]
            )
            * self.dt_s**2
        )

    def reset(self):
        """Reset delay compensation values."""
        self.num_updates = 0
        self.bool_oldest_detect = False
        self.idx_hist = -1
        self.x_hist = None
        self.P_hist = None
        self.egopos_hist = None
        self.x_post = None
        self.P_post = None
