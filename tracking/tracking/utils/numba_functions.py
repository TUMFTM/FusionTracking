"""Collections of numba function for performance improvements."""
import numpy as np
from numba.pycc import CC

cc = CC("util_numba_compiled")
cc.verbose = True


@cc.export("predict_numba", "(f8[:, :], f8[:], f8, f8[:, :], f8[:, :], f8[:, :])")
def predict_numba(F, x, dt_s, Q, Q_xy, P):
    """Conduct prediction step for state and covariance."""
    F[0, 2] = -x[3] * dt_s * np.cos(x[2])
    F[0, 3] = -dt_s * np.sin(x[2])
    F[1, 2] = -x[3] * dt_s * np.sin(x[2])
    F[1, 3] = dt_s * np.cos(x[2])
    rot_mat = np.array(
        [
            [-np.sin(x[2]), -np.cos(x[2])],
            [np.cos(x[2]), -np.sin(x[2])],
        ]
    )
    Q[:2, :2] = np.dot(np.dot(rot_mat, Q_xy), rot_mat.T)
    x[0] -= x[3] * dt_s * np.sin(x[2])
    x[1] += x[3] * dt_s * np.cos(x[2])
    x[2] += x[4] * dt_s
    x[3] += x[5] * dt_s
    x[2] = np.clip(-np.pi, np.pi, x[2])
    P = np.dot(F, P).dot(F.T) + Q
    return F, x, Q, P


if __name__ == "__main__":
    cc.compile()
