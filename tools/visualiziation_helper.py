"""Main script to visualize tracking logs."""
import sys
from matplotlib.patches import Ellipse
from matplotlib import transforms
import numpy as np


def confidence_ellipse(
    mu,
    cov,
    ax,
    n_std=3.0,
    facecolor="None",
    edgecolor="fuchsia",
    linestyle="--",
    label_ell=None,
    **kwargs
):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    mu_x = mu[0]
    mu_y = mu[1]

    pearson = cov[0, 1] / (np.sqrt(cov[0, 0] * cov[1, 1]) + sys.float_info.epsilon)
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linestyle=linestyle,
        alpha=0.5,
        **kwargs
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mu_x, mu_y)
    )

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)
    if label_ell is not None:
        ellipse.set(label=label_ell)
