"""Berek Polarization Compensator:  Model 5540 New Focus."""

import numpy as np

from pulselaser.sellmeier import mgf2


def retardance(lambda_micron: float, tilt_angle_degree: float) -> float:
    """Return the retardance R (in waves).

    Parameters
    ----------
    lambda_micron : float
        wave length of the light
    tilt_angle_degree : float
        the tiltangle


    Returns
    -------
    float
        _description_

    """
    theta = np.deg2rad(tilt_angle_degree)
    return (
        (2000 / lambda_micron)
        * np.sqrt(mgf2(lambda_micron)[0] ** 2 - np.sin(theta) ** 2)
        * (
            np.sqrt(
                (1 - (1 / mgf2(lambda_micron)[1] ** (2)) * np.sin(theta) ** 2)
                / (1 - (1 / mgf2(lambda_micron)[0] ** (2)) * np.sin(theta) ** 2),
            )
            - 1
        )
    )


def tilt_angle_deg(retardation_indicator: float) -> float:
    """Return the tilt angle of MgF2.

    Parameters
    ----------
    retardation_indicator : float
        indicator value

    Returns
    -------
    float
        tilt angle in degree.

    """
    theta_r_rad = np.pi / 4 - np.arcsin((50.22 - retardation_indicator) / 71)
    return np.rad2deg(theta_r_rad)
