"""Calculate the NLO crystal characteristics."""

from collections.abc import Callable

import numpy as np

from pulselaser import sellmeier


def cut_angle_deg(
    input_wavelength_micron: float = 0.800,
    material: Callable[[float], tuple[float, float]] = sellmeier.beta_bbo,
) -> float:
    """Return the appropriate cutting angle of the NLO crystal.

    Parameters
    ----------
    input_wavelength_micron: float
        wave length of the input light
    material: Callable
        [TODO:description]

    Returns
    -------
    float
        appropriate cutting angle

    """
    no_1: float = material(input_wavelength_micron)[0]
    """ne_1: float = material(input_wavelength_micron)[1] """
    no_2: float = material(input_wavelength_micron / 2)[0]
    ne_2: float = material(input_wavelength_micron / 2)[1]

    return np.rad2deg(
        np.arcsin(
            np.sqrt(
                ((ne_2**2) * (no_2**2 - no_1**2)) / ((no_1**2) * (no_2**2 - ne_2**2)),
            ),
        ),
    )
