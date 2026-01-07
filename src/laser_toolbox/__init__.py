"""Basic functions for pulselaser module."""

from typing import overload

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from .sellmeier import DISPERSION_FUNCS
from .types import ScalarOrArray, Scalar


def gaussian_pulse(
    t: NDArray[np.float64],
    fwhm: float,
    t0: float = 0,
) -> NDArray[np.float64]:
    """Gaussian function defined by FWHM.

    The height is unity.

    Parameters
    ----------
    t: NDArray[np.float64]
        time
    fwhm: float
        Full width at half maximum.
    t0: float
        The center offset.

    Returns
    -------
    NDArray[np.float64]
        [TODO:description]

    """
    sigma: float = fwhm / (2.0 * np.sqrt(np.log(2.0)))
    return np.exp(-((t - t0) ** 2) / sigma**2)


def sech2(x: float, x0: float, width: float) -> float:
    r"""Return :math:`\mathrm{sech}^2\left(\frac{x-x0}{\tau}\right)`.

    .. note::

    This function does not include the amplitude.

    Parameters
    ----------
    x: float
        x
    x0: float
        center position
    width: float
        width of the function :math:`\tau`. Not FWHM. (FWHM= :math:`1.7627 \tau` )

    Returns
    -------
    float

    """
    return (1 / np.cosh((x - x0) / width)) ** 2


def broadening(initial_width_fs: float, gdd: float) -> float:
    """Return pulse broadening due to GDD.

    Parameters
    ----------
    initial_width_fs: float
        initial pulse width (fs unit)
    gdd: float
        Group delay dispersion (fs^2 unit)

    Returns
    -------
    float
        the output pulse width (fs unit)

    """
    assert initial_width_fs > 0
    assert gdd > 0
    return (
        np.sqrt(initial_width_fs**4 + (gdd**2) * 16 * np.log(2) ** 2) / initial_width_fs
    )


def broadening_after_n(
    initial_width_fs: float,
    gdd: float,
    iteration: int = 1,
) -> float:
    """Return pulse broadening due to GDD after N iteration.

    Parameters
    ----------
    initial_width_fs: float
        initial pulse width (fs unit)
    gdd: float
        Group delay dispersion (fs^2 unit)
    iteration: int
        Number of iteration

    Returns
    -------
    float
        the output pulse width (fs unit)

    """
    assert isinstance(iteration, int)
    assert iteration > 0
    if iteration == 1:
        return broadening(initial_width_fs, gdd)
    return broadening(broadening_after_n(initial_width_fs, gdd, iteration - 1), gdd)


def gdd(
    input_pulse_duration_fs: float,
    output_pulse_duration_fs: float,
) -> float:
    """Return the GDD value of the optics.

    Parameters
    ----------
    input_pulse_duration_fs: float
        The duration of the input pulse
    output_pulse_duration_fs: float
        The duration of the output pulse

    Returns
    -------
    float
        GDD value

    """
    return (
        np.sqrt(output_pulse_duration_fs**2 - input_pulse_duration_fs**2)
        * input_pulse_duration_fs
        / (4 * np.log(2))
    )


@overload
def gvd(lambda_micron: Scalar, material: str) -> np.floating: ...


@overload
def gvd(lambda_micron: NDArray[np.floating], material: str) -> NDArray[np.floating]: ...


def gvd(
    lambda_micron: ScalarOrArray,
    material: str,
) -> np.floating | NDArray[np.floating]:
    """Return GVD in fs^2/mm units.

    Notes
    -----
    - Isotropic materials: shape (...)
    - Birefringent materials: shape (..., 2) with (o, e)

    """
    light_speed_micron_fs: float = 0.299792458
    try:
        disp_func = DISPERSION_FUNCS[material.lower()]
    except KeyError as exc:
        msg = f"Unknown material: {material}"
        raise ValueError(msg) from exc
    d2n = disp_func(lambda_micron, 2)  # second derivative
    assert not isinstance(d2n, sp.Expr)

    lm = np.asarray(lambda_micron, dtype=float)
    factor = lm**3 / (2 * np.pi * light_speed_micron_fs**2) * 1e3
    factor = factor[..., None] if np.ndim(d2n) == lm.ndim + 1 else factor
    return factor * d2n
