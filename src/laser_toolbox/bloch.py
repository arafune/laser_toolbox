"""Bloch equation.

To describe the temporal evolution of the excited state in two level system.
"""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from . import gaussian_pulse

if TYPE_CHECKING:
    from collections.abc import Callable

    from scipy.integrate._ivp import OdeResult


def bloch(  # noqa: PLR0913
    t: NDArray[np.float64],
    rho: NDArray[np.complex128],
    fwhm: float,
    t1: float,
    omega12_minus_omega: float,
    amplitude: float,
) -> NDArray[np.complex128]:
    r"""Bloch equation for two level system with a pulse excitation.

    Parameters
    ----------
    t
        time in fs unit.
    rho
        :math:`\rho_{22}` and :math:`\tilde{\rho}_{12}`:
            Note that :math:`\rho_{22}` is real, :math:`\tilde{\rho}_{12}` is complex.
    fwhm
        FWHM of input pluse.
    t1
        Population decay time (:math:`T_1`).  The Dephasing time (:math:`T_2` is assumed
        as :math:`2T_1`  (The pure dephasing time is assumed as infinity.)
    omega12_minus_omega
        :math:`\omega_{12}-\omega`, 0 means the resonant condition, while !=0 is off
        resonant.
    amplitude
        [TODO:description]

    Returns
    -------
    NDArray[np.complex128]
        :math:`\frac{d\rho_{22}}{dt}` and :math:`\frac{d\tilde{\rho}_{12}}{dt}`

    """
    e_field: NDArray[np.float64] = gaussian_pulse(
        t=t,
        fwhm=fwhm,
        t0=0,
    )
    t2 = 2 * t1
    r22: NDArray[np.float64] = np.real(rho[0])
    r11: NDArray[np.float64] = 1.0 - r22
    r12t: NDArray[np.complex128] = rho[1]
    r21t: NDArray[np.complex128] = np.conjugate(r12t)
    dr22dt = -1.0j * amplitude * e_field * (r12t - r21t) - r22 / t1
    dr12tdt = (
        -1.0j * amplitude * e_field * (r22 - r11)
        + (1.0j * omega12_minus_omega - 1 / t2) * r12t
    )
    return np.array([dr22dt, dr12tdt])


def rho22(
    t: float,
    t_span: tuple[float, float],
    fwhm: float,
    t1: float,
    omega12_minus_omega: float,
    amplitude: float,
    num_t: int = 5000,
    coeff_a: float = 1e-3,
) -> np.float64:
    r""":math:`\rho_{22}` from bloch equation.

    Parameters
    ----------
    t
        the time
    t_span
        time span, the first value should be minus and suffifiently low comparing with
        FWHM of the input pulse
    fwhm
        FWHM of the input pulse in fs.
    t1
        Population decay time (:math:`T_1`).  The Dephasing time (:math:`T_2` is assumed
        as :math:`2T_1`  (The pure dephasing time is assumed as infinity.)
    omega12_minus_omega
        :math:`\omega_{12}-\omega`, 0 means the resonant condition, while !=0 is off
        resonant.
    amplitude
        the maximum value of :math:`rho_{22}`
    num_t
        default is 5000.
    coeff_a
        coefficient corresponding to the transition dipole.

    Returns
    -------
    np.float64
        [TODO:description]

    """
    init = [0 + 0j, 0 + 0j]
    sol: OdeResult = solve_ivp(
        bloch,
        t_span=t_span,
        y0=init,
        args=(fwhm, t1, omega12_minus_omega, coeff_a),
        t_eval=np.linspace(*t_span, num_t),
    )
    rho22: Callable[[float], np.float64] = interp1d(
        x=sol.t,
        y=sol.y[0] / np.max(sol.y[0]),
        assume_sorted=True,
        bounds_error=False,
        fill_value=0.0,
    )
    return amplitude * np.real(rho22(t))
