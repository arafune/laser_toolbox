"""Collection of Sellmeier equation."""

from collections.abc import Callable
from enum import Enum
from functools import partial
from typing import Literal, Protocol, overload, runtime_checkable

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from .types import Scalar, ScalarOrArray

DispersionResult = float | np.ndarray | sp.Expr


@overload
def three_term_sellmeier(
    lambda_micron: Scalar,
    coeff_a: float,
    coeffs_b: tuple[float, float, float],
    coeffs_c: tuple[float, float, float],
    derivative_order: int = 0,
    *,
    as_sympy: Literal[True],
) -> sp.Expr: ...


@overload
def three_term_sellmeier(
    lambda_micron: NDArray[np.floating],
    coeff_a: float,
    coeffs_b: tuple[float, float, float],
    coeffs_c: tuple[float, float, float],
    derivative_order: int = 0,
    *,
    as_sympy: Literal[True],
) -> sp.Expr: ...


@overload
def three_term_sellmeier(
    lambda_micron: Scalar,
    coeff_a: float,
    coeffs_b: tuple[float, float, float],
    coeffs_c: tuple[float, float, float],
    derivative_order: int = 0,
    *,
    as_sympy: Literal[False] = False,
) -> np.floating: ...


@overload
def three_term_sellmeier(
    lambda_micron: NDArray[np.floating],
    coeff_a: float,
    coeffs_b: tuple[float, float, float],
    coeffs_c: tuple[float, float, float],
    derivative_order: int = 0,
    *,
    as_sympy: Literal[False] = False,
) -> NDArray[np.floating]: ...


def three_term_sellmeier(  # noqa: PLR0913
    lambda_micron: ScalarOrArray,
    coeff_a: float,
    coeffs_b: tuple[float, float, float],
    coeffs_c: tuple[float, float, float],
    derivative_order: int = 0,
    *,
    as_sympy: bool = False,
) -> ScalarOrArray | sp.Expr:
    r"""Return Sellmeier function.

    :math:`n^2 -1 = A + \frac{B_1 \lambda^2}{\lambda^2 - C_1} +
    \frac{B_2 \lambda^2}{\lambda^2 - C_2} + \frac{B_3 \lambda^2}{\lambda^2 - C_3}`.


    Parameters
    ----------
    lambda_micron: float
        wavelength in micron
    coeff_a: float
        Coefficient A
    coeffs_b: tuple[float, float, float]
        Coefficient B
    coeffs_c: tuple[float, float, float]
        Coefficient C
    derivative_order: int
        Derivative order
    as_sympy: bool
        if True, return Sympy object

    Returns
    -------
    float | sp.Expr
        Calculated refractive index

    """
    if derivative_order < 0:
        msg = "derivative must be equal or greater than zero"
        raise ValueError(msg)
    lam, a, b1, b2, b3, c1, c2, c3 = sp.symbols(
        "lambda, a, b1, b2, b3, c1, c2, c3",
        real=True,
    )
    refractive_index = sp.sqrt(
        1
        + a
        + b1 * lam**2 / (lam**2 - c1)
        + b2 * lam**2 / (lam**2 - c2)
        + b3 * lam**2 / (lam**2 - c3),
    )
    delta_n = sp.diff(
        refractive_index,
        lam,
        derivative_order,
    )
    if as_sympy:
        return delta_n.subs(
            {
                a: coeff_a,
                b1: coeffs_b[0],
                b2: coeffs_b[1],
                b3: coeffs_b[2],
                c1: coeffs_c[0],
                c2: coeffs_c[1],
                c3: coeffs_c[2],
            },
        )
    delta_n_np = sp.lambdify(
        (lam, a, b1, b2, b3, c1, c2, c3),
        delta_n,
        "numpy",
    )
    return delta_n_np(lambda_micron, coeff_a, *coeffs_b, *coeffs_c)


@overload
def two_term_sellmeier(
    lambda_micron: Scalar,
    coeff_a: float,
    coeffs_b: tuple[float, float],
    coeffs_c: tuple[float, float],
    derivative_order: int = 0,
    *,
    as_sympy: Literal[True],
) -> sp.Expr: ...


@overload
def two_term_sellmeier(
    lambda_micron: NDArray[np.floating],
    coeff_a: float,
    coeffs_b: tuple[float, float],
    coeffs_c: tuple[float, float],
    derivative_order: int = 0,
    *,
    as_sympy: Literal[True],
) -> sp.Expr: ...


@overload
def two_term_sellmeier(
    lambda_micron: NDArray[np.floating],
    coeff_a: float,
    coeffs_b: tuple[float, float],
    coeffs_c: tuple[float, float],
    derivative_order: int = 0,
    *,
    as_sympy: Literal[False] = False,
) -> NDArray[np.floating]: ...


@overload
def two_term_sellmeier(
    lambda_micron: Scalar,
    coeff_a: float,
    coeffs_b: tuple[float, float],
    coeffs_c: tuple[float, float],
    derivative_order: int = 0,
    *,
    as_sympy: Literal[False] = False,
) -> np.floating: ...


def two_term_sellmeier(  # noqa: PLR0913
    lambda_micron: ScalarOrArray,
    coeff_a: float,
    coeffs_b: tuple[float, float],
    coeffs_c: tuple[float, float],
    derivative_order: int = 0,
    *,
    as_sympy: bool = False,
) -> ScalarOrArray | sp.Expr:
    r"""Return the Sellmeier function represented by two terms.

    :math:`n^2 -1 = A + \frac{B_1 \lambda^2}{\lambda^2 - C_1} +
    \frac{B_2 \lambda^2}{\lambda^2 - C_2}`.

    Parameters
    ----------
    lambda_micron: float
        wavelength in micron
    coeff_a: float
        coefficient A
    coeffs_b: tuple[float, float]
        Coefficients B_n
    coeffs_c: tuple[float, float]
        Coefficients C_n
    derivative_order: int
        Derivative order
    as_sympy: bool
        if True, return Sympy object


    Returns
    -------
    float
        Calculated refractive index

    """
    return three_term_sellmeier(
        lambda_micron,
        coeff_a,
        (*coeffs_b, 0.0),
        (*coeffs_c, 0.0),
        derivative_order=derivative_order,
        as_sympy=as_sympy,
    )


def air_dispersion(
    lambda_micron: ScalarOrArray,
    coeffs_b: tuple[float, float],
    coeffs_c: tuple[float, float],
    derivative_order: int = 0,
    *,
    as_sympy: bool = False,
) -> ScalarOrArray | sp.Expr:
    """Return the Dispersion formula/value of the air.

    Ciddor-type formula, not Sellmeier.

    Parameters
    ----------
    lambda_micron: float
        wavelength in micron
    coeffs_b: tuple[float, float]
        Coefficients B_n
    coeffs_c: tuple[float, float]
        Coefficients C_n
    derivative_order: int
        Derivative order
    as_sympy: bool
        if True, return Sympy object

    """
    if derivative_order < 0:
        msg = "derivative must be equal or greater than zero"
        raise ValueError(msg)
    lam, b1, b2, c1, c2 = sp.symbols(
        "lambda, b1, b2,  c1, c2",
        real=True,
    )
    refractive_index = 1 + b1 / (c1 - lam ** (-2)) + b2 / (c2 - lam ** (-2))
    delta_n = sp.diff(refractive_index, lam, derivative_order)
    if as_sympy:
        return delta_n.subs(
            {
                b1: coeffs_b[0],
                b2: coeffs_b[1],
                c1: coeffs_c[0],
                c2: coeffs_c[1],
            },
        )
    delta_n_np = sp.lambdify(
        (lam, b1, b2, c1, c2),
        delta_n,
        "numpy",
    )
    return delta_n_np(
        lambda_micron,
        *coeffs_b,
        *coeffs_c,
    )


# -----------


def bk7(
    lambda_micron: ScalarOrArray,
    derivative: int = 0,
    *,
    as_sympy: bool = False,
) -> ScalarOrArray | sp.Expr:
    r"""Dispersion of BK7.

    https://refractiveindex.info/?shelf=glass&book=BK7&page=SCHOTT

    The values of the coefficients are taken from the SCHOTT catalog, as listed on the
    Thorlabs web.

    .. math::

    n^2(\lambda) = 1
    + \sum_{i=1}^3 \frac{B_i \lambda^2}{\lambda^2 - C_i}

    Parameters
    ----------
    lambda_micron: float
        wavelength (:math:`\lambda`) in micron (:math:`\mu m`) unit.
    derivative: int
        Derivative order
    as_sympy: bool
        If True return the equation

    """
    b = (1.03961212, 0.231792344, 1.01046945)
    c = (0.00600069867, 0.0200179144, 103.560653)
    return three_term_sellmeier(
        lambda_micron,
        0,
        b,
        c,
        derivative_order=derivative,
        as_sympy=as_sympy,
    )


def fused_silica(
    lambda_micron: ScalarOrArray,
    derivative: int = 0,
    *,
    as_sympy: bool = False,
) -> ScalarOrArray | sp.Expr:
    r"""Dispersion of Fused Silica (0.21- 3.71 micron).

    https://refractiveindex.info/?shelf=glass&book=fused_silica&page=Malitson

    Parameters
    ----------
    lambda_micron: float
        wavelength (:math:`\lambda`) in micron (:math:`\mu m`) unit.
    derivative: int
        the derivative order
    as_sympy: bool
        If True return the equation

    Returns
    -------
    float:
        the refractive index or its derivative

    """
    b = (0.6961663, 0.4079426, 0.8974794)
    # Malitson form: C_i = lambda_i^2
    c = (0.06840432**2, 0.11624142**2, 9.8961612**2)
    return three_term_sellmeier(
        lambda_micron,
        0,
        b,
        c,
        derivative_order=derivative,
        as_sympy=as_sympy,
    )


def caf2(
    lambda_micron: ScalarOrArray,
    derivative: int = 0,
    *,
    as_sympy: bool = False,
) -> ScalarOrArray | sp.Expr:
    r"""Dispersion of CaF2 (0.15 - 12 micron).

    J. Phys. Chem. Ref. Data 9 161 (1980).

    Parameters
    ----------
    lambda_micron: float
        wavelength (:math:`\lambda`) in micron (:math:`\mu m`) unit.
    derivative: int
        the derivative order
    as_sympy: bool
        If True return the equation

    """
    a = 0.33973
    b = (0.69913, 0.11994, 4.35181)
    c = (0.09374**2, 21.18**2, 38.46**2)
    return three_term_sellmeier(
        lambda_micron,
        a,
        b,
        c,
        derivative_order=derivative,
        as_sympy=as_sympy,
    )


def sf10(
    lambda_micron: ScalarOrArray,
    derivative: int = 0,
    *,
    as_sympy: bool = False,
) -> ScalarOrArray | sp.Expr:
    r"""Dispersion of SF10 (0.15 - 12 micron).

    https://refractiveindex.info/?shelf=popular_glass&book=SF10&page=SCHOTT

    Parameters
    ----------
    lambda_micron: float
        wavelength (:math:`\lambda`) in micron (:math:`\mu m`) unit.
    derivative: int
        The derivative order
    as_sympy: bool
        If True return the equation

    """
    b = (1.6215390, 0.256287842, 1.64447552)
    c = (0.0122241457, 0.0595736775, 147.468793)
    return three_term_sellmeier(
        lambda_micron,
        0,
        b,
        c,
        derivative_order=derivative,
        as_sympy=as_sympy,
    )


def sf11(
    lambda_micron: ScalarOrArray,
    derivative: int = 0,
    *,
    as_sympy: bool = False,
) -> ScalarOrArray | sp.Expr:
    r"""Dispersion of SF11 (0.37 - 2.5 micron).

    https://refractiveindex.info/?shelf=specs&book=SCHOTT-optical&page=N-SF11

    Parameters
    ----------
    lambda_micron: float
        wavelength (:math:`\lambda`) in micron (:math:`\mu m`) unit.
    derivative: int
        The derivative order
    as_sympy: bool
        If True return the equation

    """
    b = (1.73759695, 0.313747346, 1.89878101)
    c = (0.013188707, 0.0623068142, 155.23629)
    return three_term_sellmeier(
        lambda_micron,
        0,
        b,
        c,
        derivative_order=derivative,
        as_sympy=as_sympy,
    )


def air(
    lambda_micron: ScalarOrArray,
    derivative: int = 0,
    *,
    as_sympy: bool = False,
) -> ScalarOrArray | sp.Expr:
    r"""Dispersion of air.

    https://refractiveindelambda_micron.info/?shelf=other&book=air&page=Ciddor

    Parameters
    ----------
    lambda_micron: float
        wavelength (:math:`\lambda`) in micron (:math:`\mu m`) unit.
    derivative: int
        The derivative order
    as_sympy: bool
        If True return the equation

    Returns
    -------
    float:
        :math:`n`

    """
    b = (0.05792105, 0.00167917)
    c = (238.0185, 57.362)
    return air_dispersion(
        lambda_micron,
        b,
        c,
        derivative_order=derivative,
        as_sympy=as_sympy,
    )


def bbo_sellmeier(  # noqa: PLR0913
    lambda_micron: ScalarOrArray,
    coeff_a: float,
    coeff_b: float,
    coeff_c: float,
    coeff_d: float,
    derivative_order: int = 0,
    *,
    as_sympy: bool = False,
) -> ScalarOrArray | sp.Expr:
    r"""Sellmeier equation for BBO.

    :math:`n^2(\lambda) = a + \frac{b}{\lambda^2 - c} - d \lambda^2`

    where :math:`\lambda` is in microns.

    If ``derivative_order = n``, this function returns
    :math:`\mathrm{d}^n n / \mathrm{d}\lambda^n`.
    """
    lam, a, b, c, d = sp.symbols("lambda, a, b, c, d", real=True)
    refractive_index = sp.sqrt(a - d * lam**2 + b / (lam**2 - c))
    delta_n = sp.diff(refractive_index, lam, derivative_order)
    if as_sympy:
        return delta_n.subs(
            {
                a: coeff_a,
                b: coeff_b,
                c: coeff_c,
                d: coeff_d,
            },
        )
    delta_n_np = sp.lambdify(
        (lam, a, b, c, d),
        delta_n,
        "numpy",
    )
    return delta_n_np(
        lambda_micron,
        coeff_a,
        coeff_b,
        coeff_c,
        coeff_d,
    )


def alpha_bbo(
    lambda_micron: ScalarOrArray,
    derivative: int = 0,
    *,
    as_sympy: bool = False,
) -> NDArray[np.floating] | tuple[sp.Expr, sp.Expr]:
    r"""Dispersion of :math:`$\alpha$`-BBO.

    These coefficients are taken from . Applied Optics, 41(13), 2474.

    coeffs_o = (2.7471, 0.01878, 0.01822, 0.01354)
    coeffs_e = (2.3174, 0.01224, 0.01667, 0.001516)
    * Negative birefringence

    On Thorlabs web:
    @587.6
    n_o = 1.533
    n_e = 1.673

    Parameters
    ----------
    lambda_micron: float
        wavelength (:math:`\lambda`) in micron (:math:`\mu m`) unit.
    derivative: int
        the order of derivative
    as_sympy: bool
        If True return the equation


    Return
    -------
    ndarray
        shape (..., 2) with (:math:`n_o`, :math:`n_e`)

    """
    # The following coefficients are taken from
    # http://www.newlightphotonics.com/Birefringent-Crystals/alpha-BBO-Crystals
    # coeffs_o = (2.67579, 0.02099, 0.00470, 0.00528)  # noqa: ERA001
    # coeffs_e = (2.31197, 0.01184, 0.016070, 0.00400)  # noqa: ERA001
    #
    # These values are taken from . Applied Optics, 41(13), 2474.
    # which is also used in Thorlabs's web site:
    coeffs_o = (2.7471, 0.01878, 0.01822, 0.01354)
    coeffs_e = (2.3174, 0.01224, 0.01667, 0.001516)

    n_o = bbo_sellmeier(
        lambda_micron,
        *coeffs_o,
        derivative_order=derivative,
        as_sympy=as_sympy,
    )
    n_e = bbo_sellmeier(
        lambda_micron,
        *coeffs_e,
        derivative_order=derivative,
        as_sympy=as_sympy,
    )
    if as_sympy:
        assert isinstance(n_o, sp.Expr)
        assert isinstance(n_e, sp.Expr)
        return (n_o, n_e)
    return np.stack(
        [
            np.asarray(n_o, dtype=float),
            np.asarray(n_e, dtype=float),
        ],
        axis=-1,
    )


def beta_bbo(
    lambda_micron: ScalarOrArray,
    derivative: int = 0,
    *,
    as_sympy: bool = False,
) -> ScalarOrArray | tuple[sp.Expr, sp.Expr]:
    r"""Return :math:`n_o` and :math:`n_e` of :math:`\beta`-BBO.

    https://refractiveindex.info/?shelf=main&book=BaB2O4&page=Eimerl-o
    https://refractiveindex.info/?shelf=main&book=BaB2O4&page=Eimerl-e

    coeffs_o = (2.7405, 0.0184, 0.0179, 0.0155)
    coeffs_e = (2.3730, 0.0128, 0.0156, 0.0044)

    * Negative birefringence

    Parameters
    ----------
    lambda_micron: float
        wavelength (:math:`\lambda`) in micron (:math:`\mu m`) unit.
    derivative: DERIVATIVE_ORDER
        Order of derivative (0, 1, 2)
    as_sympy: bool
        If True return the equation

    Returns
    -------
    ndarray
        shape (..., 2) with (:math:`n_o`, :math:`n_e`)

    """
    # the coeeficients taken from  IEEE J. Quantum Electron QE-22, 1013 (1986)
    # K. Kato.
    # This is essentially same as the value below.
    #
    # coeffs_o = (2.7359, 0.01878, 0.01822, 0.01354)  # noqa: ERA001
    # coeffs_e = (2.3753, 0.01224, 0.01667, 0.01516)  # noqa: ERA001

    # The coefficients taken from J. Appl. 62, 1968 (1987).
    # D. Eimerl et al.,
    #
    coeffs_o = (2.7405, 0.0184, 0.0179, 0.0155)
    coeffs_e = (2.3730, 0.0128, 0.0156, 0.0044)
    n_o = bbo_sellmeier(
        lambda_micron,
        *coeffs_o,
        derivative_order=derivative,
        as_sympy=as_sympy,
    )
    n_e = bbo_sellmeier(
        lambda_micron,
        *coeffs_e,
        derivative_order=derivative,
        as_sympy=as_sympy,
    )
    if as_sympy:
        assert isinstance(n_o, sp.Expr)
        assert isinstance(n_e, sp.Expr)
        return (n_o, n_e)
    return np.stack(
        [
            np.asarray(n_o, dtype=float),
            np.asarray(n_e, dtype=float),
        ],
        axis=-1,
    )


def quartz(
    lambda_micron: ScalarOrArray,
    derivative: int = 0,
    *,
    as_sympy: bool = False,
) -> ScalarOrArray | tuple[sp.Expr, sp.Expr]:
    r"""Dispersion of crystal quartz.

    Optics communications. 2011, vol. 284, issue 12, p. 2683-2686.

    Opt. Commun. 163, 95-102 (1999).
    https://refractiveindex.info/?shelf=main&book=SiO2&page=Ghosh-e


    Parameters
    ----------
    lambda_micron: float
        wavelength (:math:`\lambda`) in micron (:math:`\mu m`) unit.
    derivative: DERIVATIVE_ORDER
        Order of derivative (0, 1, 2)
    as_sympy: bool
        If True return the equation

    Returns
    -------
    ndarray
        shape (..., 2) with (:math:`n_o`, :math:`n_e`)

    """
    coeff_e_a = 0.28851804
    coeffs_e_b = (1.09509924, 1.15662475)
    coeffs_e_c = (1.02101864 * 1e-2, 100)

    coeff_o_a = 0.28604141
    coeffs_o_b = (1.07044083, 1.10202242)
    coeffs_o_c = (1.00585997 * 1e-2, 100)

    n_o = two_term_sellmeier(
        lambda_micron,
        coeff_o_a,
        coeffs_o_b,
        coeffs_o_c,
        derivative_order=derivative,
        as_sympy=as_sympy,
    )
    n_e = two_term_sellmeier(
        lambda_micron,
        coeff_e_a,
        coeffs_e_b,
        coeffs_e_c,
        derivative_order=derivative,
        as_sympy=as_sympy,
    )
    if as_sympy:
        assert isinstance(n_o, sp.Expr)
        assert isinstance(n_e, sp.Expr)
        return (n_o, n_e)
    return np.stack(
        [
            np.asarray(n_o, dtype=float),
            np.asarray(n_e, dtype=float),
        ],
        axis=-1,
    )


def calcite(
    lambda_micron: ScalarOrArray,
    derivative: int = 0,
    *,
    as_sympy: bool = False,
) -> ScalarOrArray | tuple[sp.Expr, sp.Expr]:
    r"""Dispersion of calcite.  (:math:`\mathrm{CaCO}_3`).

    Opt. Commun. 163, 95-102 (1999)

    @1.064 n_e=1.480, n_o= 1.642

    * Negative birefringence

    Parameters
    ----------
    lambda_micron: float
        wavelength (:math:`\lambda`) in micron (:math:`\mu m`) unit.
    derivative: DERIVATIVE_ORDER
        Order of derivative (0, 1, 2)
    as_sympy: bool
        If True return the equation

    Returns
    -------
    ndarray
        shape (..., 2) with (:math:`n_o`, :math:`n_e`)

    """
    coeff_e_a = 0.35859695
    coeffs_e_b = (0.82427380, 0.14429128)
    coeffs_e_c = (1.06689543 * 1e-2, 120)

    coeff_o_a = 0.73358749
    coeffs_o_b = (0.96464345, 1.82831454)
    coeffs_o_c = (1.94325203 * 1e-2, 120)

    n_o = two_term_sellmeier(
        lambda_micron,
        coeff_o_a,
        coeffs_o_b,
        coeffs_o_c,
        derivative_order=derivative,
        as_sympy=as_sympy,
    )
    n_e = two_term_sellmeier(
        lambda_micron,
        coeff_e_a,
        coeffs_e_b,
        coeffs_e_c,
        derivative_order=derivative,
        as_sympy=as_sympy,
    )
    if as_sympy:
        assert isinstance(n_o, sp.Expr)
        assert isinstance(n_e, sp.Expr)
        return (n_o, n_e)
    return np.stack(
        [
            np.asarray(n_o, dtype=float),
            np.asarray(n_e, dtype=float),
        ],
        axis=-1,
    )


def mgf2(
    lambda_micron: ScalarOrArray,
    derivative: int = 0,
    *,
    as_sympy: bool = False,
) -> ScalarOrArray | tuple[sp.Expr, sp.Expr]:
    r"""Dispersion of mgf2.

    Appl. Opt. 23, 1980-1985 (1984)

    @587.6nm, n_e = 1.390, n_o = 1.378

    Parameters
    ----------
    lambda_micron: float
        wavelength (:math:`\lambda`) in micron (:math:`\mu m`) unit.
    derivative: int
        Order of derivative
    as_sympy: bool
        If True return the equation

    Returns
    -------
    ndarray
        shape (..., 2) with (:math:`n_o`, :math:`n_e`)

    """
    coeff_e_a = 0
    coeffs_e_b = (0.41344023, 0.50497499, 2.4904862)
    coeffs_e_c = (0.03684262**2, 0.09076162**2, 23.771995**2)

    coeff_o_a = 0
    coeffs_o_b = (0.48755108, 0.39875031, 2.3120353)
    coeffs_o_c = (0.04338408**2, 0.09461442**2, 23.793604**2)
    n_o = three_term_sellmeier(
        lambda_micron,
        coeff_o_a,
        coeffs_o_b,
        coeffs_o_c,
        derivative_order=derivative,
        as_sympy=as_sympy,
    )
    n_e = three_term_sellmeier(
        lambda_micron,
        coeff_e_a,
        coeffs_e_b,
        coeffs_e_c,
        derivative_order=derivative,
        as_sympy=as_sympy,
    )
    if as_sympy:
        assert isinstance(n_o, sp.Expr)
        assert isinstance(n_e, sp.Expr)
        return (n_o, n_e)
    return np.stack(
        [
            np.asarray(n_o, dtype=float),
            np.asarray(n_e, dtype=float),
        ],
        axis=-1,
    )


def phase_matching_angle_bbo(fundamental_micron: float) -> float:
    """Phase matching angle of beta-BBO for SHG.

    Parameters
    ----------
    fundamental_micron : float
        wavelength of fundamental light

    Returns
    -------
    float
        Phase matching angle (Unit: Degree)

    """
    n_f = beta_bbo(fundamental_micron)
    n_sh = beta_bbo(fundamental_micron / 2)
    n_o_f = n_f[..., 0]
    n_o_sh = n_sh[..., 0]
    n_e_sh = n_sh[..., 1]
    sin2_theta = (n_o_f ** (-2) - n_o_sh ** (-2)) / (n_e_sh ** (-2) - n_o_sh ** (-2))
    if not (0 < sin2_theta < 1):
        msg = f"Invalid phase matching condition sin^2(theta)={sin2_theta}"
        raise ValueError(msg)
    return np.rad2deg(np.arcsin(np.sqrt(sin2_theta)))


DISPERSION_FUNCS: dict[
    str,
    Callable[
        [ScalarOrArray, int, bool],
        ScalarOrArray | sp.Expr | tuple[sp.Expr, sp.Expr],
    ],
] = {
    "bk7": bk7,
    "fused_silica": fused_silica,
    "caf2": caf2,
    "sf10": sf10,
    "air": air,
    "alpha_bbo": alpha_bbo,
    "beta_bbo": beta_bbo,
    "quartz": quartz,
    "calcite": calcite,
    "mgf2": mgf2,
    "sf11": sf11,
}


@runtime_checkable
class DispersionProtocol(Protocol):
    def __call__(
        self,
        lambda_micron: ScalarOrArray,
        derivative: int = 0,
        *,
        as_sympy: bool = False,
    ) -> DispersionResult: ...


class Material(Enum):
    BK7 = partial(bk7)
    FUSED_SILICA = partial(fused_silica)
    UVFS = partial(fused_silica)  # noqa: PIE796
    CAF2 = partial(caf2)
    SF10 = partial(sf10)
    AIR = partial(air)
    ALPHA_BBO = partial(alpha_bbo)
    BETA_BBO = partial(beta_bbo)
    QUARTZ = partial(quartz)
    CALCITE = partial(calcite)
    MGF2 = partial(mgf2)
    SF11 = partial(sf11)

    def __call__(
        self,
        *args,
        **kwargs,
    ):
        func = self.value
        return func(*args, **kwargs)

    @classmethod
    def from_str(cls, name: str) -> "Material":
        try:
            return Material[name.strip().upper()]
        except KeyError as e:
            msg = f"Unkonwn material: {name}. Available: {[m.name for m in cls]}"
            raise ValueError(msg) from e
