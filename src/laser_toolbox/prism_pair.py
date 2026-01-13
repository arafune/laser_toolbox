"""Module for prism pair pulse duration compression."""

from typing_extensions import override
from typing import NamedTuple, Final
import numpy as np

from .sellmeier import Material
from .types import ScalarOrArray, Scalar


class _PrismInsert(NamedTuple):
    first: float
    second: float


def ideal_apex_deg(
    wavelength_nm: ScalarOrArray = 800.0,
    material: str | Material = "SF11",
) -> ScalarOrArray:
    """Calculate the ideal apex angle for minimum deviation.

    Parameters
    ----------
    wavelength_nm
        Wavelength in nm.

    material
        The material of the prism. Default is "SF11".

    Returns
    -------
    Scalar
        Ideal apex angle in degrees.

    """

    prism_material: Material = (
        Material.from_str(material) if isinstance(material, str) else material
    )
    if isinstance(wavelength_nm, np.ndarray):
        assert np.all(wavelength_nm > 180), (
            "wavelength_nm must be in nm and greater than 180 nm"
        )
        brewster_angle_deg_ = brewster_angle_deg(
            wavelength_nm=wavelength_nm, material=material
        )
        alpha_rad = 2 * np.arcsin(
            np.sin(np.deg2rad(brewster_angle_deg_))
            / prism_material(wavelength_nm * 1e-3)
        )
        return np.rad2deg(alpha_rad)
    assert isinstance(wavelength_nm, (float, np.floating, int))
    assert wavelength_nm > 180, "wavelength_nm must be in nm and greater than 180 nm"
    brewster_angle_deg_ = brewster_angle_deg(
        wavelength_nm=wavelength_nm, material=material
    )
    alpha_rad = 2 * np.arcsin(
        np.sin(np.deg2rad(brewster_angle_deg_)) / prism_material(wavelength_nm * 1e-3)
    )
    return float(np.rad2deg(alpha_rad))


def brewster_angle_deg(
    wavelength_nm: ScalarOrArray = 800.0,
    material: str | Material = "SF11",
) -> ScalarOrArray:
    """Calculate the Brewster angle for a given refractive index.

    Parameters
    ----------
    wavelength_nm
        Wavelength in nm.

    material
        The material of the prism. Default is "SF11".


    Returns
    -------
    Scalar
        Brewster angle in degrees.

    """

    prism_material: Material = (
        Material.from_str(material) if isinstance(material, str) else material
    )
    if isinstance(wavelength_nm, np.ndarray):
        assert np.all(wavelength_nm > 180), (
            "wavelength_nm must be in nm and greater than 180 nm"
        )
        n = prism_material(wavelength_nm * 1e-3)
        return np.rad2deg(np.arctan(n))
    assert isinstance(wavelength_nm, (float, np.floating, int))
    assert wavelength_nm > 180, "wavelength_nm must be in nm and greater than 180 nm"
    n = prism_material(wavelength_nm * 1e-3)
    return float(np.rad2deg(np.arctan(n)))


class PrismPair:
    """Prism pair for dispersion control.

    This class represents a prism-pair optical setup used as a pulse
    compressor or stretcher, and provides methods to calculate
    dispersion quantities such as group-delay dispersion (GDD).

    The model is based on geometric optics and wavelength-dependent
    refractive indices given by Sellmeier equations.

    Notes
    -----
    - The properties of two prisms are identical.
    - The base of the two prisms are parallel.

    - Wavelengths are given in nanometers, and internally converted to microns.
    - the Unit of time is in femtoseconds (defined in c)
    - Angles are specified in degrees at initialization and internally
      converted to radians.
    - The prism separation is defined as the distance between the two
      prism apexes along the optical axis.
    - The prism material can be specified either by name (string) or by
      the ``Material`` enum.

    Examples
    --------
    >>> pair = PrismPair(
    ...     wavelength_micron=0.8,
    ...     incident_angle_deg=60.0,
    ...     separation=300.0,
    ...     prism_insert=(10.0, 10.0),
    ...     prism_material="SF11",
    ... )
    >>> gdd = pair.gdd()

    """

    c: Final = 0.299792458  # µm/fs

    def __init__(  # noqa: PLR0913
        self,
        incident_angle_deg: Scalar,
        separation_mm: Scalar,
        prism_insert_mm: tuple[float, float],
        apex_deg: float | None = None,
        wavelength_nm: ScalarOrArray | None = None,
        material: str | Material = "SF11",
    ) -> None:
        r"""Initialize PrismPair class.

        Parameters
        ----------
        incident_angle_deg
            Incident angle of the light to the first prism (degrees).
        separation
            The distance two prism (the distance two apexes of the prism).
        prism_insert
            The tuple for the insert length (mm) of the prism (l_1, l_2).
        wavelength_nm
            Wavelength of light, if None, use 800 nm.
        apex_deg
            The apex angle of the prism (degrees) if None, use Brewster angle of the input wavelength.
        material
            The material of the prism. Default is "SF11".

        """
        assert separation_mm >= 0
        assert prism_insert_mm[0] >= 0
        assert prism_insert_mm[1] >= 0

        self.wavelength: ScalarOrArray = (
            0.8 if wavelength_nm is None else wavelength_nm * 1e-3
        )
        assert np.all(self.wavelength < 3)

        self.material: Material = (
            Material.from_str(material) if isinstance(material, str) else material
        )
        self.theta_0: Scalar = np.deg2rad(incident_angle_deg)
        self.separation: Scalar = separation_mm * 1e3
        self.prism_insert: _PrismInsert = _PrismInsert(
            first=prism_insert_mm[0] * 1e3,
            second=prism_insert_mm[1] * 1e3,
        )
        if apex_deg is None:
            msg = f"wavelength_nm is {wavelength_nm}, and type is {type(wavelength_nm)}, which must be scalar when apex_deg is not set."
            assert isinstance(self.wavelength, (np.floating, float, int)), msg
            brewster_angle_deg_: float = brewster_angle_deg(
                wavelength_nm=self.wavelength * 1e3, material=material
            )
            self.alpha: float | np.floating = 2 * np.arcsin(
                np.sin(np.deg2rad(brewster_angle_deg_)) / self.material(self.wavelength)
            )
        else:
            self.alpha = np.deg2rad(apex_deg)
        assert isinstance(self.alpha, (np.floating, float)), (
            f" apex is {self.alpha}. type of alpha is {type(self.alpha)}, not float / np.floating"
        )

    @override
    def __str__(self) -> str:
        if isinstance(self.wavelength, (float, np.floating, int)):
            wl_str = f"{self.wavelength * 1e3:.2f} nm"
        else:
            assert isinstance(self.wavelength, np.ndarray)
            wl_str = f"array({self.wavelength.size}), {self.wavelength.min() * 1e3:.2f}–{self.wavelength.max() * 1e3:.2f} nm"
        return (
            f"PrismPair(\n"
            f"  wavelength_nm={wl_str},\n"
            f"  incident_angle_deg={np.rad2deg(self.theta_0):.2f},\n"
            f"  separation_mm={self.separation / 1e3:.2f},\n"
            f"  prism_insert_m=({self.prism_insert.first / 1e3:.2f}, {self.prism_insert.second / 1e3:.2f}),\n"
            f"  apex_deg={np.rad2deg(self.alpha):.2f},\n"
            f"  material={self.material.name}\n"
            f")"
        )

    @override
    def __repr__(self) -> str:
        if isinstance(self.wavelength, (float, np.floating)):
            wl_str = f"{self.wavelength * 1e3:.2f} nm"
        else:
            assert isinstance(self.wavelength, np.ndarray)
            wl_str = f"array({self.wavelength.size}), {self.wavelength.min() * 1e3:.2f}–{self.wavelength.max() * 1e3:.2f} nm"
        return (
            f"PrismPair("
            f"wavelength_nm={wl_str}, "
            f"incident_angle_deg={np.rad2deg(self.theta_0):.2f}, "
            f"separation={self.separation / 1e3:.2f}, "
            f"prism_insert_mm=({self.prism_insert.first / 1e3:.2f}, {self.prism_insert.second / 1e3:.2f}), "
            f"apex={np.rad2deg(self.alpha):.2f}, "
            f"material='{self.material.name}')"
        )

    @property
    def n(self) -> ScalarOrArray:
        """Return the refractive index of the prism material at the given wavelength.

        Returns
        -------
        ScalarOrArray
            The refractive index.

        """
        n = self.material(self.wavelength)
        if isinstance(n, (float, np.floating, np.ndarray)):
            return n

        raise TypeError("Index must be Scalar or Array")  # pragma: no cover

    @property
    def theta_1(self) -> ScalarOrArray:
        """Return the refraction angle inside the prism.

        Returns
        -------
        ScalarOrArray
            The refraction angle inside the prism.

        """
        return np.arcsin(np.sin(self.theta_0) / self.n)

    @property
    def theta_2(self) -> ScalarOrArray:
        """Return the exit angle from the prism.

        Returns
        -------
        ScalarOrArray
            The exit angle from the prism.

        """
        return self.alpha - self.theta_1

    @property
    def theta_3(self) -> ScalarOrArray:
        """Return the exit angle from the prism.

        Returns
        -------
        ScalarOrArray
            The exit angle from the prism.

        """
        return np.arcsin(self.n * np.sin(self.theta_2))

    @property
    def p_ab(self) -> ScalarOrArray:
        r"""Reuturn the path of P_AB

        $\theta_2 = \alpha - \thata_1$
        $P_{AB} \cos \theta_2 = l_1 \sin(\alpha)
        """

        l1 = self.prism_insert.first
        return l1 * np.sin(self.alpha) / np.cos(self.theta_2)

    @property
    def p_ob(self) -> ScalarOrArray:
        r"""Return the path of P_OB

        P_OB = l1 cos \alpha + AB \sin \theta_2 = l1 cos \theta_1 / cos \theta_2

        For the brewster angle prism, $P_{OB} = =P_{OA} = l_1$
        """

        l1 = self.prism_insert.first
        return l1 * np.cos(self.theta_1) / np.cos(self.theta_2)

    @property
    def p_od(self) -> ScalarOrArray:
        r"""Return the path of P_OD"""

        l2 = self.prism_insert.second
        return l2 * np.cos(self.theta_2) / np.cos(self.theta_1)

    @property
    def lg(self) -> ScalarOrArray:
        r"""Return the path length in the prism.

        $  \frac{\left(P_{O_1 A} \frac{\cos \theta_1}{\cos\theta_2} + P_{O_2 C} \right) \sin \alpha}{\cos \theta_1}$
        """
        l2 = self.prism_insert.second
        g = (self.p_ob + l2) * np.sin(self.alpha)
        return g / np.cos(self.theta_1)

    @property
    def dn_dOmega(self) -> ScalarOrArray:
        """Return the derivative of refractive index with respect to angular frequency.

        Returns
        -------
        ScalarOrArray
            The derivative of refractive index with respect to angular frequency.

        """
        dn_dlambda = self.material(self.wavelength, derivative=1)
        if isinstance(dn_dlambda, (float, np.floating, np.ndarray)):
            return -(self.wavelength**2) / (2 * np.pi * self.c) * dn_dlambda
        raise TypeError("Index must be Scalar or Array")  # pragma: no cover

    @property
    def dtheta1_dOmega(self) -> ScalarOrArray:
        r"""Return the derivative of theta_1 with respect to angular frequency.

        $\frac{d}{d\Omega} \theta_1= - \left[n^2 - \sin^2\theta_0\right]^{-\frac{1}{2}}\frac{\sin\theta_0}{n} \frac{dn}{d\Omega}$

        Returns
        -------
        ScalarOrArray
            The derivative of theta_1 with respect to angular frequency.

        """
        return (
            -((self.n**2 - np.sin(self.theta_0) ** 2) ** (-1 / 2))
            * np.sin(self.theta_0)
            / self.n
            * self.dn_dOmega
        )

    @property
    def dtheta3_dOmega(self) -> ScalarOrArray:
        r"""Return the derivative of theta_3 with respect to angular frequency.

        $\frac{d}{d\Omega}\theta_3$

        Returns
        -------
        ScalarOrArray
            The derivative of theta_3 with respect to angular frequency.

        """
        n: ScalarOrArray = self.n
        return ((1 - n**2 * np.sin(self.theta_2) ** 2) ** (-1 / 2)) * (
            -n * np.cos(self.theta_2) * self.dtheta1_dOmega
            + np.sin(self.theta_2) * self.dn_dOmega
        )

    @property
    def gdd_positive(self) -> ScalarOrArray:
        r"""Return the positive GDD contribution from the prism material.

        $ \mathrm{GDD}_{\mathrm{positive}} = \frac{\lambda^3}{2 \pi c^2} \frac{d^2 n}{d \lambda^2} \cdot lg $

        Returns
        -------
        ScalarOrArray
            The positive GDD contribution from the prism material.

        """
        d2n_dlambda2 = self.material(self.wavelength, derivative=2)
        if isinstance(d2n_dlambda2, (float, np.floating, np.ndarray)):
            return (
                2
                * (self.wavelength**3)
                / (2 * np.pi * self.c**2)
                * d2n_dlambda2
                * self.lg
            )
        raise TypeError("Index must be Scalar or Array")  # pragma: no cover

    @property
    def gdd_negative(self) -> ScalarOrArray:
        r"""Return the negative GDD contribution from the prism pair geometry.

        $ \mathrm{GDD}_{\mathrm{negative}} = -\frac{2 \lambda}{\pi c^2} \left(\frac{d n}{d \lambda}\right)^2 \cdot p_{od} $

        Returns
        -------
        ScalarOrArray
            The negative GDD contribution from the prism pair geometry.

        """
        dn_dlambda = self.material(self.wavelength, derivative=1)
        if isinstance(dn_dlambda, (float, np.floating, np.ndarray)):
            gap_component = (
                -(2 * np.pi / self.wavelength)
                * self.separation
                * (self.dtheta3_dOmega) ** 2
            )
            prism_component = (
                -self.n
                * (2 * np.pi / self.wavelength)
                * self.lg
                * (self.dtheta1_dOmega) ** 2
            )
            return 2 * (gap_component + prism_component)
        raise TypeError("Index must be Scalar or Array")  # pragma: no cover

    @property
    def gdd(self) -> ScalarOrArray:
        """Return GDD of prism pairs.

        Returns
        -------
        ScalarOrArray
            GDD of the prism pair including reflection mirror.

        """
        return self.gdd_positive + self.gdd_negative
