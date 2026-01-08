"""Module for prism pair pulse duration compression."""

from typing import NamedTuple

import numpy as np

from .sellmeier import Material
from .types import ScalarOrArray, Scalar


class _PrismInsert(NamedTuple):
    first: float
    second: float


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

    - Wavelengths are given in microns.
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

    def __init__(  # noqa: PLR0913
        self,
        wavelength_micron: ScalarOrArray,
        incident_angle_deg: Scalar,
        separation: Scalar,
        prism_insert: tuple[float, float],
        prism_apex: float = 59,
        prism_material: str | Material = "SF11",
    ) -> None:
        r"""Initialize PrismPair class.

        Parameters
        ----------
        wavelength_micron
            Wavelength of light
        incident_angle_deg
            Incident angle of the light to the first prism (degrees).
        separation
            The distance two prism (the distance two apexes of the prism).
        prism_insert
            The tuple for the insert length of the prism (l_1, l_2).
        prism_apex
            The apex angle of the prism (degrees)
        prism_material
            The material of the prism.

        """
        assert separation >= 0
        assert prism_insert[0] >= 0
        assert prism_insert[1] >= 0

        self.material: Material = (
            Material.from_str(prism_material)
            if isinstance(prism_material, str)
            else prism_material
        )
        self.wavelength: ScalarOrArray = wavelength_micron
        self.theta_0: Scalar = np.deg2rad(incident_angle_deg)
        self.separation: Scalar = separation
        self.prism_insert: _PrismInsert = _PrismInsert(
            first=prism_insert[0],
            second=prism_insert[1],
        )
        self.alpha: Scalar = np.deg2rad(prism_apex)

    @property
    def theta_1(self) -> ScalarOrArray:
        """Return the refraction angle inside the prism.

        Returns
        -------
        ScalarOrArray
            The refraction angle inside the prism.

        """
        n = self.material(self.wavelength)
        return np.arcsin(np.sin(self.theta_0) / n)

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

    def gdd(self) -> float:
        """Return GDD of prism pairs.

        Returns
        -------
        float
            GDD of the prism pair including reflection mirror.

        """
