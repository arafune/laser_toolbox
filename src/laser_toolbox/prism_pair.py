"""Module for prism pair pulse duration compression."""

from typing import NamedTuple

import numpy as np

from .sellmeier import Material


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
        wavelength_micron: float,
        incident_angle_deg: float,
        separation: float,
        prism_insert: tuple[float, float],
        prism_apex: float = 59,
        prism_material: str | Material = "SF11",
    ) -> None:
        """Initialize PrismPair class.

        Parameters
        ----------
        wavelength_micron
            Wavelength of light
        incident_angle_deg
            Incident angle of the light to the first prism (degrees).
        separation
            The distance two prism (the distance two apexes of the prism).
        prism_insert
            The tuple for the insert length of the prism (1st, 2nd).
        prism_apex
            The apex angle of the prism (degrees)
        prism_material
            The material of the prism.

        """
        assert separation >= 0
        assert prism_insert[0] >= 0
        assert prism_insert[1] >= 0

        self.material = (
            Material.from_str(prism_material)
            if isinstance(prism_material, str)
            else prism_material
        )
        self.wavelength = wavelength_micron
        self.theta_0 = np.deg2rad(incident_angle_deg)
        self.separation = separation
        self.prism_insert = _PrismInsert(
            first=prism_insert[0],
            second=prism_insert[1],
        )
        self.alpha = np.deg2rad(prism_apex)

    def gdd(self) -> float:
        """Return GDD of prism pairs.

        Returns
        -------
        float
            GDD of the prism pair including reflection mirror.

        """
