"""Unit test for sellmeier/__init__.py."""  # noqa: INP001

import numpy as np
import pytest

import laser_toolbox as ltb


@pytest.mark.parametrize(
    ("material", "wavelength", "gvd"),
    [
        ("bk7", 0.800, 44.651),
        ("beta_bbo", 0.800, (75.5438, 58.831262)),
    ],
)
def test_gvd(
    material: str,
    wavelength: float,
    gvd: float,
) -> None:
    """Test for gvd calculation of 'material' at 'wavelength'."""
    np.testing.assert_allclose(
        ltb.gvd(wavelength, material),
        gvd,
        rtol=0.0001,
    )


def test_gvd_unknown_material() -> None:
    """Ensure ValueError is raised when an unsupported material name is provided."""
    msg = "Unknown material: unknown_material"
    with pytest.raises(ValueError, match=msg):
        ltb.gvd(0.80, "unknown_material")
