"""Unit tests for the pulselaser.prism_pair module."""

import numpy as np
import pytest

from laser_toolbox.prism_pair import PrismPair
from laser_toolbox.sellmeier import Material


@pytest.fixture
def sf11prisms() -> PrismPair:
    """SF11 prism pair with a typical configuration.

    AB = 4.930 , BC=287.256, CD = 9.838, h= 161.762, w=253.166
    """
    return PrismPair(
        wavelength_micron=0.8,
        incident_angle_deg=60.0,
        separation=300.0,
        prism_insert=(5.0, 10.0),
        prism_material="sf11",
    )


@pytest.fixture
def sf11prisms_woinsert() -> PrismPair:
    """SF11 prism pair without laser insert.

    AB =0, BC = 300.128, CD = 0, h = 155.361,  256.788
    GDD -7450.099
    """
    return PrismPair(
        wavelength_micron=0.8,
        incident_angle_deg=60.0,
        separation=300.0,
        prism_insert=(0.0, 0.0),
        prism_material="sf11",
    )


@pytest.fixture
def sf11prism_brewster() -> PrismPair:
    """SF11 prism pair at Brewster angle @800nm."""
    n: np.floating = Material.SF11(0.8)
    incident_angle_deg: np.floating = np.rad2deg(np.arctan(n))
    alpha: np.floating = 2 * np.arcsin(np.sin(np.deg2rad(incident_angle_deg)) / n)
    return PrismPair(
        wavelength_micron=0.8,
        incident_angle_deg=incident_angle_deg,
        separation=300.0,
        prism_insert=(5, 10),
        prism_apex=np.rad2deg(alpha),
        prism_material="sf11",
    )


def test_theta_1(sf11prisms: PrismPair) -> None:
    np.testing.assert_allclose(sf11prisms.theta_1, 0.51297, rtol=1e-4)


def test_p_ab(sf11prisms: PrismPair, sf11prisms_woinsert: PrismPair) -> None:
    "Check p_ab."
    np.testing.assert_allclose(sf11prisms.p_ab, 4.930, rtol=1e-4)
    np.testing.assert_allclose(sf11prisms_woinsert.p_ab, 0)


def test_p_ob(sf11prism_brewster: PrismPair) -> None:
    "Check p_ob at Brewster angle."
    np.testing.assert_allclose(sf11prism_brewster.p_ob, 5.0, rtol=1e-6)


def test_prism_pair_init_basic() -> None:
    pair = PrismPair(
        wavelength_micron=0.8,
        incident_angle_deg=60.0,
        separation=300.0,
        prism_insert=(10.0, 20.0),
        prism_material="sf11",
    )

    assert pair.wavelength == 0.8
    assert pair.separation == 300.0
    assert pair.prism_insert.first == 10.0
    assert pair.prism_insert.second == 20.0
    assert pair.material == Material.SF11


def test_prism_pair_init_material_enum() -> None:
    pair = PrismPair(
        wavelength_micron=0.8,
        incident_angle_deg=60.0,
        separation=100.0,
        prism_insert=(5.0, 5.0),
        prism_material=Material.SF11,
    )

    assert pair.material is Material.SF11


def test_prism_pair_negative_separation() -> None:
    with pytest.raises(AssertionError):
        PrismPair(
            wavelength_micron=0.8,
            incident_angle_deg=60.0,
            separation=-1.0,
            prism_insert=(5.0, 5.0),
        )


def test_prism_pair_negative_prism_insert_first() -> None:
    with pytest.raises(AssertionError):
        PrismPair(
            wavelength_micron=0.8,
            incident_angle_deg=60.0,
            separation=100.0,
            prism_insert=(-1.0, 5.0),
        )


def test_prism_pair_negative_prism_insert_second() -> None:
    with pytest.raises(AssertionError):
        PrismPair(
            wavelength_micron=0.8,
            incident_angle_deg=60.0,
            separation=100.0,
            prism_insert=(5.0, -1.0),
        )


@pytest.mark.skip
def test_prism_pair_gdd_returns_float() -> None:
    pair = PrismPair(
        wavelength_micron=0.8,
        incident_angle_deg=60.0,
        separation=300.0,
        prism_insert=(10.0, 10.0),
    )

    gdd = pair.gdd()
    assert isinstance(gdd, float)


def test_prism_pair_gdd_is_deterministic() -> None:
    pair = PrismPair(
        wavelength_micron=0.8,
        incident_angle_deg=60.0,
        separation=300.0,
        prism_insert=(10.0, 10.0),
    )

    assert pair.gdd() == pair.gdd()


@pytest.mark.skip
def test_prism_pair_zero_insert_gdd_is_zero() -> None:
    pair = PrismPair(
        wavelength_micron=0.8,
        incident_angle_deg=60.0,
        separation=300.0,
        prism_insert=(0.0, 0.0),
    )

    assert abs(pair.gdd()) < 1e-12


@pytest.mark.skip
def test_prism_pair_zero_separation_gdd_is_zero() -> None:
    pair = PrismPair(
        wavelength_micron=0.8,
        incident_angle_deg=60.0,
        separation=0.0,
        prism_insert=(10.0, 10.0),
    )

    assert abs(pair.gdd()) < 1e-12
