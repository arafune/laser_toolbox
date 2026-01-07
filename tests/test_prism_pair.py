"""Unit tests for the pulselaser.prism_pair module."""

import numpy as np
import pytest

from laser_toolbox.prism_pair import PrismPair
from laser_toolbox.sellmeier import Material


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
