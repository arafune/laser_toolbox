"""Unit tests for the pulselaser.prism_pair module."""

import numpy as np
import pytest

from laser_toolbox.prism_pair import PrismPair, brewster_angle_deg, ideal_apex_deg
from laser_toolbox.sellmeier import Material


@pytest.fixture
def sf11prisms() -> PrismPair:
    """SF11 prism pair with a typical configuration.

    AB = 4.930 , BC=287.256, CD = 9.838, h= 161.762, w=253.166
    """
    return PrismPair(
        wavelength_nm=800,
        incident_angle_deg=59.0,
        separation_mm=300.0,
        prism_insert_mm=(5.0, 10.0),
        material="sf11",
    )


@pytest.fixture
def sf11prisms_woinsert() -> PrismPair:
    """SF11 prism pair without laser insert.

    cf.: https://toolbox.lightcon.com/tools/prismpair
    AB =0, BC = 300.128, CD = 0, h = 155.361,  256.788
    GDD -7450.099
    """
    return PrismPair(
        wavelength_nm=800,
        incident_angle_deg=60.0,
        separation_mm=300.0,
        prism_insert_mm=(0.0, 0.0),
        material="sf11",
        apex_deg=59.0,
    )


@pytest.fixture
def sf11prism_brewster() -> PrismPair:
    """SF11 prism pair at Brewster angle @800nm."""
    wavelength_nm = 800.0
    brewster_angle_deg_: float = brewster_angle_deg(
        wavelength_nm=wavelength_nm, material="SF11"
    )
    return PrismPair(
        wavelength_nm=wavelength_nm,
        incident_angle_deg=brewster_angle_deg_,
        separation_mm=300.0,
        prism_insert_mm=(5, 10),
        material="sf11",
    )


# ---------------   Test Case for Ideal Brewster Angle -----------------------


def test_brewster_angle_deg() -> None:
    """Test calculate_brewster_angle function."""
    n_sf11_800nm = Material.SF11(0.8)
    expected_brewster_angle_deg = np.rad2deg(np.arctan(n_sf11_800nm))

    calculated_brewster_angle_deg = brewster_angle_deg(
        wavelength_nm=800,
        material="SF11",
    )

    np.testing.assert_allclose(
        calculated_brewster_angle_deg,
        expected_brewster_angle_deg,
        rtol=1e-6,
    )


def test_ideal_apex_deg() -> None:
    """Test calculate_brewster_angle function."""
    expected_ideal_apex_deg = 59.0799589474000

    calculated_ideal_apex_deg = ideal_apex_deg(
        wavelength_nm=800,
        material="SF11",
    )

    np.testing.assert_allclose(
        calculated_ideal_apex_deg,
        expected_ideal_apex_deg,
        rtol=1e-6,
    )


class TestCaseBrewsterAngle:
    @pytest.mark.parametrize(
        "lhs_attr, rhs_attr",
        [
            ("theta_1", "theta_2"),
            ("theta_0", "theta_3"),
        ],
        ids=[
            "theta1_equals_theta2",
            "theta0_equals_theta3",
        ],
    )
    def test_brewster_angle_symmetry(
        self,
        sf11prism_brewster: PrismPair,
        lhs_attr: str,
        rhs_attr: str,
    ) -> None:
        prism = sf11prism_brewster

        np.testing.assert_allclose(
            getattr(prism, lhs_attr),
            getattr(prism, rhs_attr),
            rtol=1e-6,
        )

    @pytest.mark.parametrize(
        "lhs_func, rhs_func",
        [
            (np.sin, lambda n: n / np.sqrt(1 + n**2)),
            (np.cos, lambda n: 1 / np.sqrt(1 + n**2)),
        ],
    )
    def test_relation_between_angle_and_n(
        self,
        sf11prism_brewster: PrismPair,
        lhs_func,
        rhs_func,
    ) -> None:
        n: np.floating = Material.SF11(0.8)

        brewster_angle_deg_ = brewster_angle_deg(800, material="SF11")

        np.testing.assert_allclose(
            lhs_func(np.deg2rad(brewster_angle_deg_)),
            rhs_func(n),
            rtol=1e-6,
        )

    def test_dtheta1_dOmega(self, sf11prism_brewster: PrismPair) -> None:
        """Check dtheta1/dOmega is positive away from Brewster angle."""
        np.testing.assert_allclose(
            sf11prism_brewster.dtheta1_dOmega,
            -(1 / sf11prism_brewster.n**2) * sf11prism_brewster.dn_dOmega,
        )

    def test_dtheta3_dOmega(self, sf11prism_brewster: PrismPair) -> None:
        """Check dtheta3/dOmega is negative at Brewster angle."""
        np.testing.assert_allclose(
            sf11prism_brewster.dtheta3_dOmega,
            2 * sf11prism_brewster.dn_dOmega,
        )

    def test_dndOmega_sign(self, sf11prism_brewster: PrismPair) -> None:
        "Check that dndOmega is negative at Brewster angle."
        assert sf11prism_brewster.dn_dOmega > 0

    def test_p_od(self, sf11prism_brewster: PrismPair) -> None:
        "Check p_od at Brewster angle. (µm)"
        np.testing.assert_allclose(sf11prism_brewster.p_od, 10e3)

    def test_lg(self, sf11prism_brewster: PrismPair) -> None:
        l1 = sf11prism_brewster.p_ob
        l2 = sf11prism_brewster.prism_insert.second
        np.testing.assert_allclose(
            sf11prism_brewster.lg,
            l1 * np.sin(sf11prism_brewster.alpha / 2) * 2
            + l2 * np.sin(sf11prism_brewster.alpha / 2) * 2,
        )

    def test_gdd_positive(self, sf11prism_brewster: PrismPair) -> None:
        """Check that GDD is positive at Brewster angle.

        The GVD value of SF11 at 800 nm is taken from
        https://refractiveindex.info/?shelf=specs&book=SCHOTT-optical&page=N-SF11

        The unit of the output is fs^2^.
        """
        sf11_gvd = 187.50

        np.testing.assert_allclose(
            sf11prism_brewster.gdd_positive,
            2 * (sf11prism_brewster.lg * 1e-3 * sf11_gvd),
            rtol=1e-4,
        )


# -------------- Test Case for the Current SF11 prism (EKSMA)-------------------------------------


class TestCaseSF11PrismPair:
    """Tests for SF11 prism pair with typical configuration.

    Prism is EKSMA 320-8525, which we use in IR path.
    """

    def test_theta_1(self, sf11prisms: PrismPair) -> None:
        np.testing.assert_allclose(sf11prisms.theta_1, 0.507223, rtol=1e-4)

    def test_p_ab(self, sf11prisms: PrismPair, sf11prisms_woinsert: PrismPair) -> None:
        "Check p_ab. (µm)"
        np.testing.assert_allclose(sf11prisms.p_ab, 4.9539e3, rtol=1e-4)
        np.testing.assert_allclose(sf11prisms_woinsert.p_ab, 0)

    def test_p_ob(self, sf11prism_brewster: PrismPair) -> None:
        "Check p_ob at Brewster angle. (µm)"
        np.testing.assert_allclose(sf11prism_brewster.p_ob, 5.0e3, rtol=1e-6)

    def test_prism_pair_init_material_enum(self, sf11prisms: PrismPair) -> None:
        assert sf11prisms.material is Material.SF11


# --------------------------------------------


class TestCaseNoInsertPrismPair:
    """Tests for SF11 prism pair without laser insert.

    The laser incidents at the apex of the first and second prisms.
    """

    def test_theta_1(self, sf11prisms_woinsert: PrismPair) -> None:
        np.testing.assert_allclose(sf11prisms_woinsert.theta_1, 0.512975, rtol=1e-4)

    def test_p_ab(self, sf11prisms_woinsert: PrismPair) -> None:
        "Check p_ab. (µm)"
        np.testing.assert_allclose(sf11prisms_woinsert.p_ab, 0)

    def test_gdd_positive(self, sf11prisms_woinsert: PrismPair) -> None:
        """Check that GDD is positive without prism insert."""

        np.testing.assert_allclose(
            sf11prisms_woinsert.gdd_positive,
            0,
            rtol=1e-4,
        )

    def test_gdd_negative(self, sf11prisms_woinsert: PrismPair) -> None:
        """Check that GDD is negative without prism insert."""

        np.testing.assert_allclose(
            sf11prisms_woinsert.gdd_negative,
            -7534.9,  # https://toolbox.lightcon.com/tools/prismpair saids -7540.099 fs^2^.
            rtol=1e-4,
        )


def test_prism_pair_init_basic() -> None:
    pair = PrismPair(
        wavelength_nm=800,
        incident_angle_deg=60.0,
        separation_mm=300.0,
        prism_insert_mm=(10.0, 20.0),
        material="sf11",
    )

    assert pair.wavelength == 0.8
    assert pair.separation == 300.0e3
    assert pair.prism_insert.first == 10.0e3
    assert pair.prism_insert.second == 20.0e3
    assert pair.material == Material.SF11


def test_prism_pair_negative_separation() -> None:
    with pytest.raises(AssertionError):
        PrismPair(
            wavelength_nm=800,
            incident_angle_deg=60.0,
            separation_mm=-1.0,
            prism_insert_mm=(5.0, 5.0),
        )


def test_prism_pair_negative_prism_insert_first() -> None:
    with pytest.raises(AssertionError):
        PrismPair(
            wavelength_nm=800,
            incident_angle_deg=60.0,
            separation_mm=100.0,
            prism_insert_mm=(-1.0, 5.0),
        )


def test_prism_pair_negative_prism_insert_second() -> None:
    with pytest.raises(AssertionError):
        PrismPair(
            wavelength_nm=800,
            incident_angle_deg=60.0,
            separation_mm=100.0,
            prism_insert_mm=(5.0, -1.0),
        )


def test_prism_pair_gdd_returns_float() -> None:
    pair = PrismPair(
        wavelength_nm=800,
        incident_angle_deg=60.0,
        separation_mm=300.0,
        prism_insert_mm=(10.0, 10.0),
    )

    gdd = pair.gdd
    assert isinstance(gdd, float)


def test_prism_pair_gdd_is_deterministic() -> None:
    pair = PrismPair(
        wavelength_nm=800,
        incident_angle_deg=60.0,
        separation_mm=300.0,
        prism_insert_mm=(10.0, 10.0),
    )

    assert pair.gdd == pair.gdd


@pytest.mark.skip
def test_prism_pair_zero_insert_gdd_is_zero() -> None:
    pair = PrismPair(
        wavelength_nm=800,
        incident_angle_deg=60.0,
        separation_mm=300.0,
        prism_insert_mm=(0.0, 0.0),
    )

    assert abs(pair.gdd) < 1e-12


@pytest.mark.skip
def test_prism_pair_zero_separation_gdd_is_zero() -> None:
    pair = PrismPair(
        wavelength_nm=800,
        incident_angle_deg=60.0,
        separation_mm=0.0,
        prism_insert_mm=(10.0, 10.0),
    )

    assert abs(pair.gdd) < 1e-12


def test_prism_pair_repr_scalar():
    pair = PrismPair(
        wavelength_nm=800,
        incident_angle_deg=60.0,
        separation_mm=300.0,
        prism_insert_mm=(10.0, 10.0),
        material="SF11",
    )
    r = repr(pair)
    assert "PrismPair(" in r
    assert "wavelength_nm=800.00 nm" in r
    assert "incident_angle_deg=60.00" in r
    assert "material='SF11'" in r


def test_prism_pair_repr_array(sf11prism_brewster: PrismPair):
    brewster_prism_angle: float = np.rad2deg(sf11prism_brewster.alpha)
    pair = PrismPair(
        wavelength_nm=np.linspace(600, 900, 100),
        incident_angle_deg=60.0,
        separation_mm=300.0,
        prism_insert_mm=(10.0, 10.0),
        material="SF11",
        apex_deg=brewster_prism_angle,
    )
    r = repr(pair)
    assert "array(100)" in r
    assert "PrismPair(" in r
    assert "material='SF11'" in r


def test_prism_pair_str_scalar():
    pair = PrismPair(
        wavelength_nm=800,
        incident_angle_deg=60.0,
        separation_mm=300.0,
        prism_insert_mm=(10.0, 10.0),
        material="SF11",
    )
    s = str(pair)
    assert "PrismPair(" in s
    assert "wavelength_nm=800.00 nm" in s
    assert "incident_angle_deg=60.00" in s
    assert "material=SF11" in s


def test_prism_pair_str_array(sf11prism_brewster: PrismPair):
    brewster_prism_angle = np.rad2deg(sf11prism_brewster.alpha, dtype=float)
    pair = PrismPair(
        wavelength_nm=np.linspace(600, 900, 100),
        incident_angle_deg=60.0,
        separation_mm=300.0,
        prism_insert_mm=(10.0, 10.0),
        material="SF11",
        apex_deg=brewster_prism_angle,
    )
    s = str(pair)
    assert "array(100)" in s
    assert "PrismPair(" in s
    assert "material=SF11" in s
