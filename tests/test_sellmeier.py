"""Unit tests for the pulselaser.sellmeier module.

This module validates refractive index calculations for gases, optical glasses,
and birefringent crystals using Sellmeier equations against reference data
from sources like refractiveindex.info and Thorlabs.
"""

import numpy as np
import pytest
import sympy as sp

from laser_toolbox import sellmeier
from laser_toolbox.sellmeier import Material


class TestAir:
    """Tests for the refractive index of air."""

    def test_at_dline(self) -> None:
        """Verify the refractive index of air at the Sodium D-line (587.6 nm)."""
        # Value source: https://refractiveindex.info/?shelf=other&book=air&page=Ciddor
        np.testing.assert_allclose(sellmeier.air(0.5876), 1.00027717)

    def test_for_negative_derivative(self):
        """Ensure a ValueError is raised when a negative derivative order is requested."""
        with pytest.raises(ValueError):
            sellmeier.air(0.5876, derivative=-1)

    def test_for_sympy(self):
        """Ensure sympy object is returned when as_sympy=True."""
        isinstance(sellmeier.air(0.5876, derivative=0, as_sympy=True), sp.Expr)


class TestBK7:
    """Tests for N-BK7 borosilicate glass."""

    def test_at_800nm(self) -> None:
        """Verify the refractive index of BK7 at 800 nm."""
        assert sellmeier.bk7(0.80) == 1.5107762314198743

    def test_negative_derivative(self) -> None:
        """Ensure a ValueError is raised for invalid derivative orders."""
        with pytest.raises(
            ValueError,
            match="derivative must be equal or greater than zero",
        ):
            sellmeier.bk7(0.5876, derivative=-1)

    def test_sympy_objet(self) -> None:
        """Verify that the as_sympy flag returns a SymPy expression object."""
        assert isinstance(sellmeier.bk7(0.80, as_sympy=True), sp.Expr)


class TestFusedSilica:
    """Tests for Fused Silica (UV-grade)."""

    def test_at_800nm(self) -> None:
        """Verify the refractive index of Fused Silica at 800 nm."""
        assert sellmeier.fused_silica(0.80) == 1.4533172570445876


class TestCaF2:
    """Tests for Calcium Fluoride (CaF2)."""

    @pytest.mark.parametrize(
        ("wavelength", "refractive_index"),
        [
            (0.8, 1.4305724647561817),
            (1.064, 1.428),
        ],
    )
    def test_refractive_index(
        self,
        wavelength: float,
        refractive_index: float,
    ) -> None:
        """Verify the refractive index of CaF2 at 800 nm."""
        np.testing.assert_allclose(
            sellmeier.caf2(wavelength),
            refractive_index,
            atol=0.001,
            rtol=0.001,
        )

    def test_for_sympy(self) -> None:
        """Ensure sympy object is returned when as_sympy=True."""
        isinstance(sellmeier.caf2(0.5876, derivative=0, as_sympy=True), sp.Expr)


class TestSF10:
    """Tests for SF10 dense flint glass."""

    def test_at_800nm(self) -> None:
        """Verify the refractive index of SF10 at 800 nm with a tolerance of 0.1%."""
        np.testing.assert_allclose(
            sellmeier.sf10(0.80),
            1.7112,
            atol=0.001,
            rtol=0.001,
        )


class TestSF11:
    """Tests for SF11 dense flint glass."""

    @pytest.mark.parametrize(
        ("wavelength", "refractive_index"),
        [
            (0.8, 1.7646),
            (0.5876, 1.7865),
        ],
    )
    def test_refractive_index(
        self,
        wavelength: float,
        refractive_index: float,
    ) -> None:
        """Verify the refractive index of CaF2 at 800 nm."""
        np.testing.assert_allclose(
            sellmeier.sf11(wavelength),
            refractive_index,
            atol=0.001,
            rtol=0.001,
        )
        np.testing.assert_allclose(
            Material.SF11(wavelength),
            refractive_index,
            atol=0.001,
            rtol=0.001,
        )


class TestMgF2:
    """Tests for Magnesium Fluoride (MgF2) birefringence."""

    def test_at_dline(self) -> None:
        """Verify ordinary (no) and extraordinary (ne) indices at 586.7 nm.

        Reference values from Thorlabs: ne = 1.390, no = 1.378.
        """
        np.testing.assert_allclose(
            sellmeier.mgf2(0.5867),
            (1.378, 1.390),
            atol=0.001,
            rtol=0.001,
        )

    def test_for_sympy(self) -> None:
        """Ensure sympy object is returned when as_sympy=True."""
        isinstance(sellmeier.mgf2(0.5876, derivative=0, as_sympy=True), tuple)
        isinstance(sellmeier.mgf2(0.5876, derivative=0, as_sympy=True)[0], sp.Expr)
        isinstance(sellmeier.mgf2(0.5876, derivative=0, as_sympy=True)[1], sp.Expr)


class TestCalcite:
    """Tests for Calcite (CaCO3) birefringence."""

    def test_at_YAG(self) -> None:  # noqa: N802
        """Verify refractive indices at the Nd:YAG wavelength (1064 nm).

        Reference values from Thorlabs: ne = 1.480, no = 1.642.
        """
        np.testing.assert_allclose(
            sellmeier.calcite(1.064),
            (1.642, 1.480),
            atol=0.001,
            rtol=0.001,
        )

    def test_for_sympy(self):
        """Ensure sympy object is returned when as_sympy=True."""
        isinstance(sellmeier.calcite(0.5876, derivative=0, as_sympy=True), tuple)
        isinstance(sellmeier.calcite(0.5876, derivative=0, as_sympy=True)[0], sp.Expr)
        isinstance(sellmeier.calcite(0.5876, derivative=0, as_sympy=True)[1], sp.Expr)


class TestQuartz:
    """Tests for Crystalline Quartz."""

    def test_at_800nm(self) -> None:
        """Verify ordinary and extraordinary indices of Quartz at 800 nm."""
        np.testing.assert_allclose(
            sellmeier.quartz(0.80),
            (1.5383355123424691, 1.5472301086112594),
        )

    def test_for_sympy(self):
        """Ensure sympy object is returned when as_sympy=True."""
        isinstance(sellmeier.quartz(0.5876, derivative=0, as_sympy=True), tuple)
        isinstance(sellmeier.quartz(0.5876, derivative=0, as_sympy=True)[0], sp.Expr)
        isinstance(sellmeier.quartz(0.5876, derivative=0, as_sympy=True)[1], sp.Expr)


class TestAlphaBBO:
    """Tests for Alpha-Phase Barium Borate (a-BBO)."""

    @pytest.mark.parametrize(
        ("wavelength", "refractive_index"),
        [
            (0.5876, (1.673, 1.533)),
            (0.8000, (1.6639, 1.5284)),
            (0.4000, (1.6962, 1.5500)),
        ],
    )
    def test_refractive_indexe(
        self,
        wavelength: float,
        refractive_index: tuple[float, float],
    ) -> None:
        """Verify the birefringent refractive indices of a-BBO.

        Multiple wavelengths.
        """
        np.testing.assert_allclose(
            sellmeier.alpha_bbo(wavelength),
            refractive_index,
            rtol=0.001,
            atol=0.001,
        )

    def test_for_sympy(self):
        """Ensure sympy object is returned when as_sympy=True."""
        isinstance(sellmeier.alpha_bbo(0.5876, derivative=0, as_sympy=True), tuple)
        isinstance(sellmeier.alpha_bbo(0.5876, derivative=0, as_sympy=True)[0], sp.Expr)
        isinstance(sellmeier.alpha_bbo(0.5876, derivative=0, as_sympy=True)[1], sp.Expr)


class TestBetaBBO:
    """Tests for Beta-Phase Barium Borate (b-BBO)."""

    def test_at_800nm(self) -> None:
        """Verify ordinary and extraordinary indices of b-BBO at 800 nm."""
        np.testing.assert_allclose(
            sellmeier.beta_bbo(0.800),
            (1.6614, 1.5462),
            atol=0.001,
            rtol=0.001,
        )

    def test_for_sympy(self):
        """Ensure sympy object is returned when as_sympy=True."""
        isinstance(sellmeier.beta_bbo(0.5876, derivative=0, as_sympy=True), tuple)
        isinstance(sellmeier.beta_bbo(0.5876, derivative=0, as_sympy=True)[0], sp.Expr)
        isinstance(sellmeier.beta_bbo(0.5876, derivative=0, as_sympy=True)[1], sp.Expr)


def test_phase_matching_angle_bbo_at_800() -> None:
    """Verify the SHG phase-matching angle for BBO at 800 nm fundamental wavelength."""
    np.testing.assert_allclose(
        sellmeier.phase_matching_angle_bbo(0.800),
        29.21,
        atol=0.01,
    )


def test_phase_matching_angle_bbo_at_790() -> None:
    """Verify the SHG phase-matching angle for BBO at 790 nm fundamental wavelength."""
    np.testing.assert_allclose(
        sellmeier.phase_matching_angle_bbo(0.790),
        29.58,
        atol=0.01,
    )


def test_enum_from_str() -> None:
    mat = Material.from_str("sf11")
    assert isinstance(mat, Material)


def test_unknown_material() -> None:
    with pytest.raises(AttributeError):
        Material.UNKONWN(0.8)
    with pytest.raises(ValueError, match="Unkonwn material:"):
        Material.from_str("unknown")(0.8)
