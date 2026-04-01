"""
Port of test_box.cpp - tests Minkowski tensors on box geometries.
"""

import os
import math
import numpy as np
import pytest

from pykarambola.triangulation import LABEL_UNASSIGNED
from pykarambola.io_poly import parse_poly_file
from pykarambola.surface import check_surface
from pykarambola.results import CalcOptions
from pykarambola.minkowski import (
    calculate_w000, calculate_w100, calculate_w200, calculate_w300,
    calculate_w010, calculate_w110, calculate_w210, calculate_w310,
    calculate_w020, calculate_w120, calculate_w220, calculate_w320,
    calculate_w102, calculate_w202, calculate_w203, calculate_w303,
    calculate_w204, calculate_w304,
)
from pykarambola.eigensystem import calculate_eigensystem

TEST_INPUTS = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_box(filename):
    """Load and prepare a box surface from a .poly file."""
    filepath = os.path.join(TEST_INPUTS, filename)
    surface = parse_poly_file(filepath, with_labels=False)
    surface.create_vertex_polygon_lookup_table()
    surface.create_polygon_polygon_lookup_table()
    co = CalcOptions()
    co.labels_set = False
    check_surface(co, surface)
    return surface


def _sorted_eigenvalues(w_matrix, label):
    """Get sorted eigenvalues for a matrix result."""
    eigsys = calculate_eigensystem(w_matrix)
    vals = sorted(eigsys[label].result.eigen_values)
    return np.array(vals)


class TestBox:
    """Test case: axis-aligned box with a=2, b=3, c=4."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.surface = _load_box("box_a=2_b=3_c=4.poly")
        self.a, self.b, self.c = 2.0, 3.0, 4.0
        self.pi = math.pi
        self.label = LABEL_UNASSIGNED

    def test_w000(self):
        """Volume = a*b*c."""
        w000 = calculate_w000(self.surface)
        assert w000[self.label].result == pytest.approx(
            self.a * self.b * self.c, rel=1e-4)

    def test_w100(self):
        """Surface area = 2/3 * (ab + ac + bc)."""
        w100 = calculate_w100(self.surface)
        expected = 2.0 / 3.0 * (self.a * self.b + self.a * self.c + self.b * self.c)
        assert w100[self.label].result == pytest.approx(expected, rel=1e-4)

    def test_w200(self):
        """Mean curvature = pi/3 * (a+b+c)."""
        w200 = calculate_w200(self.surface)
        expected = self.pi / 3.0 * (self.a + self.b + self.c)
        assert w200[self.label].result == pytest.approx(expected, rel=1e-4)

    def test_w300(self):
        """Euler characteristic = 4*pi/3."""
        w300 = calculate_w300(self.surface)
        expected = 4.0 * self.pi / 3.0
        assert w300[self.label].result == pytest.approx(expected, rel=1e-4)

    def test_w010(self):
        """Position vector = (0, 0, 0) for centered box."""
        w010 = calculate_w010(self.surface)
        np.testing.assert_allclose(w010[self.label].result, [0, 0, 0], atol=1e-3)

    def test_w110(self):
        """Surface-weighted position = (0, 0, 0)."""
        w110 = calculate_w110(self.surface)
        np.testing.assert_allclose(w110[self.label].result, [0, 0, 0], atol=1e-3)

    def test_w210(self):
        """Curvature-weighted position = (0, 0, 0)."""
        w210 = calculate_w210(self.surface)
        np.testing.assert_allclose(w210[self.label].result, [0, 0, 0], atol=1e-3)

    def test_w310(self):
        """Gaussian curvature-weighted position = (0, 0, 0)."""
        w310 = calculate_w310(self.surface)
        np.testing.assert_allclose(w310[self.label].result, [0, 0, 0], atol=1e-3)

    def test_w020_eigenvalues(self):
        """w020 eigenvalues = [a^3*b*c/12, b^3*a*c/12, c^3*a*b/12]."""
        a, b, c = self.a, self.b, self.c
        w020 = calculate_w020(self.surface)
        expected = sorted([a**3 * b * c / 12, b**3 * a * c / 12, c**3 * a * b / 12])
        actual = _sorted_eigenvalues(w020, self.label)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w120_eigenvalues(self):
        """w120 eigenvalues."""
        a, b, c = self.a, self.b, self.c
        w120 = calculate_w120(self.surface)
        expected = sorted([
            1.0/6 * (1.0/3 * a**3 * (b + c) + a**2 * b * c),
            1.0/6 * (1.0/3 * b**3 * (a + c) + b**2 * a * c),
            1.0/6 * (1.0/3 * c**3 * (a + b) + c**2 * a * b),
        ])
        actual = _sorted_eigenvalues(w120, self.label)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w220_eigenvalues(self):
        """w220 eigenvalues."""
        a, b, c = self.a, self.b, self.c
        pi = self.pi
        w220 = calculate_w220(self.surface)
        expected = sorted([
            pi / 36 * (a**3 + 3 * a**2 * (b + c)),
            pi / 36 * (b**3 + 3 * b**2 * (a + c)),
            pi / 36 * (c**3 + 3 * c**2 * (a + b)),
        ])
        actual = _sorted_eigenvalues(w220, self.label)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w320_eigenvalues(self):
        """w320 eigenvalues = [pi/3*a^2, pi/3*b^2, pi/3*c^2]."""
        a, b, c = self.a, self.b, self.c
        pi = self.pi
        w320 = calculate_w320(self.surface)
        expected = sorted([pi / 3 * a**2, pi / 3 * b**2, pi / 3 * c**2])
        actual = _sorted_eigenvalues(w320, self.label)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w102_eigenvalues(self):
        """w102 eigenvalues = [2ab/3, 2ac/3, 2bc/3]."""
        a, b, c = self.a, self.b, self.c
        w102 = calculate_w102(self.surface)
        expected = sorted([2.0/3 * a * b, 2.0/3 * a * c, 2.0/3 * b * c])
        actual = _sorted_eigenvalues(w102, self.label)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w202_eigenvalues(self):
        """w202 eigenvalues = [pi/6*(a+b), pi/6*(a+c), pi/6*(b+c)]."""
        a, b, c = self.a, self.b, self.c
        pi = self.pi
        w202 = calculate_w202(self.surface)
        expected = sorted([pi / 6 * (a + b), pi / 6 * (a + c), pi / 6 * (b + c)])
        actual = _sorted_eigenvalues(w202, self.label)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)


class TestSchiefeBox:
    """Test case: skewed box with a=2, b=3, c=4 (rotation invariance)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.surface = _load_box("schiefebox_a=2_b=3_c=4.poly")
        self.a, self.b, self.c = 2.0, 3.0, 4.0
        self.pi = math.pi
        self.label = LABEL_UNASSIGNED

    def test_w000(self):
        w000 = calculate_w000(self.surface)
        assert w000[self.label].result == pytest.approx(
            self.a * self.b * self.c, rel=1e-4)

    def test_w100(self):
        w100 = calculate_w100(self.surface)
        expected = 2.0 / 3.0 * (self.a * self.b + self.a * self.c + self.b * self.c)
        assert w100[self.label].result == pytest.approx(expected, rel=1e-4)

    def test_w200(self):
        w200 = calculate_w200(self.surface)
        expected = self.pi / 3.0 * (self.a + self.b + self.c)
        assert w200[self.label].result == pytest.approx(expected, rel=1e-4)

    def test_w300(self):
        w300 = calculate_w300(self.surface)
        expected = 4.0 * self.pi / 3.0
        assert w300[self.label].result == pytest.approx(expected, rel=1e-4)

    def test_w010(self):
        w010 = calculate_w010(self.surface)
        np.testing.assert_allclose(w010[self.label].result, [0, 0, 0], atol=1e-3)

    def test_w110(self):
        w110 = calculate_w110(self.surface)
        np.testing.assert_allclose(w110[self.label].result, [0, 0, 0], atol=1e-3)

    def test_w210(self):
        w210 = calculate_w210(self.surface)
        np.testing.assert_allclose(w210[self.label].result, [0, 0, 0], atol=1e-3)

    def test_w310(self):
        w310 = calculate_w310(self.surface)
        np.testing.assert_allclose(w310[self.label].result, [0, 0, 0], atol=1e-3)

    def test_w020_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        w020 = calculate_w020(self.surface)
        expected = sorted([a**3 * b * c / 12, b**3 * a * c / 12, c**3 * a * b / 12])
        actual = _sorted_eigenvalues(w020, self.label)
        np.testing.assert_allclose(actual, expected, rtol=5e-3)

    def test_w120_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        w120 = calculate_w120(self.surface)
        expected = sorted([
            1.0/6 * (1.0/3 * a**3 * (b + c) + a**2 * b * c),
            1.0/6 * (1.0/3 * b**3 * (a + c) + b**2 * a * c),
            1.0/6 * (1.0/3 * c**3 * (a + b) + c**2 * a * b),
        ])
        actual = _sorted_eigenvalues(w120, self.label)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w220_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        pi = self.pi
        w220 = calculate_w220(self.surface)
        expected = sorted([
            pi / 36 * (a**3 + 3 * a**2 * (b + c)),
            pi / 36 * (b**3 + 3 * b**2 * (a + c)),
            pi / 36 * (c**3 + 3 * c**2 * (a + b)),
        ])
        actual = _sorted_eigenvalues(w220, self.label)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w320_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        pi = self.pi
        w320 = calculate_w320(self.surface)
        expected = sorted([pi / 3 * a**2, pi / 3 * b**2, pi / 3 * c**2])
        actual = _sorted_eigenvalues(w320, self.label)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w102_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        w102 = calculate_w102(self.surface)
        expected = sorted([2.0/3 * a * b, 2.0/3 * a * c, 2.0/3 * b * c])
        actual = _sorted_eigenvalues(w102, self.label)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w202_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        pi = self.pi
        w202 = calculate_w202(self.surface)
        expected = sorted([pi / 6 * (a + b), pi / 6 * (a + c), pi / 6 * (b + c)])
        actual = _sorted_eigenvalues(w202, self.label)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)


# ============================================================================
# Rotational invariance tests for new rank-3 and rank-4 normal tensors
# ============================================================================
# These compare Frobenius norms on axis-aligned vs. skewed boxes.
# Frobenius norm is SO(3)-invariant, so it should match (up to numerical error).


def _frobenius_norm(tensor_result_dict, label):
    """Compute Frobenius norm of a tensor result."""
    result = tensor_result_dict[label].result
    arr = result.to_numpy() if hasattr(result, 'to_numpy') else np.asarray(result)
    return float(np.linalg.norm(arr))


class TestW203RotInvariance:
    """Verify w203 (curvature-weighted rank-3 normal tensor) is rotation invariant.

    Note: For an axis-aligned box, w203 is exactly zero because averaged normals
    across edges cancel. For the skewed box, it is typically small but nonzero.
    This test verifies that when both norms are small, the difference is acceptable.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.surface_aligned = _load_box("box_a=2_b=3_c=4.poly")
        self.surface_skewed = _load_box("schiefebox_a=2_b=3_c=4.poly")
        self.label = LABEL_UNASSIGNED

    def test_w203_rotinvariance(self):
        w203_aligned = calculate_w203(self.surface_aligned)
        w203_skewed = calculate_w203(self.surface_skewed)
        norm_aligned = _frobenius_norm(w203_aligned, self.label)
        norm_skewed = _frobenius_norm(w203_skewed, self.label)
        # Both norms should be small; absolute difference acceptable
        assert abs(norm_skewed - norm_aligned) < 1e-4


class TestW303RotInvariance:
    """Verify w303 (Gaussian-weighted rank-3 normal tensor) is rotation invariant.

    Note: For an axis-aligned box, w303 is essentially zero due to symmetry.
    This test verifies that when both norms are small, the difference is acceptable.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.surface_aligned = _load_box("box_a=2_b=3_c=4.poly")
        self.surface_skewed = _load_box("schiefebox_a=2_b=3_c=4.poly")
        self.label = LABEL_UNASSIGNED

    def test_w303_rotinvariance(self):
        w303_aligned = calculate_w303(self.surface_aligned)
        w303_skewed = calculate_w303(self.surface_skewed)
        norm_aligned = _frobenius_norm(w303_aligned, self.label)
        norm_skewed = _frobenius_norm(w303_skewed, self.label)
        # Both norms should be small; absolute difference acceptable
        assert abs(norm_skewed - norm_aligned) < 1e-4


class TestW204RotInvariance:
    """Verify w204 (curvature-weighted rank-4 normal tensor) is rotation invariant."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.surface_aligned = _load_box("box_a=2_b=3_c=4.poly")
        self.surface_skewed = _load_box("schiefebox_a=2_b=3_c=4.poly")
        self.label = LABEL_UNASSIGNED

    def test_w204_rotinvariance(self):
        w204_aligned = calculate_w204(self.surface_aligned)
        w204_skewed = calculate_w204(self.surface_skewed)
        norm_aligned = _frobenius_norm(w204_aligned, self.label)
        norm_skewed = _frobenius_norm(w204_skewed, self.label)
        np.testing.assert_allclose(norm_skewed, norm_aligned, rtol=5e-3)


class TestW304RotInvariance:
    """Verify w304 (Gaussian-weighted rank-4 normal tensor) is rotation invariant."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.surface_aligned = _load_box("box_a=2_b=3_c=4.poly")
        self.surface_skewed = _load_box("schiefebox_a=2_b=3_c=4.poly")
        self.label = LABEL_UNASSIGNED

    def test_w304_rotinvariance(self):
        w304_aligned = calculate_w304(self.surface_aligned)
        w304_skewed = calculate_w304(self.surface_skewed)
        norm_aligned = _frobenius_norm(w304_aligned, self.label)
        norm_skewed = _frobenius_norm(w304_skewed, self.label)
        np.testing.assert_allclose(norm_skewed, norm_aligned, rtol=5e-3)
