"""Tests for the pure-Python Wigner 3j implementation."""

import math
import pytest
from pykarambola.spherical import _wigner3j


class TestSelectionRules:
    """Verify that selection rules correctly return 0."""

    def test_m_sum_nonzero(self):
        assert _wigner3j(1, 1, 1, 1, 1, 0) == 0.0

    def test_m_exceeds_j(self):
        assert _wigner3j(2, 2, 2, 3, -1, -2) == 0.0

    def test_triangle_inequality(self):
        assert _wigner3j(1, 1, 3, 0, 0, 0) == 0.0

    def test_odd_j_sum(self):
        # j1+j2+j3 = 3 is odd, all m=0
        assert _wigner3j(1, 1, 1, 0, 0, 0) == 0.0


class TestKnownValues:
    """Verify against known analytical values."""

    def test_110_1m10(self):
        # (1,1,0; 1,-1,0) = (-1)^(1-1-0) / sqrt(3) = 1/sqrt(3)
        expected = 1.0 / math.sqrt(3)
        assert _wigner3j(1, 1, 0, 1, -1, 0) == pytest.approx(expected)

    def test_222_000(self):
        # (2,2,2; 0,0,0) -- verified against sympy
        expected = -0.23904572186687872
        assert _wigner3j(2, 2, 2, 0, 0, 0) == pytest.approx(expected)

    def test_224_000(self):
        # (2,2,4; 0,0,0) -- verified against sympy
        expected = 0.23904572186687872
        assert _wigner3j(2, 2, 4, 0, 0, 0) == pytest.approx(expected)

    def test_110_m110(self):
        # (1,1,0; -1,1,0) = (-1)^(1-1-0) * 1/sqrt(3) ... sign from (-1)^(j1-j2-m3)
        # Actually (1,1,0; -1,1,0): (-1)^(1-1-0) / sqrt(3) = 1/sqrt(3)
        expected = 1.0 / math.sqrt(3)
        assert _wigner3j(1, 1, 0, -1, 1, 0) == pytest.approx(expected)

    def test_symmetry(self):
        # Even permutation of columns: unchanged
        # (j1,j2,j3;m1,m2,m3) = (j2,j3,j1;m2,m3,m1)
        val1 = _wigner3j(2, 3, 4, 1, -2, 1)
        val2 = _wigner3j(3, 4, 2, -2, 1, 1)
        assert val1 == pytest.approx(val2)

    def test_all_zero_m_l246(self):
        # (2,4,6; 0,0,0)
        # Known: (-1)^6 * sqrt(delta) * sum
        # Use the fact that this is a well-known value
        val = _wigner3j(2, 4, 6, 0, 0, 0)
        # Verify it's nonzero (J=12 is even, triangle ok)
        assert val != 0.0
        # Check sign: should be positive for (2,4,6;0,0,0)
        assert val > 0


class TestConsistency:
    """Cross-check properties of the implementation."""

    def test_orthogonality_sum(self):
        # Sum over m1,m2 of (2*j3+1) * 3j(j1,j2,j3;m1,m2,m3)^2 = 1
        # for valid j3, m3
        j1, j2, j3, m3 = 2, 2, 2, 0
        s = 0.0
        for m1 in range(-j1, j1 + 1):
            m2 = -m1 - m3
            if abs(m2) > j2:
                continue
            w = _wigner3j(j1, j2, j3, m1, m2, m3)
            s += w * w
        expected = 1.0 / (2 * j3 + 1)
        assert s == pytest.approx(expected)
