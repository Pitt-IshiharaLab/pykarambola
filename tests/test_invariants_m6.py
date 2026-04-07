"""
Tests for Milestone 6: Public API, Labels, and Translation Invariance.

Tests verify:
- API contract: len(compute_invariants()) == len(compute_invariant_labels())
- max_degree gating: correct counts at each degree
- Deterministic ordering: identical labels across calls
- Translation behavior: document which invariants are translation-invariant
- Edge cases: error handling for invalid inputs
"""

import numpy as np
import pytest

from pykarambola.api import minkowski_tensors
from pykarambola.invariants import (
    compute_invariants, compute_invariant_labels, _enumerate_invariant_contractions,
)


# -----------------------------------------------------------------------------
# Test fixtures: mesh generation
# -----------------------------------------------------------------------------

def _box_mesh(a, b, c):
    """Build a triangulated axis-aligned box centered at the origin."""
    ha, hb, hc = a / 2.0, b / 2.0, c / 2.0
    verts = np.array([
        [-ha, -hb, -hc], [ha, -hb, -hc], [ha, hb, -hc], [-ha, hb, -hc],
        [-ha, -hb, hc], [ha, -hb, hc], [ha, hb, hc], [-ha, hb, hc],
    ], dtype=np.float64)
    faces = np.array([
        [0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ], dtype=np.int64)
    return verts, faces


def _icosphere_mesh(radius=1.0, subdivisions=2):
    """Build an icosphere mesh."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float64)
    verts = verts / np.linalg.norm(verts[0])
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)
    for _ in range(subdivisions):
        verts, faces = _subdivide_icosphere(verts, faces)
    return verts * radius, faces


def _subdivide_icosphere(verts, faces):
    """Subdivide an icosphere."""
    edge_midpoints = {}
    new_verts = list(verts)

    def get_midpoint(i, j):
        key = (min(i, j), max(i, j))
        if key in edge_midpoints:
            return edge_midpoints[key]
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        idx = len(new_verts)
        new_verts.append(mid)
        edge_midpoints[key] = idx
        return idx

    new_faces = []
    for v0, v1, v2 in faces:
        m01 = get_midpoint(v0, v1)
        m12 = get_midpoint(v1, v2)
        m20 = get_midpoint(v2, v0)
        new_faces.extend([
            [v0, m01, m20], [v1, m12, m01], [v2, m20, m12], [m01, m12, m20],
        ])
    return np.array(new_verts), np.array(new_faces, dtype=np.int64)


def _ellipsoid_mesh(a, b, c, subdivisions=2):
    """Build an ellipsoid mesh."""
    verts, faces = _icosphere_mesh(radius=1.0, subdivisions=subdivisions)
    verts = verts * np.array([a, b, c])
    return verts, faces


# -----------------------------------------------------------------------------
# Test: API contract
# -----------------------------------------------------------------------------

class TestAPIContract:
    """Test that compute_invariants and compute_invariant_labels match."""

    @pytest.fixture
    def tensors(self):
        """Tensors from a box mesh."""
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        return minkowski_tensors(verts, faces, compute='standard')

    @pytest.mark.parametrize("max_degree,symmetry", [
        (1, 'O3'),
        (1, 'SO3'),
        (2, 'O3'),
        (2, 'SO3'),
        (3, 'O3'),
        (3, 'SO3'),
    ])
    def test_invariants_and_labels_same_length(self, tensors, max_degree, symmetry):
        """len(compute_invariants()) == len(compute_invariant_labels())."""
        invariants = compute_invariants(tensors, max_degree=max_degree, symmetry=symmetry)
        labels = compute_invariant_labels(max_degree=max_degree, symmetry=symmetry)
        assert len(invariants) == len(labels), \
            f"Mismatch: {len(invariants)} invariants vs {len(labels)} labels"


# -----------------------------------------------------------------------------
# Test: max_degree gating
# -----------------------------------------------------------------------------

class TestMaxDegreeGating:
    """Test that max_degree controls which invariants are included."""

    @pytest.fixture
    def tensors(self):
        """Tensors from a box mesh."""
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        return minkowski_tensors(verts, faces, compute='standard')

    def test_degree1_count(self, tensors):
        """max_degree=1 should return 8 invariants."""
        inv = compute_invariants(tensors, max_degree=1)
        assert len(inv) == 8

    def test_degree2_count(self, tensors):
        """max_degree=2 should return 8 + 10 + 21 = 39 invariants."""
        inv = compute_invariants(tensors, max_degree=2)
        assert len(inv) == 39

    def test_degree3_o3_count(self, tensors):
        """max_degree=3, symmetry='O3' should return 155 invariants."""
        inv = compute_invariants(tensors, max_degree=3, symmetry='O3')
        assert len(inv) == 155

    def test_degree3_so3_count(self, tensors):
        """max_degree=3, symmetry='SO3' should return 219 invariants."""
        inv = compute_invariants(tensors, max_degree=3, symmetry='SO3')
        assert len(inv) == 219


# -----------------------------------------------------------------------------
# Test: Deterministic ordering
# -----------------------------------------------------------------------------

class TestDeterministicOrdering:
    """Test that labels are deterministic and stable."""

    def test_labels_deterministic_across_calls(self):
        """Two calls should produce identical label lists."""
        labels1 = compute_invariant_labels(max_degree=3, symmetry='SO3')
        labels2 = compute_invariant_labels(max_degree=3, symmetry='SO3')
        assert labels1 == labels2

    def test_invariants_deterministic_across_calls(self):
        """Two calls with same input should produce identical invariants."""
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')

        inv1 = compute_invariants(tensors, max_degree=3, symmetry='SO3')
        inv2 = compute_invariants(tensors, max_degree=3, symmetry='SO3')
        np.testing.assert_array_equal(inv1, inv2)

    def test_all_labels_unique(self):
        """All labels should be unique."""
        labels = compute_invariant_labels(max_degree=3, symmetry='SO3')
        assert len(labels) == len(set(labels)), "Duplicate labels found"


# -----------------------------------------------------------------------------
# Test: Translation behavior
# -----------------------------------------------------------------------------

class TestTranslationBehavior:
    """Test and document translation covariance of invariants."""

    def test_translation_changes_invariants(self):
        """Invariants should change when mesh is translated."""
        verts_orig, faces = _ellipsoid_mesh(1.0, 2.0, 3.0)
        verts_shifted = verts_orig + np.array([10.0, 20.0, 30.0])

        tensors_orig = minkowski_tensors(verts_orig, faces, compute='standard')
        tensors_shifted = minkowski_tensors(verts_shifted, faces, compute='standard')

        inv_orig = compute_invariants(tensors_orig, symmetry='O3')
        inv_shifted = compute_invariants(tensors_shifted, symmetry='O3')

        # Overall they should differ
        assert not np.allclose(inv_orig, inv_shifted)

    def test_translation_invariant_subset(self):
        """Some invariants should be translation-invariant.

        The following are built from w000, w100, w200, w300, w102, w202 only:
        - s0-s3 (first 4 scalars, but NOT s4-s7 which involve w020-w320)
        - frob_T4_T4, frob_T4_T5, frob_T5_T5 (indices 33, 34, 38 at degree 2)
        """
        verts_orig, faces = _ellipsoid_mesh(1.0, 2.0, 3.0)
        verts_shifted = verts_orig + np.array([10.0, 20.0, 30.0])

        tensors_orig = minkowski_tensors(verts_orig, faces, compute='standard')
        tensors_shifted = minkowski_tensors(verts_shifted, faces, compute='standard')

        inv_orig = compute_invariants(tensors_orig, max_degree=2, symmetry='O3')
        inv_shifted = compute_invariants(tensors_shifted, max_degree=2, symmetry='O3')

        # s0-s3 (indices 0-3) should be translation-invariant
        np.testing.assert_allclose(inv_orig[:4], inv_shifted[:4], rtol=1e-10,
            err_msg="s0-s3 should be translation-invariant")

    def test_s4_s7_are_translation_covariant(self):
        """s4-s7 (traces of w020-w320) should change under translation."""
        verts_orig, faces = _ellipsoid_mesh(1.0, 2.0, 3.0)
        verts_shifted = verts_orig + np.array([10.0, 20.0, 30.0])

        tensors_orig = minkowski_tensors(verts_orig, faces, compute='standard')
        tensors_shifted = minkowski_tensors(verts_shifted, faces, compute='standard')

        inv_orig = compute_invariants(tensors_orig, max_degree=1, symmetry='O3')
        inv_shifted = compute_invariants(tensors_shifted, max_degree=1, symmetry='O3')

        # s4-s7 (indices 4-7) should change under translation
        assert not np.allclose(inv_orig[4:8], inv_shifted[4:8]), \
            "s4-s7 should be translation-covariant"


# -----------------------------------------------------------------------------
# Test: Edge cases and error handling
# -----------------------------------------------------------------------------

class TestEdgeCases:
    """Test error handling for invalid inputs."""

    @pytest.fixture
    def tensors(self):
        """Tensors from a box mesh."""
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        return minkowski_tensors(verts, faces, compute='standard')

    def test_invalid_max_degree_raises(self, tensors):
        """Invalid max_degree should raise ValueError."""
        with pytest.raises(ValueError, match="max_degree must be 1, 2, or 3"):
            compute_invariants(tensors, max_degree=0)
        with pytest.raises(ValueError, match="max_degree must be 1, 2, or 3"):
            compute_invariants(tensors, max_degree=4)

    def test_invalid_symmetry_raises(self, tensors):
        """Invalid symmetry should raise ValueError."""
        with pytest.raises(ValueError, match="symmetry must be 'SO3' or 'O3'"):
            compute_invariants(tensors, symmetry='invalid')

    def test_missing_tensor_raises(self):
        """Missing required tensor should raise KeyError."""
        incomplete = {'w000': 1.0}  # Missing most tensors
        with pytest.raises(KeyError):
            compute_invariants(incomplete)

    def test_labels_invalid_max_degree_raises(self):
        """compute_invariant_labels with invalid max_degree should raise."""
        with pytest.raises(ValueError):
            compute_invariant_labels(max_degree=0)

    def test_labels_invalid_symmetry_raises(self):
        """compute_invariant_labels with invalid symmetry should raise."""
        with pytest.raises(ValueError):
            compute_invariant_labels(symmetry='invalid')


# -----------------------------------------------------------------------------
# Test: Enumeration helper
# -----------------------------------------------------------------------------

class TestEnumerationHelper:
    """Test the _enumerate_invariant_contractions helper."""

    def test_o3_counts(self):
        """O(3) counts should match expected values."""
        counts = _enumerate_invariant_contractions(symmetry='O3')
        assert counts['degree1_scalars'] == 8
        assert counts['degree2_dot_products'] == 10
        assert counts['degree2_frobenius'] == 21
        assert counts['degree3_quadratic_forms'] == 60
        assert counts['degree3_triple_traces'] == 56
        assert counts['total'] == 155

    def test_so3_counts(self):
        """SO(3) counts should match expected values."""
        counts = _enumerate_invariant_contractions(symmetry='SO3')
        assert counts['degree1_scalars'] == 8
        assert counts['degree2_dot_products'] == 10
        assert counts['degree2_frobenius'] == 21
        assert counts['degree3_quadratic_forms'] == 60
        assert counts['degree3_triple_traces'] == 56
        assert counts['degree3_triple_vector_dets'] == 4
        assert counts['degree3_commutator_pseudoscalars'] == 60
        assert counts['total'] == 219
