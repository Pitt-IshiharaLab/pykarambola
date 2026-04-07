"""
Tests for Milestone 3: Degree-2 Invariants (Dot Products & Frobenius Inner Products).

Tests verify:
- Exactly 10 dot products and 21 Frobenius products (31 total)
- Symmetry: dot(vi, vj) == dot(vj, vi) and frob(Ti, Tj) == frob(Tj, Ti)
- Rotational invariance: degree-1 + degree-2 vector stable under SO(3) rotations
- Deterministic ordering of labels
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pykarambola.api import minkowski_tensors
from pykarambola.invariants import (
    VECTORS, RANK2_TENSORS,
    decompose_all, _degree1_scalars, _degree2_contractions,
    _vector_dot_products, _frobenius_inner_products,
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


def _random_convex_hull(n_points=50, seed=42):
    """Generate a random convex hull mesh."""
    from scipy.spatial import ConvexHull
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n_points, 3))
    hull = ConvexHull(points)
    return points, hull.simplices.astype(np.int64)


# -----------------------------------------------------------------------------
# Helper: rotate tensors
# -----------------------------------------------------------------------------

def _rotate_tensors(tensors, R):
    """Apply rotation matrix R to all tensors in a tensors_dict."""
    rotated = {}

    # Scalars are invariant
    for name in ['w000', 'w100', 'w200', 'w300']:
        rotated[name] = tensors[name]

    # Vectors transform as v' = R @ v
    for name in VECTORS:
        rotated[name] = R @ tensors[name]

    # Rank-2 tensors transform as M' = R @ M @ R^T
    for name in RANK2_TENSORS:
        M = np.asarray(tensors[name])
        rotated[name] = np.einsum('ia,jb,ab->ij', R, R, M)

    return rotated


# -----------------------------------------------------------------------------
# Test: Count checks
# -----------------------------------------------------------------------------

class TestDegree2Counts:
    """Test that the correct number of invariants are produced."""

    @pytest.fixture
    def box_decomposed(self):
        """Decomposed tensors from a box mesh."""
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        return decompose_all(tensors)

    def test_dot_products_count(self, box_decomposed):
        """Should produce exactly 10 dot product invariants."""
        invariants, labels = _vector_dot_products(box_decomposed)
        assert len(invariants) == 10
        assert len(labels) == 10

    def test_frobenius_products_count(self, box_decomposed):
        """Should produce exactly 21 Frobenius inner product invariants."""
        invariants, labels = _frobenius_inner_products(box_decomposed)
        assert len(invariants) == 21
        assert len(labels) == 21

    def test_degree2_total_count(self, box_decomposed):
        """Should produce exactly 31 degree-2 invariants."""
        invariants, labels = _degree2_contractions(box_decomposed)
        assert len(invariants) == 31
        assert len(labels) == 31


# -----------------------------------------------------------------------------
# Test: Symmetry properties
# -----------------------------------------------------------------------------

class TestDegree2Symmetry:
    """Test symmetry properties of degree-2 contractions."""

    @pytest.fixture
    def decomposed(self):
        """Decomposed tensors from an ellipsoid mesh."""
        verts, faces = _ellipsoid_mesh(1.0, 2.0, 3.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        return decompose_all(tensors)

    def test_dot_product_symmetry(self, decomposed):
        """dot(vi, vj) == dot(vj, vi)."""
        vectors = []
        for name in VECTORS:
            vectors.append(decomposed[(name, '1o')])

        for i in range(4):
            for j in range(4):
                dot_ij = np.dot(vectors[i], vectors[j])
                dot_ji = np.dot(vectors[j], vectors[i])
                assert dot_ij == pytest.approx(dot_ji, rel=1e-14)

    def test_frobenius_symmetry(self, decomposed):
        """frob(Ti, Tj) == frob(Tj, Ti)."""
        traceless = []
        for name in RANK2_TENSORS:
            traceless.append(decomposed[(name, '2e')])

        for i in range(6):
            for j in range(6):
                frob_ij = np.einsum('ab,ab->', traceless[i], traceless[j])
                frob_ji = np.einsum('ab,ab->', traceless[j], traceless[i])
                assert frob_ij == pytest.approx(frob_ji, rel=1e-14)


# -----------------------------------------------------------------------------
# Test: Deterministic ordering
# -----------------------------------------------------------------------------

class TestDegree2DeterministicOrdering:
    """Test that labels are deterministic across calls."""

    def test_dot_product_labels_deterministic(self):
        """Two calls should produce identical dot product labels."""
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        decomposed = decompose_all(tensors)

        _, labels1 = _vector_dot_products(decomposed)
        _, labels2 = _vector_dot_products(decomposed)
        assert labels1 == labels2

    def test_frobenius_labels_deterministic(self):
        """Two calls should produce identical Frobenius labels."""
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        decomposed = decompose_all(tensors)

        _, labels1 = _frobenius_inner_products(decomposed)
        _, labels2 = _frobenius_inner_products(decomposed)
        assert labels1 == labels2

    def test_degree2_labels_deterministic(self):
        """Two calls should produce identical combined labels."""
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        decomposed = decompose_all(tensors)

        _, labels1 = _degree2_contractions(decomposed)
        _, labels2 = _degree2_contractions(decomposed)
        assert labels1 == labels2


# -----------------------------------------------------------------------------
# Test: Rotational invariance
# -----------------------------------------------------------------------------

class TestRotationalInvariance:
    """Test that degree-1 + degree-2 invariants are stable under SO(3) rotations."""

    @pytest.mark.parametrize("mesh_type", ['box', 'icosphere', 'ellipsoid'])
    def test_rotational_invariance_100_trials(self, mesh_type):
        """Invariants should be stable under 100 random SO(3) rotations."""
        # Generate reference mesh
        if mesh_type == 'box':
            verts, faces = _box_mesh(2.0, 3.0, 4.0)
        elif mesh_type == 'icosphere':
            verts, faces = _icosphere_mesh(1.0, 2)
        else:
            verts, faces = _ellipsoid_mesh(1.0, 2.0, 3.0, 2)

        # Compute reference tensors and invariants
        tensors_ref = minkowski_tensors(verts, faces, compute='standard')
        decomposed_ref = decompose_all(tensors_ref)
        scalars_ref, _ = _degree1_scalars(decomposed_ref)
        degree2_ref, _ = _degree2_contractions(decomposed_ref)
        invariants_ref = np.concatenate([scalars_ref, degree2_ref])

        # Test 100 random rotations
        max_deviation = 0.0
        for trial in range(100):
            R = Rotation.random(random_state=trial).as_matrix()

            # Rotate tensors
            tensors_rotated = _rotate_tensors(tensors_ref, R)

            # Compute invariants from rotated tensors
            decomposed_rot = decompose_all(tensors_rotated)
            scalars_rot, _ = _degree1_scalars(decomposed_rot)
            degree2_rot, _ = _degree2_contractions(decomposed_rot)
            invariants_rot = np.concatenate([scalars_rot, degree2_rot])

            # Compute relative deviation
            ref_norm = np.linalg.norm(invariants_ref)
            if ref_norm > 1e-14:
                deviation = np.linalg.norm(invariants_rot - invariants_ref) / ref_norm
            else:
                deviation = np.linalg.norm(invariants_rot - invariants_ref)

            max_deviation = max(max_deviation, deviation)

        # Tolerance: 1e-8
        assert max_deviation < 1e-8, \
            f"Max deviation {max_deviation:.2e} exceeds tolerance 1e-8 for {mesh_type}"

    def test_per_invariant_stability(self):
        """Check each invariant individually for rotational stability."""
        verts, faces = _ellipsoid_mesh(1.0, 2.0, 3.0)
        tensors_ref = minkowski_tensors(verts, faces, compute='standard')
        decomposed_ref = decompose_all(tensors_ref)

        scalars_ref, scalar_labels = _degree1_scalars(decomposed_ref)
        degree2_ref, degree2_labels = _degree2_contractions(decomposed_ref)
        all_labels = scalar_labels + degree2_labels
        invariants_ref = np.concatenate([scalars_ref, degree2_ref])

        # Apply 50 random rotations and track per-invariant max deviation
        per_inv_max = np.zeros(len(invariants_ref))
        for trial in range(50):
            R = Rotation.random(random_state=trial).as_matrix()
            tensors_rotated = _rotate_tensors(tensors_ref, R)
            decomposed_rot = decompose_all(tensors_rotated)
            scalars_rot, _ = _degree1_scalars(decomposed_rot)
            degree2_rot, _ = _degree2_contractions(decomposed_rot)
            invariants_rot = np.concatenate([scalars_rot, degree2_rot])

            # Compute per-invariant relative deviation
            for i in range(len(invariants_ref)):
                ref_val = abs(invariants_ref[i])
                if ref_val > 1e-14:
                    dev = abs(invariants_rot[i] - invariants_ref[i]) / ref_val
                else:
                    dev = abs(invariants_rot[i] - invariants_ref[i])
                per_inv_max[i] = max(per_inv_max[i], dev)

        # All invariants should be stable
        for i, (label, dev) in enumerate(zip(all_labels, per_inv_max)):
            assert dev < 1e-8, f"Invariant {label} has max deviation {dev:.2e}"


# -----------------------------------------------------------------------------
# Test: Combined degree-1 + degree-2 vector
# -----------------------------------------------------------------------------

class TestCombinedInvariants:
    """Test the combined degree-1 and degree-2 invariant vector."""

    def test_combined_count(self):
        """Combined degree-1 + degree-2 should have 8 + 31 = 39 invariants."""
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        decomposed = decompose_all(tensors)

        scalars, scalar_labels = _degree1_scalars(decomposed)
        degree2, degree2_labels = _degree2_contractions(decomposed)

        assert len(scalars) + len(degree2) == 39
        assert len(scalar_labels) + len(degree2_labels) == 39

    def test_no_label_collisions(self):
        """Labels should be unique across degree-1 and degree-2."""
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        decomposed = decompose_all(tensors)

        _, scalar_labels = _degree1_scalars(decomposed)
        _, degree2_labels = _degree2_contractions(decomposed)

        all_labels = scalar_labels + degree2_labels
        assert len(all_labels) == len(set(all_labels)), "Label collision detected"
