"""
Tests for Milestone 4: Degree-3 O(3) Invariants (Quadratic Forms + Triple Traces).

Tests verify:
- Exactly 60 quadratic forms and 56 triple traces (116 total)
- Rotational invariance: full O(3) vector stable under SO(3) rotations
- Reflection invariance: O(3) invariants unchanged under improper rotations
- Burnside count verification for triple traces
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pykarambola.api import minkowski_tensors
from pykarambola.invariants import (
    VECTORS, RANK2_TENSORS,
    decompose_all, _degree1_scalars, _degree2_contractions,
    _quadratic_forms, _triple_traces, _degree3_o3_contractions,
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


def _torus_mesh(R=3.0, r=0.5, n_major=32, n_minor=16):
    """Build a torus mesh (non-convex)."""
    verts = []
    faces = []

    for i in range(n_major):
        theta = 2 * np.pi * i / n_major
        for j in range(n_minor):
            phi = 2 * np.pi * j / n_minor
            x = (R + r * np.cos(phi)) * np.cos(theta)
            y = (R + r * np.cos(phi)) * np.sin(theta)
            z = r * np.sin(phi)
            verts.append([x, y, z])

    for i in range(n_major):
        for j in range(n_minor):
            i_next = (i + 1) % n_major
            j_next = (j + 1) % n_minor
            v0 = i * n_minor + j
            v1 = i_next * n_minor + j
            v2 = i_next * n_minor + j_next
            v3 = i * n_minor + j_next
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


# -----------------------------------------------------------------------------
# Helper: rotate/reflect tensors
# -----------------------------------------------------------------------------

def _transform_tensors(tensors, R):
    """Apply transformation matrix R to all tensors in a tensors_dict."""
    transformed = {}

    # Scalars are invariant
    for name in ['w000', 'w100', 'w200', 'w300']:
        transformed[name] = tensors[name]

    # Vectors transform as v' = R @ v
    for name in VECTORS:
        transformed[name] = R @ tensors[name]

    # Rank-2 tensors transform as M' = R @ M @ R^T
    for name in RANK2_TENSORS:
        M = np.asarray(tensors[name])
        transformed[name] = np.einsum('ia,jb,ab->ij', R, R, M)

    return transformed


# -----------------------------------------------------------------------------
# Test: Count checks
# -----------------------------------------------------------------------------

class TestDegree3Counts:
    """Test that the correct number of invariants are produced."""

    @pytest.fixture
    def box_decomposed(self):
        """Decomposed tensors from a box mesh."""
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        return decompose_all(tensors)

    def test_quadratic_forms_count(self, box_decomposed):
        """Should produce exactly 60 quadratic form invariants."""
        invariants, labels = _quadratic_forms(box_decomposed)
        assert len(invariants) == 60
        assert len(labels) == 60

    def test_triple_traces_count(self, box_decomposed):
        """Should produce exactly 56 triple trace invariants (Burnside: C(8,3))."""
        invariants, labels = _triple_traces(box_decomposed)
        assert len(invariants) == 56
        assert len(labels) == 56

    def test_degree3_o3_total_count(self, box_decomposed):
        """Should produce exactly 116 degree-3 O(3) invariants."""
        invariants, labels = _degree3_o3_contractions(box_decomposed)
        assert len(invariants) == 116
        assert len(labels) == 116


# -----------------------------------------------------------------------------
# Test: Burnside count verification
# -----------------------------------------------------------------------------

class TestBurnsideCount:
    """Verify the triple trace count matches Burnside's lemma prediction."""

    def test_burnside_formula(self):
        """C(n+2,3) = n(n+1)(n+2)/6 for n=6 gives 56."""
        n = 6
        expected = n * (n + 1) * (n + 2) // 6
        assert expected == 56

    def test_multiset_enumeration(self):
        """Verify we enumerate exactly C(8,3) = 56 multisets."""
        from itertools import combinations_with_replacement
        multisets = list(combinations_with_replacement(range(6), 3))
        assert len(multisets) == 56


# -----------------------------------------------------------------------------
# Test: Quadratic form symmetry
# -----------------------------------------------------------------------------

class TestQuadraticFormSymmetry:
    """Test symmetry properties of quadratic forms."""

    @pytest.fixture
    def decomposed(self):
        """Decomposed tensors from an ellipsoid mesh."""
        verts, faces = _ellipsoid_mesh(1.0, 2.0, 3.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        return decompose_all(tensors)

    def test_quadratic_form_symmetry_in_vectors(self, decomposed):
        """v_i^T T_k v_j == v_j^T T_k v_i for symmetric T_k."""
        from pykarambola.invariants import _VECTOR_ALIASES, _TRACELESS_ALIASES

        vectors = [decomposed[(_VECTOR_ALIASES[f'v{i}'], '1o')] for i in range(4)]
        traceless = [decomposed[(_TRACELESS_ALIASES[f'T{k}'], '2e')] for k in range(6)]

        for k in range(6):
            T_k = traceless[k]
            for i in range(4):
                for j in range(4):
                    qf_ij = np.einsum('a,ab,b->', vectors[i], T_k, vectors[j])
                    qf_ji = np.einsum('a,ab,b->', vectors[j], T_k, vectors[i])
                    assert qf_ij == pytest.approx(qf_ji, rel=1e-14), \
                        f"Asymmetry in qf_v{i}_T{k}_v{j}"


# -----------------------------------------------------------------------------
# Test: Rotational invariance
# -----------------------------------------------------------------------------

class TestRotationalInvariance:
    """Test that O(3) invariants are stable under SO(3) rotations."""

    @pytest.mark.parametrize("mesh_type", ['box', 'icosphere', 'ellipsoid', 'torus'])
    def test_rotational_invariance_100_trials(self, mesh_type):
        """Full invariant vector should be stable under 100 random SO(3) rotations."""
        # Generate reference mesh
        if mesh_type == 'box':
            verts, faces = _box_mesh(2.0, 3.0, 4.0)
        elif mesh_type == 'icosphere':
            verts, faces = _icosphere_mesh(1.0, 2)
        elif mesh_type == 'ellipsoid':
            verts, faces = _ellipsoid_mesh(1.0, 2.0, 3.0, 2)
        else:
            verts, faces = _torus_mesh(R=3.0, r=0.5)

        # Compute reference tensors and invariants
        tensors_ref = minkowski_tensors(verts, faces, compute='standard')
        decomposed_ref = decompose_all(tensors_ref)

        scalars_ref, _ = _degree1_scalars(decomposed_ref)
        degree2_ref, _ = _degree2_contractions(decomposed_ref)
        degree3_ref, _ = _degree3_o3_contractions(decomposed_ref)
        invariants_ref = np.concatenate([scalars_ref, degree2_ref, degree3_ref])

        # Test 100 random rotations
        max_deviation = 0.0
        for trial in range(100):
            R = Rotation.random(random_state=trial).as_matrix()

            # Rotate tensors
            tensors_rotated = _transform_tensors(tensors_ref, R)

            # Compute invariants from rotated tensors
            decomposed_rot = decompose_all(tensors_rotated)
            scalars_rot, _ = _degree1_scalars(decomposed_rot)
            degree2_rot, _ = _degree2_contractions(decomposed_rot)
            degree3_rot, _ = _degree3_o3_contractions(decomposed_rot)
            invariants_rot = np.concatenate([scalars_rot, degree2_rot, degree3_rot])

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


# -----------------------------------------------------------------------------
# Test: Reflection invariance
# -----------------------------------------------------------------------------

class TestReflectionInvariance:
    """Test that O(3) invariants are unchanged under reflections."""

    @pytest.mark.parametrize("reflection", [
        np.diag([-1, 1, 1]),   # Reflection in yz-plane
        np.diag([1, -1, 1]),   # Reflection in xz-plane
        np.diag([1, 1, -1]),   # Reflection in xy-plane
        np.diag([-1, -1, 1]),  # Improper rotation (det = -1)
    ])
    def test_reflection_invariance(self, reflection):
        """O(3) invariants should be unchanged under reflections (det = -1)."""
        verts, faces = _ellipsoid_mesh(1.0, 2.0, 3.0)
        tensors_ref = minkowski_tensors(verts, faces, compute='standard')
        decomposed_ref = decompose_all(tensors_ref)

        scalars_ref, _ = _degree1_scalars(decomposed_ref)
        degree2_ref, _ = _degree2_contractions(decomposed_ref)
        degree3_ref, _ = _degree3_o3_contractions(decomposed_ref)
        invariants_ref = np.concatenate([scalars_ref, degree2_ref, degree3_ref])

        # Apply reflection
        tensors_reflected = _transform_tensors(tensors_ref, reflection)
        decomposed_refl = decompose_all(tensors_reflected)

        scalars_refl, _ = _degree1_scalars(decomposed_refl)
        degree2_refl, _ = _degree2_contractions(decomposed_refl)
        degree3_refl, _ = _degree3_o3_contractions(decomposed_refl)
        invariants_refl = np.concatenate([scalars_refl, degree2_refl, degree3_refl])

        # O(3) invariants should match exactly
        np.testing.assert_allclose(invariants_refl, invariants_ref, rtol=1e-12,
            err_msg=f"O(3) invariants changed under reflection {reflection.tolist()}")


# -----------------------------------------------------------------------------
# Test: Combined invariant vector
# -----------------------------------------------------------------------------

class TestCombinedO3Invariants:
    """Test the combined O(3) invariant vector."""

    def test_o3_total_count(self):
        """Total O(3) invariants: 8 + 31 + 116 = 155."""
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        decomposed = decompose_all(tensors)

        scalars, _ = _degree1_scalars(decomposed)
        degree2, _ = _degree2_contractions(decomposed)
        degree3, _ = _degree3_o3_contractions(decomposed)

        total = len(scalars) + len(degree2) + len(degree3)
        assert total == 155, f"Expected 155 O(3) invariants, got {total}"

    def test_no_label_collisions(self):
        """All labels should be unique."""
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        decomposed = decompose_all(tensors)

        _, scalar_labels = _degree1_scalars(decomposed)
        _, degree2_labels = _degree2_contractions(decomposed)
        _, degree3_labels = _degree3_o3_contractions(decomposed)

        all_labels = scalar_labels + degree2_labels + degree3_labels
        assert len(all_labels) == len(set(all_labels)), "Label collision detected"

    def test_deterministic_ordering(self):
        """Labels should be deterministic across calls."""
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        tensors = minkowski_tensors(verts, faces, compute='standard')
        decomposed = decompose_all(tensors)

        _, labels1 = _degree3_o3_contractions(decomposed)
        _, labels2 = _degree3_o3_contractions(decomposed)
        assert labels1 == labels2
