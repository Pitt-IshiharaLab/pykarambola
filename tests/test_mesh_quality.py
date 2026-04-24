"""
Tests for pykarambola behavior on meshes with known quality issues.

Covers the seven scenarios from issue #78:
  1. Open surface (boundary edges present)
  2. Degenerate (zero-area) triangles
  3. Isolated vertices (angle sum = 0)
  4. Flat open patch (near-zero volume — tests reference-centroid fallback)
  5. Non-manifold mesh
  6. Single-triangle mesh
  7. Baseline: fully closed clean mesh (sanity check)

Also extends the TestBothPathsAgree pattern from test_accel.py to confirm
that the Python fallback and Cython accelerated paths produce identical
neighbour tables and vertex-triangle lookups for every quality scenario.
All path-agreement tests are skipped when _accel is not compiled.
"""

import os
import unittest.mock
import warnings

import numpy as np
import pytest

import pykarambola.triangulation as tri_module
from pykarambola.triangulation import Triangulation, NEIGHBOUR_UNASSIGNED
from pykarambola.api import minkowski_tensors, minkowski_tensors_from_label_image
from pykarambola.io_off import parse_off_file
from pykarambola.io_obj import parse_obj_file
from pykarambola.io_poly import parse_poly_file

TEST_INPUTS = os.path.join(os.path.dirname(__file__), "fixtures")


# ---------------------------------------------------------------------------
# Mesh construction helpers
# ---------------------------------------------------------------------------

def _box_mesh(a, b, c):
    """Closed box centered at the origin (same geometry as test_api.py)."""
    ha, hb, hc = a / 2.0, b / 2.0, c / 2.0
    verts = np.array([
        [-ha, -hb, -hc],  # 0
        [ ha, -hb, -hc],  # 1
        [ ha,  hb, -hc],  # 2
        [-ha,  hb, -hc],  # 3
        [-ha, -hb,  hc],  # 4
        [ ha, -hb,  hc],  # 5
        [ ha,  hb,  hc],  # 6
        [-ha,  hb,  hc],  # 7
    ], dtype=np.float64)
    faces = np.array([
        [0, 3, 2], [0, 2, 1],   # -z face
        [4, 5, 6], [4, 6, 7],   # +z face
        [0, 1, 5], [0, 5, 4],   # -y face
        [2, 3, 7], [2, 7, 6],   # +y face
        [0, 4, 7], [0, 7, 3],   # -x face
        [1, 2, 6], [1, 6, 5],   # +x face
    ], dtype=np.int64)
    return verts, faces


def _open_box_mesh():
    """Box with the -z face (first 2 triangles) removed, leaving 4 open
    boundary edges on the bottom rim."""
    verts, faces = _box_mesh(2, 3, 4)
    return verts, faces[2:]  # drop triangles 0 and 1 (-z face)


def _single_triangle_mesh():
    """Minimal mesh: one right-angled triangle in the XY plane.
    All three edges are boundary edges."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    return verts, faces


def _flat_patch_mesh():
    """Open flat patch: two triangles forming a unit square in the XY plane.
    Volume is exactly zero (w000 = 0), exercising the reference-centroid
    near-zero denominator fallback."""
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return verts, faces


def _box_with_degenerate_face():
    """Closed box with one extra zero-area (collinear) triangle appended.

    Three new disconnected vertices are added and connected by a single
    degenerate face; they appear in no other triangle, so their
    vertex_angle_sum is 0, exercising the #43 guard in w300/w310/w320.
    The box itself remains topologically closed.
    """
    verts, faces = _box_mesh(2, 3, 4)
    n = len(verts)
    extra_verts = np.array([
        [10.0, 0.0, 0.0],
        [11.0, 0.0, 0.0],
        [12.0, 0.0, 0.0],  # all three collinear -> zero area
    ], dtype=np.float64)
    degen_face = np.array([[n, n + 1, n + 2]], dtype=np.int64)
    return np.vstack([verts, extra_verts]), np.vstack([faces, degen_face])


def _non_manifold_mesh():
    """Two triangles sharing only a single vertex (non-manifold configuration).

    NOTE: all six edges are boundary edges, so the fan traversal sets
    neigh_un=True and cannot confirm the non-manifold condition for the shared
    vertex. This mesh is used for path-agreement tests and no-crash tests only.
    Use _bowtie_non_manifold_mesh() for warning detection tests.
    """
    verts = np.array([
        [ 0.0,  0.0, 0.0],  # 0 — shared non-manifold vertex
        [ 1.0,  0.0, 0.0],  # 1
        [ 0.0,  1.0, 0.0],  # 2
        [-1.0,  0.0, 0.0],  # 3
        [ 0.0, -1.0, 0.0],  # 4
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2],  # triangle 1
        [0, 3, 4],  # triangle 2 — shares only vertex 0 with triangle 1
    ], dtype=np.int64)
    return verts, faces


def _bowtie_non_manifold_mesh():
    """Two closed triangle fans sharing only vertex 0 — a detectable non-manifold.

    Fan 1 ([0,1,2], [0,2,3], [0,3,1]) and fan 2 ([0,4,5], [0,5,6], [0,6,4])
    each form a closed loop around vertex 0. All edges at vertex 0 have
    neighbors (neigh_un=False), but the traversal closes after 3 triangles
    while vertex 0 has 6 incident triangles. This triggers the
    sum_of_triangles != len(tris) check in _get_open_and_nonmanifold.
    """
    verts = np.array([
        [ 0.0,  0.0,  0.0],  # 0 — shared non-manifold vertex
        [ 1.0,  0.0,  0.0],  # 1 — fan 1
        [ 0.5,  1.0,  0.0],  # 2
        [-1.0,  0.0,  0.0],  # 3
        [ 0.0,  0.0,  1.0],  # 4 — fan 2
        [ 0.5,  0.0,  1.5],  # 5
        [-0.5,  0.0,  1.5],  # 6
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2],  # fan 1 — triangle 0
        [0, 2, 3],  # fan 1 — triangle 1
        [0, 3, 1],  # fan 1 — triangle 2 (closes fan 1 at vertex 0)
        [0, 4, 5],  # fan 2 — triangle 0
        [0, 5, 6],  # fan 2 — triangle 1
        [0, 6, 4],  # fan 2 — triangle 2 (closes fan 2 at vertex 0)
    ], dtype=np.int64)
    return verts, faces


# ---------------------------------------------------------------------------
# Path-agreement helpers (mirror test_accel.py)
# ---------------------------------------------------------------------------

def _make_python(verts, faces):
    """Build a Triangulation forcing the pure-Python code path."""
    with unittest.mock.patch.object(tri_module, '_HAS_ACCEL', False):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            return Triangulation.from_arrays(verts, faces)


def _make_accel(verts, faces):
    """Build a Triangulation using the Cython path; skip if not compiled."""
    pytest.importorskip("pykarambola._accel")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return Triangulation.from_arrays(verts, faces)


# ---------------------------------------------------------------------------
# 1. Open surface
# ---------------------------------------------------------------------------

class TestOpenSurface:
    """API behavior on a mesh with open boundary edges.

    check_surface() is wired into minkowski_tensors() (issue #94): a UserWarning
    must be raised and volume-dependent quantities (w000, w020) must be NaN.
    """

    def test_warns_open_surface(self):
        """minkowski_tensors must emit a UserWarning for open meshes."""
        verts, faces = _open_box_mesh()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            minkowski_tensors(verts, faces)
        assert any(
            issubclass(warning.category, UserWarning)
            and "open surface" in str(warning.message).lower()
            for warning in w
        )

    def test_w000_is_nan_for_open_mesh(self):
        """Volume (w000) must be NaN for an open mesh."""
        verts, faces = _open_box_mesh()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = minkowski_tensors(verts, faces)
        assert np.isnan(result['w000'])

    def test_w020_is_nan_for_open_mesh(self):
        """Rank-2 volume tensor (w020) must be all-NaN for an open mesh."""
        verts, faces = _open_box_mesh()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = minkowski_tensors(verts, faces)
        assert np.all(np.isnan(result['w020']))

    def test_surface_area_is_finite_and_positive(self):
        """w100 (area) is defined for open meshes and must be positive."""
        verts, faces = _open_box_mesh()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = minkowski_tensors(verts, faces)
        assert np.isfinite(result['w100'])
        assert result['w100'] > 0

    def test_boundary_edges_present_in_neighbour_table(self):
        """Removing a face must leave boundary edges recorded as NEIGHBOUR_UNASSIGNED."""
        verts, faces = _open_box_mesh()
        t = _make_python(verts, faces)
        assert np.any(t._neighbours == NEIGHBOUR_UNASSIGNED)


# ---------------------------------------------------------------------------
# 2. Degenerate (zero-area) triangles
# ---------------------------------------------------------------------------

class TestDegenerateTriangles:
    """Collinear faces (zero area) must produce a UserWarning and not crash."""

    def test_warns_on_degenerate_face(self):
        verts, faces = _box_with_degenerate_face()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            minkowski_tensors(verts, faces)
        assert any(
            issubclass(warning.category, UserWarning)
            and "degenerate" in str(warning.message).lower()
            for warning in w
        )

    def test_no_crash(self):
        verts, faces = _box_with_degenerate_face()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = minkowski_tensors(verts, faces)
        assert isinstance(result, dict)

    def test_scalar_results_are_finite(self):
        """w100, w200, w300 are always defined; w000 is NaN because the isolated
        degenerate triangle creates open boundary edges (open-surface NaN rule)."""
        verts, faces = _box_with_degenerate_face()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = minkowski_tensors(verts, faces)
        for key in ('w100', 'w200', 'w300'):
            assert np.isfinite(result[key]), f"{key} is not finite"

    def test_rank2_results_are_finite(self):
        """w120, w220, w320, w102, w202 are always defined; w020 is NaN for the
        same reason as w000 (open boundary from the isolated degenerate face)."""
        verts, faces = _box_with_degenerate_face()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = minkowski_tensors(verts, faces)
        for key in ('w120', 'w220', 'w320', 'w102', 'w202'):
            assert np.all(np.isfinite(result[key])), f"{key} contains non-finite values"


# ---------------------------------------------------------------------------
# 3. Isolated vertices (zero angle sum) — issue #43
# ---------------------------------------------------------------------------

class TestIsolatedVertices:
    """Vertices appearing only in zero-area faces have vertex_angle_sum = 0.

    Exercises the guard in calculate_w300/w310/w320 (#43) through a real
    degenerate mesh rather than by directly patching _vertex_angle_sums.
    The three extra collinear vertices in _box_with_degenerate_face() have
    angle_sum = 0 because no non-degenerate triangle is incident on them.
    """

    def test_w300_is_finite(self):
        verts, faces = _box_with_degenerate_face()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = minkowski_tensors(verts, faces)
        assert np.isfinite(result['w300'])

    def test_zero_angle_sum_warning_is_raised(self):
        verts, faces = _box_with_degenerate_face()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            minkowski_tensors(verts, faces)
        messages = [
            str(warning.message).lower()
            for warning in w
            if issubclass(warning.category, UserWarning)
        ]
        assert any("angle" in m or "degenerate" in m for m in messages)


# ---------------------------------------------------------------------------
# 4. Flat open patch (near-zero volume) — issue #42
# ---------------------------------------------------------------------------

class TestFlatPatch:
    """A flat open surface has w000 = 0, exercising the near-zero denominator
    fallback in get_ref_vec (#42)."""

    def test_no_crash(self):
        verts, faces = _flat_patch_mesh()
        result = minkowski_tensors(verts, faces)
        assert isinstance(result, dict)

    def test_w100_is_finite_and_positive(self):
        """Area is always defined, even for flat open patches."""
        verts, faces = _flat_patch_mesh()
        result = minkowski_tensors(verts, faces)
        assert np.isfinite(result['w100'])
        assert result['w100'] > 0

    def test_reference_centroid_no_crash_and_warns(self):
        """With center='reference_centroid' and w000 = 0, get_ref_vec must
        warn about a near-zero denominator and return zeros rather than
        raising ZeroDivisionError."""
        verts, faces = _flat_patch_mesh()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = minkowski_tensors(verts, faces, center='reference_centroid')
        assert isinstance(result, dict)
        assert any(
            issubclass(warning.category, UserWarning)
            and "near zero" in str(warning.message).lower()
            for warning in w
        )


# ---------------------------------------------------------------------------
# 5. Non-manifold mesh
# ---------------------------------------------------------------------------

class TestNonManifoldMesh:
    """Non-manifold meshes must not crash via the API (issue #94).

    check_surface() is wired into minkowski_tensors(): a UserWarning must be
    raised but computation must continue (unlike the CLI which raises RuntimeError).
    """

    def test_no_crash(self):
        verts, faces = _non_manifold_mesh()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = minkowski_tensors(verts, faces)
        assert isinstance(result, dict)

    def test_warns_non_manifold(self):
        """minkowski_tensors must emit a UserWarning for detectable non-manifold meshes.

        Uses _bowtie_non_manifold_mesh: two closed fans sharing vertex 0.
        The fan traversal closes after 3 triangles but vertex 0 has 6,
        triggering the non-manifold warning without open-edge interference.
        """
        verts, faces = _bowtie_non_manifold_mesh()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            minkowski_tensors(verts, faces)
        assert any(
            issubclass(warning.category, UserWarning)
            and "non-manifold" in str(warning.message).lower()
            for warning in w
        )

    def test_w100_and_w200_are_finite(self):
        """Surface-area and curvature integrals are defined even for
        non-manifold meshes."""
        verts, faces = _non_manifold_mesh()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = minkowski_tensors(verts, faces)
        for key in ('w100', 'w200'):
            assert np.isfinite(result[key]), f"{key} is not finite"


# ---------------------------------------------------------------------------
# 6. Single-triangle mesh
# ---------------------------------------------------------------------------

class TestSingleTriangle:
    """A single triangle is the most degenerate valid mesh; must not crash."""

    def test_no_crash(self):
        verts, faces = _single_triangle_mesh()
        result = minkowski_tensors(verts, faces)
        assert isinstance(result, dict)

    def test_w100_is_finite_and_positive(self):
        verts, faces = _single_triangle_mesh()
        result = minkowski_tensors(verts, faces)
        assert np.isfinite(result['w100'])
        assert result['w100'] > 0

    def test_all_edges_are_boundary(self):
        """All three edges of a single triangle have no neighbour."""
        verts, faces = _single_triangle_mesh()
        t = _make_python(verts, faces)
        assert np.all(t._neighbours == NEIGHBOUR_UNASSIGNED)


# ---------------------------------------------------------------------------
# 7. Baseline: fully closed clean mesh
# ---------------------------------------------------------------------------

class TestBaselineClosed:
    """Sanity check: a clean closed mesh produces no warnings and correct scalars."""

    def test_no_warnings(self):
        verts, faces = _box_mesh(2, 3, 4)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            minkowski_tensors(verts, faces)
        assert len(w) == 0, f"Unexpected warnings: {[str(x.message) for x in w]}"

    def test_no_boundary_edges(self):
        verts, faces = _box_mesh(2, 3, 4)
        t = _make_python(verts, faces)
        assert np.all(t._neighbours != NEIGHBOUR_UNASSIGNED)


# ---------------------------------------------------------------------------
# Path-agreement tests across mesh quality scenarios
# (extends TestBothPathsAgree from test_accel.py)
# All tests skipped when _accel is not compiled.
# ---------------------------------------------------------------------------

_QUALITY_MESHES = [
    ("open_box",         _open_box_mesh),
    ("single_triangle",  _single_triangle_mesh),
    ("flat_patch",       _flat_patch_mesh),
    ("degenerate_face",  _box_with_degenerate_face),
    ("non_manifold",     _non_manifold_mesh),
]


class TestBothPathsAgreeMeshQuality:
    """Python fallback and Cython accelerated paths must produce identical
    neighbour tables and vertex-triangle lookups for all quality scenarios.

    Skipped when _accel is not compiled.
    """

    @pytest.mark.parametrize("label,mesh_fn", _QUALITY_MESHES)
    def test_neighbour_tables_match(self, label, mesh_fn):
        verts, faces = mesh_fn()
        t_py = _make_python(verts, faces)
        t_ac = _make_accel(verts, faces)
        np.testing.assert_array_equal(
            t_py._neighbours, t_ac._neighbours,
            err_msg=f"Neighbour tables differ for '{label}'",
        )

    @pytest.mark.parametrize("label,mesh_fn", _QUALITY_MESHES)
    def test_vertex_triangle_lists_match(self, label, mesh_fn):
        verts, faces = mesh_fn()
        t_py = _make_python(verts, faces)
        t_ac = _make_accel(verts, faces)
        for v in range(len(verts)):
            py_tris = sorted(t_py.get_triangles_of_vertex(v))
            ac_tris = sorted(t_ac.get_triangles_of_vertex(v))
            assert py_tris == ac_tris, (
                f"'{label}' vertex {v}: python={py_tris}, accel={ac_tris}"
            )


# ---------------------------------------------------------------------------
# Issue #90: Triangulation objects can be passed directly to minkowski_tensors
# ---------------------------------------------------------------------------

class TestTriangulationInput:
    """minkowski_tensors must accept a Triangulation object as its first argument.

    Tests cover both array-mode (from_arrays) and append-mode (parser) Triangulations.
    """

    def test_from_arrays_triangulation_accepted(self):
        """Triangulation built with from_arrays can be passed directly."""
        verts, faces = _box_mesh(2, 3, 4)
        tri = Triangulation.from_arrays(verts, faces)
        result = minkowski_tensors(tri)
        assert isinstance(result, dict)
        assert np.isfinite(result['w000'])

    def test_from_arrays_matches_array_input(self):
        """Results from Triangulation input must match results from raw arrays."""
        verts, faces = _box_mesh(2, 3, 4)
        tri = Triangulation.from_arrays(verts, faces)
        result_tri = minkowski_tensors(tri)
        result_arr = minkowski_tensors(verts, faces)
        assert result_tri['w000'] == pytest.approx(result_arr['w000'], rel=1e-10)
        assert result_tri['w100'] == pytest.approx(result_arr['w100'], rel=1e-10)

    def test_off_parser_triangulation_accepted(self):
        """Triangulation returned by parse_off_file can be passed directly."""
        filepath = os.path.join(TEST_INPUTS, "cuboid-colorsNoLabels.off")
        tri = parse_off_file(filepath)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = minkowski_tensors(tri)
        assert isinstance(result, dict)
        assert np.isfinite(result['w100'])

    def test_obj_parser_triangulation_accepted(self):
        """Triangulation returned by parse_obj_file can be passed directly."""
        filepath = os.path.join(TEST_INPUTS, "box.obj")
        tri = parse_obj_file(filepath)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = minkowski_tensors(tri)
        assert isinstance(result, dict)
        assert np.isfinite(result['w100'])

    def test_poly_parser_triangulation_accepted(self):
        """Triangulation returned by parse_poly_file can be passed directly."""
        filepath = os.path.join(TEST_INPUTS, "box_a=2_b=3_c=4.poly")
        tri = parse_poly_file(filepath)
        result = minkowski_tensors(tri)
        assert isinstance(result, dict)
        assert np.isfinite(result['w100'])

    def test_glb_parser_triangulation_accepted(self, tmp_path):
        """Triangulation returned by parse_glb_file can be passed directly."""
        trimesh = pytest.importorskip("trimesh")
        from pykarambola.io_glb import parse_glb_file
        box = trimesh.creation.box()
        glb_path = tmp_path / "box.glb"
        box.export(str(glb_path))
        tri = parse_glb_file(str(glb_path))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = minkowski_tensors(tri)
        assert isinstance(result, dict)
        assert np.isfinite(result['w100'])

    def test_faces_none_raises_without_triangulation(self):
        """Omitting faces when verts is a plain array must raise TypeError."""
        verts, _ = _box_mesh(2, 3, 4)
        with pytest.raises(TypeError, match="faces must be provided"):
            minkowski_tensors(verts)


# ---------------------------------------------------------------------------
# Issue #99: pad=True/False for boundary-touching objects
# ---------------------------------------------------------------------------

class TestPadParameter:
    """Tests for the pad parameter in minkowski_tensors_from_label_image.

    pad=True (default) adds a 1-voxel zero border before marching_cubes,
    ensuring objects touching the array boundary produce closed surfaces.
    pad=False leaves boundary-touching objects open, triggering the
    open-surface warning wired in via issue #94.
    """

    def test_pad_closes_boundary_touching_object(self):
        """pad=True must suppress open-surface warnings for boundary-touching objects."""
        pytest.importorskip("skimage")
        vol = np.zeros((10, 10, 10), dtype=int)
        vol[0:5, 0:5, 0:5] = 1  # touches three boundary faces
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            minkowski_tensors_from_label_image(vol, pad=True)
        assert not any("open surface" in str(x.message).lower() for x in w)

    def test_no_pad_boundary_touching_warns(self):
        """pad=False must emit an open-surface warning for boundary-touching objects."""
        pytest.importorskip("skimage")
        vol = np.zeros((10, 10, 10), dtype=int)
        vol[0:5, 0:5, 0:5] = 1
        with pytest.warns(UserWarning, match="[Oo]pen surface"):
            minkowski_tensors_from_label_image(vol, pad=False)
