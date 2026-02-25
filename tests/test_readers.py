"""
Tests for .poly, .off, .obj, and .glb file readers.
"""

import os
import tempfile
import pytest
import numpy as np

from pykarambola.io_poly import parse_poly_file
from pykarambola.io_off import parse_off_file, is_off_file
from pykarambola.io_obj import parse_obj_file, is_obj_file
from pykarambola.io_glb import parse_glb_file, is_glb_file
from pykarambola.triangulation import LABEL_UNASSIGNED
from pykarambola.surface import check_surface
from pykarambola.results import CalcOptions
from pykarambola.minkowski import calculate_w000, calculate_w100

TEST_INPUTS = os.path.join(os.path.dirname(__file__), "fixtures")


class TestPolyReader:
    """Tests for .poly file parsing."""

    def test_box_poly(self):
        """Read box.poly and verify vertex/triangle counts."""
        filepath = os.path.join(TEST_INPUTS, "box_a=2_b=3_c=4.poly")
        surface = parse_poly_file(filepath)
        assert surface.n_vertices() == 8
        assert surface.n_triangles() == 12

    def test_box_poly_volume(self):
        """Read box.poly and verify volume computation."""
        filepath = os.path.join(TEST_INPUTS, "box_a=2_b=3_c=4.poly")
        surface = parse_poly_file(filepath)
        surface.create_vertex_polygon_lookup_table()
        surface.create_polygon_polygon_lookup_table()
        w000 = calculate_w000(surface)
        assert w000[LABEL_UNASSIGNED].result == pytest.approx(24.0, rel=1e-4)

    def test_empty_poly(self):
        """Empty poly should raise an error during surface check."""
        filepath = os.path.join(TEST_INPUTS, "empty.poly")
        surface = parse_poly_file(filepath)
        surface.create_vertex_polygon_lookup_table()
        surface.create_polygon_polygon_lookup_table()
        co = CalcOptions()
        with pytest.raises(RuntimeError, match="no polygons"):
            check_surface(co, surface)


class TestOffReader:
    """Tests for .off file parsing."""

    def test_cuboid_off(self):
        """Read cuboid.off and verify vertex/triangle counts."""
        filepath = os.path.join(TEST_INPUTS, "cuboid.off")
        surface = parse_off_file(filepath)
        assert surface.n_vertices() == 8
        # 6 faces * 2 triangles each = 12 triangles (quads are triangulated)
        assert surface.n_triangles() == 12

    def test_cuboid_off_volume(self):
        """Read cuboid.off and verify volume."""
        filepath = os.path.join(TEST_INPUTS, "cuboid.off")
        surface = parse_off_file(filepath)
        surface.create_vertex_polygon_lookup_table()
        surface.create_polygon_polygon_lookup_table()
        w000 = calculate_w000(surface)
        # Cuboid volume should be positive
        assert w000[LABEL_UNASSIGNED].result > 0

    def test_cuboid_with_labels(self):
        """Read cuboid with labels from alpha channel."""
        filepath = os.path.join(TEST_INPUTS, "cuboid-labels.off")
        if not os.path.exists(filepath):
            pytest.skip("cuboid-labels.off not found")
        surface = parse_off_file(filepath, with_labels=True)
        assert surface.n_vertices() >= 8

    def test_cuboid_with_comments(self):
        """Read cuboid with comments in .off file."""
        filepath = os.path.join(TEST_INPUTS, "cuboid-labelsWithComments.off")
        if not os.path.exists(filepath):
            pytest.skip("cuboid-labelsWithComments.off not found")
        surface = parse_off_file(filepath)
        assert surface.n_vertices() >= 8

    def test_is_off_file(self):
        assert is_off_file("test.off")
        assert is_off_file("test.OFF")
        assert not is_off_file("test.poly")
        assert not is_off_file("test.txt")


class TestObjReader:
    """Tests for .obj file parsing."""

    def test_box_obj(self):
        """Read box.obj and verify vertex/triangle counts."""
        filepath = os.path.join(TEST_INPUTS, "box.obj")
        surface = parse_obj_file(filepath)
        assert surface.n_vertices() == 8
        assert surface.n_triangles() == 12

    def test_box_obj_volume(self):
        """Read box.obj and verify volume = 24."""
        filepath = os.path.join(TEST_INPUTS, "box.obj")
        surface = parse_obj_file(filepath)
        surface.create_vertex_polygon_lookup_table()
        surface.create_polygon_polygon_lookup_table()
        w000 = calculate_w000(surface)
        assert w000[LABEL_UNASSIGNED].result == pytest.approx(24.0, rel=1e-4)

    def test_is_obj_file(self):
        assert is_obj_file("test.obj")
        assert is_obj_file("test.OBJ")
        assert not is_obj_file("test.poly")
        assert not is_obj_file("test.off")


class TestGlbReader:
    """Tests for .glb file parsing."""

    @pytest.fixture
    def box_glb(self, tmp_path):
        """Create a GLB fixture of the same box via trimesh."""
        trimesh = pytest.importorskip("trimesh")
        vertices = np.array([
            [ 1.0, -1.5,  2.0],
            [-1.0, -1.5,  2.0],
            [ 1.0,  1.5,  2.0],
            [-1.0,  1.5,  2.0],
            [-1.0, -1.5, -2.0],
            [ 1.0, -1.5, -2.0],
            [-1.0,  1.5, -2.0],
            [ 1.0,  1.5, -2.0],
        ])
        faces = np.array([
            [3, 1, 0], [2, 3, 0], [7, 5, 4], [6, 7, 4],
            [2, 7, 6], [3, 2, 6], [1, 4, 5], [0, 1, 5],
            [2, 0, 5], [7, 2, 5], [6, 4, 1], [3, 6, 1],
        ])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        filepath = str(tmp_path / "box.glb")
        mesh.export(filepath)
        return filepath

    def test_box_glb(self, box_glb):
        """Read GLB box and verify vertex/triangle counts."""
        surface = parse_glb_file(box_glb)
        assert surface.n_vertices() == 8
        assert surface.n_triangles() == 12

    def test_box_glb_volume(self, box_glb):
        """Read GLB box and verify volume = 24."""
        surface = parse_glb_file(box_glb)
        surface.create_vertex_polygon_lookup_table()
        surface.create_polygon_polygon_lookup_table()
        w000 = calculate_w000(surface)
        assert w000[LABEL_UNASSIGNED].result == pytest.approx(24.0, rel=1e-4)

    def test_is_glb_file(self):
        assert is_glb_file("test.glb")
        assert is_glb_file("test.gltf")
        assert is_glb_file("test.GLB")
        assert not is_glb_file("test.obj")
        assert not is_glb_file("test.off")
