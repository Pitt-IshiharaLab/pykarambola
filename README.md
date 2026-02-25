# pykarambola

Python implementation of [karambola](https://github.com/morphometry/karambola) — a package for computing **Minkowski functionals and tensors** on 3D triangulated surfaces.

Minkowski functionals (volume, surface area, integrated mean curvature, Euler characteristic) and their tensor generalizations characterize the geometry and morphology of 3D shapes. They are widely used in structural analysis, materials science, and computational physics.

## Installation

```bash
pip install pykarambola
```

For optional Cython acceleration:

```bash
pip install "pykarambola[accel]"
```

For development (includes pytest and scikit-image):

```bash
pip install "pykarambola[dev]"
```

## Quick start

### From a mesh file

```python
import pykarambola as pk

surface = pk.parse_poly_file("my_surface.poly")
# also: parse_off_file, parse_obj_file, parse_glb_file

result = pk.minkowski_functionals(
    surface.vertices,   # (V, 3) array
    surface.triangles,  # (F, 3) array of vertex indices
)

print(result["w000"])   # volume
print(result["w100"])   # surface area
print(result["w200"])   # integrated mean curvature
print(result["w300"])   # Euler characteristic
```

### From a 3D label image (requires scikit-image)

```python
import numpy as np
import pykarambola as pk

label_image = np.zeros((64, 64, 64), dtype=int)
label_image[16:48, 16:48, 16:48] = 1   # a cube

result = pk.minkowski_functionals_from_label_image(label_image)
print(result[1]["w000"])   # volume of label 1
```

## Computed quantities

| Name | Type | Description |
|------|------|-------------|
| `w000` | scalar | Volume |
| `w100` | scalar | Surface area |
| `w200` | scalar | Integrated mean curvature |
| `w300` | scalar | Euler characteristic |
| `w010` | vector | Minkowski vector (volume) |
| `w110` | vector | Minkowski vector (surface) |
| `w210` | vector | Minkowski vector (curvature) |
| `w310` | vector | Minkowski vector (topology) |
| `w020` | rank-2 tensor | Minkowski tensor (volume) |
| `w120` | rank-2 tensor | Minkowski tensor (surface) |
| `w220` | rank-2 tensor | Minkowski tensor (curvature) |
| `w320` | rank-2 tensor | Minkowski tensor (topology) |
| `w102` | rank-2 tensor | Minkowski tensor (surface, normal-normal) |
| `w202` | rank-2 tensor | Minkowski tensor (curvature, normal-normal) |
| `w103` | rank-3 tensor | Higher-order tensor |
| `w104` | rank-4 tensor | Higher-order tensor |
| `msm_ql`, `msm_wl` | arrays | Minkowski structure metrics (spherical) |

Rank-2 tensors additionally yield `{name}_eigvals` and `{name}_eigvecs` entries.

By default, `compute='standard'` computes the 14 base functionals plus eigensystems. Use `compute='all'` to include `w103`, `w104`, and `msm`.

## Multi-label meshes

Pass per-face integer labels to analyse multiple bodies in one mesh:

```python
result = pk.minkowski_functionals(verts, faces, labels=face_labels)
# result is a dict keyed by label value
print(result[1]["w000"])
print(result[2]["w000"])
```

## Command-line interface

```
python -m pykarambola [options] <surface_file>
```

Supported input formats: `.poly`, `.off`, `.obj`, `.glb`.

## File formats

| Extension | Description |
|-----------|-------------|
| `.poly`   | karambola native format |
| `.off`    | Object File Format |
| `.obj`    | Wavefront OBJ |
| `.glb`    | GL Transmission Format (binary) |

## Citation

If you use pykarambola in published work, please cite the original karambola package:

> Schaller, F. M., Kapfer, S. C., & Schröder-Turk, G. E.
> *karambola — 3D Minkowski Tensor Package* (v2.0).
> https://github.com/morphometry/karambola

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

## License

See [`LICENSE`](LICENSE).
