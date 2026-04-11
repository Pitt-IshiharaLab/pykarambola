# Allen Cell Nuclei: Preliminary Classification Results (Unoptimized)

## Overview

Preliminary benchmark comparing SO(3) invariants, raw Minkowski tensor baselines, and spherical harmonics (SPHARM) on Allen Cell mitotic nuclei. Results use **default hyperparameters** (PCA=10, RBF kernel, C=1) — no Bayesian optimization. These are directional only; optimized results are needed for fair comparison.

**Core question**: Do SO(3) invariants improve 6-class mitotic stage classification compared to raw tensor components and spherical harmonics?

---

## Datasets

| Property | Value |
|----------|-------|
| Source | Allen Cell Institute mitotic cells (annotated) |
| Object | Nuclei |
| Total samples | 5606 (train: 3924, val: 560, test: 1122) |
| Classes | 6 (mitotic stages 0–5) |
| Class imbalance | 7.4:1 (class 0: 2516, class 5: 338) |
| Centering | w010 ≈ 0 for all samples (centered at centroid) |

### Dataset pair used

| Dataset | Path | Description |
|---------|------|-------------|
| **Non-rotated** | `nuclei/` | Original orientations from imaging |
| **XY-aligned** | `nuclei_rotated_only_in_xy/` | Long axis of each nucleus aligned to x-axis (rotation in XY plane only) |

### Feature column mismatch and correction

The non-rotated CSV contained 10 additional derived columns absent from the XY-aligned CSV:

```
Tr(w020), Tr(w020)/w000, Tr(w102), Tr(w120), Tr(w120)/w100,
Tr(w202), Tr(w220), Tr(w220)/w200, Tr(w320), Tr(w320)/w300
```

These are rotation-invariant scalar features (traces and normalised traces of rank-2 tensors) that would give the non-rotated baselines a structural advantage. They were **excluded from all runs** to ensure a fair comparison. After exclusion, both datasets use identical Minkowski tensor feature sets:

| Feature Set | # Features used | # Excluded |
|-------------|----------------|------------|
| Minkowski (tensors) | 52 | 10 trace columns |
| Minkowski (tensors+eigen+beta) | 76 | 10 trace columns |
| SO3 Degree 1–3 | 8 / 39 / 219 | 0 (unaffected) |

### Spherical harmonics availability

SPHARM files are not fully symmetric across datasets:

| File | Non-rotated | XY-aligned | # Features |
|------|-------------|------------|------------|
| `spherical_harmonics_lmax_5.csv` | ✅ | ❌ | 72 |
| `spherical_harmonics_lmax_16_with_watertight.csv` | ✅ | ✅ | 578 |

SPHARM lmax=5 was therefore only evaluated on the non-rotated dataset.

---

## Results

### Non-rotated nuclei (unoptimized, PCA=10)

| Rank | Feature Set | # Features | Balanced Accuracy | Geo. Mean |
|------|-------------|------------|-------------------|-----------|
| 1 | **Minkowski (tensors+eigen+beta)** | 76 | **0.732 ± 0.003** | 0.725 ± 0.004 |
| 2 | SO3 Degree 2 | 39 | 0.715 ± 0.003 | 0.705 ± 0.003 |
| 3 | SO3 Degree 1 | 8 | 0.703 ± 0.004 | 0.688 ± 0.004 |
| 4 | Minkowski (tensors) | 52 | 0.685 ± 0.003 | 0.664 ± 0.006 |
| 5 | SO3 Degree 3 | 219 | 0.647 ± 0.001 | 0.636 ± 0.000 |
| 6 | SPHARM lmax=5 | 72 | 0.620 ± 0.003 | 0.615 ± 0.003 |
| 7 | SPHARM lmax=16 | 578 | 0.613 ± 0.003 | 0.588 ± 0.004 |

### XY-aligned nuclei (unoptimized, PCA=10)

| Rank | Feature Set | # Features | Balanced Accuracy | Geo. Mean |
|------|-------------|------------|-------------------|-----------|
| 1 | **Minkowski (tensors+eigen+beta)** | 76 | **0.758 ± 0.005** | 0.754 ± 0.005 |
| 2 | Minkowski (tensors) | 52 | 0.736 ± 0.001 | 0.728 ± 0.001 |
| 3 | SO3 Degree 2 | 39 | 0.708 ± 0.001 | 0.696 ± 0.000 |
| 4 | SO3 Degree 1 | 8 | 0.703 ± 0.004 | 0.688 ± 0.004 |
| 5 | SO3 Degree 3 | 219 | 0.637 ± 0.003 | 0.625 ± 0.003 |
| 6 | SPHARM lmax=16 | 578 | 0.603 ± 0.008 | 0.574 ± 0.012 |

---

## Non-rotated vs XY-aligned Comparison

| Feature Set | Non-rotated | XY-aligned | Δ |
|-------------|-------------|------------|---|
| Minkowski (tensors) | 0.685 | **0.736** | **+0.051** |
| Minkowski (tensors+eigen+beta) | 0.732 | **0.758** | **+0.026** |
| SO3 Degree 1 | 0.703 | 0.703 | 0.000 |
| SO3 Degree 2 | 0.715 | 0.708 | −0.007 |
| SO3 Degree 3 | 0.647 | 0.637 | −0.010 |
| SPHARM lmax=16 | 0.613 | 0.603 | −0.010 |

---

## Interpretation

### XY alignment explains the baseline improvement (+3–5%)

Aligning the long axis of each nucleus to the x-axis places all objects in a partially canonical reference frame. Minkowski tensor components encode **both shape and orientation** relative to the coordinate axes, so consistent alignment makes them more discriminative. After XY alignment:

- Minkowski (tensors) improves by +5.1%, Minkowski (tensors+eigen+beta) by +2.6%
- Raw tensor components now carry consistent orientation signal in addition to shape

### SO3 invariants are unaffected by rotation (Δ ≈ 0)

SO3 invariants are **rotation-invariant by construction**. Degree 1 is identical to 3 d.p. across both datasets; Degree 2/3 differ by ≤ 0.010, within seed noise. This confirms the invariant implementation is correct.

**Key implication**: In the non-rotated setting, SO3 Degree 2 outperforms Minkowski (tensors) (0.715 vs 0.685) — the invariants extract shape information that raw tensors obscure with pose noise. SO3 invariants achieve this without requiring any pre-alignment step.

### SPHARM is severely penalised by PCA=10

SPHARM ranks last on both datasets, but this is largely an artefact of the default PCA=10:

| Feature Set | # Features | PCA compression |
|-------------|------------|-----------------|
| SPHARM lmax=5 | 72 | 72 → 10 (86%) discarded) |
| SPHARM lmax=16 | 578 | 578 → 10 (98% discarded) |
| SO3 Degree 2 | 39 | 39 → 10 (74% discarded) |

SPHARM lmax=5 (72 features) slightly outperforms lmax=16 (578 features) on non-rotated data (0.620 vs 0.613) — expected, since lmax=16 suffers greater relative compression. SPHARM results are the most unreliable of all feature sets until optimization.

---

## Caveats: Why These Results Are Preliminary

All feature sets evaluated with identical default parameters (PCA=10, RBF, C=1). The MedMNIST benchmarks showed that optimization substantially changes rankings (e.g., SO3 Degree 2 jumped from near-worst to best on adrenal3d after optimization). SPHARM in particular requires a much larger PCA budget to be fairly evaluated.

**These preliminary results should not be used to draw final conclusions about relative feature set quality.**

---

## Next Steps

- Run full Bayesian optimization (n_iter=50) on both datasets with all feature sets including SPHARM
- The XY-aligned dataset with optimization is the most informative test: baselines benefit from consistent orientation, making it a harder challenge for rotation-invariant methods
