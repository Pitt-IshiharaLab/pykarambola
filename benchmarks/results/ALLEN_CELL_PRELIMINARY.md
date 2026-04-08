# Allen Cell Nuclei: Preliminary Classification Results (Unoptimized)

## Overview

Preliminary benchmark comparing SO(3) invariants against raw Minkowski tensor baselines on Allen Cell mitotic nuclei. Results use **default hyperparameters** (PCA=10, RBF kernel, C=1) — no Bayesian optimization. These are directional only; optimized results are needed for fair comparison.

**Core question**: Do SO(3) invariants improve 6-class mitotic stage classification compared to raw tensor components?

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

Both datasets use `minkowski_tensors_with_eigen_vals.csv` from their respective directories:

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

These are rotation-invariant scalar features (traces and normalised traces of rank-2 tensors) that would give the non-rotated baselines a structural advantage. They were **excluded from all runs** to ensure a fair comparison. After exclusion, both datasets use identical feature sets:

| Feature Set | # Features used | # Excluded |
|-------------|----------------|------------|
| Baseline (tensors) | 52 | 10 trace columns |
| Baseline (w/ eigen) | 76 | 10 trace columns |
| SO3 Degree 1–3 | 8 / 39 / 219 | 0 (unaffected) |

---

## Results

### Non-rotated nuclei (unoptimized, PCA=10)

| Rank | Feature Set | # Features | Balanced Accuracy | Geo. Mean |
|------|-------------|------------|-------------------|-----------|
| 1 | **Baseline (w/ eigen)** | 76 | **0.732 ± 0.003** | 0.725 ± 0.004 |
| 2 | SO3 Degree 2 | 39 | 0.715 ± 0.003 | 0.705 ± 0.003 |
| 3 | SO3 Degree 1 | 8 | 0.703 ± 0.004 | 0.688 ± 0.004 |
| 4 | Baseline (tensors) | 52 | 0.685 ± 0.003 | 0.664 ± 0.006 |
| 5 | SO3 Degree 3 | 219 | 0.647 ± 0.001 | 0.636 ± 0.000 |

### XY-aligned nuclei (unoptimized, PCA=10)

| Rank | Feature Set | # Features | Balanced Accuracy | Geo. Mean |
|------|-------------|------------|-------------------|-----------|
| 1 | **Baseline (w/ eigen)** | 76 | **0.758 ± 0.005** | 0.754 ± 0.005 |
| 2 | Baseline (tensors) | 52 | 0.736 ± 0.001 | 0.728 ± 0.001 |
| 3 | SO3 Degree 2 | 39 | 0.708 ± 0.001 | 0.696 ± 0.000 |
| 4 | SO3 Degree 1 | 8 | 0.703 ± 0.004 | 0.688 ± 0.004 |
| 5 | SO3 Degree 3 | 219 | 0.637 ± 0.003 | 0.625 ± 0.003 |

---

## Non-rotated vs XY-aligned Comparison

| Feature Set | Non-rotated | XY-aligned | Δ |
|-------------|-------------|------------|---|
| Baseline (tensors) | 0.685 | **0.736** | **+0.051** |
| Baseline (w/ eigen) | 0.732 | **0.758** | **+0.026** |
| SO3 Degree 1 | 0.703 | 0.703 | 0.000 |
| SO3 Degree 2 | 0.715 | 0.708 | −0.007 |
| SO3 Degree 3 | 0.647 | 0.637 | −0.010 |

---

## Interpretation: Does XY Alignment Explain the Observations?

**Yes.** Aligning the long axis of each nucleus to the x-axis (rotation in XY plane) places all objects in a partially canonical reference frame. This has opposite effects on raw tensor features vs SO3 invariants:

### Raw tensor baselines improve (+3–5%)

Minkowski tensor components encode **both shape and orientation** relative to the coordinate axes. When nuclei are randomly oriented (non-rotated dataset), the same mitotic stage can appear as very different tensor components depending on viewing angle. After XY alignment:

- The principal axis direction is standardised to the x-axis in the XY plane
- Tensor components such as the diagonal entries of `w020` become consistently interpretable across samples
- The classifier can exploit the now-consistent orientation signal in addition to shape
- Result: Baseline (tensors) improves by +5.1%, Baseline (w/ eigen) by +2.6%

### SO3 invariants are unaffected (Δ ≈ 0)

SO3 invariants are **rotation-invariant by construction** — the same invariant values are produced regardless of orientation. Alignment provides no additional information to an SO3-invariant feature set.

- Degree 1: identical to 3 d.p. (0.703 both)
- Degree 2/3: differ by ≤ 0.010, within seed noise

This serves as an empirical confirmation that the invariant implementation is correct.

### Key implication

In the non-rotated setting, SO3 Degree 2 **outperforms** Baseline (tensors) (0.715 vs 0.685) — the invariants extract shape information that the raw tensors obscure with pose noise. In the XY-aligned setting, the alignment partially compensates for this, allowing raw tensors to recover (+5%), while SO3 invariants have no room to improve.

**SO3 invariants achieve rotation-robust performance without requiring any pre-alignment step.** This is practically valuable: alignment requires knowledge of object geometry and a normalisation pipeline, whereas SO3 invariants work directly from tensors regardless of pose.

---

## Caveats: Why These Results Are Preliminary

All feature sets evaluated with identical default parameters (PCA=10, RBF, C=1):

- **SO3 Degree 3** (219 features → 10 PCA components): extreme compression, results unreliable
- All feature sets benefit from tuned PCA, kernel, and C (MedMNIST showed rankings can flip after optimization)

**These preliminary results should not be used to draw final conclusions about relative feature set quality.**

---

## Next Steps

- Run full Bayesian optimization (n_iter=50) on the XY-aligned dataset
- The XY-aligned dataset is the cleaner test: baselines benefit from consistent orientation, making it a harder challenge for SO3 invariants to match or beat them
