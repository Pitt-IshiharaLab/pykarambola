# Allen Cell Nuclei: Invariants Classification Summary

## Overview

Comprehensive comparison of rotation-invariant polynomial features (SO3/SO2, degree 1–3)
against Minkowski tensor baselines, with and without explicit eigenvalue augmentation.

**Classifier**: LinearSVC (liblinear)
**Dataset**: Non-rotated nuclei only (`nuclei/`)
**Optimization**: BayesSearchCV, n_iter=20, 5-fold stratified CV
**Evaluation**: 3 seeds, balanced accuracy + geometric mean

---

## Results

### Minkowski Baselines

| Feature Set | # Feat | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|-------------|--------|-------------------|-----------|--------|----------|
| Minkowski (tensors) | 62 | 0.746 ± 0.006 | 0.737 ± 0.006 | 1000 | 54 |
| Minkowski (tensors+eigen) | 80 | 0.806 ± 0.007 | 0.802 ± 0.008 | 977 | 66 |
| Eigen + Beta | 24 | **0.808 ± 0.001** | **0.804 ± 0.001** | 225 | 24 |

### SO3 Invariants (full rotation invariance)

| Feature Set | # Feat | Balanced Accuracy | Geo. Mean | Best C | Best PCA | Δ Eigen |
|-------------|--------|-------------------|-----------|--------|----------|---------|
| SO3 Degree 1 | 8 | 0.667 ± 0.004 | 0.636 ± 0.005 | 995 | 8 | — |
| SO3 Degree 1 + Eigenvalues | 26 | 0.793 ± 0.006 | 0.789 ± 0.007 | 1000 | 26 | **+12.6 pp** |
| SO3 Degree 2 | 39 | 0.783 ± 0.002 | 0.778 ± 0.002 | 225 | 39 | — |
| SO3 Degree 2 + Eigenvalues | 57 | **0.817 ± 0.003** | **0.814 ± 0.003** | 980 | 53 | **+3.4 pp** |
| SO3 Degree 3 | 219 | 0.795 ± 0.004 | 0.786 ± 0.005 | 14.9 | 195 | — |
| SO3 Degree 3 + Eigenvalues | 237 | 0.804 ± 0.005 | 0.797 ± 0.005 | 988 | 213 | **+0.9 pp** |

### SO2 Invariants (axial symmetry, z-axis preserved)

| Feature Set | # Feat | Balanced Accuracy | Geo. Mean | Best C | Best PCA | Δ Eigen |
|-------------|--------|-------------------|-----------|--------|----------|---------|
| SO2 Degree 1 | 18 | 0.674 ± 0.006 | 0.649 ± 0.009 | 21.8 | 16 | — |
| SO2 Degree 1 + Eigenvalues | 36 | 0.799 ± 0.005 | 0.795 ± 0.006 | 13.3 | 25 | **+12.5 pp** |
| SO2 Degree 2 | 94 | 0.757 ± 0.008 | 0.751 ± 0.009 | 1000 | 94 | — |
| SO2 Degree 2 + Eigenvalues | 112 | 0.787 ± 0.001 | 0.781 ± 0.001 | 225 | 111 | **+3.0 pp** |
| SO2 Degree 3 | — | not run | — | — | — | — |
| SO2 Degree 3 + Eigenvalues | — | not run | — | — | — | — |

---

## Interpretation

### Eigenvalue gain diminishes with polynomial degree

The most consistent pattern across both symmetry groups is that explicitly adding eigenvalues
(the three sorted eigenvalues of each rank-2 tensor) produces diminishing gains as polynomial
degree increases:

| Degree | SO3 Δ Eigen | SO2 Δ Eigen |
|--------|------------|------------|
| 1 | +12.6 pp | +12.5 pp |
| 2 | +3.4 pp | +3.0 pp |
| 3 | +0.9 pp | (not run) |

**Why**: Each rank-2 tensor W has three fundamental SO(3) invariants — I₁ = Tr(W),
I₂ = (Tr(W)² − Tr(W²))/2, and I₃ = det(W). Degree-1 polynomial invariants capture only I₁;
eigenvalues add I₂ and I₃, hence the large +12.6 pp gain. Degree-2 already recovers I₂
through cross-tensor products, so eigenvalues only add I₃, giving +3.4 pp. Degree-3
encodes all three invariants, leaving eigenvalues to provide only a pre-computed, better-
conditioned representation — +0.9 pp.

### SO3 Degree 2 + Eigenvalues is the optimal polynomial invariant set

At 57 features and 0.817, SO3 Degree 2 + Eigenvalues is the best-performing polynomial
invariant set:
- Better than SO3 Degree 3 + Eigenvalues (237 features, 0.804) — degree-3 adds collinear
  cubic terms that the classifier must regularise away (C=14.9 for SO3D3 vs C=980 for SO3D2+E)
- Better than SO2 Degree 1/2 + Eigenvalues — full SO3 invariance encodes richer cross-tensor
  quadratic products than axial SO2 invariance

The sudden drop from C=225 (SO3 D2) to C=14.9 (SO3 D3) without eigenvalues signals that
the cubic terms are highly collinear; with eigenvalues the classifier recovers to C=988,
confirming the eigenvalue subspace is clean.

### SO3 outperforms SO2 at the same degree

| Degree | SO3 (no eigen) | SO2 (no eigen) | SO3+E | SO2+E |
|--------|---------------|---------------|-------|-------|
| 1 | 0.667 | 0.674 (+0.7) | 0.793 | 0.799 (+0.6) |
| 2 | 0.783 | 0.757 (−2.6) | **0.817** | 0.787 (−3.0) |

At degree 1, SO2 has a slight edge (+0.6–0.7 pp): the z-axis asymmetry features add useful
information about nuclear orientation along the imaging axis.
At degree 2, SO3 dominates (+2.6 pp without eigen, +3.0 pp with eigen): the quadratic
Tr(WᵢWⱼ) cross-tensor products under full SO(3) invariance encode richer shape correlations
than the restricted z-axis-preserving SO2 products.

### Feature count vs performance: diminishing returns above 57 features

| # Feat | Feature Set | Bal. Acc | Δ from prev |
|--------|-------------|----------|-------------|
| 8 | SO3 Degree 1 | 0.667 | — |
| 18 | SO2 Degree 1 | 0.674 | +0.7 |
| 24 | Eigen + Beta | 0.808 | +13.4 |
| 26 | SO3 D1 + Eigen | 0.793 | — |
| 36 | SO2 D1 + Eigen | 0.799 | — |
| 39 | SO3 Degree 2 | 0.783 | — |
| 57 | SO3 D2 + Eigen | **0.817** | peak |
| 80 | Mink (tensors+eigen) | 0.806 | −1.1 vs peak |
| 94 | SO2 Degree 2 | 0.757 | — |
| 112 | SO2 D2 + Eigen | 0.787 | — |
| 219 | SO3 Degree 3 | 0.795 | −2.2 vs peak |
| 237 | SO3 D3 + Eigen | 0.804 | −1.3 vs peak |

Performance peaks at 57 features (SO3 D2+E) and then declines or plateaus as features are
added. All sets above 57 features perform worse despite 4–5× the feature count.

### Where Minkowski baselines sit

- **Minkowski (tensors, 62)** = 0.746: below SO3 D2 (39 features, 0.783) — raw tensor components
  are a worse representation than polynomial invariants for linear classification
- **Minkowski (tensors+eigen, 80)** = 0.806: between SO3 D2+E and SO3 D3+E, but with 23 more
  features than the better-performing SO3 D2+E (57 features)
- **Eigen + Beta (24)** = 0.808: effectively ties Minkowski (tensors+eigen) at 3.3× fewer features,
  showing that the 62 raw tensor components add negligible information once eigenvalues and betas
  are present

---

## Practical Recommendations

| Priority | Feature Set | # Feat | Bal. Acc | When to use |
|----------|-------------|--------|----------|-------------|
| 1st | SO3 Degree 2 + Eigenvalues | 57 | **0.817** | Best overall performance |
| 2nd | Eigen + Beta | 24 | 0.808 | Best efficiency (3.3× fewer features) |
| 3rd | SO2 Degree 1 + Eigenvalues | 36 | 0.799 | Simple, fast, captures z-axis asymmetry |
| 4th | SO3 Degree 1 + Eigenvalues | 26 | 0.793 | Simplest full-SO3 set |
| Avoid | SO3 Degree 3 (+E) | 219–237 | 0.795–0.804 | No gain over D2+E, 4× feature count |
| Avoid | Minkowski (tensors) | 62 | 0.746 | Dominated by any eigen-augmented set |
