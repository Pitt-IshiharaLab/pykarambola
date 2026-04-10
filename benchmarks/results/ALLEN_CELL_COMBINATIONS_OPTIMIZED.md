# Allen Cell Nuclei: Combination Feature Sets — Optimized Results

## Overview

Bayesian-optimized results for two combination feature sets, directly comparable to `ALLEN_CELL_OPTIMIZED.md` and `ALLEN_CELL_SO2_OPTIMIZED.md`.

**Classifier**: LinearSVC (liblinear)
**Dataset**: Non-rotated nuclei only (`nuclei/`)
**Optimization**: BayesSearchCV, n_iter=20, 5-fold stratified CV
**Evaluation**: 3 seeds, balanced accuracy + geometric mean

---

## New Combination Results

| Feature Set | # Features | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|-------------|------------|-------------------|-----------|--------|----------|
| SO3 Degree 2 + SO2 z-scalars | 49 | 0.784 ± 0.007 | 0.778 ± 0.008 | 225 | 49 |
| SO3 Degree 2 + Eigenvalues | 57 | **0.817 ± 0.003** | **0.814 ± 0.003** | 980 | 53 |

---

## Full Combined Ranking

| Rank | Feature Set | # Features | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|------|-------------|------------|-------------------|-----------|--------|----------|
| 1 | Baseline (w/ eigen) | 86 | 0.818 ± 0.004 | 0.815 ± 0.004 | 1.08 | 84 |
| 1 | **SO3 Degree 2 + Eigenvalues** | **57** | **0.817 ± 0.003** | **0.814 ± 0.003** | **980** | **53** |
| 3 | **SO3 Degree 2 + SO2 z-scalars** | **49** | **0.784 ± 0.007** | **0.778 ± 0.008** | **225** | **49** |
| 3 | SO3 Degree 2 | 39 | 0.783 ± 0.002 | 0.778 ± 0.002 | 225 | 39 |
| 5 | SO2 Degree 2 | 94 | 0.757 ± 0.008 | 0.751 ± 0.009 | 1000 | 94 |
| 6 | Baseline (tensors) | 62 | 0.746 ± 0.006 | 0.737 ± 0.006 | 1000 | 54 |
| 7 | SPHARM Inv lmax=5 | 75 | 0.726 ± 0.004 | 0.713 ± 0.006 | 739 | 57 |
| 8 | SO2 Degree 1 | 18 | 0.674 ± 0.006 | 0.649 ± 0.009 | 21.8 | 16 |
| 9 | SO3 Degree 1 | 8 | 0.667 ± 0.004 | 0.636 ± 0.005 | 995 | 8 |
| 10 | SPHARM lmax=5 | 72 | 0.597 ± 0.003 | 0.584 ± 0.004 | 0.39 | 57 |

---

## Interpretation

### SO3 Degree 2 + Eigenvalues matches Baseline (w/ eigen) with 34% fewer features

At 0.817 vs 0.818, the two are statistically indistinguishable (overlapping error bars).
SO3 Degree 2 + Eigenvalues achieves equivalent performance using 57 features vs 86, while being a more principled feature set: it replaces orientation-dependent raw tensor components with rotation-invariant degree-2 polynomial invariants, retaining only the eigenvalues as the shape-encoding complement.

The C contrast confirms this interpretation:
- **Baseline (w/ eigen): C=1.08** — strong regularisation forced by collinearity between raw tensor components and their derived eigenvalues
- **SO3 Degree 2 + Eigenvalues: C=980** — hard margin optimal because SO3 invariants and eigenvalues are non-collinear; each dimension contributes independently

Same performance, opposite regularisation regimes — the underlying reason is that replacing raw tensors with their rotation invariants eliminates the source of collinearity.

### SO3 Degree 2 + SO2 z-scalars adds nothing over SO3 Degree 2 alone

0.784 vs 0.783 — identical within noise (same C=225, same 100% PCA retention).
The 10 extra z-axis scalars (v_z components and M_zz components) provide no additional discriminative signal beyond what SO3 Degree 2 already captures at the degree-2 level.
This contrasts with the preliminary RBF finding where SO2 Degree 1 beat SO3 Degree 1 — the degree-2 invariants appear to already encode sufficient z-axis shape information.

### The eigenvalues are the key ingredient in Baseline (w/ eigen)

Comparing Baseline (tensors) (0.746) → Baseline (w/ eigen) (0.818): adding eigenvalues to raw tensors gains +7.2 pp but introduces collinearity (C drops from 1000 to 1.08).
Comparing SO3 Degree 2 (0.783) → SO3 Degree 2 + Eigenvalues (0.817): adding eigenvalues to rotation invariants gains +3.4 pp with no collinearity penalty (C stays high at 980).

The eigenvalues contribute ~3–7 pp regardless of what they are combined with.
The difference is that the raw tensor baseline needs much stronger regularisation to handle the redundancy — and in doing so, likely discards some useful signal.

---

## Runtime

- Total: ~33 min (n_iter=20, 2 feature sets, 5-fold CV, 3 seeds, LinearSVC, n_jobs=5, Apple Silicon)

---

## Configuration

```bash
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --output benchmarks/results/allen_cell_nuclei_combinations_optimized \
    --max-so3-degree 2 \
    --include "SO3 Degree 2 + SO2 z-scalars" "SO3 Degree 2 + Eigenvalues" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5
```
