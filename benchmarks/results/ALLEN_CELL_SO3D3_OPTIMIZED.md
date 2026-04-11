# Allen Cell Nuclei: SO3 Degree 3 — Optimized Results

## Overview

Bayesian-optimized classification result for SO3 Degree 3 invariants,
directly comparable to all previous optimized runs.

**Classifier**: LinearSVC (liblinear)
**Dataset**: Non-rotated nuclei only (`nuclei/`)
**Optimization**: BayesSearchCV, n_iter=20, 5-fold stratified CV
**Evaluation**: 3 seeds, balanced accuracy + geometric mean

---

## Result

| Feature Set | # Features | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|-------------|------------|-------------------|-----------|--------|----------|
| SO3 Degree 3 | 219 | 0.795 ± 0.004 | 0.786 ± 0.005 | 14.9 | 195 |

---

## Full Combined Ranking

| Rank | Feature Set | # Features | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|------|-------------|------------|-------------------|-----------|--------|----------|
| 1 | Baseline (w/ eigen) | 86 | 0.818 ± 0.004 | 0.815 ± 0.004 | 1.08 | 84 |
| 1 | SO3 Degree 2 + Eigenvalues | 57 | 0.817 ± 0.003 | 0.814 ± 0.003 | 980 | 53 |
| 3 | SO2 Degree 1 + Eigenvalues | 36 | 0.799 ± 0.005 | 0.795 ± 0.006 | 13.3 | 25 |
| 4 | **SO3 Degree 3** | **219** | **0.795 ± 0.004** | **0.786 ± 0.005** | **14.9** | **195** |
| 5 | SO3 Degree 1 + Eigenvalues | 26 | 0.793 ± 0.006 | 0.789 ± 0.007 | 1000 | 26 |
| 6 | SO2 Degree 2 + Eigenvalues | 112 | 0.787 ± 0.001 | 0.781 ± 0.001 | 225 | 111 |
| 7 | SO3 Degree 2 + SO2 z-scalars | 49 | 0.784 ± 0.007 | 0.778 ± 0.008 | 225 | 49 |
| 7 | SO3 Degree 2 | 39 | 0.783 ± 0.002 | 0.778 ± 0.002 | 225 | 39 |
| 9 | CellProfiler | 22 | 0.769 ± 0.003 | 0.761 ± 0.003 | 225 | 22 |
| 10 | SO2 Degree 2 | 94 | 0.757 ± 0.008 | 0.751 ± 0.009 | 1000 | 94 |
| 11 | Baseline (tensors) | 62 | 0.746 ± 0.006 | 0.737 ± 0.006 | 1000 | 54 |
| 12 | SPHARM Inv lmax=5 | 75 | 0.726 ± 0.004 | 0.713 ± 0.006 | 739 | 57 |
| 13 | SO2 Degree 1 | 18 | 0.674 ± 0.006 | 0.649 ± 0.009 | 21.8 | 16 |
| 14 | SO3 Degree 1 | 8 | 0.667 ± 0.004 | 0.636 ± 0.005 | 995 | 8 |
| 15 | SPHARM lmax=5 | 72 | 0.597 ± 0.003 | 0.584 ± 0.004 | 0.39 | 57 |

---

## Interpretation

### SO3 Degree 3 improves over SO3 Degree 2 but not efficiently

SO3 Degree 3 (0.795) gains +1.2 pp over SO3 Degree 2 (0.783), confirming that the
cross-tensor cubic invariants (Tr(WᵢWⱼWₖ)) and per-tensor det(W) = I₃ terms do carry
additional discriminative signal.
However, the gain is modest relative to the 180-feature increase (39 → 219), and comes
at a 34-minute optimization cost vs ~3 min for SO3 Degree 2.

### Eigenvalues are a more efficient route to the same signal

SO3 Degree 2 + Eigenvalues (57 features, 0.817) gains +3.4 pp over SO3 Degree 2 alone —
nearly 3× the gain of SO3 Degree 3 (+1.2 pp) at one quarter of the features (57 vs 219).
Both routes give the classifier access to det(W) per tensor (I₃), but eigenvalues deliver
it directly and non-redundantly, while SO3 Degree 3 buries it among 180 additional cubic
cross-tensor terms, many of which are partially redundant.

### Low C signals collinearity among cubic invariants

C=14.9 for SO3 Degree 3 vs C=225 for SO3 Degree 2 — a 15× drop in the regularisation
scale, despite better performance.
The 219 cubic polynomial invariants include many near-dependent combinations (e.g.,
Tr(W₁W₂W₃) is partially determined by lower-degree products), forcing the optimizer to
apply substantial regularisation to suppress the redundant directions.
PCA retaining 195/219 (89%) rather than 100% also reflects this: ~24 directions were
identified as uninformative, but the remaining 195 still required strong regularisation.

### SO3 Degree 3 is statistically tied with SO2 Degree 1 + Eigenvalues

0.795 ± 0.004 vs 0.799 ± 0.005 — overlapping error bars, indistinguishable within noise.
SO2 Degree 1 + Eigenvalues achieves the same classification power with 36 features (vs 219)
and ~1 min optimization time (vs 34 min), using a principled compact descriptor rather than
an exhaustive polynomial expansion.

---

## Runtime

- Total: ~35 min (n_iter=20, 1 feature set, 5-fold CV, 3 seeds, LinearSVC, n_jobs=5, Apple Silicon)

---

## Configuration

```bash
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --output benchmarks/results/allen_cell_nuclei_so3d3_optimized \
    --include "SO3 Degree 3" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5
```
