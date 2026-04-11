# Allen Cell Nuclei: Eigenvalues Only — Optimized Results

## Overview

Bayesian-optimized classification result for the 18 eigenvalue columns alone
(3 eigenvalues × 6 rank-2 tensors), directly comparable to all previous optimized runs.

**Classifier**: LinearSVC (liblinear)
**Dataset**: Non-rotated nuclei only (`nuclei/`)
**Optimization**: BayesSearchCV, n_iter=20, 5-fold stratified CV
**Evaluation**: 3 seeds, balanced accuracy + geometric mean
**Features**: `w020_EVal1/2/3`, `w102_EVal1/2/3`, `w120_EVal1/2/3`,
             `w202_EVal1/2/3`, `w220_EVal1/2/3`, `w320_EVal1/2/3`

---

## Result

| Feature Set | # Features | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|-------------|------------|-------------------|-----------|--------|----------|
| Eigenvalues only | 18 | **0.791 ± 0.003** | **0.784 ± 0.005** | 37.2 | 16 |

---

## Full Combined Ranking

| Rank | Feature Set | # Features | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|------|-------------|------------|-------------------|-----------|--------|----------|
| 1 | Minkowski (tensors+eigen+beta) | 86 | 0.818 ± 0.004 | 0.815 ± 0.004 | 1.08 | 84 |
| 1 | SO3 Degree 2 + Eigenvalues | 57 | 0.817 ± 0.003 | 0.814 ± 0.003 | 980 | 53 |
| 3 | SO2 Degree 1 + Eigenvalues | 36 | 0.799 ± 0.005 | 0.795 ± 0.006 | 13.3 | 25 |
| 4 | SO3 Degree 3 | 219 | 0.795 ± 0.004 | 0.786 ± 0.005 | 14.9 | 195 |
| 5 | SO3 Degree 1 + Eigenvalues | 26 | 0.793 ± 0.006 | 0.789 ± 0.007 | 1000 | 26 |
| 5 | **Eigenvalues only** | **18** | **0.791 ± 0.003** | **0.784 ± 0.005** | **37.2** | **16** |
| 7 | SO2 Degree 2 + Eigenvalues | 112 | 0.787 ± 0.001 | 0.781 ± 0.001 | 225 | 111 |
| 8 | SO3 Degree 2 + SO2 z-scalars | 49 | 0.784 ± 0.007 | 0.778 ± 0.008 | 225 | 49 |
| 8 | SO3 Degree 2 | 39 | 0.783 ± 0.002 | 0.778 ± 0.002 | 225 | 39 |
| 10 | CellProfiler | 22 | 0.769 ± 0.003 | 0.761 ± 0.003 | 225 | 22 |
| 11 | SO2 Degree 2 | 94 | 0.757 ± 0.008 | 0.751 ± 0.009 | 1000 | 94 |
| 12 | Minkowski (tensors) | 62 | 0.746 ± 0.006 | 0.737 ± 0.006 | 1000 | 54 |
| 13 | SPHARM Inv lmax=5 | 75 | 0.726 ± 0.004 | 0.713 ± 0.006 | 739 | 57 |
| 14 | SO2 Degree 1 | 18 | 0.674 ± 0.006 | 0.649 ± 0.009 | 21.8 | 16 |
| 15 | SO3 Degree 1 | 8 | 0.667 ± 0.004 | 0.636 ± 0.005 | 995 | 8 |
| 16 | SPHARM lmax=5 | 72 | 0.597 ± 0.003 | 0.584 ± 0.004 | 0.39 | 57 |

---

## Interpretation

### 18 eigenvalues match 219 polynomial invariants

Eigenvalues only (0.791) is statistically tied with SO3 Degree 3 (0.795 ± 0.004) —
the entire benefit of expanding SO3 invariants to degree 3 (8 → 39 → 219 features,
with exponentially growing computation) is reproduced by 18 numbers read directly
from the CSV.

This is not a coincidence: SO3 Degree 3 invariants recover det(W) = I₃ = λ₁λ₂λ₃
via Tr(W³), exactly the information that eigenvalues provide directly. The 180 extra
cubic cross-tensor terms in SO3 Degree 3 beyond what degree 2 already has add only
~1.2 pp over SO3 Degree 2 — the same incremental gain that eigenvalues provide
when appended to any degree-1 base (see ALLEN_CELL_EIGEN_COMBINATIONS_OPTIMIZED.md).

### Eigenvalues outperform SO3 Degree 2 with half the features

0.791 (18 features) vs 0.783 (39 features) — eigenvalues alone beat the full set of
SO3 Degree 2 polynomial invariants. SO3 Degree 2 captures I₁ and I₂ per tensor plus
cross-tensor quadratic products Tr(WᵢWⱼ), but misses I₃ = det(W). Eigenvalues capture
the complete per-tensor spectrum {I₁, I₂, I₃} for all 6 rank-2 tensors but miss
cross-tensor correlations. The per-tensor completeness wins over the cross-tensor
correlations for this task.

### SO3 Degree 1 + Eigenvalues adds nothing over Eigenvalues alone

0.793 ± 0.006 vs 0.791 ± 0.003 — statistically identical. The 8 degree-1 traces
(Tr(W) = I₁ per tensor) are already encoded in the eigenvalue sum λ₁ + λ₂ + λ₃,
so they add no new information. This confirms that the classifier extracts I₁ from
the eigenvalue columns without needing it provided separately.

### Eigenvalues are the primary information-bearing features in this benchmark

The performance ladder from 0.667 (SO3 Degree 1) to 0.817 (SO3 Degree 2 + Eigenvalues)
can be decomposed as:

| Step | Gain | What is added |
|------|------|---------------|
| SO3 Degree 1 → Eigenvalues only | +12.4 pp | I₂ + I₃ per tensor (direct) |
| Eigenvalues only → SO3 Degree 2 + Eigenvalues | +2.6 pp | Cross-tensor quadratic Tr(WᵢWⱼ) |

Cross-tensor correlations contribute only ~2.6 pp on top of the per-tensor eigenvalue
spectra. Shape, as measured by this mitotic stage classification task, is almost entirely
captured by how each individual Minkowski tensor is shaped — not by how different tensors
relate to each other.

### Runtime

Eigenvalue features are pre-computed in the CSV — feature extraction takes 0s.
Total benchmark runtime: 32 seconds.

---

## Configuration

```bash
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --output benchmarks/results/allen_cell_nuclei_eigen_only_optimized \
    --include "Eigenvalues only" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5
```
