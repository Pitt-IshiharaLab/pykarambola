# Allen Cell Nuclei: SO(2) Invariants — Optimized Results

## Overview

Bayesian-optimized classification results for SO(2) Degree 1 and 2, directly comparable to `ALLEN_CELL_OPTIMIZED.md`.
SO(2) feature sets were run in isolation using `--include`; all other conditions are identical.

**Classifier**: LinearSVC (liblinear)
**Dataset**: Non-rotated nuclei only (`nuclei/`)
**Optimization**: BayesSearchCV, n_iter=20, 5-fold stratified CV
**Evaluation**: 3 seeds, balanced accuracy + geometric mean

---

## SO(2) New Results

| Feature Set | # Features | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|-------------|------------|-------------------|-----------|--------|----------|
| SO2 Degree 1 | 18 | 0.674 ± 0.006 | 0.649 ± 0.009 | 21.8 | 16 |
| SO2 Degree 2 | 94 | **0.757 ± 0.008** | **0.751 ± 0.009** | 1000 | 94 |

---

## Combined Ranking (SO(2) + existing ALLEN_CELL_OPTIMIZED.md)

| Rank | Feature Set | # Features | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|------|-------------|------------|-------------------|-----------|--------|----------|
| 1 | Baseline (w/ eigen) | 86 | 0.818 ± 0.004 | 0.815 ± 0.004 | 1.08 | 84 |
| 2 | SO3 Degree 2 | 39 | 0.783 ± 0.002 | 0.778 ± 0.002 | 225 | 39 |
| 3 | **SO2 Degree 2** | **94** | **0.757 ± 0.008** | **0.751 ± 0.009** | **1000** | **94** |
| 4 | Baseline (tensors) | 62 | 0.746 ± 0.006 | 0.737 ± 0.006 | 1000 | 54 |
| 5 | SPHARM Inv lmax=5 | 75 | 0.726 ± 0.004 | 0.713 ± 0.006 | 739 | 57 |
| 6 | **SO2 Degree 1** | **18** | **0.674 ± 0.006** | **0.649 ± 0.009** | **21.8** | **16** |
| 7 | SO3 Degree 1 | 8 | 0.667 ± 0.004 | 0.636 ± 0.005 | 995 | 8 |
| 8 | SPHARM lmax=5 | 72 | 0.597 ± 0.003 | 0.584 ± 0.004 | 0.39 | 57 |

---

## Interpretation

### SO2 Degree 2 is the standout result

At 0.757, SO2 Degree 2 ranks 3rd overall — above Baseline (tensors) (0.746) and SPHARM Inv lmax=5 (0.726), and only 2.6 pp below SO3 Degree 2 (0.783).
The optimizer selected **full PCA (94/94, 100% retained)** and **maximum C (1000)**, the same hard-margin pattern as Baseline (tensors) (C=1000) and SPHARM Inv lmax=5 (C=739).
This confirms that SO2 Degree 2 features are well-conditioned and benefit from retaining all dimensions.

### SO2 Degree 2 gains massively from optimization (+8.6 pp vs unoptimized RBF)

Compared to the unoptimized RBF run (0.671), optimization adds +8.6 pp — the largest absolute gain across any feature set in this benchmark. The default PCA=10 discarded 89% of the 94 features; with full PCA, all degree-2 z-axis invariants contribute.

### SO2 Degree 1 improves modestly (+0.7 pp over SO3 Degree 1)

With C=21.8 and PCA=16/18 (89% retained), SO2 Degree 1 (0.674) slightly outperforms SO3 Degree 1 (0.667).
The z-axis scalars (v_z, M_zz) provide a small but consistent advantage over SO(3) degree-1 traces alone on non-rotated nuclei, consistent with the preliminary finding.

### C contrast within SO(2) degrees

SO2 Degree 1 (C=21.8) vs SO2 Degree 2 (C=1000) — a 46× increase going from degree 1 to degree 2.
Degree-2 inner products of doublets form a more discriminative but higher-dimensional space that benefits from a hard margin, analogous to the behaviour seen in SPHARM Inv lmax=5 (C=739) vs SPHARM lmax=5 (C=0.39).

---

## Runtime

- Total: ~33 min (n_iter=20, 2 feature sets, 5-fold CV, 3 seeds, LinearSVC, n_jobs=5, Apple Silicon)

---

## Configuration

```bash
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --output benchmarks/results/allen_cell_nuclei_so2_optimized \
    --max-so2-degree 2 \
    --include "SO2 Degree 1" "SO2 Degree 2" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5
```
