# Allen Cell Nuclei: Systematic SPHARM Benchmark — lmax 1–5

## Overview

Bayesian-optimized classification results for raw SPHARM coefficients and
rotation-invariant SPHARM features (power spectrum + bispectrum) at lmax=1–5,
using the lmax=16 CSV truncated at each lmax value.

**Classifier**: LinearSVC (liblinear)
**Dataset**: Non-rotated nuclei only (`nuclei/`)
**Optimization**: BayesSearchCV, n_iter=20, 5-fold stratified CV
**Evaluation**: 3 seeds, balanced accuracy + geometric mean
**Input**: `spherical_harmonics_lmax_16.csv` (filtered to L≤lmax, M≤lmax per run)

---

## Results

### Raw SPHARM (orientation-dependent coefficients)

| lmax | # Features | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|------|------------|-------------------|-----------|--------|----------|
| 1 | 8 | 0.423 ± 0.001 | 0.352 ± 0.009 | 0.10 | 4 |
| 2 | 18 | 0.495 ± 0.006 | 0.460 ± 0.007 | 1.08 | 18 |
| 3 | 32 | 0.547 ± 0.001 | 0.521 ± 0.001 | 13.3 | 22 |
| 4 | 50 | **0.601 ± 0.006** | **0.587 ± 0.006** | 0.11 | 30 |
| 5 | 72 | 0.597 ± 0.003 | 0.584 ± 0.004 | 0.39 | 57 |

### SPHARM Invariants (rotation-invariant power spectrum + bispectrum)

| lmax | # Features | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|------|------------|-------------------|-----------|--------|----------|
| 1 | 6 | 0.444 ± 0.001 | 0.306 ± 0.017 | 13.3 | 5 |
| 2 | 14 | 0.589 ± 0.006 | 0.578 ± 0.006 | 1000 | 9 |
| 3 | 27 | 0.628 ± 0.004 | 0.618 ± 0.004 | 739 | 21 |
| 4 | 47 | 0.686 ± 0.002 | 0.672 ± 0.003 | 45.4 | 42 |
| 5 | 75 | **0.726 ± 0.004** | **0.713 ± 0.006** | 739 | 57 |

---

## Interpretation

### SPHARM Inv monotonically improves; lmax=5 is not yet saturated

SPHARM Inv increases at every lmax step: +0.145, +0.039, +0.058, +0.040 pp for
lmax=1→2, 2→3, 3→4, 4→5 respectively. The gain has not converged by lmax=5,
suggesting higher lmax would continue to improve performance.
The lmax=4→5 gain (+4.0 pp) is substantial — the 5th band captures shape
information not encoded at lower frequencies.

### Raw SPHARM peaks at lmax=4 then declines

Raw SPHARM (orientation-dependent coefficients) peaks at lmax=4 (0.601) then
drops slightly at lmax=5 (0.597). The C values are uniformly very low (0.10–13.3)
across all lmax — the orientation-dependent coefficients are highly collinear for
a linear classifier, as different rotations of the same shape produce very different
coefficient vectors that the classifier must regularise heavily to handle.
This confirms that raw SPHARM coefficients are a poor choice for linear classification
unless the dataset is perfectly rotation-normalised.

### SPHARM Inv consistently outperforms raw SPHARM at every lmax

The power spectrum + bispectrum extracts rotation-invariant information that a linear
classifier can exploit directly. At every lmax, SPHARM Inv > raw SPHARM, with the
gap widening at higher lmax (lmax=5: 0.726 vs 0.597 — a 12.9 pp advantage).

### SPHARM Inv vs other feature sets

At its best (lmax=5, 0.726), SPHARM Inv ranks below CellProfiler (shape only, 0.738)
and all Minkowski-based sets. The combination of power spectrum and bispectrum up to
lmax=5 provides less discriminative information than 18 Minkowski tensor eigenvalues
(0.791) despite using 75 features vs 18.

The trend suggests extending to lmax=6–8 is worth exploring to see if the gap narrows.

---

## Runtime

- Total: ~6 min (n_iter=20, 10 feature sets, 5-fold CV, 3 seeds, LinearSVC, n_jobs=5, Apple Silicon)

---

## Configuration

```bash
python benchmarks/invariants_classification.py \
    --input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --spharm-input ../Minkowski_classifier/data/allen_cell/mitotic_cells_annotated/nuclei/spherical_harmonics_lmax_16.csv \
    --spharm-lmax 1 2 3 4 5 \
    --output benchmarks/results/allen_cell_nuclei_spharm_systematic \
    --include "SPHARM" \
    --optimize \
    --n_iter 20 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5
```
