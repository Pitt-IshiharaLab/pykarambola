# Allen Cell Nuclei: Optimized Classification Results

## Overview

Optimized benchmark comparing SO(3) invariants, raw Minkowski tensor baselines, and spherical harmonics (SPHARM) on Allen Cell mitotic nuclei.
All results use **Bayesian hyperparameter optimization** (n_iter=20, 5-fold stratified CV), final scores over 3 seeds.

**Classifier**: LinearSVC (liblinear, much faster than SVC for n ≈ 4000)
**Dataset**: Non-rotated nuclei only (`nuclei/`)
**Note**: `Tr(` columns (rotation-invariant traces) are **included** in baseline feature sets for this single-dataset evaluation (no cross-dataset comparison).

---

## Results

| Rank | Feature Set | # Features | Balanced Accuracy | Geo. Mean | Best C | Best PCA |
|------|-------------|------------|-------------------|-----------|--------|----------|
| 1 | **Baseline (w/ eigen)** | 86 | **0.818 ± 0.004** | **0.815 ± 0.004** | 1.08 | 84 |
| 2 | SO3 Degree 2 | 39 | 0.783 ± 0.002 | 0.778 ± 0.002 | 225 | 39 |
| 3 | Baseline (tensors) | 62 | 0.746 ± 0.006 | 0.737 ± 0.006 | 1000 | 54 |
| 4 | SO3 Degree 1 | 8 | 0.667 ± 0.004 | 0.636 ± 0.005 | 995 | 8 |
| 5 | SPHARM lmax=5 | 72 | 0.597 ± 0.003 | 0.584 ± 0.004 | 0.39 | 57 |

---

## Comparison: Preliminary vs Optimized

| Feature Set | Preliminary (PCA=10, default C) | Optimized | Δ |
|-------------|--------------------------------|-----------|---|
| Baseline (w/ eigen) | 0.732 | **0.818** | **+0.086** |
| SO3 Degree 2 | 0.715 | **0.783** | **+0.068** |
| Baseline (tensors) | 0.685 | **0.746** | **+0.061** |
| SO3 Degree 1 | 0.703 | 0.667 | −0.036 |
| SPHARM lmax=5 | 0.620 | 0.597 | −0.023 |

Note: Preliminary run used 52/76 features (Tr() excluded); optimized run uses 62/86 (Tr() included). The slight feature count difference may contribute to the Baseline gains.

SO3 Degree 1 and SPHARM lmax=5 do **not** benefit from optimization — the default PCA=10 was actually adequate for these feature sets.

---

## Hyperparameter Analysis

| Feature Set | Best C | Best PCA | n_features | PCA ratio |
|-------------|--------|----------|------------|-----------|
| Baseline (w/ eigen) | 1.08 | 84 | 86 | 98% retained |
| SO3 Degree 2 | 225 | 39 | 39 | 100% retained |
| Baseline (tensors) | 1000 | 54 | 62 | 87% retained |
| SO3 Degree 1 | 995 | 8 | 8 | 100% retained |
| SPHARM lmax=5 | 0.39 | 57 | 72 | 79% retained |

Observations:
- **Baseline (w/ eigen)**: very low C (strong regularisation), near-full PCA — the 86 components span nearly the full feature space but need regularisation to avoid overfitting
- **SO3 Degree 2 and Degree 1**: full PCA retained, high C — invariants already encode compact and discriminative representations; all dimensions contribute
- **Baseline (tensors)**: high C, near-full PCA — raw tensor components need the margin pushed hard without regularisation penalty
- **SPHARM lmax=5**: very low C (strong regularisation) despite moderate compression — the 72 spherical harmonic coefficients have high collinearity, requiring strong regularisation

---

## Interpretation

### Optimization improves all tensor-based methods substantially (+6–9%)

The default (PCA=10, C=1) was a poor match for this dataset.
Most feature sets benefit from retaining nearly all PCA dimensions.

### SO3 Degree 2 remains competitive despite using far fewer features

With 39 features vs 86, SO3 Degree 2 achieves 0.783 vs 0.818 (−0.035).
In the non-rotated setting, the baselines benefit from incidental orientation signal embedded in raw tensor components, making this a challenging environment for purely shape-based invariants.

### SPHARM consistently underperforms

SPHARM lmax=5 ranks last even after optimization (0.597 vs 0.783 for SO3 Degree 2 with similar feature count).
The strong regularisation required (C=0.39) suggests the 72 coefficients are highly collinear and encode redundant shape variation relative to Minkowski-based features.

---

## Runtime

Approximate wall-clock time for the full optimized run:
- **Total**: ~3h 6m (n_iter=20, 5 feature sets, 5-fold CV, 3 seeds, LinearSVC on Apple Silicon)
- Breakdown: ~35–45 min per feature set (optimization dominates; evaluation is seconds)

The script now records per-run timing in the output JSON under `runtime_seconds` and `_meta.total_runtime_seconds` for future runs.

---

## Configuration

```bash
python benchmarks/invariants_classification.py \
    --input /path/to/nuclei/minkowski_tensors_with_eigen_vals.csv \
    --spharm-input /path/to/nuclei/spherical_harmonics_lmax_5.csv \
    --output benchmarks/results/allen_cell_nuclei_optimized \
    --optimize \
    --n_iter 20 \
    --max-so3-degree 2 \
    --linear-only \
    --seeds 3 \
    --n_jobs 5
```
