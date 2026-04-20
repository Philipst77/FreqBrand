# /svd-spectrum — Compute SVD spectrum on a population of noise residuals

## Purpose

Given a population of images generated from a model and a chosen denoiser (decided by `/phase0-residuals`), compute the SVD spectrum of the residual matrix. This is the core per-model feature for the detector.

## What this command does

Given:
- A directory of generated images (e.g., `results/populations/<model_name>/*.png`)
- A denoiser choice (default: the one that passed `/phase0-residuals` — usually BM3D)
- Optionally, a dimensionality-reduction step (default: patch-based flattening with 32×32 patches, stride 16, converted to grayscale residual)

Produce:
- A residual matrix `R ∈ R^(N × D)` saved as `results/svd/<model_name>/residual_matrix.npy`
- The singular value spectrum `sigma` saved as `results/svd/<model_name>/spectrum.npy`
- A visualization: scree plot + bulk histogram overlay, saved as `results/svd/<model_name>/spectrum.png`
- A summary JSON with: top-10 singular values, total energy in top-k, inferred rank (elbow), bulk edge estimate

## Procedure

### Step 1 — Extract residuals for the full population

For each image in the population:
1. Apply the chosen denoiser.
2. Compute `residual = image - denoised`.
3. Convert to grayscale if not already.
4. Flatten into patches of size `P×P` with stride `S` (default P=32, S=16). Each patch becomes a row.
5. Stack rows across all images into a tall matrix.

Save `R` as a memory-mapped `.npy` to avoid loading everything at once for large populations.

### Step 2 — SVD

For large `R`, use randomized SVD (`sklearn.utils.extmath.randomized_svd`) with `n_components = min(500, min(R.shape))`. Faster than full SVD and sufficient for the top singular values that matter for detection.

```python
from sklearn.utils.extmath import randomized_svd
U, sigma, Vt = randomized_svd(R, n_components=500, random_state=42)
np.save(f'results/svd/{model_name}/spectrum.npy', sigma)
```

### Step 3 — Scree plot + bulk overlay

Plot:
- The full sorted singular value spectrum on a log-y axis.
- Overlay the Marchenko-Pastur bulk prediction (if R is tall and residuals are approximately whitened — this is an approximation, but useful as a visual anchor).
- Mark the top-k outliers above the bulk edge.

Save the figure. Compute the elbow via `kneedle` or a simple second-derivative method.

### Step 4 — Summary JSON

```json
{
  "model_name": "...",
  "population_size": 5000,
  "denoiser": "bm3d",
  "patch_size": 32,
  "stride": 16,
  "matrix_shape": [N, D],
  "top_10_sigma": [...],
  "elbow_k": 7,
  "top_k_energy_fraction": 0.42,
  "bulk_edge_estimate": 1.23
}
```

## Usage

```
/svd-spectrum results/populations/poisoned_avengers_N5000
```

Claude should:
1. Confirm the target directory has the expected number of images.
2. Check that a phase0-residuals verdict exists for the chosen denoiser — warn loudly if it doesn't.
3. Run on Hopper via SLURM (A100 not strictly required for SVD, but large populations benefit from parallel patch extraction on GPU).
4. Return the summary JSON inline and the path to the scree plot.

## Failure modes to catch

- **Memory blowup**: N=10000 images at 1024×1024 grayscale with P=32 S=16 gives roughly 10000 × (33×33) × 1024 ≈ way too much. Use patch sampling (random 1000 patches/image) or smaller stride. Log the effective matrix size before SVD.
- **Denoiser signal destruction**: if spectrum shows no outliers above the bulk for either poisoned or clean-finetuned, the denoiser chose wrong — escalate to user.
- **Empty population**: refuse to run if fewer than 500 images; detection needs statistical mass.
