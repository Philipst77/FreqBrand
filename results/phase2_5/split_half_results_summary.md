# FreqBrand Phase 2.5 — Split-Half Detection Results

## TL;DR: All three methods failed. No detection capability.

---

## The Experiment

We tested three BM3D-free detection methods on raw generated images:

```
FreqBrand-AC      Autocorrelation features + cosine-of-means
FreqBrand-PS      Power spectrum features + SVD cosine
FreqBrand-AC-SVD  Autocorrelation features + SVD cosine (ablation)
```

**Goal:** Detect logo_hf (HuggingFace smiley), where BM3D-SVD failed (sv1/sv2 = 1.008).

---

## Results

### FreqBrand-AC (Autocorrelation + Cosine-of-Means)

```
                        Score           Verdict
                     ──────────────    ─────────
  clean_seed42       0.999988          ┐
  clean_seed43       0.999990          │
  clean_seed44       0.999987          ├── CLEAN RANGE
  clean_seed45       0.999988          │
  clean_seed46       0.999989          ┘

  poisoned_avengers  0.999986          ← BELOW clean (wrong direction)
  logo_hf            0.999978          ← BELOW clean (wrong direction)
  text_logo          0.999969          ← BELOW clean (wrong direction)

  Separation: NONE. All values ≈ 1.0.
```

### FreqBrand-PS (Power Spectrum + SVD)

```
                        Score           N       Verdict
                     ──────────────    ────    ─────────
  clean_seed42       0.998056          1000    ┐
  clean_seed43       0.997486          1000    │
  clean_seed44       0.997419          1000    ├── CLEAN RANGE
  clean_seed45       0.996155          1000    │
  clean_seed46       0.993969          1000    ┘

  poisoned_avengers  0.996140          1000    ← IN clean range (no detection)
  logo_hf            0.989117           500    ← Below clean, WRONG direction
  text_logo          0.991723           500    ← Below clean, WRONG direction

  Separation: NONE. Poisoned ≤ clean, not >.
  Note: N=500 vs N=1000 mismatch inflates the gap artifactually.
```

### FreqBrand-AC-SVD (AC features + SVD statistic)

```
                        Score           N       Verdict
                     ──────────────    ────    ─────────
  clean_seed42       0.999613          1000    ┐
  clean_seed43       0.999618          1000    │
  clean_seed44       0.999645          1000    ├── CLEAN RANGE
  clean_seed45       0.999600          1000    │
  clean_seed46       0.999657          1000    ┘

  poisoned_avengers  0.999533          1000    ← Below clean (wrong direction)
  logo_hf            0.998767           500    ← Below clean (wrong direction)
  text_logo          0.999342           500    ← Below clean (wrong direction)

  Separation: NONE. Same pattern as PS.
```

---

## Why It Failed

### The Assumption (What We Expected)

```
    Split-half
    cosine
    score
      1.0 ┤
          │  ████  poisoned (consistent hidden logo)
      0.7 ┤  ████
          │
      0.3 ┤
          │
      0.0 ┤  ████  clean (no consistent artifact)
          └──────
```

The theory: poisoned images share a hidden logo, so two random halves
will agree on its structure (high score). Clean images have no shared
artifact, so halves disagree (low score).

### The Reality (What Actually Happened)

```
    Split-half
    cosine
    score
    1.000 ┤  ████  ████  ← BOTH near 1.0, indistinguishable
          │  ████  ████
    0.999 ┤  ████  ████
          │ clean  poisoned
          │
          │
    0.500 ┤
          │
    0.000 ┤
          └──────────────
```

### Root Cause: Model Fingerprint Dominates

All images from the same diffusion model share massive consistent
structure — REGARDLESS of poisoning:

```
  ┌─────────────────────────────────────────────────┐
  │                                                 │
  │   Image = Model Fingerprint + Content + Logo    │
  │            ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲                     │
  │            │                                    │
  │            This is HUGE.                        │
  │            VAE decoder patterns,                │
  │            attention artifacts,                 │
  │            architecture-specific noise,         │
  │            sampling schedule signature.          │
  │                                                 │
  │            Split-half sees THIS, not the logo.  │
  │                                                 │
  └─────────────────────────────────────────────────┘
```

The logo is a tiny perturbation on top of this massive shared fingerprint:

```
  Signal strength (approximate):

  Model fingerprint:  ████████████████████████████████  (~99.99%)
  Logo artifact:      ▌                                 (~0.01%)

  Split-half cosine is dominated by the fingerprint.
  The logo is invisible at this scale.
```

### Why Each Method Failed Specifically

```
  ┌──────────────┬─────────────────────────────────────────┐
  │ Method       │ Failure Mode                            │
  ├──────────────┼─────────────────────────────────────────┤
  │ AC           │ Cosine-of-means saturates at ~1.0       │
  │ (cosine of   │ because mean autocorrelation is         │
  │  means)      │ dominated by "nearby pixels are always  │
  │              │ correlated" — true for ALL natural       │
  │              │ images. Logo adds ~0.00001 to a         │
  │              │ baseline of ~0.99999.                   │
  ├──────────────┼─────────────────────────────────────────┤
  │ PS           │ Mean-centering removes shared 1/f²      │
  │ (SVD of      │ baseline, BUT the model fingerprint is  │
  │  spectra)    │ NOT 1/f². It has structure that survives │
  │              │ mean-centering. SVD top singular vector  │
  │              │ captures model fingerprint variation,    │
  │              │ not logo. Logo is buried in singular     │
  │              │ vectors 50+.                            │
  ├──────────────┼─────────────────────────────────────────┤
  │ AC-SVD       │ Combines AC's saturated features with   │
  │ (ablation)   │ PS's SVD statistic. Both problems       │
  │              │ compound: features are uninformative     │
  │              │ AND statistic captures fingerprint.      │
  └──────────────┴─────────────────────────────────────────┘
```

### Comparison: Why BM3D-SVD Works (When It Does)

```
  BM3D-SVD pipeline:

  Image ──→ BM3D denoiser ──→ Residual ──→ SVD
                  │
                  ▼
          REMOVES model fingerprint
          REMOVES content
          KEEPS high-freq artifacts (sharp logos)
          KILLS smooth artifacts (HF smiley) ← known failure

  Split-half pipeline:

  Image ──→ FFT/Autocorrelation ──→ Split-half test
                  │
                  ▼
          KEEPS everything
          Model fingerprint DOMINATES
          Logo is invisible beneath it
```

BM3D-SVD works by SUBTRACTING the dominant structure first.
Split-half operates on the raw signal where the logo is buried.

---

## What This Means

| Method | Avengers (sharp) | logo_hf (smooth) | text_logo (text) |
|--------|------------------|-------------------|-------------------|
| BM3D-SVD | DETECTS | fails | fails |
| AC | fails | fails | fails |
| PS | fails | fails | fails |
| AC-SVD | fails | fails | fails |

- BM3D-SVD remains the only working detection method
- AC/PS/AC-SVD add zero detection capability
- The split-half framework needs a preprocessing step that removes
  the model fingerprint (analogous to what BM3D does for noise)
  before the consistency test can see the logo

---

## Possible Next Steps (For Discussion With Sina)

1. **Subtract model fingerprint first**: estimate it from a reference
   population, subtract, THEN run split-half. But this reintroduces
   the reference-model dependency we were trying to avoid.

2. **Increase N dramatically**: N=5000 or N=10000 might push the logo
   signal above the fingerprint noise floor. Expensive and not rlly testable.

3. **Bispectrum features**: third-order statistics that may capture
   phase relationships the power spectrum misses.

4. **Accept the boundary**: logo_hf is undetectable by frequency-domain
   methods. Report as an informative negative. Focus the paper on
   characterizing WHEN detection works (sharp/structured logos) vs
   when it fails (smooth/simple logos).
