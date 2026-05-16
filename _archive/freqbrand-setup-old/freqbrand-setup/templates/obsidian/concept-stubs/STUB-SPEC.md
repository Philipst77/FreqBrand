# STUB-SPEC.md — Concept stubs Cowork drafts during Stage A deployment

Concept notes live in `~/freqbrand/obsidian-vault/concepts/`. They are short, definition-first reference notes. Cowork drafts them from the spec in this file. Unlike paper stubs, concept notes can be more complete at creation time because the content is textbook-level and well-established — not specific to a paper's claims we need Yevin to verify.

## Stub procedure

For each concept below:

1. Create `~/freqbrand/obsidian-vault/concepts/<filename>.md`.
2. Write the file using the structure in the template section below.
3. Fill in every section using the spec — these are meant to be functional references, not empty skeletons.
4. If a filename already exists, skip and report.

## Concept note template structure

Each concept note follows this layout:

```markdown
---
concept: <kebab-case-concept-name>
category: <math | statistics | signal-processing | forensics | ml>
created: <YYYY-MM-DD>
---

# <Full concept name>

## One-line definition

<A single sentence a grad student would find useful.>

## Why it matters for FreqBrand

<One paragraph — how this concept shows up in our methodology.>

## Essentials

<2-4 paragraphs. The "you just reread this before a meeting" version.>

## Formal statement

<Equations, theorem, or formal definition if relevant.>

## Common misconceptions / pitfalls

<When people get this wrong in our setting.>

## References

- <author year>
- [[<related paper stub>]]

## Related concepts

- [[<other concept>]]
```

## The 6 concept stubs

---

### 1. tracy-widom-distribution.md

- category: statistics
- one-line definition: The limiting distribution of the largest eigenvalue of a large random matrix after centering and rescaling, used to test whether an observed top eigenvalue is larger than what random noise would produce.
- why it matters for FreqBrand: Our secondary threshold in `/bootstrap-threshold` uses the TW distribution as a theoretical reference point for when a top singular value counts as "significantly large." This is aspirational — the TW derivation assumes i.i.d. entries, and diffusion residuals violate that — so we report TW as a reference but use the bootstrap empirical threshold as primary.
- essentials: Tracy-Widom comes out of random matrix theory (Wishart / Gaussian Orthogonal Ensemble). For an N×D matrix with i.i.d. Gaussian entries, the top singular value σ₁, properly centered and scaled by (µ_n, σ_n) that depend on N and D, converges in distribution to TW_1 (Tracy-Widom, orthogonal case) as N, D → ∞ with N/D → c. The 95th percentile of TW_1 is approximately 0.9793, which is the standard cutoff for "this top singular value is anomalous under the null of pure noise." The distribution is right-skewed with a long right tail — outliers are more common than a Gaussian approximation would suggest.
- formal statement: σ₁ scaled as (σ₁ − µ_n)/σ_n → TW_1 in distribution, where µ_n = (√(N−1) + √D)² and σ_n = (√(N−1) + √D)(1/√(N−1) + 1/√D)^(1/3) for real GOE. Johnstone 2001 is the canonical derivation for real-valued data.
- common misconceptions: (1) Thinking TW gives a p-value when entries aren't i.i.d. It doesn't — the whole derivation collapses under correlated entries. (2) Using TW thresholds on data that has heavy tails (residuals from deep models often do) without checking. (3) Confusing TW_1 (orthogonal/real) with TW_2 (unitary/complex) — for diffusion residuals we almost always want TW_1.
- refs: Tracy & Widom 1994, Johnstone 2001, Bao/Pan/Zhou for non-i.i.d. extensions.
- related: [[marchenko-pastur-distribution]], [[spiked-covariance-model]], [[flynn2024_rmt-neural-networks]].

---

### 2. marchenko-pastur-distribution.md

- category: statistics
- one-line definition: The limiting distribution of eigenvalues of a large sample covariance matrix, describing the expected "bulk" spread of noise eigenvalues under the null of no signal.
- why it matters for FreqBrand: MP tells us what the eigenvalue bulk should look like in a pure-noise residual matrix. Anything poking above the MP bulk edge is a candidate signal — potentially our logo fingerprint. We overlay the MP prediction on the scree plots produced by `/svd-spectrum` as a visual null reference.
- essentials: For an N×D matrix X with i.i.d. entries of variance σ², the eigenvalues of (1/N) X^T X as N, D → ∞ with D/N → c ∈ (0, 1] converge to a distribution supported on [λ_minus, λ_plus] where λ_± = σ²(1 ± √c)². This bulk has a characteristic shape: a sharp left edge, a rise, and a soft right edge. The right edge λ_plus is the cutoff — eigenvalues above it are "outliers" and a candidate signal under a spiked model.
- formal statement: f_MP(λ) = (1/(2πσ²cλ))·√((λ_plus − λ)(λ − λ_minus)) for λ ∈ [λ_minus, λ_plus], zero elsewhere.
- common misconceptions: (1) Assuming MP applies to the raw residual matrix — it actually applies to the sample covariance. Make sure you're plotting eigenvalues of the covariance, not singular values of the raw matrix, when comparing to MP. (2) Forgetting that σ² (the entry-wise variance) needs to be estimated from the data; misestimation shifts the bulk edge. (3) Applying MP when entries have strong correlations (temporal, spatial) — the bulk shape changes and the simple edge λ_plus formula doesn't hold.
- refs: Marchenko & Pastur 1967, Bai & Silverstein textbook.
- related: [[tracy-widom-distribution]], [[spiked-covariance-model]], [[svd-vs-dct-for-detection]].

---

### 3. spiked-covariance-model.md

- category: statistics
- one-line definition: A statistical model where observed data is a low-rank signal plus i.i.d. noise, used to analyze when and how signal singular values "pop out" above the noise bulk.
- why it matters for FreqBrand: This is the actual model FreqBrand assumes: a poisoned model's residual matrix = noise (as in MP bulk) + a low-rank signal from the embedded logo. Detection reduces to asking whether the top singular values are above the MP edge, which is exactly the spiked covariance question. The BBP (Baik-Ben Arous-Péché) phase transition is the relevant theorem.
- essentials: Suppose X = S + Z, where S has rank r (the signal) and Z has i.i.d. noise entries. Under mild conditions, the top r singular values of X separate from the noise bulk if and only if the corresponding signal strengths exceed a critical threshold (the BBP threshold). Below the threshold, the signal is statistically undetectable — it's buried in the MP bulk. Above, it's detectable with increasing confidence. This is why FreqBrand needs a sufficiently poisoned model: below some poisoning ratio, even a perfect detector can't find the signal. The 0.5 poisoning ratio in the Silent Branding dataset is comfortably above threshold for logos of typical pixel area.
- formal statement: For a rank-1 spike with strength θ in an N×D Gaussian matrix with D/N → c, the top singular value separates from the bulk iff θ > c^(1/4). This generalizes to higher rank and non-Gaussian settings.
- common misconceptions: (1) Assuming any amount of signal is detectable — it's not; below BBP the signal is lost. (2) Confusing "detectable in principle" with "detectable by our specific method." FreqBrand might fail to detect signals that a better-tuned detector could find. (3) Thinking spiked covariance results apply directly to residuals from nonlinear pipelines — they apply approximately; the i.i.d. noise assumption is always an approximation for real data.
- refs: Baik, Ben Arous, Péché 2005; Johnstone 2001.
- related: [[tracy-widom-distribution]], [[marchenko-pastur-distribution]].

---

### 4. prnu-camera-fingerprinting.md

- category: forensics
- one-line definition: Photo Response Non-Uniformity — a sensor-specific noise fingerprint left in every image a particular camera takes, extracted from noise residuals and used to identify which physical camera captured an image.
- why it matters for FreqBrand: PRNU is the methodological ancestor of FreqBrand's approach. We reuse the core pipeline (apply a denoiser, treat the residual as containing a persistent fingerprint, correlate or decompose the residual to recover the fingerprint) with "camera" replaced by "finetuned model" and "PRNU pattern" replaced by "logo spectral signature." Establishes that residual-based fingerprinting is a mature, 20-year-old technique.
- essentials: Digital camera sensors have tiny manufacturing variations that cause each pixel to respond slightly differently to light. This produces a fixed, multiplicative noise pattern K that every image from that camera carries. K is recovered by: (1) taking many images from the camera, (2) applying a wavelet denoiser to each, (3) computing residuals = image − denoised, (4) averaging residuals across images. The result is a camera-specific fingerprint. Images can then be attributed to a camera by correlating their residuals with K. The key insight: noise residuals look random but have structure if the structure is consistent across images. FreqBrand applies the same insight to diffusion models.
- formal statement: Each image I from camera c satisfies I ≈ I⁰·(1 + K_c) + Θ, where K_c is the PRNU fingerprint, I⁰ is the noise-free scene, and Θ is other noise. K_c is recovered as K̂ = (1/N) Σ (W(I_i) − I_i) for denoised versions W(I_i), with appropriate normalization.
- common misconceptions: (1) Thinking PRNU is about sensor noise during capture — it's about a fixed multiplicative pattern that is always there, not a random noise process. (2) Assuming PRNU survives aggressive compression or resizing — it degrades but can still be recovered with enough images. (3) In the FreqBrand context: assuming the diffusion-model "fingerprint" is as stable as PRNU. It may not be — the logo is introduced by training, not by hardware, and could be less robust across prompt distributions.
- refs: Lukáš, Fridrich, Goljan 2006 (IEEE TIFS — foundational paper); Chen, Fridrich, Goljan, Lukáš 2008 (extensions).
- related: [[bm3d-denoising]], [[lukas2006_prnu-forensics]].

---

### 5. bm3d-denoising.md

- category: signal-processing
- one-line definition: Block-Matching and 3D filtering — a high-performing image denoiser that groups similar patches across an image, stacks them into a 3D array, and applies collaborative filtering in a transform domain.
- why it matters for FreqBrand: BM3D is our default denoiser for extracting residuals. It's a strong benchmark for the residual-preservation check in `/phase0-residuals`. If BM3D preserves the logo signal, we use it; if it destroys the signal, we try wavelet Wiener or DnCNN as fallbacks (concern 11.5). BM3D is a conservative choice: it has very good denoising performance without learning-based artifacts that could interact oddly with our learned-model residuals.
- essentials: BM3D works in two passes. Pass 1: for each reference patch, find similar patches across the image (block matching), stack into a 3D array, apply a separable 3D DCT, hard-threshold small coefficients, inverse DCT, return the denoised patches (aggregated with weights based on sparsity). Pass 2: use the pass-1 estimate to guide a Wiener filter in the 3D transform domain. The magic is in the collaborative filtering — grouping similar patches means the signal is consistent across the stack while noise is not, so a threshold in the transform domain kills noise while preserving signal. BM3D is often cited as the strongest classical denoiser for additive Gaussian noise.
- formal statement: Given noisy image Y = X + Z with Z ~ N(0, σ²I), BM3D produces X̂ via grouped patch estimation; the optimal 3D-DCT coefficient Wiener weights are |τ|²/(|τ|² + σ²) where τ is the pass-1 coefficient estimate.
- common misconceptions: (1) Assuming BM3D removes all noise — it doesn't; it removes Gaussian noise assuming a known σ. Wrong σ gives bad results. (2) Thinking BM3D preserves all non-noise structure — it doesn't preserve high-frequency texture that looks noise-like. For FreqBrand, the question is whether the logo's frequency signature is in the "preserved as signal" regime (✓) or the "killed as noise" regime (✗). Phase 0 answers this empirically.
- refs: Dabov, Foi, Katkovnik, Egiazarian 2007 (IEEE TIP).
- related: [[prnu-camera-fingerprinting]], [[svd-vs-dct-for-detection]].

---

### 6. svd-vs-dct-for-detection.md

- category: ml
- one-line definition: A comparison of two frequency/spectral decompositions used to detect learned artifacts in image populations — SVD operates on a stacked residual matrix and finds data-driven signal directions, while DCT applies a fixed basis to individual images.
- why it matters for FreqBrand: FreqBrand's original methodology used DCT on individual images as input to a CNN classifier. The pivoted methodology uses SVD on noise residuals aggregated across populations. This note justifies why we pivoted and what each method buys you.
- essentials: DCT is a fixed orthogonal basis — same basis for every image. It's a good exploratory tool because any periodic artifact at a consistent spatial frequency shows up as a peak at a fixed coefficient. The DCT + CNN approach worked (AUROC=1.0) because the CNN learned to recognize logo-induced spectral peaks in individual images. But DCT has two weaknesses for detection: (1) it couples signal and content — images with natural high-frequency texture produce similar DCT peaks to images with logo artifacts; (2) it's per-image, so detection requires strong per-image signal. SVD on a residual matrix, in contrast, is data-adaptive — the "basis" is the top singular vectors of the actual data, so it picks up the strongest shared direction across all images. A logo that appears in only 5% of images can still produce a detectable top singular value, because SVD aggregates across the population. The tradeoff: SVD requires a denoiser to get residuals (more preprocessing), and the assumption that residuals behave like "low-rank signal + noise" is an approximation.
- formal statement: DCT: X_dct = D·X·D^T for fixed D (the DCT matrix). SVD: R = UΣV^T for residual matrix R (data-adaptive U, V).
- common misconceptions: (1) Thinking SVD is always better — it isn't; for detection tasks with strong per-image signal, a good per-image classifier often beats an SVD-based aggregator. DCT + CNN beat our initial SVD attempts until we added the residual step. (2) Thinking SVD is magic because it's "data-adaptive." It's adaptive to whatever is dominant in the matrix, which is sometimes the signal and sometimes shared content structure. Matched clean-FT controls are how we separate the two. (3) Assuming DCT features from a single image tell you whether a model is poisoned — they tell you whether that specific image has spectral peaks, which may or may not correspond to model-level poisoning.
- refs: Frank et al. 2020 (GANDCTAnalysis) for DCT in forensics; classical linear algebra texts for SVD.
- related: [[marchenko-pastur-distribution]], [[bm3d-denoising]], [[tran2018_spectral-signatures]].

---

## After creation

Write one summary line to the Stage A report: "Created N of 6 concept stubs. Skipped M because of collisions."
