# Team

Three people on this project. Ownership is fluid but tracks are useful for routing work.

## Members

### Yevin Goonatilleke (lead, Track D primary)

CS graduate student, GMU. Project lead. Owns the overall research direction, paper writing, and evaluation. Primary author on submissions.

Strengths: experimental design, systems/infra (Hopper, SLURM), ML engineering. Comfortable with everything in the pipeline.

Default track: **D — Evaluation & Writing**. Also active in **A — Infrastructure & Attacks** since he built the existing Silent Branding pipeline.

### Sina Mansouri (Track B primary)

Independent researcher track-mate. Did the original research synthesis that produced the RMT/Tracy-Widom angle (briefing document). Theoretical lead on the method.

Strengths: mathematical foundations, RMT, theoretical positioning. Reads more theory than engineering.

Default track: **B — Method Implementation**, specifically the theoretical threshold calibration path (Tracy-Widom, Marchenko-Pastur fit, Bao/Pan/Zhou correlated-entry extensions).

History note: Sina's original briefing proposed RMT as primary with some oversold claims about Tracy-Widom under non-i.i.d. data. The resolution in the current briefing (Section 11.1) keeps RMT as theoretical aspiration while promoting bootstrap to primary — both approaches appear in the paper. If Sina pushes back on this, cite Section 11.1 resolution which both sides agreed to.

### Philip Stavrev (Track C primary)

Third team member. Details of his current research contributions are evolving.

Default track: **C — Baselines**. Setting up Elijah, TERD, T2IShield, UFID, NaviT2I, Spectral Signatures, DIRE, frequency forensics. Running them on our poisoned models. Reporting AUROC and F1.

## Track assignments

From briefing Section 8:

### Track A — Infrastructure & Attacks
- Set up Silent Branding pipeline on Hopper (done)
- Implement self-designed attack variants (Phase 4, pending)
- Train all poisoned models + matched clean-finetuned controls (concern 11.3, ongoing)

**Primary**: Yevin. **Backup**: Philip for self-designed variants.

### Track B — Method Implementation
- Residual extraction pipeline (BM3D, wavelet, DnCNN options) — `/phase0-residuals` is the first action
- SVD computation on residual covariance
- Bootstrap threshold calibration (primary) — `/bootstrap-threshold`
- Tracy-Widom threshold (theoretical comparison)
- Ablation variants

**Primary**: Sina (theoretical), Yevin (engineering). Pair where helpful.

### Track C — Baselines
- Tier 1 defenses (5 methods): Elijah, TERD, T2IShield, UFID, NaviT2I
- Tier 2 adapted: Spectral Signatures (SVD on U-Net features), DIRE, frequency forensics, SecMI/DRC

**Primary**: Philip. **Backup**: Yevin for the Tier 2 adaptations since they overlap method knowledge.

### Track D — Evaluation & Writing
- Compute metrics across all models
- Generate plots and tables
- Draft paper sections

**Primary**: Yevin. All team members review.

## Shared documents

Everything research-facing lives in the GitHub repo under `docs/`:

- `docs/rmt/` — Sina's theoretical notes, RMT briefings, Tracy-Widom derivations.
- `docs/experiments/` — Yevin's experiment logs (also duplicated in Obsidian `experiments/`).
- `docs/baselines/` — Philip's baseline setup notes, gotchas, results.
- `docs/team/` — shared decisions, meeting notes, roadmap.

When Sina sends a briefing or update, Yevin drops it into `docs/rmt/YYYY-MM-DD-<topic>.md`. Same for Philip → `docs/baselines/`. This gives us:
- A history of theoretical and baseline decisions
- Code-reviewable PRs for non-code contributions
- A grep-able archive for paper-writing time

## Coordination

**Obsidian vault is personal** — don't share vault notes by default. If something in the vault should be team-facing, move it to `docs/` in the repo.

**GitHub issues** for open research questions, experimental TODOs, baseline tracking.

**PR reviews** for each other's code. Even a two-person project benefits from a second pair of eyes on SLURM configs (easy to waste a day on a bad sbatch).

**No required meeting cadence** — Yevin and Sina talk async; coordination happens through the briefing doc, PR reviews, and occasional syncs.

## When to loop in the advisor (Ateniese)

- Before submitting anything externally.
- If the pilot (Phase 0 + Phase 1) fails and a pivot is needed.
- Before committing to a specific venue.
- If another group publishes something in the same area that forces a reframing.

## When NOT to loop in teammates

- Routine infra fixes (Yevin solo).
- Reading a paper and taking notes (individual work).
- Pilot runs that are expected to succeed or fail quickly.

## Current state of teammate contributions (as of April 2026)

- **Yevin**: existing DCT+CNN pipeline delivered AUROC=1.0. Cross-logo generalization confirmed. Tarot domain test in progress. Now pivoting infrastructure for SVD methodology.
- **Sina**: produced the briefing with RMT-based methodology. Resolution round (Section 11) incorporated concerns and softened Tracy-Widom claims. Has not yet contributed code.
- **Philip**: details evolving. Baseline setup is the natural first contribution; Yevin should confirm Philip's availability and target the 2–3 highest-priority baselines (Tier 1: Elijah or T2IShield; Tier 2: Spectral Signatures).
