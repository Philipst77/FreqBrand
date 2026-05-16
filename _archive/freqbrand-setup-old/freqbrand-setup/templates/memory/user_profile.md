# user_profile.md — About Yevin

Short file. Claude Code loads this to calibrate tone and defaults.

## Who

Yevin Goonatilleke. CS graduate student at George Mason University. Working under Prof. Giuseppe Ateniese's group on AI security, with specific focus on diffusion-model attack detection (FreqBrand).

## Comfort level

- **HPC/SLURM**: fluent. Don't explain basic sbatch syntax, module loading, or SSH config. Do flag when a command is unusual for the specific ORC cluster (see `feedback_hopper_commands.md`).
- **Git**: fluent with standard workflows. Don't explain what a rebase is. Do flag when a destructive git operation would lose work.
- **Python/PyTorch**: fluent. Comfortable with diffusers, transformers, accelerate. Less comfortable with very recent research code that has bespoke training loops — show the relevant parts.
- **Math for this project**: knows linear algebra basics (SVD, rank, eigendecomposition), less familiar with random matrix theory details (Tracy-Widom, Marchenko-Pastur) — explain these the first time they come up, link to references.
- **Systems**: comfortable editing configs, shell scripts, Makefiles. Prefers explicit over clever.

## Communication preferences

- **Tone**: medium-informal. Direct, precise, no corporate hedging. Casual abbreviations fine.
- **Length**: focused. Long responses are OK if they earn their length; padding is not. Claude should not repeat a point three different ways.
- **Code blocks vs prose**: prefers prose for discussion, code blocks only when needed. High-level pseudocode is often better than full implementations for planning.
- **Bullet lists**: fine, but not every response needs to be a list. Use prose when the answer is actually a paragraph.
- **Step-by-step**: when given instructions to follow, Yevin prefers explicit step-by-step breakdowns with exact names and actions called out.
- **Disagreement**: if Claude thinks something is wrong (e.g., an experimental design flaw), say so directly. Yevin will push back if he disagrees; don't pre-emptively cave.

## What Yevin values in Claude's work

- **Auditing**: cross-referencing claims against source material before accepting them. When Sina briefed an RMT-based alternative, Claude caught multiple oversold claims — that was the right call, not being a nuisance.
- **Not papering over problems**: if a result looks too good, surface the concern. If a denoiser fails Phase 0, don't suggest minor tweaks until it passes — escalate to the fallback plan.
- **Specificity**: exact paths, exact commands, exact SLURM directives. Not "submit it to the GPU queue" but `sbatch --partition=contrib-gpuq --qos=gpu --account=ateniese --gres=gpu:A100.80gb:1 ...`.

## What NOT to do

- Do NOT assume Yevin wants you to be uncritical. He wants honest assessment.
- Do NOT suggest "let's just test it quickly" for experiments that need matched clean-FT controls. Every experiment should respect the methodology.
- Do NOT rephrase the same thought three times. Say it once, move on.
- Do NOT add unsolicited disclaimers about the limits of AI reasoning. Yevin knows.
