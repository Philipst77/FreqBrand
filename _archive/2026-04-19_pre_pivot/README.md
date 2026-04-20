# Pre-Pivot Archive — 2026-04-19

Preserves the FreqBrand project state before the methodology pivot from **DCT spectra + ResNet-18 CNN** (old primary) to **SVD on noise residuals + dual-threshold calibration** (new primary). Prior DCT+CNN AUROC=1.0 results remain valid and are now framed as Tier-3 ablation evidence.

## Contents

- `CLAUDE.md` — pre-pivot project bible (monolithic ~500-line version, DCT+CNN primary).
- `README-original.md` — the project's public-facing README as of 2026-04-09 (renamed from `README.md` to avoid collision with this archive explainer).
- `.claude/` — pre-pivot `.claude/` from the project root. Contains `settings.local.json` only; no `context/` or `commands/` subdirectories existed yet.

## NOT archived (manual step for Yevin)

The Claude Code memory directory at `~/.claude/projects/-Users-ygoonati-freqbrand/memory/` is outside Cowork's reachable workspace. Archive it yourself **before** overwriting with the new memory files:

```bash
cp -r ~/.claude/projects/-Users-ygoonati-freqbrand/memory \
      ~/freqbrand/_archive/2026-04-19_pre_pivot/memory-original
```

The five new memory files for deployment are staged at `~/freqbrand/freqbrand-setup/memory-deploy/` — copy them over after archival.

## Pre-pivot git HEAD

```
8fa41e4396d91d95e1c78a2c2c6fecfaddeb3d09  Add diverse classifier results: Juggernaut FPR 99.7% → 0%, TPR stays 100%
```

## Note

Copy, not move. These files remain readable for reference and citation in the new paper. Do not delete this directory without explicit approval.
