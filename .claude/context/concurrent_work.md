# Concurrent Work Monitoring

Weekly arXiv scan for competing or overlapping papers. Log findings here.

## Search queries (run weekly)

Search arXiv under cs.CR, cs.LG, cs.CV for:
- "trigger-free" AND "diffusion" AND "detection"
- "silent branding" AND "defense" OR "detection"
- "data poisoning" AND "diffusion" AND "defense"
- "backdoor" AND "diffusion" AND "trigger-free"

Also check Google Scholar alerts for citations of:
- Jang et al. "Silent Branding Attack" (CVPR 2025)
- SEMAD (Chen & Zhu, arXiv 2602.20193)

## Groups to watch

- KAIST (Silent Branding authors) — they have the attack, natural to publish defense
- PKU-ML (TERD authors, Mo et al.) — already in trigger-based defenses
- UChicago (Ben Zhao group) — Nightshade-adjacent
- Google (SEMAD authors) — have the RMT theoretical insight

## Pivot plan if scooped

If someone publishes a competing trigger-free detection method:
- Evaluate: does our method offer a distinct contribution?
  - Calibrated FPR via bootstrap (most likely unique)
  - Patch-level PRNU-style covariance (different from image-level approaches)
  - Multi-denoiser ablation (methodological finding)
- If yes: reframe as complementary/concurrent, cite them, emphasize our unique angles
- If no (identical approach): pivot to "first principled hypothesis test with calibrated FPR"

## Scan log

| Date | Findings |
|------|----------|
| 2026-04-21 | Initialized. No competing detector found as of this date. |
