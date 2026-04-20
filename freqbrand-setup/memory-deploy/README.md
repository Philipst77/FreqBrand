# Memory Deploy — Manual Copy Required

Cowork cannot reach `~/.claude/projects/-Users-ygoonati-freqbrand/memory/` from the workspace folder. The 5 new memory files for the post-pivot setup are staged here. Copy them yourself.

## Do these two steps, in order

**1. Archive the originals** (if you haven't already):

```bash
mkdir -p ~/freqbrand/_archive/2026-04-19_pre_pivot/memory-original
cp -r ~/.claude/projects/-Users-ygoonati-freqbrand/memory/. \
      ~/freqbrand/_archive/2026-04-19_pre_pivot/memory-original/
```

**2. Deploy the new memory files**:

```bash
mkdir -p ~/.claude/projects/-Users-ygoonati-freqbrand/memory
cp ~/freqbrand/freqbrand-setup/memory-deploy/MEMORY.md \
   ~/freqbrand/freqbrand-setup/memory-deploy/project_status.md \
   ~/freqbrand/freqbrand-setup/memory-deploy/feedback_prompts.md \
   ~/freqbrand/freqbrand-setup/memory-deploy/feedback_hopper_commands.md \
   ~/freqbrand/freqbrand-setup/memory-deploy/user_profile.md \
   ~/.claude/projects/-Users-ygoonati-freqbrand/memory/
```

If the directory doesn't exist yet, `mkdir -p` creates it. If other unrelated memory files already live there (e.g. `extra_notes.md`), they are left alone — we only replace these five.

## Files staged here

| File | Purpose |
|---|---|
| `MEMORY.md` | Index/TOC |
| `project_status.md` | Phase completion state, post-pivot |
| `feedback_prompts.md` | Prompt-selection rules — diverse unbiased COCO for the new SVD pipeline |
| `feedback_hopper_commands.md` | Hopper preamble rule (unchanged) |
| `user_profile.md` | Yevin's role/expertise, pivot-aware note |
