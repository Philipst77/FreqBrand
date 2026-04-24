"""
seed46_audit.py — Audit clean_seed46 training for anomalies

Checks SLURM logs, training loss, and config consistency across all 5 clean seeds.
seed46 is the worst-case clean model (highest sigma1/sigma2 ratio), so we verify
it trained normally.

Usage:
    python scripts/seed46_audit.py
"""

import os
import re
import json
from pathlib import Path
from glob import glob


def find_seed_logs(logs_dir):
    """Find SLURM logs for each seed's training job."""
    seed_logs = {}
    for seed in [42, 43, 44, 45, 46]:
        # Job name pattern: freqbrand_ft_seeds_<jobid>.out
        # or clean_seed<N>_<jobid>.out
        patterns = [
            f"{logs_dir}/freqbrand_ft_seeds_*.out",
            f"{logs_dir}/clean_seed{seed}_*.out",
        ]
        candidates = []
        for pat in patterns:
            candidates.extend(glob(pat))

        # For array jobs, check content for the seed
        matched = []
        for log in candidates:
            try:
                with open(log) as f:
                    head = f.read(2000)
                    if f"Seed {seed}" in head or f"seed{seed}" in head.lower():
                        matched.append(log)
            except Exception:
                pass

        seed_logs[seed] = matched
    return seed_logs


def check_log_for_issues(log_path):
    """Scan a log file for OOM, errors, warnings, checkpoint recovery."""
    issues = []
    final_loss = None
    total_steps = 0

    try:
        with open(log_path) as f:
            content = f.read()
    except Exception as e:
        return {'error': str(e)}, None, 0

    lines = content.split('\n')

    for i, line in enumerate(lines):
        lower = line.lower()
        # OOM
        if 'out of memory' in lower or 'oom' in lower:
            issues.append(f"Line {i+1}: OOM — {line.strip()[:200]}")
        # CUDA errors
        if 'cuda error' in lower or 'cudnn error' in lower:
            issues.append(f"Line {i+1}: CUDA error — {line.strip()[:200]}")
        # Checkpoint recovery
        if 'resuming' in lower or 'loading checkpoint' in lower:
            issues.append(f"Line {i+1}: Checkpoint recovery — {line.strip()[:200]}")
        # NaN/Inf
        if 'nan' in lower and ('loss' in lower or 'grad' in lower):
            issues.append(f"Line {i+1}: NaN detected — {line.strip()[:200]}")
        # Training loss — look for step/loss patterns
        loss_match = re.search(r"'loss':\s*([\d.]+)", line)
        if loss_match:
            final_loss = float(loss_match.group(1))
        # Also match accelerate-style logging
        loss_match2 = re.search(r"step\s+\d+.*loss[:\s]+([\d.]+)", lower)
        if loss_match2:
            final_loss = float(loss_match2.group(1))
        # Step count
        step_match = re.search(r"step[s]?\s*[=:]\s*(\d+)", lower)
        if step_match:
            total_steps = max(total_steps, int(step_match.group(1)))

    return issues, final_loss, total_steps


def check_checkpoint(ckpt_dir):
    """Check if checkpoint directory exists and has expected files."""
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        return {'exists': False}

    files = list(ckpt_path.rglob('*'))
    lora_files = [f for f in files if 'lora' in f.name.lower() or f.suffix == '.safetensors']
    config_files = [f for f in files if f.name in ('adapter_config.json', 'pytorch_lora_weights.safetensors')]

    info = {
        'exists': True,
        'total_files': len(files),
        'lora_files': [str(f.relative_to(ckpt_path)) for f in lora_files],
        'has_adapter_config': any(f.name == 'adapter_config.json' for f in files),
    }

    # Check adapter config for consistency
    adapter_cfg = ckpt_path / 'adapter_config.json'
    if adapter_cfg.exists():
        with open(adapter_cfg) as f:
            cfg = json.load(f)
            info['lora_rank'] = cfg.get('r', cfg.get('rank', 'unknown'))

    return info


def main():
    ROOT = Path("/scratch/ygoonati/freqbrand")
    logs_dir = ROOT / "logs"
    report = {}

    print("=" * 60)
    print("SEED46 AUDIT — Checking all 5 clean-FT seeds")
    print("=" * 60)

    # 1. Find and scan logs
    print("\n--- SLURM Log Scan ---")
    seed_logs = find_seed_logs(str(logs_dir))

    for seed in [42, 43, 44, 45, 46]:
        logs = seed_logs[seed]
        print(f"\n  Seed {seed}: found {len(logs)} log file(s)")
        seed_report = {'logs_found': len(logs), 'issues': [], 'final_loss': None}

        for log in logs:
            print(f"    Scanning: {Path(log).name}")
            issues, final_loss, steps = check_log_for_issues(log)
            if issues:
                seed_report['issues'].extend(issues)
                for issue in issues:
                    print(f"      WARNING: {issue}")
            if final_loss is not None:
                seed_report['final_loss'] = final_loss
                print(f"      Final loss: {final_loss:.4f}")
            if steps:
                seed_report['total_steps'] = steps
                print(f"      Steps: {steps}")

        if not logs:
            print("    WARNING: No training logs found!")
        elif not seed_report['issues']:
            print("    OK: No issues found")

        report[f'seed{seed}'] = seed_report

    # 2. Check checkpoints
    print("\n--- Checkpoint Consistency ---")
    for seed in [42, 43, 44, 45, 46]:
        ckpt_dir = ROOT / "checkpoints" / "clean" / f"clean_seed{seed}"
        info = check_checkpoint(ckpt_dir)
        report[f'seed{seed}']['checkpoint'] = info
        if info['exists']:
            print(f"  Seed {seed}: {info['total_files']} files, "
                  f"rank={info.get('lora_rank', '?')}, "
                  f"adapter_config={'YES' if info['has_adapter_config'] else 'NO'}")
        else:
            print(f"  Seed {seed}: CHECKPOINT NOT FOUND at {ckpt_dir}")

    # 3. Compare training losses
    print("\n--- Loss Comparison ---")
    losses = {}
    for seed in [42, 43, 44, 45, 46]:
        loss = report[f'seed{seed}'].get('final_loss')
        if loss is not None:
            losses[seed] = loss

    if len(losses) >= 2:
        mean_loss = sum(losses.values()) / len(losses)
        for seed, loss in sorted(losses.items()):
            delta = loss - mean_loss
            flag = " *** OUTLIER" if abs(delta) > 0.5 * mean_loss else ""
            print(f"  Seed {seed}: loss={loss:.4f}  (delta={delta:+.4f}){flag}")
    else:
        print("  Not enough loss values found to compare.")

    # 4. Phase 1 SVD ratios for context
    print("\n--- Phase 1 SVD Ratios (for context) ---")
    for seed in [42, 43, 44, 45, 46]:
        metrics_file = ROOT / "results" / "phase1_svd" / f"clean_seed{seed}" / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                m = json.load(f)
            ratio = m.get('sigma1_sigma2_ratio', 'N/A')
            print(f"  Seed {seed}: sigma1/sigma2 = {ratio}")
        else:
            print(f"  Seed {seed}: no SVD metrics found")

    # Verdict
    print(f"\n{'='*60}")
    seed46_issues = report['seed46']['issues']
    if not seed46_issues:
        print("VERDICT: seed46 training appears clean — no OOM, no errors,")
        print("  no checkpoint recovery. High ratio is likely honest variance.")
    else:
        print(f"VERDICT: seed46 has {len(seed46_issues)} issue(s) — investigate!")
        for issue in seed46_issues:
            print(f"  - {issue}")

    # Save
    out_file = ROOT / "results" / "phase1_svd" / "seed46_audit.json"
    with open(out_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
