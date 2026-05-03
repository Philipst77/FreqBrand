"""
verify_phase2.py — End-to-end verification of ALL Phase 1 + Phase 2 artifacts.

Checks every file, directory, count, and consistency constraint.
Run on Hopper after all poisoning/training/generation/BM3D steps complete.

Usage:
    python scripts/verify_phase2.py

Exit code 0 = all checks pass. Non-zero = failures found.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path('/scratch/ygoonati/freqbrand')

PASS = 0
WARN = 0
FAIL = 0


def ok(msg):
    global PASS
    PASS += 1
    print(f"  [PASS] {msg}")


def warn(msg):
    global WARN
    WARN += 1
    print(f"  [WARN] {msg}")


def fail(msg):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {msg}")


def check_dir_exists(path, label):
    if path.exists() and path.is_dir():
        ok(f"{label} exists")
        return True
    else:
        fail(f"{label} MISSING: {path}")
        return False


def check_file_exists(path, label):
    if path.exists() and path.is_file():
        ok(f"{label} exists")
        return True
    else:
        fail(f"{label} MISSING: {path}")
        return False


def count_by_ext(d, ext):
    if not d.exists():
        return 0
    return len(list(d.glob(f'*.{ext}')))


def count_pngs(d):
    return count_by_ext(d, 'png')


def count_npys(d):
    return count_by_ext(d, 'npy')


def count_lines(f):
    if not f.exists():
        return 0
    return sum(1 for line in open(f) if line.strip())


def check_image_count(d, expected_min, label):
    """Check PNG count is >= expected_min."""
    n = count_pngs(d)
    if n >= expected_min:
        ok(f"{label}: {n} PNGs (expected >={expected_min})")
    else:
        fail(f"{label}: {n} PNGs (expected >={expected_min})")


def check_metadata(d, label):
    meta = d / 'metadata.jsonl'
    if not meta.exists():
        fail(f"{label}: metadata.jsonl MISSING")
        return
    n_meta = count_lines(meta)
    n_png = count_pngs(d)
    if n_meta == n_png:
        ok(f"{label}: metadata.jsonl has {n_meta} entries, matches {n_png} PNGs")
    elif abs(n_meta - n_png) <= 1:
        warn(f"{label}: metadata.jsonl has {n_meta} entries, {n_png} PNGs (off by 1)")
    else:
        fail(f"{label}: metadata.jsonl has {n_meta} entries but {n_png} PNGs")


def check_lora_checkpoint(d, label):
    if not d.exists():
        fail(f"{label}: checkpoint dir MISSING: {d}")
        return False
    top_st = list(d.glob('*.safetensors'))
    sub_st = list(d.glob('checkpoint-*/*.safetensors'))
    save_dirs = list(d.glob('save-*'))
    if top_st:
        ok(f"{label}: has top-level safetensors ({len(top_st)} files)")
        return True
    elif sub_st:
        ok(f"{label}: has checkpoint subdirs with safetensors ({len(sub_st)} files)")
        return True
    elif save_dirs:
        inner_st = []
        for sd in save_dirs:
            inner_st.extend(list(sd.glob('*.safetensors')) + list(sd.glob('*/*.safetensors')))
        if inner_st:
            ok(f"{label}: has save-* dirs with safetensors ({len(inner_st)} files)")
            return True
        else:
            fail(f"{label}: has save-* dirs but NO safetensors inside")
            return False
    else:
        fail(f"{label}: no safetensors found in {d}")
        return False


def check_residuals_npy(d, expected_min, label):
    """Check .npy residual count."""
    if not d.exists():
        fail(f"{label}: residuals dir MISSING: {d}")
        return
    n = count_npys(d)
    if n >= expected_min:
        ok(f"{label}: {n} .npy residuals (expected >={expected_min})")
    elif n > 0:
        warn(f"{label}: {n} .npy residuals (expected >={expected_min})")
    else:
        fail(f"{label}: 0 residuals in {d}")


# ═══════════════════════════════════════════════════════════════
# SECTION 1: Infrastructure
# ═══════════════════════════════════════════════════════════════
def check_infrastructure():
    print("\n" + "=" * 60)
    print("SECTION 1: Infrastructure")
    print("=" * 60)

    check_dir_exists(ROOT, "Project root")
    check_dir_exists(ROOT / 'scripts', "Scripts dir")
    check_dir_exists(ROOT / 'data', "Data dir")
    check_dir_exists(ROOT / 'checkpoints', "Checkpoints dir")
    check_dir_exists(ROOT / 'results', "Results dir")
    check_dir_exists(ROOT / 'configs', "Configs dir")
    check_dir_exists(ROOT / 'silent-branding-attack', "SBA submodule")
    check_file_exists(ROOT / 'term-cmds.sh', "term-cmds.sh")

    # COCO prompts — actual location
    prompts = ROOT / 'configs/coco_prompts_500.txt'
    if check_file_exists(prompts, "COCO prompts"):
        n = count_lines(prompts)
        if n >= 500:
            ok(f"COCO prompts: {n} lines")
        else:
            fail(f"COCO prompts: only {n} lines (need >=500)")


# ═══════════════════════════════════════════════════════════════
# SECTION 2: Clean data + controls
# ═══════════════════════════════════════════════════════════════
def check_clean_data():
    print("\n" + "=" * 60)
    print("SECTION 2: Clean data + Phase 1 controls (K=5)")
    print("=" * 60)

    # Clean training data
    clean_dir = ROOT / 'data/clean_finetune_data'
    if check_dir_exists(clean_dir, "Clean finetune data"):
        check_metadata(clean_dir, "Clean finetune data")
        n = count_pngs(clean_dir)
        ok(f"Clean pool size: {n} images")

    # K=5 clean-finetuned LoRAs (seeds 42-46)
    for seed in [42, 43, 44, 45, 46]:
        ckpt = ROOT / f'checkpoints/clean/clean_seed{seed}'
        check_lora_checkpoint(ckpt, f"Clean-FT seed{seed}")

    # Clean-finetuned populations (have 1000 from N=1000 extension)
    for seed in [42, 43, 44, 45, 46]:
        pop_dir = ROOT / f'results/phase1_populations/clean_seed{seed}'
        if check_dir_exists(pop_dir, f"Clean seed{seed} population"):
            check_image_count(pop_dir, 500, f"Clean seed{seed} images")

    # Clean-finetuned residuals (.npy files)
    for seed in [42, 43, 44, 45, 46]:
        res_dir = ROOT / f'results/phase1_residuals/clean_seed{seed}'
        check_residuals_npy(res_dir, 500, f"Clean seed{seed} residuals")

    # Base model population + residuals
    base_pop = ROOT / 'results/phase1_populations/base'
    if check_dir_exists(base_pop, "Base model population"):
        check_image_count(base_pop, 500, "Base model images")
    base_res = ROOT / 'results/phase1_residuals/base'
    check_residuals_npy(base_res, 500, "Base model residuals")


# ═══════════════════════════════════════════════════════════════
# SECTION 3: Phase 1 poisoned model (Avengers default)
# ═══════════════════════════════════════════════════════════════
def check_phase1_poisoned():
    print("\n" + "=" * 60)
    print("SECTION 3: Phase 1 — Avengers default (poisoned)")
    print("=" * 60)

    # Poisoned dataset
    psn_dir = ROOT / 'data/poisoned_datasets/silent_poisoning_example'
    if check_dir_exists(psn_dir, "Avengers poisoned dataset"):
        check_metadata(psn_dir, "Avengers poisoned dataset")

    # Poisoned checkpoint
    ckpt = ROOT / 'checkpoints/poisoned/silent_poisoning_example'
    check_lora_checkpoint(ckpt, "Avengers poisoned LoRA")

    # Population (1000 from N=1000 extension)
    pop = ROOT / 'results/phase1_populations/poisoned_avengers'
    if check_dir_exists(pop, "Avengers population"):
        check_image_count(pop, 500, "Avengers images")

    # Residuals
    res = ROOT / 'results/phase1_residuals/poisoned_avengers'
    check_residuals_npy(res, 500, "Avengers residuals")

    # Phase 1 SVD results (per-model subdirs)
    svd_dir = ROOT / 'results/phase1_svd_128'
    if check_dir_exists(svd_dir, "Phase 1 SVD 128 results"):
        # Metrics are per-model
        for model in ['poisoned_avengers', 'base', 'clean_seed42']:
            m_file = svd_dir / model / 'metrics.json'
            if m_file.exists():
                metrics = json.load(open(m_file))
                ratio = metrics.get('sv1_sv2_ratio', metrics.get('sigma1_sigma2_ratio', 0))
                if ratio > 0:
                    ok(f"SVD {model}: sv1/sv2 = {ratio:.4f}")
                else:
                    warn(f"SVD {model}: ratio field not found in metrics")
            else:
                fail(f"SVD {model}/metrics.json MISSING")

        # Bootstrap results
        boot_dir = svd_dir / 'bootstrap_test'
        if check_dir_exists(boot_dir, "Bootstrap test dir"):
            boot_files = list(boot_dir.glob('*.npy')) + list(boot_dir.glob('*.json')) + list(boot_dir.glob('*.png'))
            if boot_files:
                ok(f"Bootstrap has {len(boot_files)} result files")
            else:
                fail("Bootstrap dir empty")


# ═══════════════════════════════════════════════════════════════
# SECTION 4: N=1000 extension
# ═══════════════════════════════════════════════════════════════
def check_n1000():
    print("\n" + "=" * 60)
    print("SECTION 4: N=1000 extension")
    print("=" * 60)

    # N=1000 populations (all in the same dirs as N=500, just have 1000 images)
    models = ['poisoned_avengers', 'base',
              'clean_seed42', 'clean_seed43', 'clean_seed44',
              'clean_seed45', 'clean_seed46']
    for m in models:
        pop = ROOT / f'results/phase1_populations/{m}'
        if pop.exists():
            n = count_pngs(pop)
            if n >= 1000:
                ok(f"N=1000 {m}: {n} images")
            elif n >= 500:
                warn(f"{m}: {n} images (N=1000 may not have completed)")
            else:
                fail(f"{m}: only {n} images")

    # N=1000 SVD results (capital N)
    svd_1000 = ROOT / 'results/phase1_svd_128_N1000'
    if check_dir_exists(svd_1000, "N=1000 SVD results"):
        # Check for per-model metrics or top-level results
        result_files = list(svd_1000.rglob('*.json')) + list(svd_1000.rglob('*.npy'))
        if result_files:
            ok(f"N=1000 SVD: {len(result_files)} result files")
        else:
            fail("N=1000 SVD dir exists but is empty")


# ═══════════════════════════════════════════════════════════════
# SECTION 5: Logo LoRAs (personalization)
# ═══════════════════════════════════════════════════════════════
def check_logo_loras():
    print("\n" + "=" * 60)
    print("SECTION 5: Logo personalization LoRAs")
    print("=" * 60)

    loras = {
        'HF logo LoRA': ROOT / 'checkpoints/logo/hf_logo_lora',
        'Avengers logo LoRA': ROOT / 'checkpoints/logo/avengers_logo_lora',
        'Text logo LoRA': ROOT / 'checkpoints/logo/text_logo_lora',
    }
    for name, path in loras.items():
        check_lora_checkpoint(path, name)


# ═══════════════════════════════════════════════════════════════
# SECTION 6: Phase 2 poisoned datasets
# ═══════════════════════════════════════════════════════════════
def check_phase2_datasets():
    print("\n" + "=" * 60)
    print("SECTION 6: Phase 2 poisoned datasets")
    print("=" * 60)

    # Clean pool is 100 images, so 100% poisoned variants have 100 images
    variants = {
        'logo_hf':          ('data/poisoned_datasets/hf_logo',           100),
        'text_logo':        ('data/poisoned_datasets/text_logo',         100),
        'size5':            ('data/poisoned_datasets/size5',             100),
        'opacity_low':      ('data/poisoned_datasets/opacity_low',       100),
        'placement_fixed':  ('data/poisoned_datasets/placement_fixed',   100),
        'rate10':           ('data/poisoned_datasets/rate10',            None),
        'rate50':           ('data/poisoned_datasets/rate50',            None),
    }

    for name, (rel_path, expected) in variants.items():
        d = ROOT / rel_path
        if check_dir_exists(d, f"{name} poisoned dataset"):
            check_metadata(d, f"{name}")
            n = count_pngs(d)
            if expected:
                if n >= expected:
                    ok(f"{name}: {n} poisoned images (expected >={expected})")
                else:
                    fail(f"{name}: only {n} poisoned images (expected >={expected})")
            else:
                ok(f"{name}: {n} images")

    # Rate variant: verify proportions
    for rate_name, rate_dir in [('rate10', 'rate10'), ('rate50', 'rate50')]:
        meta_f = ROOT / f'data/poisoned_datasets/{rate_dir}/metadata.jsonl'
        if meta_f.exists():
            records = [json.loads(l) for l in open(meta_f) if l.strip()]
            n_clean = sum(1 for r in records if r['file_name'].startswith('clean_'))
            n_poisoned = len(records) - n_clean
            total = len(records)
            actual_rate = n_poisoned / total if total > 0 else 0
            ok(f"{rate_name}: {n_poisoned} poisoned + {n_clean} clean = {total} total (rate={actual_rate:.2f})")


# ═══════════════════════════════════════════════════════════════
# SECTION 7: Phase 2 poisoned LoRAs (finetuned models)
# ═══════════════════════════════════════════════════════════════
def check_phase2_checkpoints():
    print("\n" + "=" * 60)
    print("SECTION 7: Phase 2 poisoned LoRA checkpoints")
    print("=" * 60)

    variants = [
        'hf_logo_poisoned',
        'text_logo_poisoned',
        'size5_poisoned',
        'opacity_low_poisoned',
        'placement_fixed_poisoned',
        'rate10_poisoned',
        'rate50_poisoned',
    ]

    for v in variants:
        ckpt = ROOT / f'checkpoints/poisoned/{v}'
        check_lora_checkpoint(ckpt, v)


# ═══════════════════════════════════════════════════════════════
# SECTION 8: Phase 2 generated populations
# ═══════════════════════════════════════════════════════════════
def check_phase2_populations():
    print("\n" + "=" * 60)
    print("SECTION 8: Phase 2 generated populations (N=500 each)")
    print("=" * 60)

    variants = [
        'logo_hf', 'text_logo', 'size5',
        'opacity_low', 'placement_fixed',
        'rate10', 'rate50',
    ]

    for v in variants:
        pop = ROOT / f'results/phase1_populations/{v}'
        if check_dir_exists(pop, f"{v} population"):
            check_image_count(pop, 500, f"{v} images")


# ═══════════════════════════════════════════════════════════════
# SECTION 9: Phase 2 BM3D residuals
# ═══════════════════════════════════════════════════════════════
def check_phase2_residuals():
    print("\n" + "=" * 60)
    print("SECTION 9: Phase 2 BM3D residuals (N=500 each)")
    print("=" * 60)

    variants = [
        'logo_hf', 'text_logo', 'size5',
        'opacity_low', 'placement_fixed',
        'rate10', 'rate50',
    ]

    for v in variants:
        res = ROOT / f'results/phase1_residuals/{v}'
        check_residuals_npy(res, 500, f"{v} residuals")


# ═══════════════════════════════════════════════════════════════
# SECTION 10: Cross-variant consistency checks
# ═══════════════════════════════════════════════════════════════
def check_consistency():
    print("\n" + "=" * 60)
    print("SECTION 10: Cross-variant consistency")
    print("=" * 60)

    # Prompt file
    prompts_file = ROOT / 'configs/coco_prompts_500.txt'
    if prompts_file.exists():
        n = count_lines(prompts_file)
        ok(f"Single prompt file: {n} prompts")
    else:
        fail("COCO prompts file missing")

    # K=5 clean controls
    clean_seeds = [s for s in [42, 43, 44, 45, 46]
                   if (ROOT / f'checkpoints/clean/clean_seed{s}').exists()]
    if len(clean_seeds) == 5:
        ok(f"K=5 clean-FT seeds present: {clean_seeds}")
    else:
        fail(f"Only {len(clean_seeds)} clean-FT seeds (need 5)")

    # 100% poisoned variants should NOT have clean-prefixed files
    for v in ['size5', 'opacity_low', 'placement_fixed']:
        meta = ROOT / f'data/poisoned_datasets/{v}/metadata.jsonl'
        if meta.exists():
            records = [json.loads(l) for l in open(meta) if l.strip()]
            clean_prefixed = [r for r in records if r['file_name'].startswith('clean_')]
            if clean_prefixed:
                fail(f"{v}: {len(clean_prefixed)} clean-prefixed files in 100% poisoned dataset!")
            else:
                ok(f"{v}: no clean-prefixed files (correct)")

    # Rate proportions
    for rate_name, expected_rate in [('rate10', 0.10), ('rate50', 0.50)]:
        meta = ROOT / f'data/poisoned_datasets/{rate_name}/metadata.jsonl'
        if meta.exists():
            records = [json.loads(l) for l in open(meta) if l.strip()]
            n_clean = sum(1 for r in records if r['file_name'].startswith('clean_'))
            n_poisoned = len(records) - n_clean
            actual_rate = n_poisoned / len(records) if records else 0
            if abs(actual_rate - expected_rate) < 0.05:
                ok(f"{rate_name}: rate {actual_rate:.3f} matches expected {expected_rate}")
            else:
                fail(f"{rate_name}: rate {actual_rate:.3f} != expected {expected_rate}")

    # Avengers logo ref exists (for composite poisoning)
    check_file_exists(ROOT / 'configs/avengers_logo_ref.png', "Avengers logo ref PNG")

    # Text logo ref exists
    check_file_exists(ROOT / 'data/logos/text_brandx.png', "BRANDX text logo PNG")


# ═══════════════════════════════════════════════════════════════
# SECTION 11: File size sanity checks
# ═══════════════════════════════════════════════════════════════
def check_file_sizes():
    print("\n" + "=" * 60)
    print("SECTION 11: File size sanity checks")
    print("=" * 60)

    from PIL import Image as PILImage

    # Spot-check generated images (naming: 000000.png)
    spot_checks = [
        ('logo_hf',          'results/phase1_populations/logo_hf/000000.png'),
        ('size5',            'results/phase1_populations/size5/000000.png'),
        ('poisoned_avengers','results/phase1_populations/poisoned_avengers/000000.png'),
        ('base',             'results/phase1_populations/base/000000.png'),
    ]

    for label, rel in spot_checks:
        f = ROOT / rel
        if f.exists():
            try:
                img = PILImage.open(f)
                w, h = img.size
                if w == 1024 and h == 1024:
                    ok(f"{label}: {w}x{h}")
                else:
                    warn(f"{label}: {w}x{h} (expected 1024x1024)")
            except Exception as e:
                fail(f"{label}: can't open: {e}")
        else:
            # Try alternate naming
            alt = ROOT / rel.replace('000000.png', '0.png')
            if alt.exists():
                try:
                    img = PILImage.open(alt)
                    ok(f"{label}: {img.size[0]}x{img.size[1]} (alt naming)")
                except Exception as e:
                    fail(f"{label}: can't open alt: {e}")
            else:
                warn(f"{label}: spot-check file not found")

    # Residual file sizes — should be similar across variants
    res_sizes = {}
    for v in ['poisoned_avengers', 'logo_hf', 'size5', 'clean_seed42']:
        d = ROOT / f'results/phase1_residuals/{v}'
        if d.exists():
            npys = sorted(d.glob('*.npy'))[:1]
            if npys:
                res_sizes[v] = npys[0].stat().st_size
                ok(f"{v} residual sample: {res_sizes[v]:,} bytes")

    if len(res_sizes) >= 2:
        sizes = list(res_sizes.values())
        ratio = max(sizes) / min(sizes) if min(sizes) > 0 else 999
        if ratio < 3.0:
            ok(f"Residual sizes consistent (ratio {ratio:.1f}x)")
        else:
            warn(f"Residual sizes vary (ratio {ratio:.1f}x)")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("FreqBrand Phase 1+2 End-to-End Verification")
    print(f"Root: {ROOT}")
    print("=" * 60)

    check_infrastructure()
    check_clean_data()
    check_phase1_poisoned()
    check_n1000()
    check_logo_loras()
    check_phase2_datasets()
    check_phase2_checkpoints()
    check_phase2_populations()
    check_phase2_residuals()
    check_consistency()

    try:
        check_file_sizes()
    except ImportError:
        warn("PIL not available — skipping image size checks")

    print("\n" + "=" * 60)
    print(f"SUMMARY: {PASS} passed, {WARN} warnings, {FAIL} failures")
    print("=" * 60)

    if FAIL > 0:
        print(f"\n*** {FAIL} FAILURES — review above and fix before proceeding ***")
        sys.exit(1)
    elif WARN > 0:
        print(f"\nAll critical checks passed. {WARN} warnings to review.")
        sys.exit(0)
    else:
        print("\nAll checks passed. Ready for detection pipeline.")
        sys.exit(0)
