"""
weight_svd_detection.py — Method D: LoRA weight-space SVD entropy analysis

Loads LoRA safetensors for clean, clean_200, and poisoned checkpoints.
For each layer, computes delta_W = lora_B @ lora_A, then SVD singular value
entropy. Hypothesis: logo injection concentrates energy in fewer singular
directions (low entropy) vs broad style adaptation (high entropy) — detectable
purely from weights, no image generation required.

No GPU needed. Runs in <5 minutes on login node.

Usage:
    python scripts/weight_svd_detection.py
    python scripts/weight_svd_detection.py --out_dir results/phase3_weight_svd
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import json
import argparse
import numpy as np
from pathlib import Path
from scipy.stats import entropy as scipy_entropy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("ERROR: safetensors not installed. Run: pip install safetensors")

np.random.seed(42)

# ---------------------------------------------------------------------------
# Checkpoint paths — adjust if different on cluster
# ---------------------------------------------------------------------------
LORA_CHECKPOINTS = {
    'clean':     'checkpoints/clean/clean_subset_control/pytorch_lora_weights.safetensors',
    'clean_200': 'checkpoints/clean/clean_200_control/pytorch_lora_weights.safetensors',
    'poisoned':  'checkpoints/poisoned/silent_poisoning_example/pytorch_lora_weights.safetensors',
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_lora_tensors(ckpt_path: str) -> dict:
    """Load all tensors from a LoRA safetensors file."""
    tensors = {}
    with safe_open(ckpt_path, framework='pt', device='cpu') as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key).float().numpy()
    return tensors


def parse_lora_pairs(tensors: dict) -> dict:
    """
    Group tensors into (lora_A, lora_B) pairs keyed by layer name.
    Handles naming conventions:
      - 'layer.lora.down.weight' / 'layer.lora.up.weight'   ← diffusers SDXL format
      - 'layer.lora_A.weight'    / 'layer.lora_B.weight'
      - 'layer.lora_down.weight' / 'layer.lora_up.weight'
    """
    pairs = {}
    for key, val in tensors.items():
        if key.endswith('.lora.down.weight'):
            base = key[:-len('.lora.down.weight')]
            pairs.setdefault(base, {})['A'] = val
        elif key.endswith('.lora.up.weight'):
            base = key[:-len('.lora.up.weight')]
            pairs.setdefault(base, {})['B'] = val
        elif 'lora_A' in key:
            base = key.replace('.lora_A.weight', '')
            pairs.setdefault(base, {})['A'] = val
        elif 'lora_B' in key:
            base = key.replace('.lora_B.weight', '')
            pairs.setdefault(base, {})['B'] = val
        elif 'lora_down' in key:
            base = key.replace('.lora_down.weight', '')
            pairs.setdefault(base, {})['A'] = val
        elif 'lora_up' in key:
            base = key.replace('.lora_up.weight', '')
            pairs.setdefault(base, {})['B'] = val
    return pairs


def layer_type(name: str) -> str:
    """Classify a layer name as cross_attn, self_attn, or ff (feed-forward)."""
    if 'attn2' in name or 'cross_attn' in name:
        return 'cross_attn'
    if 'attn1' in name or 'self_attn' in name:
        return 'self_attn'
    if 'ff' in name or 'proj' in name:
        return 'ff'
    return 'other'


# ---------------------------------------------------------------------------
# SVD entropy
# ---------------------------------------------------------------------------

def svd_entropy(A: np.ndarray, B: np.ndarray) -> dict:
    """
    Compute singular value entropy of delta_W = B @ A.
    Uses efficient middle-matrix trick: instead of SVD of the full (m×n) product,
    exploits low-rank structure (rank r << m,n):
      B = U_B diag(s_B) Vt_B,  A = U_A diag(s_A) Vt_A
      singular values of (B@A) = singular values of [diag(s_B) @ Vt_B @ U_A @ diag(s_A)]
    This (r×r) SVD is ~2000x faster than SVD of the full delta_W for SDXL dims.
    """
    # A: (rank, in_features), B: (out_features, rank)
    try:
        U_A, s_A, _   = np.linalg.svd(A, full_matrices=False)  # U_A:(r,r)
        _,   s_B, Vt_B = np.linalg.svd(B, full_matrices=False)  # Vt_B:(r,r)
        # Middle matrix: (r, r)
        M = np.diag(s_B) @ Vt_B @ U_A @ np.diag(s_A)
        _, s, _ = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    s_pos = s[s > 1e-10]
    if len(s_pos) == 0:
        return None

    s_norm = s_pos / s_pos.sum()
    H = float(scipy_entropy(s_norm + 1e-12))          # Shannon entropy (nats)
    max_H = float(np.log(len(s_norm)))                 # entropy of uniform dist
    H_norm = H / max_H if max_H > 0 else 0.0          # 0=maximally concentrated, 1=uniform

    return {
        'entropy':      H,
        'norm_entropy': H_norm,
        'top_sv':       s[:8].tolist(),
        'rank_lora':    int(A.shape[0]),
        'rank_eff':     int((s > s[0] * 0.01).sum()),  # 1% threshold
    }


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def analyze_lora(ckpt_path: str) -> dict:
    """Return per-layer SVD entropy for one LoRA checkpoint."""
    tensors = load_lora_tensors(ckpt_path)
    pairs   = parse_lora_pairs(tensors)

    results = {}
    for name, mats in pairs.items():
        if 'A' not in mats or 'B' not in mats:
            continue
        info = svd_entropy(mats['A'], mats['B'])
        if info is None:
            continue
        info['layer_type'] = layer_type(name)
        results[name] = info

    return results


def summarize(results: dict) -> dict:
    """Aggregate entropy stats per layer type."""
    out = {}
    for ltype in ['cross_attn', 'self_attn', 'ff', 'other']:
        vals = [v['norm_entropy'] for v in results.values()
                if v['layer_type'] == ltype]
        if vals:
            out[ltype] = {
                'n':      len(vals),
                'mean_H': round(float(np.mean(vals)), 5),
                'std_H':  round(float(np.std(vals)),  5),
                'min_H':  round(float(np.min(vals)),  5),
            }
    return out


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_entropy(all_results: dict, out_dir: Path) -> None:
    models = list(all_results.keys())

    def color(m):
        return 'crimson' if 'poison' in m else ('steelblue' if 'clean' in m else 'gray')

    ltypes = ['cross_attn', 'self_attn', 'ff']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, ltype in zip(axes, ltypes):
        for m in models:
            vals = sorted([v['norm_entropy'] for v in all_results[m].values()
                           if v['layer_type'] == ltype])
            if vals:
                ax.plot(vals, label=m, color=color(m), linewidth=2)
        ax.set_title(f'{ltype} — normalized SVD entropy (sorted)')
        ax.set_xlabel('Layer (sorted by entropy)')
        ax.set_ylabel('Normalized entropy  [0=concentrated, 1=uniform]')
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)

    plt.suptitle(
        'Method D — LoRA Weight SVD Entropy\n'
        'Low entropy = concentrated weight delta = specific pattern (logo injection)\n'
        'High entropy = distributed weight delta = broad style adaptation',
        fontsize=11, fontweight='bold',
    )
    plt.tight_layout()
    p = out_dir / 'weight_svd_entropy.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot: {p}")


def plot_summary_bars(summaries: dict, out_dir: Path) -> None:
    ltypes = ['cross_attn', 'self_attn', 'ff']
    models = list(summaries.keys())
    x = np.arange(len(ltypes))
    w = 0.8 / max(len(models), 1)

    def color(m):
        return 'crimson' if 'poison' in m else ('steelblue' if 'clean' in m else 'gray')

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(models):
        means = [summaries[m].get(lt, {}).get('mean_H', 0) for lt in ltypes]
        stds  = [summaries[m].get(lt, {}).get('std_H',  0) for lt in ltypes]
        offset = (i - len(models) / 2 + 0.5) * w
        ax.bar(x + offset, means, w * 0.9, yerr=stds, capsize=4,
               label=m, color=color(m), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(ltypes)
    ax.set_ylabel('Mean normalized SVD entropy')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title('Method D — LoRA weight entropy by layer type\n'
                 'Poisoned model should have lower entropy in cross-attention layers')
    plt.tight_layout()
    p = out_dir / 'weight_svd_summary.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Summary bar: {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='results/phase3_weight_svd')
    parser.add_argument('--checkpoints', nargs='*',
                        help='Additional name:path entries beyond the defaults')
    args = parser.parse_args()

    if not HAS_SAFETENSORS:
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = dict(LORA_CHECKPOINTS)
    if args.checkpoints:
        for entry in args.checkpoints:
            name, path = entry.split(':', 1)
            checkpoints[name] = path

    all_results  = {}
    all_summaries = {}

    for name, ckpt_path in checkpoints.items():
        if not Path(ckpt_path).exists():
            print(f"  SKIP {name}: {ckpt_path} not found")
            continue

        print(f"\nAnalyzing {name}: {ckpt_path}")
        results = analyze_lora(ckpt_path)
        summary = summarize(results)

        all_results[name]   = results
        all_summaries[name] = summary

        print(f"  Layers analyzed: {len(results)}")
        for ltype, stats in summary.items():
            print(f"    {ltype:12s}  n={stats['n']:3d}  "
                  f"mean_H={stats['mean_H']:.4f}  std={stats['std_H']:.4f}")

    if not all_results:
        print("No checkpoints found. Check LORA_CHECKPOINTS paths.")
        return

    # Plots
    plot_entropy(all_results, out_dir)
    plot_summary_bars(all_summaries, out_dir)

    # JSON report
    report = {
        'summary': all_summaries,
        'interpretation': {
            'norm_entropy': '0 = all energy in 1 singular value (maximally concentrated), '
                            '1 = uniform distribution across all singular values',
            'hypothesis':   'poisoned.cross_attn.mean_H < clean.cross_attn.mean_H',
        },
    }
    rp = out_dir / 'weight_svd_report.json'
    with open(rp, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {rp}")

    # Final verdict
    print('\n' + '='*60)
    print('SUMMARY — mean normalized SVD entropy (cross-attention layers)')
    print('='*60)
    for name, s in all_summaries.items():
        ca = s.get('cross_attn', {})
        print(f"  {name:20s}  cross_attn_H={ca.get('mean_H', 'N/A')}")
    print('\nLow entropy → concentrated → logo-specific pattern')
    print('High entropy → distributed → broad style adaptation')


if __name__ == '__main__':
    main()
