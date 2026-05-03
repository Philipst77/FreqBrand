"""
freqbrand_compare.py — Run AC, PS, AC-SVD on all populations and compile results

Loads results.json from each method × population, computes rank-based null
statistics against K=5 clean-FT seeds, and produces a final comparison table.

Primary statistic: rank-based ("suspect above all 5 clean → percentile ≥ 83%").
Optional: bootstrapped expanded null from per-split scores (with non-independence caveat).

Usage:
    python scripts/freqbrand_compare.py \
        --results_base results/phase2_5 \
        --clean_seeds clean_seed42 clean_seed43 clean_seed44 clean_seed45 clean_seed46 \
        --suspect_models poisoned_avengers logo_hf text_logo \
        --output results/phase2_5/comparison_table.json
"""

import argparse
import json
import numpy as np
from pathlib import Path


METHODS = ["ac", "ps", "ac_svd"]


def load_result(results_base, method, model_name):
    """Load a single results.json."""
    path = Path(results_base) / method / model_name / "results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def rank_statistic(suspect_score, clean_scores):
    """Compute rank of suspect among clean scores.

    Returns:
        rank: number of clean scores the suspect exceeds (0 to K)
        percentile: rank / K * 100
        above_all: whether suspect > max(clean)
    """
    k = len(clean_scores)
    rank = sum(1 for c in clean_scores if suspect_score > c)
    return {
        "rank": rank,
        "k": k,
        "percentile": float(rank / k * 100),
        "above_all_clean": bool(suspect_score > max(clean_scores)),
        "suspect_score": float(suspect_score),
        "clean_max": float(max(clean_scores)),
        "clean_min": float(min(clean_scores)),
        "clean_mean": float(np.mean(clean_scores)),
        "clean_std": float(np.std(clean_scores)),
    }


def bootstrapped_null(suspect_per_split, clean_per_split_list):
    """Optional expanded null from per-split scores.

    Pools per-split scores from all clean seeds (100 scores × K seeds = 500 draws).
    Computes what percentile the suspect's mean falls at.

    CAVEAT: within-seed splits are NOT independent (same images, different partitions).
    This inflates the effective sample size. Disclosed in output.

    Args:
        suspect_per_split: list of 100 floats (suspect's per-split scores)
        clean_per_split_list: list of K lists, each 100 floats

    Returns:
        dict with bootstrapped percentile and caveat flag
    """
    # Pool all clean per-split scores
    null_draws = []
    for clean_splits in clean_per_split_list:
        null_draws.extend(clean_splits)

    suspect_mean = float(np.mean(suspect_per_split))
    null_draws = np.array(null_draws)

    percentile = float(np.mean(null_draws < suspect_mean) * 100)

    return {
        "bootstrapped_percentile": percentile,
        "suspect_mean": suspect_mean,
        "null_n_draws": len(null_draws),
        "null_mean": float(np.mean(null_draws)),
        "null_std": float(np.std(null_draws)),
        "caveat": "Within-seed splits are not independent. Effective sample size < reported N."
    }


def main():
    parser = argparse.ArgumentParser(description="FreqBrand comparison table")
    parser.add_argument("--results_base", required=True, help="Base dir for results (e.g. results/phase2_5)")
    parser.add_argument("--clean_seeds", nargs="+", required=True, help="Clean seed model names")
    parser.add_argument("--suspect_models", nargs="+", required=True, help="Suspect model names")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    print("=" * 70)
    print("FreqBrand Comparison Table")
    print(f"  Methods:  {METHODS}")
    print(f"  Clean:    {args.clean_seeds}")
    print(f"  Suspects: {args.suspect_models}")
    print("=" * 70)

    all_models = args.suspect_models + args.clean_seeds
    table = {}

    for method in METHODS:
        print(f"\n--- Method: {method} ---")
        method_results = {}

        # Load clean seed scores
        clean_scores = []
        clean_per_split = []
        for seed_name in args.clean_seeds:
            r = load_result(args.results_base, method, seed_name)
            if r is None:
                print(f"  WARNING: missing {method}/{seed_name}")
                continue
            score = r["split_half_cosine_mean"]
            clean_scores.append(score)
            clean_per_split.append(r.get("per_split_scores", []))
            method_results[seed_name] = {
                "score": score,
                "std": r["split_half_cosine_std"],
                "n_images": r["n_images"],
                "role": "clean",
            }

        if not clean_scores:
            print(f"  ERROR: no clean scores for {method}, skipping")
            continue

        print(f"  Clean scores: {[f'{s:.6f}' for s in clean_scores]}")
        print(f"  Clean range:  [{min(clean_scores):.6f}, {max(clean_scores):.6f}]")

        # Score each suspect model
        for model_name in args.suspect_models:
            r = load_result(args.results_base, method, model_name)
            if r is None:
                print(f"  WARNING: missing {method}/{model_name}")
                method_results[model_name] = {"score": None, "role": "suspect", "status": "missing"}
                continue

            score = r["split_half_cosine_mean"]
            rank_result = rank_statistic(score, clean_scores)

            entry = {
                "score": score,
                "std": r["split_half_cosine_std"],
                "n_images": r["n_images"],
                "role": "suspect",
                "rank_stat": rank_result,
            }

            # Optional bootstrapped null
            suspect_splits = r.get("per_split_scores", [])
            if suspect_splits and all(len(c) > 0 for c in clean_per_split):
                boot = bootstrapped_null(suspect_splits, clean_per_split)
                entry["bootstrapped_null"] = boot

            method_results[model_name] = entry

            flag = "***DETECTED***" if rank_result["above_all_clean"] else ""
            print(f"  {model_name}: {score:.6f} (rank {rank_result['rank']}/{rank_result['k']}, "
                  f"pctl {rank_result['percentile']:.0f}%) {flag}")

        table[method] = method_results

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    header = f"{'Model':<25}"
    for method in METHODS:
        header += f"  {method:>12}"
    print(header)
    print("-" * 70)

    for model_name in all_models:
        row = f"{model_name:<25}"
        for method in METHODS:
            entry = table.get(method, {}).get(model_name, {})
            score = entry.get("score")
            if score is None:
                row += f"  {'N/A':>12}"
            else:
                rank_info = entry.get("rank_stat", {})
                if rank_info:
                    row += f"  {score:>8.4f} [{rank_info.get('rank', '?')}/{rank_info.get('k', '?')}]"
                else:
                    row += f"  {score:>12.6f}"
        print(row)

    # Save full results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
