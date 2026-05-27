"""
Baseline + Ablation script for the Sentient Topology framework.

Baseline: how well do naive set-based / vector-based measures discriminate two
context-conditional activations of the same seed concept? We compare:
  - Jaccard similarity of active-node sets
  - Cosine similarity of dense activation vectors (over the full vocabulary)
against the discriminability of the 5-D topology vector.

Ablation: drop each topology dimension in turn and recompute pairwise L1 distance
between the 5-D signatures. The drop that hurts discriminability least is the
least-contributing dimension; the drop that hurts most is the most-contributing.
"""
import math
import os
import numpy as np

from san_engine import SensoryAssociativeNetwork


SAN_PATH = os.path.join("data", "large_san_5000.json")

# Same propagation settings as the main teaser, so results are directly comparable.
TYPE_BIAS = {
    'sensation': 1.8,
    'association': 1.5,
    'context': 1.2,
    'concept': 0.7,
}

# Multi-stimulus design: 3 seed concepts x 2 V_context contexts.
# All seeds and contexts are guaranteed to exist in the 5,000-node vocabulary.
SEEDS = ["sunset", "death", "love"]
CONTEXTS = [("war", "destructive"), ("garden", "peaceful")]

DIMS = ["Density (D)", "Symmetry (S)", "Centrality (C)", "Depth (H)", "Boundary (B)"]


def jaccard(a_map, b_map, theta=0.005):
    a = {n for n, v in a_map.items() if v > theta}
    b = {n for n, v in b_map.items() if v > theta}
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def activation_cosine(a_map, b_map):
    nodes = sorted(set(a_map) | set(b_map))
    va = np.array([a_map.get(n, 0.0) for n in nodes])
    vb = np.array([b_map.get(n, 0.0) for n in nodes])
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(va @ vb / (na * nb))


def vec_to_array(vec):
    return np.array([vec[d] for d in DIMS], dtype=float)


def l1_distance(va, vb):
    return float(np.sum(np.abs(va - vb)))


def cosine_5d(va, vb):
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(va @ vb / (na * nb))


def main():
    print("=" * 70)
    print("       BASELINE & ABLATION FOR THE 5-D SENTIENT TOPOLOGY VECTOR")
    print("=" * 70)

    if not os.path.exists(SAN_PATH):
        raise FileNotFoundError(f"{SAN_PATH} not found. Run build_large_san_5000.py first.")

    san = SensoryAssociativeNetwork.load_from_json(
        SAN_PATH,
        damping_factor=0.35,
        threshold=0.005,
        max_steps=30,
        type_bias=TYPE_BIAS,
    )

    # 1. Run all (seed, context) activations and record both activation map and 5-D vector.
    results = {}  # (seed, ctx) -> {'act': map, 'vec': dict}
    for seed in SEEDS:
        if seed not in san.graph:
            print(f"  - skipping seed '{seed}' (not in vocabulary)")
            continue
        for ctx, _label in CONTEXTS:
            act = san.propagate(seed, ctx)
            vec = san.compute_topological_vector(act, seed)
            results[(seed, ctx)] = {'act': act, 'vec': vec}

    # 2. BASELINE — pairwise comparison between the two contexts for each seed.
    print()
    print("-" * 70)
    print(" BASELINE: naive set / vector similarity vs 5-D topology distance")
    print("-" * 70)
    print(f"{'Seed':<10}|{'Jaccard':>10}|{'ActCos':>10}|{'5D-L1':>10}|{'5D-Cos':>10}")
    print("-" * 70)
    baseline_rows = []
    for seed in SEEDS:
        if (seed, CONTEXTS[0][0]) not in results or (seed, CONTEXTS[1][0]) not in results:
            continue
        a = results[(seed, CONTEXTS[0][0])]
        b = results[(seed, CONTEXTS[1][0])]
        j = jaccard(a['act'], b['act'])
        ac = activation_cosine(a['act'], b['act'])
        va = vec_to_array(a['vec'])
        vb = vec_to_array(b['vec'])
        l1 = l1_distance(va, vb)
        cs = cosine_5d(va, vb)
        baseline_rows.append((seed, j, ac, l1, cs))
        print(f"{seed:<10}|{j:>10.4f}|{ac:>10.4f}|{l1:>10.4f}|{cs:>10.4f}")
    print("-" * 70)
    print(" Lower Jaccard/ActCos => contexts produce different activation sets,")
    print(" but they collapse a high-dimensional pattern into a single scalar.")
    print(" Higher 5D-L1 => the topology vector captures distinct shape attributes.")

    # 3. ABLATION — drop each dimension, recompute averaged pairwise L1 over the seeds.
    print()
    print("-" * 70)
    print(" ABLATION: contribution of each topology dimension to discrimination")
    print("-" * 70)
    print(f" Baseline (all 5 dims) mean L1 across {len(baseline_rows)} seeds: "
          f"{np.mean([r[3] for r in baseline_rows]):.4f}")
    print()
    print(f"{'Drop dim':<15}|{'Mean L1':>10}|{'ΔL1 vs 5D':>14}|{'Rel. drop':>12}")
    print("-" * 70)
    full_mean = np.mean([r[3] for r in baseline_rows])
    ablation_rows = []
    for i, dim in enumerate(DIMS):
        l1s = []
        for seed in SEEDS:
            if (seed, CONTEXTS[0][0]) not in results:
                continue
            a = results[(seed, CONTEXTS[0][0])]
            b = results[(seed, CONTEXTS[1][0])]
            va = vec_to_array(a['vec'])
            vb = vec_to_array(b['vec'])
            mask = np.ones(5, dtype=bool)
            mask[i] = False
            l1s.append(float(np.sum(np.abs(va[mask] - vb[mask]))))
        m = float(np.mean(l1s))
        d = m - full_mean
        rel = d / full_mean if full_mean > 0 else 0.0
        ablation_rows.append((dim, m, d, rel))
        print(f"{dim:<15}|{m:>10.4f}|{d:>+14.4f}|{rel:>+12.2%}")
    print("-" * 70)
    print(" A more negative ΔL1 means dropping that dimension hurts discrimination more")
    print(" => that dimension carries the most discriminative information.")
    print()
    # Rank
    ranked = sorted(ablation_rows, key=lambda r: r[2])
    print(" Dimension ranking by contribution (most -> least):")
    for i, (d, m, delta, rel) in enumerate(ranked, 1):
        print(f"   {i}. {d:<15} ΔL1={delta:+.4f} ({rel:+.2%})")
    print("=" * 70)

    return {
        'baseline_rows': baseline_rows,
        'ablation_rows': ablation_rows,
        'full_mean_l1': full_mean,
    }


if __name__ == "__main__":
    main()
