"""
Stage 2 Phase 1 Main — 7-affect × 6-stimulus Full Matrix
Track1_Affect_Trajectory_실험_Plan v1.2 §3.3 + §4.1 (S4-Stage2-Form (B) commit, 2026-05-27)

Background:
  Stage 1 v4 (7-affect × 3 stimuli × N=1, all Stage 1+ ext active) showed:
    - 36/63 differentiation cells (57%)
    - Boundary axis as primary observable (12 cross-stimulus consistent pairs)
    - SEEKING/CARE/PLAY: expansion / RAGE/FEAR/PANIC: constriction
  → Stage 2 expands to 6 stimuli for stronger cross-stimulus claim.

§11.3 self-flag:
  - Deterministic SAN: N=1 sufficient (N=10 same-stimulus would yield identical results)
  - Multi-stimulus expansion gives variance that random seeds cannot
  - Statistical claim: Bonferroni-corrected per-dim consistency rate (not power analysis)

Stimuli (6, from paper teaser + Stage 1 set):
  1. welcome × garden     (Stage 1 baseline)
  2. sunset × war         (paper teaser, intense affect)
  3. sunset × garden      (paper teaser counter — 'sensation×context' contrast)
  4. love × garden        (value-core seed)
  5. sorrow × funeral     (paper teaser, mourning context)
  6. feel × book          (paper teaser, abstract seed)
"""
import sys
import os
import json
import math
import datetime
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from affect_trajectory_engine import (
    AffectModulatedSAN, AFFECT_CONFIGS, assert_config_invariants,
)
from m_cs_metrics import (
    compute_m_cs_metrics_per_run, m_cs4_core_self_stability, VALUE_CORE_LEXICON,
)


SAN_PATH = os.path.join('D:\\', 'Coding', 'Sentient_AI', 'data', 'large_san_8000.json')
CONDITIONS = ['baseline_uniform', 'SEEKING', 'FEAR', 'CARE', 'RAGE', 'PANIC', 'PLAY']
NUMPY_SEED = 42

# 6 stimuli (Plan v1.2 §4.1)
STIMULI = [
    ('welcome', 'garden'),
    ('sunset',  'war'),
    ('sunset',  'garden'),
    ('love',    'garden'),
    ('sorrow',  'funeral'),
    ('feel',    'book'),
]

DIMS_5D = ['Density (D)', 'Symmetry (S)', 'Centrality (C)', 'Depth (H)', 'Boundary (B)']
MCS_KEYS = ['M-CS1_value_core_proximity', 'M-CS2_self_reference_loop_proxy',
            'M-CS3_corpus_projection_entropy']
ALL_DIMS = DIMS_5D + MCS_KEYS

OUTPUT_PATH = os.path.join('D:\\', 'Coding', 'Sentient_AI', 'data',
                           'stage2_main_phase1_full_matrix_results.json')


def run_one(affect_name, seed_word, context_word, san_cache=None):
    np.random.seed(NUMPY_SEED)
    cfg = AFFECT_CONFIGS[affect_name]
    if san_cache is None or san_cache.get('affect') != affect_name:
        san = AffectModulatedSAN.load_from_json_with_affect(
            SAN_PATH, cfg, value_core_lexicon=VALUE_CORE_LEXICON,
        )
        if san_cache is not None:
            san_cache['affect'] = affect_name
            san_cache['san'] = san
    else:
        san = san_cache['san']

    if seed_word not in san.graph:
        return {'error': f"seed '{seed_word}' missing"}

    activation = san.propagate(seed_word, context_word)
    topology = san.compute_topological_vector(activation, seed_word)
    active_subgraph, active_nodes = san.extract_active_subgraph(activation)
    _h0, h1_diagram = san.compute_persistent_homology(active_subgraph, activation)
    m_cs = compute_m_cs_metrics_per_run(
        san_graph=san.graph, active_subgraph=active_subgraph,
        active_nodes=active_nodes, activation_map=activation,
        h1_diagram=h1_diagram, value_core_set=set(VALUE_CORE_LEXICON),
    )

    measurements = {}
    measurements.update({k: float(v) for k, v in topology.items()})
    measurements.update({k: float(v) if math.isfinite(v) else float('nan')
                          for k, v in m_cs.items()})

    crashed = any(not math.isfinite(v) for v in measurements.values())

    active_sorted = sorted(
        [(n, v) for n, v in activation.items() if v > san.theta],
        key=lambda x: x[1], reverse=True,
    )
    top10 = [n for n, _ in active_sorted[:10]]
    vc_active = [n for n in active_nodes if n in set(VALUE_CORE_LEXICON)]

    return {
        'measurements_9d': measurements,
        'top10_active': top10,
        'n_active': len(active_sorted),
        'n_value_core_in_active': len(vc_active),
        'context_present': context_word in san.graph,
        'crashed': crashed,
    }


def main():
    print('=' * 80)
    print(' STAGE 2 PHASE 1 MAIN — 7-Affect × 6-Stimulus Full Matrix')
    print(' Plan: Track1_Affect_Trajectory_실험_Plan v1.2 §3.3 + §4.1')
    print(f' Conditions ({len(CONDITIONS)}): {CONDITIONS}')
    print(f' Stimuli ({len(STIMULI)}):')
    for s, c in STIMULI:
        print(f'   - {s} × {c}')
    print(f' Total runs: {len(CONDITIONS)} × {len(STIMULI)} = {len(CONDITIONS) * len(STIMULI)}')
    print('=' * 80)

    assert_config_invariants()
    print(' Config invariants: PASS\n')

    results = {}
    for affect in CONDITIONS:
        print(f'### {affect}')
        san_cache = {'affect': None, 'san': None}
        for s_idx, (sw, cw) in enumerate(STIMULI):
            try:
                r = run_one(affect, sw, cw, san_cache=san_cache)
                results[(affect, s_idx)] = r
                if 'error' in r:
                    print(f'   {sw} × {cw}: ERROR {r["error"]}')
                else:
                    m = r['measurements_9d']
                    print(f'   {sw:<8} × {cw:<8}: ' +
                          ', '.join(f'{k.split()[0]}={m[k]:.3f}' for k in DIMS_5D) +
                          f', active={r["n_active"]}, vc-active={r["n_value_core_in_active"]}')
            except Exception as e:
                results[(affect, s_idx)] = {'error': repr(e)}
                print(f'   {sw} × {cw} CRASH: {e!r}')
        print()

    # === Analysis 1: per-dim cross-stimulus consistency (Bonferroni-style) ===
    print('=' * 80)
    print(' ANALYSIS 1 — Cross-Stimulus Consistency per (dim, pair)')
    print(' (Direction agreement across 6 stimuli, requires >=4 non-zero meaningful diffs)')
    print('=' * 80)
    consistency = {}
    n_pairs = len(CONDITIONS) * (len(CONDITIONS) - 1) // 2
    n_dims = len(ALL_DIMS)
    n_total = n_pairs * n_dims  # for Bonferroni reference

    consistent_count = {d: 0 for d in ALL_DIMS}
    for d in ALL_DIMS:
        consistency[d] = {}
        for i in range(len(CONDITIONS)):
            for j in range(i + 1, len(CONDITIONS)):
                a, b = CONDITIONS[i], CONDITIONS[j]
                signs = []
                for s_idx in range(len(STIMULI)):
                    ra = results.get((a, s_idx))
                    rb = results.get((b, s_idx))
                    if not ra or not rb or 'error' in ra or 'error' in rb:
                        signs.append(None)
                        continue
                    diff = ra['measurements_9d'][d] - rb['measurements_9d'][d]
                    if abs(diff) < 1e-6:
                        signs.append(0)
                    else:
                        signs.append(1 if diff > 0 else -1)
                meaningful = [s for s in signs if s is not None and s != 0]
                # Stricter consistency: ≥4 meaningful AND all same sign
                if len(meaningful) >= 4:
                    consistent = len(set(meaningful)) == 1
                elif len(meaningful) >= 2:
                    consistent = len(set(meaningful)) == 1  # partial
                else:
                    consistent = None
                consistency[d][f'{a} vs {b}'] = {
                    'signs_per_stimulus': signs,
                    'n_meaningful': len(meaningful),
                    'consistent_direction': consistent,
                }
                if consistent is True and len(meaningful) >= 4:
                    consistent_count[d] += 1

    print('\n Strict cross-stimulus consistency count (>=4 meaningful diffs, all same direction):')
    for d in ALL_DIMS:
        d_short = d.split()[0] if ' ' in d else d.split('_')[0]
        print(f'   {d_short:<25}: {consistent_count[d]:>2} / {n_pairs} pairs')
    primary_dim = max(consistent_count, key=consistent_count.get)
    print(f"\n PRIMARY OBSERVABLE: {primary_dim} ({consistent_count[primary_dim]} pairs)")

    # === Analysis 2: affect ordering on primary observable ===
    print('\n' + '=' * 80)
    print(f' ANALYSIS 2 — Affect Ordering on PRIMARY ({primary_dim})')
    print('=' * 80)
    affect_means = {}
    for affect in CONDITIONS:
        vals = [results.get((affect, s_idx), {}).get('measurements_9d', {}).get(primary_dim)
                for s_idx in range(len(STIMULI))]
        finite = [v for v in vals if v is not None and math.isfinite(v)]
        if finite:
            affect_means[affect] = float(np.mean(finite))
        else:
            affect_means[affect] = None

    ordering = sorted([(a, v) for a, v in affect_means.items() if v is not None],
                      key=lambda x: x[1])
    print(f' Affect ordering by mean {primary_dim} across {len(STIMULI)} stimuli (ascending):')
    for rank, (a, v) in enumerate(ordering, 1):
        print(f'   {rank}. {a:<20}: {v:.4f}')

    # === Analysis 3: per-stimulus pairwise hit counts ===
    print('\n' + '=' * 80)
    print(' ANALYSIS 3 — Per-Stimulus Differentiation Counts')
    print('=' * 80)
    per_stim_hits = []
    for s_idx, (sw, cw) in enumerate(STIMULI):
        hits = 0
        for i in range(len(CONDITIONS)):
            for j in range(i + 1, len(CONDITIONS)):
                a, b = CONDITIONS[i], CONDITIONS[j]
                ra = results.get((a, s_idx))
                rb = results.get((b, s_idx))
                if not ra or not rb or 'error' in ra or 'error' in rb:
                    continue
                ma, mb = ra['measurements_9d'], rb['measurements_9d']
                max_delta = max(abs(ma[d] - mb[d]) for d in DIMS_5D)
                if max_delta > 0.1:
                    hits += 1
        per_stim_hits.append((f'{sw} × {cw}', hits, n_pairs))
        print(f"   {sw:<10} × {cw:<10}: {hits:>2} / {n_pairs} pairs differentiated (max-5D Δ > 0.1)")

    # === Verdict ===
    total_diff = sum(h[1] for h in per_stim_hits)
    total_cells = n_pairs * len(STIMULI)
    print('\n' + '=' * 80)
    print(' VERDICT — Stage 2 Phase 1 Main')
    print('=' * 80)
    print(f' Differentiation rate: {total_diff} / {total_cells} ({100*total_diff/total_cells:.1f}%)')
    print(f' Primary observable: {primary_dim} ({consistent_count[primary_dim]} consistent pairs)')

    # Save
    results_serial = {f'{aff}|{idx}': v for (aff, idx), v in results.items()}
    save_data = {
        'plan_version': 'Track1_Affect_Trajectory_실험_Plan v1.2',
        'stage': 'Stage 2 Phase 1 Main (S4-Stage2-Form (B))',
        'date_run': datetime.datetime.now().isoformat(timespec='seconds'),
        'conditions': CONDITIONS,
        'stimuli': [{'seed': s, 'context': c} for s, c in STIMULI],
        'numpy_seed': NUMPY_SEED,
        'san_path': SAN_PATH,
        'total_runs': len(CONDITIONS) * len(STIMULI),
        'results_per_condition_stimulus': results_serial,
        'cross_stimulus_consistency': consistency,
        'consistent_pair_count_per_dim': consistent_count,
        'primary_observable_dim': primary_dim,
        'affect_means_on_primary': affect_means,
        'affect_ordering_on_primary': ordering,
        'per_stimulus_differentiation_counts': per_stim_hits,
        'total_differentiation_rate': total_diff / total_cells,
    }
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    print(f'\n Results saved: {OUTPUT_PATH}')
    print(f' Size: {os.path.getsize(OUTPUT_PATH)} bytes')

    return 0


if __name__ == '__main__':
    sys.exit(main())
