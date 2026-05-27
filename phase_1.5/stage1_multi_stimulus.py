"""
Stage 1 (Redesigned) — Multi-Stimulus Affect Trajectory Test
Track1_Affect_Trajectory_실험_Plan v1.1 §3.2 + §3.3 (재설계, 2026-05-26 밤)

Background:
  Stage 0 v1.1 revealed:
    (i) CARE config produced IDENTICAL results to baseline on (welcome × garden)
    (ii) Deterministic SAN → N=10 same-stimulus seeds yield identical results
  → Plan §3.2 N=10 same-stimulus power analysis would be uninformative.

S2-Redesign (호철님 컨펌, 2026-05-26 밤):
  Stage 1 redesigned as 4 conditions × 3 stimuli × N=1 (deterministic).
  Tests whether:
    (A) CARE produces differentiation in OTHER stimuli (context-rich)
    (B) Affect patterns are CROSS-STIMULUS CONSISTENT (vs stimulus-specific noise)
    (C) Damasio M-CS metrics generalize across stimuli

Stimuli (Plan §3.3 stage2_main_phase1):
  - welcome × garden  (Stage 0 baseline)
  - sunset × war      (paper teaser, high-affect contrast)
  - love × garden     (value-core seed)

§11.3 self-flag:
  - "Power analysis" 단어 미사용 (N=1로 power 무의미)
  - "Stimulus-conditional differentiation analysis"가 정확
  - Cross-stimulus pattern consistency를 *suggestive* 표현
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
# Updated 2026-05-27 per S2-7Affect: 7-affect (Panksepp, LUST excluded per Plan §2.2)
CONDITIONS = ['baseline_uniform', 'SEEKING', 'FEAR', 'CARE', 'RAGE', 'PANIC', 'PLAY']
NUMPY_SEED = 42

# Plan §3.3 stage2_main_phase1.stimuli (3 stimuli)
STIMULI = [
    ('welcome', 'garden'),
    ('sunset',  'war'),
    ('love',    'garden'),
]

DIMS_5D = ['Density (D)', 'Symmetry (S)', 'Centrality (C)', 'Depth (H)', 'Boundary (B)']
MCS_KEYS = ['M-CS1_value_core_proximity', 'M-CS2_self_reference_loop_proxy',
            'M-CS3_corpus_projection_entropy']
ALL_DIMS = DIMS_5D + MCS_KEYS

OUTPUT_PATH = os.path.join('D:\\', 'Coding', 'Sentient_AI', 'data',
                           'stage1_multi_stimulus_v4_7affect_ext_results.json')


def run_one(affect_name, seed_word, context_word, san_cache=None):
    """One propagation run with reuse of SAN if same affect.

    For CARE (Stage 1+ ext): value_core_lexicon passed at load to enable proximity precompute.
    """
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
    context_present = context_word in san.graph

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
        'value_core_in_active': vc_active,
        'n_value_core_in_active': len(vc_active),
        'context_present': context_present,
        'crashed': crashed,
    }


def main():
    print('=' * 78)
    print(' STAGE 1 (Redesigned) — Multi-Stimulus Affect Trajectory Test')
    print(' Plan: Track1_Affect_Trajectory_실험_Plan v1.1 §3.2 + §3.3')
    print(f' Conditions: {CONDITIONS}')
    print(f' Stimuli: {STIMULI}')
    print(f' Seed: numpy={NUMPY_SEED} (deterministic, single-shot)')
    print(f' Total runs: {len(CONDITIONS)} × {len(STIMULI)} = {len(CONDITIONS) * len(STIMULI)}')
    print('=' * 78)

    assert_config_invariants()
    print(' Config invariants: PASS\n')

    # results[(affect, stimulus_idx)] = run_data
    results = {}
    for affect in CONDITIONS:
        print(f'### {affect}')
        san_cache = {'affect': None, 'san': None}
        for s_idx, (seed_w, ctx_w) in enumerate(STIMULI):
            stim_label = f'{seed_w} × {ctx_w}'
            print(f'  --- stimulus: {stim_label} ---')
            try:
                r = run_one(affect, seed_w, ctx_w, san_cache=san_cache)
                results[(affect, s_idx)] = r
                if 'error' in r:
                    print(f'     ERROR: {r["error"]}')
                else:
                    m = r['measurements_9d']
                    print(f'     5-D:  ' +
                          ', '.join(f'{k.split()[0]}={m[k]:.3f}' for k in DIMS_5D))
                    print(f'     M-CS: ' +
                          ', '.join(f'{k.split("_")[0]}={m[k]:.3f}' if math.isfinite(m[k]) else f'{k.split("_")[0]}=nan'
                                     for k in MCS_KEYS))
                    print(f'     Active: {r["n_active"]} (vc-active: {r["n_value_core_in_active"]})')
                    print(f'     Top 10: {r["top10_active"]}')
            except Exception as e:
                results[(affect, s_idx)] = {'error': repr(e)}
                print(f'     CRASH: {e!r}')
        print()

    # === Analysis 1: Per-stimulus pairwise differentiation ===
    print('=' * 78)
    print(' ANALYSIS 1 — Per-Stimulus Pairwise Differentiation (5-D)')
    print('=' * 78)
    per_stim_differentiation = {}
    for s_idx, (sw, cw) in enumerate(STIMULI):
        stim_label = f'{sw} × {cw}'
        print(f'\n Stimulus: {stim_label}')
        per_stim_differentiation[stim_label] = {}
        for i in range(len(CONDITIONS)):
            for j in range(i + 1, len(CONDITIONS)):
                a, b = CONDITIONS[i], CONDITIONS[j]
                ra = results.get((a, s_idx))
                rb = results.get((b, s_idx))
                if not ra or not rb or 'error' in ra or 'error' in rb:
                    continue
                ma = ra['measurements_9d']
                mb = rb['measurements_9d']
                deltas = {d: abs(ma[d] - mb[d]) for d in ALL_DIMS}
                max_dim_5d = max(DIMS_5D, key=lambda d: deltas[d])
                max_delta_5d = deltas[max_dim_5d]
                top10_diff = 1.0 - (
                    len(set(ra['top10_active']) & set(rb['top10_active'])) /
                    max(len(set(ra['top10_active']) | set(rb['top10_active'])), 1)
                )
                per_stim_differentiation[stim_label][f'{a} vs {b}'] = {
                    'deltas_all_9d': {k: float(v) for k, v in deltas.items()},
                    'max_5d_dim': max_dim_5d,
                    'max_5d_delta': float(max_delta_5d),
                    'top10_diff_ratio': float(top10_diff),
                    'differentiation_passed': max_delta_5d > 0.1 or top10_diff >= 0.3,
                }
                marker = ' ★' if max_delta_5d > 0.1 else ''
                print(f'   {a:<18} vs {b:<18}: max-5D Δ={max_delta_5d:.3f} ({max_dim_5d.split()[0]}), '
                      f'top10-diff={top10_diff:.1%}{marker}')

    # === Analysis 2: CARE differentiation across stimuli (key Stage 0 finding) ===
    print('\n' + '=' * 78)
    print(' ANALYSIS 2 — CARE vs baseline: Differentiation Across Stimuli (key check)')
    print('=' * 78)
    care_check = {}
    for s_idx, (sw, cw) in enumerate(STIMULI):
        stim_label = f'{sw} × {cw}'
        ra = results.get(('baseline_uniform', s_idx))
        rb = results.get(('CARE', s_idx))
        if not ra or not rb or 'error' in ra or 'error' in rb:
            continue
        ma = ra['measurements_9d']
        mb = rb['measurements_9d']
        deltas_5d = {d: abs(ma[d] - mb[d]) for d in DIMS_5D}
        all_zero = all(v < 1e-9 for v in deltas_5d.values())
        max_delta = max(deltas_5d.values())
        max_dim = max(deltas_5d, key=deltas_5d.get)
        care_check[stim_label] = {
            'CARE_equals_baseline': all_zero,
            'max_5d_delta': float(max_delta),
            'max_5d_dim': max_dim,
        }
        status = 'IDENTICAL (no CARE effect)' if all_zero else f'DIFFERENTIATED (max Δ={max_delta:.3f} on {max_dim.split()[0]})'
        print(f'   {stim_label:<20}: {status}')

    care_effective_stimuli = [s for s, c in care_check.items() if not c['CARE_equals_baseline']]
    print(f'\n CARE shows differentiation in: {care_effective_stimuli or "(none — CARE minimum config insufficient)"}')

    # === Analysis 3: Cross-stimulus consistency of affect patterns ===
    print('\n' + '=' * 78)
    print(' ANALYSIS 3 — Cross-Stimulus Consistency of Affect Patterns')
    print(' (Same affect-pair direction across stimuli?)')
    print('=' * 78)
    consistency = {}
    for d in DIMS_5D:
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
                    # S3-Sign-Code-Fix: relaxed threshold for floating-point noise
                    if abs(diff) < 1e-6:
                        signs.append(0)
                    else:
                        signs.append(1 if diff > 0 else -1)
                # Count agreement: all same sign (excluding 0 / None)
                meaningful = [s for s in signs if s is not None and s != 0]
                if len(meaningful) >= 2:
                    consistent = len(set(meaningful)) == 1
                else:
                    consistent = None
                consistency[d][f'{a} vs {b}'] = {
                    'signs_per_stimulus': signs,
                    'consistent_direction': consistent,
                }

    print('\n Direction consistency per (dim, pair) — *True* = same sign across all stimuli with non-zero diff:')
    for d in DIMS_5D:
        print(f'\n  {d}:')
        for pair, c in consistency[d].items():
            sig_str = '/'.join(['+' if s == 1 else '-' if s == -1 else '0' if s == 0 else '?'
                                for s in c['signs_per_stimulus']])
            print(f'    {pair:<40}: signs=[{sig_str}], consistent={c["consistent_direction"]}')

    # === Verdict ===
    print('\n' + '=' * 78)
    print(' VERDICT — Stage 1 Multi-Stimulus')
    print('=' * 78)
    # How many (stimulus, pair) cells show differentiation (max Δ > 0.1 OR top10 diff ≥ 30%)
    total_cells = 0
    differentiated_cells = 0
    for stim_label, pairs in per_stim_differentiation.items():
        for p_label, p_data in pairs.items():
            total_cells += 1
            if p_data['differentiation_passed']:
                differentiated_cells += 1
    pct = (differentiated_cells / total_cells * 100) if total_cells else 0
    print(f' Differentiation cells: {differentiated_cells} / {total_cells} ({pct:.1f}%)')
    print(f' CARE effective stimuli: {len(care_effective_stimuli)} / {len(STIMULI)}')

    # Save
    # Serialize tuple keys as strings
    results_serial = {f'{aff}|{idx}': v for (aff, idx), v in results.items()}
    save_data = {
        'plan_version': 'Track1_Affect_Trajectory_실험_Plan v1.1',
        'stage': 'Stage 1 Multi-Stimulus (S2-Redesign, 2026-05-26 밤)',
        'date_run': datetime.datetime.now().isoformat(timespec='seconds'),
        'conditions': CONDITIONS,
        'stimuli': [{'seed': s, 'context': c} for s, c in STIMULI],
        'numpy_seed': NUMPY_SEED,
        'san_path': SAN_PATH,
        'value_core_lexicon_size': len(VALUE_CORE_LEXICON),
        'total_runs': len(CONDITIONS) * len(STIMULI),
        'results_per_condition_stimulus': results_serial,
        'analysis_1_per_stimulus_differentiation': per_stim_differentiation,
        'analysis_2_CARE_vs_baseline_per_stimulus': care_check,
        'analysis_3_cross_stimulus_consistency': consistency,
        'CARE_effective_stimuli': care_effective_stimuli,
        'differentiation_cells': differentiated_cells,
        'total_cells': total_cells,
    }
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    print(f'\n Results saved: {OUTPUT_PATH}')
    print(f' Size: {os.path.getsize(OUTPUT_PATH)} bytes')

    return 0


if __name__ == '__main__':
    sys.exit(main())
