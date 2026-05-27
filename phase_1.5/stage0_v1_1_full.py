"""
Stage 0 v1.1 — Affect Trajectory + Damasio Core Self (full 9-D)
Track1_Affect_Trajectory_실험_Plan v1.1 §3.1 + §14.4

Extends Stage 0 Pilot (v1.0) with:
  - CARE condition added (S1-CARE Pilot, F2 + 호철님 권고)
  - M-CS metrics integrated (S1-MCSPostHoc, F2 (a) Damasio core self)
  - Total measurement dim: 5-D topology + 3 M-CS (per-run) + 1 M-CS4 (cross-condition) = 9-D

Stimulus: welcome × garden (single, Phase 1 Pilot scope)
Conditions: baseline_uniform / SEEKING / FEAR / CARE
Seed: numpy=42 (single, Stage 1에서 N=10 확장)

§11.3 표현 강도 검문 self-flag:
  본 결과는 N=1 단일 seed. Stage 1에서 N=10 power analysis 후 statistical claim 가능.
  Gate 1 평가는 mechanical pre-registered thresholds — self-assessment 없음.
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
STIMULUS_SEED = 'welcome'
STIMULUS_CONTEXT = 'garden'
NUMPY_SEED = 42
STAGE0_CONDITIONS = ['baseline_uniform', 'SEEKING', 'FEAR', 'CARE']

DIMS_5D = ['Density (D)', 'Symmetry (S)', 'Centrality (C)', 'Depth (H)', 'Boundary (B)']
MCS_KEYS = ['M-CS1_value_core_proximity', 'M-CS2_self_reference_loop_proxy',
            'M-CS3_corpus_projection_entropy']

GATE1_DELTA_THRESHOLD = 0.1
GATE1_TOP10_DIFF_RATIO = 0.3

OUTPUT_PATH = os.path.join('D:\\', 'Coding', 'Sentient_AI', 'data',
                           'stage0_v1_1_full_results.json')


def run_one_condition(affect_name, san_path=SAN_PATH, seed=NUMPY_SEED):
    """Single propagation + 5-D topology + 3 M-CS metrics."""
    np.random.seed(seed)
    cfg = AFFECT_CONFIGS[affect_name]
    san = AffectModulatedSAN.load_from_json_with_affect(san_path, cfg)

    if STIMULUS_SEED not in san.graph:
        return {'affect': affect_name, 'error': f"seed '{STIMULUS_SEED}' missing"}

    activation = san.propagate(STIMULUS_SEED, STIMULUS_CONTEXT)
    topology = san.compute_topological_vector(activation, STIMULUS_SEED)

    # Active subgraph + H1 for M-CS
    active_subgraph, active_nodes = san.extract_active_subgraph(activation)
    _h0, h1_diagram = san.compute_persistent_homology(active_subgraph, activation)
    m_cs = compute_m_cs_metrics_per_run(
        san_graph=san.graph,
        active_subgraph=active_subgraph,
        active_nodes=active_nodes,
        activation_map=activation,
        h1_diagram=h1_diagram,
        value_core_set=set(VALUE_CORE_LEXICON),
    )

    # NaN/inf check
    all_vals = list(topology.values()) + list(m_cs.values())
    crashed = any(not math.isfinite(v) for v in all_vals)

    # Top 10 active nodes
    active_sorted = sorted(
        [(n, v) for n, v in activation.items() if v > san.theta],
        key=lambda x: x[1], reverse=True,
    )
    top10 = [n for n, _ in active_sorted[:10]]

    # Value-core in active
    vc_active = [n for n in active_nodes if n in set(VALUE_CORE_LEXICON)]

    return {
        'affect': affect_name,
        'config_summary': san.get_config_summary(),
        'context_present': STIMULUS_CONTEXT in san.graph,
        'topology_5d': {k: float(v) for k, v in topology.items()},
        'm_cs_metrics': {k: float(v) if math.isfinite(v) else None for k, v in m_cs.items()},
        'top10_active': top10,
        'top10_activations': [(n, float(v)) for n, v in active_sorted[:10]],
        'n_active': int(len(active_sorted)),
        'value_core_in_active': vc_active,
        'n_value_core_in_active': len(vc_active),
        'crashed_numeric': crashed,
    }


def evaluate_gate1(results):
    """Pre-registered Gate 1 evaluation (5-D thresholds, M-CS reported but not gated)."""
    pair_results = []
    affects = list(results.keys())
    for i in range(len(affects)):
        for j in range(i + 1, len(affects)):
            a, b = affects[i], affects[j]
            ra, rb = results[a], results[b]
            if 'error' in ra or 'error' in rb:
                continue

            deltas_5d = {d: abs(ra['topology_5d'][d] - rb['topology_5d'][d]) for d in DIMS_5D}
            max_dim = max(deltas_5d, key=deltas_5d.get)
            max_delta = deltas_5d[max_dim]

            deltas_mcs = {}
            for k in MCS_KEYS:
                va = ra['m_cs_metrics'].get(k)
                vb = rb['m_cs_metrics'].get(k)
                if va is not None and vb is not None:
                    deltas_mcs[k] = abs(va - vb)
                else:
                    deltas_mcs[k] = None

            sa, sb = set(ra['top10_active']), set(rb['top10_active'])
            union = sa | sb
            overlap = (len(sa & sb) / len(union)) if union else 1.0
            top10_diff = 1.0 - overlap

            pair_results.append({
                'pair': f'{a} vs {b}',
                'deltas_5d': {k: float(v) for k, v in deltas_5d.items()},
                'deltas_m_cs': deltas_mcs,
                'max_delta_5d_dim': max_dim,
                'max_delta_5d_value': float(max_delta),
                'top10_overlap': float(overlap),
                'top10_diff_ratio': float(top10_diff),
                'criterion_a_pass': max_delta > GATE1_DELTA_THRESHOLD,
                'criterion_b_pass': top10_diff >= GATE1_TOP10_DIFF_RATIO,
            })

    any_a = any(p['criterion_a_pass'] for p in pair_results)
    any_b = any(p['criterion_b_pass'] for p in pair_results)
    any_crash = any(r.get('crashed_numeric') or ('error' in r) for r in results.values())

    if any_a and any_b and not any_crash:
        verdict = 'PASS'
    elif (any_a or any_b) and not any_crash:
        verdict = 'PARTIAL'
    else:
        verdict = 'FAIL'

    return {
        'pair_results': pair_results,
        'criterion_a_any_pair_pass': any_a,
        'criterion_b_any_pair_pass': any_b,
        'criterion_c_no_crash': not any_crash,
        'overall_verdict': verdict,
    }


def main():
    print('=' * 72)
    print(' STAGE 0 v1.1 — Affect Trajectory + Damasio Core Self (full 9-D)')
    print(' Plan: Track1_Affect_Trajectory_실험_Plan v1.1 §3.1 + §14.4')
    print(f' Stimulus: {STIMULUS_SEED} × {STIMULUS_CONTEXT}')
    print(f' Conditions: {STAGE0_CONDITIONS}')
    print(f' Numpy seed: {NUMPY_SEED}')
    print('=' * 72)

    assert_config_invariants()
    print(' Config invariant check (4 conditions): PASS\n')

    results = {}
    topology_5d_per_condition = {}
    for affect in STAGE0_CONDITIONS:
        print(f'--- {affect} ---')
        try:
            r = run_one_condition(affect)
            results[affect] = r
            if 'error' in r:
                print(f'   ERROR: {r["error"]}')
            else:
                cfg = r['config_summary']
                print(f'   gamma={cfg["gamma"]:.3f}, inh_mult={cfg["inhibitory_multiplier"]}')
                print(f'   Active nodes: {r["n_active"]} (value-core in active: {r["n_value_core_in_active"]})')
                print(f'   5-D Topology: ' +
                      ', '.join(f'{k.split()[0]}={v:.4f}' for k, v in r['topology_5d'].items()))
                print(f'   M-CS metrics:')
                for k, v in r['m_cs_metrics'].items():
                    short = k.split('_')[0]
                    print(f'      {short}: {v}')
                print(f'   Top 10: {r["top10_active"]}')
                print(f'   Value-core in active: {r["value_core_in_active"]}')
                topology_5d_per_condition[affect] = np.array(list(r['topology_5d'].values()))
        except Exception as e:
            results[affect] = {'affect': affect, 'error': repr(e)}
            print(f'   CRASH: {e!r}')
        print()

    # M-CS4: cross-condition core self stability
    mcs4 = m_cs4_core_self_stability(topology_5d_per_condition)
    print('=' * 72)
    print(f' M-CS4 (cross-condition core self stability, MAD on 5-D): {mcs4:.4f}')
    print('   Lower = more stable core self across affect conditions (Damasio invariance)')
    print('=' * 72)

    # Gate 1 evaluation
    print('\n GATE 1 EVALUATION (5-D pre-registered thresholds, M-CS reported)')
    gate1 = evaluate_gate1(results)

    for pr in gate1['pair_results']:
        print(f"\n {pr['pair']}:")
        for d, v in pr['deltas_5d'].items():
            mark = ' ★' if d == pr['max_delta_5d_dim'] else ''
            print(f'   |Δ {d:<15}| = {v:.4f}{mark}')
        for k, v in pr['deltas_m_cs'].items():
            short = k.split('_')[0]
            print(f'   |Δ {short:<6}| = {v}')
        print(f"   Top-10 diff ratio: {pr['top10_diff_ratio']:.2%}")
        print(f"   (a) max-5D |Δ|>0.1:  {'PASS' if pr['criterion_a_pass'] else 'FAIL'}")
        print(f"   (b) Top10 diff>=30%: {'PASS' if pr['criterion_b_pass'] else 'FAIL'}")

    print('\n' + '-' * 72)
    print(' OVERALL Gate 1 (any pair passing — pre-registered thresholds):')
    print(f"   (a) max-5D |Δ| > 0.1:    {'PASS' if gate1['criterion_a_any_pair_pass'] else 'FAIL'}")
    print(f"   (b) Top-10 diff >= 30%:  {'PASS' if gate1['criterion_b_any_pair_pass'] else 'FAIL'}")
    print(f"   (c) No crashes/NaN:      {'PASS' if gate1['criterion_c_no_crash'] else 'FAIL'}")
    print(f"   VERDICT: {gate1['overall_verdict']}")
    print('-' * 72)

    # Save
    save_data = {
        'plan_version': 'Track1_Affect_Trajectory_실험_Plan v1.1',
        'stage': 'Stage 0 v1.1 (CARE added, M-CS metrics integrated)',
        'date_run': datetime.datetime.now().isoformat(timespec='seconds'),
        'stimulus': {'seed': STIMULUS_SEED, 'context': STIMULUS_CONTEXT},
        'numpy_seed': NUMPY_SEED,
        'san_path': SAN_PATH,
        'conditions': STAGE0_CONDITIONS,
        'value_core_lexicon_size': len(VALUE_CORE_LEXICON),
        'gate1_thresholds_preregistered': {
            'a_max_5d_delta': GATE1_DELTA_THRESHOLD,
            'b_top10_diff_ratio': GATE1_TOP10_DIFF_RATIO,
        },
        'results_per_condition': results,
        'm_cs4_cross_condition_stability': float(mcs4) if math.isfinite(mcs4) else None,
        'gate1_evaluation': gate1,
    }
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    print(f'\n Results saved: {OUTPUT_PATH}')
    print(f' Size: {os.path.getsize(OUTPUT_PATH)} bytes')

    return gate1['overall_verdict']


if __name__ == '__main__':
    verdict = main()
    sys.exit({'PASS': 0, 'PARTIAL': 1, 'FAIL': 2}[verdict])
