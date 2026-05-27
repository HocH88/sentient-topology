"""
Stage 0 Pilot — Affect Trajectory Differentiation
Track1_Affect_Trajectory_실험_Plan v1.0 §3.1

Minimum-cost feasibility check: do 3 affect conditions (baseline/SEEKING/FEAR)
produce distinguishable topology signatures on a single stimulus?

Stimulus: welcome × garden (proven working on 198-book 8000-node SAN per
Track1_연구노트 2026-05-26 entry: D=0.250, S=0.319, C=0.196, H=2.0, B=0.429).

Gate 1 criteria (auto-evaluated, NO self-assessment shortcuts per §9):
  (a) At least 1 dim with |Δ| > 0.1 between SOME pair of conditions
  (b) Top-10 active vocab differs ≥ 30% between SOME pair
  (c) No crashes / NaN / infinity

Outputs:
  - Console summary
  - D:/Coding/Sentient_AI/data/stage0_pilot_results.json (replaces deleted
    statistical_rigor_results.json placeholder)
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


# === Stage 0 Configuration (from data/affect_trajectory_config.yaml stage0_pilot) ===
SAN_PATH = os.path.join('D:\\', 'Coding', 'Sentient_AI', 'data', 'large_san_8000.json')
STIMULUS_SEED = 'welcome'
STIMULUS_CONTEXT = 'garden'
NUMPY_SEED = 42
STAGE0_CONDITIONS = ['baseline_uniform', 'SEEKING', 'FEAR']

DIMS = ['Density (D)', 'Symmetry (S)', 'Centrality (C)', 'Depth (H)', 'Boundary (B)']

GATE1_DELTA_THRESHOLD = 0.1      # criterion (a)
GATE1_TOP10_DIFF_RATIO = 0.3     # criterion (b)

OUTPUT_PATH = os.path.join('D:\\', 'Coding', 'Sentient_AI', 'data', 'stage0_pilot_results.json')


def run_one_condition(affect_name, san_path=SAN_PATH, seed=NUMPY_SEED):
    """Execute propagate for one affect condition and return measurements.

    Reproducibility: numpy.random is seeded before each run for determinism.
    """
    np.random.seed(seed)
    cfg = AFFECT_CONFIGS[affect_name]
    san = AffectModulatedSAN.load_from_json_with_affect(san_path, cfg)

    # Validate stimulus
    if STIMULUS_SEED not in san.graph:
        return {'affect': affect_name, 'error': f"seed '{STIMULUS_SEED}' missing"}
    context_present = STIMULUS_CONTEXT in san.graph

    # Propagate
    activation = san.propagate(STIMULUS_SEED, STIMULUS_CONTEXT)

    # 5-D topology
    topology = san.compute_topological_vector(activation, STIMULUS_SEED)

    # NaN/inf check (Gate criterion c)
    crashed = False
    for dim_name, val in topology.items():
        if not math.isfinite(val):
            crashed = True
            break

    # Top 10 active nodes
    active_sorted = sorted(
        [(n, v) for n, v in activation.items() if v > san.theta],
        key=lambda x: x[1], reverse=True,
    )
    top10 = [n for n, _ in active_sorted[:10]]
    n_active = len(active_sorted)

    return {
        'affect': affect_name,
        'config_summary': san.get_config_summary(),
        'context_present': context_present,
        'topology': {k: float(v) for k, v in topology.items()},
        'top10_active': top10,
        'top10_activations': [(n, float(v)) for n, v in active_sorted[:10]],
        'n_active': int(n_active),
        'crashed_numeric': crashed,
    }


def evaluate_gate1(results):
    """Pairwise Gate 1 evaluation. Returns list of pair results and overall verdict.

    §9 정합: This is mechanical evaluation against pre-registered thresholds.
    No self-assessment, no human override of the criteria.
    """
    pair_results = []
    affects = list(results.keys())
    for i in range(len(affects)):
        for j in range(i + 1, len(affects)):
            a, b = affects[i], affects[j]
            ra, rb = results[a], results[b]
            if 'error' in ra or 'error' in rb:
                continue

            # Per-dim deltas
            deltas = {d: abs(ra['topology'][d] - rb['topology'][d]) for d in DIMS}
            max_dim = max(deltas, key=deltas.get)
            max_delta = deltas[max_dim]

            # Top-10 difference (Jaccard complement)
            sa, sb = set(ra['top10_active']), set(rb['top10_active'])
            union = sa | sb
            overlap = (len(sa & sb) / len(union)) if union else 1.0
            top10_diff_ratio = 1.0 - overlap

            pair_results.append({
                'pair': f'{a} vs {b}',
                'deltas_per_dim': {k: float(v) for k, v in deltas.items()},
                'max_delta_dim': max_dim,
                'max_delta_value': float(max_delta),
                'top10_overlap': float(overlap),
                'top10_diff_ratio': float(top10_diff_ratio),
                'criterion_a_pass': max_delta > GATE1_DELTA_THRESHOLD,
                'criterion_b_pass': top10_diff_ratio >= GATE1_TOP10_DIFF_RATIO,
            })

    # Overall: ANY pair passing criterion → criterion overall PASS
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
    print(' STAGE 0 PILOT — Affect Trajectory Differentiation')
    print(' Plan: Track1_Affect_Trajectory_실험_Plan v1.0 §3.1')
    print(f' Stimulus: {STIMULUS_SEED} × {STIMULUS_CONTEXT}')
    print(f' Conditions: {STAGE0_CONDITIONS}')
    print(f' Numpy seed: {NUMPY_SEED}')
    print(f' SAN path: {SAN_PATH}')
    print('=' * 72)

    # Config invariant check (§9 — drift detection)
    assert_config_invariants()
    print(' Config invariant check: PASS\n')

    # Execute each condition
    results = {}
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
                print(f'   Active nodes: {r["n_active"]}')
                print(f'   Topology: ' +
                      ', '.join(f'{k.split()[0]}={v:.4f}' for k, v in r['topology'].items()))
                print(f'   Top 10: {r["top10_active"]}')
        except Exception as e:
            results[affect] = {'affect': affect, 'error': repr(e)}
            print(f'   CRASH: {e!r}')
        print()

    # Gate 1 evaluation (mechanical, pre-registered thresholds)
    print('=' * 72)
    print(' GATE 1 EVALUATION (Plan §3.1, thresholds pre-registered)')
    print('=' * 72)
    gate1 = evaluate_gate1(results)

    for pr in gate1['pair_results']:
        print(f"\n {pr['pair']}:")
        for d, v in pr['deltas_per_dim'].items():
            mark = ' ★' if d == pr['max_delta_dim'] else ''
            print(f'   |Δ {d:<15}| = {v:.4f}{mark}')
        print(f"   Max-delta dim: {pr['max_delta_dim']} ({pr['max_delta_value']:.4f})")
        print(f"   Top-10 overlap: {pr['top10_overlap']:.2%}, diff ratio: {pr['top10_diff_ratio']:.2%}")
        print(f"   Criterion (a) |Δ|>{GATE1_DELTA_THRESHOLD}:           "
              f"{'PASS' if pr['criterion_a_pass'] else 'FAIL'}")
        print(f"   Criterion (b) Top10 diff>={GATE1_TOP10_DIFF_RATIO:.0%}:      "
              f"{'PASS' if pr['criterion_b_pass'] else 'FAIL'}")

    print('\n' + '-' * 72)
    print(' OVERALL Gate 1 (any pair passing criteria):')
    print(f"   (a) max |Δ| > {GATE1_DELTA_THRESHOLD}:        "
          f"{'PASS' if gate1['criterion_a_any_pair_pass'] else 'FAIL'}")
    print(f"   (b) Top-10 diff >= {GATE1_TOP10_DIFF_RATIO:.0%}:  "
          f"{'PASS' if gate1['criterion_b_any_pair_pass'] else 'FAIL'}")
    print(f"   (c) No crashes/NaN:    "
          f"{'PASS' if gate1['criterion_c_no_crash'] else 'FAIL'}")
    print(f"   VERDICT: {gate1['overall_verdict']}")
    print('-' * 72)

    # Persist results
    save_data = {
        'plan_version': 'Track1_Affect_Trajectory_실험_Plan v1.0',
        'stage': 'Stage 0 Pilot',
        'date_run': datetime.datetime.now().isoformat(timespec='seconds'),
        'stimulus': {'seed': STIMULUS_SEED, 'context': STIMULUS_CONTEXT},
        'numpy_seed': NUMPY_SEED,
        'san_path': SAN_PATH,
        'conditions': STAGE0_CONDITIONS,
        'gate1_thresholds': {
            'a_max_delta': GATE1_DELTA_THRESHOLD,
            'b_top10_diff_ratio': GATE1_TOP10_DIFF_RATIO,
        },
        'results_per_condition': results,
        'gate1_evaluation': gate1,
    }
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    print(f'\n Results saved: {OUTPUT_PATH}')
    print(f' Size: {os.path.getsize(OUTPUT_PATH)} bytes')

    return gate1['overall_verdict']


if __name__ == '__main__':
    verdict = main()
    # Exit code: 0 if PASS, 1 if PARTIAL, 2 if FAIL — for CI integration
    sys.exit({'PASS': 0, 'PARTIAL': 1, 'FAIL': 2}[verdict])
