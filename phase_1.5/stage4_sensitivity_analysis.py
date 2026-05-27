"""
Stage 4 — Sensitivity Analysis (S4-Stage4-Sensitivity)
Track1_Affect_Trajectory_실험_Plan v1.2 §3.5

Purpose: Robustness check on mapping formulas (Plan §2.4). For each affect's
parameters, vary ±50% and measure how the primary observable (Boundary axis)
changes. Affect parameter sets that are *brittle* (effect collapses under
small variation) need re-design; *robust* parameters can be cited as
empirical mapping commitments.

Scope (Phase 1 only):
  - 7 affects × varied parameters × ±50% × 1 stimulus (welcome × garden)
  - Single stimulus to keep run count manageable; multi-stimulus sensitivity
    is Phase 2 NeurIPS scope.

Parameters varied per affect (Plan §2.4):
  - SEEKING:  type_bias['concept']            (base 0.9, ±50%)
              gamma                           (base 0.525, ±50%)
  - FEAR:     type_bias['sensation']          (base 2.2, ±50%)
              gamma                           (base 0.21, ±50%)
              inhibitory_multiplier           (base 1.5, ±50%)
  - CARE:     type_bias['context']            (base 1.6, ±50%)
              proximity_node_weight           (base 0.5, ±50%)
              context_affect_map_multiplier   (base 1.3, ±50%)
  - RAGE:     type_bias['association']        (base 1.8, ±50%)
              inhibitory_propagation_bonus    (base 0.4, ±50%)
              boundary_crossing_penalty       (base 0.3, ±50%)
  - PANIC:    gamma                           (base 0.28, ±50%)
              value_core_anchor_weight        (base 1.3, ±50%)
              loss_affect_node_bias           (base 0.4, ±50%)
  - PLAY:     gamma                           (base 0.42, ±50%)
              cross_category_edge_bonus       (base 0.2, ±50%)

Variants count: 17 params × 2 directions = 34 variant configs + 7 baselines = ~41 runs.

§11.3 self-flag:
  - "Sensitivity" is a *robustness*  check, not "validation of formula correctness"
  - Findings expressed as "brittle"/"robust" observations, not "proof of mapping"
"""
import sys
import os
import json
import math
import copy
import datetime
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from affect_trajectory_engine import (
    AffectModulatedSAN, AFFECT_CONFIGS, assert_config_invariants,
)
from m_cs_metrics import (
    compute_m_cs_metrics_per_run, VALUE_CORE_LEXICON,
)


SAN_PATH = os.path.join('D:\\', 'Coding', 'Sentient_AI', 'data', 'large_san_8000.json')
STIMULUS = ('welcome', 'garden')
NUMPY_SEED = 42
PRIMARY_DIM = 'Boundary (B)'

DIMS_5D = ['Density (D)', 'Symmetry (S)', 'Centrality (C)', 'Depth (H)', 'Boundary (B)']

OUTPUT_PATH = os.path.join('D:\\', 'Coding', 'Sentient_AI', 'data',
                           'stage4_sensitivity_results.json')

# Parameter variation specification — accessed via dotted path or type_bias key
VARIATIONS = {
    'SEEKING': [
        ('type_bias.concept',                    0.9),
        ('gamma',                                0.525),
    ],
    'FEAR': [
        ('type_bias.sensation',                  2.2),
        ('gamma',                                0.21),
        ('inhibitory_multiplier',                1.5),
    ],
    'CARE': [
        ('type_bias.context',                    1.6),
        ('proximity_node_weight',                0.5),
        ('context_affect_map_multiplier',        1.3),
    ],
    'RAGE': [
        ('type_bias.association',                1.8),
        ('inhibitory_propagation_bonus',         0.4),
        ('boundary_crossing_penalty',            0.3),
    ],
    'PANIC': [
        ('gamma',                                0.28),
        ('value_core_anchor_weight',             1.3),
        ('loss_affect_node_bias',                0.4),
    ],
    'PLAY': [
        ('gamma',                                0.42),
        ('cross_category_edge_bonus',            0.2),
    ],
}


def set_nested(cfg, path, value):
    """Set nested config value by 'foo.bar' path."""
    parts = path.split('.')
    target = cfg
    for p in parts[:-1]:
        target = target[p]
    target[parts[-1]] = value


def run_with_config(cfg):
    """Run one propagation with given full config dict; return 5-D + M-CS measurements."""
    np.random.seed(NUMPY_SEED)
    san = AffectModulatedSAN.load_from_json_with_affect(
        SAN_PATH, cfg, value_core_lexicon=VALUE_CORE_LEXICON,
    )
    seed_w, ctx_w = STIMULUS
    if seed_w not in san.graph:
        return {'error': f"seed '{seed_w}' missing"}
    activation = san.propagate(seed_w, ctx_w)
    topology = san.compute_topological_vector(activation, seed_w)
    active_subgraph, active_nodes = san.extract_active_subgraph(activation)
    _h0, h1_diagram = san.compute_persistent_homology(active_subgraph, activation)
    m_cs = compute_m_cs_metrics_per_run(
        san_graph=san.graph, active_subgraph=active_subgraph,
        active_nodes=active_nodes, activation_map=activation,
        h1_diagram=h1_diagram, value_core_set=set(VALUE_CORE_LEXICON),
    )
    out = {k: float(v) for k, v in topology.items()}
    out.update({k: float(v) if math.isfinite(v) else None for k, v in m_cs.items()})
    return out


def main():
    print('=' * 80)
    print(' STAGE 4 — Sensitivity Analysis (parameter ±50% on welcome × garden)')
    print(' Plan: Track1_Affect_Trajectory_실험_Plan v1.2 §3.5')
    print(f' Primary observable: {PRIMARY_DIM}')
    print(f' Total variants: ~{sum(len(v) for v in VARIATIONS.values()) * 2 + len(VARIATIONS)}')
    print('=' * 80)

    assert_config_invariants()
    print(' Config invariants: PASS\n')

    # Baseline runs (each affect at original config)
    print('--- Baseline runs (original Plan §2.4 configs) ---')
    baseline_per_affect = {}
    for affect in VARIATIONS:
        cfg = copy.deepcopy(AFFECT_CONFIGS[affect])
        r = run_with_config(cfg)
        baseline_per_affect[affect] = r
        if 'error' in r:
            print(f'   {affect}: ERROR {r["error"]}')
        else:
            print(f'   {affect:<10}: ' + ', '.join(f'{k.split()[0]}={r[k]:.3f}'
                                                     for k in DIMS_5D if k in r))
    print()

    # Variants: each param ±50%
    variants_results = {}
    for affect, params in VARIATIONS.items():
        print(f'--- {affect} variants ---')
        variants_results[affect] = {}
        for path, base_val in params:
            for sign, factor in [('+50%', 1.5), ('-50%', 0.5)]:
                new_val = base_val * factor
                cfg = copy.deepcopy(AFFECT_CONFIGS[affect])
                set_nested(cfg, path, new_val)
                r = run_with_config(cfg)
                key = f'{path}_{sign}'
                variants_results[affect][key] = {
                    'param_path': path,
                    'base_value': base_val,
                    'variant_value': new_val,
                    'direction': sign,
                    'result': r,
                }
                base_primary = baseline_per_affect[affect].get(PRIMARY_DIM)
                var_primary = r.get(PRIMARY_DIM) if 'error' not in r else None
                if base_primary is not None and var_primary is not None:
                    delta = var_primary - base_primary
                    rel_pct = (100 * delta / base_primary) if abs(base_primary) > 1e-9 else 0
                    sensitivity_label = (
                        'ROBUST' if abs(rel_pct) < 10 else
                        'MODERATE' if abs(rel_pct) < 30 else
                        'BRITTLE'
                    )
                    print(f'   {path:<40} {sign}: '
                          f'{PRIMARY_DIM.split()[0]}={var_primary:.3f} '
                          f'(Δ={delta:+.3f}, {rel_pct:+.1f}%) [{sensitivity_label}]')
                else:
                    print(f'   {path:<40} {sign}: ERROR or N/A')
        print()

    # Summary: per-parameter robustness category
    print('=' * 80)
    print(' SUMMARY — Parameter Robustness on Primary Observable (Boundary)')
    print('=' * 80)
    sensitivity_summary = {}
    for affect, var_dict in variants_results.items():
        base_b = baseline_per_affect[affect].get(PRIMARY_DIM)
        # Group by param_path
        per_path = {}
        for key, v in var_dict.items():
            p = v['param_path']
            per_path.setdefault(p, []).append(v)
        sensitivity_summary[affect] = {}
        for p, vs in per_path.items():
            rels = []
            for v in vs:
                vp = v['result'].get(PRIMARY_DIM) if 'error' not in v['result'] else None
                if vp is not None and base_b is not None and abs(base_b) > 1e-9:
                    rels.append(abs(100 * (vp - base_b) / base_b))
            max_rel = max(rels) if rels else 0
            label = ('ROBUST' if max_rel < 10 else
                     'MODERATE' if max_rel < 30 else
                     'BRITTLE')
            sensitivity_summary[affect][p] = {
                'max_relative_change_pct': max_rel,
                'label': label,
            }
            print(f'   {affect:<10} {p:<40}: max {max_rel:5.1f}% [{label}]')

    save_data = {
        'plan_version': 'Track1_Affect_Trajectory_실험_Plan v1.2',
        'stage': 'Stage 4 Sensitivity Analysis',
        'date_run': datetime.datetime.now().isoformat(timespec='seconds'),
        'stimulus': {'seed': STIMULUS[0], 'context': STIMULUS[1]},
        'primary_observable': PRIMARY_DIM,
        'numpy_seed': NUMPY_SEED,
        'baseline_per_affect': {a: r for a, r in baseline_per_affect.items()},
        'variants_results': variants_results,
        'sensitivity_summary': sensitivity_summary,
    }
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    print(f'\n Results saved: {OUTPUT_PATH}')
    print(f' Size: {os.path.getsize(OUTPUT_PATH)} bytes')

    return 0


if __name__ == '__main__':
    sys.exit(main())
