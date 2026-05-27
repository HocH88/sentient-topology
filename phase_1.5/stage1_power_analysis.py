"""
Stage 1 — Power Analysis (9-D, 4 conditions, N=10 seeds)
Track1_Affect_Trajectory_실험_Plan v1.1 §3.2

Purpose: With N=10 numpy seeds per condition, estimate effect sizes
(Cohen's d) and bootstrap 95% CIs for each (condition_pair, dimension).
Determine whether N=10 is sufficient for Stage 2 main, or N must be increased.

Conditions: baseline_uniform / SEEKING / FEAR / CARE
Stimulus: welcome × garden (single, fixed)
Seeds: pre-registered numpy_seeds_main from data/affect_trajectory_config.yaml
Total runs: 4 conditions × 10 seeds = 40

Note: Random seed affects edge ordering during JSON load (no order in dict),
but for deterministic propagate(), seed is set BEFORE each run. Per-condition
variability in this setup mainly comes from any stochastic element in
SAN_engine; current implementation is deterministic given the graph, so
"N=10 seeds" tests reproducibility + future stochastic element robustness.
This is still informative for §11.3 power analysis (variance bound check).

§11.3 표현 강도 검문:
  - "Power analysis"라는 강한 단어 — 단순 N=10 effect size 계산.
  - "Statistical significance"는 단일 stimulus N=10이라 *주장 안 함*.
  - Stage 2 main에서 multi-stimulus + multi-seed로 본격 statistical claim 가능.
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
CONDITIONS = ['baseline_uniform', 'SEEKING', 'FEAR', 'CARE']

# Pre-registered numpy seeds (Plan §2.4 YAML: reproducibility.numpy_seeds_main)
NUMPY_SEEDS_N10 = [42, 137, 271, 314, 1024, 1729, 2718, 3141, 6022, 9999]
N = len(NUMPY_SEEDS_N10)

DIMS_5D = ['Density (D)', 'Symmetry (S)', 'Centrality (C)', 'Depth (H)', 'Boundary (B)']
MCS_KEYS = ['M-CS1_value_core_proximity', 'M-CS2_self_reference_loop_proxy',
            'M-CS3_corpus_projection_entropy']
ALL_DIMS = DIMS_5D + MCS_KEYS  # 8 per-run dims (+M-CS4 cross-condition)

OUTPUT_PATH = os.path.join('D:\\', 'Coding', 'Sentient_AI', 'data',
                           'stage1_power_analysis_results.json')

# Bootstrap params
BOOTSTRAP_RESAMPLES = 1000
CI_ALPHA = 0.05


def run_one(affect_name, seed, san_cache=None):
    """One propagation run with given seed. san_cache enables reuse across seeds for one condition."""
    np.random.seed(seed)
    cfg = AFFECT_CONFIGS[affect_name]
    if san_cache is None or san_cache.get('affect') != affect_name:
        san = AffectModulatedSAN.load_from_json_with_affect(SAN_PATH, cfg)
        if san_cache is not None:
            san_cache['affect'] = affect_name
            san_cache['san'] = san
    else:
        san = san_cache['san']

    activation = san.propagate(STIMULUS_SEED, STIMULUS_CONTEXT)
    topology = san.compute_topological_vector(activation, STIMULUS_SEED)

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
    return measurements


def bootstrap_ci(values, n_resamples=BOOTSTRAP_RESAMPLES, alpha=CI_ALPHA):
    """Percentile bootstrap 95% CI on the mean. Returns (lo, hi)."""
    arr = np.array([v for v in values if math.isfinite(v)], dtype=float)
    if len(arr) < 2:
        return (float('nan'), float('nan'))
    n = len(arr)
    rng = np.random.default_rng(seed=12345)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def cohen_d(a_vals, b_vals):
    """Pooled-SD Cohen's d. NaN-safe."""
    a = np.array([v for v in a_vals if math.isfinite(v)], dtype=float)
    b = np.array([v for v in b_vals if math.isfinite(v)], dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    mean_diff = a.mean() - b.mean()
    var_pooled = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    sd_pooled = math.sqrt(var_pooled) if var_pooled > 0 else 0.0
    if sd_pooled == 0:
        # Zero variance — d is infinite if means differ, else 0
        return 0.0 if abs(mean_diff) < 1e-12 else float('inf') * (1 if mean_diff > 0 else -1)
    return float(mean_diff / sd_pooled)


def interpret_cohen_d(d):
    """Cohen's d interpretation (Cohen 1988)."""
    if not math.isfinite(d):
        return 'undefined'
    ad = abs(d)
    if ad < 0.2:
        return 'negligible'
    elif ad < 0.5:
        return 'small'
    elif ad < 0.8:
        return 'medium'
    elif ad < 1.2:
        return 'large'
    else:
        return 'very large'


def main():
    print('=' * 72)
    print(' STAGE 1 — POWER ANALYSIS (9-D, 4 conditions, N=10 seeds)')
    print(' Plan: Track1_Affect_Trajectory_실험_Plan v1.1 §3.2')
    print(f' Conditions: {CONDITIONS}')
    print(f' Stimulus: {STIMULUS_SEED} × {STIMULUS_CONTEXT}')
    print(f' Seeds (pre-registered, N=10): {NUMPY_SEEDS_N10}')
    print(f' Total runs: {len(CONDITIONS)} × {N} = {len(CONDITIONS) * N}')
    print('=' * 72)

    assert_config_invariants()
    print(' Config invariants check: PASS\n')

    # Collect measurements: results[condition][dim] = [val per seed]
    results_per_condition = {}
    for affect in CONDITIONS:
        print(f'--- {affect} (N={N} seeds) ---')
        per_dim = {d: [] for d in ALL_DIMS}
        san_cache = {'affect': None, 'san': None}
        for s in NUMPY_SEEDS_N10:
            try:
                m = run_one(affect, s, san_cache=san_cache)
                for d in ALL_DIMS:
                    per_dim[d].append(m.get(d, float('nan')))
                d_summary = ', '.join(f'{k.split()[0]}={m.get(k, float("nan")):.3f}'
                                       for k in DIMS_5D)
                print(f'   seed={s:5d}: {d_summary}')
            except Exception as e:
                print(f'   seed={s} CRASH: {e!r}')
                for d in ALL_DIMS:
                    per_dim[d].append(float('nan'))
        results_per_condition[affect] = per_dim
        print()

    # Summary stats per (condition, dim): mean, std, 95% CI
    print('=' * 72)
    print(' SUMMARY STATS PER CONDITION × DIM (mean ± std [95% CI bootstrap])')
    print('=' * 72)
    summary = {}
    for affect in CONDITIONS:
        summary[affect] = {}
        print(f'\n {affect}:')
        for d in ALL_DIMS:
            vals = results_per_condition[affect][d]
            finite_vals = [v for v in vals if math.isfinite(v)]
            if len(finite_vals) < 2:
                summary[affect][d] = {'mean': None, 'std': None, 'ci_lo': None, 'ci_hi': None, 'n_finite': len(finite_vals)}
                print(f'   {d:<40}: insufficient finite data ({len(finite_vals)}/{N})')
                continue
            arr = np.array(finite_vals)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1))
            ci_lo, ci_hi = bootstrap_ci(vals)
            summary[affect][d] = {
                'mean': mean, 'std': std, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
                'n_finite': len(finite_vals),
            }
            print(f'   {d:<40}: {mean:>8.4f} ± {std:>6.4f}  [95% CI: {ci_lo:>7.4f}, {ci_hi:>7.4f}]')

    # Pairwise Cohen's d per dim
    print('\n' + '=' * 72)
    print(" PAIRWISE COHEN'S D PER DIM (effect size between conditions)")
    print(' Interpretation: |d|<0.2 negligible, <0.5 small, <0.8 medium, <1.2 large, >=1.2 very large')
    print('=' * 72)
    pairs = []
    for i in range(len(CONDITIONS)):
        for j in range(i + 1, len(CONDITIONS)):
            a, b = CONDITIONS[i], CONDITIONS[j]
            print(f'\n {a} vs {b}:')
            pair_entry = {'pair': f'{a} vs {b}', 'cohen_d_per_dim': {}}
            for d in ALL_DIMS:
                va = results_per_condition[a][d]
                vb = results_per_condition[b][d]
                d_val = cohen_d(va, vb)
                interp = interpret_cohen_d(d_val)
                pair_entry['cohen_d_per_dim'][d] = {
                    'cohen_d': float(d_val) if math.isfinite(d_val) else None,
                    'interpretation': interp,
                }
                d_str = f'{d_val:>+7.3f}' if math.isfinite(d_val) else '   inf '
                print(f'   {d:<40}: d={d_str}  ({interp})')
            pairs.append(pair_entry)

    # M-CS4 cross-condition stability per seed
    print('\n' + '=' * 72)
    print(' M-CS4 cross-condition stability (per seed, MAD on 5-D across 4 conditions)')
    print('=' * 72)
    mcs4_per_seed = []
    for s_idx in range(N):
        topology_5d_per_condition = {}
        for affect in CONDITIONS:
            vec = np.array([results_per_condition[affect][d][s_idx] for d in DIMS_5D])
            if np.all(np.isfinite(vec)):
                topology_5d_per_condition[affect] = vec
        if len(topology_5d_per_condition) >= 2:
            mcs4 = m_cs4_core_self_stability(topology_5d_per_condition)
            mcs4_per_seed.append(float(mcs4))
            print(f'   seed={NUMPY_SEEDS_N10[s_idx]:5d}: M-CS4 = {mcs4:.4f}')
        else:
            mcs4_per_seed.append(float('nan'))
    mcs4_finite = [v for v in mcs4_per_seed if math.isfinite(v)]
    if mcs4_finite:
        mcs4_mean = float(np.mean(mcs4_finite))
        mcs4_std = float(np.std(mcs4_finite, ddof=1)) if len(mcs4_finite) > 1 else 0.0
        print(f'\n   M-CS4 across N=10 seeds: {mcs4_mean:.4f} ± {mcs4_std:.4f}')
    else:
        mcs4_mean, mcs4_std = None, None

    # Verdict — is N=10 sufficient for Stage 2?
    print('\n' + '=' * 72)
    print(' STAGE 1 VERDICT — Is N=10 sufficient for Stage 2 Main?')
    print('=' * 72)
    # Mechanical criteria: at least 1 pair × 1 dim with |d| >= 0.8 (medium/large effect)
    strong_effects = []
    for p in pairs:
        for d, dd in p['cohen_d_per_dim'].items():
            if dd['cohen_d'] is not None and abs(dd['cohen_d']) >= 0.8:
                strong_effects.append((p['pair'], d, dd['cohen_d']))
    n_strong = len(strong_effects)
    print(f' Strong effects (|d| >= 0.8): {n_strong}')
    for pair_name, d_name, d_val in strong_effects:
        print(f'   - {pair_name} on {d_name}: d={d_val:+.3f}')

    # Note: deterministic SAN means N=10 may show ZERO variance per condition.
    # In that case Cohen's d is either 0 or infinite — Stage 2 must add multi-stimulus.
    zero_var_count = 0
    for affect in CONDITIONS:
        for d in ALL_DIMS:
            s = summary[affect][d]
            if s.get('std') is not None and s['std'] < 1e-9:
                zero_var_count += 1
    print(f'\n Zero-variance (condition × dim) cells: {zero_var_count} / {len(CONDITIONS) * len(ALL_DIMS)}')
    if zero_var_count > 0:
        print(' → Deterministic SAN: N=10 same-stimulus seeds yield identical results.')
        print(' → §11.3 self-flag: Power analysis requires *stochastic propagation* or *multi-stimulus*.')
        print(' → Stage 2 must add multi-stimulus (Plan §3.3) and/or propagation noise.')

    # Save
    save_data = {
        'plan_version': 'Track1_Affect_Trajectory_실험_Plan v1.1',
        'stage': 'Stage 1 Power Analysis',
        'date_run': datetime.datetime.now().isoformat(timespec='seconds'),
        'stimulus': {'seed': STIMULUS_SEED, 'context': STIMULUS_CONTEXT},
        'numpy_seeds_n10': NUMPY_SEEDS_N10,
        'san_path': SAN_PATH,
        'conditions': CONDITIONS,
        'total_runs': len(CONDITIONS) * N,
        'value_core_lexicon_size': len(VALUE_CORE_LEXICON),
        'measurements_per_condition_per_dim': {
            aff: {d: [float(v) if math.isfinite(v) else None for v in vals]
                  for d, vals in d_dict.items()}
            for aff, d_dict in results_per_condition.items()
        },
        'summary_stats': summary,
        'pairwise_cohen_d': pairs,
        'm_cs4_per_seed': [float(v) if math.isfinite(v) else None for v in mcs4_per_seed],
        'm_cs4_mean': mcs4_mean,
        'm_cs4_std': mcs4_std,
        'strong_effects_count': n_strong,
        'zero_variance_cells': zero_var_count,
        'verdict_note': (
            'If zero_variance_cells > 0: Stage 2 must add multi-stimulus and/or '
            'propagation noise (Plan §3.3 + §11.3 self-flag)'
        ),
    }
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    print(f'\n Results saved: {OUTPUT_PATH}')
    print(f' Size: {os.path.getsize(OUTPUT_PATH)} bytes')

    return 0 if zero_var_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
