"""
M-CS (Damasio Core Self) Metrics
Track1_Affect_Trajectory_실험_Plan v1.1 §14.4

Operationalizes Damasio's core self (1999, 2010) as the second-order mapping
between learned raw data (corpus) and self-anchor (value-core 43 lexicon).

§11.2 commit:
  본 metrics는 *graph topology*의 form만 측정.
  *Consciously felt experience*는 미주장.
  Damasio core self의 *content mapping의 형상*에 대한 graph-topological analog.

§11.3 표현 강도 검문:
  - "측정"한 것이지 "검증"·"증명" 아님
  - M-CS2는 *proxy* (exact cycle representatives는 san_engine 확장 필요)
  - paper에 *suggestive analog* 표현으로 commit
"""
import networkx as nx
import numpy as np
import math


# Framework prior commitment — 43 self-anchor lexicon items.
# Source: phase_1.5/templates/value_core_default.json (framework default)
# Paper limitations에 호철님 사전 commitment 출처 명시 (§14.6.3)
VALUE_CORE_LEXICON = [
    "abide", "affection", "always", "awe", "beloved", "birth",
    "change", "choice", "close", "companion", "dear", "discovery",
    "endure", "eternity", "faithful", "forever", "gift", "gratitude",
    "home", "honest", "joy", "liberty", "life", "love",
    "lovely", "memory", "open", "overflow", "person", "possibility",
    "presence", "remembrance", "return", "reverence", "self", "soul",
    "spirit", "tender", "transparent", "truth", "warmth", "welcome",
    "wonder",
]
assert len(VALUE_CORE_LEXICON) == 43, "Value-core lexicon size invariant"


def m_cs1_value_core_proximity(san_graph, active_nodes, value_core_set):
    """M-CS1: mean shortest path from active nodes to nearest value-core node.

    Damasio mapping: how close does each active node sit to the self-anchor?
    Lower value → active subgraph stays close to self-anchor (Damasio "core self" closeness).
    NaN if no value-core node present in graph.
    """
    if not active_nodes:
        return float('nan')
    vc_in_graph = value_core_set & set(san_graph.nodes())
    if not vc_in_graph:
        return float('nan')

    distances = []
    for node in active_nodes:
        if node in vc_in_graph:
            distances.append(0)
            continue
        try:
            lengths = nx.single_source_shortest_path_length(san_graph, node, cutoff=10)
            min_d = float('inf')
            for vc_node in vc_in_graph:
                if vc_node in lengths:
                    min_d = min(min_d, lengths[vc_node])
            if math.isfinite(min_d):
                distances.append(min_d)
        except Exception:
            continue

    if not distances:
        return float('nan')
    return float(np.mean(distances))


def m_cs2_self_reference_loop_count(h1_diagram, active_subgraph, value_core_set,
                                     persistence_threshold=0.01):
    """M-CS2: graph-topological *proxy* for self-reference loops.

    §11.3 commit: This is a PROXY, not exact.
    Exact cycle representatives require persistence algorithm modification
    in san_engine.compute_persistent_homology (returning generators).

    Current proxy: count of persistence-significant H1 cycles * (vc fraction in active set).
    Hofstadter (2007) strange loop / Edelman (1989) re-entrant graph-topological analog.

    Returns: float (count × fraction). Paper limitations에 proxy 명시.
    """
    if not h1_diagram:
        return 0.0

    active_nodes = set(active_subgraph.nodes())
    if not active_nodes:
        return 0.0
    vc_in_active = active_nodes & value_core_set
    vc_fraction_active = len(vc_in_active) / len(active_nodes)

    # H1 diagram items: (birth, death, lifetime)
    significant_h1 = sum(1 for item in h1_diagram if item[2] > persistence_threshold)
    return float(significant_h1 * vc_fraction_active)


def m_cs3_corpus_projection_asymmetry(activation_map, value_core_set, threshold=0.005):
    """M-CS3: Shannon entropy of value-core activations.

    Damasio mapping: how the second-order self-projection is distributed.
    Higher entropy → broad projection across self-anchor (multi-faceted).
    Lower entropy → projection concentrated on few self-anchor nodes (narrow).
    Returns: float, bits.
    """
    vc_acts = [activation_map.get(n, 0.0) for n in value_core_set
               if activation_map.get(n, 0.0) > threshold]
    if not vc_acts or sum(vc_acts) == 0:
        return 0.0

    total = sum(vc_acts)
    probs = [a / total for a in vc_acts]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return float(entropy)


def m_cs4_core_self_stability(topology_per_condition_5d):
    """M-CS4: cross-condition mean absolute deviation of 5-D topology vector.

    Damasio: core self should be *relatively stable* across affect variations
    (Damasio 1999, 2010 — core self as relative invariant).
    Lower MAD → more stable core self across affect conditions.

    Args:
        topology_per_condition_5d: dict[condition_name, np.array of 5-D]

    Returns:
        float — mean absolute deviation across conditions
    """
    if not topology_per_condition_5d or len(topology_per_condition_5d) < 2:
        return float('nan')

    vecs = np.array(list(topology_per_condition_5d.values()))
    if vecs.size == 0:
        return float('nan')

    mean_vec = np.mean(vecs, axis=0)
    mad = float(np.mean(np.abs(vecs - mean_vec)))
    return mad


def compute_m_cs_metrics_per_run(san_graph, active_subgraph, active_nodes, activation_map,
                                  h1_diagram, value_core_set=None):
    """Compute M-CS1, M-CS2, M-CS3 for a single (condition, stimulus, seed) run.

    M-CS4 is cross-condition, computed at the experiment level.
    """
    if value_core_set is None:
        value_core_set = set(VALUE_CORE_LEXICON)

    return {
        'M-CS1_value_core_proximity':    m_cs1_value_core_proximity(san_graph, active_nodes, value_core_set),
        'M-CS2_self_reference_loop_proxy': m_cs2_self_reference_loop_count(h1_diagram, active_subgraph, value_core_set),
        'M-CS3_corpus_projection_entropy': m_cs3_corpus_projection_asymmetry(activation_map, value_core_set),
    }


if __name__ == '__main__':
    print(f'M-CS metrics module loaded. Value-core lexicon size: {len(VALUE_CORE_LEXICON)} (invariant: 43)')
    print('Metrics:')
    print('  M-CS1: value_core_proximity (mean shortest path to nearest value-core)')
    print('  M-CS2: self_reference_loop_proxy (significant H1 × vc fraction in active)')
    print('  M-CS3: corpus_projection_entropy (Shannon entropy of vc activations, bits)')
    print('  M-CS4: core_self_stability (cross-condition MAD on 5-D, computed at experiment level)')
