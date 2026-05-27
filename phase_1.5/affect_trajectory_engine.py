"""
Affect Trajectory Modulated SAN Propagation
Track1_Affect_Trajectory_실험_Plan v1.0 §2.4

Implements Panksepp 7-affect (LUST excluded) as propagation parameter modulation.

Stage 0 Pilot scope (this file):
  - baseline_uniform / SEEKING / FEAR — core parameters (gamma, type_bias, inhibitory_multiplier)
  - Minimum implementation for feasibility check (Gate 1)

Stage 1+ TODO (deferred):
  - RAGE, CARE, PANIC, PLAY full implementation
  - random_walk_variance_boost (Gaussian noise on input_sum)
  - novel_node_visit_bonus (track visit history)
  - boundary_crossing_penalty (cross-type edge penalty)
  - proximity_node_weight (value-core distance)
  - context_affect_map_multiplier (edge attribute filter)
  - value_core_anchor_weight (value-core list lookup)
  - loss_affect_node_bias (sensation_negative subset)
  - cross_category_edge_bonus (cross-type edge bonus)

§9·§10 정합 commit:
  - 명시 사양 (Plan §2.4 YAML) 그대로 hard-code
  - "잘 안 됨" 시 임의 환원 금지 — 보고 후 결정
  - Self-assessment 자동 수용 금지
"""
import sys
import os
import json
import numpy as np

# Import SensoryAssociativeNetwork from code/ sibling directory
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_THIS_DIR, '..', 'code')
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
from san_engine import SensoryAssociativeNetwork


# === Base Parameters (from code/baseline_and_ablation.py) ===
BASE_GAMMA = 0.35
BASE_THRESHOLD = 0.005
BASE_MAX_STEPS = 30
BASE_TOLERANCE = 1e-5
BASE_TYPE_BIAS = {
    'sensation': 1.8,
    'association': 1.5,
    'context': 1.2,
    'concept': 0.7,
}


# === Affect Configs (Plan §2.4 — hard-coded to match YAML pre-registration) ===
# Each config is the *complete* spec read by AffectModulatedSAN.
# Stage 0 Pilot uses only baseline_uniform / SEEKING / FEAR.
# Stage 1+ extends to RAGE/CARE/PANIC/PLAY (TODO in this file, present in YAML).

AFFECT_CONFIGS = {
    'baseline_uniform': {
        'panksepp_circuit': 'none (control)',
        'propagation_temperature': 1.0,
        'gamma': BASE_GAMMA * 1.0,                       # 0.35
        'type_bias': dict(BASE_TYPE_BIAS),               # unchanged
        'inhibitory_multiplier': 1.0,
    },
    'SEEKING': {
        'panksepp_circuit': 'VTA dopamine, mesolimbic — exploration, foraging, curiosity',
        'propagation_temperature': 1.5,
        'gamma': BASE_GAMMA * 1.5,                       # 0.525
        'type_bias': {**BASE_TYPE_BIAS, 'concept': 0.9}, # base 0.7 → 0.9
        'inhibitory_multiplier': 1.0,
    },
    'FEAR': {
        'panksepp_circuit': 'Central amygdala, PAG — avoidance, freeze, contraction',
        'propagation_temperature': 0.6,
        'gamma': BASE_GAMMA * 0.6,                       # 0.21
        'type_bias': {**BASE_TYPE_BIAS, 'sensation': 2.2}, # base 1.8 → 2.2
        'inhibitory_multiplier': 1.5,
    },
    'CARE': {  # Updated 2026-05-26 (밤) per S2-CARE-Ext — Stage 1+ extensions added
        'panksepp_circuit': 'Oxytocin, anterior cingulate — bonding, nurturance, proximity',
        'propagation_temperature': 1.0,
        'gamma': BASE_GAMMA * 1.0,                       # 0.35
        'type_bias': {**BASE_TYPE_BIAS, 'context': 1.6}, # base 1.2 → 1.6
        'inhibitory_multiplier': 1.0,
        # Stage 1+ extensions (Plan §2.4 YAML):
        'proximity_node_weight': 0.5,            # additive bonus for 1-hop neighbors of value-core
        'context_affect_map_multiplier': 1.3,    # edge × 1.3 for context-aware seed edges
    },
    # Updated 2026-05-27 per S3-RAGE-PANIC-PLAY-Ext: Stage 1+ extensions added (CARE-Ext pattern).
    'RAGE': {
        'panksepp_circuit': 'Medial amygdala, hypothalamus — aggression, irritability',
        'propagation_temperature': 1.0,
        'gamma': BASE_GAMMA * 1.0,                          # 0.35
        'type_bias': {**BASE_TYPE_BIAS, 'association': 1.8}, # base 1.5 → 1.8
        'inhibitory_multiplier': 1.0,
        # Stage 1+ ext (Plan §2.4):
        'inhibitory_propagation_bonus': 0.4,    # contribution × 1.4 on inhibitory (negative-weight) edges
        'boundary_crossing_penalty': 0.3,       # contribution × 0.7 on cross-type (u_type ≠ v_type) edges
    },
    'PANIC': {
        'panksepp_circuit': 'Dorsal periaqueductal gray, anterior cingulate — separation distress, sadness',
        'propagation_temperature': 0.8,
        'gamma': BASE_GAMMA * 0.8,                          # 0.28
        'type_bias': dict(BASE_TYPE_BIAS),                  # unchanged from base
        'inhibitory_multiplier': 1.0,
        # Stage 1+ ext (Plan §2.4):
        'value_core_anchor_weight': 1.3,        # input_sum × 1.3 when receiver (u) ∈ value-core
        'loss_affect_node_bias': 0.4,           # input_sum += 0.4 × alpha[u] when u ∈ LOSS_AFFECT_LEXICON
    },
    'PLAY': {
        'panksepp_circuit': 'Frontoparietal, basal ganglia — joy, social engagement',
        'propagation_temperature': 1.2,
        'gamma': BASE_GAMMA * 1.2,                          # 0.42
        'type_bias': dict(BASE_TYPE_BIAS),                  # unchanged from base
        'inhibitory_multiplier': 1.0,
        # Stage 1+ ext (Plan §2.4):
        'cross_category_edge_bonus': 0.2,       # contribution × 1.2 on cross-type (u_type ≠ v_type) edges
    },
    # LUST: excluded (Plan §2.2)
}

# PANIC's loss_affect_node_bias requires a lexicon of separation-distress / loss vocabulary
# (Panksepp PANIC circuit behavioral signature). Lexicon committed here for paper §14.6.3 transparency.
# §11.2 commit: precise definition before use.
LOSS_AFFECT_LEXICON = [
    "sorrow", "grief", "loss", "weep", "mourn", "sad", "lonely", "loneliness",
    "despair", "anguish", "lament", "tear", "agony", "misery",
    "forsaken", "desolate", "abandon", "separation",
]  # 18 items, committed prior to Stage 1 v4 execution


# === Verification: YAML ↔ Python config invariant ===
# §9 정합 — drift 발생 시 즉시 detect

def assert_config_invariants():
    """Verify hard-coded configs match expected Plan §2.4 values.
    Raises AssertionError if any drift is detected.
    """
    expected = {
        'baseline_uniform': {'gamma': 0.35,  'concept_bias': 0.7,  'inh_mult': 1.0},
        'SEEKING':          {'gamma': 0.525, 'concept_bias': 0.9,  'inh_mult': 1.0},
        'FEAR':             {'gamma': 0.21,  'sensation_bias': 2.2,'inh_mult': 1.5},
        'CARE':             {'gamma': 0.35,  'context_bias': 1.6,  'inh_mult': 1.0,
                              'proximity_weight': 0.5, 'ctx_aff_mult': 1.3},
        'RAGE':             {'gamma': 0.35,  'association_bias': 1.8, 'inh_mult': 1.0,
                              'inh_prop_bonus': 0.4, 'boundary_penalty': 0.3},
        'PANIC':            {'gamma': 0.28,  'inh_mult': 1.0,
                              'vc_anchor': 1.3, 'loss_bias': 0.4},
        'PLAY':             {'gamma': 0.42,  'inh_mult': 1.0,
                              'cross_bonus': 0.2},
    }
    for name, exp in expected.items():
        cfg = AFFECT_CONFIGS[name]
        assert abs(cfg['gamma'] - exp['gamma']) < 1e-9, \
            f"{name} gamma drift: {cfg['gamma']} vs expected {exp['gamma']}"
        if 'concept_bias' in exp:
            assert abs(cfg['type_bias']['concept'] - exp['concept_bias']) < 1e-9, \
                f"{name} concept bias drift"
        if 'sensation_bias' in exp:
            assert abs(cfg['type_bias']['sensation'] - exp['sensation_bias']) < 1e-9, \
                f"{name} sensation bias drift"
        if 'context_bias' in exp:
            assert abs(cfg['type_bias']['context'] - exp['context_bias']) < 1e-9, \
                f"{name} context bias drift"
        if 'association_bias' in exp:
            assert abs(cfg['type_bias']['association'] - exp['association_bias']) < 1e-9, \
                f"{name} association bias drift"
        assert abs(cfg['inhibitory_multiplier'] - exp['inh_mult']) < 1e-9, \
            f"{name} inhibitory_multiplier drift"
        if 'proximity_weight' in exp:
            assert abs(cfg.get('proximity_node_weight', 0.0) - exp['proximity_weight']) < 1e-9, \
                f"{name} proximity_node_weight drift"
        if 'ctx_aff_mult' in exp:
            assert abs(cfg.get('context_affect_map_multiplier', 1.0) - exp['ctx_aff_mult']) < 1e-9, \
                f"{name} context_affect_map_multiplier drift"
        if 'inh_prop_bonus' in exp:
            assert abs(cfg.get('inhibitory_propagation_bonus', 0.0) - exp['inh_prop_bonus']) < 1e-9, \
                f"{name} inhibitory_propagation_bonus drift"
        if 'boundary_penalty' in exp:
            assert abs(cfg.get('boundary_crossing_penalty', 0.0) - exp['boundary_penalty']) < 1e-9, \
                f"{name} boundary_crossing_penalty drift"
        if 'vc_anchor' in exp:
            assert abs(cfg.get('value_core_anchor_weight', 1.0) - exp['vc_anchor']) < 1e-9, \
                f"{name} value_core_anchor_weight drift"
        if 'loss_bias' in exp:
            assert abs(cfg.get('loss_affect_node_bias', 0.0) - exp['loss_bias']) < 1e-9, \
                f"{name} loss_affect_node_bias drift"
        if 'cross_bonus' in exp:
            assert abs(cfg.get('cross_category_edge_bonus', 0.0) - exp['cross_bonus']) < 1e-9, \
                f"{name} cross_category_edge_bonus drift"
    return True


class AffectModulatedSAN(SensoryAssociativeNetwork):
    """SAN with Panksepp affect trajectory modulation.

    Modulates propagation via:
      - gamma (damping_factor): propagation_temperature × BASE_GAMMA
      - type_bias: per-affect adjusted dict
      - inhibitory_multiplier: amplify inhibitory (negative-weight) edge effects
      - proximity_node_weight (Stage 1+, CARE): additive bonus for 1-hop neighbors of value-core
      - context_affect_map_multiplier (Stage 1+, CARE): edge × multiplier for context-aware seed edges

    Operationalization commitments (§11.2):
      - "proximity node" := 1-hop neighbor of any value-core lexicon node
      - "context-aware seed edge" := edge with non-empty compatible_contexts AND active stimulus context ∈ compatible_contexts

    See Track1_Affect_Trajectory_실험_Plan v1.1 §2.4 for YAML spec.
    """

    def __init__(self, affect_config, threshold=BASE_THRESHOLD, max_steps=BASE_MAX_STEPS,
                 tolerance=BASE_TOLERANCE):
        super().__init__(
            damping_factor=affect_config['gamma'],
            threshold=threshold,
            max_steps=max_steps,
            tolerance=tolerance,
            type_bias=affect_config['type_bias'],
        )
        self.affect_name = affect_config.get('panksepp_circuit', 'unknown')
        self.inhibitory_multiplier = affect_config.get('inhibitory_multiplier', 1.0)
        # Stage 1+ extensions (default no-op)
        self.proximity_node_weight = affect_config.get('proximity_node_weight', 0.0)         # CARE
        self.context_affect_multiplier = affect_config.get('context_affect_map_multiplier', 1.0)  # CARE
        # RAGE Stage 1+ ext
        self.inhibitory_propagation_bonus = affect_config.get('inhibitory_propagation_bonus', 0.0)
        self.boundary_crossing_penalty = affect_config.get('boundary_crossing_penalty', 0.0)
        # PANIC Stage 1+ ext
        self.value_core_anchor_weight = affect_config.get('value_core_anchor_weight', 1.0)
        self.loss_affect_node_bias = affect_config.get('loss_affect_node_bias', 0.0)
        # PLAY Stage 1+ ext
        self.cross_category_edge_bonus = affect_config.get('cross_category_edge_bonus', 0.0)
        # State containers
        self._proximity_set = set()
        self._value_core_set = set()
        self._loss_affect_set = set()

    def set_value_core(self, value_core_lexicon):
        """Pre-compute proximity set: 1-hop neighbors of value-core nodes in graph.

        Operationalization §11.2: proximity := 1-hop neighbor of any value-core node.
        Also used by PANIC value_core_anchor_weight (receiver ∈ value_core).
        Call AFTER graph loaded; before propagate().
        """
        self._value_core_set = set(value_core_lexicon)
        self._proximity_set = set()
        for vc in self._value_core_set:
            if vc in self.graph:
                self._proximity_set.add(vc)
                self._proximity_set.update(self.graph.neighbors(vc))
        return len(self._proximity_set)

    def set_loss_affect(self, loss_affect_lexicon):
        """Set loss-affect lexicon for PANIC Stage 1+ ext (loss_affect_node_bias).

        Operationalization §11.2: u ∈ LOSS_AFFECT_LEXICON triggers additive input bonus.
        """
        self._loss_affect_set = set(loss_affect_lexicon)
        return len(self._loss_affect_set & set(self.graph.nodes())) if self.graph else 0

    def add_association(self, u, v, weight, compatibility_context=None):
        """Override: apply inhibitory_multiplier to negative weights only."""
        if weight < 0 and self.inhibitory_multiplier != 1.0:
            weight = weight * self.inhibitory_multiplier
        super().add_association(u, v, weight, compatibility_context)

    def _any_stage1_ext_active(self):
        """Check if any Stage 1+ extension is active for this affect."""
        return (
            self.proximity_node_weight > 0.0
            or self.context_affect_multiplier != 1.0
            or self.inhibitory_propagation_bonus > 0.0
            or self.boundary_crossing_penalty > 0.0
            or self.value_core_anchor_weight != 1.0
            or self.loss_affect_node_bias > 0.0
            or self.cross_category_edge_bonus > 0.0
        )

    def propagate(self, seed_concept, context):
        """Override base propagate to inject Stage 1+ extensions when active.

        Fast path: if no Stage 1+ ext, defer to base (baseline/SEEKING/FEAR minimum).

        Stage 1+ extensions handled per affect (Plan §2.4 + §11.2 operationalizations):
          CARE:
            - Edge contribution × context_affect_multiplier when compat_contexts non-empty
              AND active context ∈ compat_contexts
            - input_sum += proximity_node_weight × alpha[u] when u ∈ proximity_set
          RAGE:
            - Inhibitory edge contribution × (1 + inhibitory_propagation_bonus)
            - Cross-type edge contribution × (1 - boundary_crossing_penalty)
          PANIC:
            - input_sum × value_core_anchor_weight when receiver u ∈ value_core_set
            - input_sum += loss_affect_node_bias × alpha[u] when u ∈ loss_affect_set
          PLAY:
            - Cross-type edge contribution × (1 + cross_category_edge_bonus)

        Note: RAGE boundary_crossing_penalty and PLAY cross_category_edge_bonus are
        *opposite-direction* modulations on the same edge predicate (cross-type edges).
        For mixed configs, both apply multiplicatively.
        """
        if not self._any_stage1_ext_active():
            return super().propagate(seed_concept, context)

        import math
        if seed_concept not in self.graph:
            raise ValueError(f"Seed concept '{seed_concept}' not in network.")

        nodes = list(self.graph.nodes())
        num_nodes = len(nodes)
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        alpha = np.zeros(num_nodes)
        idx_seed = node_to_idx[seed_concept]
        alpha[idx_seed] = 1.0

        idx_context = None
        if context in self.graph:
            idx_context = node_to_idx[context]
            alpha[idx_context] = 1.0

        # Cache active flags
        proximity_active = self.proximity_node_weight > 0 and len(self._proximity_set) > 0
        ctx_mult_active = self.context_affect_multiplier != 1.0
        inh_bonus_active = self.inhibitory_propagation_bonus > 0
        boundary_penalty_active = self.boundary_crossing_penalty > 0
        vc_anchor_active = self.value_core_anchor_weight != 1.0 and len(self._value_core_set) > 0
        loss_bias_active = self.loss_affect_node_bias > 0 and len(self._loss_affect_set) > 0
        cross_bonus_active = self.cross_category_edge_bonus > 0

        for step in range(self.max_steps):
            alpha_next = np.zeros(num_nodes)

            for u in self.graph.nodes():
                idx_u = node_to_idx[u]
                if idx_u == idx_seed or idx_u == idx_context:
                    alpha_next[idx_u] = 1.0
                    continue

                u_type = self.graph.nodes[u].get('type', 'concept')
                u_bias = self.type_bias.get(u_type, 1.0)

                input_sum = 0.0
                for v in self.graph.neighbors(u):
                    idx_v = node_to_idx[v]
                    if alpha[idx_v] <= 0.0:
                        continue
                    edge_data = self.graph.get_edge_data(u, v)
                    weight = edge_data['weight']
                    compat_contexts = edge_data.get('compatible_contexts', [])
                    is_compatible = (len(compat_contexts) == 0 or context in compat_contexts)

                    if is_compatible:
                        deg_u = self.graph.degree(u)
                        deg_v = self.graph.degree(v)
                        norm = math.sqrt(deg_u * deg_v) if deg_u * deg_v > 0 else 1.0
                        contrib = (weight / norm) * alpha[idx_v]

                        # CARE Stage 1+: context_affect_multiplier on context-aware seed edges
                        if ctx_mult_active and compat_contexts and context in compat_contexts:
                            contrib *= self.context_affect_multiplier

                        # RAGE Stage 1+: inhibitory_propagation_bonus on negative-weight edges
                        # (Note: inhibitory_multiplier already applied at add_association load time;
                        #  this is an *additional* propagation-time amplification.)
                        if inh_bonus_active and weight < 0:
                            contrib *= (1.0 + self.inhibitory_propagation_bonus)

                        # RAGE / PLAY Stage 1+: cross-type edge modulation (opposite signs)
                        v_type = self.graph.nodes[v].get('type', 'concept')
                        if u_type != v_type:
                            if boundary_penalty_active:
                                contrib *= (1.0 - self.boundary_crossing_penalty)
                            if cross_bonus_active:
                                contrib *= (1.0 + self.cross_category_edge_bonus)

                        input_sum += contrib

                input_sum *= u_bias

                # CARE Stage 1+: proximity bonus
                if proximity_active and u in self._proximity_set:
                    input_sum += self.proximity_node_weight * alpha[idx_u]

                # PANIC Stage 1+: value_core_anchor_weight on receivers ∈ value_core
                if vc_anchor_active and u in self._value_core_set:
                    input_sum *= self.value_core_anchor_weight

                # PANIC Stage 1+: loss_affect_node_bias additive bonus
                if loss_bias_active and u in self._loss_affect_set:
                    input_sum += self.loss_affect_node_bias * alpha[idx_u]

                val = (1 - self.gamma) * alpha[idx_u] + self.gamma * input_sum
                val = max(0.0, min(1.0, val))
                alpha_next[idx_u] = val if val >= self.theta else 0.0

            alpha_next[idx_seed] = 1.0
            if idx_context is not None:
                alpha_next[idx_context] = 1.0

            diff = np.linalg.norm(alpha_next - alpha, ord=1)
            alpha = alpha_next
            if diff < self.tolerance:
                break

        return {nodes[i]: alpha[i] for i in range(num_nodes)}

    @classmethod
    def load_from_json_with_affect(cls, path, affect_config, threshold=BASE_THRESHOLD,
                                    max_steps=BASE_MAX_STEPS, tolerance=BASE_TOLERANCE,
                                    value_core_lexicon=None):
        """Load SAN with Panksepp affect modulation applied at edge ingest.

        If value_core_lexicon is provided, value_core and proximity sets are precomputed
        (used by CARE proximity + PANIC value_core_anchor).
        LOSS_AFFECT_LEXICON is always set (used by PANIC loss_affect_node_bias).
        """
        san = cls(affect_config, threshold=threshold, max_steps=max_steps, tolerance=tolerance)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for node in data['nodes']:
            san.add_node(node['id'], node['type'], node.get('description', ''))
        for edge in data['edges']:
            san.add_association(
                edge['source'], edge['target'], edge['weight'],
                edge.get('compatible_contexts', None)
            )
        # Precompute Stage 1+ ext supporting sets (no-op if ext not active for this affect)
        if value_core_lexicon is not None:
            san.set_value_core(value_core_lexicon)
        san.set_loss_affect(LOSS_AFFECT_LEXICON)
        return san

    def get_config_summary(self):
        return {
            'affect_name': self.affect_name,
            'gamma': self.gamma,
            'type_bias': dict(self.type_bias),
            'inhibitory_multiplier': self.inhibitory_multiplier,
            'proximity_node_weight': self.proximity_node_weight,
            'context_affect_multiplier': self.context_affect_multiplier,
            'proximity_set_size': len(self._proximity_set),
            'threshold': self.theta,
            'max_steps': self.max_steps,
            'n_nodes': len(self.graph.nodes),
            'n_edges': len(self.graph.edges),
        }


if __name__ == '__main__':
    # Sanity: verify config invariants on import
    assert_config_invariants()
    print('AFFECT_CONFIGS invariant check: PASS')
    print('Available conditions:', list(AFFECT_CONFIGS.keys()))
    for name, cfg in AFFECT_CONFIGS.items():
        print(f'  {name}: gamma={cfg["gamma"]:.3f}, '
              f'inh_mult={cfg["inhibitory_multiplier"]}, '
              f'circuit="{cfg["panksepp_circuit"]}"')
