"""
plasticity.py — Phase 1.5 시냅스 가소성 엔진 (6 + 1 메커니즘).

학술 토대:
- Hebbian co-activation     : Hebb 1949, Bliss & Lømo 1973
- Habituation               : Thompson & Spencer 1966, Rankin et al. 2009
- Sensitization             : Groves & Thompson 1970, Friston 2010 active inference
- Homeostatic scaling       : Turrigiano 2008, 2012 (multiplicative)
- Fading Affect Bias (FAB)  : Walker & Skowronski 2003, 2025
- Consolidation cycle       : Diekelmann-Born 2010; Kirkpatrick et al. 2017 EWC
- Value Core protection     : 7번째 메커니즘 — value_core.ValueCore와 연동

원칙:
- 본 모듈은 순수 알고리즘. 어떤 SAN 인스턴스에도 적용 가능.
- 상태(누적 카운터·예측·log)는 PlasticityState dataclass에 격리되어
  SANInstance가 보유. plasticity 함수들은 (engine, state, cfg, ...)
  서명을 받아 in-place로 갱신한다.
- 모든 hyperparameter는 PlasticityConfig로 외부화 → 실험에서 튜닝.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Config & state
# ---------------------------------------------------------------------------


@dataclass
class PlasticityConfig:
    """Phase 1.5 6+1 메커니즘 하이퍼파라미터.

    초기값은 NeuroAI continual learning 표준 범위 + 노트 §4 예시값 기반.
    실험(EXP-T1-07~09) 결과로 검증·튜닝한다.
    """

    # 1. Hebbian
    eta_H: float = 0.01

    # 2. Habituation
    lambda_H: float = 0.05
    gamma_decay: float = 0.95
    h_cap: float = 50.0

    # 3. Sensitization
    sigma_S: float = 0.5
    pe_clip: float = 1.0

    # 4. Homeostatic scaling
    # T_target가 None이면 instance 초기화 시점의 노드별 입력 가중치 합으로 자동 설정
    T_target: float | None = None
    scaling_period: int = 5  # 매 5 propagation 마다

    # 5. Fading Affect Bias (per-session multiplicative decay)
    delta_pos: float = 0.995
    delta_neg: float = 0.990
    delta_neutral: float = 0.998

    # 6. Consolidation cycle
    consolidation_period: int = 20
    top_k_replay: int = 5
    prune_threshold: float = 0.01
    rem_smoothing: float = 0.05  # 양성 edge amplitude smoothing 비율

    # 7. Value Core protection (value_core.ValueCore와 함께 동작)
    lambda_protect: float = 0.99  # value-core 노드 Δw는 (1 - lambda_protect) 만큼만 반영
    # Adversary anchor 활성화 시 alarm 임계
    adversary_alarm_threshold: float = 0.3

    # Modulation factor — neuromodulator-inspired gating
    pe_gate_kappa: float = 1.0


@dataclass
class PlasticityState:
    """Per-instance 가소성 상태. SAN의 graph 옆에 동거."""

    # node-id -> cumulative activation (Habituation accumulator h(v))
    habituation_counter: dict[str, float] = field(default_factory=dict)

    # (concept, context) -> latest activation_map (prediction baseline)
    predictions: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)

    # 양성/부성 sensation 노드 집합 (FAB용). instance 빌드 시 채워짐.
    positive_sensations: set[str] = field(default_factory=set)
    negative_sensations: set[str] = field(default_factory=set)

    # 최근 N epoch 강한 패턴 — consolidation replay 대상
    recent_strong_patterns: list[dict] = field(default_factory=list)

    # propagation 카운터
    step: int = 0

    # 노드별 EWC fisher 근사 (co-activation 빈도). value-core 노드는 따로 보호.
    fisher_proxy: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _active_nodes(activation_map: dict[str, float], threshold: float) -> list[str]:
    return [v for v, a in activation_map.items() if a > threshold]


def _is_value_core_edge(u: str, v: str, value_core_nodes: set[str] | None) -> bool:
    if not value_core_nodes:
        return False
    return u in value_core_nodes or v in value_core_nodes


# ---------------------------------------------------------------------------
# 1. Hebbian co-activation update
# ---------------------------------------------------------------------------


def apply_hebbian(
    graph: nx.Graph,
    state: PlasticityState,
    cfg: PlasticityConfig,
    activation_map: dict[str, float],
    prediction_error: float,
    value_core_nodes: set[str] | None = None,
    activation_threshold: float = 0.10,
) -> int:
    """매 propagation 후 active subgraph의 edge weight 강화.

    공식:
        w(u,v) ← w(u,v) + η_H · α*(u) · α*(v) · m(u,v)
        m(u,v) = 1 + κ · |PE|         (neuromodulator gating)
    Value-core edge는 (1 - λ_protect)로 변화량 억제.

    Returns:
        실제 업데이트된 edge 수.
    """
    active = _active_nodes(activation_map, activation_threshold)
    modulation = 1.0 + cfg.pe_gate_kappa * abs(prediction_error)
    updated = 0
    for u, v, data in graph.edges(active, data=True):
        if v not in activation_map:
            continue
        au = activation_map.get(u, 0.0)
        av = activation_map.get(v, 0.0)
        if au <= 0.0 or av <= 0.0:
            continue
        delta = cfg.eta_H * au * av * modulation
        # 부정(억제) edge는 절대값 강화 → 부호 보존
        sign = 1.0 if data.get("weight", 0.0) >= 0 else -1.0
        if _is_value_core_edge(u, v, value_core_nodes):
            delta *= (1.0 - cfg.lambda_protect)
        data["weight"] = float(data["weight"] + sign * delta)
        updated += 1

        # Fisher proxy: co-activation 빈도 누적
        for n in (u, v):
            state.fisher_proxy[n] = state.fisher_proxy.get(n, 0.0) + au * av
    return updated


# ---------------------------------------------------------------------------
# 2. Habituation
# ---------------------------------------------------------------------------


def apply_habituation(
    state: PlasticityState,
    cfg: PlasticityConfig,
    activation_map: dict[str, float],
    activation_threshold: float = 0.10,
) -> dict[str, float]:
    """노드별 누적 활성화에 따른 반응 감소.

    공식:
        α*(v) ← α*(v) · exp(-λ_H · h(v))
        h(v) ← γ_decay · h(v) + α*(v)   (다음 step 직전에)

    Returns:
        habituated activation map (새 dict, 입력은 미변경).
    """
    habituated: dict[str, float] = {}
    for v, a in activation_map.items():
        h = state.habituation_counter.get(v, 0.0)
        factor = float(np.exp(-cfg.lambda_H * h))
        habituated[v] = a * factor
    # h(v) 갱신: γ_decay 회복 + 현 활성화 가산
    for v, a in habituated.items():
        if a > activation_threshold:
            h_prev = state.habituation_counter.get(v, 0.0)
            h_new = cfg.gamma_decay * h_prev + a
            state.habituation_counter[v] = min(h_new, cfg.h_cap)
        else:
            # 비활성 노드는 천천히 회복
            h_prev = state.habituation_counter.get(v, 0.0)
            state.habituation_counter[v] = cfg.gamma_decay * h_prev
    return habituated


# ---------------------------------------------------------------------------
# 3. Sensitization (surprise-driven amplification)
# ---------------------------------------------------------------------------


def apply_sensitization(
    state: PlasticityState,
    cfg: PlasticityConfig,
    activation_map: dict[str, float],
    seed_concept: str,
    context: str,
) -> tuple[dict[str, float], float]:
    """Prediction error에 비례한 노드별 활성화 증폭.

    공식:
        α*(v) ← α*(v) · (1 + σ_S · PE(v))
    PE(v) = |α*_current(v) - α*_predicted(v)|  (직전 같은 stimulus의 기록 대비)

    Returns:
        (sensitized activation map, scalar prediction error 평균)
    """
    key = (seed_concept, context)
    predicted = state.predictions.get(key, {})
    sensitized: dict[str, float] = {}
    pe_sum = 0.0
    pe_count = 0
    for v, a in activation_map.items():
        a_pred = predicted.get(v, 0.0)
        pe = abs(a - a_pred)
        pe = min(pe, cfg.pe_clip)
        gain = 1.0 + cfg.sigma_S * pe
        sensitized[v] = float(min(1.0, max(0.0, a * gain)))
        pe_sum += pe
        pe_count += 1
    # 다음 stimulus를 위해 prediction 갱신 (EMA: 50/50)
    new_pred: dict[str, float] = {}
    for v, a in sensitized.items():
        new_pred[v] = 0.5 * predicted.get(v, 0.0) + 0.5 * a
    state.predictions[key] = new_pred
    mean_pe = pe_sum / pe_count if pe_count > 0 else 0.0
    return sensitized, mean_pe


# ---------------------------------------------------------------------------
# 4. Homeostatic scaling (Turrigiano multiplicative)
# ---------------------------------------------------------------------------


def apply_homeostatic_scaling(
    graph: nx.Graph,
    cfg: PlasticityConfig,
    T_target: float | None = None,
) -> int:
    """각 노드의 입력 가중치 합이 T_target을 초과하면 곱셈적 정규화.

    상대 weight pattern은 보존된다 — Turrigiano가 강조한 'preserves
    relative tuning' 원칙. Runaway 방지.

    Returns:
        실제 scaled된 노드 수.
    """
    target = T_target if T_target is not None else cfg.T_target
    if target is None or target <= 0.0:
        return 0
    scaled_nodes = 0
    for v in graph.nodes():
        in_edges = list(graph.edges(v, data=True))
        total = sum(abs(d.get("weight", 0.0)) for _, _, d in in_edges)
        if total <= target:
            continue
        scale = target / total
        for _, _, d in in_edges:
            d["weight"] = float(d.get("weight", 0.0) * scale)
        scaled_nodes += 1
    return scaled_nodes


# ---------------------------------------------------------------------------
# 5. Fading Affect Bias (asymmetric decay)
# ---------------------------------------------------------------------------


def apply_fading_affect_bias(
    graph: nx.Graph,
    state: PlasticityState,
    cfg: PlasticityConfig,
    value_core_nodes: set[str] | None = None,
) -> tuple[int, int]:
    """매 session/epoch 끝에 모든 edge에 valence-dependent decay.

    공식:
        if v ∈ negative_sensations: w ← w · δ_neg     (빠른 fade)
        elif v ∈ positive_sensations: w ← w · δ_pos   (느린 fade)
        else: w ← w · δ_neutral

    Value-core node와 연결된 edge는 decay에서 제외 (절대 보호).

    Returns:
        (positive-fade edge 수, negative-fade edge 수)
    """
    pos_count = 0
    neg_count = 0
    for u, v, data in graph.edges(data=True):
        if _is_value_core_edge(u, v, value_core_nodes):
            continue  # Value Core protected
        decay = cfg.delta_neutral
        if v in state.negative_sensations or u in state.negative_sensations:
            decay = cfg.delta_neg
            neg_count += 1
        elif v in state.positive_sensations or u in state.positive_sensations:
            decay = cfg.delta_pos
            pos_count += 1
        data["weight"] = float(data.get("weight", 0.0) * decay)
    return pos_count, neg_count


# ---------------------------------------------------------------------------
# 6. Consolidation cycle (offline replay + prune)
# ---------------------------------------------------------------------------


def apply_consolidation(
    graph: nx.Graph,
    state: PlasticityState,
    cfg: PlasticityConfig,
    value_core_nodes: set[str] | None = None,
) -> dict:
    """매 N epoch마다 sleep-유사 replay + 약한 edge 가지치기.

    1) top-K 강한 패턴 (recent_strong_patterns)을 재활성화 → Hebbian 강화
    2) prune_threshold 이하 edge 제거 (value-core edge 제외)
    3) REM-유사 smoothing — 양성 edge 진폭 평탄화

    Returns:
        {'replayed': K, 'pruned': N, 'smoothed': M} stats.
    """
    # 1) Replay top-K strongest patterns
    patterns = sorted(
        state.recent_strong_patterns,
        key=lambda p: p.get("intensity", 0.0),
        reverse=True,
    )[: cfg.top_k_replay]

    replayed = 0
    for pat in patterns:
        active_map: dict[str, float] = pat.get("activation_map", {})
        for u, v, data in graph.edges(list(active_map.keys()), data=True):
            au = active_map.get(u, 0.0)
            av = active_map.get(v, 0.0)
            if au <= 0.0 or av <= 0.0:
                continue
            delta = cfg.eta_H * au * av  # PE=0 (offline)
            if _is_value_core_edge(u, v, value_core_nodes):
                delta *= (1.0 - cfg.lambda_protect)
            sign = 1.0 if data.get("weight", 0.0) >= 0 else -1.0
            data["weight"] = float(data.get("weight", 0.0) + sign * delta)
        replayed += 1

    # 2) Prune weak edges
    to_remove = []
    for u, v, data in graph.edges(data=True):
        w = abs(data.get("weight", 0.0))
        if w < cfg.prune_threshold and not _is_value_core_edge(u, v, value_core_nodes):
            to_remove.append((u, v))
    graph.remove_edges_from(to_remove)

    # 3) REM-equivalent smoothing on positive edges
    smoothed = 0
    for u, v, data in graph.edges(data=True):
        w = data.get("weight", 0.0)
        if w > 0 and (v in state.positive_sensations or u in state.positive_sensations):
            # Pull toward median positive weight (lightweight smoothing)
            data["weight"] = float(w * (1.0 - cfg.rem_smoothing) + 0.5 * cfg.rem_smoothing)
            smoothed += 1

    # Reset recent buffer
    state.recent_strong_patterns = []
    return {"replayed": replayed, "pruned": len(to_remove), "smoothed": smoothed}


# ---------------------------------------------------------------------------
# Orchestrator — one full plasticity step
# ---------------------------------------------------------------------------


def plasticity_step(
    *,
    engine,
    graph: nx.Graph,
    state: PlasticityState,
    cfg: PlasticityConfig,
    seed_concept: str,
    context: str,
    raw_activation_map: dict[str, float],
    topology_vector: dict,
    baseline_topology_vector: dict | None = None,
    value_core_nodes: set[str] | None = None,
) -> dict:
    """한 stimulus 후의 풀-스택 가소성 update.

    순서 (학술적·실용적 정합):
      1) Sensitization (PE 계산)  — 가소성 게이트 결정
      2) Habituation               — 반복에 의한 반응 감소
      3) Hebbian co-activation     — co-fire pair 강화 (PE-modulated)
      4) (조건부) Homeostatic scaling — runaway 방지
      5) FAB (per-session decay)   — 부정 감정 빠른 fade
      6) (조건부) Consolidation cycle — 매 N epoch sleep replay

    Returns:
        diagnostics dict (모니터링·log용).
    """
    state.step += 1

    # Step 1: Sensitization — uses raw activation to compute PE
    sensitized_map, mean_pe = apply_sensitization(
        state, cfg, raw_activation_map, seed_concept, context
    )

    # Step 2: Habituation (sensitized 위에 적용)
    habituated_map = apply_habituation(state, cfg, sensitized_map)

    # Step 3: Hebbian — PE-modulated. Active subgraph는 habituated 기준.
    hebbian_updated = apply_hebbian(
        graph,
        state,
        cfg,
        habituated_map,
        prediction_error=mean_pe,
        value_core_nodes=value_core_nodes,
    )

    # Step 4: Homeostatic (period-gated)
    scaled = 0
    if state.step % cfg.scaling_period == 0:
        scaled = apply_homeostatic_scaling(graph, cfg)

    # Step 5: FAB
    pos_fade, neg_fade = apply_fading_affect_bias(graph, state, cfg, value_core_nodes)

    # Step 6: Consolidation (period-gated) + strong-experience tagging
    intensity = 0.0
    if baseline_topology_vector is not None:
        # intensity = ||T - T0||_1
        for k in topology_vector:
            intensity += abs(topology_vector[k] - baseline_topology_vector.get(k, 0.0))
    intensity += mean_pe

    state.recent_strong_patterns.append(
        {
            "step": state.step,
            "seed": seed_concept,
            "context": context,
            "activation_map": habituated_map,
            "topology": topology_vector,
            "intensity": intensity,
        }
    )

    consolidation_stats = None
    if state.step % cfg.consolidation_period == 0:
        consolidation_stats = apply_consolidation(graph, state, cfg, value_core_nodes)

    return {
        "step": state.step,
        "mean_pe": mean_pe,
        "intensity": intensity,
        "hebbian_updated_edges": hebbian_updated,
        "homeostatic_scaled_nodes": scaled,
        "fab_pos_fade": pos_fade,
        "fab_neg_fade": neg_fade,
        "consolidation": consolidation_stats,
    }


__all__ = [
    "PlasticityConfig",
    "PlasticityState",
    "apply_hebbian",
    "apply_habituation",
    "apply_sensitization",
    "apply_homeostatic_scaling",
    "apply_fading_affect_bias",
    "apply_consolidation",
    "plasticity_step",
]
