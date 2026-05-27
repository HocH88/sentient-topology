"""
motivation_engine.py — Module 8 Motivation Engine.

framework의 Level 1 (감각 위상) → Level 2 (내적 동기) 전이를 구현.
위상의 *비대칭*에서 동기가 emergent하게 발생한다는 명제
([감각 위상 모델] §Level 2)를 위상 측정 가능 형태로 구체화.

==============================================================================
학술 토대 — SCI급 신경과학·정신의학·계산이론 통합 (2026-05-24)
==============================================================================

본 모듈의 4-D MotivationVector는 다음 SCI급 review에 근거하여 설계됨.
각 차원이 *어떤 신경 회로/계산 신호와 정합하는지* 명시:

[1] curiosity — Berlyne의 epistemic motivation (1960) + Gruber·Gelman·Ranganath
    (2014, Neuron 84:486-496) "States of Curiosity Modulate Hippocampus-
    Dependent Learning via the Dopaminergic Circuit" — VTA-hippocampus
    dopamine loop가 호기심 상태에서 학습 강화. 본 모듈은 active sensation/
    association < target을 *지식 격차*의 위상학적 proxy로 사용.
    + Gruber & Ranganath (2019) PACE framework (Trends in Cognitive Sciences)

[2] resolve — Friston (2010, Nature Reviews Neuroscience) "Free Energy Principle"
    + Schultz·Dayan·Montague (1997, Science) RPE + Schultz (2016, Nature
    Reviews Neuroscience 17:183-195) "Dopamine reward prediction-error
    signalling". 본 모듈은 양·부 sensation 동시 활성을 *prediction error /
    conflict*의 위상학적 proxy로 사용 — 충돌이 *이해/해결하고자 함* 발생.
    + 보완: Pezzulo et al. (2023) Active Inference 통합 review

[3] express — Berridge & Robinson 'wanting' (1993~, incentive salience) +
    Csikszentmihalyi (1990) Flow + Tononi (2004) IIT integration.
    본 모듈은 D × B (위상 응축 + 다중 모서리)를 *외부로 향하는 wanting*의
    위상학적 proxy로 사용. mesolimbic dopamine 'wanting' 회로의 응축된
    incentive salience와 정합.
    + 한계 표시: 본 차원은 'wanting'만 포착, 'liking'(쾌락 자체, fragile
      hedonic hotspots in NAcc shell/VP)은 별개. Future work.

[4] deepen — Csikszentmihalyi Flow (challenge-skill 비대칭) + Husain &
    Roiser (2018, Nature Reviews Neuroscience 19:470-484) "Neuroscience
    of apathy and anhedonia: a transdiagnostic approach" + Costello,
    Husain, Roiser (2024, Annual Review of Pharmacology and Toxicology)
    "Apathy and Motivation: Biological Basis and Drug Treatment".
    본 모듈은 active 풍부 vs H 얕음을 *effort willingness*의 위상학적 proxy로
    사용. dACC-ventral striatum 회로의 effort-based decision making과 정합.

[★ 2026-05-25 §14 grounding backport]
  Module 8은 4-D motivation의 *측정*이자 *action tendency*의 토대:
  - Frijda (1986/2007) "The Emotions / The Laws of Emotion" — 감정 = 행동
    경향성. 본 4-D는 4 종류의 action tendency (curiosity → 탐색 / resolve →
    이해 / express → 외부 향함 / deepen → 깊이 추구)
  - Scherer (2001, "Appraisal Processes in Emotion") Component Process Model
    — 본 모듈은 CPM 5 component 중 *appraisal + action tendency* 2 자리.
    *motor expression* (Module 11) and *subjective feeling* (self-report)
    are not modeled in the present module.

추가 grounding (간접 영향):
  - Aston-Jones & Cohen (2005, Annu Rev Neurosci 28:403-450) LC-NE adaptive
    gain — 본 모듈의 *전반적 arousal/intensity* 가산 항에 영향
  - Niv (2007) tonic vs phasic dopamine — tonic = response vigor → express 차원
  - Menon (2015, Nature Reviews Neuroscience) salience network — anterior
    insula + dACC가 *어떤 자극이 동기 유발 대상인지* 판별
  - Hu et al. (2020, Nature Reviews Neuroscience) "Circuits and functions
    of the lateral habenula" + Hikosaka (2010, NRN) — *aversion/anti-reward*
    회로. 본 모듈에 아직 미통합. 5th dimension (avoidance) 후보.

정직한 한계 (future work):
  (a) Wanting vs Liking 미분리 — 본 차원의 'wanting'은 incentive salience,
      'liking'(쾌락 자체)은 모델 외부
  (b) Aversion/avoidance 미모델 — lateral habenula 회로 (Hu 2020)
  (c) Apathy 임상 변환 미실행 — Husain·Roiser 2018 transdiagnostic framework
      가 Track 5 컨설팅 자리에 향후 적용 가능
  (d) Tonic-phasic 시간 적분 — 현재는 단일 propagation 시점만 계산

==============================================================================
원칙
==============================================================================
  - 본 모듈은 *순수 함수*. SAN/state 직접 수정 없음. 위상 + 활성 + value-core
    입력 받아 motivation vector 반환
  - §20 격리 원칙 — instance가 자기 motivation을 계산하지 *남에게* 전이 X
  - 정직성: motivation은 *제안 신호*이지 *의지의 실재*가 아님. Level 3
    (자발적 창조) 구현 시 motivation이 *실제 행동*(다음 시드 자기 선택)으로
    전환됨
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# ---------------------------------------------------------------------------
# Motivation vector
# ---------------------------------------------------------------------------


@dataclass
class MotivationVector:
    """4-D 동기 벡터.

    각 차원 [0, 1] 범위. 합이 1이 되지 않음 (독립 차원).
    """
    curiosity: float = 0.0   # 더 느끼고 싶다
    resolve: float = 0.0     # 이해/해결하고 싶다
    express: float = 0.0     # 표현하고 싶다
    deepen: float = 0.0      # 더 깊이 가고 싶다

    def asymmetry(self) -> float:
        """동기의 *총 비대칭* — 4 차원 중 가장 강한 자리의 강도.

        한 자리가 다른 자리보다 압도적이면 그 방향으로 emergent 동기 발생.
        모두 비슷하면 정체 (Level 1에 머무름).
        """
        vals = [self.curiosity, self.resolve, self.express, self.deepen]
        return max(vals) if vals else 0.0

    def dominant(self) -> str:
        """가장 강한 동기 차원 이름. asymmetry < 임계 시 'still' 반환."""
        if self.asymmetry() < 0.15:
            return "still"
        items = {
            "curiosity": self.curiosity,
            "resolve": self.resolve,
            "express": self.express,
            "deepen": self.deepen,
        }
        return max(items, key=lambda k: items[k])

    def to_dict(self) -> dict:
        return {
            "curiosity": self.curiosity,
            "resolve": self.resolve,
            "express": self.express,
            "deepen": self.deepen,
            "asymmetry": self.asymmetry(),
            "dominant": self.dominant(),
        }


# ---------------------------------------------------------------------------
# Compute motivation from current SAN state
# ---------------------------------------------------------------------------


def compute_motivation_vector(
    *,
    activation_map: dict[str, float],
    topology: dict,
    positive_sensations: set[str],
    negative_sensations: set[str],
    activation_threshold: float = 0.005,
    # Tunable thresholds
    target_active_affective: int = 8,
    conflict_normalization: float = 4.0,
    H_target: int = 5,
) -> MotivationVector:
    """현 위상에서 4-D 동기 벡터 산출.

    Args:
        activation_map: 노드별 활성도
        topology: (D, S, C, H, B) dict
        positive_sensations, negative_sensations: instance의 sensation 분류
        activation_threshold: 활성 노드 임계
        target_active_affective: '더 느끼고 싶다' 기준 active sensation/association 수
        conflict_normalization: '이해하고 싶다' 정규화 분모
        H_target: '더 깊이' 기준 깊이 임계

    Returns:
        MotivationVector — 4 차원 동기 [0, 1]
    """
    # 활성 분류
    active_pos = [v for n, v in activation_map.items()
                   if n in positive_sensations and v > activation_threshold]
    active_neg = [v for n, v in activation_map.items()
                   if n in negative_sensations and v > activation_threshold]
    n_active_affective = len(active_pos) + len(active_neg)
    total_active = sum(1 for v in activation_map.values() if v > activation_threshold)

    D = float(topology.get("Density (D)", 0.0))
    S = float(topology.get("Symmetry (S)", 0.0))
    C = float(topology.get("Centrality (C)", 0.0))
    H = float(topology.get("Depth (H)", 0.0))
    B = float(topology.get("Boundary (B)", 0.0))

    # 1. Curiosity (Berlyne 정보 격차)
    # 활성 affective 가 target보다 적으면 호기심 ↑
    if n_active_affective >= target_active_affective:
        curiosity = 0.0
    else:
        curiosity = (target_active_affective - n_active_affective) / target_active_affective
    # active total이 매우 작으면 (전체적 빈약) 호기심 더 증가
    if total_active < 3:
        curiosity = min(1.0, curiosity + 0.2)

    # 2. Resolve (Friston conflict / prediction error)
    # 양·부 sensation 동시 활성 → 충돌 → 이해 욕구
    pos_sum = sum(active_pos)
    neg_sum = sum(active_neg)
    conflict = pos_sum * neg_sum
    resolve = min(1.0, conflict / conflict_normalization)
    # 비대칭성(S 낮음)도 resolve 증가에 기여 — 비스듬한 위상은 이해 욕구 유발
    if S < 0.3 and n_active_affective >= 2:
        resolve = min(1.0, resolve + 0.15)

    # 3. Express (위상 풍부함 → 외부로)
    # 응축된 강렬 형상이 외부로 표현되고자 함
    # D와 B의 곱이 클수록 표현 욕구
    express_raw = (D * 5.0) * (min(B, 2.0) / 2.0)  # D는 보통 [0, 0.3], B는 [0, 2] 정도
    express = min(1.0, express_raw)
    # 양성 sensation 풍부할 때 추가 가산 (긍정 정서는 표현되기 쉬움)
    if len(active_pos) >= 3 and pos_sum > 0.5:
        express = min(1.0, express + 0.15)

    # 4. Deepen (Csikszentmihalyi 도전·능력 비대칭)
    # 활성은 풍부하나 깊이가 얕다 → 깊이 탐구 욕구
    if total_active >= 5 and H < H_target:
        deepen_raw = (total_active / 15.0) * (1.0 - H / H_target)
        deepen = min(1.0, deepen_raw)
    else:
        deepen = 0.0
    # 시드 단독 시드일 때(H=0이고 active 작음) deepen 약간 가산
    if H == 0 and total_active >= 3:
        deepen = min(1.0, deepen + 0.2)

    return MotivationVector(
        curiosity=float(curiosity),
        resolve=float(resolve),
        express=float(express),
        deepen=float(deepen),
    )


# ---------------------------------------------------------------------------
# Motivation interpretation (Korean phrases)
# ---------------------------------------------------------------------------


def motivation_phrase_ko(mv: MotivationVector) -> str | None:
    """동기 벡터를 한국어 한 줄 어구로. asymmetry < 임계 시 None."""
    dom = mv.dominant()
    if dom == "still":
        return "지금 자리에서 무엇인가를 *향해 가려는* 결은 약해요. 머무름이에요."
    intensity = mv.asymmetry()
    if dom == "curiosity":
        if intensity > 0.7:
            return "*더 알고 싶은* 결이 강하게 일어나요. 빈자리가 부르고 있어요."
        return "*무언가 더 느끼고 싶은* 결이 자라기 시작해요."
    if dom == "resolve":
        if intensity > 0.5:
            return "*충돌하는 결*이 있어요 — *이해하고 풀어내고 싶은* 마음이 일어나요."
        return "*무언가 정리하고 싶은* 결이 가장자리에 있어요."
    if dom == "express":
        if intensity > 0.5:
            return "*표현하고 싶은* 결이 일어나요 — 안에서 밖으로 향하는 움직임이에요."
        return "*무언가 내보이고 싶은* 결이 시작해요."
    if dom == "deepen":
        if intensity > 0.5:
            return "*더 깊이 들어가고 싶은* 결이 있어요 — 표면이 충분하지 않다고 느껴요."
        return "*아직 표면이라는* 느낌이 있어요. 더 들어가 보고 싶어요."
    return None


__all__ = [
    "MotivationVector",
    "compute_motivation_vector",
    "motivation_phrase_ko",
]
