"""
sentient_engine.py — Phase 1.5 공유 엔진 wrapper.

Phase 1.1~1.3에서 작성된 `san_engine.SensoryAssociativeNetwork`을
공유 알고리즘(state 없음)으로 노출. Plasticity engine은 이 클래스의
public API(propagate, compute_topological_vector, graph 접근)에만
의존한다.

본 파일은 Phase 1.5의 인스턴스 격리 원칙(§20)에 따라
- 알고리즘만 import / re-export
- state(graph 가중치, 활성도, log)는 SANInstance가 보유

Phase 1.3 san_engine.py가 동일 디렉터리·상위 디렉터리·data/ 어디에
있어도 import 가능하도록 fallback 경로를 시도한다. 없으면 minimal
StubEngine을 제공해 smoke test만 가능.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Phase 1.3 san_engine.SensoryAssociativeNetwork import
# ---------------------------------------------------------------------------

_CANDIDATE_PATHS = [
    # 1) PYTHONPATH 안에 이미 있는 경우
    None,
    # 2) Phase 1.5 폴더와 sibling
    os.path.join(os.path.dirname(__file__), "..", "san_engine.py"),
    os.path.join(os.path.dirname(__file__), "..", "code", "san_engine.py"),
    # 3) D:\Coding\Sentient_AI\code\san_engine.py
    os.path.join(os.path.dirname(__file__), "..", "..", "code", "san_engine.py"),
]


def _load_san_engine_module() -> Any:
    """기존 Phase 1.3 san_engine을 찾아 import. 실패하면 None."""
    try:
        return importlib.import_module("san_engine")
    except ImportError:
        pass

    for candidate in _CANDIDATE_PATHS[1:]:
        if candidate is None:
            continue
        path = os.path.abspath(candidate)
        if not os.path.exists(path):
            continue
        spec = importlib.util.spec_from_file_location("san_engine", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules["san_engine"] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module
    return None


_san_engine_mod = _load_san_engine_module()


if _san_engine_mod is not None and hasattr(_san_engine_mod, "SensoryAssociativeNetwork"):
    # Re-export the real Phase 1.3 engine
    SensoryAssociativeNetwork = _san_engine_mod.SensoryAssociativeNetwork
    USING_STUB_ENGINE = False
else:
    # ---------------------------------------------------------------------
    # Fallback: minimal stub engine so Phase 1.5 modules can be imported
    # and smoke-tested in isolation. NOT a research-grade engine.
    # ---------------------------------------------------------------------
    import math
    import json

    import networkx as nx  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    class SensoryAssociativeNetwork:  # type: ignore[no-redef]
        """Minimal SAN stub for Phase 1.5 smoke testing.

        Honors only the surface API that plasticity.py and san_instance.py
        require. Designed to make `tests/test_plasticity_smoke.py` runnable
        without depending on the real Phase 1.3 implementation.
        """

        def __init__(
            self,
            damping_factor: float = 0.15,
            threshold: float = 0.10,
            max_steps: int = 50,
            tolerance: float = 1e-5,
            type_bias: dict | None = None,
        ) -> None:
            self.graph = nx.Graph()
            self.gamma = damping_factor
            self.theta = threshold
            self.max_steps = max_steps
            self.tolerance = tolerance
            self.type_bias = type_bias or {
                "sensation": 1.0,
                "association": 1.0,
                "context": 1.0,
                "concept": 1.0,
            }

        def add_node(self, node_id: str, node_type: str, description: str = "") -> None:
            self.graph.add_node(node_id, type=node_type, description=description)

        def add_association(
            self,
            u: str,
            v: str,
            weight: float,
            compatibility_context=None,
        ) -> None:
            if compatibility_context is None:
                contexts: list[str] = []
            elif isinstance(compatibility_context, str):
                contexts = [compatibility_context]
            else:
                contexts = list(compatibility_context)
            self.graph.add_edge(u, v, weight=weight, compatible_contexts=contexts)

        @classmethod
        def load_from_json(
            cls,
            path: str,
            damping_factor: float = 0.15,
            threshold: float = 0.10,
            max_steps: int = 50,
            tolerance: float = 1e-5,
            type_bias: dict | None = None,
        ) -> "SensoryAssociativeNetwork":
            san = cls(damping_factor, threshold, max_steps, tolerance, type_bias)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for node in data["nodes"]:
                san.add_node(node["id"], node["type"], node.get("description", ""))
            for edge in data["edges"]:
                san.add_association(
                    edge["source"],
                    edge["target"],
                    edge["weight"],
                    edge.get("compatible_contexts"),
                )
            return san

        def propagate(self, seed_concept: str, context: str) -> dict:
            """Simplified damped clamped spreading activation (Phase 1.3 형식 모사)."""
            if seed_concept not in self.graph:
                raise ValueError(f"Seed '{seed_concept}' not in network.")

            nodes = list(self.graph.nodes())
            idx = {n: i for i, n in enumerate(nodes)}
            n = len(nodes)
            alpha = np.zeros(n)
            seed_i = idx[seed_concept]
            alpha[seed_i] = 1.0
            ctx_i = idx[context] if context in self.graph else None
            if ctx_i is not None:
                alpha[ctx_i] = 1.0

            for _ in range(self.max_steps):
                nxt = np.zeros(n)
                for u in self.graph.nodes():
                    iu = idx[u]
                    if iu == seed_i or iu == ctx_i:
                        nxt[iu] = 1.0
                        continue
                    u_type = self.graph.nodes[u].get("type", "concept")
                    u_bias = self.type_bias.get(u_type, 1.0)
                    inp = 0.0
                    for v in self.graph.neighbors(u):
                        iv = idx[v]
                        if alpha[iv] <= 0.0:
                            continue
                        ed = self.graph.get_edge_data(u, v)
                        compat = ed.get("compatible_contexts", [])
                        if compat and context not in compat:
                            continue
                        du = max(self.graph.degree(u), 1)
                        dv = max(self.graph.degree(v), 1)
                        inp += (ed["weight"] / math.sqrt(du * dv)) * alpha[iv]
                    inp *= u_bias
                    val = (1 - self.gamma) * alpha[iu] + self.gamma * inp
                    val = max(0.0, min(1.0, val))
                    nxt[iu] = val if val >= self.theta else 0.0
                nxt[seed_i] = 1.0
                if ctx_i is not None:
                    nxt[ctx_i] = 1.0
                if np.linalg.norm(nxt - alpha, 1) < self.tolerance:
                    alpha = nxt
                    break
                alpha = nxt
            return {nodes[i]: float(alpha[i]) for i in range(n)}

        def extract_active_subgraph(self, activation_map: dict):
            active = [v for v, a in activation_map.items() if a > self.theta]
            return self.graph.subgraph(active).copy(), active

        def compute_topological_vector(
            self,
            activation_map: dict,
            seed_concept: str,
        ) -> dict:
            sub, active = self.extract_active_subgraph(activation_map)
            num = len(active)
            if num == 0:
                return {
                    "Density (D)": 0.0,
                    "Symmetry (S)": 1.0,
                    "Centrality (C)": 0.0,
                    "Depth (H)": 0.0,
                    "Boundary (B)": 0.0,
                }
            max_e = num * (num - 1) / 2.0
            density = sub.number_of_edges() / max_e if max_e > 0 else 0.0
            try:
                fiedler = nx.algebraic_connectivity(sub, normalized=True)
                symmetry = float(max(0.0, min(1.0, fiedler)))
            except Exception:
                symmetry = 0.0
            if num > 1:
                try:
                    cent = nx.eigenvector_centrality_numpy(sub)
                except Exception:
                    cent = nx.degree_centrality(sub)
                vals = sorted(cent.values())
                idx = np.arange(1, len(vals) + 1)
                s = sum(vals)
                gini = (np.sum((2 * idx - len(vals) - 1) * np.array(vals)) / (len(vals) * s)) if s > 0 else 0.0
            else:
                gini = 0.0
            depth = 0.0
            if seed_concept in active and len(sub) > 1:
                try:
                    lengths = nx.single_source_shortest_path_length(sub, seed_concept)
                    depth = float(max(lengths.values()))
                except Exception:
                    depth = 0.0
            return {
                "Density (D)": float(density),
                "Symmetry (S)": float(symmetry),
                "Centrality (C)": float(gini),
                "Depth (H)": float(depth),
                "Boundary (B)": 0.0,  # stub: 정식 PH는 Phase 1.3 엔진에서
            }

    USING_STUB_ENGINE = True


__all__ = ["SensoryAssociativeNetwork", "USING_STUB_ENGINE"]
