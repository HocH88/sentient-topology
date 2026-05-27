"""
experience_log.py — Phase 1.5 영속 경험 로그 (JSONL).

매 propagation의 (input, T, top_active_nodes, plasticity_delta, intensity)를
날짜별 jsonl 파일에 append. 인스턴스별 격리 (§20).

§16 비-소멸 정책 / strong-experience tagging:
  intensity(e) = ||T(e) - T_0||_1 + max(prediction_error)
  → 상위 quantile 경험은 EWC-style 보호 대상으로 마킹.

본 모듈은 *순수 I/O*. plasticity 메커니즘이 본 로거에 push 하면
SANInstance가 적절한 경로에 영속화.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Iterable


class ExperienceLog:
    """JSONL 기반 append-only 경험 로그."""

    def __init__(self, log_dir: str, instance_id: str = "default"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.instance_id = instance_id

    # ------------------------------------------------------------------ I/O

    def _today_path(self) -> Path:
        date = _dt.date.today().isoformat()
        return self.log_dir / f"{date}.jsonl"

    def append(
        self,
        *,
        seed_concept: str,
        context: str,
        activation_top: list[tuple[str, float]],
        topology_vector: dict,
        plasticity_diagnostics: dict,
        intensity: float,
        extra: dict | None = None,
    ) -> None:
        """한 propagation event 기록."""
        record = {
            "instance_id": self.instance_id,
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "seed_concept": seed_concept,
            "context": context,
            "top_active": [{"node": n, "act": float(a)} for n, a in activation_top],
            "topology": {k: float(v) for k, v in topology_vector.items()},
            "plasticity": plasticity_diagnostics,
            "intensity": float(intensity),
        }
        if extra:
            record["extra"] = extra
        with open(self._today_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ read

    def read_all(self) -> list[dict]:
        """모든 jsonl을 시간순으로 읽어 합친다."""
        records: list[dict] = []
        for path in sorted(self.log_dir.glob("*.jsonl")):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def read_strong(self, top_quantile: float = 0.05) -> list[dict]:
        """상위 quantile intensity 경험만 반환 (strong-experience tag)."""
        records = self.read_all()
        if not records:
            return []
        records.sort(key=lambda r: r.get("intensity", 0.0), reverse=True)
        cutoff = max(1, int(len(records) * top_quantile))
        return records[:cutoff]


__all__ = ["ExperienceLog"]
