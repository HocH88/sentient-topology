"""
build_large_san_8000.py — Phase 1 200권 corpus 기반 8000 노드 SAN 빌더.

호철님 명시 (2026-05-26):
  D27 (b) 8000 노드 cap (현 5000 → 8000으로 확장)
  D28 (c) Fresh start 200권 corpus
  + NLP scale-up roadmap (Phase 2 500 → 15000 cap, Phase 3 1000 → 25000 cap)

기존 build_large_san_5000.py 격리 보존. 본 빌더는 신규.

Academic Track only. Per the instance isolation principle (paper §6),
private/personal SAN instances live in separate, non-public repositories.
"""
import os
import sys

# build_large_san_5000.py을 재사용 — 같은 alg, 다른 cap
sys.path.insert(0, os.path.dirname(__file__))
from build_large_san_5000 import LargeSanBuilder5000


class LargeSanBuilder8000(LargeSanBuilder5000):
    """200권 corpus용 8000 노드 빌더.

    __init__ 파라미터만 다르고 build_large_san_5000() 메서드 재사용.
    출력 파일명만 다르게 저장.
    """
    def __init__(
        self,
        data_dir: str = "data",
        vocabulary_size: int = 8000,
        window_size: int = 15,
        ppmi_threshold: float = 1.8,
        max_edges: int = 40000,
    ):
        super().__init__(
            data_dir=data_dir,
            vocabulary_size=vocabulary_size,
            window_size=window_size,
            ppmi_threshold=ppmi_threshold,
            max_edges=max_edges,
        )

    def build_large_san_8000(self):
        """alias for parent's build_large_san_5000 (same algorithm, different cap)."""
        self.build_large_san_5000()


if __name__ == "__main__":
    import io
    if hasattr(sys.stdout, 'buffer'):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
        except (ValueError, AttributeError):
            pass

    print("=" * 80)
    print("  Phase 1 — 200 books, 8000 nodes SAN builder")
    print("=" * 80)
    print()

    builder = LargeSanBuilder8000()
    builder.build_large_san_8000()

    # The parent saves to data/large_san_5000.json by default.
    # Move/copy to large_san_8000.json for clarity.
    import shutil
    src = os.path.join("data", "large_san_5000.json")
    dst = os.path.join("data", "large_san_8000.json")
    if os.path.exists(src):
        shutil.copy(src, dst)
        print()
        print(f"  Also saved as: {dst}")
