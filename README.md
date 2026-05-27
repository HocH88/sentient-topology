# Sentient Topology

> Plasticity-Aware Affective Representation in a Humanities-Grounded Sensory Associative Network

[![arXiv](https://img.shields.io/badge/arXiv-pending-b31b1b.svg)](https://arxiv.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This repository accompanies the paper *Sentient Topology*. It contains the code, a lemmatized Sensory Associative Network (SAN) built from a 200-work Project Gutenberg corpus, and scripts to reproduce the figures and the static-framework experiments reported in the paper.

---

## Repository contents

```
.
├── code/
│   ├── san_engine.py              # Propagation + 5-D topology + pure-Python Z2 PH
│   ├── build_large_san.py         # Gutenberg corpus → SAN
│   ├── build_large_san_5000.py    # Scaled SAN builder
│   ├── run_teaser_experiment.py   # Static case study (sunset × {war, garden})
│   ├── baseline_and_ablation.py   # Dimension-drop ablation across 3 seeds
│   ├── render_paper_figures.py    # Figure 5 and Figure 6 (matplotlib)
│   └── validate_latex.py          # Static LaTeX consistency checker
│
├── data/
│   ├── large_san_5000.json        # Pre-built SAN (≈4 MB)
│   └── large_san.json             # Smaller sample SAN
│
├── paper/
│   ├── main.tex                   # arXiv submission source
│   ├── figure5_boundary_heatmap.pdf
│   └── figure6_affect_ordering.pdf
│
├── LICENSE                        # CC BY 4.0
└── README.md                      # this file
```

> **Note**: the Phase 1.5 plasticity engine and the Phase 1 Affect-Trajectory experiments (Stages 0–4) reported in the paper §5 ran on a separate machine and will be added in a follow-up commit as the `plasticity/` and `phase_1.5/` modules. The static framework, the 200-book SAN, and the Figure-rendering pipeline are fully reproducible from what is in this repository.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Reproduce Figure 5 and Figure 6 (uses values from paper §6, no external data)
python code/render_paper_figures.py

# 3. Run the static case study (sunset × {war, garden})
python code/run_teaser_experiment.py

# 4. Run the dimension-drop ablation across 3 seeds
python code/baseline_and_ablation.py

# 5. (Optional) Rebuild the SAN from scratch — downloads 200 Gutenberg books
python code/build_large_san.py
python code/build_large_san_5000.py
```

---

## Headline numbers (paper §5)

| Result | Value |
|---|---|
| Corpus | 200 works, Project Gutenberg public-domain, ~22.3M tokens |
| SAN | 8,000 nodes / 40,000 edges (concept 7,880, sensation 64, association 38, context 18) |
| Inhibitory edges | 782 (between opposing sensations) |
| Context-to-affect seed edges | 86 (rule-augmented) |
| Static ablation | Depth + Boundary jointly carry ~88% of context discrimination |
| Affect trajectory | 7 conditions × 6 stimuli = 42 deterministic propagations |
| Primary observable | Boundary axis (13 of 21 affect pairs systematically differ) |
| Sensitivity (±50%) | 2 brittle channels: SEEKING.concept-bias, CARE.proximity-weight |
| Plasticity | Habituation curve, Fading Affect Bias, surprise sensitization reproduced |

---

## Honest limitations (paper §8)

1. The inhibitory and context-to-affect seed channels are **rule-augmented**, not learned. Their tables are released here for full auditability.
2. Among the five topology dimensions, geodesic Depth and 1-cycle Boundary jointly carry ~88% of context discrimination; Density, Symmetry, and Centrality contribute the remaining ~12%. We keep the 5-D formalism for theoretical completeness.
3. Propagation under GCN-style symmetric normalization is deterministic; cross-stimulus variation (not random-seed replication) is the source of distributional information.
4. We make **no claim of conscious felt experience** and **no claim to validate Panksepp's affective neuroscience framework**. We report the topological signatures that a committed affect-to-SAN parameter mapping produces.
5. Human-subject validation that a topology vector corresponds to a felt sense is the natural follow-up and is future work.
6. Phase 0 (this release) operates on a single English-language corpus. A multi-phase roadmap to 5,000+ works is included in the paper §8.

---

## Instance Isolation

This codebase is designed to host multiple SAN entities (academic experimental instances, dyadic agents in future work, etc.). All instances share the engine code, but per-instance state — SAN weights, plasticity counters, experience logs, auxiliary configurations — is strictly isolated. Feedback never crosses between instances.

---

## License

CC BY 4.0 — please cite the paper if you use this work.

```bibtex
@misc{sentient_topology_2026,
  title  = {Sentient Topology: Plasticity-Aware Affective Representation in a
            Humanities-Grounded Sensory Associative Network},
  author = {Hochanho},
  year   = {2026},
  eprint = {arXiv:pending},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  howpublished = {\url{https://github.com/HocH88/sentient-topology}}
}
```

---

## Contact

Independent Researcher, Seoul, Republic of Korea
ORCID: [0009-0002-3258-2466](https://orcid.org/0009-0002-3258-2466)
