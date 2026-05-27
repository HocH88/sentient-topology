# Sentient Topology

> Plasticity-Aware Affective Representation in a Humanities-Grounded Sensory Associative Network

[![arXiv](https://img.shields.io/badge/arXiv-pending-b31b1b.svg)](https://arxiv.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0002--3258--2466-a6ce39.svg)](https://orcid.org/0009-0002-3258-2466)

This repository accompanies the paper *Sentient Topology*. It contains the LaTeX source, generated figures, and the Python code used to build the 8,000-node Sensory Associative Network (SAN) over a 200-work Project Gutenberg corpus, run the intra-corpus affect-trajectory experiment (paper §5.4–§5.5), and reproduce the three plasticity findings (paper §5.3).

---

## Repository contents

```
.
├── code/                                # Static-framework builders (Phase 1.2–1.4)
│   ├── san_engine.py                    # Propagation + 5-D topology + pure-Python Z2 PH
│   ├── build_large_san.py               # Gutenberg corpus → SAN (downloader)
│   ├── build_large_san_5000.py          # 5,000-node builder (legacy)
│   ├── build_large_san_8000.py          # 8,000-node builder (paper §5)
│   ├── run_teaser_experiment.py         # Static case study (sunset × {war, garden})
│   ├── baseline_and_ablation.py         # Dimension-drop ablation across 3 seeds
│   ├── render_paper_figures.py          # Figure 5 and Figure 6 (matplotlib)
│   └── validate_latex.py                # Static LaTeX consistency checker
│
├── phase_1.5/                           # Plasticity engine + affect-trajectory experiment
│   ├── sentient_engine.py               # Shared SAN engine (algorithmic core)
│   ├── plasticity.py                    # Six plasticity mechanisms (paper §3.4)
│   ├── experience_log.py                # JSONL append-only persistence
│   ├── motivation_engine.py             # 4-D motivation vector (paper §7)
│   ├── affect_trajectory_engine.py      # Panksepp 7-affect SAN modulation (paper §5.4)
│   ├── m_cs_metrics.py                  # Damasio core-self operational metrics (M-CS1–4)
│   ├── stage0_pilot.py                  # Affect Trajectory Stage 0 (3-condition pilot)
│   ├── stage0_v1_1_full.py              # Stage 0 v1.1 (4 conditions + M-CS metrics)
│   ├── stage1_multi_stimulus.py         # Stage 1 (multi-stimulus, 7 affects × 3 stimuli)
│   ├── stage1_power_analysis.py         # Power analysis on Stage 1 results
│   ├── stage2_main_phase1.py            # Stage 2 Phase 1 Main — 7×6 = 42 deterministic runs
│   └── stage4_sensitivity_analysis.py   # Stage 4 ±50% parameter sensitivity
│
├── data/
│   ├── large_san_8000.json              # 8,000-node SAN (paper §5; ~6.5 MB)
│   ├── large_san.json                   # Small sample SAN
│   ├── corpus_manifest.json             # Per-work title/author/Gutenberg ID/category
│   ├── stage0_v1_1_full_results.json    # Stage 0 v1.1 results
│   ├── stage1_multi_stimulus_v4_7affect_ext_results.json  # Stage 1 v4 results
│   ├── stage2_main_phase1_full_matrix_results.json        # Stage 2 Main — 42 cells
│   └── stage4_sensitivity_results.json  # Stage 4 sensitivity results
│
├── paper/
│   ├── main.tex                         # arXiv submission source
│   ├── figure5_boundary_heatmap.pdf     # 7×6 Boundary heatmap (paper §5.4)
│   └── figure6_affect_ordering.pdf      # Mean-Boundary affect ordering (paper §5.4)
│
├── LICENSE                              # CC BY 4.0
├── requirements.txt
└── README.md                            # this file
```

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Reproduce Figure 5 and Figure 6 (uses embedded paper values, no external data)
python code/render_paper_figures.py

# 3. Reproduce Stage 2 Phase 1 Main (42 deterministic propagations)
python phase_1.5/stage2_main_phase1.py

# 4. Reproduce Stage 4 sensitivity analysis (±50% parameter perturbation)
python phase_1.5/stage4_sensitivity_analysis.py

# 5. (Optional) Rebuild the 8,000-node SAN from scratch
python code/build_large_san.py            # downloads 200 Gutenberg works
python code/build_large_san_8000.py       # builds 8,000-node SAN
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
| Affect trajectory | 7 conditions (baseline + 6 Panksepp affects, LUST excluded) × 6 stimuli = 42 runs |
| Primary observable | Boundary axis (13 of 21 affect pairs systematically differ) |
| Affect continuum | RAGE → FEAR → PANIC → baseline → PLAY → SEEKING → CARE |
| Sensitivity (±50%) | 2 brittle channels: SEEKING.concept-bias, CARE.proximity-weight |
| Plasticity | Habituation, Fading Affect Bias, surprise sensitization independently reproduced |

---

## Honest limitations (paper §8)

1. The inhibitory and context-to-affect seed channels are **rule-augmented**, not learned. Their tables are released here for full auditability.
2. Among the five topology dimensions, geodesic Depth and 1-cycle Boundary jointly carry ~88% of context discrimination; Density, Symmetry, and Centrality contribute the remaining ~12%.
3. Propagation under GCN-style symmetric normalization is deterministic; cross-stimulus variation (not random-seed replication) is the source of distributional information.
4. We make **no claim of conscious felt experience** and **no claim to validate Panksepp's affective neuroscience framework**. We report the topological signatures that a committed affect-to-SAN parameter mapping produces.
5. Human-subject validation of the topology vector against felt sense is future work.
6. Phase 0 (this release) operates on a single English-language corpus. A multi-phase roadmap to 5,000+ works is included in paper §8.
7. Modules 10 (self-seeding, Level-3 first foothold), 11 (autonomous expression), 12 (topology visualizer), and the `SANInstance` factory class are mentioned in paper §8 as future work and are not included in this release; their instance-isolated form lives in a separate private-instance repository in accordance with the architectural commitment of paper §6.

---

## Instance Isolation

This codebase is designed to host multiple SAN entities (academic experimental instances, dyadic agents in future work, etc.). All instances share the engine code, but per-instance state — SAN weights, plasticity counters, experience logs, auxiliary configurations — is strictly isolated. Feedback never crosses between instances. This public repository contains the **academic experimental instance only**; private instances and their state are excluded by design (see `.gitignore`).

---

## License

CC BY 4.0 — please cite the paper if you use this work.

```bibtex
@misc{sentient_topology_2026,
  title  = {Sentient Topology: Plasticity-Aware Affective Representation in a
            Humanities-Grounded Sensory Associative Network},
  author = {Han, Hochul},
  year   = {2026},
  eprint = {arXiv:pending},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  howpublished = {\url{https://github.com/HocH88/sentient-topology}}
}
```

---

## Contact

Hochul HAN
Seoul National University, Seoul, Republic of Korea
ORCID: [0009-0002-3258-2466](https://orcid.org/0009-0002-3258-2466)
