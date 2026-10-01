# Neuroevolution & Evolutionary Algorithms with Neural Networks (2020-2026)

**Consolidated collection** of arXiv papers on neuroevolution, evolutionary algorithms with neural networks, evolutionary neural architecture search, evolution strategies, quality-diversity optimization, and related topics.

## Statistics
- **Total markdown papers**: 202 (organized by year)
- **Date range**: 2020-2026 (through Sept 26, 2026)
- **Plus**: 150+ PDF papers (arXiv originals)

## Papers by Year (Markdown)

| Year | Count | Key Topics |
|------|-------|------------|
| 2020 | 5 | AutoML-Zero, Novelty Search, Differential Evolution for NAS, GA-optimized CNNs |
| 2021 | 7 | Neuroevolution of RNNs, EEEA-Net, EAGAN, NEAT for Autonomous Driving |
| 2022 | 9 | EvoPruneDeepTL, QD-suite, MAP-Elites, Evolution through Large Models |
| 2023 | 18 | EvoPrompting, InstaTune, Quality-Diversity (Human/AI Feedback), Spiking NN Evolution |
| 2024 | 101 | TensorNEAT, LLaMA-NAS, Evolutionary RL (CoERL), ES for DRL, Survey papers, **many NEAT/HyperNEAT** |
| 2025 | 19 | TensorNEAT library, ES at Scale (EGGROLL), LLM Optimization (EA4LLM), GigaEvo, LoongFlow |
| 2026 | 43 | ES for LLM Reasoning, QD for LLM Safety, Microcosmos, IDEAgent, EvoScientist, EvoX |

## Sub-Collections (also available in separate directories)

| Collection | Papers | Focus |
|------------|--------|-------|
| `neuroevolution_ERL/` | 17 | Evolutionary Reinforcement Learning (CoERL, DERL, E-SPL, P²O, etc.) |
| `neuroevolution_LLM_GenAI/` | 46 | Neuroevolution + LLM/GenAI (AlphaEvolve, GEPA, EvoX, etc.) |
| `neuroevolution_NEAT/` | 12 | NEAT, HyperNEAT, ES-HyperNEAT, CPPN, TensorNEAT |
| `neuroevolution_trading/` | 23 | Trading/Finance + EA/Neuroevolution |

## Major Topics Covered

### Core Neuroevolution Algorithms
- NEAT / HyperNEAT / ES-HyperNEAT / CPPN / TensorNEAT (GPU/JAX)
- Novelty Search / Quality-Diversity (MAP-Elites) / MAP-Elites variants
- Deep Neuroevolution (GA for DNN weights)

### Evolutionary Neural Architecture Search (NAS)
- Differential Evolution for NAS
- Multi-objective Evolutionary NAS (MOEA-BUS, EvoNAS)
- Hardware-aware NAS with Evolutionary Algorithms
- LLM-guided NAS (EvoPrompting, LLaMA-NAS)

### Evolution Strategies (ES) at Scale
- ES for Deep RL (linear policies, Atari, MuJoCo)
- ES for LLM Fine-tuning (beyond RL)
- EGGROLL: Low-rank ES for hyperscale
- Quantized ES for Quantized LLMs
- ES vs GRPO for LLM Reasoning

### Quality-Diversity (QD) / MAP-Elites
- QD-suite benchmarks
- MAP-Elites with gradients (PGA-MAP-Elites, DCG-MAP-Elites)
- QD through Human/AI Feedback (QDHF, QDAIF)
- QD for LLM Vulnerability Discovery (Red-teaming)
- MoDA: Mode-conditioned Diversity Alignment

### Evolutionary Reinforcement Learning (ERL)
- Cooperative Coevolutionary RL (CoERL)
- Differentiable Evolutionary RL (DERL)
- Survey: Bridging EA and RL

### Spiking Neural Networks + Evolution
- Evolving SNNs for robot control (blimps)
- Evolutionary feature selection for SNNs
- Neuromorphic hardware (NeuroCoreX, Spark)

### LLM-based Evolutionary Systems
- EvoPrompting: LMs as mutation/crossover operators
- GigaEvo: AlphaEvolve-inspired framework
- LoongFlow: Cognitive Plan-Execute-Summarize
- IDEAgent: Agentic QD for Research Ideas
- EvoScientist: Multi-Agent Evolving AI Scientists
- EvoX: Meta-Evolution for Automated Discovery

## Directory Structure
```
neuroevolution/
├── 2020/ (5 papers)
├── 2021/ (7 papers)
├── 2022/ (9 papers)
├── 2023/ (18 papers)
├── 2024/ (101 papers - includes NEAT/HyperNEAT focus)
├── 2025/ (19 papers)
├── 2026/ (43 papers)
├── neuroevolution_ERL/ (17 papers)
├── neuroevolution_LLM_GenAI/ (46 papers)
├── neuroevolution_NEAT/ (12 papers)
├── neuroevolution_trading/ (23 papers)
├── *.pdf (150+ original arXiv PDFs)
└── README.md
```

## Source
Papers downloaded from Hugging Face Hub (`hf papers read`) and arXiv HTML (`arxiv.org/html/{id}`).
All papers saved as Markdown files organized by year.
