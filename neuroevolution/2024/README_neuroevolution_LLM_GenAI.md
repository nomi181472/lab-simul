# Neuroevolution + LLM / Generative AI Papers (2022-2026)

Collection of papers combining **neuroevolution/evolutionary algorithms** with **Large Language Models (LLMs)** and **Generative AI** for automated discovery, coding agents, prompt optimization, and more.

## Statistics
- **Total papers**: 46
- **Date range**: 2022-2026
- **Total size**: ~4.3 MB

## Major Categories

### 🧬 AlphaEvolve & LLM Evolutionary Coding Agents
| Paper | Year | Focus |
|-------|------|-------|
| **AlphaEvolve** (2506.13131) | 2025 | Google's evolutionary coding agent for scientific/algorithmic discovery |
| **DeepEvolve** (2510.06056) | 2025 | Augments AlphaEvolve with deep research (external knowledge retrieval) |
| **AlphaEvolve for Math** (2511.02864) | 2025 | 67 mathematical problems, rediscovered/improved solutions |
| **AlphaEvolve for MARL** (2602.16928) | 2026 | Discovers new multiagent learning algorithms (VAD-CFR, SHOR-PSRO) |
| **SATLUTION** (2509.07367) | 2025 | Repository-scale code evolution for SAT solving (NP-complete) |
| **CodeEvolve** (2510.14150) | 2025 | Open-source evolutionary coding agent, island-based GA + inspiration crossover |
| **LoongFlow** (2512.24077) | 2025 | Plan-Execute-Summarize paradigm, Multi-Island + MAP-Elites memory |
| **EvoX** (2602.23413) | 2026 | **Meta-Evolution**: evolves both solutions AND search strategies |
| **LEVI** (2605.09764) | 2026 | Stronger search architectures substitute larger LLMs (3.3-6.7x budget reduction) |
| **OpenEvolve Bijection** (2511.20987) | 2025 | OpenEvolve for combinatorial bijection discovery |
| **No Universal Harness** (2607.18235) | 2026 | Systematic decomposition of OpenEvolve components, harness as hyperparameter |
| **Matrix Multiplication ω** (2608.16884) | 2026 | AlphaEvolve improves matrix multiplication exponent bound |

### 🔍 GEPA & Prompt Evolution
| Paper | Year | Focus |
|-------|------|-------|
| **GEPA** (2507.19457) | 2025 | **Genetic-Pareto**: Reflective prompt evolution, outperforms RL (GRPO) with 35x fewer rollouts |
| **E-SPL** (2602.14697) | 2026 | Evolutionary System Prompt Learning: joint RL weight updates + evolutionary prompt updates |
| **P²O** (2603.21877) | 2026 | Joint Policy + Prompt Optimization: alternates policy updates (RL) with prompt evolution (GEPA) |
| **What Makes LLM Good Optimizer** (2604.19440) | 2026 | Trajectory analysis of LLM-guided evolutionary search |

### 📊 Quality-Diversity (QD) + LLMs
| Paper | Year | Focus |
|-------|------|-------|
| **QDAIF** (2310.13032) | 2023 | Quality-Diversity through AI Feedback: LLM evaluates quality/diversity of text |
| **QDHF** (2310.12103) | 2023 | Quality-Diversity through Human Feedback: infers diversity metrics from human judgments |
| **IDEAgent** (2607.22375) | 2026 | Agentic QD Search for Research Idea Generation: manages evolution through lineages |
| **RainbowPlus** (2504.15047) | 2025 | QD search for adversarial prompt generation (red-teaming) |
| **Forty Shades of Blue** (2609.14896) | 2026 | MoDA: Mode-conditioned Diversity Alignment via MARL perspective |
| **QD for LLM Safety** (2606.00801) | 2026 | MAP-Elites for discovering diverse vulnerabilities in LLMs |
| **Heuresis** (2606.25198) | 2026 | Search strategies for autonomous AI research agents (MAP-Elites, Islands, etc.) |
| **CreativeBench** (2603.11863) | 2026 | Benchmarking machine creativity via self-evolving challenges |

### 🧪 Evolution Through Large Models (ELM)
| Paper | Year | Focus |
|-------|------|-------|
| **Evolution Through Large Models** (2206.08896) | 2022 | **Foundational**: LLMs as mutation operators for genetic programming (Sodarace robots) |
| **EvoPrompting** (2302.14838) | 2023 | LLM as adaptive mutation/crossover for Neural Architecture Search |
| **GigaEvo** (2511.17592) | 2025 | Open-source AlphaEvolve-inspired framework (MAP-Elites, DAG pipeline, LLM mutation) |
| **DataEvolve** (2603.14420) | 2026 | AI autonomously evolves pretraining data curation strategies (Darwin-CC dataset) |
| **EvoScientist** (2603.08127) | 2026 | Multi-agent evolving AI scientists with persistent memory |
| **EvoAgent** (2604.20133) | 2026 | Evolvable agent framework with skill learning + hierarchical delegation |

### ⚡ Evolution Strategies (ES) for LLM Optimization
| Paper | Year | Focus |
|-------|------|-------|
| **LLMs as Evolution Strategies** (2402.18381) | 2024 | LLMs as in-context black-box recombination operators (EvoLLM) |
| **EGGROLL** (2511.16652) | 2025 | **Low-rank ES at hyperscale**: 1M+ population, integer pretraining, HFT fine-tuning |
| **ES at Scale: LLM Fine-tuning** (2509.24372) | 2025 | ES for full-parameter LLM fine-tuning, outperforms RL |
| **ES for LLM Reasoning** (2608.27351) | 2026 | ES achieves broader reasoning coverage than GRPO |
| **Beyond Best Guess** (2608.12679) | 2026 | ES improves pass@k, broader solution coverage |
| **Quantized ES** (2602.03120) | 2026 | QES: Fine-tuning quantized LLMs directly in quantized space |
| **ES Catastrophic Forgetting** (2601.20861) | 2026 | ES causes significant forgetting in LLMs vs GRPO |
| **EA4LLM** (2510.10603) | 2025 | Gradient-free full-parameter LLM optimization via EA (0.5B-32B models) |

### 🤝 Co-Evolution & Adversarial Evolution
| Paper | Year | Focus |
|-------|------|-------|
| **AdvEvo-MARL** (2510.01586) | 2025 | Adversarial co-evolution in MARL: attackers vs defenders |
| **EvoEmo** (2509.04310) | 2025 | Evolved emotional policies for adversarial LLM negotiation |
| **EvoRubric** (2605.29847) | 2026 | Self-evolving rubric-driven RL: co-evolves response + rubric generator |
| **Cross-Generational Attack Transfer** (2606.00813) | 2026 | Attack transfer across Gemma model generations via QD evolution |

### 🧠 Neuroevolution + GenAI for Specific Domains
| Paper | Year | Focus |
|-------|------|-------|
| **Differentiable Evolutionary RL (DERL)** (2512.13399) | 2025 | Bilevel: evolves reward functions via differentiable meta-optimization |
| **Microcosmos** (2607.02954) | 2026 | Artificial Life simulator for GPU era: neuroevolution + QD for embodied evolution |
| **Algorithm Discovery** (2504.05108) | 2025 | Evolutionary search + RL fine-tuning of LLM for algorithm design |
| **LIMEN** (2605.03408) | 2026 | LLM-guided evolution of RL interfaces (observations + rewards) |
| **EvoScientist** (2603.08127) | 2026 | Multi-agent AI scientists with persistent memory for end-to-end discovery |

## Key Evolutionary Techniques Enhanced by LLMs/GenAI

| Technique | LLM/GenAI Role | Papers |
|-----------|----------------|--------|
| **Mutation/Crossover** | LLM generates code/prompt mutations | AlphaEvolve, EvoPrompting, ELM, GigaEvo, CodeEvolve |
| **Prompt Evolution** | Evolve prompts as genomes | GEPA, E-SPL, P²O, EvoLLM |
| **Reward Evolution** | LLM designs/evolves reward functions | DERL, EvoRubric |
| **Search Strategy Evolution** | Meta-evolution of search strategies | EvoX, LEVI, LoongFlow |
| **Quality-Diversity Evaluation** | LLM as judge for quality/diversity | QDAIF, QDHF, IDEAgent, RainbowPlus |
| **Fitness Evaluation** | LLM evaluates candidate solutions | AlphaEvolve, EvoScientist, Algorithm Discovery |

## Related Directories
| Directory | Papers | Focus |
|-----------|--------|-------|
| `neuroevolution/` | 116 | General neuroevolution (2020-2026) |
| `neuroevolution_trading/` | 23 | Trading/Finance + EA/Neuroevolution |
| `neuroevolution_ERL/` | 17 | Core Evolutionary RL papers |
| `neuroevolution_LLM_GenAI/` | 46 | **This directory: Neuroevolution + LLM/GenAI** |

## Source
Downloaded from Hugging Face Hub (`hf papers read`) and arXiv HTML (`arxiv.org/html/{id}`).
