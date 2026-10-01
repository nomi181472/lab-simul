# Evolutionary Reinforcement Learning (ERL) Papers

Collection of papers combining **Evolutionary Algorithms / Neuroevolution** with **Reinforcement Learning**.

## Statistics
- **Total papers**: 17
- **Date range**: 2020-2026
- **Total size**: ~1.4 MB

## Core ERL Papers (The Foundation)

| Paper | Year | Method | Key Contribution |
|-------|------|--------|------------------|
| **Bridging EA & RL: Comprehensive Survey** (2401.11963) | 2024 | Survey | **Must-read survey** - 3 research directions: EA-assisted RL, RL-assisted EA, Synergistic EA+RL. GitHub: Awesome-Evolutionary-RL |
| **Evolutionary RL via Cooperative Coevolution (CoERL)** (2404.14763) | 2024 | Cooperative Coevolution | Decomposes policy optimization into subproblems, evolves population per subproblem, uses partial gradients instead of genetic operators |
| **Differentiable Evolutionary RL (DERL)** (2512.13399) | 2025 | Bilevel Meta-Optimization | Evolves reward functions (Meta-Reward) via differentiable meta-optimization using RL as inner loop |

## Three Research Directions (from Survey)

### 1. EA-assisted Optimization of RL
| Paper | Year | Focus |
|-------|------|-------|
| **CoERL** (2404.14763) | 2024 | Cooperative coevolution for policy optimization |
| **Effective Diversity in Population-Based RL** (2002.00632) | 2020 | Diversity via Determinants (DvD) - volume-based diversity |
| **Population-Based RL with Quality-Diversity** (2305.13795) | 2023 | Proximal Policy Gradient Arborescence (PPGA) |
| **MetaDE: Evolving DE by DE** (2502.10470) | 2025 | Meta-evolution of DE hyperparameters for robot control |

### 2. RL-assisted Optimization of EA
| Paper | Year | Focus |
|-------|------|-------|
| **EARLI: RL-initialized GA for Vehicle Routing** (2504.06126) | 2025 | RL agent generates initial solutions for GA |
| **Algorithm Discovery: Evolutionary Search meets RL** (2504.05108) | 2025 | RL fine-tunes LLM (evolutionary search operator) |
| **LIMEN: Discovering RL Interfaces with LLMs** (2605.03408) | 2026 | LLM-guided evolution of RL task interfaces |
| **Representation-Driven RL** (2305.19922) | 2023 | Policy representation for exploration-exploitation |

### 3. Synergistic EA + RL (Tight Integration)
| Paper | Year | Focus |
|-------|------|-------|
| **DERL: Differentiable Evolutionary RL** (2512.13399) | 2025 | Bilevel: evolve reward functions via RL meta-optimization |
| **EvoEmo: Evolved Emotional Policies** (2509.04310) | 2025 | GA evolves emotion policies in MDP for negotiation |
| **EvoRubric: Self-Evolving Rubric-Driven RL** (2605.29847) | 2026 | Co-evolution of response generator + rubric generator |
| **E-SPL: Evolutionary System Prompt Learning for RL** (2602.14697) | 2026 | Joint RL weight updates + evolutionary prompt updates |
| **P²O: Joint Policy and Prompt Optimization** (2603.21877) | 2026 | Alternates policy updates (RL) with prompt evolution (GEPA) |
| **AdvEvo-MARL: Adversarial Co-Evolution in MARL** (2510.01586) | 2025 | Co-evolution of attackers + defenders in multi-agent RL |
| **Execution-Grounded Automated AI Research** (2601.14525) | 2026 | Evolutionary search vs RL for learning from execution feedback |

## Key Algorithms & Frameworks

| Algorithm | Paper | Type |
|-----------|-------|------|
| **CoERL** | 2404.14763 | Cooperative Coevolution + Policy Gradients |
| **DERL** | 2512.13399 | Bilevel Meta-Reward Evolution |
| **PPGA** | 2305.13795 | PPO + Differentiable QD |
| **DvD** | 2002.00632 | Diversity via Determinants |
| **EARLI** | 2504.06126 | RL-initialized GA |
| **E-SPL** | 2602.14697 | RL weights + Evolutionary Prompts |
| **P²O** | 2603.21877 | Policy (RL) + Prompt (Evolution) |
| **AdvEvo-MARL** | 2510.01586 | Adversarial Co-Evolution MARL |
| **MetaDE** | 2502.10470 | Meta-Evolution of DE |
| **EvoRubric** | 2605.29847 | Co-evolutionary Rubric RL |
| **LIMEN** | 2605.03408 | LLM-guided RL Interface Evolution |

## Applications
- **Robotics/Locomotion**: CoERL, MetaDE, PPGA
- **LLM Agent Alignment**: E-SPL, P²O, EvoRubric, AdvEvo-MARL
- **Algorithm Discovery**: Algorithm Discovery paper
- **Trading/Negotiation**: EvoEmo (price negotiation)
- **Vehicle Routing**: EARLI
- **GUI Agents**: Enhancing Visual Grounding
- **RL Interface Design**: LIMEN
- **Automated AI Research**: Execution-Grounded Research

## Related Directories
- `/neuroevolution/` - 116 general neuroevolution papers
- `/neuroevolution_trading/` - 23 trading/finance papers with EA/RL
- `/neuroevolution_ERL/` - This directory (17 core ERL papers)

## Source
Downloaded from Hugging Face Hub (`hf papers read`).
