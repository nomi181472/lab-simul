# NEAT-Based Algorithm Papers (NeuroEvolution of Augmenting Topologies)

Collection of papers on **NEAT** and its variants: **HyperNEAT**, **ES-HyperNEAT**, **CPPN**, **TensorNEAT**, and related neuroevolution algorithms.

## Statistics
- **Total papers**: 12
- **Date range**: 2016-2026
- **Total size**: ~0.9 MB

## Papers by Category

### 🔧 Core NEAT Implementations & Acceleration
| Paper | Year | Focus |
|-------|------|-------|
| **Tensorized NEAT for GPU Acceleration** (2404.01817) | 2024 | Tensorization of NEAT for parallel execution, JAX-based TensorNEAT library, 500x speedup |
| **TensorNEAT: GPU-accelerated Library** (2504.08339) | 2025 | JAX-based library for NEAT, CPPN, HyperNEAT; Gym/Brax/gymnax integration |

### 🌐 HyperNEAT & ES-HyperNEAT (Indirect Encoding)
| Paper | Year | Focus |
|-------|------|-------|
| **ES-HyperNEAT Hyperparameter Optimization** (2609.00449) | 2026 | TPE optimization of ES-HyperNEAT on MNIST (3B search space), transfer to Fashion-MNIST |
| **Population Dynamics with HyperNEAT** (2604.26822) | 2026 | HyperNEAT controllers for ARIEL quadrupeds in MuJoCo, spatial evolutionary dynamics |

### 🎨 CPPN (Compositional Pattern Producing Networks)
| Paper | Year | Focus |
|-------|------|-------|
| **Fourier-CPPNs for Image Synthesis** (1909.09273) | 2019 | Extends CPPNs with frequency modeling for high-detail image synthesis |
| **HyperNetworks** (1609.09106) | 2016 | Related: hypernetworks generate weights for other networks, similar genotype-phenotype mapping |

### 🧬 Neuroevolution Applications
| Paper | Year | Focus |
|-------|------|-------|
| **Neuroevolution of RNN for Spatial/Working Memory** (2102.12638) | 2021 | Evolves RNN weights for robot navigation in triple T-maze (Webots) |
| **Neuroevolutionary Feature Representations for Causal Inference** (2205.10435) | 2022 | GA optimizes neural representations for CATE estimation |
| **Deep Neuroevolution for Land-Use Planning** (2311.12304) | 2023 | Neuroevolution discovers Pareto-optimal land-use policies (Project Resilience) |
| **Single/Multi-Agent Private Active Sensing** (2403.10112) | 2024 | Deep Neuroevolution for collaborative multi-agent hypothesis testing |
| **Novelty Search Makes Evolvability Inevitable** (2005.06224) | 2020 | Theoretical: Novelty Search promotes evolvability in bounded behavior spaces |
| **Microcosmos: Artificial Life for GPU Era** (2607.02954) | 2026 | GPU-based ALife simulator, neuroevolution + QD for embodied evolution |

## NEAT Algorithm Family Tree

```
NEAT (NeuroEvolution of Augmenting Topologies)
│
├── Direct Encoding Variants
│   ├── rtNEAT (Real-time NEAT)
│   ├── NEAT-Python / SharpNEAT / etc.
│   └── TensorNEAT (JAX/GPU accelerated) ← 2404.01817, 2504.08339
│
├── Indirect Encoding (CPPN-based)
│   ├── CPPN (Compositional Pattern Producing Networks)
│   │   └── Fourier-CPPNs ← 1909.09273
│   ├── HyperNEAT (Hypercube-based NEAT)
│   │   ├── ES-HyperNEAT (Evolvable-Substrate) ← 2609.00449
│   │   └── HyperNEAT for Robotics ← 2604.26822
│   └── HyperNetworks (Backprop-trained) ← 1609.09106
│
└── Theoretical Foundations
    ├── Novelty Search ← 2005.06224
    └── Quality-Diversity (MAP-Elites) ← Used in Microcosmos (2607.02954)
```

## Key Technical Concepts

| Concept | Description | Papers |
|---------|-------------|--------|
| **Direct Encoding** | Genome directly specifies network topology/weights | TensorNEAT papers |
| **Indirect Encoding** | Genome specifies pattern (CPPN) that generates network | HyperNEAT, CPPN papers |
| **Substrate** | Geometric space where CPPN queries connections | ES-HyperNEAT |
| **Tensorization** | Reformulate variable topologies as fixed-shape tensors | TensorNEAT (2404.01817, 2504.08339) |
| **Novelty Search** | Reward behavioral novelty instead of fitness | 2005.06224 |
| **Quality-Diversity** | Maintain archive of diverse high-performing solutions | Microcosmos (2607.02954) |

## Software Implementations
| Implementation | Language | Features | Paper |
|----------------|----------|----------|-------|
| **TensorNEAT** | Python/JAX | GPU, NEAT/CPPN/HyperNEAT, Gym/Brax | 2504.08339 |
| **NEAT-Python** | Python | Classic NEAT | Baseline for TensorNEAT |
| **SharpNEAT** | C# | NEAT + HyperNEAT | - |
| **MultiNEAT** | C++/Python | NEAT + HyperNEAT + ES-HyperNEAT | - |

## Related Directories
| Directory | Papers | Focus |
|-----------|--------|-------|
| `neuroevolution/` | 116 | General neuroevolution (2020-2026) |
| `neuroevolution_NEAT/` | 12 | **This directory: NEAT family algorithms** |
| `neuroevolution_LLM_GenAI/` | 46 | NEAT/ES + LLM/Generative AI |
| `neuroevolution_ERL/` | 17 | Evolutionary Reinforcement Learning |
| `neuroevolution_trading/` | 23 | Trading/Finance applications |

## Source
Downloaded from Hugging Face Hub (`hf papers read`) and arXiv HTML (`arxiv.org/html/{id}`).
