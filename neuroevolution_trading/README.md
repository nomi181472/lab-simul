# Neuroevolution & Evolutionary Methods in Trading/Finance (2020-2026)

Collection of arXiv papers applying evolutionary algorithms, neuroevolution, evolution strategies, genetic algorithms, and quality-diversity methods to trading, portfolio optimization, and quantitative finance.

## Statistics
- **Total papers**: 23
- **Date range**: 2023-2026
- **Total size**: ~1.4 MB

## Papers by Category

### 🧬 Evolutionary/Genetic Algorithms for Trading
| Paper | Year | Method | Focus |
|-------|------|--------|-------|
| **EvolveTrade** (2609.17632) | 2026 | Self-evolving LLM policy | LLM trading agents with evolving system prompts |
| **ContestTrade** (2508.00554) | 2025 | Multi-agent internal competition | Competitive mechanism for robust trading |
| **TradingGroup** (2508.17565) | 2025 | Multi-agent + self-reflection | Reflection + data synthesis pipeline |
| **QuantAgent** (2509.09995) | 2025 | Multi-agent LLM for HFT | Indicator/Pattern/Trend/Risk agents for HFT |
| **ATLAS** (2510.15949) | 2025 | Adaptive-OPRO prompt optimization | Dynamic prompt optimization for trading |
| **EvoAgent** (2604.20133) | 2026 | Evolvable agent + skill learning | Structured skill evolution for trading |

### 📈 Evolutionary Portfolio Optimization
| Paper | Year | Method | Focus |
|-------|------|--------|-------|
| **Multi-Objective Portfolio Opt** (2507.16717) | 2025 | Gradient descent (benchmarks vs EA) | CVaR, Sharpe, constraints, UCITS |
| **Generative Meta-Learning QD Portfolio** (2307.07811) | 2023 | Generative + Quality-Diversity | Robust ensemble portfolios |
| **Deep RL vs MVO** (2602.17098) | 2026 | DRL (SAC/PPO) vs Mean-Variance | Portfolio allocation comparison |

### 🤖 Neuroevolution / Evolution Strategies in Finance
| Paper | Year | Method | Focus |
|-------|------|--------|-------|
| **EGGROLL** (2511.16652) | 2025 | Low-rank Evolution Strategies | **Section M: HFT agent fine-tuning for PnL** |
| **Deep Q-Learning HFT** (2311.10718) | 2023 | DRL (DQN) | Statistical arbitrage in HFT |
| **Deep RL Trader (No Offline)** (2303.00356) | 2023 | Double DQN + Fast Learning Nets | Fully online trading on Cardano |
| **Neural Network Algo Trading** (2508.02356) | 2025 | Multi-timeframe NN | Crypto HFT execution |
| **AI-Powered Energy Trading** (2407.19858) | 2024 | HMM + NN + Black-Litterman | Energy sector, QuantConnect |

### 🔬 LLM Trading Agents (Evolutionary/Adaptive)
| Paper | Year | Method | Focus |
|-------|------|--------|-------|
| **Orchestration Framework** (2512.02227) | 2025 | Multi-agent orchestration | Planner, alpha, risk, portfolio, execution agents |
| **TradeTrap** (2512.02261) | 2025 | Stress-testing framework | Robustness evaluation of LLM traders |
| **QuantCode-Bench** (2604.15151) | 2026 | Code generation benchmark | Backtrader strategy generation |
| **Language Model Guided RL** (2508.02366) | 2025 | LLM → RL hybrid | LLM strategies guiding RL agents |
| **Bayesian Robust Trading** (2601.17008) | 2026 | Bayesian game + GAN | Adversarial macro-conditioned simulation |
| **AI-Trader** (2512.10971) | 2025 | Live benchmark | Real-time evaluation across markets |
| **Representation Signatures** (2605.28850) | 2026 | TradeArena testbed | Risk-feedback alignment analysis |
| **TRADES** (2502.07071) | 2025 | Diffusion models | Realistic LOB market simulation |

## Key Evolutionary Techniques Applied

| Technique | Papers |
|-----------|--------|
| **Evolution Strategies (ES)** | EGGROLL (HFT fine-tuning) |
| **Genetic Algorithms** | ContestTrade, TradingGroup internal contest |
| **Quality-Diversity (MAP-Elites)** | Generative Meta-Learning QD Portfolio |
| **Neuroevolution (weight evolution)** | Deep Q-Learning HFT, Deep RL Trader |
| **Self-Evolving Prompts/Policies** | EvolveTrade, ATLAS (Adaptive-OPRO), EvoAgent |
| **Multi-Agent Evolution** | ContestTrade, TradingGroup, QuantAgent, Orchestration |

## Markets & Assets Covered
- **Equities**: US stocks, A-shares, NIFTY 50, energy sector
- **Cryptocurrency**: Bitcoin, crypto markets (minute data)
- **HFT/LOB**: Limit order book, high-frequency trading, statistical arbitrage
- **Portfolio**: Multi-asset, multi-objective (Sharpe, CVaR, drawdown)
- **Derivatives**: Nasdaq futures

## Directory Structure
```
neuroevolution_trading/
├── *.md (23 papers)
└── README.md
```

## Source
Downloaded from Hugging Face Hub (`hf papers read`) and arXiv HTML (`arxiv.org/html/{id}`).
