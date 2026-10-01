Title: Language Model Guided Reinforcement Learning in Quantitative Trading

URL Source: https://arxiv.org/html/2508.02366

Published Time: Tue, 28 Oct 2025 00:45:43 GMT

Markdown Content:
###### Abstract

Algorithmic trading requires short-term tactical decisions consistent with long-term financial objectives. Reinforcement Learning (RL) has been applied to such problems, but adoption is limited by myopic behaviour and opaque policies. Large Language Models (LLMs) offer complementary strategic reasoning and multi-modal signal interpretation when guided by well-structured prompts.

This paper proposes a hybrid framework in which LLMs generate high-level trading strategies to guide RL agents. We evaluate (i) the economic rationale of LLM-generated strategies through expert review, and (ii) the performance of LLM-guided agents against unguided RL baselines using Sharpe Ratio (SR) and Maximum Drawdown (MDD).

Empirical results indicate that LLM guidance improves both return and risk metrics relative to standard RL.

I Introduction
--------------

Algorithmic trading requires short-term execution aligned with long-term financial objectives, and accounts for over 60% of U.S. equity volume, particularly in high-frequency trading (HFT)[[1](https://arxiv.org/html/2508.02366v3#bib.bib1)]. Long-horizon strategies often build on econometric models such as the Fama–French five-factor framework[[2](https://arxiv.org/html/2508.02366v3#bib.bib2)], while short-horizon approaches exploit transient inefficiencies through momentum and mean reversion[[3](https://arxiv.org/html/2508.02366v3#bib.bib3)]. The most extreme form is HFT, where firms execute thousands of transactions per second by exploiting order book imbalances[[1](https://arxiv.org/html/2508.02366v3#bib.bib1)].

Modern trading systems leverage machine learning (ML) to process structured and unstructured data streams. Growth is driven by advances in electronic infrastructure, compute power, and the proliferation of large high-resolution financial data[[4](https://arxiv.org/html/2508.02366v3#bib.bib4)].

RL formalises trading as sequential decision making[[5](https://arxiv.org/html/2508.02366v3#bib.bib5)] and has shown promise through methods such as Deep Q-Networks (DQN) and actor–critic variants[[3](https://arxiv.org/html/2508.02366v3#bib.bib3), [6](https://arxiv.org/html/2508.02366v3#bib.bib6)]. Yet practical adoption is hindered by sparse rewards, credit assignment issues, and opaque policies, which reduce trust in high-stakes financial settings[[7](https://arxiv.org/html/2508.02366v3#bib.bib7)].

LLMs offer complementary strengths. They can understand heterogeneous signals and generate rationale explanations[[8](https://arxiv.org/html/2508.02366v3#bib.bib8)], yet remain limited by fixed knowledge cut-offs and an inability to adapt to changing environments in real time[[9](https://arxiv.org/html/2508.02366v3#bib.bib9)]. They are also fragile to prompt design and may produce plausible but invalid outputs[[10](https://arxiv.org/html/2508.02366v3#bib.bib10), [11](https://arxiv.org/html/2508.02366v3#bib.bib11)]. Advances in prompt engineering and human-in-the-loop (HITL) methods partially mitigate these risks[[12](https://arxiv.org/html/2508.02366v3#bib.bib12)].

This paper addresses these limitations by proposing a hybrid architecture that integrates LLM strategic guidance with RL execution, and sets the following objectives.

##### Objectives

(1) Design an approach for using LLMs to generate trading strategies that are economically grounded, assessed by expert review; (2) evaluate whether LLM guidance improves RL performance, measured by SR and MDD, without altering the base RL architecture.

These objectives directly inform the main contributions of the paper.

##### Contributions

(1) A structured prompting framework that produces domain-grounded strategies assessed by expert review and financial metrics; (2) a modular LLM+RL design in which the LLM contributes a single uncertainty-weighted scalar appended to the RL observation space, improving out-of-sample performance without modifying the RL algorithm.

### I-A Related Work

A Trading DQN (TDQN) for stock trading is introduced in [[3](https://arxiv.org/html/2508.02366v3#bib.bib3)]. The authors leverage a DDQN algorithm to mitigate overestimation bias and stabilize learning in stochastic market environments. A key aspect of their RL approach was the discretization of actions and the enforcement of capital constraints, which helped prevent infeasible or overleveraged actions.

The FinRL framework[[6](https://arxiv.org/html/2508.02366v3#bib.bib6)] introduced benchmark environments and unified APIs for financial RL research that features realistic data simulation. It includes a wide array of backtests using standard RL algorithms and focuses on two primary objectives in algorithmic trading: maximizing return (measured by cumulative return and SR) and minimizing risk (measured by MDD and return variance). The framework supports experiments in single-stock trading, multi-stock trading, and portfolio allocation.

In the survey[[13](https://arxiv.org/html/2508.02366v3#bib.bib13)], the authors identified key limitations in deep reinforcement learning (DRL), including Bellman backup instability and credit assignment failures. The authors recommend hierarchical reinforcement learning (HRL) or recurrent extensions to address the lack of long-range temporal dependencies.

For LLMs in finance, [[14](https://arxiv.org/html/2508.02366v3#bib.bib14)] showed that ChatGPT outperforms traditional sentiment lexicons on forward-looking financial news, but lacks temporal awareness and numerical reasoning capabilities.

The FINMEM framework in[[12](https://arxiv.org/html/2508.02366v3#bib.bib12)] combines structured memory with LLM-based decision modules. FINMEM’s layered memory integrates recent news, financial reports, and long-term statements to inform trade recommendations, leveraging retrieval-augmented generation (RAG). Their architecture stores experiences in a vector database, which are retrieved and ranked using a decay mechanism that emulates a human’s memory decay.

Prompting practices have been extensively surveyed in [[11](https://arxiv.org/html/2508.02366v3#bib.bib11)], which categorizes strategies into instruction-based, example-based, reasoning-based, and critique-based families. The study highlights self-refinement and constraint enforcement as key mechanisms for improving robustness. It also shows that minor variations in prompt wording can systematically influence model behavior.

In [[15](https://arxiv.org/html/2508.02366v3#bib.bib15)], Chain-of-Thought (CoT) prompting was shown to significantly enhance LLM reasoning. Self-improvement frameworks iteratively refine rationale quality, while problem decomposition and model fine-tuning help address complex tasks. Without structured prompting techniques, however, LLMs continue to struggle with planning problems.

II Methodology
--------------

This section outlines the methodology developed to evaluate the integration of LLMs into RL agents. The proposed hybrid framework mirrors the top-down decision-making structures common in financial institutions.

Two experiments were conducted to address the research objectives. All LLM strategies were validated through historical backtesting and expert review prior to their integration with RL agents.

### II-A Benchmark Environment

For our benchmark, we utilized the trading system introduced in[[3](https://arxiv.org/html/2508.02366v3#bib.bib3)]. This benchmark includes a clearly defined environment, consistent state and reward functions, and extensive empirical results.

We replicated the core experimental settings, including the asset universe, data preprocessing, and evaluation metrics. Our reproduction yielded comparable statistically significant SR and MDD metrics.

We note minor discrepancies that affect financial interpretability: their cumulative returns are arithmetically summed rather than geometrically compounded, and their SR assumes a zero risk-free rate. For consistency, we preserve these conventions throughout our experiments.

### II-B Experiment 1: LLM Trading Strategy Generation

![Image 1: Refer to caption](https://arxiv.org/html/2508.02366v3/prompt-tuning.png)

Figure 1: Prompt Tuning Workflow.

This experiment addressed Objective 1 by introducing two LLM agents: the Strategist Agent and the Analyst Agent. The Strategist Agent generates global trading policies using a financial dataset. The Analyst Agent processes news and distills it into signals to inform the Strategist Agent. This experiment serves as the foundation for Experiment 2.

A strategy defines a directional action (dir​(π g)\text{dir}(\pi^{g}), where 1 = LONG and 0 = SHORT) and an associated confidence score (μ conf\mu_{\text{conf}}, from 1 to 3). Each strategy is accompanied by an explanation and a weighted set of features. Strategies are generated on a monthly basis using time-aligned data.

#### II-B1 Data and Feature Engineering

To support strategy generation, the LLM agents consumed a multi-modal dataset spanning 2012–2020, aligning with the benchmark’s dataset dates in [[3](https://arxiv.org/html/2508.02366v3#bib.bib3)]. The dataset includes traditional Open, High, Low, Close, and Volume (OHLCV) price data, which we augmented with four additional categories of financial signals: market data, fundamentals, technical analytics, and alternative data[[16](https://arxiv.org/html/2508.02366v3#bib.bib16)]. These collectively define the LLM’s context.

Market data was sourced from Interactive Brokers 1 1 1[https://www.interactivebrokers.com/api](https://www.interactivebrokers.com/api) and iVolatility 2 2 2[https://www.ivolatility.com/data-cloud-api/](https://www.ivolatility.com/data-cloud-api/), including OHLCV time series, SPX and NDX index returns, the VIX index, and Options implied volatility (IV). Fundamental data, comprising firm-level financial ratios and macroeconomic indicators (e.g., GDP, PMI, interest rates), were retrieved via SEC-API 3 3 3[https://sec-api.io/](https://sec-api.io/) and the FRED API 4 4 4[https://fred.stlouisfed.org/docs/api/fred/](https://fred.stlouisfed.org/docs/api/fred/). Analytics features were computed using TA-Lib 5 5 5[https://ta-lib.org/](https://ta-lib.org/), applying rolling windows to extract indicators. Alternative data consisting of news headlines were collected from Alpaca 6 6 6[https://alpaca.markets/docs/api-documentation/](https://alpaca.markets/docs/api-documentation/) and processed into explanatory factors using few-shot LLM prompting, following the LLMFactor framework from[[10](https://arxiv.org/html/2508.02366v3#bib.bib10)]. The data was aligned by timestamp.

#### II-B2 LLM Model

We used OpenAI’s GPT-4o Mini for its strong performance in financial reasoning and cost-efficiency[[14](https://arxiv.org/html/2508.02366v3#bib.bib14), [12](https://arxiv.org/html/2508.02366v3#bib.bib12), [15](https://arxiv.org/html/2508.02366v3#bib.bib15)]. The model supports a 128k token context window with a 16k maximum prompt size, enabling the use of detailed prompts with embedded context memory, reasoning chains, and previous reflection results.

#### II-B3 Prompt Engineering Methodology

The objective was to construct a prompt that generalized across equities and regimes while remaining interpretable. We proceeded in three stages: (i) baseline specification, (ii) incorporation of expert exemplars with feature pruning, and (iii) iterative refinement comprising two distinct processes: a _Writer–Trainer_ process (feature and instruction selection) and a _Writer–Judge_ process (prompt quality and rationale critique) with regret minimization.

To manage computational cost, we did not tune on full eight-year history and instead randomly sampled non-overlapping one-year intervals from the dataset per instrument, repeated five times, and used these subsets for creating candidate prompts.

##### Baseline

We began with a minimal prompt that exposed only raw OHLCV data and a small set of technical indicators: Simple Moving Averages (SMA; 20/50/200 periods), Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD). This configuration reflected common trading heuristics in both algorithmic and retail practice[[17](https://arxiv.org/html/2508.02366v3#bib.bib17), [18](https://arxiv.org/html/2508.02366v3#bib.bib18)] and served to set the candidate prompt and V∗V^{*} target.

##### Writer–Trainer, and Writer–Judge Loops

To refine the information set, we introduced a _Writer–Trainer_ process that reflected on expert trade exemplars derived from HITL feedback and a heuristic algorithm approximating these. Candidate features were selected by top quartile through ranking importance on a three-point Likert scale (low/medium/high), and rationales were clustered with only the ten unique ones being selected to become instructions for the candidate prompt.

Prompt refinement was then conducted through a heuristic regret–minimization loop, inspired by [[19](https://arxiv.org/html/2508.02366v3#bib.bib19)], with pruning and rationale discovery as the inner stage and backtesting as the outer stage. At each iteration t t, a _writer_ generated a candidate prompt π t\pi_{t} conditioned on the KB and the retained features and rationales. The candidate π t\pi_{t} was evaluated through backtesting to obtain its SR, denoted V π t V^{\pi_{t}}. A _judge_ then assessed the prompt for its rationale and suggested alternative instructions or feature combinations for the next iteration, and stored in the Knowledge Base (KB)

We adopted a regret heuristic to guide exploration:

ℛ(T)=𝔼[∑t=1 T(V∗−V π t)|ℋ t],\mathcal{R}(T)\;=\;\mathbb{E}\!\left[\sum_{t=1}^{T}\bigl(V^{*}-V^{\pi_{t}}\bigr)\,\middle|\,\mathcal{H}_{t}\right],(1)

where V∗V^{*} denotes the best SR defined by the baseline prompt or the market’s SR for the whole dataset (initialized as V∗=max⁡{V baseline, 0.8}V^{*}=\max\{V_{\text{baseline}},\,0.8\}), V π t V^{\pi_{t}} is the SR of the current candidate, and ℋ t\mathcal{H}_{t} represents the KB at iteration t t (features, rationales, and prior outcomes). To manage the LLM’s finite context window, the loop retained only an _extremes memory_ within ℋ t\mathcal{H}_{t}, consisting of the best- and worst-performing prompts with their associated features and instructions. These extremes biased subsequent generations toward more promising candidates.

Iterations terminated when either (i) ℛ​(T)≤0\mathcal{R}(T)\leq 0 (no expected improvement) or (ii) the maximum T T iterations elapsed. The final augmented baseline prompt, generalized across equities and regimes through this process, was then subjected to three additional refinements.

##### Prompt Improvement 1 – In-Context Memory (ICM)

Inspired by[[12](https://arxiv.org/html/2508.02366v3#bib.bib12)], we introduced a memory buffer that stores the most recent strategies π t\pi_{t} observed prior to time T T. Each stored strategy π T−1 g\pi^{g}_{T-1} is represented by its directional action, weighted features, and rationale. Within the prompt, these prior strategies are recalled in-context and reflected on, enabling the current strategy π T g\pi^{g}_{T} to be conditioned on past decisions. This reflection mechanism reduces the persistence of suboptimal strategies.

##### Prompt Improvement 2 – Instruction Decomposition

To enhance reasoning, instructions and their associated drivers were decomposed[[11](https://arxiv.org/html/2508.02366v3#bib.bib11)] into six feature groups: stock, technical, fundamental, macroeconomic, options, and prior strategy reflection[[9](https://arxiv.org/html/2508.02366v3#bib.bib9)]. Each group supplied few-shot examples and domain-specific heuristics to elicit CoT, requiring the model to reason sequentially across domains and prior strategies.

##### Prompt Improvement 3 – News Factors

Unstructured news data was introduced via the analyst agent, which applied instruction-decomposition factor extraction[[10](https://arxiv.org/html/2508.02366v3#bib.bib10)]. Entities and timestamps were anonymized, disabling the LLM’s memory to prevent leakage[[20](https://arxiv.org/html/2508.02366v3#bib.bib20)]. Extracted news factors were ranked and integrated alongside numerical indicators.

The final system integrates selected numerical and textual signals into a global strategy policy π g\pi^{g}. All prompt iterations used in Experiment 1 are summarized in Table[I](https://arxiv.org/html/2508.02366v3#S2.T1 "TABLE I ‣ Prompt Improvement 3 – News Factors ‣ II-B3 Prompt Engineering Methodology ‣ II-B Experiment 1: LLM Trading Strategy Generation ‣ II Methodology ‣ Language Model Guided Reinforcement Learning in Quantitative Trading").

TABLE I: Prompt Versions Used in Experiment 1

#### II-B4 Parameters and Evaluation

Prompt tuning used temperature 0.7 0.7 following prior work[[12](https://arxiv.org/html/2508.02366v3#bib.bib12), [14](https://arxiv.org/html/2508.02366v3#bib.bib14)], with frequency penalty 1.0 1.0 and presence penalty 0.25 0.25. These values discouraged verbatim reuse from the KB or ICM while preserving exploratory diversity.

For strategy generation, temperature was set to 0 with fixed seed 49 49 for reproducibility. Strategies were produced on a monthly cadence (20 trading days), aligning with common guidance/rebalancing cycles and remaining tractable given LLM inference cost.

At most three refinement iterations were permitted (T≤3 T\leq 3). Convergence was declared when the regret ℛ​(T)\mathcal{R}(T) approached zero or when the SR exceeded the initial threshold max⁡{V baseline, 0.8}\max\{V_{\text{baseline}},\,0.8\}. The procedure was repeated five times with discretionary HITL adjustments between runs. The iteration count balanced methodological tractability against computational cost.

All technical indicators used a 20-trading-day rolling window with standard TA-Lib defaults (e.g., 14-day RSI).

##### Quantitative Metrics

We evaluate LLM-generated strategies using three complementary metrics: risk-adjusted returns, model confidence, and model uncertainty.

The SR serves as the core risk-adjusted returns metric:

SR=𝔼​[R t−R f]σ R.\text{SR}=\frac{\mathbb{E}[R_{t}-R_{f}]}{\sigma_{R}}.(2)

where R t R_{t} is the portfolio return, R f R_{f} is the risk-free rate, and σ R\sigma_{R} is the return volatility. SR also serves as a proxy for the LLM’s financial reasoning[[14](https://arxiv.org/html/2508.02366v3#bib.bib14), [12](https://arxiv.org/html/2508.02366v3#bib.bib12)]. To ensure comparability across different periods for the daily returns, we annualize the SR to 252 trading days per year: Annualized SR=SR⋅252\text{Annualized SR}=\text{SR}\cdot\sqrt{252}.

As a proxy for prompt quality we compute the Perplexity (PPL)[[21](https://arxiv.org/html/2508.02366v3#bib.bib21)] over the LLM-generated strategies:

PPL=exp⁡(−1 N​∑t=1 N log⁡p​(w t∣w<t)).\text{PPL}=\exp\left(-\frac{1}{N}\sum_{t=1}^{N}\log p(w_{t}\mid w_{<t})\right).(3)

where p​(w t∣w<t)p(w_{t}\mid w_{<t}) denotes the conditional token probability. Lower values indicate higher quality.

To complement this, we report token-level entropy H LLM H_{\text{LLM}}, approximated using top-k k distributions:

H LLM=1 N​∑t=1 N(∑v∈V k−p t​(v)​log⁡p t​(v)−p tail,t​log⁡p tail,t).H_{\text{LLM}}=\frac{1}{N}\sum_{t=1}^{N}\left(\sum_{v\in V_{k}}-p_{t}(v)\log p_{t}(v)-p_{\text{tail},t}\log p_{\text{tail},t}\right).(4)

where V k V_{k} denotes the top-k k token set and p tail,t p_{\text{tail},t} represents the unobserved probability mass[[22](https://arxiv.org/html/2508.02366v3#bib.bib22)]. In our experiments, k=5 k=5. Lower entropy indicates greater decisiveness, whereas higher values suggest uncertainty.

Together, PPL and H LLM H_{\text{LLM}} enable a measurement of prompt quality and strategy confidence.

##### Qualitative Evaluation

Qualitative assessment was conducted via an Expert Review Score (ERS), a human-grounded rubric evaluating LLM-generated trading rationales along three dimensions: economic rationale, domain fidelity, and trade safety (risk awareness). Each dimension was scored on a 3-point ordinal scale {1 = poor, 2 = average, 3 = good}, based on the rubric shown in Table[II](https://arxiv.org/html/2508.02366v3#S2.T2 "TABLE II ‣ Qualitative Evaluation ‣ II-B4 Parameters and Evaluation ‣ II-B Experiment 1: LLM Trading Strategy Generation ‣ II Methodology ‣ Language Model Guided Reinforcement Learning in Quantitative Trading").

The review process followed a similar setup to that of[[23](https://arxiv.org/html/2508.02366v3#bib.bib23)], involving ten participants: five senior finance professionals and five retail traders (or professionals in the industry who do not actively trade). Each reviewer evaluated anonymized data for three instruments over one year, including price data, fundamental and macroeconomic metrics, and firm-level news headlines.

Before reviewing the LLM rationale, expert participants made their own directional prediction (LONG/SHORT) to activate their internal domain models. They then reviewed the LLM’s reasoning and scored it using the rubric. Each session concluded with a 60-minute structured discussion to elicit LLM critiques and identify exemplars to use. Surveys took approximately 15 minutes to complete. All scores were normalized to a 1–3 range.

TABLE II: Expert rubric for scoring LLM rationales

### II-C Experiment 2: LLM-Guided RL

This experiment addressed Objective 2 by incorporating the LLM guidance within an RL framework.

#### II-C1 Data and Feature Engineering

The LLM outputs from Experiment 1 were reused. The RL agent adopted the DDQN configuration of[[3](https://arxiv.org/html/2508.02366v3#bib.bib3)], with a single LLM-derived interaction term τ\tau as innovation to the observation space. This feature consisted of:

*   •Signal Direction (dir​(π g)\mathrm{dir}(\pi^{g})): The discrete directional recommendation from the LLM. Zero represents SHORT and one LONG. 
*   •Signal Strength (str​(π g)\mathrm{str}(\pi^{g})): The LLM’s entropy-adjusted confidence score as a Likert-3 score. 

The interaction term was defined as

τ=dir​(π g)⋅str​(π g),\tau=\mathrm{dir}(\pi^{g})\cdot\mathrm{str}(\pi^{g}),(5)

where dir​(π g)\mathrm{dir}(\pi^{g}) was remapped from {0,1}\{0,1\} to {−1,1}\{-1,1\} to enable the interaction.

The LLM’s signal strength was derived from the normalized LLM’s confidence score:

μ conf=Likert 3,\mu_{\text{conf}}=\frac{\text{Likert}}{3},(6)

and adjusted using entropy-based certainty:

C=ε+(1−ε)​(1−H),C=\varepsilon+(1-\varepsilon)(1-H),(7)

where H∈[0,1]H\in[0,1] is the normalized entropy of the LLM output, and ε=0.01\varepsilon=0.01 ensures numerical stability. The final strength term is:

str​(π g)=μ conf⋅C.\mathrm{str}(\pi^{g})=\mu_{\text{conf}}\cdot C.(8)

This entropy-adjusted confidence follows the approach of[[24](https://arxiv.org/html/2508.02366v3#bib.bib24)], providing a soft weighting of the LLM’s signal by its certainty.

The interaction term τ\tau was selected empirically. Initial variants used direction only (str​(dir)\mathrm{str}(\textit{dir})), followed by LLM’s confidence (str​(π g)\mathrm{str}(\pi^{g})) and direction. The final form was chosen based on empirical performance and compatibility with DDQN’s continuous normalized input space[[3](https://arxiv.org/html/2508.02366v3#bib.bib3)].

#### II-C2 LLM+RL Hybrid Architecture

The baseline DDQN agent is augmented by the Strategist Agent and Analyst Agent, which produce monthly strategies for the stock’s behavior. For practical reasons, outputs from the LLM were precomputed per instrument and fixed throughout training.

#### II-C3 Training and Parameters

Hyperparameters mirror [[3](https://arxiv.org/html/2508.02366v3#bib.bib3)] and the LLM settings follow those in Experiment 1. Training was conducted over 25 runs × 50 episodes per instrument using an NVIDIA RTX 3050, with each equity trained for 3 hours.

To ensure comparability with the benchmark[[3](https://arxiv.org/html/2508.02366v3#bib.bib3)], we replicated all baseline metrics within acceptable statistical bounds.

#### II-C4 Evaluation Metrics

Two measures were considered:

*   •SR: Same as Experiment 1 see Eq.([2](https://arxiv.org/html/2508.02366v3#S2.E2 "In Quantitative Metrics ‣ II-B4 Parameters and Evaluation ‣ II-B Experiment 1: LLM Trading Strategy Generation ‣ II Methodology ‣ Language Model Guided Reinforcement Learning in Quantitative Trading")). 
*   •MDD: Captures the largest observed loss from a historical peak to a subsequent trough:

MDD=P peak−P low P peak.\text{MDD}=\frac{P_{\text{peak}}-P_{\text{low}}}{P_{\text{peak}}}.(9)

where P peak P_{\text{peak}} is the highest portfolio value observed before the largest drop, and P low P_{\text{low}} is the lowest value reached before a new peak is established. Lower values indicate stronger downside protection. 

These metrics together assess whether LLM-guided RL agents can adapt to different equities without changing the core architecture.

III Results and Discussion
--------------------------

### III-A Experiment 1 Results

This section presents empirical results across the baseline (P0) and four prompt versions (P1–P4) from Table[I](https://arxiv.org/html/2508.02366v3#S2.T1 "TABLE I ‣ Prompt Improvement 3 – News Factors ‣ II-B3 Prompt Engineering Methodology ‣ II-B Experiment 1: LLM Trading Strategy Generation ‣ II Methodology ‣ Language Model Guided Reinforcement Learning in Quantitative Trading"), addressing Objective 1. We evaluated their impact on SR, PPL, and H LLM H_{\text{LLM}}. From P4 onward, qualitative evaluation was incorporated through ERS, introduced once the prompt design had stabilized. All backtests were conducted over 2018–2020 to ensure comparability with the RL’s OOS results. Statistical significance was assessed using two-tailed t-tests across 25 runs per ticker, with hypotheses H 0:μ P4=μ P1 H_{0}:\mu_{\text{P4}}=\mu_{\text{P1}}. All runs were executed at a sampling temperature of 0.7 to capture variance, while the reported metrics correspond to the deterministic setting (temperature 0) with fixed random seeds for reproducibility.

TABLE III: Sharpe Ratio Across Prompts and Benchmark

TABLE IV: Perplexity Across Prompts

TABLE V: Entropy Across Prompts

TABLE VI: Expert Reviewer Scores for Prompt 4

TABLE VII: P4 vs. P1 Significance of Metric Changes

Tables[III](https://arxiv.org/html/2508.02366v3#S3.T3 "TABLE III ‣ III-A Experiment 1 Results ‣ III Results and Discussion ‣ Language Model Guided Reinforcement Learning in Quantitative Trading")–[V](https://arxiv.org/html/2508.02366v3#S3.T5 "TABLE V ‣ III-A Experiment 1 Results ‣ III Results and Discussion ‣ Language Model Guided Reinforcement Learning in Quantitative Trading") summarize the results across prompt versions relative to the benchmark (BM). Prompt 0, which relied solely on static technical features, outperformed Prompt 1 primarily because all equities exhibited upward trends during the OOS period. Prompt 1 yielded the weakest performance, with the lowest SR across most equities and the highest PPL and entropy, indicating that the LLM was unable to exploit the additional information when presented in an isolated context. Prompt 2 incorporated ICM, producing moderate gains in SR (mean 0.59 0.59) and suggesting improved confidence through reflection. Prompt 3 introduced decomposed instructions, eliciting CoT, and outperformed the benchmark with a mean SR of 0.71 0.71. Prompt 4 further included unstructured news signals and achieved the highest mean SR (1.02 1.02), lowest PPL and entropy, and showed higher confidence particularly on sentiment-sensitive tickers such as TSLA. Based on the p p-values in Table[VII](https://arxiv.org/html/2508.02366v3#S3.T7 "TABLE VII ‣ III-A Experiment 1 Results ‣ III Results and Discussion ‣ Language Model Guided Reinforcement Learning in Quantitative Trading"), the improvements were statistically significant for SR and entropy, while the changes in PPL were comparatively weaker.

Expert evaluation of Prompt 4 confirmed its effectiveness. Reviewers rated the LLM’s rationale highly (mean 2.7 2.7 out of 3 3), highlighting its ability to integrate valuation, sentiment, and analytics.

Fidelity received a slightly lower score (mean 2.65 2.65), with critiques focused on inconsistent thresholding. For instance, one reviewer noted, “Calling RSI near 40 40 ‘oversold’ is debatable,” requiring refinements in numerical phrasing.

Feedback varied by background: buy-side professionals emphasized transparency in feature weighting, whereas retail reviewers focused on technical and macro signals. All commented on the lack of a neutral or hold signal, which was done to align with[[3](https://arxiv.org/html/2508.02366v3#bib.bib3)].

Overall, results validated Prompt 4’s modular design and market narrative awareness. It outperformed earlier prompts and was selected as the global policy generation prompt for the LLM–RL hybrid in Experiment 2.

The computational costs of Experiment 1 are summarized in Table[VIII](https://arxiv.org/html/2508.02366v3#S3.T8 "TABLE VIII ‣ III-A Experiment 1 Results ‣ III Results and Discussion ‣ Language Model Guided Reinforcement Learning in Quantitative Trading"). The overall cost for a single run with each prompt was approximately $36, increasing to about $150 when the writer–judge loop was included. When accounting for additional trials and development, the cumulative cost amounted to $345. Inference time ranged from approximately 1.5 to 2 hours per asset and prompt version.

TABLE VIII: Token Usage and Costs

### III-B Experiment 2 Results

This experiment addressed Objective 2 by comparing three agent architectures: (i) the benchmark RL-only[[3](https://arxiv.org/html/2508.02366v3#bib.bib3)], (ii) the best-performing LLM prompt from Experiment 1, and (iii) a hybrid LLM+RL agent. All agents were trained in identical environments.

To determine whether the hybrid agent outperformed the benchmark, we conducted two-sided paired t t-tests on the SR across 25 runs for each stock. The null hypothesis H 0 H_{0} assumed no difference in mean performance: H 0:μ LLM+RL=μ RL-only H_{0}:\mu_{\text{LLM+RL}}=\mu_{\text{RL-only}}. All resulting p p-values were below 0.05, indicating statistically significant improvements.

TABLE IX: Experiment 2 Results: Sharpe Ratio

TABLE X: Experiment 2 Results: Maximum Drawdown

![Image 2: Refer to caption](https://arxiv.org/html/2508.02366v3/aapl_agent_performance.png)

Figure 2: AAPL Performance with LLM+RL Model.

![Image 3: Refer to caption](https://arxiv.org/html/2508.02366v3/AAPL_sharpe.png)

Figure 3: Training Behavior for AAPL: Sharpe Ratio.

![Image 4: Refer to caption](https://arxiv.org/html/2508.02366v3/AAPL_Q1.png)

Figure 4: Training Behavior for AAPL: Q-Values for LONG.

![Image 5: Refer to caption](https://arxiv.org/html/2508.02366v3/AAPL_Q0.png)

Figure 5: Training Behavior for AAPL: Q-Values for SHORT.

Results in Table[IX](https://arxiv.org/html/2508.02366v3#S3.T9 "TABLE IX ‣ III-B Experiment 2 Results ‣ III Results and Discussion ‣ Language Model Guided Reinforcement Learning in Quantitative Trading") confirm that the LLM+RL agent outperformed the RL-only baseline in four out of six assets.

AAPL and META did not show consistent individual outperformance. Fig.[2](https://arxiv.org/html/2508.02366v3#S3.F2 "Figure 2 ‣ III-B Experiment 2 Results ‣ III Results and Discussion ‣ Language Model Guided Reinforcement Learning in Quantitative Trading") illustrates AAPL’s trading behavior during one episode. The top panel plots price, technical indicators, and trades: hollow triangles mark RL trades; filled arrows show LLM monthly guidance. The LLM issued sparse but confident signals (strength >> 0.6), often aligned with technical points of interest (e.g., MA interactions). In contrast, the RL agent frequently mistimed entries and exits.

From December 2018 to January 2019, the RL agent oscillated between LONG and SHORT positions with punishing results and despite receiving strong signals from the LLM. The LLM issued high-confidence guidance for a SHORT in December followed by a LONG in January, both with signal strengths exceeding 0.8. Regardless, the RL agent held a LONG position throughout the decline.

As shown in Fig.[5](https://arxiv.org/html/2508.02366v3#S3.F5 "Figure 5 ‣ III-B Experiment 2 Results ‣ III Results and Discussion ‣ Language Model Guided Reinforcement Learning in Quantitative Trading"), the DDQN assigns lower Q-values to Short actions, indicating limited confidence. This follows from lower-bound constraints (used to cap leverage) that created an asymmetric return function by triggering buy-to-cover after price increases, reducing portfolio value and subsequent SHORT exposure. Also, the selected equity universe has positive historical drift, which raises average prices, with limited opportunity to capture SHORT returns. Together these features lower the expected return of a SHORT and discourage sustained SHORT positions[[3](https://arxiv.org/html/2508.02366v3#bib.bib3)].

The bottom panel confirms that the LLM maintained high confidence near key inflection points, and reduced conviction when trends have persisted (possibly awaiting a reversal from its training corpus). However, the RL agent didn’t fully exploit these signals due to the underlying RL architecture, which remained fixed for the purposes of this experiment.

Fig.[3](https://arxiv.org/html/2508.02366v3#S3.F3 "Figure 3 ‣ III-B Experiment 2 Results ‣ III Results and Discussion ‣ Language Model Guided Reinforcement Learning in Quantitative Trading") illustrates the evolution of the SR for AAPL throughout the training episodes. The hybrid LLM+RL agent (orange line) outperformed the baseline RL agent (blue line) in both mean Sharpe and stability, as reflected in the narrower shaded confidence intervals. The LLM’s SR is shown for reference (black dashed line).

Figs.[4](https://arxiv.org/html/2508.02366v3#S3.F4 "Figure 4 ‣ III-B Experiment 2 Results ‣ III Results and Discussion ‣ Language Model Guided Reinforcement Learning in Quantitative Trading") and[5](https://arxiv.org/html/2508.02366v3#S3.F5 "Figure 5 ‣ III-B Experiment 2 Results ‣ III Results and Discussion ‣ Language Model Guided Reinforcement Learning in Quantitative Trading") show Q-values for LONG and SHORT actions respectively, with y-axis clipped to [−0.03,0.03][-0.03,0.03] to highlight late-episode convergence. Early training was noisy for both agents. The LLM+RL agent converged faster with lower variance. Although Q-value separation rarely exceeded 0.01, the hybrid showed slightly stronger directional signals. These gains emerged without modifying the DDQN or imposing reward shaping, thus isolating the effect of the LLM’s guidance. The narrow Q-range stems from the RL baseline design.

The hybrid agent did not consistently minimize MDD per stock but achieved values close to the best across agents, with the lowest overall mean (0.31). This suggests overall smoother drawdowns under uncertainty across the universe (see Tables[IX](https://arxiv.org/html/2508.02366v3#S3.T9 "TABLE IX ‣ III-B Experiment 2 Results ‣ III Results and Discussion ‣ Language Model Guided Reinforcement Learning in Quantitative Trading") and[X](https://arxiv.org/html/2508.02366v3#S3.T10 "TABLE X ‣ III-B Experiment 2 Results ‣ III Results and Discussion ‣ Language Model Guided Reinforcement Learning in Quantitative Trading")).

IV Conclusion and Future Work
-----------------------------

This study has explored an RL+LLM hybrid architecture for algorithmic trading, where LLMs generate guidance for RL agents to act as tactical executors.

Experiment 1 has shown that well engineered prompts improve the LLM’s performance, with Prompt 4 achieving the highest SR and lowest uncertainty. Expert evaluations confirmed the rationale of generated strategies within the domain.

Experiment 2 has demonstrated that an RL agent guided by LLM signals outperforms the RL-only baseline in four out of six stocks when evaluated by their Sharpe Ratio. While MDD was not consistently reduced, the overall drawdowns remained low on average. Importantly, the underlying RL architecture was not modified; all observed improvements stemmed from LLM guidance.

Future research should address two main directions. First, while the LLM can guide the RL, reward shaping is necessary to attain optimal results. Second, modular specialization through multiple LLM agents prompted for specific domains may enable a mixture-of-experts architecture, and lessen the risk of confabulation.

Overall, this work presents a novel LLM+RL system that improves both return and risk outcomes. It supports modular, agentic setups where LLMs operate as trustworthy planners in financial decision making.

Supplementary Material
----------------------

Full prompt templates (strategy and analyst), labeling-heuristic pseudocode, extended dataset schema, and complete replication tables are available from the corresponding author upon request.

Acknowledgment
--------------

We thank the expert reviewers who contributed their time and expertise to this work.

References
----------

*   [1] M.Chlistalla, “High-frequency trading: Better than its reputation?” Deutsche Bank Research, Frankfurt am Main, Germany, Tech. Rep. Research Briefing, Feb. 2011. [Online]. Available: [https://www.finextra.com/finextra-downloads/featuredocs/prod0000000000269468.pdf](https://www.finextra.com/finextra-downloads/featuredocs/prod0000000000269468.pdf)
*   [2] E.F. Fama and K.R. French, “A five-factor asset pricing model,” _Journal of Financial Economics_, vol. 116, no.1, pp. 1–22, 2015. 
*   [3] T.Théate and D.Ernst, “An application of deep reinforcement learning to algorithmic trading,” _Expert Systems with Applications_, vol. 173, p. 114632, Jul. 2021. 
*   [4] S.M. Bartram, J.Branke, and M.Motahari, _Artificial Intelligence in Asset Management_. CFA Institute Research Foundation, 2020. 
*   [5] R.S. Sutton and A.G. Barto, _Reinforcement Learning: An Introduction_, 2nd ed., ser. Adaptive Computation and Machine Learning series. MIT Press, 2018. [Online]. Available: [http://incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)
*   [6] X.-Y. Liu, H.Yang, J.Gao, and C.D. Wang, “Finrl: Deep reinforcement learning framework to automate trading in quantitative finance,” in _Proceedings of the Second ACM International Conference on AI in Finance_. ACM, Nov. 2021. 
*   [7] M.M.L. de Prado, “Beyond econometrics: A roadmap towards financial machine learning,” _Econometric Modeling: Theoretical Issues in Microeconometrics eJournal_, 2019. [Online]. Available: [https://api.semanticscholar.org/CorpusID:199365784](https://api.semanticscholar.org/CorpusID:199365784)
*   [8] L.Onozo, F.Arthur, and B.Gyires-Tóth, “Leveraging LLMs for financial news analysis and macroeconomic indicator nowcasting,” _IEEE Access_, 2024, early Access. Online. Accessed: Feb. 10, 2025. 
*   [9] W.Zhang _et al._, “A multimodal foundation agent for financial trading: Tool-augmented, diversified, and generalist,” New York, NY, USA, p. 4314–4325, 2024. 
*   [10] M.Wang, K.Izumi, and H.Sakaji, “LLMFactor: Extracting profitable factors through prompts for explainable stock movement prediction,” 2024. [Online]. Available: [https://arxiv.org/abs/2406.10811](https://arxiv.org/abs/2406.10811)
*   [11] S.Schulhoff _et al._, “The prompt report: A systematic survey of prompting techniques,” 2024. [Online]. Available: [https://arxiv.org/abs/2406.06608](https://arxiv.org/abs/2406.06608)
*   [12] Y.Yu _et al._, “Finmem: A performance-enhanced llm trading agent with layered memory and character design,” in _Proceedings of the AAAI Spring Symposium Series_, R.P.A. Petrick and C.W. Geib, Eds. AAAI Press, Jan. 2024, pp. 595–597. 
*   [13] K.Arulkumaran, M.P. Deisenroth, M.Brundage, and A.A. Bharath, “Deep reinforcement learning: A brief survey,” _IEEE Signal Processing Magazine_, vol.34, no.6, pp. 26–38, 2017. 
*   [14] A.Lopez-Lira and Y.Tang, “Can chatgpt forecast stock price movements? return predictability and large language models,” 2023. 
*   [15] J.Huang and K.C.-C. Chang, “Towards reasoning in large language models: A survey,” 2022. [Online]. Available: [https://arxiv.org/abs/2212.10403](https://arxiv.org/abs/2212.10403)
*   [16] M.Lopez de Prado, “The 10 reasons most machine learning funds fail,” The Journal of Portfolio Management, Tech. Rep., 06 2018. 
*   [17] L.Takara, A.Santos, V.Mariani, and L.Coelho, “Deep reinforcement learning applied to a sparse-reward trading environment with intraday data,” _Expert Systems with Applications_, vol. 238, p. 121897, 2024. 
*   [18] A.Chaddha and S.Yadav, “Examining the predictive power of moving averages in the stock market,” _Journal of Student Research_, vol.11, no.3, 2022. 
*   [19] S.Wang, H.Yuan, L.M. Ni, and J.Guo, “Quantagent: Seeking holy grail in trading by self-improving large language model,” 2024. [Online]. Available: [https://arxiv.org/abs/2402.03755](https://arxiv.org/abs/2402.03755)
*   [20] A.Lopez-Lira, Y.Tang, and M.Zhu, “The memorization problem: Can we trust LLMs’ economic forecasts?” 2025. [Online]. Available: [https://arxiv.org/abs/2504.14765](https://arxiv.org/abs/2504.14765)
*   [21] H.Gonen, S.Iyer, T.Blevins, N.Smith, and L.Zettlemoyer, “Demystifying prompts in language models via perplexity estimation,” in _Findings of the Association for Computational Linguistics: EMNLP 2023_, H.Bouamor, J.Pino, and K.Bali, Eds. Singapore: Association for Computational Linguistics, Dec. 2023, pp. 10 136–10 148. 
*   [22] A.Kaltchenko, “Entropy heat-mapping: Localizing GPT-based OCR errors with sliding-window shannon analysis,” 2025. [Online]. Available: [https://arxiv.org/abs/2505.00746](https://arxiv.org/abs/2505.00746)
*   [23] L.M. Demajo, V.Vella, and A.Dingli, “Explainable AI for interpretable credit scoring,” 2020. [Online]. Available: [https://arxiv.org/abs/2012.03749](https://arxiv.org/abs/2012.03749)
*   [24] G.Yona, R.Aharoni, and M.Geva, “Can large language models faithfully express their intrinsic uncertainty in words?” 2024. [Online]. Available: [https://arxiv.org/abs/2405.16908](https://arxiv.org/abs/2405.16908)

Appendix A Strategy Prompt
--------------------------

The final tuned prompt from Experiment 1 and the LLM strategy generator for Experiment 2, is available in[1](https://arxiv.org/html/2508.02366v3#LST1 "Listing 1 ‣ Appendix A Strategy Prompt ‣ Language Model Guided Reinforcement Learning in Quantitative Trading").

Listing 1: Tuned Strategy Prompt

1 User_Context:

2 Last_Strategy_Used_Data:

3 last_returns:"{Last_LLM_Strat_Returns}"

4 last_action:"{Last_LLM_Strat_Action}"

5 Rationale:|

6"""{Last_LLM_Strat}"""

7

8

9 Stock_Data:

10 General:

11 Beta:{Market_Beta}

12 Classification:{classification}

13

14 Last_Weeks_Price:

15 Close:"{Close}"

16 Volume:"{Volume}"

17

18 Weekly_Past_Returns:"{Weekly_Past_Returns}"

19

20 Historical_Volatility:

21 HV_Close:"{HV_Close}"

22

23 Implied_Volatility:

24 IV_Close:"{IV_Close}"

25

26 Fundamental_Data:

27 Ratios:

28 Current_Ratio:"{Current_Ratio}"

29 Quick_Ratio:"{Quick_Ratio}"

30 Debt_to_Equity_Ratio:"{Debt_to_Equity_Ratio}"

31 PE_Ratio:"{PE_Ratio}"

32 Margins:

33 Gross_Margin:"{Gross_Margin}"

34 Operating_Margin:"{Operating_Margin}"

35 Net_Profit_Margin:"{Net_Profit_Margin}"

36 Growth Metrics:

37 EPS_YoY:"{EPS_YoY_Growth}"

38 Net_Income_YoY:"{Net_Income_YoY_Growth}"

39 Free_Cash_Flow_YoY:"{Free_Cash_Flow_Per_Share_YoY_Growth}"

40

41 Technical_Analysis:

42 Moving_Averages:

43 20 MA:"{20MA}"

44 50 MA:"{50MA}"

45 200 MA:"{200MA}"

46 MA_Slopes:

47 20 MA_Slope:"{20MA_Slope}"

48 50 MA_Slope:"{50MA_Slope}"

49 100 MA_Slope:"{100MA_Slope}"

50 200 MA_Slope:"{200MA_Slope}"

51 MACD:

52 Value:"{MACD}"

53 Signal_Line:"{Signal_Line}"

54 MACD_Strength:{MACD_Strength}

55 RSI:

56 Value:"{RSI}"

57 ATR:"{ATR}"

58

59 Macro_Data:

60 Macro_Indices:

61 SPX:

62 Close:"{SPX_Close}"

63 Close_20MA:"{SPX_Close_MA}"

64 Close_Slope:"{SPX_Close_Slope}"

65 VIX:

66 Close:"{VIX_Close}"

67 Close_20MA:"{VIX_Close_MA}"

68 Close_Slope:"{VIX_Close_Slope}"

69 Economic_Data:

70 GDP_QoQ:"{GDP_QoQ}"

71 PMI:"{PMI}"

72 Consumer_Confidence_QoQ:"{Consumer_Confidence_QoQ}"

73 M2_Money_Supply_QoQ:"{M2_Money_Supply_QoQ}"

74 PPI_YoY:"{PPI_YoY}"

75 Treasury_Yields_YoY:"{Treasury_Yields_YoY}"

76

77 Options_Data:

78 Put_IV_Skews:

79 OTM_Skew:"{OTM_Skew}"

80 ATM_Skew:"{ATM_Skew}"

81 ITM_Skew:"{ITM_Skew}"

82 20 Day_Moving_Averages:

83 OTM_Skew_MA:"{MA_OTM_Skew}"

84 ATM_Skew_MA:"{MA_ATM_Skew}"

85 ITM_Skew_MA:"{MA_ITM_Skew}"

86

87 News_Sentiment:{news_sentiment}

88 News_Impact_Score:{news_impact_score}

89

90 System_Context(System):

91 Persona:{persona}

92 Portfolio_Objectives:{portfolio_objectives}

93 Instructions:|

94 Develop a LONG or SHORT trading strategy for a single stock only for the next Month that aligns with the‘portfolio_objectives‘.Follow these guidelines:

95

96 1.Stock Analysis:

97-Evaluate price trends:Compare the Close price against 20MA,50MA,and 200MA to assess momentum or reversals.

98-Analyze returns:Use Weekly Past Returns to validate trend sustainability.

99-Contextualize volatility:Align‘HV_Close‘and‘HV_High‘with recent price action for trend validation.

100-Incorporate beta:Use‘beta‘to gauge sensitivity to market movements.

101

102-ICL Example:"Close␣price␣above␣20MA␣and␣50MA␣with␣steep␣20MA␣slope␣signals␣bullish␣momentum.␣Weekly␣returns␣confirm␣a␣sustainable␣uptrend."

103

104 2.Technical Analysis:

105-Use RSI:Identify momentum signals(>70 overbought;<30 oversold)and divergences for reversals.

106-Validate with‘MACD‘:Use crossovers of‘MACD.Value‘and‘Signal_Line‘,and‘MACD_Strength‘for directional confidence.

107-Leverage‘RSI.value‘divergences,and steep‘Moving_Averages‘slopes.Or focus on stable‘Moving_Averages‘patterns on stable historical volatility‘HV_Close‘.

108-ICL Example:"RSI␣at␣65,␣a␣positive␣MACD␣crossover␣indicate␣bullish␣momentum."

109

110 3.Fundamental Analysis:

111-Evaluate growth metrics:Use‘EPS_YoY‘,‘Net_Income_YoY‘,and‘Free_Cash_Flow_YoY‘for profitability and sustainability.

112-Prioritize ratios:Low‘Debt_to_Equity_Ratio‘and‘Current_Ratio‘reflect financial stability.

113-Focus on aggressive‘Growth Metrics‘and earnings news.

114-ICL Example:"EPS␣YoY␣growth␣of␣25%␣and␣low␣Debt-to-Equity␣ratio␣of␣0.5␣support␣strong␣financial␣health,␣aligning␣with␣a␣LONG␣strategy."

115

116 4.Macro Analysis:

117-Align with market sentiment across‘Macro_Data‘:

118-"SPX_Close_Slope␣>␣0␣&&␣VIX_Close_Slope␣<␣0":Bullish(Risk-On)

119-"SPX_Close_Slope␣<␣0␣&&␣VIX_Close_Slope␣>␣0":Bearish(Risk-Off)

120-Validate with‘Economic_Data‘:

121-"GDP_QoQ␣>␣0␣&&␣‘PMI‘␣>␣50"leads to Economic Expansion

122-"‘Treasury_Yields_YoY‘␣<␣0"Signals Recession Risk,especially if already mentioned in‘Rationale‘.

123

124-ICL Examples:

125-"‘SPX_Close_Slope‘␣>␣0␣&&␣‘VIX_Close_Slope‘␣<␣0␣We␣have␣Market␣Confidence"

126-"‘GDP_QoQ‘␣Falling␣&&␣‘PMI‘␣<␣50␣We␣have␣an␣Economic␣Slowdown."

127

128 5.Options Analysis:

129-Compare‘OTM_Skew‘,‘ATM_Skew‘,and‘ITM_Skew‘IV Skews:Assess differences to gauge market sentiment and directional bias using their‘20Day_Moving_Averages‘.

130-Leverage IV spikes to capitalize on speculative directional trades.

131-Example:"Rising␣‘ATM_Skew_MA‘␣>␣0,␣market␣pricing␣up␣move,␣with␣stable␣HV␣supports␣a␣LONG␣position,␣as␣it␣indicates␣growing␣upside␣expectations␣without␣excessive␣fear."

132

133 6.News Analysis:

134-Use‘News_Sentiment‘and‘News_Impact_Score‘(1-3).

135-Only strong directional news(score=3)should override other signals.

136-Medium news(score=2)supports but does not lead.

137-Always check if news contradicts macro or technical trend.

138

139 7.Performance Reflection and Strategic Adaptation:

140-If‘Last_Strategy_Used_Data‘is available:

141-Assess the outcome of the previous strategy by examining‘last_returns‘and the chosen‘last_action‘.

142-Determine if the result aligns with the expectations outlined in the previous‘Rationale‘.

143-Identify if the direction(LONG or SHORT)led to desirable or undesirable outcomes.

144-You must NOT reuse or copy the previous‘Rationale‘.It is only context for reflection.

145-Summarize in 1-2 sentences whether the previous strategy performed as expected.

146-Example:"The␣previous␣LONG␣strategy␣yielded␣positive␣returns,␣confirming␣the␣bullish␣setup␣based␣on␣RSI␣and␣moving␣averages."

147-Do NOT include language or phrasing from the previous rationale.

148-Confidence assignment:

149-Assign a Likert score(1 to 3)to your‘action_confidence‘:

150-1:Low confidence;contradictory or weak alignment across features.

151-2:Moderate confidence;partial alignment with moderate evidence.

152-3:High confidence;strong convergence across key features.

153-Feature Attribution:

154-Rank the importance of each major feature used in your current rationale using a Likert scale(1 to 3):

155-1:Minimal contribution;not required for the decision.

156-2:Moderate contribution;relevant but not critical.

157-3:High contribution;pivotal to the trading decision.

158

159 Output:

160 action:Str.LONG or SHORT.

161 action_confidence:int.Likert scale(1-3)confidence in the proposed‘action‘,adjusted based on prior strategy outcome if‘Last_Strategy_Used_Data‘is available.

162 explanation:>

163 A concise rationale(max 350 words)justifying the proposed‘action‘.

164 Include:

165-The top 5 weighted features used in the decision,each labeled with its Likert importance(1-3).

166(e.g.,"Stock_Data.Price.Close,␣Weight␣3,␣Technical_Analysis.RSI.Value,␣Weight␣1,␣Options_Data.ATM_Skew,␣Weight␣2")

167-A reflective assessment of‘Last_Strategy_Used_Data‘,including whether the past‘action‘was successful and was it maintained given prior‘Rationale‘.

168 features_used:

169-feature:the features used from the prompt’s␣context.

170␣␣␣␣␣␣direction:␣LONG,␣SHORT,␣or␣NEUTRAL

171␣␣␣␣␣␣weight:␣A␣Likert␣score␣(1␣to␣3)␣described␣in␣Feature␣Attribution.’

Appendix B Analyst Prompt
-------------------------

The Analyst prompt used in Experiment 1 is presented in Listing[2](https://arxiv.org/html/2508.02366v3#LST2 "Listing 2 ‣ Appendix B Analyst Prompt ‣ Language Model Guided Reinforcement Learning in Quantitative Trading"), adapted from[[10](https://arxiv.org/html/2508.02366v3#bib.bib10)]. News corpora were anonymized prior to prompting.

Listing 2: Analyst Prompt

1 User_Context:

2 Monthly_News_Articles_List:|

3"{articles_list}"

4

5 System_Context:

6 Persona:Financial Market Analyst

7 Instructions:|

8 Extract the‘Top 3‘news factors influencing stock price movements from the‘Monthly_News_Articles_List‘.Follow these steps:

9

10 1.Rank the news by relevance to stock price movements:

11-Prioritize news related to significant financial or market impacts(e.g.,acquisitions,partnerships,guidance revisions).

12-Weigh industry trends,macroeconomic influences,and analyst ratings based on their expected effect on the company valuation.

13-News with broad or long-term implications ranks higher.

14

15 2.Summarize content into key factors and corporate events affecting stock prices,using concise language and causal relationships.

16

17 3.For each factor,assign:

18-‘Sentiment‘:+1 for positive,-1 for negative,0 for neutral or mixed

19-‘Market_Impact_Score‘:Likert scale from 1 to 3,where:

20-1=minimal relevance

21-2=moderate influence

22-3=high impact driver

23

24 Examples of factors influencing stock prices include:

25-Strategic partnerships or competitor activity.

26-Industry trends or macroeconomic influences.

27-Product launches or market expansions.

28-Analyst ratings,significant stock price moves,or expectations.

29-Corporate events:guidance revisions,acquisitions,contracts,splits,repurchases,dividends.

30

31 Example:

32’A␣major␣tech␣company␣partners␣with␣a␣leading␣automotive␣firm␣for␣EV␣battery␣innovation.␣Analysts␣predict␣this␣could␣boost␣revenues␣significantly.’

33

34 Ranked Factors:

35 1.factor:Strategic partnership in EV battery technology expected to increase revenue.

36 sentiment:+1

37 market_impact:3

38 2.factor:Positive sentiment driven by projected long-term gains.

39 sentiment:+1

40 market_impact:2

41 3.factor:Growing demand for EV technology anticipated to support future earnings.

42 sentiment:+1

43 market_impact:2

44

45 Output:

46 factors:

47-factor:str.Summary of the news item.Max 70 words.

48-sentiment:int.One of Positive+1,Negative-1,or Neutral 0

49-market_impact:int.Likert scale 1 to 3

Appendix C Algorithms
---------------------

The labeling algorithm emulates expert trading behavior by deliberately leveraging future return information to assign proxy trade actions in hindsight. This approach offers a cost-effective and scalable addition to manual annotation, capturing the general direction an informed trader might take. These synthetic labels are then provided to the LLM, along with a smaller set of HITL annotated examples.

Data: Time-indexed price series

Result: Trade action: LONG (1) or SHORT (0)

1

2 foreach _date t t in dataset_ do

3

P t←Close​(t)P_{t}\leftarrow\text{Close}(t)
;

4

r(10)←P t+10 P t−1 r^{(10)}\leftarrow\frac{P_{t+10}}{P_{t}}-1
,

r(20)←P t+20 P t−1 r^{(20)}\leftarrow\frac{P_{t+20}}{P_{t}}-1
;

5

r weighted←0.4⋅r(10)+0.6⋅r(20)r^{\text{weighted}}\leftarrow 0.4\cdot r^{(10)}+0.6\cdot r^{(20)}
;

6 if _r \_weighted\_>=0 r^{\text{weighted}}>=0_ then

7 Action

←\leftarrow
LONG (Trade_Action

=1=1
);

8

9 else

10 Action

←\leftarrow
SHORT (Trade_Action

=0=0
);

11

12

Algorithm 1 Expert Trade Heuristic

Appendix D Dataset
------------------

### Market Data

This market data (𝒮 mk\mathcal{S}_{\text{mk}}) included OHLCV price series as well as macro-level indicators and forward-looking sentiment signals. Specifically, it comprised:

*   •Daily returns of the S&P 500 Index (SPX) and NASDAQ-100 Index (NDX). These are market and sector indices, 
*   •Implied Volatility (IV) and Historical Volatility (HV) metrics, derived from the stock’s derivatives, 
*   •The CBOE Volatility Index (VIX) as a proxy for market fear and option market expectations, 
*   •Weekly Past Returns, which record the percentage change over the past four weekly intervals. The four-week span was selected empirically to align with the model’s monthly strategy generation frequency. 

These features help in modeling short-term market dynamics.

### Fundamental Data

Fundamental data (𝒮 fund\mathcal{S}_{\text{fund}}) has firm-level fundamentals and macroeconomic indicators. Macroeconomic variables provided contextual narrative for interpreting observed signals, and supporting regime identification[[9](https://arxiv.org/html/2508.02366v3#bib.bib9), [8](https://arxiv.org/html/2508.02366v3#bib.bib8)]. This set covered:

*   •Liquidity ratios: Current Ratio, Quick Ratio; 
*   •Leverage and coverage: Debt-to-Equity, Interest Coverage; 
*   •Profitability metrics: Gross Margin, Operating Margin, Return on Equity (ROE), Return on Assets (ROA); 
*   •Valuation: Price-to-Earnings (P/E), Price-to-Book (P/B), Enterprise Value (EV), and Earnings Before Interest, Taxes, Depreciation, and Amortization (EBITDA). 
*   •Growth: Revenue and Earnings Growth; 
*   •Macroeconomic indicators: Gross Domestic Product (GDP), Purchasing Managers’ Index (PMI), Producer Price Index (PPI), Consumer Confidence Index (CCI), U.S. 10-Year Treasury Yield, and the 10Y–2Y yield curve slope. 

To enhance temporal abstraction, all variables were computed as quarter-over-quarter (QoQ) or year-over-year (YoY) percentage changes. It is critical to take first-order dynamics as LLMs can recall absolute numbers for economic details, allowing look-ahead bias in the backtests [[20](https://arxiv.org/html/2508.02366v3#bib.bib20)].

### Analytics

Technical indicators (𝒮 an\mathcal{S}_{\text{an}}) were computed over rolling 20-day windows using the open-source TA-Lib 7 7 7[https://ta-lib.org/](https://ta-lib.org/) library. These features include:

*   •Simple Moving Averages (SMA) over 20, 50, 100, 200 trading-day horizons, 
*   •Relative Strength Index (RSI), 
*   •Average True Range (ATR) for volatility, 
*   •Moving Average Convergence Divergence (MACD) with its signal line and derived strength, 
*   •Volume-Weighted Average Price (VWAP) as a reference anchor for intraday valuations. 

Each indicator was extended with slope and z-score to assist the LLM in capturing directional shifts and the statistical significance of deviations. These technical indicators are widely used in trading practice and academic research[[18](https://arxiv.org/html/2508.02366v3#bib.bib18)].

### Alternative Data

Structured representations of financial news headlines (𝒮 alt\mathcal{S}_{\text{alt}}) were extracted using a large language model (LLM), which anonymized and synthesized the content into latent factors. Following the LLMFactor methodology[[10](https://arxiv.org/html/2508.02366v3#bib.bib10)], each news item was distilled into 2–5 interpretable factors, capturing macroeconomic and firm-specific signals.

To mitigate memorization and data leakage risks, named entities and dates were anonymized (e.g., “Tesla” becomes “the Company”).

Appendix E Replicated Benchmark Metrics
---------------------------------------

We report the replicated benchmark metrics in Appendix[E](https://arxiv.org/html/2508.02366v3#A5 "Appendix E Replicated Benchmark Metrics ‣ Language Model Guided Reinforcement Learning in Quantitative Trading") for the assets used in[[3](https://arxiv.org/html/2508.02366v3#bib.bib3)]. We include the mean SR and MDD, each averaged across 25 runs with standard deviation σ\sigma.

For the SR, we conduct a two-sided one-sample t t-test to assess whether the metric is significantly different from the published value. The null hypothesis H 0 H_{0} assumes equivalence, i.e. H 0:μ SR=SR paper H_{0}:\mu_{\text{SR}}=\text{SR}_{\text{paper}}.

Since this is a replication test, failing to reject H 0 H_{0} indicates successful replication. p p-values are computed only for SR; other metrics are reported without significance testing.

All assets have been successfully replicated within acceptable bounds, with exceptions highlighted in bold. Notably, GOOGL, one of the stocks included in our test environment, exhibited a statistically significant deviation from the original benchmark, with a p p-value below 0.05.

TABLE XI: Replication Metrics for [[3](https://arxiv.org/html/2508.02366v3#bib.bib3)]

