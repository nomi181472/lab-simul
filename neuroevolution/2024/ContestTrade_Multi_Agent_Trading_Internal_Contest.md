Title: ContestTrade: A Multi-Agent Trading System Based on Internal Contest Mechanism

URL Source: https://arxiv.org/html/2508.00554

Markdown Content:
Li Zhao 1, Rui Sun 1, Zuoyou Jiang 1, Bo Yang 1, Yuxiao Bai 2, 

Mengting Chen 2, Xinyang Wang 1, Jing Li 1, Zuo Bai 1 2

###### Abstract

In financial trading, large language model (LLM)-based agents demonstrate significant potential. However, the high sensitivity to market noise undermines the performance of LLM-based trading systems. To address this limitation, we propose a novel multi-agent system featuring an internal competitive mechanism inspired by modern corporate management structures. The system consists of two specialized teams: (1) Data Team - responsible for processing and condensing massive market data into diversified text factors, ensuring they fit the model’s constrained context. (2) Research Team - tasked with making parallelized multipath trading decisions based on deep research methods. The core innovation lies in implementing a real-time evaluation and ranking mechanism within each team, driven by authentic market feedback. Each agent’s performance undergoes continuous scoring and ranking, with only outputs from top-performing agents being adopted. The design enables the system to adaptively adjust to dynamic environment, enhances robustness against market noise and ultimately delivers superior trading performance. Experimental results demonstrate that our proposed system significantly outperforms prevailing multi-agent systems and traditional quantitative investment methods across diverse evaluation metrics. ContestTrade is open-sourced on GitHub at https://github.com/FinStep-AI/ContestTrade.

Introduction
------------

The financial sector is undergoing a profound transformation with the rise of LLM-based agents (Wu et al. [2023](https://arxiv.org/html/2508.00554v3#bib.bib22); Bai et al. [2023](https://arxiv.org/html/2508.00554v3#bib.bib1); Liu et al. [2021](https://arxiv.org/html/2508.00554v3#bib.bib14)). These agents (Ding et al. [2024](https://arxiv.org/html/2508.00554v3#bib.bib5)) excel at processing complex market information and assisting human analysts within a broader decision-making pipeline, offering interpretable outputs through natural language explanations and, when integrated with external tools, can flexibly incorporate diverse information sources, including news, numerical data, and sentiment indicators. Recent advancements highlight LLMs’ potential to automate complex trading decisions and achieve competitive performance in dynamic markets (Lopez-Lira and Tang [2024](https://arxiv.org/html/2508.00554v3#bib.bib15); Fatouros et al. [2024](https://arxiv.org/html/2508.00554v3#bib.bib7); Zhang et al. [2024a](https://arxiv.org/html/2508.00554v3#bib.bib29)).

However, market volatility and noise (Malkiel [1973](https://arxiv.org/html/2508.00554v3#bib.bib17); Engle [1982](https://arxiv.org/html/2508.00554v3#bib.bib6)) pose significant challenges. High sensitivity to noise often leads to inconsistent decision-making and undermines performance. Traditional single-agent approaches, while processing vast data, struggle to capture intricate temporal dependencies and resolve conflicting signals, especially during market turbulence, where noise obscures patterns and leads to suboptimal decisions.

Awareness of these challenges has fueled interest in multi-agent systems (LeBaron [2006](https://arxiv.org/html/2508.00554v3#bib.bib11)), leveraging role specialization for enhanced robustness (Byrd, Hybinette, and Balch [2020](https://arxiv.org/html/2508.00554v3#bib.bib2)). Inspired by collaborative investment firms, research agents are exploring frameworks where specialized agents collectively process information more effectively, distributing cognitive load and enabling complementary analytical perspectives.

Despite these advances, current multi-agent frameworks face limitations. Existing systems often use fixed data pipelines, struggling to adapt to shifting market regimes. Many frameworks make decisions based solely on individual agents’ historical returns, which is often insufficient for generating robust, high-quality signals in dynamic markets. Furthermore, current LLM-only agents often lack the sophisticated analytical tools and quantitative reasoning needed for complex market scenarios.

To address these limitations, we propose ContestTrade (A Multi-Agent Trading System Based on Internal Contest Mechanism). This novel framework enhances trading performance in noisy environments by integrating real-time competitive evaluation with a Deep Research methodology empowering agents with comprehensive financial tools.

The main contributions of this work are threefold:

1.   1.We adapt the recently popularized Deep Research methodology to the financial trading domain. Our approach equips LLM agents to autonomously plan and utilize specialized financial tools, thereby significantly enhancing trading signal quality. 
2.   2.We establish a novel internal contest mechanism driven by authentic market feedback. This mechanism operates within each team to ensure only optimal outputs are adopted, fostering continuous self-optimization for robustness against market noise. 
3.   3.We integrate these innovations within an efficient two-tiered, team-based multi-agent framework that addresses context limitations and demonstrates a new paradigm for collaborative and competitive financial AI. 

![Image 1: Refer to caption](https://arxiv.org/html/2508.00554v3/figures/main_frame.png)

Figure 1: The ContestTrade Framework Architecture, showing the complete pipeline from multi-source data input to final signals.

Related Works
-------------

LLMs are revolutionizing trading workflows as autonomous decision-making agents and sophisticated signal discovery tools. LLMFactor (Wang, Izumi, and Sakaji [2024](https://arxiv.org/html/2508.00554v3#bib.bib19)) extracts factors from text for explainable stock movement prediction. Frameworks like FinGPT (Luukkonen et al. [2023](https://arxiv.org/html/2508.00554v3#bib.bib16)) and FinRobot (Yang et al. [2024](https://arxiv.org/html/2508.00554v3#bib.bib25)) provide open-source resources, while TradingGPT (Li et al. [2023](https://arxiv.org/html/2508.00554v3#bib.bib13)) emulates human cognition for trading. The SEP framework (Koa et al. [2024](https://arxiv.org/html/2508.00554v3#bib.bib10)) further enhances this by enabling explainable predictions via self-reflection. As alpha miners, QuantAgent (Wang, Yuan, and Ni [2024](https://arxiv.org/html/2508.00554v3#bib.bib20)) refines its knowledge through real-world testing. AlphaGPT (Wang et al. [2023](https://arxiv.org/html/2508.00554v3#bib.bib21)) and AlphaGPT 2.0 (Yuan, Wang, and Guo [2024](https://arxiv.org/html/2508.00554v3#bib.bib28)) pioneer ”Human-in-the-Loop” strategies, translating human insights into effective alphas through interactive prompt engineering.

A critical challenge is agent-level adaptability in volatile markets, often addressed by self-reflection. FinMem (Li et al. [2024](https://arxiv.org/html/2508.00554v3#bib.bib12)) introduces a memory module for data assimilation and continuous decision refinement. Similarly, FinAgent (Zhang et al. [2024b](https://arxiv.org/html/2508.00554v3#bib.bib30)) uses a dual-level reflection and diversified memory retrieval for rapid adaptation from historical data, focusing on individual agent robustness.

Beyond individual capabilities, recent research focuses on structuring multi-agent interactions. HAD (Xing [2024](https://arxiv.org/html/2508.00554v3#bib.bib24)) employs agents (e.g., mood, rhetoric) that collaborate and synthesize insights through discussions. TradingAgents (Xiao et al. [2024](https://arxiv.org/html/2508.00554v3#bib.bib23)) employs specialized agents (e.g., analysts, traders) that collaborate and synthesize insights through debates. FinCon (Yu et al. [2025](https://arxiv.org/html/2508.00554v3#bib.bib27)) implements a manager-analyst hierarchy, enabling synchronized collaboration with dual-level risk control. These systems establish a strong baseline for collaborative multi-agent financial systems.

Finally, achieving a deep understanding of complex market dynamics is a fundamental challenge. An emerging direction leverages large-scale multi-agent simulation to model emergent behaviors. The MASS framework (Guo et al. [2025](https://arxiv.org/html/2508.00554v3#bib.bib9)), for instance, aims for superior market understanding by progressively increasing agent numbers and optimizing their distribution through reverse optimization. These studies underscore the importance of scale and emergent dynamics for profound market comprehension.

Architecture
------------

As illustrated in Figure[1](https://arxiv.org/html/2508.00554v3#Sx1.F1 "Figure 1 ‣ Introduction ‣ ContestTrade: A Multi-Agent Trading System Based on Internal Contest Mechanism"), our ContestTrade multi-agent trading system operates through a structured, dual-stage pipeline, emulating investment firm dynamics. The architecture comprises two specialized teams: the Data Team and the Research Team.

Our framework begins with Data Agents processing raw market data into textual factors. A key innovation is our internal contest mechanism, which continuously evaluates and forecasts the performance of each agent. The system then constructs an optimized factor portfolio by strategically aggregating the Data Analysis Agents’ outputs based on their predicted efficacy. This portfolio is then passed to Research Agents, who conduct parallel analyses and enter a second stage of competition. From their resulting proposals, a single, actionable asset allocation strategy is synthesized. This dual-stage competitive framework, centered on predictive evaluation, ensures that final decisions are guided only by the most robust insights, thereby enhancing adaptivity and filtering out market noise.

### Data Team Design

![Image 2: Refer to caption](https://arxiv.org/html/2508.00554v3/figures/data_analyst_agent_workflow.jpg)

Figure 2: The Data Team Architecture, showing the workflow of market data analysis.

The Data Team plays a critical upstream role in our multi-agent trading system, designed to distill vast volumes of raw market data into high-quality, context-friendly textual factors that are optimized for the constrained context windows of large language models (LLMs). By doing so, it directly mitigates the challenge of information overload and satisfies the need for concise, high-signal representations suitable for downstream reasoning.

#### Team Composition and Operational Workflow

The Data Team comprises multiple Data Analysis Agents operating in parallel, each following a similar workflow from data ingestion to textual factor generation, while processing distinct slices of market data. This parallel architecture improves efficiency and broadens the scope of the information coverage. The operational workflow for each agent is illustrated in Figure[2](https://arxiv.org/html/2508.00554v3#Sx3.F2 "Figure 2 ‣ Data Team Design ‣ Architecture ‣ ContestTrade: A Multi-Agent Trading System Based on Internal Contest Mechanism").

1.   1.Dynamic Information Prioritization and Extensive Reading: Each day, every agent dynamically generates a specific focus–or ”preference”–aligned with the short-term trading strategy. This preference guides the intelligent filtering of vast market information, allowing the model to focus on core readings. For example, an agent might prioritize companies that exhibit significant earnings growth or recent product launches. Guided by this preference, the agent extensively reads titles from the most relevant raw data sources, including real-time news, corporate financial statements, company announcements, and other diverse data sources. From the preliminary review, the agent eventually narrows down to several hundred high-relevance reading items for a deeper analysis. 
2.   2.Parallel Intensive Reading and Summarization: After initial filtering, the agents engage in parallel intensive reading and summarization. This stage leverages the intrinsic capabilities of LLM, eliminating the need for conventional natural language processing (NLP) tools and enabling seamless end-to-end information synthesis. 
3.   3.Textual Factor Generation through Context-Engineering: The culmination of each agent’s process is the creation of a textual factor: an unstructured natural language summary that encapsulates the agent’s synthesized insights from the day’s market information. To ensure compatibility with downstream models and adhere to strict input limitations, each agent’s output is rigorously capped at 4k tokens. This is precisely managed through specific max_token_len settings and meticulous context engineering via crafted prompts within the LLM. 

The collective output of the Data Team is a series of independent textual factors, generated by each individual agent. These factors are then aggregated to form a combined textual input. This consolidated input is directly fed into the Research Team, which is also composed of LLM agents, providing them with a comprehensive and distilled market overview to inform parallelized multipath trading decisions. Each team incorporates independent internal evaluation mechanisms to optimize their respective outputs.

The collection of textual factors from all agents in the Data Team serves as raw material for the team’s key deliverable. Through an internal evaluation process that assesses the effectiveness of each agent’s contributions, an optimized portfolio of the most promising factors is constructed. This distilled portfolio, representing the team’s collective market information, is then passed to the Research Team as the foundation for their subsequent analysis and decision-making.

### Research Team Design

![Image 3: Refer to caption](https://arxiv.org/html/2508.00554v3/figures/research_agent_workflow.png)

Figure 3: The Trading Strategy Architecture, showing the workflow of trading signal making.

The Research Team serves as a critical bridge between distilled market insights and actionable trading signals. Its primary objective is to generate precise and well-supported trading recommendations by leveraging the textual factors from the Data Team and conducting further in-depth analysis. This team is designed to produce parallelized multipath trading decisions. It explores diverse trading opportunities across various assets or strategies concurrently.

#### Team Composition and Operational Workflow

The Research Team is composed of multiple autonomous Research Agents, each operating independently. Every agent is initialized with a distinct “trading beliefs“, dynamically generated by large language models conditioned on predefined trading principles. This approach allows for a diverse set of perspectives and strategic approaches, fostering heterogeneity and robustness in the team’s overall decision-making process.

As illustrated in Figure[3](https://arxiv.org/html/2508.00554v3#Sx3.F3 "Figure 3 ‣ Research Team Design ‣ Architecture ‣ ContestTrade: A Multi-Agent Trading System Based on Internal Contest Mechanism"), each Research Agent follows a sophisticated Plan + ReAct (Yao et al. [2023](https://arxiv.org/html/2508.00554v3#bib.bib26)) framework to enable iterative planning, reasoning, and tool usage in order to formulate informed trading decisions:

1.   1.Input Reception and Initial Planning: Agents receive the aggregated textual factors from the Data Team. Guided by their pre-configured “trading beliefs“, they autonomously plan their next steps to identify potential trading opportunities aligned with their strategic bias. 
2.   2.Information Gathering (React Loop): As part of their planning and subsequent reacting, agents leverage a comprehensive suite of specialized financial tools to acquire additional information crucial for their decision-making. These tools facilitate a deeper dive into market conditions, company specifics, and broader economic trends. 
3.   3.In-depth Analysis and Signal Generation: With the gathered information, agents conduct thorough analyses, integrating the textual factors with the newly-retrieved data through their tools. This step culminates in the generation of a specific trading signal. 

Each Research Agent’s final deliverable is a structured trading signal, comprising a Trading Symbol (the specific financial instrument), a clear Action (buy, hold, or sell), an Evidence List containing data-backed justifications, and a Limitation Claim outlining any assumptions or uncertainties underlying the recommendation.

Table 1: Description of the specialized financial tool suite available to Research Agents.

All tools are implemented with strict temporal constraints, allowing agents to query data within specified time ranges to ensure both relevance and temporal consistency.

### Contest Mechanism: A General Adaptive Framework

#### The Contest Mechanism

The core of ContestTrade is an internal contest mechanism designed to enhance system adaptivity in dynamic markets. Its primary objective is to channel resources preferentially towards agents with consistently proven effectiveness, ensuring that the system’s final output is driven by the most robust strategies. This mechanism is formalized as a three-phase ”Quantify-Predict-Allocate” model. The entire process can be conceptualized as a pipeline that transforms a set of competing agents (𝒜\mathcal{A}) into a final allocation decision, represented by a weight vector (𝒲 t\mathcal{W}_{t}):

𝒜→f quant{q i,t}→f predict{u^i,t+n}→π allocate 𝒲 t\mathcal{A}\xrightarrow{f_{\text{quant}}}\{q_{i,t}\}\xrightarrow{f_{\text{predict}}}\{\hat{u}_{i,t+n}\}\xrightarrow{\pi_{\text{allocate}}}\mathcal{W}_{t}(1)

This pipeline first quantifies the historical performance of each a​g​e​n​t i agent_{i}, yielding a unified set of scores {q i,t}\{q_{i,t}\} at time t t. It then predicts the future utility {u^i,t+n}\{\hat{u}_{i,t+n}\} over the next n n steps. Finally, allocates resources based on these predictions. This general mechanism provides a unified foundation for the two distinct teams within the proposed system:

1.   1.Data Analyst Contest: Data Analysis Agents compete to construct an optimal information portfolio for the research agents. 
2.   2.Researcher Contest: Research Agents compete to achieve optimal capital allocation for the final trading. 

#### Data Analyst Contest

##### Optimization Objective

The objective of the Data Analyst Contest is to construct an optimal factor portfolio, ℱ t\mathcal{F}_{t}, from the universe of all available factors, 𝔽 t\mathbb{F}_{t}, to maximize the Research Agent’s final Decision Value (D​V DV) (Chroma [2024](https://arxiv.org/html/2508.00554v3#bib.bib3)). This value is modeled as a product of Information Value (V V) and Decision Capability (D​C DC):

D​V​(ℱ t)=V​(ℱ t)⋅D​C where V​(ℱ t)=∑i∈ℱ t v i.DV(\mathcal{F}_{t})=V(\mathcal{F}_{t})\cdot DC\quad\text{where}\quad V(\mathcal{F}_{t})=\sum_{i\in\mathcal{F}_{t}}v_{i}.(2)

Considering the complexity of investment analysis, research shows that an LLM’s Decision Capability (D​C DC) exhibits decay with respect to the total context length L L(liu2023lost; Modarressi et al. [2025](https://arxiv.org/html/2508.00554v3#bib.bib18)). This phenomenon can be effectively approximated by a sigmoid function with a performance inflection point of L 0 L_{0}(Zhou et al. [2025](https://arxiv.org/html/2508.00554v3#bib.bib31)):

D​C​(L)=1 1+e k​(L−L 0)DC(L)=\frac{1}{1+e^{k(L-L_{0})}}(3)

This capability constraint implies that we cannot naively maximize information value by simply expanding the portfolio. Therefore, the optimization objective becomes selecting a subset ℱ t⊆𝔽 t\mathcal{F}_{t}\subseteq\mathbb{F}_{t} that maximizes:

max ℱ t⊆𝔽 t⁡(∑i∈ℱ t v i)⋅1 1+e k​(∑i∈ℱ t l i−L 0)\max_{\mathcal{F}_{t}\subseteq\mathbb{F}_{t}}\left(\sum_{i\in\mathcal{F}_{t}}v_{i}\right)\cdot\frac{1}{1+e^{k\left(\sum_{i\in\mathcal{F}_{t}}l_{i}-L_{0}\right)}}(4)

where v i v_{i} and l i l_{i} denote the latent value and length of an individual factor i i, respectively.

Owing to the shape of the sigmoid function, it is clearly shown that an optimal effective context length, L∗<L 0 L^{*}<L_{0}, exists. This insight reduces our overall goal into two primary challenges: (1) finding a quantifiable proxy, q i q_{i}, for the unobservable latent value v i v_{i}, and (2) developing an allocation policy, π t\pi_{t}, to dynamically form the optimal portfolio.

##### Quantification via a Zero-Intelligence Trader

To quantify a factor’s latent value (v i v_{i}) with an objective proxy, q i,t q_{i,t}, we simulate a Zero-Intelligence (ZI) Trader (Gode and Sunder [1993](https://arxiv.org/html/2508.00554v3#bib.bib8)). This approach is inspired by the premise that a high-value factor should be profitable even without any external information or complex reasoning. Our simulated trader therefore operates on each atomic statement (”Observation”) extracted from the factor, intentionally operating under the constraint of limited analysis and no external context. This ensures q i,t q_{i,t} reflects the inherent predictive power of the information itself, independent of agent-specific strategies or external signals. We formally define the score as:

q i,t=∑obs∈F i,t ZI​(obs)q_{i,t}=\sum_{\text{obs}\in F_{i,t}}\text{ZI}(\text{obs})(5)

where F i,t F_{i,t} is the set of observations comprising factor i i at time t t, and the ZI​(⋅)\text{ZI}(\cdot) function is detailed in Algorithm[1](https://arxiv.org/html/2508.00554v3#alg1 "Algorithm 1 ‣ Quantification via a Zero-Intelligence Trader ‣ Data Analyst Contest ‣ Contest Mechanism: A General Adaptive Framework ‣ Architecture ‣ ContestTrade: A Multi-Agent Trading System Based on Internal Contest Mechanism").

Algorithm 1 ZI Trader

Input: A single Observation obs 

Output: The quantified value (reward) for obs

1:obs_reward

←0\leftarrow 0

2:RatedSymbols

←\leftarrow
GroundAndRate(obs)

3:// Each rating is an integer in

{−2,−1,0,1,2}\{-2,-1,0,1,2\}

4:for each

s s
in RatedSymbols do

5:reward

=s=s
.rating

×\times
PriceChange(

s s
.code,

t+1 t+1
)

6:obs_reward

==
obs_reward + reward

7:end for

8:return obs_reward

##### Prediction

The non-stationary nature of financial markets often manifesting as style rotation—periodic shifts in dominant investment factors—renders static factor-selection policies suboptimal. To overcome these challenges, our framework must dynamically identify factors poised to perform well. We hypothesize that factor performance exhibits short-term momentum. This empirical hypothesis is validated by the Rank Information Coefficient (RIC) between average factor scores across different time horizons. We observe:

RIC​(q¯t−m:t,q¯t:t+n)≫RIC​(q¯t−M:t,q¯t:t+N)\text{RIC}(\overline{q}_{t-m:t},\overline{q}_{t:t+n})\gg\text{RIC}(\overline{q}_{t-M:t},\overline{q}_{t:t+N})(6)

where M≫m,N≫n M\gg m,N\gg n. This indicates the correlation is significant for short-term windows (with optimal values found at m=5,n=3 m=5,n=3) but decays rapidly over longer horizons.

This allows us to frame the prediction as a supervised learning problem. We train a model to map a feature vector derived from a factor’s recent score sequence, x i,t=Φ​(q i,t−m+1:t)x_{i,t}=\Phi(q_{i,t-m+1:t}), to its expected future score μ^i,t+n\hat{\mu}_{i,t+n} and volatility σ^i,t+n\hat{\sigma}_{i,t+n}. We then define the factor’s predicted utility u^i,t+n\hat{u}_{i,t+n} as its predicted risk-adjusted score, u^i,t+n=μ^i,t+n/σ^i,t+n\hat{u}_{i,t+n}=\hat{\mu}_{i,t+n}/\hat{\sigma}_{i,t+n}. To mitigate overfitting and maintain model simplicity, we use LightGBM with a small number of estimators and default hyperparameters.

##### Allocation

The allocation policy π allocate\pi_{\text{allocate}} determines the final weight vector 𝒲 t\mathbf{\mathcal{W}}_{t} for the Data Analyst Contest. The allocation is a binary selection, represented by a weight vector 𝒲 t\mathbf{\mathcal{W}}_{t} with elements w i,t∈{0,1}w_{i,t}\in\{0,1\}.

This selection is formulated as a 0/1 Knapsack problem, where the objective is to find the optimal weight vector 𝒲 t\mathbf{\mathcal{W}}_{t} that maximizes the total predicted utility, subject to the effective context length constraint, L∗L^{*}.:

𝒲 t=argmax 𝒲 t∈{0,1}N​∑i=1 N u^i,t+n⋅w i,t s.t.∑i=1 N l i⋅w i,t≤L∗\mathbf{\mathcal{W}}_{t}=\underset{\mathbf{\mathcal{W}}_{t}\in\{0,1\}^{N}}{\text{argmax}}\sum_{i=1}^{N}\hat{u}_{i,t+n}\cdot w_{i,t}\quad\text{s.t.}\quad\sum_{i=1}^{N}l_{i}\cdot w_{i,t}\leq L^{*}(7)

Informed by prior work on effective LLM context limits, we set L 0 L_{0} to 32k (Modarressi et al. [2025](https://arxiv.org/html/2508.00554v3#bib.bib18)). We then set the capacity for the factor portfolio to L∗=16​k L^{*}=16\text{k}, which reserves the remaining context (L 0−L∗L_{0}-L^{*}) for downstream reasoning within the Research Agents. This optimization problem is solved using a standard dynamic programming algorithm, and the portfolio is reconstructed every n n days to ensure the factors are continually adapted to market dynamics.

#### Researcher Contest

##### Optimization Objective

The objective is to dynamically allocate capital among research agents to maximize the portfolio’s future risk-adjusted return. While theoretically a classic portfolio optimization problem, accurately forecasting the inter-strategy covariance matrix (Σ t+n\Sigma_{t+n}) is prohibitively difficult. Therefore, our framework focuses on robustly predicting the performance of each individual agent.

##### Quantification via Hybrid Assessment

To effectively quantify an agent’s potential, we move beyond simple historical metrics. We construct a judger-augmented performance score, q i,t q_{i,t}, that provides a more holistic assessment. This score vector is composed of two parts:

*   •Realized Performance: Standard quantitative metrics calculated over a trailing m m-day window (e.g., realized Sharpe Ratio). 
*   •Judgmental Quality: A vector of qualitative scores from an LLM Judger Panel, which assesses the logical soundness and evidence quality of each agent’s submitted trading signal. 

##### Prediction

Agent’s performance, as measured by our hybrid score q i,t q_{i,t}, exhibits short-term momentum. Our analysis reveals an optimal prediction window of n=5 n=5 days for strategies, longer than the n=3 n=3 for information factors. This aligns with the intuition that reasoned investment strategies possess greater performance inertia. The prediction task is thus to learn a function that maps the historical sequence of judger-augmented scores, q i,t−m+1:t q_{i,t-m+1:t}, to an agent’s future utility, u^i,t+n\hat{u}_{i,t+n} (its predicted Sharpe Ratio). Consistent with the approach used for factor prediction, we employ LightGBM for this task, reusing the same simple and robust setup.

##### Allocation

For capital allocation, we employ a practical heuristic policy, π t\pi_{t}, based on the predicted utilities. This Predicted Sharpe Ratio-Weighted approach allocates capital proportionally to agents with positive predicted Sharpe Ratios. The weight w i,t w_{i,t} for agent i i is an instance of the framework’s weight vector 𝒲 t\mathcal{W}_{t} and is calculated as:

w i,t=max⁡(0,u^i,t+n)∑j=1 N max⁡(0,u^j,t+n)w_{i,t}=\frac{\max(0,\hat{u}_{i,t+n})}{\sum_{j=1}^{N}\max(0,\hat{u}_{j,t+n})}(8)

Experiments
-----------

### Experiment Setup

This section outlines our comprehensive experimental design, detailing the datasets, baselines, model configurations, and metrics used to evaluate our multi-agent trading system.

Our experiments utilize a real-world financial dataset encompassing news, corporate financials, and market data. To ensure a robust and leak-free evaluation, we strictly partition the data by time. The testing period (January-June 2025) is chosen to be entirely after the knowledge cutoff of our LLMs, eliminating potential leakage from their pre-training data. Correspondingly, all internal model training and parameter calibration, such as for the LightGBM models and momentum windows (m,n), are confined exclusively to the preceding training period (July-December 2024) to avoid any look-ahead bias. Trading simulations are conducted at a daily frequency on the A-share market, strictly adhering to T+1 settlement, daily price limits, and a 0.001 transaction cost.

We benchmark our system’s short-term, multi-stock trading performance against diverse strategies, including Broad Market Index (CSI ALL Share), Rule-based Methods (MACD, RSI&KDJ), Machine Learning (LGBM), Deep Learning (LSTM), Deep Reinforcement Learning(A2C,PPO) and Multi-Agent Systems like MASS (Guo et al. [2025](https://arxiv.org/html/2508.00554v3#bib.bib9)).

For LLM configuration, we primarily use DeepSeek-V3 (DeepSeek-AI et al. [2024](https://arxiv.org/html/2508.00554v3#bib.bib4)) as the backbone LLM since it is open-sourced, so that the experiments can be easily reproduced. Data Analysis Agents utilize DeepSeek-V3 model. Research Agents in the Research Team also use DeepSeek-V3 for Plan+React stages, but switch to DeepSeek-R1 for critical signal generation due to its enhanced reasoning capabilities.

We employ two categories of metrics for evaluation. Strategy Performance Metrics assess the final portfolio’s profitability and risk, including Cumulative Return (CR), Sharpe Ratio (SR), and Maximum Drawdown (MDD). The second category, Contest Effectiveness Metrics, evaluates the predictive power of our internal contest mechanisms. These include the Rank Information Coefficient (Rank IC) and Information Coefficient Information Ratio (ICIR), which are used to validate the effectiveness of the contests in both the Data and Research teams.While Rank IC and ICIR are not direct measures of profitability, they are crucial for validating the predictive quality of the factors and signals selected by our contests. The consistently high scores in these metrics demonstrate that our mechanism effectively identifies high-quality inputs, which in turn is the primary driver of the final portfolio’s outperformance.

![Image 4: Refer to caption](https://arxiv.org/html/2508.00554v3/figures/main_result.jpg)

Figure 4: Portfolio value over time. This figure compares the net value of the ContestTrade portfolio against various baseline strategies, demonstrating its performance over the experimental period.

### Main Results

Table 2: Strategy performance comparison with baseline models. The best performance for each metric is highlighted in bold.

As shown in Table[2](https://arxiv.org/html/2508.00554v3#Sx4.T2 "Table 2 ‣ Main Results ‣ Experiments ‣ ContestTrade: A Multi-Agent Trading System Based on Internal Contest Mechanism") and Figure[4](https://arxiv.org/html/2508.00554v3#Sx4.F4 "Figure 4 ‣ Experiment Setup ‣ Experiments ‣ ContestTrade: A Multi-Agent Trading System Based on Internal Contest Mechanism"), our proposed framework, ContestTrade, significantly outperforms all baseline models across all strategy performance metrics. It achieves a Cumulative Return (CR) of 52.80%, a Sharpe Ratio (SR) of 3.12, and a Maximum Drawdown (MDD) of only 12.41%. Compared to other multi-agent approaches like MASS, which employs fixed-agent cooperation without competitive selection, ContestTrade demonstrates vastly superior profitability and significantly better risk-adjusted returns. Even against strong traditional methods like RSI&KDJ, LSTM and PPO, ContestTrade shows substantial improvements in both return generation and risk management. The results clearly highlight ContestTrade’s robust performance, validating the efficacy of our proposed competitive, multi-agent framework in navigating complex financial markets.

To further investigate the source of ContestTrade’s superior performance, we evaluated the effectiveness of each internal contest mechanism within our framework. Table[3](https://arxiv.org/html/2508.00554v3#Sx4.T3 "Table 3 ‣ Main Results ‣ Experiments ‣ ContestTrade: A Multi-Agent Trading System Based on Internal Contest Mechanism") presents these results, reporting the predictive power (Rank IC and ICIR) of the final trading signals or factors selected by each team’s contest. The data demonstrates the high effectiveness of both contest mechanisms. The factor ranks predicted by the Data Analyst Contest achieved a strong mean Rank IC of 0.054 and an ICIR of 0.13, indicating not only high-quality factor identification but also remarkable consistency. Similarly, The signal ranks predicted by the Research Agent performed strongly with a Rank IC of 0.079 and ICIR of 0.18. Collectively, these results validate that our internal contest mechanisms are crucial drivers, effectively distilling noisy market information into valuable strategy.

Table 3: Experiments on the effectiveness of the internal contest mechanism. We report the predictive performance for both the Data Analyst and Researcher contests.

Ablation Studies
----------------

To validate the effectiveness and necessity of the key components within our ContestTrade framework, we conduct a comprehensive ablation study. We design several variants of our full model by removing one critical component at a time and then evaluate the impact on the overall strategy performance, measured by CR, SR, and MDD. The configurations are as follows:

*   •w/o LLM Judge: We disabled the LLM-based judging in the Researcher Contest. Final signals were randomly chosen from a Research Agent’s proposal, quantifying the LLM judge’s contribution. 
*   •w/o Contest - Researcher: This variant removes the entire competitive evaluation mechanism from the Research Team, with final signals selected randomly. This isolates the impact of inter-agent contests on performance. 
*   •w/o Contest - Data Analyst: We disabled the competitive evaluation within the Data Team. A randomly selected agent’s textual factor served as input, quantifying the contest mechanism’s contribution to data processing and denoising. 
*   •w/o Deep Research: Research Agents in the Research Team formulated signals solely on initial plans and textual factors, without using specialized financial tools for Deep Research. This evaluates autonomous information gathering. 
*   •w/o All: This most aggressive ablation removes both Data Analyst Contest and Researcher Contest, plus Research Agents’ Deep Research capability. This provides a baseline understanding of performance without any core proposed mechanisms. 

![Image 5: Refer to caption](https://arxiv.org/html/2508.00554v3/figures/ablation_study.jpg)

Figure 5: Portfolio value over time. This figure compares the portfolio value of the full ContestTrade model against various ablated configurations over time.

Table 4: Ablation study of the key components within our ContestTrade framework. ”w/o” indicates removing the specified component.

As shown in Figure [5](https://arxiv.org/html/2508.00554v3#Sx5.F5 "Figure 5 ‣ Ablation Studies ‣ ContestTrade: A Multi-Agent Trading System Based on Internal Contest Mechanism") and Table [4](https://arxiv.org/html/2508.00554v3#Sx5.T4 "Table 4 ‣ Ablation Studies ‣ ContestTrade: A Multi-Agent Trading System Based on Internal Contest Mechanism"), our ablation study clearly demonstrates that every component within ContestTrade is crucial for its superior performance. The Researcher Contest mechanism, particularly its Deep Research and LLM Judge, works synergistically to deliver high Cumulative Return (CR), Sharpe Ratio (SR), and low Maximum Drawdown (MDD). Removing any part significantly degrades results, with full removal leading to catastrophic performance drops. This highlights the indispensable role of each design element in robust portfolio management.

Conclusion & Future works
-------------------------

In this paper, we introduced ContestTrade, a novel multi-agent framework addressing the challenges of inconsistent decision-making and market noise in LLM-based trading systems. Drawing inspiration from institutional investment practices, ContestTrade features Data team, Research team and internal contest mechanisms that continuously evaluate agent and select high-quality outputs. Our experiments confirm ContestTrade’s superior performance over a range of baseline strategies across key metrics, showing higher returns, better risk-adjusted performance, and lower downside risk. Furthermore, the high Rank IC and ICIR from internal contest validate its competitive, performance-based quantification and prediction.

Our contributions include a dynamic multi-agent architecture with internal contest, Deep Research methodology with financial toolkits, and robust information denoising. Future work involves larger-scale simulations, stronger reasoning frameworks, broader market applications (e.g., U.S. equities, forex), and diverse data integration, establishing ContestTrade as a generalizable and scalable paradigm for intelligent, autonomous trading.

References
----------

*   Bai et al. (2023) Bai, J.; Bai, S.; Chu, Y.; Cui, Z.; Dang, K.; Deng, X.; Fan, Y.; Ge, W.; Han, Y.; Huang, F.; Hui, B.; et al. 2023. QWEN TECHNICAL REPORT. _arXiv preprint arXiv:2309.16609_. 
*   Byrd, Hybinette, and Balch (2020) Byrd, D.; Hybinette, M.; and Balch, T.H. 2020. ABIDES: Towards High-Fidelity Multi-Agent Market Simulation. In _Proceedings of the 2020 ACM SIGSIM Conference on Principles of Advanced Discrete Simulation_, SIGSIM-PADS ’20, 11–22. New York, NY, USA: Association for Computing Machinery. ISBN 9781450375924. 
*   Chroma (2024) Chroma. 2024. Context Rot: How LLMs Degrade with Context. https://research.trychroma.com/context-rot. 
*   DeepSeek-AI et al. (2024) DeepSeek-AI; Liu, A.; Feng, B.; Xue, B.; Wang, B.; Wu, B.; Lu, C.; Zhao, C.; Deng, C.; Zhang, C.; Ruan, C.; Dai, D.; Guo, D.; Yang, D.; Chen, D.; Ji, D.; Li, E.; Lin, F.; Dai, F.; Luo, F.; Hao, G.; Chen, G.; Li, G.; Zhang, H.; Bao, H.; Xu, H.; Wang, H.; Zhang, H.; Ding, H.; Xin, H.; Gao, H.; Li, H.; Qu, H.; Cai, J.L.; Liang, J.; Guo, J.; Ni, J.; Li, J.; Wang, J.; Chen, J.; Chen, J.; Yuan, J.; Qiu, J.; Li, J.; Song, J.; Dong, K.; Hu, K.; Gao, K.; Guan, K.; Huang, K.; Yu, K.; Wang, L.; Zhang, L.; Xu, L.; Xia, L.; Zhao, L.; Wang, L.; Zhang, L.; Li, M.; Wang, M.; Zhang, M.; Zhang, M.; Tang, M.; Li, M.; Tian, N.; Huang, P.; Wang, P.; Zhang, P.; Wang, Q.; Zhu, Q.; Chen, Q.; Du, Q.; Chen, R.J.; Jin, R.L.; Ge, R.; Zhang, R.; Pan, R.; Wang, R.; Xu, R.; Zhang, R.; Chen, R.; Li, S.S.; Lu, S.; Zhou, S.; Chen, S.; Wu, S.; Ye, S.; Ye, S.; Ma, S.; Wang, S.; Zhou, S.; Yu, S.; Zhou, S.; Pan, S.; Wang, T.; Yun, T.; Pei, T.; Sun, T.; Xiao, W.L.; and Zeng, W. 2024. DeepSeek-V3 Technical Report. _CoRR_, abs/2412.19437. 
*   Ding et al. (2024) Ding, H.; Wang, J.; Li, Y.; and Chen, H. 2024. Large Language Model Agent in Financial Trading: A Survey. _arXiv preprint arXiv:2408.06361_. 
*   Engle (1982) Engle, R.F. 1982. Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of UK Inflation. _Econometrica_, 50(4): 987–1008. 
*   Fatouros et al. (2024) Fatouros, G.; Metaxas, K.; Soldatos, J.; and Kyriazis, D. 2024. Can Large Language Models Beat Wall Street? Unveiling the Potential of AI in Stock Selection. _arXiv preprint arXiv:2401.03737_. 
*   Gode and Sunder (1993) Gode, D.K.; and Sunder, S. 1993. Allocative Efficiency of Markets with Zero-Intelligence Traders: Market as a Partial Substitute for Individual Rationality. _Journal of Political Economy_, 101(1): 119–137. 
*   Guo et al. (2025) Guo, T.; Shen, H.; Huang, J.; Mao, Z.; Luo, J.; Chen, Z.; Liu, X.; Xia, B.; Liu, L.; Ma, Y.; and Zhang, M. 2025. MASS: Multi-Agent Simulation Scaling for Portfolio Construction. _arXiv preprint arXiv:2505.10278_. 
*   Koa et al. (2024) Koa, K.J.; Ma, Y.; Ng, R.; and Chua, T.-S. 2024. Learning to Generate Explainable Stock Predictions using Self-Reflective Large Language Models. In _The Web Conference 2024_. 
*   LeBaron (2006) LeBaron, B. 2006. Agent-Based Computational Finance. In Tesfatsion, L.; and Judd, K.L., eds., _Handbook of Computational Economics_, volume 2, 1187–1233. Elsevier. 
*   Li et al. (2024) Li, H.; Yu, Y.; Chen, Z.; Jiang, Y.; Li, Y.; Zhang, D.; Liu, R.; Suchow, J.W.; and Khashanah, K. 2024. FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design. In _ICLR 2024 Workshop on Large Language Model (LLM) Agents_. 
*   Li et al. (2023) Li, Y.; Yu, Y.; Li, H.; Chen, Z.; and Khashanah, K. 2023. TradingGPT: Multi-Agent System with Layered Memory and Distinct Characters for Enhanced Financial Trading Performance. Papers 2309.03736, arXiv.org. 
*   Liu et al. (2021) Liu, Z.; Huang, D.; Huang, K.; Li, Z.; and Zhao, J. 2021. FinBERT: a pre-trained financial language representation model for financial text mining. In _Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence_, IJCAI’20. ISBN 9780999241165. 
*   Lopez-Lira and Tang (2024) Lopez-Lira, A.; and Tang, Y. 2024. Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models. _arXiv preprint arXiv:2304.07619_. 
*   Luukkonen et al. (2023) Luukkonen, R.; Komulainen, V.; Luoma, J.; Eskelinen, A.; Kanerva, J.; Kupari, H.-M.; Ginter, F.; Laippala, V.; Muennighoff, N.; Piktus, A.; Wang, T.; Tazi, N.; Scao, T.; Wolf, T.; Suominen, O.; Sairanen, S.; Merioksa, M.; Heinonen, J.; Vahtola, A.; Antao, S.; and Pyysalo, S. 2023. FinGPT: Large Generative Models for a Small Language. In Bouamor, H.; Pino, J.; and Bali, K., eds., _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_, 2710–2726. Singapore: Association for Computational Linguistics. 
*   Malkiel (1973) Malkiel, B.G. 1973. _A Random Walk Down Wall Street_. W.W. Norton & Company. ISBN 978-0393358384. 
*   Modarressi et al. (2025) Modarressi, A.; Deilamsalehy, H.; Dernoncourt, F.; Bui, T.; Rossi, R.A.; Yoon, S.; and Schuetze, H. 2025. NoLiMa: Long-Context Evaluation Beyond Literal Matching. In _Forty-second International Conference on Machine Learning_. 
*   Wang, Izumi, and Sakaji (2024) Wang, M.; Izumi, K.; and Sakaji, H. 2024. LLMFactor: Extracting Profitable Factors through Prompts for Explainable Stock Movement Prediction. In Ku, L.-W.; Martins, A.; and Srikumar, V., eds., _Findings of the Association for Computational Linguistics: ACL 2024_, 3120–3131. Bangkok, Thailand: Association for Computational Linguistics. 
*   Wang, Yuan, and Ni (2024) Wang, S.; Yuan, H.; and Ni, J., Lionel M.and Guo. 2024. QuantAgent: Seeking Holy Grail in Trading by Self-Improving Large Language Model. _arXiv preprint arXiv:2402.03755_. 
*   Wang et al. (2023) Wang, S.; Yuan, H.; Zhou, L.; Ni, H.-Y., Lionel M.and Shum; and Guo, J. 2023. Alpha-GPT: Human-AI Interactive Alpha Mining for Quantitative Investment. _arXiv preprint arXiv:2308.00016_. 
*   Wu et al. (2023) Wu, S.; Irsoy, O.; Lu, S.; Dabravolski, V.; Dredze, M.; Gehrmann, S.; Kambadur, P.; Rosenberg, D.; and Mann, G. 2023. BloombergGPT: A Large Language Model for Finance. _arXiv preprint arXiv:2303.17564_. 
*   Xiao et al. (2024) Xiao, Y.; Sun, E.; Luo, D.; and Wang, W. 2024. TradingAgents: Multi-Agents LLM Financial Trading Framework. _arXiv preprint arXiv:2412.20138_. 
*   Xing (2024) Xing, F.Z. 2024. HAD: Heterogeneous multi-Agent framework for Financial sentiment analysis. _arXiv preprint arXiv:2401.05799_. 
*   Yang et al. (2024) Yang, H.; Zhang, B.; Wang, N.; Guo, C.; Zhang, X.; Lin, L.; Wang, J.; Zhou, T.; Guan, M.; Zhang, R.; and Wang, C.D. 2024. FinRobot: An Open-Source AI Agent Platform for Financial Applications using Large Language Models. _arXiv preprint arXiv:2405.14767_. 
*   Yao et al. (2023) Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan, K.; and Cao, Y. 2023. React: Synergizing reasoning and acting in language models. In _International Conference on Learning Representations (ICLR)_. 
*   Yu et al. (2025) Yu, Y.; Yao, Z.; Li, H.; Deng, Z.; Jiang, Y.; Cao, Y.; Chen, Z.; Suchow, J.W.; Cui, Z.; Liu, R.; Xu, Z.; Zhang, D.; Subbalakshmi, K.; Xiong, G.; He, Y.; Huang, J.; Li, D.; and Xie, Q. 2025. FINCON: a synthesized LLM multi-agent system with conceptual verbal reinforcement for enhanced financial decision making. In _Proceedings of the 38th International Conference on Neural Information Processing Systems_, NIPS ’24. Red Hook, NY, USA: Curran Associates Inc. ISBN 9798331314385. 
*   Yuan, Wang, and Guo (2024) Yuan, H.; Wang, S.; and Guo, J. 2024. Alpha-GPT 2.0: Human-in-the-Loop AI for Quantitative Investment. _arXiv preprint arXiv:2402.09746_. 
*   Zhang et al. (2024a) Zhang, H.; Hua, F.; Xu, C.; Kong, H.; Zuo, R.; and Guo, J. 2024a. Unveiling the Potential of Sentiment: Can Large Language Models Predict Chinese Stock Price Movements? _arXiv preprint arXiv:2306.14222_. 
*   Zhang et al. (2024b) Zhang, W.; Zhao, L.; Xia, H.; Sun, S.; Sun, J.; Qin, M.; Li, X.; Zhao, Y.; Zhao, Y.; Cai, X.; Zheng, L.; Wang, X.; and An, B. 2024b. A Multimodal Foundation Agent for Financial Trading: Tool-Augmented, Diversified, and Generalist. In _Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_, KDD ’24, 4314–4325. New York, NY, USA: Association for Computing Machinery. ISBN 9798400704901. 
*   Zhou et al. (2025) Zhou, Y.; Liu, H.; Chen, Z.; Tian, Y.; and Chen, B. 2025. GSM-Infinite: How Do your LLMs Behave over Infinitely Increasing Reasoning Complexity and Context Length? In _Forty-second International Conference on Machine Learning_.

