Title: QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading

URL Source: https://arxiv.org/html/2509.09995

Markdown Content:
1]Stony Brook University 2]Carnegie Mellon University 3]University of British Columbia 4]Yale University 5]Fudan University \contribution[†]Equal contribution

###### Abstract

Recent advances in Large Language Models (LLMs) have shown remarkable capabilities in financial reasoning and market understanding. Multi-agent LLM frameworks such as TradingAgent and FINMEM augment these models to long-horizon investment tasks by leveraging fundamental and sentiment-based inputs for strategic decision-making. However, these approaches are ill-suited for the high-speed, precision-critical demands of High-Frequency Trading (HFT). HFT typically requires rapid, risk-aware decisions driven by structured, short-horizon signals, such as technical indicators, chart patterns, and trend features. These signals stand in sharp contrast to the long-horizon, text-driven reasoning that characterizes most existing LLM-based systems in finance. To bridge this gap, we introduce QuantAgent, the first multi-agent LLM framework explicitly designed for high-frequency algorithmic trading. The system decomposes trading into four specialized agents, Indicator, Pattern, Trend, and Risk, each equipped with domain-specific tools and structured reasoning capabilities to capture distinct aspects of market dynamics over short temporal windows. Extensive experiments across nine financial instruments, including Bitcoin and Nasdaq futures, demonstrate that QuantAgent consistently outperforms baseline methods, achieving higher predictive accuracy at both 1-hour and 4-hour trading intervals across multiple evaluation metrics. Our findings suggest that coupling structured trading signals with LLM-based reasoning provides a viable path for traceable, real-time decision systems in high-frequency financial markets.

1 Introduction
--------------

In quantitative finance, technical analysis treats historical price action as the most immediate and information-dense reflection of market conditions (Pring, [1991](https://arxiv.org/html/2509.09995v3#bib.bib1)). The central premise is that market dynamics, including fundamentals, macro events, institutional flows, and collective sentiment, are ultimately embedded in price movements (Murphy, [1999](https://arxiv.org/html/2509.09995v3#bib.bib2)). Each bar, defined by its open, high, low, and close (OHLC), provides a compact yet universal representation of short-horizon market behavior. This structure enables systematic detection of recurring setups such as trends, reversals, breakouts, and momentum shifts across asset classes ranging from equities and commodities to digital assets (Moskowitz et al., [2012](https://arxiv.org/html/2509.09995v3#bib.bib3)). Under the efficient market hypothesis (Fama, [1970](https://arxiv.org/html/2509.09995v3#bib.bib4)), prices adjust rapidly to public information, making patterns in OHLC bars a natural substrate for short-term prediction without reliance on lagging textual inputs.

![Image 1: Refer to caption](https://arxiv.org/html/2509.09995v3/x1.png)

Figure 1: Workflows of IndicatorAgent, PatternAgent, and TrendAgent. IndicatorAgent interprets signals from MACD, RSI, ROC, and Williams %R; PatternAgent detects formations such as double bottoms; TrendAgent extracts directional flow via support and resistance channels.

Large Language Models (LLMs) have recently demonstrated impressive capabilities in multi-step reasoning, tool use, and interpretable decision-making (OpenAI et al., [2024](https://arxiv.org/html/2509.09995v3#bib.bib5)). These capabilities are directly relevant to quantitative trading (Yang et al., [2023](https://arxiv.org/html/2509.09995v3#bib.bib6)), which heavily depends on integrating heterogeneous signals, applying systematic trading rules, and controlling execution risks. However, most existing LLM-driven financial frameworks operate primarily on textual inputs, such as news articles, social media streams, or earnings reports (Nguyen et al., [2015](https://arxiv.org/html/2509.09995v3#bib.bib7); Xiao et al., [2025](https://arxiv.org/html/2509.09995v3#bib.bib8); Zakir et al., [2025](https://arxiv.org/html/2509.09995v3#bib.bib9); Zhang et al., [2023a](https://arxiv.org/html/2509.09995v3#bib.bib10)). This reliance introduces two major limitations: (i) textual signals typically lag price discovery and are incorporated into markets only after the fact (Chordia et al., [2013](https://arxiv.org/html/2509.09995v3#bib.bib11)), and (ii) such data is noisy, unstructured, and difficult to validate (Liu et al., [2022a](https://arxiv.org/html/2509.09995v3#bib.bib12)). Since short-horizon market dynamics are already encoded in OHLC bars, a more direct approach is to align LLM reasoning with structured price-based signals. To the best of our knowledge, no prior work has developed an LLM-based framework for high-frequency trading (HFT) that operates directly on OHLC data.

In this paper, we propose QuantAgent (Figure [1](https://arxiv.org/html/2509.09995v3#S1.F1 "Figure 1 ‣ 1 Introduction ‣ QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading")), the first multi-agent LLM framework tailored to high-frequency algorithmic trading. Specifically, QuantAgent decomposes the trading process into four specialized agents – IndicatorAgent, PatternAgent, TrendAgent, and RiskAgent – each designed to capture a complementary dimension of technical analysis. IndicatorAgent condenses raw OHLC bars into robust technical indicators, providing a noise-resistant summary of recent market behavior. PatternAgent chart formations such as peaks, troughs, and consolidations, leveraging the multimodal reasoning abilities of LLMs (Nison, [2001](https://arxiv.org/html/2509.09995v3#bib.bib13)). TrendAgent identifies directional bias from short-horizon price dynamics, while RiskAgent integrates all signals into a coherent risk–reward profile. Final trade decisions emerge from the interaction of these agents, yielding traceable, language-native rationales that can be inspected alongside execution (Schick et al., [2023](https://arxiv.org/html/2509.09995v3#bib.bib14)).

We evaluate QuantAgent on a multi-asset benchmark spanning commodities, equities, cryptocurrencies, and volatility indices. At 1-hour and 4-hour bar resolutions, QuantAgent consistently outperforms baselines across both directional accuracy and return-based metrics, with particularly pronounced gains in equity markets. Rolling-window validation further demonstrates robust generalization, achieving up to 80% directional accuracy in forecasting short-term price movements. Besides its strong empirical performance, QuantAgent provides natural-language rationales for trading decisions, enabling a degree of traceability and interpretability often missing in traditional algorithmic strategies.

2 Related Works
---------------

Agent-Based LLMs for Financial Decision-Making. The design of QuantAgent builds on recent work that organizes LLMs into multi-agent systems for financial decision-making. FINCON (Yu et al., [2024](https://arxiv.org/html/2509.09995v3#bib.bib15)) introduces a manager–analyst hierarchy trained via verbal reinforcement learning, while TradingAgents (Xiao et al., [2025](https://arxiv.org/html/2509.09995v3#bib.bib8)) models institutional workflows through agent communication, prioritizing interpretability over the low-latency, price-driven logic required in high-frequency trading (Tumarkin and Whitelaw, [2001](https://arxiv.org/html/2509.09995v3#bib.bib16)). As a line of work, these systems demonstrate the potential of LLM-based agents in finance, but their heavy reliance on textual inputs leaves them ill-suited for the structured, low-latency signals required in HFT scenarios. More recently, RD-Agent(Q) (Li et al., [2025](https://arxiv.org/html/2509.09995v3#bib.bib17)) takes a significant step forward by shifting to structured, data-centric signals and automating factor–model co-optimization.However, RD-Agent(Q) remains constrained to daily-resolution strategies and slower research-feedback cycles, making it less suitable for real-time decision-making in high-frequency contexts.

Quantitative Trading Based on Indicators and Patterns. Prior to LLM-based agents, quantitative trading systems are predominantly built on technical indicators such as trends, volatility, and momentum for intraday decision-making. Early studies show that nonlinear price patterns can exhibit predictive power (Lo et al., [2000](https://arxiv.org/html/2509.09995v3#bib.bib18)), but subsequent work highlight challenges including overfitting and researcher bias (Chen and Chen, [2016](https://arxiv.org/html/2509.09995v3#bib.bib19)). Momentum strategies (Jegadeesh and Titman, [1993](https://arxiv.org/html/2509.09995v3#bib.bib20); Moskowitz et al., [2012](https://arxiv.org/html/2509.09995v3#bib.bib3)) are widely adopted to capture trend persistence, while heuristics such as Elliott wave theory (Prechter, [2005](https://arxiv.org/html/2509.09995v3#bib.bib21)) and curated pattern libraries attempt to model higher-order market structures. Although these indicator- and pattern-based approaches are interpretable and computationally efficient, they often struggle in volatile or noisy environments, undermining their effectiveness in high-frequency settings. These limitations motivate us to design a framework that fuses structured price signals with LLM-based reasoning, enabling more adaptive and interpretable trading systems.

3 QuantAgent
------------

To bridge the gap between traditional high-frequency quantitative trading and recent advances in multi-agent LLM systems, we introduce QuantAgent, a collaborative framework for low-latency market decision-making. QuantAgent integrates classical technical analysis with prompt-structured LLM reasoning, enabling modular and interpretable financial intelligence. Built on LangGraph (LangChain, [2025](https://arxiv.org/html/2509.09995v3#bib.bib22)), the system simulates the workflow of institutional trading desks, where specialized agents execute distinct analytical roles to support rapid and coordinated decision-making.

In contrast to prior LLM-based frameworks that incorporate external sources such as news or social media sentiment, QuantAgent operates solely on price-derived market signals. This design choice reflects the efficient market hypothesis, which posits that asset prices incorporate available information by aggregating the actions and beliefs of market participants over time (Murphy, [1999](https://arxiv.org/html/2509.09995v3#bib.bib2)). By grounding analysis exclusively in OHLC data and technical indicators, QuantAgent avoids the latency, noise, and unpredictability of textual sources, while remaining fast, interpretable, and directly aligned with the demands of high-frequency trading.

The system decomposes trading into four specialized agent, IndicatorAgent, PatternAgent, TrendAgent, and RiskAgent, that communicate through structured prompts. Each agent captures a complementary perspective on short-horizon market dynamics: numerical indicators, geometric patterns, directional momentum, and integrated decision-making. In the following subsections, we describe the design of each agent in detail and formalize the technical components underlying our framework.

Input:

P 0:T−1,N,τ P_{0:T-1},\,N,\,\tau

1 for _t=N−1 t=N-1 to T−1 T-1_ do

2 Fit OLS on highs/lows to get

m r,m s m_{r},m_{s}
;

3

κ t←(m r+m s)/2\kappa_{t}\leftarrow(m_{r}+m_{s})/2
;

4 if _κ t>τ\kappa\_{t}>\tau_ then Trend

←\leftarrow
Uptrend;

5 else if _κ t<−τ\kappa\_{t}<-\tau_ then Trend

←\leftarrow
Downtrend;

6 else Trend

←\leftarrow
Sideways;

7

8 end for

9 Render chart

𝒦 t​(P t,κ t,Trend)\mathcal{K}_{t}(P_{t},\kappa_{t},\text{Trend})
;

Algorithm 1 Slope-aware trend detection over a candlestick sequence P 0:T−1 P_{0:T-1}

![Image 2: Refer to caption](https://arxiv.org/html/2509.09995v3/x2.png)

Figure 2: Workflow of RiskAgent. Signals from Trend, Indicator, and Pattern are aggregated into a radar chart. DecisionAgent predicts with stop-loss and take-profit.

### 3.1 IndicatorAgent

IndicatorAgent constitutes the initial analytical module of our framework, responsible for transforming raw OHLC sequences into structured quantitative signals (Figure [1](https://arxiv.org/html/2509.09995v3#S1.F1 "Figure 1 ‣ 1 Introduction ‣ QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading")). In high-frequency trading (HFT), where decisions must be executed under strict latency constraints, technical indicators provide compact representations that highlight shifts in market momentum and sentiment. Formally, this process can be viewed as a mapping from price tuples to an interpretable signal space, (O,H,L,C)↦𝒮(O,H,L,C)\mapsto\mathcal{S}, where 𝒮\mathcal{S} denotes a set of derived features summarizing short-horizon market dynamics (Lo et al., [2000](https://arxiv.org/html/2509.09995v3#bib.bib18)). By abstracting low-level price data into high-level features, IndicatorAgent facilitates fast and interpretable downstream reasoning.

To achieve this, it converts raw OHLC values into a compact set of informative technical indicators. Specifically, IndicatorAgent uses five widely used technical indicators to extract signals from market data. RSI captures momentum and flags overbought or oversold zones (Wilder, [1978](https://arxiv.org/html/2509.09995v3#bib.bib23)), while MACD tracks convergence or divergence between short and long term price trends (Appel, [2005](https://arxiv.org/html/2509.09995v3#bib.bib24)). RoC measures the speed of price changes (Murphy, [1999](https://arxiv.org/html/2509.09995v3#bib.bib2)), STOCH identifies turning points based on recent highs and lows (Investopedia, [n.d.](https://arxiv.org/html/2509.09995v3#bib.bib25)), and WILLR detects price drops from recent peaks to signal possible reversals (Williams, [2011](https://arxiv.org/html/2509.09995v3#bib.bib26)).

Together, these indicators capture both short-term volatility and longer-term momentum. IndicatorAgent integrates them into a structured summary that reflects current market conditions. Rather than applying fixed rules, it interprets signals in context such as highlighting dynamics like momentum shifts, overbought or oversold zones, and sudden reversals (Murphy, [1999](https://arxiv.org/html/2509.09995v3#bib.bib2)). This contextual reasoning enables DecisionAgent to act on timely, relevant insights, improving responsiveness in fast-moving trading environments (Lo et al., [2000](https://arxiv.org/html/2509.09995v3#bib.bib18)).

#### 3.1.1 PatternAgent Design

While IndicatorAgent offers useful signals, these numerical indicators can become unclear—especially when price movement stalls or enters a new regime (Murphy, [1999](https://arxiv.org/html/2509.09995v3#bib.bib2); Lo et al., [2000](https://arxiv.org/html/2509.09995v3#bib.bib18)). To address these limitations, our PatternAgent introduces a more visual and structural multi-modal reasoning perspective (Figure [1](https://arxiv.org/html/2509.09995v3#S1.F1 "Figure 1 ‣ 1 Introduction ‣ QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading")).

Upon receiving a request to analyze market patterns, PatternAgent first utilizes LLM-binded tools to generate clear, simplified candlestick charts directly from raw price data. These visualizations present recent market behavior without explicitly highlighting specific geometric shapes or details. Our agent, instead, automatically detects essential visual features from price movements, such as significant highs and lows, structural symmetry, and potential reversal formations,effectively capturing key visual patterns used in technical analysis (Wang et al., [2023](https://arxiv.org/html/2509.09995v3#bib.bib27)).

Using this information, PatternAgent compares the current market structure to an extensive library of well-known patterns described in clear language. Each pattern in this library includes concise yet detailed descriptions of its visual form and the market behaviors it typically signals. Through this comparison, PatternAgent identifies the most plausible match and evaluates its relevance to the current context. This process blends visual understanding with language-based reasoning to recognize patterns and assess whether their context—such as preceding trends or volatility—makes them likely to signal a meaningful price move.

By translating complex visual signals into concise and interpretable summaries, PatternAgent plays a key role in bridging raw chart data and high-level reasoning, allowing the system to integrate visual structure into its overall market understanding (Lo et al., [2000](https://arxiv.org/html/2509.09995v3#bib.bib18)).

#### 3.1.2 TrendAgent Design

Canonical chart patterns, such as double bottoms or flags, can be reliably interpreted only when evaluated within the context of a well-defined trend (Pring, [1991](https://arxiv.org/html/2509.09995v3#bib.bib1)). By tracking both the direction and steepness of price movements over time, TrendAgent provides a structural representation of trend dynamics, clarifying whether a detected pattern is consistent with the prevailing trend, signals a potential reversal, or indicates a phase of non-directional price congestion (Elder, [2002](https://arxiv.org/html/2509.09995v3#bib.bib28)).

As shown in Figure [1](https://arxiv.org/html/2509.09995v3#S1.F1 "Figure 1 ‣ 1 Introduction ‣ QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading"), TrendAgent generates an annotated K-line chart 𝒦 t\mathcal{K}_{t}, which includes a trend channel 𝒞 t\mathcal{C}_{t} that captures the price trajectory through two fitted lines: the upper resistance line ℛ t​(x)=m r​x+b r\mathcal{R}_{t}(x)=m_{r}x+b_{r}, drawn through recent local highs, and the lower support line 𝒮 t​(x)=m s​x+b s\mathcal{S}_{t}(x)=m_{s}x+b_{s}, drawn through recent local lows (Lo et al., [2000](https://arxiv.org/html/2509.09995v3#bib.bib18)). The trendlines, computed using ordinary least squares regression as outlined in Algorithm [1](https://arxiv.org/html/2509.09995v3#alg1 "Algorithm 1 ‣ 3 QuantAgent ‣ QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading"), serve to characterize price direction, strength, and potential breakout zones. The average slope κ t=m r+m s 2\kappa_{t}=\frac{m_{r}+m_{s}}{2} provides a basic estimate of directional drift. However, reliable trend classification requires more than just the slope sign, as market noise can obscure short-term movements.

To address that, TrendAgent examines the geometric relationship between the lines—such as parallel upward slopes indicating strong trends, or converging lines forming a wedge that suggests indecision or accumulation. These structural cues allow the agent to reason about not just direction but also the confidence and stability of the trend. They work in conjunction with shape-based patterns and momentum signals identified by other agents, improving decision-making and reinforcing signal coherence across the system (Kirkpatrick and Dahlquist, [2015](https://arxiv.org/html/2509.09995v3#bib.bib29)).

#### 3.1.3 RiskAgent and DecisionAgent

RiskAgent, shown in Figure [2](https://arxiv.org/html/2509.09995v3#S3.F2 "Figure 2 ‣ 3 QuantAgent ‣ QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading"), translates technical insights into risk-aware trade boundaries, reflecting the central role of risk control in real-world trading. Instead of signal generation, RiskAgent integrates other agents’ output into a unified risk-reward framework. It sets a fixed stop-loss value ρ=0.0005\rho=0.0005 to account for short-term volatility and computes a take-profit level ℛ=r⋅ρ\mathcal{R}=r\cdot\rho, where r∈[1.2,1.8]r\in[1.2,1.8] is predicted by the LLM. This forms a structured decision zone bounded by stop-loss, entry, and take-profit levels. Within this zone, the agent reasons over signal quality and predefined risk exposure to ensure consistent and informed actions. By aligning domain knowledge with real-time constraints, RiskAgent grounds high-level analysis in practical execution under uncertain and fast-moving market environment.

![Image 3: Refer to caption](https://arxiv.org/html/2509.09995v3/x3.png)

Figure 3: Workflow of DecisionAgent. LLM summarizes upstream signals and consolidates them into structured outputs: direction, reasoning, trade setup, and post-trade reflection. It then formulates an executable order with rationale, ready for market submission.

The final stage of the framework is DecisionAgent, which functions as the reasoning and execution layer. As illustrated in Figure [3](https://arxiv.org/html/2509.09995v3#S3.F3 "Figure 3 ‣ 3.1.3 RiskAgent and DecisionAgent ‣ 3.1 IndicatorAgent ‣ 3 QuantAgent ‣ QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading"), it integrates the outputs of upstream agents to decide between a LONG or SHORT position. Since holding is not permitted, the agent forecasts short-term market direction over the next three candlesticks and generates actionable decisions aligned with the aggregated signals from the overall system (Kissell, [2013](https://arxiv.org/html/2509.09995v3#bib.bib30)).

Specifically, DecisionAgent takes in an aggregated signal state from IndicatorAgent, PatternAgent, and TrendAgent, and outputs a structured trading decision comprising the predicted direction (LONG or SHORT), a concise justification, and a risk–reward ratio conditioned on market context (Kissell, [2013](https://arxiv.org/html/2509.09995v3#bib.bib30)). The agent integrates heterogeneous evidence and evaluates the consistency across upstream signals, proceeding only when the majority align and are reinforced by confirmations such as indicator crossovers, completed breakout formations, or price interactions with major trend boundaries. This layered reasoning filters out noisy or conflicting inputs and yields confident, high-quality decisions. Consequently, the outputs are not only optimized for execution in high-frequency settings but also more robust and interpretable than those produced by traditional rule-based systems.

4 Experiments
-------------

We evaluate our QuantAgent framework in a fair and comprehensive manner, with trading decisions generated autonomously without prior demonstrations or supervised fine-tuning. Building on the structured reasoning provided by upstream agents, our system uses only recent candlestick data and basic context (e.g., asset type and time interval) to predict short-term market direction. It then generates clear trade suggestions with human-readable explanations, allowing us to evaluate its performance in realistic settings. The experiment is designed to test the framework’s effectiveness in realistic, data-limited environments where fast, adaptive decision-making is required.

Benchmark Construction and Evaluation Protocol. To support evaluation, we build benchmarks of 4-hour and 1-hour OHLC data across key asset classes such as cryptocurrencies, equity indices, and commodities. For each asset, 5000 historical bars are collected via a public TradingView data extraction tool API. From this, 100 evaluation segments per asset are sampled, each with 100 consecutive candlesticks—the last three withheld to prevent test-time leakage. Details of benchmark are illustrated in Appendix [E](https://arxiv.org/html/2509.09995v3#A5 "Appendix E Benchmark Detail ‣ QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading").

The system processes agents’ analysis to generate a structured trade report containing a directional prediction (LONG or SHORT), a brief textual rationale, and an estimated risk–reward ratio. Among these outputs, the directional decision and risk–reward estimate are used for quantitative evaluation.

Baselines. We evaluate four categories of baselines: (i) Random Methods, which performs random selection of market trend between LONG and SHORT and risk-reward ratio ∈[1.2,1.8]\in[1.2,1.8]. (ii) Linear Regression, which serves as a rule-based baseline. It fits a linear model to a 40-bar window of recent closing prices and use the slope of the fitted line to classify future trend. If the slope exceeds 0, the system predicts LONG. Otherwise, it predicts SHORT. (iii) XGBoost, which serves as a tree-based supervised learning model. It uses technical indicators extracted via a public API TA-Lib (TA‑Lib Development Team, [2025](https://arxiv.org/html/2509.09995v3#bib.bib31)) including RSI, MACD, and SMA. This model is trained on hundreds of sliding-window sample across 50 randomly selected csv files. The trained model is tested on the rest 50 csv files with the same metrics as other methods. A majority-vote rule is applied across predictions to produce a final LONG/SHORT/HOLD decision where HOLD decision is disregarded during final average calculation. (iv) QuantAgent, our LLM-based approach, perform short-term prediction using multi-modal multi-agent cooperation.

Evaluation Metrics. To evaluate prediction accuracy, we compare the LLM’s directional forecast to the next three candlesticks. For a LONG decision, each candle that closes above the last close counts as a correct hit (max 3); for SHORT, each close below the current close counts. Accuracy is defined as α=ℂ 𝒯\mathcal{\alpha}=\frac{\mathbb{C}}{\mathcal{T}}, where ℂ\mathbb{C} is the number of correct predictions and 𝒯\mathcal{T} is the total evaluated. Each test yields a score from 0 to 3, and the average is computed over all samples. This aligns with the Mean Directional Accuracy metric used in forecasting (Pesaran and Timmermann, [2004](https://arxiv.org/html/2509.09995v3#bib.bib32)).

In addition, we evaluate trade outcomes based on multiple Rate of Return (RoR) estimators (Fama and MacBeth, [1973](https://arxiv.org/html/2509.09995v3#bib.bib33)) commonly used in HFT, each is to quantify the profitability of a trade by measuring the relative gain or loss between the entry price and exit. All rate-of-return metrics in our framework, including ℛ c​c\mathcal{R}_{cc}, ℛ max\mathcal{R}_{\text{max}}, and ℛ min\mathcal{R}_{\text{min}}, incorporate risk-constrained execution, simulating realistic stop-loss and take-profit behavior. Specifically, a trade is exited at the first price among the next three candlesticks that hits either the stop-loss or reward threshold. We adopt a fixed stop-loss threshold ρ=0.0005\rho=0.0005 (i.e., 0.05%), selected to reflect the relatively small fluctuations typical within a short three-candlestick forecast horizon, consistent with prior work (Kissell, [2013](https://arxiv.org/html/2509.09995v3#bib.bib30)). The corresponding reward threshold is determined using the LLM-generated risk–reward ratio r=ℛ ρ r=\frac{\mathcal{R}}{\rho}, where ℛ=r∗ρ\mathcal{R}=r*\rho is the maximum allowed gain and −r-r is the maximum allowed loss.

ℛ max\mathcal{R}_{\text{max}} represents the best-case rate of return (RoR) achievable over the next three candlesticks under the current LLM-issued trading decision (either LONG or SHORT). It assumes the optimal exit occurs at the most favorable intra-candle price point, i.e., the maximum high for a long position or the minimum low for a short position. Conversely, ℛ min\mathcal{R}_{\text{min}} captures the most adverse price movement during the same interval. These two metrics represent a bounded range of maximum profit or loss outcomes under realistic, risk-managed execution (Lo, [2001](https://arxiv.org/html/2509.09995v3#bib.bib34)).

![Image 4: Refer to caption](https://arxiv.org/html/2509.09995v3/x4.png)

Figure 4: Case sample of the PatternAgent on CL (2024). The agent extracts swing pivots, fits a declining resistance line through lower highs, and identifies flat support near 78. As the gap narrows, it classifies the formation as a descending triangle and generates three structured summaries: Structure (“lower highs” vs. “flat support”), Trend (bearish breakdown bias), and Symmetry (triangular convergence). Dashed edges and EMA overlays are visual aids only; the classification is derived solely from bar geometry.

Table 1: Performance comparison across trading symbols. Results are shown for random(Baseline), Linear Regression(LR), XGBoost, and our QuantAgent. Bold values indicate the best performance for each metric across methods. Upward arrows (↑) denote metrics where higher values are better. 

5 Results
---------

### 5.1 Main Results

In Table [1](https://arxiv.org/html/2509.09995v3#S4.T1 "Table 1 ‣ 4 Experiments ‣ QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading"), we compare our agent-based LLM trader to the three baselines, Random Baseline, Linear Regression(LR) and XGBoost, across eight widely traded markets.

From the table, we draw several key observations: (i) Our accuracy outperforms all methods across the eight evaluated markets especially on NQ, where we achieve a 26.5% increase over the random baseline and a clear margin over both LR and XGBoost. (ii) Despite the presence of risk caps, our ℛ cc\mathcal{R}_{\text{cc}} still achieves the best performance in 7 out of 8 assets, suggesting that our model can consistently capture profitable short-term trends under realistic trading constraints. (iii) We obtain the highest ℛ max\mathcal{R}_{\text{max}} in 6 out of the 8 markets and are nearly tied with the best performer in the remaining two, indicating our system effectively captures potential upside while respecting risk bounds. (iv) Similarly, our ℛ min\mathcal{R}_{\text{min}} shows strong robustness, ranking among the least negative values across most assets. This implies that our method not only captures gains but also limits downside risk effectively.

Overall, the results highlight that our approach generalizes well across diverse asset classes in 4-hour time frame, balancing accuracy and return while maintaining robust risk control.

![Image 5: Refer to caption](https://arxiv.org/html/2509.09995v3/x5.png)

Figure 5: 1-hour performance comparison across eight assets. Results are shown for random (Baseline), Linear Regression, XGBoost, and our QuantAgent. Arrows indicate higher values are better.

Furthermore, Figure [5](https://arxiv.org/html/2509.09995v3#S5.F5 "Figure 5 ‣ 5.1 Main Results ‣ 5 Results ‣ QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading") shows a comparative performance trend with same metrics for each asset in the 1-hour time frame. Our method (QuantAgent) consistently outperforms all baselines across most metrics and markets, especially in SPX, QQQ, and BTC where our method shows the most pronounced performance gap. The red line (QuantAgent) dominates across most plots, indicating both higher directional accuracy and better risk-aware returns, though it shows less satisfactory ℛ min\mathcal{R}_{\text{min}} in some assets. Baseline, Linear Regression, and XGBoost exhibit weaker and less stable patterns, often lagging across most metrics. This visualization highlights the robustness and generalization capability of our approach under both profit-seeking and risk-constrained conditions in 1-hour time horizon.

![Image 6: Refer to caption](https://arxiv.org/html/2509.09995v3/x6.png)

Figure 6: Case study of high-frequency prediction on SPX (2025). Correct forecasts (8/10) are marked with green Buy or red Sell badges, while mispredictions are shown in grey (2/10).

### 5.2 Case Study on Continuous Short-Term Prediction

To evaluate short-horizon prediction consistency, the LLM’s directional accuracy was further tested on a randomly selected 100-bar SPX segment using 10 overlapping windows, each offset by 5 bars (Qin et al., [2017](https://arxiv.org/html/2509.09995v3#bib.bib35)). Predictions were verified against actual price trends, achieving an overall accuracy of 80%, as shown in Figure [6](https://arxiv.org/html/2509.09995v3#S5.F6 "Figure 6 ‣ 5.1 Main Results ‣ 5 Results ‣ QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading"). The system correctly issued sell signals at indices 0 and 3 during resistance stalls and negative momentum shifts, and buy signals at indices 1, 4, 5, 6, 8, and 9 by detecting early momentum flips, support bounces, and recognizable recovery patterns. Errors at indices 2 and 7 were due to overreliance on incomplete patterns and premature bullish calls, highlighting the model’s tendency to prioritize emerging signals. Adjusting these weightings could enhance reliability.

### 5.3 Case Studies of Agent Reasoning

We present a representative Descending Triangle case study to illustrate PatternAgent’s reasoning. Figure [4](https://arxiv.org/html/2509.09995v3#S4.F4 "Figure 4 ‣ 4 Experiments ‣ QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading") illustrates how PatternAgent reasons over bar geometry. In this case, it recognizes a Descending Triangle, producing structured outputs that capture lower highs over flat support, a bearish breakdown bias, and triangular convergence. This example highlights the agent’s ability to decompose raw price action into interpretable features. See Appendix [F](https://arxiv.org/html/2509.09995v3#A6 "Appendix F Case Studies ‣ QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading") for additional case studies.

6 Conclusion
------------

QuantAgent illustrates how decomposing trading into specialized LLM agents grounded in price data enables accurate, transparent, and risk-aware decisions for high-frequency trading. The multi-agent structure not only enhances interpretability but also promotes robustness through cross-agent validation and specialization. Our results across diverse markets underscore the viability of this approach, suggesting that structured agent collaboration grounded purely in price data can serve as a scalable foundation for future real-time, data-efficient financial systems operating without external sentiment or supervision.

7 Limitations and Future Work
-----------------------------

QuantAgent’s key constraints center on speed and micro‑horizon accuracy. First, its predictive precision drops on ultra‑short candles (≈1\approx 1–15 min). The price series at this scale are dominated by noise and rapid regime shifts, making it difficult for the current zero‑shot LLM ensemble to separate transient spikes from tradable signals; empirical tests show a sizable degradation in prediction accuracy relative to 30min–4h bars.

Second, the architecture is not truly real‑time. Each inference cycle involves an LLM call plus several bound indicator/pattern tools, introducing latencies that can exceed the window in which a 1‑minute opportunity remains exploitable. Streamlining tool orchestration, caching intermediate features, or moving critical logic to lighter‑weight models on the edge are promising directions to close this gap (Hasbrouck and Saar, [2013](https://arxiv.org/html/2509.09995v3#bib.bib36)).

References
----------

*   Pring (1991) Martin J Pring. Technical analysis explained: The successful investor’s guide to spotting investment trends and turning points. _(No Title)_, 1991. 
*   Murphy (1999) John J Murphy. _Technical analysis of the financial markets: A comprehensive guide to trading methods and applications_. Penguin, 1999. 
*   Moskowitz et al. (2012) Tobias J Moskowitz, Yao Hua Ooi, and Lasse Heje Pedersen. Time series momentum. _Journal of financial economics_, 104(2):228–250, 2012. 
*   Fama (1970) Eugene F. Fama. Efficient capital markets: A review of theory and empirical work. _The Journal of Finance_, 25(2):383–417, 1970. [10.2307/2325486](https://arxiv.org/doi.org/10.2307/2325486). URL [https://doi.org/10.2307/2325486](https://doi.org/10.2307/2325486). 
*   OpenAI et al. (2024) OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, and et al. Gpt-4 technical report, 2024. URL [https://arxiv.org/abs/2303.08774](https://arxiv.org/abs/2303.08774). 
*   Yang et al. (2023) Hongyang Yang, Xiao-Yang Liu, and Christina Dan Wang. Fingpt: Open-source financial large language models, 2023. URL [https://arxiv.org/abs/2306.06031](https://arxiv.org/abs/2306.06031). 
*   Nguyen et al. (2015) Thien Hai Nguyen, Kiyoaki Shirai, and Julien Velcin. Sentiment analysis on social media for stock movement prediction. _Expert Systems with Applications_, 42(24):9603–9611, 2015. ISSN 0957-4174. [https://doi.org/10.1016/j.eswa.2015.07.052](https://arxiv.org/doi.org/https://doi.org/10.1016/j.eswa.2015.07.052). URL [https://www.sciencedirect.com/science/article/pii/S0957417415005126](https://www.sciencedirect.com/science/article/pii/S0957417415005126). 
*   Xiao et al. (2025) Yijia Xiao, Edward Sun, Di Luo, and Wei Wang. Tradingagents: Multi-agents llm financial trading framework, 2025. URL [https://arxiv.org/abs/2412.20138](https://arxiv.org/abs/2412.20138). 
*   Zakir et al. (2025) Umair Zakir, Evan Daykin, Amssatou Diagne, and Jacob Faile. Advanced deep learning techniques for analyzing earnings call transcripts: Methodologies and applications, 2025. URL [https://arxiv.org/abs/2503.01886](https://arxiv.org/abs/2503.01886). 
*   Zhang et al. (2023a) Xiang Zhang, Senyu Li, Bradley Hauer, Ning Shi, and Grzegorz Kondrak. Don’t trust chatgpt when your question is not in english: a study of multilingual abilities and types of llms. _arXiv preprint arXiv:2305.16339_, 2023a. 
*   Chordia et al. (2013) Tarun Chordia, Amit Goyal, Bruce N. Lehmann, and Gideon Saar. High-frequency trading. _Journal of Financial Markets_, 16(4):637–645, 2013. ISSN 1386-4181. [https://doi.org/10.1016/j.finmar.2013.06.004](https://arxiv.org/doi.org/https://doi.org/10.1016/j.finmar.2013.06.004). URL [https://www.sciencedirect.com/science/article/pii/S1386418113000268](https://www.sciencedirect.com/science/article/pii/S1386418113000268). High-Frequency Trading. 
*   Liu et al. (2022a) Xiao-Yang Liu, Ziyi Xia, Jingyang Rui, Jiechao Gao, Hongyang Yang, Ming Zhu, Christina Dan Wang, Zhaoran Wang, and Jian Guo. Finrl-meta: Market environments and benchmarks for data-driven financial reinforcement learning, 2022a. URL [https://arxiv.org/abs/2211.03107](https://arxiv.org/abs/2211.03107). 
*   Nison (2001) Steve Nison. _Japanese Candlestick Charting Techniques_. Prentice Hall Press, New York, 2nd edition, October 2001. 
*   Schick et al. (2023) Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools, 2023. URL [https://arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761). 
*   Yu et al. (2024) Yangyang Yu, Zhiyuan Yao, Haohang Li, Zhiyang Deng, Yuechen Jiang, Yupeng Cao, Zhi Chen, Jordan W. Suchow, Zhenyu Cui, Rong Liu, Zhaozhuo Xu, Denghui Zhang, Koduvayur Subbalakshmi, Guojun Xiong, Yueru He, Jimin Huang, Dong Li, and Qianqian Xie. Fincon: A synthesized llm multi-agent system with conceptual verbal reinforcement for enhanced financial decision making. In _Proceedings of the 38th Conference on Neural Information Processing Systems (NeurIPS)_, 2024. To appear. 
*   Tumarkin and Whitelaw (2001) Robert Tumarkin and Robert F. Whitelaw. News or noise? internet postings and stock prices. _Financial Analysts Journal_, 57(3):41–51, 2001. URL [http://www.jstor.org/stable/4480315](http://www.jstor.org/stable/4480315). 
*   Li et al. (2025) Yuante Li, Xu Yang, Xiao Yang, Minrui Xu, Xisen Wang, Weiqing Liu, and Jiang Bian. R&d-agent-quant: A multi-agent framework for data-centric factors and model joint optimization, 2025. 
*   Lo et al. (2000) Andrew W. Lo, Harry Mamaysky, and Jiang Wang. Foundations of technical analysis: Computational algorithms, statistical inference, and empirical implementation. NBER Working Paper 7613, National Bureau of Economic Research, March 2000. URL [https://ssrn.com/abstract=228099](https://ssrn.com/abstract=228099). Available at SSRN. 
*   Chen and Chen (2016) Tai-liang Chen and Feng-yu Chen. An intelligent pattern recognition model for supporting investment decisions in stock market. _Information Sciences_, 346–347:261–274, 2016. [10.1016/j.ins.2016.01.060](https://arxiv.org/doi.org/10.1016/j.ins.2016.01.060). 
*   Jegadeesh and Titman (1993) Narasimhan Jegadeesh and Sheridan Titman. Returns to buying winners and selling losers: Implications for stock market efficiency. _The Journal of Finance_, 48(1):65–91, 1993. ISSN 00221082, 15406261. URL [http://www.jstor.org/stable/2328882](http://www.jstor.org/stable/2328882). 
*   Prechter (2005) Robert R. Prechter. _Elliott Wave Principle: Key to Market Behavior_. New Classics Library, Gainesville, GA, February 2005. 
*   LangChain (2025) Inc. LangChain. Langgraph: A low‑level orchestration framework for building, managing, and deploying long‑running, stateful agents. [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/), 2025. Accessed: 2025‑07-01. 
*   Wilder (1978) Welles Wilder. _New Concepts in Technical Trading Systems_. Trend Research, 1978. 
*   Appel (2005) Gerald Appel. _Technical Analysis: Power Tools for Active Investors_. FT Press, reprint edition, 2005. ISBN 9780132930048. 
*   Investopedia (n.d.) Investopedia. Stochastic oscillator: What it is, how it works, how to …. [https://www.investopedia.com/terms/s/stochasticoscillator.asp](https://www.investopedia.com/terms/s/stochasticoscillator.asp), n.d. Accessed: 2025‑08‑14. 
*   Williams (2011) Larry Williams. _The Long-Term Secrets to Short-Term Trading_. Wiley, 2nd edition, 2011. ISBN 9780470915738. 
*   Wang et al. (2023) Ziao Wang, Yuhang Li, Junda Wu, Jaehyeon Soon, and Xiaofeng Zhang. Finvis-gpt: A multimodal large language model for financial chart analysis, 2023. URL [https://arxiv.org/abs/2308.01430](https://arxiv.org/abs/2308.01430). 
*   Elder (2002) Alexander Elder. _Come Into My Trading Room: A Complete Guide to Trading_. John Wiley & Sons, 2002. ISBN 9780471225348. 
*   Kirkpatrick and Dahlquist (2015) Charles Kirkpatrick and Julie Dahlquist. _Technical Analysis: The Complete Resource for Financial Market Technicians_. FT Press, 3rd edition, 2015. ISBN 978-0134137049. 
*   Kissell (2013) Robert Kissell. _The Science of Algorithmic Trading and Portfolio Management_. Academic Press, San Diego, 1st edition, 2013. ISBN 978-0124016897. 
*   TA‑Lib Development Team (2025) TA‑Lib Development Team. Ta‑lib: Technical analysis library (core c library). [https://github.com/TA‑Lib/ta‑lib](https://github.com/TA%E2%80%91Lib/ta%E2%80%91lib), 2025. 
*   Pesaran and Timmermann (2004) M.Hashem Pesaran and Allan Timmermann. How costly is it to ignore breaks when forecasting the direction of a time series? _International Journal of Forecasting_, 20(3):411–425, 2004. ISSN 0169-2070. [https://doi.org/10.1016/S0169-2070(03)00068-2](https://arxiv.org/doi.org/https://doi.org/10.1016/S0169-2070(03)00068-2). URL [https://www.sciencedirect.com/science/article/pii/S0169207003000682](https://www.sciencedirect.com/science/article/pii/S0169207003000682). 
*   Fama and MacBeth (1973) Eugene F. Fama and James D. MacBeth. Risk, return, and equilibrium: Empirical tests. _Journal of Political Economy_, 81(3):607–636, 1973. ISSN 00223808, 1537534X. URL [http://www.jstor.org/stable/1831028](http://www.jstor.org/stable/1831028). 
*   Lo (2001) Andrew W. Lo. Risk management for hedge funds: Introduction and overview. _Financial Analysts Journal_, 57(6):16–33, 2001. ISSN 0015198X. URL [http://www.jstor.org/stable/4480353](http://www.jstor.org/stable/4480353). 
*   Qin et al. (2017) Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, and Garrison Cottrell. A dual-stage attention-based recurrent neural network for time series prediction, 2017. URL [https://arxiv.org/abs/1704.02971](https://arxiv.org/abs/1704.02971). 
*   Hasbrouck and Saar (2013) Joel Hasbrouck and Gideon Saar. Low-latency trading. _Journal of Financial Markets_, 16(4):646–679, 2013. [10.1016/j.finmar.2013.05.003](https://arxiv.org/doi.org/10.1016/j.finmar.2013.05.003). 
*   Kahneman and Tversky (1979) Daniel Kahneman and Amos Tversky. Prospect theory: An analysis of decision under risk. _Econometrica_, 47(2):263–291, 1979. ISSN 00129682, 14680262. URL [http://www.jstor.org/stable/1914185](http://www.jstor.org/stable/1914185). 
*   Edwards et al. (2018) Robert D. Edwards, John Magee, and W. H. C. Bassetti. _Technical Analysis of Stock Trends_. CRC Press, 11th edition, 2018. ISBN 9781138069411. 
*   Bommasani et al. (2021) Rishi Bommasani, Drew A. Hudson, Ehsan Adeli, Russ B. Altman, Simran Arora, Sydney von Arx, Michael S. Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, Erik Brynjolfsson, Shyamal Buch, Dallas Card, Rodrigo Castellon, Niladri S. Chatterji, Annie S. Chen, Kathleen Creel, Jared Quincy Davis, Dorottya Demszky, Chris Donahue, Moussa Doumbouya, Esin Durmus, Stefano Ermon, John Etchemendy, Kawin Ethayarajh, Li Fei-Fei, Chelsea Finn, Trevor Gale, Lauren E. Gillespie, Karan Goel, Noah D. Goodman, Shelby Grossman, Neel Guha, Tatsunori Hashimoto, Peter Henderson, John Hewitt, Daniel E. Ho, Jenny Hong, Kyle Hsu, Jing Huang, Thomas Icard, Saahil Jain, Dan Jurafsky, Pratyusha Kalluri, Siddharth Karamcheti, Geoff Keeling, Fereshte Khani, Omar Khattab, Pang Wei Koh, Mark S. Krass, Ranjay Krishna, Rohith Kuditipudi, and et al. On the opportunities and risks of foundation models. _CoRR_, abs/2108.07258, 2021. URL [https://arxiv.org/abs/2108.07258](https://arxiv.org/abs/2108.07258). 
*   Chakravarty et al. (1998) Sugato Chakravarty, Asani Sarkar, and Lifan Wu. Information asymmetry, market segmentation and the pricing of cross-listed shares: theory and evidence from chinese a and b shares. _Journal of International Financial Markets, Institutions and Money_, 8(3):325–356, 1998. ISSN 1042-4431. [https://doi.org/10.1016/S1042-4431(98)00041-9](https://arxiv.org/doi.org/https://doi.org/10.1016/S1042-4431(98)00041-9). URL [https://www.sciencedirect.com/science/article/pii/S1042443198000419](https://www.sciencedirect.com/science/article/pii/S1042443198000419). 
*   Cao et al. (2025) Juntai Cao, Xiang Zhang, Raymond Li, Chuyuan Li, Chenyu You, Shafiq Joty, and Giuseppe Carenini. Multi2: Multi-agent test-time scalable framework for multi-document processing. _arXiv preprint arXiv:2502.20592_, 2025. 
*   Brock et al. (1992) William Brock, Josef Lakonishok, and Blake LeBaron. Simple technical trading rules and the stochastic properties of stock returns. _The Journal of Finance_, 47(5):1731–1764, 1992. [https://doi.org/10.1111/j.1540-6261.1992.tb04681.x](https://arxiv.org/doi.org/https://doi.org/10.1111/j.1540-6261.1992.tb04681.x). URL [https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1992.tb04681.x](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1992.tb04681.x). 
*   Wang et al. (2024) Chaojie Wang, Yanchen Deng, Zhiyi Lyu, Liang Zeng, Jujie He, Shuicheng Yan, and Bo An. Q*: Improving multi-step reasoning for llms with deliberative planning. _arXiv preprint arXiv:2406.14283_, 2024. 
*   Zhang et al. (2024a) Xiang Zhang, Muhammad Abdul-Mageed, and Laks VS Lakshmanan. Autoregressive+ chain of thought= recurrent: Recurrence’s role in language models’ computability and a revisit of recurrent transformer. _arXiv preprint arXiv:2409.09239_, 2024a. 
*   Liu et al. (2022b) Puyuan Liu, Xiang Zhang, and Lili Mou. A character-level length-control algorithm for non-autoregressive sentence summarization. _Advances in Neural Information Processing Systems_, 35:29101–29112, 2022b. 
*   Zhang et al. (2024b) Xiang Zhang, Senyu Li, Ning Shi, Bradley Hauer, Zijun Wu, Grzegorz Kondrak, Muhammad Abdul-Mageed, and Laks VS Lakshmanan. Cross-modal consistency in multimodal large language models. _arXiv preprint arXiv:2411.09273_, 2024b. 
*   Lu et al. (2022) Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. _Advances in Neural Information Processing Systems_, 35:2507–2521, 2022. 
*   Zhang et al. (2023b) Xiang Zhang, Ning Shi, Bradley Hauer, and Grzegorz Kondrak. Bridging the gap between babelnet and hownet: Unsupervised sense alignment and sememe prediction. In _Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics_, pages 2789–2798, 2023b. 
*   Gong et al. (2025) Mingyang Gong, Jing Fan, Guohui Lin, Bing Su, Zihan Su, and Xiang Zhang. Multiprocessor scheduling with testing: improved online algorithms and numerical experiments. _Journal of Scheduling_, pages 1–15, 2025. 
*   Wei et al. (2025a) Jiaqi Wei, Hao Zhou, Xiang Zhang, Di Zhang, Zijie Qiu, Wei Wei, Jinzhe Li, Wanli Ouyang, and Siqi Sun. Alignrag: Leveraging critique learning for evidence-sensitive retrieval-augmented reasoning. _arXiv preprint arXiv:2504.14858_, 2025a. 
*   Wei et al. (2025b) Jiaqi Wei, Yuejin Yang, Xiang Zhang, Yuhan Chen, Xiang Zhuang, Zhangyang Gao, Dongzhan Zhou, Guangshuai Wang, Zhiqiang Gao, Juntai Cao, et al. From ai for science to agentic science: A survey on autonomous scientific discovery. _arXiv preprint arXiv:2508.14111_, 2025b. 
*   Zhang et al. (2025) Xiang Zhang, Juntai Cao, Jiaqi Wei, Chenyu You, and Dujian Ding. Why prompt design matters and works: A complexity analysis of prompt search space in llms. _arXiv preprint arXiv:2503.10084_, 2025. 
*   Achelis (2013) Steven B. Achelis. _Technical Analysis from A to Z_. McGraw-Hill, 2nd edition, 2013. ISBN 9780071826297. Publication date: December 13, 2013. 

Appendix A Why Technical Analysis Alone Can Suffice for Trading
---------------------------------------------------------------

QuantAgent is a multi-modal, multi-agent high-frequency trading LLM framework that provides market prediction based solely on price data, disregarding other information such as news, social media, etc. This strategy is referred to as technical analysis (Pring, [1991](https://arxiv.org/html/2509.09995v3#bib.bib1)). Technical analysis is based on the premise that price alone is enough for capturing market movement and predicting future trends, and has been extensively studied by previous research (Murphy, [1999](https://arxiv.org/html/2509.09995v3#bib.bib2)). In this section, we present in detail why technical analysis alone can suffice for trading.

The first principle of technical analysis is that all relevant information, whether economic, political, psychological, or otherwise, is already reflected in market prices (Fama, [1970](https://arxiv.org/html/2509.09995v3#bib.bib4)). In other words, prices adjust quickly to new developments because people act on the information they receive by buying or selling (Kahneman and Tversky, [1979](https://arxiv.org/html/2509.09995v3#bib.bib37)). These actions are recorded in price changes (Lo et al., [2000](https://arxiv.org/html/2509.09995v3#bib.bib18)). Therefore, by observing how prices move, it is possible to indirectly understand how the market as a whole is reacting to both public and private information, without needing to process that information explicitly (Edwards et al., [2018](https://arxiv.org/html/2509.09995v3#bib.bib38)). Our system therefore also follows this principle and has each of its agents perform analysis solely based on price data.

Technical analysis assumes that price movements are not entirely random (Lo et al., [2000](https://arxiv.org/html/2509.09995v3#bib.bib18)). Instead, they tend to follow patterns over time. When prices begin to rise, they often continue to rise for some period, and similarly, downward trends can persist before reversing. These trends often reflect collective human behavior, such as fear during declines or optimism during rallies. By identifying such trends early, technical traders aim to make decisions that align with the general direction of the market (Jegadeesh and Titman, [1993](https://arxiv.org/html/2509.09995v3#bib.bib20)). QuantAgent operates such that it captures market patterns and leverages this price movement assumption.

Notably, many existing technical analysis methods are based on the observation that certain price patterns tend to appear repeatedly. This repetition is attributed to stable behavioral tendencies in market participants. For example, traders often react similarly to price increases or decreases, leading to recurring patterns such as peaks, dips, and reversals. Recognizing and responding to these familiar structures allows technical systems to make predictions without needing to understand the specific causes of each movement (Edwards et al., [2018](https://arxiv.org/html/2509.09995v3#bib.bib38)). Such observable repetition is a natural fit for an agentic framework, as LLM agents have shown strong capability in reasoning over patterns and trends, achieving human-like capabilities (Bommasani et al., [2021](https://arxiv.org/html/2509.09995v3#bib.bib39)).

Occasionally, price changes occur well before any official information is made available to the public (Chakravarty et al., [1998](https://arxiv.org/html/2509.09995v3#bib.bib40)). For example, a stock’s price might begin rising days or even weeks before a company announces strong earnings. This can happen because certain investors—such as employees, suppliers, or professional analysts—may already have insights into the company’s performance, such as increased sales activity or unusually high production volumes. Similarly, prices may fall before news of a scandal becomes public. If there are rumors of legal investigations or unusual management behavior, informed traders might start selling early, causing the price to decline in advance. In both cases, the price moves ahead of the news because the market collectively reacts to early signals, expectations, or private information (Chakravarty et al., [1998](https://arxiv.org/html/2509.09995v3#bib.bib40); Cao et al., [2025](https://arxiv.org/html/2509.09995v3#bib.bib41)). Technical analysis captures these movements directly through price behavior, without requiring access to the underlying cause (Brock et al., [1992](https://arxiv.org/html/2509.09995v3#bib.bib42)). This allows trading systems to respond to changes as soon as they appear in the market, rather than waiting for delayed or incomplete public disclosures (Lo et al., [2000](https://arxiv.org/html/2509.09995v3#bib.bib18)).

In summary, our agent works under the principles of technical analysis, which offers a practical and self-contained approach to understanding market behavior. By assuming that all available information is already incorporated into price data, and that human reactions to price movements tend to follow consistent patterns, it becomes possible for our agent to forecast future price directions without relying on external inputs (Murphy, [1999](https://arxiv.org/html/2509.09995v3#bib.bib2)).

Appendix B Why LLMs Are Well-Suited for Price-Based Technical Analysis
----------------------------------------------------------------------

This appendix explains why large language models (LLMs) are not only capable of conducting technical analysis from price data but are _especially_ well matched to it. The core claim is methodological: technical analysis is a structured, short-horizon reasoning problem over standardized inputs (OHLC bars, indicators, and chart geometry), and modern LLM capabilities—multi-step reasoning (Wang et al., [2024](https://arxiv.org/html/2509.09995v3#bib.bib43); Zhang et al., [2024a](https://arxiv.org/html/2509.09995v3#bib.bib44); Liu et al., [2022b](https://arxiv.org/html/2509.09995v3#bib.bib45)), multimodal perception (Zhang et al., [2024b](https://arxiv.org/html/2509.09995v3#bib.bib46); Lu et al., [2022](https://arxiv.org/html/2509.09995v3#bib.bib47); Zhang et al., [2023b](https://arxiv.org/html/2509.09995v3#bib.bib48)), tool use Gong et al. ([2025](https://arxiv.org/html/2509.09995v3#bib.bib49)), retrieval Wei et al. ([2025a](https://arxiv.org/html/2509.09995v3#bib.bib50)), and agentic Wei et al. ([2025b](https://arxiv.org/html/2509.09995v3#bib.bib51)) coordination—map directly onto these requirements.

Technical analysis converts recent OHLC sequences into a compact set of signals—momentum oscillators, moving-average relations, rate-of-change, and shape/level interactions. These signals compose into rules of the form “_if MACD crosses down, RSI exits overbought, and price rejects resistance, then short with risk r r_.” LLMs excel at synthesizing such heterogeneous but _symbolic_ cues into consistent, short-horizon judgments, producing language-native rationales and machine-checkable action schemas. This yields decisions that are both human-auditable and executable (see our DecisionAgent design in the main text).

A large share of technical analysis is visual: trend channels, swing pivots, triangles, flags, double bottoms, and wedge compressions are geometric concepts. Modern multimodal LLMs can parse candlestick charts, identify pivots and boundaries, and align the detected structure with canonical pattern libraries, enabling the system to reason about _structure + context_ rather than isolated indicators. Our PatternAgent and TrendAgent instantiate exactly this: tool-generated charts are analyzed for support, resistance, slope, and convergence before any prediction is issued.

Purely textual reasoning can drift; technical analysis benefits from _grounding_ via tools. By binding indicator calculators (RSI, MACD, ROC, Stoch, Williams %R), trendline estimators, and execution simulators to the LLM, we constrain outputs to numerically verifiable quantities, reduce hallucination risk, and accelerate decisions—key in latency-sensitive settings. Tools also standardize feature extraction, so the model reasons over stable, low-dimensional summaries instead of raw, noisy prices.

Technical analysis is naturally modular. Splitting responsibilities across specialized agents—Indicator (numerical momentum/oscillators), Pattern (geometric formations), Trend (direction and slope), and Risk (position sizing and boundaries)—yields complementary views that can be _cross-validated_. An agentic LLM stack (e.g., via LangGraph) supports: (i) division of labor for lower latency, (ii) explicit debate/consensus protocols to down-weight conflicting signals, and (iii) clean hand-offs to an execution layer that emits structured orders with stops and take-profits. Our QuantAgent architecture operationalizes this workflow and demonstrates consistent gains over rule-based and ML baselines on 1h/4h horizons.

In summary, Price-based technical analysis poses a _structured, tool-grounded, multimodal, and modular_ reasoning task. These properties align tightly with modern LLM strengths in stepwise reasoning, chart perception, retrieval, tool invocation, and agentic coordination—yielding decisions that are fast, interpretable, and risk-aware. Consequently, LLMs are natural engines for technical analysis on price data, especially on 30min–4h horizons where structure dominates noise.

Appendix C Prompt Template
--------------------------

### C.1 Indicator Agent

To operationalize the role of IndicatorAgent, we design a task-specific prompt Zhang et al. ([2025](https://arxiv.org/html/2509.09995v3#bib.bib52)) that guides the agent to extract and interpret technical indicators under strict latency constraints.

Prompt for IndicatorAgent in our multi-agent LLM framework. The agent receives recent OHLC data as input and interprets it through tool-assisted analysis. Variables such as kline_data and time_frame are dynamically instantiated, enabling the agent to extract meaningful price movements and adapt its outputs across diverse market conditions.

### C.2 Pattern Agent

To instantiate the PatternAgent, we construct a prompt that directs the agent to identify geometric formations (e.g., peaks, troughs, consolidations) from OHLC sequences, leveraging the LLM’s multi-modal reasoning capacity for candlestick and chart-pattern analysis.

Prompt for PatternAgent in our multi-agent LLM framework. The agent receives OHLC data as input, transforms it into a visual representation, and analyzes the sequence from a pattern-recognition perspective.

Graph-analysis prompt for PatternAgent in our multi-agent LLM framework. The agent is provided with a tool-generated chart and a textual library of canonical structural patterns (e.g., “U” shapes, “W” shapes, triangles). It automatically evaluates whether the chart matches any of these patterns and explains its reasoning along three dimensions: structure, direction, and symmetry.

### C.3 Trend Agent

For the TrendAgent, we provide a prompt that emphasizes detection of directional momentum across multiple horizons, enabling the agent to reason about medium- to long-term trends while remaining responsive to short-horizon signals.

Prompt for TrendAgent in our multi-agent LLM framework. The agent converts time-series OHLC data into a tool-generated chart and performs trend analysis on the visualization to identify directional momentum and potential breakouts.

Graph-based prompt for TrendAgent in our multi-agent LLM framework. The agent is provided with a tool-generated chart containing two reference lines: a blue support line and a red resistance line. It analyzes how price action interacts with these lines and produces a short-term trend prediction (upward, downward, or sideways), accompanied by structured outputs covering prediction, reasoning, and signals.

### C.4 Decision Agent

To implement the DecisionAgent, we design a prompt that compels the DecisionAgent to integrate signals from all specialized agents into coherent trading actions, balancing profitability, risk, and interpretability in high-frequency market settings.

Prompt for DecisionAgent in our multi-agent LLM framework. The agent integrates three upstream perspectives, indicator signals, structural patterns, and trend interactions, and outputs a short-term directional decision (LONG or SHORT). The prompt instructs the agent to prioritize consistent evidence, avoid speculative outputs, and provide structured justification, including an estimated risk–reward ratio.

### C.5 Pattern Tool Sample Output

![Image 7: Refer to caption](https://arxiv.org/html/2509.09995v3/x7.png)

Figure 7: Tool-generated chart for PatternAgent on NQ (2025). Raw intraday candlesticks from the July 7–8 window are shown prior to overlaying reference lines. The sequence of lower highs and higher lows indicates a contracting trading range, suggesting latent pressure that often precedes a breakout once a boundary is breached.

### C.6 Trend Tool Sample Output

![Image 8: Refer to caption](https://arxiv.org/html/2509.09995v3/x8.png)

Figure 8: Tool-generated chart for TrendAgent on NQ (2025). Intraday candlesticks compress between an upward-sloping support line (blue) and a downward-sloping resistance line (red), forming a symmetrical-triangle wedge. The converging boundaries indicate a consolidation phase where buying pressure gradually builds while sellers cap rallies, often preceding a decisive breakout once a boundary is breached.

Appendix D Web Demo
-------------------

![Image 9: Refer to caption](https://arxiv.org/html/2509.09995v3/x9.png)

Figure 9: User interface of QuantAgent. The configuration panel enables selection of trading asset (e.g., AAPL), timeframe, and analysis window. It supports live data input, fine-grained control over historical candlestick ranges, and secure local execution. By default, the system optimizes context using the most recent 40–50 price bars to balance relevance and computational efficiency.

![Image 10: Refer to caption](https://arxiv.org/html/2509.09995v3/x10.png)

Figure 10: Trading decision interface of QuantAgent. The system produces a final directional decision along with the forecast horizon, risk–reward ratio, and a textual justification grounded in pattern recognition (e.g., Rounded Bottom reversal).

![Image 11: Refer to caption](https://arxiv.org/html/2509.09995v3/x11.png)

Figure 11: IndicatorAgent interface of QuantAgent. A structured IndicatorAgent report is displayed, summarizing key technical indicators, MACD, RoC, Stochastic Oscillator, and Williams %R, to support interpretability and validate the decision process.

![Image 12: Refer to caption](https://arxiv.org/html/2509.09995v3/x12.png)

Figure 12: Pattern and trend report generated by QuantAgent. The top panel presents a detected Double Bottom pattern, supported by structural symmetry, a preceding downtrend, and subsequent rebound. The accompanying chart overlay highlights the pattern geometry. The bottom panel provides trend diagnostics, including ADX, RSI divergence, and volume analysis, alongside a visualization of support and resistance levels. Together, the pattern and trend modules offer complementary perspectives on potential trend reversal and market recovery.

Appendix E Benchmark Detail
---------------------------

### E.1 Benchmark Construction

To evaluate the proposed QuantAgent framework, we design a benchmark composed of diverse, well-known financial assets. This benchmark allows us to systematically test the system’s ability to generalize across asset classes and trading environments. The benchmark also facilitates controlled comparisons across different decision-making models and enables reproducibility.

### E.2 Data Collection and Asset Selection

All historical price data used in the benchmark are obtained via the publicly available APIs, specifically YFinance and TradingView’s historical market data services. We use 1-hour and 4-hour OHLC (Open, High, Low, Close) candlestick data for all assets to maintain consistency in time resolution. The benchmark covers a diverse mix of asset types, including cryptocurrency (BTC/USD), crude oil (CL), equity index futures (ES and NQ), and exchange-traded or spot indices such as QQQ, SPX, DJI, and VIX. Each asset is widely traded and highly liquid, helping avoid noise from low trading activity and making sure the price movements reflect real market behavior. Besides, the selected assets include both relatively stable, trend-following instruments such as SPX and ES, which often exhibit smoother directional movement, and more volatile assets such as BTC/USD, which are known for rapid swings and high short-term variability.

For each asset, we collect 5,000 historical 1-hour and 4-hour bars. To ensure fairness and consistency across assets, we apply the same fixed bar count to all instruments—including those with limited trading hours, such as QQQ. As a result, assets with lower intraday trading frequency span a longer historical period (up to ten years), reflecting the reduced density of available candlestick intervals. From this data, we randomly sample 100 evaluation segments per asset. Each segment consists of 100 consecutive candlesticks, with the final 3 candlesticks removed during inference to ensure the system does not observe the true market outcome during prediction. The final three candlesticks are reserved for validating the correctness of the LLM’s prediction.

### E.3 Benchmark Asset Properties

Table 2: Overview of 4-hour benchmark asset properties. Each asset is characterized by its name, market type, the start and end dates of the data collection window, and the total number of bars sampled.

Table 3: Overview of 1-hour benchmark asset properties. Each asset is characterized by its name, market type, the start and end dates of the data collection window, and the total number of bars sampled. 

### E.4 Benchmark Assets Detail

We evaluate our models on a diverse set of benchmark assets drawn from major areas of the global financial markets.

BTC/USD (Bitcoin) shows how much one Bitcoin is worth in U.S. dollars. It represents the broader cryptocurrency market and operates continuously with high trading volume.

CL (Crude Oil) tracks the price of West Texas Intermediate crude oil, a key benchmark for U.S. energy prices and a global indicator influenced by supply, demand, and geopolitical factors.

ES (E-mini S&P 500) is a futures contract tied to the S&P 500 Index, which includes 500 large publicly traded U.S. companies. It gives a broad picture of the U.S. stock market’s performance.

NQ (E-mini Nasdaq-100) follows the Nasdaq-100 Index, which focuses on large non-financial companies listed on the Nasdaq exchange, especially in the technology and innovation sectors.

QQQ is an exchange-traded fund (ETF) that aims to match the performance of the Nasdaq-100 Index. It offers a simple way for investors to gain exposure to major U.S. tech and growth stocks.

SPX (S&P 500 Index) directly tracks the S&P 500 Index and is widely used as a benchmark for measuring the overall performance of U.S. equities.

DJI (Dow Jones Industrial Average) includes 30 large and well-known U.S. companies across different industries. It is commonly used as an indicator of the broader U.S. economy.

VIX (Volatility Index) reflects the market’s expectation of near-term volatility, often referred to as the ”fear gauge” and widely used by investors to assess risk sentiment during periods of market uncertainty.

DAX (DAX Volatility Index) represents the market’s expectation of short-term volatility in the German equity market. It is widely monitored by investors to assess risk sentiment and uncertainty surrounding the DAX 40 Index and broader eurozone conditions.

### E.5 Conclusion

This benchmark offers a consistent and comprehensive setting for evaluating trading systems across a range of asset classes. By standardizing the data resolution and segment format, it ensures fair and reproducible comparisons while still capturing the variety found in real-world markets. The inclusion of both stable, trend-following assets and more volatile instruments enables meaningful stress testing of model performance within multi-agent high-frequency trading frameworks like QuantAgent.

Appendix F Case Studies
-----------------------

When presented with the unannotated K‑line window in Figure [4](https://arxiv.org/html/2509.09995v3#S4.F4 "Figure 4 ‣ 4 Experiments ‣ QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading"), the Pattern Agent first extracts swing pivots from recent bars and ranks successive local highs. The resulting pivot sequence forms a monotonic decline; a least‑squares fit through those highs yields a negatively sloped resistance trajectory. In parallel, repeated lows cluster within a narrow tolerance band near the 78 78 price mark, producing an effectively horizontal support shelf. The vertical distance between the declining highs and flat support narrows over time, flagging range compression characteristic of a Descending Triangle. From these primitives the agent composes its three summaries: the Structure Summary reports “lower highs” over “relatively flat support”; the Trend Summary maps the recognized class to its empirical bearish bias, heightened breakdown probability once support is retested multiple times; and the Symmetry Summary abstracts the converging lines into a triangular shape descriptor used downstream for trigger/invalid level setting. The post‑figure callouts (Lower Highs, dashed triangle edges, EMA overlays) are illustrative aids added for the reader; the pattern classification itself arises strictly from the bar‑geometry analysis described above.

![Image 13: Refer to caption](https://arxiv.org/html/2509.09995v3/x13.png)

Figure 13: Case sample of the TrendAgent on DJI (2022). The agent fits an upward-sloping resistance line and flat support to recent closes, confirming successive higher highs and higher lows. It then generates three structured summaries: Resistance Line (upward slope), Support Line (untested flat base), and Price Behavior (higher highs/lows with pullback). The green “Bullish Signal” badge denotes the agent’s final bullish assessment.

When the Trend Agent ingests the recent close‑anchored K‑line window in Figure [13](https://arxiv.org/html/2509.09995v3#A6.F13 "Figure 13 ‣ Appendix F Case Studies ‣ QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading"), it fits a two‑sided price channel to the rolling closes: a positively sloped upper boundary (resistance) and a relatively flat lower boundary (support). Repeated touches and near‑touch rejections along the upper fit, combined with a long gap since the last interaction with support, signal that price action is tracking the upper rail of an advancing channel rather than oscillating symmetrically about its midline. The agent also tallies successive swing highs and lows; the resulting sequence is net higher, reinforcing an upward trend classification. These geometric diagnostics are distilled into the three text panels shown beneath the chart: (1) the Resistance Line summary notes a recent probe above the red boundary and its upward slope; (2) the Support Line summary records that price is far from a relatively flat blue base, implying untested downside room; and (3) the Price Behavior summary highlights the cluster of higher highs/higher lows and a pullback after resistance contact. The green “Bullish Signal” callout in the figure reflects this composite assessment: trend up, price extended near resistance, watch for either a breakout continuation or a tactical pullback entry toward support.

![Image 14: Refer to caption](https://arxiv.org/html/2509.09995v3/x14.png)

Figure 14: Case sample of the IndicatorAgent on DJI (2019–2020). From top to bottom, the panels show: (i) the four-hour RSI with neutral (50) and overbought thresholds, (ii) the MACD line and its signal line with histogram, (iii) the Stochastic %K/%D oscillator, (iv) the RoC rate-of-change, and (v) Williams %R. This multi-panel visualization presents the raw indicator series that define the momentum and oscillator primitives underlying the agent’s “bullish but extended” assessment.

Given the latest OHLCV window, the Indicator Agent applies its momentum/oscillator transform suite (RSI, MACD, ROC, Stochastic, Williams %R) and aggregates the resulting state vector into a concise risk read. RSI has held above the neutral 50 line for most of the lookback and is presently in the high–60s (∼68\sim 68), signaling sustained upside participation but proximity to the classic overbought band. MACD remains above its signal line with a positive histogram bulge, confirming that upside momentum is still in force. ROC readings are modestly positive (≈1%\approx 1\%), reinforcing a persistent upward rate of change rather than an exhausted spike. At the same time, fast oscillators cluster in warning territory: both Stochastic %K/%D and Williams %R sit in overbought zones (>80​and>−20>80\ \text{and}\ >-20, respectively), indicating that the advance is stretched and vulnerable to a pause or mean reversion (Achelis, [2013](https://arxiv.org/html/2509.09995v3#bib.bib53)). The agent therefore issues a composite summary of “bullish but extended”: upside bias intact, yet tactical entries should respect exhaustion risk, tighten stops, scale position size, or await a reset toward support before adding exposure.

Appendix G Indicator Explanation
--------------------------------

Among the selected indicators, MACD is particularly representative due to its strengths in trend-following. MACD is designed to indicate momentum shifts by analyzing the divergence between two exponential moving averages (EMAs). It is calculated as:

ℰ t=α⋅𝒫 t+(1−α)⋅ℰ t−1 ℳ t=ℰ t(f)−ℰ t(s)𝒮 t=ℰ ℳ t\mathcal{E}_{t}=\alpha\cdot\mathcal{P}_{t}+(1-\alpha)\cdot\mathcal{E}_{t-1}\quad\quad\quad\quad\mathcal{M}_{t}=\mathcal{E}^{(f)}_{t}-\mathcal{E}^{(s)}_{t}\quad\quad\quad\quad\mathcal{S}_{t}=\mathcal{E}_{\mathcal{M}_{t}}

The exponential moving average (EMA), denoted as ℰ t\mathcal{E}_{t}, is a weighted average of past prices that assigns greater significance to more recent observations, thereby making it more responsive to recent price changes. Specifically, 𝒫 t\mathcal{P}_{t} represents the current price at time t t, ℰ t−1\mathcal{E}_{t-1} is the EMA value from previous timestep, and α∈(0,1)\alpha\in(0,1) is the smoothing factor that determines the relative weight of the current price versus past EMA values. The factor α\alpha is typically computed as α=2 N+1\alpha=\frac{2}{N+1} where N N is the number of time periods (Achelis, [2013](https://arxiv.org/html/2509.09995v3#bib.bib53)). Overall, EMA offers a smoothed representation of price trends, emphasizing recent movements while retaining historical context.

In our system, we define a fast EMA ℰ t(f)\mathcal{E}^{(f)}_{t} with N=12 N=12 and a slow EMA ℰ t(s)\mathcal{E}^{(s)}_{t} with N=26 N=26. The momentum signal ℳ t\mathcal{M}_{t} is then computed as the difference: ℳ t=ℰ t(f)−ℰ t(s)\mathcal{M}_{t}=\mathcal{E}^{(f)}_{t}-\mathcal{E}^{(s)}_{t} The signal line 𝒮 t\mathcal{S}_{t} is constructed as a 9-period EMA over the MACD sequence ℰ ℳ t\mathcal{E}_{\mathcal{M}_{t}}: 𝒮 t=ℰ ℳ t\mathcal{S}_{t}=\mathcal{E}_{\mathcal{M}_{t}}. A bullish signal occurs when ℳ t\mathcal{M}_{t} crosses above 𝒮 t\mathcal{S}_{t}, indicating upward momentum, whereas a downward crossover suggests growing bearish pressure. This formulation makes ℳ t\mathcal{M}_{t} effective for capturing mid-term trend shifts (e.g., on 4-hour charts), while filtering out high-frequency price noise (Appel, [2005](https://arxiv.org/html/2509.09995v3#bib.bib24)).

Appendix H The Use of Large Language Models (LLMs)
--------------------------------------------------

Large Language Models (LLMs) were employed only as supportive instruments to enhance the readability and grammatical precision of our academic writing. In particular, GPT-4o was utilized to aid in refining portions of the manuscript, including the introduction and methodology. The authors maintain complete responsibility for the intellectual content, encompassing the formulation of research questions, the design of methods, and the verification of experimental findings.

