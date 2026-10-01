Title: QuantCode-Bench: A Benchmark for Evaluating the Ability of Large Language Models to Generate Executable Algorithmic Trading Strategies

URL Source: https://arxiv.org/html/2604.15151

Markdown Content:
###### Abstract

Large language models have demonstrated strong performance on general-purpose programming tasks, yet their ability to generate executable algorithmic trading strategies remains underexplored. Unlike standard code benchmarks, trading-strategy generation requires simultaneous mastery of domain-specific financial logic, knowledge of a specialized API, and the ability to produce code that is not only syntactically correct but also leads to actual trades on historical data. In this work, we present QuantCode-Bench, a benchmark for the systematic evaluation of modern LLMs in generating strategies for the Backtrader framework from textual descriptions in English. The benchmark contains 400 tasks of varying difficulty collected from Reddit, TradingView, StackExchange, GitHub, and synthetic sources. Evaluation is conducted through a multi-stage pipeline that checks syntactic correctness, successful backtest execution, the presence of trades, and semantic alignment with the task description using an LLM judge. We compare state-of-the-art models in two settings: single-turn, where the strategy must be generated correctly on the first attempt, and agentic multi-turn, where the model receives iterative feedback and may repair its errors. Results show that even the best frontier models achieve only about 70–76% Judge Pass in the single-turn setting, whereas the best models in the agentic setting reach 95–98%. We analyze the failure modes across different stages of the pipeline and show that the main limitations of current models are not related to syntax, but rather to the correct operationalization of trading logic, proper API usage, and adherence to task semantics. These findings suggest that trading strategy generation constitutes a distinct class of domain-specific code generation tasks in which success requires not only technical correctness, but also alignment between natural-language descriptions, financial logic, and the observable behavior of the strategy on data.

## 1 Introduction

The ability of large language models to generate code has rapidly become one of the central areas for evaluating modern AI systems. However, most existing benchmarks focus either on general programming tasks, code repair, or repository-level software engineering problems. Such benchmarks include, for example, SWE-Bench Verified, LiveCodeBench, Terminal-Bench 2.0, and SWE-rebench, which have substantially advanced the evaluation of LLMs in code repair, self-repair, and agentic software engineering[[7](https://arxiv.org/html/2604.15151#bib.bib7), [6](https://arxiv.org/html/2604.15151#bib.bib6), [1](https://arxiv.org/html/2604.15151#bib.bib1), [9](https://arxiv.org/html/2604.15151#bib.bib9)]. Nevertheless, these benchmarks do not fully capture model behavior in domain-specific applied settings, where a model must simultaneously understand the subject matter, follow a specialized API, and produce code that exhibits meaningful behavior when executed.

One such task is the generation of algorithmic trading strategies. A model must interpret a textual description of a trading idea, identify the indicators, entry and exit conditions, position management rules, and possible parameters. It must then translate this description into correct code for a specific framework, in our case Backtrader, while respecting its interfaces, indexing conventions, indicator syntax, and execution semantics[[12](https://arxiv.org/html/2604.15151#bib.bib12)]. Finally, the generated code must not only execute successfully, but also produce actual trading signals on historical data. Even when syntax is fully correct, a strategy may still be non-functional because of overly strict thresholds, incorrect interpretation of conditions, or the absence of a link between indicators and trading actions.

This combination of requirements makes trading strategy generation fundamentally different from most existing code benchmarks. In standard coding tasks, successful compilation and passing tests often serve as sufficient indicators of quality[[7](https://arxiv.org/html/2604.15151#bib.bib7), [6](https://arxiv.org/html/2604.15151#bib.bib6)]. In algorithmic trading, this is not enough. Code may be technically correct, may pass a backtest successfully, and still be functionally useless if it generates no trades. Moreover, a strategy may be executable and even place trades, yet still fail to match the original task description. As a result, this setting requires a stricter and more layered evaluation protocol.

Interest in applying LLMs to finance is growing rapidly, but most of the literature focuses on financial NLP, question answering, document analysis, information extraction, forecasting, and agentic retrieval. This is reflected in works such as PIXIU, FinBen, FinanceBench, Fin-R1, Fino1, Finance Agent Benchmark, and FinAgentBench[[13](https://arxiv.org/html/2604.15151#bib.bib13), [14](https://arxiv.org/html/2604.15151#bib.bib14), [5](https://arxiv.org/html/2604.15151#bib.bib5), [8](https://arxiv.org/html/2604.15151#bib.bib8), [10](https://arxiv.org/html/2604.15151#bib.bib10), [2](https://arxiv.org/html/2604.15151#bib.bib2), [3](https://arxiv.org/html/2604.15151#bib.bib3)]. Against this background, there is still a lack of benchmarks that specifically measure the ability of LLMs to translate natural-language strategy descriptions into executable trading-system code.

In this work, we present QuantCode-Bench, a benchmark for evaluating the ability of LLMs to generate executable trading strategies from textual descriptions. The benchmark is built around the Backtrader framework and includes 400 tasks. We consider two interaction settings. In the single-turn setting, the model must solve the task on the first attempt, without any opportunity for revision. In the agentic multi-turn setting, the model receives structured feedback after each failure and may iteratively improve the code. This setup makes it possible to separately evaluate a model’s one-shot generation ability and its capacity to repair its own errors in an interactive scenario.

The core idea of QuantCode-Bench is that successful trading strategy generation should not be defined by a single criterion, but by a sequence of nested requirements. To this end, we use a four-stage evaluation pipeline: syntactic correctness, successful execution on historical data, the presence of at least one trade, and final semantic validation with an LLM judge. The use of LLM-as-a-Judge in open-ended tasks has already become widespread[[15](https://arxiv.org/html/2604.15151#bib.bib15), [4](https://arxiv.org/html/2604.15151#bib.bib4)]. This evaluation design makes it possible to distinguish at least four types of model capability: the ability to generate correct code, the ability to construct an executable strategy, the ability to formulate conditions that lead to real trading signals, and the ability to implement the actual trading idea described in the prompt.

Our experiments show that the task remains challenging even for the strongest models. In the single-turn setting, frontier models achieve nearly perfect compilation rates but degrade substantially at later stages of the pipeline. The best Judge Pass does not exceed roughly half of the benchmark. This means that the main challenge lies not in surface-level syntax, but in the deeper operationalization of trading logic. In the agentic setting, performance improves sharply: many errors prove to be locally repairable when feedback is available. However, a portion of the final failures in this setting still stem from incorrect interpretation of the natural-language specification rather than merely from local code defects.

Our contributions are as follows. First, we introduce a benchmark specifically designed for the generation of executable algorithmic trading strategies. Second, we propose a multi-level evaluation framework that distinguishes technical executability, the presence of trading behavior, and semantic alignment with the task specification. Third, we perform a broad comparison of modern models in both single-turn and agentic settings. Fourth, we conduct a detailed error analysis and identify the dominant failure modes at different stages of the pipeline. Fifth, we release the benchmark as a reproducible foundation for future research in domain-specific code generation for financial applications.

## 2 QuantCode-Bench

### 2.1 Task Definition

QuantCode-Bench evaluates a model’s ability to generate a Backtrader trading strategy from a textual description, subject to four nested requirements. First, the strategy must be syntactically correct. Second, it must execute successfully within the backtesting environment. Third, it must place at least one trade on the provided historical data. Fourth, it must match the described trading idea rather than merely producing an arbitrary working template.

This formulation makes the benchmark substantially stricter than typical coding tasks. Each successive validation stage strengthens the notion of success. Successful compilation does not guarantee successful execution. Successful execution does not guarantee the presence of trading signals. The presence of trades does not guarantee compliance with the textual specification. Therefore, our primary overall metric is Judge Pass, defined as the proportion of tasks for which the generated strategy passes the entire evaluation pipeline.

### 2.2 Dataset

The QuantCode-Bench dataset contains 400 trading-strategy generation tasks. The descriptions were collected from multiple sources that differ in formality, structure, and level of detail:

Table 1: Source distribution of QuantCode-Bench tasks.

Each task underwent structural enrichment. From the original description, we extracted the indicators used, the entry and exit conditions, and any additional rules, whether stated explicitly or implicitly. Each task was then assigned a difficulty category: easy, medium, or hard.

The distribution of tasks by source and difficulty is shown in Table[2](https://arxiv.org/html/2604.15151#S2.T2 "Table 2 ‣ 2.2 Dataset ‣ 2 QuantCode-Bench ‣ QuantCode-Bench: A Benchmark for Evaluating the Ability of Large Language Models to Generate Executable Algorithmic Trading Strategies").

Table 2: Difficulty distribution of QuantCode-Bench tasks by source.

### 2.3 Why Backtrader

We selected Backtrader because of its widespread use as an open-source framework for backtesting and prototyping trading strategies, as well as the nontrivial complexity of its API[[12](https://arxiv.org/html/2604.15151#bib.bib12)]. Unlike simpler educational interfaces, Backtrader requires the model to correctly handle indicators, data lines, order execution methods, and indexing conventions. This makes the benchmark realistic for applied code generation and reduces the chance that success is achieved through superficial reproduction of standard templates.

## 3 Evaluation Methodology

### 3.1 Validation Pipeline

Evaluation in QuantCode-Bench is performed using a four-stage pipeline. A strategy is counted as successful only if it passes all stages sequentially:

1.   1.
Compilation — the code is syntactically correct and can be interpreted without errors.

2.   2.
Backtest — the strategy executes successfully in the evaluation environment on benchmark-provided historical market data spanning diverse assets and timeframes without runtime errors.

3.   3.
Trade — the strategy places at least one trade.

4.   4.
Judge — an LLM judge confirms that the implemented strategy matches the textual task description[[15](https://arxiv.org/html/2604.15151#bib.bib15), [4](https://arxiv.org/html/2604.15151#bib.bib4)].

This pipeline makes it possible to localize the failure point and decompose unsuccessful generations by level. For example, one model may achieve nearly perfect compilation but often fail at execution; another may consistently pass the backtest but generate strategies with no trades; a third may trade successfully but implement the wrong logic. This decomposition is especially important in domain-specific tasks, where a single aggregate metric obscures qualitatively different causes of failure.

### 3.2 LLM Judge

The final stage of the pipeline is designed to verify the semantic alignment between the generated strategy and the original task description. This stage is necessary because a strategy may be technically functional but substantively incorrect. For example, instead of an RSI-based strategy, a model might generate an SMA crossover strategy that compiles, passes the backtest, and produces trades, but does not solve the requested task.

To address this issue, we use an LLM judge that evaluates the code according to three criteria:

*   •
whether the indicators used correspond to those in the original description or to an equivalent formalization;

*   •
whether the key entry, exit, and behavior logic of the strategy is implemented;

*   •
whether the code constitutes a relevant implementation of the given task rather than a generic template substitution.

This approach is consistent with the broader literature on LLM-as-a-Judge, in which strong models are used as scalable proxies for expert evaluation in open-ended tasks[[15](https://arxiv.org/html/2604.15151#bib.bib15), [4](https://arxiv.org/html/2604.15151#bib.bib4)].

### 3.3 Evaluation Settings

We consider two interaction settings.

In the single-turn setting, the model receives the task description and must generate a correct strategy on the first attempt. This scenario measures one-shot generation quality and is sensitive to the model’s initial knowledge of the domain, the library, and common strategy templates.

In the agentic multi-turn setting, after each unsuccessful attempt the model receives structured feedback containing the error type and the corresponding system message. The model may revise the code and retry up to 10 times. This setting measures the model’s ability to iteratively repair errors, perform local search, and use diagnostic information. Similar evaluation regimes have already proven informative in broader benchmarks for code and agentic software engineering[[7](https://arxiv.org/html/2604.15151#bib.bib7), [6](https://arxiv.org/html/2604.15151#bib.bib6), [1](https://arxiv.org/html/2604.15151#bib.bib1)].

## 4 Results

We evaluate models of different capacities in two settings: single-turn and agentic multi-turn. Tables[3](https://arxiv.org/html/2604.15151#S4.T3 "Table 3 ‣ 4.1 Single-turn ‣ 4 Results ‣ QuantCode-Bench: A Benchmark for Evaluating the Ability of Large Language Models to Generate Executable Algorithmic Trading Strategies") and[4](https://arxiv.org/html/2604.15151#S4.T4 "Table 4 ‣ 4.2 Agentic multi-turn ‣ 4 Results ‣ QuantCode-Bench: A Benchmark for Evaluating the Ability of Large Language Models to Generate Executable Algorithmic Trading Strategies") summarize the corresponding results.

### 4.1 Single-turn

Table[3](https://arxiv.org/html/2604.15151#S4.T3 "Table 3 ‣ 4.1 Single-turn ‣ 4 Results ‣ QuantCode-Bench: A Benchmark for Evaluating the Ability of Large Language Models to Generate Executable Algorithmic Trading Strategies") reports the single-turn results on QuantCode-Bench. The ranking already shows the central pattern of the benchmark: frontier models are almost uniformly strong on compilation, but substantially more dispersed on the later stages of the evaluation pipeline.

Table 3: Single-turn results on QuantCode-Bench (multi-timeframe).

As shown in Table[3](https://arxiv.org/html/2604.15151#S4.T3 "Table 3 ‣ 4.1 Single-turn ‣ 4 Results ‣ QuantCode-Bench: A Benchmark for Evaluating the Ability of Large Language Models to Generate Executable Algorithmic Trading Strategies"), the single-turn results reveal a sharp divergence between the early and late stages of the pipeline. For most strong models, compilation has almost ceased to be a bottleneck. However, a high Compilation Rate does not automatically translate into a high Judge Pass. This indicates that producing a syntactically correct strategy scaffold is no longer the main limitation for modern frontier models; the major quality losses occur at the Backtest and Trade stages.

### 4.2 Agentic multi-turn

Table[4](https://arxiv.org/html/2604.15151#S4.T4 "Table 4 ‣ 4.2 Agentic multi-turn ‣ 4 Results ‣ QuantCode-Bench: A Benchmark for Evaluating the Ability of Large Language Models to Generate Executable Algorithmic Trading Strategies") reports the cumulative performance in the agentic setting. Relative to Table[3](https://arxiv.org/html/2604.15151#S4.T3 "Table 3 ‣ 4.1 Single-turn ‣ 4 Results ‣ QuantCode-Bench: A Benchmark for Evaluating the Ability of Large Language Models to Generate Executable Algorithmic Trading Strategies"), the multi-turn protocol makes it possible to observe how quickly each model converts partial failures into final task success.

Table 4: Agentic multi-turn results on QuantCode-Bench (multi-timeframe). T1–T10 denote cumulative success by turn.

Taken together, the results in Table[4](https://arxiv.org/html/2604.15151#S4.T4 "Table 4 ‣ 4.2 Agentic multi-turn ‣ 4 Results ‣ QuantCode-Bench: A Benchmark for Evaluating the Ability of Large Language Models to Generate Executable Algorithmic Trading Strategies") show that the main differences between models emerge not at the level of syntactic correctness, but at the levels of execution, generation of trading signals, and semantic compliance with the task. Iterative feedback is especially effective for strong models, for which a substantial fraction of errors are locally repairable within a small number of attempts.

## 5 Error Analysis

### 5.1 Distribution by Failure Stage (single-turn)

Table[5](https://arxiv.org/html/2604.15151#S5.T5 "Table 5 ‣ 5.1 Distribution by Failure Stage (single-turn) ‣ 5 Error Analysis ‣ QuantCode-Bench: A Benchmark for Evaluating the Ability of Large Language Models to Generate Executable Algorithmic Trading Strategies") aggregates single-turn outcomes by the first stage at which a generation fails. This view complements the model-level results in Section 4 by showing where difficulty concentrates in the pipeline overall.

Table 5: Failure stage distribution in the single-turn setting.

As summarized in Table[5](https://arxiv.org/html/2604.15151#S5.T5 "Table 5 ‣ 5.1 Distribution by Failure Stage (single-turn) ‣ 5 Error Analysis ‣ QuantCode-Bench: A Benchmark for Evaluating the Ability of Large Language Models to Generate Executable Algorithmic Trading Strategies"), the key finding is that compilation has almost ceased to be the main problem for modern models. The main failure points lie at later stages of the pipeline, namely Backtest and No trades. This indicates that in trading-strategy generation tasks, the core difficulty is no longer Python syntax, but rather the correct operationalization of the strategy within a domain-specific execution environment.

### 5.2 Classification of Backtest Errors and Late-Stage Failures

Table LABEL:tab:error_types provides a finer-grained taxonomy of runtime and late-stage failures. Unlike Table[5](https://arxiv.org/html/2604.15151#S5.T5 "Table 5 ‣ 5.1 Distribution by Failure Stage (single-turn) ‣ 5 Error Analysis ‣ QuantCode-Bench: A Benchmark for Evaluating the Ability of Large Language Models to Generate Executable Algorithmic Trading Strategies"), which records only the first failed stage, this breakdown exposes the dominant technical and semantic patterns inside the broad Backtest, No trades, and late-judge failure categories.

Table 6: Taxonomy of backtest and late-stage failures.

| Error type | % | Description |
| --- | --- | --- |
| Signal conditions do not activate on data | 17.8 | Code compiles and backtests, but entry or exit conditions never trigger. |
| __bool__ / Line object errors | 13.1 | Using Backtrader Line objects in boolean context without [0] indexing. |
| Missing attribute/method | 3.9 | Access to nonexistent attributes of the Backtrader API. |
| Wrong API params | 3.9 | Incorrect constructor arguments for indicators (e.g., MACD period_fast). |
| Strategy doesn’t match task | 2.7 | Strategy is executable and trades but does not match the specification. |
| Type/NoneType errors | 2.1 | Operations involving None or incompatible types. |
| Other runtime errors | 1.4 | Other runtime failures. |
| Syntax errors (runtime) | 1.2 | Syntax errors caught at runtime in exec(). |
| Index out of range | 1.0 | Accessing data before enough history accumulated. |
| Compilation error | 0.3 | Structural compilation errors. |
| Execution timeout | 0.2 | Infinite loops or time limit exceeded. |

Table 6: Taxonomy of backtest and late-stage failures (continued).

As shown in Table LABEL:tab:error_types, the most frequent failure type involves strategies that compile successfully and pass the backtest, but place no trades on the data. This is typically caused by overly strict or unrealistic entry conditions, insufficient historical context for feature computation, or incorrect operationalization of indicator logic. The second most frequent category is  __bool__  / Line object errors, which reflects incorrect handling of Backtrader line objects in boolean conditions. Missing attribute/method errors account for a smaller share of failures than the leading categories, indicating that direct API hallucinations are less prevalent than logic-activation and line-object handling failures.

### 5.3 Errors in the Agentic Setting

Tables[7](https://arxiv.org/html/2604.15151#S5.T7 "Table 7 ‣ 5.3 Errors in the Agentic Setting ‣ 5 Error Analysis ‣ QuantCode-Bench: A Benchmark for Evaluating the Ability of Large Language Models to Generate Executable Algorithmic Trading Strategies") and[8](https://arxiv.org/html/2604.15151#S5.T8 "Table 8 ‣ 5.3 Errors in the Agentic Setting ‣ 5 Error Analysis ‣ QuantCode-Bench: A Benchmark for Evaluating the Ability of Large Language Models to Generate Executable Algorithmic Trading Strategies") summarize the final outcome distribution in the agentic setting and compare selected single-turn error categories with the last-turn composition for failed agentic trajectories.

Table 7: Final outcome distribution in the agentic setting.

Table 8: Comparison of selected error categories in the single-turn setting and at the last turn for failed agentic trajectories. The Turn 1 column reports the corresponding single-turn shares from Table LABEL:tab:error_types.

Compared with the single-turn profile, the composition of unresolved failures at the last agentic turn shifts toward categories that reflect persistent semantic and logic-level problems. In particular, the shares of Strategy doesn’t match task, Signal conditions do not activate on data, and  __bool__  / Line object errors are larger among the remaining failures at the last turn, whereas Missing attribute/method remains comparatively infrequent. For strategies that remain unsolved after 10 attempts, Judge rejection becomes one of the main causes of final failure.

This shows that iterative debugging is effective primarily for repairing technical errors, but substantially less effective when the model misunderstands the task itself. Therefore, the agentic setting mainly addresses program-repair problems, but does not fully eliminate limitations in the semantic interpretation of natural-language specifications.

## 6 Discussion

The generation of algorithmic trading strategies constitutes a distinct class of tasks at the intersection of programming, financial logic, and agentic search. The results of QuantCode-Bench show that even very strong models have already mastered some components of this problem, while remaining significantly limited in others.

The first important conclusion is that modern LLMs have largely solved the problem of surface-level syntactic generation. For strong models, compilation is no longer a meaningful bottleneck. The main challenge has shifted to the level of operational formalization: the model must not merely express a trading idea in code, but do so in a way that is executable, activatable on data, and semantically correct. This shift is particularly important for understanding what kinds of benchmarks are needed for the next stage of code-generation evaluation.

The second conclusion concerns the contrast between single-turn and agentic settings. The large improvement observed under iterative feedback shows that a substantial fraction of errors in QuantCode-Bench belongs to the class of locally repairable specification or API violations rather than to a fundamental inability to generate a strategy at all. This means that the practical usefulness of a model in this setting is determined not only by its single-turn accuracy, but also by its effectiveness in iterative code repair. This conclusion is conceptually aligned with observations from SWE-Bench Verified and SWE-rebench, where interactivity and repair play a major role in realistic evaluation of model capability[[7](https://arxiv.org/html/2604.15151#bib.bib7), [1](https://arxiv.org/html/2604.15151#bib.bib1)].

The third conclusion concerns the nature of difficulty. In benchmarks of the natural-language-to-code or natural-language-to-strategy type, difficulty is not simply a function of conceptual depth. Specification quality plays a major role. Vague and conversational descriptions are often harder for models than more formal and parameterized formulations, even when the latter are conceptually more complex.

The fourth conclusion concerns the distinction between general-purpose models and code-specialized models. The results of QuantCode-Bench show that specialization in programming does not guarantee superiority in domain-specific strategy generation. The likely reason is that the task requires not only technical discipline, but also precise interpretation of financial intent, translation of textual descriptions into behaviorally meaningful logic, and selection of realistic trigger conditions. In this setting, general-purpose models with stronger semantic and instruction-following capabilities often outperform specialized coding models.

The fifth conclusion concerns the role of the judge. Without semantic validation, Trade Rate systematically overestimates true success, because some strategies pass all technical checks yet still do not match the task. This is particularly important for open-ended benchmarks, where a model may generate a technically correct but semantically irrelevant implementation instead of the requested strategy. This is also supported by our reinforcement-learning experiments: when the reward function includes only technical pipeline completion and the presence of a trade, the model tends to exploit the reward by repeatedly generating the same working template that trades on the data but is unrelated to the original task. Adding the Judge stage eliminates this behavior by making semantic compliance part of the reward objective. In this sense, QuantCode-Bench shows that for open-ended domain-specific tasks, technical metrics alone are insufficient, and semantic validation must be part of the main evaluation procedure[[15](https://arxiv.org/html/2604.15151#bib.bib15), [4](https://arxiv.org/html/2604.15151#bib.bib4)].

## 7 Limitations

Although QuantCode-Bench covers an important and practically relevant class of tasks, the current version of the benchmark has several limitations.

First, all strategies are evaluated within a single framework and a single execution environment, namely Backtrader[[12](https://arxiv.org/html/2604.15151#bib.bib12)]. This improves experimental control and reduces infrastructural variability, but also limits the transferability of the results to other algorithmic-trading libraries and environments. Natural directions for extending the benchmark include QuantConnect/LEAN and Zipline[[11](https://arxiv.org/html/2604.15151#bib.bib11), [16](https://arxiv.org/html/2604.15151#bib.bib16)]. Evaluation across multiple frameworks would make it possible to better disentangle a model’s capacity for domain-specific strategy synthesis from its adaptation to a particular API.

Second, the final semantic evaluation relies on an LLM judge. Although this approach substantially strengthens the benchmark relative to purely technical validation, it does not provide an absolute guarantee of semantic correctness. The judge may overlook subtle mismatches, especially in cases where the strategy partially aligns with the task but diverges in details of the logic. In addition, the usual concerns associated with LLM-as-a-Judge remain relevant, including positional, stylistic, and model-specific biases that may influence the final evaluation[[15](https://arxiv.org/html/2604.15151#bib.bib15), [4](https://arxiv.org/html/2604.15151#bib.bib4)].

Third, the benchmark does not evaluate profitability, risk robustness, or the economic quality of the generated strategy. The presence of trades and alignment with the text do not imply that a strategy is effective as a trading system. In the present work, our focus is specifically on the ability of models to generate executable strategies from descriptions, not on the investment quality of those strategies.

## 8 Conclusion

In this work, we introduced QuantCode-Bench, a benchmark for evaluating the ability of large language models to generate executable algorithmic trading strategies. The benchmark formalizes the task as a sequence of nested requirements: syntactic correctness, successful execution, the presence of trades, and semantic alignment with the original description. This structure makes it possible to evaluate not only surface-level code quality, but also the deeper ability of a model to translate a natural-language trading idea into a behaviorally valid implementation.

The results show that even frontier models remain far from fully solving the task in the one-shot setting: the maximum single-turn Judge Pass now reaches roughly three quarters of the benchmark, but still falls well short of saturation. At the same time, the agentic setting with iterative feedback yields a sharp and consistent improvement, raising the best models to 95–98%. This indicates that a substantial fraction of errors is repairable and that model behavior in an interactive debugging loop may be at least as important as its accuracy in single-turn generation.

Taken together, the results of QuantCode-Bench show that trading-strategy generation requires simultaneous command of a specialized API, the ability to construct executable code, the capacity to formulate realistic trading logic, and adherence to the semantics of a natural-language specification. Modern models already perform well on the syntactic and basic infrastructural layers of the task, but still exhibit limitations in robust one-shot formalization of trading intent and in the precise implementation of the requested strategy.

QuantCode-Bench can serve as a useful tool for future research in domain-specific code generation, agentic software repair, and evaluation of LLMs in the financial domain.

## References

*   [1] Ibragim Badertdinov et al. Swe-rebench: An automated pipeline for task collection and decontaminated evaluation of software engineering agents, 2025. 
*   [2] Antoine Bigeard et al. Finance agent benchmark: Benchmarking llms on real-world financial research tasks, 2025. 
*   [3] Chanyeol Choi et al. Finagentbench: A benchmark dataset for agentic retrieval in financial question answering, 2025. 
*   [4] Jiawei Gu et al. A survey on LLM-as-a-judge, 2024. 
*   [5] Pranab Islam et al. Financebench: A new benchmark for financial question answering, 2023. 
*   [6] Naman Jain et al. Livecodebench: Holistic and contamination-free evaluation of large language models for code, 2024. 
*   [7] Carlos E. Jimenez et al. Swe-bench: Can language models resolve real-world github issues?, 2023. 
*   [8] Zhaowei Liu et al. Fin-r1: A large language model for financial reasoning through reinforcement learning, 2025. 
*   [9] Miles A. Merrill et al. Terminal-bench: Benchmarking agents on hard, realistic tasks, 2026. 
*   [10] Lingfei Qian et al. Fino1: On the transferability of reasoning-enhanced llms and reinforcement learning to finance, 2025. 
*   [11] QuantConnect. Quantconnect LEAN documentation. [https://www.quantconnect.com/docs/](https://www.quantconnect.com/docs/), 2026. 
*   [12] Daniel Rodriguez. Backtrader. [https://www.backtrader.com/](https://www.backtrader.com/), 2015. 
*   [13] Qianqian Xie et al. Pixiu: A large language model, instruction data and evaluation benchmark for finance, 2023. 
*   [14] Qianqian Xie et al. Finben: A holistic financial benchmark for large language models, 2024. 
*   [15] Lianmin Zheng et al. Judging LLM-as-a-judge with MT-bench and chatbot arena, 2023. 
*   [16] Zipline. Zipline documentation. [https://zipline.ml4trading.io/](https://zipline.ml4trading.io/), 2026. 

## Appendix

## Appendix A Example Tasks from QuantCode-Bench

Below we present one representative example for each difficulty level in QuantCode-Bench.

### A.1 Easy example

### A.2 Medium example

### A.3 Hard example

