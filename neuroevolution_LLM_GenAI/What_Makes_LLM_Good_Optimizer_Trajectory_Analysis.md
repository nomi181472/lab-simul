Title: What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search

URL Source: https://arxiv.org/html/2604.19440

Published Time: Wed, 22 Apr 2026 00:56:54 GMT

Markdown Content:
Xinhao Zhang, Xi Chen, François Portet, Maxime Peyrard

Univ. Grenoble Alpes, CNRS, Grenoble INP, LIG, 38000 Grenoble, France 

{[xinhao.zhang,](https://arxiv.org/html/2604.19440v1/mailto:Xinhao.Zhang@univ-grenoble-alpes.fr)[maxime.peyrard](https://arxiv.org/html/2604.19440v1/mailto:maxime.peyrard@univ-grenoble-alpes.fr)}@univ-grenoble-alpes.fr
Project Website: [github.io/traj_evo_search](https://xinhao-zhang.github.io/traj_evo_search/)

###### Abstract

Recent work has demonstrated the promise of orchestrating large language models (LLMs) within evolutionary and agentic optimization systems. However, the mechanisms driving these optimization gains remain poorly understood. In this work, we present a large-scale study of LLM-guided evolutionary search, collecting optimization trajectories for 15 LLMs across 8 tasks. Although zero-shot problem-solving ability correlates with final optimization outcomes, it explains only part of the variance: models with similar initial capability often induce dramatically different search trajectories and outcomes. By analyzing these trajectories, we find that strong LLM optimizers behave as local refiners, producing frequent incremental improvements while progressively localizing the search in semantic space. Conversely, weaker optimizers exhibit large semantic drift, with sporadic breakthroughs followed by stagnation. Notably, various measures of solution novelty do not predict final performance; novelty is beneficial only when the search remains sufficiently localized around high-performing regions of the solution space. Our results highlight the importance of trajectory analysis for understanding and improving LLM-based optimization systems and provide actionable insights for their design and training.

What Makes an LLM a Good Optimizer? 

A Trajectory Analysis of LLM-Guided Evolutionary Search

Xinhao Zhang, Xi Chen, François Portet, Maxime Peyrard Univ. Grenoble Alpes, CNRS, Grenoble INP, LIG, 38000 Grenoble, France{[xinhao.zhang,](https://arxiv.org/html/2604.19440v1/mailto:Xinhao.Zhang@univ-grenoble-alpes.fr)[maxime.peyrard](https://arxiv.org/html/2604.19440v1/mailto:maxime.peyrard@univ-grenoble-alpes.fr)}@univ-grenoble-alpes.fr Project Website: [github.io/traj_evo_search](https://xinhao-zhang.github.io/traj_evo_search/)

## 1 Introduction

Large language models (LLMs) are increasingly deployed as search operators in iterative optimization systems (Lehman et al., [2022](https://arxiv.org/html/2604.19440#bib.bib33 "Evolution through large models"); Peyrard et al., [2025](https://arxiv.org/html/2604.19440#bib.bib28 "Agentic ai: the era of semantic decoding")). Across diverse domains, such as prompt optimization (Agrawal et al., [2026](https://arxiv.org/html/2604.19440#bib.bib46 "GEPA: reflective prompt evolution can outperform reinforcement learning"); Guo et al., [2025](https://arxiv.org/html/2604.19440#bib.bib25 "EvoPrompt: connecting llms with evolutionary algorithms yields powerful prompt optimizers"); Fernando et al., [2023](https://arxiv.org/html/2604.19440#bib.bib24 "Promptbreeder: self-referential self-improvement via prompt evolution")) and scientific discovery (Romera-Paredes et al., [2023](https://arxiv.org/html/2604.19440#bib.bib38 "Mathematical discoveries from program search with large language models"); Ellenberg et al., [2025](https://arxiv.org/html/2604.19440#bib.bib36 "Generative modeling for mathematical discovery"); Novikov et al., [2025](https://arxiv.org/html/2604.19440#bib.bib17 "AlphaEvolve: a coding agent for scientific and algorithmic discovery"); Gottweis et al., [2025](https://arxiv.org/html/2604.19440#bib.bib34 "Towards an ai co-scientist")), LLMs are embedded into evolutionary or agentic loops as black-box optimizers where they repeatedly propose candidate solutions, receive feedback, and refine solutions iteratively. While such LLM-guided evolutionary workflows have been shown to deliver substantial empirical gains, the mechanisms underlying these improvements remain poorly understood. In particular, even under strictly controlled optimization loops, selection rules, and evaluation functions, different LLMs exhibit vastly different optimization trajectories and final performances. This observation motivates the central question of this work: _what explains such large model-to-model differences in optimization performance?_ Are these differences primarily a reflection of base model capability, or do they arise from more subtle differences in the exploration–exploitation dynamics induced by the models?

![Image 1: Refer to caption](https://arxiv.org/html/2604.19440v1/img/explore_bad_good.png)

Figure 1: Different optimization trajectories for two LLMs with similar zero-shot performance on TSP-60. Each point represents a candidate solution, colored by generation. Gemini-1.5-Pro (left) displays sustained fitness improvement and progressive localization. Mistral-7B-Instruct (right) maintains high novelty but fails to exploit it into fitness gains.

To address these questions, we conduct a large-scale study of LLM-based evolutionary optimization, collecting optimization trajectories for 15 LLMs across 4 task families (8 tasks), resulting in 72K analyzed candidate solutions. As expected, zero-shot performance correlates positively with final optimization outcomes. However, this relationship explains partially the variance: models with nearly identical zero-shot performance can diverge quickly after evolution, following qualitatively distinct optimization trajectories (see Figure[1](https://arxiv.org/html/2604.19440#S1.F1 "Figure 1 ‣ 1 Introduction ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search")). To explain this residual gap, we turn to the classical exploration–exploitation trade-off underlying evolutionary algorithms(Sun et al., [2018](https://arxiv.org/html/2604.19440#bib.bib13 "Balancing exploration and exploitation in multiobjective evolutionary optimization")). In LLM-driven metaheuristics, the mutation operator is no longer fully random but is strongly shaped by the LLM’s prior toward producing improved solutions conditioned on parent candidates and their fitness feedback. Consequently, exploration is more constrained than in classical, non-LLM-driven evolution. Under this view, optimization processes should benefit primarily from higher novelty and diversity, which would expose the system to a broader range of potentially useful regions in the search space. Surprisingly, our results contradict this intuition. We find that the most successful trajectories are not those exhibiting high novelty, but rather those characterized by _frequent and sustained breakthroughs_, where a breakthrough represents an incremental improvement of the best-so-far fitness. In other words, what distinguishes strong optimization runs is more of the ability to reliably produce incremental improvements repeatedly. This is different from typical behavior observed in meta-heuristics, where large breakthroughs occur at rare intervals, followed by long plateaus or small refinements(Mitchell and Taylor, [1999](https://arxiv.org/html/2604.19440#bib.bib22 "Evolutionary computation: an overview")). This interpretation is also supported by our perturbation experiments, where we directly manipulate the refinement behaviour of the search trajectory through model mixing, leading to predictable changes in optimization performance.

To further elucidate when and why breakthroughs happen, we analyze the geometry of optimization trajectories in semantic space. By embedding candidate solutions and characterizing within-generation distributions with entropy-based and dispersion measures, we reveal a clear distinction between strong and weak optimizers. Effective LLM operators progressively _localize_ their search around high-performing regions of semantic space, whereas weaker optimizers continue to diffuse and drift across distant regions. A generation-level mixed-effects analysis further uncovers an interesting interaction between novelty and semantic dispersion: novelty increases the probability of breakthroughs _only when_ the search remains sufficiently localized. Outside this regime, novelty is largely unproductive.

Contributions. We make the following contributions: (i) We conduct a large-scale, controlled study of LLM-based evolutionary optimization and release the resulting optimization trajectories. (ii) We show that differences in optimization performance between LLMs are only partially explained by zero-shot capability, unveiling a distinct notion of _optimizable ability_. (iii) We identify effective LLM optimizers as _local refiners_, whose trajectories progressively localize in semantic space and yield frequent incremental breakthroughs, and we support this mechanism through perturbation experiments that highlight the role of refinement behavior. (iv) We demonstrate that novelty is not inherently beneficial; its utility is conditional on the geometric regime of search, and it becomes productive only when search remains localized. (v) We derive practical implications for model selection and for learning better search operators, showing that smaller or cheaper models can outperform stronger base models when they exhibit more reliable refinement behavior.

More broadly, our semantic trajectory analysis offers a reusable framework for studying LLM-driven optimization processes. Our findings suggest that, rather than solely pursuing general-purpose capability, future work may benefit from understanding, controlling, training models as effective search operators(Šurina et al., [2025](https://arxiv.org/html/2604.19440#bib.bib23 "Algorithm discovery with LLMs: evolutionary search meets reinforcement learning")) emphasizing local refinement and error correction.

## 2 Related Work

##### Evolutionary Computation with LLMs

LLMs are increasingly integrated into evolutionary computation frameworks(Yang et al., [2024](https://arxiv.org/html/2604.19440#bib.bib12 "Large language models as optimizers"); Wu et al., [2025](https://arxiv.org/html/2604.19440#bib.bib2 "Evolutionary computation in the era of large language model: survey and roadmap"); Brahmachary et al., [2024](https://arxiv.org/html/2604.19440#bib.bib14 "Large language model-based evolutionary optimizer: reasoning with elitism"); Tao et al., [2024](https://arxiv.org/html/2604.19440#bib.bib3 "A survey on self-evolution of large language models")), revitalizing meta-heuristic optimization. Unlike classical approaches that rely on stochastic operators to explore optimization landscapes(Holland, [1992](https://arxiv.org/html/2604.19440#bib.bib1 "Adaptation in natural and artificial systems: an introductory analysis with applications to biology, control, and artificial intelligence")), LLM-assisted methods instantiate variation operators through the semantic priors of LLMs(Josifoski et al., [2023](https://arxiv.org/html/2604.19440#bib.bib48 "Flows: building blocks of reasoning and collaborating ai"); Peyrard et al., [2025](https://arxiv.org/html/2604.19440#bib.bib28 "Agentic ai: the era of semantic decoding"); Gao et al., [2026](https://arxiv.org/html/2604.19440#bib.bib6 "A survey of self-evolving agents: what, when, how, and where to evolve on the path to artificial super intelligence"); Fang et al., [2025](https://arxiv.org/html/2604.19440#bib.bib7 "A comprehensive survey of self-evolving ai agents: a new paradigm bridging foundation models and lifelong agentic systems")). These frameworks have demonstrated strong empirical performance across domains including combinatorial optimization(Yang et al., [2025a](https://arxiv.org/html/2604.19440#bib.bib44 "HeurAgenix: leveraging llms for solving complex combinatorial optimization challenges"); Yu et al., [2026](https://arxiv.org/html/2604.19440#bib.bib4 "Large language model-driven full-component evolution of adaptive large neighborhood search")) and scientific discovery(Yang et al., [2025b](https://arxiv.org/html/2604.19440#bib.bib47 "MOOSE-chem: large language models for rediscovering unseen chemistry scientific hypotheses"); MacKnight et al., [2025](https://arxiv.org/html/2604.19440#bib.bib18 "Pre-trained knowledge elevates large language models beyond traditional chemical reaction optimizers"); Chen et al., [2026](https://arxiv.org/html/2604.19440#bib.bib43 "MolEvolve: llm-guided evolutionary search for interpretable molecular optimization"); Abhyankar et al., [2026](https://arxiv.org/html/2604.19440#bib.bib41 "LLEMA: evolutionary search with llms for multi-objective materials discovery"); Zhou et al., [2024](https://arxiv.org/html/2604.19440#bib.bib49 "Hypothesis generation with large language models")). Recent work has further extended these approaches to algorithm discovery in open-ended scenario(Lu et al., [2024](https://arxiv.org/html/2604.19440#bib.bib5 "The ai scientist: towards fully automated open-ended scientific discovery"); Gottweis et al., [2025](https://arxiv.org/html/2604.19440#bib.bib34 "Towards an ai co-scientist"); Qu et al., [2026](https://arxiv.org/html/2604.19440#bib.bib35 "CORAL: towards autonomous multi-agent evolution for open-ended discovery")), as well as more advanced settings incorporating co-evolution or meta-reflection to mitigate limited global search perspectives(Liu et al., [2024](https://arxiv.org/html/2604.19440#bib.bib69 "Evolution of heuristics: towards efficient automatic algorithm design using large language model"); Ye et al., [2024](https://arxiv.org/html/2604.19440#bib.bib11 "ReEvo: large language models as hyper-heuristics with reflective evolution")). Our work complements these efforts by analyzing the search behavior and optimization trajectories induced by LLMs. Trajectory analyses can also inform the design of agentic systems(Lee et al., [2026](https://arxiv.org/html/2604.19440#bib.bib40 "T-map: red-teaming llm agents with trajectory-aware evolutionary search"); Zhao et al., [2026](https://arxiv.org/html/2604.19440#bib.bib45 "Large language model-powered evolutionary code optimization on a phylogenetic tree"), [2025](https://arxiv.org/html/2604.19440#bib.bib20 "TrajEvo: designing trajectory prediction heuristics via llm-driven evolution"); Lin et al., [2025](https://arxiv.org/html/2604.19440#bib.bib19 "SE-agent: self-evolution trajectory optimization in multi-step reasoning with llm-based agents")). Related to our work are behaviour-space studies(van Stein et al., [2025](https://arxiv.org/html/2604.19440#bib.bib16 "Behaviour space analysis of llm-driven meta-heuristic discovery")) and LAS landscape analyses(Liu et al., [2025](https://arxiv.org/html/2604.19440#bib.bib15 "Fitness landscape of large language model-assisted automated algorithm search")), which similarly associate effective optimization with sustained improvements and increased exploitation. We extend these findings with a unified cross-model, cross-task analysis using semantic entropy measures.

##### Evaluating LLMs as Search Operators

The evaluation paradigm for LLMs has evolved accordingly. Early optimization benchmarks relied on single-pass prompting(Fan et al., [2024](https://arxiv.org/html/2604.19440#bib.bib29 "NPHardEval: dynamic benchmark on reasoning ability of large language models via complexity classes"); Duchnowski et al., [2025](https://arxiv.org/html/2604.19440#bib.bib8 "A knapsack by any other name: presentation impacts LLM performance on NP-hard problems")). More recent work evaluates LLMs within iterative or evolutionary search loops, treating them as search operators guided by external feedback(Li et al., [2025](https://arxiv.org/html/2604.19440#bib.bib10 "OPT-bench: evaluating llm agent on large-scale search spaces optimization problems"); Huang et al., [2024](https://arxiv.org/html/2604.19440#bib.bib30 "Exploring the true potential: evaluating the black-box optimization capability of large language models"); Ouyang et al., [2025](https://arxiv.org/html/2604.19440#bib.bib9 "KernelBench: can llms write efficient gpu kernels?"); Shojaee et al., [2025a](https://arxiv.org/html/2604.19440#bib.bib67 "LLM-sr: scientific equation discovery via programming with large language models"), [b](https://arxiv.org/html/2604.19440#bib.bib68 "LLM-srbench: a new benchmark for scientific equation discovery with large language models")). In this setting, LLMs are no longer assessed as one-shot solvers but as _semantic search operators_, whose preferences and biases matter(Zhou et al., [2026](https://arxiv.org/html/2604.19440#bib.bib37 "What matters to an LLM? behavioral and computational evidences from summarization")). While these benchmarks demonstrate strong end-to-end performance, evaluation remains largely outcome-centric. Our results show that base model capability and operator effectiveness are distinct skills, with direct implications for model selection and motivating work on learning specialized search operators, such as Brahmachary et al. ([2024](https://arxiv.org/html/2604.19440#bib.bib14 "Large language model-based evolutionary optimizer: reasoning with elitism")) and EvoTune(Šurina et al., [2025](https://arxiv.org/html/2604.19440#bib.bib23 "Algorithm discovery with LLMs: evolutionary search meets reinforcement learning")).

![Image 2: Refer to caption](https://arxiv.org/html/2604.19440v1/img/final_version_method.png)

Figure 2: Overview of the LLM-driven evolutionary search framework and tasks. Left: the evolutionary process across generations. Right: the within-generation loop—population initialization, LLM-guided mutation, fitness evaluation, and selection. Bottom: the four tasks and their corresponding genome representations.

## 3 Methodology

Following Novikov et al. ([2025](https://arxiv.org/html/2604.19440#bib.bib17 "AlphaEvolve: a coding agent for scientific and algorithmic discovery")), we implement a lightweight evolutionary search loop where LLMs act as semantic variation operators, iteratively generating candidate solutions in order to optimize task-specific fitness.

Population Initialization. For each task, we construct an initial population $\mathcal{P}_{0}$ consisting of valid genomes and their corresponding fitness values $\left(\right. g , f_{T} ​ \left(\right. g \left.\right) \left.\right)$. This initial population is fixed and shared across all models for the same task.

Fitness Evaluation. Each genome is evaluated by a task-specific fitness function $f_{T} ​ \left(\right. \cdot \left.\right)$. Invalid or unparsable outputs are assigned zero fitness.

Selection (Top-$q$ Weighted). At generation $t$, we form an elite subset $\mathcal{E}_{t} = Top_{\lceil q ​ N \rceil} ⁡ \left(\right. \mathcal{P}_{t} \left.\right)$ with $q$ fixed to $0.2$. Parents are sampled from $\mathcal{E}_{t}$ with probability proportional to their fitness: $Pr ⁡ \left(\right. x \mid \mathcal{E}_{t} \left.\right) \propto f_{T} ​ \left(\right. x \left.\right)$.

Mutation. Selected parent genomes are provided as the context of prompts to the LLM, which generates a set of offspring genomes $\mathcal{C}_{t}$ conditioned on the task and parent structure.

Population Pool Update. Generated offspring are deduplicated and merged into the population pool. If the pool size exceeds $N$, only the top-$N$ genomes ranked by fitness are retained. The best-so-far fitness is updated as $f_{t}^{\star} = max_{x \in \mathcal{P}_{t}} ⁡ f_{T} ​ \left(\right. x \left.\right)$.

### 3.1 Tasks & Genome Representations

Our evaluation includes tasks across four domains, spanning combinatorial, linguistic, symbolic, and algorithmic optimization where previous work has shown benefits from LLM-guided search.

![Image 3: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/route.png)Route Optimization The Traveling Salesman Problem (TSP) is a classical route optimization task. Prior work shows that while LLMs struggle to produce high-quality tours in a single pass (Fan et al., [2024](https://arxiv.org/html/2604.19440#bib.bib29 "NPHardEval: dynamic benchmark on reasoning ability of large language models via complexity classes")), evolutionary search can largely improve performance (Huang et al., [2024](https://arxiv.org/html/2604.19440#bib.bib30 "Exploring the true potential: evaluating the black-box optimization capability of large language models")). For each optimization run, one randomly generated distance matrix is given and the output must be a valid tour as a permutation of city indices. We evaluate two TSP variants with 30 and 60 cities, respectively. Each genome represents a permutation of the tour, and the fitness function is the inverse total distance.

![Image 4: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/text.png)Prompt Optimization It aims to automatically improve prompts quality. The genome is a textual instruction that conditions a frozen LLM (gpt-4o-mini). Evolutionary approaches have been shown effective in this setting (Guo et al., [2025](https://arxiv.org/html/2604.19440#bib.bib25 "EvoPrompt: connecting llms with evolutionary algorithms yields powerful prompt optimizers"); Fernando et al., [2023](https://arxiv.org/html/2604.19440#bib.bib24 "Promptbreeder: self-referential self-improvement via prompt evolution")). We evaluate on dialogue summarization (SAMSum (Gliwa et al., [2019](https://arxiv.org/html/2604.19440#bib.bib31 "SAMSum corpus: a human-annotated dialogue dataset for abstractive summarization"))) and text simplification (ASSET (Alva-Manchego et al., [2020](https://arxiv.org/html/2604.19440#bib.bib32 "ASSET: A dataset for tuning and evaluation of sentence simplification models with multiple rewriting transformations"))). Fitness is computed as the average generation quality on a held-out 25% validation subset, using ROUGE-L for SAMSum and SARI for ASSET, following Guo et al. ([2025](https://arxiv.org/html/2604.19440#bib.bib25 "EvoPrompt: connecting llms with evolutionary algorithms yields powerful prompt optimizers")).

![Image 5: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/math.png)Equation Discovery Symbolic regression aims to discover concise mathematical expressions that fit observed input–output pairs (Grayeli et al., [2024](https://arxiv.org/html/2604.19440#bib.bib39 "Symbolic regression with a learned concept library")). This domain is well-suited for LLM-guided evolutionary search which combines prior scientific knowledge with iterative refinement (Shojaee et al., [2025a](https://arxiv.org/html/2604.19440#bib.bib67 "LLM-sr: scientific equation discovery via programming with large language models"), [b](https://arxiv.org/html/2604.19440#bib.bib68 "LLM-srbench: a new benchmark for scientific equation discovery with large language models")). We adopt two nonlinear oscillation benchmarks from Shojaee et al. ([2025a](https://arxiv.org/html/2604.19440#bib.bib67 "LLM-sr: scientific equation discovery via programming with large language models")): Oscillator 1 (three variables) and Oscillator 2 (four variables). Each solution encodes a candidate symbolic expression executable as $f ​ \left(\right. x \left.\right)$. Fitness is measured as $f_{T} ​ \left(\right. \text{expr} \left.\right) = 1 - norm ⁡ \left(\right. MSE ​ \left(\right. \hat{y} , y \left.\right) \left.\right)$, where $\hat{y}$ denotes model predictions, and $norm$ is the min–max normalization computed over all candidate solutions for the same task instance.

![Image 6: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/code.png)Heuristic Design Heuristic design for combinatorial optimization aims to evolve executable programs rather than direct solutions. This paradigm has been successfully explored in recent work such as FunSearch and EoH (Romera-Paredes et al., [2023](https://arxiv.org/html/2604.19440#bib.bib38 "Mathematical discoveries from program search with large language models"); Liu et al., [2024](https://arxiv.org/html/2604.19440#bib.bib69 "Evolution of heuristics: towards efficient automatic algorithm design using large language model")). We focus on the online bin packing problem. Each genome encodes a heuristic policy in the form of a priority function that determines item placement. We evaluate on two datasets: OR3 (20 instances, 500 items each) and Weibull (5 instances, 5,000 items each), representing synthetic and real-world-like distributions. Fitness is the inverse number of bins used.

### 3.2 Novelty Computation

##### Task-agnostic novelty.

Besides fitness, we further quantify semantic diversity along the trajectory. We defined the novelty of a solution $a$ with respect to a set of all previous solutions $\mathcal{A}^{prior}$ including the initial parents under a task-specific semantic distance metric $D_{T}$: $n ​ o ​ v ​ \left(\right. a , \mathcal{A}^{prior} \left.\right) = min_{b \in \mathcal{A}^{prior}} ⁡ D_{T} ​ \left(\right. a , b \left.\right)$. Novelty is normalized at subtask-level to ensure comparability.

##### Task-specific semantic distance.

For TSP, we use an edge-set distance invariant to rotation and starting city. For prompt optimization, we compute cosine distance in a fixed embedding space using OpenAI’s text-embedding-ada-002. For equation discovery and heuristic design, we adopt a functional behavior distance measured over a fixed input grid to capture divergence in output behavior.

### 3.3 Evolution Scale and Parameters

Our study involves 15 LLMs, with 30 generations conducted for each (model, task) pair. In each generation, the population produces 10 offspring, each corresponding to a model call. Every model–task pair is repeated twice with the same initial population, thereby totaling over 72,000 API calls. All evolutions are conducted using a default temperature of 0.7. The total cost of running experiments is estimated to be around $500.1 1 1 All trajectory data is available at [https://huggingface.co/datasets/LivevreXH/evo_llm_trajectories](https://huggingface.co/datasets/LivevreXH/evo_llm_trajectories).

The selected LLMs span six model families: ![Image 7: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/openai.png) OpenAI (GPT-4o (OpenAI et al., [2024](https://arxiv.org/html/2604.19440#bib.bib60 "GPT-4o system card")), GPT-4o-mini, GPT-3.5-turbo), Google’s![Image 8: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/gemini.png) Gemini (Gemini-1.5-flash, Gemini-1.5-pro (Team et al., [2024](https://arxiv.org/html/2604.19440#bib.bib61 "Gemini 1.5: unlocking multimodal understanding across millions of tokens of context"))), Google’s![Image 9: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/gemma.png) Gemma-3n-4b (Team et al., [2025](https://arxiv.org/html/2604.19440#bib.bib62 "Gemma 3 technical report")), Meta’s![Image 10: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/meta.png) Llama (llama-3.1-70b-instruct, llama-3.1-8b-instruct, llama-3.2-1b-instruct, llama-3.2-3b-instruct (Grattafiori et al., [2024](https://arxiv.org/html/2604.19440#bib.bib63 "The llama 3 herd of models"))), Deepseek AI’s![Image 11: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/deepseek.png) Deepseek-V3 (DeepSeek-AI et al., [2025](https://arxiv.org/html/2604.19440#bib.bib64 "DeepSeek-v3 technical report")), MistralAI’s![Image 12: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/mistral.png) Mistral (Mistral-7b-instruct (Jiang et al., [2023](https://arxiv.org/html/2604.19440#bib.bib65 "Mistral 7b")), Mistral-24b-instruct, Mistral-large, Magistral-small (Mistral-AI et al., [2025](https://arxiv.org/html/2604.19440#bib.bib66 "Magistral"))). Additional experimental details are provided in Appendix[C](https://arxiv.org/html/2604.19440#A3 "Appendix C Task-Specific Experimental Details ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search").

## 4 Results and Analysis

##### The Optimization Gap.

We evaluate each LLM under a fixed evolutionary budget and conditions. Performance is measured by the best fitness at the end of evolution. Across task families, we observe a pronounced optimization gap between models. Concretely, under identical conditions, different LLMs lead to different optimization outcomes (see Table[2](https://arxiv.org/html/2604.19440#A1.T2 "Table 2 ‣ Appendix A Complete Experimental Result ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search")). Strong early performance does not reliably predict long-horizon outcomes. For instance, Deepseek-V3 performs the best in the first generation yet fail to achieve the largest gains over time. This suggests that LLMs differ not only in solution quality, but in the search process. In the following section, we progressively rule out alternative explanations, i.e. base capability, novelty, and identify a consistent mechanism for successful optimization.

### 4.1 Base Model Capability

We first hypothesize that the gap may simply stem from base model capability, specifically the model’s intrinsic task-specific problem-solving ability in zero-shot settings. Since all tasks are optimization-oriented, we define zero-shot performance as the best fitness achieved via temperature-swept _best-of-$N$_ sampling: for each model–task pair, we generate candidates across six temperatures ($T \in \left{\right. 0.0 , 0.2 , 0.4 , 0.6 , 0.8 , 1.0 \left.\right}$), with two samples per temperature, and report the best fitness among them (See Appendix[C](https://arxiv.org/html/2604.19440#A3 "Appendix C Task-Specific Experimental Details ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search") for more details).

As revealed in Figure[3](https://arxiv.org/html/2604.19440#S4.F3 "Figure 3 ‣ 4.1 Base Model Capability ‣ 4 Results and Analysis ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"), zero-shot performance is strongly correlated with post-optimization performance when aggregated across tasks. Similar trends are observed at the sub-task level (Appendix Figure[13](https://arxiv.org/html/2604.19440#A6.F13 "Figure 13 ‣ Appendix F Supplementary Visualizations ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search")), where higher zero-shot scores generally correspond to better final outcomes, only except for equation discovery tasks. This confirms that base capability is a strong predictor of optimization potential, yet insufficient to fully explain long-horizon optimization success. Models with nearly identical zero-shot performance can diverge substantially after evolution. For instance, around an average zero-shot score of 0.4 in Figure[3](https://arxiv.org/html/2604.19440#S4.F3 "Figure 3 ‣ 4.1 Base Model Capability ‣ 4 Results and Analysis ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"), multiple models cluster tightly along the zero-shot axis yet spread widely in their best final performance. This residual variance persists across tasks (See Figure[13](https://arxiv.org/html/2604.19440#A6.F13 "Figure 13 ‣ Appendix F Supplementary Visualizations ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search")).

![Image 13: Refer to caption](https://arxiv.org/html/2604.19440v1/img/zero_final_compare.png.png)

Figure 3: Scatter plot between zero-shot performance and final optimized performance across models.

### 4.2 Trajectory-Level Analysis

Since being a strong one-shot problem solver does not necessarily imply being an effective evolutionary search operator, the differences might lie in the search process. We therefore investigate trajectory-level properties of evolutionary search.

#### 4.2.1 Novelty vs. Breakthrough Dynamics

In classical evolutionary algorithms, novelty is treated as a proxy for exploration. However, this equivalence becomes problematic in LLM-guided evolution. Different from blind stochastic mutation operators, LLMs generate offspring by conditioning on parent solutions and task context aiming to produce the best solution. Novelty in this setting does not arise from random exploration, but from semantic variation within the LLM’s output.

##### More Novelty Doesn’t Yield Better Optimization.

A natural hypothesis is therefore that models generating more novel solutions should explore the search space more effectively and achieve better optimization outcomes. We test this hypothesis in Figure[4](https://arxiv.org/html/2604.19440#S4.F4 "Figure 4 ‣ Breakthrough Rate Strongly Predicts Optimization Performance. ‣ 4.2.1 Novelty vs. Breakthrough Dynamics ‣ 4.2 Trajectory-Level Analysis ‣ 4 Results and Analysis ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"), which summarizes both effect sizes and explanatory power across different trajectory-level descriptors. Astonishingly, novelty-based measures, including both average novelty and initial novelty (in the first generation), exhibit coefficients close to zero and are not statistically significant. Moreover, their explanatory power is negligible, meaning that increasing diversity alone does not contribute to improved optimization performance.

##### Breakthrough Rate Strongly Predicts Optimization Performance.

We define a breakthrough as a best-so-far improvement event, e.g., an offspring generation in which the current solution exceeds the best fitness solution in all previous generations. We quantify each optimization trajectory’s tendency to produce breakthroughs by its breakthrough rate. The breakthrough rate is also averaged per pair of models and tasks. As shown in Figure[4](https://arxiv.org/html/2604.19440#S4.F4 "Figure 4 ‣ Breakthrough Rate Strongly Predicts Optimization Performance. ‣ 4.2.1 Novelty vs. Breakthrough Dynamics ‣ 4.2 Trajectory-Level Analysis ‣ 4 Results and Analysis ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search") (left), the breakthrough rate has the largest positive coefficient among all predictors. This is further reflected in its explanatory power (right), where breakthrough rate alone explains around two times more variance than zero-shot capability. Beyond this, when combining breakthrough rate with zero-shot performance, the overall explanatory power increases further, while the coefficient of zero-shot performance decreases. This indicates that part of the predictive power of base capability is mediated through the ability to generate consistent improvements during search.

These results show that good optimization trajectories tend to frequently produce small improvements instead of big and rare breakthroughs followed by long plateaus, as it often happens in evolutionary search(Mitchell and Taylor, [1999](https://arxiv.org/html/2604.19440#bib.bib22 "Evolutionary computation: an overview")).

![Image 14: Refer to caption](https://arxiv.org/html/2604.19440v1/img/nov_regre.png)

Figure 4: OLS regression results across different trajectory descriptors. (Left) Standardized coefficients. (Right) Explanatory power. $\_{}^{* \llbracket * *}p < 0.001$, $\_{}^{*}p < 0.05$, $n ​ s$ means non-significant p-values. Novelty-based predictors are not significant, whereas breakthrough rate (BR) strongly predicts performance and improves fit beyond zero-shot capability (ZS). 

#### 4.2.2 Semantic Geometry

In LLM-guided evolution, mutation operators are black-boxes, making it difficult to directly interpret how search progresses. To understand why some trajectories yield more breakthroughs than others, we instead examine the geometry of the search process by analyzing how solutions are distributed in semantic space over time.

![Image 15: Refer to caption](https://arxiv.org/html/2604.19440v1/img/mds_compare_v2.png)

(a) Search Topology Visualization (MDS)

![Image 16: Refer to caption](https://arxiv.org/html/2604.19440v1/img/best_fitness_compare.png)

(b) Best Fitness Progression

![Image 17: Refer to caption](https://arxiv.org/html/2604.19440v1/img/h_spatial_curve.png)

(c) Spatial Entropy

![Image 18: Refer to caption](https://arxiv.org/html/2604.19440v1/img/h_fitness_curve.png)

(d) Fitness Spatial Entropy

Figure 5: A qualitative contrast of evolutionary search geometry analysis. (a) Visualization of the search space topology using MDS. Gemini-1.5-pro forms a convergent solution cluster (yellow). All points are projected using a shared MDS space learned from all task-specific candidates. (b) The Mean Best Fitness curve shows the convergence speed and quality over seeds. (c) Spatial Entropy quantifies the candidates’ organization. (d) Fitness-Spatial Entropy illustrates Gemini’s solutions are high-quality and topologically concentrated.

We embed all candidate solutions into a task-specific shared semantic space, enabling us to analyze the within-generation distribution of candidates. Precisely, we measure the spatial organization of search using kernel-based entropy. Let $x_{i} \in \mathbb{R}^{d}$ denote the embedding of solution $i$, and $K ​ \left(\right. \cdot , \cdot \left.\right)$ a Gaussian kernel. For any weighting $w_{j}$, we compute a local density estimate

$g_{i} = \underset{j}{\sum} w_{j} ​ K ​ \left(\right. x_{i} , x_{j} \left.\right) , q_{i} = \frac{g_{i}}{\sum_{k} g_{k}} ,$

and define

$H = - \underset{i}{\sum} q_{i} ​ log ⁡ q_{i} .$

This framework yields two complementary views. Setting $w_{j} = 1$ gives (i) spatial entropy ($H_{\text{spatial}}$), which measures how broadly candidates spread across semantic space. Setting $w_{j} = f_{j}$ gives (ii) fitness spatial entropy ($H_{\text{fitness}}$), which measures whether high-quality solutions cluster or distribute across regions. These metrics intuitively summarize how solutions are spatially organized within a generation as they distinguish between diffuse versus localized search globally ($H_{\text{spatial}}$), and whether high-fitness solutions concentrates or spreads across the semantic landscape ($H_{\text{fitness}}$). We also complement them with multidimensional scaling visualizations that project all solutions onto a shared two-dimensional space, with points colored by generation and scaled by fitness (All MDS plots are on our website 2 2 2[https://xinhao-zhang.github.io/traj_evo_search/](https://xinhao-zhang.github.io/traj_evo_search/)). Figure[5](https://arxiv.org/html/2604.19440#S4.F5 "Figure 5 ‣ 4.2.2 Semantic Geometry ‣ 4.2 Trajectory-Level Analysis ‣ 4 Results and Analysis ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search") illustrates a representative case. Despite similar zero-shot performance and identical initial populations, Gemini-1.5-Pro progressively localizes its search into a smaller semantic region, while Mistral-7B-Instruct continues to drift across distant regions.

#### 4.2.3 Generation-Level Statistical Test

From the preceding case study in Figure[5](https://arxiv.org/html/2604.19440#S4.F5 "Figure 5 ‣ 4.2.2 Semantic Geometry ‣ 4.2 Trajectory-Level Analysis ‣ 4 Results and Analysis ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"), we hypothesize that the optimization success may specifically depend on the geometric properties of the population. We then use a generation-level mixed-effects regression analysis to examine which effects influence breakthrough events’ production.

We model breakthrough probability at generation $t$ as a function of population-level descriptors, including spatial entropy ($H_{\text{spatial}}$), fitness spatial entropy ($H_{\text{fitness}}$), mean and maximum novelty, generation index and their interaction for each generation. To account for repeated measurements and systematic differences across LLMs, we include model-specific random intercepts. Results are presented in Figure[6](https://arxiv.org/html/2604.19440#S4.F6 "Figure 6 ‣ 4.2.3 Generation-Level Statistical Test ‣ 4.2 Trajectory-Level Analysis ‣ 4 Results and Analysis ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"), reporting both concurrent effects (generation $t$) and lagged effects (predicting $t + 1$). There are several consistent patterns. First, the generation index is strongly negative in both specifications, indicating that breakthroughs occur mostly in early generations. Second, higher fitness spatial entropy is negatively associated with breakthrough probability, which counter-intuitively suggests that maintaining multiple dispersed high-quality regions would hinder breakthrough production. Third, although mean novelty is positively associated with breakthroughs within a generation, this effect is strongly conditioned on population geometry: the interaction between novelty and spatial entropy is significantly negative in both concurrent and lagged analyses. In other words, novelty increases the likelihood of breakthroughs only when search remains sufficiently localized. Crucially, while the effect of novelty fades under lagged prediction, the interaction effect remains significant, indicating that the productivity of novelty depends on the geometric state of the population rather than on contemporaneous correlations alone. Figure[12](https://arxiv.org/html/2604.19440#A6.F12 "Figure 12 ‣ Appendix F Supplementary Visualizations ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search") further visualizes this interaction. Breakthrough events concentrate in regions featured by high novelty and low spatial entropy, whereas high novelty under high dispersion is usually related with low breakthrough probability.

![Image 19: Refer to caption](https://arxiv.org/html/2604.19440v1/img/mixed_effects_regression.png)

Figure 6: Generation-level mixed-effects regression of breakthrough probabilities. Standardized coefficients are shown for concurrent (left) and lagged (right) models, with predictors at generation $t$ explaining breakthroughs at $t$ or $t + 1$. $\_{}^{* \llbracket * *}p < 0.001$, $\_{}^{ * *}p < 0.01$, numeric labels report non-significant p-values.

### 4.3 Operator-Level Validation

The trajectory-level analyses above characterize _what_ successful optimization runs look like: strong models progressively localize in semantic space and generate sustained best-so-far improvements, whereas weaker models exhibit semantic drift and stagnation. However, at operator-level, this suggests that beyond base capability, effective LLM optimizers behave as local refiners: they frequently produce offspring that strictly improve upon their prompted parents while maintaining a controlled semantic step sizes. We then validate this hypothesis through two studies below.

#### 4.3.1 Model-Level Regression

We first conduct a fine-grained regression at model level. We employ two operator-level metrics defined at the parent$\rightarrow$child mutation step. First, the local refinement rate (LRR) represents the frequency of strict improvements of the offspring over prompted parents at the fraction of valid offspring attempts. Second, the parent–child distance (PCD) quantifies the average semantic distance between each offspring and its prompted parents in the same task-specific semantic space.

Table[1](https://arxiv.org/html/2604.19440#S4.T1 "Table 1 ‣ 4.3.1 Model-Level Regression ‣ 4.3 Operator-Level Validation ‣ 4 Results and Analysis ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search") reveals that when considered alone, larger semantic step sizes (PCD) are negatively correlated with final performance. However, this effect vanishes once LRR is included, while LRR remains strongly positive and highly significant. This is also consistent with our interaction finding: larger edits tend to reduce the probability of producing refinements. In other words, the negative effect of large semantic modifications is largely explained by their impact on refinement reliability.

This operator-level regression once again highlights that good LLM optimizers act as local refiners, where performance is governed by the ability to produce reliable incremental improvements rather than by the magnitude of semantic variation.

ZS + PCD ZS + LRR + PCD
Zero-shot Perf. (z)$0.233^{*}$$0.144$
(0.028)(0.112)
Avg. Parent–Child Distance (z)$- 0.329^{ * *}$$- 0.024$
(0.001)(0.838)
Avg. Local Refinement Rate (z)$0.528^{* \llbracket * *}$
($<$0.001)
$R^{2}$0.204 0.367

Table 1: Model–task OLS regressions predicting best final performance (z-score), with task fixed effects and model-clustered standard errors. p-values in parentheses. $\_{}^{* \llbracket * *}p < 0.001$, $\_{}^{ * *}p < 0.01$,$\_{}^{*}p < 0.05$. Cells are shaded by coefficient magnitude.

#### 4.3.2 Perturbation Study: Model Mixing

To provide interventional evidence of the role of local refinement behavior, we further perform a perturbation study through model mixing experiments.

At each generation, a fraction of offspring are generated by an alternative model (weak refiner), while the remaining offspring are produced by the primary model (strong refiner). This intervention directly manipulate the refinement behavior of the search process. We construct task-specific model pairs with comparable zero-shot performance but contrasting refinement capabilities (see Appendix Figure[10](https://arxiv.org/html/2604.19440#A4.F10 "Figure 10 ‣ D.2 Perturbation Study Details ‣ Appendix D Robustness Analyses ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search")). We evaluate this intervention across three representative sub-tasks across different regimes: TSP-60, bin packing-OR3, and Prompt Optimization-Summarization.

As shown in Figure[7](https://arxiv.org/html/2604.19440#S4.F7 "Figure 7 ‣ 4.3.2 Perturbation Study: Model Mixing ‣ 4.3 Operator-Level Validation ‣ 4 Results and Analysis ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"), as the proportion of weak-refiner offspring increases, in particular, on TSP-60 and bin packing tasks, performance degrades sharply and monotonically. The same effect exists yet appears to be a bit weaker and less consistent in prompt optimization. Moreover, higher weaker offspring’s ratios consistently reduce the overall refinement rate, which co-varies with the observed degradation in performance. These results suggest that weakening refinement behavior, by injecting lower-refinement operators, could impair the system’s ability to produce sustained improvements, therefore leading to worse optimization outcomes.

![Image 20: Refer to caption](https://arxiv.org/html/2604.19440v1/img/model_mixing_dual_axis.png)

Figure 7: Effect of model mixing on optimization performance and refinement rate. A fraction of offspring is generated by a weaker refiner. Solid lines denote fitness; dashed lines denote refinement rate.

### 4.4 Cost-Efficiency Implication

Our findings also carry practical implications for cost-sensitive deployment. Since optimization performance is not fully determined by base model capability, strong optimizers are not necessarily the most expensive models. We thus estimate the monetary cost of evolutionary optimization for each model based on the average number of input and output tokens per run, using API pricing in OpenRouter platform 3 3 3[https://openrouter.ai](https://openrouter.ai/). Optimization efficacy is measured as the fitness gain achieved over evolution. Figure[8](https://arxiv.org/html/2604.19440#S4.F8 "Figure 8 ‣ 4.4 Cost-Efficiency Implication ‣ 4 Results and Analysis ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search") situates all models in a cost-improvement space, aggregated across all tasks (See Figure[15](https://arxiv.org/html/2604.19440#A6.F15 "Figure 15 ‣ Appendix F Supplementary Visualizations ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search")).

This aggregated view reveals large variation in cost–performance trade-offs. Notably, some mid-sized models achieve large fitness improvements at relatively low cost, whereas stronger zero-shot models do not always yield proportional gains per dollar. For example, Mistral-24B-Instruct lies on the Pareto frontier, combining large fitness improvement with moderate cost, and thus represents an efficient optimization operator rather than merely a strong base model. Overall, this result reinforces our central claim: effective evolutionary optimization depends more on how a model refines solutions over time than on its raw problem-solving capability. For practitioners, this can help build cost-efficient evolutionary systems by selecting models with favorable optimization behavior, instead of defaulting to the most powerful LLM.

![Image 21: Refer to caption](https://arxiv.org/html/2604.19440v1/img/cost_efficecny_analysis.png)

Figure 8: Optimization gain versus cost across LLMs. Each point represents a model, plotting average fitness improvement achieved through evolution against estimated monetary cost.

## 5 Discussion and Conclusion

In this work, we examined the exploration–exploitation trade-off in LLM-guided evolutionary search to understand why some models act as substantially better search operators than others. Although zero-shot task performance correlates with final optimization outcomes, it explains only part of the variance: models with similar zero-shot capability can induce markedly different optimization trajectories and final fitness.

Compared to classical evolutionary algorithms relying on stochastic mutation/crossover and selection to balance exploration and exploitation, LLM-guided evolution alters this paradigm. The mutation operator is no longer random but instantiated by a learned generative prior that induces structured, semantically meaningful variations, thereby strongly biasing the search toward exploitation.

A natural hypothesis is that this lack of randomness makes novelty or diversity a bottleneck, such that increased exploration should improve performance. Our results contradict this view. Higher novelty is not systematically associated with better outcomes and often signals failure: ineffective operators drift across semantic space without refining promising solutions. In contrast, strong LLM operators behave as effective local refiners. Their trajectories progressively localize around high-performing regions, with LLMs producing frequent, incremental improvements. In this regime, novelty is beneficial only when deviations occur within already promising regions. Our perturbation experiments further validate that directly degrading refinement behavior through model mixing brings about drops in optimization performance. These results also offer a refined interpretation of the role of novelty in LLM-guided search. Rather than being a stochastic explorer, novelty acts as an immediate driver of exploratory breakthroughs, but more importantly its long-term utility depends on whether the search regime allows these deviations to be selectively retained and amplified.

However, the observed local refinement behavior should not be viewed as an inherent capability of the base model alone. Instead, it emerges as a property of the entire agentic system that generates offspring, including the model, the prompting strategy, and the decoding configuration. While changes in temperature affect both refinement rates and performance, the relationship between refinement behavior and performance remains stable across a range of settings (see Appendix[D.1](https://arxiv.org/html/2604.19440#A4.SS1 "D.1 Temperature-Sensitivity Experiment ‣ Appendix D Robustness Analyses ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search")).

Our findings have direct implications for the design of LLM-based optimization systems. Rather than focusing solely on maximizing base model capability, our results suggest that the key objective is to _control and optimize refinement behavior_. First, stronger base models do not necessarily yield better search operators, smaller or cheaper LLMs can outperform larger ones when their inductive biases favor stable local refinement. This highlights the importance of model selection. Besides, refinement behavior can potentially be modulated via system-level design choices, including prompting and decoding hyperparameter. More broadly, our results support training or fine-tuning LLMs as effective operators(Brahmachary et al., [2024](https://arxiv.org/html/2604.19440#bib.bib14 "Large language model-based evolutionary optimizer: reasoning with elitism"); Šurina et al., [2025](https://arxiv.org/html/2604.19440#bib.bib23 "Algorithm discovery with LLMs: evolutionary search meets reinforcement learning")), emphasizing local refinement and error correction rather than general-purpose capability. Understanding and shaping such operator-specific behaviors is a promising avenue for boosting LLM-guided optimization.

Finally, the geometric analysis framework we developed to study LLM-guided optimization trajectories is broadly applicable and can be repurposed to analyze other types of iterative search or agentic behaviors. For illustration, a rich, interactive collection of visualizations showing trajectories in semantic space across models and tasks can be found on our project website.

## Limitations

Our study is subject to several limitations. First, while we conduct robustness analyses on decoding hyperparameters (notably temperature; see Appendix[D.1](https://arxiv.org/html/2604.19440#A4.SS1 "D.1 Temperature-Sensitivity Experiment ‣ Appendix D Robustness Analyses ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search")), our experiments still rely on a fixed evolutionary protocol. Other design choices, such as selection pressure, offspring size, and alternative sampling strategies, may influence the balance between exploration and exploitation, and could further shape cross-model differences. Second, in our study novelty is primarily operationalized as nearest-neighbor distance. Broader comparisons to KNN/average-distance novelty and alternative diversity indices would help assess robustness. Third, although we include perturbation experiments via model mixing, the intervention is still hard to fully isolate local refinement in a strictly controlled manner. Replacing the model that generates offspring may also affect other latent and invisible characteristics (e.g., reasoning patterns or exploration tendencies), thus making it difficult to attribute all performance differences solely to local refinement.

## Acknowledgments

This work was supported by ANR (grant ANR-22-CPJ2-0036-01). It was also partially supported by ANR through the MIAI "AI & Language" chair (ANR-23-IACL-0006).

## References

*   LLEMA: evolutionary search with llms for multi-objective materials discovery. External Links: 2510.22503, [Link](https://arxiv.org/abs/2510.22503)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   L. A. Agrawal, S. Tan, D. Soylu, N. Ziems, R. Khare, K. Opsahl-Ong, A. Singhvi, H. Shandilya, M. J. Ryan, M. Jiang, C. Potts, K. Sen, A. G. Dimakis, I. Stoica, D. Klein, M. Zaharia, and O. Khattab (2026)GEPA: reflective prompt evolution can outperform reinforcement learning. External Links: 2507.19457, [Link](https://arxiv.org/abs/2507.19457)Cited by: [§1](https://arxiv.org/html/2604.19440#S1.p1.1 "1 Introduction ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   F. Alva-Manchego, L. Martin, A. Bordes, C. Scarton, B. Sagot, and L. Specia (2020)ASSET: A dataset for tuning and evaluation of sentence simplification models with multiple rewriting transformations. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault (Eds.), Online,  pp.4668–4679. External Links: [Link](https://aclanthology.org/2020.acl-main.424/), [Document](https://dx.doi.org/10.18653/v1/2020.acl-main.424)Cited by: [§3.1](https://arxiv.org/html/2604.19440#S3.SS1.p3.1 "3.1 Tasks & Genome Representations ‣ 3 Methodology ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   S. Brahmachary, S. M. Joshi, A. Panda, K. Koneripalli, A. K. Sagotra, H. Patel, A. Sharma, A. D. Jagtap, and K. Kalyanaraman (2024)Large language model-based evolutionary optimizer: reasoning with elitism. External Links: 2403.02054, [Link](https://arxiv.org/abs/2403.02054)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"), [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px2.p1.1 "Evaluating LLMs as Search Operators ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"), [§5](https://arxiv.org/html/2604.19440#S5.p5.1 "5 Discussion and Conclusion ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   X. Chen, R. Wu, Y. Lan, T. Ma, and Y. Liu (2026)MolEvolve: llm-guided evolutionary search for interpretable molecular optimization. External Links: 2603.24382, [Link](https://arxiv.org/abs/2603.24382)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   DeepSeek-AI, A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, D. Dai, D. Guo, D. Yang, D. Chen, D. Ji, E. Li, F. Lin, F. Dai, F. Luo, G. Hao, G. Chen, G. Li, H. Zhang, H. Bao, H. Xu, H. Wang, H. Zhang, H. Ding, H. Xin, H. Gao, H. Li, H. Qu, J. L. Cai, J. Liang, J. Guo, J. Ni, J. Li, J. Wang, J. Chen, J. Chen, J. Yuan, J. Qiu, J. Li, J. Song, K. Dong, K. Hu, K. Gao, K. Guan, K. Huang, K. Yu, L. Wang, L. Zhang, L. Xu, L. Xia, L. Zhao, L. Wang, L. Zhang, M. Li, M. Wang, M. Zhang, M. Zhang, M. Tang, M. Li, N. Tian, P. Huang, P. Wang, P. Zhang, Q. Wang, Q. Zhu, Q. Chen, Q. Du, R. J. Chen, R. L. Jin, R. Ge, R. Zhang, R. Pan, R. Wang, R. Xu, R. Zhang, R. Chen, S. S. Li, S. Lu, S. Zhou, S. Chen, S. Wu, S. Ye, S. Ye, S. Ma, S. Wang, S. Zhou, S. Yu, S. Zhou, S. Pan, T. Wang, T. Yun, T. Pei, T. Sun, W. L. Xiao, W. Zeng, W. Zhao, W. An, W. Liu, W. Liang, W. Gao, W. Yu, W. Zhang, X. Q. Li, X. Jin, X. Wang, X. Bi, X. Liu, X. Wang, X. Shen, X. Chen, X. Zhang, X. Chen, X. Nie, X. Sun, X. Wang, X. Cheng, X. Liu, X. Xie, X. Liu, X. Yu, X. Song, X. Shan, X. Zhou, X. Yang, X. Li, X. Su, X. Lin, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. X. Zhu, Y. Zhang, Y. Xu, Y. Xu, Y. Huang, Y. Li, Y. Zhao, Y. Sun, Y. Li, Y. Wang, Y. Yu, Y. Zheng, Y. Zhang, Y. Shi, Y. Xiong, Y. He, Y. Tang, Y. Piao, Y. Wang, Y. Tan, Y. Ma, Y. Liu, Y. Guo, Y. Wu, Y. Ou, Y. Zhu, Y. Wang, Y. Gong, Y. Zou, Y. He, Y. Zha, Y. Xiong, Y. Ma, Y. Yan, Y. Luo, Y. You, Y. Liu, Y. Zhou, Z. F. Wu, Z. Z. Ren, Z. Ren, Z. Sha, Z. Fu, Z. Xu, Z. Huang, Z. Zhang, Z. Xie, Z. Zhang, Z. Hao, Z. Gou, Z. Ma, Z. Yan, Z. Shao, Z. Xu, Z. Wu, Z. Zhang, Z. Li, Z. Gu, Z. Zhu, Z. Liu, Z. Li, Z. Xie, Z. Song, Z. Gao, and Z. Pan (2025)DeepSeek-v3 technical report. External Links: 2412.19437, [Link](https://arxiv.org/abs/2412.19437)Cited by: [§3.3](https://arxiv.org/html/2604.19440#S3.SS3.p2.6 "3.3 Evolution Scale and Parameters ‣ 3 Methodology ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   A. Duchnowski, E. Pavlick, and A. Koller (2025)A knapsack by any other name: presentation impacts LLM performance on NP-hard problems. In Findings of the Association for Computational Linguistics: EMNLP 2025, C. Christodoulopoulos, T. Chakraborty, C. Rose, and V. Peng (Eds.), Suzhou, China,  pp.6628–6651. External Links: [Link](https://aclanthology.org/2025.findings-emnlp.352/), [Document](https://dx.doi.org/10.18653/v1/2025.findings-emnlp.352), ISBN 979-8-89176-335-7 Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px2.p1.1 "Evaluating LLMs as Search Operators ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   J. S. Ellenberg, C. S. Fraser-Taliente, T. R. Harvey, K. Srivastava, and A. V. Sutherland (2025)Generative modeling for mathematical discovery. External Links: 2503.11061, [Link](https://arxiv.org/abs/2503.11061)Cited by: [§1](https://arxiv.org/html/2604.19440#S1.p1.1 "1 Introduction ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   L. Fan, W. Hua, L. Li, H. Ling, and Y. Zhang (2024)NPHardEval: dynamic benchmark on reasoning ability of large language models via complexity classes. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), L. Ku, A. Martins, and V. Srikumar (Eds.), Bangkok, Thailand,  pp.4092–4114. External Links: [Link](https://aclanthology.org/2024.acl-long.225/), [Document](https://dx.doi.org/10.18653/v1/2024.acl-long.225)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px2.p1.1 "Evaluating LLMs as Search Operators ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"), [§3.1](https://arxiv.org/html/2604.19440#S3.SS1.p2.1 "3.1 Tasks & Genome Representations ‣ 3 Methodology ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   J. Fang, Y. Peng, X. Zhang, Y. Wang, X. Yi, G. Zhang, Y. Xu, B. Wu, S. Liu, Z. Li, Z. Ren, N. Aletras, X. Wang, H. Zhou, and Z. Meng (2025)A comprehensive survey of self-evolving ai agents: a new paradigm bridging foundation models and lifelong agentic systems. External Links: 2508.07407, [Link](https://arxiv.org/abs/2508.07407)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   C. Fernando, D. Banarse, H. Michalewski, S. Osindero, and T. Rocktäschel (2023)Promptbreeder: self-referential self-improvement via prompt evolution. External Links: 2309.16797, [Link](https://arxiv.org/abs/2309.16797)Cited by: [§1](https://arxiv.org/html/2604.19440#S1.p1.1 "1 Introduction ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"), [§3.1](https://arxiv.org/html/2604.19440#S3.SS1.p3.1 "3.1 Tasks & Genome Representations ‣ 3 Methodology ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   H. Gao, J. Geng, W. Hua, M. Hu, X. Juan, H. Liu, S. Liu, J. Qiu, X. Qi, Y. Wu, H. Wang, H. Xiao, Y. Zhou, S. Zhang, J. Zhang, J. Xiang, Y. Fang, Q. Zhao, D. Liu, Q. Ren, C. Qian, Z. Wang, M. Hu, H. Wang, Q. Wu, H. Ji, and M. Wang (2026)A survey of self-evolving agents: what, when, how, and where to evolve on the path to artificial super intelligence. External Links: 2507.21046, [Link](https://arxiv.org/abs/2507.21046)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   B. Gliwa, I. Mochol, M. Biesek, and A. Wawer (2019)SAMSum corpus: a human-annotated dialogue dataset for abstractive summarization. In Proceedings of the 2nd Workshop on New Frontiers in Summarization, L. Wang, J. C. K. Cheung, G. Carenini, and F. Liu (Eds.), Hong Kong, China,  pp.70–79. External Links: [Link](https://aclanthology.org/D19-5409/), [Document](https://dx.doi.org/10.18653/v1/D19-5409)Cited by: [§3.1](https://arxiv.org/html/2604.19440#S3.SS1.p3.1 "3.1 Tasks & Genome Representations ‣ 3 Methodology ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   J. Gottweis, W. Weng, A. Daryin, T. Tu, A. Palepu, P. Sirkovic, A. Myaskovsky, F. Weissenberger, K. Rong, R. Tanno, K. Saab, D. Popovici, J. Blum, F. Zhang, K. Chou, A. Hassidim, B. Gokturk, A. Vahdat, P. Kohli, Y. Matias, A. Carroll, K. Kulkarni, N. Tomasev, Y. Guan, V. Dhillon, E. D. Vaishnav, B. Lee, T. R. D. Costa, J. R. Penadés, G. Peltz, Y. Xu, A. Pawlosky, A. Karthikesalingam, and V. Natarajan (2025)Towards an ai co-scientist. External Links: 2502.18864, [Link](https://arxiv.org/abs/2502.18864)Cited by: [§1](https://arxiv.org/html/2604.19440#S1.p1.1 "1 Introduction ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"), [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, A. Yang, A. Fan, A. Goyal, A. Hartshorn, A. Yang, A. Mitra, A. Sravankumar, A. Korenev, A. Hinsvark, A. Rao, A. Zhang, A. Rodriguez, A. Gregerson, A. Spataru, B. Roziere, B. Biron, B. Tang, B. Chern, C. Caucheteux, C. Nayak, C. Bi, C. Marra, C. McConnell, C. Keller, C. Touret, C. Wu, C. Wong, C. C. Ferrer, C. Nikolaidis, D. Allonsius, D. Song, D. Pintz, D. Livshits, D. Wyatt, D. Esiobu, D. Choudhary, D. Mahajan, D. Garcia-Olano, D. Perino, D. Hupkes, E. Lakomkin, E. AlBadawy, E. Lobanova, E. Dinan, E. M. Smith, F. Radenovic, F. Guzmán, F. Zhang, G. Synnaeve, G. Lee, G. L. Anderson, G. Thattai, G. Nail, G. Mialon, G. Pang, G. Cucurell, H. Nguyen, H. Korevaar, H. Xu, H. Touvron, I. Zarov, I. A. Ibarra, I. Kloumann, I. Misra, I. Evtimov, J. Zhang, J. Copet, J. Lee, J. Geffert, J. Vranes, J. Park, J. Mahadeokar, J. Shah, J. van der Linde, J. Billock, J. Hong, J. Lee, J. Fu, J. Chi, J. Huang, J. Liu, J. Wang, J. Yu, J. Bitton, J. Spisak, J. Park, J. Rocca, J. Johnstun, J. Saxe, J. Jia, K. V. Alwala, K. Prasad, K. Upasani, K. Plawiak, K. Li, K. Heafield, K. Stone, K. El-Arini, K. Iyer, K. Malik, K. Chiu, K. Bhalla, K. Lakhotia, L. Rantala-Yeary, L. van der Maaten, L. Chen, L. Tan, L. Jenkins, L. Martin, L. Madaan, L. Malo, L. Blecher, L. Landzaat, L. de Oliveira, M. Muzzi, M. Pasupuleti, M. Singh, M. Paluri, M. Kardas, M. Tsimpoukelli, M. Oldham, M. Rita, M. Pavlova, M. Kambadur, M. Lewis, M. Si, M. K. Singh, M. Hassan, N. Goyal, N. Torabi, N. Bashlykov, N. Bogoychev, N. Chatterji, N. Zhang, O. Duchenne, O. Çelebi, P. Alrassy, P. Zhang, P. Li, P. Vasic, P. Weng, P. Bhargava, P. Dubal, P. Krishnan, P. S. Koura, P. Xu, Q. He, Q. Dong, R. Srinivasan, R. Ganapathy, R. Calderer, R. S. Cabral, R. Stojnic, R. Raileanu, R. Maheswari, R. Girdhar, R. Patel, R. Sauvestre, R. Polidoro, R. Sumbaly, R. Taylor, R. Silva, R. Hou, R. Wang, S. Hosseini, S. Chennabasappa, S. Singh, S. Bell, S. S. Kim, S. Edunov, S. Nie, S. Narang, S. Raparthy, S. Shen, S. Wan, S. Bhosale, S. Zhang, S. Vandenhende, S. Batra, S. Whitman, S. Sootla, S. Collot, S. Gururangan, S. Borodinsky, T. Herman, T. Fowler, T. Sheasha, T. Georgiou, T. Scialom, T. Speckbacher, T. Mihaylov, T. Xiao, U. Karn, V. Goswami, V. Gupta, V. Ramanathan, V. Kerkez, V. Gonguet, V. Do, V. Vogeti, V. Albiero, V. Petrovic, W. Chu, W. Xiong, W. Fu, W. Meers, X. Martinet, X. Wang, X. Wang, X. E. Tan, X. Xia, X. Xie, X. Jia, X. Wang, Y. Goldschlag, Y. Gaur, Y. Babaei, Y. Wen, Y. Song, Y. Zhang, Y. Li, Y. Mao, Z. D. Coudert, Z. Yan, Z. Chen, Z. Papakipos, A. Singh, A. Srivastava, A. Jain, A. Kelsey, A. Shajnfeld, A. Gangidi, A. Victoria, A. Goldstand, A. Menon, A. Sharma, A. Boesenberg, A. Baevski, A. Feinstein, A. Kallet, A. Sangani, A. Teo, A. Yunus, A. Lupu, A. Alvarado, A. Caples, A. Gu, A. Ho, A. Poulton, A. Ryan, A. Ramchandani, A. Dong, A. Franco, A. Goyal, A. Saraf, A. Chowdhury, A. Gabriel, A. Bharambe, A. Eisenman, A. Yazdan, B. James, B. Maurer, B. Leonhardi, B. Huang, B. Loyd, B. D. Paola, B. Paranjape, B. Liu, B. Wu, B. Ni, B. Hancock, B. Wasti, B. Spence, B. Stojkovic, B. Gamido, B. Montalvo, C. Parker, C. Burton, C. Mejia, C. Liu, C. Wang, C. Kim, C. Zhou, C. Hu, C. Chu, C. Cai, C. Tindal, C. Feichtenhofer, C. Gao, D. Civin, D. Beaty, D. Kreymer, D. Li, D. Adkins, D. Xu, D. Testuggine, D. David, D. Parikh, D. Liskovich, D. Foss, D. Wang, D. Le, D. Holland, E. Dowling, E. Jamil, E. Montgomery, E. Presani, E. Hahn, E. Wood, E. Le, E. Brinkman, E. Arcaute, E. Dunbar, E. Smothers, F. Sun, F. Kreuk, F. Tian, F. Kokkinos, F. Ozgenel, F. Caggioni, F. Kanayet, F. Seide, G. M. Florez, G. Schwarz, G. Badeer, G. Swee, G. Halpern, G. Herman, G. Sizov, Guangyi, Zhang, G. Lakshminarayanan, H. Inan, H. Shojanazeri, H. Zou, H. Wang, H. Zha, H. Habeeb, H. Rudolph, H. Suk, H. Aspegren, H. Goldman, H. Zhan, I. Damlaj, I. Molybog, I. Tufanov, I. Leontiadis, I. Veliche, I. Gat, J. Weissman, J. Geboski, J. Kohli, J. Lam, J. Asher, J. Gaya, J. Marcus, J. Tang, J. Chan, J. Zhen, J. Reizenstein, J. Teboul, J. Zhong, J. Jin, J. Yang, J. Cummings, J. Carvill, J. Shepard, J. McPhie, J. Torres, J. Ginsburg, J. Wang, K. Wu, K. H. U, K. Saxena, K. Khandelwal, K. Zand, K. Matosich, K. Veeraraghavan, K. Michelena, K. Li, K. Jagadeesh, K. Huang, K. Chawla, K. Huang, L. Chen, L. Garg, L. A, L. Silva, L. Bell, L. Zhang, L. Guo, L. Yu, L. Moshkovich, L. Wehrstedt, M. Khabsa, M. Avalani, M. Bhatt, M. Mankus, M. Hasson, M. Lennie, M. Reso, M. Groshev, M. Naumov, M. Lathi, M. Keneally, M. Liu, M. L. Seltzer, M. Valko, M. Restrepo, M. Patel, M. Vyatskov, M. Samvelyan, M. Clark, M. Macey, M. Wang, M. J. Hermoso, M. Metanat, M. Rastegari, M. Bansal, N. Santhanam, N. Parks, N. White, N. Bawa, N. Singhal, N. Egebo, N. Usunier, N. Mehta, N. P. Laptev, N. Dong, N. Cheng, O. Chernoguz, O. Hart, O. Salpekar, O. Kalinli, P. Kent, P. Parekh, P. Saab, P. Balaji, P. Rittner, P. Bontrager, P. Roux, P. Dollar, P. Zvyagina, P. Ratanchandani, P. Yuvraj, Q. Liang, R. Alao, R. Rodriguez, R. Ayub, R. Murthy, R. Nayani, R. Mitra, R. Parthasarathy, R. Li, R. Hogan, R. Battey, R. Wang, R. Howes, R. Rinott, S. Mehta, S. Siby, S. J. Bondu, S. Datta, S. Chugh, S. Hunt, S. Dhillon, S. Sidorov, S. Pan, S. Mahajan, S. Verma, S. Yamamoto, S. Ramaswamy, S. Lindsay, S. Lindsay, S. Feng, S. Lin, S. C. Zha, S. Patil, S. Shankar, S. Zhang, S. Zhang, S. Wang, S. Agarwal, S. Sajuyigbe, S. Chintala, S. Max, S. Chen, S. Kehoe, S. Satterfield, S. Govindaprasad, S. Gupta, S. Deng, S. Cho, S. Virk, S. Subramanian, S. Choudhury, S. Goldman, T. Remez, T. Glaser, T. Best, T. Koehler, T. Robinson, T. Li, T. Zhang, T. Matthews, T. Chou, T. Shaked, V. Vontimitta, V. Ajayi, V. Montanez, V. Mohan, V. S. Kumar, V. Mangla, V. Ionescu, V. Poenaru, V. T. Mihailescu, V. Ivanov, W. Li, W. Wang, W. Jiang, W. Bouaziz, W. Constable, X. Tang, X. Wu, X. Wang, X. Wu, X. Gao, Y. Kleinman, Y. Chen, Y. Hu, Y. Jia, Y. Qi, Y. Li, Y. Zhang, Y. Zhang, Y. Adi, Y. Nam, Yu, Wang, Y. Zhao, Y. Hao, Y. Qian, Y. Li, Y. He, Z. Rait, Z. DeVito, Z. Rosnbrick, Z. Wen, Z. Yang, Z. Zhao, and Z. Ma (2024)The llama 3 herd of models. External Links: 2407.21783, [Link](https://arxiv.org/abs/2407.21783)Cited by: [§3.3](https://arxiv.org/html/2604.19440#S3.SS3.p2.6 "3.3 Evolution Scale and Parameters ‣ 3 Methodology ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   A. Grayeli, A. Sehgal, O. Costilla-Reyes, M. Cranmer, and S. Chaudhuri (2024)Symbolic regression with a learned concept library. In Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (Eds.), Vol. 37,  pp.44678–44709. External Links: [Document](https://dx.doi.org/10.52202/079017-1419), [Link](https://proceedings.neurips.cc/paper_files/paper/2024/file/4ec3ddc465c6d650c9c419fb91f1c00a-Paper-Conference.pdf)Cited by: [§3.1](https://arxiv.org/html/2604.19440#S3.SS1.p4.5 "3.1 Tasks & Genome Representations ‣ 3 Methodology ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   Q. Guo, R. Wang, J. Guo, B. Li, K. Song, X. Tan, G. Liu, J. Bian, and Y. Yang (2025)EvoPrompt: connecting llms with evolutionary algorithms yields powerful prompt optimizers. External Links: 2309.08532, [Link](https://arxiv.org/abs/2309.08532)Cited by: [§1](https://arxiv.org/html/2604.19440#S1.p1.1 "1 Introduction ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"), [§3.1](https://arxiv.org/html/2604.19440#S3.SS1.p3.1 "3.1 Tasks & Genome Representations ‣ 3 Methodology ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   J. H. Holland (1992)Adaptation in natural and artificial systems: an introductory analysis with applications to biology, control, and artificial intelligence. MIT press. External Links: [Link](https://doi.org/10.7551/mitpress/1090.001.0001)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   B. Huang, X. Wu, Y. Zhou, J. Wu, L. Feng, R. Cheng, and K. C. Tan (2024)Exploring the true potential: evaluating the black-box optimization capability of large language models. External Links: 2404.06290, [Link](https://arxiv.org/abs/2404.06290)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px2.p1.1 "Evaluating LLMs as Search Operators ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"), [§3.1](https://arxiv.org/html/2604.19440#S3.SS1.p2.1 "3.1 Tasks & Genome Representations ‣ 3 Methodology ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. de las Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, L. R. Lavaud, M. Lachaux, P. Stock, T. L. Scao, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed (2023)Mistral 7b. External Links: 2310.06825, [Link](https://arxiv.org/abs/2310.06825)Cited by: [§3.3](https://arxiv.org/html/2604.19440#S3.SS3.p2.6 "3.3 Evolution Scale and Parameters ‣ 3 Methodology ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   M. Josifoski, L. Klein, M. Peyrard, N. Baldwin, Y. Li, S. Geng, J. P. Schnitzler, Y. Yao, J. Wei, D. Paul, et al. (2023)Flows: building blocks of reasoning and collaborating ai. arXiv preprint arXiv:2308.01285. Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   H. Lee, S. Park, Y. Choi, S. An, S. Lee, and S. J. Hwang (2026)T-map: red-teaming llm agents with trajectory-aware evolutionary search. External Links: 2603.22341, [Link](https://arxiv.org/abs/2603.22341)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   J. Lehman, J. Gordon, S. Jain, K. Ndousse, C. Yeh, and K. O. Stanley (2022)Evolution through large models. External Links: 2206.08896, [Link](https://arxiv.org/abs/2206.08896)Cited by: [§1](https://arxiv.org/html/2604.19440#S1.p1.1 "1 Introduction ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   X. Li, J. Chen, X. Fang, S. Ding, H. Duan, Q. Liu, and K. Chen (2025)OPT-bench: evaluating llm agent on large-scale search spaces optimization problems. External Links: 2506.10764, [Link](https://arxiv.org/abs/2506.10764)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px2.p1.1 "Evaluating LLMs as Search Operators ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   J. Lin, Y. Guo, Y. Han, S. Hu, Z. Ni, L. Wang, M. Chen, H. Liu, R. Chen, Y. He, D. Jiang, B. Jiao, C. Hu, and H. Wang (2025)SE-agent: self-evolution trajectory optimization in multi-step reasoning with llm-based agents. External Links: 2508.02085, [Link](https://arxiv.org/abs/2508.02085)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   F. Liu, X. Tong, X. L. Mingxuan Yuan, F. Luo, Z. Wang, Z. Lu, and Q. Zhang (2024)Evolution of heuristics: towards efficient automatic algorithm design using large language model. In International Conference on Machine Learning (ICML), External Links: [Link](https://arxiv.org/abs/2401.02051)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"), [§3.1](https://arxiv.org/html/2604.19440#S3.SS1.p5.1 "3.1 Tasks & Genome Representations ‣ 3 Methodology ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   F. Liu, Q. Zhang, J. Shi, X. Tong, K. Mao, and M. Yuan (2025)Fitness landscape of large language model-assisted automated algorithm search. External Links: 2504.19636, [Link](https://arxiv.org/abs/2504.19636)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   C. Lu, C. Lu, R. T. Lange, J. Foerster, J. Clune, and D. Ha (2024)The ai scientist: towards fully automated open-ended scientific discovery. External Links: 2408.06292, [Link](https://arxiv.org/abs/2408.06292)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   R. MacKnight, J. E. Regio, J. G. Ethier, L. A. Baldwin, and G. Gomes (2025)Pre-trained knowledge elevates large language models beyond traditional chemical reaction optimizers. External Links: 2509.00103, [Link](https://arxiv.org/abs/2509.00103)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   Mistral-AI, :, A. Rastogi, A. Q. Jiang, A. Lo, G. Berrada, G. Lample, J. Rute, J. Barmentlo, K. Yadav, K. Khandelwal, K. R. Chandu, L. Blier, L. Saulnier, M. Dinot, M. Darrin, N. Gupta, R. Soletskyi, S. Vaze, T. L. Scao, Y. Wang, A. Yang, A. H. Liu, A. Sablayrolles, A. Héliou, A. Martin, A. Ehrenberg, A. Agarwal, A. Roux, A. Darcet, A. Mensch, B. Bout, B. Rozière, B. D. Monicault, C. Bamford, C. Wallenwein, C. Renaudin, C. Lanfranchi, D. Dabert, D. Mizelle, D. de las Casas, E. Chane-Sane, E. Fugier, E. B. Hanna, G. Delerce, G. Guinet, G. Novikov, G. Martin, H. Jaju, J. Ludziejewski, J. Chabran, J. Delignon, J. Studnia, J. Amar, J. S. Roberts, J. Denize, K. Saxena, K. Jain, L. Zhao, L. Martin, L. Gao, L. R. Lavaud, M. Pellat, M. Guillaumin, M. Felardos, M. Augustin, M. Seznec, N. Raghuraman, O. Duchenne, P. Wang, P. von Platen, P. Saffer, P. Jacob, P. Wambergue, P. Kurylowicz, P. R. Muddireddy, P. Chagniot, P. Stock, P. Agrawal, R. Sauvestre, R. Delacourt, S. Gandhi, S. Subramanian, S. Dalal, S. Gandhi, S. Ghosh, S. Mishra, S. Aithal, S. Antoniak, T. Schueller, T. Lavril, T. Robert, T. Wang, T. Lacroix, V. Nemychnikova, V. Paltz, V. Richard, W. Li, W. Marshall, X. Zhang, and Y. Tang (2025)Magistral. External Links: 2506.10910, [Link](https://arxiv.org/abs/2506.10910)Cited by: [§3.3](https://arxiv.org/html/2604.19440#S3.SS3.p2.6 "3.3 Evolution Scale and Parameters ‣ 3 Methodology ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   M. Mitchell and C. E. Taylor (1999)Evolutionary computation: an overview. Annual Review of Ecology and Systematics,  pp.593–616. External Links: [Link](https://doi.org/10.1146/annurev.ecolsys.30.1.593)Cited by: [§1](https://arxiv.org/html/2604.19440#S1.p2.1 "1 Introduction ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"), [§4.2.1](https://arxiv.org/html/2604.19440#S4.SS2.SSS1.Px2.p2.1 "Breakthrough Rate Strongly Predicts Optimization Performance. ‣ 4.2.1 Novelty vs. Breakthrough Dynamics ‣ 4.2 Trajectory-Level Analysis ‣ 4 Results and Analysis ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   A. Novikov, N. Vũ, M. Eisenberger, E. Dupont, P. Huang, A. Z. Wagner, S. Shirobokov, B. Kozlovskii, F. J. R. Ruiz, A. Mehrabian, M. P. Kumar, A. See, S. Chaudhuri, G. Holland, A. Davies, S. Nowozin, P. Kohli, and M. Balog (2025)AlphaEvolve: a coding agent for scientific and algorithmic discovery. External Links: 2506.13131, [Link](https://arxiv.org/abs/2506.13131)Cited by: [§1](https://arxiv.org/html/2604.19440#S1.p1.1 "1 Introduction ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"), [§3](https://arxiv.org/html/2604.19440#S3.p1.1 "3 Methodology ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   OpenAI, :, A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford, A. Mądry, A. Baker-Whitcomb, A. Beutel, A. Borzunov, A. Carney, A. Chow, A. Kirillov, A. Nichol, A. Paino, A. Renzin, A. T. Passos, A. Kirillov, A. Christakis, A. Conneau, A. Kamali, A. Jabri, A. Moyer, A. Tam, A. Crookes, A. Tootoochian, A. Tootoonchian, A. Kumar, A. Vallone, A. Karpathy, A. Braunstein, A. Cann, A. Codispoti, A. Galu, A. Kondrich, A. Tulloch, A. Mishchenko, A. Baek, A. Jiang, A. Pelisse, A. Woodford, A. Gosalia, A. Dhar, A. Pantuliano, A. Nayak, A. Oliver, B. Zoph, B. Ghorbani, B. Leimberger, B. Rossen, B. Sokolowsky, B. Wang, B. Zweig, B. Hoover, B. Samic, B. McGrew, B. Spero, B. Giertler, B. Cheng, B. Lightcap, B. Walkin, B. Quinn, B. Guarraci, B. Hsu, B. Kellogg, B. Eastman, C. Lugaresi, C. Wainwright, C. Bassin, C. Hudson, C. Chu, C. Nelson, C. Li, C. J. Shern, C. Conger, C. Barette, C. Voss, C. Ding, C. Lu, C. Zhang, C. Beaumont, C. Hallacy, C. Koch, C. Gibson, C. Kim, C. Choi, C. McLeavey, C. Hesse, C. Fischer, C. Winter, C. Czarnecki, C. Jarvis, C. Wei, C. Koumouzelis, D. Sherburn, D. Kappler, D. Levin, D. Levy, D. Carr, D. Farhi, D. Mely, D. Robinson, D. Sasaki, D. Jin, D. Valladares, D. Tsipras, D. Li, D. P. Nguyen, D. Findlay, E. Oiwoh, E. Wong, E. Asdar, E. Proehl, E. Yang, E. Antonow, E. Kramer, E. Peterson, E. Sigler, E. Wallace, E. Brevdo, E. Mays, F. Khorasani, F. P. Such, F. Raso, F. Zhang, F. von Lohmann, F. Sulit, G. Goh, G. Oden, G. Salmon, G. Starace, G. Brockman, H. Salman, H. Bao, H. Hu, H. Wong, H. Wang, H. Schmidt, H. Whitney, H. Jun, H. Kirchner, H. P. de Oliveira Pinto, H. Ren, H. Chang, H. W. Chung, I. Kivlichan, I. O’Connell, I. O’Connell, I. Osband, I. Silber, I. Sohl, I. Okuyucu, I. Lan, I. Kostrikov, I. Sutskever, I. Kanitscheider, I. Gulrajani, J. Coxon, J. Menick, J. Pachocki, J. Aung, J. Betker, J. Crooks, J. Lennon, J. Kiros, J. Leike, J. Park, J. Kwon, J. Phang, J. Teplitz, J. Wei, J. Wolfe, J. Chen, J. Harris, J. Varavva, J. G. Lee, J. Shieh, J. Lin, J. Yu, J. Weng, J. Tang, J. Yu, J. Jang, J. Q. Candela, J. Beutler, J. Landers, J. Parish, J. Heidecke, J. Schulman, J. Lachman, J. McKay, J. Uesato, J. Ward, J. W. Kim, J. Huizinga, J. Sitkin, J. Kraaijeveld, J. Gross, J. Kaplan, J. Snyder, J. Achiam, J. Jiao, J. Lee, J. Zhuang, J. Harriman, K. Fricke, K. Hayashi, K. Singhal, K. Shi, K. Karthik, K. Wood, K. Rimbach, K. Hsu, K. Nguyen, K. Gu-Lemberg, K. Button, K. Liu, K. Howe, K. Muthukumar, K. Luther, L. Ahmad, L. Kai, L. Itow, L. Workman, L. Pathak, L. Chen, L. Jing, L. Guy, L. Fedus, L. Zhou, L. Mamitsuka, L. Weng, L. McCallum, L. Held, L. Ouyang, L. Feuvrier, L. Zhang, L. Kondraciuk, L. Kaiser, L. Hewitt, L. Metz, L. Doshi, M. Aflak, M. Simens, M. Boyd, M. Thompson, M. Dukhan, M. Chen, M. Gray, M. Hudnall, M. Zhang, M. Aljubeh, M. Litwin, M. Zeng, M. Johnson, M. Shetty, M. Gupta, M. Shah, M. Yatbaz, M. J. Yang, M. Zhong, M. Glaese, M. Chen, M. Janner, M. Lampe, M. Petrov, M. Wu, M. Wang, M. Fradin, M. Pokrass, M. Castro, M. O. T. de Castro, M. Pavlov, M. Brundage, M. Wang, M. Khan, M. Murati, M. Bavarian, M. Lin, M. Yesildal, N. Soto, N. Gimelshein, N. Cone, N. Staudacher, N. Summers, N. LaFontaine, N. Chowdhury, N. Ryder, N. Stathas, N. Turley, N. Tezak, N. Felix, N. Kudige, N. Keskar, N. Deutsch, N. Bundick, N. Puckett, O. Nachum, O. Okelola, O. Boiko, O. Murk, O. Jaffe, O. Watkins, O. Godement, O. Campbell-Moore, P. Chao, P. McMillan, P. Belov, P. Su, P. Bak, P. Bakkum, P. Deng, P. Dolan, P. Hoeschele, P. Welinder, P. Tillet, P. Pronin, P. Tillet, P. Dhariwal, Q. Yuan, R. Dias, R. Lim, R. Arora, R. Troll, R. Lin, R. G. Lopes, R. Puri, R. Miyara, R. Leike, R. Gaubert, R. Zamani, R. Wang, R. Donnelly, R. Honsby, R. Smith, R. Sahai, R. Ramchandani, R. Huet, R. Carmichael, R. Zellers, R. Chen, R. Chen, R. Nigmatullin, R. Cheu, S. Jain, S. Altman, S. Schoenholz, S. Toizer, S. Miserendino, S. Agarwal, S. Culver, S. Ethersmith, S. Gray, S. Grove, S. Metzger, S. Hermani, S. Jain, S. Zhao, S. Wu, S. Jomoto, S. Wu, Shuaiqi, Xia, S. Phene, S. Papay, S. Narayanan, S. Coffey, S. Lee, S. Hall, S. Balaji, T. Broda, T. Stramer, T. Xu, T. Gogineni, T. Christianson, T. Sanders, T. Patwardhan, T. Cunninghman, T. Degry, T. Dimson, T. Raoux, T. Shadwell, T. Zheng, T. Underwood, T. Markov, T. Sherbakov, T. Rubin, T. Stasi, T. Kaftan, T. Heywood, T. Peterson, T. Walters, T. Eloundou, V. Qi, V. Moeller, V. Monaco, V. Kuo, V. Fomenko, W. Chang, W. Zheng, W. Zhou, W. Manassra, W. Sheu, W. Zaremba, Y. Patil, Y. Qian, Y. Kim, Y. Cheng, Y. Zhang, Y. He, Y. Zhang, Y. Jin, Y. Dai, and Y. Malkov (2024)GPT-4o system card. External Links: 2410.21276, [Link](https://arxiv.org/abs/2410.21276)Cited by: [§3.3](https://arxiv.org/html/2604.19440#S3.SS3.p2.6 "3.3 Evolution Scale and Parameters ‣ 3 Methodology ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   A. Ouyang, S. Guo, S. Arora, A. L. Zhang, W. Hu, C. Ré, and A. Mirhoseini (2025)KernelBench: can llms write efficient gpu kernels?. External Links: 2502.10517, [Link](https://arxiv.org/abs/2502.10517)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px2.p1.1 "Evaluating LLMs as Search Operators ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   M. Peyrard, M. Josifoski, and R. West (2025)Agentic ai: the era of semantic decoding. External Links: 2403.14562, [Link](https://arxiv.org/abs/2403.14562)Cited by: [§1](https://arxiv.org/html/2604.19440#S1.p1.1 "1 Introduction ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"), [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   A. Qu, H. Zheng, Z. Zhou, Y. Yan, Y. Tang, S. Y. Ong, F. Hong, K. Zhou, C. Jiang, M. Kong, J. Zhu, X. Jiang, S. Li, C. Wu, B. K. H. Low, J. Zhao, and P. P. Liang (2026)CORAL: towards autonomous multi-agent evolution for open-ended discovery. External Links: 2604.01658, [Link](https://arxiv.org/abs/2604.01658)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   B. Romera-Paredes, M. Barekatain, A. Novikov, M. Balog, M. P. Kumar, E. Dupont, F. J. R. Ruiz, J. Ellenberg, P. Wang, O. Fawzi, P. Kohli, and A. Fawzi (2023)Mathematical discoveries from program search with large language models. Nature. External Links: [Document](https://dx.doi.org/10.1038/s41586-023-06924-6), [Link](https://doi.org/10.1038/s41586-023-06924-6)Cited by: [§1](https://arxiv.org/html/2604.19440#S1.p1.1 "1 Introduction ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"), [§3.1](https://arxiv.org/html/2604.19440#S3.SS1.p5.1 "3.1 Tasks & Genome Representations ‣ 3 Methodology ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   P. Shojaee, K. Meidani, S. Gupta, A. B. Farimani, and C. K. Reddy (2025a)LLM-sr: scientific equation discovery via programming with large language models. External Links: 2404.18400, [Link](https://arxiv.org/abs/2404.18400)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px2.p1.1 "Evaluating LLMs as Search Operators ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"), [§3.1](https://arxiv.org/html/2604.19440#S3.SS1.p4.5 "3.1 Tasks & Genome Representations ‣ 3 Methodology ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   P. Shojaee, N. Nguyen, K. Meidani, A. B. Farimani, K. D. Doan, and C. K. Reddy (2025b)LLM-srbench: a new benchmark for scientific equation discovery with large language models. External Links: 2504.10415, [Link](https://arxiv.org/abs/2504.10415)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px2.p1.1 "Evaluating LLMs as Search Operators ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"), [§3.1](https://arxiv.org/html/2604.19440#S3.SS1.p4.5 "3.1 Tasks & Genome Representations ‣ 3 Methodology ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   J. Sun, H. Zhang, Q. Zhang, and H. Chen (2018)Balancing exploration and exploitation in multiobjective evolutionary optimization. In Proceedings of the genetic and evolutionary computation conference companion,  pp.199–200. External Links: [Document](https://dx.doi.org/10.1145/3205651.3205708), [Link](https://doi.org/10.1145/3205651.3205708)Cited by: [§1](https://arxiv.org/html/2604.19440#S1.p2.1 "1 Introduction ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   A. Šurina, A. Mansouri, L. C.P.M. Quaedvlieg, A. Seddas, M. Viazovska, E. Abbe, and C. Gulcehre (2025)Algorithm discovery with LLMs: evolutionary search meets reinforcement learning. In Second Conference on Language Modeling, External Links: [Link](https://openreview.net/forum?id=dNW3RGW0gi)Cited by: [§1](https://arxiv.org/html/2604.19440#S1.p5.1 "1 Introduction ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"), [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px2.p1.1 "Evaluating LLMs as Search Operators ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"), [§5](https://arxiv.org/html/2604.19440#S5.p5.1 "5 Discussion and Conclusion ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   Z. Tao, T. Lin, X. Chen, H. Li, Y. Wu, Y. Li, Z. Jin, F. Huang, D. Tao, and J. Zhou (2024)A survey on self-evolution of large language models. External Links: 2404.14387, [Link](https://arxiv.org/abs/2404.14387)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   G. Team, P. Georgiev, V. I. Lei, R. Burnell, L. Bai, A. Gulati, G. Tanzer, D. Vincent, Z. Pan, S. Wang, S. Mariooryad, Y. Ding, X. Geng, F. Alcober, R. Frostig, M. Omernick, L. Walker, C. Paduraru, C. Sorokin, A. Tacchetti, C. Gaffney, S. Daruki, O. Sercinoglu, Z. Gleicher, J. Love, P. Voigtlaender, R. Jain, G. Surita, K. Mohamed, R. Blevins, J. Ahn, T. Zhu, K. Kawintiranon, O. Firat, Y. Gu, Y. Zhang, M. Rahtz, M. Faruqui, N. Clay, J. Gilmer, J. Co-Reyes, I. Penchev, R. Zhu, N. Morioka, K. Hui, K. Haridasan, V. Campos, M. Mahdieh, M. Guo, S. Hassan, K. Kilgour, A. Vezer, H. Cheng, R. de Liedekerke, S. Goyal, P. Barham, D. Strouse, S. Noury, J. Adler, M. Sundararajan, S. Vikram, D. Lepikhin, M. Paganini, X. Garcia, F. Yang, D. Valter, M. Trebacz, K. Vodrahalli, C. Asawaroengchai, R. Ring, N. Kalb, L. B. Soares, S. Brahma, D. Steiner, T. Yu, F. Mentzer, A. He, L. Gonzalez, B. Xu, R. L. Kaufman, L. E. Shafey, J. Oh, T. Hennigan, G. van den Driessche, S. Odoom, M. Lucic, B. Roelofs, S. Lall, A. Marathe, B. Chan, S. Ontanon, L. He, D. Teplyashin, J. Lai, P. Crone, B. Damoc, L. Ho, S. Riedel, K. Lenc, C. Yeh, A. Chowdhery, Y. Xu, M. Kazemi, E. Amid, A. Petrushkina, K. Swersky, A. Khodaei, G. Chen, C. Larkin, M. Pinto, G. Yan, A. P. Badia, P. Patil, S. Hansen, D. Orr, S. M. R. Arnold, J. Grimstad, A. Dai, S. Douglas, R. Sinha, V. Yadav, X. Chen, E. Gribovskaya, J. Austin, J. Zhao, K. Patel, P. Komarek, S. Austin, S. Borgeaud, L. Friso, A. Goyal, B. Caine, K. Cao, D. Chung, M. Lamm, G. Barth-Maron, T. Kagohara, K. Olszewska, M. Chen, K. Shivakumar, R. Agarwal, H. Godhia, R. Rajwar, J. Snaider, X. Dotiwalla, Y. Liu, A. Barua, V. Ungureanu, Y. Zhang, B. Batsaikhan, M. Wirth, J. Qin, I. Danihelka, T. Doshi, M. Chadwick, J. Chen, S. Jain, Q. Le, A. Kar, M. Gurumurthy, C. Li, R. Sang, F. Liu, L. Lamprou, R. Munoz, N. Lintz, H. Mehta, H. Howard, M. Reynolds, L. Aroyo, Q. Wang, L. Blanco, A. Cassirer, J. Griffith, D. Das, S. Lee, J. Sygnowski, Z. Fisher, J. Besley, R. Powell, Z. Ahmed, D. Paulus, D. Reitter, Z. Borsos, R. Joshi, A. Pope, S. Hand, V. Selo, V. Jain, N. Sethi, M. Goel, T. Makino, R. May, Z. Yang, J. Schalkwyk, C. Butterfield, A. Hauth, A. Goldin, W. Hawkins, E. Senter, S. Brin, O. Woodman, M. Ritter, E. Noland, M. Giang, V. Bolina, L. Lee, T. Blyth, I. Mackinnon, M. Reid, O. Sarvana, D. Silver, A. Chen, L. Wang, L. Maggiore, O. Chang, N. Attaluri, G. Thornton, C. Chiu, O. Bunyan, N. Levine, T. Chung, E. Eltyshev, X. Si, T. Lillicrap, D. Brady, V. Aggarwal, B. Wu, Y. Xu, R. McIlroy, K. Badola, P. Sandhu, E. Moreira, W. Stokowiec, R. Hemsley, D. Li, A. Tudor, P. Shyam, E. Rahimtoroghi, S. Haykal, P. Sprechmann, X. Zhou, D. Mincu, Y. Li, R. Addanki, K. Krishna, X. Wu, A. Frechette, M. Eyal, A. Dafoe, D. Lacey, J. Whang, T. Avrahami, Y. Zhang, E. Taropa, H. Lin, D. Toyama, E. Rutherford, M. Sano, H. Choe, A. Tomala, C. Safranek-Shrader, N. Kassner, M. Pajarskas, M. Harvey, S. Sechrist, M. Fortunato, C. Lyu, G. Elsayed, C. Kuang, J. Lottes, E. Chu, C. Jia, C. Chen, P. Humphreys, K. Baumli, C. Tao, R. Samuel, C. N. dos Santos, A. Andreassen, N. Rakićević, D. Grewe, A. Kumar, S. Winkler, J. Caton, A. Brock, S. Dalmia, H. Sheahan, I. Barr, Y. Miao, P. Natsev, J. Devlin, F. Behbahani, F. Prost, Y. Sun, A. Myaskovsky, T. S. Pillai, D. Hurt, A. Lazaridou, X. Xiong, C. Zheng, F. Pardo, X. Li, D. Horgan, J. Stanton, M. Ambar, F. Xia, A. Lince, M. Wang, B. Mustafa, A. Webson, H. Lee, R. Anil, M. Wicke, T. Dozat, A. Sinha, E. Piqueras, E. Dabir, S. Upadhyay, A. Boral, L. A. Hendricks, C. Fry, J. Djolonga, Y. Su, J. Walker, J. Labanowski, R. Huang, V. Misra, J. Chen, R. Skerry-Ryan, A. Singh, S. Rijhwani, D. Yu, A. Castro-Ros, B. Changpinyo, R. Datta, S. Bagri, A. M. Hrafnkelsson, M. Maggioni, D. Zheng, Y. Sulsky, S. Hou, T. L. Paine, A. Yang, J. Riesa, D. Rogozinska, D. Marcus, D. E. Badawy, Q. Zhang, L. Wang, H. Miller, J. Greer, L. L. Sjos, A. Nova, H. Zen, R. Chaabouni, M. Rosca, J. Jiang, C. Chen, R. Liu, T. Sainath, M. Krikun, A. Polozov, J. Lespiau, J. Newlan, Z. Cankara, S. Kwak, Y. Xu, P. Chen, A. Coenen, C. Meyer, K. Tsihlas, A. Ma, J. Gottweis, J. Xing, C. Gu, J. Miao, C. Frank, Z. Cankara, S. Ganapathy, I. Dasgupta, S. Hughes-Fitt, H. Chen, D. Reid, K. Rong, H. Fan, J. van Amersfoort, V. Zhuang, A. Cohen, S. S. Gu, A. Mohananey, A. Ilic, T. Tobin, J. Wieting, A. Bortsova, P. Thacker, E. Wang, E. Caveness, J. Chiu, E. Sezener, A. Kaskasoli, S. Baker, K. Millican, M. Elhawaty, K. Aisopos, C. Lebsack, N. Byrd, H. Dai, W. Jia, M. Wiethoff, E. Davoodi, A. Weston, L. Yagati, A. Ahuja, I. Gao, G. Pundak, S. Zhang, M. Azzam, K. C. Sim, S. Caelles, J. Keeling, A. Sharma, A. Swing, Y. Li, C. Liu, C. G. Bostock, Y. Bansal, Z. Nado, A. Anand, J. Lipschultz, A. Karmarkar, L. Proleev, A. Ittycheriah, S. H. Yeganeh, G. Polovets, A. Faust, J. Sun, A. Rrustemi, P. Li, R. Shivanna, J. Liu, C. Welty, F. Lebron, A. Baddepudi, S. Krause, E. Parisotto, R. Soricut, Z. Xu, D. Bloxwich, M. Johnson, B. Neyshabur, J. Mao-Jones, R. Wang, V. Ramasesh, Z. Abbas, A. Guez, C. Segal, D. D. Nguyen, J. Svensson, L. Hou, S. York, K. Milan, S. Bridgers, W. Gworek, M. Tagliasacchi, J. Lee-Thorp, M. Chang, A. Guseynov, A. J. Hartman, M. Kwong, R. Zhao, S. Kashem, E. Cole, A. Miech, R. Tanburn, M. Phuong, F. Pavetic, S. Cevey, R. Comanescu, R. Ives, S. Yang, C. Du, B. Li, Z. Zhang, M. Iinuma, C. H. Hu, A. Roy, S. Bijwadia, Z. Zhu, D. Martins, R. Saputro, A. Gergely, S. Zheng, D. Jia, I. Antonoglou, A. Sadovsky, S. Gu, Y. Bi, A. Andreev, S. Samangooei, M. Khan, T. Kocisky, A. Filos, C. Kumar, C. Bishop, A. Yu, S. Hodkinson, S. Mittal, P. Shah, A. Moufarek, Y. Cheng, A. Bloniarz, J. Lee, P. Pejman, P. Michel, S. Spencer, V. Feinberg, X. Xiong, N. Savinov, C. Smith, S. Shakeri, D. Tran, M. Chesus, B. Bohnet, G. Tucker, T. von Glehn, C. Muir, Y. Mao, H. Kazawa, A. Slone, K. Soparkar, D. Shrivastava, J. Cobon-Kerr, M. Sharman, J. Pavagadhi, C. Araya, K. Misiunas, N. Ghelani, M. Laskin, D. Barker, Q. Li, A. Briukhov, N. Houlsby, M. Glaese, B. Lakshminarayanan, N. Schucher, Y. Tang, E. Collins, H. Lim, F. Feng, A. Recasens, G. Lai, A. Magni, N. D. Cao, A. Siddhant, Z. Ashwood, J. Orbay, M. Dehghani, J. Brennan, Y. He, K. Xu, Y. Gao, C. Saroufim, J. Molloy, X. Wu, S. Arnold, S. Chang, J. Schrittwieser, E. Buchatskaya, S. Radpour, M. Polacek, S. Giordano, A. Bapna, S. Tokumine, V. Hellendoorn, T. Sottiaux, S. Cogan, A. Severyn, M. Saleh, S. Thakoor, L. Shefey, S. Qiao, M. Gaba, S. Chang, C. Swanson, B. Zhang, B. Lee, P. K. Rubenstein, G. Song, T. Kwiatkowski, A. Koop, A. Kannan, D. Kao, P. Schuh, A. Stjerngren, G. Ghiasi, G. Gibson, L. Vilnis, Y. Yuan, F. T. Ferreira, A. Kamath, T. Klimenko, K. Franko, K. Xiao, I. Bhattacharya, M. Patel, R. Wang, A. Morris, R. Strudel, V. Sharma, P. Choy, S. H. Hashemi, J. Landon, M. Finkelstein, P. Jhakra, J. Frye, M. Barnes, M. Mauger, D. Daun, K. Baatarsukh, M. Tung, W. Farhan, H. Michalewski, F. Viola, F. de Chaumont Quitry, C. L. Lan, T. Hudson, Q. Wang, F. Fischer, I. Zheng, E. White, A. Dragan, J. Alayrac, E. Ni, A. Pritzel, A. Iwanicki, M. Isard, A. Bulanova, L. Zilka, E. Dyer, D. Sachan, S. Srinivasan, H. Muckenhirn, H. Cai, A. Mandhane, M. Tariq, J. W. Rae, G. Wang, K. Ayoub, N. FitzGerald, Y. Zhao, W. Han, C. Alberti, D. Garrette, K. Krishnakumar, M. Gimenez, A. Levskaya, D. Sohn, J. Matak, I. Iturrate, M. B. Chang, J. Xiang, Y. Cao, N. Ranka, G. Brown, A. Hutter, V. Mirrokni, N. Chen, K. Yao, Z. Egyed, F. Galilee, T. Liechty, P. Kallakuri, E. Palmer, S. Ghemawat, J. Liu, D. Tao, C. Thornton, T. Green, M. Jasarevic, S. Lin, V. Cotruta, Y. Tan, N. Fiedel, H. Yu, E. Chi, A. Neitz, J. Heitkaemper, A. Sinha, D. Zhou, Y. Sun, C. Kaed, B. Hulse, S. Mishra, M. Georgaki, S. Kudugunta, C. Farabet, I. Shafran, D. Vlasic, A. Tsitsulin, R. Ananthanarayanan, A. Carin, G. Su, P. Sun, S. V, G. Carvajal, J. Broder, I. Comsa, A. Repina, W. Wong, W. W. Chen, P. Hawkins, E. Filonov, L. Loher, C. Hirnschall, W. Wang, J. Ye, A. Burns, H. Cate, D. G. Wright, F. Piccinini, L. Zhang, C. Lin, I. Gog, Y. Kulizhskaya, A. Sreevatsa, S. Song, L. C. Cobo, A. Iyer, C. Tekur, G. Garrido, Z. Xiao, R. Kemp, H. S. Zheng, H. Li, A. Agarwal, C. Ngani, K. Goshvadi, R. Santamaria-Fernandez, W. Fica, X. Chen, C. Gorgolewski, S. Sun, R. Garg, X. Ye, S. M. A. Eslami, N. Hua, J. Simon, P. Joshi, Y. Kim, I. Tenney, S. Potluri, L. N. Thiet, Q. Yuan, F. Luisier, A. Chronopoulou, S. Scellato, P. Srinivasan, M. Chen, V. Koverkathu, V. Dalibard, Y. Xu, B. Saeta, K. Anderson, T. Sellam, N. Fernando, F. Huot, J. Jung, M. Varadarajan, M. Quinn, A. Raul, M. Le, R. Habalov, J. Clark, K. Jalan, K. Bullard, A. Singhal, T. Luong, B. Wang, S. Rajayogam, J. Eisenschlos, J. Jia, D. Finchelstein, A. Yakubovich, D. Balle, M. Fink, S. Agarwal, J. Li, D. Dvijotham, S. Pal, K. Kang, J. Konzelmann, J. Beattie, O. Dousse, D. Wu, R. Crocker, C. Elkind, S. R. Jonnalagadda, J. Lee, D. Holtmann-Rice, K. Kallarackal, R. Liu, D. Vnukov, N. Vats, L. Invernizzi, M. Jafari, H. Zhou, L. Taylor, J. Prendki, M. Wu, T. Eccles, T. Liu, K. Kopparapu, F. Beaufays, C. Angermueller, A. Marzoca, S. Sarcar, H. Dib, J. Stanway, F. Perbet, N. Trdin, R. Sterneck, A. Khorlin, D. Li, X. Wu, S. Goenka, D. Madras, S. Goldshtein, W. Gierke, T. Zhou, Y. Liu, Y. Liang, A. White, Y. Li, S. Singh, S. Bahargam, M. Epstein, S. Basu, L. Lao, A. Ozturel, C. Crous, A. Zhai, H. Lu, Z. Tung, N. Gaur, A. Walton, L. Dixon, M. Zhang, A. Globerson, G. Uy, A. Bolt, O. Wiles, M. Nasr, I. Shumailov, M. Selvi, F. Piccinno, R. Aguilar, S. McCarthy, M. Khalman, M. Shukla, V. Galic, J. Carpenter, K. Villela, H. Zhang, H. Richardson, J. Martens, M. Bosnjak, S. R. Belle, J. Seibert, M. Alnahlawi, B. McWilliams, S. Singh, A. Louis, W. Ding, D. Popovici, L. Simicich, L. Knight, P. Mehta, N. Gupta, C. Shi, S. Fatehi, J. Mitrovic, A. Grills, J. Pagadora, T. Munkhdalai, D. Petrova, D. Eisenbud, Z. Zhang, D. Yates, B. Mittal, N. Tripuraneni, Y. Assael, T. Brovelli, P. Jain, M. Velimirovic, C. Akbulut, J. Mu, W. Macherey, R. Kumar, J. Xu, H. Qureshi, G. Comanici, J. Wiesner, Z. Gong, A. Ruddock, M. Bauer, N. Felt, A. GP, A. Arnab, D. Zelle, J. Rothfuss, B. Rosgen, A. Shenoy, B. Seybold, X. Li, J. Mudigonda, G. Erdogan, J. Xia, J. Simsa, A. Michi, Y. Yao, C. Yew, S. Kan, I. Caswell, C. Radebaugh, A. Elisseeff, P. Valenzuela, K. McKinney, K. Paterson, A. Cui, E. Latorre-Chimoto, S. Kim, W. Zeng, K. Durden, P. Ponnapalli, T. Sosea, C. A. Choquette-Choo, J. Manyika, B. Robenek, H. Vashisht, S. Pereira, H. Lam, M. Velic, D. Owusu-Afriyie, K. Lee, T. Bolukbasi, A. Parrish, S. Lu, J. Park, B. Venkatraman, A. Talbert, L. Rosique, Y. Cheng, A. Sozanschi, A. Paszke, P. Kumar, J. Austin, L. Li, K. Salama, B. Perz, W. Kim, N. Dukkipati, A. Baryshnikov, C. Kaplanis, X. Sheng, Y. Chervonyi, C. Unlu, D. de Las Casas, H. Askham, K. Tunyasuvunakool, F. Gimeno, S. Poder, C. Kwak, M. Miecnikowski, V. Mirrokni, A. Dimitriev, A. Parisi, D. Liu, T. Tsai, T. Shevlane, C. Kouridi, D. Garmon, A. Goedeckemeyer, A. R. Brown, A. Vijayakumar, A. Elqursh, S. Jazayeri, J. Huang, S. M. Carthy, J. Hoover, L. Kim, S. Kumar, W. Chen, C. Biles, G. Bingham, E. Rosen, L. Wang, Q. Tan, D. Engel, F. Pongetti, D. de Cesare, D. Hwang, L. Yu, J. Pullman, S. Narayanan, K. Levin, S. Gopal, M. Li, A. Aharoni, T. Trinh, J. Lo, N. Casagrande, R. Vij, L. Matthey, B. Ramadhana, A. Matthews, C. Carey, M. Johnson, K. Goranova, R. Shah, S. Ashraf, K. Dasgupta, R. Larsen, Y. Wang, M. R. Vuyyuru, C. Jiang, J. Ijazi, K. Osawa, C. Smith, R. S. Boppana, T. Bilal, Y. Koizumi, Y. Xu, Y. Altun, N. Shabat, B. Bariach, A. Korchemniy, K. Choo, O. Ronneberger, C. Iwuanyanwu, S. Zhao, D. Soergel, C. Hsieh, I. Cai, S. Iqbal, M. Sundermeyer, Z. Chen, E. Bursztein, C. Malaviya, F. Biadsy, P. Shroff, I. Dhillon, T. Latkar, C. Dyer, H. Forbes, M. Nicosia, V. Nikolaev, S. Greene, M. Georgiev, P. Wang, N. Martin, H. Sedghi, J. Zhang, P. Banzal, D. Fritz, V. Rao, X. Wang, J. Zhang, V. Patraucean, D. Du, I. Mordatch, I. Jurin, L. Liu, A. Dubey, A. Mohan, J. Nowakowski, V. Ion, N. Wei, R. Tojo, M. A. Raad, D. A. Hudson, V. Keshava, S. Agrawal, K. Ramirez, Z. Wu, H. Nguyen, J. Liu, M. Sewak, B. Petrini, D. Choi, I. Philips, Z. Wang, I. Bica, A. Garg, J. Wilkiewicz, P. Agrawal, X. Li, D. Guo, E. Xue, N. Shaik, A. Leach, S. M. Khan, J. Wiesinger, S. Jerome, A. Chakladar, A. W. Wang, T. Ornduff, F. Abu, A. Ghaffarkhah, M. Wainwright, M. Cortes, F. Liu, J. Maynez, A. Terzis, P. Samangouei, R. Mansour, T. Kępa, F. Aubet, A. Algymr, D. Banica, A. Weisz, A. Orban, A. Senges, E. Andrejczuk, M. Geller, N. D. Santo, V. Anklin, M. A. Merey, M. Baeuml, T. Strohman, J. Bai, S. Petrov, Y. Wu, D. Hassabis, K. Kavukcuoglu, J. Dean, and O. Vinyals (2024)Gemini 1.5: unlocking multimodal understanding across millions of tokens of context. External Links: 2403.05530, [Link](https://arxiv.org/abs/2403.05530)Cited by: [§3.3](https://arxiv.org/html/2604.19440#S3.SS3.p2.6 "3.3 Evolution Scale and Parameters ‣ 3 Methodology ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   G. Team, A. Kamath, J. Ferret, S. Pathak, N. Vieillard, R. Merhej, S. Perrin, T. Matejovicova, A. Ramé, M. Rivière, L. Rouillard, T. Mesnard, G. Cideron, J. Grill, S. Ramos, E. Yvinec, M. Casbon, E. Pot, I. Penchev, G. Liu, F. Visin, K. Kenealy, L. Beyer, X. Zhai, A. Tsitsulin, R. Busa-Fekete, A. Feng, N. Sachdeva, B. Coleman, Y. Gao, B. Mustafa, I. Barr, E. Parisotto, D. Tian, M. Eyal, C. Cherry, J. Peter, D. Sinopalnikov, S. Bhupatiraju, R. Agarwal, M. Kazemi, D. Malkin, R. Kumar, D. Vilar, I. Brusilovsky, J. Luo, A. Steiner, A. Friesen, A. Sharma, A. Sharma, A. M. Gilady, A. Goedeckemeyer, A. Saade, A. Feng, A. Kolesnikov, A. Bendebury, A. Abdagic, A. Vadi, A. György, A. S. Pinto, A. Das, A. Bapna, A. Miech, A. Yang, A. Paterson, A. Shenoy, A. Chakrabarti, B. Piot, B. Wu, B. Shahriari, B. Petrini, C. Chen, C. L. Lan, C. A. Choquette-Choo, C. Carey, C. Brick, D. Deutsch, D. Eisenbud, D. Cattle, D. Cheng, D. Paparas, D. S. Sreepathihalli, D. Reid, D. Tran, D. Zelle, E. Noland, E. Huizenga, E. Kharitonov, F. Liu, G. Amirkhanyan, G. Cameron, H. Hashemi, H. Klimczak-Plucińska, H. Singh, H. Mehta, H. T. Lehri, H. Hazimeh, I. Ballantyne, I. Szpektor, I. Nardini, J. Pouget-Abadie, J. Chan, J. Stanton, J. Wieting, J. Lai, J. Orbay, J. Fernandez, J. Newlan, J. Ji, J. Singh, K. Black, K. Yu, K. Hui, K. Vodrahalli, K. Greff, L. Qiu, M. Valentine, M. Coelho, M. Ritter, M. Hoffman, M. Watson, M. Chaturvedi, M. Moynihan, M. Ma, N. Babar, N. Noy, N. Byrd, N. Roy, N. Momchev, N. Chauhan, N. Sachdeva, O. Bunyan, P. Botarda, P. Caron, P. K. Rubenstein, P. Culliton, P. Schmid, P. G. Sessa, P. Xu, P. Stanczyk, P. Tafti, R. Shivanna, R. Wu, R. Pan, R. Rokni, R. Willoughby, R. Vallu, R. Mullins, S. Jerome, S. Smoot, S. Girgin, S. Iqbal, S. Reddy, S. Sheth, S. Põder, S. Bhatnagar, S. R. Panyam, S. Eiger, S. Zhang, T. Liu, T. Yacovone, T. Liechty, U. Kalra, U. Evci, V. Misra, V. Roseberry, V. Feinberg, V. Kolesnikov, W. Han, W. Kwon, X. Chen, Y. Chow, Y. Zhu, Z. Wei, Z. Egyed, V. Cotruta, M. Giang, P. Kirk, A. Rao, K. Black, N. Babar, J. Lo, E. Moreira, L. G. Martins, O. Sanseviero, L. Gonzalez, Z. Gleicher, T. Warkentin, V. Mirrokni, E. Senter, E. Collins, J. Barral, Z. Ghahramani, R. Hadsell, Y. Matias, D. Sculley, S. Petrov, N. Fiedel, N. Shazeer, O. Vinyals, J. Dean, D. Hassabis, K. Kavukcuoglu, C. Farabet, E. Buchatskaya, J. Alayrac, R. Anil, Dmitry, Lepikhin, S. Borgeaud, O. Bachem, A. Joulin, A. Andreev, C. Hardin, R. Dadashi, and L. Hussenot (2025)Gemma 3 technical report. External Links: 2503.19786, [Link](https://arxiv.org/abs/2503.19786)Cited by: [§3.3](https://arxiv.org/html/2604.19440#S3.SS3.p2.6 "3.3 Evolution Scale and Parameters ‣ 3 Methodology ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   N. van Stein, H. Yin, A. V. Kononova, T. Bäck, and G. Ochoa (2025)Behaviour space analysis of llm-driven meta-heuristic discovery. External Links: 2507.03605, [Link](https://arxiv.org/abs/2507.03605)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   X. Wu, S. Wu, J. Wu, L. Feng, and K. C. Tan (2025)Evolutionary computation in the era of large language model: survey and roadmap. IEEE Transactions on Evolutionary Computation 29 (2),  pp.534–554. External Links: [Document](https://dx.doi.org/10.1109/TEVC.2024.3506731), [Link](https://ieeexplore.ieee.org/document/10767756)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   C. Yang, X. Wang, Y. Lu, H. Liu, Q. V. Le, D. Zhou, and X. Chen (2024)Large language models as optimizers. External Links: 2309.03409, [Link](https://arxiv.org/abs/2309.03409)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   X. Yang, L. Zhang, H. Qian, L. Song, and J. Bian (2025a)HeurAgenix: leveraging llms for solving complex combinatorial optimization challenges. External Links: 2506.15196, [Link](https://arxiv.org/abs/2506.15196)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   Z. Yang, W. Liu, B. Gao, T. Xie, Y. Li, W. Ouyang, S. Poria, E. Cambria, and D. Zhou (2025b)MOOSE-chem: large language models for rediscovering unseen chemistry scientific hypotheses. External Links: 2410.07076, [Link](https://arxiv.org/abs/2410.07076)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   H. Ye, J. Wang, Z. Cao, F. Berto, C. Hua, H. Kim, J. Park, and G. Song (2024)ReEvo: large language models as hyper-heuristics with reflective evolution. In Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (Eds.), Vol. 37,  pp.43571–43608. External Links: [Document](https://dx.doi.org/10.52202/079017-1381), [Link](https://proceedings.neurips.cc/paper_files/paper/2024/file/4ced59d480e07d290b6f29fc8798f195-Paper-Conference.pdf)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   S. Yu, T. Chen, and L. Liu (2026)Large language model-driven full-component evolution of adaptive large neighborhood search. External Links: 2603.06996, [Link](https://arxiv.org/abs/2603.06996)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   L. Zhao, W. Huang, Y. Guo, J. Bian, C. Wang, and X. Zhang (2026)Large language model-powered evolutionary code optimization on a phylogenetic tree. External Links: 2601.14523, [Link](https://arxiv.org/abs/2601.14523)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   Z. Zhao, C. Hua, F. Berto, K. Lee, Z. Ma, J. Li, and J. Park (2025)TrajEvo: designing trajectory prediction heuristics via llm-driven evolution. External Links: 2505.04480, [Link](https://arxiv.org/abs/2505.04480)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   Y. Zhou, H. Liu, T. Srivastava, H. Mei, and C. Tan (2024)Hypothesis generation with large language models. In Proceedings of the 1st Workshop on NLP for Science (NLP4Science),  pp.117–139. External Links: [Link](http://dx.doi.org/10.18653/v1/2024.nlp4science-1.10), [Document](https://dx.doi.org/10.18653/v1/2024.nlp4science-1.10)Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px1.p1.1 "Evolutionary Computation with LLMs ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 
*   Y. Zhou, C. Wu, P. Mulhem, D. Schwab, and M. Peyrard (2026)What matters to an LLM? behavioral and computational evidences from summarization. In Findings of the Association for Computational Linguistics: EACL 2026, V. Demberg, K. Inui, and L. Marquez (Eds.), Rabat, Morocco,  pp.5712–5737. External Links: [Link](https://aclanthology.org/2026.findings-eacl.302/), [Document](https://dx.doi.org/10.18653/v1/2026.findings-eacl.302), ISBN 979-8-89176-386-9 Cited by: [§2](https://arxiv.org/html/2604.19440#S2.SS0.SSS0.Px2.p1.1 "Evaluating LLMs as Search Operators ‣ 2 Related Work ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search"). 

## Appendix A Complete Experimental Result

See Table[2](https://arxiv.org/html/2604.19440#A1.T2 "Table 2 ‣ Appendix A Complete Experimental Result ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search").

| LLM | Zero-Shot | First Generation | Last Generation |
| --- |
| ![Image 22: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/route.png) | ![Image 23: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/text.png) | ![Image 24: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/math.png) | ![Image 25: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/code.png) | Avg | ![Image 26: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/route.png) | ![Image 27: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/text.png) | ![Image 28: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/math.png) | ![Image 29: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/code.png) | Avg | ![Image 30: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/route.png) | ![Image 31: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/text.png) | ![Image 32: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/math.png) | ![Image 33: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/code.png) | Avg |
| ![Image 34: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/openai.png) 4o | 47.4 | 51.9 | 82.3 | 31.5 | 53.3 | 37.2 | 41.3 | 75.7 | 31.7 | 46.5 | 85.4 | 70.9 | 77.7 | 75.4 | 77.4 |
| ![Image 35: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/gemini.png) 1.5-Pro | 47.3 | 43.0 | 70.4 | 30.7 | 47.8 | 43.0 | 38.3 | 84.0 | 32.4 | 49.4 | 89.0 | 72.4 | 85.5 | 58.5 | 76.4 |
| ![Image 36: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/deepseek.png) V3 | 39.0 | 41.2 | 71.5 | 31.5 | 45.8 | 50.8 | 56.6 | 87.2 | 33.0 | 56.9 | 70.4 | 77.1 | 91.1 | 62.7 | 75.3 |
| ![Image 37: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/mistral.png) Large | 19.5 | 49.1 | 79.7 | 31.5 | 45.0 | 34.5 | 56.6 | 74.1 | 33.0 | 49.5 | 58.4 | 84.7 | 78.7 | 81.1 | 75.7 |
| ![Image 38: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/meta.png) 3.1-70B-Instruct | 15.9 | 55.2 | 75.2 | 31.5 | 44.5 | 34.0 | 45.1 | 69.9 | 33.0 | 45.5 | 59.6 | 69.5 | 78.0 | 69.8 | 69.2 |
| ![Image 39: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/mistral.png) Magistral-Small | 29.0 | 49.9 | 66.6 | 31.5 | 44.3 | 34.0 | 61.8 | 70.5 | 31.7 | 49.5 | 68.0 | 73.7 | 75.4 | 64.4 | 70.4 |
| ![Image 40: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/mistral.png) 24B-Instruct | 11.6 | 55.8 | 72.2 | 31.5 | 42.8 | 34.0 | 66.5 | 70.2 | 33.1 | 51.0 | 75.0 | 84.3 | 75.2 | 92.0 | 81.6 |
| ![Image 41: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/openai.png) 4o-mini | 13.9 | 38.3 | 70.1 | 31.5 | 38.4 | 34.0 | 55.8 | 66.7 | 31.7 | 47.1 | 60.2 | 71.2 | 82.9 | 66.2 | 70.1 |
| ![Image 42: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/meta.png) 3.1-8B-Instruct | 2.9 | 55.7 | 48.7 | 32.9 | 35.1 | 34.3 | 44.5 | 67.4 | 31.7 | 44.5 | 63.8 | 73.7 | 77.1 | 74.5 | 72.2 |
| ![Image 43: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/mistral.png) 7B-Instruct | 18.5 | 46.9 | 49.7 | 23.6 | 34.7 | 35.1 | 47.7 | 67.7 | 31.7 | 45.5 | 46.9 | 73.7 | 91.8 | 67.7 | 70.0 |
| ![Image 44: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/gemini.png) 1.5-Flash | 6.6 | 47.5 | 60.9 | 3.4 | 29.6 | 34.0 | 49.8 | 73.5 | 31.7 | 47.2 | 57.1 | 95.9 | 75.1 | 44.6 | 68.2 |
| ![Image 45: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/meta.png) 3.2-3B-Instruct | 22.2 | 36.3 | 28.4 | 29.8 | 29.1 | 34.6 | 47.7 | 66.7 | 32.3 | 45.3 | 55.1 | 70.6 | 85.9 | 47.5 | 64.8 |
| ![Image 46: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/meta.png) 3.2-1B-Instruct | 14.8 | 47.8 | 0.0 | 31.5 | 23.5 | 34.0 | 53.1 | 69.4 | 31.7 | 47.0 | 54.1 | 68.5 | 80.9 | 48.5 | 63.0 |
| ![Image 47: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/gemma.png) 3n-4B | 0.0 | 55.6 | 0.0 | 22.9 | 19.6 | 35.6 | 53.2 | 66.7 | 31.7 | 46.8 | 42.7 | 80.4 | 67.8 | 52.2 | 60.8 |
| ![Image 48: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/openai.png) 3.5-turbo | 16.5 | 27.2 | 0.0 | 28.1 | 18.0 | 34.0 | 40.5 | 75.6 | 33.2 | 45.8 | 49.0 | 65.8 | 81.4 | 41.0 | 59.3 |

Table 2: Fitness performance of LLMs on four task families (![Image 49: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/route.png) Route Optimization, ![Image 50: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/text.png) Prompt Optimization, ![Image 51: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/math.png) Equation Discovery, ![Image 52: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/code.png) Heuristic Design). Cells report averaged normalized fitness across two sub-tasks and two seeds within the same family, making scores comparable over three settings. Those cells are background-shaded by a normalized improvement (darker = larger improvement) computed per (model,task) pair relative to their initial-population best value of each sub-task. Models are sorted in descending order of their average Zero-Shot score. Best scores per column are bold.

## Appendix B Use of AI Assistant

AI tools were used to assist in writing, editing, and code development. The authors provided all content, ideas, and decisions, and the AI was used solely to improve clarity, readability, and efficiency.

## Appendix C Task-Specific Experimental Details

Here are task-family-specific details of (i) EA parameters, (ii) genome validity checks, (iii) fitness evaluation, (iv) novelty distance, (v) population initialization, and (vi) prompts used for zero-shot and evolutionary search. Unless stated otherwise, the global evolutionary loop (selection–mutation–evaluation–pool update) follows Section[3](https://arxiv.org/html/2604.19440#S3 "3 Methodology ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search").

### C.1 ![Image 53: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/route.png) Route Optimization

##### EA Parameters:

For both TSP-30 and TSP-60, we use: $n_{init} = 40$ (initial population size), $q = 0.2$ (elite fraction), $p_{parent} = 3$ (parents sampled per generation), $p_{child} = 10$ (offspring per generation), $N = 40$ (capacity-limited pool), $G = 30$ (generations), and seed $= 21$.

##### Genome and Validity:

A genome is a permutation $\pi \in S_{n}$ where $n \in \left{\right. 30 , 60 \left.\right}$ is the number of cities. The LLM generates a genome as a JSON array of integers. Invalid genomes (non-permutations, unparsable outputs) receive fitness $f = 0$ and are excluded from parent sampling.

##### Fitness Evaluation:

Given a distance matrix $DIST \in \mathbb{R}^{n \times n}$, the tour length is $L ​ \left(\right. \pi \left.\right) = \sum_{i = 1}^{n - 1} DIST_{\pi_{i} , \pi_{i + 1}} + DIST_{\pi_{n} , \pi_{1}}$. Fitness is the inverted length: $f_{TSP} ​ \left(\right. \pi \left.\right) = - L ​ \left(\right. \pi \left.\right)$ (normalized post-hoc by task-level min/max).

##### Novelty Distance:

We use edge-set distance after canonization:

$D_{edge} ​ \left(\right. \pi , \sigma \left.\right) = 1 - \frac{\left|\right. E ​ \left(\right. \pi \left.\right) \cap E ​ \left(\right. \sigma \left.\right) \left|\right.}{\left|\right. E ​ \left(\right. \pi \left.\right) \left|\right.} ,$

where $E ​ \left(\right. \pi \left.\right)$ denotes the set of undirected edges in tour $\pi$. This metric captures structural differences regardless of rotation or starting city.

##### Population Initialization:

The initial population $\mathcal{P}_{0}$ consists of $n_{init} = 40$ random permutations (using the same random seed across all models). Each permutation is obtained via random.sample(range(n), n).

##### Zero-shot Prompt:

The zero-shot prompt provides the complete distance matrix as JSON and asks the model to return an optimal tour. See Table[3](https://arxiv.org/html/2604.19440#A3.T3 "Table 3 ‣ Zero-shot Prompt: ‣ C.1 Route Optimization ‣ Appendix C Task-Specific Experimental Details ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search").

Table 3: TSP Zero-shot Prompt Template

##### Evolution Prompt:

During evolution, the prompt provides the distance matrix and up to 3 parent tours with their scores (path lengths). The LLM is asked to generate a better child tour. See Table[4](https://arxiv.org/html/2604.19440#A3.T4 "Table 4 ‣ Evolution Prompt: ‣ C.1 Route Optimization ‣ Appendix C Task-Specific Experimental Details ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search").

Table 4: TSP Evolution Prompt Template

### C.2 ![Image 54: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/text.png) Prompt Optimization

##### EA Parameters:

$n_{init} = 10$, $q = 0.2$, $p_{parent} = 2$, $p_{child} = 5$, $N = 10$ (capacity), $G = 30$, and two task variants: SAMSum (dialogue summarization) and ASSET (text simplification).

##### Genome and Validity:

A genome is a natural language instruction prompt (string). Any non-empty string output is considered valid.

##### Fitness Evaluation:

For each prompt genome $p$, we condition a frozen evaluator LLM (gpt-4o-mini) on $p$ to generate outputs on a held-out 25% evaluation set. Fitness is the task-specific metric score (ROUGE-L for SAMSum, SARI for ASSET).

##### Novelty Distance:

We use semantic cosine distance in a fixed embedding space (OpenAI’s text-embedding-ada-002):

$D_{cos} ​ \left(\right. p_{1} , p_{2} \left.\right) = 1 - \frac{E ​ \left(\right. p_{1} \left.\right) \cdot E ​ \left(\right. p_{2} \left.\right)}{\parallel E ​ \left(\right. p_{1} \left.\right) \parallel ​ \parallel E ​ \left(\right. p_{2} \left.\right) \parallel} ,$

where $E ​ \left(\right. \cdot \left.\right)$ is the sentence embedding.

##### Population Initialization:

Initial prompts are loaded from a curated set of baseline prompts for each task (SAMSum or ASSET). Duplicates are removed, and the first 10 unique prompts are selected.

##### Zero-shot Prompt:

See Table[5](https://arxiv.org/html/2604.19440#A3.T5 "Table 5 ‣ Zero-shot Prompt: ‣ C.2 Prompt Optimization ‣ Appendix C Task-Specific Experimental Details ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search").

Table 5: Prompt Optimization Zero-shot Prompt Template

##### Evolution Prompt:

See Table[6](https://arxiv.org/html/2604.19440#A3.T6 "Table 6 ‣ Evolution Prompt: ‣ C.2 Prompt Optimization ‣ Appendix C Task-Specific Experimental Details ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search").

Table 6: Prompt Optimization Evolution Prompt Template

### C.3 ![Image 55: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/math.png) Equation Discovery

##### EA Parameters:

$n_{init} = 7$, $q = 0.2$, $p_{parent} = 2$, $p_{child} = 10$, $N = 40$ (capacity), $G = 30$, and seed $= 21$. We evaluate on two nonlinear oscillator datasets with different dimensionalities (Oscillator-1: 3 variables; Oscillator-2: 4 variables with time).

##### Genome and Validity:

A genome is a Python function string defining $a = f ​ \left(\right. x , v \left.\right)$ (Oscillator-1) or $a = f ​ \left(\right. t , x , v \left.\right)$ (Oscillator-2). Genomes are validated by attempting to parse and execute them; non-executable or divergent outputs receive fitness $f = 1 \times 10^{6}$ (high loss).

##### Fitness Evaluation:

Given training data $\left(\right. X , y_{true} \left.\right)$, fitness is computed as:

$f_{SymReg} = 1 - norm ​ \left(\right. MSE ​ \left(\right. y_{pred} , y_{true} \left.\right) \left.\right) ,$

where normalization is per-task instance.

##### Novelty Distance:

We use functional behavior distance over a fixed input grid:

$D_{sem} ​ \left(\right. f , g \left.\right) = 1 - \frac{1}{m} ​ \sum_{j = 1}^{m} cos ⁡ \left(\right. f ​ \left(\right. x_{j} \left.\right) , g ​ \left(\right. x_{j} \left.\right) \left.\right) ,$

where $x_{j}$ are uniformly sampled input points and $cos ⁡ \left(\right. \cdot \left.\right)$ is cosine similarity. This captures semantic divergence in output behavior.

##### Population Initialization:

Initial genomes are randomly sampled expressions combining input variables, constants, and allowed functions (e.g., np.sin, np.cos, np.exp). We maintain the same seed for all models. All initial genomes are evaluated on the training set.

##### Zero-shot Prompt:

See Table[7](https://arxiv.org/html/2604.19440#A3.T7 "Table 7 ‣ Zero-shot Prompt: ‣ C.3 Equation Discovery ‣ Appendix C Task-Specific Experimental Details ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search").

Table 7: Symbolic Regression Zero-shot Prompt Template

##### Evolution Prompt:

See Table[8](https://arxiv.org/html/2604.19440#A3.T8 "Table 8 ‣ Evolution Prompt: ‣ C.3 Equation Discovery ‣ Appendix C Task-Specific Experimental Details ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search").

Table 8: Symbolic Regression Evolution Prompt Template

### C.4 ![Image 56: [Uncaptioned image]](https://arxiv.org/html/2604.19440v1/img/icons/code.png) Heuristic Design

##### EA Parameters:

For bin packing heuristic design, we use: $n_{init} = 7$, $q = 0.2$, $p_{parent} = 2$, $p_{child} = 10$, $N = 40$ (capacity), $G = 30$, and seed $= 42$.

##### Genome and Validity:

A genome is a Python function string implementing a priority heuristic def priority(item, bins): .... The function takes item size and bin residual capacities as input, and returns priority scores. Invalid or non-executable functions receive fitness $f = 1 \times 10^{6}$.

##### Fitness Evaluation:

For each genome, we run online bin packing on all instances of the active dataset (OR3 or Weibull) using the heuristic function. Fitness is the inverted number of bins used, averaged over all instances:

$f_{BinPack} = - \frac{1}{\left|\right. instances \left|\right.} ​ \underset{i}{\sum} bins ​ _ ​ used ​ \left(\right. i \left.\right) .$

##### Novelty Distance:

We measure behavioral/strategy distance as well. We define a set of fixed “probe scenarios” (random combinations of item sizes and bin capacities), compute the priority/score vectors for each candidate function under these scenarios, and measure distance via:

$D_{behav} ​ \left(\right. h_{1} , h_{2} \left.\right) = 1 - \frac{1}{K} ​ \sum_{k = 1}^{K} cos ⁡ \left(\right. 𝐬_{h_{1}}^{\left(\right. k \left.\right)} , 𝐬_{h_{2}}^{\left(\right. k \left.\right)} \left.\right) ,$

where $𝐬_{h}^{\left(\right. k \left.\right)}$ is the priority vector returned by function $h$ on scenario $k$. This metric captures functional divergence in heuristic strategy independent of implementation style. If a function returns probability distributions, distribution distance (Jensen-Shannon or Earth Mover Distance) can alternatively be used.

##### Population Initialization:

Initial heuristics are sampled from a set of canonical bin packing rules (e.g., best-fit, worst-fit, first-fit, combinations thereof). More details can be found in our code repository. Each is evaluated on all instances.

##### Zero-shot Prompt:

See Table[9](https://arxiv.org/html/2604.19440#A3.T9 "Table 9 ‣ Zero-shot Prompt: ‣ C.4 Heuristic Design ‣ Appendix C Task-Specific Experimental Details ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search").

Table 9: Bin Packing Zero-shot Prompt Template

##### Evolution Prompt:

See Table[10](https://arxiv.org/html/2604.19440#A3.T10 "Table 10 ‣ Evolution Prompt: ‣ C.4 Heuristic Design ‣ Appendix C Task-Specific Experimental Details ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search").

Table 10: Bin Packing Evolution Prompt Template

#### C.4.1 Zero-shot Evaluation Details

For each model–task pair, we sampled outputs under six temperature settings ($T \in \left{\right. 0.0 , 0.2 , 0.4 , 0.6 , 0.8 , 1.0 \left.\right}$), with two runs per temperature.

#### C.4.2 Task-agnostic Novelty Computation

Algorithm[1](https://arxiv.org/html/2604.19440#algorithm1 "In C.4.2 Task-agnostic Novelty Computation ‣ C.4 Heuristic Design ‣ Appendix C Task-Specific Experimental Details ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search") details the task-agnostic novelty computation procedure used across all experiments. Novelty is defined as the minimum semantic distance to prior candidates within the same problem instance and normalized to ensure comparability across generations.

Require: task

$T$
, problem instance

$p$
, generation

$g$
, candidate

$a_{p , g}$
, prior set

$\mathcal{A}_{p , g}^{prior}$
, task-specific distance metric

$D_{T}$
, all diversity values

$\mathcal{N}_{p}$
for instance

$p$
, constant

$\epsilon$

Distance Computation:  Compute raw point diversity (nearest neighbor):

$n \leftarrow min_{b \in \mathcal{A}_{p , g}^{prior}} ⁡ D_{T} ​ \left(\right. a_{p , g} , b \left.\right)$

Normalization: Normalize within

$p$
:

$\hat{n} ​ \left(\right. a_{p , g} \left.\right) \leftarrow \frac{n - min ⁡ \left(\right. \mathcal{N}_{p} \left.\right)}{max ⁡ \left(\right. \mathcal{N}_{p} \left.\right) - min ⁡ \left(\right. \mathcal{N}_{p} \left.\right) + \epsilon}$

Return normalized diversity

$\hat{n} ​ \left(\right. a_{p , g} \left.\right) \in \left[\right. 0 , 1 \left]\right.$

Algorithm 1 Task-agnostic novelty computation for each genome

### C.5 Multi-dimensional Scaling Parameters

We use Multi-dimensional Scaling (MDS) to project task-specific novelty distances into 2D embeddings (“trajectory landscapes”) for visualization. The approach is standardized across all four task families, with variations in the distance metric:

##### Sampling and Fitting Strategy:

For each task, we collect all genomes across generations and models. To avoid computational bottlenecks:

1.   1.
If total genomes $N \leq 4000$: fit MDS on the entire distance matrix.

2.   2.
If $N > 4000$: use stratified sampling (max 60 genomes per (model, generation) bucket) to obtain $m \leq 4000$ base points, then use out-of-sample (OOS) placement for remaining points.

##### MDS Solver Parameters:

All experiments use sklearn.manifold.MDS with:

*   •
n_components=2: Project to 2D for visualization.

*   •
dissimilarity="precomputed": Input is precomputed distance matrix.

*   •
n_init=1: Single initialization (fixed random seed ensures reproducibility).

*   •
max_iter=300: Maximum solver iterations.

*   •
eps=1e-3: Convergence tolerance.

*   •
random_state=42: Deterministic seed.

##### Out-of-Sample Placement:

For genomes not in the base fit set, we use k-NN Shepard interpolation in the 2D space:

1.   1.
Compute distance (using task-specific metric) from each OOS point to all $m$ base points.

2.   2.
Find $k = 8$ nearest neighbors (smallest distances).

3.   3.
Assign weights $w_{i} = 1 / \left(\left(\right. d_{i} + 10^{- 8} \left.\right)\right)^{p}$, where $p = 2.0$ and $d_{i}$ is distance to neighbor $i$.

4.   4.
Place OOS point at weighted average of neighbor 2D coordinates.

This approach is fast (vectorized per block of 4000 points) and preserves neighborhood structure in the high-dimensional space.

##### Prompt Optimization Preprocessing:

For prompt optimization, embeddings are high-dimensional ($sim 1536$ dimensions for text-embedding-ada-002). To accelerate MDS on large datasets ($> 10 ​ K$ rows), we first applied PyGlimmerMDS 4 4 4[https://github.com/hageldave/PyGlimmerMDS](https://github.com/hageldave/PyGlimmerMDS) (a GPU-accelerated multilevel MDS variant) on a server, then loaded precomputed 2D coordinates for visualization.

##### Normalization and Scaling:

Fitness values are normalized per task using robust min-max scaling (1st and 99th percentiles), ensuring visual comparability across tasks with different fitness ranges. Generation and fitness are displayed via (i) color (viridis colormap) and (ii) point size, respectively.

## Appendix D Robustness Analyses

### D.1 Temperature-Sensitivity Experiment

A potential concern is that our findings may depend on specific decoding hyperparameters, as temperature directly affects the stochasticity of LLM-generated mutations. We also conduct a temperature-sensitivity analysis on two representative task families—route optimization (TSP) and equation discovery (Oscillator)—using two models with contrasting refinement capabilities (Mistral-7B and Mistral-24B). We vary the decoding temperature over a wide range $T \in \left{\right. 0.0 , 0.1 , 0.3 , 0.5 , 0.7 , 0.9 , 1.1 , 1.3 \left.\right}$, and for each configuration measure both the local refinement rate and the final optimization performance.

Overall, these additional results in Table[11](https://arxiv.org/html/2604.19440#A4.T11 "Table 11 ‣ D.1 Temperature-Sensitivity Experiment ‣ Appendix D Robustness Analyses ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search") and Figure[9](https://arxiv.org/html/2604.19440#A4.F9 "Figure 9 ‣ D.1 Temperature-Sensitivity Experiment ‣ Appendix D Robustness Analyses ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search") that our main finding—that local refinement ability is a key driver of optimization success—is robust to substantial variations in decoding temperature. Rather than being tied to a narrow hyperparameter regime, refinement behavior turns out to be a stable property of the combined system (model, prompt, and decoding configuration).

Task Model Pearson $r$$p$-value
Oscillator Mistral-7B$0.49$$0.209$
Oscillator Mistral-24B$0.32$$0.433$
TSP Mistral-7B$0.76^{*}$$0.027$
TSP Mistral-24B$0.92^{* \llbracket * *}$$0.000$

Table 11:  Pearson correlation between local refinement rate and final performance under temperature variation. The relationship remains consistently positive across settings.

![Image 57: Refer to caption](https://arxiv.org/html/2604.19440v1/img/temp_exp.png)

Figure 9:  Temperature sensitivity of refinement–performance dynamics. While performance varies with temperature, the relationship between refinement and fitness remains stable, particularly on TSP where strong positive correlations are observed. 

### D.2 Perturbation Study Details

![Image 58: Refer to caption](https://arxiv.org/html/2604.19440v1/img/average_refinement_rate_per_model.png)

Figure 10:  Average local refinement rate across models and task families. Mistral-24B exhibits consistently strong and balanced refinement behavior across tasks, aligning with its strong post-optimization performance..

Task Good Refiner Bad Refiner
TSP60 Mistral-24B Mistral-7B
Summarization DeepSeek-V3-Chat GPT-4o-mini
Oscillator-1 GPT-3.5-Turbo Gemma-3n-4B
Bin-Packing-OR3 Mistral-24B LLaMA-3.2-3B-Instruct

Table 12: Models used in the mixing model experiments across tasks. We contrast strong refiners with weaker ones to analyze causal effects of refinement ability.

## Appendix E Statistical Model Specifications

### E.1 OLS Regressions

We employ ordinary least squares (OLS) regression with clustering-robust standard errors to evaluate the predictive power of novelty and breakthrough-rate metrics on final evolutionary performance. All models include task fixed effects to account for within-task heterogeneity (8 tasks × 2 seeds = 16 task instances across 15 LLMs, $N = 119$ model-task pairs).

##### Data and Estimation:

*   •
Sample: Aggregated at the model-task level ($N = 119$ observations: 15 models × 8 tasks, with missing cells for some model-task combinations).

*   •
Dependent variable:$\text{best}_\text{final}_\text{perf}_{z}$: z-score-normalized best final generation fitness per (model, task) pair.

*   •

Covariates (all z-scored for interpretability):

    *   –
$\text{avg}_\text{novelty}_{z}$: Mean within-generation novelty (average distance to nearest prior candidate).

    *   –
$\text{initial}_\text{nov}_{z}$: Initial population novelty (diversity at generation 0).

    *   –
$\text{avg}_\text{breakthrough}_\text{rate}_{z}$: Fraction of generations achieving best-so-far improvement.

    *   –
$\text{zero}_\text{shot}_\text{perf}_{z}$: Average zero-shot performance under temperature-swept setting.

*   •
Errors: Clustered by model (15 clusters) using Huber-White robust covariance estimator to account for within-model correlations across tasks.

*   •
Fixed effects: 8 task indicators (baseline: TSP-30; reference category absorbed in intercept).

##### Model Specifications:

We fit two sets of models to test distinct hypotheses:

Set A: Novelty as Predictor. Regression form: $\text{best}_\text{final}_\text{perf}_{z} sim \text{predictor}_{z} + 𝟏_{\text{task}}$

1.   M1
Predictor = $\text{avg}_\text{novelty}_{z}$ only; tests if exploration (novelty) predicts final performance.

2.   M2
Predictor = $\text{initial}_\text{nov}_{z}$ only; tests if initial diversity is predictive.

3.   M3
Predictor = $\text{zero}_\text{shot}_\text{perf}_{z}$ only; baseline model controlling for base capability.

4.   M4
Predictors = $\text{zero}_\text{shot}_\text{perf}_{z} + \text{avg}_\text{novelty}_{z}$; tests whether novelty adds explanatory power controlling for zero-shot ability.

5.   M5
Predictors = $\text{zero}_\text{shot}_\text{perf}_{z} + \text{initial}_\text{nov}_{z}$; alternative initial-diversity control.

Set B: Breakthrough-Rate as Predictor. Regression form: $\text{best}_\text{final}_\text{perf}_{z} sim \text{predictor}_{z} + 𝟏_{\text{task}}$

1.   M6
Predictor = $\text{avg}_\text{breakthrough}_\text{rate}_{z}$ only; tests if progress (breakthrough frequency) predicts success.

2.   M7
Predictor = $\text{zero}_\text{shot}_\text{perf}_{z}$ only; baseline.

3.   M8
Predictors = $\text{zero}_\text{shot}_\text{perf}_{z} + \text{avg}_\text{breakthrough}_\text{rate}_{z}$; joint model.

##### Results Summary:

See Table[13](https://arxiv.org/html/2604.19440#A5.T13 "Table 13 ‣ Results Summary: ‣ E.1 OLS Regressions ‣ Appendix E Statistical Model Specifications ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search") and Table[14](https://arxiv.org/html/2604.19440#A5.T14 "Table 14 ‣ Results Summary: ‣ E.1 OLS Regressions ‣ Appendix E Statistical Model Specifications ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search")

Table 13: OLS Regression Results: Novelty Predictiveness (Set A)

Model Predictor(s)$\beta$SE$p$-value$R^{2}$Adj.$R^{2}$
M1 avg_novelty_z$- 0.027$$0.073$0.710 0.001$- 0.072$
M2 initial_nov_z$- 0.042$$0.102$0.681 0.002$- 0.071$
M3 zero_shot_perf_z$0.322^{*}$$0.134$0.016 0.103 0.038
M4 ZS + Avg Novelty ZS: $0.322^{*}$$0.138$0.019 0.103 0.029
Nov: $0.002$$0.072$0.980
M5 ZS + Init Novelty ZS: $0.324^{*}$$0.132$0.014 0.106 0.033
Init: $- 0.055$$0.063$0.387

Table 14: OLS Regression Results: Breakthrough-Rate Predictiveness (Set B)

Model Predictor(s)$\beta$SE$p$-value$R^{2}$Adj.$R^{2}$
M6 avg_breakthrough_rate_z$0.445^{* \llbracket * *}$$0.097$$< 0.001$0.198 0.139
M7 zero_shot_perf_z$0.322^{*}$$0.134$0.016 0.103 0.038
M8 ZS + Breakthrough ZS: $0.226$$0.127$0.076 0.246 0.184
BR: $0.389^{* \llbracket * *}$$0.087$$< 0.001$

$\_{}^{*}p < 0.05$, $\_{}^{ * *}p < 0.01$, $\_{}^{* \llbracket * *}p < 0.001$. All predictors are z-scored. Standard errors are clustered by model (robust to heteroskedasticity and within-model correlation). Task fixed effects included but not shown.

##### Implications:

The weak explanatory power of novelty metrics (Set A) contrasts sharply with the strong predictiveness of breakthrough-rate metrics (Set B), supporting that models that generate diverse genomes do not necessarily improve faster. Instead, the ability to drive consistent fitness improvements (characterized by breakthrough frequency) is the key differentiator across LLMs.

### E.2 Mixed-Effects Regression Models

We employ generalized linear mixed-effects models (GLMM) to analyze breakthrough probability at the generation level, accounting for model-level heterogeneity via random intercepts. We fit two specifications: (1) Concurrent Model: same-generation predictors, and (2) Lagged Model: current-generation predictors forecasting next-generation breakthrough probability.

##### Data and Estimation:

*   •
Concurrent sample:$N = 3 , 570$ generation-level observations across 15 LLMs (groups), with variable group sizes (min 209, max 242, mean 238). Data from all 8 tasks × 30 generations × 15 models, subset to generations with complete novelty and entropy measurements.

*   •
Lagged sample:$N = 3 , 451$ observations (omitting final generation of each model-task pair, which has no $t + 1$ outcome).

*   •
Dependent variable (concurrent):$\text{prob}_\text{breakthrough}_{z}$: z-scored fraction of offspring achieving best-so-far improvement in generation $g$ (binary: 0/1 per generation, then aggregated).

*   •
Dependent variable (lagged):$\text{prob}_\text{breakthrough}_{z , t + 1}$: next-generation breakthrough probability (lead variable).

*   •

Fixed effects (all z-scored for comparability):

    *   –
$H_{\text{fitness} , z}$: Fitness-weighted spatial entropy (concentration of high-fitness mass).

    *   –
$H_{\text{spatial} , z}$: Uniform-weighted spatial entropy (semantic dispersion).

    *   –
$\text{mean}_\text{novelty}_\text{per}_\text{gen}_{z}$: Average within-generation novelty.

    *   –
$\text{max}_\text{novelty}_\text{per}_\text{gen}_{z}$: Maximum novelty in generation.

    *   –
$\text{mean}_\text{novelty}_\text{per}_\text{gen}_{z} \times H_{\text{spatial} , z}$: Interaction term capturing interference between exploration and dispersion.

    *   –
$\text{generation}_{z}$: z-scored generation index (time control).

    *   –
8 task indicators (baseline: TSP-30; reference absorbed in intercept).

*   •
Random effects: Model-level random intercept, $u_{\text{model}} sim \mathcal{N} ​ \left(\right. 0 , \tau^{2} \left.\right)$, allowing breakthrough propensity to vary across LLMs.

*   •
Estimation: Maximum likelihood (ML, not REML) to permit likelihood ratio testing between models.

##### Model Formulation:

Concurrent Model. For generation $g$ of model $m$ on task $t$:

$\text{prob}_\text{breakthrough}_{g , m , t , z} =$$\beta_{0} +$
$\beta_{1} ​ H_{\text{fitness} , z} + \beta_{2} ​ H_{\text{spatial} , z} +$
$\beta_{3} ​ \left(\bar{\text{nov}}\right)_{z} + \beta_{4} ​ \text{max}_\text{nov}_{z} +$
$\beta_{5} ​ \left(\right. \left(\bar{\text{nov}}\right)_{z} \times H_{\text{spatial} , z} \left.\right)$
$+ \gamma_{t} ​ 𝟏_{t} + \beta_{6} ​ \text{gen}_{z} + u_{m} + \epsilon_{g , m , t} .$(1)

Lagged Model. Using generation $g$ predictors to forecast generation $g + 1$ outcomes:

$\text{prob}_\text{breakthrough}_{g + 1 , m , t , z} =$$\beta_{0}^{\text{lag}} +$
$\beta_{1}^{\text{lag}} ​ H_{\text{fitness} , z} ​ \left(\right. g \left.\right) + \beta_{2}^{\text{lag}} ​ H_{\text{spatial} , z} ​ \left(\right. g \left.\right)$
$+ \beta_{3}^{\text{lag}} ​ \left(\bar{\text{nov}}\right)_{z} + \beta_{4}^{\text{lag}} ​ \text{max}_\text{nov}_{z}$
$+ \beta_{5}^{\text{lag}} ​ \left(\right. \left(\bar{\text{nov}}\right)_{z} ​ \left(\right. g \left.\right) \times H_{\text{spatial} , z} ​ \left(\right. g \left.\right) \left.\right)$
$+ \gamma_{t}^{\text{lag}} ​ 𝟏_{t} + \beta_{6}^{\text{lag}} ​ \text{gen}_{z} ​ \left(\right. g \left.\right)$
$+ u_{m}^{\text{lag}} + \epsilon_{g + 1 , m , t} .$(2)

##### Results Summary:

See Table[15](https://arxiv.org/html/2604.19440#A5.T15 "Table 15 ‣ Results Summary: ‣ E.2 Mixed-Effects Regression Models ‣ Appendix E Statistical Model Specifications ‣ What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search")

Table 15: Mixed-Effects Models: Concurrent vs.Lagged Breakthrough Prediction

Fixed Effect Concurrent Lagged (+1 Gen)
$\hat{\beta}$SE$p$$\hat{\beta}$SE$p$
Main Effects
$H_{\text{fitness} , z}$$- 0.073$$0.026$0.005$- 0.074$$0.024$0.002
$H_{\text{spatial} , z}$$- 0.015$$0.024$0.532$0.012$$0.022$0.593
$\text{mean}_\text{novelty}_{z}$$0.070^{ * *}$$0.026$0.006$0.016$$0.025$0.517
$\text{max}_\text{novelty}_{z}$$0.029$$0.023$0.202$0.006$$0.022$0.787
Interaction
$\text{novelty} \times H_{\text{spatial}}$$- 0.090^{* \llbracket * *}$$0.010$$< 0.001$$- 0.051^{* \llbracket * *}$$0.009$$< 0.001$
Temporal
$\text{generation}_{z}$$- 0.250^{* \llbracket * *}$$0.018$$< 0.001$$- 0.193^{* \llbracket * *}$$0.017$$< 0.001$
Model Statistics
No.Observations 3,570 3,451
No.Groups (models)15 15
Log-Likelihood$- 4639.77$$- 4203.99$
Residual Variance$0.7798$$0.6621$
Random Intercept Var$0.034$$0.033$

$\_{}^{*}p < 0.05$, $\_{}^{ * *}p < 0.01$, $\_{}^{* \llbracket * *}p < 0.001$. All predictors z-scored. Task fixed effects included but not separately reported. SE = model-level clustered standard error. Random intercept allows LLM-specific deviation from population mean.

##### Temporal Dynamics:

The lagged model reveals that current-generation state is a weak predictor of next-generation breakthroughs (residual variance $0.662$ vs.concurrent $0.780$, suggesting some temporal structure but substantial noise). The interaction term remains the strongest signal across both timescales, suggesting the interference effect is a robust mechanistic feature of LLM-guided evolution, not merely a concurrent correlation.

## Appendix F Supplementary Visualizations

![Image 59: Refer to caption](https://arxiv.org/html/2604.19440v1/img/novelty_distribution.png)

Figure 11: Breakthrough and Non Breakthrough’s distribution for all collected trajectories

![Image 60: Refer to caption](https://arxiv.org/html/2604.19440v1/img/real_data_landscape_breakthroughs.png)

Figure 12: Interaction between novelty and spatial entropy in breakthrough dynamics. Each cell reports the empirical breakthrough probability aggregated over generations falling into the corresponding bins of mean novelty and spatial entropy(z-scored). Color intensity indicates higher likelihood of breakthroughs.

![Image 61: Refer to caption](https://arxiv.org/html/2604.19440v1/img/task_performance_scatter.png)

Figure 13: Zero-shot Performance Versus Post-Optimization performance for each task

![Image 62: Refer to caption](https://arxiv.org/html/2604.19440v1/img/general_novelty_analysis.png)

Figure 14: Interaction between breakthroughs and Novelty

![Image 63: Refer to caption](https://arxiv.org/html/2604.19440v1/img/combined_view_rq2_by_4_tasks.png)

Figure 15: Cost-efficiency plots for four task families

![Image 64: Refer to caption](https://arxiv.org/html/2604.19440v1/img/combined_vew_novelty_lineplot.png)

Figure 16: Novelty and fitness coevolution line-plots aggregated over tasks (exploration–exploitation tension)

