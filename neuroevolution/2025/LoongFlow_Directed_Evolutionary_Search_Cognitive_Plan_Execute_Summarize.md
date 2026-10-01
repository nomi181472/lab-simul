Title: LoongFlow: Directed Evolutionary Search via a Cognitive Plan-Execute-Summarize Paradigm

URL Source: https://arxiv.org/html/2512.24077

Markdown Content:
Chunhui Wan Xunan Dai*Zhuo Wang*Minglei Li*Yanpeng Wang*Yinan Mao Yu Lan Zhiwen Xiao Baidu Inc

###### Abstract

The transition from static Large Language Models (LLMs) to self-improving agents is hindered by the lack of structured reasoning in traditional evolutionary approaches. Existing methods often struggle with premature convergence and inefficient exploration in high-dimensional code spaces. To address these challenges, we introduce LoongFlow, a self-evolving agent framework that achieves state-of-the-art solution quality with significantly reduced computational costs. Unlike "blind" mutation operators, LoongFlow integrates LLMs into a cognitive "Plan-Execute-Summarize" (PES) paradigm, effectively mapping the evolutionary search to a reasoning-heavy process. To sustain long-term architectural coherence, we incorporate a hybrid evolutionary memory system. By synergizing Multi-Island models with MAP-Elites and adaptive Boltzmann selection, this system theoretically balances the exploration-exploitation trade-off, maintaining diverse behavioral niches to prevent optimization stagnation. We instantiate LoongFlow with a General Agent for algorithmic discovery and an ML Agent for pipeline optimization. Extensive evaluations on the AlphaEvolve benchmark and Kaggle competitions demonstrate that LoongFlow outperforms leading baselines (e.g., OpenEvolve, ShinkaEvolve) by up to 60% in evolutionary efficiency while discovering superior solutions. LoongFlow marks a substantial step forward in autonomous scientific discovery, enabling the generation of expert-level solutions with reduced computational overhead.

Code:[https://github.com/baidu-baige/LoongFlow](https://github.com/baidu-baige/LoongFlow)

1 Introduction
--------------

The progression from static prompting—where humans manually engineer instructions—to autonomous, self-evolving agents marks a fundamental shift in artificial intelligence. While static approaches rely on fixed inference patterns, self-evolving agents utilize Large Language Models (LLMs) as mutation operators to iteratively modify their own code or parameters. Pioneering works have validated this paradigm in specific domains: FunSearch[romera2024mathematical] utilizes LLMs to discover novel mathematical constructions, Eureka[ma2024eureka] optimizes reward functions via evolutionary search, and AlphaEvolve[novikov2025alphaevolve] automates the discovery of heuristic algorithms. that LLMs, building on foundational code-generation capabilities[chen2021codex, li2022alphacode], can discover novel mathematical algorithms and reward functions that surpass human baselines. This "Darwinian shift" has established automated scientific discovery as a vibrant research frontier.

However, as the complexity of tasks increases, current frameworks face severe cognitive and architectural limitations. Leading open-source baselines, such as OpenEvolve and ShinkaEvolve[sakana2025shinkaevolve], effectively treat the LLM as a stochastic black box. OpenEvolve relies on high-volume random mutations, leading to a "random walk" behavior that is computationally prohibitive. ShinkaEvolve improves efficiency via novelty search but operates purely at the execution level, lacking a mechanism to analyze why a mutation failed. Consequently, these methods hit a "cognitive ceiling," struggling to maintain structural coherence over long evolutionary horizons. Specifically, they encounter three critical bottlenecks:

*   •Inefficient Exploration (The Cost Bottleneck): Existing agents lack a strategic planning layer. They engage in brute-force sampling in high-dimensional code spaces, resulting in excessive token consumption and unstable convergence rates. 
*   •Diversity Collapse (The Convergence Bottleneck): Without explicit diversity management, population-based agents tend to converge prematurely to local optima. Traditional "Top-K" sampling fails to preserve diverse but potentially high-reward "stepping stone" solutions. 
*   •Absence of Reflexive Memory (The Feedback Bottleneck): Unlike deep learning, where backpropagation provides a precise gradient for improvement, most existing evolutionary agent frameworks lack a structured reflection mechanism[shinn2023reflexion]. They function as "memory-less" searchers, repeating similar errors across generations rather than accumulating "evolutionary wisdom" through structured summarization. 

![Image 1: Refer to caption](https://arxiv.org/html/2512.24077v1/LoongFlow-overview.jpg)

Figure 1: Overview of LoongFlow.

To overcome these barriers, we introduce LoongFlow, a framework designed to bridge the gap between reasoning agents and evolutionary computation. LoongFlow distinguishes itself through two core architectural innovations. First, we propose the "Plan-Execute-Summarize" (PES) paradigm. This cognitive loop transforms random mutation into a directed hypothesis-testing process (addressing Bottleneck 1 & 3), as illustrated in the Agent Loop of Figure[1](https://arxiv.org/html/2512.24077v1#S1.F1 "Figure 1 ‣ 1 Introduction ‣ LoongFlow: Directed Evolutionary Search via a Cognitive Plan-Execute-Summarize Paradigm"). Second, to resolve the exploration-exploitation dilemma (Bottleneck 2), we design a Hybrid Evolutionary Memory. By fusing the spatial isolation of Island Models with the behavioral diversity of MAP-Elites[mouret2015illuminating] and entropy-regularized Boltzmann selection, LoongFlow dynamically maintains diverse behavioral niches. This ensures that the system can escape local optima and continuously discover novel solution architectures.

We demonstrate the versatility of LoongFlow by instantiating two domain-specific agents: a General Agent for algorithmic tasks and an ML Agent for machine learning pipelines. Experimental results on the AlphaEvolve benchmark and Kaggle competitions confirm that LoongFlow significantly surpasses OpenEvolve and ShinkaEvolve, breaking theoretical performance barriers with superior sample efficiency.

The primary contributions of this work are as follows:

*   •Structured Evolutionary Paradigm: We propose the “Plan-Execute-Summarize” paradigm, which integrates expert-level planning and retrospective summarization to reduce generation randomness and establish a sustainable feedback loop. 
*   •Advanced Memory Architecture: We design a domain-adaptive evolutionary memory that combines multi-island parallel evolution with MAP-Elites[mouret2015illuminating] and adaptive Boltzmann selection, effectively solving the premature convergence and “catastrophic forgetting” problems inherent in traditional LLM agents. 
*   •Superior Performance & Efficiency: We provide open-source, pre-built agents (GeneralAgent and MLAgent) that achieve state-of-the-art results on NP-hard mathematical problems and complex ML pipelines, surpassing existing frameworks in both stability and evolutionary speed. 

2 Related Work
--------------

The development of LoongFlow is situated at the intersection of LLM-based Evolutionary Optimization and Cognitive Agent Architectures. In this section, we review the progression of these fields and identify the specific gap that LoongFlow addresses.

### 2.1 LLM-Based Evolutionary Optimization

The paradigm of utilizing LLMs for evolutionary optimization has shifted from simple solution generation to iterative refinement. Pioneering work such as FunSearch[romera2024mathematical] demonstrated that LLMs, when coupled with an evolutionary evaluator, could solve open problems in mathematics (e.g., the Cap Set problem) by searching for "functions" rather than parameters. Similarly, AlphaEvolve[novikov2025alphaevolve] orchestrates an autonomous pipeline of LLMs to improve an algorithm by making direct changes to the program, achieving human-level performance on robot manipulation tasks. While recent methods like OPRO[yang2023opro] and PromptBreeder[fernando2023promptbreeder] have explored using LLMs as optimizers, they often treat the model as a black-box operator to mutation. They typically treat the LLM as a stochastic operator—randomly mutating code without a high-level strategy—which leads to high token costs and inefficient exploration in complex search spaces.

### 2.2 Evolutionary Agent Frameworks

To engineeringly scale LLM-based evolution, several open-source frameworks have emerged. These serve as the primary baselines for our work:

*   •OpenEvolve: As a standard implementation of the AlphaEvolve algorithm, OpenEvolve utilizes an "Island Model" to maintain population diversity. However, it treats code generation as a single-step translation task. The mutation process is largely reactive, where the agent fixes errors or makes random local changes without understanding the global algorithmic structure. This often leads to a "random walk" behavior in high-difficulty tasks. 
*   •ShinkaEvolve: ShinkaEvolve[sakana2025shinkaevolve] improves sample efficiency by integrating "code-novelty rejection" to filter out redundant solutions before execution. While it reduces computational waste, it still operates primarily at the execution level. The framework lacks a structured reflection mechanism to analyze why a specific architectural change failed, preventing the system from learning abstract principles over long evolutionary horizons. 

### 2.3 Cognitive Architectures and The Reasoning Gap

While evolutionary methods excel at population-based search, they often lack the depth of semantic reasoning found in autonomous agents.

Reasoning Agents: Frameworks like ReAct[yao2023react] and Reflexion[shinn2023reflexion] have demonstrated that interleaving reasoning traces (Thought) with actions significantly improves problem-solving capabilities. These methods enable agents to perform multi-step planning and self-correction. Similarly, Voyager[wang2023voyager] utilizes an iterative curriculum to learn complex skills in embodied environments.

The Gap: A critical gap exists in merging these two paradigms. Standard reasoning agents (like AutoGPT[richards2023autogpt] or Voyager[wang2023voyager]) generally focus on single-instance problem solving rather than population-based evolutionary search. Conversely, traditional evolutionary algorithms (like MAP-Elites[mouret2015illuminating]) excel at maintaining diverse populations but lack the semantic reasoning capabilities of ReAct-style agents.

LoongFlow bridges this gap by introducing the "Plan-Execute-Summarize" paradigm. This paradigm allows LoongFlow to maintain the diversity benefits of MAP-Elites[mouret2015illuminating] while leveraging the reasoning depth of ReAct-style agents[yao2023react], effectively moving the evolutionary process from "random mutation" to "directed evolution".

3 Background
------------

In this section, we formally frame the open-ended evolutionary process as a sequential decision-making problem and establish the mathematical foundations of the LoongFlow framework. We model the self-evolution of agents as a Markov Decision Process (MDP[sutton2018reinforcement]) over a discrete code space, guided by a parameterized Large Language Model (LLM).

### 3.1 Problem Formulation as MDP

We define the problem as a tuple ⟨𝒞,𝒜,R,π θ⟩\langle\mathcal{C},\mathcal{A},R,\pi_{\theta}\rangle:

*   •State/Code Space (𝒞\mathcal{C}): Let 𝒞\mathcal{C} be the infinite, discrete space of all valid programs in a specific language (e.g., Python). A state s t∈𝒞 s_{t}\in\mathcal{C} represents the solution code at evolutionary generation t t. 
*   •Action Space (𝒜\mathcal{A}): The action space consists of semantic modification operations (e.g., rewrite, debug, optimize) applied to the code. 
*   •Reward Function (R R): R:𝒞→ℝ R:\mathcal{C}\to\mathbb{R} is a scalar fitness function (e.g., accuracy on test cases). The environment is characterized by a sparse reward signal, where valid solutions are rare. 
*   •Policy (π θ\pi_{\theta}): The agent is an LLM parameterized by weights θ\theta. It acts as a stochastic policy π θ​(a|s)\pi_{\theta}(a|s), generating the next code state s t+1 s_{t+1} based on the current state s t s_{t} and context. 

Our objective is to find an optimal solution s∗s^{*} that maximizes the reward:

s∗=arg⁡max s∈𝒞 R​(s)s^{*}=\mathop{\arg\max}_{s\in\mathcal{C}}R(s)(1)

### 3.2 LLM as a Composite Semantic Operator

Unlike traditional Evolutionary Algorithms (EA) that use fixed, random mutation operators (denoted as 𝒯 m​u​t\mathcal{T}_{mut}), LoongFlow utilizes the LLM as a learnable Semantic Operator.

We formalize the "Plan-Execute-Summarize" (PES) paradigm as a composite transition kernel decomposing the policy π θ\pi_{\theta} into three sub-steps. Let ℳ t\mathcal{M}_{t} be the evolutionary memory at generation t t, and ℐ\mathcal{I} be the set of system instructions (prompts). The transition from parent s t s_{t} to offspring s t+1 s_{t+1} proceeds as follows:

1.   1.Planning: The Planner generates a natural language blueprint b b (an intermediate latent variable) conditioned on the parent code s t s_{t} and retrieved insights from memory ℳ t\mathcal{M}_{t}:

b∼π θ​(b∣s t,ℳ t,ℐ p​l​a​n)b\sim\pi_{\theta}(b\mid s_{t},\mathcal{M}_{t},\mathcal{I}_{plan})(2) 
2.   2.Execution: The Executor generates the executable offspring code s′s^{\prime} (a candidate for s t+1 s_{t+1}) based on the blueprint b b:

s′∼π θ​(s′∣b,s t,ℐ e​x​e​c)s^{\prime}\sim\pi_{\theta}(s^{\prime}\mid b,s_{t},\mathcal{I}_{exec})(3) 
3.   3.Summarization & Update: The Summarizer generates a reflection insight z z based on the execution feedback r=R​(s′)r=R(s^{\prime}), and updates the memory:

z∼π θ​(z∣s′,r,b,ℐ s​u​m)z\sim\pi_{\theta}(z\mid s^{\prime},r,b,\mathcal{I}_{sum})(4)

ℳ t+1←ℳ t∪{z}\mathcal{M}_{t+1}\leftarrow\mathcal{M}_{t}\cup\{z\}(5) 

### 3.3 Feature Space and Archive Management

To manage population diversity beyond raw fitness, we map the high-dimensional code space 𝒞\mathcal{C} to a lower-dimensional Feature Space ℱ⊆ℝ k\mathcal{F}\subseteq\mathbb{R}^{k}.

##### Feature Mapping.

Let Φ:𝒞→ℱ\Phi:\mathcal{C}\to\mathcal{F} be a mapping function that projects a solution s s to a feature vector 𝐯=Φ​(s)\mathbf{v}=\Phi(s). In this work, 𝐯\mathbf{v} consists of interpretable dimensions, for example, 𝐯=(Cyclomatic Complexity,Code Length)\mathbf{v}=(\text{Cyclomatic Complexity},\text{Code Length}).

##### MAP-Elites Archive.

We maintain a structured archive (Memory) 𝒜 r​c​h​i​v​e\mathcal{A}_{rchive}, discretized into a grid of cells in ℱ\mathcal{F}. Each cell, indexed by a feature vector 𝐯\mathbf{v}, stores only the single best solution found so far for that specific behavior:

𝒜 r​c​h​i​v​e​(𝐯)={s∈𝒞∣Φ​(s)∈Cell​(𝐯)∧R​(s)=max s′∈Cell​(𝐯)⁡R​(s′)}\mathcal{A}_{rchive}(\mathbf{v})=\{s\in\mathcal{C}\mid\Phi(s)\in\text{Cell}(\mathbf{v})\land R(s)=\max_{s^{\prime}\in\text{Cell}(\mathbf{v})}R(s^{\prime})\}(6)

This mechanism ensures behavioral diversity, preventing the policy from collapsing into a single local optimum.

### 3.4 Adaptive Boltzmann Selection

To select the parent s t s_{t} for the next generation from the archive 𝒜 r​c​h​i​v​e\mathcal{A}_{rchive}, we replace static greedy selection with Adaptive Boltzmann Selection.

Let {s 1,s 2,…,s N}\{s_{1},s_{2},\dots,s_{N}\} be the set of solutions currently stored in the archive. The probability P​(s i)P(s_{i}) of selecting solution s i s_{i} as the parent is:

P​(s i)=exp⁡(R​(s i)/τ)∑j=1 N exp⁡(R​(s j)/τ)P(s_{i})=\frac{\exp(R(s_{i})/\tau)}{\sum_{j=1}^{N}\exp(R(s_{j})/\tau)}(7)

where τ\tau is a temperature parameter dynamically modulated by the population entropy. This allows LoongFlow to shift smoothly between exploration (high τ\tau) and exploitation (low τ\tau).

4 LoongFlow Overview
--------------------

Designing an evolutionary agent capable of solving high-difficulty, open-ended tasks requires overcoming two fundamental systemic contradictions: the tension between search space complexity and sampling efficiency, and the trade-off between population diversity and convergence speed.

Algorithm 1 LoongFlow Main Evolutionary Loop

1:Input: Task Description

T T
, Initial Solution

s 0 s_{0}
, Max Iterations

N m​a​x N_{max}
, Islands

K K

2:Output: Best Solution

s∗s^{*}

3:// Initialization phase

4:Initialize Global Memory

ℳ←{s 0}\mathcal{M}\leftarrow\{s_{0}\}

5:Initialize

K K
Islands with MAP-Elites Archives

𝒜 1,…,𝒜 K\mathcal{A}_{1},\dots,\mathcal{A}_{K}

6:for

i​t​e​r​a​t​i​o​n=1 iteration=1
to

N m​a​x N_{max}
do

7:for

k=1 k=1
to

K K
do⊳\triangleright Parallel Evolution on Islands

8:// 1. Adaptive Selection (Sec. 4.2.3)

9:

H k←CalculateEntropy​(𝒜 k)H_{k}\leftarrow\text{CalculateEntropy}(\mathcal{A}_{k})

10:

τ←τ b​a​s​e⋅(1+α​e−β​H k)\tau\leftarrow\tau_{base}\cdot(1+\alpha e^{-\beta H_{k}})
⊳\triangleright Dynamic Temperature

11:

s p​a​r​e​n​t←BoltzmannSelect​(𝒜 k,τ)s_{parent}\leftarrow\text{BoltzmannSelect}(\mathcal{A}_{k},\tau)

12:// 2. Lineage-Based Planning (Sec. 4.1.1)

13:

c h a i n←GetLineage(s p​a​r​e​n​t.id)chain\leftarrow\text{GetLineage}(s_{parent}.\text{id})

14:

c​o​n​t​e​x​t←{p.plan,p.summary∣p∈c​h​a​i​n}context\leftarrow\{\text{p.plan},\text{p.summary}\mid p\in chain\}

15:

p​l​a​n←Planner​(s p​a​r​e​n​t,c​o​n​t​e​x​t,T)plan\leftarrow\text{Planner}(s_{parent},context,T)

16:// 3. Execution & Evaluation (Sec. 4.1.2)

17:

c​o​d​e←Executor​(p​l​a​n,s p​a​r​e​n​t)code\leftarrow\text{Executor}(plan,s_{parent})

18:if

Verify​(c​o​d​e)\text{Verify}(code)
is False then

19:continue⊳\triangleright Fast-fail on syntax errors

20:end if

21:

s​c​o​r​e,l​o​g​s←Evaluator​(c​o​d​e)score,logs\leftarrow\text{Evaluator}(code)

22:// 4. Reflection & Storage (Sec. 4.1.3)

23:

s​u​m​m​a​r​y←Summarizer​(p​l​a​n,c​o​d​e,l​o​g​s)summary\leftarrow\text{Summarizer}(plan,code,logs)

24:

s n​e​w←Solution(c o d e,s c o r e,s u m m a r y,p a r e n t=s p​a​r​e​n​t.id)s_{new}\leftarrow\text{Solution}(code,score,summary,parent=s_{parent}.\text{id})

25:

UpdateMAPElites​(𝒜 k,s n​e​w)\text{UpdateMAPElites}(\mathcal{A}_{k},s_{new})

26:end for

27:// 5. Migration Strategy (Sec. 4.2.1)

28:if

i t e r a t i o n mod M==0 iteration\mod M==0
then

29:MigrateElites(

𝒜 1,…,𝒜 K\mathcal{A}_{1},\dots,\mathcal{A}_{K}
)

30:end if

31:end for

32:return

max s∈∪𝒜 k⁡s.s​c​o​r​e\max_{s\in\cup\mathcal{A}_{k}}s.score

To resolve these, LoongFlow introduces a hierarchical architecture that decouples “Cognitive Reasoning” from “Evolutionary Dynamics”. The framework consists of two coupled subsystems: the Agent Loop, which implements the “Plan-Execute-Summarize” (PES) paradigm, and the Hybrid Evolutionary Memory, which governs population management. The overall procedure is outlined in Algorithm[1](https://arxiv.org/html/2512.24077v1#alg1 "Algorithm 1 ‣ 4 LoongFlow Overview ‣ LoongFlow: Directed Evolutionary Search via a Cognitive Plan-Execute-Summarize Paradigm").

### 4.1 The “Plan-Execute-Summarize” (PES) Paradigm

Standard LLM-based evolutionary methods (e.g., genetic programming with LLMs) typically treat the model as a “black-box mutation operator”, randomly perturbing solutions in hopes of improvement. This approach suffers from extreme sample inefficiency and a lack of directional guidance. To address this, LoongFlow formalizes the evolutionary iteration as a structured cognitive process composed of three specialized stages.

![Image 2: Refer to caption](https://arxiv.org/html/2512.24077v1/LoongFlow-frame.jpg)

Figure 2: Expanded view of the LoongFlow evolutionary process. The framework iterates through a Planner-Executor-Summarizer loop. The Planner retrieves historical insights to prune the search space; the Executor generates and verifies code; the Summarizer extracts causal knowledge to update the Evolutionary Memory.

#### 4.1.1 Planner: Strategic Search Space Pruning

In infinite solution spaces, navigating via stochastic mutation often devolves into a “random walk,” wasting vast computational resources on invalid or redundant trials. To mitigate this, the Planner functions as a strategic architect employing Lineage-Based Context Retrieval.

Unlike RAG[lewis2020rag] systems that rely on fuzzy semantic similarity, LoongFlow utilizes the explicit genealogical links inherent in the evolutionary process. As defined in the Solution data structure (see Listing[1](https://arxiv.org/html/2512.24077v1#LST1 "Listing 1 ‣ Solution Data Structure. ‣ 4.2 Hybrid Evolutionary Memory System ‣ 4 LoongFlow Overview ‣ LoongFlow: Directed Evolutionary Search via a Cognitive Plan-Execute-Summarize Paradigm")), each individual preserves its lineage via parent_id.

For a given parent solution s t s_{t}, the Planner traverses the ID chain (retrieving ancestors s t−1,s t−2​…s_{t-1},s_{t-2}... and potential descendants). It extracts the historical generate_plan (Original Intent) and summary (Retrospective Feedback) from this lineage.

*   •Intent Tracking: By reading past plans, the Planner understands the “research trajectory” intended by previous generations. 
*   •Course Correction: By reading past summaries (which contain specific advice for the next generation), the Planner identifies verified pitfalls to avoid. 

This structured recall allows the Planner to construct a context-aware blueprint b b, leveraging the Chain-of-Thought[wei2022chain] reasoning capabilities of modern LLMs, ensuring that the new plan is a logical continuation and refinement of the parent’s strategy, rather than a random jump.

#### 4.1.2 Executor: Polymorphic Implementation & Robust Verification

The translation from a high-level strategic blueprint to an executable solution is inherently non-deterministic and error-prone. The Executor acts as a robust translation engine that converts the Planner’s intent (b b) into verified artifacts (r r).

##### Polymorphic Execution Strategies.

The framework decouples the “What” (Plan) from the “How” (Execution). The Executor supports Pluggable Execution Flows tailored to the problem domain. For algorithmic tasks, it may instantiate as a logic-intensive single-pass coder; for system tasks, it may operate as a multi-stage workflow engine. This design ensures that LoongFlow is not limited to a single class of problems but is adaptable to any domain with a definable action space.

##### Local Verification Loop (Fast-Fail).

Before submitting to the global Evaluator, the Executor interacts with the Environment Interface to perform “Pre-Evaluation Checks”. This local feedback loop allows the Executor to self-correct minor errors (such as syntax typos or import errors) immediately, acting as a filter that prevents low-quality candidates from consuming expensive global evaluation resources.

#### 4.1.3 Summary: Closing the Feedback Loop

Traditional evolutionary algorithms are “memory-less” regarding causality—they know that a solution failed, but not why. This leads to “Cyclical Errors,” where the population repeatedly explores the same invalid dead-ends. The Summary module introduces a retrospective learning mechanism.

After evaluation, the Summarizer performs Abductive Reflection[shinn2023reflexion]. It compares the Planner’s intent (b b) with the execution result (r r) to infer causal relationships and generates a structured Insight (z z). These insights are stored in the Evolutionary Memory. This establishes a Long-Term Cognitive Memory, inspired by the memory architectures in Generative Agents[park2023generative], but adapted for evolutionary lineage.. By feeding these insights back to future Planners, LoongFlow achieves Meta-Learning, where the system becomes “smarter” about the domain constraints over generations.

### 4.2 Hybrid Evolutionary Memory System

A critical failure mode in evolutionary agents is Premature Convergence. LoongFlow addresses this via a multi-layered memory architecture rooted in a structured data schema.

##### Solution Data Structure.

To support the genealogical retrieval described above, LoongFlow maintains a rigorous data schema for every individual in the population. As shown in Listing[1](https://arxiv.org/html/2512.24077v1#LST1 "Listing 1 ‣ Solution Data Structure. ‣ 4.2 Hybrid Evolutionary Memory System ‣ 4 LoongFlow Overview ‣ LoongFlow: Directed Evolutionary Search via a Cognitive Plan-Execute-Summarize Paradigm"), the Solution class encapsulates not only the code but also the full evolutionary metadata (lineage IDs, plans, summaries, and metrics).

Listing 1: The Solution Data Structure in Evolutionary Memory

class Solution:

"""Represent a solution in the memory."""

solution:str=""

solution_id:str=""

generate_plan:str=""

parent_id:Optional[str]=""

island_id:Optional[int]=0

iteration:Optional[int]=0

timestamp:float=field(default_factory=time.time)

generation:int=0

sample_cnt:int=0

sample_weight:float=0.0

score:Optional[float]=0.0

evaluation:Optional[str]=""

summary:str=""

metadata:Dict[str,Any]=field(default_factory=dict)

This comprehensive schema transforms the memory from a simple “High Score List” into a Structured Knowledge Graph, enabling the Planner to query causal relationships (Why did parent X fail?) rather than just outcomes.

#### 4.2.1 Multi-Island Distributed Topology

Single-population models are prone to “dominance,” where one successful strategy outcompetes all others. LoongFlow employs a Multi-Island Model[whitley1994cellular] with a Ring Topology. The population is partitioned into N N isolated islands. Each island evolves independently, allowing distinct algorithmic “species” to cultivate. Migration occurs only when the diversity difference Δ​D\Delta D between neighbors exceeds a threshold. The top k%k\% elites are copied to adjacent islands, acting as “invasive species” to shake up stagnation. This spatial isolation ensures Global Diversity Maintenance, preventing the system from getting stuck in local optima.

#### 4.2.2 MAP-Elites with Feature Grids

Objective-based selection often discards novel but unpolished solutions (“stepping stones”). Within each island, LoongFlow utilizes a MAP-Elites[mouret2015illuminating] container. Solutions are mapped to a feature grid 𝒜\mathcal{A} based on behavioral descriptors Φ​(s)\Phi(s) (e.g., Code Complexity ×\times Memory Usage). The system preserves the best individual for each cell in the grid, not just the global best. This guarantees Niche Preservation, providing a diverse “gene pool” for the Planner to cross-pollinate.

#### 4.2.3 Adaptive Boltzmann Selection

The balance between Exploration and Exploitation is dynamic. LoongFlow implements Entropy-Regularized Boltzmann Selection[thierens1999scalability]. The selection temperature τ\tau is dynamically adjusted based on the population entropy H​(𝒫)H(\mathcal{P}):

τ​(t)∝exp⁡(−λ⋅H​(𝒫 t))\tau(t)\propto\exp(-\lambda\cdot H(\mathcal{P}_{t}))(8)

When the population is diverse (High H H), τ\tau lowers to encourage Exploitation (Greedy). When the population converges (Low H H), τ\tau rises to force Exploration (Random). This achieves Self-Adaptive Control, automatically transitioning between “searching for new ideas” and “polishing existing ones” without human intervention.

5 Experiments
-------------

To empirically validate the Eadfent framework, we conducted a comprehensive evaluation focusing on Effectiveness (Solution Quality) and Efficiency (Convergence Speed).

### 5.1 Experimental Setup

Benchmarks: We instantiated two domain-specific agents—General Agent for algorithmic discovery and Machine Learning Agent for machine learning engineering—and compared them against state-of-the-art open-source baselines.

*   •AlphaEvolve Suite (General Agent): A suite of challenging open-ended mathematical problems derived from the AlphaEvolve paper[novikov2025alphaevolve]. 
*   •MLEBench[chan2024mle] (ML Agent): Real-world machine learning competitions requiring end-to-end pipeline optimization, spanning Computer Vision, NLP, and Tabular data. 

Baselines: We compared General Agent against two primary evolutionary agent frameworks:

*   •OpenEvolve: A standard implementation of the AlphaEvolve algorithm using Island Models. 
*   •ShinkaEvolve: A recent framework emphasizing sample efficiency via novelty search. 

Models: Experiments were conducted using both open-weights models (DeepSeek-r1-0528, etc.) and commercial models (Gemini-3-Pro-Preview, etc.) to ensure the results are framework-dependent rather than model-dependent.

### 5.2 Effectiveness

#### 5.2.1 Algorithmic Discovery (General Agent)

We compared the best solutions found by LoongFlow against the baselines and known theoretical bounds. As shown in Table[1](https://arxiv.org/html/2512.24077v1#S5.T1 "Table 1 ‣ 5.2.1 Algorithmic Discovery (General Agent) ‣ 5.2 Effectiveness ‣ 5 Experiments ‣ LoongFlow: Directed Evolutionary Search via a Cognitive Plan-Execute-Summarize Paradigm"), LoongFlow achieved state-of-the-art (SOTA) results across multiple problems in the benchmark suite.

Table 1: LoongFlow Performance on AlphaEvolve Suite (Grouped by Metric Direction)

Problem Metric LLM AlphaEvolve LoongFlow
Higher is Better (↑\uparrow)
Autocorrelation II Bound (↑\uparrow)DeepSeek-R1 0.8962 0.9027
Circle Packing (Square)Radius (↑\uparrow)DeepSeek-R1 2.6358 2.6359
Circle Packing (Rectangle)Radius (↑\uparrow)DeepSeek-R1 2.3658321 2.3658322
Lower is Better (↓\downarrow)
Hexagon Packing Side Length (↓\downarrow)DeepSeek-R1 3.93 3.92
Max-to-Min Ratios Ratio (↓\downarrow)DeepSeek-R1 12.88926 12.88924
Uncertainty Inequality Bound (↓\downarrow)DeepSeek-R1 0.352099104422 0.352099104421
Erdős’ problem Bound (↓\downarrow)DeepSeek-R1 0.380924 0.380913

Notably, in the Autocorrelation II problem, LoongFlow discovered a solution with a score of 0.9027, significantly outperforming the AlphaEvolve baseline (0.8962). This indicates that the Planner’s ability to enforce global structural constraints allows LoongFlow to navigate high-dimensional spaces more effectively than random mutation.

#### 5.2.2 Machine Learning Engineering (ML Agent)

In the Machine Learning domain, MLAgent demonstrated the ability to construct robust pipelines without human intervention. As shown in Table[2](https://arxiv.org/html/2512.24077v1#S5.T2 "Table 2 ‣ 5.2.2 Machine Learning Engineering (ML Agent) ‣ 5.2 Effectiveness ‣ 5 Experiments ‣ LoongFlow: Directed Evolutionary Search via a Cognitive Plan-Execute-Summarize Paradigm"), LoongFlow achieved 14 Gold Medals.

Table 2: LoongFlow Performance on MLE Bench

### 5.3 Efficiency and Stability Analysis

To quantify efficiency, we analyzed the Circle Packing (Square) task under strict compute budgets.

#### 5.3.1 Evolve Efficiency (DeepSeek-R1-0528)

We set a time limit of 24 hours with a target score ≥0.99\geq 0.99. As shown in Table[3](https://arxiv.org/html/2512.24077v1#S5.T3 "Table 3 ‣ 5.3.1 Evolve Efficiency (DeepSeek-R1-0528) ‣ 5.3 Efficiency and Stability Analysis ‣ 5 Experiments ‣ LoongFlow: Directed Evolutionary Search via a Cognitive Plan-Execute-Summarize Paradigm"), LoongFlow demonstrated a >60%>60\% improvement in evolutionary efficiency compared to OpenEvolve.

*   •Convergence: LoongFlow required an average of 258 evaluations to reach the target threshold (0.99), whereas OpenEvolve required 783 evaluations. 
*   •Success Rate: Across 3 independent runs, LoongFlow achieved a 100% success rate in reaching the high-score region (>0.99>0.99). In contrast, OpenEvolve only succeeded once (33% rate), and ShinkaEvolve failed to break the 0.99 barrier in all attempts. 

Table 3: Efficiency Comparison (Sorted by Best Score)

#### 5.3.2 High-Difficulty Breakthrough (Gemini-3-Pro)

Under a constrained budget of 100 iterations, we evaluated the agents’ ability to break theoretical barriers. As shown in Table[4](https://arxiv.org/html/2512.24077v1#S5.T4 "Table 4 ‣ 5.3.2 High-Difficulty Breakthrough (Gemini-3-Pro) ‣ 5.3 Efficiency and Stability Analysis ‣ 5 Experiments ‣ LoongFlow: Directed Evolutionary Search via a Cognitive Plan-Execute-Summarize Paradigm"), LoongFlow completed the task three times consecutively.

*   •LoongFlow: Successfully broke the theoretical barrier (Score >1.0>1.0) in 3 out of 3 runs. 
*   •Baselines: Both OpenEvolve and ShinkaEvolve failed to reach a score of 1.0 within the budget. This confirms that LoongFlow’s PES paradigm not only finds solutions faster (6 vs 100 calls) but accesses solution subspaces that are unreachable for standard evolutionary methods under limited budgets. 

Table 4: High-Difficulty Breakthrough (Top-100 Iterations)

### 5.4 Ablations

To validate the necessity of the "Plan-Execute-Summarize" (PES) paradigm, we conducted an ablation study using General Agent—the representative instantiation of the LoongFlow framework. We evaluated the contribution of the Planner, Executor, and Summary modules on the Circle Packing task.

![Image 3: Refer to caption](https://arxiv.org/html/2512.24077v1/ablation1.png)

Figure 3: Evolutionary Effect & Efficiency.

![Image 4: Refer to caption](https://arxiv.org/html/2512.24077v1/ablation2.png)

Figure 4: Score Convergence over Time. Note: Faint lines represent individual runs (N=3 N=3), and bold lines represent the average trajectory.

#### 5.4.1 Planner: The Compass of Evolution

The Planner provides global expert guidance. Removing it forces the agent into a "blind search" mode. As shown in Figure[4](https://arxiv.org/html/2512.24077v1#S5.F4 "Figure 4 ‣ 5.4 Ablations ‣ 5 Experiments ‣ LoongFlow: Directed Evolutionary Search via a Cognitive Plan-Execute-Summarize Paradigm"), the Planner-ablated agent stagnated below 0.96. The lack of search pruning increased the average time to reach Top-1 solutions from 9.67 hours to 14.67 hours.

#### 5.4.2 Executor: Balancing Speed and Depth

The Executor employs an adaptive "Fuse Mode", switching between Chat (single-turn) and ReAct (multi-turn).

*   •Chat Mode: Computationally lightweight but highly unstable. 
*   •ReAct Mode: Stable but inefficient. 
*   •Fuse Mode: By dynamically allocating compute, Fuse Mode achieved the highest asymptotic score (0.998) with optimal sample efficiency. 

#### 5.4.3 Summary: The Evolutionary Feedback

The Summary prevents the loss of historical insights. Without it, the agent suffered from cyclical errors. One trial ran for 35 hours yet failed to break the 0.95 threshold (as shown in Figure[4](https://arxiv.org/html/2512.24077v1#S5.F4 "Figure 4 ‣ 5.4 Ablations ‣ 5 Experiments ‣ LoongFlow: Directed Evolutionary Search via a Cognitive Plan-Execute-Summarize Paradigm"), see the "No-Summary" trajectory). The absence of retrospective analysis degraded the Planner’s decision-making, confirming that the summary module is essential for breaking performance bottlenecks.

6 Conclusion
------------

In this work, we introduced LoongFlow, a cognitive evolutionary framework that fundamentally transcends the "blind watchmaker" limitations of traditional LLM-based optimization. By identifying the critical "cognitive ceiling" in existing methods—specifically their reliance on stochastic mutation and lack of historical reflection—we proposed a paradigm shift from random search to Directed Cognitive Evolution.

Our core contributions, the "Plan-Execute-Summarize" (PES) paradigm and the Hybrid Evolutionary Memory, effectively bridge the gap between reasoning agents and evolutionary computation. Theoretical analysis and extensive experiments demonstrate that LoongFlow not only preserves the diversity benefits of population-based methods but also injects the strategic depth of reasoning agents, achieving state-of-the-art results with significantly reduced computational overhead. LoongFlow establishes a new standard for sample-efficient, autonomous scientific discovery.

Future work will focus on extending LoongFlow towards fully autonomous "Meta-Agents" that can self-configure their evolutionary strategies and learning unsupervised diversity metrics for novel domains.

