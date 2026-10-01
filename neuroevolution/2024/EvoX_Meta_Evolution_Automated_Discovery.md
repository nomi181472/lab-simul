Title: EvoX: Meta-Evolution for Automated Discovery

URL Source: https://arxiv.org/html/2602.23413

Markdown Content:
Shu Liu 1∗, Shubham Agarwal 1∗, Monishwaran Maheswaran 1, Mert Cemri 1, Zhifei Li 1, Qiuyang Mang 1, Ashwin Naren 1, Ethan Boneh 2, Audrey Cheng 1, Melissa Z. Pan 1, Alexander Du 1, Kurt Keutzer 1, Alexandros G. Dimakis 1,3, Koushik Sen 1, Matei Zaharia 1, Ion Stoica 1 Affiliations 

1 UC Berkeley 2 Stanford University 3 Bespoke Labs

###### Abstract

Abstract: Recent work such as AlphaEvolve has shown that combining LLM-driven optimization with evolutionary search can effectively improve programs, prompts, and algorithms across domains. In this paradigm, previously evaluated solutions are reused to guide the model toward new candidate solutions. Crucially, the effectiveness of this evolution process depends on the search strategy: how prior solutions are selected and varied to generate new candidates. However, most existing methods rely on fixed search strategies with predefined knobs (e.g., explore–exploit ratios) that remain static throughout execution. While effective in some settings, these approaches often fail to adapt across tasks, or even within the same task as the search space changes over time.

We introduce EvoX, an adaptive evolution method that optimizes its own evolution process. EvoX jointly evolves candidate solutions and the search strategies used to generate them, continuously updating how prior solutions are selected and varied based on progress. This enables the system to dynamically shift between different search strategies during the optimization process. Across nearly 200 real-world optimization tasks, EvoX outperforms existing AI-driven evolutionary methods including AlphaEvolve, OpenEvolve, GEPA, and ShinkaEvolve on the majority of tasks.

Table 1: EvoX across optimization tasks. We compare EvoX against strong human-designed methods and prior AI discovery systems, including AlphaEvolve, GEPA, ShinkaEvolve, and OpenEvolve, across mathematical optimization, systems performance optimization (e.g., GPU-to-model placement), and algorithm engineering tasks (e.g., Frontier-CS). For Frontier-CS, we report median scores over 172 competitive programming problems. Arrows indicate whether higher (↑\uparrow) or lower (↓\downarrow) scores are better. All _open_ AI frameworks are evaluated under a fixed budget of 100 iterations, and the best result among them is reported as “AI Best.” AlphaEvolve results are taken directly from its original publication.

![Image 1: Refer to caption](https://arxiv.org/html/2602.23413v1/fig/architecturev0.png)

(a) Overall system architecture

![Image 2: Refer to caption](https://arxiv.org/html/2602.23413v1/fig/good_bad_two_in_one.png)

(b) Evolved search (blue) breakthroughs

Figure 1: Evolving the search strategy. ([1](https://arxiv.org/html/2602.23413#S0.F1 "Figure 1 ‣ EvoX: Meta-Evolution for Automated Discovery")) _System architecture._ EvoX has two coupled loops: an inner loop that evolves solutions, and an outer loop that evolves the _search strategy_ that governs generation. ([1](https://arxiv.org/html/2602.23413#S0.F1 "Figure 1 ‣ EvoX: Meta-Evolution for Automated Discovery")) _Effect of search evolution._ A strategy with fixed exploration-exploitation ratio (MAP-Elites, red) stagnates, while an evolving search strategy (blue) produces discrete performance breakthroughs. 

1 Introduction
--------------

LLM-driven optimization combined with evolutionary search has enabled numerous scientific breakthroughs across domains including mathematics[[22](https://arxiv.org/html/2602.23413#bib.bib14 "AlphaEvolve: a coding agent for scientific and algorithmic discovery")], systems performance optimization[[7](https://arxiv.org/html/2602.23413#bib.bib17 "Barbarians at the gate: how AI is upending systems research")], and competitive programming[[15](https://arxiv.org/html/2602.23413#bib.bib18 "ALE-Bench: a benchmark for long-horizon objective-driven algorithm engineering")]. Typically, LLM-driven evolutionary systems maintain a population of candidate solutions. At each step, a search strategy selects a subset of previously evaluated solutions and constructs a prompt, which is passed to a generator model (typically an LLM) to produce new candidates. These candidates are then evaluated and added back to the population.

Critically, the effectiveness of the evolution process relies on the search strategy, which determines both (i) which candidates are selected from the population and (ii) how new solutions are proposed from them (e.g., via refinement, structural variation, or combining multiple ideas). These choices directly influence what the generator attempts next and which regions of the solution space to explore.

Existing LLM-driven evolutionary systems often rely on fixed search strategies with hand-specified parameters. For example, AlphaEvolve[[22](https://arxiv.org/html/2602.23413#bib.bib14 "AlphaEvolve: a coding agent for scientific and algorithmic discovery")] employs MAP-Elites with predefined population database structures and selection ratios. OpenEvolve[[28](https://arxiv.org/html/2602.23413#bib.bib20 "OpenEvolve: an open-source evolutionary coding agent")] similarly relies on static elite and diversity heuristics. ShinkaEvolve[[17](https://arxiv.org/html/2602.23413#bib.bib19 "ShinkaEvolve: towards open-ended and sample-efficient program evolution")] takes a step toward adaptivity by incorporating bandit-based selection on LLM generators. However, key search strategy knobs remain manually configured. For example, the exploitation ratio is fixed, controlling how often top solutions are refined.

In practice, a fixed search strategy often fails to generalize across problems or across different stages of optimization. This can lead to stagnation during the search and requires manual retuning to make progress. Some tasks benefit primarily from repeated refinement of a strong candidate, where small local modifications incrementally improve an existing solution. For example, in packing problems[[22](https://arxiv.org/html/2602.23413#bib.bib14 "AlphaEvolve: a coding agent for scientific and algorithmic discovery"), [11](https://arxiv.org/html/2602.23413#bib.bib44 "Mathematical exploration and discovery at scale")], adjusting a few placements in a near-feasible configuration can yield further gains. In contrast, other tasks require qualitatively different solution structures. For instance, in a GPU-model placement problem, refining a least-loaded heuristic quickly plateaus, and further progress requires reformulating the problem as a bin-packing assignment.

Even within a single optimization run, the effectiveness of a search strategy can change over time. In a multi-objective signal processing task (Figure[1](https://arxiv.org/html/2602.23413#S0.F1 "Figure 1 ‣ EvoX: Meta-Evolution for Automated Discovery")([1](https://arxiv.org/html/2602.23413#S0.F1 "Figure 1 ‣ EvoX: Meta-Evolution for Automated Discovery"))), a MAP-Elites style strategy (red line) makes rapid early progress but later stagnates. Switching to a strategy that explicitly samples candidates along different trade-off objectives (blue line) enables continued improvement.

Motivated by this observation, EvoX frames LLM-driven optimization as a meta-learning problem in which _the search strategy itself is treated as an evolvable object_. Rather than fixing a hand-tuned search strategy, EvoX operates as a two-level evolution process comprising a _solution-evolution_ loop and a _meta-evolution_ loop (Figure[1](https://arxiv.org/html/2602.23413#S0.F1 "Figure 1 ‣ EvoX: Meta-Evolution for Automated Discovery")([1](https://arxiv.org/html/2602.23413#S0.F1 "Figure 1 ‣ EvoX: Meta-Evolution for Automated Discovery"))).

The solution-evolution loop generates candidate solutions under the standard LLM-driven evolutionary search paradigm. At each step, a search strategy selects prior candidates and determines how to generate new ones, for example by sampling high-performing solutions and applying a variation operator (e.g., refinement or structural variation). The resulting prompt is passed to the LLM, which produces a new candidate that is evaluated and added back to the population.

The meta-evolution loop periodically updates the search strategy. Each strategy is deployed for a window of solution-evolution iterations and evaluated by the progress it induces on the downstream task (e.g., improvement rate). When progress stagnates, the meta-evolution loop generates a new search strategy using the LLM, conditioned on prior strategies, their observed performance, and the current state of the solution population. This loop enables EvoX to evolve the search strategy itself: selecting, mutating, and replacing strategies based on their effectiveness. Because the performance of a strategy depends on the evolving population, EvoX explicitly conditions strategy generation on population-level signals, allowing it to adapt as the search space changes.

#### Contributions.

We evaluate EvoX across nearly 200 real-world optimization tasks spanning mathematics, algorithms, and scientific research benchmarks (Table[1](https://arxiv.org/html/2602.23413#S0.T1 "Table 1 ‣ EvoX: Meta-Evolution for Automated Discovery") and Section[6](https://arxiv.org/html/2602.23413#S6 "6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery")). Starting from a simple random-sampling search strategy, EvoX consistently outperforms existing LLM-driven evolutionary frameworks, including OpenEvolve, ShinkaEvolve, and GEPA[[28](https://arxiv.org/html/2602.23413#bib.bib20 "OpenEvolve: an open-source evolutionary coding agent"), [17](https://arxiv.org/html/2602.23413#bib.bib19 "ShinkaEvolve: towards open-ended and sample-efficient program evolution"), [1](https://arxiv.org/html/2602.23413#bib.bib11 "GEPA: reflective prompt evolution can outperform reinforcement learning")] on the majority of tasks (e.g., 96% of the math and system optimization benchmarks). It also often matches or surpasses the best human-designed solutions. In summary, we make the following contributions:

1.   1.
We formalize LLM-driven optimization as a two-level process that separates solution evolution from search strategy evolution.

2.   2.
We introduce EvoX, a method that dynamically evolves search strategies based on improvements in the optimization objective

3.   3.
We demonstrate consistent improvements over prior methods across nearly 200 problems, and characterize the cost, scaling behavior, and adaptation dynamics of the search strategy evolution.

2 Related Work
--------------

#### Context and Memory Management.

Many recent LLM systems iteratively improve solutions by incorporating feedback from prior attempts. A key challenge in this setting is how to store, summarize, and present past information to guide future generations. ACE[[37](https://arxiv.org/html/2602.23413#bib.bib39 "Agentic context engineering: evolving contexts for self-improving language models")] treats context as editable objects, MemGPT[[24](https://arxiv.org/html/2602.23413#bib.bib53 "MemGPT: towards llms as operating systems.")] introduces hierarchical memory, and Reflexion[[30](https://arxiv.org/html/2602.23413#bib.bib33 "Reflexion: language agents with verbal reinforcement learning")] and Self-Refine[[19](https://arxiv.org/html/2602.23413#bib.bib51 "Self-refine: iterative refinement with self-feedback")] incorporate model-generated feedback into future prompts. These approaches improve how prior experience is organized and reused, enabling more effective iterative refinement.

One prominent paradigm that builds on this foundation is evolutionary search, which treats prior solutions as a population and explicitly selects and varies candidates over time.

#### LLM-Guided Evolutionary Search.

Recent work applies evolutionary search to guide LLM-driven optimization, spanning prompt optimization[[10](https://arxiv.org/html/2602.23413#bib.bib49 "Promptbreeder: self-referential self-improvement via prompt evolution"), [13](https://arxiv.org/html/2602.23413#bib.bib59 "Evoprompt: connecting llms with evolutionary algorithms yields powerful prompt optimizers"), [18](https://arxiv.org/html/2602.23413#bib.bib34 "Evolving deeper llm thinking"), [35](https://arxiv.org/html/2602.23413#bib.bib58 "Reevo: large language models as hyper-heuristics with reflective evolution"), [32](https://arxiv.org/html/2602.23413#bib.bib55 "Dynamic cheatsheet: test-time learning with adaptive memory"), [9](https://arxiv.org/html/2602.23413#bib.bib46 "Feedback-aware monte carlo tree search for efficient information seeking in goal-oriented conversations")] and program or algorithm discovery[[22](https://arxiv.org/html/2602.23413#bib.bib14 "AlphaEvolve: a coding agent for scientific and algorithmic discovery"), [28](https://arxiv.org/html/2602.23413#bib.bib20 "OpenEvolve: an open-source evolutionary coding agent"), [17](https://arxiv.org/html/2602.23413#bib.bib19 "ShinkaEvolve: towards open-ended and sample-efficient program evolution"), [1](https://arxiv.org/html/2602.23413#bib.bib11 "GEPA: reflective prompt evolution can outperform reinforcement learning"), [3](https://arxiv.org/html/2602.23413#bib.bib38 "Codeevolve: an open source evolutionary coding agent for algorithm discovery and optimization"), [14](https://arxiv.org/html/2602.23413#bib.bib57 "Evolving code with a large language model"), [31](https://arxiv.org/html/2602.23413#bib.bib56 "Llm-sr: scientific equation discovery via programming with large language models")]. These systems differ primarily in how candidates are selected and varied. AlphaEvolve[[22](https://arxiv.org/html/2602.23413#bib.bib14 "AlphaEvolve: a coding agent for scientific and algorithmic discovery")] selects candidates using MAP-Elites, GEPA[[1](https://arxiv.org/html/2602.23413#bib.bib11 "GEPA: reflective prompt evolution can outperform reinforcement learning")] selects along Pareto frontiers, and OpenEvolve[[28](https://arxiv.org/html/2602.23413#bib.bib20 "OpenEvolve: an open-source evolutionary coding agent")] and ShinkaEvolve[[17](https://arxiv.org/html/2602.23413#bib.bib19 "ShinkaEvolve: towards open-ended and sample-efficient program evolution")] emphasize diversity-driven selection. CodeEvolve[[3](https://arxiv.org/html/2602.23413#bib.bib38 "Codeevolve: an open source evolutionary coding agent for algorithm discovery and optimization")] integrates LLM-based generation within an island-based genetic algorithm, while DeltaEvolve[[16](https://arxiv.org/html/2602.23413#bib.bib60 "DeltaEvolve: accelerating scientific discovery through momentum-driven evolution")] models semantic deltas between candidates. PACEvolve[[34](https://arxiv.org/html/2602.23413#bib.bib35 "PACEvolve: enabling long-horizon progress-aware consistent evolution")] introduces mechanisms such as hierarchical context pruning and momentum-based backtracking to improve stability. However, these systems still operate under predefined search strategies with manually designed knobs.

Some recent work introduces learning-based adaptation mechanisms. SOAR[[25](https://arxiv.org/html/2602.23413#bib.bib48 "Self-improving language models for evolutionary program synthesis: a case study on arc-agi")] alternates between search and model fine-tuning, while ThetaEvolve[[33](https://arxiv.org/html/2602.23413#bib.bib16 "ThetaEvolve: test-time learning on open problems")], TTT-Discover[[36](https://arxiv.org/html/2602.23413#bib.bib36 "Learning to discover at test time")], and FLEX[[4](https://arxiv.org/html/2602.23413#bib.bib37 "Flex: continuous agent evolution via forward learning from experience")] apply reinforcement learning to improve the generator model. These approaches adapt the generator model itself, but do not adapt the search strategy governing candidate selection and variation.

#### Meta-Learning and Learning to Optimize.

Meta-learning and learned optimization frameworks treat the optimization procedure itself as the object of adaptation rather than a fixed component[[21](https://arxiv.org/html/2602.23413#bib.bib2 "Understanding and correcting pathologies in the training of learned optimizers"), [5](https://arxiv.org/html/2602.23413#bib.bib1 "Learning to optimize: a primer and a benchmark")]. Prior work explores learned optimizers, symbolic discovery of update rules[[6](https://arxiv.org/html/2602.23413#bib.bib3 "Symbolic discovery of optimization algorithms")], gradient-based meta-learning[[2](https://arxiv.org/html/2602.23413#bib.bib6 "Learning to learn by gradient descent by gradient descent")], and reinforcement learning-based optimizers[[26](https://arxiv.org/html/2602.23413#bib.bib5 "Optimizing test-time compute via meta reinforcement fine-tuning")]. These works demonstrate that adapting the optimization process itself can significantly improve search efficiency. Our work extends this principle to LLM-driven evolutionary search.

#### EvoX: Meta-Evolving the Search Strategy.

EvoX builds on these lines of work by treating the search strategy itself as an evolvable object. Rather than relying on fixed candidate selection and variation mechanisms, EvoX evolves the search strategy through evolutionary feedback, dynamically adapting how candidates are selected and varied across different optimization stages and heterogeneous solution landscapes.

3 Problem Formulation
---------------------

We formalize LLM-driven evolutionary search following standard abstractions from prior work[[27](https://arxiv.org/html/2602.23413#bib.bib15 "Mathematical discoveries from program search with large language models"), [22](https://arxiv.org/html/2602.23413#bib.bib14 "AlphaEvolve: a coding agent for scientific and algorithmic discovery"), [28](https://arxiv.org/html/2602.23413#bib.bib20 "OpenEvolve: an open-source evolutionary coding agent"), [1](https://arxiv.org/html/2602.23413#bib.bib11 "GEPA: reflective prompt evolution can outperform reinforcement learning"), [17](https://arxiv.org/html/2602.23413#bib.bib19 "ShinkaEvolve: towards open-ended and sample-efficient program evolution")]. Candidate solutions are generated by a language model, evaluated by a task-specific evaluator, and iteratively evolved under a search strategy and a fixed evaluation budget.

#### Candidate solution evaluation.

Let 𝒳\mathcal{X} denote the space of candidate solutions (e.g., programs or prompts). Each candidate x∈𝒳 x\in\mathcal{X} is evaluated by an evaluator

E​(x)→(s​(x),a​(x)),E(x)\rightarrow\bigl(s(x),\,a(x)\bigr),

which returns a scalar score s​(x)∈ℝ s(x)\in\mathbb{R} together with auxiliary artifacts a​(x)a(x) (e.g., logs, traces, or any feedback).

#### Solution population database.

The optimization proceeds for T T sequential evaluation steps. Let

𝒟 t={(x i,s i,a i)}i=1 t,𝒟 0=∅,\mathcal{D}_{t}=\{(x_{i},s_{i},a_{i})\}_{i=1}^{t},\quad\mathcal{D}_{0}=\emptyset,

denote the database of all candidate solutions evaluated up to step t t. At each step, a new candidate is generated, evaluated, and appended to the database to form 𝒟 t+1\mathcal{D}_{t+1}.

#### Search strategy.

A search strategy S∈𝒮 S\in\mathcal{S} specifies how the next-generation LLM input is constructed from the current database 𝒟 t\mathcal{D}_{t}. Concretely, S S defines the construction of the generation context:

C S​(𝒟 t)→(x par,π,ℐ),C_{S}(\mathcal{D}_{t})\rightarrow\bigl(x_{\mathrm{par}},\,\pi,\,\mathcal{I}\bigr),

which selects (i) one or more _parent_ candidate(s) x par∈𝒟 t x_{\mathrm{par}}\in\mathcal{D}_{t} to be modified, (ii) a _variation operator_ π\pi expressed in the prompt that specifies how the parent can be modified, and (iii) optionally, an _inspiration set_ ℐ⊆𝒟 t\mathcal{I}\subseteq\mathcal{D}_{t} (e.g., diverse candidates) that provides additional exemplars. This abstraction aligns with prior LLM-driven evolutionary systems[[17](https://arxiv.org/html/2602.23413#bib.bib19 "ShinkaEvolve: towards open-ended and sample-efficient program evolution"), [28](https://arxiv.org/html/2602.23413#bib.bib20 "OpenEvolve: an open-source evolutionary coding agent")].

Given (x par,π,ℐ)(x_{\mathrm{par}},\pi,\mathcal{I}), the solution generator (implemented by an LLM) produces a new candidate

x′∼𝒢 sol(⋅∣x par,π,ℐ),x^{\prime}\sim\mathcal{G}_{\mathrm{sol}}(\cdot\mid x_{\mathrm{par}},\pi,\mathcal{I}),

which is evaluated and appended to the database.

_Variation operators._ The operator π\pi specifies the _kind_ of modification requested in a generation step. We use three variation operators: local refinement for fine-grained edits (exploitation), structural variation for coarse-grained redesigns (exploration), and free-form variation, which imposes no constraints on the edit scope.

The semantics of these operators are task-dependent. For example, in code or algorithmic tasks, refinement may tune parameters, reorder logic, or edit localized code blocks, whereas structural variation may switch algorithm families or restructure the overall design. At the start of each problem, we instantiate these operators using a lightweight model (Section[6](https://arxiv.org/html/2602.23413#S6 "6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery")) to generate a small set of operator-specific prompts from the problem description. During optimization, the search strategy selects among these operators to control the intended type of variation applied to the parent candidate.

#### Optimization goal.

Search strategies shape optimization behavior, and our goal is to adaptively select and improve the search strategy to maximize the final best score

max(x,s,a)∈𝒟 T⁡s\max_{(x,s,a)\in\mathcal{D}_{T}}s

under a fixed evaluation budget of T T steps.

4 Co-evolving Solution and Search Strategy
------------------------------------------

Algorithm 1 EvoX: Two-level evolution process

1:Budget

T T
, window

W W
, stagnation threshold

τ\tau

2:Evaluator

E E
; solution generator

𝒢 sol\mathcal{G}_{\mathrm{sol}}
; strategy generator

𝒢 str\mathcal{G}_{\mathrm{str}}

3:Population descriptor

ϕ​(⋅)\phi(\cdot)
; validity test

Valid​(⋅)\textsc{Valid}(\cdot)

4:Initial solution database

𝒟 0\mathcal{D}_{0}
; initial strategy

S 0 S_{0}

5:

𝒟 t←𝒟 0\mathcal{D}_{t}\leftarrow\mathcal{D}_{0}
;

S t←S 0 S_{t}\leftarrow S_{0}
;

ℋ←∅\mathcal{H}\leftarrow\emptyset
;

t←0 t\leftarrow 0

6:while

t<T t<T
do

7:Phase I: Solution evolution under S t S_{t} (one window)

8:

ϕ t←ϕ​(𝒟 t)\phi_{t}\leftarrow\phi(\mathcal{D}_{t})

9:

s start←max(x,s,a)∈𝒟 t⁡s s_{\mathrm{start}}\leftarrow\max_{(x,s,a)\in\mathcal{D}_{t}}s

10:for

i=1 i=1
to

W W
do

11:if

t≥T t\geq T
then break

12:end if

13:

(x par,π,ℐ)∼C S t​(𝒟 t)(x_{\mathrm{par}},\pi,\mathcal{I})\sim C_{S_{t}}(\mathcal{D}_{t})

14:

x′∼𝒢 sol(⋅∣x par,π,ℐ)x^{\prime}\sim\mathcal{G}_{\mathrm{sol}}(\cdot\mid x_{\mathrm{par}},\pi,\mathcal{I})

15:

(s′,a′)←E​(x′)(s^{\prime},a^{\prime})\leftarrow E(x^{\prime})

16:

𝒟 t+1←𝒟 t∪{(x′,s′,a′)}\mathcal{D}_{t+1}\leftarrow\mathcal{D}_{t}\cup\{(x^{\prime},s^{\prime},a^{\prime})\}
;

t←t+1 t\leftarrow t+1
;

𝒟 t←𝒟 t+1\mathcal{D}_{t}\leftarrow\mathcal{D}_{t+1}

17:end for

18:Phase II: Progress monitoring

19:

s end←max(x,s,a)∈𝒟 t⁡s s_{\mathrm{end}}\leftarrow\max_{(x,s,a)\in\mathcal{D}_{t}}s

20:

Δ←s end−s start\Delta\leftarrow s_{\mathrm{end}}-s_{\mathrm{start}}

21:

J t←Δ​log⁡(1+s start)/W J_{t}\leftarrow\Delta\log(1+s_{\mathrm{start}})/\sqrt{W}

22:

ℋ←ℋ∪{(S t,ϕ t,J t)}\mathcal{H}\leftarrow\mathcal{H}\cup\{(S_{t},\phi_{t},J_{t})\}

23:Phase III: Strategy evolution (on stagnation)

24:if

Δ<τ\Delta<\tau
then

25:

S′∼𝒢 str(⋅∣ℋ,ϕ(𝒟 t))S^{\prime}\sim\mathcal{G}_{\mathrm{str}}(\cdot\mid\mathcal{H},\phi(\mathcal{D}_{t}))

26:if

Valid​(S′)\textsc{Valid}(S^{\prime})
then

S t←S′S_{t}\leftarrow S^{\prime}

27:end if

28:end if

29:end while

30:return

arg⁡max(x,s,a)∈𝒟 T⁡s\arg\max_{(x,s,a)\in\mathcal{D}_{T}}s

EvoX addresses the problem of iteratively improving solution quality under a fixed evaluation budget defined in Section[3](https://arxiv.org/html/2602.23413#S3 "3 Problem Formulation ‣ EvoX: Meta-Evolution for Automated Discovery") by _co-evolving_ two components: (i) the solution population 𝒟 t\mathcal{D}_{t}, and (ii) the search strategy S S that constructs the next-generation LLM context. Specifically, EvoX proceeds through the three-step process shown in Algorithm[1](https://arxiv.org/html/2602.23413#alg1 "Algorithm 1 ‣ 4 Co-evolving Solution and Search Strategy ‣ EvoX: Meta-Evolution for Automated Discovery"): (1) evolving the solution database under a fixed search strategy, (2) monitoring population performance, and (3) updating the search strategy when progress stalls, using feedback from previous strategy deployments and the current state of the solution population.

### 4.1 Solution evolution under the current strategy

Given the current database 𝒟 t\mathcal{D}_{t} and an active strategy S t S_{t}, EvoX generates and evaluates new candidates as defined in Section[3](https://arxiv.org/html/2602.23413#S3 "3 Problem Formulation ‣ EvoX: Meta-Evolution for Automated Discovery"): the strategy constructs a generation context (x par,π,ℐ)∼C S t​(𝒟 t)(x_{\mathrm{par}},\pi,\mathcal{I})\sim C_{S_{t}}(\mathcal{D}_{t}), the generator produces x′x^{\prime}, and the evaluator appends (x′,s′,a′)(x^{\prime},s^{\prime},a^{\prime}) to the database.

A search strategy controls both _what_ the model sees (through parent selection and inspiration construction) and _how_ the selected parent is transformed (via the variation operator π\pi).

### 4.2 Progress monitoring and strategy updates

A search strategy does not affect a single candidate in isolation, but shapes a _sequence_ of generated candidates. Accordingly, its effectiveness can only be assessed over multiple evaluation steps rather than from a single outcome. For this reason, EvoX monitors progress over a sliding window of the most recent W W evaluation steps.

Let t t denote the start of the current monitoring window and define

s start=max(x,s,a)∈𝒟 t⁡s,s end=max(x,s,a)∈𝒟 t+W⁡s,Δ=s end−s start.s_{\mathrm{start}}=\max_{(x,s,a)\in\mathcal{D}_{t}}s,\quad s_{\mathrm{end}}=\max_{(x,s,a)\in\mathcal{D}_{t+W}}s,\quad\Delta=s_{\mathrm{end}}-s_{\mathrm{start}}.(1)

We treat Δ\Delta as the primary signal of strategy efficacy. If Δ\Delta falls below a stagnation threshold τ\tau, EvoX triggers a strategy update; otherwise, it continues with the current strategy. This design makes strategy switching _demand-driven_ rather than _periodic at a fixed interval_, avoiding unnecessary updates when the current strategy is effective.

#### Search strategy evaluation.

After W W evaluation steps have elapsed, EvoX computes a performance score for the strategy employed during that window:

J​(S t∣𝒟 t)\displaystyle J(S_{t}\mid\mathcal{D}_{t})=(s end−s start)​log⁡(1+s start)W\displaystyle=\frac{(s_{\mathrm{end}}-s_{\mathrm{start}})\,\log(1+s_{\mathrm{start}})}{\sqrt{W}}(2)

The log⁡(1+s start)\log(1+s_{\mathrm{start}}) term upweights improvements achieved from higher starting scores, rewarding strategies that drive progress near the frontier where gains are typically harder to obtain. The W\sqrt{W} normalization accounts for window length.

### 4.3 Meta-evolving the search strategy

Adapting search strategies is challenging because their effectiveness is _state-dependent_. For instance, a strategy that drives rapid progress at one stage of search may become ineffective as the solution population evolves. This arises from the inherently non-stationary nature of evolutionary optimization: as the database grows, the distribution of candidate quality, diversity, and variation outcomes shifts, altering which search behaviors are beneficial.

EvoX addresses this challenge by conditioning strategy updates on both _a population of past search strategies_ and _the current downstream solution population state_. Conditioning on the population descriptor ϕ​(D t)\phi(D_{t}) allows the system to interpret stagnation relative to population structure (e.g., loss of diversity, repeated parent selection, ineffective variation). Conditioning on the strategy database H H provides empirical evidence about which strategies previously induced progress under similar states.

Together, these signals enable state-conditional strategy adaptation rather than fixed or periodic strategy switching.

#### Search strategy database.

To achieve this, EvoX maintains a _search strategy database_

ℋ={(S j,ϕ j,J j)}j=1 M,\mathcal{H}=\{(S_{j},\phi_{j},J_{j})\}_{j=1}^{M},

which serves as a memory of previously deployed strategies. Each entry records a strategy S j S_{j}, a descriptor ϕ j=ϕ​(𝒟 t j)\phi_{j}=\phi(\mathcal{D}_{t_{j}}) summarizing the population state before and after deployment, and its observed performance J j J_{j} computed using Eq.[2](https://arxiv.org/html/2602.23413#S4.E2 "Equation 2 ‣ Search strategy evaluation. ‣ 4.2 Progress monitoring and strategy updates ‣ 4 Co-evolving Solution and Search Strategy ‣ EvoX: Meta-Evolution for Automated Discovery"). This database enables EvoX to reason about _which strategies tend to work under which search conditions_.

_Population state descriptor._ The descriptor ϕ​(𝒟 t)\phi(\mathcal{D}_{t}) summarizes the current state of the solution population. It includes: (i) score statistics (best value, percentiles, spread), (ii) frontier structure (e.g., top-k k scores), (iii) progress indicators (e.g., steps since last significant improvement), and (iv) recent window statistics (e.g., parent selection frequency).

#### Meta-evolution of search strategies.

When a strategy update is triggered, EvoX evolves a new search strategy by applying variation to high-performing strategies from the history of the search. The strategy database ℋ\mathcal{H} plays the role of a population, where each individual strategy S j S_{j} is associated with a score signal J j J_{j} evaluated in the population state ϕ j\phi_{j} in which it was deployed.

EvoX performs score-biased selection over ℋ\mathcal{H} to choose a high-performing parent strategy to mutate, and selects an inspirational set of strategies prioritizing those that previously induced strong progress and those that performed well under similar population descriptors. The strategy generator (i.e., an LLM), denoted 𝒢 str\mathcal{G}_{\mathrm{str}}, then applies mutation to the selected parent, conditioned on the current population descriptor ϕ​(𝒟 t)\phi(\mathcal{D}_{t}), to produce a new strategy candidate S′S^{\prime}:

S′∼𝒢 str(⋅∣S par,ϕ(𝒟 t)),S^{\prime}\sim\mathcal{G}_{\mathrm{str}}\!\left(\,\cdot\mid S_{\mathrm{par}},\phi(\mathcal{D}_{t})\right),

Mutations modify the components of a strategy (e.g., parent selection rules, construction of the inspiration set, or preferences over variation operators π\pi).

#### Strategy deployment.

Because strategies directly affect execution, EvoX validates each candidate strategy before deployment. If validation succeeds, EvoX immediately switches to S′S^{\prime}; otherwise, it retries generation up to a fixed budget and falls back to the previous strategy if all attempts fail. Importantly, switching strategies never resets the solution population: the database 𝒟\mathcal{D} is preserved, and evolution continues from the current search state.

5 Case Study: Signal Processing
-------------------------------

![Image 3: Refer to caption](https://arxiv.org/html/2602.23413v1/x1.png)

Figure 2: Evolving search strategy on the signal processing task. Starting from a uniform random sampling strategy, EvoX detects stagnation and adaptively switches to greedy search, stratified multi-objective sampling, UCB-guided structural variation, and finally local refinement. These strategy changes enable discovery of improved filtering programs and yield major gains at iterations ∼\sim 48 (+0.119), ∼\sim 70 (+0.056), and ∼\sim 96 (+0.022), achieving 34.1% higher final score than the static baseline. 

We illustrate EvoX through a case study on a signal processing task[[29](https://arxiv.org/html/2602.23413#bib.bib45 "Introduction to digital signal processing and filter design"), [28](https://arxiv.org/html/2602.23413#bib.bib20 "OpenEvolve: an open-source evolutionary coding agent")]. The goal is to build a filtering program for a noisy, changing time series. To succeed, the program must balance several competing goals: high fidelity to the signal, smoothness to reduce noise, low lag for responsiveness, and a minimal false trend changes. We evaluate candidate programs using a combined score that balances these four objectives.

Figure[2](https://arxiv.org/html/2602.23413#S5.F2 "Figure 2 ‣ 5 Case Study: Signal Processing ‣ EvoX: Meta-Evolution for Automated Discovery") compares EvoX against a static baseline under the same 100-iteration budget. The static baseline uniformly samples the parent and inspiration set from the population and applies free-form variation to generate new candidates throughout the search. EvoX achieves a 34.1% higher final score by adaptively changing its search strategy based on observed progress, as described in Section[4](https://arxiv.org/html/2602.23413#S4 "4 Co-evolving Solution and Search Strategy ‣ EvoX: Meta-Evolution for Automated Discovery").

Static Baseline. The dashed curve in Figure[2](https://arxiv.org/html/2602.23413#S5.F2 "Figure 2 ‣ 5 Case Study: Signal Processing ‣ EvoX: Meta-Evolution for Automated Discovery") illustrates the limitations of a fixed search strategy. This strategy achieves modest early improvement (0.499 →\rightarrow 0.530) but then stagnates. Because the baseline chooses its parent and inspiration programs randomly and uses generic free-form variations, it mostly produces simple filters, such as basic moving averages (MA) or exponential moving averages (EMA) using NumPy. Later generations only make tiny adjustments to these simple settings, which are not enough to handle complex noise patterns. As a result, the search hits a ceiling and stays stuck.

Phase 1: Random Search and Greedy Search. EvoX begins with the same random strategy as the baseline. At iteration 20, the system detects that progress has stopped and tries a greedy strategy that focuses entirely on refining the single best program found so far. However, because the current best program still relies on simple MA/EMA structures, these tiny refinements fail to produce a breakthrough. The search remains stuck because the underlying program structure is too limited.

Phase 2: The Breakthrough (Stratified + Multi-Objective). At iteration 40, E​v​o​X EvoX learns from its past failure and switches to a more advanced strategy, where instead of only selecting the best overall program, it selects parents and inspiration programs from diverse score tiers and objective-specific rankings.

For example, if a parent filter reaches a high smoothness score but has a high lag at the same time, EvoX selects an inspiration program that excels with lower lag. Conversely, if a parent preserves signal fidelity but leaves too much noise, it picks an inspiration program specialized in smoothing of signal. By merging these different strengths, EvoX discovers novel hybrid designs such as combining singular spectrum analysis (SSA) with Whittaker smoothing. This smart blending of complementary ideas creates the largest jump in performance (+0.119).

Phase 3: Structural Exploration (UCB + Structural Variation). By iteration 60, as the progress slows again, the population state descriptor shows that many recently generated candidates have similar scores. This suggests that simple refinements or combination of ideas is no longer working. To break this pattern, EvoX evolves a policy that increases the use of structural variation to attempt bolder, more complex changes.

At the same time, it uses a UCB selection rule that encourages exploration of programs that have been largely ignored (rarely selected as parents). The strategy continues to use multi-objective sampling for selecting the inspiration programs, as it proved effective in the previous search stage. It also employs the _structural variation operator_ to encourage large, exploratory changes. As a result, the newly generated solutions begin to use advanced SciPy tools to construct filtering pipelines, incorporating higher-order filters, smoothing kernels, and forward–backward filtering operations (e.g., filtfilt). This exploration of new solution families yields a significant further improvement (+0.056).

Phase 4: Final Polishing (UCB + Local Refinement). By iteration 90, the search enters its final stage. Large structural changes now tend to destabilize performance rather than improve it. Consequently, EvoX shifts its strategy toward local refinement. This involves making small, precise adjustments to the top discovered solutions while keeping the UCB rule to prevent the search from narrowing too quickly. These fine-tuning steps provide the final gains (+0.022) and lock in the high score.

This example demonstrates the core power of E​v​o​X EvoX: it changes its search strategy based on what happened in previous steps and the current variety of programs it has found. By choosing the right strategy for the right moment, EvoX avoids the plateaus that stall traditional methods and achieves substantially higher final performance.

6 Evaluation
------------

Benchmarks. We evaluate EvoX on 196 real-world optimization tasks spanning mathematics (8), systems (6), and algorithmic and research problems (10 from ALE-Bench-Lite[[15](https://arxiv.org/html/2602.23413#bib.bib18 "ALE-Bench: a benchmark for long-horizon objective-driven algorithm engineering")] and 172 from Frontier-CS[[20](https://arxiv.org/html/2602.23413#bib.bib61 "FrontierCS: evolving challenges for evolving intelligence")]) (Sections[6.1](https://arxiv.org/html/2602.23413#S6.SS1 "6.1 Main Results: Math Optimization Problems ‣ 6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery")–[6.3](https://arxiv.org/html/2602.23413#S6.SS3 "6.3 Main Results: Algorithmic and Research Problems ‣ 6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery")). We also report results on the ARC-AGI-2 benchmark, a widely used benchmark for evaluating reasoning and generalization capabilities (Appendix[B.1](https://arxiv.org/html/2602.23413#A2.SS1 "B.1 ARC-AGI Evaluation ‣ Appendix B Additional Results ‣ EvoX: Meta-Evolution for Automated Discovery")). Detailed descriptions on the benchmarks are in Appendix[A](https://arxiv.org/html/2602.23413#A1 "Appendix A Benchmark Details ‣ EvoX: Meta-Evolution for Automated Discovery"). Additionally, we conduct ablation studies to analyze the cost and scaling behavior of EvoX in Section[6.4](https://arxiv.org/html/2602.23413#S6.SS4 "6.4 Ablations ‣ 6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery").

Baselines and Setups. We compare EvoX against strong LLM-driven evolutionary frameworks, including OpenEvolve[[28](https://arxiv.org/html/2602.23413#bib.bib20 "OpenEvolve: an open-source evolutionary coding agent")], ShinkaEvolve[[17](https://arxiv.org/html/2602.23413#bib.bib19 "ShinkaEvolve: towards open-ended and sample-efficient program evolution")], and GEPA[[1](https://arxiv.org/html/2602.23413#bib.bib11 "GEPA: reflective prompt evolution can outperform reinforcement learning")]. For mathematical tasks, we additionally report human-best results and prior state-of-the-art AlphaEvolve[[22](https://arxiv.org/html/2602.23413#bib.bib14 "AlphaEvolve: a coding agent for scientific and algorithmic discovery")] numbers. For ADRS systems benchmarks, we report human-best results. We include further comparisons to CodeEvolve[[3](https://arxiv.org/html/2602.23413#bib.bib38 "Codeevolve: an open source evolutionary coding agent for algorithm discovery and optimization")] and ThetaEvolve[[33](https://arxiv.org/html/2602.23413#bib.bib16 "ThetaEvolve: test-time learning on open problems")] in Appendix[B](https://arxiv.org/html/2602.23413#A2 "Appendix B Additional Results ‣ EvoX: Meta-Evolution for Automated Discovery").

For EvoX, we use GPT-5 for search strategy generation with a window size of 10% of the total iteration budget to detect stagnation and trigger strategy evolution (Section[4](https://arxiv.org/html/2602.23413#S4 "4 Co-evolving Solution and Search Strategy ‣ EvoX: Meta-Evolution for Automated Discovery")). All EvoX runs start from a simple random search strategy that samples both the parent and inspirational candidates uniformly at random. The effect of different initial search strategies (e.g., Best-of-N, MAP-Elites) is shown in Section[6.4](https://arxiv.org/html/2602.23413#S6.SS4 "6.4 Ablations ‣ 6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery"). We report full results of this random search strategy in Appendix[B.2](https://arxiv.org/html/2602.23413#A2.SS2 "B.2 Full Results ‣ Appendix B Additional Results ‣ EvoX: Meta-Evolution for Automated Discovery").

All _open_ frameworks, including OpenEvolve, ShinkaEvolve, GEPA, and EvoX, are evaluated under a fixed budget of 100 iterations per task unless specified otherwise. For math tasks, human-best and AlphaEvolve[[22](https://arxiv.org/html/2602.23413#bib.bib14 "AlphaEvolve: a coding agent for scientific and algorithmic discovery")] results are taken from the original paper, as AlphaEvolve is not open-sourced and its iteration budget is not publicly available.

We report mean and best performance over three independent runs using two backbone models, GPT-5[[23](https://arxiv.org/html/2602.23413#bib.bib8 "GPT-5 system card")] and Gemini-3.0-Pro[[12](https://arxiv.org/html/2602.23413#bib.bib9 "Gemini 3 Pro model card")]. Mean scores reflect robustness across runs, while best scores capture peak solution quality. Higher (↑\uparrow) or lower (↓\downarrow) scores indicate better performance depending on the task.

### 6.1 Main Results: Math Optimization Problems

We evaluate eight mathematical optimization tasks (continuous and discrete) spanning geometric packing, extremal geometry, distance maximization, and sequence design (Table[2](https://arxiv.org/html/2602.23413#S6.T2 "Table 2 ‣ 6.1 Main Results: Math Optimization Problems ‣ 6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery")). Across these tasks, EvoX achieves the strongest overall performance among open evolutionary frameworks, including OpenEvolve, ShinkaEvolve, and GEPA.

Table 2: Main results for math optimization problems. We report mean and best over three runs. “↑\uparrow” / “↓\downarrow” indicate maximization / minimization. All open methods use 100 iterations under the same backbone model. Bold denotes the best LLM-based open framework result per model. Green cells indicate matches or improvements over human SOTA or closed-source AlphaEvolve results. In Circle Packing Rect, EvoX achieves 2.36583237, exceeding AlphaEvolve’s 2.36583213, although both appear equal when rounded. 

Under GPT-5, EvoX attains the _best_ or tied-best result on 7 of 8 tasks, and under Gemini-3.0-Pro, it achieves the best result on all 8 tasks. In terms of _mean_ performance, EvoX achieves the best score on 6 of 8 tasks under GPT-5 and 7 of 8 tasks under Gemini-3.0-Pro, demonstrating robust performance across runs. In the two cases where mean performance lags best performance (Heilbronn triangle and MinMaxMinDist d=3 d=3), early random initializations occasionally converge to strong local optima, limiting further improvement within the fixed 100 iteration budget we set.

Compared to the strongest previously reported results, including the closed-source AlphaEvolve, EvoX matches or exceeds AlphaEvolve on 5 out of 7 tasks, such as Circle Packing, Circle Packing Rect, and MinMaxMinDist (d=3 d=3), within just 100 iterations. Since AlphaEvolve’s iteration budget is not publicly specified, direct cost comparison is not possible.

Beyond aggregate scores, EvoX discovers qualitatively distinct solutions across domains. In circle packing, EvoX achieves state-of-the-art performance by constructing a hexagonal-lattice core packing and refining it via constrained SLSQP optimization. In MinMaxMinDist (d=3 d=3), EvoX discovers a 14-point configuration in ℝ 3\mathbb{R}^{3} by solving a constrained extremal-distance problem initialized from structured polyhedral seeds. For the third autocorrelation inequality problem, EvoX constructs a 1024-bin discretized function over the interval [−0.25, 0.25][-0.25,\,0.25], computes its autoconvolution via FFT, and applies gradient-based optimization to minimize the peak magnitude. These solutions outperform those discovered by existing evolutionary systems with fixed, manually designed search strategies.

### 6.2 Main Results: System Performance Problems

Table 3: Main results for system problems. We report mean and best scores over three runs. “↑\uparrow” denotes maximization and “↓\downarrow” minimization. All methods (except human / prior AI) are run for 100 iterations using GPT-5 and Gemini-3.0-Pro. Bold denotes the best LLM-based open framework result per model. Green cells indicate matches or improvements over human SOTA results. 

We also evaluate EvoX on six real-world system optimization tasks spanning expert placement load balancing (EPLB), GPU sharing (PRISM), LLM-driven analytics (LLM-SQL), multi-cloud broadcast optimization (Cloudcast), transaction scheduling, and telemetry repair (Table[3](https://arxiv.org/html/2602.23413#S6.T3 "Table 3 ‣ 6.2 Main Results: System Performance Problems ‣ 6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery")). EvoX exceeds human-best results on all six benchmarks under Gemini-3.0-Pro.

For mean performance, EvoX achieves the best score on all tasks under GPT-5 and 5 of 6 tasks under Gemini-3.0-Pro. Telemetry repair is the only case where OpenEvolve achieves a slightly higher mean. For best performance, EvoX outperforms or ties all baselines on all tasks under GPT-5 and 5 of 6 tasks under Gemini-3.0-Pro.

Qualitatively, EvoX uncovers new system optimization algorithms. For example, in Cloudcast, EvoX discovers a Steiner-tree–based multicast routing strategy that builds a shortest-path distance graph over regions and jointly optimizes all destinations, reducing transfer cost beyond prior heuristics. In PRISM, EvoX identifies a model-placement strategy that minimizes peak KV-cache pressure by binary-searching a global load threshold and applying Best-Fit-Decreasing packing.

### 6.3 Main Results: Algorithmic and Research Problems

![Image 4: Refer to caption](https://arxiv.org/html/2602.23413v1/x2.png)

(a)Average performance across 10 ALE-Bench-Lite tasks.

![Image 5: Refer to caption](https://arxiv.org/html/2602.23413v1/x3.png)

(b)Performance on 172 Frontier-CS tasks.

Figure 3: Algorithm and Research Challenges. EvoX achieves the highest average private performance with GPT-5 on 10 different ALE-Bench-Lite problems and highest median score across 172 different Frontier-CS challenges. 

We also evaluate EvoX on algorithmic and research benchmarks (Figure[3](https://arxiv.org/html/2602.23413#S6.F3 "Figure 3 ‣ 6.3 Main Results: Algorithmic and Research Problems ‣ 6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery")).

ALE-Bench-Lite. As shown in Figure[3(a)](https://arxiv.org/html/2602.23413#S6.F3.sf1 "Figure 3(a) ‣ Figure 3 ‣ 6.3 Main Results: Algorithmic and Research Problems ‣ 6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery"), we evaluate EvoX on 10 tasks from ALE-Bench-Lite[[15](https://arxiv.org/html/2602.23413#bib.bib18 "ALE-Bench: a benchmark for long-horizon objective-driven algorithm engineering")], derived from AtCoder Heuristic Contests. Following prior work[[17](https://arxiv.org/html/2602.23413#bib.bib19 "ShinkaEvolve: towards open-ended and sample-efficient program evolution")], all methods are initialized from the ALE-Agent baseline and evaluated using private scores.

EvoX achieves the strongest overall performance, attaining the highest average private score (1958.2), outperforming ALE-Agent (1874.8), OpenEvolve (1902.9), and our reproduction of ShinkaEvolve result (1914.6)[[17](https://arxiv.org/html/2602.23413#bib.bib19 "ShinkaEvolve: towards open-ended and sample-efficient program evolution")]. On AHC016 (graph classification), EvoX finds a solution that replaces graph edit distance with block-pattern signatures, a noise-aware representation enabling efficient matching. On AHC024 (map compression), it improves simulated annealing with boundary-aware tracking, prioritizing boundary updates to improve search progress. Gains from EvoX are smaller on some tasks (e.g., AHC025), where a strong initial solution might bias the search toward a local optimum.

Frontier-CS. We evaluate EvoX on 172 tasks from Frontier-CS[[20](https://arxiv.org/html/2602.23413#bib.bib61 "FrontierCS: evolving challenges for evolving intelligence")], a large-scale benchmark of open-ended computer science problems. Each task is scored from 0–100, where 100 corresponds to the best-known human or optimal solution. As shown in Figure[3(b)](https://arxiv.org/html/2602.23413#S6.F3.sf2 "Figure 3(b) ‣ Figure 3 ‣ 6.3 Main Results: Algorithmic and Research Problems ‣ 6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery"), EvoX achieves the strongest overall performance, reaching a mean score of 62.6 and median of 75.5. This improves over OpenEvolve by 24% in mean and 34% in median score, over GEPA by 22% and 34%, and over ShinkaEvolve by 30% and 63%, respectively. These gains demonstrate that EvoX discovers higher-quality solutions while maintaining stronger consistency across diverse competitive programming tasks.

### 6.4 Ablations

![Image 6: Refer to caption](https://arxiv.org/html/2602.23413v1/fig/all_algorithms_combined_higher_swapped.png)

(a)Effect of different initial search strategies

![Image 7: Refer to caption](https://arxiv.org/html/2602.23413v1/fig/cost_quality_plot.png)

(b)Cost-quality tradeoff.

Figure 4: Search strategy evolution and cost-quality tradeoffs on the Heilbronn triangle task.(a) Dashed lines show fixed strategies, while solid lines show EvoX initialized from each strategy and allowed to evolve. Regardless of initialization, EvoX continues improving beyond the fixed strategy. (b) Cost-quality tradeoff under the GPT-5 model. To exceed a score of 0.031, EvoX and GEPA both require less than $1 in LLM generation cost, compared to ShinkaEvolve ($7.6) and OpenEvolve ($15.4). 

Starting from Different Search Strategies. Figure[4(a)](https://arxiv.org/html/2602.23413#S6.F4.sf1 "Figure 4(a) ‣ Figure 4 ‣ 6.4 Ablations ‣ 6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery") evaluates EvoX initialized from different fixed search strategies (Beam Search, Best-of-N N, Top-K K, MAP-Elites) on the Heilbronn triangle task. The Heilbronn triangle task presents a highly non-convex objective landscape, where optimization behavior is sensitive to both global configuration discovery and subsequent local improvements.

Beam search and Best-of-N N provide the strongest initializations by concentrating compute on repeatedly refining top candidates: beam search via elite expansion and pruning, and Best-of-N N through large-batch selection. Top-K K instead preserves multiple candidates, diffusing optimization pressure and slowing convergence, while MAP-Elites maintains a grid of diverse solutions and lags behind.

Regardless of initialization, fixed strategies exhibit early saturation. In contrast, EvoX consistently improves solution quality by adapting the search strategy during optimization. This behavior indicates robustness to initialization and demonstrates the benefits of strategy evolution.

Cost and Scaling. Figure[4(b)](https://arxiv.org/html/2602.23413#S6.F4.sf2 "Figure 4(b) ‣ Figure 4 ‣ 6.4 Ablations ‣ 6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery") shows the cost–quality tradeoff on the Heilbronn triangle task under the GPT-5 model. To exceed a score of 0.031, EvoX and GEPA both require less than $1 in LLM generation cost, compared to $7.6 for ShinkaEvolve and $15.4 for OpenEvolve. However, GEPA plateaus at a score of 0.0323 after around 20 iterations and shows no further improvement despite continued search. In contrast, EvoX breaks through this stagnation point and reaches a peak score of 0.0339, which is the highest among all methods, by adapting its search strategy during optimization. This shows that EvoX not only reaches competitive solutions faster, but continues to improve where fixed-strategy baselines stall.

#### Search Evolution Examples.

We provide additional case studies of search strategy evolution in Appendix[D](https://arxiv.org/html/2602.23413#A4 "Appendix D Analysis of Search Evolution ‣ EvoX: Meta-Evolution for Automated Discovery"). In some tasks, strong generator models already produce high-quality candidates, allowing even simple strategies to make progress. For example, under Gemini-3.0-Pro, random sampling alone yields substantial improvements on Cloudcast (Appendix Table[9](https://arxiv.org/html/2602.23413#A2.T9 "Table 9 ‣ B.2 Full Results ‣ Appendix B Additional Results ‣ EvoX: Meta-Evolution for Automated Discovery")), without specialized selection or variation.

In contrast, other tasks rely heavily on adaptive strategy. For Signal Processing, major improvements arise during multi-objective sampling, followed by refinement-focused phases. For Circle Packing, early free-form variation yields large gains but quickly saturates; continued progress emerges through structural variation, with later refinement contributing incremental improvements. The Heilbronn Triangle task derives its improvements predominantly from strategies emphasizing local refinement, on top of a reasonable initial configuration identified in the early iterations.

Appendix Tables[10](https://arxiv.org/html/2602.23413#A4.T10 "Table 10 ‣ Appendix D Analysis of Search Evolution ‣ EvoX: Meta-Evolution for Automated Discovery"), [11](https://arxiv.org/html/2602.23413#A4.T11 "Table 11 ‣ Appendix D Analysis of Search Evolution ‣ EvoX: Meta-Evolution for Automated Discovery"), and [12](https://arxiv.org/html/2602.23413#A4.T12 "Table 12 ‣ Appendix D Analysis of Search Evolution ‣ EvoX: Meta-Evolution for Automated Discovery") provide detailed breakdowns of these patterns. Across tasks, both selection mechanisms and variation operators evolve in task- and run-dependent ways, reflecting differences in optimization dynamics.

7 Conclusion
------------

We introduced EvoX, a meta-evolution method that jointly evolves candidate solutions and the search strategies that generate them. By adapting search strategy to the structure and stage of the optimization process, EvoX delivers consistent improvements in solution quality and cost efficiency across diverse domains. As optimization problems continue to scale in size and complexity, these results point toward a future of general-purpose, self-evolving systems that continuously refine how they search, reducing reliance on fixed, manually designed procedures.

Acknowledgments
---------------

This research is supported by NSF (IFML) CCF-2019844 and gifts from Accenture, AMD, Anyscale, Broadcom Inc., Google, IBM, Intel, Intesa Sanpaolo, Lambda, Mibura Inc, Samsung SDS, and SAP.

References
----------

*   [1]L. A. Agrawal, S. Tan, D. Soylu, N. Ziems, R. Khare, K. Opsahl-Ong, A. Singhvi, H. Shandilya, M. J. Ryan, M. Jiang, et al. (2025)GEPA: reflective prompt evolution can outperform reinforcement learning. arXiv preprint arXiv:2507.19457. Cited by: [§1](https://arxiv.org/html/2602.23413#S1.SS0.SSS0.Px1.p1.1 "Contributions. ‣ 1 Introduction ‣ EvoX: Meta-Evolution for Automated Discovery"), [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px2.p1.1 "LLM-Guided Evolutionary Search. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"), [§3](https://arxiv.org/html/2602.23413#S3.p1.1 "3 Problem Formulation ‣ EvoX: Meta-Evolution for Automated Discovery"), [§6](https://arxiv.org/html/2602.23413#S6.p2.1 "6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [2]M. Andrychowicz, M. Denil, S. Gomez, M. W. Hoffman, D. Pfau, T. Schaul, B. Shillingford, and N. de Freitas (2016)Learning to learn by gradient descent by gradient descent. In Advances in Neural Information Processing Systems, Vol. 29,  pp.3981–3989. Cited by: [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px3.p1.1 "Meta-Learning and Learning to Optimize. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [3] (2025)Codeevolve: an open source evolutionary coding agent for algorithm discovery and optimization. arXiv preprint arXiv:2510.14150. Cited by: [§B.2](https://arxiv.org/html/2602.23413#A2.SS2.p1.1 "B.2 Full Results ‣ Appendix B Additional Results ‣ EvoX: Meta-Evolution for Automated Discovery"), [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px2.p1.1 "LLM-Guided Evolutionary Search. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"), [§6](https://arxiv.org/html/2602.23413#S6.p2.1 "6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [4]Z. Cai, X. Guo, Y. Pei, J. Feng, J. Su, J. Chen, Y. Zhang, W. Ma, M. Wang, and H. Zhou (2025)Flex: continuous agent evolution via forward learning from experience. arXiv preprint arXiv:2511.06449. Cited by: [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px2.p2.1 "LLM-Guided Evolutionary Search. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [5]T. Chen, X. Chen, W. Chen, H. Heaton, J. Liu, Z. Wang, and W. Yin (2022)Learning to optimize: a primer and a benchmark. Journal of Machine Learning Research 23 (189),  pp.1–59. Cited by: [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px3.p1.1 "Meta-Learning and Learning to Optimize. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [6]X. Chen, C. Liang, D. Huang, E. Real, K. Wang, H. Pham, X. Dong, T. Luong, C. Hsieh, Y. Lu, et al. (2023)Symbolic discovery of optimization algorithms. Advances in neural information processing systems 36,  pp.49205–49233. Cited by: [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px3.p1.1 "Meta-Learning and Learning to Optimize. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [7]A. Cheng, S. Liu, M. Pan, Z. Li, B. Wang, A. Krentsel, T. Xia, M. Cemri, J. Park, S. Yang, et al. (2025)Barbarians at the gate: how AI is upending systems research. arXiv preprint arXiv:2510.06189. Cited by: [Table 5](https://arxiv.org/html/2602.23413#A1.T5 "In A.2 System Performance Optimization ‣ Appendix A Benchmark Details ‣ EvoX: Meta-Evolution for Automated Discovery"), [§1](https://arxiv.org/html/2602.23413#S1.p1.1 "1 Introduction ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [8]F. Chollet, M. Knoop, G. Kamradt, B. Landers, and H. Pinkard (2025)Arc-agi-2: a new challenge for frontier ai reasoning systems. arXiv preprint arXiv:2505.11831. Cited by: [§B.1](https://arxiv.org/html/2602.23413#A2.SS1.p1.1 "B.1 ARC-AGI Evaluation ‣ Appendix B Additional Results ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [9]H. Chopra and C. Shah (2025)Feedback-aware monte carlo tree search for efficient information seeking in goal-oriented conversations. arXiv preprint arXiv:2501.15056. Cited by: [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px2.p1.1 "LLM-Guided Evolutionary Search. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [10]C. Fernando, D. Banarse, H. Michalewski, S. Osindero, and T. Rocktäschel (2023)Promptbreeder: self-referential self-improvement via prompt evolution. arXiv preprint arXiv:2309.16797. Cited by: [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px2.p1.1 "LLM-Guided Evolutionary Search. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [11]B. Georgiev, J. Gómez-Serrano, T. Tao, and A. Z. Wagner (2025)Mathematical exploration and discovery at scale. arXiv preprint arXiv:2511.02864. Cited by: [§1](https://arxiv.org/html/2602.23413#S1.p4.1 "1 Introduction ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [12]Google DeepMind (2025-12)Gemini 3 Pro model card. Technical report Google DeepMind. External Links: [Link](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Pro-Model-Card.pdf)Cited by: [§6](https://arxiv.org/html/2602.23413#S6.p5.2 "6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [13]Q. Guo, R. Wang, J. Guo, B. Li, K. Song, X. Tan, G. Liu, J. Bian, and Y. Yang (2023)Evoprompt: connecting llms with evolutionary algorithms yields powerful prompt optimizers. arXiv e-prints,  pp.arXiv–2309. Cited by: [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px2.p1.1 "LLM-Guided Evolutionary Search. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [14]E. Hemberg, S. Moskal, and U. O’Reilly (2024)Evolving code with a large language model. Genetic Programming and Evolvable Machines 25 (2),  pp.21. Cited by: [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px2.p1.1 "LLM-Guided Evolutionary Search. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [15]Y. Imajuku, K. Horie, Y. Iwata, K. Aoki, N. Takahashi, and T. Akiba (2025)ALE-Bench: a benchmark for long-horizon objective-driven algorithm engineering. arXiv preprint arXiv:2506.09050. Cited by: [Table 6](https://arxiv.org/html/2602.23413#A1.T6 "In A.3 Algorithmic and Research Problems ‣ Appendix A Benchmark Details ‣ EvoX: Meta-Evolution for Automated Discovery"), [§1](https://arxiv.org/html/2602.23413#S1.p1.1 "1 Introduction ‣ EvoX: Meta-Evolution for Automated Discovery"), [§6.3](https://arxiv.org/html/2602.23413#S6.SS3.p2.1 "6.3 Main Results: Algorithmic and Research Problems ‣ 6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery"), [§6](https://arxiv.org/html/2602.23413#S6.p1.1 "6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [16]J. Jiang, T. Ding, and Z. Zhu (2026)DeltaEvolve: accelerating scientific discovery through momentum-driven evolution. arXiv preprint arXiv:2602.02919. Cited by: [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px2.p1.1 "LLM-Guided Evolutionary Search. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [17]R. T. Lange, Y. Imajuku, and E. Cetin (2025)ShinkaEvolve: towards open-ended and sample-efficient program evolution. External Links: 2509.19349, [Link](https://arxiv.org/abs/2509.19349)Cited by: [§1](https://arxiv.org/html/2602.23413#S1.SS0.SSS0.Px1.p1.1 "Contributions. ‣ 1 Introduction ‣ EvoX: Meta-Evolution for Automated Discovery"), [§1](https://arxiv.org/html/2602.23413#S1.p3.1 "1 Introduction ‣ EvoX: Meta-Evolution for Automated Discovery"), [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px2.p1.1 "LLM-Guided Evolutionary Search. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"), [§3](https://arxiv.org/html/2602.23413#S3.SS0.SSS0.Px3.p1.6 "Search strategy. ‣ 3 Problem Formulation ‣ EvoX: Meta-Evolution for Automated Discovery"), [§3](https://arxiv.org/html/2602.23413#S3.p1.1 "3 Problem Formulation ‣ EvoX: Meta-Evolution for Automated Discovery"), [§6.3](https://arxiv.org/html/2602.23413#S6.SS3.p2.1 "6.3 Main Results: Algorithmic and Research Problems ‣ 6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery"), [§6.3](https://arxiv.org/html/2602.23413#S6.SS3.p3.1 "6.3 Main Results: Algorithmic and Research Problems ‣ 6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery"), [§6](https://arxiv.org/html/2602.23413#S6.p2.1 "6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [18]K. Lee, I. Fischer, Y. Wu, D. Marwood, S. Baluja, D. Schuurmans, and X. Chen (2025)Evolving deeper llm thinking. arXiv preprint arXiv:2501.09891. Cited by: [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px2.p1.1 "LLM-Guided Evolutionary Search. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [19]A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegreffe, U. Alon, N. Dziri, S. Prabhumoye, Y. Yang, et al. (2023)Self-refine: iterative refinement with self-feedback. Advances in neural information processing systems 36,  pp.46534–46594. Cited by: [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px1.p1.1 "Context and Memory Management. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [20]Q. Mang, W. Chai, Z. Li, H. Mao, S. Zhou, A. Du, H. Li, S. Liu, E. Chen, Y. Wang, et al. (2025)FrontierCS: evolving challenges for evolving intelligence. arXiv preprint arXiv:2512.15699. Cited by: [Table 6](https://arxiv.org/html/2602.23413#A1.T6 "In A.3 Algorithmic and Research Problems ‣ Appendix A Benchmark Details ‣ EvoX: Meta-Evolution for Automated Discovery"), [§6.3](https://arxiv.org/html/2602.23413#S6.SS3.p4.1 "6.3 Main Results: Algorithmic and Research Problems ‣ 6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery"), [§6](https://arxiv.org/html/2602.23413#S6.p1.1 "6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [21]L. Metz, N. Maheswaranathan, J. Nixon, D. Freeman, and J. Sohl-Dickstein (2019)Understanding and correcting pathologies in the training of learned optimizers. In International Conference on Machine Learning, Vol. 97,  pp.4556–4565. Cited by: [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px3.p1.1 "Meta-Learning and Learning to Optimize. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [22]A. Novikov, N. Vu, M. Eisenberger, E. Dupont, P. Huang, A. Z. Wagner, S. Shirobokov, B. Kozlovskii, F. J. R. Ruiz, A. Mehrabian, et al. (2025)AlphaEvolve: a coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131. Cited by: [§1](https://arxiv.org/html/2602.23413#S1.p1.1 "1 Introduction ‣ EvoX: Meta-Evolution for Automated Discovery"), [§1](https://arxiv.org/html/2602.23413#S1.p3.1 "1 Introduction ‣ EvoX: Meta-Evolution for Automated Discovery"), [§1](https://arxiv.org/html/2602.23413#S1.p4.1 "1 Introduction ‣ EvoX: Meta-Evolution for Automated Discovery"), [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px2.p1.1 "LLM-Guided Evolutionary Search. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"), [§3](https://arxiv.org/html/2602.23413#S3.p1.1 "3 Problem Formulation ‣ EvoX: Meta-Evolution for Automated Discovery"), [§6](https://arxiv.org/html/2602.23413#S6.p2.1 "6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery"), [§6](https://arxiv.org/html/2602.23413#S6.p4.1 "6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [23]OpenAI (2025-08)GPT-5 system card. Technical report OpenAI. External Links: [Link](https://cdn.openai.com/gpt-5-system-card.pdf)Cited by: [§6](https://arxiv.org/html/2602.23413#S6.p5.2 "6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [24]C. Packer, V. Fang, S. Patil, K. Lin, S. Wooders, and J. Gonzalez (2023)MemGPT: towards llms as operating systems.. Cited by: [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px1.p1.1 "Context and Memory Management. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [25]J. Pourcel, C. Colas, and P. Oudeyer (2025)Self-improving language models for evolutionary program synthesis: a case study on arc-agi. arXiv preprint arXiv:2507.14172. Cited by: [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px2.p2.1 "LLM-Guided Evolutionary Search. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [26]Y. Qu, M. Y. R. Yang, A. Setlur, L. Tunstall, E. E. Beeching, R. Salakhutdinov, and A. Kumar (2025)Optimizing test-time compute via meta reinforcement fine-tuning. External Links: 2503.07572, [Link](https://arxiv.org/abs/2503.07572)Cited by: [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px3.p1.1 "Meta-Learning and Learning to Optimize. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [27]B. Romera-Paredes, M. Barekatain, A. Novikov, M. Balog, M. P. Kumar, E. Dupont, F. J. R. Ruiz, J. S. Ellenberg, P. Wang, O. Fawzi, P. Kohli, and A. Fawzi (2024)Mathematical discoveries from program search with large language models. Nature 625 (7995),  pp.468–475. External Links: [Document](https://dx.doi.org/10.1038/s41586-023-06924-6), [Link](https://doi.org/10.1038/s41586-023-06924-6)Cited by: [§3](https://arxiv.org/html/2602.23413#S3.p1.1 "3 Problem Formulation ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [28]A. Sharma (2025)OpenEvolve: an open-source evolutionary coding agent. Note: GitHub External Links: [Link](https://github.com/codelion/openevolve)Cited by: [§1](https://arxiv.org/html/2602.23413#S1.SS0.SSS0.Px1.p1.1 "Contributions. ‣ 1 Introduction ‣ EvoX: Meta-Evolution for Automated Discovery"), [§1](https://arxiv.org/html/2602.23413#S1.p3.1 "1 Introduction ‣ EvoX: Meta-Evolution for Automated Discovery"), [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px2.p1.1 "LLM-Guided Evolutionary Search. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"), [§3](https://arxiv.org/html/2602.23413#S3.SS0.SSS0.Px3.p1.6 "Search strategy. ‣ 3 Problem Formulation ‣ EvoX: Meta-Evolution for Automated Discovery"), [§3](https://arxiv.org/html/2602.23413#S3.p1.1 "3 Problem Formulation ‣ EvoX: Meta-Evolution for Automated Discovery"), [§5](https://arxiv.org/html/2602.23413#S5.p1.1 "5 Case Study: Signal Processing ‣ EvoX: Meta-Evolution for Automated Discovery"), [§6](https://arxiv.org/html/2602.23413#S6.p2.1 "6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [29]B. A. Shenoi (2005)Introduction to digital signal processing and filter design. Vol. 169, John Wiley & Sons. Cited by: [§5](https://arxiv.org/html/2602.23413#S5.p1.1 "5 Case Study: Signal Processing ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [30]N. Shinn, F. Cassano, E. Berman, A. Gopinath, K. Narasimhan, and S. Yao (2023)Reflexion: language agents with verbal reinforcement learning. External Links: 2303.11366, [Link](https://arxiv.org/abs/2303.11366)Cited by: [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px1.p1.1 "Context and Memory Management. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [31]P. Shojaee, K. Meidani, S. Gupta, A. B. Farimani, and C. K. Reddy (2024)Llm-sr: scientific equation discovery via programming with large language models. arXiv preprint arXiv:2404.18400. Cited by: [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px2.p1.1 "LLM-Guided Evolutionary Search. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [32]M. Suzgun, M. Yuksekgonul, F. Bianchi, D. Jurafsky, and J. Zou (2025)Dynamic cheatsheet: test-time learning with adaptive memory. arXiv preprint arXiv:2504.07952. Cited by: [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px2.p1.1 "LLM-Guided Evolutionary Search. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [33]Y. Wang, S. Su, Z. Zeng, E. Xu, L. Ren, X. Yang, Z. Huang, X. He, L. Ma, B. Peng, et al. (2025)ThetaEvolve: test-time learning on open problems. arXiv preprint arXiv:2511.23473. Cited by: [§B.2](https://arxiv.org/html/2602.23413#A2.SS2.p1.1 "B.2 Full Results ‣ Appendix B Additional Results ‣ EvoX: Meta-Evolution for Automated Discovery"), [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px2.p2.1 "LLM-Guided Evolutionary Search. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"), [§6](https://arxiv.org/html/2602.23413#S6.p2.1 "6 Evaluation ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [34]M. Yan, B. Peng, B. Coleman, Z. Chen, Z. Xie, Z. He, N. Sachdeva, I. Ye, W. Wang, C. Wang, et al. (2026)PACEvolve: enabling long-horizon progress-aware consistent evolution. arXiv preprint arXiv:2601.10657. Cited by: [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px2.p1.1 "LLM-Guided Evolutionary Search. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [35]H. Ye, J. Wang, Z. Cao, F. Berto, C. Hua, H. Kim, J. Park, and G. Song (2024)Reevo: large language models as hyper-heuristics with reflective evolution. Advances in neural information processing systems 37,  pp.43571–43608. Cited by: [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px2.p1.1 "LLM-Guided Evolutionary Search. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [36]M. Yuksekgonul, D. Koceja, X. Li, F. Bianchi, J. McCaleb, X. Wang, J. Kautz, Y. Choi, J. Zou, C. Guestrin, et al. (2026)Learning to discover at test time. arXiv preprint arXiv:2601.16175. Cited by: [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px2.p2.1 "LLM-Guided Evolutionary Search. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"). 
*   [37]Q. Zhang, C. Hu, S. Upasani, B. Ma, F. Hong, V. Kamanuru, J. Rainton, C. Wu, M. Ji, H. Li, et al. (2025)Agentic context engineering: evolving contexts for self-improving language models. arXiv preprint arXiv:2510.04618. Cited by: [§2](https://arxiv.org/html/2602.23413#S2.SS0.SSS0.Px1.p1.1 "Context and Memory Management. ‣ 2 Related Work ‣ EvoX: Meta-Evolution for Automated Discovery"). 

\beginsupplement

Appendix

Appendix A Benchmark Details
----------------------------

We evaluate EvoX on 196 problems: 24 optimization problems spanning three domains (mathematical optimization (8 problems), system performance (6 problems), and algorithmic challenges (10 problems)) and 172 algorithmic problems from FrontierCS.

### A.1 Mathematical Optimization

Table 4: Mathematical optimization benchmarks. Summary of optimization tasks used to evaluate EvoX, including geometric, combinatorial, and signal processing problems.

### A.2 System Performance Optimization

Table 5: System performance optimization benchmarks. Benchmarks drawn from ADRS-Bench[[7](https://arxiv.org/html/2602.23413#bib.bib17 "Barbarians at the gate: how AI is upending systems research")], capturing realistic optimization problems from production systems.

### A.3 Algorithmic and Research Problems

Table 6: Algorithmic and research problem benchmarks. We evaluate on 182 open-ended problems: 10 NP-hard optimization problems from ALE-Bench-Lite[[15](https://arxiv.org/html/2602.23413#bib.bib18 "ALE-Bench: a benchmark for long-horizon objective-driven algorithm engineering")] and 172 problems from FrontierCS[[20](https://arxiv.org/html/2602.23413#bib.bib61 "FrontierCS: evolving challenges for evolving intelligence")].

Problem / Category Count Description
ALE-Bench-Lite: NP-hard optimization from AtCoder Heuristic Contests
ahc008 (Territory)1 Control agents on a grid to place fences isolating pets; max\max satisfaction.
ahc011 (Sliding Puzzle)1 Rearrange tiles so line patterns form a large connected tree; max\max connectivity.
ahc015 (Candy Clustering)1 Cluster candies of the same flavor using global tilt operations; max\max clustering score.
ahc016 (Graph Classification)1 Design reference graphs and classify noisy, permuted query graphs; min\min classification error.
ahc024 (Map Compression)1 Compress a grid map preserving district adjacency; min\min map size.
ahc025 (Weight Balancing)1 Partition items with unknown weights into balanced groups; min\min imbalance.
ahc026 (Box Stacking)1 Rearrange numbered boxes across stacks to a target configuration; min\min moves.
ahc027 (Cleaning Route)1 Design a cyclic robot route to minimize steady-state dirt; min\min average dirtiness.
ahc039 (Fishing Net)1 Construct a rectilinear polygon enclosing targets under a perimeter budget; max\max net score.
ahc046 (Skating with Blocks)1 Visit target squares in order using moves, slides, and blocks; min\min actions.
FrontierCS: open-ended CS problems
Optimization 38 Maximize or minimize a quantitative objective over a parameterized search space under resource constraints. IDs: 1, 2, 15, 16, 22, 25, 26, 27, 28, 30, 33, 35, 36, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 52, 53, 54, 57, 58, 59, 61, 64, 68, 69, 79, 81, 86, 93, 229.
Constructive 62 Synthesize a valid structured object (e.g., packing, graph, expression) under global constraints. IDs: 0, 3, 4, 5, 6, 7, 8, 9, 13, 17, 23, 24, 60, 62, 63, 70, 72, 73, 75, 77, 80, 82, 83, 85, 87, 89, 142, 174–193, 192, 193, 203, 205, 207, 209–214, 217, 220, 222, 225, 227, 228, 239, 241.
Interactive 72 Solve a hidden-instance task via an adaptive query-response protocol, minimizing queries or interaction steps. IDs: 10, 11, 14, 101, 104, 106–113, 117, 119–125, 127, 132–135, 137, 138, 140, 141, 143–145, 147–171, 226, 231, 233, 243, 245, 247–249, 252–258.

Appendix B Additional Results
-----------------------------

We report additional evaluations to examine EvoX behavior beyond the primary optimization benchmarks discussed in the main paper.

### B.1 ARC-AGI Evaluation

ARC-AGI-2 Tasks[[8](https://arxiv.org/html/2602.23413#bib.bib41 "Arc-agi-2: a new challenge for frontier ai reasoning systems")] evaluate abstract and compositional reasoning across programmatic problem-solving instances. Although ARC-AGI-2 is not explicitly designed as an optimization benchmark, it provides a useful testbed for analyzing cross-domain robustness.

Experiments follow the evaluation protocol described in Section 4.1. OpenEvolve (OE) and EvoX operate under a matched inference budget (30 LLM iterations) per task.

Table 7: EvoX performance on ARC-AGI benchmarks. Values denote final accuracy.

These results indicate that EvoX maintains performance improvements even on reasoning-oriented tasks outside traditional optimization settings. We emphasize that ARC-AGI fundamentally differs from the evolutionary optimization regime, as standard ARC evaluation assumes strict train–test separation, whereas evolutionary frameworks typically adapt during inference.

### B.2 Full Results

We additionally compare against CodeEvolve[[3](https://arxiv.org/html/2602.23413#bib.bib38 "Codeevolve: an open source evolutionary coding agent for algorithm discovery and optimization")] and ThetaEvolve[[33](https://arxiv.org/html/2602.23413#bib.bib16 "ThetaEvolve: test-time learning on open problems")], two recent systems representing strong prior approaches for LLM-driven optimization.

Across tasks, EvoX maintains consistent performance advantages over both fixed-policy and adaptive-policy baselines. Detailed per-task breakdowns are provided below.

Table 8:  Main results for math optimization problems. We report mean and best over three runs. “↑\uparrow” / “↓\downarrow” indicate maximization / minimization. 

Table 9:  Main results for system problems. We report mean and best scores over three runs. “↑\uparrow” denotes maximization and “↓\downarrow” minimization. 

Appendix C Prompt
-----------------

Below is the prompt used for search strategy evolution, together with example mutation operator labels.

Appendix D Analysis of Search Evolution
---------------------------------------

Table 10: Search strategy evolution on Signal Processing (Gemini, 100 iterations). Initial score 0.499 →\rightarrow final 0.743. Δ\Delta: improvement achieved within the phase window; W W: window length. Variation operator π∈{local refinement,structural variation,free-form variation}\pi\in\{\textit{local refinement},\textit{structural variation},\textit{free-form variation}\}, where free-form variation is the default. 

Table 11: Search strategy evolution on Circle Packing (Gemini, 100 iterations). Initial score 0.364 →\rightarrow final 1.0004. Δ\Delta: improvement achieved within the phase window; W W: window length. Variation operator π∈{local refinement,structural variation,free-form variation}\pi\in\{\textit{local refinement},\textit{structural variation},\textit{free-form variation}\}, where free-form variation is the default. 

Table 12: Search strategy evolution on Heilbronn Triangle (Gemini, 100 iterations). Initial score 0.0 →\rightarrow final min_area≈0.0365\texttt{min\_area}\approx 0.0365 (SOTA 1.0); best at iteration 91. Δ\Delta: improvement achieved within the phase window; W W: window length. Variation operator π∈{local refinement,structural variation,free-form variation}\pi\in\{\textit{local refinement},\textit{structural variation},\textit{free-form variation}\}, where free-form variation is the default. 

### D.1 Case Study: Search Evolution in Circle Packing

We now illustrate EvoX through a case study on the Circle Packing task, where the objective is to construct a program that maximizes packing efficiency under geometric constraints. The generated program must balance multiple competing factors, including achieving dense configurations, maintaining numerical stability, preventing overlap violations, and converging reliably within the evaluation budget. Candidate programs are evaluated using a normalized packing score reflecting achieved density under constraint satisfaction.

Starting from an initial score of 0.364, EvoX reaches a near-optimal score of 1.0004. Table[11](https://arxiv.org/html/2602.23413#A4.T11 "Table 11 ‣ Appendix D Analysis of Search Evolution ‣ EvoX: Meta-Evolution for Automated Discovery") summarizes the sequence of evolved strategies.

Static Baseline Behavior. Uniform parent sampling combined with free-form variation produces rapid early gains (+0.59), primarily by discovering coarse geometric heuristics such as greedy placement rules, collision-aware perturbations, and simple local displacement schemes. However, progress quickly saturates. Free-form variation largely generates parameter-level modifications, adjusting perturbation radii, iteration counts, or threshold values without introducing qualitatively new optimization mechanisms. Because uniform sampling increasingly selects structurally similar programs, variation operates over a narrowing region of the program space.

As a result, improvements exhibit diminishing returns. Local heuristic updates frequently induce oscillatory adjustments, repeated near-collisions, and unstable refinements near dense configurations. Constraint satisfaction remains externally enforced through penalties or rejection rules rather than intrinsically modeled by the update mechanism. Without adaptive strategy evolution, the baseline cannot redirect search toward mechanisms capable of coordinated global adjustments.

Evolving Search Strategy in EvoX.

Exploration-Dominated Phase. EvoX initially mirrors baseline behavior, rapidly identifying stable geometric constructions (+0.59). At this stage, improvements arise primarily from discovering viable placement schemes rather than refining precise optimization dynamics.

Diversity-Inducing Strategies. As structural redundancy emerges within the population, EvoX evolves usage-penalized and tiered sampling strategies. These mechanisms alter search dynamics by discouraging repeated selection of recently sampled parents and explicitly sampling candidates across score bands. This transition prevents over-exploitation of early high-performing but structurally limited heuristics while increasing exposure to partially successful yet structurally diverse programs. Although immediate gains are modest (+0.04, +0.004), these phases preserve population diversity, which proves critical for subsequent mechanism discovery.

Mechanism Discovery via Structural Variation. Under stagnation-aware sampling, structural variation produces programs that introduce a fundamentally different optimization paradigm based on constrained numerical optimization using SLSQP. This transition represents a qualitative shift in solution construction. Earlier heuristic programs perform sequential local adjustments, modifying circle positions independently through handcrafted displacement rules. Such updates are inherently myopic and frequently induce constraint violations or oscillatory corrections. In contrast, SLSQP-based programs formulate packing as a constrained optimization problem in which circle positions become jointly optimized variables and overlap constraints are explicitly encoded.

This paradigm-level shift fundamentally alters refinement dynamics. Constraint satisfaction becomes intrinsic to the update rule rather than externally enforced, coordinated gradient-based updates mitigate oscillatory behaviors, and the optimizer implicitly captures higher-order interactions among circles. Rather than relying on independent local perturbations, updates now reflect globally coordinated corrections across multiple variables. This mechanism discovery effectively breaks prior stagnation patterns and unlocks further improvements.

Refinement-Dominated Phase. Once SLSQP-based solutions emerge, large structural edits increasingly destabilize high-quality configurations. EvoX therefore shifts operator bias toward local refinement. Refinement now operates within a significantly stronger optimization framework, primarily adjusting convergence parameters, constraint tolerances, and step-size dynamics. Quantile-biased sampling further stabilizes search by relaxing strict elite selection, preventing overfitting to brittle local optima. Incremental gains observed in this phase reflect convergence stabilization rather than structural discovery.

This example highlights a central property of EvoX: strategy evolution enables optimization-mechanism discovery rather than merely improving sampling efficiency. Early improvements arise from discovering viable geometric heuristics, whereas later improvements require identifying mechanisms capable of coordinated global updates. Static strategies typically fail to induce such transitions because variation operators alone rarely trigger paradigm shifts, uniform sampling suppresses structurally novel candidates, and exploitative refinement reinforces local heuristic biases.

By evolving strategies conditioned on both historical feedback and population-state signals, EvoX identifies when refinement-based search becomes ineffective and increases structural variation pressure until new optimization mechanisms emerge. Overall, Circle Packing illustrates how adaptive strategy evolution governs phase transitions in search dynamics, enabling qualitative improvements that static search policies fail to realize.

