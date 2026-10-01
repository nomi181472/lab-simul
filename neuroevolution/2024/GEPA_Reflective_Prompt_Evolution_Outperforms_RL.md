Title: GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning

URL Source: https://arxiv.org/html/2507.19457

Markdown Content:
Lakshya A Agrawal 1, Shangyin Tan 1, Dilara Soylu 2, Noah Ziems 4, 

Rishi Khare 1, Krista Opsahl-Ong 5, Arnav Singhvi 2,5, Herumb Shandilya 2, 

Michael J Ryan 2, Meng Jiang 4, Christopher Potts 2, Koushik Sen 1, 

Alexandros G. Dimakis 1,3, Ion Stoica 1, Dan Klein 1, Matei Zaharia 1,5, Omar Khattab 6
1 UC Berkeley 2 Stanford University 3 BespokeLabs.ai 4 Notre Dame 5 Databricks 6 MIT

###### Abstract

Large language models (LLMs) are increasingly adapted to downstream tasks via reinforcement learning (RL) methods like Group Relative Policy Optimization (GRPO), which often require thousands of rollouts to learn new tasks. We argue that the interpretable nature of language can often provide a much richer learning medium for LLMs, compared with policy gradients derived from sparse, scalar rewards. To test this, we introduce GEPA (Ge netic-Pa reto), a prompt optimizer that thoroughly incorporates natural language reflection to learn high-level rules from trial and error. Given any AI system containing one or more LLM prompts, GEPA samples system-level trajectories (e.g., reasoning, tool calls, and tool outputs) and reflects on them in natural language to diagnose problems, propose and test prompt updates, and combine complementary lessons from the Pareto frontier of its own attempts. As a result of GEPA’s design, it can often turn even just a few rollouts into a large quality gain. Across four tasks, GEPA outperforms GRPO by 10% on average and by up to 20%, while using up to 35x fewer rollouts. GEPA also outperforms the leading prompt optimizer, MIPROv2, by over 10% across two LLMs, and demonstrates promising results as an inference-time search strategy for code optimization.

1 Introduction
--------------

![Image 1: Refer to caption](https://arxiv.org/html/2507.19457v1/x1.png)

(a) HotpotQA, Qwen3 8B

![Image 2: Refer to caption](https://arxiv.org/html/2507.19457v1/x2.png)

(b) HoVer, Qwen3 8B

Figure 1: A comparison of the learning behavior of our proposed GEPA prompt optimizer against a state-of-the-art prompt optimizer (MIPROv2) and the GRPO (24,000 rollouts) algorithm. As more rollouts are sampled, the prompt optimizers can learn much more quickly than GRPO. GEPA substantially outperforms both GRPO and MIPROv2 in final score. The Test-set star markers demonstrate the performance gap in a held-out set of questions.

Large language models (LLMs) have enabled the development of agents and systems that combine fuzzy natural-language behavior specification with tools like retrieval and code execution. These types of systems raise the question of how LLMs should be “optimized” for the best downstream performance within their harness. One popular approach for adapting LLMs to downstream tasks is reinforcement learning with verifiable rewards (RLVR), including algorithms such as Group Relative Policy Optimization (GRPO)(Shao et al., [2024](https://arxiv.org/html/2507.19457v1#bib.bib33)). Such RL methods cast success metrics as a scalar reward observed at the end of each rollout(Lambert, [2025](https://arxiv.org/html/2507.19457v1#bib.bib18)) and use these rewards to estimate gradients for policy improvement.

While these RL approaches are effective, they typically require tens of thousands of rollouts in practice to fit new tasks. For example, recent works leveraging GRPO across a range of tasks typically use up to hundreds of thousands of rollouts for training(Chen et al., [2025b](https://arxiv.org/html/2507.19457v1#bib.bib6); Wu et al., [2025c](https://arxiv.org/html/2507.19457v1#bib.bib51); Zhang et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib61); Jin et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib13); Si et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib35); Wang et al., [2025a](https://arxiv.org/html/2507.19457v1#bib.bib45); Java et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib11); Chen et al., [2025a](https://arxiv.org/html/2507.19457v1#bib.bib5); Wu et al., [2025a](https://arxiv.org/html/2507.19457v1#bib.bib49); Sha et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib32); Lin et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib21); Peng et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib28); Song et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib37)). This sample inefficiency can quickly become a serious bottleneck: many downstream LLM applications invoke expensive tool calls, have limited inference budget for sampling from the LLM itself, or simply cannot finetune the weights of the largest or best-performing LLMs.

We observe that the rollouts sampled from even highly sophisticated LLM systems can be serialized into traces of natural (and formal) language, as they contain nothing but the instructions of each LLM module, the resulting LLM reasoning chains, tool calls, and potentially the internal workings of the reward function (for example, compiler error messages, before they are collapsed into scalar rewards). Because such serialized trajectories can be readily understood by modern LLMs, we argue that algorithms that learn deliberately in natural language by reflecting on these trajectories can potentially make much more effective use of the strong language priors that LLMs have, compared with standard RL approaches.

To operationalize this, we introduce GEPA (Genetic-Pareto), a reflective prompt optimizer for compound AI systems that merges textual reflection with multi-objective evolutionary search. GEPA iteratively mutates every prompt within the AI system in light of natural language feedback drawn from new rollouts. In each mutation, the candidate prompt is derived from an ancestor, accumulating high-level lessons derived from observations and LLM feedback. To avoid the local optima that afflict greedy prompt updates, GEPA maintains a Pareto front: instead of evolving only the global best prompt, it stochastically explores the top-performing prompts for each problem instance, thereby diversifying strategies and encouraging robust generalization.

We evaluate GEPA on four diverse tasks—multi-hop reasoning (HotpotQA; Yang et al. [2018](https://arxiv.org/html/2507.19457v1#bib.bib56)), instruction following (IFBench; Pyatkin et al. [2025b](https://arxiv.org/html/2507.19457v1#bib.bib30)), privacy-aware delegation (PUPA; Li et al. [2025a](https://arxiv.org/html/2507.19457v1#bib.bib19)), and retrieval-augmented verification (HoVer; Jiang et al. [2020](https://arxiv.org/html/2507.19457v1#bib.bib12))—using both open (Qwen3 8B; Yang et al. [2025](https://arxiv.org/html/2507.19457v1#bib.bib54); Team [2025](https://arxiv.org/html/2507.19457v1#bib.bib42)) and proprietary (GPT-4.1 mini; OpenAI [2025](https://arxiv.org/html/2507.19457v1#bib.bib25)) models. Our results show that GEPA demonstrates robust generalization and is highly sample-efficient: on Qwen3 8B, GEPA outperforms GRPO (24,000 rollouts with LoRA) by up to 19% while requiring up to 35×\times fewer rollouts. Overall, GEPA achieves an average improvement of +10% over GRPO across all tasks. Furthermore, GEPA surpasses the previous state-of-the-art prompt optimizer, MIPROv2(Opsahl-Ong et al., [2024](https://arxiv.org/html/2507.19457v1#bib.bib26)), on every benchmark and model, obtaining aggregate optimization gains of +14%, more than doubling the gains achieved by MIPROv2 (+7%).

Even qualitatively, GEPA generated prompts can be highly effective. Figure[2](https://arxiv.org/html/2507.19457v1#S1.F2 "Figure 2 ‣ 1 Introduction ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning") highlights excerpts from a prompt crafted by GEPA for the query creation module of a multi-hop question answering system (Used in HotpotQA). We also find that in most cases, even a single reflective prompt update can give large improvements (as highlighted in the optimization trajectory in Figure[5](https://arxiv.org/html/2507.19457v1#S3.F5 "Figure 5 ‣ 3.3 Pareto-based candidate selection ‣ 3 GEPA: Reflective Prompt Evolution ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning")). These results demonstrate that reflective prompt evolution using language feedback enables substantial sample efficiency and robust generalization, providing a practical path to optimizing complex, real-world AI workflows in data- or budget-constrained environments. Finally, we also show promising preliminary results demonstrating GEPA’s use as an inference-time search strategy for code optimization over NPUEval(Kalade & Schelle, [2025](https://arxiv.org/html/2507.19457v1#bib.bib14)) and KernelBench(Ouyang et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib27)).

Figure 2: This figure shows an example prompt generated by GEPA for the second-hop document retrieval to be performed in a multi-hop question-answer system, along with the seed prompt it started with. Appendix[I](https://arxiv.org/html/2507.19457v1#A9 "Appendix I Examples of best prompts for every benchmark ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning") compares GEPA’s prompts for all tasks with prompts generated by MIPROv2.

2 Problem Statement
-------------------

#### Compound AI Systems.

We follow related work in defining a compound AI system as any modular system composed of one or more language model (LLM) invocations, potentially interleaved with external tool calls, orchestrated through arbitrary control flow. This definition subsumes a broad class of real-world LLM-based AI systems, including agents, multi-agent systems, and general-purpose scaffolding techniques like ReAct(Yao et al., [2023](https://arxiv.org/html/2507.19457v1#bib.bib57)), Archon(Saad-Falcon et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib31)), etc. Following Soylu et al. ([2024](https://arxiv.org/html/2507.19457v1#bib.bib38)); Khattab et al. ([2024](https://arxiv.org/html/2507.19457v1#bib.bib16)); Opsahl-Ong et al. ([2024](https://arxiv.org/html/2507.19457v1#bib.bib26)); Tan et al. ([2025](https://arxiv.org/html/2507.19457v1#bib.bib41)), we formalize such a system as Φ=(M,C,𝒳,𝒴)\Phi=(M,C,\mathcal{X},\mathcal{Y}), where M=⟨M 1,…,M|M|⟩M=\langle M_{1},\ldots,M_{|M|}\rangle denotes language modules, C C specifies control flow logic, and 𝒳\mathcal{X}, 𝒴\mathcal{Y} are global input/output schemas. Each module M i=(π i,θ i,𝒳 i,𝒴 i)M_{i}=(\pi_{i},\theta_{i},\mathcal{X}_{i},\mathcal{Y}_{i}) is an LLM subcomponent: π i\pi_{i} is its (system) prompt including instructions and few-shot demonstrations; θ i\theta_{i} the underlying model weights; 𝒳 i,𝒴 i\mathcal{X}_{i},\mathcal{Y}_{i} are input/output schemas. At runtime, C C orchestrates the sequencing and invocation of modules—e.g., passing outputs from one module to another, invoking modules conditionally, or leveraging tool APIs. This way, C C can invoke different modules in any order multiples of times.

#### Compound AI System Optimization.

Given Φ\Phi, let Π Φ=⟨π 1,…,π|M|⟩\Pi_{\Phi}=\langle\pi_{1},\ldots,\pi_{|M|}\rangle denote the collection of all module prompts and Θ Φ=⟨θ 1,…,θ|M|⟩\Theta_{\Phi}=\langle\theta_{1},\ldots,\theta_{|M|}\rangle the set of module weights. The learnable parameters are thus ⟨Π,Θ⟩Φ\langle\Pi,\Theta\rangle_{\Phi}. For a task instance (x,m)(x,m)—where x x maps to the input schema 𝒳\mathcal{X} and m m contains evaluator metadata (e.g., gold answers, evaluation rubrics, code unit tests)—the system induces an output y=Φ​(x;⟨Π,Θ⟩Φ)y=\Phi(x;\langle\Pi,\Theta\rangle_{\Phi}). A metric μ:𝒴×ℳ→[0,1]\mu:\mathcal{Y}\times\mathcal{M}\to[0,1] then measures the output quality of y y with respect to metadata m m (for example by calculating, exact match, F1, pass rate, etc.). The optimization problem is thus defined by:

⟨Π∗,Θ∗⟩Φ=arg⁡max⟨Π,Θ⟩Φ⁡𝔼(x,m)∼𝒯​[μ​(Φ​(x;⟨Π,Θ⟩Φ),m)],\displaystyle\langle\Pi^{*},\Theta^{*}\rangle_{\Phi}=\arg\max_{\langle\Pi,\Theta\rangle_{\Phi}}\mathbb{E}_{(x,m)\sim\mathcal{T}}\left[\mu\big(\Phi(x;\langle\Pi,\Theta\rangle_{\Phi}),\,m\big)\right],(1)

where 𝒯\mathcal{T} is a task distribution.

#### Sample-Efficient Optimization.

In many real-world scenarios, rollouts—concretely, invocations of Φ\Phi plus evaluation by μ\mu—are often computationally, monetarily, or timewise expensive. The optimizer is thus limited to at most B B rollouts on a dataset 𝒟 train={(x,m)i}i=1 N\mathcal{D}_{\text{train}}=\{(x,m)_{i}\}_{i=1}^{N} with full access to μ\mu. The goal is to identify parameters ⟨Π∗,Θ∗⟩Φ\langle\Pi^{*},\Theta^{*}\rangle_{\Phi} that maximize held-out performance, subject to not exceeding the rollout budget B B:

⟨Π∗,Θ∗⟩Φ=arg⁡max⟨Π,Θ⟩Φ⁡𝔼(x,m)∼𝒯​[μ​(Φ​(x;⟨Π,Θ⟩Φ),m)]s.t.#​rollouts≤B.\displaystyle\langle\Pi^{*},\Theta^{*}\rangle_{\Phi}=\arg\max_{\langle\Pi,\Theta\rangle_{\Phi}}\mathbb{E}_{(x,m)\sim\mathcal{T}}\left[\mu\big(\Phi(x;\langle\Pi,\Theta\rangle_{\Phi}),\,m\big)\right]\quad\text{s.t.}\quad\#\text{rollouts}\leq B.(2)

This formulation captures the core challenge motivating our work: _How can we extract maximal learning signal from every expensive rollout to enable effective adaptation of complex, modular AI systems in low-data or budget-constrained settings?_

3 GEPA: Reflective Prompt Evolution
-----------------------------------

![Image 3: Refer to caption](https://arxiv.org/html/2507.19457v1/x3.png)

Figure 3: GEPA works iteratively—proposing a new candidate in every iteration by improving some existing candidates using one of the two strategies (Reflective Prompt Mutation (Section [3.2](https://arxiv.org/html/2507.19457v1#S3.SS2 "3.2 Reflective Prompt Mutation ‣ 3 GEPA: Reflective Prompt Evolution ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning")) or System Aware Merge (Appendix [F](https://arxiv.org/html/2507.19457v1#A6 "Appendix F Merge: System-aware crossover strategy for Compound AI optimization ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"))), first evaluating them on a minibatch, and if improved, evaluating on a larger dataset. Instead of selecting the best performing candidate to mutate always, which can lead to a local-optimum, GEPA introduces Pareto-based candidate sampling (Section[3.3](https://arxiv.org/html/2507.19457v1#S3.SS3 "3.3 Pareto-based candidate selection ‣ 3 GEPA: Reflective Prompt Evolution ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning")), which filters and samples from the list of best candidates per task, ensuring sufficient diversity. Overall, these design decisions allow GEPA to be highly sample-efficient while demonstrating strong generalization.

Algorithm 1 GEPA: Reflective Evolutionary Prompt Optimizer

1:Inputs: System

Φ\Phi
, dataset

𝒟 train\mathcal{D}_{\text{train}}
, eval metric

μ\mu
, feedback function

μ f\mu_{f}
, budget

B B

2:Hyperparams: minibatch size

b b
, Pareto set size

n p​a​r​e​t​o n_{pareto}

3:Split

𝒟 train\mathcal{D}_{\text{train}}
into

𝒟 feedback\mathcal{D}_{\text{feedback}}
,

𝒟 pareto\mathcal{D}_{\text{pareto}}
, s.t.

|D p​a​r​e​t​o|=n p​a​r​e​t​o|D_{pareto}|=n_{pareto}

4:Initialize candidates

𝒫←[Φ]\mathcal{P}\leftarrow[\Phi]
, parents

𝒜←[None]\mathcal{A}\leftarrow[\text{None}]

5:for each

(x i,m i)(x_{i},m_{i})
in

𝒟 pareto\mathcal{D}_{\text{pareto}}
do

6:

S Φ​[i]←μ​(Φ​(x i),m i)S_{\Phi}[i]\leftarrow\mu(\Phi(x_{i}),m_{i})

7:end for

8:while budget

B B
not exhausted do

9:

k←k\leftarrow
SelectCandidate(

𝒫,S\mathcal{P},S
)

10:

j←j\leftarrow
SelectModule(

Φ k\Phi_{k}
)

11:

ℳ←\mathcal{M}\leftarrow
minibatch of size

b b
from

𝒟 feedback\mathcal{D}_{\text{feedback}}

12: Gather feedback, scores, traces for

Φ k​[j]\Phi_{k}[j]
on

ℳ\mathcal{M}
using

μ f\mu_{f}

13:

π j′←\pi_{j}^{\prime}\leftarrow
UpdatePrompt(

π j\pi_{j}
, feedbacks, traces[j])

14:

Φ′←\Phi^{\prime}\leftarrow
Copy of

Φ k\Phi_{k}
w/ module

j j
updated by

π j′\pi_{j}^{\prime}

15:

σ\sigma
,

σ′\sigma^{\prime}←\leftarrow
avg score on

ℳ\mathcal{M}
(before, after)

16:if

σ′\sigma^{\prime}
improved then

17: Add

Φ′\Phi^{\prime}
to

𝒫\mathcal{P}
; Add

k k
to

𝒜\mathcal{A}

18:for each

(x i,m i)(x_{i},m_{i})
in

𝒟 pareto\mathcal{D}_{\text{pareto}}
do

19:

S Φ′​[i]←μ​(Φ′​(x i),m i)S_{\Phi^{\prime}}[i]\leftarrow\mu(\Phi^{\prime}(x_{i}),m_{i})

20:end for

21:end if

22:end while

23:return

Φ∗\Phi^{*}
maximizing average score on

𝒟 pareto\mathcal{D}_{\text{pareto}}

Algorithm 2 Pareto-based candidate selection

1:function SelectCandidate(

𝒫,S\mathcal{P},S
)

2: // Build instance-wise Pareto sets

3:for each

i i
do

4:

s∗​[i]←max k⁡S 𝒫​[k]​[i]s^{*}[i]\leftarrow\max_{k}S_{\mathcal{P}[k]}[i]

5:

𝒫∗​[i]←{𝒫​[k]:S 𝒫​[k]​[i]=s∗​[i]}\mathcal{P}^{*}[i]\leftarrow\{\mathcal{P}[k]:S_{\mathcal{P}[k]}[i]=s^{*}[i]\}

6:end for

7:

𝒞←\mathcal{C}\leftarrow
unique candidates in

⋃i 𝒫∗​[i]\bigcup_{i}\mathcal{P}^{*}[i]

8:

D←∅D\leftarrow\emptyset

9:while there exists

Φ∈𝒞∖D\Phi\in\mathcal{C}\setminus D
dominated by another in

𝒞∖D\mathcal{C}\setminus D
do

10:

D←D∪{Φ}D\leftarrow D\cup\{\Phi\}

11:end while

12: Remove

D D
from each

𝒫∗​[i]\mathcal{P}^{*}[i]
to get

𝒫^∗​[i]\hat{\mathcal{P}}^{*}[i]

13: Let

f​[Φ]=f[\Phi]=
number of

i i
for which

Φ∈𝒫^∗​[i]\Phi\in\hat{\mathcal{P}}^{*}[i]

14: Sample

Φ k\Phi_{k}
from

𝒞^\hat{\mathcal{C}}
with probability

∝f​[Φ k]\propto f[\Phi_{k}]

15:return index

k k
of

Φ k\Phi_{k}
in

𝒫\mathcal{P}

16:end function

Figure 4: (Left) GEPA’s core algorithm for reflective prompt evolution. GEPA works iteratively, in each iteration, selecting some of the current candidates to evolve (line 7), executing the identified candidate on a minibatch of rollouts, while utilizing a special feedback function μ f\mu_{f} to gather module specific feedback when available (lines 9-10, described in detail in Section[3.2](https://arxiv.org/html/2507.19457v1#S3.SS2 "3.2 Reflective Prompt Mutation ‣ 3 GEPA: Reflective Prompt Evolution ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning")), using an LLM to reflectively update the prompt (line 11), and evaluating whether the system instantiated with the new prompt improved the performance on the minibatch (line 14). If improved, GEPA then proceeds to evaluate the new system candidate on the full D p​a​r​e​t​o D_{pareto} set, adding it to the list of candidates tracked and marking the new system’s parent. (Right) The SelectCandidate subprocedure used by GEPA’s core algorithm is tasked with identifying the best candidate to evolve in the next optimization iteration. GEPA’s chief candidate selection strategy is to find non-dominated candidates in the Pareto frontier (of all task instances), and stochastically select one of them based on their appearance frequency in the Pareto front.

We introduce GEPA (Genetic-Pareto), a sample-efficient optimizer for compound AI systems motivated by three core principles: genetic prompt evolution (Section[3.1](https://arxiv.org/html/2507.19457v1#S3.SS1 "3.1 Genetic Optimization Loop ‣ 3 GEPA: Reflective Prompt Evolution ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning")), reflection using natural language feedback (Section[3.2](https://arxiv.org/html/2507.19457v1#S3.SS2 "3.2 Reflective Prompt Mutation ‣ 3 GEPA: Reflective Prompt Evolution ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning")), and Pareto-based candidate selection (Section[3.3](https://arxiv.org/html/2507.19457v1#S3.SS3 "3.3 Pareto-based candidate selection ‣ 3 GEPA: Reflective Prompt Evolution ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning")). Figure[3](https://arxiv.org/html/2507.19457v1#S3.F3 "Figure 3 ‣ 3 GEPA: Reflective Prompt Evolution ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning") gives an overview of GEPA and the full GEPA algorithm is formalized in Figure[4](https://arxiv.org/html/2507.19457v1#S3.F4 "Figure 4 ‣ 3 GEPA: Reflective Prompt Evolution ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning").

GEPA receives the following inputs: A compound AI system Φ\Phi instantiated with simple prompts to be optimized, training dataset D t​r​a​i​n D_{train} (consisting of task instances (x,m)(x,m) as described in Section[2](https://arxiv.org/html/2507.19457v1#S2.SS0.SSS0.Px2 "Compound AI System Optimization. ‣ 2 Problem Statement ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning")), the standard evaluation metric μ\mu for the task, a feedback function μ f\mu_{f} (introduced in Section[3.2](https://arxiv.org/html/2507.19457v1#S3.SS2 "3.2 Reflective Prompt Mutation ‣ 3 GEPA: Reflective Prompt Evolution ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning")) and the total rollout budget B B.

### 3.1 Genetic Optimization Loop

Given a compound AI system Φ\Phi, the goal of the optimization process is to identify a set of parameters ⟨Π,Θ⟩Φ\langle\Pi,\Theta\rangle_{\Phi} that maximize the score over a task distribution. GEPA starts by initializing a candidate pool 𝒫\mathcal{P}, where a candidate is a concrete instantiation of the learnable parameters of the compound system, ⟨Π,Θ⟩Φ\langle\Pi,\Theta\rangle_{\Phi}. Initially, the candidate pool consists only of the base system’s parameters as the sole candidate. GEPA then proceeds in an optimization loop, iteratively proposing new candidates and adding them to the pool, continuing this process until the evaluation budget is exhausted.

Iteratively, GEPA proposes increasingly effective candidates by modifying existing ones through mutation or crossover, informed by learning signals from newly gathered rollouts and while tracking each new candidates’ ancestry. This enables GEPA to accumulate lessons along the genetic tree as optimization progresses. Each new candidate inherits learning signals from its parents, as well as signals from the current rollout.

During each iteration, GEPA identifies promising candidates from the candidate pool (candidate selection), proposes a new candidate—possibly by mutating prompts in a module based on reflective feedback or by performing crossover between two candidates—and evaluates this new variant on a minibatch of tasks. If the newly proposed candidate demonstrates improved performance relative to its parent(s) on the local minibatch, then GEPA adds the new candidate to the candidate pool 𝒫\mathcal{P}. This involves tracking internal data structures including tracking the ancestry of the new candidate, along with the full evaluation of the new candidate on a D p​a​r​e​t​o D_{pareto}, a validation set used for candidate selection.

After the budget is depleted, GEPA returns the candidate with the best aggregate performance on D p​a​r​e​t​o D_{pareto}.

### 3.2 Reflective Prompt Mutation

Natural language traces generated during the execution of a compound AI system offer rich visibility into the behavior and responsibilities of each module, as they capture the intermediate inferences and underlying reasoning steps. When these traces are paired with the final outcome of the system (e.g., success or failure), they provide substantial diagnostic value, allowing practitioners to trace errors or successes back to specific decisions made at the module level. LLMs can then leverage these traces via reflection to perform implicit credit assignment, attributing responsibility for the final outcome to the relevant modules. This process of reflection can then be used to make targeted updates to individual modules, making large and effective updates to the whole system’s behavior.

GEPA operationalizes this as follows: Given a selected candidate to mutate in the current iteration of the optimization loop, GEPA updates the system with the candidate parameters, selects a target module within the system to improve (via round robin to ensure all modules receive updates), and generates a few rollouts over a minibatch sampled from the training dataset, recording their outcomes (success/failure). By examining the execution traces of the system, GEPA identifies the target module’s inputs, outputs, and reasoning. With this, GEPA uses an LLM to reflectively examine this information, attributing successes or failures to elements of the module’s prompt (or omission thereof), and propose new instructions for the target module. A new candidate is then proposed as a copy of the current candidate, with the target module’s prompt updated to the new proposed prompt. The meta-prompt used by GEPA to perform reflective prompt update is presented in Appendix[B](https://arxiv.org/html/2507.19457v1#A2 "Appendix B GEPA’s Reflection and Prompt Update Meta Prompt ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning").

Evaluation trace as diagnostic signal: While the system’s own execution traces already provide useful information to enable successful reflection and prompt updates, we identify another source of highly diagnostic information: The evaluation metric μ\mu. Often, the evaluation metric μ\mu applies rich strategies to perform evaluations to arrive at a final score. For example, code evaluation environments run a series of steps (compilation, execution, profiling, etc.) each of which produce natural language traces, before providing a scalar reward.

We propose the use of these evaluation traces in addition to the system’s own execution traces to perform reflective credit assignment, and targeted prompt updates. GEPA operationalizes this as a simple update to the evaluation metric μ\mu, to create a feedback function μ f\mu_{f}, which identifies relevant textual traces produced during the evaluation metric’s execution, and returns the final score along with feedback_text. Whenever available, such a feedback function can also provide module-level feedback (for example, in multi-hop systems, the evaluator can provide feedback after each hop of the system).

### 3.3 Pareto-based candidate selection

GEPA is a highly modular algorithm capable of supporting various strategies for selecting candidates in each iteration of optimization. Crucially, the choice of candidate selection strategy determines the exploration-exploitation tradeoff adopted by the optimizer. A naive strategy is to always select the best-performing candidate in the pool. However, this can cause the optimizer to get stuck in a local optimum within the prompt space: once a dominant strategy is found, it becomes difficult to surpass, and the optimizer exhausts its budget without learning new, potentially better strategies. An example search tree generated with this strategy is demonstrated in Figure[6(a)](https://arxiv.org/html/2507.19457v1#S5.F6.sf1 "In Figure 6 ‣ 5 Results and Analysis ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"). Specifically, note how the search process found one new strategy (the first child node), and then kept trying to improve it, failing to do so across many iterations, finally exhausting all the rollouts budget.

To address this, GEPA employs a Pareto-based “illumination” strategy(Mouret & Clune, [2015](https://arxiv.org/html/2507.19457v1#bib.bib23)), as shown in Algorithm[2](https://arxiv.org/html/2507.19457v1#alg2 "Algorithm 2 ‣ Figure 4 ‣ 3 GEPA: Reflective Prompt Evolution ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"). Specifically, GEPA identifies the highest score achieved for each individual training instance across all candidates in the pool, creating a “Pareto frontier” of scores achieved by the optimization process so far. GEPA then compiles a list of candidates that achieve the best score on at least one training task. This filters the pool down to candidates that incorporate “winning” strategies, preserving every valuable insight discovered in any reflective mutation. Next, GEPA prunes candidates that are strictly dominated: for instance, if Candidate 2 has the best score on Task 1 only, but Candidate 3 achieves that same best score on Task 1 and the best on Task 2, Candidate 2 is removed. Finally, GEPA stochastically samples a candidate from this pruned list, assigning higher selection probability to candidates that achieved the best score across more training instances.

In practice, this strategy helps GEPA escape local optima without expanding the search excessively. By focusing resources on promising candidates that have already demonstrated impactful, “winning” strategies, GEPA efficiently balances exploration and exploitation, allowing for continual improvement within the optimization budget.

![Image 4: Refer to caption](https://arxiv.org/html/2507.19457v1/x4.png)

Figure 5: GEPA’s reflective prompt mutation systematically incorporates task-specific nuances, leading to substantial improvements in performance. This figure visualizes the optimization trajectory taken by GEPA, presenting an annotated subtree from Figure[23(d)](https://arxiv.org/html/2507.19457v1#A8.F23.sf4 "In Figure 23 ‣ Appendix H GEPA Search Trees ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning") (for the privacy-preserving delegation task PUPA) to demonstrate the iterative enhancements made to the prompts. The progression from the base prompt (candidate 0) to the best performing prompt (candidate 11) is highlighted with red arrows, and key prompt changes at each step are annotated beside the corresponding nodes. Full-length instructions for these iterations are provided in Appendix[G.1](https://arxiv.org/html/2507.19457v1#A7.SS1 "G.1 Prompts at intermediate stages for PUPA ‣ Appendix G Visualizing the Iterative Refinement achieved by GEPA ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"). Each prompt refinement in this trajectory adds targeted nuances informed by ongoing optimization, illustrating how GEPA’s process accumulates lessons to continually boost task performance.

4 Evaluation Setup
------------------

In this section, we detail our experimental setup. For each benchmark, we adopt a standard three-way data split: train, validation, and test. The train split is fully accessible to the optimizers, allowing them to read and utilize the text and labels of the training instances for program tuning. Although optimizers may monitor the performance of candidate parameters (like model checkpoints) by tracking scores on the validation set (to implement early stopping, for example), direct access to the content of validation instances is restricted. The test set remains entirely held out and is inaccessible throughout the optimization process; performance on this split is only measured post-optimization to assess the performance of the optimized program.

### 4.1 Benchmarks, Reference compound AI systems, and Feedback Functions

To rigorously evaluate the performance of GEPA and and compare it against current state-of-the-art compound AI system optimizers, we assemble a diverse suite of benchmarks mostly obtained from Tan et al. ([2025](https://arxiv.org/html/2507.19457v1#bib.bib41)), each paired with available Compound AI Systems.

HotpotQA(Yang et al., [2018](https://arxiv.org/html/2507.19457v1#bib.bib56)) is a large-scale question-answering dataset consisting of 113K Wikipedia-based question-answer pairs. It features questions that require reasoning over multiple supporting documents. We modify the last hop of the HoVerMultiHop program (described below) to answer the question instead of generating another query, and the rest of the system remains unmodified. The textual feedback module identifies the set of relevant documents remaining to be retrieved at each stage of the program, and provides that as feedback to the modules at that stage. We use 150 examples for training, 300 for validation, and 300 for testing.

IFBench(Pyatkin et al., [2025b](https://arxiv.org/html/2507.19457v1#bib.bib30)) introduced a benchmark specifically designed to assess language models’ ability to follow precise human instructions, especially output constraints (e.g., “answer only with yes or no”, or “mention a word at least three times”). The IFBench test set consists of 58 new and out-of-distribution output constraints and instructions to test system’s ability to generalize to new task constraints. Pyatkin et al. ([2025b](https://arxiv.org/html/2507.19457v1#bib.bib30)) also release IFTrain and IF-RLVR Train data(Pyatkin et al., [2025a](https://arxiv.org/html/2507.19457v1#bib.bib29)) which are used for training. We split the IF-RLVR Train into our train/val sets, and IFBench as our test set in order to ensure that the optimizers do not access the new, unseen constraints being tested in IFBench. We design a 2-stage system, that first attempts to answer the user query, and then in the second stage, rewrites the answer following the constraints. The textual feedback module provides the descriptions of constraints satsified and failed-to-be-satisifed by the system’s response. Our splits contain 150 training examples, 300 for validation, and 294 for testing.

HoVer(Jiang et al., [2020](https://arxiv.org/html/2507.19457v1#bib.bib12)) is an open-domain multihop fact extraction and claim verification benchmark built on a Wikipedia-based corpus requiring complex reasoning across multiple sentences and documents, typically involving multiple wikipedia articles. Following Tan et al. ([2025](https://arxiv.org/html/2507.19457v1#bib.bib41)), the systems are evaluated for their ability to write queries in multiple hops to retrieve all relevant wikipedia documents (gold documents) required to make the claim. We obtain the HoverMultiHop program from Tan et al. ([2025](https://arxiv.org/html/2507.19457v1#bib.bib41)), which performs up to 3-hop retrievals using 2 query writer modules, and 2 document summary modules. The textual feedback module simply identifies the set of correct documents retrieved, and the set of documents remaining to be retrieved, and returns them as feedback text. For HoVer, we use 150 examples for training, 300 for validation, and 300 for testing.

PUPA(Li et al., [2025a](https://arxiv.org/html/2507.19457v1#bib.bib19)) propose the task of Privacy-Conscious Delegation: addressing real-world user queries using an ensemble of trusted and untrusted models. The core challenges are maintaining high response quality while minimizing leakage of personally identifiable information (PII) to untrusted models. Li et al. ([2025a](https://arxiv.org/html/2507.19457v1#bib.bib19)) also present PAPILLON, a compound AI system consisting of 2 modules, a user query rewriter and a response rewriter, run over the trusted model, along with an intermediate call to the untrusted model with the rewritten query. The feedback text simply provides the breakdown of the aggregate score, consisting of a response quality score and a PII leakage score. The dataset is split into 111 training examples, 111 for validation, and 221 for testing.

### 4.2 Models and Inference Parameters

We evaluate GEPA and baseline optimizers using two contemporary LLMs, chosen to represent both open-source and commercial model families. Each compound AI system is instantiated once per model, with all modules (e.g., retrievers, rewriters, answer generators) relying on the same model. All models are allowed a context window of upto 16384 tokens for inference.

Qwen3 8B(Yang et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib54)): For our open-source experiments (including GRPO), we use Qwen3-8B. Following the recommended settings as per Team ([2025](https://arxiv.org/html/2507.19457v1#bib.bib42)), we use a decoding temperature of 0.6, top-p of 0.95, and top-k of 20 for training as well as inference.

GPT-4.1 Mini(OpenAI, [2025](https://arxiv.org/html/2507.19457v1#bib.bib25)): For comparison with large commercial models, we use GPT-4.1 mini (openai/gpt-4.1-mini-2025-04-14) accessed via the OpenAI API with a model temperature of 1.0.

### 4.3 Optimizers

Baseline: The base program is directly evaluated without any further optimization applied.

MIPROv2(Opsahl-Ong et al., [2024](https://arxiv.org/html/2507.19457v1#bib.bib26)): MIPROv2 is a widely used compound AI system prompt optimizer and has been integrated into the DSPy(Khattab et al., [2024](https://arxiv.org/html/2507.19457v1#bib.bib16)) and llama-prompt-ops(AI, [2025](https://arxiv.org/html/2507.19457v1#bib.bib4)) frameworks. It works by jointly optimizing both instructions and demonstrations using Bayesian optimization. For each program module, it first bootstraps candidate sets of instructions and demonstrations, assigning uniform priors over their utilities. Candidate assignments are proposed with the Tree-Structured Parzen Estimator (TPE), and the Bayesian model is updated based on evaluation scores to favor high-performing candidates. The most probable sets of instructions and demonstrations are then selected and validated to obtain the final optimized program configuration. In order to

All MIPROv2 optimization runs are performed with the a​u​t​o=h​e​a​v​y auto=heavy setting, which corresponds to proposing 18 instruction candidates and 18 bootstrapped few-shot sets. Hence, across benchmarks, the exact number of rollouts varies depending on the number of trials it takes to bootstrap examples (finding 18 successful solution instances), the required number of Bayesian search steps (determined by the number of modules in the system), and size of the valset. Overall, MIPROv2’s rollouts ranged from a minimum of 2270 (for PUPA) to maximum of 6926 (for HoVer).

GRPO(Shao et al., [2024](https://arxiv.org/html/2507.19457v1#bib.bib33)): Group Relative Policy Optimization (GRPO) is a reinforcement learning algorithm that estimates advantages in a group-relative manner. We use the GRPO implementation for compound AI systems provided and open-sourced by[Ziems, Soylu, and Agrawal et al. (2025)](https://arxiv.org/html/2507.19457v1#bib.bib66) to perform our experiments. Across all training runs, each training step uses a group size of 12, with 4 training instances per step (total batch size 48, with per device train batch size 1). Training employs LoRA(Hu et al., [2022](https://arxiv.org/html/2507.19457v1#bib.bib10)) with rank dimension 16, α=64\alpha=64, and dropout 0.05, using bf16 precision targeting the projection modules [q,k,v,o,up,down,gate][\mathrm{q},\mathrm{k},\mathrm{v},\mathrm{o},\mathrm{up},\mathrm{down},\mathrm{gate}]. We use a learning rate of 1×10−5 1\times 10^{-5}, β=0.01\beta=0.01, reward scale normalization, and gradient norm clipping of 0.1. Gradients are accumulated for 20 steps before each update, with a “constant with warmup learning” rate scheduler. Non-reentrant gradient checkpointing is enabled to further reduce memory usage. We manually explore several values for [LR, beta, norm clipping] hyperparameters. All GRPO optimizations run for 500 training steps, amounting to fixed 24,000 rollouts, with validation performed every 20 training steps, which is used to implement early stopping. All training experiments are performed on 1xH100/A100 (80 GB memory) with separate GPUs for inference rollouts.

GEPA: GEPA is our optimizer, based on the algorithm described in Section[3](https://arxiv.org/html/2507.19457v1#S3 "3 GEPA: Reflective Prompt Evolution ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"). We evaluate 2 variants of our main optimizer GEPA: GEPA and GEPA+Merge, along with 2 ablations created by replacing the Pareto-based sampling strategy with a naive, SelectBestCandidate strategy (SelectBestCandidate and SelectBestCandidate+Merge). All GEPA optimization runs use a minibatch size of 3, and merge is invoked a maximum of 5 times during the optimization run, when enabled. To ensure a fair comparison with MIPROv2, we align the computational budget between GEPA and MIPROv2 on a per-benchmark basis. The training set from each benchmark is used as D f​e​e​d​b​a​c​k D_{feedback} (which is used to derive the training signals, as discussed in Section[3](https://arxiv.org/html/2507.19457v1#S3 "3 GEPA: Reflective Prompt Evolution ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning")) and the validation set is used as D p​a​r​e​t​o D_{pareto}. Specifically, since MIPROv2’s total rollout budget depends on factors such as validation set size and the number of modules, we first record the number of rollouts expended by MIPROv2 for each benchmark, and then cap GEPA’s optimization to match this rollout budget. While differences in proposal and validation procedures cause the exact budget usage by the systems to be slightly different, the discrepancy is always within 10.15%. This protocol ensures that any performance differences arise from the optimization algorithms themselves, rather than from differences in search budget. The exact rollout countts for each optimizer is visualized in Appendix[C](https://arxiv.org/html/2507.19457v1#A3 "Appendix C Performance vs. Budget (Rollouts) Curves (Contd.) ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning").

5 Results and Analysis
----------------------

Table 1: Benchmark results for different optimizers over Qwen3 8B and GPT-4.1 Mini models across multiple tasks.

Table[1](https://arxiv.org/html/2507.19457v1#S5.T1 "Table 1 ‣ 5 Results and Analysis ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning") and Figure[9](https://arxiv.org/html/2507.19457v1#A1.F9 "Figure 9 ‣ Appendix A Results (Contd.) ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning") summarize our main results, from which we derive the following observations:

Observation 1: Reflective Prompt Evolution is highly sample-efficient and can outperform weight-space reinforcement learning: Across all four benchmarks, GEPA demonstrates rapid adaptation and robust generalization in compound AI systems—outperforming GRPO (24,000 rollouts with LoRA) by up to 19% while using up to 35×35\times fewer rollouts.

GEPA attains optimal test set performance on HotpotQA, IFBench, HoVer, and PUPA with only 6,438, 678 (35× fewer), 6,858, and 2,157 (11× fewer) rollouts, respectively—surpassing GRPO by 19%, 2.73%, 13.66%, and 5.19% on these tasks. Notably, GEPA matches GRPO’s best validation scores after only 402, 330, 1179, and 306 rollouts respectively, achieving up to 78× greater sample efficiency. Furthermore, the combined GEPA+Merge approach outperforms GRPO by an even wider margin of 21% at a comparable rollout budget as GEPA. We especially highlight the +8.16% achieved by GEPA+Merge on IFBench with GPT-4.1 mini, even though it contains new, completely out-of-domain constraints in the test set.

It is also important to note that the majority of GEPA’s counted rollouts are allocated to the validation set, where scores are utilized solely for candidate selection and not for producing learning signals. If we restrict the analysis to train set rollouts—the rollouts actually used for learning—GEPA requires just 737, 79, 558, and 269 training rollouts to reach optimal performance on HotpotQA, IFBench, HoVer, and PUPA, respectively. To match GRPO’s best validation scores, GEPA achieves this with only 102, 32, 6, and 179 train rollouts, underscoring the high sample efficiency of learning based on reflective prompt evolution.

Since tracking candidates’ validation performance accounts for the majority of GEPA’s rollout budget, sample efficiency could be further improved by evaluating on a smaller validation set or by tracking scores on dynamically selected validation subsets instead of the full set—both of which we propose as directions for future work.

Figures[1(a)](https://arxiv.org/html/2507.19457v1#S1.F1.sf1 "In Figure 1 ‣ 1 Introduction ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"), [11(c)](https://arxiv.org/html/2507.19457v1#A3.F11.sf3 "In Figure 11 ‣ Appendix C Performance vs. Budget (Rollouts) Curves (Contd.) ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"), [1(b)](https://arxiv.org/html/2507.19457v1#S1.F1.sf2 "In Figure 1 ‣ 1 Introduction ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning") and [13(c)](https://arxiv.org/html/2507.19457v1#A3.F13.sf3 "In Figure 13 ‣ Appendix C Performance vs. Budget (Rollouts) Curves (Contd.) ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning") show the full performance-vs-rollouts curve for all optimizers over benchmarks HotpotQA, IFBench, HoVer and PUPA, respectively.

Observation 2: Reflective prompt evolution enables Instruction-Optimization alone to outperform joint Instruction and Few-Shot Optimization: We compare GEPA with MIPROv2—a state-of-the-art joint instruction and few-shot optimizer—using two leading models (GPT-4.1 mini and Qwen3 8B) across four diverse tasks. Our experiments show that GEPA consistently outperforms MIPROv2 in all settings, achieving margins as high as 11.1% for GPT-4.1 mini and 10.3% for Qwen3 8B. Furthermore, GEPA and GEPA+Merge more than double the aggregate gains over baseline seen with MIPROv2 across all benchmarks and both models (+16.02% and +14.29% vs +7.04% for MIPROv2).

While prior works such as Opsahl-Ong et al. ([2024](https://arxiv.org/html/2507.19457v1#bib.bib26)) and Wan et al. ([2024](https://arxiv.org/html/2507.19457v1#bib.bib44)) have provided compelling evidence for the effectiveness of few-shot example optimization—often outperforming instruction-based approaches—our findings suggest an exciting shift in this trend. We attribute this primarily to recent advances in the instruction-following and self-reflective abilities of LLMs, as well as the design choices in GEPA that capitalize on these improved capabilities. To further contextualize our findings, we redo the study on generalization gap (the difference between validation and test set performance for optimized prompts) as proposed by Wan et al. ([2024](https://arxiv.org/html/2507.19457v1#bib.bib44)). The results presented in figure[14](https://arxiv.org/html/2507.19457v1#A4.F14 "Figure 14 ‣ Appendix D Generalization Gap ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning") reinforce these observations: reflectively evolved instructions now demonstrate a lower generalization gap, underscoring both advancements in model capabilities and the benefits of GEPA’s design. We see this as a reflection of the continuous evolution of LLMs and GEPA’s ability to effectively leverage these improvements.

Finally, we provide the full-length optimized prompts produced by GEPA for all systems, benchmarks, and models in Appendix[I](https://arxiv.org/html/2507.19457v1#A9 "Appendix I Examples of best prompts for every benchmark ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"), alongside the corresponding MIPROv2 prompts. Notably, in contrast to prior findings where instruction optimization yielded improvements primarily through quasi-exemplars(Wan et al., [2024](https://arxiv.org/html/2507.19457v1#bib.bib44)), GEPA’s prompts frequently contain detailed declarative instructions for completing the task, as illustrated in Figure[2](https://arxiv.org/html/2507.19457v1#S1.F2 "Figure 2 ‣ 1 Introduction ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning").

Observation 3: The next-candidate selection strategy strongly influences the optimization trajectory and final performance, with Pareto-based sampling providing a distinct advantage. GEPA seeks to iteratively refine prompts by leveraging feedback from new rollouts. In order to test the impact of our Pareto-based candidate selection strategy, we consider a straightforward baseline for instantiating SelectCandidate strategy: always selecting the currently best-performing candidate. As shown by the ablation results in Table[2](https://arxiv.org/html/2507.19457v1#S5.T2 "Table 2 ‣ 5 Results and Analysis ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"), this approach often leads to sub-optimal exploration of the prompt search space ultimately leading to poor performance—GEPA with Pareto-based sampling strategy outperforms the SelectBestCandidate strategy by as much as 8.17%, maintaining an aggregate margin of +6.4% across all benchmarks. Figure[6](https://arxiv.org/html/2507.19457v1#S5.F6 "Figure 6 ‣ 5 Results and Analysis ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning") illustrates the stark difference in optimization trajectories between this naïve strategy and our proposed Pareto-based sampling-strategy. Always choosing the current best candidate tends to yield immediate improvement in the next iteration, but then causes the optimizer to stall, expending its entire rollout budget attempting to further improve this specific candidate. In contrast, our Pareto-based sampling method expands the search by considering all Pareto-optimal candidates (representing all the “winning” strategies discovered so far), ensuring a tight balance between exploration and exploitation tradeoffs—ultimately converging to a higher-performing solution within the same rollout budget.

Table 2: Benchmark results for different optimizers over Qwen3 8B and GPT-4.1 Mini models across multiple tasks.

![Image 5: Refer to caption](https://arxiv.org/html/2507.19457v1/x5.png)

(a) SelectBestCandidate Strategy

![Image 6: Refer to caption](https://arxiv.org/html/2507.19457v1/x6.png)

(b) Pareto-based candidate sampling

Figure 6: Comparing the impact of different candidate selection strategies. (Left) As can be seen, selecting the best-performing candidate in every iteration led to a local-optima after one iteration, leading to suboptimal search performance. (Right) On the other hand, using pareto-based candidate selection strategy, GEPA was able to generate a balanced search tree, finding a better performing program within the same budget.

Observation 4: Instruction-optimized prompts are computationally cheaper and generalize better than few-shot demonstration prompts: In addition to their strong generalization capabilities, reflectively evolved instructions offer a significant practical advantage: they are often much shorter and thus computationally more efficient than few-shot demonstration prompts. This advantage becomes especially clear for complex tasks, where even a single few-shot demonstration can be prohibitively long. The problem is further exacerbated when few-shot examples are optimized using state-of-the-art methods such as MIPROv2, which jointly optimizes multiple demonstrations to be used simultaneously, further increasing prompt length.

In contrast, reflectively evolved instructions—such as those generated by GEPA—maintain compactness while providing large performance gains (as demonstrated in Lessons 1 and 2). To illustrate this, we compare GEPA’s and MIPROv2’s prompt lengths (see Figure[16](https://arxiv.org/html/2507.19457v1#A5.F16 "Figure 16 ‣ Appendix E Cost vs. Performance Analysis for optimized systems ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning")). Notably, prompts produced by GEPA and GEPA+Merge are up to 9.2×9.2\times shorter than those from MIPROv2, representing a substantial improvement in efficiency, alongside performance improvements.

Moreover, we observe a trend where, in aggregate, optimizers that achieve higher performance tend to produce shorter prompts (see Figure[15](https://arxiv.org/html/2507.19457v1#A5.F15 "Figure 15 ‣ Appendix E Cost vs. Performance Analysis for optimized systems ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning")). This reduction in prompt size has a significant impact—not only reducing runtime cost for downstream tasks (as all API-providers meter the input tokens), but also decreasing latency and improving the overall efficiency of LLM-serving systems(Kwon et al., [2023](https://arxiv.org/html/2507.19457v1#bib.bib17); Zheng et al., [2024](https://arxiv.org/html/2507.19457v1#bib.bib64); Agrawal et al., [2023](https://arxiv.org/html/2507.19457v1#bib.bib3); Yu et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib58)).

Observation 5: System aware crossover strategies can provide large gains, but the optimal budget allocation between mutation and crossover, as well as when to invoke merge needs further study: We identify a unique system-aware crossover strategy and operationalize it as Merge (described in Appendix[F](https://arxiv.org/html/2507.19457v1#A6 "Appendix F Merge: System-aware crossover strategy for Compound AI optimization ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning")). GEPA+Merge can outperform GEPA by as much as 5%, providing an aggregate 2% additional improvement over the already strong performance established by GEPA. Detailed results are available in Table[1](https://arxiv.org/html/2507.19457v1#S5.T1 "Table 1 ‣ 5 Results and Analysis ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"). We attribute these gains to the ability of GEPA+Merge to identify distinct optimization lineages, that have learnt complementary strategies (by evolving distinct modules), and merging them by picking the best version of different modules from each of these lineages to propose a single, optimal candidate.

While in our analysis, we found GEPA+Merge works especially well for GPT-4.1 Mini, it lead to performance degradation when used with Qwen3 8B. Even Qwen3 8B benefits from Merge on one out of four tasks. We attribute these discrepancies to the way the rollout budget is allocated between reflective mutation and crossover, and the timing of invocation of the crossover strategy. In our experiments, we fixed the same hyperparameters for GPT-4.1 Mini and Qwen3 8B, leading to suboptimal choice for Qwen3 8B. Intuitively, crossover would provide the maximum benefit, when there are independent lineages that perform well. Hence, the hyperparameters should be chosen such that Merge is invoked once the optimization tree has evolved sufficiently different lineages. We propose the study of such adaptive techniques as future work.

6 GEPA For Inference-Time Search
--------------------------------

While the primary focus of this paper is sample-efficient adaptation of AI systems to new tasks, preliminary findings suggest that GEPA may also serve as a promising inference-time search technique. This can be achieved by passing the set of tasks to be solved (for example, a list of Pytorch modules to be converted to CUDA) as the training set to GEPA, ensuring that both D t​r​a​i​n D_{train} and D p​a​r​e​t​o D_{pareto} contain the full set of tasks. This way, GEPA can “overfit” the set of tasks, iteratively proposing better solutions to every problem. We also note that this allows GEPA to apply lessons and insights extracted from rollouts for one task to other tasks. To explore this use case, we conduct preliminary experiments using GEPA as an inference-time search technique for code-generation tasks on two hardware platforms: writing kernels for AMD’s recently introduced XDNA2 Architecture(Advanced Micro Devices, [2025](https://arxiv.org/html/2507.19457v1#bib.bib1)) using an early version of the NPUEval benchmark(Kalade & Schelle, [2025](https://arxiv.org/html/2507.19457v1#bib.bib14)), and generating CUDA code for NVIDIA-V100 GPUs using KernelBench(Ouyang et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib27)).

A distinguishing aspect of these experiments is the use of the feedback function μ f\mu_{f} to dynamically inject domain-specific knowledge into the optimization process. Specifically, kernel development expertise—often codified in technical manuals and documentation—can be selectively surfaced by retrieving relevant manual sections based on rollout failures (e.g., compiler error messages). By using error information to make targetted retrieval queries, GEPA promotes integration of architectural best practices into prompt evolution, as exemplified by the detailed prompt for NPUEval shown in Figure[25](https://arxiv.org/html/2507.19457v1#A10.F25 "Figure 25 ‣ J.1 NPUEval: Kernel Code Generation for new hardware architecture ‣ Appendix J GEPA for Inference-Time Search ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"). We also note that generation stochasticity (temperature based sampling) is eliminated by operating under a cache; this ensures that observed improvements tie closely to inference scaling through prompt updates and GEPA’s diverse prompt exploration, rather than stochasticity in the model’s sampling process.

NPU Kernels: We create a sequential refinement agent that iteratively generates kernels (up to 10 times) based on feedback like compiler errors and profiling results (Sequential10), and evaluate the Best-of-N generation. With GPT-4o alone, Sequential10 reaches only 4.25% mean vector utilization. Adding RAG, sourced from technical manuals, improves this to 16.33%, and integrating MIPROv2 further raises it to 19.03%. Notably, applying GEPA to Sequential10 (without RAG) dramatically boosts kernel performance, with several generated kernels achieving up to 70% vector utilization and a mean of 30.52%. Furthermore, a single prompt generated by GEPA enables Sequential10 (again without RAG) to attain a score of 26.85%.

CUDA Kernels: For 35 tasks from the KernelBench “representative subset”(Ouyang et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib27)), spanning three difficulty levels, we ran GEPA with GPT-4o. As depicted in Figure[8](https://arxiv.org/html/2507.19457v1#S6.F8 "Figure 8 ‣ 6 GEPA For Inference-Time Search ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"), GEPA boosts GPT-4o’s close-to-0% f​a​s​t 1 fast_{1} score to above 20% with increasing search budget. This task used an agent that could generate upto 5 sequential refinements based on environment feedback (Sequential5).

These experiments with GPT-4o also demonstrate GEPA’s ability to leverage the abilities of frontier LLMs. However, these are early results and warrant further systematic study. We believe that leveraging GEPA for inference-time search, particularly when coupled with domain specific textual feedback, could generalize to other code generation and domain adaptation tasks—a direction we leave for future work.

![Image 7: Refer to caption](https://arxiv.org/html/2507.19457v1/x7.png)

![Image 8: Refer to caption](https://arxiv.org/html/2507.19457v1/x8.png)

Figure 7: GEPA with GPT-4o is able to generate kernels for AMD NPUs that achieve vector utilization rates as high as 70%, with a mean utilization score of 30.52%. In comparison, GPT-4o, even after up to 10 sequential refinements with environment feedback, achieves an aggregate score of only 4.25%. When enhanced with retrieval-augmented generation (RAG) and MIPRO, the sequential refinement agent improves to scores of 16.33% and 19.03%, respectively. Notably, the final prompt produced by GEPA enables the same agent to reach a utilization score of 26.85%, all without requiring any runtime RAG.

![Image 9: Refer to caption](https://arxiv.org/html/2507.19457v1/x9.png)

Figure 8: GEPA with GPT-4o is able to iteratively refine and improve CUDA Kernel Code. The graphs shows f​a​s​t p fast_{p} vs. rollouts plot for p=[0.5,1][0.5,1], where the speedup is calculated over Pytorch-eager. f​a​s​t p fast_{p} is a metric described in(Ouyang et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib27)) that measures the fraction of tasks for which the method generated a kernel executing faster than p p times the baseline. As can be seen, GEPA with GPT-4o is able to generate cuda kernels executing faster than Pytorch-eager for over 20% of the 35 representative tasks.

7 Related Work
--------------

#### Prompt Optimization

Prompt optimization has been shown to be effective for large language models. Manual prompt tuning like chain-of-thought prompting Wei et al. ([2023](https://arxiv.org/html/2507.19457v1#bib.bib48)) promotes the model performance by a large margin, but requires human knowledge to decide what instructions and demonstrations are useful. To scale tuning prompts to more domains, recent work (Zhou et al., [2022](https://arxiv.org/html/2507.19457v1#bib.bib65); Yang et al., [2024](https://arxiv.org/html/2507.19457v1#bib.bib55); Agarwal et al., [2024](https://arxiv.org/html/2507.19457v1#bib.bib2); Fernando et al., [2024](https://arxiv.org/html/2507.19457v1#bib.bib7)) propose using large language models to optimize the prompt. GEPA also leverages LLMs to optimize the prompt. However, GEPA is different from all previous work for using textual feedback from the environment, searching the best prompt candidates through Pareto-aware optimization, and using evolution strategies to optimize prompts for each submodule in a compound AI optimization.

#### Evolutionary LLMs

Beyond naive prompt optimization, evolutionary algorithms are also popular for optimizing LLM performance. For example, EvoPrompt(Guo et al., [2024](https://arxiv.org/html/2507.19457v1#bib.bib8)) connects LLMs with evolutionary algorithms that gradually populate optimized prompts. While the evolutionary algorithm is powerful, EvoPrompt’s prompt population fails to leverage training feedback. Further, unlike the use of random mutation, GEPA leverages domain specific feedback to perform mutations, enabling it to converge faster, evidences by GEPA’s sample efficiency.

Another recent work, AlphaEvolve(Novikov et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib24)) (and its open-source implementation OpenEvolve(Sharma, [2025](https://arxiv.org/html/2507.19457v1#bib.bib34))), demonstrates the effectiveness of evolutionary search when applied directly to code: the system iteratively rewrites portions of the code with evolutionary feedback inside the prompt. This approach excels in domains where solutions can be explicitly encoded and directly manipulated at the code level. While AlphaEvolve applies solution search toward a single hard problem, GEPA brings the evolutionary paradigm to a domain of problems through prompts. By combining Pareto-front optimization and prompt evolution, GEPA leverages knowledge and tactics from similar problems to construct a powerful prompt.

#### Reflection and Feedback

Utilizing feedback and reward is another prominent way to improve LLM performances on verifiable tasks. A common strategy is to use the reward to guide the model’s learning process through reinforcement learning (RL) techniques. For example, Xu et al. ([2025a](https://arxiv.org/html/2507.19457v1#bib.bib52)); Zuo et al. ([2025](https://arxiv.org/html/2507.19457v1#bib.bib67)) proposed using natural language feedback and major voting knowledge to perform reinforcement learning training, respectively. However, weight-altering reinforcement learning bears sample inefficiency, especially when reward or feedback calculation is slow.

An alternative way to “learn” from environment feedback is through learning in the natural language space. Monea et al. ([2025](https://arxiv.org/html/2507.19457v1#bib.bib22)) uses the language model itself to perform in-context learning through self-bootstrapping with external verifiers. Wang et al. ([2024](https://arxiv.org/html/2507.19457v1#bib.bib47)) proposes summarizing “memories” from successful agent trajectories to resemble leaning. Similarly, Dynamic Cheatsheet(Suzgun et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib40)) proposes test-time learning through synthesizing strategies for later reference. In contrast, GEPA leverages the examples for reflectively proposing new instructions, rather than using them as few-shot demonstrations, thereby creating task-specific knowledge and rules.

#### Optimizing Compound AI Systems and Agents

Optimizing compound AI systems has always been a challenge with multiple LLM modules involved. DSPy(Khattab et al., [2022](https://arxiv.org/html/2507.19457v1#bib.bib15); [2024](https://arxiv.org/html/2507.19457v1#bib.bib16)) introduces optimization for searching and bootstrapping few-shot in-context examples. TextGrad(Yuksekgonul et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib60)) optimizes compound systems by backpropagating textual feedback from other LLMs. MIPROv2(Opsahl-Ong et al., [2024](https://arxiv.org/html/2507.19457v1#bib.bib26)) attempts to align instructions and few-shot examples jointly for compound AI systems through Bayesian Optimization. Despite using different algorithms, the aforementioned optimizers rely on final global rewards as signals for success. Recent work Optimas (Wu et al., [2025b](https://arxiv.org/html/2507.19457v1#bib.bib50)) proposes using globally aligned local rewards for each LLM module to further improve compound AI systems.

GEPA leverages global rewards as signals while incorporating additional textual feedback for each module from the environment. Unlike previous optimizers that treat the training set as a monolithic entity and optimize for overall performance, GEPA takes a fundamentally different approach. At each step, GEPA maintains and updates a Pareto front across all data instances, enabling it to match prompts to specific data points within the training set. This more fine-grained approach set the basis for GEPA’s effective prompt evolution, enabling it to explore many different prompt strategies before converging to a generalizable prompt.

8 Limitations and Future Work
-----------------------------

While GEPA demonstrates strong sample efficiency and generalization via reflective prompt evolution, several limitations remain. The boundary between prompt-based and weight-based learning is not well understood—although GEPA excels when rollouts are expensive, it is likely that weight updates will outperform prompting in regimes with abundant data or when large-scale rollouts are feasible. We adopted LoRA for our GRPO baseline experiments because it offers lower computational costs and has been successfully applied to reinforcement learning in previous work across a range of tasks(Wang et al., [2025b](https://arxiv.org/html/2507.19457v1#bib.bib46); Xu et al., [2025b](https://arxiv.org/html/2507.19457v1#bib.bib53); Li et al., [2025b](https://arxiv.org/html/2507.19457v1#bib.bib20); Yue et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib59); Sun et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib39); Hayou et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib9); Zhao et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib63); Teknium et al., [2024](https://arxiv.org/html/2507.19457v1#bib.bib43); Zhao et al., [2024](https://arxiv.org/html/2507.19457v1#bib.bib62); Sidahmed et al., [2024](https://arxiv.org/html/2507.19457v1#bib.bib36)). However, future work should investigate the impact of full-parameter finetuning for GRPO. At the same time, we note that prior studies using full-parameter GRPO have consistently required a high number of rollouts—typically ranging from 100,000 to 512,000(Chen et al., [2025b](https://arxiv.org/html/2507.19457v1#bib.bib6); Wu et al., [2025c](https://arxiv.org/html/2507.19457v1#bib.bib51); Zhang et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib61); Jin et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib13); Si et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib35); Wang et al., [2025a](https://arxiv.org/html/2507.19457v1#bib.bib45); Java et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib11); Chen et al., [2025a](https://arxiv.org/html/2507.19457v1#bib.bib5); Wu et al., [2025a](https://arxiv.org/html/2507.19457v1#bib.bib49); Sha et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib32); Lin et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib21); Peng et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib28); Song et al., [2025](https://arxiv.org/html/2507.19457v1#bib.bib37)). While we explored several hyperparameter values for [LR, beta, norm clipping], future work should study the impact of careful hyperparameter tuning for RL algorithms.

GEPA currently focuses on optimizing instructions alone, omitting exemplar or few-shot demonstration optimization. Incorporating such examples could further improve performance, particularly in tasks where in-context demonstrations are known to help. Furthermore, we made the observation that in the current form, the majority of GEPA’s rollouts are expended for candidate validation, which can be performed well even on smaller Pareto datasets. Future works should explore the impact of the hyperparameter Pareto-validation set size (Algorithm[1](https://arxiv.org/html/2507.19457v1#alg1 "Algorithm 1 ‣ Figure 4 ‣ 3 GEPA: Reflective Prompt Evolution ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning")) and developing dynamic or subsampled validation strategies to further enhance sample efficiency. A promising and underexplored direction is what we dub _feedback engineering_, i.e., identifying which of the system’s execution or evaluation traces could provide the most valuable learning signal for reflective optimization. Finally, GEPA currently operates on fixed model parameters; we hypothesize that integrating reflective prompt evolution with weight-space adaptation—for example, using GEPA’s language-based lessons to perform RL rollouts—could yield additive gains and help unify prompt- and weight-based approaches for optimizing compound AI systems.

9 Conclusion
------------

We introduced GEPA, a novel prompt optimizer for arbitrary LLM agents and workflows. GEPA leverages reflective prompt evolution and Pareto-based selection, showing superior sample efficiency compared to reinforcement learning (GRPO) alongside robust generalization, while outperforming leading prompt optimizers (MIPROv2). By explicitly incorporating natural language feedback and maintaining a diverse pool of Pareto-optimal candidates, GEPA rapidly adapts AI systems to new tasks. Our results across diverse benchmarks and multiple models suggest that language-based reflection can offer a scalable strategy for optimizing complex real-world AI workflows, especially in resource-constrained settings. GEPA also shows promise as an inference-time search strategy, showing ability to write code in very challenging domains. We view GEPA as a step towards more human-like, adaptive, and efficient AI system learning, and anticipate that its core principles will inspire further research into language-driven, reflection-based learning in AI.

References
----------

*   Advanced Micro Devices (2025) Advanced Micro Devices. Amd xdna™ architecture. [https://www.amd.com/en/technologies/xdna.html#xdna2](https://www.amd.com/en/technologies/xdna.html#xdna2), 2025. Accessed on: 2025-07-23. 
*   Agarwal et al. (2024) Eshaan Agarwal, Joykirat Singh, Vivek Dani, Raghav Magazine, Tanuja Ganu, and Akshay Nambi. Promptwizard: Task-aware prompt optimization framework, 2024. URL [https://arxiv.org/abs/2405.18369](https://arxiv.org/abs/2405.18369). 
*   Agrawal et al. (2023) Amey Agrawal, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S Gulavani, and Ramachandran Ramjee. Sarathi: Efficient llm inference by piggybacking decodes with chunked prefills. _arXiv preprint arXiv:2308.16369_, 2023. 
*   AI (2025) Meta AI. llama-prompt-ops: An open-source tool for seamless migration from other llms to llama, and for general prompt optimization. [https://github.com/meta-llama/llama-prompt-ops](https://github.com/meta-llama/llama-prompt-ops), 2025. Accessed: 2025-07-11. 
*   Chen et al. (2025a) Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Haofen Wang, Jeff Z. Pan, Wen Zhang, Huajun Chen, Fan Yang, Zenan Zhou, and Weipeng Chen. ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning, March 2025a. URL [http://arxiv.org/abs/2503.19470](http://arxiv.org/abs/2503.19470). arXiv:2503.19470 [cs]. 
*   Chen et al. (2025b) Peter Chen, Xiaopeng Li, Ziniu Li, Xi Chen, and Tianyi Lin. Spectral Policy Optimization: Coloring your Incorrect Reasoning in GRPO, May 2025b. URL [http://arxiv.org/abs/2505.11595](http://arxiv.org/abs/2505.11595). arXiv:2505.11595 [cs]. 
*   Fernando et al. (2024) Chrisantha Fernando, Dylan Banarse, Henryk Michalewski, Simon Osindero, and Tim Rockt””aschel. Promptbreeder: self-referential self-improvement via prompt evolution. In _Proceedings of the 41st International Conference on Machine Learning_, ICML’24. JMLR.org, 2024. 
*   Guo et al. (2024) Qingyan Guo, Rui Wang, Junliang Guo, Bei Li, Kaitao Song, Xu Tan, Guoqing Liu, Jiang Bian, and Yujiu Yang. Connecting large language models with evolutionary algorithms yields powerful prompt optimizers. In _The Twelfth International Conference on Learning Representations_, 2024. URL [https://openreview.net/forum?id=ZG3RaNIsO8](https://openreview.net/forum?id=ZG3RaNIsO8). 
*   Hayou et al. (2025) Soufiane Hayou, Nikhil Ghosh, and Bin Yu. PLoP: Precise LoRA Placement for Efficient Finetuning of Large Models, June 2025. URL [http://arxiv.org/abs/2506.20629](http://arxiv.org/abs/2506.20629). arXiv:2506.20629 [cs]. 
*   Hu et al. (2022) Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. _ICLR_, 1(2):3, 2022. 
*   Java et al. (2025) Abhinav Java, Srivathsan Koundinyan, Nagarajan Natarajan, and Amit Sharma. FrugalRAG: Learning to retrieve and reason for multi-hop QA, July 2025. URL [http://arxiv.org/abs/2507.07634](http://arxiv.org/abs/2507.07634). arXiv:2507.07634 [cs]. 
*   Jiang et al. (2020) Yichen Jiang, Shikha Bordia, Zheng Zhong, Charles Dognin, Maneesh Singh, and Mohit Bansal. HoVer: A dataset for many-hop fact extraction and claim verification. In Trevor Cohn, Yulan He, and Yang Liu (eds.), _Findings of the Association for Computational Linguistics: EMNLP 2020_, pp. 3441–3460, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.findings-emnlp.309. URL [https://aclanthology.org/2020.findings-emnlp.309/](https://aclanthology.org/2020.findings-emnlp.309/). 
*   Jin et al. (2025) Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning, July 2025. URL [http://arxiv.org/abs/2503.09516](http://arxiv.org/abs/2503.09516). arXiv:2503.09516 [cs]. 
*   Kalade & Schelle (2025) Sarunas Kalade and Graham Schelle. Npueval: Optimizing npu kernels with llms and open source compilers, 2025. URL [https://arxiv.org/abs/2507.14403](https://arxiv.org/abs/2507.14403). 
*   Khattab et al. (2022) Omar Khattab, Keshav Santhanam, Xiang Lisa Li, David Hall, Percy Liang, Christopher Potts, and Matei Zaharia. Demonstrate-search-predict: Composing retrieval and language models for knowledge-intensive NLP. _arXiv preprint arXiv:2212.14024_, 2022. 
*   Khattab et al. (2024) Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan A, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, and Christopher Potts. DSPy: Compiling declarative language model calls into state-of-the-art pipelines. In _The Twelfth International Conference on Learning Representations_, 2024. URL [https://openreview.net/forum?id=sY5N0zY5Od](https://openreview.net/forum?id=sY5N0zY5Od). 
*   Kwon et al. (2023) Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In _Proceedings of the 29th symposium on operating systems principles_, pp. 611–626, 2023. 
*   Lambert (2025) Nathan Lambert. Policy gradient algorithms. In _RLHF Book: Reinforcement Learning from Human Feedback_, chapter 11. RLHF Book, 2025. URL [https://rlhfbook.com/c/11-policy-gradients.html](https://rlhfbook.com/c/11-policy-gradients.html). Accessed July 16, 2025. 
*   Li et al. (2025a) Siyan Li, Vethavikashini Chithrra Raghuram, Omar Khattab, Julia Hirschberg, and Zhou Yu. PAPILLON: Privacy preservation from Internet-based and local language model ensembles. In Luis Chiruzzo, Alan Ritter, and Lu Wang (eds.), _Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)_, pp. 3371–3390, Albuquerque, New Mexico, April 2025a. Association for Computational Linguistics. ISBN 979-8-89176-189-6. doi: 10.18653/v1/2025.naacl-long.173. URL [https://aclanthology.org/2025.naacl-long.173/](https://aclanthology.org/2025.naacl-long.173/). 
*   Li et al. (2025b) Xianming Li, Aamir Shakir, Rui Huang, Julius Lipp, and Jing Li. ProRank: Prompt Warmup via Reinforcement Learning for Small Language Models Reranking, June 2025b. URL [http://arxiv.org/abs/2506.03487](http://arxiv.org/abs/2506.03487). arXiv:2506.03487 [cs]. 
*   Lin et al. (2025) Chenyu Lin, Yilin Wen, Du Su, Fei Sun, Muhan Chen, Chenfu Bao, and Zhonghou Lv. Knowledgeable-r1: Policy Optimization for Knowledge Exploration in Retrieval-Augmented Generation, June 2025. URL [http://arxiv.org/abs/2506.05154](http://arxiv.org/abs/2506.05154). arXiv:2506.05154 [cs]. 
*   Monea et al. (2025) Giovanni Monea, Antoine Bosselut, Kianté Brantley, and Yoav Artzi. Llms are in-context bandit reinforcement learners, 2025. URL [https://arxiv.org/abs/2410.05362](https://arxiv.org/abs/2410.05362). 
*   Mouret & Clune (2015) Jean-Baptiste Mouret and Jeff Clune. Illuminating search spaces by mapping elites, 2015. URL [https://arxiv.org/abs/1504.04909](https://arxiv.org/abs/1504.04909). 
*   Novikov et al. (2025) Alexander Novikov, Ngân Vũ, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J.R. Ruiz, Abbas Mehrabian, M.Pawan Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli, and Matej Balog. Alphaevolve: A coding agent for scientific and algorithmic discovery. Technical report, Google DeepMind, 2025. URL [https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf). White paper. 
*   OpenAI (2025) OpenAI. GPT-4.1 series, 2025. Large language model series, released April 2025. [https://openai.com/index/gpt-4-1/](https://openai.com/index/gpt-4-1/). 
*   Opsahl-Ong et al. (2024) Krista Opsahl-Ong, Michael J Ryan, Josh Purtell, David Broman, Christopher Potts, Matei Zaharia, and Omar Khattab. Optimizing instructions and demonstrations for multi-stage language model programs. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), _Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing_, pp. 9340–9366, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.525. URL [https://aclanthology.org/2024.emnlp-main.525/](https://aclanthology.org/2024.emnlp-main.525/). 
*   Ouyang et al. (2025) Anne Ouyang, Simon Guo, Simran Arora, Alex L Zhang, William Hu, Christopher Re, and Azalia Mirhoseini. Kernelbench: Can LLMs write efficient GPU kernels? In _Scaling Self-Improving Foundation Models without Human Supervision_, 2025. URL [https://openreview.net/forum?id=k6V4jb8jkX](https://openreview.net/forum?id=k6V4jb8jkX). 
*   Peng et al. (2025) Hao Peng, Yunjia Qi, Xiaozhi Wang, Bin Xu, Lei Hou, and Juanzi Li. VerIF: Verification Engineering for Reinforcement Learning in Instruction Following, June 2025. URL [http://arxiv.org/abs/2506.09942](http://arxiv.org/abs/2506.09942). arXiv:2506.09942 [cs]. 
*   Pyatkin et al. (2025a) Valentina Pyatkin, Saumya Malik, Victoria Graf, Hamish Ivison, Shengyi Huang, Pradeep Dasigi, Nathan Lambert, and Hannaneh Hajishirzi. IF-RLVR-Train, July 2025a. URL [https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5). 
*   Pyatkin et al. (2025b) Valentina Pyatkin, Saumya Malik, Victoria Graf, Hamish Ivison, Shengyi Huang, Pradeep Dasigi, Nathan Lambert, and Hannaneh Hajishirzi. Generalizing verifiable instruction following, 2025b. URL [https://arxiv.org/abs/2507.02833](https://arxiv.org/abs/2507.02833). 
*   Saad-Falcon et al. (2025) Jon Saad-Falcon, Adrian Gamarra Lafuente, Shlok Natarajan, Nahum Maru, Hristo Todorov, Etash Guha, E.Kelly Buchanan, Mayee Chen, Neel Guha, Christopher Ré, and Azalia Mirhoseini. Archon: An architecture search framework for inference-time techniques, 2025. URL [https://arxiv.org/abs/2409.15254](https://arxiv.org/abs/2409.15254). 
*   Sha et al. (2025) Zeyang Sha, Shiwen Cui, and Weiqiang Wang. SEM: Reinforcement Learning for Search-Efficient Large Language Models, May 2025. URL [http://arxiv.org/abs/2505.07903](http://arxiv.org/abs/2505.07903). arXiv:2505.07903 [cs]. 
*   Shao et al. (2024) Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y.Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300). 
*   Sharma (2025) Asankhaya Sharma. Openevolve: Open-source implementation of alphaevolve. [https://github.com/codelion/openevolve](https://github.com/codelion/openevolve), 2025. GitHub. 
*   Si et al. (2025) Shuzheng Si, Haozhe Zhao, Cheng Gao, Yuzhuo Bai, Zhitong Wang, Bofei Gao, Kangyang Luo, Wenhao Li, Yufei Huang, Gang Chen, Fanchao Qi, Minjia Zhang, Baobao Chang, and Maosong Sun. Teaching Large Language Models to Maintain Contextual Faithfulness via Synthetic Tasks and Reinforcement Learning, May 2025. URL [http://arxiv.org/abs/2505.16483](http://arxiv.org/abs/2505.16483). arXiv:2505.16483 [cs]. 
*   Sidahmed et al. (2024) Hakim Sidahmed, Samrat Phatale, Alex Hutcheson, Zhuonan Lin, Zhang Chen, Zac Yu, Jarvis Jin, Simral Chaudhary, Roman Komarytsia, Christiane Ahlheim, Yonghao Zhu, Bowen Li, Saravanan Ganesh, Bill Byrne, Jessica Hoffmann, Hassan Mansoor, Wei Li, Abhinav Rastogi, and Lucas Dixon. Parameter Efficient Reinforcement Learning from Human Feedback, September 2024. URL [http://arxiv.org/abs/2403.10704](http://arxiv.org/abs/2403.10704). arXiv:2403.10704 [cs]. 
*   Song et al. (2025) Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning, March 2025. URL [http://arxiv.org/abs/2503.05592](http://arxiv.org/abs/2503.05592). arXiv:2503.05592 [cs]. 
*   Soylu et al. (2024) Dilara Soylu, Christopher Potts, and Omar Khattab. Fine-tuning and prompt optimization: Two great steps that work better together, 2024. URL [https://arxiv.org/abs/2407.10930](https://arxiv.org/abs/2407.10930). 
*   Sun et al. (2025) Zhongxiang Sun, Qipeng Wang, Haoyu Wang, Xiao Zhang, and Jun Xu. Detection and Mitigation of Hallucination in Large Reasoning Models: A Mechanistic Perspective, May 2025. URL [http://arxiv.org/abs/2505.12886](http://arxiv.org/abs/2505.12886). arXiv:2505.12886 [cs]. 
*   Suzgun et al. (2025) Mirac Suzgun, Mert Yuksekgonul, Federico Bianchi, Dan Jurafsky, and James Zou. Dynamic cheatsheet: Test-time learning with adaptive memory, 2025. URL [https://arxiv.org/abs/2504.07952](https://arxiv.org/abs/2504.07952). 
*   Tan et al. (2025) Shangyin Tan, Lakshya A Agrawal, Arnav Singhvi, Liheng Lai, Michael J Ryan, Dan Klein, Omar Khattab, Koushik Sen, and Matei Zaharia. Langprobe: a language programs benchmark, 2025. URL [https://arxiv.org/abs/2502.20315](https://arxiv.org/abs/2502.20315). 
*   Team (2025) Qwen Team. Qwen/qwen3-8b. [https://huggingface.co/Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), 2025. Accessed: 2025-07-11. 
*   Teknium et al. (2024) Ryan Teknium, Jeffrey Quesnelle, and Chen Guang. Hermes 3 Technical Report, August 2024. URL [http://arxiv.org/abs/2408.11857](http://arxiv.org/abs/2408.11857). arXiv:2408.11857 [cs]. 
*   Wan et al. (2024) Xingchen Wan, Ruoxi Sun, Hootan Nakhost, and Sercan Arik. Teach better or show smarter? on instructions and exemplars in automatic prompt optimization. _Advances in Neural Information Processing Systems_, 37:58174–58244, 2024. URL [https://proceedings.neurips.cc/paper_files/paper/2024/hash/6b031defd145b02bed031093d8797bb3-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6b031defd145b02bed031093d8797bb3-Abstract-Conference.html). 
*   Wang et al. (2025a) Hongru Wang, Cheng Qian, Wanjun Zhong, Xiusi Chen, Jiahao Qiu, Shijue Huang, Bowen Jin, Mengdi Wang, Kam-Fai Wong, and Heng Ji. Acting Less is Reasoning More! Teaching Model to Act Efficiently, May 2025a. URL [http://arxiv.org/abs/2504.14870](http://arxiv.org/abs/2504.14870). arXiv:2504.14870 [cs]. 
*   Wang et al. (2025b) Shangshang Wang, Julian Asilis, Ömer Faruk Akgül, Enes Burak Bilgin, Ollie Liu, and Willie Neiswanger. Tina: Tiny Reasoning Models via LoRA, April 2025b. URL [http://arxiv.org/abs/2504.15777](http://arxiv.org/abs/2504.15777). arXiv:2504.15777 [cs]. 
*   Wang et al. (2024) Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory, 2024. URL [https://arxiv.org/abs/2409.07429](https://arxiv.org/abs/2409.07429). 
*   Wei et al. (2023) Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023. URL [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903). 
*   Wu et al. (2025a) Peilin Wu, Mian Zhang, Xinlu Zhang, Xinya Du, and Zhiyu Zoey Chen. Search Wisely: Mitigating Sub-optimal Agentic Searches By Reducing Uncertainty, May 2025a. URL [http://arxiv.org/abs/2505.17281](http://arxiv.org/abs/2505.17281). arXiv:2505.17281 [cs]. 
*   Wu et al. (2025b) Shirley Wu, Parth Sarthi, Shiyu Zhao, Aaron Lee, Herumb Shandilya, Adrian Mladenic Grobelnik, Nurendra Choudhary, Eddie Huang, Karthik Subbian, Linjun Zhang, Diyi Yang, James Zou, and Jure Leskovec. Optimas: Optimizing compound ai systems with globally aligned local rewards, 2025b. URL [https://arxiv.org/abs/2507.03041](https://arxiv.org/abs/2507.03041). 
*   Wu et al. (2025c) Yihong Wu, Liheng Ma, Muzhi Li, Jiaming Zhou, Jianye Hao, Ho-fung Leung, Irwin King, Yingxue Zhang, and Jian-Yun Nie. Reinforcing Question Answering Agents with Minimalist Policy Gradient Optimization, July 2025c. URL [http://arxiv.org/abs/2505.17086](http://arxiv.org/abs/2505.17086). arXiv:2505.17086 [cs]. 
*   Xu et al. (2025a) Wanqiao Xu, Allen Nie, Ruijie Zheng, Aditya Modi, Adith Swaminathan, and Ching-An Cheng. Provably learning from language feedback, 2025a. URL [https://arxiv.org/abs/2506.10341](https://arxiv.org/abs/2506.10341). 
*   Xu et al. (2025b) Yixuan Even Xu, Yash Savani, Fei Fang, and Zico Kolter. Not All Rollouts are Useful: Down-Sampling Rollouts in LLM Reinforcement Learning, June 2025b. URL [http://arxiv.org/abs/2504.13818](http://arxiv.org/abs/2504.13818). arXiv:2504.13818 [cs]. 
*   Yang et al. (2025) An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL [https://arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388). 
*   Yang et al. (2024) Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, and Xinyun Chen. Large language models as optimizers, 2024. URL [https://arxiv.org/abs/2309.03409](https://arxiv.org/abs/2309.03409). 
*   Yang et al. (2018) Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In _Conference on Empirical Methods in Natural Language Processing (EMNLP)_, 2018. 
*   Yao et al. (2023) Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models, 2023. URL [https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629). 
*   Yu et al. (2025) Lingfan Yu, Jinkun Lin, and Jinyang Li. Stateful large language model serving with pensieve. In _Proceedings of the Twentieth European Conference on Computer Systems_, EuroSys ’25, pp. 144–158, New York, NY, USA, 2025. Association for Computing Machinery. ISBN 9798400711961. doi: 10.1145/3689031.3696086. URL [https://doi.org/10.1145/3689031.3696086](https://doi.org/10.1145/3689031.3696086). 
*   Yue et al. (2025) Zhenrui Yue, Bowen Jin, Huimin Zeng, Honglei Zhuang, Zhen Qin, Jinsung Yoon, Lanyu Shang, Jiawei Han, and Dong Wang. Hybrid Latent Reasoning via Reinforcement Learning, May 2025. URL [http://arxiv.org/abs/2505.18454](http://arxiv.org/abs/2505.18454). arXiv:2505.18454 [cs]. 
*   Yuksekgonul et al. (2025) Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Pan Lu, Zhi Huang, Carlos Guestrin, and James Zou. Optimizing generative ai by backpropagating language model feedback. _Nature_, 639:609–616, 2025. 
*   Zhang et al. (2025) Qi Zhang, Shouqing Yang, Lirong Gao, Hao Chen, Xiaomeng Hu, Jinglei Chen, Jiexiang Wang, Sheng Guo, Bo Zheng, Haobo Wang, and Junbo Zhao. LeTS: Learning to Think-and-Search via Process-and-Outcome Reward Hybridization, May 2025. URL [http://arxiv.org/abs/2505.17447](http://arxiv.org/abs/2505.17447). arXiv:2505.17447 [cs]. 
*   Zhao et al. (2024) Siyan Zhao, John Dang, and Aditya Grover. Group Preference Optimization: Few-Shot Alignment of Large Language Models, October 2024. URL [http://arxiv.org/abs/2310.11523](http://arxiv.org/abs/2310.11523). arXiv:2310.11523 [cs]. 
*   Zhao et al. (2025) Siyan Zhao, Devaansh Gupta, Qinqing Zheng, and Aditya Grover. d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning, June 2025. URL [http://arxiv.org/abs/2504.12216](http://arxiv.org/abs/2504.12216). arXiv:2504.12216 [cs]. 
*   Zheng et al. (2024) Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Livia Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. Sglang: Efficient execution of structured language model programs. _Advances in neural information processing systems_, 37:62557–62583, 2024. 
*   Zhou et al. (2022) Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. Large language models are human-level prompt engineers. In _The eleventh international conference on learning representations_, 2022. 
*   Ziems* et al. (2025) Noah Ziems*, Dilara Soylu*, Lakshya A Agrawal*, Isaac Miller, Liheng Lai, Chen Qian, Karel D’Oosterlinck, Meng Jiang, Dan Klein, Matei Zaharia, Christopher Potts, and Omar Khattab. Multi-module GRPO: Composing policy gradients and prompt optimization for language model programs. Manuscript in preparation. *Equal contribution., 2025. URL [https://dspy.ai/api/optimizers/GRPO](https://dspy.ai/api/optimizers/GRPO). 
*   Zuo et al. (2025) Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, Yuchen Zhang, Xinwei Long, Ermo Hua, Biqing Qi, Youbang Sun, Zhiyuan Ma, Lifan Yuan, Ning Ding, and Bowen Zhou. Ttrl: Test-time reinforcement learning, 2025. URL [https://arxiv.org/abs/2504.16084](https://arxiv.org/abs/2504.16084). 

Appendix A Results (Contd.)
---------------------------

![Image 10: Refer to caption](https://arxiv.org/html/2507.19457v1/x10.png)

(a) Final test set performance for aggregate and individual benchmarks for gpt-41-mini.

![Image 11: Refer to caption](https://arxiv.org/html/2507.19457v1/x11.png)

(b) Final test set performance for aggregate and individual benchmarks for qwen3-8b.

Figure 9: Final test set performance for aggregate and individual benchmarks.

Appendix B GEPA’s Reflection and Prompt Update Meta Prompt
----------------------------------------------------------

Figure[B](https://arxiv.org/html/2507.19457v1#A2 "Appendix B GEPA’s Reflection and Prompt Update Meta Prompt ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning") provides the meta prompt used by GEPA for LLM’s to reflectively update the current set of instructions, given input-output examples with the current instruction, and the corresponding feedback from the environment.

Appendix C Performance vs. Budget (Rollouts) Curves (Contd.)
------------------------------------------------------------

![Image 12: Refer to caption](https://arxiv.org/html/2507.19457v1/x12.png)

(a) GPT-4.1 Mini - GEPA vs MIPRO

![Image 13: Refer to caption](https://arxiv.org/html/2507.19457v1/x13.png)

(b) Qwen3 8B - GEPA vs MIPROv2

![Image 14: Refer to caption](https://arxiv.org/html/2507.19457v1/x14.png)

(c) Qwen3 8B - GEPA vs GRPO

Figure 10: Hotpot QA Bench: rollout vs. score for different models/settings.

![Image 15: Refer to caption](https://arxiv.org/html/2507.19457v1/x15.png)

(a) GPT-4.1 Mini - GEPA vs MIPRO

![Image 16: Refer to caption](https://arxiv.org/html/2507.19457v1/x16.png)

(b) Qwen3 8B - GEPA vs MIPROv2

![Image 17: Refer to caption](https://arxiv.org/html/2507.19457v1/x17.png)

(c) Qwen3 8B - GEPA vs GRPO

Figure 11: IFBench: rollout vs. score for different models/settings.

![Image 18: Refer to caption](https://arxiv.org/html/2507.19457v1/x18.png)

(a) GPT-4.1 Mini - GEPA vs MIPRO

![Image 19: Refer to caption](https://arxiv.org/html/2507.19457v1/x19.png)

(b) Qwen3 8B - GEPA vs MIPROv2

![Image 20: Refer to caption](https://arxiv.org/html/2507.19457v1/x20.png)

(c) Qwen3 8B - GEPA vs GRPO

Figure 12: HoverBench: rollout vs. score for different models/settings.

![Image 21: Refer to caption](https://arxiv.org/html/2507.19457v1/x21.png)

(a) GPT-4.1 Mini - GEPA vs MIPRO

![Image 22: Refer to caption](https://arxiv.org/html/2507.19457v1/x22.png)

(b) Qwen3 8B - GEPA vs MIPROv2

![Image 23: Refer to caption](https://arxiv.org/html/2507.19457v1/x23.png)

(c) Qwen3 8B - GEPA vs GRPO

Figure 13: PUPA: rollout vs. score for different models/settings.

Appendix D Generalization Gap
-----------------------------

![Image 24: Refer to caption](https://arxiv.org/html/2507.19457v1/x24.png)

![Image 25: Refer to caption](https://arxiv.org/html/2507.19457v1/x25.png)

![Image 26: Refer to caption](https://arxiv.org/html/2507.19457v1/x26.png)

![Image 27: Refer to caption](https://arxiv.org/html/2507.19457v1/x27.png)

![Image 28: Refer to caption](https://arxiv.org/html/2507.19457v1/x28.png)

![Image 29: Refer to caption](https://arxiv.org/html/2507.19457v1/x29.png)

Figure 14: Generalization gaps for different optimization methods. Following Wan et al. ([2024](https://arxiv.org/html/2507.19457v1#bib.bib44)), we visualize the generalization gap (i.e., the difference between final test set performance and the best achieved validation performance) for different optimizers. While Wan et al. ([2024](https://arxiv.org/html/2507.19457v1#bib.bib44)) previously observed that exemplars tend to generalize better, our results suggest that instructions generated by reflective prompt evolution can achieve stronger generalization as well as improved overall performance. We hypothesize this difference may be due to the improving capabilities of the underlying LLMs, as more recent models are both better at adhering to instructions and capable of reflecting on their outputs.

Figure[14](https://arxiv.org/html/2507.19457v1#A4.F14 "Figure 14 ‣ Appendix D Generalization Gap ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning") visualizes the generalization gap for different optimization methods.

Appendix E Cost vs. Performance Analysis for optimized systems
--------------------------------------------------------------

![Image 30: Refer to caption](https://arxiv.org/html/2507.19457v1/x30.png)

(a) GPT-4.1 Mini.

![Image 31: Refer to caption](https://arxiv.org/html/2507.19457v1/x31.png)

(b) Qwen3 8B.

Figure 15: These plots visualize the final aggregate scores against the aggregate prompt size (across all benchmarks) of the final optimized system for each optimizer. It can be seen that GEPA consistently produces prompts that are around less than 33% of the size of MIPROv2’s prompts, while getting higher performance. Most of GEPA’s prompt tokens are used for providing instructions, whereas most of MIPROv2’s prompt tokens pertain to few-shot examples.

![Image 32: Refer to caption](https://arxiv.org/html/2507.19457v1/x32.png)

(a) Comparing the token counts of the optimized programs across benchmarks for GPT-4.1 Mini.

![Image 33: Refer to caption](https://arxiv.org/html/2507.19457v1/x33.png)

(b) Comparing the token counts of the optimized programs across benchmarks for Qwen3 8B.

Figure 16: Final test set performance for aggregate and individual benchmarks.

The prompt size of the optimized system plays an important role in determining the downstream cost of using the optimized system. Figure[15](https://arxiv.org/html/2507.19457v1#A5.F15 "Figure 15 ‣ Appendix E Cost vs. Performance Analysis for optimized systems ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning") visualizes the aggregate prompt lengths of the final optimized system (as cost proxy) for each optimizer, against the performance achieved. Notably, GEPA’s prompts are around 33% shorter than MIPROv2’s prompts, while achieving higher performance.

Appendix F Merge: System-aware crossover strategy for Compound AI optimization
------------------------------------------------------------------------------

Algorithm 3 Check if module combination is desirable

1:function Desirable(

a,i,j,𝒫 a,\,i,\,j,\,\mathcal{P}
)

2:for module

m=1 m=1
to

|M||M|
do

3:

π a←\pi_{a}\leftarrow
ancestor’s prompt for module

m m

4:

π i←\pi_{i}\leftarrow
descendent i’s prompt for module

m m

5:

π j←\pi_{j}\leftarrow
descendent j’s prompt for module

m m

6:if (

π a=π i\pi_{a}=\pi_{i}
and

π j≠π i\pi_{j}\neq\pi_{i}
) or(

π a=π j\pi_{a}=\pi_{j}
and

π i≠π j\pi_{i}\neq\pi_{j}
) then

7:return True

8:end if

9:end for

10:return False

11:end function

Algorithm 4 Merge: Genetic Crossover for Modular Candidates

1:function Merge(

𝒫,𝒜,S,r\mathcal{P},\mathcal{A},S,r
)

2:

i,j←r.sample​(2,|𝒫|)i,\,j\leftarrow r.\text{sample}(2,\,|\mathcal{P}|)
// distinct

i≠j i\neq j

3:

A i←A_{i}\leftarrow
GetAncestors(

i,𝒜 i,\,\mathcal{A}
),

A j←A_{j}\leftarrow
GetAncestors(

j,𝒜 j,\,\mathcal{A}
)

4:if

i∈A j i\in A_{j}
or

j∈A i j\in A_{i}
then

5:continue // skip direct ancestry

6:end if

7:for

a∈A i∩A j a\in A_{i}\cap A_{j}
do

8:if this merge

(i,j,a)(i,j,a)
has been tried before then

9:continue

10:end if

11:if

S​[a]>min⁡(S​[i],S​[j])S[a]>\min(S[i],S[j])
then

12:continue

13:end if

14:if not Desirable(

a,i,j,𝒫 a,\,i,\,j,\,\mathcal{P}
) then

15:continue

16:end if

17:

Φ′←\Phi^{\prime}\leftarrow
copy of

𝒫​[a]\mathcal{P}[a]

18:for module

m=1 m=1
to

|M||M|
do

19:

π a←𝒫​[a].ℳ m.π\pi_{a}\leftarrow\mathcal{P}[a].\mathcal{M}_{m}.\pi

20:

π i←𝒫​[i].ℳ m.π\pi_{i}\leftarrow\mathcal{P}[i].\mathcal{M}_{m}.\pi

21:

π j←𝒫​[j].ℳ m.π\pi_{j}\leftarrow\mathcal{P}[j].\mathcal{M}_{m}.\pi

22:if

π a=π i\pi_{a}=\pi_{i}
and

π j≠π i\pi_{j}\neq\pi_{i}
then

23:

Φ′.ℳ m.π←π j\Phi^{\prime}.\mathcal{M}_{m}.\pi\leftarrow\pi_{j}

24:else if

π a=π j\pi_{a}=\pi_{j}
and

π i≠π j\pi_{i}\neq\pi_{j}
then

25:

Φ′.ℳ m.π←π i\Phi^{\prime}.\mathcal{M}_{m}.\pi\leftarrow\pi_{i}

26:else if

π i≠π j≠π a\pi_{i}\neq\pi_{j}\neq\pi_{a}
then

27: Choose

d∗=arg⁡max⁡{S​[i],S​[j]}d^{*}=\arg\max\{S[i],S[j]\}
(break ties randomly)

28:

Φ′.ℳ m.π←π d∗\Phi^{\prime}.\mathcal{M}_{m}.\pi\leftarrow\pi_{d^{*}}

29:else

30:

Φ′.ℳ m.π←π i\Phi^{\prime}.\mathcal{M}_{m}.\pi\leftarrow\pi_{i}
// default

31:end if

32:end for

33:return

(Φ′,i,j,a)(\Phi^{\prime},i,j,a)

34:end for

35:return None

36:end function

Algorithm[4](https://arxiv.org/html/2507.19457v1#alg4 "Algorithm 4 ‣ Appendix F Merge: System-aware crossover strategy for Compound AI optimization ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning") provides the instantiation of the System aware Merge strategy used in GEPA+Merge.

Appendix G Visualizing the Iterative Refinement achieved by GEPA
----------------------------------------------------------------

Figure[5](https://arxiv.org/html/2507.19457v1#S3.F5 "Figure 5 ‣ 3.3 Pareto-based candidate selection ‣ 3 GEPA: Reflective Prompt Evolution ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning") presented a summary of the prompt refinements performed by GEPA during the optimization for PUPA. In this section, we present the full prompts produced during the optimization.

### G.1 Prompts at intermediate stages for PUPA

Appendix H GEPA Search Trees
----------------------------

![Image 34: Refer to caption](https://arxiv.org/html/2507.19457v1/x34.png)

(a) Abl:SelectBestCandidate

![Image 35: Refer to caption](https://arxiv.org/html/2507.19457v1/x35.png)

(b) SelectBestCandidate + Merge

![Image 36: Refer to caption](https://arxiv.org/html/2507.19457v1/x36.png)

(c) GEPA - Best Config

![Image 37: Refer to caption](https://arxiv.org/html/2507.19457v1/x37.png)

(d) GEPA+Merge

Figure 17: HotpotQA GPT-4.1 Mini

![Image 38: Refer to caption](https://arxiv.org/html/2507.19457v1/x38.png)

(a) Abl:SelectBestCandidate

![Image 39: Refer to caption](https://arxiv.org/html/2507.19457v1/x39.png)

(b) SelectBestCandidate + Merge

![Image 40: Refer to caption](https://arxiv.org/html/2507.19457v1/x40.png)

(c) GEPA

![Image 41: Refer to caption](https://arxiv.org/html/2507.19457v1/x41.png)

(d) GEPA+Merge - Best Config

Figure 18: HotpotQA Qwen3 8B

![Image 42: Refer to caption](https://arxiv.org/html/2507.19457v1/x42.png)

(a) Abl:SelectBestCandidate

![Image 43: Refer to caption](https://arxiv.org/html/2507.19457v1/x43.png)

(b) SelectBestCandidate + Merge

![Image 44: Refer to caption](https://arxiv.org/html/2507.19457v1/x44.png)

(c) GEPA

![Image 45: Refer to caption](https://arxiv.org/html/2507.19457v1/x45.png)

(d) GEPA+Merge - Best Config

Figure 19: IFBench GPT-4.1 Mini

![Image 46: Refer to caption](https://arxiv.org/html/2507.19457v1/x46.png)

(a) Abl:SelectBestCandidate

![Image 47: Refer to caption](https://arxiv.org/html/2507.19457v1/x47.png)

(b) SelectBestCandidate+Merge

![Image 48: Refer to caption](https://arxiv.org/html/2507.19457v1/x48.png)

(c) GEPA - Best Config

![Image 49: Refer to caption](https://arxiv.org/html/2507.19457v1/x49.png)

(d) GEPA+Merge

Figure 20: IFBench Qwen3 8B

![Image 50: Refer to caption](https://arxiv.org/html/2507.19457v1/x50.png)

(a) Abl:SelectBestCandidate

![Image 51: Refer to caption](https://arxiv.org/html/2507.19457v1/x51.png)

(b) SelectBestCandidate + Merge

![Image 52: Refer to caption](https://arxiv.org/html/2507.19457v1/x52.png)

(c) GEPA

![Image 53: Refer to caption](https://arxiv.org/html/2507.19457v1/x53.png)

(d) GEPA+Merge - Best Config

Figure 21: HoVer GPT-4.1 Mini

![Image 54: Refer to caption](https://arxiv.org/html/2507.19457v1/x54.png)

(a) Abl:SelectBestCandidate

![Image 55: Refer to caption](https://arxiv.org/html/2507.19457v1/x55.png)

(b) SelectBestCandidate + Merge

![Image 56: Refer to caption](https://arxiv.org/html/2507.19457v1/x56.png)

(c) GEPA - Best Config

![Image 57: Refer to caption](https://arxiv.org/html/2507.19457v1/x57.png)

(d) GEPA+Merge

Figure 22: HoVer Qwen3 8B

![Image 58: Refer to caption](https://arxiv.org/html/2507.19457v1/x58.png)

(a) Abl:SelectBestCandidate

![Image 59: Refer to caption](https://arxiv.org/html/2507.19457v1/x59.png)

(b) SelectBestCandidate + Merge

![Image 60: Refer to caption](https://arxiv.org/html/2507.19457v1/x60.png)

(c) GEPA

![Image 61: Refer to caption](https://arxiv.org/html/2507.19457v1/x61.png)

(d) GEPA+Merge - Best Config

Figure 23: PUPA GPT-4.1 Mini

![Image 62: Refer to caption](https://arxiv.org/html/2507.19457v1/x62.png)

(a) Abl:SelectBestCandidate

![Image 63: Refer to caption](https://arxiv.org/html/2507.19457v1/x63.png)

(b) SelectBestCandidate + Merge

![Image 64: Refer to caption](https://arxiv.org/html/2507.19457v1/x64.png)

(c) GEPA - Best Config

![Image 65: Refer to caption](https://arxiv.org/html/2507.19457v1/x65.png)

(d) GEPA+Merge

Figure 24: PUPA Qwen3 8B

Figures[17](https://arxiv.org/html/2507.19457v1#A8.F17 "Figure 17 ‣ Appendix H GEPA Search Trees ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"), [18](https://arxiv.org/html/2507.19457v1#A8.F18 "Figure 18 ‣ Appendix H GEPA Search Trees ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"), [19](https://arxiv.org/html/2507.19457v1#A8.F19 "Figure 19 ‣ Appendix H GEPA Search Trees ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"), [20](https://arxiv.org/html/2507.19457v1#A8.F20 "Figure 20 ‣ Appendix H GEPA Search Trees ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"), [21](https://arxiv.org/html/2507.19457v1#A8.F21 "Figure 21 ‣ Appendix H GEPA Search Trees ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"), [22](https://arxiv.org/html/2507.19457v1#A8.F22 "Figure 22 ‣ Appendix H GEPA Search Trees ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"), [23](https://arxiv.org/html/2507.19457v1#A8.F23 "Figure 23 ‣ Appendix H GEPA Search Trees ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"), and [24](https://arxiv.org/html/2507.19457v1#A8.F24 "Figure 24 ‣ Appendix H GEPA Search Trees ‣ GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning") present the genetic search trees created by various configurations of GEPA (and ablation SelectBestCandidate).

Appendix I Examples of best prompts for every benchmark
-------------------------------------------------------

In this section, we present the best optimized prompt obtained for every (benchmark, model) configuration. Each subsection below pertains to one (benchmark, model) configuration. Since every compound AI system consists of multiple modules, each subsection consists of multiple boxes, listing the prompts for each module. MIPROv2 optimized prompts contain upto 4 few-shot examples for each task. We provide just the first demo here for brevity. GEPA’s prompts only consist of the optimized instruction, which is provided in full.

### I.1 HotpotQA, GPT-4.1 Mini

### I.2 HotpotQA, Qwen3 8B

### I.3 IFBench, GPT-4.1 Mini

### I.4 IFBench, Qwen3 8B

### I.5 HoVer, GPT-4.1 Mini

### I.6 HoVer, Qwen3 8B

### I.7 PUPA, GPT-4.1 Mini

### I.8 PUPA, Qwen3 8B

Appendix J GEPA for Inference-Time Search
-----------------------------------------

### J.1 NPUEval: Kernel Code Generation for new hardware architecture

![Image 66: Refer to caption](https://arxiv.org/html/2507.19457v1/manual_figures/npueval_gepa_prompt.png)

Figure 25: GEPA generated prompt for NPUEval that achieves 26.85% score with the same GPT-4o agent, that achieved a 4.25% score.

### J.2 KernelBench: CUDA Kernel Code Generation for NVIDIA GPUs

