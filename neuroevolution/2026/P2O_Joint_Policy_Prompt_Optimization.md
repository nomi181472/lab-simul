Title: P2O: Joint Policy and Prompt Optimization

URL Source: https://arxiv.org/html/2603.21877

Markdown Content:
Xinyu Lu 1,2&Kaiqi Zhang 1,2∗&Jinglin Yang 3,4,5&Boxi Cao 1&Yaojie Lu 1 Hongyu Lin 1 Min He 5 Xianpei Han 1,2 Le Sun 1,2

1 Chinese Information Processing Laboratory, Institute of 

Software, Chinese Academy of Sciences 

2 University of Chinese Academy of Sciences 

3 Institute of Information Engineering, Chinese Academy of Sciences 

4 School of Cyber Security, University of Chinese Academy of Sciences 

5 National Computer Network Emergency Response Technical 

Team/Coordination Center of China 

zhangkaiqi.zlk@gmail.com, luxinyu2021@iscas.ac.cn

###### Abstract

Reinforcement Learning with Verifiable Rewards (RLVR) enhances Large Language Model (LLM) reasoning but suffers from advantage collapse on “hard samples” where all rollouts fail. This lack of variance eliminates crucial learning signals. For these intractable samples, simply scaling up rollout budgets offers limited gains. We introduce Joint Policy and Prompt Optimization (P 2 O) to mitigate this collapse by alternating continuous policy updates with discrete prompt evolution. P 2 O leverages the GEPA algorithm to discover successful reasoning prompts for intractable instances. Via context distillation, the model internalizes these prompt-induced gains directly into its parameters, removing the need for inference-time prompting. Empirically, P 2 O restores critical advantage signals, significantly outperforming standard GRPO and surpassing baselines with doubled rollout budgets, ultimately yielding strong out-of-distribution generalization and an up to 9.5\% performance improvement. Our findings expose the limits of standard exploration in sparse-reward environments, illuminating the potential of unifying evolutionary algorithms with reinforcement learning. This integration of discrete semantic search and continuous parameter updates establishes a self-reinforcing paradigm for autonomous LLM alignment.

## 1 Introduction

Large Language Models (LLMs) have achieved remarkable proficiency across a spectrum of complex reasoning tasks[[7](https://arxiv.org/html/2603.21877#bib.bib22 "Openai o1 system card"), [5](https://arxiv.org/html/2603.21877#bib.bib21 "Deepseek-r1: incentivizing reasoning capability in llms via reinforcement learning")]. A defining characteristic of these tasks is the existence of objective verification mechanisms capable of providing deterministic feedback. Capitalizing on this, Reinforcement Learning with Verifiable Rewards (RLVR)[[8](https://arxiv.org/html/2603.21877#bib.bib23 "Tulu 3: pushing frontiers in open language model post-training")] has emerged as a dominant paradigm for reasoning alignment. By leveraging outcome-based supervision, RLVR enables models to autonomously explore the solution space, thereby bypassing the inherent bottlenecks of imitation learning[[10](https://arxiv.org/html/2603.21877#bib.bib4 "Prorl: prolonged reinforcement learning expands reasoning boundaries in large language models")].

Despite its potential, RLVR is fundamentally bottlenecked by advantage collapse: when all rollouts within a group exhibit zero reward variance—whether uniformly succeeding on trivial instances or uniformly failing on challenging ones—the advantage estimates vanish, thereby neutralizing the learning signal[[18](https://arxiv.org/html/2603.21877#bib.bib2 "Outcome-based exploration for llm reasoning")]. This failure mode on hard samples is particularly detrimental; although these instances carry high informational value, they yield no effective supervision. Moreover, scaling rollout budgets offers fundamentally limited gains, as a universally failing policy rarely discovers successful trajectories even under extensive sampling.

Common approaches to mitigate exploration challenges include curriculum learning strategies, which progressively introduce harder samples as training advances[[3](https://arxiv.org/html/2603.21877#bib.bib27 "Nemotron 3 nano: open, efficient mixture-of-experts hybrid mamba-transformer model for agentic reasoning")], and various reward shaping techniques that provide intermediate feedback. However, curriculum learning requires heuristic-based and computationally expensive schedule generation[[2](https://arxiv.org/html/2603.21877#bib.bib28 "Llama-nemotron: efficient reasoning models")], while reward shaping demands domain-specific expert heuristics.

![Image 1: Refer to caption](https://arxiv.org/html/2603.21877v3/x1.png)

Figure 1: Conceptual Illustration of the P 2 O Framework. Standard policy optimization often gets trapped in local optima (\rho_{\text{init}}) due to sparse rewards on hard samples. P 2 O bridges this exploration gap using optimized prompts (the red arrow) to reach high-reward regions that are inaccessible via standard exploration. Subsequently, the model consolidates these gains (the white arrows) by updating its parameters to master the new region (\rho_{\text{opt}}), effectively internalizing the prompt-induced capabilities.

The recent success of prompt optimization methods[[4](https://arxiv.org/html/2603.21877#bib.bib24 "Promptbreeder: self-referential self-improvement via prompt evolution"), [14](https://arxiv.org/html/2603.21877#bib.bib26 "Optimizing instructions and demonstrations for multi-stage language model programs"), [24](https://arxiv.org/html/2603.21877#bib.bib25 "Optimizing generative ai by backpropagating language model feedback")] offers a compelling way to break this stalemate. These methods demonstrate that even when a model fails to solve a hard problem under a standard policy, a carefully evolved prompt can often elicit the correct reasoning path. This implies that the solution lies within the model’s latent search space but is inaccessible via standard gradient ascent[[25](https://arxiv.org/html/2603.21877#bib.bib30 "Star: bootstrapping reasoning with reasoning")]. As illustrated in Figure[1](https://arxiv.org/html/2603.21877#S1.F1 "Figure 1 ‣ 1 Introduction ‣ P2O: Joint Policy and Prompt Optimization"), optimized prompts act as a bridge, enabling the model to “jump” out of the local optimum (\rho_{\text{init}}) and cross the reward-sparse valley. However, relying solely on inference-time prompts is insufficient. The “jump” must be followed by internalization, where the model parameters are updated to master the new region (\rho_{\text{opt}}).

Building on this intuition, we propose P 2 O (Joint P olicy and P rompt O ptimization), a novel framework that synergizes adaptive prompt evolution with reinforcement learning to overcome the hard sample bottleneck. Specifically, P 2 O identifies challenging instances during training and employs the Genetic-Pareto (GEPA)[[1](https://arxiv.org/html/2603.21877#bib.bib14 "Gepa: reflective prompt evolution can outperform reinforcement learning")] prompt optimization algorithm to evolve prompts that elicit successful reasoning chains. Rather than relying on inference-time prompting, we utilize these improved trajectories as supervision signals, enabling the policy to internalize reasoning patterns directly into its parameters. This creates a self-reinforcing cycle: as the model improves, previously hard samples become tractable, and GEPA focuses its efforts on the new frontier of difficulty. Experiments on representative reasoning benchmarks demonstrate that P 2 O consistently outperforms various baselines.

The core contributions of this paper include:

(1) We propose P²O, a framework that alternates continuous policy optimization with discrete prompt evolution to mitigate advantage collapse in RLVR on hard samples.

(2) We introduce a context distillation strategy that distills reasoning gains induced by optimized prompts directly into model parameters, eliminating inference-time prompt dependency.

(3) We empirically demonstrate that P 2 O breaks the rollout-scaling ceiling of vanilla GRPO, surpassing baselines with doubled rollout counts under identical budgets and simple reflection strategy.

## 2 Preliminaries

### 2.1 Policy Optimization

#### Problem Definition.

In Reinforcement Learning, the LLM is parameterized as a policy \pi_{\theta}. Given a query x\in\mathcal{D}, it generates a response y evaluated by a reward function r(x,y), typically derived from ground-truth verification or a learned reward model.

#### Optimization Objective.

The goal of policy optimization is to maximize the expected reward while ensuring the updated policy does not deviate excessively from a reference policy \pi_{\text{ref}}. The standard objective function is:

\mathrm{J}_{\text{RL}}(\theta)=\mathbb{E}_{x\sim\mathcal{D},y\sim\pi_{\theta}(\cdot|x)}\left[r(x,y)-\beta\mathbb{D}_{\text{KL}}(\pi_{\theta}(\cdot|x)\|\pi_{\text{ref}}(\cdot|x))\right]

#### Group Relative Policy Optimization (GRPO).

Traditional RL methods like PPO[[15](https://arxiv.org/html/2603.21877#bib.bib13 "Proximal policy optimization algorithms")] require a separate critic model to estimate the baseline for advantage computation, which doubles the memory overhead. GRPO[[16](https://arxiv.org/html/2603.21877#bib.bib12 "Deepseekmath: pushing the limits of mathematical reasoning in open language models")] eliminates the critic by employing a group-based baseline. For each query x, GRPO samples a group of K outputs \{y_{1},y_{2},\dots,y_{K}\} from the current policy. The advantage for the i-th output is computed by normalizing its reward against the group statistics:

A_{i}=\frac{r(x,y_{i})-\mu_{\text{group}}}{\sigma_{\text{group}}}

where \mu_{\text{group}} and \sigma_{\text{group}} are the mean and standard deviation of the rewards within the group. GRPO updates the policy by maximizing a surrogate objective based on these relative advantages, significantly improving training efficiency for reasoning tasks.

### 2.2 Prompt Optimization

#### Problem Definition.

Prompt optimization treats the LLM as a frozen black-box function \mathcal{M}. The optimization variable is the discrete prompt z from the space of all possible natural language strings \mathcal{P}. Given an input x, the model generates an output \hat{y}=\mathcal{M}(z,x). The problem is to find an optimal prompt z^{*} that maximizes a metric S(\hat{y},y) (e.g., accuracy score) over the validation dataset \mathcal{D}_{\text{val}}.

#### Optimization Objective.

This is a discrete, derivative-free optimization problem. The objective is formally defined as:

z^{*}=\arg\max_{z\in\mathcal{P}}\mathbb{E}_{(x,y)\sim\mathcal{D}_{\text{val}}}\left[S(\mathcal{M}(z,x),y)\right](1)

Unlike policy optimization, which adjusts continuous weights \theta, prompt optimization searches the discrete semantic space of instructions to elicit better capabilities from the frozen model.

#### GEPA (Genetic-Pareto).

To solve this optimization problem efficiently, we consider GEPA[[1](https://arxiv.org/html/2603.21877#bib.bib14 "Gepa: reflective prompt evolution can outperform reinforcement learning")], a state-of-the-art evolutionary framework. GEPA conceptualizes prompt optimization as a genetic process driven by language-based reflection. Instead of random mutations, it employs a “Reflection LLM” to analyze error traces from the current prompt’s performance and generate targeted semantic mutations that address specific failure modes. Furthermore, GEPA maintains a population of diverse prompts using a Pareto-based selection mechanism, stochastically exploring the top-performing prompts for each problem instance to prevent premature convergence.

## 3 P 2 O: Joint Policy and Prompt Optimization

### 3.1 Overview of P 2 O Framework

![Image 2: Refer to caption](https://arxiv.org/html/2603.21877v3/x2.png)

Figure 2: Overview of the P 2 O Framework. The training process is formulated as an alternating maximization procedure between two phases: (1) Policy Optimization with Context Distillation, where the policy \pi_{\theta} is updated to internalize reasoning patterns elicited by augmented inputs \tilde{x}; and (2) Evolutionary Prompt Optimization, where the prompt template set \mathcal{Z} is evolved using GEPA to discover successful trajectories for the remaining hard samples (\mathcal{D}_{\text{hard}}).

To recover critical learning signals on hard samples, we introduce Joint Policy and Prompt Optimization (P 2 O), a framework that synergizes the continuous parameter updates of policy optimization with the discrete semantic search of prompt optimization.

As illustrated in Figure[2](https://arxiv.org/html/2603.21877#S3.F2 "Figure 2 ‣ 3.1 Overview of P2O Framework ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"), P 2 O alternates between two phases: (1) Policy Optimization with Context Distillation: The model learns from prompt-guided trajectories but updates its parameters using the original input, forcing it to internalize the newly discovered reasoning steps. (2) Evolutionary Prompt Optimization: As the policy improves, we apply GEPA to evolve new prompts targeting the remaining hard samples.

### 3.2 Problem Formulation

Let \mathcal{D}_{\text{hard}}\subset\mathcal{D} denote hard samples where \pi_{\theta} yields vanishing gradients. We introduce prompt templates \mathcal{Z} as discrete latent variables and reformulate the RL objective as joint optimization over \theta and \mathcal{Z}:

\max_{\theta,\mathcal{Z}}\mathcal{J}_{\text{joint}}=\sum_{x\in\mathcal{D}\setminus\mathcal{D}_{\text{hard}}}\mathbb{E}_{y\sim\pi_{\theta}(\cdot|x)}[r(x,y)]+\sum_{x\in\mathcal{D}_{\text{hard}}}\max_{z\in\mathcal{Z}}\mathbb{E}_{y\sim\pi_{\theta}(\cdot|\mathcal{T}(x,z))}[r(x,y)](2)

where \mathcal{T} is a template insertion function. This formulation implies that while simple samples are optimized via standard exploration, hard samples require a latent instruction z^{*} to bridge the reward-sparse valley. Since joint optimization of discrete and continuous variables is intractable, P 2 O decouples this objective into an alternating maximization process between policy distillation and prompt evolution.

### 3.3 Phase 1: Policy Optimization with Context Distillation

In this phase, we fix the template set \mathcal{Z} and update the policy parameters \theta. A naïve approach would be to train the policy to map the augmented input \tilde{x} to the correct output y. However, this creates a dependency on inference-time prompting. Instead, we employ Context Distillation[[17](https://arxiv.org/html/2603.21877#bib.bib20 "Learning by distilling context")] to transfer the reasoning capabilities triggered by z directly into the parameters of \pi_{\theta} for the original input x.

#### Hard Sample Mining.

At epoch t, we dynamically identify the set of hard samples \mathcal{D}_{\text{hard}}^{(t+1)}. For each query x_{i}, we perform K rollouts. We define the hard sample set by filtering instances whose empirical success rate falls below a threshold \tau:

\mathcal{D}_{\text{hard}}^{(t+1)}=\left\{x_{i}\in\mathcal{D}\;\middle|\;\frac{1}{K}\sum_{k=1}^{K}r(x_{i},y_{ik})<\tau\right\}(3)

where y_{ik}\sim\pi_{\theta}(\cdot|x_{i}). These hard samples are collected to serve as the seed data for the subsequent Prompt Optimization phase.

#### Trajectory Augmentation & Distillation.

To maximize \mathcal{J}_{\text{joint}}, we compute the policy gradient by combining standard exploration on simple samples with prompt-guided exploration on hard samples. For a hard sample x\in\mathcal{D}_{\text{hard}}^{(t)}, we sample z\sim\mathcal{Z}^{(t)} to generate augmented trajectories \tilde{y}\sim\pi_{\theta}(\cdot|\tilde{x}), where \tilde{x}=\mathcal{T}(x,z).

Crucially, to prevent dependency on inference-time prompting, the gradient for \tilde{y} is evaluated against the original input x. The unified policy gradient is:

\nabla_{\theta}\mathcal{J}_{\text{joint}}\approx\sum_{x\notin\mathcal{D}_{\text{hard}}^{(t)}}\mathbb{E}_{y\sim\pi_{\theta}(\cdot|x)}\big[A(x,y)\nabla_{\theta}\log\pi_{\theta}(y|x)\big]+\sum_{x\in\mathcal{D}_{\text{hard}}^{(t)}}\mathbb{E}_{\tilde{y}\sim\pi_{\theta}(\cdot|\tilde{x})}\big[A(x,\tilde{y})\nabla_{\theta}\log\pi_{\theta}(\tilde{y}|x)\big](4)

Decoupling the rollout context (\tilde{x}) from the gradient context (x) inherently distills the prompt-elicited reasoning into the policy parameters \theta.

### 3.4 Phase 2: Evolutionary Prompt Optimization

In the second phase, we fix the policy \pi_{\theta} and optimize the template set \mathcal{Z}^{(t+1)} to address the newly identified hard set \mathcal{D}_{\text{hard}}^{(t+1)}. We employ GEPA (Algorithm[3](https://arxiv.org/html/2603.21877#alg3 "Algorithm 3 ‣ Appendix A Details on GEPA ‣ Limitations. ‣ 6 Conclusion and Limitations ‣ 5 Related Works ‣ Impact of Group Prompt Diversity. ‣ 4.4 Ablations ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization")) for this discrete optimization task.

#### Reflective Evolution.

GEPA utilizes the init policy model (or a teacher model) \pi_{\text{init}} as a mutation operator to “perform gradient descent in semantic space”[[22](https://arxiv.org/html/2603.21877#bib.bib29 "Large language models as optimizers")]. For each template z in the current Pareto front \mathcal{Z}_{\text{front}}, we sample a mini-batch \mathcal{D}_{\text{mini}} from \mathcal{D}_{\text{hard}}^{\text{train}} and construct error feedback \mathcal{F}_{z}=\{(x,\hat{y},y^{*})\}_{x\in\mathcal{D}_{\text{mini}}} containing the query, the failed prediction, and the ground truth. The mutation step then produces an improved candidate:

z^{\prime}\sim\pi_{\text{init}}\left(\cdot\mid\text{prompt}_{\text{meta}},z,\mathcal{F}_{z}\right)(5)

where \text{prompt}_{\text{meta}} denotes the meta instruction guiding the model to analyze errors and propose optimizations. We adopt the same meta-prompt as in Agrawal et al. [[1](https://arxiv.org/html/2603.21877#bib.bib14 "Gepa: reflective prompt evolution can outperform reinforcement learning")]. Candidates that demonstrate improvement on \mathcal{D}_{\text{mini}} are evaluated on the development set \mathcal{D}_{\text{hard}}^{\text{dev}} and added to the template pool \mathcal{Z}.

#### Pareto Selection.

Using only the best template is insufficient to cover the diversity of failure modes in reasoning tasks. GEPA maintains a population of templates and employs Pareto optimization on \mathcal{D}_{\text{hard}}^{\text{dev}} to identify nondominated candidates (Algorithm[4](https://arxiv.org/html/2603.21877#alg4 "Algorithm 4 ‣ Appendix A Details on GEPA ‣ Limitations. ‣ 6 Conclusion and Limitations ‣ 5 Related Works ‣ Impact of Group Prompt Diversity. ‣ 4.4 Ablations ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization")). This approach ensures that \mathcal{Z} remains diverse and generalizable across different error patterns. After evolution, we apply a greedy set cover strategy (Algorithm[2](https://arxiv.org/html/2603.21877#alg2 "Algorithm 2 ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization")) to select a minimal Pareto set \mathcal{Z}_{\text{covered}} and assign sample-specific templates to each instance in \mathcal{D}_{\text{hard}} for the next training epoch. This mechanism enables targeted intervention while maintaining template diversity.

The complete training procedure is summarized in Algorithm[1](https://arxiv.org/html/2603.21877#alg1 "Algorithm 1 ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"). By iteratively refining the policy to absorb prompt-induced capabilities and evolving prompts to target the policy’s remaining weaknesses, P 2 O establishes an iterative self-improvement process.

Algorithm 1 P 2 O

0: Training Data

\mathcal{D}
, Initial Template Set

\mathcal{Z}^{(0)}=\emptyset

0: Policy Model

\pi_{\theta}
, Reference Model

\pi_{\text{init}}

0: Epochs

N_{\text{epoch}}
, Rollout Num

K
, Threshold

\tau

0: Optimized Policy

\pi_{\theta^{*}}

1:for

t=0
to

N_{\text{epoch}}-1
do

2: Initialize next hard set

\mathcal{D}_{\text{hard}}^{(t+1)}\leftarrow\emptyset

3:// Phase 1: Policy Optimization

4:for all batch

(x_{i},\hat{y}_{i})\in\mathcal{D}
do

5:for

k=1
to

K
do

6:

x_{ik}\leftarrow x_{i}

7:if

x_{i}\in\mathcal{D}_{\text{hard}}^{(t)}
and

\mathcal{Z}^{(t)}\neq\emptyset
then

8: Sample template

z\sim\mathcal{Z}^{(t)}

9:

x_{ik}\leftarrow\mathcal{T}(x_{ik},z)

10:// \mathcal{T} is a template insertion function

11:end if

12: Sample

y_{ik}\sim\pi_{\theta}(\cdot|x_{ik})

13:

r_{ik}\leftarrow\textsc{RewardFunc}(y_{ik},\hat{y}_{i})

14:end for

15:

\bar{r}_{i}\leftarrow\frac{1}{K}\sum_{k=1}^{K}r_{ik}

16:if

\bar{r}_{i}<\tau
then

17:

\mathcal{D}_{\text{hard}}^{(t+1)}\leftarrow\mathcal{D}_{\text{hard}}^{(t+1)}\cup\{x_{i}\}

18:end if

19: Compute advantages

A_{ik}

20: Update

\pi_{\theta}
using

\{x_{i},y_{ik},A_{ik}\}

21:end for

22:// Phase 2: Prompt Optimization

23:if

\mathcal{D}_{\text{hard}}^{(t+1)}\neq\emptyset
then

24:

\mathcal{Z}^{(t+1)}\leftarrow\textsc{Gepa}(\mathcal{D}_{\text{hard}}^{(t+1)},\pi_{\theta},\pi_{\text{init}})
// Alg.3

25:else

26:

\mathcal{Z}^{(t+1)}\leftarrow\emptyset

27:end if

28:end for

Algorithm 2 GreedyPromptAssignment

0: Templates with scores

\mathcal{Z}=\{(z_{i},\mathbf{r}_{i})\}_{i=1}^{M}

0: Hard samples

\mathcal{D}_{\text{hard}}
, Rollout Num

K
// \mathbf{r}_{i}=(r_{i,1},\ldots,r_{i,N}): scores of z_{i} on N dev samples

0: Sample-specific template assignments

\{(x_{j},\mathbf{z}_{j})\}
// Step 1: Greedy set cover to find minimal Pareto set

1:Initialize Template Pareto Set

\mathcal{Z}_{\text{covered}}\leftarrow\emptyset
and Covered Sample Set

\mathcal{S}_{\text{covered}}\leftarrow\emptyset

2:while True do

3: Find

z^{*}\leftarrow\arg\max_{z_{i}\notin\mathcal{Z}_{\text{covered}}}|\mathcal{C}(z_{i})\setminus\mathcal{S}_{\text{covered}}|

4:// \mathcal{C}(z_{i})=\{x_{j}\mid r_{i,j}=1\} is the set of dev samples solved by z_{i}

5:if

|\mathcal{C}(z^{*})\setminus\mathcal{S}_{\text{covered}}|=0
then

6:break

7:end if

8:

\mathcal{Z}_{\text{covered}}\leftarrow\mathcal{Z}_{\text{covered}}\cup\{z^{*}\}

9:

\mathcal{S}_{\text{covered}}\leftarrow\mathcal{S}_{\text{covered}}\cup\mathcal{C}(z^{*})

10:end while

11:// Step 2: Assign K templates to each hard sample

12: Compute weights:

\bar{r}_{i}\leftarrow\frac{1}{N}\sum_{n=1}^{N}r_{i,n}
for each

z_{i}\in\mathcal{Z}_{\text{covered}}

13:Initialize assignment:

\mathcal{A}\leftarrow\{\}

14:for

x_{j}
in

\mathcal{D}_{\text{hard}}
do

15:if

x_{j}\in\mathcal{S}_{\text{covered}}
then

16:

\mathcal{Z}_{j}\leftarrow\{z_{i}\in\mathcal{Z}_{\text{covered}}\mid x_{j}\in\mathcal{C}(z_{i})\}

17:// Sample from templates that solve this sample

18:else

19:

\mathcal{Z}_{j}\leftarrow\mathcal{Z}_{\text{covered}}

20:end if

21: Sample

\mathbf{z}_{j}=(z_{j,1},\ldots,z_{j,K})
from

\mathcal{Z}_{j}
weighted for

K
times by

\{\bar{r}_{i}\}

22:

\mathcal{A}\leftarrow\mathcal{A}\cup\{(x_{j},\mathbf{z}_{j})\}

23:end for

24: RETURN

\mathcal{A}

Table 1: Comparative Evaluation on Mathematical Reasoning Benchmarks. We report the accuracy (%) of P 2 O against baselines. P 2 O consistently outperforms the baselines, achieving the most significant gains on challenging benchmarks such as AIME24 and AIME25. Best results are highlighted in bold.

## 4 Experiments

### 4.1 Settings

#### Datasets.

We conduct experiments on two distinct datasets: a subset randomly sampled from DeepScaler[[12](https://arxiv.org/html/2603.21877#bib.bib15 "DeepScaleR: surpassing o1-preview with a 1.5b model by scaling rl")] and a subset of DeepMath[[6](https://arxiv.org/html/2603.21877#bib.bib16 "Deepmath-103k: a large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning")] filtered to difficulty level \geq 7. For each dataset, we randomly sample two splits of N=5{,}000 and N=10{,}000 examples respectively.

#### Method Configuration.

We combine GRPO with a one-step off-policy strategy, excluding the KL divergence penalty. The training hyperparameters include a maximum learning rate of 1\times 10^{-6}, a global batch size of 128, and a maximum generation length of 12k tokens. During the rollout phase, we utilize a temperature of T=0.6 and sample K=6 trajectories per prompt. All models are trained for 10 epochs and evaluated with the best dev checkpoint. We use Qwen3-4B[[21](https://arxiv.org/html/2603.21877#bib.bib31 "Qwen3 technical report")] as the training backbone. For GEPA, we use a validation set of 300 examples. Unlike the original GEPA setting[[1](https://arxiv.org/html/2603.21877#bib.bib14 "Gepa: reflective prompt evolution can outperform reinforcement learning")], we use beam size W to parallelize the Pareto search for improved computational efficiency. The candidate selection beam size is set to W=16 (yielding \sim 40 templates per iteration). Regarding the reflection model in GEPA, we evaluate two variants: P 2 O{}_{\text{Self-Ref}}, which utilizes the reference model (Qwen3-4B) as the mutation operator, and P 2 O{}_{\text{Teacher-Ref}}, which employs Kimi-K2[[19](https://arxiv.org/html/2603.21877#bib.bib32 "Kimi k2: open agentic intelligence")] as a stronger external teacher to provide high-quality feedback and prompt mutations.

#### Reward.

We employ a strict binary reward (r\in\{0,1\}). A reward of 1 is granted solely when the response follows the \boxed{} format and matches the ground truth.

#### Evaluation Protocol.

We employ the open-source evaluation suite 1 1 1[https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation) provided by Qwen for all mathematical benchmarks. To ensure statistical robustness and encourage exploration on smaller datasets (fewer than 100 samples), we report the average performance across 16 rollouts per question, generated with a temperature of 0.6. For larger datasets, we adopt greedy sampling.

#### Baselines.

We compare P 2 O against four baselines throughout the experiments:

(1) GRPO[[16](https://arxiv.org/html/2603.21877#bib.bib12 "Deepseekmath: pushing the limits of mathematical reasoning in open language models")] with varying rollout budgets. This standard RLVR baseline serves to test the naive scaling hypothesis: whether advantage collapse on hard samples can be resolved simply by allocating more compute for exploration.

(2) DAPO[[23](https://arxiv.org/html/2603.21877#bib.bib3 "Dapo: an open-source llm reinforcement learning system at scale")]. For a fair comparison, we isolate its dynamic sampling strategy, as P 2 O is complementary to DAPO’s clip-higher and token-level policy gradient loss. This baseline is included to validate our premise that actively bridging the exploration gap on hard samples yields superior convergence compared to passively discounting them.

(3) Single-Turn Reflection. This ablation replaces GEPA with a single round of self reflection per hard sample at each epoch’s end, serving to validate the necessity of GEPA’s complex prompt evolution and search strategies.

(4) Teacher-Distill-SFT. A compute-matched baseline designed to test whether P 2 O{}_{\text{Teacher-Ref}}’s teacher budget is better spent on data synthesis. It uses an identical teacher budget to generate two rollouts per problem, filters trajectories via ground-truth-verified rejection sampling, and fine-tunes the model via behavioral cloning.

### 4.2 Main Results

![Image 3: Refer to caption](https://arxiv.org/html/2603.21877v3/x3.png)

Figure 3: Training Dynamics and Prompt Optimization Effectiveness of P 2 O.Top: Training reward (left) and validation accuracy (right) of P 2 O and the GRPO baseline throughout optimization. Bottom-left: Optimized prompts (triangles) consistently outperform the standard prompts (circles) on both Pass@1 and Pass@6, showing that GEPA helps bridge the performance gap. Bottom-right: With this assistance, the model steadily conquers hard samples, yielding a continuous decline in intractable instances across training epochs. Results are from Teacher-Ref variant on the DeepScaler-5K dataset.

Table[3.4](https://arxiv.org/html/2603.21877#S3.SS4.SSS0.Px2 "Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization") summarizes the performance of models trained on DeepMath and DeepScaler across six challenging mathematical benchmarks, comparing P 2 O against Qwen3 base models and the baselines.

#### Comparison with Baselines.

As shown in Table[3.4](https://arxiv.org/html/2603.21877#S3.SS4.SSS0.Px2 "Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"), P 2 O consistently outpaces all baselines by directly addressing the exploration bottleneck on hard samples.

First, P 2 O consistently outperforms standard GRPO and its variants. We observe that naively scaling the GRPO rollout budget (from 6 to 12) yields diminishing returns, with average accuracy plateauing near 54.8% on DeepMath-5K and 57.0% on DeepScaler-5K, while passively discounting hard samples (as in DAPO) actually underperforms standard GRPO by 3.2%. By replacing the sampling budget with prompt evolution, P 2 O breaks this scaling ceiling, achieving 64.2% on DeepMath-5K under compute-equivalent conditions—a robust 9% improvement over GRPO runs. Meanwhile, P 2 O continues to outperform the GRPO baseline on both DeepMath-10K and DeepScaler-10K.

Second, P 2 O substantially surpasses the single-turn reflection baseline, which falls below even the standard GRPO performance (scoring only 50.4% on DeepMath-5K). This severe degradation proves that simple teacher feedback is insufficient to provide high-quality guidance. Instead, it requires GEPA’s iterative evolutionary process—which systematically optimizes the prompt against a single core objective—to reliably discover and internalize the correct reasoning trajectories.

#### Impact of Reflection Models on Prompt Evolution.

The optimal reflection source proves task-dependent: on DeepScaler-5K, Teacher-Reflection dominates (65.2% vs. 62.4% avg.), while on DeepMath-5K, P 2 O{}_{\text{Self-Ref}} leads (64.2% vs. 57.9% avg.). This suggests that for certain domains, self-generated refinements better align with the policy’s reasoning capacity, whereas external teacher priors may be harder to internalize. Scaling to 10K amplifies this pattern: on DeepMath-10K, P 2 O{}_{\text{Self-Ref}} achieves the best overall result (66.2% avg.) while Teacher-Reflection drops to 55.5%; on DeepScaler-10K, both remain competitive (62.7% vs. 58.9% avg.). Crucially, all P 2 O variants consistently surpass the GRPO baseline across all datasets, confirming that GEPA-driven structural exploration robustly improves performance regardless of the reflection source.

### 4.3 Training Dynamics

As shown in Figure[3](https://arxiv.org/html/2603.21877#S4.F3 "Figure 3 ‣ 4.2 Main Results ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization")(top), the continuous integration of optimized prompts maintains a training reward consistently superior to that of the GRPO baseline. Crucially, this advantage translates into gains in validation accuracy, confirming that the model effectively internalizes the elicited reasoning patterns to achieve robust in-distribution generalization.

Meanwhile, as shown in Figure[3](https://arxiv.org/html/2603.21877#S4.F3 "Figure 3 ‣ 4.2 Main Results ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization")(bottom), as the model’s intrinsic capability grows during training, GEPA effectively assists in continuously conquering “hard samples”, as evidenced by the steady decline in the number of intractable instances (Right). Furthermore, the performance breakdown (Left) reveals that prompt optimization yields consistent gains across both Pass@1 and Pass@6 metrics. This demonstrates that the evolved templates do not merely facilitate a single lucky guess; rather, they robustly enhance the model’s solution space, boosting both the deterministic success rate and the broader exploration coverage required for effective distillation.

### 4.4 Ablations

Table 2: Ablation Study of P 2 O Components. Comparison of performance on mathematical benchmarks when excluding context distillation or group prompt diversity. Data points are derived from the Teacher-Ref variant on the DeepScaler-5K dataset.

#### Importance of Context Distillation.

We investigate the necessity of context distillation by ablating how the supervision signal is constructed. Recall that our method samples a response y\sim\pi_{\theta}(\cdot\mid\tilde{x}) from the prompt-augmented input \tilde{x}=\mathcal{T}(x,z), but computes the policy gradient on (x,y) using the _original_ query x, thereby distilling the behavior elicited by \tilde{x} back into \pi_{\theta}(\cdot\mid x). In the ablated variant (P 2 O{}_{\text{w/o context distillation}}), the gradient is instead computed on (\tilde{x},y), optimizing the model conditioned on the augmented prompt rather than on x. As shown in Table[2](https://arxiv.org/html/2603.21877#S4.T2 "Table 2 ‣ 4.4 Ablations ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"), this causes a severe drop: average accuracy falls from 65.2% (P 2 O{}_{\text{Teacher-Ref}}) to 55.6%, even 4.9% below the GRPO baseline (60.5%). Training on (\tilde{x},y) induces a “dependency” effect, where the model relies on the auxiliary prompt z rather than internalizing the reasoning logic, and thus fails to generalize when only x is available at evaluation. Context distillation is therefore a prerequisite, not merely an enhancement, for transferring the teacher reference’s capabilities into the model’s intrinsic parameters.

#### Impact of Group Prompt Diversity.

We investigate prompt diversity within rollout groups during training. Standard P 2 O naturally induces template diversity across the K rollouts per query via the greedy assignment in Algorithm[2](https://arxiv.org/html/2603.21877#alg2 "Algorithm 2 ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"), whereas the baseline variant P 2 O{}_{\text{same template in group}} fixes a single sampled template z\sim\mathcal{Z}^{(t)} for all K rollouts. As shown in Table[2](https://arxiv.org/html/2603.21877#S4.T2 "Table 2 ‣ 4.4 Ablations ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"), the diversity-driven approach achieves 65.2% average accuracy, outperforming the single-template baseline (64.2%), with pronounced gains on AIME24 (+2.5%) and AIME25 (+4.8%). This confirms that diverse Pareto-optimal prompts within a group broaden reasoning space coverage, yielding more diverse supervision signals and more effective distillation of complex reasoning capabilities.

## 5 Related Works

RLVR and GRPO are widely used for LLM reasoning alignment, stabilizing training via group baselines and verification, but still struggle with exploration in sparse or misleading landscapes. To address these scalability and stability issues, recent works have focused on algorithmic advances. For instance, DAPO [[23](https://arxiv.org/html/2603.21877#bib.bib3 "Dapo: an open-source llm reinforcement learning system at scale")] prunes prompts with accuracy equal to 1 or 0 to improve training stability. Despite these improvements, these methods primarily focus on optimizing the policy on a fixed search space, leaving the initial task prompts untouched.

To further aid the model in traversing complex reasoning paths, various “hint-based” strategies have been proposed, including strong model guidance[[13](https://arxiv.org/html/2603.21877#bib.bib8 "Adaptive guidance accelerates reinforcement learning of reasoning models"), [11](https://arxiv.org/html/2603.21877#bib.bib7 "Ghpo: adaptive guidance for stable and efficient llm reinforcement learning")], experience replay data[[26](https://arxiv.org/html/2603.21877#bib.bib11 "ExGRPO: learning to reason from experience")], and expert solutions[[28](https://arxiv.org/html/2603.21877#bib.bib10 "BREAD: branched rollouts from expert anchors bridge sft & rl for reasoning"), [20](https://arxiv.org/html/2603.21877#bib.bib5 "Learning to reason under off-policy guidance"), [9](https://arxiv.org/html/2603.21877#bib.bib9 "Questa: expanding reasoning capacity in llms via question augmentation")] attempt to scaffold the reasoning process. Similarly, Critique-GRPO[[27](https://arxiv.org/html/2603.21877#bib.bib6 "Critique-grpo: advancing llm reasoning with natural language and numerical feedback")] utilizes feedback signals to guide the policy. However, most of these methods rely heavily on external expert supervision or expert trajectory fragments. In contrast, P 2 O achieves self-improvement without external guidance. By treating the prompt as an optimizable parameter and evolving it jointly with the policy, P 2 O matches or exceeds the performance of teacher-dependent variants while remaining compatible with teacher reflection.

## 6 Conclusion and Limitations

In this paper, we introduce P 2 O, a novel framework that bridges the gap between discrete prompt evolution and continuous policy optimization to overcome the collapse of advantages in RLVR. By dynamically identifying hard samples and leveraging the GEPA algorithm, P 2 O evolves targeted prompt templates that guide the policy toward discovering successful reasoning trajectories that are otherwise inaccessible. Crucially, our context distillation mechanism ensures that these prompt-elicited capabilities are internalized directly into the model parameters, eliminating reliance on inference-time guidance. Extensive evaluations across challenging mathematical reasoning benchmarks demonstrate that P 2 O significantly outperforms standard GRPO paradigms, establishing joint optimization as a robust pathway for autonomous self-improvement in LLMs.

#### Limitations.

The primary limitation of P 2 O is the increased computational overhead during the prompt evolution phrase compared to the vanilla GRPO baseline. However, as detailed in Section[4.1](https://arxiv.org/html/2603.21877#S4.SS1.SSS0.Px5 "Baselines. ‣ 4.1 Settings ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"), the additional computation yields higher sample efficiency.

## References

*   [1]L. A. Agrawal, S. Tan, D. Soylu, N. Ziems, R. Khare, K. Opsahl-Ong, A. Singhvi, H. Shandilya, M. J. Ryan, M. Jiang, et al. (2025)Gepa: reflective prompt evolution can outperform reinforcement learning. arXiv preprint arXiv:2507.19457. Cited by: [§1](https://arxiv.org/html/2603.21877#S1.p5.3 "1 Introduction ‣ P2O: Joint Policy and Prompt Optimization"), [§2.2](https://arxiv.org/html/2603.21877#S2.SS2.SSS0.Px3.p1.1 "GEPA (Genetic-Pareto). ‣ 2.2 Prompt Optimization ‣ 2 Preliminaries ‣ P2O: Joint Policy and Prompt Optimization"), [§3.4](https://arxiv.org/html/2603.21877#S3.SS4.SSS0.Px1.p1.10 "Reflective Evolution. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"), [§4.1](https://arxiv.org/html/2603.21877#S4.SS1.SSS0.Px2.p1.10 "Method Configuration. ‣ 4.1 Settings ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [2]A. Bercovich, I. Levy, I. Golan, M. Dabbah, R. El-Yaniv, O. Puny, I. Galil, Z. Moshe, T. Ronen, N. Nabwani, et al. (2025)Llama-nemotron: efficient reasoning models. arXiv preprint arXiv:2505.00949. Cited by: [§1](https://arxiv.org/html/2603.21877#S1.p3.1 "1 Introduction ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [3]A. Blakeman, A. Grattafiori, A. Basant, A. Gupta, A. Khattar, A. Renduchintala, A. Vavre, A. Shukla, A. Bercovich, A. Ficek, et al. (2025)Nemotron 3 nano: open, efficient mixture-of-experts hybrid mamba-transformer model for agentic reasoning. arXiv preprint arXiv:2512.20848. Cited by: [§1](https://arxiv.org/html/2603.21877#S1.p3.1 "1 Introduction ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [4]C. Fernando, D. Banarse, H. Michalewski, S. Osindero, and T. Rocktäschel (2023)Promptbreeder: self-referential self-improvement via prompt evolution. arXiv preprint arXiv:2309.16797. Cited by: [§1](https://arxiv.org/html/2603.21877#S1.p4.2 "1 Introduction ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [5]D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al. (2025)Deepseek-r1: incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948. Cited by: [§1](https://arxiv.org/html/2603.21877#S1.p1.1 "1 Introduction ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [6]Z. He, T. Liang, J. Xu, Q. Liu, X. Chen, Y. Wang, L. Song, D. Yu, Z. Liang, W. Wang, et al. (2025)Deepmath-103k: a large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning. arXiv preprint arXiv:2504.11456. Cited by: [§4.1](https://arxiv.org/html/2603.21877#S4.SS1.SSS0.Px1.p1.3 "Datasets. ‣ 4.1 Settings ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [7]A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel, A. Carney, et al. (2024)Openai o1 system card. arXiv preprint arXiv:2412.16720. Cited by: [§1](https://arxiv.org/html/2603.21877#S1.p1.1 "1 Introduction ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [8]N. Lambert, J. Morrison, V. Pyatkin, S. Huang, H. Ivison, F. Brahman, L. J. V. Miranda, A. Liu, N. Dziri, S. Lyu, et al. (2024)Tulu 3: pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124. Cited by: [§1](https://arxiv.org/html/2603.21877#S1.p1.1 "1 Introduction ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [9]J. Li, H. Lin, H. Lu, K. Wen, Z. Yang, J. Gao, Y. Wu, and J. Zhang (2025)Questa: expanding reasoning capacity in llms via question augmentation. arXiv preprint arXiv:2507.13266. Cited by: [§5](https://arxiv.org/html/2603.21877#S5.p2.2 "5 Related Works ‣ Impact of Group Prompt Diversity. ‣ 4.4 Ablations ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [10]M. Liu, S. Diao, X. Lu, J. Hu, X. Dong, Y. Choi, J. Kautz, and Y. Dong (2025)Prorl: prolonged reinforcement learning expands reasoning boundaries in large language models. arXiv preprint arXiv:2505.24864. Cited by: [§1](https://arxiv.org/html/2603.21877#S1.p1.1 "1 Introduction ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [11]Z. Liu, C. Gong, X. Fu, Y. Liu, R. Chen, S. Hu, S. Zhang, R. Liu, Q. Zhang, and D. Tu (2025)Ghpo: adaptive guidance for stable and efficient llm reinforcement learning. arXiv preprint arXiv:2507.10628. Cited by: [§5](https://arxiv.org/html/2603.21877#S5.p2.2 "5 Related Works ‣ Impact of Group Prompt Diversity. ‣ 4.4 Ablations ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [12]M. Luo, S. Tan, J. Wong, X. Shi, W. Y. Tang, M. Roongta, C. Cai, J. Luo, L. E. Li, R. A. Popa, and I. Stoica (2025)DeepScaleR: surpassing o1-preview with a 1.5b model by scaling rl. Note: Notion Blog Cited by: [§4.1](https://arxiv.org/html/2603.21877#S4.SS1.SSS0.Px1.p1.3 "Datasets. ‣ 4.1 Settings ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [13]V. Nath, E. Lau, A. Gunjal, M. Sharma, N. Baharte, and S. Hendryx (2025)Adaptive guidance accelerates reinforcement learning of reasoning models. arXiv preprint arXiv:2506.13923. Cited by: [§5](https://arxiv.org/html/2603.21877#S5.p2.2 "5 Related Works ‣ Impact of Group Prompt Diversity. ‣ 4.4 Ablations ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [14]K. Opsahl-Ong, M. J. Ryan, J. Purtell, D. Broman, C. Potts, M. Zaharia, and O. Khattab (2024-11)Optimizing instructions and demonstrations for multi-stage language model programs. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Y. Al-Onaizan, M. Bansal, and Y. Chen (Eds.), Miami, Florida, USA,  pp.9340–9366. External Links: [Link](https://aclanthology.org/2024.emnlp-main.525/), [Document](https://dx.doi.org/10.18653/v1/2024.emnlp-main.525)Cited by: [§1](https://arxiv.org/html/2603.21877#S1.p4.2 "1 Introduction ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [15]J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov (2017)Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347. Cited by: [§2.1](https://arxiv.org/html/2603.21877#S2.SS1.SSS0.Px3.p1.4 "Group Relative Policy Optimization (GRPO). ‣ 2.1 Policy Optimization ‣ 2 Preliminaries ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [16]Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. (2024)Deepseekmath: pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300. Cited by: [§2.1](https://arxiv.org/html/2603.21877#S2.SS1.SSS0.Px3.p1.4 "Group Relative Policy Optimization (GRPO). ‣ 2.1 Policy Optimization ‣ 2 Preliminaries ‣ P2O: Joint Policy and Prompt Optimization"), [§4.1](https://arxiv.org/html/2603.21877#S4.SS1.SSS0.Px5.p2.1 "Baselines. ‣ 4.1 Settings ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [17]C. Snell, D. Klein, and R. Zhong (2022)Learning by distilling context. arXiv preprint arXiv:2209.15189. Cited by: [§3.3](https://arxiv.org/html/2603.21877#S3.SS3.p1.7 "3.3 Phase 1: Policy Optimization with Context Distillation ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [18]Y. Song, J. Kempe, and R. Munos (2025)Outcome-based exploration for llm reasoning. arXiv preprint arXiv:2509.06941. Cited by: [§1](https://arxiv.org/html/2603.21877#S1.p2.1 "1 Introduction ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [19]K. Team, Y. Bai, Y. Bao, G. Chen, J. Chen, N. Chen, R. Chen, Y. Chen, Y. Chen, Y. Chen, et al. (2025)Kimi k2: open agentic intelligence. arXiv preprint arXiv:2507.20534. Cited by: [§4.1](https://arxiv.org/html/2603.21877#S4.SS1.SSS0.Px2.p1.10 "Method Configuration. ‣ 4.1 Settings ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [20]J. Yan, Y. Li, Z. Hu, Z. Wang, G. Cui, X. Qu, Y. Cheng, and Y. Zhang (2025)Learning to reason under off-policy guidance. arXiv preprint arXiv:2504.14945. Cited by: [§5](https://arxiv.org/html/2603.21877#S5.p2.2 "5 Related Works ‣ Impact of Group Prompt Diversity. ‣ 4.4 Ablations ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [21]A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al. (2025)Qwen3 technical report. arXiv preprint arXiv:2505.09388. Cited by: [§4.1](https://arxiv.org/html/2603.21877#S4.SS1.SSS0.Px2.p1.10 "Method Configuration. ‣ 4.1 Settings ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [22]C. Yang, X. Wang, Y. Lu, H. Liu, Q. V. Le, D. Zhou, and X. Chen (2024)Large language models as optimizers. In ICLR, External Links: [Link](https://openreview.net/forum?id=Bb4VGOWELI)Cited by: [§3.4](https://arxiv.org/html/2603.21877#S3.SS4.SSS0.Px1.p1.6 "Reflective Evolution. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [23]Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, W. Dai, T. Fan, G. Liu, L. Liu, et al. (2025)Dapo: an open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476. Cited by: [§4.1](https://arxiv.org/html/2603.21877#S4.SS1.SSS0.Px5.p3.1 "Baselines. ‣ 4.1 Settings ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"), [§5](https://arxiv.org/html/2603.21877#S5.p1.2 "5 Related Works ‣ Impact of Group Prompt Diversity. ‣ 4.4 Ablations ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [24]M. Yuksekgonul, F. Bianchi, J. Boen, S. Liu, P. Lu, Z. Huang, C. Guestrin, and J. Zou (2025)Optimizing generative ai by backpropagating language model feedback. Nature 639,  pp.609–616. Cited by: [§1](https://arxiv.org/html/2603.21877#S1.p4.2 "1 Introduction ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [25]E. Zelikman, Y. Wu, J. Mu, and N. Goodman (2022)Star: bootstrapping reasoning with reasoning. Advances in Neural Information Processing Systems 35,  pp.15476–15488. Cited by: [§1](https://arxiv.org/html/2603.21877#S1.p4.2 "1 Introduction ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [26]R. Zhan, Y. Li, Z. Wang, X. Qu, D. Liu, J. Shao, D. F. Wong, and Y. Cheng (2025)ExGRPO: learning to reason from experience. arXiv preprint arXiv:2510.02245. Cited by: [§5](https://arxiv.org/html/2603.21877#S5.p2.2 "5 Related Works ‣ Impact of Group Prompt Diversity. ‣ 4.4 Ablations ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [27]X. Zhang, H. Sun, Y. Zhang, K. Feng, C. Lu, C. Yang, and H. Meng (2025)Critique-grpo: advancing llm reasoning with natural language and numerical feedback. arXiv preprint arXiv:2506.03106. Cited by: [§5](https://arxiv.org/html/2603.21877#S5.p2.2 "5 Related Works ‣ Impact of Group Prompt Diversity. ‣ 4.4 Ablations ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"). 
*   [28]X. Zhang, Z. Huang, Y. Li, C. Ni, J. Chen, and S. Oymak (2025)BREAD: branched rollouts from expert anchors bridge sft & rl for reasoning. arXiv preprint arXiv:2506.17211. Cited by: [§5](https://arxiv.org/html/2603.21877#S5.p2.2 "5 Related Works ‣ Impact of Group Prompt Diversity. ‣ 4.4 Ablations ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"). 

## Appendix A Details on GEPA

We provide the comprehensive pseudocode for the Genetic-Pareto (GEPA) prompt optimization process used in Phase 2 of our framework. Please refer to Algorithm[3](https://arxiv.org/html/2603.21877#alg3 "Algorithm 3 ‣ Appendix A Details on GEPA ‣ Limitations. ‣ 6 Conclusion and Limitations ‣ 5 Related Works ‣ Impact of Group Prompt Diversity. ‣ 4.4 Ablations ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization") for the detailed execution of the main evolutionary loop, and Algorithm[4](https://arxiv.org/html/2603.21877#alg4 "Algorithm 4 ‣ Appendix A Details on GEPA ‣ Limitations. ‣ 6 Conclusion and Limitations ‣ 5 Related Works ‣ Impact of Group Prompt Diversity. ‣ 4.4 Ablations ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization") for the specific implementation of the Pareto-based frontier selection strategy.

Algorithm 3 Gepa

0: Hard Data

\mathcal{D}_{\text{hard}}
, Policy Model

\pi_{\theta}
, Reference Model

\pi_{\text{init}}

0: Budget

C_{\text{total}}
, Mini-batch

B
, Width

W

0: Output Template Set

\mathcal{Z}

1: Split

\mathcal{D}_{\text{hard}}
into

\mathcal{D}_{\text{hard}}^{\text{train}}
and

\mathcal{D}_{\text{hard}}^{\text{dev}}

2: Initialize

\mathcal{Z}\leftarrow\{(\epsilon,\textsc{Eval}(\pi_{\theta},\epsilon,\mathcal{D}_{\text{hard}}^{\text{dev}}))\}

3:// \epsilon means empty template.

4:// Eval() will return reward of every sample in the given dataset by applying the given template

5:

C_{\text{left}}\leftarrow C_{\text{total}}

6:while

C_{\text{left}}>0
do

7:

\mathcal{Z}_{\text{front}}\leftarrow\textsc{SelectParetoFront}(\mathcal{Z},W)

8:for all

z\in\mathcal{Z}_{\text{front}}
do

9: Sample mini-batch

\mathcal{D}_{\text{mini}}\subset\mathcal{D}_{\text{hard}}^{\text{train}}
of size

B

10:

\bar{r}_{\text{old}}\leftarrow\operatorname{mean}(\textsc{Eval}(\pi_{\theta},z,\mathcal{D}_{\text{mini}}))

11: Generate error feedback

\mathcal{F}
from rollouts and rewards on

\mathcal{D}_{\text{mini}}

12:

z^{\prime}\leftarrow\pi_{\text{init}}(\text{``Propose Improvement''},z,\mathcal{F})

13:

\bar{r}_{\text{new}}\leftarrow\operatorname{mean}(\textsc{Eval}(\pi_{\theta},z^{\prime},\mathcal{D}_{\text{mini}}))

14: Update Cost:

C_{\text{left}}\leftarrow C_{\text{left}}-2B

15:if

\bar{r}_{\text{new}}>\bar{r}_{\text{old}}
then

16:

\mathcal{Z}\leftarrow\mathcal{Z}\cup\{(z^{\prime},\textsc{Eval}(\pi_{\theta},z^{\prime},\mathcal{D}_{\text{hard}}^{\text{dev}}))\}

17:

C_{\text{left}}\leftarrow C_{\text{left}}-|\mathcal{D}_{\text{hard}}^{\text{dev}}|

18:end if

19:end for

20:end while

21:

\mathcal{Z}\leftarrow\textsc{GreedyPromptAssignment}(\mathcal{Z},\mathcal{D}_{\text{hard}})

22: RETURN

\mathcal{Z}

Algorithm 4 SelectParetoFront

0: Template set with scores

\mathcal{Z}=\{(z_{1},(r_{1,1},\ldots,r_{1,N})),\ldots,(z_{M},(r_{M,1},\ldots,r_{M,N}))\}
, Width

W

1:// Each (r_{i,1},\ldots,r_{i,N}) contains scores of template z_{i} on N dev samples

1: Selected templates

\mathcal{Z}_{\text{front}}

2:// Step 1: Identify Pareto-optimal templates

3:

\mathcal{Z}_{\text{front}}\leftarrow\emptyset

4:for

i=1
to

M
do

5:

\text{dominated}\leftarrow\textsc{False}

6:for

j=1
to

M
do

7:if

j\neq i
and

z_{j}
dominates

z_{i}
then

8:// z_{j} dominates z_{i} if \exists n:r_{j,n}>r_{i,n} and \forall n^{\prime}:r_{j,n^{\prime}}\geq r_{i,n^{\prime}}

9:

\text{dominated}\leftarrow\textsc{True}

10:break

11:end if

12:end for

13:if not dominated then

14:

\mathcal{Z}_{\text{front}}\leftarrow\mathcal{Z}_{\text{front}}\cup\{(z_{i},(r_{i,1},\ldots,r_{i,N}))\}

15:end if

16:end for

17:// Step 2: Select W templates from Pareto front

18:if

|\mathcal{Z}_{\text{front}}|\leq W
then

19:

\mathcal{Z}_{\text{front}}\leftarrow\{z\mid(z,(r_{1},\ldots,r_{N}))\in\mathcal{Z}_{\text{front}}\}

20:else

21:// Sample based on mean dev scores

22: Compute weights:

\bar{r}_{i}\leftarrow\frac{1}{N}\sum_{n=1}^{N}r_{i,n}
for each

(z_{i},(r_{i,1},\ldots,r_{i,N}))\in\mathcal{Z}_{\text{front}}

23: Sample

W
templates from

\mathcal{Z}_{\text{front}}
weighted by

\{\bar{r}_{i}\}

24:

\mathcal{Z}_{\text{front}}\leftarrow\{\text{sampled templates}\}

25:end if

26: RETURN

\mathcal{Z}_{\text{front}}

## Appendix B Training Dynamics of GRPO

In this section, we present the training dynamics of the GRPO baseline with different rollout budgets in Figures[4](https://arxiv.org/html/2603.21877#A2.F4 "Figure 4 ‣ Appendix B Training Dynamics of GRPO ‣ Limitations. ‣ 6 Conclusion and Limitations ‣ 5 Related Works ‣ Impact of Group Prompt Diversity. ‣ 4.4 Ablations ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization") and[5](https://arxiv.org/html/2603.21877#A2.F5 "Figure 5 ‣ Appendix B Training Dynamics of GRPO ‣ Limitations. ‣ 6 Conclusion and Limitations ‣ 5 Related Works ‣ Impact of Group Prompt Diversity. ‣ 4.4 Ablations ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization").

![Image 4: Refer to caption](https://arxiv.org/html/2603.21877v3/x4.png)

Figure 4: Training Dynamics of GRPO on DeepMath-5K dataset

![Image 5: Refer to caption](https://arxiv.org/html/2603.21877v3/x5.png)

Figure 5: Training Dynamics of GRPO on DeepScaler-5K dataset

## Appendix C GEPA Iteration Dynamics

Figure[6](https://arxiv.org/html/2603.21877#A3.F6 "Figure 6 ‣ Appendix C GEPA Iteration Dynamics ‣ Limitations. ‣ 6 Conclusion and Limitations ‣ 5 Related Works ‣ Impact of Group Prompt Diversity. ‣ 4.4 Ablations ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization") illustrates the internal optimization dynamics of GEPA, where the best template score exhibits a staircase-like ascending pattern: after two brief growth phases interspersed with plateaus, the score reaches its peak around iteration 8 and remains stable thereafter, indicating that GEPA reliably converges to a near-optimal prompt template within a limited number of iterations.

![Image 6: Refer to caption](https://arxiv.org/html/2603.21877v3/x6.png)

Figure 6: Iteration Curve of GEPA: Results of the Self-Ref variant on the DeepMath-10K dataset at epoch 1, showing the change of the best template score, the number of candidate templates, and the depth of the GEPA exploration tree during the GEPA iteration process.

## Appendix D Hard Sample Threshold \tau Ablation

Table[3](https://arxiv.org/html/2603.21877#A4.T3 "Table 3 ‣ Appendix D Hard Sample Threshold 𝜏 Ablation ‣ Limitations. ‣ 6 Conclusion and Limitations ‣ 5 Related Works ‣ Impact of Group Prompt Diversity. ‣ 4.4 Ablations ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization") presents the ablation results on the hard sample threshold \tau. Compared to \tau=1, which retains only fully incorrect samples, our default setting \tau=2 achieves consistently better performance across most benchmarks, with an average gain of 2.1 points. We attribute this to the fact that a stricter threshold yields too few samples for GEPA to iterate over, causing prompt optimization to converge to a suboptimal local optimum. Setting \tau=4, which includes samples where at least half of the rollouts are incorrect, performs comparably to \tau=2 (64.1 vs. 64.2 average), suggesting that our method is robust to moderate changes in this hyperparameter.

Table 3: Hard Sample Threshold \tau Ablation. Results on the DeepMath-5K dataset using the Self-Ref variant. In this work, we set \tau=2, treating samples with at most one correct as hard samples. For reference, \tau=1 retains only fully incorrect samples, while \tau=4 includes samples where at least half of the responses are incorrect.

## Appendix E Case Study

![Image 7: Refer to caption](https://arxiv.org/html/2603.21877v3/x7.png)

Figure 7: Qualitative Analysis: Overcoming Local Optima in Geometric Reasoning.

To further demonstrate how P 2 O overcomes the exploration bottleneck, we analyze a representative “hard sample” from the training process in Figure[7](https://arxiv.org/html/2603.21877#A5.F7 "Figure 7 ‣ Appendix E Case Study ‣ Limitations. ‣ 6 Conclusion and Limitations ‣ 5 Related Works ‣ Impact of Group Prompt Diversity. ‣ 4.4 Ablations ‣ 4 Experiments ‣ Pareto Selection. ‣ 3.4 Phase 2: Evolutionary Prompt Optimization ‣ 3 P2O: Joint Policy and Prompt Optimization ‣ P2O: Joint Policy and Prompt Optimization"). The problem asks for the radius of the smallest sphere enclosing four unit spheres.

#### Failure of the Base Policy.

Without guidance, the model falls into a common cognitive trap: it defaults to a visually intuitive but mathematically suboptimal configuration. As shown in the generated trace, the model attempts to place “3 spheres on a plane… with the 4th sphere resting on top.” This corresponds to a localized packing that yields a parent sphere radius of 1+\sqrt{2}, failing to minimize the volume.

#### Mechanism of the Optimized Prompt.

The template evolved by GEPA acts as a targeted intervention in the model’s latent search space. It does not merely encourage the model to “think harder”; rather, it injects specific domain knowledge—specifically the concept of the regular tetrahedron and the precise mathematical constant for its centroid-to-vertex distance (\sqrt{3/8}). This prompt effectively prunes the invalid search space (planar configurations) and steers the reasoning trajectory toward the global optimum (tetrahedral packing), allowing the model to derive the correct radius of 1+\frac{\sqrt{6}}{2}.

## Appendix F Broader Impacts

The proposed P 2 O framework positively impacts society by democratizing the development of highly capable reasoning models and reducing the reliance on expensive, human-curated expert data. Because our contribution focuses strictly on optimizing objective mathematical and logical reasoning capabilities, we do not foresee any direct negative societal impacts arising specifically from the P 2 O framework itself.

