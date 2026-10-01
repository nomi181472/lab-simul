Title: Evolutionary Reinforcement Learning via Cooperative Coevolution

URL Source: https://arxiv.org/html/2404.14763

Published Time: Fri, 02 Aug 2024 00:41:29 GMT

Markdown Content:
Jialin Liu Xin Yao Guangdong Provincial Key Laboratory of Brain-inspired Intelligent Computation, Department of Computer Science and Engineering, Southern University of Science and Technology, Shenzhen, China Research Institute of Trustworthy Autonomous System, Southern University of Science and Technology, 

Shenzhen, China Department of Computing and Decision Sciences, Lingnan University, Hong Kong SAR, China

###### Abstract

Recently, evolutionary reinforcement learning has obtained much attention in various domains. Maintaining a population of actors, evolutionary reinforcement learning utilises the collected experiences to improve the behaviour policy through efficient exploration. However, the poor scalability of genetic operators limits the efficiency of optimising high-dimensional neural networks. To address this issue, this paper proposes a novel cooperative coevolutionary reinforcement learning (CoERL) algorithm. Inspired by cooperative coevolution, CoERL periodically and adaptively decomposes the policy optimisation problem into multiple subproblems and evolves a population of neural networks for each of the subproblems. Instead of using genetic operators, CoERL directly searches for partial gradients to update the policy. Updating policy with partial gradients maintains consistency between the behaviour spaces of parents and offspring across generations. The experiences collected by the population are then used to improve the entire policy, which enhances the sampling efficiency. Experiments on six benchmark locomotion tasks demonstrate that CoERL outperforms seven state-of-the-art algorithms and baselines. Ablation study verifies the unique contribution of CoERL’s core ingredients.

\paperid

642

1 Introduction
--------------

Reinforcement learning (RL) has made adequate advancements in various domains such as video games[[19](https://arxiv.org/html/2404.14763v3#bib.bib19), [30](https://arxiv.org/html/2404.14763v3#bib.bib30)], Go[[26](https://arxiv.org/html/2404.14763v3#bib.bib26)] and robotic control[[9](https://arxiv.org/html/2404.14763v3#bib.bib9)] with a competitive level beyond human. Evolutionary reinforcement learning (ERL)[[10](https://arxiv.org/html/2404.14763v3#bib.bib10), [1](https://arxiv.org/html/2404.14763v3#bib.bib1)] combines evolutionary computation and RL to solve sequential decision-making problems, capitalising on the benefits of evolution in exploration. ERL maintains a population of parameterised actors, called individuals, which interact with the environments to collect experiences. Individuals with higher reward-based fitness are more likely to be selected as the parents, to which mutation and crossover operators are applied for reproduction. Another RL agent called _learner_, learns from diverse experiences sampled by the maintained population of actors and substitutes the individual with the worst fitness in the population periodically, i.e., injecting its gradient information for better convergence.

However, ERL introduces a scalability problem, attributed to the genetic operators used for reproducing new individuals[[40](https://arxiv.org/html/2404.14763v3#bib.bib40), [25](https://arxiv.org/html/2404.14763v3#bib.bib25), [13](https://arxiv.org/html/2404.14763v3#bib.bib13)]. Minor perturbations in the parameter space, caused by genetic operators, lead to significant divergences in behaviour space[[12](https://arxiv.org/html/2404.14763v3#bib.bib12), [13](https://arxiv.org/html/2404.14763v3#bib.bib13)]. Consequently, an offspring might not inherit its parents’ behaviours by merely exchanging fragments of parameters. Thus, this inconsistency between parameter space and behaviour space does not align with the principles of evolution[[39](https://arxiv.org/html/2404.14763v3#bib.bib39)].

Lehman et al. [[13](https://arxiv.org/html/2404.14763v3#bib.bib13)] proposed a safe mutation operator based on the sensitivity of neural networks’ outputs, which slightly adjusts the parameters of neural networks to ensure consistency. State-based crossover[[6](https://arxiv.org/html/2404.14763v3#bib.bib6)], as well as distillation crossover and proximal mutation operator[[2](https://arxiv.org/html/2404.14763v3#bib.bib2)], improve the traditional genetic operators using backpropagation. However, those operators are gradient-based, which introduce in-negligible extra computational costs. Nevertheless, the works of [[21](https://arxiv.org/html/2404.14763v3#bib.bib21)] and [[36](https://arxiv.org/html/2404.14763v3#bib.bib36)] directly searched for the parameters of neural networks using the cooperative coevolution, in which the policy optimisation problem is decomposed into multiple low-dimensional subproblems. However, the experiences of optimising subproblems are not fully utilised, resulting in inefficient training[[10](https://arxiv.org/html/2404.14763v3#bib.bib10)].

To address the scalability issue, we propose a novel cooperative coevolutionary reinforcement learning (CoERL) algorithm. The optimisation of a high-dimensional neural network is decomposed into multiple low-dimensional subproblems periodically and adaptively by cooperative coevolution. A population is resampled for each subproblem, in which each individual maintains the same neural network architecture. CoERL directly searches for partial gradients and updates the entire policy. The partial gradient searched via the population guides a proximal update on the subproblem, diminishing inconsistency between the behaviour spaces of parents and offspring. Experiences collected during evolution are then used to enhance RL for better sampling efficiency.

Our main contributions are summarised as follows:

*   •This paper proposes CoERL, a novel cooperative coevolutionary reinforcement learning algorithm to address the scalability issue of ERL, which also improves sample efficiency during training. 
*   •This paper proposes a partially updating strategy for policy improvement based on cooperative coevolution, which costs less and maintains the consistency between the behaviour spaces of parents and offspring across the evolution. 
*   •Experimental results on six benchmark locomotion tasks demonstrate the comparable performance of CoERL, compared with seven state-of-the-art algorithms and baselines. Ablation study shows the unique contributions of CoERL’s core ingredients. 

2 Background
------------

In this section, we introduce some preliminary, cooperative coevolution and recent progress of ERL.

### 2.1 Preliminary: Markov decision process

Markov decision process (MDP)[[27](https://arxiv.org/html/2404.14763v3#bib.bib27)] is defined as a tuple (𝒮,𝒜,ℛ,𝒫,γ)𝒮 𝒜 ℛ 𝒫 𝛾(\mathcal{S},\mathcal{A},\mathcal{R},\mathcal{P},\gamma)( caligraphic_S , caligraphic_A , caligraphic_R , caligraphic_P , italic_γ ), where 𝒮 𝒮\mathcal{S}caligraphic_S is the set of states, 𝒜 𝒜\mathcal{A}caligraphic_A is the set of actions, ℛ:𝒮×𝒜×𝒮↦ℝ:ℛ maps-to 𝒮 𝒜 𝒮 ℝ\mathcal{R}:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\mapsto\mathbb{R}caligraphic_R : caligraphic_S × caligraphic_A × caligraphic_S ↦ blackboard_R is the reward function, 𝒫:𝒮×𝒜×𝒮↦[0,1]:𝒫 maps-to 𝒮 𝒜 𝒮 0 1\mathcal{P}:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\mapsto[0,1]caligraphic_P : caligraphic_S × caligraphic_A × caligraphic_S ↦ [ 0 , 1 ] is the transition probability function, and γ 𝛾\gamma italic_γ is the discount factor. A policy π 𝜋\pi italic_π is a mapping from states to probability distributions for acting, where π⁢(a t|s t)𝜋 conditional subscript 𝑎 𝑡 subscript 𝑠 𝑡\pi(a_{t}|s_{t})italic_π ( italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT | italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) is the probability of taking action a t subscript 𝑎 𝑡 a_{t}italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT in state s t subscript 𝑠 𝑡 s_{t}italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT at time t 𝑡 t italic_t. The goal of the MDP is to optimise a parameterised policy π θ subscript 𝜋 𝜃\pi_{\theta}italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT that maximises the discounted cumulative reward, formulated as:

max θ⁡J⁢(θ)=𝔼 τ∼π θ⁢[∑t=0∞γ t⁢ℛ⁢(s t,a t,s t+1)],subscript 𝜃 𝐽 𝜃 subscript 𝔼 similar-to 𝜏 subscript 𝜋 𝜃 delimited-[]superscript subscript 𝑡 0 superscript 𝛾 𝑡 ℛ subscript 𝑠 𝑡 subscript 𝑎 𝑡 subscript 𝑠 𝑡 1\max_{\theta}~{}J(\theta)=\mathbb{E}_{\tau\sim\pi_{\theta}}[\sum_{t=0}^{\infty% }\gamma^{t}\mathcal{R}(s_{t},a_{t},s_{t+1})],roman_max start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT italic_J ( italic_θ ) = blackboard_E start_POSTSUBSCRIPT italic_τ ∼ italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT end_POSTSUBSCRIPT [ ∑ start_POSTSUBSCRIPT italic_t = 0 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ∞ end_POSTSUPERSCRIPT italic_γ start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT caligraphic_R ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_s start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT ) ] ,(1)

where s 0 subscript 𝑠 0 s_{0}italic_s start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT is an initial state. τ∼π θ similar-to 𝜏 subscript 𝜋 𝜃\tau\sim\pi_{\theta}italic_τ ∼ italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT denotes a trajectory (s 0,a 0,s 1,a 1,…,s t,a t,s t+1)subscript 𝑠 0 subscript 𝑎 0 subscript 𝑠 1 subscript 𝑎 1…subscript 𝑠 𝑡 subscript 𝑎 𝑡 subscript 𝑠 𝑡 1(s_{0},a_{0},s_{1},a_{1},\dots,s_{t},a_{t},s_{t+1})( italic_s start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT , italic_a start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT , italic_s start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_a start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , … , italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_s start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT ) sampled from π θ subscript 𝜋 𝜃\pi_{\theta}italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT.

### 2.2 Cooperative coevolution

Cooperative coevolution simulates the interactions of species living in an ecosphere[[17](https://arxiv.org/html/2404.14763v3#bib.bib17)]. Potter and De Jong [[23](https://arxiv.org/html/2404.14763v3#bib.bib23)] proposed the first cooperative coevolution (CC) algorithm, called cooperative coevolutionary genetic algorithm (CCGA). CCGA decomposes an optimisation problem into multiple subproblems. A population is initialised for each subproblem with a global context vector for fitness evaluations. For example, given an optimisation problem with n 𝑛 n italic_n variables, the fitness of a partial solution x i⁢j subscript 𝑥 𝑖 𝑗 x_{ij}italic_x start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT, i.e., the j 𝑗 j italic_j-th individual in the population of i 𝑖 i italic_i-th subproblem, can be evaluated by directly replacing the variable b j subscript 𝑏 𝑗 b_{j}italic_b start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT in the context vector 𝒃 𝒃\bm{b}bold_italic_b. Each variable in 𝒃 𝒃\bm{b}bold_italic_b is then updated with the best individual for the corresponding subproblem.

Problem decomposition referring to the grouping of variables is critical in CC[[17](https://arxiv.org/html/2404.14763v3#bib.bib17)]. Incorrect grouping leads to a local optimum, which is limited by the decomposed problem itself[[22](https://arxiv.org/html/2404.14763v3#bib.bib22), [35](https://arxiv.org/html/2404.14763v3#bib.bib35)]. Intuitively, static grouping and random grouping[[23](https://arxiv.org/html/2404.14763v3#bib.bib23)] are easy to implement and require no additional computational budget or knowledge. However, the effectiveness of the general strategies is unclear and requires some presets. Interaction-based grouping[[15](https://arxiv.org/html/2404.14763v3#bib.bib15)] and landscape-aware grouping[[34](https://arxiv.org/html/2404.14763v3#bib.bib34)] decompose the problem according to the characteristics of variables such as correlation and landscape. However, additional costs for fitness evaluations are required.

Another issue concerns the evaluation of individuals within a subproblem’s population, also known as the credit assignment problem, since individuals in a population represent only part of the entire solution. Collaboration for fitness evaluation with other subproblems’ populations is required. A global context vector can be constructed by collaborators (i.e., individuals) selected from populations of subproblems. Popular selection methods, such as single best collaborator[[23](https://arxiv.org/html/2404.14763v3#bib.bib23)], elite collaborator[[7](https://arxiv.org/html/2404.14763v3#bib.bib7)], and random collaborator selection, are typically chosen based on the specific domain[[17](https://arxiv.org/html/2404.14763v3#bib.bib17)].

Cooperative coevolution, served as a useful technique for solving high-dimensional black-box optimisation problems, is expected to benefit the policy optimisation. In this paper, we apply random grouping and single best collaboration for low cost and simplicity.

### 2.3 Evolutionary reinforcement learning

Evolutionary algorithms (EAs) have been successfully applied to solve RL problems, where neuroevolution and evolution-guided policy gradient are two main trends in this area[[20](https://arxiv.org/html/2404.14763v3#bib.bib20), [40](https://arxiv.org/html/2404.14763v3#bib.bib40), [33](https://arxiv.org/html/2404.14763v3#bib.bib33), [1](https://arxiv.org/html/2404.14763v3#bib.bib1)].

#### 2.3.1 Neuroevolution

Neuroevolution directly optimises the parameters or architectures of neural networks without considering the underlying mechanism like gradients, making it suitable for evolving RL policy and tackling real-world problems[[16](https://arxiv.org/html/2404.14763v3#bib.bib16), [4](https://arxiv.org/html/2404.14763v3#bib.bib4), [14](https://arxiv.org/html/2404.14763v3#bib.bib14), [5](https://arxiv.org/html/2404.14763v3#bib.bib5), [37](https://arxiv.org/html/2404.14763v3#bib.bib37)]. Fitness metrics formulated on rewards are used to select good individuals for reproducing the next generation by mutation and crossover. Hence, neuroevolution can intuitively handle complicated rewards such as non-convex and non-differentiable cases[[25](https://arxiv.org/html/2404.14763v3#bib.bib25), [4](https://arxiv.org/html/2404.14763v3#bib.bib4)]. Population-based training also encourages exploration. However, Nesterov and Spokoiny [[21](https://arxiv.org/html/2404.14763v3#bib.bib21)] indicated that EAs scale poorly with the increase of parameters.

Gong et al. [[8](https://arxiv.org/html/2404.14763v3#bib.bib8)] combined coevolution and backpropagation to solve classification tasks, while the framework is limited by traditional reproduction and specific domains. Yang et al. [[36](https://arxiv.org/html/2404.14763v3#bib.bib36)] proposed a CC framework based on negatively correlated search to tackle this issue, which directly searches for the parameters of a neural network. However, it only utilises the final accumulative reward as fitness, resulting in a waste of experiences collected during the evolution.

Besides, traditional genetic operators limit EA’s capability of addressing large-scale optimisation problems. The direct operations on the parameter space lead to an inconsistency between the behaviour spaces of parents and offspring, where parental behaviours may be typically forgotten[[13](https://arxiv.org/html/2404.14763v3#bib.bib13), [6](https://arxiv.org/html/2404.14763v3#bib.bib6), [2](https://arxiv.org/html/2404.14763v3#bib.bib2)]. Figure [1](https://arxiv.org/html/2404.14763v3#S2.F1 "Figure 1 ‣ 2.3.1 Neuroevolution ‣ 2.3 Evolutionary reinforcement learning ‣ 2 Background ‣ Evolutionary Reinforcement Learning via Cooperative Coevolution") shows an example of applying the one-point crossover, in which behaviour spaces of actors are decompressed by t-SNE[[29](https://arxiv.org/html/2404.14763v3#bib.bib29)]. It is easy to see from Figure [1](https://arxiv.org/html/2404.14763v3#S2.F1 "Figure 1 ‣ 2.3.1 Neuroevolution ‣ 2.3 Evolutionary reinforcement learning ‣ 2 Background ‣ Evolutionary Reinforcement Learning via Cooperative Coevolution") that directly exchanging the parameter fragments of the parents leads to a forgetting phenomenon in the behaviour space of the offspring.

![Image 1: Refer to caption](https://arxiv.org/html/2404.14763v3/extracted/5768630/figures/one-pointc2.png)

Figure 1: Inconsistency phenomenon between the behaviour spaces of parents and offspring by applying one-point crossover to exchange partially the parameters of Actor 1 (red) and Actor 2 (green). Subfigures show the feature maps of behaviour spaces decompressed by t-SNE[[29](https://arxiv.org/html/2404.14763v3#bib.bib29)]. 

#### 2.3.2 Evolution-guided policy gradient

![Image 2: Refer to caption](https://arxiv.org/html/2404.14763v3/extracted/5768630/figures/coerl2.png)

Figure 2: Diagram of CoERL. The CoERL algorithm begins by decomposing the policy optimisation problem parameterised by a neural network into multiple subproblems and searching for partial gradients to update the policy. Subsequently, the explored experiences are gathered to further refine the policy outcome using the MDP-based RL.

Evolution-guided policy gradient combines neuroevolution and MDP-based RL[[10](https://arxiv.org/html/2404.14763v3#bib.bib10)]. In the EA loop, actors, also called individuals, interact with the environments and are evolved by genetic operators. Experiences collected during the evaluations, i.e., transitions, are stored in a replay buffer and leveraged in the RL loop. Khadka et al. [[11](https://arxiv.org/html/2404.14763v3#bib.bib11)] proposed a portfolio of policies and dynamically allocated computational resources to train different policies, extending ERL. CEM-RL[[24](https://arxiv.org/html/2404.14763v3#bib.bib24)] leveraged cross-entropy method (CEM) to delayed deep deterministic policy gradient (TD3). Without using genetic operators, CEM-RL samples actors from the policy distribution estimated in previous generations. Gangwani and Peng [[6](https://arxiv.org/html/2404.14763v3#bib.bib6)] applied imitation learning to the crossover operator, which use trajectories of two parents to train an offspring. Bodnar et al. [[2](https://arxiv.org/html/2404.14763v3#bib.bib2)] improved the mutation and crossover by using local replay memories and the critic of the RL learner, namely proximal distilled evolutionary reinforcement learning (PDERL). New offsprings are produced by exchanging the local replay buffer or perturbation according to the sensitivity of the actions in PDERL[[2](https://arxiv.org/html/2404.14763v3#bib.bib2)]. Genetic algorithm has also been combined with RL in the work of[[18](https://arxiv.org/html/2404.14763v3#bib.bib18)], where individuals are generated by the noisy mutation only. To tackle the computationally expensive evaluation, Wang et al. [[31](https://arxiv.org/html/2404.14763v3#bib.bib31)] extended the surrogate model to ERL, which estimates the fitness of each individual based on the approximated value function. Zhu et al. [[41](https://arxiv.org/html/2404.14763v3#bib.bib41)] addressed the exploration and exploitation issue by maintaining actor and critic populations.

Although ERL inherits the advantages of neuroevolution, enabling it to address the temporal credit assignment problem with sparse rewards, it faces the challenge of poor scalability with EAs. Recent efforts have primarily concentrated on enhancing genetic operators, but at the expense of increased computational costs.

3  Cooperative coevolutionary reinforcement learning
----------------------------------------------------

Aiming at addressing the poor scalability of ERL, we propose a novel cooperative coevolutionary reinforcement learning (CoERL) algorithm 1 1 1 Code: https://github.com/HcPlu/CoERL. Pseudo-code and diagram of CoERL are provided in Algorithm [1](https://arxiv.org/html/2404.14763v3#alg1 "Algorithm 1 ‣ 3 Cooperative coevolutionary reinforcement learning ‣ Evolutionary Reinforcement Learning via Cooperative Coevolution") and Figure [2](https://arxiv.org/html/2404.14763v3#S2.F2 "Figure 2 ‣ 2.3.2 Evolution-guided policy gradient ‣ 2.3 Evolutionary reinforcement learning ‣ 2 Background ‣ Evolutionary Reinforcement Learning via Cooperative Coevolution"), respectively.

Algorithm 1 CoERL.

1:The number of generations

T 𝑇 T italic_T
, population size

μ 𝜇\mu italic_μ
, noise strength

σ 𝜎\sigma italic_σ
, number of subproblems

m 𝑚 m italic_m
, learning rate

α 𝛼\alpha italic_α
, temperature coefficient

α s subscript 𝛼 𝑠\alpha_{s}italic_α start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT

2:Policy

π θ subscript 𝜋 𝜃\pi_{\theta}italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT

3:Initialise policy

π θ subscript 𝜋 𝜃\pi_{\theta}italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT
, two critics

Q^ϕ 1 subscript^𝑄 subscript italic-ϕ 1\hat{Q}_{\phi_{1}}over^ start_ARG italic_Q end_ARG start_POSTSUBSCRIPT italic_ϕ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT end_POSTSUBSCRIPT
and

Q^ϕ 2 subscript^𝑄 subscript italic-ϕ 2\hat{Q}_{\phi_{2}}over^ start_ARG italic_Q end_ARG start_POSTSUBSCRIPT italic_ϕ start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT end_POSTSUBSCRIPT

4:Initialise reward buffer

ℬ ℛ subscript ℬ ℛ\mathcal{B}_{\mathcal{R}}caligraphic_B start_POSTSUBSCRIPT caligraphic_R end_POSTSUBSCRIPT

5:for

n=1 𝑛 1 n=1 italic_n = 1
to

T 𝑇 T italic_T
do

6:Decompose the policy optimisation problem into

m 𝑚 m italic_m

sub-problems by grouping

|θ|𝜃|\theta|| italic_θ |
-d parameters to

m 𝑚 m italic_m
sets

of parameter indices

ℐ 1,⋯,ℐ m subscript ℐ 1⋯subscript ℐ 𝑚\mathcal{I}_{1},\cdots,\mathcal{I}_{m}caligraphic_I start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , ⋯ , caligraphic_I start_POSTSUBSCRIPT italic_m end_POSTSUBSCRIPT

7:for

j=1 𝑗 1 j=1 italic_j = 1
to

m 𝑚 m italic_m
do

8:Sample

ϵ 1,⋯,ϵ μ∼𝒩|ℐ j|⁢(0,I)similar-to subscript italic-ϵ 1⋯subscript italic-ϵ 𝜇 superscript 𝒩 subscript ℐ 𝑗 0 𝐼\epsilon_{1},\cdots,\epsilon_{\mu}\sim\mathcal{N}^{|\mathcal{I}_{j}|}(0,I)italic_ϵ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , ⋯ , italic_ϵ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT ∼ caligraphic_N start_POSTSUPERSCRIPT | caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT | end_POSTSUPERSCRIPT ( 0 , italic_I )

9:Reproduce a population

{ψ 1,⋯,ψ μ}subscript 𝜓 1⋯subscript 𝜓 𝜇\{\psi_{1},\cdots,\psi_{\mu}\}{ italic_ψ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , ⋯ , italic_ψ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT }
for sub-

problem

ℐ j subscript ℐ 𝑗\mathcal{I}_{j}caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT
by

ψ i←θ⟨ℐ j⟩+ϵ i←subscript 𝜓 𝑖 superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 subscript italic-ϵ 𝑖\psi_{i}\leftarrow\theta^{\langle\mathcal{I}_{j}\rangle}+\epsilon_{i}italic_ψ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ← italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT + italic_ϵ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT
, for

i=1,⋯,μ 𝑖 1⋯𝜇 i=1,\cdots,\mu italic_i = 1 , ⋯ , italic_μ

▷▷\triangleright▷_θ⟨ℐ j⟩superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗\theta^{\langle\mathcal{I}\_{j}\rangle}italic\_θ start\_POSTSUPERSCRIPT ⟨ caligraphic\_I start\_POSTSUBSCRIPT italic\_j end\_POSTSUBSCRIPT ⟩ end\_POSTSUPERSCRIPT denotes θ 𝜃\theta italic\_θ’s elements at indices ℐ j subscript ℐ 𝑗\mathcal{I}\_{j}caligraphic\_I start\_POSTSUBSCRIPT italic\_j end\_POSTSUBSCRIPT_

10:for

i=1 𝑖 1 i=1 italic_i = 1
to

μ 𝜇\mu italic_μ
do

11:

J ℛ π ψ i,τ π ψ i=Evaluate⁢(π ψ i)subscript superscript 𝐽 subscript 𝜋 subscript 𝜓 𝑖 ℛ superscript 𝜏 subscript 𝜋 subscript 𝜓 𝑖 Evaluate subscript 𝜋 subscript 𝜓 𝑖 J^{\pi_{\psi_{i}}}_{\mathcal{R}},\tau^{\pi_{\psi_{i}}}=\text{Evaluate}(\pi_{% \psi_{i}})italic_J start_POSTSUPERSCRIPT italic_π start_POSTSUBSCRIPT italic_ψ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT end_POSTSUPERSCRIPT start_POSTSUBSCRIPT caligraphic_R end_POSTSUBSCRIPT , italic_τ start_POSTSUPERSCRIPT italic_π start_POSTSUBSCRIPT italic_ψ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT end_POSTSUPERSCRIPT = Evaluate ( italic_π start_POSTSUBSCRIPT italic_ψ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT )

12:Store

τ π ψ i superscript 𝜏 subscript 𝜋 subscript 𝜓 𝑖\tau^{\pi_{\psi_{i}}}italic_τ start_POSTSUPERSCRIPT italic_π start_POSTSUBSCRIPT italic_ψ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT end_POSTSUPERSCRIPT
in

ℬ ℛ subscript ℬ ℛ\mathcal{B}_{\mathcal{R}}caligraphic_B start_POSTSUBSCRIPT caligraphic_R end_POSTSUBSCRIPT

13:Assign fitness

f i=J ℛ π ψ i subscript 𝑓 𝑖 subscript superscript 𝐽 subscript 𝜋 subscript 𝜓 𝑖 ℛ f_{i}=J^{\pi_{\psi_{i}}}_{\mathcal{R}}italic_f start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = italic_J start_POSTSUPERSCRIPT italic_π start_POSTSUBSCRIPT italic_ψ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT end_POSTSUPERSCRIPT start_POSTSUBSCRIPT caligraphic_R end_POSTSUBSCRIPT

14:end for

15:

θ⟨ℐ j⟩←θ⟨ℐ j⟩+α⁢1 μ⁢σ⁢∑i=1 μ f i⁢ϵ i←superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 𝛼 1 𝜇 𝜎 subscript superscript 𝜇 𝑖 1 subscript 𝑓 𝑖 subscript italic-ϵ 𝑖\theta^{\langle\mathcal{I}_{j}\rangle}\leftarrow\theta^{\langle\mathcal{I}_{j}% \rangle}+\alpha\frac{1}{\mu\sigma}\sum^{\mu}_{i=1}f_{i}\epsilon_{i}italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT ← italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT + italic_α divide start_ARG 1 end_ARG start_ARG italic_μ italic_σ end_ARG ∑ start_POSTSUPERSCRIPT italic_μ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT italic_f start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT italic_ϵ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT
▷▷\triangleright▷_Eq. ([5](https://arxiv.org/html/2404.14763v3#S3.E5 "In 3.2 Partially updating via cooperative coevolution ‣ 3 Cooperative coevolutionary reinforcement learning ‣ Evolutionary Reinforcement Learning via Cooperative Coevolution"))_

▷▷\triangleright▷_Update parameters of policy π θ subscript 𝜋 𝜃\pi\_{\theta}italic\_π start\_POSTSUBSCRIPT italic\_θ end\_POSTSUBSCRIPT indexed by ℐ j subscript ℐ 𝑗\mathcal{I}\_{j}caligraphic\_I start\_POSTSUBSCRIPT italic\_j end\_POSTSUBSCRIPT_

16:end for

17:

▷▷\triangleright▷
_Policy gradient with ℬ ℛ subscript ℬ ℛ\mathcal{B}\_{\mathcal{R}}caligraphic\_B start\_POSTSUBSCRIPT caligraphic\_R end\_POSTSUBSCRIPT:_

18:Randomly sample a minibatch

ℬ ℬ\mathcal{B}caligraphic_B
of transitions

𝒯=⟨s,a,s′,r⟩𝒯 𝑠 𝑎 superscript 𝑠′𝑟\mathcal{T}=\langle s,a,s^{\prime},r\rangle caligraphic_T = ⟨ italic_s , italic_a , italic_s start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT , italic_r ⟩
from

ℬ R subscript ℬ 𝑅\mathcal{B}_{R}caligraphic_B start_POSTSUBSCRIPT italic_R end_POSTSUBSCRIPT

19:Compute target

y=r−γ⁢(min j=1,2⁡Q^ϕ j⁢(s′,a~′)−α s⁢log⁡π θ⁢(a~′|s′))𝑦 𝑟 𝛾 subscript 𝑗 1 2 subscript^𝑄 subscript italic-ϕ 𝑗 superscript 𝑠′superscript~𝑎′subscript 𝛼 𝑠 subscript 𝜋 𝜃 conditional superscript~𝑎′superscript 𝑠′y=r-\gamma(\min\limits_{j=1,2}\hat{Q}_{\phi_{j}}(s^{\prime},\tilde{a}^{\prime}% )-\alpha_{s}\log{\pi_{\theta}(\tilde{a}^{\prime}|s^{\prime})})italic_y = italic_r - italic_γ ( roman_min start_POSTSUBSCRIPT italic_j = 1 , 2 end_POSTSUBSCRIPT over^ start_ARG italic_Q end_ARG start_POSTSUBSCRIPT italic_ϕ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( italic_s start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT , over~ start_ARG italic_a end_ARG start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ) - italic_α start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT roman_log italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( over~ start_ARG italic_a end_ARG start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT | italic_s start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ) )
,

where

a~′∼π θ(⋅|s′)\tilde{a}^{\prime}\sim\pi_{\theta}(\cdot|s^{\prime})over~ start_ARG italic_a end_ARG start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ∼ italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( ⋅ | italic_s start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT )

20:Update critics with

∇ϕ j 1|B R|⁢∑𝒯∈ℬ(y−Q^ϕ j⁢(s,a))2 subscript∇subscript italic-ϕ 𝑗 1 subscript 𝐵 𝑅 subscript 𝒯 ℬ superscript 𝑦 subscript^𝑄 subscript italic-ϕ 𝑗 𝑠 𝑎 2\nabla_{\phi_{j}}\frac{1}{|B_{R}|}\sum\limits_{\mathcal{T}\in\mathcal{B}}(y-% \hat{Q}_{\phi_{j}}(s,a))^{2}∇ start_POSTSUBSCRIPT italic_ϕ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_POSTSUBSCRIPT divide start_ARG 1 end_ARG start_ARG | italic_B start_POSTSUBSCRIPT italic_R end_POSTSUBSCRIPT | end_ARG ∑ start_POSTSUBSCRIPT caligraphic_T ∈ caligraphic_B end_POSTSUBSCRIPT ( italic_y - over^ start_ARG italic_Q end_ARG start_POSTSUBSCRIPT italic_ϕ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( italic_s , italic_a ) ) start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT
for

j=1,2 𝑗 1 2 j=1,2 italic_j = 1 , 2

21:Update actor with

∇θ 1|B R|⁢∑𝒯∈ℬ(min j=1,2⁡Q^ϕ j⁢(s,a~θ)−α s⁢log⁡π θ⁢(a~θ|s))subscript∇𝜃 1 subscript 𝐵 𝑅 subscript 𝒯 ℬ subscript 𝑗 1 2 subscript^𝑄 subscript italic-ϕ 𝑗 𝑠 subscript~𝑎 𝜃 subscript 𝛼 𝑠 subscript 𝜋 𝜃 conditional subscript~𝑎 𝜃 𝑠\nabla_{\theta}\frac{1}{|B_{R}|}\sum\limits_{\mathcal{T}\in\mathcal{B}}(\min% \limits_{j=1,2}\hat{Q}_{\phi_{j}}(s,\tilde{a}_{\theta})-\alpha_{s}\log\pi_{% \theta}(\tilde{a}_{\theta}|s))∇ start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT divide start_ARG 1 end_ARG start_ARG | italic_B start_POSTSUBSCRIPT italic_R end_POSTSUBSCRIPT | end_ARG ∑ start_POSTSUBSCRIPT caligraphic_T ∈ caligraphic_B end_POSTSUBSCRIPT ( roman_min start_POSTSUBSCRIPT italic_j = 1 , 2 end_POSTSUBSCRIPT over^ start_ARG italic_Q end_ARG start_POSTSUBSCRIPT italic_ϕ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( italic_s , over~ start_ARG italic_a end_ARG start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ) - italic_α start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT roman_log italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( over~ start_ARG italic_a end_ARG start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT | italic_s ) )
,

where

a~θ subscript~𝑎 𝜃\tilde{a}_{\theta}over~ start_ARG italic_a end_ARG start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT
is sampled from

π θ(⋅|s)\pi_{\theta}(\cdot|s)italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( ⋅ | italic_s )
via the reparame-

terisation trick

22:end for

### 3.1 Collaboration between cooperative coevolution and reinforcement learning

CoERL maintains collaborative loops between CC and RL. In the CC loop, a neural network (a policy) is converted into a position-fixed vector with real numbers. The policy optimisation problem, is divided into multiple subproblems, in which parameters of the neural network are grouped. The decomposed subproblems have no intersection, while their union collectively constitute the entire parameter space. Given a policy π θ subscript 𝜋 𝜃\pi_{\theta}italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT parameterised by θ 𝜃\theta italic_θ, it ensures that θ⟨ℐ i⟩∩θ⟨ℐ j⟩=∅,∀i≠j,1≤i≤m,1≤j≤m formulae-sequence formulae-sequence superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑖 superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 formulae-sequence for-all 𝑖 𝑗 1 𝑖 𝑚 1 𝑗 𝑚\theta^{\langle\mathcal{I}_{i}\rangle}\cap\theta^{\langle\mathcal{I}_{j}% \rangle}=\varnothing,\forall i\neq j,1\leq i\leq m,1\leq j\leq m italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT ∩ italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT = ∅ , ∀ italic_i ≠ italic_j , 1 ≤ italic_i ≤ italic_m , 1 ≤ italic_j ≤ italic_m and θ⟨ℐ 1⟩∪θ⟨ℐ 2⟩∪⋯∪θ⟨ℐ m⟩=θ superscript 𝜃 delimited-⟨⟩subscript ℐ 1 superscript 𝜃 delimited-⟨⟩subscript ℐ 2⋯superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑚 𝜃\theta^{\langle\mathcal{I}_{1}\rangle}\cup\theta^{\langle\mathcal{I}_{2}% \rangle}\cup\cdots\cup\theta^{\langle\mathcal{I}_{m}\rangle}=\theta italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT ∪ italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT ∪ ⋯ ∪ italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_m end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT = italic_θ, where m 𝑚 m italic_m is the number of subproblems and ⟨ℐ i⟩delimited-⟨⟩subscript ℐ 𝑖\langle\mathcal{I}_{i}\rangle⟨ caligraphic_I start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ⟩ denotes the parameter indices of subproblem j 𝑗 j italic_j. To optimise a subproblem, a population is maintained and generated using perturbation methods instead of traditional genetic operators. Practically, each individual in the population shares the same neural network structure with the freezing quotient set of the subproblem. Only the variables in the subproblem, which form part of the neural network, are perturbed by the noises from the distribution.

Individuals consistently interact with the environment. The cumulative reward obtained by the individual is treated as its fitness. Then, noised individuals and their corresponding fitness are used to update the entire policy. Notably, subproblems are not optimised separately. The policy optimised according to the preceding subproblem is regarded as the complementary base of the subsequent subproblem. Thus, connections between different subproblems are built.

After updating the policy within each subproblem, CoERL proceeds to optimise the policy via MDP-based RL. Experiences such as transitions produced by the individuals during the cooperative coevolution loop are collected in a replay buffer. Batches sampled from the replay buffer are then used to update the policy via policy gradient, which fully utilises the temporal information. Sections[3.2](https://arxiv.org/html/2404.14763v3#S3.SS2 "3.2 Partially updating via cooperative coevolution ‣ 3 Cooperative coevolutionary reinforcement learning ‣ Evolutionary Reinforcement Learning via Cooperative Coevolution") and[3.3](https://arxiv.org/html/2404.14763v3#S3.SS3 "3.3 Leveraging temporal information ‣ 3 Cooperative coevolutionary reinforcement learning ‣ Evolutionary Reinforcement Learning via Cooperative Coevolution") detail the two loops, respectively.

### 3.2 Partially updating via cooperative coevolution

Given a policy π θ subscript 𝜋 𝜃\pi_{\theta}italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT parameterised by θ 𝜃\theta italic_θ, CoERL decomposes the policy optimisation problem into m 𝑚 m italic_m subproblems in the CC loop. The subproblem j 𝑗 j italic_j of policy optimisation, θ⟨ℐ j⟩superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗\theta^{\langle\mathcal{I}_{j}\rangle}italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT indexed by parameters indices ⟨ℐ j⟩delimited-⟨⟩subscript ℐ 𝑗\langle\mathcal{I}_{j}\rangle⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩, and the quotient θ^⟨ℐ j⟩=θ/θ⟨ℐ j⟩superscript^𝜃 delimited-⟨⟩subscript ℐ 𝑗 𝜃 superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗\hat{\theta}^{\langle\mathcal{I}_{j}\rangle}=\theta/\theta^{\langle\mathcal{I}% _{j}\rangle}over^ start_ARG italic_θ end_ARG start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT = italic_θ / italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT constitute the entire parameter space, i.e., θ=[θ⟨ℐ j⟩:θ^⟨ℐ j⟩]\theta=\left[\theta^{\langle\mathcal{I}_{j}\rangle}:\hat{\theta}^{\langle% \mathcal{I}_{j}\rangle}\right]italic_θ = [ italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT : over^ start_ARG italic_θ end_ARG start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT ]. Then, for each subproblem j 𝑗 j italic_j, a corresponding actor π θ⟨ℐ j⟩subscript 𝜋 superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗\pi_{\theta^{\langle\mathcal{I}_{j}\rangle}}italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT end_POSTSUBSCRIPT is constructed by freezing the quotient part. CoERL partially updates the policy by iteratively optimising each subproblem. A population is first sampled via a specific distribution P⟨ℐ j⟩superscript 𝑃 delimited-⟨⟩subscript ℐ 𝑗 P^{\langle\mathcal{I}_{j}\rangle}italic_P start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT. Each individual is an actor π θ⟨ℐ j⟩subscript 𝜋 superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗\pi_{\theta^{\langle\mathcal{I}_{j}\rangle}}italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT end_POSTSUBSCRIPT, where θ⟨ℐ j⟩∼P⟨ℐ j⟩similar-to superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 superscript 𝑃 delimited-⟨⟩subscript ℐ 𝑗\theta^{\langle\mathcal{I}_{j}\rangle}\sim P^{\langle\mathcal{I}_{j}\rangle}italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT ∼ italic_P start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT. All individuals in the population of the subproblem share the same quotient part θ^⟨ℐ j⟩superscript^𝜃 delimited-⟨⟩subscript ℐ 𝑗\hat{\theta}^{\langle\mathcal{I}_{j}\rangle}over^ start_ARG italic_θ end_ARG start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT. Then, the fitness of each individual f⁢(π θ⟨ℐ j⟩)𝑓 subscript 𝜋 superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 f(\pi_{\theta^{\langle\mathcal{I}_{j}\rangle}})italic_f ( italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT end_POSTSUBSCRIPT ) is determined by the cumulative reward. The expected fitness under an arbitrary distribution is formulated as Eq. ([2](https://arxiv.org/html/2404.14763v3#S3.E2 "In 3.2 Partially updating via cooperative coevolution ‣ 3 Cooperative coevolutionary reinforcement learning ‣ Evolutionary Reinforcement Learning via Cooperative Coevolution")):

J⁢(θ⟨ℐ j⟩)=𝔼 θ⟨ℐ j⟩⁢[f⁢(π θ⟨ℐ j⟩)]=∫f⁢(π θ⟨ℐ j⟩)⁢p⁢(θ⟨ℐ j⟩)⁢𝑑 θ⟨ℐ j⟩,𝐽 superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 subscript 𝔼 superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 delimited-[]𝑓 subscript 𝜋 superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 𝑓 subscript 𝜋 superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 𝑝 superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 differential-d superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 J(\theta^{\langle\mathcal{I}_{j}\rangle})=\mathbb{E}_{\theta^{\langle\mathcal{% I}_{j}\rangle}}[f(\pi_{\theta^{\langle\mathcal{I}_{j}\rangle}})]\\ =\int f(\pi_{\theta^{\langle\mathcal{I}_{j}\rangle}})p(\theta^{\langle\mathcal% {I}_{j}\rangle})d\theta^{\langle\mathcal{I}_{j}\rangle},italic_J ( italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT ) = blackboard_E start_POSTSUBSCRIPT italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT end_POSTSUBSCRIPT [ italic_f ( italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT end_POSTSUBSCRIPT ) ] = ∫ italic_f ( italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT end_POSTSUBSCRIPT ) italic_p ( italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT ) italic_d italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT ,(2)

where p⁢(θ⟨ℐ j⟩)𝑝 superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 p(\theta^{\langle\mathcal{I}_{j}\rangle})italic_p ( italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT ) denotes the density. Then, we can write the gradient form using the “log-likelihood trick". The estimation of the gradient by maintaining a population with size μ 𝜇\mu italic_μ is shown as follows:

∇θ⟨ℐ j⟩J⁢(θ⟨ℐ j⟩)=𝔼 θ⟨ℐ j⟩⁢[f⁢(π θ⟨ℐ j⟩)⁢∇θ log⁡(p⁢(θ⟨ℐ j⟩))].subscript∇superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 𝐽 superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 subscript 𝔼 superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 delimited-[]𝑓 subscript 𝜋 superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 subscript∇𝜃 𝑝 superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗\nabla_{\theta^{\langle\mathcal{I}_{j}\rangle}}J(\theta^{\langle\mathcal{I}_{j% }\rangle})=\mathbb{E}_{\theta^{\langle\mathcal{I}_{j}\rangle}}\left[f(\pi_{% \theta^{\langle\mathcal{I}_{j}\rangle}})\nabla_{\theta}\log(p(\theta^{\langle% \mathcal{I}_{j}\rangle}))\right].∇ start_POSTSUBSCRIPT italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT end_POSTSUBSCRIPT italic_J ( italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT ) = blackboard_E start_POSTSUBSCRIPT italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT end_POSTSUBSCRIPT [ italic_f ( italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT end_POSTSUBSCRIPT ) ∇ start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT roman_log ( italic_p ( italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT ) ) ] .(3)

In the case of factored Gaussian distribution with the deviation σ 𝜎\sigma italic_σ, we can set θ⟨ℐ j⟩superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗\theta^{\langle\mathcal{I}_{j}\rangle}italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT as the mean vector. Then, the expected fitness of Eq.([2](https://arxiv.org/html/2404.14763v3#S3.E2 "In 3.2 Partially updating via cooperative coevolution ‣ 3 Cooperative coevolutionary reinforcement learning ‣ Evolutionary Reinforcement Learning via Cooperative Coevolution")) is rewritten as follows:

𝔼 θ⟨ℐ j⟩⁢[f⁢(π θ⟨ℐ j⟩)]=𝔼 ϵ∼𝒩⟨ℐ j⟩⁢(0,I)⁢[f⁢(π θ⟨ℐ j⟩+σ⁢ϵ)].subscript 𝔼 superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 delimited-[]𝑓 subscript 𝜋 superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 subscript 𝔼 similar-to italic-ϵ superscript 𝒩 delimited-⟨⟩subscript ℐ 𝑗 0 𝐼 delimited-[]𝑓 subscript 𝜋 superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 𝜎 italic-ϵ\mathbb{E}_{\theta^{\langle\mathcal{I}_{j}\rangle}}[f(\pi_{\theta^{\langle% \mathcal{I}_{j}\rangle}})]=\mathbb{E}_{\epsilon\sim\mathcal{N}^{\langle% \mathcal{I}_{j}\rangle}(0,I)}[f(\pi_{\theta^{\langle\mathcal{I}_{j}\rangle}+% \sigma\epsilon})].blackboard_E start_POSTSUBSCRIPT italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT end_POSTSUBSCRIPT [ italic_f ( italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT end_POSTSUBSCRIPT ) ] = blackboard_E start_POSTSUBSCRIPT italic_ϵ ∼ caligraphic_N start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT ( 0 , italic_I ) end_POSTSUBSCRIPT [ italic_f ( italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT + italic_σ italic_ϵ end_POSTSUBSCRIPT ) ] .(4)

Practically, each individual, i.e., π ψ i subscript 𝜋 subscript 𝜓 𝑖\pi_{\psi_{i}}italic_π start_POSTSUBSCRIPT italic_ψ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT, the actor in the population is produced by a perturbation operation ψ i=θ⟨ℐ j⟩+ϵ i subscript 𝜓 𝑖 superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 subscript italic-ϵ 𝑖\psi_{i}=\theta^{\langle\mathcal{I}_{j}\rangle}+\epsilon_{i}italic_ψ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT + italic_ϵ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT, where ϵ i∼𝒩⟨ℐ j⟩⁢(0,I)similar-to subscript italic-ϵ 𝑖 superscript 𝒩 delimited-⟨⟩subscript ℐ 𝑗 0 𝐼\epsilon_{i}\sim\mathcal{N}^{\langle\mathcal{I}_{j}\rangle}(0,I)italic_ϵ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∼ caligraphic_N start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT ( 0 , italic_I ). The average estimated gradient in this case according to Eq.([4](https://arxiv.org/html/2404.14763v3#S3.E4 "In 3.2 Partially updating via cooperative coevolution ‣ 3 Cooperative coevolutionary reinforcement learning ‣ Evolutionary Reinforcement Learning via Cooperative Coevolution")) is shown as follows:

∇θ⟨ℐ j⟩𝔼 θ⟨ℐ j⟩⁢[f⁢(π θ⟨ℐ j⟩)]≅1 μ⁢σ⁢∑i=1 μ[f⁢(π θ⟨ℐ j⟩+σ⁢ϵ i)⁢ϵ i],subscript∇superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 subscript 𝔼 superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 delimited-[]𝑓 subscript 𝜋 superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 1 𝜇 𝜎 subscript superscript 𝜇 𝑖 1 delimited-[]𝑓 subscript 𝜋 superscript 𝜃 delimited-⟨⟩subscript ℐ 𝑗 𝜎 subscript italic-ϵ 𝑖 subscript italic-ϵ 𝑖\nabla_{\theta^{\langle\mathcal{I}_{j}\rangle}}\mathbb{E}_{\theta^{\langle% \mathcal{I}_{j}\rangle}}[f(\pi_{\theta^{\langle\mathcal{I}_{j}\rangle}})]\cong% \frac{1}{\mu\sigma}\sum^{\mu}_{i=1}[f(\pi_{\theta^{\langle\mathcal{I}_{j}% \rangle}+\sigma\epsilon_{i}})\epsilon_{i}],∇ start_POSTSUBSCRIPT italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT end_POSTSUBSCRIPT blackboard_E start_POSTSUBSCRIPT italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT end_POSTSUBSCRIPT [ italic_f ( italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT end_POSTSUBSCRIPT ) ] ≅ divide start_ARG 1 end_ARG start_ARG italic_μ italic_σ end_ARG ∑ start_POSTSUPERSCRIPT italic_μ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT [ italic_f ( italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUPERSCRIPT ⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩ end_POSTSUPERSCRIPT + italic_σ italic_ϵ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT ) italic_ϵ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ] ,(5)

where μ 𝜇\mu italic_μ is the population size. The estimated partial gradient enables searching for a good optimisation direction for each subproblem. Instead of independently optimising each subproblem, we introduce the concept of coevolution by the divide-and-conquer strategy, in which each subproblem is optimised iteratively. Using Figure [3](https://arxiv.org/html/2404.14763v3#S3.F3 "Figure 3 ‣ 3.2 Partially updating via cooperative coevolution ‣ 3 Cooperative coevolutionary reinforcement learning ‣ Evolutionary Reinforcement Learning via Cooperative Coevolution") as an example, a neural network optimisation problem (weights) is decomposed into 3 subproblems (3 subsets), highlighted by red, green and blue, respectively. Each subproblem aims at optimising a subset of the weights. First, weights in red will be updated, while those in black will be frozen. Then, weights in green will be updated, while both weights in red and black will be frozen. Similar steps proceed until all weights are updated once. Optimising subproblems is not independent but in a cascade way. Optimisation of the current subproblem relies on the previous subproblem.

The idea of coevolution builds the bridge between subproblems. The estimated gradient of the preceding subproblem can be regarded as momentum, which facilitates the subsequent optimisation. Additionally, the time complexity of the partially updating is 𝒪⁢(|θ|)𝒪 𝜃\mathcal{O}(|\theta|)caligraphic_O ( | italic_θ | ) with linearly scaling the number of parameters. This complexity is easy to get since each subproblem is the complimentary set of other subproblems. This complexity is much smaller than that of current gradient-based operators such as distilled crossover[[2](https://arxiv.org/html/2404.14763v3#bib.bib2)], which requires backpropagation with batches.

![Image 3: Refer to caption](https://arxiv.org/html/2404.14763v3/extracted/5768630/figures/partial_gradients.png)

Figure 3: An example of partial updating via CC. Three subproblems are highlighted in red, green, and blue, respectively. For each subproblem, the policy inherits the partial gradient from the previous subproblem. Eventually, all parameters are updated once and only once.

### 3.3 Leveraging temporal information

To fully utilise the collected experiences, we choose soft actor-critic (SAC)[[9](https://arxiv.org/html/2404.14763v3#bib.bib9)] as the base algorithm in the RL loop, which optimises the policy in an off-policy way. Besides, the actor-critic architecture allows us to train the actor and critic separately. Hence, a population can be directly generated from an actor. The optimised actor after the CC loop is then used in policy improvement directly. SAC changes the objective of RL by adding the entropy term. The entropy of the policy is regarded as an extra rewarding signal. Larger entropy indicates a greater tendency for exploration. The Q 𝑄 Q italic_Q function is parameterised by ϕ italic-ϕ\phi italic_ϕ, and the target Q 𝑄 Q italic_Q value using reparameterisation a~′∼π θ(⋅|s′)\tilde{a}^{\prime}\sim\pi_{\theta}(\cdot|s^{\prime})over~ start_ARG italic_a end_ARG start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ∼ italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( ⋅ | italic_s start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ) is shown as follows:

y⁢(r,s′)=r+γ⁢(min j=1,2⁡Q ϕ j⁢(s′,a~′)−α s⁢log⁡π θ⁢(a~′|s′)),𝑦 𝑟 superscript 𝑠′𝑟 𝛾 subscript 𝑗 1 2 subscript 𝑄 subscript italic-ϕ 𝑗 superscript 𝑠′superscript~𝑎′subscript 𝛼 𝑠 subscript 𝜋 𝜃 conditional superscript~𝑎′superscript 𝑠′y(r,s^{\prime})=r+\gamma(\min_{j=1,2}Q_{\phi_{j}}(s^{\prime},\tilde{a}^{\prime% })-\alpha_{s}\log{\pi_{\theta}(\tilde{a}^{\prime}|s^{\prime})}),italic_y ( italic_r , italic_s start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ) = italic_r + italic_γ ( roman_min start_POSTSUBSCRIPT italic_j = 1 , 2 end_POSTSUBSCRIPT italic_Q start_POSTSUBSCRIPT italic_ϕ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( italic_s start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT , over~ start_ARG italic_a end_ARG start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ) - italic_α start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT roman_log italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( over~ start_ARG italic_a end_ARG start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT | italic_s start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ) ) ,(6)

where α s subscript 𝛼 𝑠\alpha_{s}italic_α start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT is the temperature coefficient. If s′superscript 𝑠′s^{\prime}italic_s start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT is the end of the episode, then y⁢(r,s′)=r 𝑦 𝑟 superscript 𝑠′𝑟 y(r,s^{\prime})=r italic_y ( italic_r , italic_s start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ) = italic_r. The loss function of the Q-network is shown as follows:

L⁢(ϕ i)=𝔼(s,a,r,s′)[(Q ϕ i⁢(s,a)−y⁢(r,s′))2].𝐿 subscript italic-ϕ 𝑖 subscript 𝔼 𝑠 𝑎 𝑟 superscript 𝑠′delimited-[]superscript subscript 𝑄 subscript italic-ϕ 𝑖 𝑠 𝑎 𝑦 𝑟 superscript 𝑠′2 L(\phi_{i})=\mathop{\mathbb{E}}\limits_{(s,a,r,s^{\prime})}\left[(Q_{\phi_{i}}% (s,a)-y(r,s^{\prime}))^{2}\right].italic_L ( italic_ϕ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) = blackboard_E start_POSTSUBSCRIPT ( italic_s , italic_a , italic_r , italic_s start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ) end_POSTSUBSCRIPT [ ( italic_Q start_POSTSUBSCRIPT italic_ϕ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( italic_s , italic_a ) - italic_y ( italic_r , italic_s start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ) ) start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ] .(7)

According to the additional entropy term, the objective of SAC is

max θ⁡J⁢(θ)=𝔼[min j=1,2⁡Q ϕ j⁢(s,a~θ)−α s⁢log⁡π θ⁢(a~θ|s)],subscript 𝜃 𝐽 𝜃 𝔼 delimited-[]subscript 𝑗 1 2 subscript 𝑄 subscript italic-ϕ 𝑗 𝑠 subscript~𝑎 𝜃 subscript 𝛼 𝑠 subscript 𝜋 𝜃 conditional subscript~𝑎 𝜃 𝑠\max_{\theta}~{}J(\theta)=\mathop{\mathbb{E}}\left[\min_{j=1,2}Q_{\phi_{j}}(s,% \tilde{a}_{\theta})-\alpha_{s}\log{\pi_{\theta}(\tilde{a}_{\theta}|s)}\right],roman_max start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT italic_J ( italic_θ ) = blackboard_E [ roman_min start_POSTSUBSCRIPT italic_j = 1 , 2 end_POSTSUBSCRIPT italic_Q start_POSTSUBSCRIPT italic_ϕ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( italic_s , over~ start_ARG italic_a end_ARG start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ) - italic_α start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT roman_log italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( over~ start_ARG italic_a end_ARG start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT | italic_s ) ] ,(8)

where a~θ subscript~𝑎 𝜃\tilde{a}_{\theta}over~ start_ARG italic_a end_ARG start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT is sampled via the reparameterisation trick.

4 Experiments
-------------

![Image 4: Refer to caption](https://arxiv.org/html/2404.14763v3/extracted/5768630/figures/training/HalfCheetah-v2_2.png)

![Image 5: Refer to caption](https://arxiv.org/html/2404.14763v3/extracted/5768630/figures/training/Humanoid-v2_2.png)

![Image 6: Refer to caption](https://arxiv.org/html/2404.14763v3/extracted/5768630/figures/training/Hopper-v2_2.png)

![Image 7: Refer to caption](https://arxiv.org/html/2404.14763v3/extracted/5768630/figures/training/Ant-v2_2.png)

![Image 8: Refer to caption](https://arxiv.org/html/2404.14763v3/extracted/5768630/figures/training/Walker2d-v2_2.png)

![Image 9: Refer to caption](https://arxiv.org/html/2404.14763v3/extracted/5768630/figures/training/Swimmer-v2_2.png)

![Image 10: Refer to caption](https://arxiv.org/html/2404.14763v3/extracted/5768630/figures/training/legends.png)

Figure 4: Training curves on six locomotion tasks. Each algorithm is trained for 1e6 timesteps with five different random seeds.

CoERL is compared with four advanced RL and ERL algorithms, namely SAC[[9](https://arxiv.org/html/2404.14763v3#bib.bib9)], ERL[[10](https://arxiv.org/html/2404.14763v3#bib.bib10)], PDERL[[2](https://arxiv.org/html/2404.14763v3#bib.bib2)], and CCNCS[[36](https://arxiv.org/html/2404.14763v3#bib.bib36)] on six benchmark locomotion tasks, including Ant-v2, HalfCheetah-v2, Humanoid-v2, Hopper-v2, Walker2d-v2, and Swimmer-v2 from Mujoco[[28](https://arxiv.org/html/2404.14763v3#bib.bib28)] integrated in the OpenAI gym[[3](https://arxiv.org/html/2404.14763v3#bib.bib3)]. Additionally, CoERL is split into three independent methods for conducting ablation study, namely cooperative coevolutionary evolution strategies (CoES), evolution strategies with soft actor-critic (ESSAC), and evolution strategies (ES)[[25](https://arxiv.org/html/2404.14763v3#bib.bib25)].

### 4.1 Settings

Table 1: Hyperparameters for all compared algorithms. “-" denotes that the parameter is not involved.

CoERL is implemented with the Tianshou framework[[32](https://arxiv.org/html/2404.14763v3#bib.bib32)]. The network structure is a fully connected neural network with a ⟨256,256⟩256 256\langle 256,256\rangle⟨ 256 , 256 ⟩ linear layer. γ 𝛾\gamma italic_γ is 0.99. In this paper, we apply random grouping[[38](https://arxiv.org/html/2404.14763v3#bib.bib38), [36](https://arxiv.org/html/2404.14763v3#bib.bib36)] and single best collaboration[[23](https://arxiv.org/html/2404.14763v3#bib.bib23)] for low cost and simplicity. The grouping number at each generation is randomly chosen from two, three and four, as Yang et al. [[36](https://arxiv.org/html/2404.14763v3#bib.bib36)] suggested. Hyperparameters of SAC, ERL, PDERL and CCNCS follow their original settings in [[9](https://arxiv.org/html/2404.14763v3#bib.bib9), [10](https://arxiv.org/html/2404.14763v3#bib.bib10), [2](https://arxiv.org/html/2404.14763v3#bib.bib2), [36](https://arxiv.org/html/2404.14763v3#bib.bib36)], respectively. Main hyperparameters are shown in Table[1](https://arxiv.org/html/2404.14763v3#S4.T1 "Table 1 ‣ 4.1 Settings ‣ 4 Experiments ‣ Evolutionary Reinforcement Learning via Cooperative Coevolution"). All algorithms are trained by 1e6 timesteps with five different random seeds. CoERL’s hyperparameters are arbitrarily set, and remain the same values for all tasks.

### 4.2 Comparison results

Figure [4](https://arxiv.org/html/2404.14763v3#S4.F4 "Figure 4 ‣ 4 Experiments ‣ Evolutionary Reinforcement Learning via Cooperative Coevolution") shows the training curves assembled with five different random seeds. Our CoERL, highlighted with red colour, presents comparable performance and superior convergence on _HalfCheetah-v2_, _Hopper-v2_, _Ant-v2_ and _Walker2d-v2_. Table [2](https://arxiv.org/html/2404.14763v3#S4.T2 "Table 2 ‣ 4.2 Comparison results ‣ 4 Experiments ‣ Evolutionary Reinforcement Learning via Cooperative Coevolution") presents the final reward performance of CoERL and all comprised algorithms across six locomotion tasks. CoERL achieves the best average rewards on four tasks including _HalfCheetah-v2_, _Hopper-v2_, _Ant-v2_ and _Walker2d-v2_. For example, in _Ant-v2_, CoERL obtains the best average reward of 5037.22, while the second-best algorithm SAC only gets 3654.16. Besides, CoERL achieves the highest average rank of 1.67, considering all six tasks. CoERL decomposes the neural network and searches for the partial gradient for each subproblem. Each parameter of the entire neural network is updated only once within an iteration, with a complexity of 𝒪⁢(|θ|)𝒪 𝜃\mathcal{O}(|\theta|)caligraphic_O ( | italic_θ | ). The partial gradient estimation does not rely on the delicate calculation of batches from the dataset but only requires the final outcome of evaluations, which can be easily sped up on CPUs according to Eq. ([5](https://arxiv.org/html/2404.14763v3#S3.E5 "In 3.2 Partially updating via cooperative coevolution ‣ 3 Cooperative coevolutionary reinforcement learning ‣ Evolutionary Reinforcement Learning via Cooperative Coevolution")).

Table 2: Final reward performances with five different seeds in six locomotion environments. The bold number denotes the highest average value of each column. “Avg. Rank” denotes the average rank across all six tasks. Lower rank indicates better overall performance. CoERL shows the best performance with the highest average rank 1.67.

We also notice that CoERL does not achieve outstanding performance in _Swimmer-v2_, although its performance has been improved from 55.89 to 94.87 and 128.90 with the help of CC and the utility of temporal information by ESSAC. The poor performance of SAC might explain this case, as it only achieves 37.95, surpassing only the 15.91 achieved by ES. Given that CoERL is built upon SAC, it is not surprising that CoERL encounters the same local optimum as SAC, despite ultimately reaching 128.90. PDERL, the improved version of ERL only get 765.55 in _Humanoid-v2_, while our CoERL gets an average reward of 4642.05 and ERL gets 4677.19. It is attributed to the high dimensions of _Humanoid-v2_ for the peculiar performance of PDERL. The observation dimension of _Humanoid-v2_ is 376, which is the largest among six tasks. The distilled crossover maintains the consistency of behaviour spaces while exchanging parameters using supervised learning techniques. However, this improvement leads to higher computational demands during backpropagation. Our CoERL, in contrast, searches for the partial gradient at a lower computational cost than backpropagation, while still preserving consistency.

Moreover, none of the pure evolution-based algorithms, such as ES[[25](https://arxiv.org/html/2404.14763v3#bib.bib25)] and CCNCS[[36](https://arxiv.org/html/2404.14763v3#bib.bib36)], demonstrates overall promising performance, even though CoES and CCNCS achieve superior average rewards than SAC in _Swimmer-v2_ with 55.89 and 47.71, respectively. As discussed previously, evolution-based algorithms have long struggled with the scalability issue[[17](https://arxiv.org/html/2404.14763v3#bib.bib17)]. And the temporal experiences are barely utilised in evolution, resulting in inefficient training. Therefore, more computational time is required to converge. Our CoERL overcomes those issues and achieves the highest rank with 1.67.

### 4.3 Ablation study

To fully verify the effectiveness of CoERL, we examine the contribution of its components independently. The evolution part is split into coevolution strategies (CoES) and evolution strategies (ES). Regarding the RL part, we consider pure RL algorithm, SAC, and the simplified version without coevolution, ESSAC. The performance of CoERL and its ingredients, including CoES, ES, SAC, and ESSAC, are shown in Table [2](https://arxiv.org/html/2404.14763v3#S4.T2 "Table 2 ‣ 4.2 Comparison results ‣ 4 Experiments ‣ Evolutionary Reinforcement Learning via Cooperative Coevolution") and Figure [4](https://arxiv.org/html/2404.14763v3#S4.F4 "Figure 4 ‣ 4 Experiments ‣ Evolutionary Reinforcement Learning via Cooperative Coevolution"). It is evident from the results that none of the ablated algorithms like ES, CoES and ESSAC outperforms CoERL, indicating their unique contributions to CoERL.

Evolution-based algorithms such as CoES and ES only utilise the outcome of the evaluation as the fitness. Although EAs show some advantages in tackling the credit assignment problem with sparse reward[[25](https://arxiv.org/html/2404.14763v3#bib.bib25)], it wastes adequate temporal information collected during evaluation. On the other hand, modern MDP-based RL algorithms tend to use temporal information, i.e., transition-based experiences, while ignoring the long-term cumulative reward. This dilemma is similar to the trade-off between Monte-Carlo sampling and temporal difference[[27](https://arxiv.org/html/2404.14763v3#bib.bib27)], which seeks to balance between bias and variance of the estimated value function. In our case, the usage of temporal information becomes an issue as traditional evolution-based algorithms merely provide a solution to utilise it.

Instead of a single evolution loop, CoERL maintains an additional MDP-based RL loop for better use of temporal information. The RL loop reuses the experiences collected by the actors in the evolution. Since the evolution loop still inherits the surviving technique based on fitness, a variation between the target policy and the behaviour policy, i.e., individuals, is introduced. So the choice of MDP-based RL has to be the off-policy version[[10](https://arxiv.org/html/2404.14763v3#bib.bib10)].

Intuitively, the additional RL loop proceeds to optimise the policy, which achieves an efficient sample utility for picking up the wasted experiences. The policy is improved using value approximation in a fine-grained way. At the same time, when looking into the way of collecting experiences, we find that the underlying mechanism of the reproduction shows an advantage on exploration. New individuals are generated through proximal perturbation in CoERL, ensuring a diverse range of experiences.

Besides, decompostion is the essential part of CoERL. The simple random grouping is applied to decompose the policy in CoERL. The unique performance is evaluated via the ablation study. It’s possible to improve this part by using other decomposition techniques like[[34](https://arxiv.org/html/2404.14763v3#bib.bib34)], which learns the problem structure and variable dependency for decomposition. However, it might be computationally expensive. Another possible way is to fix the number of subproblems. As the neural network’s structure is known, we can decompose it with different layers. Of course, determining the dependency of layers and merging different layers as a subproblem are crucial.

Table 3: Average running time (minutes) of algorithms over five trials with different random seeds.

### 4.4 Inheriting behaviour space

To further analyse the mechanism of CoERL, we visualise the behaviour space using t-SNE[[29](https://arxiv.org/html/2404.14763v3#bib.bib29)] during training. t-SNE compresses the high-dimensional observation space into the two-dimensional space. _Swimmer-v2_ is chosen for a case study. Figure [5](https://arxiv.org/html/2404.14763v3#S4.F5 "Figure 5 ‣ 4.4 Inheriting behaviour space ‣ 4 Experiments ‣ Evolutionary Reinforcement Learning via Cooperative Coevolution") shows an example of the visualisation in the case of being decomposed into four subproblems. From left to right, the first figure is the behaviour feature map of the agent after optimising weight subset 1, then the second one shows after optimising weight subset 2, and so on. Each subproblem is applied for one full step before the next subproblem in CoERL, however, this update is only limited to the subproblem (denoted by parameter indices ⟨ℐ j⟩delimited-⟨⟩subscript ℐ 𝑗\langle\mathcal{I}_{j}\rangle⟨ caligraphic_I start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ⟩). The gradient of the preceding subproblem remains an implicit effort on the optimisation direction of the subsequent subproblem, acting as momentum. This effort is easily observed in Figure [5](https://arxiv.org/html/2404.14763v3#S4.F5 "Figure 5 ‣ 4.4 Inheriting behaviour space ‣ 4 Experiments ‣ Evolutionary Reinforcement Learning via Cooperative Coevolution"), where the first feature map shares a high similarity with the third one (from left to right), as well as the second and fourth ones. This phenomenon implies that the behaviour space is inherited during optimisation.

Benefiting from the cooperative coevolution strategy and partial gradients, the offspring policy inherits the behaviour space of its parents after being updated in the last subproblem. Since CoERL only optimises the parameters of the subproblem each time, the remaining quotient set of parameters acts as a “memory buffer". After the updates, some old neural activations of the network in the memory buffer still connect, resulting in the emergence of certain behaviours of the new policy. Additionally, the partial gradient can be considered as a form of proximal variation. Instead of aimlessly perturbing the policy, CoERL searches for the partial gradient within a proximal area, which ensures policy updates within a promising range. Then, the sampled individuals provide a certain optimisation direction, which is assembled as a vector according to Eq. ([5](https://arxiv.org/html/2404.14763v3#S3.E5 "In 3.2 Partially updating via cooperative coevolution ‣ 3 Cooperative coevolutionary reinforcement learning ‣ Evolutionary Reinforcement Learning via Cooperative Coevolution")).

![Image 11: Refer to caption](https://arxiv.org/html/2404.14763v3/extracted/5768630/figures/cces_behaviour3.png)

Figure 5: Four exclusive subproblems are highlighted in red, green, blue and yellow, respectively. The feature maps present the behaviour spaces, reduced by t-SNE[[29](https://arxiv.org/html/2404.14763v3#bib.bib29)], of the policy after optimised in subproblems. 

### 4.5 Direct coordination or indirect coordination

A possible cause for the failure of CoERL in _Swimmer-v2_ could be attributed to the coordination between the CC loop and the RL loop. In ERL, the learning agent injects gradient information by replacing the worst individual in the population[[10](https://arxiv.org/html/2404.14763v3#bib.bib10)]. This indirect coordination requires an extra roll-out by the learning agent itself and introduces a trade-off when choosing individuals. Our CoERL avoids these issues by sampling the population at each generation instead of maintaining a permanent population, allowing for more direct coordination. The optimised policy can directly access the evolution loop. However, we have to admit that this direct coordination may stuck in a local optimum if the base learning algorithm is not a good partner, even though CoERL has demonstrated outstanding performances on almost all tasks. Balancing the trade-off between direct and indirect coordination is worth considering as a future research direction.

### 4.6 Runtime analysis

Table [3](https://arxiv.org/html/2404.14763v3#S4.T3 "Table 3 ‣ 4.3 Ablation study ‣ 4 Experiments ‣ Evolutionary Reinforcement Learning via Cooperative Coevolution") presents the average running times of algorithms in five trials. The results show that the running time of CoERL is close to SAC and much shorter than ERL. The two loops in CoERL don’t share computational resources but the evolution loop produces experiences for the RL loop. CoERL is particularly designed for addressing the scalability issue through random grouping, which actually does not introduce additional computational complexity via random grouping. The complexity of the proposed partial updating is 𝒪⁢(|θ|)𝒪 𝜃\mathcal{O}(|\theta|)caligraphic_O ( | italic_θ | ), which should not be greater than the one-time policy gradient via backpropagation. CCNCS has the shortest running time as it doesn’t apply gradient techniques and only updates weights a few times in each generation. However, its performance is relatively poor.

5 Conclusion
------------

In this paper, we propose CoERL, a novel cooperative coevolutionary reinforcement learning algorithm to address the scalability issues and enhance the efficiency during training. CoERL decomposes the policy optimisation problem into multiple subproblems using a cooperative coevolutionary strategy. For each subproblem, CoERL searches for partial gradients to update the policy. This decomposition, coupled with the use of partial gradients, ensures consistency between the behaviour spaces of parents and offspring at a reduced cost. In contrast to traditional evolution-based approaches that discard experiences, CoERL capitalises on the collected experiences within the population, as a novel hierarchy based on cooperative coevolution. Extensive experiments on six locomotion tasks demonstrate that CoERL outperforms seven state-of-the-art algorithms and baselines. The contributions of CoERL’s components are also verified through an ablation study. In the future, CoERL can be extended using knowledge-based grouping techniques and combined with other ERL algorithms like[[41](https://arxiv.org/html/2404.14763v3#bib.bib41)] to cope with the exploration-exploitation dilemma. Furthermore, exploring the explainable decomposition in neural networks via visualisation and quantification could be another interesting direction. It is also worth investigating a thorough theoretical analysis and hyperparameter sensitivity analysis.

References
----------

*   Bai et al. [2023] H.Bai, R.Cheng, and Y.Jin. Evolutionary reinforcement learning: A survey. _Intelligent Computing_, 2:0025, 2023. 
*   Bodnar et al. [2020] C.Bodnar, B.Day, and P.Lió. Proximal distilled evolutionary reinforcement learning. In _AAAI Conference on Artificial Intelligence_, volume 34, pages 3283–3290, 2020. 
*   Brockman et al. [2016] G.Brockman, V.Cheung, L.Pettersson, J.Schneider, J.Schulman, J.Tang, and W.Zaremba. OpenAI Gym. _arXiv preprint:1606.01540_, 2016. 
*   Cauwet et al. [2016] M.-L. Cauwet, J.Liu, B.Rozière, and O.Teytaud. Algorithm portfolios for noisy optimization. _Annals of Mathematics and Artificial Intelligence_, 76:143–172, 2016. 
*   Galván and Mooney [2021] E.Galván and P.Mooney. Neuroevolution in deep neural networks: Current trends and future challenges. _IEEE Transactions on Artificial Intelligence_, 2(6):476–493, 2021. 
*   Gangwani and Peng [2018] T.Gangwani and J.Peng. Genetic policy optimization. In _International Conference on Learning Representations_, pages 1–16, 2018. URL https://openreview.net/forum?id=ByOnmlWC-. 
*   Glorieux et al. [2015] E.Glorieux, B.Svensson, F.Danielsson, and B.Lennartson. Improved constructive cooperative coevolutionary differential evolution for large-scale optimisation. In _IEEE Symposium Series on Computational Intelligence_, pages 1703–1710. IEEE, 2015. 
*   Gong et al. [2020] M.Gong, J.Liu, A.K. Qin, K.Zhao, and K.C. Tan. Evolving deep neural networks via cooperative coevolution with backpropagation. _IEEE Transactions on Neural Networks and Learning Systems_, 32(1):420–434, 2020. 
*   Haarnoja et al. [2018] T.Haarnoja, A.Zhou, P.Abbeel, and S.Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In _International Conference on Machine Learning_, pages 1861–1870. PMLR, 2018. 
*   Khadka and Tumer [2018] S.Khadka and K.Tumer. Evolution-guided policy gradient in reinforcement learning. In _International Conference on Neural Information Processing Systems_, page 1196–1208. Curran Associates Inc., 2018. 
*   Khadka et al. [2019] S.Khadka, S.Majumdar, T.Nassar, Z.Dwiel, E.Tumer, S.Miret, Y.Liu, and K.Tumer. Collaborative evolutionary reinforcement learning. In _International Conference on Machine Learning_, pages 3341–3350. PMLR, 2019. 
*   Lehman et al. [2018a] J.Lehman, J.Chen, J.Clune, and K.O. Stanley. ES is more than just a traditional finite-difference approximator. In _Genetic and Evolutionary Computation Conference_, pages 450–457, 2018a. 
*   Lehman et al. [2018b] J.Lehman, J.Chen, J.Clune, and K.O. Stanley. Safe mutations for deep and recurrent neural networks through output gradients. In _Genetic and Evolutionary Computation Conference_, pages 117–124, 2018b. 
*   Liu et al. [2020] F.-Y. Liu, Z.-N. Li, and C.Qian. Self-guided evolution strategies with historical estimated gradients. In _International Joint Conferences on Artificial Intelligence_, pages 1474–1480, 2020. 
*   Liu and Tang [2013] J.Liu and K.Tang. Scaling up covariance matrix adaptation evolution strategy using cooperative coevolution. In _International Conference on Intelligent Data Engineering and Automated Learning_, pages 350–357. Springer, 2013. 
*   Liu and Teytaud [2014] J.Liu and O.Teytaud. Meta online learning: Experiments on a unit commitment problem. In _European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning_, pages 485–490, 2014. 
*   Ma et al. [2018] X.Ma, X.Li, Q.Zhang, K.Tang, Z.Liang, W.Xie, and Z.Zhu. A survey on cooperative co-evolutionary algorithms. _IEEE Transactions on Evolutionary Computation_, 23(3):421–441, 2018. 
*   Marchesini et al. [2020] E.Marchesini, D.Corsi, and A.Farinelli. Genetic soft updates for policy evolution in deep reinforcement learning. In _International Conference on Learning Representations_, pages 1–15, 2020. 
*   Mnih et al. [2015] V.Mnih, K.Kavukcuoglu, D.Silver, A.A. Rusu, J.Veness, M.G. Bellemare, A.Graves, M.Riedmiller, A.K. Fidjeland, G.Ostrovski, S.Petersen, C.Beattie, A.Sadik, I.Antonoglou, H.King, D.Kumaran, D.Wierstra, S.Legg, and D.Hassabis. Human-level control through deep reinforcement learning. _Nature_, 518(7540):529–533, 2015. 
*   Moriarty et al. [1999] D.E. Moriarty, A.C. Schultz, and J.J. Grefenstette. Evolutionary algorithms for reinforcement learning. _Journal of Artificial Intelligence Research_, 11:241–276, 1999. 
*   Nesterov and Spokoiny [2017] Y.Nesterov and V.Spokoiny. Random gradient-free minimization of convex functions. _Foundations of Computational Mathematics_, 17:527–566, 2017. 
*   Panait [2010] L.Panait. Theoretical convergence guarantees for cooperative coevolutionary algorithms. _Evolutionary Computation_, 18(4):581–615, 2010. 
*   Potter and De Jong [1994] M.A. Potter and K.A. De Jong. A cooperative coevolutionary approach to function optimization. In _International Conference on Evolutionary Computation_, pages 249–257. Springer, 1994. 
*   Pourchot and Sigaud [2019] Pourchot and Sigaud. CEM-RL: Combining evolutionary and gradient-based methods for policy search. In _International Conference on Learning Representations_, pages 1–13, 2019. URL https://openreview.net/forum?id=BkeU5j0ctQ. 
*   Salimans et al. [2017] T.Salimans, J.Ho, X.Chen, S.Sidor, and I.Sutskever. Evolution strategies as a scalable alternative to reinforcement learning. _arXiv preprint arXiv:1703.03864_, 2017. 
*   Silver et al. [2017] D.Silver, J.Schrittwieser, K.Simonyan, I.Antonoglou, A.Huang, A.Guez, T.Hubert, L.Baker, M.Lai, A.Bolton, Y.Chen, T.Lillicrap, F.Hui, L.Sifre, G.van den Driessche, T.Graepel, and D.Hassabis. Mastering the game of Go without human knowledge. _Nature_, 550:354–359, Oct. 2017. 
*   Sutton and Barto [2018] R.S. Sutton and A.G. Barto. _Reinforcement Learning: An Introduction_. MIT press, 2018. 
*   Todorov et al. [2012] E.Todorov, T.Erez, and Y.Tassa. Mujoco: A physics engine for model-based control. In _IEEE/RSJ International Conference on Intelligent Robots and Systems_, pages 5026–5033, 2012. 
*   Van der Maaten and Hinton [2008] L.Van der Maaten and G.Hinton. Visualizing data using t-SNE. _Journal of Machine Learning Research_, 9(11), 2008. 
*   Vinyals et al. [2019] O.Vinyals, I.Babuschkin, W.M. Czarnecki, M.Mathieu, A.Dudzik, J.Chung, D.H. Choi, R.Powell, T.Ewalds, P.Georgiev, J.Oh, D.Horgan, M.Kroiss, I.Danihelka, A.Huang, L.Sifre, T.Cai, J.P. Agapiou, M.Jaderberg, A.S. Vezhnevets, R.Leblond, T.Pohlen, V.Dalibard, D.Budden, Y.Sulsky, J.Molloy, T.L. Paine, C.Gulcehre, Z.Wang, T.Pfaff, Y.Wu, R.Ring, D.Yogatama, D.Wünsch, K.McKinney, O.Smith, T.Schaul, T.Lillicrap, K.Kavukcuoglu, D.Hassabis, C.Apps, and D.Silver. Grandmaster level in StarCraft II using multi-agent reinforcement learning. _Nature_, 575(7782):350–354, 2019. 
*   Wang et al. [2022] Y.Wang, T.Zhang, Y.Chang, X.Wang, B.Liang, and B.Yuan. A surrogate-assisted controller for expensive evolutionary reinforcement learning. _Information Sciences_, 616(C):539–557, 2022. 
*   Weng et al. [2022] J.Weng, H.Chen, D.Yan, K.You, A.Duburcq, M.Zhang, Y.Su, H.Su, and J.Zhu. Tianshou: A highly modularized deep reinforcement learning library. _The Journal of Machine Learning Research_, 23(1):12275–12280, 2022. 
*   Whiteson [2012] S.Whiteson. Evolutionary computation for reinforcement learning. _Reinforcement Learning: State-of-the-art_, pages 325–355, 2012. 
*   Wu et al. [2023] Y.Wu, X.Peng, H.Wang, Y.Jin, and D.Xu. Cooperative coevolutionary CMA-ES with landscape-aware grouping in noisy environments. _IEEE Transactions on Evolutionary Computation_, 27(3):686–700, 2023. [10.1109/TEVC.2022.3180224](https://arxiv.org/doi.org/10.1109/TEVC.2022.3180224). 
*   Yang et al. [2017] P.Yang, K.Tang, and X.Yao. Turning high-dimensional optimization into computationally expensive optimization. _IEEE Transactions on Evolutionary Computation_, 22(1):143–156, 2017. 
*   Yang et al. [2022] P.Yang, H.Zhang, Y.Yu, M.Li, and K.Tang. Evolutionary reinforcement learning via cooperative coevolutionary negatively correlated search. _Swarm and Evolutionary Computation_, 68:100974, 2022. 
*   Yang et al. [2023] P.Yang, L.Zhang, H.Liu, and G.Li. Reducing idleness in financial cloud services via multi-objective evolutionary reinforcement learning based load balancer. _SCIENCE CHINA Information Sciences_, pages 1–20, 2023. [https://doi.org/10.1007/s11432-023-3895-3](https://arxiv.org/doi.org/https://doi.org/10.1007/s11432-023-3895-3). 
*   Yang et al. [2008] Z.Yang, K.Tang, and X.Yao. Multilevel cooperative coevolution for large scale optimization. In _IEEE World Congress on Computational Intelligence_, pages 1663–1670. IEEE, 2008. 
*   Yao [1993] X.Yao. A review of evolutionary artificial neural networks. _International Journal of Intelligent Systems_, 8(4):539–567, 1993. 
*   Yao [1999] X.Yao. Evolving artificial neural networks. _Proceedings of the IEEE_, 87(9):1423–1447, 1999. 
*   Zhu et al. [2024] Q.Zhu, X.Wu, Q.Lin, and W.-N. Chen. Two-stage evolutionary reinforcement learning for enhancing exploration and exploitation. In _AAAI Conference on Artificial Intelligence_, volume 38, pages 20892–20900, 2024.

