Title: Proximal Policy Gradient Arborescence for Quality Diversity Reinforcement Learning

URL Source: https://arxiv.org/html/2305.13795

Published Time: Wed, 31 Jan 2024 02:01:13 GMT

Markdown Content:
Sumeet Batra 

University of Southern California 

Los Angeles, CA 90089 

ssbatra@usc.edu

&Bryon Tjanaka 

University of Southern California 

Los Angeles, CA 90089 

tjanaka@usc.edu&Matthew C. Fontaine 

University of Southern California 

Los Angeles, CA 90089 

mfontain@usc.edu

&Aleksei Petrenko 

University of Southern California 

Los Angeles, CA 90089 

petrenko@usc.edu

&Stefanos Nikolaidis 

University of Southern California 

Los Angeles, CA 90089 

nikolaid@usc.edu

&Gaurav S. Sukhatme 

University of Southern California 

Los Angeles, CA 90089 

gaurav@usc.edu

###### Abstract

Training generally capable agents that thoroughly explore their environment and learn new and diverse skills is a long-term goal of robot learning. Quality Diversity Reinforcement Learning (QD-RL) is an emerging research area that blends the best aspects of both fields – Quality Diversity (QD) provides a principled form of exploration and produces collections of behaviorally diverse agents, while Reinforcement Learning (RL) provides a powerful performance improvement operator enabling generalization across tasks and dynamic environments. Existing QD-RL approaches have been constrained to sample efficient, deterministic off-policy RL algorithms and/or evolution strategies, and struggle with highly stochastic environments. In this work, we, for the first time, adapt on-policy RL, specifically Proximal Policy Optimization (PPO), to the Differentiable Quality Diversity (DQD) framework and propose additional improvements over prior work that enable efficient optimization and discovery of novel skills on challenging locomotion tasks. Our new algorithm, Proximal Policy Gradient Arborescence (PPGA), achieves state-of-the-art results, including a 4x improvement in best reward over baselines on the challenging humanoid domain.

1 Introduction
--------------

Quality Diversity (QD) algorithms enable the exploration and discovery of diverse skills in a behavior space. For example, a QD algorithm can train different locomotion gaits for a walker(Cully et al., [2015](https://arxiv.org/html/2305.13795v2#bib.bib7)), discover different grasping trajectories for a manipulator(Morel et al., [2022](https://arxiv.org/html/2305.13795v2#bib.bib29)), or generate a diverse range of human faces(Fontaine & Nikolaidis, [2021](https://arxiv.org/html/2305.13795v2#bib.bib11)). However, since these algorithms are generally oriented towards solving exploration problems, they struggle to find performant policies in high-dimensional robot learning tasks. QD-RL is an emerging field that attempts to combine the principled exploration capabilities of QD with the powerful performance improvement capabilities of RL. Prior methods have leveraged off-policy RL, specifically TD3, to estimate the gradient of performance, and either Evolution Strategies (ES) or TD3 to estimate the gradient of diversity in order to search for diverse, high-quality policies. They have shown success in exploration problems and certain robot locomotion tasks (Nilsson & Cully, [2021](https://arxiv.org/html/2305.13795v2#bib.bib33); Pierrot et al., [2022](https://arxiv.org/html/2305.13795v2#bib.bib35); Tjanaka et al., [2022b](https://arxiv.org/html/2305.13795v2#bib.bib41)). Nonetheless, there remains a gap in performance between QD-RL and standard RL algorithms on continuous control tasks. Furthermore, off-policy RL algorithms were not designed with massive parallelization in mind, and there is little literature that explores how to leverage modern massively-parallelized simulators with these algorithms, whereas numerous works exploring on-policy RL in these regimes exist (Makoviychuk et al., [2021](https://arxiv.org/html/2305.13795v2#bib.bib27); Rudin et al., [2021](https://arxiv.org/html/2305.13795v2#bib.bib36); Handa et al., [2022](https://arxiv.org/html/2305.13795v2#bib.bib19); Batra et al., [2021](https://arxiv.org/html/2305.13795v2#bib.bib2); Huang et al., [2022](https://arxiv.org/html/2305.13795v2#bib.bib21)).

![Image 1: Refer to caption](https://arxiv.org/html/2305.13795v2/x1.png)

Figure 1: PPGA finds a diverse archive of high-performing locomotion behaviors for a humanoid agent by combining PPO gradient approximations with Differentiable Quality Diversity algorithms. The archive’s dimensions correspond to the measures m 1 subscript 𝑚 1 m_{1}italic_m start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT and m 2 subscript 𝑚 2 m_{2}italic_m start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT, i.e., the proportion of time that the left and right feet contact the ground. The color of each cell shows the objective value, i.e., how fast the humanoid moves. For instance, jumping moves the humanoid forward quickly, with the left and right feet individually contacting the ground 30% and 22% of the time, respectively.

From our investigation of prior methods, simply combining existing QD methods with an RL algorithm tends not to scale well to high-dimensional, highly dynamical systems such as Humanoid. For example, all QD-RL algorithms for locomotion to date use non-Markovian measures of behavioral diversity, which in many cases prevents direct RL-optimization. Most algorithms instead opt for policy parameter mutation, which struggles to scale well with deep neural networks. Prior methods that investigated combining Differentiable Quality Diversity and off-policy RL (Tjanaka et al., [2022b](https://arxiv.org/html/2305.13795v2#bib.bib41)) achieved similar results as other baselines. However, given the gap in performance between standard RL and QD-RL algorithms in terms of best-performing policy, we believe that DQD algorithms, under a different formulation more synergistic with its underlying mechanisms, can close this gap. To this end, we leverage Proximal Policy Optimization (PPO) (Schulman et al., [2017](https://arxiv.org/html/2305.13795v2#bib.bib39)), a popular on-policy RL algorithm, with Differentiable Quality Diversity (DQD) (Fontaine & Nikolaidis, [2021](https://arxiv.org/html/2305.13795v2#bib.bib11)) because of the already present synergy. Specifically, DQD algorithms CMA-MEGA (Fontaine & Nikolaidis, [2021](https://arxiv.org/html/2305.13795v2#bib.bib11)), and its more recent variation CMA-MAEGA (Fontaine & Nikolaidis, [2023](https://arxiv.org/html/2305.13795v2#bib.bib10)), maintain a single search point (or policy in the case of RL) that moves through the behavior space and fills in new, unexplored regions with offspring policies constructed via gradient information collected from online data. It is through this high level view that we see the emergent synergy between PPO and DQD, in that PPO can be used to collect gradient estimates from online data when one or both of the objective and measure functions are markovian and non-differentiable.

We make several key changes to CMA-MAEGA and PPO to maximally leverage their synergy. Our new algorithm, Proximal Policy Gradient Arborescence (PPGA), to the best of our knowledge, is the first QD-RL algorithm to not only achieve 4x performance in best reward on the humanoid domain, but achieve the same level of performance as PPO without sacrificing any of the diversity in the discovered policies. Specifically, we make the following contributions:

(1) We propose a vectorized implementation of PPO, VPPO, that jointly computes the objective and measure gradients with little overhead and without running separate PPO instances for each task (2) We generalize prior CMA-based DQD algorithms as instances of Natural Evolution Strategies (NES) and show that contemporary NES methods, specifically xNES, enables better training stability and performance for DQD algorithms (3) We introduce the notion of Markovian Measure Proxies (MMPs), which makes the typically non-Markovian measure functions used in QD-RL amenable to RL-optimization (4) We propose a new method to move the current search point, hereon referred to as the "search policy", to unexplored regions of the archive by iteratively "walking" it using collected online data and RL optimization of a novel multi-objective reward function.

2 Background
------------

### 2.1 Deep Reinforcement Learning

Reinforcement Learning algorithms search for policies, a mapping of states to actions, that maximize cumulative reward in an environment. RL typically assumes the discrete-time Markov Decision Process (MDP) formalism (𝒮,𝒜,ℛ,𝒫,γ)𝒮 𝒜 ℛ 𝒫 𝛾(\mathcal{S},\mathcal{A},\mathcal{R},\mathcal{P},\gamma)( caligraphic_S , caligraphic_A , caligraphic_R , caligraphic_P , italic_γ ) where 𝒮 𝒮\mathcal{S}caligraphic_S and 𝒜 𝒜\mathcal{A}caligraphic_A are the state and action spaces respectively, ℛ⁢(s,a)ℛ 𝑠 𝑎\mathcal{R}(s,a)caligraphic_R ( italic_s , italic_a ) is the reward function, 𝒫⁢(s′|s,a)𝒫 conditional superscript 𝑠′𝑠 𝑎\mathcal{P}(s^{\prime}|s,a)caligraphic_P ( italic_s start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT | italic_s , italic_a ) defines state transition probabilities, and γ 𝛾\gamma italic_γ is the discount factor. The RL objective is to maximize the discounted episodic return of a policy 𝔼⁢[∑k=0 T−1 γ k⁢R⁢(s k,a k)]𝔼 delimited-[]superscript subscript 𝑘 0 𝑇 1 superscript 𝛾 𝑘 𝑅 subscript 𝑠 𝑘 subscript 𝑎 𝑘\mathbb{E}\left[\sum_{k=0}^{T-1}\gamma^{k}R(s_{k},a_{k})\right]blackboard_E [ ∑ start_POSTSUBSCRIPT italic_k = 0 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_T - 1 end_POSTSUPERSCRIPT italic_γ start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT italic_R ( italic_s start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT , italic_a start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ) ] where T 𝑇 T italic_T is episode length. Deep Reinforcement Learning solves the RL problem by finding a policy π θ⁢(a t|s t)subscript 𝜋 𝜃 conditional subscript 𝑎 𝑡 subscript 𝑠 𝑡\pi_{\theta}(a_{t}|s_{t})italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT | italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) parameterized by a deep neural network θ 𝜃\theta italic_θ that represents a state-action mapping.

On-policy Deep RL methods directly learn the policy π θ subscript 𝜋 𝜃\pi_{\theta}italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT using experience collected by that policy or a recent version thereof. Contemporary methods(Mnih et al., [2016](https://arxiv.org/html/2305.13795v2#bib.bib28)) fit the value function V ϕ⁢(s t)subscript 𝑉 italic-ϕ subscript 𝑠 𝑡 V_{\phi}(s_{t})italic_V start_POSTSUBSCRIPT italic_ϕ end_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) to discounted returns and estimate the advantage A^t=∑k=t T−1 γ k⁢R⁢(s k,a k)−V ϕ⁢(s t)subscript^𝐴 𝑡 superscript subscript 𝑘 𝑡 𝑇 1 superscript 𝛾 𝑘 𝑅 subscript 𝑠 𝑘 subscript 𝑎 𝑘 subscript 𝑉 italic-ϕ subscript 𝑠 𝑡\hat{A}_{t}=\sum_{k=t}^{T-1}\gamma^{k}R(s_{k},a_{k})-V_{\phi}(s_{t})over^ start_ARG italic_A end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_k = italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_T - 1 end_POSTSUPERSCRIPT italic_γ start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT italic_R ( italic_s start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT , italic_a start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ) - italic_V start_POSTSUBSCRIPT italic_ϕ end_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ), which corresponds to the value of an action over the current policy(Schulman et al., [2016](https://arxiv.org/html/2305.13795v2#bib.bib38)). From here, the gradient of the objective w.r.t. θ 𝜃\theta italic_θ, or policy gradient, can be estimated as 𝔼^t⁢[∇θ log⁢π θ⁢(a t|s t)⁢A^t]subscript^𝔼 𝑡 delimited-[]subscript∇𝜃 log subscript 𝜋 𝜃 conditional subscript 𝑎 𝑡 subscript 𝑠 𝑡 subscript^𝐴 𝑡\hat{\mathbb{E}}_{t}\left[\nabla_{\theta}\text{log}\pi_{\theta}(a_{t}|s_{t})% \hat{A}_{t}\right]over^ start_ARG blackboard_E end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT [ ∇ start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT log italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT | italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) over^ start_ARG italic_A end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ], and the policy π θ subscript 𝜋 𝜃\pi_{\theta}italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT is trained using mini-batch gradient descent.

Trust region policy gradient Deep RL methods constrain the policy updates to maintain the proximity of π θ subscript 𝜋 𝜃\pi_{\theta}italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT to the behavior policy π θ o⁢l⁢d subscript 𝜋 subscript 𝜃 𝑜 𝑙 𝑑\pi_{\theta_{old}}italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT italic_o italic_l italic_d end_POSTSUBSCRIPT end_POSTSUBSCRIPT that was used to collect the experience. TRPO(Schulman et al., [2015](https://arxiv.org/html/2305.13795v2#bib.bib37)) takes the largest policy improvement step that satisfies the strict KL-divergence constraint. Proximal Policy Optimization (PPO)(Schulman et al., [2017](https://arxiv.org/html/2305.13795v2#bib.bib39)) approximates the trust region by optimizing a clipped surrogate objective where r t⁢(θ)=π θ⁢(a t|s t)π θ o⁢l⁢d⁢(a t|s t)subscript 𝑟 𝑡 𝜃 subscript 𝜋 𝜃 conditional subscript 𝑎 𝑡 subscript 𝑠 𝑡 subscript 𝜋 subscript 𝜃 𝑜 𝑙 𝑑 conditional subscript 𝑎 𝑡 subscript 𝑠 𝑡 r_{t}(\theta)=\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{old}}(a_{t}|s_{t})}italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ( italic_θ ) = divide start_ARG italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT | italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) end_ARG start_ARG italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT italic_o italic_l italic_d end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT | italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) end_ARG is the importance sampling ratio:

L⁢(θ)=𝔼^π θ⁢[min⁢(r t⁢(θ)⁢A^t),clip⁢(r t⁢(θ),1−ϵ,1+ϵ)⁢A^t].𝐿 𝜃 subscript^𝔼 subscript 𝜋 𝜃 min subscript 𝑟 𝑡 𝜃 subscript^𝐴 𝑡 clip subscript 𝑟 𝑡 𝜃 1 italic-ϵ 1 italic-ϵ subscript^𝐴 𝑡 L(\theta)=\hat{\mathbb{E}}_{\pi_{\theta}}\left[\text{min}(r_{t}(\theta)\hat{A}% _{t}),\text{clip}(r_{t}(\theta),1-\epsilon,1+\epsilon)\hat{A}_{t}\right].italic_L ( italic_θ ) = over^ start_ARG blackboard_E end_ARG start_POSTSUBSCRIPT italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT end_POSTSUBSCRIPT [ min ( italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ( italic_θ ) over^ start_ARG italic_A end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) , clip ( italic_r start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ( italic_θ ) , 1 - italic_ϵ , 1 + italic_ϵ ) over^ start_ARG italic_A end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ] .

Off-policy Deep RL algorithms learn parameterized state-action value functions Q θ⁢(s t,a t)subscript 𝑄 𝜃 subscript 𝑠 𝑡 subscript 𝑎 𝑡 Q_{\theta}(s_{t},a_{t})italic_Q start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) that estimate the value of taking action a t subscript 𝑎 𝑡 a_{t}italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT in state s t subscript 𝑠 𝑡 s_{t}italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT. Then, actions are taken with a greedy policy arg⁢max a⁡Q θ⁢(s t,a t)subscript arg max 𝑎 subscript 𝑄 𝜃 subscript 𝑠 𝑡 subscript 𝑎 𝑡\operatorname*{arg\,max}_{a}Q_{\theta}(s_{t},a_{t})start_OPERATOR roman_arg roman_max end_OPERATOR start_POSTSUBSCRIPT italic_a end_POSTSUBSCRIPT italic_Q start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ), or an ε 𝜀\varepsilon italic_ε-greedy variation thereof. Q-functions can be learned from experience collected by recent or past versions of the policy or another policy altogether.

In continuous control problems, it can be difficult to find a*=arg⁢max a⁡Q θ⁢(s t,a t)superscript 𝑎 subscript arg max 𝑎 subscript 𝑄 𝜃 subscript 𝑠 𝑡 subscript 𝑎 𝑡 a^{*}=\operatorname*{arg\,max}_{a}Q_{\theta}(s_{t},a_{t})italic_a start_POSTSUPERSCRIPT * end_POSTSUPERSCRIPT = start_OPERATOR roman_arg roman_max end_OPERATOR start_POSTSUBSCRIPT italic_a end_POSTSUBSCRIPT italic_Q start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) due to an infinite number of possible actions. To work around this issue, off-policy methods such as DDPG(Lillicrap et al., [2016](https://arxiv.org/html/2305.13795v2#bib.bib25)) and TD3(Fujimoto et al., [2018](https://arxiv.org/html/2305.13795v2#bib.bib13)) learn a deterministic policy μ ϕ⁢(s t)subscript 𝜇 italic-ϕ subscript 𝑠 𝑡\mu_{\phi}(s_{t})italic_μ start_POSTSUBSCRIPT italic_ϕ end_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) by solving m a x ϕ(Q θ(s t,μ ϕ(s t))max_{\phi}(Q_{\theta}(s_{t},\mu_{\phi}(s_{t}))italic_m italic_a italic_x start_POSTSUBSCRIPT italic_ϕ end_POSTSUBSCRIPT ( italic_Q start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_μ start_POSTSUBSCRIPT italic_ϕ end_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) ) using gradient ascent. Other off-policy methods, such as soft actor-critic (SAC) Haarnoja et al. ([2018](https://arxiv.org/html/2305.13795v2#bib.bib17)), maintain an explicit policy π θ subscript 𝜋 𝜃\pi_{\theta}italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT, but similarly derive the policy gradient from the critic, allowing them to learn from off-policy data as well.

### 2.2 Quality Diversity Optimization

Unlike single-objective optimization methods such as RL, Quality Diversity algorithms search for an archive of high-performing, diverse policies. An optimal archive essentially answers the question, "how does performance change with behavior?" by mapping out the optimization landscape of a pre-defined behavior space. The QD problem (Chatzilygeroudis et al., [2021](https://arxiv.org/html/2305.13795v2#bib.bib3)) assumes an objective function f⁢(⋅)𝑓⋅f(\cdot)italic_f ( ⋅ ) that quantifies the agent’s performance and k 𝑘 k italic_k measure functions m 1⁢(⋅)⁢…⁢m k⁢(⋅)subscript 𝑚 1⋅…subscript 𝑚 𝑘⋅m_{1}(\cdot)...m_{k}(\cdot)italic_m start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ( ⋅ ) … italic_m start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ( ⋅ ) that characterize the agent’s behavior. The measure functions, represented jointly as 𝐦⁢(⋅)𝐦⋅\textbf{m}(\cdot)m ( ⋅ ), define an embedding the QD algorithm should span with diverse policies. The QD objective is to find a policy that maximizes f 𝑓 f italic_f for every possible output of m. However, the embedding formed by m is continuous, so the embedding space is discretized into a tessellation of M 𝑀 M italic_M cells. The QD objective then becomes to maximize ∑i=1 M f⁢(θ i)superscript subscript 𝑖 1 𝑀 𝑓 subscript 𝜃 𝑖\sum_{i=1}^{M}f(\theta_{i})∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_M end_POSTSUPERSCRIPT italic_f ( italic_θ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ), where θ i subscript 𝜃 𝑖\theta_{i}italic_θ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT is a policy whose measures 𝐦⁢(θ i)𝐦 subscript 𝜃 𝑖\textbf{m}(\theta_{i})m ( italic_θ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) fall in cell i 𝑖 i italic_i of the tesselation.

QD algorithms originated with NSLC(Lehman & Stanley, [2011a](https://arxiv.org/html/2305.13795v2#bib.bib23); [b](https://arxiv.org/html/2305.13795v2#bib.bib24)) and MAP-Elites(Mouret & Clune, [2015](https://arxiv.org/html/2305.13795v2#bib.bib31); Cully et al., [2015](https://arxiv.org/html/2305.13795v2#bib.bib7)). While these early QD algorithms built on genetic algorithms, modern QD algorithms incorporate optimization techniques like evolution strategies(Fontaine et al., [2020](https://arxiv.org/html/2305.13795v2#bib.bib12); Conti et al., [2018](https://arxiv.org/html/2305.13795v2#bib.bib6); Colas et al., [2020](https://arxiv.org/html/2305.13795v2#bib.bib5)), gradient ascent(Fontaine & Nikolaidis, [2021](https://arxiv.org/html/2305.13795v2#bib.bib11); [2023](https://arxiv.org/html/2305.13795v2#bib.bib10)), and differential evolution(Choi & Togelius, [2021](https://arxiv.org/html/2305.13795v2#bib.bib4)). Several works have applied QD optimization to generative design(Hagg et al., [2020](https://arxiv.org/html/2305.13795v2#bib.bib18); Gaier et al., [2018](https://arxiv.org/html/2305.13795v2#bib.bib14)), procedural content generation(Gravina et al., [2019](https://arxiv.org/html/2305.13795v2#bib.bib16); Earle et al., [2022](https://arxiv.org/html/2305.13795v2#bib.bib8); Khalifa et al., [2018](https://arxiv.org/html/2305.13795v2#bib.bib22)), robot manipulation (Morrison et al., [2020](https://arxiv.org/html/2305.13795v2#bib.bib30)), and reinforcement learning(Nilsson & Cully, [2021](https://arxiv.org/html/2305.13795v2#bib.bib33); Tjanaka et al., [2022b](https://arxiv.org/html/2305.13795v2#bib.bib41); Pierrot & Flajolet, [2023](https://arxiv.org/html/2305.13795v2#bib.bib34)).

### 2.3 Differentiable Quality Diversity

The Differentiable Quality Diversity (DQD)(Fontaine & Nikolaidis, [2021](https://arxiv.org/html/2305.13795v2#bib.bib11)) algorithm Covariance Matrix Adaptation Map Elites via Gradient Arborescence (CMA-MEGA) considers the first-order QD problem where the objective and measure functions are differentiable, with gradients w.r.t. policy parameters represented as ∇f=∂f∂θ∇𝑓 𝑓 𝜃\nabla f=\frac{\partial f}{\partial\theta}∇ italic_f = divide start_ARG ∂ italic_f end_ARG start_ARG ∂ italic_θ end_ARG and ∇𝐦=[∂m 1∂θ,…,∂m k∂θ]∇𝐦 subscript 𝑚 1 𝜃…subscript 𝑚 𝑘 𝜃\nabla\textbf{m}=\left[\frac{\partial m_{1}}{\partial\theta},...,\frac{% \partial m_{k}}{\partial\theta}\right]∇ m = [ divide start_ARG ∂ italic_m start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT end_ARG start_ARG ∂ italic_θ end_ARG , … , divide start_ARG ∂ italic_m start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_ARG start_ARG ∂ italic_θ end_ARG ]. CMA-MEGA maintains a search policy π θ μ subscript 𝜋 subscript 𝜃 𝜇\pi_{\theta_{\mu}}italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT end_POSTSUBSCRIPT in policy parameter space (θ μ∈ℝ N subscript 𝜃 𝜇 superscript ℝ 𝑁\theta_{\mu}\in\mathbb{R}^{N}italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_N end_POSTSUPERSCRIPT) corresponding to some cell in the archive given by the measures <m 1(π θ μ),…,m k(π θ μ)><m_{1}{(\pi_{\theta_{\mu}})},...,m_{k}(\pi_{\theta_{\mu}})>< italic_m start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ( italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT end_POSTSUBSCRIPT ) , … , italic_m start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ( italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT end_POSTSUBSCRIPT ) >, and a search distribution in objective-measure gradient coefficient space maintained by CMA-ES (Hansen, [2016](https://arxiv.org/html/2305.13795v2#bib.bib20)), a zeroth-order optimizer that optimizes the coefficient distribution to produce coefficient vectors that point in the direction of greatest archive improvement. At a high level, CMA-MEGA branches off policies from the search policy in order to locally fill the archive, and then steps the search policy to new, unexplored regions of the archive. During the branching step, the gradients <∇f,∇𝐦>θ μ<\nabla f,\nabla\textbf{m}>_{\theta_{\mu}}< ∇ italic_f , ∇ m > start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT end_POSTSUBSCRIPT and λ 𝜆\lambda italic_λ gradient coefficient vectors <c 0,…,c k>1,…,<c 0,…,c k>λ<c_{0},...,c_{k}>_{1},...,<c_{0},...,c_{k}>_{\lambda}< italic_c start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT , … , italic_c start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT > start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , … , < italic_c start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT , … , italic_c start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT > start_POSTSUBSCRIPT italic_λ end_POSTSUBSCRIPT sampled from the CMA-ES search distribution 𝐜 i∼𝒩⁢(μ,Σ)∈ℝ k+1 similar-to subscript 𝐜 𝑖 𝒩 𝜇 Σ superscript ℝ 𝑘 1\textbf{c}_{i}\sim\mathcal{N}(\mu,\Sigma)\in\mathbb{R}^{k+1}c start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∼ caligraphic_N ( italic_μ , roman_Σ ) ∈ blackboard_R start_POSTSUPERSCRIPT italic_k + 1 end_POSTSUPERSCRIPT are combined via the dot product i.e. <∇f,∇𝐦>θ μ⋅𝐜 1,…<\nabla f,\nabla\textbf{m}>_{\theta_{\mu}}\cdot\textbf{c}_{1},...< ∇ italic_f , ∇ m > start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT end_POSTSUBSCRIPT ⋅ c start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , … to produce local gradients ∇1,…,∇λ subscript∇1…subscript∇𝜆\nabla_{1},...,\nabla_{\lambda}∇ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , … , ∇ start_POSTSUBSCRIPT italic_λ end_POSTSUBSCRIPT around the search policy. Applying the gradients to the search policy gives us λ 𝜆\lambda italic_λ branched policies π θ 1,…,π θ λ subscript 𝜋 subscript 𝜃 1…subscript 𝜋 subscript 𝜃 𝜆\pi_{\theta_{1}},...,\pi_{\theta_{\lambda}}italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT end_POSTSUBSCRIPT , … , italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT italic_λ end_POSTSUBSCRIPT end_POSTSUBSCRIPT. The new policies can then be ranked by how much they improve the archive, i.e., f⁢(π θ i)−f⁢(π θ o⁢l⁢d),i∈[1,λ]𝑓 subscript 𝜋 subscript 𝜃 𝑖 𝑓 subscript 𝜋 subscript 𝜃 𝑜 𝑙 𝑑 𝑖 1 𝜆 f(\pi_{\theta_{i}})-f(\pi_{\theta_{old}}),i\in[1,\lambda]italic_f ( italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT ) - italic_f ( italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT italic_o italic_l italic_d end_POSTSUBSCRIPT end_POSTSUBSCRIPT ) , italic_i ∈ [ 1 , italic_λ ], where π θ o⁢l⁢d subscript 𝜋 subscript 𝜃 𝑜 𝑙 𝑑\pi_{\theta_{old}}italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT italic_o italic_l italic_d end_POSTSUBSCRIPT end_POSTSUBSCRIPT is the incumbent policy in the archive corresponding to the same cell as π θ i subscript 𝜋 subscript 𝜃 𝑖\pi_{\theta_{i}}italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT. Branched policies that map to new, unexplored cells in the archive have f⁢(π θ o⁢l⁢d)𝑓 subscript 𝜋 subscript 𝜃 𝑜 𝑙 𝑑 f(\pi_{\theta_{old}})italic_f ( italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT italic_o italic_l italic_d end_POSTSUBSCRIPT end_POSTSUBSCRIPT ) set to some minimum threshold. This implicitly biases the ranking towards the exploration of new, unvisited cells. This ranking is given to CMA-ES, which internally performs an update that steps the search distribution in the direction of the natural gradient w.r.t. greatest archive improvement. CMA-ES returns weights w 1,…,w λ subscript 𝑤 1…subscript 𝑤 𝜆 w_{1},...,w_{\lambda}italic_w start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , … , italic_w start_POSTSUBSCRIPT italic_λ end_POSTSUBSCRIPT such that ∇s⁢t⁢e⁢p=<w 1,…,w λ>⋅<∇1,…,∇λ>\nabla_{step}=<w_{1},...,w_{\lambda}>\cdot<\nabla_{1},...,\nabla_{\lambda}>∇ start_POSTSUBSCRIPT italic_s italic_t italic_e italic_p end_POSTSUBSCRIPT = < italic_w start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , … , italic_w start_POSTSUBSCRIPT italic_λ end_POSTSUBSCRIPT > ⋅ < ∇ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , … , ∇ start_POSTSUBSCRIPT italic_λ end_POSTSUBSCRIPT > is the natural gradient in parameter space. This weighted linear recombination of the branching gradients is then used to step the search policy in the direction of greatest archive improvement θ μ←θ μ+α⁢∇s⁢t⁢e⁢p←subscript 𝜃 𝜇 subscript 𝜃 𝜇 𝛼 subscript∇𝑠 𝑡 𝑒 𝑝\theta_{\mu}\leftarrow\theta_{\mu}+\alpha\nabla_{step}italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT ← italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT + italic_α ∇ start_POSTSUBSCRIPT italic_s italic_t italic_e italic_p end_POSTSUBSCRIPT.

The current state-of-the-art DQD algorithm, Covariance Matrix Adaptation Map Annealing via Gradient Arborescence (CMA-MAEGA)(Fontaine & Nikolaidis, [2023](https://arxiv.org/html/2305.13795v2#bib.bib10)), introduced the concept of soft archives to CMA-MEGA. Instead of maintaining the best policy in each cell, the archive maintains a threshold t e subscript 𝑡 𝑒 t_{e}italic_t start_POSTSUBSCRIPT italic_e end_POSTSUBSCRIPT and updates the threshold by t e←(1−α)⁢t e+α⁢f⁢(π θ i)←subscript 𝑡 𝑒 1 𝛼 subscript 𝑡 𝑒 𝛼 𝑓 subscript 𝜋 subscript 𝜃 𝑖 t_{e}\leftarrow(1-\alpha)t_{e}+\alpha f(\pi_{\theta_{i}})italic_t start_POSTSUBSCRIPT italic_e end_POSTSUBSCRIPT ← ( 1 - italic_α ) italic_t start_POSTSUBSCRIPT italic_e end_POSTSUBSCRIPT + italic_α italic_f ( italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT ) when a new policy π θ i subscript 𝜋 subscript 𝜃 𝑖\pi_{\theta_{i}}italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT crosses the threshold of its cell e 𝑒 e italic_e. The hyperparameter 0≤α≤1 0 𝛼 1 0\leq\alpha\leq 1 0 ≤ italic_α ≤ 1, referred to as the archive learning rate, controls how much time is spent optimizing a region of the archive before exploring a new region. Soft archives have many theoretical and practical benefits discussed in prior work (Fontaine & Nikolaidis, [2023](https://arxiv.org/html/2305.13795v2#bib.bib10)). Our proposed PPGA algorithm builds directly on CMA-MAEGA.

![Image 2: Refer to caption](https://arxiv.org/html/2305.13795v2/x2.png)

Figure 2:  PPGA estimates ∇f,∇𝐦∇𝑓∇𝐦\nabla f,\nabla\textbf{m}∇ italic_f , ∇ m with PPO. We randomly sample gradient coefficients c and perform weighted linear recombination of the objective-measure gradients with c as the weights. This produces a population of gradients that, in turn, result in a population of branched policies. The policies are evaluated and inserted into the archive. xNES adapts the gradient coefficient distribution based on these insertions towards maximal archive improvement. The new mean of the coefficient distribution is used to walk the search policy towards a new, potentially unexplored region of the archive. 

### 2.4 Quality Diversity Reinforcement Learning

Unlike the standard DQD formulation in which the analytical gradients of f 𝑓 f italic_f and m can be computed, the QD-RL setting considers MDPs in which these functions are non-differentiable and must be approximated with model-free RL. The gradient approximations of f 𝑓 f italic_f and m can be used to improve the performance and diversity of agents in an archive. QD-RL methods can be roughly divided into two subgroups. The first set of approaches directly optimizes over the entire archive by sampling existing policies in the archive and applying operations to the policies’ parameters that either improve their performance or diversity. For example, PGA-ME(Nilsson & Cully, [2021](https://arxiv.org/html/2305.13795v2#bib.bib33)) collects experience from evaluated agents into a replay buffer and uses TD3 to derive a policy gradient that improves the performance of randomly sampled agents from the archive, while using genetic variation(Vassiliades & Mouret, [2018](https://arxiv.org/html/2305.13795v2#bib.bib44)) on the same set of agents to improve diversity and fill new, unexplored cells. Similarly, QDPG(Pierrot et al., [2022](https://arxiv.org/html/2305.13795v2#bib.bib35)) derives a policy and diversity gradient using TD3 and applies these operators to randomly sampled agents in the archive.

Whereas the first family of QD-RL algorithms simultaneously search the behavioral embedding in many different regions at once, the second family uses the DQD formulation i.e., maintains a single search policy that explores new local regions one at a time using objective-measure gradient approximations. In prior work(Tjanaka et al., [2022b](https://arxiv.org/html/2305.13795v2#bib.bib41)), the authors considered objective gradient approximations via TD3 and OpenAI-ES, while approximating the measure function gradients with OpenAI-ES. In this work, we notice the unique on-policy nature of DQD algorithms and present a novel formulation that exploits its synergy with PPO.

3 Proposed Method: The Proximal Policy Gradient Arborescence Algorithm
----------------------------------------------------------------------

We begin with the DQD algorithm CMA-MAEGA as our foundation. The algorithm can be roughly divided into three phases: (1) computing the objective-measure gradients for the branching phase, (2) providing the relative ranking of each branched policy w.r.t. the QD objective to CMA-ES, and (3) stepping the search policy in the direction of greatest archive improvement. Sections 3.1 and 3.2 focus on enabling RL optimization for phase one, 3.3 explores the connection between CMA-ES and NES and how this can improve training stability in phase two, and section 3.4 describes our method for walking the search policy with PPO in phase three.

### 3.1 Markovian Measure Proxies

It is often the case that QD problems contain non-Markovian measure functions. For robot locomotion tasks, the standard measure function is the proportional foot contact time with the ground m i⁢(θ)=1 T⁢∑t=0 T δ i⁢(s t)subscript 𝑚 𝑖 𝜃 1 𝑇 superscript subscript 𝑡 0 𝑇 subscript 𝛿 𝑖 subscript 𝑠 𝑡 m_{i}(\theta)=\frac{1}{T}\sum_{t=0}^{T}\delta_{i}(s_{t})italic_m start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( italic_θ ) = divide start_ARG 1 end_ARG start_ARG italic_T end_ARG ∑ start_POSTSUBSCRIPT italic_t = 0 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_T end_POSTSUPERSCRIPT italic_δ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) for each leg i,i=1⁢…⁢k 𝑖 𝑖 1…𝑘 i,i=1...k italic_i , italic_i = 1 … italic_k, where the Kronecker delta δ i⁢(s t)subscript 𝛿 𝑖 subscript 𝑠 𝑡\delta_{i}(s_{t})italic_δ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) indicates whether the i 𝑖 i italic_i’th leg is in contact with the ground or not in state s t subscript 𝑠 𝑡 s_{t}italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT, and T is the episode length. However, this measure function is defined on a trajectory (i.e., the whole episode), making it non-Markovian and thus preventing us from using RL to estimate its gradient. To solve this issue, we introduce the notion of a Markovian Measure Proxy (MMP), which is a surrogate function that obeys the Markov property and has a positive correlation with the original measure function. For locomotion tasks, we can construct an MMP by simply removing the dependency on the trajectory and making the original measure function state-dependent, i.e., setting it to be δ i⁢(s t)subscript 𝛿 𝑖 subscript 𝑠 𝑡\delta_{i}(s_{t})italic_δ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ). We can then use the exact same MDP as the standard RL formulation and replace the reward function with δ i⁢(s t)subscript 𝛿 𝑖 subscript 𝑠 𝑡\delta_{i}(s_{t})italic_δ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ).

### 3.2 Policy Gradients for Differentiable Quality Diversity Optimization

PPO is an attractive choice as our objective-measure gradient estimator because of its ability to scale with additional parallel environments. Being an approximate trust region method, the constrained policy update step provides some robustness to noisy and non-stationary objectives. This is particularly important in the QD-RL setting, where the QD-objective is highly non-stationary – that is, the QD-objective changes with the state of the archive, which is updated on each QD iteration.

We treat the RL objective and k 𝑘 k italic_k MMPs, each one optimized by an actor-critic pair, as reward functions to optimize. Rather than spawning a new PPO instance with a separate actor and critic network for the RL objective f 𝑓 f italic_f and each MMP δ i⁢(s t)subscript 𝛿 𝑖 subscript 𝑠 𝑡\delta_{i}(s_{t})italic_δ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) independently, we start with a single actor π θ μ⁢(a|s)subscript 𝜋 subscript 𝜃 𝜇 conditional 𝑎 𝑠\pi_{\theta_{\mu}}(a|s)italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( italic_a | italic_s ) parameterized by the policy parameters θ μ subscript 𝜃 𝜇\theta_{\mu}italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT of the current search policy, and k+1 𝑘 1 k+1 italic_k + 1 value functions V ϕ f,V ϕ δ 1,…,V ϕ δ k subscript 𝑉 subscript italic-ϕ 𝑓 subscript 𝑉 subscript italic-ϕ subscript 𝛿 1…subscript 𝑉 subscript italic-ϕ subscript 𝛿 𝑘 V_{\phi_{f}},V_{\phi_{\delta_{1}}},...,V_{\phi_{\delta_{k}}}italic_V start_POSTSUBSCRIPT italic_ϕ start_POSTSUBSCRIPT italic_f end_POSTSUBSCRIPT end_POSTSUBSCRIPT , italic_V start_POSTSUBSCRIPT italic_ϕ start_POSTSUBSCRIPT italic_δ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT end_POSTSUBSCRIPT end_POSTSUBSCRIPT , … , italic_V start_POSTSUBSCRIPT italic_ϕ start_POSTSUBSCRIPT italic_δ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_POSTSUBSCRIPT end_POSTSUBSCRIPT. The actor is replicated k+1 𝑘 1 k+1 italic_k + 1 times, each one paired with a corresponding value function. The actors are combined into a single vectorized policy π<θ μ 1,…,θ μ k+1>⁢(a|s)subscript 𝜋 absent superscript subscript 𝜃 𝜇 1…superscript subscript 𝜃 𝜇 𝑘 1 absent conditional 𝑎 𝑠\pi_{<\theta_{\mu}^{1},...,\theta_{\mu}^{k+1}>}(a|s)italic_π start_POSTSUBSCRIPT < italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 1 end_POSTSUPERSCRIPT , … , italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_k + 1 end_POSTSUPERSCRIPT > end_POSTSUBSCRIPT ( italic_a | italic_s ) that jointly optimizes <f,δ 1(s t),…,δ 1(s t)><f,\delta_{1}(s_{t}),...,\delta_{1}(s_{t})>< italic_f , italic_δ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) , … , italic_δ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) > for N 1 subscript 𝑁 1 N_{1}italic_N start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT iterations, where N 1 subscript 𝑁 1 N_{1}italic_N start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT is a configurable hyperparameter. We additionally modify the computation of the policy gradient into a batched policy gradient method, where intermediate gradient estimates of each function w.r.t. policy params only flow back to the parameters corresponding to respective individual policies during minibatch gradient descent. After N 1 subscript 𝑁 1 N_{1}italic_N start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT iterations, we separate the vectorized policy, giving a set of subpolicies with optimized parameters π θ f⁢(a|s),…,π θ δ k⁢(a|s)subscript 𝜋 subscript 𝜃 𝑓 conditional 𝑎 𝑠…subscript 𝜋 subscript 𝜃 subscript 𝛿 𝑘 conditional 𝑎 𝑠\pi_{\theta_{f}}(a|s),...,\pi_{\theta_{\delta_{k}}}(a|s)italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT italic_f end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( italic_a | italic_s ) , … , italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT italic_δ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( italic_a | italic_s ) that perform better w.r.t. their objectives. In the case of measure functions where m i⁢(⋅)subscript 𝑚 𝑖⋅m_{i}(\cdot)italic_m start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( ⋅ ) is the proportion foot contact time of the i′⁢t⁢h superscript 𝑖′𝑡 ℎ i^{\prime}th italic_i start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT italic_t italic_h leg, m i⁢(π θ δ k⁢(s t))>m i⁢(π θ μ)subscript 𝑚 𝑖 subscript 𝜋 subscript 𝜃 subscript 𝛿 𝑘 subscript 𝑠 𝑡 subscript 𝑚 𝑖 subscript 𝜋 subscript 𝜃 𝜇 m_{i}(\pi_{\theta_{\delta_{k}(s_{t})}})>m_{i}(\pi_{\theta_{\mu}})italic_m start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT italic_δ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ( italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) end_POSTSUBSCRIPT end_POSTSUBSCRIPT ) > italic_m start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT end_POSTSUBSCRIPT ) i.e. the resulting policy will have a higher proportion foot contact time over the starting policy after optimization. Subtracting the initial parameters (θ μ subscript 𝜃 𝜇\theta_{\mu}italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT) from each resulting policy gives us the desired objective-measure Jacobian [∂f∂θ μ,∂δ 1∂θ μ,…,∂δ k∂θ μ]𝑓 subscript 𝜃 𝜇 subscript 𝛿 1 subscript 𝜃 𝜇…subscript 𝛿 𝑘 subscript 𝜃 𝜇\left[\frac{\partial f}{\partial\theta_{\mu}},\frac{\partial\delta_{1}}{% \partial\theta_{\mu}},...,\frac{\partial\delta_{k}}{\partial\theta_{\mu}}\right][ divide start_ARG ∂ italic_f end_ARG start_ARG ∂ italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT end_ARG , divide start_ARG ∂ italic_δ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT end_ARG start_ARG ∂ italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT end_ARG , … , divide start_ARG ∂ italic_δ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_ARG start_ARG ∂ italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT end_ARG ], which can be linearly recombined in various ways to branch policies from θ μ subscript 𝜃 𝜇\theta_{\mu}italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT.

In addition to the VPPO implementation, we introduce the option to make the learnable action standard deviation parameter static. In the typical case, PPO decays this parameter over time in order to converge to a quasi-deterministic optimal policy at the expense of further exploration. In some environments, narrowing the action distribution can indeed help promote consistent optimal performance. In other environments, this effect can hinder the QD algorithm’s ability to branch policies into new cells, given that the outer QD optimization loop relies on gradient estimates produced by PPO to discover unexplored regions of the archive. In environments where we observe this negative effect, we disable gradient flow to the action standard deviation parameter.

Finally, in order to address environmental uncertainty, we insert new policies based on their performance and behavior averaged over 10 parallel environments. We leverage GPU acceleration to quickly batch process many parallel environments over a population of branched policies.

### 3.3 Connection to Natural Evolution Strategies

We replace CMA-ES with a Natural Evolution Strategy (NES) in order to increase the stability and performance of CMA-MAEGA on noisy RL environments. CMA- variants of PPGA diverged during training. Prior work (Müller & Glasmachers, [2018](https://arxiv.org/html/2305.13795v2#bib.bib32)) showed that CMA-ES struggled to evolve deep neural network controllers with dimensionality ℝ d superscript ℝ 𝑑\mathbb{R}^{d}blackboard_R start_POSTSUPERSCRIPT italic_d end_POSTSUPERSCRIPT on stochastic RL environments. However, CMA-MAEGA uses CMA-ES to maintain search distribution in objective-measure gradient coefficient space ℝ k+1<<ℝ d much-less-than superscript ℝ 𝑘 1 superscript ℝ 𝑑\mathbb{R}^{k+1}<<\mathbb{R}^{d}blackboard_R start_POSTSUPERSCRIPT italic_k + 1 end_POSTSUPERSCRIPT << blackboard_R start_POSTSUPERSCRIPT italic_d end_POSTSUPERSCRIPT, where k+1 𝑘 1 k+1 italic_k + 1 can be as small as three dimensions, implying that CMA-ES should still be effective in this low-dimensional space. It was then puzzling to find consistent divergence during the training of our CMA-based algorithm. We hypothesize that the culprit is the cumulative step-size adaptation (CSA) mechanism employed by CMA-ES. CMA-ES uses evolution paths ρ σ(g)superscript subscript 𝜌 𝜎 𝑔\rho_{\sigma}^{(g)}italic_ρ start_POSTSUBSCRIPT italic_σ end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_g ) end_POSTSUPERSCRIPT to adapt the step size σ(g)superscript 𝜎 𝑔\sigma^{(g)}italic_σ start_POSTSUPERSCRIPT ( italic_g ) end_POSTSUPERSCRIPT between successive generations (g). The mechanisms by which σ(g)superscript 𝜎 𝑔\sigma^{(g)}italic_σ start_POSTSUPERSCRIPT ( italic_g ) end_POSTSUPERSCRIPT are updated assume a fairly non-noisy and stationary objective f 𝑓 f italic_f. However, the application of CMA-ES to QD optimization on stochastic RL environments presumes the exact opposite. That is, the RL objective f R⁢L subscript 𝑓 𝑅 𝐿 f_{RL}italic_f start_POSTSUBSCRIPT italic_R italic_L end_POSTSUBSCRIPT is very noisy, and the QD-objective f Q⁢D=g⁢(f R⁢L⁢(⋅))subscript 𝑓 𝑄 𝐷 𝑔 subscript 𝑓 𝑅 𝐿⋅f_{QD}=g(f_{RL}(\cdot))italic_f start_POSTSUBSCRIPT italic_Q italic_D end_POSTSUBSCRIPT = italic_g ( italic_f start_POSTSUBSCRIPT italic_R italic_L end_POSTSUBSCRIPT ( ⋅ ) ), which is a function of the RL objective, is highly non-stationary, since the state of the archive 𝒜 𝒜\mathcal{A}caligraphic_A changes the direction of greatest archive improvement on every iteration. To address the training divergence, we propose using exponential evolution strategies (xNES) Glasmachers et al. ([2010](https://arxiv.org/html/2305.13795v2#bib.bib15)), a more recent and theoretically well-motivated method, as a drop in replacement for CMA-ES. Prior works have shown strong links between xNES and CMA-ES, and generalize both methods as instances of natural evolution strategies(Akimoto et al., [2010](https://arxiv.org/html/2305.13795v2#bib.bib1); Glasmachers et al., [2010](https://arxiv.org/html/2305.13795v2#bib.bib15)). In fact, the update step in xNES is equivalent to CMA-ES up to the use of evolution paths. We refer the reader to these prior works for an in-depth comparison. More generally, we believe that any natural evolution strategy can be used to maintain and update the search distribution over gradient coefficients in this and any prior CMA-based DQD method.

### 3.4 Walking the Search Policy

In standard DQD, ∇s⁢t⁢e⁢p subscript∇𝑠 𝑡 𝑒 𝑝\nabla_{step}∇ start_POSTSUBSCRIPT italic_s italic_t italic_e italic_p end_POSTSUBSCRIPT is computed via weighted linear recombination to produce a gradient vector that steps the search policy in the least explored direction of the archive. However, the resulting gradient vector is a linearized approximation around the current search policy θ μ subscript 𝜃 𝜇\theta_{\mu}italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT and thus cannot be reused to take multiple gradient steps in a non-convex optimization problem. It would be remiss not to leverage the highly-parallelized VPPO implementation to "walk" the search policy over many steps in the direction of greatest archive improvement. We make the key observation that the mean gradient coefficient vector 𝐜 μ subscript 𝐜 𝜇\textbf{c}_{\mu}c start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT of the updated search distribution maintained by xNES points in the direction of greatest archive improvement for the next iteration of the QD algorithm. Thus, we construct a new multi-objective reward function for VPPO to optimize by taking the dot product between the gradient coefficient vector and the objective and measure proxies <c μ 0,…,c μ k+1>⋅<f,δ 1,…,δ k><c_{\mu_{0}},...,c_{\mu_{k+1}}>\cdot<f,\delta_{1},...,\delta_{k}>< italic_c start_POSTSUBSCRIPT italic_μ start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT end_POSTSUBSCRIPT , … , italic_c start_POSTSUBSCRIPT italic_μ start_POSTSUBSCRIPT italic_k + 1 end_POSTSUBSCRIPT end_POSTSUBSCRIPT > ⋅ < italic_f , italic_δ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , … , italic_δ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT >. Optimizing this function with VPPO allows us to walk the search policy θ μ subscript 𝜃 𝜇\theta_{\mu}italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT in the direction of greatest archive improvement by iteratively taking conservative steps, where the magnitude of the movement is controllable by hyperparameter N 2 subscript 𝑁 2 N_{2}italic_N start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT. This objective is stationary for all N 2 subscript 𝑁 2 N_{2}italic_N start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT steps, and is only updated after the subsequent QD iteration. We provide pseudocode in Appendix [A](https://arxiv.org/html/2305.13795v2#A1 "Appendix A PPGA Pseudocode ‣ Proximal Policy Gradient Arborescence for Quality Diversity Reinforcement Learning").

4 Experiments
-------------

We evaluate our algorithm on four different continuous-control locomotion tasks derived from the original Mujoco environments (Todorov et al., [2012](https://arxiv.org/html/2305.13795v2#bib.bib43)): Ant, Walker2d, Half-Cheetah, and Humanoid. The standard objective in each task is to maximize forward progress and robot stability while minimizing energy consumption. We use the Brax simulator to leverage GPU acceleration and massive parallelization of the environments. The observation space sizes for these environments are 87, 17, 18, and 227, respectively, and the action space sizes are 8, 6, 6, and 17, respectively. The standard Brax environments are augmented with wrappers that determine the measures of an agent in any given rollout as implemented in QDax (Lim et al., [2022](https://arxiv.org/html/2305.13795v2#bib.bib26)), where the number of measures of an agent is equivalent to the number of legs. The measure function is the number of times a leg contacts the ground divided by the length of the trajectory. We implement PPGA in pyribs(Tjanaka et al., [2023](https://arxiv.org/html/2305.13795v2#bib.bib42)), with our VPPO implementation based on CleanRL’s implementation of PPO(Huang et al., [2022](https://arxiv.org/html/2305.13795v2#bib.bib21)). Most experiments were run on a SLURM cluster where each job had access to an NVIDIA RTX 2080Ti GPUs, 4 cores from a Intel(R) Xeon(R) Gold 6154 3.00GHz CPU, and 108GB of RAM. Some additional experiments and ablations were run on local workstations with access to an NVIDIA RTX 3090, AMD Ryzen 7900x 12 core CPU, and 64GB of RAM.

### 4.1 Comparisons

We compare our results to current state-of-the-art QD-RL algorithms: Policy Gradient Assisted MAP-Elites (PGA-ME),1 1 1 A comparison on Humanoid to PBT-ME (SAC), a recent QD-RL method, can be found in Appendix [H](https://arxiv.org/html/2305.13795v2#A8 "Appendix H Comparing to PBT-ME (SAC) ‣ Proximal Policy Gradient Arborescence for Quality Diversity Reinforcement Learning"). PBT-ME (SAC) was trained with Google TPUs. Due to computational constraints, we were only able to provide a comparison on one task. Quality Diversity Policy Gradient (QDPG) implemented in QDax(Lim et al., [2022](https://arxiv.org/html/2305.13795v2#bib.bib26)), and CMA-MAEGA(TD3, ES) implemented in pyribs(Tjanaka et al., [2023](https://arxiv.org/html/2305.13795v2#bib.bib42)). We also compare against the state-of-the-art ES-based QD-RL algorithm, separable CMA-MAE (sep-CMA-MAE) (Tjanaka et al., [2022a](https://arxiv.org/html/2305.13795v2#bib.bib40)), which allows evolutionary QD techniques to scale up to larger neural networks. Finally, in order to verify our hypothesis on the emergent synergy between PPO and DQD, we provide an ablation where TD3 is used as a drop-in replacement for PPO in PPGA, which we will refer to as TD3GA going forward. Details on the TD3GA design choices and additional ablations, such as comparing against standard PPO, can be found in the appendix. The same archive resolutions and network architectures are used for all baselines. A full list of shared hyperparameters is in Appendix [B](https://arxiv.org/html/2305.13795v2#A2 "Appendix B Hyperparameters ‣ Proximal Policy Gradient Arborescence for Quality Diversity Reinforcement Learning"). We use an archive learning rate of 0.1, 0.15, 0.1, and 1.0 on Humanoid, Walker2d, Ant, and Half-Cheetah, respectively. Adaptive standard deviation is enabled for Ant and Humanoid. We reset the action distribution standard deviation to 1.0 on each iteration in all other environments.

![Image 3: Refer to caption](https://arxiv.org/html/2305.13795v2/x3.png)

Figure 3: 2D Archive visualizations of PPGA compared to the current state-of-the-art QD-RL algorithm PGA-ME. We use 50x50 archives to show detail.

We conduct our experiments using the following criteria: QD-score, which is the sum of scores of all nonempty cells in the archive, and coverage, which is the percentage of nonempty cells in the archive, have been historically used by QD algorithms to measure performance and diversity respectively, and so we include them as metrics. However, these metrics have a number of edge cases that make them imperfect measures of performance and diversity. For example, an algorithm that fills 100% of the archive with low-performing policies can have a higher QD-score and coverage than a QD algorithm that fills fewer cells with high-performing policies. To more accurately represent the performance and diversity of a given algorithm, we additionally include plots of the Complementary Cumulative Distribution Function (CCDF), originally presented in (Vassiliades et al., [2016](https://arxiv.org/html/2305.13795v2#bib.bib45)), which shows what percentage of policies in the archive achieve a reward of R 𝑅 R italic_R or greater for all possible values of R 𝑅 R italic_R on the x 𝑥 x italic_x-axis. The CCDF attempts to capture notions of quality of policies in the archive and diversity, while also shedding light on how the policies are distributed w.r.t. performance. Finally, we include the best reward metric, denoting the highest-performing policy the algorithm was able to discover.

![Image 4: Refer to caption](https://arxiv.org/html/2305.13795v2/x4.png)

Figure 4: QD metrics and cumulative distributions for archives from PPGA compared to baselines. The CCDF plots in the last row indicate the percentage of archive policies that fall above a certain objective threshold. All plots are averaged over four seeds, and the shaded region represents a 95% bootstrapped confidence interval.

In Figures [3](https://arxiv.org/html/2305.13795v2#S4.F3 "Figure 3 ‣ 4.1 Comparisons ‣ 4 Experiments ‣ Proximal Policy Gradient Arborescence for Quality Diversity Reinforcement Learning") and [4](https://arxiv.org/html/2305.13795v2#S4.F4 "Figure 4 ‣ 4.1 Comparisons ‣ 4 Experiments ‣ Proximal Policy Gradient Arborescence for Quality Diversity Reinforcement Learning"), we see that PPGA outperforms baselines in best reward and QD-score, achieves comparable coverage scores on all tasks except for Ant, and generates much more illuminated archive heatmaps with a diverse range of higher performing policies than the current state of the art, PGA-ME. Notably, PPGA is the only algorithm capable of solving the Humanoid domain, achieving a greater than 4x jump in best-performing policy and QD score compared to baselines. More important than QD-Score and Coverage are the CCDF plots. At x=0 𝑥 0 x=0 italic_x = 0, all policies in the archive are included, i.e., x=0 𝑥 0 x=0 italic_x = 0 encapsulates the coverage score. CCDF plots additionally provide a better representation of "quality" than QD-score, since we can see how the policies in the archive are distributed. Except for Ant, PPGA consistently produces distributions where more of the mass is distributed to the right where the high-performing policies lie.

![Image 5: Refer to caption](https://arxiv.org/html/2305.13795v2/x5.png)

Figure 5: PPGA vs TD3GA on Humanoid on the standard QD metrics. All plots are averaged over 4 seeds. The shaded regions are the 95% bootstrapped confidence intervals.

In Figure [5](https://arxiv.org/html/2305.13795v2#S4.F5 "Figure 5 ‣ 4.1 Comparisons ‣ 4 Experiments ‣ Proximal Policy Gradient Arborescence for Quality Diversity Reinforcement Learning"), we find evidence that PPO indeed has an important synergy with DQD that is perhaps missing in other RL algorithms. TD3GA fails to find high performing policies on Humanoid. Achieving 100% coverage is indicative of the step size σ 𝜎\sigma italic_σ in xNES exploding and producing highly stochastic policies that, by chance, land in far away cells. This typically occurs when xNES cannot fit a covariance matrix to the data, which in this case are weighted linear combinations of ∇f,∇𝐦∇𝑓∇𝐦\nabla f,\nabla\textbf{m}∇ italic_f , ∇ m produced by TD3.

### 4.2 Post-Hoc Archive Analysis

QD algorithms are known to struggle with reproducing performance and behavior in stochastic environments. To determine the replicability of our agents, we follow the guidelines laid out in Flageat et al. ([2023](https://arxiv.org/html/2305.13795v2#bib.bib9)). That is, we re-evaluate each agent in the archive 50 times and average its performance and measures to construct a Corrected Archive and use this to produce Corrected QD metrics such as Corrected QD-Score and Corrected Coverage. Fig.[6](https://arxiv.org/html/2305.13795v2#S4.F6 "Figure 6 ‣ 4.2 Post-Hoc Archive Analysis ‣ 4 Experiments ‣ Proximal Policy Gradient Arborescence for Quality Diversity Reinforcement Learning") shows the corrected QD metrics and the corrected CCDFs, respectively. After re-evaluation, PPGA maintains the lead in best reward on all tasks, QD-score on Humanoid and Ant, and Coverage on Humanoid. The CCDF plots of the re-evaluated archives show PPGA producing better distributions of policies on all tasks except Ant, suggesting that PPGA-produced policies are robust to stochasticity.

![Image 6: Refer to caption](https://arxiv.org/html/2305.13795v2/x6.png)

Humanoid Walker2d Halfcheetah Ant
QD-Score Cov Best QD-Score Cov Best QD-Score Cov Best QD-Score Cov Best
PPGA 1.02×𝟏𝟎 𝟓 1.02 superscript 10 5\mathbf{1.02\times 10^{5}}bold_1.02 × bold_10 start_POSTSUPERSCRIPT bold_5 end_POSTSUPERSCRIPT 0.52 8324 1.06×10 5 1.06 superscript 10 5 1.06\times 10^{5}1.06 × 10 start_POSTSUPERSCRIPT 5 end_POSTSUPERSCRIPT 0.39 4702 7.26×𝟏𝟎 𝟓 7.26 superscript 10 5\mathbf{7.26\times 10^{5}}bold_7.26 × bold_10 start_POSTSUPERSCRIPT bold_5 end_POSTSUPERSCRIPT 0.58 8919 1.53×10 7 1.53 superscript 10 7 1.53\times 10^{7}1.53 × 10 start_POSTSUPERSCRIPT 7 end_POSTSUPERSCRIPT 0.34 7328
PGA-ME 1.08×10 4 1.08 superscript 10 4 1.08\times 10^{4}1.08 × 10 start_POSTSUPERSCRIPT 4 end_POSTSUPERSCRIPT 0.39 446 6.53×10 4 6.53 superscript 10 4 6.53\times 10^{4}6.53 × 10 start_POSTSUPERSCRIPT 4 end_POSTSUPERSCRIPT 0.38 1621 3.31×10 5 3.31 superscript 10 5 3.31\times 10^{5}3.31 × 10 start_POSTSUPERSCRIPT 5 end_POSTSUPERSCRIPT 0.31 4644 1.79×𝟏𝟎 𝟕 1.79 superscript 10 7\mathbf{1.79\times 10^{7}}bold_1.79 × bold_10 start_POSTSUPERSCRIPT bold_7 end_POSTSUPERSCRIPT 0.37 4571
sep-CMA-MAE 1.27×10 4 1.27 superscript 10 4 1.27\times 10^{4}1.27 × 10 start_POSTSUPERSCRIPT 4 end_POSTSUPERSCRIPT 0.42 498 1.84×𝟏𝟎 𝟓 1.84 superscript 10 5\mathbf{1.84\times 10^{5}}bold_1.84 × bold_10 start_POSTSUPERSCRIPT bold_5 end_POSTSUPERSCRIPT 0.64 2326 6.03×10 5 6.03 superscript 10 5 6.03\times 10^{5}6.03 × 10 start_POSTSUPERSCRIPT 5 end_POSTSUPERSCRIPT 0.61 228 1.16×10 7 1.16 superscript 10 7 1.16\times 10^{7}1.16 × 10 start_POSTSUPERSCRIPT 7 end_POSTSUPERSCRIPT 0.30 1629
QDPG 6.53×10 3 6.53 superscript 10 3 6.53\times 10^{3}6.53 × 10 start_POSTSUPERSCRIPT 3 end_POSTSUPERSCRIPT 0.26 412 6.59×10 4 6.59 superscript 10 4 6.59\times 10^{4}6.59 × 10 start_POSTSUPERSCRIPT 4 end_POSTSUPERSCRIPT 0.35 1490 2.53×10 5 2.53 superscript 10 5 2.53\times 10^{5}2.53 × 10 start_POSTSUPERSCRIPT 5 end_POSTSUPERSCRIPT 0.28 359 3.03×10 6 3.03 superscript 10 6 3.03\times 10^{6}3.03 × 10 start_POSTSUPERSCRIPT 6 end_POSTSUPERSCRIPT 0.25 78
CMA-MAEGA(TD3, ES)4.10×10 3 4.10 superscript 10 3 4.10\times 10^{3}4.10 × 10 start_POSTSUPERSCRIPT 3 end_POSTSUPERSCRIPT 0.21 352 3.48×10 4 3.48 superscript 10 4 3.48\times 10^{4}3.48 × 10 start_POSTSUPERSCRIPT 4 end_POSTSUPERSCRIPT 0.33 1025 5.70×10 5 5.70 superscript 10 5 5.70\times 10^{5}5.70 × 10 start_POSTSUPERSCRIPT 5 end_POSTSUPERSCRIPT 0.56 2438 3.08×10 6 3.08 superscript 10 6 3.08\times 10^{6}3.08 × 10 start_POSTSUPERSCRIPT 6 end_POSTSUPERSCRIPT 0.12 1020

Figure 6: Corrected QD metrics: QD-Score, Coverage (Cov), and Best Reward (Best), as averaged over four seeds. Plots show corrected cumulative distributions, with error bars indicating a 95% bootstrapped confidence interval.

5 Discussion and Limitations
----------------------------

We present a new method, PPGA, which is one of the first QD-RL methods to leverage on-policy RL, the first to solve the challenging Humanoid task, and the first to achieve equivalent performance in best reward compared to standard RL on all domains. We show that DQD algorithms and on-policy RL have emergent synergies that make them work particularly well with each other. However, instead of simply combining DQD and on-policy RL as is, we re-examine the fundamental assumptions and mechanisms of each component and implement changes that maximize their synergies. There are some caveats with this approach. On-policy RL algorithms such as PPO are quite sample-inefficient and require many parallel environments per agent in order to compute the stochastic policy gradient. Although GPU acceleration and massive parallelism improve wall-clock convergence over off-policy RL, this makes our approach less sample-efficient than other off-policy QD-RL methods. Secondly, enabling PPO’s adaptive standard deviation parameter (which is true by default for PPO) can have detrimental effects on PPGA’s exploration capabilities, as made evident by the coverage score on Ant. This is mainly due to the fact that PPO favors collapsing the standard deviation to achieve higher average returns. In the future, we will investigate modifying the standard deviation parameter such that it dynamically shrinks or increases the standard deviation value based on the QD-optimization landscape as opposed to the RL one. Finally, we are interested to see how this method scales to even more data-rich regimes such as distributed settings, as well as its application to harder problems such as real robotics tasks. We leave these as potential avenues of future research.

6 Reproducibility
-----------------

In the supplemental material, we provide the source code and training scripts used to produce our results. In the README, we include documentation for setting up a Conda environment, running our training scripts, and visualizing our results. In addition, we provide pre-trained archives whose results were presented in this work. Detailed pseudocode and a list of relevant hyperparameters can be found in Appendices [A](https://arxiv.org/html/2305.13795v2#A1 "Appendix A PPGA Pseudocode ‣ Proximal Policy Gradient Arborescence for Quality Diversity Reinforcement Learning") and [B](https://arxiv.org/html/2305.13795v2#A2 "Appendix B Hyperparameters ‣ Proximal Policy Gradient Arborescence for Quality Diversity Reinforcement Learning").

References
----------

*   Akimoto et al. (2010) Youhei Akimoto, Yuichi Nagata, Isao Ono, and Shigenobu Kobayashi. Bidirectional relation between cma evolution strategies and natural evolution strategies. In Robert Schaefer, Carlos Cotta, Joanna Kołodziej, and Günter Rudolph (eds.), _Parallel Problem Solving from Nature, PPSN XI_, pp. 154–163, Berlin, Heidelberg, 2010. Springer Berlin Heidelberg. ISBN 978-3-642-15844-5. 
*   Batra et al. (2021) Sumeet Batra, Zhehui Huang, Aleksei Petrenko, Tushar Kumar, Artem Molchanov, and Gaurav S. Sukhatme. Decentralized control of quadrotor swarms with end-to-end deep reinforcement learning. In Aleksandra Faust, David Hsu, and Gerhard Neumann (eds.), _Conference on Robot Learning, 8-11 November 2021, London, UK_, volume 164 of _Proceedings of Machine Learning Research_, pp. 576–586. PMLR, 2021. URL [https://proceedings.mlr.press/v164/batra22a.html](https://proceedings.mlr.press/v164/batra22a.html). 
*   Chatzilygeroudis et al. (2021) Konstantinos Chatzilygeroudis, Antoine Cully, Vassilis Vassiliades, and Jean-Baptiste Mouret. _Quality-Diversity Optimization: A Novel Branch of Stochastic Optimization_, pp. 109–135. Springer International Publishing, Cham, 2021. ISBN 978-3-030-66515-9. doi: [10.1007/978-3-030-66515-9_4](https://arxiv.org/html/2305.13795v2/10.1007/978-3-030-66515-9_4). URL [https://doi.org/10.1007/978-3-030-66515-9_4](https://doi.org/10.1007/978-3-030-66515-9_4). 
*   Choi & Togelius (2021) Tae Jong Choi and Julian Togelius. Self-referential quality diversity through differential map-elites. In _Proceedings of the Genetic and Evolutionary Computation Conference_, GECCO ’21, pp. 502–509, New York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450383509. doi: [10.1145/3449639.3459383](https://arxiv.org/html/2305.13795v2/10.1145/3449639.3459383). URL [https://doi.org/10.1145/3449639.3459383](https://doi.org/10.1145/3449639.3459383). 
*   Colas et al. (2020) Cédric Colas, Vashisht Madhavan, Joost Huizinga, and Jeff Clune. Scaling map-elites to deep neuroevolution. In _Proceedings of the 2020 Genetic and Evolutionary Computation Conference_, GECCO ’20, pp. 67–75, New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450371285. doi: [10.1145/3377930.3390217](https://arxiv.org/html/2305.13795v2/10.1145/3377930.3390217). URL [https://doi.org/10.1145/3377930.3390217](https://doi.org/10.1145/3377930.3390217). 
*   Conti et al. (2018) Edoardo Conti, Vashisht Madhavan, Felipe Petroski Such, Joel Lehman, Kenneth Stanley, and Jeff Clune. Improving exploration in evolution strategies for deep reinforcement learning via a population of novelty-seeking agents. In S.Bengio, H.Wallach, H.Larochelle, K.Grauman, N.Cesa-Bianchi, and R.Garnett (eds.), _Advances in Neural Information Processing Systems 31_, pp. 5027–5038. Curran Associates, Inc., 2018. URL [http://papers.nips.cc/paper/7750-improving-exploration-in-evolution-strategies-for-deep-reinforcement-learning-via-a-population-of-novelty-seeking-agents.pdf](http://papers.nips.cc/paper/7750-improving-exploration-in-evolution-strategies-for-deep-reinforcement-learning-via-a-population-of-novelty-seeking-agents.pdf). 
*   Cully et al. (2015) Antoine Cully, Jeff Clune, Danesh Tarapore, and Jean-Baptiste Mouret. Robots that can adapt like animals. _Nat._, 521(7553):503–507, 2015. doi: [10.1038/nature14422](https://arxiv.org/html/2305.13795v2/10.1038/nature14422). URL [https://doi.org/10.1038/nature14422](https://doi.org/10.1038/nature14422). 
*   Earle et al. (2022) Sam Earle, Justin Snider, Matthew C. Fontaine, Stefanos Nikolaidis, and Julian Togelius. Illuminating diverse neural cellular automata for level generation. In _Proceedings of the Genetic and Evolutionary Computation Conference_, GECCO ’22, pp. 68–76, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450392372. doi: [10.1145/3512290.3528754](https://arxiv.org/html/2305.13795v2/10.1145/3512290.3528754). URL [https://doi.org/10.1145/3512290.3528754](https://doi.org/10.1145/3512290.3528754). 
*   Flageat et al. (2023) Manon Flageat, Felix Chalumeau, and Antoine Cully. Empirical analysis of pga-map-elites for neuroevolution in uncertain domains. _ACM Trans. Evol. Learn. Optim._, Jan 2023. ISSN 2688-299X. doi: [10.1145/3577203](https://arxiv.org/html/2305.13795v2/10.1145/3577203). URL [https://doi.org/10.1145/3577203](https://doi.org/10.1145/3577203). Just Accepted. 
*   Fontaine & Nikolaidis (2023) Matthew Fontaine and Stefanos Nikolaidis. Covariance matrix adaptation map-annealing. In _Proceedings of the Genetic and Evolutionary Computation Conference_, GECCO ’23, pp. 456–465, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9798400701191. doi: [10.1145/3583131.3590389](https://arxiv.org/html/2305.13795v2/10.1145/3583131.3590389). URL [https://doi.org/10.1145/3583131.3590389](https://doi.org/10.1145/3583131.3590389). 
*   Fontaine & Nikolaidis (2021) Matthew C. Fontaine and Stefanos Nikolaidis. Differentiable quality diversity. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan (eds.), _Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual_, pp. 10040–10052, 2021. URL [https://proceedings.neurips.cc/paper/2021/hash/532923f11ac97d3e7cb0130315b067dc-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/532923f11ac97d3e7cb0130315b067dc-Abstract.html). 
*   Fontaine et al. (2020) Matthew C. Fontaine, Julian Togelius, Stefanos Nikolaidis, and Amy K. Hoover. Covariance matrix adaptation for the rapid illumination of behavior space. In Carlos Artemio Coello Coello (ed.), _GECCO ’20: Genetic and Evolutionary Computation Conference, Cancún Mexico, July 8-12, 2020_, pp. 94–102. ACM, 2020. doi: [10.1145/3377930.3390232](https://arxiv.org/html/2305.13795v2/10.1145/3377930.3390232). URL [https://doi.org/10.1145/3377930.3390232](https://doi.org/10.1145/3377930.3390232). 
*   Fujimoto et al. (2018) Scott Fujimoto, Herke van Hoof, and David Meger. Addressing function approximation error in actor-critic methods. In Jennifer Dy and Andreas Krause (eds.), _Proceedings of the 35th International Conference on Machine Learning_, volume 80 of _proceedings of machine learning research_, pp. 1587–1596. pmlr, 10–15 jul 2018. URL [http://proceedings.mlr.press/v80/fujimoto18a.html](http://proceedings.mlr.press/v80/fujimoto18a.html). 
*   Gaier et al. (2018) Adam Gaier, Alexander Asteroth, and Jean-Baptiste Mouret. Data-Efficient Design Exploration through Surrogate-Assisted Illumination. _Evolutionary Computation_, 26(3):381–410, 09 2018. ISSN 1063-6560. doi: [10.1162/evco_a_00231](https://arxiv.org/html/2305.13795v2/10.1162/evco_a_00231). URL [https://doi.org/10.1162/evco_a_00231](https://doi.org/10.1162/evco_a_00231). 
*   Glasmachers et al. (2010) Tobias Glasmachers, Tom Schaul, Yi Sun, Daan Wierstra, and Jürgen Schmidhuber. Exponential natural evolution strategies. In Martin Pelikan and Jürgen Branke (eds.), _Genetic and Evolutionary Computation Conference, GECCO 2010, Proceedings, Portland, Oregon, USA, July 7-11, 2010_, pp. 393–400. ACM, 2010. doi: [10.1145/1830483.1830557](https://arxiv.org/html/2305.13795v2/10.1145/1830483.1830557). URL [https://doi.org/10.1145/1830483.1830557](https://doi.org/10.1145/1830483.1830557). 
*   Gravina et al. (2019) Daniele Gravina, Ahmed Khalifa, Antonios Liapis, Julian Togelius, and Georgios N Yannakakis. Procedural content generation through quality diversity. In _2019 IEEE Conference on Games (CoG)_, pp. 1–8. IEEE, 2019. 
*   Haarnoja et al. (2018) Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In Jennifer G. Dy and Andreas Krause (eds.), _Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018_, volume 80 of _Proceedings of Machine Learning Research_, pp. 1856–1865. PMLR, 2018. URL [http://proceedings.mlr.press/v80/haarnoja18b.html](http://proceedings.mlr.press/v80/haarnoja18b.html). 
*   Hagg et al. (2020) Alexander Hagg, Dominik Wilde, Alexander Asteroth, and Thomas Bäck. Designing air flow with surrogate-assisted phenotypic niching. In _International Conference on Parallel Problem Solving from Nature_, pp. 140–153. Springer, 2020. 
*   Handa et al. (2022) Ankur Handa, Arthur Allshire, Viktor Makoviychuk, Aleksei Petrenko, Ritvik Singh, Jingzhou Liu, Denys Makoviichuk, Karl Van Wyk, Alexander Zhurkevich, Balakumar Sundaralingam, Yashraj Narang, Jean-Francois Lafleche, Dieter Fox, and Gavriel State. Dextreme: Transfer of agile in-hand manipulation from simulation to reality. _arXiv_, 2022. 
*   Hansen (2016) Nikolaus Hansen. The CMA evolution strategy: A tutorial. _CoRR_, abs/1604.00772, 2016. URL [http://arxiv.org/abs/1604.00772](http://arxiv.org/abs/1604.00772). 
*   Huang et al. (2022) Shengyi Huang, Rousslan Fernand Julien Dossa, Chang Ye, Jeff Braga, Dipam Chakraborty, Kinal Mehta, and João G.M. Araújo. Cleanrl: High-quality single-file implementations of deep reinforcement learning algorithms. _Journal of Machine Learning Research_, 23(274):1–18, 2022. URL [http://jmlr.org/papers/v23/21-1342.html](http://jmlr.org/papers/v23/21-1342.html). 
*   Khalifa et al. (2018) Ahmed Khalifa, Scott Lee, Andy Nealen, and Julian Togelius. Talakat: Bullet hell generation through constrained map-elites. In _Proceedings of The Genetic and Evolutionary Computation Conference_, pp. 1047–1054, 2018. 
*   Lehman & Stanley (2011a) Joel Lehman and Kenneth O. Stanley. Abandoning Objectives: Evolution Through the Search for Novelty Alone. _Evolutionary Computation_, 19(2):189–223, 06 2011a. ISSN 1063-6560. doi: [10.1162/EVCO_a_00025](https://arxiv.org/html/2305.13795v2/10.1162/EVCO_a_00025). URL [https://doi.org/10.1162/EVCO_a_00025](https://doi.org/10.1162/EVCO_a_00025). 
*   Lehman & Stanley (2011b) Joel Lehman and Kenneth O. Stanley. Evolving a diversity of virtual creatures through novelty search and local competition. In _Proceedings of the 13th Annual Conference on Genetic and Evolutionary Computation_, GECCO ’11, pp. 211–218, New York, NY, USA, 2011b. Association for Computing Machinery. ISBN 9781450305570. doi: [10.1145/2001576.2001606](https://arxiv.org/html/2305.13795v2/10.1145/2001576.2001606). URL [https://doi.org/10.1145/2001576.2001606](https://doi.org/10.1145/2001576.2001606). 
*   Lillicrap et al. (2016) Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. In Yoshua Bengio and Yann LeCun (eds.), _4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings_, 2016. URL [http://arxiv.org/abs/1509.02971](http://arxiv.org/abs/1509.02971). 
*   Lim et al. (2022) Bryan Lim, Maxime Allard, Luca Grillotti, and Antoine Cully. Accelerated quality-diversity for robotics through massive parallelism. _arXiv preprint arXiv:2202.01258_, 2022. 
*   Makoviychuk et al. (2021) Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles Macklin, David Hoeller, Nikita Rudin, Arthur Allshire, Ankur Handa, and Gavriel State. Isaac gym: High performance GPU based physics simulation for robot learning. In Joaquin Vanschoren and Sai-Kit Yeung (eds.), _Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual_, 2021. URL [https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/28dd2c7955ce926456240b2ff0100bde-Abstract-round2.html](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/28dd2c7955ce926456240b2ff0100bde-Abstract-round2.html). 
*   Mnih et al. (2016) Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In Maria-Florina Balcan and Kilian Q. Weinberger (eds.), _Proceedings of the 33nd International Conference on Machine Learning, ICML 2016, New York City, NY, USA, June 19-24, 2016_, volume 48 of _JMLR Workshop and Conference Proceedings_, pp. 1928–1937. JMLR.org, 2016. URL [http://proceedings.mlr.press/v48/mniha16.html](http://proceedings.mlr.press/v48/mniha16.html). 
*   Morel et al. (2022) Aurélien Morel, Yakumo Kunimoto, Alex Coninx, and Stéphane Doncieux. Automatic acquisition of a repertoire of diverse grasping trajectories through behavior shaping and novelty search. In _2022 International Conference on Robotics and Automation (ICRA)_, pp. 755–761, 2022. doi: [10.1109/ICRA46639.2022.9811837](https://arxiv.org/html/2305.13795v2/10.1109/ICRA46639.2022.9811837). 
*   Morrison et al. (2020) Douglas Morrison, Peter Corke, and Jürgen Leitner. Egad! an evolved grasping analysis dataset for diversity and reproducibility in robotic manipulation. _IEEE Robotics and Automation Letters_, 5(3):4368–4375, 2020. doi: [10.1109/LRA.2020.2992195](https://arxiv.org/html/2305.13795v2/10.1109/LRA.2020.2992195). 
*   Mouret & Clune (2015) Jean-Baptiste Mouret and Jeff Clune. Illuminating search spaces by mapping elites. _CoRR_, abs/1504.04909, 2015. URL [http://arxiv.org/abs/1504.04909](http://arxiv.org/abs/1504.04909). 
*   Müller & Glasmachers (2018) Nils Müller and Tobias Glasmachers. Challenges in high-dimensional reinforcement learning with evolution strategies. In Anne Auger, Carlos M. Fonseca, Nuno Lourenço, Penousal Machado, Luís Paquete, and L.Darrell Whitley (eds.), _Parallel Problem Solving from Nature - PPSN XV - 15th International Conference, Coimbra, Portugal, September 8-12, 2018, Proceedings, Part II_, volume 11102 of _Lecture Notes in Computer Science_, pp. 411–423. Springer, 2018. doi: [10.1007/978-3-319-99259-4_33](https://arxiv.org/html/2305.13795v2/10.1007/978-3-319-99259-4_33). URL [https://doi.org/10.1007/978-3-319-99259-4_33](https://doi.org/10.1007/978-3-319-99259-4_33). 
*   Nilsson & Cully (2021) Olle Nilsson and Antoine Cully. Policy gradient assisted map-elites. In _Proceedings of the Genetic and Evolutionary Computation Conference_, GECCO ’21, pp. 866–875, New York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450383509. doi: [10.1145/3449639.3459304](https://arxiv.org/html/2305.13795v2/10.1145/3449639.3459304). URL [https://doi.org/10.1145/3449639.3459304](https://doi.org/10.1145/3449639.3459304). 
*   Pierrot & Flajolet (2023) Thomas Pierrot and Arthur Flajolet. Evolving populations of diverse RL agents with map-elites. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_. OpenReview.net, 2023. URL [https://openreview.net/pdf?id=CBfYffLqWqb](https://openreview.net/pdf?id=CBfYffLqWqb). 
*   Pierrot et al. (2022) Thomas Pierrot, Valentin Macé, Felix Chalumeau, Arthur Flajolet, Geoffrey Cideron, Karim Beguir, Antoine Cully, Olivier Sigaud, and Nicolas Perrin-Gilbert. Diversity policy gradient for sample efficient quality-diversity optimization. In _Proceedings of the Genetic and Evolutionary Computation Conference_, GECCO ’22, pp. 1075–1083, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450392372. doi: [10.1145/3512290.3528845](https://arxiv.org/html/2305.13795v2/10.1145/3512290.3528845). URL [https://doi.org/10.1145/3512290.3528845](https://doi.org/10.1145/3512290.3528845). 
*   Rudin et al. (2021) Nikita Rudin, David Hoeller, Philipp Reist, and Marco Hutter. Learning to walk in minutes using massively parallel deep reinforcement learning. In Aleksandra Faust, David Hsu, and Gerhard Neumann (eds.), _Conference on Robot Learning, 8-11 November 2021, London, UK_, volume 164 of _Proceedings of Machine Learning Research_, pp. 91–100. PMLR, 2021. URL [https://proceedings.mlr.press/v164/rudin22a.html](https://proceedings.mlr.press/v164/rudin22a.html). 
*   Schulman et al. (2015) John Schulman, Sergey Levine, Pieter Abbeel, Michael I. Jordan, and Philipp Moritz. Trust region policy optimization. In Francis R. Bach and David M. Blei (eds.), _Proceedings of the 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July 2015_, volume 37 of _JMLR Workshop and Conference Proceedings_, pp. 1889–1897. JMLR.org, 2015. URL [http://proceedings.mlr.press/v37/schulman15.html](http://proceedings.mlr.press/v37/schulman15.html). 
*   Schulman et al. (2016) John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. In Yoshua Bengio and Yann LeCun (eds.), _4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings_, 2016. URL [http://arxiv.org/abs/1506.02438](http://arxiv.org/abs/1506.02438). 
*   Schulman et al. (2017) John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. _CoRR_, abs/1707.06347, 2017. URL [http://arxiv.org/abs/1707.06347](http://arxiv.org/abs/1707.06347). 
*   Tjanaka et al. (2022a) Bryon Tjanaka, Matthew C. Fontaine, Aniruddha Kalkar, and Stefanos Nikolaidis. Training diverse high-dimensional controllers by scaling covariance matrix adaptation map-annealing, 2022a. 
*   Tjanaka et al. (2022b) Bryon Tjanaka, Matthew C. Fontaine, Julian Togelius, and Stefanos Nikolaidis. Approximating gradients for differentiable quality diversity in reinforcement learning. In _Proceedings of the Genetic and Evolutionary Computation Conference_, GECCO ’22, pp. 1102–1111, New York, NY, USA, 2022b. Association for Computing Machinery. ISBN 9781450392372. doi: [10.1145/3512290.3528705](https://arxiv.org/html/2305.13795v2/10.1145/3512290.3528705). URL [https://doi.org/10.1145/3512290.3528705](https://doi.org/10.1145/3512290.3528705). 
*   Tjanaka et al. (2023) Bryon Tjanaka, Matthew C Fontaine, David H Lee, Yulun Zhang, Nivedit Reddy Balam, Nathaniel Dennler, Sujay S Garlanka, Nikitas Dimitri Klapsis, and Stefanos Nikolaidis. Pyribs: A bare-bones python library for quality diversity optimization. In _Proceedings of the Genetic and Evolutionary Computation Conference_, GECCO ’23, pp. 220–229, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9798400701191. doi: [10.1145/3583131.3590374](https://arxiv.org/html/2305.13795v2/10.1145/3583131.3590374). URL [https://doi.org/10.1145/3583131.3590374](https://doi.org/10.1145/3583131.3590374). 
*   Todorov et al. (2012) Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In _2012 IEEE/RSJ International Conference on Intelligent Robots and Systems_, pp. 5026–5033. IEEE, 2012. doi: [10.1109/IROS.2012.6386109](https://arxiv.org/html/2305.13795v2/10.1109/IROS.2012.6386109). 
*   Vassiliades & Mouret (2018) Vassilis Vassiliades and Jean-Baptiste Mouret. Discovering the elite hypervolume by leveraging interspecies correlation. In _Proceedings of the Genetic and Evolutionary Computation Conference_, GECCO ’18, pp. 149–156, New York, NY, USA, 2018. Association for Computing Machinery. ISBN 9781450356183. doi: [10.1145/3205455.3205602](https://arxiv.org/html/2305.13795v2/10.1145/3205455.3205602). URL [https://doi.org/10.1145/3205455.3205602](https://doi.org/10.1145/3205455.3205602). 
*   Vassiliades et al. (2016) Vassilis Vassiliades, Konstantinos Chatzilygeroudis, and Jean-Baptiste Mouret. Using centroidal voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm. _IEEE Transactions on Evolutionary Computation_, 22:623–630, 2016. URL [https://api.semanticscholar.org/CorpusID:23453919](https://api.semanticscholar.org/CorpusID:23453919). 

Appendix A PPGA Pseudocode
--------------------------

Algorithm 1 Proximal Policy Gradient Arborescence

Input: Initial policy

θ 0 subscript 𝜃 0\theta_{0}italic_θ start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT
, VPPO instance to approximate

∇f,∇𝐦∇𝑓∇𝐦\nabla f,\nabla\textbf{m}∇ italic_f , ∇ m
and move the search policy, number of QD iterations

N Q subscript 𝑁 𝑄 N_{Q}italic_N start_POSTSUBSCRIPT italic_Q end_POSTSUBSCRIPT
, number of VPPO iterations to estimate the objective-measure functions and gradients

N 1 subscript 𝑁 1 N_{1}italic_N start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT
, number of VPPO iterations to move the search policy

N 2 subscript 𝑁 2 N_{2}italic_N start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT
, branching population size

λ 𝜆\lambda italic_λ
, and an initial step size for xNES

σ g subscript 𝜎 𝑔\sigma_{g}italic_σ start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT

Initialize the search policy

θ μ=θ 0 subscript 𝜃 𝜇 subscript 𝜃 0\theta_{\mu}=\theta_{0}italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT = italic_θ start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT
. Initialize NES parameters

μ,Σ=σ g⁢I 𝜇 Σ subscript 𝜎 𝑔 𝐼\mu,\Sigma=\sigma_{g}I italic_μ , roman_Σ = italic_σ start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT italic_I

for iter

←←\leftarrow←
1 to N do

f,∇f,𝐦,∇𝐦←V⁢P⁢P⁢O.c⁢o⁢m⁢p⁢u⁢t⁢e⁢_⁢j⁢a⁢c⁢o⁢b⁢i⁢a⁢n⁢(θ μ,f⁢(⋅),𝐦⁢(⋅),N 1)formulae-sequence←𝑓∇𝑓 𝐦∇𝐦 𝑉 𝑃 𝑃 𝑂 𝑐 𝑜 𝑚 𝑝 𝑢 𝑡 𝑒 _ 𝑗 𝑎 𝑐 𝑜 𝑏 𝑖 𝑎 𝑛 subscript 𝜃 𝜇 𝑓⋅𝐦⋅subscript 𝑁 1 f,\nabla f,\textbf{m},\nabla\textbf{m}\leftarrow VPPO.compute\_jacobian(\theta% _{\mu},f(\cdot),\textbf{m}(\cdot),N_{1})italic_f , ∇ italic_f , m , ∇ m ← italic_V italic_P italic_P italic_O . italic_c italic_o italic_m italic_p italic_u italic_t italic_e _ italic_j italic_a italic_c italic_o italic_b italic_i italic_a italic_n ( italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT , italic_f ( ⋅ ) , m ( ⋅ ) , italic_N start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT )

∇f←normalize⁢(∇f),∇𝒎←normalize⁢(∇𝒎)formulae-sequence←∇𝑓 normalize∇𝑓←∇𝒎 normalize∇𝒎\nabla f\leftarrow\text{normalize}(\nabla f),\nabla\textbf{{m}}\leftarrow\text% {normalize}(\nabla\textbf{{m}})∇ italic_f ← normalize ( ∇ italic_f ) , ∇ m ← normalize ( ∇ m )

_

←←\leftarrow←
update _archive(

θ μ,f,𝒎 subscript 𝜃 𝜇 𝑓 𝒎\theta_{\mu},f,\textbf{{m}}italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT , italic_f , m
)

for

i←1⁢𝐭𝐨⁢λ←𝑖 1 𝐭𝐨 𝜆 i\leftarrow 1\;\textbf{to}\;\lambda italic_i ← 1 to italic_λ
do

c∼𝒩⁢(μ,Σ)similar-to 𝑐 𝒩 𝜇 Σ c\sim\mathcal{N}(\mu,\Sigma)italic_c ∼ caligraphic_N ( italic_μ , roman_Σ )
// sample gradient coefficients

∇i←c 0⁢∇f+∑j=1 k c j⁢∇m j←subscript∇𝑖 subscript 𝑐 0∇𝑓 superscript subscript 𝑗 1 𝑘 subscript 𝑐 𝑗∇subscript 𝑚 𝑗\nabla_{i}\leftarrow c_{0}\nabla f+\sum_{j=1}^{k}c_{j}\nabla m_{j}∇ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ← italic_c start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT ∇ italic_f + ∑ start_POSTSUBSCRIPT italic_j = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT italic_c start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ∇ italic_m start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT

θ i′←θ μ+∇i←superscript subscript 𝜃 𝑖′subscript 𝜃 𝜇 subscript∇𝑖\theta_{i}^{{}^{\prime}}\leftarrow\theta_{\mu}+\nabla_{i}italic_θ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT start_FLOATSUPERSCRIPT ′ end_FLOATSUPERSCRIPT end_POSTSUPERSCRIPT ← italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT + ∇ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT

f′,*,m′,*←rollout(θ i′)f^{\prime},*,m^{\prime},*\leftarrow\text{rollout}(\theta^{\prime}_{i})italic_f start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT , * , italic_m start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT , * ← rollout ( italic_θ start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT )

Δ i←update←subscript Δ 𝑖 update\Delta_{i}\leftarrow\text{update}roman_Δ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ← update
_

archive⁢(θ i′,f′,m’)archive superscript subscript 𝜃 𝑖′superscript 𝑓′m’\text{archive}(\theta_{i}^{{}^{\prime}},f^{\prime},\textbf{{m'}})archive ( italic_θ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT start_FLOATSUPERSCRIPT ′ end_FLOATSUPERSCRIPT end_POSTSUPERSCRIPT , italic_f start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT , m’ )

end for

rank gradient coefficients

∇i subscript∇𝑖\nabla_{i}∇ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT
by archive improvement

Δ i subscript Δ 𝑖\Delta_{i}roman_Δ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT

Adapt xNES parameters

μ=μ′,Σ=Σ′formulae-sequence 𝜇 superscript 𝜇′Σ superscript Σ′\mu=\mu^{\prime},\Sigma=\Sigma^{\prime}italic_μ = italic_μ start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT , roman_Σ = roman_Σ start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT
based on improvement ranking

Δ i subscript Δ 𝑖\Delta_{i}roman_Δ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT

f′⁢(θ μ)=c μ,0⁢f+∑j=1 k c μ,j⁢m j superscript 𝑓′subscript 𝜃 𝜇 subscript 𝑐 𝜇 0 𝑓 superscript subscript 𝑗 1 𝑘 subscript 𝑐 𝜇 𝑗 subscript 𝑚 𝑗 f^{\prime}(\theta_{\mu})=c_{\mu,0}f+\sum_{j=1}^{k}c_{\mu,j}m_{j}italic_f start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ( italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT ) = italic_c start_POSTSUBSCRIPT italic_μ , 0 end_POSTSUBSCRIPT italic_f + ∑ start_POSTSUBSCRIPT italic_j = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT italic_c start_POSTSUBSCRIPT italic_μ , italic_j end_POSTSUBSCRIPT italic_m start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT
, where

𝐜 μ=μ′subscript 𝐜 𝜇 superscript 𝜇′\textbf{c}_{\mu}=\mu^{\prime}c start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT = italic_μ start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT
// construct multi-objective reward function

θ μ′=V⁢P⁢P⁢O.t⁢r⁢a⁢i⁢n⁢(θ μ,f′,N 2)formulae-sequence superscript subscript 𝜃 𝜇′𝑉 𝑃 𝑃 𝑂 𝑡 𝑟 𝑎 𝑖 𝑛 subscript 𝜃 𝜇 superscript 𝑓′subscript 𝑁 2\theta_{\mu}^{\prime}=VPPO.train(\theta_{\mu},f^{\prime},N_{2})italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT = italic_V italic_P italic_P italic_O . italic_t italic_r italic_a italic_i italic_n ( italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT , italic_f start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT , italic_N start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT )
// standard PPO training procedure

if there is no change in the archive then

Restart xNES with

μ=0,Σ=σ g⁢I formulae-sequence 𝜇 0 Σ subscript 𝜎 𝑔 𝐼\mu=0,\Sigma=\sigma_{g}I italic_μ = 0 , roman_Σ = italic_σ start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT italic_I

Set

θ μ subscript 𝜃 𝜇\theta_{\mu}italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT
to a randomly selected existing cell

θ i subscript 𝜃 𝑖\theta_{i}italic_θ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT
from the archive

end if

end for

Algorithm 2 Update Archive

Input: Solution

θ 𝜃\theta italic_θ
to insert, episodic reward

f 𝑓 f italic_f
, measures

𝐦=<m 1,…,m k>\textbf{m}=<m_{1},...,m_{k}>m = < italic_m start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , … , italic_m start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT >
, archive

𝒜 𝒜\mathcal{A}caligraphic_A
, archive learning rate

α 𝛼\alpha italic_α

θ i⁢n⁢c,f i⁢n⁢c subscript 𝜃 𝑖 𝑛 𝑐 subscript 𝑓 𝑖 𝑛 𝑐\theta_{inc},f_{inc}italic_θ start_POSTSUBSCRIPT italic_i italic_n italic_c end_POSTSUBSCRIPT , italic_f start_POSTSUBSCRIPT italic_i italic_n italic_c end_POSTSUBSCRIPT
=

𝒜 𝒜\mathcal{A}caligraphic_A
[m] if

𝒜 𝒜\mathcal{A}caligraphic_A
[m] is nonempty else

N⁢o⁢n⁢e,0 𝑁 𝑜 𝑛 𝑒 0 None,0 italic_N italic_o italic_n italic_e , 0
// incumbent policy

Δ i=0 subscript Δ 𝑖 0\Delta_{i}=0 roman_Δ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = 0

if

f>f i⁢n⁢c 𝑓 subscript 𝑓 𝑖 𝑛 𝑐 f>f_{inc}italic_f > italic_f start_POSTSUBSCRIPT italic_i italic_n italic_c end_POSTSUBSCRIPT
then

insert

θ 𝜃\theta italic_θ
into cell

𝒜 𝒜\mathcal{A}caligraphic_A
[m]

f i⁢n⁢c←(1−α)⁢f i⁢n⁢c+α⁢f←subscript 𝑓 𝑖 𝑛 𝑐 1 𝛼 subscript 𝑓 𝑖 𝑛 𝑐 𝛼 𝑓 f_{inc}\leftarrow(1-\alpha)f_{inc}+\alpha f italic_f start_POSTSUBSCRIPT italic_i italic_n italic_c end_POSTSUBSCRIPT ← ( 1 - italic_α ) italic_f start_POSTSUBSCRIPT italic_i italic_n italic_c end_POSTSUBSCRIPT + italic_α italic_f

Δ i=f−f i⁢n⁢c subscript Δ 𝑖 𝑓 subscript 𝑓 𝑖 𝑛 𝑐\Delta_{i}=f-f_{inc}roman_Δ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = italic_f - italic_f start_POSTSUBSCRIPT italic_i italic_n italic_c end_POSTSUBSCRIPT

end if

return

Δ i subscript Δ 𝑖\Delta_{i}roman_Δ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT

Algorithm 3 Vectorized-PPO (VPPO)

Input: Initial search policy

π θ i subscript 𝜋 subscript 𝜃 𝑖\pi_{\theta_{i}}italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT
, objective functions to optimize

𝐟=f 1⁢(⋅),…,f k⁢(⋅)𝐟 subscript 𝑓 1⋅…subscript 𝑓 𝑘⋅\mathbf{f}=f_{1}(\cdot),...,f_{k}(\cdot)bold_f = italic_f start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ( ⋅ ) , … , italic_f start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ( ⋅ )
, number of VPPO iterations

N 𝑁 N italic_N
, number of parallel environments

E 𝐸 E italic_E
, rollout length

L 𝐿 L italic_L

Initialize the vectorized agent

𝐯⁢π θ 𝐢 𝐯 subscript 𝜋 subscript 𝜃 𝐢\mathbf{{\bm{v}}{\pi}_{\theta_{i}}}bold_v italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT bold_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT
= vectorized_agent(

[π θ]×(k+1)delimited-[]subscript 𝜋 𝜃 𝑘 1[\pi_{\theta}]\times(k+1)[ italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ] × ( italic_k + 1 )
)

for iter

←←\leftarrow←
1 to

N 1 subscript 𝑁 1 N_{1}italic_N start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT
do

(S, A, R, S’)←←(S, A, R, S’)absent\textbf{(S, A, R, S')}\leftarrow(S, A, R, S’) ←
rollout(vectorized_agent,

E 𝐸 E italic_E
,

L 𝐿 L italic_L
,

𝐟 𝐟\mathbf{f}bold_f
) // Note that S =

{S 1,…,S k}subscript 𝑆 1…subscript 𝑆 𝑘\{S_{1},...,S_{k}\}{ italic_S start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , … , italic_S start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT }
, etc

advantage

𝔸 𝔸\mathbb{A}blackboard_A
, returns

𝒢 𝒢\mathcal{G}caligraphic_G←←\leftarrow←
batch_calculate_rewards(S, A, R, S’,

𝐟 𝐟\mathbf{f}bold_f
)

𝐯⁢π θ′𝐯 superscript subscript 𝜋 𝜃′\mathbf{{\bm{v}}{\pi}_{\theta}}^{\prime}bold_v italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT←←\leftarrow←
batch_gradient_descent(

𝔸,𝒢,𝐯⁢π θ 𝔸 𝒢 𝐯 subscript 𝜋 𝜃\mathbb{A},\mathcal{G},\mathbf{{\bm{v}}{\pi}_{\theta}}blackboard_A , caligraphic_G , bold_v italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT
) // using the stochastic policy gradient

𝐯⁢π θ←𝐯⁢π θ′←𝐯 subscript 𝜋 𝜃 𝐯 superscript subscript 𝜋 𝜃′\mathbf{{\bm{v}}{\pi}_{\theta}}\leftarrow\mathbf{{\bm{v}}{\pi}_{\theta}}^{\prime}bold_v italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ← bold_v italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT

end for

∇𝐟←←∇𝐟 absent\mathbf{\nabla f}\leftarrow∇ bold_f ←𝐯⁢π θ′−𝐯⁢π θ 𝐢 𝐯 superscript subscript 𝜋 𝜃′𝐯 subscript 𝜋 subscript 𝜃 𝐢\mathbf{{\bm{v}}{\pi}_{\theta}}^{\prime}-\mathbf{{\bm{v}}{\pi}_{\theta_{i}}}bold_v italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT - bold_v italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT bold_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT

return

∇𝐟∇𝐟\mathbf{\nabla f}∇ bold_f

Appendix B Hyperparameters
--------------------------

Table 1: List of relevant hyperparameters for PPGA shared across all environments.

Hyperparameter Value
Actor Network[128, 128, Action Dim]
Critic Network[256, 256, 1]
N 1 subscript 𝑁 1 N_{1}italic_N start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT 10
N 2 subscript 𝑁 2 N_{2}italic_N start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT 10
PPO Num Minibatches 8
PPO Num Epochs 4
Observation Normalization True
Reward Normalization True
Rollout Length 128

Appendix C Ablation Against CMA-MAEGA
-------------------------------------

PPGA makes two key changes compared to standard DQD algorithms such as CMA-MAEGA: Walking the search policy with VPPO vs. weighted linear recombination of the gradients and replacing xNES. We compare walking the search policy with VPPO to using gradient recombination in the original formulation of CMA-MEGA. In addition, we ran an ablation using xNES as the outer-loop optimizer compared to CMA-ES. However, the step-size adaptation parameter quickly diverges with CMA-ES and destabilizes training, and thus were unable to provide plots for this ablation. Fig.[7](https://arxiv.org/html/2305.13795v2#A3.F7 "Figure 7 ‣ Appendix C Ablation Against CMA-MAEGA ‣ Proximal Policy Gradient Arborescence for Quality Diversity Reinforcement Learning") shows the comparison of walking the search policy with VPPO versus using weighted linear recombination of the gradients. We believe that the large gap in performance is due to the fact that we can take multiple steps with VPPO via the N 2 subscript 𝑁 2 N_{2}italic_N start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT hyperparameter. Weighted recombination results in a single gradient step where the updated search policy may land too close to the previous search policy’s cell. When we branch from the updated search policy, many-branched agents will fall into overlapping cells, resulting in small archive improvement, which can lead to the emitter prematurely leaving a high-performing region of the search space.

![Image 7: Refer to caption](https://arxiv.org/html/2305.13795v2/x7.png)

Figure 7: Ablation of walking the search policy with VPPO vs gradient recombination done in CMA-MEGA/CMA-MAEGA on the Humanoid environment.

Appendix D TD3GA Implementation Details
---------------------------------------

As part of our ablation study, we implemented TD3GA, which differs from PPGA by replacing all PPO mechanisms with TD3(Fujimoto et al., [2018](https://arxiv.org/html/2305.13795v2#bib.bib13)). This implementation required several algorithmic decisions, as PPGA was originally designed to integrate with an on-policy method like PPO rather than an off-policy method like TD3, In this section, we describe these decisions. In general, we intend our decisions to make TD3GA operate as closely as possible to PPGA, and we leave it to future work to explore whether variations of these decisions will further improve performance.

Background: TD3 is an off-policy actor-critic method designed for single-objective RL tasks. TD3 maintains an actor (i.e., a policy) that takes actions in the environment and a critic that estimates the action-value function. TD3 also maintains a replay buffer that stores experiences collected by the actor. Over time, the actor is trained to optimize the critic. Simultaneously, based on experience in the replay buffer, the critic learns to better predict the action-value function.

Design Decisions:

1.   1.Number of critics: In TD3GA, we maintain a TD3 critic for the objective function and one for each of the measure functions. We also create a separate critic for the weighted objective that is used when walking the search policy. We refer to these critics as the objective critic, measure critics, and walking critic. 
2.   2.Choice of actor for critic training: When training the critic, TD3 requires an actor that generates actions for states sampled from the replay buffer. Prior QD-RL methods that estimate objective gradients with TD3, e.g., PGA-ME(Nilsson & Cully, [2021](https://arxiv.org/html/2305.13795v2#bib.bib33)) and CMA-MEGA (TD3, ES)(Tjanaka et al., [2022b](https://arxiv.org/html/2305.13795v2#bib.bib41)), fulfill this role with a dedicated actor. This actor is referred to as a greedy actor since its primary purpose is to optimize its performance with respect to the critic. In theory, since TD3 is an off-policy method, any actor, including a greedy actor, can be used to train the critic. However, to make TD3GA closer to PPGA, we instead use a copy of the current search policy to train the critic. This decision provides our TD3 instances with an on-policy nature that mirrors the PPO mechanisms found in PPGA. 
3.   3.When to train critics: Similar to the PPO value functions in PPGA, we maintain all critics throughout the entire training run, updating them on every iteration. 
4.   4.Experience collection: This decision concerns which experience collected in the environment should be inserted into the replay buffer. With the settings in our paper, PPGA (and TD3GA) samples 300 policies per iteration, and each policy is evaluated for 10 episodes, with each episode having up to 1,000 timesteps of experience; in total, these policies generate 3 million timesteps per iteration. The typical replay buffer size(Fujimoto et al., [2018](https://arxiv.org/html/2305.13795v2#bib.bib13)) in TD3 is 1 million, meaning the buffer would be filled three times over if we inserted all of this experience, i.e., the critic would never be trained with the experience from two-thirds of all policies. To ensure that we collect experience from many policies across many iterations, we make two decisions. First, we only collect one episode of experience from each agent — this already cuts down experience collected on each iteration to 300,000 timesteps. Second, we increase the replay buffer size to 5 million to store experience across more iterations. Note that there is no analog to the replay buffer in PPGA since PPO is an on-policy method. Instead, PPO regresses the value function based on experience collected while evaluating the policy. 
5.   5.Gradient computation: We begin by describing how a prior method estimates the objective gradient with TD3. To estimate the objective gradient of a policy, CMA-MEGA (TD3, ES) samples a batch of experience and passes the batch through the corresponding actor. The actions outputted by the actor are then inputted to the critic. Backgpropagating through the critic and the actor then provides the objective gradient. Roughly, the above procedure corresponds to taking a single step of TD3. However, in PPGA, the gradient is computed by taking the difference after multiple steps of PPO (the number of steps is determined by the hyperparameters N 1 subscript 𝑁 1 N_{1}italic_N start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT and N 2 subscript 𝑁 2 N_{2}italic_N start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT. The straightforward approach to mirror this behavior in TD3GA is to also output a gradient after several steps of TD3. Thus, when training each critic, we also track the final state of the actor (recall that the actor used in critic training is a copy of the search policy). At the end of critic training, our gradient is then the difference between the final actor and the original search policy. For example, to compute the objective gradient, we train the objective critic with an actor that is a copy of the search policy. While training the critic, the actor is updated so that it maximizes the objective critic. After N 1 subscript 𝑁 1 N_{1}italic_N start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT steps, we compute the objective gradient as the difference between the actor and the original search policy. 
6.   6.Actor target networks: One TD3 mechanism that improved stability and performance was target networks, which are slowly updating versions of the actor and critic parameters. In TD3GA, it is straightforward to apply this mechanism to the critic parameters. However, since the actor is reset to the current search policy on every iteration, it is difficult to maintain a single target network. Thus, on every iteration, the target network for the actor is simply reset to the current search policy’s parameters before the gradient computation. 

Pseudocode: Algorithm[4](https://arxiv.org/html/2305.13795v2#alg4 "Algorithm 4 ‣ Appendix D TD3GA Implementation Details ‣ Proximal Policy Gradient Arborescence for Quality Diversity Reinforcement Learning") shows the process for computing an objective gradient in TD3GA. The same process applies to computing measure gradients. The process for walking the search policy is also similar, except that the reward is a weighted combination of the objective and measures (the weights come from the mean μ 𝜇\mu italic_μ of the emitter’s coefficient distribution). Furthermore, when walking the search policy, we return the final actor instead of a gradient.

Algorithm 4 TD3 Gradient Computation for the Objective. Adapted from TD3(Fujimoto et al., [2018](https://arxiv.org/html/2305.13795v2#bib.bib13))

Input: Current search policy params

θ μ subscript 𝜃 𝜇\theta_{\mu}italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT
, current TD3 critic networks

Q ψ 1 subscript 𝑄 subscript 𝜓 1 Q_{\psi_{1}}italic_Q start_POSTSUBSCRIPT italic_ψ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT end_POSTSUBSCRIPT
and

Q ψ 2 subscript 𝑄 subscript 𝜓 2 Q_{\psi_{2}}italic_Q start_POSTSUBSCRIPT italic_ψ start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT end_POSTSUBSCRIPT
parameterized by

ψ 1 subscript 𝜓 1\psi_{1}italic_ψ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT
and

ψ 2 subscript 𝜓 2\psi_{2}italic_ψ start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT
respectively, current critic targets

ψ 1′subscript superscript 𝜓′1\psi^{\prime}_{1}italic_ψ start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT
and

ψ 2′subscript superscript 𝜓′2\psi^{\prime}_{2}italic_ψ start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT
, replay buffer

B 𝐵 B italic_B

Hyperparameters: Training steps

N 1 subscript 𝑁 1 N_{1}italic_N start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT
or

N 2 subscript 𝑁 2 N_{2}italic_N start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT
, target network update rate

τ 𝜏\tau italic_τ
, target network update frequency

d 𝑑 d italic_d
, smoothing noise standard deviation

σ p subscript 𝜎 𝑝\sigma_{p}italic_σ start_POSTSUBSCRIPT italic_p end_POSTSUBSCRIPT
, smoothing noise clip

c 𝑐 c italic_c
, discount factor

γ 𝛾\gamma italic_γ
, batch size

n b⁢a⁢t⁢c⁢h subscript 𝑛 𝑏 𝑎 𝑡 𝑐 ℎ n_{batch}italic_n start_POSTSUBSCRIPT italic_b italic_a italic_t italic_c italic_h end_POSTSUBSCRIPT

Initialize actor

ϕ=θ μ italic-ϕ subscript 𝜃 𝜇\phi=\theta_{\mu}italic_ϕ = italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT
, actor target

ϕ′=θ μ superscript italic-ϕ′subscript 𝜃 𝜇\phi^{\prime}=\theta_{\mu}italic_ϕ start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT = italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT

{Either N 1 subscript 𝑁 1 N_{1}italic_N start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT for gradient computation or N 2 subscript 𝑁 2 N_{2}italic_N start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT for walking the search policy}

for iter

←←\leftarrow←
1 to

N 1 subscript 𝑁 1 N_{1}italic_N start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT
do

{

r 𝑟 r italic_r
is replaced with the measures for measure gradients, or a weighted combination of the reward and measures for walking the search policy}

Sample mini-batch of

n b⁢a⁢t⁢c⁢h subscript 𝑛 𝑏 𝑎 𝑡 𝑐 ℎ n_{batch}italic_n start_POSTSUBSCRIPT italic_b italic_a italic_t italic_c italic_h end_POSTSUBSCRIPT
transitions

(s,a,r,s′)𝑠 𝑎 𝑟 superscript 𝑠′(s,a,r,s^{\prime})( italic_s , italic_a , italic_r , italic_s start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT )
from

B 𝐵 B italic_B

{Train the critics}

a~←π ϕ′⁢(s′)+ϵ←~𝑎 subscript 𝜋 superscript italic-ϕ′superscript 𝑠′italic-ϵ\tilde{a}\leftarrow\pi_{\phi^{\prime}}(s^{\prime})+\epsilon over~ start_ARG italic_a end_ARG ← italic_π start_POSTSUBSCRIPT italic_ϕ start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT end_POSTSUBSCRIPT ( italic_s start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ) + italic_ϵ
,

ϵ∼clip⁢(𝒩⁢(0,σ p),−c,c)similar-to italic-ϵ clip 𝒩 0 subscript 𝜎 𝑝 𝑐 𝑐\epsilon\sim\text{clip}(\mathcal{N}(0,\sigma_{p}),-c,c)italic_ϵ ∼ clip ( caligraphic_N ( 0 , italic_σ start_POSTSUBSCRIPT italic_p end_POSTSUBSCRIPT ) , - italic_c , italic_c )

y←r+γ⁢min i=1,2⁡Q ψ i′⁢(s′,a~)←𝑦 𝑟 𝛾 subscript 𝑖 1 2 subscript 𝑄 subscript superscript 𝜓′𝑖 superscript 𝑠′~𝑎 y\leftarrow r+\gamma\min_{i=1,2}Q_{\psi^{\prime}_{i}}(s^{\prime},\tilde{a})italic_y ← italic_r + italic_γ roman_min start_POSTSUBSCRIPT italic_i = 1 , 2 end_POSTSUBSCRIPT italic_Q start_POSTSUBSCRIPT italic_ψ start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( italic_s start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT , over~ start_ARG italic_a end_ARG )

{This update is performed with Adam}

Update critics

ψ i←arg⁢min ψ i⁡1 N⁢∑(y−Q ψ i⁢(s,a))2←subscript 𝜓 𝑖 subscript arg min subscript 𝜓 𝑖 1 𝑁 superscript 𝑦 subscript 𝑄 subscript 𝜓 𝑖 𝑠 𝑎 2\psi_{i}\leftarrow\operatorname*{arg\,min}_{\psi_{i}}\frac{1}{N}\sum(y-Q_{\psi% _{i}}(s,a))^{2}italic_ψ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ← start_OPERATOR roman_arg roman_min end_OPERATOR start_POSTSUBSCRIPT italic_ψ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT divide start_ARG 1 end_ARG start_ARG italic_N end_ARG ∑ ( italic_y - italic_Q start_POSTSUBSCRIPT italic_ψ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( italic_s , italic_a ) ) start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT

if

t mod d modulo 𝑡 𝑑 t\mod d italic_t roman_mod italic_d
then

{Update ϕ italic-ϕ\phi italic_ϕ by the deterministic policy gradient with Adam}

∇ϕ J⁢(ϕ)=1 n b⁢a⁢t⁢c⁢h⁢∑∇a Q ψ 1⁢(s,a)|a=π ϕ⁢(s)⁢∇ϕ π ϕ⁢(s)subscript∇italic-ϕ 𝐽 italic-ϕ evaluated-at 1 subscript 𝑛 𝑏 𝑎 𝑡 𝑐 ℎ subscript∇𝑎 subscript 𝑄 subscript 𝜓 1 𝑠 𝑎 𝑎 subscript 𝜋 italic-ϕ 𝑠 subscript∇italic-ϕ subscript 𝜋 italic-ϕ 𝑠\nabla_{\phi}J(\phi)=\frac{1}{n_{batch}}\sum\nabla_{a}Q_{\psi_{1}}(s,a)|_{a=% \pi_{\phi}(s)}\nabla_{\phi}\pi_{\phi}(s)∇ start_POSTSUBSCRIPT italic_ϕ end_POSTSUBSCRIPT italic_J ( italic_ϕ ) = divide start_ARG 1 end_ARG start_ARG italic_n start_POSTSUBSCRIPT italic_b italic_a italic_t italic_c italic_h end_POSTSUBSCRIPT end_ARG ∑ ∇ start_POSTSUBSCRIPT italic_a end_POSTSUBSCRIPT italic_Q start_POSTSUBSCRIPT italic_ψ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( italic_s , italic_a ) | start_POSTSUBSCRIPT italic_a = italic_π start_POSTSUBSCRIPT italic_ϕ end_POSTSUBSCRIPT ( italic_s ) end_POSTSUBSCRIPT ∇ start_POSTSUBSCRIPT italic_ϕ end_POSTSUBSCRIPT italic_π start_POSTSUBSCRIPT italic_ϕ end_POSTSUBSCRIPT ( italic_s )

{Update target networks:}

ψ i′←τ⁢ψ i+(1−τ)⁢ψ i′←subscript superscript 𝜓′𝑖 𝜏 subscript 𝜓 𝑖 1 𝜏 subscript superscript 𝜓′𝑖\psi^{\prime}_{i}\leftarrow\tau\psi_{i}+(1-\tau)\psi^{\prime}_{i}italic_ψ start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ← italic_τ italic_ψ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT + ( 1 - italic_τ ) italic_ψ start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT

ϕ′←τ⁢ϕ+(1−τ)⁢ϕ′←superscript italic-ϕ′𝜏 italic-ϕ 1 𝜏 superscript italic-ϕ′\phi^{\prime}\leftarrow\tau\phi+(1-\tau)\phi^{\prime}italic_ϕ start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ← italic_τ italic_ϕ + ( 1 - italic_τ ) italic_ϕ start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT

end if

end for

{Note that ψ 1 subscript 𝜓 1\psi_{1}italic_ψ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT, ψ 2 subscript 𝜓 2\psi_{2}italic_ψ start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT, ψ 1′subscript superscript 𝜓 normal-′1\psi^{\prime}_{1}italic_ψ start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT, and ψ 2′subscript superscript 𝜓 normal-′2\psi^{\prime}_{2}italic_ψ start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT are maintained across calls to this function}

{When walking the search policy, we just return the new policy params ϕ italic-ϕ\phi italic_ϕ}

∇f←ϕ−θ μ←∇𝑓 italic-ϕ subscript 𝜃 𝜇\nabla f\leftarrow\phi-\theta_{\mu}∇ italic_f ← italic_ϕ - italic_θ start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT

return

∇f∇𝑓\nabla f∇ italic_f

Hyperparameters: TD3GA inherits all relevant hyperparameters from PPGA (Table[1](https://arxiv.org/html/2305.13795v2#A2.T1 "Table 1 ‣ Appendix B Hyperparameters ‣ Proximal Policy Gradient Arborescence for Quality Diversity Reinforcement Learning")). It also inherits TD3 hyperparameters from prior QD-RL works that incorporate TD3(Tjanaka et al., [2022b](https://arxiv.org/html/2305.13795v2#bib.bib41); Nilsson & Cully, [2021](https://arxiv.org/html/2305.13795v2#bib.bib33)), except for having a larger replay buffer. We also increase the batch size used during critic training to mimic the batch size used by PPO to regress the value function. Table[2](https://arxiv.org/html/2305.13795v2#A4.T2 "Table 2 ‣ Appendix D TD3GA Implementation Details ‣ Proximal Policy Gradient Arborescence for Quality Diversity Reinforcement Learning") lists hyperparameters shared across all environments. Note that the archive learning rate depends on the environment but is identical to that used in PPGA.

Table 2: List of hyperparameters used in TD3GA, shared across all environments.

Hyperparameter Value
Actor Network[128, 128, Action Dim]
Critic Network[256, 256, 1]
Training Steps for Objective and Measures (N 1 subscript 𝑁 1 N_{1}italic_N start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT)10
Training Steps for Walking (N 2 subscript 𝑁 2 N_{2}italic_N start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT)10
Optimizer Adam
Adam Learning Rate 3×10−4 3 superscript 10 4 3\times 10^{-4}3 × 10 start_POSTSUPERSCRIPT - 4 end_POSTSUPERSCRIPT
Target Network Update Rate (τ 𝜏\tau italic_τ)0.005
Target Network Update Frequency (d 𝑑 d italic_d)2
Smoothing Noise Standard Deviation (σ p subscript 𝜎 𝑝\sigma_{p}italic_σ start_POSTSUBSCRIPT italic_p end_POSTSUBSCRIPT)0.2
Smoothing Noise Clip (c 𝑐 c italic_c)0.5
Discount Factor (γ 𝛾\gamma italic_γ)0.99
Replay Buffer Size 5,000,000
Batch Size (n b⁢a⁢t⁢c⁢h subscript 𝑛 𝑏 𝑎 𝑡 𝑐 ℎ n_{batch}italic_n start_POSTSUBSCRIPT italic_b italic_a italic_t italic_c italic_h end_POSTSUBSCRIPT)48,000

Appendix E Hyperparameter Study
-------------------------------

We investigate the effects of changing the N 1 subscript 𝑁 1 N_{1}italic_N start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT and N 2 subscript 𝑁 2 N_{2}italic_N start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT hyperparameters on the performance of PPGA in the Humanoid domain. We run 4 seeds of PPGA on the following combinations of N 1 subscript 𝑁 1 N_{1}italic_N start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT and N 2 subscript 𝑁 2 N_{2}italic_N start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT: (10, 5), (5, 10), and (1, 1), and compare against the baseline (10, 10). With these experiments, we wish to address the following questions

1.   1.How few PPO steps on both the objective-measure Jacobian calculation and walking the current search policy can we take before noticing performance degradation? 
2.   2.How do asymmetric choices for N 1 subscript 𝑁 1 N_{1}italic_N start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT and N 2 subscript 𝑁 2 N_{2}italic_N start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT (i.e. more Jacobian calculation steps than walking steps and vice versa) affect performance? 

![Image 8: Refer to caption](https://arxiv.org/html/2305.13795v2/x8.png)

Figure 8: Study of the effect of different hyperparameter choices for N 1 subscript 𝑁 1 N_{1}italic_N start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT and N 2 subscript 𝑁 2 N_{2}italic_N start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT. N 1 subscript 𝑁 1 N_{1}italic_N start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT is the number of PPO steps used to calculate the objective-measure Jacobian, and N 2 subscript 𝑁 2 N_{2}italic_N start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT is the number of PPO steps used to walk the search policy. All plots are averaged over 4 seeds. The shaded region is the 95% boostrapped confidence interval.

(5, 10) achieves the most consistent results while achieving the same performance as the baseline, while (10, 5) results in very high run-to-run variance and lower coverage. This suggests that spending more computation on walking the current search policy is more important than approximating the objective-measure Jacobian. Nonetheless, we conclude that in a more computationally constrained setting, PPGA could be tuned to perform fewer N 1 subscript 𝑁 1 N_{1}italic_N start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT and N 2 subscript 𝑁 2 N_{2}italic_N start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT steps while still maintaining good performance. Finally, (1, 1) performs the worst, achieving slightly more than 50% of the baseline’s performance, implying that there is significant performance degradation when too few Jacobian-calculation and walking steps are taken.

Appendix F PPO Ablations
------------------------

We perform ablations against standard PPO on Humanoid, the most challenging of all domains, in order to compare PPGA’s relative performance against standard RL. We first ablate how the measure functions affect performance. The algorithm "PPGA (No Measure)" performs exactly as standard PPGA but with no measure functions. In this case, PPGA computes gradients for the RL objective when branching and deciding where in the archive to move next. Unsurprisingly, PPGA (No Measures) achieves the same best reward as PPGA, but with less than half the archive coverage.

In the second ablation, we run vanilla PPO and store the intermediate policies in between mini-batch gradient descent steps in an archive. The algorithm, dubbed "PPO + Archive", achieves the same best reward as PPGA, but with less than 1% archive coverage. Note that PPGA (No Measures) is still performing an outer-loop optimization step of the QD-objective, thus achieving better coverage than PPO + Archive, whereas PPO + Archive only optimizes for the RL objective.

Algorithm QD-Score Coverage Best Reward
PPGA 3.59×10 5 3.59 superscript 10 5 3.59\times 10^{5}3.59 × 10 start_POSTSUPERSCRIPT 5 end_POSTSUPERSCRIPT 98.67%9677
PPGA (No Measures)8.24×10 4 8.24 superscript 10 4 8.24\times 10^{4}8.24 × 10 start_POSTSUPERSCRIPT 4 end_POSTSUPERSCRIPT 32.06%9653
PPO + Archive 3.30×10 3 3.30 superscript 10 3 3.30\times 10^{3}3.30 × 10 start_POSTSUPERSCRIPT 3 end_POSTSUPERSCRIPT 0.08%9651

Table 3: Ablation study of PPGA with various components on Humanoid. PPGA (No Measures) functions exactly as PPGA, but only computes objective gradients and walks the search policy with respect to f. PPO + Archive runs standard PPO and stores the intermediate policy updates as policies into an archive. Archives are 10x10.

Appendix G Scaling Experiments
------------------------------

![Image 9: Refer to caption](https://arxiv.org/html/2305.13795v2/x9.png)

Figure 9: QD-Score, Coverage, Best Reward, and CCDF plots for PPGA and baselines, with 50x50 archives for all tasks except for Ant. Ant retains the same 10 4 superscript 10 4 10^{4}10 start_POSTSUPERSCRIPT 4 end_POSTSUPERSCRIPT archive resolution, as this is already sufficiently large.

![Image 10: Refer to caption](https://arxiv.org/html/2305.13795v2/x10.png)

Figure 10: QD Metrics for 50x50 archives of PPGA and PGA-ME trained to 1.2 million evaluations. Results are averaged over 4 seeds. The additional "average performance" metric is presented to show how performance over all policies in the archive changes with additional training.

We test the scalability of PPGA along two axes – the ability to scale to larger archives and the ability to learn with more data. To test for the first property, we scale up the archive resolution to 50x50 for locomotion tasks with two measures i.e. Humanoid, Walker2D, and Halfcheetah and compare against baselines. PPGA retains the same performance across tasks, with slight reductions in coverage. This is due to PPGA being an entirely gradient-based method, using gradients for branching and walking the search policy, whereas prior methods that employ ES can get lucky and randomly mutate policies into far away cells.

Finally, we run our algorithm on a 50x50 archive with 1.2 million evaluations, more than twice as long as the main experiments, of both PPGA and the state of the art baseline PGA-ME, on Humanoid. We report an additional metric, the average performance of the archive, which is the sum of scores of all policies divided by the total number of policies in the archive. This can also be interpreted as the normalized QD score. We find that PPGA continues to improve on this metric, indicating that the episodic returns and thus performance of many policies in the archive continues to improve with additional training. We observe this effect to a much lesser extent with PGA-ME.

Appendix H Comparing to PBT-ME (SAC)
------------------------------------

![Image 11: Refer to caption](https://arxiv.org/html/2305.13795v2/x11.png)

Figure 11: Comparison to PBT-ME (SAC) on Humanoid, which uses Soft Actor-Critic (Haarnoja et al., [2018](https://arxiv.org/html/2305.13795v2#bib.bib17)) to compute the policy gradient when improving the agents.

Population-Based Training MAP-Elites (PBT-ME)(Pierrot & Flajolet, [2023](https://arxiv.org/html/2305.13795v2#bib.bib34)) is a recent QD-RL algorithm that alleviates hyperparameter sensitivity in QD-RL algorithms by evolving populations of agents and their hyperparameters, while also using the policy gradient formulation to improve the agents’ performance. The evolved and optimized policies are added to an archive following the MAP-Elites formulation. PBT-ME was run with Google TPUs – due to computational constraints, we were only able to make comparisons against PBT-ME on the Humanoid domain, which we present here. Specifically, we compare to the PBT-ME (SAC) variant.

