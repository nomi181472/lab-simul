Title: Representation-Driven Reinforcement Learning

URL Source: https://arxiv.org/html/2305.19922

Markdown Content:
Representation-Driven Reinforcement Learning
Ofir Nabati    Guy Tennenholtz    Shie Mannor
Representation-Driven Reinforcement Learning - Appendix
Ofir Nabati    Guy Tennenholtz    Shie Mannor
Abstract

We present a representation-driven framework for reinforcement learning. By representing policies as estimates of their expected values, we leverage techniques from contextual bandits to guide exploration and exploitation. Particularly, embedding a policy network into a linear feature space allows us to reframe the exploration-exploitation problem as a representation-exploitation problem, where good policy representations enable optimal exploration. We demonstrate the effectiveness of this framework through its application to evolutionary and policy gradient-based approaches, leading to significantly improved performance compared to traditional methods. Our framework provides a new perspective on reinforcement learning, highlighting the importance of policy representation in determining optimal exploration-exploitation strategies.

Machine Learning, ICML


1 Introduction

Reinforcement learning (RL) is a field in machine learning in which an agent learns to maximize a reward through interactions with an environment. The agent maps its current state into action and receives a reward signal. Its goal is to maximize the cumulative sum of rewards over some predefined (possibly infinite) horizon (Sutton & Barto, 1998). This setting fits many real-world applications such as recommendation systems (Li et al., 2010), board games (Silver et al., 2017), computer games (Mnih et al., 2015), and robotics (Polydoros & Nalpantidis, 2017).

A large amount of contemporary research in RL focuses on gradient-based policy search methods (Sutton et al., 1999; Silver et al., 2014; Schulman et al., 2015, 2017; Haarnoja et al., 2018). Nevertheless, these methods optimize the policy locally at specific states and actions. Salimans et al. (2017) have shown that such optimization methods may cause high variance updates in long horizon problems, while Tessler et al. (2019) have shown possible convergence to suboptimal solutions in continuous regimes. Moreover, policy search methods are commonly sample inefficient, particularly in hard exploration problems, as policy gradient methods usually converge to areas of high reward, without sacrificing exploration resources to achieve a far-reaching sparse reward.

In this work, we present Representation-Driven Reinforcement Learning (RepRL) – a new framework for policy-search methods, which utilizes theoretically optimal exploration strategies in a learned latent space. Particularly, we reduce the policy search problem to a contextual bandit problem, using a mapping from policy space to a linear feature space. Our approach leverages the learned linear space to optimally tradeoff exploration and exploitation using well-established algorithms from the contextual bandit literature (Abbasi-Yadkori et al., 2011; Agrawal & Goyal, 2013). By doing so, we reframe the exploration-exploitation problem to a representation-exploitation problem, for which good policy representations enable optimal exploration.

We demonstrate the effectiveness of our approach through its application to both evolutionary and policy gradient-based approaches – demonstrating significantly improved performance compared to traditional methods. Empirical experiments on the MuJoCo (Todorov et al., 2012) and MinAtar (Young & Tian, 2019) show the benefits of our approach, particularly in sparse reward settings. While our framework does not make the exploration problem necessarily easier, it provides a new perspective on reinforcement learning, shifting the focus to policy representation in the search for optimal exploration-exploitation strategies.

2 Preliminaries

We consider the infinite-horizon discounted Markov Decision Process (MDP). An MDP is defined by the tuple 
ℳ
=
(
𝒮
,
𝒜
,
𝑟
,
𝑇
,
𝛽
,
𝛾
)
, where 
𝒮
 is the state space, 
𝒜
 is the action space, 
𝑇
:
𝒮
×
𝒜
→
Δ
⁢
(
𝒮
)
 is the transition kernel, 
𝑟
:
𝒮
×
𝒜
→
[
0
,
1
]
 is the reward function, 
𝛽
∈
Δ
⁢
(
𝒮
)
 is the initial state distribution, and 
𝛾
∈
[
0
,
1
)
 is the discount factor. A stationary policy 
𝜋
:
𝒮
→
Δ
⁢
(
𝒜
)
, maps states into a distribution over actions. We denote by 
Π
 the set of stationary stochastic policies, and the history of policies and trajectories up to episode 
𝑘
 by 
ℋ
𝑘
. Finally, we denote 
𝑆
=
|
𝒮
|
 and 
𝐴
=
|
𝒜
|
.

The return of a policy is a random variable defined as the discounted sum of rewards

	
𝐺
⁢
(
𝜋
)
=
∑
𝑡
=
0
∞
𝛾
𝑡
⁢
𝑟
⁢
(
𝑠
𝑡
,
𝑎
𝑡
)
,
		(1)

where 
𝑠
0
∼
𝛽
,
𝑎
𝑡
∼
𝜋
⁢
(
𝑠
𝑡
)
,
𝑠
𝑡
+
1
∼
𝑇
⁢
(
𝑠
𝑡
,
𝑎
𝑡
)
, and the policy’s value is its mean, i.e., 
𝑣
⁢
(
𝜋
)
=
𝔼
#
⁢
1
⁢
[
\setargs
⁢
∑
𝑡
=
0
∞
𝛾
𝑡
⁢
𝑟
⁢
(
𝑠
𝑡
,
𝑎
𝑡
)
|
𝛽
,
𝜋
,
𝑇
]
. An optimal policy maximizes the value, i.e., 
𝜋
*
∈
arg
⁢
max
𝜋
∈
Π
⁡
𝑣
⁢
(
𝜋
)
.

We similarly define the per-state value function, 
𝑣
⁢
(
𝜋
,
𝑠
)
 as 
𝑣
⁢
(
𝜋
,
𝑠
)
=
𝔼
#
⁢
1
⁢
[
\setargs
⁢
∑
𝑡
=
0
∞
𝛾
𝑡
⁢
𝑟
⁢
(
𝑠
𝑡
,
𝑎
𝑡
)
|
𝑠
0
=
𝑠
,
𝜋
,
𝑇
]
,
 and note that 
𝑣
⁢
(
𝜋
)
=
𝔼
#
⁢
1
⁢
[
\setargs
⁢
𝑣
⁢
(
𝜋
,
𝑠
)
]
.

Finally, we denote the discounted state-action frequency distribution w.r.t. 
𝜋
 by

	
𝜌
𝜋
⁢
(
𝑠
,
𝑎
)
=
(
1
−
𝛾
)
⁢
∑
𝑡
=
0
∞
𝛾
𝑡
⁢
𝑃
⁢
𝑟
⁢
(
𝑠
𝑡
=
𝑠
,
𝑎
𝑡
=
𝑎
|
𝛽
,
𝜋
,
𝑇
)
,
	

and let 
𝒦
=
{
𝜌
𝜋
:
𝜋
∈
Π
}
.

2.1 Linear Bandits

In this work, we consider the linear bandit framework as defined in Abbasi-Yadkori et al. (2011). At each time 
𝑡
, the learner is given a decision set 
𝐷
𝑡
⊆
ℝ
𝑑
, which can be adversarially and adaptively chosen. The learner chooses an action 
𝑥
𝑡
∈
𝐷
𝑡
 and receives a reward 
𝑟
𝑡
, whose mean is linear w.r.t 
𝑥
𝑡
, i.e., 
𝔼
#
⁢
1
⁢
[
\setargs
⁢
𝑟
𝑡
|
𝑥
𝑡
]
=
⟨
𝑥
𝑡
,
𝑤
⟩
 for some unknown parameter vector 
𝑤
∈
ℝ
𝑑
.

A general framework for solving the linear bandit problem is the “Optimism in the Face of Uncertainty Linear bandit algorithm” (OFUL, Abbasi-Yadkori et al. (2011)). There, a linear regression estimator is constructed each round as follows:

	
𝑤
^
𝑡
=
𝑉
𝑡
−
1
⁢
𝑏
𝑡
,
	
	
𝑉
𝑡
=
𝑉
𝑡
−
1
+
𝑥
𝑡
⁢
𝑥
𝑡
⊤
,
	
	
𝑏
𝑡
=
𝑏
𝑡
−
1
+
𝑥
𝑡
⁢
𝑦
𝑡
,
		(2)

where 
𝑦
𝑡
,
𝑥
𝑡
 are the noisy reward signal and chosen action at time 
𝑡
, respectively, and 
𝑉
0
=
𝜆
⁢
𝐼
 for some positive parameter 
𝜆
>
0
.

It can be shown that, under mild assumptions, and with high probability, the self-normalizing norm 
∥
𝑤
^
𝑡
−
𝑤
∥
𝑉
𝑡
 can be bounded from above (Abbasi-Yadkori et al., 2011). OFUL then proceeds by taking an optimistic action 
(
𝑥
𝑡
,
𝑤
¯
𝑡
)
∈
arg
⁡
max
𝑥
∈
𝐷
𝑡
,
𝑤
¯
∈
𝒞
𝑡
⁡
⟨
𝑥
,
𝑤
¯
⟩
, where 
𝒞
𝑡
 is a confidence set induced by the aforementioned bound on 
∥
𝑤
^
𝑡
−
𝑤
∥
𝑉
𝑡
. In practice, a softer version is used in Chu et al. (2011), where an action is selected optimistically according to

	
𝑥
𝑡
∈
arg
⁡
max
𝑥
∈
𝐷
𝑡
⁡
⟨
𝑥
,
𝑤
^
𝑡
⟩
+
𝛼
⁢
𝑥
𝑇
⁢
𝑉
𝑡
−
1
⁢
𝑥
,
		(OFUL)

where 
𝛼
>
0
 controls the level of optimism.

Alternatively, linear Thompson sampling (TS, Abeille & Lazaric (2017)) shows it is possible to converge to an optimal solution with sublinear regret, even with a constant probability of optimism. This is achieved through the sampling of a parameter vector from a normal distribution, which is determined by the confidence set 
𝒞
𝑡
. Specifically, linear TS selects an action according to

	
𝑥
𝑡
∈
arg
⁡
max
𝑥
∈
𝐷
𝑡
⁡
⟨
𝑥
,
𝑤
~
𝑡
⟩
,
𝑤
~
𝑡
∼
𝒩
⁢
(
𝑤
^
𝑡
,
𝜎
2
⁢
𝑉
𝑡
−
1
)
,
		(TS)

where 
𝜎
>
0
 controls the level of optimism. We note that for tight regret guarantees, both 
𝛼
 and 
𝜎
 need to be chosen to respect the confidence set 
𝒞
𝑡
. Nevertheless, it has been shown that tuning these parameters can improve performance in real-world applications (Chu et al., 2011).

3 RL as a Linear Bandit Problem
Figure 1: RepRL scheme. Composed of 4 stages: representation of the parameters, constructing a decision set, choosing the best arm using an off-the-shelf linear bandit algorithm, collect data with the chosen policy.

Classical methods for solving the RL problem attempted to use bandit formulations (Fox & Rolph, 1973). There, the set of policies 
Π
 reflects the set of arms, and the value 
𝑣
⁢
(
𝜋
)
 is the expected bandit reward. Unfortunately, such a solution is usually intractable due to the exponential number of policies (i.e., bandit actions) in 
Π
.

Alternatively, we consider a linear bandit formulation of the RL problem. Indeed, it is known that the value can be expressed in linear form as

	
𝑣
⁢
(
𝜋
)
=
𝔼
#
⁢
1
⁢
[
\setargs
⁢
𝑟
⁢
(
𝑠
,
𝑎
)
]
=
⟨
𝜌
𝜋
,
𝑟
⟩
.
		(3)

Here, any 
𝜌
𝜋
∈
𝒦
 represents a possible action in the linear bandit formulation (Abbasi-Yadkori et al., 2011). Notice that 
|
𝒦
|
=
|
Π
|
, as any policy 
𝜋
∈
Π
 can be written as 
𝜋
⁢
(
𝑎
|
𝑠
)
=
𝜌
𝜋
⁢
(
𝑠
,
𝑎
)
∑
𝑎
′
𝜌
𝜋
⁢
(
𝑠
,
𝑎
′
)
, rendering the problem intractable. Nevertheless, this formulation can be relaxed using a lower dimensional embedding of 
𝜌
𝜋
 and 
𝑟
. As such, we make the following assumption.

Assumption 3.1 (Linear Embedding).

There exist a mapping 
𝑓
:
Π
→
ℝ
𝑑
 such that 
𝑣
⁢
(
𝜋
)
=
⟨
𝑓
⁢
(
𝜋
)
,
𝑤
⟩
 for all 
𝜋
∈
Π
 and some unknown 
𝑤
∈
ℝ
𝑑
.

We note that 3.1 readily holds when 
𝑑
=
𝑆
⁢
𝐴
 for 
𝑓
⁢
(
𝜋
)
≡
𝜌
𝜋
 and 
𝑤
=
𝑟
. For efficient solutions, we consider environments for which the dimension 
𝑑
 is relatively low, i.e., 
𝑑
≪
𝑆
⁢
𝐴
.

Note that neural bandit approaches also consider linear representations (Riquelme et al., 2018). Nevertheless, these methods use mappings from states 
𝒮
↦
ℝ
𝑑
, whereas we consider mapping entire policies 
Π
↦
ℝ
𝑑
 (i.e., embedding the function 
𝜋
). Learning a mapping 
𝑓
 can be viewed as trading the effort of finding good exploration strategies in deep RL problems to finding a good representation. We emphasize that we do not claim it to be an easier task, but rather a different viewpoint of the problem, for which possible new solutions can be derived. Similar to work on neural-bandits (Riquelme et al., 2018), finding such a mapping requires alternating between representation learning and exploration.

3.1 RepRL

We formalize a representation-driven framework for RL, inspired by linear bandits (Section 2.1) and 3.1. We parameterize the policy 
𝜋
 and mapping 
𝑓
 using neural networks, 
𝜋
𝜃
 and 
𝑓
𝜙
, respectively. Here, a policy 
𝜋
𝜃
 is represented in lower-dimensional space as 
𝑓
𝜙
⁢
(
𝜋
𝜃
)
. Therefore, searching in policy space is equivalent to searching in the parameter space. With slight abuse of notation, we will denote 
𝑓
𝜙
⁢
(
𝜋
𝜃
)
=
𝑓
𝜙
⁢
(
𝜃
)
.

Pseudo code for RepRL is presented in Algorithm 1. At every episode 
𝑘
, we map the policy’s parameters 
𝜃
𝑘
−
1
 to a latent space using 
𝑓
𝜙
𝑘
−
1
⁢
(
𝜃
𝑘
−
1
)
. We then use a construction algorithm, 
𝙲𝚘𝚗𝚜𝚝𝚛𝚞𝚌𝚝𝙳𝚎𝚌𝚒𝚜𝚘𝚗𝚂𝚎𝚝
⁢
(
𝜃
𝑘
−
1
,
ℋ
𝑘
−
1
)
, which takes into account the history 
ℋ
𝑘
−
1
, to generate a new decision set 
𝐷
𝑘
. Then, to update the parameters 
𝜃
𝑘
−
1
 of the policy, we select an optimistic policy 
𝜋
𝜃
𝑘
∈
𝐷
𝑘
 using a linear bandit method, such as TS or OFUL (see Section 2.1). Finally, we rollout the policy 
𝜋
𝜃
𝑘
 and update the representation network and the bandit parameters according to the procedure outlined in Equation 2, where 
𝑥
𝑘
 are the learned representations of 
𝑓
𝜙
𝑘
. A visual schematic of our framework is depicted in Figure 1.

In the following sections, we present and discuss methods for representation learning, decision set construction, and propose two implementations of RepRL in the context of evolutionary strategies and policy gradient. We note that RepRL is a framework for addressing RL through representation, and as such, any representation learning technique or decision set algorithm can be incorporated as long as the basic structure is maintained.

Algorithm 1 RepRL
1:  Init: 
ℋ
0
←
∅
, 
𝜋
𝜃
0
, 
𝑓
𝜙
0
 randomly initialized
2:  for  
𝑘
=
1
,
2
,
…
  do
3:     Representation Stage:Map the policy network 
𝜋
𝜃
𝑘
−
1
 using representation network 
𝑓
𝜙
𝑘
−
1
⁢
(
𝜃
𝑘
−
1
)
.
4:     Decision Set Stage:
𝐷
𝑘
←
𝙲𝚘𝚗𝚜𝚝𝚛𝚞𝚌𝚝𝙳𝚎𝚌𝚒𝚜𝚘𝚗𝚂𝚎𝚝
⁢
(
𝜃
𝑘
−
1
,
ℋ
𝑘
−
1
)
.
5:     Bandit Stage:Use linear bandit algorithm to choose 
𝜋
𝜃
𝑘
 out of 
𝐷
𝑘
.
6:     Exploitation Stage:Rollout policy 
𝜋
𝜃
𝑘
 and store the return 
𝐺
𝑘
 in 
ℋ
𝑘
.  
7:     Update representation 
𝑓
𝜙
𝑘
.
8:     Update bandit parameters 
𝑤
^
𝑡
,
𝑉
𝑡
 (Equation 2) with the updated representation.
9:  end for
3.2 Learning Representations for RepRL

We learn a linear representation of a policy using tools from variational inference. Specifically, we sample a representation from a posterior distribution 
𝑧
∼
𝑓
𝜙
⁢
(
𝑧
|
𝜃
)
, and train the representation by maximizing the Evidence Lower Bound (ELBO) (Kingma & Welling, 2013) 
ℒ
⁢
(
𝜙
,
𝜅
)
=
−
𝔼
𝑧
∼
𝑓
𝜙
⁢
(
𝑧
|
𝜃
)
⁢
[
log
⁡
𝑝
𝜅
⁢
(
𝐺
|
𝑧
)
]
+
𝐷
𝐾
⁢
𝐿
⁢
(
𝑓
𝜙
⁢
(
𝑧
|
𝜃
)
∥
𝑝
⁢
(
𝑧
)
)
,
 where 
𝑓
𝜙
⁢
(
𝑧
|
𝜃
)
 acts as the encoder of the embedding, and 
𝑝
𝜅
⁢
(
𝐺
|
𝑧
)
 is the return decoder or likelihood term.

The latent representation prior 
𝑝
⁢
(
𝑧
)
 is typically chosen to be a zero-mean Gaussian distribution. In order to encourage linearity of the value (i.e the return’s mean) with respect to the learned representation, we chose the likelihood to be a Gaussian distribution with a mean that is linear in the representation, i.e., 
𝑝
𝜅
⁢
(
𝐺
|
𝑧
)
=
𝒩
⁢
(
𝜅
⊤
⁢
𝑧
,
𝜎
2
)
. When the encoder is also chosen to be a Gaussian distribution, the loss function has a closed form. The choice of the decoder to be linear is crucial, due to the fact that the value is supposed to be linear w.r.t learned embeddings. The parameters 
𝜙
 and 
𝜅
 are the learned parameters of the encoder and decoder, respectively. Note that a deterministic mapping occurs when the function 
𝑓
𝜙
⁢
(
𝑧
|
𝜃
)
 takes the form of the Dirac delta function. A schematic of the architectural framework is presented in Figure 2.

Figure 2: The diagram illustrates the structure of the networks in RepRL. The policy’s parameters are fed into the representation network, which acts as a posterior distribution for the policy’s latent representation. Sampling from this posterior, the latent representation is used by the bandits algorithm to evaluate the value that encapsulates the exploration-exploitation tradeoff.
3.3 Constructing a Decision Set

The choice of the decision set algorithm (line 4 of Algorithm 1) may have a great impact on the algorithm in terms of performance and computational complexity. Clearly, choosing 
𝐷
𝑘
=
Π
,
∀
𝑘
 will be unfeasible in terms of computational complexity. Moreover, it may be impractical to learn a linear representation for all policies at once. We present several possible choices of decision sets below.

Policy Space Decision Set.

One potential strategy is to sample a set of policies centered around the current policy

	
𝐷
𝑘
=
{
𝜃
𝑘
+
𝜖
𝑖
}
𝑖
=
1
𝑁
,
𝜖
𝑖
∼
𝒩
⁢
(
0
,
𝜈
2
⁢
𝐼
)
,
		(4)

where 
𝜈
>
0
 controls how local policy search is. This approach is motivated by the assumption that the representation of policies in the vicinity of the current policy will exhibit linear behavior with respect to the value function due to their similarity to policies encountered by the learner thus far.

Latent Space Decision Set.

An alternative approach involves sampling policies in their learned latent space, i.e.,

	
𝐷
𝑘
=
{
𝑧
𝑘
+
𝜖
𝑖
}
𝑖
=
1
𝑁
,
𝜖
𝑖
∼
𝒩
⁢
(
0
,
𝜈
2
⁢
𝐼
)
,
		(5)

where 
𝑧
𝑘
∼
𝑓
𝜙
⁢
(
𝑧
|
𝜃
𝑘
)
. The linearity of the latent space ensures that this decision set will improve the linear bandit target (UCB or the sampled value in TS), which will subsequently lead to an improvement in the actual value. This approach enables optimal exploration w.r.t. linear bandits, as it uniformly samples the eigen directions of the precision matrix 
𝑉
𝑡
, rather than only sampling specific directions as may occur when sampling in the parameter space.

Unlike Equation 4 constructing the set in Equation 5 presents several challenges. First, in order to rollout the policy 
𝜋
𝜃
𝑘
, one must construct an inverse mapping to extract the chosen policy from the selected latent representation. This can be done by training a decoder for the policy parameters 
𝑞
⁢
(
𝜃
|
𝑧
)
. Alternatively, we propose to use a decoder-free approach. Given a target embedding 
𝑧
*
∈
arg
⁢
max
𝑧
∈
𝐷
𝑡
⁡
⟨
𝑧
,
𝑤
^
⟩
, we search for a policy 
𝜃
*
∈
arg
⁢
max
𝜃
⁡
𝑓
𝜙
⁢
(
𝑧
*
|
𝜃
)
.
 This optimization problem can be solved using gradient descent-based optimization algorithms by varying the inputs to 
𝑓
𝜙
. A second challenge for latent-based decision sets involves the realizability of such policies. That is, there may exist representations 
𝑧
∈
𝐷
𝑘
, which are not mapped by any policy in 
Π
. Lastly, even for realizable policies, the restored 
𝜃
 may be too far from the learned data manifold, leading to an overestimation of its value and a degradation of the overall optimization process. One way to address these issues is to use a small enough value of 
𝜈
 during the sampling process, reducing the probability of the set members being outside the data distribution. We leave more sophisticated methods of latent-based decision sets for future work.

History-based Decision Set.

An additional approach uses the history of policies at time 
𝑘
 to design a decision set. Specifically, at time episode 
𝑘
 we sample around the set of policies observed so far, i.e.,

	
𝐷
𝑘
=
⋃
ℓ
∈
[
𝑘
]
{
𝜃
ℓ
+
𝜖
ℓ
,
𝑖
}
𝑖
=
1
𝑁
,
𝜖
ℓ
,
𝑖
∼
𝒩
⁢
(
0
,
𝜈
2
⁢
𝐼
)
,
		(6)

resulting in a decision set of size 
𝑁
⁢
𝑘
. After improving the representation over time, it may be possible to find a better policy near policies that have already been used and were missed due to poor representation or sampling mismatch. This method is quite general, as the history can be truncated only to consider a certain number of past time steps, rather than the complete set of policies observed so far. Truncating the history can help reduce the size of the decision set, making the search more computationally tractable.

In Section 5, we compare the various choices of decision sets. Nevertheless, we found that using policy space decisions is a good first choice, due to their simplicity, which leads to stable implementations. Further exploration of other decision sets is left as a topic for future research.

3.4 Inner trajectory sampling

Vanilla RepRL uses the return values of the entire trajectory. As a result, sampling the trajectories at their initial states is the natural solution for both the bandit update and representation learning. However, the discount factor diminishes learning signals beyond the 
1
1
−
𝛾
 effective horizon, preventing the algorithm from utilizing these signals, which may be critical in environments with long-term dependencies. On the other hand, using a discount factor 
𝛾
=
1
 would result in returns with a large variance, leading to poor learning. Instead of sampling from the initial state, we propose to use the discount factor and sample trajectories at various states during learning, enabling the learner to observe data from different locations along the trajectory. Under this sampling scheme, the estimated value would be an estimate of the following quantity:

	
𝑣
~
⁢
(
𝜋
)
=
𝔼
#
⁢
1
⁢
[
\setargs
⁢
𝑣
⁢
(
𝜋
,
𝑠
)
]
.
	

In the following proposition we prove that optimizing 
𝑣
~
⁢
(
𝜋
)
 is equivalent to optimizing the real value.

Proposition 3.2.

For a policy 
𝜋
∈
Π
, 
𝑣
~
⁢
(
𝜋
)
=
𝑣
⁢
(
𝜋
)
1
−
𝛾
.

The proof can be found in the Appendix C. That is, sampling along the trajectory from 
𝜌
𝜋
 approximates the scaled value, which, like 
𝑣
⁢
(
𝜋
)
, exhibits linear behavior with respect to the reward function. Thus, instead of sampling the return defined in Equation 1, we sample 
𝐺
~
⁢
(
𝜋
)
=
∑
𝑡
=
0
∞
𝛾
𝑡
⁢
𝑟
⁢
(
𝑠
𝑡
,
𝑎
𝑡
)
,
 where 
𝑠
0
∼
𝜌
𝜋
,
𝑎
𝑡
∼
𝜋
⁢
(
𝑠
𝑡
)
,
𝑠
𝑡
+
1
∼
𝑇
⁢
(
𝑠
𝑡
,
𝑎
𝑡
)
, both during representation learning and bandit updates. Empirical evidence suggests that uniformly sampling from the stored trajectory produces satisfactory results in practice.

Algorithm 2 Representation Driven Evolution Strategy
1:  Input: initial policy 
𝜋
0
=
𝜋
𝜃
0
, noise 
𝜈
, step size 
𝛼
, decision set size 
𝑁
, history 
ℋ
.
2:  for  
𝑡
=
1
,
2
,
…
,
𝑇
  do
3:     Sample an evaluation set and collect their returns.
4:     Update representation 
𝑓
𝑡
 and bandit parameters 
(
𝑤
^
𝑡
,
𝑉
𝑡
)
 using history.
5:     Construct a decision set 
𝐷
𝑡
.
6:     Use linear bandit algorithm to evaluate each policy in 
𝐷
𝑡
.
7:     Update policy using ES scheme (Section 4.1).
8:  end for
4 RepRL Algorithms

In this section we describe two possible approaches for applying the RepRL framework; namely, in Evolution Strategy (Wierstra et al., 2014) and Policy Gradients (Sutton et al., 1999).

4.1 Representation Driven Evolution Strategy

Evolutionary Strategies (ES) are used to train agents by searching through the parameter space of their policy and sampling their return. In contrast to traditional gradient-based methods, ES uses a population of candidates evolving over time through genetic operators to find the optimal parameters for the agent. Such methods have been shown to be effective in training deep RL agents in high-dimensional environments (Salimans et al., 2017; Mania et al., 2018).

At each round, the decision set is chosen over the policy space with Gaussian sampling around the current policy as described in Section 3.3. Algorithm 5 considers an ES implementation of RepRL. To improve the stability of the optimization process, we employ soft-weighted updates across the decision set. This type of update rule is similar to that used in ES algorithms (Salimans et al., 2017; Mania et al., 2018), and allows for an optimal exploration-exploitation trade-off, replacing the true sampled returns with the bandit’s value. Moreover, instead of sampling the chosen policy, we evaluate it by also sampling around it as done in ES-based algorithms. Each evaluation is used for the bandit parameters update and representation learning process. Sampling the evaluated policies around the chosen policy helps the representation avoid overfitting to a specific policy and generalize better for unseen policies - an important property when selecting the next policy.

Algorithm 3 Representation Driven Policy Gradient
1:  Input: initial policy 
𝜋
𝜃
, decision set size 
𝑁
, history 
ℋ
.
2:  for  
𝑡
=
1
,
2
,
…
,
𝑇
  do
3:     Collect trajectories using 
𝜋
𝜃
.
4:     Update representation 
𝑓
 and bandit parameters 
(
𝑤
^
𝑡
,
𝑉
𝑡
)
 using history.
5:     Compute Policy Gradient loss 
ℒ
𝑃
⁢
𝐺
⁢
(
𝜃
)
.
6:     Sample a decision set and choose the best policy 
𝜃
~
.
7:     Compute gradient of the regularized Policy Gradient loss with 
𝑑
⁢
(
𝜃
,
𝜃
~
)
 (Equation 7).
8:  end for

Unlike traditional ES, optimizing the UCB in the case of OFUL or sampling using TS can encourage the algorithm to explore unseen policies in the parameter space. This exploration is further stabilized by averaging over the sampled directions, rather than assigning the best policy in the decision set. This is particularly useful when the representation is still noisy, reducing the risk of instability caused by hard assignments. An alternative approach uses a subset of 
𝐷
𝑡
 with the highest bandit scores, as suggested in Mania et al. (2018), which biases the numerical gradient towards the direction with the highest potential return.

Figure 3: The two-dimensional t-SNE visualization depicts the policy representation in the GridWorld experiment. On the right, we observe the learned latent representation, while on the left, we see the direct representation of the policy’s weights. Each point in the visualization corresponds to a distinct policy, and the color of each point corresponds to a sample of the policy’s value.
Figure 4: GridWorld visualization experiment. Trajectories were averaged across 100 seeds at various times during training, where more recent trajectories have greater opacity. Background colors indicate the level of mean reward.
4.2 Representation Driven Policy Gradient

RepRL can also be utilized as a regularizer for policy gradient algorithms. Pseudo code for using RepRL in policy gradients is shown in Algorithm 6. At each gradient step, a weighted regularization term 
𝑑
⁢
(
𝜃
,
𝜃
~
)
 is added, where 
𝜃
~
 are the parameters output by RepRL with respect to the current parameters for a chosen metric (e.g., 
ℓ
2
):

	
ℒ
reg
⁢
(
𝜃
)
=
ℒ
PG
⁢
(
𝜃
)
+
𝜁
⁢
𝑑
⁢
(
𝜃
,
𝜃
~
)
.
		(7)

After collecting data with the chosen policy and updating the representation and bandit parameters, the regularization term is added to the loss of the policy gradient at each gradient step. The policy gradient algorithm can be either on-policy or off-policy while in our work we experiment with an on-policy algorithm.

Similar to the soft update rule in ES, using RepRL as a regularizer can significantly stabilize the representation process. Applying the regularization term biases the policy toward an optimal exploration strategy in policy space. This can be particularly useful when the representation is still weak and the optimization process is unstable, as it helps guide the update toward more promising areas of the parameter space. In our experiments, we found that using RepRL as a regularizer for policy gradients improved the stability and convergence of the optimization process.

5 Experiments

In order to evaluate the performance of RepRL, we conducted experiments on various tasks on the MuJoCo (Todorov et al., 2012) and MinAtar (Young & Tian, 2019) domains. We also used a sparse version of the MuJoCo environments, where exploration is crucial. We used linear TS as our linear bandits algorithm as it exhibited good performance during evaluation. The detailed network architecture and hyperparameters utilized in the experiments are provided in Appendix F.

Figure 5: MuJoCo experiments during training. The results are for the MuJoCo suitcase (top) and the modified sparse MuJoCo (bottom).
Grid-World Visualization.

Before presenting our results, we demonstrate the RepRL framework on a toy example. Specifically, we constructed a GridWorld environment (depicted in Figure 4) which consists of spatially changing, noisy rewards. The agent, initialized at the bottom left state 
(
𝑥
,
𝑦
)
=
(
1
,
1
)
, can choose to take one of four actions: up, down, left, or right. To focus on exploration, the rewards were distributed unevenly across the grid. Particularly, the reward for every 
(
𝑥
,
𝑦
)
 was defined by the Normal random variable 
𝑟
⁢
(
𝑥
,
𝑦
)
∼
𝒩
⁢
(
𝜇
⁢
(
𝑥
,
𝑦
)
,
𝜎
2
)
,
 where 
𝜎
>
0
 and 
𝜇
⁢
(
𝑥
,
𝑦
)
∝
𝑅
1
⁢
exp
⁡
{
−
(
𝑥
−
𝑥
1
)
2
+
(
𝑦
−
𝑦
1
)
2
𝑎
1
}
+
𝑅
2
⁢
exp
⁡
{
−
(
𝑥
−
𝑥
2
)
2
+
(
𝑦
−
𝑦
2
)
2
𝑎
2
}
+
𝑅
3
⁢
𝟙
{
(
𝑥
,
𝑦
)
=
goal
}
.
 That is, the reward consisted of Normally distributed noise, with mean defined by two spatial Gaussians, as shown in Figure 4, with 
𝑅
1
>
𝑅
2
, 
𝑎
1
<
𝑎
2
 and a goal state (depicted as a star), with 
𝑅
3
≫
𝑅
1
,
𝑅
2
. Importantly, the values of 
𝑅
1
,
𝑅
2
,
𝑅
3
,
𝑎
1
,
𝑎
2
 were chosen such that an optimal policy would take the upper root in Figure 4.

Comparing the behavior of RepRL and ES on the GridWorld environment, we found that RepRL explored the environment more efficiently, locating the optimal path to the goal. This emphasizes the varying characteristics of state-space-driven exploration vs. policy-space-driven exploration, which, in our framework, coincides with representation-driven exploration. Figure 3 illustrates a two-dimensional t-SNE plot comparing the learned latent representation of the policy with the direct representation of the policy weights.

Decision Set Comparison.

We begin by evaluating the impact of the decision set on the performance of the RepRL. For this, we tested the three decision sets outlined in Section 3.3. The evaluation was conducted using the Representation Driven Evolution Strategy variant on a sparse HalfCheetah environment. A history window of 20 policies was utilized when evaluating the history-based decision set. A gradient descent algorithm was employed to obtain the parameters that correspond to the selected latent code in the latent-based setting

As depicted in Figure 8 at Appendix E, RepRL demonstrated similar performance for the varying decision sets on the tested domains. In what follows, we focus on policy space decision sets.

MuJoCo.

We conducted experiments on the MuJoCo suitcase task using RepRL. Our approach followed the setting of Mania et al. (2018), in which a linear policy was used and demonstrated excellent performance on MuJoCo tasks. We utilized the ES variant of our algorithm (Algorithm 5). We incorporated a weighted update between the gradients using the bandit value and the zero-order gradient of the sampled returns, taking advantage of sampled information and ensuring stable updates in areas where the representation is weak.

We first evaluated RepES on the standard MuJoCo baseline (see Figure 5). RepES either significantly outperformed or performed on-par with ES. We also tested a modified, sparse variant of MuJoCo. In the sparse environment, a reward was given for reaching a goal each distance interval, denoted as 
𝑑
, where the reward function was defined as:

	
𝑟
⁢
(
𝑠
,
𝑎
)
=
{
10
−
𝑐
⁢
(
𝑎
)
,
	
|
𝑥
agent
|
mod
𝑑
=
0


−
𝑐
⁢
(
𝑎
)
,
	
o.w.
	

Here, 
𝑐
⁢
(
𝑎
)
 is the control cost associated with utilizing action 
𝑎
, and 
𝑥
𝑎
⁢
𝑔
⁢
𝑒
⁢
𝑛
⁢
𝑡
 denotes the location of the agent along the 
𝑥
-axis. The presence of a control cost function incentivized the agent to maintain its position rather than actively exploring the environment. The results of this experiment, as depicted in Figure 5, indicate that the RepRL algorithm outperformed both the ES and SAC algorithms in terms of achieving distant goals. However, it should be noted that the random search component of the ES algorithm occasionally resulted in successful goal attainment, albeit at a significantly lower rate in comparison to the RepRL algorithm.

Figure 6: MinAtar experiments during training.
MinAtar.

We compared the performance of RepRL on MinAtar (Young & Tian, 2019) with the widely used policy gradient algorithm PPO (Schulman et al., 2017). Specifically, we compared PPO against its regularized version with RepRL, as described in Algorithm 6, and refer to it as RepPG. We parametrized the policy by a neural network. Although PPO collects chunks of rollouts (i.e., uses subtrajectories), RepPG adjusted naturally due to the inner trajectory sampling (see Section 3.4). That is, the critic was used to estimate the value of the rest of the trajectory in cases where the rollouts were truncated by the algorithm.

Results are shown in Figure 6. Overall, RepRL outperforms PPO on all tasks, suggesting that RepRL is effective at solving challenging tasks with sparse rewards, such as those found in MinAtar.

6 Related Work

Policy Optimization: Policy gradient methods (Sutton et al., 1999) have shown great success at various challenging tasks, with numerous improvements over the years; most notable are policy gradient methods for deterministic policies (Silver et al., 2014; Lillicrap et al., 2015), trust region based algorithms (Schulman et al., 2015, 2017), and maximum entropy algorithms (Haarnoja et al., 2018). Despite its popularity, traditional policy gradient methods are limited in continuous action spaces. Therefore, Tessler et al. (2019) suggest optimizing the policy over the policy distribution space rather than the action space.

In recent years, finite difference gradient methods have been rediscovered by the RL community. This class of algorithms uses numerical gradient estimation by sampling random directions (Nesterov & Spokoiny, 2017). A closely related family of optimization methods is Evolution Strategies (ES) a class of black-box optimization algorithms that heuristic search by perturbing and evaluating the set members, choosing only the mutations with the highest scores until convergence. Salimans et al. (2017) used ES for RL as a zero-order gradient estimator for the policy, parameterized as a neural network. ES is robust to the choice of the reward function or the horizon length and it also does not need value function approximation as most state-of-art algorithms. Nevertheless, it suffers from low sample efficiency due to the potentially noisy returns and the usage of the final return value as the sole learning signal. Moreover, it is not effective in hard exploration tasks. Mania et al. (2018) improves ES by using only the most promising directions for gradient estimation.

Policy Search with Bandits. Fox & Rolph (1973) was one of the first works to utilize multi-arm bandits for policy search over a countable stationary policy set – a core approach for follow-up work (Burnetas & Katehakis, 1997; Agrawal et al., 1988). Nevertheless, the concept was left aside due to its difficulty to scale up with large environments.

As an alternative, Neural linear bandits (Riquelme et al., 2018; Xu et al., 2020; Nabati et al., 2021) simultaneously train a neural network policy, while interacting with the environment, using a chosen linear bandit method and are closely related to the neural-bandits literature (Zhou et al., 2020; Kassraie & Krause, 2022). In contrast to this line of work, our work maps entire policy functions into linear space, where linear bandit approaches can take effect. This induces an exploration strategy in policy space, as opposed to locally, in action space.

Representation Learning. Learning a compact and useful representation of states (Laskin et al., 2020; Schwartz et al., 2019; Tennenholtz & Mannor, ), actions (Tennenholtz & Mannor, 2019; Chandak et al., 2019), rewards (Barreto et al., 2017; Nair et al., 2018; Toro Icarte et al., 2019), and policies (Hausman et al., 2018; Eysenbach et al., 2018), has been at the core of a vast array of research. Such representations can be used to improve agents’ performance by utilizing the structure of an environment more efficiently. Policy representation has been the focus of recent studies, including the work by Tang et al. (2022), which, similar to our approach, utilizes policy representation to learn a generalized value function. They demonstrate that the generalized value function can generalize across policies and improve value estimation for actor-critic algorithms, given certain conditions. In another study, Li et al. (2022) enhance the stability and efficiency of Evolutionary Reinforcement Learning (ERL) (Khadka & Tumer, 2018) by adopting a linear policy representation with a shared state representation between the evolution and RL components. In our research, we view the representation problem as an alternative solution to the exploration-exploitation problem in RL. Although this shift does not necessarily simplify the problem, it transfers the challenge to a different domain, offering opportunities for the development of new methods.

7 Discussion and Future Work

We presented RepRL, a novel representation-driven framework for reinforcement learning. By optimizing the policy over a learned representation, we leveraged techniques from the contextual bandit literature to guide exploration and exploitation. We demonstrated the effectiveness of this framework through its application to evolutionary and policy gradient-based approaches, leading to significantly improved performance compared to traditional methods.

In this work, we suggested reframing the exploration-exploitation problem as a representation-exploitation problem. By embedding the policy network into a linear feature space, good policy representations enable optimal exploration. This framework provides a new perspective on reinforcement learning, highlighting the importance of policy representation in determining optimal exploration-exploitation strategies.

As future work, one can incorporate RepRL into more involved representation methods, including pretrained large Transformers (Devlin et al., 2018; Brown et al., 2020), which have shown great promise recently in various areas of machine learning. Another avenue for future research is the use of RepRL in scenarios where the policy is optimized in latent space using an inverse mapping (i.e., decoder), as well as more involved decision sets. Finally, while this work focused on linear bandit algorithms, future work may explore the use of general contextual bandit algorithms, (e.g., SquareCB Foster & Rakhlin (2020)), which are not restricted to linear representations.

8 Acknowledgments

This work was partially funded by the Israel Science Foundation under Contract 2199/20.

References
Abbasi-Yadkori et al. (2011) Yasin Abbasi-Yadkori, David Pal, and Csaba Szepesvari. Improved algorithms for linear stochastic bandits. In Advances in Neural Information Processing Systems, pp. 2312–2320, 2011.
Abeille & Lazaric (2017) Marc Abeille and Alessandro Lazaric. Linear thompson sampling revisited. In Artificial Intelligence and Statistics, pp.  176–184. PMLR, 2017.
Agrawal et al. (1988) Rajeev Agrawal, Demosthenis Teneketzis, and Venkatachalam Anantharam. Asymptotically efficient adaptive allocation schemes for controlled markov chains: Finite parameter space. Technical report, MICHIGAN UNIV ANN ARBOR COMMUNICATIONS AND SIGNAL PROCESSING LAB, 1988.
Agrawal & Goyal (2013) Shipra Agrawal and Navin Goyal. Thompson sampling for contextual bandits with linear payoffs. In International Conference on Machine Learning, pp. 127–135, 2013.
Barreto et al. (2017) André Barreto, Will Dabney, Rémi Munos, Jonathan J Hunt, Tom Schaul, Hado P van Hasselt, and David Silver. Successor features for transfer in reinforcement learning. Advances in neural information processing systems, 30, 2017.
Brown et al. (2020) Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
Burnetas & Katehakis (1997) Apostolos N Burnetas and Michael N Katehakis. Optimal adaptive policies for markov decision processes. Mathematics of Operations Research, 22(1):222–255, 1997.
Chandak et al. (2019) Yash Chandak, Georgios Theocharous, James Kostas, Scott Jordan, and Philip Thomas. Learning action representations for reinforcement learning. In International conference on machine learning, pp. 941–950. PMLR, 2019.
Chu et al. (2011) Wei Chu, Lihong Li, Lev Reyzin, and Robert Schapire. Contextual bandits with linear payoff functions. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pp.  208–214. JMLR Workshop and Conference Proceedings, 2011.
Devlin et al. (2018) Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.
Eysenbach et al. (2018) Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, and Sergey Levine. Diversity is all you need: Learning skills without a reward function. arXiv preprint arXiv:1802.06070, 2018.
Foster & Rakhlin (2020) Dylan Foster and Alexander Rakhlin. Beyond ucb: Optimal and efficient contextual bandits with regression oracles. In International Conference on Machine Learning, pp. 3199–3210. PMLR, 2020.
Fox & Rolph (1973) Bennett L Fox and John E Rolph. Adaptive policies for markov renewal programs. The Annals of Statistics, 1(2):334–341, 1973.
Haarnoja et al. (2018) Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. International conference on machine learning, pp. 1861–1870, 2018.
Hausman et al. (2018) Karol Hausman, Jost Tobias Springenberg, Ziyu Wang, Nicolas Heess, and Martin Riedmiller. Learning an embedding space for transferable robot skills. In International Conference on Learning Representations, 2018.
Kassraie & Krause (2022) Parnian Kassraie and Andreas Krause. Neural contextual bandits without regret. In International Conference on Artificial Intelligence and Statistics, pp.  240–278. PMLR, 2022.
Khadka & Tumer (2018) Shauharda Khadka and Kagan Tumer. Evolution-guided policy gradient in reinforcement learning. Advances in Neural Information Processing Systems, 31, 2018.
Kingma & Welling (2013) Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.
Laskin et al. (2020) Michael Laskin, Aravind Srinivas, and Pieter Abbeel. Curl: Contrastive unsupervised representations for reinforcement learning. In International Conference on Machine Learning, pp. 5639–5650. PMLR, 2020.
Li et al. (2010) Lihong Li, Wei Chu, John Langford, and Robert E Schapire. A contextual-bandit approach to personalized news article recommendation. In Proceedings of the 19th international conference on World wide web, pp.  661–670, 2010.
Li et al. (2022) Pengyi Li, Hongyao Tang, Jianye Hao, Yan Zheng, Xian Fu, and Zhaopeng Meng. Erl-re: Efficient evolutionary reinforcement learning with shared state representation and individual policy representation. arXiv preprint arXiv:2210.17375, 2022.
Lillicrap et al. (2015) Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.
Mania et al. (2018) Horia Mania, Aurelia Guy, and Benjamin Recht. Simple random search provides a competitive approach to reinforcement learning. arXiv preprint arXiv:1803.07055, 2018.
Mnih et al. (2015) Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. nature, 518(7540):529–533, 2015.
Nabati et al. (2021) Ofir Nabati, Tom Zahavy, and Shie Mannor. Online limited memory neural-linear bandits with likelihood matching. arXiv preprint arXiv:2102.03799, 2021.
Nair et al. (2018) Ashvin V Nair, Vitchyr Pong, Murtaza Dalal, Shikhar Bahl, Steven Lin, and Sergey Levine. Visual reinforcement learning with imagined goals. Advances in neural information processing systems, 31, 2018.
Navon et al. (2023) Aviv Navon, Aviv Shamsian, Idan Achituve, Ethan Fetaya, Gal Chechik, and Haggai Maron. Equivariant architectures for learning in deep weight spaces. arXiv preprint arXiv:2301.12780, 2023.
Nesterov & Spokoiny (2017) Yurii Nesterov and Vladimir Spokoiny. Random gradient-free minimization of convex functions. Foundations of Computational Mathematics, 17(2):527–566, 2017.
Polydoros & Nalpantidis (2017) Athanasios S Polydoros and Lazaros Nalpantidis. Survey of model-based reinforcement learning: Applications on robotics. Journal of Intelligent & Robotic Systems, 86(2):153–173, 2017.
Riquelme et al. (2018) Carlos Riquelme, George Tucker, and Jasper Snoek. Deep bayesian bandits showdown: An empirical comparison of bayesian deep networks for thompson sampling. arXiv preprint arXiv:1802.09127, 2018.
Salimans et al. (2017) Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, and Ilya Sutskever. Evolution strategies as a scalable alternative to reinforcement learning. arXiv preprint arXiv:1703.03864, 2017.
Schulman et al. (2015) John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. International conference on machine learning, pp. 1889–1897, 2015.
Schulman et al. (2017) John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
Schwartz et al. (2019) Erez Schwartz, Guy Tennenholtz, Chen Tessler, and Shie Mannor. Language is power: Representing states using natural language in reinforcement learning. arXiv preprint arXiv:1910.02789, 2019.
Silver et al. (2014) David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, and Martin Riedmiller. Deterministic policy gradient algorithms. International conference on machine learning, pp.  387–395, 2014.
Silver et al. (2017) David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. nature, 550(7676):354–359, 2017.
Sutton & Barto (1998) Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press Cambridge, 1998.
Sutton et al. (1999) Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12, 1999.
Tang et al. (2022) Hongyao Tang, Zhaopeng Meng, Jianye Hao, Chen Chen, Daniel Graves, Dong Li, Changmin Yu, Hangyu Mao, Wulong Liu, Yaodong Yang, et al. What about inputting policy in value function: Policy representation and policy-extended value function approximator. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pp.  8441–8449, 2022.
(40) Guy Tennenholtz and Shie Mannor. Uncertainty estimation using riemannian model dynamics for offline reinforcement learning. In Advances in Neural Information Processing Systems.
Tennenholtz & Mannor (2019) Guy Tennenholtz and Shie Mannor. The natural language of actions. In International Conference on Machine Learning, pp. 6196–6205. PMLR, 2019.
Tessler et al. (2019) Chen Tessler, Guy Tennenholtz, and Shie Mannor. Distributional policy optimization: An alternative approach for continuous control. Advances in Neural Information Processing Systems, 32, 2019.
Todorov et al. (2012) Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pp.  5026–5033, 2012. doi: 10.1109/IROS.2012.6386109.
Toro Icarte et al. (2019) Rodrigo Toro Icarte, Ethan Waldie, Toryn Klassen, Rick Valenzano, Margarita Castro, and Sheila McIlraith. Learning reward machines for partially observable reinforcement learning. Advances in neural information processing systems, 32, 2019.
Wierstra et al. (2014) Daan Wierstra, Tom Schaul, Tobias Glasmachers, Yi Sun, Jan Peters, and Jürgen Schmidhuber. Natural evolution strategies. The Journal of Machine Learning Research, 15(1):949–980, 2014.
Xu et al. (2020) Pan Xu, Zheng Wen, Handong Zhao, and Quanquan Gu. Neural contextual bandits with deep representation and shallow exploration. arXiv preprint arXiv:2012.01780, 2020.
Young & Tian (2019) Kenny Young and Tian Tian. Minatar: An atari-inspired testbed for thorough and reproducible reinforcement learning experiments. arXiv preprint arXiv:1903.03176, 2019.
Zhou et al. (2020) Dongruo Zhou, Lihong Li, and Quanquan Gu. Neural contextual bandits with ucb-based exploration. In International Conference on Machine Learning, pp. 11492–11502. PMLR, 2020.
Appendix A Algorithms
Algorithm 4 Random Search / Evolution Strategy
1:  Input: initial policy 
𝜋
0
=
𝜋
𝜃
0
, noise 
𝜈
, step size 
𝛼
, set size 
𝐾
.
2:  for  
𝑡
=
1
,
2
,
…
,
𝑇
  do
3:     Sample a decision set 
𝐷
𝑡
=
{
𝜃
𝑡
−
1
±
𝛿
𝑖
}
𝑖
=
1
𝐾
, 
𝛿
𝑖
∼
𝒩
⁢
(
0
,
𝜈
2
⁢
𝐼
)
.
4:     Collect the returns 
{
𝐺
⁢
(
𝜃
𝑡
−
1
±
𝛿
𝑖
)
}
𝑖
=
1
𝐾
 of each policy in 
𝐷
𝑡
.
5:     Update policy
	
𝜃
𝑡
=
𝜃
𝑡
−
1
+
𝛼
𝜎
𝑅
⁢
𝐾
⁢
∑
𝑖
=
1
𝐾
[
𝐺
⁢
(
𝜃
𝑡
−
1
+
𝛿
𝑖
)
−
𝐺
⁢
(
𝜃
𝑡
−
1
−
𝛿
𝑖
)
]
⁢
𝛿
𝑖
	
6:  end for
Algorithm 5 Representation Driven Evolution Strategy
1:  Input: initial policy 
𝜋
0
=
𝜋
𝜃
0
, noise 
𝜈
, step size 
𝛼
, set size 
𝐾
, decision set size 
𝑁
, history 
ℋ
.
2:  for  
𝑡
=
1
,
2
,
…
,
𝑇
  do
3:     Sample an evaluation set 
{
𝜃
𝑡
−
1
±
𝛿
𝑖
}
𝑖
=
1
𝐾
, 
𝛿
𝑖
∼
𝒩
⁢
(
0
,
𝜈
2
⁢
𝐼
)
.
4:     Collect the returns 
{
𝐺
⁢
(
𝜃
𝑡
−
1
±
𝛿
𝑖
)
}
𝑖
=
1
𝐾
 from the environment and store them in replay buffer.
5:     Update representation 
𝑓
𝑡
 and bandit parameters 
(
𝑤
^
𝑡
,
𝑉
𝑡
)
 using history.
6:     Construct a decision set 
𝐷
𝑡
=
{
𝜃
𝑡
−
1
±
𝛿
𝑖
}
𝑖
=
1
𝑁
, 
𝛿
𝑖
∼
𝒩
⁢
(
0
,
𝜈
2
⁢
𝐼
)
.
7:     Use linear bandit algorithm to evaluate each policy in 
𝐷
𝑡
: 
{
𝑣
^
⁢
(
𝜃
𝑡
−
1
±
𝛿
𝑖
)
}
𝑖
=
1
𝑁
 .
8:     Update policy
	
𝑔
𝑡
=
	
1
𝑁
⁢
∑
𝑖
=
1
𝑁
[
𝑣
^
⁢
(
𝜃
𝑡
−
1
+
𝛿
𝑖
)
−
𝑣
^
⁢
(
𝜃
𝑡
−
1
−
𝛿
𝑖
)
]
⁢
𝛿
𝑖
,
	
	
𝜃
𝑡
=
	
𝜃
𝑡
−
1
+
𝛼
⁢
𝑔
𝑡
	
9:  end for
Algorithm 6 Representation Driven Policy Gradient
1:  Input: initial policy 
𝜋
𝜃
, noise 
𝜈
, step size 
𝛼
, decision set size 
𝑁
, 
𝜁
, history 
ℋ
.
2:  for  
𝑡
=
1
,
2
,
…
,
𝑇
  do
3:     for  
1
,
2
,
…
,
𝐾
  do
4:        Collect trajectory data using 
𝜋
𝜃
.
5:        Update representation 
𝑓
 and bandit parameters 
(
𝑤
^
,
Σ
)
 using history.
6:     end for
7:     for  
1
,
2
,
…
,
𝑀
  do
8:        Sample a decision set 
𝐷
=
{
𝜃
+
𝛿
𝑖
}
𝑖
=
1
𝑁
, 
𝛿
𝑖
∼
𝒩
⁢
(
0
,
𝜈
2
⁢
𝐼
)
.
9:        Use linear bandit algorithm to choose the best parameter 
𝜃
~
∈
arg
⁢
max
𝜃
∈
𝐷
⁡
⟨
𝑧
,
𝑤
^
⟩
 for 
𝑧
∼
𝑓
⁢
(
𝑧
|
𝜃
)
.
10:        Compute
	
𝑔
=
	
∇
𝜃
[
ℒ
𝑃
⁢
𝐺
⁢
(
𝜃
)
+
𝜁
⁢
‖
𝜃
−
𝜃
~
‖
2
]
,
	
	
𝜃
=
	
𝜃
−
𝛼
⁢
𝑔
	
11:     end for
12:  end for
Appendix B Variational Interface

We present here proof of the ELBO loss for our variational interface, which was used to train the representation encoder.

Proof.
	
log
⁡
𝑝
⁢
(
𝐺
;
𝜙
,
𝜅
)
	
=
log
⁢
∫
𝑧
𝑝
𝜅
⁢
(
𝐺
|
𝑧
)
⁢
𝑝
⁢
(
𝑧
)
⁢
𝑑
𝑧
	
		
=
log
⁢
∫
𝑧
𝑝
𝜅
⁢
(
𝐺
|
𝑧
)
⁢
𝑝
⁢
(
𝑧
)
𝑓
𝜙
⁢
(
𝑧
|
𝜋
)
⁢
𝑓
𝜙
⁢
(
𝑧
|
𝜋
)
⁢
𝑑
𝑧
	
		
=
log
⁡
𝔼
𝑧
∼
𝑓
𝜙
⁢
(
𝑧
|
𝜋
)
⁢
[
𝑝
𝜅
⁢
(
𝐺
|
𝑧
)
⁢
𝑝
⁢
(
𝑧
)
𝑓
𝜙
⁢
(
𝑧
|
𝜋
)
]
	
		
≥
𝔼
𝑧
∼
𝑓
𝜙
⁢
(
𝑧
|
𝜋
)
⁢
[
log
⁡
𝑝
𝜅
⁢
(
𝐺
|
𝑧
)
]
+
𝔼
𝑧
∼
𝑓
𝜙
⁢
(
𝑧
|
𝜋
)
⁢
[
log
⁡
𝑝
⁢
(
𝑧
)
𝑓
𝜙
⁢
(
𝑧
|
𝜋
)
]
	
		
=
𝔼
𝑧
∼
𝑓
𝜙
⁢
(
𝑧
|
𝜋
)
⁢
[
log
⁡
𝑝
𝜅
⁢
(
𝐺
|
𝑧
)
]
−
𝐷
𝐾
⁢
𝐿
⁢
(
𝑓
𝜙
⁢
(
𝑧
|
𝜋
)
∥
𝑝
⁢
(
𝑧
)
)
,
	

where the inequality is due to Jensen’s inequality.

Appendix C Proof for Proposition 3.2

By definition:

	
𝑣
~
⁢
(
𝜋
)
	
=
∑
𝑠
𝜌
𝜋
⁢
(
𝑠
)
⁢
𝑣
⁢
(
𝜋
,
𝑠
)
	
		
=
∑
𝑠
𝜌
𝜋
⁢
(
𝑠
)
⁢
∑
𝑎
𝜋
⁢
(
𝑎
|
𝑠
)
⁢
{
𝑟
⁢
(
𝑠
,
𝑎
)
+
𝛾
⁢
∑
𝑠
′
𝑇
⁢
(
𝑠
′
|
𝑠
,
𝑎
)
⁢
𝑣
⁢
(
𝜋
,
𝑠
′
)
}
	
		
=
𝑣
⁢
(
𝜋
)
+
𝛾
⁢
∑
𝑠
𝜌
𝜋
⁢
(
𝑠
)
⁢
∑
𝑎
𝜋
⁢
(
𝑎
|
𝑠
)
⁢
∑
𝑠
′
𝑇
⁢
(
𝑠
′
|
𝑠
,
𝑎
)
⁢
𝑣
⁢
(
𝜋
,
𝑠
′
)
	
		
=
𝑣
⁢
(
𝜋
)
+
𝛾
⁢
∑
𝑠
′
𝜌
𝜋
⁢
(
𝑠
′
)
⁢
𝑣
⁢
(
𝜋
,
𝑠
′
)
	
		
=
𝑣
⁢
(
𝜋
)
+
𝛾
⁢
𝑣
~
⁢
(
𝜋
)
,
	

where the second equality is due to the Bellman equation and the third is from the definition. Therefore,

	
𝑣
~
⁢
(
𝜋
)
=
𝑣
⁢
(
𝜋
)
+
𝛾
⁢
𝑣
~
⁢
(
𝜋
)
⟹
𝑣
~
⁢
(
𝜋
)
=
𝑣
⁢
(
𝜋
)
1
−
𝛾
	

∎

Appendix D Full RepRL Scheme

The diagram presented below illustrates the networks employed in RepRL. The policy’s parameters are inputted into the representation network, which serves as a posterior distribution capturing the latent representation of the policy. Subsequently, a sampling procedure is performed from the representation posterior, followed by the utilization of a linear return encoder, acting as the likelihood, to forecast the return distribution with a linear mean (i.e. the policy’s value). This framework is employed to maximize the Evidence Lower Bound (ELBO).

Figure 7: The full diagram illustrates the networks in RepRL.
Appendix E Decision Set Experiment

The impact of different decision sets on the performance of RepRL was assessed in our evaluation. We conducted tests using three specific decision sets as described in Section 3.3. The evaluation was carried out on a sparse HalfCheetah environment, utilizing the RepES variant. When evaluating the history-based decision set, we considered a history window consisting of 
20
 policies. In the latent-based setting, the parameters corresponding to the selected latent code were obtained using a gradient descent algorithm. The results showed that RepRL exhibited similar performance across the various decision sets tested in different domains.

Figure 8: Plots depict experiments for three decision sets: policy space-based, latent space-based, and history-based. The experiment was conducted on the SparseHalfCheetah environment.
Appendix F Hyperparameters and Network Architecture
F.1 Grid-World

In the GridWorld environment, a 
8
×
8
 grid is utilized with a horizon of 
20
, where the reward is determined by a stochastic function as outlined in the paper: 
𝑟
⁢
(
𝑥
,
𝑦
)
∼
𝒩
⁢
(
𝜇
⁢
(
𝑥
,
𝑦
)
,
𝜎
2
)
,
 where 
𝜎
>
0
 and 
𝜇
⁢
(
𝑥
,
𝑦
)
∝
𝑅
1
⁢
exp
⁡
{
−
(
𝑥
−
𝑥
1
)
2
+
(
𝑦
−
𝑦
1
)
2
𝑎
1
}
+
𝑅
2
⁢
exp
⁡
{
−
(
𝑥
−
𝑥
2
)
2
+
(
𝑦
−
𝑦
2
)
2
𝑎
2
}
+
𝑅
3
⁢
𝟙
{
(
𝑥
,
𝑦
)
=
goal
}
.
 The parameters of the environment are set as 
𝑅
1
=
2.5
,
𝑅
2
=
0.3
,
𝑅
3
=
13
,
𝜎
=
3
,
𝑎
1
=
0.125
,
𝑎
2
=
8
.

The policy employed in this study is a fully-connected network with 3 layers, featuring the use of the 
𝑡
⁢
𝑎
⁢
𝑛
⁢
ℎ
 non-linearity operator. The hidden layers’ dimensions across the network are fixed at 
32
, followed by a 
𝑆
⁢
𝑜
⁢
𝑓
⁢
𝑡
⁢
𝑚
⁢
𝑎
⁢
𝑥
 operation. The state is represented as a one-hot vector. please rephrase the next paragraph so it will sounds more professional: The representation encoder is built from Deep Weight-Space (DWS) layers (Navon et al., 2023), which are equivariant to the permutation symmetry of fully connected networks and enable much stronger representation capacity of deep neural networks compared to standard architectures. The DWS model (DWSNet) comprises four layers with a hidden dimension of 16. Batch normalization is applied between these layers, and a subsequent fully connected layer follows. Notably, the encoder is deterministic, meaning it represents a delta function. For more details, we refer the reader to the code provided in Navon et al. (2023), which was used by us.

In the experimental phase, 
300
 rounds were executed, with 
100
 trajectories sampled at each round utilizing noisy sampling of the current policy, with a zero-mean Gaussian noise and a standard deviation of 
0.1
. The ES algorithm utilized a step size of 
0.1
, while the RepRL algorithm employed a decision set of size 
2048
 without a discount factor (
𝛾
=
1
) and 
𝜆
=
0.1
.

F.2 MuJoCo

In the MuJoCo experiments, both ES and RepES employed a linear policy, in accordance with the recommendations outlined in (Mania et al., 2018). For each environment, ES utilized the parameters specified by Mania et al. (2018), while RepES employed the same sampling strategy in order to ensure a fair comparison.

RepES utilized a representation encoder consisting of 
4
 layers of a fully-connected network, with dimensions of 
2048
 across all layers, and utilizing the 
𝑅
⁢
𝑒
⁢
𝐿
⁢
𝑈
 non-linearity operator. This was followed by a fully-connected layer for the mean and variance. The latent dimension was also chosen to be 
2048
. After each sampling round, the representation framework (encoder and decoder) were trained for 
3
 iterations on each example, utilizing an 
𝐴
⁢
𝑑
⁢
𝑎
⁢
𝑚
 optimizer and a learning rate of 
3
⁢
𝑒
−
4
. When combining learning signals of the ES with RepES, a mixture gradient approach was employed, with 20% of the gradient taken from the ES gradient and 80% taken from the RepES gradient. Across all experiments, a discount factor of 
𝛾
=
0.995
 and 
𝜆
=
0.1
 were used.

F.3 MinAtar

In the MinAtar experiments, we employed a policy model consisting of a fully-connected neural network similar to the one utilized in the GridWorld experiment, featuring a hidden dimension of 
64
. The value function was also of a similar structure, with a scalar output. The algorithms collected five rollout chunks of 
512
 between each training phase.

The regulation coefficient chosen for RepRL was 
1
, while the discount factor and the mixing factor were set as 
𝛾
=
0.995
 and 
𝜆
=
0.1
. The representation encoder used was similar to the one employed in the GridWorld experiments with two layers, followed by a symmetry invariant layer and two fully connected layers.

Generated on Thu Jul 13 18:24:55 2023 by LATExml

