# Neural Architecture Search via Combinatorial Multi-Armed Bandit

Hanxun Huang\*, Xingjun Ma<sup>†</sup>, Sarah M. Erfani\*, James Bailey\*

\* School of Computing and Information Systems, The University of Melbourne, Victoria, Australia  
 {hanxunh}@student.unimelb.edu.au

<sup>†</sup>School of Information Technology, Deakin University, Geelong, Australia

**Abstract**—Neural Architecture Search (NAS) has gained significant popularity as an effective tool for designing high performance deep neural networks (DNNs). NAS can be performed via reinforcement learning, evolutionary algorithms, differentiable architecture search or tree-search methods. While significant progress has been made for both reinforcement learning and differentiable architecture search, tree-search methods have so far failed to achieve comparable accuracy or search efficiency. In this paper, we formulate NAS as a Combinatorial Multi-Armed Bandit (CMAB) problem (CMAB-NAS). This allows the decomposition of a large search space into smaller blocks where tree-search methods can be applied more effectively and efficiently. We further leverage a tree-based method called Nested Monte-Carlo Search to tackle the CMAB-NAS problem. On CIFAR-10, our approach discovers a cell structure that achieves a low error rate that is comparable to the state-of-the-art, using only 0.58 GPU days, which is 20 times faster than current tree-search methods. Moreover, the discovered structure transfers well to large-scale datasets such as ImageNet.

**Index Terms**—Neural Architecture Search, Monte Carlo Tree Search, Multi-Armed Bandit

## I. INTRODUCTION

Deep neural networks (DNNs) have demonstrated superior performance on a wide range of complex learning problems such as image classification [1], [2] and object detection [1], [3]. However, manual design of DNN architectures for a new domain not only requires extensive domain knowledge but also demands a huge amount of time to tune the hyperparameters. In recent years, neural architecture search (NAS) has emerged as a powerful tool for automated design of high-performance DNNs. Given a predefined search space, NAS applies a search strategy to find the optimal DNN architecture according to certain performance objectives. According to the search strategy, existing NAS methods can be categorized into four types: 1) reinforcement learning, 2) evolutionary algorithms, 3) differentiable architecture search, and 4) tree-search methods.

The reinforcement learning (policy gradient) method was the first NAS method that was able to discover architectures competitive to hand-crafted DNNs [4], [5]. However, this type of approach can easily get stuck in sub-optimal solutions [6], [7]. Evolutionary algorithms are also effective methods for NAS. However, they are known to be extremely time-consuming [8]–[10]. Differentiable architecture search is so far the most effective and efficient approach, albeit it may fail in certain search spaces [11]. Compared to differentiable

architecture search, tree-search methods have so far failed to achieve comparable accuracy nor search efficiency [12]–[15]. In this paper, we propose a novel tree-search method that is as competitive as state-of-the-art differentiable architecture search methods in terms of both accuracy and efficiency, and is more robust to different search spaces.

In this work, we propose to formulate NAS as a Combinatorial Multi-Armed Bandit (CMAB) problem, a variant of the Multi-Armed Bandits (MAB) problem. We denote such a formulation of NAS as CMAB-NAS. This formulation provides a unified framework to analyze the efficiency issues of existing tree-search methods, and help identify their key bottleneck: the random sampling of candidate architectures. To address this bottleneck, we propose to use Nested Monte-Carlo Search to solve the CMAB-NAS problem and speedup the sampling of the candidate architectures. In summary, our key contributions are:

- • We propose a novel formulation of NAS as a Combinatorial Multi-Armed Bandit problem (CMAB-NAS), and use Nested Monte-Carlo Search to tackle the problem. Based on CMAB-NAS, we provide a unified view of the efficiency issues of existing tree-search methods.
- • On CIFAR-10, our approach can discover an architecture that is of a low error rate comparable to differentiable architecture search, using only 0.58 GPU days, which is 20 times faster than state-of-the-art tree-search methods. The discovered architecture scales well to ImageNet.
- • We also show that our CMAB-NAS method can perform architecture search more robustly in different search spaces than differentiable architecture search.

## II. RELATED WORK

**Reinforcement Learning.** Reinforcement Learning (Policy gradient) methods were the first NAS methods that were able to discover DNN architectures that can achieve similar or even lower error rates than hand-crafted networks. Early policy gradient methods are very time-consuming, taking 2,000 ~ 22,400 GPU days to find a good architecture [4], [5]. This has been reduced to 0.45 GPU days by weight-sharing across different child networks [6]. However, policy gradient methods can easily get stuck in local optima [6], [7], could producing less optimal architectures.**Evolutionary algorithms.** Evolutionary algorithm-based methods are known to be time-consuming, and can require 3,150 GPU days to produce the same level of error rate as reinforcement learning methods [8]. Although this search cost has been reduced to 4 ~ 8 GPU days by later works [9], [10], they are less efficient than reinforcement learning methods [6].

**Differentiable architecture search (DARTS).** In [16], the authors formulate NAS as a bi-level optimization problem in a differentiable manner, which allows more efficient search using gradient descent. A number of improved variants of DARTS have also been proposed, such as P-DARTS [17], Fair DARTS [18], MiLeNAS [19], and PC-DARTS [20]. The original DARTS method takes 1.5 ~ 4 GPU days to find a state-of-the-art architecture, which was later improved to ~ 0.3 GPU days by [17]–[19], and further to only ~ 0.1 GPU days by [20]. AdaptNAS improve the generalization gap between proxy dataset and target dataset for differentiable methods [21]. While DARTS and its variants focus on searching with proxy network and searching for convolution operations, DARTS can be directly applied on the target task [22], searching for channel dimensions [23], densely connected blocks [24], resource-aware architecture search [25] or be incorporated with continual learning [26]. DARTS and its variants are arguably the most effective and efficient NAS methods to date. However, recent work has shown that DARTS can overfit to the validation set, and thus may fail in certain search spaces [11]. [27], [28] have found that DARTS favours architectures that are of fast convergence rates, which may decrease its generalization performance. These shortcomings of DARTS motivate us to explore other types of search strategies, i.e., tree-search methods. In other words, we explore whether tree-search methods can be made as effective and efficient as DARTS methods, and at the same time, more robust to different search spaces.

**Tree-search methods.** Although tree-search methods have also been proposed for NAS [14], [15] and even be used to search for adversarially robust architectures [29], they often suffer from significant efficiency issues. Even state-of-the-art tree-search methods can take 12 ~ 225 GPU days [14], [15] to find a good architecture on CIFAR-10. In this paper, we aim to improve tree-search methods under a more unified framework.

### III. COMBINATORIAL MULTI-ARMED BANDIT FOR NEURAL ARCHITECTURE SEARCH

In this section, we first describe the search space of NAS: target network structure, cell structure and node structure. We then introduce the Combinatorial Multi-Armed Bandit (CMAB) problem and our formulation of NAS as a CMAB problem (CMAB-NAS). Finally, we propose an efficient tree-search approach to solve the CMAB-NAS problem.

**Network structure.** As illustrated in Figure 1, the target network consists of several stacked convolutional cells, and the goal of NAS is to search for the optimal *cell* structure of the target network. Following previous work [16], we perform architecture search for both the normal convolution cell ( $Cell_N$ ) and the reduction convolution cell ( $Cell_R$ ), that

is,  $(Cell_N, Cell_R)$ . The search is performed on a proxy-network, which is stacked by cells. We use the weight-sharing proxy-network [6], and consider the same basic structure and operations for the proxy network as in [16], except that we removed the zero operation which is not a frequent choice in the previous work.

Fig. 1: Architecture of the proxy network with 8 cells.

Fig. 2: Structures of a cell  $C_k$  and a  $Node_3$

TABLE I: Basic operations in a node.

<table border="1">
<thead>
<tr>
<th>Operation</th>
<th>Kernel Size</th>
</tr>
</thead>
<tbody>
<tr>
<td>Skip Connection</td>
<td>-</td>
</tr>
<tr>
<td>Separable Convolution</td>
<td><math>3 \times 3</math></td>
</tr>
<tr>
<td>Separable Convolution</td>
<td><math>5 \times 5</math></td>
</tr>
<tr>
<td>Dilated Convolution</td>
<td><math>3 \times 3</math></td>
</tr>
<tr>
<td>Dilated Convolution</td>
<td><math>5 \times 5</math></td>
</tr>
<tr>
<td>Max Pooling</td>
<td><math>3 \times 3</math></td>
</tr>
<tr>
<td>Avg Pooling</td>
<td><math>3 \times 3</math></td>
</tr>
</tbody>
</table>

**Cell structure.** Figure 2 illustrates the cell structure to be searched, i.e., a directed acyclic graph with 4 nodes and a concatenation operation. Each node takes inputs from the two previous cells (i.e. *cell inputs*  $C_{k-1}$  and  $C_{k-2}$ ), and also all its predecessor nodes (i.e. *node inputs*). Note that, the first node “Node<sub>1</sub>” does not have any node input as it does not have any predecessor nodes. The outputs of all four nodes in the current cell are then concatenated to form the *cell output*  $C_k$ .

**Node structure.** We denote a node by  $\mathcal{X}_i$ , which consists of two operations on two inputs, i.e.,  $\mathcal{X}_i = (I_1, O_1, I_2, O_2)$ , with  $I$  denoting the input and  $O$  the operation. As illustrated in Figure 2, input  $I$  can be selected from both the cell inputs and the node inputs, while operation  $O$  can be selected from a set of predefined operations. Here, we consider the 7 basic operations defined in Table I. The outputs of the two operations (i.e.  $O_1$  and  $O_2$ ) in a node are then combined into a single *node output* via an element-wise addition.

**Search space size.** We denote the search space of a cell by  $\mathcal{S}$ , the total number of nodes in a single cell by  $N$ , and the total types of operations by  $M$ . We reduce the original search space  $\mathcal{S}$  by removing the duplicates (i.e.  $(I_1, O_1) + (I_2, O_2)$  represents the same node as  $(I_2, O_2) + (I_1, O_1)$ ). This can reduce the size of the search space to  $|\mathcal{S}|$  in Equation (1),which is basically the pair of 2 distinct input-pair combinations plus the combinations where the two operations are identical, for example,  $(I_1, O_1, I_1, O_1)$ . In our setting,  $N=4$  and  $M=7$ , and  $|\mathcal{S}| \approx 6.2 \times 10^9$ . The total size of our search space is  $|\mathcal{S}|^2$  since we are searching for both the normal and the reduction cells (i.e.  $(Cell_N, Cell_R)$ ).

$$|\mathcal{S}| = \prod_{i=1}^N \binom{(i+1)M}{2} + (i+1)M \quad (1)$$

#### A. CMAB Formulation of NAS

The formal definition of CMAB has been presented in previous works as a variant of Multi-Armed Bandits (MAB) [30], [31]. MAB is a classical reinforcement learning problem. Given a finite amount of resources that can be allocated for a number of choices with unknown reward distribution, the goal of MAB is to maximize the expected reward. The choices for MAB are referred to as arms. A solution needs to balance the exploration of the arms with unknown rewards and the exploitation of the arms with known high rewards. CMAB relaxes one large MAB (referred to as a global MAB) into  $n$  smaller size MABs (referred to as local MABs). The *Naive Assumption* [31] (also known as *Monotonicity Assumption* [30]) allows the exploration of the global MAB through local MABs. We formulate NAS as a CMAB problem as follows:

- • Denote a cell (or a set of  $N$  nodes) by  $\mathcal{X} = \{\mathcal{X}_1, \dots, \mathcal{X}_N\}$  with each node  $\mathcal{X}_i$  has a search subspace with size of  $|\mathcal{S}_i|$  different architectural choices, then each choice is called a *local-arm*. The choice for all  $N$  nodes in  $\mathcal{X}$  forms a valid cell, and is called a *global-arm*.
- • The reward distribution  $\mu : \mathcal{X} \rightarrow R$  over each cell  $\mathcal{X}$  is unknown until the cell is determined.

A set of local-arms forming a global-arm in CMAB corresponds to a set of nodes forming a cell (i.e.  $\mathcal{X}$ ) in NAS. The goal of NAS is to find optimal cell structures  $(Cell_N, Cell_R)$  that leads to the best performing child network, or from the CMAB perspective, to find a valid global-arm that optimizes the expected reward. Accordingly, we can define the following local and global MAB problems for NAS.

- • Local MAB: Each node  $\mathcal{X}_i \in \mathcal{X}$  defines a local  $MAB_i$ , which selects the pairs of inputs and operations for node  $\mathcal{X}_i$ .
- • Global MAB:  $MAB_g$ , which considers the whole CMAB problem as a single MAB.  $MAB_g$  selects the pairs of inputs and operations for the entire cell (i.e. for both  $Cell_N$  and  $Cell_R$ ).

As the global-arm can be formed by a combination of local-arms, a *naive assumption* [31] can be made between the global and the local MABs: the global reward  $\mu_g$  for  $MAB_g$  can be approximated by the sum of the local rewards  $\mu_i$  for each  $MAB_i$ , and the local reward  $\mu_i$  only depends on the choices made for  $MAB_i$ . This assumption allows  $MAB_g$  to be optimized via the optimization of each  $MAB_i$ . In other

words, the reward function for a cell can be approximated by the sum of rewards for its all  $N$  nodes:

$$\mu_g(\mathcal{X}) \approx \sum_{i=1}^N \mu_i(\mathcal{X}_i) \quad (2)$$

This assumption allows the decomposition of the cell structure search problem into a set of smaller problems on the node structures.

**Necessity of the naive assumption.** It may be possible to treat each global-arm as a complex local-arm in a single MAB. Then, the cell search problem will become a traditional MAB that directly searches the cell structure without being decomposed into local nodes. However, due to the huge size of the cell search space, it is extremely difficult to apply traditional approaches, where all unplayed arms need to be explored at least once. In contrast, with the naive assumption, there is no need to expand the entire cell search space rather than the local MABs. The number of arms in a local  $MAB_i$  can be calculated using Equation (1) by taking a specific node  $i$ . For example, under the current search settings (eg.  $N=4$ ,  $M=7$ ), the first local MAB (eg.  $MAB_1$ ) only consists of 105 (eg.  $\binom{(1+1)M}{2} + (1+1)M$ ) different arms. Therefore, the CMAB with naive assumption approach is a more practical choice. We denote this formulation of NAS as *CMAB-NAS*.

**Two objectives of CMAB-NAS.** In order to obtain a higher reward (validation accuracy), the proxy network needs to update its weights through training. During the search process, the proxy network can be trained via training of the sampled child networks with weight-sharing [6]. To better train the proxy network, we need to sample a batch of *promising* child networks that have high rewards, as low reward child networks are less helpful or may even harm the next iteration's performance. Taking this into consideration, there will be two objectives for CMAB-NAS: 1) selection of the promising child networks for training the proxy network, and 2) selection of the best performing child network for higher reward. We use regret to define the two objectives. Suppose we use selection policies  $P$  and  $Q$  to select promising child networks and the best performing child network respectively, the two regrets that define the above two objectives of CMAB-NAS are:

$$\text{cumulative regret: } R_c = \sum_{t=1}^T \mu_g^* - \mu_{P(t)} \quad (3)$$

$$\text{simple regret: } R_s = \mu_g^* - \mu_{Q(t)}, \quad (4)$$

where,  $t$  stands for the  $t$ -th play of the  $MAB_g$  (for a total number of  $T$  plays),  $\mu_{P(t)}$  and  $\mu_{Q(t)}$  denote the rewards of the  $t$ -th play with respect to selection policies  $P$  and  $Q$  respectively, and  $\mu_g^*$  denotes the optimal global reward. If the classification accuracy is used as the reward, then  $\mu_g^*$  is the 100% accuracy and the regrets become the error rates. The *cumulative regret* for a total of  $T$  plays defines the objective to select promising (low error rates) child networks for training, while the *simple regret* for one play of  $MAB_g$  defines the objective to select the best child network for final evaluations.#### IV. PROPOSED TREE-SEARCH SOLUTION FOR CMAB-NAS

There already exist several CMAB sampling policies, such as *CUCB* [30] and *Naive Sampling (NS)* [31]. However, they cannot be directly applied to CMAB-NAS. Although the guaranteed logarithm cumulative regret bound of CUCB fits well to the CMAB-NAS objectives, CUCB requires an  $(\alpha, \beta)$ -approximation oracle, which in NAS means that we will need an additional performance prediction model to produce the approximation. On the other hand, *NS* has a linear cumulative regret bound, which is worse than *CUCB*. Considering the huge amount of search space in NAS, it is not optimal for CMAB-NAS neither. Moreover, the two policies consider local MABs separately while ignoring the correlations between different local MABs. However, in NAS, some combinations of operations may perform better. For example, skip connections work well only when there are enough convolution operations, if each local MAB makes independent selections, the resulting cell structure may end up with only skip connections. Alternatively, we propose to use the Nested Monte-Carlo Search (NMCS) [32] to optimize the *cumulative regret*, and the top-k selection strategy to optimize the *simple regret*. NMCS considers each local MAB in a contextual setting (aware of other local MABs' selections) and optimizes each local MAB to approximate the reward of the global MAB based on the *naive assumption*.

##### A. NMCS-guided Architecture Search for CMAB-NAS

We adapted the Nested Monte-Carlo Search (NMCS) to our problem by modifying the sampling strategy. The original NMCS uses random sampling [32], here we use UCB sampling for our problem. The UCB has been proposed to solve traditional MAB problems [33]. Given a local MAB<sub>i</sub> which is a traditional MAB problem, the UCB selection strategy can be defined as:

$$\text{UCB: } \operatorname{argmax}_{arm_j \in \mathcal{X}_i} \bar{\mu}(\mathcal{X}_i, arm_j) + \alpha \sqrt{\frac{2 \ln n_i}{n_j}} \quad (5)$$

where,  $\bar{\mu}(\mathcal{X}_i, arm_j)$  is the average reward of local-arm  $arm_j$  (i.e. the architectural choice for node  $\mathcal{X}_i$ ) in local MAB<sub>i</sub>,  $n_i$  is the number of times local MAB<sub>i</sub> has been played,  $n_j$  is the number of times a local-arm  $arm_j$  has been selected until the current search iteration, and  $\alpha$  is the parameter balancing the trade-off between exploration (i.e.  $\sqrt{\frac{2 \ln n_i}{n_j}}$ ) and exploitation (i.e.  $\bar{\mu}(\mathcal{X}_i, arm_j)$ ). The adapted NMCS maintains a tree structure for each possible architecture of a node and its reward distribution. The path from the root to a leaf in the tree structure defines the exact inputs and operations of a valid cell structure. Initially, the tree has unknown rewards, which can be iteratively estimated by a *simulation* (Algorithm 3) process (will be explained shortly). The sampled child architecture is evaluated on the validation set. The classification accuracy is used as the reward and updated using *backprop*. We assume each node contributes equally to the final reward. Therefore, when updating the reward for each  $arm_j$  in  $\mathcal{X}_i$ , we accumulatively add  $Reward/2N$  to its total rewards and

increment the  $n_i, n_j$  by one. Note, during sampling, we use the average reward (eg.  $\bar{\mu}(\mathcal{X}_i, arm_j)$ ). The overall search procedure is described in Algorithm 1-4. At a high level, it is an iteratively applied two-step process: 1) sample a batch of promising networks for training the proxy network (via the UCB selection), and 2) select the best child network as a candidate for the final evaluation (via the Top-k selection). This two-step process corresponds to the optimization of the two objectives of CMAB-NAS as follows.

---

##### Algorithm 1 NMCS-guided CMAB-NAS

---

```

1: Input: Proxy-Network  $S$ , Epochs  $E$ ,  $B$ ,  $\alpha$ 
2:  $BestChild = \text{None}$ ,  $R_{max} = 0$ 
3: for  $i = 1$  to  $E$  do
4:   Initialize Tree  $Tr$ ,  $Candidates = \text{empty}$ 
5:   for  $j = 1$  to  $B$  do ▷ Objective 1
6:      $Simulation(Tr)$ 
7:      $Child_j = \text{Sample}(Tr, \alpha)$  ▷ UCB sampling
8:      $Candidates \leftarrow Candidates \cup Child_j$ 
9:   end for
10:   $Train(S, Candidates)$ 
11:   $R = \text{Eval}(S, Candidates)$ 
12:   $Backprop(Tr, R)$  ▷ Update rewards for local MABs
13:   $BestChild_i = \text{SearchBest}(Tr)$  ▷ Objective 2
14:   $R_i = \text{Eval}(S, BestChild_i)$ 
15:  if  $R_i > R_{max}$  then
16:     $BestChild = BestChild_i, R_{max} = R_i$ 
17:  end if
18: end for
19: Output:  $BestChild$ 

```

---

##### Algorithm 2 Sample

---

```

1: Input: Root Node  $r$ ,  $\alpha$ 
2: Initialize  $V = \text{empty}$ 
3: repeat
4:   if  $r$  is not fully explored then
5:      $r = \text{Explore}(r)$  ▷ Select the unexplored arm
6:   else
7:      $r = \text{UCB}(r, \alpha)$ 
8:   end if
9:    $V \leftarrow V \cup r$ 
10: until  $V$  is valid cell structure
11: Output:  $V$ 

```

---

##### Algorithm 3 Simulation

---

```

1: Input: Proxy network  $S$ , Tree  $Tr$ , Limit  $L$ ,  $\alpha$ 
2: for  $i = 1$  to  $L$  do
3:    $Child = \text{Sample}(Tr, \alpha)$ 
4:    $R = \text{Eval}(S, Child)$ 
5:    $Backprop(Tr, R)$ 
6: end for

```

---

**Objective 1: NMCS-guided child network sampling.** This step minimizes the *cumulative regret* of CMAB-NAS, corresponds to line 5 to 10 in Algorithm 1. In each epoch (for total---

**Algorithm 4** Search Best

---

```

1: Input: Root Node  $r$ , Limit  $L$ ,  $\alpha$ 
2: Initialize  $V = \text{empty}$ 
3: repeat
4:   Simulation( $Tr, L, \alpha$ )
5:    $r = \text{Policy}(r)$ 
6:    $V \leftarrow V \cup r$ 
7: until  $V$  is valid cell structure
8: Output:  $V$ 

```

---

$E$  epochs), the tree  $Tr$  interacts with a particular snapshot of the proxy network to reinitialize the reward distribution, and sample a new set of promising child networks. In the sampling process, the algorithm visits a particular node ( $MAB_i$ ) in the tree and expands a “new” arm if that particular node is not fully explored. Otherwise, it selects an arm following UCB in Equation (5). This process is repeated until it reaches the leaf node. The selected arms from root to leaf form a valid cell structure, which further forms a valid child network. Each valid architecture is trained with one optimization (gradient) step with a randomly sampled batch of data, then evaluated on the validation set, and the reward is backpropagated to the tree  $Tr$ . This interaction process is called *simulation*. This simulation process will repeat for  $B$  times to obtain a “*ChildSet*” of  $B$  child networks, which are then used to train the proxy network.

**Objective 2: Selection policy for the final child network.** This step minimizes the *simple regret* of CMAB-NAS and corresponds to line 13 in Algorithm 1. After the NMCS-guided simulation and training of the proxy network, the reward distribution of the child networks can be obtained via evaluations on the validation set, along with previous simulations. The selection of the final child network is done by the “*SearchBest*” described in Algorithm 4. Specifically, we use *Top-k selection* (eg.  $k = 1$ ) to minimize the *simple regret* (Equation (3)). The maximum reward can be achieved by exploiting the best performing arm. Recall the *naive assumption* states that the global reward  $\mu_g$  for  $MAB_g$  can be approximated as the sum of rewards  $\mu_i$  for  $MAB_i$ . It is guaranteed that if the reward of each node in set  $\mathcal{X}$  is greater than each node in set  $\mathcal{X}'$ , then the reward of  $\mathcal{X}$  is also greater than  $\mathcal{X}'$ , as mathematically described in Equation (6) below. Based on this guarantee, we explore 3 different selection policies for the final child network: 1) *local optimal*, 2) *local suboptimal* and 3) *local random*.

$$\sum_{i=1}^N \mu_P(\mathcal{X}_i) > \sum_{i=1}^N \mu_Q(\mathcal{X}'_i) \quad (6)$$

$$\text{s.t. } \mu_P(\mathcal{X}_i) > \mu_Q(\mathcal{X}'_i) \text{ for } i = 1, \dots, N$$

$$\text{Local Optimal: } \arg\max_{arm_j \in \mathcal{X}_i} \mu_i(\mathcal{X}_i, arm_j) \quad (7)$$

$$\text{Local Suboptimal: Second-best } arm_m \text{ from } \mu(\mathcal{X}_i) \quad (8)$$

$$\text{Local Random: } \text{Random}(\mathcal{X}_i) \quad (9)$$

At each  $MAB_i$ , we refer to the policy that exploits the best and second-best arm so far as the *local optimal* and *local suboptimal* policy, respectively. We refer to the policy that exploits a random child network as the *local random* policy. The random and local suboptimal policies serve as baselines. Since the original *naive assumption* is only used for exploration of the global arm in  $MAB_g$ , the local policies could lose global optimality by using the *naive assumption* to exploit local MABs. The proposed algorithm and *local optimal* policy do not guarantee the regret bound for global  $MAB_g$ , but on each local  $MAB_i$ , the regret bound for UCB sampling is preserved. In practice, we find that the *local optimal* policy works reasonably well for CMAB-NAS.

### B. A Unified View of Current Tree-search Methods

TABLE II: Comparison with existing tree-search methods. C-Regret: cumulative regret. S-Regret: simple regret. P: the need of a prediction Model.

<table border="1">
<thead>
<tr>
<th>Method</th>
<th>C-Regret</th>
<th>S-Regret</th>
<th>Type</th>
<th>P</th>
</tr>
</thead>
<tbody>
<tr>
<td>PNAS [14]</td>
<td>P&amp;Top-K</td>
<td>Top-K</td>
<td>SMBO</td>
<td>✓</td>
</tr>
<tr>
<td>Wistuba [13]</td>
<td>Random</td>
<td>UCB</td>
<td>MCTS</td>
<td>✓</td>
</tr>
<tr>
<td>DeepArchitect [12]</td>
<td>Random</td>
<td>UCB</td>
<td>MCTS</td>
<td>✓</td>
</tr>
<tr>
<td>AlphaX [15]</td>
<td>Random</td>
<td>UCB</td>
<td>MCTS</td>
<td>✓</td>
</tr>
<tr>
<td><b>CMAB-NAS (ours)</b></td>
<td><b>UCB</b></td>
<td><b>Top-K</b></td>
<td><b>NMCS</b></td>
<td><b>✗</b></td>
</tr>
</tbody>
</table>

Our CMAB-NAS formulation provides a unified framework to understand the efficiency issues of existing tree-search methods [12]–[15]. Existing tree-search methods can be converted to CMAB-NAS based on two criteria: 1) the selection strategy of the child networks for training the proxy network or equivalent processes, and 2) the selection strategy of the final child network. These two criteria correspond to the two objectives of CMAB-NAS, i.e., the *cumulative regret* and the *simple regret*. The different strategies adopted by existing tree-search methods are summarized in Table II. Our approach uses UCB sampling to optimize the *cumulative regret*, while [13], DeepArchitect [12] and AlphaX [15] use a random strategy. Although the proposed solution does not guarantee any global regret bound, for each local  $MAB_i$ , the UCB sampling can achieve the logarithm regret bound [34]. Our empirical results in Section V-C verify that the *naive assumption* works reasonably well for NAS problems. Combining the local UCB sampling and the *naive assumption*, our approach can efficiently select a batch of promising architectures for training. For PNAS [14] which uses SMBO (an improved MCTS approach), the regret bound is related to an additional prediction model and is hard to measure. It also requires additional computational cost to train the prediction model. In fact, all existing tree-search methods rely on a prediction model or a fixed function for performance prediction. Particularly, for AlphaX [15] which is arguably the state-of-the-art tree-search method, the training of its Meta-DNN prediction model is time-consuming. Our CMAB-NAS approach with UCB and Top-k selection strategies can improve tree-search methods to an efficiency level that is as competitive as state-of-the-art DARTS methods, as we will show in the experiments.## V. EXPERIMENTS

In this section, we first compare our CMAB-NAS approach with state-of-the-art methods on both CIFAR-10 and ImageNet datasets. Then, we empirically verify the effectiveness of the *Naive Assumption*. We also show that our method can perform the search more robustly in different search spaces. We report the two standard NAS evaluation metrics: 1) the classification error of the final architecture, and 2) the search cost in GPU days. Following previous work [16], the architecture is searched on simple dataset CIFAR-10 [2] then transferred to other complex datasets such as ImageNet [1]. For all our experiments, we use the same parameter settings for the proxy network and the final network as in the original DARTS method [16]. We use the same type of data augmentations for final network training without any additional tricks.

**Proxy Network setting.** The proxy network is stacked by 6 normal cells and 2 reduction cells with channel size 16 (as illustrated in Figure 1). The proxy network is trained using Stochastic Gradient Descent (SGD) optimizer with momentum 0.9, initial learning rate 0.025 and cosine scheduler [39] without restart.

**Final Network Setting.** For training of the final architecture, for the different dataset, we use the same hyper-parameters and techniques as in [16]. On CIFAR-10, the network consists of 20 stacked cells with a filter size of 36. The network is trained for 650 epochs using SGD with momentum 0.9, initial learning rate 0.025, and cosine scheduler [39]. Typical techniques including cutout [40] and scheduled drop path are also applied. For ImageNet [1], the network consists of 14 stacked cells with filter size 48 and is trained for 250 epochs using SGD with momentum 0.9, initial learning rate 0.025 and decay by 0.97 for every epoch. Colour jittering is used for data augmentation.

**NMCS Parameter setting.** The warm-up of proxy network is performed for 5 epochs. We search for 50 epochs. The exploration parameter  $\alpha$  for UCB is set to 1.0 with decay rate of 0.95 at each epoch. We use top-1 ( $k=1$ ) selection policy for the *simple regret*. ChildNetSet size  $B$  for Algorithm 1 is set to 2500. The number of iterations in *Simulation* is set to  $L = 8$  and  $L = 800$  for Algorithm 3 and Algorithm 4, respectively. The scale up of  $L$  for Algorithm 4 is because a smaller number of iterations is sufficient for exploring shallow local MABs, however, it requires more iterations to ensure (before choosing the next node for the best architecture) that all local MABs are fully expanded.

### A. Results on CIFAR-10 and ImageNet

The results of different methods on CIFAR-10 dataset are reported in Table III. For CMAB-NAS, here we use *local optimal* policy to select the final child network. For differentiable architecture search, we consider the original DARTS [16] method as our main competitor since we are using the same proxy network, search space and hyper-parameter setting. As can be observed, our CMAB-NAS approach is on par with DARTS in terms of error rate, but is  $\sim 7\times$  (eg. 0.58 vs 4 GPU days) more efficient than DARTS and is as efficient as the P-DARTS, an accelerated version of

DARTS. This implies that our proposed backpropagation of the accumulative reward on the tree of NMCS is more efficient than the differentiable-based bilevel optimization of DARTS. Interestingly, despite being a completely different approach, our optimization of the cumulative and the simple regrets share certain similarities with the bilevel optimization of DARTS. Our *local optimal* selection strategy for the final child network over different child nodes in the tree of NMCS has the same effect as the Softmax function of DARTS over all possible operations. Compared with the state-of-the-art tree-search method AlphaX [15], our approach achieves a lower error rate, and is  $20\times$  faster. Particularly, our method can achieve a 2.58% error rate using only 0.58 GPU days, compared to the 2.78% error rate of AlphaX but using 12 GPU days. As we explained in the previous section, this is because AlphaX uses a random policy to optimize the *cumulative regret*, which is less efficient than our UCB. The training of an additional Meta-DNN network for reward prediction is another reason why AlphaX is less efficient. Overall, our CMAB-NAS formulation and the proposed NMCS approach with UCB and Top-1 selection policies has successfully improved tree-search methods to a performance level that is as effective and efficient as state-of-the-art reinforcement learning or DARTS methods. This opens up more opportunities for tree-search based NAS solutions. The best performing cell architecture discovered by our method is illustrated in Appendix.

### B. Transferring to ImageNet

The results on ImageNet [1] are reported in Table IV. Following previous works, here we transfer the discovered cell architectures on CIFAR-10 to ImageNet under the same setting as [15]–[17]. Since this is a direct architecture transfer, the efficiency does not change. The error rate of our CMAB-NAS is within the same range (eg. 24% to 26% top-1 and 7% to 9% top-5 test error) as state-of-the-art methods DARTS and AlphaX. Note that, certain variations may occur when transferring the architecture found on a target dataset to other datasets, as have been discussed in many previous works [8], [14]–[18]. For example, DARTS has lower error rate than PNAS on CIFAR-10, but has higher error rate when transfers to ImageNet. However, the transferred error rates are generally within the same range, as is also the case for our CMAB-NAS method. Overall, the results on ImageNet confirms that our CMAB-NAS approach can indeed find transferable cell architectures.

### C. Empirical Verification of the Naive Assumption

In the previous section, we introduced three possible selection policies for optimizing the *simple regret*: local-optimal (Equation (7)), local-suboptimal (Equation (8)) and local-random (Equation (9)). To show the superiority of the local-optimal policy, we run our CMAB-NAS method five times with different random seeds for each of the three selection policies. The selection policy for the *cumulative regret* is fixed to UCB. The results are reported in Table V. On average, the local-optimal policy outperforms two other policies with a significantly lower error rate. The advantage of local-optimalTABLE III: Results on CIFAR-10. The search cost of CMAB-NAS is measured on a RTX-2080Ti. The best results for non-tree-search and tree-search methods are highlighted in **bold** separately.

<table border="1">
<thead>
<tr>
<th>Method</th>
<th>Test Error (%)</th>
<th>Params (M)</th>
<th>Search Cost (GPU Days)</th>
<th>Type</th>
</tr>
</thead>
<tbody>
<tr>
<td>DenseNet-BC [35]</td>
<td>3.46</td>
<td>25.6</td>
<td>-</td>
<td>Human</td>
</tr>
<tr>
<td>NASNet-A [5] *</td>
<td>2.65</td>
<td>3.3</td>
<td>2000</td>
<td>Reinforcement Learning</td>
</tr>
<tr>
<td>ENAS [6] *</td>
<td>2.89</td>
<td>4.6</td>
<td>0.45</td>
<td>Reinforcement Learning</td>
</tr>
<tr>
<td>AmoebaNet-A [8]</td>
<td>3.34</td>
<td>3.2</td>
<td>3150</td>
<td>Evolutionary Algorithms</td>
</tr>
<tr>
<td>AmoebaNet-B [8]*</td>
<td>2.55</td>
<td>2.8</td>
<td>3150</td>
<td>Evolutionary Algorithms</td>
</tr>
<tr>
<td>EcoNAS [10]*</td>
<td>2.62</td>
<td>2.9</td>
<td>8</td>
<td>Evolutionary Algorithms</td>
</tr>
<tr>
<td>DARTS (Second Order) [16]*</td>
<td>2.76</td>
<td>3.3</td>
<td>4</td>
<td>Differentiable</td>
</tr>
<tr>
<td>SNAS (moderate) [36] *</td>
<td>2.85</td>
<td>2.8</td>
<td>1.5</td>
<td>Differentiable</td>
</tr>
<tr>
<td>P-DARTS [17] *</td>
<td><b>2.50</b></td>
<td>3.4</td>
<td><b>0.3</b></td>
<td>Differentiable</td>
</tr>
<tr>
<td>Firefly [26] *</td>
<td>2.78</td>
<td>3.3</td>
<td>1.5</td>
<td>Differentiable</td>
</tr>
<tr>
<td>AdaptNAS-S (Rot-1) [21] *</td>
<td>2.59</td>
<td>3.6</td>
<td>0.5</td>
<td>Differentiable</td>
</tr>
<tr>
<td>Wistuba [13]</td>
<td>6.45</td>
<td>-</td>
<td>5</td>
<td>Tree Search</td>
</tr>
<tr>
<td>PNAS [14]</td>
<td>3.41</td>
<td>3.2</td>
<td>225</td>
<td>Tree Search</td>
</tr>
<tr>
<td>AlphaX [15]*†</td>
<td>2.78</td>
<td>8.89</td>
<td>12</td>
<td>Tree Search</td>
</tr>
<tr>
<td><b>CMAB-NAS*</b></td>
<td><b>2.58</b></td>
<td>3.80</td>
<td><b>0.58</b></td>
<td>Tree Search</td>
</tr>
</tbody>
</table>

\* Cutout is used for augmentation. † Obtained by running open source code of the pre-trained model AlphaX-1.

TABLE IV: Results of different NAS methods on ImageNet following the same setting as [15]–[17]. The best results for non-tree-search and tree-search methods are highlighted in **bold** separately.

<table border="1">
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="2">Test Error (%)</th>
<th rowspan="2">Params (M)</th>
<th rowspan="2">#Ops</th>
<th rowspan="2">Search Cost (GPU Days)</th>
<th rowspan="2">Type</th>
</tr>
<tr>
<th>Top-1</th>
<th>Top-5</th>
</tr>
</thead>
<tbody>
<tr>
<td>Inception-v1 [37]</td>
<td>30.2</td>
<td>10.1</td>
<td>6.6</td>
<td>1448</td>
<td>-</td>
<td>Human</td>
</tr>
<tr>
<td>MobileNet [38]</td>
<td>29.4</td>
<td>10.5</td>
<td>4.2</td>
<td>569</td>
<td>-</td>
<td>Human</td>
</tr>
<tr>
<td>NASNet-A [5]</td>
<td>26.0</td>
<td>8.4</td>
<td>5.3</td>
<td>564</td>
<td>2000</td>
<td>Reinforcement Learning</td>
</tr>
<tr>
<td>NASNet-B [5]</td>
<td>27.2</td>
<td>8.7</td>
<td>5.3</td>
<td>488</td>
<td>2000</td>
<td>Reinforcement Learning</td>
</tr>
<tr>
<td>AmoebaNet-A [8]</td>
<td>25.5</td>
<td>8.0</td>
<td>5.1</td>
<td>555</td>
<td>3150</td>
<td>Evolutionary Algorithms</td>
</tr>
<tr>
<td>AmoebaNet-B [8]</td>
<td>26.0</td>
<td>8.5</td>
<td>5.3</td>
<td>555</td>
<td>3150</td>
<td>Evolutionary Algorithms</td>
</tr>
<tr>
<td>AmoebaNet-C [8]</td>
<td>24.3</td>
<td>7.6</td>
<td>6.4</td>
<td>570</td>
<td>3150</td>
<td>Evolutionary Algorithms</td>
</tr>
<tr>
<td>EcoNAS [10]</td>
<td>25.2</td>
<td>-</td>
<td>4.3</td>
<td>-</td>
<td>8</td>
<td>Evolutionary Algorithms</td>
</tr>
<tr>
<td>DARTS [16]</td>
<td>26.7</td>
<td>8.7</td>
<td>4.7</td>
<td>574</td>
<td>4</td>
<td>Differentiable</td>
</tr>
<tr>
<td>SNAS (moderate) [36]</td>
<td>27.3</td>
<td>9.2</td>
<td>4.3</td>
<td>522</td>
<td>1.5</td>
<td>Differentiable</td>
</tr>
<tr>
<td>P-DARTS [17]</td>
<td>24.4</td>
<td>7.4</td>
<td>4.9</td>
<td>557</td>
<td><b>0.3</b></td>
<td>Differentiable</td>
</tr>
<tr>
<td>AtomNAS-A [25]</td>
<td>25.4</td>
<td>7.9</td>
<td>3.9</td>
<td>258</td>
<td>112</td>
<td>Differentiable</td>
</tr>
<tr>
<td>AtomNAS-C [25]</td>
<td><b>24.1</b></td>
<td><b>7.3</b></td>
<td>4.7</td>
<td>360</td>
<td>112</td>
<td>Differentiable</td>
</tr>
<tr>
<td>AdaptNAS-S (Rot-1) [21]</td>
<td>24.7</td>
<td>7.6</td>
<td>5.2</td>
<td>575</td>
<td>0.5</td>
<td>Differentiable</td>
</tr>
<tr>
<td>PNAS [14]</td>
<td>25.8</td>
<td>8.1</td>
<td>5.1</td>
<td>588</td>
<td>225</td>
<td>Tree Search</td>
</tr>
<tr>
<td>AlphaX [15]</td>
<td><b>24.5</b></td>
<td><b>7.8</b></td>
<td>5.4</td>
<td>579</td>
<td>12</td>
<td>Tree Search</td>
</tr>
<tr>
<td><b>CMAB-NAS</b></td>
<td>25.8</td>
<td>8.4</td>
<td>5.3</td>
<td>619</td>
<td><b>0.58</b></td>
<td>Tree Search</td>
</tr>
</tbody>
</table>

TABLE V: Error rates of our CMAB-NAS on CIFAR-10 with different selection policies for the *simple regret*. The best results are in **bold**.

<table border="1">
<thead>
<tr>
<th rowspan="2">Policy</th>
<th colspan="6">Error Rate (%)</th>
</tr>
<tr>
<th>Run1</th>
<th>Run2</th>
<th>Run3</th>
<th>Run4</th>
<th>Run5</th>
<th>Avg</th>
</tr>
</thead>
<tbody>
<tr>
<td>Local-optimal</td>
<td><b>2.87</b></td>
<td><b>2.91</b></td>
<td><b>2.58</b></td>
<td><b>2.93</b></td>
<td><b>2.77</b></td>
<td><b>2.81</b></td>
</tr>
<tr>
<td>Local-suboptimal</td>
<td>3.35</td>
<td>2.99</td>
<td>3.12</td>
<td>3.19</td>
<td>3.24</td>
<td>3.18</td>
</tr>
<tr>
<td>Local-random</td>
<td>3.63</td>
<td>3.11</td>
<td>3.10</td>
<td>3.08</td>
<td>3.26</td>
<td>3.24</td>
</tr>
</tbody>
</table>

policy over the local-suboptimal policy provides an empirical proof for the *Naive Assumption* in Equation (2) and its monotonicity guarantee in Equation (6).

#### D. Robust Neural Architecture Search

TABLE VI: Error rates (%) on CIFAR-10 in different search spaces. The results of DARTS and RobustDARTS are from [11]. The best results are highlighted in **bold**.

<table border="1">
<thead>
<tr>
<th>Search Space</th>
<th>DARTS</th>
<th>RobustDARTS(L2)</th>
<th>CMAB-NAS</th>
</tr>
</thead>
<tbody>
<tr>
<td>S2</td>
<td>4.85</td>
<td>3.31</td>
<td><b>2.84 ± 0.17</b></td>
</tr>
<tr>
<td>S4</td>
<td>7.20</td>
<td><b>3.56</b></td>
<td>3.78 ± 0.27</td>
</tr>
</tbody>
</table>

It has been shown that the DARTS methods could overfit to the validation set and leads to poor test performance in certain search spaces [11], and in this case, DARTS and improved variants require additional regularization to avoid overfitting. Different from DARTS, as a non-differentiable approach, tree-search methods are less prone to such overfitting issues. Here, we empirically show that our CMAB-NAS can perform the search robustly in different search spaces. Here, we consider the two search spaces S2 and S4 tested in RobustDARTS [11], where the original DARTS method demonstrated a clear overfitting issue. In particular, S2 consists of  $\{3 \times 3 \text{ SepConv}, \text{SkipConnect}\}$ , while S4 consist of  $\{3 \times 3 \text{ SepConv}, \text{Noise}\}$ . We performed the search 3 times with different random seeds under the same hyperparameter setting as in our previous CIFAR-10 experiment. As shown in Table VI, our method does not suffer from the overfitting problem of DARTS and works reasonably well even without additional regularizing techniques like the L2 regularization (on the inner objective of DARTS) used in RobustDARTS [11].## VI. CONCLUSION

In this paper, we formulated the neural architecture search (NAS) problem as a CMAB problem (CMAB-NAS), which naturally leads us to propose the use of the Nested Monte-Carlo Search (NMCS) with UCB and Top-1 selection policies to solve its two objectives (i.e. *cumulative regret* and *simple regret*). Our CMAB-NAS formulation provides a unified view of current tree-search methods and their efficiency issues. On CIFAR-10 dataset, our approach discovers a cell structure that can achieve 2.58% error rate using only 0.58 GPU days, which is 20 times faster than the current state-of-the-art tree-search method. The cell structures discovered by our method on CIFAR-10 transfer well to large-scale dataset like ImageNet. Our work not only provides a new formulation for NAS but also improves tree-search methods to a performance level that is as effective and efficient as reinforcement learning or differentiable architecture search methods. This opens up more opportunities for tree-search NAS methods as they are robust to different search spaces and can be continuously improved by exploring more advanced sampling strategies under our CMAB-NAS framework.

## VII. ACKNOWLEDGEMENT

This research was undertaken using the LIEF HPC-GPGPU Facility hosted at the University of Melbourne. This Facility was established with the assistance of LIEF Grant LE170100200.

## REFERENCES

1. [1] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, "ImageNet: A Large-Scale Hierarchical Image Database," in *CVPR*, 2009.
2. [2] A. Krizhevsky, G. Hinton, *et al.*, "Learning multiple layers of features from tiny images," 2009.
3. [3] T. Lin, M. Maire, S. J. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, "Microsoft COCO: common objects in context," in *ECCV*, 2014.
4. [4] B. Zoph and Q. V. Le, "Neural architecture search with reinforcement learning," in *ICLR*, 2017.
5. [5] B. Zoph, V. Vasudevan, J. Shlens, and Q. V. Le, "Learning transferable architectures for scalable image recognition," in *CVPR*, 2018.
6. [6] H. Pham, M. Y. Guan, B. Zoph, and Q. V. L. and J. Dean, "Efficient neural architecture search via parameter sharing," in *ICML*, 2018.
7. [7] R. S. Sutton, D. A. McAllester, S. P. Singh, and Y. Mansour, "Policy gradient methods for reinforcement learning with function approximation," in *NeurIPS*, 1999.
8. [8] E. Real, A. Aggarwal, Y. Huang, and Q. V. Le, "Regularized evolution for image classifier architecture search," in *AAAI*, 2019.
9. [9] Z. Lu, I. Whalen, V. Boddeti, Y. D. Dhebar, K. Deb, E. D. Goodman, and W. Banzhaf, "Nsga-net: neural architecture search using multi-objective genetic algorithm," in *GECCO*, 2019.
10. [10] D. Zhou, X. Zhou, W. Zhang, C. C. Loy, S. Yi, X. Zhang, and W. Ouyang, "Econas: Finding proxies for economical neural architecture search," in *CVPR*, 2020.
11. [11] A. Zela, T. Elskens, T. Saikia, Y. Marrakchi, T. Brox, and F. Hutter, "Understanding and robustifying differentiable architecture search," in *ICLR*, 2020.
12. [12] R. Negrinho and G. Gordon, "Deeparchitect: Automatically designing and training deep architectures," *arXiv:1704.08792*, 2017.
13. [13] M. Wistuba, "Finding competitive network architectures within a day using uct," *arXiv:1712.07420*, 2017.
14. [14] C. Liu, B. Zoph, M. Neumann, J. Shlens, W. Hua, L. Li, L. Fei-Fei, A. L. Yuille, J. Huang, and K. Murphy, "Progressive neural architecture search," in *ECCV*, 2018.
15. [15] L. Wang, Y. Zhao, Y. Jinnai, Y. Tian, and R. Fonseca, "Alphax: exploring neural architectures with deep neural networks and monte carlo tree search," *arXiv:1903.11059*, 2019.
16. [16] H. Liu, K. Simonyan, and Y. Yang, "DARTS: differentiable architecture search," in *ICLR*, 2019.
17. [17] X. Chen, L. Xie, J. Wu, and Q. Tian, "Progressive differentiable architecture search: Bridging the depth gap between search and evaluation," in *ICCV*, 2019.
18. [18] X. Chu, T. Zhou, B. Zhang, and J. Li, "Fair darts: Eliminating unfair advantages in differentiable architecture search," *arXiv:1911.12126*, 2019.
19. [19] C. He, H. Ye, L. Shen, and T. Zhang, "Milenas: Efficient neural architecture search via mixed-level reformulation," in *CVPR*, 2020.
20. [20] Y. Xu, L. Xie, X. Zhang, X. Chen, G. Qi, Q. Tian, and H. Xiong, "PC-DARTS: partial channel connections for memory-efficient architecture search," in *ICLR*, 2020.
21. [21] Y. Li, Y. Wang, C. Xu, *et al.*, "Adapting neural architectures between domains," *NeurIPS*, vol. 33, 2020.
22. [22] H. Cai, L. Zhu, and S. Han, "ProxylessNAS: Direct neural architecture search on target task and hardware," in *ICLR*, 2019.
23. [23] A. Wan, X. Dai, P. Zhang, Z. He, Y. Tian, S. Xie, B. Wu, M. Yu, T. Xu, K. Chen, *et al.*, "Fbnetv2: Differentiable neural architecture search for spatial and channel dimensions," in *CVPR*, 2020.
24. [24] J. Fang, Y. Sun, Q. Zhang, Y. Li, W. Liu, and X. Wang, "Densely connected search space for more flexible neural architecture search," in *CVPR*, 2020.
25. [25] J. Mei, Y. Li, X. Lian, X. Jin, L. Yang, A. Yuille, and J. Yang, "Atomnas: Fine-grained end-to-end neural architecture search," in *ICLR*, 2020.
26. [26] L. Wu, B. Liu, P. Stone, and Q. Liu, "Firefly neural architecture descent: a general approach for growing neural networks," *NeurIPS*, vol. 33, 2020.
27. [27] Y. Shu, W. Wang, and S. Cai, "Understanding architectures learnt by cell-based neural architecture search," in *ICLR*, 2020.
28. [28] P. Zhou, C. Xiong, R. Socher, and S. C. Hoi, "Theory-inspired path-regularized differential network architecture search," *NeurIPS*, 2020.
29. [29] H. Chen, B. Zhang, S. Xue, X. Gong, H. Liu, R. Ji, and D. Doermann, "Anti-bandit neural architecture search for model defense," in *ECCV*, pp. 70–85, 2020.
30. [30] W. Chen, Y. Wang, and Y. Yuan, "Combinatorial multi-armed bandit: General framework and applications," in *ICML*, pp. 151–159, 2013.
31. [31] S. Ontañón, "The combinatorial multi-armed bandit problem and its application to real-time strategy games," in *AAAI*, 2013.
32. [32] T. Cazenave, "Nested monte-carlo search," in *IJCAI*, 2009.
33. [33] L. Kocsis and C. Szepesvári, "Bandit based monte-carlo planning," in *ECML*, 2006.
34. [34] P. Auer, N. Cesa-Bianchi, and P. Fischer, "Finite-time analysis of the multiarmed bandit problem," *Machine Learning*, 2002.
35. [35] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely connected convolutional networks," in *CVPR*, 2017.
36. [36] S. Xie, H. Zheng, C. Liu, and L. Lin, "SNAS: stochastic neural architecture search," in *ICLR*, 2019.
37. [37] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. E. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich, "Going deeper with convolutions," in *ICLR*, 2015.
38. [38] A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam, "Mobilenets: Efficient convolutional neural networks for mobile vision applications," *arXiv:1704.04861*, 2017.
39. [39] I. Loshchilov and F. Hutter, "SGDR: stochastic gradient descent with warm restarts," in *ICLR*, 2017.
40. [40] T. DeVries and G. W. Taylor, "Improved regularization of convolutional neural networks with cutout," *arXiv:1708.04552*, 2017.

## APPENDIX

(a) Normal Cell.

(b) Reduction Cell.

Fig. 3: The best performing cell discovered by our CMAB-NAS

