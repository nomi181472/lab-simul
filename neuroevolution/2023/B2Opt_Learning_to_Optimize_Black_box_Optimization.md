# B2Opt: Learning to Optimize Black-box Optimization with Little Budget

Xiaobin Li, *Member, IEEE*, Kai Wu, *Member, IEEE*, and Xiaoyu Zhang, *Member, IEEE*, Handing Wang, *Member, IEEE*, Jing Liu, *Senior Member, IEEE*

**Abstract**—The core challenge of high-dimensional and expensive black-box optimization (BBO) is how to obtain better performance faster with little function evaluation cost. The essence of the problem is how to design an efficient optimization strategy tailored to the target task. This paper designs a powerful optimization framework to automatically learn the optimization strategies from the target or cheap surrogate task without human intervention. However, current methods are weak for this due to poor representation of optimization strategy. To achieve this, 1) drawing on the mechanism of genetic algorithm, we propose a deep neural network framework called B2Opt, which has a stronger representation of optimization strategies based on survival of the fittest; 2) B2Opt can utilize the cheap surrogate functions of the target task to guide the design of the efficient optimization strategies. Compared to the state-of-the-art BBO baselines, B2Opt can achieve multiple orders of magnitude performance improvement with less function evaluation cost. We validate our proposal on high-dimensional synthetic functions and two real-world applications. We also find that deep B2Opt performs better than shallow ones.

**Index Terms**—Learning to Optimize, Black-box Optimization, Transformer, Learned Optimizer, Genetic Algorithm.

## I. INTRODUCTION

MANY tasks, such as neural architecture search [1] and hyperparameter optimization [2], [3], can be abstracted as black-box optimization (BBO) [4], which means that although we can evaluate  $f(x)$  for any  $x \in X$ , we have no access to any other information about  $f$ , such as the Hessian and gradients. Moreover, it is expensive to evaluate  $f(x)$  in most cases. Various hand-designed algorithms, such as genetic algorithms (GAs) [5]–[7], Bayesian optimization [8]–[12], and evolution strategy (ES) [13]–[16], have been designed to solve BBO. Their purpose is to design optimization strategies that allow them to continuously sample better solutions.

Recently, the learning to optimize (L2O) framework [17], [18] gives a new insight into optimization. They leverage the recurrent neural network (RNN), long short-term memory

architecture (LSTM) [19]–[23], multilayer perceptron (MLP) [24], or Transformer [25] as the optimizer to develop optimization methods, aiming at reducing the laborious iterations of hand engineering [21], [26]–[28]. The core of L2O is constructing a solid mapping from the initial solutions to the optimal solution. Moreover, several efforts [20], [29], [30] have coped with the black-box problems, their effectiveness may be hindered by the limited representation capabilities of optimization strategy and a large number of evaluations on expensive black-box functions. As such, they often deal with low-dimensional problems.

The optimization strategy in GA consists of a sequential arrangement of crossover, mutation, and selection operators to handle black-box optimization. But their fixed parameters cause inefficient sampling strategies. In this paper, we obtain a strong optimization strategy representation by coming to parameterize these modules, termed as B2Opt.

We design three modules to realize the mapping from the random solution to the optimal solution. To generate potential individuals to approach the optimal solution, we first develop a self-attention (SA)-based crossover module (SAC) to create potential individuals to approach the optimal solution. Then the output of this module is input into the proposed feed-forward network (FFN)-based mutation module (FM) to escape the optimal local solution. Moreover, the residual and selection module (RSSM) is designed to survive the fittest individuals. RSSM is a pairwise comparison between the output of SAC, FM, and the input population regarding their fitness. We develop a B2Opt Block (OB) consisting of SAC, FM, and RSSM. Finally, we simulate the iteration rule of GAs by stacking OBs to represent strong optimization strategies.

In order to adapt the optimization strategy represented by B2Opt to the target task, we update the parameters of B2Opt based on the information of target task. Moreover, to reduce the number of evaluations to an expensive black-box function, we establish a cheap surrogate function set to train B2Opt. We construct a set of cheap differentiable functions with similar properties to the targeted BBO problems. This training set contains the pair of the initial population and the designed surrogate function. Thus, we can use gradient-based methods, such as stochastic gradient descent and Adam [31], to train B2Opt. In this way, we don't need to query expensive functions during training.

We test B2Opt on six standard functions with high dimension, the neural network training problem, and the planar mechanic arm problem [32]. The experimental results demonstrate the top rank of B2Opt and the strong representation

This work was supported in part by the National Natural Science Foundation of China under Grant 62206205, in part by the Fundamental Research Funds for the Central Universities under Grant XJS211905, in part by the Guangdong High-level Innovation Research Institution Project under Grant 2021B0909050008, and in part by the Guangzhou Key Research and Development Program under Grant 202206030003. (Corresponding author: Jing Liu.)

Xiaobin Li, Kai Wu, and Handing Wang are with School of Artificial Intelligence, Xidian University, Xi'an 710071, China (e-mail: kwu@xidian.edu.cn; 22171214784@stu.xidian.edu.cn; hdwang@xidian.edu.cn).

Xiaoyu Zhang is with School of Cyber Engineering, Xidian University, Xi'an 710071, China (e-mail: xiaoyuzhang@xidian.edu.cn).

Jing Liu is with Guangzhou Institute of Technology, Xidian University, Guangzhou 510555, China (e-mail: neouma@mail.xidian.edu.cn).compared with five state-of-the-art (SOTA) EA baselines, two SOTA Bayesian baselines, and three SOTA L2O methods. The highlights of this paper are shown as follows:

- • Compared with the state-of-the-art BBO methods, B2Opt achieves multiple orders of magnitude performance improvement with fewer function evaluations even if the training dataset is a low-fidelity set of the target black-box function.
- • We design the B2Opt framework to realize the automated GA, which can better represent optimization strategies. It is easier to map random solutions to optimal solutions with fewer evaluation times during optimization.
- • We design a new training strategy to reduce the evaluation cost of expensive objective functions during training. It uses cheap surrogate functions for expensive target tasks instead of directly evaluating the target black-box function.

The rest of this paper is organized as follows. Section II summarizes the related work about learnable optimization strategy. Section III formalizes the studied problem and introduces the background of GAs and Transformer. Section IV presents the content of B2Opt. Section V conducts extensive experiments to evaluate the effectiveness of B2Opt. Finally, Section VI concludes the whole paper.

## II. RELATED WORK

We mainly focus on a learnable optimization strategy for BBO, which is divided into two parts.

### A. Meta-Learn Whole BBO Algorithm

Our proposal belongs to this type. Chen *et al.* [20] first explored meta-learning entire algorithms for low-dimensional BBO. Then, TV *et al.* [33] proposed RNN-Opt, which learned recurrent neural network (RNN)-based optimizers for optimizing real parameter single-objective continuous functions under limited budget constraints. Moreover, they train RNN-Opt with knowledge of task optima. Swarm-inspired meta-optimizer [29] learns in the algorithmic space of both point-based and population-based optimization algorithms. Gomes *et al.* [34] employed meta-learning to infer population-based black-box optimizers that can automatically adapt to specific classes of tasks. These methods parametrize the optimization algorithm by an RNN processing the raw solution vectors and their associated fitness.

### B. Meta-Learn Part BBO Algorithm

This type only learns some parameters of the algorithm, not the overall algorithm. Shala *et al.* [35] meta-learned a policy that configured the mutation step-size parameters of CMA-ES [14]. LES [30] designed a self-attention-based search strategy to discover effective update rules for evolution strategies via meta-learning. Subsequent work LGA [36] employed the above framework to discover the update rules of Gaussian genetic algorithms via Open-ES [16].

The limitations of these methods are shown as follows: 1) to train the meta-optimizer, a large number of expensive black-box functions need to be requested, which is very unrealistic;

2) the established loss function for training the meta-optimizer is challenging to optimize, resulting in a poor representation of the optimization strategy. Thus, we propose B2Opt overcome the above limitations.

## III. PRELIMINARIES

### A. Genetic Algorithms

The crossover, mutation, and selection operators form the basic framework of GAs. GA starts with a randomly generated initial population. Then, genetic operations such as crossover and mutation will be carried out. After the fitness evaluation of all individuals in the population, a selection operation is performed to identify fitter individuals to undergo reproduction to generate offspring. Such an evolutionary process will be repeated until specific predefined stopping criteria are satisfied.

*a) Crossover:* The crossover operator generates a new individual  $\hat{X}_i$  by Eq. (1), and  $cr$  is the probability of the crossover operator.

$$\hat{X}_{i,k}^c = \begin{cases} X_{j,k} & \text{rand}(0,1) < cr \\ X_{i,k} & \text{otherwise} \end{cases} \quad (1)$$

where  $k \in [1, \dots, d]$ . This operator is commonly conducted on  $n$  individuals. After an expression expansion, we reformulate Eq. (1) as  $\sum_{i=1}^n X_i W_i^c$  [37].  $W_i^c$  is the diagonal matrix. If  $W_i^c$  is full of zeros, the  $i$ th individual has no contribution.

*b) Mutation:* The mutation operator brings random changes into the population. Specifically, an individual  $X_i$  in the population goes through the mutation operator to form the new individual  $\hat{X}_i$ , formulated as follows:

$$\hat{X}_{i,k}^m = \begin{cases} \text{rand}(l_k, u_k) & \text{rand}(0,1) < mr \\ \hat{X}_{i,k}^c & \text{otherwise} \end{cases} \quad (2)$$

where  $mr$  is the probability of mutation operator and  $k \in [1, \dots, d]$ . Similarly, Equation (2) can be re-formulated as  $X_i W_i^m$ , where  $W_i^m$  is the diagonal matrix.

*c) Selection:* We introduce the binary tournament mating selection operator in Eq. (3). The selection operator survives individuals of higher quality for the next generation until the number of individuals is chosen.

$$p_i = \begin{cases} 1 & f(X_i) < f(X_k) \\ 0 & f(X_i) > f(X_k) \end{cases}, \quad (X_i, X_k) \in X, \quad (3)$$

where  $p_i$  reflects the probability that  $X_i$  is selected for the next generation, and  $(X_i, X_k)$  in Eq. 3 are randomly selected from the population  $X \cup \hat{X}^m$ .

### B. Transformer

We mainly introduce the core part of Vision Transformer [38], such as the multi-head self-attention layer (MSA), feed-forward network (FFN), layer normalization (LN), and residual connection (RC).Fig. 1. Overall architecture of B2Opt and OB.  $N_x$  stands for B2Opt is composed of  $N_x$  stacked OBs. These OBs can be set to share weights with each other or not share weights with each other.

1) *MSA*: MSA fuses several SA operations to handle the queries ( $Q$ ), keys ( $K$ ), and values ( $V$ ) that jointly attend to information from different representation subspaces. MSA is formulated as follows:  $MultiHead(Q, K, V) = Concat(H_1, H_2, \dots, H_h)W^O$ . where  $Concat$  means concatenation operation. The head feature  $H_i$  can be formulated as:

$$H_i = SA(QW_i^Q, KW_i^K, VW_i^V) \\ = Softmax\left(QW_i^Q(KW_i^K)^T/\sqrt{d_k}\right)VW_i^V = AVW_i^V$$

where  $W_i^Q \in R^{d_m \times d_q}$ ,  $W_i^K \in R^{d_m \times d_k}$ , and  $W_i^V \in R^{d_m \times d_v}$  are parameter matrices for queries, keys, and values, respectively;  $W^O \in R^{h d_v \times d_m}$  maps each head feature  $H_i$  to the output. Moreover,  $d_m$  is the input dimension, while  $d_q$ ,  $d_k$ , and  $d_v$  are hidden dimensions of the corresponding projection subspace;  $h$  is the head number.  $A \in R^{l \times l}$  is the attention matrix of  $h$ th head,  $l$  is the sequence length.

2) *FFN*: FFN employs two cascaded linear transformations with a ReLU activation to handle  $X$ , which is shown as:  $FFN(X) = \max(0, XW_1 + b_1)W_2 + b_2$ , where  $W_1$  and  $W_2$  are weights of two linear layers, and  $b_1$  and  $b_2$  are corresponding biases.

3) *LN*: LN is applied before each layer of MSA and FFN, and the output of LN is calculated by  $X + [MSA|FFN](LN(X))$ .

#### IV. B2OPT

##### A. Problem Definition

A black-box optimization problem can be transformed as a minimization problem, as shown in Eq. (4), and constraints may exist for corresponding solutions:

$$\min_x f(x), s.t. x_i \in [l_i, u_i] \quad (4)$$

where  $x = (x_1, x_2, \dots, x_d)$  represents the solution of optimization problem  $f$ , the lower and upper bounds  $l = (l_1, l_2, \dots, l_d)$  and  $u = (u_1, u_2, \dots, u_d)$ , and  $d$  is the dimension of  $x$ . Suppose  $n$  individuals of one population be  $X_1 = (X_{1,1}, X_{1,2}, \dots, X_{1,d}), \dots, X_n = (X_{n,1}, X_{n,2}, \dots, X_{n,d})$ , then B2Opt are required to find the population near the optimal solution. Note that we only have a very small number of function evaluations to achieve.

##### B. Self-Attention Crossover Module

Inspired by the crossover operator in GAs, we propose an SA-based module to generate potential solutions by maximizing information interaction among individuals in a population. Suppose a population  $X$  is arranged in a non-descending order of fitness, and  $F \in R^{n \times 1}$  be the fitness matrix of  $X$ . Then, this module can be represented as follows:

$$X^c = SAC(X, F) \quad (5)$$

where  $X^c$  is the output population of the SAC module.

Since the object processed by B2Opt is the population with several individuals, and the order of individuals does not affect the population distribution, SA does not require position coding. Standard SA projects the input sequence  $X$  into a  $d$ -dimensional space via the queries ( $Q$ ), keys ( $K$ ), and values ( $V$ ). These three mappings enable the SA module to better capture the characteristics of the problems encountered during training. In other words, these three mappings strengthen the ability of SA to focus on specific issues but do not necessarily make SA have good transferability between different problems. Therefore, we consider removing these three mappings for enhanced transferability and  $X^c = AX$ .  $A \in R^{n \times n}$  is a self-attention matrix that can be learned to maximize inter-individual information interaction based on individual ranking information. Since  $A$  uses individual ranking information instead of fitness value information,  $A$  is invariant to the problems, which is beneficial to improve the generalization performance of the model. This is why the population needs to be sorted in non-descending order.

However, designing the SAC module based solely on population ranking information is a coarse-grained approach. Because this way only considers the location information of individuals in the population but does not consider the fitness relationship between individuals. Therefore, we further introduce fitness information to assist in designing the SAC module:

$$A^F = SA(F) = Softmax\left(FW^Q(FW^K)^T/\sqrt{d_k}\right) \quad (6)$$

Thus,  $X^c = AX + A^F X$ . To better balance the roles of  $A$  and  $A^F$ , we introduce two learnable weights  $W_1^c \in R^{n \times 1}$  and$W_2^c \in R^{n \times 1}$ . Therefore, the final SAC module is shown as follows:

$$X^c = \text{tile}(W_1^c) \odot (AX) + \text{tile}(W_2^c) \odot (A^F X) \quad (7)$$

where  $\odot$  represents Hadamard product and the *tile* copy function extends the vector to a matrix.

### C. FFN-based Mutation Module

In Transformer, each patch embedding carries on directional feature transformation through the FFN module. We take one linear layer as an example:  $X = XW^F$ , where  $W^F$  is the weight of the linear layer, and it is applied to each embedding separately and identically. The similar formula format of mutation operator and FFN inspires us to design a learnable mutation module FM based on FFN with *ReLU* activation function:

$$X^m = FM(X^c) = (\text{ReLU}(X^c W_1^F + b_1)) W_2^F + b_2 \quad (8)$$

where  $X^m$  is the population after the mutation of  $X^c$ .  $W_2^F$  and  $W_1^F$  represent the weight of the second layer of FFN and the weight of the first layer of FFN, respectively.  $b_2$  and  $b_1$  represent the bias of the second layer and the first layer of FFN, respectively. FM is designed to escape the local optimum.

### D. Selection Module

The residual connection in the transformer can be analogized to the selection operation in GA [37]. The RSSM generates the offspring population according to the following equation:

$$\begin{aligned} \hat{X} &= RSSM(X, X^c, X^m) \\ &= \text{Sort}(SM(X, \text{tile}(W_1^s) \odot X \\ &\quad + \text{tile}(W_2^s) \odot X^c + \text{tile}(W_3^s) \odot X^m)) \end{aligned} \quad (9)$$

where  $\hat{X}$  is the fittest population for the next generation; the learnable weights  $W_1^s \in R^{n \times 1}$ ,  $W_2^s \in R^{n \times 1}$ , and  $W_3^s \in R^{n \times 1}$  are the weights for  $X$ ,  $X^c$ , and  $X^m$ , respectively. *Sort*( $X$ ) represents that  $X$  is sorted in non-descending order of fitness. We use quicksort to sort the population based on function fitness. These three learnable weight matrices realize the weighted summation of residual connections, thereby simulating a learnable selection strategy. Meanwhile, introducing residual structure enhances the model's representation ability, enabling B2Opt to form a deep architecture.

SM updates individuals based on a pairwise comparison between the offspring and input population regarding their fitness. Suppose that  $X$  and  $X'$  are the input populations of SM. We compare the quality of individuals from  $X$  and  $X'$  pairwise based on fitness. A binary mask matrix indicating the selected individual can be obtained based on the indicator function  $l_{x>0}(x)$ , where  $l_{x>0}(x) = 1$  if  $x > 0$  and  $l_{x>0}(x) = 0$  if  $x < 0$ . SM forms a new population  $\hat{X}$  by employing Eq. (10).

$$\begin{aligned} \hat{X} &= \text{tile}(l_{x>0}(M_{F'} - M_F)) \odot X \\ &\quad + \text{tile}(1 - l_{x>0}(M_{F'} - M_F)) \odot X' \end{aligned} \quad (10)$$

where the *tile* copy function extends the indication vector to a matrix,  $M_F(M_{F'})$  denotes the fitness matrix of  $X(X')$ .

### E. Structure of B2Opt

B2Opt comprises basic  $t$  B2Opt blocks (OBs), and parameters can be shared among these  $t$  OBs or not. The overall architecture of B2Opt and OB is shown in Figure 1. Each OB consists of SAC, FM, and RSSM in sequential order.  $X^0 \in R^{n \times d}$  represents the initial population input into B2Opt, which must be sorted in non-descending fitness order. In Eq. 11,  $X^{i-1}$  is fed into  $OB_t$  to get  $X^i$ , where  $i \in [1, t]$ . B2Opt realizes the mapping from the random initial population to the target population by stacking  $t$  OBs.

$$\begin{aligned} X^i &= OB(X^{i-1}); \quad X^c = SAC(X^{i-1}, F); \\ X^m &= FM(X^c); \quad X^i = RSSM(X^{i-1}, X^c, X^m) \end{aligned} \quad (11)$$

### F. Training of B2Opt

1) *Goal*: We introduce the parameters  $\theta$  of B2Opt, which need to be optimized. Here, we set  $\theta = \{W_1^c, W_2^c, A, W_1^F, W_2^F, b_1, b_2, W_1^s, W_2^s, W_3^s\}$ .

2) *Training Dataset*: Before introducing the details of the training dataset, fidelity [39] is defined as follows: Suppose the differentiable surrogate functions  $f_1, f_2, \dots, f_m$  are the continuous exact approximations of the black-box function  $f$ . We call these approximations fidelity, which satisfies the following conditions: 1)  $f_1, \dots, f_i, \dots, f_m$  approximate  $f$ .  $\|f - f_i\|_\infty \leq \zeta_m$ , where the fidelity bound  $\zeta_1 > \zeta_2 > \dots > \zeta_m$ . 2) Estimating approximation  $f_i$  is cheaper than estimating  $f$ . Suppose the query cost at fidelity is  $\lambda_i$ , and  $\lambda_1 < \lambda_2 < \dots < \lambda_m$ .

Training data is a crucial factor beyond the objective functions. This paper establishes the training set by constructing a set of differentiable functions related to the optimization objective. This training dataset only contains  $(X_0, f_i(x|\omega))$ , the initial population and objective function, respectively. The variance of  $\omega$  causes the shift in landscapes. The training dataset is designed as follows:

1. 1) Randomly initialize the input population  $X_0$ ;
2. 2) Randomly produce a shifted objective function  $f_i(x|\omega)$  by adjusting the parameter  $\omega$ ;
3. 3) Evaluate  $X_0$  by  $f_i(x|\omega)$ ;
4. 4) Repeat Steps 1)-3) to generate the corresponding dataset. We show the designed training and testing datasets as follows:

$$F^{train} = \{f_1(x|\omega_{1,i}^{train}), \dots, f_m(x|\omega_{m,i}^{train})\} \quad (12)$$

where  $\omega_{m,i}^{train}$  represents the  $i$ th different values of  $\omega$  in the  $m$ th function  $f_m$ .

3) *Loss Function*: B2Opt attempts to search for individuals with high quality based on the available information. The loss function tells how to obtain the parameters of B2Opt to generate individuals closer to the optimal solution by maximizing the difference between the initial population and the output population of B2Opt. The following loss function  $l_i(X^0, f(x|\omega), \theta)$  is employed,

$$l_i = \frac{\frac{1}{|X^0|} \sum_{x \in X^0} f_i(x|\omega) - \frac{1}{|E_\theta(X^0)|} \sum_{x \in E_\theta(X^0)} f_i(x|\omega)}{\left| \frac{1}{|X^0|} \sum_{x \in X^0} f_i(x|\omega) \right|} \quad (13)$$where  $\theta$  denotes parameters of B2Opt ( $E$ ). Equation (13) calculates the average fitness difference between the input and output, further normalized within  $[0, 1]$ . To encourage B2Opt to explore the fitness landscape, for example, the constructed Bayesian posterior distribution over the global optimum [40] can be added to Eq. (13). We have three ways to train B2Opt:

1. 1) When we can construct a differentiable proxy function of the objective function, we use the gradient information of Eq. (13) to train B2Opt.
2. 2) When it is difficult to construct differentiable surrogate functions of the objective function, we can use REINFORCE [41] to estimate these derivatives.
3. 3) We can use ES [26] to train B2Opt. Here, we focus on introducing the training of B2Opt through the first method.

4) *Training B2Opt*: We then train B2Opt under a supervised mode. Since the gradient is unnecessary during the test process, B2Opt can solve BBO problems. To prepare B2Opt to learn a balanced performance upon different optimization problems, we design a loss function formulated as follows:

$$\arg \min_{\theta} l_{\Omega} = -\frac{1}{K} \sum_{X^0 \in \Omega} l_i(X^0, f_i(x|\omega_i^{train}), \theta) \quad (14)$$

We employ Adam [31] method with a minibatch  $\Omega$  to train B2Opt upon training dataset.

5) *Detailed Training Process*: The goal of the training algorithm is to search for parameters  $\theta^*$  of the B2Opt. Before training starts, B2Opt is randomly initialized to get initial parameters  $\theta$ . Then the algorithm will perform the following three steps in a loop until the training termination condition is satisfied:

1. 1) **Step 1**, randomly initialize a minibatch  $\Omega$  comprised of  $K$  populations  $X^0$ ;
2. 2) **Step 2**, for each  $f_i \in F^{train}$ , given training data  $(X^0, f_i)$ , update  $\theta$  by minimizing the  $l_{\Omega}$ ;
3. 3) **Step 3**, given  $X^0$ , update  $\theta$  by minimizing  $-1/m \sum_i l_{\Omega}$ , where  $m$  is the number of functions in  $F^{train}$ . After completing the training process, the algorithm will output  $\theta^*$ .

## V. EXPERIMENTS

### A. Experimental Setup

#### 1) Datasets:

a) *Synthetic Functions*: This paper first employs nine commonly used functions to show the effectiveness of the proposed B2Opt. The characteristics of these nine functions are shown in Table I. Here, B2Opt is trained on  $F^{train}$  generated based on F1-F3, and the target functions are F4-F9. B2Opt is trained on the set of tasks with low fidelity for the target function. We test the generalization performance of B2Opt through this case. Moreover, B2Opt is trained on target functions with different function biases. We then test it on target function with biases unseen in the training stage; we refer it to B2Opt-V2.

Fig. 2. Planar Mechanical Arm.

b) *Planner Mechanic Arm*: We further evaluate the performance of the proposed scheme on the planner mechanic arm problem [42]–[45], a robotic control problem that has been widely used to evaluate the performance of BBO algorithms. It is shown in Fig. 2. The optimization goal of this problem is to search for a set of lengths  $L = (L_1, L_2, \dots, L_n)$  and a set of angles  $\alpha = (\alpha_1, \alpha_2, \dots, \alpha_n)$  so that the distance  $f(L, \alpha, p)$  from the top of the mechanic arm to the target position  $p$  is the smallest, where  $n$  represents the number of segments of the mechanic arm, and  $L_i \in (l_i, u_i)$  and  $\alpha_i \in (-\Pi, \Pi)$  represent the length and angle of the  $i$ th mechanic arm, respectively.  $r$  represents the distance from the target point to the origin of the mechanic arm. Typically,  $f = \sqrt{(\sum_{i=1}^n \cos(\alpha_i)L_i - p_x)^2 + (\sum_{i=1}^n \sin(\alpha_i)L_i - p_y)^2}$ , where  $p_x$  and  $p_y$  represent the  $x$ -coordinate and  $y$ -coordinate of the target point, respectively.

Here,  $n = 100$ ,  $l_i = 0$  and  $u_i = 10$ . We design two groups of experiments:

1. 1) *Simple Case*. We fixed the length of each mechanic arm as  $l_i = 10$  and only searched for the optimal  $\alpha$ .
2. 2) *Complex Case*. We need to search for  $L$  and  $\alpha$  simultaneously. We randomly selected 600 target points within the range of  $r \leq 1000$  to form a set  $S$ , where  $r$  represents the distance from the target point to the origin of the mechanic arm, as shown in Fig. 2. During the training process of B2Opt, a sample point set  $s$  is re-extracted from  $S$  for training every  $T$  training cycle. In the testing process, we extracted 128 target points ( $S^{test}$ ) in the range of  $r \leq 100$ ,  $r \leq 300$ , and  $r \leq 1000$ , respectively, for testing. The purpose of testing in three different regions is to explore the generalization performance of B2Opt further. We evaluate the generalization ability of the algorithm by  $(\sum_s^{S^{test}} f(L, \alpha, s)) / |S^{test}|$ .

c) *Neural Network Training*: We further analyze the performance of B2Opt in the field of neuroevolution. We evaluate the performance of training a convolutional neural network [46] using B2Opt on the MNIST classification task. The structure of this network model is shown in Table II. This task involves a large number of parameters. B2Opt and baselines are tasked with solving the parameters of this neural network to maximize test accuracy. The network does not use any bias, and the total number of parameters is 567. If the SGD algorithm directly optimizes the neural network, the networkTABLE I  
TRAINING AND TESTING FUNCTIONS. F1-F3 ARE TRAINING FUNCTIONS AND F4-F9 ARE TESTING FUNCTIONS. HERE,  $d = \{10, 100\}$ .  $z_i = x_i - b_i$ .

<table border="1">
<thead>
<tr>
<th>ID</th>
<th>Functions</th>
<th>Range</th>
</tr>
</thead>
<tbody>
<tr>
<td>F1</td>
<td><math>\sum_i |w_i \sin(x_i - b_i)|</math></td>
<td><math>x \in [-10, 10], b \in [-10, 10]</math></td>
</tr>
<tr>
<td>F2</td>
<td><math>\sum_i |x_i - b_i|</math></td>
<td><math>x \in [-10, 10], b \in [-10, 10]</math></td>
</tr>
<tr>
<td>F3</td>
<td><math>\sum_i |(x_i - b_i) + (x_{i+1} - b_{i+1})| + \sum_i |x_i - b_i|</math></td>
<td><math>x \in [-10, 10], b \in [-10, 10]</math></td>
</tr>
<tr>
<td>F4(Sphere)</td>
<td><math>\sum_i z_i^2</math></td>
<td><math>x \in [-100, 100], b \in [-50, 50]</math></td>
</tr>
<tr>
<td>F5</td>
<td><math>\{|z_i|, 1 \leq i \leq D\}</math></td>
<td><math>x \in [-100, 100], b \in [-50, 50]</math></td>
</tr>
<tr>
<td>F6(Rosenbrock)</td>
<td><math>\sum_{i=1}^{D-1} (100(z_i^2 - z_{i+1})^2 + (z_i - 1)^2)</math></td>
<td><math>x \in [-100, 100], b \in [-50, 50]</math></td>
</tr>
<tr>
<td>F7(Rastrigin)</td>
<td><math>\sum_{i=1}^D (z_i^2 - 10 \cos(2\pi z_i) + 10)</math></td>
<td><math>x \in [-5, 5], b \in [-2.5, 2.5]</math></td>
</tr>
<tr>
<td>F8(Griewank)</td>
<td><math>\sum_{i=1}^D \frac{z_i^2}{4000} - \prod_{i=1}^D \cos(\frac{z_i}{\sqrt{i}}) + 1</math></td>
<td><math>x \in [-600, 600], b \in [-300, 300]</math></td>
</tr>
<tr>
<td>F9(Ackley)</td>
<td><math>-20 \exp(-0.2 \sqrt{\frac{1}{D} \sum_{i=1}^D z_i^2}) - \exp(\frac{1}{D} \sum_{i=1}^D \cos(2\pi z_i)) + 20 + \exp(1)</math></td>
<td><math>x \in [-32, 32], b \in [-16, 16]</math></td>
</tr>
</tbody>
</table>

TABLE II  
THE SMALL-SCALE CLASSIFICATION NEURAL NETWORK FOR MNIST.

<table border="1">
<thead>
<tr>
<th>Layer ID</th>
<th>Layer Type</th>
<th>Padding</th>
<th>Stride</th>
<th>Kernel Size</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>depth-wise convolution</td>
<td>✗</td>
<td>1</td>
<td>1×5×5</td>
</tr>
<tr>
<td>2</td>
<td>point-wise convolution</td>
<td>✗</td>
<td>1</td>
<td>3×1×1×1</td>
</tr>
<tr>
<td>3</td>
<td>ReLU</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td>4</td>
<td>max pooling</td>
<td>✗</td>
<td>2</td>
<td>2×2</td>
</tr>
<tr>
<td>5</td>
<td>depth-wise convolution</td>
<td>✗</td>
<td>1</td>
<td>3×5×5</td>
</tr>
<tr>
<td>6</td>
<td>point-wise convolution</td>
<td>✗</td>
<td>1</td>
<td>16×3×1×1</td>
</tr>
<tr>
<td>7</td>
<td>ReLU</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td>8</td>
<td>max pooling</td>
<td>✗</td>
<td>2</td>
<td>2×2</td>
</tr>
<tr>
<td>9</td>
<td>depth-wise convolution</td>
<td>✗</td>
<td>1</td>
<td>16×4×4</td>
</tr>
<tr>
<td>10</td>
<td>point-wise convolution</td>
<td>✗</td>
<td>1</td>
<td>10×16×1×1</td>
</tr>
<tr>
<td>11</td>
<td>softmax</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
</tr>
</tbody>
</table>

can achieve about 90% test accuracy, which proves that the network architecture is effective. We use 1000 samples from the MNIST dataset to form the training set and another 1000 samples as the test set.

2) *Baselines*: B2Opt is compared with the state-of-the-art BBO methods, including non-L2O and L2O-based.

**EA baselines.** DE(DE/rand/1/bin) [47], ES( $(\mu, \lambda)$ -ES), IPOP-CMA-ES [48], L-SHADE [49], and CMA-ES [50], where DE [47] and ES are implemented based on Geatpy [51], CMA-ES and IPOP-CMA-ES are implemented by cmaes<sup>1</sup>, and L-SHADE is implemented by pyade<sup>2</sup>. The reasons for choosing these EA baselines are the following:

- • DE(DE/rand/1/bin): A classic numerical optimization algorithm.
- • ES( $(\mu, \lambda)$ -ES): A classic variant of the evolution strategy.
- • CMA-ES: CMA-ES is often considered the state-of-the-art method for continuous domain optimization under challenging settings (e.g., ill-conditioned, non-convex, non-continuous, multimodal).
- • IPOP-CMA-ES: The state-of-the-art variant of CMA-ES.

- • L-SHADE: The state-of-the-art variant of DE.

**Bayesian optimization.** Dragonfly [52] and SAASBO [53] are employed as a reference. Here are the reasons for choosing these two algorithms:

- • Dragonfly: A representative algorithm for Bayesian optimization.
- • SAASBO: A state-of-the-art large-scale Bayesian optimization algorithm.

**L2O-based methods.** We chose three state-of-the-art L2O-based methods for comparison with B2Opt.

- • L2O-swarm [29]: A representative L2O method for BBO.
- • LES [30]: A recently proposed learnable ES. It uses a data-driven approach to discover new ES with strong generalization performance and search efficiency.
- • LGA [36]: A recently proposed learnable GA that discovers new GA in a data-driven manner. The learned algorithm can be applied to unseen optimization problems, search dimensions, and evaluation budgets.

**B2Opt.** We design three B2Opt models, including 3 *OBs* with *WS*, 5 *OBs* without *WS*, and 30 *OBs* with *WS*. 3 *OBs* with *WS* represents that B2Opt has 3 OB modules, and these OBs share the weights of each other. Each OB consists of 1 SAC, 1 FM, and 1 RSSM. In general, B2Opt represents 30 OBs with *WS*.

3) *Parameters*: Next, we show the parameter settings of these methods:

**B2Opt.** In 30 *OBs* with *WS*, these 30 OBs share parameters. 5 *OBs* without *WS* has 5 OBs, and no parameters are shared among them. During the training process, B2Opt is trained for 1000 epochs. The initial learning rate (*lr*) was set to 0.01 and  $lr = lr \times 0.9$  every 100 cycles. The 2-norm of the gradient is clipped so that it is not larger than 10. The bias of the function is regenerated each epoch, and a new batch of random initial populations is generated.

**Baselines.** For all cases, we choose the optimal hyper-parameters. For ES and DE, the population size is set to 100,  $\lambda = 0.5$ ,  $cr = 0.5$ ,  $F = 0.5$ , and the rest of the parameters are the default ones set by geatpy. For CMA-ES, the rest of the parameters take the default values of the

<sup>1</sup><https://github.com/CyberAgentAILab>

<sup>2</sup><https://github.com/xKuZz/pyade>Fig. 3. The compared results of all methods on six functions. The structure of B2Opt is 30 OBs with WS. The left schematic diagram and the right schematic diagram are the test results when the dimensions are 10 and 100, respectively. In order to present the experimental results more intuitively, we mark the experimental results as  $a$ , and the figure shows  $\exp(-a)$  normalized by z-score. Larger values represent better performance.

package camae. The population size of L-SHADE is 100, and the rest of the parameters adopt the default parameters of the package pyade. The initial population size of IPOP-CMA-ES is 30, and the population growth factor is 2. L-SHADE and IPOP-CMA-ES are known for their fast convergence. For all EA baselines, the maximum number of evaluations is set to 3100, which is consistent with the number of evaluations of B2Opt (30 OBs with WS). We set the maximum number of generations of Dragonfly and SAASBO to 100. Because the runtime overhead of these two Bayesian optimization algorithms is very expensive, it takes several days to complete 100 generations when  $d = 100$ . Even at  $d = 10$ , it takes many hours to complete 100 evaluations. But B2Opt can finish running in 1 second, so the expensive actual running time overhead of SAASBO and Dragonfly is unacceptable. The population size of LGA and LES is 100, and the maximum number of function evaluations is 3100. We use the optimal trained model parameters officially provided by their project website<sup>3</sup>. The  $F^{Train}$  and  $F^{test}$  is the same as that of B2Opt. In the test phase, L2O-Swarm was run to convergence.

To ensure validity, all experimental results were averaged to the optimal values of the five groups of experiments, each of which performed 64 runs. All experiments were performed on a Ubuntu 20.04 PC with Intel(R) Core I7 (TM) I3-8100 CPU at 3.60GHz and NVIDIA GeForce GTX 1060.

## B. Results

*a) Synthetic Functions:* Here, the structure of B2Opt is 30 OBs with WS. The results on synthetic functions are provided in Fig. 3. We also plot the convergence curves of all methods on F4-F9 with  $d = \{10, 100\}$ , as shown in

Figs. 4 and 5. B2Opt converges quickly and can obtain better solutions with little budget. The fewer function evaluations, the greater the advantage of B2Opt. When  $d = 10$ , B2Opt achieved better performance than all SOTA methods except for L-SHADE in all cases. B2Opt loses five cases to L-SHADE and outperforms L-SHADE in one case (F7). L-SHADE is the optimal DE variant, which is heavily engineered, and intricately hand-designed by researchers combining a wealth of expert knowledge. It has been iteratively updated many times over the years. L-SHADE can obtain high-performance initialized populations on F4-F9, far better than B2Opt's. However, the convergence speed of B2Opt is much faster than that of L-SHADE, and B2Opt is highly competitive, especially with a minimal number of function evaluations. However, when  $d = 100$ , B2Opt outperforms all SOTA methods in F4-F9. When the problem dimension is 10, the problem is relatively simple, and when the problem dimension is 100, the problem becomes very complicated. B2Opt achieves the best performance on all tested functions in complex cases. L-SHADE has strong advantages over simple problems. Note that the hyperparameters of L-SHADE are optimally tuned to the performance of the target task. However, B2Opt is only trained on the low-fidelity surrogate functions F1-F3 of the target task.

LES and LGA are the latest learnable evolutionary algorithms with powerful performance. They are trained on a large number of BBOB functions, many of which are the same or similar to the test functions shown in Table I. For example, the LES and LGA training datasets include F4, F5, F6, and F8. As shown in Figs. 4 and 5, the initial population have achieved good performance. B2Opt only performs simple training on F1, F2, and F3 in Table I, and F4-F9 is unseen for B2Opt in the training stage. However, B2Opt outperforms LES and

<sup>3</sup><https://github.com/RobertTLange/evosax>Fig. 4. The convergence curves of B2Opt and other baselines. It shows the convergence curve of these algorithms on functions in Table I when  $d = 10$ .

LGA in all tested functions.

B2Opt-V2 outperforms B2Opt overall. However, B2Opt-V2 performs worse than B2Opt on F8. B2Opt-V2 is trained on a high-fidelity surrogate function of the target task. We found that F8 has a "Rotated" feature and a wide range of  $z_i$  values, which cause B2Opt-V2 to be overfitted. B2Opt-V2 outperforms B2Opt after we increase the training sample size.

These cases also show the excellent generalization ability of B2Opt on more tasks unseen during the training stage. The transferability of B2Opt is proportional to the fitness landscape similarity between the training set and the problem. Although new problem attributes are unavailable in the training set, B2Opt can perform better. However, this conclusion only holds when the similarity between the problem and training dataset is high. Now we further analyze the detailed reasons for the excellent effect of B2Opt.

We train B2Opt based on F1-F3 and make it successfully optimize F4-F9, and we also analyze the reason for this case.

**From the expression of the functions**, F1-F3 are low-fidelity surrogate functions of the target functions F4-F9. F1-F3 contain the information of objective functions (F4-F9). For example, F7 can be decomposed into  $\sum_{i=1}^D z_i^2 - \sum_{i=1}^D 10 \cos(2\pi z_i) + \sum_{i=1}^D 10$ . F2 is the low-fidelity surrogate function of  $\sum_{i=1}^D z_i^2$ . F1 is the low-fidelity surrogate function of  $\sum_{i=1}^D 10 \cos(2\pi z_i)$ . For other functions in F4-F9, we can find similar surrogate functions from F1-F3. B2Opt can use this information to maximize the matching degree between the learned optimization strategy and the objective function.

However, F6 is less similar to F1-F3 than F4, F5, and F7-F9. Therefore, although the performance of B2Opt on F6 is better than that of the comparison algorithm, it still needs improvement.

**From the perspective of landscape features**, F1-F3 include the following features: unimodal, multimodal, separable, and non-separable. The landscape features included in F4-F9 are as follows:

- • F4: Unimodal, Separable
- • F5: Unimodal, Separable
- • F6: Multimodal, Non-separable, Having a very narrow valley from local optimum to global optimum, Ill-conditioned
- • F7: Multimodal, Separable, Asymmetrical, Local optima's number is huge
- • F8: Multi-modal, Non-separable, Rotated
- • F9: Multi-modal, Non-separable, Asymmetrical

The landscape features of F4 and F5 can be found in F1-F3. F6-F9 all have new features. The interference strength of different characteristics to landscape is arranged as follows: **Having a very narrow valley from local optimum to global optimum > Asymmetrical, Local optima's number is huge > Asymmetrical > Rotated**. Therefore, B2Opt has the best generalization performance on F4, F5, F8, the second-best generalization performance on F7 and F9, and the worst on F6.

*b) Planner Mechanic Arm:* The detailed experimental results are in Tables III. Note that the parentheses in the tableFig. 5. The convergence curves of B2Opt and other baselines. It shows the convergence curve of these algorithms on functions in Table I when  $d = 100$ .

TABLE III

THE RESULTS OF ALL BASELINES ON THE PLANAR MECHANICAL ARM. THE STRUCTURE OF B2OPT IS 30 OBs with WS. SIMPLE CASE (SC): SEARCHING FOR DIFFERENT ANGLES WITH FIXED LENGTHS. COMPLEX CASE (CC): SEARCHING FOR DIFFERENT ANGLES AND LENGTHS. W/T/L REPRESENTS WIN/TIE/LOSS TO B2OPT.

<table border="1">
<thead>
<tr>
<th></th>
<th><math>r</math></th>
<th>DE</th>
<th>ES</th>
<th>CMA-ES</th>
<th>L2O-Swarm</th>
<th>L-SHADE</th>
<th>I-POP-CMA-ES</th>
<th>LGA</th>
<th>LES</th>
<th>B2Opt</th>
<th>Untrained</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">SC</td>
<td>100</td>
<td>1.60(0.08)</td>
<td>10.7(0.40)</td>
<td>1.25(0.05)</td>
<td>40.4(3.89)</td>
<td>0.66(0.03)</td>
<td>1.17(0.02)</td>
<td>540(1.97)</td>
<td>546(8.22)</td>
<td><b>0.38(0.01)</b></td>
<td>0.8(0.05)</td>
</tr>
<tr>
<td>300</td>
<td>2.65(0.08)</td>
<td>35(2.41)</td>
<td>1.84(0.07)</td>
<td>69.5(3.77)</td>
<td>0.54(0.02)</td>
<td>1.21(0.06)</td>
<td>562(2.1)</td>
<td>572(4.92)</td>
<td><b>0.52(0.02)</b></td>
<td>4.26(0.33)</td>
</tr>
<tr>
<td>1000</td>
<td>155(1.45)</td>
<td>153(2.19)</td>
<td>154(2.31)</td>
<td>176(7.20)</td>
<td><b>0.4(0.03)</b></td>
<td>36.8(1.00)</td>
<td>716(1.74)</td>
<td>719(4.03)</td>
<td>16.9(0.30)</td>
<td>126(0.85)</td>
</tr>
<tr>
<td rowspan="3">CC</td>
<td>100</td>
<td>1.17(0.03)</td>
<td>9.22(0.52)</td>
<td>0.78(0.02)</td>
<td>31.9(1.78)</td>
<td>0.25(0.02)</td>
<td>0.72(0.02)</td>
<td>108(1.34)</td>
<td>223(6.51)</td>
<td><b>0.17(0.01)</b></td>
<td>0.94(0.04)</td>
</tr>
<tr>
<td>300</td>
<td>13.6(0.53)</td>
<td>38.3(0.60)</td>
<td>4.77(0.37)</td>
<td>89.1(1.96)</td>
<td>54.5(5.23)</td>
<td>0.92(0.02)</td>
<td>161(2.67)</td>
<td>259(4.29)</td>
<td><b>0.22(0.01)</b></td>
<td>18.6(0.39)</td>
</tr>
<tr>
<td>1000</td>
<td>278(1.96)</td>
<td>235(1.41)</td>
<td>239(0.83)</td>
<td>262(2.99)</td>
<td>218(0.89)</td>
<td>167(1.15)</td>
<td>421(0.93)</td>
<td>493(1.15)</td>
<td><b>14.7(0.70)</b></td>
<td>214(0.93)</td>
</tr>
<tr>
<td>w/t/l</td>
<td></td>
<td>0/0/6</td>
<td>0/0/6</td>
<td>0/0/6</td>
<td>0/0/6</td>
<td>1/0/5</td>
<td>0/0/6</td>
<td>0/0/6</td>
<td>0/0/6</td>
<td>-</td>
<td>0/0/6</td>
</tr>
</tbody>
</table>

TABLE IV

THE DESIGNED CONVOLUTION MODULE TO REPLACE SAC. ONE REPRESENTS ONE REFLECTION PADDING.

<table border="1">
<thead>
<tr>
<th>Layer ID</th>
<th>layer type</th>
<th>padding</th>
<th>stride</th>
<th>kernel size</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>depth-wise convolution</td>
<td>one</td>
<td>1</td>
<td><math>567 \times 3 \times 3</math></td>
</tr>
<tr>
<td>2</td>
<td>ReLU</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td>3</td>
<td>depth-wise convolution</td>
<td>one</td>
<td>1</td>
<td><math>567 \times 3 \times 3</math></td>
</tr>
<tr>
<td>4</td>
<td>ReLU</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td>5</td>
<td>depth-wise convolution</td>
<td>one</td>
<td>1</td>
<td><math>567 \times 3 \times 3</math></td>
</tr>
<tr>
<td>6</td>
<td>ReLU</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td>7</td>
<td>depth-wise convolution</td>
<td>one</td>
<td>1</td>
<td><math>567 \times 3 \times 3</math></td>
</tr>
</tbody>
</table>

show the standard deviation if not otherwise specified. The

structure of B2Opt is 30 OBs with WS. Untrained represents the untrained B2Opt. We randomly selected 600 target points within the range of  $r \leq 1000$  to form a set  $S$ , where  $r$  represents the distance from the target point to the origin of the mechanic arm, as shown in Fig. 2. During the training process of B2Opt, a sample point set  $s$  is re-extracted from  $S$  for training every  $T$  training cycle. In the testing process, we extracted 128 target points ( $S^{test}$ ) in the range of  $r \leq 100$ ,  $r \leq 300$ , and  $r \leq 1000$ , respectively, for testing. The purpose of testing in three different regions is to explore the generalization performance of B2Opt further. B2Opt wins all SOTA methods and achieves the best results in comparison with other algorithms. B2Opt loses once to L-SHADE in SC with  $r = 1000$ . However, in the other 5 cases, B2Opt outperforms L-SHADE. In particular, on the complex problem (CC), B2Opt is more dominant and stable than L-SHADE.TABLE V  
THE CLASSIFICATION ACCURACY OF ALL METHODS ON THE MNIST DATASET. DATASIZE REPRESENTS THE PROPORTION OF DATA SETS INVOLVED IN TRAINING.

<table border="1">
<thead>
<tr>
<th>Datasize</th>
<th>B2Opt</th>
<th>ES</th>
<th>DE</th>
<th>CMA-ES</th>
<th>I-POP-CMA-ES</th>
<th>L-SHADE</th>
<th>LGA</th>
<th>LES</th>
</tr>
</thead>
<tbody>
<tr>
<td>0.25</td>
<td><b>0.69(0.01)</b></td>
<td>0.21(0.02)</td>
<td>0.21(0.01)</td>
<td>0.57(0.04)</td>
<td>0.33(0.02)</td>
<td>0.31(0.03)</td>
<td>0.53(0.02)</td>
<td>0.14(0.02)</td>
</tr>
<tr>
<td>0.5</td>
<td><b>0.80(0.01)</b></td>
<td>0.21(0.01)</td>
<td>0.24(0.01)</td>
<td>0.61(0.03)</td>
<td>0.37(0.04)</td>
<td>0.29(0.01)</td>
<td>0.51(0.01)</td>
<td>0.25(0.03)</td>
</tr>
<tr>
<td>0.75</td>
<td><b>0.85(0.00)</b></td>
<td>0.21(0.01)</td>
<td>0.22(0.02)</td>
<td>0.61(0.03)</td>
<td>0.45(0.02)</td>
<td>0.30(0.07)</td>
<td>0.52(0.05)</td>
<td>0.23(0.07)</td>
</tr>
<tr>
<td>1</td>
<td><b>0.86(0.00)</b></td>
<td>0.25(0.02)</td>
<td>0.23(0.02)</td>
<td>0.67(0.01)</td>
<td>0.42(0.02)</td>
<td>0.30(0.01)</td>
<td>0.54(0.04)</td>
<td>0.28(0.01)</td>
</tr>
</tbody>
</table>

The optimization objective of this problem does not exist in the training sets of LGA and LES. A sharp performance degradation of LGA and LES can be observed. We also find the surprising phenomenon that *Untrained* outperforms most of the baselines, which suggests that the randomly initialized B2Opt also possesses some ability to produce and select potential solutions.

c) *Neural Network Training*: The detailed results are shown in Table V. The evaluation metric is test accuracy on the test set. While training B2Opt, the optimization objective of B2Opt is to minimize the cross-entropy loss, which is a surrogate function for metric accuracy. However, in the testing stage, the optimization goal of B2Opt and other baselines is to maximize the accuracy of the training set. We select 25%, 50%, 75%, and 100% data from the training set for training, respectively, constituting surrogate problems with different fidelity levels. B2Opt has 3 OBs that do not share weights. We replaced SAC with a lightweight convolution module with the structure shown in Table IV. This substitution is made because we expect B2Opt to perform more stably during training for this task. The population size of B2Opt is 36, meaning its number of evaluations is  $36 \times 4 = 144$ . The maximum number of evaluations for L-SHADE and I-POP-CMA-ES is 3000, which is  $3000/144 \approx 21$  times that of B2Opt. LGA and LES have a maximum number of generations of 100, and they are evaluated  $10000/144 \approx 69$  times as many times as B2Opt. Even in this unfair case, B2Opt achieves the best results for all fidelity levels. Fig. 6 shows the convergence curve of B2Opt, LES, LGA, and CMA-ES on this task. B2Opt can achieve the best solution with the least number of evaluations.

Fig. 6. The convergence curves of B2Opt and baselines on the task of neural network training. The X-axis represents the number of evaluations, and the Y-axis represents the test accuracy. B2Opt stops iterating when the number of evaluations is 144, and the red dotted line is for convenient comparison.

TABLE VI  
THE PERFORMANCE OF DIFFERENT B2OPT STRUCTURES.

<table border="1">
<thead>
<tr>
<th><math>f</math></th>
<th><i>Untrained</i></th>
<th><i>5 OBs without WS</i></th>
<th><i>30 OBs with WS</i></th>
<th><i>3 OBs with WS</i></th>
</tr>
</thead>
<tbody>
<tr>
<td>F4</td>
<td>0.28(0.09)</td>
<td>0.08(0.03)</td>
<td><b>1.2e-4(5e-5)</b></td>
<td>8.16(3.44)</td>
</tr>
<tr>
<td>F5</td>
<td>0.37(0.05)</td>
<td>0.15(0.03)</td>
<td><b>0.008(0.002)</b></td>
<td>1.47(0.40)</td>
</tr>
<tr>
<td>F6</td>
<td>45.8(16.9)</td>
<td>15.43(2.34)</td>
<td><b>8.93(0.03)</b></td>
<td>1891(1396)</td>
</tr>
<tr>
<td>F7</td>
<td>1.08(0.72)</td>
<td>4.43(1.82)</td>
<td><b>0.01(0.03)</b></td>
<td>35.72(8.52)</td>
</tr>
<tr>
<td>F8</td>
<td>0.69(0.09)</td>
<td>0.06(0.03)</td>
<td><b>1e-5(2e-5)</b></td>
<td>0.82(0.10)</td>
</tr>
<tr>
<td>F9</td>
<td>0.85(0.20)</td>
<td>0.29(0.07)</td>
<td><b>0.01(0.003)</b></td>
<td>3.28(1.00)</td>
</tr>
</tbody>
</table>

### C. Parameter Analysis

We analyze the effect of the deep structure, learning rate, and weight sharing between OBs on B2Opt.

1) *B2Opt Architectures*: We consider the performance of different B2Opt architectures. The experimental results are shown in Table VI. They were sorted from good to worst based on their performance, and the result is *30 OBs with WS* > *5 OBs without WS* > *3 OBs with WS*. Deep architectures have better representation capabilities and also lead to better performance. However, training non-weight-sharing B2Opt with more layers is challenging due to the difficulty of training deep architectures. *Untrained* represents that the parameters of *5 OBs without WS* are randomly initialized. *5 OBs without WS* outperforms *Untrained*, which demonstrates the effectiveness of the designed training process. We also find an interesting phenomenon: *5 OBs without WS* outperforms *3 OBs with WS* in all cases. Our untrained deep architecture, *30 OBs with*

*WS*, can achieve good results on simple planner mechanic arm problems, which shows that B2Opt retains the advantages of Transformer architecture and has strong generalization ability. We use the untrained *5 OBs without WS* to test the complex planner mechanic arm problem, which performs poorly.

We have observed that B2Opt can achieve better results with deeper architectures. However, it is currently difficult for us to train deep B2Opt. Moreover, as far as we know, the use of ES to optimize deep models has been studied a lot [26], which will be an essential research prospect in the future.

2) *Learning Rate*: We train B2Opt on the F1-F3 function set with different learning rates and then test it on the F4-F9 function set. The experimental results are shown in Table VII. *5 OBs without WS* and *30 OBs with WS* perform poorly when the learning rate is 0.1, which may be because the learning rate is too large, which affects the convergenceTABLE VII  
THE EFFECT OF LEARNING RATE ON B2OPT.

<table border="1">
<thead>
<tr>
<th><math>lr</math></th>
<th>F4</th>
<th>F5</th>
<th>F6</th>
<th>F7</th>
<th>F8</th>
<th>F9</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="7" style="text-align: center;"><i>5 OBs without WS</i></td>
</tr>
<tr>
<td>0.1</td>
<td>0.93(7.42)</td>
<td>0.31(0.49)</td>
<td>2.04e7(2.03e8)</td>
<td>15.3(6.4)</td>
<td>0.36(0.16)</td>
<td>0.61(0.18)</td>
</tr>
<tr>
<td>0.01</td>
<td><b>0.01(0.003)</b></td>
<td><b>0.05(0.02)</b></td>
<td><b>9.57(0.22)</b></td>
<td><b>1.62(0.60)</b></td>
<td><b>0.03(0.01)</b></td>
<td><b>0.06(0.03)</b></td>
</tr>
<tr>
<td>0.001</td>
<td>0.88(3.41)</td>
<td>0.36(0.12)</td>
<td>226(1750)</td>
<td>6.18(2.66)</td>
<td>0.56(0.16)</td>
<td>1.36(0.36)</td>
</tr>
<tr>
<td>0.0001</td>
<td>0.06(0.03)</td>
<td>0.13(0.03)</td>
<td>13.6(2.11)</td>
<td>0.83(0.50)</td>
<td>0.17(0.04)</td>
<td>0.28(0.10)</td>
</tr>
<tr>
<td colspan="7" style="text-align: center;"><i>30 OBs with WS</i></td>
</tr>
<tr>
<td>0.1</td>
<td>1.64(1.19)</td>
<td>0.85(1.59)</td>
<td>493(3110)</td>
<td>28.4(4.35)</td>
<td>0.47(0.11)</td>
<td>2.82(0.5)</td>
</tr>
<tr>
<td>0.01</td>
<td>0.05(0.30)</td>
<td>0.09(0.03)</td>
<td>39.2(240)</td>
<td>1.05(1.40)</td>
<td>0.01(0.06)</td>
<td>0.28(0.07)</td>
</tr>
<tr>
<td>0.001</td>
<td><b>1.01e-3(0.001)</b></td>
<td>0.02(0.01)</td>
<td>9.03(0.18)</td>
<td>0.03(0.02)</td>
<td><b>0.003(0.001)</b></td>
<td>0.03(0.01)</td>
</tr>
<tr>
<td>0.0001</td>
<td>1.50e-3(0.001)</td>
<td><b>0.016(0.004)</b></td>
<td><b>9.01(0.13)</b></td>
<td><b>0.02(0.02)</b></td>
<td>0.006(0.002)</td>
<td><b>0.02(0.001)</b></td>
</tr>
<tr>
<td colspan="7" style="text-align: center;"><i>3 OBs with WS</i></td>
</tr>
<tr>
<td>0.1</td>
<td>3.04(5.55)</td>
<td>0.98(0.4)</td>
<td>1150(7930)</td>
<td>43.3(8.87)</td>
<td><b>0.64(0.12)</b></td>
<td>2.43(0.98)</td>
</tr>
<tr>
<td>0.01</td>
<td>29.9(47.7)</td>
<td>2.68(1.30)</td>
<td>6.24e4(4.87e5)</td>
<td>40.4(8.52)</td>
<td>1.05(0.06)</td>
<td>4.45(0.85)</td>
</tr>
<tr>
<td>0.001</td>
<td>1.82(1.20)</td>
<td>0.76(0.32)</td>
<td>654(4780)</td>
<td>7.00(7.11)</td>
<td>0.79(0.13)</td>
<td>1.91(0.53)</td>
</tr>
<tr>
<td>0.0001</td>
<td><b>0.39(0.21)</b></td>
<td><b>0.33(0.07)</b></td>
<td><b>46.8(79.4)</b></td>
<td><b>2.22(2.41)</b></td>
<td>0.66(0.09)</td>
<td><b>0.59(0.19)</b></td>
</tr>
</tbody>
</table>

of B2Opt during the training process. For *5 OBs without WS*, setting the learning rate to 0.01 achieves the relatively best performance. A learning rate 0.0001 would be a good choice for *30 OBs with WS* and *3 OBs with WS*. However, our experiments are coarse-grained. The learning rate has a significant impact on B2Opt. Then using AutoML to search for the optimal hyperparameter combination of the model is expected to achieve better performance.

3) *Training Data Size*: We also explore the impact of the size of the training data set on the performance of the algorithm and take F4 as an example. It is trained on various biases of F4 and tested on F4 without bias. The training data set is  $F^{train} = \{F4(x|\omega_{1,i}^{train}), \dots, F4(x|\omega_{m,i}^{train})\}$ . Experimental results are shown in Fig. 7, which show that the size of the training dataset significantly impacts the performance of B2Opt. As the amount of training data increases, the performance of B2Opt increases.

Fig. 7. The impact of training dataset size on B2Opt.

#### D. Ablation Study

This section considers the impact of different parts on B2Opt. We take B2Opt with 3 OBs and weight sharing as

TABLE VIII  
THE RESULTS OF ABLATION STUDY.  $d = 10$ .

<table border="1">
<thead>
<tr>
<th><math>f</math></th>
<th><i>Not SAC</i></th>
<th><i>Not FM</i></th>
<th><i>Not RC</i></th>
<th><i>Not RSSM</i></th>
<th>B2Opt</th>
</tr>
</thead>
<tbody>
<tr>
<td>F4</td>
<td>52.7(15.0)</td>
<td>23.3(6.68)</td>
<td>50.5(6.81)</td>
<td>271(346)</td>
<td>3.03(5.82)</td>
</tr>
<tr>
<td>F5</td>
<td>5.08(0.98)</td>
<td>2.94(0.41)</td>
<td>3.75(0.09)</td>
<td>3.51(1.49)</td>
<td>1.02(0.20)</td>
</tr>
<tr>
<td>F6</td>
<td>1e5(9e4)</td>
<td>1e4(1e4)</td>
<td>5e4(1e4)</td>
<td>7e6(1e8)</td>
<td>4e3(7e4)</td>
</tr>
<tr>
<td>F7</td>
<td>47.2(5.11)</td>
<td>19.7(4.27)</td>
<td>63.4(8.60)</td>
<td>40.6(8.10)</td>
<td>44.8(8.34)</td>
</tr>
<tr>
<td>F8</td>
<td>1.18(0.04)</td>
<td>1.19(0.07)</td>
<td>0.87(0.07)</td>
<td>3.28(1.14)</td>
<td>0.60(0.17)</td>
</tr>
<tr>
<td>F9</td>
<td>4.49(1.09)</td>
<td>3.42(0.78)</td>
<td>8.36(0.43)</td>
<td>3.56(0.72)</td>
<td>3.03(0.70)</td>
</tr>
</tbody>
</table>

an example, which is trained on F1-F3 and tested on F4-F9. We remove SAC, FM, RSSM, and RC in B2Opt, respectively, and denote them as *Not SAC*, *Not FM*, *Not RSSM*, and *Not RC*. The experimental results are shown in Table VIII. When their results are sorted from good to worst, the rank is  $B2Opt > Not FM > Not RC \approx Not SAC \approx Not RSSM$ . The role of FM is slightly weaker than that of the other three modules. Taken as a whole, the parts of SAC, RSSM, and RC are of equal importance. The absence of these core components can seriously affect the performance of B2Opt. At the same time, it also shows the effectiveness of the proposed four modules. The removal of any one of the modules in the crossover, mutation, and selection of EAs will degrade the performance of EAs. This shows that B2Opt implements a learnable EA framework that does not require human-designed parameters.

#### E. Visualization Analysis

The tested model is *5 OBs with WS* trained on F1-F3 with  $d = 100$ . The population size is set to 100.

**Visual Analysis of SAC** The crossover strategies learned by the five SAC are shown in Fig. 8. For the presentation, we select individuals with 1st, 50th, and 100th fitness rankings. The horizontal axis represents the fitness ranking of individuals, and the vertical axis represents the attention (weight when performing crossover) on these individuals. OB1 tends to crossover with lower-ranked individuals, showing a preferenceFig. 8. Crossover Strategy learned by B2Opt.

Fig. 9. Mutation strategy learned by B2Opt for (a) F4, (b) F5, (c) F6, (d) F7, (e) F8, and (f) F9.

for exploration. From OB1 to OB5, the bias of SAC gradually changes from exploration to exploitation.

**Visual Analysis of FM** We visualize the FM of B2Opt to explore its behavior as Fig. 9. As a reference, we use polynomial mutation in genetic algorithms. Given the input population (input), the mutated population (OB1) is obtained through OB1; the new population (mutpolyn) is obtained by performing polynomial mutation on the input population. We visualize F4-F9 and observe the following phenomena:

1) The population generated by performing polynomial mutation is more evenly distributed on the landscape. However, most of the solutions produced by FM in B2Opt are concentrated in "areas with greater potential", which are closer to the optimal solution. Moreover, the population distribution generated by our scheme also takes diversity into account. The non-optimal solution area is also more comprehensive than that of polynomial mutation, which is more conducive to jumping out of the local solution.2) The population produced by performing polynomial mutation moves slightly compared to the original population. However, FM can guide the input population to make big moves toward the optimal solution, significantly accelerating the algorithm's convergence.

This shows that B2Opt can use the information of the objective function to design the mutation strategy automatically, making it more applicable to the target optimization task, which is consistent with our motivation.

## VI. CONCLUSIONS

The better performance than that of EA baselines, Bayesian optimization, and the L2O-based method demonstrates the effectiveness of B2Opt. Moreover, B2Opt can be well adapted to unseen BBO. Meanwhile, we experimentally demonstrate that the proposed three modules have positive effects. However, B2Opt still has room for improvement.

1) In the loss function, we do not effectively consider the diversity of the population, and the population can be regularized in the future;

2) The training set seriously affects the performance of B2Opt. If the similarity between the training set and the optimization objective is low, it will cause the performance of B2Opt to degrade drastically. Building the dataset as relevant to the target as possible is essential.

## REFERENCES

1. [1] T. Elsken, J. H. Metzen, and F. Hutter, "Neural architecture search: A survey," *The Journal of Machine Learning Research*, vol. 20, no. 1, pp. 1997–2017, 2019.
2. [2] F. Hutter, L. Kotthoff, and J. Vanschoren, *Automated machine learning: methods, systems, challenges*. Springer Nature, 2019.
3. [3] D. Golovin, B. Solnik, S. Moitra, G. Kochanski, J. Karro, and D. Sculley, "Google vizier: A service for black-box optimization," in *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2017, pp. 1487–1495.
4. [4] J.-Y. Li, Z.-H. Zhan, and J. Zhang, "Evolutionary computation for expensive optimization: A survey," *Machine Intelligence Research*, vol. 19, no. 1, pp. 3–23, 2022.
5. [5] M. Mitchell, *An introduction to genetic algorithms*. MIT press, 1998.
6. [6] S. Khadka and K. Tumer, "Evolution-guided policy gradient in reinforcement learning," in *Advances in Neural Information Processing Systems*, S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, Eds., vol. 31. Curran Associates, Inc., 2018.
7. [7] Q. Zhang and H. Li, "Moea/d: A multiobjective evolutionary algorithm based on decomposition," *IEEE Transactions on Evolutionary Computation*, vol. 11, no. 6, pp. 712–731, 2007.
8. [8] J. Snoek, H. Larochelle, and R. P. Adams, "Practical bayesian optimization of machine learning algorithms," in *Advances in Neural Information Processing Systems*, F. Pereira, C. Burges, L. Bottou, and K. Weinberger, Eds., vol. 25. Curran Associates, Inc., 2012.
9. [9] M. Mutny and A. Krause, "Efficient high dimensional bayesian optimization with additivity and quadrature fourier features," in *Advances in Neural Information Processing Systems*, S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, Eds., vol. 31. Curran Associates, Inc., 2018.
10. [10] C. Li, S. Gupta, S. Rana, V. Nguyen, S. Venkatesh, and A. Shilton, "High dimensional bayesian optimization using dropout," in *IJCAI'17*. AAAI Press, 2017.
11. [11] K. Kandasamy, J. Schneider, and B. Póczos, "High dimensional bayesian optimisation and bandits via additive models," in *International Conference on Machine Learning*. PMLR, 2015, pp. 295–304.
12. [12] M. Balandat, B. Karrer, D. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy, "Botorch: A framework for efficient monte-carlo bayesian optimization," in *Advances in Neural Information Processing Systems*, H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, Eds., vol. 33. Curran Associates, Inc., 2020, pp. 21 524–21 538.
13. [13] D. Wierstra, T. Schaul, T. Glasmachers, Y. Sun, J. Peters, and J. Schmidhuber, "Natural evolution strategies," *Journal of Machine Learning Research*, vol. 15, no. 1, pp. 949–980, 2014.
14. [14] N. Hansen and A. Ostermeier, "Completely derandomized self-adaptation in evolution strategies," *Evolutionary Computation*, vol. 9, no. 2, pp. 159–195, 2001.
15. [15] A. Auger and N. Hansen, "A restart cma evolution strategy with increasing population size," in *2005 IEEE Congress on Evolutionary Computation*, vol. 2, 2005, pp. 1769–1776.
16. [16] T. Salimans, J. Ho, X. Chen, S. Sidor, and I. Sutskever, "Evolution strategies as a scalable alternative to reinforcement learning," *arXiv preprint arXiv:1703.03864*, 2017.
17. [17] M. Andrychowicz, M. Denil, S. Gomez, M. W. Hoffman, D. Pfau, T. Schaul, B. Shillingford, and N. De Freitas, "Learning to learn by gradient descent by gradient descent," in *Advances in Neural Information Processing Systems*, 2016, pp. 3981–3989.
18. [18] T. Chen, X. Chen, W. Chen, H. Heaton, J. Liu, Z. Wang, and W. Yin, "Learning to optimize: A primer and a benchmark," *Journal of Machine Learning Research*, vol. 23, pp. 1–59, 2022.
19. [19] T. Chen, W. Zhang, Z. Jingyang, S. Chang, S. Liu, L. Amini, and Z. Wang, "Training stronger baselines for learning to optimize," in *Advances in Neural Information Processing Systems*, H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, Eds., vol. 33. Curran Associates, Inc., 2020, pp. 7332–7343.
20. [20] Y. Chen, M. W. Hoffman, S. G. Colmenarejo, M. Denil, T. P. Lillicrap, M. Botvinick, and N. Freitas, "Learning to learn without gradient descent by gradient descent," in *International Conference on Machine Learning*. PMLR, 2017, pp. 748–756.
21. [21] K. Li and J. Malik, "Learning to optimize," *arXiv preprint arXiv:1606.01885*, 2016.
22. [22] O. Wichrowska, N. Maheswaranathan, M. W. Hoffman, S. G. Colmenarejo, M. Denil, N. Freitas, and J. Sohl-Dickstein, "Learned optimizers that scale and generalize," in *International Conference on Machine Learning*. PMLR, 2017, pp. 3751–3760.
23. [23] I. Bello, B. Zoph, V. Vasudevan, and Q. V. Le, "Neural optimizer search with reinforcement learning," in *International Conference on Machine Learning*. PMLR, 2017, pp. 459–468.
24. [24] L. Metz, N. Maheswaranathan, J. Nixon, D. Freeman, and J. Sohl-Dickstein, "Understanding and correcting pathologies in the training of learned optimizers," in *International Conference on Machine Learning*. PMLR, 2019, pp. 4556–4565.
25. [25] E. Gärtner, L. Metz, M. Andriluka, C. D. Freeman, and C. Sminchisescu, "Transformer-based learned optimization," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, June 2023, pp. 11 970–11 979.
26. [26] P. Vicol, L. Metz, and J. Sohl-Dickstein, "Unbiased gradient estimation in unrolled computation graphs with persistent evolution strategies," in *International Conference on Machine Learning*. PMLR, 2021, pp. 10 553–10 563.
27. [27] S. Flennerhag, Y. Schroecker, T. Zahavy, H. van Hasselt, D. Silver, and S. Singh, "Bootstrapped meta-learning," in *International Conference on Learning Representations*, 2022.
28. [28] H. Sun, X. Chen, Q. Shi, M. Hong, X. Fu, and N. D. Sidiropoulos, "Learning to optimize: Training deep neural networks for interference management," *IEEE Transactions on Signal Processing*, vol. 66, no. 20, pp. 5438–5453, 2018.
29. [29] Y. Cao, T. Chen, Z. Wang, and Y. Shen, "Learning to optimize in swarms," *Advances in Neural Information Processing Systems*, vol. 32, pp. 15 044–15 054, 2019.
30. [30] R. T. Lange, T. Schaul, Y. Chen, T. Zahavy, V. Dallibard, C. Lu, S. Singh, and S. Flennerhag, "Discovering evolution strategies via meta-black-box optimization," *arXiv preprint arXiv:2211.11260*, 2022.
31. [31] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," *arXiv preprint arXiv:1412.6980*, 2014.
32. [32] C. Wang, J. Liu, K. Wu, and Z. Wu, "Solving multitask optimization problems with adaptive knowledge transfer via anomaly detection," *IEEE Transactions on Evolutionary Computation*, vol. 26, no. 2, pp. 304–318, 2022.
33. [33] V. TV, P. Malhotra, J. Narwariya, L. Vig, and G. Shroff, "Meta-learning for black-box optimization," in *Joint European Conference on Machine Learning and Knowledge Discovery in Databases*. Springer, 2020, pp. 366–381.
34. [34] H. S. Gomes, B. Léger, and C. Gagné, "Meta learning black-box population-based optimizers," *arXiv preprint arXiv:2103.03526*, 2021.
35. [35] G. Shala, A. Biedenkapp, N. Awad, S. Adriaensen, M. Lindauer, and F. Hutter, "Learning step-size adaptation in cma-es," in *International**Conference on Parallel Problem Solving from Nature*. Springer, 2020, pp. 691–706.

- [36] R. Lange, T. Schaul, Y. Chen, C. Lu, T. Zahavy, V. Dalibard, and S. Flennerhag, “Discovering attention-based genetic algorithms via meta-black-box optimization,” in *Proceedings of the Genetic and Evolutionary Computation Conference*, ser. GECCO ’23. New York, NY, USA: Association for Computing Machinery, 2023, p. 929–937.
- [37] J. Zhang, C. Xu, J. Li, W. Chen, Y. Wang, Y. Tai, S. Chen, C. Wang, F. Huang, and Y. Liu, “Analogous to evolutionary algorithm: Designing a unified sequence model,” in *Advances in Neural Information Processing Systems*, M. Ranzato, A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan, Eds., vol. 34. Curran Associates, Inc., 2021, pp. 26 674–26 688.
- [38] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly *et al.*, “An image is worth 16x16 words: Transformers for image recognition at scale,” in *International Conference on Learning Representations*, 2021.
- [39] K. Kandasamy, G. Dasarathy, J. B. Oliva, J. Schneider, and B. Póczos, “Gaussian process bandit optimisation with multi-fidelity evaluations,” *Advances in Neural Information Processing Systems*, vol. 29, 2016.
- [40] Y. Cao and Y. Shen, “Bayesian active learning for optimization and uncertainty quantification in protein docking,” *Journal of Chemical Theory and Computation*, vol. 16, no. 8, pp. 5334–5347, 2020.
- [41] R. J. Williams, “Simple statistical gradient-following algorithms for connectionist reinforcement learning,” *Machine Learning*, vol. 8, no. 3, pp. 229–256, 1992.
- [42] A. Cully, J. Clune, D. Tarapore, and J.-B. Mouret, “Robots that can adapt like animals,” *Nature*, vol. 521, pp. 503–507, 2015.
- [43] V. Vassiliades, K. Chatzilygeroudis, and J.-B. Mouret, “Using centroidal voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm,” *IEEE Transactions on Evolutionary Computation*, vol. 22, pp. 623–630, 2018.
- [44] V. Vassiliades and J.-B. Mouret, “Discovering the elite hypervolume by leveraging interspecies correlation,” *Proceedings of the Genetic and Evolutionary Computation Conference*, 2018.
- [45] J.-B. Mouret and G. Maguire, “Quality diversity for multi-task optimization,” *Proceedings of the 2020 Genetic and Evolutionary Computation Conference*, 2020.
- [46] A. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam, “Mobilenets: Efficient convolutional neural networks for mobile vision applications,” *arXiv: Computer Vision and Pattern Recognition*, 2017.
- [47] S. Das and P. N. Suganthan, “Differential evolution: A survey of the state-of-the-art,” *IEEE transactions on Evolutionary Computation*, vol. 15, no. 1, pp. 4–31, 2010.
- [48] A. Auger and N. Hansen, “A restart cma evolution strategy with increasing population size,” in *2005 IEEE Congress on Evolutionary Computation*, vol. 2, 2005, pp. 1769–1776 Vol. 2.
- [49] R. Tanabe and A. S. Fukunaga, “Improving the search performance of shade using linear population size reduction,” in *2014 IEEE Congress on Evolutionary Computation (CEC)*, 2014, pp. 1658–1665.
- [50] N. Hansen, “The cma evolution strategy: A tutorial,” *ArXiv*, vol. abs/1604.00772, 2016.
- [51] Jazbin, “Geatpy: The genetic and evolutionary algorithm toolbox with high performance in python,” 2020.
- [52] K. Kandasamy, K. R. Vysyaraju, W. Neiswanger, B. Paria, C. R. Collins, J. Schneider, B. Póczos, and E. P. Xing, “Tuning hyperparameters without grad students: Scalable and robust bayesian optimisation with dragonfly,” *Journal of Machine Learning Research*, vol. 21, no. 81, pp. 1–27, 2020.
- [53] D. Eriksson and M. Jankowiak, “High-dimensional Bayesian optimization with sparse axis-aligned subspaces,” in *Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence*, ser. Proceedings of Machine Learning Research, C. de Campos and M. H. Maathuis, Eds., vol. 161. PMLR, 27–30 Jul 2021, pp. 493–503.

