Title: Improving the matrix multiplication exponent with modern optimization and AlphaEvolve

URL Source: https://arxiv.org/html/2608.16884

Markdown Content:
\uselogo

Marvin Eisenberger Affiliation: Equal contribution in alphabetical order Affiliation: \thepa Borislav Kozlovskii Affiliation: Equal contribution in alphabetical order Affiliation: \thepa Abbas Mehrabian Affiliation: Equal contribution in alphabetical order Affiliation: \thepa Francisco J. R. Ruiz Affiliation: Equal contribution in alphabetical order Affiliation: \thepa Abigail See Affiliation: Equal contribution in alphabetical order Affiliation: \thepa Renfei Zhou Affiliation: Equal contribution in alphabetical order Affiliation: Carnegie Mellon University Josh Alman Affiliation: Columbia University Virginia Vassilevska Williams Affiliation: MIT Matej Balog Affiliation: \thepa

###### Abstract

The current best bounds on the matrix multiplication exponent \omega are obtained through a refinement of the laser method called combination loss analysis ([10](https://arxiv.org/html/2608.16884#bib.bib15); [23](https://arxiv.org/html/2608.16884#bib.bib16); [1](https://arxiv.org/html/2608.16884#bib.bib17)). In this note, we address the optimization problem at the core of this approach and propose several improvements. First, we reformulate the optimization problem allowing us to solve it in a larger setting than was previously possible. Second, we leverage recent advances in machine learning to design a new optimization algorithm for this problem. Finally, we refine the resulting optimization algorithm with AlphaEvolve. Our combined approach yields an upper bound of \omega<2.371177, improving the previous best bound of 2.371339.

## 1 Introduction

From accelerating machine learning computations to enabling realistic computer graphics, matrix multiplication is a fundamental operation underpinning critical applications in computer science. Despite its prominence, the computational complexity of matrix multiplication—the number of arithmetic operations needed to multiply large matrices—is unknown, and determining it is a major open question in theoretical computer science ([4](https://arxiv.org/html/2608.16884#bib.bib24)).

The pioneering work of Strassen ([22](https://arxiv.org/html/2608.16884#bib.bib1)) showed that two n\times n matrices can be multiplied in sub-cubic time—specifically, \mathcal{O}(n^{\omega+o(1)}) operations for \omega<2.81—spurring a line of work attempting to further reduce the complexity exponent \omega([16](https://arxiv.org/html/2608.16884#bib.bib4); [3](https://arxiv.org/html/2608.16884#bib.bib5); [18](https://arxiv.org/html/2608.16884#bib.bib6); [17](https://arxiv.org/html/2608.16884#bib.bib7); [6](https://arxiv.org/html/2608.16884#bib.bib8); [21](https://arxiv.org/html/2608.16884#bib.bib9); [7](https://arxiv.org/html/2608.16884#bib.bib10); [20](https://arxiv.org/html/2608.16884#bib.bib11); [24](https://arxiv.org/html/2608.16884#bib.bib12); [14](https://arxiv.org/html/2608.16884#bib.bib13); [2](https://arxiv.org/html/2608.16884#bib.bib14); [10](https://arxiv.org/html/2608.16884#bib.bib15); [23](https://arxiv.org/html/2608.16884#bib.bib16); [1](https://arxiv.org/html/2608.16884#bib.bib17)). All improvements in the past 40 years rely on the _laser method_, a mathematical technique to indirectly design matrix multiplication algorithms. The current best bound is achieved by a refinement of the laser method called _combination loss analysis_([10](https://arxiv.org/html/2608.16884#bib.bib15)), a technique that requires solving a non-convex optimization problem as part of the computer-assisted proof, and yields \omega<2.371339([1](https://arxiv.org/html/2608.16884#bib.bib17)).

Here, we improve the bound to \omega<2.371\mathbf{177} using a two-step approach. First, we leverage recent advances in machine learning and adjacent areas to address the non-convex optimization problem of combination loss analysis using a gradient descent approach; this alone improves the previous state-of-the-art (SOTA) bound by \approx 0.97\times 10^{-4}. Second, we use AlphaEvolve ([15](https://arxiv.org/html/2608.16884#bib.bib23)) to improve our optimization algorithm; this raises the improvement over the SOTA to \approx 1.62\times 10^{-4}.

Table 1: Recent improvements to \omega.

The optimization problem at the core of combination loss analysis is formulated in [1](https://arxiv.org/html/2608.16884#bib.bib17), where it was also shown that any feasible solution provides an upper bound on \omega. To achieve our new bound, we target a slightly different optimization problem. Specifically, combination loss analysis has a parameter—the _maximum recursion level_, denoted by \ell^{*}—that introduces a trade-off between the complexity of the optimization (which grows doubly exponentially in \ell^{*}) and the best possible bound on \omega it can achieve. The previous SOTA bound was found with \ell^{*}=3([1](https://arxiv.org/html/2608.16884#bib.bib17)). In contrast, our gradient-based optimization, implemented in Jax ([5](https://arxiv.org/html/2608.16884#bib.bib3)), allows for hardware parallelization and can handle \ell^{*}=4 (the number of optimizable parameters increases from approximately 25k to 7 million when moving from \ell^{*}=3 to \ell^{*}=4).

In this note, we describe the full optimization problem from [1](https://arxiv.org/html/2608.16884#bib.bib17) in detail, explain how we numerically solved it and applied AlphaEvolve to it, and finally show how we rigorously certified the resulting omega bound.

## 2 Optimization problem

The high-level structure of the optimization problem can be captured by a rooted tree, where every node is associated with a collection of optimizable parameters. Integers q\geq 1 and \ell^{*}\geq 2 are fixed as hyperparameters. Intuitively, the tree structure describes a recursive way to decompose the tensor CW_{q}^{\otimes 2^{\ell^{*}}} into smaller tensors, where CW_{q} is the Coppersmith-Winograd tensor ([7](https://arxiv.org/html/2608.16884#bib.bib10)); see [1](https://arxiv.org/html/2608.16884#bib.bib17) for more details about this correspondence. We start by defining the tree structure, and will then define the optimizable parameters associated to its nodes.

### 2.1 Tree structure

Denote [k]\coloneqq\{1,\dots,k\}. Each non-root node T in the tree is associated with a _level_, a _shape_ s_{T}, and a _region_ r_{T}, defined as follows.

*   •
The _level_ of a non-root node is a positive integer \ell\in\{2,3,\dots,\ell^{*}\}, describing its depth in the tree; higher is closer to the root. Direct children of the root node have level \ell^{*}, which is a hyperparameter fixed in advance.

*   •A level-\ell _shape_ is a triple s=(s_{X},s_{Y},s_{Z}), where s_{X},s_{Y},s_{Z} are non-negative integers summing to 2^{\ell}. We use

\mathcal{S}_{\ell}\coloneqq\left\{(i,j,k)\in\mathbb{Z}_{\geq 0}^{3}\;\middle|\;i+j+k=2^{\ell}\right\}

to denote the set of all level-\ell shapes. We use _dimensions_ X,Y,Z to refer to the indices of the three coordinates in a shape. 
*   •
There are six _regions_, indexed by an integer r\in[6]. Each region is associated with \pi_{r}, the r-th permutation over symbols \{X,Y,Z\} in lexicographic order.

We call a non-root node T a _positive-shape node_ if _all_ coordinates in its shape s_{T} are positive. Otherwise it is a _zero-shape node_. The root node is denoted by G and does not have the aforementioned attributes—level, shape, and region.

Next, we define the tree structure by describing the child nodes for different types of nodes.

*   •
Root: For every region r\in[6] and level-\ell^{*} shape s\in\mathcal{S}_{\ell^{*}}, the root G has a child at level \ell^{*} with shape s and region r, denoted as G[s,r].

*   •Positive-shape node: For every level-\ell shape s=(s_{X},s_{Y},s_{Z}), we define

\Split(s)\coloneqq\left\{u\in\mathcal{S}_{\ell-1}\;\middle|\;0\leq u_{X}\leq s_{X},\;0\leq u_{Y}\leq s_{Y},\;0\leq u_{Z}\leq s_{Z}\right\}.

Fixing a positive-shape node T at level \ell\geq 3, for every region r\in[6] and level-(\ell-1) shape u\in\Split(s_{T}), the node T has a child at level \ell-1 with shape u and region r, denoted as T[u,r]. Note that the region index r of the child can be different from that of T itself. 
*   •
Zero-shape nodes and level-2 positive-shape nodes do not have child nodes. We call them the _leaves_.

### 2.2 Optimizable parameters

Next we list the free variables of the optimization problem, grouped by the associated node on the tree. For any finite set D, we use \Delta(D) to denote the simplex of probability distributions on D:

\Delta(D)\coloneqq\left\{p:D\rightarrow[0,1]\;\middle|\;\sum_{x\in D}p(x)=1\right\}

#### Root node.

The optimizable parameters associated with the root node G are:

*   •
A distribution A_{G}=(A_{G}^{(1)},\dots,A_{G}^{(6)})\in\Delta([6]) over the six regions;

*   •
For r\in[6], a distribution \alpha_{G}^{(r)}\in\Delta(\mathcal{S}_{\ell^{*}}) over all level-\ell^{*} shapes.

#### Positive-shape node.

Each level-\ell positive-shape node T with \ell\geq 3 is associated with the following parameters:

*   •
A distribution A_{T}=(A_{T}^{(1)},\ldots,A_{T}^{(6)})\in\Delta([6]) over the six regions;

*   •
For r\in[6], a distribution \alpha_{T}^{(r)}\in\Delta(\Split(s_{T})), where s_{T} is the shape of T.

#### Zero-shape node.

For a level \ell and an integer a with 0\leq a\leq 2^{\ell}, we define

\mathcal{C}_{\ell,a}\coloneqq\left\{L\in\{0,1,2\}^{2^{\ell-1}}\;\middle|\;\sum_{p=1}^{2^{\ell-1}}L_{p}=a\right\}.

This is the set of length-2^{\ell-1} vectors over \{0,1,2\} whose entries sum to a. A distribution over \mathcal{C}_{\ell,a} is called a level-\ell _complete split distribution_ of a.

For every level-\ell zero-shape node T, let W\in\{X,Y,Z\} be the first dimension where s_{T,W} is nonzero. T is then associated with a complete split distribution \beta_{T,W}\in\Delta(\mathcal{C}_{\ell,s_{T,W}}).

#### Level-2 node.

The only remaining nodes are level-2 nodes with shapes (1,1,2), (1,2,1), or (2,1,1), which are the only valid strictly positive shapes for level 2, since their coordinates must sum to 2^{2}=4. Each such node is associated with a scalar \mu_{T}\in[0,1/2].

### 2.3 Derived quantities

With the free variables fixed, all remaining quantities can be computed deterministically. In the following, all logarithms and entropies are in base two. For a distribution \rho with finite support, we denote its entropy by

H(\rho)\coloneqq-\sum_{x\in\supp(\rho)}\rho(x)\log\rho(x).

We also use H(p_{1},p_{2},\dots,p_{k}) to denote the entropy of a distribution over [k] with probability masses p_{1},\dots,p_{k}. If D is a set of shapes and \rho\in\Delta(D), let \rho_{W} denote the marginal of \rho in coordinate W, namely, \rho_{W}(w)\coloneqq\displaystyle\sum_{a\in D:a_{W}=w}\rho(a). Define

H_{D}^{\max}(\rho)\coloneqq\sup_{\begin{subarray}{c}\rho^{\prime}\in\Delta(D)\\
\rho^{\prime}_{W}=\rho_{W}\text{ for }W\in\{X,Y,Z\}\end{subarray}}H(\rho^{\prime}),

as well as the penalty notion

P_{D}\left(\rho\right)\coloneqq H_{D}^{\max}\left(\rho\right)-H\left(\rho\right).

#### Masses.

Each non-zero-shape node T of the tree has a real number m_{T}\in[0,1] associated with it, called its _mass_. The masses of non-root nodes are computed top-down as follows:

*   •
The root G has m_{G}=1.

*   •For the children of the root, we set

m_{G[s,r]}\coloneqq A_{G}^{(r)}\cdot\alpha_{G}^{(r)}(s),\qquad\forall\,s\in\mathcal{S}_{\ell^{*}},\ r\in[6]. 
*   •For any positive-shape node T of level \ell\geq 3, we set

m_{T[u,r]}=m_{T}\cdot A_{T}^{(r)}\cdot\left(\alpha_{T}^{(r)}(u)+\alpha_{T}^{(r)}(s_{T}\!-\!u)\right),\qquad\forall\,u\in\Split(s_{T}),\ r\in[6]. 

#### Complete split distributions.

Every level-\ell non-root node T carries, for each dimension W\in\{X,Y,Z\}, a complete split distribution \beta_{T,W}\in\Delta\big(\mathcal{C}_{\ell,s_{T,W}}\big). These distributions are calculated as follows.

Let \vec{0}\coloneqq(0,0,\dots,0) and \vec{2}\coloneqq(2,2,\dots,2) denote vectors of length 2^{\ell-1}. For a zero-shape node T, let W_{0}\in\{X,Y,Z\} be the first zero coordinate of s_{T}, W_{1} be the first nonzero coordinate of s_{T}, and W_{2} be the other coordinate. \beta_{T,W_{0}} is the point mass distribution at the length-2^{\ell-1} vector \vec{0}; \beta_{T,W_{1}} was defined as optimizable parameters in [section 2.2](https://arxiv.org/html/2608.16884#S2.SS2 "2.2 Optimizable parameters ‣ 2 Optimization problem ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"); \beta_{T,W_{2}}\coloneqq\beta^{\vee}_{T,W_{1}}, where for any complete split distribution \beta, we define

\beta^{\vee}(\vec{2}\!-\!L)\coloneqq\beta(L),\qquad\forall\,L\in{\supp(\beta)}.

(Notice, in particular, that if the components of L sum to s_{T,W_{1}}, then the components of \vec{2}-L sum to 2\cdot 2^{\ell-1}-s_{T,W_{1}}=s_{T,W_{2}}, keeping the mapping validly within the correct domain.)

For a positive-shape node T at level \ell\geq 3, we define

\beta_{T,W}^{(r)}\coloneqq\sum_{u\in\Split(s_{T})}\alpha_{T}^{(r)}(u)\cdot\bigl(\beta_{T[u,r],W}\times\beta_{T[s_{T}-u,r],W}\bigr),\qquad\forall\,r\in[6],

where \times denotes the Cartesian product of complete split distributions; that is, if L_{\text{left}} and L_{\text{right}} are two sequences of length 2^{\ell-2} in supports of \beta_{\text{left}}\coloneqq\beta_{T[u,r],W} and \beta_{\text{right}}\coloneqq\beta_{T[s_{T}-u,r],W}, respectively, then we form a new sequence L, of length 2^{\ell-1}, by concatenating L_{\text{left}} and L_{\text{right}}, and letting

(\beta_{\text{left}}\times\beta_{\text{right}})(L)\coloneqq\beta_{\text{left}}(L_{\text{left}})\cdot\beta_{\text{right}}(L_{\text{right}}).

Then, the complete split distributions of T are calculated by

\beta_{T,W}\coloneqq\sum_{r=1}^{6}A_{T}^{(r)}\beta_{T,W}^{(r)}.

The only remaining case is a level-2 positive-shape node, which must have shape (1,1,2), (2,1,1), or (1,2,1). For a node T with shape (1,1,2), recall that \mu_{T}\in[0,1/2] is the only optimizable parameter associated with T. We let \delta_{a,b} denote the point mass at (a,b)\in\{0,1,2\}^{2} and set

\beta_{T,X}=\beta_{T,Y}=\frac{1}{2}\delta_{0,1}+\frac{1}{2}\delta_{1,0},\qquad\beta_{T,Z}=\mu_{T}\delta_{0,2}+\mu_{T}\delta_{2,0}+(1-2\mu_{T})\delta_{1,1}.

Similarly, for a node T with shape (2,1,1), we set

\beta_{T,Y}=\beta_{T,Z}=\frac{1}{2}\delta_{0,1}+\frac{1}{2}\delta_{1,0},\qquad\beta_{T,X}=\mu_{T}\delta_{0,2}+\mu_{T}\delta_{2,0}+(1-2\mu_{T})\delta_{1,1},

and for a node T with shape (1,2,1), we set

\beta_{T,X}=\beta_{T,Z}=\frac{1}{2}\delta_{0,1}+\frac{1}{2}\delta_{1,0},\qquad\beta_{T,Y}=\mu_{T}\delta_{0,2}+\mu_{T}\delta_{2,0}+(1-2\mu_{T})\delta_{1,1}.

#### Retained exponent at root node.

Next, we define how to calculate the _retained exponent_ E_{G} associated with the root G, which will be obtained from intermediate quantities E_{G}^{(r)} for regions r\in[6]. We start by introducing how to compute E^{(1)}_{G} for region r=1, where \pi_{1} is the identity permutation. In all summations below, s ranges over \mathcal{S}_{\ell^{*}}. We will use the symbols \text{*},\text{+}, and < in our notation below as mnemonic placeholders corresponding to the dimensions, indicating the meanings “unconstrained / any value”, “strictly positive”, and “strictly less than the parent’s coordinate”, respectively. We define

\eta_{G,Y}^{(1)}\coloneqq\sum_{s\,:\,s_{Z}=0}\alpha_{G}^{(1)}(s)\cdot H\left(\beta_{G[s,1],Y}\right)+\sum_{j=0}^{2^{\ell^{*}}}\alpha_{G}^{(1)}(\text{*},j,\text{+})\cdot H\left(\bar{\beta}_{G,Y,\text{*},j,\text{+}}^{(1)}\right),

where

\alpha_{G}^{(1)}(\text{*},j,\text{+})\coloneqq\sum_{s\,:\,s_{Y}=j,\;s_{Z}>0}\alpha_{G}^{(1)}(s),\qquad\bar{\beta}_{G,Y,\text{*},j,\text{+}}^{(1)}\coloneqq\frac{1}{\alpha_{G}^{(1)}(\text{*},j,\text{+})}\sum_{s\,:\,s_{Y}=j,\;s_{Z}>0}\alpha_{G}^{(1)}(s)\cdot\beta_{G[s,1],Y}.

We then define

\eta_{G,Z}^{(1)}\coloneqq\sum_{s\,:\,s_{X}=0\ \mathrm{or}\ s_{Y}=0}\alpha_{G}^{(1)}(s)\cdot H\left(\beta_{G[s,1],Z}\right)+\sum_{k=0}^{2^{\ell^{*}}}\alpha_{G}^{(1)}(\text{+},\text{+},k)\cdot H\left(\bar{\beta}_{G,Z,\text{+},\text{+},k}^{(1)}\right),

where

\alpha_{G}^{(1)}(\text{+},\text{+},k)\coloneqq\sum_{s\,:\,s_{X}>0,\;s_{Y}>0,\;s_{Z}=k}\alpha_{G}^{(1)}(s),\qquad\bar{\beta}_{G,Z,\text{+},\text{+},k}^{(1)}\coloneqq\frac{1}{\alpha_{G}^{(1)}(\text{+},\text{+},k)}\sum_{s\,:\,s_{X}>0,\;s_{Y}>0,\;s_{Z}=k}\alpha_{G}^{(1)}(s)\cdot\beta_{G[s,1],Z}.

To avoid the issue of denominators \alpha_{G}^{(1)}(\cdot) being zero, we regard 0\cdot\textsf{undefined}\coloneqq 0, so that [Section 2.3](https://arxiv.org/html/2608.16884#S2.Ex19 "Retained exponent at root node. ‣ 2.3 Derived quantities ‣ 2 Optimization problem ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve") and [Section 2.3](https://arxiv.org/html/2608.16884#S2.Ex22 "Retained exponent at root node. ‣ 2.3 Derived quantities ‣ 2 Optimization problem ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve") are always well-defined.

The quantity E_{G}^{(1)} is then given by

E_{G}^{(1)}\coloneqq\min\left\{H\left(\big(\alpha_{G}^{(1)}\big)_{X}\right)-P_{\mathcal{S}_{\ell^{*}}}\left(\alpha_{G}^{(1)}\right),\;H\left(\bar{\beta}_{G,Y,\text{*},\text{*},\text{*}}^{(1)}\right)-\eta_{G,Y}^{(1)},\;H\left(\bar{\beta}_{G,Z,\text{*},\text{*},\text{*}}^{(1)}\right)-\eta_{G,Z}^{(1)}\right\},

where

\bar{\beta}_{G,W,\text{*},\text{*},\text{*}}^{(1)}\coloneqq\sum_{s\in\mathcal{S}_{\ell^{*}}}\alpha_{G}^{(1)}(s)\cdot\beta_{G[s,1],W},\qquad\forall\,W\in\{X,Y,Z\}.

For r=2,\dots,6, we define E_{G}^{(r)} by applying the same formula [Section 2.3](https://arxiv.org/html/2608.16884#S2.Ex25 "Retained exponent at root node. ‣ 2.3 Derived quantities ‣ 2 Optimization problem ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve") after relabeling the coordinates X,Y,Z as \pi_{r}(X),\pi_{r}(Y),\pi_{r}(Z), and replacing the region index 1 with r. Finally, we set

E_{G}\coloneqq\sum_{r=1}^{6}A_{G}^{(r)}\cdot E_{G}^{(r)}.

#### Retained exponents at level \ell\geq 3.

For each positive-shape node T at level \ell\geq 3, we will calculate intermediate quantities E_{T,W}^{(r)} for each region r\in[6] and W\in\{X,Y,Z\}. These quantities will later be aggregated to form the _retained exponent_ of level \ell. Note that the region index r in the calculation can be different from the region index r_{T} of node T itself. As in the above, we start by defining the quantities for region r=1 where \pi_{1} is the identity permutation. In the summations below, u ranges over \Split(s_{T}).

We first define

\displaystyle\eta_{T,Y}^{(1)}\displaystyle\coloneqq\sum_{u\,:\,u_{Z}=0}\Bigl(\alpha_{T}^{(1)}(u)+\alpha_{T}^{(1)}(s_{T}\!-\!u)\Bigr)\cdot H\left(\beta_{T[u,1],Y}\right)
\displaystyle+\sum_{j=0}^{\min\{s_{T,Y},\,2^{\ell-1}\}}\Bigl(\alpha_{T}^{(1)}(\text{*},j,\text{+})+\alpha_{T}^{(1)}(\text{*},\,s_{T,Y}\!-\!j,\,\text{<})\Bigr)\cdot H\left(\bar{\beta}_{T,Y,\text{*},j,\text{+}}^{(1)}\right),

where

\displaystyle\alpha_{T}^{(1)}(\text{*},j,\text{+})\coloneqq\sum_{u\,:\,u_{Y}=j,\;u_{Z}>0}\alpha_{T}^{(1)}(u),\qquad\alpha_{T}^{(1)}(\text{*},\,s_{T,Y}\!-\!j,\,\text{<})\coloneqq\sum_{\begin{subarray}{c}u\,:\,u_{Y}=s_{T,Y}-j,\\
u_{Z}<s_{T,Z}\end{subarray}}\alpha_{T}^{(1)}(u),

\bar{\beta}_{T,Y,\text{*},j,\text{+}}^{(1)}\coloneqq\frac{1}{\alpha_{T}^{(1)}(\text{*},j,\text{+})+\alpha_{T}^{(1)}(\text{*},\,s_{T,Y}\!-\!j,\,\text{<})}\cdot\sum_{u\,:\,u_{Y}=j,\;u_{Z}>0}\Bigl(\alpha_{T}^{(1)}(u)+\alpha_{T}^{(1)}(s_{T}\!-\!u)\Bigr)\cdot\beta_{T[u,1],Y}.

We also define

\displaystyle\eta_{T,Z}^{(1)}\displaystyle\coloneqq\sum_{u\,:\,u_{X}=0\ \mathrm{or}\ u_{Y}=0}\Bigl(\alpha_{T}^{(1)}(u)+\alpha_{T}^{(1)}(s_{T}-u)\Bigr)\cdot H\left(\beta_{T[u,1],Z}\right)
\displaystyle+\sum_{k=0}^{\min\{s_{T,Z},\,2^{\ell-1}\}}\Bigl(\alpha_{T}^{(1)}(\text{+},\text{+},k)+\alpha_{T}^{(1)}(\text{<},\,\text{<},\,s_{T,Z}\!-\!k)\Bigr)\cdot H\left(\bar{\beta}_{T,Z,\text{+},\text{+},k}^{(1)}\right),

where

\displaystyle\alpha_{T}^{(1)}(\text{+},\text{+},k)\coloneqq\sum_{u\,:\,u_{X}>0,\;u_{Y}>0,\;u_{Z}=k}\alpha_{T}^{(1)}(u),\qquad\alpha_{T}^{(1)}(\text{<},\,\text{<},\,s_{T,Z}\!-\!k)\coloneqq\sum_{\begin{subarray}{c}u\,:\,u_{X}<s_{T,X},\;u_{Y}<s_{T,Y},\\
u_{Z}=s_{T,Z}-k\end{subarray}}\alpha_{T}^{(1)}(u),

\bar{\beta}_{T,Z,\text{+},\text{+},k}^{(1)}\coloneqq\frac{1}{\alpha_{T}^{(1)}(\text{+},\text{+},k)+\alpha_{T}^{(1)}(\text{<},\,\text{<},\,s_{T,Z}\!-\!k)}\cdot\sum_{u\,:\,u_{X}>0,\;u_{Y}>0,\;u_{Z}=k}\Bigl(\alpha_{T}^{(1)}(u)+\alpha_{T}^{(1)}(s_{T}\!-\!u)\Bigr)\cdot\beta_{T[u,1],Z}.

Again, as we define 0\cdot\textsf{undefined}\coloneqq 0, [Section 2.3](https://arxiv.org/html/2608.16884#S2.Ex29 "Retained exponents at level 
            
              
                ≥
                ℓ
                3
              
            
          . ‣ 2.3 Derived quantities ‣ 2 Optimization problem ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve") and [Section 2.3](https://arxiv.org/html/2608.16884#S2.Ex33 "Retained exponents at level 
            
              
                ≥
                ℓ
                3
              
            
          . ‣ 2.3 Derived quantities ‣ 2 Optimization problem ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve") are well-defined even when some denominators \alpha_{T}^{(1)} are zeros.

Then, E^{(1)}_{T,W} are given by

\displaystyle E_{T,X}^{(1)}\displaystyle\coloneqq m_{T}A_{T}^{(1)}\cdot\Big(H\left(\big(\alpha_{T}^{(1)}\big)_{X}\right)-P_{\Split(s_{T})}\left(\alpha_{T}^{(1)}\right)\Big),
\displaystyle E_{T,Y}^{(1)}\displaystyle\coloneqq m_{T}A_{T}^{(1)}\cdot\Big(H\left(\beta_{T,Y}^{(1)}\right)-\eta_{T,Y}^{(1)}\Big),
\displaystyle E_{T,Z}^{(1)}\displaystyle\coloneqq m_{T}A_{T}^{(1)}\cdot\Big(H\left(\beta_{T,Z}^{(1)}\right)-\eta_{T,Z}^{(1)}\Big).

For regions r=2,\dots,6, we define E_{T,\pi_{r}(X)}^{(r)}, E_{T,\pi_{r}(Y)}^{(r)}, and E_{T,\pi_{r}(Z)}^{(r)} by applying [Section 2.3](https://arxiv.org/html/2608.16884#S2.Ex29 "Retained exponents at level 
            
              
                ≥
                ℓ
                3
              
            
          . ‣ 2.3 Derived quantities ‣ 2 Optimization problem ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"), [Section 2.3](https://arxiv.org/html/2608.16884#S2.Ex33 "Retained exponents at level 
            
              
                ≥
                ℓ
                3
              
            
          . ‣ 2.3 Derived quantities ‣ 2 Optimization problem ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"), and [Section 2.3](https://arxiv.org/html/2608.16884#S2.Ex37 "Retained exponents at level 
            
              
                ≥
                ℓ
                3
              
            
          . ‣ 2.3 Derived quantities ‣ 2 Optimization problem ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve") after relabeling coordinates X,Y,Z as \pi_{r}(X),\pi_{r}(Y),\pi_{r}(Z) and replacing region index 1 by r.

To calculate the _retained exponent_ for level \ell\in[3,\ell^{*}], we let \mathcal{T}_{\ell}^{+} denote the set of positive-shape nodes at level \ell. Then, the retained exponent at level \ell is given by

E_{\ell}\coloneqq\sum_{r=1}^{6}\min\left\{\sum_{T\in\mathcal{T}_{\ell}^{+}}E_{T,X}^{(r)},\;\sum_{T\in\mathcal{T}_{\ell}^{+}}E_{T,Y}^{(r)},\;\sum_{T\in\mathcal{T}_{\ell}^{+}}E_{T,Z}^{(r)}\right\}.

#### Retained exponent at level 2.

For a positive level-2 node T of shape (1,1,2), we define

\left(E_{T,X},E_{T,Y},E_{T,Z}\right)\coloneqq m_{T}\cdot\left(1,\;1,\;H\left(\mu_{T},\mu_{T},1-2\mu_{T}\right)\right).

Similarly, for a node T of shape (2,1,1), we define

\left(E_{T,X},E_{T,Y},E_{T,Z}\right)\coloneqq m_{T}\cdot\left(H\left(\mu_{T},\mu_{T},1-2\mu_{T}\right),\;1,\;1\right),

and for a node T of shape (1,2,1), we define

\left(E_{T,X},E_{T,Y},E_{T,Z}\right)\coloneqq m_{T}\cdot\left(1,\;H\left(\mu_{T},\mu_{T},1-2\mu_{T}\right),\;1\right),

Then, letting \mathcal{T}_{2}^{+} denote the set of positive-shape nodes at level 2, the _retained exponent_ at level 2 is given by

E_{2}\coloneqq\min\left\{\sum_{T\in\mathcal{T}_{2}^{+}}E_{T,X},\;\sum_{T\in\mathcal{T}_{2}^{+}}E_{T,Y},\;\sum_{T\in\mathcal{T}_{2}^{+}}E_{T,Z}\right\}.

#### Local matrix size for zero-shape nodes.

For every zero-shape node T, we let W_{0}\in\{X,Y,Z\} be the first zero coordinate of s_{T} and W_{1} be the first nonzero coordinate of s_{T}. Then, we define the _local matrix size_ of T, written (M_{T,X},M_{T,Y},M_{T,Z}), as

M_{T,W_{0}}\coloneqq m_{T}\cdot\Bigg(H\left(\beta_{T,W_{1}}\right)+\sum_{L\in\supp(\beta_{T,W_{1}})}\beta_{T,W_{1}}(L)\cdot\left|\left\{p\in[2^{\ell-1}]\;\middle|\;L_{p}=1\right\}\right|\cdot\log q\Bigg);

the other two coordinates in the local matrix size are zeros.

#### Local matrix size for (1,1,2)-nodes.

For a node T with shape (1,1,2), we define its local matrix size as

\left(M_{T,X},M_{T,Y},M_{T,Z}\right)\coloneqq m_{T}\cdot\left((1-2\mu_{T})\log q,\;(1-2\mu_{T})\log q,\;2\mu_{T}\log q\right).

Similarly, for a node T with shape (2,1,1), we define its local matrix size as

\left(M_{T,X},M_{T,Y},M_{T,Z}\right)\coloneqq m_{T}\cdot\left(2\mu_{T}\log q,\;(1-2\mu_{T})\log q,\;(1-2\mu_{T})\log q\right),

and for a node T with shape (1,2,1), we define its local matrix size as

\left(M_{T,X},M_{T,Y},M_{T,Z}\right)\coloneqq m_{T}\cdot\left((1-2\mu_{T})\log q,\;2\mu_{T}\log q,\;(1-2\mu_{T})\log q\right).

### 2.4 Final assembly

Finally, we define the _total retained exponent_ as

E_{\text{total}}\coloneqq E_{G}+E_{2}+\sum_{\ell=3}^{\ell^{*}}E_{\ell},

and define the _total matrix size_ as

M_{\text{total}}\coloneqq\min\left\{\sum_{T\in\mathcal{L}}M_{T,X},\;\sum_{T\in\mathcal{L}}M_{T,Y},\;\sum_{T\in\mathcal{L}}M_{T,Z}\right\},

where \mathcal{L} is the set of leaves. We can now write the full optimization problem.

###### Theorem 1([1](https://arxiv.org/html/2608.16884#bib.bib17)).

Any feasible solution of [Equation 11](https://arxiv.org/html/2608.16884#S2.E11 "In 2.4 Final assembly ‣ 2 Optimization problem ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve") implies \omega\leq\Omega, where \omega is the asymptotic matrix multiplication exponent.

### 2.5 Dealing with maximum entropies

Among the quantities introduced above, H_{D}^{\max}(\rho) is the only type that cannot be computed analytically. However, it is easy to see that replacing each occurrence of H_{D}^{\max} in the computation of E_{\text{total}} with its upper bound can only decrease E_{\text{total}}, so any solution that satisfies the constraints in [Equation 11](https://arxiv.org/html/2608.16884#S2.E11 "In 2.4 Final assembly ‣ 2 Optimization problem ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve") after this replacement is feasible for [Equation 11](https://arxiv.org/html/2608.16884#S2.E11 "In 2.4 Final assembly ‣ 2 Optimization problem ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve") itself, and still yields a valid upper bound \omega\leq\Omega. In this section, we describe how to derive an upper bound of H_{D}^{\max}(\rho).

For a set D of shapes and a probability distribution \rho\in\Delta(D), a _valid certificate_ for H_{D}^{\max}(\rho) consists of:

1.   1.
a distribution y\in\Delta(D) satisfying y(a)>0 for every a\in D and y_{W}=\rho_{W} for every W\in\{X,Y,Z\};

2.   2.
a real number \lambda_{0} and, for each W\in\{X,Y,Z\} and each marginal value w\in\{a_{W}\;|\;a\in D\}, a real number \lambda_{W}(w) (these values play the role of Lagrange multipliers for the maximization defining H_{D}^{\max}(\rho)); and

3.   3.a real number \varepsilon\geq 0 satisfying, for every a=(a_{X},a_{Y},a_{Z})\in D,

\big\lvert\log y(a)-\big(\lambda_{0}+\lambda_{X}(a_{X})+\lambda_{Y}(a_{Y})+\lambda_{Z}(a_{Z})\big)\big\rvert\leq\varepsilon. 

###### Lemma 1.

For any valid certificate, we have H(y)\leq H_{D}^{\max}(\rho)\leq H(y)+2\varepsilon.

###### Proof.

The left inequality follows from the definition: y has the same marginals as \rho, so it is feasible for the maximization defining H_{D}^{\max}(\rho) (see [Section 2.3](https://arxiv.org/html/2608.16884#S2.Ex6 "2.3 Derived quantities ‣ 2 Optimization problem ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve")).

For the right inequality, fix any \rho^{\prime}\in\Delta(D) with \rho^{\prime}_{W}=\rho_{W} for all W\in\{X,Y,Z\}; to prove the right inequality we need only show H(\rho^{\prime})\leq H(y)+2\varepsilon.

The entropy function H is concave on \Delta(D), and since y is strictly positive, H is differentiable at y with \nabla H(y)_{a}=-\log y(a)-\log e (recall that all logarithms are in base 2). Therefore

H(\rho^{\prime})\leq H(y)+\big\langle\nabla H(y),\,\rho^{\prime}-y\big\rangle.

Write g(a)\coloneqq\lambda_{0}+\lambda_{X}(a_{X})+\lambda_{Y}(a_{Y})+\lambda_{Z}(a_{Z}), and view g, \log y, and the constant \log e as vectors indexed by a\in D. Then,

\displaystyle\big\langle g+\log e,\,\rho^{\prime}-y\big\rangle\displaystyle=\big(\lambda_{0}+\log e\big)\cdot\sum_{a\in D}\big(\rho^{\prime}(a)-y(a)\big)+\sum_{W\in\{X,Y,Z\}}\,\sum_{a\in D}\lambda_{W}(a_{W})\cdot\big(\rho^{\prime}(a)-y(a)\big)
\displaystyle=\big(\lambda_{0}+\log e\big)\cdot(1-1)+\sum_{W\in\{X,Y,Z\}}\,\sum_{w\in\{a_{W}\mid a\in D\}}\lambda_{W}(w)\cdot\big(\rho^{\prime}_{W}(w)-y_{W}(w)\big)=0,

where the last step uses \rho^{\prime}_{W}=\rho_{W}=y_{W} for each W\in\{X,Y,Z\}. Hence,

\big\langle\nabla H(y),\,\rho^{\prime}-y\big\rangle=\big\langle g-\log y,\,\rho^{\prime}-y\big\rangle\leq\varepsilon\cdot\norm{\rho' - y}_{1}\leq 2\varepsilon,

where the first inequality uses [Item 3](https://arxiv.org/html/2608.16884#S2.Ex52 "In 2.5 Dealing with maximum entropies ‣ 2 Optimization problem ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve") and the second one uses the fact that the \ell_{1}-distance between two probability distributions is at most 2. Thus, H(\rho^{\prime})\leq H(y)+2\varepsilon, proving the lemma. ∎

## 3 Solving the optimization problem numerically

We wish to numerically minimize \Omega from [Equation 11](https://arxiv.org/html/2608.16884#S2.E11 "In 2.4 Final assembly ‣ 2 Optimization problem ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve") while satisfying all the constraints, with q=5 and \ell^{*}=4. In [1](https://arxiv.org/html/2608.16884#bib.bib17), this was done (for \ell^{*}=3) via a sequential quadratic programming (SQP) algorithm using the software package SNOPT ([12](https://arxiv.org/html/2608.16884#bib.bib18)). In contrast, we take a gradient-based approach to tackle the non-convex minimization. We develop an algorithm applying several techniques from machine learning and adjacent areas, such as optimal transport.

### 3.1 Differentiable objective

To apply a gradient-based optimization algorithm, we need a differentiable objective function to be optimized. However, obtaining a differentiable objective from [Equation 11](https://arxiv.org/html/2608.16884#S2.E11 "In 2.4 Final assembly ‣ 2 Optimization problem ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve") is challenging due to the problem constraints. Since most of the parameters to be optimized are distributions, this creates the constraints that the probabilities must be non-negative and must add up to one. To sidestep that issue and obtain unconstrained optimization, we parameterize distributions in terms of their logits, and obtain the distribution by applying the softmax operation on the free parameters. We initialize the algorithm with random (logit) parameters.

The other relevant set of constraints comes from the maximum entropy distributions. Given an input probability distribution, we must find the distribution with maximum entropy that shares the same marginals as the input. This is reminiscent of a problem that typically arises in optimal transport, where the Sinkhorn-Knopp algorithm ([19](https://arxiv.org/html/2608.16884#bib.bib20)) is used to obtain a stable and differentiable objective ([9](https://arxiv.org/html/2608.16884#bib.bib19)). Thus, unlike [1](https://arxiv.org/html/2608.16884#bib.bib17), we do not treat the maximum entropy distributions (and the corresponding Lagrange multipliers) as free parameters to be optimized together with the rest of distributions; rather, we obtain the maximum entropy distributions (and the Lagrange multipliers) using the Sinkhorn-Knopp algorithm.

We compute the gradients of the objective using automatic differentiation. To improve stability, we employ implicit differentiation for backpropagating through the Sinkhorn-Knopp algorithm ([8](https://arxiv.org/html/2608.16884#bib.bib22); [11](https://arxiv.org/html/2608.16884#bib.bib21)). We update the parameters using Adam ([13](https://arxiv.org/html/2608.16884#bib.bib2)).

### 3.2 Software implementation

To obtain a fast algorithm that can leverage hardware platforms (such as GPUs) and scale up to level \ell^{*}=4 (with nearly 7 million parameters to optimize) we implement our algorithm in Jax ([5](https://arxiv.org/html/2608.16884#bib.bib3)), applying some techniques to improve efficiency. The key to an effective speed-up is finding a new representation for the problem, switching from a loop over graph nodes (as done in the SQP implementation from [1](https://arxiv.org/html/2608.16884#bib.bib17)) to a naturally parallelizable computation over tensors. In the implementation from [1](https://arxiv.org/html/2608.16884#bib.bib17), the free parameters are stored in the nodes of a graph where updates happen through message-passing (both from child nodes to parent nodes and vice versa). The graph is non-uniform enough (e.g., different nodes have different number of children and different types of them) to make parallelization hard, but we solve this via two techniques. Firstly, we introduce phantom nodes in the graph with masking; as a trade-off, we pay the cost of increasing the number of optimization parameters by up to a factor of 3. Secondly, we cluster the nodes into a limited number of highly specialized groups (“stages”). With these techniques, we are able to represent the full graph with multi-dimensional tensors, enabling parallel processing over a number of axes (up to 10), challenging the limits of tensor-processing backends.

### 3.3 Applying AlphaEvolve

We use AlphaEvolve ([15](https://arxiv.org/html/2608.16884#bib.bib23)) to further improve the optimization algorithm. Specifically, we let AlphaEvolve modify the optimization program, which is then executed (taking approximately 5 hours on a single GPU) to output a bound on omega. AlphaEvolve then evolves the code to minimize omega. We found improved results by using AlphaEvolve’s “evolving constructions” feature, where the optimization algorithm at each generation starts at the best solution point found by the parent algorithm.

## 4 Rigorous verification of the omega upper bound

To rigorously certify our omega bound, we run a separate verification step computing all quantities in rational arithmetic to guard against floating point errors. More specifically, we round the floating point solution obtained at the end of the optimization to rational numbers, ensuring the maximum-entropy certificates stay valid. We then evaluate all derived quantities in exact rational arithmetic, and replace each logarithm with a rational bound rounded in the proper direction that ensures each constraint in [Equation 11](https://arxiv.org/html/2608.16884#S2.E11 "In 2.4 Final assembly ‣ 2 Optimization problem ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve") is satisfied, so the certified bounds are free from numerical errors. We are preparing a repository in which we will release the verification code and our discovered solution.

## 5 Discussion

In this note, we improve the SOTA upper bound on \omega to 2.371177. Our improvement is comparable in magnitude to most improvements in the last 40 years since \omega<2.376 was attained by [7](https://arxiv.org/html/2608.16884#bib.bib10). We achieve this by leveraging modern optimization techniques and AlphaEvolve to design a better optimization algorithm for the problem described by [1](https://arxiv.org/html/2608.16884#bib.bib17). While further modest improvements may be obtained in this manner, achieving larger improvements to \omega likely requires new mathematical ideas and is an exciting area of research.

#### Acknowledgments.

The work presented in this note was motivated by initial conversations during the Complexity and Linear Algebra program at Simons Institute for the Theory of Computing in Fall 2025.

## References

*   Alman et al. (2025)J. Alman, R. Duan, V. V. Williams, Y. Xu, Z. Xu, and R. Zhou More asymmetry yields faster matrix multiplication. In Proceedings of the 2025 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pp.2005–2039. External Links: [Document](https://dx.doi.org/10.1137/1.9781611978322.63), [Link](https://epubs.siam.org/doi/abs/10.1137/1.9781611978322.63), https://epubs.siam.org/doi/pdf/10.1137/1.9781611978322.63 Cited by: [Table 1](https://arxiv.org/html/2608.16884#S1.T1.2.3.1 "In 1 Introduction ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"), [§1](https://arxiv.org/html/2608.16884#S1.p2.1 "1 Introduction ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"), [§1](https://arxiv.org/html/2608.16884#S1.p4.1 "1 Introduction ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"), [§1](https://arxiv.org/html/2608.16884#S1.p5.1 "1 Introduction ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"), [§2](https://arxiv.org/html/2608.16884#S2.p1.1 "2 Optimization problem ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"), [§3.1](https://arxiv.org/html/2608.16884#S3.SS1.p2.1 "3.1 Differentiable objective ‣ 3 Solving the optimization problem numerically ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"), [§3.2](https://arxiv.org/html/2608.16884#S3.SS2.p1.1 "3.2 Software implementation ‣ 3 Solving the optimization problem numerically ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"), [§3](https://arxiv.org/html/2608.16884#S3.p1.1 "3 Solving the optimization problem numerically ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"), [§5](https://arxiv.org/html/2608.16884#S5.p1.1 "5 Discussion ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"), [Theorem 1](https://arxiv.org/html/2608.16884#Thmtheorem1 "Theorem 1 (
              (
              )
            ). ‣ 2.4 Final assembly ‣ 2 Optimization problem ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"), [Abstract](https://arxiv.org/html/2608.16884#abstract1.1 "Abstract ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"). 
*   Alman and Williams (2024)J. Alman and V. V. Williams A refined laser method and faster matrix multiplication. TheoretiCS 3 (11261). Cited by: [§1](https://arxiv.org/html/2608.16884#S1.p2.1 "1 Introduction ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"). 
*   Bini et al. (1979)D. Bini, M. Capovani, F. Romani, and G. Lotti\mathcal{O}(n^{2.7799}) Complexity for n\times n approximate matrix multiplication. Information Processing Letters 8 (5), pp.234–235. External Links: ISSN 0020-0190 Cited by: [§1](https://arxiv.org/html/2608.16884#S1.p2.1 "1 Introduction ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"). 
*   Bläser (2013)M. Bläser Fast matrix multiplication. Theory of Computing, pp.1–60. Cited by: [§1](https://arxiv.org/html/2608.16884#S1.p1.1 "1 Introduction ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"). 
*   Bradbury et al. (2018)J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, C. Leary, D. Maclaurin, G. Necula, A. Paszke, J. VanderPlas, S. Wanderman-Milne, and Q. Zhang JAX: composable transformations of Python+NumPy programs. External Links: [Link](http://github.com/jax-ml/jax)Cited by: [§1](https://arxiv.org/html/2608.16884#S1.p4.1 "1 Introduction ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"), [§3.2](https://arxiv.org/html/2608.16884#S3.SS2.p1.1 "3.2 Software implementation ‣ 3 Solving the optimization problem numerically ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"). 
*   Coppersmith and Winograd (1981)D. Coppersmith and S. Winograd On the asymptotic complexity of matrix multiplication. In 22nd Annual Symposium on Foundations of Computer Science (sfcs 1981), Vol. , pp.82–90. Cited by: [§1](https://arxiv.org/html/2608.16884#S1.p2.1 "1 Introduction ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"). 
*   Coppersmith and Winograd (1990)D. Coppersmith and S. Winograd Matrix multiplication via arithmetic progressions. Journal of Symbolic Computation 9 (3), pp.251–280. Note: Computational algebraic complexity editorial External Links: ISSN 0747-7171 Cited by: [§1](https://arxiv.org/html/2608.16884#S1.p2.1 "1 Introduction ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"), [§2](https://arxiv.org/html/2608.16884#S2.p1.1 "2 Optimization problem ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"), [§5](https://arxiv.org/html/2608.16884#S5.p1.1 "5 Discussion ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"). 
*   Cuturi et al. (2020)M. Cuturi, O. Teboul, J. Niles-Weed, and J. Vert Supervised quantile normalization for low-rank matrix approximation. In International Conference on Machine Learning, Cited by: [§3.1](https://arxiv.org/html/2608.16884#S3.SS1.p3.1 "3.1 Differentiable objective ‣ 3 Solving the optimization problem numerically ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"). 
*   Cuturi (2013)M. Cuturi Sinkhorn distances: lightspeed computation of optimal transport. In International Conference on Neural Information Processing Systems, pp.2292––2300. Cited by: [§3.1](https://arxiv.org/html/2608.16884#S3.SS1.p2.1 "3.1 Differentiable objective ‣ 3 Solving the optimization problem numerically ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"). 
*   Duan et al. (2023)R. Duan, H. Wu, and R. Zhou Faster matrix multiplication via asymmetric hashing. In 2023 IEEE 64th Annual Symposium on Foundations of Computer Science (FOCS), Vol. , pp.2129–2138. External Links: [Document](https://dx.doi.org/10.1109/FOCS57990.2023.00130)Cited by: [Table 1](https://arxiv.org/html/2608.16884#S1.T1.2.1.1 "In 1 Introduction ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"), [§1](https://arxiv.org/html/2608.16884#S1.p2.1 "1 Introduction ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"), [Abstract](https://arxiv.org/html/2608.16884#abstract1.1 "Abstract ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"). 
*   Eisenberger et al. (2022)M. Eisenberger, A. Toker, L. Leal-Taixé, F. Bernard, and D. Cremers A unified framework for implicit Sinkhorn differentiation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp.499–508. Cited by: [§3.1](https://arxiv.org/html/2608.16884#S3.SS1.p3.1 "3.1 Differentiable objective ‣ 3 Solving the optimization problem numerically ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"). 
*   Gill et al. (2002)P. E. Gill, W. Murray, and M. A. Saunders SNOPT: an SQP algorithm for large-scale constrained optimization. SIAM Journal on Optimization 12 (4), pp.979–1006. Cited by: [§3](https://arxiv.org/html/2608.16884#S3.p1.1 "3 Solving the optimization problem numerically ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"). 
*   Kingma and Ba (2015)D. P. Kingma and J. Ba Adam: a method for stochastic optimization. In International Conference on Learning Representations (ICLR), Cited by: [§3.1](https://arxiv.org/html/2608.16884#S3.SS1.p3.1 "3.1 Differentiable objective ‣ 3 Solving the optimization problem numerically ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"). 
*   Le Gall (2014)F. Le Gall Algebraic complexity theory and matrix multiplication. In Proceedings of the 39th International Symposium on Symbolic and Algebraic Computation, ISSAC ’14, New York, NY, USA, pp.23. External Links: ISBN 9781450325011 Cited by: [§1](https://arxiv.org/html/2608.16884#S1.p2.1 "1 Introduction ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"). 
*   Novikov et al. (2025)A. Novikov, N. Vũ, M. Eisenberger, E. Dupont, P. Huang, A. Z. Wagner, S. Shirobokov, B. Kozlovskii, F. J. R. Ruiz, A. Mehrabian, M. P. Kumar, A. See, S. Chaudhuri, G. Holland, A. Davies, S. Nowozin, P. Kohli, and M. Balog AlphaEvolve: a coding agent for scientific and algorithmic discovery. arXiv. External Links: 2506.13131, [Link](https://arxiv.org/abs/2506.13131)Cited by: [§1](https://arxiv.org/html/2608.16884#S1.p3.1 "1 Introduction ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"), [§3.3](https://arxiv.org/html/2608.16884#S3.SS3.p1.1 "3.3 Applying AlphaEvolve ‣ 3 Solving the optimization problem numerically ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"). 
*   Pan (1978)V. Y. Pan Strassen’s algorithm is not optimal trilinear technique of aggregating, uniting and canceling for constructing fast algorithms for matrix operations. In 19th Annual Symposium on Foundations of Computer Science, Vol. , pp.166–176. Cited by: [§1](https://arxiv.org/html/2608.16884#S1.p2.1 "1 Introduction ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"). 
*   Romani (1982)F. Romani Some properties of disjoint sums of tensors related to matrix multiplication. SIAM Journal on Computing 11 (2), pp.263–267. Cited by: [§1](https://arxiv.org/html/2608.16884#S1.p2.1 "1 Introduction ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"). 
*   Schönhage (1981)A. Schönhage Partial and total matrix multiplication. SIAM Journal on Computing 10 (3), pp.434–455. Cited by: [§1](https://arxiv.org/html/2608.16884#S1.p2.1 "1 Introduction ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"). 
*   Sinkhorn and Knopp (1967)R. Sinkhorn and P. Knopp Concerning nonnegative matrices and doubly stochastic matrices. Pacific Journal of Mathematics 21, pp.343–348. Cited by: [§3.1](https://arxiv.org/html/2608.16884#S3.SS1.p2.1 "3.1 Differentiable objective ‣ 3 Solving the optimization problem numerically ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"). 
*   Stothers (2010)A. J. Stothers On the complexity of matrix multiplication. PhD thesis, University of Edinburgh, Edinburgh, United Kingdom. Cited by: [§1](https://arxiv.org/html/2608.16884#S1.p2.1 "1 Introduction ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"). 
*   Strassen (1986)V. Strassen The asymptotic spectrum of tensors and the exponent of matrix multiplication. In 27th Annual Symposium on Foundations of Computer Science, Vol. , pp.49–54. Cited by: [§1](https://arxiv.org/html/2608.16884#S1.p2.1 "1 Introduction ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"). 
*   Strassen (1969)V. Strassen Gaussian elimination is not optimal. Numerische mathematik 13 (4), pp.354–356. Cited by: [§1](https://arxiv.org/html/2608.16884#S1.p2.1 "1 Introduction ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"). 
*   Williams et al. (2024)V. V. Williams, Y. Xu, Z. Xu, and R. Zhou New bounds for matrix multiplication: from alpha to omega. In Proceedings of the 2024 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pp.3792–3835. External Links: [Document](https://dx.doi.org/10.1137/1.9781611977912.134), [Link](https://epubs.siam.org/doi/abs/10.1137/1.9781611977912.134), https://epubs.siam.org/doi/pdf/10.1137/1.9781611977912.134 Cited by: [Table 1](https://arxiv.org/html/2608.16884#S1.T1.2.2.1 "In 1 Introduction ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"), [§1](https://arxiv.org/html/2608.16884#S1.p2.1 "1 Introduction ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"), [Abstract](https://arxiv.org/html/2608.16884#abstract1.1 "Abstract ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve"). 
*   Williams (2012)V. V. Williams Multiplying matrices faster than Coppersmith-Winograd. In Proceedings of the Forty-Fourth Annual ACM Symposium on Theory of Computing, STOC ’12, New York, NY, USA, pp.887–898. External Links: ISBN 9781450312455 Cited by: [§1](https://arxiv.org/html/2608.16884#S1.p2.1 "1 Introduction ‣ Improving the matrix multiplication exponent with modern optimization and AlphaEvolve").

