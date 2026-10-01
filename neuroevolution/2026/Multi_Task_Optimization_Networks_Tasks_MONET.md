Title: Multi-Task Optimization over Networks of Tasks

URL Source: https://arxiv.org/html/2604.21991

Markdown Content:
1 1 institutetext: Vrije Universiteit Amsterdam, Amsterdam, The Netherlands 

1 1 email: {j.hatzky, a.e.eiben, a.yaman}@vu.nl 2 2 institutetext: TH Köln, Cologne, Germany 

2 2 email: thomas.bartz-beielstein@th-koeln.de

###### Abstract

Multi-task optimization is a powerful approach for solving a large number of tasks in parallel. However, existing algorithms face distinct limitations: Population-based methods scale poorly and remain underexplored for large task sets. Approaches that do scale beyond a thousand tasks are mostly MAP-Elites variants and rely on a fixed, discretized archive that disregards the topology of the task space. We introduce MONET (Multi-Task Optimization over Networks of Tasks), a multi-task optimization algorithm that models the task space as a graph: tasks are nodes, and edges connect tasks in the task parameter space. This representation enables knowledge transfer between tasks and remains tractable for high-dimensional problems while exploiting the topology of the task space. MONET combines _social learning_, which generates candidates from neighboring nodes via crossover, with _individual learning_, which refines a node’s own solution independently via mutation. We evaluate MONET on four domains (archery, arm, and cartpole with 5,000 tasks each; hexapod with 2,000 tasks) and show that it matches or exceeds the performance of existing MAP–Elites-based baselines across all four domains.

## 1 Introduction

To maintain reliable performance, real-world embodied agents must remain operational even when their configurations change. For example, in scenarios where a legged robot loses a limb[[14](https://arxiv.org/html/2604.21991#bib.bib55 "Robots that can adapt like animals"), [11](https://arxiv.org/html/2604.21991#bib.bib100 "Resilient machines through continuous self-modeling")] or a modular robot reconfigures itself to navigate a narrow passage[[52](https://arxiv.org/html/2604.21991#bib.bib101 "Modular self-reconfigurable robot systems"), [46](https://arxiv.org/html/2604.21991#bib.bib103 "A model-free method to learn multiple skills in parallel on modular robots"), [43](https://arxiv.org/html/2604.21991#bib.bib104 "Metaheuristic based navigation of a reconfigurable robot through narrow spaces with shape changing ability")], they are required to maintain their performance. In such cases, the ability to adapt an existing control strategy instead of relearning a new strategy for the new configuration from scratch can prove to be more efficient. It is possible to achieve this by employing the multi-task optimization approach[[22](https://arxiv.org/html/2604.21991#bib.bib13 "Multifactorial evolution: toward evolutionary multitasking")], where each configuration variant (e.g. morphology) is treated as a distinct task, and solutions (e.g. controller) are pre-computed for all of them, so that when the configurations change, an appropriate controller is already available or is faster to adapt[[36](https://arxiv.org/html/2604.21991#bib.bib79 "Illuminating search spaces by mapping elites"), [12](https://arxiv.org/html/2604.21991#bib.bib102 "Quality-diversity optimization: a novel branch of stochastic optimization")]. In other words, given a parameterized family of tasks, the multi-task optimization approach aims to find a high-quality solution for each task simultaneously and in parallel[[22](https://arxiv.org/html/2604.21991#bib.bib13 "Multifactorial evolution: toward evolutionary multitasking")].

![Image 1: Refer to caption](https://arxiv.org/html/2604.21991v1/x1.png)

Figure 1: Illustration of the MONET task network structure. Each node V_{i}=\boldsymbol{\tau}_{i} represents a task, labelled by its current best solution \mathcal{A}[\boldsymbol{\tau}_{i}]\in\Theta. (a) Task network with the individual and social learning strategies that operate on it. (b) Task network and solution qualities at 500{,}000 evaluations (50\% of the total).

Existing multi-task optimization algorithms face a fundamental scalability–structure tradeoff. Population-based multi-task evolutionary methods[[28](https://arxiv.org/html/2604.21991#bib.bib45 "Ensemble Multifactorial Evolution With Biased Skill-Factor Inheritance for Many-Task Optimization"), [49](https://arxiv.org/html/2604.21991#bib.bib86 "Solving Multitask Optimization Problems With Adaptive Knowledge Transfer via Anomaly Detection")] maintain a separate population per task and quickly become impractical as the number of tasks grows into the thousands and beyond. Quality-diversity (QD) algorithms and MAP-Elites[[14](https://arxiv.org/html/2604.21991#bib.bib55 "Robots that can adapt like animals"), [36](https://arxiv.org/html/2604.21991#bib.bib79 "Illuminating search spaces by mapping elites")] in particular, bypass this by replacing per-task populations with a single structured archive. MAP-Elites has been extended to multi-task settings by Multi-Task MAP-Elites (MT-ME)[[37](https://arxiv.org/html/2604.21991#bib.bib57 "Quality diversity for multi-task optimization")], which solves a fixed set of tasks via a globally shared archive, and Parametric-Task MAP-Elites (PT-ME)[[4](https://arxiv.org/html/2604.21991#bib.bib14 "Parametric-Task MAP-Elites")], which extends this to continuous task spaces. Both have demonstrated strong performance on problems with thousands of tasks.

However, these archive-based methods exhibit two intertwined limitations. First, the archive is a _fixed discretized grid_ over the task or feature space: with c bins per dimension and a d-dimensional task space its size grows as \mathcal{O}(c^{d}), so the representation becomes infeasible already for moderate d. Second, the archive is largely _topologically flat_: in MT-ME any solution can serve as a parent for any task. PT-ME partially addresses this through a local linear regression operator that exploits Delaunay triangulation[[17](https://arxiv.org/html/2604.21991#bib.bib24 "Sur la sphere vide")] between tasks to calculate an adjacency matrix. Even so, this locality is confined to one of two variation paths, and the global Simulated Binary Crossover (SBX) operator[[15](https://arxiv.org/html/2604.21991#bib.bib81 "Simulated binary crossover for continuous search space")] still draws parents without regard for task similarity. Neither method makes the task-space topology a first-class structural element that governs all recombination.

We introduce MONET (Multi-Task Optimization over Networks of Tasks), a multi-task optimization algorithm that resolves both limitations by replacing the flat archive with an explicit graph over tasks (Figure[1](https://arxiv.org/html/2604.21991#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Multi-Task Optimization over Networks of Tasks")): each node in the vertex set \mathcal{V} represents a task and stores its current best solution, while bidirectional edges in \mathcal{E} connect tasks to enable knowledge transfer (e.g., through crossover or direct copy). Unlike a grid, this representation does not depend on a discretization of the task space and scales as \mathcal{O}(|\mathcal{V}|\,k), where k is the number of neighbors per node, remaining tractable as the task space dimensionality grows. Unlike a flat archive, the graph encodes topology directly: every node has a well-defined neighborhood that the search can exploit. The evolutionary process operates on this graph through two complementary mechanisms. _Individual learning_ improves a solution in isolation through mutation, exploring the local fitness landscape of a single task. _Social learning_ transfers information between neighboring nodes in variable ratios by selecting parent solutions from neighboring tasks and generating offspring through crossover. This separation is inspired by cultural evolution, where individual innovation and the social spread of useful innovations through a population jointly drive adaptation[[41](https://arxiv.org/html/2604.21991#bib.bib91 "Why copy others? Insights from the social learning strategies tournament"), [51](https://arxiv.org/html/2604.21991#bib.bib12 "Distributed embodied evolution over networks")].

The balance between individual and social learning, the strategy to select a node (random or best fit), the size of the neighborhood, and the type of neighborhood (closest neighbors, distance-proportional sampling, or random selection) are key design choices that shape how information flows through the graph. A small neighborhood with strong social learning concentrates exploitation among very similar tasks, while a larger neighborhood or weaker social learning permits broader exploration. Understanding which of these choices matters and whether a single robust configuration exists in different problem domains is essential for the practical application of MONET.

To this end, we conducted a hyperparameter study in four environments: archery, arm, and cartpole with 5,000 tasks each, and hexapod with 2,000 tasks. We show that MONET outperforms both MT-ME and PT-ME across all four environments, demonstrating that exploiting task-space structure through a network of task neighborhoods is an effective approach to multi-task optimization at scale. Our contributions are:

1.   1.
A unified formal taxonomy of MAP-Elites variants for multi-task optimization that exposes the structural modifications distinguishing each variant and motivates MONET (Section[2](https://arxiv.org/html/2604.21991#S2 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks")).

2.   2.
MONET, a graph-based multi-task optimizer that makes task-space neighborhoods an explicit element of recombination and combines localized social and individual learning (Section[3](https://arxiv.org/html/2604.21991#S3 "3 MONET ‣ Multi-Task Optimization over Networks of Tasks")).

3.   3.
A systematic hyperparameter study across four task domains (archery, arm, cartpole, and hexapod) analyzing the effects of neighborhood type, neighborhood size, and the social/individual balance (Sections[4](https://arxiv.org/html/2604.21991#S4 "4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks") and[5](https://arxiv.org/html/2604.21991#S5 "5 Results ‣ Multi-Task Optimization over Networks of Tasks")).

4.   4.
Empirical evidence that a single MONET configuration generalizes across all four domains and matches or outperforms MT-ME and PT-ME, showing task-space topology to be a valuable and underexploited source of information in multi-task optimization (Sections[5](https://arxiv.org/html/2604.21991#S5 "5 Results ‣ Multi-Task Optimization over Networks of Tasks") and[6](https://arxiv.org/html/2604.21991#S6 "6 Conclusion ‣ Multi-Task Optimization over Networks of Tasks")).

## 2 Related Works

Evolutionary multi-task optimization has become a highly active area of research within the computational intelligence community [[22](https://arxiv.org/html/2604.21991#bib.bib13 "Multifactorial evolution: toward evolutionary multitasking"), [23](https://arxiv.org/html/2604.21991#bib.bib27 "Half a dozen real-world applications of evolutionary multitasking, and more"), [53](https://arxiv.org/html/2604.21991#bib.bib64 "What makes evolutionary multi-task optimization better: A comprehensive survey"), [39](https://arxiv.org/html/2604.21991#bib.bib89 "Evolutionary multitask optimization: a methodological overview, challenges and future research directions")]. To concretize the description, we specify the problem setting as follows. Consider a set of n tasks, each described by a parameter vector \boldsymbol{\tau}_{i}\in\mathcal{T}, for which we want to find the corresponding solutions \boldsymbol{\theta}_{1},\ldots,\boldsymbol{\theta}_{n}\in\Theta simultaneously. The goal of multi-task optimization is to find, for every task \boldsymbol{\tau}\in\mathcal{T}, an optimal solution as:

\boldsymbol{\theta}^{*}_{\boldsymbol{\tau}}=\arg\max_{\boldsymbol{\theta}\in\Theta}f(\boldsymbol{\theta},\boldsymbol{\tau}),(1)

where \Theta is the solution space, \mathcal{T} the task parameter space and f:\Theta\times\mathcal{T}\rightarrow\mathbb{R} the fitness function.

Existing methodologies generally fall into two categories: _implicit transfer_, which utilizes a single, unified search space for all tasks, and _explicit transfer_, which operates across distinct solution spaces. Although implicit transfer is relatively straightforward to implement, its application is largely restricted to homogeneous environments. In contrast, explicit transfer accommodates heterogeneous tasks, though determining the optimal knowledge to exchange between distinct spaces remains a complex challenge. To facilitate this transfer, contemporary frameworks have leveraged probabilistic search distributions, search direction vectors, higher-order heuristics, and surrogate models. A representative example is the Multi-Factorial Evolutionary Algorithm II (MFEA-II)[[7](https://arxiv.org/html/2604.21991#bib.bib92 "Multifactorial evolutionary algorithm with online transfer parameter estimation: MFEA-II")], which learns an online inter-task transfer parameter to modulate the probabilistic crossover between tasks and thereby suppress negative transfer during multifactorial search.

Historically, the application of these methods has been limited to optimizing a small number of tasks concurrently (typically between two and ten). As catalogued in the evolutionary multi-tasking (EMT) survey of[[39](https://arxiv.org/html/2604.21991#bib.bib89 "Evolutionary multitask optimization: a methodological overview, challenges and future research directions")], examples include optimizing three robot morphologies via _neuroevolution_[[7](https://arxiv.org/html/2604.21991#bib.bib92 "Multifactorial evolutionary algorithm with online transfer parameter estimation: MFEA-II")], two symbolic regression datasets using genetic programming[[54](https://arxiv.org/html/2604.21991#bib.bib39 "Multifactorial genetic programming for symbolic regression problems")], ten numerical functions for software testing[[42](https://arxiv.org/html/2604.21991#bib.bib37 "Concurrently searching branches in software tests generation through multitask evolution")], multi-UAV path planning across three tasks[[6](https://arxiv.org/html/2604.21991#bib.bib40 "Cognizant multitasking in multiobjective multifactorial evolution: MO-MFEA-II")], power dispatch optimization for two bus systems[[30](https://arxiv.org/html/2604.21991#bib.bib30 "A multitasking electric power dispatch approach with multi-objective multifactorial optimization algorithm")], logistics planning for four package delivery problems[[21](https://arxiv.org/html/2604.21991#bib.bib33 "Explicit evolutionary multitasking for combinatorial optimization: a case study on capacitated vehicle routing problem")], and the design of three diode models[[29](https://arxiv.org/html/2604.21991#bib.bib32 "Evolutionary multi-task optimization for parameters extraction of photovoltaic models")]. Similar multi-task formulations exist in other fields, such as Bayesian optimization[[40](https://arxiv.org/html/2604.21991#bib.bib63 "Continuous multi-task bayesian optimisation with correlation")], which is frequently used to tune machine learning models across multiple datasets. However, because Bayesian approaches rely on Gaussian processes that scale cubically 1 1 1 In addition, the approaches such as Nyström approximation have limitations., they are restricted to computationally expensive fitness functions and struggle to scale beyond a thousand evaluations[[44](https://arxiv.org/html/2604.21991#bib.bib15 "Taking the human out of the loop: a review of bayesian optimization")]. To address these scalability limits, recent works in evolutionary many-task optimization have successfully tackled increasingly larger problem sets, scaling to 30, 50, 500, and up to 2,000 tasks simultaneously. Aligning with this trajectory toward large-scale optimization.

In order to show the connection of the proposed method to the existing literature, we first introduce a unified notation framework for the whole MAP-Elites algorithm family, enabling each multi-task variant to be presented as a modification of its underlying formulation 2 2 2 We adopt the maximization convention throughout; a minimization problem with cost c is handled by setting f=-c.. The same base is then extended once more in Section[3](https://arxiv.org/html/2604.21991#S3 "3 MONET ‣ Multi-Task Optimization over Networks of Tasks") when MONET is defined.

###### Definition 1(MAP-Elites)

Let \Theta denote a solution space, f:\Theta\to\mathbb{R} a fitness function to be _maximized_, \mathcal{B} a behavior space, and \mathbf{b}:\Theta\to\mathcal{B} a behavioral descriptor that maps a solution to its behavior. Fix a finite discretization of \mathcal{B} into cells. MAP-Elites[[14](https://arxiv.org/html/2604.21991#bib.bib55 "Robots that can adapt like animals"), [36](https://arxiv.org/html/2604.21991#bib.bib79 "Illuminating search spaces by mapping elites")] maintains an archive \mathcal{A}3 3 3 We use uppercase calligraphic \mathcal{A} because the archive is primarily a data structure that is _treated as_ a partial function; this convention is standard in the MAP-Elites literature and consistent with the use of uppercase for structured objects such as operators or transforms., treated as a partial function from cells to \Theta, initialized empty and updated by the following elitist rule: given a candidate \boldsymbol{\theta}\in\Theta, insert \boldsymbol{\theta} into the cell containing \mathbf{b}(\boldsymbol{\theta}) iff that cell is empty or f(\boldsymbol{\theta})>f(\mathcal{A}[\mathbf{b}(\boldsymbol{\theta})]). The run returns \mathcal{A}, which trades objective value against behavioral coverage rather than optimizing f alone.

Figure[2(a)](https://arxiv.org/html/2604.21991#S2.F2.sf1 "In Figure 2 ‣ 2 Related Works ‣ Multi-Task Optimization over Networks of Tasks") summarizes the mappings of Definition[1](https://arxiv.org/html/2604.21991#Thmdefinition1 "Definition 1(MAP-Elites) ‣ 2 Related Works ‣ Multi-Task Optimization over Networks of Tasks") as a commutative diagram.4 4 4 Note that Definition[1](https://arxiv.org/html/2604.21991#Thmdefinition1 "Definition 1(MAP-Elites) ‣ 2 Related Works ‣ Multi-Task Optimization over Networks of Tasks") uses a single-task fitness f:\Theta\to\mathbb{R}; the multi-task extensions below lift this to f:\Theta\times\mathcal{T}\to\mathbb{R} by conditioning on a task parameter \boldsymbol{\tau}\in\mathcal{T}, recovering the problem statement of Eq.([1](https://arxiv.org/html/2604.21991#S2.E1 "In 2 Related Works ‣ Multi-Task Optimization over Networks of Tasks")). Now we are ready to introduce relevant multi-task variants as extensions of this definition: each paragraph names exactly what is changed and leaves the rest of the definition in place.

![Image 2: Refer to caption](https://arxiv.org/html/2604.21991v1/x2.png)

(a)MAP-Elites (Definition[1](https://arxiv.org/html/2604.21991#Thmdefinition1 "Definition 1(MAP-Elites) ‣ 2 Related Works ‣ Multi-Task Optimization over Networks of Tasks")). Solid blue: behavioral descriptor \mathbf{b}:\Theta\to\mathcal{B}; solid red: fitness f:\Theta\to\mathbb{R}; dashed green: archive \mathcal{A}:\mathcal{B}\rightharpoonup\Theta and composed map f\circ\mathcal{A}:\mathcal{B}\to\mathbb{R}.

![Image 3: Refer to caption](https://arxiv.org/html/2604.21991v1/x3.png)

(b)MONET (Definition[2](https://arxiv.org/html/2604.21991#Thmdefinition2 "Definition 2(MONET) ‣ 3 MONET ‣ Multi-Task Optimization over Networks of Tasks")). The behavioral descriptor \mathbf{b} is absent; solid red: fitness f(\cdot,\boldsymbol{\tau}):\Theta\times\mathcal{T}\to\mathbb{R}; dashed green: archive \mathcal{A}:\mathcal{T}\to\Theta and f\circ\mathcal{A}:\mathcal{T}\to\mathbb{R}; solid blue: graph construction via task distance d; dashed blue: neighborhood lookup \mathrm{Nb}(\boldsymbol{\tau}).

Figure 2: Commutative diagrams comparing MAP-Elites and MONET.

Multi-Task MAP-Elites (MT-ME)[[37](https://arxiv.org/html/2604.21991#bib.bib57 "Quality diversity for multi-task optimization")]. MT-ME extends Definition[1](https://arxiv.org/html/2604.21991#Thmdefinition1 "Definition 1(MAP-Elites) ‣ 2 Related Works ‣ Multi-Task Optimization over Networks of Tasks") by _replacing_ the behavioral descriptor \mathbf{b}:\Theta\to\mathcal{B} with the task parameter itself. A finite set of tasks \{\boldsymbol{\tau}_{1},\ldots,\boldsymbol{\tau}_{n}\}\subset\mathcal{T} is fixed in advance, the archive becomes \mathcal{A}:\mathcal{T}\to\Theta indexed directly by task, and the fitness lifts to f:\Theta\times\mathcal{T}\to\mathbb{R}, matching the problem statement of Eq.([1](https://arxiv.org/html/2604.21991#S2.E1 "In 2 Related Works ‣ Multi-Task Optimization over Networks of Tasks")). Recombination draws parents globally from the entire task archive, so information is shared across all tasks regardless of their similarity. MT-ME uses the iso-line-dd variation operator[[47](https://arxiv.org/html/2604.21991#bib.bib46 "Discovering the elite hypervolume by leveraging interspecies correlation")], which exploits the observation that elites tend to cluster in a compact “elite hypervolume” in genotype space: given two parent elites \boldsymbol{\theta}_{i} and \boldsymbol{\theta}_{j}, a candidate is generated by adding isotropic Gaussian noise and a directional component along the line connecting the parents, thereby biasing search toward the region of genotype space where elites concentrate. Mouret and Maguire[[37](https://arxiv.org/html/2604.21991#bib.bib57 "Quality diversity for multi-task optimization")] evaluated MT-ME on arm and hexapod (Figure [3](https://arxiv.org/html/2604.21991#S4.F3 "Figure 3 ‣ 4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks")), comparing against CMA-ES run independently per task, MAP-Elites with random task assignment, and MAP-Elites evaluated on all tasks simultaneously. MT-ME outperformed all baselines on both domains. Multi-Task Multi-Behavior MAP-Elites (MTMB-ME)[[3](https://arxiv.org/html/2604.21991#bib.bib62 "Multi-task multi-behavior MAP-elites")]. MTMB-ME extends MT-ME by _restoring_ the behavioral descriptor from Definition[1](https://arxiv.org/html/2604.21991#Thmdefinition1 "Definition 1(MAP-Elites) ‣ 2 Related Works ‣ Multi-Task Optimization over Networks of Tasks") on top of the task index, yielding a two-level archive \mathcal{A}:\mathcal{T}\times\mathcal{B}\to\Theta indexed jointly by task and behavior. Task-specific adaptation and behavioral diversity are therefore maintained simultaneously within a single archive. Anne and Mouret[[3](https://arxiv.org/html/2604.21991#bib.bib62 "Multi-task multi-behavior MAP-elites")] applied MTMB-ME to a humanoid robot fault-recovery scenario, identifying successful contact positions on a wall that help the robot regain balance after a leg fault, and outperformed random search, grid search, and standard MAP-Elites both in the number of tasks solved and in the number of solutions found.

Parametric-Task MAP-Elites (PT-ME)[[4](https://arxiv.org/html/2604.21991#bib.bib14 "Parametric-Task MAP-Elites")]. PT-ME extends MT-ME to _continuous_ task spaces. Instead of fixing a finite set of tasks in advance, PT-ME draws a fresh task \boldsymbol{\tau}\sim\mathcal{T} at every iteration and stores, for each drawn task, the best solution encountered so far, so the archive becomes a dense dataset covering the entire task space. PT-ME alternates between two variation operators at equal probability. The first is SBX[[15](https://arxiv.org/html/2604.21991#bib.bib81 "Simulated binary crossover for continuous search space")] with a bandit-tuned tournament that biases task selection toward tasks close to the parents, replacing the iso-line-dd operator of MT-ME. The second, and more distinctive, operator is a _local linear regression_: PT-ME partitions the task space into cells via Centroidal Voronoi Tessellation (CVT)[[18](https://arxiv.org/html/2604.21991#bib.bib20 "Centroidal voronoi tessellations: Applications and algorithms"), [48](https://arxiv.org/html/2604.21991#bib.bib21 "Using centroidal voronoi tessellations to scale up the multi-dimensional archive of phenotypic elites algorithm")], precomputes a k-dimensional tree (k-d tree)[[10](https://arxiv.org/html/2604.21991#bib.bib109 "Multidimensional binary search trees used for associative searching")] for fast cell lookup and a Delaunay triangulation[[17](https://arxiv.org/html/2604.21991#bib.bib24 "Sur la sphere vide")] for adjacency, then fits a linear least-squares model M=(\mathbf{T}^{\top}\mathbf{T})^{-1}\mathbf{T}^{\top}\mathbf{X} to the elites in adjacent cells, where \mathbf{T} stacks the task vectors of the adjacent cells and \mathbf{X} the corresponding solution vectors; a candidate is then generated as \boldsymbol{\theta}=M\cdot\boldsymbol{\tau}+\sigma_{\mathrm{reg}}\cdot\mathcal{N}(0,\mathrm{var}(\mathbf{X})). This regression operator is notable because it explicitly exploits task-space locality: the Delaunay triangulation defines a geometric neighborhood over \mathcal{T}, and the linear model builds a local approximation of the task-to-solution mapping from that neighborhood. On archery adn arm PT-ME yields improved performance over standard MAP-Elites and Proximal Policy Optimization (PPO)[[2](https://arxiv.org/html/2604.21991#bib.bib72 "Proximal policy optimization algorithms")].

MAP-Elites with Knowledge Transfer (MMKT)[[31](https://arxiv.org/html/2604.21991#bib.bib87 "A preliminary study of multi-task map-elites with knowledge transfer for robotic arm design")]. This variant extends MT-ME by _partitioning_ the task set into groups \{\mathcal{T}_{1},\ldots,\mathcal{T}_{m}\} obtained by applying k-means clustering to the task parameters, and then promoting knowledge transfer between tasks through linear combinations of solutions drawn from the same group. In this preliminary study, MMKT outperforms MT-ME on the arm task.

Adaptive Group-based Collaborative Optimization (AGCO)[[26](https://arxiv.org/html/2604.21991#bib.bib44 "A Group-Based Many-Task Collaborative Optimization Framework for Evolutionary Robots Design")]. This framework extends the group-based view further with a two-stage transfer protocol. An _inter-group knowledge separation_ stage employs an adaptive selection mechanism for crossover operators to control information exchange between groups, and an _intra-group knowledge reunion_ stage leverages Thompson sampling to weight within-group recombination so that target tasks assimilate insights from multiple in-group sources according to their evolutionary preferences.

Most of the variants above treat the archive as fully connected: every elite can serve as a parent for any cell regardless of the relationship between the corresponding tasks. PT-ME is a partial exception: its regression operator builds a local linear model from Delaunay-adjacent cells, thereby exploiting geometric locality in \mathcal{T} for candidate generation. However, this locality is confined to one of two variation operators (applied with 50% probability), the adjacency structure is a by-product of the CVT tessellation rather than an explicit design element, and the SBX path still draws parents globally. An alternative that makes locality a first-class principle is familiar from _cellular evolutionary algorithms_ (cEAs)[[1](https://arxiv.org/html/2604.21991#bib.bib99 "Cellular genetic algorithms")], which organize the population into local neighborhoods and restrict recombination accordingly, so that information propagates through the population only via those neighborhoods. MONET (Section[3](https://arxiv.org/html/2604.21991#S3 "3 MONET ‣ Multi-Task Optimization over Networks of Tasks")) follows the same intuition but imposes the graph over tasks rather than over individuals: it extends MT-ME by replacing the implicit fully-connected archive \mathcal{A}:\mathcal{T}\to\Theta with an explicit graph structure over \mathcal{T}, inducing neighborhoods from task-space proximity.

## 3 MONET

Building on the taxonomy introduced in Section[2](https://arxiv.org/html/2604.21991#S2 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"), we now introduce MONET. Figure[1](https://arxiv.org/html/2604.21991#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Multi-Task Optimization over Networks of Tasks") illustrates the task-graph representation at the core of MONET: nodes are tasks, edges connect neighboring tasks in the task parameter space, and each node stores the best solution found so far.

###### Definition 2(MONET)

Let \{\boldsymbol{\tau}_{1},\ldots,\boldsymbol{\tau}_{n}\}\subset\mathcal{T} be a finite task set, f:\Theta\times\mathcal{T}\to\mathbb{R} a task-parameterized fitness function, and d:\mathcal{T}\times\mathcal{T}\to\mathbb{R}_{\geq 0} a task distance. MONET represents the optimization state as a labeled graph \mathcal{G}=(\mathcal{V},\mathcal{E},\mathcal{A}) whose nodes \mathcal{V}=\{\boldsymbol{\tau}_{1},\ldots,\boldsymbol{\tau}_{n}\} are the tasks themselves, whose edges \mathcal{E}=\{(\boldsymbol{\tau}_{i},\boldsymbol{\tau}_{j}):\boldsymbol{\tau}_{j}\text{ is among the }N\text{ nearest neighbors of }\boldsymbol{\tau}_{i}\text{ under }d\} encode task-space proximity for a user-chosen neighborhood size N, and whose labeling \mathcal{A}:\mathcal{V}\to\Theta stores the best solution found so far for each task. Candidate solutions are admitted by the elitist rule: a candidate \boldsymbol{\theta} generated for task \boldsymbol{\tau}_{i} replaces \mathcal{A}[\boldsymbol{\tau}_{i}] iff f(\boldsymbol{\theta},\boldsymbol{\tau}_{i})\geq f(\mathcal{A}[\boldsymbol{\tau}_{i}],\boldsymbol{\tau}_{i}).

Figure[2(b)](https://arxiv.org/html/2604.21991#S2.F2.sf2 "In Figure 2 ‣ 2 Related Works ‣ Multi-Task Optimization over Networks of Tasks") summarizes the mappings of Definition[2](https://arxiv.org/html/2604.21991#Thmdefinition2 "Definition 2(MONET) ‣ 3 MONET ‣ Multi-Task Optimization over Networks of Tasks"). Compared with the base MAP-Elites diagram (Figure[2(a)](https://arxiv.org/html/2604.21991#S2.F2.sf1 "In Figure 2 ‣ 2 Related Works ‣ Multi-Task Optimization over Networks of Tasks")), the behavioral descriptor \mathbf{b} is absent and the archive is indexed directly by task; the fitness is conditioned on a task parameter; and the bottom row shows the graph construction from the task distance d and the neighborhood lookup \mathrm{Nb}(\boldsymbol{\tau}) that mediates social learning.

Candidates are generated by one of two complementary mechanisms, selected at each step with probability p_{\text{ind}}. _Individual learning_ mutates the current label \mathcal{A}[\boldsymbol{\tau}_{i}] of a focal task, exploring the local fitness landscape of \boldsymbol{\tau}_{i} in isolation. _Social learning_ draws a neighbor \boldsymbol{\tau}_{j} from the graph neighborhood of \boldsymbol{\tau}_{i} and recombines \mathcal{A}[\boldsymbol{\tau}_{i}] with \mathcal{A}[\boldsymbol{\tau}_{j}], transferring information between similar tasks along the edges of \mathcal{G}. Empty nodes are initialized by individual learning with a uniform random solution, or by social learning by copying from a non-empty neighbor (with a uniform random fallback if no such neighbor exists); social learning is skipped at a step where the chosen neighbor is itself empty.

In this work, we use the Euclidean distance in \mathcal{T} as the task distance d, and set the neighborhood size N (the number of nearest neighbors retained per task, Def.[2](https://arxiv.org/html/2604.21991#Thmdefinition2 "Definition 2(MONET) ‣ 3 MONET ‣ Multi-Task Optimization over Networks of Tasks")) to a small fraction of |\mathcal{V}| (see Section[4](https://arxiv.org/html/2604.21991#S4 "4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks")), so that \mathcal{G} is a sparse, partially connected graph whose density is controlled by the chosen neighborhood size. Algorithm[1](https://arxiv.org/html/2604.21991#alg1 "Algorithm 1 ‣ 3 MONET ‣ Multi-Task Optimization over Networks of Tasks") provides the pseudocode of MONET and the code is publicly accessible[[24](https://arxiv.org/html/2604.21991#bib.bib106 "Repository for MONET: Multi-Task Optimization over Networks of Tasks")].

Algorithm 1 MONET

1:Task set

T=\{\boldsymbol{\tau}_{1},\dots,\boldsymbol{\tau}_{n}\}
, fitness function

f
, task distance

d

2:Budget

E
, init fraction

K_{\text{init}}
, neighborhood size

N
, selection strategy Select(

\cdot
), probability

p_{\text{ind}}

3:Archive

\mathcal{A}
mapping each task to its best solution

4:

\mathcal{A}[\boldsymbol{\tau}]\leftarrow\emptyset
for all

\boldsymbol{\tau}\in T
\triangleright empty network

5:for each

\boldsymbol{\tau}_{i}\in T
do\triangleright build k-NN task graph

6:

\mathrm{Nb}(\boldsymbol{\tau}_{i})\leftarrow N
nearest tasks under

d

7:end for

8:for

i\leftarrow 1
to

K_{\text{init}}
do\triangleright random initialization

9:

\boldsymbol{\tau}\leftarrow\textsc{RandomTask}(T)

10:

\boldsymbol{\theta}\sim\mathcal{U}(p_{\min},p_{\max})

11:

f_{x}\leftarrow f(\boldsymbol{\theta},\boldsymbol{\tau})

12:if

\mathcal{A}[\boldsymbol{\tau}]=\emptyset
or

f_{x}\geq\mathcal{A}[\boldsymbol{\tau}].\text{fitness}
then\triangleright elitist update

13:

\mathcal{A}[\boldsymbol{\tau}]\leftarrow(\boldsymbol{\theta},f_{x})

14:end if

15:end for

16:

e\leftarrow K_{\text{init}}
\triangleright main loop

17:while

e<E
do

18:

\boldsymbol{\tau}\leftarrow\textsc{RandomTask}(T)
\triangleright sample focal task

19:if

\mathrm{rand}()<p_{\text{ind}}
then\triangleright individual learning

20:Mutation(

\boldsymbol{\tau},\mathcal{A}
) \triangleright IndividualLearning(\boldsymbol{\tau},\mathcal{A})

21:

e\leftarrow e+1

22:else\triangleright social learning

23:

\boldsymbol{\tau}_{j}\leftarrow\textsc{Select}(\mathrm{Nb}(\boldsymbol{\tau}))
\triangleright pick one neighbor

24:if

\boldsymbol{\tau}_{j}\neq\text{nil}
then\triangleright skip if no graph neighbor exists

25:SBX(

\boldsymbol{\tau},\boldsymbol{\tau}_{j},\mathcal{A}
) \triangleright SocialLearning(\boldsymbol{\tau},\boldsymbol{\tau}_{j},\mathcal{A})

26:

e\leftarrow e+1

27:end if

28:end if

29:end while

30:return

\mathcal{A}

## 4 Experimental Setup

We compare MONET against PT-ME and MT-ME across four task environments: archery, arm, cartpole, and hexapod (Figure [3](https://arxiv.org/html/2604.21991#S4.F3 "Figure 3 ‣ 4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks")).

PT-ME samples tasks continuously from \mathcal{T} rather than operating over a fixed task set: it partitions \mathcal{T} into n cells via Centroidal Voronoi Tessellation (CVT)[[18](https://arxiv.org/html/2604.21991#bib.bib20 "Centroidal voronoi tessellations: Applications and algorithms"), [48](https://arxiv.org/html/2604.21991#bib.bib21 "Using centroidal voronoi tessellations to scale up the multi-dimensional archive of phenotypic elites algorithm")] and, by the end of its run, each cell holds the highest-fitness task-solution pair it has sampled. To place the three algorithms on the same footing, we take these n cell elites (n=5000 for archery, arm, and cartpole, and n=2000 for hexapod) as the fixed task set on which MT-ME and MONET are then run. This removes task selection as a confound in the cross-algorithm comparison. One asymmetry remains: under this protocol each PT-ME task is evaluated essentially once (the sample whose solution became its cell’s elite), whereas MT-ME and MONET revisit each task many times to refine its solution, so the total evaluation budget is identical across algorithms but the per-task refinement opportunity is not.

For PT-ME and MT-ME we use the default parameters from [[4](https://arxiv.org/html/2604.21991#bib.bib14 "Parametric-Task MAP-Elites"), [37](https://arxiv.org/html/2604.21991#bib.bib57 "Quality diversity for multi-task optimization")]. For MONET, we first conduct a hyperparameter sensitivity analysis. We consider the following hyperparameters and ranges: p_{\text{ind}}\in\{0,\,0.3,\,0.5,\,0.7,\,1\}, \text{strategy}\in\{\text{random},\,\text{best\_fitness}\}, \text{neighborhood}\in\{\text{random},\,\text{closest},\,\text{distance\_proportional}\}, and \text{neighbor\_percentage}\in\{1\%,\,5\%,\,10\%\}. Here p_{\text{ind}} is the per-step probability of performing individual learning (mutation) rather than social learning (SBX); strategy is the rule used to pick the neighbor during social learning, either uniformly at random or deterministically the neighbor with the highest current fitness; neighborhood determines how the neighbor set of each task is built, taking the N closest tasks in \mathcal{T}, N uniformly random tasks, or N tasks drawn without replacement with probability proportional to task-space similarity; and neighbor_percentage sets this neighborhood size as a fraction N=\lceil\text{neighbor\_percentage}\cdot|\mathcal{V}|\rceil of the total task set.

For each hyperparameter combination we perform up to 20 independent runs with different random seeds and report the mean fitness with standard deviation as well as the area under the learning curve (AUC), capturing both final performance and cumulative learning efficiency. The sensitivity sweep is concentrated on archery and arm (20 seeds per configuration; cartpole and hexapod are evaluated on a smaller subset of configurations because of simulator cost.

To quantify the importance of individual hyperparameters and their interactions, both per task and aggregated across all tasks, we fit a gradient-boosted tree surrogate model and compute SHapley Additive exPlanations (SHAP) values[[33](https://arxiv.org/html/2604.21991#bib.bib93 "A unified approach to interpreting model predictions"), [32](https://arxiv.org/html/2604.21991#bib.bib94 "From local explanations to global understanding with explainable AI for trees")], following the surrogate-based hyperparameter importance methodology of[[27](https://arxiv.org/html/2604.21991#bib.bib95 "An efficient approach for assessing hyperparameter importance")]. Guided by this analysis, we then assign reasonable values to each hyperparameter, yielding a single configuration of MONET that we refer to as the _default_ configuration and compare it against PT-ME and MT-ME. Additionally, we employ the Sequential Parameter Optimization Toolbox (SPOT)[[9](https://arxiv.org/html/2604.21991#bib.bib105 "Optimization with spotoptim")], a surrogate-based optimizer, to automatically tune MONET’s hyperparameters per task and include these task-specific configurations in the comparison.

To assess statistical significance of the performance differences between algorithms, we apply the Mann-Whitney U test[[34](https://arxiv.org/html/2604.21991#bib.bib96 "On a test of whether one of two random variables is stochastically larger than the other")] to the final fitness values and AUC values across all seeds and report the resulting p-values with Holm-Bonferroni correction[[25](https://arxiv.org/html/2604.21991#bib.bib98 "A simple sequentially rejective multiple test procedure"), [19](https://arxiv.org/html/2604.21991#bib.bib97 "Multiple comparisons among means")] for multiple comparisons.

![Image 4: Refer to caption](https://arxiv.org/html/2604.21991v1/x4.png)

Figure 3: Overview of benchmark environments.

Higher fitness is better in all four benchmarks, but the scale differs across them: archery and arm return values in [0,1]; cartpole reports the mean balanced-step count across 10 rollouts and therefore lies in [0,t_{\max}] with t_{\max}=1000; hexapod reports the forward centre-of-mass displacement in metres over a 3 s episode with no formal upper bound (in practice below 1.5\,\mathrm{m}). Closed-form fitness expressions are defined in the Supplementary Information (SI)[[24](https://arxiv.org/html/2604.21991#bib.bib106 "Repository for MONET: Multi-Task Optimization over Networks of Tasks")]. All four benchmarks are run for 10^{6} evaluations and 20 seeds per run.

#### Archery.

A projectile-aiming task in which the archer compensates for target distance and wind (Figure[3](https://arxiv.org/html/2604.21991#S4.F3 "Figure 3 ‣ 4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks")a)[[4](https://arxiv.org/html/2604.21991#bib.bib14 "Parametric-Task MAP-Elites")]. Tasks (2D) vary the distance D\in[5,40] m and horizontal wind W\in[-10,10]\,\mathrm{m/s^{2}}; solutions (2D) are yaw and pitch in [-\pi/12,\pi/12] at fixed launch speed v_{0}=70\,\mathrm{m/s}.

#### Arm.

A planar 10-degree-of-freedom (10-DoF) arm places its end-effector at a fixed target (Figure[3](https://arxiv.org/html/2604.21991#S4.F3 "Figure 3 ‣ 4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks")b)[[37](https://arxiv.org/html/2604.21991#bib.bib57 "Quality diversity for multi-task optimization"), [4](https://arxiv.org/html/2604.21991#bib.bib14 "Parametric-Task MAP-Elites")]. Tasks (2D) vary the link length L and maximum joint angle \alpha_{\max}, both in [0,1]; solutions (10D) are the 10 joint angles.

#### Cartpole.

Cart-pole balancing under varying physical parameters (Figure[3](https://arxiv.org/html/2604.21991#S4.F3 "Figure 3 ‣ 4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks")c)[[35](https://arxiv.org/html/2604.21991#bib.bib8 "BOXES: an experiment in adaptive control"), [8](https://arxiv.org/html/2604.21991#bib.bib9 "Neuronlike adaptive elements that can solve difficult learning control problems")]. Tasks (2D) vary pole mass m_{p}\in[0.05,0.5] kg and pole length L\in[0.3,1.2] m; solutions (16D) are a direct encoding of the weights of a small multi-layer perceptron controller (4 inputs, 2 hidden tanh units, 2 output logits, with bias units on the input and hidden layers).

#### Hexapod.

A simulated hexapod walks forward across 2 000 random morphologies (Figure[3](https://arxiv.org/html/2604.21991#S4.F3 "Figure 3 ‣ 4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks")d)[[14](https://arxiv.org/html/2604.21991#bib.bib55 "Robots that can adapt like animals"), [37](https://arxiv.org/html/2604.21991#bib.bib57 "Quality diversity for multi-task optimization")], simulated in PyBullet[[13](https://arxiv.org/html/2604.21991#bib.bib3 "PyBullet, a Python module for physics simulation for games, robotics and machine learning")]. Tasks (12D) are 12 leg-segment length changes; solutions (36D) parameterise a gait controller.

## 5 Results

Across the four domains, MONET matches or exceeds the strongest MAP-Elites baseline on final mean fitness, with the largest gap on the hexapod task where the per-domain tuned configuration nearly doubles the final fitness of MT-ME (Figure[4](https://arxiv.org/html/2604.21991#S5.F4 "Figure 4 ‣ Hexapod. ‣ 5.1 Algorithm Comparison ‣ 5 Results ‣ Multi-Task Optimization over Networks of Tasks"), Table[1](https://arxiv.org/html/2604.21991#S5.T1 "Table 1 ‣ 5.2 Hyperparameter Sensitivity and Default Configuration ‣ 5 Results ‣ Multi-Task Optimization over Networks of Tasks")). We first walk through the per-domain comparison, then unpack the hyperparameter analysis that produced the shared default configuration.

### 5.1 Algorithm Comparison

We report two MONET variants: _MONET default_, a single configuration shared across domains (defined in Section[5.2](https://arxiv.org/html/2604.21991#S5.SS2 "5.2 Hyperparameter Sensitivity and Default Configuration ‣ 5 Results ‣ Multi-Task Optimization over Networks of Tasks")), and _MONET tuned_, a per-domain tuning produced with SPOT [[9](https://arxiv.org/html/2604.21991#bib.bib105 "Optimization with spotoptim")]. Both are compared against PT-ME and MT-ME with their published defaults. Each box in Figure[4](https://arxiv.org/html/2604.21991#S5.F4 "Figure 4 ‣ Hexapod. ‣ 5.1 Algorithm Comparison ‣ 5 Results ‣ Multi-Task Optimization over Networks of Tasks") aggregates 20 seeds per (algorithm, domain) pair; brackets indicate pairwise comparisons where one algorithm is significantly better (Mann-Whitney U, Holm-Bonferroni correction, p<0.05).

#### Archery.

MT-ME, PT-ME, and MONET tuned all saturate at 1.000; MONET default stops short at 0.996\pm 0.000 (p<0.001 vs. all three). Final fitness is thus tied at the top, and AUC is the discriminating signal. MONET default reaches an AUC of 976.3\pm 0.6, significantly above MT-ME (957.2\pm 3.3, p<0.001) and PT-ME (924.1\pm 1.1, p<0.001). MONET tuned improves AUC further to 979.2\pm 1.9 (p<0.001 vs. MONET default). MONET therefore converges to near-optimal performance substantially earlier than the baselines.

#### Arm.

MONET default reaches 0.770\pm 0.001 on final fitness, just below MT-ME (0.772\pm 0.002, p<0.001) and above PT-ME (0.747\pm 0.001, p<0.001). On AUC it is the strongest of the four algorithms: 762.7\pm 1.2, ahead of MT-ME (760.1\pm 3.4, p=0.003) and PT-ME (734.0\pm 2.9, p<0.001). This reflects faster convergence in the early phase (Figure[4](https://arxiv.org/html/2604.21991#S5.F4 "Figure 4 ‣ Hexapod. ‣ 5.1 Algorithm Comparison ‣ 5 Results ‣ Multi-Task Optimization over Networks of Tasks")b). MONET tuned achieves the highest final fitness at 0.773\pm 0.000, significantly above MT-ME (p<0.001).

#### Cartpole.

MONET default reaches 997.2\pm 2.6, statistically indistinguishable from MT-ME (994.5\pm 8.5) and PT-ME (992.5\pm 8.9). MT-ME retains a significant AUC lead (978.4\pm 11.4 vs. 969.0\pm 7.2, p<0.001): it converges earlier while MONET terminates higher. Tuning nudges final fitness to 998.5\pm 1.0 (not significant vs. baselines) but significantly lowers AUC to 941.4\pm 17.4 (p<0.001 vs. MONET default).

#### Hexapod.

MONET default (0.690\pm 0.101) performs significantly better than MT-ME (0.530\pm 0.032, p<0.001) and statistically indistinguishable from PT-ME (0.727\pm 0.078, p=0.15). Tuning improves performance further to 0.956\pm 0.082, significantly above both baselines (p<0.001 vs. MT-ME; p<0.001 vs. PT-ME).

Previous work reported that PT-ME outperformed MT-ME[[4](https://arxiv.org/html/2604.21991#bib.bib14 "Parametric-Task MAP-Elites")], whereas in our experiments, MT-ME achieved better performance than PT-ME. This discrepancy arises because PT-ME samples tasks continuously from the task space, which can result in selecting tasks that are easier to solve. To ensure a fair comparison, we fixed the task set to the final tasks discovered by PT-ME and evaluated both methods on this same set of tasks.

![Image 5: Refer to caption](https://arxiv.org/html/2604.21991v1/x5.png)

Figure 4: Algorithm comparison across the four benchmark domains.

### 5.2 Hyperparameter Sensitivity and Default Configuration

The most important finding from the SHAP-based sensitivity analysis is that the probability of individual learning p_{\text{ind}} accounts for most of the variation in performance, with a mean absolute SHAP value \overline{|\text{SHAP}|}=0.49, followed by neighbor_percentage (0.16), strategy (0.13), and neighborhood (0.09); see Figure[5](https://arxiv.org/html/2604.21991#S5.F5 "Figure 5 ‣ 5.2 Hyperparameter Sensitivity and Default Configuration ‣ 5 Results ‣ Multi-Task Optimization over Networks of Tasks"). Low values of p_{\text{ind}} (favoring social learning over individual mutation) consistently improve both mean fitness and AUC across all domains, supporting the central design intuition behind MONET that information transfer between similar tasks is the dominant lever. The remaining hyperparameters show weaker and more domain-dependent effects (see SI for detailed SHAP analysis per domain[[24](https://arxiv.org/html/2604.21991#bib.bib106 "Repository for MONET: Multi-Task Optimization over Networks of Tasks")]).

The default parameters of MONET were defined based on the SHAP analysis as: strategy = best_fitness, neighborhood = closest, p_{\text{ind}}=0.3, and neighbor_percentage =5\% (obtained by ranking configurations on the z-scored mean fitness across all domains, see SI[[24](https://arxiv.org/html/2604.21991#bib.bib106 "Repository for MONET: Multi-Task Optimization over Networks of Tasks")]). This configuration picks the fittest neighbor for crossover, draws neighbors from the k nearest tasks in task space, and mixes mostly social learning (70\%) with occasional individual refinement (30\%). The SPOT-tuned per-domain configurations differ markedly from the default and from each other: archery uses a large neighborhood (39\%) with moderate p_{\text{ind}}=0.22; arm uses a very small neighborhood (0.1\%) with high p_{\text{ind}}=0.54 (favouring individual refinement on this well-conditioned task); cartpole uses a small neighborhood (1\%) with a balanced p_{\text{ind}}=0.5; and hexapod uses a small neighborhood (1\%) with pure social learning (p_{\text{ind}}=0). This spread confirms that the optimal individual vs. social learning balance is task-dependent. The default configuration is competitive with the baselines and achieves the best AUC on three of four domains (archery, arm, hexapod); it is outperformed only on archery final fitness, where PT-ME and MT-ME saturate at 1.000, and on cartpole AUC, where MT-ME converges faster (Fig.[4](https://arxiv.org/html/2604.21991#S5.F4 "Figure 4 ‣ Hexapod. ‣ 5.1 Algorithm Comparison ‣ 5 Results ‣ Multi-Task Optimization over Networks of Tasks"), Table[1](https://arxiv.org/html/2604.21991#S5.T1 "Table 1 ‣ 5.2 Hyperparameter Sensitivity and Default Configuration ‣ 5 Results ‣ Multi-Task Optimization over Networks of Tasks"))

![Image 6: Refer to caption](https://arxiv.org/html/2604.21991v1/x6.png)

Figure 5: Global hyperparameter importance for MONET: mean absolute SHAP value on a gradient-boosted tree surrogate.

Table 1: Final mean fitness and AUC (\times 10^{3} on the [0,1] domains) per domain. Mean \pm std over 20 random seeds per (algorithm, domain) pair.

## 6 Conclusion

In this work, we propose MONET to find high-quality solutions for a large set of related tasks simultaneously and in parallel. We propose structuring the task space as a network to establish topological relations and neighborhoods, which aims to exploit task similarities for knowledge transfer. As shown by the results in four domains, this approach has an advantage over flat archive representation, where these topological relations are ignored. We introduced two operators, namely individual and social learning, that aim to improve tasks in isolation or exploit knowledge transfer across neighboring related tasks.

The extensive sensitivity analysis of the MONET algorithm and its parameters showed that the probability of individual learning p_{\text{ind}} influences the performance the most, with high values leading to lower performance. This shows that social learning is a significant contributor to performance by exploiting knowledge transfer across tasks. However, this analysis was on a pre-selected set of p_{\text{ind}} parameters. Thus, using a more balanced ratio between individual and social learning parameters can improve performance, particularly in task spaces where similarities between tasks are relatively low. In fact, SPOT-based per-domain tuning selected low p_{\text{ind}} on hexapod (0) and archery (0.54). On arm (0.5) and cartpole 0.5, a more balanced mix was preferred.

We establish connectivity between tasks on the basis of the task parameter similarities. This relies on the idea that tasks sharing similar parameters are themselves similar, making them more likely to benefit from knowledge transfer and to converge toward similar solutions. However, this assumption may not hold in many domains. In the hexapod domain, for example, small changes to front-right and back-left leg lengths yield nearby task vectors but can demand qualitatively different solutions to perform the control tasks. Future work should therefore explore alternative connectivity definitions and assess their effect on knowledge transfer.

Currently, the MONET algorithm has four hyperparameters. Another interesting direction is online self-adaptive parameter control (e.g., adapting the neighborhood size as in MT-ME’s task-competition and replacement rule[[37](https://arxiv.org/html/2604.21991#bib.bib57 "Quality diversity for multi-task optimization")], or dynamically switching between individual and social learning strategies[[50](https://arxiv.org/html/2604.21991#bib.bib47 "Meta-control of social learning strategies")]). Although this can introduce complexity, it may provide more generalized performance across different domains and task spaces. Another promising direction is to extend the types of individual and social learning strategies that can improve performance. Furthermore, some of these strategies can converge to more general solutions that can lead to satisfactory performance for a wide range of tasks in the task space [[45](https://arxiv.org/html/2604.21991#bib.bib48 "Evolving generalist controllers to handle a wide range of morphological variations")].

## References

*   [1]E. Alba and B. Dorronsoro (2008)Cellular genetic algorithms. Operations Research/Computer Science Interfaces Series, Vol. 42, Springer US, Boston, MA. External Links: [Document](https://dx.doi.org/10.1007/978-0-387-77610-1)Cited by: [§2](https://arxiv.org/html/2604.21991#S2.p10.3 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [2]J. S. and Filip Wolski and Prafulla Dhariwal and Alec Radford and Oleg Klimov (2017)Proximal policy optimization algorithms. CoRR abs/1707.06347. Cited by: [§2](https://arxiv.org/html/2604.21991#S2.p7.8 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [3]T. Anne and J. Mouret (2023)Multi-task multi-behavior MAP-elites. In Proceedings of the Companion Conference on Genetic and Evolutionary Computation, GECCO ’23 Companion, New York, NY, USA,  pp.111–114. External Links: [Document](https://dx.doi.org/10.1145/3583133.3590730), ISBN 979-8-4007-0120-7 Cited by: [§2](https://arxiv.org/html/2604.21991#S2.p6.7 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"), [§2](https://arxiv.org/html/2604.21991#S2.p6.7.3 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [4]T. Anne and J. Mouret (2024-07)Parametric-Task MAP-Elites. In Proceedings of the Genetic and Evolutionary Computation Conference,  pp.68–77. External Links: 2402.01275, [Document](https://dx.doi.org/10.1145/3638529.3653993)Cited by: [Table 7](https://arxiv.org/html/2604.21991#Pt0.A7.T7.4.5.4.2 "In Appendix 0.G Code Availability ‣ Multi-Task Optimization over Networks of Tasks"), [§1](https://arxiv.org/html/2604.21991#S1.p2.1 "1 Introduction ‣ Multi-Task Optimization over Networks of Tasks"), [§2](https://arxiv.org/html/2604.21991#S2.p7.8.1 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"), [§4](https://arxiv.org/html/2604.21991#S4.SS0.SSS0.Px1.p1.4 "Archery. ‣ 4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks"), [§4](https://arxiv.org/html/2604.21991#S4.SS0.SSS0.Px2.p1.3 "Arm. ‣ 4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks"), [§4](https://arxiv.org/html/2604.21991#S4.p3.13 "4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks"), [§5.1](https://arxiv.org/html/2604.21991#S5.SS1.SSS0.Px4.p2.1 "Hexapod. ‣ 5.1 Algorithm Comparison ‣ 5 Results ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [5]T. Anne and J. Mouret (2024)Parametric-Task MAP-Elites. Note: [https://github.com/hucebot/Parametric-Task_MAP-Elites](https://github.com/hucebot/Parametric-Task_MAP-Elites)Cited by: [Table 7](https://arxiv.org/html/2604.21991#Pt0.A7.T7.4.5.4.3 "In Appendix 0.G Code Availability ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [6]K. K. Bali, A. Gupta, Y. Ong, and P. S. Tan (2021)Cognizant multitasking in multiobjective multifactorial evolution: MO-MFEA-II. IEEE Transactions on Cybernetics 51 (4),  pp.1784–1796. External Links: [Document](https://dx.doi.org/10.1109/TCYB.2020.2981733)Cited by: [§2](https://arxiv.org/html/2604.21991#S2.p3.1 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [7]K. K. Bali, Y. Ong, A. Gupta, and P. S. Tan (2020)Multifactorial evolutionary algorithm with online transfer parameter estimation: MFEA-II. IEEE Transactions on Evolutionary Computation 24 (1),  pp.69–83. External Links: [Document](https://dx.doi.org/10.1109/TEVC.2019.2906927)Cited by: [§2](https://arxiv.org/html/2604.21991#S2.p2.1 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"), [§2](https://arxiv.org/html/2604.21991#S2.p3.1 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [8]A. G. Barto, R. S. Sutton, and C. W. Anderson (1983)Neuronlike adaptive elements that can solve difficult learning control problems. IEEE Transactions on Systems, Man, and Cybernetics SMC-13 (5),  pp.834–846. External Links: [Document](https://dx.doi.org/10.1109/TSMC.1983.6313077)Cited by: [§4](https://arxiv.org/html/2604.21991#S4.SS0.SSS0.Px3.p1.2 "Cartpole. ‣ 4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [9]T. Bartz-Beielstein (2026)Optimization with spotoptim. External Links: 2604.13672, [Link](https://arxiv.org/abs/2604.13672)Cited by: [§4](https://arxiv.org/html/2604.21991#S4.p5.1 "4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks"), [§5.1](https://arxiv.org/html/2604.21991#S5.SS1.p1.3 "5.1 Algorithm Comparison ‣ 5 Results ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [10]J. L. Bentley (1975)Multidimensional binary search trees used for associative searching. Communications of the ACM 18 (9),  pp.509–517. External Links: [Document](https://dx.doi.org/10.1145/361002.361007)Cited by: [§2](https://arxiv.org/html/2604.21991#S2.p7.8 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [11]J. Bongard, V. Zykov, and H. Lipson (2006)Resilient machines through continuous self-modeling. Science 314 (5802),  pp.1118–1121. External Links: [Document](https://dx.doi.org/10.1126/science.1133687)Cited by: [§1](https://arxiv.org/html/2604.21991#S1.p1.1 "1 Introduction ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [12]K. Chatzilygeroudis, A. Cully, V. Vassiliades, and J. Mouret (2021)Quality-diversity optimization: a novel branch of stochastic optimization. In Black Box Optimization, Machine Learning, and No-Free Lunch Theorems, Springer Optimization and Its Applications,  pp.109–135. External Links: [Document](https://dx.doi.org/10.1007/978-3-030-66515-9%5F4)Cited by: [§1](https://arxiv.org/html/2604.21991#S1.p1.1 "1 Introduction ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [13]E. Coumans and Y. Bai (2021)PyBullet, a Python module for physics simulation for games, robotics and machine learning. Note: [http://pybullet.org](http://pybullet.org/)Cited by: [§4](https://arxiv.org/html/2604.21991#S4.SS0.SSS0.Px4.p1.1 "Hexapod. ‣ 4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [14]A. Cully, J. Clune, D. Tarapore, and J. Mouret (2015/05/01)Robots that can adapt like animals. Nature 521 (7553),  pp.503–507. External Links: [Document](https://dx.doi.org/10.1038/nature14422), ISBN 1476-4687, [Link](https://doi.org/10.1038/nature14422)Cited by: [Table 7](https://arxiv.org/html/2604.21991#Pt0.A7.T7.4.3.2.2 "In Appendix 0.G Code Availability ‣ Multi-Task Optimization over Networks of Tasks"), [§1](https://arxiv.org/html/2604.21991#S1.p1.1 "1 Introduction ‣ Multi-Task Optimization over Networks of Tasks"), [§1](https://arxiv.org/html/2604.21991#S1.p2.1 "1 Introduction ‣ Multi-Task Optimization over Networks of Tasks"), [§4](https://arxiv.org/html/2604.21991#S4.SS0.SSS0.Px4.p1.1 "Hexapod. ‣ 4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks"), [Definition 1](https://arxiv.org/html/2604.21991#Thmdefinition1.p1.13 "Definition 1(MAP-Elites) ‣ 2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [15]K. Deb and R. B. Agrawal (1995)Simulated binary crossover for continuous search space. Complex Systems 9 (2),  pp.115–148. Cited by: [Appendix 0.B](https://arxiv.org/html/2604.21991#Pt0.A2.SS0.SSS0.Px2.p1.1 "Social Learning. ‣ Appendix 0.B MONET Algorithm ‣ Multi-Task Optimization over Networks of Tasks"), [§1](https://arxiv.org/html/2604.21991#S1.p3.4 "1 Introduction ‣ Multi-Task Optimization over Networks of Tasks"), [§2](https://arxiv.org/html/2604.21991#S2.p7.8 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [16]K. Deb (2001)Multi-objective optimization using evolutionary algorithms. Wiley, Chichester, UK. External Links: ISBN 978-0-471-87339-6 Cited by: [Appendix 0.B](https://arxiv.org/html/2604.21991#Pt0.A2.SS0.SSS0.Px2.p1.1 "Social Learning. ‣ Appendix 0.B MONET Algorithm ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [17]B. Delaunay (1928)Sur la sphere vide. In Proceedings of the Mathematics, Toronto, Toronto,  pp.695–700. Cited by: [§1](https://arxiv.org/html/2604.21991#S1.p3.4 "1 Introduction ‣ Multi-Task Optimization over Networks of Tasks"), [§2](https://arxiv.org/html/2604.21991#S2.p7.8 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [18]Q. Du, V. Faber, and M. Gunzburger (1999)Centroidal voronoi tessellations: Applications and algorithms. SIAM Review 41 (4),  pp.637–676. External Links: https://doi.org/10.1137/S0036144599352836, [Document](https://dx.doi.org/10.1137/S0036144599352836)Cited by: [§2](https://arxiv.org/html/2604.21991#S2.p7.8 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"), [§4](https://arxiv.org/html/2604.21991#S4.p2.6 "4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [19]O. J. Dunn (1961)Multiple comparisons among means. Journal of the American Statistical Association 56 (293),  pp.52–64. External Links: [Document](https://dx.doi.org/10.1080/01621459.1961.10482090)Cited by: [§4](https://arxiv.org/html/2604.21991#S4.p6.1 "4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [20]W. Feller (1968)An introduction to probability theory and its applications. 3 edition, Vol. 1, Wiley. Cited by: [Appendix 0.A](https://arxiv.org/html/2604.21991#Pt0.A1.p2.2 "Appendix 0.A Node Coverage Guarantees Under Random Sampling ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [21]L. Feng, Y. Huang, L. Zhou, J. Zhong, A. Gupta, K. Tang, and K. C. Tan (2021)Explicit evolutionary multitasking for combinatorial optimization: a case study on capacitated vehicle routing problem. IEEE Transactions on Cybernetics 51 (6),  pp.3143–3156. External Links: [Document](https://dx.doi.org/10.1109/TCYB.2019.2962865)Cited by: [§2](https://arxiv.org/html/2604.21991#S2.p3.1 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [22]A. Gupta, Y. Ong, and L. Feng (2016)Multifactorial evolution: toward evolutionary multitasking. IEEE Transactions on Evolutionary Computation 20 (3),  pp.343–357. External Links: [Document](https://dx.doi.org/10.1109/TEVC.2015.2458037)Cited by: [§1](https://arxiv.org/html/2604.21991#S1.p1.1 "1 Introduction ‣ Multi-Task Optimization over Networks of Tasks"), [§2](https://arxiv.org/html/2604.21991#S2.p1.4 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [23]A. Gupta, L. Zhou, Y. Ong, Z. Chen, and Y. Hou (2022-05)Half a dozen real-world applications of evolutionary multitasking, and more. IEEE Computational Intelligence Magazine 17,  pp.49–66. External Links: [Document](https://dx.doi.org/10.1109/MCI.2022.3155332)Cited by: [§2](https://arxiv.org/html/2604.21991#S2.p1.4 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [24]J. Hatzky, T. Bartz-Beielstein, A. E. Eiben, and A. Yaman (2026)Repository for MONET: Multi-Task Optimization over Networks of Tasks. Note: [https://github.com/ju2ez/MONET/tree/PPSN-MONET-introduction](https://github.com/ju2ez/MONET/tree/PPSN-MONET-introduction)Cited by: [Table 7](https://arxiv.org/html/2604.21991#Pt0.A7.T7.4.2.1.3 "In Appendix 0.G Code Availability ‣ Multi-Task Optimization over Networks of Tasks"), [§3](https://arxiv.org/html/2604.21991#S3.p4.5 "3 MONET ‣ Multi-Task Optimization over Networks of Tasks"), [§4](https://arxiv.org/html/2604.21991#S4.p7.5 "4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks"), [§5.2](https://arxiv.org/html/2604.21991#S5.SS2.p1.9 "5.2 Hyperparameter Sensitivity and Default Configuration ‣ 5 Results ‣ Multi-Task Optimization over Networks of Tasks"), [§5.2](https://arxiv.org/html/2604.21991#S5.SS2.p2.16 "5.2 Hyperparameter Sensitivity and Default Configuration ‣ 5 Results ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [25]S. Holm (1979)A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics 6 (2),  pp.65–70. Cited by: [§4](https://arxiv.org/html/2604.21991#S4.p6.1 "4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [26]Y. Hou, Z. Yu, Z. Guo, W. Pei, Y. Wu, H. Ge, B. Xue, and M. Zhang (2025-05)A Group-Based Many-Task Collaborative Optimization Framework for Evolutionary Robots Design. IEEE Transactions on Systems, Man, and Cybernetics: Systems 55 (5),  pp.3492–3505. External Links: ISSN 2168-2216, 2168-2232, [Document](https://dx.doi.org/10.1109/TSMC.2025.3541002)Cited by: [§2](https://arxiv.org/html/2604.21991#S2.p9.1.1 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [27]F. Hutter, H. Hoos, and K. Leyton-Brown (2014)An efficient approach for assessing hyperparameter importance. In Proceedings of the 31st International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 32,  pp.754–762. Cited by: [§4](https://arxiv.org/html/2604.21991#S4.p5.1 "4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [28]B. Huynh Thi Thanh, L. Van Cuong, T. B. Thang, and N. H. Long (2023-12)Ensemble Multifactorial Evolution With Biased Skill-Factor Inheritance for Many-Task Optimization. IEEE Transactions on Evolutionary Computation 27 (6),  pp.1735–1749. External Links: ISSN 1089-778X, 1089-778X, 1941-0026, [Document](https://dx.doi.org/10.1109/TEVC.2022.3227120)Cited by: [§1](https://arxiv.org/html/2604.21991#S1.p2.1 "1 Introduction ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [29]J. Liang, K. Qiao, M. Yuan, K. Yu, B. Qu, S. Ge, Y. Li, and G. Chen (2020-03)Evolutionary multi-task optimization for parameters extraction of photovoltaic models. Energy Conversion and Management 207,  pp.112509. External Links: [Document](https://dx.doi.org/10.1016/j.enconman.2020.112509)Cited by: [§2](https://arxiv.org/html/2604.21991#S2.p3.1 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [30]J. Liu, P. Li, G. Wang, Y. Zha, J. Peng, and G. Xu (2020)A multitasking electric power dispatch approach with multi-objective multifactorial optimization algorithm. IEEE access : practical innovations, open solutions 8,  pp.155902–155911. External Links: [Document](https://dx.doi.org/10.1109/ACCESS.2020.3018484)Cited by: [§2](https://arxiv.org/html/2604.21991#S2.p3.1 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [31]P. Liu, Z. Guo, H. Yu, H. Linghu, Y. Li, Y. Hou, H. Ge, and Q. Zhang (2022)A preliminary study of multi-task map-elites with knowledge transfer for robotic arm design. In 2022 IEEE Congress on Evolutionary Computation (CEC), Vol. ,  pp.1–8. External Links: [Document](https://dx.doi.org/10.1109/CEC55065.2022.9870374)Cited by: [§2](https://arxiv.org/html/2604.21991#S2.p8.2.1 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [32]S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, and S. Lee (2020)From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence 2 (1),  pp.56–67. External Links: [Document](https://dx.doi.org/10.1038/s42256-019-0138-9)Cited by: [§4](https://arxiv.org/html/2604.21991#S4.p5.1 "4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [33]S. M. Lundberg and S. Lee (2017)A unified approach to interpreting model predictions. In Advances in Neural Information Processing Systems, Vol. 30. Cited by: [§4](https://arxiv.org/html/2604.21991#S4.p5.1 "4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [34]H. B. Mann and D. R. Whitney (1947)On a test of whether one of two random variables is stochastically larger than the other. The Annals of Mathematical Statistics 18 (1),  pp.50–60. External Links: [Document](https://dx.doi.org/10.1214/aoms/1177730491)Cited by: [§4](https://arxiv.org/html/2604.21991#S4.p6.1 "4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [35]D. Michie and R. A. Chambers (1968)BOXES: an experiment in adaptive control. In Machine Intelligence 2, E. Dale and D. Michie (Eds.),  pp.137–152. Cited by: [§4](https://arxiv.org/html/2604.21991#S4.SS0.SSS0.Px3.p1.2 "Cartpole. ‣ 4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [36]J. Mouret and J. Clune (2015)Illuminating search spaces by mapping elites. CoRR abs/1504.04909. External Links: [Link](http://arxiv.org/abs/1504.04909), 1504.04909 Cited by: [Table 7](https://arxiv.org/html/2604.21991#Pt0.A7.T7.4.3.2.2 "In Appendix 0.G Code Availability ‣ Multi-Task Optimization over Networks of Tasks"), [§1](https://arxiv.org/html/2604.21991#S1.p1.1 "1 Introduction ‣ Multi-Task Optimization over Networks of Tasks"), [§1](https://arxiv.org/html/2604.21991#S1.p2.1 "1 Introduction ‣ Multi-Task Optimization over Networks of Tasks"), [Definition 1](https://arxiv.org/html/2604.21991#Thmdefinition1.p1.13 "Definition 1(MAP-Elites) ‣ 2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [37]J. Mouret and G. Maguire (2020-06)Quality diversity for multi-task optimization. In Proceedings of the 2020 Genetic and Evolutionary Computation Conference, Cancún Mexico,  pp.121–129. External Links: [Document](https://dx.doi.org/10.1145/3377930.3390203), ISBN 978-1-4503-7128-5 Cited by: [Table 7](https://arxiv.org/html/2604.21991#Pt0.A7.T7.4.4.3.2 "In Appendix 0.G Code Availability ‣ Multi-Task Optimization over Networks of Tasks"), [§1](https://arxiv.org/html/2604.21991#S1.p2.1 "1 Introduction ‣ Multi-Task Optimization over Networks of Tasks"), [§2](https://arxiv.org/html/2604.21991#S2.p6.7 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"), [§2](https://arxiv.org/html/2604.21991#S2.p6.7.1 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"), [§4](https://arxiv.org/html/2604.21991#S4.SS0.SSS0.Px2.p1.3 "Arm. ‣ 4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks"), [§4](https://arxiv.org/html/2604.21991#S4.SS0.SSS0.Px4.p1.1 "Hexapod. ‣ 4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks"), [§4](https://arxiv.org/html/2604.21991#S4.p3.13 "4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks"), [§6](https://arxiv.org/html/2604.21991#S6.p4.1 "6 Conclusion ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [38]J. Mouret pymap_elites: Python reference implementation of MAP-Elites and Multi-Task MAP-Elites. Note: [https://github.com/resibots/pymap_elites](https://github.com/resibots/pymap_elites)Cited by: [Table 7](https://arxiv.org/html/2604.21991#Pt0.A7.T7.4.3.2.3 "In Appendix 0.G Code Availability ‣ Multi-Task Optimization over Networks of Tasks"), [Table 7](https://arxiv.org/html/2604.21991#Pt0.A7.T7.4.4.3.3 "In Appendix 0.G Code Availability ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [39]E. Osaba, A. D. Martinez, and J. D. Ser (2021)Evolutionary multitask optimization: a methodological overview, challenges and future research directions. External Links: 2102.02558, [Link](https://arxiv.org/abs/2102.02558)Cited by: [§2](https://arxiv.org/html/2604.21991#S2.p1.4 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"), [§2](https://arxiv.org/html/2604.21991#S2.p3.1 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [40]M. Pearce and J. Branke (2018)Continuous multi-task bayesian optimisation with correlation. European Journal of Operational Research 270 (3),  pp.1074–1085. External Links: ISSN 0377-2217, [Document](https://dx.doi.org/10.1016/j.ejor.2018.03.017)Cited by: [§2](https://arxiv.org/html/2604.21991#S2.p3.1 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [41]L. Rendell, R. Boyd, D. Cownden, M. Enquist, K. Eriksson, M. W. Feldman, L. Fogarty, S. Ghirlanda, T. Lillicrap, and K. N. Laland (2010)Why copy others? Insights from the social learning strategies tournament. Science 328 (5975),  pp.208–213. External Links: [Document](https://dx.doi.org/10.1126/science.1184719)Cited by: [§1](https://arxiv.org/html/2604.21991#S1.p4.4 "1 Introduction ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [42]R. Sagarna and Y. Ong (2016)Concurrently searching branches in software tests generation through multitask evolution. In 2016 IEEE Symposium Series on Computational Intelligence (SSCI),  pp.1–8. External Links: [Document](https://dx.doi.org/10.1109/SSCI.2016.7850040)Cited by: [§2](https://arxiv.org/html/2604.21991#S2.p3.1 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [43]S. M. B. P. Samarakoon, M. A. V. J. Muthugala, and M. R. Elara (2022)Metaheuristic based navigation of a reconfigurable robot through narrow spaces with shape changing ability. Expert Systems with Applications 201,  pp.117060. External Links: [Document](https://dx.doi.org/10.1016/j.eswa.2022.117060)Cited by: [§1](https://arxiv.org/html/2604.21991#S1.p1.1 "1 Introduction ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [44]B. Shahriari, K. Swersky, Z. Wang, R. P. Adams, and N. de Freitas (2016)Taking the human out of the loop: a review of bayesian optimization. Proceedings of the IEEE 104 (1),  pp.148–175. External Links: [Document](https://dx.doi.org/10.1109/JPROC.2015.2494218)Cited by: [§2](https://arxiv.org/html/2604.21991#S2.p3.1 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [45]C. Triebold and A. Yaman (2024)Evolving generalist controllers to handle a wide range of morphological variations. In Proceedings of the Genetic and Evolutionary Computation Conference,  pp.1137–1145. Cited by: [§6](https://arxiv.org/html/2604.21991#S6.p4.1 "6 Conclusion ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [46]F. van Diggelen, N. Cambier, E. Ferrante, and A. E. Eiben (2024)A model-free method to learn multiple skills in parallel on modular robots. Nature Communications 15,  pp.6267. External Links: [Document](https://dx.doi.org/10.1038/s41467-024-50131-4)Cited by: [§1](https://arxiv.org/html/2604.21991#S1.p1.1 "1 Introduction ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [47]V. Vassiliades and J. Mouret (2018)Discovering the elite hypervolume by leveraging interspecies correlation. In Proceedings of the Genetic and Evolutionary Computation Conference, Gecco ’18, New York, NY, USA,  pp.149–156. External Links: [Document](https://dx.doi.org/10.1145/3205455.3205602), ISBN 978-1-4503-5618-3 Cited by: [§2](https://arxiv.org/html/2604.21991#S2.p6.7 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [48]V. Vassiliades, K. Chatzilygeroudis, and J. Mouret (2017-08)Using centroidal voronoi tessellations to scale up the multi-dimensional archive of phenotypic elites algorithm. IEEE Transactions on Evolutionary Computation PP,  pp.1–1. External Links: [Document](https://dx.doi.org/10.1109/TEVC.2017.2735550)Cited by: [§2](https://arxiv.org/html/2604.21991#S2.p7.8 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"), [§4](https://arxiv.org/html/2604.21991#S4.p2.6 "4 Experimental Setup ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [49]C. Wang, J. Liu, K. Wu, and Z. Wu (2022-04)Solving Multitask Optimization Problems With Adaptive Knowledge Transfer via Anomaly Detection. IEEE Transactions on Evolutionary Computation 26 (2),  pp.304–318. External Links: ISSN 1089-778X, 1089-778X, 1941-0026, [Document](https://dx.doi.org/10.1109/TEVC.2021.3068157)Cited by: [§1](https://arxiv.org/html/2604.21991#S1.p2.1 "1 Introduction ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [50]A. Yaman, N. Bredeche, O. Çaylak, J. Z. Leibo, and S. W. Lee (2022)Meta-control of social learning strategies. PLoS computational biology 18 (2),  pp.e1009882. Cited by: [§6](https://arxiv.org/html/2604.21991#S6.p4.1 "6 Conclusion ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [51]A. Yaman and G. Iacca (2021)Distributed embodied evolution over networks. Applied Soft Computing 101,  pp.106993. Cited by: [§1](https://arxiv.org/html/2604.21991#S1.p4.4 "1 Introduction ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [52]M. Yim, W. Shen, B. Salemi, D. Rus, M. Moll, H. Lipson, E. Klavins, and G. S. Chirikjian (2007)Modular self-reconfigurable robot systems. IEEE Robotics & Automation Magazine 14 (1),  pp.43–52. External Links: [Document](https://dx.doi.org/10.1109/MRA.2007.339623)Cited by: [§1](https://arxiv.org/html/2604.21991#S1.p1.1 "1 Introduction ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [53]H. Zhao, X. Ning, X. Liu, C. Wang, and J. Liu (2023)What makes evolutionary multi-task optimization better: A comprehensive survey. Applied Soft Computing 145,  pp.110545. External Links: ISSN 1568-4946, [Document](https://dx.doi.org/10.1016/j.asoc.2023.110545)Cited by: [§2](https://arxiv.org/html/2604.21991#S2.p1.4 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"). 
*   [54]J. Zhong, L. Feng, W. Cai, and Y. Ong (2020)Multifactorial genetic programming for symbolic regression problems. IEEE Transactions on Systems, Man, and Cybernetics: Systems 50 (11),  pp.4492–4505. External Links: [Document](https://dx.doi.org/10.1109/TSMC.2018.2853719)Cited by: [§2](https://arxiv.org/html/2604.21991#S2.p3.1 "2 Related Works ‣ Multi-Task Optimization over Networks of Tasks"). 

## Appendix 0.A Node Coverage Guarantees Under Random Sampling

If we draw at each evaluation step a random node and perform an action, we may ask how many evaluations are required so that we visit each of the \nu nodes at least once with high certainty.

This problem can be framed as a _coupon collector’s problem_[[20](https://arxiv.org/html/2604.21991#bib.bib2 "An introduction to probability theory and its applications")]. Writing M for the number of draws, the expected number of steps needed to visit all \nu nodes at least once is

\mathbb{E}[M]=\nu H_{\nu}=\nu\left(\ln\nu+\gamma+O\!\left(\frac{1}{\nu}\right)\right),

where H_{\nu} is the \nu-th harmonic number and \gamma\approx 0.57721 is the Euler-Mascheroni constant.

Beyond the expectation, one may also ask for a high-probability guarantee. Using the asymptotic limit for the coupon collector’s completion time, the probability that all nodes have been visited by draw

M=\nu(\ln\nu+c)

converges to

\mathbb{P}\bigl(M\leq\nu(\ln\nu+c)\bigr)\approx e^{-e^{-c}}.

To achieve at least 99\% probability of having visited all nodes, we solve for c

\displaystyle e^{-e^{-c}}\displaystyle=p
\displaystyle\ln\!\left(e^{-e^{-c}}\right)\displaystyle=\ln(p)
\displaystyle-e^{-c}\displaystyle=\ln(p)
\displaystyle\text{For }p=99:\quad c\displaystyle=-\ln\!\bigl(-\ln(99)\bigr)\approx 60015\approx\ln(00).

Since each node is selected independently with probability \tfrac{1}{\nu} at every draw, the visit count of any particular node over M draws follows a Binomial distribution,

X_{i}\sim\mathrm{Binomial}\!\left(M,\tfrac{1}{\nu}\right),

from which the expected number of visits per node is

\mathbb{E}[X_{i}]=\frac{M}{\nu}.

We evaluate four environments with differing numbers of distinct tasks and a fixed amount of draws (number of evaluations) M=1{,}000{,}000:

Table 2: Coupon collector bounds for the four benchmark environments with M=1{,}000{,}000 total evaluations.

## Appendix 0.B MONET Algorithm

Algorithm[2](https://arxiv.org/html/2604.21991#alg2 "Algorithm 2 ‣ Individual Learning. ‣ Appendix 0.B MONET Algorithm ‣ Multi-Task Optimization over Networks of Tasks") and Algorithm[3](https://arxiv.org/html/2604.21991#alg3 "Algorithm 3 ‣ Social Learning. ‣ Appendix 0.B MONET Algorithm ‣ Multi-Task Optimization over Networks of Tasks") detail the two learning subroutines invoked in Algorithm[1](https://arxiv.org/html/2604.21991#alg1 "Algorithm 1 ‣ 3 MONET ‣ Multi-Task Optimization over Networks of Tasks").

#### Individual Learning.

When individual learning is selected, the current elite at node \boldsymbol{\tau}_{i} is mutated independently of any neighbor. If no solution exists at the node, a random solution is drawn from \mathcal{U}(p_{\min},p_{\max}) and evaluated. Otherwise, Gaussian mutation with standard deviation \sigma_{m} scaled by the parameter range is applied. The offspring replaces the current elite only if its fitness is at least as high.

Algorithm 2 Individual Learning

1:Task

\boldsymbol{\tau}
, archive

\mathcal{A}
, bounds

p_{\min},p_{\max}
, mutation std

\sigma_{m}

2:Updated

\mathcal{A}[\boldsymbol{\tau}]

3:if

\mathcal{A}[\boldsymbol{\tau}]=\emptyset
then\triangleright initialize empty node

4:

\boldsymbol{\theta}\sim\mathcal{U}(p_{\min},p_{\max})
\triangleright random solution

5:

\mathcal{A}[\boldsymbol{\tau}]\leftarrow(\boldsymbol{\theta},\,f(\boldsymbol{\theta},\boldsymbol{\tau}))

6:return

7:end if

8:

\boldsymbol{\theta}\leftarrow\mathcal{A}[\boldsymbol{\tau}].\text{solution}
\triangleright current elite

9:

\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\,\sigma_{m}(p_{\max}-p_{\min}))
\triangleright Gaussian perturbation

10:

\mathbf{y}\leftarrow\text{clip}(\boldsymbol{\theta}+\boldsymbol{\epsilon},\;p_{\min},\;p_{\max})
\triangleright mutant offspring

11:

f_{y}\leftarrow f(\mathbf{y},\boldsymbol{\tau})

12:if

f_{y}\geq\mathcal{A}[\boldsymbol{\tau}].\text{fitness}
then\triangleright elitist replacement

13:

\mathcal{A}[\boldsymbol{\tau}]\leftarrow(\mathbf{y},f_{y})

14:end if

#### Social Learning.

When social learning is selected, the current elite is recombined with a solution drawn from a neighboring node. If the current node is empty, the neighbor’s solution is copied as an initial seed (or a random solution is used when no neighbor has a solution). Recombination is performed via Simulated Binary Crossover (SBX)[[15](https://arxiv.org/html/2604.21991#bib.bib81 "Simulated binary crossover for continuous search space")] with distribution index \eta_{c}=10[[16](https://arxiv.org/html/2604.21991#bib.bib1 "Multi-objective optimization using evolutionary algorithms")]. As with individual learning, the offspring replaces the elite only upon improvement.

Algorithm 3 Social Learning

1:Task

\boldsymbol{\tau}
, neighbor task

\boldsymbol{\tau}_{j}
, archive

\mathcal{A}
, bounds

p_{\min},p_{\max}

2:Updated

\mathcal{A}[\boldsymbol{\tau}]

3:if

\mathcal{A}[\boldsymbol{\tau}]=\emptyset
then\triangleright initialize empty node

4:if

\mathcal{A}[\boldsymbol{\tau}_{j}]\neq\emptyset
then

5:

\boldsymbol{\theta}\leftarrow\mathcal{A}[\boldsymbol{\tau}_{j}].\text{solution}
\triangleright copy from neighbor

6:else

7:

\boldsymbol{\theta}\sim\mathcal{U}(p_{\min},p_{\max})
\triangleright random fallback

8:end if

9:

\mathcal{A}[\boldsymbol{\tau}]\leftarrow(\boldsymbol{\theta},\,f(\boldsymbol{\theta},\boldsymbol{\tau}))

10:return

11:end if

12:if

\mathcal{A}[\boldsymbol{\tau}_{j}]=\emptyset
then

13:return\triangleright no valid neighbor solution

14:end if

15:

\mathbf{y}\leftarrow\text{SBX}\bigl(\mathcal{A}[\boldsymbol{\tau}].\text{solution},\;\mathcal{A}[\boldsymbol{\tau}_{j}].\text{solution},\;\eta_{c}\!=\!10\bigr)
\triangleright crossover

16:

\mathbf{y}\leftarrow\text{clip}(\mathbf{y},\,p_{\min},\,p_{\max})

17:

f_{y}\leftarrow f(\mathbf{y},\boldsymbol{\tau})

18:if

f_{y}\geq\mathcal{A}[\boldsymbol{\tau}].\text{fitness}
then\triangleright elitist replacement

19:

\mathcal{A}[\boldsymbol{\tau}]\leftarrow(\mathbf{y},f_{y})

20:end if

## Appendix 0.C Closed-Form Task Fitness Definitions

This appendix gives the explicit fitness expressions for the four benchmarks.

### 0.C.1 Archery

With initial velocity vector

\mathbf{v}=v_{0}\begin{bmatrix}-\sin(\text{yaw})\\
\cos(\text{yaw})\cos(\text{pitch})\\
\cos(\text{yaw})\sin(\text{pitch})\end{bmatrix},\quad v_{0}=70\,\mathrm{m/s},

the target-plane hit time is t_{\text{hit}}=D/v_{y} whenever v_{y}>0; if v_{y}\leq 0 the shot misses and f=0. Taking g=9.8\,\mathrm{m/s^{2}} and wind W\in[-10,10]\,\mathrm{m/s^{2}} along x, the miss coordinates on the target plane (y=D) are

m_{x}=\tfrac{1}{2}\,W\,t_{\text{hit}}^{2}+v_{x}\,t_{\text{hit}},\qquad m_{z}=-\tfrac{1}{2}\,g\,t_{\text{hit}}^{2}+v_{z}\,t_{\text{hit}},\qquad r=\sqrt{m_{x}^{2}+m_{z}^{2}}.

Fitness is computed on a 10-ring target of ring width 0.061\,\mathrm{m}:

f=\frac{\max\!\bigl(0,\;10-\lfloor r/0.061\rfloor\bigr)}{10}\in\{0,0.1,\ldots,1.0\}.

### 0.C.2 Arm

With end-effector position \mathbf{p}_{\text{ee}}\in\mathbb{R}^{2} and a fixed target \mathbf{p}_{\text{tar}}=(0.5,0.5),

f=\exp\!\left(-\lVert\mathbf{p}_{\text{ee}}-\mathbf{p}_{\text{tar}}\rVert_{2}\right)\in(0,1].

The joint commands are \mathbf{q}=(\boldsymbol{\theta}-\tfrac{1}{2})\cdot 2\pi\,\alpha_{\max}/10 where \boldsymbol{\theta}\in[0,1]^{10} is the solution vector and \alpha_{\max} is the task-dependent maximum joint-angle fraction; each of the 10 links has length L/10 with L the task-dependent total arm length. \mathbf{p}_{\text{ee}} is the end-effector position returned by forward kinematics.

### 0.C.3 Cartpole

The task uses the CartPole-v1 dynamics from OpenAI Gym with the pole mass m_{p} and pole length L replaced by the per-task values; the default cart mass m_{c} is retained. With pole angle \phi, cart position x_{c}, control force F\in\{-10,+10\}\,\mathrm{N}, and gravity g=9.8\,\mathrm{m/s^{2}},

\ddot{\phi}=\frac{g\sin\phi+\cos\phi\left(-F-m_{p}L\dot{\phi}^{2}\sin\phi\right)/(m_{c}+m_{p})}{L\left(\tfrac{4}{3}-\tfrac{m_{p}\cos^{2}\phi}{m_{c}+m_{p}}\right)},\qquad\ddot{x}_{c}=\frac{F+m_{p}L\left(\dot{\phi}^{2}\sin\phi-\ddot{\phi}\cos\phi\right)}{m_{c}+m_{p}}.

A solution is evaluated over n_{\text{rollouts}}=10 rollouts with distinct Gym seeds. Each rollout runs until the pole leaves the balanced region (|\phi|\geq 12^{\circ} or |x_{c}|\geq 2.4\,\mathrm{m}) or until the step cap t_{\max}=1000 is reached. Writing t_{\text{balanced}}^{(k)} for the number of surviving steps on rollout k, the reported fitness is the arithmetic mean

f=\frac{1}{n_{\text{rollouts}}}\sum_{k=1}^{n_{\text{rollouts}}}\min\!\bigl(t_{\max},\,t_{\text{balanced}}^{(k)}\bigr)\in[0,t_{\max}].

### 0.C.4 Hexapod

With gait parameters \boldsymbol{\theta}\in[0,1]^{36} and morphology \boldsymbol{\tau}\in\mathcal{T} instantiated through a per-task URDF, a PyBullet simulation is run for up to 3\,\mathrm{s} of wall-clock simulator time. The episode terminates early if any body-axis angle (roll, pitch, yaw) exceeds \pi/8 or if the lateral deviation satisfies |y|>0.5\,\mathrm{m}. Writing p_{x}(t) for the forward position of the center of mass and T\leq 3\,\mathrm{s} for the time of termination or episode end,

f(\boldsymbol{\theta},\boldsymbol{\tau})=p_{x}(T),

reported in metres. The controller returns f=0 if any entry of \boldsymbol{\theta} lies outside [0,1].

## Appendix 0.D Detailed Sensitivity Analysis

This appendix provides per-domain SHAP beeswarm plots and the SPOT-tuned hyperparameter configurations that complement the global sensitivity analysis.

### 0.D.1 Per-Domain SHAP Analysis

Figures[6](https://arxiv.org/html/2604.21991#Pt0.A4.F6 "Figure 6 ‣ 0.D.1 Per-Domain SHAP Analysis ‣ Appendix 0.D Detailed Sensitivity Analysis ‣ Multi-Task Optimization over Networks of Tasks") and[7](https://arxiv.org/html/2604.21991#Pt0.A4.F7 "Figure 7 ‣ 0.D.1 Per-Domain SHAP Analysis ‣ Appendix 0.D Detailed Sensitivity Analysis ‣ Multi-Task Optimization over Networks of Tasks") show the SHAP beeswarm plots for the archery and arm domains, respectively. Each point represents one MONET run; its horizontal position indicates the SHAP value (contribution to the predicted metric), and its color encodes the feature value (red = high, blue = low).

In both domains, p_{\text{ind}} exhibits the strongest effect: high values of p_{\text{ind}} (more individual learning) consistently decrease predicted fitness, corroborating the global finding that social learning is the primary driver of performance. The remaining parameters show smaller, more task-specific effects.

![Image 7: Refer to caption](https://arxiv.org/html/2604.21991v1/x7.png)

Figure 6: SHAP beeswarm plot for the archery domain. Each row is a hyperparameter; each dot is a MONET run. Color indicates the feature value (red = high, blue = low); horizontal position indicates the SHAP value.

![Image 8: Refer to caption](https://arxiv.org/html/2604.21991v1/x8.png)

Figure 7: SHAP beeswarm plot for the arm domain. Panels as in Figure[6](https://arxiv.org/html/2604.21991#Pt0.A4.F6 "Figure 6 ‣ 0.D.1 Per-Domain SHAP Analysis ‣ Appendix 0.D Detailed Sensitivity Analysis ‣ Multi-Task Optimization over Networks of Tasks").

![Image 9: Refer to caption](https://arxiv.org/html/2604.21991v1/x9.png)

Figure 8: SHAP beeswarm plot for the cartpole domain. Panels as in Figure[6](https://arxiv.org/html/2604.21991#Pt0.A4.F6 "Figure 6 ‣ 0.D.1 Per-Domain SHAP Analysis ‣ Appendix 0.D Detailed Sensitivity Analysis ‣ Multi-Task Optimization over Networks of Tasks").

![Image 10: Refer to caption](https://arxiv.org/html/2604.21991v1/x10.png)

Figure 9: SHAP beeswarm plot for the hexapod domain. Panels as in Figure[6](https://arxiv.org/html/2604.21991#Pt0.A4.F6 "Figure 6 ‣ 0.D.1 Per-Domain SHAP Analysis ‣ Appendix 0.D Detailed Sensitivity Analysis ‣ Multi-Task Optimization over Networks of Tasks").

### 0.D.2 SPOT-Tuned Configurations

Table[3](https://arxiv.org/html/2604.21991#Pt0.A4.T3 "Table 3 ‣ 0.D.2 SPOT-Tuned Configurations ‣ Appendix 0.D Detailed Sensitivity Analysis ‣ Multi-Task Optimization over Networks of Tasks") lists the per-domain configurations found by the SPOT hyperparameter tuner.

Table 3: SPOT-tuned MONET configurations per domain.

## Appendix 0.E MONET Hyperparameter Descriptions

Table[4](https://arxiv.org/html/2604.21991#Pt0.A5.T4 "Table 4 ‣ Appendix 0.E MONET Hyperparameter Descriptions ‣ Multi-Task Optimization over Networks of Tasks") summarizes the four hyperparameters varied in the sensitivity analysis. All other algorithmic parameters (mutation operator, crossover operator, initialization strategy) are held fixed throughout. See Algorithm[1](https://arxiv.org/html/2604.21991#alg1 "Algorithm 1 ‣ 3 MONET ‣ Multi-Task Optimization over Networks of Tasks") for the points at which each parameter enters the loop.

Table 4: MONET hyperparameters and their roles.

## Appendix 0.F Hyperparameter Configuration Ranking

Table[5](https://arxiv.org/html/2604.21991#Pt0.A6.T5 "Table 5 ‣ Appendix 0.F Hyperparameter Configuration Ranking ‣ Multi-Task Optimization over Networks of Tasks") lists the ten highest-ranked MONET hyperparameter configurations (across the 2–4 domains on which they were run) by combined z-score, computed as the mean of the within-domain z-scores of final mean fitness and AUC. Table[6](https://arxiv.org/html/2604.21991#Pt0.A6.T6 "Table 6 ‣ Appendix 0.F Hyperparameter Configuration Ranking ‣ Multi-Task Optimization over Networks of Tasks") shows the five worst-ranked configurations for comparison. Only configurations evaluated on at least two domains are included in the ranking. The counts n_{\text{Arm}},n_{\text{Archery}},n_{\text{Cartpole}},n_{\text{Hexapod}} report the number of deduplicated seed runs per domain.

Table 5: Top 10 MONET configurations by combined z-score (n_{\text{domains}}\geq 2).

Table 6: Bottom 5 MONET configurations by combined z-score (n_{\text{domains}}\geq 2).

Two patterns are clear from Table[5](https://arxiv.org/html/2604.21991#Pt0.A6.T5 "Table 5 ‣ Appendix 0.F Hyperparameter Configuration Ranking ‣ Multi-Task Optimization over Networks of Tasks"): (i)the closest neighborhood dominates the top of the ranking, and (ii)small neighborhood sizes (\leq 2\% of the task set) are preferred. Poorly-performing configurations (Table[6](https://arxiv.org/html/2604.21991#Pt0.A6.T6 "Table 6 ‣ Appendix 0.F Hyperparameter Configuration Ranking ‣ Multi-Task Optimization over Networks of Tasks")) combine the distance_proportional neighborhood with either very high p_{\text{ind}} (near pure individual learning) or very large neighborhoods, both of which disrupt local knowledge transfer.

## Appendix 0.G Code Availability

Table[7](https://arxiv.org/html/2604.21991#Pt0.A7.T7 "Table 7 ‣ Appendix 0.G Code Availability ‣ Multi-Task Optimization over Networks of Tasks") lists the reference papers and public code repositories for MONET and the three baselines.

Table 7: Reference papers and public code repositories for MONET and the baseline algorithms.

