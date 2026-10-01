Title: Hierarchical Hybrid Neural Architecture Search using Adaptive Mutation Policies

URL Source: https://arxiv.org/html/2508.14946

Markdown Content:
Anurag Tripathi 1, Ajeet Kumar Singh 1, Rajsabi Surya 1, Aum Gupta 2, Sahiinii Lemaina Veikho 2, Dorien Herremans 3, Sudhir Bisane 1

{anurag.tripathi, ajeetkumar.singh, rajsabi.surya, sudhirb}@infoorigin.com 1 , sahiinii.linguistics@gmail.com 2, dorien_herremans@sutd.edu.sg 3

###### Abstract

Neural Architecture Search (NAS) has garnered significant research interest due to its capability to discover architectures superior to manually designed ones. Learning text representation is crucial for text classification and other language-related tasks. The NAS model used in text classification does not have a Hybrid hierarchical structure, and there is no restriction on the architecture structure, due to which the search space becomes very large and mostly redundant, so the existing RL models are not able to navigate the search space effectively. Also, doing a flat architecture search leads to an unorganised search space, which is difficult to traverse. For this purpose, we propose HHNAS-AM (Hierarchical Hybrid Neural Architecture Search with Adaptive Mutation Policies), a novel approach that efficiently explores diverse architectural configurations. We introduce a few architectural templates to search on which organise the search spaces, where search spaces are designed on the basis of domain-specific cues. Our method employs mutation strategies that dynamically adapt based on performance feedback from previous iterations using Q-learning, enabling a more effective and accelerated traversal of the search space.The proposed model is fully probabilistic, enabling effective exploration of the search space. We evaluate our approach on the database id (db_id) prediction task, where it consistently discovers high-performing architectures across multiple experiments. On the Spider dataset, our method achieves an 8% improvement in test accuracy over existing baselines.

Introduction
------------

NAS has emerged as a powerful approach to automating neural network design, outperforming manually crafted architectures in domains such as computer vision, reinforcement learning (Zoph and Le [2017](https://arxiv.org/html/2508.14946v1#bib.bib20))(Zoph et al. [2018](https://arxiv.org/html/2508.14946v1#bib.bib22)), and natural language processing (NLP). Despite this success, most NAS efforts have focused predominantly on vision tasks, often overlooking the structural and semantic characteristics unique to text data (Liu, Simonyan, and Yang [2018](https://arxiv.org/html/2508.14946v1#bib.bib5)). In text classification, where representation learning is crucial, this oversight limits the effectiveness of conventional NAS methods.

![Image 1: Refer to caption](https://arxiv.org/html/2508.14946v1/pipeline_new.png)

Figure 1: Proposed HHNAS-AM Pipeline.

Recent efforts in NAS for NLP have introduced tailored search spaces by integrating attention mechanisms and variable kernel sizes to better suit textual data (Wang et al. [2020](https://arxiv.org/html/2508.14946v1#bib.bib12)). Nevertheless, two key limitations persist. First, most approaches rely on flat architecture searches (Maziarz et al. [2019](https://arxiv.org/html/2508.14946v1#bib.bib6)), neglecting hierarchical or hybrid structures that are essential for capturing multi-scale dependencies in text. Second, the vast and unconstrained search spaces often result in a combinatorial explosion of candidate architectures, many of which are redundant or suboptimal, thus impeding search efficiency (Elsken, Metzen, and Hutter [2019](https://arxiv.org/html/2508.14946v1#bib.bib3)). To navigate such expansive design spaces, researchers have employed various optimization strategies, including random search, evolutionary algorithms, Bayesian optimization, and reinforcement learning (RL). Among these, RL-based methods (Zoph and Le [2016](https://arxiv.org/html/2508.14946v1#bib.bib21))(Talaat and Gamel [2023](https://arxiv.org/html/2508.14946v1#bib.bib9)) have shown particular promise by training a controller network to iteratively generate high-performing architectures through performance-guided feedback. In parallel, evolutionary algorithms have also demonstrated strong performance in architecture search, with some approaches rivaling or even surpassing RL-based techniques (So, Le, and Liang [2019](https://arxiv.org/html/2508.14946v1#bib.bib8)). Evolutionary methods are particularly adept at fine-tuning promising models by iteratively mutating them to create similar, but potentially improved variants. In contrast, RL methods tend to sample from a learned distribution over architectures, which can make it harder to incrementally improve high-performing candidates unless they are consistently rewarded. However, a key shortcoming of many evolutionary approaches is their reliance on manually defined or random mutation operators, which do not adapt based on prior experience and, therefore, lack the capacity to learn better strategies over time. 

Reinforcement learning (RL)-based NAS models, while popular for architecture generation, struggle to efficiently traverse such large and unstructured spaces due to sparse rewards and slow convergence (Pham et al. [2018](https://arxiv.org/html/2508.14946v1#bib.bib7)). These limitations motivate the need for structured and adaptive search strategies that can guide exploration more effectively. 

To address the limitations of flat and unconstrained search strategies in existing NAS methods for text classification, we propose HHNAS-AM, a novel framework that introduces structured architectural templates and adaptive search mechanisms. HHNAS-AM operates across multiple hierarchical levels of architectural design, enabling the search to capture both local and global structural patterns relevant to textual data. By constraining the search space using domain-informed hybrid templates, the method reduces redundancy and enhances the search tractability.

Crucially, HHNAS-AM incorporates adaptive mutation strategies governed by Q-learning, which dynamically adjust exploration behavior based on performance feedback from prior iterations. This feedback-driven adaptation allows the search to progressively focus on promising subspaces, improving efficiency without sacrificing diversity. Furthermore, the framework employs a fully probabilistic search process, striking a principled balance between exploration and exploitation. Our key contributions of HHNAS-AM include:

*   •Hybrid Hierarchical Search Space: We design a domain-informed, hierarchical search space tailored for text classification, which combines structural templates with parameter-level flexibility. This significantly reduces redundant exploration and enables more effective architecture discovery. 
*   •Adaptive Mutation Strategy via Q-Learning: We propose a Q-learning-based adaptive mutation mechanism that dynamically adjusts mutation probabilities based on performance feedback from prior iterations. This facilitates a more targeted and efficient search, leading to faster convergence toward high-performing models. 
*   •Empirical Performance and Efficiency Gains: Through extensive experiments on the Spider benchmark dataset (Yu et al. [2018](https://arxiv.org/html/2508.14946v1#bib.bib18)) for db_id (database id) prediction, we demonstrate that HHNAS-AM consistently achieves competitive or superior accuracy while significantly reducing computational overhead. The automated search eliminates the need for manual architecture and hyperparameter tuning. 
*   •Performance Improvement over existing Work: Compared to previous manually optimized models on the same dataset, HHNAS-AM yields a 8%jump in classification accuracy, underscoring the effectiveness of hierarchical search combined with adaptive learning strategies. 

Related Work
------------

Neural Architecture Search (NAS) has evolved significantly to address the demands of text classification, where traditional vision-based NAS approaches often fall short due to the unique structural and semantic characteristics of textual data. TextNAS (Wang et al. [2020](https://arxiv.org/html/2508.14946v1#bib.bib12)) presents an early effort to define NAS search spaces tailored for NLP by incorporating depth, width, kernel size, and attention-based modules. Their architecture-specific adaptations yield superior performance over hand-designed baselines. Extending this direction, Xue et al. ([2021](https://arxiv.org/html/2508.14946v1#bib.bib16)) propose a block-level NAS strategy with self-adaptive mutation, leveraging evolutionary feedback to improve convergence and structural diversity. Similarly, Zhang et al. ([2022](https://arxiv.org/html/2508.14946v1#bib.bib19)) apply a genetic algorithm within a one-shot NAS framework to efficiently explore macro-level network structures using hypergraph representations and parameter sharing. Benchmarking has also played a critical role in advancing NAS research. NAS-Bench-NLP (Klyuchnikov et al. [2022](https://arxiv.org/html/2508.14946v1#bib.bib4)) provides a large-scale benchmark of pre-evaluated Transformer architectures on GLUE tasks, greatly reducing the overhead of training from scratch and facilitating fair NAS comparisons. Meanwhile, ECGP-NAS (Wu et al. [2023](https://arxiv.org/html/2508.14946v1#bib.bib13)) introduces Cartesian Genetic Programming to evolve compact architectures under constrained environments, making NAS feasible without GPUs. Recent innovations also explore structural reformulations. BGNAS (Yan et al. [2024](https://arxiv.org/html/2508.14946v1#bib.bib17)) replaces DAG-based encoding with bipartite graphs, enabling more expressive architecture modeling via submodular optimization. MOEA(Xue, Chen, and Słowik [2023](https://arxiv.org/html/2508.14946v1#bib.bib15)) balances multiple objectives like accuracy and efficiency using customized genetic operators and Pareto optimization. In contrast,Chauhan, Bhattacharyya, and Vadivel ([2023](https://arxiv.org/html/2508.14946v1#bib.bib1)) employs Double Deep Q-Networks for reinforcement learning-based NAS with prioritized replay and one-shot training.Beyond architectural design, efficiency-centric approaches have emerged. Wang et al. ([2025](https://arxiv.org/html/2508.14946v1#bib.bib11)) ABG-NAS fuses Bayesian optimization with gradient-based updates to reduce evaluation cost, while DDNAS (Chen, Li, and Lee [2023](https://arxiv.org/html/2508.14946v1#bib.bib2)) introduces dynamic depth selection into a differentiable supernet for inference efficiency. On a different front, ATLAS (Xing et al. [2024](https://arxiv.org/html/2508.14946v1#bib.bib14)) addresses tabular data using a zero-cost proxy with budget-aware refinement, highlighting cross-domain transferability of NAS techniques. While the above approaches offer significant contributions, they either assume flat architecture spaces, rely on fixed mutation strategies, or are limited by task-specific assumptions. In contrast, our proposed HHNAS-AM introduces a hybrid hierarchical NAS framework specifically for text classification. It organizes the search space into macro and micro levels, allowing architectural templates and parameter configurations to co-evolve. By integrating a Q-learning-based adaptive mutation strategy, HHNAS-AM dynamically adjusts its exploration policy based on performance feedback, promoting more targeted and efficient architecture discovery. Furthermore, unlike traditional NAS methods that require exhaustive search or fixed mutation rules, our method learns which parameters to mutate and how, driven by a learned mutation probability model. On the Spyder benchmark dataset for db_id prediction, HHNAS-AM not only outperforms prior baselines but also achieves a 8% gain in accuracy over our previous best, while significantly reducing manual design effort and search cost.

Our Approach
------------

Our objective is to employ NAS with adaptive mutation strategies to discover optimal neural architectures and parameter configurations that yield high accuracy on a text classification task, specifically, db_id prediction. Unlike prior NAS efforts, which are primarily designed for vision tasks or rely on uniform architecture families, our method embraces architectural diversity and domain-specific cues to address the unique challenges of textual data. To this end, we introduce HHNAS-AS, a framework that explores both Transformer-based architectures (e.g., RoBERTa) and their hybrid variants formed by combining RoBERTa with convolutional layers either in parallel or in series. Additionally, we incorporate LLM-generated logical rules and entity-level signals (true/false classification), further enriching the architectural search space. This integration of heterogeneous components is what we refer to as a hybrid search paradigm.

Our HHNAS-AM framework operates on a two-level hierarchical search space:

*   •Macro-level: Defines high-level architectural templates (e.g., RoBERTa alone, RoBERTa+CNN in parallel or series). 
*   •Micro-level: Searches within each macro-template for optimal hyperparameters (e.g., layer sizes, dropout rates, kernel sizes). 

We introduce a curated set of architectural templates based on domain-specific insights, which significantly constrains and organizes the search space, improving both relevance and efficiency. At the core of our search process is a Q-learning-based adaptive mutation strategy. In each iteration, the framework generates a new architecture by probabilistically mutating a subset of parameters from the previously evaluated model. The decision to mutate each parameter is guided by a Q-table, which is updated based on the validation accuracy of the last sampled architecture. This performance-driven feedback loop enables the mutation policy to adapt over time, balancing exploration of novel configurations with exploitation of promising regions in the search space. Finally, the entire framework is fully probabilistic, ensuring a diverse and dynamic search trajectory that mitigates premature convergence and supports robust architecture discovery. In the following subsections, we detail the components of our approach.

### Hierarchical Search Space

The proposed HHNAS-AM framework organizes the architectural search space hierarchically into two distinct levels: macro and micro. This organization enables structured exploration of high-level architectural designs while allowing fine-grained parameter tuning within each selected design.

#### Macro-Level Search: Architecture Selection via Binary Encoding

At the macro level, the search space is defined by three binary decision variables, denoted as p 1,p 2,p 3∈{0,1}p_{1},p_{2},p_{3}\in\{0,1\}, each representing a key architectural component. Together, these binary parameters form a 3-bit code, giving rise to a total of 2 3=8 2^{3}=8 possible architectural configurations. Each combination corresponds to a unique high-level architecture. The initial macro configuration be represented by a binary vector:

𝐚(t)=[p 1(t),p 2(t),p 3(t)]\mathbf{a}^{(t)}=[p_{1}^{(t)},p_{2}^{(t)},p_{3}^{(t)}]

where t t denotes the current iteration. Each parameter p i(t)p_{i}^{(t)} is associated with a mutation probability π i(t)∈[0,1]\pi_{i}^{(t)}\in[0,1]. At each iteration, mutation is applied independently to each p i p_{i} based on its probability π i(t)\pi_{i}^{(t)}. A mutation operation flips the bit from 0 to 1 or vice versa.

𝐚(t)=[0,0,0],𝝅(t)=[0.5,0.75,0.3]\mathbf{a}^{(t)}=[0,0,0],\quad\boldsymbol{\pi}^{(t)}=[0.5,0.75,0.3]

Suppose a mutation occurs on p 1 p_{1} and p 3 p_{3}, the updated vector becomes:

𝐚(t+1)=[1,0,1]\mathbf{a}^{(t+1)}=[1,0,1]

This binary vector can be interpreted as a base-2 integer:

ArchIndex(t+1)=∑i=1 3 p i(t+1)⋅2 3−i=1⋅2 2+0⋅2 1+1⋅2 0=5\text{ArchIndex}^{(t+1)}=\sum_{i=1}^{3}p_{i}^{(t+1)}\cdot 2^{3-i}=1\cdot 2^{2}+0\cdot 2^{1}+1\cdot 2^{0}=5

Thus, the framework selects the 5th architecture (out of 8) for evaluation in the next round.

The probabilities π i(t)\pi_{i}^{(t)} for each macro parameter are dynamically updated using a Q-learning-inspired strategy based on the performance feedback of the selected architecture. If a parameter mutation led to performance improvement, the corresponding Q-value—and hence its mutation probability—is increased, promoting adaptive focus on influential components.

However, In the Existing experiments on the Spider dataset (Yu et al. [2018](https://arxiv.org/html/2508.14946v1#bib.bib18)) for the db_id prediction task have shown that RoBERTa alone achieves over 90% accuracy, indicating its strong standalone performance (Tripathi et al. [2025](https://arxiv.org/html/2508.14946v1#bib.bib10)). Consequently, in our work, we fix the first bit p 1=1 p_{1}=1, corresponding to the inclusion of RoBERTa in every candidate model. This design choice reduces the effective macro-level search space to four distinct configurations, defined by the remaining two mutable bits p 2,p 3 p_{2},p_{3}. Each of these configurations represents a unique hybrid composition involving RoBERTa and auxiliary components (e.g., CNN layers, LLM rule-based modules).

𝐚(t)=[1,p 2(t),p 3(t)],p 2,p 3∈{0,1}\mathbf{a}^{(t)}=[1,p_{2}^{(t)},p_{3}^{(t)}],\quad p_{2},p_{3}\in\{0,1\}

Then the effective architecture index becomes:

ArchIndex(t)=∑i=2 3 p i(t)⋅2 3−i\text{ArchIndex}^{(t)}=\sum_{i=2}^{3}p_{i}^{(t)}\cdot 2^{3-i}

yielding architecture indices ∈{0,1,2,3}\in\{0,1,2,3\}, corresponding to four distinct hybrid models built on a fixed RoBERTa backbone.

By allowing for reducing the macro-level search space in this principled way, HHNAS-AM concentrates its exploration on the most impactful architectural variants, ensuring computational efficiency while leveraging the proven efficacy of RoBERTa in the example of db_id.

#### Micro-Level Search: Fine-Grained Parameter Mutation and Adaptation

Once a macro-level architecture is selected, the micro-level search operates (Algorithm[1](https://arxiv.org/html/2508.14946v1#alg1 "Algorithm 1 ‣ Methods for performing Mutation ‣ Our Approach ‣ HHNAS-AM: Hierarchical Hybrid Neural Architecture Search using Adaptive Mutation Policies")) within its associated parameter subspace to optimize hyperparameters that significantly influence model performance. Each architecture is coupled with a unique set of mutable parameters 𝜽={θ 1,θ 2,…,θ k}\boldsymbol{\theta}=\{\theta_{1},\theta_{2},\dots,\theta_{k}\}, which may include both continuous (e.g., learning rate, dropout rate) and discrete (e.g., number of filters, kernel size, hidden layer size) variables.

Let 𝒜 j\mathcal{A}_{j} denote the architecture selected at iteration t t from the macro-level search, and let 𝜽 𝒜 j(t)\boldsymbol{\theta}^{(t)}_{\mathcal{A}_{j}} be its associated parameter vector. For each parameter θ i∈𝜽 𝒜 j\theta_{i}\in\boldsymbol{\theta}_{\mathcal{A}_{j}}, there exists a mutation probability π i(t)∈[0,1]\pi_{i}^{(t)}\in[0,1], and a candidate mutation is sampled as:

θ i(t+1)={Mutate​(θ i(t)),with probability​π i(t)θ i(t),otherwise\theta_{i}^{(t+1)}=\begin{cases}\text{Mutate}(\theta_{i}^{(t)}),&\text{with probability }\pi_{i}^{(t)}\\ \theta_{i}^{(t)},&\text{otherwise}\end{cases}

Here, Mutate​(⋅)\text{Mutate}(\cdot) refers to a perturbation function, which depends on the type of parameter—i.e., Gaussian noise for continuous variables or categorical sampling for discrete parameters.

After applying the selected mutations, the resulting architecture 𝒜 j\mathcal{A}_{j} with updated micro-level parameters 𝜽 𝒜 j(t+1)\boldsymbol{\theta}^{(t+1)}_{\mathcal{A}_{j}} is trained on the Spider dataset, and the validation accuracy α(t+1)\alpha^{(t+1)} is computed. This accuracy serves as a reward signal for updating the corresponding Q-values associated with each mutated parameter.

This adaptive micro-level strategy enables HHNAS-AM to progressively refine its parameter choices for each architecture, guided by empirical feedback from the learning task. Together with the macro-level hierarchy, it forms a probabilistic and feedback-driven framework for architecture and hyperparameter co-optimization in text classification.

### Modelling Mutation Probabilities via Q-Table

To enable performance-driven adaptation during the architecture search, we utilize a Q-learning-based mechanism to model mutation probabilities for all mutable parameters in the search space. Specifically, a Q-table is maintained to estimate the expected reward for performing an action on a given parameter, where each mutable feature s∈S s\in S is associated with a set of two possible actions: increase (denoted “+”) and decrease (denoted “-”) by x∈[ℤ+={1,2,3,…}]x\in[\mathbb{Z}^{+}=\{1,2,3,\ldots\}].

Formally, let Q​(s,a)Q(s,a) denote the Q-value corresponding to applying action a∈A s={+,−}a\in A_{s}=\{+,-\} on feature s s, A s A_{s} denotes the set of all possible actions that can be performed on the feature s s. The cumulative Q-value for a feature s s is computed as:

Q​(s)=∑a∈A s Q​(s,a)Q(s)=\sum_{a\in A_{s}}Q(s,a)

These cumulative Q-values reflect the historical utility of modifying each parameter and are used to determine the likelihood of selecting a parameter for mutation. The mutation probability P​(s)P(s) for each feature s s is defined as a normalized score relative to the most influential feature in the current Q-table:

P​(s)=Q​(s)max s′∈S⁡Q​(s′)⋅maxProb P(s)=\frac{Q(s)}{\max_{s^{\prime}\in S}Q(s^{\prime})}\cdot\text{maxProb}

Here, maxProb∈(0,1]\text{maxProb}\in(0,1] is a user-defined hyperparameter that sets the upper bound on the mutation probability. This formulation ensures that features with consistently high impact on performance are prioritized for mutation, while still allowing occasional exploration of less frequently updated parameters.

Importantly, each parameter is evaluated and mutated independently, allowing the algorithm to explore diverse subspaces of the search space simultaneously. At each iteration, the parameters to be mutated are sampled probabilistically based on P​(s)P(s), and the selected mutations are applied to construct a new candidate architecture.

This Q-learning-guided mutation model serves as the backbone of our adaptive search strategy, enabling the system to refine its mutation policies over time based on observed performance trends, thereby improving convergence efficiency and search quality.

### Methods for performing Mutation

Mutation is performed on each parameter independently. First, the parameter to be mutated is chosen. Then, mutation is applied depending on whether the parameter is continuous or discrete, as per below.

*   •Binary Parameter: The only action that can be performed on a binary parameter is to flip it from ’0’ to ’1’ and vice versa. 
*   •Discrete Parameter: For a discrete parameter, only two actions are possible, which are ’+’ or ’-’, which correspond to an increase by 1 and decrease by 1, respectively. The probabilities for each are:

+⇒Q(s,+)/(Q(s,+)+Q(s,−))+\Rightarrow Q(s,+)/(Q(s,+)+Q(s,-))

−⇒Q(s,−)/(Q(s,+)+Q(s,−))-\Rightarrow Q(s,-)/(Q(s,+)+Q(s,-)) 
*   •Continuous Parameter: For a continuous parameter, only two actions are possible, which are ’+’ or ’-’, which correspond to increasing or decreasing the parameter. Let the parameter be ’s’. The probabilities for each are:

+⇒Q(s,+)/(Q(s,+)+Q(s,−))+\Rightarrow Q(s,+)/(Q(s,+)+Q(s,-))

−⇒Q(s,−)/(Q(s,+)+Q(s,−))-\Rightarrow Q(s,-)/(Q(s,+)+Q(s,-)) The amount by which the parameter s s increases or decreases is sampled from the distribution x∼𝒩​(0,v​a​r s)x\sim\mathcal{N}(0,var_{s}) where v​a​r s var_{s} is the stored variance corresponding to the parameter ’s’. To increase the parameter, it is set to μ+a​b​s​(x)\mu+abs(x) and otherwise to μ−a​b​s​(x)\mu-abs(x), where μ\mu is the mean value stored corresponding to the parameter s s. 

Algorithm 1 Architecture Search with Mutation and Q-Learning at Micro Level

1:Input: Initial architecture features

A A
, mutation probability

p p
, iterations

i i

2: {Initialize Q-table and mean and variance values for features}

3: Initialize Q-values, mean

μ\mu
, variance

σ 2\sigma^{2}

4:for

i=1 i=1
to

n n
do

5:

A m​u​t​a​t​e​d←Mutate​(A)A_{mutated}\leftarrow\text{Mutate}(A)
{Architecture features}

6:for each feature

a s∈A m​u​t​a​t​e​d a_{s}\in A_{mutated}
do

7: Mutate

a s a_{s}
with probability

p s p_{s}
{Independent feature mutation}

8: Add mutated feature to

A′A^{\prime}

9:end for

10: Train model with architecture

A′A^{\prime}
, compute accuracy

a​c​c acc

11: {Update statistics}

12:

UpdateQValues​(a​c​c)\text{UpdateQValues}(acc)

13:

UpdateMeanValues​(a​c​c,μ)\text{UpdateMeanValues}(acc,\mu)

14:

UpdateVarValues​(a​c​c,μ,σ 2)\text{UpdateVarValues}(acc,\mu,\sigma^{2})

15:

A←SelectFeatures​(A′)A\leftarrow\text{SelectFeatures}(A^{\prime})

16:end for

17:return Optimized architecture

A A

### Mean and Variance Update Rules

To effectively model adaptive mutation behavior for continuous parameters, we implement dynamic updates of the mean and variance associated with each such parameter. These statistics govern how future mutations are sampled and updated based on the performance of the current architecture relative to its historical average.

#### Mean Update

The update rule for the historical mean μ\mu of a parameter is designed to reflect performance improvement relative to the current sample value. The intuition is as follows:

If the sampled value is greater than the current mean, and the model’s performance exceeds its historical average, the mean should increase, and vice versa. The magnitude of the change is proportional to the difference between the accuracy of the current model p i p_{i} and the running average accuracy p i^\hat{p_{i}}.

The mean update rule is thus defined as:

μ new=μ old+k⋅(p i−p i^)\mu_{\text{new}}=\mu_{\text{old}}+k\cdot(p_{i}-\hat{p_{i}})

where k k is a scaling hyperparameter controlling the sensitivity of the update.

#### Variance Update

Variance σ 2\sigma^{2} controls the spread of sampling for a given parameter. The update strategy reflects how “surprising” or “off-distribution” the sampled value is relative to the current mean. Two update strategies are considered:

_(i) Distance-based update (heuristic):_

*   •Case 1: If the sampled value lies outside the interval (μ s−σ s,μ s+σ s)(\mu_{s}-\sigma_{s},\mu_{s}+\sigma_{s}): σ new 2=σ old 2+k​(|s−μ s σ old|−1)​(p i−p i^)\sigma^{2}_{\text{new}}=\sigma^{2}_{\text{old}}+k\left(\left|\frac{s-\mu_{s}}{\sigma_{\text{old}}}\right|-1\right)(p_{i}-\hat{p_{i}}) 
*   •Case 2: If the sampled value lies within (μ s−σ s,μ s+σ s)(\mu_{s}-\sigma_{s},\mu_{s}+\sigma_{s}): σ new 2=σ old 2+k​(1−|s−μ s σ old|)​(p i−p i^)\sigma^{2}_{\text{new}}=\sigma^{2}_{\text{old}}+k\left(1-\left|\frac{s-\mu_{s}}{\sigma_{\text{old}}}\right|\right)(p_{i}-\hat{p_{i}}) 

These updates encourage wider exploration when beneficial mutations come from less likely values, and narrower focus when optimal values are close to the mean.

_(ii) Statistical moment-based update:_

An alternative, more statistics-oriented update rule based on the deviation from expected variance is given by:

σ new 2=σ old 2+k⋅((s−μ s)2−σ old 2 σ old 2)​(p i−p i^)\sigma^{2}_{\text{new}}=\sigma^{2}_{\text{old}}+k\cdot\left(\frac{(s-\mu_{s})^{2}-\sigma^{2}_{\text{old}}}{\sigma^{2}_{\text{old}}}\right)(p_{i}-\hat{p_{i}})

This formulation treats the squared error as a sample variance estimator and adjusts the current variance accordingly.

Both of these update mechanisms ensure that the mutation process for continuous parameters evolves in a data-driven manner, adapting to performance feedback while maintaining a controlled balance between exploration and stability. Combined with the Q-learning-guided mutation strategy, these updates make the search process more responsive and efficient over time.

Experimental Setup
------------------

We evaluate HHNAS-AM on the db_id prediction task using the publicly available Spider benchmark dataset (Yu et al. [2018](https://arxiv.org/html/2508.14946v1#bib.bib18)), which is widely recognized for its structural and semantic complexity. To further assess the generalizability of the proposed framework, we also experiment on a confidential industrial dataset representative of real-world application scenarios. The architecture search process is carried out over 50 iterations, where both macro-level architectural decisions and micro-level hyperparameters are adapted using a Q-learning-based mutation policy. Each candidate architecture is trained and evaluated using consistent protocols on both datasets. Notably, to the best of our knowledge, prior NAS studies for text classification have not addressed classification problems involving this many classes. For instance, in prior work such as TextNAS (Wang et al. [2020](https://arxiv.org/html/2508.14946v1#bib.bib12)), the classification tasks involved relatively fewer class labels, making direct comparison on those datasets inappropriate. Therefore, we contextualize our evaluation by referencing performance benchmarks reported in related text-to-SQL paper(Tripathi et al. [2025](https://arxiv.org/html/2508.14946v1#bib.bib10)), which also utilize the Spider dataset.

#### Dataset

We evaluate our approach on two datasets. The first is the Spider dataset, a standard benchmark originally introduced in the context of text-to-SQL tasks. After preprocessing for the db_id prediction task, the dataset comprises 86 classes. We observe significant class overlap in the feature space, as visualized through t-SNE (Figure.[2](https://arxiv.org/html/2508.14946v1#Sx4.F2 "Figure 2 ‣ Dataset ‣ Experimental Setup ‣ HHNAS-AM: Hierarchical Hybrid Neural Architecture Search using Adaptive Mutation Policies")). The Spider dataset has a total of 6,998 rows and is split into 70%-15%-15% training, validation, and testing. The second dataset is a confidential industrial dataset that has 7,696 rows with 28 classes after preprocessing, following the same train-validation-test split protocol as the Spider dataset.

![Image 2: Refer to caption](https://arxiv.org/html/2508.14946v1/after_merging.png)

Figure 2: Overlapping Spider dataset classes after merging

#### Implementation Details

Our framework operates over a structured macro-level search space consisting of four distinct hybrid architectures, each containing a fixed RoBERTa backbone and additional components integrated in parallel. The four architectures evaluated are: Model 1: A standalone RoBERTa encoder. Model 2: RoBERTa with a parallel CNN processing large language model (LLM) features. Model 3: RoBERTa with a parallel CNN processing rule-based features. Model 4: RoBERTa with two parallel CNNs—one for LLM features and one for rule-based features. These architectures are selected at the macro-level using three binary decision variables, corresponding to the presence or absence of the CNN branches. Given the fixed RoBERTa core, the resulting effective macro-level search space contains 4 unique architecture combinations, each encoded via binary flags. During each iteration, the macro-level configuration is determined via a probabilistic bit-flip mutation policy. Once an architecture is selected, micro-level mutation is applied to a set of continuous and discrete hyperparameters, including: learning_rate, criterion_num, layer_size1, layer_size2, dropout_rate, and kernel_size. Each parameter is independently mutated using a Q-table-guided policy, where mutation probabilities are updated based on performance feedback using accuracy as the reward signal. This two-tier adaptive process is executed for 50 iterations, with each sampled model trained for 20 epochs using a batch size of 16. Optimizers used are AdamW, SGD with momentum and RMSprop when criterion_num is 0, 1, 2 respectively.

All experiments were conducted on an NVIDIA RTX A6000 GPU (48 GiB) and 256 GB RAM with the Ubuntu operating system. The average runtime per experiment was approximately 5 days. To evaluate search space exploration and stability, we conducted four independent runs on the Spider dataset. In all runs, Model4 (Figure[3](https://arxiv.org/html/2508.14946v1#Sx5.F3 "Figure 3 ‣ Results ‣ HHNAS-AM: Hierarchical Hybrid Neural Architecture Search using Adaptive Mutation Policies")) consistently emerged as the top-performing architecture, demonstrating both high validation accuracy and stable convergence. In contrast, Model2 (Figure[4](https://arxiv.org/html/2508.14946v1#Sx5.F4 "Figure 4 ‣ Results ‣ HHNAS-AM: Hierarchical Hybrid Neural Architecture Search using Adaptive Mutation Policies")) exhibited the poorest performance, even after extensive exploration. Models 1 and 3 showed moderate and relatively stable performance across runs.

To assess the generalization capacity of HHNAS-AM, we further conducted two experiments on a confidential industrial dataset comprising 28 classes. Interestingly, in this setting, Model3 emerged as the optimal architecture, outperforming the others in terms of validation accuracy. This result highlights the adaptive nature of the framework, which effectively allocates mutation efforts and architecture selection based on task-specific performance feedback. These observations collectively demonstrate that HHNAS-AM not only promotes diverse exploration of the search space but also adapts to different datasets by converging on distinct architecture choices in a task-sensitive manner.

Results
-------

We evaluate the effectiveness of the proposed HHNAS-AM framework through a two-level search: macro-level architecture selection and micro-level parameter optimization. As described in the experimental setup, we conducted four independent experiments on the Spider dataset (Yu et al. [2018](https://arxiv.org/html/2508.14946v1#bib.bib18)) to assess the convergence behavior, architecture stability, and overall model performance.

Across all experiments, Model4 consistently emerged as the best-performing architecture, demonstrating superior validation accuracy and robust convergence. The detailed results of each model across the four experiments are reported in Table[3](https://arxiv.org/html/2508.14946v1#Sx5.T3 "Table 3 ‣ Results ‣ HHNAS-AM: Hierarchical Hybrid Neural Architecture Search using Adaptive Mutation Policies"). The accuracy trajectories of the top-performing models over search iterations are illustrated in (Figure[3](https://arxiv.org/html/2508.14946v1#Sx5.F3 "Figure 3 ‣ Results ‣ HHNAS-AM: Hierarchical Hybrid Neural Architecture Search using Adaptive Mutation Policies")), which highlights the exploration-to-convergence trend of Model4, confirming its reliability and dominance within the search space.

![Image 3: Refer to caption](https://arxiv.org/html/2508.14946v1/final_M4_new.png)

Figure 3: Trends of Model4 across experiments on Spider dataset- exploration-to-convergence trend of Model4

Interestingly, although Model2 underwent substantial exploration, it failed to achieve consistent performance, exhibiting high variance across experiments. This behavior is captured in (Figure [4](https://arxiv.org/html/2508.14946v1#Sx5.F4 "Figure 4 ‣ Results ‣ HHNAS-AM: Hierarchical Hybrid Neural Architecture Search using Adaptive Mutation Policies")), where the fluctuation in validation accuracy underscores the model’s lack of convergence, despite frequent mutation.

![Image 4: Refer to caption](https://arxiv.org/html/2508.14946v1/final_M2.png)

Figure 4: Trends of Model2 across experiments on Spider dataset model’s lack of convergence

To evaluate the generalization capability of HHNAS-AM, we further conducted experiments on a confidential industrial dataset, as described in experimental details section In this setting, Model 3 demonstrated the best accuracy, outperforming others under the same NAS framework. These findings are summarized in Table [1](https://arxiv.org/html/2508.14946v1#Sx5.T1 "Table 1 ‣ Results ‣ HHNAS-AM: Hierarchical Hybrid Neural Architecture Search using Adaptive Mutation Policies"), emphasizing the adaptability of HHNAS-AM to varying data distributions.

Table 1: Test accuracy of models across HHNAS-AM Experiments on the Industry confidential dataset. Bold values indicate the best-performing model per experiment.

For comparative analysis, we also benchmark our results against prior work on db_id prediction in the text-to-SQL (Tripathi et al. [2025](https://arxiv.org/html/2508.14946v1#bib.bib10)) domain. The best reported accuracy from manual training approaches on the Spider dataset is 89.71. In contrast, HHNAS-AM achieves a performance gain of approximately 8.07%, reaching up to 97.78% accuracy—without requiring manual architectural tuning. This performance boost demonstrates not only the strength of our hierarchical search strategy but also its practicality in real-world model development pipelines.

Table 2: Comparison of db_id Prediction Accuracy with Text-to-SQL(Tripathi et al. [2025](https://arxiv.org/html/2508.14946v1#bib.bib10)) Baseline on spider dataset

Table 3: Validation accuracy of models across HHNAS-AM experiments on the Spider dataset. Bold values indicate the best-performing model per experiment.

Conclusion
----------

In this work, we proposed HHNAS-AM, a novel Hierarchical Hybrid Neural Architecture Search framework with Adaptive Mutation Policies for efficient and scalable model discovery in text classification tasks. Unlike conventional NAS methods, HHNAS-AM operates across two levels of abstraction: macro-level architecture selection and micro-level parameter optimization, both guided by a Q-learning-based mutation strategy. This design enables the framework to balance exploration and exploitation, adaptively refining its search trajectory based on performance feedback. Empirical evaluations on the complex Spider benchmark and a confidential industrial dataset demonstrate that HHNAS-AM consistently identifies high-performing and stable architectures with minimal manual intervention. Our approach achieves a substantial accuracy gain of over 8 compared to state-of-the-art manually designed baselines for db_id prediction, validating its practical utility and generalizability across domains.

References
----------

*   Chauhan, Bhattacharyya, and Vadivel (2023) Chauhan, A.; Bhattacharyya, S.; and Vadivel, S. 2023. Dqnas: Neural architecture search using reinforcement learning. _arXiv preprint arXiv:2301.06687_. 
*   Chen, Li, and Lee (2023) Chen, K.-C.; Li, C.-T.; and Lee, K.-J. 2023. DDNAS: Discretized Differentiable Neural Architecture Search for Text Classification. _ACM Transactions on Intelligent Systems and Technology_, 14(5): 1–22. 
*   Elsken, Metzen, and Hutter (2019) Elsken, T.; Metzen, J.H.; and Hutter, F. 2019. Neural architecture search: A survey. _Journal of Machine Learning Research_, 20(55): 1–21. 
*   Klyuchnikov et al. (2022) Klyuchnikov, N.; Trofimov, I.; Artemova, E.; Salnikov, M.; Fedorov, M.; Filippov, A.; and Burnaev, E. 2022. Nas-bench-nlp: neural architecture search benchmark for natural language processing. _IEEE Access_, 10: 45736–45747. 
*   Liu, Simonyan, and Yang (2018) Liu, H.; Simonyan, K.; and Yang, Y. 2018. Darts: Differentiable architecture search. _arXiv preprint arXiv:1806.09055_. 
*   Maziarz et al. (2019) Maziarz, K.; Tan, M.; Khorlin, A.; Chang, K.-Y.S.; and Gesmundo, A. 2019. Evo-nas: Evolutionary-neural hybrid agent for architecture search. 
*   Pham et al. (2018) Pham, H.; Guan, M.; Zoph, B.; Le, Q.; and Dean, J. 2018. Efficient neural architecture search via parameters sharing. In _International conference on machine learning_, 4095–4104. PMLR. 
*   So, Le, and Liang (2019) So, D.; Le, Q.; and Liang, C. 2019. The evolved transformer. In _International conference on machine learning_, 5877–5886. PMLR. 
*   Talaat and Gamel (2023) Talaat, F.M.; and Gamel, S.A. 2023. RL based hyper-parameters optimization algorithm (ROA) for convolutional neural network. _Journal of Ambient Intelligence and Humanized Computing_, 14(10): 13349–13359. 
*   Tripathi et al. (2025) Tripathi, A.; Patle, V.; Jain, A.; Pundir, A.; Menon, S.; Singh, A.K.; and Herremans, D. 2025. End-to-End Text-to-SQL with Dataset Selection: Leveraging LLMs for Adaptive Query Generation. In _Proceedings of IJCNN, Rome, Italy_. 
*   Wang et al. (2025) Wang, S.; Yin, J.; Cao, J.; Tang, M.; Wang, H.; and Zhang, Y. 2025. ABG-NAS: Adaptive Bayesian Genetic Neural Architecture Search for Graph Representation Learning. _arXiv preprint arXiv:2504.21254_. 
*   Wang et al. (2020) Wang, Y.; Yang, Y.; Chen, Y.; Bai, J.; Zhang, C.; Su, G.; Kou, X.; Tong, Y.; Yang, M.; and Zhou, L. 2020. Textnas: A neural architecture search space tailored for text representation. In _Proceedings of the AAAI conference on artificial intelligence_, volume 34, 9242–9249. 
*   Wu et al. (2023) Wu, X.; Wang, D.; Chen, H.; Yan, L.; Xiao, Y.; Miao, C.; Ge, H.; Xu, D.; Liang, Y.; Wang, K.; et al. 2023. Neural architecture search for text classification with limited computing resources using efficient Cartesian genetic programming. _IEEE Transactions on Evolutionary Computation_, 28(3): 638–652. 
*   Xing et al. (2024) Xing, N.; Cai, S.; Luo, Z.; Ooi, B.C.; and Pei, J. 2024. Anytime neural architecture search on tabular data. _arXiv preprint arXiv:2403.10318_. 
*   Xue, Chen, and Słowik (2023) Xue, Y.; Chen, C.; and Słowik, A. 2023. Neural architecture search based on a multi-objective evolutionary algorithm with probability stack. _IEEE transactions on evolutionary computation_, 27(4): 778–786. 
*   Xue et al. (2021) Xue, Y.; Wang, Y.; Liang, J.; and Slowik, A. 2021. A self-adaptive mutation neural architecture search algorithm based on blocks. _IEEE Computational Intelligence Magazine_, 16(3): 67–78. 
*   Yan et al. (2024) Yan, X.; Huang, H.; Jin, Y.; Wang, Z.; and Hao, Z. 2024. Neural architecture search based on bipartite graphs for text classification. _IEEE Transactions on Neural Networks and Learning Systems_. 
*   Yu et al. (2018) Yu, T.; Zhang, R.; Yang, K.; Yasunaga, M.; Wang, D.; Li, Z.; Ma, J.; Li, I.; Yao, Q.; Roman, S.; et al. 2018. Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-sql task. _arXiv preprint arXiv:1809.08887_. 
*   Zhang et al. (2022) Zhang, S.; Guo, L.; Fan, J.; Zhang, X.; and Zhang, W. 2022. Exploring neural architecture search for text classification. In _7th International Symposium on Advances in Electrical, Electronics, and Computer Engineering_, volume 12294, 1430–1436. SPIE. 
*   Zoph and Le (2017) Zoph, B.; and Le, Q. 2017. Neural Architecture Search with Reinforcement Learning. In _International Conference on Learning Representations_. 
*   Zoph and Le (2016) Zoph, B.; and Le, Q.V. 2016. Neural architecture search with reinforcement learning. _arXiv preprint arXiv:1611.01578_. 
*   Zoph et al. (2018) Zoph, B.; Vasudevan, V.; Shlens, J.; and Le, Q.V. 2018. Learning transferable architectures for scalable image recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_, 8697–8710.

