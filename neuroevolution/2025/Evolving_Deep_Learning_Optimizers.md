Title: Evolving Deep Learning Optimizers

URL Source: https://arxiv.org/html/2512.11853

Markdown Content:
###### Abstract

We present a genetic algorithm framework for automatically discovering deep learning optimization algorithms. Our approach encodes optimizers as genomes that specify combinations of primitive update terms (gradient, momentum, RMS normalization, Adam-style adaptive terms, and sign-based updates) along with hyperparameters and scheduling options. Through evolutionary search over 50 generations with a population of 50 individuals, evaluated across multiple vision tasks, we discover an evolved optimizer that outperforms Adam by 2.6% in aggregate fitness and achieves a 7.7% relative improvement on CIFAR-10. The evolved optimizer combines sign-based gradient terms with adaptive moment estimation, uses lower momentum coefficients than Adam (β 1=0.86\beta_{1}=0.86, β 2=0.94\beta_{2}=0.94), and notably disables bias correction while enabling learning rate warmup and cosine decay. Our results demonstrate that evolutionary search can discover competitive optimization algorithms and reveal design principles that differ from hand-crafted optimizers. Code is available at [https://github.com/mmarfinetz/evo-optimizer](https://github.com/mmarfinetz/evo-optimizer).

1 Introduction
--------------

The choice of optimizer is critical to deep learning success, yet the design of optimization algorithms remains largely a manual process guided by intuition and mathematical analysis. While optimizers like SGD with momentum (Polyak, [1964](https://arxiv.org/html/2512.11853v1#bib.bib8)), Adam (Kingma and Ba, [2014](https://arxiv.org/html/2512.11853v1#bib.bib5)), and more recently Lion (Chen et al., [2023](https://arxiv.org/html/2512.11853v1#bib.bib3)) have emerged from human insight, the space of possible update rules is vast and largely unexplored.

We propose an evolutionary approach to optimizer discovery that treats the optimization algorithm itself as the object of optimization. Our key insight is that many successful optimizers can be expressed as weighted combinations of a small set of primitive operations applied to gradients and their running statistics. By encoding these combinations as genomes and evolving them based on training performance across diverse tasks, we can search for evolved optimizers without requiring manual derivation.

Our contributions are:

1.   1.A genome representation for optimization algorithms that captures the essential components of modern optimizers in a compact, evolvable form. 
2.   2.A multi-task fitness evaluation protocol that encourages discovery of optimizers that generalize across datasets and architectures. 
3.   3.Empirical evidence that evolutionary search discovers optimizers competitive with hand-designed algorithms, achieving improvements over Adam on vision benchmarks. 
4.   4.Analysis of the evolved optimizer’s structure, revealing design choices (sign-based updates, disabled bias correction, aggressive scheduling) that differ from conventional wisdom. 

2 Related Work
--------------

#### Hand-Designed Optimizers.

The progression from SGD (Robbins and Monro, [1951](https://arxiv.org/html/2512.11853v1#bib.bib10)) to momentum (Polyak, [1964](https://arxiv.org/html/2512.11853v1#bib.bib8)), AdaGrad (Duchi et al., [2011](https://arxiv.org/html/2512.11853v1#bib.bib4)), RMSProp (Tieleman and Hinton, [2012](https://arxiv.org/html/2512.11853v1#bib.bib11)), and Adam (Kingma and Ba, [2014](https://arxiv.org/html/2512.11853v1#bib.bib5)) represents decades of manual optimizer engineering. Recent work has produced AdamW (Loshchilov and Hutter, [2017](https://arxiv.org/html/2512.11853v1#bib.bib7)), which decouples weight decay from gradient updates, and Lion (Chen et al., [2023](https://arxiv.org/html/2512.11853v1#bib.bib3)), which uses sign-based updates discovered through symbolic search.

#### Learning to Optimize.

Andrychowicz et al. ([2016](https://arxiv.org/html/2512.11853v1#bib.bib1)) introduced the idea of using neural networks to learn optimization algorithms, training an LSTM to output parameter updates. Subsequent work has explored meta-learning optimizers for specific domains (Li and Malik, [2017](https://arxiv.org/html/2512.11853v1#bib.bib6); Wichrowska et al., [2017](https://arxiv.org/html/2512.11853v1#bib.bib12)) and using reinforcement learning to discover update rules (Bello et al., [2017](https://arxiv.org/html/2512.11853v1#bib.bib2)). Our approach differs by using genetic algorithms with an explicit, interpretable genome rather than black-box neural networks.

#### Symbolic and Evolutionary Search.

Chen et al. ([2023](https://arxiv.org/html/2512.11853v1#bib.bib3)) used program search to discover Lion, demonstrating that simple symbolic expressions can outperform complex optimizers. Real et al. ([2019](https://arxiv.org/html/2512.11853v1#bib.bib9)) showed evolutionary methods can discover neural architectures competitive with hand-designed ones. We apply similar evolutionary principles to the optimizer search space.

3 Method
--------

### 3.1 Genome Representation

We represent an optimizer as a genome 𝒢\mathcal{G} that specifies how to compute parameter updates from gradients. The core update rule is:

Δ​w t=−η t​∑k=1 K α k⋅T k​(g t,m t,v t,ϵ)\Delta w_{t}=-\eta_{t}\sum_{k=1}^{K}\alpha_{k}\cdot T_{k}(g_{t},m_{t},v_{t},\epsilon)(1)

where η t\eta_{t} is the (possibly scheduled) learning rate, α k\alpha_{k} are learned coefficients, and T k T_{k} are primitive terms selected from a catalog.

#### Primitive Terms.

We define seven primitive operations that encompass the building blocks of modern optimizers:

*   •Grad: g t g_{t} — raw gradient 
*   •Momentum: m t m_{t} — exponential moving average of gradients 
*   •RmsNorm: g t/(v t+ϵ)g_{t}/(\sqrt{v_{t}}+\epsilon) — RMSProp-style normalization 
*   •AdamTerm: m t/(v t+ϵ)m_{t}/(\sqrt{v_{t}}+\epsilon) — Adam-style adaptive term 
*   •SignGrad: sign​(g t)\text{sign}(g_{t}) — sign of gradient 
*   •UnitGrad: g t/(|g t|+ϵ)g_{t}/(|g_{t}|+\epsilon) — unit gradient 
*   •Nesterov: m t+β 1​(g t−m t)m_{t}+\beta_{1}(g_{t}-m_{t}) — Nesterov-style lookahead 

#### Genome Components.

A complete genome consists of:

*   •Terms: List of 1–4 primitive types with corresponding α\alpha coefficients 
*   •Hyperparameters: log 10⁡(η)\log_{10}(\eta), β 1\beta_{1}, β 2\beta_{2}, log 10⁡(ϵ)\log_{10}(\epsilon), log 10⁡(λ)\log_{10}(\lambda) (decoupled weight decay applied in AdamW style, i.e., w←(1−η t​λ)​w−Δ​w t w\leftarrow(1-\eta_{t}\lambda)w-\Delta w_{t}) 
*   •Flags: Use momentum (m t m_{t}), use second moment (v t v_{t}), bias correction, gradient clipping 
*   •Schedule: Warmup steps, cosine decay 

When gradient clipping is enabled in the genome, we apply elementwise clipping to each gradient component: g t←clip​(g t,−c,c)g_{t}\leftarrow\mathrm{clip}(g_{t},-c,c) with a fixed threshold c c. In the evolved optimizer, clipping is disabled.

This representation can express SGD (K=1 K=1, Grad), Adam (K=1 K=1, AdamTerm with bias correction), and novel combinations not previously explored.

### 3.2 Fitness Evaluation

To encourage discovery of general-purpose optimizers, we evaluate each genome across multiple tasks:

Fitness​(𝒢)=1|T|​∑τ∈T 1 S​∑s=1 S f​(𝒢,τ,s)\text{Fitness}(\mathcal{G})=\frac{1}{|T|}\sum_{\tau\in T}\frac{1}{S}\sum_{s=1}^{S}f(\mathcal{G},\tau,s)(2)

where T T is the set of tasks, S S is the number of random seeds, and f​(𝒢,τ,s)f(\mathcal{G},\tau,s) is the scalar fitness on task τ\tau with seed s s. Concretely, we define

f​(𝒢,τ,s)={acc test+λ loss⋅max⁡(0,1−ℓ¯train,last 50)if run is stable,−1 if run diverges,f(\mathcal{G},\tau,s)=\begin{cases}\mathrm{acc}_{\text{test}}+\lambda_{\text{loss}}\cdot\max\bigl(0,1-\overline{\ell}_{\text{train,last 50}}\bigr)&\text{if run is stable},\\ -1&\text{if run diverges},\end{cases}(3)

where acc test\mathrm{acc}_{\text{test}} is test accuracy at the final step, ℓ¯train,last 50\overline{\ell}_{\text{train,last 50}} is the mean training loss over the last 50 steps, and λ loss=0.05\lambda_{\text{loss}}=0.05. This bonus is capped at +0.05+0.05 and is only positive when the final training loss is below 1.0 1.0, which explains why fitness values can slightly exceed 1.0 1.0 on MNIST. We mark a run as divergent if the training loss becomes NaN or Inf, exceeds 50 50 at any point, or if the training loop raises a numerical RuntimeError; such runs receive fitness −1-1.

### 3.3 Genetic Algorithm

We evolve a population of N=50 N=50 genomes over G=50 G=50 generations using:

#### Selection.

Tournament selection with k=4 k=4 candidates, plus elitism preserving the top 3 individuals.

#### Crossover.

Single-point crossover for term lists; uniform crossover for scalar hyperparameters and flags.

#### Mutation.

Adaptive mutation rates that decay over generations:

*   •Numeric mutations: Gaussian perturbation of hyperparameters 
*   •Structural mutations: Add, remove, or change primitive terms 
*   •Flag mutations: Flip boolean settings 

#### Initialization.

The initial population includes known optimizers (SGD, Adam, AdamW, RMSProp) as seeds, with the remainder randomly generated.

4 Experiments
-------------

### 4.1 Setup

#### Tasks.

We evaluate on three vision classification tasks:

*   •Fashion-MNIST: 28×\times 28 grayscale, 10 classes, SmallCNN 
*   •CIFAR-10: 32×\times 32 RGB, 10 classes, deeper CNN with BatchNorm 
*   •MNIST: 28×\times 28 grayscale, 10 classes, SmallCNN 

#### Training.

During evolution, each evaluation trains for 500 optimization steps with batch size 128. For the final comparison in Table[1](https://arxiv.org/html/2512.11853v1#S4.T1 "Table 1 ‣ 4.2 Results ‣ 4 Experiments ‣ Evolving Deep Learning Optimizers"), we re-train each optimizer for 1000 steps using the same architectures, batch size, and data subsampling protocol. Evolution uses 2 seeds per task, while the final comparison averages over 3 seeds. Total evolution time was approximately 13 hours on an NVIDIA Tesla T4.

For each dataset we sample a fixed random subset of 15,000 training examples to speed up evaluation. All optimizers, including baselines, are trained on the same subset for a given task.

#### Architectures and Implementation Details.

For MNIST and Fashion-MNIST we use a SmallCNN with two convolutional layers (32 and 64 channels, 3×3 3\times 3 kernels, ReLU activations), each followed by 2×2 2\times 2 max-pooling, and a classifier consisting of a fully connected layer with 256 hidden units, ReLU, dropout, and a final linear layer to 10 classes. For CIFAR-10 we use a CNN with two convolutional blocks: a 32-channel block and a 64-channel block, each containing two 3×3 3\times 3 convolutions with Batch Normalization and ReLU, followed by 2×2 2\times 2 max-pooling and dropout, and a classifier with a 512-unit fully connected layer, dropout, and a final linear classifier. All models use cross-entropy loss. We implement optimizers in PyTorch and apply decoupled weight decay and (optionally) gradient clipping as described in Sec.[3](https://arxiv.org/html/2512.11853v1#S3 "3 Method ‣ Evolving Deep Learning Optimizers").

### 4.2 Results

Table 1: Comparison of evolved optimizer with baselines. Fitness combines test accuracy with a small training-loss bonus (Eq.[3](https://arxiv.org/html/2512.11853v1#S3.E3 "In 3.2 Fitness Evaluation ‣ 3 Method ‣ Evolving Deep Learning Optimizers")). Results averaged over 3 seeds with 1000 training steps.

Table[1](https://arxiv.org/html/2512.11853v1#S4.T1 "Table 1 ‣ 4.2 Results ‣ 4 Experiments ‣ Evolving Deep Learning Optimizers") shows that the evolved optimizer outperforms all baselines, with the largest improvement on CIFAR-10 (+7.7% relative to Adam).

### 4.3 Analysis of Evolved Optimizer

The best genome after 50 generations has the following structure:

#### Update Rule.

Δ​w=−η​(0.73​sign​(g)+3.63​m v+ϵ)\Delta w=-\eta\left(0.73\,\mathrm{sign}(g)+3.63\,\frac{m}{\sqrt{v}+\epsilon}\right)(4)

This combines sign-based updates (total weight 0.73) with Adam-style adaptive terms (total weight 3.63). The underlying genome contained duplicate primitive terms (two SignGrad and two AdamTerm entries); we merge them here for clarity by summing their coefficients.

#### Hyperparameters.

*   •Learning rate: 1.2×10−3 1.2\times 10^{-3} (slightly higher than Adam’s typical 10−3 10^{-3}) 
*   •β 1=0.855\beta_{1}=0.855 (vs. Adam’s 0.9) — less momentum smoothing 
*   •β 2=0.936\beta_{2}=0.936 (vs. Adam’s 0.999) — faster adaptation to gradient magnitude 
*   •Weight decay: 9.7×10−4 9.7\times 10^{-4} 

#### Notable Design Choices.

*   •Bias correction disabled: Unlike Adam, the evolved optimizer does not correct for initialization bias in moment estimates. 
*   •Warmup enabled: 100 steps of linear warmup, which may compensate for disabled bias correction. 
*   •Cosine decay: Learning rate annealing over training. 
*   •Sign gradient component: Provides magnitude-invariant updates, similar to Lion. 

#### Population Dynamics.

Figure[1](https://arxiv.org/html/2512.11853v1#S4.F1 "Figure 1 ‣ Population Dynamics. ‣ 4.3 Analysis of Evolved Optimizer ‣ 4 Experiments ‣ Evolving Deep Learning Optimizers") shows that SignGrad and AdamTerm dominated the population by generation 50, with 102 and 88 occurrences respectively. Other primitives were largely eliminated by selection pressure, suggesting these two components are particularly effective.

![Image 1: Refer to caption](https://arxiv.org/html/2512.11853v1/figures/evolution_progress.png)

Figure 1: Evolution progress showing best and average fitness over 50 generations.

5 Discussion
------------

#### Relation to Lion.

The evolved optimizer’s use of sign-based updates parallels the Lion optimizer (Chen et al., [2023](https://arxiv.org/html/2512.11853v1#bib.bib3)), which was discovered through symbolic program search. However, our approach discovered this independently through evolutionary pressure, providing convergent evidence for the effectiveness of sign-based updates. Unlike Lion, our evolved optimizer retains Adam-style adaptive terms, suggesting a hybrid approach may be beneficial.

#### Why Disable Bias Correction?

Adam’s bias correction compensates for the zero-initialization of moment estimates. Our evolved optimizer disables this but enables warmup, which serves a similar purpose by using small learning rates early in training. This suggests the two mechanisms may be redundant.

#### Lower Momentum Coefficients.

The evolved β 1=0.855\beta_{1}=0.855 and β 2=0.936\beta_{2}=0.936 make the optimizer more responsive to recent gradients than Adam. This may be beneficial for the relatively short training runs (500 steps) used during evolution, though it warrants investigation on longer training.

6 Limitations and Future Work
-----------------------------

*   •Scale: We evaluated on small CNNs with short training runs. Validation on larger models (ResNets, Transformers) and longer training is needed. 
*   •Task diversity: Only vision classification tasks were used. Language modeling and reinforcement learning would test generalization. 
*   •Single evolution run: Results are from one evolutionary run. Multiple runs would establish robustness. 
*   •Missing baselines: We did not compare to Lion, AdaFactor, or other modern optimizers. 
*   •Theoretical analysis: We provide no convergence guarantees or theoretical justification for the evolved update rule. 

Future work could address these limitations and explore whether the genome representation can be extended to capture more complex scheduling strategies or per-layer adaptation.

7 Conclusion
------------

We presented a genetic algorithm framework for discovering deep learning optimizers. By encoding optimizers as genomes specifying combinations of primitive update terms, we evolved an optimizer that outperforms Adam on vision benchmarks. The evolved algorithm combines sign-based and adaptive updates, uses aggressive scheduling, and makes design choices (disabled bias correction, lower momentum) that differ from hand-designed optimizers. Our results suggest that evolutionary search is a viable approach to optimizer discovery and can reveal design principles not apparent from manual analysis.

References
----------

*   Andrychowicz et al. [2016] Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, and Nando De Freitas. Learning to learn by gradient descent by gradient descent. In _Advances in Neural Information Processing Systems_, pages 3981–3989, 2016. 
*   Bello et al. [2017] Irwan Bello, Barret Zoph, Vijay Vasudevan, and Quoc V Le. Neural optimizer search with reinforcement learning. In _International Conference on Machine Learning_, pages 459–468, 2017. 
*   Chen et al. [2023] Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Yao Liu, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, and Quoc V Le. Symbolic discovery of optimization algorithms. _arXiv preprint arXiv:2302.06675_, 2023. 
*   Duchi et al. [2011] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. _Journal of Machine Learning Research_, 12:2121–2159, 2011. 
*   Kingma and Ba [2014] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_, 2014. 
*   Li and Malik [2017] Ke Li and Jitendra Malik. Learning to optimize. In _International Conference on Learning Representations_, 2017. 
*   Loshchilov and Hutter [2017] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_, 2017. 
*   Polyak [1964] Boris T Polyak. Some methods of speeding up the convergence of iteration methods. _USSR Computational Mathematics and Mathematical Physics_, 4(5):1–17, 1964. 
*   Real et al. [2019] Esteban Real, Alok Aggarwal, Yanping Huang, and Quoc V Le. Regularized evolution for image classifier architecture search. In _AAAI Conference on Artificial Intelligence_, volume 33, pages 4780–4789, 2019. 
*   Robbins and Monro [1951] Herbert Robbins and Sutton Monro. A stochastic approximation method. _The Annals of Mathematical Statistics_, pages 400–407, 1951. 
*   Tieleman and Hinton [2012] Tijmen Tieleman and Geoffrey Hinton. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning, 2012. 
*   Wichrowska et al. [2017] Olga Wichrowska, Niru Maheswaranathan, Matthew W Hoffman, Sergio Gomez Colmenarejo, Misha Denil, Nando de Freitas, and Jascha Sohl-Dickstein. Learned optimizers that scale and generalize. In _International Conference on Machine Learning_, pages 3751–3760, 2017.

