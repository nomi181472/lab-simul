Title: EA4LLM: A Gradient-Free Approach to Large Language Model Optimization via Evolutionary Algorithms

URL Source: https://arxiv.org/html/2510.10603

Published Time: Fri, 24 Oct 2025 00:23:37 GMT

Markdown Content:
###### Abstract

In recent years, large language models (LLMs) have made remarkable progress, with model optimization primarily relying on gradient-based optimizers such as Adam. However, these gradient-based methods impose stringent hardware requirements, demanding high-concurrency, high-memory GPUs. Moreover, they require all neural network operations to be differentiable, thereby excluding many promising non-differentiable architectures from practical use.

To address these limitations, we propose EA4LLM, an evolutionary algorithm for optimizing LLMs, and, for the first time, empirically verify full-parameter optimization from the pretraining stage across model sizes ranging from 0.5B to 32B. We conduct extensive experiments and provide key insights into how evolutionary algorithms can effectively optimize neural networks.

Our work challenges the prevailing assumption that gradient-based optimization is the only viable approach for training neural networks. It also holds significant potential to reduce the computational cost of training large language models, thereby enabling groups with limited computational resources to participate in deep learning research.

Machine Learning, ICML

![Image 1: Refer to caption](https://arxiv.org/html/2510.10603v2/x1.png)

Figure 1: Loss Curve of EAGPT Optimized from Pretraining Using Evolutionary Algorithms.

1 Introduction
--------------

Over the past few decades, gradient-based optimization has been the dominant—and often the main approach for training neural networks. A series of improved gradient-based algorithms have been successively proposed, including Stochastic Gradient Descent (SGD)(Robbins & Monro, [1951](https://arxiv.org/html/2510.10603v2#bib.bib15)), Adaptive Moment Estimation (Adam)(Kingma & Ba, [2015](https://arxiv.org/html/2510.10603v2#bib.bib11)), and more recently, EvoLved Sign Momentum (Lion)(Chen et al., [2023](https://arxiv.org/html/2510.10603v2#bib.bib3)).

However, gradient-based optimization suffers from two major limitations. First, it requires computing and storing gradients during training, leading to substantial memory and computational overhead. In practice, training typically incurs 3–8×\times the model parameter size in GPU memory (for parameters, gradients, and optimizer states) and 2–3×\times the computational cost of a forward pass (due to forward propagation, backpropagation, and gradient updates). Second, gradient-based methods require the model to be differentiable, which excludes many promising non-differentiable architectures. For example, in attention network design, using FAISS (Johnson et al., [2019](https://arxiv.org/html/2510.10603v2#bib.bib10)) for approximate nearest neighbor hashing to retrieve context from the key-value (KV) cache assumes differentiability, which prevents the use of this approach to implement efficient sparse attention mechanisms.

Igel et al.(Igel, [2003](https://arxiv.org/html/2510.10603v2#bib.bib8)) and Tim Salimans et al.(Salimans et al., [2017](https://arxiv.org/html/2510.10603v2#bib.bib16)) explore using Evolution Strategies (ES) as an alternative to Markov Decision Process (MDP)-based methods for optimizing neural networks in reinforcement learning. In recent years, some methods have utilized EAs to optimize large models’ prompts(Sun et al., [2022a](https://arxiv.org/html/2510.10603v2#bib.bib18), [b](https://arxiv.org/html/2510.10603v2#bib.bib19); Zhao et al., [2023](https://arxiv.org/html/2510.10603v2#bib.bib21); Guo et al., [2023](https://arxiv.org/html/2510.10603v2#bib.bib5)) and small parameter spaces(Jin et al., [2024](https://arxiv.org/html/2510.10603v2#bib.bib9); Carmona et al., [2024](https://arxiv.org/html/2510.10603v2#bib.bib2); Huang et al., [2025](https://arxiv.org/html/2510.10603v2#bib.bib7)), validating the ability of EAs to optimize large model parameters in smaller-scale settings. More recently, Xin Qiu et al.(Qiu et al., [2025](https://arxiv.org/html/2510.10603v2#bib.bib13)) adapted the reward computation from GRPO (Guo et al., [2025](https://arxiv.org/html/2510.10603v2#bib.bib4)) to design the fitness evaluation and population update mechanism in their ES algorithm, successfully extending ES to full-parameter fine-tuning of LLMs in the reinforcement learning stage for the first time. However, these works only apply ES within the reinforcement learning context, and their effectiveness heavily depends on the design of the RL reward function.

We further investigate the capability of Evolution Strategies for direct neural network optimization beyond reinforcement learning. We propose EA4LLM, a method that leverages evolutionary algorithms to optimize LLMs, and demonstrate for the first time that ES can effectively perform full-parameter optimization of LLMs starting from pretraining—validated across model sizes from 0.5B to 32B—as illustrated in [1](https://arxiv.org/html/2510.10603v2#S0.F1 "Figure 1 ‣ EA4LLM: A Gradient-Free Approach to Large Language Model Optimization via Evolutionary Algorithms"). In our implementation, we introduce a simple yet effective approach that links the model’s output logits to the fitness computation in ES. This implies that virtually any neural network with a probabilistic training objective P P can, in principle, be optimized using ES. Different from prior ES in supervised learning that typically treats mini-batch loss or performance as a black-box objective, we explicitly aggregate token-level log-probabilities (log-softmax) into the ES fitness and carry out full-parameter optimization for LLMs in pretraining; combined with antithetic sampling, rank shaping, grouped/layered controls, and cross-process synchronized subsampled evaluation, this substantially improves the practicality and efficiency of zeroth-order optimization in this setting. Building upon standard ES(Salimans et al., [2017](https://arxiv.org/html/2510.10603v2#bib.bib16)), we conduct extensive exploratory experiments and provide key insights—as well as identified shortcomings—regarding the direct application of ES to neural network optimization.

Our work challenges the assumption that gradient-based methods are the sole viable option for training neural networks. It also holds significant potential to reduce the computational cost of training large language models, thereby enabling research groups with limited computational resources—even those without access to GPUs—to conduct meaningful deep learning research.

2 Related Work
--------------

Evolutionary Algorithms (EAs) form a vast and diverse family of optimization techniques inspired by the process of natural selection(Back, [1996](https://arxiv.org/html/2510.10603v2#bib.bib1); Zhou et al., [2011](https://arxiv.org/html/2510.10603v2#bib.bib22); Zhang et al., [2008](https://arxiv.org/html/2510.10603v2#bib.bib20)). Among EAs, Evolution Strategies (ES)(Rechenberg, [1973](https://arxiv.org/html/2510.10603v2#bib.bib14)), which we utilize in our work, constitute merely a small fraction of this extensive domain. ES methods focus on optimizing parameters through random perturbations, crossover, mutation, and other mechanisms to sample a set of solutions. These solutions (individuals) are then evaluated based on their fitness, and the population is updated accordingly. This process repeats until a termination condition is met, such as reaching a maximum number of iterations. Since ES does not require gradient information, they are particularly suitable for non-differentiable problems or scenarios where gradient computation is impractical(Hansen, [2003](https://arxiv.org/html/2510.10603v2#bib.bib6)).

Despite the significant potential of using EAs for neural network optimization, research in this area has been relatively limited compared to the widespread adoption of gradient-based methods. Early efforts by Igel et al.(Igel, [2003](https://arxiv.org/html/2510.10603v2#bib.bib8)) and Tim Salimans et al.(Salimans et al., [2017](https://arxiv.org/html/2510.10603v2#bib.bib16)) demonstrated that ES could serve as an effective alternative to traditional reinforcement learning approaches, paving the way for subsequent studies(Lorenc & Neruda, [2025](https://arxiv.org/html/2510.10603v2#bib.bib12); Schulman et al., [2017](https://arxiv.org/html/2510.10603v2#bib.bib17)). In recent years, some methods have utilized EAs to optimize large models’ prompts(Sun et al., [2022a](https://arxiv.org/html/2510.10603v2#bib.bib18), [b](https://arxiv.org/html/2510.10603v2#bib.bib19); Zhao et al., [2023](https://arxiv.org/html/2510.10603v2#bib.bib21); Guo et al., [2023](https://arxiv.org/html/2510.10603v2#bib.bib5)) and small parameter spaces(Jin et al., [2024](https://arxiv.org/html/2510.10603v2#bib.bib9); Carmona et al., [2024](https://arxiv.org/html/2510.10603v2#bib.bib2); Huang et al., [2025](https://arxiv.org/html/2510.10603v2#bib.bib7)), validating the ability of EAs to optimize large model parameters in smaller-scale settings. More recently, Xin Qiu et al.(Qiu et al., [2025](https://arxiv.org/html/2510.10603v2#bib.bib13)) extended the application of ES to fine-tuning large language models (LLMs) in reinforcement learning, highlighting the algorithm’s capability to handle complex, high-dimensional optimization landscapes.

However, these pioneering works primarily focused on specific contexts such as reinforcement learning, leaving ample room for exploring the broader applicability of ES in direct neural network optimization. Our work builds upon this foundation, aiming to unlock the full potential of evolutionary strategies for training neural networks beyond the confines of reinforcement learning. By doing so, we not only challenge the prevailing paradigm of gradient-based optimization but also open new avenues for efficient, resource-conscious deep learning research.

3 Method
--------

### 3.1 Linking Logits to ES Fitness

We optimize the model by connecting its output logits to an Evolution Strategies (ES) fitness function that measures next-token predictive quality on a text corpus.

Let a tokenized sequence be x=(x 1,x 2,…,x T)x=(x_{1},x_{2},\dots,x_{T}) with vocabulary 𝒱\mathcal{V}. Given parameters θ\theta, the model produces logits z t∈ℝ|𝒱|z_{t}\in\mathbb{R}^{|\mathcal{V}|} at position t t. We convert logits to probabilities via softmax:

p θ​(x t+1∣x≤t)=softmax​(z t)​[x t+1]=exp⁡(z t,x t+1)∑v∈𝒱 exp⁡(z t,v).p_{\theta}(x_{t+1}\mid x_{\leq t})=\mathrm{softmax}(z_{t})[x_{t+1}]=\frac{\exp\big(z_{t,x_{t+1}}\big)}{\sum_{v\in\mathcal{V}}\exp\big(z_{t,v}\big)}.(1)

To handle padding and truncation, we introduce a binary mask m t∈{0,1}m_{t}\in\{0,1\} that excludes pad positions. The per-sample reward (fitness contribution) is the mean log-probability of the true next token:

R​(x;θ)=1∑t=1 T−1 m t​∑t=1 T−1 m t​log⁡p θ​(x t+1∣x≤t).R(x;\theta)=\frac{1}{\sum_{t=1}^{T-1}m_{t}}\sum_{t=1}^{T-1}m_{t}\,\log p_{\theta}(x_{t+1}\mid x_{\leq t}).(2)

Given a dataset 𝒟\mathcal{D}, we define the overall fitness as the average across samples:

F​(θ)=1|𝒟|​∑x∈𝒟 R​(x;θ).F(\theta)=\frac{1}{|\mathcal{D}|}\sum_{x\in\mathcal{D}}R(x;\theta).(3)

In ES, a perturbation ϵ\epsilon is sampled and applied to parameters as θ′=θ+σ​ϵ\theta^{\prime}=\theta+\sigma\,\epsilon, yielding a perturbed fitness

r​(ϵ)≜F​(θ+σ​ϵ)=1|𝒟|​∑x∈𝒟 R​(x;θ+σ​ϵ),r(\epsilon)\triangleq F(\theta+\sigma\,\epsilon)=\frac{1}{|\mathcal{D}|}\sum_{x\in\mathcal{D}}R\big(x;\theta+\sigma\,\epsilon\big),(4)

which directly ties the model’s logits—through p θ p_{\theta}—to the ES objective. In our implementation, we evaluate R​(x;θ)R(x;\theta) via log-softmax of shifted logits and label-gathering, and average rewards across the dataset, consistent with next-token causal language modeling.

### 3.2 Evolution Strategies and Our Extensions

#### Baseline ES Operator.

We adopt a standard ES estimator(Salimans et al., [2017](https://arxiv.org/html/2510.10603v2#bib.bib16)). Let ϵ j∼𝒩​(0,I)\epsilon_{j}\sim\mathcal{N}(0,I) for j=1,…,N j=1,\dots,N denote i.i.d. Gaussian noise samples. We evaluate perturbed fitness values

r j=F​(θ+σ​ϵ j).r_{j}=F\big(\theta+\sigma\,\epsilon_{j}\big).(5)

We build normalized weights w j w_{j} from r j r_{j} to form a gradient-like update. We support either z-score normalization

w j z\displaystyle w_{j}^{\text{z}}=r j−μ r σ r+10−8,\displaystyle=\frac{r_{j}-\mu_{r}}{\sigma_{r}+10^{-8}},(6)
μ r\displaystyle\mu_{r}=1 N​∑k=1 N r k,\displaystyle=\frac{1}{N}\sum_{k=1}^{N}r_{k},
σ r 2\displaystyle\sigma_{r}^{2}=1 N​∑k=1 N(r k−μ r)2\displaystyle=\frac{1}{N}\sum_{k=1}^{N}(r_{k}-\mu_{r})^{2}

or rank-based shaping

w j rank=2⋅rank​(r j)N−1−1∈[−1,1],w_{j}^{\text{rank}}=2\cdot\frac{\mathrm{rank}(r_{j})}{N-1}-1\in[-1,1],(7)

where rank​(⋅)\mathrm{rank}(\cdot) assigns ranks 0,…,N−1 0,\dots,N-1 after sorting.

The ES update then aggregates noise weighted by w j w_{j}:

g^=1 N​∑j=1 N w j​ϵ j.\hat{g}=\frac{1}{N}\sum_{j=1}^{N}w_{j}\,\epsilon_{j}.(8)

We apply either the standard scaling or the NES-style 1/σ 1/\sigma scaling:

θ←θ+α​g^(standard),θ←θ+α σ​g^(NES with 1/σ).\theta\leftarrow\theta+\alpha\,\hat{g}\quad\text{(standard)},\qquad\theta\leftarrow\theta+\frac{\alpha}{\sigma}\,\hat{g}\quad\text{(NES with $1/\sigma$)}.(9)

#### Subsampled Evaluation.

To reduce per-iteration computation, we evaluate fitness on a random subset of the dataset instead of all samples. At iteration t t, let 𝒟 t⊂𝒟\mathcal{D}_{t}\subset\mathcal{D} be a uniformly sampled subset with size M M (without replacement). We define the mini-batch fitness

F M​(θ)=1 M​∑x∈𝒟 t R​(x;θ),M≪|𝒟|.F_{M}(\theta)=\frac{1}{M}\sum_{x\in\mathcal{D}_{t}}R(x;\theta),\qquad M\ll|\mathcal{D}|.(10)

Perturbed evaluations use the same subset:

r j(t)=F M​(θ+σ​ϵ j),j=1,…,N.r_{j}^{(t)}=F_{M}\big(\theta+\sigma\,\epsilon_{j}\big),\qquad j=1,\dots,N.(11)

This yields a stochastic ES estimator that is unbiased under uniform sampling, i.e., 𝔼 𝒟 t​[F M​(θ)]=F​(θ)\mathbb{E}_{\mathcal{D}_{t}}[F_{M}(\theta)]=F(\theta), while substantially reducing wall-clock time and memory. In our implementation, the subset indices are broadcast across processes so all workers evaluate the same 𝒟 t\mathcal{D}_{t}, and antithetic sampling plus rank shaping further mitigates the variance introduced by sub-sampling.

Algorithm 1 EA4LLM: Zeroth-Order Pretraining via ES

Input: Pretrained LLM parameters

θ 0\theta_{0}
; token-level fitness

R​(x;θ)R(x;\theta)
via log-softmax; iterations

T T
; population size

N N
; noise scale

σ\sigma
; learning rate

α\alpha
; subset size

M M
; normalization (z-score or rank shaping).

for

t=1 t=1
to

T T
do

Sample a uniform subset

𝒟 t⊂𝒟\mathcal{D}_{t}\subset\mathcal{D}
with

|𝒟 t|=M|\mathcal{D}_{t}|=M
and broadcast indices across processes.

Draw

{ϵ j}j=1 N\{\epsilon_{j}\}_{j=1}^{N}
with

ϵ j∼𝒩​(0,I)\epsilon_{j}\sim\mathcal{N}(0,I)
.

for

j=1 j=1
to

N N
do

Form temporary parameters:

θ tmp=θ+σ​ϵ j\theta_{\text{tmp}}=\theta+\sigma\,\epsilon_{j}
.

Evaluate reward on the subset:

r j=1 M​∑x∈𝒟 t R​(x;θ tmp)r_{j}=\frac{1}{M}\sum_{x\in\mathcal{D}_{t}}R\big(x;\theta_{\text{tmp}}\big)
.

end for

Compute weights

w j w_{j}
from

{r j}\{r_{j}\}
by z-score normalization or rank shaping.

Aggregate noise:

g^=1 N​∑j=1 N w j​ϵ j\hat{g}=\frac{1}{N}\sum_{j=1}^{N}w_{j}\,\epsilon_{j}
.

Update parameters:

θ←θ+α​g^\theta\leftarrow\theta+\alpha\,\hat{g}
.

end for

Return:

θ T\theta_{T}
.

The pseudocode of our final algorithm is shown in Algorithm[1](https://arxiv.org/html/2510.10603v2#alg1 "Algorithm 1 ‣ Subsampled Evaluation. ‣ 3.2 Evolution Strategies and Our Extensions ‣ 3 Method ‣ EA4LLM: A Gradient-Free Approach to Large Language Model Optimization via Evolutionary Algorithms").

4 Experiment
------------

### 4.1 Experimental Setups

#### Global Hyperparameters and Setup.

Unless otherwise stated, we adopt the following configuration drawn from our experiment runner and training script:

*   •Population: P=30 P=30; noise scale: σ=10−3\sigma=10^{-3}; step size: α=5×10−4\alpha=5\times 10^{-4}. 
*   •Iterations: 1000 1000 with 5000 5000 evaluation samples per iteration; seeds are synchronized across processes. 
*   •Micro-batch: 16 16; max sequence length: 256 256; 
*   •Models: Qwen3 configurations spanning 0.5B–32B (see Appendix[A.4](https://arxiv.org/html/2510.10603v2#A1.SS4 "A.4 Model Configuration Details ‣ Appendix A Comparative Analysis: Differences and Significance ‣ EA4LLM: A Gradient-Free Approach to Large Language Model Optimization via Evolutionary Algorithms")). 
*   •Data sampling: up to 10,000 10{,}000 lines from a novel corpus. 
*   •Baseline toggles: antithetic off, 1/σ 1/\sigma scaling off, rank shaping off, grouped multipliers off; scheduler decays disabled (γ α=γ σ=1.0\gamma_{\alpha}=\gamma_{\sigma}=1.0, S α=S σ=0 S_{\alpha}=S_{\sigma}=0). 

### 4.2 Experimental Results

#### Cross-Scale Training Validation (0.5B–32B).

We pretrain Qwen3-style models from 0.5B to 32B with full-parameter updates and consistently observe stable loss reduction; the trend matches the loss curve shown earlier in Figure[1](https://arxiv.org/html/2510.10603v2#S0.F1 "Figure 1 ‣ EA4LLM: A Gradient-Free Approach to Large Language Model Optimization via Evolutionary Algorithms"), indicating ES effectively optimizes across model scales.

#### Memory Footprint Comparison: Evolutionary vs Gradient Optimization.

We compare the minimal GPU memory usage when training 0.5B–32B models using ES versus gradient-based pipelines. By avoiding backpropagation and the storage of gradients/optimizer states, ES exhibits lower memory usage across all scales; the relative advantage increases with model size. Moreover, as model parameters increase, the memory advantage becomes more pronounced; for models of 4B and above, ES-based pretraining reduces GPU memory usage by more than 4×4\times compared to gradient-based methods. See Figure[2](https://arxiv.org/html/2510.10603v2#S4.F2 "Figure 2 ‣ Memory Footprint Comparison: Evolutionary vs Gradient Optimization. ‣ 4.2 Experimental Results ‣ 4 Experiment ‣ EA4LLM: A Gradient-Free Approach to Large Language Model Optimization via Evolutionary Algorithms").

![Image 2: Refer to caption](https://arxiv.org/html/2510.10603v2/x2.png)

Figure 2: Minimal GPU memory usage across model sizes (0.5B–32B), comparing ES with gradient-based optimization. ES consistently uses less memory, with a larger margin as the model size increases.

#### Ablations: Evaluation Budget and Population Size.

On the 1.7B model, we study the effect of evaluation samples per iteration (1000/3000/5000/7000) and population size (20/30/40). Results match expectations: larger evaluation budgets and larger populations provide more accurate optimization signals, leading to more pronounced loss reduction. Considering time and performance constraints, in other experiments we primarily adopt evaluation budget 5000 and population size 30. See Figure[3](https://arxiv.org/html/2510.10603v2#S4.F3 "Figure 3 ‣ Ablations: Evaluation Budget and Population Size. ‣ 4.2 Experimental Results ‣ 4 Experiment ‣ EA4LLM: A Gradient-Free Approach to Large Language Model Optimization via Evolutionary Algorithms") and Figure[4](https://arxiv.org/html/2510.10603v2#S4.F4 "Figure 4 ‣ Ablations: Evaluation Budget and Population Size. ‣ 4.2 Experimental Results ‣ 4 Experiment ‣ EA4LLM: A Gradient-Free Approach to Large Language Model Optimization via Evolutionary Algorithms").

![Image 3: Refer to caption](https://arxiv.org/html/2510.10603v2/x3.png)

Figure 3: Different evaluation budgets (1.7B). Larger budgets yield more accurate optimization signals and stronger loss reduction.

![Image 4: Refer to caption](https://arxiv.org/html/2510.10603v2/x4.png)

Figure 4: Different population sizes (eval=5000). Larger populations yield more accurate optimization signals and stronger loss reduction.

5 Conclusion
------------

#### The Significance of Our Work

Our approach provides a practical, architecture-agnostic zeroth-order pretraining path for large language models by tying token-level log-softmax to an ES fitness (Section[3](https://arxiv.org/html/2510.10603v2#S3 "3 Method ‣ EA4LLM: A Gradient-Free Approach to Large Language Model Optimization via Evolutionary Algorithms")). We employ subsampled evaluation to reduce per-iteration cost while maintaining an unbiased estimator under uniform sampling, and aggregate perturbations via weights from either z-score or rank shaping. Experimentally (Section[4](https://arxiv.org/html/2510.10603v2#S4 "4 Experiment ‣ EA4LLM: A Gradient-Free Approach to Large Language Model Optimization via Evolutionary Algorithms")), ES delivers stable loss reduction from 0.5B to 32B with full-parameter updates, and achieves substantial memory advantages over gradient-based training: for models of 4B and above, ES-based pretraining reduces GPU memory usage by more than 4×4\times. Ablations on a 1.7B model confirm expected trends: larger evaluation budgets and populations provide more accurate optimization signals and stronger loss reduction; in practice, we primarily adopt an evaluation budget of 5000 and a population of 30.

#### Limitations and Future Work

While promising, zeroth-order ES pretraining has limitations. Variance can be higher than first-order gradients and performance is sensitive to population size N N, noise scale σ\sigma, learning rate α\alpha, and normalization choice; subsampling, though unbiased, introduces additional stochasticity. Sample efficiency may trail gradient-based methods because each iteration requires multiple forward evaluations. Scaling to extremely large models depends on efficient parameter perturbation and IO. In our experiments, certain variance-reduction and control mechanisms (e.g., antithetic sampling, NES-style 1/σ 1/\sigma scaling, grouped multipliers) were disabled; exploring these and hybrid pipelines (alternating ES and gradient steps) is a natural direction. Future work includes adaptive schedules for α\alpha and σ\sigma, improved variance reduction, and task-aware fitness shaping beyond next-token log-likelihood to better align with downstream objectives.

References
----------

*   Back (1996) Back, T. _Evolutionary Algorithms in Theory and Practice: Evolution Strategies, Evolutionary Programming, Genetic Algorithms_. Oxford University Press, 1996. 
*   Carmona et al. (2024) Carmona, V. I.S., Jiang, S., and Dong, B. How well can a genetic algorithm fine-tune transformer encoders? a first approach. In _Proceedings of the Fifth Workshop on Insights from Negative Results in NLP_, pp. 25–33, 2024. 
*   Chen et al. (2023) Chen, X., Liu, B., Yuan, R., Song, R., He, L., Wang, J., Wang, H., He, J., and Liu, T.-Y. Symbolic discovery of optimization algorithms. In _Proceedings of the 40th International Conference on Machine Learning (ICML)_, pp. 5288–5305. PMLR, 2023. 
*   Guo et al. (2025) Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. _arXiv preprint arXiv:2501.12948_, 2025. 
*   Guo et al. (2023) Guo, Q., Wang, R., Guo, J., Li, B., Song, K., Tan, X., Liu, G., Bian, J., and Yang, Y. Connecting large language models with evolutionary algorithms yields powerful prompt optimizers. _arXiv preprint arXiv:2309.08532_, 2023. 
*   Hansen (2003) Hansen, N. Reducing the building blocks of an EA. In _Proceedings of the 2003 conference on Genetic and evolutionary computation_, pp. 1420–1431. Springer-Verlag, 2003. 
*   Huang et al. (2025) Huang, B., Jiang, Y., Chen, M., Wang, Y., Chen, H., and Wang, W. When evolution strategy meets language models tuning. In _Proceedings of the 31st International Conference on Computational Linguistics_, pp. 5333–5344, 2025. 
*   Igel (2003) Igel, C. Neuroevolution for reinforcement learning using evolution strategies. In _Proceedings of the 2003 Congress on Evolutionary Computation_, pp. 1978–1985. IEEE, 2003. 
*   Jin et al. (2024) Jin, F., Liu, Y., and Tan, Y. Derivative-free optimization for low-rank adaptation in large language models. _IEEE/ACM Transactions on Audio, Speech, and Language Processing_, 2024. 
*   Johnson et al. (2019) Johnson, J., Douze, M., and Jégou, H. Billion-scale similarity search with gpus, 2019. 
*   Kingma & Ba (2015) Kingma, D.P. and Ba, J. Adam: A method for stochastic optimization. In _Proceedings of the 3rd International Conference on Learning Representations (ICLR)_, 2015. 
*   Lorenc & Neruda (2025) Lorenc, M. and Neruda, R. Utilizing novelty-based evolution strategies to train transformers in reinforcement learning, 2025. URL [https://arxiv.org/abs/2502.06301](https://arxiv.org/abs/2502.06301). 
*   Qiu et al. (2025) Qiu, X., Gan, Y., Hayes, C.F., Liang, Q., Meyerson, E., Hodjat, B., and Miikkulainen, R. Evolution strategies at scale: Llm fine-tuning beyond reinforcement learning. _arXiv preprint arXiv:2509.24372_, 2025. 
*   Rechenberg (1973) Rechenberg, I. Evolutionsstrategie : Optimierung technischer systeme nach prinzipien der biologischen evolution. 1973. 
*   Robbins & Monro (1951) Robbins, H. and Monro, S. A stochastic approximation method. _The annals of mathematical statistics_, pp. 400–407, 1951. 
*   Salimans et al. (2017) Salimans, T., Ho, J., Chen, X., Sidor, S., and Sutskever, I. Evolution strategies as a scalable alternative to reinforcement learning. _arXiv preprint arXiv:1703.03864_, 2017. 
*   Schulman et al. (2017) Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms, 2017. URL [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347). 
*   Sun et al. (2022a) Sun, T., He, Z., Qian, H., Zhou, Y., Huang, X., and Qiu, X. Bbtv2: Towards a gradient-free future with large language models. _arXiv preprint arXiv:2205.11200_, 2022a. 
*   Sun et al. (2022b) Sun, T., Shao, Y., Qian, H., Huang, X., and Qiu, X. Black-box tuning for language-model-as-a-service. In _International Conference on Machine Learning_, pp. 20841–20855. PMLR, 2022b. 
*   Zhang et al. (2008) Zhang, Q., Zhou, A., and Jin, Y. Rm-meda: A regularity model-based multiobjective estimation of distribution algorithm. _IEEE Transactions on Evolutionary Computation_, 12(1):41–63, 2008. 
*   Zhao et al. (2023) Zhao, J., Wang, Z., and Yang, F. Genetic prompt search via exploiting language model probabilities. In _Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence_, pp. 5296–5305, 2023. 
*   Zhou et al. (2011) Zhou, A., Qu, B.-Y., Li, H., Zhao, S.-Z., Suganthan, P.N., and Zhang, Q. Multiobjective evolutionary algorithms: A survey of the state of the art. _Swarm and evolutionary computation_, 1(1):32–49, 2011. 

Appendix
--------

Appendix A Comparative Analysis: Differences and Significance
-------------------------------------------------------------

### A.1 Differences from ES in Supervised Learning

Salimans et al. (2017) studied, on small supervised datasets (e.g., MNIST), the correlation between ES gradient estimates g t E​S g_{t}^{ES} obtained via random perturbations and the true gradients g t T​r​u​e g_{t}^{True} from backpropagation, using the mini-batch performance/loss as a black-box objective f​(θ+σ​ϵ)f(\theta+\sigma\epsilon). While that objective is computed from logits, it treats the loss as an opaque function and does not offer a full-parameter ES optimization of large-scale LLMs for pretraining.

Our approach explicitly reformulates the maximum-likelihood language modeling objective as ES fitness. Let the model outputs (logits) be z t z_{t}. We define

R​(x;θ)=1 T​∑t=1 T m t⋅log⁡softmax​(z t​(θ))y t,F​(θ)=1|𝒟|​∑x∈𝒟 R​(x;θ),R(x;\theta)=\frac{1}{T}\sum_{t=1}^{T}m_{t}\cdot\log\,\mathrm{softmax}\big(z_{t}(\theta)\big)_{y_{t}},\quad F(\theta)=\frac{1}{|\mathcal{D}|}\sum_{x\in\mathcal{D}}R(x;\theta),(12)

where m t m_{t} can be used for masking or weighting, and F​(θ)F(\theta) aligns with the standard log-likelihood objective in language modeling. We use a simple population ES with i.i.d. Gaussian perturbations ϵ j∼𝒩​(0,I)\epsilon_{j}\sim\mathcal{N}(0,I); for each iteration we evaluate F​(θ+σ​ϵ j)F(\theta+\sigma\epsilon_{j}), apply either z-score normalization or rank-based shaping to the rewards, estimate the ES direction, and update parameters with the core ES step θ←θ+α​g^\theta\leftarrow\theta+\alpha\,\hat{g}. This design removes antithetic sampling, grouped/layered operators, and step-size multipliers, matching the actual algorithm we employ for large-scale pretraining while remaining effective empirically.

### A.2 Differences from ES in the RL Stage

RL-stage ES and GRPO-style methods focus on task-specific rewards for fine-tuning and depend on reward design and sampling environments. In contrast, we apply ES _from pretraining_ to full-parameter optimization with fitness defined by the aggregation of token-level log-probabilities consistent with maximum likelihood, independent of external environments or rewards. This directly targets the pretraining objective and avoids reliance on RL-specific reward shaping.

### A.3 Differences from MeZO and Other Zeroth-Order Optimizers

MeZO is a memory-efficient zeroth-order optimizer that typically uses two function evaluations f​(θ±δ​u)f(\theta\pm\delta u) to approximate directional derivatives and then updates parameters in a first-order style, with the objective treated as a black-box mini-batch loss. In contrast, our method:

*   •uses _population-based_ ES with i.i.d. Gaussian perturbations, without antithetic pairs or grouped/layered operators; 
*   •integrates the _structured token-level log-softmax_ objective into ES fitness, rather than treating the loss as opaque; 
*   •stabilizes updates via _z-score_ or _rank-based_ reward normalization to reduce scale sensitivity and outlier effects; 
*   •demonstrates _cross-scale effectiveness_ (0.5B–32B) and a _memory footprint reduction_ of over 4×\times for models ≥\geq 4B during pretraining; in other experiments we primarily adopt evaluation budget 5000 and population size 30. 

Therefore, our method directly optimizes the structured probabilistic objective of language modeling with a simple ES update and practical normalization strategies, making zeroth-order pretraining viable at LLM scale.

### A.4 Model Configuration Details

We instantiate a series of Qwen3-style causal language models using the Transformers implementation (Qwen3Config) spanning 0.5B–32B parameters in our experiments. The key configuration settings for each size are summarized below.

Size Layers Hidden Heads KV Heads Head Dim Intermediate Max Pos Tie Emb Use Cache Vocab
0.5B 24 1024 8 4 128 4096 32768 true false 151936
1.7B 28 2048 16 8 128 6144 32768 true true 151936
4B 36 4096 32 8 128 9728 32768 true true 151936
8B 36 4096 32 8 128 12288 32768 false true 151936
14B 40 5120 40 8 128 17408 32768 false true 151936
32B 64 8192 64 8 128 25600 40960 false true 151936

Table 1: Qwen3-style model configurations used for ES pretraining experiments across 0.5B–32B.

