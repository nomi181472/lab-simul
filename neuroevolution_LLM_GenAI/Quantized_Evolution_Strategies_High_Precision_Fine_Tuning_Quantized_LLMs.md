Title: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost

URL Source: https://arxiv.org/html/2602.03120

Published Time: Wed, 04 Feb 2026 01:33:31 GMT

Markdown Content:
###### Abstract

Post-Training Quantization (PTQ) is essential for deploying Large Language Models (LLMs) on memory-constrained devices, yet it renders models static and difficult to fine-tune. Standard fine-tuning paradigms, including Reinforcement Learning (RL), fundamentally rely on backpropagation and high-precision weights to compute gradients. Thus they cannot be used on quantized models, where the parameter space is discrete and non-differentiable. While Evolution Strategies (ES) offer a backpropagation-free alternative, optimization of the quantized parameters can still fail due to vanishing or inaccurate gradient. This paper introduces Quantized Evolution Strategies (QES), an optimization paradigm that performs full-parameter fine-tuning directly in the quantized space. QES is based on two innovations: (1) it integrates accumulated error feedback to preserve high-precision gradient signals, and (2) it utilizes a stateless seed replay to reduce memory usage to low-precision inference levels. QES significantly outperforms the state-of-the-art zeroth-order fine-tuning method on arithmetic reasoning tasks, making direct fine-tuning for quantized models possible. It therefore opens up the possibility for scaling up LLMs entirely in the quantized space. The source code is available at [https://github.com/dibbla/Quantized-Evolution-Strategies](https://github.com/dibbla/Quantized-Evolution-Strategies).

Evolution Strategies, Fine-tuning, Large Language Models, Quantization

1 Introduction
--------------

![Image 1: Refer to caption](https://arxiv.org/html/2602.03120v1/figure/main_figure_updated_curve.png)

Figure 1: An overview of Quantized Evolution Strategies (QES). The goal is to optimize quantized LLMs directly on discrete parameter space in a memory-efficient manner. Error residuals are accumulated at each iteration until they reach a threshold for making a discrete change. QES achieves temporal equivalence to high-precision optimization trajectory, while maintaining a memory at the same level as inference-only quantized models.

The scaling of Large Language Models (LLMs) has unlocked emergent capabilities in mathematical reasoning, coding, and general problem-solving (Shao et al., [2024](https://arxiv.org/html/2602.03120v1#bib.bib1 "DeepSeekMath: pushing the limits of mathematical reasoning in open language models"); OpenAI and others, [2024](https://arxiv.org/html/2602.03120v1#bib.bib9 "OpenAI o1 system card"); Guo and others, [2025](https://arxiv.org/html/2602.03120v1#bib.bib5 "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning")). However, this performance comes at a significant computational cost. To mitigate the memory bottleneck of large-scale deployment, Post-Training Quantization (PTQ) has become standard practice. Techniques such as activation smoothing (e.g. SmoothQuant, Xiao et al., [2024](https://arxiv.org/html/2602.03120v1#bib.bib20 "SmoothQuant: accurate and efficient post-training quantization for large language models")) and layer-wise weight compression (GPTQ; Frantar et al., [2023](https://arxiv.org/html/2602.03120v1#bib.bib19 "GPTQ: accurate post-training quantization for generative pre-trained transformers"); AWQ; Lin et al., [2024](https://arxiv.org/html/2602.03120v1#bib.bib22 "AWQ: activation-aware weight quantization for LLM compression and acceleration")) make 3-4 bit precision inference possible with negligible performance degradation. It is then possible to deploy such models on consumer-grade hardware.

Whereas PQT democratizes inference, the model becomes essentially a static artifact. Any fine-tuning is typically done before the model is quantized, and requires massive, training-grade compute clusters that PTQ practitioners do not typically have. Methods that allow fine-tuning quantized models directly do not fare much better. For instance, Standard Quantization-Aware Training (QAT) relies on backpropagation and high-precision optimizer states, which often consume more memory than the model itself (Liu et al., [2024](https://arxiv.org/html/2602.03120v1#bib.bib23 "LLM-QAT: data-free quantization aware training for large language models"); Dettmers et al., [2023](https://arxiv.org/html/2602.03120v1#bib.bib21 "QLoA: efficient finetuning of quantized LLMs")). While Zeroth-Order (ZO) optimization and Evolution Strategies (ES) have emerged as memory-efficient alternatives to pre-quantization fine tuning (Malladi et al., [2024](https://arxiv.org/html/2602.03120v1#bib.bib11 "Fine-tuning language models with just forward passes"); Qiu et al., [2025](https://arxiv.org/html/2602.03120v1#bib.bib7 "Evolution strategies at scale: LLM fine-tuning beyond reinforcement learning"); Sarkar et al., [2025](https://arxiv.org/html/2602.03120v1#bib.bib8 "Evolution strategies at the hyperscale")), they are severely limitated in discrete parameter spaces. Backpropagation-free approaches like QuZO and others (Zhou et al., [2025](https://arxiv.org/html/2602.03120v1#bib.bib13 "QuZO: quantized zeroth-order fine-tuning for large language models"); Feng et al., [2024](https://arxiv.org/html/2602.03120v1#bib.bib14 "Stepping forward on the last mile")) have shown promise in Supervised Fine-Tuning (SFT) on quantized models, but struggle with reasoning tasks. The problem, identified in this paper as the stagnation problem, is that the discrete nature of the parameter space causes gradient signals to vanish, leading the optimization process to collapse. Moreover, inaccurate gradient due to the discretization errors severely reduces the optimization efficiency.

This paper introduces Quantized Evolution Strategies (QES; Figure[1](https://arxiv.org/html/2602.03120v1#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost")), a novel optimization paradigm designed to perform full-parameter fine-tuning directly on the quantized model. By bridging quantized evolution with signal processing principles, specifically Delta-Sigma modulation (Inose et al., [1962](https://arxiv.org/html/2602.03120v1#bib.bib32 "A telemetering system by code modulation - ⁢ΔΣ modulation"); Razavi, [2016](https://arxiv.org/html/2602.03120v1#bib.bib33 "The delta-sigma modulator [a circuit for all seasons]")), it proposes an accumulated error feedback mechanism that allows the optimizer to sense and traverse high-precision gradients even in ultra-low bit settings (e.g., four-bit integers denoted as INT4). This mechanism mirrors historical techniques in communication-efficient training, such as 1-bit stochastic gradient descent (Seide et al., [2014](https://arxiv.org/html/2602.03120v1#bib.bib34 "1-bit stochastic gradient descent and its application to data-parallel distributed training of speech dnns."); Ström, [2015](https://arxiv.org/html/2602.03120v1#bib.bib35 "Scalable distributed dnn training using commodity GPU cloud computing"); Karimireddy et al., [2019](https://arxiv.org/html/2602.03120v1#bib.bib26 "Error feedback fixes SignSGD and other gradient compression schemes")), but adapts them for the first time to the strict constraints of backpropagation-free quantized optimization. Furthermore, to address the memory overhead of tracking these error residuals, Stateless Seed Replay mechanism is introduced. It reconstructs optimization states on-the-fly reducing GPU memory requirements of QES to low-precision inference level.

The paper makes three main contributions:

1.   1.A Mechanism for Optimization in Quantized Space: QES enables fine-tuning of LLMs directly within their low-precision quantized weights, yet maintains high-precision learning dynamics. Unlike gradient-based methods that require differentiable operations, QES utilizes accumulated error feedback to ensure progress on low-precision non-differentiable landscapes. 
2.   2.Inference-Level Memory Footprint: By replacing the storage of high-precision optimizer states with stateless seed replay, QES reduces GPU memory consumption to the order of low-precision inference. This level allows full-parameter learning on hardware that previously could only accommodate quantized inference. 
3.   3.Overcoming Gradient Stagnation and Discretization inaccuracy: QES resolves two challenges for optimization in quantized space: (1) stagnation of learning due to small gradient signals, and (2) inaccurate parameter updates due to gradient discretization errors. 

Evaluated in the Countdown task (Gandhi et al., [2024](https://arxiv.org/html/2602.03120v1#bib.bib45 "Stream of search (SoS): learning to search in language"); Pan et al., [2025](https://arxiv.org/html/2602.03120v1#bib.bib28 "TinyZero")), QES significantly improves arithmetic reasoning in quantized baseline models, and does so significantly better than the state-of-the-art quantized fine tuning method QuZO. It therefore democratizes fine tuning, making it possible in low resource environments. The results also suggest a new avenue for scaling LLMs in the future by including more quantized parameters in the same amount of memory.

2 Related Work
--------------

In prior work, several methods have been developed for quantizing LLMs, for fine-tuning them without gradients, and for reducing errors resulting from quantization. These methods are reviewed in this section.

### 2.1 LLM Quantization and Low-Bit Fine-Tuning

To mitigate the memory bottleneck of large-scale deployment, Post-Training Quantization (PTQ) has become standard practice. Techniques such as activation smoothing (e.g., SmoothQuant; Xiao et al., [2024](https://arxiv.org/html/2602.03120v1#bib.bib20 "SmoothQuant: accurate and efficient post-training quantization for large language models"))) and layer-wise weight compression (GPTQ; Frantar et al., [2023](https://arxiv.org/html/2602.03120v1#bib.bib19 "GPTQ: accurate post-training quantization for generative pre-trained transformers"); AWQ; Lin et al., [2024](https://arxiv.org/html/2602.03120v1#bib.bib22 "AWQ: activation-aware weight quantization for LLM compression and acceleration")) enable 3-4 bit precision inference with negligible degradation in performance.

However, while PTQ excels at static inference, optimizing quantized models requires Quantization-Aware Training (QAT). Existing QAT approaches, such as LLM-QAT (Liu et al., [2024](https://arxiv.org/html/2602.03120v1#bib.bib23 "LLM-QAT: data-free quantization aware training for large language models")) or QLoRA (Dettmers et al., [2023](https://arxiv.org/html/2602.03120v1#bib.bib21 "QLoA: efficient finetuning of quantized LLMs")), rely heavily on first-order approximations. These methods need either Straight-Through Estimators (STE) to approximate gradients through non-differentiable steps or backpropagation through frozen weights into high-precision adapters. Consequently, they suffer from two critical limitations: (1) the backward pass requires dequantization, which in turen requires significant memory, and (2) STE is inherently unstable in deep networks. Therefore, optimization paradigms capable of performing full-parameter fine-tuning directly on the quantized model are needed.

### 2.2 Zeroth Order Fine-tuning for Quantized Models

Improving alignment and reasoning ability remains the central focus of LLM fine tuning. The standard approach is Reinforcement Learning with Human Feedback (RLHF), and it has been successful in general preference alignment (Ouyang et al., [2022](https://arxiv.org/html/2602.03120v1#bib.bib16 "Training language models to follow instructions with human feedback")), mathematical reasoning (Shao et al., [2024](https://arxiv.org/html/2602.03120v1#bib.bib1 "DeepSeekMath: pushing the limits of mathematical reasoning in open language models"); Yu et al., [2025](https://arxiv.org/html/2602.03120v1#bib.bib2 "DAPO: an open-source LLM reinforcement learning system at scale"); Liu et al., [2025](https://arxiv.org/html/2602.03120v1#bib.bib3 "Understanding r1-zero-like training: a critical perspective")), code generation (Gehring et al., [2024](https://arxiv.org/html/2602.03120v1#bib.bib4 "RLEF: grounding code LLMs in execution feedback with reinforcement learning"); Guo and others, [2025](https://arxiv.org/html/2602.03120v1#bib.bib5 "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning")), and general reasoning (OpenAI and others, [2024](https://arxiv.org/html/2602.03120v1#bib.bib9 "OpenAI o1 system card"); An and others, [2025](https://arxiv.org/html/2602.03120v1#bib.bib10 "Qwen3 technical report")). However, RLHF relies fundamentally on backpropagation, restricting its application to differentiable, full-precision architectures with significant memory.

Recently, Zeroth-Order (ZO) optimizer and Evolution Strategies (ES) have emerged as backpropagation-free alternatives. Building on simultaneous perturbation stochastic approximation (SPSA; Spall, [1992](https://arxiv.org/html/2602.03120v1#bib.bib27 "Multivariate stochastic approximation using a simultaneous perturbation gradient approximation")), the MeZO method (Malladi et al., [2024](https://arxiv.org/html/2602.03120v1#bib.bib11 "Fine-tuning language models with just forward passes")) pioneered the scaling of ZO-SGD (Spall, [2002](https://arxiv.org/html/2602.03120v1#bib.bib15 "Multivariate stochastic approximation using a simultaneous perturbation gradient approximation")) to billion-parameter models. Similarly, recent work has demonstrated that ES can match or exceed RL performance in similar high-dimensional spaces (Qiu et al., [2025](https://arxiv.org/html/2602.03120v1#bib.bib7 "Evolution strategies at scale: LLM fine-tuning beyond reinforcement learning")). However, these methods were originally designed for high-precision continuous parameters, and cannot be directly used on quantized models.

Nevertheless, the backpropagation-free paradigm offers a natural basis for quantized fine-tuning. Recent approaches like QuZO (Zhou et al., [2025](https://arxiv.org/html/2602.03120v1#bib.bib13 "QuZO: quantized zeroth-order fine-tuning for large language models")) and QZO (Feng et al., [2024](https://arxiv.org/html/2602.03120v1#bib.bib14 "Stepping forward on the last mile")) extended ZO optimizers to supervised fine-tuning of quantized models. However, so far their success in reasoning tasks has been limited; they struggle to capture sparse signals, often leading to no meaningful convergence. QES addresses this gap by connecting quantized evolution to signal processing principles.

### 2.3 Techniques for Reducing Discretization Errors

The fundamental challenge of approximating a continuous signal with discrete steps is well-studied in the context of Delta-Sigma (Δ​Σ\Delta\Sigma) modulation (Inose et al., [1962](https://arxiv.org/html/2602.03120v1#bib.bib32 "A telemetering system by code modulation - ⁢ΔΣ modulation"); Razavi, [2016](https://arxiv.org/html/2602.03120v1#bib.bib33 "The delta-sigma modulator [a circuit for all seasons]")). In this framework, the error introduced by taking a discrete step (i.e. quantization) is not discarded but accumulated and applied to subsequent steps. This mechanism, known as noise shaping in DAC design, ensures that while individual steps may be coarse, the time-averaged behavior of the system tracks the continuous gradient accurately.

In the domain of gradient compression, this classical principle can be instantiated as residual accumulation, or error feedback (Seide et al., [2014](https://arxiv.org/html/2602.03120v1#bib.bib34 "1-bit stochastic gradient descent and its application to data-parallel distributed training of speech dnns."); Ström, [2015](https://arxiv.org/html/2602.03120v1#bib.bib35 "Scalable distributed dnn training using commodity GPU cloud computing"); Karimireddy et al., [2019](https://arxiv.org/html/2602.03120v1#bib.bib26 "Error feedback fixes SignSGD and other gradient compression schemes")). These methods compress gradient information for efficiency but accumulate the resulting error for correction in later steps. Unlike Stochastic Rounding (Gupta et al., [2015](https://arxiv.org/html/2602.03120v1#bib.bib39 "Deep learning with limited numerical precision"); Zhou et al., [2025](https://arxiv.org/html/2602.03120v1#bib.bib13 "QuZO: quantized zeroth-order fine-tuning for large language models")), which achieves unbiased updates via probabilistic sampling but suffers from high variance, this method ensures convergence via deterministic accumulation of residuals. This process effectively ”shapes” the quantization noise of the optimization trajectory, making it possible to apply quantized evolution to fine-tuning without the instability of prior QAT methods.

3 Method
--------

The challenge of quantized optimization is described first, followed by the QES methods with accumulated error feedback and seed replay.

### 3.1 The Optimization Challenge

In LLM reasoning tasks, the objective is to maximize a reward function J​(𝐖)J(\mathbf{W}) defined over a reasoning task:

max 𝐖∈𝒲⁡𝔼​[J​(𝐖)].\max_{\mathbf{W}\in\mathcal{W}}\mathbb{E}[J(\mathbf{W})].(1)

Standard first-order optimization methods (e.g., SGD, Adam) cannot be applied to this setting for two reasons: (1) the quantization operator is non-differentiable, and (2) there is insufficient memory for storing high-precision gradients and optimizer states in many edge applications. Consequently, Evolution Strategies (ES) is used to estimate descent directions via parameter-space exploration.

In the standard continuous setting (Salimans et al., [2017](https://arxiv.org/html/2602.03120v1#bib.bib17 "Evolution strategies as a scalable alternative to reinforcement learning"); Qiu et al., [2025](https://arxiv.org/html/2602.03120v1#bib.bib7 "Evolution strategies at scale: LLM fine-tuning beyond reinforcement learning")), ES approximates the gradient of the expected reward ∇𝔼​[J​(𝐖)]\nabla\mathbb{E}[J(\mathbf{W})] using a population of N N perturbations. For a current parameter state 𝐖 t\mathbf{W}_{t}, the gradient estimate g^\hat{g} is computed as:

g^=1 N​σ​∑i=1 N F i⋅ϵ i,\hat{g}=\frac{1}{N\sigma}\sum_{i=1}^{N}F_{i}\cdot\epsilon_{i},(2)

where ϵ i∼𝒩​(𝟎,𝐈)\epsilon_{i}\sim\mathcal{N}(\mathbf{0},\mathbf{I}) is a random perturbation sampled from a standard normal distribution, σ\sigma is the standard deviation for scaling the perturbation, and F i=F​(𝐖 t+σ​ϵ i)F_{i}=F(\mathbf{W}_{t}+\sigma\epsilon_{i}) is the fitness (reward) corresponding to the perturbed weights. Note that fitness is thus the normalized reward score derived from J​(𝐖 t+σ​ϵ i)J(\mathbf{W}_{t}+\sigma\epsilon_{i}) to makre sure optimization is stable. Fitness is evaluated on a problem set according to the reinforcement learning from verifiable rewards (RLVR) framework (Shao et al., [2024](https://arxiv.org/html/2602.03120v1#bib.bib1 "DeepSeekMath: pushing the limits of mathematical reasoning in open language models")), where the model is queried and assessed based on whether its responses are correct.

Standard ES updates the parameters via gradient ascent: 𝐖 t+1←𝐖 t+α​g^\mathbf{W}_{t+1}\leftarrow\mathbf{W}_{t}+\alpha\hat{g} with learning rate α\alpha. While effective in continuous spaces at large scale (Qiu et al., [2025](https://arxiv.org/html/2602.03120v1#bib.bib7 "Evolution strategies at scale: LLM fine-tuning beyond reinforcement learning")), this update rule fails in quantized spaces due to the vanishing magnitude of the update steps relative to the discretization granularity, or errors introduced during discretization. The solution is to provide error feedback in an accumulated manner, as will be described next.

### 3.2 Accumulated Error Feedback

QES optimizes quantized LLMs directly within the discrete integer space 𝐖∈{0,…,2 B−1}d\mathbf{W}\in\{0,\dots,2^{B}-1\}^{d}, where d d is the number of model parameters and B B is the number of bits to represent each parameter, Thus, the parameters 𝐖\mathbf{W} lie on a low-precision lattice 𝒲 Q\mathcal{W}_{Q} (e.g., INT4) defined by 𝒬​(⋅)\mathcal{Q}(\cdot). Since the continuous perturbation σ​ϵ\sigma\epsilon from Eq. [2](https://arxiv.org/html/2602.03120v1#S3.E2 "Equation 2 ‣ 3.1 The Optimization Challenge ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost") violates quantization constraints, a stochastic perturbation strategy is adapted from prior work (Connolly et al., [2021](https://arxiv.org/html/2602.03120v1#bib.bib42 "Stochastic rounding and its probabilistic backward error analysis"); Zhou et al., [2025](https://arxiv.org/html/2602.03120v1#bib.bib13 "QuZO: quantized zeroth-order fine-tuning for large language models")).

A discrete perturbation 𝜹\boldsymbol{\delta} is created by stochastically rounding the scaled Gaussian noise σ​ϵ\sigma\epsilon (where ϵ∼𝒩​(𝟎,𝐈)\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})):

𝜹=⌊σ​ϵ⌋+𝐛,where​𝐛∼Bernoulli​(σ​ϵ−⌊σ​ϵ⌋).\boldsymbol{\delta}=\lfloor\sigma\epsilon\rfloor+\mathbf{b},\quad\text{where }\mathbf{b}\sim\text{Bernoulli}(\sigma\epsilon-\lfloor\sigma\epsilon\rfloor).(3)

By recording the random seeds used to generate ϵ\epsilon and 𝐛\mathbf{b}, the δ\delta can be reproduced during optimization without the memory overhead of stored perturbation vectors (Qiu et al., [2025](https://arxiv.org/html/2602.03120v1#bib.bib7 "Evolution strategies at scale: LLM fine-tuning beyond reinforcement learning"); Salimans et al., [2017](https://arxiv.org/html/2602.03120v1#bib.bib17 "Evolution strategies as a scalable alternative to reinforcement learning")). To strictly enforce the codebook limits defined by 𝒲 Q\mathcal{W}_{Q}, boundary gating is applied to mask invalid updates:

𝐖 i​j′={𝐖 i​j+δ i​j if​0≤𝐖 i​j+δ i​j<2 B,𝐖 i​j otherwise.\mathbf{W}^{\prime}_{ij}=\begin{cases}\mathbf{W}_{ij}+\delta_{ij}&\text{if }0\leq\mathbf{W}_{ij}+\delta_{ij}<2^{B},\\ \mathbf{W}_{ij}&\text{otherwise}.\end{cases}(4)

Algorithm 1 QES with Accumulated Error Feedback

1:Input: Integer Weights

𝐖 0∈{0,…,2 B−1}d\mathbf{W}_{0}\in\{0,\dots,2^{B}-1\}^{d}
, Learning Rate

α\alpha
, Decay

γ\gamma
, Population

N N
, Maximum Iteration Number

T T

2:Initialize: Residuals

𝐞 0←𝟎\mathbf{e}_{0}\leftarrow\mathbf{0}
(FP16)

3:for

t=0 t=0
to

T−1 T-1
do

4:for each perturbation

i∈{1,…,N}i\in\{1,\dots,N\}
in parallel do

5: Reconstruct

δ i\delta_{i}
using Eq. [3](https://arxiv.org/html/2602.03120v1#S3.E3 "Equation 3 ‣ 3.2 Accumulated Error Feedback ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost") and seed

s i s_{i}

6: Apply perturbation:

𝐖′←Gate​(𝐖 t+δ i)\mathbf{W}^{\prime}\leftarrow\text{Gate}(\mathbf{W}_{t}+\delta_{i})

7: Execute inference and compute reward

F i F_{i}

8:end for

9: Normalize reward for population

10: Estimate gradient

g^t←1 N​σ​∑i=1 N F i⋅δ i\hat{g}_{t}\leftarrow\frac{1}{N\sigma}\sum_{i=1}^{N}F_{i}\cdot\delta_{i}

11:

𝐮 t←α​g^t+γ​𝐞 t\mathbf{u}_{t}\leftarrow\alpha\hat{g}_{t}+\gamma\mathbf{e}_{t}
// Apply accumulated error

12:

Δ​𝐖 t←Round​(𝐮 t)\Delta\mathbf{W}_{t}\leftarrow\text{Round}(\mathbf{u}_{t})
// Discretize update

13:

𝐞 t+1←𝐮 t−Δ​𝐖 t\mathbf{e}_{t+1}\leftarrow\mathbf{u}_{t}-\Delta\mathbf{W}_{t}
// Update accumulated error

14:

𝐖 t+1←𝐖 t+Δ​𝐖 t\mathbf{W}_{t+1}\leftarrow\mathbf{W}_{t}+\Delta\mathbf{W}_{t}

15:end for

The gradient direction g^\hat{g} is then estimated by aggregating the discrete search directions weighted by their rewards:

g^=1 N​σ​∑i=1 N F i⋅δ i.\hat{g}=\frac{1}{N\sigma}\sum_{i=1}^{N}F_{i}\cdot\delta_{i}.(5)

A critical challenge can be seen in Equation[5](https://arxiv.org/html/2602.03120v1#S3.E5 "Equation 5 ‣ 3.2 Accumulated Error Feedback ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"): The scaled update step α​g^\alpha\hat{g} is often smaller than the minimum discretization gap of the parameter lattice (i.e., ‖α​g^‖∞<1\|\alpha\hat{g}\|_{\infty}<1 for integer weights). Naively rounding this update either results in Δ​𝐖=𝟎\Delta\mathbf{W}=\mathbf{0}, causing optimization to stagnate, or inaccurate gradient, reducing optimization efficiency.

Interestingly, this challenge is well known in signal processing. The standard solution, Delta-Sigma (Δ​Σ\Delta\Sigma) modulation (Inose et al., [1962](https://arxiv.org/html/2602.03120v1#bib.bib32 "A telemetering system by code modulation - ⁢ΔΣ modulation"); Razavi, [2016](https://arxiv.org/html/2602.03120v1#bib.bib33 "The delta-sigma modulator [a circuit for all seasons]")), employs a feedback loop to accumulate quantization error over time, preserving signal fidelity. This approach is already used in communication-efficient training, such as 1-bit SGD, where uncompressed errors are carried forward to ensure convergence (Seide et al., [2014](https://arxiv.org/html/2602.03120v1#bib.bib34 "1-bit stochastic gradient descent and its application to data-parallel distributed training of speech dnns."); Ström, [2015](https://arxiv.org/html/2602.03120v1#bib.bib35 "Scalable distributed dnn training using commodity GPU cloud computing"); Karimireddy et al., [2019](https://arxiv.org/html/2602.03120v1#bib.bib26 "Error feedback fixes SignSGD and other gradient compression schemes")). It can be used to solve the stagnation problem and preserve gradient accuracy in QES.

In QES, a high-precision error vector 𝐞 t\mathbf{e}_{t} (typically FP16) that accumulates the quantization error from previous steps is maintained. Rather than discarding the fractional component of the update, it is carried forward. The update dynamics at step t t are thus defined as:

𝐮 t\displaystyle\mathbf{u}_{t}=α​g^t+γ​𝐞 t−1\displaystyle=\alpha\hat{g}_{t}+\gamma\mathbf{e}_{t-1}(6)
Δ​𝐖 t\displaystyle\Delta\mathbf{W}_{t}=Round​(𝐮 t)\displaystyle=\text{Round}(\mathbf{u}_{t})(7)
𝐞 t\displaystyle\mathbf{e}_{t}=𝐮 t−Δ​𝐖 t,\displaystyle=\mathbf{u}_{t}-\Delta\mathbf{W}_{t},(8)

where 𝐮 t\mathbf{u}_{t} represents the desired high-precision update, and γ∈(0,1]\gamma\in(0,1] is a decay factor that stabilizes the history. This mechanism allows infinitesimal gradient signals to accumulate in 𝐞\mathbf{e} over multiple iterations until they cross the rounding threshold (|𝐮|≥0.5|\mathbf{u}|\geq 0.5), effectively simulating a lower learning rate on the integer lattice. The full quantized evolution strategy with accumulated error feedback is listed in Algorithm [1](https://arxiv.org/html/2602.03120v1#alg1 "Algorithm 1 ‣ 3.2 Accumulated Error Feedback ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost").

Algorithm 2 Stateless QES Update with Seed Replay

1:Input: Current Weights

𝐖 t\mathbf{W}_{t}
, New Seeds

S t S_{t}
, New Rewards

𝐅 t\mathbf{F}_{t}
, History

ℋ\mathcal{H}
, Window

K K
, Sigma

σ\sigma

2:Output: Updated

𝐖 t+1\mathbf{W}_{t+1}
, New History

ℋ′\mathcal{H}^{\prime}

3: Initialize proxy residual

𝐞~←𝟎\tilde{\mathbf{e}}\leftarrow\mathbf{0}

4:for

(S^,𝐅^)∈ℋ(\hat{S},\hat{\mathbf{F}})\in\mathcal{H}
do

5: Re-generate noise with history seed

ϵ←RNG​(S^)\epsilon\leftarrow\text{RNG}(\hat{S})

6: Re-compute grad

g^←Agg​(ϵ,𝐅^,σ)\hat{g}\leftarrow\text{Agg}(\epsilon,\hat{\mathbf{F}},\sigma)

7:

𝐮 s​i​m←α​g^+γ​𝐞~\mathbf{u}_{sim}\leftarrow\alpha\hat{g}+\gamma\tilde{\mathbf{e}}

8:

Δ s​i​m←Round​(𝐮 s​i​m)\Delta_{sim}\leftarrow\text{Round}(\mathbf{u}_{sim})

9:

Δ f​i​n​a​l←Gate​(𝐖 t+Δ s​i​m)\Delta_{final}\leftarrow\text{Gate}(\mathbf{W}_{t}+\Delta_{sim})

10:

𝐞~←𝐮 s​i​m−Δ f​i​n​a​l\tilde{\mathbf{e}}\leftarrow\mathbf{u}_{sim}-\Delta_{final}
// Update proxy error

11:end for

12: Generate current noise

ϵ t←RNG​(S t)\epsilon_{t}\leftarrow\text{RNG}(S_{t})

13:

g^t←Agg​(ϵ t,𝐅 t,σ)\hat{g}_{t}\leftarrow\text{Agg}(\epsilon_{t},\mathbf{F}_{t},\sigma)

14:

𝐮 t←α​g^t+γ​𝐞~\mathbf{u}_{t}\leftarrow\alpha\hat{g}_{t}+\gamma\tilde{\mathbf{e}}
// Add rematerialized error

15:

Δ​𝐖←Round​(𝐮 t)\Delta\mathbf{W}\leftarrow\text{Round}(\mathbf{u}_{t})

16:

𝐖 t+1←Gate​(𝐖 t+Δ​𝐖)\mathbf{W}_{t+1}\leftarrow\text{Gate}(\mathbf{W}_{t}+\Delta\mathbf{W})

17:

ℋ′←Enqueue​(ℋ,(S t,𝐅 t))\mathcal{H}^{\prime}\leftarrow\text{Enqueue}(\mathcal{H},(S_{t},\mathbf{F}_{t}))

18:return

𝐖 t+1,ℋ′\mathbf{W}_{t+1},\mathcal{H}^{\prime}

Table 1: Countdown Task Accuracy (%). Comparison of QES against the original base models, state-of-the-art quantized zeroth-order method QuZO, and the variant with full residuals. The QuZO results were obtained with the best-performing configuration found through hyperparameter search. Whereas QuZO struggled to improve significantly over the base models especially in low-bit settings (INT4), QES improved the reasoning capabilities of the base models significantly across all quantization methods and model scales. Its performance was only slightly lower than with full residuals, indicating that the approximation method is effective. Thus,… 

### 3.3 Stateless Error Tracking via Seed Replay

While Algorithm [1](https://arxiv.org/html/2602.03120v1#alg1 "Algorithm 1 ‣ 3.2 Accumulated Error Feedback ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost") ensures progress, it incurs a significant memory penalty. Storing the dense high-precision error vector 𝐞 t∈ℝ d\mathbf{e}_{t}\in\mathbb{R}^{d} (typically FP16) often consumes more VRAM than the quantized model weights themselves, canceling the primary advantage of quantization.

To resolve this bottleneck, QES includes Stateless Error Tracking. Note that the error state 𝐞 t\mathbf{e}_{t} is deterministic given the initial condition and the history of optimization steps. Rather than persisting 𝐞 t\mathbf{e}_{t}, it can be rematerialized on-the-fly by replaying a limited history window. To do that, a lightweight history buffer ℋ t={(S τ,𝐅 τ)}τ=t−K t−1\mathcal{H}_{t}=\{(S_{\tau},\mathbf{F}_{\tau})\}_{\tau=t-K}^{t-1} is maintained, containing only the random seeds S S and the scalar rewards 𝐅\mathbf{F} for the past K K steps’ populations. To perform an update at step t t, the error accumulation process is re-simulated starting from an assumed zero error state at step t−K t-K. Since the decay factor γ∈(0,1)\gamma\in(0,1), the contribution of errors from steps τ<t−K\tau<t-K vanishes exponentially (γ K≈0\gamma^{K}\approx 0).

During the replay of step τ\tau, boundary constraints are checked using the current weights 𝐖 t\mathbf{W}_{t} rather than reconstructing the historical weights 𝐖 τ\mathbf{W}_{\tau}. Since discrete updates are sparse, 𝐖 τ≈𝐖 t\mathbf{W}_{\tau}\approx\mathbf{W}_{t}, and the discrepancy in boundary masking is minimal.

This approach, detailed in Algorithm [2](https://arxiv.org/html/2602.03120v1#alg2 "Algorithm 2 ‣ 3.2 Accumulated Error Feedback ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), trades computation for memory. By performing K K additional reconstruction operations per update, the optimizer state memory complexity is reduced from O​(d)O(d) to O​(K)O(K). This tradeoff is highly favorable because K K is orders of magnitude smaller than the parameter dimension d d. In a typical LLM optimization scenario, d d may exceed 10 9 10^{9}, whereas a short window of K≈50 K\approx 50 is sufficient to recover the accumulated error. For standard values (e.g., γ=0.9,K=50\gamma=0.9,K=50), the influence introduced by truncating history is negligible.

4 Experiments
-------------

QES was evaluated experimentally on arithmetic reasoning under strict memory constraints. This setup aims to demonstrate that the method can effectively fine-tune LLMs directly in their quantized versions without the need for full-precision gradients or auxiliary memory for optimizer states. The experimental setup is described first, followed by the main results on performance, and an analysis of accumulation hyperparameters and the fidelity of the stateless seed replay mechanism.

### 4.1 Experimental Setup

Countdown (Pan et al., [2025](https://arxiv.org/html/2602.03120v1#bib.bib28 "TinyZero")) is a compact reasoning task that is still challenging for LLMs. Given a set of source numbers, the model must generate a valid arithmetic expression using the operators +-*/ that equals the given target number. For instance, given the numbers 3, 4, and 52, and the target 44, the solution is 28 + 52/4 + 3 = 44. Performance was measured based on whether the expression is correct. The same prompt for the LLMs was used as in prior experiments with GRPO-Zero (Policy-gradient, [2025](https://arxiv.org/html/2602.03120v1#bib.bib41 "GRPO-Zero: implementing deepseek r1’s grpo algorithm from scratch")).

Qwen2.5 models (Qwen et al., [2025](https://arxiv.org/html/2602.03120v1#bib.bib29 "Qwen2.5 technical report")) were used as the base LLM, quantized into INT4, INT8 formats using GPTQ (Frantar et al., [2023](https://arxiv.org/html/2602.03120v1#bib.bib19 "GPTQ: accurate post-training quantization for generative pre-trained transformers")), and to W8A8 format using LLM-Compressor (RedHatAI and vLLM Project, [2024](https://arxiv.org/html/2602.03120v1#bib.bib30 "LLM compressor")). These formats represent varying degrees of precision and hardware compatibility.

QES on these quantized models was then compared against the following systems:

*   •Base Model: The pre-trained quantized model without fine-tuning (a zero-shot baseline). 
*   •QuZO(Zhou et al., [2025](https://arxiv.org/html/2602.03120v1#bib.bib13 "QuZO: quantized zeroth-order fine-tuning for large language models")): A quantized zeroth-order fine-tuning method, serving as the primary comparison to the state-of-the-art in quantized fine tuning. 
*   •Full Residual: A variant of QES that stores full-precision (FP16) residuals, to measure the performance impact without the memory-saving Seed Replay optimization. 

In all experiments, QES was run for 300 generations.

![Image 2: Refer to caption](https://arxiv.org/html/2602.03120v1/figure/reward_mean_grid_2x3_1.5B_over_3B_updated.png)

Figure 2: Training curves for QUZO, QES, and Full-Residual QES compared to the Base Model. QuZO (Orange) performance is unstable and training collapses especially in the coarser INT4 landscape and with the smaller base model. In contrast, QES (Green) progresses steadily closely tracking the Full Residual Oracle (Blue) despite using significantly less memory.

### 4.2 Performance

Table [1](https://arxiv.org/html/2602.03120v1#S3.T1 "Table 1 ‣ 3.2 Accumulated Error Feedback ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost") summarizes the performance of QES compared to baselines across different model sizes and quantization formats. As mentioned in Section[3.2](https://arxiv.org/html/2602.03120v1#S3.SS2 "3.2 Accumulated Error Feedback ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), the primary challenge in quantized optimization is the loss of information caused by the discrete parameter landscape. This problem is seen in Figure [2](https://arxiv.org/html/2602.03120v1#S4.F2 "Figure 2 ‣ 4.1 Experimental Setup ‣ 4 Experiments ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), where the Zeroth-Order method QuZO struggled to escape the initial performance plateau in all cases. For instance in INT4 with Qwen2.5-1.5B, it achieved only marginal gains over the base model, from 3.50% to 5.25% correct answers. In contrast, QES successfully navigated this landscape, achieving a decisive performance improvement to 18.00%. This trend held for the larger 3B model as well, where QES more than doubled the performance of the base model (from 14.25% to 31.85%). This result empirically validates that the error-accumulation mechanism is essential for learning when the quantization is coarse.

QuZO’s performance depends strongly on the size of the base model. As shown in Table [1](https://arxiv.org/html/2602.03120v1#S3.T1 "Table 1 ‣ 3.2 Accumulated Error Feedback ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), while QuZO nearly stagnated on the 1.5B model (improving INT4 accuracy by only 1.75%), it managed to achieve meaningful gains on the larger 3B model (improving INT4 accuracy by 11.45%). This discrepancy aligns with the general intuition that smaller quantized models are significantly harder to optimize than their larger counterparts (Li et al., [2018](https://arxiv.org/html/2602.03120v1#bib.bib40 "Visualizing the loss landscape of neural nets")). Smaller models possess fewer redundant parameters and sharper loss landscapes, making the search for valid descent directions on a discrete lattice far more brittle. In contrast, QES demonstrates robustness across scales, achieving high performance even with the 1.5B model.

### 4.3 Effect of Accumulation Hyperparameters

Table[2](https://arxiv.org/html/2602.03120v1#S4.T2 "Table 2 ‣ 4.3 Effect of Accumulation Hyperparameters ‣ 4 Experiments ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost") evaluates the impact of two crucial accumulation hyperparameters, the replay window size (K K) and decay factor (γ\gamma). Two regimes were compared: one where γ\gamma was scaled such that historical errors effectively vanished within the replay horizon (γ K≈0\gamma^{K}\approx 0), and another where γ\gamma was held constant at 0.90 0.90.

When γ\gamma was scaled to make the residual to vanish, performance was highly sensitive to the decay rate. With a window of K=50 K=50 (γ=0.90\gamma=0.90), the method achieved a 16.00% accuracy. However, shrinking the window to K=10 K=10 needed an aggressive decay (γ=0.58\gamma=0.58), causing performance to collapse to 4.55%.

The results with a fixed decay of γ=0.90\gamma=0.90 revealed that this collapse was not due to the short window itself. Even at K=10 K=10, maintaining a strong memory (γ=0.90\gamma=0.90) retained a robust accuracy of 13.05%, despite the reconstruction error for residuals introduced by truncating the history.

Table 2: Impact of Accumulation Hyperparameters on Performance. In the top table, the decay ratio γ\gamma scales with the window size K K); in the bottom table, it is fixed at 0.90 for all window sizes.

### 4.4 Fidelity of Stateless Seed Replay

In Stateless Seed Replay, the perturbation noise ϵ\epsilon is re-generated accurately from the preserved seed. However, the boundary gating conditions are created using the current weights W t W_{t} rather than the historical weights W τ W_{\tau}. In this section, the accuracy of this approximation is evaluated empirically.

A reconstruction error—where the replayed optimization path diverges from the original—can only occur if the gating status of a parameter changes. This change requires a parameter to actively cross a quantization boundary within the replay window. Crucially, for each parameter update step, an update ratio can be defined as the ratio of parameters that are changed, and the boundary hit ratio (ρ\rho) as the proportion of these changing parameters that hit a quantization boundary. As shown in Table [3](https://arxiv.org/html/2602.03120v1#S4.T3 "Table 3 ‣ 4.4 Fidelity of Stateless Seed Replay ‣ 4 Experiments ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), the total ratio of updated parameters is consistently small (≈10−2\approx 10^{-2} across all formats). In addition, ρ\rho is negligible across all formats (e.g., <1×10−5<1\times 10^{-5} for INT4). This observation indicates that even within the small subset of parameters that actually change, virtually none are located at the boundary where the state approximation would fail.

Consequently, the intersection of these two events, i.e. an active update occurring precisely at a boundary, is exceedingly rare. As a result, the Stateless Seed Replay mechanism tracks the performance of the memory-heavy Full Residual variant (listed in Table [1](https://arxiv.org/html/2602.03120v1#S3.T1 "Table 1 ‣ 3.2 Accumulated Error Feedback ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost")) with near-perfect fidelity.

Table 3: Update Ratio and Boundary Hit Ratio ρ\rho in Stateless Seed Replay. The proportion of parameters updated per step alongside the fraction of those updates that encounter quantization boundaries. The updates are consistenly sparse (≈10−2\approx 10^{-2}) and the hit ratio negligible across all formats. Since reconstruction errors only occur when these conditions coincide, the error in Stateless Seed replay is vanishingly small.

### 4.5 Accelerating State Reconstruction

While Stateless Seed Replay only uses negligible memory for saving the history (i.e. a few KB), its computational cost can also be flexibly managed. Reconstruction of the accumulated error feedback scales linearly with the window size K K. As demonstrated in Table [2](https://arxiv.org/html/2602.03120v1#S4.T2 "Table 2 ‣ 4.3 Effect of Accumulation Hyperparameters ‣ 4 Experiments ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), reducing the window size from K=50 K=50 to K=20 K=20 reduces reconstruction cost by 60% while largely preserving performance (i.e. 14.75% instead of 16.00%, with fixed decay). This result demonstrates that the replay horizon acts as a tunable parameter, allowing users to trade marginal accuracy for significant gains in training throughput.

Furthermore, since the model is not serving inference requests during the update phase, the reconstruction process can be parallelized. One can potentially reconstruct multiple layers simultaneously by leveraging otherwise idle memory resources, such as offloaded KV caches or prefixing caches, to further mitigate the computational impact of the replay mechanism.

5 Temporal Equivalence to Continuous Optimization
-------------------------------------------------

To understand why QES succeeds while other quantized methods stagnate or converge slowly, it is useful to characterize QES’s optimization dynamics relative to an ideal underlying continuous model. While the quantized parameters W W are constrained to a discrete grid 𝒲 Q\mathcal{W}_{Q} with spacing Δ\Delta, the reward function J​(W)J(W) is defined over a continuous domain. For instance, assume a Gaussian smoothed objective J σ​(Θ)=𝔼 ϵ∼𝒩​(0,I)​[J​(Θ+σ​ϵ)]J_{\sigma}(\Theta)=\mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)}[J(\Theta+\sigma\epsilon)] (Figure [3](https://arxiv.org/html/2602.03120v1#S5.F3 "Figure 3 ‣ 5 Temporal Equivalence to Continuous Optimization ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost")). It can be optimized by high-precision gradient ascent, but such a mechanism is not available on the discrete grid. Instead, the gradient needs to be approximated.

![Image 3: Refer to caption](https://arxiv.org/html/2602.03120v1/figure/bed_of_nailsa.png)

![Image 4: Refer to caption](https://arxiv.org/html/2602.03120v1/figure/bed_of_nailsb.png)

Figure 3: A continuous reward function and its instantiation on a discrete grid. While the reward function can be optimized with a high-precision gradient ascent, that method cannot be applied to the discrete case. Instead, the gradient needs to be estimated without vanishing signals or inaccurate moves between grid points. That is the challenge that QES is designed to solve.

One approach is to use a stochastic gradient estimator g^t\hat{g}_{t}, like QuZO does (Zhou et al., [2025](https://arxiv.org/html/2602.03120v1#bib.bib13 "QuZO: quantized zeroth-order fine-tuning for large language models")). Utilizing stochastic perturbation (double quantization) yields an unbiased estimator of the smoothed gradient: 𝔼​[g^t]=∇J σ​(W t)\mathbb{E}[\hat{g}_{t}]=\nabla J_{\sigma}(W_{t}). However, while an unbiased gradient is necessary, it is not sufficient for high performance on a discrete grid. The critical failure occurs at the application of the gradient update.

Consider the standard update rule at step t t with learning rate α\alpha and a quantization operator 𝒬\mathcal{Q}:

W t+1=W t+𝒬​(α​g^t).W_{t+1}=W_{t}+\mathcal{Q}(\alpha\hat{g}_{t}).(9)

The quantization operator can be decomposed into the identity plus an error term ξ t\xi_{t}, such that 𝒬​(x)=x+ξ t​(x)\mathcal{Q}(x)=x+\xi_{t}(x). Expanding the trajectory over T T steps yields

W T=W 0+∑t=0 T−1 α​g^t⏟Ideal Continuous Update+∑t=0 T−1 ξ t.⏟Accumulated Quantization Loss W_{T}=W_{0}+\underbrace{\sum_{t=0}^{T-1}\alpha\hat{g}_{t}}_{\text{\tiny Ideal Continuous Update}}+\underbrace{\sum_{t=0}^{T-1}\xi_{t}.}_{\text{\tiny Accumulated Quantization Loss}}(10)

The mechanisms behind the failure of stateless updates can be seen in this equation. First, stagnation occurs because the update magnitude in fine-tuning is often smaller than the grid precision (‖α​g^t‖∞<Δ/2\|\alpha\hat{g}_{t}\|_{\infty}<\Delta/2). Consequently, 𝒬​(u)→0\mathcal{Q}(u)\to 0, implying the error is ξ t=−α​g^t\xi_{t}=-\alpha\hat{g}_{t}; thus, the Accumulated Quantization Loss exactly cancels the Ideal Continuous Update, resulting in W T=W 0 W_{T}=W_{0}. Second, variance explosion arises if 𝒬\mathcal{Q} is stochastic. In this case, ξ t\xi_{t} becomes a zero-mean noise variable with standard deviation proportional to Δ\Delta. These errors accumulate as a random walk, scale with T​Δ\sqrt{T}\Delta and create a noise floor that drowns out the subtle fine-tuning signal in the long run.

QES solves this problem by introducing a residual state e t e_{t} to enforce temporal equivalence between the discrete and continuous domains. First, Virtual Continuous Parameters Θ t\Theta_{t} are defined as the sum of the physical discrete weights and the carry-over residual error:

Θ t≜W t+e t.\Theta_{t}\triangleq W_{t}+e_{t}.(11)

Then, expanding this equation with the QES update rules in Equations[6](https://arxiv.org/html/2602.03120v1#S3.E6 "Equation 6 ‣ 3.2 Accumulated Error Feedback ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost")–[8](https://arxiv.org/html/2602.03120v1#S3.E8 "Equation 8 ‣ 3.2 Accumulated Error Feedback ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost") gives

Θ t+1\displaystyle\Theta_{t+1}=W t+1+e t+1\displaystyle=W_{t+1}+e_{t+1}(12)
=(W t+Δ​W t)+(u t−Δ​W t)\displaystyle=(W_{t}+\Delta W_{t})+(u_{t}-\Delta W_{t})
=W t+u t=W t+(α​g^t+e t)\displaystyle=W_{t}+u_{t}=W_{t}+(\alpha\hat{g}_{t}+e_{t})
=(W t+e t)+α​g^t=Θ t+α​g^t.\displaystyle=(W_{t}+e_{t})+\alpha\hat{g}_{t}=\Theta_{t}+\alpha\hat{g}_{t}.

Thus, the virtual parameters Θ t\Theta_{t} evolve according to the dynamics of unconstrained, high-precision gradient ascent. The physical quantized weights W T W_{T} at any step T T are related to this ideal trajectory by

W T=Θ T−e T=(W 0+∑t=0 T−1 α​g^t)−e T.W_{T}=\Theta_{T}-e_{T}=\left(W_{0}+\sum_{t=0}^{T-1}\alpha\hat{g}_{t}\right)-e_{T}.(13)

Crucially, unlike the stateless baseline where errors sum linearly or as a random walk, the deviation in QES is determined solely by the single final residual e T e_{T}. Since Θ T\Theta_{T} rounds to the nearest grid point, this error is strictly bounded by the grid resolution: ‖e T‖∞≤Δ/2\|e_{T}\|_{\infty}\leq\Delta/2.

Consequently, QES guarantees that the quantized model W t W_{t} never deviates more than half a grid step from the ideal high-precision trajectory Θ t\Theta_{t}. The residual e t e_{t} effectively integrates infinitesimal gradient signals over time until they cross the quantization threshold, triggering a discrete update Δ​W≠0\Delta W\neq 0 that aligns the physical model with the virtual continuous path.

6 Future Work
-------------

Several promising directions exist in extending the capabilities of QES. First, while this paper validated QES on standard linear integer quantization (INT4, INT8, W8A8), it should be possible to extend it to more aggressive and non-uniform quantization paradigms, such as binary networks and floating-point formats (e.g., FP4). Second, the current stateless replay relies on a fixed lookback window K K to manage the compute-memory trade-off. An adaptive algorithm could be created that automatically tunes K K and the decay rate γ\gamma based on real-time convergence stability or available hardware resources, effectively removing the need for manual hyperparameter selection.

While the primary motivation for quantization so far has been to run existing large models on limited hardware, QES fine tuning opens up a new frontier: given the same hardware, a model with significantly more parameters can be quantized in the beginning, and QES be used to train the model directly in quantized space. Such scale-up is possible based on two aspects: (1) precision can be traded for more parameters, e.g. four-fold in case of INT4, and (2) only low-precision inference is needed during training, requiring about 1/12 of the memory of backpropagation (Malladi et al., [2024](https://arxiv.org/html/2602.03120v1#bib.bib11 "Fine-tuning language models with just forward passes")). Stacking these two benefits together opens up possibilities to train one or two orders of magnitude larger models with the same hardware.

7 Conclusion
------------

Several barriers make effective fine-tuning of quantized LLMs difficult: gradient stagnation, inaccurate parameter update due to discretization, and prohibitive memory requirements. This paper introduced QES, a novel backpropagation-free optimization framework designed to perform full-parameter optimization directly on quantized integer weights. First, vanishing or inaccurate gradients were identified as the primary cause of failure for existing ZO methods. Inspired by Delta-Sigma modulation, an accumulated error feedback mechanism was constructed to preserve and leverage these minuscule learning signals effectively. Second, to eliminate the memory overhead of storing high-precision error states, a Stateless Seed Replay mechanism was developed. This mechanism makes full-parameter fine-tuning possible within the strict memory constraints of standard quantized inference. These mechanisms were evaluated empirically on a challenging reasoning task. QES successfully overcame the quantized gradient issues and memory restrictions, significantly outperforming the state-of-the-art quantized fine-tuning baseline and achieving parity with memory-intensive oracles. Thus, by enabling the adaptation of large models on consumer-grade hardware, QES represents a significant step toward democratizing access to LLM fine-tuning.

Impact Statement
----------------

This paper presents Quantized Evolution Strategies (QES), an optimization framework designed to democratize access to Large Language Model (LLM) fine-tuning. By enabling high-precision learning directly within a quantized parameter space, QES allows full-parameter adaptation on consumer-grade hardware that was previously restricted to static inference. Furthermore, the significant reduction in memory overhead contributes to more resource-efficient AI development, potentially lowering the energy footprint and environmental impact of scaling large models.

References
----------

*   Y. An et al. (2025)Qwen3 technical report. External Links: 2505.09388, [Link](https://arxiv.org/abs/2505.09388)Cited by: [§2.2](https://arxiv.org/html/2602.03120v1#S2.SS2.p1.1 "2.2 Zeroth Order Fine-tuning for Quantized Models ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   M. P. Connolly, N. J. Higham, and T. Mary (2021)Stochastic rounding and its probabilistic backward error analysis. SIAM Journal on Scientific Computing 43 (1),  pp.A566–A585. External Links: [Document](https://dx.doi.org/10.1137/20M1334796), [Link](https://doi.org/10.1137/20M1334796), https://doi.org/10.1137/20M1334796 Cited by: [§3.2](https://arxiv.org/html/2602.03120v1#S3.SS2.p1.7 "3.2 Accumulated Error Feedback ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer (2023)QLoA: efficient finetuning of quantized LLMs. Advances in neural information processing systems 36,  pp.10088–10115. Cited by: [§1](https://arxiv.org/html/2602.03120v1#S1.p2.1 "1 Introduction ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§2.1](https://arxiv.org/html/2602.03120v1#S2.SS1.p2.1 "2.1 LLM Quantization and Low-Bit Fine-Tuning ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   C. Feng, S. Zhuo, X. Zhang, R. K. Ramakrishnan, Z. Yuan, and A. Z. Li (2024)Stepping forward on the last mile. Advances in Neural Information Processing Systems 37,  pp.94851–94870. Cited by: [§1](https://arxiv.org/html/2602.03120v1#S1.p2.1 "1 Introduction ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§2.2](https://arxiv.org/html/2602.03120v1#S2.SS2.p3.1 "2.2 Zeroth Order Fine-tuning for Quantized Models ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh (2023)GPTQ: accurate post-training quantization for generative pre-trained transformers. External Links: 2210.17323, [Link](https://arxiv.org/abs/2210.17323)Cited by: [§1](https://arxiv.org/html/2602.03120v1#S1.p1.1 "1 Introduction ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§2.1](https://arxiv.org/html/2602.03120v1#S2.SS1.p1.1 "2.1 LLM Quantization and Low-Bit Fine-Tuning ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§4.1](https://arxiv.org/html/2602.03120v1#S4.SS1.p2.1 "4.1 Experimental Setup ‣ 4 Experiments ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   K. Gandhi, D. Lee, G. Grand, M. Liu, W. Cheng, A. Sharma, and N. D. Goodman (2024)Stream of search (SoS): learning to search in language. External Links: 2404.03683, [Link](https://arxiv.org/abs/2404.03683)Cited by: [§1](https://arxiv.org/html/2602.03120v1#S1.p5.1 "1 Introduction ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   J. Gehring, K. Zheng, J. Copet, V. Mella, Q. Carbonneaux, T. Cohen, and G. Synnaeve (2024)RLEF: grounding code LLMs in execution feedback with reinforcement learning. arXiv preprint arXiv:2410.02089. Cited by: [§2.2](https://arxiv.org/html/2602.03120v1#S2.SS2.p1.1 "2.2 Zeroth Order Fine-tuning for Quantized Models ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   D. Guo et al. (2025)DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. Nature 645 (8081),  pp.633–638. External Links: ISSN 1476-4687, [Link](http://dx.doi.org/10.1038/s41586-025-09422-z), [Document](https://dx.doi.org/10.1038/s41586-025-09422-z)Cited by: [§1](https://arxiv.org/html/2602.03120v1#S1.p1.1 "1 Introduction ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§2.2](https://arxiv.org/html/2602.03120v1#S2.SS2.p1.1 "2.2 Zeroth Order Fine-tuning for Quantized Models ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   S. Gupta, A. Agrawal, K. Gopalakrishnan, and P. Narayanan (2015)Deep learning with limited numerical precision. In International conference on machine learning,  pp.1737–1746. Cited by: [§2.3](https://arxiv.org/html/2602.03120v1#S2.SS3.p2.1 "2.3 Techniques for Reducing Discretization Errors ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   H. Inose, Y. Yasuda, and J. Murakami (1962)A telemetering system by code modulation - Δ​Σ\Delta\Sigma modulation. IRE Transactions on Space Electronics and Telemetry SET-8 (3),  pp.204–209. External Links: [Document](https://dx.doi.org/10.1109/IRET-SET.1962.5008839)Cited by: [§1](https://arxiv.org/html/2602.03120v1#S1.p3.1 "1 Introduction ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§2.3](https://arxiv.org/html/2602.03120v1#S2.SS3.p1.1 "2.3 Techniques for Reducing Discretization Errors ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§3.2](https://arxiv.org/html/2602.03120v1#S3.SS2.p6.1 "3.2 Accumulated Error Feedback ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   S. P. Karimireddy, Q. Rebjock, S. Stich, and M. Jaggi (2019)Error feedback fixes SignSGD and other gradient compression schemes. In International conference on machine learning,  pp.3252–3261. Cited by: [§1](https://arxiv.org/html/2602.03120v1#S1.p3.1 "1 Introduction ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§2.3](https://arxiv.org/html/2602.03120v1#S2.SS3.p2.1 "2.3 Techniques for Reducing Discretization Errors ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§3.2](https://arxiv.org/html/2602.03120v1#S3.SS2.p6.1 "3.2 Accumulated Error Feedback ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   H. Li, Z. Xu, G. Taylor, C. Studer, and T. Goldstein (2018)Visualizing the loss landscape of neural nets. Advances in neural information processing systems 31. Cited by: [§4.2](https://arxiv.org/html/2602.03120v1#S4.SS2.p2.1 "4.2 Performance ‣ 4 Experiments ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   J. Lin, J. Tang, H. Tang, S. Yang, W. Chen, W. Wang, G. Xiao, X. Dang, C. Gan, and S. Han (2024)AWQ: activation-aware weight quantization for LLM compression and acceleration. External Links: 2306.00978, [Link](https://arxiv.org/abs/2306.00978)Cited by: [§1](https://arxiv.org/html/2602.03120v1#S1.p1.1 "1 Introduction ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§2.1](https://arxiv.org/html/2602.03120v1#S2.SS1.p1.1 "2.1 LLM Quantization and Low-Bit Fine-Tuning ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   Z. Liu, B. Oguz, C. Zhao, E. Chang, P. Stock, Y. Mehdad, Y. Shi, R. Krishnamoorthi, and V. Chandra (2024)LLM-QAT: data-free quantization aware training for large language models. In Findings of the Association for Computational Linguistics: ACL 2024,  pp.467–484. Cited by: [§1](https://arxiv.org/html/2602.03120v1#S1.p2.1 "1 Introduction ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§2.1](https://arxiv.org/html/2602.03120v1#S2.SS1.p2.1 "2.1 LLM Quantization and Low-Bit Fine-Tuning ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   Z. Liu, C. Chen, W. Li, P. Qi, T. Pang, C. Du, W. S. Lee, and M. Lin (2025)Understanding r1-zero-like training: a critical perspective. arXiv preprint arXiv:2503.20783. Cited by: [§2.2](https://arxiv.org/html/2602.03120v1#S2.SS2.p1.1 "2.2 Zeroth Order Fine-tuning for Quantized Models ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   S. Malladi, T. Gao, E. Nichani, A. Damian, J. D. Lee, D. Chen, and S. Arora (2024)Fine-tuning language models with just forward passes. External Links: 2305.17333, [Link](https://arxiv.org/abs/2305.17333)Cited by: [§1](https://arxiv.org/html/2602.03120v1#S1.p2.1 "1 Introduction ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§2.2](https://arxiv.org/html/2602.03120v1#S2.SS2.p2.1 "2.2 Zeroth Order Fine-tuning for Quantized Models ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§6](https://arxiv.org/html/2602.03120v1#S6.p2.1 "6 Future Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   OpenAI et al. (2024)OpenAI o1 system card. External Links: 2412.16720, [Link](https://arxiv.org/abs/2412.16720)Cited by: [§1](https://arxiv.org/html/2602.03120v1#S1.p1.1 "1 Introduction ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§2.2](https://arxiv.org/html/2602.03120v1#S2.SS2.p1.1 "2.2 Zeroth Order Fine-tuning for Quantized Models ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe (2022)Training language models to follow instructions with human feedback. External Links: 2203.02155, [Link](https://arxiv.org/abs/2203.02155)Cited by: [§2.2](https://arxiv.org/html/2602.03120v1#S2.SS2.p1.1 "2.2 Zeroth Order Fine-tuning for Quantized Models ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   J. Pan, J. Zhang, X. Wang, L. Yuan, H. Peng, and A. Suhr (2025)TinyZero. Note: https://github.com/Jiayi-Pan/TinyZeroAccessed: 2025-01-24 Cited by: [§1](https://arxiv.org/html/2602.03120v1#S1.p5.1 "1 Introduction ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§4.1](https://arxiv.org/html/2602.03120v1#S4.SS1.p1.1 "4.1 Experimental Setup ‣ 4 Experiments ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   Policy-gradient (2025)GRPO-Zero: implementing deepseek r1’s grpo algorithm from scratch. GitHub. Note: [https://github.com/policy-gradient/GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero)Cited by: [§4.1](https://arxiv.org/html/2602.03120v1#S4.SS1.p1.1 "4.1 Experimental Setup ‣ 4 Experiments ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   X. Qiu, Y. Gan, C. F. Hayes, Q. Liang, E. Meyerson, B. Hodjat, and R. Miikkulainen (2025)Evolution strategies at scale: LLM fine-tuning beyond reinforcement learning. External Links: 2509.24372, [Link](https://arxiv.org/abs/2509.24372)Cited by: [§1](https://arxiv.org/html/2602.03120v1#S1.p2.1 "1 Introduction ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§2.2](https://arxiv.org/html/2602.03120v1#S2.SS2.p2.1 "2.2 Zeroth Order Fine-tuning for Quantized Models ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§3.1](https://arxiv.org/html/2602.03120v1#S3.SS1.p2.4 "3.1 The Optimization Challenge ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§3.1](https://arxiv.org/html/2602.03120v1#S3.SS1.p3.2 "3.1 The Optimization Challenge ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§3.2](https://arxiv.org/html/2602.03120v1#S3.SS2.p3.4 "3.2 Accumulated Error Feedback ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   Qwen, :, A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, H. Lin, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Lin, K. Dang, K. Lu, K. Bao, K. Yang, L. Yu, M. Li, M. Xue, P. Zhang, Q. Zhu, R. Men, R. Lin, T. Li, T. Tang, T. Xia, X. Ren, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Wan, Y. Liu, Z. Cui, Z. Zhang, and Z. Qiu (2025)Qwen2.5 technical report. External Links: 2412.15115, [Link](https://arxiv.org/abs/2412.15115)Cited by: [§4.1](https://arxiv.org/html/2602.03120v1#S4.SS1.p2.1 "4.1 Experimental Setup ‣ 4 Experiments ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   B. Razavi (2016)The delta-sigma modulator [a circuit for all seasons]. IEEE Solid-State Circuits Magazine 8 (2),  pp.10–15. Cited by: [§1](https://arxiv.org/html/2602.03120v1#S1.p3.1 "1 Introduction ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§2.3](https://arxiv.org/html/2602.03120v1#S2.SS3.p1.1 "2.3 Techniques for Reducing Discretization Errors ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§3.2](https://arxiv.org/html/2602.03120v1#S3.SS2.p6.1 "3.2 Accumulated Error Feedback ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   RedHatAI and vLLM Project (2024)LLM compressor. External Links: [Link](https://github.com/vllm-project/llm-compressor)Cited by: [§4.1](https://arxiv.org/html/2602.03120v1#S4.SS1.p2.1 "4.1 Experimental Setup ‣ 4 Experiments ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   T. Salimans, J. Ho, X. Chen, S. Sidor, and I. Sutskever (2017)Evolution strategies as a scalable alternative to reinforcement learning. External Links: 1703.03864, [Link](https://arxiv.org/abs/1703.03864)Cited by: [§3.1](https://arxiv.org/html/2602.03120v1#S3.SS1.p2.4 "3.1 The Optimization Challenge ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§3.2](https://arxiv.org/html/2602.03120v1#S3.SS2.p3.4 "3.2 Accumulated Error Feedback ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   B. Sarkar, M. Fellows, J. A. Duque, A. Letcher, A. L. Villares, A. Sims, D. Cope, J. Liesen, L. Seier, T. Wolf, U. Berdica, A. D. Goldie, A. Courville, K. Sevegnani, S. Whiteson, and J. N. Foerster (2025)Evolution strategies at the hyperscale. External Links: 2511.16652, [Link](https://arxiv.org/abs/2511.16652)Cited by: [§1](https://arxiv.org/html/2602.03120v1#S1.p2.1 "1 Introduction ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   F. Seide, H. Fu, J. Droppo, G. Li, and D. Yu (2014)1-bit stochastic gradient descent and its application to data-parallel distributed training of speech dnns.. In Interspeech, Vol. 2014,  pp.1058–1062. Cited by: [§1](https://arxiv.org/html/2602.03120v1#S1.p3.1 "1 Introduction ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§2.3](https://arxiv.org/html/2602.03120v1#S2.SS3.p2.1 "2.3 Techniques for Reducing Discretization Errors ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§3.2](https://arxiv.org/html/2602.03120v1#S3.SS2.p6.1 "3.2 Accumulated Error Feedback ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo (2024)DeepSeekMath: pushing the limits of mathematical reasoning in open language models. External Links: 2402.03300, [Link](https://arxiv.org/abs/2402.03300)Cited by: [§1](https://arxiv.org/html/2602.03120v1#S1.p1.1 "1 Introduction ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§2.2](https://arxiv.org/html/2602.03120v1#S2.SS2.p1.1 "2.2 Zeroth Order Fine-tuning for Quantized Models ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§3.1](https://arxiv.org/html/2602.03120v1#S3.SS1.p2.8 "3.1 The Optimization Challenge ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   J.C. Spall (1992)Multivariate stochastic approximation using a simultaneous perturbation gradient approximation. IEEE Transactions on Automatic Control 37 (3),  pp.332–341. External Links: [Document](https://dx.doi.org/10.1109/9.119632)Cited by: [§2.2](https://arxiv.org/html/2602.03120v1#S2.SS2.p2.1 "2.2 Zeroth Order Fine-tuning for Quantized Models ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   J. C. Spall (2002)Multivariate stochastic approximation using a simultaneous perturbation gradient approximation. IEEE transactions on automatic control 37 (3),  pp.332–341. Cited by: [§2.2](https://arxiv.org/html/2602.03120v1#S2.SS2.p2.1 "2.2 Zeroth Order Fine-tuning for Quantized Models ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   N. Ström (2015)Scalable distributed dnn training using commodity GPU cloud computing. In Sixteenth Annual Conference of the International Speech Communication Association, External Links: [Link](https://www.isca-archive.org/interspeech_2015/strom15_interspeech.pdf)Cited by: [§1](https://arxiv.org/html/2602.03120v1#S1.p3.1 "1 Introduction ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§2.3](https://arxiv.org/html/2602.03120v1#S2.SS3.p2.1 "2.3 Techniques for Reducing Discretization Errors ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§3.2](https://arxiv.org/html/2602.03120v1#S3.SS2.p6.1 "3.2 Accumulated Error Feedback ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han (2024)SmoothQuant: accurate and efficient post-training quantization for large language models. External Links: 2211.10438, [Link](https://arxiv.org/abs/2211.10438)Cited by: [§1](https://arxiv.org/html/2602.03120v1#S1.p1.1 "1 Introduction ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§2.1](https://arxiv.org/html/2602.03120v1#S2.SS1.p1.1 "2.1 LLM Quantization and Low-Bit Fine-Tuning ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, W. Dai, T. Fan, G. Liu, L. Liu, X. Liu, H. Lin, Z. Lin, B. Ma, G. Sheng, Y. Tong, C. Zhang, M. Zhang, W. Zhang, H. Zhu, J. Zhu, J. Chen, J. Chen, C. Wang, H. Yu, Y. Song, X. Wei, H. Zhou, J. Liu, W. Ma, Y. Zhang, L. Yan, M. Qiao, Y. Wu, and M. Wang (2025)DAPO: an open-source LLM reinforcement learning system at scale. External Links: 2503.14476, [Link](https://arxiv.org/abs/2503.14476)Cited by: [§2.2](https://arxiv.org/html/2602.03120v1#S2.SS2.p1.1 "2.2 Zeroth Order Fine-tuning for Quantized Models ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"). 
*   J. Zhou, Y. Yang, K. Zhen, Z. Liu, Y. Zhao, E. Banijamali, A. Mouchtaris, N. Wong, and Z. Zhang (2025)QuZO: quantized zeroth-order fine-tuning for large language models. arXiv preprint arXiv:2502.12346. Cited by: [§1](https://arxiv.org/html/2602.03120v1#S1.p2.1 "1 Introduction ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§2.2](https://arxiv.org/html/2602.03120v1#S2.SS2.p3.1 "2.2 Zeroth Order Fine-tuning for Quantized Models ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§2.3](https://arxiv.org/html/2602.03120v1#S2.SS3.p2.1 "2.3 Techniques for Reducing Discretization Errors ‣ 2 Related Work ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§3.2](https://arxiv.org/html/2602.03120v1#S3.SS2.p1.7 "3.2 Accumulated Error Feedback ‣ 3 Method ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [2nd item](https://arxiv.org/html/2602.03120v1#S4.I1.i2.p1.1 "In 4.1 Experimental Setup ‣ 4 Experiments ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost"), [§5](https://arxiv.org/html/2602.03120v1#S5.p2.2 "5 Temporal Equivalence to Continuous Optimization ‣ Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost").

