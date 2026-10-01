# Evolution Strategies at the Hyperscale

Bidipta Sarkar<sup>\*1,2</sup>, Mattie Fellows<sup>\*1</sup>, Juan Agustin Duque<sup>\*2,3</sup>,  
 Alistair Letcher<sup>†1</sup>, Antonio León Villares<sup>†1</sup>, Anya Sims<sup>†1</sup>, Clarisse Wibault<sup>†1</sup>,  
 Dmitry Samsonov<sup>†6</sup>, Dylan Cope<sup>†1</sup>, Jarek Liesen<sup>†1</sup>, Kang Li<sup>†1</sup>, Lukas Seier<sup>†1</sup>, Theo Wolf<sup>†1</sup>,  
 Uljad Berdica<sup>†1</sup>, Valentin Mohl<sup>†1</sup>,  
 Alexander David Goldie<sup>1,2</sup>, Aaron Courville<sup>3,5</sup>, Karin Sevegnani<sup>4</sup>,  
 Shimon Whiteson<sup>‡2</sup>, Jakob Nicolaus Foerster<sup>†1</sup>.

<sup>1</sup> FLAIR - University of Oxford, <sup>2</sup> WhiRL - University of Oxford, <sup>3</sup> MILA– Québec AI Institute

<sup>4</sup> NVIDIA AI Technology Center, <sup>5</sup> CIFAR AI Chair, <sup>6</sup> NormaCore.dev

{bidipta.sarkar, matthew.fellows, jakob.foerster}@eng.ox.ac.uk

juan.duque@mila.quebec, shimon.whiteson@cs.ox.ac.uk

## Abstract

Evolution Strategies (ES) is a class of powerful black-box optimisation methods that are highly parallelisable and can handle non-differentiable and noisy objectives. However, naïve ES becomes prohibitively expensive at scale on GPUs due to the low arithmetic intensity of batched matrix multiplications with unstructured random perturbations. We introduce Evolution Guided GeneRAL Optimisation via Low-rank Learning (EGGROLL), which improves arithmetic intensity by structuring individual perturbations as rank- $r$  matrices, resulting in a hundredfold increase in training speed for billion-parameter models at large population sizes, achieving up to 91% of the throughput of pure batch inference. We provide a rigorous theoretical analysis of Gaussian ES for high-dimensional parameter objectives, investigating conditions needed for ES updates to converge in high dimensions. Our results reveal a linearising effect, and proving consistency between EGGROLL and ES as parameter dimension increases. Our experiments show that EGGROLL: (1) enables the stable pretraining of nonlinear recurrent language models that operate purely in integer datatypes, (2) is competitive with GRPO for post-training LLMs on reasoning tasks, and (3) does not compromise performance compared to ES in tabula rasa RL settings, despite being faster. Our code is available at <https://eshyperscale.github.io/>.

Figure 1: Schematic visualisation of EGGROLL using  $N$  workers.

## 1 Introduction

Evolution Strategies (ES) (Rechenberg, 1978; Beyer, 1995; Beyer & Schwefel, 2002) is an attractive alternative to first-order methods based on gradient backpropagation for several reasons. First, ES does not require differentiability; it can optimise a broader class of models, like those with discrete parametrisations (cellular

<sup>\*</sup>Equal Contribution <sup>†</sup> Core Contributor, sorted by alphabetical order in first names <sup>‡</sup>Equal Senior AuthorsFigure 2: (a) Relative speed of our method, EGGROLL, in terms of experience throughput versus prior methods, where 100 is the maximum batch inference throughput. See Appendix E for more details. (b) We use EGGROLL to train an int8 RNN language model from scratch, scaling population size from 2 to 1,048,576 with a fixed data batch size of 16. The dotted line is a fp32 Transformer trained with backprop SGD. EGGROLL’s test next-token cross-entropy of 3.40 bits/byte while backprop only gets 3.58 bits/byte.

automata) or objectives for which gradients are unavailable or noisy, such as outcome-only rewards in LLM fine-tuning (Qiu et al., 2025). Second, ES can be more robust to noisy and ill-conditioned optimisation landscapes (Wierstra et al., 2011; Xue et al., 2021). Population-based exploration smooths irregularities (Salimans et al., 2017), tolerates discontinuities, and mitigates issues like ill-conditioned curvature or vanishing and exploding gradients in long-range or recurrent settings (Hansen, 2023). Third, ES is highly amenable to parallel scaling, since fitness evaluations are independent across population members and require only the communication of scalar fitnesses, which maps cleanly onto modern inference infrastructure and yields near-linear speedups on large clusters (Salimans et al., 2017). By contrast, backpropagation requires communicating and aggregating gradients across devices, yielding updates with high memory and computational costs. Furthermore, backpropagation requires special care when training models with low-precision datatypes (Fishman et al., 2025), whereas ES can directly optimise any model with the same datatypes used at inference time. Together, these properties position ES as a potentially powerful tool for training large, discrete, or hybrid architectures, and end-to-end systems with non-differentiable components, including LLMs (Brown et al., 2020; Chowdhery et al., 2023; Du et al., 2022; Fedus et al., 2022).

However, there are currently practical obstacles to employing ES at scale. In deep learning architectures (Goodfellow et al., 2016), the majority of trainable parameters form linear mappings represented by matrices (Rosenblatt, 1962; Hochreiter & Schmidhuber, 1996; Bengio et al., 2000; Krizhevsky et al., 2012; Goodfellow et al., 2014; Kingma & Welling, 2014; Vaswani et al., 2017). Naïvely adapting ES therefore requires generating full-rank matrix perturbations that replicate the entire parameter set for every population member. This inflates memory costs and forces frequent movement of large weight tensors. Evaluating these perturbations then requires a separate sequence of matrix multiplications per member, so the total compute and wall-clock time scale roughly with the population size and sequence length since batched matrix multiplication has a low arithmetic intensity, i.e., the ratio of arithmetic operations to memory traffic (Williams, 2008). In billion-parameter regimes, these two costs dominate, limiting ES to small models and small populations (Qiu et al., 2025; Korotyshova et al., 2025).

To mitigate both memory and computational bottlenecks, we introduce Evolution Guided GeneRal Optimisation via Low-rank Learning (EGGROLL), an ES algorithm that allows for the efficient training of neural network architectures with billions of parameters. Analogous to LoRA’s low-rank adapters in gradient-based training (Hu et al., 2022), EGGROLL generates *low-rank* parameter-space perturbations for ES; instead of sampling a full-rank matrix  $E \in \mathbb{R}^{m \times n}$ , we sample  $A \in \mathbb{R}^{m \times r}$  and  $B \in \mathbb{R}^{n \times r}$  with  $r \ll \min(m, n)$  and form  $E = \frac{1}{\sqrt{r}} AB^\top$ . This reduces auxiliary perturbation matrix storage from  $mn$  to  $(m + n)r$  per layer, and proportionally reduces tensor movement.

Moreover, we use a counter-based deterministic random number generator (RNG) (Salmon et al., 2011; Bradbury et al., 2018) to reconstruct noise on demand, so matrix perturbations need not persist in memory. When evaluating the fitness of members of multiple perturbations in parallel, EGGROLL batches a population of low-rank adapters and shares the base activations, enabling a single forward pass that applies all  $AB^\top$  updates via specialised batched matrix multiplications with significantly higher arithmetic intensity, resultingin over a hundredfold increase in training throughput for large neural networks at large population sizes, as shown in Fig. 2a. Crucially, EGGROLL does not restrict updates to be low-rank, as the overall update is a weighted average of rank  $r$  matrices across the population, making the matrix parameter update rank  $\min(Nr, m, n)$ .

To understand ES when applied to large parameter models, we analyse the convergence properties of general Gaussian ES in high dimensions, showing there exists a critical noise scaling  $\sigma_d = o(d^{-1/2})$  under which the update provably linearises and converges to the first-order derivative for a broad class of (possibly discontinuous) objectives. We identify three distinct regimes—linearisation, critical, and divergence—and establish provably tight conditions for stable ES optimisation in large models. Building on this, we extend the analysis to EGGROLL and prove that even fixed low-rank updates (including rank-1) converge to the true ES gradient as dimension grows, despite heavier-tailed perturbations. Our results explain the empirical success of EGGROLL in high-dimensional neural networks and connect its behaviour to neural tangent kernel-style linearisation (Jacot et al., 2018), yielding explicit convergence rates under standard overparameterised regimes. We also provide a rigorous theoretical analysis of the low-rank approximation accuracy, proving that EGGROLL updates converge to the full-rank Gaussian ES updates at a fast  $\mathcal{O}(r^{-1})$  rate.

Furthermore, in our extensive empirical evaluation, we test this hypothesis across a wide range of domains. In tabula rasa and multi-agent RL (MARL) settings, we show that EGGROLL does not compromise performance compared to naïve ES despite being faster. We demonstrate the scalability of EGGROLL for LLM fine-tuning with experiments on pretrained RWKV7 (Peng et al., 2025) models, modern recurrent language models that enable large batch inference due to their constant state size. Finally, we develop a nonlinear RNN language model that operates purely in integer datatypes, and demonstrate that EGGROLL can stably pretrain this language model, a feat which is only feasible due to the large population sizes enabled by EGGROLL.

## 2 Preliminaries

### 2.1 Low-Rank Matrix Approximations

When adapting high-dimensional foundation models for specific tasks, updating the parameters using gradient-based methods has high memory requirements. LoRA (Hu et al., 2022) applies low-rank approximations to the matrix multiplications to reduce these costs. For each matrix  $M_i \in \mathbb{R}^{m \times n}$  in the model, a low-rank approximation can be made by decomposing each matrix:

$$M_i \approx M_i^0 + A_i B_i^\top,$$

where  $M_i^0 := \text{StopGrad}(M_i)$  is the imported matrix from the foundation model with frozen parameters and  $A_i \in \mathbb{R}^{m \times r}$  and  $B_i \in \mathbb{R}^{n \times r}$  are low-width column matrices (i.e.,  $r \ll \min(m, n)$ ) whose parameters are updated through gradient-based optimisation during task-specific adaptation. This reduces the number of optimisation parameters for each matrix from  $mn$  to  $r(m + n)$ . EGGROLL uses a similar low-rank approximation for evolutionary strategies.

### 2.2 Evolution Strategies

Evolution strategies (ES) (Rechenberg, 1978; Beyer, 1995; Beyer & Schwefel, 2002) is a set of black-box optimisation methods that has emerged as a useful alternative to first-order gradient-based methods like stochastic gradient descent (SGD), particularly for noisy or non-differentiable systems. Let  $f : \mathbb{R}^d \rightarrow \mathbb{R}$  denote an objective to be optimised, known as the *fitness*, where the goal is to find an optimising set of parameters  $x^* \in \arg \max_{x \in \mathbb{R}^d} f(x)$ . Each set of parameters is collected into a  $d$ -dimensional vector known as a genotype. We denote the derivative of the fitness  $\nabla_x f(x)|_{x=a}$  evaluated at  $x = a$  as  $\nabla f(a)$ . Unlike first-order gradient-based methods, which query derivatives  $\nabla f(x)$  to update the vector of parameters  $x$ , evolutionary methods update a parametric population distribution over the fitness parameter space  $\pi(x|\theta)$ , which is smoothly parametrised by a separate set of parameters  $\theta \in \Theta$ . The population distribution generates perturbations  $x \sim \pi(x|\theta)$  known as mutations. The problem of optimising the fitness  $f(x)$  for  $x$  reduces to optimising the parameters of the population distribution  $\theta$ . This is achieved by solving a *secondary* optimisation problem to maximise the expected fitness under  $\pi(x|\theta)$  for  $\theta$ :

$$J(\theta) = \mathbb{E}_{x \sim \pi(x|\theta)} [f(x)].$$Introducing a population distribution *smooths* the fitness landscape; since  $\pi(x|\theta)$  is smooth in  $\theta$ , the resulting objective  $J(\theta)$  is also smooth in  $\theta$ , provided  $f(x)$  is measurable and integrable but not necessarily differentiable. Evolution strategies can therefore optimise black-box problems that may be non-differentiable as the derivatives of  $J(\theta)$  exist for fitness functions that are discontinuous, yielding a gradient with respect to  $\theta$ :

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{x \sim \pi(x|\theta)} [\nabla_{\theta} \log \pi(x|\theta) f(x)],$$

where  $\nabla_{\theta} \log \pi(x|\theta)$  is known as the score function. A Monte Carlo estimate is formed by sampling  $N$  search mutations  $x_i \sim \pi(x_i|\theta)$  and computing an average of the score-weighted fitnesses:

$$\hat{\nabla}_{\theta} J(\theta) = \frac{1}{N} \sum_{i=1}^N \nabla_{\theta} \log \pi(x_i|\theta) f(x_i), \quad (1)$$

with which we update  $\theta$  via stochastic gradient ascent with a suitable stepsize  $\alpha_t$ :

$$\theta_{t+1} \leftarrow \theta_t + \alpha_t \hat{\nabla}_{\theta} J(\theta_t).$$

ES does not require taking derivatives directly through the fitness function; instead the Monte Carlo update in Eq. (1) only requires evaluation of  $f(x_i)$  for each mutation  $x_i$  to estimate  $\nabla_{\theta} J(\theta)$ . As ES only queries  $f(x)$  and not  $\nabla f(\mu)$ , it is a *zeroth-order* optimisation method.

In this paper, we study ES using Gaussian population distributions:  $\pi(x|\theta) = \mathcal{N}(\mu, I_d \sigma^2)$ . In addition to its mathematical convenience, the central limit theorem means that the Gaussian distribution emerges naturally from the EGGROLL low-rank approximation as rank increases, even if the matrices  $A$  and  $B$  are themselves non-Gaussian. Moreover, most widely-used ES algorithms assume Gaussian population distributions (Rechenberg, 1978; Schwefel, 1995; Hansen & Ostermeier, 2001a; Beyer & Schwefel, 2002; Auger & Hansen, 2011; Wierstra et al., 2011; Salimans et al., 2017). In our setting, ES optimises over the population mean  $\mu \in \mathbb{R}^d$ , which acts as a proxy for the true maximum of the fitness function, and the variance parameter  $\sigma^2 \geq 0$  is treated as a hyperparameter to be tuned.

For the Gaussian population distribution we study in this paper, the ES update can be written using an expectation under a standard normal distribution by making a transformation of variables  $v = \frac{x - \mu}{\sigma}$  (Wierstra et al., 2011; Salimans et al., 2017):

$$\begin{aligned} \nabla_{\mu} J(\theta) &= -\frac{1}{\sigma} \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\nabla_v \log p(v) \cdot f(\mu + \sigma v)], \\ &= \frac{1}{\sigma} \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [v \cdot f(\mu + \sigma v)], \end{aligned} \quad (2)$$

where  $v \sim P(v) = \mathcal{N}(0, I_d)$  and  $p(v)$  denotes the density of  $P(v)$ . In this form, Eq. (2) shows that Gaussian ES methods optimise the fitness by generating search vectors from a standard normal distribution  $\mathcal{N}(0, I_d)$  around the mean parameter  $\mu$ .

### 2.3 Evolution Strategies for Matrix Parameters

A key focus of this paper is to develop efficient methods for evolution strategies that target *matrix parameters*. When working in matrix space, it is convenient to use the matrix Gaussian distribution (Dawid, 1981), which is defined directly over matrices  $X \in \mathbb{R}^{m \times n}$ :

$$\mathcal{N}(M, U, V) = \frac{1}{(2\pi)^{\frac{mn}{2}} \det(U)^{\frac{n}{2}} \det(V)^{\frac{m}{2}}} \exp \left( -\frac{1}{2} \text{tr} (V^{-1} (X - M)^{\top} U^{-1} (X - M)) \right),$$

where  $M \in \mathbb{R}^{m \times n}$  is the mean matrix,  $U \in \mathbb{R}^{m \times m}$  is the row covariance matrix and  $V \in \mathbb{R}^{n \times n}$  is the column covariance matrix. We use  $\text{vec}(\cdot)$  to denote the vectorisation operator:

$$\text{vec}(X) := [x_{1,1}, \dots, x_{m,1}, x_{1,2}, \dots, x_{m,n}]^{\top}.$$

The matrix Gaussian distribution is a generalisation of the multivariate Gaussian distribution  $\mathcal{N}(\mu, \Sigma)$  defined over vector space. Sampling a matrix  $X \sim \mathcal{N}(M, U, V)$  from a matrix Gaussian distribution is equivalentto sampling a vector  $\text{vec}(X) \sim \mathcal{N}(\mu, \Sigma)$  from a multivariate Gaussian distribution with mean  $\mu = \text{vec}(M)$  and covariance matrix  $\Sigma = V \otimes U$  where  $\otimes$  denotes the Kronecker product. For isotropic matrix Gaussian distributions with covariance matrices  $U = \sigma I_m$  and  $V = \sigma I_n$ , the equivalent multivariate Gaussian distribution is also isotropic with  $\Sigma = \sigma^2 I_{mn}$ . We denote the  $\ell^2$  vector norm as  $\|\cdot\|$  and to measure distance between matrices, we use the Frobenius norm:

$$\|M\|_F := \sqrt{\sum_{i,j} m_{i,j}^2} = \|\text{vec}(M)\|,$$

which provides an upper bound on the matrix 2-norm (Petersen & Pedersen, 2012). Let  $W \in \mathbb{R}^{m \times n}$  be a set of matrix parameters where  $\text{vec}(W)$  forms a subset of the full parameter vector  $x$ , typically parametrising the weights of a linear layer in a neural network. As we derive in Section B, the Gaussian ES update associated with the matrix  $W$  is:

$$\begin{aligned} \nabla_M J(\theta) &= -\frac{1}{\sigma} \mathbb{E}_{E \sim P(E)} [\nabla_E \log p(E) \cdot f(W = M + \sigma E)], \\ &= \frac{1}{\sigma} \mathbb{E}_{E \sim P(E)} [E \cdot f(W = M + \sigma E)], \end{aligned} \quad (3)$$

where  $M$  is the mean matrix associated with  $W$ , i.e.  $\text{vec}(M)$  forms a subset of  $\mu$ , and  $P(E)$  is a zero-mean standard normal matrix distribution:  $p(E) = \mathcal{N}(0, I_m, I_n)$ . The gradient in Eq. (3) is estimated using the Monte Carlo estimate:

$$\hat{\nabla}_M J(\theta) = \frac{1}{\sigma N} \sum_{i=1}^N E_i \cdot f(W = M + \sigma E_i),$$

by sampling  $N$  search matrices  $E_i \sim P(E_i)$  from a standard matrix normal distribution  $\mathcal{N}(0, I_m, I_n)$  around the mean parameter matrix  $M$ , which is updated via stochastic gradient ascent:

$$M_{t+1} \leftarrow M_t + \alpha_t \hat{\nabla}_M J(\theta_t).$$

### 3 Related Work

#### 3.1 Evolutionary Algorithms

Evolutionary algorithms have long been a compelling alternative to backpropagation-based training methods (e.g., genetic algorithms (Such et al., 2018) or symbolic evolution (Koza, 1994)). Much research in evolution has focused on developing algorithms for deep learning that scale well to distributed parallel computation (Jaderberg et al., 2017; Hansen & Ostermeier, 2001b; Salimans et al., 2017). These approaches have increased in popularity following the application of ES to policy learning in deep RL environments (Salimans et al., 2017). Since then, evolution has been widely applied in other domains, such as meta-learning (e.g., (Lu et al., 2022; Metz et al., 2022; Lange et al., 2023; Goldie et al., 2024; 2025)), hyperparameter tuning (e.g., (Parker-Holder et al., 2021; Tani et al., 2021; Vincent & Jidesh, 2023)), and drug discovery (Towers et al., 2025). ES has also enabled the development of neural network architectures that are unsuitable for backpropagation, such as activation-free models that exploit floating point rounding error as an implicit nonlinearity (Foerster, 2017). Here, we consider how to apply ES at a scale beyond the small networks and population sizes of prior work. For example, Salimans et al. (2017) use a maximum population size of 1440, whereas we use over a million.

While low-rank structures have been used in prior evolutionary algorithms, they have been applied to different ends, with different trade-offs, relative to EGGROLL. Choromanski et al. (2019) use a low-rank search space found via principal component analysis, which provides a better search direction to more efficiently use small populations. Garbus & Pollack (2025) optimise a low-rank factorisation instead of the full dense matrix with neuroevolution, achieving similar computational gains to EGGROLL but is limited to the low-rank structure regardless of population size.### 3.2 Evolution Strategies for LLMs

Although gradient backpropagation is typically used for LLM training and fine-tuning, prior work explores ES variants for fine-tuning. In particular, Zhang et al. (2024)’s two-point zeroth-order gradient estimator, which can be viewed as an ES-inspired method using a single perturbation direction and two function queries per update, is used by Malladi et al. (2023) for memory-efficient LLM fine-tuning. Yu et al. (2025) extend this approach by projecting perturbations to a low-rank subspace, improving convergence. Jin et al. (2024) perform ES directly on LoRA matrices. These works focus on supervised fine-tuning and report performance comparable to full fine-tuning, but do not address whether pretraining is possible with two-point zeroth-order methods; we find that large population sizes are necessary for pretraining, indicating such methods are unsuitable here.

Recent work also explores ES in the context of LLM reasoning. Korotyshova et al. (2025) first train LoRA adapters using supervised fine-tuning (SFT) before decomposing them into fixed SVD bases alongside singular values that are trained using CMA-ES. They achieve comparable performance to GRPO (Shao et al., 2024) in significantly less wall-clock time on maths reasoning benchmarks. Qiu et al. (2025) directly use ES to optimise all LLM parameters for reasoning, with stronger performance than GRPO on the countdown reasoning task. However, both of these approaches use relatively small population sizes, on the order of a hundred unique perturbations per update, and instead collect hundreds of rollouts per perturbation to efficiently use GPUs. By contrast, our approach allows all generations to use different perturbations, such that our maximum population size per update is orders of magnitude larger (equal to the maximum inference batch size), without compromising token generation throughput.

## 4 EGGROLL

We now introduce EGGROLL (Algorithm 1). A practical issue with using a low-rank matrix approximation is that its distribution and score function have no analytic solution except for degenerate cases, so in Section 4.1 we derive the EGGROLL approximate score function from the limiting high-rank Gaussian. Section 4.2 describes how to efficiently implement EGGROLL on modern hardware.

### 4.1 Low-Rank Evolution Strategies

Recall the Gaussian matrix ES update from Eq. (3). Our goal is to introduce a tractable approximation to generating full-rank matrices by using low-rank matrices  $AB^\top$  as our search matrices instead. Let  $p(A)$  and  $p(B)$  denote the distribution of  $A \in \mathbb{R}^{m \times r}$  and  $B \in \mathbb{R}^{n \times r}$ .

**Assumption 1 (I.I.D. Sampling).** *Assume all elements  $a_{i,j} \in A$  and  $b_{i,j} \in B$  are continuous, identically and independently distributed random variables according to some zero-mean, symmetric, absolutely continuous distribution  $p_0(\cdot)$  with finite fourth-order moments and unit variance.*

#### Algorithm 1 EGGROLL( $r, \alpha, \sigma, T_{\max}, N_{\text{workers}}$ )

```

initialise  $M$  and workers with known random seeds  $\varsigma$ 
for  $T_{\max}$  timesteps do
  for each worker  $i \in \{1, \dots, N_{\text{workers}}\}$  in parallel do
     $A_i \sim p(A_i), B_i \sim p(B_i)$ 
     $E_i \leftarrow \frac{1}{\sqrt{r}} A_i B_i^\top$ 
     $f_i \leftarrow f(W = M + \sigma E_i)$ 
  end for
  workers share scalar fitness  $f_i$  with other workers
  for each worker  $i \in \{1, \dots, N_{\text{workers}}\}$  in parallel do
    reconstruct  $E_j$  for  $j \in \{1, \dots, N_{\text{workers}}\}$  from  $\varsigma$ 
     $M \leftarrow M + \alpha \frac{1}{N_{\text{workers}}} \sum_{j=1}^{N_{\text{workers}}} E_j f_j$ 
  end for
end for

```

This assumption is easily satisfied for most perturbation distributions used by ES, including members from the set of generalised Gaussian distributions like Laplace, normal, and uniform distributions. We then form a low-rank search matrix:  $E = \frac{1}{\sqrt{r}} AB^\top$ . The  $\frac{1}{\sqrt{r}}$  scaling ensures the variance of  $E$  remains bounded for all  $r$ . We denote the induced distribution of  $E$  as  $P(E)$ .  $E = \frac{1}{\sqrt{r}} AB^\top$  maps to the manifold  $\mathbb{M}^r \subset \mathbb{R}^{m \times n}$  of rank- $r$  matrices. Hence, the density  $p(E)$  is defined with respect to a unit volume on the manifold and cannot be defined with respect to the standard unit volume in Euclidean space. For the corresponding score function, gradients with respect to  $\log p(E)$  are not defined over the usual Euclidean space. Instead, we use an approximation  $\hat{S}(E) : \mathbb{R}^{m \times n} \rightarrow \mathbb{R}^{m \times n}$  for the score function, yielding our low-rank update:

$$\hat{g}_{\text{LR}} = -\frac{1}{\sigma} \mathbb{E}_{E \sim p(E)} \left[ \hat{S}(E) \cdot f(W = M + \sigma E) \right]. \quad (4)$$

In our experiments, analysis and Algorithm 1, we use a Gaussian approximate score function:

$$\hat{S}(E) = -E, \quad (5)$$which is the score function for the Gaussian distribution  $\mathcal{N}(0, I_m, I_n)$ . This choice is motivated by two theoretical insights from Section 5. The matrix  $AB^\top$  can be decomposed as a sum of independent, zero-mean vector outer products. Under Assumption 1, the central limit theorem applies to this sum of variables, proving that  $P(E)$  converges in distribution to a Gaussian  $\mathcal{N}(0, I_m, I_n)$  as rank  $r$  increases, recovering the approximate Gaussian score in the limit. Secondly, we investigate the convergence of ES and EGGROLL as the number of parameters grows, proving both updates converge to a linearised form that is consistent with the EGGROLL update using the Gaussian approximate score function.

EGGROLL is not wedded to any particular score function approximator and we derive and explore a set of mean-field approximators in Appendix D.1 as alternatives. However, our experiments show that the Gaussian approximator has the best overall performance on the tasks we consider. To optimise the ES objective using the EGGROLL update, we adapt the parallelised evolutionary strategies algorithm from Salimans et al. (2017). We make a Monte Carlo estimate of the expectation in Eq. (4) with  $N_{\text{workers}}$  samples to optimise the mean matrix parameters  $M$  using (approximate) stochastic gradient ascent. This yields the Gaussian EGGROLL update:

**EGGROLL UPDATE:** For each worker  $i$  (in parallel), sample  $A_{i,t} \sim p(A_{i,t}), B_{i,t} \sim p(B_{i,t})$  and form a low-rank perturbation  $E_{i,t} = \frac{1}{\sqrt{r}} A_{i,t} B_{i,t}^\top$ . Update matrix parameters using:

$$M_{t+1} \leftarrow M_t + \frac{\alpha_t}{N_{\text{workers}}} \sum_{i=1}^{N_{\text{workers}}} E_{i,t} f(W = M_t + \sigma E_{i,t}). \quad (6)$$

Here we absorb the constant  $\frac{1}{\sigma}$  into the tunable learning rate  $\alpha_t$ . As each random matrix  $E_{i,t}$  in Eq. (6) has rank  $r$  almost surely and the matrix is updated using a sum of  $N_{\text{worker}}$  such matrices, the overall EGGROLL matrix parameter update has rank  $\min(Nr, m, n)$  almost surely, i.e., the overall parameter update is not restricted to be low-rank. For all experiments in Section 6,  $Nr > \min(m, n)$ , i.e., EGGROLL parameter updates are full-rank.

## 4.2 Hardware-Efficient Implementation

A key reason to use EGGROLL over standard ES is that large populations can be simulated in parallel on a GPU thanks to the low-rank perturbations. For the sake of exposition, we write equations from the perspective of a single worker,  $i$ , and explain in text how this corresponds to batched GPU operations. Consider the task of computing a batched forward pass over inputs  $u_i \in \mathbb{R}^{d_{in}}$  for a linear layer with mean parameter  $M \in \mathbb{R}^{d_{out} \times d_{in}}$ . The standard forward pass is just a regular matrix multiplication,  $u_i M^\top$ , since  $M$  is constant across all threads. In contrast, naively applying ES by trying to compute  $u_i (M + \sigma E_i)^\top$  becomes a batched matrix multiplication, which is inefficient on GPUs since every element of  $M + \sigma E_i$  is only used in a single multiplication, yielding poor arithmetic intensity.

However, with EGGROLL we know that  $u_i (M + \sigma E_i)^\top = u_i M^\top + \frac{\sigma}{\sqrt{r}} (u_i B_i) A_i^\top$ , which improves arithmetic intensity since it preserves the efficient general matrix multiplication used in batched inference while adding some additional cheap work per perturbation. In this context, the bulk of compute is spent on the efficient calculation of  $u_i M^\top$  using regular matrix multiplication. Meanwhile, when  $r = 1$ ,  $u_i B_i$  simply becomes an inexpensive batch of  $N$  vector-vector dot products of length  $d_{in}$  to get a batch of  $N$  scalars, which is then processed by a batched scalar-vector multiplication when multiplying by  $A_i^\top$ . This decomposition is key to efficient batched LoRA inference, such as those used by vLLM (Kwon et al., 2023), which is why EGGROLL achieves the same speeds as batched LoRA inference systems. The batched LoRA inference enables high arithmetic intensity, enabling us to saturate compute with many unique perturbations per input. Note that this is impossible with naïve ES because each perturbation requires a separate matrix-vector multiplication, setting an upper bound of 1 for arithmetic intensity regardless of population size; see Appendix F for a full derivation. We additionally optimise the update by not explicitly materialising the individual  $E_i$  in the computation of  $\sum_{i=1}^N E_i f_i$ , the key term in the Gaussian approximate score function. In particular, when the rank is 1, we reconstruct  $A \in \mathbb{R}^{N \times d_{out}}$  and  $B \in \mathbb{R}^{N \times d_{in}}$  and calculate the expression as  $(\text{diag}(f) A)^\top B$ , a simple matrix multiplication.## 5 Analysis

*Proofs for all theorems can be found in Appendices A to D.*

In this section, we investigate the theoretical properties of the ES and EGGROLL updates. In Section 5.1, we study the convergence properties of the general Gaussian ES update as the parameter dimension  $d \rightarrow \infty$ , obtaining the conditions required for convergence to a linearised form. We then extend this analysis to the EGGROLL update in Section 5.2. Finally, in Section 5.3 we provide an analysis investigating the effect that increasing the rank of the EGGROLL approximation, proving convergence to the true ES update in the limit.

### 5.1 High-Dimensional Gaussian ES

We first analyse the general ES update under Gaussian perturbations from Eq. (2):

$$\nabla_{\mu} J(\theta) = \frac{1}{\sigma_d} \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [v \cdot f(\mu + \sigma_d v)],$$

where  $v \in \mathbb{R}^d$ . In high dimensions, the Gaussian annulus theorem (Vershynin, 2018; Wegner, 2024) proves that the probability mass of standard Gaussian distributions concentrates in thin shells of radius  $\sqrt{d}$ , which place probability mass further from the origin as dimension  $d$  increases. To counter this, we let  $\sigma_d$  depend on  $d$  and analyse the *critical decay rate* of  $\sigma_d$  that yields convergence of the ES updates. We make the following mild regularity assumptions:

**Assumption 2** (Locally Continuous Fitness). *With probability 1 with respect to the random initialisation of  $\mu$ , assume there exists a ball  $B_{\rho}(\mu) := \{x' \mid \|x' - \mu\| < \rho\}$  of fixed radius  $\rho > 0$  where  $f(x)$  is  $C^1$ -continuous for all  $x \in B_{\rho}(\mu)$ . Within this ball, let  $\nabla f(x)$  be  $\alpha$ -Hölder continuous, i.e.,  $\|\nabla f(x) - \nabla f(y)\| \leq L\|x - y\|^{\alpha}$  for all  $x, y \in B_{\rho}(\mu)$ ,  $\alpha \in (0, 1]$  and  $L = \mathcal{O}(1)$ .*

Assumption 2 does not restrict the fitness to be globally continuous; with probability one with respect to the initialisation distribution there must exist an arbitrarily small  $C^1$ -continuous ball around  $\mu$ . In particular, discontinuities, kinks, and non-differentiable regions may exist in the domain, provided they are not encountered with nonzero probability in the local region explored by the algorithm.  $\alpha$ -Hölder is the weakest simple, dimension-robust assumption that guarantees vanishing local gradient variation under Gaussian perturbations; it is weaker than Lipschitz continuity, which is recovered with  $\alpha = 1$ .

**Assumption 3** (Global Polynomial Growth). *Assume that there exists some constant  $0 < C < \infty$  that is  $\mathcal{O}(1)$  in  $d$  and finite polynomial degree  $p \geq 0$  such that  $|f(\mu + \sigma_d v)| \leq C(1 + \|\mu + \sigma_d v\|^p)$  and  $\|\nabla f(\mu + \sigma_d v)\| \leq C(1 + \|\mu + \sigma_d v\|^p)$  almost surely under  $v \sim \mathcal{N}(0, I_d)$ .*

Unlike Assumption 2, this is a *global* assumption. Again, discontinuities can exist. The assumption is weaker than boundedness, is satisfied by essentially all fitness functions used in ES, and ensures that both the objective and its gradient are integrable under Gaussian perturbations; objectives violating this condition typically exhibit super-polynomial growth and derivative growth, which leads to ill-defined or highly unstable ES updates. Moreover, if the condition is not satisfied almost surely, then the function and its gradients are undefined in regions that have nonzero Gaussian measure.

**Assumption 4** (Bounded Derivative). *With probability 1 with respect to the random initialisation of  $\mu$ , assume that  $\|\mu\| = \mathcal{O}(1)$  and  $\|\nabla f(\mu)\| = \mathcal{O}(1)$ , i.e.  $\|\mu\|$  and  $\|\nabla f(\mu)\|$  do not grow with increasing  $d$ .*

This assumption is standard in high-dimensional analysis proving convergence to linearity, as proving convergence to  $\nabla f(\mu)$  becomes meaningless if  $\|\nabla f(\mu)\| \rightarrow \infty$ . Moreover, the ES update as a whole can diverge if Assumption 4 is not satisfied. It can be ensured by scaling, typically by scaling networks parameters by  $d^{-\frac{1}{2}}$  or using an appropriate scaled initialisation, commonly Gaussian initialisation  $\mu \sim \mathcal{N}(0, \frac{1}{d} I_d)$ . This is precisely the scaling employed in the neural tangent kernel (NTK) regime (Jacot et al., 2018; Lee et al., 2019; Chizat et al., 2019), where it guarantees dimension-independent gradients and stable training dynamics.

These assumptions encompass essentially all objectives encountered in modern machine learning, including networks with finitely many ReLU activations, max- and hinge-based losses, and other piecewise-smooth or discontinuous models. Our first theorem proves convergence of a Gaussian ES update to a linearised form, that is to the local first-order derivative  $\nabla f(\mu)$ , with a tight convergence rate for any function satisfying these assumptions:---

**Theorem 1** (Convergence to Linearity). *Let Assumptions 2, 3, and 4 hold and  $\sigma_d = o\left(d^{-\frac{1}{2}}\right)$ . Then:*  
 $\|\nabla_{\mu} J(\theta) - \nabla f(\mu)\| = \Theta\left(\left(\sigma_d \sqrt{d}\right)^{\alpha}\right) = o(1)$ , almost surely with respect to the distribution over  $\mu$ .

To understand the effect that breaching the  $\sigma_d = o\left(d^{-\frac{1}{2}}\right)$  rate has on the convergence of Gaussian ES, we study the space of functions that can be represented by cubic polynomials of the form:

$$f(x) = a^{\top} x + \frac{1}{2} x^{\top} B x + \frac{1}{6} C(x, x, x), \quad (7)$$

where  $a \in \mathbb{R}^d$ ,  $B \in \mathbb{R}^{d \times d}$  is a symmetric matrix and  $C(x, x, x) = \sum_{i,j,k} c_{i,j,k} x_i x_j x_k$  denotes a symmetric 3-linear map represented by the symmetric 3-tensor  $C \in \mathbb{R}^{d \times d \times d}$ , which generalises cubic equations of the form  $f(x) = ax + bx^2 + cx^3$  to vector-valued  $x$ . These are non-pathological, well-behaved, analytic  $C^{\infty}$ -continuous functions, and include a rich subclass of convex optimisation problems, for instance, cubic perturbations of strictly convex quadratics. Moreover, any convex  $C^3$ -continuous objective admits a local third-order Taylor expansion of this form around a minimiser.

**Theorem 2** (Exact Divergence for Cubic Objectives). *Let  $f(x)$  denote the cubic polynomial in Eq. (7). Assume  $\|a\| = \mathcal{O}(1)$ ,  $\|B\| = \mathcal{O}(1)$ ,  $\|C\| = \mathcal{O}(1)$  where  $\|\cdot\|$  denotes operator norm for  $i$ -tensor  $T(x_1, \dots, x_i)$ :  $\|T\| := \sup_{\|x_1\|=\dots=\|x_i\|=1} |T(x_1, \dots, x_i)|$ . Let Assumption 4 hold, then:*

$$\nabla_{\mu} J(\theta) = \nabla f(\mu) + \frac{\sigma_d^2}{2} \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [C(v, v, \cdot)].$$

Moreover:

$$\left\| \frac{\sigma_d^2}{2} \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [C(v, v, \cdot)] \right\| = \Theta(\sigma_d^2 d), \quad (8)$$

$$\|\nabla_{\mu} J(\theta) - \nabla f(\mu)\| = \Theta(\sigma_d^2 d). \quad (9)$$

Together, Theorems 1 and 2 prove Gaussian ES has a *critical convergence rate* of  $\sigma_d = o\left(d^{-\frac{1}{2}}\right)$  in high dimensions, and operates in three regimes:

**Regime I (Convergence to Linearity):** For  $\sigma_d = o\left(d^{-\frac{1}{2}}\right)$ , ES converges to a linearised form, recovering a local first-order gradient update  $\nabla f(\mu)$ . This result is *analogous to neural tangent kernel (NTK) type theorems*, which prove that neural networks linearise in high dimensions (Jacot et al., 2018) and results from the concentration of the population distribution as  $d \rightarrow \infty$ , but applies to a more general set of objectives including discontinuous architectures. Moreover, Theorem 1 proves that the  $(\sigma_d \sqrt{d})^{\alpha}$  rate at which Gaussian ES converges is tight and cannot in general be improved upon without strengthening continuity or introducing specific structure into the objective to ensure the Hölder constant  $L$  decays with  $d$ ; for the class of cubic functions we consider in Theorem 2, the faster  $\sigma_d^2 d$  convergence rate found in Eq. (9) is possible due to the  $C^{\infty}$ -continuity of this function class, which means the converge rate is governed by third order derivative terms.

**Regime II (Critical):** For  $\sigma_d \asymp d^{-\frac{1}{2}}$ , Gaussian ES converges to a nonlinear limiting update that may retain higher-order derivative terms when they exist; for our cubic example, Eq. (8) proves that at this critical rate, the second-order term associated with the matrix  $B$  vanishes due to symmetry and the third-order term associated with the tensor  $C$  remains:

$$\left\| \frac{\sigma_d^2}{2} \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [C(v, v, \cdot)] \right\| = \Theta(1).$$

As the polynomial form is representative of general Taylor expansions, this implies that the limiting high dimensional update retains third-order derivatives (and higher order odd derivatives) as  $d \rightarrow \infty$ .**Regime III (Divergence):** For  $d^{-\frac{1}{2}} = o(\sigma_d)$ , Theorem 2 shows that there exist smooth cubic objectives with bounded coefficients for which:

$$\|\nabla_{\mu} J(\theta)\| = \Theta(\sigma_d^2 d) \rightarrow \infty.$$

In particular, divergence occurs whenever the cubic tensor has a non-vanishing Gaussian contraction (equivalently, non-zero partial trace), i.e. in non-degenerate cases; only in the exceptional trace-free case does the cubic contribution vanish.

In practice,  $\sigma_d$  is often absorbed into the ES update stepsize, and its scale is adjusted automatically as part of the hyperparameter regime to ensure stability.

## 5.2 High-Dimensional EGGROLL

We now extend our high-dimensional analysis to study the EGGROLL update using the Gaussian approximate score function  $\hat{g}_{\text{LR}}$  from Eq. (5). Taking  $r$  as fixed, we consider the Gaussian matrix ES setting outlined in Section 2.3. We take  $x = \text{Vec}(W)$  where  $W \in \mathbb{R}^{m \times n}$  and analyse the effect of increasing the total number of matrix parameters  $d = mn$ . Recall the true ES Gaussian matrix update is:

$$\nabla_M J(\theta) = \frac{1}{\sigma} \mathbb{E}_{E \sim P(E)} [E \cdot f(W = M + \sigma E)],$$

where  $M$  is the set of mean matrix parameters associated with the matrix  $W$  and  $P(E)$  is a zero-mean standard normal  $p(E) = \mathcal{N}(0, I_m, I_n)$ .

Two key differences between full-rank Gaussian ES and EGGROLL are that  $\hat{g}_{\text{LR}}$  is an approximation to a true gradient and  $P(E)$  may have heavier tails than a Gaussian. To account for these differences, we require a slightly stricter local continuity control assumption:

**Assumption 5** (EGGROLL Locally Continuous Fitness). *With probability 1 with respect to the random initialisation of  $\mu$ , assume there exists a ball  $B_{\rho}(\mu) := \{x' \mid \|x' - \mu\| < \rho\}$  of fixed radius  $\rho > 0$  where  $f(x)$  is  $C^2$ -continuous for all  $x \in B_{\rho}(\mu)$  and  $\|\nabla^2 f(\mu)\|$  be polynomial bounded in  $d$ . Within this ball, let  $\nabla^2 f(x)$  be Lipschitz continuous, i.e.  $\|\nabla^2 f(x) - \nabla^2 f(y)\| \leq L_d \|x - y\|$  for all  $x, y \in B_{\rho}(\mu)$ .*

This assumption still permits discontinuous objectives. We also assume that  $p_0(\cdot)$  generates sub-Gaussian elements with uniform tail control:

**Assumption 6** (Sub-Gaussian Tails). *In addition to Assumption 1, assume that  $p_0(\cdot)$  generates variables that have sub-Gaussian tails, i.e. for  $x_i \sim p_0(x_i)$ :*

$$\mathbb{P}(|x_i| > t) \leq 2 \exp(-Ct^2),$$

for some  $0 \leq C < \infty$  that does not depend on  $d$ .

We discuss sub-Gaussian variables and their properties in Section C.3 The assumption is trivially satisfied for Gaussian distributions  $a \sim \mathcal{N}(0, I_m)$  and  $b \sim \mathcal{N}(0, I_n)$ , and holds more generally, for example for bounded distributions, uniform distributions and generalised Gaussian distributions with shape parameter greater than two. This flexibility is particularly relevant for the models in Section 6.1, where heavier-shouldered distributions may be preferred over the Gaussian.

**Theorem 3** (EGGROLL Convergence to Linearity). *Let  $W \in \mathbb{R}^{m \times n}$ ,  $d = mn$  and  $x = \text{Vec}(W)$ . Let Assumptions 3, 4, 5 and 6 hold,  $\sigma_d = o(d^{-1/2})$ , and  $L_d(\sigma_d d)^2 = o(1)$ . Then there exists some  $K > 0$  such that:*

$$\|\hat{g}_{\text{LR}} - \nabla_W f(W = M)\|_F = \mathcal{O}(L_d(\sigma_d d)^2) + \mathcal{O}\left(\frac{\sqrt{d}}{\sigma_d^2} \exp\left(-K \frac{\rho}{\sqrt{d}\sigma_d}\right)\right) = o(1), \quad (10)$$

and

$$\|\hat{g}_{\text{LR}} - \nabla_M J(\theta)\|_F = \mathcal{O}\left(\sigma_d \sqrt{d} \cdot \left(1 + L_d \sigma_d d^{\frac{3}{2}}\right)\right) = o(1). \quad (11)$$

almost surely with respect to the distribution over  $\mu$ .Our theory explains the success of EGGROLL in high dimensions with rank as small as  $r = 1$ ; Eq. (11) proves EGGROLL converges to the true update matrix ES update  $\nabla_M J(\theta)$  as  $d \rightarrow \infty$  regardless of  $r$ . In addition, Eq. (10) proves that under the same conditions, the EGGROLL update also linearises like the true Gaussian ES update analysed in Section 5.1, recovering a local first-order derivative as  $d \rightarrow \infty$ . For high-dimensional neural networks, standard parametrisations place training in the NTK regime, in which the network behaves approximately linearly in its parameters and gradient descent converges to a global minimum (Jacot et al., 2018; Lee et al., 2019; Chizat et al., 2019). Recent results show that the spectral norm of the Hessian decays polynomially with width, and that higher-order derivatives governing the variation of the Hessian also vanish (Liu et al., 2020). Consequently, the Lipschitz constant  $L_d = o(1)$ , typically at rate  $d^{-\frac{1}{2}}$  or  $d^{-1}$  depending on the network architecture. Substituting these rates into our upper bound in Eq. (10) yields convergence rates of  $\mathcal{O}(\sigma_d^2 d^{\frac{3}{2}})$  or  $\mathcal{O}(\sigma_d^2 d)$  respectively.

### 5.3 Rank Analysis

We now analyse how fast the low-rank update from Eq. (4) with Gaussian score approximation converges to the true Gaussian ES matrix gradient in Eq. (3) as the rank of the update  $r$  increases. We make notation explicit in  $r$  in this subsection, for example writing  $E^r = \frac{1}{\sqrt{r}} A^r B^{r\top}$ . We introduce the following formal regularity assumption for the fitness function:

**Assumption 7** (Bounded Fitness). *Assume that  $f(W)$  is bounded, that is  $\sup_W |f(W)| < \infty$ .*

Our key theoretical result characterises the error rate between the Gaussian score approximator in the low-rank update  $\hat{g}_{\text{LR}}^r$  from Eq. (4) and the true gradient using the matrix Frobenius norm:

**Theorem 4** (EGGROLL Rank Convergence). *Let Assumptions 1 and 7 hold, then:*

$$\|\hat{g}_{\text{LR}}^r - \nabla_{\mu} J(\theta)\|_F = \mathcal{O}(r^{-1}). \quad (12)$$

The convergence rate in Eq. (12) is faster than the typical  $\mathcal{O}(r^{-\frac{1}{2}})$  rate dictated by the general parametric central limit theorem. Our analysis shows that this is due to the symmetry in our problem under Assumption 1. To obtain our results, we make an Edgeworth expansion (Bhattacharya & Ranga Rao, 1976) of the distribution  $P(E^r)$ , which expands  $P(E^r)$  as the limiting Gaussian distribution plus a sum of decaying terms that are controlled by the 3rd order and higher cumulants of  $P(E^r)$ . Each  $i$ th order cumulant term is multiplied by a factor that decays at rate  $\mathcal{O}(r^{-\frac{i-2}{2}})$ . For symmetric zero-mean distributions, all odd cumulants are zero (for the same reason that all odd moments of a symmetric distribution are zero). Hence, the rate of convergence to the limiting distribution is controlled by the 4th order term, which has rate  $\mathcal{O}(r^{-1})$ .

Although the full distribution  $P(E^r)$  has no general closed-form solution, the distribution over marginals  $P(E_{i,j})$  is more amenable to analysis. We derive the density of the marginal distribution  $P(E_{i,j})$  for generalised Gaussian distributed  $a_{i,j}$  and  $b_{i,j}$  in Section D.1. To illustrate the fast convergence rate, we plot the negative density  $\times$  score function  $p(E_{i,j})E_{i,j}$  for the marginal density  $p(E_{i,j})$  in Fig. 3 using Gaussian distributed  $a_{i,j}$  and  $b_{i,j}$  (see Theorem 6 for a derivation). The figure shows that  $p(E_{i,j})E_{i,j}$  quickly converges to the limiting function  $\frac{E_{i,j}}{\sqrt{2\pi}} \exp\left(-\frac{E_{i,j}^2}{2}\right)$ , recovering the Gaussian form from the true Gaussian ES update. Even at  $r = 1$ , the function is not a poor approximation. After  $r = 10$ , the function has nearly converged and after  $r = 50$ , the function is visually indistinguishable from the limit, providing evidence for the hypothesis that the low-rank approximation is accurate even for very low-rank regimes  $r \ll \min(m, n)$ .

Figure 3: Plot of Marginal Score Multiplied by Density for Increasing  $r$Figure 4: (a) Comparison of reinforcement learning returns normalised by PPO performance across 16 environments for 10 seeds. The shaded region is the standard error of the mean.(b) Validation score of 3 seeds of EGGROLL v.s. 3 seeds of GRPO in countdown task with an RWKV 7g1.5B model on a single GPU. EGGROLL allows 1024 parallel generations per GPU (618 updates) whereas GRPO only 64 (915 updates).

## 6 Experiments

In the following section we showcase the effectiveness of EGGROLL in a variety of tasks that position it as a strong alternative to back-propagation for the end-to-end training of foundation models.

### 6.1 Pure Integer Language Model Pretraining

To demonstrate the potential of EGGROLL as a general optimisation method, we apply it to language model pretraining. Since EGGROLL does not rely on gradients, we explicitly design a language model architecture to be efficient and hardware-friendly at inference time. To highlight EGGROLL’s flexibility, we train a nonlinear recurrent neural network (RNN) in pure integer datatypes with no explicit activation functions, relying only on the implicit nonlinearity of clipping in int8 operations. We call the resulting language model EGG, the Evolved Generative GRU, an EGGROLL-friendly architecture with all weights in int8. See Appendix G for more details on the architecture and motivation behind EGG.

We train an EGG model with 6 layers and hidden dimension 256 (6L-256D) to do character-level prediction on the minipile dataset (Kaddour, 2023). We update parameters after 100 tokens for each population member, applying truncated ES by keeping the hidden state and only resetting at document boundaries. We plot the test loss in Fig. 2b over training steps across a range of population sizes with a fixed data batch size of 16 sequences per step, where the best test loss is 3.40 bits/byte. With a sufficiently large population size, EGG outperforms a dense 6L-256D Transformer trained with backprop SGD using the same data batch size. Note that larger population sizes require more parallel compute for the same amount of data; our largest population size of  $2^{20} = 1048576$  requires around 180 times more GPU-hours than the backprop baseline, demonstrating the potential for compute-only scaling in limited data regimes using EGGROLL.

Moreover, our largest population size of  $2^{20}$  is three orders of magnitude larger than the largest experiment done by Salimans et al. (2017) while only requiring a single GPU to train, highlighting EGGROLL’s computational efficiency. We note that large population sizes are critical for pretraining; a population size of 2, analogous to MeZO (Malladi et al., 2023), significantly underperforms larger population sizes despite having access to the same data batch. We conduct more ablations in Appendix I, analysing the tradeoff between population size and data batch size.

### 6.2 Reinforcement Learning Tasks

To verify that low-rank perturbations do not change the optimisation behavior of ES in standard control settings, we benchmark EGGROLL against OpenES (Salimans et al., 2017) across 16 tabula rasa environments spanning Navix, Craftax, Brax, Kinetix, and Jumanji. We use a fixed 3-layer MLP policy (256 hidden units) and perform per-environment hyperparameter optimisation for each method before evaluating the selected configuration over 10 random seeds, reporting mean performance (normalised by PPO) and uncertainty. Overall, EGGROLL is competitive with OpenES on 7/16 environments, underperforms on 2/16, and outperforms on 7/16, while often delivering substantial wall-clock improvements due to its batched low-rank structure (full environmentFigure 5: (a) Comparison of the validation score of 3 seeds of EGGROLL v.s. 3 seeds of GRPO in GSM8K task with an RWKV 7g7B model on 8 GPUs. EGGROLL allows 8192 parallel generations (1024 per GPU with 260 updates) whereas GRPO only 256 (32 per GPU with 340 updates). (b) Performance of our finetuned RWKV 7G 7 billion model on hard reasoning tasks using 128 GPUs for 12 hours. The model was trained using the DeepScaleR dataset and the best checkpoint was chosen by evaluating on AIME24.

list, learning curves, timing comparisons, and complete HPO ranges/settings are provided in Appendix N.4). Figure 4a shows the averaged normalised return across the 16 environments with 10 seeds per environment. We additionally report MARL results in Section N.1.

### 6.3 Foundation Model Fine-tuning

We apply EGGROLL to finetune an RWKV-7 (Peng et al., 2025) LLM on two reasoning tasks: countdown (Gandhi et al., 2024) and GSM8K (Cobbe et al., 2021). RWKV is a recurrent model that is better suited to parallelisation than transformers because any memory otherwise spent on the KV cache is used to evaluate population members. Figure 4b shows that EGGROLL fine-tuning on an RWKV-7 1.5B model converges to a higher validation accuracy of 35% (vs. 23%) given the same hardware and wall-clock time in the countdown task. Similarly, Figure 5a shows that EGGROLL outperforms GRPO on GSM8K fine-tuning. Our scoring function draws parallels to the group relative advantage of GRPO. In particular, to score a set of noise directions,  $E \equiv \{E_1, \dots, E_n\}$ , we first compute their accuracies,  $\{s_{1,q_i}, \dots, s_{n,q_i}\}$ , on  $|q| = m$  questions, creating a matrix of scores  $S \in \mathbb{R}^{m \times n}$ . We then compute the normalised score per question, with the main difference that we use the global variance  $\bar{\sigma}$ , and average over all the questions to compute a score for the noise direction  $E_i$ :

$$\bar{s}_i = \frac{1}{m} \sum_{j=1}^m z_{i,q_j} = \frac{1}{m} \sum_{j=1}^m \frac{s_{i,j} - \mu_{q_j}}{\bar{\sigma}}.$$

This scoring function weights all questions within the same batch the same across population members. We use this recipe to train a 14 billion parameter RWKV 7 model on the DeepScaleR dataset and evaluate in more challenging maths reasoning tasks. In this regime, GRPO is infeasible due to the extra memory used by the Adam optimiser Kingma & Ba (2014). Using a thinking budget of 5000 tokens for training and evaluation, our fine-tuned 14B model improves from 13% to 30% accuracy on AIME24, from 7% to 33% accuracy on AIME25 and from 11% to 13% accuracy on HMMT25 after training on 32 GPUs for 12 hours (Figure 13b). On 7B models, we outperform GRPO using 128 GPUs for 24 hours (Figure 5b).

In Section L, we achieve similar performance to GRPO when fine-tuning Qwen Transformer models, and additionally demonstrate that EGGROLL can directly optimise for pass@k, a known limitation of GRPO (Yue et al., 2025). Beyond language models, we also fine-tune a finance world model into an agent for high-frequency trading that directly optimises for PnL; see Section M for more details.

### 6.4 Fine-tuning Integer Quantised LLMs

We follow the same procedure as Jacob et al. (2017) to quantise the RWKV-7 family of models by dividing by the maximum *per-channel* value on each weight matrix and mapping into the int8 range of  $[-127, 127]$ . We then apply EGGROLL with Adam to do model distillation from the original, non-quantised RWKV-7, into the resulting int8 quantised model using examples from GSM8K. See Appendix K for full details about theFigure 6: (a) Average per token perplexity (during training) of 3 seeds of a quantised (int8) RWKV 7G 7 billion parameter model on distillation from the non quantised model using examples from GSM8K. (b) Validation score on unseen examples of GSM8K of 3 seeds of a quantised RWKV 7G 7 billion parameter model. Initially the model is unable to solve any problems, but progressively it is capable of solving more problems. The baseline here indicates the validation score of a quantised model without any further training.

specifics of quantisation and fine-tuning. The distillation is done by matching the distributions between the quantised and non-quantised models on teacher forced examples (with solutions) from the GSM8K dataset. More specifically, the fitness for a given set of parameters,  $\mu_i$ , is computed as follows:

$$f_{\mu_i}(x_{1:T}) = \sum_{t=1}^T \text{KL}(p_t || q_t(\cdot; \mu_i)),$$

where  $x_{1:T}$  is a subsequence of tokens taken from the solutions of GSM8K and  $\text{KL}(p_t || q_t(\cdot; \mu_i))$  is the Kullback-Leibler divergence between the distribution of the non-quantised model,  $p_t$ , and the distribution of the quantised model  $q_t$  over the vocabulary at token  $t$ . Figure 6a shows the average per token perplexity of 3 seeds of a quantised RWKV 7G 7 billion parameter model compared to that of the original non-quantised model over the same sequence, as a baseline. Progressively, the quantised model recovers the capability to solve a subset of the GSM8K dataset (Figure 6b).

## 7 Conclusion

We introduce EGGROLL, a powerful method for black-box optimisation that scales evolutionary strategies to billion-parameter models and beyond using low-rank search matrices. Our experiments demonstrate that EGGROLL is effective with a rank of 1, giving substantial computational and memory savings for negligible decrease in performance when compared to the full-rank perturbations. Empirically, EGGROLL delivers large speedups over naïve ES in tabula rasa and multi-agent RL, and can power end-to-end training pipelines for foundation models. Our theoretical analysis shows that the EGGROLL update converges towards the Gaussian ES update with increasing rank  $r$  and parameter dimension  $d = mn$ , and we provide a rigorous study of general ES at high dimensions, deriving necessary and sufficient conditions for convergence and linearisation.

Looking forward, we can use EGGROLL for other problems beyond the reach of modern first-order gradient-based techniques. In particular, EGGROLL can enable the training of large scale end-to-end neurosymbolic systems (Sarker et al., 2021) with non-differentiable components. For instance, we can train neural networks that interface with symbolic modules for specific functions, like memory or calculations. We can also optimise end-to-end systems of language models, training them to be aware of inference-time harnesses and interactions with other agents in complex systems.

## Acknowledgements

Compute for this project is graciously provided by the Isambard-AI National AI Research Resource, under the projects “FLAIR 2025 Moonshot Projects” and “Robustness via Self-Play RL.” Some experiments also used compute generously given by JASMIN, the UK’s collaborative data analysis environment (<https://www.jasmin.ac.uk>).---

Bidipta Sarkar is supported by the Clarendon Fund Scholarship in partnership with a Department of Engineering Science Studentship for his Oxford DPhil. Mattie Fellows is funded by a generous grant from the UKRI Engineering and Physical Sciences Research Council EP/Y028481/1. Juan Agustin Duque is supported by the St-Pierre-Larochelle Scholarship at the University of Montreal and by Aaron Courville’s CIFAR AI Chair in Representations that Generalize Systematically. Jarek Liesen and Theo Wolf are supported by the EPSRC Centre for Doctoral Training in Autonomous Intelligent Machines & Systems EP/Y035070/1. Jarek Liesen is also supported by Sony Interactive Entertainment Europe Ltd. Uljad Berdica is supported by the EPSRC Centre for Doctoral Training in Autonomous Intelligent Machines & Systems EP/S024050/1 and the Rhodes Scholarship. Lukas Seier is supported by the Intelligent Earth CDT with funding from the UKRI grant number EP/Y030907/1. Alexander D. Goldie is funded by the EPSRC Centre for Doctoral Training in Autonomous Intelligent Machines and Systems EP/S024050/1. Jakob Nicolaus Foerster is partially funded by the UKRI grant EP/Y028481/1 (originally selected for funding by the ERC). Jakob Nicolaus Foerster is also supported by the JPMC Research Award and the Amazon Research Award.

We thank Andreas Kirsch for discovering an emergent log-linear scaling law for EEG loss with respect to int8 OPs in [this tweet](#) along with other community members for their comments and recommendations during the first arXiv release of this work.

## References

Agentica Organization, Michael Luo, Sijun Tan, and Justin Wong. DeepScaler-preview-dataset. <https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset>, 2025. Accessed: 2025-01-14.

R. Askey and R. (eds.) Roy. Nist digital library of mathematical functions, chapter 5: Gamma function. Online: <https://dlmf.nist.gov/5>, 2020-2026. Section 5.11 (Stirling / asymptotic expansions), release 1.1.16.

Anne Auger and Nikolaus Hansen. Theory of evolution strategies: A new perspective. In Anne Auger and Benjamin Doerr (eds.), *Theory of Randomized Search Heuristics: Foundations and Recent Developments*, pp. 289–325. World Scientific, Singapore, 2011.

Mislav Balunović, Jasper Dekoninck, Ivo Petrov, Nikola Jovanović, and Martin Vechev. Matharena: Evaluating llms on uncontaminated math competitions, 2026. URL <https://arxiv.org/abs/2505.23281>.

A. B. Basset. *A Treatise on Hydrodynamics: with numerous examples*, volume 2. Deighton, Bell, and Co., Cambridge, UK, 1888.

Yoshua Bengio, Réjean Ducharme, and Pascal Vincent. A neural probabilistic language model. In T. Leen, T. Dietterich, and V. Tresp (eds.), *Advances in Neural Information Processing Systems*, volume 13. MIT Press, 2000. URL [https://proceedings.neurips.cc/paper\\_files/paper/2000/file/728f206c2a01bf572b5940d7d9a8fa4c-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2000/file/728f206c2a01bf572b5940d7d9a8fa4c-Paper.pdf).

Hans-Georg Beyer. Toward a theory of evolution strategies: Self-adaptation. *Evolutionary Computation*, 3: 311–347, 1995. URL <https://api.semanticscholar.org/CorpusID:17416734>.

Hans-Georg Beyer and Hans-Paul Schwefel. Evolution strategies – a comprehensive introduction. *Natural Computing*, 1(1):3–52, 2002.

R. N. Bhattacharya and R. Ranga Rao. *Normal approximation and asymptotic expansions*. Wiley series in probability and mathematical statistics. Wiley, New York, 1976. ISBN 047107201X.

Clément Bonnet, Daniel Luo, Donal Byrne, Shikha Surana, Sasha Abramowitz, Paul Duckworth, Vincent Coyette, Laurence I. Midgley, Elshadai Tegegn, Tristan Kalloniatis, Omayma Mahjoub, Matthew Macfarlane, Andries P. Smit, Nathan Grinsztajn, Raphael Boige, Cemlyn N. Waters, Mohamed A. Mimouni, Ulrich A. Mbou Sob, Ruan de Kock, Siddarth Singh, Daniel Furelos-Blanco, Victor Le, Arnu Pretorius, and Alexandre Laterre. Jumanji: a diverse suite of scalable reinforcement learning environments in jax, 2024. URL <https://arxiv.org/abs/2306.09884>.---

Jean-Philippe Bouchaud, Julius Bonart, Jonathan Donier, and Martin Gould. *Trades, quotes and prices: financial markets under the microscope*. Cambridge University Press, 2018.

James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL <http://github.com/jax-ml/jax>.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020. URL <https://arxiv.org/abs/2005.14165>.

Lénaïc Chizat, Edouard Oyallon, and Francis Bach. On lazy training in differentiable programming. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (eds.), *Advances in Neural Information Processing Systems*, volume 32. Curran Associates, Inc., 2019. URL [https://proceedings.neurips.cc/paper\\_files/paper/2019/file/ae614c557843b1df326cb29c57225459-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2019/file/ae614c557843b1df326cb29c57225459-Paper.pdf).

Krzysztof M Choromanski, Aldo Pacchiano, Jack Parker-Holder, Yunhao Tang, and Vikas Sindhwani. From complexity to simplicity: Adaptive es-active subspaces for blackbox optimization. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (eds.), *Advances in Neural Information Processing Systems*, volume 32. Curran Associates, Inc., 2019. URL [https://proceedings.neurips.cc/paper\\_files/paper/2019/file/88bade49e98db8790df275fcebb37a13-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2019/file/88bade49e98db8790df275fcebb37a13-Paper.pdf).

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrman, Parker Schuh, Kensen Shi, Sashank Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayanan Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways. *J. Mach. Learn. Res.*, 24(1144), 2023. URL <https://jmlr.org/papers/volume24/22-1144/22-1144.pdf>.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021.

A. P. Dawid. Some matrix-variate distribution theory: Notational considerations and a bayesian application. *Biometrika*, 68(1):265–274, 1981. ISSN 0006-3444.

Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten P. Bosma, Zongwei Zhou, Tao Wang, Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, Kathleen Meier-Hellstern, Toju Duke, Lucas Dixon, Kun Zhang, Quoc Le, Yonghui Wu, Zhifeng Chen, and Claire Cui. Glam: Efficient scaling of language models with mixture-of-experts. In *Proceedings of the 39th International Conference on Machine Learning*, volume 162 of *Proceedings of Machine Learning Research*, pp. 5547–5569, Jul 2022. URL <https://proceedings.mlr.press/v162/du22c.html>.

William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: scaling to trillion parameter models with simple and efficient sparsity. *J. Mach. Learn. Res.*, 23(1):1–39, January 2022. ISSN 1532-4435. URL <https://jmlr.org/papers/volume23/21-0998/21-0998.pdf>.---

Maxim Fishman, Brian Chmiel, Ron Banner, and Daniel Soudry. Scaling fp8 training to trillion-token llms, 2025. URL <https://arxiv.org/abs/2409.12517>.

Jakob Nicolaus Foerster. Nonlinear computation in deep linear networks, sep 2017. URL <https://blog.openai.com/nonlinear-computation-in-linear-networks/>. Accessed: 2025-11-20.

Gerald B. Folland. *Real Analysis: Modern Techniques and Their Applications*. John Wiley & Sons, New York, 2nd edition, 1999. See Theorem 8.22 (Riemann–Lebesgue Lemma).

Catherine Forbes, Merran Evans, Nicholas Hastings, and Brian Peacock. *Statistical Distributions*. Wiley Series in Probability and Statistics. John Wiley & Sons, Hoboken, NJ, USA, 4th edition, 2011. ISBN 9780470390634.

C. Daniel Freeman, Erik Frey, Anton Raichuk, Sertan Girgin, Igor Mordatch, and Olivier Bachem. Brax – a differentiable physics engine for large scale rigid body simulation, 2021. URL <https://arxiv.org/abs/2106.13281>.

Sascha Yves Frey, Kang Li, Peer Nagy, Silvia Saporà, Christopher Lu, Stefan Zohren, Jakob Foerster, and Anisoara Calinescu. Jax-lob: A gpu-accelerated limit order book simulator to unlock large scale reinforcement learning for trading. In *Proceedings of the Fourth ACM International Conference on AI in Finance*, pp. 583–591, 2023.

Kevin Galim, Wonjun Kang, Yuchen Zeng, Hyung Il Koo, and Kangwook Lee. Parameter-efficient fine-tuning of state space models, 2025. URL <https://arxiv.org/abs/2410.09016>.

Matteo Gallici, Mattie Fellows, Benjamin Ellis, Bartomeu Pou, Ivan Masmitja, Jakob Nicolaus Foerster, and Mario Martin. Simplifying deep temporal difference learning. In *The Thirteenth International Conference on Learning Representations*, 2025. URL <https://openreview.net/forum?id=7IzeL0kflu>.

Kanishk Gandhi, Denise Lee, Gabriel Grand, Muxin Liu, Winson Cheng, Archit Sharma, and Noah D. Goodman. Stream of search (sos): Learning to search in language, 2024. URL <https://arxiv.org/abs/2404.03683>.

Jack Garbus and Jordan Pollack. Low rank factorizations are indirect encodings for deep neuroevolution. In *Proceedings of the Genetic and Evolutionary Computation Conference Companion*, GECCO ’25 Companion, pp. 2371–2379, New York, NY, USA, 2025. Association for Computing Machinery. ISBN 9798400714641. doi: 10.1145/3712255.3734297. URL <https://doi.org/10.1145/3712255.3734297>.

Alexander D. Goldie, Chris Lu, Matthew T. Jackson, Shimon Whiteson, and Jakob N. Foerster. Can Learned Optimization Make Reinforcement Learning Less Difficult? In *Advances in Neural Information Processing Systems*, volume 37, pp. 5454–5497, 2024.

Alexander David Goldie, Zilin Wang, Jaron Cohen, Jakob Nicolaus Foerster, and Shimon Whiteson. How Should We Meta-Learn Reinforcement Learning Algorithms? May 2025. URL <https://openreview.net/forum?id=jKzQ6af2DU>.

Ian Goodfellow, Yoshua Bengio, and Aaron Courville. *Deep Learning*. MIT Press, 2016. <http://www.deeplearningbook.org>.

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K.Q. Weinberger (eds.), *Advances in Neural Information Processing Systems*, volume 27. Curran Associates, Inc., 2014. URL [https://proceedings.neurips.cc/paper\\_files/paper/2014/file/f033ed80deb0234979a61f95710dbe25-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2014/file/f033ed80deb0234979a61f95710dbe25-Paper.pdf).

Martin D Gould, Mason A Porter, Stacy Williams, Mark McDonald, Daniel J Fenn, and Sam D Howison. Limit order books. *Quantitative Finance*, 13(11):1709–1742, 2013.---

I. S. (Izrail Solomonovich) Gradshtein, I. M. (Iosif Moiseevich) Ryzhik, Daniel Zwillinger, Victor Moll, and Inc Scripta Technica. *Table of integrals, series, and products*. Academic Press, San Diego ; Tokyo, 8 edition, 2015. ISBN 0123849330.

G R Grimmett and D R Stirzaker. Probability and random processes. *Journal of the Royal Statistical Society. Series A, Statistics in society*, 156(3):503–503, 1993. ISSN 0964-1998.

Peter Hall. *The bootstrap and Edgeworth expansion*. Springer series in statistics. Springer-Verlag, New York, 1992. ISBN 9780387945088.

Nikolaus Hansen. The cma evolution strategy: A tutorial, 2023. URL <https://arxiv.org/abs/1604.00772>.

Nikolaus Hansen and Andreas Ostermeier. Completely derandomized self-adaptation in evolution strategies. *Evolutionary Computation*, 9(2):159–195, 2001a.

Nikolaus Hansen and Andreas Ostermeier. Completely Derandomized Self-Adaptation in Evolution Strategies. *Evolutionary Computation*, 9(2):159–195, June 2001b. ISSN 1063-6560. doi: 10.1162/106365601750190398. URL <https://ieeexplore.ieee.org/document/6790628>.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems, 2024. URL <https://arxiv.org/abs/2402.14008>.

Joel Heck and Fathi M. Salem. Simplified minimal gated unit variations for recurrent neural networks, 2017. URL <https://arxiv.org/abs/1701.03452>.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset, 2021. URL <https://arxiv.org/abs/2103.03874>.

Sepp Hochreiter and Jürgen Schmidhuber. Lstm can solve hard long time lag problems. In M.C. Mozer, M. Jordan, and T. Petsche (eds.), *Advances in Neural Information Processing Systems*, volume 9. MIT Press, 1996. URL [https://proceedings.neurips.cc/paper\\_files/paper/1996/file/a4d2f0d23dcc84ce983ff9157f8b7f88-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/1996/file/a4d2f0d23dcc84ce983ff9157f8b7f88-Paper.pdf).

Mark Horowitz. 1.1 computing’s energy problem (and what we can do about it). In *2014 IEEE International Solid-State Circuits Conference Digest of Technical Papers (ISSCC)*, pp. 10–14, 2014. doi: 10.1109/ISSCC.2014.6757323.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In *ICLR*. OpenReview.net, 2022.

Ruihong Huang and Tomas Polak. LOBSTER: Limit order book reconstruction system. *Available at SSRN 1977207*, 2011.

Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. Quantization and training of neural networks for efficient integer-arithmetic-only inference, 2017. URL <https://arxiv.org/abs/1712.05877>.

Arthur Jacot, Franck Gabriel, and Clement Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), *Advances in Neural Information Processing Systems*, volume 31. Curran Associates, Inc., 2018. URL [https://proceedings.neurips.cc/paper\\_files/paper/2018/file/5a4belfa34e62bb8a6ec6b91d2462f5a-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2018/file/5a4belfa34e62bb8a6ec6b91d2462f5a-Paper.pdf).

Max Jaderberg, Valentin Dalibard, Simon Osindero, Wojciech M. Czarnecki, Jeff Donahue, Ali Razavi, Oriol Vinyals, Tim Green, Iain Dunning, Karen Simonyan, Chrisantha Fernando, and Koray Kavukcuoglu. Population Based Training of Neural Networks, November 2017. URL <http://arxiv.org/abs/1711.09846>. arXiv:1711.09846 [cs].---

Feihu Jin, Yifan Liu, and Ying Tan. Derivative-free optimization for low-rank adaptation in large language models. *IEEE/ACM Trans. Audio, Speech and Lang. Proc.*, 32:4607–4616, October 2024. ISSN 2329-9290. doi: 10.1109/TASLP.2024.3477330. URL <https://doi.org/10.1109/TASLP.2024.3477330>.

Jean Kaddour. The minipile challenge for data-efficient language models. *arXiv preprint arXiv:2304.08442*, 2023.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*, 2014.

Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In Yoshua Bengio and Yann LeCun (eds.), *2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings*, 2014. URL <http://arxiv.org/abs/1312.6114>.

Daria Korotyshova, Boris Shaposhnikov, Alexey Malakhov, Alexey Khokhulin, Nikita Surnachev, Kirill Ovcharenko, George Bredis, Alexey Gorbatovski, Viacheslav Sinii, and Daniil Gavrilov. Essa: Evolutionary strategies for scalable alignment, 2025. URL <https://arxiv.org/abs/2507.04453>.

John R. Koza. Genetic programming as a means for programming computers by natural selection. *Statistics and Computing*, 4(2):87–112, June 1994. ISSN 1573-1375. doi: 10.1007/BF00175355. URL <https://doi.org/10.1007/BF00175355>.

Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In F. Pereira, C.J. Burges, L. Bottou, and K.Q. Weinberger (eds.), *Advances in Neural Information Processing Systems*, volume 25. Curran Associates, Inc., 2012. URL [https://proceedings.neurips.cc/paper\\_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf).

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In *Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles*, 2023.

Robert Tjarko Lange, Tom Schaul, Yutian Chen, Tom Zahavy, Valentin Dallibard, Chris Lu, Satinder Singh, and Sebastian Flennerhag. Discovering Evolution Strategies via Meta-Black-Box Optimization, March 2023. URL <http://arxiv.org/abs/2211.11260>. arXiv:2211.11260 [cs].

Pierre-Simon Laplace. Mémoire sur les intégrales définies et leur application aux probabilités, et spécialement à la recherche du milieu qu’il faut choisir entre les résultats des observations. *Mémoires de la Classe des Sciences Mathématiques et Physiques de l’Institut Impérial de France*, 1<sup>re</sup> série, 11(1<sup>re</sup> partie):297–347, 1811.

Jaehoon Lee, Lechao Xiao, Samuel S. Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, and Jeffrey Pennington. Wide neural networks of any depth evolve as linear models under gradient descent. In *Proceedings of the 33rd International Conference on Neural Information Processing Systems*, Red Hook, NY, USA, 2019. Curran Associates Inc.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models, 2022. URL <https://arxiv.org/abs/2206.14858>.

Junjie Li, Yang Liu, Weiqing Liu, Shikai Fang, Lewen Wang, Chang Xu, and Jiang Bian. Mars: a financial market simulation engine powered by generative foundation model. In *The Thirteenth International Conference on Learning Representations*, 2025. URL <https://openreview.net/forum?id=Yqk7EyT52H>.---

Oscar Li, James Harrison, Jascha Sohl-Dickstein, Virginia Smith, and Luke Metz. Variance-reduced gradient estimation via noise-reuse in online evolution strategies. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023.

Elliott H. Lieb and Michael Loss. *Analysis*. Graduate studies in mathematics ; volume 14. American Mathematical Society, Providence, Rhode Island, 2nd ed. edition, 2010 - 2010. ISBN 1-4704-1143-1.

Jarek Liesen, Chris Lu, and Robert Lange. rejax, 2024. URL <https://github.com/kerajLi/rejax>.

Chaoyue Liu, Libin Zhu, and Mikhail Belkin. On the linearity of large non-linear models: when and why the tangent kernel is constant. In *Proceedings of the 34th International Conference on Neural Information Processing Systems*, NeurIPS 2020, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN 9781713829546.

Jun S. Liu. Siegel's formula via stein's identities. *Statistics and probability letters*, 21(3):247–251, 1994. ISSN 0167-7152.

Zichen Liu, Anya Sims, Keyu Duan, Changyu Chen, Simon Yu, Xiangxin Zhou, Haotian Xu, Shaopan Xiong, Bo Liu, Chenmien Tan, Chuen Yang Beh, Weixun Wang, Hao Zhu, Weiyuan Shi, Diyi Yang, Michael Shieh, Yee Whye Teh, Wee Sun Lee, and Min Lin. Gem: A gym for agentic llms, 2025. URL <https://arxiv.org/abs/2510.01051>.

Ryan Lowe, Aviv Tamar, Jean Harb, OpenAI Pieter Abbeel, and Igor Mordatch. Multi-agent actor-critic for mixed cooperative-competitive environments. *Advances in neural information processing systems*, 30, 2017.

Chris Lu, Jakub Kuba, Alistair Letcher, Luke Metz, Christian Schroeder de Witt, and Jakob Foerster. Discovered policy optimisation. *Advances in Neural Information Processing Systems*, 35:16455–16468, 2022.

H. M. Macdonald. Zeroes of the bessel functions. *Proceedings of the London Mathematical Society*, 30: 165–179, 1899. doi: 10.1112/plms/s1-30.1.165.

Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason D. Lee, Danqi Chen, and Sanjeev Arora. Fine-tuning language models with just forward passes. In *Proceedings of the 37th International Conference on Neural Information Processing Systems*, NIPS '23, Red Hook, NY, USA, 2023. Curran Associates Inc.

Michael Matthews, Michael Beukman, Benjamin Ellis, Mikayel Samvelyan, Matthew Jackson, Samuel Coward, and Jakob Foerster. Craftax: A lightning-fast benchmark for open-ended reinforcement learning. *arXiv preprint arXiv:2402.16801*, 2024.

Michael T. Matthews, Michael Beukman, Chris Lu, and Jakob Nicolaus Foerster. Kinetix: Investigating the training of general agents through open-ended physics-based control tasks. In *ICLR*, 2025. URL <https://openreview.net/forum?id=zCxGCDzreM>.

William Merrill, Jackson Petty, and Ashish Sabharwal. The illusion of state in state-space models. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*, pp. 35492–35506. PMLR, 21–27 Jul 2024. URL <https://proceedings.mlr.press/v235/merrill124a.html>.

Luke Metz, James Harrison, C. Daniel Freeman, Amil Merchant, Lucas Beyer, James Bradbury, Naman Agrawal, Ben Poole, Igor Mordatch, Adam Roberts, and Jascha Sohl-Dickstein. VeLO: Training Versatile Learned Optimizers by Scaling Up, November 2022. URL <http://arxiv.org/abs/2211.09760>. arXiv:2211.09760 [cs, math, stat].

Valentin Mohl, Sascha Frey, Reuben Leyland, Kang Li, George Nigmatulin, Mihai Cucuringu, Stefan Zohren, Jakob Foerster, and Anisoara Calinescu. Jaxmarl-hft: Gpu-accelerated large-scale multi-agent reinforcement learning for high-frequency trading. In *Proceedings of the 6th ACM International Conference on AI in Finance*, pp. 18–26, 2025. URL <https://doi.org/10.1145/3768292.3770416>.---

Peer Nagy, Sascha Frey, Silvia Sapora, Kang Li, Anisoara Calinescu, Stefan Zohren, and Jakob Foerster. Generative ai for end-to-end limit order book modelling: A token-level autoregressive generative model of message flow using a deep state space network. In *Proceedings of the Fourth ACM International Conference on AI in Finance*, ICAIF '23, pp. 91–99, 2023.

Peer Nagy, Sascha Yves Frey, Kang Li, Bidipta Sarkar, Svitlana Vyetrenko, Stefan Zohren, Ani Calinescu, and Jakob Nicolaus Foerster. LOB-bench: Benchmarking generative AI for finance - an application to limit order book data. In *Forty-second International Conference on Machine Learning*, 2025. URL <https://openreview.net/forum?id=CXPpYJpYXQ>.

Brian Ning, Franco Ho Ting Lin, and Sebastian Jaimungal. Double deep q-learning for optimal execution. *Applied Mathematical Finance*, 28(4):361–380, 2021.

Jack Parker-Holder, Vu Nguyen, and Stephen Roberts. Provably Efficient Online Hyperparameter Optimization with Population-Based Bandits, June 2021. URL <http://arxiv.org/abs/2002.02518>. arXiv:2002.02518 [cs].

Bo Peng, Ruichong Zhang, Daniel Goldstein, Eric Alcaide, Xingjian Du, Haowen Hou, Jiaju Lin, Jiaxing Liu, Janna Lu, William Merrill, Guangyu Song, Kaifeng Tan, Saiteja Utpala, Nathan Wilce, Johan S. Wind, Tianyi Wu, Daniel Wuttke, and Christian Zhou-Zheng. Rkv-7 "goose" with expressive dynamic state evolution, 2025. URL <https://arxiv.org/abs/2503.14456>.

K. B. Petersen and M. S. Pedersen. The matrix cookbook, nov 2012. URL <http://localhost/pubdb/p.php?3274>. Version 20121115.

Eduardo Pignatelli, Jarek Liesen, Robert Tjarko Lange, Chris Lu, Pablo Samuel Castro, and Laura Toni. Navix: Scaling minigrid environments with jax, 2024. URL <https://arxiv.org/abs/2407.19396>.

Xin Qiu, Yulu Gan, Conor F. Hayes, Qiyao Liang, Elliot Meyerson, Babak Hodjat, and Risto Miikkulainen. Evolution strategies at scale: Llm fine-tuning beyond reinforcement learning, 2025. URL <https://arxiv.org/abs/2509.24372>.

I. Rechenberg. Evolutionsstrategien. In Berthold Schneider and Ulrich Ranft (eds.), *Simulationsmethoden in der Medizin und Biologie*, pp. 83–114, Berlin, Heidelberg, 1978. Springer Berlin Heidelberg. ISBN 978-3-642-81283-5.

V. K. Rohatgi. *An introduction to probability theory and mathematical statistics*. Wiley series in probability and mathematical statistics. Wiley, New York, 1976. ISBN 0471731358.

Frank. Rosenblatt. *Principles of neurodynamics : perceptrons and the theory of brain mechanisms*. Spartan Books, Washington, 1962.

Alexander Rutherford, Benjamin Ellis, Matteo Gallici, Jonathan Cook, Andrei Lupu, Garðar Ingvarsson, Timon Willi, Ravi Hammond, Akbir Khan, Christian Schroeder de Witt, Alexandra Souly, Saptarashmi Bandyopadhyay, Mikayel Samvelyan, Minqi Jiang, Robert Tjarko Lange, Shimon Whiteson, Bruno Lacerda, Nick Hawes, Tim Rocktäschel, Chris Lu, and Jakob Nicolaus Foerster. JaxMARL: Multi-agent RL environments and algorithms in JAX. In *The Thirty-eighth Conference on Neural Information Processing Systems Datasets and Benchmarks Track*, 2024.

Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, and Ilya Sutskever. Evolution strategies as a scalable alternative to reinforcement learning, 2017. URL <https://arxiv.org/abs/1703.03864>.

John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw. Parallel random numbers: As easy as 1, 2, 3. In *SC '11: Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and Analysis*, pp. 1–12, 2011. doi: 10.1145/2063384.2063405.

Md Kamruzzaman Sarker, Lu Zhou, Aaron Eberhart, and Pascal Hitzler. Neuro-symbolic artificial intelligence: Current trends, 2021. URL <https://arxiv.org/abs/2105.05330>.---

Hans-Paul Schwefel. *Evolution and Optimum Seeking*. John Wiley & Sons, New York, 1995.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL <https://arxiv.org/abs/2402.03300>.

Zhihong Shao, Yuxiang Luo, Chengda Lu, Z. Z. Ren, Jiewen Hu, Tian Ye, Zhibin Gou, Shirong Ma, and Xiaokang Zhang. Deepseekmath-v2: Towards self-verifiable mathematical reasoning, 2025. URL <https://arxiv.org/abs/2511.22570>.

Jimmy TH Smith, Andrew Warrington, and Scott W Linderman. Simplified state space layers for sequence modeling. In *International Conference on Learning Representations*, 2023.

Jasper Snoek, Hugo Larochelle, and Ryan P. Adams. Practical bayesian optimization of machine learning algorithms, 2012. URL <https://arxiv.org/abs/1206.2944>.

Charles Stein. A bound for the error in the normal approximation to the distribution of a sum of dependent random variables. In *Proceedings of the Sixth Berkeley Symposium on Mathematical Statistics and Probability*, volume 2, pp. 583–602, Berkeley, CA, 1972. University of California Press.

Felipe Petroski Such, Vashisht Madhavan, Edoardo Conti, Joel Lehman, Kenneth O. Stanley, and Jeff Clune. Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning, April 2018. URL <http://arxiv.org/abs/1712.06567>. arXiv:1712.06567 [cs].

Laurits Tani, Diana Rand, Christian Veelken, and Mario Kadastik. Evolutionary algorithms for hyperparameter optimization in machine learning for application in high energy physics. *The European Physical Journal C*, 81(2):170, February 2021. ISSN 1434-6044, 1434-6052. doi: 10.1140/epjc/s10052-021-08950-y. URL <http://arxiv.org/abs/2011.04434>. arXiv:2011.04434 [hep-ex].

Nico M Temme. *Bessel Functions*, chapter 9, pp. 219–255. John Wiley and Sons, Ltd, 1996. ISBN 9781118032572. doi: <https://doi.org/10.1002/9781118032572.ch9>. URL <https://onlinelibrary.wiley.com/doi/abs/10.1002/9781118032572.ch9>.

Sebastian Towers, Aleksandra Kalisz, Philippe A. Robert, Alicia Higueroelo, Francesca Vianello, Ming-Han Chloe Tsai, Harrison Steel, and Jakob N. Foerster. ADIOS: Antibody Development via Opponent Shaping, June 2025. URL <http://arxiv.org/abs/2409.10588>. arXiv:2409.10588 [q-bio].

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), *Advances in Neural Information Processing Systems*, volume 30. Curran Associates, Inc., 2017. URL [https://proceedings.neurips.cc/paper\\_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).

Roman Vershynin. *High-Dimensional Probability: An Introduction with Applications in Data Science*. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, Cambridge, UK, 2018. ISBN 9781108415194. Foundational text covering concentration of norms and high-dimensional Gaussian phenomena.

Amala Mary Vincent and P. Jidesh. An improved hyperparameter optimization framework for AutoML systems using evolutionary algorithms. *Scientific Reports*, 13(1):4737, March 2023. ISSN 2045-2322. doi: 10.1038/s41598-023-32027-3. URL <https://doi.org/10.1038/s41598-023-32027-3>.

Martin J. Wainwright. *Basic tail and concentration bounds*, pp. 21–57. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2019.

Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Huaijie Wang, Lingxiao Ma, Fan Yang, Ruiping Wang, Yi Wu, and Furu Wei. Bitnet: Scaling 1-bit transformers for large language models, 2023. URL <https://arxiv.org/abs/2310.11453>.---

G. N. Watson. *A Treatise on the Theory of Bessel Functions*. Cambridge University Press, Cambridge, 2 edition, 1944. Reprinted with corrections, various later printings.

Sven A. Wegner. Gaussian random vectors in high dimensions. In *Mathematical Introduction to Data Science*, pp. 139–149. Springer, Berlin, Heidelberg, 2024. doi: 10.1007/978-3-662-69426-8\_10. Chapter proving and discussing the Gaussian annulus theorem.

G. B. Whitham. *Linear and nonlinear waves*. Pure and applied mathematics. Wiley-Interscience, New York, 1999. ISBN 9786613306241.

Daan Wierstra, Tom Schaul, Tobias Glasmachers, Yi Sun, and Jürgen Schmidhuber. Natural evolution strategies, 2011. URL <https://arxiv.org/abs/1106.4487>.

Samuel Webb Williams. *Auto-tuning performance on multicore computers*. PhD thesis, USA, 2008. AAI3353349.

C.S. Withers. A simple expression for the multivariate hermite polynomials. *Statistics and Probability Letters*, 47(2):165–169, 2000. ISSN 0167-7152. doi: [https://doi.org/10.1016/S0167-7152\(99\)00153-4](https://doi.org/10.1016/S0167-7152(99)00153-4). URL <https://www.sciencedirect.com/science/article/pii/S0167715299001534>.

Ke Xue, Chao Qian, Ling Xu, and Xudong Fei. Evolutionary gradient descent for non-convex optimization. In Zhi-Hua Zhou (ed.), *Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21*, pp. 3221–3227. International Joint Conferences on Artificial Intelligence Organization, 8 2021. doi: 10.24963/ijcai.2021/443. URL <https://doi.org/10.24963/ijcai.2021/443>. Main Track.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL <https://arxiv.org/abs/2505.09388>.

Ziming Yu, Pan Zhou, Sike Wang, Jia Li, Mi Tian, and Hua Huang. Zeroth-order fine-tuning of llms in random subspaces. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 4475–4485, October 2025.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? *arXiv preprint arXiv:2504.13837*, 2025.

Yihua Zhang, Pingzhi Li, Junyuan Hong, Jiaxiang Li, Yimeng Zhang, Wenqing Zheng, Pin-Yu Chen, Jason D. Lee, Wotao Yin, Mingyi Hong, Zhangyang Wang, Sijia Liu, and Tianlong Chen. Revisiting zeroth-order optimization for memory-efficient llm fine-tuning: A benchmark, 2024.---

## Appendix

<table><tr><td><b>A</b></td><td><b>Notation</b></td><td><b>26</b></td></tr><tr><td><b>B</b></td><td><b>ES Matrix Gradient Deviations</b></td><td><b>26</b></td></tr><tr><td><b>C</b></td><td><b>High-Dimensional Analysis</b></td><td><b>27</b></td></tr><tr><td>C.1</td><td>High-Dimensional Gaussian ES and Convergence . . . . .</td><td>27</td></tr><tr><td>C.2</td><td>Critical Convergence Rate . . . . .</td><td>32</td></tr><tr><td>C.3</td><td>EGGROLL Linearisation . . . . .</td><td>34</td></tr><tr><td><b>D</b></td><td><b>Asymptotic Rank Analysis</b></td><td><b>42</b></td></tr><tr><td>D.1</td><td>Mean Field Score Function Approximator . . . . .</td><td>46</td></tr><tr><td>D.2</td><td>Derivation of Mean-field Approximators . . . . .</td><td>47</td></tr><tr><td><b>E</b></td><td><b>EGGROLL Speed</b></td><td><b>52</b></td></tr><tr><td><b>F</b></td><td><b>Arithmetic Intensity Analysis</b></td><td><b>53</b></td></tr><tr><td>F.1</td><td>Arithmetic Intensity of Standard Batched Inference . . . . .</td><td>53</td></tr><tr><td>F.2</td><td>Arithmetic Intensity of Gaussian Matrix ES . . . . .</td><td>53</td></tr><tr><td>F.3</td><td>Arithmetic Intensity of EGGROLL . . . . .</td><td>54</td></tr><tr><td><b>G</b></td><td><b>EGG Architecture</b></td><td><b>55</b></td></tr><tr><td>G.1</td><td>Motivation . . . . .</td><td>55</td></tr><tr><td>G.2</td><td>Notation and Operations . . . . .</td><td>55</td></tr><tr><td>G.3</td><td>Parameter Initialisation . . . . .</td><td>56</td></tr><tr><td>G.4</td><td>Matrix Multiplication . . . . .</td><td>56</td></tr><tr><td>G.5</td><td>Embedding . . . . .</td><td>57</td></tr><tr><td>G.6</td><td>Layer Normalisation (LN) . . . . .</td><td>57</td></tr><tr><td>G.7</td><td>MLP . . . . .</td><td>57</td></tr><tr><td>G.8</td><td>GRU . . . . .</td><td>57</td></tr><tr><td>G.9</td><td>Fitness Calculation in Integer Types . . . . .</td><td>58</td></tr><tr><td><b>H</b></td><td><b>EGG Pretraining with Integer EGGROLL</b></td><td><b>58</b></td></tr><tr><td>H.1</td><td>Adding EGGROLL Perturbations . . . . .</td><td>58</td></tr><tr><td>H.2</td><td>Fitness Shaping . . . . .</td><td>58</td></tr><tr><td>H.3</td><td>Parameter Update . . . . .</td><td>58</td></tr><tr><td><b>I</b></td><td><b>EGG Ablations</b></td><td><b>59</b></td></tr><tr><td><b>J</b></td><td><b>Distributed EGGROLL Framework</b></td><td><b>60</b></td></tr><tr><td>J.1</td><td>Base-3 Fitness Packing and Bandwidth Efficiency . . . . .</td><td>60</td></tr><tr><td>J.2</td><td>System Architecture . . . . .</td><td>60</td></tr><tr><td><b>K</b></td><td><b>Fine-tuning of Integer Quantised Models</b></td><td><b>60</b></td></tr><tr><td>K.1</td><td>Quantisation Procedure . . . . .</td><td>60</td></tr></table>---

<table><tr><td>K.2</td><td>Integrating integer-quantised EGGROLL with Adam</td><td>60</td></tr><tr><td><b>L</b></td><td><b>Fine-tuning Pretrained Transformer LLMs with Verifiable Rewards</b></td><td><b>61</b></td></tr><tr><td>L.1</td><td>Results</td><td>61</td></tr><tr><td>L.2</td><td>Training Infrastructure for Large-Scale Transformer LLMs</td><td>62</td></tr><tr><td><b>M</b></td><td><b>Fine-tuning Time Series Foundation Model: High-Frequency Trading</b></td><td><b>64</b></td></tr><tr><td><b>N</b></td><td><b>Experimental Details</b></td><td><b>66</b></td></tr><tr><td>N.1</td><td>Multi Agent Reinforcement Learning Experiments</td><td>66</td></tr><tr><td>N.2</td><td>Reasoning Fine-tuning Experiments: Countdown</td><td>68</td></tr><tr><td>N.3</td><td>Reasoning Fine-tuning Experiments: GSM8K</td><td>69</td></tr><tr><td>N.4</td><td>Reinforcement Learning Experiments</td><td>69</td></tr></table>## A Notation

In our proofs, we use the integral notation  $\int$  to denote the integral over the corresponding  $\mathbb{R}^d$  space, for example, for a matrix  $E \in \mathbb{R}^{m \times n}$ ,  $\int f(E) dE = \int_{\mathbb{R}^{m \times n}} f(E) dE$  and for a vector  $E \in \mathbb{R}^{mn}$ ,  $\int f(v) dv = \int_{\mathbb{R}^{mn}} f(v) dv$ . For  $f : \mathbb{R}^d \rightarrow \mathbb{R}$ , we use  $\nabla f(x)$  to denote the derivative of  $f(\cdot)$  evaluated at  $x$ . For a vector  $v \in \mathbb{R}^{mn}$ , we define the mat operator as:

$$\text{mat}(v) = \begin{bmatrix} v_1 & v_{m+1} & \cdots & v_{(n-1)m+1} \\ v_2 & v_{m+2} & \cdots & v_{(n-1)m+2} \\ \vdots & \vdots & \ddots & \vdots \\ v_m & v_{2m} & \cdots & v_{mn} \end{bmatrix},$$

so  $\text{mat}(\text{vec}(M)) = M$ . We will use the fact that the Frobenius norm becomes the  $\ell_2$  norm in vector space:

$$\|M\|_F = \sqrt{\sum_{i,j} m_{i,j}^2} = \sqrt{\sum_k \text{vec}(M)_k^2} = \|\text{vec}(M)\|. \quad (13)$$

Our proofs make use of Fourier analysis. For a vector-valued function  $f(v) : \mathbb{R}^d \rightarrow \mathbb{R}$ , we define the Fourier transform as:

$$\tilde{f}(\omega) = \mathcal{F}[f](\omega) := \int f(v) \exp(-i\omega^\top v) dv,$$

and the inverse Fourier transform as:

$$f(v) = \mathcal{F}^{-1}[\tilde{f}](v) := \frac{1}{(2\pi)^d} \int \tilde{f}(\omega) \exp(i\omega^\top v) d\omega,$$

## B ES Matrix Gradient Deviations

Let  $\mu_M = \text{vec}(M) \in \mathbb{R}^{mn}$  be the vector of mean parameters associated with the matrix  $M$ . Let  $v_M \in \mathbb{R}^{mn}$  denote the corresponding search vector associated with  $\mu_M$ . As each element of  $v$  is generated independently from a standard normal  $\mathcal{N}(0, 1)$ , the search vector  $v_M$  is generated from the standard multivariate norm:  $v_M \sim \mathcal{N}(0, I_{mn})$ . From Eq. (2), the update for  $\mu_M$  is:

$$\begin{aligned} \sigma \nabla_{\mu_M} J(\theta) &= \mathbb{E}_{v_M \sim \mathcal{N}(0, I_{mn})} [v_M \cdot f(W = \text{mat}(\mu_M) + \sigma \text{mat}(v_M))], \\ &= \mathbb{E}_{v_M \sim \mathcal{N}(0, I_{mn})} [\text{vec}(\text{mat}(v_M)) \cdot f(W = \text{mat}(\mu_M) + \sigma \text{mat}(v_M))], \\ &= \mathbb{E}_{E \sim \mathcal{N}(0, I_m, I_n)} [\text{vec}(E) \cdot f(W = M + \sigma E)], \end{aligned}$$

where  $E = \text{mat}(v_M)$  and we have used the fact that sampling  $v_M \sim \mathcal{N}(0, I_{mn})$  is equivalent to sampling  $E \sim \mathcal{N}(0, I_m, I_n)$  and applying  $v_M = \text{vec}(E)$ . Now

$$\begin{aligned} \nabla_M J(\theta) &= \text{mat}(\nabla_{\mu_M} J(\theta)), \\ &= \frac{1}{\sigma} \mathbb{E}_{E \sim \mathcal{N}(0, I_m, I_n)} [\text{mat}(\text{vec}(E)) \cdot f(W = M + \sigma E)], \\ &= \frac{1}{\sigma} \mathbb{E}_{E \sim \mathcal{N}(0, I_m, I_n)} [E \cdot f(W = M + \sigma E)], \\ &= -\frac{1}{\sigma} \mathbb{E}_{E \sim \mathcal{N}(0, I_m, I_n)} [\nabla_E \log p(E) \cdot f(W = M + \sigma E)]. \end{aligned}$$## C High-Dimensional Analysis

### C.1 High-Dimensional Gaussian ES and Convergence

We use insights from the Gaussian annulus theorem when investigating the convergence properties of high-dimensional ES: our proof relies on the fact that all probability mass converges to the interior of the ball  $B_\epsilon(\mu) := \{x' \mid \|x' - \mu\| < \epsilon\}$  where  $\epsilon = \frac{\rho}{2}$  in the limit  $d \rightarrow \infty$ , where  $\rho$  is the radius of the local ball from Assumption 2, meaning we only need to consider the smooth region around  $\mu$  in this limit. Our first result proves that the mass outside of the ball for any polynomially bounded function tends to zero at an exponential rate.

**Lemma 1** (Polynomial Tail Bounds). *Let  $g(x)$  be polynomial bounded as:*

$$\|g(\mu + \sigma_d v)\| \leq C\|v\|^q(1 + \|\mu + \sigma_d v\|^p),$$

for some finite polynomial of orders  $p$  and  $q$  and constant  $C > 0$ . Let  $A_d := \{\|\sigma_d v\| \geq \epsilon\}$  denote the event that a mutation lies outside the a local ball of radius  $\epsilon$  around  $\mu$ . Assume  $\sigma_d = o(d^{-1/2})$ . Then for some constant  $K > 0$ :

$$\|\mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [g(\mu + \sigma_d v) \mathbf{1}(A_d)]\| = \mathcal{O} \left( d^{\frac{q}{2}} \exp \left( -K \left( \frac{\epsilon}{\sigma_d} \right)^2 \right) \right),$$

and in particular the right-hand side is  $o(1)$  as  $d \rightarrow \infty$ .

*Proof.* We start by bounding the integrand using the polynomial bound. Denote  $\mathbb{P}(A_d) := \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\mathbf{1}(A_d)]$ . Then, by Jensen's inequality in the first line, polynomial boundedness in the second and  $\|a + b\|^p \leq 2^{p-1}(\|a\|^p + \|b\|^p)$  in the third:

$$\begin{aligned} \|\mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [g(\mu + \sigma_d v) \mathbf{1}(A_d)]\| &\leq \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\|g(\mu + \sigma_d v)\| \mathbf{1}(A_d)], \\ &\leq C \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\|v\|^q (1 + \|\mu + \sigma_d v\|^p) \mathbf{1}(A_d)], \\ &\leq C \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\|v\|^q (1 + 2^{p-1} \|\mu\|^p) \mathbf{1}(A_d) + 2^{p-1} \sigma_d^p \|v\|^{p+q} \mathbf{1}(A_d)], \\ &= C' \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\|v\|^q \mathbf{1}(A_d)] + C'' \sigma_d^p \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\|v\|^{p+q} \mathbf{1}(A_d)]. \end{aligned}$$

where  $C' = C(1 + 2^{p-1} \|\mu\|^p)$  and  $C'' = C 2^{p-1}$  are constants independent of  $d$ . Applying the Cauchy-Schwarz inequality to the second expectation gives:

$$\mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\|v\|^{p+q} \mathbf{1}(A_d)] \leq \sqrt{\mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\|v\|^{2(p+q)}]} \cdot \sqrt{\mathbb{P}(A_d)}.$$

Now, the variable  $\|v\|$  is  $\chi_d$ -distributed. Using the formula for the  $i$ -th central moment of  $\|v\|$  about the origin (Forbes et al., 2011, Chapter 11.3) yields:

$$\mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\|v\|^i] = 2^{\frac{i}{2}} \frac{\Gamma(\frac{1}{2}(d+i))}{\Gamma(\frac{1}{2}d)}.$$

Applying the identity  $\frac{\Gamma(z+a)}{\Gamma(z+b)} \sim z^{a-b}$  (Askey & Roy, 2020-2026, Eq. 5.11.12):

$$\mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\|v\|^i] \sim 2^{\frac{i}{2}} \left( \frac{d}{2} \right)^{\frac{i}{2}} = d^{\frac{i}{2}}, \quad (14)$$

where  $\sim$  denotes asymptotic equivalence. For  $i = 2(p+q)$ , this yields the bound:

$$\mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\|v\|^{2(p+q)}] = \mathcal{O}(d^{p+q}),$$

hence:

$$\begin{aligned} \|\mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [g(\mu + \sigma_d v) \mathbf{1}(A_d)]\| &\leq C' d^{\frac{q}{2}} \sqrt{\mathbb{P}(A_d)} + C'' \sigma_d^p d^{\frac{p+q}{2}} \sqrt{\mathbb{P}(A_d)}, \\ &= (C' + C'' \sigma_d^p d^{\frac{p}{2}}) d^{\frac{q}{2}} \sqrt{\mathbb{P}(A_d)}, \end{aligned} \quad (15)$$We use the Gaussian concentration inequality for the Euclidean norm (Vershynin, 2018, Theorem 3.1.1), which states that for  $x \sim \mathcal{N}(0, I_d)$  there exists an absolute constant  $K > 0$  such that for all  $t \geq 0$ ,

$$\mathbb{P}\left(\left|\|x\| - \sqrt{d}\right| \geq t\right) \leq 2 \exp(-Kt^2).$$

In our setting, we need to bound:

$$\mathbb{P}(A_d) = \mathbb{P}(\|\sigma_d v\| \geq \epsilon) = \mathbb{P}\left(\|v\| \geq \frac{\epsilon}{\sigma_d}\right) = \mathbb{P}\left(\|v\| - \sqrt{d} \geq \frac{\epsilon}{\sigma_d} - \sqrt{d}\right).$$

Setting  $t = \frac{\epsilon}{\sigma_d} - \sqrt{d}$ , the assumption  $\sqrt{d}\sigma_d = o(1)$  implies for sufficiently large  $d$  that  $\sqrt{d}\sigma_d \leq \epsilon$  and therefore  $t \geq 0$ , so we can apply the concentration bound to obtain:

$$\begin{aligned} \mathbb{P}(A_d) &= \mathbb{P}\left(\|v\| - \sqrt{d} \geq t\right) \leq \mathbb{P}\left(\left|\|v\| - \sqrt{d}\right| \geq t\right), \\ &= \mathcal{O}\left(\exp\left(-K\left(\frac{\epsilon}{\sigma_d} - \sqrt{d}\right)^2\right)\right) = \mathcal{O}\left(\exp\left(-K\left(\frac{\epsilon}{\sigma_d}\right)^2\left(1 - \frac{\sigma_d\sqrt{d}}{\epsilon}\right)^2\right)\right). \end{aligned} \quad (16)$$

Now, as  $\sqrt{d}\sigma_d = o(1)$ , it follows  $\frac{\sigma_d\sqrt{d}}{\epsilon} = o(1)$ , yielding:

$$\begin{aligned} \mathbb{P}(A_d) &= \mathcal{O}\left(\exp\left(-K\left(\frac{\epsilon}{\sigma_d}\right)^2\right)\right), \\ \implies \sqrt{\mathbb{P}(A_d)} &= \mathcal{O}\left(\exp\left(-\frac{K}{2}\left(\frac{\epsilon}{\sigma_d}\right)^2\right)\right), \\ \implies d^{\frac{q}{2}}\sqrt{\mathbb{P}(A_d)} &= \mathcal{O}\left(d^{\frac{q}{2}}\exp\left(-\frac{K}{2}\left(\frac{\epsilon}{\sigma_d}\right)^2\right)\right), \end{aligned}$$

Applying these results to Eq. (15), along with  $\sigma_d^p d^{\frac{p}{2}} = \mathcal{O}(d^{\frac{-p}{2}})d^{\frac{p}{2}} = \mathcal{O}(1)$ , yields our desired result:

$$\begin{aligned} \|\mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [g(\mu + \sigma_d v) \mathbf{1}(A_d)]\| &\leq C' \mathcal{O}\left(d^{\frac{q}{2}} \exp\left(-\frac{K}{2}\left(\frac{\epsilon}{\sigma_d}\right)^2\right)\right) \\ &\quad + C'' \mathcal{O}(1) \mathcal{O}\left(d^{\frac{q}{2}} \exp\left(-\frac{K}{2}\left(\frac{\epsilon}{\sigma_d}\right)^2\right)\right), \\ &= \mathcal{O}\left(d^{\frac{q}{2}} \exp\left(-K\left(\frac{\epsilon}{\sigma_d}\right)^2\right)\right). \end{aligned}$$

where we have absorbed the factor of  $\frac{1}{2}$  into the constant  $K$ .  $\square$

Our proof in Lemma 1 reveals the necessity of the condition  $\sigma_d\sqrt{d} = o(1)$  for convergence as we can only apply the Gaussian concentration inequality in Eq. (16) for  $\sigma_d\sqrt{d} = o(1)$ ; this is a direct consequence of the Gaussian annulus theorem, as for slower rates  $1 = o(\sigma_d\sqrt{d})$ , the Gaussian probability mass will exit any local ball around  $\mu$  and flood the tail, meaning that the tail probability will grow with increasing  $d$ . Having bounded the tail, convergence to linearity follows by proving convergence within the ball, which allows us to exploit the local  $C^1$  smoothness of  $f(x)$ :

**Theorem 1** (Convergence to Linearity). *Let Assumptions 2, 3 and 4 hold and  $\sigma_d = o\left(d^{-\frac{1}{2}}\right)$ . Then:*

$$\|\nabla_{\mu} J(\theta) - \nabla f(\mu)\| = \Theta\left((\sigma_d\sqrt{d})^{\alpha}\right) = o(1),$$

*almost surely with respect to the distribution over  $\mu$ .**Proof.* We start with the definition of the ES update:

$$\nabla_{\mu} J(\theta) = \frac{1}{\sigma_d} \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [v \cdot f(\mu + \sigma_d v)].$$

Now let  $\epsilon = \frac{\rho}{2}$  where  $\rho$  is the radius of the ball from Assumption 2. Consider the hinge function:

$$\phi(x) = \begin{cases} 1, & \|x\| \leq \epsilon, \\ 2 - \frac{\|x\|}{\epsilon}, & \epsilon < \|x\| < 2\epsilon, \\ 0, & \|x\| \geq 2\epsilon, \end{cases}$$

which interpolates between 1 and 0 in the region  $\epsilon < \|x\| < 2\epsilon$ . Our first goal is to use  $\phi(x)$  to generate a function  $\tilde{f}(x)$  that is absolutely continuous and has integrable derivatives outside of  $B_{\rho}(\mu)$  to allow us to apply Stein's lemma (Stein, 1972). We define  $\tilde{f}(x)$  as:

$$\tilde{f}(x) = f(x) \cdot \phi(x - \mu)$$

Consider the closed ball  $B_{\epsilon}(\mu) := \{x' \mid \|x' - \mu\| \leq \epsilon\}$ . We note that within the ball  $f(\mu + \sigma_d v)$  remains unchanged:

$$\tilde{f}(\mu + \sigma_d v) = \begin{cases} f(\mu + \sigma_d v), & \|\sigma_d v\| \leq \epsilon, \\ f(\mu + \sigma_d v) \cdot \left(2 - \frac{\|\sigma_d v\|}{\epsilon}\right), & \epsilon < \|\sigma_d v\| < 2\epsilon, \\ 0, & \|\sigma_d v\| \geq 2\epsilon. \end{cases} \quad (17)$$

The derivative of the function with respect to  $v$  is:

$$\nabla_v \tilde{f}(\mu + \sigma_d v) = \begin{cases} \sigma_d \nabla f(\mu + \sigma_d v), & \|\sigma_d v\| \leq \epsilon, \\ \sigma_d \nabla f(\mu + \sigma_d v) \cdot \left(2 - \frac{\|\sigma_d v\|}{\epsilon}\right) - \frac{\sigma_d v}{\epsilon \|v\|} \cdot f(\mu + \sigma_d v), & \epsilon < \|\sigma_d v\| < 2\epsilon, \\ 0, & \|\sigma_d v\| \geq 2\epsilon. \end{cases} \quad (18)$$

where the gradient fails to exist only on the sets  $\|\sigma_d v\| \in \{\epsilon, 2\epsilon\}$ , which have Lebesgue measure zero. We start by using this function to decompose  $J(\mu)$  into a smoothed part and a remainder:

$$\nabla_{\mu} J(\theta) = \underbrace{\frac{1}{\sigma_d} \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [v \cdot \tilde{f}(\mu + \sigma_d v)]}_{:= \nabla_{\mu} \tilde{J}(\mu)} + \underbrace{\frac{1}{\sigma_d} \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [v \cdot (f(\mu + \sigma_d v) - \tilde{f}(\mu + \sigma_d v))]}_{:= \Delta(\mu)},$$

Hence:

$$\|\nabla_{\mu} J(\theta) - \nabla f(\mu)\| \leq \|\nabla_{\mu} \tilde{J}(\mu) - \nabla f(\mu)\| + \|\Delta(\mu)\|. \quad (19)$$

Consider the smoothed part:

$$\nabla_{\mu} \tilde{J}(\mu) := \frac{1}{\sigma_d} \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [v \cdot \tilde{f}(\mu + \sigma_d v)].$$

Our goal is to apply Stein's lemma (Stein, 1972) in its multivariate form (Liu, 1994, Lemma 1). The assumptions of (Liu, 1994, Lemma 1) require that the partial derivatives  $\partial_{v_i} \tilde{f}(\mu + \sigma_d v)$  are absolutely continuous almost everywhere and:

$$\mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [|\partial_{v_i} \tilde{f}(\mu + \sigma_d v)|] < \infty.$$

These two conditions are satisfied by construction. Indeed, under Assumption 2,  $f(\cdot)$  is  $C^1$  continuous on  $B_{\rho}(\mu)$ , hence from Eq. (17),  $\tilde{f}(\cdot)$  coincides with a compactly supported, piecewise  $C^1$  function whose gradient (Eq. (18)) exists almost everywhere. Moreover, under Assumption 3, both  $f(\mu + \sigma_d v)$  and  $\nabla f(\mu + \sigma_d v)$  are polynomially bounded, and since  $\nabla \tilde{f}(\mu + \sigma_d v)$  is supported on  $\|\sigma_d v\| \leq 2\epsilon$ , it follows that:

$$\mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\|\nabla \tilde{f}(\mu + \sigma_d v)\|] < \infty.$$Applying (Liu, 1994, Lemma 1):

$$\begin{aligned}
\frac{1}{\sigma_d} \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [v \cdot f(\mu + \sigma_d v)] &= \frac{1}{\sigma_d} \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\nabla_v \tilde{f}(\mu + \sigma_d v)], \\
&= \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\nabla \tilde{f}(\mu + \sigma_d v)], \\
\implies \|\nabla_\mu \tilde{J}(\mu) - \nabla f(\mu)\| &= \|\mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\nabla \tilde{f}(\mu + \sigma_d v) - \nabla f(\mu)]\| \\
&\leq \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\|\nabla \tilde{f}(\mu + \sigma_d v) - \nabla f(\mu)\|].
\end{aligned} \tag{20}$$

Let  $\{\mu + \sigma_d v \in B_\epsilon(\mu)\} = \{\|\sigma_d v\| \leq \epsilon\}$  denote the event that a mutation lies within the ball  $B_\epsilon(\mu)$ . We now split the integral into two regions, the first within the ball and the second outside:

$$\begin{aligned}
\mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\|\nabla f(\mu + \sigma_d v) - \nabla f(\mu)\|] &= \underbrace{\mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\|\nabla \tilde{f}(\mu + \sigma_d v) - \nabla f(\mu)\| \mathbf{1}(\|\sigma_d v\| \leq \epsilon)]}_{:= I_{\text{loc}}} \\
&\quad + \underbrace{\mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\|\nabla \tilde{f}(\mu + \sigma_d v) - \nabla f(\mu)\| \mathbf{1}(\|\sigma_d v\| > \epsilon)]}_{:= I_{\text{tail}}}.
\end{aligned}$$

Consider the region inside the ball,  $I_{\text{loc}}$ . From Eq. (18),  $\nabla \tilde{f}(\mu + \sigma_d v) = \nabla f(\mu + \sigma_d v)$  within this region. Using the local  $\alpha$ -Hölder continuity from Assumption 2:

$$\begin{aligned}
I_{\text{loc}} &= \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\|\nabla f(\mu + \sigma_d v) - \nabla f(\mu)\| \mathbf{1}(\|\sigma_d v\| \leq \epsilon)], \\
&\leq L \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\|\sigma_d v\|^\alpha \mathbf{1}(\|\sigma_d v\| \leq \epsilon)], \\
&\leq \sigma_d^\alpha L \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\|v\|^\alpha].
\end{aligned}$$

Now, applying the identity  $\mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\|v\|^i] \sim d^{\frac{i}{2}}$ , from Eq. (14):

$$I_{\text{loc}} = \mathcal{O} \left( (\sigma_d \sqrt{d})^\alpha \right).$$

We now bound the tail region outside the ball:

$$\begin{aligned}
I_{\text{tail}} &= \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\|\nabla \tilde{f}(\mu + \sigma_d v) - \nabla f(\mu)\| \mathbf{1}(\|\sigma_d v\| > \epsilon)], \\
&\leq \mathbb{E}_{v \sim \mathcal{N}(0, I_d)} [\|\nabla \tilde{f}(\mu + \sigma_d v) - \nabla f(\mu)\| \mathbf{1}(\|\sigma_d v\| \geq \epsilon)].
\end{aligned}$$

Now, as  $\|\nabla f(\mu)\| = \mathcal{O}(1)$  from Assumption 4 and we have established that  $\|\nabla \tilde{f}(\mu + \sigma_d v)\|$  is polynomial bounded under Assumption 3 when applying Stein's lemma, it follows that  $\|\nabla \tilde{f}(\mu + \sigma_d v) - \nabla f(\mu)\|$  is also polynomial bounded, that is there exists some constant  $C > 0$  and finite polynomial order  $p$  such that:

$$\|\nabla \tilde{f}(\mu + \sigma_d v) - \nabla f(\mu)\| \leq C(1 + \|\mu + \sigma_d v\|^p).$$

Applying Lemma 1, it follows:

$$I_{\text{tail}} = \mathcal{O} \left( \exp \left( -K \left( \frac{\epsilon}{\sigma_d} \right)^2 \right) \right),$$

for some constant  $K > 0$ . Together, this yields:

$$\begin{aligned}
\|\nabla_\mu \tilde{J}(\mu) - \nabla f(\mu)\| &= I_{\text{loc}} + I_{\text{tail}}, \\
&= \mathcal{O} \left( (\sigma_d \sqrt{d})^\alpha \right) + \mathcal{O} \left( \exp \left( -K \left( \frac{\epsilon}{\sigma_d} \right)^2 \right) \right).
\end{aligned}$$

As  $\exp(-x) = o(x^{-a})$  for any  $a > 0$ , we take  $a = \alpha/2$  to obtain a weakened bound matching the first term:

$$\exp \left( -K \left( \frac{\epsilon}{\sigma_d} \right)^2 \right) = o \left( \left( \frac{\sigma_d}{\epsilon} \right)^\alpha \right) = o \left( (\sigma_d \sqrt{d})^\alpha \right).$$

