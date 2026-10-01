Title: iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter

URL Source: https://arxiv.org/html/2506.00731

Markdown Content:
Binghang Lu 

School of Electrical and Computer Engineering 

Purdue University 

610 Purdue Mall, West Lafayette, 47907, IN 

lu895@purdue.edu

&Changhong Mou 

Department of Mathematics 

Purdue University 

610 Purdue Mall, West Lafayette, 47907, IN 

mouc@purdue.edu

&Guang Lin 

Department of Mathematics and School of Mechanical Engineering 

Purdue University 

610 Purdue Mall, West Lafayette, 47907, IN 

guanglin@purdue.edu

###### Abstract

Physics-informed neural networks (PINNs) have emerged as a powerful tool for solving forward and inverse problems involving partial differential equations (PDEs) by incorporating physical laws into the training process. However, the performance of PINNs is often hindered in real-world scenarios involving noisy observational data and missing physics, particularly in inverse problems. In this work, we propose an iterative multi-objective PINN ensemble Kalman filter (iPINNER) framework that improves the robustness and accuracy of PINNs in both forward and inverse problems by using the ensemble Kalman filter and the non-dominated sorting genetic algorithm III (NSGA-III). Specifically, NSGA-III is used as a multi-objective optimizer that can generate various ensemble members of PINNs along the optimal Pareto front, while accounting the model uncertainty in the solution space. These ensemble members are then utilized within the EnKF to assimilate noisy observational data. The EnKF’s analysis is subsequently used to refine the data loss component for retraining the PINNs, thereby iteratively updating their parameters. The iterative procedure generates improved solutions to the PDEs. The proposed method is tested on two benchmark problems: the one-dimensional viscous Burgers equation and the time-fractional mixed diffusion-wave equation (TFMDWE). The numerical results show it outperforms standard PINNs in handling noisy data and missing physics.

1 Introduction
--------------

The rapid advancement of machine learning and artificial intelligence has profoundly influenced a wide range of scientific and engineering disciplines [olier2021transformational, thiyagalingam2022scientific, karpatne2018machine, cuomo2022scientific, cai2021physics, guo2023learn, lin2025energy, qi2020using, kharazmi2021identifiability, zheng2022data, mou2023combining]. Among these advancements, physics-informed neural network (PINN) [cai2021physics, cuomo2022scientific, mao2020physics] has emerged as a powerful tool for solving complex partial differential equations (PDEs) by integrating physical laws directly into the learning process. PINN leverages the expressive capabilities of neural networks to approximate solutions to PDEs, offering a mesh-free and flexible alternative to traditional numerical methods. However, the effectiveness of PINN can be significantly impaired in the presence of noisy and sparse observational data (forward problem) [cai2021physics, yang2021b, satyadharma2024assessing], or missing physics, such as unknown coefficients in PDEs (inverse problem) [zou2024correcting, jiang2021hybrid, chung2024hybrid, guo2021construct], which are common challenges in real-world applications.

With perfect data, PINN is successfully used in forward and inverse problems. To train such a PINN, one needs to minimize a multi-objective loss function that includes the PDE residual (residual loss), initial conditions, boundary conditions (boundary loss), and data discrepancies (data loss) [raissi2019physics, mao2020physics, shukla2020physics, lu2021physics, zhang2022analyses, rasht2022physics, xu2023transfer, zhou2023damageIMAC, lu2025evolutionary, zhang2025constrained]. They are widely used in many different problems. For example, Mao et al. [mao2020physics] utilized PINNs to infer density, velocity, and pressure fields for the one-dimensional Euler equations based on observed density gradient data. Similarly, Rasht et al. [rasht2022physics] employed PINNs for full waveform inversions in seismic imaging to determine wave speed from observational data. Despite the success of PINNs, there are still several challenges. First, PINN uses the soft constraints which tend to minimize the sum of the PDE residual, boundary, and data losses with appropriate weights, however, the imbalance among different loss functions during training period may result in unusually expensive training costs [wong2022learning]. This can happen when certain terms dominate or vanish prematurely, resulting in inefficient training and therefore leading to non-optimal results. Second, PINNs are usually sensitive to available noisy data, which in fact is a very common setting in real-world applications. Indeed, noisy or imperfect data can “mislead” the training and therefore generate inaccurate results with inappropriate neural network parameters; the model errors in forward problem or only missing physics in inverse problem put the problem more challenging. Third, in inverse problems where only partial physics are known, i.e., PDEs with unknown parameters, the accuracy of traditional PINNs diminishes. This is because the PDE residual loss relies on both the neural network derivatives and the unknown parameters, which makes accurate inference challenging without balanced losses in training and access to high-quality data.

This paper proposes an integrated iPINNER framework that combines the I terative P hysics I nformed N eural N etwork with E nsemble Kalman Filte r (EnKF) [Evensen_1996, evensen2003ensemble, KF_original] to solve PDEs in both forward and inverse settings with noisy observational data. The iPINNER uses reference-point-based non-dominated sorting approach (hereby, referred to as NSGA-III) [deb2000fast, deb2001controlled, deb2013evolutionary] to solve the multi-objective loss function in the original PINN. Specifically, iPINNER employs NSGA-III to generate ensemble members of PINNs within the optimal Pareto front where these ensemble members are further used as forecast model results in EnKF, together with available observation data to iteratively refines the PINNs by updating its data loss function. The iPINNER framework integrates the advantages of two methods, evolution multi-objective optimizer, i.e., NSGA-III and ensemble Kalman filter (EnKF). The former can provide more balanced and effective training for PINNs with a multi-objective loss function while the optimal solutions from NSGA-III consist of various members on the optimal Pareto front, expressing the model uncertainty. In particular, NSGA-III treats each component of PINN loss as distinct objectives and in the training process, the non-dominated sorting and crowding distance calculation methods is employed [lu2023nsga]. On the other hand, the latter, i.e., EnKF can be used to assimilate model and observational data to find the optimal solutions in the Bayesian sense. While the original Kalman filter only handles linear systems [KF_original], the ensemble Kalman filter (EnKF) and its variant [Anderson2001, Evensen_1996, evensen2003ensemble, mou2023efficient, chen2020predicting, popov2021multifidelity] extends it to a wider range of problems by using a Monte Carlo approach. Essentially, the EnKF begins with a probability distribution (represented by ensemble members in the forecast) and a likelihood function for observed data, then applies Bayes’ theorem to update this distribution (the “analysis” or posterior) once new observations are introduced. However, neural network–based PDE solvers often struggle to generate ensemble members that capture model uncertainty (model errors). The EnKF has therefore, this iPINNER approach iterately refines the PINNs while leveraging the strengths of NSGA-III’s multi-objective optimization capabilities and EnKF’s denoising to improve the accuracy and robustness. The proposed iPINNER framework is general and can be reformulated with different trial spaces once the PDE problem is recast as an optimization problem. In this work, we adopt a neural representation u​(x,t;θ)u(x,t;\theta), where θ\theta is the neural parameters, for the two reasons: (1) it can encode physical laws (e.g., PDE constraints, energy dissipation) and (2) it provides mesh-free automatic differentiation for residual evaluation.

This method can be used in both forward and inverse problem settings for the given PDEs. It is also important to note that, in the inverse problem setting for PINNs, there are two different approaches: (I) The unknown physical parameters are treated as additional independent variables and included as inputs to the neural network, which is trained over a range of parameter values [chen2023reduced, chen2022autodifferentiable]; (II) The physical parameters are treated as trainable variables. While they do not explicitly appear in the network architecture, they affect the training process through their contribution to the loss function via the PDE residual. In this paper, we use the second approach for the inverse problem, i.e., putting unknown physics term as the trainable variables which directly contributes to the PDE residual in PINN’s loss function. In brief summary, our primary contribution includes the following:

1.   1.
We employ the NSGA-III algorithm to treat each component of the PINN loss as an individual objective and use Non-dominated Sorting (NDS) and gradient decent method to optimize these objectives. Numerical tests show that this multi-objective approach helps the PINN avoid local minima and better satisfy physical and data constraints.

2.   2.
We proposed the novel iPINNER framework that integrates Ensemble Kalman Filter (EnKF) with the NSGA-III-optimized PINN ensemble. The EnKF utilizes solutions from the Pareto front (as predictions) and assimilates observational data to update the state variables, which in turn refines the PINN training process. Experiments on both forward and inverse PDE/FPDE problems show that the framework significantly improves prediction accuracy in the presence of model imperfections and noisy data.

3.   3.
We demonstrate that iPINNER significantly improves prediction accuracy over traditional PINNs in scenarios with incomplete physical knowledge and noisy observations. While conventional PINNs often fail under these conditions due to loss imbalance and sensitivity to noise, our framework leverages filtered observational data via EnKF to guide the model toward the ground truth.

4.   4.
iPINNER also achieves strong performance in inverse PDE and FPDE problems, effectively recovering missing physical information by combining ensemble-based data assimilation with multi-objective training.

Compared to methods that rely solely on multi-objective optimization, the iPINNER framework shows substantial improvements when only noisy data are available. The proposed framework introduces a novel iterative scheme that integrates multi-objective optimization with the Kalman filter to identify optimal solutions under noise-dominated conditions—an approach not previously explored in the PINN literature. The rest of the paper is organized as follows. In Section [2](https://arxiv.org/html/2506.00731v2#S2 "2 General framework ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter"), we introduce the proposed integrated model that combines NSGA-III optimizer (Section [2.2](https://arxiv.org/html/2506.00731v2#S2.SS2 "2.2 Non‐Dominated Sorting Genetic Algorithm-III (NSGA-III) ‣ 2 General framework ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter")) and ensemble Kalman filter (Section [2.3](https://arxiv.org/html/2506.00731v2#S2.SS3 "2.3 Ensemble Kalman Filter (EnKF) ‣ 2 General framework ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter")) in the PINN framework (Section [2.1](https://arxiv.org/html/2506.00731v2#S2.SS1 "2.1 Physics Informed Neural Network (PINN) ‣ 2 General framework ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter")). Section [3](https://arxiv.org/html/2506.00731v2#S3 "3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter") presents numerical test results for both forward and inverse problems for the proposed framework. Specifically, we test two different problems: (1). one-dimensional viscous Burgers equation in Section [3.1](https://arxiv.org/html/2506.00731v2#S3.SS1 "3.1 Burgers Equation ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter"), and (2). time-fractional mixed diffusion-wave equations (TFMDWEs) in Section [3.2](https://arxiv.org/html/2506.00731v2#S3.SS2 "3.2 Time-fractional Mixed Diffusion-Wave Equations (TFMDWEs) ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter"). Finally, conclusions and future research directions are discussed in Section [4](https://arxiv.org/html/2506.00731v2#S4 "4 Conclusion and Future Work ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter").

2 General framework
-------------------

In this section, we introduce the physics‐informed neural network (PINN), the Kalman filter, and the Non-Dominated Sorting Genetic Algorithm-III (NSGA-III), and then describe the framework that integrates these components as illustrated in Figure[1](https://arxiv.org/html/2506.00731v2#S2.F1 "Figure 1 ‣ 2.1 Physics Informed Neural Network (PINN) ‣ 2 General framework ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter") and Algorithm[2](https://arxiv.org/html/2506.00731v2#alg2 "In 2.4 A brief summary of the framework ‣ 2 General framework ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter"). The framework employs PINNs to integrate potentially incomplete physical information through partial differential equation (PDE)‐based loss functions, while the Kalman filter assimilates observational data in real time to optimally estimate state variables with PINN models. Subsequently, NSGA-III is employed to optimize the PINN ensemble by treating various loss components (e.g., PDE residual, data mismatch) as separate objectives, efficiently exploring the parameter space to identify a set of non-dominated solutions (the Pareto front). This integrated approach ensures that each component informs and improves the others, leading to a comprehensive, data‐driven methodology that remains faithful to the underlying physics.

### 2.1 Physics Informed Neural Network (PINN)

To illustrate the framework of physics informed neural networks (PINNs), we start with the general nonlinear PDE which takes the form [lu2021physics, karniadakis2021physics, mao2020physics]:

u t+𝒩​[u]=0,x∈Ω,t∈[0,T],\displaystyle u_{t}+\mathcal{N}[u]=0,\ x\in\Omega,\ t\in[0,T],(1)

with a suitable initial condition and Dirichlet boundary conditions, where u​(t,x)u(t,x) denotes the latent (hidden) solution, 𝒩​[⋅]\mathcal{N}[\cdot] is a nonlinear differential operator, and Ω\Omega is a subset of ℝ D\mathbb{R}^{D}. We define the PDE residual as a function f f:

f=u t+𝒩​[u].\displaystyle f=u_{t}+\mathcal{N}[u].(2)

PINN framework finds a neural network (NN), which is parametrized by a set of parameters θ\theta, i.e., u^θ​(z)\hat{u}_{\theta}(z) to approximate the solution to the PDE. To determine the parameter set θ\theta that defines the model, PINN solves a optimization problem minimizing a suitably constructed loss function, which incorporates contributions from the differential equation ℒ ℱ\mathcal{L}_{\mathcal{F}}, the boundary conditions ℒ ℬ\mathcal{L}_{\mathcal{B}}, and any available data ℒ data\mathcal{L}_{\mathrm{data}}, with each component appropriately weighted. Specifically, the optimization problem yields the following:

θ∗=arg⁡min θ⁡(ω i​c​ℒ i​c​(θ)+ω b​c​ℒ b​c​(θ)+ω r​e​s​ℒ r​e​s​(θ)+ω d​a​t​a​ℒ data​(θ)).\displaystyle\theta^{*}\;=\;\arg\min_{\theta}\Bigl(\omega_{ic}\,\mathcal{L}_{ic}(\theta)\;+\;\omega_{bc}\,\mathcal{L}_{bc}(\theta)\;+\;\omega_{res}\,\mathcal{L}_{res}(\theta)\;+\;\omega_{data}\,\mathcal{L}_{\text{data}}(\theta)\Bigr).(3)

where

ℒ i​c=1 N i​c​∑i=1 N i​c(u​(x i,t i)−u i)2,\displaystyle\mathcal{L}_{ic}=\frac{1}{N_{ic}}\sum_{i=1}^{N_{ic}}\left(u(x_{i},t_{i})-u_{i}\right)^{2},(4)

ℒ b​c=1 N b​c​∑i=1 N b​c(u​(x i,t i)−u i)2,\displaystyle\mathcal{L}_{bc}=\frac{1}{N_{bc}}\sum_{i=1}^{N_{bc}}\left(u(x_{i},t_{i})-u_{i}\right)^{2},(5)

ℒ r​e​s=1 N r​e​s​∑j=1 N r​e​s(u t+𝒩​[u])2|(x j,t j).\displaystyle\mathcal{L}_{res}=\frac{1}{N_{res}}\sum_{j=1}^{N_{res}}\left(u_{t}+\mathcal{N}[u]\right)^{2}\bigg|_{(x_{j},t_{j})}.(6)

and

![Image 1: Refer to caption](https://arxiv.org/html/2506.00731v2/MoPINNEnKF_diagram.png)

Figure 1: Schematic of the proposed workflow for iPINNER framework for forward and inverse problem (parameter estimation or missing physics). (a) Firstly, the neural network takes spatiotemporal inputs (x,t)(x,t) and, through automatic differentiation, enforces partial differential equation (PDE) constraints, boundary and initial conditions (ICs/BCs), as well as noisy observational data via loss functions. (b) Secondly. a multi‐objective loss and multiple candidate networks are then refined using a multi‐objective optimizer (NSGA-III), yielding robust solutions consistent with both data and governing physics. (c) The main flow of iPINNER that uses ensemble output from NSGA-PINN as initial observation and combine 

ℒ data=1 N data​∑k=1 N data(u​(x k d,t k d)−u k d)2.\displaystyle\mathcal{L}_{\text{data}}=\frac{1}{N_{\text{data}}}\sum_{k=1}^{N_{\text{data}}}\left(u(x^{d}_{k},t^{d}_{k})-u^{d}_{k}\right)^{2}.(7)

Here {(x i,t i)}\{(x_{i},t_{i})\} are the number of points sampled at the initial/boundary locations and in the entire domain, respectively, and {u k d}\{u^{d}_{k}\} is a set of u u values that are accessible at {(x i d,t i d)}\{(x^{d}_{i},t^{d}_{i})\}; w r​e​s w_{res}, w i​c w_{ic}, w b​c w_{bc}, and w data w_{\text{data}} are the weights used to balance the interplay among the four loss terms. These weights affects both the convergence rates and the obtained minimum for the optimization problem([3](https://arxiv.org/html/2506.00731v2#S2.E3 "In 2.1 Physics Informed Neural Network (PINN) ‣ 2 General framework ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter")) [psaros2022meta, bai2023physics], therefore it is crucial to choose their values. In [lu2023nsga], the authors integrate the NSGA-III algorithm into the optimization framework for PINNs, and we use the same strategy in this paper.

### 2.2 Non‐Dominated Sorting Genetic Algorithm-III (NSGA-III)

Multi-objective optimization problems frequently arise in various scientific and engineering fields, including artificial intelligence [tian2021evolutionary], data mining [hong2021evolutionary], software engineering [li2019quality], scheduling [jozefowiez2008multi], bioinformatics [rockendorf2022design], and economics [park2024techno], many of which involve high-dimensional spaces. In what follows, we will introduce the nondominated sorting genetic algorithm (NSGA-III) for multi-objective optimization problems and how it is used to reduce uncertainty in PINN frameworks.

#### 2.2.1 Multi-objective Optimization

The multi-objective optimization problem is defined as follows: Given m∈𝐍 m\in\mathbf{N}, the _m m-objective function_ is defined as f​(x)=(f 1​(x),…,f m​(x))f(x)=(f_{1}(x),\ldots,f_{m}(x)) where x∈Ω x\in\Omega, and f i:Ω→𝐑 f_{i}\colon\Omega\rightarrow\mathbf{R} for a given search space Ω\Omega. Other than in single-objective optimization, there is usually no solution that minimizes all m m objective functions simultaneously. Suppose that there are two solutions x,y x,y, x x _dominates_ y y referring to x⪯y x\preceq y if and only if f j​(x)≤f j​(y)f_{j}(x)\leq f_{j}(y) for all 1≤j≤m 1\leq j\leq m. If there exists a j 0 j_{0} such that f j 0​(x)<f j 0​(y)f_{j_{0}}(x)<f_{j_{0}}(y), we refer that x x _strictly dominates_ y y, denoted by x≺y x\prec y. A solution is defined as _Pareto-optimal_ if it is not strictly dominated by any other solution. Here, the set of objective values of Pareto-optimal solutions is denoted as _Pareto front_.

To solve such optimization problems, one can consider population-based evolutionary algorithms, which are usually more effective than traditional mathematical programming in finding a set of solutions that balance different objective functions. However, it is becoming increasingly clear that expecting a single population-based optimization method to achieve convergence near the Pareto-optimal front and maintain a uniform distribution across the entire front in high-dimensional problems is impractical [deb2011multi, zitzler2000comparison, goldberg1989messy]. One remedy to tackle such an issue is to incorporate external mechanisms to support diversity maintenance, which could also alleviate the computational burden. Instead of exhaustively exploring the entire search space for Pareto-optimal solutions, the algorithm can initiate multiple predefined, targeted searches. The nondominated sorting genetic algorithm (NSGA-III), proposed by Deb and Jain [deb2013evolutionary], is based on this principle and has been shown to be an effective multi-objective optimization method within the evolutionary optimization framework. NSGA-III modifies NSGA-II [deb2002fast] by incorporating a reference-point–based mechanism for many-objective problems, emphasizing population members that are non-dominated yet lie close to a set of specified reference points. NSGA-III initializes with a random population of size N N. In each iteration, the user generates an offspring population of size N N with mutation and/or crossover operators. With a fixed population size, out of this total of 2​N 2N individuals, NSGA-III selects N N for the next iteration. Because non-dominated solutions are preferred, the following ranking scheme establishes the dominance relation as the principal criterion for individual survival. Individuals that are not strictly dominated by any other in the population are assigned rank 1 1. Subsequent ranks are determined recursively: each unranked individual that is strictly dominated only by those with ranks 1,…,k−1 1,\ldots,k-1 is assigned rank k k. Intuitively, the lower an individual’s rank, the more interesting it is. Denote F i F_{i} be the set of individuals with rank i i, and let i∗i^{*} be the smallest integer such that

∑i=1 i∗|F i|≥N.\displaystyle\sum_{i=1}^{i^{*}}|F_{i}|\geq N.(8)

All individuals with rank at most i∗−1 i^{*}-1 retain to the next generation. In addition, 0<k≤N 0<k\leq N individuals of rank i∗i^{*} must be selected so that the new population remains N N, allowing the next iteration to proceed.

#### 2.2.2 Uncertainty

In [lu2023nsga], the authors use the non-dominated sorting genetic algorithm (NSGA) to improve traditional stochastic gradient optimization methods (e.g., ADAM), enabling them to escape local minima more effectively. Indeed, when complete data or fully known physics information is available, PINNs combined with NSGA can effectively solve PDEs in both forward and inverse problem settings [moya2025conformalized]. However, when data availability is limited or the data are noisy, the framework may inadvertently incorporate uncertainties originating from noisy data in forward problems. Moreover, incomplete or missing physics information can lead to wrong solutions in inverse Problems. On the other hand, NSGA-III provides an alternative method for quantifying uncertainty arising from data-driven loss functions and potentially incomplete physical models. Specifically, NSGA-III generates a set of Pareto-optimal solutions (see Section[2.2.1](https://arxiv.org/html/2506.00731v2#S2.SS2.SSS1 "2.2.1 Multi-objective Optimization ‣ 2.2 Non‐Dominated Sorting Genetic Algorithm-III (NSGA-III) ‣ 2 General framework ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter")), which can be viewed as multiple realizations or candidate parameter configurations within the PINN framework. These solutions can be integrated into an ensemble Kalman filter (in Section[2.3](https://arxiv.org/html/2506.00731v2#S2.SS3 "2.3 Ensemble Kalman Filter (EnKF) ‣ 2 General framework ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter")), together with observational data, to approximate the posterior probability distribution. This approximation can then be iteratively utilized back to refine and correct the data loss in the PINN framework. The NSGA-III algorithm used in PINN is outlined in Algorithm[1](https://arxiv.org/html/2506.00731v2#alg1 "In 2.2.2 Uncertainty ‣ 2.2 Non‐Dominated Sorting Genetic Algorithm-III (NSGA-III) ‣ 2 General framework ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter").

1 Initialize a population P 0 P_{0} consisting of N N individuals chosen independently and uniformly:

P 0={θ 0(i)}i=1 N P_{0}=\{\theta_{0}^{(i)}\}_{i=1}^{N}

where θ 0(i)\theta_{0}^{(i)} denotes the parameters of the i t​h i^{th} neural network individual.

2 for _t=0,1,2,…t=0,1,2,\ldots_ do

3 Generate offspring population

Q t Q_{t}
of size

N N
using selection, crossover, and mutation operations.

4

5 Set combined population

R t←P t∪Q t R_{t}\leftarrow P_{t}\cup Q_{t}
.

6

7 Apply fast non-dominated sorting[deb2002fast] to partition

R t R_{t}
into non-dominated fronts

ℒ 1,ℒ 2,…\mathcal{L}_{1},\mathcal{L}_{2},\dots
based on the multi-objective loss function defined in ([3](https://arxiv.org/html/2506.00731v2#S2.E3 "In 2.1 Physics Informed Neural Network (PINN) ‣ 2 General framework ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter")).

8

9 Identify the front index

i∗≥1 i^{*}\geq 1
such that:

∑i=1 i∗−1|ℒ i|<N and∑i=1 i∗|ℒ i|≥N\sum_{i=1}^{i^{*}-1}|\mathcal{L}_{i}|<N\quad\text{and}\quad\sum_{i=1}^{i^{*}}|\mathcal{L}_{i}|\geq N

10

11 Set

Z t←⋃i=1 i∗−1 ℒ i Z_{t}\leftarrow\displaystyle\bigcup_{i=1}^{i^{*}-1}\mathcal{L}_{i}
.

12

13 Select subset

ℒ~i∗⊆ℒ i∗\tilde{\mathcal{L}}_{i^{*}}\subseteq\mathcal{L}_{i^{*}}
satisfying:

|Z t∪ℒ~i∗|=N\displaystyle\bigl|Z_{t}\cup\tilde{\mathcal{L}}_{i^{*}}\bigr|=N
Selection based on reference points R R when maximizing the function ℒ\mathcal{L}.

14

15 Update the population:

P t+1←Z t∪ℒ~i∗P_{t+1}\leftarrow Z_{t}\cup\tilde{\mathcal{L}}_{i^{*}}

16

17 end for

Algorithm 1 NSGA-III in PINN’s multi-objective optimization 

### 2.3 Ensemble Kalman Filter (EnKF)

Since the ensemble Kalman filter (EnKF) was first developed in geophysics [evensen2003ensemble, Evensen_1996], it has been widely used in atmospheric science, oceanography, and climate modeling. The EnKF involves two steps: (1). forecast (prediction) and (2). analysis (filtering). The forecast step is usually temporal propagation of state-space models:

x t=F​(x t−1)\displaystyle x_{t}=F(x_{t-1})(9)

In the update step, we assume that each observation is a linear combination of the state, perturbed by Gaussian noise. Formally, for a given state x t x_{t}, the observation Y t Y_{t} follows a normal distribution:

Y t∣x t∼𝒩​(H​x t,σ o),\displaystyle Y_{t}\mid x_{t}\sim\mathcal{N}\bigl(Hx_{t},\ \sigma^{o}\bigr),(10)

where H H is the obervational operator and σ o\sigma^{o} denotes the amplitude of the observation noise. If the prediction distribution is normal,

π t∣t−1=𝒩​(m t∣t−1,P t∣t−1),\displaystyle\pi_{t\mid t-1}\;=\;\mathcal{N}\bigl(m_{t\mid t-1},\ P_{t\mid t-1}\bigr),(11)

then the filter distribution is also normal,

π t=𝒩​(m t,P t),\displaystyle\pi_{t}\;=\;\mathcal{N}\bigl(m_{t},\ P_{t}\bigr),(12)

with

m t=m t∣t−1+K t​(y t−H​m t∣t−1),P t=(I−K t​H)​P t∣t−1,\displaystyle m_{t}\;=\;m_{t\mid t-1}\;+\;K_{t}\bigl(y_{t}-H\,m_{t\mid t-1}\bigr),\quad P_{t}\;=\;\bigl(I\;-\;K_{t}\,H\bigr)\,P_{t\mid t-1},(13)

where

K t=K​(P t∣t−1,R)=P t∣t−1​H⊤​(H​P t∣t−1​H⊤+R)−1.\displaystyle K_{t}\;=\;K\!\bigl(P_{t\mid t-1},\,R\bigr)\;=\;P_{t\mid t-1}\,H^{\top}\bigl(H\,P_{t\mid t-1}\,H^{\top}+R\bigr)^{-1}.(14)

is the Kalman gain. In the ensemble Kalman filter (EnKF), both π t∣t−1\pi_{t\mid t-1} and π t\pi_{t} are approximated by equally weighted ensembles {x t(i)}\{x_{t}^{(i)}\} and {x~t(i)}\{\widetilde{x}_{t}^{(i)}\}. During the update step, one first employs the prediction sample to estimate m t∣t−1 m_{t\mid t-1} and P t∣t−1 P_{t\mid t-1}. The filter sample is then constructed by transforming the prediction sample so that its mean and covariance align with the update equations. There are several possible approaches to carry out this transformation [evensen2003ensemble, evensen2022]. In this paper, we use the plain ensemble Kalman filter (EnKF) and the forecast step, the prior is generated using the PINNs from NSGA-III algorithm.

### 2.4 A brief summary of the framework

Algorithm[2](https://arxiv.org/html/2506.00731v2#alg2 "In 2.4 A brief summary of the framework ‣ 2 General framework ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter") summarize the framework that infers model on the fly using generic-PINN-based ensemble Kalman filter. In brief, the algorithm integrates physics-informed neural networks (PINNs) with NSGA-III and the ensemble Kalman filter (EnKF) to iteratively infer models from observational data. Initially, the PDE solution is approximated by neural networks, whose parameters are optimized using gradient descent coupled with NSGA-III to produce a few PINNs with different neural network parameters which lies in the optimal Pareto front. These networks are then evaluated at observational points, and the ensemble Kalman filter updates the solution based on observed data, resulting in a posterior estimate. This posterior is then used to refine the PINN loss function, iteratively improving the model until convergence is reached based on a prescribed threshold.

1 1:_Represent the PDE solution by a neural network._

2 2:_Formulate the weighted loss function according to the PDE system:_

ℒ=(ω ℱ​ℒ ℱ​(θ)+ω ℬ​ℒ ℬ​(θ)+ω d​ℒ data​(θ)).\displaystyle\mathcal{L}\;=\;\Bigl(\omega_{\mathcal{F}}\,\mathcal{L}_{\mathcal{F}}(\theta)\;+\;\omega_{\mathcal{B}}\,\mathcal{L}_{\mathcal{B}}(\theta)\;+\;\omega_{d}\,\mathcal{L}_{\text{data}}(\theta)\Bigr).

3 3:_Use S S steps of a gradient descent algorithm together with NSGA-III to update the parameters θ\theta, obtaining a cluster of PINNs:_

{u θ l(1):l=1,…,N s},\{\,u_{\theta_{l}}^{(1)}:\;l=1,\dots,N_{s}\},

_where N s N\_{s} is the number of offsprings in NSGA-III._

4 4:_Evaluate the PINNs at observational points (x k,t k)(x\_{k},t\_{k}), k=1,⋯,N obs k=1,\cdots,N\_{\mathrm{obs}}:_

u θ l(1)​(x 1,t 1),u θ l(1)​(x 2,t 2),…,u θ l(1)​(x N obs,t N obs).u_{\theta_{l}}^{(1)}(x_{1},t_{1}),\;u_{\theta_{l}}^{(1)}(x_{2},t_{2}),\;\dots,\;u_{\theta_{l}}^{(1)}(x_{N_{\mathrm{obs}}},t_{N_{\mathrm{obs}}}).

_Here, u θ l(1)​(x 1,t 1)u\_{\theta\_{l}}^{(1)}(x\_{1},t\_{1}) is interpreted in an ensemble sense._

5 5:_Use the ensemble Kalman filter (EnKF) and the observational data set to obtain the posterior estimate of the data set:_

𝒟(1)​(u)={u~θ(1)​(x 1,t 1),u~θ(1)​(x 2,t 2),…,u~θ(1)​(x N obs,t N obs)}.\mathcal{D}^{(1)}(u)=\bigl\{\widetilde{u}_{\theta}^{(1)}(x_{1},t_{1}),\;\widetilde{u}_{\theta}^{(1)}(x_{2},t_{2}),\;\dots,\;\widetilde{u}_{\theta}^{(1)}(x_{N_{\mathrm{obs}}},t_{N_{\mathrm{obs}}})\bigr\}.

6 6:_Use the data set 𝒟(1)​(u)\mathcal{D}^{(1)}(u) to update the loss function in PINN:_

ℒ o=1 N obs​∑k=1 N obs(u~θ(1)​(x k,t k)−u k)2.\displaystyle\mathcal{L}_{\mathrm{o}}\;=\;\frac{1}{N_{\mathrm{obs}}}\sum_{k=1}^{N_{\mathrm{obs}}}\bigl(\widetilde{u}_{\theta}^{(1)}(x_{k},t_{k})\;-\;u_{k}\bigr)^{2}.(15)

7 7:_Repeat from Step 2 until_

1 N obs​∑k=1 N obs|u θ(m+1)​(x k,t k)−u θ(m)​(x k,t k)|2<ϵ iter,\frac{1}{N_{\mathrm{obs}}}\sum_{k=1}^{N_{\mathrm{obs}}}\bigl\lvert u_{\theta}^{(m+1)}(x_{k},t_{k})\;-\;u_{\theta}^{(m)}(x_{k},t_{k})\bigr\rvert^{2}\;<\;\epsilon_{\mathrm{iter}},

_where ϵ iter\epsilon\_{\mathrm{iter}} is a prescribed threshold._

Algorithm 2 Algorithm for Inferring model on the fly using MoPINNEnKF 

3 Numerical Results
-------------------

In this section, we test the proposed framework, hereafter referred to as iPINNER, using two different benchmark problems: (1) the one-dimensional viscous Burgers equation and (2) the one-dimensional time-fractional mixed diffusion-wave equations (TFMDWEs). For each of these problems, we consider two scenarios: (i) the forward problem and (ii) the inverse problem. Specifically, in the inverse problem setting, we assume the diffusion coefficient is unknown for the viscous Burgers equation, while the fractional order is unknown for the TFMDWEs. Also, the observational data, {u o​b​s​(x,t)}\{u^{obs}(x,t)\} are generated by introducing observational noise at sparse grid point, with a noise level of η\eta. Specifically, this observational noise is drawn from a Gaussian distribution with zero mean, and its standard deviation is set equal to η\eta of the standard deviation of the true solution, s​t​d​(u t​r​u​t​h​(x,t))std(u^{truth}(x,t)), at grid x x. The mathematical formulation yields the following:

u o​b​s​(x,t)=u​(x,t)+η⋅𝒩​(0,s​t​d​(u t​r​u​t​h)).\displaystyle u^{obs}(x,t)=u(x,t)+\eta\,\cdot\mathcal{N}(0,std(u^{truth})).(16)

To investigate the impact of noise levels on solutions of both forward and inverse problems, we consider three different values of the parameter η\eta, corresponding to scenarios of small, medium, and large observational uncertainties, respectively:

1.   1.
η=20%\eta=20\%, representing observations with small uncertainty;

2.   2.
η=50%\eta=50\%, representing observations with medium uncertainty;

3.   3.
η=80%\eta=80\%, representing observations with large uncertainty.

In addition, we compare our proposed iPINNER method with two other approaches: (a) a PINN employing the traditional Adam optimizer, denoted as ADAM-PINN, and (b) a PINN employing only the NSGA-III optimizer, denoted as NSGA-III-PINN.

### 3.1 Burgers Equation

The one-dimensional viscous Burgers equation is a nonlinear partial differential equation frequently used as a benchmark [lu2021deepxde, lu2021physics, raissi2019physics]. The equation with Dirichlet boundary conditions is defined on the spatial domain Ω=[−1,1]\Omega=[-1,1] and temporal domain [0,T][0,T] given by the following:

∂u∂t−ν​∂2 u∂x 2+u​∂u∂x=0,x∈Ω,t∈[0,T],\displaystyle\frac{\partial u}{\partial t}-\nu\frac{\partial^{2}u}{\partial x^{2}}+u\frac{\partial u}{\partial x}=0,\quad x\in\Omega,\;t\in[0,T],(17)
u​(x,t)=0,∀x∈∂Ω,\displaystyle u(x,t)=0,\quad\forall x\in\partial\Omega,(18)
u​(x,0)=−sin⁡(π​x).\displaystyle u(x,0)=-\sin(\pi x).(19)

Here, u​(x,t)u(x,t) represents the solution over space and time, and ν\nu is the viscosity chosen to be 0.01/π 0.01/\pi.

#### 3.1.1 PINN’s Settings

##### Observational data

To generate observational data used in PINN framework, we choose 100 100 data points evenly for the initial and boundary conditions and 10 4 10^{4} interior points. Figure[2](https://arxiv.org/html/2506.00731v2#S3.F2 "Figure 2 ‣ 3.1.2 Forward Problem ‣ 3.1 Burgers Equation ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter") shows the different noise level observation at different time instance for the one-dimensional viscous Burgers equation.

##### Neural network architecture

In order to make a fair comparison, with different optimizers, the PINN employs a consistent deep neural network architecture comprising 8 layers with 20 neurons in each layer. During the training phase, models corresponding to each optimizer were collected until the training loss converged below a prescribed threshold ϵ\epsilon. Specifically, the ADAM-PINN requires 5000 5000 epochs to achieve convergence, whereas the NSGA-III-PINN arrives at convergence within 4 4 generations. The proposed iPINNER requires of 3 3 generations with 1000 1000 epochs to ensure the convergence of training loss.

##### Training and Testing Data

The training data for the Burgers equation consist of three main components: Initial Condition (IC) points, Boundary Condition (BC) points, and Collocation Points. The IC points are sampled in space at the initial time t=0 t=0, while the BC points are sampled in time along the spatial boundaries (x=−1 x=-1 and x=1 x=1). The collocation points are randomly selected from the interior of the spatiotemporal domain, with 100 points used in this study. The testing data are sampled over the domain Ω×[0,T]\Omega\times[0,T] with Ω=[0,1]\Omega=[0,1]. The spatial mesh size is of Δ​x=1/100\Delta x=1/100 and a temporal step size is Δ​t=0.01\Delta t=0.01.

##### Error criteria

We evaluate models using the mean squared error (MSE) between the benchmark solutions and the predictions of different models.

Table 2: Sensitivity analysis for the ensemble size (N N). The table shows the trade-off between the final Mean Squared Error (MSE) and the total wall-clock time. The chosen value, N=8 N=8, offers the best balance.

Ensemble Size (N N)Final MSE Wall-Clock Time (s)
4 0.0019 298
6 0.0016 336
8 0.0014 385
10 0.0014 415

#### 3.1.2 Forward Problem

In the forward problem, the model is assumed to be not perfect which yields the model error. In particular, we assume that the viscosity term v v is different from its exact value which represents the model errors, i.e., we choose ν=0.02/π\nu=0.02/\pi which is different from its true value ν=0.01/π\nu=0.01/\pi. In this setting, we evaluate the performance of three models: ADAM-PINN, NSGA-PINN, and MoPONNEnKF.

Table 3: Testing error of ADAM-PINN with different Gaussian noises in the presence of imperfect model.

Table[3](https://arxiv.org/html/2506.00731v2#S3.T3 "Table 3 ‣ 3.1.2 Forward Problem ‣ 3.1 Burgers Equation ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter") shows the testing errors of ADAM-PINNs with different Gaussian noise in available observational data, with imperfect model. Because the imperfect model introduces the incorrect viscosity in Burgers equation, the PINN yields model errors. By combining noisy observation data with the model, iPINNER finds the optimal between the two. It is also noted that when the noise level is high, i.e., 80%, both the model error and observational error dominate and hence the iPINNER’s accuracy diminishes.

Table 4: Mean square errors (MSEs) of three different models, ADAM-PINN, NSGA-III-PINN, iPINNER, with different noise levels in the one-dimensional Burgers Equation forward test problem.

Table [4](https://arxiv.org/html/2506.00731v2#S3.T4 "Table 4 ‣ 3.1.2 Forward Problem ‣ 3.1 Burgers Equation ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter") shows the mean square error (MSE) in the forward problem setting for (1). ADAM-PINN, (2). NSGA-III-PINN, and (3). iPINNER with different noise level of obervational data. It is shown that, when the observational noise level is 20%20\% and 50%50\%, iPINNER are at least one order more accurate than ADAM-PINN and NSGA-III-PINN. Even when the observational data become substantially noisier (80%80\% noise level), it remains at least twice as accurate as the other two methods. This is not surprising because both ADAM-PINN and NSGA-III-PINN use noise-contaminated observations directly in their loss functions during training, without an explicit noise-filtering mechanism. In contrast, the iPINNER framework incorporates an additional "purification" step through the Kalman filter, thereby effectively mitigating the impact of observational noise. Nonetheless, the NSGA-III-PINN exhibits slightly better robustness than ADAM-PINN, as the NSGA-III algorithm more effectively balances multiple objectives within the loss function, thereby partially mitigating the adverse effects of noisy observations.

Table 5: Mean square errors (MSEs) of three different models, ADAM-PINN, NSGA-III-PINN, iPINNER, with 80%80\% noise levels in the one-dimensional Burgers Equation forward test problem.

Table[5](https://arxiv.org/html/2506.00731v2#S3.T5 "Table 5 ‣ 3.1.2 Forward Problem ‣ 3.1 Burgers Equation ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter") shows that the NSGA-III algorithm effectively balances the loss terms in the PINN to comparable orders of magnitude within a reasonable training time. As a result, during training, the proposed iPINNER method achieves relatively low residuals for the PDE residual loss, boundary loss, and data loss—performance that neither ADAM-PINN nor NSGA-III-PINN alone can attain.

![Image 2: Refer to caption](https://arxiv.org/html/2506.00731v2/burgers_noise_ut_t0.09.png)

(a) 

![Image 3: Refer to caption](https://arxiv.org/html/2506.00731v2/burgers_noise_ut_t0.39.png)

(b) 

![Image 4: Refer to caption](https://arxiv.org/html/2506.00731v2/burgers_noise_ut_t0.99.png)

(c)

Figure 2:  Observation data with different noise levels at time instances (a)t=0.1 t=0.1, (b)t=0.4 t=0.4, and (c)t=1 t=1 for the one-dimensional Burgers equation. 

Figure[3](https://arxiv.org/html/2506.00731v2#S3.F3 "Figure 3 ‣ 3.1.2 Forward Problem ‣ 3.1 Burgers Equation ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter") shows the spatiotemporal solutions and corresponding errors obtained using (1) ADAM-PINN, (2) NSGA-III-PINN, and (3) iPINNER, under different observational noise levels. Consistent with the findings summarized in Table[4](https://arxiv.org/html/2506.00731v2#S3.T4 "Table 4 ‣ 3.1.2 Forward Problem ‣ 3.1 Burgers Equation ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter"), iPINNER is the most accurate model among the three approaches, with its accuracy becoming more remarkable as the noise level increases.

![Image 5: Refer to caption](https://arxiv.org/html/2506.00731v2/nsga_pinn_adam_noise_20.png)

![Image 6: Refer to caption](https://arxiv.org/html/2506.00731v2/nsga_pinn_adam_noise_50.png)

![Image 7: Refer to caption](https://arxiv.org/html/2506.00731v2/nsga_pinn_adam_noise_80.png)

Figure 3: Forward problem solutions (top) and errors (bottom) for comparison of ADAM-PINN, NSGA-III-PINN and iPINNER solution for one dimensional Burgers equation; observation data noise level are chosen to be 20%20\%, 50%50\%, and 80%80\% respectively.

#### 3.1.3 Inverse Problem

In the inverse problem, we assume that the diffusion term is unknown, i.e., the diffusion coefficient ν\nu in equation ([17](https://arxiv.org/html/2506.00731v2#S3.E17 "In 3.1 Burgers Equation ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter")), is not known, and its initial estimate are chosen randomly in a certain range from 0 to 1 1. The diffusion coefficient is considered to be a trained parameter in the PINN framework and we use data with varying noise levels—similar to those in the forward problem—to train each PINN model. The other neural network settings are the same as in Section [3.2.1](https://arxiv.org/html/2506.00731v2#S3.SS2.SSS1 "3.2.1 PINN’s Settings ‣ 3.2 Time-fractional Mixed Diffusion-Wave Equations (TFMDWEs) ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter").

Table [6](https://arxiv.org/html/2506.00731v2#S3.T6 "Table 6 ‣ 3.1.3 Inverse Problem ‣ 3.1 Burgers Equation ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter") shows the estimated diffusion parameters with L 1 L^{1} errors for different models. It is shown that for all different observational uncertainties, the proposed iPINNER gives the best estimate. Specifically when the observational uncertainty is small and medium, i.e., η=20%\eta=20\% and 50%50\% respectively, the proposed iPINNER gives the much better estimate when comparing to PINNs with ADAM optimizer and with NSGA-III optimizer; and between the two, the later is slightly better. Yet, when the observational uncertainty is large, i.e., η=80%\eta=80\%, the proposed iPINNER gives similar inaccurate results as the other two. This is because when the data is noisy enough, the diffusion coefficient which is the trainable parameter together with PINN’s weights converges to the wrong that still can represent the limited noisy data. Figure [4](https://arxiv.org/html/2506.00731v2#S3.F4 "Figure 4 ‣ 3.1.3 Inverse Problem ‣ 3.1 Burgers Equation ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter") shows errors (left) and solutions (right) for iPINNER solution for one dimensional Burgers equation with different observation data noise level are chosen to be 20%20\%, 50%50\%, and 80%80\%. It is consistent with the results shown in Table [6](https://arxiv.org/html/2506.00731v2#S3.T6 "Table 6 ‣ 3.1.3 Inverse Problem ‣ 3.1 Burgers Equation ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter").

Table 6: Comparison of ADAM-PINN, NSGA-III-PINN and the proposed iPINNER for estimating the viscosity term ν\nu in the Burgers equation. For reference, the true value of the viscosity term is ν=0.01/π≈0.00318\nu=0.01/\pi\approx 0.00318.

![Image 8: Refer to caption](https://arxiv.org/html/2506.00731v2/nsga_pinn_adam_inverse_noise_20.png)

![Image 9: Refer to caption](https://arxiv.org/html/2506.00731v2/nsga_pinn_adam_inverse_noise_50.png)

![Image 10: Refer to caption](https://arxiv.org/html/2506.00731v2/nsga_pinn_adam_inverse_noise_80.png)

Figure 4: Inverse problem errors (left) and solutions (right) for iPINNER solution for one dimensional Burgers equation; observation data noise level are chosen to be 20%20\%, 50%50\%, and 80%80\% respectively.

### 3.2 Time-fractional Mixed Diffusion-Wave Equations (TFMDWEs)

In this section, we consider the time-fractional mixed diffusion-wave equations (TFMDWEs), which generalize classical diffusion and wave equations by incorporating fractional-order time derivatives [liu2019alternating, du2021temporal, lu2025fpinn]. The TFMDWEs with Dirichlet boundary conditions are defined as follows:

D t α​u​(x,t)=∂2 u∂x 2+f​(x,t),t∈[0,1],x∈Ω≡[0,π],\displaystyle D_{t}^{\alpha}u(x,t)=\frac{\partial^{2}u}{\partial x^{2}}+f(x,t),\quad t\in[0,1],\;x\in\Omega\equiv[0,\pi],(20)
u​(x,t)=0,∀x∈∂Ω,\displaystyle u(x,t)=0,\quad\forall x\in\partial\Omega,(21)
u​(x,0)=0,x∈Ω,\displaystyle u(x,0)=0,\quad x\in\Omega,(22)

where the fractional order α\alpha yields:

α∈[0,1],∀t∈[0,1],\displaystyle\alpha\in[0,1],\quad\forall t\in[0,1],(23)

and the forcing term f​(x,t)f(x,t) is explicitly defined as:

f​(x,t)=Γ​(4)Γ​(4−α)​t 3−α​sin⁡(x)+t 3​sin⁡(x),\displaystyle f(x,t)=\frac{\Gamma(4)}{\Gamma(4-\alpha)}\,t^{3-\alpha}\sin(x)+t^{3}\sin(x),(24)

with Γ​(⋅)\Gamma(\cdot) denoting the Gamma function. This equation introduces fractional-order temporal dynamics, combining features of both diffusion and wave phenomena. The fractional order α\alpha controls the transition between diffusive and wave-like behaviors, making TFMDWEs particularly useful in modeling complex systems exhibiting anomalous transport and non-local temporal interactions.

#### 3.2.1 PINN’s Settings

##### Observational data

To generate observational data used in PINN framework, we choose 100 100 data points evenly for the initial and boundary conditions and 10 4 10^{4} interior points. Figure[5](https://arxiv.org/html/2506.00731v2#S3.F5 "Figure 5 ‣ Neural network architecture ‣ 3.2.1 PINN’s Settings ‣ 3.2 Time-fractional Mixed Diffusion-Wave Equations (TFMDWEs) ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter") shows the different noise level observation at different time instance for the time-fractional mixed diffusion-wave equations (TFMDWEs).

##### Neural network architecture

In order to make a fair comparison, with different optimizers, the PINN employs a consistent deep neural network architecture comprising 2 hidden layers with 50 neurons in each layer. During the training phase, models corresponding to each optimizer were collected until the training loss converged below a prescribed threshold ϵ\epsilon. Specifically, training ADAM-PINN requires 5000 5000 epochs to achieve convergence, whereas the NSGA-III-PINN reaches convergence within 4 4 generations. The proposed iPINNER requires of 3 3 generations with 2000 2000 epochs to ensure the convergence of training loss.

![Image 11: Refer to caption](https://arxiv.org/html/2506.00731v2/noise_ut_t0.30.png)

(a) 

![Image 12: Refer to caption](https://arxiv.org/html/2506.00731v2/noise_ut_t0.60.png)

(b) 

![Image 13: Refer to caption](https://arxiv.org/html/2506.00731v2/noise_ut_t1.00.png)

(c)

Figure 5:  Observation data with different noise levels at time instances (a)t=0.3 t=0.3, (b)t=0.6 t=0.6, and (c)t=1 t=1 for the time-fractional mixed diffusion-wave equations (TFMDWEs). 

#### 3.2.2 Forward Problem

Similar to the Burgers equation test case, we assume that only an imperfect model is available. In particular, for TFMDWEs, the source term f f is assumed to be inaccurate by adding 50% Gaussian noise and thereby mimicking the model errors.

Table 7: Testing error of PINN (MAE) under different levels of Gaussian noise with incorrect viscosity.

Table[7](https://arxiv.org/html/2506.00731v2#S3.T7 "Table 7 ‣ 3.2.2 Forward Problem ‣ 3.2 Time-fractional Mixed Diffusion-Wave Equations (TFMDWEs) ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter") shows the testing error of PINN model with different Gaussian noises in observational data. With the imperfect model, the PINN leads to inaccurate prediction without data. Incorporating observational data with low-level noise (20%) can improve the correction of model predictions. However, when the observational data are highly contaminated by noise, the predictive accuracy of the model deteriorates.

Table 8: Mean absolute errors (MAEs) of three different models, ADAM-PINN, NSGA-III-PINN, iPINNER, with different noise levels in the time-fractional mixed diffusion-wave equations (TFMDWEs).

Table [8](https://arxiv.org/html/2506.00731v2#S3.T8 "Table 8 ‣ 3.2.2 Forward Problem ‣ 3.2 Time-fractional Mixed Diffusion-Wave Equations (TFMDWEs) ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter") shows the mean absolute error (MAE) in the forward problem setting for (1). ADAM-PINN, (2). NSGA-III-PINN, and (3). iPINNER with different noise level of observational data. It is shown that in all cases, iPINNER is at least more accurate than ADAM-PINN and NSGA-III-PINN. Nonetheless, the NSGA-III-PINN exhibits slightly better robustness than ADAM-PINN, as the NSGA-III algorithm more effectively balances multiple objectives within the loss function, thereby partially mitigating the adverse effects of noisy observations. This is also reflected in the figures [6](https://arxiv.org/html/2506.00731v2#S3.F6 "Figure 6 ‣ 3.2.2 Forward Problem ‣ 3.2 Time-fractional Mixed Diffusion-Wave Equations (TFMDWEs) ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter")–[8](https://arxiv.org/html/2506.00731v2#S3.F8 "Figure 8 ‣ 3.2.2 Forward Problem ‣ 3.2 Time-fractional Mixed Diffusion-Wave Equations (TFMDWEs) ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter").

Table 9: Mean square errors (MSEs) of three different models, ADAM-PINN, NSGA-III-PINN, iPINNER, with 80%80\% noise levels in the one-dimensional Burgers Equation forward test problem.

Table[9](https://arxiv.org/html/2506.00731v2#S3.T9 "Table 9 ‣ 3.2.2 Forward Problem ‣ 3.2 Time-fractional Mixed Diffusion-Wave Equations (TFMDWEs) ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter") compares the mean square error of three different models, ADAM-PINN, NSGA-III-PINN, and iPINNER, with 80% Gaussian noise in the forward problem. The results show that iPINNER consistently obtains the lowest errors across all loss components, i.e., PDE residual, boundary, and observation losses. Figure [9](https://arxiv.org/html/2506.00731v2#S3.F9 "Figure 9 ‣ 3.2.2 Forward Problem ‣ 3.2 Time-fractional Mixed Diffusion-Wave Equations (TFMDWEs) ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter") shows at different time instances (top: t=0.5 t=0.5, bottom: t=1 t=1), the true solution (black) and forward problem solutions obtained using ADAM-PINN (red), NSGA-III-PINN (blue), and iPINNER (green) for the time-fractional mixed diffusion-wave equations (TFMDWEs) with different noise levels. The results show that iPINNER (green) outperforms both ADAM-PINN (red) and NSGA-III-PINN (blue) in terms of accuracy. Moreover, its advantage becomes even more pronounced at higher noise levels.

![Image 14: Refer to caption](https://arxiv.org/html/2506.00731v2/TFMDWEs_true_forward_noise20.png)

(a) 

![Image 15: Refer to caption](https://arxiv.org/html/2506.00731v2/TFMDWEs_MoPINNenKF_forward_noise20.png)

(b) 

![Image 16: Refer to caption](https://arxiv.org/html/2506.00731v2/err_TFMDWEs_data_adam_forward_noise20.png)

(c) 

![Image 17: Refer to caption](https://arxiv.org/html/2506.00731v2/err_TFMDWEs_MoPINNenKF_forward_noise20.png)

(d)

Figure 6:  Comparison of (a) the true solution and (b) the iPINNER solution as well as the L 1 L_{1} errors of (c) the ADAM-PINN and (d) the iPINNER for the time-fractional mixed diffusion-wave equations (TFMDWEs) in forward problem. Observation noise level is 20%20\%. 

![Image 18: Refer to caption](https://arxiv.org/html/2506.00731v2/TFMDWEs_true_forward_noise20.png)

(a) 

![Image 19: Refer to caption](https://arxiv.org/html/2506.00731v2/TFMDWEs_MoPINNenKF_forward_noise50.png)

(b) 

![Image 20: Refer to caption](https://arxiv.org/html/2506.00731v2/err_TFMDWEs_data_adam_forward_noise50.png)

(c) 

![Image 21: Refer to caption](https://arxiv.org/html/2506.00731v2/err_TFMDWEs_MoPINNenKF_forward_noise50.png)

(d)

Figure 7:  Comparison of (a) the true solution and (b) the iPINNER solution as well as the L 1 L_{1} errors of (c) the ADAM-PINN and (d) the iPINNER for the time-fractional mixed diffusion-wave equations (TFMDWEs) in forward problem. Observation noise level is 50%50\%. 

![Image 22: Refer to caption](https://arxiv.org/html/2506.00731v2/TFMDWEs_true_forward_noise20.png)

(a) 

![Image 23: Refer to caption](https://arxiv.org/html/2506.00731v2/TFMDWEs_MoPINNenKF_forward_noise80.png)

(b) 

![Image 24: Refer to caption](https://arxiv.org/html/2506.00731v2/err_TFMDWEs_data_adam_forward_noise80.png)

(c) 

![Image 25: Refer to caption](https://arxiv.org/html/2506.00731v2/err_TFMDWEs_MoPINNenKF_forward_noise80.png)

(d)

Figure 8:  Comparison of (a) the true solution and (b) the iPINNER solution as well as the L 1 L_{1} errors of (c) the ADAM-PINN and (d) the iPINNER for the time-fractional mixed diffusion-wave equations (TFMDWEs) in forward problem. Observation noise level is 80%80\%. 

![Image 26: Refer to caption](https://arxiv.org/html/2506.00731v2/forward_noise_level_20_ut_t0.51.png)

![Image 27: Refer to caption](https://arxiv.org/html/2506.00731v2/forward_noise_level_20_ut_t0.71.png)

(a) 20%20\% noise level 

![Image 28: Refer to caption](https://arxiv.org/html/2506.00731v2/forward_noise_level_50_ut_t0.51.png)

![Image 29: Refer to caption](https://arxiv.org/html/2506.00731v2/forward_noise_level_50_ut_t0.71.png)

(b) 50%50\% noise level 

![Image 30: Refer to caption](https://arxiv.org/html/2506.00731v2/forward_noise_level_80_ut_t0.51.png)

![Image 31: Refer to caption](https://arxiv.org/html/2506.00731v2/forward_noise_level_80_ut_t0.71.png)

(c) 80%80\% noise level 

Figure 9: Comparison at different time instances (top: t=0.5 t=0.5, bottom: t=0.7 t=0.7) between the true solution (black) and forward problem solutions obtained using ADAM-PINN (red), NSGA-III-PINN (blue), and iPINNER (green) for the time-fractional mixed diffusion-wave equations (TFMDWEs) under varying noise levels: Panel (a) — 20%20\% noise, Panel (b) — 50%50\% noise, and Panel (c) — 80%80\% noise. 

In Figure [10](https://arxiv.org/html/2506.00731v2#S3.F10 "Figure 10 ‣ 3.2.2 Forward Problem ‣ 3.2 Time-fractional Mixed Diffusion-Wave Equations (TFMDWEs) ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter"), we show the finite element method (FEM) results with different mesh sizes, which demonstrate good agreement for the forward solution compared with our iPINNER. However, FEM is primarily restricted to solving forward problems. While FEM itself does not require data, with available observational data, it requires additional data assimilation techniques. On the other hand iPINNER can simultaneously addresses both forward and inverse problem settings with available noisy data.

![Image 32: Refer to caption](https://arxiv.org/html/2506.00731v2/TFMDWEs_FEM_forward_noise_Nh25.png)

(a) 

![Image 33: Refer to caption](https://arxiv.org/html/2506.00731v2/TFMDWEs_FEM_forward_noise_Nh50.png)

(b) 

![Image 34: Refer to caption](https://arxiv.org/html/2506.00731v2/TFMDWEs_FEM_forward_noise_Nh100.png)

(c) 

![Image 35: Refer to caption](https://arxiv.org/html/2506.00731v2/err_TFMDWEs_FEM_forward_Nh25.png)

(d) 

![Image 36: Refer to caption](https://arxiv.org/html/2506.00731v2/err_TFMDWEs_FEM_forward_Nh50.png)

(e) 

![Image 37: Refer to caption](https://arxiv.org/html/2506.00731v2/err_TFMDWEs_FEM_forward_Nh100.png)

(f)

Figure 10:  Comparison of finite element method (FEM) solution with different mesh size for the time-fractional mixed diffusion-wave equations (TFMDWEs) in forward problem. 

#### 3.2.3 Inverse Problem

Table [10](https://arxiv.org/html/2506.00731v2#S3.T10 "Table 10 ‣ 3.2.3 Inverse Problem ‣ 3.2 Time-fractional Mixed Diffusion-Wave Equations (TFMDWEs) ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter") shows the estimated α\alpha values of the TFMDWEs and their L 1 L^{1} error in the inverse problem setting for (1). ADAM-PINN, (2). NSGA-III-PINN, and (3). iPINNER with different noise level of observational data. It is shown that in all cases, iPINNER is more accurate in estimating the missing parameters in TFMDWEs than ADAM-PINN and NSGA-III-PINN. Similar as in the forward problem setting, the NSGA-III-PINN shows more accurate estimations than ADAM-PINN, as the NSGA-III algorithm more effectively balances multiple objectives within the loss function, thereby partially mitigating the adverse effects of noisy observations even in the missing parameters. This is also reflected in the figures [11](https://arxiv.org/html/2506.00731v2#S3.F11 "Figure 11 ‣ 3.2.3 Inverse Problem ‣ 3.2 Time-fractional Mixed Diffusion-Wave Equations (TFMDWEs) ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter")–[13](https://arxiv.org/html/2506.00731v2#S3.F13 "Figure 13 ‣ 3.2.3 Inverse Problem ‣ 3.2 Time-fractional Mixed Diffusion-Wave Equations (TFMDWEs) ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter") as well as Figure [14](https://arxiv.org/html/2506.00731v2#S3.F14 "Figure 14 ‣ 3.2.3 Inverse Problem ‣ 3.2 Time-fractional Mixed Diffusion-Wave Equations (TFMDWEs) ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter") which shows at different time instances (top: t=0.5 t=0.5, bottom: t=1 t=1), the true solution (black) and inverse problem solutions obtained using ADAM-PINN (red), NSGA-III-PINN (blue), and iPINNER (green) for the time-fractional mixed diffusion-wave equations (TFMDWEs) with different noise levels. The results show that iPINNER (green) outperforms both ADAM-PINN (red) and NSGA-III-PINN (blue) in terms of estimating parameters and PDE solutions.

![Image 38: Refer to caption](https://arxiv.org/html/2506.00731v2/TFMDWEs_true_forward_noise20.png)

(a) 

![Image 39: Refer to caption](https://arxiv.org/html/2506.00731v2/TFMDWEs_MoPINNenKF_inverse_noise20.png)

(b) 

![Image 40: Refer to caption](https://arxiv.org/html/2506.00731v2/err_TFMDWEs_data_adam_inverse_noise20.png)

(c) 

![Image 41: Refer to caption](https://arxiv.org/html/2506.00731v2/err_TFMDWEs_MoPINNenKF_inverse_noise20.png)

(d)

Figure 11:  Comparison of (a) the true solution and (b) the iPINNER solution as well as the L 1 L_{1} errors of (c) the ADAM-PINN and (d) the iPINNER for the time-fractional mixed diffusion-wave equations (TFMDWEs) in inverse problem. Observation noise level is 20%20\%. 

![Image 42: Refer to caption](https://arxiv.org/html/2506.00731v2/TFMDWEs_true_forward_noise20.png)

(a) 

![Image 43: Refer to caption](https://arxiv.org/html/2506.00731v2/TFMDWEs_MoPINNenKF_inverse_noise50.png)

(b) 

![Image 44: Refer to caption](https://arxiv.org/html/2506.00731v2/err_TFMDWEs_data_adam_inverse_noise50.png)

(c) 

![Image 45: Refer to caption](https://arxiv.org/html/2506.00731v2/err_TFMDWEs_MoPINNenKF_inverse_noise50.png)

(d)

Figure 12:  Comparison of (a) the true solution and (b) the iPINNER solution as well as the L 1 L_{1} errors of (c) the ADAM-PINN and (d) the iPINNER for the time-fractional mixed diffusion-wave equations (TFMDWEs) in inverse problem. Observation noise level is 50%50\%. 

![Image 46: Refer to caption](https://arxiv.org/html/2506.00731v2/TFMDWEs_true_forward_noise20.png)

(a) 

![Image 47: Refer to caption](https://arxiv.org/html/2506.00731v2/TFMDWEs_MoPINNenKF_inverse_noise80.png)

(b) 

![Image 48: Refer to caption](https://arxiv.org/html/2506.00731v2/err_TFMDWEs_data_adam_inverse_noise80.png)

(c) 

![Image 49: Refer to caption](https://arxiv.org/html/2506.00731v2/err_TFMDWEs_MoPINNenKF_inverse_noise80.png)

(d)

Figure 13:  Comparison of (a) the true solution and (b) the iPINNER solution as well as the L 1 L_{1} errors of (c) the ADAM-PINN and (d) the iPINNER for the time-fractional mixed diffusion-wave equations (TFMDWEs) in inverse problem. Observation noise level is 80%80\%. 

![Image 50: Refer to caption](https://arxiv.org/html/2506.00731v2/inverse_noise_level_20_ut_t0.51.png)

![Image 51: Refer to caption](https://arxiv.org/html/2506.00731v2/inverse_noise_level_20_ut_t0.71.png)

(a) 20%20\% noise level 

![Image 52: Refer to caption](https://arxiv.org/html/2506.00731v2/inverse_noise_level_50_ut_t0.51.png)

![Image 53: Refer to caption](https://arxiv.org/html/2506.00731v2/inverse_noise_level_50_ut_t0.71.png)

(b) 50%50\% noise level 

![Image 54: Refer to caption](https://arxiv.org/html/2506.00731v2/inverse_noise_level_80_ut_t0.51.png)

![Image 55: Refer to caption](https://arxiv.org/html/2506.00731v2/inverse_noise_level_80_ut_t0.71.png)

(c) 80%80\% noise level 

Figure 14: Comparison at different time instances (top: t=0.5 t=0.5, bottom: t=0.7 t=0.7) between the true solution (black) and inverse problem solutions obtained using ADAM-PINN (red), NSGA-III-PINN (blue), and iPINNER (green) for the time-fractional mixed diffusion-wave equations (TFMDWEs) under varying noise levels: Panel (a) — 20%20\% noise, Panel (b) — 50%50\% noise, and Panel (c) — 80%80\% noise. 

Table 10: Comparison of PINN with Adam optimizer (ADAM-PINN), PINN with NSGA-III optimizer (NSGA-III-PINN) and the proposed integrated method (iPINNER) for estimating the fractional term α\alpha in the TFMDWEs. For reference, the true value of the fractional term is α=0.5\alpha=0.5.

### 3.3 Two-Dimensional Heat Equation

We next consider the classical two-dimensional heat equation on the unit square with homogeneous Dirichlet boundaries:

∂t u​(x,y,t)−κ​Δ​u​(x,y,t)=f​(x,y,t),(x,y)∈Ω≡[0,1]2,t∈[0,1],\displaystyle\partial_{t}u(x,y,t)-\kappa\,\Delta u(x,y,t)\;=\;f(x,y,t),\qquad(x,y)\in\Omega\equiv[0,1]^{2},\;t\in[0,1],(25)
u|∂Ω=0,\displaystyle u|_{\partial\Omega}=0,(26)
u​(x,y,0)=u 0​(x,y).\displaystyle u(x,y,0)=u_{0}(x,y).(27)

To enable quantitative comparison, we consider the analytical solution

u​(x,y,t)=sin⁡(π​x)​sin⁡(π​y)​e−2​π 2​κ​t,u(x,y,t)=\sin(\pi x)\sin(\pi y)\,\mathrm{e}^{-2\pi^{2}\kappa t},(28)

which corresponds to non forcing case, i.e., f≡0 f\equiv 0 and the initial condition u 0​(x,y)=sin⁡(π​x)​sin⁡(π​y)u_{0}(x,y)=\sin(\pi x)\sin(\pi y).

##### Neural network architecture

In order to make a fair comparison, with different optimizers, the PINN employs a consistent deep neural network architecture comprising 6 layers with 64 neurons in each layer. During the training phase, models corresponding to each optimizer were collected until the training loss converged below a prescribed threshold ϵ\epsilon. Specifically, the ADAM-PINN requires 30000 30000 epochs to achieve convergence. The proposed iPINNER requires of 5 5 generations with 5000 5000 epochs to ensure the convergence of training loss.

##### Training and Testing Data

The training data for the 2D Heat equation consist of three main components: Initial Condition (IC) points, Boundary Condition (BC) points, and Collocation Points. The IC points are sampled in space at the initial time t=0 t=0, while the BC points are sampled in time along the spatial boundaries ∂Ω\partial\Omega. The collocation points are randomly selected from the interior of the spatiotemporal domain, with 100 points used in this study. The testing data are sampled over the domain Ω×[0,T]\Omega\times[0,T] with Ω=[0,1]\Omega=[0,1]. The spatial mesh size is of h=Δ​x=Δ​y=1/10 h=\Delta x=\Delta y=1/10 and a temporal step size is Δ​t=0.01\Delta t=0.01.

##### Error criteria

We evaluate models using the L 2 L_{2} relative error between the benchmark solutions and the predictions of different models.

In the forward problem setting, we samples the observational data at N obs N_{\text{obs}} scattered space–time locations, which are then added with zero-mean Gaussian noise with standard deviation equal to a prescribed fraction η∈{20%,50%}\eta\in\{20\%,50\%\} of std​(u)\mathrm{std}(u) evaluated at those points (consistent with Sections[3.1](https://arxiv.org/html/2506.00731v2#S3.SS1 "3.1 Burgers Equation ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter")–[3.2](https://arxiv.org/html/2506.00731v2#S3.SS2 "3.2 Time-fractional Mixed Diffusion-Wave Equations (TFMDWEs) ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter")).

![Image 56: Refer to caption](https://arxiv.org/html/2506.00731v2/comparison_t0.5_20.png)

![Image 57: Refer to caption](https://arxiv.org/html/2506.00731v2/comparison_t1.0_20.png)

Figure 15: iPINNER prediction of 2D Heat equation with 20% noise data.

![Image 58: Refer to caption](https://arxiv.org/html/2506.00731v2/comparison_t0.5_50.png)

![Image 59: Refer to caption](https://arxiv.org/html/2506.00731v2/comparison_t1.0_50.png)

Figure 16: iPINNER prediction of 2D Heat equation with 50% noise data.

Figures[15](https://arxiv.org/html/2506.00731v2#S3.F15 "Figure 15 ‣ Error criteria ‣ 3.3 Two-Dimensional Heat Equation ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter")–[16](https://arxiv.org/html/2506.00731v2#S3.F16 "Figure 16 ‣ Error criteria ‣ 3.3 Two-Dimensional Heat Equation ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter") show the forward problem solutions using iPINNER (left) and the corresponding true solutions (middle) of the 2D heat equation at two different time instances (top: t=0.5 t=0.5, bottom: t=1 t=1) under different noise levels. The right panels show the corresponding L 1 L^{1} errors between the predicted and true solutions. iPINNER shows higher accuracy than ADAM-PINN, particularly under higher noise levels. With 50% data noise, the iPINNER has 0.0180 L 2 L_{2} relative error whereas ADAM-PINN has 0.0285 L 2 L_{2} relative error shown in Table[11](https://arxiv.org/html/2506.00731v2#S3.T11 "Table 11 ‣ Error criteria ‣ 3.3 Two-Dimensional Heat Equation ‣ 3 Numerical Results ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter")

Table 11: Wall time of training PINN and iPINNER with 50% noise data.

4 Conclusion and Future Work
----------------------------

In this paper, we introduce a novel iPINNER framework that integrates the physics informed neural network (PINN) with ensemble Kalman filter (EnKF) with the NSGA-III multi-objective optimizer. This framework can be used to address both forward problem and inverse problem with only limited noisy data in the context of partial differential equations (PDE). The iPINNER framework has several advantages: (i) The framework utilizes the multi-objective optimizer NSGA-III to find an optimal cluster of PINNs, and (ii) The uncertainty within this cluster, inherent from the NSGA-III optimization, can be integrated with observational noisy data using the ensemble Kalman filter. This process can then be iteratively applied to update the data loss during PINN training. (iii) The framework can be applied in both the forward and inverse problems in solving PDEs.

Building on the iPINNER framework’s effectiveness in solving forward and inverse problems with noisy data, several future directions necessitate further investigation. First, in the inverse problem setting, the unknown parameters are treated as trainable variables within the neural networks and are implicitly embedded in the PDE residual loss of the PINN. To quantify the uncertainty of the inferred parameters, these parameters together with the independent variables can be used as inputs to the neural network, allowing it to be trained across a wide range of physical parameters [harlim2021machine, gottwald2021combining]. Second, a Bayesian PINN framework for both inverse and forward models, as proposed in [yang2021b], can be integrated with the current framework. This integration could potentially improve the performance of the model in the presence of highly noisy data, as the Bayesian approach naturally quantifies the uncertainties arising from scattered noisy data. Third, the iPINNER framework can be extended to a continual learning framework, allowing it to adapt and improve over time [purvine2017comparative, howard2024multifidelity, roy2024exact]. In this approach, PINNs optimized with the NSGA-III algorithm can be incrementally updated with new observational data. This continual learning process allows the model to refine its predictions, improve accuracy, and maintain robustness when additional data becomes available. Moreover, it facilitates the dynamic update of systems and uncertainties, making it particularly valuable for real-time prediction and long time simulation of complex physical systems [ghunaim2023real, shaheen2022continual, yang2012adaptive].

Acknowledgment
--------------

Guang Lin acknowledges the National Science Foundation under grants DMS-2053746, DMS-2134209, ECCS-2328241, CBET-2347401, and OAC-2311848. The U.S. Department of Energy also supports this work through the Office of Science Advanced Scientific Computing Research program (DE-SC0023161) and the Office of Fusion Energy Sciences (DE-SC0024583).

Appendix A Mathematical Formulation of Ensemble Kalman Filter (EnKF)
--------------------------------------------------------------------

Consistent with the notation in Section[2.3](https://arxiv.org/html/2506.00731v2#S2.SS3 "2.3 Ensemble Kalman Filter (EnKF) ‣ 2 General framework ‣ iPINNER: An Iterative Physics-Informed Neural Network with Ensemble Kalman Filter"), let the forecast ensemble at time t t be denoted as follows

X t|t−1=[x t|t−1(1),,x t|t−1(2),,…,,x t|t−1(N)]∈ℝ d×N,\displaystyle X_{t|t-1}=\bigl[x_{t|t-1}^{(1)},,x_{t|t-1}^{(2)},,\dots,,x_{t|t-1}^{(N)}\bigr]\in\mathbb{R}^{d\times N},(29)

where N N is the ensemble size and d d denotes the dimension of the state variable. Each column x t|t−1(i)x_{t|t-1}^{(i)} represents one realization of the state vector in forecast models. Then, the prior mean x¯t|t−1\bar{x}_{t|t-1} and covariance R t|t−1 R_{t|t-1} (model forecasts) are then estimated by the following [evensen2003ensemble]

x¯t|t−1\displaystyle\bar{x}_{t|t-1}=1 N​∑i=1 N​x t|t−1(i),\displaystyle=\frac{1}{N}\sum{i=1}^{N}x_{t|t-1}^{(i)},(30)
R t|t−1\displaystyle R_{t|t-1}=1 N−1​∑i=1 N(x t|t−1(i)−x¯t|t−1)​(x t|t−1(i)−x¯t|t−1)⊤,\displaystyle=\frac{1}{N-1}\sum_{i=1}^{N}\bigl(x_{t|t-1}^{(i)}-\bar{x}_{t|t-1}\bigr)\bigl(x_{t|t-1}^{(i)}-\bar{x}_{t|t-1}\bigr)^{\top},(31)

which provide Monte Carlo approximations to the prior mean and covariance. Then, given an observation y t y_{t}, the ensemble members can be updated through Bayesian formula and yields the following:

x t(i)=x t|t−1(i)+K t​(y t(i)−H​x t|t−1(i)),i=1,…,N,\displaystyle x_{t}^{(i)}=x_{t|t-1}^{(i)}+K_{t}\bigl(y_{t}^{(i)}-Hx_{t|t-1}^{(i)}\bigr),\quad i=1,\dots,N,(32)

where y t(i)=y t+ϵ t(i)y_{t}^{(i)}=y_{t}+\epsilon_{t}^{(i)} are observations with observation error ϵ t(i)∼𝒩​(0,R)\epsilon_{t}^{(i)}\sim\mathcal{N}(0,R). The ensemble Kalman gain can be derived as follows:

K t=P t|t−1​H⊤​(H​P t|t−1​H⊤+R)−1.\displaystyle K_{t}=P_{t|t-1}H^{\top}\bigl(HP_{t|t-1}H^{\top}+R\bigr)^{-1}.(33)

