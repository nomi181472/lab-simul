# 2608.27351

Source: https://arxiv.org/html/2608.27351



Understanding Evolution Strategies for LLM Reasoning:Broader Reasoning Coverage than GRPO

Report GitHub Issue
×

Title:

Content selection saved. Describe the issue below:
Description:

Submit without GitHub
Submit in GitHub

arXiv is now an independent nonprofit!
Learn more
×

Back to arXiv

Why HTML?

Report Issue

Back to Abstract

Download PDF

Abstract
1 Introduction
2 Preliminaries

2.1 SFT & OPD Fine-tuning
2.2 GRPO Fine-tuning

Entropy Collapse.
Correct Answers under Repeated Sampling.

2.3 ES Optimization

GRPO versus ES.

3 RQ1: Does ES Exhibit the Same Post-Training Characteristics as GRPO?

3.1 Theoretically, How ES Population Diversity Can Support Pass@KK

ES Perturbations Induce Policy Diversity.
Multiple Policies in the ES Population Increase the Chance of Finding a Correct Answer.
Reward Weighting Can Exploit Population Heterogeneity.
Sufficient Conditions for Pass@KK Improvement of the ES Center.

3.2 Empirically, ES Achieves Higher Pass@KK Than GRPO and Base LLM

4 RQ2: Does ES Necessarily Cause Catastrophic Forgetting?

4.1 ES Induces Greater Parameter Drift
4.2 ES Effects Are Concentrated in a Small Subset of Larger-Magnitude Updates

Our Explanation for Why ES can Work on Full-Parameter LLM

4.3 Large Parameter Drift Does Not Necessarily Indicate Broad Forgetting

5 RQ3: What Parameter Settings and Estimators Make ES Effective and Scalable?

5.1 Reward Normalization, Perturbation Scale, and Population-Size Scaling

Reward Normalization.
Perturbation Scale.
Population-Size Scaling.

5.2 Why Two-Point ZO Intuition Does Not Directly Transfer to ES for Reasoning

6 Conclusion and Future Work

Conclusion.
Future Work.

References
A Proofs for the RQ1 Diversity–Coverage Analysis

A.1 Proof of Lemma 
A.2 Proof of Lemma 
A.3 Proof and Illustration of Lemma 
A.4 Proof of Proposition 

B Complete Majority-Vote Results
C Detailed Held-Out Performance Changes
D Complete Results for Magnitude-Thresholded ES
E ES–ZO Estimator Equivalence and Its Finite-Sample Boundary

E.1 A Unified Smoothed-Objective View
E.2 Covariance Required by Paired Subtraction
E.3 Local Perturbation Diagnostic

Protocol.
Objectives.
Randomness Coupling.

E.4 Diagnostic Results
E.5 Scope of the Evidence

F Population-Size Scaling Trajectories
G Experimental Details

G.1 Implementation-Level Training Procedure

Full-parameter one-point ES.

G.2 ES–GRPO FLOP Matching
G.3 Training Hyperparameters
G.4 Evaluation Configuration
G.5 Prompt Templates
G.6 Datasets and Setting Rationale
G.7 Evaluation Metrics

H Related Work

H.1 Evolution Strategies for LLM Reasoning Post-Training
H.2 Reinforcement Learning with Verifiable Rewards
H.3 Zeroth-Order Optimization for LLMs

I Pass@K Profiles of Individual Post-Training Methods and Sequential Compositions

 License: arXiv.org perpetual non-exclusive license
 

arXiv:2608.27351v2 [cs.LG] 28 Aug 2026

Understanding Evolution Strategies for LLM Reasoning:Broader Reasoning Coverage than GRPO

Yunpeng Ba

Affiliation: Southern University of Science and Technology

  
Zhi Zheng

Affiliation: National University of Singapore

  
Yue Xie

Affiliation: Southern University of Science and Technology

  
Jiaqing Li

Affiliation: Harbin Institute of Technology, Weihai

  
Xialiang Tong

Affiliation: Huawei Noah’s Ark Lab

  
Tao Zhong

Affiliation: Huawei Noah’s Ark Lab

  
Mingxuan Yuan

Affiliation: Huawei Noah’s Ark Lab

  
Zhichao Lu

Affiliation: City University of Hong Kong

  
Xuyang Wu

Affiliation: Southern University of Science and Technology

  
Zhenkun Wang

Affiliation: Southern University of Science and Technology

Abstract
Evolution Strategies (ES) have recently emerged as a memory-efficient post-training paradigm for LLM reasoning.
However, the optimization behavior of ES remains understudied, making it hard to define its advantage scope compared to mainstream post-training paradigms (e.g., Group Relative Policy Optimization (GRPO)).
By systematically investigating ES dynamics and mechanisms, this paper first identifies a performance advantage of ES over GRPO, theoretically and empirically showing that ES can lead to broader reasoning coverage, thereby better exploiting the reasoning capabilities of pretrained LLMs.
Theoretically, we show that verifier-projected Jensen–Shannon diversity across the ES population is helpful to higher Pass@K performances.
Empirically, unlike GRPO, which exhibits entropy collapse, ES improves Pass@1 while attaining higher Pass@K than GRPO. We further develop a sequential GRPO–ES training strategy that combines GRPO’s strength in Pass@1 with ES’s gains in Pass@K.
Second, we find that despite substantial whole-model parameter drift, the task-performance gains of ES are only contributed to a sparse subset of larger-magnitude updates. This functional sparsity suggests that large parameter movement need not imply widespread functional change, and held-out evaluations further show that it does not necessarily lead to catastrophic forgetting.
Finally, we study how hyperparameter design affects the effectiveness of ES, demonstrating that ES requires a smaller population size in a larger LLM.
These findings position ES as a distinct reasoning post-training paradigm rather than a less effective, memory-efficient alternative to GRPO.
††GitHub \protect\githubemoji: https://github.com/yunpengba7/understanding-es††correspondence: zhi.zheng@u.nus.edu, wangzhenkun90@gmail.com

Figure 1: Overview of the three research questions and main findings. Panel (a) contrasts ES and GRPO post-training behavior; Panel (b) shows that keeping larger ES updates preserves target-task performance and reports held-out Maj@32 changes; Panel (c) summarizes normalization, population-size, and estimator choices.

1 Introduction

Recent work has shown that Evolution Strategies (ES) (Salimans et al., 2017) can fine-tune large language models (LLMs) to improve their reasoning capabilities (Qiu et al., 2026; Sarkar et al., 2026; Zheng et al., 2026a).
ES optimizes a model by perturbing its parameters, evaluating the perturbed models through forward passes, and aggregating reward-weighted perturbations to estimate an update direction.
By avoiding backpropagation, ES provides a memory-efficient and highly parallelizable approach to LLM post-training.
Despite these efficiency advantages, the practical viability of ES remains unclear: its post-training behavior, susceptibility to catastrophic forgetting (Hoy et al., 2026; Abdi et al., 2026), and conditions for effective and scalable optimization are still poorly understood.

We compare ES with Group Relative Policy Optimization (GRPO) (Shao et al., 2024), a mainstream RL method that likewise optimizes verifier rewards from sampled responses.
ES performs population-based search over parameter-perturbed policies, whereas GRPO samples multiple responses from a single policy and backpropagates a token-level objective based on their relative advantages.
While GRPO effectively improves Pass@1, it can exhibit entropy collapse, leading to increasingly concentrated exploration (Cui et al., 2025; Petrenko et al., 2026; Jin et al., 2026).
Consistently, prior analyses find that RL post-training can reduce reasoning-path diversity and even lower large-KK Pass@K relative to the base model (Yue et al., 2025; Zhao et al., 2025; Wu et al., 2025).
This suggests that GRPO may concentrate probability on a narrower set of successful reasoning modes, making low-probability correct solutions less accessible.
We therefore investigate whether ES exhibits the same narrowing behavior or preserves broader reasoning coverage through population-based parameter-space exploration.

Beyond reasoning coverage, another important criterion for post-training is whether the model preserves its pretrained capabilities.
Compared with supervised fine-tuning (SFT), RL-based post-training methods such as GRPO have been shown to better preserve pretrained capabilities and to be less prone to catastrophic forgetting (Jin et al., 2025; Yuan et al., 2025).
This makes capability preservation an important reference point when evaluating whether ES can serve as a practical alternative to mainstream RL post-training.

For ES, however, whether such capability preservation holds remains unclear.
Prior work observes substantial parameter drift alongside catastrophic forgetting under ES and attributes the latter to the former (Abdi et al., 2026).
Yet whether parameter drift itself causes forgetting remains unverified, as existing evidence is limited to specific tasks and small training sets.
Beyond capability preservation, deploying ES effectively also requires understanding how its hyperparameters and estimator designs affect optimization stability and scalability, particularly as model size increases.
Together with the distinct exploration behavior discussed above, these open questions motivate our systematic study of ES. So, this paper summarizes our three research questions and main findings in Figure 1 and as follows:

1.

RQ1: Does ES exhibit the same post-training characteristics as GRPO?
We find that ES maintains broader reasoning coverage than GRPO.
Across models post-trained on GSM8K and DeepScaleR, ES improves Pass@1 while achieving higher Pass@KK than GRPO, without exhibiting the same entropy collapse.
Theoretically, we show that verifier-projected Jensen–Shannon diversity across the ES population improves repeated-sampling success and can translate into higher Pass@KK in the ES-updated policy.
We further develop two sequential compositions, ES→\rightarrowGRPO and GRPO→\rightarrowES, that combine GRPO’s strength in Pass@1 with ES’s gains in Pass@KK.

2.

RQ2: Does ES necessarily cause catastrophic forgetting?
By examining the distribution of parameter changes, we find that the task-relevant effects of ES are concentrated in a small subset of larger-magnitude updates, while most parameter changes contribute little after perturbation cancellation.
This functional sparsity suggests that substantial whole-model drift need not correspond to widespread functional change. Consistently, held-out capabilities remain largely preserved under appropriate training settings, indicating that large parameter movement alone does not imply catastrophic forgetting and that prior observations are better explained by training-set overfitting.

3.

RQ3: What hyperparameter settings and estimators make ES effective and scalable?
We systematically evaluate ES hyperparameters and estimator designs to identify stable and effective configurations. We find that z-score reward normalization is a key ingredient for effective ES training. Due to the discrete reward in reasoning, the two-point estimator commonly favored in zeroth-order SFT provides no advantage for ES. We further find that the population size required for effective optimization decreases as pretrained model scale increases.

2 Preliminaries

Let θ\theta denote the model parameters and 𝒟train\mathcal{D}_{\mathrm{train}} the distribution of reasoning prompts.
Given x∼𝒟trainx\sim\mathcal{D}_{\mathrm{train}}, an LLM typically generates a chain-of-thought (CoT) trajectory cc followed by a final answer aa:

c∼πθ(⋅∣x),a∼πθ(⋅∣x,c),c\sim\pi_{\theta}(\cdot\mid x),\qquad a\sim\pi_{\theta}(\cdot\mid x,c),

(1)

and we denote the complete response as y=(c,a)y=(c,a).
A verifier assigns a scalar reward r⁡(x,y)r(x,y), yielding the objective as follows:

F(θ)=𝔼x∼𝒟train𝔼y∼πθ(⋅∣x)[r(x,y)].F(\theta)=\mathbb{E}_{x\sim\mathcal{D}_{\mathrm{train}}}\mathbb{E}_{y\sim\pi_{\theta}(\cdot\mid x)}[r(x,y)].

(2)

Several post-training paradigms have been used to improve LLM reasoning capabilities, including SFT, On-Policy Distillation (OPD), GRPO, and ES. These methods differ primarily in the supervision used to optimize the generated reasoning trajectories.

2.1 SFT & OPD Fine-tuning

SFT learns from expert CoT trajectories (Chu et al., 2025; Zheng and Lee, 2025), while OPD learns from token distributions of stronger teacher models (Song and Zheng, 2026).
Both require supervision beyond scalar verifier rewards, so we exclude them from our main comparison.

2.2 GRPO Fine-tuning

GRPO instead learns directly from verifier rewards.
We use GRPO as the primary gradient-based comparator to ES, as both optimize Eq. (2) but use fundamentally different update mechanisms.

For each reasoning prompt xx, GRPO (Shao et al., 2024) samples a group of GG responses
{yi=(ci,ai)}i=1G\{y_{i}=(c_{i},a_{i})\}_{i=1}^{G} from the behavior policy
πθold\pi_{\theta_{\mathrm{old}}} and obtains verifier rewards
ri=r⁡(x,yi)r_{i}=r(x,y_{i}).
Let r¯G\bar{r}_{G} and sGs_{G} denote the mean and population standard deviation of the group rewards.
For sG>0s_{G}>0, GRPO computes the response-level advantage

A^i=ri−r¯GsG.\widehat{A}_{i}=\frac{r_{i}-\bar{r}_{G}}{s_{G}}.

(3)

It then applies the same A^i\widehat{A}_{i} to all tokens in yiy_{i} through a PPO-style clipped objective (Schulman et al., 2017).
Let Ti=|yi|T_{i}=|y_{i}| and

ϱi,t​(θ)=πθ​(yi,t∣x,yi,<t)πθold​(yi,t∣x,yi,<t).\varrho_{i,t}(\theta)=\frac{\pi_{\theta}(y_{i,t}\mid x,y_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid x,y_{i,<t})}.

The GRPO objective is

ℒGRPO(θ)=−1G∑i=1G1Ti∑t=1Timin(ϱi,tA^i,clip(ϱi,t,1−εclip,1+εclip)A^i).\mathcal{L}_{\mathrm{GRPO}}(\theta)=-\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_{i}}\sum_{t=1}^{T_{i}}\min\!\left(\varrho_{i,t}\widehat{A}_{i},\,\operatorname{clip}(\varrho_{i,t},1-\varepsilon_{\mathrm{clip}},1+\varepsilon_{\mathrm{clip}})\widehat{A}_{i}\right).

(4)

The group reward provides a prompt-specific baseline without a learned critic, while each response-level advantage supervises the entire CoT trajectory and final answer.

Entropy Collapse.

A known issue of GRPO-style policy-gradient training is entropy collapse, where the policy distribution becomes increasingly concentrated during optimization.
For a tabular softmax policy at state ss, let pa=πθ​(a∣s)p_{a}=\pi_{\theta}(a\mid s) and Aa=A⁡(s,a)A_{a}=A(s,a), with
𝔼a∼πθ​[Aa]=0\mathbb{E}_{a\sim\pi_{\theta}}[A_{a}]=0.
For policy entropy
H(s)=−∑apalogpaH(s)=-\sum_{a}p_{a}\log p_{a}, prior work (Cui et al., 2025) gives the local relation

Δ​H​(s)≈−ηPG​Cov⁡(log⁡pa,pa​Aa).\Delta H(s)\approx-\eta_{\mathrm{PG}}\,\operatorname{Cov}\!\left(\log p_{a},p_{a}A_{a}\right).

(5)

When high-probability actions tend to receive positive advantages, the covariance becomes positive, yielding Δ​H​(s)<0\Delta H(s)<0 (Cui et al., 2025; Petrenko et al., 2026).
Repeated updates can therefore reduce policy entropy and response diversity (Jin et al., 2026).

Correct Answers under Repeated Sampling.

Entropy measures overall distributional concentration, but does not directly show whether repeated sampling can produce a correct response.
We therefore use Pass@K, which measures whether at least one of KK sampled responses is correct.
Recent work shows that GRPO can reduce Pass@K relative to the Base Model, indicating a lower probability of finding a correct response through repeated sampling even when single-sample accuracy improves (Yue et al., 2025).

2.3 ES Optimization

ES optimizes F⁡(θ)F(\theta) through parameter-space evaluations (Salimans et al., 2017).
For perturbation scale σ>0\sigma>0, it considers the Gaussian-smoothed objective as follows:

Fσ​(θ)=𝔼ϵ∼𝒩⁡(0,I)​[F⁡(θ+σ​ϵ)].F_{\sigma}(\theta)=\mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)}[F(\theta+\sigma\epsilon)].

At each update, ES samples NN perturbations {ϵi}i=1N\{\epsilon_{i}\}_{i=1}^{N} and evaluates the corresponding perturbed models θ+σ​ϵi\theta+\sigma\epsilon_{i}, obtaining rollout rewards RiR_{i}. The standard one-point estimator is as follows:

g^ESraw=1N​σ​∑i=1NRi​ϵi,𝔼⁡[g^ESraw]=∇Fσ​(θ).\widehat{g}_{\mathrm{ES}}^{\mathrm{raw}}=\frac{1}{N\sigma}\sum_{i=1}^{N}R_{i}\epsilon_{i},\qquad\mathbb{E}\!\left[\widehat{g}_{\mathrm{ES}}^{\mathrm{raw}}\right]=\nabla F_{\sigma}(\theta).

(6)

In practice, following Qiu et al. (2026), we standardize rewards within each population, analogous to the group normalization in Eq. (3).
Let ziz_{i} denote the standardized reward.
The ES search direction and center-model update are as follows:

d^ES=1N​∑i=1Nzi​ϵi,θ+=θ+α​d^ES,\widehat{d}_{\mathrm{ES}}=\frac{1}{N}\sum_{i=1}^{N}z_{i}\epsilon_{i},\qquad\theta^{+}=\theta+\alpha\widehat{d}_{\mathrm{ES}},

(7)

where α>0\alpha>0 is the update scale and absorbs the fixed 1/σ1/\sigma factor.

Aspect

ES

GRPO

Signal path

Ri→zi→θR_{i}\!\rightarrow\!z_{i}\!\rightarrow\!\theta

ri→A^i→θr_{i}\!\rightarrow\!\widehat{A}_{i}\!\rightarrow\!\theta

Update rule

Standardized perturbation average

Clipped policy-gradient surrogate

Execution

Perturbed-policy rollouts; no backprop

Policy rollouts and backprop

Backward state

Not retained

Retained on GPU

Table 1: Update and efficiency comparison of ES and GRPO.

GRPO versus ES.

Table 1 highlights the key distinction: GRPO backpropagates token-level advantages through a single policy, whereas ES converts population-level reward differences directly into a parameter-space update.

This mechanism-level distinction motivates our comparison of post-training behavior, while broader concerns about capability preservation and practical optimization motivate two additional questions. Accordingly, we investigate three questions about ES: (1) Does ES exhibit the same post-training characteristics as GRPO, particularly in terms of reasoning coverage? (2) Does large parameter movement under ES necessarily lead to catastrophic forgetting? (3) What hyperparameter settings and estimators make ES effective and scalable? The following sections investigate these questions in turn.

3 RQ1: Does ES Exhibit the Same Post-Training Characteristics as GRPO?

RQ1 examines whether ES can improve Pass@1 without degrading Pass@KK at larger sampling budgets, unlike the entropy collapse and large-KK Pass@KK degradation that can occur under GRPO.

3.1 Theoretically, How ES Population Diversity Can Support Pass@KK

Theoretically, we find that population diversity in ES can increase the probability that at least one of multiple sampled responses is correct, and that this advantage can transfer to the updated center policy under suitable conditions. The reason is that parameter perturbations induce heterogeneous policies with different success probabilities; sampling across these policies increases the chance of discovering a correct response, while reward weighting can preferentially emphasize more successful members. This subsection formalizes these effects and characterizes the conditions under which the resulting population-level advantage is preserved after the ES update.

ES Perturbations Induce Policy Diversity.

We first quantify how ES parameter perturbations translate into diversity at the policy level.
To characterize the diversity induced by parameter perturbations, let
sθ​(y∣x)=∇θ​log​πθ​(y∣x)s_{\theta}(y\mid x)=\nabla_{\theta}\log\pi_{\theta}(y\mid x)
and πθx=πθ(⋅∣x)\pi_{\theta}^{x}=\pi_{\theta}(\cdot\mid x).
For a local displacement δ\delta, the prompt-conditioned Fisher information is

ℐx(θ)=𝔼Y∼πθ(⋅∣x)[sθ(Y∣x)sθ(Y∣x)⊤],\displaystyle\mathcal{I}_{x}(\theta)=\mathbb{E}_{Y\sim\pi_{\theta}(\cdot\mid x)}\left[s_{\theta}(Y\mid x)s_{\theta}(Y\mid x)^{\top}\right],

DKL(πθ+δx∥πθx)=12δ⊤ℐx(θ)δ+o(∥δ∥2).\displaystyle D_{\mathrm{KL}}\left(\pi_{\theta+\delta}^{x}\,\|\,\pi_{\theta}^{x}\right)=\frac{1}{2}\delta^{\top}\mathcal{I}_{x}(\theta)\delta+o(\|\delta\|^{2}).

(8)

For δ=σ​ϵ\delta=\sigma\epsilon, the expected local policy displacement is proportional to
σ22​tr⁡ℐx​(θ)\frac{\sigma^{2}}{2}\operatorname{tr}\mathcal{I}_{x}(\theta).

Let 𝒟\mathcal{D} be the evaluation-prompt distribution, let v⁡(x,y)∈{0,1}v(x,y)\in\{0,1\} be a correctness verifier, and define the correct-response set 𝒞⁡(x)={y:v⁡(x,y)=1}\mathcal{C}(x)=\{y:v(x,y)=1\}.
For any policy π\pi, let pπ(x)=PrY∼π(⋅∣x)[Y∈𝒞(x)]p_{\pi}(x)=\Pr_{Y\sim\pi(\cdot\mid x)}[Y\in\mathcal{C}(x)].
Unless an expectation is shown, quantities below are conditioned on the realized population, fitness values, and resulting update. So, we provide the lemma as follows:

Lemma 1 (Perturbations induce policy diversity).

For a fixed prompt xx and population size NN, draw
ϵi​∼iid​𝒩​(0,I)\epsilon_{i}\overset{\mathrm{iid}}{\sim}\mathcal{N}(0,I), define
πi​(y∣x)=πθ+σ​ϵi​(y∣x)\pi_{i}(y\mid x)=\pi_{\theta+\sigma\epsilon_{i}}(y\mid x) and
π¯N=N−1​∑iπi\bar{\pi}_{N}=N^{-1}\sum_{i}\pi_{i}, and let
JSNpol(x)=N−1∑iDKL(πi∥π¯N)\operatorname{JS}^{\mathrm{pol}}_{N}(x)=N^{-1}\sum_{i}D_{\mathrm{KL}}(\pi_{i}\,\|\,\bar{\pi}_{N}).
The prompt-conditioned Fisher information ℐx​(θ)\mathcal{I}_{x}(\theta) is defined in Equation (8).
Assuming finite entropy and Fisher information, common local support, and sufficient smoothness as detailed in Appendix A.1, as σ→0\sigma\to 0,

𝔼ϵ1:N[JSNpol(x)]=σ22(1−1N)trℐx(θ)+O(σ4).\mathbb{E}_{\epsilon_{1:N}}\!\left[\operatorname{JS}^{\mathrm{pol}}_{N}(x)\right]=\frac{\sigma^{2}}{2}\!\left(1-\frac{1}{N}\right)\operatorname{tr}\mathcal{I}_{x}(\theta)+O(\sigma^{4}).

(9)

Multiple Policies in the ES Population Increase the Chance of Finding a Correct Answer.

The benefit of sampling across different ES members is then established through comparison with matched sampling from a single policy.

Lemma 2 (Policy diversity improves correct-answer discovery).

For these policies, define
pi​(x)=pπi​(x)p_{i}(x)=p_{\pi_{i}}(x)
and p¯​(x)=N−1​∑ipi​(x)\bar{p}(x)=N^{-1}\sum_{i}p_{i}(x).
Using binary entropy hh, define
JSNsucc⁡(x)=h⁡(p¯​(x))−N−1​∑ih⁡(pi​(x))\operatorname{JS}^{\mathrm{succ}}_{N}(x)=h(\bar{p}(x))-N^{-1}\sum_{i}h(p_{i}(x)).
Data processing, followed by one independent sample per member, yields

0≤JSNsucc⁡(x)≤JSNpol⁡(x),\displaystyle 0\leq\operatorname{JS}^{\mathrm{succ}}_{N}(x)\leq\operatorname{JS}^{\mathrm{pol}}_{N}(x),

(10)

PNpop​(x):=1−∏i=1N(1−pi​(x))≥1−(1−p¯​(x))N=:PNsame​(x).\displaystyle P_{N}^{\mathrm{pop}}(x):=1-\prod_{i=1}^{N}(1-p_{i}(x))\geq 1-(1-\bar{p}(x))^{N}=:P_{N}^{\mathrm{same}}(x).

(11)

Here PNpopP_{N}^{\mathrm{pop}} and PNsameP_{N}^{\mathrm{same}} denote one-response-per-member success and its matched single-policy baseline.
Equality holds if and only if p1​(x)=⋯=pN​(x)p_{1}(x)=\cdots=p_{N}(x), and the local gap is proportional to JSNsucc⁡(x)\operatorname{JS}^{\mathrm{succ}}_{N}(x) as derived in Appendix A.2.

Reward Weighting Can Exploit Population Heterogeneity.

The condition under which reward weighting improves the population-level success rate is then characterized.

Lemma 3 (Reward weighting improves success under positive correlation).

Let wi≥0w_{i}\geq 0, ∑iwi=1\sum_{i}w_{i}=1, be normalized weights obtained by a monotone transformation of realized fitness RiR_{i}.
Define the reward-weighted policy mixture πw=∑iwi​πi\pi_{w}=\sum_{i}w_{i}\pi_{i} and its success probability pw​(x)=∑iwi​pi​(x)p_{w}(x)=\sum_{i}w_{i}p_{i}(x).
With Covi\operatorname{Cov}_{i} taken under a uniform member draw,
pw​(x)−p¯​(x)=N​Covi⁡(wi,pi​(x))p_{w}(x)-\bar{p}(x)=N\operatorname{Cov}_{i}(w_{i},p_{i}(x)).

The analytical comparator πw\pi_{w} improves over the uniform population average exactly when the weights and member success are positively correlated.
It is not the practical ES update or a comparison against πθ\pi_{\theta}.

Sufficient Conditions for Pass@KK Improvement of the ES Center.

Finally, sufficient conditions are established under which the ES-updated model achieves higher Pass@KK than the model before the update.

Proposition 1 (ES updates can improve Pass@KK).

Let Bw​(x)B_{w}(x) and Bθ+​(x)B_{\theta^{+}}(x) denote Bernoulli outcome distributions with success probabilities pw​(x)p_{w}(x) and pθ+​(x)p_{\theta^{+}}(x), respectively.
For a repeated-sampling budget KK, let
JK​(π)=𝔼x∼𝒟​[1−(1−pπ​(x))K]J_{K}(\pi)=\mathbb{E}_{x\sim\mathcal{D}}[1-(1-p_{\pi}(x))^{K}] denote expected Pass@KK.
Define the comparator margin mK=JK​(πw)−JK​(πθ)m_{K}=J_{K}(\pi_{w})-J_{K}(\pi_{\theta}), and let εsucc≥0\varepsilon_{\mathrm{succ}}\geq 0 be the center-transfer error budget.
Assume

𝔼x∼𝒟DKL(Bw(x)∥Bθ+(x))≤εsucc,\displaystyle\mathbb{E}_{x\sim\mathcal{D}}D_{\mathrm{KL}}\!\left(B_{w}(x)\,\|\,B_{\theta^{+}}(x)\right)\leq\varepsilon_{\mathrm{succ}},

(12)

mK>K​εsucc/2.\displaystyle m_{K}>K\sqrt{\varepsilon_{\mathrm{succ}}/2}.

(13)

Then

JK​(πθ+)≥JK​(πw)−K​εsucc/2>JK​(πθ).J_{K}(\pi_{\theta^{+}})\geq J_{K}(\pi_{w})-K\sqrt{\varepsilon_{\mathrm{succ}}/2}>J_{K}(\pi_{\theta}).

(14)

Thus, when verifier-relevant heterogeneity improves population coverage, reward-aligned selection produces a sufficient margin over the initial policy, and the center update preserves this margin; ES training can improve the model’s Pass@KK.
Proofs and implementation-related qualifications are provided in Appendix A.1–A.4.

3.2 Empirically, ES Achieves Higher Pass@KK Than GRPO and Base LLM

Experimental Setup.
In this part, we empirically evaluate whether ES can improve Pass@1 while preserving Pass@KK under two post-training settings.
The Easy Setting compares ES and GRPO on Qwen2.5-1.5B-Instruct, Llama-3.2-3B-Instruct, and Qwen2.5-7B-Instruct after two epochs of GSM8K post-training (Cobbe et al., 2021).
The Hard Setting compares them on DeepSeek-R1-Distill-Qwen-1.5B after one epoch of DeepScaleR post-training (Luo et al., 2025).
We FLOP-match ES and GRPO by using N=32N=32 perturbation directions for ES and G=8G=8 responses per prompt for GRPO. Appendix G.2 gives the complete accounting.

Training Process: GSM8K →\rightarrow GPQA

Test Results: Training Task →\rightarrow Test Task

(a) GRPO

Qwen2.5-1.5B-Instruct

(b) ES

Qwen2.5-1.5B-Instruct

(c) GSM8K →\rightarrow GPQA

Llama-3.2-3B-Instruct

(d) DeepScaleR →\rightarrow MATH-500

DeepSeek-R1-Distill-Qwen-1.5B

Figure 2: 
Comparison of ES and GRPO during training and testing.
During GSM8K post-training, GPQA token-level entropy drops substantially under GRPO but remains largely stable under ES; GRPO finishes below the base model on Pass@16 and Pass@32, whereas ES finishes above it.
On the representative task pairs, GRPO raises Pass@1 but lowers Pass@16 and Pass@32, whereas ES improves all three metrics.

Training Dynamics.
Figure 2(a–b) tracks held-out GPQA performance during GSM8K post-training of Qwen2.5-1.5B-Instruct.
Under GRPO, token-level entropy declines substantially, while Pass@16 and Pass@32 finish below the base model.
This pattern is consistent with prior reports linking entropy collapse under GRPO-style optimization to reduced reasoning diversity and weaker Pass@KK scaling (Jang et al., 2026).
In contrast, under ES, entropy changes modestly, and both metrics finish above the base model.

Test Results.
The representative task pairs in Figure 2(c–d) expose the Pass@1–Pass@KK difference directly.
On GPQA after GSM8K post-training and MATH-500 after DeepScaleR post-training, GRPO improves Pass@1 but falls below the corresponding base model on Pass@16 and Pass@32, whereas ES improves all three metrics.
The complete results in Tables 2 and 3 show that ES improves average Pass@1, Pass@16, and Pass@32 relative to the base model in both settings.
Although GRPO achieves larger average Pass@1 gains, ES attains higher average Pass@16 and Pass@32 in both settings.
Pass@KK degradation is particularly common in the Easy Setting: GRPO falls below the corresponding base model on both Pass@16 and Pass@32 in 15 of 18 comparisons.
These results indicate that ES yields gains at both K=1K=1 and larger KK, whereas GRPO’s gains are concentrated at K=1K=1 and often accompanied by lower large-KK performance.
Concurrent work reports a similar observation: ES achieves higher Pass@KK than RL and preserves solution coverage, whereas RL plateaus earlier and can be overtaken by the Base Model as KK increases (Hayes et al., 2026).

Methodology: Sequentially Mixed Training of ES and GRPO
ES leads in Pass@KK over base and GRPO, while GRPO improve much Pass@1 than ES. So, to combine their respective advantages, we propose an intuitive method by sequentially mixing the training process of ES and GRPO. Under the same total update budget, we additionally split training equally between two stages and evaluate both sequential compositions: ES→\rightarrowGRPO applies ES first and GRPO second, whereas GRPO→\rightarrowES reverses the order.
Figure 2 presents a representative training trajectory and two representative training-to-test task pairs, while Tables 2 and 3 report the complete Pass@KK results.
The corresponding Maj@KK results are reported in Appendix B.
For each prompt, Pass@1 measures single-sample correctness, Pass@KK records whether at least one of KK independent responses is correct, and Maj@KK reports majority-vote accuracy after answer normalization.

Pass Metrics

GSM8K
CSQA
HotpotQA
Countdown
GPQA
MBPP
Average

Method
@1
@16
@32
@1
@16
@32
@1
@16
@32
@1
@16
@32
@1
@16
@32
@1
@16
@32
@1
@16
@32

Qwen2.5-1.5B-Instruct

Base
70.3
94.8
96.4
60.6
93.4
96.1
27.6
46.6
50.3
7.5
46.9
56.9
21.4
87.1
94.4
58.6
83.8
87.2
41.0
75.4
80.2

GRPO
78.1
93.9
95.3
63.7
93.2
95.5
25.0
44.3
48.1
8.5
49.5
58.8
22.2
87.3
96.0
60.0
82.4
86.0
42.9
75.1
79.9

ES
73.2
94.9
96.6
60.7
94.1
96.8
24.9
45.8
49.9
8.8
49.4
59.5
22.9
88.5
96.5
58.7
83.0
86.4
41.5
76.0
80.9

ES→\rightarrowGRPO
77.1
94.4
96.0
62.1
93.9
96.5
24.3
44.0
47.9
8.3
47.9
57.1
22.2
88.1
96.0
59.8
83.4
86.8
42.3
75.3
80.0

GRPO→\rightarrowES
76.1
94.6
96.1
62.5
93.7
96.0
24.6
44.7
48.8
7.8
48.4
58.0
22.8
86.6
95.5
58.5
81.6
84.8
42.1
75.0
79.9

Llama-3.2-3B-Instruct

Base
77.7
96.8
98.1
67.3
92.8
95.3
26.6
44.2
47.6
7.7
47.4
56.6
23.0
82.0
90.4
62.1
80.6
83.7
44.1
74.0
78.6

GRPO
84.7
95.6
96.4
72.1
90.5
92.8
27.5
40.4
43.0
9.6
52.9
61.6
26.9
80.3
88.4
62.1
76.7
79.8
47.1
72.7
77.0

ES
81.6
96.4
97.9
69.9
93.0
95.5
29.0
45.7
48.9
9.0
51.3
60.6
25.1
86.9
95.5
61.1
81.2
84.1
45.9
75.8
80.4

ES→\rightarrowGRPO
84.1
95.0
96.2
72.0
91.3
93.7
26.1
41.0
44.1
10.3
53.5
63.4
24.4
71.4
79.3
63.5
80.4
82.1
46.7
72.1
76.5

GRPO→\rightarrowES
84.1
95.9
96.9
71.3
91.8
94.3
27.5
43.6
47.0
8.5
49.9
59.5
26.6
84.5
92.9
61.5
77.2
80.5
46.6
73.8
78.5

Qwen2.5-7B-Instruct

Base
91.8
97.6
98.0
80.0
92.9
94.3
45.4
52.4
53.6
34.1
70.3
74.8
33.3
82.2
89.4
76.1
86.4
87.9
60.1
80.3
83.0

GRPO
92.3
97.0
97.4
80.8
91.7
92.7
46.2
52.3
53.3
36.8
68.4
72.9
34.8
79.3
85.4
74.8
85.5
87.6
61.0
79.0
81.5

ES
91.0
97.6
98.1
79.0
92.6
94.3
45.1
52.1
53.4
33.0
71.9
76.7
34.4
83.6
88.9
75.1
85.7
87.6
59.6
80.6
83.1

ES→\rightarrowGRPO
92.1
97.6
98.0
80.2
92.4
93.7
46.0
52.1
53.1
36.0
70.7
75.5
35.5
79.9
85.9
75.2
85.1
86.0
60.8
79.7
82.0

GRPO→\rightarrowES
92.3
97.4
98.0
80.2
91.9
93.2
45.9
52.4
53.6
35.0
69.4
73.8
34.2
80.8
87.4
75.1
86.0
87.9
60.4
79.6
82.3

Table 2: 
Pass@KK results (×100\times 100) in the Easy Setting after two epochs of GSM8K post-training.
Underlining marks the best Base/GRPO/ES result in each task cell; bold and shading mark the best and second-best Average cells across all five methods.

Pass Metrics

AIME24
AIME25
AMC23
MATH500
Average

Method
@1
@16
@32
@1
@16
@32
@1
@16
@32
@1
@16
@32
@1
@16
@32

DeepSeek-R1-Distill-Qwen-1.5B

Base
23.4
59.9
66.7
22.6
43.8
50.0
64.1
93.6
95.0
80.9
96.7
98.0
47.7
73.5
77.4

GRPO
27.6
62.2
66.7
25.4
45.8
53.3
73.9
94.7
95.0
84.8
96.1
96.8
52.9
74.7
78.0

ES
22.8
60.3
70.0
23.2
46.7
50.0
70.5
96.0
97.5
83.2
97.2
98.2
49.9
75.0
78.9

ES→\rightarrowGRPO
29.0
70.6
76.7
23.8
43.0
50.0
72.3
92.5
92.5
84.4
97.1
97.8
52.3
75.8
79.2

GRPO→\rightarrowES
25.8
61.8
63.3
24.5
51.0
56.7
71.6
94.9
95.0
84.5
97.0
97.6
51.6
76.2
78.2

Table 3: 
Pass@KK results (×100\times 100) on mathematical benchmarks in the Hard Setting.
Underlining marks the best Base/GRPO/ES result in each task cell; bold and shading mark the best and second-best Average cells across all five methods.

Pareto Trade-offs from Sequential Compositions.
Figure 3 treats Pass@1 and Pass@KK as paired objectives in three representative model–task settings.
In each panel, the reference Pareto front formed by Base, GRPO, and ES reflects the trade-off between GRPO’s higher Pass@1 and ES’s higher large-KK coverage.
Adding the endpoints from the two sequential compositions introduces additional non-dominated points.
In particular, ES→\rightarrowGRPO attains the highest Pass@32 on the Hard-Setting math average while retaining most of GRPO’s Pass@1 gain, and GRPO→\rightarrowES supplies an intermediate trade-off on GPQA with Llama-3.2-3B-Instruct.

(a) GPQA

Llama-3.2-3B-Instruct

(b) Math Average

DeepSeek-R1-Distill-Qwen-1.5B

(c) GSM8K

Qwen2.5-1.5B-Instruct

Figure 3: 
Representative Pass@1–Pass@KK Pareto fronts across models and tasks.
The math average is computed equally over AIME24, AIME25, AMC23, and MATH-500; higher values are better on both axes.
Gray dashed lines connect the reference Pareto fronts formed by Base, GRPO, and ES.
Solid dark lines trace the full Pareto fronts after adding the endpoints from the two sequential compositions.
Faded markers are dominated in the corresponding pairwise comparison.
The two sequential compositions add non-dominated trade-off points in all three settings.

Takeaways.

•

Sampling across ES perturbations can increase Pass@KK. ES perturbations produce population members with different response distributions and success rates. Sampling once from each member is at least as likely to include a correct answer as drawing the same number of responses from a single policy with the same average success rate. Under the conditions in Section 3.1, the ES update can preserve this advantage in the updated model.

•

ES improves Pass@1 and attains higher Pass@KK than GRPO. Across the Easy and Hard Settings, ES improves average Pass@1, Pass@16, and Pass@32 over the Base Model and exceeds GRPO on average Pass@16 and Pass@32. Representative training dynamics further show that ES largely avoids the entropy decline and large-KK degradation observed under GRPO.

•

The two sequential compositions add new Pass@1–Pass@KK trade-offs. Under the same total update budget, adding the endpoints from ES→\rightarrowGRPO and GRPO→\rightarrowES expands the Pareto fronts in all three representative settings and provides additional non-dominated points.

4 RQ2: Does ES Necessarily Cause Catastrophic Forgetting?

RQ2 examines the scale and distribution of ES-induced parameter changes, identifies which updates account for performance gains, and evaluates whether large parameter drift necessarily leads to catastrophic forgetting.

4.1 ES Induces Greater Parameter Drift

ES evaluates full-parameter perturbations and can accumulate substantial whole-model parameter drift over successive updates. Prior works identify this large parameter movement as a characteristic difference between ES and gradient-based post-training (Hoy et al., 2026; Abdi et al., 2026). We first quantify this movement relative to matched GRPO training.

\FloatBarrier

Model
GRPO
Full ES
Magnitude-Thresholded ES

τ=1.0×10−3\tau=1.0\times 10^{-3}
τ=1.5×10−3\tau=1.5\times 10^{-3}
τ=2.0×10−3\tau=2.0\times 10^{-3}

Qwen2.5-1.5B-Instruct
0.0475
1.933 (40.7×\times)
1.615 (79.11%)
1.242 (92.47%)
0.871 (97.57%)

Llama-3.2-3B-Instruct
0.0954
4.185 (43.9×\times)
3.482 (78.08%)
2.589 (92.64%)
1.707 (97.89%)

Qwen2.5-7B-Instruct
0.0831
3.664 (44.1×\times)
3.032 (78.41%)
2.218 (93.02%)
1.427 (98.11%)

DeepSeek-R1-Distill-Qwen-1.5B
0.0548
2.338 (42.7×\times)
2.201 (62.18%)
2.017 (77.61%)
1.772 (87.29%)

Table 4: Relative whole-model L2L_{2} distance (×10−2\times 10^{-2}) and the distribution of nonzero ES updates across magnitude thresholds. Parentheses in the Full ES column report the distance relative to GRPO; parentheses in the magnitude-thresholded columns report sτs_{\tau}, the percentage of nonzero updates with magnitudes in (0,τ](0,\tau].

Following prior analyses of parameter drift of ES (Hoy et al., 2026; Abdi et al., 2026), we measure whole-model movement using relative L2L_{2} distance,

Drel​(θ,θ0)=∥θ−θ0∥2∥θ0∥2.D_{\mathrm{rel}}(\theta,\theta_{0})=\frac{\lVert\theta-\theta_{0}\rVert_{2}}{\lVert\theta_{0}\rVert_{2}}.

(15)

Here, θ0\theta_{0} and θ\theta denote the parameters before and after post-training, respectively.
Table 4 shows that Full ES is 40.7–44.1 times
farther from initialization than GRPO across the four models, confirming the substantially different parameter geometry reported in prior work.
Hoy et al. (2026) explain this movement by decomposing ES dynamics into reward-aligned progress and diffusion along locally flat or weakly reward-relevant directions.
Along these directions, the expected optimization signal is small, but stochastic perturbation aggregation can accumulate as a high-dimensional random walk.
Thus, the large total drift of ES may partly reflect accumulated movement in many directions with little effect on reward.

4.2 ES Effects Are Concentrated in a Small Subset of Larger-Magnitude Updates

However, relative whole-model distance aggregates changes across all parameter coordinates and does not reveal their magnitude distribution.
We therefore quantify the proportions of parameter changes within and beyond a range of magnitude thresholds.

Let Δ​θi=θi−θ0,i\Delta\theta_{i}=\theta_{i}-\theta_{0,i} denote the change of parameter ii from the Base Model.
For a threshold τ\tau, we first quantify the fraction of nonzero updates whose magnitudes fall within (0,τ](0,\tau]:

sτ=|{i:0<|Δ​θi|≤τ}||{i:|Δ​θi|>0}|,s_{\tau}=\frac{\left|\left\{i:0<|\Delta\theta_{i}|\leq\tau\right\}\right|}{\left|\left\{i:|\Delta\theta_{i}|>0\right\}\right|},

(16)

The complementary fraction, 1−sτ1-s_{\tau}, consists of updates with magnitudes greater than τ\tau.
Table 4 shows that most changed parameter coordinates have relatively small update magnitudes, whereas larger-magnitude updates form a smaller subset.
For example, at τ=1.5×10−3\tau=1.5\times 10^{-3}, which falls within the magnitude range of a single ES update, the interval (0,τ](0,\tau] contains 77.6–93.0% of the nonzero updates across the four models, leaving 7.0–22.4% above the threshold.

(a) GSM8K

Llama-3.2-3B-Instruct

(b) MATH-500

DeepSeek-R1-Distill-Qwen-1.5B

Figure 4: Target-task Pass@1 across update-sparsity levels in the Easy and Hard Settings. The dashed lines indicate
previously recorded Base Model performance, which also forms the connected
endpoint at 100% update sparsity.

Having established that larger-magnitude updates form a smaller subset, we next test whether removing the more numerous small-magnitude updates preserves task performance.
For each τ\tau, we construct a magnitude-thresholded checkpoint by setting Δ​θi=0\Delta\theta_{i}=0 for coordinates satisfying 0<|Δ​θi|≤τ0<|\Delta\theta_{i}|\leq\tau.
Under this operation, sτs_{\tau} is the Update Sparsity, and Full ES corresponds to sτ=0s_{\tau}=0.
Figure 4 shows that target-task Pass@1
remains broadly stable as progressively more small-magnitude updates are set to zero,
with noticeable degradation emerging only at high update sparsity.
Together, the update-magnitude statistics and thresholding experiments reveal magnitude sparsity with corresponding functional concentration: despite perturbing all parameters, ES produces larger-magnitude updates in a limited parameter subset, and retaining these updates preserves most of the performance gains.
The retained coordinates can therefore be viewed as defining an approximately performance-preserving coordinate subspace identified after training.

Our Explanation for Why ES can Work on Full-Parameter LLM

In our view, this post-hoc structure helps explain why full-parameter ES can remain effective in the high-dimensional parameter space of an LLM, because its performance gains do not depend equally on updates throughout the entire space. Frankle and Carbin (2018) mentioned that larger models contain more such effective structures, so ES can work on LLMs with billions of parameters.

We further compare the locations and scales of the largest ES and GRPO updates. The largest ES updates occur
mainly in LayerNorm weights and attention
projections. In Llama-3.2-3B-Instruct, the maximum magnitude is 0.011718750.01171875:
five of the nine maximum coordinates are input-LayerNorm weights, while the remaining maxima include attention and MLP projections.
In DeepSeek-R1-Distill-Qwen-1.5B, the maximum is 0.0156250.015625; 117 of its 144
maximum coordinates are LayerNorm weights and 24 are attention-projection
biases. Normalization parameters further account for 72 and 80 of the 100
largest updates in
Llama-3.2-3B-Instruct and DeepSeek-R1-Distill-Qwen-1.5B, respectively.
By contrast, under GRPO, the
largest Llama-3.2-3B-Instruct update is 0.000244140.00024414 (48×48\times smaller)
and its top 100 all occur in token embeddings; the
DeepSeek-R1-Distill-Qwen-1.5B GRPO endpoint peaks at 0.000976560.00097656
(16×16\times smaller), with its top 100 all in the language-model head. This
contrast suggests that ES may adapt through normalization and attention
parameters: LayerNorm rescales hidden states, while
attention projections route information. GRPO instead concentrates its largest changes in token
embeddings and the language-model head, adjacent to input
representations and output logits, consistent with token-level
gradient optimization.

4.3 Large Parameter Drift Does Not Necessarily Indicate Broad Forgetting

We next test whether the larger parameter movement of ES leads to held-out performance degradation.
Tables 2 and 5 report held-out-task performance for the evaluated models in the Easy and Hard Settings, respectively.
Table 10 in Appendix C additionally reports metric-wise performance changes for each Easy Setting model.

Pass Metrics

GPQA
MBPP
CSQA
Countdown
Average

Method
@1
@16
@32
@1
@16
@32
@1
@16
@32
@1
@16
@32
@1
@16
@32

DeepSeek-R1-Distill-Qwen-1.5B

Base
24.2
80.8
91.4
62.9
85.6
87.6
46.8
89.2
93.7
37.5
79.9
85.7
42.9
83.9
89.6

GRPO
31.8
86.4
91.9
66.6
87.0
88.3
47.5
88.9
93.6
36.6
77.5
82.2
45.6
85.0
89.0

ES
28.0
84.6
92.4
63.5
85.9
87.2
46.1
88.9
93.4
38.5
79.4
84.4
44.0
84.7
89.3

ES→\rightarrowGRPO
30.6
84.5
91.4
64.6
86.7
87.9
46.8
88.8
93.3
38.8
80.6
86.8
45.2
85.1
89.9

GRPO→\rightarrowES
30.6
86.7
93.4
64.2
85.2
87.2
47.3
88.7
93.2
38.1
79.9
85.3
45.1
85.1
89.8

Table 5: Pass@KK results (×100\times 100) on four held-out benchmarks in the one-epoch Hard Setting. Underlining marks the best Base/GRPO/ES result in each task cell; bold and shading mark the best and second-best Average cells across all five methods.

For all three Easy Setting models, the Pass@32 change averaged over the
five held-out tasks is positive under ES but negative under GRPO
(Table 10).
In the Hard Setting, both methods improve average Pass@1 and Pass@16 on the held-out non-mathematical benchmarks, while ES achieves higher average Pass@32 than GRPO (Table 5).
The corresponding majority-vote results show that ES exceeds both the Base Model and GRPO in average Maj@16 and Maj@32 (Table 9).

Taken together, these results suggest that ES generally maintains held-out performance despite large parameter drift.
This finding differs from Abdi et al. (2026), who report catastrophic forgetting under ES.
Their evidence was limited to a small training set and a single model, training task, and held-out benchmark, making broad capability forgetting difficult to distinguish from overfitting to the training set.

However, performance drops on some tasks are especially consequential in practically relevant continual-learning scenarios, where deployed models must repeatedly adapt to new tasks and knowledge while reliably preserving prior capabilities.
Prior exploratory work suggests that the effects of ES on prior capabilities may depend on the task and training configuration (Hoy et al., 2026).
This raises concern that ES may preserve capabilities inconsistently across tasks and repeated updates.

Takeaways.

•

ES induces greater whole-model parameter drift than GRPO. ES-trained models move substantially farther from their initial parameters than their GRPO-trained counterparts.

•

ES updates exhibit magnitude sparsity. Larger-magnitude updates occupy a small subset of parameters, while ignoring smaller updates largely preserves task performance. These larger updates occur mainly in normalization and attention parameters.

•

Large parameter drift does not necessarily lead to catastrophic forgetting. ES generally preserves held-out performance and performs better than GRPO across the Easy and Hard Settings.

5 RQ3: What Parameter Settings and Estimators Make ES Effective and Scalable?

RQ3 examines how reward normalization, perturbation scale, population size, and estimator choice affect effective and scalable ES training.

5.1 Reward Normalization, Perturbation Scale, and Population-Size Scaling

Qwen2.5-0.5B-Instruct
Qwen2.5-1.5B-Instruct
Qwen2.5-3B-Instruct

Update
N=8N=8
N=16N=16
N=32N=32
N=64N=64 (ref.)
N=8N=8
N=16N=16
N=32N=32
N=64N=64 (ref.)
N=8N=8
N=16N=16
N=32N=32
N=64N=64 (ref.)

100
0.4782
0.5021
0.5100
0.5085
0.8211
0.8250
0.8259
0.8244
0.8912
0.8945
0.8960
0.8957

200
0.4614
0.4995
0.5180
0.5201
0.8242
0.8314
0.8366
0.8352
0.8903
0.8970
0.8966
0.8992

300
0.4421
0.4935
0.5253
0.5287
0.8276
0.8387
0.8449
0.8438
0.8901
0.9001
0.9004
0.9031

Table 6: Smoothed GSM8K rewards across model and population sizes. Shading marks reduced-population results within 0.010.01 of the N=64N=64 reference; values use debiased exponential smoothing with weight 0.990.99.

Reward Normalization.

We z-score population rewards before using them to weight perturbation directions, making each update depend on relative rather than absolute reward scales.
In the matched single-point ES ablation, z-score normalization yields higher mean rewards than no normalization throughout the evaluated updates after initialization, indicating improved reward-guided optimization in this setting (Figure 1(c), item (1)).

Perturbation Scale.

The perturbation scale σ\sigma sets both the radius of parameter-space exploration and the smoothing level of the effective objective FσF_{\sigma}.
When σ\sigma is too small, ES explores a narrow neighborhood and may overfit the observed rewards, causing the search to become trapped around a local optimum.
When σ\sigma is too large, perturbations span an overly broad region, leading to excessive exploration that can destabilize training and collapse performance.
During training, σ\sigma should therefore be selected to limit early reward overfitting while maintaining stable reward-guided progress.

Population-Size Scaling.

Section 4.2 shows that ES can preserve task performance with a small subset of larger-magnitude updates.
Larger models may contain more such subsets with similar effects, making task-improving perturbations denser around pretrained weights, consistent with Frankle and Carbin (2018); Gan and Isola (2026).
We therefore compare matched GSM8K runs of Qwen2.5-{0.5,1.5,3}\{0.5,1.5,3\}B-Instruct with N∈{8,16,32,64}N\in\{8,16,32,64\} to test whether fewer directions suffice at larger scales.
At update 300, Table 6 shows that only N=32N=32 is within 0.010.01 of N=64N=64 for 0.5B, while both N=16N=16 and N=32N=32 meet this threshold for 1.5B and 3B; correspondingly, the N=16N=16-to-N=64N=64 gap falls from 0.03520.0352 to 0.00510.0051 and 0.00300.0030.
Together with the trajectories in Appendix F, these results indicate that smaller populations approach the N=64N=64 reward as model scale grows.
Accordingly, the population required for stable ES training appears to decrease with model scale.

5.2 Why Two-Point ZO Intuition Does Not Directly Transfer to ES for Reasoning

Zeroth-order (ZO) optimization is a popular gradient-free approach that typically estimates gradients by subtracting objective values at two symmetric perturbations and is commonly evaluated on non-reasoning NLP tasks such as SST-2 (Malladi et al., 2023).
ES can instead use one evaluation per direction, as in our one-point implementation, or the same antithetic two-point estimator; under matched objectives and perturbations, antithetic ES is algebraically identical to two-point ZO.
The benefit of subtraction depends on evaluation: supervised tasks can reuse fixed data, whereas reasoning tasks regenerate autoregressive responses whose early-token divergence can weaken paired covariance.
In our matched GSM8K run, two-point ES provides no training-reward or held-out advantage (Figure 1(c), item (3)), while Appendix E finds raw variance reduction for SST-2 but not reliably for regenerated GSM8K rewards.
Thus, two-point ZO gains on non-reasoning tasks do not necessarily transfer to ES for reasoning; we favor one-point estimation when coupling is weak.

Takeaways.

•

Z-score normalization yields higher training rewards. In the matched one-point ES ablation, z-score normalization consistently outperforms no normalization after initialization.

•

Larger models can use smaller ES populations. As model scale increases, reduced-population runs more closely approach the N=64N=64 reference. At update 300, N=16N=16 is within 0.010.01 of N=64N=64 for the 1.5B and 3B models, whereas only N=32N=32 meets this criterion for the 0.5B model.

•

Two-point estimation provides no advantage over one-point estimation. In the matched GSM8K experiment, two-point ES improves neither training reward nor held-out performance.

6 Conclusion and Future Work

Conclusion.

This work examines ES’s reasoning ability gains, possible forgetting as a side effect, and conditions for effective training for LLMs.
Theoretically and empirically, we demonstrate that ES improves Pass@1 while maintaining broader Pass@KK coverage than GRPO, which can exhibit entropy collapse alongside a decline in Pass@KK.
Held-out evaluations further indicate that large parameter movement does not necessarily erase existing capabilities, in contrast to previous reports.
Our design study identifies effective parameter and estimator choices and finds that larger models can train stably with smaller populations.
Together, these results position ES as a distinct reasoning post-training paradigm rather than a memory-efficient GRPO alternative.

Future Work.

Future work should better exploit ES’s incentivized-reasoning advantage by expanding access to correct paths assigned low probability by the base policy.
Continual learning should be further explored to clarify how parameter drift affects previously acquired capabilities over longer training horizons spanning multiple tasks, and to investigate whether approaches such as parameter-efficient updates can suppress harmful drift without constraining beneficial exploration.

References

Abdi et al. (2026)
I. Abdi, A. Gupta, M. Mok, A. Lu, N. Lee, and G. Anumanchipalli

Evolutionary strategies at scale lead to catastrophic forgetting.

In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers),

pp. 194–204.

External Links: Document,
Link

Cited by: §H.1,
§1,
§1,
§4.1,
§4.1,
§4.3.

Austin et al. (2021)
J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry, Q. Le, and C. Sutton

Program synthesis with large language models.

arXiv preprint arXiv:2108.07732.

External Links: Document,
Link

Cited by: §G.6.

Chu et al. (2025)
T. Chu, Y. Zhai, J. Yang, S. Tong, S. Xie, D. Schuurmans, Q. V. Le, S. Levine, and Y. Ma

Sft memorizes, rl generalizes: a comparative study of foundation model post-training.

arXiv preprint arXiv:2501.17161.

Cited by: §2.1.

Cobbe et al. (2021)
K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman

Training verifiers to solve math word problems.

arXiv preprint arXiv:2110.14168.

External Links: Document,
Link

Cited by: §G.6,
§3.2.

Cover and Thomas (2006)
T. M. Cover and J. A. Thomas

Elements of information theory.

2 edition, Wiley-Interscience.

External Links: Document,
Link

Cited by: Appendix A.

Cui et al. (2025)
G. Cui, Y. Zhang, J. Chen, L. Yuan, Z. Wang, Y. Zuo, H. Li, Y. Fan, H. Chen, W. Chen, Z. Liu, H. Peng, L. Bai, W. Ouyang, Y. Cheng, B. Zhou, and N. Ding

The entropy mechanism of reinforcement learning for reasoning language models.

arXiv preprint arXiv:2505.22617.

External Links: Document,
Link

Cited by: §H.2,
§1,
§2.2,
§2.2.

Frankle and Carbin (2018)
J. Frankle and M. Carbin

The lottery ticket hypothesis: finding sparse, trainable neural networks.

arXiv preprint arXiv:1803.03635.

Cited by: §4.2,
§5.1.

Gan and Isola (2026)
Y. Gan and P. Isola

Neural thickets: diverse task experts are dense around pretrained weights.

In Proceedings of the 43rd International Conference on Machine Learning,

External Links: Link

Cited by: §G.2,
§5.1.

Gautam et al. (2024)
T. Gautam, Y. Park, H. Zhou, P. Raman, and W. Ha

Variance-reduced zeroth-order methods for fine-tuning language models.

In Proceedings of the 41st International Conference on Machine Learning,

Proceedings of Machine Learning Research, Vol. 235, pp. 15180–15208.

External Links: Link

Cited by: §H.3.

Hayes et al. (2026)
C. F. Hayes, E. Meyerson, K. Schweighofer, R. Dailey, B. Hodjat, R. Miikkulainen, and X. Qiu

Beyond the best guess: improving llm solution coverage with evolution strategies.

 arXiv.

External Links: Document,
Link

Cited by: §3.2.

He et al. (2025)
B. He, Z. Qu, Z. Liu, Y. Chen, Y. Zuo, C. Qian, K. Zhang, W. Chen, C. Xiao, G. Cui, N. Ding, and Z. Liu

JustRL: scaling a 1.5b LLM with a simple RL recipe.

arXiv preprint arXiv:2512.16649.

External Links: Document,
Link

Cited by: §G.6.

Hoy et al. (2026)
W. Hoy, B. Wang, and X. Pan

Matching accuracy, different geometry: evolution strategies vs GRPO in LLM post-training.

arXiv preprint arXiv:2604.01499.

External Links: Document,
Link

Cited by: §H.1,
§1,
§4.1,
§4.1,
§4.1,
§4.3.

HuggingFaceTB (2025)
HuggingFaceTB

Countdown-task-gold.

Note: Hugging Face dataset

External Links: Link

Cited by: §G.6.

Jang et al. (2026)
J. Jang, H. Lee, and S. Kim

A few bad apples spoil the bunch: preventing global entropy collapse driven by a small set of tokens in LLM reasoning.

In Findings of the Association for Computational Linguistics: ACL 2026,

pp. 13134–13154.

External Links: Document,
Link

Cited by: §H.2,
§3.2.

Jin et al. (2025)
H. Jin, S. Luan, T. Ni, S. Lyu, G. Rabusseau, R. Rabbany, D. Precup, and M. Hamdaqa

Rl fine-tuning heals ood forgetting in sft.

arXiv preprint arXiv:2509.12235.

Cited by: §1.

Jin et al. (2026)
R. Jin, P. Gao, Y. Ren, Z. Han, T. Zhang, W. Huang, W. Liu, J. Luan, and D. Xiong

Revisiting entropy in reinforcement learning for large reasoning models.

In Findings of the Association for Computational Linguistics: ACL 2026,

pp. 25300–25322.

External Links: Document,
Link

Cited by: §H.2,
§1,
§2.2.

Korotyshova et al. (2025)
D. Korotyshova, B. Shaposhnikov, A. Malakhov, A. Khokhulin, N. Surnachev, K. Ovcharenko, G. Bredis, A. Gorbatovski, V. Sinii, and D. Gavrilov

ESSA: evolutionary strategies for scalable alignment.

arXiv preprint arXiv:2507.04453.

External Links: 2507.04453,
Link

Cited by: §H.1.

Lightman et al. (2023)
H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, and K. Cobbe

Let’s verify step by step.

arXiv preprint arXiv:2305.20050.

External Links: Document,
Link

Cited by: §G.6.

Lin (1991)
J. Lin

Divergence measures based on the shannon entropy.

IEEE Transactions on Information Theory 37 (1), pp. 145–151.

External Links: Document,
Link

Cited by: Appendix A.

Luo et al. (2025)
M. Luo, S. Tan, J. Wong, X. Shi, W. Y. Tang, M. Roongta, C. Cai, and J. Luo

DeepScaleR: surpassing o1-preview with a 1.5b model by scaling RL.

Note: Notion Blog

External Links: Link

Cited by: §G.6,
§3.2.

Malladi et al. (2023)
S. Malladi, T. Gao, E. Nichani, A. Damian, J. D. Lee, D. Chen, and S. Arora

Fine-tuning language models with just forward passes.

In Advances in Neural Information Processing Systems,

Vol. 36, pp. 53038–53075.

External Links: Document,
Link

Cited by: §H.3,
§5.2.

math-ai (2025)
math-ai

AMC 2023.

Note: Hugging Face dataset

External Links: Link

Cited by: §G.6.

Nesterov and Spokoiny (2017)
Y. E. Nesterov and V. G. Spokoiny

Random gradient-free minimization of convex functions.

Foundations of Computational Mathematics 17 (2), pp. 527–566.

External Links: Document,
Link

Cited by: Appendix E,
§H.3.

Petrenko et al. (2026)
A. Petrenko, B. Lipkin, K. Chen, E. Wijmans, M. F. Cusumano-Towner, R. Giryes, and P. Krähenbühl

Entropy-preserving reinforcement learning.

In The Fourteenth International Conference on Learning Representations,

External Links: Link

Cited by: §H.2,
§1,
§2.2.

Qiu et al. (2026)
X. Qiu, Y. Gan, C. F. Hayes, Q. Liang, Y. Xu, R. Dailey, E. Meyerson, B. Hodjat, and R. Miikkulainen

Evolution strategies at scale: LLM fine-tuning beyond reinforcement learning.

In Forty-third International Conference on Machine Learning,

External Links: Link

Cited by: §H.1,
§1,
§2.3.

Rein et al. (2023)
D. Rein, B. L. Hou, A. C. Stickland, J. Petty, R. Y. Pang, J. Dirani, J. Michael, and S. R. Bowman

GPQA: a graduate-level google-proof Q&A benchmark.

arXiv preprint arXiv:2311.12022.

External Links: Document,
Link

Cited by: §G.6.

Salimans et al. (2017)
T. Salimans, J. Ho, X. Chen, S. Sidor, and I. Sutskever

Evolution strategies as a scalable alternative to reinforcement learning.

arXiv preprint arXiv:1703.03864.

External Links: Document,
Link

Cited by: Appendix E,
§H.1,
§1,
§2.3.

Sarkar et al. (2026)
B. Sarkar, M. Fellows, J. A. Duque, A. Letcher, A. L. Villares, A. Sims, C. Wibault, D. Samsonov, D. Cope, J. Liesen, K. Li, L. Seier, T. Wolf, U. Berdica, V. Mohl, A. D. Goldie, A. Courville, K. Sevegnani, S. Whiteson, and J. N. Foerster

Evolution strategies at the hyperscale.

In Proceedings of the 43rd International Conference on Machine Learning,

External Links: Link

Cited by: §H.1,
§1.

Schulman et al. (2017)
J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov

Proximal policy optimization algorithms.

arXiv preprint arXiv:1707.06347.

External Links: Document,
Link

Cited by: §2.2.

Shao et al. (2024)
Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo

DeepSeekMath: pushing the limits of mathematical reasoning in open language models.

arXiv preprint arXiv:2402.03300.

External Links: Document,
Link

Cited by: §H.2,
§1,
§2.2.

Song and Zheng (2026)
M. Song and M. Zheng

A survey of on-policy distillation for large languagere models.

arXiv preprint arXiv:2604.00626.

Cited by: §2.1.

Sun et al. (2026)
Z. Sun, S. Dang, G. Dai, and H. Ye

ESSAM: a novel competitive evolution strategies approach to reinforcement learning for memory efficient LLMs fine-tuning.

arXiv preprint arXiv:2602.01003.

External Links: 2602.01003,
Link

Cited by: §H.1.

Talmor et al. (2019)
A. Talmor, J. Herzig, N. Lourie, and J. Berant

CommonsenseQA: a question answering challenge targeting commonsense knowledge.

In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies,

pp. 4149–4158.

External Links: Document,
Link

Cited by: §G.6.

Wu et al. (2025)
F. Wu, W. Xuan, X. Lu, Z. Harchaoui, and Y. Choi

The invisible leash: why RLVR may not escape its origin.

In The 2nd AI for MATH Workshop at the 42nd International Conference on Machine Learning,

External Links: Link

Cited by: §H.2,
§1.

Xu et al. (2026)
H. Xu, S. Zhao, X. Wu, and A. T. Luu

Understanding and preventing entropy collapse in RLVR with on-policy entropy flow optimization.

In Findings of the Association for Computational Linguistics: ACL 2026,

pp. 17759–17771.

External Links: Document,
Link

Cited by: §H.2.

Yang et al. (2018)
Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. Cohen, R. Salakhutdinov, and C. D. Manning

HotpotQA: a dataset for diverse, explainable multi-hop question answering.

In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing,

pp. 2369–2380.

External Links: Document,
Link

Cited by: §G.6.

Yuan et al. (2025)
X. Yuan, X. Chen, T. Yu, D. Shi, C. Jin, W. Lee, and S. Mitra

Mitigating forgetting between supervised and reinforcement learning yields stronger reasoners.

arXiv preprint arXiv:2510.04454.

Cited by: §1.

Yue et al. (2025)
Y. Yue, Z. Chen, R. Lu, A. Zhao, Z. Wang, Y. Yue, S. Song, and G. Huang

Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model?.

In Advances in Neural Information Processing Systems,

Vol. 38, pp. 57654–57689.

External Links: Link

Cited by: §H.2,
§1,
§2.2.

Zhang and Math-AI Team (2024)
Y. Zhang and Math-AI Team

American invitational mathematics examination (AIME) 2024.

Note: Hugging Face dataset

External Links: Link

Cited by: §G.6.

Zhang and Math-AI Team (2025)
Y. Zhang and Math-AI Team

American invitational mathematics examination (AIME) 2025.

Note: Hugging Face dataset

External Links: Link

Cited by: §G.6.

Zhang et al. (2024)
Y. Zhang, P. Li, J. Hong, J. Li, Y. Zhang, W. Zheng, P. Chen, J. D. Lee, W. Yin, M. Hong, Z. Wang, S. Liu, and T. Chen

Revisiting zeroth-order optimization for memory-efficient LLM fine-tuning: a benchmark.

In Proceedings of the 41st International Conference on Machine Learning,

pp. 59173–59190.

External Links: Link

Cited by: §H.3.

Zhao et al. (2025)
R. Zhao, A. Meterez, S. M. Kakade, C. Pehlevan, S. Jelassi, and E. Malach

Echo chamber: RL post-training amplifies behaviors learned in pretraining.

In Proceedings of the 2nd Conference on Language Modeling,

External Links: Link

Cited by: §H.2,
§1.

Zheng et al. (2026a)
Z. Zheng, R. Chen, Y. Ba, Z. Wang, Y. W. Teh, and W. S. Lee

Agentic esopt: fine-tuning long-horizon llm agents with minimal gpu requirements.

arXiv preprint arXiv:2608.17310.

Cited by: §1.

Zheng et al. (2026b)
Z. Zheng, Y. Gu, W. Liu, Y. W. Teh, and W. S. Lee

SofT-GRPO: surpassing discrete-token LLM reinforcement learning via gumbel-reparameterized soft-thinking policy optimization.

External Links: 2511.06411,
Link

Cited by: Table 14,
Table 14.

Zheng and Lee (2025)
Z. Zheng and W. S. Lee

Reasoning-cv: fine-tuning powerful reasoning llms for knowledge-assisted claim verification.

arXiv preprint arXiv:2505.12348.

Cited by: §2.1.

Appendix A Proofs for the RQ1 Diversity–Coverage Analysis

This appendix provides the auxiliary identities, qualifications, and proofs behind Lemmas 1–3 and Proposition 1.
The Jensen–Shannon (JS) definitions and entropy representation follow the standard formulation of Lin (1991).
We use the usual information-theoretic forms of the data-processing and Pinsker inequalities (Cover and Thomas, 2006); the arithmetic–geometric mean, Taylor, and Jensen inequalities are invoked explicitly at the steps where they are used.

A.1 Proof of Lemma 1

Finite response-distribution entropies imply this identity:

H⁡(π¯N)=1N​∑i=1NH⁡(πi)+JSNpol⁡(x).H(\bar{\pi}_{N})=\frac{1}{N}\sum_{i=1}^{N}H(\pi_{i})+\operatorname{JS}^{\mathrm{pol}}_{N}(x).

(17)

Thus, JSNpol\operatorname{JS}^{\mathrm{pol}}_{N} measures diversity across population members rather than sampling entropy within one policy.
For a GRPO rollout group drawn from one behavior policy, all member policies are identical and the corresponding cross-parameter JS is zero, although within-policy sampling entropy can remain nonzero.

Take NN and the parameter dimension to be fixed and finite, and let 𝒴x\mathcal{Y}_{x} be the countable response space for prompt xx.
Write qΔ​(y)=πθ+Δ​(y∣x)q_{\Delta}(y)=\pi_{\theta+\Delta}(y\mid x), q0=qΔ=0q_{0}=q_{\Delta=0},
sθ=sθ​(y∣x)s_{\theta}=s_{\theta}(y\mid x), Δ¯=N−1​∑iΔi\bar{\Delta}=N^{-1}\sum_{i}\Delta_{i},
q¯=N−1​∑iqΔi\bar{q}=N^{-1}\sum_{i}q_{\Delta_{i}}, and
∥Δ1:N∥=maxi∥Δi∥\lVert\Delta_{1:N}\rVert=\max_{i}\lVert\Delta_{i}\rVert.
For some r>0r>0, assume that qΔq_{\Delta} has common positive support and finite entropy on {∥Δ∥≤r}\{\lVert\Delta\rVert\leq r\}, is four times continuously differentiable in Δ\Delta, and has finite Fisher information at Δ=0\Delta=0.
Define the per-response JS integrand
ℓy(Δ1:N)=N−1∑iqΔi(y)log[qΔi(y)/q¯(y)]\ell_{y}(\Delta_{1:N})=N^{-1}\sum_{i}q_{\Delta_{i}}(y)\log[q_{\Delta_{i}}(y)/\bar{q}(y)].
For every multi-index |ν|≤4|\nu|\leq 4, assume
sup∥Δ1:N∥≤r|∂νℓy(Δ1:N)|≤Mν(y)\sup_{\lVert\Delta_{1:N}\rVert\leq r}|\partial^{\nu}\ell_{y}(\Delta_{1:N})|\leq M_{\nu}(y) for an envelope satisfying
∑y∈𝒴xMν​(y)<∞\sum_{y\in\mathcal{Y}_{x}}M_{\nu}(y)<\infty.
These conditions justify termwise differentiation of the response sum and a uniform fourth-order Taylor remainder.
For local offsets Δ1,…,ΔN\Delta_{1},\ldots,\Delta_{N}, Taylor expansion gives
qΔ=q0​[1+sθ⊤​Δ]+O⁡(∥Δ∥2)q_{\Delta}=q_{0}[1+s_{\theta}^{\top}\Delta]+O(\lVert\Delta\rVert^{2}) and
q¯=q0[1+sθ⊤Δ¯]+O(∥Δ1:N∥2)\bar{q}=q_{0}[1+s_{\theta}^{\top}\bar{\Delta}]+O(\lVert\Delta_{1:N}\rVert^{2}).
Substituting into the KL divergence and using
𝔼q0​[sθ]=0\mathbb{E}_{q_{0}}[s_{\theta}]=0 and
𝔼q0​[sθ​sθ⊤]=ℐx​(θ)\mathbb{E}_{q_{0}}[s_{\theta}s_{\theta}^{\top}]=\mathcal{I}_{x}(\theta) yields the quadratic term below.
Retaining the third- and fourth-order terms gives

JSNpol⁡(x)=12​N​∑i=1N(Δi−Δ¯)⊤​ℐx​(θ)​(Δi−Δ¯)+𝒯3(Δ1:N)+ℛ4(Δ1:N),\begin{gathered}\operatorname{JS}^{\mathrm{pol}}_{N}(x)=\frac{1}{2N}\sum_{i=1}^{N}(\Delta_{i}-\bar{\Delta})^{\top}\mathcal{I}_{x}(\theta)(\Delta_{i}-\bar{\Delta})\\[-0.86108pt]
{}+\mathcal{T}_{3}(\Delta_{1:N})+\mathcal{R}_{4}(\Delta_{1:N}),\end{gathered}

(18)

where 𝒯3\mathcal{T}_{3} is homogeneous of degree three and
|ℛ4(Δ1:N)|≤C∥Δ1:N∥4|\mathcal{R}_{4}(\Delta_{1:N})|\leq C\lVert\Delta_{1:N}\rVert^{4}
whenever ∥Δ1:N∥≤r\lVert\Delta_{1:N}\rVert\leq r.
For Δi=σ​ϵi\Delta_{i}=\sigma\epsilon_{i}, the covariance becomes

𝔼⁡[(Δi−Δ¯)​(Δi−Δ¯)⊤]=σ2​(1−1N)​I.\mathbb{E}\left[(\Delta_{i}-\bar{\Delta})(\Delta_{i}-\bar{\Delta})^{\top}\right]=\sigma^{2}\left(1-\frac{1}{N}\right)I.

(19)

Therefore, the expected quadratic term in Equation (18) is

σ22​(1−1N)​tr⁡ℐx​(θ).\frac{\sigma^{2}}{2}\left(1-\frac{1}{N}\right)\operatorname{tr}\mathcal{I}_{x}(\theta).

(20)

Let Eσ={maxi∥σϵi∥≤r}E_{\sigma}=\{\max_{i}\lVert\sigma\epsilon_{i}\rVert\leq r\}.
It is sign invariant, so
𝔼[𝒯3(σϵ1:N)𝟏Eσ]=0\mathbb{E}[\mathcal{T}_{3}(\sigma\epsilon_{1:N})\mathbf{1}_{E_{\sigma}}]=0.
The remainder on EσE_{\sigma} has expectation O⁡(σ4)O(\sigma^{4}).
Moreover, JSNpol≤log⁡N\operatorname{JS}^{\mathrm{pol}}_{N}\leq\log N and
Pr(Eσc)=O(e−c/σ2)\Pr(E_{\sigma}^{c})=O(e^{-c/\sigma^{2}}), so the JS tail and quadratic truncation are o⁡(σ4)o(\sigma^{4}), establishing the expansion in Equation (9).

A.2 Proof of Lemma 2

The correctness verifier deterministically maps each response distribution πi(⋅∣x)\pi_{i}(\cdot\mid x) to
Bi​(x)=Bernoulli⁡(pi​(x))B_{i}(x)=\operatorname{Bernoulli}(p_{i}(x)).
The data-processing inequality for generalized JS divergence therefore proves Equation (10).
This projection is necessary because full policy JS may arise solely from variation among incorrect responses.

For a fixed prompt xx, suppress the dependence on xx and write pi=pi​(x)p_{i}=p_{i}(x) and p¯=N−1​∑ipi\bar{p}=N^{-1}\sum_{i}p_{i}.
By the arithmetic–geometric mean inequality,

∏i=1N(1−pi)≤(1N​∑i=1N(1−pi))N=(1−p¯)N.\prod_{i=1}^{N}(1-p_{i})\leq\left(\frac{1}{N}\sum_{i=1}^{N}(1-p_{i})\right)^{N}=(1-\bar{p})^{N}.

(21)

Subtracting both sides from one proves
PNpop≥PNsameP_{N}^{\mathrm{pop}}\geq P_{N}^{\mathrm{same}}.
Equality in the arithmetic–geometric mean inequality holds exactly when
1−p1=⋯=1−pN1-p_{1}=\cdots=1-p_{N}, equivalently p1=⋯=pNp_{1}=\cdots=p_{N}.

For the local relation, let pi=p¯+ξip_{i}=\bar{p}+\xi_{i}, ∑iξi=0\sum_{i}\xi_{i}=0, and β=1−p¯\beta=1-\bar{p}, where ξi\xi_{i} is the centered member-success deviation and β\beta the average failure probability.
Assume that p¯\bar{p} remains in a compact subset of (0,1)(0,1).
At ξ=0\xi=0, expand the product:

∏i=1N(β−ξi)=βN−βN−22​∑i=1Nξi2+O⁡(∥ξ∥3),\prod_{i=1}^{N}(\beta-\xi_{i})=\beta^{N}-\frac{\beta^{N-2}}{2}\sum_{i=1}^{N}\xi_{i}^{2}+O(\lVert\xi\rVert^{3}),

(22)

Subtracting from one gives the coverage difference:

PNpop−PNsame=βN−22​∑i=1Nξi2+O⁡(∥ξ∥3).P_{N}^{\mathrm{pop}}-P_{N}^{\mathrm{same}}=\frac{\beta^{N-2}}{2}\sum_{i=1}^{N}\xi_{i}^{2}+O(\lVert\xi\rVert^{3}).

(23)

For h⁡(p)=−p​log⁡p−(1−p)​log⁡(1−p)h(p)=-p\log p-(1-p)\log(1-p), the binary entropy satisfies h′′(p)=−1/[p(1−p)]h^{\prime\prime}(p)=-1/[p(1-p)].
A Taylor expansion of the success-JS definition in Lemma 2, with the linear term canceled by ∑iξi=0\sum_{i}\xi_{i}=0, yields

JSNsucc=12​N​p¯​(1−p¯)​∑i=1Nξi2+O⁡(∥ξ∥3).\operatorname{JS}^{\mathrm{succ}}_{N}=\frac{1}{2N\bar{p}(1-\bar{p})}\sum_{i=1}^{N}\xi_{i}^{2}+O(\lVert\xi\rVert^{3}).

(24)

Eliminating ∑iξi2\sum_{i}\xi_{i}^{2} between Equations (23) and (24) gives

PNpop​(x)−PNsame​(x)=N​p¯​(x)​(1−p¯​(x))N−1​JSNsucc⁡(x)+O⁡(∥ξ⁡(x)∥3).P_{N}^{\mathrm{pop}}(x)-P_{N}^{\mathrm{same}}(x)=N\bar{p}(x)(1-\bar{p}(x))^{N-1}\operatorname{JS}^{\mathrm{succ}}_{N}(x){}\,+O(\lVert\xi(x)\rVert^{3}).

(25)

A.3 Proof and Illustration of Lemma 3

Under a uniform draw of member index ii, 𝔼i​[wi]=1/N\mathbb{E}_{i}[w_{i}]=1/N and 𝔼i​[pi]=p¯\mathbb{E}_{i}[p_{i}]=\bar{p}. Hence

N​Covi⁡(wi,pi)=∑i=1Nwi​pi−p¯=pw−p¯,N\operatorname{Cov}_{i}(w_{i},p_{i})=\sum_{i=1}^{N}w_{i}p_{i}-\bar{p}=p_{w}-\bar{p},

(26)

which proves the covariance identity in Lemma 3.

For intuition only, define the prompt-adaptive oracle gate

w~i​(x)=exp⁡(λ​pi​(x))∑j=1Nexp⁡(λ​pj​(x)),p~w​(x)=∑i=1Nw~i​(x)​pi​(x),\begin{gathered}\widetilde{w}_{i}(x)=\frac{\exp(\lambda p_{i}(x))}{\sum_{j=1}^{N}\exp(\lambda p_{j}(x))},\\
\widetilde{p}_{w}(x)=\sum_{i=1}^{N}\widetilde{w}_{i}(x)p_{i}(x),\end{gathered}

(27)

where λ>0\lambda>0.
For fixed N≥2N\geq 2, let pi=p¯+ξip_{i}=\bar{p}+\xi_{i} and assume that p¯​(x)\bar{p}(x) is bounded away from zero and one.
This parameterization gives the following normalized weights:

w~i=exp⁡(λ​ξi)∑j=1Nexp⁡(λ​ξj).\widetilde{w}_{i}=\frac{\exp(\lambda\xi_{i})}{\sum_{j=1}^{N}\exp(\lambda\xi_{j})}.

(28)

Since ∑iξi=0\sum_{i}\xi_{i}=0, the ratio expands as

w~i=1N+λN​ξi+O⁡(∥ξ∥2).\widetilde{w}_{i}=\frac{1}{N}+\frac{\lambda}{N}\xi_{i}+O(\lVert\xi\rVert^{2}).

(29)

This expansion gives the selected-policy shift:

p~w−p¯=∑i=1Nw~i​ξi=λN​∑i=1Nξi2+O⁡(∥ξ∥3).\widetilde{p}_{w}-\bar{p}=\sum_{i=1}^{N}\widetilde{w}_{i}\xi_{i}=\frac{\lambda}{N}\sum_{i=1}^{N}\xi_{i}^{2}+O(\lVert\xi\rVert^{3}).

(30)

Substituting Equation (24) into Equation (30) gives

p~w​(x)−p¯​(x)=2​λ​p¯​(x)​(1−p¯​(x))​JSNsucc⁡(x)+O⁡(∥ξ⁡(x)∥3).\widetilde{p}_{w}(x)-\bar{p}(x)=2\lambda\bar{p}(x)(1-\bar{p}(x))\operatorname{JS}^{\mathrm{succ}}_{N}(x)+O(\lVert\xi(x)\rVert^{3}).

(31)

This calculation illustrates how success/failure predictive JS becomes exploitable when fitness is calibrated to success.

A.4 Proof of Proposition 1

Data processing yields the relaxed full-policy KL bound:

DKL(Bw∥Bθ+)≤DKL(πw∥πθ+).D_{\mathrm{KL}}(B_{w}\,\|\,B_{\theta^{+}})\leq D_{\mathrm{KL}}(\pi_{w}\,\|\,\pi_{\theta^{+}}).

(32)

It allows the center policy to use different responses while preserving total verifier success probability.
This empirically testable condition connects the analytical comparator to the actual ES center.

Define the prompt-wise transfer divergence as

Dx=DKL(Bw(x)∥Bθ+(x)).D_{x}=D_{\mathrm{KL}}\left(B_{w}(x)\,\|\,B_{\theta^{+}}(x)\right).

(33)

For Bernoulli distributions, their total variation distance is the absolute difference between their success probabilities.
Pinsker’s inequality bounds this difference:

|pw​(x)−pθ+​(x)|≤Dx2.\left|p_{w}(x)-p_{\theta^{+}}(x)\right|\leq\sqrt{\frac{D_{x}}{2}}.

(34)

For the KK-sample success map fK​(p)=1−(1−p)Kf_{K}(p)=1-(1-p)^{K}, the derivative is bounded as follows:

0≤fK′​(p)=K​(1−p)K−1≤Kfor ​p∈[0,1],0\leq f_{K}^{\prime}(p)=K(1-p)^{K-1}\leq K\qquad\text{for }p\in[0,1],

(35)

Thus, fKf_{K} is KK-Lipschitz.
From Equation (34), Lipschitz continuity implies

fK​(pθ+​(x))≥fK​(pw​(x))−K​Dx2.f_{K}(p_{\theta^{+}}(x))\geq f_{K}(p_{w}(x))-K\sqrt{\frac{D_{x}}{2}}.

(36)

Taking expectation over xx and applying Jensen’s inequality to the concave square-root function gives

JK​(πθ+)≥JK​(πw)−K​𝔼x∼𝒟​Dx2\displaystyle J_{K}(\pi_{\theta^{+}})\geq J_{K}(\pi_{w})-K\mathbb{E}_{x\sim\mathcal{D}}\sqrt{\frac{D_{x}}{2}}

(37)

JK​(πθ+)≥JK​(πw)−K​𝔼x∼𝒟​[Dx]2\displaystyle J_{K}(\pi_{\theta^{+}})\geq J_{K}(\pi_{w})-K\sqrt{\frac{\mathbb{E}_{x\sim\mathcal{D}}[D_{x}]}{2}}

(38)

JK​(πθ+)≥JK​(πw)−K​εsucc2,\displaystyle J_{K}(\pi_{\theta^{+}})\geq J_{K}(\pi_{w})-K\sqrt{\frac{\varepsilon_{\mathrm{succ}}}{2}},

(39)

where the final inequality uses Equation (12).
Combining this bound with the margin condition in Equation (13) proves the chained conclusion in Equation (14).
\FloatBarrier

Appendix B Complete Majority-Vote Results

This appendix reports the Maj@16 and Maj@32 results corresponding to the
Pass@KK results in Tables 2,
3, and 5.
Maj@KK applies deterministic plurality voting after task-specific answer
normalization, as defined in Appendix G.7.

GSM8K
CSQA
HotpotQA
Countdown
GPQA
MBPP
Average

Method
M@16
M@32
M@16
M@32
M@16
M@32
M@16
M@32
M@16
M@32
M@16
M@32
M@16
M@32

Qwen2.5-1.5B-Instruct

Base
82.6
84.3
67.8
68.8
31.1
31.1
21.7
27.0
20.7
20.2
61.9
61.5
47.6
48.8

GRPO
84.2
84.8
70.0
71.0
28.8
29.0
27.8
31.5
22.7
22.7
60.3
61.9
49.0
50.2

ES
82.3
83.9
67.2
68.9
29.8
29.8
27.6
33.1
25.3
24.2
58.4
62.3
48.4
50.4

ES→\rightarrowGRPO
83.9
84.2
69.2
69.6
28.8
29.1
26.2
31.6
20.7
18.7
61.5
62.7
48.4
49.3

GRPO→\rightarrowES
83.1
83.6
68.7
70.0
29.2
29.4
20.7
24.6
23.7
23.7
58.8
61.9
47.4
48.9

Llama-3.2-3B-Instruct

Base
86.9
87.8
74.1
74.6
28.1
28.5
39.2
45.7
33.8
33.8
62.3
64.6
54.1
55.8

GRPO
88.9
89.5
74.3
74.6
27.4
27.7
44.6
50.8
35.4
33.8
62.3
63.0
55.5
56.6

ES
88.6
88.9
74.9
75.5
29.5
30.0
40.4
48.9
30.3
30.8
62.3
61.9
54.4
56.0

ES→\rightarrowGRPO
88.4
88.2
75.9
75.8
26.6
26.8
41.9
48.7
29.8
29.3
63.8
62.3
54.4
55.2

GRPO→\rightarrowES
89.1
89.8
74.9
75.6
28.5
28.6
40.0
47.5
30.8
31.3
63.0
62.7
54.4
55.9

Qwen2.5-7B-Instruct

Base
94.5
94.5
81.4
81.7
45.9
46.0
56.6
58.4
37.4
38.4
77.0
77.0
65.5
66.0

GRPO
93.8
93.6
81.9
81.4
46.8
46.9
56.6
57.4
34.9
37.9
75.9
75.9
65.0
65.5

ES
94.2
94.2
80.8
81.2
45.7
45.7
57.9
60.1
40.4
41.9
75.1
75.9
65.7
66.5

ES→\rightarrowGRPO
94.3
94.6
81.4
81.4
46.5
46.5
57.3
59.2
37.4
37.9
75.1
74.7
65.3
65.7

GRPO→\rightarrowES
93.7
94.0
82.2
81.7
46.2
46.3
56.1
57.2
40.4
39.4
75.1
75.9
65.6
65.8

Table 7: Maj@KK results (×100\times 100) in the Easy Setting after two epochs of GSM8K post-training, corresponding to the Pass@KK results in Table 2. Underlining marks the best Base/GRPO/ES result in each task cell; bold and shading mark the best and second-best Average cells across all five methods.

AIME24  
AIME25  
AMC23  
MATH500  
Average  

Method
M@16
M@32
M@16
M@32
M@16
M@32
M@16
M@32
M@16
M@32

DeepSeek-R1-Distill-Qwen-1.5B  

Base
36.7
40.0
33.3
33.3
82.5
82.5
90.0
91.0
60.6
61.7

GRPO
43.3
46.7
30.0
33.3
85.0
87.5
91.2
91.8
62.4
64.8

ES
36.7
43.3
36.7
33.3
90.0
92.5
91.8
92.2
63.8
65.3

ES→\rightarrowGRPO
46.7
53.3
33.3
33.3
85.0
92.5
91.6
92.6
64.2
67.9

GRPO→\rightarrowES
46.7
46.7
40.0
36.7
90.0
85.0
91.4
92.2
67.0
65.1

Table 8: Maj@KK results (×100\times 100) on mathematical benchmarks in the Hard Setting, corresponding to the Pass@KK results in Table 3. Underlining marks the best Base/GRPO/ES result in each task cell; bold and shading mark the best and second-best Average cells across all five methods.

GPQA
MBPP
CSQA
Countdown
Average

Method
M@16
M@32
M@16
M@32
M@16
M@32
M@16
M@32
M@16
M@32

DeepSeek-R1-Distill-Qwen-1.5B

Base
33.3
33.3
69.7
70.4
54.1
54.1
71.1
75.1
57.0
58.2

GRPO
35.9
35.4
69.7
70.8
53.7
52.7
68.5
72.8
56.9
57.9

ES
39.4
39.4
70.8
69.7
52.7
53.6
71.7
76.2
58.6
59.7

ES→\rightarrowGRPO
36.4
37.4
71.6
72.0
52.4
53.0
70.9
73.6
57.8
59.0

GRPO→\rightarrowES
33.8
34.9
66.5
69.7
53.3
54.1
72.2
76.9
56.5
58.9

Table 9: Maj@KK results (×100\times 100) on four held-out benchmarks in the one-epoch Hard Setting, corresponding to the Pass@KK results in Table 5. Underlining marks the best Base/GRPO/ES result in each task cell; bold and shading mark the best and second-best Average cells across all five methods.

\FloatBarrier

Appendix C Detailed Held-Out Performance Changes

Table 10 reports the metric-wise Easy Setting results underlying the held-out summary in Figure 1(b).
For each model and metric, we compute post-training minus initial performance on each held-out benchmark and then average these changes equally over the five non-GSM8K tasks.

Pass changes
Majority-vote changes

Method
Δ\Delta@1
Δ\Delta@16
Δ\Delta@32
Δ\DeltaM@16
Δ\DeltaM@32

Qwen2.5-1.5B-Instruct

GRPO
+0.7+0.7
−0.2-0.2
−0.1-0.1
+1.3+1.3
+1.5+1.5

ES
+0.0+0.0
+0.6+0.6
+0.8+0.8
+1.0+1.0
+2.0+2.0

Llama-3.2-3B-Instruct

GRPO
+2.3+2.3
−1.3-1.3
−1.6-1.6
+1.3+1.3
+0.6+0.6

ES
+1.4+1.4
+2.2+2.2
+2.2+2.2
−0.0-0.0
−0.0-0.0

Qwen2.5-7B-Instruct

GRPO
+0.9+0.9
−1.4-1.4
−1.6-1.6
−0.5-0.5
−0.4-0.4

ES
−0.5-0.5
+0.3+0.3
+0.2+0.2
+0.3+0.3
+0.7+0.7

Table 10: Held-out performance changes for matched Easy Setting runs, averaged equally over five non-GSM8K tasks and reported in percentage points. M@16 and M@32 denote Maj@16 and Maj@32. Positive values denote gains.

Averaged across the three models, all five ES metric changes are positive, whereas GRPO decreases Pass@16 and Pass@32.

\FloatBarrier

Appendix D Complete Results for Magnitude-Thresholded ES

This appendix reports the complete target-task evaluation results for
magnitude-thresholded ES across all four models. For a threshold τ\tau,
coordinates satisfying
0<|Δ​θi|≤τ0<|\Delta\theta_{i}|\leq\tau are set to zero, equivalently resetting the corresponding parameters to their Base Model values, and
update sparsity is computed over nonzero ES changes according to
Equation (16). Coordinates with zero change are
excluded from both the sparsity denominator and the ablated set. Pass@1 is the empirical single-sample accuracy
estimated from 32 retained responses per problem. The Δ\Delta Full ES column
reports the signed absolute difference from the corresponding unablated ES
endpoint on the same percentage scale.

Table 11: Complete GSM8K Pass@1 results for magnitude-thresholded ES in the Easy Setting. Base Model values are included as absolute references.

Threshold τ\tau
Update sparsity (%)
Pass@1 (%)
Δ\Delta Full ES (%)

Qwen2.5-1.5B-Instruct

Full ES
0.00
73.351
+0.000+0.000

2.5×10−42.5\times 10^{-4}
27.82
73.330
−0.021-0.021

5.0×10−45.0\times 10^{-4}
50.77
73.102
−0.249-0.249

1.0×10−31.0\times 10^{-3}
79.11
73.000
−0.351-0.351

1.5×10−31.5\times 10^{-3}
92.47
72.091
−1.260-1.260

2.0×10−32.0\times 10^{-3}
97.57
70.681
−2.670-2.670

Base Model
100.00
70.271
–

Llama-3.2-3B-Instruct

Full ES
0.00
81.691
+0.000+0.000

2.5×10−42.5\times 10^{-4}
25.27
81.658
−0.033-0.033

5.0×10−45.0\times 10^{-4}
47.74
81.440
−0.251-0.251

1.0×10−31.0\times 10^{-3}
78.08
81.560
−0.130-0.130

1.5×10−31.5\times 10^{-3}
92.64
78.433
−3.258-3.258

2.0×10−32.0\times 10^{-3}
97.89
77.883
−3.807-3.807

Base Model
100.00
77.670
–

Qwen2.5-7B-Instruct

Full ES
0.00
91.016
+0.000+0.000

2.5×10−42.5\times 10^{-4}
25.43
91.087
+0.071+0.071

5.0×10−45.0\times 10^{-4}
47.51
91.028
+0.012+0.012

1.0×10−31.0\times 10^{-3}
78.41
91.504
+0.488+0.488

1.5×10−31.5\times 10^{-3}
93.02
91.736
+0.720+0.720

2.0×10−32.0\times 10^{-3}
98.11
91.824
+0.808+0.808

Base Model
100.00
91.774
–

Table 12: Complete MATH-500 Pass@1 results for magnitude-thresholded ES with DeepSeek-R1-Distill-Qwen-1.5B in the Hard Setting. The Base Model value is included as an absolute reference.

Threshold τ\tau
Update sparsity (%)
Pass@1 (%)
Δ\Delta Full ES (%)

Full ES
0.00
82.844
+0.000+0.000

2.5×10−42.5\times 10^{-4}
17.80
83.112
+0.269+0.269

5.0×10−45.0\times 10^{-4}
38.32
82.919
+0.075+0.075

1.0×10−31.0\times 10^{-3}
62.18
83.075
+0.231+0.231

1.5×10−31.5\times 10^{-3}
77.61
83.013
+0.169+0.169

2.0×10−32.0\times 10^{-3}
87.29
82.394
−0.450-0.450

2.5×10−32.5\times 10^{-3}
93.16
82.213
−0.631-0.631

3.0×10−33.0\times 10^{-3}
96.64
81.388
−1.456-1.456

Base Model
100.00
80.856
–

At comparable sparsity near 78%, all four endpoints remain close to Full ES:
the Pass@1 changes are −0.351-0.351, −0.130-0.130, +0.488+0.488, and +0.169+0.169
percentage points for Qwen2.5-1.5B-Instruct, Llama-3.2-3B-Instruct,
Qwen2.5-7B-Instruct, and DeepSeek-R1-Distill-Qwen-1.5B, respectively.
Their behavior diverges as the threshold expands farther into the update
distribution. Qwen2.5-1.5B-Instruct and Llama-3.2-3B-Instruct degrade at
approximately 92% sparsity, DeepSeek degrades more gradually, and Qwen2.5-7B
improves across the tested range.

\FloatBarrier

Appendix E ES–ZO Estimator Equivalence and Its Finite-Sample Boundary

This appendix separates an ideal estimator identity from a finite-sample property of stochastic reasoning evaluations.
The identity shows that ES and ZO can estimate the same smoothed objective; the diagnostic tests whether paired evaluations retain enough covariance to reduce variance.
The Gaussian smoothing and score-function construction below are standard tools in gradient-free optimization and ES (Nesterov and Spokoiny, 2017; Salimans et al., 2017).

E.1 A Unified Smoothed-Objective View

For a measurable scalar objective FF, we use a maximization convention throughout.
For reward maximization, we take F⁡(θ)=R⁡(θ)F(\theta)=R(\theta); for loss minimization, we take
F⁡(θ)=−ℒ⁡(θ)F(\theta)=-\mathcal{L}(\theta), so that all estimators below are written in ascent form.
Assume that some neighborhood UU of θ\theta satisfies
supθ′∈U𝔼ϵ​[|F⁡(θ′+σ​ϵ)|​(1+∥ϵ∥)]<∞\sup_{\theta^{\prime}\in U}\mathbb{E}_{\epsilon}[|F(\theta^{\prime}+\sigma\epsilon)|(1+\lVert\epsilon\rVert)]<\infty,
and define its Gaussian smoothing as

Fσ​(θ)=𝔼ϵ∼𝒩⁡(0,I)​[F⁡(θ+σ​ϵ)].F_{\sigma}(\theta)=\mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)}\left[F(\theta+\sigma\epsilon)\right].

(40)

The Gaussian score-function identity gives

∇Fσ​(θ)=1σ​𝔼ϵ​[F⁡(θ+σ​ϵ)​ϵ].\nabla F_{\sigma}(\theta)=\frac{1}{\sigma}\mathbb{E}_{\epsilon}\left[F(\theta+\sigma\epsilon)\epsilon\right].

(41)

The resulting one-point estimator over NN directions is

g^1​p=1N​σ​∑i=1NF⁡(θ+σ​ϵi)​ϵi,𝔼⁡[g^1​p]=∇Fσ​(θ).\widehat{g}^{1p}=\frac{1}{N\sigma}\sum_{i=1}^{N}F(\theta+\sigma\epsilon_{i})\epsilon_{i},\qquad\mathbb{E}\!\left[\widehat{g}^{1p}\right]=\nabla F_{\sigma}(\theta).

(42)

The multi-direction two-point estimator is

g^2​p=12​N​σ​∑i=1N[F⁡(θ+σ​ϵi)−F⁡(θ−σ​ϵi)]​ϵi.\widehat{g}^{2p}=\frac{1}{2N\sigma}\sum_{i=1}^{N}\left[F(\theta+\sigma\epsilon_{i})-F(\theta-\sigma\epsilon_{i})\right]\epsilon_{i}.

(43)

Because ϵ\epsilon and −ϵ-\epsilon have the same distribution,

𝔼ϵ​[F⁡(θ−σ​ϵ)​ϵ]=−𝔼ϵ​[F⁡(θ+σ​ϵ)​ϵ].\mathbb{E}_{\epsilon}\left[F(\theta-\sigma\epsilon)\epsilon\right]=-\mathbb{E}_{\epsilon}\left[F(\theta+\sigma\epsilon)\epsilon\right].

(44)

Substitution into Equation (43) yields

𝔼⁡[g^2​p]=∇Fσ​(θ)=𝔼⁡[g^1​p].\mathbb{E}\!\left[\widehat{g}^{2p}\right]=\nabla F_{\sigma}(\theta)=\mathbb{E}\!\left[\widehat{g}^{1p}\right].

(45)

If ES uses the same positive–negative perturbation pairs, its antithetic estimator is exactly Equation (43).
Thus, antithetic ES and two-point ZO are the same estimator when they share the objective, perturbation distribution, scale, and raw paired-difference rule.
This identity does not imply equal finite-budget variance because one pair consumes two function evaluations, whereas one one-point sample consumes one.
Thus, NN counts perturbation directions in both estimators, while their perturbed-evaluation counts are Neval=NN_{\mathrm{eval}}=N and Neval=2​NN_{\mathrm{eval}}=2N, respectively.

(a) Objective comparison at σ=10−3\sigma=10^{-3}

Objective
Evaluation coupling
Corr⁡(X+,X0)\operatorname{Corr}(X_{+},X_{0})
κbase\kappa_{\mathrm{base}}
Corr⁡(X+,X−)\operatorname{Corr}(X_{+},X_{-})
κpair\kappa_{\mathrm{pair}}

GSM8K regenerated-rollout reward
Common seed
0.2777
1.4445
0.1304
1.9870

GSM8K regenerated-rollout reward
Independent seed
0.0561
1.3577
0.0224
2.7847

SST-2 supervised CE
Fixed batch
0.9776
0.0447
0.9196
0.1557

Mean token log-probability
Fixed rollout
0.9939
0.0125
0.9857
0.0285

Policy sequence loss
Fixed rollout
0.9939
0.0167
0.9903
0.0194

GRPO surrogate
Fixed rollout
0.9998
0.0006
0.9997
0.0007

(b) Perturbation-scale sweep for GSM8K regenerated-rollout reward

σ\sigma
Seed coupling
Corr⁡(X+,X0)\operatorname{Corr}(X_{+},X_{0})
κbase\kappa_{\mathrm{base}}
Corr⁡(X+,X−)\operatorname{Corr}(X_{+},X_{-})
κpair\kappa_{\mathrm{pair}}

10−410^{-4}
Common
0.8797
0.2288
0.5537
0.8194

10−410^{-4}
Independent
-0.0245
1.4080
0.0450
2.3822

3×10−43\times 10^{-4}
Common
0.5607
1.0424
0.3026
1.9434

3×10−43\times 10^{-4}
Independent
-0.0303
1.6687
0.0532
2.4379

10−310^{-3}
Common
0.2777
1.4445
0.1304
1.9870

10−310^{-3}
Independent
0.0561
1.3577
0.0224
2.7847

Table 13: Local subtraction diagnostic at the Qwen2.5-1.5B-Instruct base checkpoint. κbase\kappa_{\mathrm{base}} and κpair\kappa_{\mathrm{pair}} are the raw scalar variance ratios in Equation (51); lower is better and values below 11 indicate variance reduction. Panel (a) compares objectives at σ=10−3\sigma=10^{-3}, and Panel (b) sweeps the scale for GSM8K regenerated-rollout reward.

E.2 Covariance Required by Paired Subtraction

Suppose an objective evaluation is stochastic because it includes prompt sampling, autoregressive generation, or verifier noise.
Let ξ\xi collect this evaluation randomness and let F^​(ϑ,ξ)\widehat{F}(\vartheta;\xi) be an unbiased scalar evaluation of the objective, such that

𝔼ξ​[F^​(ϑ,ξ)]=F⁡(ϑ)\mathbb{E}_{\xi}\!\left[\widehat{F}(\vartheta;\xi)\right]=F(\vartheta)

(46)

for every evaluated parameter point ϑ\vartheta.
For fixed (θ,ϵ,σ)(\theta,\epsilon,\sigma), the repeated evaluations are

X+=F^​(θ+σ​ϵ,ξ+),X0=F^​(θ,ξ0),X−=F^​(θ−σ​ϵ,ξ−).\begin{gathered}X_{+}=\widehat{F}(\theta+\sigma\epsilon;\xi_{+}),\\
X_{0}=\widehat{F}(\theta;\xi_{0}),\\
X_{-}=\widehat{F}(\theta-\sigma\epsilon;\xi_{-}).\end{gathered}

(47)

The random variables (ξ+,ξ0,ξ−)(\xi_{+},\xi_{0},\xi_{-}) may be coupled through common random numbers or sampled independently.
Their raw difference variances satisfy

Var⁡(X+−X0)=Var⁡(X+)+Var⁡(X0)−2​Cov⁡(X+,X0),\displaystyle\operatorname{Var}(X_{+}-X_{0})=\operatorname{Var}(X_{+})+\operatorname{Var}(X_{0})-2\operatorname{Cov}(X_{+},X_{0}),

(48)

Var⁡(X+−X−)=Var⁡(X+)+Var⁡(X−)−2​Cov⁡(X+,X−).\displaystyle\operatorname{Var}(X_{+}-X_{-})=\operatorname{Var}(X_{+})+\operatorname{Var}(X_{-})-2\operatorname{Cov}(X_{+},X_{-}).

(49)

In particular, Var⁡(X+−X−)<Var⁡(X+)\operatorname{Var}(X_{+}-X_{-})<\operatorname{Var}(X_{+}) if and only if

Cov⁡(X+,X−)>12​Var⁡(X−).\operatorname{Cov}(X_{+},X_{-})>\frac{1}{2}\operatorname{Var}(X_{-}).

(50)

We report the correlations and the raw variance ratios

κbase=Var⁡(X+−X0)Var⁡(X+),κpair=Var⁡(X+−X−)Var⁡(X+).\kappa_{\mathrm{base}}=\frac{\operatorname{Var}(X_{+}-X_{0})}{\operatorname{Var}(X_{+})},\qquad\kappa_{\mathrm{pair}}=\frac{\operatorname{Var}(X_{+}-X_{-})}{\operatorname{Var}(X_{+})}.

(51)

A ratio below one means that subtraction reduces the variance of the scalar objective values in this diagnostic.

E.3 Local Perturbation Diagnostic

Protocol.

We conduct a local diagnostic around the Qwen2.5-1.5B-Instruct base checkpoint.
For each sampled full-parameter direction, we evaluate Equation (47) at several perturbation scales.
The reported statistics pool scalar observations across prompts, generation seeds, and directions.

Objectives.

We compare three evaluation protocols.
First, GSM8K regenerated-rollout reward regenerates a response independently at each of the positive, center, and negative parameter points and assigns a terminal correctness reward.
Second, SST-2 supervised CE evaluates a differentiable cross-entropy objective on the same fixed batch at all three parameter points.
Third, GSM8K fixed rollout generates responses once at the base checkpoint, freezes them, and recomputes mean token log-probability, policy sequence loss, and a GRPO surrogate at the perturbed points.
Fixed rollouts isolate rescoring from autoregressive generation noise.

Randomness Coupling.

For regenerated rollouts, common uses the same generation seed at the three parameter points, whereas independent uses different seeds.
Common seeds implement common random numbers, but they do not guarantee identical autoregressive trajectories after a parameter perturbation changes an early token distribution.

E.4 Diagnostic Results

Table 13(a) compares the three objective classes at σ=10−3\sigma=10^{-3}.
The repeated-direction regenerated-rollout run uses 1616 prompts, 1616 generation seeds per prompt, and 44 perturbation directions.
The fixed-rollout run uses 1616 prompts, 88 seeds per prompt, and 44 directions.
The supervised control contains 128128 pooled scalar observations.

GSM8K regenerated-rollout rewards are weakly correlated across the positive and negative perturbations at this scale, and paired subtraction increases their raw scalar variance.
The same perturbation implementation yields strong correlations and ratios below one for supervised CE and all fixed-rollout objectives.
This contrast localizes the observed instability to the combination of regenerating autoregressive responses and evaluating terminal rewards rather than to paired parameter perturbations alone.
The very small fixed-rollout GRPO ratio further shows that a GRPO surrogate is not intrinsically incompatible with paired ZO evaluation when the responses are held fixed.

Table 13(b) shows that coupling regenerated-rollout reward evaluations with common random numbers is scale dependent.
At σ=10−4\sigma=10^{-4}, common seeds produce κpair=0.8194\kappa_{\mathrm{pair}}=0.8194, whereas independent seeds remain above one.
At the two larger scales, paired subtraction does not reduce raw variance under either seed mode.

\FloatBarrier

E.5 Scope of the Evidence

The diagnostic supports a conclusion: near the tested base checkpoint, regenerated-rollout correctness rewards provide substantially weaker positive–negative coupling than fixed-batch supervised or fixed-rollout objectives, so raw paired subtraction does not reliably cancel evaluation variance except at small scales.

Figure 1(c), item (3), complements the local diagnostic with a three-epoch comparison under matched perturbed evaluations and reward normalization.

Appendix F Population-Size Scaling Trajectories

Figure 5 reports the complete per-update mean-reward trajectories summarized by Table 6.
The colored curves use debiased exponential smoothing with weight 0.990.99.
Distinct colors and line styles identify N∈{8,16,32,64}N\in\{8,16,32,64\}.
The same-color shaded region spans one local standard deviation of the raw-minus-smoothed residuals.
This band visualizes within-trajectory fluctuation.

(a) Qwen2.5-0.5B-Instruct

(b) Qwen2.5-1.5B-Instruct

(c) Qwen2.5-3B-Instruct

Figure 5: Full GSM8K population-size trajectories across three model scales. Lines show debiased exponential smoothing with weight 0.990.99. Same-color shading shows the local standard deviation of raw-minus-smoothed residuals; it describes within-trajectory fluctuation. Model-specific vertical ranges expose within-model differences.

For Qwen2.5-0.5B-Instruct, stable late-stage improvement requires a larger population: N=8N=8 and N=16N=16 decline over later updates, whereas N=32N=32 and N=64N=64 continue improving.
For Qwen2.5-1.5B-Instruct and Qwen2.5-3B-Instruct, the N=16N=16, N=32N=32, and N=64N=64 curves remain substantially closer throughout training.

\FloatBarrier

Appendix G Experimental Details

This appendix records the settings used to produce the reported results.
We separate training hyperparameters, evaluation settings, prompt templates, and dataset provenance.
The locally trained ES/GRPO runs use full-parameter updates.
The training and testing hardware used NVIDIA A100-SXM4-80GB GPUs.

G.1 Implementation-Level Training Procedure

Full-parameter one-point ES.

The following procedure describes the implementation used for all locally trained ES models.
It is the practical one-point, reward-standardized update studied in the paper.
The Easy and Hard settings use the same update procedure and differ only in the model, dataset, epoch budget, and generation settings listed in Table 14.

Implementation Procedure: Full-parameter one-point ES training

1.

Initialize the center model θ0\theta_{0}, shuffle the training split, and construct one keep-tail mini-batch schedule per epoch.

2.

At update tt, draw NN independent 32-bit perturbation seeds. A seed and the stable name of each parameter tensor deterministically reconstruct an independent standard-Gaussian tensor, yielding a full-model direction ϵi\epsilon_{i}.

3.

Distribute the NN directions over eight single-GPU vLLM engines. For direction ii, add σ​ϵi\sigma\epsilon_{i} to every model parameter in place, generate one response for every prompt in the shared mini-batch, compute the task-verifier rewards, average them into RiR_{i}, and then subtract the same seeded perturbation to restore the center.

4.

Standardize the population rewards with

zi=Ri−R¯N−1​∑j=1N(Rj−R¯)2+εnum.z_{i}=\frac{R_{i}-\bar{R}}{\sqrt{N^{-1}\sum_{j=1}^{N}(R_{j}-\bar{R})^{2}}+\varepsilon_{\mathrm{num}}}.

5.

Reconstruct the same directions and update every engine locally by

θt+1=θt+αN​∑i=1Nzi​ϵi.\theta_{t+1}=\theta_{t}+\frac{\alpha}{N}\sum_{i=1}^{N}z_{i}\epsilon_{i}.

Here α\alpha is the center-update scale in Table 14.

The response-level ES training reward is computed by the task-specific verifier.
For the reported mathematical runs, a verifier-correct response receives reward 11; a response that follows the required final-answer format but is incorrect receives the configured format reward 0.10.1; and an answer that fails the required format receives 00.

G.2 ES–GRPO FLOP Matching

We match the estimated model FLOPs of ES and GRPO following the standard accounting used by Gan and Isola (2026).
For a model with PP parameters and a response length of LL tokens, one forward pass requires approximately 2​P​L2PL FLOPs and one backward pass requires approximately 4​P​L4PL FLOPs.
A GRPO response requires a policy forward, a reference-policy forward, and a policy backward, giving

FLOPsGRPO=8​TGRPO​B​G​P​L,\operatorname{FLOPs}_{\mathrm{GRPO}}=8T_{\mathrm{GRPO}}BGPL,

(52)

where TGRPOT_{\mathrm{GRPO}} is the number of updates, BB is the prompt batch size, and GG is the number of responses per prompt.
ES evaluates each population member using only a forward pass, giving

FLOPsES=2​TES​N​D​P​L,\operatorname{FLOPs}_{\mathrm{ES}}=2T_{\mathrm{ES}}NDPL,

(53)

where NN is the population size and DD is the number of prompts evaluated per direction.
With matched update counts TES=TGRPOT_{\mathrm{ES}}=T_{\mathrm{GRPO}}, matched prompt batches D=BD=B, and N=32=4​GN=32=4G for G=8G=8, Equations (52) and (53) are equal.
Thus, the reported ES and GRPO configurations are FLOP-matched under this model-compute accounting.

G.3 Training Hyperparameters

Table 14 summarizes the training configurations used for the Easy and Hard ES/GRPO comparisons.

Parameter

Easy ES

Easy GRPO

Hard ES

Hard GRPO

Model

 

Qwen2.5-1.5B-Instruct,

Llama-3.2-3B-Instruct,

Qwen2.5-7B-Instruct

Same as Easy ES

 

DeepSeek-R1-Distill-

Qwen-1.5B

Same as Hard ES

Training data

GSM8K

GSM8K

DeepScaleR

DeepScaleR

Epochs

2

2

1

1

Global train batch

64

64

64

64

Population / rollout group

32 directions

8 responses

32 directions

8 responses

Learning scale

σ=1.5×10−3\sigma=1.5{\times}10^{-3}, α=2.5×10−4\alpha=2.5{\times}10^{-4}

1.0×10−61.0{\times}10^{-6} learning rate

σ=1.5×10−3\sigma=1.5{\times}10^{-3}, α=2.5×10−4\alpha=2.5{\times}10^{-4}

1.0×10−61.0{\times}10^{-6} learning rate

Train-time sampling

τ=0\tau=0

τ=1.0\tau=1.0

τ=0.6\tau=0.6

τ=1.0\tau=1.0

Response limit

2,048

2,048

8,192

8,192

Reward normalization

Population zz-score

Group-relative advantages

Population zz-score

Group-relative standardized advantages

GRPO update details

–

 

PPO minibatch 32; microbatch 2;

clip 0.2; KL coefficient 0.001

–

 

clip 0.2; KL coefficient 0.001;

verl with vLLM rollout

Table 14: Training settings used for the reported comparisons. Hard GRPO is the released discrete-token GRPO checkpoint. Its training parameters are transcribed from Appendix D.1 of the SofT-GRPO paper (Zheng et al., 2026b).

G.4 Evaluation Configuration

Parameter
Easy
Hard

Samples per problem
32
32

Temperature
0.6
0.6

Maximum new tokens
2,048
8,192

Reported KK
16, 32
16, 32

Execution
Single GPU
Single GPU

Table 15: Offline sampled-evaluation settings.

Table 15 summarizes the offline evaluation configurations used for the Easy and Hard comparisons.
Both protocols retain 32 responses per problem at temperature 0.60.6 and report K∈{16,32}K\in\{16,32\}.
The Easy Setting limits each response to 2,048 new tokens, whereas the Hard Setting allows 8,192 to accommodate longer reasoning traces.
Both protocols run on a single GPU.
The Easy protocol additionally includes a one-sample greedy evaluation at temperature 00; the main Pass@KK and majority-vote tables use the sampled protocol shown here.

G.5 Prompt Templates

The following messages are rendered with each model’s official chat template.

Prompt for Mathematical Reasoning

system
Please reason step by step, and put your final answer within
\boxed{}.

user
{Question}

Prompt for Multiple-Choice Reasoning

system
Please reason step by step to solve the following multiple-choice question.
Put your final choice letter inside \boxed{}, e.g.,
\boxed{C}.

user
{Question and labeled choices}

Prompt for Open Question Answering

system
Answer the question using the provided context.
Keep the final answer short, and put it inside
<answer> </answer> tags.

user
{Context and question}

Prompt for MBPP Code Generation

system
You are a Python programming assistant.
Write clean, correct Python code that passes the given tests.
The function name and signature must match exactly as specified in the test cases.
The final code block must not include the tests; do not include the tests in your answer.
Do not include explanations or text outside the code block.
Output exactly one Python markdown code block containing only the final solution code.

user
Task: {Programming task}
Test Cases: {Public tests}

Countdown uses the role/content messages distributed with the dataset and renders them without rewriting the task statement.
Answer extraction follows the same task-family contract as scoring: boxed answers for mathematics and multiple choice, <answer> tags for open QA and Countdown, and one Python code block for MBPP.

G.6 Datasets and Setting Rationale

Setting

Role

Dataset / split

Purpose

Easy

Post-training

GSM8K train

Standard grade-school mathematical word problems with rule-verifiable numeric answers.

In-domain test

GSM8K test

Measures transfer to unseen problems from the post-training task family.

Held-out test

CommonsenseQA validation; HotpotQA distractor validation; Countdown test; GPQA Diamond; MBPP sanitized test

Covers commonsense choice, multi-hop QA, arithmetic planning, scientific reasoning, and code generation.

Hard

Post-training

DeepScaleR Preview

Mathematical RL training data used in the DeepScaleR line of work.

Mathematical test

AIME 2024; AIME 2025; AMC 2023; MATH-500

Competition and advanced mathematical reasoning benchmarks commonly used for 1.5B reasoning-RL evaluation.

Held-out test

GPQA Diamond; MBPP sanitized test; CommonsenseQA validation; Countdown test

Tests whether mathematical post-training transfers to or degrades non-training task families.

Table 16: Datasets and splits used in the reported comparisons. No held-out dataset contributes post-training rewards.

Dataset

Distributed source used for provenance

License declared by the source

GSM8K

openai/gsm8k

MIT

CommonsenseQA

tau/commonsense_qa

MIT

HotpotQA

Official HotpotQA dataset release

CC BY-SA 4.0

Countdown

HuggingFaceTB/Countdown-Task-GOLD; the fixed test construction corresponds to Jiayi-Pan/Countdown-Tasks-3to4

No license specified in either dataset card

GPQA

Idavidrein/gpqa

CC BY 4.0

MBPP

google-research-datasets/mbpp

CC BY 4.0

DeepScaleR Preview

agentica-org/DeepScaleR-Preview-Dataset

MIT

AIME 2024 / AIME 2025

math-ai/aime24; math-ai/aime25

Apache-2.0

AMC 2023

math-ai/amc23

No license specified in the dataset card

MATH-500

HuggingFaceH4/MATH-500

No license specified in the dataset card

Table 17: Licenses declared by the dataset distributions used in this work.

The Easy Setting uses GSM8K because it is a widely used, readily verifiable mathematical-reasoning task and supports a controlled comparison across three instruction-tuned model scales (Cobbe et al., 2021).
Its held-out suite uses CommonsenseQA (Talmor et al., 2019), HotpotQA (Yang et al., 2018), Countdown-Task-GOLD (HuggingFaceTB, 2025), GPQA (Rein et al., 2023), and MBPP (Austin et al., 2021) to test cross-domain retention.

The Hard Setting is a harder post-training protocol, not a claim that every included item is uniformly harder than every GSM8K item.
It combines the DeepSeek-R1-Distill-Qwen-1.5B backbone with DeepScaleR mathematical RL training data (Luo et al., 2025).
This choice aligns the training regime with DeepScaleR and the model/evaluation regime with representative mathematical GRPO work.
In particular, JustRL uses the same DeepSeek 1.5B backbone and reports AIME 2024, AIME 2025, AMC 2023, and MATH-500 among its standard mathematical benchmarks (He et al., 2025).
The Hard mathematical suite uses the math-ai AIME and AMC releases (Zhang and Math-AI Team, 2024; Zhang and Math-AI Team, 2025; math-ai, 2025) and MATH-500 (Lightman et al., 2023).

G.7 Evaluation Metrics

For a problem with M=32M=32 retained responses and cc verifier-correct responses, Pass@1 is the average per-response correctness c/Mc/M.
For K≤MK\leq M, Pass@KK uses the standard without-replacement estimator

Pass​@⁡K^=1−(M−cK)(MK),\widehat{\operatorname{Pass@}K}=1-\frac{\binom{M-c}{K}}{\binom{M}{K}},

(54)

so Pass@32 is the fraction of problems with at least one correct response.
Maj@KK normalizes task-specific answers, takes a deterministic plurality vote over the first KK retained responses, treats failed extraction as abstention, and scores the selected answer against the gold answer.
Reported task scores are arithmetic means over problems, with each problem weighted equally; multi-task averages likewise assign equal weight to each benchmark.

Appendix H Related Work

H.1 Evolution Strategies for LLM Reasoning Post-Training

Evolution strategies (ES) are population-based black-box optimizers that estimate parameter updates from forward-evaluated perturbations and naturally distribute candidate evaluations across workers (Salimans et al., 2017).
Recent work has adapted this paradigm to LLM reasoning post-training.
Qiu et al. demonstrate that full-parameter ES can operate directly in billion-dimensional parameter spaces and optimize LLMs with outcome-level rewards (Qiu et al., 2026).
ESSA instead restricts evolutionary search to low-rank adapters and further reduces the search space by optimizing their singular values, enabling quantized forward-only alignment on mathematical reasoning and instruction-following tasks (Korotyshova et al., 2025).
EGGROLL structures perturbations as low-rank matrices to improve the arithmetic intensity of population evaluation and applies the resulting system to reasoning post-training at large population sizes (Sarkar et al., 2026).
ESSAM combines full-parameter ES with a sharpness-aware objective and evaluates this design on GSM8K, emphasizing generalization and memory-efficient reasoning fine-tuning (Sun et al., 2026).
These methods establish several viable implementations of ES for LLM post-training, spanning direct full-parameter search, parameter-efficient search, and structured perturbations.

A related line of work asks whether the geometry of ES updates predicts capability retention.
Abdi et al. report that ES updates have larger norms and affect a larger fraction of parameter elements than GRPO updates (Abdi et al., 2026).
They associate these update properties with held-out degradation.
However, their forgetting result is obtained from a restricted experimental configuration comprising a small training set, one model, one training task, and one held-out benchmark, making training-set overfitting difficult to distinguish from broad capability forgetting.
Hoy et al. provide complementary evidence that larger parameter movement does not necessarily lead to consistent behavioral degradation: even when ES and GRPO attain similar target-task accuracy, ES travels much farther in weight space and induces broader off-task KL shifts, while the resulting accuracy changes vary across held-out benchmarks and ES iteration budgets (Hoy et al., 2026).
Taken together, these studies show that large parameter displacement can accompany capability degradation, but neither weight-space distance nor policy KL has been established as a general predictor, much less a cause, of forgetting.

H.2 Reinforcement Learning with Verifiable Rewards

Reinforcement learning with verifiable rewards (RLVR) improves reasoning models using automatically checked outcome rewards from domains such as mathematics and code.
Group Relative Policy Optimization (GRPO) removes the learned critic by normalizing rewards within a group of responses and optimizing a clipped token-level policy objective (Shao et al., 2024).
Although this recipe can improve single-sample accuracy, recent work identifies policy-entropy collapse as a recurring limitation of RLVR.
The entropy-mechanism analysis relates entropy change under policy-gradient updates to a probability–update covariance and observes sustained entropy decrease during reasoning training (Cui et al., 2025).
Subsequent studies attribute collapse to factors including positive-advantage tokens, imbalanced token-level entropy flows, clipping and off-policy reuse, and premature confidence at a small set of structurally important decision points (Xu et al., 2026; Petrenko et al., 2026; Jin et al., 2026; Jang et al., 2026).

A second line of work evaluates coverage behaviorally through large-budget sampling and asks whether RLVR expands the set of correct reasoning paths accessible from the base model.
Large-budget sampling studies find that RLVR can improve Pass@1 while underperforming the base model at larger KK, indicating more efficient exploitation of high-probability correct paths but weaker coverage of less likely solutions (Yue et al., 2025).
Controlled pretraining experiments further show that RL post-training tends to amplify behaviors already represented in the pretraining distribution (Zhao et al., 2025).
Support-based analyses similarly find that RLVR may shrink access to some correct solutions even as it raises single-attempt accuracy (Wu et al., 2025).
These findings motivate evaluating Pass@1 together with Pass@KK, rather than interpreting an increase in single-sample accuracy as an unconditional expansion of reasoning capability.

H.3 Zeroth-Order Optimization for LLMs

Zeroth-order (ZO) optimization replaces backpropagated derivatives with a gradient-shaped estimate constructed solely from scalar objective values.
Given a Gaussian direction u∼𝒩⁡(0,I)u\sim\mathcal{N}(0,I) and perturbation radius μ\mu, two forward evaluations first estimate the directional derivative as δu​f​(θ)=[f⁡(θ+μ​u)−f⁡(θ−μ​u)]/(2​μ)\delta_{u}f(\theta)=[f(\theta+\mu u)-f(\theta-\mu u)]/(2\mu), then map this scalar back to parameter space as the pseudo-gradient g^ZO​(θ,u)=δu​f​(θ)​u\widehat{g}_{\mathrm{ZO}}(\theta;u)=\delta_{u}f(\theta)u.
Averaging these pseudo-gradients across directions estimates the gradient of a Gaussian-smoothed objective, enabling parameter updates without differentiating through the model (Nesterov and Spokoiny, 2017).
MeZO adapts simultaneous perturbation-based ZO optimization to operate in place, reducing memory to approximately the inference footprint and supporting both full-parameter and parameter-efficient language-model fine-tuning (Malladi et al., 2023).
Variance-reduced extensions such as MeZO-SVRG improve the stability and convergence of forward-only language-model fine-tuning by periodically constructing lower-variance update estimates (Gautam et al., 2024).
ZO-Bench broadens the comparison across model families, task types, and fine-tuning schemes, and studies blockwise descent, hybrid optimization, forward-gradient estimators, and gradient sparsity (Zhang et al., 2024).

ES and ZO are not disjoint estimator families.
With the same objective, perturbation distribution, scale, and raw positive–negative difference, antithetic ES is algebraically identical to a two-point ZO estimator.
The practical distinction in LLM reasoning arises from the evaluation protocol: much of the ZO fine-tuning literature studies fixed downstream objectives, whereas reasoning post-training repeatedly generates stochastic autoregressive responses and scores them with terminal verifiers.
As analyzed in Appendix E, resampling responses at the perturbed parameter points can weaken the covariance required for paired subtraction to reduce variance.
This boundary makes one- versus two-point estimation an empirical design choice under rollout rewards, rather than one whose behavior can be inferred unchanged from fixed supervised objectives.

Appendix I Pass@K Profiles of Individual Post-Training Methods and Sequential Compositions

Figure 6 first compares the Pass@KK profiles of ES and GRPO when applied individually, and then examines whether the two sequential compositions can exploit their complementary strengths.
In panels (a–b), GRPO leads at Pass@1, whereas ES overtakes it as KK grows while remaining above the Base Model across all reported KK values.
In panels (c–d), ES→\rightarrowGRPO gives the strongest AIME24 profile, while GRPO→\rightarrowES leads on AIME25 for K≥2K\geq 2.
Thus, the two sequential compositions can better exploit the complementary strengths of ES and GRPO, although the effective training order is task-dependent.

(a) MATH-500

DeepSeek-R1-Distill-Qwen-1.5B

(b) GPQA

Llama-3.2-3B-Instruct

(c) AIME24

DeepSeek-R1-Distill-Qwen-1.5B

(d) AIME25

DeepSeek-R1-Distill-Qwen-1.5B

Figure 6: Pass@KK profiles of individual post-training methods and sequential compositions. For individual post-training, GRPO leads at K=1K=1, whereas ES overtakes it at larger KK while remaining above the Base Model throughout. Among the two sequential compositions, ES→\rightarrowGRPO gives the strongest AIME24 profile, while GRPO→\rightarrowES leads on AIME25 for K≥2K\geq 2, showing complementary but order-dependent benefits.

 Experimental support, please
 view the build logs
 for errors. Generated by
 

 L
 A
 T
 E

xml

.
 

Instructions for reporting errors
We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile
 support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the
 methods listed below:

Click the "Report Issue" (

) button, located in the page header.

Tip: You can select the relevant text first, to include it in your report.
Our team has already identified the following issues. We appreciate your time reviewing and reporting rendering errors we
 may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability
 should not be a barrier to accessing research. Thank you for your continued support in championing open access for
 all.
Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a list of packages that need conversion, and welcome developer contributions.

 We gratefully acknowledge support from
 our major funders,
 member institutions, ,
 and all contributors.
 

About
·
Help
·
Contact
·
Subscribe
·
Copyright
·
Privacy
·
Accessibility
·
Operational Status (opens in new tab)

Major funding support from

