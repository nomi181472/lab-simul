Title: Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization

URL Source: https://arxiv.org/html/2310.12103

Published Time: Wed, 05 Jun 2024 00:39:00 GMT

Markdown Content:
###### Abstract

Reinforcement Learning from Human Feedback (RLHF) has shown potential in qualitative tasks where easily defined performance measures are lacking. However, there are drawbacks when RLHF is commonly used to optimize for average human preferences, especially in generative tasks that demand diverse model responses. Meanwhile, Quality Diversity (QD) algorithms excel at identifying diverse and high-quality solutions but often rely on manually crafted diversity metrics. This paper introduces Quality Diversity through Human Feedback (QDHF), a novel approach that progressively infers diversity metrics from human judgments of similarity among solutions, thereby enhancing the applicability and effectiveness of QD algorithms in complex and open-ended domains. Empirical studies show that QDHF significantly outperforms state-of-the-art methods in automatic diversity discovery and matches the efficacy of QD with manually crafted diversity metrics on standard benchmarks in robotics and reinforcement learning. Notably, in open-ended generative tasks, QDHF substantially enhances the diversity of text-to-image generation from a diffusion model and is more favorably received in user studies. We conclude by analyzing QDHF’s scalability, robustness, and quality of derived diversity metrics, emphasizing its strength in open-ended optimization tasks. Code and tutorials are available at [https://liding.info/qdhf](https://liding.info/qdhf).

Machine Learning, ICML

1 Introduction
--------------

Foundation models such as large language models (LLMs) and text-to-image generation models in effect compress vast archives of human knowledge into powerful and flexible tools, serving as a foundation for down-stream applications(Brown et al., [2020](https://arxiv.org/html/2310.12103v3#bib.bib6); Bommasani et al., [2021](https://arxiv.org/html/2310.12103v3#bib.bib5)). Their promise includes helping individuals better meet their varied goals, such as exploring their creativity in different modalities and coming up with novel solutions. One mechanism to build upon such foundational knowledge is reinforcement learning from human feedback (RLHF)(Christiano et al., [2017](https://arxiv.org/html/2310.12103v3#bib.bib11)), which can make models both easier to use (by aligning them to human instructions), and more competent (by improving their capabilities based on human preferences).

RLHF is a relatively new paradigm, and its deployments often follow the relatively narrow recipe of maximizing a learned reward model of averaged human preferences over model responses. This work aims to broaden that recipe to include optimizing for interesting diversity among responses, which is of practical importance for many creative applications such as text-to-image generation(Rombach et al., [2022](https://arxiv.org/html/2310.12103v3#bib.bib46)). Such increased diversity can also improve optimization for complex and open-ended tasks (through improved exploration), personalization (serving individual rather than average human preference), and fairness (to offset algorithmic biases in gender, ethnicity, and more).

Diversity encourages exploration, which is essential for finding novel and effective solutions to many complex problems. Without diversity, optimization algorithms can converge prematurely, resulting in getting stuck in local optima or producing only a limited set of responses (i.e., mode collapse). Diversity-seeking is a core aspect of Quality Diversity (QD) algorithms(Pugh et al., [2016](https://arxiv.org/html/2310.12103v3#bib.bib44); Cully et al., [2015](https://arxiv.org/html/2310.12103v3#bib.bib15); Lehman & Stanley, [2011b](https://arxiv.org/html/2310.12103v3#bib.bib33); Mouret & Clune, [2015](https://arxiv.org/html/2310.12103v3#bib.bib36)), where diversity metrics are explicitly defined and utilized to encourage the variation of solutions during optimization.

Our main idea is to derive distinct representations of what humans find interestingly different and use such diversity representations to support optimization. We introduce a novel method, Quality Diversity through Human Feedback (QDHF), which empowers QD algorithms with diversity metrics actively learned from human feedback during optimization. QDHF serves as an online, diversity-driven optimizer for open-ended tasks. It navigates the search space through the application of progressively learned diversity, which is achieved by concurrently fine-tuning inferred diversity by aligning it to human feedback in response to the discovery of novel solutions.

To illustrate its capability, we propose a generic implementation of QDHF, which is capable of formulating arbitrary diversity metrics using latent space projection, and aligning them to human feedback through contrastive learning(Hadsell et al., [2006](https://arxiv.org/html/2310.12103v3#bib.bib27); Dosovitskiy et al., [2014](https://arxiv.org/html/2310.12103v3#bib.bib18)). Experiments are conducted on benchmarks across three domains: robotics, reinforcement learning (RL), and generative modeling, demonstrating QDHF’s superior performance in producing diverse, high-quality responses.

This work is inspired by the considerable benefits that learned reward models have unlocked for RLHF. Analogous to reward functions, diversity metrics are often qualitative, complex, and hard to specify. Existing QD algorithms demonstrate proficiency in addressing complex search tasks, but their reliance on manually crafted diversity metrics restricts their applicability in real-world open-ended tasks. QDHF aims to bridge this gap, allowing QD algorithms to easily adapt to more challenging tasks by learning diversity metrics through iterative exploration and alignment. In summary, the main contributions of this work are:

1.   1.Introducing QDHF and its implementation leveraging latent projection and contrastive learning. 
2.   2.Demonstrating that QDHF mirrors the search capabilities of QD algorithms with manually crafted diversity metrics in benchmark robotics and RL tasks. 
3.   3.Implementing QDHF in the latent space illumination (LSI) task for open-ended text-to-image generation, showing its capability in improving diffusion model to generate more diverse and preferable responses. 
4.   4.Ablation studies on QDHF’s sample efficiency, robustness, and the quality of its learned diversity metrics. 

2 Preliminaries
---------------

This section covers the basic and most relevant aspects of QD algorithms. More detailed descriptions are included in Appendix[A](https://arxiv.org/html/2310.12103v3#A1 "Appendix A Preliminaries on Quality Diversity ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization").

### 2.1 Quality Diversity

QD algorithms effectively explore the search space by maintaining diverse high-quality solutions and using them to drive optimization. Given a solution space 𝒳 𝒳\mathcal{X}caligraphic_X, QD considers an objective function J:𝒳→ℝ:𝐽→𝒳 ℝ J:\mathcal{X}\rightarrow\mathbb{R}italic_J : caligraphic_X → blackboard_R and k 𝑘 k italic_k diversity metrics M i:𝒳→ℝ:subscript 𝑀 𝑖→𝒳 ℝ M_{i}:\mathcal{X}\rightarrow\mathbb{R}italic_M start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT : caligraphic_X → blackboard_R, i=1,2,⋯,k 𝑖 1 2⋯𝑘 i=1,2,\cdots,k italic_i = 1 , 2 , ⋯ , italic_k. The diversity metrics jointly form a measurement space M⁢(𝒳)𝑀 𝒳 M(\mathcal{X})italic_M ( caligraphic_X ), which quantifies the diversity of samples. For each unique measurement in M⁢(𝒳)𝑀 𝒳 M(\mathcal{X})italic_M ( caligraphic_X ), the global objective J∗superscript 𝐽 J^{*}italic_J start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT of QD is to find a solution x∈𝒳 𝑥 𝒳 x\in\mathcal{X}italic_x ∈ caligraphic_X that has a maximum J⁢(x)𝐽 𝑥 J(x)italic_J ( italic_x ) among all solutions with the same diversity measurement.

Considering that the measurement space M⁢(𝒳)𝑀 𝒳 M(\mathcal{X})italic_M ( caligraphic_X ) is usually continuous, a QD algorithm will ultimately need to store an infinite number of solutions corresponding to each solution in the measurement space. One common way to mitigate this is to discretize M⁢(𝒳)𝑀 𝒳 M(\mathcal{X})italic_M ( caligraphic_X ) into an archive of s 𝑠 s italic_s cells {C 1,C 2,⋯,C s}subscript 𝐶 1 subscript 𝐶 2⋯subscript 𝐶 𝑠\{C_{1},C_{2},\cdots,C_{s}\}{ italic_C start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_C start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , ⋯ , italic_C start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT }, which was introduced in MAP-Elites(Mouret & Clune, [2015](https://arxiv.org/html/2310.12103v3#bib.bib36); Cully et al., [2015](https://arxiv.org/html/2310.12103v3#bib.bib15)) and has been widely adopted. The QD objective is thus approximated by a relaxation to finding a set of solutions {x i},i∈{1,…,s}subscript 𝑥 𝑖 𝑖 1…𝑠\{x_{i}\},i\in\{1,...,s\}{ italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT } , italic_i ∈ { 1 , … , italic_s }, and each x i subscript 𝑥 𝑖 x_{i}italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT is the best solution for one unique cell C i subscript 𝐶 𝑖 C_{i}italic_C start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT. This (approximated) J∗superscript 𝐽 J^{*}italic_J start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT is:

J∗=∑i=1 s max x∈𝒳,M⁢(x)∈C i⁡J⁢(x).superscript 𝐽 superscript subscript 𝑖 1 𝑠 subscript formulae-sequence 𝑥 𝒳 𝑀 𝑥 subscript 𝐶 𝑖 𝐽 𝑥 J^{*}=\sum_{i=1}^{s}\max_{x\in\mathcal{X},M(x)\in C_{i}}J(x).italic_J start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT = ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT roman_max start_POSTSUBSCRIPT italic_x ∈ caligraphic_X , italic_M ( italic_x ) ∈ italic_C start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT italic_J ( italic_x ) .(1)

### 2.2 Quality Diversity w/o Predefined Diversity Metrics

Diversity maintenance in QD usually relies on manually designed diversity metrics to ensure a varied solution archive. However, such a requirement restricts the applicability of QD in more complex and open-ended domains, where the notion of diversity is likely to be abstract and qualitative. To address this, QD with automatic discovery of diversity has become popular. Instead of using pre-defined diversity metrics, recent work such as AURORA(Cully, [2019](https://arxiv.org/html/2310.12103v3#bib.bib14); Grillotti & Cully, [2022b](https://arxiv.org/html/2310.12103v3#bib.bib25)) and RUDA(Grillotti & Cully, [2022a](https://arxiv.org/html/2310.12103v3#bib.bib24)) utilize unsupervised dimension reduction techniques to learn diversity metrics directly from the data.

One limitation of unsupervised methods is that the derived diversity metrics often capture the overall variance in current solutions, which may not align well with the diversity needed for discovering novel and superior solutions. Such misalignment requires explicit mechanisms (e.g., RUDA) to guide the search, making the methods less applicable in complex and open-ended environments. We introduce the new paradigm of QDHF, which offers greater flexibility compared to manually designing diversity metrics and outperforms unsupervised methods by leveraging diversity metrics that align to human intuition.

3 Methods
---------

This section first introduces a formulation of arbitrary diversity metrics from a generic learning perspective using latent projection, then describes how to align such metrics to human intuition of diversity using contrastive learning. Finally, we introduce quality diversity through human feedback (QDHF), an online method for open-ended diversity-driven optimization. An overview is shown in Fig.[1](https://arxiv.org/html/2310.12103v3#S3.F1 "Figure 1 ‣ 3 Methods ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization").

![Image 1: Refer to caption](https://arxiv.org/html/2310.12103v3/extracted/5642546/figures/qdhf.png)

Figure 1: Overview of the proposed QDHF method.

### 3.1 Diversity Characterization with Latent Projection

Prior work(Meyerson et al., [2016](https://arxiv.org/html/2310.12103v3#bib.bib35); Cully, [2019](https://arxiv.org/html/2310.12103v3#bib.bib14); Grillotti & Cully, [2022b](https://arxiv.org/html/2310.12103v3#bib.bib25)) has shown that unsupervised dimensionality reduction algorithms such as principal component analysis (PCA) and auto-encoders (AE) can be utilized to automatically learn robot behavioral descriptors based on sensory data. In our framework, we first introduce a concept of diversity characterization, which uses latent projection to transform arbitrary data into a semantically meaningful latent space that represents diversity. More specifically, given an input vector x 𝑥 x italic_x, we first employ a feature extractor f:𝒳→𝒴:𝑓→𝒳 𝒴 f:\mathcal{X}\rightarrow\mathcal{Y}italic_f : caligraphic_X → caligraphic_Y where 𝒳 𝒳\mathcal{X}caligraphic_X denotes the input space and 𝒴 𝒴\mathcal{Y}caligraphic_Y the feature space. Post extraction, a projection function parameterized with θ 𝜃\theta italic_θ, denoted as D r⁢(y,θ):𝒴→𝒵:subscript 𝐷 𝑟 𝑦 𝜃→𝒴 𝒵 D_{r}(y,\theta):\mathcal{Y}\rightarrow\mathcal{Z}italic_D start_POSTSUBSCRIPT italic_r end_POSTSUBSCRIPT ( italic_y , italic_θ ) : caligraphic_Y → caligraphic_Z, is applied to project y 𝑦 y italic_y into a more compact latent representation:

z=D r⁢(y,θ),where⁢y=f⁢(x).formulae-sequence 𝑧 subscript 𝐷 𝑟 𝑦 𝜃 where 𝑦 𝑓 𝑥 z=D_{r}(y,\theta),\text{where }y=f(x).italic_z = italic_D start_POSTSUBSCRIPT italic_r end_POSTSUBSCRIPT ( italic_y , italic_θ ) , where italic_y = italic_f ( italic_x ) .(2)

𝒵 𝒵\mathcal{Z}caligraphic_Z represents the latent space, wherein each dimension corresponds to a diversity metric. The magnitude and direction capture various notions of diversity, offering a compact yet informative representation of x 𝑥 x italic_x. For example, on a single dimension, smaller and larger values could indicate variations in certain characteristics such as the size of the object. In this work, we use linear projection for simplicity and generality, although non-linear multi-layer projections are also compatible with our framework. The parameters for the projection are learned through a contrastive learning process described in the following section.

### 3.2 Aligning Diversity Metrics to Human Intuition

While unsupervised dimensionality reduction methods capture significant data variances, the resulting latents may not consistently offer useful semantics for optimization. Recognizing this shortcoming, QDHF effectively aligns the latents to human notions of diversity through contrastive learning(Chopra et al., [2005](https://arxiv.org/html/2310.12103v3#bib.bib10); Schroff et al., [2015](https://arxiv.org/html/2310.12103v3#bib.bib48)).

#### Contrastive learning.

Recent work has explored using contrastive learning as a type of self-supervised representation learning strategy(Radford et al., [2021](https://arxiv.org/html/2310.12103v3#bib.bib45); Chen et al., [2020](https://arxiv.org/html/2310.12103v3#bib.bib9); He et al., [2020](https://arxiv.org/html/2310.12103v3#bib.bib28); Tian et al., [2020](https://arxiv.org/html/2310.12103v3#bib.bib53)), and has demonstrated success of using it in modeling human preferences in RLHF(Christiano et al., [2017](https://arxiv.org/html/2310.12103v3#bib.bib11)) and image similarity(Fu et al., [2023](https://arxiv.org/html/2310.12103v3#bib.bib23)). Our framework takes a similar approach. Given a triplet of latent embeddings z 1 subscript 𝑧 1 z_{1}italic_z start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT, z 2 subscript 𝑧 2 z_{2}italic_z start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT, and z 3 subscript 𝑧 3 z_{3}italic_z start_POSTSUBSCRIPT 3 end_POSTSUBSCRIPT, and supposing that the human input indicates that z 1 subscript 𝑧 1 z_{1}italic_z start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT is more similar to z 2 subscript 𝑧 2 z_{2}italic_z start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT rather than z 3 subscript 𝑧 3 z_{3}italic_z start_POSTSUBSCRIPT 3 end_POSTSUBSCRIPT, our intention is to optimize the spatial relations of z 1 subscript 𝑧 1 z_{1}italic_z start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT and z 2 subscript 𝑧 2 z_{2}italic_z start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT in the latent space relative to z 3 subscript 𝑧 3 z_{3}italic_z start_POSTSUBSCRIPT 3 end_POSTSUBSCRIPT. We use a triplet loss mechanism, i.e., to minimize the distance between z 1 subscript 𝑧 1 z_{1}italic_z start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT and z 2 subscript 𝑧 2 z_{2}italic_z start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT while maximizing between z 1 subscript 𝑧 1 z_{1}italic_z start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT and z 3 subscript 𝑧 3 z_{3}italic_z start_POSTSUBSCRIPT 3 end_POSTSUBSCRIPT via a hinge loss. This objective can be formalized as:

ℒ⁢(z 1,z 2,z 3)=max⁡(0,m+D⁢(z 1,z 2)−D⁢(z 1,z 3))ℒ subscript 𝑧 1 subscript 𝑧 2 subscript 𝑧 3 0 𝑚 𝐷 subscript 𝑧 1 subscript 𝑧 2 𝐷 subscript 𝑧 1 subscript 𝑧 3\mathcal{L}(z_{1},z_{2},z_{3})=\max(0,m+D(z_{1},z_{2})-D(z_{1},z_{3}))caligraphic_L ( italic_z start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_z start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , italic_z start_POSTSUBSCRIPT 3 end_POSTSUBSCRIPT ) = roman_max ( 0 , italic_m + italic_D ( italic_z start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_z start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT ) - italic_D ( italic_z start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_z start_POSTSUBSCRIPT 3 end_POSTSUBSCRIPT ) )(3)

where D⁢(⋅,⋅)𝐷⋅⋅D(\cdot,\cdot)italic_D ( ⋅ , ⋅ ) represents a distance metric in the embedding space, and m 𝑚 m italic_m acts as a predetermined margin. Through this approach, the latent projections are aligned to both the inherent structure of data and human notions of similarity.

#### Human judgment on similarity.

To accommodate our design, we use Two Alternative Forced Choice (2AFC)(Zhang et al., [2018](https://arxiv.org/html/2310.12103v3#bib.bib58); Fu et al., [2023](https://arxiv.org/html/2310.12103v3#bib.bib23)) to obtain human judgments on the similarity of solutions. When presented with a triplet {x 1,x 2,x 3}subscript 𝑥 1 subscript 𝑥 2 subscript 𝑥 3\{x_{1},x_{2},x_{3}\}{ italic_x start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_x start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , italic_x start_POSTSUBSCRIPT 3 end_POSTSUBSCRIPT }, an evaluator is prompted to discern whether x 2 subscript 𝑥 2 x_{2}italic_x start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT or x 3 subscript 𝑥 3 x_{3}italic_x start_POSTSUBSCRIPT 3 end_POSTSUBSCRIPT is more similar to the reference solution, x 1 subscript 𝑥 1 x_{1}italic_x start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT. Importantly, this mechanism works not only with human judgment but also with judgment produced by heuristics and AI systems, meaning that our framework remains universally applicable across different feedback modalities.

### 3.3 Quality Diversity through Human Feedback

We hereby introduce QDHF and its implementation through latent space projection and contrastive learning.

#### Objective.

In QDHF, the diversity metrics M hf,i:𝒳→ℝ:subscript 𝑀 hf 𝑖→𝒳 ℝ M_{\text{hf},i}:\mathcal{X}\rightarrow\mathbb{R}italic_M start_POSTSUBSCRIPT hf , italic_i end_POSTSUBSCRIPT : caligraphic_X → blackboard_R are derived from human feedback on the similarity of solutions. Given a solution x∈𝒳 𝑥 𝒳 x\in\mathcal{X}italic_x ∈ caligraphic_X, we define M hf,i⁢(x)=z subscript 𝑀 hf 𝑖 𝑥 𝑧 M_{\text{hf},i}(x)=z italic_M start_POSTSUBSCRIPT hf , italic_i end_POSTSUBSCRIPT ( italic_x ) = italic_z where z∈𝒵 𝑧 𝒵 z\in\mathcal{Z}italic_z ∈ caligraphic_Z is the latent representation defined in Eq.[2](https://arxiv.org/html/2310.12103v3#S3.E2 "Equation 2 ‣ 3.1 Diversity Characterization with Latent Projection ‣ 3 Methods ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization"). The latent space 𝒵 𝒵\mathcal{Z}caligraphic_Z is used as the measurement space M hf⁢(𝒳)subscript 𝑀 hf 𝒳 M_{\text{hf}}(\mathcal{X})italic_M start_POSTSUBSCRIPT hf end_POSTSUBSCRIPT ( caligraphic_X ), where each dimension i 𝑖 i italic_i in 𝒵 𝒵\mathcal{Z}caligraphic_Z corresponds to a diversity metric M hf,i subscript 𝑀 hf 𝑖 M_{\text{hf},i}italic_M start_POSTSUBSCRIPT hf , italic_i end_POSTSUBSCRIPT. We can now specialize Eq.[1](https://arxiv.org/html/2310.12103v3#S2.E1 "Equation 1 ‣ 2.1 Quality Diversity ‣ 2 Preliminaries ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization") for the objective of QDHF, J hf∗superscript subscript 𝐽 hf J_{\text{hf}}^{*}italic_J start_POSTSUBSCRIPT hf end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT, which is formulated as:

J hf∗=∑i=1 s max x∈𝒳,z∈C i⁡J⁢(x)superscript subscript 𝐽 hf superscript subscript 𝑖 1 𝑠 subscript formulae-sequence 𝑥 𝒳 𝑧 subscript 𝐶 𝑖 𝐽 𝑥 J_{\text{hf}}^{*}=\sum_{i=1}^{s}\max_{x\in\mathcal{X},z\in C_{i}}J(x)italic_J start_POSTSUBSCRIPT hf end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ∗ end_POSTSUPERSCRIPT = ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT roman_max start_POSTSUBSCRIPT italic_x ∈ caligraphic_X , italic_z ∈ italic_C start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT italic_J ( italic_x )(4)

where z=D r⁢(f⁢(x),θ)𝑧 subscript 𝐷 𝑟 𝑓 𝑥 𝜃 z=D_{r}(f(x),\theta)italic_z = italic_D start_POSTSUBSCRIPT italic_r end_POSTSUBSCRIPT ( italic_f ( italic_x ) , italic_θ ) (Eq.[2](https://arxiv.org/html/2310.12103v3#S3.E2 "Equation 2 ‣ 3.1 Diversity Characterization with Latent Projection ‣ 3 Methods ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization")). To effectively learn θ 𝜃\theta italic_θ, we use the contrastive learning mechanism (Eq.[3](https://arxiv.org/html/2310.12103v3#S3.E3 "Equation 3 ‣ Contrastive learning. ‣ 3.2 Aligning Diversity Metrics to Human Intuition ‣ 3 Methods ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization")).

#### Training.

One essential component of QDHF is the effective utilization of human feedback. The initial diversity representations are learned from feedback collected on randomly generated solutions, which are often low-quality. But as optimization proceeds, novel and higher-quality solutions are found, and the initial diversity metrics are unlikely to characterize them correctly.

We propose a progressive training strategy for QDHF where the latent projection is iteratively fine-tuned throughout the QD process. As QD continually adds new solutions to its archive, we select samples from this evolving archive to gather human judgments on their similarity, producing online feedback used to fine-tune the latent projection. Subsequently, the QD algorithm is re-initialized with current solutions but integrated with updated diversity representations. This method ensures that the diversity representations remain relevant and reflective of the improved quality of solutions as optimization advances, supporting continuous and open-ended optimization for complex tasks.

#### Baseline method.

We also propose a strong baseline method, ‘QDHF-Base,’ which integrates the general design principles of RLHF and QD algorithms but learns diversity metrics offline. This method first collects human judgments on the similarity of randomly generated solutions, then learns the diversity representations (akin to reward model learning in RLHF), and finally uses these representations as diversity metrics in QD. QDHF-Base, although a baseline, is a novel algorithm for combining RLHF with QD optimization, leveraging the proven effectiveness of the RLHF framework.

4 Theoretical Analysis
----------------------

In this section, we include further interpretations of QDHF, provide theoretical backing, and relate the advantages of QDHF to prior work in online learning.

### 4.1 Learning Distinct Diversity Representations through Information Bottleneck

When the dimensionality of the feature space, particularly in the last layer of a network, is significantly reduced, the process compels the network to encapsulate essential discriminative information within a limited set of features. This process is effectively framed by the Information Bottleneck (IB) principle(Tishby et al., [2000](https://arxiv.org/html/2310.12103v3#bib.bib54)), which suggests that an optimal representation, Z 𝑍 Z italic_Z, of input data X 𝑋 X italic_X, should conserve only the information necessary for predicting the output Y 𝑌 Y italic_Y. The IB principle articulates this as an optimization task, aiming to balance the mutual information between Z 𝑍 Z italic_Z and Y 𝑌 Y italic_Y while constraining the information between X 𝑋 X italic_X and Z 𝑍 Z italic_Z. Mathematically, this balance is captured by:

ℒ I⁢B=I⁢(X;Z)−β⁢I⁢(Z;Y).subscript ℒ 𝐼 𝐵 𝐼 𝑋 𝑍 𝛽 𝐼 𝑍 𝑌\mathcal{L}_{IB}=I(X;Z)-\beta I(Z;Y).caligraphic_L start_POSTSUBSCRIPT italic_I italic_B end_POSTSUBSCRIPT = italic_I ( italic_X ; italic_Z ) - italic_β italic_I ( italic_Z ; italic_Y ) .(5)

Here, a Lagrange multiplier β 𝛽\beta italic_β serves as a trade-off parameter, moderating the amount of information about X 𝑋 X italic_X retained in Z 𝑍 Z italic_Z and ensuring the relevance of Z 𝑍 Z italic_Z for predicting Y 𝑌 Y italic_Y. More recent work(Alemi et al., [2017](https://arxiv.org/html/2310.12103v3#bib.bib1)) has further shown the IB principle’s pivotal role in deep neural networks’ capacity to compress and selectively filter information.

In the context of contrastive learning, the objective is to optimize representations that carry the necessary information for predicting human preferences for similarity. Conventional approaches often employ large embedding dimensions to maximize the capability for similarity estimation, such as embeddings with 512 dimensions in DreamSim(Fu et al., [2023](https://arxiv.org/html/2310.12103v3#bib.bib23)). In contrast, QDHF opts for significantly reduced dimensions to produce compact representations, imposing a constraint on the mutual information between Z 𝑍 Z italic_Z and X 𝑋 X italic_X, formulated by:

I⁢(X;Z)≤I c 𝐼 𝑋 𝑍 subscript 𝐼 𝑐 I(X;Z)\leq I_{c}italic_I ( italic_X ; italic_Z ) ≤ italic_I start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT(6)

where I c subscript 𝐼 𝑐 I_{c}italic_I start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT is the information constraint. Intuitively, I c subscript 𝐼 𝑐 I_{c}italic_I start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT forces Z 𝑍 Z italic_Z to learn a minimal sufficient statistic of X 𝑋 X italic_X for predicting Y 𝑌 Y italic_Y, resulting in Z 𝑍 Z italic_Z being concise yet information-rich representations. The IB principle suggests that such representations should be both discriminative and succinct if ℒ I⁢B subscript ℒ 𝐼 𝐵\mathcal{L}_{IB}caligraphic_L start_POSTSUBSCRIPT italic_I italic_B end_POSTSUBSCRIPT is minimal. Our empirical analysis, as described in Sec.[5.4](https://arxiv.org/html/2310.12103v3#S5.SS4 "5.4 Ablation Studies and Analysis ‣ 5 Experiments ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization"), further validates that when I⁢(Z;Y)𝐼 𝑍 𝑌 I(Z;Y)italic_I ( italic_Z ; italic_Y ) - the accuracy of predicting human judgments on the validation set - is high, QDHF can learn robust and informative diversity metrics and closely approaches the capability of QD using ground truth diversity.

Complementing this, the contrastive loss function (Eq.[3](https://arxiv.org/html/2310.12103v3#S3.E3 "Equation 3 ‣ Contrastive learning. ‣ 3.2 Aligning Diversity Metrics to Human Intuition ‣ 3 Methods ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization")) minimizes the distance between similar pairs and maximizes it between dissimilar pairs, thereby encouraging the learning of meaningful representations for diversity that are aligned with human intuition. These representations can thus serve as robust diversity metrics in QD optimization, enhancing the model’s generalization and creativity.

### 4.2 Quality Diversity as an Active Sampling Strategy for Online Learning

We also provide an analysis of QDHF from the perspective of online learning. Prior work has explored using online learning for similarity, especially in large-scale data environments. QDHF introduces a nuanced methodology, enhancing traditional models like the Online Algorithm for Scalable Image Similarity (OASIS)(Chechik et al., [2010](https://arxiv.org/html/2310.12103v3#bib.bib7)). OASIS, part of the Passive-Aggressive (PA)(Crammer et al., [2006](https://arxiv.org/html/2310.12103v3#bib.bib13)) algorithm family, optimizes similarity across image triplets (p i,p i+,p i−)subscript 𝑝 𝑖 superscript subscript 𝑝 𝑖 superscript subscript 𝑝 𝑖(p_{i},p_{i}^{+},p_{i}^{-})( italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT + end_POSTSUPERSCRIPT , italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT - end_POSTSUPERSCRIPT ) to ensure a predefined margin in parameterized similarity scores S W subscript 𝑆 𝑊 S_{W}italic_S start_POSTSUBSCRIPT italic_W end_POSTSUBSCRIPT, formalized as:

l W⁢(p i,p i+,p i−)=max⁡(0,1−S W⁢(p i,p i+)+S W⁢(p i,p i−)).subscript 𝑙 𝑊 subscript 𝑝 𝑖 superscript subscript 𝑝 𝑖 superscript subscript 𝑝 𝑖 0 1 subscript 𝑆 𝑊 subscript 𝑝 𝑖 superscript subscript 𝑝 𝑖 subscript 𝑆 𝑊 subscript 𝑝 𝑖 superscript subscript 𝑝 𝑖 l_{W}(p_{i},p_{i}^{+},p_{i}^{-})=\max\left(0,1-S_{W}(p_{i},p_{i}^{+})+S_{W}(p_% {i},p_{i}^{-})\right).italic_l start_POSTSUBSCRIPT italic_W end_POSTSUBSCRIPT ( italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT + end_POSTSUPERSCRIPT , italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT - end_POSTSUPERSCRIPT ) = roman_max ( 0 , 1 - italic_S start_POSTSUBSCRIPT italic_W end_POSTSUBSCRIPT ( italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT + end_POSTSUPERSCRIPT ) + italic_S start_POSTSUBSCRIPT italic_W end_POSTSUBSCRIPT ( italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT - end_POSTSUPERSCRIPT ) ) .(7)

This contrastive loss (equivalent to Eq.[3](https://arxiv.org/html/2310.12103v3#S3.E3 "Equation 3 ‣ Contrastive learning. ‣ 3.2 Aligning Diversity Metrics to Human Intuition ‣ 3 Methods ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization") in the main paper by having a margin of 1 1 1 1) leads to the global loss L W subscript 𝐿 𝑊 L_{W}italic_L start_POSTSUBSCRIPT italic_W end_POSTSUBSCRIPT defined as the sum of hinge losses over all triplets. The loss is minimized through iterative PA updates, where each iteration i 𝑖 i italic_i involves optimizing W 𝑊 W italic_W via:

W i=argmin 𝑊⁢(1 2⁢‖W−W i−1‖F 2+C⁢ξ),subscript 𝑊 𝑖 𝑊 argmin 1 2 superscript subscript norm 𝑊 subscript 𝑊 𝑖 1 𝐹 2 𝐶 𝜉 W_{i}=\underset{W}{\text{argmin}}\left(\frac{1}{2}\|W-W_{i-1}\|_{F}^{2}+C\xi% \right),italic_W start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = underitalic_W start_ARG argmin end_ARG ( divide start_ARG 1 end_ARG start_ARG 2 end_ARG ∥ italic_W - italic_W start_POSTSUBSCRIPT italic_i - 1 end_POSTSUBSCRIPT ∥ start_POSTSUBSCRIPT italic_F end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT + italic_C italic_ξ ) ,(8)

subject to l W⁢(p i,p i+,p i−)≤ξ subscript 𝑙 𝑊 subscript 𝑝 𝑖 superscript subscript 𝑝 𝑖 superscript subscript 𝑝 𝑖 𝜉 l_{W}(p_{i},p_{i}^{+},p_{i}^{-})\leq\xi italic_l start_POSTSUBSCRIPT italic_W end_POSTSUBSCRIPT ( italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT + end_POSTSUPERSCRIPT , italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT - end_POSTSUPERSCRIPT ) ≤ italic_ξ and ξ≥0 𝜉 0\xi\geq 0 italic_ξ ≥ 0, where ∥⋅∥F\|\cdot\|_{F}∥ ⋅ ∥ start_POSTSUBSCRIPT italic_F end_POSTSUBSCRIPT denotes the Frobenius norm. The updates make sure that W i subscript 𝑊 𝑖 W_{i}italic_W start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT optimizes the balance between staying close to the previous parameters W i−1 subscript 𝑊 𝑖 1 W_{i-1}italic_W start_POSTSUBSCRIPT italic_i - 1 end_POSTSUBSCRIPT and reducing loss on the current triplet l W⁢(p i,p i+,p i−)subscript 𝑙 𝑊 subscript 𝑝 𝑖 superscript subscript 𝑝 𝑖 superscript subscript 𝑝 𝑖 l_{W}(p_{i},p_{i}^{+},p_{i}^{-})italic_l start_POSTSUBSCRIPT italic_W end_POSTSUBSCRIPT ( italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT + end_POSTSUPERSCRIPT , italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT - end_POSTSUPERSCRIPT ), with C 𝐶 C italic_C modulating this trade-off.

Conversely, QDHF focuses on QD as its core mechanism, utilizing contrastive learning to derive representations that serve as both benchmarks for diversity and guides for exploration. Unlike OASIS’s fixed-margin strategy, QDHF employs an adaptive scheme that refines its diversity metrics through continuous feedback and new data integration, fostering a progressing representation space that captures complex diversity-exploration interrelations. Mathematically, QDHF seeks to minimize the global loss as a blend of sustained learning from historical preference data and online data with feedback based on new triplet samplings:

L W=∑old triplets l W+∑new triplets l W,subscript 𝐿 𝑊 subscript old triplets subscript 𝑙 𝑊 subscript new triplets subscript 𝑙 𝑊 L_{W}=\sum_{\text{old triplets}}l_{W}+\sum_{\text{new triplets}}l_{W},italic_L start_POSTSUBSCRIPT italic_W end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT old triplets end_POSTSUBSCRIPT italic_l start_POSTSUBSCRIPT italic_W end_POSTSUBSCRIPT + ∑ start_POSTSUBSCRIPT new triplets end_POSTSUBSCRIPT italic_l start_POSTSUBSCRIPT italic_W end_POSTSUBSCRIPT ,(9)

where ∑old triplets l W subscript old triplets subscript 𝑙 𝑊\sum_{\text{old triplets}}l_{W}∑ start_POSTSUBSCRIPT old triplets end_POSTSUBSCRIPT italic_l start_POSTSUBSCRIPT italic_W end_POSTSUBSCRIPT aims to retain the learned diversity, similar to how the first term ‖W−W i−1‖F 2 superscript subscript norm 𝑊 subscript 𝑊 𝑖 1 𝐹 2\|W-W_{i-1}\|_{F}^{2}∥ italic_W - italic_W start_POSTSUBSCRIPT italic_i - 1 end_POSTSUBSCRIPT ∥ start_POSTSUBSCRIPT italic_F end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT in Eq.[8](https://arxiv.org/html/2310.12103v3#S4.E8 "Equation 8 ‣ 4.2 Quality Diversity as an Active Sampling Strategy for Online Learning ‣ 4 Theoretical Analysis ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization") ensures continuity. Theoretical analysis and empirical evidence from past studies(Crammer et al., [2006](https://arxiv.org/html/2310.12103v3#bib.bib13)) suggest that such iterative online algorithms can achieve a cumulative online loss that remains small and converges over time.

While OASIS leverages a structured update rule to ensure that the learned similarity matrix progressively aligns with the target function, QDHF employs continuous feedback loops to develop a learning environment for diversity and exploration. This strategic feedback mechanism can be viewed as using QD as a sampling strategy for actively discovering unseen data for refining the learned diversity metrics, which ensures a vibrant, adaptive optimization landscape for online learning in more complex, open-ended environments.

5 Experiments
-------------

### 5.1 Tasks and Benchmarks

We describe our experimental setup across three benchmark tasks in the domains of robotics, RL, and generative modeling. For all experiments, we use MAP-Elites(Mouret & Clune, [2015](https://arxiv.org/html/2310.12103v3#bib.bib36)) as the QD algorithm for fair comparison. More implementation details can be found in Appendix[B](https://arxiv.org/html/2310.12103v3#A2 "Appendix B Additional Implementation Details ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization").

#### Robotic arm.

We use the robotic arm domain from Cully et al. ([2015](https://arxiv.org/html/2310.12103v3#bib.bib15)); Vassiliades & Mouret ([2018](https://arxiv.org/html/2310.12103v3#bib.bib55)). The goal is to identify an inverse kinematics solution for each accessible position of a planar robotic arm. The objective function is to minimize the variance of the joint angles. The standard way of measuring diversity is the endpoint’s positions of the arm in the 2D space, which is computed using the forward kinematics of the arm, as detailed in Murray et al. ([2017](https://arxiv.org/html/2310.12103v3#bib.bib38)).

#### Maze navigation (RL).

We adopt the Kheperax(Grillotti & Cully, [2023](https://arxiv.org/html/2310.12103v3#bib.bib26)) RL environment, which features a maze navigation task originally proposed in Mouret & Doncieux ([2012](https://arxiv.org/html/2310.12103v3#bib.bib37)). The goal is to discover a collection of neural network policy that controls the agent to navigate across varying positions in a maze. A Khepera-like robot is used as the agent, which is equipped with laser sensors positioned at the robot’s facing directions for computing the distance. The maze is designed to be deceptive with immovable walls, making navigation a challenging optimization task.

#### Latent space illumination.

The latent space illumination (LSI) task(Fontaine et al., [2021](https://arxiv.org/html/2310.12103v3#bib.bib22)) is designed for exploring the latent space of a generative model. LSI initially aims to generate different gameplay levels and has later been extended to generate images of human faces that correspond to specific text prompts as diversity(Fontaine & Nikolaidis, [2021](https://arxiv.org/html/2310.12103v3#bib.bib19)). We introduce a new LSI benchmark on text-to-image generation with Stable Diffusion(Rombach et al., [2022](https://arxiv.org/html/2310.12103v3#bib.bib46)), a latent diffusion model with state-of-the-art capability. In this task, the goal is to find high-quality (to match a text prompt, scored by CLIP(Radford et al., [2021](https://arxiv.org/html/2310.12103v3#bib.bib45))) and diverse images by optimizing the latent vectors, where the diffusion model is treated as a black box. There are no pre-defined diversity metrics, making this task a more challenging and open-ended optimization problem in a real-world setting.

This benchmark also features two types of text prompts as subtasks: singular and compositional prompts. Singular prompts elicit responses about a single concept, usually taking the form of “an image of a category” where category is drawn from labels in popular image datasets such as COCO(Lin et al., [2014](https://arxiv.org/html/2310.12103v3#bib.bib34)) and ImageNet(Russakovsky et al., [2015](https://arxiv.org/html/2310.12103v3#bib.bib47)). We also include a few popular text prompts used in prior work(Rombach et al., [2022](https://arxiv.org/html/2310.12103v3#bib.bib46)) that have a similar form. For singular prompts, diversity gives more variation in images that have the same semantics, which can better meet the diverse preferences of users.

Compositional prompts are complex conjunctions of multiple concepts. We use the MCC-250 benchmark(Bellagente et al., [2023](https://arxiv.org/html/2310.12103v3#bib.bib3)), which consists of prompts describing two objects with respective attributes, e.g., “a red apple and a yellow banana”. Recent work has shown that diffusion models often fail to correctly compose the specific objects and attributes(Chefer et al., [2023](https://arxiv.org/html/2310.12103v3#bib.bib8); Bellagente et al., [2023](https://arxiv.org/html/2310.12103v3#bib.bib3)), but enhanced diversity may explore different ways of composing, and thus improve the chance of generating the correct image given a limited budget of responses.

Table 1: Results for robotic arm. We report the QD score (normalized to a scale of 0-100) and coverage for “All Solutions” (solutions found throughout training) and “Archive Solutions” (solutions in the final archive). QDHF significantly outperforms QDHF-Base and AURORA, and closely approaches the search capability of QD using ground truth diversity metrics when considering all solutions.

### 5.2 Experimental Design and Evaluation

In our experiments, we design two distinct scenarios: structured tasks, where ground truth diversity metrics are available, and open-ended tasks, where such metrics are not.

#### Structured tasks.

The first scenario leverages a predefined ground truth diversity metric to simulate human feedback. The primary reason for using simulated feedback is to facilitate evaluation and comparisons, which requires measuring diversity consistently across different methods. The robotic arm and maze navigation tasks fall under this category, where the ground truth diversity metrics correspond to the position (x and y values) of the arm and of the robot in the 2D space, respectively. The “human judgment” is determined by the similarity of the positions, calculated as the L2 distance from the ground truth measurements.

To validate the effectiveness of QDHF, we benchmark against AURORA(Grillotti & Cully, [2022b](https://arxiv.org/html/2310.12103v3#bib.bib25); Cully, [2019](https://arxiv.org/html/2310.12103v3#bib.bib14)) and standard QD. The standard QD uses ground truth diversity metrics, which offers an oracle control representing the best possible performance of QD. For comprehensiveness, we implement four variants of AURORA, encompassing two dimension-reduction techniques: PCA and AE, and two training strategies: pre-trained (Pre) and incremental (Inc).

For evaluation, solutions of each method are additionally stored in a separate archive corresponding to the ground truth diversity metrics. We report QD score(Pugh et al., [2015](https://arxiv.org/html/2310.12103v3#bib.bib43)) (sum of objective values, Eq.[1](https://arxiv.org/html/2310.12103v3#S2.E1 "Equation 1 ‣ 2.1 Quality Diversity ‣ 2 Preliminaries ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization")) and coverage (ratio of filled cells to total cells), which are standard metrics for assessing both quality and diversity of solutions. The evaluation is conducted in two settings: 1) solutions in the final archive, and 2) solutions discovered by the algorithm throughout its entire search process. The first setting evaluates the alignment of learned diversity metrics with the ground truth metrics, and the second setting provides insights into the overall efficacy of the search process regardless of how the diversity is measured and maintained.

#### Open-ended tasks.

In the second scenario, there is no ground truth diversity metric and real human feedback data is used in training and evaluation, which applies to the LSI task. To facilitate the efficiency and scalability of our method, instead of having human labelers in the loop, we train a preference model with real human feedback data and use the preference model to source feedback for training and evaluation. Similar approaches have been widely used in recent RLHF applications(Stiennon et al., [2020](https://arxiv.org/html/2310.12103v3#bib.bib52); Ouyang et al., [2022](https://arxiv.org/html/2310.12103v3#bib.bib42)).

We use human judgment data from the NIGHTS dataset, and the DreamSim model to estimate image similarity, both from Fu et al. ([2023](https://arxiv.org/html/2310.12103v3#bib.bib23)). For comparisons, we implement a best-of-n approach on Stable Diffusion as the baseline, which generates the same number of images as QDHF with random latents, and select a solution set with top CLIP scores. We also propose a stronger baseline (Best-of-n+) featuring a heuristic that increases the diversity of the latents by sampling the next latent to be distant from previous ones.

The LSI experiments are conducted using 100 singular and 100 compositional prompts. Quantitatively, we report the average CLIP score for assessing how well the generated images match the prompt. We also use DreamSim to calculate the pairwise distance between images in the solution to measure diversity. The mean pairwise distance indicates the average separation of images, and the standard deviation indicates the variability in how the images are distributed. We also conduct human evaluation (n=50 𝑛 50 n=50 italic_n = 50) using Amazon Mechanical Turk to qualitatively evaluate the results, and show examples of solutions for qualitative assessment.

Table 2: Results for maze navigation. QDHF shows superior performance compared to QDHF-Base and AURORA, as well as a resemblance of the search capability of QD with ground-truth diversity.

Table 3: Quantitative Results for LSI (singular prompts). We report the CLIP score and DreamSim pairwise distance (mean and std.) as quantitative metrics. QDHF demonstrates competitive quality to Stable Diffusion (Best-of-n) measured by CLIP score, and significantly better diversity measured by pariwise distance.

Table 4: Human evaluation results for LSI. In a blind user experience study, QDHF outperforms Stable Diffusion (best-of-n) by considerable margins in terms of user preference and perceived diversity for singular prompts and shows a clear advantage in correctness by human judgment for compositional prompts.

![Image 2: Refer to caption](https://arxiv.org/html/2310.12103v3/extracted/5642546/figures/2.jpg)

Figure 2: Qualitative result for LSI (singular prompts). The target prompt is “an image of a bear in a national park”. The left 4x4 grid displays Stable Diffusion (Best-of-n) results, i.e., randomly generated images with the highest CLIP scores. The right grid displays a uniformly sampled subset of QDHF solutions. Qualitatively, images generated by QDHF have more variations and show visible trends of diversity such as object sizes (large to small along the x-axis) and landscape types (rocky to verdant along the y-axis, terrestrial to aquatic along the x-axis). The selected example represents the average user preference ratio observed in our user experience study, where QDHF results are about twice as preferred and three times considered as more diverse.

![Image 3: Refer to caption](https://arxiv.org/html/2310.12103v3/extracted/5642546/figures/obj3.jpg)

Figure 3: Qualitative result for LSI (compositional prompts). The target prompt is “a brown car and a red clock”. The left 3x3 grid displays SD (Best-of-n) results, where we can observe the issue of attribute leakage, i.e., the color of the clock (red) leaks to the car. However, QDHF (on the right) can generate conceptually correct responses by having more variation in the responses.

### 5.3 Results

We present our major results and findings below, with more detailed results available in Appendix[C](https://arxiv.org/html/2310.12103v3#A3 "Appendix C Additional Results ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization").

#### Robotic arm.

For the robotic arm task, detailed results are presented in Table[1](https://arxiv.org/html/2310.12103v3#S5.T1 "Table 1 ‣ Latent space illumination. ‣ 5.1 Tasks and Benchmarks ‣ 5 Experiments ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization"). The statistics are accumulated over 20 repeated trials. QDHF significantly surpasses AURORA in terms of both QD score and coverage. The results highlight QDHF’s capability to enhance QD algorithms over unsupervised diversity discovery methods. QDHF also outperforms QDHF-Base, a variant that separates the processes of diversity learning and QD optimization. Such results validate our hypothesis that diversity representations need to be refined during optimization, as finding more high-quality solutions changes the distribution of the solutions in the search space. It is worth highlighting that QDHF closely matches the performance of QD with the ground truth metric when evaluating all solutions, which indicates that QDHF is a competitive alternative for QD in situations where manually designed metrics are not available.

#### Maze navigation.

Results are shown in Table[2](https://arxiv.org/html/2310.12103v3#S5.T2 "Table 2 ‣ Open-ended tasks. ‣ 5.2 Experimental Design and Evaluation ‣ 5 Experiments ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization"). The statistics are accumulated over 10 repeated trials since maze navigation is more computationally expensive. Similar to the robotic arm task, we observe that QDHF surpasses QDHF-Base and AURORA, and closely matches the performance of standard QD. The results further support the validity of QDHF as an alternative to standard QD with improved flexibility and competitive performance.

#### Latent space illumination.

Results are shown in Table[3](https://arxiv.org/html/2310.12103v3#S5.T3 "Table 3 ‣ Open-ended tasks. ‣ 5.2 Experimental Design and Evaluation ‣ 5 Experiments ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization"). The results are summarized over 100 text prompts, and we include more details in Appendix[C.2](https://arxiv.org/html/2310.12103v3#A3.SS2 "C.2 Latent Space Illumination ‣ Appendix C Additional Results ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization"). SD (Best-of-n) is the best-of-n responses produced by Stable Diffusion, and SD (Best-of-n+) is the best-of-n with a heuristic to enhance diversity in sampling (Sec. [5.2](https://arxiv.org/html/2310.12103v3#S5.SS2 "5.2 Experimental Design and Evaluation ‣ 5 Experiments ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization")). Quantitatively, QDHF has a similar CLIP score to both baseline methods, but much higher mean and standard deviation of pairwise distance, which indicates that QDHF is able to generate more diverse solutions while maintaining the high-quality. We also observe that SD (Best-of-n+) does not produce better diversity, which indicates that there exists a strong inductive bias in the diffusion model and that generating diverse images is a challenging latent optimization problem.

We also compare QDHF and SD (Best-of-n) with human evaluation in a blind user study. For singular prompts, QDHF outperforms SD (Best-of-n) with a considerable margin on both the user preference ratio and user diversity perception ratio. Notably, we find that most users think QDHF generates more diverse images, and a majority of them also prefer to have such diversity in the solution. An example of the LSI results is depicted in Fig.[2](https://arxiv.org/html/2310.12103v3#S5.F2 "Figure 2 ‣ Open-ended tasks. ‣ 5.2 Experimental Design and Evaluation ‣ 5 Experiments ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization"), where QDHF results have more variations and show visible trends of diversity.

For compositional prompts, users are asked to select which 3x3 set of generated images most accurately responds to the prompt. We find that QDHF is more accurate in generating images with correct concepts as evaluated by humans. Fig.[3](https://arxiv.org/html/2310.12103v3#S5.F3 "Figure 3 ‣ Open-ended tasks. ‣ 5.2 Experimental Design and Evaluation ‣ 5 Experiments ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization") shows an example, where QDHF mitigates the common attribute leakage issue in compositional image generation by improving the diversity of responses. It is worth highlighting that QDHF serves as a flexible black-box optimization method, compatible with any existing generative models, such as Stable Diffusion, without requiring access to or fine-tuning the model parameters.

![Image 4: Refer to caption](https://arxiv.org/html/2310.12103v3/extracted/5642546/figures/qd1.png)

![Image 5: Refer to caption](https://arxiv.org/html/2310.12103v3/extracted/5642546/figures/qd2.png)

Figure 4: Analysis of varying human feedback sample sizes on robotic arm. “Judgment prediction val acc” is the accuracy of the latent projection in predicting human preferences based on a validation set. There is a direct correlation between QD score and sample size, with QDHF’s performance closely tied to the accuracy of latent projection in reflecting human judgment.

### 5.4 Ablation Studies and Analysis

We highlight the analysis of QDHF’s scalability and quality of learned diversity metrics. Additional analysis on robustness is provided in Appendix[C.1](https://arxiv.org/html/2310.12103v3#A3.SS1 "C.1 Robotic Arm ‣ Appendix C Additional Results ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization").

#### Scalability.

To ensure scalability, determining the necessary amount of human feedback to achieve sufficient performance is critical. To investigate this, we perform a study of varying sample sizes of human judgments, shown in Fig.[4](https://arxiv.org/html/2310.12103v3#S5.F4 "Figure 4 ‣ Latent space illumination. ‣ 5.3 Results ‣ 5 Experiments ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization"). The left plot demonstrates the relationship between QD score and sample size, showing a strong correlation. Diving deeper into this relationship, we evaluate how accurately the latent projection captures the diversity and mirrors human judgment using the accuracy of predicting human judgment on a validation set. The right plot reveals a strong correlation between accuracy and QD score, which suggests that QDHF relies on learning useful diversity metrics. In other words, by evaluating the accuracy of judgment prediction during training, we can estimate whether current feedback data is sufficient for QDHF to perform well.

Secondly, we note that the judgment data for QDHF does not need to come entirely from humans. In the LSI experiments, we use a preference model (DreamSim) to source human feedback during training QDHF, and we show that an accurate preference model can be used for QDHF as an alternative to human labelers. Combining these two observations, we conclude that QDHF has good scalability towards more complex tasks because 1) we can anticipate its performance through online validation, and 2) the samples used by QDHF can be sourced from a preference model trained on a fixed amount of human labor.

#### Alignment between learned and ground truth diversity.

We evaluate the alignment between diversity metrics derived by QDHF and the underlying ground truth diversity metrics. In Fig.[5](https://arxiv.org/html/2310.12103v3#S6.F5 "Figure 5 ‣ Learning from human feedback. ‣ 6 Related Work ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization"), the archives of QDHF and AURORA-Inc (PCA) are visualized for maze navigation. Both AURORA and QDHF appear to effectively learn a diversity space reflective of the ground truth. However, QDHF exhibits an enhanced capability to discern the relative distances between solutions, especially in under-explored areas. This indicates QDHF’s strength in identifying novel solutions, and this efficacy stems from QDHF’s ability to better align its learned diversity space to the ground truth diversity, especially concerning the scales. A similar analysis for the robotic arm task is included in Appendix[C.1](https://arxiv.org/html/2310.12103v3#A3.SS1 "C.1 Robotic Arm ‣ Appendix C Additional Results ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization"), where QDHF also shows its advantage in modeling the scales of diversity representations for enhanced discovery of novel solutions.

6 Related Work
--------------

#### Learning from human feedback.

This work expands upon recent developments in methodologies for aligning models to human objectives, especially reinforcement learning from human feedback (RLHF)(Christiano et al., [2017](https://arxiv.org/html/2310.12103v3#bib.bib11); Ibarz et al., [2018](https://arxiv.org/html/2310.12103v3#bib.bib29)). RLHF was initially proposed to train RL agents in simulated environments such as Atari games. It has since been applied to fine-tune or perform one-shot learning on language models for tasks including text summarization(Ziegler et al., [2019](https://arxiv.org/html/2310.12103v3#bib.bib59); Stiennon et al., [2020](https://arxiv.org/html/2310.12103v3#bib.bib52); Wu et al., [2021](https://arxiv.org/html/2310.12103v3#bib.bib57)), dialogue(Jaques et al., [2019](https://arxiv.org/html/2310.12103v3#bib.bib30)), and question-answering(Nakano et al., [2021](https://arxiv.org/html/2310.12103v3#bib.bib39); Bai et al., [2022](https://arxiv.org/html/2310.12103v3#bib.bib2); Ouyang et al., [2022](https://arxiv.org/html/2310.12103v3#bib.bib42)), as well as vision tasks such as measuring perceptual similarity(Fu et al., [2023](https://arxiv.org/html/2310.12103v3#bib.bib23)). While past efforts have focused on learning reward or preference models from human intentions, we propose to learn diversity metrics through human feedback, which then drives the optimization process in QD algorithms.

![Image 6: Refer to caption](https://arxiv.org/html/2310.12103v3/extracted/5642546/figures/heat_maze.png)

Figure 5: Visualization of maze navigation solutions in different diversity spaces. “GT” stands for ground truth. Each point on the heatmap is a solution with its objective value visualized in color. QDHF fills up the archives with more solutions than AURORA. While both methods learned a rotated version of the maze as the diversity metrics (first column), QDHF more accurately captures the scale of the maze, especially in under-explored areas.

#### Diversity-driven optimization.

Instead of optimizing for a single optimal solution, diversity-driven optimization methods such as QD(Lehman & Stanley, [2011b](https://arxiv.org/html/2310.12103v3#bib.bib33); Cully et al., [2015](https://arxiv.org/html/2310.12103v3#bib.bib15); Mouret & Clune, [2015](https://arxiv.org/html/2310.12103v3#bib.bib36); Pugh et al., [2016](https://arxiv.org/html/2310.12103v3#bib.bib44)) and Novelty Search(Lehman & Stanley, [2011a](https://arxiv.org/html/2310.12103v3#bib.bib32)) aim to identify a variety of (top-performing) solutions that exhibit novelty or diversity. Prior work expands on QD by enhancing diversity maintenance(Fontaine et al., [2019](https://arxiv.org/html/2310.12103v3#bib.bib20); Smith et al., [2016](https://arxiv.org/html/2310.12103v3#bib.bib50); Vassiliades et al., [2017](https://arxiv.org/html/2310.12103v3#bib.bib56)), the search process(Fontaine et al., [2020](https://arxiv.org/html/2310.12103v3#bib.bib21); Vassiliades & Mouret, [2018](https://arxiv.org/html/2310.12103v3#bib.bib55); Nordmoen et al., [2018](https://arxiv.org/html/2310.12103v3#bib.bib41); Sfikas et al., [2021](https://arxiv.org/html/2310.12103v3#bib.bib49)), optimization mechanism(Kent & Branke, [2020](https://arxiv.org/html/2310.12103v3#bib.bib31); Conti et al., [2018](https://arxiv.org/html/2310.12103v3#bib.bib12); Fontaine & Nikolaidis, [2021](https://arxiv.org/html/2310.12103v3#bib.bib19)), and exploring implicit methods for diversity-driven optimization(Ding & Spector, [2022](https://arxiv.org/html/2310.12103v3#bib.bib16); Boldi et al., [2023](https://arxiv.org/html/2310.12103v3#bib.bib4); Ding et al., [2023](https://arxiv.org/html/2310.12103v3#bib.bib17); Ni et al., [2024](https://arxiv.org/html/2310.12103v3#bib.bib40); Spector et al., [2024](https://arxiv.org/html/2310.12103v3#bib.bib51)). Recent work(Grillotti & Cully, [2022a](https://arxiv.org/html/2310.12103v3#bib.bib24), [b](https://arxiv.org/html/2310.12103v3#bib.bib25); Cully, [2019](https://arxiv.org/html/2310.12103v3#bib.bib14)) also explores unsupervised methods for diversity discovery. Our work differs by leveraging human feedback to derive diversity metrics that are aligned with human interest, and thus often more beneficial for optimization.

7 Conclusion
------------

This paper introduces Quality Diversity through Human Feedback (QDHF), which expands the reach of QD by leveraging human feedback to effectively derive diversity metrics. Empirical results show that QDHF outperforms current unsupervised diversity discovery methods, and compares well to standard QD that uses manually crafted diversity metrics. In particular, applying QDHF in an open-ended generative task substantially enhances the diversity of text-to-image generations. We provide an analysis of QDHF’s scalability, robustness, and the quality of diversity metrics learned. Future work will focus on applying QDHF to more challenging tasks in complex and open-ended environments.

Acknowledgments
---------------

Li Ding and Lee Spector were supported by the National Science Foundation under Grant No. 2117377. Jeff Clune and Jenny Zhang were supported by the Vector Institute, a grant from Schmidt Futures, an NSERC Discovery Grant, and a generous donation from Rafael Cosman. Any opinions, findings, and conclusions or recommendations expressed in this publication are those of the authors and do not necessarily reflect the views of the funding agencies. The authors would like to thank Andrew Dai and Herbie Bradley for insightful suggestions, Bryon Tjanaka for help preparing the tutorial, members of the PUSH lab at Amherst College, University of Massachusetts Amherst, and Hampshire College for helpful discussions, and anonymous reviewers for their thoughtful feedback. This work utilized resources from Unity, a collaborative, multi-institutional high-performance computing cluster at the Massachusetts Green High Performance Computing Center (MGHPCC).

Impact Statement
----------------

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

References
----------

*   Alemi et al. (2017) Alemi, A.A., Fischer, I., Dillon, J.V., and Murphy, K. Deep variational information bottleneck. In _International Conference on Learning Representations_, 2017. 
*   Bai et al. (2022) Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. _arXiv preprint arXiv:2204.05862_, 2022. 
*   Bellagente et al. (2023) Bellagente, M., Brack, M., Teufel, H., Friedrich, F., Deiseroth, B., Eichenberg, C., Dai, A., Baldock, R., Nanda, S., Oostermeijer, K., et al. Multifusion: Fusing pre-trained models for multi-lingual, multi-modal image generation. _arXiv preprint arXiv:2305.15296_, 2023. 
*   Boldi et al. (2023) Boldi, R., Ding, L., and Spector, L. Objectives are all you need: Solving deceptive problems without explicit diversity maintenance. In _Second Agent Learning in Open-Endedness Workshop_, 2023. 
*   Bommasani et al. (2021) Bommasani, R., Hudson, D.A., Adeli, E., Altman, R., Arora, S., von Arx, S., Bernstein, M.S., Bohg, J., Bosselut, A., Brunskill, E., et al. On the opportunities and risks of foundation models. _arXiv preprint arXiv:2108.07258_, 2021. 
*   Brown et al. (2020) Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. _Advances in neural information processing systems_, 33:1877–1901, 2020. 
*   Chechik et al. (2010) Chechik, G., Sharma, V., Shalit, U., and Bengio, S. Large scale online learning of image similarity through ranking. _Journal of Machine Learning Research_, 11(3), 2010. 
*   Chefer et al. (2023) Chefer, H., Alaluf, Y., Vinker, Y., Wolf, L., and Cohen-Or, D. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. _ACM Transactions on Graphics (TOG)_, 42(4):1–10, 2023. 
*   Chen et al. (2020) Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. A simple framework for contrastive learning of visual representations. In _International conference on machine learning_, pp.1597–1607. PMLR, 2020. 
*   Chopra et al. (2005) Chopra, S., Hadsell, R., and LeCun, Y. Learning a similarity metric discriminatively, with application to face verification. In _2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05)_, volume 1, pp. 539–546. IEEE, 2005. 
*   Christiano et al. (2017) Christiano, P.F., Leike, J., Brown, T., Martic, M., Legg, S., and Amodei, D. Deep reinforcement learning from human preferences. _Advances in neural information processing systems_, 30, 2017. 
*   Conti et al. (2018) Conti, E., Madhavan, V., Petroski Such, F., Lehman, J., Stanley, K., and Clune, J. Improving exploration in evolution strategies for deep reinforcement learning via a population of novelty-seeking agents. _Advances in neural information processing systems_, 31, 2018. 
*   Crammer et al. (2006) Crammer, K., Dekel, O., Keshet, J., Shalev-Shwartz, S., Singer, Y., and Warmuth, M.K. Online passive-aggressive algorithms. _Journal of Machine Learning Research_, 7(3), 2006. 
*   Cully (2019) Cully, A. Autonomous skill discovery with quality-diversity and unsupervised descriptors. In _Proceedings of the Genetic and Evolutionary Computation Conference_, pp. 81–89, 2019. 
*   Cully et al. (2015) Cully, A., Clune, J., Tarapore, D., and Mouret, J.-B. Robots that can adapt like animals. _Nature_, 521(7553):503–507, 2015. 
*   Ding & Spector (2022) Ding, L. and Spector, L. Optimizing neural networks with gradient lexicase selection. In _International Conference on Learning Representations_, 2022. URL [https://openreview.net/forum?id=J_2xNmVcY4](https://openreview.net/forum?id=J_2xNmVcY4). 
*   Ding et al. (2023) Ding, L., Pantridge, E., and Spector, L. Probabilistic lexicase selection. In _Proceedings of the Genetic and Evolutionary Computation Conference_, pp. 1073–1081, 2023. 
*   Dosovitskiy et al. (2014) Dosovitskiy, A., Springenberg, J.T., Riedmiller, M., and Brox, T. Discriminative unsupervised feature learning with convolutional neural networks. _Advances in neural information processing systems_, 27, 2014. 
*   Fontaine & Nikolaidis (2021) Fontaine, M. and Nikolaidis, S. Differentiable quality diversity. _Advances in Neural Information Processing Systems_, 34:10040–10052, 2021. 
*   Fontaine et al. (2019) Fontaine, M.C., Lee, S., Soros, L.B., de Mesentier Silva, F., Togelius, J., and Hoover, A.K. Mapping hearthstone deck spaces through map-elites with sliding boundaries. In _Proceedings of The Genetic and Evolutionary Computation Conference_, pp. 161–169, 2019. 
*   Fontaine et al. (2020) Fontaine, M.C., Togelius, J., Nikolaidis, S., and Hoover, A.K. Covariance matrix adaptation for the rapid illumination of behavior space. In _Proceedings of the 2020 genetic and evolutionary computation conference_, pp. 94–102, 2020. 
*   Fontaine et al. (2021) Fontaine, M.C., Liu, R., Khalifa, A., Modi, J., Togelius, J., Hoover, A.K., and Nikolaidis, S. Illuminating mario scenes in the latent space of a generative adversarial network. In _Proceedings of the AAAI Conference on Artificial Intelligence_, volume 35, pp. 5922–5930, 2021. 
*   Fu et al. (2023) Fu, S., Tamir, N., Sundaram, S., Chai, L., Zhang, R., Dekel, T., and Isola, P. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. _Advances in Neural Information Processing Systems_, 36, 2023. 
*   Grillotti & Cully (2022a) Grillotti, L. and Cully, A. Relevance-guided unsupervised discovery of abilities with quality-diversity algorithms. In _Proceedings of the Genetic and Evolutionary Computation Conference_, pp. 77–85, 2022a. 
*   Grillotti & Cully (2022b) Grillotti, L. and Cully, A. Unsupervised behavior discovery with quality-diversity optimization. _IEEE Transactions on Evolutionary Computation_, 26(6):1539–1552, 2022b. 
*   Grillotti & Cully (2023) Grillotti, L. and Cully, A. Kheperax: a lightweight jax-based robot control environment for benchmarking quality-diversity algorithms. In _Proceedings of the Companion Conference on Genetic and Evolutionary Computation_, pp. 2163–2165, 2023. 
*   Hadsell et al. (2006) Hadsell, R., Chopra, S., and LeCun, Y. Dimensionality reduction by learning an invariant mapping. In _2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’06)_, volume 2, pp. 1735–1742. IEEE, 2006. 
*   He et al. (2020) He, K., Fan, H., Wu, Y., Xie, S., and Girshick, R. Momentum contrast for unsupervised visual representation learning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_, pp. 9729–9738, 2020. 
*   Ibarz et al. (2018) Ibarz, B., Leike, J., Pohlen, T., Irving, G., Legg, S., and Amodei, D. Reward learning from human preferences and demonstrations in atari. _Advances in neural information processing systems_, 31, 2018. 
*   Jaques et al. (2019) Jaques, N., Ghandeharioun, A., Shen, J.H., Ferguson, C., Lapedriza, A., Jones, N., Gu, S., and Picard, R. Way off-policy batch deep reinforcement learning of implicit human preferences in dialog. _arXiv preprint arXiv:1907.00456_, 2019. 
*   Kent & Branke (2020) Kent, P. and Branke, J. Bop-elites, a bayesian optimisation algorithm for quality-diversity search. _arXiv preprint arXiv:2005.04320_, 2020. 
*   Lehman & Stanley (2011a) Lehman, J. and Stanley, K.O. Abandoning objectives: Evolution through the search for novelty alone. _Evolutionary computation_, 19(2):189–223, 2011a. 
*   Lehman & Stanley (2011b) Lehman, J. and Stanley, K.O. Evolving a diversity of virtual creatures through novelty search and local competition. In _Proceedings of the 13th annual conference on Genetic and evolutionary computation_, pp. 211–218, 2011b. 
*   Lin et al. (2014) Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., and Zitnick, C.L. Microsoft coco: Common objects in context. In _Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13_, pp.740–755. Springer, 2014. 
*   Meyerson et al. (2016) Meyerson, E., Lehman, J., and Miikkulainen, R. Learning behavior characterizations for novelty search. In _Proceedings of the Genetic and Evolutionary Computation Conference 2016_, pp. 149–156, 2016. 
*   Mouret & Clune (2015) Mouret, J.-B. and Clune, J. Illuminating search spaces by mapping elites. _arXiv preprint arXiv:1504.04909_, 2015. 
*   Mouret & Doncieux (2012) Mouret, J.-B. and Doncieux, S. Encouraging behavioral diversity in evolutionary robotics: An empirical study. _Evolutionary computation_, 20(1):91–133, 2012. 
*   Murray et al. (2017) Murray, R.M., Li, Z., and Sastry, S.S. _A mathematical introduction to robotic manipulation_. CRC press, 2017. 
*   Nakano et al. (2021) Nakano, R., Hilton, J., Balaji, S., Wu, J., Ouyang, L., Kim, C., Hesse, C., Jain, S., Kosaraju, V., Saunders, W., et al. Webgpt: Browser-assisted question-answering with human feedback. _arXiv preprint arXiv:2112.09332_, 2021. 
*   Ni et al. (2024) Ni, A., Ding, L., and Spector, L. Dalex: Lexicase-like selection via diverse aggregation. In _European Conference on Genetic Programming (Part of EvoStar)_, pp. 90–107. Springer, 2024. 
*   Nordmoen et al. (2018) Nordmoen, J., Samuelsen, E., Ellefsen, K.O., and Glette, K. Dynamic mutation in map-elites for robotic repertoire generation. In _Artificial Life Conference Proceedings_, pp. 598–605. MIT Press One Rogers Street, Cambridge, MA 02142-1209, USA journals-info…, 2018. 
*   Ouyang et al. (2022) Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. _Advances in Neural Information Processing Systems_, 35:27730–27744, 2022. 
*   Pugh et al. (2015) Pugh, J.K., Soros, L.B., Szerlip, P.A., and Stanley, K.O. Confronting the challenge of quality diversity. In _Proceedings of the 2015 Annual Conference on Genetic and Evolutionary Computation_, pp. 967–974, 2015. 
*   Pugh et al. (2016) Pugh, J.K., Soros, L.B., and Stanley, K.O. Quality diversity: A new frontier for evolutionary computation. _Frontiers in Robotics and AI_, pp.40, 2016. 
*   Radford et al. (2021) Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_, pp.8748–8763. PMLR, 2021. 
*   Rombach et al. (2022) Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pp. 10684–10695, 2022. 
*   Russakovsky et al. (2015) Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., et al. Imagenet large scale visual recognition challenge. _International journal of computer vision_, 115(3):211–252, 2015. 
*   Schroff et al. (2015) Schroff, F., Kalenichenko, D., and Philbin, J. Facenet: A unified embedding for face recognition and clustering. In _Proceedings of the IEEE conference on computer vision and pattern recognition_, pp. 815–823, 2015. 
*   Sfikas et al. (2021) Sfikas, K., Liapis, A., and Yannakakis, G.N. Monte carlo elites: Quality-diversity selection as a multi-armed bandit problem. In _Proceedings of the Genetic and Evolutionary Computation Conference_, pp. 180–188, 2021. 
*   Smith et al. (2016) Smith, D., Tokarchuk, L., and Wiggins, G. Rapid phenotypic landscape exploration through hierarchical spatial partitioning. In _Parallel Problem Solving from Nature–PPSN XIV: 14th International Conference, Edinburgh, UK, September 17-21, 2016, Proceedings 14_, pp. 911–920. Springer, 2016. 
*   Spector et al. (2024) Spector, L., Ding, L., and Boldi, R. Particularity. In _Genetic Programming Theory and Practice XX_, pp. 159–176. Springer, 2024. 
*   Stiennon et al. (2020) Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P.F. Learning to summarize with human feedback. _Advances in Neural Information Processing Systems_, 33:3008–3021, 2020. 
*   Tian et al. (2020) Tian, Y., Krishnan, D., and Isola, P. Contrastive multiview coding. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XI 16_, pp. 776–794. Springer, 2020. 
*   Tishby et al. (2000) Tishby, N., Pereira, F.C., and Bialek, W. The information bottleneck method. _arXiv preprint physics/0004057_, 2000. 
*   Vassiliades & Mouret (2018) Vassiliades, V. and Mouret, J.-B. Discovering the elite hypervolume by leveraging interspecies correlation. In _Proceedings of the Genetic and Evolutionary Computation Conference_, pp. 149–156, 2018. 
*   Vassiliades et al. (2017) Vassiliades, V., Chatzilygeroudis, K., and Mouret, J.-B. Using centroidal voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm. _IEEE Transactions on Evolutionary Computation_, 22(4):623–630, 2017. 
*   Wu et al. (2021) Wu, J., Ouyang, L., Ziegler, D.M., Stiennon, N., Lowe, R., Leike, J., and Christiano, P. Recursively summarizing books with human feedback. _arXiv preprint arXiv:2109.10862_, 2021. 
*   Zhang et al. (2018) Zhang, R., Isola, P., Efros, A.A., Shechtman, E., and Wang, O. The unreasonable effectiveness of deep features as a perceptual metric. In _Proceedings of the IEEE conference on computer vision and pattern recognition_, pp. 586–595, 2018. 
*   Ziegler et al. (2019) Ziegler, D.M., Stiennon, N., Wu, J., Brown, T.B., Radford, A., Amodei, D., Christiano, P., and Irving, G. Fine-tuning language models from human preferences. _arXiv preprint arXiv:1909.08593_, 2019. 

Appendix A Preliminaries on Quality Diversity
---------------------------------------------

Quality Diversity (QD)(Mouret & Clune, [2015](https://arxiv.org/html/2310.12103v3#bib.bib36); Cully et al., [2015](https://arxiv.org/html/2310.12103v3#bib.bib15); Pugh et al., [2016](https://arxiv.org/html/2310.12103v3#bib.bib44); Lehman & Stanley, [2011b](https://arxiv.org/html/2310.12103v3#bib.bib33)) is a concept in the field of optimization and artificial intelligence that emphasizes not just finding the best possible solution to a problem (quality), but also discovering a variety of good solutions that are diverse in their characteristics (diversity). This approach is particularly valuable in complex problem-solving scenarios where there might be multiple good solutions, each with its unique benefits.

### A.1 Quality Diversity

The QD process aims to find the best representative samples, not only seeking the absolute best but also ensuring that the selections are varied and uniquely excellent in their own ways. Intuitively, imagine assembling a soccer team with QD: it meticulously recruits top-tier players across various positions to build a well-rounded team, rather than simply gathering the most renowned players regardless of their specialized roles. Key aspects of QD include:

*   •Quality: This refers to how well a solution meets the desired criteria or objectives. In QD, the aim is to identify solutions that are highly effective or optimal with respect to the goals of the task. 
*   •Diversity: Unlike traditional optimization that focuses on the single best solution, QD seeks a range of good solutions that are different from each other. This diversity can be in terms of features, approaches, or strategies the solutions employ. 
*   •Exploration and Exploitation: QD balances exploration (searching for new, diverse solutions) and exploitation (refining known good solutions). This balance helps navigate the solution space and uncover unique solutions that may be overlooked by conventional methods. 

### A.2 MAP-Elites

Algorithm 1 MAP-Elites Algorithm

1:Initialize a map of solutions, each cell representing a unique feature combination

2:while not converged do

3:Generate new solutions via mutation and crossover

4:for each solution do

5:Evaluate the solution for its performance and feature characteristics

6:Identify the corresponding cell in the map based on features

7:if solution is better than the current cell occupant then

8:Replace the cell’s solution with the new solution

9:end if

10:end for

11:end while

12:Return the map of elite solutions

MAP-Elites(Mouret & Clune, [2015](https://arxiv.org/html/2310.12103v3#bib.bib36)) stands out in evolutionary computation for its unique approach to exploring solution spaces. Unlike traditional algorithms that target a single optimal solution, MAP-Elites focuses on revealing a broad spectrum of high-performing solutions, categorized by distinct features. A high-level view of MAP-Elites, outlining its core steps, is shown in Algorithm[1](https://arxiv.org/html/2310.12103v3#alg1 "Algorithm 1 ‣ A.2 MAP-Elites ‣ Appendix A Preliminaries on Quality Diversity ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization").

The algorithm uses a grid or map where each cell corresponds to a unique combination of feature descriptors. New solutions are generated through mutation, and are evaluated for their performance and feature characteristics. The map is updated continually, with each cell holding the best-performing solution for its feature combination, ensuring a rich diversity of high-quality solutions.

MAP-Elites is particularly advantageous in domains requiring adaptability and robustness, such as robotics, or in areas where creativity and a wide range of solutions are beneficial, like design and art. It provides insights into the solution space, highlighting the relationship between different solution features and their trade-offs.

Appendix B Additional Implementation Details
--------------------------------------------

In this section, we detail our implementation and hyperparameters used in the experiments.

#### For all tasks.

The frequency of diversity representation fine-tuning in QDHF decreases exponentially over time as the learned metrics become more robust. In our experiments, the latent projection is updated 4 times at iteration 1 1 1 1, 10%⋅n⋅percent 10 𝑛 10\%\cdot n 10 % ⋅ italic_n, 25%⋅n⋅percent 25 𝑛 25\%\cdot n 25 % ⋅ italic_n, and 50%⋅n⋅percent 50 𝑛 50\%\cdot n 50 % ⋅ italic_n for a total of n 𝑛 n italic_n iterations. To fairly compare with QDHF-base, each update consumes 1/4 of the total budget of human feedback.

#### Robotic arm.

The robotic arm repertoire task is configured to have 10 degrees of freedom, i.e., the solution is a vector of 10 values, each specifying a joint angle. For QDHF and AURORA, we extract the features from the raw solution by running the accumulated sum on the solution vector and applying sin\sin roman_sin and cos\cos roman_cos transformations, resulting in a 20-dim feature vector. The latent projection in QDHF transforms the feature into a 2-dim embedding. For AURORA, the auto-encoder has an architecture of 64-32-2-32-64 neurons in each layer, where the mid-layer is the embedding used for QD. For QDHF, we use 1,000 judgments of simulated human feedback. The ground truth diversity is given by the end-point of the arm. For all experiments, we run MAP-Elites for 1000 iterations, and for each iteration, we generate a batch of 100 solutions with Gaussian mutation (adding Gaussian noises sampled from 𝒩⁢(0,0.1 2)𝒩 0 superscript 0.1 2\mathcal{N}(0,0.1^{2})caligraphic_N ( 0 , 0.1 start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT )), and evaluate them. The archive has a shape of (50,50)50 50(50,50)( 50 , 50 ), i.e., each of the 2 dimensions is discretized into 50 equal-sized bins.

#### Maze navigation.

For the maze navigation task, the robot features two contact sensors, and each of these sensors yields a value of 1 upon contact with a wall and -1 in the absence of any contact. The robot operates within a bounded square maze comprised of immovable, straight walls, which prevents the robot from moving upon collision. The episode length of the environment is 250. The solution is the network parameters of the default MLP policy network with a hidden-layer size of 8. We evaluate the policy and obtain the state descriptors. The objective is the accumulated reward at each state. For diversity measures, the ground truth diversity is the end-position of the agent, i.e., the position at the last state. For QDHF and AURORA, we extract features from the state descriptor as the x and y positions of the agent at each state. The latent projection in QDHF transforms the feature into a 2-dim embedding. For AURORA, the auto-encoder has the same architecture of 64-32-2-32-64 nodes in each layer, where the mid-layer is the embedding used for QD. For QDHF, we use 200 judgments of simulated human feedback. For all experiments, we run MAP-Elites for 1000 iterations, and for each iteration, we generate a batch of 200 solutions with Gaussian mutation (adding Gaussian noises sampled from 𝒩⁢(0,0.2 2)𝒩 0 superscript 0.2 2\mathcal{N}(0,0.2^{2})caligraphic_N ( 0 , 0.2 start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT )), and evaluate them. The archive has a shape of (50,50)50 50(50,50)( 50 , 50 ).

#### Latent space illumination.

In the LSI task, we use human judgment data from the NIGHTS dataset, and the DreamSim model to estimate image similarity, both from Fu et al. ([2023](https://arxiv.org/html/2310.12103v3#bib.bib23)). For comparisons, we implement a best-of-n approach on Stable Diffusion as the baseline, which generates a total of 2,000 images (the same number generated by QDHF) with latents randomly sampled from uniform 𝒰⁢(0,1)𝒰 0 1\mathcal{U}(0,1)caligraphic_U ( 0 , 1 ), and select a solution set of 400 images (the same number of solutions to be produced by QDHF) with the top CLIP scores. We also propose a stronger baseline featuring a heuristic (Best-of-n+), which works as follows: Starting from the second image to be generated, we sample 100 latents randomly and choose the one with a maximal l2 distance to the previously sampled latent. This approach increases the diversity of the latents and could potentially also increase the diversity of responses.

We run QDHF for 200 iterations with a batch size of 5 solutions per iteration. The solutions are generated with Gaussian mutation (adding Gaussian noises sampled from 𝒩⁢(0,0.1 2)𝒩 0 superscript 0.1 2\mathcal{N}(0,0.1^{2})caligraphic_N ( 0 , 0.1 start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT )). The archive has a shape of (20,20)20 20(20,20)( 20 , 20 ). The solution is the latent vector used as the input to Stable Diffusion, which has a shape of (4,64,64)4 64 64(4,64,64)( 4 , 64 , 64 ). We use Stable Diffusion v2.1-base, which generates images at a resolution of 512x512. The feature extractor is a CLIP model with ViT-B/16 backbone, which returns a 512-dim feature vector. QDHF learns a latent projection from 512-d to 2-d. To gather online human feedback, we use DreamSim with the DINO-ViT-B/16 backbone. The DreamSim model is trained on the NIGHTS dataset, which consists of 20k synthetic image triplets annotated with human judgments as labels. For QDHF, we use 10000 judgments of predicted human feedback. The objective is the CLIP score of the image and the text prompt. The text prompt is also input to the Stable Diffusion model to condition the generation towards more relevant content.

Appendix C Additional Results
-----------------------------

In this section, we include more experimental results to for further analysis of the proposed method.

### C.1 Robotic Arm

#### Alignment between learned and ground truth diversity.

We evaluate the alignment of diversity metrics derived by QDHF with the underlying ground truth diversity metrics. In Fig.[6](https://arxiv.org/html/2310.12103v3#A3.F6 "Figure 6 ‣ Alignment between learned and ground truth diversity. ‣ C.1 Robotic Arm ‣ Appendix C Additional Results ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization"), the archives of QDHF and AURORA-Inc (PCA) are visualized. Both AURORA and QDHF appear to effectively learn a diversity space reflective of the ground truth. However, QDHF exhibits enhanced capability in discerning the relative distances between solutions. This efficacy stems from QDHF’s ability to better align its learned diversity space with the ground truth diversity, especially concerning the scales on each axis.

![Image 7: Refer to caption](https://arxiv.org/html/2310.12103v3/extracted/5642546/figures/heat_arm.png)

Figure 6: Visualization of the solution archives on different diversity spaces for the robotic arm task. “GT” stands for ground truth. Each point on the heatmap is a solution with its objective value visualized in color. QDHF fills up the archives with more solutions than AURORA, and more accurately learns the scale of the ground truth diversity metrics.

![Image 8: Refer to caption](https://arxiv.org/html/2310.12103v3/extracted/5642546/figures/turk.jpg)

Figure 7: Example of the human evaluation interface on Amazon Mechanical Turk.

Table 5: Results for robotic arm with noisy labels. We report the QD score (normalized to a scale of 0-100) and coverage for “All Solutions” (solutions found throughout training) and “Archive Solutions” (solutions in the final archive). We additionally report the performance of QDHF at different noise levels, where 5% Noise means 5% of the preference labels being randomly sampled are flipped. QDHF’s performance is not significantly affected at the low noise level (5%), and still outperforms QDHF-Base (with perfect data) and AURORA at the highest noise level of 20%.

#### Noisy feedback.

To investigate the impact of noise in human feedback data on the performance of QDHF, we conducted an ablation study on the robotic arm task. In this study, we manually injected noise into the feedback data, which was simulated from a ground truth diversity metric. We selected three noise levels—5%, 10%, and 20%—and at each level, a corresponding percentage of the feedback data, randomly sampled from the entire dataset, was flipped. This simulates potential bias or errors by human annotators during the annotation process.

The results, as presented in Table[5](https://arxiv.org/html/2310.12103v3#A3.T5 "Table 5 ‣ Alignment between learned and ground truth diversity. ‣ C.1 Robotic Arm ‣ Appendix C Additional Results ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization"), indicate that QDHF’s performance is not significantly affected at the lower noise level of 5%. Although QDHF’s performance declines at higher noise levels, it still outperforms the baseline method (with perfect data) and all variants of AURORA, even at the highest noise level of 20%. These findings validate QDHF’s robustness against potential noise in the feedback data from humans.

### C.2 Latent Space Illumination

We include more detailed experimental setups and results for the LSI task.

#### Quantitative results on compositional prompts.

We include the quantitative evaluations of results on compositional prompts in Table[6](https://arxiv.org/html/2310.12103v3#A3.T6 "Table 6 ‣ Quantitative results on compositional prompts. ‣ C.2 Latent Space Illumination ‣ Appendix C Additional Results ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization"), which are summarized over 100 text prompts. QDHF has a similar CLIP score to SD (Best-of-n), but much higher mean and standard deviation of pairwise distance, which indicates that QDHF is able to generate more diverse solutions while maintaining the high-quality. We can see that even though the CLIP scores is high, the issue of attribute leakage or interchanged attributes still exists in the results. However, QDHF is able to mitigate this issue by exploring more diverse responses, which results in a higher chance of approaching the correct response. For a more accurate evaluation of correctness, we rely on human evaluation as shown in Table[4](https://arxiv.org/html/2310.12103v3#S5.T4 "Table 4 ‣ Open-ended tasks. ‣ 5.2 Experimental Design and Evaluation ‣ 5 Experiments ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization").

Table 6: Quantitative Results for LSI (compositional prompts). We report the CLIP score and DreamSim pairwise distance (mean and std.) as quantitative metrics. QDHF demonstrates competitive quality to Stable Diffusion (Best-of-n) measured by CLIP score, and significantly better diversity measured by pariwise distance.

#### User study details.

The LSI experiments are conducted with singular prompts and compositional prompts. We use human evaluation to assess the results through a “blind” user study, where the users are unaware of the origin of each output when making the decision. For singular prompts, we use 100 prompts and each prompt produces results of two 4x4 grids of images for SD (Best-of-N) and QDHF, respectively. We evaluate the results by asking the users to answer two questions: (1) which set of generated images is more preferred, and (2) which set of generated images is more diverse. The study is conducted on Amazon Mechanical Turk, where each evaluation task is created as a Human Intelligence Task (HIT). For each prompt, we seek 50 evaluations from different users, and a user is qualified if they have a HIT Approval Rate over 95%percent 95 95\%95 %. We follow the Amazon MTurk guidelines to fairly compensate the annotators for $0.01 per evaluation per text prompt. An example of the MTurk annotation interface is shown in Fig.[7](https://arxiv.org/html/2310.12103v3#A3.F7 "Figure 7 ‣ Alignment between learned and ground truth diversity. ‣ C.1 Robotic Arm ‣ Appendix C Additional Results ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization").

For compositional prompts, we follow a similar human evaluation procedure, and the question is: which set of generated images has the most accurate response. We also conduct experiments on 100 text prompts, sampled from the MCC-250 benchmark(Bellagente et al., [2023](https://arxiv.org/html/2310.12103v3#bib.bib3)), with 50 evaluations per prompt. The comparisons are done on 3x3 grids of images since the evaluation target is on correctness rather than diversity.

#### More qualitative results on singular prompts.

We show more qualitative results of generated images with singular prompts in Fig.[8](https://arxiv.org/html/2310.12103v3#A3.F8 "Figure 8 ‣ More qualitative results on singular prompts. ‣ C.2 Latent Space Illumination ‣ Appendix C Additional Results ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization") to Fig[11](https://arxiv.org/html/2310.12103v3#A3.F11 "Figure 11 ‣ More qualitative results on singular prompts. ‣ C.2 Latent Space Illumination ‣ Appendix C Additional Results ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization"). For all the figures, the left 4x4 grid displays images with the highest CLIP scores from randomly generated images, and the right grid displays a uniformly sampled subset of QDHF solutions. Qualitatively, images generated by QDHF have more variations and show visible trends of diversity.

Notably, while QDHF significantly outperforms the baseline on most cases, we find that Fig[11](https://arxiv.org/html/2310.12103v3#A3.F11 "Figure 11 ‣ More qualitative results on singular prompts. ‣ C.2 Latent Space Illumination ‣ Appendix C Additional Results ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization") is a sub-performing case for QDHF. Although QDHF is able to generate both day and night scenes, the diversity between the scenes is not apparent. The most likely reason is that the preference model (DreamSim) does not generalize well to cases such as different appearances of cityscapes. We aim to solve the above issues in future work where human feedback needs to be collected in a more diverse and strategic way to facilitate better generalization of the preference model and thus improve the performance of QDHF.

Another interesting finding is that for prompt Fig[8](https://arxiv.org/html/2310.12103v3#A3.F8 "Figure 8 ‣ More qualitative results on singular prompts. ‣ C.2 Latent Space Illumination ‣ Appendix C Additional Results ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization"), while most users find QDHF results are more diverse, more than half of the users actually prefer the less diverse baseline results. According to the feedback from users, people may prefer less diverse but more content-focused results in some specific cases. The relationship between diversity and user preference under different use cases in generative AI applications remains an open question, and we look forward to exploring this topic in future work.

![Image 9: Refer to caption](https://arxiv.org/html/2310.12103v3/extracted/5642546/figures/1.jpg)

Figure 8: Qualitative results for singular prompt: “a photo of an astronaut riding a horse on mars”.

![Image 10: Refer to caption](https://arxiv.org/html/2310.12103v3/extracted/5642546/figures/3.jpg)

Figure 9: Qualitative results for singular prompt: “an image of a cat on the sofa”.

![Image 11: Refer to caption](https://arxiv.org/html/2310.12103v3/extracted/5642546/figures/4.jpg)

Figure 10: Qualitative results for singular prompt: “an image of a person playing guitar”.

![Image 12: Refer to caption](https://arxiv.org/html/2310.12103v3/extracted/5642546/figures/5.jpg)

Figure 11: Qualitative results for singular prompt: “an image of urban downtown”.

#### More qualitative results on compositional prompts.

We also include more qualitative results of generated images with compositional prompts in Fig.[12](https://arxiv.org/html/2310.12103v3#A3.F12 "Figure 12 ‣ More qualitative results on compositional prompts. ‣ C.2 Latent Space Illumination ‣ Appendix C Additional Results ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization") and Fig[13](https://arxiv.org/html/2310.12103v3#A3.F13 "Figure 13 ‣ More qualitative results on compositional prompts. ‣ C.2 Latent Space Illumination ‣ Appendix C Additional Results ‣ Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization"). For all the figures, the left 3x3 grid displays images with the highest CLIP scores from randomly generated images, and the right grid displays a uniformly sampled subset of QDHF solutions. Qualitatively, images generated by QDHF have more variations and show a higher frequency of correctness.

![Image 13: Refer to caption](https://arxiv.org/html/2310.12103v3/extracted/5642546/figures/obj1.jpg)

Figure 12: Qualitative results for compositional prompt: “a red boat and a blue car”.

![Image 14: Refer to caption](https://arxiv.org/html/2310.12103v3/extracted/5642546/figures/obj2.jpg)

Figure 13: Qualitative results for compositional prompt: “a blue cake and a gold clock”.

