Title: Jointly Reinforcing Diversity and Quality in Language Model Generations

URL Source: https://arxiv.org/html/2509.02534

Published Time: Wed, 03 Sep 2025 02:21:00 GMT

Markdown Content:
♡]Meta FAIR ♣]Carnegie Mellon University ♢]Johns Hopkins University \contribution[†]Work done during an internship at Meta

Yiming Zhang Ping Yu Swarnadeep Saha Daniel Khashabi Jason Weston 

Jack Lanchantin Tianlu Wang [ [ [ [tli104@jhu.edu](mailto:tli104@jhu.edu)[tianluwang@meta.com](mailto:tianluwang@meta.com)

(September 2, 2025)

###### Abstract

Post-training of Large Language Models (LMs) often prioritizes accuracy and helpfulness at the expense of diversity. This creates a tension: while post-training improves response quality, it also sharpens output distributions and reduces the range of ideas, limiting the usefulness of LMs in creative and exploratory tasks such as brainstorming, storytelling, or problem solving. We address this challenge with Diversity-Aware Reinforcement Learning (Darling), a framework that jointly optimizes for response quality and semantic diversity. At its core, Darling introduces a learned partition function to measure diversity beyond surface-level lexical variations. This diversity signal is then combined with a quality reward during online reinforcement learning, encouraging models to generate outputs that are both high-quality and distinct. Experiments across multiple model families and sizes show that Darling generalizes to two regimes: non-verifiable tasks (instruction following and creative writing) and verifiable tasks (competition math). On five benchmarks in the first setting, Darling consistently outperforms quality-only RL baselines, producing outputs that are simultaneously of higher quality and novelty. In the second setting, it achieves higher pass@1 (solution quality) and pass@k k (solution variety). Most strikingly, explicitly optimizing for diversity catalyzes exploration in online RL, which manifests itself as higher-quality responses.

1 Introduction
--------------

Diversity plays a critical role in numerous real-world applications (Lu et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib50)), directly influencing their effectiveness, utility, and innovation potential (Nagarajan et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib55); Zhang et al., [2025b](https://arxiv.org/html/2509.02534v1#bib.bib87)). For example, in scientific discovery, diverse hypotheses or experimental outcomes enable researchers to explore a broader solution space, potentially uncovering novel insights and breakthroughs (Gruver et al., [2023](https://arxiv.org/html/2509.02534v1#bib.bib18); Romera-Paredes et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib62); Si et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib68)). Similarly, in other tasks such as creative writing (Fan et al., [2018](https://arxiv.org/html/2509.02534v1#bib.bib15)) and natural conversations (Li et al., [2016a](https://arxiv.org/html/2509.02534v1#bib.bib37)), diverse outputs are essential for innovation that requires avoiding repetitive or predictable outcomes. In reinforcement learning (RL) and self-training loops of LMs, diversity is also crucial. Policies that produce diverse outputs enable thorough exploration of the action space, critical for discovering novel and effective strategies (Chen et al., [2025a](https://arxiv.org/html/2509.02534v1#bib.bib7); Cheng et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib9); Wu et al., [2025a](https://arxiv.org/html/2509.02534v1#bib.bib76); He et al., [2025a](https://arxiv.org/html/2509.02534v1#bib.bib19)).

However, recent developments in Language Models (LMs) have revealed a significant issue: post-training of LMs often result in overly sharpened output distributions (Huang et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib23); Li et al., [2025b](https://arxiv.org/html/2509.02534v1#bib.bib41)), leading to significant reduction of diversity among generated responses (Padmakumar and He, [2024](https://arxiv.org/html/2509.02534v1#bib.bib57); Shypula et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib67)), even sharing identical prefixes (Ji et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib28)) or becoming near duplicates (Mahony et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib53); Zhang et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib86)), reducing the overall informativeness of outputs (Lin et al., [2021](https://arxiv.org/html/2509.02534v1#bib.bib44); Kirk et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib30); West and Potts, [2025](https://arxiv.org/html/2509.02534v1#bib.bib73); Yang and Holtzman, [2025](https://arxiv.org/html/2509.02534v1#bib.bib79); Yun et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib82)).

To address the loss of diversity during LM post-training, we propose Diversity-Aware Reinforcement Learning (Darling), an online RL objective that (a) measures diversity at the _semantic_ level via a learned classifier, and (b) fuses diversity and quality to condition gradient updates on “usefully different” trajectories. As illustrated in [Figure 1](https://arxiv.org/html/2509.02534v1#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations"), Darling first partitions rollouts from a single user prompt into distinct semantic clusters using a semantic classifier, capturing diversity beyond superficial lexical differences (§[3.1](https://arxiv.org/html/2509.02534v1#S3.SS1 "3.1 Partitioning the Responses into Semantic Equivalence Classes ‣ 3 Method: DARLING ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations")). It then combines (multiplies) the diversity assessment with a quality reward, amplifying the advantage of log-probabilities for responses that are both high-quality and semantically diverse (§[3.2](https://arxiv.org/html/2509.02534v1#S3.SS2 "3.2 DARLING: Diversity Aware Reinforcement Learning ‣ 3 Method: DARLING ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations")).

We validate Darling’s effectiveness and generalizability across both non-verifiable and verifiable tasks, using various language model families and sizes. Experimental results demonstrate that Darling preserves the original model’s diversity and achieves improved benchmark performance in both non-verifiable instruction-following and creative writing tasks, as well as verifiable math problems.

In summary, our contributions are three-fold:

1.   (1)We propose Darling, an RL framework that simultaneously optimizes quality and diversity, preventing diversity collapse during post-training. 
2.   (2)We demonstrate that a learned semantic classifier can serve as a scalable and generalizable signal of diversity to integrate into online RL training. 
3.   (3)We show that explicitly optimizing for diversity promotes greater exploration, often leading to improvements in quality in both non-verifiable (creative writing) and verifiable (competition math) benchmarks. 

![Image 1: Refer to caption](https://arxiv.org/html/2509.02534v1/x1.png)

Figure 1:  Diversity-Aware Reinforcement Learning (Darling): We first partition LLM generations into semantically equivalent clusters (represented by colors). While standard GRPO (Shao et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib64)) increases probabilities based on response quality only, Darling amplifies the increase in probability of diverse and high-quality responses.

2 Notations and Preliminaries
-----------------------------

Let 𝒮\mathcal{S} denote the set of natural language token sequences, a language model π(⋅∣x)\pi(\cdot\mid x) takes a token sequence x∈𝒮 x\in\mathcal{S} as its input and outputs a probability distribution over 𝒮\mathcal{S}. We denote the probability of a specific token sequence y∈𝒮 y\in\mathcal{S} as π​(y∣x)\pi(y\mid x) and denote the token at position t t as y t y^{t}. Given a reward function r:𝒮×𝒮→ℝ r:\mathcal{S}\times\mathcal{S}\rightarrow\mathbb{R} which maps a pair of natural language instructions and responses x,y∈𝒮 x,y\in\mathcal{S} to a scalar value r​(x,y)∈ℝ r(x,y)\in\mathbb{R}, LM post-training aims to solve the following KL constrained optimization problem:

max π 𝔼 x∼𝒟,y∼π(⋅∣x)[r(x,y)−β π​(y∣x)π ref​(y∣x)]=max π 𝔼 x∼𝒟,y∼π(⋅∣x)[r(x,y)]−β 𝔻 KL(π(⋅∣x)||π ref(⋅∣x)),\max_{\pi}\mathbb{E}_{x\sim\mathcal{D},y\sim\pi(\cdot\mid x)}\left[r(x,y)-\beta\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\right]=\max_{\pi}\mathbb{E}_{x\sim\mathcal{D},y\sim\pi(\cdot\mid x)}[r(x,y)]-\beta\mathbb{D}_{\text{KL}}\left(\pi(\cdot\mid x)||\pi_{\text{ref}}(\cdot\mid x)\right),(1)

where 𝒟\mathcal{D} is a dataset of prompts and π ref\pi_{\text{ref}} is a reference model from which we do not want to deviate too much, usually implemented as the LM before the optimization process. Group Relative Policy Optimization (Shao et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib64)) optimizes ([1](https://arxiv.org/html/2509.02534v1#S2.E1 "Equation 1 ‣ 2 Notations and Preliminaries ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations")) by maximizing the following objective:

𝔼 x∼𝒟,{y i}i=1 n∼π act(⋅∣x)[1 n∑i=1 n 1|y i|∑t=1|y i|(min(IS i,t⋅A i,t,clip(IS i,t,1−ε,1+ε)⋅A i,t)−β 𝔻 KL(π θ||π ref))]\mathbb{E}_{x\sim\mathcal{D},\{y_{i}\}_{i=1}^{n}\sim\pi_{\text{act}}(\cdot\mid x)}\left[\frac{1}{n}\sum_{i=1}^{n}\frac{1}{|y_{i}|}\sum_{t=1}^{|y_{i}|}\Big{(}\min\big{(}\text{IS}_{i,t}\cdot A_{i,t},\leavevmode\nobreak\ \text{clip}(\text{IS}_{i,t},1-\varepsilon,1+\varepsilon)\cdot A_{i,t}\big{)}-\beta\mathbb{D}_{\text{KL}}(\pi_{\theta}||\pi_{\text{ref}})\Big{)}\right](2)

where n n is the number of responses per prompt, and

IS i,t=π θ​(y i t∣y i<t,x i)π act​(y i t∣y i<t,x i)\text{IS}_{i,t}=\frac{\pi_{\theta}(y_{i}^{t}\mid y_{i}^{<t},x_{i})}{\pi_{\text{act}}(y_{i}^{t}\mid y_{i}^{<t},x_{i})}

is the importance sampling (Kloek and van Dijk, [1978](https://arxiv.org/html/2509.02534v1#bib.bib31)) ratio between the current policy π θ\pi_{\theta} and the actor π act\pi_{\text{act}} (the model used to generate y i y_{i}), and

A i,t=r​(x,y i)−mean j=1 n​(r​(x,y j))std j=1 n​(r​(x,y j))A_{i,t}=\frac{r(x,y_{i})-\text{mean}_{j=1}^{n}(r(x,y_{j}))}{\text{std}_{j=1}^{n}(r(x,y_{j}))}(3)

is the advantage of the response y i y_{i}, measuring how much better (or worse) is y i y_{i} over an average response, and ε\varepsilon is a hyperparameter preventing the importance sampling term IS i,t\text{IS}_{i,t} from being too large or small. GRPO and its variants (Yu et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib81); Liu et al., [2025c](https://arxiv.org/html/2509.02534v1#bib.bib47); Hu, [2025](https://arxiv.org/html/2509.02534v1#bib.bib22)) are widely adopted as some of the go-to algorithms for LM post-training (DeepSeek-AI et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib13); Liu et al., [2025a](https://arxiv.org/html/2509.02534v1#bib.bib45); Yang et al., [2025a](https://arxiv.org/html/2509.02534v1#bib.bib78)) due to its simplicity and stability. In our work, we use GRPO as our starting baseline.

3 Method: DARLING
-----------------

[Figure 1](https://arxiv.org/html/2509.02534v1#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations") illustrates our method Darling(Diversity Aware Reinforcement Learning). We first partition the responses using our developed classifier (§[3.1](https://arxiv.org/html/2509.02534v1#S3.SS1 "3.1 Partitioning the Responses into Semantic Equivalence Classes ‣ 3 Method: DARLING ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations")) that captures semantic similarity, then combine diversity and quality signals in an RL framework to generate diverse and high-quality responses (§[3.2](https://arxiv.org/html/2509.02534v1#S3.SS2 "3.2 DARLING: Diversity Aware Reinforcement Learning ‣ 3 Method: DARLING ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations")).

### 3.1 Partitioning the Responses into Semantic Equivalence Classes

![Image 2: Refer to caption](https://arxiv.org/html/2509.02534v1/x2.png)

Figure 2: Example of partitioning a group of responses into semantically equivalent subgroups and evaluating diversity. Diversity is calculated as the normalized count of responses that is distinct from a given response.

We begin by formally defining _diversity_, as used in our work: Given a pairwise distance metric d:𝒮×𝒮→ℝ+d:\mathcal{S}\times\mathcal{S}\rightarrow\mathbb{R}^{+} between two generations, and a group of n n generations y 1,⋯,y n y_{1},\cdots,y_{n} we define the diversity of a generation y i y_{i} with respect to all other generations as the average pairwise distance between y i y_{i} and all other generations y j y_{j} (j≠i j\neq i):

Div d​(y i∣y 1,⋯,y n)=1 n−1​∑j=1 j≠i n d​(y i,y j).\text{Div}_{d}(y_{i}\mid y_{1},\cdots,y_{n})=\frac{1}{n-1}\sum_{\begin{subarray}{c}j=1\\ j\neq i\end{subarray}}^{n}d(y_{i},y_{j}).(4)

We aim to incorporate a scalable metric of semantic diversity that captures deeper differences beyond surface-level variation into our training process. Following Zhang et al. ([2025b](https://arxiv.org/html/2509.02534v1#bib.bib87)), we train a binary classifier to determine whether two responses convey equivalent semantics:

classify​(y i,y j)=𝟏​(y i​semantically equivalent to​y j).\textbf{classify}(y_{i},y_{j})=\mathbf{1}(y_{i}\text{ semantically equivalent to }y_{j}).

Responses predicted as equivalent are clustered to form a partition of all responses into semantic clusters, where multiple members provide little additional value beyond a single representative.

We directly set diversity metric d=classify​(⋅,⋅)d=\textbf{classify}(\cdot,\cdot). [Figure 2](https://arxiv.org/html/2509.02534v1#S3.F2 "Figure 2 ‣ 3.1 Partitioning the Responses into Semantic Equivalence Classes ‣ 3 Method: DARLING ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations") illustrates an example of our diversity calculation from partitions. For a single prompt _“Write a short joke about programming.”_, responses in the left column (blue) are classified as semantically equivalent, both utilizing that the word “bug” has multiple meanings. The responses in the right column (purple and yellow) are distinct from the three other responses. For each of the individual responses in blue boxes, there are only two other responses that are distinct: purple and yellow. Therefore, using ([4](https://arxiv.org/html/2509.02534v1#S3.E4 "Equation 4 ‣ 3.1 Partitioning the Responses into Semantic Equivalence Classes ‣ 3 Method: DARLING ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations")), we derive the diversity of both blue responses as 2/3 2/3. Similarly, the yellow and purple responses have diversity 3/3 3/3 because they are distinct from all other responses.

### 3.2 DARLING: Diversity Aware Reinforcement Learning

Given a diversity function Div d\text{Div}_{d} and a reward function r r, we define the diversity-aware reward r darling r_{\text{darling}} as

r darling​(x,y i∣y 1,⋯,y n):=r​(x,y i)×Norm​(Div d​(y i∣y 1,⋯,y n)),r_{\text{darling}}(x,y_{i}\mid y_{1},\cdots,y_{n}):=r(x,y_{i})\times\text{Norm}\big{(}\text{Div}_{d}(y_{i}\mid y_{1},\cdots,y_{n})\big{)},(5)

where Norm​(⋅)\text{Norm}(\cdot) normalizes diversity values to be between 0 and 1.

We choose to multiply the two reward scores instead of adding them. While simply adding the quality and diversity rewards is an alternative approach, this method poses challenges due to the differing scales of the two rewards. Naively summing the two rewards can lead the model to prioritize one reward over the other. An ablation study of varying methods for fusing reward scores is provided in §[6.1](https://arxiv.org/html/2509.02534v1#S6.SS1 "6.1 Ablations on Multiplicative v.s. Additive Aggregation ‣ 6 Ablations ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations").

Darling plugs ([5](https://arxiv.org/html/2509.02534v1#S3.E5 "Equation 5 ‣ 3.2 DARLING: Diversity Aware Reinforcement Learning ‣ 3 Method: DARLING ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations")) into ([1](https://arxiv.org/html/2509.02534v1#S2.E1 "Equation 1 ‣ 2 Notations and Preliminaries ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations")), which amplifies the effective reward r darling r_{\text{darling}} of high reward responses that are diverse from others. Motivated by prior work (Liu et al., [2025c](https://arxiv.org/html/2509.02534v1#bib.bib47); Yu et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib81)), we also make the following modifications: changing sequence-level loss averaging to token-level averaging in ([2](https://arxiv.org/html/2509.02534v1#S2.E2 "Equation 2 ‣ 2 Notations and Preliminaries ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations")) as the former exhibits bias towards longer sequences, and removing normalization by standard deviation in ([3](https://arxiv.org/html/2509.02534v1#S2.E3 "Equation 3 ‣ 2 Notations and Preliminaries ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations")) since it amplifies the noise in dense rewards. We leave detailed ablations on the effect of normalization to §[6.3](https://arxiv.org/html/2509.02534v1#S6.SS3 "6.3 Ablations on Advantage Normalization in GRPO ‣ 6 Ablations ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations").

The overall loss function for Darling is thus defined as:

𝔼 x∼𝒟,{y i}i=1 n∼π act(⋅∣x)[1∑i=1 n|y i|∑t=1|y i|(min(IS i,t⋅A i,t,clip(IS i,t,1−ε,1+ε)⋅A i,t)−β 𝔻 KL(π θ||π ref))],\mathbb{E}_{x\sim\mathcal{D},\{y_{i}\}_{i=1}^{n}\sim\pi_{\text{act}}(\cdot\mid x)}\left[\frac{1}{\sum_{i=1}^{n}|y_{i}|}\sum_{t=1}^{|y_{i}|}\Big{(}\min\big{(}\text{IS}_{i,t}\cdot A_{i,t},\leavevmode\nobreak\ \text{clip}(\text{IS}_{i,t},1-\varepsilon,1+\varepsilon)\cdot A_{i,t}\big{)}-\beta\mathbb{D}_{\text{KL}}(\pi_{\theta}||\pi_{\text{ref}})\Big{)}\right],(6)

where we use the diversity aware reward r darling r_{\text{darling}} as the effective reward:

A i,t=r darling​(x,y i∣y 1,⋯,y n)−mean j=1 n​(r darling​(x,y j∣y 1,⋯,y n)).A_{i,t}=r_{\text{darling}}(x,y_{i}\mid y_{1},\cdots,y_{n})-\text{mean}_{j=1}^{n}\big{(}{r_{\text{darling}}(x,y_{j}\mid y_{1},\cdots,y_{n})}\big{)}.

Compared to standard GRPO, our main modification is that we multiply a normalized diversity reward Norm​(Div d​(y i∣y 1,⋯,y n))\text{Norm}\big{(}\text{Div}_{d}(y_{i}\mid y_{1},\cdots,y_{n})\big{)} by the quality reward r​(x,y)r(x,y) to promote high-quality and diverse rewards during training. This amplifies the increase in the log-likelihood of responses that are both of high-quality and diverse — jointly reinforcing quality and diversity.

4 DARLING on Non-verifiable Tasks
---------------------------------

We first show the experimental effectiveness of Darling on general non-verifiable instruction following tasks. We describe our setup in §[4.1](https://arxiv.org/html/2509.02534v1#S4.SS1 "4.1 Setup ‣ 4 DARLING on Non-verifiable Tasks ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations"). We show and analyze our results in §[4.2](https://arxiv.org/html/2509.02534v1#S4.SS2 "4.2 Experimental Results ‣ 4 DARLING on Non-verifiable Tasks ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations").

### 4.1 Setup

Models and Baselines We use Llama-3.1-8B-Instruct and Llama-3.3-70B-Instruct(Llama Team, [2024](https://arxiv.org/html/2509.02534v1#bib.bib49)) as reference models and perform training on top of them. We train on a randomly sampled subset of 10k prompts in WildChat(Zhao et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib88)), which is the same setup as used in Lanchantin et al. ([2025a](https://arxiv.org/html/2509.02534v1#bib.bib34)). We use Nexusflow/Athene-RM-8B(Frick et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib16)) as the reward function in ([5](https://arxiv.org/html/2509.02534v1#S3.E5 "Equation 5 ‣ 3.2 DARLING: Diversity Aware Reinforcement Learning ‣ 3 Method: DARLING ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations")) for quality. We use a batch size of {32 (8B), 64 (70B)} prompts ×\times 8 rollouts per prompt, and a max rollout length of 1024 tokens. Other training hyperparameters can be found in [section 10](https://arxiv.org/html/2509.02534v1#S10 "10 Hyperparameters ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations").

We compare our method against the following baselines:

*   •GRPO (Shao et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib64)): The standard GRPO method described in §[1](https://arxiv.org/html/2509.02534v1#S2.E1 "Equation 1 ‣ 2 Notations and Preliminaries ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations"), with token-level mean aggregation; 
*   •DivPO (Lanchantin et al., [2025a](https://arxiv.org/html/2509.02534v1#bib.bib34)): a DPO-based (Rafailov et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib61)) optimization method that selects the _most_ diverse response among the high-quality ones as the chosen response and the _least_ diverse response as the rejected response; 
*   •GRPO-Unlikeliness (He et al., [2025a](https://arxiv.org/html/2509.02534v1#bib.bib19)): a revised GRPO algorithm that re-weights the rewards of responses according to their likelihood. Responses that have a low likelihood receive a higher reward. 

We implement Darling using the verl codebase (Sheng et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib65)), using vLLM (Kwon et al., [2023](https://arxiv.org/html/2509.02534v1#bib.bib33)) for inference and FSDP (Zhao et al., [2023](https://arxiv.org/html/2509.02534v1#bib.bib89)) for training. The original classifier of Zhang et al. ([2025b](https://arxiv.org/html/2509.02534v1#bib.bib87)) was limited to a context length of 512 tokens 1 1 1 https://github.com/novelty-bench/novelty-bench/blob/main/src/partition.py#L69. In our work, we extend their method by training a classifier with an 8192-token context window, using the same human-annotated data. Details of this training procedure are provided in Appendix [9.1](https://arxiv.org/html/2509.02534v1#S9.SS1 "9.1 Classifier for Non-verifiable Tasks ‣ 9 Partitioning the Responses ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations").

Evaluation Benchmarks and Metrics For evaluating response quality, we employ standard benchmarks: AlpacaEval 2.0 (Li et al., [2023](https://arxiv.org/html/2509.02534v1#bib.bib40); Dubois et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib14)), ArenaHard v2.0 (Li et al., [2025a](https://arxiv.org/html/2509.02534v1#bib.bib39)), and EQ-Bench (Creative Writing) (Paech, [2023](https://arxiv.org/html/2509.02534v1#bib.bib59)). We report the length-controlled win rate (LCWR) for AlpacaEval 2.0 and the win rate with style control (markdown, length) for ArenaHard v2.0 on the creative writing prompts. We report the normalized ELO score for EQ-Bench. For both AlpacaEval and ArenaHard, we follow (Lanchantin et al., [2025b](https://arxiv.org/html/2509.02534v1#bib.bib35)) and use GPT-4o (OpenAI et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib56)) as the judge. For EQ-bench, we use Claude 3.7 Sonnet (Anthropic, [2024](https://arxiv.org/html/2509.02534v1#bib.bib3)) as the judge. For evaluating diversity, we use NoveltyBench (Zhang et al., [2025b](https://arxiv.org/html/2509.02534v1#bib.bib87)). We report the number of _semantically_ distinct generations (Distinct) and the average number of distinct 4grams (Distinct-4) normalized by length. We provide detailed descriptions of the benchmarks in [section 12](https://arxiv.org/html/2509.02534v1#S12 "12 Benchmark Descriptions ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations").

### 4.2 Experimental Results

AE 2.0*AH v2.0*AH v1.0*EQ-Bench NoveltyBench
LCWR (%)WR (%)WR (%)ELO Distinct (#)Distinct-4 (%)
Llama-3.1-8B-Instruct 31.9 7.1 30.9 636 5.28 93.9
GRPO 48.7 61.1 45.5 659 2.08 92.8
DivPO 43.5 54.4 39.7 639 4.34 94.1
GRPO-Unlikeliness 45.6 59.5 46.2 724 3.53 93.2
Darling 55.2 68.8 63.7 905 5.49 96.0
Llama-3.3-70B-Instruct 44.6 17.7 64.9 737 2.95 91.7
GRPO 73.3 89.7 79.2 1261 2.31 94.6
GRPO-Unlikeliness 69.5 84.2 76.4 1346 3.15 95.2
Darling 80.4 91.2 85.7 1531 4.26 95.3

Table 1: Non-verifiable Task Evaluations. For each method, we train a single model on 10,000 WildChat prompts. We evaluate the models on AE (AlpacaEval 2.0 Length-Controlled Win Rate), AH v2.0/v1.0 (ArenaHard, creative writing subset), EQ-Bench (ELO), and NoveltyBench. * indicates we used GPT-4o as the judge. All metrics are the higher the better. We find that models trained with Darling achieve the best quality measured by both AlpacaEval/ArenaHard win rates and EQ-Bench ELO, and simultaneously are the most diverse, as measured by NoveltyBench.

DARLING achieves both the best quality and diversity across all benchmarks.[Table 1](https://arxiv.org/html/2509.02534v1#S4.T1 "Table 1 ‣ 4.2 Experimental Results ‣ 4 DARLING on Non-verifiable Tasks ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations") shows our main results: we observe that Darling is able to jointly optimize for both quality and diversity. Specifically, Darling results in the best quality scores (AlpacaEval and ArenaHard win rates) across our baselines, while also achieving the best diversity in both semantic level (Distinct) and lexical level (Distinct-4), showcasing the effectiveness of our method. Moreover, although we did not explicitly train on creative writing prompts, Darling achieves the best ELO score compared to all baselines in EQ-Bench (creative writing), demonstrating the effectiveness of improving diversity on creative tasks.

DARLING improves the pareto front between quality and diversity by varying sampling temperature. We further investigate the effect of sampling temperature on the quality-diversity pareto front. We vary the sampling temperature (T={0.2,0.4,0.6,0.8,1.2}T=\{0.2,0.4,0.6,0.8,1.2\}) of two models (Llama-3.1-8B-Instruct and Llama-3.3-70B-Instruct), after being trained with GRPO and Darling. [Figure 3](https://arxiv.org/html/2509.02534v1#S4.F3 "Figure 3 ‣ 4.2 Experimental Results ‣ 4 DARLING on Non-verifiable Tasks ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations") shows that Darling(blue) exhibits both better quality and better diversity than both the baseline (green) and GRPO (orange) at both scales, pushing forward the pareto-front of the “quality-diversity tradeoff” (Zhang et al., [2021](https://arxiv.org/html/2509.02534v1#bib.bib84); Padmakumar et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib58)).

![Image 3: Refer to caption](https://arxiv.org/html/2509.02534v1/x3.png)

Figure 3: The quality-diversity tradeoff when using different sampling temperatures (T T) for models (at 8B and 70B scales) trained with standard GRPO and Darling. X X-axis: Distinct metric in NoveltyBench; Y Y-axis: Reward score used in NoveltyBench measuring quality of responses. Darling (blue) simultaneously achieves better quality (y-axis) and diversity (x-axis) as demonstrated by the improved Pareto fronts on both the 8B and 70B scale.

### 4.3 Qualitative Analysis

![Image 4: Refer to caption](https://arxiv.org/html/2509.02534v1/x4.png)

Figure 4: Detailed win rates of the top-3 and the bottom-3 rubrics of Llama-3.1-8B-Instruct trained with Darling against models with similar ELO points. Darling s strength lies in being "Interesting and Original" and "Avoids Cliche" due to being able to generate creative responses.

We show qualitative analysis on EQ-Bench (Paech, [2023](https://arxiv.org/html/2509.02534v1#bib.bib59)). EQ-bench provides detailed evaluation rubrics and asks Claude-3.7-Sonnet to score model generations according to these rubrics. We breakdown the rubrics where Darling has the most and least win rates over models with similar ELO in [Figure 4](https://arxiv.org/html/2509.02534v1#S4.F4 "Figure 4 ‣ 4.3 Qualitative Analysis ‣ 4 DARLING on Non-verifiable Tasks ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations"). Darling’s strength lies in being able to generate diverse outputs, thus it wins the most on being “Interesting and Original” and “Avoids Cliche”. We show the output for a creative writing prompt generated by our model trained with Darling in Appendix [11.1](https://arxiv.org/html/2509.02534v1#S11.SS1 "11.1 Example Generation in EQBench ‣ 11 Generation Examples ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations").

Figure 5: Example generations of Llama-3.3-70B-Instruct before and after Darling training. We sample 4 parallel generations with temperature=1.0 for both models. Models trained with Darling exhibit better diversity.

We further illustrate these findings with qualitative examples from NoveltyBench (Zhang et al., [2025b](https://arxiv.org/html/2509.02534v1#bib.bib87)), shown in [Figure 5](https://arxiv.org/html/2509.02534v1#S4.F5 "Figure 5 ‣ 4.3 Qualitative Analysis ‣ 4 DARLING on Non-verifiable Tasks ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations") and Appendix [11.2](https://arxiv.org/html/2509.02534v1#S11.SS2 "11.2 Example generations in NoveltyBench ‣ 11 Generation Examples ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations"). Across four parallel generations, models trained with Darling consistently exhibit higher semantic diversity. Even when repetitions occur—for example, in the second and fourth generations of [Figure 5](https://arxiv.org/html/2509.02534v1#S4.F5 "Figure 5 ‣ 4.3 Qualitative Analysis ‣ 4 DARLING on Non-verifiable Tasks ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations")—the outputs remain meaningfully distinct: both suggest Bellroy, yet each provides a different rationale. A similar trend appears in the examples in Appendix [11.2](https://arxiv.org/html/2509.02534v1#S11.SS2 "11.2 Example generations in NoveltyBench ‣ 11 Generation Examples ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations"), where Darling-trained models not only produce more diverse generations overall, but also introduce variation in their explanations when repeating simple outputs such as random numbers or animals.

5 DARLING on Verifiable Tasks
-----------------------------

In this section we present experimental results of Darling on verifiable math problems. We describe our setup in §[5.1](https://arxiv.org/html/2509.02534v1#S5.SS1 "5.1 Setup ‣ 5 DARLING on Verifiable Tasks ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations") and show our results in §[5.2](https://arxiv.org/html/2509.02534v1#S5.SS2 "5.2 Experimental Results ‣ 5 DARLING on Verifiable Tasks ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations").

### 5.1 Setup

Models, Baselines, and Benchmarks We train models on top of Qwen3-4B-Base and Qwen3-14B-Base(Yang et al., [2025a](https://arxiv.org/html/2509.02534v1#bib.bib78)) using the the DeepscaleR dataset (Luo et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib52)), where we first filter out questions that are unanswerable due to missing figures, and then subsample 10,000 examples. We compare our method against GRPO (Shao et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib64)) on four competition math benchmarks: AIME25 (Art of Problem Solving, [2025](https://arxiv.org/html/2509.02534v1#bib.bib4)), OlympiadBench (He et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib20)), HMMT 2025 (Balunović et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib5)), and Brumo 2025 (Balunović et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib5)). We delibrately choose these benchmarks because they were released concurrently with Qwen3, preventing the effect of potential data contamination. We use the Hugging Face Math-Verify 2 2 2 https://github.com/huggingface/Math-Verify library to automatically check the correctness of model answers, assigning a binary reward of r=1 r=1 for correct and r=0 r=0 for incorrect solutions. We report pass@1 scores as a measure of quality and pass@k k scores as a measure of diversity. To evaluate pass@k k up to k=128 k=128, we sample n=256 n=256 responses for each prompt, and we average the performance of the 256 256 examples for calculating pass@1 to account for the variance introduced by the relatively small sizes of these benchmarks. We use the method in Chen et al. ([2021](https://arxiv.org/html/2509.02534v1#bib.bib6)) for an unbiased estimate of pass@k k from n=256 n=256 examples:

pass​@​k:=𝔼​[1−(n−c k)(n k)],\text{pass}@k:=\mathbb{E}\left[1-\frac{\binom{n-c}{k}}{\binom{n}{k}}\right],(7)

where c c is the number of correct generations. With increased diversity, we expect to see an improved performance of pass@k k as the model is more likely to hit the correct answer when it generates more diverse responses. Additional hyperparameters can be found at [section 10](https://arxiv.org/html/2509.02534v1#S10 "10 Hyperparameters ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations").

Training an equivalence classifier for math For building the diversity classifier, Zhang et al. ([2025b](https://arxiv.org/html/2509.02534v1#bib.bib87)) collected annotated training data which are sampled prompts from WildChat (Zhao et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib88)) filtered for non-verifiable tasks. To adapt their method to math, we sample prompts from DeepscaleR and collect trajectories generated by 8 different models spanning multiple model families and sizes. We annotate whether a pair of trajectories is semantically equivalent using Llama-3.3-70B-Instruct(Llama Team, [2024](https://arxiv.org/html/2509.02534v1#bib.bib49)). We then finetune a Qwen3-Embedding-4B(Zhang et al., [2025a](https://arxiv.org/html/2509.02534v1#bib.bib85)) model using the annotations to obtain our math semantic equivalence classifier. Details on how we perform trajectory sampling, annotations, and classifier training can be found in Appendix [9.2](https://arxiv.org/html/2509.02534v1#S9.SS2 "9.2 Classifier for Verifiable Tasks ‣ 9 Partitioning the Responses ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations").

### 5.2 Experimental Results

DARLING improves both pass@1 and pass@k in competition math.[Figure 6](https://arxiv.org/html/2509.02534v1#S5.F6 "Figure 6 ‣ 5.2 Experimental Results ‣ 5 DARLING on Verifiable Tasks ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations") shows our main results: we plot pass@k k from k=1 to k=128. We observe that Darling outperforms the GRPO baseline in both quality and diversity. First, for pass@1 (as a measure of quality), Darling outperforms GRPO by +3.51/1.90% averaged across 4 benchmarks for 4B and 14B models respectively. Next, for pass@128 (as a measure of diversity), Darling outperforms GRPO by +7.62/10.16%. This shows that by jointly reinforcing quality and diversity, Darling is able to achieve the best of both worlds in competition math benchmarks — simultaneously achieving better pass@1 and pass@k k. Furthermore, we observe the largest gains on HMMT, the most challenging of the four datasets, suggesting that enhanced exploration yields greater improvements on harder datasets. We report accuracy for each of the datasets in Appendix [13](https://arxiv.org/html/2509.02534v1#S13 "13 Full Results on Math ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations").

![Image 5: Refer to caption](https://arxiv.org/html/2509.02534v1/x5.png)

Figure 6: Comparison of different pass@k k values by applying GRPO and Darling on Qwen3-4B-Base and Qwen3-14B-Base on competition math benchmarks. Darling outperforms GRPO simultaneously for pass@1 (+3.51/+1.9% on avg.) and pass@128 (+7.62/10.16% on avg.) on 4B and 14B models respectively. Darling simultaneously achieves the best quality and diversity averaged across 4 competition math benchmarks.

6 Ablations
-----------

In this section, we perform ablations on the design choices of Darling. In particular, we compare additive aggregation of quality and diversity rewards v.s. multiplicative aggregation (ours) in §[6.1](https://arxiv.org/html/2509.02534v1#S6.SS1 "6.1 Ablations on Multiplicative v.s. Additive Aggregation ‣ 6 Ablations ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations"). We compare our partition classifier which measures semantic diversity with traditional lexical diversity in §[6.2](https://arxiv.org/html/2509.02534v1#S6.SS2 "6.2 Ablations on Lexical Metrics for Diversity ‣ 6 Ablations ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations") and we ablate the effect of the standard deviation term in GRPO normalization ([3](https://arxiv.org/html/2509.02534v1#S2.E3 "Equation 3 ‣ 2 Notations and Preliminaries ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations")) in §[6.3](https://arxiv.org/html/2509.02534v1#S6.SS3 "6.3 Ablations on Advantage Normalization in GRPO ‣ 6 Ablations ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations").

### 6.1 Ablations on Multiplicative v.s. Additive Aggregation

A naive aggregation of the quality reward r r and the diversity signal Div d\text{Div}_{d} is to add the two rewards, as many existing works perform additive aggregation of the quality-based reward and auxiliary rewards such as length (Aggarwal and Welleck, [2025](https://arxiv.org/html/2509.02534v1#bib.bib1); Liu et al., [2025b](https://arxiv.org/html/2509.02534v1#bib.bib46)), entropy (Cheng et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib9)), and format (Wu et al., [2025b](https://arxiv.org/html/2509.02534v1#bib.bib77)).

We ablate the effect of adding or multiplying the quality reward r r and the diversity reward Norm(Div) in equation [5](https://arxiv.org/html/2509.02534v1#S3.E5 "Equation 5 ‣ 3.2 DARLING: Diversity Aware Reinforcement Learning ‣ 3 Method: DARLING ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations") in our non-verifiable setting and report the results in [Table 2](https://arxiv.org/html/2509.02534v1#S6.T2 "Table 2 ‣ 6.1 Ablations on Multiplicative v.s. Additive Aggregation ‣ 6 Ablations ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations"). To aggregate the two rewards, we manually normalize (subtract mean and divide by standard deviation) both r r (the quality reward) and Div d\text{Div}_{d}.

AlpacaEval 2.0 ArenaHard v2.0 NoveltyBench
LCWR(%)WR(%)creative writing hard prompts Distinct (#)
Llama-3.1-8B-Instruct 31.92 32.61 7.1 (-1.4 / +1.8)6.4 (-0.8 / +1.3)5.28
Quality only (GRPO)48.74 57.01 61.1 (-3.5 / +4.5)33.9(-2.3 / +2.5)2.08
Quality + partition 53.17 60.82 69.2(-3.6 / +3.9)32.7 (-3.2 / +2.9)5.23
Darling=Quality ×\times partition 55.15 65.34 68.8 (-3.3 / +2.9)31.1 (-2.0 / +2.1)5.49

Table 2:  Ablation comparing the way we aggregate the quality and diversity reward: additive v.s. multiplicative in [Equation 5](https://arxiv.org/html/2509.02534v1#S3.E5 "Equation 5 ‣ 3.2 DARLING: Diversity Aware Reinforcement Learning ‣ 3 Method: DARLING ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations"), evaluating on AlpacaEval 2.0, ArenaHard v2.0 (with Style Control), and NoveltyBench.

We observe that multiplicative aggregation (Darling) outperforms additive aggregation in AlpacaEval 2.0, and performs similarly in ArenaHard v2.0 and NoveltyBench. We opt for multiplicative aggregation due to its simplicity as it does not require additional handling of mismatched reward scales and hyperparameter tuning of mixing weights on individual rewards.

### 6.2 Ablations on Lexical Metrics for Diversity

We study whether our proposed partition classifier can be replaced by a simple lexical diversity metric. We replace our semantic equivalence classifier with the number of distinct N N-grams in online RL training. Specifically, we set the diversity of a response y i y_{i} w.r.t all other responses for the same input prompt as:

Div ngram​(y i∣y 1,⋯,y n)=number of distinct N-grams that only appear in​y i total number of N-grams in​y i.\text{Div}_{\text{ngram}}(y_{i}\mid y_{1},\cdots,y_{n})=\frac{\text{number of distinct $N$-grams that only appear in\leavevmode\nobreak\ }y_{i}}{\text{total number of $N$-grams in\leavevmode\nobreak\ }y_{i}}.

This means that if no N N-gram in y i y_{i} appears in any other response y j​(j≠i)y_{j}(j\neq i), then the diversity Div N-gram​(y i∣y 1,⋯,y n)=1\text{Div}_{\text{$N$-gram}}(y_{i}\mid y_{1},\cdots,y_{n})=1. Similarly, Div N-gram​(y i∣y 1,⋯,y n)=0\text{Div}_{\text{$N$-gram}}(y_{i}\mid y_{1},\cdots,y_{n})=0 if all N N-grams in y i y_{i} appear in at least one other response. In our experiments, we set N=4 N=4 and denote this setting as “Quality ×\times 4gram”. We report experimental results on non-verifiable tasks in [Table 3](https://arxiv.org/html/2509.02534v1#S6.T3 "Table 3 ‣ 6.2 Ablations on Lexical Metrics for Diversity ‣ 6 Ablations ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations").

AlpacaEval 2.0 ArenaHard v2.0 NoveltyBench
LCWR(%)WR(%)creative writing hard prompts Distinct (#)
Llama-3.1-8B-Instruct 31.92 32.61 7.1 (-1.4 / +1.8)6.4 (-0.8 / +1.3)5.28
Quality only (GRPO)48.74 57.01 61.1 (-3.5 / +4.5)33.9(-2.3 / +2.5)2.08
Quality ×\times 4gram 53.82 66.46 71.9(-3.3 / +3.6)31.3 (-2.3 / +2.9)3.59
Darling=Quality ×\times partition 55.15 65.34 68.8 (-3.3 / +2.9)31.1 (-2.0 / +2.1)5.49

Table 3: Comparison of N N-gram diversity loss to Darling. The N N-gram diversity loss (N=4) performs similarly to Darling in terms of quality, but underperforms Darling in terms of diversity in NoveltyBench.

We observe that while 4gram diversity integrated with quality is able to match the performance of Darling in LM-as-a-Judge evaluations (AlpacaEval 2.0, ArenaHard v2.0), it significantly underperforms Darling in semantic diversity assessment (NoveltyBench).

Additionally, we evaluate the performance of using 4gram diversity in competition math and report the results in [Table 4](https://arxiv.org/html/2509.02534v1#S6.T4 "Table 4 ‣ 6.2 Ablations on Lexical Metrics for Diversity ‣ 6 Ablations ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations"). We found that in math questions, using lexical diversity as a reward underperforms the GRPO baseline in terms of pass@1 performance. We analyze the the generations and observe that the policy often hacks the ngram diversity reward by generating texts that are of a different language, or self-reflections about the difficulty of the problem. We provide an example of such ngram reward hacking in [section 15](https://arxiv.org/html/2509.02534v1#S15 "15 Diversity Reward Hacking ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations").

Pass@128 128 Pass@1
Model AIME HMMT Olympiad Brumo Avg.AIME HMMT Olympiad Brumo Avg.
Qwen3-4B-Base 47.35 27.12 71.11 55.10 50.17 8.17 1.28 31.13 16.68 14.32
Quality only (GRPO)53.33 26.72 70.37 63.24 53.42 19.51 7.14 42.27 24.66 23.40
Quality ×\times 4gram 57.47 32.35 67.47 60.55 54.46 17.44 6.95 40.03 25.55 22.49
Darling=Quality ×\times partition 62.28 39.19 74.41 68.27 61.04 20.06 10.32 45.53 31.73 26.91

Table 4: Comparison of n-gram diversity loss to Darling on Competition Math. Using 4gram as the diversity reward underperforms the baseline GRPO (no diversity reward), indicating that lexical diversity reward can harm performance in Competition Math tasks.

### 6.3 Ablations on Advantage Normalization in GRPO

Liu et al. ([2025c](https://arxiv.org/html/2509.02534v1#bib.bib47)) show that in equation ([3](https://arxiv.org/html/2509.02534v1#S2.E3 "Equation 3 ‣ 2 Notations and Preliminaries ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations")), dividing by std j=1 n​(r​(x,y j))\text{std}_{j=1}^{n}(r(x,y_{j})) effectively upweights prompts whose rewards have low variance (e.g., when rewards are nearly all 1 or all 0). We extend their analysis to a more general setting where rewards are arbitrary scalars, as is typical in Bradley–Terry style reward models.

Formally, let the reward for response y i y_{i} be

r i=f i+ε i,r_{i}=f_{i}+\varepsilon_{i},

where f i f_{i} is the true underlying utility and ε i\varepsilon_{i} is noise with variance τ 2\tau^{2}. GRPO with normalization computes

r^i=r i−r¯σ r,σ r 2≈Var​(f)+τ 2,\hat{r}_{i}=\frac{r_{i}-\bar{r}}{\sigma_{r}},\quad\sigma_{r}^{2}\approx\text{Var}(f)+\tau^{2},

so each prompt contributes unit variance to the gradient update. This has the effect of amplifying noise when τ 2\tau^{2} is large relative to Var​(f)\text{Var}(f) (dense but noisy rewards), because even very small differences between responses get magnified into values of order one. By contrast, removing the normalization yields

r~i=r i−r¯,\tilde{r}_{i}=r_{i}-\bar{r},

which preserves the true scale of reward differences. Thus, normalization is helpful when rewards are reliable (high signal-to-noise ratio), but harmful when they are noisy and tightly clustered.

Empirically, we find that removing the standard deviation term improves performance in settings with dense and noisy rewards. [Table 5](https://arxiv.org/html/2509.02534v1#S6.T5 "Table 5 ‣ 6.3 Ablations on Advantage Normalization in GRPO ‣ 6 Ablations ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations") shows results in our non-verifiable setting with a Bradley–Terry style reward: removing normalization (“w/o norm”) improves not only quality (AlpacaEval and Arena-Hard win rates) but also diversity (NoveltyBench Distinct and distinct n-grams).

AlpacaEval 2.0*ArenaHard v2.0*NoveltyBench
LCWR(%)WR(%)Creative Writing (%)Distinct (#)Distinct-4 (%)
GRPO 48.74 57.01 61.1 (-3.5 / +4.5)2.08 92.84
GRPO (w/o norm)52.57 (+3.83)61.18 (+4.17)68.1 (-3.5 / +2.7)(+7.0)2.28 (+0.20)94.05 (+1.21)
4gram 48.48 57.76 65.3 (-3.3 / +3.6)2.79 93.87
4gram (w/o norm)53.82 (+5.34)66.46 (+8.70)71.9 (-3.3 / +3.6)(+6.6)3.59 (+0.80)95.63 (+1.76)
partition 51.64 62.17 69.7 (-3.3 / +4.0)3.35 94.93
Darling= partition (w/o norm)55.15 (+3.51)65.34 (+3.17)68.8 (-3.3 / +2.9)(-0.9)5.49 (+2.14)96.04 (+1.11)

Table 5: Ablation study on normalization: Results for GRPO baseline, 4-gram, and partition mixing, each with and without normalization. All metrics are the higher the better. * indicates GPT-4o was used as the judge. Removing normalization (w/o norm) prevents the amplification of tiny differences in dense rewards, resulting in improved performance on both quality and diversity metrics.

In contrast, in settings where the reward is sparse and noise-free, normalization has little effect. [section 14](https://arxiv.org/html/2509.02534v1#S14 "14 Removing Normalization in Math Tasks ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations") reports results on Math, where rewards are binary (0,1 0,1) and deterministic. In this case, the variance comes entirely from the true differences, so normalization is neither helpful nor harmful.

7 Related Work
--------------

In this section, we provide an overview of previous work that elicits diverse responses from LMs during _training_ and _inference_, and clarify the distinction from our work. We defer additional related work on diversity evaluation metrics and RL for LMs to [section 16](https://arxiv.org/html/2509.02534v1#S16 "16 Additional Related Works ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations"). For a more comprehensive survey on LM creativity, we also refer readers to Ismayilzada et al. ([2024](https://arxiv.org/html/2509.02534v1#bib.bib26)).

Training-time strategies for diversity Neural language models often generate repetitive outputs, a long-standing challenge in the community(Li et al., [2016b](https://arxiv.org/html/2509.02534v1#bib.bib38); Zhang et al., [2021](https://arxiv.org/html/2509.02534v1#bib.bib84)). Prior work addresses this by modifying the maximum likelihood training objective to encourage diversity. For example, Li et al. ([2016b](https://arxiv.org/html/2509.02534v1#bib.bib38)) maximize mutual information to avoid generic responses (e.g., _I don’t know_). Welleck et al. ([2020](https://arxiv.org/html/2509.02534v1#bib.bib72)) penalize repetitions to improve lexical variety within a response. Other approaches smoothen or modify the one-hot target distribution: Li et al. ([2020](https://arxiv.org/html/2509.02534v1#bib.bib42)) introduce a Gaussian prior, Zhang et al. ([2024](https://arxiv.org/html/2509.02534v1#bib.bib86)) match outputs to high-entropy distributions, and Li et al. ([2025b](https://arxiv.org/html/2509.02534v1#bib.bib41)) apply sparse logit updates. Beyond cross-entropy, DivPO (Lanchantin et al., [2025a](https://arxiv.org/html/2509.02534v1#bib.bib34)) and its “soft” variants (Chung et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib11); Ismayilzada et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib27)) optimize preferences for both quality and diversity. In online RL, He et al. ([2025a](https://arxiv.org/html/2509.02534v1#bib.bib19)) perform re-weighting of rewards by likelihood to promote diverse proofs, Lanchantin et al. ([2025b](https://arxiv.org/html/2509.02534v1#bib.bib35)) show that using simple entropy regularization is a non-trivial task, and Slocum et al. ([2025](https://arxiv.org/html/2509.02534v1#bib.bib69)) attribute diversity loss to KL regularization and decouple its terms. Concurrent to our work, Chen et al. ([2025a](https://arxiv.org/html/2509.02534v1#bib.bib7)) down-weigh uncertain model solutions in math. In contrast, our work measures uncertainty at the trajectory level and up-weigh diverse responses. Darling also differs from other approaches in two important ways: (1) it employs a semantic-level diversity signal, going beyond surface-level lexical variations, and (2) it directly shapes the reward during online RL, unlike prior work that modifies cross-entropy loss in pre-training (Li et al., [2020](https://arxiv.org/html/2509.02534v1#bib.bib42)) or offline fine-tuning (Li et al., [2025b](https://arxiv.org/html/2509.02534v1#bib.bib41); Lanchantin et al., [2025a](https://arxiv.org/html/2509.02534v1#bib.bib34); Chung et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib11); Ismayilzada et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib27)).

Inference-time strategies for diversity Decoding diverse outputs from neural LMs has been a well-studied problem in the literature. A body of prior work has proposed modifications to beam search (Cho, [2016](https://arxiv.org/html/2509.02534v1#bib.bib10); Li and Jurafsky, [2016](https://arxiv.org/html/2509.02534v1#bib.bib36); Li et al., [2016b](https://arxiv.org/html/2509.02534v1#bib.bib38); Vijayakumar et al., [2018](https://arxiv.org/html/2509.02534v1#bib.bib70); Kulikov et al., [2019](https://arxiv.org/html/2509.02534v1#bib.bib32)).Ippolito et al. ([2019a](https://arxiv.org/html/2509.02534v1#bib.bib24)), in their work, compare such methods with those that simply increase the sampling temperature (Peeperkorn et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib60); Shur-Ofry et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib66)).Apart from modifying the beam search process, many methods have proposed to harness the prompt to elicit diverse responses, which includes conditioning on random seeds (Nagarajan et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib55)), on different persona (Shur-Ofry et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib66); Ge et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib17)), on past generations (Lu et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib51)), and directly prompting the model to “be diverse” (Zhang et al., [2025b](https://arxiv.org/html/2509.02534v1#bib.bib87)). Both Padmakumar et al. ([2025](https://arxiv.org/html/2509.02534v1#bib.bib58)) and Zhang et al. ([2025b](https://arxiv.org/html/2509.02534v1#bib.bib87)) present a comprehensive evaluation of such prompting methods, revealing improved diversity often comes at the cost of degraded quality. Our work directly modifies the training objective which is orthogonal to and compatible with decoding methods that elicit diversity at test time.

8 Conclusion
------------

In this work we introduced Darling, an online RL method that jointly optimizes for both quality and diversity. Unlike prior RL approaches that often lead to diversity collapse, Darling effectively preserves diversity in model generations. Through various qualitative and quantitative experiments, we demonstrated its effectiveness with different model families and sizes across both verifiable and non-verifiable tasks.

References
----------

*   Aggarwal and Welleck (2025) Pranjal Aggarwal and Sean Welleck. L1: Controlling how long a reasoning model thinks with reinforcement learning. In _Second Conference on Language Modeling_, 2025. [https://arxiv.org/abs/2503.04697](https://arxiv.org/abs/2503.04697). 
*   An et al. (2025) Chenxin An, Zhihui Xie, Xiaonan Li, Lei Li, Jun Zhang, Shansan Gong, Ming Zhong, Jingjing Xu, Xipeng Qiu, Mingxuan Wang, and Lingpeng Kong. Polaris: A post-training recipe for scaling reinforcement learning on advanced reasoning models, 2025. [https://hkunlp.github.io/blog/2025/Polaris](https://hkunlp.github.io/blog/2025/Polaris). 
*   Anthropic (2024) Anthropic. Claude 3.7 sonnet system card, 2024. [https://assets.anthropic.com/m/785e231869ea8b3b/original/claude-3-7-sonnet-system-card.pdf](https://assets.anthropic.com/m/785e231869ea8b3b/original/claude-3-7-sonnet-system-card.pdf). 
*   Art of Problem Solving (2025) Art of Problem Solving. Aime problems and solutions, 2025. [https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions). 
*   Balunović et al. (2025) Mislav Balunović, Jasper Dekoninck, Ivo Petrov, Nikola Jovanović, and Martin Vechev. Matharena: Evaluating llms on uncontaminated math competitions, February 2025. [https://matharena.ai/](https://matharena.ai/). 
*   Chen et al. (2021) Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, and Michael Petrov et al. Evaluating large language models trained on code, 2021. [https://arxiv.org/abs/2107.03374](https://arxiv.org/abs/2107.03374). 
*   Chen et al. (2025a) Minghan Chen, Guikun Chen, Wenguan Wang, and Yi Yang. Seed-grpo: Semantic entropy enhanced grpo for uncertainty-aware policy optimization, 2025a. [https://arxiv.org/abs/2505.12346](https://arxiv.org/abs/2505.12346). 
*   Chen et al. (2025b) Zhipeng Chen, Xiaobo Qin, Youbin Wu, Yue Ling, Qinghao Ye, Wayne Xin Zhao, and Guang Shi. Pass@k training for adaptively balancing exploration and exploitation of large reasoning models, 2025b. [https://arxiv.org/abs/2508.10751](https://arxiv.org/abs/2508.10751). 
*   Cheng et al. (2025) Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, and Furu Wei. Reasoning with exploration: An entropy perspective, 2025. [https://arxiv.org/abs/2506.14758](https://arxiv.org/abs/2506.14758). 
*   Cho (2016) Kyunghyun Cho. Noisy parallel approximate decoding for conditional recurrent language model, 2016. [https://arxiv.org/abs/1605.03835](https://arxiv.org/abs/1605.03835). 
*   Chung et al. (2025) John Joon Young Chung, Vishakh Padmakumar, Melissa Roemmele, Yuqian Sun, and Max Kreminski. Modifying large language model post-training for diverse creative writing, 2025. [https://arxiv.org/abs/2503.17126](https://arxiv.org/abs/2503.17126). 
*   Cui et al. (2025) Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, Zhiyuan Liu, Hao Peng, Lei Bai, Wanli Ouyang, Yu Cheng, Bowen Zhou, and Ning Ding. The entropy mechanism of reinforcement learning for reasoning language models, 2025. [https://arxiv.org/abs/2505.22617](https://arxiv.org/abs/2505.22617). 
*   DeepSeek-AI et al. (2025) DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z.F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H.Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J.L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R.J. Chen, R.L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S.S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T.Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W.L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X.Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y.K. Li, Y.Q. Wang, Y.X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y.X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z.Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. [https://arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948). 
*   Dubois et al. (2024) Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B Hashimoto. Length-controlled alpacaeval: A simple way to debias automatic evaluators. _arXiv preprint arXiv:2404.04475_, 2024. 
*   Fan et al. (2018) Angela Fan, Mike Lewis, and Yann Dauphin. Hierarchical neural story generation. In Iryna Gurevych and Yusuke Miyao, editors, _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, pages 889–898, Melbourne, Australia, July 2018. Association for Computational Linguistics. [10.18653/v1/P18-1082](https://arxiv.org/doi.org/10.18653/v1/P18-1082). [https://aclanthology.org/P18-1082/](https://aclanthology.org/P18-1082/). 
*   Frick et al. (2024) Evan Frick, Peter Jin, Tianle Li, Karthik Ganesan, Jian Zhang, Jiantao Jiao, and Banghua Zhu. Athene-70b: Redefining the boundaries of post-training for open models, July 2024. [https://nexusflow.ai/blogs/athene](https://nexusflow.ai/blogs/athene). 
*   Ge et al. (2025) Tao Ge, Xin Chan, Xiaoyang Wang, Dian Yu, Haitao Mi, and Dong Yu. Scaling synthetic data creation with 1,000,000,000 personas, 2025. [https://arxiv.org/abs/2406.20094](https://arxiv.org/abs/2406.20094). 
*   Gruver et al. (2023) Nate Gruver, Samuel Don Stanton, Nathan C. Frey, Tim G.J. Rudner, Isidro Hotzel, Julien Lafrance-Vanasse, Arvind Rajpal, Kyunghyun Cho, and Andrew Gordon Wilson. Protein design with guided discrete diffusion. In _Thirty-seventh Conference on Neural Information Processing Systems_, 2023. [https://openreview.net/forum?id=MfiK69Ga6p](https://openreview.net/forum?id=MfiK69Ga6p). 
*   He et al. (2025a) Andre He, Daniel Fried, and Sean Welleck. Rewarding the unlikely: Lifting grpo beyond distribution sharpening, 2025a. [https://arxiv.org/abs/2506.02355](https://arxiv.org/abs/2506.02355). 
*   He et al. (2024) Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, pages 3828–3850, Bangkok, Thailand, August 2024. Association for Computational Linguistics. [10.18653/v1/2024.acl-long.211](https://arxiv.org/doi.org/10.18653/v1/2024.acl-long.211). [https://aclanthology.org/2024.acl-long.211/](https://aclanthology.org/2024.acl-long.211/). 
*   He et al. (2025b) Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, Siyuan Li, Liang Zeng, Tianwen Wei, Cheng Cheng, Bo An, Yang Liu, and Yahui Zhou. Skywork open reasoner 1 technical report, 2025b. [https://arxiv.org/abs/2505.22312](https://arxiv.org/abs/2505.22312). 
*   Hu (2025) Jian Hu. Reinforce++: A simple and efficient approach for aligning large language models. _arXiv preprint arXiv:2501.03262_, 2025. 
*   Huang et al. (2025) Audrey Huang, Adam Block, Dylan J Foster, Dhruv Rohatgi, Cyril Zhang, Max Simchowitz, Jordan T. Ash, and Akshay Krishnamurthy. Self-Improvement in Language Models: The Sharpening Mechanism. In _International Conference on Learning Representations (iclr)_, 2025. [https://openreview.net/forum?id=WJaUkwci9o](https://openreview.net/forum?id=WJaUkwci9o). 
*   Ippolito et al. (2019a) Daphne Ippolito, Reno Kriz, João Sedoc, Maria Kustikova, and Chris Callison-Burch. Comparison of diverse decoding methods from conditional language models. In Anna Korhonen, David Traum, and Lluís Màrquez, editors, _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_, pages 3752–3762, Florence, Italy, July 2019a. Association for Computational Linguistics. [10.18653/v1/P19-1365](https://arxiv.org/doi.org/10.18653/v1/P19-1365). [https://aclanthology.org/P19-1365/](https://aclanthology.org/P19-1365/). 
*   Ippolito et al. (2019b) Daphne Ippolito, Reno Kriz, João Sedoc, Maria Kustikova, and Chris Callison-Burch. Comparison of Diverse Decoding Methods from Conditional Language Models. In _Annual Meeting of the Association for Computational Linguistics (ACL)_, 2019b. 
*   Ismayilzada et al. (2024) Mete Ismayilzada, Debjit Paul, Antoine Bosselut, and Lonneke van der Plas. Creativity in ai: Progresses and challenges, 2024. [https://arxiv.org/abs/2410.17218](https://arxiv.org/abs/2410.17218). 
*   Ismayilzada et al. (2025) Mete Ismayilzada, Antonio Laverghetta Jr., Simone A. Luchini, Reet Patel, Antoine Bosselut, Lonneke van der Plas, and Roger Beaty. Creative preference optimization, 2025. [https://arxiv.org/abs/2505.14442](https://arxiv.org/abs/2505.14442). 
*   Ji et al. (2025) Ke Ji, Jiahao Xu, Tian Liang, Qiuzhi Liu, Zhiwei He, Xingyu Chen, Xiaoyuan Liu, Zhijie Wang, Junying Chen, Benyou Wang, et al. The first few tokens are all you need: An efficient and effective unsupervised prefix fine-tuning method for reasoning models. _arXiv preprint arXiv:2503.02875_, 2025. 
*   Jung et al. (2025) Jaehun Jung, Seungju Han, Ximing Lu, Skyler Hallinan, David Acuna, Shrimai Prabhumoye, Mostafa Patwary, Mohammad Shoeybi, Bryan Catanzaro, and Yejin Choi. Prismatic synthesis: Gradient-based data diversification boosts generalization in llm reasoning, 2025. [https://arxiv.org/abs/2505.20161](https://arxiv.org/abs/2505.20161). 
*   Kirk et al. (2024) Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, and Roberta Raileanu. Understanding the effects of RLHF on LLM generalisation and diversity. In _International Conference on Learning Representations (iclr)_, 2024. [https://openreview.net/forum?id=PXD3FAVHJT](https://openreview.net/forum?id=PXD3FAVHJT). 
*   Kloek and van Dijk (1978) T.Kloek and H.K. van Dijk. Bayesian estimates of equation system parameters: An application of integration by monte carlo. _Econometrica_, 46(1):1–19, 1978. ISSN 00129682, 14680262. [http://www.jstor.org/stable/1913641](http://www.jstor.org/stable/1913641). 
*   Kulikov et al. (2019) Ilia Kulikov, Alexander Miller, Kyunghyun Cho, and Jason Weston. Importance of search and evaluation strategies in neural dialogue modeling. In Kees van Deemter, Chenghua Lin, and Hiroya Takamura, editors, _Proceedings of the 12th International Conference on Natural Language Generation_, pages 76–87, Tokyo, Japan, October–November 2019. Association for Computational Linguistics. [10.18653/v1/W19-8609](https://arxiv.org/doi.org/10.18653/v1/W19-8609). [https://aclanthology.org/W19-8609/](https://aclanthology.org/W19-8609/). 
*   Kwon et al. (2023) Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In _Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles_, 2023. 
*   Lanchantin et al. (2025a) Jack Lanchantin, Angelica Chen, Shehzaad Dhuliawala, Ping Yu, Jason Weston, Sainbayar Sukhbaatar, and Ilia Kulikov. Diverse preference optimization, 2025a. [https://arxiv.org/abs/2501.18101](https://arxiv.org/abs/2501.18101). 
*   Lanchantin et al. (2025b) Jack Lanchantin, Angelica Chen, Janice Lan, Xian Li, Swarnadeep Saha, Tianlu Wang, Jing Xu, Ping Yu, Weizhe Yuan, Jason E Weston, et al. Bridging offline and online reinforcement learning for llms. _arXiv preprint arXiv:2506.21495_, 2025b. 
*   Li and Jurafsky (2016) Jiwei Li and Dan Jurafsky. Mutual information and diverse decoding improve neural machine translation, 2016. [https://arxiv.org/abs/1601.00372](https://arxiv.org/abs/1601.00372). 
*   Li et al. (2016a) Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, and Bill Dolan. A diversity-promoting objective function for neural conversation models. In Kevin Knight, Ani Nenkova, and Owen Rambow, editors, _Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_, pages 110–119, San Diego, California, June 2016a. Association for Computational Linguistics. [10.18653/v1/N16-1014](https://arxiv.org/doi.org/10.18653/v1/N16-1014). [https://aclanthology.org/N16-1014/](https://aclanthology.org/N16-1014/). 
*   Li et al. (2016b) Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, and Bill Dolan. A diversity-promoting objective function for neural conversation models, 2016b. [https://arxiv.org/abs/1510.03055](https://arxiv.org/abs/1510.03055). 
*   Li et al. (2025a) Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E. Gonzalez, and Ion Stoica. From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline. In _Forty-second International Conference on Machine Learning_, 2025a. [https://openreview.net/forum?id=KfTf9vFvSn](https://openreview.net/forum?id=KfTf9vFvSn). 
*   Li et al. (2023) Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Alpacaeval: An automatic evaluator of instruction-following models. [https://github.com/tatsu-lab/alpaca_eval](https://github.com/tatsu-lab/alpaca_eval), 5 2023. 
*   Li et al. (2025b) Ziniu Li, Congliang Chen, Tian Xu, Zeyu Qin, Jiancong Xiao, Zhi-Quan Luo, and Ruoyu Sun. Preserving diversity in supervised fine-tuning of large language models. In _The Thirteenth International Conference on Learning Representations_, 2025b. [https://openreview.net/forum?id=NQEe7B7bSw](https://openreview.net/forum?id=NQEe7B7bSw). 
*   Li et al. (2020) Zuchao Li, Rui Wang, Kehai Chen, Masso Utiyama, Eiichiro Sumita, Zhuosheng Zhang, and Hai Zhao. Data-dependent gaussian prior objective for language generation. In _International Conference on Learning Representations_, 2020. [https://openreview.net/forum?id=S1efxTVYDr](https://openreview.net/forum?id=S1efxTVYDr). 
*   Liang et al. (2025) Xiao Liang, Zhongzhi Li, Yeyun Gong, Yelong Shen, Ying Nian Wu, Zhijiang Guo, and Weizhu Chen. Beyond pass@1: Self-play with variational problem synthesis sustains rlvr, 2025. [https://arxiv.org/abs/2508.14029](https://arxiv.org/abs/2508.14029). 
*   Lin et al. (2021) Xiang Lin, Simeng Han, and Shafiq Joty. Straight to the gradient: Learning to use novel tokens for neural text generation. In Marina Meila and Tong Zhang, editors, _Proceedings of the 38th International Conference on Machine Learning_, volume 139 of _Proceedings of Machine Learning Research_, pages 6642–6653. PMLR, 18–24 Jul 2021. [https://proceedings.mlr.press/v139/lin21b.html](https://proceedings.mlr.press/v139/lin21b.html). 
*   Liu et al. (2025a) Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, and Yi Dong. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models. _arXiv preprint_, 2025a. [https://arxiv.org/abs/2505.24864](https://arxiv.org/abs/2505.24864). 
*   Liu et al. (2025b) Wei Liu, Ruochen Zhou, Yiyun Deng, Yuzhen Huang, Junteng Liu, Yuntian Deng, Yizhe Zhang, and Junxian He. Learn to reason efficiently with adaptive length-based reward shaping, 2025b. [https://arxiv.org/abs/2505.15612](https://arxiv.org/abs/2505.15612). 
*   Liu et al. (2025c) Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. _arXiv preprint arXiv:2503.20783_, 2025c. 
*   Liu et al. (2025d) Zihe Liu, Jiashun Liu, Yancheng He, Weixun Wang, Jiaheng Liu, Ling Pan, Xinyu Hu, Shaopan Xiong, Ju Huang, Jian Hu, Shengyi Huang, Siran Yang, Jiamang Wang, Wenbo Su, and Bo Zheng. Part i: Tricks or traps? a deep dive into rl for llm reasoning, 2025d. [https://arxiv.org/abs/2508.08221](https://arxiv.org/abs/2508.08221). 
*   Llama Team (2024) Llama Team. The llama 3 herd of models, 2024. [https://arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783). 
*   Lu et al. (2025) Ximing Lu, Melanie Sclar, Skyler Hallinan, Niloofar Mireshghallah, Jiacheng Liu, Seungju Han, Allyson Ettinger, Liwei Jiang, Khyathi Chandu, Nouha Dziri, and Yejin Choi. AI as humanity’s salieri: Quantifying linguistic creativity of language models via systematic attribution of machine text against web text. In _The Thirteenth International Conference on Learning Representations_, 2025. [https://openreview.net/forum?id=ilOEOIqolQ](https://openreview.net/forum?id=ilOEOIqolQ). 
*   Lu et al. (2024) Yining Lu, Dixuan Wang, Tianjian Li, Dongwei Jiang, and Daniel Khashabi. Benchmarking language model creativity: A case study on code generation. _arXiv preprint arXiv:2407.09007_, 2024. [https://arxiv.org/abs/2407.09007](https://arxiv.org/abs/2407.09007). 
*   Luo et al. (2025) Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Tianjun Zhang, Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl. [https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2), 2025. Notion Blog. 
*   Mahony et al. (2024) Laura O'Mahony, Leo Grinsztajn, Hailey Schoelkopf, and Stella Biderman. Attributing Mode Collapse in the fine-tuning of Large Language Models. In _ICLR 2024 Workshop on Mathematical and Empirical Understanding of Foundation Models_, 2024. [https://openreview.net/forum?id=3pDMYjpOxk](https://openreview.net/forum?id=3pDMYjpOxk). 
*   Mikolov et al. (2013) Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. In _International Conference on Learning Representations (iclr)_, 2013. [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781). 
*   Nagarajan et al. (2025) Vaishnavh Nagarajan, Chen Henry Wu, Charles Ding, and Aditi Raghunathan. Roll the dice & look before you leap: Going beyond the creative limits of next-token prediction. In _Forty-second International Conference on Machine Learning_, 2025. [https://openreview.net/forum?id=Hi0SyHMmkd](https://openreview.net/forum?id=Hi0SyHMmkd). 
*   OpenAI et al. (2024) OpenAI, :, Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, Aleksander Mądry, Alex Baker-Whitcomb, Alex Beutel, Alex Borzunov, Alex Carney, Alex Chow, Alex Kirillov, Alex Nichol, Alex Paino, Alex Renzin, Alex Tachard Passos, Alexander Kirillov, Alexi Christakis, Alexis Conneau, Ali Kamali, Allan Jabri, Allison Moyer, Allison Tam, Amadou Crookes, Amin Tootoochian, Amin Tootoonchian, Ananya Kumar, Andrea Vallone, Andrej Karpathy, Andrew Braunstein, Andrew Cann, Andrew Codispoti, Andrew Galu, Andrew Kondrich, Andrew Tulloch, Andrey Mishchenko, Angela Baek, Angela Jiang, Antoine Pelisse, Antonia Woodford, Anuj Gosalia, Arka Dhar, Ashley Pantuliano, Avi Nayak, Avital Oliver, Barret Zoph, Behrooz Ghorbani, Ben Leimberger, Ben Rossen, Ben Sokolowsky, Ben Wang, Benjamin Zweig, Beth Hoover, Blake Samic, Bob McGrew, Bobby Spero, Bogo Giertler, Bowen Cheng, Brad Lightcap, Brandon Walkin, Brendan Quinn, Brian Guarraci, Brian Hsu, Bright Kellogg, Brydon Eastman, Camillo Lugaresi, Carroll Wainwright, Cary Bassin, Cary Hudson, Casey Chu, Chad Nelson, Chak Li, Chan Jun Shern, Channing Conger, Charlotte Barette, Chelsea Voss, Chen Ding, Cheng Lu, Chong Zhang, Chris Beaumont, Chris Hallacy, Chris Koch, Christian Gibson, Christina Kim, Christine Choi, Christine McLeavey, Christopher Hesse, Claudia Fischer, Clemens Winter, Coley Czarnecki, Colin Jarvis, Colin Wei, Constantin Koumouzelis, Dane Sherburn, Daniel Kappler, Daniel Levin, Daniel Levy, David Carr, David Farhi, David Mely, David Robinson, David Sasaki, Denny Jin, Dev Valladares, Dimitris Tsipras, Doug Li, Duc Phong Nguyen, Duncan Findlay, Edede Oiwoh, Edmund Wong, Ehsan Asdar, Elizabeth Proehl, Elizabeth Yang, Eric Antonow, Eric Kramer, Eric Peterson, Eric Sigler, Eric Wallace, Eugene Brevdo, Evan Mays, Farzad Khorasani, Felipe Petroski Such, Filippo Raso, Francis Zhang, Fred von Lohmann, Freddie Sulit, Gabriel Goh, Gene Oden, Geoff Salmon, Giulio Starace, Greg Brockman, Hadi Salman, Haiming Bao, Haitang Hu, Hannah Wong, Haoyu Wang, Heather Schmidt, Heather Whitney, Heewoo Jun, Hendrik Kirchner, Henrique Ponde de Oliveira Pinto, Hongyu Ren, Huiwen Chang, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian O’Connell, Ian Osband, Ian Silber, Ian Sohl, Ibrahim Okuyucu, Ikai Lan, Ilya Kostrikov, Ilya Sutskever, Ingmar Kanitscheider, Ishaan Gulrajani, Jacob Coxon, Jacob Menick, Jakub Pachocki, James Aung, James Betker, James Crooks, James Lennon, Jamie Kiros, Jan Leike, Jane Park, Jason Kwon, Jason Phang, Jason Teplitz, Jason Wei, Jason Wolfe, Jay Chen, Jeff Harris, Jenia Varavva, Jessica Gan Lee, Jessica Shieh, Ji Lin, Jiahui Yu, Jiayi Weng, Jie Tang, Jieqi Yu, Joanne Jang, Joaquin Quinonero Candela, Joe Beutler, Joe Landers, Joel Parish, Johannes Heidecke, John Schulman, Jonathan Lachman, Jonathan McKay, Jonathan Uesato, Jonathan Ward, Jong Wook Kim, Joost Huizinga, Jordan Sitkin, Jos Kraaijeveld, Josh Gross, Josh Kaplan, Josh Snyder, Joshua Achiam, Joy Jiao, Joyce Lee, Juntang Zhuang, Justyn Harriman, Kai Fricke, Kai Hayashi, Karan Singhal, Katy Shi, Kavin Karthik, Kayla Wood, Kendra Rimbach, Kenny Hsu, Kenny Nguyen, Keren Gu-Lemberg, Kevin Button, Kevin Liu, Kiel Howe, Krithika Muthukumar, Kyle Luther, Lama Ahmad, Larry Kai, Lauren Itow, Lauren Workman, Leher Pathak, Leo Chen, Li Jing, Lia Guy, Liam Fedus, Liang Zhou, Lien Mamitsuka, Lilian Weng, Lindsay McCallum, Lindsey Held, Long Ouyang, Louis Feuvrier, Lu Zhang, Lukas Kondraciuk, Lukasz Kaiser, Luke Hewitt, Luke Metz, Lyric Doshi, Mada Aflak, Maddie Simens, Madelaine Boyd, Madeleine Thompson, Marat Dukhan, Mark Chen, Mark Gray, Mark Hudnall, Marvin Zhang, Marwan Aljubeh, Mateusz Litwin, Matthew Zeng, Max Johnson, Maya Shetty, Mayank Gupta, Meghan Shah, Mehmet Yatbaz, Meng Jia Yang, Mengchao Zhong, Mia Glaese, Mianna Chen, Michael Janner, Michael Lampe, Michael Petrov, Michael Wu, Michele Wang, Michelle Fradin, Michelle Pokrass, Miguel Castro, Miguel Oom Temudo de Castro, Mikhail Pavlov, Miles Brundage, Miles Wang, Minal Khan, Mira Murati, Mo Bavarian, Molly Lin, Murat Yesildal, Nacho Soto, Natalia Gimelshein, Natalie Cone, Natalie Staudacher, Natalie Summers, Natan LaFontaine, Neil Chowdhury, Nick Ryder, Nick Stathas, Nick Turley, Nik Tezak, Niko Felix, Nithanth Kudige, Nitish Keskar, Noah Deutsch, Noel Bundick, Nora Puckett, Ofir Nachum, Ola Okelola, Oleg Boiko, Oleg Murk, Oliver Jaffe, Olivia Watkins, Olivier Godement, Owen Campbell-Moore, Patrick Chao, Paul McMillan, Pavel Belov, Peng Su, Peter Bak, Peter Bakkum, Peter Deng, Peter Dolan, Peter Hoeschele, Peter Welinder, Phil Tillet, Philip Pronin, Philippe Tillet, Prafulla Dhariwal, Qiming Yuan, Rachel Dias, Rachel Lim, Rahul Arora, Rajan Troll, Randall Lin, Rapha Gontijo Lopes, Raul Puri, Reah Miyara, Reimar Leike, Renaud Gaubert, Reza Zamani, Ricky Wang, Rob Donnelly, Rob Honsby, Rocky Smith, Rohan Sahai, Rohit Ramchandani, Romain Huet, Rory Carmichael, Rowan Zellers, Roy Chen, Ruby Chen, Ruslan Nigmatullin, Ryan Cheu, Saachi Jain, Sam Altman, Sam Schoenholz, Sam Toizer, Samuel Miserendino, Sandhini Agarwal, Sara Culver, Scott Ethersmith, Scott Gray, Sean Grove, Sean Metzger, Shamez Hermani, Shantanu Jain, Shengjia Zhao, Sherwin Wu, Shino Jomoto, Shirong Wu, Shuaiqi, Xia, Sonia Phene, Spencer Papay, Srinivas Narayanan, Steve Coffey, Steve Lee, Stewart Hall, Suchir Balaji, Tal Broda, Tal Stramer, Tao Xu, Tarun Gogineni, Taya Christianson, Ted Sanders, Tejal Patwardhan, Thomas Cunninghman, Thomas Degry, Thomas Dimson, Thomas Raoux, Thomas Shadwell, Tianhao Zheng, Todd Underwood, Todor Markov, Toki Sherbakov, Tom Rubin, Tom Stasi, Tomer Kaftan, Tristan Heywood, Troy Peterson, Tyce Walters, Tyna Eloundou, Valerie Qi, Veit Moeller, Vinnie Monaco, Vishal Kuo, Vlad Fomenko, Wayne Chang, Weiyi Zheng, Wenda Zhou, Wesam Manassra, Will Sheu, Wojciech Zaremba, Yash Patil, Yilei Qian, Yongjik Kim, Youlong Cheng, Yu Zhang, Yuchen He, Yuchen Zhang, Yujia Jin, Yunxing Dai, and Yury Malkov. Gpt-4o system card, 2024. [https://arxiv.org/abs/2410.21276](https://arxiv.org/abs/2410.21276). 
*   Padmakumar and He (2024) Vishakh Padmakumar and He He. Does writing with language models reduce content diversity? In _The Twelfth International Conference on Learning Representations_, 2024. [https://openreview.net/forum?id=Feiz5HtCD0](https://openreview.net/forum?id=Feiz5HtCD0). 
*   Padmakumar et al. (2025) Vishakh Padmakumar, Chen Yueh-Han, Jane Pan, Valerie Chen, and He He. Beyond memorization: Mapping the originality-quality frontier of language models, 2025. [https://arxiv.org/abs/2504.09389](https://arxiv.org/abs/2504.09389). 
*   Paech (2023) Samuel J. Paech. Eq-bench: An emotional intelligence benchmark for large language models, 2023. 
*   Peeperkorn et al. (2024) Max Peeperkorn, Tom Kouwenhoven, Dan Brown, and Anna Jordanous. Is temperature the creativity parameter of large language models?, 2024. [https://arxiv.org/abs/2405.00492](https://arxiv.org/abs/2405.00492). 
*   Rafailov et al. (2024) Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. _Advances in Neural Information Processing Systems (NeurIPS)_, 36, 2024. [https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290). 
*   Romera-Paredes et al. (2024) Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M.Pawan Kumar, Emilien Dupont, Francisco J.R. Ruiz, Jordan S. Ellenberg, Pengming Wang, Omar Fawzi, Pushmeet Kohli, and Alhussein Fawzi. Mathematical discoveries from program search with large language models. _Nat._, 625(7995):468–475, January 2024. [https://doi.org/10.1038/s41586-023-06924-6](https://doi.org/10.1038/s41586-023-06924-6). 
*   Salton and Buckley (1988) Gerard Salton and Christopher Buckley. Term-weighting approaches in automatic text retrieval. _Information Processing & Management_, 24(5):513–523, 1988. ISSN 0306-4573. [https://doi.org/10.1016/0306-4573(88)90021-0](https://arxiv.org/doi.org/https://doi.org/10.1016/0306-4573(88)90021-0). [https://www.sciencedirect.com/science/article/pii/0306457388900210](https://www.sciencedirect.com/science/article/pii/0306457388900210). 
*   Shao et al. (2024) Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y.K. Li, Y.Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300). 
*   Sheng et al. (2024) Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. _arXiv preprint arXiv: 2409.19256_, 2024. 
*   Shur-Ofry et al. (2024) Michal Shur-Ofry, Bar Horowitz-Amsalem, Adir Rahamim, and Yonatan Belinkov. Growing a tail: Increasing output diversity in large language models, 2024. [https://arxiv.org/abs/2411.02989](https://arxiv.org/abs/2411.02989). 
*   Shypula et al. (2025) Alexander Shypula, Shuo Li, Botong Zhang, Vishakh Padmakumar, Kayo Yin, and Osbert Bastani. Evaluating the diversity and quality of llm generated content. In _Conference on Language Modeling_, 2025. [https://arxiv.org/abs/2504.12522](https://arxiv.org/abs/2504.12522). 
*   Si et al. (2025) Chenglei Si, Diyi Yang, and Tatsunori Hashimoto. Can LLMs generate novel research ideas? a large-scale human study with 100+ NLP researchers. In _The Thirteenth International Conference on Learning Representations_, 2025. [https://openreview.net/forum?id=M23dTGWCZy](https://openreview.net/forum?id=M23dTGWCZy). 
*   Slocum et al. (2025) Stewart Slocum, Asher Parker-Sartori, and Dylan Hadfield-Menell. Diverse preference learning for capabilities and alignment. In _The Thirteenth International Conference on Learning Representations_, 2025. [https://openreview.net/forum?id=pOq9vDIYev](https://openreview.net/forum?id=pOq9vDIYev). 
*   Vijayakumar et al. (2018) Ashwin K Vijayakumar, Michael Cogswell, Ramprasaath R. Selvaraju, Qing Sun, Stefan Lee, David Crandall, and Dhruv Batra. Diverse beam search: Decoding diverse solutions from neural sequence models. In _Conference on Artificial Intelligence (AAAI)_, 2018. 
*   Warner et al. (2024) Benjamin Warner, Antoine Chaffin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, Nathan Cooper, Griffin Adams, Jeremy Howard, and Iacopo Poli. Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference, 2024. [https://arxiv.org/abs/2412.13663](https://arxiv.org/abs/2412.13663). 
*   Welleck et al. (2020) Sean Welleck, Ilia Kulikov, Stephen Roller, Emily Dinan, Kyunghyun Cho, and Jason Weston. Neural Text Generation With Unlikelihood Training. In _International Conference on Learning Representations (iclr)_, 2020. [https://openreview.net/forum?id=SJeYe0NtvH](https://openreview.net/forum?id=SJeYe0NtvH). 
*   West and Potts (2025) Peter West and Christopher Potts. Base models beat aligned models at randomness and creativity, 2025. [https://arxiv.org/abs/2505.00047](https://arxiv.org/abs/2505.00047). 
*   Wieting and Gimpel (2018) John Wieting and Kevin Gimpel. ParaNMT-50M: Pushing the limits of paraphrastic sentence embeddings with millions of machine translations. In Iryna Gurevych and Yusuke Miyao, editors, _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, pages 451–462, Melbourne, Australia, July 2018. Association for Computational Linguistics. [10.18653/v1/P18-1042](https://arxiv.org/doi.org/10.18653/v1/P18-1042). [https://aclanthology.org/P18-1042/](https://aclanthology.org/P18-1042/). 
*   Wieting et al. (2019) John Wieting, Taylor Berg-Kirkpatrick, Kevin Gimpel, and Graham Neubig. Beyond BleU: Training Neural Machine Translation with Semantic Similarity. In _Annual Meeting of the Association for Computational Linguistics (ACL)_, 2019. 
*   Wu et al. (2025a) Fang Wu, Weihao Xuan, Ximing Lu, Zaid Harchaoui, and Yejin Choi. The invisible leash: Why rlvr may not escape its origin, 2025a. [https://arxiv.org/abs/2507.14843](https://arxiv.org/abs/2507.14843). 
*   Wu et al. (2025b) Yuhao Wu, Yushi Bai, Zhiqiang Hu, Roy Ka-Wei Lee, and Juanzi Li. Longwriter-zero: Mastering ultra-long text generation via reinforcement learning, 2025b. [https://arxiv.org/abs/2506.18841](https://arxiv.org/abs/2506.18841). 
*   Yang et al. (2025a) An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025a. [https://arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388). 
*   Yang and Holtzman (2025) Chenghao Yang and Ari Holtzman. How alignment shrinks the generative horizon, 2025. [https://arxiv.org/abs/2506.17871](https://arxiv.org/abs/2506.17871). 
*   Yang et al. (2025b) Zhicheng Yang, Zhijiang Guo, Yinya Huang, Yongxin Wang, Dongchun Xie, Yiwei Wang, Xiaodan Liang, and Jing Tang. Depth-breadth synergy in rlvr: Unlocking llm reasoning gains with adaptive exploration, 2025b. [https://arxiv.org/abs/2508.13755](https://arxiv.org/abs/2508.13755). 
*   Yu et al. (2025) Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale, 2025. [https://arxiv.org/abs/2503.14476](https://arxiv.org/abs/2503.14476). 
*   Yun et al. (2025) Longfei Yun, Chenyang An, Zilong Wang, Letian Peng, and Jingbo Shang. The price of format: Diversity collapse in llms, 2025. [https://arxiv.org/abs/2505.18949](https://arxiv.org/abs/2505.18949). 
*   Zeng et al. (2025) Weihao Zeng, Yuzhen Huang, Lulu Zhao, Yijun Wang, Zifei Shan, and Junxian He. B-STar: Monitoring and balancing exploration and exploitation in self-taught reasoners. In _The Thirteenth International Conference on Learning Representations_, 2025. [https://openreview.net/forum?id=P6dwZJpJ4m](https://openreview.net/forum?id=P6dwZJpJ4m). 
*   Zhang et al. (2021) Hugh Zhang, Daniel Duckworth, Daphne Ippolito, and Arvind Neelakantan. Trading off diversity and quality in natural language generation. In Anya Belz, Shubham Agarwal, Yvette Graham, Ehud Reiter, and Anastasia Shimorina, editors, _Proceedings of the Workshop on Human Evaluation of NLP Systems (HumEval)_, pages 25–33, Online, April 2021. Association for Computational Linguistics. [https://aclanthology.org/2021.humeval-1.3/](https://aclanthology.org/2021.humeval-1.3/). 
*   Zhang et al. (2025a) Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embedding: Advancing text embedding and reranking through foundation models. _arXiv preprint arXiv:2506.05176_, 2025a. 
*   Zhang et al. (2024) Yiming Zhang, Avi Schwarzschild, Nicholas Carlini, J Zico Kolter, and Daphne Ippolito. Forcing Diffuse Distributions out of Language Models. In _Conference on Language Modeling_, 2024. [https://openreview.net/forum?id=9JY1QLVFPZ](https://openreview.net/forum?id=9JY1QLVFPZ). 
*   Zhang et al. (2025b) Yiming Zhang, Harshita Diddee, Susan Holm, Hanchen Liu, Xinyue Liu, Vinay Samuel, Barry Wang, and Daphne Ippolito. Noveltybench: Evaluating language models for humanlike diversity. In _Conference on Language Modeling_, 2025b. [https://arxiv.org/abs/2504.05228](https://arxiv.org/abs/2504.05228). 
*   Zhao et al. (2024) Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. Wildchat: 1m chatGPT interaction logs in the wild. In _The Twelfth International Conference on Learning Representations_, 2024. [https://openreview.net/forum?id=Bl8u7ZRlbM](https://openreview.net/forum?id=Bl8u7ZRlbM). 
*   Zhao et al. (2023) Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can Balioglu, Pritam Damania, Bernard Nguyen, Geeta Chauhan, Yuchen Hao, Ajit Mathews, and Shen Li. Pytorch fsdp: Experiences on scaling fully sharded data parallel, 2023. [https://arxiv.org/abs/2304.11277](https://arxiv.org/abs/2304.11277). 
*   Zhu et al. (2018) Yaoming Zhu, Sidi Lu, Lei Zheng, Jiaxian Guo, Weinan Zhang, Jun Wang, and Yong Yu. Texygen: A benchmarking platform for text generation models. In _The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval_, SIGIR ’18, page 1097–1100, New York, NY, USA, 2018. Association for Computing Machinery. ISBN 9781450356572. [10.1145/3209978.3210080](https://arxiv.org/doi.org/10.1145/3209978.3210080). [https://doi.org/10.1145/3209978.3210080](https://doi.org/10.1145/3209978.3210080). 

\beginappendix

9 Partitioning the Responses
----------------------------

### 9.1 Classifier for Non-verifiable Tasks

Zhang et al. ([2025b](https://arxiv.org/html/2509.02534v1#bib.bib87)) had human annotators judge whether pairs of model-generated responses were semantically equivalent, across 1,100 prompts (2,200 responses in total). We directly use their annotations and concatenate the two responses to be classified as semantically similar or not:

[CLS] response 1 [SEP] response 2 [CLS]

and perform classification on top of the second [CLS] token. We train a ModernBERT-base(Warner et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib71)) model with 1000 NoveltyBench annotations (2000 responses) to support a max context length of 8192 tokens. We evaluate the performance of the classifier using an held out set of 100 prompts (200 responses) and plot the performance in [Figure 7](https://arxiv.org/html/2509.02534v1#S9.F7 "Figure 7 ‣ 9.1 Classifier for Non-verifiable Tasks ‣ 9 Partitioning the Responses ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations"). We found that (1) Our trained classifier (Acc.=78%) achieves similar performance with the original NoveltyBench classifier (Acc.=79%), and (2) Using proprietary models (e.g. GPT-4o and o1-mini) performs worse in terms of determining whether two responses are semantically equivalent to humans. We provide the detailed prompt we used for asking an LM to determine whether two responses are semantically similar in [Figure 8](https://arxiv.org/html/2509.02534v1#S9.F8 "Figure 8 ‣ 9.2 Classifier for Verifiable Tasks ‣ 9 Partitioning the Responses ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations").

![Image 6: Refer to caption](https://arxiv.org/html/2509.02534v1/x6.png)

Figure 7: Performance of different classifiers on 100 held out human annotated data of whether two responses are similar. Classifier based approaches outperform proprietary models in determining whether two responses are semantically similar to humans.

### 9.2 Classifier for Verifiable Tasks

The original NoveltyBench (Zhang et al., [2025b](https://arxiv.org/html/2509.02534v1#bib.bib87)) only supports non-verifiable tasks as the prompts was filtered to only be creative-writing prompts from WildChat (Zhao et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib88)). Therefore, we additionally trained a classifier using Qwen3-4B-Embeddings(Zhang et al., [2025a](https://arxiv.org/html/2509.02534v1#bib.bib85)) on top of model generated solution traces. In particlular, we performed inference using prompts from DeepscaleR (Luo et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib52)) with the following models: Qwen3-4B-Base, Qwen3-8B-Base, Qwen3-4B (without thinking), Qwen3-8B (without thinking), OctoThinker-8B-Long-Base, Qwen2.5-Math-7B-Instruct, QwQ-32B, and Llama-4-Maverick to cover diverse solution traces with different model types (base, instruct), families and sizes. We prompted Llama-3.3-70B-Instruct using the prompt in [Figure 9](https://arxiv.org/html/2509.02534v1#S9.F9 "Figure 9 ‣ 9.2 Classifier for Verifiable Tasks ‣ 9 Partitioning the Responses ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations") as the ground-truth of whether two math solutions are similar. Our trained classifier achieves a 89% accuracy on a held out validation set of 200 examples.

Figure 8: The prompt with chain-of-thought to ask an language model whether two responses are semantically similar.

Figure 9: Prompt to Llama-3.3-70B-Instruct on whether two math solution traces are similar.

10 Hyperparameters
------------------

Hyperparameters for Non-verifiable Tasks[Table 6](https://arxiv.org/html/2509.02534v1#S10.T6 "Table 6 ‣ 10 Hyperparameters ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations") shows key hyperparameters for our GRPO training on non-verifiable tasks (WildChat). We train our models using 1 nodes / 4 nodes of NVIDIA H200 for the 8B and 70B model, respectively.

Category Hyperparameter Value
Data Train file WildChat
Max prompt length 512
Max response length 1024
Filter overlong prompts True
Actor Model Base model 1 Llama-3.1-8B-Instruct
Base model 2 Llama-3.3-70B-Instruct
LR 1×10−6 1\times 10^{-6}
KL loss coefficient β\beta 0.001
KL loss type low_var_kl
Use dynamic batch size True
Rollout Rollout engine vllm
GPU mem utilization 0.8
Train rollout n 8
Temperature 1.0
Reward Model RM model Athene-RM-8B
Trainer Mini Batch size 32 & 64
Full Batch size 32 & 64 (Fully on-policy)
Critic Warmup 0
GPUs/node 8
Nodes 1 (8B), 4 (70B)
Total epochs 10

Table 6: Key hyperparameters used for GRPO training for non-verifiable tasks used in the verl (Sheng et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib65)) framework. 

Training Hyperparameters for Verifiable Tasks (Math)[Table 7](https://arxiv.org/html/2509.02534v1#S10.T7 "Table 7 ‣ 10 Hyperparameters ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations") shows key hyperparameters for our GRPO training on verifiable tasks (Math).

Category Hyperparameter Value
Data Train file DeepscaleR (10k)
Max prompt length 1024
Max response length 8192
Filter overlong prompts True
Actor Model Base model 1 Qwen3-4B-Base
Base model 2 Qwen3-14B-Base
LR 1×10−6 1\times 10^{-6}
KL loss coefficient β\beta 0
KL loss type N/A
Use dynamic batch size True
Rollout Rollout engine vllm
GPU mem utilization 0.7
Train rollout n 8
Temperature 1.0
Reward Model Rule Based Math_Verify
Trainer Mini Batch size 64
Full Batch size 256 (4 step off-policy)
Critic Warmup 0
GPUs/node 8
Nodes 8
Total epochs 10
Clip Ratio(0.2, 0.2)

Table 7: Key hyperparameters used for GRPO training on DeepScaleR (Luo et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib52)) in the verl (Sheng et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib65)) framework. We use the huggingface math_verify library to extract and verify whether the model response matches the ground-truth answer.

Hyperparameters for Evaluations[Table 8](https://arxiv.org/html/2509.02534v1#S10.T8 "Table 8 ‣ 10 Hyperparameters ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations") shows the hyperparameters we used for evaluation. We used the official code-bases for each benchmark except competition math, which we adopt the Qwen2.5-Math codebase for evaluation.

Category Hyperparameter Value
AlpacaEval 2.0 (Dubois et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib14))Judge GPT-4o
Max generation length 8192
Temperature 0.6
Top-p 0.9
ArenaHard v1.0/v2.0 (Li et al., [2025a](https://arxiv.org/html/2509.02534v1#bib.bib39))Judge GPT-4o
Max generation length 8192
Temperature 0.6
Top-p 0.9
EQ-Bench (Creative Writing) (Paech, [2023](https://arxiv.org/html/2509.02534v1#bib.bib59))Judge Claude-3.7-Sonnet
Max generation length 4096
Temperature 1.0
Min-p 0.1
NoveltyBench (Zhang et al., [2025b](https://arxiv.org/html/2509.02534v1#bib.bib87))Temperature 1.0
Max generation length 4096
Patience 1.0
Partition model deberta-v3-large-generation-similarity
Reward model Skywork-Reward-Gemma-2-27B-v0.2
Competition Math Temperature 0.6
Top-p 0.95
Max generation length 12000

Table 8: Evaluation parameters by benchmark. Competition Math contains 4 benchmarks: OlympiadBench (He et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib20)), AIME25 (Art of Problem Solving, [2025](https://arxiv.org/html/2509.02534v1#bib.bib4)), Brumo Math (Balunović et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib5)) and HMMT (Balunović et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib5)). We used vLLM (Kwon et al., [2023](https://arxiv.org/html/2509.02534v1#bib.bib33)) for inference.

11 Generation Examples
----------------------

### 11.1 Example Generation in EQBench

Figure 10: Example outputs from Llama-3.1-405B-Instruct and Llama-3.1-8B-Instruct enhanced with Darling. The former produces plain conversations, while models trained with Darling use art metaphors.

### 11.2 Example generations in NoveltyBench

12 Benchmark Descriptions
-------------------------

Non-verifiable We provide detailed descriptions and statistics of the benchmarks in our non-verifiable task experiments (§[4](https://arxiv.org/html/2509.02534v1#S4 "4 DARLING on Non-verifiable Tasks ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations")):

*   •AlpacaEval 2.0(Dubois et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib14)) is a benchmark of 805 prompts, each paired with a GPT-4-turbo response. To evaluate a model, it generates responses to the same prompts, and a judge compares them against the GPT-4-turbo outputs. Higher win rate (WR) or length-controlled win rate (LCWR) indicates better performance. 
*   •Arena-Hard v1.0/v2.0(Li et al., [2025a](https://arxiv.org/html/2509.02534v1#bib.bib39)) is a benchmark of 750 prompts, evenly split between challenging math/coding tasks and creative writing tasks. As in AlpacaEval 2.0, a judge compares model responses against a baseline, with higher win rates indicating stronger performance. 
*   •EQBench (Creative Writing v3) (Paech, [2023](https://arxiv.org/html/2509.02534v1#bib.bib59)) evaluates models on 32 creative writing prompts, judged by Claude Sonnet. Responses are scored both by rubric and through pairwise comparisons, with Elo ratings computed from the latter. The benchmark emphasizes challenging prompts (e.g., humor, romance, unusual perspectives) to expose weaknesses, and higher Elo or rubric scores indicate stronger creative writing ability. 
*   •NoveltyBench(Zhang et al., [2025b](https://arxiv.org/html/2509.02534v1#bib.bib87)) consists of 1,100 prompts from WildChat (Zhao et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib88)) that require diverse responses. Diversity is measured using a partition classifier (deberta-v3-large-generation-similarity), while response quality is assessed with a reward model (Skywork/Skywork-Reward-Gemma-2-27B-v0.2). In our work, we primarily use the distinct classifier, as it is trained on human annotations, whereas the reward model is vulnerable to reward hacking. 

Verifiable We used 4 competition math benchmarks in §[5](https://arxiv.org/html/2509.02534v1#S5 "5 DARLING on Verifiable Tasks ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations"): OlympiadBench (He et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib20)) contains 675 questions, AIME 25 (Art of Problem Solving, [2025](https://arxiv.org/html/2509.02534v1#bib.bib4)), Brumo (Balunović et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib5)) and HMMT (Balunović et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib5)) each contains 30 examples.

13 Full Results on Math
-----------------------

[Table 9](https://arxiv.org/html/2509.02534v1#S13.T9 "Table 9 ‣ 13 Full Results on Math ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations") and [Table 10](https://arxiv.org/html/2509.02534v1#S13.T10 "Table 10 ‣ 13 Full Results on Math ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations") shows the Math results for training on Qwen-4B-Base and Qwen-14B-Base, respectively.

Experiment Dataset Pass@1 Pass@2 Pass@4 Pass@8 Pass@16 Pass@32 Pass@64 Pass@128
Qwen3-4B-Base Olympiadbench 33.30 40.29 47.68 53.80 59.12 63.71 67.63 71.11
Qwen3-4B-Base AIME 25 8.17 13.52 19.92 26.16 31.95 37.63 42.98 47.35
Qwen3-4B-Base Brumo 25 16.68 22.95 28.85 33.98 38.51 43.19 48.73 55.10
Qwen3-4B-Base HMMT 25 3.30 3.45 4.54 7.90 12.52 17.84 22.98 27.12
GRPO Olympiadbench 42.27 48.12 53.10 57.42 61.25 64.63 67.59 70.37
GRPO AIME 25 19.51 23.93 27.79 31.36 35.55 41.44 48.37 53.33
GRPO Brumo 25 24.66 30.58 35.12 39.03 43.42 48.51 55.03 63.24
GRPO HMMT 25 7.14 10.29 13.67 17.50 20.78 22.74 24.59 26.72
Darling Olympiadbench 45.53 51.90 56.97 60.90 64.07 66.80 70.19 74.41
Darling AIME 25 20.06 26.11 32.42 39.29 46.17 52.29 57.45 62.28
Darling Brumo 25 31.73 39.09 45.25 50.46 55.49 60.42 64.72 68.27
Darling HMMT 25 10.32 13.65 17.90 22.66 27.03 30.82 34.69 39.19

Table 9: Full math results of training on Qwen3-4B-Base. Values represent pass@k k performance (up to pass@128).

Experiment Dataset Pass@1 Pass@2 Pass@4 Pass@8 Pass@16 Pass@32 Pass@64 Pass@128
Qwen3-14B-Base Olympiadbench 41.30 46.41 52.81 58.16 62.77 66.83 70.39 73.78
Qwen3-14B-Base AIME 25 12.23 18.84 25.44 31.17 36.77 42.81 48.68 53.88
Qwen3-14B-Base Brumo 25 20.62 27.48 33.93 39.66 44.45 48.96 54.48 60.94
Qwen3-14B-Base HMMT 25 3.05 5.30 8.41 12.10 16.20 20.86 26.38 32.77
GRPO Olympiadbench 51.80 57.19 60.99 63.77 65.93 67.77 69.57 71.56
GRPO AIME 25 25.87 31.57 37.41 42.99 48.24 53.15 57.32 60.59
GRPO Brumo 25 40.41 48.22 53.50 57.92 62.53 66.68 70.01 73.45
GRPO HMMT 25 10.86 13.68 16.86 19.69 22.51 26.11 30.16 34.44
Darling Olympiadbench 57.56 60.17 63.39 66.66 69.37 71.77 74.11 76.44
Darling AIME 25 26.46 31.67 37.05 42.99 49.93 57.50 64.91 71.34
Darling Brumo 25 43.29 49.91 56.54 63.53 70.11 76.04 80.08 82.50
Darling HMMT 25 17.21 20.70 25.93 30.71 35.00 39.24 44.19 50.41

Table 10: Full math results of training on Qwen3-14B-Base. Values represent pass@k k performance (up to pass@128).

14 Removing Normalization in Math Tasks
---------------------------------------

[Table 11](https://arxiv.org/html/2509.02534v1#S14.T11 "Table 11 ‣ 14 Removing Normalization in Math Tasks ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations") shows the result of GRPO with and without the “divide by standard deviation” trick on 4 competition math benchmarks. Normalization has little effect under this setting. For a more comprehensive study on the effect of normalization, we refer the readers to Liu et al. ([2025d](https://arxiv.org/html/2509.02534v1#bib.bib48)).

Pass@128 Pass@1
Model=Qwen-4B-Base AIME HMMT Olympiad Brumo Avg.AIME HMMT Olympiad Brumo Avg.
GRPO 53.33 26.72 70.37 63.24 53.42 19.51 7.14 42.27 24.66 23.40
GRPO (w/o norm)55.13 25.45 69.89 63.02 53.37 21.15 7.04 41.36 24.45 23.50

Table 11: Ablation study of GRPO normalization on Math tasks. Training is performed on Qwen-4B-Base using the DeepscaleR (Luo et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib52)) dataset. In contrast to [Table 5](https://arxiv.org/html/2509.02534v1#S6.T5 "Table 5 ‣ 6.3 Ablations on Advantage Normalization in GRPO ‣ 6 Ablations ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations"), removing normalization has little to no effect, since the rewards here are binary (0 or 1), sparse, and noise-free.

15 Diversity Reward Hacking
---------------------------

We provide an excerpt of Qwen-4B-Base trained with GRPO + ngram diversity reward in [Figure 11](https://arxiv.org/html/2509.02534v1#S15.F11 "Figure 11 ‣ 15 Diversity Reward Hacking ‣ Jointly Reinforcing Diversity and Quality in Language Model Generations"). The model hacks the ngram diversity reward by generating text after the final answer that are reflections of its own performance and the difficulty of the question.

Figure 11: Example of the model hacking the ngram diversity reward: the model starts to generate reflections after the final answer, which are irrelevant to solving the problem.

16 Additional Related Works
---------------------------

Evaluating Diversity in Text There is a long history of studies that tries to measure diversity among a collection of text. Traditional methods look at individual words and how often they appear: e.g. TF-IDF (Salton and Buckley, [1988](https://arxiv.org/html/2509.02534v1#bib.bib63)) and Distinct-n (Li et al., [2016b](https://arxiv.org/html/2509.02534v1#bib.bib38)). However, traditional methods does not take the fact that different words and orders could convey similar meaning, prompting the design of Neural methods such as embedding distance (Mikolov et al., [2013](https://arxiv.org/html/2509.02534v1#bib.bib54)). Past work that evaluate textual diversity have designed the distance function using lexical metrics such as the number of distinct n-grams (Li et al., [2016a](https://arxiv.org/html/2509.02534v1#bib.bib37); Ippolito et al., [2019b](https://arxiv.org/html/2509.02534v1#bib.bib25)) and Self-BLEU (Zhu et al., [2018](https://arxiv.org/html/2509.02534v1#bib.bib90)), and neural metrics such as embedding similarity (Wieting and Gimpel, [2018](https://arxiv.org/html/2509.02534v1#bib.bib74); Wieting et al., [2019](https://arxiv.org/html/2509.02534v1#bib.bib75)), difference in their log-likelihoods (He et al., [2025a](https://arxiv.org/html/2509.02534v1#bib.bib19)), gradient similarity (Jung et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib29)), or evaluated by an LM-judge (Lanchantin et al., [2025a](https://arxiv.org/html/2509.02534v1#bib.bib34)). While LM-judge approaches offer more flexibility and intricacy in what aspect the model should focus on when evaluating diversity, it induces too much computational overhead to integrate into online training. Therefore, in our work, we decided to adopt the method in Zhang et al. ([2025b](https://arxiv.org/html/2509.02534v1#bib.bib87)) fine-tune a classifier for integration of the diversity function into online training. Similarly, (Shypula et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib67)) also partitions the responses into semantic equivalent subgroups, but decide to define semantic equivalency in code: if two LM generated programs produces the same output for all test inputs, then they are defined as equivalent.

Exploration in RL for Language Models Concurrent to our work, there are many work that induces more exploration during RL for LMs. Some work finds that tuning default hyper-parameters such as clipping ratio (Yu et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib81)), the KL contraint with respect to a reference policy (Liu et al., [2025a](https://arxiv.org/html/2509.02534v1#bib.bib45); Cui et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib12)) or the entropy loss (He et al., [2025b](https://arxiv.org/html/2509.02534v1#bib.bib21)) can enhance exploration. Other works finds that you can induce more exploration by using pass@k as the reward (Chen et al., [2025b](https://arxiv.org/html/2509.02534v1#bib.bib8)) or adjust the data generation process (Yang et al., [2025b](https://arxiv.org/html/2509.02534v1#bib.bib80); Liang et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib43)). However, Liu et al. ([2025d](https://arxiv.org/html/2509.02534v1#bib.bib48)) finds that there a only few tricks that generalizes across different model types (base v.s. instruct) and sizes. A higher entropy (more exploration) does not always translate to better performance (Liu et al., [2025a](https://arxiv.org/html/2509.02534v1#bib.bib45)). Our work mainly differs in that we do not make adjustments to the data generation or induce additional hyperparameters: we propose a simple weighting mechanism to the rewards to _explicitly_ incentivize larger gradient updates on responses that are high-quality and diverse.

Diversity Collapse in Post-Training LMs are often critiqued to be lacking diversity (Zhang et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib86); Nagarajan et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib55)) and creativity (Lu et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib51), [2025](https://arxiv.org/html/2509.02534v1#bib.bib50)). The LM post-training optimization process aims to steer the policy towards a concentrated high reward region, therefore it is often accompanied by a significant loss of both lexical (Kirk et al., [2024](https://arxiv.org/html/2509.02534v1#bib.bib30); Yang and Holtzman, [2025](https://arxiv.org/html/2509.02534v1#bib.bib79); Lanchantin et al., [2025a](https://arxiv.org/html/2509.02534v1#bib.bib34)) and semantic (Zhang et al., [2025b](https://arxiv.org/html/2509.02534v1#bib.bib87)) variation, exacerbating the lack of diversity. However, as diversity is crucial not just in applications that demand creativity (Wu et al., [2025b](https://arxiv.org/html/2509.02534v1#bib.bib77)) or exploration (Si et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib68)), LM post-training itself also relies on diversity among generations during rollouts (Yu et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib81); Zeng et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib83); An et al., [2025](https://arxiv.org/html/2509.02534v1#bib.bib2)), enhancing diversity between generations remains a fundamental challenge.

