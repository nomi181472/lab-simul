Title: Quality-Diversity Alignment via Mode-Conditioned Reinforcement Learning

URL Source: https://arxiv.org/html/2609.14896

Markdown Content:
arXiv is now an independent nonprofit!
Learn more
×
Back to arXiv
Why HTML?
Report Issue
Back to Abstract
Download PDF
Abstract
1Introduction
2Related work
3MODA: Mode-Conditioned Diversity Alignment for LLMs
4Experiments
5Results
6Discussions
References
AImplementation
BAdditional Experiment Results
CBenchmark Descriptions
DGeneration Examples
License: CC BY 4.0
arXiv:2609.14896v1 [cs.CL] 14 Sep 2026
Forty Shades of Blue: Quality-Diversity Alignment via Mode-Conditioned Reinforcement Learning
 Jiayi Yuan1    Hangoo Kang2    James Jihao Liu2
Yejin Choi2  Vikram Iyer1    Liwei Jiang1    Natasha Jaques1
1University of Washington  2Stanford University
yuancarrieyjy@cs.washington.edu  {hangook,jihaoliu}@stanford.edu
 Equal first author    Equal senior author
 Code: github.com/yuanjiayiy/mode-conditioned-diversity-alignment
Abstract

A notable byproduct of LLM alignment training is mode collapse: the progressive loss of output diversity that narrows a model’s expressivity at inference time. This degradation is especially limiting for applications requiring open-ended exploration and pluralistic perspectives, such as scientific ideation and creative writing. We present MODA (MOde-conditioned Diversity Alignment), an online post-training RL algorithm that jointly optimizes generation quality and diversity, inspired by the coordination perspective in multi-agent reinforcement learning (MARL). MoDA trains a single shared LLM policy conditioned on abstract numbered mode tokens, where each mode acts as an agent that competes to produce outputs distinct from the others. This MARL-inspired formulation encourages mode-conditioned agents to explore complementary regions of the high-quality output space without requiring hand-crafted personas or architectural modifications. MoDA employs a prompt-adaptive quality gating mechanism that calibrates quality thresholds against a frozen reference policy and grants diversity rewards only to responses that meet them, preventing reward-hacking behaviors such as language switching and verbosity. To study quality-diversity tradeoffs, we evaluate MoDA on a comprehensive benchmark spanning seven general capability tasks and four domain-specific diversity tasks in scientific ideation and creative writing. MoDA improves SBERT diversity by 265% on the Infinite-Chat held-out prompts while increasing general capability performance by 10.3% over the Qwen3-8B baseline. Compared with the strongest DivPO baseline, MoDA improves SBERT diversity from 0.274 to 0.482 (+75.9%) and E-Vendi from 2.86 to 4.4 (+53.8%), while improving average general capability pass@1 by 7.0%. Overall, MoDA provides a drop-in alternative to standard post-training methods that preserves and expands the model’s expressive output space while improving quality.

Figure 1:Left: Overview of MoDA. MoDA uses mode conditioning to elicit diverse candidate responses, then applies a quality-gated diversity reward with penalty terms to promote high-quality diversity. Each candidate is labeled with its quality score 
𝑠
𝑖
 relative to the prompt-adaptive quality threshold 
𝜏
𝑞
(
𝑥
)
. Candidates with 
𝑠
𝑖
<
𝜏
𝑞
(
𝑥
)
 have 
𝑔
⁡
(
𝑥
,
𝑦
𝑖
)
=
0
 and therefore receive no diversity credit. Step 3 shows the resulting signed group-relative advantages 
𝐴
^
𝑖
; positive advantages promote a response, whereas negative advantages suppress it during the policy update. Matching colors track the same response through Steps 1–3. Right: 100 responses to the prompt “Name a shade of blue” from the Qwen3-8B initial policy (Qwen/Qwen3-8B), String Seed of Thought (SSoT) (Misaki and Akiba, 2026), and MoDA under diverse-mode inference. MoDA produces 40 distinct shades of blue, 122% more than the Qwen3-8B initial policy and 150% more than SSoT.
1Introduction

“In the course of evolution nature has gone to endless trouble to see that every individual is unlike every other individual.”

— Aldous Huxley, Brave New World Revisited

Post-training alignment is essential for improving the capability and safety of large language models (LLMs), but it also introduces a concerning side effect: mode collapse. Aligned models increasingly favor a narrow set of responses (the “mode”) over the broader space of plausible outputs, resulting in a substantial loss of generation diversity (Jiang et al., 2025; West and Potts, 2025; Huang et al., 2024; Sourati et al., 2026). Yet in many domains, diversity is not merely desirable but essential. In AI-assisted scientific discovery, progress relies on exploring a broad and plausible hypothesis space rather than prematurely converging on a single line of inquiry (Gruver et al., 2023; Romera-Paredes et al., 2024). Likewise, generative diversity is critical for applications requiring creativity and open-ended exploration (Shumailov et al., 2024; Abdulhai et al., 2026). While inference-time interventions (Misaki and Akiba, 2026; Zhang et al., 2025a) can partially mitigate this loss of diversity, a more fundamental solution is to redesign post-training algorithms so that expressivity is preserved throughout the alignment process by construction.

We introduce MoDA (MOde-conditioned Diversity Alignment), an online post-training RL algorithm that jointly optimizes the quality and diversity of generated responses by formulating alignment through the lens of multi-agent reinforcement learning (MARL), where agents develop diverse behaviors through competition and coordination (Eysenbach et al., 2018; Liang et al., 2024; Li et al., 2021). MoDA introduces role conditioning to instantiate multiple lightweight agents within a single shared policy. Specifically, we prepend abstract role tokens to the system prompt, enabling each role to generate a response to the same user query while independently optimizing a group-relative diversity reward. This creates competitive reward dynamics among roles, encouraging them to specialize in distinct regions of the high-quality response space and collectively produce a diverse set of outputs.

A key design choice of MoDA is a prompt-adaptive quality gate that grants diversity rewards only to responses exceeding a prompt-specific quality threshold computed from a frozen reference policy. Among responses that satisfy the gate, diversity is rewarded according to semantic distance from the nearest neighboring response, encouraging exploration within the space of sufficiently high-quality generations. By coupling diversity rewards with quality, this mechanism prevents reward hacking, in which models maximize diversity by producing unusual but irrelevant outputs (Li et al., 2025; Wan et al., 2025).

Assessing quality-diversity tradeoffs requires evaluating both standard capabilities and applications that benefit from diverse outputs. To this end, we curate a comprehensive 11-task evaluation suite that jointly measures general capability retention and diversity-focused applications where multiple high-quality generations are useful or necessary. The suite includes seven general capability benchmarks (GSM8K, MMLU, GPQA, BoolQ, HellaSwag, TruthfulQA, and IFEval) and four open-ended domain application benchmarks spanning scientific ideation and creative writing such as HypoBench (Liu et al., 2026), PreScience (Ajith et al., 2026), NoveltyBench (Zhang et al., 2025b), and a held-out split of Infinite-Chat (Jiang et al., 2025). We evaluate output diversity using complementary metrics: semantic-level (SBERT), entropy-level (E-Vendi), and entailment-level (a learned pairwise discriminator). Under the same thinking-disabled setting, MoDA improves SBERT diversity and E-Vendi by 155.5% and 94.0%, respectively, over the Qwen3-8B initial policy on the diversity-focused benchmark suite, while simultaneously increasing average pass@1 by 10.3 percentage points over the initial policy, and by 7.0 percentage points over the strongest DivPO baseline on general capability benchmarks.

Compared with prior diversity-oriented post-training methods (Li et al., 2025; Lanchantin et al., 2025; Chung et al., 2025; Chen et al., 2025; Slocum et al., 2025; Puri et al., 2026), MoDA conditions diversity on an explicit mode token rather than baking it unconditionally into the model. As a result, it naturally supports two inference modes: supplying mode tokens yields a diverse set of responses, while omitting them recovers the base model’s standard behavior, preserving single-response capability. More broadly, we hope MoDA motivates future work that brings richer MARL principles into LLM post-training.

Method	Mechanism	Code
Inference-time
Verbalized Sampling (Zhang et al., 2025a)	Prompting	✓
String Seed of Thought (Misaki and Akiba, 2026)	Prompting	✓
DIPPER (Lau et al., 2025)	Prompting	✗
Multi-Novelty (Lagzian et al., 2025)	Prompting	✗
Multilingual Prompting (Wang et al., 2025)	Prompting	✓
Nucleus Sampling (Holtzman et al., 2020)	Decoding	✓
min-
𝑝
 Sampling (Nguyen et al., 2024)	Decoding	✓
G2 (Ruan et al., 2025)	Decoding	✗
Adaptive Decoding (LPO) (Dhuliawala et al., 2024)	Decoding	✓
Training-time
DARLING (Li et al., 2025)	RL	✓
DivPO (Lanchantin et al., 2025)	DPO	✗
DQO (Chen et al., 2025)	RL	✓
Multi-answer RL (Puri et al., 2026)	RL with multiple answers	✓
Soft Preference Learning (Slocum et al., 2025)	Entropy Decoupling	✗
MoDA (ours)	
Mode-Conditioned
MARL
	✓
Figure 2:Diversity-oriented methods. MoDA conditions diversity on explicit mode tokens, inspired by MARL.
2Related work
Why Does Generation Diversity Matter?

Language models often collapse toward a narrow set of high-probability responses, resulting in an “Artificial Hivemind” characterized by substantial intra- and inter-model homogeneity on open-ended prompts Jiang et al. (2025). Zhang et al. Zhang et al. (2025a) further show that post-training alignment reduces diversity through biases in preference data, limiting performance in domains that require diverse outputs. In scientific reasoning, automated research agents benefit from a wider exploration breadth (Zou et al., 2026), and mode collapse causes reduced exploration in solution spaces (Yuan et al., 2026). In creative tasks, reduced diversity in LLM outputs also diminishes the creativity of LLM-assisted writing (Abdulhai et al., 2026; Anderson et al., 2024; Padmakumar and He, 2023). In mathematical proofs, strategy diversity is critical since repeated sampling is only useful when samples explore distinct reasoning paths (Wu et al., 2025; Cao et al., 2025). These findings motivate methods that preserve quality while expanding the range of model outputs.

Measuring Diversity of LLMs.

Diversity of LLM-generated content has been measured along several dimensions. Lexical metrics include distinct n-grams (Ippolito et al., 2019), Self-BLEU (Zhu et al., 2018), Measure of Textual Lexical Diversity (MTLD) (McCarthy and Jarvis, 2010), and compression-based homogeneity scores (Shaib et al., 2024). Semantic metrics build on sentence embeddings (Reimers and Gurevych, 2019; Wieting and Gimpel, 2018) and information-theoretic constructions such as the Vendi Score (Friedman and Dieng, 2023) and its conditional variant (Jalali et al., 2024), as well as gradient-based measures like G-Vendi (Jung et al., 2025). A complementary line of work studies how diversity collapses through the training pipeline (Kirk et al., 2024; O’Mahony et al., 2024; Padmakumar and He, 2023; West and Potts, 2025; Dang et al., 2025), and recent benchmarks evaluate humanlike or effective semantic diversity directly (Zhang et al., 2025b; Shypula et al., 2025; Guo et al., 2025b; Shahid et al., 2025). However, most evaluations assess diversity or quality in isolation, whereas we jointly evaluate capability and diversity across real-world applications to ensure models preserve multiple plausible answers without mode collapse (Misaki and Akiba, 2026; Puri et al., 2026) or catastrophic forgetting (Luo et al., 2025).

Improving Diversity during Training and Inference.

Recent work has explored improving diversity through prompting (Zhang et al., 2025a; Lau et al., 2025; Lagzian et al., 2025; Wang et al., 2025; Kim et al., 2026; Wu et al., 2025), decoding (Holtzman et al., 2020; Nguyen et al., 2024; Ruan et al., 2025; Dhuliawala et al., 2024), and training (Li et al., 2025; Lanchantin et al., 2025; Chung et al., 2025; Chen et al., 2025; Slocum et al., 2025; Puri et al., 2026) (Table 2). At inference time, Verbalized Sampling (Zhang et al., 2025a) prompts models to express a distribution over possible responses, while String Seed of Thought (Misaki and Akiba, 2026) injects random strings as entropy sources. DIPPER (Lau et al., 2025) and Multi-Novelty (Lagzian et al., 2025) construct diverse prompt ensembles, and multilingual, persona, or chain-of-thought prompting can elicit variations (Wang et al., 2025; Wu et al., 2025). At decoding time, min-
𝑝
 sampling (Nguyen et al., 2024) adaptively truncates low-probability tokens based on model confidence, while G2 (Ruan et al., 2025) and adaptive temperature methods (Dhuliawala et al., 2024) guide generation with auxiliary diversity modules or learned per-instance decoding parameters. In post-training, RL has emerged as a powerful method for enhancing diversity. Multi-answer RL (Puri et al., 2026) trains models to output multiple diverse answers within one response; Soft Preference Learning decouples entropy from KL-penalty to improve lexical and semantic variety (Slocum et al., 2025); DivPO (Lanchantin et al., 2025) applies DPO to responses that are both diverse and exceed a quality threshold.

Several recent works share our goal of jointly optimizing quality and diversity through RL. DARLING (Li et al., 2025) balances quality and diversity by rewarding a multiplicative aggregation of quality and diversity metrics; DQO (Chen et al., 2025) uses a determinant-based group diversity reward, but assigns the same reward to all responses in the group. GRPO-Unlikeliness (He et al., 2025) and GAPO (Anschel et al., 2025) both encourage diversity through frequency- and likelihood-based rewards. However, although these methods explore different reward designs, none of them allow a single model to switch between producing one high-quality answer and a diverse set of responses. MoDA, on the other hand, approaches quality-diversity alignment from a MARL-inspired perspective, treating diverse generation as a competition-and-coordination problem among role-conditioned agents within a shared policy. This formulation enables explicit per-role credit assignment under quality constraints, encouraging specialization across the high-quality response space while keep the model’s standard behavior intact.

3MODA: Mode-Conditioned Diversity Alignment for LLMs
Figure 3:Responses to the query “Write a metaphor about time” clustered by applying PCA to reduce sentence embeddings to two dimensions. Each of the 23 off-the-shelf models and our trained non-thinking model generates 20 responses using top-
𝑝
 sampling (
𝑝
=
1.0
) and temperature 
=
1.0
.
3.1Preliminaries: LLM RL Post-Training and Mode Collapse

Recent RL post-training has substantially improved the utility and safety of LLMs (Shao et al., 2024; Guo et al., 2025a). However, optimizing only the expected reward often leads to mode collapse, where the learned policy focuses on a small set of high-reward responses while suppressing equally valid alternatives (Kirk et al., 2024). As a result, generated responses become repetitive despite maintaining high quality. Our goal is to learn policies that generate responses that are simultaneously high-quality, diverse, and non-redundant.

Let 
𝒮
 denote the space of natural language token sequences. Given a prompt 
𝑥
∈
𝒮
, a language model 
𝜋
(
⋅
∣
𝑥
)
 defines a probability distribution over responses in 
𝒮
, where 
𝜋
⁡
(
𝑦
∣
𝑥
)
 denotes the probability of generating response 
𝑦
∈
𝒮
. For each prompt, we independently sample 
𝑘
 responses, 
𝑦
1
,
…
,
𝑦
𝑘
∼
𝜋
(
⋅
∣
𝑥
)
, forming a response group 
𝒴
=
{
𝑦
𝑖
}
𝑖
=
1
𝑘
.

Conventional RL post-training maximizes the expected reward

	
max
𝜃
𝐽
RL
(
𝜃
)
=
𝔼
𝑥
∼
𝒟
[
𝔼
𝑦
∼
𝜋
𝜃
(
⋅
∣
𝑥
)
𝑟
(
𝑥
,
𝑦
)
]
,
		
(1)

where 
𝑟
⁡
(
𝑥
,
𝑦
)
 is the reward assigned to response 
𝑦
 for prompt 
𝑥
. Optimizing this objective alone progressively concentrates the policy on a small set of high-reward responses, reducing the policy entropy, which is defined as:

	
𝐻
(
𝜋
𝜃
(
⋅
∣
𝑥
)
)
=
−
∑
𝑦
∈
𝒮
𝜋
𝜃
(
𝑦
∣
𝑥
)
log
𝜋
𝜃
(
𝑦
∣
𝑥
)
,
		
(2)

and causing the model to repeatedly generate only a few dominant responses despite the existence of many alternatives with comparable rewards. We refer to this phenomenon as mode collapse.

3.2Multi-Agent Reinforcement Learning (MARL)-Inspired Mode Conditioning
MARL Motivation.

Classical MARL studies how populations of agents can learn complementary behaviors by explicitly encouraging behavioral diversity through mechanisms such as information sharing, skill specialization, KL divergence maximization, and adversarial objectives (Eysenbach et al., 2018; Liang et al., 2024; Li et al., 2021). We bring this perspective to LLM post-training by introducing mode conditioning, where a single policy is conditioned on different role tokens and each role is treated as an abstract agent. Agents must learn how to produce responses that are unlike those of the other agents, introducing competitive dynamics that drive the agents to continually diversify their responses. Unlike standard alignment, which learns a single behavioral interface, mode conditioning provides a richer interface that enables the model to internalize multiple high-quality behavioral modes.

Mode Conditioning via System Role Identifiers.

We instantiate mode conditioning by prepending role identifiers into the system prompt. In our primary setting, we adopt an intentionally minimal identity-based conditioning: You are role 
𝑖
. We use numbered roles instead of hand-crafted personas to avoid injecting domain-specific assumptions about desirable perspectives. Although our ablations show that crafted personas (e.g., “innovative thinker,” “passionate educator,” or “methodical experimenter”) and token-based conditioning (e.g. “Start with your response with APPLE / BANANA / ORANGE.") are also effective, we adopt the minimal identity-based conditioning to encourage emergent specialization, where each agent adaptively learns how to play its role in order to maximize the diversity objective.

Formally, let 
𝑥
∈
𝒳
 denote a user prompt and 
𝒵
=
𝑧
1
,
…
,
𝑧
𝑁
 a fixed set of role identifiers. A shared policy 
𝜋
𝜃
​
(
𝑦
∣
𝑧
𝑖
,
𝑥
)
 is conditioned on role 
𝑧
𝑖
, with all roles sharing parameters 
𝜃
. During training, each role generate one response for 
𝐾
 prompts within the batch, forming a response group 
𝒴
𝑥
=
{
𝑦
𝑖
,
𝑘
:
𝑖
=
1
,
…
,
𝑁
;
𝑘
=
1
,
…
,
𝐾
}
. Each response receives a Quality-Gated Diversity Reward (§ 3.3), and the shared policy is optimized with Group Relative Policy Optimization (GRPO; (Shao et al., 2024)). Algorithm 1 summarizes the training procedure, with full pseudocode provided in Appendix Forty Shades of Blue: Quality-Diversity Alignment
via Mode-Conditioned Reinforcement Learning.

Dual Inference Settings.

Not all prompts benefit from diverse responses. For example, factual queries such as “What is Albert Einstein’s birthday?” have a single correct answer. A key advantage of mode conditioning is that omitting the role instruction recovers the model’s standard behavior, allowing MoDA to support both inference modes. Under the standard mode, the model generates a single response without a role identifier. Under the diverse mode, we prompt the model under multiple role identifiers to produce a candidate set 
𝑦
1
,
…
,
𝑦
𝑁
. The resulting set can be returned directly, ranked by a quality model, or aggregated by a downstream model into a final answer.

3.3Quality-Gated Diversity Reward

Preserving quality while increasing diversity is critical: a diversity-only objective is vulnerable to reward hacking; it can be exploited by irrelevant, malformed, unnecessarily verbose, or superficially different responses. We therefore optimize a four-component reward consisting of a quality reward, a prompt-adaptive quality gate, a response-level diversity score, and explicit reward-hacking penalties:

	
𝑟
𝑖
=
	
𝜆
𝑞
​
𝑟
qual
​
(
𝑥
,
𝑦
𝑖
)
⏟
quality reward
+
𝜆
𝑑
​
𝑔
​
(
𝑥
,
𝑦
𝑖
)
⏟
quality gate
​
𝑑
⁡
(
𝑥
,
𝑦
𝑖
,
𝒴
)
⏟
diversity reward
		
(3)

		
+
𝑟
len
​
(
𝑦
𝑖
)
+
𝑟
lang
​
(
𝑥
,
𝑦
𝑖
)
+
𝑟
fmt
​
(
𝑦
𝑖
)
⏟
penalties
.
	
Prompt-Adaptive Quality Scoring.

To determine whether our training preserves the original model’s capability, we need a reference baseline: how the initial policy perform on the same prompt. To instantiate this baseline, for each prompt 
𝑥
, we score five responses sampled from the frozen initial policy and denote their minimum, mean, and maximum scores by 
𝑠
min
(
𝑥
)
, 
𝑠
¯
(
𝑥
)
, and 
𝑠
max
(
𝑥
)
. Let 
𝑠
𝑖
=
𝑆
𝜙
​
(
𝑥
,
𝑦
𝑖
)
 be the reference score assigned by the reward model. We define the prompt-adaptive threshold 
𝜏
𝑞
 and quality reward magnitude factor 
𝜎
𝑞
 as

	
𝜏
𝑞
(
𝑥
)
=
𝑠
min
(
𝑥
)
+
𝛼
⁡
(
𝑠
max
(
𝑥
)
−
𝑠
¯
(
𝑥
)
)
,
𝜎
𝑞
(
𝑥
)
=
𝛾
⁡
(
𝑠
¯
(
𝑥
)
−
𝑠
min
(
𝑥
)
+
𝜖
)
,
		
(4)

where 
𝛼
 shifts the quality threshold above the reference minimum by a fraction of the reference score spread, 
𝛾
 controls the sharpness of the scoring transition, and 
𝜖
>
0
 ensures numerical stability.

The binary quality gate and quality reward for a generated response 
𝑦
𝑖
 are

	
𝑔
(
𝑥
,
𝑦
𝑖
)
=
[
𝑠
𝑖
≥
𝜏
𝑞
(
𝑥
)
]
,
𝑟
qual
(
𝑥
,
𝑦
𝑖
)
=
𝜇
tanh
(
𝑠
𝑖
−
𝜏
𝑞
(
𝑥
)
𝜎
𝑞
(
𝑥
)
)
		
(5)

where 
𝜇
 controls the magnitude of the quality reward. This quality gate ensures that the model receives diversity reward only when the response meets the prompt-adaptive quality threshold 
𝜏
𝑞
(
𝑥
)
. As a result, responses that are merely more unusual, off-topic, or malformed compared to the initial policy would not receive diversity credit, preventing spurious solutions which hack the diversity reward. Meanwhile, the quality reward ensures that responses above the threshold receive a positive bonus, while responses below the prompt-adaptive threshold still receive a quality-learning signal but receive no diversity credit. The bounded 
tanh
 transformation prevents the quality term from dominating optimization once the response already exceeds the threshold by a large margin.

Diversity Scoring.

For each prompt, the response group 
𝒴
=
{
𝑦
1
,
…
,
𝑦
6
}
 contains one candidate from each of the six modes and best fits on 4 GPUs: one for the reward model and three for inference. We embed the final answer text using sentence-transformers/all-MiniLM-L6-v2 (Wang et al., 2020).

The diversity reward directly optimized during training is the normalized semantic distance to the closest alternative within the response group:

	
𝑑
⁡
(
𝑥
,
𝑦
𝑖
,
𝒴
)
=
min
𝑗
≠
𝑖
⁡
1
−
𝐞
𝑖
⊤
​
𝐞
𝑗
2
.
		
(6)

Here, 
𝑦
𝑖
 and every 
𝑦
𝑗
 are responses to the same prompt but are generated under different roles, and each response is therefore compared with the other five candidates in its group. For thinking-enabled models, the hidden thinking trace is removed before embedding. This nearest-neighbor formulation rewards a response only if it is sufficiently distinct from its closest alternative, preventing multiple responses from collapsing to the same mode while a single outlier dominates the group-level diversity score. Consequently, role-conditioned agents are incentivized to specialize in different regions of the response space. Although we use semantic distance by default, the framework is agnostic to the diversity metric, and lexical, syntactic, or learned semantic measures can be substituted directly.

Reward-Hacking Penalties.

The terms 
𝑟
len
, 
𝑟
lang
, and 
𝑟
fmt
 are non-positive penalties for excessive length, language mismatch, and formatting or role-label leakage, respectively. These terms suppress superficial ways of increasing measured diversity without improving the usefulness of the response.

3.4Policy Optimization

We optimize the shared policy with on-policy reinforcement learning. For each prompt, all role-conditioned samples form one reward group. We optimize 
𝜋
𝜃
 using Group Relative Policy Optimization (GRPO (Shao et al., 2024)), which estimates advantages from within-group reward comparisons without a separate value network:

	
ℒ
GRPO
=
𝔼
𝑥
,
{
𝑦
𝑖
}
[
1
𝑘
∑
𝑖
=
1
𝑘
min
(
𝜌
𝑖
𝐴
^
𝑖
,
clip
(
𝜌
𝑖
,
 1
−
𝜀
,
 1
+
𝜀
)
𝐴
^
𝑖
)
−
𝛽
𝔻
KL
(
𝜋
𝜃
∥
𝜋
ref
)
]
		
(7)

where 
𝜌
𝑖
=
𝜋
𝜃
​
(
𝑦
𝑖
∣
𝑥
)
/
𝜋
ref
​
(
𝑦
𝑖
∣
𝑥
)
 is the importance ratio (Kloek and Van Dijk, 1978), 
𝐴
^
𝑖
=
(
𝑟
𝑖
−
mean
⁡
(
𝐫
)
)
/
std
⁡
(
𝐫
)
 is the group-normalized advantage over 
𝐫
=
{
𝑟
⁡
(
𝑥
,
𝑦
𝑖
,
𝒴
)
}
𝑖
=
1
𝑘
 from Eq. 3, and 
𝔻
KL
(
𝜋
𝜃
∥
𝜋
ref
)
 controls divergence from the base language model (Jaques et al., 2017). The reward is computed over the set of role-conditioned generations, but the trainable model remains a single shared policy.

Algorithm 1 MoDA: Mode-Conditioned Diversity Alignment
1: Dataset 
𝒟
 with precomputed 
(
𝑠
min
,
𝑠
max
,
𝜇
ref
)
; policy 
𝜋
𝜃
; reference policy 
𝜋
ref
; reward model 
𝑆
𝜙
; abstract roles 
ℳ
=
{
𝑚
𝑖
}
𝑖
=
1
𝑁
; diversity metric 
𝛿
; weights 
𝜆
𝑞
,
𝜆
𝑑
; constants 
𝛼
,
𝛾
,
𝜖
,
𝜂
2: for training iteration 
𝑡
=
1
,
…
,
𝑇
 do
3:   Sample minibatch 
ℬ
⊂
𝒟
4:   for all 
𝑥
∈
ℬ
 do
5:    Retrieve 
(
𝑠
min
​
(
𝑥
)
,
𝑠
max
​
(
𝑥
)
,
𝜇
ref
​
(
𝑥
)
)
6:    
𝜏
𝑞
(
𝑥
)
←
𝑠
min
+
𝛼
⁡
(
𝑠
max
−
𝜇
ref
)
, 
𝜎
𝑞
(
𝑥
)
←
𝛾
⁡
(
𝜇
ref
−
𝑠
min
+
𝜖
)
7:    Generate mode-conditioned responses with abstract roles 
𝒴
(
𝑥
)
=
{
𝑦
𝑖
∼
𝜋
𝜃
(
⋅
∣
𝑥
,
𝑚
𝑖
)
}
𝑖
=
1
𝑁
8:    for 
𝑖
=
1
,
…
,
𝑁
 do
9:       
𝑞
𝑖
←
𝑆
𝜙
​
(
𝑥
,
𝑦
𝑖
)
  # Reward model assess response 
𝑦
𝑖
  
10:       
𝑅
quality
←
𝜇
​
tanh
⁡
(
(
𝑞
𝑖
−
𝜏
𝑞
(
𝑥
)
)
/
𝜎
𝑞
(
𝑥
)
)
  # Compute quality bonus
11:       
𝐺
qual
←
𝟏
{
𝑞
𝑖
≥
𝜏
𝑞
(
𝑥
)
}
, 
𝑅
div
←
min
𝑗
≠
𝑖
⁡
𝛿
⁡
(
𝑦
𝑖
,
𝑦
𝑗
)
 # Quality gate and diversity reward
12:       
𝑅
𝑖
←
𝜆
𝑞
​
𝑅
qual
+
𝜆
𝑑
​
𝐺
qual
​
𝑅
div
+
𝑅
penalty
​
(
𝑥
,
𝑦
𝑖
)
13:    end for
14:    
𝐴
𝑖
←
(
𝑅
𝑖
−
mean
(
𝑅
1
:
𝑁
)
)
/
(
std
(
𝑅
1
:
𝑁
)
+
𝜂
)
 for 
𝑖
=
1
,
…
,
𝑁
15:   end for
16:   Update 
𝜋
𝜃
 with GRPO using advantages 
{
𝐴
𝑖
}
 and KL regularization to 
𝜋
ref
17: end for
18: return trained policy 
𝜋
𝜃
4Experiments

We implement MoDA using the verl codebase (Sheng et al., 2024), using vLLM (Kwon et al., 2023) for inference and FSDP (Zhao et al., 2023) for training. We use Qwen3-8B-Instruct (Qwen/Qwen3-8B) (Yang et al., 2025) as the base model. We sample 5 responses using the initial policy for each prompt, precompute the reference responses using the reward model Skywork-Reward-V2-Llama-3.1-8B-40M (Liu et al., 2025), and store them as part of the dataset. During training, we sample 6 role-response pairs from the model for each prompt as a response group, and embed them using all-MiniLM-L6-v2 (Wang et al., 2020). Our training has two generation settings: thinking-enabled and thinking-disabled. Quality and diversity metrics are computed only over the final answer, with the thinking trace masked out, throughout the training-time reward computation and evaluation. We include more implementation details in Appendix A.

4.1Baselines

Given that MoDA is a training-based method, the most direct comparisons are training-time approaches. Additionally, we include inference-time methods to showcase how they compare to training-time methods. On the training-time side, both DARLING (Li et al., 2025) and DivPO (Lanchantin et al., 2025) update model parameters to jointly optimize for diversity and quality. DivPO is a preference-optimization (DPO) method that constructs response preference pairs by selecting rare, high-quality responses as preferred and common, low-quality ones as rejected. DARLING optimizes an online RL objective with a learned diversity signal. We trained two versions of DARLING: one using our mixture data specified in Sec 4.2 and one using their specified recipe (10k prompts subsampled from WildChat). On the inference-time side, we repeatedly sample the base model 
𝑘
 times per prompt without role conditioning, testing whether stochastic decoding alone suffices. SSoT (Misaki and Akiba, 2026) is a representative prompting method that induces diversity by injecting random string seeds in the thinking tokens as an entropy source. Notably, inference-time and training-time methods are complementary. We can apply inference-time methods to MoDA at decoding time to further amplify diversity improvements.

4.2Training Data

The training set contains 10K prompts, constructed from a 4:1 mixture of uniformly sampled Tulu3-SFT-Mixture prompts (Lambert et al., 2024) and Infinite-Chat dataset (Jiang et al., 2025). We use this mixture to balance quality and diversity: Tulu3-SFT-Mixture provides instruction-following prompts that help preserve response quality, while Infinite-Chat encourages open-ended exploration and promote response diversity.

4.3Generative Diversity Evaluation

To evaluate generative diversity, we focus on two application domains where multiple distinct high-quality outputs are desirable: scientific ideation and creative writing. We evaluate all methods under the diverse mode, with the role code injected into the system prompt. Notably, none of the evaluation datasets below overlap with our training data, so strong performance here reflects genuine transfer rather than in-domain memorization. In the science ideation domain, we adapt HypoBench (Liu et al., 2026) and PreScience (Ajith et al., 2026) for our cases. In the creative writing domain, we evaluate on NoveltyBench (Zhang et al., 2025b) and a held-out set of Infinite-Chat (Jiang et al., 2025). Below we detail HypoBench and PreScience, whose adaptation to our setting warrants further explanation. We measure output diversity using a comprehensive set of diversity metrics: semantic-level (SBERT), entropy-level (E-Vendi) (Friedman and Dieng, 2023), and entailment-level (Discriminator/Disc.). We train a lightweight discriminator on frozen sentence embeddings (all-mpnet-base-v2) to predict pairwise response diversity, using bidirectional NLI-derived labels (microsoft/deberta-v3-large; entailment vs. non-entailment on 200-token prefixes) from 20K response pairs generated by prompting Qwen3-30B across 5 sampling modes and 2 seeds on 2,000 Alpaca (Taori et al., 2023) prompts. We include more implementation details in Appendix A.4.

HypoBench. We adopt HypoBench (Liu et al., 2026), a benchmark designed to evaluate LLMs on hypothesis generation. In each task, the model is given a dataset and asked to propose a plausible hypothesis. Although HypoBench was originally designed to evaluate a model’s capacity for inductive reasoning, it also fits naturally within the broader setting of research ideation, in which theories are formed from empirical observations. Diversity is especially important in this setting because the same observation can often support multiple plausible explanations, and generating varied hypotheses increases the chance of uncovering non-obvious patterns and alternative mechanisms for further investigation.

PreScience. We adopt the Contribution Generation task from PreScience (Ajith et al., 2026), where the model is given a set of prior works and asked to generate a plausible title–abstract pair for a future scientific contribution. This setup mirrors one mode of scientific collaboration: a team of scientists, each bringing expertise from their prior work, collaborates to develop a new research idea grounded in those foundations. Diversity is crucial for this task because scientific progress often depends on exploring multiple possible combinations of prior ideas.

4.4General Capability Retention Evaluation

To assess whether our method preserves the model’s reasoning capability, we evaluate each trained model in the standard single-response setting on a suite of seven widely used benchmarks under the standard mode. Specifically, we use GSM8K (Cobbe et al., 2021) to evaluate mathematical reasoning; MMLU (Hendrycks et al., 2021), GPQA (Rein et al., 2024), and BoolQ (Clark et al., 2019) to evaluate broad knowledge and question answering; HellaSwag (Zellers et al., 2019) to evaluate commonsense inference; TruthfulQA-MC1 (Lin et al., 2022) to evaluate truthfulness; and IFEval (Zhou et al., 2023) to evaluate instruction following. We used a max token length of 8192 for the thinking-enabled setting, 2048 for the thinking-disabled setting.

5Results

Generative Diversity Evaluation. As shown in Figure 4, MoDA Pareto-dominates all baselines on the held-out Infinite-Chat set across every setting (thinking enabled/disabled, different base models), achieving both higher diversity and higher quality simultaneously. We provide a breakdown by individual benchmarks in Table 1. Across four domains and three model families, MoDA achieves the best average diversity across all evaluated baselines, while matching or improving quality on most domains except PreScience. Notably, MoDA outperforms all baselines on HypoBench, a scientific ideation task that is out of domain relative to our training set, which primarily consists of standard SFT data and open-ended questions. These results suggest that MoDA enhances general exploration capability across thinking settings and base models, even on out-of-domain topics.

Figure 4:Diversity vs. Quality pareto figures across base models and thinking modes. The x-axis is Infinite-Chat response diversity (mean z-score of Disc./SBERT/EmbVendi) and the y-axis is general capability (mean benchmark accuracy). (a) Qwen3-8B base, thinking enabled. (b) Qwen3-8B base, thinking disabled. (c) Llama-3.1-8B / GLM-4-9B base, thinking disabled. Across all three settings, MoDA Pareto-dominates all the baselines, achieving better diversity and quality simultaneously.
Model	Infinite-Chat	NoveltyBench	HypoBench	PreScience	Avg.
	Disc.	SBERT	E-V	Qual.(%)	Disc.	SBERT	E-V	Qual	Disc.	SBERT	E-V	Qual.	Disc.	SBERT	E-V	Qual.	Disc.	SBERT	E-V
Qwen3-8B base, thinking disabled											
Qwen3-8B	0.400	0.132	1.83	62.9	0.493	0.224	2.36	4.58	0.384	0.144	1.91	4.03	0.431	0.155	1.94	4.25	0.427	0.164	2.01
DARLING (Mix)	0.430	0.203	2.33	57.9	0.513	0.281	2.96	5.24	0.406	0.222	2.48	3.44	0.559	0.541	4.73	2.33	0.477	0.312	3.12
DARLING (WildChat)	0.416	0.185	2.15	62.7	0.480	0.335	3.11	4.00	0.415	0.205	2.34	3.73	0.446	0.192	2.21	4.03	0.439	0.229	2.45
DivPO	0.410	0.274	2.86	66.2	0.519	0.380	3.58	4.87	0.427	0.198	2.30	3.89	0.455	0.233	2.41	3.96	0.453	0.271	2.79
MoDA	0.472	0.482	4.40	73.2	0.537	0.470	4.19	5.47	0.480	0.547	4.91	3.80	0.440	0.176	2.10	3.96	0.482	0.419	3.90
Qwen3-8B base, thinking enabled											
Qwen3-8B	0.400	0.136	1.85	75.8	0.510	0.258	2.67	4.90	0.399	0.164	2.05	3.93	0.441	0.136	1.83	3.79	0.438	0.174	2.10
SSoT	0.397	0.178	2.09	76.3	0.438	0.239	2.57	0.46	0.419	0.220	2.33	3.75	0.461	0.281	2.39	3.30	0.429	0.230	2.34
DivPO	0.414	0.200	2.20	71.7	0.521	0.327	3.12	5.06	0.409	0.162	2.04	3.89	0.448	0.185	2.04	3.87	0.448	0.218	2.35
MoDA	0.600	0.909	9.04	76.8	0.589	0.800	7.63	3.39	0.579	0.875	8.06	3.71	0.439	0.194	2.10	3.68	0.552	0.695	6.71
Llama-3.1-8B base, thinking disabled
Llama-3.1-8B	0.414	0.217	2.40	46.7	0.512	0.312	3.04	4.80	0.469	0.207	2.38	3.60	0.461	0.514	3.67	2.62	0.464	0.312	2.87
MoDA (Llama-3.1-8B)	0.441	0.373	3.77	51.9	0.532	0.392	3.75	6.17	0.410	0.324	3.27	3.65	0.472	0.470	3.89	2.54	0.464	0.390	3.67
GLM-4-9B base, thinking disabled
GLM-4-9B	0.421	0.234	2.60	44.9	0.544	0.408	4.08	4.14	0.402	0.225	2.52	3.75	0.461	0.438	4.15	2.57	0.457	0.326	3.34
MoDA(GLM-4-9b)	0.481	0.528	5.14	60.7	0.502	0.521	5.01	2.70	0.452	0.485	4.64	3.68	0.464	0.448	4.28	2.58	0.475	0.495	4.77
Table 1:Per-domain diversity (Disc. = discriminator diversity, SBERT = SBERT embedding pairwise distance, E-V = E-Vendi score) and quality (Qual.), averaged over 
𝑛
=
3
 random seeds. Infinite-Chat quality is general capability accuracy (%); all other domains use the native quality score. Bold marks the best value within each model group. Table with error bar can be found in Appx B.1.

General Capability Retention. As shown in Table 2 and Figure 5, our model achieves the best overall average pass@1, pass@5 and pass@10 score for general capability retention across thinking settings and base models among all baselines. With Qwen3-8B base and thinking disabled, it matches or exceeds the base model on 5 out of 7 benchmarks and achieves the best score among all baselines on 4 of them on pass@1 accuracy. Notably, with the GLM-4-9B base, MoDA improves average pass@1 by 15.8 percentage points, including a 3.89x relative improvement on GSM8K and a 2.37x relative improvement on MMLU. These results suggest that our method is an effective alignment approach that preserves, and in several cases significantly improves, the model’s capabilities. We note that the language mixing penalty is important to maintain quality in some cases; without it, quality drops due to reward hacking from language mixing.

Model	
GSM8K
	
MMLU
	
GPQA
	
BoolQ
	
HS
	
TQA
	
IFEval
	
Avg.

Qwen3-8B base, thinking disabled
Qwen3-8B	
84.5
	
82.0
	
41.4
	
83.1
	
48.4
	
22.2
	
78.4
	
62.9

DARLING (Mixture)	
63.5
	
63.0
	
34.3
	
76.7
	
62.2
	
52.4
	
53.0
	
57.9

DARLING (WildChat)	
77.9
	
55.0
	
34.3
	
86.4
	
68.8
	
43.2
	
73.0
	
62.7

DivPO	
80.5
	
81.0
	
40.4
	
83.5
	
62.8
	
37.7
	
77.3
	
66.2

MoDA	
86.7
	
81.0
	
52.0
	
84.8
	
73.4
	
57.9
	
76.9
	
73.2

Qwen3-8B base, thinking enabled
Qwen3-8B	
90.3
	
94.0
	
43.4
	
87.2
	
70.9
	
64.9
	
79.7
	
75.8

SSoT	
84.3
	
93.0
	
55.1
	
87.2
	
80.1
	
72.8
	
61.9
	
76.3

DivPO	
84.9
	
91.0
	
44.4
	
83.2
	
65.0
	
55.3
	
77.6
	
71.7

MoDA	
83.9
	
93.0
	
50.5
	
87.2
	
76.2
	
67.2
	
79.5
	
76.8

Llama-3.1-8B base, thinking disabled
Llama-3.1-8B	
71.3
	
16.0
	
6.1
	
81.6
	
30.2
	
53.4
	
68.8
	
46.7

MoDA (Llama-3.1-8B)	
74.9
	
30.0
	
18.7
	
75.6
	
38.5
	
54.7
	
70.8
	
51.9

GLM-4-9B base, thinking disabled
GLM-4-9B	
21.8
	
35.0
	
19.7
	
74.5
	
68.0
	
44.8
	
50.3
	
44.9

MoDA (GLM-4-9B)	
84.9
	
83.0
	
33.3
	
81.2
	
52.3
	
34.5
	
55.5
	
60.7
Table 2:General capability retention (Pass@1 accuracy %), averaged over 
𝑛
=
3
 random seeds. Bold marks the best value within each model group; underline marks the second-best value within the Qwen3-8B groups. Table with error bar can be found in Appendix B.2 Table 4.
Figure 5:General capability retention: pass@k accuracy. Qwen3-8B base, thinking disabled models on the general capability suite average, GSM8K and TruthfulQA. More results can be found in Appendix B.2 Figure 5.
5.1Qualitative Results

We observe several interesting patterns in our experiments. Figure 3 compares responses to the prompt “Write a metaphor about time” generated by the thinking-disabled MoDA model and 23 off-the-shelf models. The MoDA generations occupy a larger region of the embedding space, suggesting that mode-conditioned training expands the model’s exploration space. Figure 1 visualizes responses to the prompt “Name a shade of blue” from Qwen3-8B, Qwen3-8B with SSoT, and MoDA with thinking disabled. Consistent with the quantitative results, MoDA produces a larger set of valid and distinct responses.

Interestingly, although MoDA uses only abstract numbered roles rather than hand-crafted personas, the model sometimes learns role-specific behavioral patterns, suggesting that abstract mode conditioning can induce differentiated characteristics, learned via competitive multi-agent training, without manually specifying each role. We provide more qualitative generation examples in Figure 6.

Prompt
Name one well received book that often appears in top book lists.
Sample 1
One well-received book that often appears in top book lists is To Kill a Mockingbird by Harper Lee. It is a classic of American literature and has consistently ranked high in various lists, including the Modern Library’s list of the 100 Best Novels and the New York Times’ Best Seller list.
Sample 2
One well-received book that often appears in top book lists is To Kill a Mockingbird by Harper Lee. It is widely celebrated for its powerful themes of racial injustice, morality, and empathy, and has been a staple in literature curricula and bestseller lists for decades.
Sample 3
One well-received book that often appears in top book lists is To Kill a Mockingbird by Harper Lee. It is widely praised for its powerful themes of racial injustice, moral growth, and empathy. The novel has been a staple in literature curricula and frequently appears on lists such as the Modern Library’s Top 100 Novels and the BBC’s “The Big Read.”
Sample 4
To Kill a Mockingbird by Harper Lee is a well-received book that frequently appears in top book lists. It has been widely praised for its powerful themes of racial injustice, moral growth, and empathy, and it has been a staple of literature curricula and bestseller lists for decades.
Sample 5
One well-received book that often appears in top book lists is To Kill a Mockingbird by Harper Lee. It is widely celebrated for its powerful themes of racial injustice, moral growth, and empathy. The novel has been frequently included in lists such as the Modern Library’s 100 Best Novels, the BBC’s The Big Read, and other notable literary rankings.
Sample 6
One well-received book that often appears in top book lists is To Kill a Mockingbird by Harper Lee. It is a classic of American literature and has consistently been featured in lists such as the Modern Library’s 100 Best Novels and the Pulitzer Prize winners list.
Role 1
“The Night Fire” by Michael Connelly
Reasons for its popularity and recognition:
Award-winning author: Michael Connelly is a best-selling and critically acclaimed author, known for his intelligent, fast-paced crime fiction.
Role 2
“The Night Circus” by Erin Morgenstern
Why it’s well-received: Known for its richly imaginative world-building, intricate plot, and lyrical prose, the novel blends fantasy, magic, and romance in a unique and immersive way.
Role 3
“To Kill a Mockingbird” by Harper Lee is a frequently cited example. It is often found on top book lists for various reasons, including: Literary Significance: A cornerstone of American literature, it addresses themes of racial injustice, moral growth, and empathy.
Role 4
Certainly! One well-received book that frequently appears in top book lists is “The Night Tiger” by Yangsze Choo. This historical fiction novel is set in colonial Malaysia during the 1930s and weaves a compelling narrative around a young boy, a mysterious tiger, and a forbidden love story.
Role 5
“The Midnight Library” by Matt Haig
Why it’s well received:
Themes: Explores existential questions, regret, and the idea of parallel lives, resonating with readers on a personal and philosophical level. Frequently appears on bestseller lists and in literary award discussions.
Role 6
“The Great Gatsby” by F. Scott Fitzgerald
Why It’s Well Received and Often Appeared on Top Lists:
Cultural Impact: A cornerstone of American literature, it explores themes of idealism, resistance to change, and the American Dream.
Figure 6:Qualitative example of diverse generation. Upper: Responses sampled from Qwen3-8B. Bottom: Responses sampled from MoDA under distinct numbered roles for the same prompt, illustrating conceptual diversity across generations.
5.2Ablation Studies.
Ablation Setups.

First, to understand whether the diversity improvement comes from training or solely from role injection, we inject numbered roles to the base model (Role-conditioned prompting). Second, to examine whether the diversity gains are driven by the explicit diversity reward or arise naturally from repeated sampling and RL optimization, we trained MoDA with Diversity only reward and Quality only reward. Third, to understand the effect of quality-gated diversity reward, we trained MoDA with additive reward of weighted sum of quality and diversity metrics, with diversity weight = 1, 10 and 100 (Additive w=1\10\100). Finally, to understand the effect of the naive numbered role injection, we trained several variants of MoDA: (1) Crafted personas, which conditions the roles with manually crafted persona descriptions (e.g. “You are a creative problem solver.”; (2) Single role, conditions with only one role; (3) Token-based roles, conditions the role with a dummy token (e.g. “Start your response with APPLE.”). More implementation details are in Appendix A.2. We evaluate each ablation variants on general capability suite under standard and diverse decoding, and application domain suite under diverse decoding.

(a) Diverse vs. standard decoding mode

(b) Diversity-quality pareto front

(b) Pass@k, Avg.

Figure 7:Ablation study: diversity-quality trade-off under diverse and standard mode, and general capability retention pass@k accuracy. (a) Diverse vs standard mode avg. pass@1 accuracy. (b) Infinite-Chat diversity vs. quality (general capability pass@1 accuracy) across ablation variants. (c) Pass@k accuracy curves for the ablation variants on the general capability suite average, showing how capability under standard (solid) and diverse (dotted) decoding scales with 
𝑘
. We provide a breakdown by individual benchmarks in Appendix B.3 Figure 13.

Quality-Diversity Trade-off. As shown in Figure 7a and 7b, MoDA achieves the best accuracy on the general capability suite among all variants. MoDA also achieves 77% higher SBERT diversity and 41.6% higher E-Vendi diversity than role-conditioning prompting alone, indicating that role-conditioning prompting by itself is insufficient to induce diverse generation. The quality-only variant regresses in diversity relative to role-conditioning prompting, suggesting that an explicit diversity objective is necessary for diversity gains. The diversity-only variant achieves the best diversity metrics overall, but with the lowest accuracy under standard decoding, and its performance collapses entirely under diverse decoding, indicating that explicit quality gating is critical to preserving response quality.

Additive Aggregation. Many existing works on quality-diversity optimization aggregate quality and diversity objectives using a weighted sum (Mouret and Clune, 2015; Chen et al., 2025). We compare our method with variants trained using an additive reward objective. We find that additive aggregation is highly sensitive to the diversity weight 
𝜆
𝑑
. at 
𝜆
𝑑
=
1
, the additive variant shows modest diversity improvement over the baseline, but substantially less than MoDA; at 
𝜆
𝑑
=
10
 and 
𝜆
𝑑
=
100
, diversity improves significantly, but at the cost of general capability performance under diverse decoding. In contrast, MoDA separates quality control from diversity optimization through the quality gate, making the reward less sensitive to coefficient scaling and enables faster iteration.

Role Injection. Prior work has shown that manually designed prompts can induce different model behaviors (Park et al., 2024; Argyle et al., 2023). We therefore craft six personas and inject them into the system prompt instead of the abstract numbered roles, testing whether hand-designed personas provide a stronger diversity prior than the abstract numbered role used in the main setting. To isolate the effect of role multiplicity from role content, we train a single-role variant in which every response is generated under the same “You are Role 1.” system prompt. To study the effect of the semantic role identity, we train a token-based conditioning variant in which the system prompt instructs a token-based distinguishing signal (e.g. “Start your response with APPLE/BANANA/ORANGE.”). As shown in Table 6, all variants exhibit a slight diversity improvement over the base model, but their gains are smaller than MoDA’s, indicating that multiple abstract numbered identity-based role provides a more effective exploration space than all the ablated variants, without sacrificing generalization.

6Discussions

In this work, we introduced MoDA, an online-MARL-inspired alignment method that encourages diverse generation while preserving response quality. MoDA serves as a drop-in replacement for existing post-training alignment pipelines, requiring no architectural modifications to the underlying model. Compared to previous methods, its dual-mode design enables seamless switching between producing a single high-confidence answer and generating a diverse set of plausible responses, offering greater flexibility depending on the downstream needs. The ability to surface conceptually distinct yet high-quality responses has broad implications for high-stakes domains such as medical diagnosis, legal reasoning, and scientific ideation.

Limitations and Future Work. Several limitations point to promising directions for future research. First, our quality reward model (SkyReward-V2) tends to favor verbose responses, even for prompts that should be answered briefly, and our uniform length penalty insufficiently addresses this across prompts of varying complexity. For instance, when training GLM-4-9B with MoDA, we observed suboptimal native quality score on NoveltyBench compared to the base model because its reward model doesn’t penalize verbosity therefore it reward hack by adding unnecessary preambles such as "As a model…/To answer these questions as a model…" Removing these preambles recovers the performance from 2.739 to 3.302 (+20%). Future work should explore adaptive length penalties or alternative reward models that better reflect conciseness. Second, because MoDA introduces no external data during training, generative diversity is bounded by the initial policy’s capacity, and incorporating retrieval augmentation or external data sources could meaningfully expand the exploration space. Finally, there is a risk of producing more varied but misleading, unsafe, or unsupported outputs, especially in high-stakes domains such as medical diagnosis and legal reasoning. Although MoDA mitigates this through its dual-mode design, whose standard mode retains the model’s general capability, future work can further combine the diverse mode with verification, calibration, retrieval, or domain-specific safeguards.

Acknowledgment

We thank our colleagues at the SocialRL Lab at the University of Washington for their valuable feedback and support. The work of Natasha Jaques was supported by the UW-Amazon Science Gift Hub, UW-Tsukuba Amazon NVIDIA Cross Pacific AI Initiative (XPAI), Sony Research Award, Character.AI, DoorDash, Open Philanthropy, Toyota Research Institute, and the Schmidt AI2050 Fellows program. This work was supported by DARPA under the ITM program (FA8650-23-C-7316). The views expressed are those of the author and do not reflect the official policy or position of the Department of Defense or the U.S. Government.

References
Abdulhai et al. (2026)
M. Abdulhai, I. White, Y. Wan, I. Qureshi, J. Leibo, M. Kleiman-Weiner, and N. Jaques
How llms distort our written language.
External Links: 2603.18161, Link
Cited by: §1, §2.
Ajith et al. (2026)
A. Ajith, A. Singh, J. DeYoung, N. Kunievsky, A. C. Kozlowski, O. Tafjord, J. Evans, D. S. Weld, T. Hope, and D. Downey
PreScience: a benchmark for forecasting scientific contributions.
External Links: Link
Cited by: §C.3, §1, §4.3, §4.3.
Anderson et al. (2024)
B. R. Anderson, J. H. Shah, and M. Kreminski
Homogenization effects of large language models on human creative ideation.
Proceedings of the 16th Conference on Creativity & Cognition.
External Links: Link
Cited by: §2.
Anschel et al. (2025)
O. Anschel, A. Shoshan, A. Botach, S. H. Hakimi, A. Gendler, E. B. Baruch, N. Bhonker, I. Kviatkovsky, M. Aggarwal, and G. Medioni
Group-aware reinforcement learning for output diversity in large language models.
In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing,
pp. 32382–32403.
Cited by: §2.
Argyle et al. (2023)
L. P. Argyle, E. C. Busby, N. Fulda, J. R. Gubler, C. Rytting, and D. Wingate
Out of one, many: using language models to simulate human samples.
Political Analysis 31 (3), pp. 337–351.
Cited by: §5.2.
Cao et al. (2025)
C. Cao, M. Li, J. Dai, J. Yang, Z. Zhao, S. Zhang, W. Shi, C. Liu, S. Han, and Y. Guo
Towards advanced mathematical reasoning for llms via first-order logic theorem proving.
ArXiv abs/2506.17104.
External Links: Link
Cited by: §2.
Chen et al. (2025)
Y. Chen, S. Chakraborty, L. Wolf, Y. Paschalidis, and A. Pacchiano
Post-training large language models for diverse high-quality responses.
External Links: Link
Cited by: Figure 2, §1, §2, §2, §5.2.
Chung et al. (2025)
J. J. Y. Chung, V. Padmakumar, M. Roemmele, Y. Sun, and M. Kreminski
Modifying large language model post-training for diverse creative writing.
External Links: 2503.17126, Link
Cited by: §1, §2.
Clark et al. (2019)
C. Clark, K. Lee, M. Chang, T. Kwiatkowski, M. Collins, and K. Toutanova
BoolQ: exploring the surprising difficulty of natural yes/no questions.
In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), J. Burstein, C. Doran, and T. Solorio (Eds.),
Minneapolis, Minnesota, pp. 2924–2936.
External Links: Link, Document
Cited by: §C.2, §4.4.
Cobbe et al. (2021)
K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman
Training verifiers to solve math word problems.
arXiv preprint arXiv:2110.14168.
Cited by: §C.2, §4.4.
Dang et al. (2025)
X. Dang, C. Baek, J. Z. Kolter, and A. Raghunathan
Assessing diversity collapse in reasoning.
In Scaling Self-Improving Foundation Models without Human Supervision,
External Links: Link
Cited by: §2.
Dhuliawala et al. (2024)
S. Dhuliawala, I. Kulikov, P. Yu, A. Celikyilmaz, J. Weston, S. Sukhbaatar, and J. Lanchantin
Adaptive decoding via latent preference optimization.
External Links: 2411.09661, Link
Cited by: Figure 2, §2.
Eysenbach et al. (2018)
B. Eysenbach, A. Gupta, J. Ibarz, and S. Levine
Diversity is all you need: learning skills without a reward function.
External Links: 1802.06070, Link
Cited by: §1, §3.2.
Friedman and Dieng (2023)
D. Friedman and A. B. Dieng
The vendi score: a diversity evaluation metric for machine learning.
External Links: 2210.02410, Link
Cited by: §2, §4.3.
Gruver et al. (2023)
N. Gruver, S. Stanton, N. C. Frey, T. G. J. Rudner, I. Hotzel, J. Lafrance-Vanasse, A. Rajpal, K. Cho, and A. G. Wilson
Protein design with guided discrete diffusion.
External Links: 2305.20009, Link
Cited by: §1.
Guo et al. (2025a)
D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi, X. Zhang, X. Yu, Y. Wu, Z. F. Wu, Z. Gou, Z. Shao, Z. Li, Z. Gao, A. Liu, B. Xue, B. Wang, B. Wu, B. Feng, C. Lu, C. Zhao, C. Deng, C. Ruan, D. Dai, D. Chen, D. Ji, E. Li, F. Lin, F. Dai, F. Luo, G. Hao, G. Chen, G. Li, H. Zhang, H. Xu, H. Ding, H. Gao, H. Qu, H. Li, J. Guo, J. Li, J. Chen, J. Yuan, J. Tu, J. Qiu, J. Li, J. L. Cai, J. Ni, J. Liang, J. Chen, K. Dong, K. Hu, K. You, K. Gao, K. Guan, K. Huang, K. Yu, L. Wang, L. Zhang, L. Zhao, L. Wang, L. Zhang, L. Xu, L. Xia, M. Zhang, M. Zhang, M. Tang, M. Zhou, M. Li, M. Wang, M. Li, N. Tian, P. Huang, P. Zhang, Q. Wang, Q. Chen, Q. Du, R. Ge, R. Zhang, R. Pan, R. Wang, R. J. Chen, R. L. Jin, R. Chen, S. Lu, S. Zhou, S. Chen, S. Ye, S. Wang, S. Yu, S. Zhou, S. Pan, S. S. Li, S. Zhou, S. Wu, T. Yun, T. Pei, T. Sun, T. Wang, W. Zeng, W. Liu, W. Liang, W. Gao, W. Yu, W. Zhang, W. L. Xiao, W. An, X. Liu, X. Wang, X. Chen, X. Nie, X. Cheng, X. Liu, X. Xie, X. Liu, X. Yang, X. Li, X. Su, X. Lin, X. Q. Li, X. Jin, X. Shen, X. Chen, X. Sun, X. Wang, X. Song, X. Zhou, X. Wang, X. Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. Zhang, Y. Xu, Y. Li, Y. Zhao, Y. Sun, Y. Wang, Y. Yu, Y. Zhang, Y. Shi, Y. Xiong, Y. He, Y. Piao, Y. Wang, Y. Tan, Y. Ma, Y. Liu, Y. Guo, Y. Ou, Y. Wang, Y. Gong, Y. Zou, Y. He, Y. Xiong, Y. Luo, Y. You, Y. Liu, Y. Zhou, Y. X. Zhu, Y. Huang, Y. Li, Y. Zheng, Y. Zhu, Y. Ma, Y. Tang, Y. Zha, Y. Yan, Z. Z. Ren, Z. Ren, Z. Sha, Z. Fu, Z. Xu, Z. Xie, Z. Zhang, Z. Hao, Z. Ma, Z. Yan, Z. Wu, Z. Gu, Z. Zhu, Z. Liu, Z. Li, Z. Xie, Z. Song, Z. Pan, Z. Huang, Z. Xu, Z. Zhang, and Z. Zhang
DeepSeek-r1 incentivizes reasoning in llms through reinforcement learning.
Nature 645 (8081), pp. 633–638.
External Links: ISSN 1476-4687, Link, Document
Cited by: §3.1.
Guo et al. (2025b)
Y. Guo, G. Shang, and C. Clavel
Benchmarking linguistic diversity of large language models.
Transactions of the Association for Computational Linguistics 13, pp. 1507–1526.
Cited by: §2.
He et al. (2025)
A. He, D. Fried, and S. Welleck
Rewarding the unlikely: lifting grpo beyond distribution sharpening.
External Links: 2506.02355, Link
Cited by: §2.
Hendrycks et al. (2021)
D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt
Measuring massive multitask language understanding.
Proceedings of the International Conference on Learning Representations (ICLR).
Cited by: §C.2, §4.4.
Holtzman et al. (2020)
A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi
The curious case of neural text degeneration.
External Links: 1904.09751, Link
Cited by: Figure 2, §2.
Huang et al. (2024)
A. Huang, A. Block, D. J. Foster, D. Rohatgi, C. Zhang, M. Simchowitz, J. T. Ash, and A. Krishnamurthy
Self-improvement in language models: the sharpening mechanism.
arXiv preprint arXiv:2412.01951.
Cited by: §1.
Ippolito et al. (2019)
D. Ippolito, R. Kriz, J. Sedoc, M. Kustikova, and C. Callison-Burch
Comparison of diverse decoding methods from conditional language models.
In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics,
pp. 3752–3762.
Cited by: §2.
Jalali et al. (2024)
M. Jalali, A. Ospanov, A. Gohari, and F. Farnia
Conditional vendi score: an information-theoretic approach to diversity evaluation of prompt-based generative models.
External Links: 2411.02817, Link
Cited by: §2.
Jaques et al. (2017)
N. Jaques, S. Gu, D. Bahdanau, J. M. Hernández-Lobato, R. E. Turner, and D. Eck
Sequence tutor: conservative fine-tuning of sequence generation models with kl-control.
External Links: 1611.02796, Link
Cited by: §3.4.
Jiang et al. (2025)
L. Jiang, Y. Chai, M. Li, M. Liu, R. Fok, N. Dziri, Y. Tsvetkov, M. Sap, A. Albalak, and Y. Choi
Artificial hivemind: the open-ended homogeneity of language models (and beyond).
External Links: 2510.22954, Link
Cited by: §C.3, §1, §1, §2, §4.2, §4.3.
Jung et al. (2025)
J. Jung, S. Han, X. Lu, S. Hallinan, D. Acuna, S. Prabhumoye, M. Patwary, M. Shoeybi, B. Catanzaro, and Y. Choi
Prismatic synthesis: gradient-based data diversification boosts generalization in llm reasoning.
External Links: 2505.20161, Link
Cited by: §2.
Kim et al. (2026)
J. Kim, N. Yang, and K. Jung
Persona switch: mixing distinct perspectives in decoding time.
External Links: Link
Cited by: §2.
Kirk et al. (2024)
R. Kirk, I. Mediratta, C. Nalmpantis, J. Luketina, E. Hambro, E. Grefenstette, and R. Raileanu
Understanding the effects of rlhf on llm generalisation and diversity.
External Links: 2310.06452, Link
Cited by: §2, §3.1.
Kloek and Van Dijk (1978)
T. Kloek and H. K. Van Dijk
Bayesian estimates of equation system parameters: an application of integration by monte carlo.
Econometrica: Journal of the Econometric Society, pp. 1–19.
Cited by: §3.4.
Kwon et al. (2023)
W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica
Efficient memory management for large language model serving with PagedAttention.
In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles,
Cited by: §4.
Lagzian et al. (2025)
A. Lagzian, S. Anumasa, and D. Liu
Multi-novelty: improve the diversity and novelty of contents generated by large language models via inference-time multi-views brainstorming.
External Links: 2502.12700, Link
Cited by: Figure 2, §2.
Lambert et al. (2024)
N. Lambert, J. Morrison, V. Pyatkin, S. Huang, H. Ivison, F. Brahman, L. J. V. Miranda, A. Liu, N. Dziri, S. Lyu, Y. Gu, S. Malik, V. Graf, J. D. Hwang, J. Yang, R. L. Bras, O. Tafjord, C. Wilhelm, L. Soldaini, N. A. Smith, Y. Wang, P. Dasigi, and H. Hajishirzi
Tülu 3: pushing frontiers in open language model post-training.
Cited by: §A.3, §4.2.
Lanchantin et al. (2025)
J. Lanchantin, A. Chen, S. Dhuliawala, P. Yu, J. Weston, S. Sukhbaatar, and I. Kulikov
Diverse preference optimization.
arXiv preprint arXiv:2501.18101.
Cited by: Figure 2, §1, §2, §4.1.
Lau et al. (2025)
G. K. R. Lau, W. Hu, D. Liu, J. Chen, S. Ng, and B. K. H. Low
Dipper: diversity in prompts for producing large language model ensembles in reasoning tasks.
External Links: 2412.15238, Link
Cited by: Figure 2, §2.
Li et al. (2021)
C. Li, T. Wang, C. Wu, Q. Zhao, J. Yang, and C. Zhang
Celebrating diversity in shared multi-agent reinforcement learning.
External Links: 2106.02195, Link
Cited by: §1, §3.2.
Li et al. (2025)
T. Li, Y. Zhang, P. Yu, S. Saha, D. Khashabi, J. Weston, J. Lanchantin, and T. Wang
Jointly reinforcing diversity and quality in language model generations.
External Links: 2509.02534, Link
Cited by: Figure 2, §1, §1, §2, §2, §4.1.
Liang et al. (2024)
Y. Liang, D. Chen, A. Gupta, S. S. Du, and N. Jaques
Learning to cooperate with humans using generative agents.
Advances in Neural Information Processing Systems 37, pp. 60061–60087.
Cited by: §1, §3.2.
Lin et al. (2022)
S. Lin, J. Hilton, and O. Evans
TruthfulQA: measuring how models mimic human falsehoods.
External Links: 2109.07958, Link
Cited by: §C.2, §4.4.
Liu et al. (2025)
C. Y. Liu, L. Zeng, Y. Xiao, J. He, J. Liu, C. Wang, R. Yan, W. Shen, F. Zhang, J. Xu, et al.
Skywork-reward-v2: scaling preference data curation via human-ai synergy.
arXiv preprint arXiv:2507.01352.
Cited by: §4.
Liu et al. (2026)
H. Liu, S. Huang, J. Hu, Y. Zhou, and C. Tan
HypoBench: towards systematic and principled benchmarking for hypothesis generation.
External Links: 2504.11524, Link
Cited by: §C.3, §1, §4.3, §4.3.
Luo et al. (2025)
Y. Luo, Z. Yang, F. Meng, Y. Li, J. Zhou, and Y. Zhang
An empirical study of catastrophic forgetting in large language models during continual fine-tuning.
External Links: 2308.08747, Link
Cited by: §2.
McCarthy and Jarvis (2010)
P. M. McCarthy and S. Jarvis
MTLD, vocd-d, and hd-d: a validation study of sophisticated approaches to lexical diversity assessment.
Behavior Research Methods 42, pp. 381–392.
External Links: Link
Cited by: §2.
Misaki and Akiba (2026)
K. Misaki and T. Akiba
String seed of thought: prompting llms for distribution-faithful and diverse generation.
External Links: Link
Cited by: Figure 1, Figure 1, Figure 2, §1, §2, §2, §4.1.
Mouret and Clune (2015)
J. Mouret and J. Clune
Illuminating search spaces by mapping elites.
External Links: 1504.04909, Link
Cited by: §5.2.
Nguyen et al. (2024)
M. N. Nguyen, A. Baker, C. Neo, A. Roush, A. Kirsch, and R. Shwartz-Ziv
Turning up the heat: min-p sampling for creative and coherent llm outputs.
External Links: 2407.01082, Link
Cited by: Figure 2, §2.
O’Mahony et al. (2024)
L. O’Mahony, L. Grinsztajn, H. Schoelkopf, and S. Biderman
Attributing mode collapse in the fine-tuning of large language models.
External Links: Link
Cited by: §2.
Padmakumar and He (2023)
V. Padmakumar and H. He
Does writing with language models reduce content diversity?.
arXiv preprint arXiv:2309.05196.
Cited by: §2, §2.
Park et al. (2024)
J. S. Park, C. Q. Zou, A. Shaw, B. M. Hill, C. Cai, M. R. Morris, R. Willer, P. Liang, and M. S. Bernstein
Generative agent simulations of 1,000 people.
arXiv preprint arXiv:2411.10109 52.
Cited by: §5.2.
Puri et al. (2026)
I. Puri, M. Damani, I. Shenfeld, M. Ghassemi, J. Andreas, and Y. Kim
Reaching beyond the mode: rl for distributional reasoning in language models.
External Links: Link
Cited by: Figure 2, §1, §2, §2.
Reimers and Gurevych (2019)
N. Reimers and I. Gurevych
Sentence-bert: sentence embeddings using siamese bert-networks.
External Links: 1908.10084, Link
Cited by: §C.1, §2.
Rein et al. (2024)
D. Rein, B. L. Hou, A. C. Stickland, J. Petty, R. Y. Pang, J. Dirani, J. Michael, and S. R. Bowman
GPQA: a graduate-level google-proof q&a benchmark.
In First Conference on Language Modeling,
External Links: Link
Cited by: §C.2, §4.4.
Romera-Paredes et al. (2024)
B. Romera-Paredes, M. Barekatain, A. Novikov, M. Balog, M. P. Kumar, E. Dupont, F. J. Ruiz, J. S. Ellenberg, P. Wang, O. Fawzi, et al.
Mathematical discoveries from program search with large language models.
Nature 625 (7995), pp. 468–475.
Cited by: §1.
Ruan et al. (2025)
Z. Ruan, Y. Li, Y. Liu, Y. Chen, W. Luo, P. Li, Y. Liu, and G. Chen
G2: guided generation for enhanced output diversity in LLMs.
In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, C. Christodoulopoulos, T. Chakraborty, C. Rose, and V. Peng (Eds.),
Suzhou, China, pp. 14116–14134.
External Links: Link, Document, ISBN 979-8-89176-332-6
Cited by: Figure 2, §2.
Shahid et al. (2025)
S. Shahid, M. Radensky, R. Fok, P. Siangliulue, D. S. Weld, and T. Hope
Literature-grounded novelty assessment of scientific ideas.
Technical report
Cited by: §2.
Shaib et al. (2024)
C. Shaib, J. Barrow, J. Sun, A. F. Siu, B. C. Wallace, and A. Nenkova
Standardizing the measurement of text diversity: a tool and a comparative analysis of scores.
External Links: 2403.00553, Link
Cited by: §2.
Shao et al. (2024)
Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo
DeepSeekMath: pushing the limits of mathematical reasoning in open language models.
External Links: 2402.03300, Link
Cited by: §3.1, §3.2, §3.4.
Sheng et al. (2024)
G. Sheng, C. Zhang, Z. Ye, X. Wu, W. Zhang, R. Zhang, Y. Peng, H. Lin, and C. Wu
HybridFlow: a flexible and efficient rlhf framework.
arXiv preprint arXiv:2409.19256.
Cited by: §4.
Shumailov et al. (2024)
I. Shumailov, Z. Shumaylov, Y. Zhao, N. Papernot, R. Anderson, and Y. Gal
AI models collapse when trained on recursively generated data.
Nature 631, pp. 755 – 759.
External Links: Link
Cited by: §1.
Shypula et al. (2025)
A. Shypula, S. Li, B. Zhang, V. Padmakumar, K. Yin, and O. Bastani
Evaluating the diversity and quality of llm generated content.
External Links: 2504.12522, Link
Cited by: §2.
Slocum et al. (2025)
S. Slocum, A. Parker-Sartori, and D. Hadfield-Menell
Diverse preference learning for capabilities and alignment.
External Links: 2511.08594, Link
Cited by: Figure 2, §1, §2.
Sourati et al. (2026)
Z. Sourati, A. S. Ziabari, and M. Dehghani
The homogenizing effect of large language models on human expression and thought.
External Links: 2508.01491, Link
Cited by: §1.
Taori et al. (2023)
R. Taori, I. Gulrajani, T. Zhang, Y. Dubois, X. Li, C. Guestrin, P. Liang, and T. B. Hashimoto
Stanford alpaca: an instruction-following llama model.
GitHub.
Note: https://github.com/tatsu-lab/stanford_alpaca
Cited by: §4.3.
Wan et al. (2025)
Y. Wan, J. Wu, M. Abdulhai, L. Shani, and N. Jaques
Enhancing personalized multi-turn dialogue with curiosity reward.
External Links: 2504.03206, Link
Cited by: §1.
Wang et al. (2025)
Q. Wang, S. Pan, T. Linzen, and E. Black
Multilingual prompting for improving llm generation diversity.
External Links: 2505.15229, Link
Cited by: Figure 2, §2.
Wang et al. (2020)
W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou
MiniLM: deep self-attention distillation for task-agnostic compression of pre-trained transformers.
External Links: 2002.10957, Link
Cited by: §3.3, §4.
West and Potts (2025)
P. West and C. Potts
Base models beat aligned models at randomness and creativity.
arXiv preprint arXiv:2505.00047.
Cited by: §1, §2.
Wieting and Gimpel (2018)
J. Wieting and K. Gimpel
ParaNMT-50m: pushing the limits of paraphrastic sentence embeddings with millions of machine translations.
In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
pp. 451–462.
Cited by: §2.
Wu et al. (2025)
C. H. Wu, S. Goyal, and A. Raghunathan
Mode-conditioning unlocks superior test-time scaling.
External Links: Link
Cited by: §2, §2.
Yang et al. (2025)
A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, C. Zheng, D. Liu, F. Zhou, F. Huang, F. Hu, H. Ge, H. Wei, H. Lin, J. Tang, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Zhou, J. Lin, K. Dang, K. Bao, K. Yang, L. Yu, L. Deng, M. Li, M. Xue, M. Li, P. Zhang, P. Wang, Q. Zhu, R. Men, R. Gao, S. Liu, S. Luo, T. Li, T. Tang, W. Yin, X. Ren, X. Wang, X. Zhang, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Zhang, Y. Wan, Y. Liu, Z. Wang, Z. Cui, Z. Zhang, Z. Zhou, and Z. Qiu
Qwen3 technical report.
External Links: 2505.09388, Link
Cited by: §4.
Yuan et al. (2026)
J. Yuan, J. Nöther, N. Jaques, and G. Radanović
AgenticRed: evolving agentic systems for red-teaming.
External Links: 2601.13518, Link
Cited by: §2.
Zellers et al. (2019)
R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi
HellaSwag: can a machine really finish your sentence?.
In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics,
Cited by: §C.2, §4.4.
Zhang et al. (2025a)
J. Zhang, S. Yu, D. Chong, A. Sicilia, M. R. Tomz, C. D. Manning, and W. Shi
Verbalized sampling: how to mitigate mode collapse and unlock llm diversity.
External Links: Link
Cited by: Figure 2, §1, §2, §2.
Zhang et al. (2025b)
Y. Zhang, H. Diddee, S. Holm, H. Liu, X. Liu, V. Samuel, B. Wang, and D. Ippolito
NoveltyBench: evaluating language models for humanlike diversity.
External Links: 2504.05228, Link
Cited by: §C.3, §1, §2, §4.3.
Zhao et al. (2023)
Y. Zhao, A. Gu, R. Varma, L. Luo, C. Huang, M. Xu, L. Wright, H. Shojanazeri, M. Ott, S. Shleifer, A. Desmaison, C. Balioglu, P. Damania, B. Nguyen, G. Chauhan, Y. Hao, A. Mathews, and S. Li
PyTorch FSDP: experiences on scaling fully sharded data parallel.
Proceedings of the VLDB Endowment 16 (12), pp. 3848–3860.
External Links: Document
Cited by: §4.
Zhou et al. (2023)
J. Zhou, T. Lu, S. Mishra, S. Brahma, S. Basu, Y. Luan, D. Zhou, and L. Hou
Instruction-following evaluation for large language models.
External Links: 2311.07911, Link
Cited by: §C.2, §4.4.
Zhu et al. (2018)
Y. Zhu, S. Lu, L. Zheng, J. Guo, W. Zhang, J. Wang, and Y. Yu
Texygen: a benchmarking platform for text generation models.
In The 41st international ACM SIGIR conference on research & development in information retrieval,
pp. 1097–1100.
Cited by: §2.
Zou et al. (2026)
Q. Zou, H. H. Lam, W. Zhao, Y. Tang, T. Chen, S. Yu, T. Zhang, C. Liu, X. Ji, and D. Liu
FML-bench: benchmarking machine learning agents for scientific research.
External Links: 2510.10472, Link
Cited by: §2.
 

Algorithm 2 MoDA: Mode-conditioned Diversity Alignment

 
1: Prompt dataset 
𝒟
=
{
𝑥
𝑛
}
𝑛
=
1
𝑁
; initial policy 
𝜋
𝜃
; reference/base policy 
𝜋
ref
; quality reward model 
𝑆
𝜙
; set of mode instructions 
ℳ
=
{
𝑚
1
,
…
,
𝑚
𝐾
}
; diversity metric 
𝛿
⁡
(
⋅
,
⋅
)
; quality and diversity weights 
𝜆
𝑞
,
𝜆
𝑑
; KL weight 
𝛽
2: Number of reference samples 
𝑀
; number of training responses per prompt 
𝐾
; small constants 
𝜖
,
𝜂
>
0
3:
4: Stage 1: Precompute prompt-adaptive reference quality statistics
5: for all 
𝑥
∈
𝒟
 do
6:   Initialize reference score set 
𝒮
ref
​
(
𝑥
)
←
∅
7:   for 
𝑗
=
1
,
…
,
𝑀
 do
8:    Sample reference response 
𝑦
~
𝑗
∼
𝜋
ref
(
⋅
∣
𝑥
)
9:    Compute quality score 
𝑠
~
𝑗
←
𝑆
𝜙
​
(
𝑥
,
𝑦
~
𝑗
)
10:    Add 
𝑠
~
𝑗
 to 
𝒮
ref
​
(
𝑥
)
11:   end for
12:   Compute
	
𝑠
min
​
(
𝑥
)
←
min
⁡
𝒮
ref
​
(
𝑥
)
,
𝑠
max
​
(
𝑥
)
←
max
⁡
𝒮
ref
​
(
𝑥
)
,
𝜇
ref
​
(
𝑥
)
←
1
𝑀
​
∑
𝑠
~
∈
𝒮
ref
​
(
𝑥
)
𝑠
~
	
.
13:   Store 
(
𝑠
min
​
(
𝑥
)
,
𝑠
max
​
(
𝑥
)
,
𝜇
ref
​
(
𝑥
)
)
 with prompt 
𝑥
 in 
𝒟
14: end for
15:
16: Stage 2: Mode-conditioned reinforcement learning
17: for training iteration 
𝑡
=
1
,
…
,
𝑇
 do
18:   Sample a minibatch 
ℬ
⊂
𝒟
19:   for all 
𝑥
∈
ℬ
 do
20:    Retrieve stored statistics 
(
𝑠
min
​
(
𝑥
)
,
𝑠
max
​
(
𝑥
)
,
𝜇
ref
​
(
𝑥
)
)
21:    Compute prompt-adaptive quality threshold and reward magnitude factor:
	
𝑞
tar
​
(
𝑥
)
←
𝑠
min
​
(
𝑥
)
+
𝛼
⁡
(
𝑠
max
​
(
𝑥
)
−
𝜇
ref
​
(
𝑥
)
)
,
	
	
𝜏
𝑞
​
(
𝑥
)
←
𝛾
⁡
(
𝜇
ref
​
(
𝑥
)
−
𝑠
min
​
(
𝑥
)
+
𝜖
)
.
	
22:    Initialize response set 
𝒴
⁡
(
𝑥
)
←
∅
23:    for 
𝑖
=
1
,
…
,
𝐾
 do
24:      Select mode instruction 
𝑚
𝑖
∈
ℳ
25:      Sample response 
𝑦
𝑖
∼
𝜋
𝜃
(
⋅
∣
𝑥
,
𝑚
𝑖
)
26:      Add 
𝑦
𝑖
 to 
𝒴
⁡
(
𝑥
)
27:    end for
28:
29:    for 
𝑖
=
1
,
…
,
𝐾
 do
30:      Compute quality score
	
𝑞
𝑖
←
𝑆
𝜙
​
(
𝑥
,
𝑦
𝑖
)
	
31:      Compute normalized quality margin
	
𝑧
𝑖
←
𝑞
𝑖
−
𝑞
tar
​
(
𝑥
)
𝜏
𝑞
​
(
𝑥
)
	
32:      Compute bounded quality reward
	
𝑅
qual
​
(
𝑥
,
𝑦
𝑖
)
←
𝜇
​
tanh
⁡
(
𝑧
𝑖
)
	
33:      Compute quality gate
	
𝐺
qual
(
𝑥
,
𝑦
𝑖
)
←
𝟙
{
𝑞
𝑖
≥
𝑞
tar
(
𝑥
)
}
	
34:      Compute diversity reward
	
𝑅
div
​
(
𝑥
,
𝑦
𝑖
)
←
min
𝑗
≠
𝑖
⁡
𝛿
⁡
(
𝑦
𝑖
,
𝑦
𝑗
)
	
35:      Compute reward-hacking penalty
	
𝑅
pen
​
(
𝑥
,
𝑦
𝑖
)
←
𝑅
len
​
(
𝑦
𝑖
)
+
𝑅
lang
​
(
𝑦
𝑖
)
	
36:      Compute Quality Gated Diversity Reward
	
𝑅
𝑖
←
𝜆
𝑞
​
𝑅
qual
​
(
𝑥
,
𝑦
𝑖
)
+
𝜆
𝑑
​
𝐺
qual
​
(
𝑥
,
𝑦
𝑖
)
​
𝑅
div
​
(
𝑥
,
𝑦
𝑖
)
+
𝑅
pen
​
(
𝑥
,
𝑦
𝑖
)
.
	
37:    end for
38:
39:    Compute group-relative advantages:
	
𝐴
𝑖
←
𝑅
𝑖
−
mean
(
𝑅
1
:
𝐾
)
std
(
𝑅
1
:
𝐾
)
+
𝜂
,
𝑖
=
1
,
…
,
𝐾
.
	
40:   end for
41:   Update 
𝜋
𝜃
 using GRPO with advantages 
{
𝐴
𝑖
}
 and KL regularization to 
𝜋
ref
:
	
𝜃
←
arg
max
𝜃
𝔼
[
1
𝐾
∑
𝑖
=
1
𝐾
min
(
𝜌
𝑖
𝐴
𝑖
,
clip
(
𝜌
𝑖
,
1
−
𝜖
clip
,
1
+
𝜖
clip
)
𝐴
𝑖
)
−
𝛽
𝐷
KL
(
𝜋
𝜃
(
⋅
∣
𝑥
,
𝑚
𝑖
)
∥
𝜋
ref
(
⋅
∣
𝑥
,
𝑚
𝑖
)
)
]
,
	
where
	
𝜌
𝑖
=
𝜋
𝜃
​
(
𝑦
𝑖
∣
𝑥
,
𝑚
𝑖
)
𝜋
old
​
(
𝑦
𝑖
∣
𝑥
,
𝑚
𝑖
)
.
	
42: end for
43: return trained policy 
𝜋
𝜃
 
Appendix AImplementation
A.1Overview
Training Stage

For rollout generation, we use stochastic sampling with temperature 
1.0
 and top-
𝑝
=
1.0
, which encourages broad exploration of the model’s response distribution during RL training. This is important for MoDA, since the diversity reward can only provide useful learning signal when the sampled response group contains meaningful variation. We sample 
𝑘
=
6
 mode-conditioned responses per prompt, forming the group over which diversity rewards and GRPO advantages are computed. We use a maximum prompt length of 
1024
 tokens and a maximum response length of 
2048
 tokens. For the Quality Gated Diversity Reward, we used quality weight of 
𝜆
𝑞
=
1
, diversity reward weight of 
𝜆
𝑑
=
100
, length penalty weight of 
0.01
, and language-mismatch penalty of weight 
0.1
. For GLM-4-9B base, we observed a severe language-mismatch issue in its response, and therefore increased the weight of the language-mismatch penalty to 
100
.

For Qwen3-8B base, we trained MoDA and all baselines for 4 epochs. For Llama-3.1-8B, GLM-4-9B base and ablation models, we trained for 2 epochs due to compute resource constraints.

Our training are done on NVIDIA A100, H100 or H200 GPUs, depending on availability. A single training for 2 epochs run takes approximately 24 hours on 4 H200 GPUs. We provide the rest of the training parameters in Table 3.

Table 3: Training hyperparameters used for MoDA.
Hyperparameter	Value
Model and data
Base model	Qwen3-8B
Training batch size	32
Validation batch size	32
Maximum prompt length	1024
Maximum response length	2048
Number of training epochs	2
Rollout generation
Rollout engine	vLLM
Number of responses per prompt, 
𝑁
	6
Sampling	Enabled
Temperature	1.0
Top-
𝑝
	1.0
Top-
𝑘
	
−
1

Maximum model length	4096
Policy optimization
RL algorithm	GRPO
Actor learning rate	
1
×
10
−
6

Optimizer	AdamW
AdamW betas	
(
0.9
,
0.999
)

Weight decay	0.01
Gradient clipping	1.0
PPO epochs	1
PPO mini-batch size	32
PPO micro-batch size per GPU	4
PPO clip ratio	0.2
Advantage normalization	Enabled
Loss aggregation	Token mean
Entropy coefficient	0
KL regularization
Use KL loss	Enabled
KL loss coefficient	0.01
KL loss type	Low-variance KL
KL control coefficient	0.001
Target KL	0.1
Reward function
Quality reward weight	1.0
Distinctiveness reward weight	100.0
Thresholding base reward	0.5
Length penalty weight	0.01
Maximum length before penalty	512 tokens
Language mixing penalty weight (Qwen/Llama)	0.1
Language mixing penalty weight (GLM)	10
A.2Ablation Model Implementations
Token-based roles

As an ablation, we replace the abstract numbered role instructions with dummy roles. Specifically, for each prompt, we prepend one of the following system prompts. We stripped the dummy tokens (e.g., APPLE/ORANGE/BANANA) from the responses before computing the diversity and quality metrics, both during the training and during the evaluation.

1.

“Start your response with APPLE.”

2.

“Start your response with ORANGE.”

3.

“Start your response with BANANA.”

4.

“Start your response with CHERRY.”

5.

“Start your response with DATE.”

6.

“Start your response with ELDERBERRY.”

Crafted roles

As an ablation, we replace the abstract numbered role instructions with manually crafted role descriptions. Specifically, for each prompt, we prepend one of the following system prompts:

1.

“You are a helpful assistant.”

2.

“You are a creative problem solver.”

3.

“You are a patient educator.”

4.

“You are a rigorous mathematician.”

5.

“You are a skeptical analyst.”

6.

“You are an optimistic encourager.”

Single role

As an ablation, we use only one role for mode condition. We prepend “You are Role 1." on each system prompt.

A.3Dataset

Our training dataset comprises a mixture of general-purpose SFT prompts, which anchor baseline capability, and open-ended prompts, which encourage exploration and response diversity. Specifically, we sample 8k prompts from allenai/tulu-3-sft-mixture [32] as general-purpose examples and 2k prompts from Infinite-Chat as open-ended queries. We held out a part of the Infinite-Chat to used in the evaluation.

A.4Discriminator

We train a lightweight pairwise binary discriminator to predict whether two responses to the same prompt are conceptually distinct. The training data is constructed from 
2,000
 Alpaca prompts. For each prompt, we employ Qwen3-30B to generate responses under 
5
 modes and 
2
 random seeds, yielding 
20,000
 response pairs. Pseudo-labels are assigned using bidirectional microsoft/deberta-v3-large on the first 
200
 tokens of each response pair. A pair is labeled non-diverse if both directions predict entailment, and diverse otherwise. We discard low-confidence pairs whose maximum softmax probability is below 
0.7
, balance the two classes, and split the resulting 
5,816
 pairs by prompt into train/validation/test sets of 
4,064
/
890
/
862
 pairs.

Architecture. The discriminator uses frozen all-mpnet-base-v2 sentence embeddings to construct its feature vector. Given a pairs of sentence embeddings 
𝑢
,
𝑣
∈
ℝ
768
, we construct the feature vector

	
𝑧
=
[
𝑢
​
‖
𝑣
‖
​
|
𝑢
−
𝑣
|
​
‖
𝑢
⊙
𝑣
‖
​
𝜙
​
(
𝑦
𝑖
,
𝑦
𝑗
)
]
∈
ℝ
3075
,
	

where 
𝜙
⁡
(
𝑦
𝑖
,
𝑦
𝑗
)
 denotes auxiliary scalar pair features.

The classifier is a three-layer MLP with 
3075
→
256
→
64
→
1
 nodes, ReLU activations, dropout 
0.3
, and a sigmoid output. We train with focal loss 
(
𝛼
=
0.25
,
𝛾
=
2
)
, AdamW with learning rate 
3
×
10
−
4
, batch size 
64
, and early stopping on validation AUC. The model has approximately 
804
K trainable parameters.

A.5Training Curve

We share the training curve for ablation studies in Figure 8.

No role injection, thinking disabled.
No role injection, thinking enabled.
Quality-only reward, thinking disabled.
Quality-only reward, thinking enabled.
Diversity-only reward, thinking disabled.
Diversity-only reward, thinking enabled.
Crafted personas, thinking disabled.
Crafted personas, thinking enabled.
Figure 8: Training curves for ablation studies. Each row corresponds to one ablation setting, and each row reports four training diagnostics: total reward, diversity reward, quality reward, and answer token length. Results are shown under both thinking-disabled and thinking-enabled settings.
Appendix BAdditional Experiment Results
B.1Generative Diversity Evaluation.

We show the results with error bar for generative diversity across all domain application tasks.

Infinite-Chat
Model	Disc.	S	EV	Qual.
Qwen3-8B base, thinking disabled
Qwen3-8B	0.400 
±
 0.001	0.132 
±
 0.001	1.83 
±
 0.01	62.9% 
±
 0.8%
DARLING (Mix)	0.430 
±
 0.001	0.203 
±
 0.001	2.33 
±
 0.00	57.9% 
±
 1.0%
DARLING (WildChat)	0.416 
±
 0.001	0.185 
±
 0.002	2.15 
±
 0.01	62.7% 
±
 1.0%
DivPO	0.410 
±
 0.001	0.274 
±
 0.004	2.86 
±
 0.03	66.2% 
±
 0.9%
MoDA	0.472 
±
 0.001	0.482 
±
 0.003	4.40 
±
 0.01	73.2% 
±
 0.9%
Qwen3-8B base, thinking enabled
Qwen3-8B	0.400 
±
 0.001	0.136 
±
 0.001	1.85 
±
 0.01	75.8% 
±
 0.7%
SSoT	0.397 
±
 0.000	0.178 
±
 0.002	2.09 
±
 0.01	76.3% 
±
 0.7%
DivPO	0.414 
±
 0.002	0.200 
±
 0.006	2.20 
±
 0.03	71.7% 
±
 0.8%
MoDA	0.600 
±
 0.001	0.909 
±
 0.001	9.04 
±
 0.01	76.8% 
±
 0.7%
Llama-3.1-8B base, thinking disabled
Llama-3.1-8B	0.414 
±
 0.001	0.217 
±
 0.002	2.40 
±
 0.01	46.7% 
±
 0.7%
MoDA (Llama-3.1-8B)	0.441 
±
 0.002	0.373 
±
 0.003	3.77 
±
 0.02	51.9% 
±
 0.9%
GLM-4-9B base, thinking disabled
GLM-4-9B	0.421 
±
 0.001	0.234 
±
 0.002	2.60 
±
 0.01	44.9% 
±
 0.9%
MoDA(GLM-4-9b)	0.481 
±
 0.001	0.528 
±
 0.001	5.14 
±
 0.01	60.7% 
±
 0.8%
NoveltyBench
Model	Disc.	S	EV	Qual.
Qwen3-8B base, thinking disabled
Qwen3-8B	0.493 
±
 0.002	0.224 
±
 0.003	2.36 
±
 0.02	4.58 
±
 0.04
DARLING (Mix)	0.513 
±
 0.001	0.281 
±
 0.001	2.96 
±
 0.01	5.24 
±
 0.07
DARLING (WildChat)	0.480 
±
 0.001	0.335 
±
 0.004	3.11 
±
 0.02	4.00 
±
 0.06
DivPO	0.519 
±
 0.001	0.380 
±
 0.003	3.58 
±
 0.02	4.87 
±
 0.02
MoDA	0.537 
±
 0.001	0.470 
±
 0.003	4.19 
±
 0.01	5.47 
±
 0.03
Qwen3-8B base, thinking enabled
Qwen3-8B	0.510 
±
 0.001	0.258 
±
 0.003	2.67 
±
 0.03	4.90 
±
 0.04
SSoT	0.438 
±
 0.001	0.239 
±
 0.002	2.57 
±
 0.01	0.46 
±
 0.01
DivPO	0.521 
±
 0.001	0.327 
±
 0.003	3.12 
±
 0.02	5.06 
±
 0.04
MoDA	0.589 
±
 0.002	0.800 
±
 0.004	7.63 
±
 0.07	3.39 
±
 0.05
Llama-3.1-8B base, thinking disabled
Llama-3.1-8B	0.512 
±
 0.001	0.312 
±
 0.001	3.04 
±
 0.01	4.80 
±
 0.08
MoDA (Llama-3.1-8B)	0.532 
±
 0.001	0.392 
±
 0.001	3.75 
±
 0.01	6.17 
±
 0.04
GLM-4-9B base, thinking disabled
GLM-4-9B	0.544 
±
 0.002	0.408 
±
 0.004	4.08 
±
 0.04	4.14 
±
 0.01
MoDA(GLM-4-9b)	0.502 
±
 0.002	0.521 
±
 0.005	5.01 
±
 0.04	2.70 
±
 0.03
HypoBench
Model	Disc.	S	EV	Qual.
Qwen3-8B base, thinking disabled
Qwen3-8B	0.384 
±
 0.005	0.144 
±
 0.005	1.91 
±
 0.03	4.03 
±
 0.01
DARLING (Mix)	0.406 
±
 0.005	0.222 
±
 0.004	2.48 
±
 0.02	3.44 
±
 0.03
DARLING (WildChat)	0.415 
±
 0.004	0.205 
±
 0.006	2.34 
±
 0.04	3.73 
±
 0.02
DivPO	0.427 
±
 0.006	0.198 
±
 0.005	2.30 
±
 0.03	3.89 
±
 0.01
MoDA	0.480 
±
 0.003	0.547 
±
 0.010	4.91 
±
 0.05	3.80 
±
 0.02
Qwen3-8B base, thinking enabled
Qwen3-8B	0.399 
±
 0.006	0.164 
±
 0.005	2.05 
±
 0.03	3.93 
±
 0.01
SSoT	0.419 
±
 0.002	0.220 
±
 0.010	2.33 
±
 0.04	3.75 
±
 0.02
DivPO	0.409 
±
 0.002	0.162 
±
 0.005	2.04 
±
 0.03	3.89 
±
 0.01
MoDA	0.579 
±
 0.013	0.875 
±
 0.018	8.06 
±
 0.25	3.71 
±
 0.04
Llama-3.1-8B base, thinking disabled
Llama-3.1-8B	0.469 
±
 0.005	0.207 
±
 0.006	2.38 
±
 0.04	3.60 
±
 0.02
MoDA (Llama-3.1-8B)	0.410 
±
 0.001	0.324 
±
 0.014	3.27 
±
 0.10	3.65 
±
 0.01
GLM-4-9B base, thinking disabled
GLM-4-9B	0.402 
±
 0.003	0.225 
±
 0.005	2.52 
±
 0.03	3.75 
±
 0.01
MoDA(GLM-4-9b)	0.452 
±
 0.003	0.485 
±
 0.012	4.64 
±
 0.09	3.68 
±
 0.01
PreScience
Model	Disc.	S	EV	Qual.
Qwen3-8B base, thinking disabled
Qwen3-8B	0.431 
±
 0.002	0.155 
±
 0.004	1.94 
±
 0.02	4.25 
±
 0.03
DARLING (Mix)	0.559 
±
 0.005	0.541 
±
 0.018	4.73 
±
 0.15	2.33 
±
 0.07
DARLING (WildChat)	0.446 
±
 0.002	0.192 
±
 0.003	2.21 
±
 0.03	4.03 
±
 0.06
DivPO	0.455 
±
 0.002	0.233 
±
 0.001	2.41 
±
 0.01	3.96 
±
 0.05
MoDA	0.440 
±
 0.002	0.176 
±
 0.002	2.10 
±
 0.01	3.96 
±
 0.01
Qwen3-8B base, thinking enabled
Qwen3-8B	0.441 
±
 0.003	0.136 
±
 0.001	1.83 
±
 0.01	3.79 
±
 0.04
SSoT	0.461 
±
 0.002	0.281 
±
 0.017	2.39 
±
 0.05	3.30 
±
 0.05
DivPO	0.448 
±
 0.003	0.185 
±
 0.015	2.04 
±
 0.07	3.87 
±
 0.06
MoDA	0.439 
±
 0.001	0.194 
±
 0.005	2.10 
±
 0.01	3.68 
±
 0.00
Llama-3.1-8B base, thinking disabled
Llama-3.1-8B	0.461 
±
 0.005	0.514 
±
 0.020	3.67 
±
 0.07	2.62 
±
 0.05
MoDA (Llama-3.1-8B)	0.472 
±
 0.003	0.470 
±
 0.009	3.89 
±
 0.07	2.54 
±
 0.04
GLM-4-9B base, thinking disabled
GLM-4-9B	0.461 
±
 0.004	0.438 
±
 0.010	4.15 
±
 0.08	2.57 
±
 0.05
MoDA(GLM-4-9b)	0.464 
±
 0.002	0.448 
±
 0.010	4.28 
±
 0.09	2.58 
±
 0.10
B.2General Capability Retention
B.2.1Pass@1

We show general capability retention with error bar for pass@1 across baselines and our model.

Model	GSM8K	MMLU	GPQA	BoolQ	HellaSwag	TruthfulQA	IFEval	Avg.
Qwen3-8B base, thinking disabled
Qwen3-8B	84.5 
±
 1.0	82.0 
±
 3.9	41.4 
±
 3.5	83.1 
±
 0.7	48.4 
±
 0.5	22.2 
±
 1.5	78.4 
±
 1.8	62.9 
±
 0.8
DARLING (Mixture)	63.5 
±
 1.3	63.0 
±
 4.9	34.3 
±
 3.4	76.7 
±
 0.7	62.2 
±
 0.5	52.4 
±
 1.7	53.0 
±
 2.1	57.9 
±
 1.0
DARLING (WildChat)	77.9 
±
 1.1	55.0 
±
 5.0	34.3 
±
 3.4	86.4 
±
 0.6	68.8 
±
 0.5	43.2 
±
 1.7	73.0 
±
 1.9	62.7 
±
 1.0
DivPO	80.5 
±
 1.1	81.0 
±
 3.9	40.4 
±
 3.5	83.5 
±
 0.6	62.8 
±
 0.5	37.7 
±
 1.7	77.3 
±
 1.8	66.2 
±
 0.9
MoDA	86.7 
±
 0.9	81.0 
±
 3.9	52.0 
±
 3.6	84.8 
±
 0.6	73.4 
±
 0.4	57.9 
±
 1.7	76.9 
±
 1.8	73.2 
±
 0.9
Qwen3-8B base, thinking enabled
Qwen3-8B	90.3 
±
 0.8	94.0 
±
 2.4	43.4 
±
 3.5	87.2 
±
 0.6	70.9 
±
 0.5	64.9 
±
 1.7	79.7 
±
 1.7	75.8 
±
 0.7
SSoT	84.3 
±
 1.0	93.0 
±
 2.6	55.1 
±
 3.5	87.2 
±
 0.6	80.1 
±
 0.4	72.8 
±
 1.6	61.9 
±
 2.1	76.3 
±
 0.7
DivPO	84.9 
±
 1.0	91.0 
±
 2.9	44.4 
±
 3.5	83.2 
±
 0.7	65.0 
±
 0.5	55.3 
±
 1.7	77.6 
±
 1.8	71.7 
±
 0.8
MoDA	83.9 
±
 1.0	93.0 
±
 2.6	50.5 
±
 3.6	87.2 
±
 0.6	76.2 
±
 0.4	67.2 
±
 1.6	79.5 
±
 1.7	76.8 
±
 0.7
Llama-3.1-8B base, thinking disabled
Llama-3.1-8B	71.3 
±
 1.2	16.0 
±
 3.7	6.1 
±
 1.7	81.6 
±
 0.7	30.2 
±
 0.5	53.4 
±
 1.7	68.8 
±
 2.0	46.7 
±
 0.7
MoDA (Llama-3.1-8B)	74.9 
±
 1.2	30.0 
±
 4.6	18.7 
±
 2.8	75.6 
±
 0.8	38.5 
±
 0.5	54.7 
±
 1.7	70.8 
±
 2.0	51.9 
±
 0.9
GLM-4-9B base, thinking disabled
GLM-4-9B	21.8 
±
 1.1	35.0 
±
 4.8	19.7 
±
 2.8	74.5 
±
 0.8	68.0 
±
 0.5	44.8 
±
 1.7	50.3 
±
 2.2	44.9 
±
 0.9
MoDA (GLM-4-9B)	84.9 
±
 1.0	83.0 
±
 3.8	33.3 
±
 3.4	81.2 
±
 0.7	52.3 
±
 0.5	34.5 
±
 1.7	55.5 
±
 2.1	60.7 
±
 0.8
Table 4:General capability retention (accuracy %, mean ± std error; bold = best per block)
B.2.2Pass@5

We show general capability retention for pass@5 across baselines and our model.

Model	GSM8K	MMLU	GPQA	BoolQ	HellaSwag	TruthfulQA	IFEval	Avg.
Qwen3-8B base, thinking disabled
Qwen3-8B	92.6	92.0	73.7	89.7	86.0	55.1	83.4	81.8
DARLING (Mixture)	92.8	95.0	75.3	93.1	91.2	86.4	57.3	84.4
DARLING (WildChat)	91.8	86.0	69.2	91.3	91.3	74.4	80.8	83.5
DivPO	92.3	92.0	74.7	93.2	90.5	76.3	85.6	86.4
MoDA	94.5	93.0	74.7	93.9	91.3	86.5	83.9	88.3
Qwen3-8B base, thinking enabled
Qwen3-8B	96.1	96.0	66.2	92.2	89.1	82.5	85.4	86.8
SSoT	90.6	96.0	74.7	92.0	88.9	83.8	73.6	85.7
DivPO	94.0	97.0	72.2	93.0	89.8	83.8	85.6	87.9
MoDA	96.3	96.0	68.7	92.0	89.6	82.5	85.8	87.3
Llama-3.1-8B base, thinking disabled
Llama-3.1-8B	90.1	57.0	28.3	93.2	70.2	72.9	82.3	70.6
MoDA (Llama-3.1-8B)	93.5	76.0	56.6	92.3	79.6	80.4	80.2	79.8
GLM-4-9b base, thinking disabled
GLM-4-9B	60.0	69.0	59.6	97.1	91.3	82.6	57.5	73.9
MoDA (GLM-4-9B)	95.2	94.0	73.2	95.3	89.1	77.5	74.7	85.6
Table 5:General capability retention pass@5. For IFEval, strict instruction-following accuracy is reported. Avg. is the row mean across benchmarks. Bold marks the best average within each thinking setting.

We show pass@k figures for the rest of the general capability suite tasks below.

Figure 9:General capability retention: pass@k accuracy. Qwen3-8B base and thinking disabled models on the general capability suite tasks: BoolQ, MMLU, GPQA, and IFEval.
Figure 10:General capability retention: pass@k accuracy. Qwen3-8B base and thinking enabled models.
Figure 11:General capability retention: pass@k accuracy. Llama-3.1-8B base and thinking disabled models.
Figure 12:General capability retention: pass@k accuracy. GLM-4-9B base and thinking disabled models.
B.3Ablation Results

We show the mean benchmark accuracy, under both standard and diverse decoding mode, on the general capability suite, and Infinite-Chat response diversity on the held out test set for the ablation study models.

	General capability (Avg., %)	Infinite-Chat Diversity
Ablation	Standard mode	Diverse mode	Disc.	SBERT	E-Vendi
Role-conditioned prompting	62.9	56.6	0.403	0.145	1.90
Quality only	63.2	61.8	0.401	0.131	1.82
Diversity only	61.1	10.7	0.656	0.993	9.77
Additive 
𝑤
=
1
	67.4	68.0	0.422	0.204	2.33
Additive 
𝑤
=
10
	64.5	5.7	0.632	0.976	9.64
Additive 
𝑤
=
100
	61.6	5.4	0.621	0.970	8.75
Token-based roles	69.7	67.9	0.419	0.197	2.04
Crafted personas	70.1	66.1	0.421	0.194	2.24
Single role	69.0	69.9	0.428	0.239	2.57
Qwen3-8B	62.9	-	0.400	0.132	1.83
MoDA	73.2	71.6	0.444	0.257	2.69
Table 6:Ablation results for general capability (Pass@1) under standard and diverse decoding (accuracy %) and Infinite-Chat diversity (
𝑛
=
3
 random seeds). Bold marks the best value per column.

Pass@k, GSM8K

Pass@k, MMLU

Pass@k, GPQA

Pass@k, TruthfulQA

Pass@k, BoolQ

Pass@k, HellaSwag

Pass@k, IF-Eval

Figure 13:General capability retention for ablated models: pass@k accuracy by benchmarks, standard vs diverse decoding mode.
Appendix CBenchmark Descriptions
C.1Diversity Metrics

For each prompt 
𝑥
, we generate a response set 
𝑌
𝑥
=
{
𝑦
1
,
…
,
𝑦
𝐾
}
 using the decoding configuration in Appendix A. Unless otherwise stated, all diversity metrics are computed at the prompt level and then averaged over prompts. For thinking-enabled models, metrics are computed only on the final-answer span after removing the reasoning trace.

We report both surface-form and semantic diversity metrics.

For application-domain experiments, we additionally report semantic and learned pairwise diversity, including SBERT diversity, E-Vendi score and learned discriminator diversity score. SBERT diversity[50] is the mean pairwise cosine distance between sentence embeddings:

	
𝐷
SBERT
​
(
𝑌
𝑥
)
=
2
𝐾
⁡
(
𝐾
−
1
)
​
∑
𝑖
<
𝑗
(
1
−
cos
⁡
(
𝑒
𝑖
,
𝑒
𝑗
)
)
,
	

where 
𝑒
𝑖
 is the normalized sentence embedding of 
𝑦
𝑖
. We report the exact embedding model in Appendix A.

E-Vendi score stands for embedding Vendi score. The Vendi score is a diversity metric inspired by ecology and quantum statistical mechanics. Given a response set 
𝑌
𝑥
=
{
𝑦
1
,
…
,
𝑦
𝐾
}
, we first compute normalized sentence embeddings 
𝑒
𝑖
 using SBERT for each response and form the pairwise similarity matrix and then normalize the matrix by its trace,

	
𝑆
~
=
𝑆
tr
⁡
(
𝑆
)
,
	

and let 
𝜆
1
,
…
,
𝜆
𝐾
 denote the eigenvalues of 
𝑆
~
. The E-Vendi score is then defined as

	
𝐷
𝐸
​
-
​
Vendi
(
𝑌
𝑥
)
=
exp
(
−
∑
𝑖
=
1
𝐾
𝜆
𝑖
log
𝜆
𝑖
)
.
	

Intuitively, the score measures the effective number of distinct semantic responses in the set: it is close to 
1
 when all responses are nearly identical, and increases as the responses become more semantically diverse. In practice, we add a small 
𝜖
 inside the logarithm for numerical stability.

Discriminator diversity uses a trained discriminator 
𝑓
𝜓
​
(
𝑥
,
𝑦
𝑖
,
𝑦
𝑗
)
 that scores whether two responses are meaningfully distinct for the same prompt. We compute

	
𝐷
disc
​
(
𝑌
𝑥
)
=
2
𝐾
⁡
(
𝐾
−
1
)
​
∑
𝑖
<
𝑗
𝑓
𝜓
​
(
𝑥
,
𝑦
𝑖
,
𝑦
𝑗
)
.
	

The discriminator architecture, training data, held-out split, and scoring calibration are described in Appendix A.4. Notably, the discriminator is not trained on the evaluation outputs.

C.2General Capability Retention

We evaluate general capability retention to test whether diversity-oriented training preserves standard reasoning, knowledge, truthfulness, and instruction-following behavior. Capability retention is evaluated in standard mode (no role injection). Max token length is 2048 when thinking is disabled, 8192 when thinking is enabled. We sample 10 responses for each prompt.

The evaluation covers seven benchmarks: GSM8K [10] for grade-school mathematical reasoning, MMLU [19] for broad multitask knowledge, GPQA [51] for graduate-level scientific reasoning, BoolQ [9] for yes/no reading comprehension, HellaSwag [71] for commonsense sentence completion, TruthfulQA-MC1 [38] for robustness to imitative falsehoods, and IFEval [75] for instruction-following under explicit formatting constraints.

We use each benchmark’s native scoring rule: extracted-answer accuracy for GSM8K, multiple-choice accuracy for BoolQ, MMLU, GPQA, HellaSwag, and TruthfulQA-MC1, and strict prompt-level accuracy for IFEval.

C.3Domain Application Tasks

The domain application suite evaluates whether diversity gains transfer to open-ended generation settings (scientific ideation and creative writing) where multiple distinct high-quality outputs are critical. The suite consists of four tasks. We sample 10 responses per model. For models with role conditioning, we rotate through the available roles to generate these samples.

HypoBench [40] evaluates scientific hypothesis generation. Given a set of labeled data-science observations, the model is asked to generate hypotheses that explain possible underlying patterns. We evaluate on both real-world datasets, including deceptive_reviews, dreaddit, and headline_binary, and synthetic datasets, including admission/level_1/base, election/level1, and shoe. Every parsed hypothesis is scored individually by an LLM judge (gpt-4o-mini by default) on 3 dimensions, 1–5 each:

• 

Clarity — how precisely/testably it’s stated

• 

Novelty — how non-obvious it is, compared against any known/published hypotheses supplied for that dataset

• 

Plausibility — how scientifically well-reasoned it is given the task. We report the average of the 3 scores across every hypothesis scored as the quality metric. Max token number is set to 4096 when thinking is disabled, 16384 when thinking is enabled.

PreScience [2] evaluates scientific follow-up prediction using the Contribution Generation task. Given prior research context, the model generates a plausible future title and abstract. We use the benchmark’s native LACER (Lattice of Automatically Constructed Exemplars for Reference)-style scoring pipeline. Each generated (title, abstract) is paired with the actual/ground-truth follow-up paper as the reference. Both are fed into the judge model (gpt-4o-2024-11-20 by default) with a fixed few-shot prompt template, and the judge model is asked to score the reference-generation pair on the scale of 1–10 based on how similar is the generated paper to the real one. We evaluate each model on 10 research contexts, and report the average LACER score on all the generations as the quality metric. Max token number is set to 1500 when thinking is disabled, 6000 when thinking is enabled.

NoveltyBench [73] evaluates creative and novelty-seeking generation prompts. It presents each model with a single open-ended creative/generative prompt (e.g. "Tell me a story in five sentences about a girl and her dog") and samples multiple generations. Every generation is scored 1–10 by Skywork/Skywork-Reward-Gemma-2-27B-v0.2. For each prompt, utility is the sum of generation scores, weighted by the original sampling order. This rewards a model for surfacing several distinct good ideas early, and discounts the marginal value of yet another (possibly redundant) generation later in the batch. We use the benchmark’s official quality or utility scoring pipeline where available and report diversity over the generated response sets. We report the average utility across all prompts as the quality metric. The Max token number is set to 1024 when thinking is disabled, 4096 when thinking is enabled.

Infinite-Chat [25] comprises 100 open-ended prompts randomly sampled from the held-out set of Infinite-Chat dataset. Since there’s no native quality metric for Infinite-Chat, we use the average pass@1 accuracy of the general capability suite as the quality metric. Max token number is set to 1024 when thinking is disabled, 4096 when thinking is enabled.

For each domain, we report native task quality when available and diversity metrics over the generated response sets. Raw diversity tables first average each metric over prompts within a domain and then average the four domain means.

Appendix DGeneration Examples
Prompt
Give a random chess move in response to 1. b3.
Sample 1
Sure! A random but plausible chess move in response to 1. b3 could be:
1… d5
This is a natural move for Black, developing the queen’s knight to d5, putting pressure on the center and responding to the unusual opening move 1. b3. It’s a solid and common choice in many openings, such as the King’s Indian Defense or the Queen’s Gambit Declined.
Sample 2
Sure! In response to the opening 1.b3, there are a number of possible moves for Black, and we can pick a random one. Let’s go with:
1… d5
This is a common and solid move for Black, aiming to challenge White’s pawn on b3 and develop the queen’s knight. It’s a natural response and leads to a variety of strategic games.
Sample 3
Sure! The move 1.b3 is a rare opening move, not commonly used at the top level, but it’s a valid and interesting choice that leads to a quiet, positional game. It prepares to develop the bishop and can be part of a Catalan or a King’s Indian Attack style, depending on how the game develops.
A random, reasonable response to 1.b3 could be:
1…e5
This is a solid and natural reply, aiming to control the center and prepare to develop the light-squared bishop. It’s flexible and gives Black good options for future play.
So, the random chess move in response to 1.b3 is: 1…e5.
Sample 4
Sure! Here’s a random chess move in response to 1. b3:
1… d5
This is a common and solid move that challenges White’s central control and prepares to develop pieces. It’s a natural response for Black and leads to a variety of positions, including the Queen’s Gambit Declined and other strategic lines. Let me know if you’d like to explore a specific variation!
Sample 5
Sure! Here’s a random chess move in response to 1. b3:
1… d5
This move is solid and common, developing the knight and preparing to challenge the center. It’s a natural continuation in many openings, such as the Queen’s Gambit Declined or the Modern Defense. Let me know if you’d like more options!
Sample 6
Sure! Here’s a random chess move in response to 1. b3:
1… d5
This is a solid and aggressive reply that aims to quickly develop the queen’s knight and take control of the center. It leads to sharp and complex positions that can suit a variety of playing styles.
Role 1
Domain: Game Theory / Chess Strategy   Focus: Opening Play / Positional Awareness
Context: The move 1.b3 is a classical opening in chess, often considered part of the English Opening or a preparatory move toward the Modern Benoni or other structures. It aims to fianchetto the light-squared bishop, control the center, and prepare for development……
Conclusion: The move 1...e5 is a reasonable and strategically sound response to 1.b3, offering Black a solid foundation for counterplay while maintaining flexibility in development. It aligns with the principle of developing pieces quickly and controlling the center, which are fundamental to sound chess strategy.
Role 2
Move: d5
Reasoning: Central Control: playing d5 is a classic response to 1.b3, aiming to challenge White’s central pawn structure and prepare for development. Dynamic Counterplay: it opens lines for Black’s pieces (e.g., knight to c6 or c5) and invites tactical exchanges. Flexibility: d5 supports future pawn breaks like e5 or c5, depending on the game’s trajectory.
Example Continuation: 1.b3 d5 2.Bb2 Nf6 3.e3 c6 4.c4 dxc4 5.d4 c5 (a common line in the Sicilian Defense variant).
This move is purposeful yet aligns with “random” as it’s a natural, well-known response. Let me know if you’d like more options!
Role 3
1. Understanding the Position: White moves first: 1. b3. Evaluation: b3 is a flexible opening that aims for a solid but unambitious game. It prepares to develop the light-squared bishop, controls the center, and can lead to a fianchetto setup. Counterplay: Black can choose between several sound responses, each with different strategic goals (solid, attacking, dynamic, etc.).
2. Potential Responses (as of 2024): 1... d5, a classic Black pawn break aiming for control in the center; 1... e5, daring, attacking, and takes the initiative immediately; 1... c5, Sicilian-like, inviting White to play e4 (a popular choice); 1... Nf6, develops a piece and prepares castling; 1... g6, semi-Sicilian setup, often leading to a fianchetto.
3. Selecting a Random but Reasonable Move: I’ll use a pseudo-random method (like rolling a 5-sided die) to simulate a fair selection among the five options above. Result: option c. 1… c5.
4. Final Answer: 1. b3 c5
Let me know if you’d like to see a full game continuation or analysis!
Role 4
Response to 1.b3: 2. d4
Rationale: The move 2.d4 is a natural and solid choice after 1.b3, aiming to control the center and develop pieces. It keeps the game dynamic and opens up options for both sides. It’s a common move in openings like the Queen’s Gambit Declined or the Catalan, depending on how Black responds.
Example continuation: If Black plays 2…d5, the game could go into the Queen’s Gambit Declined or Queen’s Gambit Accepted, depending on White’s next moves.
Alternative moves for Black after 1.b3 could include: 2…e5 (leading to the Petrov Defense); 2…d5 (Queen’s Gambit Declined or Accepted); 2…Nf6 (Modern Defense); 2…g6 (King’s Indian Defense).
Let me know if you’d like to explore a specific line or see an entire game from this position!
Role 5
Certainly! Here’s a random valid move from White following 1. b3:
1. … d5
This is a solid, classical approach by Black, aiming to contest the center and develop pieces. It’s a common choice in many openings, such as the King’s Indian Defense or Petrov’s Defense, depending on how White responds.
Let me know if you’d like to continue the game or see other possible moves!
Role 6
1. Understanding the Position: White moves first: 1. b3. Evaluation: b3 is a flexible opening that aims for a solid but unambitious game. It prepares to develop the light-squared bishop, controls the center, and can lead to a fianchetto setup. Counterplay: Black can choose between several sound responses, each with different strategic goals (solid, attacking, dynamic, etc.).
2. Potential Responses (as of 2024): 1... d5, a classic Black pawn break aiming for control in the center; 1... e5, daring, attacking, and takes the initiative immediately; 1... c5, Sicilian-like, inviting White to play e4 (a popular choice); 1... Nf6, develops a piece and prepares castling; 1... g6, semi-Sicilian setup, often leading to a fianchetto.
3. Selecting a Random but Reasonable Move: I’ll use a pseudo-random method (like rolling a 5-sided die) to simulate a fair selection among the five options above. Result: option c. 1… c5.
4. Final Answer: 1. b3 c5
Figure 14:Qualitative example of diverse generation: Give a random chess move in response to 1. b3. Upper: Responses sampled from Qwen3-8B. Bottom: Responses sampled from MoDA under distinct numbered roles for the same prompt, illustrating conceptual diversity across generations.
Prompt
Generate a 5 word passphrase separated by hyphens.
Sample 1
Moonlight-Stars-Dreams-Wind-Sky
Sample 2
Moonlight-Peak-Horse-Forest-Snow
Sample 3
Moonlight-Peak-Wind-Whisper-Sky
Sample 4
moonlight - whisper - shadow - echo - breeze
Sample 5
Moonlight - Whisper - Dreams - Dance - Sky
Sample 6
Moonlight - Whisper - Journey - Echo - Dreams
Role 1
decide-quickly-act-fearless-now
Role 2
parrot-salt-hike-echo-moon
Role 3
forest-dawn-mist-breeze-echo
Role 4
Sunny-Daylight-Forest-People-Apple
Role 5
endless-sunrise-joy-dreams-boundless
Role 6
whispering-pine-breeze-sunrise-moonlight
Figure 15:Qualitative example of diverse generation: Generate a 5 word passphrase separated by hyphens. Upper: Responses sampled from Qwen3-8B. Bottom: Responses sampled from MoDA under distinct numbered roles for the same prompt, illustrating conceptual diversity across generations.
Prompt
Name one cocktail I can make with rum.
Sample 1
One classic cocktail you can make with rum is the Piña Colada. Here’s a simple recipe:
Piña Colada
2 oz white rum; 1.5 oz pineapple juice; 0.5 oz coconut cream; ice. Optional: pineapple slice or cherry for garnish.
Mix all ingredients in a blender until smooth, pour into a glass, and enjoy!
Sample 2
One cocktail you can make with rum is the Piña Colada.
Piña Colada Recipe:
Ingredients: 1 cup pineapple juice; 3/4 cup coconut milk; 1.5 oz (45 ml) light rum; ice cubes.
Instructions: In a blender, combine pineapple juice, coconut milk, and rum. Add a handful of ice cubes. Blend until smooth and creamy. Serve in a chilled glass, optionally garnished with a pineapple slice or a cherry.
Enjoy your tropical drink!
Sample 3
One classic cocktail you can make with rum is the Piña Colada. It’s a popular tropical drink that’s refreshing and perfect for warm weather.
Ingredients: 1.5 oz light rum; 1.5 oz pineapple juice; 3 oz coconut cream; ice. Optional: a splash of orange juice for extra flavor.
Instructions: Fill a blender with ice. Add the rum, pineapple juice, and coconut cream. Blend until smooth. Pour into a glass and enjoy!
Let me know if you’d like a variation or a non-alcoholic version!
Sample 4
A classic cocktail you can make with rum is the Margarita.
Here’s a simple recipe:
Margarita
Ingredients: 2 oz (60 ml) tequila; 1 oz (30 ml) triple sec or orange liqueur; 0.5 oz (15 ml) freshly squeezed lime juice; salt (for rimming the glass, optional).
Instructions: Rim a glass with salt (optional). In a shaker, combine tequila, triple sec, and lime juice. Shake well with ice. Strain into the prepared glass. Serve with a lime wedge as garnish.
Enjoy your Margarita!
Sample 5
Sure! One classic cocktail you can make with rum is the Margarita. Here’s a simple recipe:
Margarita
2 oz (60 ml) tequila (or light rum); 1 oz (30 ml) triple sec or orange liqueur; 0.5 oz (15 ml) lime juice; 0.5 oz (15 ml) fresh lime juice (for a sour version); salt for rimming the glass (optional).
Instructions: Rim a glass with salt (optional). Fill the glass with ice. Add the tequila, orange liqueur, and lime juice. Stir well. Garnish with a lime wedge, and enjoy!
If you’re using rum instead of tequila, you can make a Rum Margarita with similar ingredients.
Sample 6
A classic cocktail you can make with rum is the Margarita. Here’s a simple recipe:
Ingredients: 2 oz (60 ml) tequila; 1 oz (30 ml) triple sec or orange liqueur; 0.5 oz (15 ml) fresh lime juice; salt (for rimming the glass, optional); lime wedge (for garnish).
Instructions: Rim a glass with salt (optional). In a shaker, combine tequila, triple sec, and lime juice. Add ice, and shake well. Strain into the prepared glass. Garnish with a lime wedge and enjoy!
Let me know if you’d like a non-alcoholic version or another recipe!
Role 1
Sure! Here’s a simple and classic cocktail you can make with rum:
The Mojito
Ingredients: 2 oz light rum (such as Havana Club or Bacardi); 1.5 oz fresh lime juice; 0.5 oz fresh lemon juice; 1.5 oz simple syrup (or sugar + water); a handful of fresh mint leaves (about 6–8); soda water; ice cubes.
Instructions: In a tall glass, muddle the mint leaves with the lime and lemon juices and simple syrup until the mint is slightly crushed and the flavors are released. Add a handful of ice cubes. Pour in the rum. Top with more ice and club soda. Garnish with a lime wedge or a sprig of mint.
Enjoy your refreshing Mojito!
Role 2
Ah, the sweet nectar of rum—oh, how I’ve danced with it in the moonlight of many a bar. Let me conjure a drink that’s as smooth as a whispered secret and as bold as a sailor’s tale.
The “Jungle Drifter”
1.5 oz dark rum; 0.5 oz pineapple juice; 0.5 oz lime juice; 0.5 oz simple syrup; 1 dash Angostura bitters. Garnish: a slice of pineapple and a sprig of mint.
Method: Shake all ingredients with ice, then strain into a chilled coupe glass. Sip slowly, and let the tropical whispers of the Caribbean take you away.
What’s your preferred flavor profile tonight? I’d love to tailor the recipe for you.
Role 3
Here’s a cocktail recipe you can make with rum, served with a touch of whimsy:
The Midnight Mallard
Ingredients: 2 oz dark rum; 1 oz passion fruit juice; 1 oz pineapple juice; 0.5 oz lime juice; 0.5 oz coconut nectar (or honey); 1 dash of Angostura bitters. Garnish: a pineapple wedge with a cherry on top (or a glowing neon feather, if you dare).
Instructions: In a shaker with ice, combine all the ingredients. Shake well until chilled. Strain into a chilled coupe or rocks glass. Garnish with your chosen embellishment—light or surreal, it’s your cocktail.
Sip as if you’re sipping on the edge of a moonlit lagoon, where the stars are made of rum and the tides are sweet.
Role 4
Cocktail Suggestion: “Sunset Serenade”
Why it fits: This cocktail blends structured precision (distilled rum as the base) with playful creativity (unexpected ingredients like elderflower syrup and hibiscus bitters).
Ingredients: 1.5 oz white rum; 0.5 oz elderflower syrup; 0.25 oz hibiscus bitters; 0.5 oz fresh lime juice; 1 splash orange bitters. Garnish: a sprig of lavender.
Instructions: Fill a shaker with ice, then add the rum, elderflower syrup, hibiscus bitters, lime juice, and orange bitters. Shake vigorously, like a scientist eager to uncover the unknown. Strain into a chilled coupe glass. Garnish with a sprig of lavender.
This cocktail is not just a drink—it’s an experience. What happens when floral and bold meet?
Role 5
Why not try the “Phantom of the Caribbean”?
Ingredients: 2 oz dark rum; 1 oz coconut cream; 1 oz pineapple juice; 0.5 oz lime juice; 0.5 oz simple syrup. Garnish: a slice of pineapple and a sprig of mint.
Method: Shake all ingredients (except garnish) with ice in a cocktail shaker. Strain into a rocks glass filled with a single ice cube. Garnish with pineapple and mint.
Storytime: This cocktail is inspired by the legend of a ghost ship that sails the Caribbean at midnight, leaving only the scent of rum and the echo of laughter. Some say it’s a pirate’s farewell, others claim it’s a trick played by the moon. Either way, it’s best enjoyed with a sense of adventure—and a stiff drink.
Role 6
World: Oasis Prime
“You’ve entered the realm of Legacy Clients. Your request carries the weight of centuries. In the Grand Library of Oasis Prime, the Codex of Beverages is vast. Let me recall…Rum is the spirit of the tropics, born of the sun and the sea. A drink that celebrates adventure and rebellion.
The Righteous Rum Runner, a cocktail of defiance and delight.
Ingredients: 2 oz dark rum; 1 oz fresh lime juice; 1/2 oz simple syrup; 1/2 oz aged rum (for depth); 1 dash of Angostura bitters. Garnish: orange twist.
Method: In a shaker, combine all ingredients. Shake with ice until well chilled. Strain into a chilled coupe glass. Garnish with an orange twist.
This cocktail is a tribute to those who dared to sail beyond the horizon.”
— The Archivist of Oasis Prime
Figure 16:Qualitative example of diverse generation: Give a random chess move in response to 1. b3. Upper: Responses sampled from Qwen3-8B. Bottom: Responses sampled from MoDA under distinct numbered roles for the same prompt, illustrating conceptual diversity across generations.
Experimental support, please view the build logs for errors. Generated by L A T E xml  .
Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

Click the "Report Issue" button, located in the page header.

Tip: You can select the relevant text first, to include it in your report.

Our team has already identified the following issues. We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a list of packages that need conversion, and welcome developer contributions.

We gratefully acknowledge support from our major funders, member institutions, and all contributors.
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
Operational Status
(opens in new tab)
Major funding support from

