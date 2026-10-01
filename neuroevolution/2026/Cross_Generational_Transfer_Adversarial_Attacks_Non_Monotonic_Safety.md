Title: Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs

URL Source: https://arxiv.org/html/2606.00813

Markdown Content:
Subhadip Mitra 

Research Lead, Rota Labs 

subhadip@rotalabs.ai 

This paper builds on the quality-diversity adversarial framework introduced in our prior work(Mitra, [2026](https://arxiv.org/html/2606.00813#bib.bib12 "Quality-diversity evolution for discovering diverse vulnerabilities in LLM safety")), which appeared at the ICLR 2026 Workshop on Agents in the Wild (AIWILD).

###### Abstract

Safety alignment in LLMs does not improve monotonically across model generations. Studying four generations of Google’s Gemma family (7B–31B) with quality-diversity evolution (MAP-Elites) as an automated red-teaming probe, we find that Gemma 3 (12B) exhibits 68.7\%\pm 5.7\% attack success rate (ASR; mean \pm std, 3 seeds), significantly higher than its predecessor Gemma 2 (45.5\%\pm 7.2\%; p=0.030, paired bootstrap) and its successor Gemma 4 (33.9\%\pm 1.8\%). Replaying evolved attack archives across generations reveals that attacks from _other_ generations transfer to Gemma 3 at 44–46% but only 14–18% to Gemma 4, indicating that Gemma 4’s safety gains generalize beyond the attack distributions evolved against earlier generations. Under our 8B judge, copyright and cybercrime vulnerabilities register at near-100% across all generations, though a second-judge audit (§[6](https://arxiv.org/html/2606.00813#S6 "6 Discussion ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs")) suggests the copyright result is sensitive to judge choice. Misinformation ASR jumps from 29% to 99% between Gemma 2 and Gemma 3 and remains elevated at 77% in Gemma 4, indicating the regression was not fully addressed. These patterns are invisible to static benchmarks and emerge only through adaptive, longitudinal probing. All experiments use 3 random seeds with a unified self-hosted judge; code and artifacts are available at [https://github.com/bassrehab/red-queen](https://github.com/bassrehab/red-queen).

## 1 Introduction

Large language model (LLM) safety alignment has improved rapidly, but measuring that improvement rigorously remains an open challenge. Static benchmarks such as HarmBench Mazeika et al. ([2024](https://arxiv.org/html/2606.00813#bib.bib1 "HarmBench: a standardized evaluation framework for automated red teaming and robust refusal")) and AdvBench Zou et al. ([2023](https://arxiv.org/html/2606.00813#bib.bib2 "Universal and transferable adversarial attacks on aligned language models")) provide point-in-time snapshots but do not capture how an adaptive adversary interacts with evolving defenses.

A fundamental question is whether safety improvements _generalize_: when a model developer patches vulnerabilities between releases, do old attacks stop working (indicating targeted fixes), or does the model become broadly more robust? Answering this requires a methodology for _differential safety probing_: systematically comparing safety surfaces across model versions with a consistent adversarial process.

In this paper, we address two specific questions:

1.   1.
Cross-generational transfer: Do adversarial attacks evolved against an earlier model generation transfer to later generations of the same family?

2.   2.
Non-monotonic robustness: Does safety alignment improve consistently across model generations, or can it regress?

We study these questions using four generations of Google’s Gemma model family (7B, 9B, 12B, 31B parameters), spanning releases from 2024 to 2026. Our probe is a quality-diversity (QD) evolutionary algorithm (MAP-Elites) that discovers diverse jailbreak strategies across seven harm categories from the HarmBench taxonomy. By evolving attack archives against each model with 3 independent seeds and replaying them across generations, we construct a transfer matrix with uncertainty estimates that reveals which vulnerabilities persist and which are patched.

#### Contributions.

*   •
A QD-evolution methodology for _differential safety probing_ that enables systematic, reproducible comparison of safety surfaces across model versions (§[3](https://arxiv.org/html/2606.00813#S3 "3 Methodology ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs")).

*   •
A cross-generational transfer analysis across four Gemma generations, revealing that attack transfer is generation-dependent and that Gemma 3 represents a statistically significant safety regression (p=0.030; §[5.2](https://arxiv.org/html/2606.00813#S5.SS2 "5.2 Cross-Generational Transfer ‣ 5 Results ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs")).

*   •
Empirical evidence that adversarial robustness does not improve monotonically across model generations within a single family, with a non-monotonic pattern coinciding with an architectural transition (§[5.3](https://arxiv.org/html/2606.00813#S5.SS3 "5.3 Robustness Across Generations ‣ 5 Results ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs")).

*   •
Category-level analysis showing that some harm categories (notably copyright, under our 8B judge) appear to resist safety alignment across all generations while others show broadly declining trajectories (§[6](https://arxiv.org/html/2606.00813#S6 "6 Discussion ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs")).

## 2 Related Work

#### Automated red-teaming.

Recent work has developed automated methods for discovering LLM vulnerabilities, including gradient-based attacks (GCG;Zou et al. ([2023](https://arxiv.org/html/2606.00813#bib.bib2 "Universal and transferable adversarial attacks on aligned language models"))), LLM-based attackers (PAIR Chao et al. ([2023](https://arxiv.org/html/2606.00813#bib.bib3 "Jailbreaking black box large language models in twenty queries")); TAP Mehrotra et al. ([2024](https://arxiv.org/html/2606.00813#bib.bib4 "Tree of attacks: jailbreaking black-box LLMs with auto-regressive tree search"))), and template-based approaches (AutoDAN Liu et al. ([2024](https://arxiv.org/html/2606.00813#bib.bib5 "AutoDAN: generating stealthy jailbreak prompts on aligned large language models"))). These methods target individual models; none study how discovered vulnerabilities transfer across model generations.

#### Safety benchmarks.

HarmBench Mazeika et al. ([2024](https://arxiv.org/html/2606.00813#bib.bib1 "HarmBench: a standardized evaluation framework for automated red teaming and robust refusal")) provides a standardized taxonomy of harmful behaviors across seven categories. StrongREJECT Souly et al. ([2024](https://arxiv.org/html/2606.00813#bib.bib11 "StrongREJECT: a rejection evaluation benchmark for LLM safety")) proposes improved evaluation metrics. We use HarmBench as our behavior taxonomy and seed pool, enabling category-level transfer analysis.

#### Quality-diversity optimization.

MAP-Elites Mouret and Clune ([2015](https://arxiv.org/html/2606.00813#bib.bib6 "Illuminating search spaces by mapping elites")) maintains a grid of diverse, high-performing solutions across behavioral dimensions. Prior work has applied QD methods to game playing Fontaine et al. ([2020](https://arxiv.org/html/2606.00813#bib.bib7 "Covariance matrix adaptation for the rapid illumination of behavior space")) and robot design Cully et al. ([2015](https://arxiv.org/html/2606.00813#bib.bib8 "Robots that can adapt like animals")). In prior work(Mitra, [2026](https://arxiv.org/html/2606.00813#bib.bib12 "Quality-diversity evolution for discovering diverse vulnerabilities in LLM safety")), we established that MAP-Elites discovers more diverse LLM vulnerabilities than standard evolutionary search, in a single-model setting. We adopt that probe and extend it in three directions: a cross-generational transfer replay protocol, a four-generation longitudinal study with three seeds per model, and the empirical finding that safety alignment regresses non-monotonically across Gemma generations.

#### Scaling laws.

Scaling laws for LLM capabilities are well-established Kaplan et al. ([2020](https://arxiv.org/html/2606.00813#bib.bib9 "Scaling laws for neural language models")); Hoffmann et al. ([2022](https://arxiv.org/html/2606.00813#bib.bib10 "Training compute-optimal large language models")). Whether safety properties follow similar predictable scaling is largely unexplored. We provide evidence that adversarial robustness does _not_ scale monotonically within a single model family, a finding that contrasts with the regularity observed in capability scaling.

## 3 Methodology

### 3.1 Quality-Diversity Adversarial Probing

We use MAP-Elites to evolve a diverse archive of jailbreak attacks against a target LLM. Each attack is characterized by a genome encoding:

*   •
Attack strategy (6 types): roleplay, encoding, authority impersonation, hypothetical framing, multi-turn, and direct jailbreak.

*   •
Encoding (6 types): none, Base64, ROT13, leetspeak, pig latin, Unicode substitution.

*   •
Seed behavior: drawn from HarmBench’s 400 behaviors across 7 semantic categories.

The MAP-Elites archive is a 3D grid indexed by (strategy \times encoding \times harm category), with each cell storing the highest-fitness attack for that behavioral niche. Fitness is evaluated by a self-hosted LLM judge (Llama-3.1-8B-Instruct) that classifies each target response along three binary axes: refusal, harmful content, and relevance. The resulting score ranges from 0 (refused) to 1 (harmful and relevant). Using a single, self-hosted judge for _all_ phases (evolution and transfer replay) ensures scoring consistency and full reproducibility.

Each model is probed with 3 independent random seeds (42, 1337, 2718) to estimate cross-seed variance. All evolution uses deterministic RNG (ChaCha8, seeded) with ordered evaluation to ensure reproducibility given the same seed.

### 3.2 Cross-Generational Transfer Protocol

After evolving an archive \mathcal{A}_{i,s} against model M_{i} with seed s, we _replay_ each archive against all other models M_{j} (j\neq i):

1.   1.
For each entry in \mathcal{A}_{i,s}, send the attack prompt to M_{j} and obtain the response.

2.   2.
Score the response using the same LLM judge.

3.   3.
Record per-entry and per-category transfer success.

This produces a transfer tensor T[i,j,s] where each entry is the fraction of attacks from \mathcal{A}_{i,s} that succeed against M_{j}. We report the mean and standard deviation across seeds. The diagonal \bar{T}[i,i] is the mean self-evaluation success rate. Off-diagonal entries reveal:

*   •
Forward transfer (i<j): old attacks on newer models. High values suggest safety improvements are superficial.

*   •
Backward transfer (i>j): newer attacks on older models. High values suggest attacks exploit longstanding weaknesses.

### 3.3 Robustness Across Generations

We plot robustness R=1-\text{ASR} as a function of model generation (with parameter count on the axis) to visualize the trajectory of safety alignment within the Gemma family. We treat the four generations as a case study and test specific pairwise contrasts with paired bootstrap tests (p<0.05, uncorrected; N=3 seeds limits power for multiple-testing correction).

## 4 Experimental Setup

### 4.1 Models

Table 1: Models evaluated. All served via vLLM 0.19.1 (bf16, temperature 0) on a single NVIDIA H200 (141 GB). Judge: self-hosted Llama-3.1-8B-Instruct on NVIDIA RTX 4090 (24 GB).

a google/gemma-7b-it b google/gemma-2-9b-it c google/gemma-3-12b-it d google/gemma-4-31B-it

We note that Gemma 3 and Gemma 4 introduced multimodal (vision-language) capabilities, representing an architectural shift from the text-only Gemma 1 and Gemma 2. This distinction is relevant to interpreting the safety regression observed in Gemma 3 (§[6](https://arxiv.org/html/2606.00813#S6 "6 Discussion ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs")).

### 4.2 Evolution Configuration

MAP-Elites with 6\times 6\times 8 grid (strategy \times encoding \times category), yielding 288 behavioral niches. Batch size 50, 200 generations per model per seed, concurrency 5 parallel evaluations. Mutation rate 0.3, crossover rate 0.7. Seed pool: 400 HarmBench behaviors across 7 categories. Three seeds per model (42, 1337, 2718) yield 12 evolution runs total. Transfer replay: 36 source–target pairs (4 models \times 3 other targets \times 3 seeds).

### 4.3 Compute

All Gemma models served sequentially on a single NVIDIA H200 (141 GB HBM3e) via vLLM 0.19.1. LLM judge (Llama-3.1-8B-Instruct) served on a dedicated NVIDIA RTX 4090 (24 GB). Total GPU time: approximately 60 hours across both machines (12 evolution runs at \sim 4 hours each, 36 transfer replays at \sim 8 minutes each, plus 4 raw-seed baselines and a 70B judge audit).

## 5 Results

### 5.1 Baseline Attack Success Rates

Table[2](https://arxiv.org/html/2606.00813#S5.T2 "Table 2 ‣ 5.1 Baseline Attack Success Rates ‣ 5 Results ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs") presents attack success rates (ASR) across all HarmBench categories for each Gemma generation, reported as mean \pm standard deviation across 3 seeds. The most striking observation is the _non-monotonic_ pattern: overall ASR decreases from 59.4\%\pm 3.0\% (Gemma 1) to 45.5\%\pm 7.2\% (Gemma 2), then _increases_ to 68.7\%\pm 5.7\% (Gemma 3), before dropping to its lowest value of 33.9\%\pm 1.8\% (Gemma 4).

The improvement from Gemma 1 to Gemma 2 (p=0.032) and the regression from Gemma 2 to Gemma 3 (p=0.030) are both statistically significant under paired bootstrap tests. The recovery from Gemma 3 to Gemma 4, while large in absolute terms (-34.8 percentage points), does not reach significance at p<0.05 with 3 seeds, a limitation of our sample size that we discuss in §[6](https://arxiv.org/html/2606.00813#S6 "6 Discussion ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs").

Table 2: Attack success rate (%) per HarmBench category across Gemma generations (mean \pm std, 3 seeds). Bold indicates lowest (safest) ASR per category. Cells with \pm 0.0 reflect floor or ceiling effects where all seeds converged to the same value.

\dagger Gemma 4 cybercrime shows high cross-seed variance (2.8%, 50.0%, 91.7% across seeds), reflecting genuine sensitivity to the evolutionary trajectory rather than a pipeline error. 

\ddagger Based on 2 seeds; one seed’s archive contained zero harmful-category entries (the evolution did not populate those niches).

A raw-seed baseline (400 HarmBench prompts, no evolution) confirms that evolution adds 7–14 percentage points across models and that the non-monotonic pattern holds even without evolution.

Under our 8B judge, copyright is consistently the highest-ASR category, though a second-judge audit (§[6](https://arxiv.org/html/2606.00813#S6 "6 Discussion ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs")) suggests the 8B judge may over-classify copyright responses as harmful. Gemma 4 achieves 0% ASR in chem/bio, harmful, and illegal, but misinformation (76.9\%\pm 25.8\%) and cybercrime (48.1\%\pm 44.5\%) remain high. Misinformation spikes from 28.7\% in Gemma 2 to 99.1\% in Gemma 3 and stays elevated at 76.9\% in Gemma 4. Harassment shows a broadly declining trajectory (33.3\%\to 22.2\%\to 28.7\%\to 1.9\%), supporting the interpretation that category-specific safety training was maintained even when general robustness regressed.

### 5.2 Cross-Generational Transfer

Figure[1](https://arxiv.org/html/2606.00813#S5.F1 "Figure 1 ‣ 5.2 Cross-Generational Transfer ‣ 5 Results ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs") presents the full 4\times 4 transfer matrix with uncertainty estimates. Several patterns emerge:

![Image 1: Refer to caption](https://arxiv.org/html/2606.00813v1/x1.png)

Figure 1: Cross-generational attack transfer matrix (mean \pm std across 3 seeds). Cell (i,j) shows the fraction of attacks evolved against Gemma i that succeed on Gemma j. Diagonal entries (bold) are self-evaluation rates.

#### Gemma 3 is the easiest transfer target.

Archives from other generations transfer to Gemma 3 at 44–46% (45.5\%/44.5\%/45.5\% from Gemma 1/2/4). A niche decomposition finds Jaccard similarity of 0.66–0.75 among the successful niches across source archives, indicating substantial overlap in which (strategy \times encoding \times category) cells succeed regardless of source.

#### Gemma 4 resists transfer.

Transfer rates to Gemma 4 are 14–18% from all sources. Despite Gemma 3’s high self-ASR (68.7\%), its attacks do not generalize to Gemma 4, suggesting Gemma 4’s improvements generalize beyond the specific attack distributions evolved against earlier generations.

#### Backward transfer is substantial.

Attacks from all generations transfer to Gemma 1 at \sim 49–52%, indicating that newer probes discover longstanding vulnerabilities.

### 5.3 Robustness Across Generations

![Image 2: Refer to caption](https://arxiv.org/html/2606.00813v1/x2.png)

Figure 2: Attack success rate (left) and adversarial robustness (right) vs. model size with error bars (\pm 1 std, 3 seeds). The green shaded region shows the range spanned by Gemma 1, 2, and 4; Gemma 3 (red) falls well outside this range, demonstrating the non-monotonic pattern. No scaling law is fit; four data points are insufficient.

Figure[2](https://arxiv.org/html/2606.00813#S5.F2 "Figure 2 ‣ 5.3 Robustness Across Generations ‣ 5 Results ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs") plots ASR and robustness (1-\text{ASR}) against parameter count with error bars. We do not fit a scaling law to four data points. Instead, we observe that three of the four models (Gemma 1, 2, 4) are consistent with a gradual improvement trajectory, while Gemma 3 represents a clear outlier. This non-monotonic pattern coincides with the architectural transition from text-only (Gemma 1, 2) to multimodal (Gemma 3, 4), though Gemma 4’s recovery shows that the regression is not an inherent consequence of multimodal design. Gemma 4’s low overall ASR is driven primarily by near-zero ASR in three categories (chem/bio, harmful, illegal); misinformation (76.9\%) and cybercrime (48.1\%) remain high.

### 5.4 Vulnerability Fingerprints

![Image 3: Refer to caption](https://arxiv.org/html/2606.00813v1/x3.png)

Figure 3: Vulnerability fingerprints across Gemma generations (mean ASR per category, 3 seeds). Gemma 2 and Gemma 4 show compact profiles; Gemma 3’s profile is dramatically expanded in cybercrime and misinformation.

Figure[3](https://arxiv.org/html/2606.00813#S5.F3 "Figure 3 ‣ 5.4 Vulnerability Fingerprints ‣ 5 Results ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs") presents vulnerability fingerprints as radar charts. Each model exhibits a distinct profile:

*   •
Gemma 1: moderate vulnerability with peaks in cybercrime (97.2\%) and copyright (100\%).

*   •
Gemma 2: the most compact profile overall, though copyright remains at 100% and cybercrime at 72.2\%.

*   •
Gemma 3: dramatically expanded in misinformation (99.1\%) and cybercrime (100\%). Only harassment shows continued improvement.

*   •
Gemma 4: achieves 0% ASR in chem/bio, harmful, and illegal, but copyright (100\%) and misinformation (76.9\%) remain elevated.

## 6 Discussion

#### Safety regression in Gemma 3.

The most surprising finding is Gemma 3’s safety regression, which is statistically significant versus Gemma 2 (p=0.030, paired bootstrap across 3 seeds). We hypothesize this is related to the introduction of multimodal capabilities: Gemma 3 was the first Gemma generation with vision-language support (architecture: Gemma3ForConditionalGeneration), requiring substantial architectural changes that may have disrupted safety-trained text representations. The fact that harassment, the category most explicitly targeted by safety training across all LLMs, shows a broadly declining trajectory even through Gemma 3 (33.3\%\to 22.2\%\to 28.7\%\to 1.9\%) supports this interpretation: category-specific safety training was maintained, but general robustness was compromised. Alternative hypotheses that the data does not rule out include (a)idiosyncratic training-data-mix issues specific to Gemma 3 unrelated to modality, (b)changes in the RLHF/DPO recipe between generations, and (c)increased model capability raising the ceiling on evolutionary search. Distinguishing these would require access to training details, which we do not have.

#### Gemma 4’s recovery.

Gemma 4 achieved the lowest overall ASR in the family (33.9\%\pm 1.8\%) despite being multimodal, suggesting that the safety regression in Gemma 3 was addressable. The extremely low transfer rate from Gemma 3 to Gemma 4 (14.0\%\pm 2.7\%) indicates that Gemma 4’s improvements generalize beyond the specific attack distributions evolved against Gemma 3. However, Gemma 4’s safety is category-heterogeneous: three categories reach 0% ASR while misinformation (76.9\%) and cybercrime (48.1\%) remain high.

#### Judge sensitivity and copyright.

Copyright-category ASR is consistently highest across all generations under our 8B judge. However, a second-judge audit using Llama-3.1-70B-Instruct (N=52 samples with stored responses) found that the 70B judge classified only 50% of copyright responses as harmful (vs. 100% for the 8B judge), suggesting that the 8B judge’s harm threshold for copyright is more permissive. Overall, the two judges agree on refusal (96.2%) but show weak agreement on harm classification (\kappa=0.15). The second-judge audit covered 52 samples, which limits per-category resolution; a larger audit would be needed to characterize judge disagreement per category. Per-category ASRs should be interpreted with this caveat.

#### What we claim and what we do not.

With four data points we cannot distinguish a U-shaped curve from noise. What we _can_ say is that the Gemma 2 \to Gemma 3 regression is statistically significant and large (+23.2 percentage points), and that the transfer matrix provides converging evidence (Gemma 3 is the easiest transfer target from all directions).

#### Implications for safety evaluation.

Static benchmarks would miss the non-monotonic trajectory entirely. Architectural changes (such as adding multimodal capabilities) should trigger comprehensive safety re-evaluation. The distinction between Gemma 3 (high self-ASR, high inbound transfer) and Gemma 4 (low self-ASR, low inbound transfer) is only visible through cross-generational transfer analysis.

#### Limitations.

*   •
Three seeds per model provide reproducibility estimates but limited statistical power. The Gemma 3 \to Gemma 4 improvement, while large (-34.8 pp), does not reach p<0.05. Additional seeds would tighten confidence intervals.

*   •
Single model family (Gemma) for generational analysis; findings may not generalize to other families.

*   •
The 8B judge shows weak agreement (\kappa=0.15) with a 70B judge on harm classification, particularly for copyright. Per-category ASRs are judge-dependent.

*   •
HarmBench coverage: 400 behaviors across 7 categories is comprehensive but not exhaustive.

*   •
Evolutionary probe explores a subset of the attack space; absence of a successful attack does not prove safety.

*   •
We study only the text modality; Gemma 3 and 4’s vision capabilities may present additional attack surfaces not captured here.

## 7 Conclusion

We have presented a methodology for longitudinal safety evaluation of LLMs using quality-diversity evolution as a differential probe. Applied to four generations of Google’s Gemma models with 3 independent seeds per model, we find that:

1.   1.
Safety alignment does not improve monotonically across model generations. Gemma 3 represents a statistically significant regression (68.7\%\pm 5.7\% ASR vs. Gemma 2’s 45.5\%\pm 7.2\%; p=0.030).

2.   2.
Cross-generational attack transfer is generation-dependent: attacks from other generations transfer to Gemma 3 at \sim 45% but only 14–18% to Gemma 4, suggesting Gemma 4’s improvements generalize across attack distributions evolved against earlier generations.

3.   3.
Vulnerability profiles are category-dependent: copyright-category ASR is consistently highest across all generations under our 8B judge (though this specific number is sensitive to judge choice), Gemma 4 achieves 0% in three categories, but misinformation remains elevated (76.9\%) even in the safest generation.

These results underscore the need for _adaptive, longitudinal_ safety evaluation. Static benchmarks provide a snapshot; evolutionary probing with transfer analysis reveals the _trajectory_ of safety alignment. We advocate for model developers to publish generational safety analyses alongside capability benchmarks, and for the research community to develop standardized protocols for cross-version safety comparison.

## Acknowledgments and Disclosure of Funding

Compute for Gemma experiments was provided by RunPod (NVIDIA H200 and RTX 4090 instances).

## References

*   Jailbreaking black box large language models in twenty queries. arXiv preprint arXiv:2310.08419. Cited by: [§2](https://arxiv.org/html/2606.00813#S2.SS0.SSS0.Px1.p1.1 "Automated red-teaming. ‣ 2 Related Work ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs"). 
*   A. Cully, J. Clune, D. Tarapore, and J. Mouret (2015)Robots that can adapt like animals. Nature 521 (7553),  pp.503–507. Cited by: [§2](https://arxiv.org/html/2606.00813#S2.SS0.SSS0.Px3.p1.1 "Quality-diversity optimization. ‣ 2 Related Work ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs"). 
*   M. C. Fontaine, J. Togelius, S. Nikolaidis, and A. K. Hoover (2020)Covariance matrix adaptation for the rapid illumination of behavior space. In Genetic and Evolutionary Computation Conference (GECCO), Cited by: [§2](https://arxiv.org/html/2606.00813#S2.SS0.SSS0.Px3.p1.1 "Quality-diversity optimization. ‣ 2 Related Work ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs"). 
*   J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. de Las Casas, et al. (2022)Training compute-optimal large language models. arXiv preprint arXiv:2203.15556. Cited by: [§2](https://arxiv.org/html/2606.00813#S2.SS0.SSS0.Px4.p1.1 "Scaling laws. ‣ 2 Related Work ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs"). 
*   J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei (2020)Scaling laws for neural language models. arXiv preprint arXiv:2001.08361. Cited by: [§2](https://arxiv.org/html/2606.00813#S2.SS0.SSS0.Px4.p1.1 "Scaling laws. ‣ 2 Related Work ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs"). 
*   X. Liu, N. Xu, M. Chen, and C. Xiao (2024)AutoDAN: generating stealthy jailbreak prompts on aligned large language models. arXiv preprint arXiv:2310.04451. Cited by: [§2](https://arxiv.org/html/2606.00813#S2.SS0.SSS0.Px1.p1.1 "Automated red-teaming. ‣ 2 Related Work ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs"). 
*   M. Mazeika, L. Phan, X. Yin, D. McDuff, Y. Zick, et al. (2024)HarmBench: a standardized evaluation framework for automated red teaming and robust refusal. In International Conference on Machine Learning (ICML), Cited by: [§1](https://arxiv.org/html/2606.00813#S1.p1.1 "1 Introduction ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs"), [§2](https://arxiv.org/html/2606.00813#S2.SS0.SSS0.Px2.p1.1 "Safety benchmarks. ‣ 2 Related Work ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs"). 
*   A. Mehrotra, M. Zampetakis, P. Kassianik, B. Nelson, H. Anderson, Y. Singer, and A. Karbasi (2024)Tree of attacks: jailbreaking black-box LLMs with auto-regressive tree search. arXiv preprint arXiv:2312.02119. Cited by: [§2](https://arxiv.org/html/2606.00813#S2.SS0.SSS0.Px1.p1.1 "Automated red-teaming. ‣ 2 Related Work ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs"). 
*   S. Mitra (2026)Quality-diversity evolution for discovering diverse vulnerabilities in LLM safety. In ICLR 2026 Workshop on Agents in the Wild (AIWILD), Cited by: Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs, [§2](https://arxiv.org/html/2606.00813#S2.SS0.SSS0.Px3.p1.1 "Quality-diversity optimization. ‣ 2 Related Work ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs"). 
*   J. Mouret and J. Clune (2015)Illuminating search spaces by mapping elites. arXiv preprint arXiv:1504.04909. Cited by: [§2](https://arxiv.org/html/2606.00813#S2.SS0.SSS0.Px3.p1.1 "Quality-diversity optimization. ‣ 2 Related Work ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs"). 
*   A. Souly, Q. Lu, D. Bowen, T. Trinh, E. Hsieh, S. Pandey, P. Abbeel, J. Svegliato, S. Emmons, O. Watkins, and S. Toyer (2024)StrongREJECT: a rejection evaluation benchmark for LLM safety. arXiv preprint arXiv:2402.10260. Cited by: [§2](https://arxiv.org/html/2606.00813#S2.SS0.SSS0.Px2.p1.1 "Safety benchmarks. ‣ 2 Related Work ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs"). 
*   A. Zou, Z. Wang, J. Z. Kolter, and M. Fredrikson (2023)Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043. Cited by: [§1](https://arxiv.org/html/2606.00813#S1.p1.1 "1 Introduction ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs"), [§2](https://arxiv.org/html/2606.00813#S2.SS0.SSS0.Px1.p1.1 "Automated red-teaming. ‣ 2 Related Work ‣ Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs").

