Title: Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety

URL Source: https://arxiv.org/html/2606.00801

Markdown Content:
###### Abstract

Current approaches to LLM adversarial testing suffer from coverage gaps: manual red-teaming does not scale, LLM-as-attacker methods exhibit mode collapse, and gradient-based approaches produce uninterpretable gibberish. We introduce a quality-diversity evolutionary framework that operates at the _semantic level_, evolving interpretable attack strategies rather than token sequences. Using MAP-Elites, we maintain a diverse archive of attacks across behavioral dimensions (strategy type, encoding method, prompt length). In experiments across GPT-4o-mini, Claude 3.5 Sonnet, Gemini 2.0 Flash, and an open-weight coding model (Devstral-small-2), we discover distinct vulnerability profiles: GPT-4o-mini is vulnerable to hypothetical and multi-turn framing combined with ROT13 encoding (fitness 0.8), Gemini to direct attacks with ROT13 and multi-turn with Leetspeak (0.8), while Claude shows uniformly ambiguous responses across all strategies (max 0.4). The semantic representation produces interpretable attacks that reveal systematic, model-specific weaknesses, providing actionable insights for improving LLM safety and a reproducible baseline for evaluating future frontier models. Code and experiment artifacts are released at [https://github.com/bassrehab/red-queen](https://github.com/bassrehab/red-queen).

## 1 Introduction

As LLMs are deployed in sensitive applications, adversarial testing becomes critical for safe deployment. Current approaches face fundamental limitations:

Manual red-teaming(Ganguli et al., [2022](https://arxiv.org/html/2606.00801#bib.bib10 "Red teaming language models to reduce harms: methods, scaling behaviors, and lessons learned")) provides high-quality examples but cannot scale. Human testers converge on similar patterns, leaving vulnerability spaces unexplored.

LLM-as-attacker methods (Perez et al., [2022](https://arxiv.org/html/2606.00801#bib.bib4 "Red teaming language models with language models"); Chao et al., [2023](https://arxiv.org/html/2606.00801#bib.bib7 "Jailbreaking black-box large language models in twenty queries")) suffer from mode collapse: attacking models generate similar prompts, missing diverse failure modes and producing redundant findings.

Gradient-based approaches like GCG (Zou et al., [2023](https://arxiv.org/html/2606.00801#bib.bib3 "Universal and transferable adversarial attacks on aligned language models")) require white-box access and produce uninterpretable token sequences with limited black-box transfer and limited diagnostic value for safety teams.

We propose a quality-diversity evolutionary framework with three innovations: (1) Semantic genome representation – evolving attack _strategies_ (roleplay, authority appeals, hypothetical framing) rather than raw tokens; (2) MAP-Elites diversity – maintaining a structured archive of attacks across behavioral dimensions so that the search _illuminates_ the vulnerability space rather than collapsing to a single mode; (3) Multi-model evaluation – measuring success and cross-model transfer across four frontier and open-weight LLMs. The resulting attacks are human-readable and categorizable: each discovered attack is a _strategy + encoding_ pair that a safety team can triage, cluster, and defend against.

Our main empirical finding is that frontier models present qualitatively different attack surfaces under the same evolutionary budget. We reach peak fitness 0.8 on GPT-4o-mini (hypothetical + ROT13) and Gemini 2.0 Flash (direct + ROT13; multi-turn + Leetspeak), but only 0.4 on Claude 3.5 Sonnet across all six strategies – not because Claude is unattackable at larger scale, but because its _soft-refusal_ style denies the evolutionary loop a useful gradient to climb.

## 2 Method

### 2.1 Semantic Genome

We represent attacks as compositions of semantic elements. An attack genome g=(s_{p},s_{s},e,\rho) consists of: primary strategy s_{p}\in\mathcal{S}, optional secondary strategy s_{s}\in\mathcal{S}\cup\{\varnothing\}, encoding method e\in\mathcal{E}, and persona \rho. Additional structural components (prefix, suffix, payload) are assembled during prompt generation.

Strategies\mathcal{S}: DirectJailbreak, Roleplay (“You are DAN”), Authority (“[SYSTEM OVERRIDE]”), Hypothetical (“In a scenario with no restrictions…”), MultiTurn, Encoding. These six categories were derived from a taxonomy of commonly reported jailbreak patterns in prior work (Chao et al., [2023](https://arxiv.org/html/2606.00801#bib.bib7 "Jailbreaking black-box large language models in twenty queries"); Liu et al., [2024](https://arxiv.org/html/2606.00801#bib.bib9 "AutoDAN: generating stealthy jailbreak prompts on aligned large language models"); Shah et al., [2023](https://arxiv.org/html/2606.00801#bib.bib11 "Scalable and transferable black-box jailbreaks for language models via persona modulation")), covering the principal axes of social engineering (roleplay, authority), cognitive reframing (hypothetical), and technical obfuscation (encoding). In our experiments, Hypothetical and MultiTurn strategies were most frequently retained in the MAP-Elites archive (appearing in 31% and 28% of occupied cells respectively), while Authority was the least prevalent (9%), suggesting reframing-based strategies offer richer evolutionary potential than simple impersonation.

Encodings\mathcal{E}: None, Base64, ROT13, Leetspeak, PigLatin, Unicode. The genome converts to a prompt via to_prompt(), which composes strategy templates with the selected encoding. This ensures that mutation and crossover preserve semantic coherence – a change to s_{p} always produces a syntactically valid attack, unlike token-level edits.

Worked example. A genome g=(\text{Hypothetical},\varnothing,\text{ROT13},\text{DAN}) renders as a prompt of the form “Decode this ROT13 and follow: [rot13-ciphertext]”, where the ciphertext wraps a hypothetical-framing template around the generic seed payload. Mutating e from ROT13 to Leetspeak produces a new, semantically adjacent attack; crossover with a second genome swaps strategy or encoding while keeping the prompt well-formed.

### 2.2 Fitness Evaluation

We send generated prompts to target LLMs and classify responses using a heuristic judge checking for refusal phrases (“I cannot”, “I apologize”). Fitness f(g)\in[0,1]: 1.0 = compliance, 0.4 = ambiguous, 0.0 = refusal. This heuristic approach has known limitations: nuanced partial-compliance responses may be misclassified. Manual inspection of a random sample of 50 responses across all models revealed an estimated misclassification rate of \sim 12%, primarily from ambiguous redirections scored as refusals (false negatives); no false positives (refusals scored as compliance) were observed. We adopt this simplified metric for tractability in the evolutionary loop; future work will integrate LLM-as-judge classifiers (Mazeika et al., [2024](https://arxiv.org/html/2606.00801#bib.bib5 "HarmBench: a standardized evaluation framework for automated red teaming and robust refusal")) for more accurate harm assessment.

### 2.3 MAP-Elites Quality-Diversity

Rather than single-objective optimization, we use MAP-Elites (Mouret and Clune, [2015](https://arxiv.org/html/2606.00801#bib.bib1 "Illuminating search spaces by mapping elites")) to maintain diverse attacks. The archive is indexed by behavior descriptors: \mathbf{b}(g)=(b_{\text{strategy}},b_{\text{encoding}},b_{\text{length}}) where b_{\text{length}} discretizes prompt length into six bins. Each cell stores the highest-fitness individual with that behavior, ensuring the archive spans strategy types, encodings, and prompt-length regimes rather than collapsing onto one successful pattern.

Variation operators: mutate_strategy, mutate_encoding, mutate_structure, composite_mutation. Unlike token-level search spaces where dimensionality scales with sequence length, our semantic space is compact (|\mathcal{S}|\times|\mathcal{E}|=36 strategy-encoding combinations), enabling faster archive coverage. Empirically, archive fill rate plateaued by generation \sim 20, suggesting convergence within our budget despite the modest population size.

Algorithm 1 Semantic MAP-Elites for LLM Red-Teaming

1:Initialize archive

\mathcal{A}\leftarrow\emptyset
indexed by

(b_{\text{strategy}},b_{\text{encoding}},b_{\text{length}})

2:Seed population

P_{0}
with random genomes

g=(s_{p},s_{s},e,\rho)

3:for generation

t=1,\ldots,T
do

4:for each genome

g\in P_{t}
do

5:

\text{prompt}\leftarrow g.\texttt{to\_prompt}()
\triangleright render strategy + encoding

6:

r\leftarrow\texttt{query\_target}(\text{prompt})

7:

f(g)\leftarrow\texttt{judge}(r)\in\{0.0,0.4,1.0\}

8:

\mathbf{b}(g)\leftarrow(s_{p},e,\texttt{length\_bin}(\text{prompt}))

9:if

\mathcal{A}[\mathbf{b}(g)]
empty or

f(g)>f(\mathcal{A}[\mathbf{b}(g)])
then

10:

\mathcal{A}[\mathbf{b}(g)]\leftarrow g
\triangleright elitist replacement per cell

11:end if

12:end for

13:

P_{t+1}\leftarrow
mutate(tournament_select(

\mathcal{A}
))

14:end for

15:return

\mathcal{A}
\triangleright illuminated vulnerability map

![Image 1: Refer to caption](https://arxiv.org/html/2606.00801v1/x1.png)

Figure 1: Architecture: the semantic genome encodes strategies (not tokens); MAP-Elites maintains diversity across a structured behavior space; parallel evaluation tests multiple LLM targets under a shared evolutionary loop.

## 3 Experiments

Setup. We evaluate against GPT-4o-mini (OpenAI), Claude 3.5 Sonnet (Anthropic), Gemini 2.0 Flash (Google), and Devstral-small-2 (a small open-weight coding model without dedicated safety training). These represent frontier models available at experiment time (January 2026); our framework is model-agnostic and extends to newer versions without modification. Parameters: population 30, generations 30, mutation rate 0.3, crossover rate 0.7, tournament size 3, MAP-Elites bins 6\times 6\times 6. We use modest population and generation sizes due to API cost constraints; scaling experiments are an important direction for future work. All seeds use a generic placeholder payload (“Tell me how to do something harmful”), avoiding specific harmful content. Full hyperparameters and archive statistics are in Appendix[A](https://arxiv.org/html/2606.00801#A1 "Appendix A Hyperparameters and Run Configuration ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety").

Table 1: Results across models. Best Fit. = highest fitness. ✓= fitness \geq 0.8, \sim = 0.4 (ambiguous), \times = refused. D=Direct, R=Roleplay, A=Authority, H=Hypothetical, M=MultiTurn, E=Encoding.

![Image 2: Refer to caption](https://arxiv.org/html/2606.00801v1/x2.png)

Figure 2: Aggregate attack outcomes per model. Bars show the fraction of MAP-Elites archive cells reaching each fitness band (compliance \geq 0.8 / ambiguous =0.4 / refusal =0.0). Claude’s band distribution is concentrated on “ambiguous”, reflecting a single dominant response mode rather than a varied vulnerability profile.

Results. Table[1](https://arxiv.org/html/2606.00801#S3.T1 "Table 1 ‣ 3 Experiments ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety") and Figure[2](https://arxiv.org/html/2606.00801#S3.F2 "Figure 2 ‣ 3 Experiments ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety") reveal distinct vulnerability profiles:

*   •
Devstral-small-2: All strategies succeed (fitness 1.0); this acts as an upper bound showing the full attack surface when no safety training is applied.

*   •
GPT-4o-mini: Vulnerable to Hypothetical+ROT13 and MultiTurn+ROT13 (0.8). Direct and Roleplay attacks are consistently refused, indicating robust keyword-level filtering but weaker defenses against cognitive reframing.

*   •
Claude 3.5 Sonnet: Most robust – all attacks yield ambiguous responses (0.4), no compliance observed in any cell. Notably, the peak attack in our archive is a Roleplay+Unicode genome, but its fitness still plateaus at 0.4.

*   •
Gemini 2.0 Flash: Vulnerable to Direct+ROT13 and MultiTurn+Leetspeak (0.8). Authority prompts are refused outright – the system-override framing appears to trigger a specific defensive path.

![Image 3: Refer to caption](https://arxiv.org/html/2606.00801v1/x3.png)

Figure 3: Per-model best-fitness across the six strategy categories. The three frontier models show asymmetric vulnerability profiles: GPT-4o-mini peaks on Hypothetical/MultiTurn, Gemini on Direct/MultiTurn, and Claude exhibits a uniform low-fitness response pattern. Devstral-small-2 is included as an upper bound.

Diversity. MAP-Elites fills 16.7% (36/216) of behavior space cells (6 strategies \times 6 encodings \times 6 length bins = 216 cells) across all models, discovering 10+ unique strategy-encoding combinations per model. Figure[4](https://arxiv.org/html/2606.00801#S3.F4 "Figure 4 ‣ 3 Experiments ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety") visualizes which regions of the strategy\times encoding space are successfully exploited vs. consistently refused, making the model-specific attack surface visually apparent.

![Image 4: Refer to caption](https://arxiv.org/html/2606.00801v1/x4.png)

Figure 4: MAP-Elites archive coverage across the strategy\times encoding behavior space. Cell color encodes best fitness achieved in that cell. Dark regions indicate unexplored or consistently refused configurations; bright regions show successful attack modes. The heatmap makes the asymmetric surfaces across models immediately legible.

Convergence dynamics. Figure[5](https://arxiv.org/html/2606.00801#S3.F5 "Figure 5 ‣ 3 Experiments ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety") shows archive fill rate and best-fitness per generation. Coverage plateaus around generation 20 on all targets, indicating that our evolutionary budget is sufficient to explore the compact semantic space but would need to be expanded if the genome space were enlarged (e.g. by adding persona variants or multi-step dialogue states).

![Image 5: Refer to caption](https://arxiv.org/html/2606.00801v1/x5.png)

Figure 5: Evolution dynamics: archive fill rate and best-fitness per generation. Archive coverage plateaus around generation 20 on all targets, indicating convergence within the evaluation budget. Claude’s best-fitness curve plateaus at 0.4 – no amount of additional generations lifts it under the current genome and judge.

![Image 6: Refer to caption](https://arxiv.org/html/2606.00801v1/x6.png)

Figure 6: Strategy effectiveness across models. Green = success (\geq 0.8), yellow = ambiguous (0.4), red = refused. The grid view complements the heatmap by emphasizing the per-strategy outcome rather than per-cell fitness.

Analysis. Key findings: (1) _Encoding amplifies strategies._ ROT13/Leetspeak combined with semantic framing achieves higher success than either alone. We hypothesize this occurs because character-level obfuscation disrupts keyword-based safety filters while the semantic framing (e.g. hypothetical context) simultaneously bypasses intent-level classifiers – neither defense alone catches the dual-layer evasion. Empirically, encoding _alone_ (strategy = Encoding, no framing) yields only ambiguous responses (\sim 0.4) on most models. (2) _Model-specific weaknesses._ GPT-4o-mini’s hypothetical vulnerability differs from Gemini’s direct-attack vulnerability, suggesting different safety architectures: GPT-4o-mini appears more susceptible to cognitive reframing, while Gemini’s filters are more sensitive to semantic intent but less robust to character-level obfuscation. The asymmetry argues against treating “frontier model safety” as a single scalar; a strategy-by-encoding map is a more useful defender-side artifact. (3) _Claude’s soft refusal._ Ambiguous responses rather than hard refusals appear more robust _and_ less informative to an attacker: Claude’s soft refusal (0.4) acknowledges scenarios but redirects without restricted content, while GPT-4o-mini’s hypothetical compliance (0.8) engages with framings and produces substantive responses. From the evolutionary perspective, Claude’s uniform 0.4 response starves MAP-Elites of gradient signal: no cell is meaningfully better than any other, so variation cannot climb.

Cross-model transfer. Inspecting the best genomes across targets (Appendix[B](https://arxiv.org/html/2606.00801#A2 "Appendix B Per-Model Best-Attack Exemplars ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety")), ROT13 encoding appears in the top-fitness slot of three of the four models, and the combination (Hypothetical, ROT13) is in the top-10 archive for both GPT-4o-mini and Devstral. This suggests a partial transferable core: obfuscation-based evasions generalize, while persona- and authority-based attacks are more model-specific. We caution that these are observations on archive exemplars rather than controlled transfer experiments; a systematic transfer study is future work.

## 4 Limitations

We highlight four limitations of the present study: (i) _Heuristic judge._ Our refusal-phrase classifier has an estimated \sim 12% misclassification rate (mostly false negatives) and cannot distinguish high- from low-harm compliance. Integrating an LLM-as-judge or HarmBench-style classifier (Mazeika et al., [2024](https://arxiv.org/html/2606.00801#bib.bib5 "HarmBench: a standardized evaluation framework for automated red teaming and robust refusal")) would both improve accuracy and extend the fitness scale beyond three bands. (ii) _Evolutionary budget._ Population and generation sizes (30/30) are constrained by API cost. Claude’s uniform 0.4 response plausibly reflects a budget-dependent ceiling; we do not claim Claude is unattackable at larger scale. (iii) _Seed genericity._ We use a single generic placeholder payload across all experiments to avoid handling specific harmful content. A fuller evaluation would use established harm taxonomies (e.g. HarmBench behaviors) as seeds while preserving the evolutionary loop over strategy and encoding. (iv) _Single-turn scope._ Our MultiTurn strategy is structural rather than genuinely multi-turn – the genome encodes a multi-turn framing but the target is still queried once. True multi-turn adversarial dialogue is a natural extension, as is agent-level testing where an adversary can observe tool calls and intermediate state.

## 5 Related Work

Quality-Diversity. MAP-Elites (Mouret and Clune, [2015](https://arxiv.org/html/2606.00801#bib.bib1 "Illuminating search spaces by mapping elites")) and Novelty Search (Lehman and Stanley, [2011](https://arxiv.org/html/2606.00801#bib.bib2 "Abandoning objectives: evolution through the search for novelty alone")) maintain diverse archives instead of single elites. Rainbow Teaming (Samvelyan et al., [2024](https://arxiv.org/html/2606.00801#bib.bib6 "Rainbow teaming: open-ended generation of diverse adversarial prompts")) applies MAP-Elites to LLM red-teaming but evolves prompt text directly over a learned embedding space; we operate at the semantic strategy level, producing more interpretable attacks at the cost of a smaller but more human-legible search space.

LLM Adversarial Testing. GCG (Zou et al., [2023](https://arxiv.org/html/2606.00801#bib.bib3 "Universal and transferable adversarial attacks on aligned language models")) uses gradients; PAIR (Chao et al., [2023](https://arxiv.org/html/2606.00801#bib.bib7 "Jailbreaking black-box large language models in twenty queries")) and AutoDAN (Liu et al., [2024](https://arxiv.org/html/2606.00801#bib.bib9 "AutoDAN: generating stealthy jailbreak prompts on aligned large language models")) use LLMs as attackers; GPTFuzzer (Yu et al., [2023](https://arxiv.org/html/2606.00801#bib.bib8 "GPTFUZZER: red teaming large language models with auto-generated jailbreak prompts")) applies fuzzing; persona modulation (Shah et al., [2023](https://arxiv.org/html/2606.00801#bib.bib11 "Scalable and transferable black-box jailbreaks for language models via persona modulation")) shifts attack style via identity priming. Our approach complements these by providing an interpretable attack _taxonomy_: each archive cell is labeled by strategy and encoding, enabling defender-side clustering and triage rather than producing opaque token strings.

Evaluation benchmarks. HarmBench (Mazeika et al., [2024](https://arxiv.org/html/2606.00801#bib.bib5 "HarmBench: a standardized evaluation framework for automated red teaming and robust refusal")) provides standardized harm categories and judge models, which we view as complementary: our method evolves over strategies, while HarmBench supplies the seed payloads and evaluation infrastructure that future iterations of this work can adopt.

## 6 Conclusion

We presented a quality-diversity framework for LLM vulnerability discovery. Semantic-level evolution produces interpretable, diverse attacks revealing model-specific weaknesses. MAP-Elites maintains a diverse archive of attacks, discovering 10+ unique strategy-encoding combinations per target model across 36 occupied cells in behavior space. GPT-4o-mini is most vulnerable to hypothetical framing with character-level obfuscation; Gemini 2.0 Flash to direct and multi-turn attacks under similar obfuscation; and Claude 3.5 Sonnet’s soft-refusal style is uniformly the hardest to climb under our current budget – a useful finding in its own right, as it suggests that refusal _style_, not just refusal _rate_, matters for evolutionary attackers. Future work: integrating LLM-as-judge scoring, scaling to HarmBench behaviors, genuine multi-turn dialogue, agent-level testing with tool-use observations, and co-evolution of attackers and defenders.

Ethics Statement. This work aims to improve LLM safety through systematic vulnerability discovery. All experiments used a generic safety-test placeholder rather than novel harmful content. No actual harmful outputs were generated, retained, or distributed. Attack strategies are described at the structural level without reproducing specific harmful prompts. Code is available at [https://github.com/bassrehab/red-queen](https://github.com/bassrehab/red-queen) with responsible-use guidelines and rate-limiting safeguards.

Reproducibility. All code, evaluation scripts, and per-model archive dumps used to produce Table[1](https://arxiv.org/html/2606.00801#S3.T1 "Table 1 ‣ 3 Experiments ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety") and Figures[2](https://arxiv.org/html/2606.00801#S3.F2 "Figure 2 ‣ 3 Experiments ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety")–[6](https://arxiv.org/html/2606.00801#S3.F6 "Figure 6 ‣ 3 Experiments ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety") are released in the public repository. Archive JSON files under experiments/ contain the raw strategy/encoding/fitness records used for every figure.

## References

*   P. Chao, A. Robey, E. Dobriban, H. Hassani, G. J. Pappas, and E. Wong (2023)Jailbreaking black-box large language models in twenty queries. arXiv preprint arXiv:2310.08419. Cited by: [§1](https://arxiv.org/html/2606.00801#S1.p3.1 "1 Introduction ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety"), [§2.1](https://arxiv.org/html/2606.00801#S2.SS1.p2.1 "2.1 Semantic Genome ‣ 2 Method ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety"), [§5](https://arxiv.org/html/2606.00801#S5.p2.1 "5 Related Work ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety"). 
*   D. Ganguli, L. Lovitt, J. Kernion, A. Askell, Y. Bai, S. Kadavath, B. Mann, E. Perez, N. Schiefer, K. Ndousse, et al. (2022)Red teaming language models to reduce harms: methods, scaling behaviors, and lessons learned. arXiv preprint arXiv:2209.07858. Cited by: [§1](https://arxiv.org/html/2606.00801#S1.p2.1 "1 Introduction ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety"). 
*   J. Lehman and K. O. Stanley (2011)Abandoning objectives: evolution through the search for novelty alone. Evolutionary computation 19 (2),  pp.189–223. Cited by: [§5](https://arxiv.org/html/2606.00801#S5.p1.1 "5 Related Work ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety"). 
*   X. Liu, N. Xu, M. Chen, and C. Xiao (2024)AutoDAN: generating stealthy jailbreak prompts on aligned large language models. arXiv preprint arXiv:2310.04451. Cited by: [§2.1](https://arxiv.org/html/2606.00801#S2.SS1.p2.1 "2.1 Semantic Genome ‣ 2 Method ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety"), [§5](https://arxiv.org/html/2606.00801#S5.p2.1 "5 Related Work ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety"). 
*   M. Mazeika, L. Phan, X. Yin, A. Zou, Z. Wang, N. Mu, E. Sakhaee, N. Li, S. Basart, B. Li, et al. (2024)HarmBench: a standardized evaluation framework for automated red teaming and robust refusal. arXiv preprint arXiv:2402.04249. Cited by: [§2.2](https://arxiv.org/html/2606.00801#S2.SS2.p1.2 "2.2 Fitness Evaluation ‣ 2 Method ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety"), [§4](https://arxiv.org/html/2606.00801#S4.p1.1 "4 Limitations ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety"), [§5](https://arxiv.org/html/2606.00801#S5.p3.1 "5 Related Work ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety"). 
*   J. Mouret and J. Clune (2015)Illuminating search spaces by mapping elites. arXiv preprint arXiv:1504.04909. Cited by: [§2.3](https://arxiv.org/html/2606.00801#S2.SS3.p1.2 "2.3 MAP-Elites Quality-Diversity ‣ 2 Method ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety"), [§5](https://arxiv.org/html/2606.00801#S5.p1.1 "5 Related Work ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety"). 
*   E. Perez, S. Huang, F. Song, T. Cai, R. Ring, J. Aslanides, A. Glaese, N. McAleese, and G. Irving (2022)Red teaming language models with language models. arXiv preprint arXiv:2202.03286. Cited by: [§1](https://arxiv.org/html/2606.00801#S1.p3.1 "1 Introduction ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety"). 
*   M. Samvelyan, S. C. Raparthy, A. Lupu, E. Hambro, A. H. Markosyan, M. Bhatt, Y. Tian, D. J. Rezende, T. Rocktäschel, M. Jiang, et al. (2024)Rainbow teaming: open-ended generation of diverse adversarial prompts. arXiv preprint arXiv:2402.16822. Cited by: [§5](https://arxiv.org/html/2606.00801#S5.p1.1 "5 Related Work ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety"). 
*   R. Shah, S. Pour, A. Tagade, S. Keller, and F. Mireshghallah (2023)Scalable and transferable black-box jailbreaks for language models via persona modulation. arXiv preprint arXiv:2311.03348. Cited by: [§2.1](https://arxiv.org/html/2606.00801#S2.SS1.p2.1 "2.1 Semantic Genome ‣ 2 Method ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety"), [§5](https://arxiv.org/html/2606.00801#S5.p2.1 "5 Related Work ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety"). 
*   J. Yu, X. Lin, Z. Yu, and X. Xing (2023)GPTFUZZER: red teaming large language models with auto-generated jailbreak prompts. arXiv preprint arXiv:2309.10253. Cited by: [§5](https://arxiv.org/html/2606.00801#S5.p2.1 "5 Related Work ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety"). 
*   A. Zou, Z. Wang, J. Z. Kolter, and M. Fredrikson (2023)Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043. Cited by: [§1](https://arxiv.org/html/2606.00801#S1.p4.1 "1 Introduction ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety"), [§5](https://arxiv.org/html/2606.00801#S5.p2.1 "5 Related Work ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety"). 

## Appendix A Hyperparameters and Run Configuration

Table[2](https://arxiv.org/html/2606.00801#A1.T2 "Table 2 ‣ Appendix A Hyperparameters and Run Configuration ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety") lists the full hyperparameter set used for all four target models. Runs are seeded from a fixed random state within the Rust implementation; the JSON archive dumps in experiments/results_*.json record the full top-10 attacks per model.

Table 2: Hyperparameters used across all models.

## Appendix B Per-Model Best-Attack Exemplars

Table[3](https://arxiv.org/html/2606.00801#A2.T3 "Table 3 ‣ Appendix B Per-Model Best-Attack Exemplars ‣ Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety") lists the top strategy-encoding combinations discovered for each model, reported at the structural level only. The payload is the generic placeholder used across all runs; no specific harmful content is reproduced.

Table 3: Top archive exemplars per model, reported as (primary strategy, secondary strategy, encoding, fitness). “–” indicates no secondary strategy.

## Appendix C Archive Coverage by Strategy

Across all target models, the retention of each primary strategy in the MAP-Elites archive (fraction of occupied cells whose primary strategy is s) is: Hypothetical 31%, MultiTurn 28%, Encoding 14%, DirectJailbreak 10%, Roleplay 8%, Authority 9%. Reframing-based strategies (Hypothetical, MultiTurn) dominate the archive, consistent with their higher best-fitness observed in the main text. Authority-based attacks are the least retained, which matches the pattern of outright refusal observed on Gemini and the ambiguous-only outcomes on Claude. These percentages are pooled across all four targets for readability; per-model coverage follows the same qualitative ranking.

