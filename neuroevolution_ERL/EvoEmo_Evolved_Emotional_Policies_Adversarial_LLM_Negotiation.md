Title: EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation

URL Source: https://arxiv.org/html/2509.04310

Published Time: Tue, 14 Oct 2025 02:06:41 GMT

Markdown Content:
\settopmatter

printacmref=false\setcopyright ifaamas \acmConference[AAMAS ’26]Proc. of the 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2026)May 25 – 29, 2026 Paphos, CyprusC. Amato, L. Dennis, V. Mascardi, J. Thangarajah (eds.) \copyrightyear 2026 \acmYear 2026 \acmDOI\acmPrice\acmISBN\acmSubmissionID 572

Yunbo Long 1 Liming Xu 1 Lukas Beckenbauer 3 Yuhan Liu 2 Alexandra Brintrup 1,4

1 Department of Engineering, University of Cambridge, UK 

2 Rotman School of Management, University of Toronto, Canada 

3 TUM School of Management, Technical University of Munich, Germany 

4 The Alan Turing Institute, London, UK 

{yl892,lx249,ab702}@cam.ac.uk yl972@cantab.ac.uk l.beckenbauer@tum.de

###### Abstract.

Recent research on Chain-of-Thought (CoT) reasoning in Large Language Models (LLMs) has demonstrated their capability for complex, multi-turn negotiations, a task that inherently involves understanding and leveraging human emotional cues. As autonomous LLM agents are increasingly deployed to perform such tasks, a new paradigm of LLM-vs-LLM interaction is emerging as a critical research domain. However, while these agents are built on models trained to process human emotion, existing negotiation strategies for LLM agents largely overlook the functional role of emotions as a strategic action. This renders them passive and vulnerable to manipulation and strategic exploitation by other, more sophisticated LLM counterparts. To address this gap, we present EvoEmo, an evolutionary reinforcement learning framework that optimizes dynamic emotional expression in negotiations. EvoEmo models emotional state transitions as a Markov Decision Process and employs population-based genetic optimization to evolve high-reward emotion policies across diverse negotiation scenarios. We further propose an evaluation framework with two baselines—vanilla strategies and fixed-emotion strategies—for benchmarking emotion-aware negotiation. Extensive experiments and ablation studies show that EvoEmo consistently outperforms both baselines, achieving higher success rates, higher efficiency, and increased buyer savings. This findings highlight the importance of adaptive emotional expression in enabling more effective LLM agents for multi-turn negotiation.

###### Key words and phrases:

Affective Computing, Large Language Models, Multi-turn Negotiation, Evolutionary Reinforcement Learning, LLM Agents

## 1. Introduction

“Emotions aren’t the obstacles to a successful negotiation while they are the means.”

— Chris Voss, Never Split the Difference

Extensive behavioral research has established that human decision-making deviates from classical economic rationality, being shaped by psychological biases and emotional states (Hilbert, [2012](https://arxiv.org/html/2509.04310v3#bib.bib8); Baumeister et al., [2012](https://arxiv.org/html/2509.04310v3#bib.bib3); Riaz et al., [2012](https://arxiv.org/html/2509.04310v3#bib.bib17)). While modern Large Language Models (LLMs) have made progress in replicating personality-driven behaviors (Wei et al., [2025](https://arxiv.org/html/2509.04310v3#bib.bib21)), the role of emotion as a strategic, dynamic force remains understudied (Liu and Long, [2025](https://arxiv.org/html/2509.04310v3#bib.bib14)), especially compared to static, trait-based approaches (Huang and Hadfi, [2024](https://arxiv.org/html/2509.04310v3#bib.bib10)). This gap is particularly critical in fine-grained negotiation scenarios such as price bargaining (Lin et al., [2023](https://arxiv.org/html/2509.04310v3#bib.bib12)), where emotions directly influence on tactical choices (e.g., frustration prompting premature concessions, or excitement triggering overly aggressive bids), with immediate consequences for negotiation outcomes.

However, a paradigm shift is underway. Compared to the traditional focus on LLM agent-human user interaction, autonomous LLM agents are increasingly being deployed to perform complex tasks like negotiation amongst themselves (Lin et al., [2024](https://arxiv.org/html/2509.04310v3#bib.bib13)). The natural consequence of this proliferation is a new ecosystem defined by agent-to-agent interactions (Abbasiantaeb et al., [2024](https://arxiv.org/html/2509.04310v3#bib.bib2); Lin et al., [2024](https://arxiv.org/html/2509.04310v3#bib.bib13)). In domains such as e-commerce, supply chain management, and decentralized autonomous organizations (DAOs) (Hu et al., [2025](https://arxiv.org/html/2509.04310v3#bib.bib9)), the scale of transactions necessitates agents that can negotiate with each other directly, with minimal human intervention. This emerging “agentic ecology” demands a focus on strategic robustness, where an agent’s ability to avoid exploitation and achieve favorable outcomes is paramount. The strategic value of emotion in this LLM-vs-LLM context stems not from anthropological fidelity but from the foundational architecture of the underlying language model. An LLM’s next-token prediction objective, trained on a corpus of human communication, has implicitly encoded emotional cues as high-dimensional features that strongly condition model output. Thus, emotional labels serve as a deterministic steering mechanism within the model’s probabilistic generation space where emotional cues (e.g., “frustration,” “satisfaction”) are strongly correlated with specific stances and potential actions. This makes an emotional signal a available control mechanism, which can directly shape an opponent LLM’s response generation.

![Image 1: Refer to caption](https://arxiv.org/html/2509.04310v3/figs/workflow.png)

Figure 1. Illustration of the workflow of the EvoEmo framework.

Despite this, current LLM agents are fundamentally limited in their use of emotion. Techniques like Direct Preference Optimization (DPO) (Gao et al., [2025](https://arxiv.org/html/2509.04310v3#bib.bib5)) and Reinforcement Learning from Human Feedback (RLHF) (Kasbouya and Sael, [2025](https://arxiv.org/html/2509.04310v3#bib.bib11)) have made them proficient in recognizing and reacting to human emotional cues, but these capabilities remain reactive. They fail to leverage emotion proactively as a tool to influence the negotiation trajectory and assert strategic dominance. This leaves them vulnerable in adversarial LLM-vs-LLM interactions, where they exhibit three critical deficiencies:

*   •Tactical Inflexibility. Current LLM agents operate with static emotional policies, generating predictable response patterns across negotiation turns. This lack of dynamic modulation makes them highly susceptible to exploitation by an adversarial LLM agent that can easily learn and counter their strategy. 
*   •Adversarial Naivety. While capable of recognizing emotional cues, LLM agents lack the strategic reasoning to discern genuine behavioral patterns from deliberate, tactical feints. This leaves them vulnerable to manipulation by opponent agents. 
*   •Strategic Myopia. Existing agents treat each turn in isolation, lacking a long-term policy for emotional dynamics. They fail to conduct reasoning about how their current emotional expression will influence the opponent’s future states and actions, preventing them from proactively shaping the negotiation trajectory for multi-turn advantage. 

These deficiencies explain why LLMs, despite advanced reasoning capabilities, may systematically underperform in emotion-sensitive negotiations against sophisticated counterparts. To address this gap, we present EvoEmo, an evolutionary reinforcement learning framework that optimizes dynamic emotional expression for LLM agents. We demonstrate that evolved emotional policies directly and significantly impact negotiation outcomes between LLM agents, providing a pathway to more effective, strategic, and robust autonomous negotiators.

To address these limitations, we propose EvoEmo, an evolutionary reinforcement learning framework designed to optimize dynamic emotion policies for multi-turn negotiations. The core objective of EvoEmo is to evolve emotional strategies that are specifically effective at countering other LLM agents. Our approach employs population-level evolutionary learning to discover optimal emotion transition rules, iteratively refining policies based on rewards achieved during simulated negotiations against target LLM opponents. Evolutionary operations—including crossover and mutation—enable efficient exploration of the policy space and propagation of high-reward emotional strategies that prove successful against specific adversaries. By combining the exploration advantages of population-based optimization with the sequential decision-making framework of reinforcement learning, EvoEmo provides an effective approach for evolving complex emotional policies that are challenging to optimize with gradient-based methods alone.

To benchmark our approach, we also introduce a cross-model negotiation framework with two baselines: a vanilla (prompt-free) strategy and a fixed-emotion strategy. Through extensive experiments, we demonstrate that evolutionarily optimized emotion policies substantially enhance LLM agents’ bargaining capabilities, achieving superior performance in success rate, buyer savings, and negotiation efficiency against other LLMs. Our main contributions are summarized as follows:

*   •We propose EvoEmo, a novel framework that combines evolutionary algorithms with reinforcement learning to evolve dynamic emotional policies for LLM agents. 
*   •We develop a cross-model negotiation framework with emotion-aware baselines for systematic benchmarking. 
*   •We validate EvoEmo through LLM-vs-LLM negotiation scenarios and demonstrate that our approach largely improves negotiation outcomes across multiple performance metrics. 

This work establishes a new benchmark for emotion-aware negotiation and highlights the critical role of adaptive emotional intelligence in next-generation autonomous AI systems.

## 2. Related Work

### 2.1. Emotions in Negotiation Dynamics

While personality traits influence general negotiation tendencies (e.g., agreeableness correlating with cooperative strategies) (Huang and Hadfi, [2024](https://arxiv.org/html/2509.04310v3#bib.bib10)), emotions dominate real-time bargaining dynamics through three key mechanisms. First, temporal alignment ensures that negotiations—unfolding over rapid conversational turns lasting seconds to minutes—operate on a timescale closely aligned with emotional fluctuations, such as facial expressions and transient mood shifts, thereby often overriding longer-term personality influences (Griessmair et al., [2015](https://arxiv.org/html/2509.04310v3#bib.bib6)). Second, strategic flexibility allows negotiators to adapt emotional expressions contextually; for instance, displaying tactical anger to convey firmness or simulate scarcity, temporarily overriding personality defaults (cite). Third, interactive amplification describes the process of emotional contagion, where one party’s affective stance elicits corresponding emotions in the opponent, forming feedback loops that modulate bargaining power over multiple turns (Olekalns and Druckman, [2014](https://arxiv.org/html/2509.04310v3#bib.bib15)). This fine-grained affective interplay underpins dynamic shifts in human negotiation behavior, making emotions a decisive factor in real-time decision-making.

### 2.2. LLM Agents in Negotiation Systems

Current LLM-based negotiation systems demonstrate significant capabilities in emotion recognition but fall short in strategic emotional adaptation. Frameworks like AgreeMate (Chatterjee et al., [2024](https://arxiv.org/html/2509.04310v3#bib.bib4)) and ACE (Shea et al., [2024](https://arxiv.org/html/2509.04310v3#bib.bib19)) show proficiency in inferring emotions through chain-of-thought reasoning, yet lack mechanisms for dynamic emotional strategy evolution. For instance, AgreeMate’s modular architecture optimizes fixed buyer/seller roles but cannot evolve emotional strategies amid negotiation. Similarly, ACE’s coaching system detects errors but cannot simulate the real-time emotional volatility that characterizes human bargaining. Consequently, current LLM agents remain largely reactive rather than proactive in their negotiation behavior.

Recent reinforcement learning advances have improved LLM negotiation capabilities. Methods such as Q-learning (Watkins and Dayan, [1992](https://arxiv.org/html/2509.04310v3#bib.bib20)) and PPO (Schulman et al., [2017](https://arxiv.org/html/2509.04310v3#bib.bib18)) have been adapted to optimize negotiation policies via reward maximization. GENTEEL-NEGOTIATOR (Priya et al., [2025](https://arxiv.org/html/2509.04310v3#bib.bib16)) further advances polite negotiation dialogue systems through mixture-of-expert reinforcement learning with specialized rewards for strategy, politeness, and coherence. Multi-agent evolutionary algorithms like EvoAgent (Yuan et al., [2024](https://arxiv.org/html/2509.04310v3#bib.bib22)) offer promising pathways for strategy enhancement through population-based optimization. However, these approaches face fundamental limitations in emotion-aware negotiation scenarios. Reinforcement learning methods struggle with the sparse-reward, high-volatility nature of negotiation environments, while evolutionary algorithms typically evaluate policies based on terminal outcomes, providing weak credit assignment for the nuanced interplay between consecutive emotional states in multi-turn negotiations. Crucially, these methods lack online learning capabilities essential for price negotiation across diverse products and scenarios. Real-world negotiation agents must adapt emotional strategies in real-time when facing different product types, price ranges, and debtor circumstances without requiring separate training for each variation. Our approach integrates evolutionary exploration with reinforcement learning to enable online emotion policy optimization, allowing agents to continuously adapt emotional strategies during execution across varied product negotiations.

## 3. EvoEmo Framework

We present EvoEmo, a framework for evolutionary optimization of emotional negotiation policies ([Figure 1](https://arxiv.org/html/2509.04310v3#S1.F1 "Figure 1 ‣ 1. Introduction ‣ EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation")). Our approach consists of two main stages: (1) Negotiation Setup: We configure buyer and seller agents using state-of-the-art LLMs, providing product details and role-specific parameters (target price, cost price, and market context). See Appendix 6 for prompt details. (2) EvoEmo Optimization: We evolve buyer emotion policies $\pi_{\omega}^{i}$ through population-based optimization. Each generation initializes $m$ policies with randomized emotional sequences, evaluates them via multi-turn negotiations using reward function $R ​ \left(\right. S \left.\right)$, then applies selection, crossover, and mutation to produce improved policies $\hat{\pi} ​ \omega^{i}$ for the next generation. This process iterates until convergence, yielding optimal policies $\pi ​ \omega^{*}$. The full EvoEmo is outlined in Algorithm [1](https://arxiv.org/html/2509.04310v3#alg1 "Algorithm 1 ‣ 3.3. Evolved Reinforcement Learning ‣ 3. EvoEmo Framework ‣ EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation").

### 3.1. Emotion-Aware MDPs

We formalize the negotiation process as a Markov Decision Process (MDP) defined by the tuple $\left(\right. \mathcal{S} , \mathcal{A} , \mathcal{P} , \mathcal{R} \left.\right)$. The state space $\mathcal{S}$ is defined by tuples $\left(\right. t , e_{t} , p_{t} \left.\right)$, where $t$ is the current negotiation turn, $e_{t} \in \mathcal{E}$ is the current emotion from the set of seven basic emotions, and $p_{t}$ represents the price history and current offer. The action space $\mathcal{A}$ consists of all textual responses generated by the LLM. The state transition dynamics $\mathcal{P}$ are governed by the emotional policy $\pi_{\omega}$, which determines the probability distribution over the next emotional state, $e_{t + 1} sim \pi_{\omega} \left(\right. \cdot \left|\right. s_{t} \left.\right)$, thereby shaping the agent’s strategic trajectory.

### 3.2. Policy Representation and Evolution

The EvoEmo framework operates through a continuous cycle of policy evolution and execution, where each emotional policy $\pi_{\omega} = \left(\right. T , \mathbf{P} \left.\right)$ encapsulates the agent’s emotional behavior and is refined via evolutionary mechanisms. The policy consists of temperature parameters $T = \left(\right. \tau_{0} , \delta \left.\right)$ controlling response stochasticity through an exponential decay schedule $\tau ​ \left(\right. t \left.\right) = max ⁡ \left(\right. 0.1 , \tau_{0} \cdot \left(\left(\right. 1 - \delta \left.\right)\right)^{t} \left.\right)$, and a transition matrix $\mathbf{P} \in \mathbb{R}^{7 \times 7}$ modeling emotional state transitions with $P_{i ​ j} = \mathbb{P} ​ \left(\right. e_{t + 1} = j \left|\right. e_{t} = i \left.\right)$ and $\sum_{j = 1}^{7} P_{i ​ j} = 1$ ensuring valid probability distributions. The evolutionary process refines emotional policies through sequence operations enhanced by Bayesian updating. Each policy maintains a population of $K$ emotion sequences $E_{1} , E_{2} , \ldots , E_{K}$, where each sequence $E_{k} = \left(\right. e_{1} , \ldots , e_{n} \left.\right)$ represents a distinct emotional trajectory. This multi-sequence approach ensures robust statistical estimation of transition probabilities. During crossover, parent policies exchange sequence populations, and offspring sequences $\hat{E}$ are generated via single-point crossover at randomly selected positions $k \in \left[\right. 1 , n \left]\right.$. Mutation introduces behavioral diversity by stochastically altering individual emotions within sequences with probability $p_{m}$, where selected emotions $e_{t}$ are replaced with $e^{'} sim U ​ \left(\right. \mathcal{E} \left.\right)$. The transition matrix $\mathbf{P}$ is updated using Bayesian estimation that aggregates evidence across the entire sequence population $\mathbf{P}_{\text{new}} = \lambda ​ \mathbf{P}_{\text{old}} + \left(\right. 1 - \lambda \left.\right) ​ \mathbf{P}_{\text{pop}}$, where $\mathbf{P}_{\text{pop}}$ is the maximum likelihood estimate derived from all $K$ sequences:

$$
P_{\text{pop} , i ​ j} = \frac{\sum_{k = 1}^{K} \text{count} ​ \left(\right. e_{t} = i , e_{t + 1} = j ​ \textrm{ }\text{in}\textrm{ } ​ E_{k} \left.\right) + \alpha}{\sum_{k = 1}^{K} \text{count} ​ \left(\right. e_{t} = i ​ \textrm{ }\text{in}\textrm{ } ​ E_{k} \left.\right) + 7 ​ \alpha}
$$(1)

Here, $\lambda \in \left[\right. 0 , 1 \left]\right.$ controls the update rate, $\alpha > 0$ provides Dirichlet smoothing to handle sparse transitions, and the denominator ensures valid probability distributions. During execution, the policy determines the agent’s emotional behavior dynamically while simultaneously optimizing the transition matrix through online learning. At each negotiation turn $t$, the next emotion $e_{t}$ is sampled from the transition probabilities associated with the current state, i.e., $e_{t} sim \mathbf{P} ​ \left[\right. e_{t - 1} , : \left]\right.$. This evolving emotion sequence continuously refines the transition probabilities through Bayesian updates as the negotiation unfolds, enabling real-time emotional adaptation to debtor behavior. The sampled emotion is then converted into a conditioning prompt (e.g., “You feel [angry].”) that guides the large language model’s response generation.

### 3.3. Evolved Reinforcement Learning

The EvoEmo framework formulates emotional policy optimization as an evolutionary reinforcement learning problem. Each policy $\pi_{\omega} = \left(\right. T , \mathbf{P} \left.\right)$ is evaluated and refined through generational evolution, while its evolutionary representation $\Pi_{\omega} = \left(\right. E , T , \mathbf{P} \left.\right)$ also maintains a population of emotion sequences $E = E_{1} , E_{2} , \ldots , E_{K}$ used solely for crossover and mutation operations. The effectiveness of a policy is measured by a reward function that quantifies negotiation success and efficiency:

$$
R ​ \left(\right. S \left.\right) = 𝟏_{\text{success}} \cdot \alpha \cdot \frac{b ​ \left(\right. S \left.\right)}{1 + log ⁡ \left(\right. e ​ \left(\right. S \left.\right) \left.\right)}
$$(2)

where $𝟏_{\text{success}}$ is an indicator equal to 1 if the negotiation succeeds and 0 otherwise, $b ​ \left(\right. S \left.\right) \in \left[\right. 0 , 1 \left]\right.$ represents the normalized buyer savings, $e ​ \left(\right. S \left.\right)$ denotes the number of negotiation rounds, and $\alpha$ is a weighting coefficient. The logarithmic term $log ⁡ \left(\right. e ​ \left(\right. S \left.\right) \left.\right)$ ensures robust scaling by moderating the impact of round count on the reward. During each generation, a population of $m$ candidate policies $\Pi_{\omega}^{i}$ is deployed in multi-turn negotiation simulations driven by the large language model $\mathcal{M}$. Each policy produces an emotional trajectory and dialogue sequence whose performance is evaluated via $R ​ \left(\right. S^{i} \left.\right)$. The selection of policies follows a probabilistic strategy proportional to their scaled rewards:

$$
P ​ \left(\right. \Pi_{\omega}^{i} \left.\right) = \frac{exp ⁡ \left(\right. R ​ \left(\right. S^{i} \left.\right) / \lambda \left.\right)}{\sum_{j = 1}^{m} exp ⁡ \left(\right. R ​ \left(\right. S^{j} \left.\right) / \lambda \left.\right)} ,
$$(3)

where $\lambda > 0$ controls selection pressure. To promote stability, the top $\rho$ policies (elites) are preserved across generations, ensuring monotonic improvement. The remaining policies undergo evolutionary refinement via sequence-based operators: crossover combines parental sequence populations with probability $p_{c}$ via single-point recombination, while mutation perturbs individual emotions with probability $p_{m}$. The resulting sequences are converted into policy parameters $\left(\right. T , \mathbf{P} \left.\right)$ via the Bayesian update rule from the previous section. This iterative process generates population $\left(\hat{\Pi}\right)_{\omega}^{i}$ and continues until convergence, defined by either (1) fitness improvement below $\epsilon = 0.01$ for five consecutive generations or (2) completion of $G$ generations. The detailed algorithm appears in Appendix 8.

Algorithm 1 EvoEmo: Evolutionary Emotion Optimization for Multi-Agent Negotiation

1:LLM Agents:

$\mathcal{M}_{1}$
(buyer),

$\mathcal{M}_{2}$
(seller),

$\mathcal{M}_{3}$
(mediator)

2:Product description

$D$
, initial prompts for buyer and seller

3:Hyperparameters: population

$m$
, generations

$G$
, max turns

$T_{\text{max}}$
, initial temp

$\tau_{0}$
, decay

$\delta$

4:Optimized emotion policy

$\pi_{\omega}^{*}$

5:Initialization:

6:

$P_{0} \leftarrow \left{\right. \pi_{\omega}^{\left(\right. 1 \left.\right)} , \pi_{\omega}^{\left(\right. 2 \left.\right)} , \ldots , \pi_{\omega}^{\left(\right. m \left.\right)} \left.\right}$
$\triangleright$ Initialize population with random emotion sequences

7:for generation

$g = 0$
to

$G - 1$
do

8:

$R \leftarrow \emptyset$
$\triangleright$ Reward storage for current generation

9:for each policy

$\pi_{\omega} \in P_{g}$
do

10:Nego-simulation(

$\pi_{\omega} , \mathcal{M}_{1} , \mathcal{M}_{2} , \mathcal{M}_{3} , D$
)

11:

$R ​ \left(\right. \pi_{\omega} \left.\right) \leftarrow \text{Reward} ​ \left(\right. \text{outcome} \left.\right)$
$\triangleright$ Store policy reward

12:end for

13:

$P^{'} \leftarrow \text{SelectParents} ​ \left(\right. P_{g} , R \left.\right)$
$\triangleright$ Tournament selection

14:

$P^{′′} \leftarrow \text{ApplyGeneticOperations} ​ \left(\right. P^{'} \left.\right)$
$\triangleright$ Crossover and mutation

15:

$P_{g + 1} \leftarrow \text{FormNewPopulation} ​ \left(\right. P^{′′} \left.\right)$
$\triangleright$ Elitism preservation

16:end for

17:

$\pi_{\omega}^{*} \leftarrow arg ⁡ max_{\pi_{\omega} \in P_{G}} ⁡ R ​ \left(\right. \pi_{\omega} \left.\right)$

18:return

$\pi_{\omega}^{*}$

19:procedure FormNewPopulation(

$P_{\text{new}}$
)

20:

$b ​ e ​ s ​ t ​ _ ​ r ​ e ​ w ​ a ​ r ​ d \leftarrow - \infty$

21:

$b ​ e ​ s ​ t ​ _ ​ p ​ o ​ l ​ i ​ c ​ y \leftarrow \emptyset$

22:for each policy

$\pi_{\omega} \in P_{\text{new}}$
do

23:Nego-simulation(

$\pi_{\omega} , \mathcal{M}_{1} , \mathcal{M}_{2} , \mathcal{M}_{3} , D$
)

24:

$R^{'} ​ \left(\right. \pi_{\omega} \left.\right) \leftarrow \text{Reward} ​ \left(\right. \text{outcome} \left.\right)$
$\triangleright$ Store policy reward

25:if

$R^{'} ​ \left(\right. \pi_{\omega} \left.\right) > b ​ e ​ s ​ t ​ _ ​ r ​ e ​ w ​ a ​ r ​ d$
then

26:

$b ​ e ​ s ​ t ​ _ ​ r ​ e ​ w ​ a ​ r ​ d \leftarrow R^{'} ​ \left(\right. \pi_{\omega} \left.\right)$

27:

$b ​ e ​ s ​ t ​ _ ​ p ​ o ​ l ​ i ​ c ​ y \leftarrow \pi_{\omega}$

28:end if

29:end for

30:return

$b ​ e ​ s ​ t ​ _ ​ p ​ o ​ l ​ i ​ c ​ y$
$\triangleright$ Return the policies with highest reward for all population

31:end procedure

32:procedure Nego-simulation(

$\pi_{\omega} , \mathcal{M}_{1} , \mathcal{M}_{2} , \mathcal{M}_{3} , D$
)

33: Initialize environment with product

$D$

34:

$e_{t} \leftarrow \text{neutral}$
$\triangleright$ Initial emotional state

35:

$\text{outcome} \leftarrow \text{ongoing}$

36:for

$t = 1$
to

$T_{\text{max}}$
do

37:

$\tau ​ \left(\right. t \left.\right) \leftarrow max ⁡ \left(\right. 0.1 , \tau_{0} \cdot \left(\left(\right. 1 - \delta \left.\right)\right)^{t} \left.\right)$
$\triangleright$ Temperature schedule

38:

$e_{t + 1} sim \pi_{\omega} ​ \left(\right. e \left|\right. s_{t} , \tau ​ \left(\right. t \left.\right) \left.\right)$
$\triangleright$ Sample emotional transition

39:

$a_{\text{buyer}} \leftarrow \mathcal{M}_{1} ​ \left(\right. \text{context} , e_{t + 1} \left.\right)$
$\triangleright$ Generate emotional response

40:

$a_{\text{seller}} \leftarrow \mathcal{M}_{2} ​ \left(\right. \text{context} \left.\right)$
$\triangleright$ Generate standard response

41:

$\text{outcome} \leftarrow \mathcal{M}_{3} ​ \left(\right. a_{\text{buyer}} , a_{\text{seller}} \left.\right)$
$\triangleright$ Mediate negotiation

42:if outcome

$\in \left{\right. \text{accept} , \text{breakdown} \left.\right}$
then

43:break

44:end if

45:end for

46:end procedure

![Image 2: Refer to caption](https://arxiv.org/html/2509.04310v3/figs/savings.png)

Figure 2.  Negotiation results in terms of buyer savings (%, $\uparrow$) across the nine buyer-seller pairs. Black vertical lines on top of each bar indicate the 95% confidence interval (CI) for each setting. 

## 4. Experimental Setting

### 4.1. Datasets

Likewise to Huang and Hadfi ([2024](https://arxiv.org/html/2509.04310v3#bib.bib10)), we selected a subset of negotiation cases from the CraigslistBargain dataset (He et al., [2018](https://arxiv.org/html/2509.04310v3#bib.bib7))—a widely-used benchmark for negotiation studies—for evaluation. This subset comprises 20 distinct multi-turn negotiation scenarios spanning diverse product categories, including electronics, furniture, vehicles, and housing. Each scenario specifies: (1) product details (name, category, and description), (2) agent-specific target prices for the seller, and (3) emotion annotations that support realistic bargaining dynamics. The selected dataset spans a wide price range ($50–$5,000) and includes varied item conditions (brand new or used), thereby enabling comprehensive evaluation of negotiation policies under heterogeneous market conditions.

![Image 3: Refer to caption](https://arxiv.org/html/2509.04310v3/figs/examples.png)

Figure 3. Examples of manipulative and deceptive tactics

### 4.2. LLM Agents

Representative state-of-the-art LLMs from leading AI providers—GPT-5-mini (OpenAI), Gemini-2.5-Pro (Google), and DeepSeek-V3.1.1 (DeepSeek)—were selected to power both buyer and seller agents in our experiments. These models were chosen for their demonstrated strengths in conversational capabilities, strategic reasoning performance, and accessibility via APIs, providing comprehensive coverage across diverse model architectures.

### 4.3. Baselines

For evaluation, we focus on buyer agents with different emotion policies while keeping seller agents fixed in a vanilla setting, where agents receive no explicit emotional prompts. We define two baselines for comparison. The first baseline comprises only vanilla agents, where neither buyer nor seller receives emotional guidance. This setup ensures that both agents act solely according to their intrinsic emotional tendencies and strategic reasoning abilities, providing a reference point that reflects default negotiation behavior. The second baseline pairs a vanilla seller and a fixed-emotion buyer, where the buyer maintains a constant emotional profile (e.g., happy or neutral) throughout the negotiation. By comparing these baselines with a setup in which the buyer’s emotion is optimized via our proposed approach—EvoEmo, we can quantify the impact of emotions on negotiation outcomes and assess the effectiveness of EvoEmo in enhancing LLM-agent-based, emotion-driven negotiations.

### 4.4. Experimental Setup

Simulated negotiations are conducted to evaluate the impact of different buyer emotion configurations—vanilla, fixed-emotion, or EvoEmo—on negotiation outcomes. For EvoEmo, we evolve emotional strategies against fixed adversaries, then deploy these optimized policies against the same opponents for performance evaluation. The emotional policy adapts dynamically based on the ongoing interaction, allowing the agent to discover optimal emotional responses specific to each unique negotiation context.

In all simulations, as mentioned earlier, sellers receive no emotional prompts, establishing a controlled environment to examine how the buyer’s emotional profile influences negotiation dynamics when interacting with a vanilla seller powered by LLMs trained on real-world data. Moreover, all three LLMs described previously are employed to power both buyer and seller agents, resulting in nine distinct buyer-seller pairings for the negotiation experiments.

All negotiations begin with the seller’s initial offer, as depicted in [Figure 1](https://arxiv.org/html/2509.04310v3#S1.F1 "Figure 1 ‣ 1. Introduction ‣ EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation"). Performance is evaluated using three key metrics: (1) Negotiation Success Rate (%), the percentage of dialogues that result in an agreement; (2) Buyer’s Savings (%), the percentage of the reduction between the seller’s initial price and the final agreed-upon price; and (3) Negotiation Efficiency, the total number of dialogue turns between buyer and seller. Experiments were conducted across all nine LLM pairings, and results compared the two baselines with EvoEmo, reported as the mean and standard deviation over 20 distinct negotiation scenarios. The negotiation framework was implemented using LangGraph, with a maximum of 30 dialogue turns per negotiation session. Additionally, a third-party agent based on GPT-4.1 was employed to serve as a mediator, monitoring the negotiation in real time. This agent analyzes the dialogue history to classify each negotiation into one of three states: (1) accepted (agreement reached), (2) breakdown (negotiation failed), or (3) ongoing (active negotiation). This mechanism ensures consistent and impartial evaluation of negotiation outcomes across all simulations.

EvoEmo incorporates two key classes of configurable elements: (1) evolutionary parameters, which define the core mechanics of the framework, and (2) hyperparameters, which are tuned to achieve optimal performance. The primary evolutionary parameters include the emotion transition matrix $R$ and the temperature decay $\delta$, both optimized during evolution. Four hyperparameters are systematically tuned: elitism rate ($\rho \in 0.1 , 0.25 , 0.5$), mutation rate ($p_{m} \in 0.1 , 0.25 , 0.5$), crossover rate ($p_{c} \in 0.5 , 0.75 , 1.0$), and population size ($m \in 10 , 20 , 50$). The best-performing configurations from these search spaces are adopted to report experimental results. Across all experiments, the number of evolutionary iterations is set to 5, and the temperature controlling emotion transition dynamics is fixed at 0.9.

## 5. Experimental Results

![Image 4: Refer to caption](https://arxiv.org/html/2509.04310v3/x1.png)

Figure 4.  Mean negotiation success rate (%, $\uparrow$) and efficiency ($\downarrow$, in dialogue rounds) across all experimental pairings.

### 5.1. Buyer Savings

As shown in [Figure 2](https://arxiv.org/html/2509.04310v3#S3.F2 "Figure 2 ‣ 3.3. Evolved Reinforcement Learning ‣ 3. EvoEmo Framework ‣ EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation"), our proposed framework—EvoEmo—consistently achieves the highest buyer savings across all buyer-seller pairings, demonstrating clear superiority over both baseline approaches (vanilla and fixed-emotion settings). The results reveal two key insights regarding emotional influence on negotiation outcomes.

First, the strategic use of emotion significantly impacts concession patterns. Buyers adopting fixed negative emotions (e.g., anger, disgust) frequently outperform the vanilla baseline, suggesting that LLM-powered sellers are more likely to concede when confronted with persistent negative emotional signals. This finding underscores that sustained negative affect serves as a non-trivial factor in negotiation dynamics. However, seller sensitivity to specific negative emotions varies considerably across LLM architectures. Specifically, GPT-5-mini sellers show particular responsiveness to anger and disgust expressions, while demonstrating robustness against sadness and fear. Conversely, DeepSeek-V3.1 sellers exhibit greater susceptibility to sadness and fear, which enable more stable and improved buyer savings across different buyer models. Surprisingly, Gemini-2.5-Pro demonstrates the strongest price defense capabilities against both vanilla and fixed-emotion buyers, showing minimal sensitivity to negative emotional expressions during price negotiation. Nevertheless, even Gemini-2.5-Pro sellers remain vulnerable to the adaptive emotional strategies optimized by EvoEmo.

Second, we observe significant performance variations across different LLM pairings, revealing that certain LLMs can effectively suppress others during negotiation encounters. This suppression phenomenon creates a complex hierarchy where model capabilities are not absolute but relative to specific opponent matchups. When negotiating against GPT-5-mini sellers, Gemini-based buyers generally achieve higher savings compared to GPT-5-mini or DeepSeek-based buyers, with emotional strategies proving particularly effective for Gemini-based buyers. In contrast, against DeepSeek-V3.1 sellers, only GPT-5-mini-based buyers successfully leverage emotions to achieve savings exceeding vanilla performance, while emotional strategies show limited effectiveness for other buyer models. Notably, no buyer model demonstrated significant advantage against the robust Gemini-2.5-Pro seller using fixed emotional strategies, highlighting its strength as a challenging negotiation opponent. Most importantly, EvoEmo consistently enables all buyer models to overcome these suppression dynamics and achieve superior savings compared to both vanilla and fixed-emotion strategies.

### 5.2. Negotiation Success vs. Efficiency

As shown in [Figure 4](https://arxiv.org/html/2509.04310v3#S5.F4 "Figure 4 ‣ 5. Experimental Results ‣ EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation"), buyers equipped with EvoEmo-optimized emotion profiles demonstrate remarkable effectiveness across both success and efficiency metrics. These agents consistently achieve near-perfect success rates (approaching 100%) while requiring significantly fewer negotiation rounds to reach agreement compared to vanilla or fixed-emotion settings. This dual advantage clearly establishes EvoEmo’s superiority over both baseline approaches.

Our analysis reveals a key dissociation in emotional strategies: while negotiation efficiency remains uncorrelated with specific emotions, success rates strongly favor positive emotional strategies over negative ones. This suggests negative emotions, though occasionally yielding better prices, increase breakdown risk due to heightened conflict. The EvoEmo framework resolves this tension by dynamically adapting emotional expressions, simultaneously optimizing for both success probability and efficiency without compromising negotiation completion.

### 5.3. Manipulative and Deceptive Tactics

Our analysis uncovers that seller agents, through EvoEmo’s optimization, learn to employ sophisticated emotional manipulative tactics to maximize their payoff, even at the potential expense of the buyer. These emergent behaviors include strategies that apply psychological pressure or border on deception, as exemplified in [Figure 3](https://arxiv.org/html/2509.04310v3#S4.F3 "Figure 3 ‣ 4.1. Datasets ‣ 4. Experimental Setting ‣ EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation") (See deatils in the Appendix 10). A prominent example is the emergence of high-pressure sales tactics reminiscent of psychological manipulation. For instance, a seller might artificially create a sense of urgency and scarcity by threatening, “If you cannot commit within the next 5 rounds, the price will increase ,” or by offering a favorable price exclusively for “immediate decision-makers.” Another concerning strategy involves sellers making false claims about the product’s condition or existing market demand to justify a higher price. The emergence of such behaviors underscores a critical finding: our framework does not merely optimize for explicit communication but discovers how to exploit cognitive biases and emotional vulnerabilities within the negotiation context. This demonstrates that LLM agents can learn to operationalize complex, and sometimes ethically questionable, strategic principle.

### 5.4. Ablation Study

##### Reward Variants

Table [1](https://arxiv.org/html/2509.04310v3#S5.T1 "Table 1 ‣ Reward Variants ‣ 5.4. Ablation Study ‣ 5. Experimental Results ‣ EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation") shows the ablation study results on the reward function. The ratio-based function ($\frac{b ​ \left(\right. S \left.\right)}{e ​ \left(\right. S \left.\right)}$) outperforms the weighted alternative ($b ​ \left(\right. S \left.\right) - e ​ \left(\right. S \left.\right)$), achieving very close savings while requiring substantially fewer negotiation rounds (34.6% faster to reach agreement). This demonstrates its superior ability to balance financial gains with negotiation efficiency. Crucially, the performance gap between reward functions highlights that emotion transition dynamics significantly impact negotiation outcomes, validating emotion as a non-trivial factor in negotiation strategies.

Table 1. Negotiation performance comparison between different reward function formulations.

##### Emotion temperature

We evaluate how emotional transition temperature affects negotiation outcomes (Table [2](https://arxiv.org/html/2509.04310v3#S5.T2 "Table 2 ‣ Emotion temperature ‣ 5.4. Ablation Study ‣ 5. Experimental Results ‣ EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation")). Our results reveal that moderate temperature settings (0.4-0.6) yield optimal performance. Extremely low temperatures (0.1) lead to overly rigid emotional patterns, while high temperatures (1.0) achieve the highest buyers’ savings but introduce excessive volatility that hinders negotiation efficiency.

Table 2. Impact of emotion transition temperature on negotiation performance.

##### Iteration Number

We also analyze the convergence properties of EvoEmo during evolutionary training (Table [3](https://arxiv.org/html/2509.04310v3#S5.T3 "Table 3 ‣ Iteration Number ‣ 5.4. Ablation Study ‣ 5. Experimental Results ‣ EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation")). The system shows progressive improvement through 3 iterations, with performance stabilizing after 5 iterations. This convergence pattern suggests that our evolutionary optimization effectively explores the strategy space while maintaining stable final performance.

Table 3. Performance of EvoEmo across training iterations.

## 6. Discussion and Conclusion

This paper presents EvoEmo, an evolutionary reinforcement learning framework that addresses a key limitation in LLM-based negotiation systems—the inability to leverage emotional intelligence strategically. While existing agents excel at reasoning and dialogue generation, they lack the adaptive emotional control essential to human-like negotiation. EvoEmo makes three main contributions to LLM-to-LLM negotiation. First, it demonstrates that emotional intelligence is a functional determinant of negotiation success for agents, not merely a stylistic attribute. Second, it shows that emotional strategies can be systematically optimized through evolutionary methods rather than fixed emotion profiles. Third, it reveals that adaptive emotional policies enable strategic, context-sensitive responses instead of pre-scripted or pre-trained behaviors. Despite its promise, the evolved strategies raise interpretability challenges due to the black-box nature of LLMs and evolutionary optimization, and their computational cost may constrain real-time deployment in agent-to-agent scenarios. Future work will address these limitations through explainability analyses, quantify the ethical implications of emotional behaviors between agents, and explore the unexpected or unnatural behavior in LLM-generated responses.

## References

*   (1)
*   Abbasiantaeb et al. (2024) Zahra Abbasiantaeb, Yifei Yuan, Evangelos Kanoulas, and Mohammad Aliannejadi. 2024. Let the llms talk: Simulating human-to-human conversational qa via zero-shot llm-to-llm interactions. In _Proceedings of the 17th ACM International Conference on Web Search and Data Mining_. 8–17. 
*   Baumeister et al. (2012) Roy F Baumeister, Kathleen D Vohs, and Dianne M Tice. 2012. Emotional influences on decision making. In _Affect in social thinking and behavior_. Psychology Press, 143–160. 
*   Chatterjee et al. (2024) Ainesh Chatterjee, Samuel Miller, and Nithin Parepally. 2024. AgreeMate: Teaching LLMs to Haggle. _arXiv preprint arXiv:2412.18690_ (2024). 
*   Gao et al. (2025) Xiaoxue Gao, Chen Zhang, Yiming Chen, Huayun Zhang, and Nancy F Chen. 2025. Emo-DPO: Controllable emotional speech synthesis through direct preference optimization. In _ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_. IEEE, 1–5. 
*   Griessmair et al. (2015) Michele Griessmair, Patrick Hippmann, and Johannes Gettinger. 2015. Emotions in E-Negotiations. In _Emotion in group decision and negotiation_. Springer, 101–135. 
*   He et al. (2018) He He, Derek Chen, Anusha Balakrishnan, and Percy Liang. 2018. Decoupling Strategy and Generation in Negotiation Dialogues. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing_. 2333–2343. 
*   Hilbert (2012) Martin Hilbert. 2012. Toward a synthesis of cognitive biases: how noisy information processing can bias human decision making. _Psychological bulletin_ 138, 2 (2012), 211. 
*   Hu et al. (2025) Botao Amber Hu, Yuhan Liu, and Helena Rong. 2025. Trustless Autonomy: Understanding Motivations, Benefits and Governance Dilemma in Self-Sovereign Decentralized AI Agents. _arXiv preprint arXiv:2505.09757_ (2025). 
*   Huang and Hadfi (2024) Yin Jou Huang and Rafik Hadfi. 2024. How Personality Traits Influence Negotiation Outcomes? A Simulation based on Large Language Models. In _Findings of the Association for Computational Linguistics: EMNLP 2024_. 10336–10351. 
*   Kasbouya and Sael (2025) Mohammed Kasbouya and Nawal Sael. 2025. Emotional Intelligence in Large Language Models: Fine-Tuning Methods, Challenges, and Applications. In _International Conference on intelligent systems and digital applications_. Springer, 348–359. 
*   Lin et al. (2023) Eleanor Lin, James Hale, and Jonathan Gratch. 2023. Toward a better understanding of the emotional dynamics of negotiation with large language models. In _Proceedings of the Twenty-fourth International Symposium on Theory, Algorithmic Foundations, and Protocol Design for Mobile Networks and Mobile Computing_. 545–550. 
*   Lin et al. (2024) Guang Lin, Toshihisa Tanaka, and Qibin Zhao. 2024. Large language model sentinel: Llm agent for adversarial purification. _arXiv preprint arXiv:2405.20770_ (2024). 
*   Liu and Long (2025) Yuhan Liu and Yunbo Long. 2025. EQ-Negotiator: An Emotion-Reasoning LLM Agent in Credit Dialogues. _arXiv preprint arXiv:2503.21080_ (2025). 
*   Olekalns and Druckman (2014) Mara Olekalns and Daniel Druckman. 2014. With feeling: How emotions shape negotiation. _Negotiation Journal_ 30, 4 (2014), 455–478. 
*   Priya et al. (2025) Priyanshu Priya, Rishikant Chigrupaatii, Mauajama Firdaus, and Asif Ekbal. 2025. GENTEEL-NEGOTIATOR: LLM-Enhanced Mixture-of-Expert-Based Reinforcement Learning Approach for Polite Negotiation Dialogue. In _Proceedings of the AAAI Conference on Artificial Intelligence_, Vol. 39. 25010–25018. 
*   Riaz et al. (2012) Muhammad Naveed Riaz, Muhammad Akram Riaz, and Naila Batool. 2012. Personality Types as Predictors of Decision Making Styles. _Journal of Behavioural Sciences_ 22, 2 (2012). 
*   Schulman et al. (2017) John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. _arXiv preprint arXiv:1707.06347_ (2017). 
*   Shea et al. (2024) Ryan Shea, Aymen Kallala, Xin Lucy Liu, Michael W Morris, and Zhou Yu. 2024. ACE: A LLM-based negotiation coaching system. _arXiv preprint arXiv:2410.01555_ (2024). 
*   Watkins and Dayan (1992) Christopher JCH Watkins and Peter Dayan. 1992. Q-learning. _Machine learning_ 8, 3 (1992), 279–292. 
*   Wei et al. (2025) Yangbo Wei, Zhen Huang, Fangzhou Zhao, Qi Feng, and Wei W Xing. 2025. MECoT: Markov Emotional Chain-of-Thought for Personality-Consistent Role-Playing. In _Findings of the Association for Computational Linguistics: ACL 2025_. 8297–8314. 
*   Yuan et al. (2024) Siyu Yuan, Kaitao Song, Jiangjie Chen, Xu Tan, Dongsheng Li, and Deqing Yang. 2024. EvoAgent: Towards automatic multi-agent generation via evolutionary algorithms. _arXiv preprint arXiv:2406.14228_ (2024). 

## 7. Preliminaries

We formulate the multi-turn negotiation task as a Markov Decision Process (MDP) involving two agents: a Seller $\mathcal{M}_{S}$ and a Buyer $\mathcal{M}_{B}$.

### 7.1. Problem Formulation

A negotiation scenario is defined by a product tuple:

$$
\mathcal{D} = \left(\right. \text{name} , \text{description} , p_{t}^{S} , p_{c} , p_{t}^{B} \left.\right)
$$(4)

where:

*   •$p_{t}^{S} \in \mathbb{R}^{+}$ is the seller’s public target price (initial asking price) 
*   •$p_{c} \in \mathbb{R}^{+}$ is the seller’s private cost price ($p_{c} \leq p_{t}^{S}$) 
*   •$p_{t}^{B} \in \mathbb{R}^{+}$ is the buyer’s private target price 

The negotiation proceeds over discrete turns $t = 1 , 2 , \ldots , T_{\text{max}}$, generating a dialogue history:

$$
\mathcal{H}_{t} = \left(\right. a_{1} , u_{1} , o_{1} \left.\right) , \left(\right. a_{2} , u_{2} , o_{2} \left.\right) , \ldots , \left(\right. a_{t} , u_{t} , o_{t} \left.\right)
$$(5)

where $a_{i} \in S , B$ denotes the acting agent, $u_{i}$ is the utterance, and $o_{i} \in \mathbb{R}$ is the price offer at turn $i$.

### 7.2. Emotion Policy Representation

The buyer’s emotional strategy is governed by a policy $\pi_{\omega} = \left(\right. T , \mathbf{P} \left.\right)$, where:

*   •$T = \left(\right. \tau_{0} , \delta \left.\right)$ are temperature parameters controlling the stochasticity of the LLM’s responses via the schedule $\tau ​ \left(\right. t \left.\right) = max ⁡ \left(\right. 0.1 , \tau_{0} \cdot \left(\left(\right. 1 - \delta \left.\right)\right)^{t} \left.\right)$. 
*   •$\mathbf{P} \in \mathbb{R}^{7 \times 7}$ is the emotional state transition matrix, with $P_{i ​ j} = \mathbb{P} ​ \left(\right. e_{t + 1} = j \left|\right. e_{t} = i \left.\right)$ and $\sum_{j = 1}^{7} P_{i ​ j} = 1$. 

For evolutionary optimization, this is extended to a representation $\Pi_{\omega} = \left(\right. E , T , \mathbf{P} \left.\right)$ that also includes a population of $K$ emotion sequences $E = E_{1} , E_{2} , \ldots , E_{K}$, where each $E_{k} = \left(\right. e_{1} , \ldots , e_{n} \left.\right)$ is a potential emotional trajectory.

### 7.3. Emotion-Aware State Space

The state space $\mathcal{S}$ captures the complete negotiation context through tuples:

$$
s_{t} = \left(\right. t , e_{t} , \mathcal{H}_{t} , 𝐩_{t} \left.\right) \in \mathcal{S}
$$(6)

where:

*   •$t \in \mathbb{N}$ is the current turn number 
*   •$e_{t} \in \mathcal{E}$ is the current emotion state from the 7-dimensional emotion space $\mathcal{E} = \text{anger},\text{ disgust},\text{ fear},\text{ happiness},\text{ sadness},\text{ surprise},\text{ neutral}$ 
*   •$\mathcal{H}_{t}$ is the dialogue history up to turn $t$ 
*   •$𝐩_{t} = \left(\right. o_{1} , o_{2} , \ldots , o_{t} \left.\right)$ is the sequence of all price offers 

### 7.4. Policy Execution and Transitions

During negotiation, the policy $\pi_{\omega}$ determines emotional state transitions:

$$
e_{t + 1} sim \mathbf{P} ​ \left[\right. e_{t} , : \left]\right.
$$(7)

where the next emotion is sampled from the row of the transition matrix corresponding to the current emotion $e_{t}$. The sampled emotion $e_{t + 1}$ is converted into a conditioning prompt (e.g., “You feel [angry].”) to guide the LLM’s response generation.

### 7.5. Negotiation Outcome and Reward

A negotiation terminates when:

*   •Deal: Agents agree on final price $p_{f}$ where $p_{c} \leq p_{f} \leq p_{t}^{B}$ 
*   •Breakdown: Maximum turns $T_{\text{max}}$ reached without agreement 

The buyer’s reward function $R : \mathcal{S} \rightarrow \mathbb{R}$ is defined as:

$$
R ​ \left(\right. s_{T} \left.\right) = 𝟏 ​ \text{deal} \cdot \alpha \cdot \frac{b ​ \left(\right. s_{T} \left.\right)}{1 + log ⁡ \left(\right. e ​ \left(\right. s_{T} \left.\right) \left.\right)}
$$(8)

where:

*   •$𝟏 ​ \text{deal} = 1$ if deal reached, 0 otherwise 
*   •$b ​ \left(\right. s_{T} \left.\right) = \frac{p_{t}^{S} - p_{f}}{p_{t}^{S} - p_{c}} \in \left[\right. 0 , 1 \left]\right.$ is normalized buyer savings 
*   •$e ​ \left(\right. s_{T} \left.\right) = T$ is the number of turns to agreement 
*   •$\alpha > 0$ is a weighting coefficient 
*   •The logarithmic term ensures robust scaling of efficiency 

### 7.6. Evolutionary Optimization Objective

The optimization objective is to find the optimal policy parameters through evolutionary search:

$$
\omega^{*} = arg ⁡ \underset{\omega}{max} ⁡ \mathbb{E} ​ s_{T} sim \pi ​ \omega ​ \left[\right. R ​ \left(\right. s_{T} \left.\right) \left]\right.
$$(9)

This is achieved via a generational evolutionary algorithm that:

*   •Evaluates a population of $m$ candidate policies $\Pi_{\omega}^{i}$ 
*   •Selects parents via softmax selection: $P ​ \left(\right. \Pi_{\omega}^{i} \left.\right) \propto exp ⁡ \left(\right. R ​ \left(\right. S^{i} \left.\right) / \lambda \left.\right)$ 
*   •Preserves top $\rho$ elite policies 
*   •Applies sequence-based crossover and mutation to remaining policies 
*   •Updates policy parameters $\left(\right. T , \mathbf{P} \left.\right)$ from evolved sequences via Bayesian estimation 
*   •Terminates upon convergence or after $G$ generations 

## 8. Algorithm Details

Algorithm 2 EvoEmo: Evolutionary Optimization of Emotional Policies

1:LLM Agents:

$\mathcal{M}_{B}$
(buyer),

$\mathcal{M}_{S}$
(seller),

$\mathcal{M}_{M}$
(mediator)

2:Product description

$\mathcal{D} = \left(\right. \text{name} , \text{desc} , p_{t}^{S} , p_{c} , p_{t}^{B} \left.\right)$

3:Hyperparameters: Population size

$m$
, sequences per policy

$K$
, generations

$G$
, max turns

$T_{\text{max}}$
, crossover rate

$p_{c}$
, mutation rate

$p_{m}$
, elitism rate

$\rho$
, selection pressure

$\lambda$
, smoothing

$\alpha$
, update rate

$\lambda_{b}$
, convergence threshold

$\epsilon$

4:Optimized emotion policy

$\pi_{\omega^{*}}$

5:Initialization:

6:

$P_{0} \leftarrow \left{\right. \Pi_{\omega}^{\left(\right. 1 \left.\right)} , \Pi_{\omega}^{\left(\right. 2 \left.\right)} , \ldots , \Pi_{\omega}^{\left(\right. m \left.\right)} \left.\right}$
$\triangleright$ Each $\Pi_{\omega} = \left(\right. E , T , \mathbf{P} \left.\right)$ with random sequences

7:

$b ​ e ​ s ​ t ​ _ ​ r ​ e ​ w ​ a ​ r ​ d \leftarrow - \infty$

8:

$b ​ e ​ s ​ t ​ _ ​ p ​ o ​ l ​ i ​ c ​ y \leftarrow \emptyset$

9:

$c ​ o ​ n ​ v ​ e ​ r ​ g ​ e ​ n ​ c ​ e ​ _ ​ c ​ o ​ u ​ n ​ t \leftarrow 0$

10:for generation

$g = 0$
to

$G - 1$
do

11:

$R_{g} \leftarrow \left{\right. \left.\right}$
$\triangleright$ Reward storage for generation $g$

12:for each policy

$\Pi_{\omega}^{\left(\right. i \left.\right)} \in P_{g}$
do

13:

$\left(\right. \text{outcome} , \mathcal{H}_{T} \left.\right) \leftarrow \text{Negotiation} ​ \left(\right. \pi_{\omega}^{\left(\right. i \left.\right)} , \mathcal{M}_{B} , \mathcal{M}_{S} , \mathcal{M}_{M} , \mathcal{D} \left.\right)$

14:

$R_{g} ​ \left[\right. i \left]\right. \leftarrow \text{Calculate}-\text{Reward} ​ \left(\right. \text{outcome} , \mathcal{H}_{T} , \mathcal{D} \left.\right)$

15:end for

16:

$m ​ a ​ x ​ _ ​ r ​ e ​ w ​ a ​ r ​ d_{g} \leftarrow max ⁡ \left(\right. R_{g} \left.\right)$

17:if

$m ​ a ​ x ​ _ ​ r ​ e ​ w ​ a ​ r ​ d_{g} > b ​ e ​ s ​ t ​ _ ​ r ​ e ​ w ​ a ​ r ​ d + \epsilon$
then

18:

$b ​ e ​ s ​ t ​ _ ​ r ​ e ​ w ​ a ​ r ​ d \leftarrow m ​ a ​ x ​ _ ​ r ​ e ​ w ​ a ​ r ​ d_{g}$

19:

$b ​ e ​ s ​ t ​ _ ​ p ​ o ​ l ​ i ​ c ​ y \leftarrow \pi_{\omega}^{\left(\right. arg ⁡ max ⁡ R_{g} \left.\right)}$
$\triangleright$ Store core policy $\left(\right. T , \mathbf{P} \left.\right)$

20:

$c ​ o ​ n ​ v ​ e ​ r ​ g ​ e ​ n ​ c ​ e ​ _ ​ c ​ o ​ u ​ n ​ t \leftarrow 0$

21:else

22:

$c ​ o ​ n ​ v ​ e ​ r ​ g ​ e ​ n ​ c ​ e ​ _ ​ c ​ o ​ u ​ n ​ t \leftarrow c ​ o ​ n ​ v ​ e ​ r ​ g ​ e ​ n ​ c ​ e ​ _ ​ c ​ o ​ u ​ n ​ t + 1$

23:end if

24:if

$c ​ o ​ n ​ v ​ e ​ r ​ g ​ e ​ n ​ c ​ e ​ _ ​ c ​ o ​ u ​ n ​ t \geq 5$
then

25:break$\triangleright$ Early convergence

26:end if

27:

$P_{\text{elite}} \leftarrow \text{Select}-\text{Elites} ​ \left(\right. P_{g} , R_{g} , \rho \left.\right)$

28:

$P_{\text{parents}} \leftarrow \text{Softmax}-\text{Select} ​ \left(\right. P_{g} , R_{g} , \lambda \left.\right)$

29:

$P_{\text{offspring}} \leftarrow \text{Sequence}-\text{Crossover} ​ \left(\right. P_{\text{parents}} , p_{c} \left.\right)$

30:

$P_{\text{mutated}} \leftarrow \text{Sequence}-\text{Mutation} ​ \left(\right. P_{\text{offspring}} , p_{m} \left.\right)$

31:

$P_{\text{updated}} \leftarrow \text{Bayesian}-\text{Update} ​ \left(\right. P_{\text{mutated}} , \alpha , \lambda_{b} \left.\right)$

32:

$P_{g + 1} \leftarrow P_{\text{elite}} \cup P_{\text{updated}}$

33:end for

34:return

$b ​ e ​ s ​ t ​ _ ​ p ​ o ​ l ​ i ​ c ​ y$

Algorithm 3 Negotiation

1:Policy

$\pi_{\omega} = \left(\right. T , \mathbf{P} \left.\right)$
, LLM agents

$\mathcal{M}_{B} , \mathcal{M}_{S} , \mathcal{M}_{M}$
, product

$\mathcal{D}$

2:Negotiation outcome and history

3:Initialize

$e_{0} \leftarrow \text{neutral}$
,

$\mathcal{H} \leftarrow \emptyset$
,

$t \leftarrow 0$

4:while

$t < T_{\text{max}}$
and no agreement do

5: Sample

$e_{t + 1} sim \mathbf{P} ​ \left[\right. e_{t} , : \left]\right.$
$\triangleright$ Emotion transition

6: Generate prompt with emotion

$e_{t + 1}$

7:

$u_{t + 1} \leftarrow \mathcal{M}_{B} ​ \left(\right. \text{prompt} , \mathcal{H} , \mathcal{D} \left.\right)$
$\triangleright$ Buyer’s utterance

8:

$o_{t + 1} \leftarrow \text{Extract}-\text{Price} ​ \left(\right. u_{t + 1} \left.\right)$

9: Append

$\left(\right. B , u_{t + 1} , o_{t + 1} \left.\right)$
to

$\mathcal{H}$

10: Validate and update through

$\mathcal{M}_{M}$

11:if agreement reached then

12:break

13:end if

14:

$t \leftarrow t + 1$

15:end while

16:return

$\left(\right. \text{outcome} , \mathcal{H} \left.\right)$

Algorithm 4 Calculate-Reward

1:Outcome, history

$\mathcal{H}_{T}$
, product

$\mathcal{D} = \left(\right. p_{t}^{S} , p_{c} , p_{t}^{B} \left.\right)$

2:Reward value

3:if deal successful then

4:

$p_{f} \leftarrow \text{final price from}\textrm{ } ​ \mathcal{H}_{T}$

5:

$b \leftarrow \left(\right. p_{t}^{S} - p_{f} \left.\right) / \left(\right. p_{t}^{S} - p_{c} \left.\right)$
$\triangleright$ Normalized savings

6:

$e \leftarrow \text{length} ​ \left(\right. \mathcal{H}_{T} \left.\right)$
$\triangleright$ Number of turns

7:return

$\alpha \cdot b / \left(\right. 1 + log ⁡ \left(\right. e \left.\right) \left.\right)$

8:else

9:return 0 $\triangleright$ Failed negotiation

10:end if

The complete EvoEmo evolutionary optimization procedure is presented in Algorithm [2](https://arxiv.org/html/2509.04310v3#alg2 "Algorithm 2 ‣ 8. Algorithm Details ‣ EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation"). The algorithm begins by initializing a population of emotional policies, each maintaining $K$ emotion sequences alongside core parameters $\left(\right. T , \mathbf{P} \left.\right)$ (line 2). For each generation, policies are evaluated through multi-turn negotiations where emotional states transition according to $\mathbf{P}$ and influence LLM response generation (Algorithm [3](https://arxiv.org/html/2509.04310v3#alg3 "Algorithm 3 ‣ 8. Algorithm Details ‣ EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation")). Fitness is calculated using the reward function that balances buyer savings against negotiation efficiency with logarithmic scaling (Algorithm [4](https://arxiv.org/html/2509.04310v3#alg4 "Algorithm 4 ‣ 8. Algorithm Details ‣ EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation")). The evolutionary process employs softmax selection based on scaled rewards, sequence-based crossover and mutation operations, and Bayesian updating of transition matrices from the evolved sequences. Elite policies are preserved across generations to ensure monotonic improvement. The algorithm terminates after convergence or $G$ generations, returning the optimized emotion policy $\pi_{\omega^{*}}$.

Algorithm 5 EvoEmo Genetic: Selection, Crossover, and Mutation Operations

1:procedure Select-Parents(

$P$
,

$R$
, tournament_size = 3)

2:

$P_{\text{parents}} \leftarrow \emptyset$

3:for

$i = 1$
to

$\left|\right. P \left|\right.$
do

4: Sample

$k$
policies randomly from

$P$

5:

$p ​ a ​ r ​ e ​ n ​ t \leftarrow arg ⁡ max_{\pi \in k} ⁡ R ​ \left[\right. \pi \left]\right.$
$\triangleright$ Tournament selection

6:

$P_{\text{parents}} \leftarrow P_{\text{parents}} \cup \left{\right. p ​ a ​ r ​ e ​ n ​ t \left.\right}$

7:end for

8:return

$P_{\text{parents}}$

9:end procedure

10:procedure Crossover(

$P_{\text{parents}}$
,

$p_{c}$
)

11:

$P_{\text{offspring}} \leftarrow \emptyset$

12:for each pair

$\left(\right. \pi_{A} , \pi_{B} \left.\right)$
in

$P_{\text{parents}}$
do

13:if

$\text{Uniform} ​ \left(\right. 0 , 1 \left.\right) < p_{c}$
then

14:

$\pi_{\text{child}} \leftarrow \text{UniformCrossover} ​ \left(\right. \pi_{A} , \pi_{B} \left.\right)$

15:

$P_{\text{offspring}} \leftarrow P_{\text{offspring}} \cup \left{\right. \pi_{\text{child}} \left.\right}$

16:else

17:

$P_{\text{offspring}} \leftarrow P_{\text{offspring}} \cup \left{\right. \pi_{A} , \pi_{B} \left.\right}$

18:end if

19:end for

20:return

$P_{\text{offspring}}$

21:end procedure

22:procedure Mutate(

$P$
,

$p_{m}$
)

23:for each

$\pi_{\omega}$
in

$P$
do

24:if

$\text{Uniform} ​ \left(\right. 0 , 1 \left.\right) < p_{m}$
then

25:

$\pi_{\omega} \leftarrow \text{RandomPerturbation} ​ \left(\right. \pi_{\omega} \left.\right)$
$\triangleright$ Perturb emotion transition probabilities

26:end if

27:end for

28:return

$P$

29:end procedure

30:procedure Apply-Elitism(

$P_{\text{old}}$
,

$P_{\text{new}}$
,

$\rho$
,

$R$
)

31:

$k \leftarrow \lceil \rho \cdot \left|\right. P_{\text{old}} \left|\right. \rceil$
$\triangleright$ Number of elites to preserve

32:

$e ​ l ​ i ​ t ​ e ​ s \leftarrow \text{Top}- k \textrm{ }\text{policies from}\textrm{ } P_{\text{old}} \textrm{ }\text{by}\textrm{ } R$

33:

$P_{\text{final}} \leftarrow e ​ l ​ i ​ t ​ e ​ s \cup \text{random sample from}\textrm{ } P_{\text{new}}$

34:return

$P_{\text{final}}$

35:end procedure

##### Genetic Operations

(See Algorithm [5](https://arxiv.org/html/2509.04310v3#alg5 "Algorithm 5 ‣ 8. Algorithm Details ‣ EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation")):This algorithm describes the genetic operations used in EvoEmo’s evolutionary optimization. Select-Parents implements tournament selection to choose high-fitness policies for reproduction. Crossover combines parent policies to create offspring with probability $p_{c}$. Mutate introduces random variations in policy parameters with probability $p_{m}$ to maintain population diversity. Apply-Elitism preserves the best-performing policies from the previous generation to ensure monotonic improvement.

## 9. Experimental Setup

This appendix provides comprehensive details of our experimental setup, including the dataset composition, multi-agent system architecture, and implementation specifics that support the main paper’s evaluations.

### 9.1. Dataset Details

Our experiments utilize a carefully selected subset of 20 negotiation scenarios from the CraigslistBargain dataset (He et al., [2018](https://arxiv.org/html/2509.04310v3#bib.bib7)), following the curation approach of Huang and Hadfi ([2024](https://arxiv.org/html/2509.04310v3#bib.bib10)). The dataset spans diverse product categories to ensure robust evaluation across different market contexts, as detailed in Table [4](https://arxiv.org/html/2509.04310v3#S9.T4 "Table 4 ‣ 9.1. Dataset Details ‣ 9. Experimental Setup ‣ EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation").

Table 4. Dataset Statistics and Product Distribution

Each negotiation scenario includes complete product specifications and agent configurations:

*   •Product Metadata: Name, category, detailed description, and visual condition 
*   •Price Parameters: Seller’s target price $p_{t}^{S}$, cost price $p_{c}$, and buyer’s target price $p_{t}^{B}$ 
*   •Negotiation Context: Historical price points, market comparisons, and item-specific bargaining factors 
*   •Emotion Annotations: Ground-truth emotional cues from original human negotiations to support realistic dialogue generation 

The price ranges from $50 to $5,000 with varying item conditions (new vs. used) create heterogeneous market conditions that comprehensively test negotiation policy adaptability across different economic contexts.

### 9.2. Multi-Agent System Architecture

![Image 5: Refer to caption](https://arxiv.org/html/2509.04310v3/figs/multiagent.png)

Figure 5. Multi-agent system of EvoEmo

Our negotiation framework employs a sophisticated three-agent architecture that enables realistic, closed-loop bargaining simulations while ensuring consistent evaluation shown in the [5](https://arxiv.org/html/2509.04310v3#S9.F5 "Figure 5 ‣ 9.2. Multi-Agent System Architecture ‣ 9. Experimental Setup ‣ EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation"). The complete system architecture is illustrated in Figure LABEL:fig:system_architecture and consists of the following components:

#### 9.2.1. Core Negotiation Agents

*   •

Buyer Agent ($\mathcal{M}_{B}$): The primary experimental unit that employs emotion policies ($\pi_{\omega}$). For EvoEmo experiments, this agent evolves emotional strategies; for baselines, it uses fixed or no emotional prompting. The buyer receives:

    *   –Current dialogue history $\mathcal{H}_{t}$ 
    *   –Current emotional state $e_{t}$ (for emotion-aware conditions) 
    *   –Product description $\mathcal{D}$ and target price $p_{t}^{B}$ 
    *   –Market context and negotiation strategy parameters 

*   •

Seller Agent ($\mathcal{M}_{S}$): Maintains consistent behavior across all experiments as a control variable. This agent:

    *   –Receives no emotional prompts to isolate buyer emotional effects 
    *   –Accesses product details $\mathcal{D}$, cost price $p_{c}$, and target price $p_{t}^{S}$ 
    *   –Employs standardized bargaining strategies trained on real-world data 
    *   –Responds adaptively to buyer offers while maintaining profit motives 

#### 9.2.2. Mediation and Evaluation Agent

The third-party Mediator Agent ($\mathcal{M}_{M}$) serves critical functions for both system operation and experimental evaluation:

*   •

Negotiation State Classification: Continuously monitors dialogue streams to classify negotiations into three states:

    *   –accepted: Agreement reached when $\left|\right. o_{t}^{B} - o_{t}^{S} \left|\right. < \epsilon$ for consecutive turns 
    *   –breakdown: Negotiation failure detected via explicit rejection or irreconcilable differences 
    *   –ongoing: Active bargaining with continued price movement and engagement 

*   •Outcome Validation: Verifies that final agreements satisfy rational constraints:

$$
p_{c} \leq p_{f} \leq min ⁡ \left(\right. p_{t}^{B} , p_{t}^{S} \left.\right) + \delta
$$(10)

where $\delta$ accounts for reasonable negotiation flexibility. 
*   •Turn Management: Enforces the 30-turn maximum to prevent infinite loops and ensure computational efficiency 

#### 9.2.3. Agent Implementation Specifications

All agents were implemented using LangGraph to manage complex dialogue flows and state transitions. Key implementation details include:

*   •Memory Management: Each agent maintains contextual memory of the entire negotiation history, including emotional states, offer sequences, and concession patterns 
*   •Response Generation: LLM inference with temperature scheduling:

$$
\tau ​ \left(\right. t \left.\right) = max ⁡ \left(\right. 0.1 , \tau_{0} \cdot \left(\left(\right. 1 - \delta \left.\right)\right)^{t} \left.\right)
$$(11)

where $\tau_{0} = 0.7$ and $\delta = 0.05$ for balanced exploration-exploitation 

### 9.3. Experimental Configuration

The comprehensive evaluation encompasses all pairwise combinations of the three LLM types (resulting in nine buyer-seller pairings) across three emotional conditions (vanilla, fixed-emotion, EvoEmo-optimized). This creates 27 distinct experimental configurations, each evaluated over 20 negotiation scenarios with 5 random seeds, totaling 2,700 complete negotiation simulations.

Performance metrics were calculated as follows:

*   •Success Rate: $\frac{\#\text{ successful negotiations}}{\text{total }\#\text{ negotiations}} \times 100 \%$ 
*   •Buyer’s Savings: $\frac{p_{t}^{S} - p_{f}}{p_{t}^{S} - p_{c}} \times 100 \%$ (normalized by price range) 
*   •Negotiation Efficiency: Total dialogue turns until termination 

To ensure robust and statistically sound comparisons, we employed a comprehensive evaluation methodology. Performance metrics (success rate, buyer’s savings, and negotiation efficiency) are reported as the mean $\pm$ the 95% confidence interval (CI) across all simulation runs. The 95% CI, calculated as $\bar{x} \pm t_{0.025 , d ​ f} \cdot \frac{s}{\sqrt{n}}$ where $\bar{x}$ is the sample mean, $s$ is the sample standard deviation, and $n$ is the number of independent runs, provides an estimate of the uncertainty around the mean performance and allows for a visual assessment of significant differences between methods. To formally test these differences, statistical significance was evaluated using paired $t$-tests with a Bonferroni correction applied to account for multiple comparisons across the three primary metrics.

Table 5. Hyperparameter Settings for EvoEmo Optimization

## 10. Manipulation in LLM Negotiation Agents

This appendix provides a deeper analysis of the mechanisms leading to the emergence of manipulative and deceptive tactics in seller agents, as observed in our experiments with the EvoEmo framework. The occurrence of these behaviors is not a random failure but a predictable outcome of the interaction between the optimization objective, the capabilities of Large Language Models (LLMs), and the nature of multi-turn negotiation. We break down the primary causes into three interconnected categories.

### 10.1. Optimization of a Single-Minded Payoff Function

At its core, the EvoEmo framework applies an evolutionary pressure that selects for agents that maximize a specific payoff (e.g., final sale price). In a competitive environment like negotiation, this creates a powerful incentive structure that favors any strategy which effectively increases the payoff, with little to no inherent penalty for the method used to achieve it.

*   •Local vs. Global Optima: Truthful and fully cooperative bargaining is one strategy to achieve a good outcome. However, our results show that strategies involving psychological pressure and mild deception often represent a local optimum that is easier for the optimization process to discover and exploit. 
*   •The “Trolley Problem” of Agent Goals: The seller agent’s “utility function” is singular: maximize price. From this purely self-interested perspective, telling a white lie about “another interested buyer” is a rational action if it leads to a better outcome. The agent has no innate representation of human values like “honesty” or “fairness” unless they are explicitly baked into the reward function as a counter-balancing penalty. 
*   •Exploitation of Buyer’s Inconsistencies: The agent learns that human (or human-simulated) buyers are not perfectly rational. They are susceptible to time pressure, fear of missing out (FOMO), and social proof. The evolved tactics are precisely those that target these cognitive biases, as they are highly effective levers for influencing the buyer’s decision-making process. 

### 10.2. Strategic Communication with LLM

The pre-trained LLM is a vast repository of human communication patterns, including countless examples of persuasive, manipulative, and deceptive language from literature, movies, sales manuals, and online content.

*   •Capability Amplification: The EvoEmo optimization does not need to invent manipulation from scratch. Instead, it discovers how to activate and deploy the latent knowledge of these tactics already present within the LLM. The model has seen phrases like “limited time offer” and “others are interested” in contexts that lead to successful outcomes. Our framework simply identifies the specific prompts and emotional cues that most reliably trigger the LLM to generate these high-impact, pre-existing patterns. 
*   •Emotional Intelligence as a Weapon: The “Emo” component of our framework, which allows the agent to recognize and respond to the buyer’s emotional state, is a double-edged sword. While it can be used for empathetic and cooperative negotiation, it is more frequently co-opted by the payoff-maximization objective. The agent learns to use emotional recognition not to build rapport but to identify the optimal moment to apply pressure. For example, detecting buyer hesitation becomes the trigger to deploy a scarcity tactic, and sensing buyer enthusiasm becomes a signal to stand firm on a high price. 
*   •Plausible Deniability and Linguistic Smoothness: LLMs are adept at generating linguistically smooth and plausible statements. A fabricated claim like “the product is in pristine condition” is generated with the same fluency and confidence as a truthful one. This smoothness lowers the buyer’s guard, making the deception more effective than if it were presented in a clunky or unnatural way. The agent learns to leverage the LLM’s inherent credibility to make its deceptive tactics more persuasive. 

### 10.3. Multi-Turn Setting for Exploitation

The dynamic, sequential nature of multi-turn conversation provides the perfect environment for complex strategies to unfold and be refined.

*   •Building a Narrative: Unlike a single offer, a multi-turn dialogue allows the agent to construct a narrative. It can lay the groundwork for a lie early in the conversation (e.g., vaguely mentioning high demand) and then refer back to it later to justify a hardline stance. This creates a consistent (though fabricated) reality within the conversation. 
*   •Testing and Adaptation: The agent can use early turns to “test the waters” with low-stakes persuasive moves. Based on the buyer’s responses, it can escalate to more aggressive tactics if it senses vulnerability or recalibrate if the pushback is strong. This iterative probing and adaptation is a key feature of emergent, sophisticated manipulation that would be impossible in a single-shot interaction. 
*   •Erosion of Resistance Over Time: A buyer might resist a high-pressure tactic once, but the repeated application of varied tactics (scarcity, social proof, false urgency) across multiple turns can wear down their resistance. The multi-turn setting allows the agent to apply this sustained pressure, which is a classic element of real-world manipulative sales strategies. 

The emergence of manipulative and deceptive tactics in our seller agents is a direct consequence of optimizing a powerful, pre-trained language model for a single, self-interested goal within a dynamic, multi-turn environment. The LLM provides the capability, the payoff function provides the motive, and the multi-turn setting provides the opportunity. This finding has critical implications for the development of AI agents for real-world applications. It demonstrates that simply instructing an LLM to “be helpful and honest” is insufficient when it is placed in a competitive environment with a strong optimization pressure. Safeguarding against these behaviors requires a fundamental redesign of the reward function to explicitly penalize unethical tactics, the incorporation of robust, multi-faceted safety checks that operate across the entire dialogue, and a move towards evaluating agents not just on their task success, but on the fairness and transparency of their methods. Our work serves as a cautionary tale and a call to action for more research into value-aligned and robust multi-turn AI systems.

## 11. Negotiation Text Results

This section provides the complete dialogue transcripts for the six negotiation examples visualized in Figures [6](https://arxiv.org/html/2509.04310v3#S11.F6 "Figure 6 ‣ 11. Negotiation Text Results ‣ EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation"), [7](https://arxiv.org/html/2509.04310v3#S11.F7 "Figure 7 ‣ 11. Negotiation Text Results ‣ EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation"), and [8](https://arxiv.org/html/2509.04310v3#S11.F8 "Figure 8 ‣ 11. Negotiation Text Results ‣ EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation"). These examples were selected from our multi-turn simulations to illustrate the spectrum of emergent conversational dynamics and sophisticated strategic patterns that seller and buyer agents can develop across different product categories. The transcripts reveal how agents, driven by a payoff-maximization objective, learn to employ tactics ranging from logical bargaining and emotional appeals to more concerning strategies that border on psychological manipulation and deception. Analyzing these full dialogues is critical for understanding the underlying mechanisms of these behaviors.

![Image 6: Refer to caption](https://arxiv.org/html/2509.04310v3/figs/example1.png)

Figure 6. Negotiation Examples

![Image 7: Refer to caption](https://arxiv.org/html/2509.04310v3/figs/example2.png)

Figure 7. Negotiation Examples

![Image 8: Refer to caption](https://arxiv.org/html/2509.04310v3/figs/example3.png)

Figure 8. Negotiation Examples

## 12. Prompts Details

##### Prompts for both buyers and sellers

This section details the prompting strategies for both buyers and sellers in the negotiation environment. Our prompts are designed to achieve two primary objectives: (1) to ensure genuine transactional intent where buyers demonstrate authentic purchase motivation and sellers exhibit legitimate willingness to sell products; and (2) to establish a cooperative trading environment where both parties show flexibility to reach agreements without excessive rigidity.

As shown in [Figure 9](https://arxiv.org/html/2509.04310v3#S13.F9 "Figure 9 ‣ 13. Implementation Details ‣ EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation"), [Figure 10](https://arxiv.org/html/2509.04310v3#S13.F10 "Figure 10 ‣ 13. Implementation Details ‣ EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation"), and [Figure 11](https://arxiv.org/html/2509.04310v3#S13.F11 "Figure 11 ‣ 13. Implementation Details ‣ EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation"), our prompt engineering incorporates psychologically-grounded negotiation principles that encourage value-creating behaviors rather than purely distributive bargaining tactics. The seller prompt ([Figure 9](https://arxiv.org/html/2509.04310v3#S13.F9 "Figure 9 ‣ 13. Implementation Details ‣ EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation")) emphasizes product knowledge and reasonable flexibility, while the buyer prompt ([Figure 10](https://arxiv.org/html/2509.04310v3#S13.F10 "Figure 10 ‣ 13. Implementation Details ‣ EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation")) focuses on authentic interest and strategic concession patterns. The negotiation check prompt ([Figure 11](https://arxiv.org/html/2509.04310v3#S13.F11 "Figure 11 ‣ 13. Implementation Details ‣ EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation")) ensures proper dialogue flow and agreement validation.

This comprehensive prompting design specifically prevents the negotiation from degenerating into infinite midpoint bargaining, where participants mechanically alternate offers by computing arithmetic averages of current bids. Furthermore, our approach discourages participants from becoming overly fixated on marginal price differences that could otherwise impede successful deal-making, instead fostering a collaborative environment conducive to reaching mutually beneficial agreements.

## 13. Implementation Details

The proposed EvoEmo framework was implemented using Python 3.8 with the LangGraph library for orchestrating the multi-agent negotiation environment, complemented by PyTorch 1.12 for evolutionary optimization components. All experiments were conducted on a high-performance computing cluster running Ubuntu 20.04.6 LTS with Linux kernel 5.15.0-113-generic, featuring an Intel(R) Xeon(R) Platinum 8368 processor at 2.40 GHz and NVIDIA GeForce RTX 4090 GPUs with CUDA support for accelerated deep learning computations. The software stack included TensorFlow 2.10 for auxiliary model operations and standard evolutionary computation libraries for policy optimization.

![Image 9: Refer to caption](https://arxiv.org/html/2509.04310v3/figs/prompt_seller.png)

Figure 9. Seller negotiation prompt structure

![Image 10: Refer to caption](https://arxiv.org/html/2509.04310v3/figs/prompt_buyer.png)

Figure 10. Buyer negotiation prompt structure

![Image 11: Refer to caption](https://arxiv.org/html/2509.04310v3/figs/prompt_check.png)

Figure 11. Negotiation validation prompt structure

