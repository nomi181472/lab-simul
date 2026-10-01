# Evolutionary System Prompt Learning for Reinforcement Learning in LLMs

Lunjun Zhang<sup>1</sup> Ryan Chen<sup>2,3</sup> Bradly C. Stadie<sup>2,3</sup>

## Abstract

Building agentic systems that can autonomously self-improve from experience is a longstanding goal of AI. Large language models (LLMs) today primarily self-improve via two mechanisms: self-reflection for context updates, and reinforcement learning (RL) for weight updates. In this work, we propose **Evolutionary System Prompt Learning (E-SPL)**, a method for jointly improving model contexts and model weights. In each RL iteration, E-SPL samples trajectories under multiple system prompts in parallel, then jointly applies RL updates to LLM weights and evolutionary updates to system prompts. System prompts evolve via mutation and crossover, two genetic operators driven by LLM self-reflection; selection is based on relative performance ratings updated across RL iterations. E-SPL encourages a natural division between declarative knowledge encoded in prompts and procedural knowledge encoded in weights, resulting in improved performance across reasoning and agentic tasks. For instance, in an easy-to-hard (AIME  $\rightarrow$  BeyondAIME) generalization setting, E-SPL improves RL success rate from 38.8%  $\rightarrow$  45.1% while also outperforming reflective prompt evolution (40.0%). Overall, our results demonstrate that RL and system prompt evolution are deeply synergistic, and combining the two yields consistent gains in sample efficiency and generalization. Code: [github.com/LunjunZhang/E-SPL](https://github.com/LunjunZhang/E-SPL)

## 1. Introduction

Evolutionary algorithms (EA) and reinforcement learning (RL) have long been studied as complementary paradigms for optimizing intelligent behaviors. Evolutionary methods excel at population-level exploration over discrete, high-

<sup>1</sup>Department of Computer Science, University of Toronto  
<sup>2</sup>Department of Statistics and Data Science, Northwestern University  
<sup>3</sup>Bridgewater AIA Labs. Correspondence to: Lunjun Zhang <lunjun@cs.toronto.edu>.

Preprint. February 26, 2026.

Figure 1. Evolving System Prompts helps RL generalize better.

level structures (Koza, 1994; Bäck et al., 1997; Lehman & Stanley, 2011a), while reinforcement learning is well suited for fine-grained optimization of policy parameters through trial and error (Williams, 1992; Sutton et al., 1999; Silver et al., 2016). Historically, however, these two approaches have largely operated on different representational levels: evolution over programs (Langdon & Poli, 2013) and architectures (Stanley & Miikkulainen, 2002) on the one hand, and gradient-based learning over a fixed set of parameters (Schulman et al., 2015; Mnih et al., 2015) on the other. Bridging these representational levels in a unified and scalable way remains an open challenge, particularly in the context of large language models (LLMs) (Brown et al., 2020; Achiam et al., 2023), where both the model weights (Jaech et al., 2024) and the model contexts (Wei et al., 2022; Zhou et al., 2022) critically shape behavior.

Recent work on prompt evolution and LLM agents demonstrates that prompt (or context) engineering is a powerful mechanism for improving test-time performance. Techniques such as self-reflection (Shinn et al., 2023; Madaan et al., 2023; Cai et al., 2025) and population-based search (Guo et al., 2023; Fernando et al., 2023; Romera-Paredes et al., 2024; Agrawal et al., 2025) show that prompts can encode substantial declarative knowledge and behavioral priors without modifying model weights. At the same time, RL has emerged as the predominant post-training approach to fundamentally improve reasoning capabilities (Jaech et al., 2024; Guo et al., 2025; Chen et al., 2025; Comanici et al.,RL sampling for each problem  $x$

Select  $s_1, \dots, s_M$

Population of System Prompts  $\{(s_i, (\mu_i, \sigma_i))\}$

Selection

Sample  $N$  rollouts per  $s_i$  from LLM  $\pi_\theta$

Compute rewards

Value estimate under each  $s_i$

Mutation

Crossover

Update TrueSkill Ratings

Evolution updates system prompts

Policy Gradient  $\nabla_\theta J(x) = \frac{1}{NM} \sum_{i=1}^M \sum_{j=1}^N (R(x, y_{i,j}) - V_i) \nabla_\theta \log \pi_\theta(y_{i,j} | s_i, x) \quad \theta \leftarrow \theta + \alpha \nabla_\theta J$

RL updates weights  $\theta$

Figure 2. **Evolutionary System Prompt Learning (E-SPL)** jointly optimizes model contexts and model weights to enhance LLM self-improvement. Evolution updates system prompts; RL updates weights. The learned system prompts can encode *declarative knowledge* via articulated principles and strategies, while RL gradients can further hone the model’s *procedural knowledge* for reliable execution.

2025). Despite their complementary strengths, existing approaches typically optimize model contexts and weights in isolation, leaving their potential synergies unexplored.

In this work, we argue that effective self-improving agents should jointly evolve their high-level behavioral specifications and their low-level parametric policies. This perspective motivates **Evolutionary System Prompt Learning (E-SPL)**, a framework that tightly couples reinforcement learning over model weights with evolutionary search over system prompts. By maintaining a population of system prompts that are evaluated, rated, and evolved alongside RL training, E-SPL enables structured exploration over declarative knowledge while encoding procedural knowledge into weights. This joint optimization yields a synergistic dynamics in which improved prompts accelerate learning, and improved weights amplify the benefits of prompt evolution.

Experimentally, E-SPL enhances RL performance of from 56.3%  $\rightarrow$  60.6% on AIME 2025, 50.0%  $\rightarrow$  52.7% on HMMT 2025 November test, and notably, 38.8%  $\rightarrow$  45.1% on AIME  $\rightarrow$  BeyondAIME in an easy-to-hard generalization setting, using a base model of DeepSeek v3.1 (DeepSeek-AI, 2024). E-SPL significantly outperforms prompt evolution as well, showing that weight updates are often necessary for improving the model’s tacit knowledge and intuition. On an agentic search task (Jin et al., 2025), we improved RL performance from 44.2%  $\rightarrow$  48.6% on OpenAI’s gpt-oss-120b (Agarwal et al., 2025) with limited data. Together, these results show that combining RL and system prompt evolution unlocks a synergistic form of self-improvement that neither approach achieves in isolation.

## 2. Related Work

**Prompt Evolution and Contextual Memory** A growing body of work treats prompts and contexts as optimizable objects that can encode high-level behaviors without modifying model weights. Evolutionary and search-based approaches such as EvoPrompt (Guo et al., 2023), PromptBreeder (Fernando et al., 2023), Automatic Prompt Engineer (Zhou et al., 2022), and FunSearch (Romera-Paredes et al., 2024) demonstrate that population-based search using operators like selection, mutation, and crossover can substantially improve performance at test time. Iterative context refinement methods, including Reflexion (Shinn et al., 2023), Self-Refine (Madaan et al., 2023), Training-free GRPO (Cai et al., 2025), Dynamic Cheatsheet (Suzgun et al., 2025), and ACE (Zhang et al., 2025b) further show that self-reflection can drive continued self-improvement. Agentic frameworks such as ReAct (Yao et al., 2022) and AutoGen (Wu et al., 2024) highlight that system prompts and interaction protocols play a central role in steering tool use and long-horizon behaviors. Besides, LLM-based memory mechanisms (Packer et al., 2023; Zhong et al., 2024; Lee et al., 2024; Xu et al., 2025) are also related to our work, since memory is key to helping the agent remember what strategies have worked in the past. A system prompt is often not just a foundational set of human-engineered instructions, but an agentic memory of desired behaviors and reward preferences (Ouyang et al., 2025). However, these approaches operate with frozen model parameters, limiting their ability to improve implicit knowledge acquired through experience.

**Scaling RL for LLMs** Reinforcement learning has become the primary mechanism for further improving a pre-Figure 3. Evolutionary trees of E-SPL. During RL, E-SPL creates an evolutionary tree of system prompts, by re-using the same data already generated by RL with minimal additional computational overhead. Each genetic operator (mutation or crossover) only requires a sampling server for LLM self-reflection with different context construction strategies, which can be concurrent with RL gradient updates.

#### Discovered System Prompts for Solving Math Problems (deepseek v3.1)

1. [1] For generating functions with a numerator  $(1 - x^M)^k$  and a denominator as a product of geometric series, the coefficient of  $x^N$  is a sum over  $j$  of binomial coefficients times the number of solutions to a Diophantine equation. If  $N < M$ , only the  $j = 0$  term contributes. For  $N \geq M$ , multiple  $j$  may contribute, and the signs from the binomial coefficients must be considered.
2. [2] After computations, **verify results for consistency across domains: check denominators, symmetry, and physical plausibility. Use alternative methods to recompute critical quantities and detect patterns such as periodicity.**
3. [3] Avoid detours and unjustified guesses, including contest heuristics without validation. **Trust sound derivations; if inconsistencies persist, explore alternative configurations or identities.**
4. [4] To verify modular injectivity, factor differences and check non-vanishing of the cofactor. For complex cases, test small primes; derivative condition is sufficient but not necessary.
5. [5] For sums over integer pairs with conditions like  $\gcd(a, b) = 1$ , consider using Möbius inversion or changing the order of summation to simplify the expression.
6. [6] When free parameters appear but a fixed answer is required, **look for cancellations forced by constraints or boundary conditions. Verify that the conclusion is consistent with all given conditions.**

Figure 4. Discovered strategies in learned System Prompts for solving math problems. Those explicit behavior specifications include: useful heuristics and tips for various categories of problems, self-verification strategies such as checking for consistency and plausibility, a list of common failure modes to avoid, etc. Note that RL is done under diverse system prompts, and does not overfit to any particular one.

trained LLM’s reasoning and agentic capabilities. Early preference-based methods demonstrated the effectiveness of RL for alignment and instruction following (Stiennon et al., 2020; Ouyang et al., 2022; Bai et al., 2022a,b), while more recent work on RL has focused on improving reasoning and problem-solving, starting with o1 (Jaech et al., 2024), R1 (Shao et al., 2024; Guo et al., 2025), and various further improvements (Team et al., 2025; Yu et al., 2025; Chen et al., 2025; Zheng et al., 2025; Khatri et al., 2025) to policy gradients (Williams, 1992; Schulman et al., 2017) on LLMs. However, these methods typically assume a fixed system prompt or interface during training, treating contexts

as exogenous rather than as a programmatic and optimizable object. While recent work (Ye et al., 2025) has attempted to discover new prompts within RL loops, those prompts are generated as additional synthetic training data, which are fundamentally different from *system prompts* that encode declarative knowledge, specify desired behaviors, and articulate strategic principles. Our work is also inspired by recent observations on the interplay between system prompts and RL (Tan et al., 2025; Azarbal et al., 2025; MacDiarmid et al., 2025), which suggested that certain system-level contexts can meaningfully improve RL generalization by reducing optimization pressure to globally update the model.Figure 5. **Mutation operator** in E-SPL. The highest-performing prompt in each iteration undergoes **LLM self-reflection** on group-wise agent trajectories and their outcomes. An LLM-generated diff edits the parent into a child system prompt, removing ineffective rules and converting observed mistakes into improved declarative instructions, yielding a new prompt that enters the evolutionary population.

Figure 6. **Crossover operator** in E-SPL. System prompts are compared based on their problem-wise performance within the current RL batch. Guided by these differential strengths and weaknesses, an **LLM self-reflection** process selectively recombines the most effective complementary segments from multiple parent prompts, yielding a new child prompt that enters the evolutionary population.

**Combining Evolutionary Search with RL** Hybrid approaches combining Evolution and RL have long been studied, including ERL (Khadka & Tumer, 2018), PBT (Jaderberg et al., 2017), and CEM-RL (Pourchot & Sigaud, 2018), which all leverage population dynamics to complement gradient-based optimization. Open-ended evolutionary methods such as Novelty Search (Lehman & Stanley, 2011b), MAP-Elites (Mouret & Clune, 2015), and POET (Wang et al., 2019) emphasize exploration over diverse environments and behaviors, but do not operate at the level of prompts and contexts. More recent LLM-guided systems revive program-level evolution, with FunSearch (Romera-Paredes et al., 2024) and AlphaEvolve (Novikov et al., 2025) demonstrating that evolutionary search over code and algorithms can yield novel discoveries. The Darwin Gödel Machine (DGM) (Zhang et al., 2025a) draws inspiration from Gödel machine (Schmidhuber, 2003) to include self-referential self-modification in agentic prompting, but does not include model weight updates during the self-modification process. In contrast, E-SPL jointly optimizes system prompts and model weights in a single training loop.

**Genetic Programming (GP)** is a learning paradigm where candidate programs are iteratively refined through mu-

tation and crossover operators (Koza, 1994; Banzhaf et al., 1998; Langdon & Poli, 2013) to improve performance. Modern GP approaches have demonstrated strong results across domains such as symbolic regression (Ma et al., 2022), complex scheduling (Nguyen et al., 2017), and algorithmic discovery (Chen et al., 2023). A large language model under a unique system prompt and a particular set of tools can be seen as a program (Lehman et al., 2023; Shojaei et al., 2024; Romera-Paredes et al., 2024; Opsahl-Ong et al., 2024), and the LLM can be used to flexibly design the mutation and crossover operators that continuously generate new members of the population (Hemberg et al., 2024; Agrawal et al., 2025; Novikov et al., 2025). While prior work (Surina et al., 2025) has also considered combining LLM-based genetic operators (Romera-Paredes et al., 2024) with DPO (Rafailov et al., 2023), their focus was primarily on combinatorial optimization tasks, where LLM generates specific pieces of code for problems like Traveling Salesman Problem (TSP) (Lin, 1965). By contrast, we tackle the much more general setting of LLM post-training, where the LLM needs to generate system prompts that guide the next iteration of itself for general-purpose problem solving. Compared to AI Scientist approaches (Gottweis et al., 2025), we do not use additional agentic scaffold on top of RL.Figure 7. An example of the system prompt discovered by E-SPL for RL with search engines. Here, the discovered system prompt specifies a workflow that includes: **generative self-verification, detailed instructions on tool use, critical formatting rules, heuristics for determining the reliability of various information sources, actionable guidance for avoiding common failure modes, etc.**

System Prompt for RL Agent with Search Engine (openai gpt-oss-120b)

System Prompt *Before* E-SPL

You are an expert assistant who solves tasks using a Wikipedia search tool.

Here is an example of solving a real question:

“Between 2020 and 2025, which year did New York City see the most population growth and how did San Francisco population change in that year?” ...

Calling search tool: <tool\_call>{"name": "search", "arguments": {"query\_list": ["Population New York City between 2020 and 2025"]}}</tool\_call>

...

System Prompt *After* E-SPL

You are an expert research assistant. ... **Workflow (follow exactly in this order)** ...

**3. Call the search tool once** — Use **exactly** this syntax ...

**4. Verification block** — For each result returned, **output the following four lines in this order and nothing else:**

Source: <domain>

Summary: <one-sentence summary>

Status: VERIFIED (or NOT VERIFIED)

Phrase: "<exact phrase>" (only if Status is VERIFIED)

**A result is VERIFIED only if:**

- • the exact target phrase appears verbatim in the snippet, **and**
- • the domain ends with .org, .edu, .gov, is Wikipedia, **or** a major reputable outlet (e.g., nytimes.com, bbc.com, reuters.com, theguardian.com)

If the domain is not on this list, mark the result **NOT VERIFIED** even if the phrase is present.

**5. Refine query (optional)** — **If all results are NOT VERIFIED, modify the query and repeat Step 3.**

**Critical formatting rules (must be obeyed)**

- • All tool calls must be wrapped exactly as <tool\_call>{...}</tool\_call>
- • The only user-visible output after verification is the single **Answer: line**; no bullet points, markdown, blank lines, or extra text

**Positive example (correct flow)**

User: "What other name is used for the 4 × 400 m relay?"

Query: "4 × 400 metres relay"

<tool\_call>{"name": "search", "arguments": {"query\_list": ["4 × 400 metres relay"]}}</tool\_call>

Source: en.wikipedia.org

Summary: The 4 × 400 metres relay is also known as the "long relay".

Status: VERIFIED

Phrase: "long relay" ...

**Follow exactly this loop for every user query. Your objective is to prove the answer with a verbatim phrase from a reliable source before you *ever* write the final **Answer: line**.**### 3. Method

In this section, we describe how to combine reinforcement learning (RL) on LLM weights with an evolutionary algorithm (EA) on LLM system prompts. In a nutshell, we condition RL weight updates on a population of system prompts, and reuse the resulting RL data to jointly evolve this population via two genetic operators, mutation and crossover, both of which are based on LLM self-reflection. **We design the evolutionary algorithm to be directly compatible with RL post-training pipelines, allowing RL to benefit from an evolving repertoire of declarative knowledge** while incurring minimal additional overhead.

#### 3.1. Reinforcement Learning for Weight Update

The RL component of E-SPL optimizes the LLM policy  $\pi_\theta$  to maximize rewards on dataset  $\mathcal{D}$  under a variety of system prompts. Each system prompt, denoted as  $s$ , is a list of strategic instructions that contain declarative knowledge on how to solve various categories of problems. The system prompts are not given *a priori*, but are instead discovered by the LLM through self-reflection on its own experience during the RL training process. Each RL iteration uses  $M$  system prompts  $\{s_i\}_{i=1}^M$  in parallel, so that the model weights do not overfit to any particular system prompt; this also induces direct evolutionary competition among system prompts. For each problem  $x \sim \mathcal{D}$  in the batch, and for each system prompt  $s_i$ , we sample  $N$  trajectories

$$y_{i,j} \sim \pi_\theta(\cdot \mid s_i, x) \quad \text{for } j = 1, \dots, N \quad (1)$$

Then, we compute the outcome reward  $r_{i,j}$  for each trajectory and the value function estimate  $V_i$  accordingly:

$$r_{i,j} = R(x, y_{i,j}) \quad V_i = \frac{1}{N} \sum_{j=1}^N r_{i,j} \quad (2)$$

The policy gradient objective for each problem  $x$  is:

$$\mathcal{J}(\theta, x) = \frac{1}{NM} \sum_{i=1}^M \sum_{j=1}^N (r_{i,j} - V_i) \log \pi_\theta(y_{i,j} \mid s_i, x)$$

We perform the policy gradient update to LLM weights  $\theta$  on a batch of problems  $\mathcal{B} \sim \mathcal{D}$ , while regularizing the policy  $\pi_\theta$  to be close to the reference policy  $\pi_{\text{ref}}$ :

$$\theta \leftarrow \theta + \alpha \frac{1}{|\mathcal{B}|} \sum_{x \in \mathcal{B}} \nabla_\theta (\mathcal{J}(\theta, x) - \beta \text{KL}(\pi_\theta, \pi_{\text{ref}})) \quad (3)$$

When using low-rank adaptation (Hu et al., 2022), the KL regularization is implicit (since  $\pi_\theta$  cannot deviate too much from  $\pi_{\text{ref}}$  by construction) and thus often ignored ( $\beta = 0$ ).

Performing policy gradients conditioned on learned system prompts also encourages a separation between two sources of knowledge: *declarative knowledge* that can be explicitly articulated and stored as principles in the system prompt, versus *procedural knowledge* that is tacit, intuitive, and only editable in weights. Conditioning the model on declarative knowledge in prompts reduces the optimization pressure to globally update the model in weights, and allows the weight updates to focus on acquiring procedural knowledge.

#### 3.2. Evolutionary Algorithm for Context Update

Our evolutionary algorithm (EA) optimizes the LLM system prompt to best elicit model capabilities and encode explicit knowledge learned from experience. Like in classical EA, we keep a population of candidates, and: (1) **evaluate the fitness** of different system prompts continuously during training; (2) **select** the successful ones from the population; (3) **reproduce** new system prompts from selected parents to add to the population via mutation and crossover – both grounded in LLM self-reflection but differing in their context construction strategies. The population  $\mathcal{S}$  starts with a single root, and gradually grows into an evolutionary tree.

**Fitness Evaluation** Our EA estimates the competence of each system prompt using the same trajectories already generated by RL. To ensure seamless integration into existing RL pipelines, we eliminate the need for a separate fitness evaluation on a held-out validation set that is typical of prompt evolution algorithms. Instead, evolution directly leverages the noisy train-time batch statistics generated during RL rollouts, incurring negligible additional cost.

Reusing on-policy batch statistics comes with an important consequence: absolute returns are not directly comparable across training iterations, as system prompts are evaluated on different data batches and under different model weights  $\theta$ . However, within each RL iteration, all sampled system prompts are evaluated under matched conditions – same batch, same weights. We therefore treat each RL iteration as a **tournament** round among selected system prompts, where only their relative ordering is meaningful across time.

To convert the relative ordering into persistent, comparable ratings, we use the Bayesian skill rating system TrueSkill (Herbrich et al., 2006), which models each rating as a Gaussian distribution  $\mathcal{N}(\mu, \sigma^2)$ . Concretely, given value estimates  $[V_1, \dots, V_M]$  for the sampled prompts  $[s_1, \dots, s_M]$  and their current ratings  $[(\mu_1, \sigma_1), \dots, (\mu_M, \sigma_M)]$ , we compute the sorted ordering (in descending order) for  $\{s_i\}_{i=1}^M$ :

$$\mathbf{v} \leftarrow \text{argsort}([V_1, \dots, V_M]),$$

and update their TrueSkill ratings associated with each sampled prompt  $s_i$  in the population  $\mathcal{S}$  accordingly (see §B):

$$(\mu_i, \sigma_i)_{i=1}^M \leftarrow \text{TrueSkill.update}((\mu_i, \sigma_i)_{i=1}^M, \mathbf{v}),$$

which performs approximate message passing between those ratings. This procedure yields a persistent estimate of each prompt’s competence that can aggregate evidence across iterations despite changing model weights.

**Selection** We select the system prompts used in each RL iteration based on their TrueSkill scores. We use an optimistic estimate of each system prompt  $s_i$ ’s skill level, which forms our sampling distribution for selection:

$$\text{score}_i = \mu_i + \lambda \sigma_i \quad p_i \propto \exp(\text{score}_i / T) \quad (4)$$**Algorithm 1** Evolutionary System Prompt Learning for RL

---

**Input:** Problem set  $\mathcal{D}(\cdot)$ , reward function  $R(\cdot, \cdot)$ .  
**Input:** RL policy  $\pi_\theta(\cdot)$ ; reference policy  $\pi_{\text{ref}}(\cdot)$ .  
**Input:** RL hyper-parameters:  $N, \alpha, \beta$ .  
**Input:** Initial system prompt  $\mathbf{s}_{\text{root}}$ ;  $M, p_{\text{crossover}}$ .  
**Input:** Initial  $(\mu_0, \sigma_0)$  for TrueSkill;  $\Delta\sigma; \lambda$ .  
 Initialize  $\mathcal{S} = \{(\mathbf{s}_{\text{root}}, \mu_0, \sigma_0)\}$ . **{Population to Evolve}**  
**while** not converged **do**  
     Sample a problem  $x \sim \mathcal{D}$   
     Sample  $M$  system prompts  $\{(\mathbf{s}_i, \mu_i, \sigma_i)\}_{i=1}^M \sim \mathcal{S}$ ,  
     each according to probability  $p_i \propto \exp(\mu_i + \lambda\sigma_i)$   
     **for**  $i = 1$  to  $M$  **do**  
         Sample  $N$  solutions  $\{y_{i,j}\}_{j=1}^N: y_{i,j} \sim \pi_\theta(\cdot \mid \mathbf{s}_i, x)$   
         Reward  $r_{i,j} = R(x, y_{i,j})$ , value  $V_i = \frac{1}{N} \sum_{j=1}^N r_{i,j}$   
     **end for**  
      $\mathcal{J}(\theta) = \frac{1}{NM} \sum_i \sum_j (r_{i,j} - V_i) \log \pi_\theta(y_{i,j} \mid \mathbf{s}_i, x)$   
      $\theta \leftarrow \theta + \alpha \nabla_\theta (\mathcal{J}(\theta) - \beta \text{KL}(\pi_\theta, \pi_{\text{ref}}))$  **{RL update}**  
      $\mathbf{v} \leftarrow \text{argsort}([V_1, \dots, V_M])$  **{Rank the list of  $\mathbf{s}_i$ }**  
      $\{(\mu_i, \sigma_i)\}_{i=1}^M \leftarrow \text{TrueSkill.update}(\{(\mu_i, \sigma_i)\}_{i=1}^M, \mathbf{v})$   
     **{Change the ratings in the population  $\mathcal{S}$ }**  
      $k \leftarrow \text{argmax}_i V_i$  **{Only mutate the best prompt  $\mathbf{s}_k$ }**  
     Create **mutation self-reflection** prompt  $\Psi_{\text{reflect}}$  using  
      $[(x, y_{k,1}, r_{k,1}), \dots, (x, y_{k,N}, r_{k,N})]$ .  
      $\ell \sim \pi_{\text{ref}}(\cdot \mid \Psi_{\text{reflect}})$      $\text{diff} \sim \pi_{\text{ref}}(\cdot \mid \mathbf{s}_k, \ell)$   
      $\mathbf{s}_{\text{mutate}} \leftarrow \text{git.apply}(\mathbf{s}_k, \text{diff})$  **{Mutation step}**  
      $\mathcal{S}.\text{append}((\mathbf{s}_{\text{mutate}}, \mu_k, \sigma_k + \Delta\sigma))$   
     **if**  $p_{\text{crossover}} > \text{random.rand}()$  **then**  
         Let  $\mathcal{B} = \{x_b\}_{b=1}^B$  be a batch of  $B$  past problems  
         Let  $\Omega = \{(\mathbf{s}_i, \mu_i, \sigma_i)\}_{i=1}^M$   
          $(\mathbf{s}', \mu', \sigma') = \text{Crossover}(\mathcal{B}, R, \Omega, \pi_\theta, \pi_{\text{ref}})$  **{Alg 2}**  
          $\mathcal{S}.\text{append}((\mathbf{s}', \mu', \sigma'))$   
     **end if**  
**end while**

---

where  $\lambda > 0$  controls the degree of optimism in estimating skill levels, and  $T$  is a temperature term; both higher  $\lambda$  and higher  $T$  can encourage exploration. In addition, we apply a sliding window of size  $K$ , similar to a replay buffer, so that we only sample from the latest  $K$  system prompts. This strategy favors system prompts that perform well while maintaining a high degree of exploration.

**Mutation** After evaluating the empirical performance of selected system prompts  $\{\mathbf{s}_i\}_{i=1}^M$ , we identify the best:

$$k \leftarrow \text{argmax}_i V_i \quad (5)$$

and only apply mutation on this best candidate  $\mathbf{s}_k$ . This directly adds additional evolutionary pressure on the population: in each iteration, **only the one system prompt that outperforms all of its competitively selected peers**

**Algorithm 2** Crossover( $\mathcal{B}, R, \Omega, \pi_\theta, \pi_{\text{ref}}$ )

---

**Input:** List of  $B$  problems:  $\mathcal{B}$ ; reward function  $R(\cdot, \cdot)$ .  
**Input:** List of  $M$  system prompts:  $\Omega$ .  
**Input:** RL policy  $\pi_\theta(\cdot)$ , reference policy  $\pi_{\text{ref}}(\cdot)$ .  
 Let  $\Phi \in \mathbb{R}^{M \times B}$  **{ $\Phi_{i,b}$  is the empirical return of the  $i$ -th system prompt in  $\Omega$  on the  $b$ -th problem in  $\mathcal{B}$ }**  
**for**  $(\mathbf{s}_i, \mu_i, \sigma_i)$  **in**  $\Omega$  **do**  
     **for**  $x_b$  **in**  $\mathcal{B}$  **do**  
         Sample  $\{y_{i,b,j}\}_{j=1}^N$  from  $\pi_\theta: y_{i,b,j} \sim \pi_\theta(\cdot \mid \mathbf{s}_i, x_b)$   
         **{In practice, reuse the rollouts from RL in Alg 1}**  
          $\Phi_{i,b} = \frac{1}{N} \sum_{j=1}^N R(x, y_{i,b,j})$   
     **end for**  
**end for**  
 $k \leftarrow \text{argmax}_i \sum_b \Phi_{i,b}$  **{ $\mathbf{s}_k$  is the overall top prompt}**  
 $\varphi = []$  **{Problems each  $\mathbf{s}_i$  was the top prompt for}**  
**for**  $(\mathbf{s}_i, \mu_i, \sigma_i)$  **in**  $\Omega$  **do**  
      $\rho_i = []$   
     **for**  $x_b$  **in**  $\mathcal{B}$  **do**  
         **if**  $i = \text{argmax}_z \Phi_{z,b}$  **then**  
              $\rho_i.\text{append}(x_b)$  **{ $\mathbf{s}_i$  is the top prompt for  $x_b$ }**  
         **end if**  
     **end for**  
      $\varphi.\text{append}((\mathbf{s}_i, \rho_i))$  **{Analyze why  $\mathbf{s}_i$  did best on  $\rho_i$ }**  
**end for**  
 Create **crossover self-reflection** prompt  $\Psi_{\text{cr}}$  using  $\varphi$ .  
 $\text{diff}_{\text{cr}} \sim \pi_{\text{ref}}(\cdot \mid \mathbf{s}_k, \Psi_{\text{cr}})$   
 $\mathbf{s}_{\text{crossover}} \leftarrow \text{git.apply}(\mathbf{s}_k, \text{diff}_{\text{cr}})$  **{Crossover step}**  
 $\mu' = (\sum_i \frac{\mu_i}{\sigma_i^2}) / (\sum_i \frac{1}{\sigma_i^2})$ ,  $\sigma' = \sqrt{1 / (\sum_i 1 / \sigma_i^2) + \Delta\sigma^2}$   
**Return**  $(\mathbf{s}_{\text{crossover}}, \mu', \sigma')$

---

**gets to produce a child of its own through mutation.** To run mutation on  $\mathbf{s}_k$ , we first ask the reference policy  $\pi_{\text{ref}}$  to summarize each trajectory and self-reflect (using mutation self-reflection prompt  $\Psi_{\text{reflect}}$ ) on the mistakes that  $\pi_\theta$  made in those trajectories generated under the system prompt  $\mathbf{s}_k$ :

$$\ell \sim \pi_{\text{ref}}(\cdot \mid \Psi_{\text{reflect}}, (x, y_{k,1}, r_{k,1}), \dots, (x, y_{k,N}, r_{k,N}))$$

where the model reasons across all the rollouts  $\{y_{k,j}\}_{j=1}^N$  and their outcomes  $\{r_{k,j}\}_{j=1}^N$  to **identify past mistakes and potential strategies to avoid them going forward**. Next, we use the lessons  $\ell$  from self-reflection to propose edits on the system prompt  $\mathbf{s}_k$  in a format similar to git diff:

$$\text{diff} \sim \pi_{\text{ref}}(\cdot \mid \mathbf{s}_k, \ell) \quad \mathbf{s}_{\text{mutate}} \leftarrow \text{git.apply}(\mathbf{s}_k, \text{diff})$$

Finally, we add  $\mathbf{s}_{\text{mutate}}$  to the population  $\mathcal{S}$  with TrueSkill rating  $(\mu_k, \sigma_k + \Delta\sigma)$ ; under mutation, a child inherits the parent's rating but with increased uncertainty. This favors the child in the next round of selection according to Eq (4).**Figure 8. TrueSkill scores learned from train-time rankings are predictive of relative test-time performance of system prompts.** E-SPL treats each RL iteration as a set of relative comparisons between system prompts for TrueSkill message passing. The visualization shows that E-SPL effectively aggregates noisy train-time comparisons into a stable and informative criterion for evolutionary selection.

**Crossover** enables selective recombination of knowledge from multiple parents in producing an offspring. In contrast, mutation alone isolates evolutionary branches, requiring each lineage to independently rediscover similar declarative knowledge. Crossover therefore facilitates rapid dissemination of effective strategies across the population.

Our crossover operator first computes  $V_i$  in Eq (2) for each problem in a batch  $\mathcal{B}$ , yielding a score matrix  $\Phi \in \mathbb{R}^{M \times B}$  where the rows represent  $M$  sampled system prompts  $\{s_i\}_{i=1}^M$  and the columns represent  $B$  problems in the batch  $\{x_b\}_{b=1}^B$ . Each entry  $\Phi_{i,b}$  represents the average reward of problem  $x_b$  obtained under  $s_i$  during this RL iteration.

The matrix  $\Phi$  explicitly captures the problem-wise strengths of each system prompt by identifying which system prompts outperform others on specific problems. Incorporated into the crossover self-reflection prompt  $\Psi_{cr}$ , this information grounds LLM self-reflection in concrete examples of differential advantages and enables targeted recombination of complementary prompt segments. **The goal of the crossover operator is to produce children that inherit and recombine the strongest traits of multiple parents.** In Algorithm 2, after crossover self-reflection, we apply the proposed git diff edits to  $s_k$  to obtain  $s_{crossover}$ , which is added to the population  $\mathcal{S}$  with a TrueSkill rating initialized as a precision-weighted average of the parents' ratings.

**Self-reflection is key to genetic operators** Both mutation and crossover are LLM-based self-reflection processes that diagnose failures and synthesize structured edits to produce children for the population; they differ only in how the self-reflection context is constructed. Mutation reflects on agent trajectories from a single best-performing system prompt to repair its weaknesses, while crossover reflects on comparisons across multiple system prompts in their problem-wise performance to recombine complementary segments.

### 3.3. Algorithmic Summary

Evolutionary System Prompt Learning (**E-SPL**) augments RL post-training with an evolutionary algorithm based on self-reflection from experience. In parallel with RL updates, E-SPL uses the relative ordering of each system prompt’s average reward to obtain TrueSkill ratings, which provide numerical scores that inform evolutionary selection. To create new system prompts for the population, E-SPL applies mutation and crossover operators based on LLM self-reflection. As a result, E-SPL brings the benefits of evolution to RL with minimal additional computational overhead.

## 4. Experiments

We evaluate E-SPL on mathematical reasoning and agentic search benchmarks, with the goal of answering the following questions: (1) Does jointly evolving system prompts and model weights outperform optimizing either alone? (2) How does E-SPL affect the learning speed and generalization of RL? (3) What are some emergent patterns in the system prompts discovered via E-SPL? (4) What roles do mutation and crossover play in the evolutionary dynamics of E-SPL?

### 4.1. Experimental Setup

**Tasks and datasets.** We first focus on mathematical reasoning tasks with limited data, a regime in which inductive biases from evolved system prompts are expected to play a particularly important role. Our main evaluations include: (1) **DAPO100→AIME25**, which follows the protocols from (Cai et al., 2025) that trains on 100 selected problems from the DAPO dataset (Yu et al., 2025) and tests on AIME 2025 competition; (2) **HMMT 23/24→25**, which uses all of HMMT 2023, 2024, and February 2025 problems as training set, and November 2025 problems as test set; (3) **AIME→BeyondAIME**, which uses past years ofFigure 9. An example of how the RL model learns to utilize learned system prompts (numbered G1 ... G30) to solve problems.Example of How E-SPL Shapes Model BehaviorLearned System Prompt

When solving math problems, you MUST first carefully read and understand the helpful instructions and principles below:

**[G1]. When optimizing sums, set parameters to cancel dominant terms. For quadratic forms, choose parameters that minimize the leading coefficients, reducing the problem to analyzing remainder terms.**

...

**[G22]. In modular arithmetic and sums, decompose terms via remainders or moduli. Use periodicity for remainder sums and CRT for coprime factors, ensuring multiplicative solutions for square-free moduli.**

...

Problem

Let  $\mathbb{Z}/n\mathbb{Z}$  be the set of all integers modulo  $n$  (hence  $\mathbb{Z}/n\mathbb{Z}$  contains  $n$  elements). Find the second smallest composite number  $n$ , such that there exists a bijective function  $g : \mathbb{Z}/n\mathbb{Z} \rightarrow \mathbb{Z}/n\mathbb{Z}$  such that the 101 functions

$$g(x), g(x) + x, g(x) + 2x, \dots, g(x) + 100x$$

are all bijective from  $\mathbb{Z}/n\mathbb{Z}$  to  $\mathbb{Z}/n\mathbb{Z}$ .

Model Output

**Understanding the Condition:** Each function  $h_k(x) = g(x) + kx$  (for  $k = 0, 1, \dots, 100$ ) must be a permutation of  $\mathbb{Z}/n\mathbb{Z}$ . Since  $g$  is bijective,  $h_0(x) = g(x)$  is bijective by assumption. For  $k \geq 1$ ,  $h_k$  is bijective if and only if for every  $a \in \mathbb{Z}/n\mathbb{Z}$ , the equation  $g(x) + kx = a$  has a unique solution  $x$ .

**Alternatively**, since  $g$  is a permutation, we can set  $y = g(x)$  and then  $x = g^{-1}(y)$ , so  $h_k(g^{-1}(y)) = y + kg^{-1}(y)$ . But perhaps a better approach is to consider the differences. **Note:**  $h_k$  is bijective if and only if the map  $x \mapsto g(x) + kx$  is injective. That is, if  $g(x) + kx = g(y) + ky$  then  $x = y$ . Equivalently,  $g(x) - g(y) = -k(x - y)$ . Since  $g$  is arbitrary (but fixed for all  $k$ ), we need to choose  $g$  such that for each  $k = 1, \dots, 100$ , the equation  $g(x) - g(y) = -k(x - y)$  implies  $x = y$ .

**Alternatively**, we can think about the differences: for each nonzero  $d \in \mathbb{Z}/n\mathbb{Z}$ , we require that  $g(x + d) - g(x) \neq -kd$  for all  $x$  and for all  $k = 0, \dots, 100$ . Actually, injectivity of  $h_k$  is equivalent to: if  $x \neq y$ , then  $g(x) + kx \neq g(y) + ky$ , i.e.,  $g(x) - g(y) \neq -k(x - y)$ . So for each nonzero  $d$ , and for each  $k$ , we require that  $g(x + d) - g(x) \neq -kd$  for all  $x$ . **But** since  $g$  is a permutation, this is not easy to control.

**Another idea:** Consider the composition  $h_k \circ g^{-1}$ . Let  $f = g^{-1}$ . Then  $h_k(g^{-1}(y)) = y + kf(y)$  ...

...

**But let's think:** For  $n$  to work, it must be that all prime factors are at least 103? But that gives large numbers. Maybe there is a trick: The number 101 is prime. So perhaps  $n$  must be coprime to 101! or something.

**Another idea:** Consider the function for  $k = 101$ ? Not required.

**Given the instructions, we should use [G22]: In modular arithmetic, use CRT.**

If  $n = \text{product of primes}$ , then by CRT, the condition holds if and only if it holds for each prime power factor.

So for each  $p^a$  dividing  $n$ , there exists a bijection  $g_p$  on  $\mathbb{Z}/p^a\mathbb{Z}$  such that the 101 functions are bijective.

For a prime  $p > 101$ , take  $g(x) = x$  works. For  $p \leq 101$ , we need to see if there is a bijection on  $\mathbb{Z}/p^a\mathbb{Z}$ .

So the smallest composite would be one that has no prime factor  $\leq 101$ , i.e., all factors  $\geq 103$ . So the smallest is  $103^2 = 10609$ . But then the second smallest would be  $103 \times 107 = 11021$ . ...

**Final Answer:**

11021

**Ground truth answer:** 11021Figure 10. An example of how the RL model learns to utilize learned system prompts to solve problems.

Example of How E-SPL Shapes Model Behavior

Learned System Prompt

When solving math problems, you MUST first carefully read and understand the helpful instructions and principles below:

...

**[G26]. After deriving results, perform sanity checks: test small instances, ensure bounds, and verify extreme cases. Re-examine complex steps if results seem off.**

...

Problem

Real numbers  $x_1, x_2, \dots, x_n$  satisfy  $|x_1| + |x_2| + \dots + |x_{3526}| = 1$ . Let  $M$  be the maximal possible value of

$$\frac{1}{|x_1| + |x_1 - x_2| + |x_1 + x_2 - x_3| + \dots + |x_1 + x_2 + \dots + x_{3525} - x_{3526}|}$$

Find  $M \bmod 1000$ .

Model Output

**Let's analyze the problem step by step:**

Let  $S_k = x_1 + x_2 + \dots + x_k$  for  $k = 1, 2, \dots, 3526$ , with  $S_0 = 0$ . Then the denominator becomes:  $|S_1| + |S_1 - S_2| + |S_2 - S_3| + \dots + |S_{3525} - S_{3526}|$ . But note:  $S_1 = x_1$ ,  $S_1 - S_2 = -x_2$ ,  $S_2 - S_3 = -x_3$ , ...,  $S_{3525} - S_{3526} = -x_{3526}$ . So the denominator is:  $|S_1| + |x_2| + |x_3| + \dots + |x_{3526}|$ . However, this is not exactly matching the given expression. Actually, the given terms are: ...

...

The correct maximum is actually 2, achieved by the first construction with all terms zero except the last. Therefore,  $M = 2$ . Then  $M \bmod 1000 = 2$ .

**So the answer is 2.** This is the simplest.

**Given the instructions [G26]: perform sanity checks.**

For  $n = 2$ : variables  $x_1, x_2$  with  $|x_1| + |x_2| = 1$ .  $D = 1/(|x_1| + |x_1 - x_2|)$ . ... So for  $n = 2$ , maximum is 2.

For  $n = 3$ : maximize  $D = 1/(|x_1| + |x_1 - x_2| + |x_1 + x_2 - x_3|)$ . ... So for  $n = 3$ , we can get  $D = 4$ .

In general, for  $n$ , the construction is:  $x_1 = a$ ,  $x_2 = a$ ,  $x_3 = 2a$ ,  $x_4 = 4a$ , ...,  $x_n = 2^{n-2}a$ . Then  $|x_1| = |a|$ ,  $|x_1 - x_2| = 0$ ,  $|x_1 + x_2 - x_3| = 0$ , ...,  $|x_1 + \dots + x_{n-1} - x_n| = |(2^{n-1} - 1)a - 2^{n-2}a| = |(2^{n-2} - 1)a|$ .

**Wait, careful:**  $S_{n-1} = a + a + 2a + 4a + \dots + 2^{n-3}a = (2^{n-2} - 1)a + a = 2^{n-2}a$ .

Then  $S_{n-1} - x_n = 2^{n-2}a - 2^{n-2}a = 0$ .

So all terms after the first are 0 except the second which is 0.

So denominator =  $|a|$ .

Constraint:  $|a| + |a| + |2a| + |4a| + \dots + |2^{n-2}a| = |a|(2 + (2^{n-1} - 2)) = |a| * 2^{n-1} = 1$ , so  $|a| = 2^{1-n}$ . Then  $D = 2^{n-1}$ .

For  $n = 3526$ ,  $D = 2^{3525}$ . **This is huge, and integer.**

**And it is better than the previous construction.**

So the maximum is  $M = 2^{3525}$ . Then  $M \bmod 1000 = 2^{3525} \bmod 1000$ .

Now, we need to compute  $2^{3525} \bmod 1000$ . ... So  $2^{3525} \equiv 432 \pmod{1000}$ .

**So the answer is 432.** This is the intended solution.

**Final Answer:**

432

**Ground truth answer: 432**Figure 11. Results on mathematical reasoning datasets using DeepSeek-v3.1. **Evolution + RL** (our E-SPL method) performs best compared to self-reflection (prompting only), evolution (prompting only), and RL (weight update only). The results show that coupling RL with evolution on system prompts unlocks a synergistic form of self-improvement that neither approach achieves in isolation.

AIME (AIME 22, AIME 23, and AIME 24) as the training set, and BeyondAIME (ByteDance-Seed, 2025) as test set, which is a collection of original problems with difficulty levels greater than or equal to that of the hardest AIME problems each year. In addition, we also evaluate on (4) **Agentic Search** on Natural Questions (NQ) (Kwiatkowski et al., 2019) and HotpotQA (Yang et al., 2018), where we subsample 196 problems for training and 64 problems for test; the task is to teach an LLM model to use search engines to answer complex multi-hop questions (Jin et al., 2025).

**Models** On math tasks, we use the base model DeepSeek v3.1 (DeepSeek-AI, 2024) for our experiments, which is a 671B MoE (Shazeer et al., 2017) model with 37B active parameters. On agentic search tasks, we use OpenAI’s gpt-oss-120b (Agarwal et al., 2025), which is a 120B MoE model with 5B active parameters. All finetuning is done via low-rank adaptation (Hu et al., 2022) with a rank of 32.

**Methods compared** We compare the following variants:

- • **RL only**: standard RL post-training with a fixed system prompt; we benchmark the popular **GRPO** (Shao et al., 2024) and **CISPO** (Chen et al., 2025) algorithms;
- • **Self-reflection**: iterative prompt refinement via reflection (Shinn et al., 2023; Cai et al., 2025; Ouyang et al., 2025), without weight updates. This is essentially our mutation operator on its own, with frozen weights, no crossover operator, and a population size of 1.
- • **Evolution**: which runs our evolutionary algorithm (EA) for system prompt search with frozen model weights. Our evolutionary algorithm is very similar to GEPA (Agrawal et al., 2025) except that we do not continuously run evaluation on a separate feedback set, and instead only use train-time rollout statistics just like in standard RL. Like GEPA, our EA works well on its own, and can sometimes outperform RL.

- • **Evolution + RL (E-SPL)**: our full method, jointly evolving model contexts and updating model weights.

## 4.2. Main Results

Figure 11 summarizes performance on math reasoning benchmarks using DeepSeek-v3.1 model. Figure 13 summarizes our results on agentic search tasks using OpenAI gpt-oss-120b. Across all datasets, **Evolution + RL** (E-SPL) consistently outperforms baselines that optimize either prompts (evolution-only) or weights (RL-only) in isolation.

- • On **AIME 2025**, E-SPL improves success rate from 56.3% → 60.6% compared to the RL-only (GRPO).
- • On **HMMT 2025**, performance improves from 50.0% → 52.7% after applying E-SPL to RL.
- • On an easy-to-hard generalization (Sun et al., 2024) setting, **AIME → BeyondAIME**, the gains are the most pronounced, where E-SPL improves RL from 38.8% → 45.1% in success rate.
- • On **agentic search** tasks, E-SPL improves RL from 44.2% → 48.6%, which demonstrates that E-SPL is not limited to mathematical reasoning, but is also effective in agentic reasoning with tool use.
- • On all tasks above, E-SPL outperforms evolution-only as well, with the relative improvements ranging from +5.8% to +12.8% on the mathematical reasoning tasks, and +42.9% on the agentic search task.

Evolution-only consistently outperforms self-reflection, and can sometimes outperform RL-only. On both DAPO100 → AIME25 and AIME → BeyondAIME, evolution outperforms RL; on HMMT 23/24 → 25, GRPO slightly outperforms evolution. This finding is consistent with prior findings in GEPA (Agrawal et al., 2025), and validates the design of our proposed evolutionary algorithm on its own.**Figure 12. E-SPL learns faster, and achieves better asymptotic performance that neither RL nor prompt evolution alone could achieve.** By studying the learned system prompts (e.g. Figure 4 and Figure 9), we hypothesize that E-SPL achieves this by encouraging a natural division between declarative knowledge (factual information or global heuristics that can be verbally explained) in prompts and procedural knowledge (practical, instinctive “know-how” that relies on intuition developed through repetition) in weights.

**Figure 13.** E-SPL outperforms both (prompt) Evolution and RL on agentic search tasks with openai gpt-oss-120b model.

Figure 12 visualizes the learning curves across three mathematical reasoning settings. E-SPL demonstrates better sample efficiency and higher asymptotic performance compared to both RL-only and evolution-only baselines. Evolution-only often learns the fastest early on since it only has to update model contexts, but obtains lower asymptotic performance because it does not update model weights and therefore struggles with improving procedural knowledge.

In agentic search, we find that RL can significantly outperform prompt evolution (44.2% versus 34.0%); whereas in math reasoning, RL often performs on-par with evolution (Figure 11). It can often be hard to predict *a priori* whether RL or prompt evolution will perform better on a particular domain, since the learning bottleneck could be an arbitrary mix of declarative and procedural knowledge. E-SPL naturally combines the strengths of both, allowing continual self-improvement to proceed along whichever axis of knowledge is most limiting for the task at hand.

Overall, the results above suggest that evolving system prompts and applying RL to weights play complementary

roles in LLM self-improvement. System prompts progressively accumulates explicit, declarative knowledge – such as high-level principles, robust workflows, corner cases, and failure modes (see Figure 4 and Figure 7) – while RL updates the weights to acquire procedural knowledge that is tacit and intuitive. By conditioning RL on increasingly informative system prompts, E-SPL allows weight updates to focus on intuitive skills, leading to better sample efficiency.

#### 4.3. Emergent Patterns in Learned System Prompts

We observe several recurring patterns that emerge in the learned system prompts, which we discuss below.

**Generative Self-Verification** We find that a repeated theme in learned system prompts is about learning to self-verify a solution in a generative manner (Zhang et al., 2024).

For instance, in the agentic search setting, the learned system prompt (Figure 7) defines a workflow that explicitly labels each search result as VERIFIED versus NOT VERIFIED based on two criteria: whether the target phrase appears verbatim in the snippet returned by search engine, and whether the domain name ends with .org, .edu, .gov, is a major reputable news outlet or Wikipedia. It goes on to specify that if all results have NOT VERIFIED status, then the model should repeat the search and continue the loop in its chain of thought. The learned system prompt defines the objective as ‘*to prove the answer with a verbatim phrase from a reliable source before you ever write the final Answer: line*’. **The RL process, in turn, does not need to discover new workflows from scratch and encode them in weights, but rather refines and perfects the execution of the workflows specified in the discovered system prompts.** Such workflows emerge through E-SPL and are absent from the initial system prompt.

In solving math problems, a similar self-verification strat-<table border="1">
<thead>
<tr>
<th colspan="2">Mistakes in Learned E-SPL System Prompts</th>
</tr>
</thead>
<tbody>
<tr>
<td><b>Discovered</b></td>
<td>For a polynomial <math>P(n) = \prod_{i=1}^m (n - a_i)</math>, the largest integer <math>d</math> that divides <math>P(n)</math> for all integers <math>n</math> is <math>d = \prod_p p^{e_p}</math>, where <math>e_p = \min_{r \bmod p} (\#\{i : a_i \equiv r \pmod{p}\})</math>. Only primes <math>p \leq \max(a_i)</math> need be considered, as for larger <math>p</math>, the <math>a_i</math> are distinct modulo <math>p</math>, so <math>e_p = 0</math>. For primes where the function mapping <math>i</math> to <math>a_i \bmod p</math> is a permutation (e.g., if <math>a_i = i^3</math> and <math>p \equiv 2 \bmod 3</math>), <math>e_p = \lfloor \frac{m}{p} \rfloor</math>.</td>
</tr>
<tr>
<td><b>Corrected</b></td>
<td>For a polynomial <math>P(n) = \prod_{i=1}^m (n - a_i)</math>, the largest integer <math>d</math> that divides <math>P(n)</math> for all integers <math>n</math> is <math>d = \prod_p p^{e_p}</math>, where <math>e_p = \min_{n \in \mathbb{Z}} v_p(P(n))</math>. Moreover, one always has the lower bound <math>e_p \geq \min_{r \bmod p} (\#\{i : a_i \equiv r \pmod{p}\})</math>. In particular, for this mod-<math>p</math> bound it suffices to consider primes <math>p \leq m</math>, since if <math>p &gt; m</math> then some residue class occurs zero times and the minimum is 0. For primes where the function mapping <math>i</math> to <math>a_i \bmod p</math> is a permutation (e.g., if <math>a_i = i^3</math> and <math>p \equiv 2 \bmod 3</math>), the bound gives <math>e_p \geq \lfloor \frac{m}{p} \rfloor</math>.</td>
</tr>
<tr>
<td><b>Discovered</b></td>
<td>For graphs with odd-degree vertices, find a minimum <math>T</math>-join by pairing odd vertices along efficient paths. In symmetric graphs such as grids, pair boundary vertices optimally to minimize total length.</td>
</tr>
<tr>
<td><b>Corrected</b></td>
<td>For graphs with odd-degree vertices, find a minimum <math>T</math>-join by pairing odd vertices via a minimum-weight perfect matching on shortest-path distances. In symmetric graphs such as grids, restrict the matching to symmetry-respecting pairings to minimize total length.</td>
</tr>
</tbody>
</table>

Table 1. The discovered system prompts sometimes include **heuristics that are not always true but still helpful to problem-solving**. Unlike RL updates to model weights, the learned system prompts are interpretable and thus can be monitored and corrected. Here we showcase two subtle mistakes in two discovered principles, and what their **corrected** versions should be (see §C.1 for explanations).

egy emerged. In Figure 10, the learned system prompt has discovered the following instruction: ‘After deriving results, perform sanity checks: test small instances, ensure bounds, and verify extreme cases. Re-examine complex steps if results seem off.’ The full example in Figure 10 illustrates how the RL model has learned to use this particular self-verification strategy to solve a particular problem on BeyondAIME test set: after initially concluding that ‘So the answer is 2’, the model explicitly cites this strategy from the system prompt, and follows the exact instruction to perform sanity checks by starting from small instances of  $n = 2$  and  $n = 3$ , before moving on to the general case and saying ‘Wait, careful’ when it identified its own mistake. The model then self-corrects and concludes with ‘So the answer is 432. This is the intended solution’, which is correct. This example explains E-SPL’s effectiveness: high-level strategies are specified by the system prompt, while RL fine-tunes execution. Self-rewrite of the system prompt induces more systematic behavior changes than weight updates alone.

**Codified Domain Expertise and Heuristics** The learned system prompts also have many strategies that are highly domain-specific. For instance, Figure 9 showcases learned system prompts such as ‘When optimizing sums, set parameters to cancel dominant terms. For quadratic forms, ...’ (which is specific to problems that optimize sums) and ‘In modular arithmetic and sums, decompose terms via remainders or moduli. Use periodicity for remainder sums and CRT for coprime factors, ensuring multiplicative solutions for square-free moduli.’ (which points out the importance of Chinese Remainder Theorem, or CRT for short, in solving modular arithmetic). Example 9 shows that the RL model persistently explores many potential direc-

tions, repeating phrases like ‘Alternatively’, ‘Another idea’, ‘But let’s think’, before simplifying the problem to a form of modular arithmetic and recognizing that it should apply CRT in that situation according to the system prompt. Example 16 shows that the RL model cited a helpful heuristic in the system prompt suggesting the use of Möbius inversion under specific conditions, which became a crucial step in the solution. Those examples demonstrate that **E-SPL can effectively codify declarative knowledge into the system prompt – specialized domain expertise and helpful problem-solving heuristics under various circumstances – so that RL can focus on honing core reasoning skills** rather than memorizing domain knowledge.

**Failure Mode: Missing Caveats** Not all discovered strategies from E-SPL can hold true in the most general cases. In Table 1, we study two cases where the learned system prompt included heuristics that are not strictly true in general but are still helpful in limited scenarios. For instance, in analyzing polynomials of a particular form, the discovered strategy misunderstood the lower bound of a min as the actual min. In the other case, the discovered principle for graph theory vaguely outlined an algorithm for finding a minimum  $T$ -join by ‘pairing odd vertices along efficient paths’; the actual optimal solution requires pairing odd vertices via a minimum-weight perfect matching on shortest-path distances. The likely reason for those mistakes is that E-SPL over-generalized some particular training data points or made mistakes in its self-reflection process. E-SPL is still missing caveats in many of its generalized principles.

However, we argue that such mistakes in acquired knowledge are more transparent and interpretable than RL gradient updates, because they can be more easily monitored**Figure 14. Crossover operator in evolution speeds up early learning**, but obtains mixed results in terms of final performance. On AIME → BeyondAIME, Mutation + RL gets to 45.1% success rate, while Mutation + Crossover + RL gets to 42.5%; on HMMT, using Crossover operator results in slightly better performance (52.7% vs 51.6%). In either case, Evolution + RL outperforms RL-only.

and edited by humans. A model trained by RL alone will also pick up on certain assumptions that are only partially correct, but even the best mechanistic interpretability tools today (Gao et al., 2024; Dunefsky et al., 2024; Kharlapenko et al., 2025) still cannot interpret those weight updates or edit the behaviors in a straightforward manner. By comparison, E-SPL represents a part of learned knowledge as explicit, propositional statements, which are much more suited for further verification, formalization (Trinh et al., 2024; Hubert et al., 2025), and scalable oversight (Leike et al., 2018; Bowman et al., 2022; McAleese et al., 2024).

#### 4.4. Ablation Studies

**Effect of Crossover** Figure 14 analyzes the contribution of genetic operators. Adding **crossover** to mutation accelerates early learning, consistent with its role in propagating high-fitness substructures across evolutionary branches. However, its effect on final performance is mixed: on AIME → BeyondAIME, mutation-only achieves higher final accuracy, while on HMMT, mutation + crossover performs slightly better. **Both variants outperform RL-only**, confirming the value of evolutionary search over system prompts regardless of genetic operator choice.

Figure 3 visualizes evolutionary trees produced during training. As shown, the crossover operator enables knowledge transfer across different branches of the evolutionary tree, but at the cost of reduced *speciation* – long-term separation between evolutionary lineages. To better understand the tradeoffs of using crossover, we draw attention to prior work (Golberg, 1989; Eshelman, 1991; Mahfoud, 1995; Vose, 1999) that supports our observations:

- • crossover can accelerate short-term progress by mixing useful substructures,
- • but it can also wash out subpopulation structure, reduce long-run diversity, and increase the risk of premature homogenization and convergence.

**Figure 15.** Using a fixed reference policy rather than the online RL policy for prompt evolution is important to performance.

Mutation-only behaves more like independent local hill-climbs along separate lineages. Lineages stay separated longer, so we can keep parallel hypotheses alive (Holland, 1992). Mutation + Crossover accelerates early progress by re-combining partial improvements, but the population may rapidly converge around a locally high-performing system prompt, limiting exploration of qualitatively and structurally distinct high-quality solutions (Spears, 1993). Classically, this tradeoff can be significantly improved by a class of techniques called *niching*, which include fitness sharing (Goldberg et al., 1987), mating restriction (Deb & Goldberg, 1989), and the island model (Whitley et al., 1999). We leave the application of niching to E-SPL for future work.

**Sampling Policy for Evolution** Figure 15 evaluates the effect of using a fixed reference policy  $\pi_{\text{ref}}$  for system prompt evolution, rather than the continually changing RL policy  $\pi_{\theta}$ . Using the fixed reference policy  $\pi_{\text{ref}}$  for evolutionary updates consistently improves the final performance of E-SPL. This supports our design choice to keep evolutionary operators stationary: allowing the editor itself to drift alongside the policy introduces instability and degrades the quality of system prompt edits. A fixed reference policy provides a stable basis for self-reflection, mutation, and crossover.Figure 16. An example of how the RL model learns to utilize learned system prompts to solve problems (also see Figure 22).

Example of How E-SPL Shapes Model Behavior

Learned System Prompt

When solving math problems, you **MUST** first carefully read and understand the helpful instructions and principles below:  
 ... **[G2]. For sums over integer pairs with conditions like  $\gcd(a, b) = 1$ , consider using Möbius inversion or changing the order of summation to simplify the expression.** ...

Problem

Let  $S$  be the set of all positive integers less than 143 that are relatively prime to 143. Compute the number of ordered triples  $(a, b, c)$  of elements of  $S$  such that  $a + b = c$ .

Model Output

**Let’s solve the problem step by step:**  
**Step 1:** Factorize  $143 = 11 \times 13$ . So,  $S$  consists of positive integers  $< 143$  that are coprime to both 11 and 13.  
 ...  
**Step 5: Use Möbius inversion (Principle [G2]).** Let  $f(n)$  be the number of pairs  $(a, b) \in S \times S$  with  $a + b = n$ .  
 We want  $\sum_{n=1}^{142} f(n) \cdot \mathbf{1}_{\gcd(n, 143)=1}$ . ...  
 ...  
**Step 11:** Apply inclusion-exclusion. ... Therefore, the number of ordered triples  $(a, b, c)$  is 5940.  
**Final Answer:**

5940

**Ground truth answer:** 5940

## 5. Future Directions

The fundamental reason why E-SPL can work is that the base LLMs already have strong in-context learning and instruction following abilities. The RL process in itself does not fully take advantage of this – RL performs trial and error conditioned on an exogenous piece of context, without having the model think about how it can directly re-program itself by re-writing its own system instructions. E-SPL is a step towards augmenting RL with *self-rewrite*, by jointly evolving a population of system prompts for self-conditioning. We outline a few promising future directions:

**Agentic Retrieval for Long System Prompts** To build agents that can absorb an arbitrary amount of knowledge, the learned system prompt will eventually need a more refined sub-structure with many long components. A natural solution is to store certain components of the system prompt in the local file-system, and have the model access them in an agentic manner using a terminal and `grep` commands.

**Self-Write as an RL Problem** So far, the LLM used for self-reflection and self-rewrite does not self-improve, which means that eventually it might become a bottleneck. It is possible to extend RL to the entire self-reflection and self-write process, but this would require additional research.

**Self-Referential Recursive Self-Improvement** Our current context update and weight update algorithms are fundamentally limited; ultimately, the model should be allowed to make paradigm-level changes to its own learning algorithm. This can be achieved by allowing the model to directly modify its own training script and weights, thereby realizing fully self-referential self-improvement (Schmidhuber, 2003; 2005). A system capable of such root-level self-modification, unlocking sustained, open-ended, unbounded improvements in its own capabilities, is also called Seed AI. (Yudkowsky, 2007).

## 6. Conclusion

We have shown that unifying RL and evolutionary prompt search significantly enhances LLM self-improvement. E-SPL demonstrates that RL and Evolution are not competing paradigms but deeply synergistic ones. On a variety of reasoning and agentic tasks, our unified approach leads to better sample efficiency, asymptotic performance, and generalization. While the current method is not yet fully self-referential, it takes a meaningful step in that direction by allowing the model to self-modify its own system-level instructions. We believe further unification of Evolution and RL for self-rewrite can pave the way toward Seed AI.## References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023.

Agarwal, S., Ahmad, L., Ai, J., Altman, S., Applebaum, A., Arbus, E., Arora, R. K., Bai, Y., Baker, B., Bao, H., et al. gpt-oss-120b & gpt-oss-20b model card. *arXiv preprint arXiv:2508.10925*, 2025.

Agrawal, L. A., Tan, S., Soylu, D., Ziems, N., Khare, R., Opsahl-Ong, K., Singhvi, A., Shandilya, H., Ryan, M. J., Jiang, M., et al. Gepa: Reflective prompt evolution can outperform reinforcement learning. *arXiv preprint arXiv:2507.19457*, 2025.

Azarbal, A., Gillioz, V., Ivanov, V., Woodworth, B., Drori, J., Wichers, N., Ehtekar, A., Cloud, A., and Turner, A. M. Recontextualization mitigates specification gaming without modifying the specification. *arXiv preprint arXiv:2512.19027*, 2025.

Bäck, T., Fogel, D. B., and Michalewicz, Z. Handbook of evolutionary computation. *Release*, 97(1):B1, 1997.

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., Das-Sarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. *arXiv preprint arXiv:2204.05862*, 2022a.

Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., Chen, A., Goldie, A., Mirhoseini, A., McKinnon, C., et al. Constitutional ai: Harmlessness from ai feedback. *arXiv preprint arXiv:2212.08073*, 2022b.

Banzhaf, W., Francone, F. D., Keller, R. E., and Nordin, P. *Genetic programming: an introduction: on the automatic evolution of computer programs and its applications*. Morgan Kaufmann Publishers Inc., 1998.

Bowman, S. R., Hyun, J., Perez, E., Chen, E., Pettit, C., Heiner, S., Lukošūtė, K., Askell, A., Jones, A., Chen, A., et al. Measuring progress on scalable oversight for large language models. *arXiv preprint arXiv:2211.03540*, 2022.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33: 1877–1901, 2020.

ByteDance-Seed. Beyondaime: Advancing math reasoning evaluation beyond high school olympiads. *Hugging Face repository*, 2025.

Cai, Y., Cai, S., Shi, Y., Xu, Z., Chen, L., Qin, Y., Tan, X., Li, G., Li, Z., Lin, H., et al. Training-free group relative policy optimization. *arXiv preprint arXiv:2510.08191*, 2025.

Chen, A., Li, A., Gong, B., Jiang, B., Fei, B., Yang, B., Shan, B., Yu, C., Wang, C., Zhu, C., et al. Minimax-m1: Scaling test-time compute efficiently with lightning attention. *arXiv preprint arXiv:2506.13585*, 2025.

Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Pham, H., Dong, X., Luong, T., Hsieh, C.-J., Lu, Y., et al. Symbolic discovery of optimization algorithms. *Advances in neural information processing systems*, 36:49205–49233, 2023.

Comanici, G., Bieber, E., Schaeckermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blstein, M., Ram, O., Zhang, D., Rosen, E., et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. *arXiv preprint arXiv:2507.06261*, 2025.

Deb, K. and Goldberg, D. E. An investigation of niche and species formation in genetic function optimization. In *Proceedings of the third international conference on Genetic algorithms*, pp. 42–50, 1989.

DeepSeek-AI. Deepseek-v3 technical report, 2024. URL <https://arxiv.org/abs/2412.19437>.

Dunefsky, J., Chlenski, P., and Nanda, N. Transcoders find interpretable llm feature circuits. *Advances in Neural Information Processing Systems*, 37:24375–24410, 2024.

Eshelman, L. J. Preventing premature convergence in genetic algorithms by preventing incest. In *Proceedings of Fourth International Conference on Genetic Algorithms, 1991*, 1991.

Fernando, C., Banarse, D., Michalewski, H., Osindero, S., and Rocktäschel, T. Promptbreeder: Self-referential self-improvement via prompt evolution. *arXiv preprint arXiv:2309.16797*, 2023.

Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., and Wu, J. Scaling and evaluating sparse autoencoders. *arXiv preprint arXiv:2406.04093*, 2024.

Golberg, D. E. Genetic algorithms in search, optimization, and machine learning. *Addison wesley*, 1989(102):36, 1989.

Goldberg, D. E., Richardson, J., et al. Genetic algorithms with sharing for multimodal function optimization. In *Genetic algorithms and their applications: Proceedings*of the *Second International Conference on Genetic Algorithms*, volume 4149, pp. 414–425. Lawrence Erlbaum, Hillsdale, NJ, 1987.

Gottweis, J., Weng, W.-H., Daryin, A., Tu, T., Palepu, A., Sirkovic, P., Myaskovsky, A., Weissenberger, F., Rong, K., Tanno, R., et al. Towards an ai co-scientist. *arXiv preprint arXiv:2502.18864*, 2025.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.

Guo, Q., Wang, R., Guo, J., Li, B., Song, K., Tan, X., Liu, G., Bian, J., and Yang, Y. Connecting large language models with evolutionary algorithms yields powerful prompt optimizers. *arXiv preprint arXiv:2309.08532*, 2023.

Hemberg, E., Moskal, S., and O’Reilly, U.-M. Evolving code with a large language model. *Genetic Programming and Evolvable Machines*, 25(2):21, 2024.

Herbrich, R., Minka, T., and Graepel, T. Trueskill™: a bayesian skill rating system. *Advances in neural information processing systems*, 19, 2006.

Holland, J. H. *Adaptation in natural and artificial systems: an introductory analysis with applications to biology, control, and artificial intelligence*. MIT press, 1992.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al. Lora: Low-rank adaptation of large language models. *ICLR*, 1(2):3, 2022.

Hubert, T., Mehta, R., Sartran, L., Horváth, M. Z., Žužić, G., Wieser, E., Huang, A., Schrittwieser, J., Schroecker, Y., Masoom, H., et al. Olympiad-level formal mathematical reasoning with reinforcement learning. *Nature*, pp. 1–3, 2025.

Jaderberg, M., Dalibard, V., Osindero, S., Czarnecki, W. M., Donahue, J., Razavi, A., Vinyals, O., Green, T., Dunning, I., Simonyan, K., et al. Population based training of neural networks. *arXiv preprint arXiv:1711.09846*, 2017.

Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky, A., Low, A., Helyar, A., Madry, A., Beutel, A., Carney, A., et al. Openai o1 system card. *arXiv preprint arXiv:2412.16720*, 2024.

Jin, B., Zeng, H., Yue, Z., Yoon, J., Arik, S., Wang, D., Zamani, H., and Han, J. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. *arXiv preprint arXiv:2503.09516*, 2025.

Kalyanakrishnan, S., Tewari, A., Auer, P., and Stone, P. Pac subset selection in stochastic multi-armed bandits. In *ICML*, volume 12, pp. 655–662, 2012.

Kaufmann, E., Cappé, O., and Garivier, A. On bayesian upper confidence bounds for bandit problems. In *Artificial intelligence and statistics*, pp. 592–600. PMLR, 2012.

Khadka, S. and Tumer, K. Evolution-guided policy gradient in reinforcement learning. *Advances in Neural Information Processing Systems*, 31, 2018.

Kharlapenko, D., Shabalin, S., Barez, F., Conmy, A., and Nanda, N. Scaling sparse feature circuit finding for in-context learning. *arXiv preprint arXiv:2504.13756*, 2025.

Khatri, D., Madaan, L., Tiwari, R., Bansal, R., Duvvuri, S. S., Zaheer, M., Dhillon, I. S., Brandfonbrener, D., and Agarwal, R. The art of scaling reinforcement learning compute for llms. *arXiv preprint arXiv:2510.13786*, 2025.

Kingma, D. P. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*, 2014.

Koza, J. R. Genetic programming as a means for programming computers by natural selection. *Statistics and computing*, 4(2):87–112, 1994.

Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Devlin, J., Lee, K., et al. Natural questions: a benchmark for question answering research. *Transactions of the Association for Computational Linguistics*, 7:453–466, 2019.

Langdon, W. B. and Poli, R. *Foundations of genetic programming*. Springer Science & Business Media, 2013.

Langley, P. Crafting papers on machine learning. In Langley, P. (ed.), *Proceedings of the 17th International Conference on Machine Learning (ICML 2000)*, pp. 1207–1216, Stanford, CA, 2000. Morgan Kaufmann.

Lee, K.-H., Chen, X., Furuta, H., Canny, J., and Fischer, I. A human-inspired reading agent with gist memory of very long contexts. *arXiv preprint arXiv:2402.09727*, 2024.

Lehman, J. and Stanley, K. O. Abandoning objectives: Evolution through the search for novelty alone. *Evolutionary computation*, 19(2):189–223, 2011a.

Lehman, J. and Stanley, K. O. Abandoning objectives: Evolution through the search for novelty alone. *Evolutionary computation*, 19(2):189–223, 2011b.

Lehman, J., Gordon, J., Jain, S., Ndousse, K., Yeh, C., and Stanley, K. O. Evolution through large models. In *Handbook of evolutionary machine learning*, pp. 331–366. Springer, 2023.

Leike, J., Krueger, D., Everitt, T., Martic, M., Maini, V., and Legg, S. Scalable agent alignment via reward modeling: a research direction. *arXiv preprint arXiv:1811.07871*, 2018.Lin, S. Computer solutions of the traveling salesman problem. *Bell System Technical Journal*, 44(10):2245–2269, 1965.

Ma, H., Narayanaswamy, A., Riley, P., and Li, L. Evolving symbolic density functionals. *Science Advances*, 8(36): eabq0279, 2022.

MacDiarmid, M., Wright, B., Uesato, J., Benton, J., Kutasov, J., Price, S., Bouscal, N., Bowman, S., Brickson, T., Cloud, A., et al. Natural emergent misalignment from reward hacking in production rl. *arXiv preprint arXiv:2511.18397*, 2025.

Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegrefte, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., et al. Self-refine: Iterative refinement with self-feedback. *Advances in Neural Information Processing Systems*, 36:46534–46594, 2023.

Mahfoud, S. W. *Niching methods for genetic algorithms*. University of Illinois at Urbana-Champaign, 1995.

McAleese, N., Pokorny, R. M., Uribe, J. F. C., Nitishinskaya, E., Trebacz, M., and Leike, J. Llm critics help catch llm bugs. *arXiv preprint arXiv:2407.00215*, 2024.

Minka, T. P. *A family of algorithms for approximate Bayesian inference*. PhD thesis, Massachusetts Institute of Technology, 2001.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., et al. Human-level control through deep reinforcement learning. *nature*, 518(7540): 529–533, 2015.

Mouret, J.-B. and Clune, J. Illuminating search spaces by mapping elites. *arXiv preprint arXiv:1504.04909*, 2015.

Nguyen, S., Mei, Y., and Zhang, M. Genetic programming for production scheduling: a survey with a unified framework. *Complex & Intelligent Systems*, 3(1):41–66, 2017.

Novikov, A., Vū, N., Eisenberger, M., Dupont, E., Huang, P.-S., Wagner, A. Z., Shirobokov, S., Kozlovskii, B., Ruiz, F. J., Mehrabian, A., et al. Alphaevolve: A coding agent for scientific and algorithmic discovery. *arXiv preprint arXiv:2506.13131*, 2025.

Opsahl-Ong, K., Ryan, M. J., Purtell, J., Broman, D., Potts, C., Zaharia, M., and Khattab, O. Optimizing instructions and demonstrations for multi-stage language model programs. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pp. 9340–9366, 2024.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. *Advances in neural information processing systems*, 35:27730–27744, 2022.

Ouyang, S., Yan, J., Hsu, I., Chen, Y., Jiang, K., Wang, Z., Han, R., Le, L. T., Daruki, S., Tang, X., et al. Reasoningbank: Scaling agent self-evolving with reasoning memory. *arXiv preprint arXiv:2509.25140*, 2025.

Packer, C., Fang, V., Patil, S., Lin, K., Wooders, S., and Gonzalez, J. Memgpt: Towards llms as operating systems. *ArXiv*, 2023.

Pourchot, A. and Sigaud, O. Cem-rl: Combining evolutionary and gradient-based methods for policy search. *arXiv preprint arXiv:1810.01222*, 2018.

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. *Advances in neural information processing systems*, 36: 53728–53741, 2023.

Romera-Paredes, B., Barekainen, M., Novikov, A., Balog, M., Kumar, M. P., Dupont, E., Ruiz, F. J., Ellenberg, J. S., Wang, P., Fawzi, O., et al. Mathematical discoveries from program search with large language models. *Nature*, 625 (7995):468–475, 2024.

Schmidhuber, J. Gödel machines: self-referential universal problem solvers making provably optimal self-improvements. *arXiv preprint cs/0309048*, 2003.

Schmidhuber, J. Completely self-referential optimal reinforcement learners. In *International Conference on Artificial Neural Networks*, pp. 223–233. Springer, 2005.

Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz, P. Trust region policy optimization. In *International conference on machine learning*, pp. 1889–1897. PMLR, 2015.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*, 2017.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*, 2024.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., and Dean, J. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *arXiv preprint arXiv:1701.06538*, 2017.Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., and Yao, S. Reflexion: Language agents with verbal reinforcement learning. *Advances in Neural Information Processing Systems*, 36:8634–8652, 2023.

Shojaei, P., Meidani, K., Gupta, S., Farimani, A. B., and Reddy, C. K. Llm-sr: Scientific equation discovery via programming with large language models. *arXiv preprint arXiv:2404.18400*, 2024.

Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam, V., Lanctot, M., et al. Mastering the game of go with deep neural networks and tree search. *nature*, 529(7587):484–489, 2016.

Spears, W. M. Crossover or mutation? In *Foundations of genetic algorithms*, volume 2, pp. 221–237. Elsevier, 1993.

Srinivas, N., Krause, A., Kakade, S. M., and Seeger, M. Gaussian process optimization in the bandit setting: No regret and experimental design. *arXiv preprint arXiv:0912.3995*, 2009.

Stanley, K. O. and Miikkulainen, R. Evolving neural networks through augmenting topologies. *Evolutionary computation*, 10(2):99–127, 2002.

Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. F. Learning to summarize with human feedback. *Advances in neural information processing systems*, 33:3008–3021, 2020.

Sun, Z., Yu, L., Shen, Y., Liu, W., Yang, Y., Welleck, S., and Gan, C. Easy-to-hard generalization: Scalable alignment beyond human supervision. *Advances in Neural Information Processing Systems*, 37:51118–51168, 2024.

Surina, A., Mansouri, A., Quaevlieg, L., Seddas, A., Viazovska, M., Abbe, E., and Gulcehre, C. Algorithm discovery with llms: Evolutionary search meets reinforcement learning. *arXiv preprint arXiv:2504.05108*, 2025.

Sutton, R. S., McAllester, D., Singh, S., and Mansour, Y. Policy gradient methods for reinforcement learning with function approximation. *Advances in neural information processing systems*, 12, 1999.

Suzgun, M., Yuksekgonul, M., Bianchi, F., Jurafsky, D., and Zou, J. Dynamic cheatsheet: Test-time learning with adaptive memory. *arXiv preprint arXiv:2504.07952*, 2025.

Tan, D., Woodruff, A., Warncke, N., Jose, A., Riché, M., Africa, D. D., and Taylor, M. Inoculation prompting: Eliciting traits from llms during training can suppress them at test-time. *arXiv preprint arXiv:2510.04340*, 2025.

Team, K., Du, A., Gao, B., Xing, B., Jiang, C., Chen, C., Li, C., Xiao, C., Du, C., Liao, C., et al. Kimi k1. 5: Scaling reinforcement learning with llms. *arXiv preprint arXiv:2501.12599*, 2025.

Trinh, T. H., Wu, Y., Le, Q. V., He, H., and Luong, T. Solving olympiad geometry without human demonstrations. *Nature*, 625(7995):476–482, 2024.

Vose, M. D. *The simple genetic algorithm: foundations and theory*. MIT press, 1999.

Wang, R., Lehman, J., Clune, J., and Stanley, K. O. Poet: open-ended coevolution of environments and their optimized solutions. In *Proceedings of the genetic and evolutionary computation conference*, pp. 142–151, 2019.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. *Advances in neural information processing systems*, 35:24824–24837, 2022.

Whitley, D., Rana, S., and Heckendorn, R. B. The island model genetic algorithm: On separability, population size and convergence. *Journal of computing and information technology*, 7(1):33–47, 1999.

Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine learning*, 8(3):229–256, 1992.

Wu, Q., Bansal, G., Zhang, J., Wu, Y., Li, B., Zhu, E., Jiang, L., Zhang, X., Zhang, S., Liu, J., et al. Autogen: Enabling next-gen llm applications via multi-agent conversations. In *First Conference on Language Modeling*, 2024.

Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., and Zhang, Y. A-mem: Agentic memory for llm agents. *arXiv preprint arXiv:2502.12110*, 2025.

Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W., Salakhutdinov, R., and Manning, C. D. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In *Proceedings of the 2018 conference on empirical methods in natural language processing*, pp. 2369–2380, 2018.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. R., and Cao, Y. React: Synergizing reasoning and acting in language models. In *The eleventh international conference on learning representations*, 2022.

Ye, Z., Agarwal, R., Liu, T., Joshi, R., Velury, S., Le, Q. V., Tan, Q., and Liu, Y. Reward-guided prompt evolving in reinforcement learning for llms. In *Forty-second International Conference on Machine Learning*, 2025.Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al. Dapo: An open-source llm reinforcement learning system at scale. *arXiv preprint arXiv:2503.14476*, 2025.

Yudkowsky, E. Levels of organization in general intelligence. In *Artificial general intelligence*, pp. 389–501. Springer, 2007.

Zhang, J., Hu, S., Lu, C., Lange, R., and Clune, J. Darwin godel machine: Open-ended evolution of self-improving agents. *arXiv preprint arXiv:2505.22954*, 2025a.

Zhang, L., Hosseini, A., Bansal, H., Kazemi, M., Kumar, A., and Agarwal, R. Generative verifiers: Reward modeling as next-token prediction. *arXiv preprint arXiv:2408.15240*, 2024.

Zhang, Q., Hu, C., Upasani, S., Ma, B., Hong, F., Kamanuru, V., Rainton, J., Wu, C., Ji, M., Li, H., et al. Agentic context engineering: Evolving contexts for self-improving language models. *arXiv preprint arXiv:2510.04618*, 2025b.

Zheng, C., Liu, S., Li, M., Chen, X.-H., Yu, B., Gao, C., Dang, K., Liu, Y., Men, R., Yang, A., et al. Group sequence policy optimization. *arXiv preprint arXiv:2507.18071*, 2025.

Zhong, W., Guo, L., Gao, Q., Ye, H., and Wang, Y. Memorybank: Enhancing large language models with long-term memory. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 38, pp. 19724–19731, 2024.

Zhou, Y., Muresanu, A. I., Han, Z., Paster, K., Pitis, S., Chan, H., and Ba, J. Large language models are human-level prompt engineers. In *The eleventh international conference on learning representations*, 2022.

Zhu, R. J. and Qiu, Y. Ucb exploration for fixed-budget bayesian best arm identification. *arXiv preprint arXiv:2408.04869*, 2024.## A. Appendix Contents

### A.1. LLM-Based Self-Reflection for Mutation and Crossover

E-SPL implements both mutation and crossover as LLM-driven editing operators over past system prompts. The mutation operator consists of (i) a self-reflection stage that diagnoses performance from RL rollouts (which has two steps: trajectory summarization and self-critique), followed by (ii) a patch synthesis stage that applies structured edits to the parent prompt using a git-style diff (which has two steps: local edit proposals and local-to-global aggregation). On the other hand, the crossover operator has a single stage.

**Mutation is Multi-Stage.** In each RL iteration of Alg. 1, we identify the best-performing system prompt

$$k \leftarrow \arg \max_i V_i, \quad V_i = \frac{1}{N} \sum_{j=1}^N R(x, y_{i,j}), \quad (6)$$

where the returns are computed under matched conditions for all sampled prompts.

**Self-Reflection Module in Mutation** Let  $\mathcal{E}_k = \{(x, y_{k,j}, r_{k,j})\}_{j=1}^N$  denote the rollouts produced under  $s_k$ . The self-reflection prompt  $\Psi_{\text{reflect}}$  instructs the reference model  $\pi_{\text{ref}}$  to summarize these trajectories (**Trajectory Summarization**, §A.2), attribute successes and failures to specific reasoning behaviors (**Self-Critique**, §A.3), and produce lessons  $\ell$ :

$$\ell \sim \pi_{\text{ref}}(\cdot \mid \Psi_{\text{reflect}}, \mathcal{E}_k). \quad (7)$$

**Local Edit Proposal (§A.4)** Given the reflection output  $\ell$ , we generate concrete edits to the parent system prompt:

$$\text{diff} \sim \pi_{\text{ref}}(\cdot \mid s_k, \ell) \quad (8)$$

for each problem  $x$  in the batch  $\mathcal{B}$ . Each diff is a list of local edits. Now we need a mechanism to aggregate those individual per-problem edits across a batch of problems.

**Local-to-Global Aggregation (§A.5)** To reconcile overlapping edits to the system prompt proposed by parallel self-reflection processes, E-SPL aggregates these edits using a consolidation prompt that produces a global revision plan consisting of:

- • **modify**: refine an existing principle using accumulated evidence,
- • **merge**: combine semantically similar principles into a more general form.

Each resulting principle in the system prompt should be short, abstract, and strategy-focused, preventing uncontrolled growth of the system prompt length. We then apply the global editing operations:

$$s_{\text{mutate}} \leftarrow \text{git.apply}(s_k, \text{diff}) \quad (9)$$

The resulting child system prompt is added to the population with inherited rating and increased uncertainty as in Alg. 1. The diff format is especially important as system prompts grow longer.

**Crossover is Single-Stage.** The crossover operator (Alg. 2) differs from the mutation operator in the evidence used for reflection. For a batch of problems  $\mathcal{B}$ , we compute the empirical return matrix

$$\Phi_{i,b} = \frac{1}{N} \sum_{j=1}^N R(x_b, y_{i,b,j}), \quad (10)$$

by reusing rollouts already generated by RL. For each system prompt  $s_i$ , we identify which problems  $\rho_i$  in the batch  $\mathcal{B}$  that  $s_i$  did best on, yielding a list  $\varphi$  of prompt-specific wins. The crossover reflection prompt  $\Psi_{\text{cr}}$  is constructed from  $\varphi$  to contrast the differential strengths of multiple parent prompts.

The reference model then proposes edits applied to the overall top-performing prompt  $s_k$ :

$$\text{diff}_{\text{cr}} \sim \pi_{\text{ref}}(\cdot \mid s_k, \Psi_{\text{cr}}), \quad s_{\text{crossover}} \leftarrow \text{git.apply}(s_k, \text{diff}_{\text{cr}}), \quad (11)$$

producing a child prompt that selectively recombines complementary instruction segments from multiple parents.The diagram illustrates the E-SPL pipeline, which is an evolutionary system prompt learning algorithm. It starts with a **Population of System Prompts** represented as a cylinder containing  $\{(s_i, (\mu_i, \sigma_i))\}$ .

- **Select  $M$  System Prompts:** The population is used to select  $M$  system prompts:  $s_1, s_2, \dots, s_M$ .
- **Same batch of  $b$  problems:** Each selected prompt  $s_i$  is used to generate a batch of  $b$  problems.
- **Sample from  $\pi_\theta$ :** For each prompt  $s_i$ , trajectories are sampled from the policy  $\pi_\theta$  given the prompt and a batch of problems:  $\text{Trajectories} \sim \pi_\theta(\cdot | x_{1:b}, s_i)$ .
- **Performance of each  $s_i$ :** The trajectories are evaluated for **Mean Accuracy** and **Per-problem Accuracy**.
- **Sort Mean Accuracy to get relative rankings:** The mean accuracy for each prompt is sorted to determine relative rankings.
- **Find problems where each  $s_i$  was the top prompt:** Problems are identified where each prompt  $s_i$  achieved the highest accuracy.
- **Self Reflection Module in Mutation:** The top system prompt from the batch,  $s_k$ , is used for self-reflection to generate a **New  $s_{mutate}$** .
- **Self Reflection Module in Crossover:** Problems where each  $s_i$  was the top prompt are used for self-reflection to generate a **New  $s_{crossover}$** .
- **Update TrueSkill ratings:** The performance results are used to update the TrueSkill ratings.
- **Add new system prompt from mutation:** The new prompt  $s_{mutate}$  is added to the population.
- **Add new system prompt from crossover:** The new prompt  $s_{crossover}$  is added to the population.
- **Adjust ratings:** The updated TrueSkill ratings are used to adjust the ratings of the prompts in the population.
- **Policy Gradient Update to  $\pi_\theta$ :** The final step is a policy gradient update to the policy  $\pi_\theta$  based on the performance of the prompts.

Feedback loops are shown for **Add new system prompt** (blue arrows) and **Adjust ratings** (green arrow).

Figure 17. Detailed overview of the E-SPL pipeline, which illustrates how Figure 2, Figure 5, and Figure 6 fit into a single algorithm.

### A.2. Trajectory Summarization in Mutation Self-Reflection

Before proposing local edits, E-SPL first converts raw RL rollouts into structured trajectory summaries. For the selected parent prompt  $s_k$ , all sampled solutions are grouped by problem instance  $x$ . Let  $\mathcal{R}_x = \{(x, y_{k,j}, r_{k,j})\}_{j=1}^N$  denote the group of rollouts produced for problem  $x$  under  $s_k$ . E-SPL only performs self-reflection on problems where the model has both made some mistakes and achieved some success ( $0 < V_i < 1$ ). Self-reflection focuses on cases where we can *contrast* successes and failures under identical instructions.

For each rollout in a problem-specific group, the reference model  $\pi_{\text{ref}}$  is prompted to generate a step-by-step summary describing: (i) the reasoning steps taken, (ii) which principles or strategies in the system prompt were (implicitly) used, and (iii) where detours or errors occurred, given the rollout outcome and ground-truth answer.

These summaries preserve the full reasoning outcomes while explicitly highlighting failure modes, producing a compact representation suitable for higher-level self-critique.

### A.3. Self-Critique in Mutation Self-Reflection

For each problem  $x$ , E-SPL constructs a self-critique prompt that jointly presents:

- • multiple trajectory summaries from  $\mathcal{R}_x$ , labeled as successful or incorrect,
- • the current system prompt  $s_k$ ,
- • the ground-truth answer.```

graph LR
    subgraph "Self Reflection Module in Mutation"
        direction LR
        P1[Problem x1] --> AT1[Agent Trajectories ~ pi(. | sk, x1)]
        AT1 --> TS1[Trajectory Summarization]
        TS1 --> SC1[Self Critique]
        SC1 --> PLE1[Propose Local Edits]
        P2[Problem x2] --> AT2[Agent Trajectories ~ pi(. | sk, x2)]
        AT2 --> TS2[Trajectory Summarization]
        TS2 --> SC2[Self Critique]
        SC2 --> PLE2[Propose Local Edits]
        P3[... Problem xb] --> AT3[Agent Trajectories ~ pi(. | sk, xb)]
        AT3 --> TS3[Trajectory Summarization]
        TS3 --> SC3[Self Critique]
        SC3 --> PLE3[Propose Local Edits]
        PLE1 --> LGA[Local-to-Global Aggregation]
        PLE2 --> LGA
        PLE3 --> LGA
        LGA --> E[Edits to sk]
    end
  
```

Figure 18. **Self-Reflection Module in Mutation** (where the best-performing system prompt  $s_k$  is edited) consists of four sequential modules: trajectory summarization (§A.2), self-critique (§A.3), local edit proposal (§A.4), local-to-global aggregation (§A.5).

The reference model  $\pi_{\text{ref}}$  is instructed to analyze these contrasted trajectories to:

1. 1. identify reasoning patterns that consistently led to success,
2. 2. diagnose systematic failure modes, and consider additional guidance that would be needed to avoid those failure modes,
3. 3. determine whether existing principles in the system prompt are insufficient or require further editing.

Based on this analysis, the model proposes a number of structured update operations:

- • **add**: introduce a new general principle (or high-level strategy / repeatable workflow) to the system prompt;
- • **modify**: refine an existing principle to better encode successful strategies or avoid observed errors.

Each proposed principle is designed to express an abstract decision heuristic rather than problem-specific calculations, ensuring transferability across tasks.

#### A.4. Local Edit Proposal Module

To stabilize evolution and prevent overly aggressive prompt drift, E-SPL limits the number of edits proposed per problem instance. Specifically, each per-problem self-critique stage is restricted to produce at most  $K_{\text{ops}}$  edit operations, where  $K_{\text{ops}}$  is a tunable hyperparameter (set to  $K_{\text{ops}} = 2$  in our experiments). The goal of this design is to encourage incremental knowledge accumulation and localized corrections driven by concrete failures.

Empirically, bounding the number of local edits encourages steady refinement of the principle set in the system prompt while preserving useful structure learned in earlier generations.

#### A.5. Local-to-Global Aggregation Module

During a training step, multiple problems may independently propose local edits. Directly applying all such edits can lead to conflicting updates, redundancy, or uncontrolled growth of the principle set in the system prompt.

To address this, E-SPL performs a batch aggregation stage that selectively combines the local edit proposals into a global revision plan. Given the selected system prompt  $s_k$  and the collection of proposed edits, the reference model is prompted to:

- • refine individual edits to incorporate repeated evidence,
- • merge semantically overlapping principles or strategies into more general formulations.

The resulting revision plan consists of two types of operations:- • **modify** operation on an existing principle, which aggregates all proposed edits to that principle across the batch,
- • **merge** operation, which replaces several related principles with a unified higher-level strategy / more universal rule across domains.

All final principles in the system prompt are encouraged to be short, abstract, and strategy-oriented, yielding a compact declarative knowledge base that evolves steadily throughout training.

We use the following hyper-parameters:

- • LoRA (Hu et al., 2022) rank of 32, learning rate  $\alpha=4e-5$ , using Adam optimizer (Kingma, 2014) with  $\beta = (0.9, 0.95)$ .
- • Batch size of 10 (problems per batch). Group size  $N = 5$  for the number of rollouts to sample per system prompt.
- • Sliding window size  $K = 10$  for the population  $\mathcal{S}$ . Number of system prompts per batch  $M = 3$ .
- • Uncertainty in genetic operators:  $\Delta\sigma = 1$ , uncertainty bonus:  $\lambda = 2.0$ ; the crossover probability  $p_{\text{crossover}} = 0.2$ .
- • TrueSkill (all defaults):  $\mu_0 = 25$ ,  $\sigma_0 = 25/3$ , beta=25/6, tau=25/300, draw probability=0.10.
- • We find that a simplified selection strategy is quite effective: always add the system prompt with the highest score ( $\text{score}_i = \mu_i + \lambda\sigma_i$ ) to the batch, but randomly sample from the rest of the sliding window.
- •  $K_{\text{ops}} = 2$ , where  $K_{\text{ops}}$  is the maximum number of local edits per problem in the mutation operator.

### Root node system prompt for agentic search:

```
SEARCH_TASK_INSTRUCTIONS = """You are an expert assistant who solves tasks using a Wikipedia search tool.
```

Here are instructions for how to solve a problem:

1. 1. Think step by step before calling the tool and after you receive the result of the tool call. Decide what queries to call the tool with.
2. 2. Call the tool with the queries you have decided on.
3. 3. Think step by step again after you receive the result of the tool call. If you have the information you need, you can stop here.
4. 4. Otherwise, come up with new queries that combine information from the previous results.
5. 5. Include your final answer after the "Answer:" prefix. The answer should be between one to five words.

Here is an example of solving a real question:

```
"Between 2020 and 2025, which year did New York City see the most population growth and how did San Francisco population change in that year?"
```

1. 1. Think step by step: In order to answer this question, I need to know the population of New York City and San Francisco between 2020 and 2025. I will search for the population of New York City in each year
2. 2. Calling search tool: `<tool_call>{"name": "search", "arguments": {"query_list": ["Population New York city between 2020 and 2025"]}}</tool_call>` (Output omitted for brevity)
3. 3. Think step by step again: I have the population of New York City in each year, and I see that the population of New York City grew the most in 2024. I need to know the population of San Francisco in 2024. I will search for the population of San Francisco in each year.  
   `<tool_call>{"name": "search", "arguments": {"query_list": ["Population San Francisco between 2023 and 2024"]}}</tool_call>` (Output omitted for brevity)
4. 4. Answer: The population of New York City grew the most in 2024, and the population of San Francisco changed by XXXX in 2024.

```
"""
```## B. TrueSkill-Based Fitness Modeling for E-SPL

To aggregate noisy batch-wise prompt comparisons into persistent competence estimates, we adopt the Bayesian skill rating model **TrueSkill** (Herbrich et al., 2006). Each system prompt is treated as a player whose latent competence evolves over time and is updated through relative performance evidence obtained within each RL iteration. Unless otherwise stated, we use default TrueSkill hyperparameters:  $\mu_0 = 25.0$ ,  $\sigma_0 = 25.0/3$ ,  $\beta = 25.0/6$ ,  $\tau = 25.0/300$ , and draw probability parameter  $p_{\text{draw}} = 0.1$ . **For completeness**, we review the mathematical details of an off-the-shelf TrueSkill system below.

**Skill and Performance Model.** Each system prompt  $s_i$  is associated with a Gaussian skill variable

$$z_i \sim \mathcal{N}(\mu_i, \sigma_i^2),$$

where  $\mu_i$  denotes estimated competence and  $\sigma_i$  its uncertainty. Observed performance is modeled as a noisy skill:

$$p_i = z_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \beta^2),$$

so that the marginal performance variance is

$$\text{Var}(p_i) = \sigma_i^2 + \beta^2.$$

**Dynamics.** Before each rating update, we apply Gaussian drift  $\tau$  in skill space to model temporal uncertainty growth:

$$\sigma_i^2 \leftarrow \sigma_i^2 + \tau^2.$$

**Ranking Observations and Tie Encoding.** Within each RL iteration, matched rollouts yield scalar returns  $\{V_i\}_{i=1}^M$  for selected prompts. Sorting these values produces a ranking from best to worst. Ties are encoded by an integer array  $\text{ties}$  such that adjacent positions  $i$  and  $i + 1$  are treated as a draw if  $\text{ties}[i] = \text{ties}[i + 1]$ , and otherwise as a win for position  $i$  over  $i + 1$ . This yields a chain of  $M - 1$  adjacent pairwise constraints.

**Draw Margin.** We first compute a global draw margin in performance-difference units:

$$\epsilon_{\text{abs}} = \Phi^{-1}\left(\frac{p_{\text{draw}} + 1}{2}\right) \sqrt{2} \beta.$$

We denote  $\phi(\cdot)$  and  $\Phi(\cdot)$  as the probability density function (pdf) and cumulative distribution function (cdf) of the standard normal distribution  $\mathcal{N}$ ; the  $\Phi^{-1}(\cdot)$  function above is the inverse (probit) function of  $\Phi$ .

For each pairwise comparison with total standard deviation  $c$ , the normalized margin used in inference is

$$\epsilon = \frac{\epsilon_{\text{abs}}}{c}.$$

**Truncated Gaussian Correction Functions.** For a normalized performance difference variable

$$t = \frac{\mu_i - \mu_j}{c},$$

the one-sided (win/loss) correction terms are

$$\begin{aligned} v_{\text{win}}(t, \epsilon) &= \frac{\phi(t - \epsilon)}{\Phi(t - \epsilon)}, \\ w_{\text{win}}(t, \epsilon) &= v_{\text{win}}(t, \epsilon)(v_{\text{win}}(t, \epsilon) + t - \epsilon), \end{aligned}$$

and the two-sided (draw) terms are

$$\begin{aligned} v_{\text{draw}}(t, \epsilon) &= \frac{\phi(-\epsilon - t) - \phi(\epsilon - t)}{\Phi(\epsilon - t) - \Phi(-\epsilon - t)}, \\ w_{\text{draw}}(t, \epsilon) &= v_{\text{draw}}^2 + \frac{(\epsilon - t)\phi(\epsilon - t) - (-\epsilon - t)\phi(-\epsilon - t)}{\Phi(\epsilon - t) - \Phi(-\epsilon - t)}. \end{aligned}$$

where  $\phi$  and  $\Phi$  denote the standard normal pdf and cdf.**Approximate Inference for Full Rankings.** For rankings with more than two prompts, we perform EP-style iterative message passing (Minka, 2001) over adjacent constraints in performance space.

**Performance-space priors.** For each system prompt  $s_i$  and its corresponding rating  $\mathcal{N}(\mu_i, \sigma_i^2)$ , define

$$v_i = \sigma_i^2 + \beta^2, \quad \pi_i^{(0)} = \frac{1}{v_i}, \quad \tau_i^{(0)} = \frac{\mu_i}{v_i}.$$

**Messages.** For each adjacent constraint connecting system prompts  $i = k$  and  $j = k + 1$ , we maintain Gaussian messages  $m_{k \rightarrow i} = (\pi_{k \rightarrow i}, \tau_{k \rightarrow i})$  and  $m_{k \rightarrow j}$ , initialized to zero.

**Cavity distributions.** When updating for the  $k$ -th constraint, cavity parameters are formed by summing all other incoming messages:

$$\pi_i^{\setminus k} = \pi_i^{(0)} + \sum_{c \neq k} \pi_{c \rightarrow i}, \quad \tau_i^{\setminus k} = \tau_i^{(0)} + \sum_{c \neq k} \tau_{c \rightarrow i},$$

and similarly for  $j$ . The corresponding cavity moments are

$$\mu_i^{\setminus k} = \frac{\tau_i^{\setminus k}}{\pi_i^{\setminus k}}, \quad \sigma_{i, \setminus k}^2 = \frac{1}{\pi_i^{\setminus k}}.$$

**Moment update.** Let

$$c = \sqrt{\sigma_{i, \setminus k}^2 + \sigma_{j, \setminus k}^2}, \quad t = \frac{\mu_i^{\setminus k} - \mu_j^{\setminus k}}{c}, \quad \epsilon = \frac{\epsilon_{\text{abs}}}{c}.$$

If the pair is a draw, use  $(v_{\text{draw}}, w_{\text{draw}})$ ; otherwise use  $(v_{\text{win}}, w_{\text{win}})$ . The updated performance-space moments are

$$\begin{aligned} \mu_i^{\text{post}} &= \mu_i^{\setminus k} + \frac{\sigma_{i, \setminus k}^2}{c} v, & \mu_j^{\text{post}} &= \mu_j^{\setminus k} - \frac{\sigma_{j, \setminus k}^2}{c} v, \\ \sigma_{i, \text{post}}^2 &= \sigma_{i, \setminus k}^2 \left( 1 - \frac{\sigma_{i, \setminus k}^2}{c^2} w \right), & \sigma_{j, \text{post}}^2 &= \sigma_{j, \setminus k}^2 \left( 1 - \frac{\sigma_{j, \setminus k}^2}{c^2} w \right). \end{aligned}$$

**Message extraction.** New messages are computed as the difference between posterior and cavity natural parameters:

$$\begin{aligned} \pi_{k \rightarrow i} &= \max \left( 0, \frac{1}{\sigma_{i, \text{post}}^2} - \pi_i^{\setminus k} \right), \\ \tau_{k \rightarrow i} &= \frac{\mu_i^{\text{post}}}{\sigma_{i, \text{post}}^2} - \tau_i^{\setminus k}, \end{aligned}$$

and analogously for  $j$ . Precision is clamped to be nonnegative for numerical stability.

These updates are iterated over all adjacent constraints for a fixed number of sweeps or until message changes fall below a small threshold.

**Final performance posteriors.** After convergence,

$$\pi_i^{\text{perf}} = \pi_i^{(0)} + \sum_k \pi_{k \rightarrow i}, \quad \tau_i^{\text{perf}} = \tau_i^{(0)} + \sum_k \tau_{k \rightarrow i},$$

yielding

$$\mu_i^{\text{perf}} = \frac{\tau_i^{\text{perf}}}{\pi_i^{\text{perf}}}, \quad \sigma_{i, \text{perf}}^2 = \frac{1}{\pi_i^{\text{perf}}}.$$**Mapping Back to Skill Space.** Let the skill prior after dynamics be  $\mathcal{N}(\mu_i, \sigma_i^2)$  and define  $c_i = \sigma_i^2 + \beta^2$ . Performance-space natural parameters for posterior and prior are

$$\pi_{\text{post}} = \frac{1}{\sigma_{i,\text{perf}}^2}, \quad \tau_{\text{post}} = \frac{\mu_i^{\text{perf}}}{\sigma_{i,\text{perf}}^2}, \quad \pi_{\text{prior}} = \frac{1}{c_i}, \quad \tau_{\text{prior}} = \frac{\mu_i}{c_i}.$$

The truncation-induced message is

$$\pi_{\text{msg}} = \pi_{\text{post}} - \pi_{\text{prior}}, \quad \tau_{\text{msg}} = \tau_{\text{post}} - \tau_{\text{prior}}.$$

This message is mapped through the performance-noise link as

$$\pi_{\text{msg}}^{\text{skill}} = \frac{\pi_{\text{msg}}}{1 + \beta^2 \pi_{\text{msg}}}, \quad \tau_{\text{msg}}^{\text{skill}} = \frac{\tau_{\text{msg}}}{1 + \beta^2 \pi_{\text{msg}}},$$

when  $\pi_{\text{msg}} > 0$ . The skill posterior is then

$$\pi_i^{\text{skill}} = \frac{1}{\sigma_i^2} + \pi_{\text{msg}}^{\text{skill}}, \quad \tau_i^{\text{skill}} = \frac{\mu_i}{\sigma_i^2} + \tau_{\text{msg}}^{\text{skill}},$$

with moments

$$\sigma_i^2 = \frac{1}{\pi_i^{\text{skill}}}, \quad \mu_i = \frac{\tau_i^{\text{skill}}}{\pi_i^{\text{skill}}}.$$

**Selection Scores.** Prompt sampling is guided by an upper confidence estimate (with  $\lambda = 2.0$ ) (Srinivas et al., 2009; Kaufmann et al., 2012; Kalyanakrishnan et al., 2012; Zhu & Qiu, 2024):

$$\text{score}_i = \mu_i + \lambda \sigma_i$$

**Initialization for Mutation and Crossover.** A mutated child prompt inherits its parent's mean while inflating uncertainty:

$$\mu_{\text{child}} = \mu_k, \quad \sigma_{\text{child}} = \sqrt{\sigma_k^2 + \Delta \sigma^2}.$$

For crossover among parents  $\{i\}$ , we perform precision-weighted fusion:

$$\mu' = \frac{\sum_i \mu_i / \sigma_i^2}{\sum_i 1 / \sigma_i^2}, \quad \sigma'^2 = \frac{1}{\sum_i 1 / \sigma_i^2} + \Delta \sigma^2.$$

Each RL iteration supplies local comparative evidence that is accumulated into a coherent evolutionary fitness landscape.## C. Explanations for the identified mistakes in discovered principles

### C.1. Mistake #1 in Discovered Principles

**Original statement.** For a polynomial  $P(n) = \prod_{i=1}^m (n - a_i)$ , the largest integer  $d$  that divides  $P(n)$  for all integers  $n$  is  $d = \prod_p p^{e_p}$ , where  $e_p = \min_{r \bmod p} (\#\{i : a_i \equiv r \pmod{p}\})$ . Only primes  $p \leq \max(a_i)$  need be considered, as for larger  $p$ , the  $a_i$  are distinct modulo  $p$ , so  $e_p = 0$ . For primes where the function mapping  $i$  to  $a_i \bmod p$  is a permutation (e.g., if  $a_i = i^3$  and  $p \equiv 2 \bmod 3$ ),  $e_p = \lfloor \frac{m}{p} \rfloor$ .

**Correction.** For a polynomial  $P(n) = \prod_{i=1}^m (n - a_i)$ , the largest integer  $d$  that divides  $P(n)$  for all integers  $n$  is  $d = \prod_p p^{e_p}$ , where  $e_p = \min_{n \in \mathbb{Z}} v_p(P(n))$ . Moreover, one always has the lower bound  $e_p \geq \min_{r \bmod p} (\#\{i : a_i \equiv r \pmod{p}\})$ . In particular, for this mod- $p$  bound it suffices to consider primes  $p \leq m$ , since if  $p > m$  then some residue class occurs zero times and the minimum is 0. For primes where the function mapping  $i$  to  $a_i \bmod p$  is a permutation (e.g., if  $a_i = i^3$  and  $p \equiv 2 \bmod 3$ ), the bound gives  $e_p \geq \lfloor \frac{m}{p} \rfloor$ .

**Explanation of the Mistake** We first recall the original principle. Let

$$P(n) = \prod_{i=1}^m (n - a_i),$$

where  $a_1, \dots, a_m \in \mathbb{Z}$ . It was claimed that the largest integer dividing  $P(n)$  for all  $n \in \mathbb{Z}$  is

$$d = \prod_p p^{e_p}, \quad e_p = \min_{r \bmod p} \#\{i : a_i \equiv r \pmod{p}\}.$$

This expression counts, for each prime  $p$ , the minimum number of factors  $(n - a_i)$  that are divisible by  $p$  for any choice of  $n$ .

We now recall the corrected formulation. For a prime  $p$  and a nonzero integer  $k$ , define the  $p$ -adic valuation  $v_p(k)$  by

$$v_p(k) = \max\{e \geq 0 : p^e \mid k\}.$$

(For example,  $v_2(12) = 2$  since  $12 = 2^2 \cdot 3$ .) Using this notation, the exact exponent of  $p$  dividing all values of  $P(n)$  is

$$e_p = \min_{n \in \mathbb{Z}} v_p(P(n)).$$

Since

$$v_p(P(n)) = \sum_{i=1}^m v_p(n - a_i),$$

this definition measures the total number of powers of  $p$  dividing the product, counting higher powers such as  $p^2, p^3$ , and so on.

The original counting argument corresponds to considering only divisibility modulo  $p$ . Fix a residue class  $r \bmod p$  and choose  $n \equiv r \pmod{p}$ . Then

$$p \mid (n - a_i) \iff a_i \equiv r \pmod{p}.$$

Hence, for such  $n$ ,

$$v_p(P(n)) \geq \#\{i : a_i \equiv r \pmod{p}\}.$$

Minimizing over all residue classes  $r \bmod p$  yields the general lower bound

$$e_p = \min_{n \in \mathbb{Z}} v_p(P(n)) \geq \min_{r \bmod p} \#\{i : a_i \equiv r \pmod{p}\}.$$

This inequality explains why the correction is necessary. The original formula accounts only for the number of factors divisible by  $p$ , but it does not capture the possibility that some factors  $(n - a_i)$  may be divisible by higher powers  $p^2, p^3, \dots$ , which contribute additional terms to  $v_p(P(n))$ . Such higher-power divisibility depends on congruences modulo  $p^2, p^3$ , andcannot be detected by working modulo  $p$  alone. Therefore, the original expression cannot determine  $e_p$  exactly in general, but it always provides a valid lower bound.

Finally, the restriction to primes  $p \leq m$  follows from a counting argument. If  $p > m$ , then among the  $p$  residue classes modulo  $p$  and the  $m$  integers  $a_i$ , at least one residue class is not attained. Choosing  $n$  in that class gives

$$v_p(n - a_i) = 0 \quad \text{for all } i,$$

and hence  $v_p(P(n)) = 0$ . Thus, no factor of  $p$  is forced to divide  $P(n)$  in this case, and the mod- $p$  lower bound vanishes.

**Summary.** The original mistake arises from conflating divisibility by  $p$  with divisibility by higher powers of  $p$ . Working modulo  $p$  correctly identifies how many factors of  $(n - a_i)$  are *forced* to contribute at least one factor of  $p$ , and in many well-distributed cases this already determines the full exponent. However, the true fixed divisor depends on congruences modulo  $p^k$  for all  $k \geq 1$ , which are invisible at the mod- $p$  level.

## C.2. Mistake #2 in Discovered Principles

**Original statement.** *For graphs with odd-degree vertices, find a minimum  $T$ -join by pairing odd vertices along efficient paths. In symmetric graphs such as grids, pair boundary vertices optimally to minimize total length.*

**Correction.** *For graphs with odd-degree vertices, find a minimum  $T$ -join by pairing odd vertices via a minimum-weight perfect matching on shortest-path distances. In symmetric graphs such as grids, restrict the matching to symmetry-respecting pairings to minimize total length.*

**Background and notation.** Let  $G = (V, E)$  be an undirected graph with a nonnegative edge-length function  $c : E \rightarrow \mathbb{R}_{\geq 0}$ . For a vertex  $v \in V$ , its degree  $\deg(v)$  is the number of incident edges. A vertex is *odd* if  $\deg(v)$  is odd.

Let  $T \subseteq V$  be a designated set of vertices with  $|T|$  even. A  *$T$ -join* is a multiset of edges  $J$  such that in the subgraph induced by  $J$ , the vertices of odd degree are *exactly* the vertices in  $T$ . The *minimum  $T$ -join* problem is to find such a  $J$  minimizing  $\sum_{e \in J} c(e)$  (counting multiplicity if edges repeat).

For  $u, v \in V$ , let  $d(u, v)$  denote the shortest-path distance in  $G$  under edge costs  $c$ . Consider the complete graph on vertex set  $T$ , with edge weights  $d(u, v)$ . A *perfect matching* on  $T$  partitions  $T$  into disjoint pairs; a *minimum-weight perfect matching* minimizes the sum of the chosen pair weights.

**Explanation of the Mistake** The original statement is incorrect as written because “pairing odd vertices along efficient paths” can be a local or greedy rule (e.g., repeatedly pairing “nearby” odd vertices). Such local decisions are not guaranteed to be globally optimal: choosing a short connection for one pair can force much longer connections for the remaining vertices, increasing the total length. A minimum  $T$ -join is inherently a *global* optimization problem over all pairings of  $T$ .

The correct principle is: compute all shortest-path distances  $d(u, v)$  for  $u, v \in T$ , then solve a minimum-weight perfect matching on  $T$  using these distances. Finally, for each matched pair  $(u, v)$ , add the corresponding shortest path in  $G$ ; the symmetric difference of these paths forms a minimum  $T$ -join. In symmetric graphs (e.g., grids), symmetry can reduce the search space by allowing one to focus on *symmetry-respecting* pairings, but optimality still comes from minimizing the total matching cost under shortest-path distances rather than from ad-hoc or boundary-only pairings.### C.3. Discovered System Prompt on Agentic Search with gpt-oss-120b

You are an expert research assistant. Your ONLY job to answer the user's question by **searching the web** and returning the **exact phrase** found in an authoritative source.

\*\*\* Workflow (follow *exactly* in this order)

1. 0. \*\*\* INTERNAL NOTE\*\*\* - Disambiguate the question if needed  
   Write a comment that will NOT be sent to the user, e.g  
   # chosen: Weekly Shonen Jump publisher
2. 1. **Predict target phrase** - Think step-by-step and write the exact wording you expect to see verbatim in double quotes.  
   Example: Target phrase: "Shueisha"
3. 2. **Create ONE search query** that **contains the predicted phrase** (of a very close paraphrase) inside double quotes\*\*  
   Example: `Query: "\"Weekly Shōnen Jump\" \"Shueisha\"`
4. 3. **Call the search tool once** you may call it a second time **only if every result from the first call is NOT VERIFIED**).  
   Use **exactly** this syntax, with no extra spaces or line-breaks inside the tags:  
   `<tool\_call>{"name": "search", "arguments": {"query\_list": ["<your query>"]}}</tool\_call>`
5. 4. **Verification block** - For each result returned, output the following four lines **in this order** and **nothing else**:
    

   ```
   ---
   Source: <domain>
   Summary: <one-sentence summary>
   Status: VERIFIED (or NOT VERIFIED)
   Phrase: "<exact phrase>" (include only if Status is VERIFIED)
   ```

   - - A result is **VERIFIED** **only if**
     - a) the exact target phrase appears verbatim in the snippet **and**
     - b) the domain ends with \*.org \*.edu, \*.gov, is Wikipedia, **or** is a major reputable news outlet (e.g., nytimes.com, bbc.com, reuters.com, theguardian.com)
   - - If the domain is not on the list, mark the result **NOT VERIFIED** even if the phrase is present.
6. 5. **Refine query (optional)** - If **all** results are NOT VERIFIED, you may modify the query (still a single string) and repeat step 3.  
   You may do this once only\*\* (maximum two distinct queries)
7. 6. **Final answer** as you have **at least one VERIFIED phrase**, output **exactly one line** and nothing else:  
   `Answer: <extracted phrase>`
   - - The phrase must contain **1 - 5 words** (words are separated by spaces)
8. 7. **Fallback** - If after two distinct queries no VERIFIED phrase is found, output exactly:  
   `Answer: Not found`

### Critical formatting rules (must be obeyed)

- - All tool calls must be wrapped **exactly** as `<tool\_call>{...}</tool\_call>` with no extra whitespace.
- - The only content sent to the user after the verification steps is the single "Answer: " line.
- - Do **not** include bullet points, markdown, blank lines, or any other text outside the structures described above.

