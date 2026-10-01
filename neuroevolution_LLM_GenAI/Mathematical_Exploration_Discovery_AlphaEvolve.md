# MATHEMATICAL EXPLORATION AND DISCOVERY AT SCALE

BOGDAN GEORGIEV, JAVIER GÓMEZ-SERRANO, TERENCE TAO, AND ADAM ZSOLT WAGNER

**ABSTRACT.** `AlphaEvolve`, introduced in [224], is a generic evolutionary coding agent that combines the generative capabilities of LLMs with automated evaluation in an iterative evolutionary framework that proposes, tests, and refines algorithmic solutions to challenging scientific and practical problems. In this paper we showcase `AlphaEvolve` as a tool for autonomously discovering novel mathematical constructions and advancing our understanding of long-standing open problems.

To demonstrate its breadth, we considered a list of 67 problems spanning mathematical analysis, combinatorics, geometry, and number theory. The system rediscovered the best known solutions in most of the cases and discovered improved solutions in several. In some instances, `AlphaEvolve` is also able to *generalize* results for a finite number of input values into a formula valid for all input values. Furthermore, we are able to combine this methodology with `Deep Think` [149] and `AlphaProof` [148] in a broader framework where the additional proof-assistants and reasoning systems provide automated proof generation and further mathematical insights.

These results demonstrate that large language model-guided evolutionary search can autonomously discover mathematical constructions that complement human intuition, at times matching or even improving the best known results, highlighting the potential for significant new ways of interaction between mathematicians and AI systems. We present `AlphaEvolve` as a powerful tool for mathematical discovery, capable of exploring vast search spaces to solve complex optimization problems at scale, often with significantly reduced requirements on preparation and computation time.

## 1. INTRODUCTION

The landscape of mathematical discovery has been fundamentally transformed by the emergence of computational tools that can autonomously explore mathematical spaces and generate novel constructions [56, 120, 242, 291]. `AlphaEvolve` (see [224]) represents a step in this evolution, demonstrating that large language models, when combined with evolutionary computation and rigorous automated evaluation, can discover explicit constructions that either match or improve upon the best-known bounds to long-standing mathematical problems, at large scales.

`AlphaEvolve` is not a general-purpose solver for all types of mathematical problems; it was primarily designed to attack problems in which a key objective is to construct a complex mathematical object that satisfies good quantitative properties, such as obeying a certain inequality with a good numerical constant. In this follow-up paper, we report on our experiments testing the performance of `AlphaEvolve` on a wide variety of such problems, primarily in the areas of analysis, combinatorics, and geometry. In many cases, the constructions provided by `AlphaEvolve` were not merely numerical in nature, but can be interpreted and generalized by human mathematicians, by other tools such as `Deep Think`, and even by `AlphaEvolve` itself. `AlphaEvolve` was not able to match or exceed previous results in all cases, and some of the individual improvements it was able to achieve could likely also have been matched by more traditional computational or theoretical methods performed by human experts. However, in contrast to such methods, we have found that `AlphaEvolve` can be readily scaled up to study large classes of problems at a time, without requiring extensive expert supervision for each new problem. This demonstrates that evolutionary computational approaches can systematically explore the space of mathematical objects in ways that complement traditional techniques, thus helping answer questions about the relationship between computational search and mathematical existence proofs.

We have also seen that in many cases, besides the scaling, in order to get `AlphaEvolve` to output comparable results to the literature and in contrast to traditional ways of doing mathematics, very little overhead is needed:

---

The authors are listed in alphabetical order.on average the usual preparation time for the setup of a problem using AlphaEvolve took only up to a few hours. We expect that without prior knowledge, information or code, an equivalent traditional setup would typically take significantly longer. This has led us to use the term *constructive mathematics at scale*.

A crucial mathematical insight underlying AlphaEvolve’s effectiveness is its ability to operate across multiple levels of abstraction simultaneously. The system can optimize not just the specific parameters of a mathematical construction, but also the algorithmic strategy for discovering such constructions. This meta-level evolution represents a new form of recursion where the optimization process itself becomes the object of optimization. For example, AlphaEvolve might evolve a program that uses a set of heuristics, a SAT solver, a second order method without convergence guarantee, or combinations of them. This hierarchical approach is particularly evident in AlphaEvolve’s treatment of complex mathematical problems (suggested by the user), where the system often discovers specialized search heuristics for different phases of the optimization process. Early-stage heuristics excel at making large improvements from random or simple initial states, while later-stage heuristics focus on fine-tuning near-optimal configurations. This emergent specialization mirrors the intuitive approaches employed by human mathematicians.

**1.1. Comparison with [224].** The white paper [224] introduced AlphaEvolve and highlighted its general broad applicability, including to mathematics and including some details of our results. In this follow-up paper we expand on the list of considered mathematical problems in terms of their breadth, hardness, and importance, and we now give full details for all of them. The problems below are arranged in no particular order. For reasons of space, we do not attempt to exhaustively survey the history of each of the problems listed here, and refer the reader to the references provided for each problem for a more in-depth discussion of known results.

Along with this paper, we will also release a live [Repository of Problems](#) with code containing some experiments and extended details of the problems. While the presence of randomness in the evolution process may make reproducibility harder, we expect our results to be fully reproducible with the information given and enough experiments.

**1.2. AI and Mathematical Discovery.** The emergence of artificial intelligence as a transformative force in mathematical discovery has marked a paradigm shift in how we approach some of mathematics’ most challenging problems. Recent breakthroughs [87, 165, 97, 77, 296, 6, 271, 295] have demonstrated AI’s capability to assist mathematicians. AlphaGeometry solved 25 out of 30 Olympiad geometry problems within standard time limits [287]. AlphaProof and AlphaGeometry 2 [148] achieved silver-medal performance at the 2024 International Mathematical Olympiad followed by a gold-medal performance of an advanced Gemini Deep Think framework at the 2025 International Mathematical Olympiad [149]. See [297] for a gold-medal performance by a model from OpenAI. Beyond competition performance, AI has begun making genuine mathematical discoveries, as demonstrated by FunSearch [242], discovering new solutions to the cap set problem and more effective bin-packing algorithms (see also [100]), or PatternBoost [56] disproving a 30-year old conjecture (see also [291]), or precursors such as Graffiti [119] generating conjectures. Other instances of AI helping mathematicians are for example [70, 283, 302, 301], in the context of finding formal and informal proofs of mathematical statements. While AlphaEvolve is geared more towards exploration and discovery, we have been able to pipeline it with other systems in a way that allows us not only to explore but also to combine our findings with a mathematically rigorous proof as well as a formalization of it.

**1.3. Evolving Algorithms to Find Constructions.** At its core, AlphaEvolve is a sophisticated search algorithm. To understand its design, it is helpful to start with a familiar idea: local search. To solve a problem like finding a graph on 50 vertices with no triangles and no cycles of length four, and the maximum number of edges, a standard approach would be to start with a random graph, and then iteratively make small changes (e.g., adding or removing an edge) that improve its score (in this case, the edge count, penalized for any triangles or four-cycles). We keep ‘hill-climbing’ until we can no longer improve.<table border="1">
<thead>
<tr>
<th>FunSearch [242]</th>
<th>AlphaEvolve [224]</th>
</tr>
</thead>
<tbody>
<tr>
<td>evolves single function</td>
<td>evolves entire code file</td>
</tr>
<tr>
<td>evolves up to 10-20 lines of code</td>
<td>evolves up to hundreds of lines of code</td>
</tr>
<tr>
<td>evolves code in Python</td>
<td>evolves any language</td>
</tr>
<tr>
<td>needs fast evaluation (<math>\leq 20</math>min on 1 CPU)</td>
<td>can evaluate for hours, in parallel, on accelerators</td>
</tr>
<tr>
<td>millions of LLM samples used</td>
<td>thousands of LLM samples suffice</td>
</tr>
<tr>
<td>small LLMs used; no benefit from larger</td>
<td>benefits from SotA LLMs</td>
</tr>
<tr>
<td>minimal context (only previous solutions)</td>
<td>rich context and feedback in prompts</td>
</tr>
<tr>
<td>optimizes single metric</td>
<td>can simultaneously optimize multiple metrics</td>
</tr>
</tbody>
</table>

TABLE 1. Capabilities and typical behaviors of AlphaEvolve and FunSearch. Table reproduced from [224].

The first key idea, inherited from AlphaEvolve’s predecessor, FunSearch [242] (see Table 1 for a head to head comparison) and its reimplementation [100], is to perform this local search not in the space of graphs, but in the space of Python programs that *generate* graphs. We start with a simple program, then use a large language model (LLM) to generate many similar but slightly different programs (‘mutations’). We score each program by running it and evaluating the graph it produces. It is natural to wonder why this approach would be beneficial. An LLM call is usually vastly more expensive than adding an edge or evaluating a graph, so this way we can often explore thousands or even millions of times fewer candidates than with standard local search methods. Many ‘nice’ mathematical objects, like the optimal Hoffman-Singleton graph for the aforementioned problem [142], have short, elegant descriptions as code. Moreover even if there is only one optimal construction for a problem, there can be many different, natural programs that generate it. Conversely, the countless ‘ugly’ graphs that are local optima might not correspond to any simple program. Searching in program space might act as a powerful prior for simplicity and structure, helping us navigate away from messy local maxima towards elegant, often optimal, solutions. In the case where the optimal solution does not admit a simple description, even by a program, and the best way to find it is via heuristic methods, we have found that AlphaEvolve excels at this task as well.

Still, for problems where the scoring function is cheap to compute, the sheer brute-force advantage of traditional methods can be hard to overcome. Our proposed solution to this problem is as follows. Instead of evolving programs that directly *generate* a construction, AlphaEvolve evolves programs that *search for* a construction. This is what we refer to as the *search mode* of AlphaEvolve, and it was the standard mode we used for all the problems where the goal was to find good constructions, and we did not care about their interpretability and generalizability.

Each program in AlphaEvolve’s population is a search heuristic. It is given a fixed time budget (say, 100 seconds) and tasked with finding the best possible construction within that time. The score of the heuristic is the score of the best object it finds. This resolves the speed disparity: a single, slow LLM call to generate a new search heuristic can trigger a massive cheap computation, where that heuristic explores millions of candidate constructions on its own.

We emphasize that the search does not have to start from scratch each time. Instead, a new heuristic is evaluated on its ability to *improve the best construction found so far*. We are thus evolving a population of ‘improver’ functions. This creates a dynamic, adaptive search process. In the beginning, heuristics that perform broad, exploratory searches might be favored. As we get closer to a good solution, heuristics that perform clever, problem-specific refinements might take over. The final result is often a sequence of specialized heuristics that, when chained together, produce a state-of-the-art construction. The downside is a potential loss of interpretability in the search *process*, but the final *object* it discovers remains a well-defined mathematical entity for us to study. This addition seems to be particularly useful for more difficult problems, where a single search function may not be able to discover a good solution by itself.**1.4. Generalizing from Examples to Formulas: the *generalizer mode*.** Beyond finding constructions for a fixed problem size (e.g., packing for  $n = 11$ ) on which the above *search mode* excelled, we have experimented with a more ambitious *generalizer mode*. Here, we tasked AlphaEvolve with writing a program that can solve the problem for any given  $n$ . We evaluate the program based on its performance across a range of  $n$  values. The hope is that by seeing its own (often optimal) solutions for small  $n$ , AlphaEvolve can spot a pattern and generalize it into a construction that works for all  $n$ .

This mode is more challenging, but it has produced some of our most exciting results. In one case, AlphaEvolve’s proposed construction for the Nikodym problem (see Problem 6.1) inspired a new paper by the third author [281]. On the other hand, when using the *search mode*, the evolved programs can not easily be interpreted. Still, the final *constructions* themselves can be analyzed, and in the case of the arithmetic Kakeya problem (Problem 6.30) they inspired another paper by the third author [282].

**1.5. Building a pipeline of several AI tools.** Even more strikingly, for the finite field Kakeya problem (cf. Problem 6.1), AlphaEvolve discovered an interesting general construction. When we fed this programmatic solution to the agent called Deep Think [149], it successfully derived a proof of its correctness and a closed-form formula for its size. This proof was then fully formalized in the Lean proof assistant using another AI tool, AlphaProof [148]. This workflow, combining pattern discovery (AlphaEvolve), symbolic proof generation (Deep Think), and formal verification (AlphaProof), serves as a concrete example of how specialized AI systems can be integrated. It suggests a future potential methodology where a combination of AI tools can assist in the process of moving from an empirically observed pattern (suggested by the model) to a formally verified mathematical result, fully automated or semi-automated.

**1.6. Limitations.** We would also like to point out that while AlphaEvolve excels at problems that can be clearly formulated as the optimization of a smooth score function that is possible to ‘hill-climbing’ on, it sometimes struggles otherwise. In particular, we have encountered several instances where AlphaEvolve failed to attain an optimal or close to optimal result. We also report these cases below. In general, we have found AlphaEvolve most effective when applied at a large scale across a broad portfolio of loosely related problems such as, for example, packing problems or Sendov’s conjecture and its variants.

In Section 6, we will detail the new mathematical results discovered with this approach, along with all the examples we found where AlphaEvolve did not manage to find the previously best known construction. We hope that this work will not only provide new insights into these specific problems but also inspire other scientists to explore how these tools can be adapted to their own areas of research.

## 2. OVERVIEW OF ALPHAEVOLVE AND USAGE

As introduced in [224], AlphaEvolve establishes a framework that combines the creativity of LLMs with automated evaluators. Some of its description and usage appears there and we discuss it here in order for this paper to be self-contained. At its heart, AlphaEvolve is an evolutionary system. The system maintains a population of programs, each encoding a potential solution to a given problem. This population is iteratively improved through a loop that mimics natural selection.

The evolutionary process consists of two main components:

1. (1) **A Generator (LLM):** This component is responsible for introducing variation. It takes some of the better-performing programs from the current population and ‘mutates’ them to create new candidate solutions. This process can be parallelized across several CPUs. By leveraging an LLM, these mutations are not random character flips but intelligent, syntactically-aware modifications to the code, inspired by the logic of the parent programs and the expert advice given by the human user.- (2) An Evaluator (typically provided by the user): This is the ‘fitness function’. It is a deterministic piece of code that takes a program from the population, runs it, and assigns it a numerical score based on its performance. For a mathematical construction problem, this score could be how well the construction satisfies certain properties (e.g., the number of edges in a graph, or the density of a packing).

The process begins with a few simple initial programs. In each generation, some of the better-scoring programs are selected and fed to the LLM to generate new, potentially better, offspring. These offspring are then evaluated, scored, and the higher scoring ones among them will form the basis of the future programs. This cycle of generation and selection allows the population to ‘evolve’ over time towards programs that produce increasingly high-quality solutions. Note that since every evaluator has a fixed time budget, the total CPU hours spent by the evaluators is directly proportional to the total number of LLM calls made in the experiment. For more details and applications beyond mathematical problems, we refer the reader to [224]. Nagda et al. [221] apply AlphaEvolve to establish new hardness of approximation results for problems such as the Metric Traveling Salesman Problem and MAX-k-CUT. After AlphaEvolve was released, other open-source implementations of frameworks leveraging LLMs for scientific discovery were developed such as OpenEvolve [257], ShinkaEvolve [190] or DeepEvolve [202].

When applied to mathematics, this framework is particularly powerful for finding constructions with extremal properties. As described in the introduction, we primarily use it in a *search mode*, where the programs being evolved are not direct constructions but are themselves heuristic search algorithms. The evaluator gives one of these evolved heuristics a fixed time budget and scores it based on the quality of the best construction it can find in that time. This method turns the expensive, creative power of the LLM towards designing efficient search strategies, which can then be executed cheaply and at scale. This allows AlphaEvolve to effectively navigate vast and complex mathematical landscapes, discovering the novel constructions we detail in this paper.

### 3. META-ANALYSIS AND ABLATIONS

To better understand the behavior and sensitivities of AlphaEvolve, we conducted a series of meta-analyses and ablation studies. These experiments are designed to answer practical questions about the method: How do computational resources affect the search? What is the role of the underlying LLM? What are the typical costs involved? For consistency, many of these experiments use the autocorrelation inequality (Problem 6.2) as a testbed, as it provides a clean, fast-to-evaluate objective.

**3.1. The Trade-off Between Speed of Discovery and Evaluation Cost.** A key parameter in any AlphaEvolve run is the amount of parallel computation used (e.g., the number of CPU threads). Intuitively, more parallelism should lead to faster discoveries. We investigated this by running Problem 6.2 with varying numbers of parallel threads (from 2 up to 20).

Our findings (see Figure 1), while noisy, seem to align with this expected trade-off. Increasing the number of parallel threads significantly accelerated the time-to-discovery. Runs with 20 threads consistently surpassed the state-of-the-art bound much faster than those with 2 threads. However, this speed comes at a higher total cost. Since each thread operates semi-independently and makes its own calls to the LLM to generate new heuristics, doubling the threads roughly doubles the rate of LLM queries. Even though the threads communicate with each other and build upon each other’s best constructions, achieving the result faster requires a greater total number of LLM calls. The optimal strategy depends on the researcher’s priority: for rapid exploration, high parallelism is effective; for minimizing direct costs, fewer threads over a longer period is the more economical choice.

**3.2. The Role of Model Choice: Large vs. Cheap LLMs.** AlphaEvolve’s performance is fundamentally tied to the LLM used for generating code mutations. We compared the effectiveness of a high-performance LLMFIGURE 1. Performance on Problem 6.2: running AlphaEvolve with more parallel threads leads to the discovery of good constructions faster, but at a greater total compute cost. The results displayed are the averages of 100 experiments with 2 CPU threads, 40 experiments with 5 CPU threads, 20 experiments with 10 CPU threads, and 10 experiments with 20 CPU threads.

against a much smaller, cheaper model (with a price difference of roughly 15x per input token and 30x per output token).

We observed that the more capable LLM tends to produce higher-quality suggestions (see Figure 2), often leading to better scores with fewer evolutionary steps. However, the most effective strategy was not always to use the most powerful model exclusively. For this simple autocorrelation problem, the most cost-effective strategy to beat the literature bound was to use the cheapest model across many runs. The total LLM cost for this was remarkably low: a few USD. However, for the more difficult problem of Nikodym sets (see Problem 6.1), the cheap model was not able to get the most elaborate constructions.

We also observed that an experiment using only high-end models can sometimes perform worse than a run that occasionally used cheaper models as well. One explanation for this is that different models might suggest very different approaches, and even though a worse model generally suggests lower quality ideas, it does add variance. This suggests a potential benefit to injecting a degree of randomness or “naive creativity” into the evolutionary process. We suspect that for problems requiring deeper mathematical insight, the value of the smarter LLM would become more pronounced, but for many optimization landscapes, diversity from cheaper models is a powerful and economical tool.FIGURE 2. Comparison of 50 experiments on Problem 6.2 using a cheap LLM and 20 experiments using a more expensive LLM. The experiments using a cheaper LLM required about twice as many calls as the ones using expensive ones, and this ratio tends to be even larger for more difficult problems.

#### 4. CONCLUSIONS

Our exploration of AlphaEvolve has yielded several key insights, which are summarized below. We have found that the selection of the verifier is a critical component that significantly influences the system’s performance and the quality of the discovered results. For example, sometimes the optimizer will be drawn more towards more stable (trivial) solutions which we want to avoid. Designing a clever verifier that avoids this behavior is key to discover new results.

Similarly, employing continuous (as opposed to discrete) loss functions proved to be a more effective strategy for guiding the evolutionary search process in some cases. For example, for Problem 6.54 we could have designed our scoring function as the number of touching cylinders of any given configuration (or  $-\infty$  if the configuration is illegal). By looking at a continuous scoring function depending on the distances led to a more successful and faster optimization process.

During our experiments, we also observed a “cheating phenomenon”, where the system would find loopholes or exploit artifacts (leaky verifier when approximating global constraints such as positivity by discrete versions of them, unreliable LLM queries to cheap models, etc.) in the problem setup rather than genuine solutions, highlighting the need for carefully designed and robust evaluation environments.

Another important component is the advice given in the prompt and the experience of the prompter. We have found that we got better at knowing how to prompt AlphaEvolve the more we tried. For example, prompting as in our *search mode* versus trying to find the construction directly resulted in more efficient programs and much better results in the former case. Moreover, in the hands of a user who is a subject expert in the particular problem that is being attempted, AlphaEvolve has always performed much better than in the hands of another user who is not a subject expert: we have found that the advice one gives to AlphaEvolve in the prompt has a significant impact on the quality of the final construction. Giving AlphaEvolve an insightful piece of expert advice in the prompt almost always led to significantly better results: indeed, AlphaEvolve will always simply try to squeeze the most out of the advice it was given, while retaining the gist of the original advice. We stress that we think that, in general, it was the combination of human expertise and the computational capabilities of AlphaEvolve that led to the best results overall.An interesting finding for promoting the discovery of broadly applicable algorithms is that generalization improves when the system is provided with a more constrained set of inputs or features. Having access to a large amount of data does not necessarily imply better generalization performance. Instead, when we were looking for interpretable programs that generalize across a wide range of the parameters, we constrained AlphaEvolve to have access to less data by showing it the previous best solutions only for small values of  $n$  (see for example Problems 6.29, 6.65, 6.1). This “less is more” approach appears to encourage the emergence of more fundamental ideas. Looking ahead, a significant step toward greater autonomy for the system would be to enable AlphaEvolve to select its own hyperparameters, adapting its search strategy dynamically.

Results are also significantly improved when the system is trained on correlated problems or a family of related problem instances within a single experiment. For example, when exploring geometric problems, tackling configurations with various numbers of points  $n$  and dimensions  $d$  simultaneously is highly effective. A search heuristic that performs well for a specific  $(n, d)$  pair will likely be a strong foundation for others, guiding the system toward more universal principles.

We have found that AlphaEvolve excels at discovering constructions that were already within reach of current mathematics, but had not yet been discovered due to the amount of time and effort required to find the right combination of standard ideas that works well for a particular problem. On the other hand, for problems where genuinely new, deep insights are required to make progress, AlphaEvolve is likely not the right tool to use. In the future, we envision that tools like AlphaEvolve could be used to systematically assess the difficulty of large classes of mathematical bounds or conjectures. This could lead to a new type of classification, allowing researchers to semi-automatically label certain inequalities as “AlphaEvolve-hard”, indicating their resistance to AlphaEvolve-based methods. Conversely, other problems could be flagged as being amenable to further attacks by both theoretical and computer-assisted techniques, thereby directing future research efforts more effectively.

## 5. FUTURE WORK

The mathematical developments in AlphaEvolve represent a significant step toward automated mathematical discovery, though there are many future directions that are wide open. Given the nature of the human-machine interface, we imagine a further incorporation of a computer-assisted proof into the output of AlphaEvolve in the future, leading to AlphaEvolve first finding the candidate, then providing the e.g. Lean code of such computer-assisted proof to validate it, all in an automatic fashion. In this work, we have demonstrated that in rare cases this is already possible, by providing an example of a full pipeline from discovery to formalization, leading to further insights that when combined with human expertise yield stronger results. This paper represents a first step of a long-term goal that is still in progress, and we expect to explore more in this direction. The line drawn by this paper is solely due to human time and paper length constraints, but not by our computational capabilities. Specifically, in some of the problems we believe that (ongoing and future) further exploration might lead to more and better results.

**Acknowledgements:** JGS has been partially supported by the MICINN (Spain) research grant number PID2021–125021NA–I00; by NSF under Grants DMS-2245017, DMS-2247537 and DMS-2434314; and by a Simons Fellowship. This material is based upon work supported by a grant from the Institute for Advanced Study School of Mathematics. TT was supported by the James and Carol Collins Chair, the Mathematical Analysis & Application Research Fund, and by NSF grants DMS-2347850, and is particularly grateful to recent donors to the Research Fund.

We are grateful for contributions, conversations and support from Matej Balog, Henry Cohn, Alex Davies, Demis Hassabis, Ray Jiang, Pushmeet Kohli, Freddie Manners, Alexander Novikov, Joaquim Ortega-Cerdà, Abigail See, Eric Wieser, Junyan Xu, Daniel Zheng, and Goran Žužić. We are also grateful to Alex Bäuerle, Adam Connors, Lucas Dixon, Fernanda Viegas, and Martin Wattenberg for their work on creating the user interface for AlphaEvolve that lets us publish our experiments so others can explore them. Finally, we thank David Woodruff for corrections.## 6. MATHEMATICAL PROBLEMS WHERE ALPHAEVOLVE WAS TESTED

In our experiments we took 67 problems (both solved and unsolved) from the mathematical literature, most of which could be reformulated in terms of obtaining upper and/or lower bounds on some numerical quantity (which could depend on one or more parameters, and in a few cases was multi-dimensional instead of scalar-valued). Many of these quantities could be expressed as a supremum or infimum of some score function over some set (which could be finite, finite dimensional, or infinite dimensional). While both upper and lower bounds are of interest, in many cases only one of the two types of bounds was amenable to an AlphaEvolve approach, as it is a tool designed to find interesting mathematical constructions, i.e., examples that attempt to optimize the score function, rather than prove bounds that are valid for all possible such examples. In the cases where the domain of the score function was infinite-dimensional (e.g., a function space), an additional restriction or projection to a finite dimensional space (e.g., via discretization or regularization) was used before AlphaEvolve was applied to the problem.

In many cases, AlphaEvolve was able to match (or nearly match) existing bounds (some of which are known or conjectured to be sharp), often with an interpretable description of the extremizers, and in several cases could improve upon the state of the art. In other cases, AlphaEvolve did not even match the literature bounds, but we have endeavored to document both the positive and negative results for our experiments here to give a more accurate portrait of the strengths and weaknesses of AlphaEvolve as a tool. Our goal is to share the results on all problems we tried, even on those we attempted only very briefly, to give an honest account of what works and what does not.

In the cases where AlphaEvolve improved upon the state of the art, it is likely that further work, using either a version of AlphaEvolve with improved prompting and setup, a more customized approach guided by theoretical considerations or traditional numerics, or a hybrid of the two approaches, could lead to further improvements; this has already occurred in some of the AlphaEvolve results that were previously announced in [224]. We hope that the results reported here can stimulate further such progress on these problems by a broad variety of methods.

Throughout this section, we will use the following notation: We will say that  $A \lesssim B$  (resp.  $A \gtrsim B$ ) whenever there exists a constant  $C$  independent of  $A, B$  such that  $|A| \leq CB$  (resp.  $|A| \geq CB$ ).

### Contents.

<table>
<tr>
<td><a href="#">Contents</a></td>
<td>9</td>
</tr>
<tr>
<td><a href="#">1. Finite field Kakeya and Nikodym sets</a></td>
<td>11</td>
</tr>
<tr>
<td><a href="#">2. Autocorrelation inequalities</a></td>
<td>13</td>
</tr>
<tr>
<td><a href="#">3. Difference bases</a></td>
<td>17</td>
</tr>
<tr>
<td><a href="#">4. Kissing numbers</a></td>
<td>17</td>
</tr>
<tr>
<td><a href="#">5. Kakeya needle problem</a></td>
<td>18</td>
</tr>
<tr>
<td><a href="#">6. Sphere packing and uncertainty principles</a></td>
<td>23</td>
</tr>
<tr>
<td><a href="#">7. Classical inequalities</a></td>
<td>27</td>
</tr>
<tr>
<td><a href="#">8. The Ovals problem</a></td>
<td>29</td>
</tr>
</table><table><tr><td>9.</td><td>Sendov's conjecture and its variants</td><td>30</td></tr><tr><td>10.</td><td>Crouzeix's conjecture</td><td>34</td></tr><tr><td>11.</td><td>Sidorenko's conjecture</td><td>35</td></tr><tr><td>12.</td><td>The prime number theorem</td><td>35</td></tr><tr><td>13.</td><td>Flat polynomials and Golay's merit factor conjecture</td><td>36</td></tr><tr><td>14.</td><td>Blocks Stacking</td><td>38</td></tr><tr><td>15.</td><td>The arithmetic Kakeya conjecture</td><td>41</td></tr><tr><td>16.</td><td>Furstenberg–Sárközy theorem</td><td>41</td></tr><tr><td>17.</td><td>Spherical designs</td><td>42</td></tr><tr><td>18.</td><td>The Thomson and Tammes problems</td><td>44</td></tr><tr><td>19.</td><td>Packing problems</td><td>46</td></tr><tr><td>20.</td><td>The Turán number of the tetrahedron</td><td>48</td></tr><tr><td>21.</td><td>Factoring <math>N!</math> into <math>N</math> numbers</td><td>49</td></tr><tr><td>22.</td><td>Beat the average game</td><td>50</td></tr><tr><td>23.</td><td>Erdős discrepancy problem</td><td>51</td></tr><tr><td>24.</td><td>Points on sphere maximizing the volume</td><td>51</td></tr><tr><td>25.</td><td>Sums and differences problems</td><td>52</td></tr><tr><td>26.</td><td>Sum-product problems</td><td>53</td></tr><tr><td>27.</td><td>Triangle density in graphs</td><td>54</td></tr><tr><td>28.</td><td>Matrix multiplications and AM-GM inequalities</td><td>55</td></tr><tr><td>29.</td><td>Heilbronn problems</td><td>56</td></tr><tr><td>30.</td><td>Max to min ratios</td><td>57</td></tr><tr><td>31.</td><td>Erdős–Gyárfás conjecture</td><td>58</td></tr><tr><td>32.</td><td>Erdős squarefree problem</td><td>58</td></tr><tr><td>33.</td><td>Equidistant points in convex polygons</td><td>59</td></tr></table><table>
<tr>
<td>34.</td>
<td>Pairwise touching cylinders</td>
<td>59</td>
</tr>
<tr>
<td>35.</td>
<td>Erdős squares in a square problem</td>
<td>60</td>
</tr>
<tr>
<td>36.</td>
<td>Good asymptotic constructions of Szemerédi–Trotter</td>
<td>60</td>
</tr>
<tr>
<td>37.</td>
<td>Rudin problem for polynomials</td>
<td>61</td>
</tr>
<tr>
<td>38.</td>
<td>Erdős–Szekeres Happy Ending problem</td>
<td>62</td>
</tr>
<tr>
<td>39.</td>
<td>Subsets of the grid with no isosceles triangles</td>
<td>63</td>
</tr>
<tr>
<td>40.</td>
<td>The “no 5 on a sphere” problem</td>
<td>63</td>
</tr>
<tr>
<td>41.</td>
<td>The Ring Loading Problem</td>
<td>64</td>
</tr>
<tr>
<td>42.</td>
<td>Moving sofa problem</td>
<td>65</td>
</tr>
<tr>
<td>43.</td>
<td>International Mathematical Olympiad (IMO) 2025: Problem 6</td>
<td>66</td>
</tr>
<tr>
<td>44.</td>
<td>Bonus: Letting AlphaEvolve write code that can call LLMs</td>
<td>69</td>
</tr>
<tr>
<td>44.1.</td>
<td>The function guessing game</td>
<td>69</td>
</tr>
<tr>
<td>44.2.</td>
<td>Smullyan-type logic puzzles</td>
<td>70</td>
</tr>
</table>

## 1. Finite field Kakeya and Nikodym sets.

**Problem 6.1 (Kakeya and Nikodym sets).** *Let  $d \geq 1$ , and let  $q$  be a prime power. Let  $\mathbf{F}_q$  be a finite field of order  $q$ . A Kakeya set is a set  $K$  that contains a line in every direction, and a Nikodym set  $N$  is a set with the property that every point  $x$  in  $\mathbf{F}_q^d$  is contained in a line that is contained in  $N \cup \{x\}$ . Let  $C_{6.1}^K(d, q), C_{6.1}^N(d, q)$  denote the least size of a Kakeya or Nikodym set in  $\mathbf{F}_q^d$  respectively.*

These quantities have been extensively studied in the literature, due to connections with block designs, the polynomial method in combinatorics, and a strong analogy with the Kakeya conjecture in other settings such as Euclidean space. The previous best known bounds for large  $q$  can be summarized as follows:

- • We have the general inequality

$$C_{6.1}^N(d, q) \geq C_{6.1}^K(d, q) - \frac{2q^{d-1} - q^{d-2} - q}{q^{d-1} - 1} q^{d-1} \geq C_{6.1}^K(d, q) - 2q^{d-1} \quad (6.1)$$

which reflects the fact that a projective transformation of a Nikodym set is essentially a Kakeya set; see [281].

- • We trivially have  $C_{6.1}^K(1, q) = C_{6.1}^N(1, q) = q$ .
- •  $C_{6.1}^K(2, q)$  is equal to  $q(q+1)/2 + (q-1)/2$  when  $q$  is odd and  $q(q+1)/2$  when  $q$  is even [205, 32].
- • In contrast, from the theory of blocking sets,  $C_{6.1}^N(2, q)$  is known to be at least  $q^2 - q^{3/2} - 1 + \frac{1}{4}s(1-s)q$ , where  $s$  is the fractional part of  $\sqrt{q}$  [276]. When  $q$  is a perfect square, this bound is sharp up to a lower order error  $O(q \log q)$  [31]<sup>1</sup>. However, there is no obvious way to adapt such results to the non-perfect-square case.

<sup>1</sup>In the notation of that paper, Nikodym sets are the “green” portion of a “green–black coloring”.- • In general, we have the bounds

$$\left(2 - \frac{1}{q}\right)^{-(d-1)} q^d \leq C_{6.1}^K(d, q) \leq \frac{1}{2^{d-1}} q^d \left(1 + \frac{d+1-2^{-d+2}}{q} + O\left(\frac{1}{q^2}\right)\right);$$

see [49]. In particular,  $C_{6.1}^K(d, q) = \frac{1}{2^{d-1}} q^d + O(q^{d-1})$  and thus also  $C_{6.1}^N(d, q) \geq \frac{1}{2^{d-1}} q^d + O(q^{d-1})$ , thanks to (6.1).

- • It is conjectured that  $C_{6.1}^N(d, q) = q^d - o(q^d)$  [205, Conjecture 1.2]. In the regime when  $q$  goes to infinity while the characteristic stays bounded (which in particular includes the case of even  $q$ ) the stronger bound  $C_{6.1}^N(d, q) = q^d - O(q^{(1-\varepsilon)d})$  is known [156, Theorem 1.6]. In three dimensions the conjecture would be implied by a further conjecture on unions of lines [205, Conjecture 1.4].
- • The classes of Kakeya and Nikodym sets can both be checked to be closed under Cartesian products, giving rise to the inequalities  $C_{6.1}^K(d_1 + d_2, q) \leq C_{6.1}^K(d_1, q)C_{6.1}^K(d_2, q)$  and  $C_{6.1}^N(d_1 + d_2, q) \leq C_{6.1}^N(d_1, q)C_{6.1}^N(d_2, q)$  for any  $d_1, d_2 \geq 1$ . When  $q$  is a perfect square, one can combine this observation with the constructions in [31] (and the trivial bound  $C_{6.1}^N(1, q) = q$ ) to obtain an upper bound

$$C_{6.1}^N(d, q) \leq q^d - \left\lfloor \frac{d}{2} \right\rfloor q^{d-1/2} + O(q^{d-1} \log q)$$

for any fixed  $d \geq 1$ .

We applied AlphaEvolve to search for new constructions of Kakeya and Nikodym sets in  $\mathbf{F}_p^d$  and  $\mathbf{F}_q^d$ , for various values of  $d$ . Since we were after a construction that works for all primes  $p$  / prime powers  $q$  (or at least an infinite class of primes / prime powers), we used the *generalizer mode* of AlphaEvolve. That is, every construction of AlphaEvolve was evaluated on many large values of  $p$  or  $q$ , and the final score was the average normalized size of all these constructions. This encouraged AlphaEvolve to find constructions that worked for many values of  $p$  or  $q$  simultaneously.

Throughout all of these experiments, whenever AlphaEvolve found a construction that worked well on a large range of primes, we asked Deep Think to give us an explicit formula for the sizes of the sets constructed. If Deep Think succeeded in deriving a closed form expression, we would check if this formula matched our records for several primes, and if it did, it gave us some confidence that the Deep Think produced proof was likely correct. To gain absolute confidence, in one instance we then used AlphaProof to turn this natural language proof into a fully formalized Lean proof. Unfortunately, this last step was possible only when the proof was simple enough; in particular all of its necessary steps needed to have already been implemented in the Lean library `mathlib`.

This investigation into Kakeya sets yielded new constructions with lower-order improvements in dimensions 3, 4, and 5. In three dimensions, AlphaEvolve discovered multiple new constructions, such as one demonstrating the bound  $C_{6.1}^K(3, p) \leq \frac{1}{4}p^3 + \frac{7}{8}p^2 - \frac{1}{8}$  that worked for all primes  $p \equiv 1 \pmod{4}$ , via the explicit Kakeya set

$$\left\{ \left( x, \frac{q_1 + q_2}{2} - x^2 - g, \frac{q_1 - q_2}{2} \right) : x \in \mathbf{F}_p; q_1, q_2 \in S \right\} \cup \{(0, y, z) : y + z^2 \in S\} \cup \{(0, y, 0) : y \in \mathbf{F}_p\}$$

where  $g := \frac{p-1}{4}$  and  $S$  is the set of quadratic residues (including 0). This slightly refines the previously best known bound  $C_{6.1}^K(3, p) \leq \frac{1}{4}p^3 + \frac{7}{8}p^2 + O(p)$  from [49]. Since we found so many promising constructions that would have been tedious to verify manually, we found it useful to have Deep Think produce proofs of formulas for the sizes of the produced sets, which we could then cross-reference with the actual sizes for several primes  $p$ . When we wanted to be absolutely certain that the proof was correct, here we used AlphaProof to produce a fully formal Lean proof as well. This was only possible because the proofs typically used reasonably elementary, though quite long, number theoretic inclusion-exclusion computations.

In four dimensions, the difficulty ramped up quite a bit, and many of the methods that worked for  $d = 3$  stopped working altogether. AlphaEvolve came up with a construction demonstrating the bound  $C_{6.1}^K(4, p) \leq \frac{1}{8}p^4 + \frac{19}{32}p^3 + \frac{11}{16}p^2 + O(p^{\frac{3}{2}})$ , again for primes  $p \equiv 1 \pmod{4}$ . As in the  $d = 3$  case, the coefficients in the leading two terms match the best-known construction in [49] (and may have a modest improvement in the  $p^2$  term). In theproof of this construction, Deep Think revealed a link to elliptic curves, which explains why the lower-order error terms grow like  $O(p^{\frac{3}{2}})$  instead of being simple polynomials. Unfortunately, this also meant that the proofs were too difficult for AlphaProof to handle, and since there was no exact formula for the size of the sets, we could not even cross-reference the asymptotic formula claimed by Deep Think with our actual computed numbers. As such, in stark contrast to the  $d = 3$  case, we had to resort to manually checking the proofs ourselves.

On closer inspection, the construction AlphaEvolve found for the  $d = 4$  case of the finite field Kakeya problem was not too far from the constructions in the literature, which also involved various polynomial constraints involving quadratic residues; up to trivial changes of variable, AlphaEvolve matched the construction in [49] exactly outside of a three-dimensional subspace of  $\mathbb{F}_p^4$ , and was fairly similar to that construction inside that subspace as well. While it is possible that with more classical numerical experimentation and trial and error one could have found such a construction, it would have been rather time-consuming to do so. Overall, we felt this was a great example of AlphaEvolve finding structures with deep number-theoretic properties, especially since the reference [49] was not explicitly made available to AlphaEvolve.

The same pattern held in  $d = 5$ , where we found a construction establishing  $C_{6.1}^K(5, p)$  of size  $\frac{1}{16}p^5 + \frac{47}{128}p^4 + \frac{177}{256}p^3 + O(p^{\frac{5}{2}})$  for primes  $p \equiv 1 \pmod{4}$  with a Deep Think proof that we verified by hand. In both the  $d = 4$  and  $d = 5$  cases, our results matched the leading two coefficients from [49], but refined the lower order terms (which was not the focus of [49]).

The story with Nikodym sets was a bit different and showed more of a back-and-forth between the AI and us. AlphaEvolve's first attempt in three dimensions gave a promising construction by building complicated high-degree surfaces that Deep Think had a hard time analyzing. By simplifying the approach by hand to use lower-degree surfaces and more probabilistic ideas, we were able to find a better construction establishing the upper bound  $C_{6.1}^N(d, p) \leq p^d - (((d-2)/\log 2) + 1 + o(1))p^{d-1} \log p$  for fixed  $d \geq 3$ , improving on the best known construction. AlphaEvolve's construction, while not optimal, was a great jumping-off point for human intuition. The details of this proof will appear in a separate paper by the third author [281].

Another experiment highlighted how important expert guidance can be. As noted earlier in this section, for fields of square order  $q = p^2$ , there are Nikodym sets in two dimensions giving the bound  $C_{6.1}^N(2, q) \leq q^2 - q^{3/2} + O(q \log q)$ . At first we asked AlphaEvolve to solve this problem without any hints, and it only managed to find constructions of size  $q^2 - O(q \log q)$ . Next, we ran the same experiment again, but this time telling AlphaEvolve that a construction of size  $q^2 - q^{3/2} + O(q \log q)$  was possible. Curiously, this small bit of extra information had a huge impact on the performance: AlphaEvolve now immediately found constructions of size  $q^2 - cq^{3/2}$  for a small constant  $c > 0$ , and eventually it discovered various different constructions of size  $q^2 - q^{3/2} + O(q \log q)$ .

We also experimented with giving AlphaEvolve hints from a relevant paper ([276]) and asked it to reproduce the complicated construction in it via code. We measured its progress just as before, by looking simply at the size of the construction it created on a wide range of primes. After a few hundred iterations AlphaEvolve managed to reproduce the constructions in the paper (and even slightly improve on it via some small heuristics that happen to work well for small primes).

**2. Autocorrelation inequalities.** The convolution  $f * g$  of two (absolutely integrable) functions  $f, g : \mathbb{R} \rightarrow \mathbb{R}$  is defined by the formula

$$(f * g)(t) = \int_{\mathbb{R}} f(x)g(t-x) dx.$$

When  $g$  is either equal to  $f$  or a reflection of  $f$ , we informally refer to such convolutions as *autocorrelations*. There has been some literature on obtaining sharp constants on various functional inequalities involving autocorrelations; see [90] for a general survey. In this paper, AlphaEvolve was applied to some of them via its standard *search mode*, evolving a heuristic search function that produces a good function within a fixed time budget, given the best construction so far as input. We now set out some notation for some of these inequalities.**Problem 6.2.** Let  $C_{6.2}$  denote the largest constant for which one has

$$\max_{-1/2 \leq t \leq 1/2} \int_{\mathbb{R}} f(t-x)f(x) dx \geq C_{6.2} \left( \int_{-1/4}^{1/4} f(x) dx \right)^2 \quad (6.2)$$

for all non-negative  $f : \mathbb{R} \rightarrow \mathbb{R}$ . What is  $C_{6.2}$ ?

Problem 6.2 arises in additive combinatorics, relating to the size of Sidon sets. Prior to this work, the best known upper and lower bounds were

$$1.28 \leq C_{6.2} \leq 1.50992$$

with the lower bound achieved in [59] and the upper bound achieved in [210]; we refer the reader to these references for prior bounds on the problem.

Upper and lower bounds for  $C_{6.2}$  can both be achieved by computational methods, and so both types of bounds are potential use cases for AlphaEvolve. For lower bounds, we refer to [59]. For upper bounds, one needs to produce specific counterexamples  $f$ . The explicit choice

$$f(x) = \frac{1}{\sqrt{2x + 1/2}} 1_{(-1/4, 1/4)}(x)$$

already gives the upper bound  $C_{6.2} \leq \pi/2 = 1.57079\dots$ , which at one point was conjectured to be optimal. The improvement comes from a numerical search involving functions that are piecewise constant on a fixed partition of  $(-1/4, 1/4)$  into some finite number  $n$  of intervals ( $n = 10$  is already enough to improve the  $\pi/2$  bound), and optimizing. There are some tricks to speed up the optimization, in particular there is a Newton type method in which one selects an intelligent direction in which to perturb a candidate  $f$ , and then moves optimally in that direction. See [210] for details. After we told AlphaEvolve about this Newton type method, it found heuristic search methods using “cubic backtracking” that produced constructions reducing the upper bound to  $C_{6.2} \leq 1.5032$ . See [Repository of Problems](#) for several constructions and some of the search functions that got evolved.

After our results, Damek Davis performed a very thorough meta-analysis [88] using different optimization methods and was not able to improve on the results, perhaps due to the highly irregular nature of the numerical optimizers (see Figure 3). This is an example of how much AlphaEvolve can reduce the effort required to optimize a problem.

The following problem, studied in particular in [210], concerns the extent to which an autocorrelation  $f * f$  of a non-negative function  $f$  can resemble an indicator function.

**Problem 6.3.** Let  $C_{6.3}$  be the best constant for which one has

$$\|f * f\|_{L^2(\mathbb{R})}^2 \leq C_{6.3} \|f * f\|_{L^1(\mathbb{R})} \|f * f\|_{L^\infty(\mathbb{R})}$$

for non-negative  $f : \mathbb{R} \rightarrow \mathbb{R}$ . What is  $C_{6.3}$ ?

It is known that

$$0.88922 \leq C_{6.3} \leq 1$$

with the upper bound being immediate from Hölder’s inequality, and the lower bound coming from a piecewise constant counterexample. It is tentatively conjectured in [210] that  $C_{6.3} < 1$ .

The lower bound requires exhibiting a specific function  $f$ , and is thus a use case for AlphaEvolve. Similarly to how we approached Problem 6.2, we can restrict ourselves to piecewise constant functions, with a fixed number of equal sized parts. With this simple setup, AlphaEvolve improved the lower bound to  $C_{6.3} \geq 0.8962$  in a quick experiment. A recent work of Boyer and Li [42] independently used gradient-based methods to obtain the further improvement  $C_{6.3} \geq 0.901564$ . Seeing this result, we ran our experiment for a bit longer. After a few hours AlphaEvolve also discovered that gradient-based methods work well for this problem. Letting it run forFIGURE 3. Left: the constructions produced by AlphaEvolve for Problem 6.2, Right: their autoconvolutions. From top to bottom, their scores are 1.5053, 1.5040, and 1.5032 (smaller is better).

FIGURE 4. Left: the best construction for Problem 6.3 discovered by AlphaEvolve. Right: its autoconvolution. Both functions are highly irregular and difficult to plot.

several hours longer, it found some extra heuristics that seemed to work well together with the gradient-based methods, and it eventually improved the lower bound to  $C_{6.3} \geq 0.961$  using a step function consisting of 50,000 parts. We believe that with even more parts, this lower bound can be further improved.

Figure 4 shows the discovered step function consisting of 50,000 parts and its autoconvolution. We believe that the irregular nature of the extremizers is one of the reasons why this optimization problem is difficult to accomplish by traditional means.One can remove the non-negativity hypothesis in Problem 6.2, giving a new problem:

**Problem 6.4.** Let  $C_{6.4}$  and  $C'_{6.4}$  be the best constants for which one has

$$(a) \quad \max_{-1/2 \leq t \leq 1/2} \left| \int_{\mathbb{R}} f(t-x)f(x) dx \right| \geq C_{6.4} \left( \int_{-1/4}^{1/4} f(x) dx \right)^2$$

$$(b) \quad \left| \max_{-1/2 \leq t \leq 1/2} \int_{\mathbb{R}} f(t-x)f(x) dx \right| \geq C'_{6.4} \left( \int_{-1/4}^{1/4} f(x) dx \right)^2$$

for all  $f : [-1/4, 1/4] \rightarrow \mathbb{R}$  (note  $f$  can now take negative values). What are  $C_{6.4}$  and  $C'_{6.4}$ ?

Trivially one has  $C_{6.4}, C'_{6.4} \leq C_{6.2}$ . However, there are better examples that gives a new upper bound on  $C_{6.4}$  and  $C'_{6.4}$ , namely  $C_{6.4} \leq 1.4993$  [210] and  $C'_{6.4} \leq 1.45810$  [290]. With the same setup as the previous autocorrelation problems, in a quick experiment AlphaEvolve improved these to  $C_{6.4} \leq 1.4688$  and  $C'_{6.4} \leq 1.4557$ .

**Problem 6.5.** Let  $C_{6.5}$  be the largest constant for which

$$\sup_{x \in [-2,2]} \int_{-1}^1 f(t)g(x+t) dt \geq C_{6.5}$$

for all non-negative  $f, g : [-1, 1] \rightarrow [0, 1]$  with  $f + g = 1$  on  $[-1, 1]$  and  $\int_{\mathbb{R}} f = 1$ , where we extend  $f, g$  by zero outside of  $[-1, 1]$ . What is  $C_{6.5}$ ?

The constant  $C_{6.5}$  controls the asymptotics of the “minimum overlap problem” of Erdős [103], [118, Problem 36]. The bounds

$$0.379005 \leq C_{6.5} \leq 0.3809268534330870$$

are known; the lower bound was obtained in [299] via convex programming methods, and the upper bound obtained in [164] by a step function construction. AlphaEvolve managed to improve the upper bound ever so slightly to  $C_{6.5} \leq 0.380924$ .

The following problem is motivated by a problem in additive combinatorics regarding difference bases.

**Problem 6.6.** Let  $C_{6.6}$  be the smallest constant such that

$$\min_{0 \leq t \leq 1} \int_{\mathbb{R}} f(x)f(x+t) dx \leq C_{6.6} \|f\|_{L^1(\mathbb{R})}^2 \quad (6.3)$$

for  $f \in L^1(\mathbb{R})$ . What is  $C_{6.6}$ ?

In [17] it was shown that

$$0.37 \leq C_{6.6} \leq 0.411.$$

To prove the upper bound, one can assume that  $f$  is non-negative, and one studies the Fourier coefficients  $\hat{g}(\xi)$  of the autocorrelation  $g(t) = \int_{\mathbb{R}} f(x)f(x+t) dx$ . On the one hand, the autocorrelation structure guarantees that these Fourier coefficients are nonnegative. On the other hand, if the minimum in (6.3) is large, then one can use the Hardy–Littlewood rearrangement inequality to lower bound  $\hat{g}(\xi)$  in terms of the  $L^1$  norm of  $g$ , which is  $\|f\|_{L^1(\mathbb{R})}^2$ . Optimizing in  $\xi$  gives the result.

The lower bound was obtained by using an arcsine distribution  $f(x) = \frac{1_{[-1/2, 1/2]}(x)}{\sqrt{1-4x^2}}$  (with some epsilon modifications to avoid some technical boundary issues). The authors in [17] reported that attacking this problem numerically “appears to be difficult”.This problem was the very first one we attempted to tackle in this entire project, when we were still unfamiliar with the best practices of using AlphaEvolve. Since we had not come up with the idea of the *search mode* for AlphaEvolve yet, instead we simply asked AlphaEvolve to suggest a mathematical function directly. Since this way every LLM call only corresponded to one single construction and we were heavily bottlenecked by LLM calls, we tried to artificially make the evaluation more expensive: instead of just computing the score for the function AlphaEvolve suggested, we also computed the scores of thousands of other functions we obtained from the original function via simple transformations. This was the precursor of our *search mode* idea that we developed after attempting this problem.

The results highlighted our inexperience. Since we forced our own heuristic search method (trying the predefined set of simple transformations) onto AlphaEvolve, it was much more restricted and did not do well. Moreover, since we let AlphaEvolve suggest arbitrary functions instead of just bounded step functions with fixed step sizes, it always eventually figured out a way to cheat by suggesting a highly irregular function that exploited the numerical integration methods in our scoring function in just the right way, and got impossibly high scores.

If we were to try this problem again, we would try the *search mode* in the space of bounded step functions with fixed step sizes, since this setup managed to improve all the previous bounds in this section.

**3. Difference bases.** This problem was suggested by a custom literature search pipeline based on Gemini 2.5 [71]. We thank Daniel Zheng for providing us with support for it. We plan to explore further literature suggestions provided by AI tools (including open problems) in the future.

**Problem 6.7 (Difference bases).** *For any natural number  $n$ , let  $\Delta(n)$  be the size of the smallest set  $B$  of integers such that every natural number from 1 to  $n$  is expressible as a difference of two elements of  $B$  (such sets are known as difference bases for the interval  $\{1, \dots, n\}$ ). Write  $C_{6.7}(n) := \Delta^2(n)/n$ , and  $C_{6.7} := \inf_{n \geq 1} C_{6.7}(n)$ . Establish upper and lower bounds on  $C_{6.7}$  that are as strong as possible.*

It was shown in [240] that  $C_{6.7}(n)$  converges to  $C_{6.7}$  as  $n \rightarrow \infty$ , which is also the infimum of this sequence. The previous best bounds (see [16]) on this quantity were

$$2.434\dots = 2 + \max_{0 < \phi < \pi} \frac{2 \sin \phi}{\phi + \pi} \leq C_{6.7} \leq \frac{128^2}{6166} = 2.6571\dots;$$

see [192], [143]. While the lower bound requires some non-trivial mathematical argument, the upper bound proceeds simply by exhibiting a difference set for  $n = 6166$  of cardinality 128, thus demonstrating that  $\Delta(6166) \leq 128$ .

We tasked AlphaEvolve to come up with an integer  $n$  and a difference set for it, that would yield an improved upper bound. AlphaEvolve by itself, with no expert advice, was not able to beat the 2.6571 upper bound. In order to get a better result we had to show it the correct code for generating Singer difference sets [260]. Using this code AlphaEvolve managed to find a substantial improvement in the upper bound from 2.6571 to 2.6390. The construction can be found in the [Repository of Problems](#).

#### 4. Kissing numbers.

**Problem 6.8 (Kissing numbers).** *For a dimension  $n \geq 1$ , define the kissing number  $C_{6.8}(n)$  to be the maximum number of non-overlapping unit spheres that can be arranged to simultaneously touch a central unit sphere in  $n$ -dimensional space. Establish upper and lower bounds on  $C_{6.8}(n)$  that are as strong as possible.*

This problem has been studied as early as 1694 when Isaac Newton and David Gregory discussed what  $C_{6.8}(3)$  would be. The cases  $C_{6.8}(1) = 2$  and  $C_{6.8}(2) = 6$  are trivial. The four-dimensional problem was solved by Musin [218], who proved that  $C_{6.8}(4) = 24$ , using a clever modification of Delsarte's linear programming method [92]. In dimensions 8 and 24, the problem is also solved and the extrema are the  $E_8$  lattice and the Leech latticerespectively, giving kissing numbers of  $C_{6.8}(8) = 240$  and  $C_{6.8}(24) = 196\,560$  respectively [226, 195]. In recent years, Ganzhinov [137], de Laat–Leijenhorst [193] and Cohn–Li [69] managed to improve upper and lower bounds for  $C_{6.8}(n)$  in dimensions  $n \in \{10, 11, 14\}$ ,  $11 \leq n \leq 23$ , and  $17 \leq n \leq 21$  respectively. AlphaEvolve was able to improve on the lower bound for  $C_{6.8}(11)$ , raising it from 592 to 593. See Table 2 for the current best known upper and lower bounds for  $C_{6.8}(n)$ :

<table border="1">
<thead>
<tr>
<th>Dim. <math>n</math></th>
<th>1</th>
<th>2</th>
<th>3</th>
<th>4</th>
<th>5</th>
<th>6</th>
<th>7</th>
<th>8</th>
<th>9</th>
<th>10</th>
<th>11</th>
</tr>
</thead>
<tbody>
<tr>
<td>Lower</td>
<td>2</td>
<td>6</td>
<td>12</td>
<td>24</td>
<td>40</td>
<td>72</td>
<td>126</td>
<td>240</td>
<td>306</td>
<td>510</td>
<td>593</td>
</tr>
<tr>
<td>Upper</td>
<td>2</td>
<td>6</td>
<td>12</td>
<td>24</td>
<td>44</td>
<td>77</td>
<td>134</td>
<td>240</td>
<td>363</td>
<td>553</td>
<td>868</td>
</tr>
</tbody>
</table>

TABLE 2. Upper and lower bounds of the kissing numbers  $C_{6.8}(n)$ . See [66]. Orange cells indicate where AlphaEvolve matched the best results; green cells indicate where AlphaEvolve improved them. (We did not have a framework for deploying AlphaEvolve to establish strong upper bounds.)

Lower bounds on  $C_{6.8}(n)$  can be generated by producing a finite configuration of spheres, and thus form a potential use case for AlphaEvolve. We tasked AlphaEvolve to generate a fixed number of vectors, and we placed unit spheres in those directions at distance 2 from the origin. For a pair of spheres, if the distance  $d$  of their centers was less than 2, we defined their penalty to be  $2 - d$ , and the loss function of a particular configuration of spheres was simply the sum of all these pairwise penalties. A loss of zero would mean a correct kissing configuration in theory, and this is possible to achieve numerically if e.g. there is a solution where each sphere has some slack. In practice, since we are working with floating point numbers, often the best we can hope for is a loss that is small enough (below  $O(10^{-20})$  was enough) so that we can use simple mathematical results to prove that this approximate solution can then be turned into an exact solution to the problem (for details, see [224, 1]).

## 5. Kakeya needle problem.

**Problem 6.9 (Kakeya needle problem).** Let  $n \geq 2$ . Let  $C_{6.9}^T(n)$  denote the minimal area  $|\bigcup_{j=1}^n T_j|$  of a union of triangles  $T_j$  with vertices  $(x_j, 0)$ ,  $(x_j + 1/n, 0)$ ,  $(x_j + j/n, 1)$  for some real numbers  $x_1, \dots, x_n$ , and similarly define  $C_{6.9}^P(n)$  denote the minimal area  $|\bigcup_{j=1}^n P_j|$  of a union of parallelograms  $P_j$  with vertices  $(x_j, 0)$ ,  $(x_j + 1/n, 0)$ ,  $(x_j + j/n, 1)$ ,  $(x_j + (j + 1)/n, 0)$  for some real numbers  $x_1, \dots, x_n$ . Finally, define  $S_{6.9}^T(n)$  to be the maximal “score”

$$\frac{\sum_{i=1}^n |T_i|}{\left(\sum_{i=1}^n \sum_{j=1}^n |T_i \cap T_j|\right)^{1/2} |\bigcup_{i=1}^n T_i|^{1/2}}$$

over triangles  $T_i$  as above, and define  $S_{6.9}^P(n)$  similarly. Establish upper and lower bounds for  $C_{6.9}^T(n)$ ,  $C_{6.9}^P(n)$ ,  $S_{6.9}^T(n)$ ,  $S_{6.9}^P(n)$  that are as strong as possible.

The observation of Besicovitch [28] that solved the Kakeya needle problem (can a unit needle be rotated in the plane using arbitrarily small area?) implied that  $C_{6.9}^T(n)$  and  $C_{6.9}^P(n)$  both converged to zero as  $n \rightarrow \infty$ . It is known that

$$\frac{1}{\log n} \lesssim C_{6.9}^T(n) \leq C_{6.9}^P(n) \lesssim \frac{1}{\log n},$$

with the lower bound due to Córdoba [78], and the upper bound due to Keich [178]. Since  $\sum_{i=1}^n |T_i| = \frac{1}{2}$  and  $\sum_{i=1}^n \sum_{j=1}^n |T_i \cap T_j| \asymp \log n$ , we have

$$C_{6.9}^T(n) \gtrsim \frac{1}{S_{6.9}^T(n)^2 \log n}$$

and similarly

$$C_{6.9}^P(n) \gtrsim \frac{1}{S_{6.9}^P(n)^2 \log n}$$

and so the lower bound of Córdoba in fact follows from the trivial Cauchy–Schwarz bound

$$S_{6.9}^P(n), S_{6.9}^T(n) \leq 1,$$and the construction of Keich shows that

$$1 \lesssim S_{6.9}^P(n), S_{6.9}^T(n).$$

We explored the extent to which AlphaEvolve could reproduce or improve upon the known upper bounds on  $C_{6.9}^T(n)$ ,  $C_{6.9}^P(n)$  and lower bounds on  $S_{6.9}^T(n)$ ,  $S_{6.9}^P(n)$

First, we explored the problem in the context of our search mode. We started with the goal to minimize the total union area where we prompted AlphaEvolve with no additional hints or expert guidance. Here AlphaEvolve was expected to evolve a program that given a positive integer  $n$  returns an optimized sequence of points  $x_1, \dots, x_n$ . Our evaluation computed the total triangle (respectively, parallelogram) area - we used tools from computational geometry such as the `shapely` library; we also validated the constructions using evaluation from first principles based on Monte Carlo or regular mesh dense sampling to approximate the areas. The areas and  $S^T, S^P$  scores of several AlphaEvolve constructions are presented in Figure 5. As a guiding baseline we used the construction of Keich [178] which takes  $n = 2^k$  to be a power of two, and for  $a_i = i/n$  expressed in binary as  $a_i = \sum_{j=1}^k \epsilon_j 2^{-j}$ , sets the position  $x_i$  to be

$$x_i := \sum_{j=1}^k \frac{1-j}{k} \epsilon_j 2^{-j}.$$

AlphaEvolve was able to obtain constructions with better union area within 5 to 10 evolution steps (approximately, 1 to 2 hours wall-clock time) - moreover, with longer runtime and guided prompting (e.g. hinting towards patterns in found constructions/programs) we expect that the results for given  $n$  could be improved even further. Examples of a few of the evolved programs are provided in the [Repository of Problems](#). We present illustrations of constructions obtained by AlphaEvolve in Figures 7 and 8 - curiously, most of the found sets of triangles and polygons visibly have an "irregular" structure in contrast to previous schemes by Keich and Besicovich. While there seems to be some basic resemblance from the distance, the patterns are very different and not self-similar in our case. In an additional experiment we explored further the relationship between the union area and the  $S^T$  score whereby we tasked AlphaEvolve to focus on optimizing the score  $S^T$  - results are summarized in Figure 6 where we observed an improved performance with respect to Keich's construction.

The mentioned results illustrate the ability to obtain configurations of triangles and parallelograms that optimize area/score for a given fixed set of inputs  $n$ . As a second step we experimented with AlphaEvolve's ability to obtain *generalizable* programs - in the prompt we task AlphaEvolve to search for concise, fast, reproducible and human-readable algorithms that avoid black-box optimization. Similarly to other scenarios, we also gave the instruction that the scoring of a proposed algorithm would be done by evaluating its performance on a mixture of small and large inputs  $n$  and taking the average.

At first AlphaEvolve proposed algorithms that typically generated a collection of  $x_1, \dots, x_n$  from a uniform mesh that is perturbed by some heuristics (e.g. explicitly adjusting the endpoints). Those configurations fell short of the performance of Keich sets, especially in the asymptotic regime as  $n$  becomes larger. Additional hints in the prompt to avoid such constructions led AlphaEvolve to suggest other algorithms, e.g. based on geometric progressions, that, similarly, did not reach the total union areas of Keich sets for large  $n$ .

In a further experiment we provided a hint in the prompt that suggested Keich's construction as potential inspiration and a good starting point. As a result AlphaEvolve produced programs based on similar bit-wise manipulations with additional offsets and weighting; these constructions do not assume  $n$  being a power of 2. An illustration of the performance of such a program is depicted in the top row of Figure 9 - here one observes certain "jumps" in performance around the powers of 2; a closer inspection of the configurations (shown visually in Figure 10) reveals the intuitively suboptimal addition of triangles for  $n = 2^k + 1$ . This led us to prompt AlphaEvolve to mitigate this behavior - results of these experiments with improved performance are presented in the bottom row in Figure 9. Examples of such constructions are provided in the [Repository of Problems](#).FIGURE 5. AlphaEvolve applied for optimization of total union area of (top) triangles and (bottom) parallelograms using our search method: (left) Total area of AlphaEvolve's constructions compared with Keich's construction and (right) monitoring the corresponding  $S^T, S^P$  scores for both.

FIGURE 6. AlphaEvolve applied for optimization of the score  $S^T$ : a comparison between AlphaEvolve and Keich's constructions.

One can also pose a similar problem in three dimensions:FIGURE 7. Parallelogram constructions towards minimizing total area for  $n = 16, 32, 64$  (left, middle and right): (Top) Keich's method and (Bottom) AlphaEvolve's constructions.

FIGURE 8. Triangle constructions towards minimizing total area for  $n = 16, 32, 64$  (left, middle and right): (Top) Keich's method and (Bottom) AlphaEvolve's constructions. More examples are provided in the [Repository of Problems](#) .FIGURE 9. AlphaEvolve generalizing Keich’s construction to non-powers of 2. The found programs are based on Keich’s bitwise structure with some additional weighting. (Top) A construction that extrapolates beyond powers of 2 introducing jumps in performance; (Bottom) An example with mitigated jumps obtained by more guidance in the prompt.

**Problem 6.10 (3D Kakeya problem).** Let  $n \geq 2$ . Let  $C_{6.10}(n)$  denote the minimal volume  $|\bigcup_{j=1}^n \bigcup_{k=1}^n P_{j,k}|$  of prisms  $P_{j,k}$  with vertices

$$\begin{aligned} & (x_{j,k}, y_{j,k}, 0), \left(x_{j,k} + \frac{1}{n}, y_{j,k}, 0\right), \left(x_{j,k}, y_{j,k} + \frac{1}{n}, 0\right), \left(x_{j,k} + \frac{1}{n}, y_{j,k} + \frac{1}{n}, 0\right), \\ & \left(x_{j,k} + \frac{j}{n}, y_{j,k} + \frac{k}{n}, 1\right), \left(x_{j,k} + \frac{j+1}{n}, y_{j,k} + \frac{k}{n}, 1\right), \left(x_{j,k} + \frac{j}{n}, y_{j,k} + \frac{k+1}{n}, 0\right), \left(x_{j,k} + \frac{j+1}{n}, y_{j,k} + \frac{k+1}{n}, 1\right) \end{aligned}$$

for some real numbers  $x_{j,k}, y_{j,k}$ . Establish upper and lower bounds for  $C_{6.10}(n)$  that are as strong as possible.

It is known that

$$n^{-o(1)} \lesssim C_{6.10}(n) \lesssim \frac{1}{\log^2 n}$$

asymptotically as  $n \rightarrow \infty$ , with the lower bound being a remarkable recent result of Wang and Zahl [294], and the upper bound a forthcoming result of Iqra Altaf<sup>2</sup>, building on recent work of Lai and Wong [188]. The lower bound is not feasible to reproduce with AlphaEvolve, but we tested its ability to produce upper bounds.

<sup>2</sup>Private communication.FIGURE 10. AlphaEvolve generalizing Keich’s construction to non-powers of 2: (top) illustrating potential suboptimal schemes near powers of 2 where a (right-most) triangle is added “far” from the union; (bottom) prompting AlphaEvolve to pack more densely and mitigate such jumps.

In a similar fashion to the 2D case, we initially explored how the AlphaEvolve search mode could be used to obtain optimized constructions (with respect to volume). The prompt did not contain any specific hints or expert guidance. The evaluation produces an approximation of the volume based on sufficiently dense Monte Carlo sampling (implemented in the `jax` framework and ran on GPUs) - for the purposes of optimization over a bounded set of inputs (e.g.  $n \leq 128$ ) this setup yields a reasonable and tractable scoring mechanism implemented from first principles. For inputs  $n \leq 64$  AlphaEvolve was able to find improvements with respect to Keich’s construction - the found volumes are represented in Figure 11; a visualization of the AlphaEvolve tube placements is depicted in Figure 12.

In ongoing work (for both the cases of 2D and higher dimensions) we continue to explore ways of finding better generalizable constructions that would provide further insights for asymptotics as  $n \rightarrow \infty$ .

## 6. Sphere packing and uncertainty principles.

**Problem 6.11 (Uncertainty principle).** Given a function  $f \in L^1(\mathbb{R})$ , set

$$A(f) := \inf \{r > 0 : f(x) \geq 0 \text{ for all } |x| \geq r\}.$$FIGURE 11. Kakeya needle problem in 3D: improving upon Keich’s constructions in terms of lower volume.

FIGURE 12. Kakeya needle problem in 3D. Examples of constructions of three-dimensional parallelograms obtained by AlphaEvolve: the cases of  $n = 8$  (left) and  $n = 16$  (right).

Let  $C_{6.11}$  be the largest constant for which one has

$$A(f)A(\hat{f}) \geq C_{6.11}$$

for all even  $f$  with  $f(0), \hat{f}(0) < 0$ . Establish upper and lower bounds for  $C_{6.11}$  that are as strong as possible.

Over the last decade several works have explored upper and lower bounds on  $C_{6.11}$ . For example, in [145] the authors obtained

$$0.2025 \leq C_{6.11} \leq 0.353.$$

and established further results in other dimensions. Later on, further improvements in [62] led to  $C_{6.11} \leq 0.32831$  and, more recently, in unpublished work by Cohn, de Laat and Gonçalves (announced in [146]) the authors have been able to obtain an upper bound  $C_{6.11} \leq 0.3102$ .

One way towards obtaining upper bounds on  $C_{6.11}$  is based on a linear programming approach - a celebrated instance of which is the application towards sphere packing bounds developed by Cohn and Elkies [61]. Roughly speaking, it is sufficient to construct a suitable auxiliary test function whose largest sign change is as close to 0 as possible. To this end, one can focus on studying normalized families of candidate functions (e.g. satisfying$f = \hat{f}$  and certain pointwise constraints) parametrized by Fourier eigenbases such as Hermite [145] or Laguerre polynomials [62].

In our framework we prompted AlphaEvolve to construct test functions of the form  $f = p(2\pi|x|^2)e^{-\pi|x|^2}$  where  $p$  is a linear combination of the polynomial Fourier eigenbasis constrained to ensure that  $f = \hat{f}$  and  $f(0) = 0$ . We experimented using both the Hermite and Laguerre approaches: in the case of Hermite polynomials AlphaEvolve specified the coefficients in the linear combination ([145]) whereas for Laguerre polynomials the setup specified the roots ([62]). From another perspective, the search for optimal polynomials is an interesting benchmark for AlphaEvolve since there exists a polynomial-time search algorithm that becomes quite expensive as the degrees of the polynomials grow.

For a given size of the linear combination  $k$  we employed our *search mode* that gives AlphaEvolve a time budget to design a search strategy making use of the corresponding scoring function. The scoring function (verifier) estimated the last sign change of the corresponding test function. Additionally, we explored tradeoffs between the speed and accuracy of the verifiers - a fast and less accurate (*leaky*) verifier based on floating point arithmetic and a more reliable but slower verifier written using rational arithmetic.

As reported in [224], AlphaEvolve was able to obtain a refinement of the configuration in [145] using a linear combination of three Hermite polynomials with coefficients  $[0.32925, -0.01159, -8.9216 \times 10^{-5}]$  yielding an upper bound  $C_{6.11} \leq 0.3521$ . Furthermore, using the Laguerre polynomial formulation (and prompting AlphaEvolve to search over the positions of double roots) we obtained the following constructions and upper bounds on  $C_{6.11}$ :

<table border="1">
<thead>
<tr>
<th><math>k</math></th>
<th>Prescribed Double Roots</th>
<th><math>C_{6.11}</math></th>
</tr>
</thead>
<tbody>
<tr>
<td>6</td>
<td>[3.64273649, 5.68246114, 33.00463486, 40.97185579, 50.1028231, 53.76768016]</td>
<td><math>\leq 0.32831</math></td>
</tr>
<tr>
<td>7</td>
<td>[3.64913287, 5.67235784, 38.79096469, 32.62677356, 45.48028355, 52.97276933, 106.77886152]</td>
<td><math>\leq 0.32800</math></td>
</tr>
<tr>
<td>8</td>
<td>[3.64386938, 5.69329786, 32.38322129, 38.90891377, 45.14892756, 53.11575866, 99.06784500, 122.102121266]</td>
<td><math>\leq 0.327917</math></td>
</tr>
<tr>
<td>9</td>
<td>[3.65229523, 5.69674475, 32.13629449, 38.30580848, 44.53027128, 52.78630070, 98.67722817, 118.22167413, 133.59986194]</td>
<td><math>\leq 0.32786</math></td>
</tr>
<tr>
<td>10</td>
<td>[3.6331003, 5.6714292, 33.09981679, 38.35917516, 41.1543366, 50.98385922, 59.75317169, 94.27439607, 119.86075361, 136.35793559]</td>
<td><math>\leq 0.32784</math></td>
</tr>
<tr>
<td>11</td>
<td>[3.5, 5.5, 30.0, 35.0, 40.0, 45.0, 48.74067499, 50.0, 97.46491651, 114.80158990, 134.07379552]</td>
<td><math>\leq 0.324228</math></td>
</tr>
<tr>
<td>12</td>
<td>[3.6331003, 5.6714292, 33.09981679, 38.84994289, 41.1543366, 43.18733473, 50.98385922, 58.63890192, 96.02371844, 111.21606458, 118.90258668, 141.44196227]</td>
<td><math>\leq 0.321591</math></td>
</tr>
</tbody>
</table>

TABLE 3. Prescribed double roots for different values of  $k$  with corresponding  $C_{6.11}$  bounds

We remark that these estimates do not outperform the state of the art announced in [146] - interestingly, the structure of the maximizer function the authors propose suggests it is not analytic; this might require a different setup for AlphaEvolve than the one above based on double roots. However, the bounds in Table 3 are competitive with respect to prior bounds e.g. in [62] - moreover, an advantage of AlphaEvolve we observe here is the efficiency and speed of the experimental work that could lead to a good bound.

As alluded to above, there exists a close connection between these types of uncertainty principles and estimates on sphere packing - this is a fundamental problem in mathematics, open in all dimensions other than  $\{1, 2, 3, 8, 24\}$  [159, 289, 68, 183].

**Problem 6.12 (Sphere packing).** For any dimension  $n$ , let  $C_{6.12}(n)$  denote the maximal density of a packing of  $\mathbb{R}^n$  by unit spheres. Establish upper and lower bounds on  $C_{6.12}(n)$  that are as strong as possible.FIGURE 13. AlphaEvolve applied towards linear programming upper bounds  $C_{6.13}(n)$  for the center sphere packing density  $\delta$ . Here  $\delta$  is given by  $\Delta(n/2)!/\pi^{n/2}$  with  $\Delta$  denoting the packing's density, i.e. the fraction of space covered by balls in the packing [61]. (Left) Benchmark for lower dimensions with AlphaEvolve matching the Cohn-Elkies baseline up to 4 digits. (Right) Benchmark for higher dimensions with AlphaEvolve improving Cohn-Elkies baselines.

**Problem 6.13 (Linear programming bound).** For any dimension  $n$ , let  $C_{6.13}(n)$  denote the quantity

$$C_{6.13}(n) := \frac{\pi^{n/2}}{\Gamma(n/2 + 1)} \inf_f \frac{(r/2)^n f(0)}{\hat{f}(0)}$$

where  $f$  ranges over integrable continuous functions  $f := \mathbb{R}^n \rightarrow \mathbb{R}$ , not identically zero, with  $\hat{f}(\xi) \geq 0$  for all  $\xi$  and  $f(x) \leq 0$  for all  $|x| \geq r$  for some  $r > 0$ . Establish upper and lower bounds on  $C_{6.13}(n)$  that are as strong as possible.

It was shown in [61] that  $C_{6.12}(n) \leq C_{6.13}(n)$ , thus upper bounds on  $C_{6.13}(n)$  give rise to upper bounds on the sphere packing problem. Remarkably, this bound is known to be tight for  $n = 1, 8, 24$  (with extremizer  $f(x) = (1 - |x|)_+$  and  $r = 1$  in the  $n = 1$  case), although it is not believed to be tight for other values of  $n$ . Additionally, the problem has been extensively studied numerically with important baselines presented in [61].

Upper bounds for  $C_{6.13}(n)$  can be obtained by exhibiting a function  $f$  for which both  $f$  and  $\hat{f}$  have a tractable form that permits the verification of the constraints stated in Problem 6.13, and thus a potential use case for AlphaEvolve. Following the approach of Cohn and Elkies [61], we represent  $f$  as a spherically symmetric function that is a linear combination of Laguerre polynomials  $L_k^\alpha$  times a gaussian, specifically of the form

$$f(x) = \sum_{k \text{ odd}} a_k L_k^\alpha(\pi|x|^2) e^{-\pi|x|^2} \quad (6.4)$$

where  $a_k$  are real coefficients and  $\alpha := n/2 - 1$ . In practice it was helpful to force  $f$  to have single and double roots at various locations that one optimizes in. We had to resort to extended precision and rational arithmetic in order to define the verifier; see Figure 13.

An additional feature in our experiments here is given by the reduced effort to prepare a numerical experiment that would produce a competitive bound - one only needs to prepare the verifier and prompt (computing the estimate of the largest sign change given a polynomial linear combination) leaving the optimization schemes to be handled by AlphaEvolve. In summary, although so far AlphaEvolve has not obtained qualitatively newstate-of-the-art results, it demonstrated competitive performance when instructed and compared against similar optimization setups from the literature.

**7. Classical inequalities.** As a benchmark for our setup, we explored several scenarios where the theoretical optimal bounds are known [198, 124] - these include the Hausdorff–Young inequality, the Gagliardo–Nirenberg inequality, Young’s inequality, and the Hardy–Littlewood maximal inequality.

**Problem 6.14 (Hausdorff–Young).** For  $1 \leq p \leq 2$ , let  $C_{6.14}(p)$  be the best constant such that

$$\|\hat{f}\|_{L^{p'}(\mathbb{R})} \leq C_{6.14}(p) \|f\|_{L^p(\mathbb{R})} \quad (6.5)$$

holds for all test functions  $f : \mathbb{R} \rightarrow \mathbb{R}$ . Here  $p' := \frac{p}{p-1}$  is the dual exponent of  $p$ . What is  $C_{6.14}(p)$ ?

It was proven by Beckner [20] (with some special cases previously worked out in [9]) that

$$C_{6.14}(p) = (p^{1/p}/(p')^{1/p'})^{1/2}.$$

The extremizer is obtained by choosing  $f$  to be a Gaussian.

We tested the ability for AlphaEvolve to obtain an efficient lower bound for  $C_{6.14}(p)$  by producing code for a function  $f : \mathbb{R} \rightarrow \mathbb{R}$  with the aim of extremizing (6.5). Given a candidate function  $f$  proposed by AlphaEvolve, the corresponding evaluator estimates the ratio  $Q(f) := \|\hat{f}\|_{L^{p'}(\mathbb{R})}/\|f\|_{L^p(\mathbb{R})}$  using a step function approximation of  $f$ . More precisely, for truncation parameters  $R_1, R_2$  and discretization parameter  $J$ , we work with an explicitly truncated discretized version of  $f$ , e.g., the piecewise constant approximation

$$f_{R_1,J}(x) := \sum_{j=-J}^{J-1} f(jR_1/J) 1_{[jR_1/J, (j+1)R_1/J)}(x)$$

In particular, in this representation  $f_{R_1,J}$  is compactly supported, the Fourier transform is an explicit trigonometric polynomial and the numerator of  $Q$  could be computed to a high precision using a Gaussian quadrature.

Being a well-known result in analysis, we experimented designing various prompts where we gave AlphaEvolve different amounts of context about the problem as well as the numerical evaluation setup, i.e. the approximation of  $f$  via  $f_{R_1,J}$  and the option to allow AlphaEvolve to choose the truncation and discretization parameters  $R_1, R_2, J$ . Furthermore, we tested several options for  $p = 1 + k/10$  where  $k$  ranged over  $[1, 2, \dots, 10]$ . In all cases the setup guessed the Gaussian extremizer either immediately or after one or two iterations, signifying the LLM’s ability to recognize  $Q(f)$  and recall its relation to Hausdorff–Young’s inequality. This can be compared with more traditional optimization algorithms, which would produce a discretized approximation to the Gaussian as the numerical extremizer, but which would not explicitly state the Gaussian structure.

**Problem 6.15 (Gagliardo–Nirenberg).** Let  $1 \leq q \leq \infty$ , and let  $j$  and  $m$  be non-negative integers such that  $j < m$ . Furthermore, let  $1 \leq r \leq \infty, p \geq 1$  be real and  $\theta \in [0, 1]$  such that the following relations hold:

$$\frac{1}{p} = j + \theta \left( \frac{1}{r} - m \right) + \frac{1-\theta}{q}, \quad \frac{j}{m} \leq \theta < 1.$$

Let  $C_{6.15}(j, p, q, r, m)$  be the best constant such that

$$\|D^j u\|_{L^p(\mathbb{R})} \leq C_{6.15}(j, p, q, r, m) \|D^m u\|_{L^r(\mathbb{R})}^\theta \|u\|_{L^q(\mathbb{R})}^{1-\theta}$$

for all test functions  $u$ , where  $D$  denotes the derivative operator  $\frac{d}{dx}$ . Then  $C_{6.15}(j, p, q, r, m)$  is finite. Establish lower and upper bounds on  $C_{6.15}(j, p, q, r, m)$  that are as strong as possible.

To reduce the number of parameters, we only considered the following variant:**Problem 6.16 (Special case of Gagliardo–Nirenberg).** Let  $2 < p < \infty$ . Let  $C_{6.16}(p)$  denote the supremum of the quantities

$$Q_{6.16}(f) := \frac{\|f\|_{L^p(\mathbb{R})}^{4p}}{\|f\|_{L^2(\mathbb{R})}^{2(p+2)} \|f'\|_{L^2(\mathbb{R})}^{2(p-2)}}$$

for all smooth rapidly decaying  $f$ , not identically zero. Establish upper and lower bounds for  $C_{6.16}(p)$  that are as strong as possible.

A brief calculation shows that

$$C_{6.15}(0, p, 2, 2, 1) = C_{6.16}(p)^{4p}.$$

Clearly one can obtain lower bounds on  $C_{6.16}(p)$  by evaluating  $Q_{6.16}(f)$  at specific  $f$ . It is known that  $Q_{6.16}(f)$  is extremized when  $f(x) = 1/(\cosh x)^{2/(p-2)}$  is the hyperbolic secant function [298], thus allowing for  $C_{6.16}(p)$  to be computed exactly. In our setup AlphaEvolve produces a one-dimensional real function  $f$  where one can compute  $f(x)$  for every  $x \in \mathbb{R}$  - to evaluate  $Q_{6.16}(f)$  numerically we approximate a given candidate  $f$  by using piecewise linear splines. Similarly to the Hausdorff–Young outcome, we experimented with several options for  $p$  in  $(2, 10]$  and in each case AlphaEvolve guessed the correct form of the extremizer in at most two iterations.

**Problem 6.17 (Young’s convolution inequality).** Let  $1 \leq p, q, r \leq \infty$  with  $1/r+1 = 1/p+1/q$ . Let  $C_{6.17}(p, q, r)$  denote the supremum of the quantity

$$Q_{6.17}(f, g) := \frac{\|f * g\|_r}{\|f\|_p \|g\|_q}$$

over all non-zero test functions  $f, g$ . What is  $C_{6.17}(p, q, r)$ ?

It is known [20] that  $Q_{6.17}(f, g)$  is extremized when  $f, g$  are Gaussians  $e^{-\alpha x^2}, e^{-\beta x^2}$  (see [20]) which satisfy  $\alpha/\beta = \sqrt{q/p}$ . Thus, we have

$$C_{6.17}(p, q, r) = C_{6.14}(p)C_{6.14}(q)C_{6.14}(r').$$

We tested the ability of AlphaEvolve to produce lower bounds for  $C_{6.17}(p, q, r)$ , by prompting AlphaEvolve to propose two functions that optimize the quotient  $Q_{6.17}(f, g)$  keeping the prompting instructions as minimal as possible. Numerically, we kept a similar setup as for the Hausdorff–Young inequality and work with step functions and discretization parameters. AlphaEvolve consistently came up with the following pattern that proceeds in the following three steps: (1) propose two standard Gaussians  $f = e^{-x^2}, g = e^{-x^2}$  as a first guess; (2) Introduce variations by means of parameters  $a, b, c, d \in \mathbb{R}$  such as  $f = ae^{-bx^2}, g = ce^{-dx^2}$ ; (3) Introduce an optimization loop that numerically fine-tunes the parameters  $a, b, c, d$  before defining  $f, g$  - in most runs these are based on gradient descent that optimizes  $Q_{6.17}(ae^{-bx^2}, ce^{-dx^2})$  in terms of the parameters  $a, b, c, d$ . After the optimization loop one obtains the theoretically optimal coupling between the parameters.

We remark again that in most of the above runs AlphaEvolve is able to almost instantly solve or guess the correct structure of the extremizers highlighting the ability of the system to recover or recognize the scoring function.

Next, we evaluated AlphaEvolve against the (centered) one-dimensional Hardy–Littlewood inequality.

**Problem 6.18 (Hardy–Littlewood maximal inequality).** Let  $C_{6.18}$  denote the best constant for which

$$\left| \left\{ x : \sup_{h>0} \frac{1}{2h} \int_{x-h}^{x+h} f(y) dy \geq \lambda \right\} \right| \leq \frac{C_{6.18}}{\lambda} \int_{\mathbb{R}} f(x) dx$$

for absolutely integrable non-negative  $f : \mathbb{R} \rightarrow \mathbb{R}$ . What is  $C_{6.18}$ ?

This problem was solved completely in [212, 213], which established

$$C_{6.18} = \frac{11 + \sqrt{61}}{12} = 1.5675208 \dots$$Both the upper and lower bounds here were non-trivial to obtain; in particular, natural candidate functions such as Gaussians or step functions turn out not to be extremizers.

We use an equivalent form of the inequality which is computationally more tractable:  $C_{6.18}$  is the best constant such that for any real numbers  $y_1 < \dots < y_n$  and  $k_1, \dots, k_n > 0$ , one has

$$\left| \bigcup_{1 \leq i \leq j \leq n} [y_j - k_i - \dots - k_j, y_i + k_i + \dots + k_j] \right| \leq 2C_{6.18}(k_1 + \dots + k_n)$$

(with the convention that  $[a, b]$  is empty for  $a > b$ ; see [212, Lemma 1]).

For instance, setting  $n = 1$  we have

$$2k_1 = |[y_1 - k_1, y_1 + k_1]| \leq 2C_{6.18}k_1$$

leading to the lower bound  $C_{6.18} \geq 1$ . If we instead set  $k_1 = \dots = k_n = 1$  and  $y_i = 3i$  then we have

$$3n - 1 = \left| \bigcup_{i=1}^n [y_i - 1, y_i + 1] \cup \bigcup_{i=1}^{n-1} [y_{i+1} - 2, y_i + 2] \right| \leq 2C_{6.18}n$$

leading to  $C_{6.18} \geq 3/2 - 1/2n$  for all  $n \in \mathbb{N}$ . In fact, for some time it had been conjectured that  $C_{6.18}$  was  $3/2$  until a tighter lower bound was found by Aldaz; see [4].

In our setup we prompted AlphaEvolve to produce two sequences  $y = \{y_i\}_{i=1}^n, k = \{k_i\}_{i=1}^n$  that respect the above negativity and monotonicity conditions and maximize the ratio  $Q(y, k)$  between the left-hand and right-hand sides of the inequality. Candidates of this form serve to produce lower bounds for  $C_{6.18}$ . As an initial guess AlphaEvolve started with a program that produced suboptimal  $y, k$  and yielded lower bounds less than 1.

AlphaEvolve was tested using both our search and generalization approaches. In terms of data contamination, we note that unlike other benchmarks (such as e.g. the inequalities of Hausdorff–Young or Gagliardo–Nirenberg) the underlying large language models did not seem to draw direct relations between the quotient  $Q(y, k)$  and results in the literature related to the Hardy–Littlewood maximal inequality.

In the *search mode* AlphaEvolve was able to obtain a lower bound  $C_{6.18} \geq 1.5080$ , surpassing the  $3/2$  barrier but not fully reaching  $C_{6.18}$ . The construction of  $y, k$  found by AlphaEvolve was largely based on heuristics coupled with randomized mutation of the sequences and large-scale search. Regarding the generalization approach, AlphaEvolve swiftly obtained the  $3/2$  bound using the argument above. However, further improvement was not observed without additional guidance in the prompt. Giving more hints (e.g. related to the construction in [4]) led AlphaEvolve to explore more configurations where  $y, k$  are built from shorter, repeated patterns - the obtained sequences were essentially variations of the initial hints leading to improvements up to  $\sim 1.533$ .

## 8. The Ovals problem.

**Problem 6.19 (Ovals problem).** Let  $C_{6.19}$  denote the infimal value of  $\lambda_0(\gamma)$ , the least eigenvalue of the Schrödinger operator

$$H_\gamma = -\frac{d^2}{ds^2} + \kappa^2(s)$$

associated with a simple closed convex curve  $\gamma$  parameterized by arclength and normalized to have length  $2\pi$ , where  $\kappa(s)$  is the curvature. Obtain upper and lower bounds for  $C_{6.19}$  that are as strong as possible.

Benguria and Loss [22] showed that  $C_{6.19}$  determines the smallest constant in a one-dimensional Lieb–Thirring inequality for a Schrödinger operator with two bound states, and showed that

$$\frac{1}{2} < C_{6.19} \leq 1,$$with the upper bound coming from the example of the unit circle, and more generally on a two-parameter family of geometrically distinct ovals containing the round circle and collapsing to a multiplicity-two line segment. The quantity  $C_{6.19}$  was also implicitly introduced slightly earlier by Burchard and Thomas in their work on the local existence for a dynamical Euler elastica [50]. They showed that  $C_{6.19} \geq \frac{1}{4}$ , which is in fact optimal if one allows curves to be open rather than closed; see also [51].

It was conjectured in [22] that the upper bound was in fact sharp, thus  $C_{6.19} = 1$ . The best lower bound was obtained by Linde [199] as  $(1 + \frac{\pi}{\pi+8})^{-2} \sim 0.60847$ . See the reports [2, 7] for further comments and strategies on this problem.

We can characterize this eigenvalue in a variational way. Given a closed curve of length  $2\pi$ , parametrized by arclength with curvature  $\kappa$ , then

$$\lambda_0 = \inf_{\Phi \neq 0} \frac{\int |\Phi'|^2 + \kappa^2 |\Phi|^2 ds}{\int |\Phi|^2 ds}$$

The eigenvalue problem can be phrased as the variational problem:

$$I[x, \phi] := \int_0^{2\pi} (\phi'^2 + |x''|^2 \phi^2) ds, \quad (6.6)$$

$$\lambda_0 = \inf_{x, \phi} \left\{ I[x, \phi] \mid x \in W^{2,2}(S^1 \rightarrow \mathbb{R}^n), \phi \in W^{1,2}(S^1), |x'| = 1, \|\phi\|_{L^2}^2 = 1 \right\},$$

where  $W^{2,2}$  and  $W^{1,2}$  are Sobolev spaces.

In other words, the problem of upper bounding  $C_{6.19}$  reduces to the search for three one-dimensional functions:  $x_1, x_2$  (the components of  $x$ ), and  $\phi$ , satisfying certain normalization conditions. We used splines to model the functions numerically - AlphaEvolve was prompted to produce three sequences of real numbers in the interval  $[0, 2\pi)$  which served as the spline interpolation points. Evaluation was done by computing an approximation of  $I[x, \phi]$  by means of quadratures and exact derivative computations. Here for a closed curve  $c(t)$  we passed to the natural parametrization by computing the arc-length  $s = s(t)$  and taking the inverse  $t = t(s)$  by interpolating samples  $(t_i, s_i)_{i=1}^{10000}$ . We used JAX and scipy as tools for automatic differentiation, quadratures, splines and one-dimensional interpolation. The prompting strategy for AlphaEvolve was based on our standard search approach where AlphaEvolve can access the scoring function multiple times and update its guesses multiple times before producing the three sequences.

In most runs AlphaEvolve was able to obtain the circle as a candidate curve in a few iterations (along with a constant function  $\phi$ ) - this corresponds to the conjectured lower bound of 1 for  $\lambda_0(\gamma)$ . AlphaEvolve did not obtain the ovals as an additional class of optimal curves.

**9. Sendov's conjecture and its variants.** We tested AlphaEvolve on a well known conjecture of Sendov, as well as some of its variants in the literature.

**Problem 6.20 (Sendov's conjecture).** For each  $n \geq 2$ , let  $C_{6.20}(n)$  be the smallest constant such that for any complex polynomial  $f$  of degree  $n \geq 2$  with zeros  $z_1, \dots, z_n$  in the unit disk and critical points  $w_1, \dots, w_{n-1}$ ,

$$\max_{1 \leq k \leq n} \min_{1 \leq j \leq n-1} |z_k - w_j| \leq C_{6.20}(n).$$

Sendov [256] conjectured that  $C_{6.20}(n) = 1$ .

It is known that

$$1 \leq C_{6.20}(n) \leq 2^{1/n},$$

