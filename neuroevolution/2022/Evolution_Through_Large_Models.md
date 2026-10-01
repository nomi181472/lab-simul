# Evolution through Large Models

Joel Lehman  
OpenAI  
joel@openai.com

Jonathan Gordon  
OpenAI  
gordonjo@openai.com

Shawn Jain  
OpenAI  
jains@openai.com

Kamal Ndousse  
Anthropic\*  
kamal.ndousse@gmail.com

Cathy Yeh  
OpenAI  
cathy@openai.com

Kenneth O. Stanley  
OpenAI  
kennethstanley@gmail.com

June 20, 2022

## Abstract

This paper pursues the insight that large language models (LLMs) trained to generate code can vastly improve the effectiveness of mutation operators applied to programs in genetic programming (GP). Because such LLMs benefit from training data that includes sequential changes and modifications, they can approximate likely changes that humans would make. To highlight the breadth of implications of such *evolution through large models* (ELM), in the main experiment ELM combined with MAP-Elites generates hundreds of thousands of functional examples of Python programs that output working ambulating robots in the Sodarace domain, which the original LLM had never seen in pre-training. These examples then help to bootstrap training a new conditional language model that can output the right walker for a particular terrain. The ability to bootstrap new models that can output appropriate artifacts for a given context in a domain where zero training data was previously available carries implications for open-endedness, deep learning, and reinforcement learning. These implications are explored here in depth in the hope of inspiring new directions of research now opened up by ELM.

## 1 Introduction

For many in the evolutionary computation (EC) community, the rise of deep learning (DL) has raised questions on its implications for EC. Both approaches

---

\*Work done at OpenAI.scale well with compute and both can yield useful discoveries and meaningful surprises. Yet are they ultimately competing paradigms, or rather are they complementary? In this paper we explore the latter possibility, of considerable synergy, by highlighting an untapped implication of large language models (LLMs; [1, 2]) for both genetic programming (GP; [3, 4]) and open-endedness [5–7].

In particular, in this new Evolution through Large Models (ELM) approach, an LLM trained on code can suggest mutations that are intelligent, thereby facilitating a dramatically more effective mutation operator that sidesteps many of the challenges that previously existed for evolving programs [8]. Interestingly, the benefits of ELM are also reciprocal back to deep learning: the set of samples generated through the LLM can eventually constitute a new training set in a novel domain that can then fine-tune the LLM to perform well in the new domain, a novel data-generation procedure. Furthermore, this approach ultimately opens up new opportunities in the pursuit of open-endedness by increasing the generative capabilities of the LLM solely through its own generated data.

LLMs have recently yielded impressive results in automated code generation [9, 10]. These models bootstrap from human knowledge by learning from very large datasets to achieve general coding competency. The fact that such bootstrapping is possible is clearly relevant to GP. After all, GP is in effect a generative approach to programming. While it might seem at first glance that LLMs therefore might out-compete or subsume GP, in fact GP does still offer an advantage in situations where the particular class of programs targeted by the search is far (or even completely lacking) from the training distribution of the LLM. In such cases, the LLM offers limited recourse (prompt engineering to learn an entirely new domain would be prohibitive), while GP can in principle evolve in any space (though in practice some spaces may be intractable due to the amount of mutation necessary to get consistent signal on fitness).

Interestingly (and perhaps surprisingly), the best of both worlds is easily attainable by combining them: simply by prompting the LLM to generate changes the LLM can serve as a highly sophisticated mutation operator embedded within an overarching evolutionary algorithm. This way, the LLM in concert with evolution can steer each other towards the right region of the solution space even though neither evolution with a conventional mutation operator nor the LLM on its own could generate anything close. In effect, program evolution using LLM-based perturbation begins to bridge the divide between evolutionary algorithms and those that operate on the level of human ideas. That is, LLMs can be trained to approximate how humans intentionally change programs, while staying on the manifold of functionality. Furthermore, such LLMs can be further fine-tuned on successful perturbations for the purposes of self-improvement, culminating in a novel technique for iteratively enhancing the performance of ELM.

To highlight the potential of this approach, in this paper an entire dataset in a novel domain is generated from only a single mediocre starting example designed by hand by humans. In particular, the domain is called Sodarace[11, 12], where two-dimensional ambulating robots of arbitrary morphology are constructed for diverse terrains. Sodarace is cheap to simulate, allowing fast iteration, and also makes it easy to appreciate the sophistication of designs intuitively by simply watching the robot walk. In this way, it facilitates quick assessment of whether a design is successful both quantitatively and qualitatively.

To make the contribution of ELM explicit in the experiments in this paper, the Sodaracers are encoded as raw Python programs that output an enumeration of the ambulating robots’ components. That way, it is possible to demonstrate that ELM is a form of GP that can operate on a modern programming language directly, with no special provisions needed beyond the generic (i.e. *not* previously trained in Sodarace) existing code-generating LLM.

A final important insight unlocked by this approach is that the ability to generate diverse solutions in a domain or part of the search space where there was little to no training data is foundational to bootstrapping an open-ended process [6, 13, 14]. After all, open-endedness is fundamentally about searching outside the distribution of previous experience, which is exactly what ELM helps the LLM to do. Because this novel capability has potentially far-reaching implications, we have chosen in this work to focus on the implications of the generated data that can be produced by ELM. Of course, ELM is applicable in many other contexts that will undoubtedly be explored in the future.

More specifically, experiments that follow show that generated data is sufficiently rich that it can serve as training data for fine-tuning LLMs to generate code for viable new Sodaracers consistently, and furthermore that reinforcement learning (RL) can even fine-tune an augmented LLM to output Sodaracers *conditionally*, depending on the terrain. In the future, such conditional invention has the potential to unlock entirely new kinds of open-ended processes, just as humans have open-endedly built civilization over centuries by conditionally inventing its constituents.

In short, the main contributions of this paper are (1) the ELM method for efficiently evolving programs through LLMs, (2) a technique for improving ELM’s ability to search over time by fine-tuning its LLM-based mutation operator, (3) a demonstration of ELM in a domain not included in the training data of the LLM, and (4) validation that data generated through ELM can bootstrap enhanced LLMs that offer a novel path towards open-endedness.

## 2 Background

This section reviews previous work in genetic programming, large language models, and open-endedness.

### 2.1 Genetic Programming

The field of genetic programming (GP) applies evolutionary algorithms to evolve computer programs to solve problems [3, 4, 15]. The promise of GP is that com-puter code is a computationally universal representation that underlies much modern technology, including artificial intelligence itself. Therefore it is conceivable for GP to automatically evolve programs that achieve human-level (or beyond) performance across diverse application domains [16]. However, there are obstacles in practice to its successful and widespread application to challenging problems.

One obstacle is that scaling GP to evolve increasingly complicated programs can be challenging [8], and that effectively applying GP to a new domain can require significant domain expertise. A researcher often must explicitly specify what functions, variables, and control structures are available to evolution [3, 17], which limits what ultimately can be evolved. In contrast, a human programmer can open-endedly decide what libraries to import and how to write many interdependent subroutines or classes. Research aims to lift these constraints, often through enabling modular reuse of code: e.g. through automatically defined functions [3], data-mining populations to find common sub-components [18], or attempts to use solutions to previous problems when solving new ones [19]. However, no method yet enables GP to scalably operate on human-designed programming languages with a minimum of domain-specific tweaking.

A second obstacle is that nearly all GP methods explore through random perturbations of code, unlike humans, who through active practice improve their proficiency in making deliberate, complex, and coupled modifications to programs [20, 21]. Unlike perturbing e.g. neural network weights, wherein continuous parameters subject to small enough perturbations can predictably generate small changes in functionality [22, 23], perturbing code requires discrete changes that often dramatically shift functionality [24], thereby complicating search. While there exist approaches towards more directed generation of offspring (e.g. building probabilistic models of high-performing programs [25], evolving reproduction operators [26], or applying less-impactful mutation operators [24]), the problem remains at core unsolved.

In contrast to GP, humans learn to reason about code in its full complexity through experimentation and learning. This iterative effort leaves a permanent signature in repositories of code, such as GitHub. The next section describes progress in training large language models upon such repositories as a potential way to bypass the above obstacles.

## 2.2 Large Language Models

Large language models (LLMs; [1, 2, 27]), trained on internet-scale data, have progressed at an impressive pace in recent years. The main idea (in autoregressive models such as GPT-3 [2]) is to train increasingly-large neural networks (built on the popular transformer architecture [28], sometimes with billions of parameters) on the seemingly simple task of next-token prediction (i.e. given a sequence of tokens seen so far, predict the proceeding token). Scaling such LLMs (and formulating problems of interest as natural language processing tasks) has resulted in groundbreaking performance across a wide range of tasks[2, 29], including program synthesis [9, 10, 30].

In particular, by training LLMs on large-scale coding data, e.g. from GitHub, it is possible to produce models with impressive function-synthesis capabilities [9, 10], highlighting the possibility to bootstrap the ability to fluently code from large-scale data. A further development are *diff* models that are trained on diffs from GitHub [31]. A diff is an incremental change to a file that is committed to a version control system such as GitHub, accompanied by a *commit message* describing the intent of the change. In this way, diff models are trained how, given a piece of code and any potential commit message, to suggest an informed *change*. Through the lens of evolutionary algorithms, such diff models can be viewed as intelligent perturbation operators, providing a way to walk over the manifold of code (in a controllable way) through mimicking human programmers. An interesting further possibility is that such models are amenable to further training through gradient descent, implying a potentially-powerful mechanism for self-adaptation (e.g. through reinforcing successful diffs during evolution). Both diff models and their capacity for self-adaptation are explored in this work as a way to improve GP. However, it is also important to note that general language models not trained directly on diffs can also act in effect like diff models when given the right kinds of prompts (see Section 3.1).

### 2.3 Open-endedness

With origins in the open-ended evolution community [6, 13, 32, 33] within artificial life, the field of open-endedness seeks to create algorithmic systems that produce never-ending innovation [5]. Given the primacy of search with ML, research within open-endedness naturally has focused on refining algorithms for open-ended search, such as those driven by novelty [34, 35] or curiosity [36, 37]. While such focus has indeed lead to algorithmic progress, there is a growing awareness of the criticality of the *environment* in which open-ended algorithms are applied [38–41].

That is, the environment limits what can arise within the system and for how long its products can remain interesting. As a result, some have argued for more complex environments for open-endedness, such as video games [38, 39], and others have argued that features of the environment should co-evolve with agents [40, 42]. Yet a theory for what specific forms of additional such complexity is needed for enduring open-endedness has been lacking. This paper contributes a possible theory, arguing that agents outputting inventions into the environment in response to previous inventions may be a principled route to such continuing open-endedness.

One challenge in evolving aspects of the environment (such as inventions), is how they are encoded. Most research applies encodings that are specifically fit to describe some fixed part of a larger environment, e.g. a fixed way of describing edges within a maze [43], or the shape of a 2-D landscape [40]. While sometimes the *encodings* of these parts are universal (e.g. the CPPN encoding of landscapes in [40] can describe any landscape, and the RNN encoding of Dennis et al. [42] can describe any maze), it is unclear how to expand the representationto include more of the environment without relying upon ad-hoc principles. This paper argues that computer programs are a general and powerful encoding for continually expanding the richness of an existing environment.

### 3 Approach: Evolution through Large Models

Three distinct components facilitate ELM. First is the novel mutation operator driven by an LLM. Second is an evolutionary outer loop that calls this mutation operator. Finally, the third component is a method for updating the LLM to improve based on its preceding performance. Each of these is detailed in this section.

#### 3.1 Mutation through Diff

The main idea behind ELM centers on rethinking the mutation operator for code by exploiting the capabilities of LLMs. In conventional GP, the language of the code and the types of changes allowed through mutation are both chosen intentionally to yield a reasonable chance that perturbations can lead to useful functional changes [3]. In contrast, LLMs unlock an entirely different basis for mutation: it would be more ideal if the mutation operator *understood* the code and how it can be changed in interesting ways, more like a human than a stochastic event.

LLMs can indeed be trained to output code in an autoregressive manner by exposing them to extensive programming examples [9, 10]. A *diff model* [31] can similarly be autoregressively trained on a collection of code diffs (e.g. from GitHub). Each diff targets a single file, where the file and diff are short enough to fit into the context of the LLM. The model is trained to predict the diff (formatted, for example, in unified diff format [44]) from the concatenation of the file and the commit message, where the loss includes only the tokens that make up the diff, thereby encouraging the model to predict the diff but not to memorize the file and commit message. In other words, the model learns to predict plausible changes to code from examples of changes made to code by human programmers. It is important to note that the idea of diff models (or their initial training) [31] is not a contribution of this paper, but diff models are rather a tool applied here in a new context (to produce mutations).

To achieve meaningful mutations, ELM can choose among a set of *commit messages*, which convey to the LLM the details of the operation it should perform in lieu of mutation. These messages offer significant power and nuance for calibrating mutation operators that is likely highly novel to anyone familiar with implementing mutation in GP or evolutionary algorithms in general. In the experiment in this paper, the three commit messages and their respective probabilities of being chosen are:

- • Changed `make_walker` function. (40% chance)
- • Changed parameters in `make_walker` function. (30% chance)- • Small change to `make_walker` function. (30% chance)

Of course, any commit message is conceivable. The LLM’s ability to interpret general natural language means that the scope for research exploration (and domain-specificity) here is vast.

As a simple experiment to highlight diff models’ ability to intelligently modify code, an implementation of a function with an adjustable amount of bugs is perturbed with either a simple GP mutation operator or with a 300M parameter diff model. The hypothesis is that an intelligent perturbation operator will be better able to make multiple correlated changes to code (in this case to correct the bugs). The *4-Parity* task (which is inspired by a standard GP benchmark [3]) serves as a representative test-bed. Note that a correct implementation of 4-Parity returns the sum of the four input bits, modulo two. Up to five bugs are introduced to 4-Parity, first by incrementally misnaming each of the variables in the sum calculation; and for the fifth bug, the modulo is changed from two to three. Then, perturbation operators are tested for their ability to (in one perturbation step) change the buggy version of the code to one that successfully passes unit tests. Results in figure 1 highlight how with increasing bugs GP mutation becomes exponentially more unlikely to produce a successful solution (note that *no* mutation from GP solves all five, given 100,000 trials). In contrast, the diff operator is able to fix all five bugs, and its performance is impacted more by the number of *different* types of bugs (i.e. the fifth bug affects the modulo calculation rather than renaming variables) than by the raw number of bugs itself. Further details (including a supporting experiment with another task with similar results) are given in Appendix A.

Because the tools involved in an ELM implementation are unconventional, we finally wanted to highlight here several alternatives for implementing such systems in practice today. One option is to use models available on the OpenAI API that can edit through following instructions [45, 46]. A second option is to create an intelligent mutation operator through few-shot prompting instead of through explicit training (as in the diff model). That is, one could design prompts for a model trained on code (like Codex [9] or GPT-6-J [47]). To show the potential to replicate (or improve upon) the results in this paper, we conducted a simple experiment comparing (on the 4-Parity problem) prompt engineering and edit mode to the diff model. Figure 2 shows how models from the API outperform the diff model used in the paper. Further experimental details can be found in Appendix A.

### 3.2 The Evolutionary Algorithm and Implications for Open-Endedness

Because the mutation operator is effectively a modular component for many evolutionary algorithms [48, 49], ELM can be implemented within a diversity of contexts. Of course, the approach is most applicable to a case where the genetic encoding is through a known programming language, because that is how the benefits of the LLM will be realized. Genetic encodings in natural languageFigure 1: **Comparing diff mutation to GP mutation in fixing 4-Parity bugs.** The figure shows how the ability of a single mutation to produce correct solutions changes as bugs are incrementally added to a working 4-Parity implementation. Note that success percentage is shown in *log scale*, i.e. success for GP mutation decreases exponentially in the number of mutations (and produces no solutions when there are five bugs). In contrast, diff mutation degrades only with the fifth bug. The conclusion is that LLM-based mutation can indeed make multiple sensible coupled changes to code.

or any other language at which LLMs excel are also conceivable, but of course the utility of such encodings would depend on how they are applied and their mapping to a potentially useful phenotype. The experiments in this paper focus on Python 3 genotypes, which are also by their nature variable in length. The ability to use modern programming languages as genotypes without the need for any special accommodation is a key benefit of ELM.

While there are many options for the evolutionary algorithm in the outer loop, we have chosen in this paper to implement ELM within a quality diversity (QD) algorithm [50, 51]. An important motivation for this choice is that the emergence of the ability to search intelligently for arbitrarily complex programs is tantalizingly close to overcoming some of the key obstacles to open-endedness [14], and ELM is an opportunity to highlight this opportunity.

Recall that we do not yet know how to make an algorithm that exhibits genuinely open-ended divergence. While there has been progress towards open-endedness in recent years, the state of the art remains *weak open-endedness*, wherein novel and interesting discovery continues only for a brief time, eventually ending in a plateau as the possibilities are exhausted [5, 40, 43, 52–54]. In contrast, in *strong open-endedness*, the process would never plateau—if we left and returned a year later, or even a *million* years later, its products would continue to become more interesting over time. No algorithm comes close to such an achievement, though it is evidently possible in nature.

The question then is what stands between today’s algorithms and tractable strong open-endedness. This gap remains despite that recent work in open-Figure 2: **Comparing alternate LLM-based mutations in fixing 4-Parity bugs.** The performance of different mutation operators in fixing bugs is shown as bugs are incrementally added to a working 4-Parity implementation. Both edit mode and prompt-engineering approaches outperform the diff model applied in this paper’s experiments. The conclusion is that both prompt-engineering on LLMs trained on code and using edit mode models from the OpenAI API are viable options to build upon the work in this paper.

endedness appears to make progress. For example, the Enhanced POET algorithm continues to generate diverse and increasingly complex terrains for bipedal robots to solve [40]. In their hide-and-seek experiment, Baker et al. [54] show agents discovering increasingly complex strategies like assembling blocks into a hideout. Yet despite such algorithms clearly demonstrating the capability to continue to invent new solutions, all such demonstrations share a singular downfall: they slow down and eventually end. Formalizing ELM within a QD framework in effect offers a novel opportunity to address this challenge.

This opportunity connects to the difficulty of formulating an artificial environment that imposes no limit on what even the most capable open-ended algorithm can achieve, as noted in the Background. The challenge of devising artificial environments with unbounded potential raises the intriguing question of what property our universe and planet possess that is lacking in current artificial environments. This question is critically important for open-endedness because in the absence of that property, open-ended algorithms cannot demonstrate their full potential. If the problem indeed stems from the fact that artificial environments to date offer only finite possible experiences until their potential is exhausted, then to overcome this bottleneck the environment itself needs to possess the potential to change forever.

Since the emergence of intelligence in nature, much environmental change has been driven by the intelligent agents themselves. Eventually, humans acquired the ability to leave *detached* artifacts in the environment that radically alter its potential for themselves and other agents, like a house, a vehicle, or even a *program*. Unlike new organisms that are evolved over generations, such *detached conditional things* (DCTs) are generated intentionally as a conditionof the observations of the agent. Once DCTs enter the world, open-endedness accelerates because the environment is rapidly updating even within the course of a single lifetime.

Each DCT creates an opportunity for further DCTs. For example, the invention of the door creates the opportunity for keys to be invented, which then sets the stage for lock picks, and so on. And because they are detached, DCTs can leave a permanent legacy in the environment well beyond the lifetime of their inventor. In this way, invention in the era of DCTs is open-ended, and accordingly has continued for thousands of years, from fire and wheels to space stations and computers.

This theory of DCTs supplies an abstract answer to the problem of a limited environment: Agents must be able to imprint the environment with DCTs in response to those already present within it. However, realizing DCTs in practice requires addressing a separate question: how can agents be enabled to efficiently invent DCTs of limitless complexity in a new domain?

Interestingly, computer programs are universal representations, meaning that the procedure of assembling new artifacts can naturally be described algorithmically. For example, programmers have leveraged code to help create enormously complex artifacts (like the layouts of computer chips or instructions for 3-D printers to produce complex physical objects). Of course, programs themselves can function as DCTs. In this way, a procedure that can search through modern program space and ultimately generate such programs conditionally is a candidate for creating open-ended environments of unlimited capacity. The experiment in this paper will demonstrate in more detail how ELM makes such a construct conceivable; the significance of QD is that its ability to generate a diverse space of artifacts can serve as the bootstrap to obtaining agents capable of generating DCTs. In short, the QD algorithm is generating *training data* that can transform the LLM into a kind of DCT generator.

While any QD algorithm can work with ELM, the specific algorithm in the experiment in this paper is MAP-Elites [51, 55] (Figure 3). The core of MAP-Elites is a uniformly-spaced grid of niches (called the *map*), that spans user-specified dimensions of solution diversity, called the *behavior characterization*. Upon initialization, a single pre-existing (hand-designed in this paper) solution is evaluated and placed into the map. In each iteration thereafter, an inhabited niche is randomly chosen and the solution within that niche is perturbed by the diff model and evaluated. The new candidate solution is assigned its niche from its behavior characterization, and if that niche is unfilled or the new solution outperforms the niche’s current inhabitant, it becomes the champion of that niche; otherwise, the candidate is discarded. In this way, over iterations of search, the map gradually fills with increasingly diverse and high-quality solutions.

### 3.3 Fine-tuning the Diff Operator

Interestingly, because the mutation (diff) operator is itself an LLM, it has the potential to be improved with respect to the domain. While self-adaptation [56–58] has a long tradition in evolutionary computation, including algorithms suchThe diagram illustrates the MAP-Elites with ELM process. It features a central grid labeled 'Map of Diverse Champions' with axes 'Height of Sodaracer' and 'Width of Sodaracer'. Two red arrows originate from the grid and point to two separate 'Python Program' boxes. These two 'Python Program' boxes are connected by a vertical 'Diff Model' box, with red arrows pointing upwards from the bottom 'Python Program' to the 'Diff Model' and then to the top 'Python Program'.

Figure 3: **MAP-Elites with ELM**. In each iteration, an existing Python solution is sampled from the map of solutions for each independent replica of a diff model. Each replica generates a batch of diffs that are applied to the sampled solution to generate modified candidate solutions. These candidates are evaluated and are then inserted into the map if they either establish a new niche or outperform the niche’s current champion. Over iterations, a single initial seed program gives rise to a diversity of high-performing Python programs.

as CMA-ES [58] and natural evolution strategies [59], the kinds of improvements possible in ELM are unique by offering the possibility of the LLM learning *how to think about change*. That is, ideas for changes that are most promising in one domain might be different than in another, and the richness of the LLM offers the potential to capture such nuance through experience. In particular, the pre-trained diff model can be trained further (which is called *fine-tuning*) with accepted diffs (by MAP-Elites) from initial iterations or runs of ELM. That way, the diff operator updates to understand better the kinds of modifications that lead to either higher quality, more novelty, or both. This fine-tuning technique can cause ELM itself to improve over iterations. Of course, over a long run, the ideal kinds of changes might change; continually fine-tuning based on recent experience can potentially track such drifting opportunities. In this paper, the potential of fine-tuning is demonstrated through a single fine-tuning iteration, but the investigation of such continual refinement is an open research opportunity.nity. Note that the prompt-engineering approach to LLM mutation described at the end of Section 3.1 can also benefit from fine-tuning in this way.

## 4 Experiment and Results

The primary motivation for the experiment that follows is to give a taste of the breadth of implications of ELM, to evolutionary computation, to deep learning, and to open-endedness. For this purpose, this experiment focuses on the problem of the *invention* of complex artifacts (which could eventually serve as DCTs in a future more ambitious experiment). While the potential scope of applications for ELM is broad, the opportunity to learn to invent complex artifacts in an arbitrary domain extends directly from the augmented ability to search through programs; seeing this inventive capability realized thereby highlights novel opportunities opening up.

The experiment will aim to bootstrap from a few hand-written (and barely functional) examples of an invention into an LLM-based inventor that can fluidly output appropriate inventions conditioned on its environment. This concept is demonstrated in the domain of *Sodarace* [11, 12], a physics-based invention domain that serves as a cheap-to-simulate microcosm of invention. The goal in Sodarace is to construct from collections of masses and oscillating springs two-dimensional robots that can locomote competently. A wide range of interesting Sodaracer robots are possible, as highlighted by previous ML research [12] and the origins of the domain: Sodarace began as a web application called Sodaconstructor, wherein the human design of Sodaracers was sufficiently compelling for an online community to form around the endeavor [11].

An individual Sodaracer (Figure 4) is composed of a variable-sized collection of point masses (each fully-described by its initial 2-D position) and oscillating springs that connect masses together. The motion of the robot is driven by the oscillation of its springs, and each spring has parameters specifying the amplitude and phase of its oscillation (by convention all springs have the same period). To evaluate a particular Sodaracer, it is simulated in a specific terrain for a fixed amount of time and its ability to traverse that terrain is measured (i.e. how far the Sodaracer’s center of mass moves along the x-axis); additionally, to measure the diversity of solutions for MAP-Elites, features capturing gross aspects of the robot’s morphology (i.e. its initial height, width, and total mass) are recorded. While a search algorithm could operate directly in the space of masses and springs, here instead LLMs output Python code that describes the morphology of the Sodaracer. For examples of such source code, see Appendix B and G. In this way, the programs evolved by ELM are in effect *indirect encodings* [60–63] for Sodaracers. That is, in principle any indirect encoding expressible through code could be evolved from scratch or modified by ELM.

More ambitiously than only generating a repertoire of Sodaracer designs, the experiment will attempt to implement an entire *invention pipeline* that ultimately yields a novel kind of conditional LLM that can input a terrain and output an appropriate Sodaracer for that terrain. ELM thereby serves as theFigure 4: **An Example Sodaracer.** The objective in the Sodarace domain is to design a Sodaracer that locomotes effectively across the ground terrain. Labeled in the image are examples of a mass and a spring that connects two masses together. A Sodaracer design consists of a variable number of masses and springs, where springs also have oscillatory parameters that determine the Sodaracer’s motion.

initial *data generation phase* of this pipeline, showing in this way how ELM can serve in general as a way of generating domain data for downstream deep learning where it did not previously exist. Furthermore, in the future the ability to train such conditional inventors could serve as a foundation for an open-ended world of DCT-generating agents.

In practice, the aim of the invention pipeline is to create an agent that can output complex artifacts conditionally, based on its observation of the environment. If *invention* is conceived as an action, then learning to invent conditionally can be viewed as a reinforcement learning (RL) problem [64]. That is, for any given observation, the agent can be rewarded depending on the success of the resultant invention. For example, in Sodarace, the agent might observe a specific terrain such as a hill and then output a design for a Sodaracer artifact. The reward then depends upon the performance of the Sodaracer in the observed terrain.

This approach sounds straightforward—it is simply RL with complex outputs—but there is a problem. If the agent has no prior experience in the domain (e.g. in Sodarace), then outputting even a valid (let alone working) artifact is effectively impossible. As a result, there is no gradient for RL and it cannot bootstrap into the new domain.

Therefore, to get RL started, some form of pretraining is necessary. In effect, the RL fine-tuning described above is actually the last step in a *pipeline*, where the preceding step is to teach the agent something preliminary about its domain. For that purpose, an LLM can be trained on a large set of *examples* from the target domain. For example, these examples could be Sodarace walker designs. After exposure to enough such designs, in principle the LLM knows something about the domain and can output sample artifacts from the training distribution. With such knowledge later passed on to RL, it should now bepossible to bootstrap into conditional fine-tuning.

However, there is *still* a problem: where did all the examples come from for training the LLM? If the hope is for the conditional inventor eventually to invent in a novel domain like Sodarace where a generic LLM is unlikely to have any exposure, then the source for all the examples needed to train the LLM is itself elusive. As a consequence, the pipeline needs yet one more preceding step—which is where ELM comes in—to generate a set of example artifacts from scratch, which could then train the LLM that will eventually bootstrap RL.

Generating a diverse and large set of initial training examples is a search problem. However, because no LLM yet has any exposure to the right kind of data, it is a search problem within the invention space rather than within the weight space of neural networks. Searching for diverse functional examples (to get a wide pre-training distribution) within the space of artifacts is the natural role of QD (i.e. MAP-Elites in this paper). Combined with the diff function, the result is ELM, which yields a novel approach to generating training examples, thereby bootstrapping the entire process.

To recap, what emerges is a three-stage *invention pipeline* for training conditional inventors (Figure 5):

1. 1. **ELM.** Search for a diverse set of example artifacts (e.g. Sodaracers on flat ground).
2. 2. **Pre-train the LLM with examples from ELM.** The LLM accordingly learns to output example inventions from the training distribution.
3. 3. **Learn to invent conditionally.** Splice new conditional inputs onto the LLM and fine tune it through RL to produce appropriate inventions for the conditions it observes.

## 4.1 Encoding Sodaracers with Python

Previous experiments targeting Sodarace have leveraged specialized evolutionary encodings [12]. Instead, in this work plain-text Python code acts as a generic representation for inventions. By showing how Python can be used to represent artifacts in an arbitrary domain, it opens up the possibility of using it as a generic encoding in diverse future domains. More specifically, in the experiments an individual is evaluated by executing its code through the Python interpreter. The product of the interpreter (for a viable individual) is a data structure containing the description of a Sodaracer (i.e. a Python dictionary containing lists of both point masses and springs), which can then be passed to the Sodarace simulator to evaluate the encoded Sodaracer’s behavior. Note that Sodaracers are encoded in Python throughout the invention pipeline, i.e. ELM evolves Python programs and the language models in both latter stages of the pipeline are trained to output Python programs.

Preliminary experiments showed that the diff model’s initial performance (i.e. before fine-tuning) in creating useful perturbations depended upon the design of the “interface” through which Sodaracers were procedurally constructed.```

graph LR
    subgraph Training_Pipeline [Training Pipeline]
        direction LR
        A[Run QD algorithm to generate initial dataset (ELM)] --> B[Pre-train LM on many examples in simple starting context]
        B --> C[Conditional RL on LM]
    end

    subgraph Invention_Pipeline [Invention Pipeline]
        direction LR
        D[Conditional Inventors] --> E[Open-ended Process]
    end

    D --- A
    D --- B
    D --- C
    D --- E

    subgraph Sodarace_Domain [Sodarace Domain]
        direction TB
        F[Invention Spec] --> G[LM]
        G --> H[Terrain]
    end

```

Figure 5: **The Invention Pipeline.** (left) A three-staged training pipeline bootstraps from a single example of an invention to an LLM that can output an invention tailored to its current condition. The hope for the future is for such a conditional inventor agent to help seed an open-ended process, wherein interactions between agents and their inventions spur continual innovation. (right) In the Sodarace domain, the conditional inventor observes the terrain, which conditions the LLM to output the specification of the desired invention.

That is, while a Sodaracer can be constructed in Python by directly adding elements to a Python dictionary with keys such as “joints” and “muscles,” a more Pythonic interface (which was more effective and is what is used in the experiments) is to create a simple class with two methods: “add\_joint” (to add a spring) and “add\_muscle” (to add a point mass.) The idea is that such an interface (here encapsulated in a class called “walker\_creator”) is closer to the training distribution of Python code (although still no Sodarace examples in this format exist). For example, below is the encoding of a simple hand-designed square Sodaracer (that is also used in the experiments as a seed), as well as its translation after being executed into a dictionary of joints and muscles. The interface also includes logic for ensuring that the Sodaracer will not break the underlying Box2D physics engine, e.g. that each joint is connected only to so many muscles, that the strength of muscles is limited, and that there is a minimum distance between joints. Note that the benefit of evolving a program that produces a data structure rather than directly evolving the data structure itself relates to the benefits of indirect encoding (i.e. a program can leverage regularities through loops, conditionals, and functions, to efficiently encode large complex structures) [60]. Figure 6 shows an image of this walker when instantiated.---

```
1 from walk_creator import walker_creator
2
3 def make_square(wc, x0, y0, x1, y1):
4     """ Make a square with top left x0,y0 and top right x1,y1 """
5     j0 = wc.add_joint(x0, y0)
6     j1 = wc.add_joint(x0, y1)
7     j2 = wc.add_joint(x1, y1)
8     j3 = wc.add_joint(x1, y0)
9
10    return j0, j1, j2, j3
11
12
13 def make_walker():
14     wc = walker_creator()
15
16     # the main body is a square
17     sides = make_square(wc, 0, 0, 10, 10)
18     center = wc.add_joint(5, 5)
19
20     # connect the square with distance muscles
21     for k in range(len(sides)-1):
22         wc.add_muscle(sides[k], sides[k+1])
23     wc.add_muscle(sides[3], sides[0])
24
25     # one prong of the square is a distance muscle
26     wc.add_muscle(sides[3], center)
27
28     # the other prongs from the center of the square are active
29     wc.add_muscle(sides[0], center, False, 5.0, 0.0)
30     wc.add_muscle(sides[1], center, False, 10.0, 0.0)
31     wc.add_muscle(sides[2], center, False, 2.0, 0.0)
32
33     return wc.get_walker()
```

---

Listing 1: Example Sodaracer-generating program.

## 5 Pipeline Stage 1: Data Generation through ELM

Recall that the aim in this first stage is to generate a large variety of high-quality examples from a single example starter program through ELM. In this stage of the pipeline, the Sodarace environment is a simple flat terrain.

Recall that ELM in this experiment will evolve through MAP-Elites (Fig----

```

1 {
2     "joints": [(0, 0), (0, 10), (10, 10), (10, 0), (5, 5)],
3     "muscles": [
4         [0, 1, {"type": "distance"}],
5         [1, 2, {"type": "distance"}],
6         [2, 3, {"type": "distance"}],
7         [3, 0, {"type": "distance"}],
8         [3, 4, {"type": "distance"}],
9         [0, 4, {"type": "muscle", "amplitude": 2.12, "phase": 0.0}],
10        [1, 4, {"type": "muscle", "amplitude": 2.12, "phase": 0.0}],
11        [2, 4, {"type": "muscle", "amplitude": 2.12, "phase": 0.0}],
12    ],
13 }

```

---

Listing 2: Intermediate Sodaracer representation from running the above Python seed program.

Figure 6: **Instantiation of a hand-designed square Sodaracer.** A video of this walker can be viewed at <https://y2u.be/jeP8Nsulu48>

ure 3) [51]. The core of MAP-Elites is a uniformly-spaced grid of niches (called the *map*), that spans user-specified dimensions of solution diversity, called the *behavior characterization*. In experiments here, the behavior characterization consists of the height, width, and mass of Sodaracers, and the map is a  $12 \times 12 \times 12$  grid into which any behavioral characterization can be mapped. Upon initialization, a single hand-designed solution is evaluated and placed into the map. In each iteration thereafter, an inhabited niche is randomly chosen and the solution within that niche is perturbed by the diff model and evaluated. The new candidate solution is assigned its niche from its behavior characterization, and if that niche is unfilled or the new solution outperforms the niche’s current inhabitant, it becomes the champion of that niche; otherwise, the candidate is discarded. In this way, over iterations of search, the map gradually fills with increasingly diverse and high-quality solutions.

To address the need for seed solutions, four simple seeds were written that explore different architectural motifs: the Square seed, the Radial seed, and two CPPN-like seeds (CPPN stands for *compositional pattern-producing network* [61]); note that source code for these seeds is provided in Appendix B. The Square seed instantiates a polygon-like bias, by including a function that createsFigure 7: **The three seed solutions.** From top to bottom: CPPN seed, radial seed, and square seed. A video of these walkers is at <https://y2u.be/jeP8Nsulu48> (same video as for Figure 6).

a square composed of four masses from two coordinates, and code that calls that function and connects the masses together with a for-loop. The Radial seed instead implements a radial bias by replacing the square-generating function with a function that places a given number of masses in a circular shape. Finally, the CPPN-like seeds roughly implement the CPPN-based encoding used by previous work in Sodarace [12], i.e. it places masses and connects springs between them as a mathematical function of their coordinates. The CPPN-based seed’s code can be neatly divided into (1) implementing the core functionality of a CPPN, and (2) describing a particular instantiation of a CPPN, and thus enables exploring the consequences of letting core functionality of the encoding itself evolve. To this end, there are two CPPN seeds, one in which the CPPN encoding is fixed, called the CPPN-Fixed seed, and one where it is mutable, called the CPPN-Mutable seed. Note that these seed programs were not highly-tuned as the videos in Figure 7 highlight.

## 5.1 Experimental Details and Results

Three independent runs of ELM were conducted with each seed, running for 1,024,000 evaluations each (composed of 2,000 iterations of 512 diffis per iteration). A 300M parameter pretrained diff model [31] served as the perturbation operator in these experiments.

One metric of success for ELM is the number of niches filled, which represents the diversity of data generated by ELM, under the hypothesis that such diverse data will benefit later pipeline stages. Figure 8 shows that runs of ELM tend to discover a large proportion of niches, highlighting how the system can bootstrap from a single user-provided example to fill the space of desired possibilities. However, the speed of spreading through niches varies across seeds; in particular, introducing loops and/or function composition is required for the Square seedFigure 8: **Amount of niches filled across seeds.** The figure shows the percentage of all niches (1,728 in total) that are filled by the end of ELM runs across different seeds. Results are averaged across three independent runs for each seed. In general, nearly all seeds fill the map, although the Square seed proceeds more slowly than other seeds.

to spread into high-mass niches (e.g. to connect many squares together), which emerges slowly in some runs.

Beyond diversity, the quality of solutions is also important. A gross measure of quality is the maximum fitness discovered by runs, shown in Figure 9. A more nuanced metric that takes both quality and diversity into account is the QD score [50], calculated as the sum of the performance of all champions in the final map. This metric, shown averaged over runs in Figure 10, rewards both quality (having higher scores in each niche) and diversity (having discovered more niches), and thus serves as a succinct measure of ELM’s goal of accumulating diverse, high-quality solutions (and in later stages in the pipeline, of how well an LLM has modeled the distribution of solutions that ELM has uncovered). Attainment of QD differs across seeds; while the CPPN seed uncovers diversity most quickly, the Radial seed generates higher-quality solutions on average. The relationship between the seed and the products of search is complex and deserves further future study (see also Appendix D for further analysis of seed robustness).

Fine-tuning the diff model on accepted diffs from an initial series of runs greatly increased performance (Figure 11); while Sodarace-generating programs are out-of-distribution for the pretrained diff model (applying a Python encoding to this domain is a novel enterprise), fine-tuning effectively aligns the diff model with the domain, an interesting result. Figure 11c shows how the fine-tuned diff model produces a significantly higher percentage of diffs that are valid (i.e. able to be applied) and runnable (i.e. the patched program is executable). Because of their higher performance, the output of runs applying the fine-tuned diff model are the ones passed to later stages in the pipeline.

Note that further rounds of fine-tuning are possible (e.g. fine-tuning the diffFigure 9: **Maximum fitness across seeds.** The maximum performance attained on average by different seeds is shown. The results suggest that ELM’s capacity to find high-fitness solutions is somewhat robust to seed design.

model again from the improved products of the first round); however preliminary experiments showed diminishing returns. Future work could explore how to continually improve such models, such as by identifying and encouraging more impactful perturbations instead of including and weighting equally all accepted diff.

The seeds and fine-tuned diff model also qualitatively impact the kinds of solutions discovered by ELM. While the Radial seed performs well quantitatively (in terms of quality and diversity), it turns out that its products tend to exploit chaotic dynamics that seem overfit to the flat terrain (this hypothesis is tentatively validated in the Stage 3 experiments). The Square and CPPN seeds in contrast are more likely to output inventions that leverage more predictable dynamics. For these reasons, the Radial seed runs were not ultimately used in future stages.

A video selection of the highest-quality Sodaracers from these initial runs that showcases the considerable diversity uncovered can be viewed at <https://y2u.be/QNyNtvwA9FI>. An example of a lineage of Sodaracers progressing from the Square seed to a high-quality final Sodaracer can be seen at <https://y2u.be/M9pAJuX6dyM>. In short, ELM shows that by combining the an intelligent LLM-based mutation operator with a QD algorithm it is possible to generate hundreds of thousands of working training examples in a completely novel domain where no data was previously available.

## 6 Pipeline Stage 2: Language Model Training

The product of Stage 1 is a collection of programs, whereas Stage 3 RL requires an initial model that can output valid Sodaracer-generating programs. Thus, the second stage of the invention pipeline fine-tunes an LLM on the productsFigure 10: **Quality diversity score across seeds.** Shown is the average final QD score attained by runs initialized from different seeds. The conclusion is that fine-tuning the diff model has a significant impact on attained QD score, as does the choice of seed.

of ELM, which serves as the initialization for an RL-based conditional inventor. To do so first requires compiling the results of Stage 1 into a fine-tuning dataset.

While there are many ways to distill a dataset of programs from runs of ELM, a simple thresholded approach is adopted here (although see Appendix E for another simple approach that did not work in practice). The main idea is to append all reasonably-capable solutions for each niche.

In more detail, from each run all solutions ever admitted to the map are included, subject to meeting a minimal bar for performance. Some parts of the behavior space offer more stringent challenges (i.e. it is more difficult to locomote when required to be tall but not wide and to have low mass), and yet in some terrains encountered in Stage 3, those kinds of solutions may yet be most effective despite their low level of absolute performance. Thus, for each niche, the maximum performance across all runs is calculated, and the minimal bar for inclusion is set as a percentage of that per-niche score. With a higher percentage threshold, less data is included, but the quality of that data will be higher.

As noted in the previous section, solutions from the Radial seed were qualitatively chaotic. Furthermore, preliminary experiments suggest that such chaotic behavior significantly harms downstream Stage 3 performance. For these reasons Radial runs of ELM were excluded from the LLM datasets. Datasets for each of the remaining treatments were compiled from 9 runs from ELM with the fine-tuned diff model (3 runs for each of the Square, CPPN-Fixed, and CPPN-Mutable seeds). In total, the 50% cut-off threshold dataset consisted of 280K examples, and the 80% cut-off threshold dataset contained a subset of 95K of those examples.

A variety of pretrained code-generating models were then fine-tuned with these examples (using the standard LLM log-probability loss), leaving out 5%(a) Niches Reached

(b) QD Score

(c) Diff Quality

Figure 11: **The impact of fine-tuning the diff model on the performance of ELM.** For both the pretrained diff model and the fine-tuned one, shown are (a) the number of niches reached, (b) QD score of the produced map, and (c) percentage of valid/runnable diffs proposed. The experiments demonstrate that fine-tuning the diff model improves performance of the evolutionary process across all three metrics.Figure 12: **Test loss across model sizes.** The minimum test loss achieved by training runs on the 80% Percentage Threshold dataset across model sizes is shown. Model sizes above 85M may not better-fit the data, and random initialization hurts performance.

of the data to serve as a test set. Models ranging from 0.1M to 680M parameters were trained (architectural details for these models can be seen in Appendix C). Also, as a control to support the hypothesis that Sodarace models benefit from code-generation pretraining, a 300M model was also trained instead from a random initialization (signified with “RI” in charts that follow).

Minimum test-losses (i.e. loss on generated Sodaracers held-out from the fine-tuning dataset) of the 80% Percentage Threshold models are shown in Figure 12. The 50% Percentage Threshold models exhibit qualitatively similar results across model size (but as both thresholds represent different datasets, loss values are not directly comparable between them). The conclusions are that model sizes above 85M may not better fit the data, and that random initialization does hurt performance relative to fine-tuning from a model pretrained on code.

However, loss is not the whole story. The interesting question for Stage 2 is whether the LLMs trained from the data generated in Stage 1 can generate the same diversity and quality of data. Therefore, the QD score metric and number of niches discovered (both of which were also reported for Stage 1) are calculated for samples taken from trained LLMs. Because these metrics can be maximized by a model that memorizes the data, and because empirically QD score was more correlated with loss on the training set rather than the test set, the LLM checkpoint for each model is selected on the basis of lowest training loss. In particular, 1,024 samples are taken from each model, which are then evaluated and inserted into a new MAP-Elites map. For comparison, the same metrics are calculated using the Stage 1 dataset, by taking the same number of samples from it and evaluating them in the same way. These results are shown in Figure 13, highlighting that the model samples achieve a similar level of performance as dataset samples, suggesting that they have modeled the data well. Also, there is a slight but consistent QD benefit from models trained on the 80% cutoff dataset, reflecting the higher average QD of that dataset.(a) Number of Niches Filled

(b) QD Score

Figure 13: **Measuring the quality and diversity of model samples.** Two metrics evaluating samples from trained LLMs are shown (across model size and training dataset): (a) the percentage of niches discovered and (b) the QD score achieved. The 80% threshold dataset is on average less diverse but of higher quality than the 50% threshold dataset, and induces the same properties in models trained upon it. There is not a trend in increasing quality or diversity as model size increases beyond 85M, and random initialization hurts performance.Figure 14: **Generalization tests.** In this test, the model is asked to complete samples, taken from the first half of the dataset from the unseen runs. These unseen originals are shown in the videos at <https://y2u.be/8C2K5fk28HI>. From top to bottom: Wheel, from radial seed; Galloper, from square seed; Runner, from CPPN seed.

A natural further question is how well the model will do when taken out of distribution, i.e. how well has it really internalized the dynamics of Sodarace? That is, the training and test set for fine-tuning are taken from the same runs, and thus the model will likely have encountered all of the motifs in the test set, and so it may not be a representative test of how well the model will generalize in the future. A preliminary test in this spirit is to take the first half of the Python programs describing several inventions from unseen runs, and explore the capacity of different models to generate functional completions. Though the Radial seed usually produced chaotic Sodaracers, in one preliminary run of ELM with the Radial seed, a functional wheel was discovered. As noted previously data from this run (or any other radial runs) was not used to train the models in Stage 2, nor was it used to fine-tune the diff model in Stage 1; thus the ability to complete the wheel can serve as a proxy for generalization. Similarly, two other high-performing individuals were taken from other preliminary runs of the CPPN seed and the Square seed, to create a set of three out-of-distribution completion tests. See Figure 14 for visualizations of these walkers, including videos; source code for these generalization examples can be found in Appendix F). Note that further tests of generalization are documented in Appendix H.

For each of the three completion tasks, 1,024 completion samples are taken from each model and then evaluated in simulation. In contrast to the in-distribution metrics, in this generalization-focused test, performance was more correlated with the model’s test loss rather than training loss, and thus what checkpoint to evaluate for each model was selected on the basis of lowest testFigure 15: **Out of distribution completion performance.** Shown is the percentage of the original solutions’ performance that is attained by completions from trained LLMs. The percentage shown is the maximum attained over 1,024 independent completion samples from each model. The results are averaged over three out-of-distribution solutions (taken from runs not included in LLM training). The conclusion is that the 80% threshold models perform better than the 50% threshold, and that there is no obvious trend in performance once model size reaches 85M parameters.

loss. Results are shown in Figure 15, highlighting that larger models, and those trained on the 80% threshold, generally perform better at this task. Note that the randomly-initialized (RI) 300M model significantly underperforms, providing more evidence that pretraining on code provides a valuable prior.

Videos of the best-performing sample for the Wheel completion from each model are at <https://y2u.be/-LW2cCwSdRU> (for the 80% threshold dataset; the random-initialized 300M model is not shown because it generated no valid samples for this completion). For the Galloper and Runner completions, the structure and/or behavior of completions often does not match the original sample (especially for the Galloper). In the following linked video, a higher-performing completion is shown for both of the Galloper and the Runner: <https://y2u.be/XR3L4cZ83xU>.

Overall, these results show that an LLM can effectively integrate synthetic data generated through ELM in a novel domain.

## 7 Pipeline Stage 3: Conditional RL

In the final stage, reinforcement learning (RL) is invoked to fine-tune the pre-trained LLM output by Stage 2 of the pipeline. The goal is to produce a model that outputs Python programs representing Sodaracers in response to *particular terrains*. Importantly, the output of Stage 2 is an *unconditional* model, in the sense that it samples Sodaracers from a distribution defined by the output of Stage 1, without considering the terrain in which the samples will be deployed. The first step in Stage 3 is thus to convert the model to a *conditional* one, i.e. amodel that accepts terrains as inputs, and produces samples of Sodaracers in response.

To achieve this functional form, we first introduce the notion of a *terrain embedding network* (TEN). The role of the TEN is to map a representation of the terrain to a representation that can be used by the model to sample conditionally. In particular, the output of TENs is a vector (or sequence of vectors) in  $d$ , the dimension in which the model embeds tokens. That way, the output of the TEN can be treated as the activation from a given prefix, and the model can proceed in effect now sampling conditioned on the output of the TEN.

Concretely, an unconditional autoregressive LLM defines a sampling distribution over a sequence of tokens  $\mathbf{x} = (x_1, \dots, x_n)$  as  $p_{\theta}(\mathbf{x}) = \prod_{i=1}^n p_{\theta}(x_i | x_{<i})$ . In this stage, we introduce the additional module  $f_{\text{TEN}}$ , which represents terrains  $t$  in  $\mathbb{R}^d$ . As  $f_{\text{TEN}}(t) \in \mathbb{R}^d$ , we can consider the resulting conditional model without further modification:

$$p_{\theta}(\mathbf{x} | t) = \prod_{i=1}^n p_{\theta}(x_i | x_{<i}, f_{\text{TEN}}(t)). \quad (1)$$

This approach is similar to the controllable transformer proposed by Keskar et al. [65], but with the conditional codes being the output of a TEN, rather than particular tokens from the existing vocabulary.

Given a distribution over terrains  $p(t)$ , an RL setting is constructed to train the parameters of the TEN and further finetune the LLM parameters to the conditional setting. In particular, an episode now consists of sampling  $t \sim p(t)$ , and sampling a program from the conditional distribution defined in Equation (1). The program is converted to a Sodaracer, evaluated in simulation with the terrain  $t$ , and the reward is defined as the absolute distance traversed by the Sodaracer in a given period of time.

## 7.1 Terrain Distributions

In this experiment, the distribution over terrains that the model is exposed to is chosen to explore the viability of producing conditional inventors with the Invention Pipeline. The future vision is to lay the groundwork for the ability to deploy agents capable of conditional invention in rich, potentially multi-agent environments that support the development of open-ended processes. In such settings, it stands to reason that learning to output complex artifacts conditioned on observations of the environment would be a prerequisite to ongoing open-ended innovation.

However, in preliminary experiments in the Sodarace domain, learning tended to “gravitate” towards collapsed solutions, wherein a single program is produced that achieves reasonable performance on a subset of the terrains in the distribution support. To reduce the viability of such an outcome and simulate a scenario where conditionality is essential, a small and discrete set of terrains for which a single program *cannot* achieve good performance provides a test where conditional solutions should be significantly more advantageous.Figure 16: **Terrains used in experiments.** A small set of terrains from which distributions that force *conditionality* can be constructed. The terrains are (a) LEFT-WALL, (b) RIGHT-WALL, (c) BUMPY, and (d) TUNNEL. The Sodaracers produced by the models are incapable of scaling the walls in LEFT-WALL and RIGHT-WALL, and therefore must produce different Sodaracers for these two terrains. Similarly, achieving good performance in the TUNNEL terrain can only be achieved with short Sodaracers, which struggle to locomote as quickly as taller ones, encouraging the model to distinguish between these terrains. Finally, Sodaracers that are proficient in locomotion on flat terrains tend to perform poorly on BUMPY, encouraging the model to produce yet another Sodaracer for this terrain. In contrast to TUNNEL, which requires Sodaracers with a particular morphology, achieving good performance on BUMPY requires modifying the way Sodaracers *locomote*. Example Sodaracers are added to the figures to illustrate the scale of the terrains.

In the experiments, uniform distributions are considered over sets of terrains as illustrated in Figure 16. Two subsets are considered, both of which contain LEFT-WALL and RIGHT-WALL. One set additionally contains TUNNEL, and the other includes BUMPY. These sets were specifically chosen such that the models are incapable of producing a single Sodaracer that achieves good performance on all terrains; to maximize the learning objective, the model must leverage the TEN to incorporate conditionality.

## 7.2 Parametrizing TENs

Two parametrizations for the TEN are explored.

**Discrete Codes.** The terrain distribution has a discrete and finite support. As such, a simple parametrization wherein the terrains are treated as additional tokens in the existing vocabulary, and the embedding for each terrain is learned separately may be used. The advantage of such a parametrization is that itintroduces a relatively small number of new parameters to be optimized with RL, and it is conceptually simple to understand and debug. However, the main disadvantages of such a parameterization are that (i) the number of parameters scales with the size of the terrain set, and (ii) it does not allow the model to naturally generalize to unseen terrains at test-time, which may be an important constraint for downstream open-ended processes.

**ResNets.** An alternative parametrization is visual representations of the terrains, which can then be processed by visual recognition models. In particular, a ResNet50 [66] embeds images into  $\mathbb{R}^d$  as a TEN when experimenting with visual representations of terrains. The main advantages of this parametrization are that it is quite general, could conceivably be used in multiple settings (e.g. teaching a code-generating LLM to write programs in response to visual input, and in theory can generalize to unseen terrains. The main drawback of this approach is that it introduces a large number of new parameters that must be optimized using a sparse RL signal. Conversely, for large terrain distributions, this approach makes it possible to amortize the number of additional parameters necessary for designing conditional inventors.

### 7.3 Experimental Details and Results

Each RL episode consists of sampling a batch of terrains from the distribution, producing samples from the conditional LLM, and evaluating them in simulation to produce the reward.

Proximal policy optimization [PPO; 67] is the RL algorithm, in conjunction with generalized advantage estimation [GAE; 68], with default hyperparameters. In preliminary experiments, we found it important to add a KL term (between the policy network and the pre-trained LLM from Stage 2) *to the reward function*, as proposed by Christiano et al. [69] and Stiennon et al. [70]. The value network is parametrized as a scalar-function version of the policy network, i.e. a separate LLM with a separate prepended TEN initialized from the Stage 2 models. Figure 17 illustrates the architectures and pipelines for the policy and value-function networks. Each iteration consists of batches of 1,024 samples (distributed over 32 GPUs), and training runs consist of 100 iterations.

RL is run on pretrained, 300M-parameter LLMs trained with datasets having cutoff thresholds in  $\{50\%, 80\%\}$ . Recall that we use the cutoff threshold to control the tradeoff between data quality and quantity, such that higher thresholds result in smaller pretraining datasets with a higher density of quality instances. For each dataset and terrain distribution combination, three runs are performed using different seeds, and the performance is averaged over samples from the resulting model for each terrain, from over all runs, though we exclude a small number of runs that diverged during training. To compute a measure of the performance of datasets and pretrained LLMs, we invoke test-time compute: 1,024 Sodaracers are sampled uniformly and evaluated from each dataset/model (recall that there is one model for both cutoff thresholds), and the best-performingThe diagram illustrates the RL architecture, divided into two parts: (a) Policy network and (b) Value-function network.

**(a) Policy network:** This part shows the process of generating a policy. It starts with a terrain image (a green and pink rectangle) and a prompt (a text box with "one with creator more better creation"). The terrain image is processed by a TEN (Terrain Embedding Network) neural network. The prompt is processed by a Token Embedding layer. Both the TEN output and the Token Embedding output are fed into a policy network (represented by a neural network icon). The policy network outputs a sample of Python code (a code block). This code is then used to compile a Sodaracer, which is evaluated in a simulation (labeled "Sim") to produce a reward  $R$ .

**(b) Value-function network:** This part shows the process of estimating the value function. It starts with the same terrain image and prompt as the policy network. The terrain image is processed by a TEN neural network. The prompt is processed by a Token Embedding layer. Both the TEN output and the Token Embedding output are fed into a value-function network (represented by a neural network icon). The value-function network outputs an estimation of the value  $V$ .

Figure 17: **Illustration of the RL architecture.** The conditional policy (a) and value-function (b) are depicted, both augmented with a separate TEN for terrain embeddings. The policy is conditioned on a particular terrain (via the TEN) and prompt, and produces a sample, which is interpreted as Python code. The code is then used to compile a Sodaracer, which is evaluated in a simulation to produce a reward  $R$ . The value function is conditioned on the same terrain (via its own TEN) and prompt, and outputs an estimation of the value ( $V$ ) of every token output by the policy sample. During learning, the value-function is trained to predict advantages estimated using GAE [68].

