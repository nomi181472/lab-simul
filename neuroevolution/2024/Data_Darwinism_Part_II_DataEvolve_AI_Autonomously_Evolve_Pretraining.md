Title: AI can Autonomously Evolve Pretraining Data Curation

URL Source: https://arxiv.org/html/2603.14420

Markdown Content:
showstringspaces = false, keywords = false,true, alsoletter = 0123456789., morestring = [s]””, stringstyle = , MoreSelectCharTable =\lst@DefSaveDef‘:\colon@json\processColon@json, basicstyle = , keywordstyle = ,

††affiliationtext: 1 SII 2 FDU 3 SJTU 4 GAIR 5 KPS
Dongming Shan Zhen Huang Yiwei Qin Muhang Xie 

Yuxuan Qiao Yixiu Liu Chenyang Zhou Pengfei Liu†

###### Abstract

Data Darwinism (Part I) established a systematic ten-level hierarchy for data processing, demonstrating that ascending this hierarchy—from basic filtering to model-driven enrichment—unlocks latent data value. However, that work relied on manually designed strategies for a single category. Modern pretraining corpora comprise hundreds of heterogeneous categories spanning domains, content types, and quality levels—each demanding specialized treatment. At this scale, manual strategy design becomes prohibitive: experts must analyze each category, identify quality issues, and iteratively refine approaches. This raises a fundamental question: can strategies themselves evolve in an automated way?

We introduce DataEvolve, a framework that enables strategies to evolve through iterative optimization rather than manual design. For each data category, DataEvolve operates in a closed evolutionary loop: it observes data to identify quality issues, generates candidate strategies, executes them on sampled data, evaluates results with diagnostic feedback, and refines approaches across generations. This evolutionary process accumulates knowledge through an experience pool of discovered issues and a strategy pool tracking performance across iterations, enabling each generation to build upon previous insights. Applied to 8 categories spanning 672B tokens from Nemotron-CC, DataEvolve produces Darwin-CC, a 504B-token dataset with strategies evolved through 30 iterations per category. Training 3B models on 500B tokens, Darwin-CC substantially outperforms raw data (+3.96 points) and achieves 44.13 average score across 18 benchmarks—surpassing DCLM (42.42), Ultra-FineWeb (36.29), and FineWeb-Edu (36.52), with pronounced gains on knowledge-intensive tasks (MMLU: +18.64, MedQA: +13.48). Analysis reveals evolved strategies converge on cleaning-focused approaches: targeted noise removal and format normalization with domain-aware preservation—echoing the L4 (Generative Refinement) principles from Part I. Ablation studies confirm iterative evolution is essential: optimized strategies outperform suboptimal ones by 2.93 points, demonstrating that not all automated processing is equally effective. Our work establishes evolutionary strategy design as both feasible and necessary at pretraining scale, shifting the paradigm from manual expertise to automated evolution in data curation.

††footnotetext: † Corresponding author.![Image 1: Refer to caption](https://arxiv.org/html/2603.14420v1/figures/teaser-final.png)

Figure 1: Performance comparison across pretraining datasets on 18 selected benchmarks.

## 1 Introduction

The performance of large language models is fundamentally determined by their pretraining data (Longpre et al., [2023](https://arxiv.org/html/2603.14420#bib.bib112 "A pretrainer’s guide to training data: measuring the effects of data age, domain coverage, quality, & toxicity")). Data Darwinism (Part I) (Qin et al., [2026](https://arxiv.org/html/2603.14420#bib.bib2 "Data darwinism part i: unlocking the value of scientific data for pre-training"))established this principle through a systematic ten-level processing hierarchy (L0–L9), demonstrating that data quality is not static but evolves through progressively sophisticated transformations—from basic filtering (L0–L3) to model-driven refinement (L4–L5) and beyond. Part I validated this hierarchy on scientific literature, showing that raw data suffers from severe learnability gaps that only higher-level processing can bridge. Yet the strategies driving this data evolution remained manually designed for a single category, leaving a critical question unanswered: can the strategies themselves evolve?

Modern pretraining relies on web crawling to achieve trillion-token scale, but this introduces substantial noise. Meanwhile, high-quality curated sources such as textbooks and publications remain limited and grow slowly (Villalobos et al., [2024](https://arxiv.org/html/2603.14420#bib.bib111 "Will we run out of data? limits of llm scaling based on human-generated data")). Quality-based filtering methods like FineWeb-Edu (Penedo et al., [2024](https://arxiv.org/html/2603.14420#bib.bib67 "FineWeb: decanting the web for the finest text data at scale")) and DCLM (Li et al., [2024](https://arxiv.org/html/2603.14420#bib.bib12 "DataComp-lm: in search of the next generation of training sets for language models")) address this by retaining high-quality web pages while discarding low-quality content—an approach corresponding to the lower levels (L0–L3) of the Data Darwinism hierarchy. However, as Part I demonstrated, filtering-only approaches fail to unlock the full value of conceptually dense content. Aggressive filtering substantially reduces data quantity and risks losing valuable content obscured by surface-level noise, motivating data refinement: transforming noisy sources into usable material through higher-level processing (Nguyen et al., [2025b](https://arxiv.org/html/2603.14420#bib.bib110 "Recycling the web: a method to enhance pre-training data quality and quantity for language models")).

Recent work demonstrates that large language models can automate data refinement at the L4–L5 levels by executing specified strategies at scale. Approaches including rephrasing web text into structured formats (Mahabadi et al., [2025](https://arxiv.org/html/2603.14420#bib.bib37 "Nemotron-cc-math: a 133 billion-token-scale high quality math pretraining dataset"); Maini et al., [2024](https://arxiv.org/html/2603.14420#bib.bib78 "Rephrasing the web: a recipe for compute and data-efficient language modeling")) and enriching content through guided rewriting (Jiang et al., [2025](https://arxiv.org/html/2603.14420#bib.bib6 "Generative data refinement: just ask for better data")) establish LLM-based curation feasibility. Modern pretraining corpora span diverse categories with distinct characteristics, and recent work explores tailored strategies: SwallowCode and SwallowMath (Fujii et al., [2025](https://arxiv.org/html/2603.14420#bib.bib108 "Rewriting pre-training data boosts llm performance in math and code")) for code and mathematics, MegaMath (Zhou et al., [2025](https://arxiv.org/html/2603.14420#bib.bib33 "MegaMath: pushing the limits of open math corpora")) for mathematical web pages, and MegaScience (Fan et al., [2025](https://arxiv.org/html/2603.14420#bib.bib34 "MegaScience: pushing the frontiers of post-training datasets for science reasoning")) with discipline-specific extraction strategies.

While these tailored approaches demonstrate clear benefits, they reveal a fundamental scalability bottleneck. Designing effective strategies requires analyzing category-specific characteristics, identifying quality issues, formulating appropriate operations, and iteratively refining approaches—demanding substantial expertise per category. Even adding a single few-shot example to a curation strategy can be painstaking, as it may require sifting through dozens of candidate examples before identifying a single high-quality one that effectively guides the model (Jiang et al., [2022](https://arxiv.org/html/2603.14420#bib.bib3 "PromptMaker: prompt-based prototyping with large language models")). Moreover, even validating a single candidate strategy requires cleaning the data and training a model at scale, making iterative refinement practically infeasible (Li et al., [2024](https://arxiv.org/html/2603.14420#bib.bib12 "DataComp-lm: in search of the next generation of training sets for language models")). However, modern pretraining corpora comprise hundreds or thousands of categories formed by intersecting subject domains, content types, quality levels, and other dimensions, each exhibiting distinct characteristics (AI et al., [2025](https://arxiv.org/html/2603.14420#bib.bib24 "Essential-web v1.0: 24t tokens of organized web data"); Wettig et al., [2025](https://arxiv.org/html/2603.14420#bib.bib5 "Organize the web: constructing domains enhances pre-training data curation")). At this scale, comprehensive per-category manual design becomes prohibitive. This raises our central question: can strategies themselves evolve in an automated way? Recent advances show AI systems can autonomously design solutions—from optimizing prompts (Zhou et al., [2023](https://arxiv.org/html/2603.14420#bib.bib105 "Large language models are human-level prompt engineers")) to discovering architectures (Liu et al., [2025](https://arxiv.org/html/2603.14420#bib.bib7 "AlphaGo moment for model architecture discovery"))—suggesting that strategy design, like the data it processes, can undergo evolutionary refinement.

We introduce DataEvolve, a framework enabling strategies to evolve through iterative optimization rather than manual design. DataEvolve operates in a closed evolutionary loop: for each category, it observes data to identify quality issues, generates candidate strategies, executes them on sampled documents, and refines approaches based on diagnostic feedback. Rather than full model training, DataEvolve approximates strategy fitness through sample-based quality assessment, enabling rapid iterative refinement without prohibitive training overhead. Critically, the system evolves knowledge across generations through two mechanisms: an experience pool accumulating discovered quality issues, and a strategy pool tracking performance and analyses across iterations. Each generation builds upon previous insights, with successful strategies serving as parents for the next generation while unsuccessful ones are pruned—a natural selection process for data curation strategies. This transforms strategy design from a manual, expertise-intensive process into an evolutionary optimization that scales across hundreds of categories.

We apply DataEvolve to 8 categories spanning 672B tokens from Nemotron-CC (Mahabadi et al., [2025](https://arxiv.org/html/2603.14420#bib.bib37 "Nemotron-cc-math: a 133 billion-token-scale high quality math pretraining dataset")), evolving strategies through 30 generations per category to produce Darwin-CC, a 504B-token dataset. Training 3B models on 500B tokens, Darwin-CC substantially outperforms raw data (+3.96 points average across 18 benchmarks) and gains are particularly pronounced on knowledge-intensive tasks (MMLU +18.64, MedQA +13.48), and achieves 44.13 average score, surpassing established corpora including DCLM (42.42), Ultra-FineWeb (36.29), and FineWeb-Edu (36.52) under identical training budgets.

Ablation studies confirm that evolution is essential: optimized strategies outperform suboptimal ones by 2.93 points, demonstrating that not all automated processing is equally effective—only evolved strategies unlock full data value, and that our sample-based fitness approximation reliably reflects true downstream performance. Analysis reveals that evolved strategies converge on cleaning-focused approaches: targeted noise removal with domain-aware preservation, echoing the L4 (Generative Refinement) principles from Part I. This convergence demonstrates that systematic cleaning, when properly evolved for each category’s characteristics, suffices for substantial quality improvements without expensive content transformation.

Our contributions are:(1) We demonstrate that strategies can evolve: by formulating strategy design as an evolutionary optimization problem with diagnostic feedback and cross-generation knowledge accumulation, we enable automated discovery of effective curation strategies validated at pretraining scale. (2) We release DataEvolve, an end-to-end framework for evolutionary strategy design, and Darwin-CC, a 504B-token dataset where each category’s strategy has evolved through 30 generations, outperforming established corpora. (3) We reveal evolutionary convergence: independently evolved strategies across diverse categories converge on cleaning-focused approaches with domain-aware preservation, offering a simpler and more scalable path than transformation-based methods for pretraining data curation.

## 2 Problem Formulation

Figure 2: Naively evaluating candidate curation strategies requires cleaning data at full scale and training a model to convergence for each candidate strategy—demanding thousands of GPU hours per evaluation. Across n n candidate strategies and m m categories, the total cost becomes computationally intractable.

Modern pretraining corpora span hundreds of heterogeneous categories, each with distinct quality characteristics requiring tailored curation strategies. We formalize the problem of designing effective strategies at this scale.

Let 𝒞={c 1,c 2,…,c m}\mathcal{C}=\{c_{1},c_{2},\ldots,c_{m}\} denote a set of data categories, where each category c i c_{i} contains a corpus 𝒟 i={d 1,d 2,…,d n}\mathcal{D}_{i}=\{d_{1},d_{2},\ldots,d_{n}\} of raw documents. A curation strategy s i s_{i} for category c i c_{i} specifies processing operations—what to remove, preserve, and normalize—that an LLM-based executor f​(⋅,⋅)f(\cdot,\cdot) applies to produce cleaned data:

𝒟 i′={f​(d,s i)∣d∈𝒟 i}\mathcal{D}^{\prime}_{i}=\{f(d,s_{i})\mid d\in\mathcal{D}_{i}\}(1)

Ideally, strategy quality would be determined by downstream model performance. Let M 𝒟′M_{\mathcal{D}^{\prime}} denote a language model trained on 𝒟′=⋃i=1 m 𝒟 i′\mathcal{D}^{\prime}=\bigcup_{i=1}^{m}\mathcal{D}^{\prime}_{i} with performance Q​(M)Q(M) on evaluation benchmarks. The optimal strategies would satisfy:

(s 1∗,…,s m∗)=arg​max s 1,…,s m⁡Q​(M 𝒟′)(s^{*}_{1},\ldots,s^{*}_{m})=\operatorname*{arg\,max}_{s_{1},\ldots,s_{m}}Q(M_{\mathcal{D}^{\prime}})(2)

However, directly optimizing this objective is prohibitively expensive (Figure [2](https://arxiv.org/html/2603.14420#S2.F2 "Figure 2 ‣ 2 Problem Formulation ‣ AI can Autonomously Evolve Pretraining Data Curation")): each evaluation requires cleaning data, training a model from scratch, and measuring performance—demanding thousands of GPU hours. Moreover, quality differences between strategies often emerge only at scales of hundreds of billions of tokens during training. This cost is further compounded in the pretraining setting: unlike fine-tuning where quality differences emerge quickly, pretraining requires processing hundreds of billions of tokens before the effect of a curation strategy becomes statistically distinguishable, rendering exhaustive search over candidate strategies computationally intractable.

To make the problem tractable, we approximate the objective using a quality function S​(𝒟 i′)S(\mathcal{D}^{\prime}_{i}) that evaluates cleaned data through sample-based assessment (rating document quality on a 1-10 scale):

s i∗=arg​max s i⁡S​(𝒟 i′)s^{*}_{i}=\operatorname*{arg\,max}_{s_{i}}S(\mathcal{D}^{\prime}_{i})(3)

This reformulation enables iterative strategy refinement without model training at each step. However, the problem remains challenging: the strategy space 𝒮\mathcal{S} is discrete and combinatorially large, with strategies varying in operation choices, decision criteria, and quality issue coverage. Moreover, effective design requires analyzing category-specific characteristics, identifying quality issues, and iterative refinement—effort that compounds across hundreds of heterogeneous categories.

## 3 DataEvolve: Evolutionary Strategy Design

### 3.1 Overview

![Image 2: Refer to caption](https://arxiv.org/html/2603.14420v1/figures/new-pipeline.png)

Figure 3: Overview of the DataEvolve framework. The system enables strategies to evolve through an iterative feedback loop involving four core components: (1) the data observer identifies category-specific quality issues; (2) the strategy designer generates and refines cleaning strategies; (3) the data cleaner executes strategies on sample data; and (4) the quality judge provides scoring and diagnostic feedback. Discovered issues and evolved strategies are archived in the experience pool and strategy pool, enabling cross-generation knowledge transfer to guide evolutionary progression.

DataEvolve enables category-specific strategies to evolve through iterative optimization rather than manual design. The framework operates as an evolutionary system with four core components in a closed feedback loop: a data observer identifies quality issues in sampled data, a strategy designer generates and refines curation strategies, a data cleaner executes these strategies, and a quality judge evaluates results and provides diagnostic feedback. Two repositories enable cross-generation knowledge transfer: an experience pool accumulates discovered quality issues across all iterations, and a strategy pool tracks evolved strategies with their performance scores and diagnostic analyses.

The evolutionary process operates in two phases. During initialization, the system observes the target category to identify quality problems, establishing the initial knowledge base. Through evolutionary refinement, strategies evolve over multiple generations: each iteration selects the best-performing strategy as a parent, generates refined variants through mutation (guided by diagnostic feedback), evaluates offspring on new samples, and propagates successful strategies to the next generation. Each category evolves independently, enabling specialized strategies tailored to distinct data characteristics. The complete evolutionary workflow is illustrated in Figure [3](https://arxiv.org/html/2603.14420#S3.F3 "Figure 3 ‣ 3.1 Overview ‣ 3 DataEvolve: Evolutionary Strategy Design ‣ AI can Autonomously Evolve Pretraining Data Curation").

### 3.2 Initialization: Establishing the Knowledge Base

Before strategies can evolve, DataEvolve establishes an initial understanding of the target category. The data observer examines sampled documents and produces a structured assessment revealing category-specific quality patterns—such as equation rendering errors and broken citations in stem content, ambiguous medical terminology in medicine, or code snippets mixed with corrupted symbols in computer science.

These initial observations seed the experience pool—a repository of quality issues that guides all subsequent evolutionary cycles. Rather than each generation starting from scratch, this accumulated knowledge ensures that evolved strategies address both initially observed problems and issues discovered during execution, enabling genuine evolutionary progress rather than random search.

### 3.3 Evolutionary Strategy Optimization

With the knowledge base established, DataEvolve enters its core evolutionary loop: generate strategy variants, execute them on sample data, evaluate fitness, and select survivors for the next generation. This cycle repeats for multiple generations, with each generation building upon the insights of its predecessors.

#### Strategy generation.

The strategy designer operates differently across iterations. In the first iteration, it synthesizes observations from the experience pool into an initial strategy. From the second iteration onward, it refines the best-performing strategy from the strategy pool, guided by diagnostic feedback explaining what worked, what gaps remain, and what needs clarification.

#### Execution.

The data cleaner applies the designed strategy to a batch of sampled documents. To ensure generalization, different documents are sampled in each iteration, preventing strategies from overfitting to specific examples. This produces pairs of original and cleaned documents that reveal the strategy’s actual behavior.

#### Evaluation.

The quality judge evaluates results in three steps. First, it scores each (original, cleaned) pair on a 1-10 scale with comments on improvements and remaining issues; the average becomes the strategy’s overall score. Second, it analyzes strategy coverage (which issues from the experience pool were addressed) and executability (whether instructions were clear and consistently followed). Third, it often discovers new quality issues during evaluation, which are added to the experience pool for future iterations.

#### Closing the loop.

After each iteration, results are stored in both repositories: the strategy pool records the strategy with its score and diagnostic analysis, while the experience pool accumulates newly discovered issues. When the next iteration begins, the strategy designer accesses both an enriched experience pool (with more quality issues identified) and the best-performing strategy with its diagnostic feedback from the strategy pool. This feedback loop enables each iteration to produce more targeted and effective strategies than the previous one. After completing all iterations, the highest-scoring strategy from the strategy pool is selected for deployment on the full dataset.

## 4 Experiments

### 4.1 Dataset Construction

#### Data Source

We construct our dataset from Nemotron-CC (Mahabadi et al., [2025](https://arxiv.org/html/2603.14420#bib.bib37 "Nemotron-cc-math: a 133 billion-token-scale high quality math pretraining dataset")), a large-scale web corpus derived from Common Crawl. We use only the real (non-synthetic) part, which provides quality annotations at five levels. To simplify our experimental design, we merge these into two categories: high and not-high.

Beyond quality, we require finer-grained categorization to enable domain-specific prompt design. We employ the EAI-Distill-0.5B(AI et al., [2025](https://arxiv.org/html/2603.14420#bib.bib24 "Essential-web v1.0: 24t tokens of organized web data")), a specialized document classification model fine-tuned from Qwen2.5-0.5B-Instruct that provides annotations across 12 taxonomic dimensions, including Free Decimal Correspondence (FDC), Bloom’s Taxonomy, Document Type, Content Quality, and Educational Metadata. We focus on two of these dimensions for our categorization scheme.(1) For the FDC , which captures subject domain, we perform a mapping to derive five broad disciplines: human-social, mathematics, computer science, medicine, and other stem. (2) Similarly, for the Document Type, we map the original classifications into five content categories: academic, code, fragment, social media, and text. Details of these mappings are provided in Appendix [A](https://arxiv.org/html/2603.14420#A1 "Appendix A Classification ‣ AI can Autonomously Evolve Pretraining Data Curation").

For our experiments, we select a focused subset from this categorization space: we use only academic content across four technical disciplines (mathematics, computer science, medicine, and other stem), each at two quality levels (high, not-high). This selection yields 8 distinct categories comprising approximately 672B tokens in total.

Table 1: Category composition of Darwin-CC

Content Type Quality Domain Raw Cleaned
Academic High Mathematics 7B 5B
Computer Science 32B 24B
Medicine 52B 29B
Other STEM 71B 51B
Not-High Mathematics 6B 6B
Computer Science 107B 78B
Medicine 164B 132B
Other STEM 233B 179B
Total 672B 504B

#### DataEvolve Configuration

We apply DataEvolve independently to each of the 8 categories for 30 iterations. Each iteration operates as follows: the data observer, implemented with GPT-4o-mini(OpenAI, [2024](https://arxiv.org/html/2603.14420#bib.bib107 "GPT-4o mini: advancing cost-efficient intelligence")), analyzes 100 sampled documents to identify quality issues; the strategy designer, using o4-mini([26](https://arxiv.org/html/2603.14420#bib.bib61 "OpenAI o3 and o4-mini system card")), synthesizes these observations with accumulated feedback to generate a refined cleaning prompt; the data cleaner executes this prompt via gpt-oss-120b(OpenAI, [2025a](https://arxiv.org/html/2603.14420#bib.bib58 "Gpt-oss-120b & gpt-oss-20b model card")) on 500 documents; and the quality judge, also using gpt-5-mini(OpenAI, [2025b](https://arxiv.org/html/2603.14420#bib.bib109 "Introducing gpt-5")), evaluates 50 randomly sampled pairs from the cleaned results, producing both scores and diagnostic feedback.

#### Final Dataset.

After 30 iterations per category, we select highest-scoring strategy from each strategy pool for deployment. Each optimized strategy is then applied to its corresponding full category corpus using the same LLM-based data cleaner. This large-scale cleaning process, executed on the complete 672B-token raw data, produces Darwin-CC, comprising 504B tokens. The 25% token reduction reflects targeted removal of low-quality content—duplicates, formatting artifacts, incomplete fragments, and noise—while preserving domain-specific valuable content through the discovered preservation rules. Table [1](https://arxiv.org/html/2603.14420#S4.T1 "Table 1 ‣ Data Source ‣ 4.1 Dataset Construction ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation") shows the token distribution across the 8 categories before and after cleaning. We do not perform explicit benchmark decontamination, consistent with the practice of the upstream Nemotron-CC corpus and comparable datasets (Mahabadi et al., [2025](https://arxiv.org/html/2603.14420#bib.bib37 "Nemotron-cc-math: a 133 billion-token-scale high quality math pretraining dataset"); Li et al., [2024](https://arxiv.org/html/2603.14420#bib.bib12 "DataComp-lm: in search of the next generation of training sets for language models"); Lozhkov et al., [2024](https://arxiv.org/html/2603.14420#bib.bib11 "FineWeb-edu: the finest collection of educational content")). We note that our cleaning process targets removing artifacts, duplicates, and low-quality fragments, and thus is unlikely to systematically alter benchmark contamination levels relative to the source corpus.

### 4.2 Training Configuration

To ensure fair comparison, we train all models from scratch on 500B tokens using identical configurations. We employ a 3B-parameter Qwen2.5 architecture (Team, [2024](https://arxiv.org/html/2603.14420#bib.bib40 "Qwen2.5: a party of foundation models")) with 4,096-token context length. Training uses AdamW (Loshchilov and Hutter, [2019](https://arxiv.org/html/2603.14420#bib.bib85 "Decoupled weight decay regularization")) with a learning rate that warms up to 3×10−4 3\times 10^{-4} over 6,000 steps and remains constant thereafter, a global batch size of 1,024, and runs for 120,000 steps total, approximately 500B tokens.

### 4.3 Evaluation Benchmarks

We evaluate all trained models on a comprehensive benchmark suite, including MMLU (5-shot) (Hendrycks et al., [2020](https://arxiv.org/html/2603.14420#bib.bib43 "Measuring massive multitask language understanding")), ARC-Easy and ARC-Challenge (10-shot) (Clark et al., [2018](https://arxiv.org/html/2603.14420#bib.bib42 "Think you have solved question answering? try arc, the ai2 reasoning challenge")), OpenBookQA (5-shot) (Mihaylov et al., [2018](https://arxiv.org/html/2603.14420#bib.bib46 "Can a suit of armor conduct electricity? a new dataset for open book question answering")), PIQA (10-shot) (Bisk et al., [2020](https://arxiv.org/html/2603.14420#bib.bib47 "Piqa: reasoning about physical commonsense in natural language")), HellaSwag (10-shot) (Zellers et al., [2019](https://arxiv.org/html/2603.14420#bib.bib16 "HellaSwag: can a machine really finish your sentence?")), WinoGrande (0-shot) ([43](https://arxiv.org/html/2603.14420#bib.bib9 "WinoGrande: an adversarial winograd schema challenge at scale")), SIQA (5-shot) (Sap et al., [2019](https://arxiv.org/html/2603.14420#bib.bib4 "Social iqa: commonsense reasoning about social interactions")), RACE (0-shot) (Lai et al., [2017](https://arxiv.org/html/2603.14420#bib.bib15 "RACE: large-scale reading comprehension dataset from examinations")), BBH (3-shot) (Suzgun et al., [2022](https://arxiv.org/html/2603.14420#bib.bib41 "Challenging big-bench tasks and whether chain-of-thought can solve them")), DROP (5-shot) (Dua et al., [2019](https://arxiv.org/html/2603.14420#bib.bib45 "DROP: a reading comprehension benchmark requiring discrete reasoning over paragraphs")), AGIEval-En (0-shot) (Zhong et al., [2023](https://arxiv.org/html/2603.14420#bib.bib10 "AGIEval: a human-centric benchmark for evaluating foundation models")), CSQA(0-shot) (Talmor et al., [2019](https://arxiv.org/html/2603.14420#bib.bib8 "CommonsenseQA: a question answering challenge targeting commonsense knowledge")), TriviaQA (5-shot) (Joshi et al., [2017](https://arxiv.org/html/2603.14420#bib.bib17 "triviaqa: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension")), GPQA-Main (5-shot) (Rein et al., [2024](https://arxiv.org/html/2603.14420#bib.bib50 "Gpqa: a graduate-level google-proof q&a benchmark")), MedQA (0-shot) (Jin et al., [2021](https://arxiv.org/html/2603.14420#bib.bib54 "What disease does this patient have? a large-scale open domain question answering dataset from medical exams")), MedMCQA (0-shot) (Pal et al., [2022](https://arxiv.org/html/2603.14420#bib.bib56 "Medmcqa: a large-scale multi-subject multi-choice dataset for medical domain question answering")), and PubMedQA (0-shot) (Jin et al., [2019](https://arxiv.org/html/2603.14420#bib.bib57 "Pubmedqa: a dataset for biomedical research question answering")), totally 18 benchmarks. All evaluations are conducted using the lm-evaluation-harness(Gao et al., [2024](https://arxiv.org/html/2603.14420#bib.bib14 "The language model evaluation harness")) framework.

## 5 Results

### 5.1 Validating Automated Strategy Design

Table 2: Performance comparison across curation strategies. Models trained on raw data (Raw), data with suboptimal strategy (Sub-opt), data cleaned with DataEvolve’s best strategy (Best), which generates Darwin-CC. All models are 3B parameters trained on 500B tokens.

Metrics Raw Sub-opt Best
BBH 26.82 26.69-0.12 26.16-0.65
ARC-E 74.94 77.55+2.61 78.59+3.65
ARC-C 43.52 48.41+4.90 49.32+5.80
MMLU 27.49 32.55+5.05 46.13+18.64
AGIEval 18.15 18.30+0.15 18.21+0.07
HellaSwag 65.32 64.36-0.96 62.21-3.11
TriviaQA 25.33 26.96+1.63 26.65+1.32
RACE 35.04 35.63+0.59 34.28-0.76
DROP 19.57 19.48-0.09 18.49-1.08
WinoGrande 57.96 59.89+1.93 58.09+0.13
PIQA 76.79 76.80+0.01 76.15-0.64
CSQA 20.31 20.61-0.30 39.12+18.80
SIQA 44.36 43.58-0.78 43.57-0.79
OpenBookQA 39.80 42.20+2.40 41.44+1.64
GPQA-Main 24.37 23.93-0.45 27.10+2.72
MedQA 26.77 26.11-0.66 40.25+13.48
MedMCQA 28.86 30.28+1.42 40.97+12.10
PubMedQA 67.68 68.32+0.64 67.68+0.00
Average 40.17 41.20+1.03 44.13+3.96

To validate DataEvolve’s effectiveness, we compare three data configurations under identical training setups (3B model, 500B tokens): raw unprocessed data, data cleaned with a lower-scoring strategy from the strategy pool, and Darwin-CC cleaned with DataEvolve’s best-performing strategies. Table [2](https://arxiv.org/html/2603.14420#S5.T2 "Table 2 ‣ 5.1 Validating Automated Strategy Design ‣ 5 Results ‣ AI can Autonomously Evolve Pretraining Data Curation") presents results across 18 benchmarks, and Figure [5](https://arxiv.org/html/2603.14420#S5.F5 "Figure 5 ‣ Automated design improves data quality. ‣ 5.1 Validating Automated Strategy Design ‣ 5 Results ‣ AI can Autonomously Evolve Pretraining Data Curation") shows the learning curves throughout training.

#### Automated design improves data quality.

Models trained on Darwin-CC achieve an average score of 44.13, substantially outperforming raw data by 3.96 points. This consistent improvement demonstrates that automated strategy design effectively enhances pretraining data quality. Gains are particularly pronounced on knowledge-intensive tasks: MMLU improves by 18.64 points, CSQA by 18.80 points, and MedMCQA by 12.11 points, suggesting that DataEvolve successfully preserves factual information while removing noise. Beyond final performance, Figure [5](https://arxiv.org/html/2603.14420#S5.F5 "Figure 5 ‣ Automated design improves data quality. ‣ 5.1 Validating Automated Strategy Design ‣ 5 Results ‣ AI can Autonomously Evolve Pretraining Data Curation") reveals superior data efficiency throughout training. Darwin-CC achieves steeper early improvements and maintains widening advantages as training progresses—the gap over raw data grows from 2.49 points at 250B tokens to 3.96 points at 500B tokens. This scaling behavior demonstrates that data quality differences compound during training, with optimized strategies enabling more effective learning at scale.

![Image 3: Refer to caption](https://arxiv.org/html/2603.14420v1/figures/performance_curve_left.png)

Figure 4: Learning curves of 3B models trained on raw data, suboptimal strategy, and optimized strategy (Darwin-CC) over 500B tokens.

![Image 4: Refer to caption](https://arxiv.org/html/2603.14420v1/figures/performance_curve-new-right.png)

Figure 5: Learning curves comparison across pretraining corpora. All models are 3B parameters trained for 500B tokens.

#### Iterative optimization is essential.

Comparison with the lower-scoring strategy reveals that not all automated cleaning is equally effective. Data cleaned with the suboptimal strategy achieves only 41.20 average score, barely above raw data (+1.03 points) and significantly below Darwin-CC (gap: 2.93 points). This pattern is also especially stark on knowledge tasks: on MMLU, Darwin-CC gains +18.64 over raw data while the suboptimal strategy gains only +5.05, leaving a 13.59-point gap attributable to strategy quality alone. These disparities confirm that iterative optimization, not just LLM-based processing, drives effective data curation.

### 5.2 Comparison with Established Pretraining Corpora

To assess the practical value of Darwin-CC, we compare it against three widely-used pretraining datasets: FineWeb-Edu (Lozhkov et al., [2024](https://arxiv.org/html/2603.14420#bib.bib11 "FineWeb-edu: the finest collection of educational content")), a high-quality educational content subset; Ultra-FineWeb (Wang et al., [2025](https://arxiv.org/html/2603.14420#bib.bib13 "Ultra-FineWeb: efficient data filtering and verification for high-quality llm training data")), an aggressively filtered version emphasizing quality over scale; and DCLM (Li et al., [2024](https://arxiv.org/html/2603.14420#bib.bib12 "DataComp-lm: in search of the next generation of training sets for language models")), a large-scale curated corpus using classifier-based filtering. All models are trained under identical configurations (3B parameters, 500B tokens) and evaluated on 18 benchmarks. Table [3](https://arxiv.org/html/2603.14420#S5.T3 "Table 3 ‣ 5.2 Comparison with Established Pretraining Corpora ‣ 5 Results ‣ AI can Autonomously Evolve Pretraining Data Curation") presents the results.

Darwin-CC achieves the highest average score at 44.13, outperforming the second-best baseline DCLM by 1.71 points. This improvement is particularly pronounced on knowledge-intensive tasks such as MMLU. Performance is generally competitive or superior across most benchmarks, though a few tasks such as HellaSwag and TriviaQA show variability reflecting different curation emphases. Overall, these results demonstrate that Darwin-CC achieves quality competitive with or superior to established pretraining datasets, validating both DataEvolve’s automated strategy design approach and the practical value of the resulting corpus.

Beyond final performance, Figure [5](https://arxiv.org/html/2603.14420#S5.F5 "Figure 5 ‣ Automated design improves data quality. ‣ 5.1 Validating Automated Strategy Design ‣ 5 Results ‣ AI can Autonomously Evolve Pretraining Data Curation") show that Darwin-CC achieves steeper improvements in early training stages and maintains consistent gains as training progresses, suggesting that automated strategy design produces not only higher-quality data but also more sample-efficient pretraining corpora.

Table 3: Comparison of models trained on different pretraining corpora. All models are 3B parameters trained on 500B tokens.

Metrics Fineweb-Edu Ultra-Fineweb DCLM Darwin-CC
BBH 3.01 7.42 24.16 26.16
ARC-E 73.39 73.96 75.13 78.59
ARC-C 43.45 43.77 45.02 49.32
MMLU 28.38 25.53 28.54 46.13
AGIEval 16.96 17.72 17.90 18.21
HellaSwag 64.33 65.32 70.39 62.21
TriviaQA 0.67 0.42 42.85 26.65
RACE 35.43 34.28 36.08 34.28
DROP 6.78 7.78 24.31 18.49
WinoGrande 61.02 60.93 64.99 58.09
PIQA 75.80 75.63 77.93 76.15
CSQA 19.54 19.90 20.16 39.12
SIQA 45.02 43.43 47.51 43.57
OpenBookQA 39.92 39.84 43.36 41.44
GPQA-Main 24.51 23.04 25.67 27.10
MedQA 26.36 24.84 24.88 40.25
MedMCQA 25.80 24.92 28.15 40.97
PubMedQA 67.04 64.44 66.56 67.68
Average 36.52 36.29 42.42 44.13

## 6 Analysis

### 6.1 What Do DataEvolve Strategies Actually Do?

Examining the category-specific strategies reveals a clear pattern: DataEvolve learns to clean, not to transform. Contrary to prior approaches that rephrase web text into idealized formats (textbook-style exposition, Wikipedia articles, or question-answer pairs), discovered strategies focus on removing unwanted content and normalizing formatting while preserving original text.

The cleaning operations fall into three categories: (1) removing web artifacts (HTML tags, navigation menus, advertisements, duplicate sentences, PII); (2) normalizing formatting (whitespace, punctuation, numeric expressions); and (3) applying domain-specific preservation rules. These preservation rules reveal category-specific tailoring: stem strategies preserve scientific nomenclature and equations with LaTeX handling; mathematics strategies maintain theorem statements and proofs; medicine strategies preserve clinical units and drug names; computer science strategies protect code blocks and technical syntax.

This cleaning-focused approach yields substantial improvements by removing artifacts and normalizing formatting errors while preserving original content. The score gain demonstrates that systematic cleaning enhances quality without rewriting text into homogeneous formats such as textbook-style exposition or question-answer pairs. Unlike transformation-based approaches that impose uniform writing styles, cleaning preserves the diverse content and structure of web sources while removing technical corruption. This offers a practical path for scaling data curation to heterogeneous pretraining corpora.

Table 4: Diversity metrics for raw and cleaned datasets.

Metric Raw Cleaned Change
Self-ROUGE-2 ↓0.0148 0.0116-21.7%
L2 Distance ↑1.376 1.374-0.15%
Shannon Entropy ↑10.84 10.87+0.3%

### 6.2 What Makes an Effective Curation Strategy?

Comparing high-scoring and low-scoring strategies from each category’s strategy pool reveals consistent patterns. Effective strategies share four key characteristics: (1) concrete criteria—specifying measurable thresholds (e.g., “over 80% identical” for deduplication) rather than vague judgments like “clearly truncated”; (2) targeted deletion—removing specific elements (HTML tags, PII, promotional content) rather than wholesale document filtering; (3) explicit preservation rules—protecting domain-critical content alongside deletion instructions, such as “Preserve theorem statements and proofs” in mathematics or “Preserve clinical units and drug names” in medicine, preventing over-aggressive filtering; and (4) conservative operations—preferring deletion of uncertain fragments over risky transformations that may introduce errors.

### 6.3 Task-Specific Performance

To understand where cleaning-focused strategies provide the most benefit, we analyze performance across all 18 benchmarks and identify three distinct patterns, reflecting how different task types respond to data cleaning.

#### Substantial Improvement (Cleaned >> Raw).

The most pronounced gains appear on benchmarks centered on knowledge memorization and recall, including MMLU (57 academic subjects across science, law and medicine), CSQA (everyday conceptual knowledge), ARC-Challenge (challenging science questions), OpenBookQA (science facts combined with world knowledge), and medical benchmarks MedQA and MedMCQA. This finding is consistent with concurrent work on guided web rewriting (Nguyen et al., [2025a](https://arxiv.org/html/2603.14420#bib.bib94 "Recycling the web: a method to enhance pre-training data quality and quantity for language models"); Mahabadi et al., [2025](https://arxiv.org/html/2603.14420#bib.bib37 "Nemotron-cc-math: a 133 billion-token-scale high quality math pretraining dataset")), which also observes large MMLU gains when synthetic or cleaned data enriches the factual content available during pretraining. This pattern points to a clear underlying mechanism: raw web documents contain rich factual content obscured by formatting artifacts, HTML noise, and boilerplate text. By removing such surface-level corruption while preserving knowledge-bearing content, DataEvolve allows models to more effectively memorize and retain factual knowledge during pretraining. As shown in Figure [6](https://arxiv.org/html/2603.14420#S6.F6 "Figure 6 ‣ Substantial Improvement (Cleaned > Raw). ‣ 6.3 Task-Specific Performance ‣ 6 Analysis ‣ AI can Autonomously Evolve Pretraining Data Curation"), the advantage of cleaned data emerges early and continues to widen throughout training.

![Image 5: Refer to caption](https://arxiv.org/html/2603.14420v1/figures/benchmark_subfig/clean_higher/ARC-Challenge.png)

![Image 6: Refer to caption](https://arxiv.org/html/2603.14420v1/figures/benchmark_subfig/clean_higher/CSQA.png)

![Image 7: Refer to caption](https://arxiv.org/html/2603.14420v1/figures/benchmark_subfig/clean_higher/MMLU.png)

![Image 8: Refer to caption](https://arxiv.org/html/2603.14420v1/figures/benchmark_subfig/clean_higher/OpenBookQA.png)

Figure 6: Representative benchmarks where cleaned data substantially outperforms raw data throughout training. The gap between cleaned (blue) and raw (red) widens consistently as training progresses.

#### Performance Degradation (Cleaned << Raw).

A different pattern emerges on benchmarks evaluating informal and situated language understanding, including HellaSwag (plausible continuation of everyday activity descriptions), SIQA (social commonsense reasoning), PIQA (physical commonsense through everyday procedures), and DROP (numerical reasoning over diverse paragraphs). As shown in Figure [7](https://arxiv.org/html/2603.14420#S6.F7 "Figure 7 ‣ Performance Degradation (Cleaned < Raw). ‣ 6.3 Task-Specific Performance ‣ 6 Analysis ‣ AI can Autonomously Evolve Pretraining Data Curation"), the performance gap appears from the very beginning of training, suggesting a fundamental difference in what the two data distributions teach the model rather than a matter of convergence speed. We hypothesize that cleaning removes colloquial and conversational expressions from the training corpus, shifting the language distribution toward more formal and structured text. As a result, the model becomes less familiar with the informal registers these benchmarks rely on, producing outputs that are stylistically misaligned with how these tasks naturally present language.

![Image 9: Refer to caption](https://arxiv.org/html/2603.14420v1/figures/benchmark_subfig/raw_higher/DROP.png)

![Image 10: Refer to caption](https://arxiv.org/html/2603.14420v1/figures/benchmark_subfig/raw_higher/HellaSwag.png)

![Image 11: Refer to caption](https://arxiv.org/html/2603.14420v1/figures/benchmark_subfig/raw_higher/PIQA.png)

![Image 12: Refer to caption](https://arxiv.org/html/2603.14420v1/figures/benchmark_subfig/raw_higher/SIQA.png)

Figure 7: Representative benchmarks where raw data outperforms cleaned data. The gap between raw (red) and cleaned (blue) is established early and remains stable throughout training.

#### Minimal Change (Cleaned ≈\approx Raw).

Beyond these two groups, a number of benchmarks show negligible differences between cleaned and raw data, with the two training curves interleaving throughout without a consistent directional gap—representative examples include BBH (multi-step logical and algorithmic reasoning), WinoGrande (commonsense reasoning through pronoun resolution), AGIEval (general cognitive abilities from standardized exams), and RACE (reading comprehension grounded in provided passages). Unlike the clear separation seen in the previous two groups, neither data configuration holds a sustained advantage here, suggesting these capabilities are largely robust to the surface-level changes DataEvolve introduces. One possible explanation is that these tasks depend less on the specific textual properties that cleaning modifies.

![Image 13: Refer to caption](https://arxiv.org/html/2603.14420v1/figures/benchmark_subfig/sim/AGIEval.png)

![Image 14: Refer to caption](https://arxiv.org/html/2603.14420v1/figures/benchmark_subfig/sim/BBH.png)

![Image 15: Refer to caption](https://arxiv.org/html/2603.14420v1/figures/benchmark_subfig/sim/RACE.png)

![Image 16: Refer to caption](https://arxiv.org/html/2603.14420v1/figures/benchmark_subfig/sim/WinoGrande.png)

Figure 8: Representative benchmarks where cleaned and raw data yield comparable performance, with the two curves interleaving throughout training without a consistent directional gap.

Taken together, these three groups suggest that the impact of data cleaning is not uniform across tasks, and may be mediated by how much a benchmark relies on the specific textual properties that cleaning modifies—such as knowledge density, language register, and surface-level formatting.

### 6.4 Semantic Preservation and Data Diversity

![Image 17: Refer to caption](https://arxiv.org/html/2603.14420v1/figures/similarity_distribution.png)

Figure 9: Semantic similarity distribution before and after processing. DataEvolve’s cleaning strategies (Raw1-Cleaned) preserve substantially more semantic content than transformation-based methods (Raw1-Wiki, Raw1-QA), with similarity distributions approaching the identity baseline. The tight, right-skewed distribution indicates minimal content modification during cleaning.

Semantic preservation. To validate DataEvolve’s design principle, we analyze how processing affects document semantics. We sample 1,500 documents and compare semantic similarity before and after processing using four methods: DataEvolve’s cleaning strategies, Wikipedia-style rephrasing, question-answer generation, and a random baseline pairing unrelated documents. Figure [9](https://arxiv.org/html/2603.14420#S6.F9 "Figure 9 ‣ 6.4 Semantic Preservation and Data Diversity ‣ 6 Analysis ‣ AI can Autonomously Evolve Pretraining Data Curation") reveals that DataEvolve-cleaned documents maintain substantially higher semantic similarity to their originals (mean: 0.958) compared to transformation-based methods (Wiki: 0.937, QA: 0.882) and far above the random baseline (0.619). The narrow, right-skewed distribution confirms that cleaning operations focus on targeted noise removal rather than content rewriting. This contrasts sharply with methods like question-answer generation, whose broader, left-shifted distribution indicates more aggressive content transformation.

Corpus diversity. While individual documents are minimally modified, does cleaning affect overall corpus diversity? Table [4](https://arxiv.org/html/2603.14420#S6.T4 "Table 4 ‣ 6.1 What Do DataEvolve Strategies Actually Do? ‣ 6 Analysis ‣ AI can Autonomously Evolve Pretraining Data Curation") shows that Self-ROUGE-2 (measuring pairwise n-gram overlap) decreases by 21.7%, indicating documents become more distinct after cleaning—likely due to removal of repeated boilerplate and near-duplicates. Critically, semantic diversity is preserved: mean L2 distance in embedding space remains nearly unchanged (1.376 → 1.374, -0.15%), and Shannon entropy slightly increases (+0.3%), suggesting marginally more balanced vocabulary distributions after noise removal.

Together, these results confirm DataEvolve’s dual objective: high per-document semantic similarity (Figure [9](https://arxiv.org/html/2603.14420#S6.F9 "Figure 9 ‣ 6.4 Semantic Preservation and Data Diversity ‣ 6 Analysis ‣ AI can Autonomously Evolve Pretraining Data Curation")) demonstrates that cleaning preserves original content, while stable corpus-level diversity metrics (Table [4](https://arxiv.org/html/2603.14420#S6.T4 "Table 4 ‣ 6.1 What Do DataEvolve Strategies Actually Do? ‣ 6 Analysis ‣ AI can Autonomously Evolve Pretraining Data Curation")) show that this preservation maintains the broad coverage essential for pretraining.

## 7 Related Work

The performance of Large Language Models (LLMs) is fundamentally tied to the quality and scale of their pre-training data. Early efforts focused on curating massive web crawls through heuristic rules and simple classifiers, resulting in datasets like RefinedWeb (Penedo et al., [2023](https://arxiv.org/html/2603.14420#bib.bib65 "The refinedweb dataset for falcon llm: outperforming curated corpora with web data, and web data only")) and Dolma (Soldaini et al., [2024](https://arxiv.org/html/2603.14420#bib.bib66 "Dolma: an open corpus of three trillion tokens for language model pretraining research")). Recent advancements have shifted toward more rigorous quality filtering and semantic benchmarking. For instance, DCLM (Li et al., [2024](https://arxiv.org/html/2603.14420#bib.bib12 "DataComp-lm: in search of the next generation of training sets for language models")) formalizes data curation as a standardized competition, while FineWeb-Edu (Penedo et al., [2024](https://arxiv.org/html/2603.14420#bib.bib67 "FineWeb: decanting the web for the finest text data at scale")) and Ultra-FineWeb (Wang et al., [2025](https://arxiv.org/html/2603.14420#bib.bib13 "Ultra-FineWeb: efficient data filtering and verification for high-quality llm training data")) utilize model-based scoring to extract high-educational-value content. However, these corpora primarily rely on filtering—either retaining or removing documents—which often leads to the loss of valuable domain-specific information that is merely “noisy” in its raw form.

Beyond simple filtering, recent studies explore using LLMs as active editors to refine training data. Nemotron-CC (Mahabadi et al., [2025](https://arxiv.org/html/2603.14420#bib.bib37 "Nemotron-cc-math: a 133 billion-token-scale high quality math pretraining dataset")) and WRAP (Maini et al., [2024](https://arxiv.org/html/2603.14420#bib.bib78 "Rephrasing the web: a recipe for compute and data-efficient language modeling")) demonstrate that rephrasing web text into structured formats (e.g., Wikipedia-style or QA) can significantly improve signal density. More fine-grained approaches like ProX (Zhou et al., [2024](https://arxiv.org/html/2603.14420#bib.bib76 "Programming every example: lifting pre-training data quality like experts at scale")) and REFINEX (Bi et al., [2025](https://arxiv.org/html/2603.14420#bib.bib77 "Refinex: learning to refine pre-training data at scale from expert-guided programs")) treat data curation as “programming every example,” executing expert-guided edit programs to perform precise corpus surgery. Similarly, Recycling the Web (Nguyen et al., [2025a](https://arxiv.org/html/2603.14420#bib.bib94 "Recycling the web: a method to enhance pre-training data quality and quantity for language models")) attempts to recover low-quality data through synthetic rephrasing, and GDR (Jiang et al., [2025](https://arxiv.org/html/2603.14420#bib.bib6 "Generative data refinement: just ask for better data")) leverages pretrained models to automatically transform unsafe or private raw content into high-quality training data. While effective, these methods typically require labor-intensive manual design of prompts or programs for each data category, creating a bottleneck when scaling to hundreds of heterogeneous domains.

## 8 Conclusion

This paper presents DataEvolve, a framework that transforms pretraining data curation from a manual, expertise-intensive process into an automated evolutionary system. By operating a closed loop of data observation, strategy generation, execution, and diagnostic feedback—with knowledge accumulated across generations through an experience pool and a strategy pool—DataEvolve autonomously discovers high-quality, category-specific cleaning strategies at scale. Applied to 8 categories spanning 672B tokens from Nemotron-CC, the resulting Darwin-CC dataset achieves 44.13 average score across 18 benchmarks, outperforming raw data by 3.96 points and surpassing established corpora including DCLM, Ultra-FineWeb, and FineWeb-Edu, with pronounced gains on knowledge-intensive tasks. Ablation studies further confirm that iterative evolution is essential, demonstrating that only systematically evolved strategies unlock the full latent value of data.

Analysis reveals that strategies evolved independently across diverse categories consistently converge on a cleaning-centric paradigm—targeted noise removal and format normalization with domain-aware content preservation—rather than rewriting text into homogeneous formats. This convergence suggests that systematic cleaning, when guided by properly evolved strategies, suffices for substantial quality improvements without expensive content transformation, offering a simpler and more scalable path for large-scale pretraining data curation. Looking ahead, DataEvolve establishes evolutionary strategy design as a promising new direction, with future work to explore broader content types, improved stopping criteria, and optimization objectives more directly tied to downstream task performance.

## 9 Limitations and Future Directions

Despite promising results, DataEvolve has several limitations that point to natural directions for future work. First, due to computational constraints, Darwin-CC covers only 8 categories from Nemotron-CC, all of which are academic content in scientific domains. This also means that when comparing Darwin-CC against other general-purpose pretraining corpora, the comparison may not be entirely fair, as those datasets cover a much broader range of topics. We note that our primary contribution is the processed-vs-raw comparison, which directly validates DataEvolve’s effectiveness; the cross-dataset comparison is intended to demonstrate Darwin-CC’s value as an open-source resource rather than to claim general superiority. Extending DataEvolve to broader content types remains an important direction for future work. Second, each category is evolved for a fixed 30 iterations without an adaptive stopping criterion, which may not be sufficient for full convergence; more iterations and principled stopping criteria would likely yield further gains. Finally, a more fundamental challenge lies in our fitness approximation: while sample-based quality scoring enables efficient strategy evolution, it introduces noise relative to true downstream performance. Incorporating lightweight model training as a more direct feedback signal may be a natural next step, though our experiments reveal a subtle difficulty—quality differences between strategies often only emerge after training on hundreds of billions of tokens, and performance rankings can even reverse at earlier stages of training. This suggests that small-scale training signals may themselves be unreliable proxies, and designing an efficient yet accurate evaluation mechanism remains an open challenge for the community.

## References

*   E. AI, :, A. Hojel, M. Pust, T. Romanski, Y. Vanjani, R. Kapila, M. Parmar, A. Chaluvaraju, A. Tripathy, A. Thomas, A. Tanwer, D. J. Shah, I. Shah, K. Stratos, K. Nguyen, K. Smith, M. Callahan, P. Rushton, P. Monk, P. Mazarakis, S. Jamal, S. Srivastava, S. Singla, and A. Vaswani (2025)Essential-web v1.0: 24t tokens of organized web data. External Links: 2506.14111, [Link](https://arxiv.org/abs/2506.14111)Cited by: [§1](https://arxiv.org/html/2603.14420#S1.p4.1 "1 Introduction ‣ AI can Autonomously Evolve Pretraining Data Curation"), [§4.1](https://arxiv.org/html/2603.14420#S4.SS1.SSS0.Px1.p2.1 "Data Source ‣ 4.1 Dataset Construction ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   B. Bi, S. Liu, X. Ren, D. Liu, J. Lin, Y. Wang, L. Mei, J. Fang, J. Guo, and X. Cheng (2025)Refinex: learning to refine pre-training data at scale from expert-guided programs. arXiv preprint arXiv:2507.03253. Cited by: [§7](https://arxiv.org/html/2603.14420#S7.p2.1 "7 Related Work ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   Y. Bisk, R. Zellers, J. Gao, Y. Choi, et al. (2020)Piqa: reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, Vol. 34,  pp.7432–7439. Cited by: [§4.3](https://arxiv.org/html/2603.14420#S4.SS3.p1.1 "4.3 Evaluation Benchmarks ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord (2018)Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457. Cited by: [§4.3](https://arxiv.org/html/2603.14420#S4.SS3.p1.1 "4.3 Evaluation Benchmarks ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   D. Dua, Y. Wang, P. Dasigi, G. Stanovsky, S. Singh, and M. Gardner (2019)DROP: a reading comprehension benchmark requiring discrete reasoning over paragraphs. arXiv preprint arXiv:1903.00161. Cited by: [§4.3](https://arxiv.org/html/2603.14420#S4.SS3.p1.1 "4.3 Evaluation Benchmarks ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   R. Fan, Z. Wang, and P. Liu (2025)MegaScience: pushing the frontiers of post-training datasets for science reasoning. External Links: 2507.16812, [Link](https://arxiv.org/abs/2507.16812)Cited by: [§1](https://arxiv.org/html/2603.14420#S1.p3.1 "1 Introduction ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   K. Fujii, Y. Tajima, S. Mizuki, H. Shimada, T. Shiotani, K. Saito, M. Ohi, M. Kawamura, T. Nakamura, T. Okamoto, S. Ishida, K. Hattori, Y. Ma, H. Takamura, R. Yokota, and N. Okazaki (2025)Rewriting pre-training data boosts llm performance in math and code. External Links: 2505.02881, [Link](https://arxiv.org/abs/2505.02881)Cited by: [§1](https://arxiv.org/html/2603.14420#S1.p3.1 "1 Introduction ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   L. Gao, J. Tow, B. Abbasi, S. Biderman, S. Black, A. DiPofi, C. Foster, L. Golding, J. Hsu, A. Le Noac’h, H. Li, K. McDonell, N. Muennighoff, C. Ociepa, J. Phang, L. Reynolds, H. Schoelkopf, A. Skowron, L. Sutawika, E. Tang, A. Thite, B. Wang, K. Wang, and A. Zou (2024)The language model evaluation harness. Zenodo. External Links: [Document](https://dx.doi.org/10.5281/zenodo.12608602), [Link](https://zenodo.org/records/12608602)Cited by: [§4.3](https://arxiv.org/html/2603.14420#S4.SS3.p1.1 "4.3 Evaluation Benchmarks ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt (2020)Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300. Cited by: [§4.3](https://arxiv.org/html/2603.14420#S4.SS3.p1.1 "4.3 Evaluation Benchmarks ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   E. Jiang, K. Olson, E. Toh, A. Molina, A. Donsbach, M. Terry, and C. J. Cai (2022)PromptMaker: prompt-based prototyping with large language models. CHI Conference on Human Factors in Computing Systems Extended Abstracts. External Links: [Link](https://api.semanticscholar.org/CorpusID:248419856)Cited by: [§1](https://arxiv.org/html/2603.14420#S1.p4.1 "1 Introduction ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   M. Jiang, J. G. M. Araújo, W. Ellsworth, S. Gooding, and E. Grefenstette (2025)Generative data refinement: just ask for better data. External Links: 2509.08653, [Link](https://arxiv.org/abs/2509.08653)Cited by: [§1](https://arxiv.org/html/2603.14420#S1.p3.1 "1 Introduction ‣ AI can Autonomously Evolve Pretraining Data Curation"), [§7](https://arxiv.org/html/2603.14420#S7.p2.1 "7 Related Work ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   D. Jin, E. Pan, N. Oufattole, W. Weng, H. Fang, and P. Szolovits (2021)What disease does this patient have? a large-scale open domain question answering dataset from medical exams. Applied Sciences 11 (14),  pp.6421. Cited by: [§4.3](https://arxiv.org/html/2603.14420#S4.SS3.p1.1 "4.3 Evaluation Benchmarks ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   Q. Jin, B. Dhingra, Z. Liu, W. W. Cohen, and X. Lu (2019)Pubmedqa: a dataset for biomedical research question answering. arXiv preprint arXiv:1909.06146. Cited by: [§4.3](https://arxiv.org/html/2603.14420#S4.SS3.p1.1 "4.3 Evaluation Benchmarks ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   M. Joshi, E. Choi, D. Weld, and L. Zettlemoyer (2017)triviaqa: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. arXiv e-prints,  pp.arXiv:1705.03551. External Links: 1705.03551 Cited by: [§4.3](https://arxiv.org/html/2603.14420#S4.SS3.p1.1 "4.3 Evaluation Benchmarks ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   G. Lai, Q. Xie, H. Liu, Y. Yang, and E. Hovy (2017)RACE: large-scale reading comprehension dataset from examinations. External Links: 1704.04683, [Link](https://arxiv.org/abs/1704.04683)Cited by: [§4.3](https://arxiv.org/html/2603.14420#S4.SS3.p1.1 "4.3 Evaluation Benchmarks ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   J. Li, A. Fang, G. Smyrnis, M. Ivgi, M. Jordan, S. Gadre, H. Bansal, E. Guha, S. Keh, K. Arora, S. Garg, R. Xin, N. Muennighoff, R. Heckel, J. Mercat, M. Chen, S. Gururangan, M. Wortsman, A. Albalak, Y. Bitton, M. Nezhurina, A. Abbas, C. Hsieh, D. Ghosh, J. Gardner, M. Kilian, H. Zhang, R. Shao, S. Pratt, S. Sanyal, G. Ilharco, G. Daras, K. Marathe, A. Gokaslan, J. Zhang, K. Chandu, T. Nguyen, I. Vasiljevic, S. Kakade, S. Song, S. Sanghavi, F. Faghri, S. Oh, L. Zettlemoyer, K. Lo, A. El-Nouby, H. Pouransari, A. Toshev, S. Wang, D. Groeneveld, L. Soldaini, P. W. Koh, J. Jitsev, T. Kollar, A. G. Dimakis, Y. Carmon, A. Dave, L. Schmidt, and V. Shankar (2024)DataComp-lm: in search of the next generation of training sets for language models. arXiv preprint arXiv:2406.11794. Cited by: [§1](https://arxiv.org/html/2603.14420#S1.p2.1 "1 Introduction ‣ AI can Autonomously Evolve Pretraining Data Curation"), [§1](https://arxiv.org/html/2603.14420#S1.p4.1 "1 Introduction ‣ AI can Autonomously Evolve Pretraining Data Curation"), [§4.1](https://arxiv.org/html/2603.14420#S4.SS1.SSS0.Px3.p1.1 "Final Dataset. ‣ 4.1 Dataset Construction ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"), [§5.2](https://arxiv.org/html/2603.14420#S5.SS2.p1.1 "5.2 Comparison with Established Pretraining Corpora ‣ 5 Results ‣ AI can Autonomously Evolve Pretraining Data Curation"), [§7](https://arxiv.org/html/2603.14420#S7.p1.1 "7 Related Work ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   Y. Liu, Y. Nan, W. Xu, X. Hu, L. Ye, Z. Qin, and P. Liu (2025)AlphaGo moment for model architecture discovery. External Links: 2507.18074, [Link](https://arxiv.org/abs/2507.18074)Cited by: [§1](https://arxiv.org/html/2603.14420#S1.p4.1 "1 Introduction ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   S. Longpre, G. Yauney, E. Reif, K. Lee, A. Roberts, B. Zoph, D. Zhou, J. Wei, K. Robinson, D. Mimno, and D. Ippolito (2023)A pretrainer’s guide to training data: measuring the effects of data age, domain coverage, quality, & toxicity. External Links: 2305.13169, [Link](https://arxiv.org/abs/2305.13169)Cited by: [§1](https://arxiv.org/html/2603.14420#S1.p1.1 "1 Introduction ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   I. Loshchilov and F. Hutter (2019)Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR), Cited by: [§4.2](https://arxiv.org/html/2603.14420#S4.SS2.p1.1 "4.2 Training Configuration ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   A. Lozhkov, L. Ben Allal, L. von Werra, and T. Wolf (2024)FineWeb-edu: the finest collection of educational content. Hugging Face. External Links: [Link](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), [Document](https://dx.doi.org/10.57967/hf/2497)Cited by: [§4.1](https://arxiv.org/html/2603.14420#S4.SS1.SSS0.Px3.p1.1 "Final Dataset. ‣ 4.1 Dataset Construction ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"), [§5.2](https://arxiv.org/html/2603.14420#S5.SS2.p1.1 "5.2 Comparison with Established Pretraining Corpora ‣ 5 Results ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   R. K. Mahabadi, S. Satheesh, S. Prabhumoye, M. Patwary, M. Shoeybi, and B. Catanzaro (2025)Nemotron-cc-math: a 133 billion-token-scale high quality math pretraining dataset. External Links: [Link](https://arxiv.org/abs/2508.15096)Cited by: [§1](https://arxiv.org/html/2603.14420#S1.p3.1 "1 Introduction ‣ AI can Autonomously Evolve Pretraining Data Curation"), [§1](https://arxiv.org/html/2603.14420#S1.p6.1 "1 Introduction ‣ AI can Autonomously Evolve Pretraining Data Curation"), [§4.1](https://arxiv.org/html/2603.14420#S4.SS1.SSS0.Px1.p1.1 "Data Source ‣ 4.1 Dataset Construction ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"), [§4.1](https://arxiv.org/html/2603.14420#S4.SS1.SSS0.Px3.p1.1 "Final Dataset. ‣ 4.1 Dataset Construction ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"), [§6.3](https://arxiv.org/html/2603.14420#S6.SS3.SSS0.Px1.p1.1 "Substantial Improvement (Cleaned > Raw). ‣ 6.3 Task-Specific Performance ‣ 6 Analysis ‣ AI can Autonomously Evolve Pretraining Data Curation"), [§7](https://arxiv.org/html/2603.14420#S7.p2.1 "7 Related Work ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   P. Maini, S. Seto, H. Bai, D. Grangier, Y. Zhang, and N. Jaitly (2024)Rephrasing the web: a recipe for compute and data-efficient language modeling. arXiv preprint arXiv:2401.16380. Cited by: [§1](https://arxiv.org/html/2603.14420#S1.p3.1 "1 Introduction ‣ AI can Autonomously Evolve Pretraining Data Curation"), [§7](https://arxiv.org/html/2603.14420#S7.p2.1 "7 Related Work ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   T. Mihaylov, P. Clark, T. Khot, and A. Sabharwal (2018)Can a suit of armor conduct electricity? a new dataset for open book question answering. In EMNLP, Cited by: [§4.3](https://arxiv.org/html/2603.14420#S4.SS3.p1.1 "4.3 Evaluation Benchmarks ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   T. Nguyen, Y. Li, O. Golovneva, L. Zettlemoyer, S. Oh, L. Schmidt, and X. Li (2025a)Recycling the web: a method to enhance pre-training data quality and quantity for language models. arXiv preprint arXiv:2506.04689. Cited by: [§6.3](https://arxiv.org/html/2603.14420#S6.SS3.SSS0.Px1.p1.1 "Substantial Improvement (Cleaned > Raw). ‣ 6.3 Task-Specific Performance ‣ 6 Analysis ‣ AI can Autonomously Evolve Pretraining Data Curation"), [§7](https://arxiv.org/html/2603.14420#S7.p2.1 "7 Related Work ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   T. Nguyen, Y. Li, O. Golovneva, L. Zettlemoyer, S. Oh, L. Schmidt, and X. Li (2025b)Recycling the web: a method to enhance pre-training data quality and quantity for language models. External Links: 2506.04689, [Link](https://arxiv.org/abs/2506.04689)Cited by: [§1](https://arxiv.org/html/2603.14420#S1.p2.1 "1 Introduction ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   [26] (2025)OpenAI o3 and o4-mini system card. External Links: [Link](https://cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4-mini-system-card.pdf)Cited by: [§4.1](https://arxiv.org/html/2603.14420#S4.SS1.SSS0.Px2.p1.1 "DataEvolve Configuration ‣ 4.1 Dataset Construction ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   OpenAI (2024)GPT-4o mini: advancing cost-efficient intelligence. Note: [https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/)Accessed: 2024-07-18 Cited by: [§4.1](https://arxiv.org/html/2603.14420#S4.SS1.SSS0.Px2.p1.1 "DataEvolve Configuration ‣ 4.1 Dataset Construction ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   OpenAI (2025a)Gpt-oss-120b & gpt-oss-20b model card. External Links: 2508.10925, [Link](https://arxiv.org/abs/2508.10925)Cited by: [§4.1](https://arxiv.org/html/2603.14420#S4.SS1.SSS0.Px2.p1.1 "DataEvolve Configuration ‣ 4.1 Dataset Construction ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   OpenAI (2025b)Introducing gpt-5. Note: [https://openai.com/index/introducing-gpt-5/](https://openai.com/index/introducing-gpt-5/)Accessed: 2025-08-07 Cited by: [§4.1](https://arxiv.org/html/2603.14420#S4.SS1.SSS0.Px2.p1.1 "DataEvolve Configuration ‣ 4.1 Dataset Construction ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   A. Pal, L. K. Umapathi, and M. Sankarasubbu (2022)Medmcqa: a large-scale multi-subject multi-choice dataset for medical domain question answering. In Conference on health, inference, and learning,  pp.248–260. Cited by: [§4.3](https://arxiv.org/html/2603.14420#S4.SS3.p1.1 "4.3 Evaluation Benchmarks ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   G. Penedo, H. Kydlícek, L. B. Allal, and T. Wolf (2024)FineWeb: decanting the web for the finest text data at scale. HuggingFace. Accessed: Jul 12. Cited by: [§1](https://arxiv.org/html/2603.14420#S1.p2.1 "1 Introduction ‣ AI can Autonomously Evolve Pretraining Data Curation"), [§7](https://arxiv.org/html/2603.14420#S7.p1.1 "7 Related Work ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   G. Penedo, Q. Malartic, D. Hesslow, R. Cojocaru, A. Cappelli, H. Alobeidli, B. Pannier, E. Almazrouei, and J. Launay (2023)The refinedweb dataset for falcon llm: outperforming curated corpora with web data, and web data only. arXiv preprint arXiv:2306.01116. Cited by: [§7](https://arxiv.org/html/2603.14420#S7.p1.1 "7 Related Work ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   Y. Qin, Z. Huang, T. Mi, W. Si, C. Zhou, Q. Guo, S. Feng, and P. Liu (2026)Data darwinism part i: unlocking the value of scientific data for pre-training. External Links: 2602.07824, [Link](https://arxiv.org/abs/2602.07824)Cited by: [§1](https://arxiv.org/html/2603.14420#S1.p1.1 "1 Introduction ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   D. Rein, B. L. Hou, A. C. Stickland, J. Petty, R. Y. Pang, J. Dirani, J. Michael, and S. R. Bowman (2024)Gpqa: a graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, Cited by: [§4.3](https://arxiv.org/html/2603.14420#S4.SS3.p1.1 "4.3 Evaluation Benchmarks ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   M. Sap, H. Rashkin, D. Chen, R. LeBras, and Y. Choi (2019)Social iqa: commonsense reasoning about social interactions. In EMNLP, External Links: [Link](https://www.aclweb.org/anthology/D19-1454)Cited by: [§4.3](https://arxiv.org/html/2603.14420#S4.SS3.p1.1 "4.3 Evaluation Benchmarks ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   L. Soldaini, R. Kinney, A. Bhagia, D. Schwenk, D. Atkinson, R. Authur, B. Bogin, K. Chandu, J. Dumas, Y. Elazar, et al. (2024)Dolma: an open corpus of three trillion tokens for language model pretraining research. arXiv preprint arXiv:2402.00159. Cited by: [§7](https://arxiv.org/html/2603.14420#S7.p1.1 "7 Related Work ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   M. Suzgun, N. Scales, N. Schärli, S. Gehrmann, Y. Tay, H. W. Chung, A. Chowdhery, Q. V. Le, E. H. Chi, D. Zhou, et al. (2022)Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261. Cited by: [§4.3](https://arxiv.org/html/2603.14420#S4.SS3.p1.1 "4.3 Evaluation Benchmarks ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   A. Talmor, J. Herzig, N. Lourie, and J. Berant (2019)CommonsenseQA: a question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), Minneapolis, Minnesota,  pp.4149–4158. External Links: [Link](https://aclanthology.org/N19-1421), [Document](https://dx.doi.org/10.18653/v1/N19-1421), 1811.00937 Cited by: [§4.3](https://arxiv.org/html/2603.14420#S4.SS3.p1.1 "4.3 Evaluation Benchmarks ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   Q. Team (2024)Qwen2.5: a party of foundation models. External Links: [Link](https://qwenlm.github.io/blog/qwen2.5/)Cited by: [§4.2](https://arxiv.org/html/2603.14420#S4.SS2.p1.1 "4.2 Training Configuration ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   P. Villalobos, A. Ho, J. Sevilla, T. Besiroglu, L. Heim, and M. Hobbhahn (2024)Will we run out of data? limits of llm scaling based on human-generated data. External Links: 2211.04325, [Link](https://arxiv.org/abs/2211.04325)Cited by: [§1](https://arxiv.org/html/2603.14420#S1.p2.1 "1 Introduction ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   Y. Wang, Z. Fu, J. Cai, P. Tang, H. Lyu, Y. Fang, Z. Zheng, J. Zhou, G. Zeng, C. Xiao, X. Han, and Z. Liu (2025)Ultra-FineWeb: efficient data filtering and verification for high-quality llm training data. External Links: 2505.05427 Cited by: [§5.2](https://arxiv.org/html/2603.14420#S5.SS2.p1.1 "5.2 Comparison with Established Pretraining Corpora ‣ 5 Results ‣ AI can Autonomously Evolve Pretraining Data Curation"), [§7](https://arxiv.org/html/2603.14420#S7.p1.1 "7 Related Work ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   A. Wettig, K. Lo, S. Min, H. Hajishirzi, D. Chen, and L. Soldaini (2025)Organize the web: constructing domains enhances pre-training data curation. External Links: 2502.10341, [Link](https://arxiv.org/abs/2502.10341)Cited by: [§1](https://arxiv.org/html/2603.14420#S1.p4.1 "1 Introduction ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   [43] (2019)WinoGrande: an adversarial winograd schema challenge at scale. Cited by: [§4.3](https://arxiv.org/html/2603.14420#S4.SS3.p1.1 "4.3 Evaluation Benchmarks ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi (2019)HellaSwag: can a machine really finish your sentence?. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, Cited by: [§4.3](https://arxiv.org/html/2603.14420#S4.SS3.p1.1 "4.3 Evaluation Benchmarks ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   W. Zhong, R. Cui, Y. Guo, Y. Liang, S. Lu, Y. Wang, A. Saied, W. Chen, and N. Duan (2023)AGIEval: a human-centric benchmark for evaluating foundation models. External Links: 2304.06364 Cited by: [§4.3](https://arxiv.org/html/2603.14420#S4.SS3.p1.1 "4.3 Evaluation Benchmarks ‣ 4 Experiments ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   F. Zhou, Z. Wang, Q. Liu, J. Li, and P. Liu (2024)Programming every example: lifting pre-training data quality like experts at scale. arXiv preprint arXiv:2409.17115. Cited by: [§7](https://arxiv.org/html/2603.14420#S7.p2.1 "7 Related Work ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   F. Zhou, Z. Wang, N. Ranjan, Z. Cheng, L. Tang, G. He, Z. Liu, and E. P. Xing (2025)MegaMath: pushing the limits of open math corpora. arXiv preprint arXiv:2504.02807. Note: Preprint Cited by: [§1](https://arxiv.org/html/2603.14420#S1.p3.1 "1 Introduction ‣ AI can Autonomously Evolve Pretraining Data Curation"). 
*   Y. Zhou, A. I. Muresanu, Z. Han, K. Paster, S. Pitis, H. Chan, and J. Ba (2023)Large language models are human-level prompt engineers. External Links: 2211.01910, [Link](https://arxiv.org/abs/2211.01910)Cited by: [§1](https://arxiv.org/html/2603.14420#S1.p4.1 "1 Introduction ‣ AI can Autonomously Evolve Pretraining Data Curation"). 

## Appendix A Classification

The Dewey Decimal Classification (DDC) is a widely adopted library classification system that systematically organizes knowledge through decimal numerical codes. We merged and remapped the numerical codes from FDC labels to align them with disciplines suitable for current research needs.

computer science 000-009 computer_science
mathematics 500-519 mathematics
medicine 610-619 medicine
stem-others 355-359 military_science
520-529 natural_sciences_astronomy
530-539 physics
540-549 chemistry
550-559 natural_sciences_earth
560-569 natural_sciences_paleontology
570-579 biology
580-589 natural_sciences_botany
590-599 natural_sciences_zoology
600-610, 620-621, 626, 629 engineering
622 engineering_mining
623 engineering_maritime
624 engineering_civil
625 engineering_railway
627 engineering_water
628 engineering_environment
630-631, 632-635, 636-639 agriculture
660-669 engineering_chemical
670-689 manufacturing
690-699 construction
910-919 natural_sciences_geography
humansocial 010-099, 350-354, 640-649, 650-659 management
100-129, 140-149, 160-199 philosophy
130-139, 150-159 psychology
200-299 religion
300-319, 360-369, 380-399 sociology
320-329 political_science
330-339 economics
340-349 law
370-379 education
400-499 linguistics
700-709, 750-769 art_fine_arts
710-729 art_architecture
730-739 art_artifacts
740-749 art_design
770-779 art_photography
780-789 art_music
790-799 art_sports
800-899 literature
900-909, 920-999 history

