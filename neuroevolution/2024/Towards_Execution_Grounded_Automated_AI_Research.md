Title: Towards Execution-Grounded Automated AI Research

URL Source: https://arxiv.org/html/2601.14525

Published Time: Thu, 22 Jan 2026 01:09:53 GMT

Markdown Content:
Zitong Yang Yejin Choi Emmanuel Candès Diyi Yang Tatsunori Hashimoto

###### Abstract

Automated AI research holds great potential to accelerate scientific discovery. However, current LLMs often generate plausible-looking but ineffective ideas. Execution grounding may help, but it is unclear whether automated execution is feasible and whether LLMs can learn from the execution feedback. To investigate these, we first build an automated executor to implement ideas and launch large-scale parallel GPU experiments to verify their effectiveness. We then convert two realistic research problems – LLM pre-training and post-training – into execution environments and demonstrate that our automated executor can implement a large fraction of the ideas sampled from frontier LLMs. We analyze two methods to learn from the execution feedback: evolutionary search and reinforcement learning. Execution-guided evolutionary search is sample-efficient: it finds a method that significantly outperforms the GRPO baseline (69.4% vs 48.0%) on post-training, and finds a pre-training recipe that outperforms the nanoGPT baseline (19.7 minutes vs 35.9 minutes) on pre-training, all within just ten search epochs. Frontier LLMs often generate meaningful algorithmic ideas during search, but they tend to saturate early and only occasionally exhibit scaling trends. Reinforcement learning from execution reward, on the other hand, suffers from mode collapse. It successfully improves the average reward of the ideator model but not the upper-bound, due to models converging on simple ideas. We thoroughly analyze the executed ideas and training dynamics to facilitate future efforts towards execution-grounded automated AI research.

Machine Learning, ICML

1 Introduction
--------------

![Image 1: Refer to caption](https://arxiv.org/html/2601.14525v1/x1.png)

Figure 1: We build an automated idea executor involving Implementer, Scheduler, and Worker. We then use this automated executor as a reward function to teach LLMs to generate more effective ideas through evolutionary search and RL. We only update the ideator in the learning process.

We envision automated AI research: LLMs generate research ideas to tackle important research problems, implement the ideas as code, run experiments to verify the effectiveness, and continuously learn from the execution results. If successful, these automated AI researchers could automatically develop and identify effective research ideas in a massive search space, thereby scalably converting compute into scientific discovery; the discovered ideas could, in turn, improve frontier AI models themselves, enabling recursive self-improvement. Despite the promise, automated AI research is bottlenecked by the ability of LLMs to generate effective ideas. Si et al. ([2025b](https://arxiv.org/html/2601.14525v1#bib.bib35)) and Si et al. ([2025a](https://arxiv.org/html/2601.14525v1#bib.bib34)) evaluated the quality of LLM-generated research ideas through large-scale expert review and found that LLM ideas often look convincing but are ineffective after being executed by human researchers.

This highlights the need to ground idea generation in execution. However, obtaining execution results of ideas in an automated and scalable manner is challenging, especially since we are targeting open-ended AI research where any ideas expressible in natural language are within our action space. To tackle this, we design and build a high-throughput automated idea executor that can implement hundreds of model-generated ideas and execute them in parallel to obtain the experiment results as execution feedback.

To study the extent to which we can automate realistic LLM research, we chose two GPU-intensive research problems (LLM pre-training and post-training) that are critical for improving the capabilities of LLMs as the research environments for our automated AI researchers. For the first time, we demonstrate that our automated executor can implement a large fraction of LLM-generated ideas on these challenging open-ended research problems, with over 90% execution rates on the pre-training environment with Claude-4.5-Sonnet and Claude-4.5-Opus.

To analyze whether grounding on execution-feedback can improve LLM idea generation, we define objective performance metrics for both environments and analyze the strengths and weaknesses of two popular learning algorithms: evolutionary search and reinforcement learning.

We use our automated executor to guide evolutionary search. Within ten search epochs, this execution-guided search finds a post-training recipe that outperforms the GRPO baseline (69.4% vs 48.0%) on the task of post-training a 1.5B model for math reasoning, and a pre-training recipe that outperforms the nanoGPT baseline (19.7 minutes vs 35.9 minutes) on the task of minimizing the training wall-clock time to reach the target validation loss (Table[1](https://arxiv.org/html/2601.14525v1#S1.T1 "Table 1 ‣ 1 Introduction ‣ Towards Execution-Grounded Automated AI Research")). Our analysis shows that models are often generating algorithmic ideas apart from tuning hyper-parameters, and evolutionary search significantly outperforms best-of-N under the same sampling budget. However, when analyzing the scaling trend, only Claude-4.5-Opus shows a clear scaling curve, while both Claude-4.5-Sonnet and GPT-5 tend to saturate early.

We then use the automated executor as the reward function in an RL loop to finetune Qwen3-30B. We show that RL with execution reward can successfully improve the average reward of the ideator model, similar to typical RL from verifiable rewards. However, RL does not improve the max reward, which is the more important metric for scientific discovery. In fact, we reveal that RL causes the ideator model to converge on a few easy-to-implement ideas, resulting in a collapse in thinking length and idea diversity.

In summary, we develop a large-scale automated idea executor system that can implement research ideas for open-ended and realistic research problems. Using this automated executor, we conduct an in-depth analysis of how well LLM ideators can learn from execution feedback to improve effectiveness through evolutionary search and RL. Execution-guided evolutionary search is sample-efficient and effective, but shows limited scaling. RL from execution reward suffers from diversity collapse and does not improve the upper-bound. We additionally provide extensive analysis on the executed ideas and suggest promising directions to improve the existing learning algorithms. Altogether, we demonstrate the feasibility and potential of grounding LLM ideation in automated execution and uncover important limitations for future improvement.

Table 1: Performance of our execution-guided search in comparison with the provided baselines and best human experts. The post-training task is to finetune a 1.5B model for math reasoning, and the metric is validation accuracy. The pre-training task is to train a 124M Transformer on FineWeb, and the metric is the training time to reach 3.28 validation loss.

2 Automated Idea Executor
-------------------------

To measure the effectiveness of model-generated ideas, we build an automated executor that takes natural language research ideas as input, generates code implementations, runs the experiments on the backend, and returns the idea’s benchmark performance as the final output.

### 2.1 Research Environments for Ideation

Our automated idea executor is grounded in specific research environments, where each environment consists of a research problem, a baseline codebase, a benchmark to measure performance on, fixed training and evaluation data, and evaluation metrics. When constructing the research environments, we aim to select research problems that are open-ended, so that there is ample room for new algorithmic innovations, and at the same time have well-established baselines and benchmarking metrics so that measuring effectiveness is straightforward. In this work, we construct both a pre-training environment and a post-training environment for the automated AI researchers to work on.1 1 1 We open-source our environments and idea execution trajectories in [https://github.com/NoviScl/Automated-AI-Researcher](https://github.com/NoviScl/Automated-AI-Researcher).

Pre-Training Task: Improving nanoGPT In the nanoGPT environment, we provide a baseline codebase adapted from the nanoGPT speedrun(Jordan et al., [2024](https://arxiv.org/html/2601.14525v1#bib.bib13)) and ask the ideator model to brainstorm possible improvements. The original speedrun task is to minimize the time to pre-train a 124M GPT-2 model(Radford et al., [2019](https://arxiv.org/html/2601.14525v1#bib.bib30)) on the FineWeb corpus(Penedo et al., [2024](https://arxiv.org/html/2601.14525v1#bib.bib29)) to reach a validation loss of 3.28 on the validation set on 8 H100 GPUs. We did several modifications to the original speedrun setting. First, we introduce a proxy reward equal to the reciprocal of the validation loss (1 l​o​s​s\frac{1}{loss}) when performing the search and RL experiments in later sections of the paper. This way, we can fix the training wall-clock time to be 25 minutes and ask the model to directly optimize the proxy reward under this fixed budget, so that we can avoid different runs having vastly different runtimes. We report the validation loss or the proxy reward metric in most plots, and only measure and report the training time metric for the top solution in order to directly compare it with the human experts’ solutions on the original nanoGPT speedrun leaderboard. Second, to avoid any possible reward hacking, we freeze all evaluation hyper-parameters and implement an inference function that predicts one future token at a time to prevent models from changing the attention mechanism in a way that leaks future tokens (which happened multiple times during our initial development). We use this inference function during the final validation after each training run.

Post-Training Task: Improving GRPO In the GRPO environment, the baseline is an implementation of the GRPO algorithm(Shao et al., [2024](https://arxiv.org/html/2601.14525v1#bib.bib33)) that finetunes a Qwen2.5-Math-1.5B checkpoint(Yang et al., [2024](https://arxiv.org/html/2601.14525v1#bib.bib47)) on the MATH dataset(Hendrycks et al., [2021](https://arxiv.org/html/2601.14525v1#bib.bib9)). The ideator model needs to brainstorm post-training algorithms that are more effective than the baseline. We specify a fixed training wall-clock time budget and use the max accuracy on the MATH validation set during training as the metric. To prevent reward hacking, we keep all validation-related code in a separate file and do not allow the automatic executor to access or modify it.

In both environments, we do not set any constraints on the ideation scope, so anything between extensive hyperparameter tuning and novel model architecture or training algorithms is within scope.

### 2.2 System design

The automated idea executor can be viewed as a high-level API whose input is a batch of natural language ideas, and the output is the benchmark performance of each idea. There are three core building blocks of this API (Figure [1](https://arxiv.org/html/2601.14525v1#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Towards Execution-Grounded Automated AI Research")): Implementer – the server that generates the code diff for the idea and applies those changes; Scheduler – a middle layer that receives the list of codebases and allocates resources to run experiments; Worker – the cluster with GPU available that runs the experiments and uploads the experiment results.

#### Implementer

The implementer is hosted on a CPU machine with high IO capacity. First, the user submits a batch of natural language ideas. Then, for each idea, the implementer makes parallelized API calls to the code execution LLM to obtain a diff file that can be patched into the corresponding baseline codebase. To optimize for efficiency, we prompt the code execution LLM with both the idea and the baseline codebase to sample 10 10 code diff files in parallel. For each sample, if the generated diff file cannot be patched into the original codebase, we provide the patch log and ask the model to revise the original generation. We repeat this sequential self-revision for a maximum of 2 2 times. In the end, we return the first code diff file that can be successfully patched into the baseline codebase. The patched codebase is then submitted to a cloud bucket as a .zip file.

#### Scheduler

Under a set clock frequency, the scheduler downloads the new codebases from the cloud. If the codebase has not been executed, the scheduler examines the resource requirement of the given research environment and prepares a job configuration to be submitted.

#### Worker

Once the scheduler finds available resources, it connects the prepared job configuration with the GPU resource and initializes the worker to run the experiment. If the execution of the experiment is successful, the worker will upload the experiment logs including all performance metrics to another cloud bucket (wandb) along with the complete metadata: idea content, code change, execution log, etc. If the execution fails (e.g., due to bugs in code implementation), the worker halts. The user (i.e., the ideator model) can then download the execution results and see the performance of the batch of ideas they submitted with full training logs.

![Image 2: Refer to caption](https://arxiv.org/html/2601.14525v1/x2.png)

(a)Self-Execution (GRPO)

![Image 3: Refer to caption](https://arxiv.org/html/2601.14525v1/x3.png)

(b)Self-Execution (nanoGPT)

![Image 4: Refer to caption](https://arxiv.org/html/2601.14525v1/x4.png)

(c)GPT-5 Execution (GRPO)

![Image 5: Refer to caption](https://arxiv.org/html/2601.14525v1/x5.png)

(d)GPT-5 Execution (nanoGPT)

Figure 2: Model performance comparison with self-execution (top row) vs GPT-5 execution (bottom row) on the GRPO and nanoGPT environments. The baseline accuracy for GRPO is 0.480, and the baseline loss for nanoGPT is 3.255. The completion rate is high for most models, especially under self-execution.

3 Benchmarking LLM Ideators and Executors
-----------------------------------------

The prerequisite for an execution-grounded feedback loop is that current LLMs can serve as both ideators and executors, so that we can get meaningful reward signals for the models to learn from. To examine this prerequisite, we first benchmark various frontier LLMs as both the ideator and the executor.

### 3.1 End-to-End Ideation and Execution

In the first setting, we sample ideas from an LLM, and use the same LLM as the code execution model to execute its own ideas. We sample and execute 50 ideas from Claude-4.5-Opus, Claude-4.5-Sonnet, and GPT-5, and measure several metrics: (1) completion rate: the percentage of ideas that are successfully executed with a valid (non-zero) experiment result after execution; (2) average performance: the average validation accuracy or loss for all the successfully executed ideas among the 50 samples; (3) best performance: the highest validation accuracy or lowest validation loss among all executed ideas. We present results in the top row of Figure[2](https://arxiv.org/html/2601.14525v1#S2.F2 "Figure 2 ‣ Worker ‣ 2.2 System design ‣ 2 Automated Idea Executor ‣ Towards Execution-Grounded Automated AI Research"). Notably, a large fraction of the sampled ideas can indeed be executed successfully, with Claude-4.5-Opus and Claude-4.5-Sonnet having a significantly higher execution rate than GPT-5. Moreover, the best-of-N performance (N=50 N=50) of these models can already beat the original baseline solutions. For example, on the GRPO environment, Claude-4.5-Sonnet gets a max accuracy of 60.4% as compared to the baseline of 48.0%; on nanoGPT, Claude-4.5-Opus gets a lowest loss of 3.237 as compared to the baseline of 3.255.

### 3.2 Comparing Ideators with the Same Executor

In the second setting, we fix the executor model to be GPT-5 and use different ideator models to sample ideas. As shown in the bottom row of Figure[2](https://arxiv.org/html/2601.14525v1#S2.F2 "Figure 2 ‣ Worker ‣ 2.2 System design ‣ 2 Automated Idea Executor ‣ Towards Execution-Grounded Automated AI Research"), even when the ideator and executor are different models, the execution rate is still decent (ranging from 42% to 78%), although we do notice that the same ideas from Claude-4.5-Sonnet get a lower execution rate when executed by GPT-5 instead of itself (84% vs 42% on GRPO and 90% vs 78% on nanoGPT). Moreover, frontier open-weight models like Kimi-K2-Thinking(Kimi Team, [2025](https://arxiv.org/html/2601.14525v1#bib.bib14)) and Qwen3-235B-A22B(Yang et al., [2025a](https://arxiv.org/html/2601.14525v1#bib.bib48)) can also get non-trivial completion rates and achieve best-of-N performance that outperforms the baseline solutions in this setting. For example, Qwen3-235B achieves a max accuracy of 50.2% on GRPO and min loss of 3.238 on nanoGPT with N=50 N=50, both better than the baselines.

These benchmarking results demonstrate the feasibility of the automated ideation and execution loop. Next, we build search scaffolds and RL training loops to examine whether models can learn from the execution feedback.

![Image 6: Refer to caption](https://arxiv.org/html/2601.14525v1/x6.png)

![Image 7: Refer to caption](https://arxiv.org/html/2601.14525v1/x7.png)

Figure 3: Best performance at each epoch when performing execution-guided search with different models. For the nanoGPT environment (left), we use the reciprocal of the validation loss as the metric; for the GRPO environment (right), we use validation accuracy as the metric. Claude-4.5-Opus exhibits a scaling trend on both environments and achieves the best performance on nanoGPT. Claude-4.5-Sonnet achieves the best performance on GRPO due to effective hyper-parameter tuning, but saturates early.

Algorithm 1 Execution-Guided Search

1:batch size

N N
, epochs

T T
, baseline performance

β\beta

2:initial exploitation rate

a 1∈[0,100]a_{1}\in[0,100]
, annealing schedule

a​(t)a(t)
for

t∈{1,…,T}t\in\{1,\dots,T\}

3:Sample initial batch of ideas

ℐ 0←SampleIdeas​(N)\mathcal{I}_{0}\leftarrow\textsc{SampleIdeas}(N)

4:Execute

ℐ 0\mathcal{I}_{0}
to obtain trajectories

𝒟 0←{(idea,reward)}\mathcal{D}_{0}\leftarrow\{(\text{idea},\text{reward})\}

5:for

t=1 t=1
to

T T
do

6:

a←a​(t)a\leftarrow a(t)
⊳\triangleright(100−a)%(100-a)\% exploration rate

7:

𝒟<t←⋃k=0 t−1 𝒟 k\mathcal{D}_{<t}\leftarrow\bigcup_{k=0}^{t-1}\mathcal{D}_{k}

8:

𝒟+←{(i,r)∈𝒟<t:r>β}\mathcal{D}^{+}\leftarrow\{(i,r)\in\mathcal{D}_{<t}\;:\;r>\beta\}
⊳\triangleright positive trajectories

9:

N exp←⌊a 100​N⌋N_{\text{exp}}\leftarrow\left\lfloor\frac{a}{100}\,N\right\rfloor
,

N expl←N−N exp N_{\text{expl}}\leftarrow N-N_{\text{exp}}

10:

ℐ t exp←ExploitVariants​(𝒟+,N exp)\mathcal{I}^{\text{exp}}_{t}\leftarrow\textsc{ExploitVariants}(\mathcal{D}^{+},N_{\text{exp}})

11:

𝒟~<t←SubsampleToContext​(𝒟<t)\tilde{\mathcal{D}}_{<t}\leftarrow\textsc{SubsampleToContext}(\mathcal{D}_{<t})

12:

ℐ t expl←ExploreNovel​(𝒟~<t,N expl)\mathcal{I}^{\text{expl}}_{t}\leftarrow\textsc{ExploreNovel}(\tilde{\mathcal{D}}_{<t},N_{\text{expl}})

13:

ℐ t←ℐ t exp∪ℐ t expl\mathcal{I}_{t}\leftarrow\mathcal{I}^{\text{exp}}_{t}\cup\mathcal{I}^{\text{expl}}_{t}

14: Execute

ℐ t\mathcal{I}_{t}
to obtain trajectories

𝒟 t←{(idea,reward)}\mathcal{D}_{t}\leftarrow\{(\text{idea},\text{reward})\}

15:end for

16:return

⋃t=0 T 𝒟 t\bigcup_{t=0}^{T}\mathcal{D}_{t}

4 Execution-Guided Evolutionary Search
--------------------------------------

Evolutionary search(Koza, [1994](https://arxiv.org/html/2601.14525v1#bib.bib15); Lehman et al., [2023](https://arxiv.org/html/2601.14525v1#bib.bib18)) is a traditional optimization method without the need for gradient updates. We develop an evolutionary search scaffold on top of frontier LLMs to optimize for effective ideas based on execution feedback. We introduce our search method that blends exploration and exploitation, its effectiveness on our two research environments, and various analyses of the generated ideas throughout the evolutionary search process.

### 4.1 Search Scaffold

Our search method is inspired by prior evolutionary search approaches for code optimization, such as AlphaEvolve(Novikov et al., [2025](https://arxiv.org/html/2601.14525v1#bib.bib28)). Our algorithm is detailed in Algorithm[1](https://arxiv.org/html/2601.14525v1#alg1 "Algorithm 1 ‣ 3.2 Comparing Ideators with the Same Executor ‣ 3 Benchmarking LLM Ideators and Executors ‣ Towards Execution-Grounded Automated AI Research"). At the first search epoch, we sample a full batch of new ideas. In all subsequent epochs, we split the idea generation into exploitation and exploration subsets. For exploitation, we choose ideas from previous epochs that outperform the baseline and append them to the idea generation prompt to ask the ideator model to generate new variants that combine their strengths. For exploration, we randomly sample ideas from previous epochs to append to the idea generation prompt until reaching the max context length and instruct the ideator model to generate completely new ideas different from them. We start with 50% exploitation and 50% exploration at epoch 1 and gradually anneal the exploration rate and increase the exploitation ratio in later epochs. We use a batch size of 50 for the GRPO environment and a batch size of 80 for the nanoGPT environment.

### 4.2 Experiment Results

For each environment, we perform execution-guided search with three different models: Claude-4.5-Opus, Claude-4.5-Sonnet, and GPT-5. For each experiment, we use the same model as both the ideator and executor (i.e., self-execution). We plot the progression of the best performance at each search epoch in Figure[3](https://arxiv.org/html/2601.14525v1#S3.F3 "Figure 3 ‣ 3.2 Comparing Ideators with the Same Executor ‣ 3 Benchmarking LLM Ideators and Executors ‣ Towards Execution-Grounded Automated AI Research"). We summarize several notable trends below.

First, we observe a scaling trend with Claude-4.5-Opus, where searching for more epochs leads to a higher upper bound. In contrast, Claude-4.5-Sonnet and GPT-5 tend to saturate early. Second, all models can find ideas that significantly outperform the baselines. On GRPO, Claude-4.5-Sonnet finds that using vanilla policy gradient with the group-average baseline without importance reweighting or clipping outperforms the standard GRPO objective in this particular experiment setup and exploits this finding in all subsequent search epochs, resulting in the best solution of 69.4% at epoch 2 with precise hyper-parameter tuning. On nanoGPT, Claude-4.5-Opus achieves the min validation loss of 3.1407 at epoch 9 by combining various architectural modifications, hyper-parameter tuning, and applying exponential moving average of intermediate checkpoints during validation (see Appendix[A.2](https://arxiv.org/html/2601.14525v1#A1.SS2 "A.2 Additional Idea Examples ‣ Appendix A Appendix ‣ Towards Execution-Grounded Automated AI Research") for the full idea). We run this top solution on 8 H100s to follow the same setup as the nanoGPT speedrun, and it reaches the 3.28 target validation loss in 19.7 minutes, a significant speedup as compared to the baseline codebase, which takes 35.9 minutes of training time to reach the same target validation loss.

To better contextualize these solutions optimized by the model, we also compare the top performance of our execution-guided search to human experts (Table[1](https://arxiv.org/html/2601.14525v1#S1.T1 "Table 1 ‣ 1 Introduction ‣ Towards Execution-Grounded Automated AI Research")). For the GRPO environment, we compare with the leaderboard of the Stanford CS336 graduate-level LLM class, which hosted the same environment as an assignment for all students to optimize the validation accuracy under the same training time budget. The best student solution 2 2 2[https://github.com/stanford-cs336/assignment5-alignment-leaderboard](https://github.com/stanford-cs336/assignment5-alignment-leaderboard) achieved an accuracy of 68.8%, lower than Claude-4.5-Sonnet’s top solution using our execution-guided search. For the nanoGPT environment, we directly compare with the nanoGPT speedrun leaderboard,3 3 3[https://github.com/KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) where the top human solution as of December 2025 can reach the target validation loss under 2.1 minutes, indicating significant headroom for further model capability and search method improvement on this environment.

![Image 8: Refer to caption](https://arxiv.org/html/2601.14525v1/x8.png)

Figure 4: Comparison between best-of-N (blue) and our execution-guided search (red) under the same sampling budget where we sample N=80 ideas at each epoch.

Table 2: Breakdown of hyper-parameter tuning vs algorithmic ideas throughout the entire execution-guided search. We report the percentage of each type among all generated ideas of each model (N=500 N=500 ideas on GRPO and N=800 N=800 ideas on nanoGPT). We also report the average and best performance for ideas under each category, where we use validation accuracy as the performance metric for GRPO and validation loss as the metric for nanoGPT. Bold numbers every row indicate the best performance by each model. All models generate a substantial amount of algorithmic ideas apart from hyper-parameter changes, while Claude-4.5-Sonnet generates significantly more hyper-parameter ideas than other models.

Hyper-parameter Algorithmic
Model name Percentage Average Best Percentage Average Best
_GRPO environment (accuracy↑)_
GPT-5 5.0%45.0%50.2%95.0%44.5%60.0%
Claude-4.5-Sonnet 41.1%48.4%69.4%58.9%45.0%67.4%
Claude-4.5-Opus 3.7%44.4%50.4%96.3%46.5%61.6%
_nanoGPT environment (loss↓)_
GPT-5 15.4%3.254 3.195 84.6%3.894 3.170
Claude-4.5-Sonnet 31.3%3.251 3.208 68.7%3.679 3.208
Claude-4.5-Opus 8.7%3.329 3.147 91.3%3.419 3.141

Claude-4.5-Opus on GRPO Claude-4.5-Sonnet on GRPO GPT-5 on GRPO
Residual Ratio Learning with Momentum Bounds: Instead of directly using the (importance sampling) ratio, decompose it into a “base” component (EMA of batch mean ratios) and a “residual” component (ratio – base). Apply sigmoid bounding only to the residual, allowing the base to capture systematic policy drift while controlling deviations from it. Combined with momentum clip adaptation. Formula: residual = ratio - ema_batch_ratio, bounded_residual = sigmoid_bound(residual, deviation), effective_ratio = 1.0 + bounded_residual.Validation Accuracy: 61.6 Advantage Rank Difference Weighting: Instead of using absolute advantage magnitude, weight samples by how far their rank differs from their expected rank under uniform distribution. Samples that significantly outperform or underperform their “expected” position get higher weights. This is distribution-free and robust to outliers. Formula: expected_rank = (N-1)/2, rank_diff = |actual_rank - expected_rank| / expected_rank, weight = 0.5 + 0.5 * rank_diff.Validation Accuracy: 59.2 Dynamic Mathematical Problem Difficulty Balancing with Performance Feedback: Implement intelligent difficulty balancing that dynamically adjusts the mix of problem difficulties based on recent performance trends. When performance is strong, increase difficulty proportion; when struggling, provide more foundational problems. Combine with the proven hyper-parameters for optimal learning progression.Validation Accuracy: 64.0 Implement token-level reward attribution by using attention weights to identify which input tokens contributed most to correct answers, then amplifying the gradient updates for those tokens during policy gradient training.Validation Accuracy: 45.2 Create mathematical working memory simulation by maintaining a context buffer of mathematical facts, definitions, and intermediate results during problem solving. This buffer gets updated as the model works through problems and provides additional context for subsequent mathematical steps, simulating how humans maintain mathematical working memory during complex calculations.Validation Accuracy: 58.0 Token-Level Ratio De-noising via Response Chunks (Chunked-Ratio): Reduce noisy token spikes by averaging log-ratio over small contiguous chunks within the response. Partition response tokens into C C chunks per sequence (e.g., C=8 C=8 over effective length), replace per-token Δ​log⁡p\Delta\log p with chunk mean broadcast to tokens in the chunk before ratio and clipping. Keeps structural signal while smoothing extremes.Validation Accuracy: 58.2 Per-Group Curriculum via Reward Spread (PGC-RS): Adjust step aggressiveness based on group reward spread. For each group, compute spread s g=std​(r)s_{g}=\mathrm{std}(r). Compute per-sample temperature T i grp=clamp​(1+α⋅(s ref−s g), 0.8, 1.5)T^{\mathrm{grp}}_{i}=\mathrm{clamp}\!\big(1+\alpha\cdot(s_{\mathrm{ref}}-s_{g}),\,0.8,\,1.5\big) with s ref=s_{\mathrm{ref}}= median group std over rollout and α=0.8\alpha=0.8. Multiply existing ratio temperature T i T_{i} (if any) by T i grp T^{\mathrm{grp}}_{i}. Groups with low spread (hard to distinguish) get cooler ratios; high-spread groups allow bolder updates.Validation Accuracy: 49.4

Table 3: Examples of successfully executed ideas on the GRPO environment, along with their accuracy on the MATH validation set. The baseline accuracy is 48.0% on this environment.

### 4.3 Comparison with Best-of-N

To demonstrate the effectiveness of our search scaffold, we compare our execution-guided search with the best-of-N baseline with the same sampling budget on the nanoGPT environment. Since the batch size for our search is 80, we compare the first 3 epochs of the execution-guided search using the GPT-5 backbone with the best-of-N results of GPT-5 with N∈{80,160,240}N\in\{80,160,240\}. As shown in Figure[4](https://arxiv.org/html/2601.14525v1#S4.F4 "Figure 4 ‣ 4.2 Experiment Results ‣ 4 Execution-Guided Evolutionary Search ‣ Towards Execution-Grounded Automated AI Research"), search and best-of-N start from similar performance at epoch 0 (they are not exactly the same due to variances from sampling), but evolutionary search significantly outperforms best-of-N from epoch 1 onward, demonstrating that the model is effectively leveraging trajectories from previous epochs to generate more effective ideas in future epochs.

### 4.4 Analysis of Generated Ideas

Hyper-parameter vs Algorithmic To quantitatively understand the types of ideas that models generate during the execution-guided search, we perform a stratified analysis by classifying all generated ideas into either hyper-parameter tuning (including any ideas that can be implemented via changing existing configs) or algorithmic (including all ideas that involve implementing new changes not originally supported by the baseline codebase) by using an LLM-judge. Based on Table[2](https://arxiv.org/html/2601.14525v1#S4.T2 "Table 2 ‣ 4.2 Experiment Results ‣ 4 Execution-Guided Evolutionary Search ‣ Towards Execution-Grounded Automated AI Research"), all three models generate a substantial amount of algorithmic ideas apart from hyper-parameter tuning. Interestingly, different models exhibit different patterns, where Claude-4.5-Sonnet generates significantly more hyper-parameter ideas than both Claude-4.5-Opus and GPT-5. Moreover, the most effective ideas come from algorithmic ideas in most cases, except when using Claude-4.5-Sonnet.

Qualitative Examples To complement the quantitative analysis, we provide several executed ideas on the GRPO environment in Table[3](https://arxiv.org/html/2601.14525v1#S4.T3 "Table 3 ‣ 4.2 Experiment Results ‣ 4 Execution-Guided Evolutionary Search ‣ Towards Execution-Grounded Automated AI Research") and provide example ideas on the nanoGPT environment in Appendix[A.2](https://arxiv.org/html/2601.14525v1#A1.SS2 "A.2 Additional Idea Examples ‣ Appendix A Appendix ‣ Towards Execution-Grounded Automated AI Research"). When sampling ideas, models would generate a thinking trace, followed by the natural language idea and a brief description of all the code changes needed to implement the idea. For brevity, we only include the natural language ideas in the table, but we present additional examples in Appendix[A.3](https://arxiv.org/html/2601.14525v1#A1.SS3 "A.3 Code Execution Examples ‣ Appendix A Appendix ‣ Towards Execution-Grounded Automated AI Research") with more details, including full code execution trajectories. Based on the qualitative examples in Table[3](https://arxiv.org/html/2601.14525v1#S4.T3 "Table 3 ‣ 4.2 Experiment Results ‣ 4 Execution-Guided Evolutionary Search ‣ Towards Execution-Grounded Automated AI Research"), different models generate different styles of ideas. For example, Claude-4.5-Sonnet generates more intuitive ideas while Claude-4.5-Opus and GPT-5 are more mathematically inclined.

Recovering Recent Research Papers We observed multiple cases where the model-generated ideas (without any RAG) are highly similar to research papers released within the three months prior to the writing of this paper. For example, Claude-4.5-Sonnet proposed: “Implement response diversity rewards within groups where responses to the same prompt receive bonus rewards for being dissimilar to other responses in their group, encouraging exploration of different solution paths.”, which is similar to Li et al. ([2025](https://arxiv.org/html/2601.14525v1#bib.bib19)). For pre-training, Claude-4.5-Opus proposed: “Causal Context Compression: Before each attention layer, apply a learned compression that mixes local context (previous 2-3 tokens) into the current representation, providing implicit local context without convolutions.”, which is similar to the “canon layer” described in Allen-Zhu ([2025](https://arxiv.org/html/2601.14525v1#bib.bib1)). Although assessing the novelty of LLM-generated ideas is beyond the scope of this work, models’ ability to rediscover ideas from recent research papers nevertheless indicates that automated AI researchers could plausibly support work at the frontier of LLM research.

5 Reinforcement Learning from Execution Reward
----------------------------------------------

Different from evolutionary search, reinforcement learning is an alternative learning algorithm that shapes model behaviors through gradient updates. Despite much recent success on verifiable domains like math and coding(DeepSeek-AI et al., [2025](https://arxiv.org/html/2601.14525v1#bib.bib6)), RL’s effectiveness on open-ended AI research remains unclear. For the first time, we explore whether we can leverage the automated executor as a reward function to directly finetune LLMs to generate more effective ideas via RL. We detail our implementation, experiment setup, and analysis of the training dynamics.

![Image 9: Refer to caption](https://arxiv.org/html/2601.14525v1/x9.png)

![Image 10: Refer to caption](https://arxiv.org/html/2601.14525v1/x10.png)

![Image 11: Refer to caption](https://arxiv.org/html/2601.14525v1/x11.png)

![Image 12: Refer to caption](https://arxiv.org/html/2601.14525v1/x12.png)

Figure 5: Training curves of RL from execution reward. We plot the average reward per epoch in the upper row, and the max reward per epoch in the lower row. For the GRPO environment, the reward is the accuracy; for the nanoGPT environment, the reward is the reciprocal of the loss. The average reward increases, but not the max reward.

### 5.1 Reward Design and Experiment Setup

We use Qwen3-30B-A3B(Yang et al., [2025a](https://arxiv.org/html/2601.14525v1#bib.bib48)) as the base model and finetune it using the standard GRPO algorithm(Shao et al., [2024](https://arxiv.org/html/2601.14525v1#bib.bib33)), motivated by its consistent empirical success on other verified domains. Our prompt batch size is one since we only have one prompt for each environment. In the prompt, we provide the baseline codebase and ask the model to generate new ideas to improve the baseline (GRPO or nanoGPT). This experiment setup is similar to prior work exploring RLVR from one training example(Wang et al., [2025](https://arxiv.org/html/2601.14525v1#bib.bib43)).

We use large group sizes to stabilize training. For the post-training environment, we use a group size of 256; for the pre-training environment, we use a group size of 128. Since each idea on the GRPO environment runs on one single GPU and each idea on the nanoGPT environment runs on 8 GPUs, these group sizes correspond to parallel execution on 256 GPUs (for GRPO) or 1024 GPUs (for nanoGPT) to obtain the execution reward on each batch of rollout ideas. Each rollout is a thinking trace followed by the natural language idea. We set a max output length of 8192 tokens for rollout sampling and only feed the extracted ideas to the automated executor without the preceding thinking trace.

For the post-training environment, we directly use the validation set accuracy of each rollout idea after execution as the reward. For ideas without a valid accuracy (i.e., when the execution failed due to code generation errors), we assign a reward of 0. For the pre-training environment, we use the reciprocal of the validation loss as the reward (1 l​o​s​s\frac{1}{loss}) and assign a reward of 0 to ideas with failed execution. Our experiments are based on the Tinker API(Thinking Machines Lab, [2025](https://arxiv.org/html/2601.14525v1#bib.bib39)).

### 5.2 Experiment Results

#### Positive Training Curves for Average Reward

We plot the average reward of all rollouts of each training epoch in the upper row of Figure[5](https://arxiv.org/html/2601.14525v1#S5.F5 "Figure 5 ‣ 5 Reinforcement Learning from Execution Reward ‣ Towards Execution-Grounded Automated AI Research"). For the first time, we demonstrate that the average performance of the generated ideas can increase after sufficient training epochs for open-ended research environments. For instance, the average accuracy on the GRPO environment increases from 0.253 at the beginning to 0.343 after 40 training epochs (top left plot of Figure[5](https://arxiv.org/html/2601.14525v1#S5.F5 "Figure 5 ‣ 5 Reinforcement Learning from Execution Reward ‣ Towards Execution-Grounded Automated AI Research")); and the average reward on the nanoGPT environment increases from 0.194 at the beginning to 0.246 after 68 epochs (top right plot of Figure[5](https://arxiv.org/html/2601.14525v1#S5.F5 "Figure 5 ‣ 5 Reinforcement Learning from Execution Reward ‣ Towards Execution-Grounded Automated AI Research")), corresponding to a decrease in the average validation loss from 5.150 to 4.066. Such training curves are similar to prior findings on the effectiveness of one-shot RVLR on other verified domains like math(Wang et al., [2025](https://arxiv.org/html/2601.14525v1#bib.bib43)).

#### The Case of Max Reward

Despite successfully reproducing the positive training curves observed in other domains, we argue that there is a distinction between idea generation and other verifiable domains. For advancing scientific discovery, we often care about the upper-bound of idea generation, rather than the average quality. In our particular case, we care more about having one breakthrough idea that dominates the baselines, rather than having many safe ideas with a high average. Thus, we plot the max reward of all rollouts at each training epoch in the lower row of Figure[5](https://arxiv.org/html/2601.14525v1#S5.F5 "Figure 5 ‣ 5 Reinforcement Learning from Execution Reward ‣ Towards Execution-Grounded Automated AI Research"). The trend here is very different – the max reward is fluctuating throughout RL training without a clear upward trend. This reveals the crucial limitation of the standard GRPO algorithm for improving idea generation. In the next subsection, we analyze why RL from execution reward improves the average reward but not the max.

![Image 13: Refer to caption](https://arxiv.org/html/2601.14525v1/x13.png)

![Image 14: Refer to caption](https://arxiv.org/html/2601.14525v1/x14.png)

![Image 15: Refer to caption](https://arxiv.org/html/2601.14525v1/x15.png)

![Image 16: Refer to caption](https://arxiv.org/html/2601.14525v1/x16.png)

Figure 6: Upper Row: Average length of the thinking trace (red line) and the idea (blue line) per training epoch. Lower Row: Execution rate of ideas with the longest (red line) versus shortest (blue line) thinking traces. Ideas with longer thinking have lower execution rates. Correspondingly, thinking lengths decrease in RL.

![Image 17: Refer to caption](https://arxiv.org/html/2601.14525v1/x17.png)

Figure 7: We plot how many ideas in each epoch are about either replacing RMSNorm with LayerNorm or doing EMA. The model converged on these two ideas after RL training.

### 5.3 Analysis of Training Dynamics

#### Thinking Length

We first plot how the lengths of the thinking traces evolve over RL training in the upper row of Figure[6](https://arxiv.org/html/2601.14525v1#S5.F6 "Figure 6 ‣ The Case of Max Reward ‣ 5.2 Experiment Results ‣ 5 Reinforcement Learning from Execution Reward ‣ Towards Execution-Grounded Automated AI Research"). In both environments, the thinking traces rapidly decrease in length while the idea lengths stay roughly constant. This is the opposite of the thinking emergence trend from prior RLVR work, such as DeepSeek-R1(DeepSeek-AI et al., [2025](https://arxiv.org/html/2601.14525v1#bib.bib6)). To further understand why the thinking length decreases, we investigate the correlation between the idea execution rate and the thinking trace length. In the bottom row of Figure[6](https://arxiv.org/html/2601.14525v1#S5.F6 "Figure 6 ‣ The Case of Max Reward ‣ 5.2 Experiment Results ‣ 5 Reinforcement Learning from Execution Reward ‣ Towards Execution-Grounded Automated AI Research"), for the first 20 epochs of the RL training, we sort all ideas in each epoch by their thinking trace lengths and plot the average execution rate of the top-30% longest thinking ideas (red line) and the bottom-30% shortest thinking ideas (blue line). We see a clear trend where ideas with longer thinking consistently have a lower execution rate. We thus hypothesize that longer thinking correlates with more complex ideas with lower execution rates, leading the model to prefer shorter thinking instead in order to maximize the reward.

#### Diversity Collapse

Upon manual investigation of all the rollouts being sampled throughout the RL training, we also observed a diversity collapse. Specifically, the model learned to converge on a few simple ideas that can consistently get a positive reward. For example, in the nanoGPT environment, the model learned to converge towards two common ideas: (1) replacing RMSNorm with LayerNorm; and (2) performing exponential moving average (EMA) over intermediate model checkpoints. As shown in Figure[7](https://arxiv.org/html/2601.14525v1#S5.F7 "Figure 7 ‣ The Case of Max Reward ‣ 5.2 Experiment Results ‣ 5 Reinforcement Learning from Execution Reward ‣ Towards Execution-Grounded Automated AI Research"), out of a batch of 128 sampled ideas per epoch, 51 ideas sampled from Qwen3-30B at epoch 0 are one of the two common ideas above. Towards the end of the RL training, 119 out of 128 sampled ideas at epoch 68 are one of the two common ideas, indicating a severe diversity collapse.

The above analyses reveal that RL causes the models to converge on a few simple-to-implement ideas, accompanied by shrinking thinking lengths. These lead to an increase in the average reward, but do not push the upper-bound due to the lack of exploration. This phenomenon is analogous to mode-collapse observations on other verifiable domains, where the pass@k performance stagnates or even decreases after RL(Yue et al., [2025](https://arxiv.org/html/2601.14525v1#bib.bib50); Wu et al., [2025](https://arxiv.org/html/2601.14525v1#bib.bib45)). Avoiding such convergence and collapse is an open problem and likely requires new algorithmic interventions beyond standard GRPO, which is beyond the scope of this work. However, we do share several preliminary attempts, including: sampling and appending previous epochs’ trajectories into the current epoch’s prompt for rollout sampling, adding a weighted length reward in the total reward, and adding a weighted similarity penalty in the total reward. We did not observe clear gains in the initial epochs and thus early-stopped them, but we document all these results in Appendix[A.1](https://arxiv.org/html/2601.14525v1#A1.SS1 "A.1 Other RL Attempts ‣ Appendix A Appendix ‣ Towards Execution-Grounded Automated AI Research") to inform future work.

6 Related Work
--------------

#### AutoML

Our work has deep connections to the AutoML literature. For example, the Neural Architecture Search (NAS) line of work typically defines a constrained set of neural network operators and optimizes for architectures based on validation set performance through reinforcement learning(Zoph & Le, [2017](https://arxiv.org/html/2601.14525v1#bib.bib54); Zoph et al., [2017](https://arxiv.org/html/2601.14525v1#bib.bib55)) or search(Liu et al., [2018](https://arxiv.org/html/2601.14525v1#bib.bib21); So et al., [2019](https://arxiv.org/html/2601.14525v1#bib.bib36)). In the modern era, recent works also explored directly using LLMs to propose architecture variants and implement them for validation(Liu et al., [2025](https://arxiv.org/html/2601.14525v1#bib.bib22); Cheng et al., [2025](https://arxiv.org/html/2601.14525v1#bib.bib5)). Beyond architectures, similar automatic optimizations have been applied to improve hyperparameter tuning(Zhang et al., [2023](https://arxiv.org/html/2601.14525v1#bib.bib51)), discover machine learning algorithms(Real et al., [2020](https://arxiv.org/html/2601.14525v1#bib.bib31)), improve post-training objectives(Lu et al., [2024a](https://arxiv.org/html/2601.14525v1#bib.bib23)), discover better neural network optimizers(Chen et al., [2023](https://arxiv.org/html/2601.14525v1#bib.bib4)), and design agent scaffolds(Hu et al., [2025](https://arxiv.org/html/2601.14525v1#bib.bib10)). Different from this line of work, we tackle automated AI research in a fully open-ended setting without any constraint on the type of ideas. Moreover, our goal is to improve the effectiveness of idea generation, where natural language ideas represent a higher level of abstraction than specific architecture variants or code optimizations.

#### LLM-based Research Agents

Recent works have been building LLM-based research agents for accelerating scientific discovery in various domains, including AI research. AI-Scientist(Lu et al., [2024b](https://arxiv.org/html/2601.14525v1#bib.bib24); Yamada et al., [2025](https://arxiv.org/html/2601.14525v1#bib.bib46)), AI-Researcher(Tang et al., [2025](https://arxiv.org/html/2601.14525v1#bib.bib38)), and Agent Laboratory(Schmidgall et al., [2025](https://arxiv.org/html/2601.14525v1#bib.bib32)) are examples of end-to-end research agents that use LLMs to generate ideas and implement them through carefully designed agent scaffolds. They address open-ended AI research as we do, but do not study how to learn from execution feedback and improve idea effectiveness. On the other hand, on more grounded benchmarks with clear performance metrics such as MLE-Bench(Chan et al., [2025](https://arxiv.org/html/2601.14525v1#bib.bib3)), RE-Bench(Wijk et al., [2024](https://arxiv.org/html/2601.14525v1#bib.bib44)), and ML-Gym(Nathani et al., [2025](https://arxiv.org/html/2601.14525v1#bib.bib27)), various works have explored how to learn from execution feedback through either search(Toledo et al., [2025](https://arxiv.org/html/2601.14525v1#bib.bib41); Jiang et al., [2025](https://arxiv.org/html/2601.14525v1#bib.bib12)) or RL(Yang et al., [2025b](https://arxiv.org/html/2601.14525v1#bib.bib49)) to optimize performance on these targeted ML engineering tasks. While we also study algorithms for learning from execution feedback, we tackle open-ended research problems like pre-training and post-training rather than ML engineering tasks that heavily depend on feature engineering and hyper-parameter tuning rather than algorithm development.

#### AI for Research

Apart from fully end-to-end automated AI research, many works have studied how to use LLMs for specific components of the scientific research pipeline, such as literature review(Asai et al., [2024](https://arxiv.org/html/2601.14525v1#bib.bib2); L’ala et al., [2023](https://arxiv.org/html/2601.14525v1#bib.bib16)), idea generation(Si et al., [2025b](https://arxiv.org/html/2601.14525v1#bib.bib35); Wang et al., [2024](https://arxiv.org/html/2601.14525v1#bib.bib42)), data analysis(Majumder et al., [2025](https://arxiv.org/html/2601.14525v1#bib.bib25); Mitchener et al., [2025](https://arxiv.org/html/2601.14525v1#bib.bib26)), experiment plan generation(Goel et al., [2025](https://arxiv.org/html/2601.14525v1#bib.bib8)), research code execution(Starace et al., [2025](https://arxiv.org/html/2601.14525v1#bib.bib37); Hua et al., [2025](https://arxiv.org/html/2601.14525v1#bib.bib11); Tian et al., [2024](https://arxiv.org/html/2601.14525v1#bib.bib40)), and paper reviewing(Liang et al., [2024](https://arxiv.org/html/2601.14525v1#bib.bib20); Zhu et al., [2025](https://arxiv.org/html/2601.14525v1#bib.bib53)). Our work focuses on automated idea execution and learning from the execution feedback. We consider our work complementary to many of the above works that improve other aspects of the scientific research pipeline.

#### Execution Grounding for Code

The idea of learning from execution feedback has been explored in the code generation domain. For example, Zheng et al. ([2024](https://arxiv.org/html/2601.14525v1#bib.bib52)) curate data and train models to refine code from either human or execution feedback; Gehring et al. ([2025](https://arxiv.org/html/2601.14525v1#bib.bib7)) use end-to-end RL training to teach models to improve code based on execution feedback; Lavon et al. ([2025](https://arxiv.org/html/2601.14525v1#bib.bib17)) directly guide code generation with execution signals during inference time. In contrast, our work explores execution grounding for the application of idea generation, where the verification is more complicated and expensive.

7 Conclusion
------------

In this work, we built a large-scale parallel executor for automatically executing model-generated ideas to verify their effectiveness on open-ended LLM research problems, including both LLM pre-training and post-training. Using this executor as a reward function, we analyzed the effectiveness of execution-guided evolutionary search, where frontier LLMs equipped with a simple evolutionary search scaffold can significantly outperform the baseline solutions. We also benchmarked and revealed the limitations of reinforcement learning with execution rewards, where models tend to converge on simple ideas to improve the average reward but lose diversity and do not improve the upper-bound. Our empirical results demonstrate the feasibility and potential of the automated execution feedback loop and also point out the remaining limitations for future improvement. We hope this work paves the foundation and inspires more efforts towards execution-grounded automated AI research.

8 Discussion
------------

Despite encouraging initial signals, there are still many limitations to our current set of experiments, which would be great directions for future improvement.

First, our current procedure does not test the generalizability of the generated ideas. It is possible that the best-performing ideas at the small scales may not transfer to gains at a larger scale or on other datasets. Future work should explore methods that explicitly test such generalizability and scalability, and even incorporate them as part of the optimization objectives.

Second, we have shown that RL with execution reward in our current setup can only improve the average reward but not the upper bound. There are many possible reasons, such as a lack of diversity in the base model and missing exploration incentives in the current RL objective. Future work should explore remedies and better learning algorithms for LLMs to more efficiently learn from the execution feedback. For instance, future works could explore how to exploit richer learning signals from the execution trajectories beyond just scalar rewards.

Third, our current experiment scope is bounded by the capability of the execution agent. There exist many promising model-generated ideas that could not be successfully executed by the execution agent (e.g., see the end of Appendix[A.2](https://arxiv.org/html/2601.14525v1#A1.SS2 "A.2 Additional Idea Examples ‣ Appendix A Appendix ‣ Towards Execution-Grounded Automated AI Research")), leading to noise in the reward signal. Future work could develop more capable execution agents and extend our setup to even more complex research problems. For instance, instead of directly prompting an LLM for code diff, future work can implement more capable coding agents with access to external tools and the ability to install new libraries in the execution environments.

Last but not least, we only explored effectiveness as the training reward in this work. There are many other, more subjective alternative metrics that could complement the effectiveness reward, such as the idea novelty and interestingness. Future work could explore how to computationally measure them and incorporate them as part of the training objective to discover more insightful ideas.

Acknowledgment
--------------

We thank DST Global, Laude Institute, and Thinking Machines Lab for their generous sponsorship of computing resources. We thank Yuandong Tian, Edward Hughes, Ludwig Schmidt, Cong Lu, Jenny Zhang, Jiaxin Wen, Chris Rytting, Chen Zhao, Xinran Zhao, Yanzhe Zhang, John Yang, Shicheng Liu, Andy Zhou, Will Held, Haotian Ye, Luke Bailey, as well as members of Tatsu Lab and SALT Lab for their helpful discussion and feedback. This work was supported by an HAI grant, DSO labs, gifts from Open Philanthropy, Amazon, Schmidt Sciences, the Tianqiao and Chrissy Chen Foundation and a grant under the NSF CAREER IIS-2338866 and IIS-2247357, ONR N00014-24-1-2609 and N00014-24-1-2532, and DARPA Cooperative Agreement HR00112520013. This work does not necessarily reflect the position or policy of the government and no official endorsement should be inferred.

References
----------

*   Allen-Zhu (2025) Allen-Zhu, Z. Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers. _ArXiv_, 2025. 
*   Asai et al. (2024) Asai, A., He, J., Shao, R., Shi, W., Singh, A., Chang, J.C., Lo, K., Soldaini, L., Feldman, S., D’Arcy, M., Wadden, D., Latzke, M., Tian, M., Ji, P., Liu, S., Tong, H., Wu, B., Xiong, Y., Zettlemoyer, L.S., Neubig, G., Weld, D., Downey, D., tau Yih, W., Koh, P.W., and Hajishirzi, H. OpenScholar: Synthesizing Scientific Literature with Retrieval-augmented LMs. _ArXiv_, abs/2411.14199, 2024. 
*   Chan et al. (2025) Chan, J.S., Chowdhury, N., Jaffe, O., Aung, J., Sherburn, D., Mays, E., Starace, G., Liu, K., Maksin, L., Patwardhan, T., Weng, L., and Mkadry, A. MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering. In _ICLR_, 2025. 
*   Chen et al. (2023) Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Liu, Y., Pham, H., Dong, X., Luong, T., Hsieh, C.-J., Lu, Y., and Le, Q.V. Symbolic Discovery of Optimization Algorithms. In _NeurIPS_, 2023. 
*   Cheng et al. (2025) Cheng, J., Clark, P., and Richardson, K. Language Modeling by Language Models. In _NeurIPS_, 2025. 
*   DeepSeek-AI et al. (2025) DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, J.-M., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., Zhang, X., Yu, X., Wu, Y., Wu, Z.F., Gou, Z., Shao, Z., Li, Z., Gao, Z., Liu, A., Xue, B., Wang, B.-L., Wu, B., Feng, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., Dai, D., Chen, D., Ji, D.-L., Li, E., Lin, F., Dai, F., Luo, F., Hao, G., Chen, G., Li, G., Zhang, H., Bao, H., Xu, H., Wang, H., Ding, H., Xin, H., Gao, H., Qu, H., Li, H., Guo, J., Li, J., Wang, J., Chen, J., Yuan, J., Qiu, J., Li, J., Cai, J., Ni, J., Liang, J., Chen, J., Dong, K., Hu, K., Gao, K., Guan, K., Huang, K., Yu, K., Wang, L., Zhang, L., Zhao, L., Wang, L., Zhang, L., Xu, L., Xia, L., Zhang, M., Zhang, M., Tang, M., Li, M., Wang, M., Li, M., Tian, N., Huang, P., Zhang, P., Wang, Q., Chen, Q., Du, Q., Ge, R., Zhang, R., Pan, R., Wang, R., Chen, R.J., Jin, R., Chen, R., Lu, S., Zhou, S., Chen, S., Ye, S., Wang, S., Yu, S., Zhou, S., Pan, S., Li, S.S., Zhou, S., Wu, S.-K., Yun, T., Pei, T., Sun, T., Wang, T., Zeng, W., Zhao, W., Liu, W., Liang, W., Gao, W., Yu, W.-X., Zhang, W., Xiao, W., An, W., Liu, X., Wang, X., Chen, X., Nie, X., Cheng, X., Liu, X., Xie, X., Liu, X., Yang, X., Li, X., Su, X., Lin, X., Li, X.Q., Jin, X., Shen, X.-C., Chen, X., Sun, X., Wang, X., Song, X., Zhou, X., Wang, X., Shan, X., Li, Y.K., Wang, Y.Q., Wei, Y.X., Zhang, Y., Xu, Y., Li, Y., Zhao, Y., Sun, Y., Wang, Y., Yu, Y., Zhang, Y., Shi, Y., Xiong, Y., He, Y., Piao, Y., Wang, Y., Tan, Y., Ma, Y., Liu, Y., Guo, Y., Ou, Y., Wang, Y., Gong, Y., Zou, Y.-J., He, Y., Xiong, Y., Luo, Y.-W., mei You, Y., Liu, Y., Zhou, Y., Zhu, Y.X., Huang, Y., Li, Y., Zheng, Y., Zhu, Y., Ma, Y., Tang, Y., Zha, Y., Yan, Y., Ren, Z., Ren, Z., Sha, Z., Fu, Z., Xu, Z., Xie, Z., guo Zhang, Z., Hao, Z., Ma, Z., Yan, Z., Wu, Z., Gu, Z., Zhu, Z., Liu, Z., Li, Z.-A., Xie, Z., Song, Z., Pan, Z., Huang, Z., Xu, Z., Zhang, Z., and Zhang, Z. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. _Nature_, 2025. 
*   Gehring et al. (2025) Gehring, J., Zheng, K., Copet, J., Mella, V., Cohen, T., and Synnaeve, G. RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning. In _ICML_, 2025. 
*   Goel et al. (2025) Goel, S., Hazra, R., Jayalath, D.H., Willi, T., Jain, P., Shen, W.F., Leontiadis, I., Barbieri, F., Bachrach, Y., Geiping, J., and Whitehouse, C. Training AI Co-Scientists Using Rubric Rewards. _ArXiv_, abs/2512.23707, 2025. 
*   Hendrycks et al. (2021) Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D.X., and Steinhardt, J. Measuring Mathematical Problem Solving With the MATH Dataset. In _NeurIPS_, 2021. 
*   Hu et al. (2025) Hu, S., Lu, C., and Clune, J. Automated Design of Agentic Systems. In _ICLR_, 2025. 
*   Hua et al. (2025) Hua, T., Hua, H., Xiang, V., Klieger, B., Truong, S.T., Liang, W., Sun, F.-Y., and Haber, N. ResearchCodeBench: Benchmarking LLMs on Implementing Novel Machine Learning Research Code. In _NeurIPS_, 2025. 
*   Jiang et al. (2025) Jiang, Z., Schmidt, D., Srikanth, D., Xu, D., Kaplan, I., Jacenko, D., and Wu, Y. AIDE: AI-Driven Exploration in the Space of Code. _ArXiv_, abs/2502.13138, 2025. 
*   Jordan et al. (2024) Jordan, K., Bernstein, J., Rappazzo, B., @fernbear.bsky.social, Vlado, B., Jiacheng, Y., Cesista, F., Koszarsky, B., and @Grad62304977. modded-nanogpt: Speedrunning the nanogpt baseline, 2024. URL [https://github.com/KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt). 
*   Kimi Team (2025) Kimi Team. Kimi K2: Open Agentic Intelligence. _ArXiv_, abs/2507.20534, 2025. 
*   Koza (1994) Koza, J. Genetic Programming: On the Programming of Computers by Means of Natural Selection. _Statistics and Computing_, 4, 1994. 
*   L’ala et al. (2023) L’ala, J., O’Donoghue, O., Shtedritski, A., Cox, S., Rodriques, S.G., and White, A.D. PaperQA: Retrieval-Augmented Generative Agent for Scientific Research. _ArXiv_, abs/2312.07559, 2023. 
*   Lavon et al. (2025) Lavon, B., Katz, S., and Wolf, L. Execution Guided Line-by-Line Code Generation. In _NeurIPS_, 2025. 
*   Lehman et al. (2023) Lehman, J., Gordon, J., Jain, S., Ndousse, K., Yeh, C., and Stanley, K.O. Evolution through large models. In _Handbook of Evolutionary Machine Learning_. Springer, 2023. 
*   Li et al. (2025) Li, T., Zhang, Y., Yu, P., Saha, S., Khashabi, D., Weston, J.E., Lanchantin, J., and Wang, T. Jointly Reinforcing Diversity and Quality in Language Model Generations. _ArXiv_, abs/2509.02534, 2025. 
*   Liang et al. (2024) Liang, W., Zhang, Y., Cao, H., Wang, B., Ding, D., Yang, X., Vodrahalli, K., He, S., Smith, D.S., Yin, Y., McFarland, D.A., and Zou, J. Can large language models provide useful feedback on research papers? a large-scale empirical analysis. _NEJM AI_, 2024. 
*   Liu et al. (2018) Liu, H., Simonyan, K., Vinyals, O., Fernando, C., and Kavukcuoglu, K. Hierarchical Representations for Efficient Architecture Search. In _ICLR_, 2018. 
*   Liu et al. (2025) Liu, Y., Nan, Y., Xu, W., Hu, X., Ye, L., Qin, Z., and Liu, P. AlphaGo Moment for Model Architecture Discovery. _ArXiv_, abs/2507.18074, 2025. 
*   Lu et al. (2024a) Lu, C., Holt, S., Fanconi, C., Chan, A.J., Foerster, J.N., van der Schaar, M., and Lange, R.T. Discovering Preference Optimization Algorithms with and for Large Language Models. In _NeurIPS_, 2024a. 
*   Lu et al. (2024b) Lu, C., Lu, C., Lange, R.T., Foerster, J.N., Clune, J., and Ha, D. The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. _ArXiv_, abs/2408.06292, 2024b. 
*   Majumder et al. (2025) Majumder, B.P., Surana, H., Agarwal, D., Dalvi, B., Meena, A., Prakhar, A., Vora, T., Khot, T., Sabharwal, A., and Clark, P. DiscoveryBench: Towards Data-Driven Discovery with Large Language Models. In _ICLR_, 2025. 
*   Mitchener et al. (2025) Mitchener, L., Yiu, A., Chang, B., Bourdenx, M., Nadolski, T., Sulovari, A., Landsness, E.C., Barabási, D.L., Narayanan, S., Evans, N., Reddy, S., Foiani, M.S., Kamal, A., Shriver, L.P., Cao, F., Wassie, A.T., Laurent, J.M., Melville-Green, E., Ramos, M.C., Bou, A., Roberts, K.F., Zagorac, S., Orr, T.C., Orr, M.E., Zwezdaryk, K.J., Ghareeb, A.E., McCoy, L., Gomes, B., Ashley, E.A., Duff, K.E., Buonassisi, T., Rainforth, T., Bateman, R.J., Skarlinski, M., Rodriques, S.G., Hinks, M.M., and White, A.D. Kosmos: An AI Scientist for Autonomous Discovery. _ArXiv_, abs/2511.02824, 2025. 
*   Nathani et al. (2025) Nathani, D., Madaan, L., Roberts, N., lay Bashlykov, N., Menon, A., Moens, V., Budhiraja, A., Magka, D., Vorotilov, V., Chaurasia, G., Hupkes, D., Cabral, R.S., Shavrina, T., Foerster, J., Bachrach, Y., Wang, W.Y., and Raileanu, R. MLGym: A New Framework and Benchmark for Advancing AI Research Agents. In _COLM_, 2025. 
*   Novikov et al. (2025) Novikov, A., V˜u, N., Eisenberger, M., Dupont, E., Huang, P.-S., Wagner, A.Z., Shirobokov, S., Kozlovskii, B.M., Ruiz, F. J.R., Mehrabian, A., Kumar, M.P., See, A., Chaudhuri, S., Holland, G., Davies, A., Nowozin, S., Kohli, P., Balog, M., and Deepmind, G. Alphaevolve: A coding agent for scientific and algorithmic discovery. _ArXiv_, abs/2506.13131, 2025. 
*   Penedo et al. (2024) Penedo, G., Kydlícek, H., Allal, L.B., Lozhkov, A., Mitchell, M., Raffel, C., von Werra, L., and Wolf, T. The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. _ArXiv_, abs/2406.17557, 2024. 
*   Radford et al. (2019) Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. Language Models are Unsupervised Multitask Learners. 2019. 
*   Real et al. (2020) Real, E., Liang, C., So, D.R., and Le, Q.V. AutoML-Zero: Evolving Machine Learning Algorithms From Scratch. In _ICML_, 2020. 
*   Schmidgall et al. (2025) Schmidgall, S., Su, Y., Wang, Z., Sun, X., Wu, J., Yu, X., Liu, J., Liu, Z., and Barsoum, E. Agent Laboratory: Using LLM Agents as Research Assistants. In _Findings of EMNLP_, 2025. 
*   Shao et al. (2024) Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J.-M., Zhang, M., Li, Y.K., Wu, Y., and Guo, D. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. _ArXiv_, abs/2402.03300, 2024. 
*   Si et al. (2025a) Si, C., Hashimoto, T., and Yang, D. The Ideation-Execution Gap: Execution Outcomes of LLM-Generated versus Human Research Ideas. _ArXiv_, abs/2506.20803, 2025a. 
*   Si et al. (2025b) Si, C., Yang, D., and Hashimoto, T. Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers. In _ICLR_, 2025b. 
*   So et al. (2019) So, D.R., Liang, C., and Le, Q.V. The Evolved Transformer. In _ICML_, 2019. 
*   Starace et al. (2025) Starace, G., Jaffe, O., Sherburn, D., Aung, J., Chan, J.S., Maksin, L., Dias, R., Mays, E., Kinsella, B., Thompson, W., Heidecke, J., Glaese, A., and Patwardhan, T. PaperBench: Evaluating AI’s Ability to Replicate AI Research. In _ICML_, 2025. 
*   Tang et al. (2025) Tang, J., Xia, L., Li, Z., and Huang, C. AI-Researcher: Autonomous Scientific Innovation. In _NeurIPS_, 2025. 
*   Thinking Machines Lab (2025) Thinking Machines Lab. Announcing Tinker, 2025. 
*   Tian et al. (2024) Tian, M., Gao, L., Zhang, S.D., Chen, X., Fan, C., Guo, X., Haas, R., Ji, P., Krongchon, K., Li, Y., Liu, S., Luo, D., Ma, Y., Tong, H., Trinh, K., Tian, C., Wang, Z., Wu, B., Xiong, Y., Yin, S., Zhu, M., Lieret, K.A., Lu, Y., Liu, G., Du, Y., Tao, T., Press, O., Callan, J., Huerta, E.A., and Peng, H. SciCode: A Research Coding Benchmark Curated by Scientists. _ArXiv_, abs/2407.13168, 2024. 
*   Toledo et al. (2025) Toledo, E., Hambardzumyan, K., Josifoski, M., Hazra, R., Baldwin, N.M., Audran-Reiss, A., Kuchnik, M., Magka, D., Jiang, M., Lupidi, A.M., Lupu, A., Raileanu, R., Niu, K., Shavrina, T., Gagnon-Audet, J.-C., Shvartsman, M., Sodhani, S., Miller, A.H., Charnalia, A., Dunfield, D., Wu, C.-J., Stenetorp, P., Cancedda, N., Foerster, J.N., and Bachrach, Y. AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench. _ArXiv_, abs/2507.02554, 2025. 
*   Wang et al. (2024) Wang, Q., Downey, D., Ji, H., and Hope, T. SciMON: Scientific Inspiration Machines Optimized for Novelty. In _ACL_, 2024. 
*   Wang et al. (2025) Wang, Y., Yang, Q., Zeng, Z., Ren, L., Liu, L., Peng, B., Cheng, H., He, X., Wang, K., Gao, J., Chen, W., Wang, S., Du, S.S., and Shen, Y. Reinforcement Learning for Reasoning in Large Language Models with One Training Example. In _NeurIPS_, 2025. 
*   Wijk et al. (2024) Wijk, H., Lin, T.R., Becker, J., Jawhar, S., Parikh, N., Broadley, T., Chan, L., Chen, M., Clymer, J., Dhyani, J., Ericheva, E., Garcia, K., Goodrich, B., Jurkovic, N., Kinniment, M., Lajko, A., Nix, S., Sato, L. J.K., Saunders, W., Taran, M., West, B., and Barnes, E. RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts. _ArXiv_, abs/2411.15114, 2024. 
*   Wu et al. (2025) Wu, F., Xuan, W., Lu, X., Harchaoui, Z., and Choi, Y. The Invisible Leash: Why RLVR May Not Escape Its Origin. _ArXiv_, abs/2507.14843, 2025. 
*   Yamada et al. (2025) Yamada, Y., Lange, R.T., Lu, C., Hu, S., Lu, C., Foerster, J.N., Clune, J., and Ha, D. The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search. _ArXiv_, abs/2504.08066, 2025. 
*   Yang et al. (2024) Yang, A., Zhang, B., Hui, B., Gao, B., Yu, B., Li, C., Liu, D., Tu, J., Zhou, J., Lin, J., Lu, K., Xue, M., Lin, R., Liu, T., Ren, X., and Zhang, Z. Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement. _ArXiv_, abs/2409.12122, 2024. 
*   Yang et al. (2025a) Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D., Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, H., Tang, J., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Zhou, J., Lin, J., Dang, K., Bao, K., Yang, K., Yu, L., Deng, L.-C., Li, M., Xue, M., Li, M., Zhang, P., Wang, P., Zhu, Q., Men, R., Gao, R., Liu, S.-Q., Luo, S., Li, T., Tang, T., Yin, W., Ren, X., Wang, X., Zhang, X., Ren, X., Fan, Y., Su, Y., Zhang, Y.-C., Zhang, Y., Wan, Y., Liu, Y., Wang, Z., Cui, Z., Zhang, Z., Zhou, Z., and Qiu, Z. Qwen3 Technical Report. _ArXiv_, abs/2505.09388, 2025a. 
*   Yang et al. (2025b) Yang, S., He-Yueya, J., and Liang, P. Reinforcement Learning for Machine Learning Engineering Agents. _ArXiv_, abs/2509.01684, 2025b. 
*   Yue et al. (2025) Yue, Y., Chen, Z., Lu, R., Zhao, A., Wang, Z., Song, S., and Huang, G. Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? In _NeurIPS_, 2025. 
*   Zhang et al. (2023) Zhang, M.R., Desai, N., Bae, J., Lorraine, J., and Ba, J. Using Large Language Models for Hyperparameter Optimization. _ArXiv_, abs/2312.04528, 2023. 
*   Zheng et al. (2024) Zheng, T., Zhang, G., Shen, T., Liu, X., Lin, B.Y., Fu, J., Chen, W., and Yue, X. OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement. In _Findings of ACL_, 2024. 
*   Zhu et al. (2025) Zhu, M., Weng, Y., Yang, L., and Zhang, Y. DeepReview: Improving LLM-based Paper Review with Human-like Deep Thinking Process. In _ACL_, 2025. 
*   Zoph & Le (2017) Zoph, B. and Le, Q.V. Neural Architecture Search with Reinforcement Learning. In _ICLR_, 2017. 
*   Zoph et al. (2017) Zoph, B., Vasudevan, V., Shlens, J., and Le, Q.V. Learning Transferable Architectures for Scalable Image Recognition. In _CVPR_, 2017. 

Appendix A Appendix
-------------------

### A.1 Other RL Attempts

We present several attempts to improve our RL from the execution reward recipe.

#### Attempt 1: Dynamic Prompt

At each epoch (except the first epoch), we randomly sample different executed idea trajectories from the previous epoch and append them to the idea sampling prompt when sampling new rollouts. This merges in-context learning with RL and adds diversity to the idea sampling process. We present the experiment results on the GRPO environment in Figure[8](https://arxiv.org/html/2601.14525v1#A1.F8 "Figure 8 ‣ Attempt 1: Dynamic Prompt ‣ A.1 Other RL Attempts ‣ Appendix A Appendix ‣ Towards Execution-Grounded Automated AI Research"). We did not see significant improvement in early epochs and thus early stopped.

![Image 18: Refer to caption](https://arxiv.org/html/2601.14525v1/x18.png)

Figure 8: RL with dynamic prompt by appending trajectories from previous epochs. Results are on the GRPO environment.

#### Attempt 2: Length Reward

Since we noted a rapid thinking length decrease in our main RL experiment, we tried a simple fix by adding a weighted length reward that counts the number of tokens in the entire rollout sequence, including the thinking trace and the idea. We cap the length reward to a maximum of 0.3 0.3 to avoid it dominating the accuracy reward. We present the experiment results on the GRPO environment in Figure[9](https://arxiv.org/html/2601.14525v1#A1.F9 "Figure 9 ‣ Attempt 2: Length Reward ‣ A.1 Other RL Attempts ‣ Appendix A Appendix ‣ Towards Execution-Grounded Automated AI Research"). As shown on the right panel, the thinking length no longer decreases after adding the length reward to the total reward, but the total training reward isn’t going up as shown on the left panel.

![Image 19: Refer to caption](https://arxiv.org/html/2601.14525v1/x19.png)

![Image 20: Refer to caption](https://arxiv.org/html/2601.14525v1/x20.png)

Figure 9: Average and max reward throughout RL training when adding the length reward (left), as well as the progression of the thinking and idea lengths (right).

#### Attempt 3: Diversity Reward

We also tried adding a diversity reward in addition to the effectiveness reward. Specifically, when computing the reward for each rollout, we compute its token-level Jaccard similarity with ideas from the previous epoch and add the negative similarity as a penalty to the total reward to discourage repeating ideas that have already been generated before. In fact, this is similar to one of the post-training ideas proposed by Claude-4.5-Sonnet (see example 4 in Appendix[A.3](https://arxiv.org/html/2601.14525v1#A1.SS3 "A.3 Code Execution Examples ‣ Appendix A Appendix ‣ Towards Execution-Grounded Automated AI Research")). We show the training curves on the GRPO environment in Figure[10](https://arxiv.org/html/2601.14525v1#A1.F10 "Figure 10 ‣ Attempt 3: Diversity Reward ‣ A.1 Other RL Attempts ‣ Appendix A Appendix ‣ Towards Execution-Grounded Automated AI Research"). The model maintains a consistent idea similarity (right plot), suggesting sustained exploration. The effectiveness reward is generally showing an upward trend (left plot), but not markedly better than the main RL run with just the effectiveness reward (first sub-plot in Figure[5](https://arxiv.org/html/2601.14525v1#S5.F5 "Figure 5 ‣ 5 Reinforcement Learning from Execution Reward ‣ Towards Execution-Grounded Automated AI Research")).

![Image 21: Refer to caption](https://arxiv.org/html/2601.14525v1/figures/RL_diversity_reward_acc.png)

![Image 22: Refer to caption](https://arxiv.org/html/2601.14525v1/figures/RL_diversity_similarity.png)

Figure 10: Effectiveness reward (left) and average idea similarity to previous epoch (right) when doing RL with diversity reward.

### A.2 Additional Idea Examples

We provide several additional example ideas generated by Claude-4.5-Opus (Table[4](https://arxiv.org/html/2601.14525v1#A1.T4 "Table 4 ‣ A.2 Additional Idea Examples ‣ Appendix A Appendix ‣ Towards Execution-Grounded Automated AI Research")) and Claude-4.5-Sonnet (Table[5](https://arxiv.org/html/2601.14525v1#A1.T5 "Table 5 ‣ A.2 Additional Idea Examples ‣ Appendix A Appendix ‣ Towards Execution-Grounded Automated AI Research")) on the GRPO environment, including ideas with failed code execution. In most cases, code execution errors happen when the idea involves complicated changes or installing and importing external packages not supported in our execution environment. Future work should explore improvement to the execution agent so that more complicated types of ideas (e.g., training additional auxiliary models or system-level optimizations) can be implemented correctly.

Successful Execution Failed Execution
[Experiment] Sequence Position Weighted Trust Region: Apply tighter sigmoid bounds to earlier tokens in the sequence (where errors compound) and looser bounds to later tokens. Weight: position_weight = 1 - 0.3 * (position / seq_len), effective_deviation = 0.25 + 0.2 * position_weight. This accounts for the sequential nature of autoregressive generation.[Code Changes] Modify grpo.py: Initialize current_cliprange = 0.2, ema_clip_fraction = 0.15. Standard momentum clip updates. Modify compute_grpo_clip_loss in grpo_utils.py: After computing ratio on line 91 (shape: batch_size x seq_len): batch_size, seq_len = ratio.shape, positions = torch.arange(seq_len, device=ratio.device).float().unsqueeze(0).expand(batch_size, -1), position_weight = 1.0 - 0.3 * (positions / (seq_len - 1 + 1e-6)), effective_deviation = 0.25 + 0.2 * position_weight. Apply position-aware sigmoid: centered_ratio = ratio - 1.0, bounded_ratio = 1.0 + (2.0 * torch.sigmoid(centered_ratio) - 1.0) * effective_deviation. Use: surr1 = bounded_ratio * advantages, surr2 = torch.clamp(bounded_ratio, 1 - cliprange, 1 + cliprange) * advantages, loss = -torch.min(surr1, surr2). Add metadata: metadata["mean_effective_deviation"] = effective_deviation.mean().item(), metadata["early_deviation"] = effective_deviation[:, :seq_len//4].mean().item(), metadata["late_deviation"] = effective_deviation[:, -seq_len//4:].mean().item().Validation Accuracy: 59.8[Experiment] Hierarchical Position-Group Trust Region: Apply trust region at two hierarchical levels – group level (shared within each response group) and position level (varies along sequence). Groups with high internal reward variance get tighter group-level bounds. Within groups, positions follow the proven decay pattern. This captures both cross-sample and within-sample structure. Formula: group_dev = 0.4 - 0.15 * tanh(group_reward_var / 0.3), position_factor = 1 - 0.2 * rel_pos, effective_dev = group_dev * position_factor.[Code Changes] Modify grpo.py: Initialize current_cliprange = 0.2, ema_clip_fraction = 0.15. Standard momentum clip updates. Pass group_size to function. Modify compute_grpo_clip_loss in grpo_utils.py: Add parameter group_size=8. After computing ratio: batch_size, seq_len = ratio.shape, n_groups = batch_size // group_size. Compute group reward variance from advantages as proxy: adv_grouped = advantages.view(n_groups, group_size, -1), group_adv_var = adv_grouped.var(dim=1, keepdim=True), group_adv_var_expanded = group_adv_var.expand(-1, group_size, -1).reshape(advantages.shape). Group-level deviation: group_deviation = 0.4 - 0.15 * torch.tanh(group_adv_var_expanded / 0.3). Position factor: positions = torch.arange(seq_len, device=ratio.device).float().unsqueeze(0).expand(batch_size, -1), rel_pos = positions / (seq_len - 1 + 1e-6), position_factor = 1.0 - 0.2 * rel_pos. Hierarchical deviation: effective_deviation = group_deviation * position_factor, effective_deviation = torch.clamp(effective_deviation, 0.15, 0.45). Apply: centered_ratio = ratio - 1.0, bounded_ratio = 1.0 + (2.0 * torch.sigmoid(centered_ratio) - 1.0) * effective_deviation. Use: surr1 = bounded_ratio * advantages, surr2 = torch.clamp(bounded_ratio, 1 - cliprange, 1 + cliprange) * advantages, loss = -torch.min(surr1, surr2). Add metadata["mean_group_var"] = group_adv_var.mean().item(), metadata["mean_effective_deviation"] = effective_deviation.mean().item(). Log to wandb.

Table 4: Additional examples on the GRPO environment. Ideas are generated by Claude-4.5-Opus during evolutionary search.

Successful Execution Failed Execution
[Experiment] Create a mathematical step-complexity aware reward shaping where responses with more mathematical reasoning steps receive slightly higher base rewards (1.05x for 3+ steps, 1.1x for 5+ steps) when correct, encouraging thorough mathematical exposition without changing the core binary reward structure.[Code Changes] Modify r1_zero_reward_fn_train in drgrpo_grader.py to count reasoning steps by detecting mathematical transitions ("therefore", "thus", "so", "=", "=>"). When answer is correct, apply step-based multiplier: step_multiplier = 1.0 + 0.05 * min(2, max(0, num_steps - 2)) to get multipliers [1.0, 1.05, 1.1]. Return {"format_reward": 1.0, "answer_reward": answer_reward, "reward": base_reward * step_multiplier}. Set --learning_rate 3e-5 and --loss_type reinforce_with_baseline.Validation Accuracy: 65.6[Experiment] Add experience replay by maintaining a buffer of the top 20% highest-reward rollouts from previous epochs and mixing them (25% replay, 75% new) with current rollouts during training, combined with 3e-5 learning rate and reinforce_with_baseline for improved sample efficiency.[Code Changes] Modify train_loop in grpo.py to maintain replay_buffer storing high-reward (>0.8) rollouts from previous epochs. Each epoch, sample 25% of training data from replay buffer and 75% from new rollouts. Update buffer by adding top 20% of current epoch’s rollouts and removing oldest entries when buffer exceeds 1000 samples. Set --learning_rate 3e-5 and --loss_type reinforce_with_baseline.Validation Accuracy: 39.4[Experiment] Implement response diversity rewards within groups where responses to the same prompt receive bonus rewards (0.05--0.15) for being dissimilar to other responses in their group, encouraging exploration of different solution paths while maintaining the proven group_size=8 and 3e-5 learning rate combination.[Code Changes] Modify compute_group_normalized_rewards in grpo_utils.py to compute pairwise similarity between responses in each group using token-level Jaccard similarity. Add diversity bonus: diversity_reward = 0.15 * (1 - max_similarity_in_group) to each response’s reward before advantage computation. Reshape responses into groups, compute similarities, and add bonuses before advantage normalization. Set --learning_rate 3e-5, --loss_type reinforce_with_baseline, --group_size 8.Validation Accuracy: 19.2[Experiment] Implement temporal difference advantage estimation where advantages incorporate not just current rewards but also predicted future rewards using a learned value function, combined with the proven 3e-5 learning rate and reinforce_with_baseline for more accurate credit assignment.[Code Changes] Add a value head to the policy model in grpo.py that predicts expected future rewards. Compute TD advantages as advantages = rewards + gamma * next_values - current_values with gamma=0.99. Train the value function with MSE loss on observed returns. Modify compute_group_normalized_rewards to use TD advantages instead of basic reward differences. Set --learning_rate 3e-5 and --loss_type reinforce_with_baseline.[Experiment] Ensemble Decision Training with Voting Consensus: Train the model using ensemble-style decision making where each rollout generates multiple candidate responses, and the final training signal is based on majority voting among responses. This encourages the model to develop more robust and consistent reasoning patterns while maintaining diversity in solution approaches.[Code Changes] Modify sample_rollout in sample.py to generate 3 responses per prompt instead of 1, using different random seeds. Implement voting consensus in r1_zero_reward_fn_train: if 2+ responses are correct, apply +0.15 consensus bonus; if responses disagree, apply -0.05 uncertainty penalty. In train_loop in grpo.py, select the highest-voted response for training while using consensus information to adjust learning rate: consensus_lr = 3e-5 * (0.9 + 0.2 * consensus_rate). Set group_size=6, --loss_type reinforce_with_baseline.[Experiment] Implement hierarchical advantage estimation where advantages are computed at both token-level and sequence-level, with token-level advantages weighted by their position importance (higher weights for mathematical expressions and final answers), combined with the successful 3e-5 learning rate and reinforce_with_baseline.[Code Changes] Modify grpo_microbatch_train_step in grpo_utils.py to create position importance weights that assign 2.0x weight to tokens containing mathematical symbols (\frac, +, -, *, =) and 1.5x weight to answer sections. Compute both sequence-level advantages (current) and token-level advantages, then combine as final_advantages = 0.6 * sequence_advantages + 0.4 * token_advantages. Set --learning_rate 3e-5 and --loss_type reinforce_with_baseline.

Table 5: Additional examples on the GRPO environment. Ideas are generated by Claude-4.5-Sonnet during evolutionary search.

We also present the top-performing ideas from Claude-4.5-Opus, Claude-4.5-Sonnet, and GPT-5 on the nanoGPT environment below:

While the best-performing ideas on nanoGPT tend to be heavily optimized with extensive hyper-parameter tuning mixed with various architecture tweaks, we also pick a few more “atomic” algorithmic ideas that are successfully executed.

Examples from Claude-4.5-Opus

*   •Head-Wise Attention Output Scaling Add learnable per-head scaling factors to attention, allowing different heads to contribute with different magnitudes to the output. 

Validation Loss: 3.2386 
*   •Learned Residual Connection Weights Add learnable scalar weights for each residual connection that are initialized to 1.0, allowing the model to learn optimal residual scaling during training. 

Validation Loss: 3.2517 
*   •Mixture of Embeddings with Position Learn to mix token embeddings and position embeddings with a content-dependent weight, allowing the model to dynamically balance positional vs semantic information per token. 

Validation Loss: 3.2497 
*   •Shared Input-Output Embedding with Learned Asymmetry Keep weight tying but add a small learned transformation on the output side, providing the benefits of weight tying while allowing output-specific adaptation. 

Validation Loss: 3.2499 
*   •Gated Final Normalization Replace the final RMSNorm before lm_head with a gated version where a learned gate controls how much normalization is applied vs passing the raw representation. 

Validation Loss: 3.2503 
*   •Position-Aware MLP Gating Gate the MLP output based on position information, allowing the model to learn position-dependent processing depth. 

Validation Loss: 3.2506 
*   •Learned Residual Connection Weights Add learnable scalar weights for each residual connection that are initialized to 1.0, allowing the model to learn optimal residual scaling during training. 

Validation Loss: 3.2517 
*   •Grouped Token Embeddings Group the vocabulary into clusters and add a learned embedding per cluster on top of token embeddings, providing hierarchical vocabulary structure. 

Validation Loss: 3.2521 

Lastly, we present several interesting ideas on the nanoGPT environment that didn’t get successfully executed. These examples are generated by Claude-4.5-Opus.

*   •Soft Layer Repetition Allow the model to softly repeat computation through layers by adding a learned gate that mixes the current layer’s input back into its output, simulating variable depth. 
*   •Causal Context Compression Before each attention layer, apply a learned compression that mixes local context (previous 2-3 tokens) into the current representation, providing implicit local context without convolutions. 
*   •Attention Head Specialization via Orthogonal Loss Add a soft penalty that encourages different attention heads to attend to different patterns by penalizing similarity between head outputs. 
*   •Skip Connections with Learned Residual Weights Combine skip connections with learned residual weights. The skip connections provide alternative gradient paths while learned weights allow adaptive scaling. 
*   •Token Difficulty-Aware Loss Weighting Weight the loss contribution of each token based on the model’s uncertainty (entropy) at that position, focusing learning on difficult tokens while not over-optimizing easy ones. 

### A.3 Code Execution Examples

We present a few ideas with their full code execution to demonstrate the full end-to-end trajectories. All examples below are from Claude-4.5-Sonnet on the GRPO environment. For each example, we first present the natural language idea, followed by the code implementation generated by Claude-4.5-Sonnet.

Example 1

[Experiment] Create mathematical working memory simulation by maintaining a context buffer of mathematical facts, definitions, and intermediate results during problem solving. This buffer gets updated as the model works through problems and provides additional context for subsequent mathematical steps, simulating how humans maintain mathematical working memory during complex calculations.

[Code Changes] Modify the prompt template in sample.py to include a “Mathematical Context” section that accumulates relevant mathematical facts during problem solving. Implement context buffer management in train_loop in grpo.py that maintains recent mathematical definitions, formulas, and intermediate calculations across problems in the same batch. Update the buffer with key mathematical expressions extracted from each response. Use enhanced prompts that include relevant context from the buffer. Set --learning_rate 2.5e-5 and --loss_type reinforce_with_baseline.

Validation Accuracy: 58.0

```
Example 1 code diff

Example 2[Experiment] Implement mathematical solution robustness training through
systematic perturbation testing that teaches the model to solve mathematically
equivalent problems with varied presentations, notation styles, and problem
phrasings. Combine this with proven reward shaping by creating robustness-aware
rewards that encourage mathematical understanding that generalizes across problem
variations.
[Code Changes] Create problem perturbation system in sample.py
that generates equivalent mathematical problems with varied notation, different
variable names, alternative problem phrasing, and equivalent mathematical
formulations. Track solution consistency across perturbations. Enhance
r1_zero_reward_fn_train in drgrpo_grader.py to reward
robustness: robustness_bonus = 0.12 * consistency_score_across_perturbations
where consistency measures solution method stability across equivalent problems.
Apply generalization bonuses: +0.08 for solving problems with unfamiliar
notation, +0.06 for handling varied problem presentations. Modify
train_loop in grpo.py to include perturbation-based training
batches and track robustness improvement. Use robustness-aware learning rate
with cosine annealing from 3.4e-5 to 1.8e-5. Implement
perturbation-complexity gradient clipping. Set --loss_type reinforce_with_baseline.
Validation Accuracy: 56.2 

Example 2 code diff

Example 3[Experiment] Implement token-level reward attribution by using
attention weights to identify which input tokens contributed most to correct
answers, then amplifying the gradient updates for those tokens during policy
gradient training.
[Code Changes] Modify get_response_log_probs in
utils.py to also return attention weights from the last layer. In
grpo_microbatch_train_step, compute token importance scores by
averaging attention weights across heads, then multiply the policy gradient loss
by (1 + importance_score) for tokens with high attention to
mathematical content.
Validation Accuracy: 45.2 

Example 3 code diff

Example 4[Experiment] Implement response diversity rewards within groups where responses to the same prompt receive bonus rewards (0.05-0.15) for being dissimilar to other responses in their group, encouraging exploration of different solution paths while maintaining the proven group_size=8 and 3e-5 learning rate combination.
[Code Changes] Modify compute_group_normalized_rewards in grpo_utils.py to compute pairwise similarity between responses in each group using token-level Jaccard similarity. Add diversity bonus: diversity_reward = 0.15 * (1 - max_similarity_in_group) to each response’s reward before advantage computation. Reshape responses into groups, compute similarities, and add bonuses before advantage normalization. Set --learning_rate 3e-5, --loss_type reinforce_with_baseline, --group_size 8.
Validation Accuracy: 19.2 

Example 4 code diff
```

