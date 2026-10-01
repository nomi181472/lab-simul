Title: Enhancing Visual Grounding for GUI Agents via Self-Evolutionary Reinforcement Learning

URL Source: https://arxiv.org/html/2505.12370

Markdown Content:
Xinbin Yuan 1,2 1 1 1 This work was done during Xinbin Yuan’s internship at vivo. Jian Zhang 2 3 3 3 Project lead. Kaixin Li 2 Zhuoxuan Cai 2 Lujian Yao 2 Jie Chen 2

Enguang Wang 2 Qibin Hou 1 2 2 2 Corresponding authors.Jinwei Chen 2 Peng-Tao Jiang 2 Bo Li 2 2 2 footnotemark: 2
1 VCIP, School of Computer Science, NKU 2 vivo Mobile Communication Co., Ltd 

yxb@mail.nankai.edu.cn, houqb@nankai.edu.cn, libra@vivo.com

Project page: [https://github.com/YXB-NKU/SE-GUI](https://github.com/YXB-NKU/SE-GUI)

###### Abstract

Graphical User Interface (GUI) agents have made substantial strides in understanding and executing user instructions across diverse platforms. Yet, grounding these instructions to precise interface elements remains challenging—especially in complex, high-resolution, professional environments. Traditional supervised fine-tuning (SFT) methods often require large volumes of diverse data and exhibit weak generalization. To overcome these limitations, we introduce a reinforcement learning (RL)-based framework that incorporates three core strategies: (1) seed data curation to ensure high-quality training samples, (2) a dense policy gradient that provides continuous feedback based on prediction accuracy, and (3) a self-evolutionary reinforcement finetuning mechanism that iteratively refines the model using attention maps. With only 3k training samples, our 7B-parameter model achieves state-of-the-art results among similarly sized models on three grounding benchmarks. Notably, it attains 47.3% accuracy on the ScreenSpot-Pro dataset—outperforming much larger models, such as UI-TARS-72B, by a margin of 24.2%. These findings underscore the effectiveness of RL-based approaches in enhancing GUI agent performance, particularly in high-resolution, complex environments.

1 Introduction
--------------

Graphical User Interface (GUI) agents have become increasingly capable of executing user commands across diverse platforms[[1](https://arxiv.org/html/2505.12370v2#bib.bib1); [2](https://arxiv.org/html/2505.12370v2#bib.bib2)]. Yet, a core challenge remains: accurately grounding natural language instructions to the correct elements on the interface[[3](https://arxiv.org/html/2505.12370v2#bib.bib3); [4](https://arxiv.org/html/2505.12370v2#bib.bib4); [5](https://arxiv.org/html/2505.12370v2#bib.bib5)]. While Supervised Fine-Tuning (SFT) has been proven to be effective in simple scenarios[[6](https://arxiv.org/html/2505.12370v2#bib.bib6); [7](https://arxiv.org/html/2505.12370v2#bib.bib7); [8](https://arxiv.org/html/2505.12370v2#bib.bib8)], it faces two major limitations: requiring large volumes of diverse data and exhibiting weak generalization in complex and high-resolution professional settings[[9](https://arxiv.org/html/2505.12370v2#bib.bib9); [10](https://arxiv.org/html/2505.12370v2#bib.bib10)].

To address these issues, reinforcement learning (RL) offers a promising alternative. Unlike SFT, RL enables models to learn from structured and incremental feedback via reward functions, guiding them toward more precise grounding. Among RL-based methods[[11](https://arxiv.org/html/2505.12370v2#bib.bib11); [12](https://arxiv.org/html/2505.12370v2#bib.bib12); [13](https://arxiv.org/html/2505.12370v2#bib.bib13)], Group Relative Policy Optimization (GRPO)[[13](https://arxiv.org/html/2505.12370v2#bib.bib13)] stands out for its efficiency. It replaces heavy value models with a simple rule-based reward system, reducing the computational burden. However, typical approaches[[9](https://arxiv.org/html/2505.12370v2#bib.bib9); [10](https://arxiv.org/html/2505.12370v2#bib.bib10)] rely on binary (0/1) rewards, which are often sparse. Such sparsity impedes early-stage training, as incorrect predictions commonly receive identical zero rewards, resulting in uniform advantage estimates and limited gradient information for effective optimization.

Furthermore, recent studies[[14](https://arxiv.org/html/2505.12370v2#bib.bib14); [15](https://arxiv.org/html/2505.12370v2#bib.bib15); [16](https://arxiv.org/html/2505.12370v2#bib.bib16)] also emphasize the importance of training data quality for RL-based methods, which is not explored for GUI tasks. However, grounding datasets are typically collected through automated pipelines that extract data from noisy UI accessibility trees or raw HTML, often without thorough verification[[7](https://arxiv.org/html/2505.12370v2#bib.bib7); [6](https://arxiv.org/html/2505.12370v2#bib.bib6); [8](https://arxiv.org/html/2505.12370v2#bib.bib8); [17](https://arxiv.org/html/2505.12370v2#bib.bib17)]. This process introduces significant noise, including low-quality instructions (e.g., generic labels like <PushButton>) and bounding boxes corresponding to elements that appear in the DOM but are not visually rendered in the UI. This raises a critical question: _How can we assess the quality of each training sample?_

To investigate the above problem, we analyze model failures by examining layer attention across the transformer architecture. We found two common patterns: In some cases, the model roughly identifies the target region, but fails to localize it precisely; in others, it completely overlooks the relevant area. These errors often stem from weak alignment between the instruction and the ground truth location.

Motivated by these insights, we propose a Self-Evolutionary reinforcement fine-tuning algorithm that leverages layer attention from a trained model to guide further learning. As grounding accuracy improves, so does the quality of attention, enabling the model to iteratively supervise its own training. At each stage, the best-performing model generates attention maps that inform the next iteration, continuing until performance converges.

![Image 1: Refer to caption](https://arxiv.org/html/2505.12370v2/x1.png)

Figure 1: Performance comparison of various GUI agents on both the grounding benchmark (ScreenSpot[[18](https://arxiv.org/html/2505.12370v2#bib.bib18)], ScreenSpot-v2[[7](https://arxiv.org/html/2505.12370v2#bib.bib7)], ScreenSpot-Pro[[4](https://arxiv.org/html/2505.12370v2#bib.bib4)]) and the agent benchmark (Android_Control-High[[19](https://arxiv.org/html/2505.12370v2#bib.bib19)], Android_Control-Low[[19](https://arxiv.org/html/2505.12370v2#bib.bib19)], OmniACT[[20](https://arxiv.org/html/2505.12370v2#bib.bib20)]). Our model, SE-GUI marked with a star, demonstrates state-of-the-art performance against models with larger parameter counts.

We summarize our contributions as follows:

1.   1.Seed Data Curation. We curate a 3,018-sample dataset by filtering out vague, inaccurate, or overly simple tasks from a larger candidate pool. This ensures linguistic consistency and balanced task complexity, promoting better generalization and stable performance across scenarios. 
2.   2.Group Relative Policy Optimization with Dense Point Reward. To combat sparse rewards, we designed a continuous reward mechanism that evaluates the proximity between predictions and ground truth. This provides smoother feedback, enabling the model to learn from near-misses and gradually refine its grounding behavior. 
3.   3.Self-Evolutionary Reinforcement Fine-Tuning. We implement an iterative learning loop, where attention maps serve as intermediate supervision signals. These maps highlight which visual tokens the model attends to for each instruction, helping align its focus with relevant interface elements over time. 

We evaluate our SE-GUI on six diverse grounding and agent benchmarks across desktop, mobile, and web environments. As illustrated in Fig.[1](https://arxiv.org/html/2505.12370v2#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Enhancing Visual Grounding for GUI Agents via Self-Evolutionary Reinforcement Learning"), SE-GUI achieves state-of-the-art performance, most notably attaining 47.3% accuracy on the challenging ScreenSpot-Pro benchmark, surpassing the previous best (UI-TARS-72B) by 24.2% at 7B parameter scale with only 3k training samples. These results demonstrate the effectiveness of our RL-based approach for grounding in complex GUI environments.

2 Related Work
--------------

### 2.1 GUI Agents

GUI agents, as a specialized class of autonomous AI systems, have rapidly evolved with the advent of large foundation models such as LLMs and vision-language models (VLMs)[[21](https://arxiv.org/html/2505.12370v2#bib.bib21); [22](https://arxiv.org/html/2505.12370v2#bib.bib22); [23](https://arxiv.org/html/2505.12370v2#bib.bib23)], enabling them to interact with graphical user interfaces in increasingly sophisticated ways. Unlike traditional programmatic agents that rely on API calls or internal code access, GUI agents[[24](https://arxiv.org/html/2505.12370v2#bib.bib24); [25](https://arxiv.org/html/2505.12370v2#bib.bib25); [26](https://arxiv.org/html/2505.12370v2#bib.bib26); [27](https://arxiv.org/html/2505.12370v2#bib.bib27); [28](https://arxiv.org/html/2505.12370v2#bib.bib28)] simulate human-like interactions through mouse clicks, keyboard inputs, and visual perception, offering greater flexibility in automating tasks across diverse platforms. Early approaches[[29](https://arxiv.org/html/2505.12370v2#bib.bib29)] focused on structured representations like HTML code, but recent advancements[[6](https://arxiv.org/html/2505.12370v2#bib.bib6); [8](https://arxiv.org/html/2505.12370v2#bib.bib8); [7](https://arxiv.org/html/2505.12370v2#bib.bib7)] demonstrate superior performance when agents directly process visual forms of GUIs, leveraging high-resolution encoders and unified vision-language interfaces . Systems like AppAgent[[24](https://arxiv.org/html/2505.12370v2#bib.bib24); [25](https://arxiv.org/html/2505.12370v2#bib.bib25)], and CogAgent[[26](https://arxiv.org/html/2505.12370v2#bib.bib26)] have pioneered this field by enhancing GUI comprehension and interaction precision, while newer frameworks such as UI-TARS[[30](https://arxiv.org/html/2505.12370v2#bib.bib30)] and AgentS2[[3](https://arxiv.org/html/2505.12370v2#bib.bib3)] introduce modular architectures and generalist-specialist designs for improved task planning and cross-platform generalization. Despite these strides, challenges remain, particularly in grounding accuracy and data efficiency, as highlighted by methods like UGround[[6](https://arxiv.org/html/2505.12370v2#bib.bib6)] and OS-Atlas[[7](https://arxiv.org/html/2505.12370v2#bib.bib7)]. However, the reliance on supervised fine-tuning (SFT) poses limitations in scalability and adaptability, motivating the exploration of advanced learning paradigms to overcome the need for vast labeled datasets and improve generalization to unseen interfaces.

### 2.2 Reinforcement Learning

Reinforcement Learning (RL) has emerged as a transformative approach in the domain of GUI agents[[9](https://arxiv.org/html/2505.12370v2#bib.bib9); [10](https://arxiv.org/html/2505.12370v2#bib.bib10)], leveraging rule-based reward functions to guide model behavior and enhance performance. Frameworks like DeepSeek-R1[[23](https://arxiv.org/html/2505.12370v2#bib.bib23)] have demonstrated the efficacy of RL in tasks such as mathematical reasoning and code generation, with subsequent studies[[31](https://arxiv.org/html/2505.12370v2#bib.bib31); [32](https://arxiv.org/html/2505.12370v2#bib.bib32)] extending its application to multimodal models for visual tasks, including image classification and object detection. as to the application of Reinforcement Fine-Tuning (RFT) in gui tasks. UI-R1[[9](https://arxiv.org/html/2505.12370v2#bib.bib9)] and GUI-R1[[10](https://arxiv.org/html/2505.12370v2#bib.bib10)] represents a pioneering effort in this direction, showcasing the potential of RFT to improve action prediction accuracy and grounding in GUI environments, even with limited data. These developments highlight RL’s adaptability and scalability, positioning it as a promising paradigm for future innovations in intelligent GUI agent. However, these approaches typically rely on binary (0/1) rewards, which are often sparse. Such sparsity impedes early-stage training, as incorrect predictions commonly receive identical zero rewards, resulting in uniform advantage estimates and limited gradient information for effective optimization. So, how to design better RL algorithms remains a question worth exploring.

3 Method
--------

![Image 2: Refer to caption](https://arxiv.org/html/2505.12370v2/x2.png)

Figure 2: Overview of the SE-GUI Framework. Given high-level instructions and the corresponding screenshots, SE-GUI generates multiple candidate responses and is optimized via GRPO using verifiable rewards as feedback signals. The updated model is then leveraged to identify high-quality samples in the training set, which yields cleaner learning signal for subsequent iterations. This iterative process enables SE-GUI to progressively improve itself by refining its predictions and training data.

In this section, we describe our proposed framework SE-GUI. It comprises three key components as shown in Fig.[2](https://arxiv.org/html/2505.12370v2#S3.F2 "Figure 2 ‣ 3 Method ‣ Enhancing Visual Grounding for GUI Agents via Self-Evolutionary Reinforcement Learning"). First, we introduce _a data filtering strategy_ aimed at constructing a clean training set, addressing the need for high-quality data in RL. Second, to mitigate the problem of sparse binary reward signals, we introduce _a dense point reward mechanism_ to guide the model more effectively towards the target even when initial predictions are imprecise. Finally, we propose _a self-evolutionary training strategy_ that iteratively refines the model and data with an attention-based filtering method.

### 3.1 Seed Data Curation

Our work commences with the collection of diverse open-source grounding datasets including ShowUI[[33](https://arxiv.org/html/2505.12370v2#bib.bib33)], UGround[[6](https://arxiv.org/html/2505.12370v2#bib.bib6)] and AriaUI[[17](https://arxiv.org/html/2505.12370v2#bib.bib17)]. This leads to approximately 300k examples of grounding data encompassing desktop, web, and mobile platforms. These datasets are primarily created through automated pipelines, which often introduce noise, such as inaccurate or off-screen bounding boxes. To address these issues and ensure high training quality, we implemented a three-fold filtering strategy to construct a cleaner and more reliable seed dataset:

Instruction quality. We use regular expressions to remove low-quality entries such as raw python object names. We then employ a multi-modal large language model to score the remaining samples(The prompt is provided in appendix), evaluating the clarity of the instruction, its alignment with the UI elements, and potential ambiguity.

Bounding box accuracy. Many original annotations are misaligned or include irrelevant components. To correct this, we train a bounding box quality scorer based on the Qwen2.5-VL-7B model. This scorer is fine-tuned on a curated set of high-quality bounding boxes from our collected 100k web pages as positive examples and randomly positioned ones as negatives, allowing it to effectively filter out inaccurate annotations.

Task difficulty. We utilize the Qwen2.5-VL-7B base model to conduct zero-shot testing on each data point, generating 8 output results for each instruction. If all results are correct, the data point is considered too simple and filtered out.

After applying these three filtering steps, we curated a final set of 3,018 high-quality samples, which we refer to as SE-GUI-3k.

### 3.2 Group Relative Policy Optimization with Dense Point Reward

In our RL training process, the model generates a set of N 𝑁 N italic_N potential position predictions {o 1,o 2,…,o N}subscript 𝑜 1 subscript 𝑜 2…subscript 𝑜 𝑁\{o_{1},o_{2},\dots,o_{N}\}{ italic_o start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_o start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_o start_POSTSUBSCRIPT italic_N end_POSTSUBSCRIPT }. Each response is evaluated by format reward and point reward {r 1,r 2,…,r N}subscript 𝑟 1 subscript 𝑟 2…subscript 𝑟 𝑁\{r_{1},r_{2},\dots,r_{N}\}{ italic_r start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_r start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_r start_POSTSUBSCRIPT italic_N end_POSTSUBSCRIPT }. These rewards are then normalized to calculate the relative advantage of each response. The relative quality A i subscript 𝐴 𝑖 A_{i}italic_A start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT of the i 𝑖 i italic_i-th response is computed as

A i=r i−Mean⁢({r 1,r 2,…,r N})Std⁢({r 1,r 2,…,r N}),subscript 𝐴 𝑖 subscript 𝑟 𝑖 Mean subscript 𝑟 1 subscript 𝑟 2…subscript 𝑟 𝑁 Std subscript 𝑟 1 subscript 𝑟 2…subscript 𝑟 𝑁 A_{i}=\frac{r_{i}-\mathrm{Mean}(\{r_{1},r_{2},\dots,r_{N}\})}{\mathrm{Std}(\{r% _{1},r_{2},\dots,r_{N}\})},italic_A start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = divide start_ARG italic_r start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - roman_Mean ( { italic_r start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_r start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_r start_POSTSUBSCRIPT italic_N end_POSTSUBSCRIPT } ) end_ARG start_ARG roman_Std ( { italic_r start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_r start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_r start_POSTSUBSCRIPT italic_N end_POSTSUBSCRIPT } ) end_ARG ,(1)

where Mean Mean\mathrm{Mean}roman_Mean and Std Std\mathrm{Std}roman_Std are the mean and standard deviation of the rewards. This normalization step ensures that responses are compared within the context of the group, allowing model to better capture nuanced differences between candidates. Policy updates are further constrained by minimizing the KL divergence between the updated and reference models, ensuring stable RL learning.

However, during early-stage training, incorrect predictions are common. If we rely solely on binary rewards (e.g., success or failure), the resulting advantage estimates tend to be uniform, providing limited gradient signals for effective learning. To address the issue of sparse reward feedback, we introduce a dense point reward mechanism. This reward R p subscript 𝑅 p R_{\text{p}}italic_R start_POSTSUBSCRIPT p end_POSTSUBSCRIPT is calculated by comparing the predicted click point (x,y)𝑥 𝑦(x,y)( italic_x , italic_y ) with the ground truth bounding box g⁢t bbox=[x 1,y 1,x 2,y 2]𝑔 superscript 𝑡 bbox subscript 𝑥 1 subscript 𝑦 1 subscript 𝑥 2 subscript 𝑦 2 gt^{\text{bbox}}=[x_{1},y_{1},x_{2},y_{2}]italic_g italic_t start_POSTSUPERSCRIPT bbox end_POSTSUPERSCRIPT = [ italic_x start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_y start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_x start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , italic_y start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT ]. The calculation formula is as follows:

d=(x W−x 1+x 2 2⋅W)2+(y H−y 1+y 2 2⋅H)2,𝑑 superscript 𝑥 𝑊 subscript 𝑥 1 subscript 𝑥 2⋅2 𝑊 2 superscript 𝑦 𝐻 subscript 𝑦 1 subscript 𝑦 2⋅2 𝐻 2 d=\sqrt{\left(\frac{x}{\text{$W$}}-\frac{x_{1}+x_{\text{2}}}{2\cdot\text{$W$}}% \right)^{2}+\left(\frac{y}{\text{$H$}}-\frac{y_{\text{1}}+y_{\text{2}}}{2\cdot% \text{$H$}}\right)^{2}},italic_d = square-root start_ARG ( divide start_ARG italic_x end_ARG start_ARG italic_W end_ARG - divide start_ARG italic_x start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT + italic_x start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT end_ARG start_ARG 2 ⋅ italic_W end_ARG ) start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT + ( divide start_ARG italic_y end_ARG start_ARG italic_H end_ARG - divide start_ARG italic_y start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT + italic_y start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT end_ARG start_ARG 2 ⋅ italic_H end_ARG ) start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT end_ARG ,(2)

R p={1.0+(1−d d max)2,if⁢{x 1≤x≤x 2,y 1≤y≤y 2(1−d d max)2,otherwise subscript 𝑅 p cases 1.0 superscript 1 𝑑 subscript 𝑑 max 2 if cases subscript 𝑥 1 𝑥 subscript 𝑥 2 otherwise subscript 𝑦 1 𝑦 subscript 𝑦 2 otherwise superscript 1 𝑑 subscript 𝑑 max 2 otherwise R_{\text{p}}=\begin{cases}1.0+\left(1-\frac{d}{d_{\text{max}}}\right)^{2},&% \text{if }\begin{cases}x_{\text{1}}\leq x\leq x_{\text{2}},\\ y_{\text{1}}\leq y\leq y_{\text{2}}\end{cases}\\ \left(1-\frac{d}{d_{\text{max}}}\right)^{2},&\text{otherwise}\end{cases}italic_R start_POSTSUBSCRIPT p end_POSTSUBSCRIPT = { start_ROW start_CELL 1.0 + ( 1 - divide start_ARG italic_d end_ARG start_ARG italic_d start_POSTSUBSCRIPT max end_POSTSUBSCRIPT end_ARG ) start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT , end_CELL start_CELL if { start_ROW start_CELL italic_x start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ≤ italic_x ≤ italic_x start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , end_CELL start_CELL end_CELL end_ROW start_ROW start_CELL italic_y start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ≤ italic_y ≤ italic_y start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT end_CELL start_CELL end_CELL end_ROW end_CELL end_ROW start_ROW start_CELL ( 1 - divide start_ARG italic_d end_ARG start_ARG italic_d start_POSTSUBSCRIPT max end_POSTSUBSCRIPT end_ARG ) start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT , end_CELL start_CELL otherwise end_CELL end_ROW(3)

where H 𝐻 H italic_H and W 𝑊 W italic_W are image width and height, respectively, d means the normalized distance between the click point and the center point of the ground truth bounding box and d m⁢a⁢x subscript 𝑑 𝑚 𝑎 𝑥 d_{max}italic_d start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT is the normalized maximum distance between the center point of the ground truth bounding box and the four vertices of the image. Essentially, this formula assigns a reward R p subscript 𝑅 𝑝 R_{p}italic_R start_POSTSUBSCRIPT italic_p end_POSTSUBSCRIPT based on whether a predicted point (x,y)𝑥 𝑦(x,y)( italic_x , italic_y ) lies within a bounding box and its normalized distance from the box’s center. If the point is inside the box (i.e., x 1≤x≤x 2 subscript 𝑥 1 𝑥 subscript 𝑥 2 x_{\text{1}}\leq x\leq x_{\text{2}}italic_x start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ≤ italic_x ≤ italic_x start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT and y 1≤y≤y 2 subscript 𝑦 1 𝑦 subscript 𝑦 2 y_{\text{1}}\leq y\leq y_{\text{2}}italic_y start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ≤ italic_y ≤ italic_y start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT), the base reward is 1.0 1.0 1.0 1.0, and an additional decay term reduces the reward as the point moves farther from the center. If the point is outside the box, the reward only depends on the normalized distance between the point and the box center.

Finally, following previous works[[34](https://arxiv.org/html/2505.12370v2#bib.bib34); [23](https://arxiv.org/html/2505.12370v2#bib.bib23); [32](https://arxiv.org/html/2505.12370v2#bib.bib32)], we introduce format rewards during training to evaluate whether the generated output adheres to the expected output format. The final response reward is composed of format rewards and point rewards, defined as: R o=α⁢R f+β⁢R Point,subscript 𝑅 𝑜 𝛼 subscript 𝑅 f 𝛽 subscript 𝑅 Point R_{o}=\alpha R_{\text{f}}+\beta R_{\text{Point}},italic_R start_POSTSUBSCRIPT italic_o end_POSTSUBSCRIPT = italic_α italic_R start_POSTSUBSCRIPT f end_POSTSUBSCRIPT + italic_β italic_R start_POSTSUBSCRIPT Point end_POSTSUBSCRIPT , where R f subscript 𝑅 f R_{\text{f}}italic_R start_POSTSUBSCRIPT f end_POSTSUBSCRIPT represents the format reward, R Point subscript 𝑅 Point R_{\text{Point}}italic_R start_POSTSUBSCRIPT Point end_POSTSUBSCRIPT represents the point reward, and α 𝛼\alpha italic_α and β 𝛽\beta italic_β are weighting parameters, respectively.

### 3.3 Self-Evolutionary Reinforcement Fine-Tuning

The proposed algorithm, SE-RFT (Self-Evolutionary Reinforcement Fine-Tuning), outlines a principled approach to iteratively improving a vision-language GUI agent through attention-guided self-supervision. We begin by training an initial model using seed data. In each subsequent training iteration, the model from the previous round is used to generate attention maps for the current training samples. To do this, we generate the model’s output sequence and collect self-attention weights from the decoder across all transformer layers. Subsequently, the attention vector for each generated token is normalized, retaining only the components related to the visual tokens. Next, an average is computed across all layers to obtain an aggregated token-to-vision attention weight matrix. Finally, these attention values are projected back onto the original image resolution to produce spatial attention maps. These attention maps highlight the visual tokens on which the current instruction focuses.

If an attention map fails to focus appropriately on the correct UI element, it may indicate that _the model lacks relevant prior knowledge or that the sample presents a challenge beyond the model’s current capacity_. We found that such cases frequently lead to inaccurate localization.

Taking this into account, we propose to guide the loss function with the attention maps. If attention maps cannot correctly attend to the target UI element, these data may lead to incorrect guidance during training, hindering model optimization. To address this, we set the loss for these data points to zero, ensuring that they do not negatively impact the training process. The formula below is used to compute whether the attention map is devoted to the correct target UI element.

f⁢(attn,g⁢t bbox,τ)={1,if⁢P peak∧P global 0,otherwise 𝑓 attn 𝑔 superscript 𝑡 bbox 𝜏 cases 1 if subscript 𝑃 peak subscript 𝑃 global 0 otherwise f(\text{attn},gt^{\text{bbox}},\tau)=\begin{cases}1,&\text{if}~{}~{}P_{\text{% peak}}\land P_{\text{global}}\\ 0,&\text{otherwise}\end{cases}italic_f ( attn , italic_g italic_t start_POSTSUPERSCRIPT bbox end_POSTSUPERSCRIPT , italic_τ ) = { start_ROW start_CELL 1 , end_CELL start_CELL if italic_P start_POSTSUBSCRIPT peak end_POSTSUBSCRIPT ∧ italic_P start_POSTSUBSCRIPT global end_POSTSUBSCRIPT end_CELL end_ROW start_ROW start_CELL 0 , end_CELL start_CELL otherwise end_CELL end_ROW(4)

where attn means the spatial attention maps, τ 𝜏\tau italic_τ is the filtering threshold, P peak subscript 𝑃 peak P_{\text{peak}}italic_P start_POSTSUBSCRIPT peak end_POSTSUBSCRIPT indicates if there exists any significant activation point in the target region:

P peak=𝟏⁢(max(i,j)∈[x 1,x 2]×[y 1,y 2]⁡attn⁢[i,j]>τ),subscript 𝑃 peak 1 subscript 𝑖 𝑗 subscript 𝑥 1 subscript 𝑥 2 subscript 𝑦 1 subscript 𝑦 2 attn 𝑖 𝑗 𝜏 P_{\text{peak}}=\mathbf{1}\left(\max_{(i,j)\in[x_{1},x_{2}]\times[y_{1},y_{2}]% }\text{attn}[i,j]>\tau\right),italic_P start_POSTSUBSCRIPT peak end_POSTSUBSCRIPT = bold_1 ( roman_max start_POSTSUBSCRIPT ( italic_i , italic_j ) ∈ [ italic_x start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_x start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT ] × [ italic_y start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_y start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT ] end_POSTSUBSCRIPT attn [ italic_i , italic_j ] > italic_τ ) ,(5)

and P global subscript 𝑃 global P_{\text{global}}italic_P start_POSTSUBSCRIPT global end_POSTSUBSCRIPT indicates if the average attention weight in the target region is higher than the global average:

P global=𝟏⁢(1 H gt⁢W gt⁢∑i=x 1 x 2∑j=y 1 y 2 attn⁢[i,j]>1 H⁢W⁢∑i=0 H−1∑j=0 W−1 attn⁢[i,j]),subscript 𝑃 global 1 1 subscript 𝐻 gt subscript 𝑊 gt superscript subscript 𝑖 subscript 𝑥 1 subscript 𝑥 2 superscript subscript 𝑗 subscript 𝑦 1 subscript 𝑦 2 attn 𝑖 𝑗 1 𝐻 𝑊 superscript subscript 𝑖 0 𝐻 1 superscript subscript 𝑗 0 𝑊 1 attn 𝑖 𝑗 P_{\text{global}}=\mathbf{1}\left(\frac{1}{H_{\text{gt}}W_{\text{gt}}}\sum_{i=% x_{1}}^{x_{2}}\sum_{j=y_{1}}^{y_{2}}\text{attn}[i,j]>\frac{1}{HW}\sum_{i=0}^{H% -1}\sum_{j=0}^{W-1}\text{attn}[i,j]\right),italic_P start_POSTSUBSCRIPT global end_POSTSUBSCRIPT = bold_1 ( divide start_ARG 1 end_ARG start_ARG italic_H start_POSTSUBSCRIPT gt end_POSTSUBSCRIPT italic_W start_POSTSUBSCRIPT gt end_POSTSUBSCRIPT end_ARG ∑ start_POSTSUBSCRIPT italic_i = italic_x start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_x start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT end_POSTSUPERSCRIPT ∑ start_POSTSUBSCRIPT italic_j = italic_y start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_y start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT end_POSTSUPERSCRIPT attn [ italic_i , italic_j ] > divide start_ARG 1 end_ARG start_ARG italic_H italic_W end_ARG ∑ start_POSTSUBSCRIPT italic_i = 0 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_H - 1 end_POSTSUPERSCRIPT ∑ start_POSTSUBSCRIPT italic_j = 0 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_W - 1 end_POSTSUPERSCRIPT attn [ italic_i , italic_j ] ) ,(6)

where H gt subscript 𝐻 gt H_{\text{gt}}italic_H start_POSTSUBSCRIPT gt end_POSTSUBSCRIPT and W gt subscript 𝑊 gt W_{\text{gt}}italic_W start_POSTSUBSCRIPT gt end_POSTSUBSCRIPT are grounding truth bounding box width and height, respectively. First, P peak subscript 𝑃 peak P_{\text{peak}}italic_P start_POSTSUBSCRIPT peak end_POSTSUBSCRIPT checks if there exists at least one significant activation point within the target region by verifying if any pixel’s attention weight exceeds a given threshold τ 𝜏\tau italic_τ. Second, P global subscript 𝑃 global P_{\text{global}}italic_P start_POSTSUBSCRIPT global end_POSTSUBSCRIPT ensures that the average attention weight within the target region is higher than the global average attention weight across the entire attention map. Together, these criteria ensure that the attention mechanism not only highlights specific points within the target area but also prioritizes the region as a whole compared to the rest of the map. The final loss function is as follows:

ℒ(θ)=f(attn,g t bbox,τ)×𝔼 t[−π θ⁢(a t|s t)π old⁢(a t|s t)A t+γ⋅KL[π old(⋅|s t),π θ(⋅|s t)]],\mathcal{L}(\theta)=f(\text{attn},gt^{\text{bbox}},\tau)\times\mathbb{E}_{t}% \left[-\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\text{old}}(a_{t}|s_{t})}A_{t}+% \gamma\cdot\text{KL}[\pi_{\text{old}}(\cdot|s_{t}),\pi_{\theta}(\cdot|s_{t})]% \right],caligraphic_L ( italic_θ ) = italic_f ( attn , italic_g italic_t start_POSTSUPERSCRIPT bbox end_POSTSUPERSCRIPT , italic_τ ) × blackboard_E start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT [ - divide start_ARG italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT | italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) end_ARG start_ARG italic_π start_POSTSUBSCRIPT old end_POSTSUBSCRIPT ( italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT | italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) end_ARG italic_A start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT + italic_γ ⋅ KL [ italic_π start_POSTSUBSCRIPT old end_POSTSUBSCRIPT ( ⋅ | italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) , italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( ⋅ | italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) ] ] ,(7)

where π θ⁢(a t|s t)subscript 𝜋 𝜃 conditional subscript 𝑎 𝑡 subscript 𝑠 𝑡\pi_{\theta}(a_{t}|s_{t})italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT | italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) means the current policy, π old⁢(a t|s t)subscript 𝜋 old conditional subscript 𝑎 𝑡 subscript 𝑠 𝑡\pi_{\text{old}}(a_{t}|s_{t})italic_π start_POSTSUBSCRIPT old end_POSTSUBSCRIPT ( italic_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT | italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) means the old policy. The advantage function A t subscript 𝐴 𝑡 A_{t}italic_A start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT measures the relative value of an action, guiding the optimization process. Additionally, the hyperparameter γ 𝛾\gamma italic_γ controls the influence of the KL divergence term, KL[π old(⋅|s t),π θ(⋅|s t)]\text{KL}[\pi_{\text{old}}(\cdot|s_{t}),\pi_{\theta}(\cdot|s_{t})]KL [ italic_π start_POSTSUBSCRIPT old end_POSTSUBSCRIPT ( ⋅ | italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) , italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( ⋅ | italic_s start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) ], which penalizes large deviations between the old and new policies, thereby maintaining training stability.

4 Experiments
-------------

### 4.1 Implementation Details

For Reinforcement finetuning, we use the QwenVL2.5-3B/7B[[35](https://arxiv.org/html/2505.12370v2#bib.bib35)] model as the base model and train ten epochs. The α 𝛼\alpha italic_α, β 𝛽\beta italic_β, γ 𝛾\gamma italic_γ, τ 𝜏\tau italic_τ used in the formula are set as 1, 2, 0.004, and 0.2, respectively (More details can be found in appendix). During inference, to ensure fairness, we apply a unified and simple prompt across all benchmarks under zero-shot configurations. All experiments are conducted using 8×\times×NVIDIA A100-80G GPUs. We evaluate our model on six benchmarks that cover grounding and agents on three different platforms, including ScreenSpot[[18](https://arxiv.org/html/2505.12370v2#bib.bib18)], ScreenSpot-v2[[7](https://arxiv.org/html/2505.12370v2#bib.bib7)], ScreenSpot-Pro[[4](https://arxiv.org/html/2505.12370v2#bib.bib4)], AndroidControl-Low[[19](https://arxiv.org/html/2505.12370v2#bib.bib19)], AndroidControl-High[[19](https://arxiv.org/html/2505.12370v2#bib.bib19)] and OmniAct[[20](https://arxiv.org/html/2505.12370v2#bib.bib20)]. Following UGround[[6](https://arxiv.org/html/2505.12370v2#bib.bib6)], we use two commonly adopted metrics for GUI agents in evaluation: click point prediction accuracy, and step success rate, denoted as Grounding, and SR, respectively. In more detail, Grounding evaluates the performance of GUI grounding in downstream tasks. Besides, SR represents the step-wise success rate, where a step is deemed successful only if both the predicted action and its associated arguments (e.g., point for click actions and input text for scroll actions) are correct.

### 4.2 Experimental Results

We here evaluate our SE-GUI model by comparing it with current state-of-the-art (SOTA) models on various tasks including GUI grounding tasks, GUI low-level tasks, and GUI high-level tasks.

Grounding capability. We evaluate the grounding capability of SE-GUI using ScreenSpot[[18](https://arxiv.org/html/2505.12370v2#bib.bib18)], ScreenSpot-v2[[7](https://arxiv.org/html/2505.12370v2#bib.bib7)] and ScreenSpot-Pro[[4](https://arxiv.org/html/2505.12370v2#bib.bib4)]. ScreenSpot and ScreenSpot-v2 assesses GUI grounding performance across mobile, desktop, and web platforms, while ScreenSpot-Pro focuses on high-resolution professional environments, featuring expert-annotated tasks spanning 23 applications, five industries, and three operating systems.

As shown in Tab.[1](https://arxiv.org/html/2505.12370v2#S4.T1 "Table 1 ‣ 4.2 Experimental Results ‣ 4 Experiments ‣ Enhancing Visual Grounding for GUI Agents via Self-Evolutionary Reinforcement Learning"), compared to the previous SOTA model UI-TARS-72B, which is trained with large-scale data using supervised finetuning (SFT), our SE-GUI-7B achieves 8.5% improvement using only 0.2% of the data (3K vs. 14M). Furthermore, compared to the base models Qwen2.5-VL-7B and the SFT-trained Qwen2.5-VL-7B models using the same dataset, the RFT-based SE-GUI demonstrates much better performance in GUI grounding tasks. Moreover, the results in Tab.[3](https://arxiv.org/html/2505.12370v2#S4.T3 "Table 3 ‣ 4.2 Experimental Results ‣ 4 Experiments ‣ Enhancing Visual Grounding for GUI Agents via Self-Evolutionary Reinforcement Learning") reveals that SE-GUI-7B achieves 88.2% on ScreenSpot and 90.25% on ScreenSpot-v2, respectively. This highlights the effectiveness of our method in leveraging small-scale datasets to achieve significant performance improvements, which demonstrates its potential as a data-efficient and scalable approach for model training in resource-constrained environments.

Table 1: Performance comparison of different agent models across various task categories based on Text, Icon, and Average scores on ScreenSpot-Pro. Results marked in bold represent the best performance, and those underlined indicate the second-best performance.

Model CAD Dev Creative Scientific Office OS Avg.
Text Icon Text Icon Text Icon Text Icon Text Icon Text Icon Text Icon Avg.
Proprietary Models
GPT-4o[[36](https://arxiv.org/html/2505.12370v2#bib.bib36)]2.0 0.0 1.3 0.0 1.0 0.0 2.1 0.0 1.1 0.0 0.0 0.0 1.3 0.0 0.8
Claude Computer Use[[37](https://arxiv.org/html/2505.12370v2#bib.bib37)]14.5 3.7 22.0 3.9 25.9 3.4 33.9 15.8 30.1 16.3 11.0 4.5 23.4 7.1 17.1
General Open-source Models
Qwen2.5-VL-3B[[22](https://arxiv.org/html/2505.12370v2#bib.bib22)]9.1 7.3 22.1 1.4 26.8 2.1 38.2 7.3 33.9 15.1 10.3 1.1 23.6 3.8 16.1
Qwen2.5-VL-7B[[22](https://arxiv.org/html/2505.12370v2#bib.bib22)]16.8 1.6 46.8 4.1 35.9 7.7 49.3 7.3 52.5 20.8 37.4 6.7 38.9 7.1 26.8
GUI-specific Models
CogAgent-18B[[1](https://arxiv.org/html/2505.12370v2#bib.bib1)]7.1 3.1 14.9 0.7 9.6 0.0 22.2 1.8 13.0 0.0 5.6 0.0 12.0 0.8 7.7
Aria-UI[[17](https://arxiv.org/html/2505.12370v2#bib.bib17)]7.6 1.6 16.2 0.0 23.7 2.1 27.1 6.4 20.3 1.9 4.7 0.0 17.1 2.0 11.3
OS-Atlas-7B[[38](https://arxiv.org/html/2505.12370v2#bib.bib38)]12.2 4.7 33.1 1.4 28.8 2.8 37.5 7.3 33.9 5.7 27.1 4.5 28.1 4.0 18.9
ShowUI-2B[[33](https://arxiv.org/html/2505.12370v2#bib.bib33)]2.5 0.0 16.9 1.4 9.1 0.0 13.2 7.3 15.3 7.5 10.3 2.2 10.8 2.6 7.7
UGround-7B [[39](https://arxiv.org/html/2505.12370v2#bib.bib39)]14.2 1.6 26.6 2.1 27.3 2.8 31.9 2.7 31.6 11.3 17.8 0.0 25.0 2.8 16.5
UGround-V1-7B[[39](https://arxiv.org/html/2505.12370v2#bib.bib39)]15.8 1.2 51.9 2.8 47.5 9.7 57.6 14.5 60.5 13.2 38.3 7.9 45.2 8.1 31.1
UI-R1-3B[[40](https://arxiv.org/html/2505.12370v2#bib.bib40)]11.2 6.3 22.7 4.1 27.3 3.5 42.4 11.8 32.2 11.3 13.1 4.5 24.9 6.4 17.8
GUI-R1-3B[[10](https://arxiv.org/html/2505.12370v2#bib.bib10)]26.4 7.8 33.8 4.8 40.9 5.6 61.8 17.3 53.6 17.0 28.1 5.6---
GUI-R1-7B[[10](https://arxiv.org/html/2505.12370v2#bib.bib10)]23.9 6.3 49.4 4.8 38.9 8.4 55.6 11.8 58.7 26.4 42.1 16.9---
UI-TARS-2B[[2](https://arxiv.org/html/2505.12370v2#bib.bib2)]17.8 4.7 47.4 4.1 42.9 6.3 56.9 17.3 50.3 17.0 21.5 5.6 39.6 8.4 27.7
UI-TARS-7B[[2](https://arxiv.org/html/2505.12370v2#bib.bib2)]20.8 9.4 58.4 12.4 50.0 9.1 63.9 31.8 63.3 20.8 30.8 16.9 47.8 16.2 35.7
UI-TARS-72B[[2](https://arxiv.org/html/2505.12370v2#bib.bib2)]18.8 12.5 62.9 17.2 57.1 15.4 64.6 20.9 63.3 26.4 42.1 15.7 50.9 17.6 38.1
Ours
SE-GUI-3B 38.1 12.5 55.8 7.6 47.0 4.9 61.8 16.4 59.9 24.5 40.2 12.4 50.4 11.8 35.9
SE-GUI-7B 51.3 42.2 68.2 19.3 57.6 9.1 75.0 28.2 78.5 43.4 49.5 25.8 63.5 21.0 47.3

Table 2: Step accuracy on AndroidControl over 500 random actions from the test split. Baseline results are from [[19](https://arxiv.org/html/2505.12370v2#bib.bib19)]. Note that previous work are trained on AndroidControl and GUIACT, while our methods in the zero-shot setting are only trained on grounding data.

Planner Grounding AndroidControl_High AndroidControl_Low OmniACT
Grounding SR Grounding SR AS
Supervised Setting
GPT-4o SeeClick[[18](https://arxiv.org/html/2505.12370v2#bib.bib18)]-39.4-47.2 29.6
GPT-4o UGroundV1-7B[[6](https://arxiv.org/html/2505.12370v2#bib.bib6)]-48.4-62.4 32.8
Zero-shot Setting
GPT-4o Qwen2.5-VL-7B[[35](https://arxiv.org/html/2505.12370v2#bib.bib35)]31.6 36.0 62.6 58.0 24.1
GPT-4o SE-GUI-7B 65.6 52.8 79.6 68.2 34.4

Table 3: Performances on various platforms (Mobile, Desktop, Web) on ScreenSpot and ScreenSpot-v2. All experiments were conducted using raw screenshot information. Results marked in bold represent the best performance, and those underlined indicate the second-best performance.

Model ScreenSpot Accuracy (%)ScreenSpot-v2 Accuracy (%)
Mobile Desktop Web Avg.Mobile Desktop Web Avg.
Proprietary Models
GPT-4o[[41](https://arxiv.org/html/2505.12370v2#bib.bib41)]21.9 17.8 9.4 18.8 22.5 22.2 12.4 20.1
General Open-source Models
Qwen2-VL-7B[[22](https://arxiv.org/html/2505.12370v2#bib.bib22)]50.3 40.4 27.4 42.9 39.4 50.1 27.7 39.8
Qwen2.5-VL-3B [[22](https://arxiv.org/html/2505.12370v2#bib.bib22)]---55.5 55.5 44.0 39.1 46.9
Qwen2.5-VL-7B [[22](https://arxiv.org/html/2505.12370v2#bib.bib22)]---84.7 92.8 78.4 85.4 86.5
GUI-specific Models
CogAgent-18B[[26](https://arxiv.org/html/2505.12370v2#bib.bib26)]57.8 31.6 40.1 47.4 50.6 51.6 54.1 52.8
SeeClick-7B[[18](https://arxiv.org/html/2505.12370v2#bib.bib18)]68.1 48.8 41.8 53.4 51.8 65.5 40.7 53.9
OSAtlas-4B[[7](https://arxiv.org/html/2505.12370v2#bib.bib7)]56.2 74.9 69.9 68.5 74.9 56.9 70.0 68.5
UGround-7B[[6](https://arxiv.org/html/2505.12370v2#bib.bib6)]75.9 75.8 78.3 73.3 74.3 74.9 78.6 76.3
ShowUI-2B[[33](https://arxiv.org/html/2505.12370v2#bib.bib33)]84.8 70.8 76.2 75.1 70.0 85.1 73.3 77.3
OSAtlas-7B[[7](https://arxiv.org/html/2505.12370v2#bib.bib7)]85.0 78.8 84.5 82.5 78.3 85.5 83.8 83.3
Aguvis-7B[[8](https://arxiv.org/html/2505.12370v2#bib.bib8)]86.9 82.4 84.7 84.4 89.6 86.8 84.9 87.3
UI-TARS-2B [[2](https://arxiv.org/html/2505.12370v2#bib.bib2)]85.0 81.4 79.8 82.3 87.9 81.4 82.9 84.7
Ours
SE-GUI-7B 85.6 91.4 86.5 88.2 95.2 87.1 87.0 90.3

Agent evaluation. We evaluate our method, SE-GUI, on three challenging GUI agent benchmarks: AndroidControl_High, AndroidControl_Low and OmniACT. The AndroidControl benchmark [[19](https://arxiv.org/html/2505.12370v2#bib.bib19)] consists of 15K tasks across 833 Android apps. Each task contains a sequence of actions grounded in screenshots and a11y trees, with both high-level intents and optional low-level step-by-step instructions. Following the evaluation protocol in [[6](https://arxiv.org/html/2505.12370v2#bib.bib6)], we randomly sample 500 steps from the test split and report step-wise accuracy (SR), where a step is considered correct only if all predicted actions and arguments match the ground truth. As shown in Table[2](https://arxiv.org/html/2505.12370v2#S4.T2 "Table 2 ‣ 4.2 Experimental Results ‣ 4 Experiments ‣ Enhancing Visual Grounding for GUI Agents via Self-Evolutionary Reinforcement Learning"), SE-GUI-7B significantly outperforms all baselines across both task settings. In the low-level setting, where each step includes a fine-grained instruction, SE-GUI achieves 52.8% accuracy, outperforming GPT-4o (39.4%) and UGround-V1 (48.4%). In the high-level setting, where only the task goal is given, SE-GUI achieves 68.2%, a +5.8% improvement over the strongest baseline, UGround-V1 (62.4%). Notably, unlike prior methods such as UGround-V1 and SeeClick, which are trained on AndroidControl and GUIACT, SE-GUI is trained purely on grounding data, yet generalizes effectively to both task regimes.

For desktop and web environments, we evaluate on the OmniACT dataset [[42](https://arxiv.org/html/2505.12370v2#bib.bib42)], which comprises 9,802 tasks spanning 38 applications and 27 websites on macOS, Windows, and Linux. Each task involves generating a complete PyAutoGUI script based on a single screenshot. We follow the DetACT pipeline for a fair comparison, but replace its multimodal detection modules with an MLLM-based grounding system. Specifically, we prompt an MLLM to generate textual descriptions of target UI elements, then use SE-GUI to predict their screen coordinates. These coordinates are composed into scripts using the original prompts and retrieval setup of DetACT, including in-context demonstrations selected by task similarity. On this benchmark, SE-GUI achieves an action score of 34.4, surpassing all other baselines, including GPT-4o (29.6) and UGround-V1 (32.8). These results demonstrate the strong generalization and grounding capabilities of SE-GUI in both mobile and desktop GUI environments, even under zero-shot settings.

### 4.3 Ablation Study

Image resolution and data quality. To investigate the impact of image resolution and data quality on GUI RFT, we conduct corresponding ablation experiments, with the results shown in Fig.[3](https://arxiv.org/html/2505.12370v2#S4.F3 "Figure 3 ‣ 4.3 Ablation Study ‣ 4 Experiments ‣ Enhancing Visual Grounding for GUI Agents via Self-Evolutionary Reinforcement Learning")(a) and Tab.[4](https://arxiv.org/html/2505.12370v2#S4.T4 "Table 4 ‣ 4.3 Ablation Study ‣ 4 Experiments ‣ Enhancing Visual Grounding for GUI Agents via Self-Evolutionary Reinforcement Learning"). When training on a filtered, higher quality version of the dataset, the model achieves a 4.45% increase in accuracy. In contrast, using unfiltered, lower-quality data significantly degrades performance, reducing accuracy to just 31.31%. This highlights the importance of data quality in supporting effective model learning. We further examine the effect of image resolution by varying the maximum number of pixels from 1 million to 5 million. As the resolution increases, the model’s ability to recognize small but critical UI elements can be improved, leading to steady performance gains. This suggests that higher-resolution inputs provide more detailed visual information, which is especially valuable in complex and professional GUI tasks.

Due to computational constraints, our experiments are limited to a maximum of 5 million pixels. However, the observed trend implies that further improvements in resolution may yield additional performance benefits. Overall, these results emphasize the combined importance of clean, high-quality data and high-resolution visual inputs in advancing GUI grounding capabilities.

The effectiveness of dense point reward function. To explore the impact of the coefficients for format rewards and accuracy rewards in the reward function on the final performance, we conduct relevant ablation experiments, as shown in Table[4](https://arxiv.org/html/2505.12370v2#S4.T4 "Table 4 ‣ 4.3 Ablation Study ‣ 4 Experiments ‣ Enhancing Visual Grounding for GUI Agents via Self-Evolutionary Reinforcement Learning"). The results indicate that, compared to sparse rewards, dense rewards bring a 4.21% improvement . This is because dense rewards allow the model to receive more reward signals during the early stages of training, thus guiding the model to focus on the correct positions of UI elements.

The effectiveness of self-evolutionary reinforcment fine-tuning. To investigate the impact of self-evolutionary training on the final performance, we conducted relevant ablation experiments, as shown in Fig.[3](https://arxiv.org/html/2505.12370v2#S4.F3 "Figure 3 ‣ 4.3 Ablation Study ‣ 4 Experiments ‣ Enhancing Visual Grounding for GUI Agents via Self-Evolutionary Reinforcement Learning")(b). The results indicate that the first-stage model, trained solely on the filtered dataset SE-GUI-3k, achieved a performance of 39.97%. Subsequently, by leveraging the pre-trained model from the first stage to generate attention maps for supervising the training of the second-stage model, the performance improved by 3%. Continuing this process, where each subsequent stage utilizes the previous stage’s model to generate attention maps for supervision, the third-stage model achieved a performance of 46.55%. In the fourth stage, the improvement was minimal, indicating that the model’s performance had largely converged.

Furthermore, we visualize the reward curves for each training stage to better illustrate the learning dynamics. The curves are smoothed to clearly reveal the overall training trends and convergence behavior. As shown in Fig.[3](https://arxiv.org/html/2505.12370v2#S4.F3 "Figure 3 ‣ 4.3 Ablation Study ‣ 4 Experiments ‣ Enhancing Visual Grounding for GUI Agents via Self-Evolutionary Reinforcement Learning")(c), with the progression of training, each subsequent stage in the self-evolutionary process consistently achieves higher rewards compared to its predecessor. The final stage’s reward curve closely aligns with that of the preceding stage, indicating that the model has almost converged.

In addition, as shown in Fig.[3](https://arxiv.org/html/2505.12370v2#S4.F3 "Figure 3 ‣ 4.3 Ablation Study ‣ 4 Experiments ‣ Enhancing Visual Grounding for GUI Agents via Self-Evolutionary Reinforcement Learning")(d), we observe that on the ScreenSpot-Pro dataset, the previous state-of-the-art model, UI-TARS-72B, performs poorly in certain specialized scenarios, with accuracy dropping below 10%. In contrast, our model demonstrates strong generalization capabilities, achieving significant improvements in these challenging settings. Specifically, it boosts accuracy by nearly 40% on SolidWorks (SW) and Inventor (INV), which validates the effectiveness of our approach.

Table 4: Ablations for data quality and reward function on ScreenSpot-Pro.

Data quality reward function CAD Dev Creative Scientific Office OS Avg.
original w/filtering sparse dense
✓✗✓✗14.6 26.4 28.6 40.1 42.6 14.3 27.1
✓✗✗✓20.3 30.8 26.1 38.2 54.3 25.0 31.3
✗✓✓✗14.8 31.1 35.8 50.0 51.3 25.5 35.8
✗✓✗✓36.4 34.5 32.3 47.6 57.8 29.1 40.0
![Image 3: Refer to caption](https://arxiv.org/html/2505.12370v2/x3.png)

Figure 3: More experiments. (a) Perfromance of SE-GUI-3B improves as max pixel increases. (b) Performance in each training stage of self-evolutionary process. (c) The reward curves in each training stage of SE-GUI-7B in self-evolutionary process. (d) Performance comparison between UI-TARS-72B and our SE-GUI-7B in professional environments.

5 Conclusions and Limitations
-----------------------------

Conclusions. In this work, we explore how to more effectively leverage reinforcement learning to unlock the potential of large multimodal models as GUI Agents. Motivated by recent studies that emphasize the importance of training data quality for RL-based methods, we employ a data filtering strategy to curate a high-quality dataset, on which we train a base model. Building upon this, we adopt a self-evolutionary reinforcement fine-tuning paradigm to progressively enhance model performance. Our approach achieves state-of-the-art results across three grounding benchmarks. We hope this study provides new insights for future research in the field of GUI Agents.

Limitations and future work. Due to hardware limitations, this paper explores only two model scales: 3B and 7B. For the 7B model, we further constrained the maximum input resolution to 2 million pixels to limit the number of visual tokens, which may result in the loss of fine-grained details in high-resolution images. Nevertheless, we believe that our method exhibits strong scalability and generalization capabilities, and has the potential to achieve even better performance when applied to larger models such as 32B or 72B.

References
----------

*   [1] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14281–14290, 2024. 
*   [2] Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326, 2025. 
*   [3] Saaket Agashe, Kyle Wong, Vincent Tu, Jiachen Yang, Ang Li, and Xin Eric Wang. Agent s2: A compositional generalist-specialist framework for computer use agents, 2025. 
*   [4] Kaixin Li, Ziyang Meng, Hongzhan Lin, Ziyang Luo, Yuchen Tian, Jing Ma, Zhiyong Huang, and Tat-Seng Chua. Screenspot-pro: Gui grounding for professional high-resolution computer use. Workshop on Reasoning and Planning for Large Language Models, 2025. 
*   [5] Shravan Nayak, Xiangru Jian, Kevin Qinghong Lin, Juan A Rodriguez, Montek Kalsi, Rabiul Awal, Nicolas Chapados, M Tamer Özsu, Aishwarya Agrawal, David Vazquez, et al. Ui-vision: A desktop-centric gui benchmark for visual perception and interaction. arXiv preprint arXiv:2503.15661, 2025. 
*   [6] Boyu Gou, Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. Navigating the digital world as humans do: Universal visual grounding for gui agents. arXiv preprint arXiv:2410.05243, 2024. 
*   [7] Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, et al. Os-atlas: A foundation action model for generalist gui agents. arXiv preprint arXiv:2410.23218, 2024. 
*   [8] Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. Aguvis: Unified pure vision agents for autonomous gui interaction. arXiv preprint arXiv:2412.04454, 2024. 
*   [9] Zhengxi Lu, Yuxiang Chai, Yaxuan Guo, Xi Yin, Liang Liu, Hao Wang, Guanjing Xiong, and Hongsheng Li. Ui-r1: Enhancing action prediction of gui agents by reinforcement learning. arXiv preprint arXiv:2503.21620, 2025. 
*   [10] Xiaobo Xia and Run Luo. Gui-r1: A generalist r1-style vision-language action model for gui agents. arXiv preprint arXiv:2504.10458, 2025. 
*   [11] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023. 
*   [12] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 
*   [13] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 
*   [14] Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. Limo: Less is more for reasoning. arXiv preprint arXiv:2502.03387, 2025. 
*   [15] Xuefeng Li, Haoyang Zou, and Pengfei Liu. Limr: Less is more for rl scaling. arXiv preprint arXiv:2502.11886, 2025. 
*   [16] Chong Zhang, Yue Deng, Xiang Lin, Bin Wang, Dianwen Ng, Hai Ye, Xingxuan Li, Yao Xiao, Zhanfeng Mo, Qi Zhang, et al. 100 days after deepseek-r1: A survey on replication studies and more directions for reasoning language models. arXiv preprint arXiv:2505.00551, 2025. 
*   [17] Yuhao Yang, Yue Wang, Dongxu Li, Ziyang Luo, Bei Chen, Chao Huang, and Junnan Li. Aria-ui: Visual grounding for gui instructions. arXiv preprint arXiv:2412.16256, 2024. 
*   [18] Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Yantao Li, Jianbing Zhang, and Zhiyong Wu. Seeclick: Harnessing gui grounding for advanced visual gui agents. arXiv preprint arXiv:2401.10935, 2024. 
*   [19] Wei Li, William Bishop, Alice Li, Chris Rawles, Folawiyo Campbell-Ajala, Divya Tyamagundlu, and Oriana Riva. On the effects of data scale on computer control agents. arXiv preprint arXiv:2406.03679, 2024. 
*   [20] Raghav Kapoor, Yash Parag Butala, Melisa Russak, Jing Yu Koh, Kiran Kamble, Waseem AlShikh, and Ruslan Salakhutdinov. Omniact: A dataset and benchmark for enabling multimodal generalist autonomous agents for desktop and web. In ECCV, pages 161–178. Springer, 2024. 
*   [21] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024. 
*   [22] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 
*   [23] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 
*   [24] Chi Zhang, Zhao Yang, Jiaxuan Liu, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. Appagent: Multimodal agents as smartphone users. arXiv preprint arXiv:2312.13771, 2023. 
*   [25] Yanda Li, Chi Zhang, Wanqi Yang, Bin Fu, Pei Cheng, Xin Chen, Ling Chen, and Yunchao Wei. Appagent v2: Advanced agent for flexible mobile interactions. arXiv preprint arXiv:2408.11824, 2024. 
*   [26] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14281–14290, 2024. 
*   [27] Junyang Wang, Haiyang Xu, Jiabo Ye, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-agent: Autonomous multi-modal mobile device agent with visual perception. arXiv preprint arXiv:2401.16158, 2024. 
*   [28] Junyang Wang, Haiyang Xu, Haitao Jia, Xi Zhang, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-agent-v2: Mobile device operation assistant with effective navigation via multi-agent collaboration. arXiv preprint arXiv:2406.01014, 2024. 
*   [29] Hao Wen, Yuanchun Li, Guohong Liu, Shanhui Zhao, Tao Yu, Toby Jia-Jun Li, Shiqi Jiang, Yunhao Liu, Yaqin Zhang, and Yunxin Liu. Autodroid: Llm-powered task automation in android. arXiv preprint arXiv:2308.15272, 2023. 
*   [30] Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326, 2025. 
*   [31] Haozhan Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, Ruochen Xu, and Tiancheng Zhao. Vlm-r1: A stable and generalizable r1-style large vision-language model. [https://github.com/om-ai-lab/VLM-R1](https://github.com/om-ai-lab/VLM-R1), 2025. Accessed: 2025-02-15. 
*   [32] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025. 
*   [33] Kevin Qinghong Lin, Linjie Li, Difei Gao, Zhengyuan Yang, Zechen Bai, Weixian Lei, Lijuan Wang, and Mike Zheng Shou. Showui: One vision-language-action model for generalist gui agent. In NeurIPS 2024 Workshop on Open-World Agents, 2024. 
*   [34] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, et al. Mm-eureka: Exploring visual aha moment with rule-based large-scale reinforcement learning. arXiv preprint arXiv:2503.07365, 2025. 
*   [35] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 
*   [36] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 
*   [37] Anthropic. Developing a computer use model. [https://www.anthropic.com/news/developing-computer-use](https://www.anthropic.com/news/developing-computer-use), 2024. Accessed: 2025-04-12. 
*   [38] Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, et al. Os-atlas: A foundation action model for generalist gui agents. arXiv preprint arXiv:2410.23218, 2024. 
*   [39] Boyu Gou, Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. Navigating the digital world as humans do: Universal visual grounding for gui agents. arXiv preprint arXiv:2410.05243, 2024. 
*   [40] Zhengxi Lu, Yuxiang Chai, Yaxuan Guo, Xi Yin, Liang Liu, Hao Wang, Guanjing Xiong, and Hongsheng Li. Ui-r1: Enhancing action prediction of gui agents by reinforcement learning. arXiv preprint arXiv:2503.21620, 2025. 
*   [41] OpenAI. Gpt-4o, 2024. Accessed: 2025-01-03. 
*   [42] Raghav Kapoor, Yash Parag Butala, Melisa Russak, Jing Yu Koh, Kiran Kamble, Waseem Alshikh, and Ruslan Salakutdinov. Omniact: A dataset and benchmark for enabling multimodal generalist autonomous agents for desktop and web. arXiv preprint arXiv:2402.17553, 2024. 
*   [43] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 
*   [44] Weihao Zeng, Yuzhen Huang, Wei Liu, Keqing He, Qian Liu, Zejun Ma, and Junxian He. 7b model and 8k examples: Emerging reasoning with reinforcement learning is both effective and efficient. [https://hkust-nlp.notion.site/simplerl-reason](https://hkust-nlp.notion.site/simplerl-reason), 2025. Notion Blog. 
*   [45] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025. 

Appendix A GRPO preliminary
---------------------------

Many rule-based RL works[[43](https://arxiv.org/html/2505.12370v2#bib.bib43), [44](https://arxiv.org/html/2505.12370v2#bib.bib44), [45](https://arxiv.org/html/2505.12370v2#bib.bib45)] adopt the Group Relative Policy Optimization (GRPO) algorithm[[13](https://arxiv.org/html/2505.12370v2#bib.bib13)] for RL training. GRPO offers an alternative to commonly used Proximal Policy Optimization (PPO)[[12](https://arxiv.org/html/2505.12370v2#bib.bib12)] by eliminating the need for a critic model. Instead, GRPO directly compares a group of candidate responses to determine their relative quality.

In GRPO, given a task question, the model generates a set of N 𝑁 N italic_N potential responses {o 1,o 2,…,o N}subscript 𝑜 1 subscript 𝑜 2…subscript 𝑜 𝑁\{o_{1},o_{2},\dots,o_{N}\}{ italic_o start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_o start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_o start_POSTSUBSCRIPT italic_N end_POSTSUBSCRIPT }. Each response is evaluated by taking the corresponding actions and computing its reward {r 1,r 2,…,r N}subscript 𝑟 1 subscript 𝑟 2…subscript 𝑟 𝑁\{r_{1},r_{2},\dots,r_{N}\}{ italic_r start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_r start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_r start_POSTSUBSCRIPT italic_N end_POSTSUBSCRIPT }. Unlike PPO, which relies on a single reward signal and a critic to estimate the value function, GRPO normalizes these rewards to calculate the relative advantage of each response. The relative quality A i subscript 𝐴 𝑖 A_{i}italic_A start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT of the i 𝑖 i italic_i-th response is computed as

A i=r i−Mean⁢({r 1,r 2,…,r N})Std⁢({r 1,r 2,…,r N}),subscript 𝐴 𝑖 subscript 𝑟 𝑖 Mean subscript 𝑟 1 subscript 𝑟 2…subscript 𝑟 𝑁 Std subscript 𝑟 1 subscript 𝑟 2…subscript 𝑟 𝑁 A_{i}=\frac{r_{i}-\mathrm{Mean}(\{r_{1},r_{2},\dots,r_{N}\})}{\mathrm{Std}(\{r% _{1},r_{2},\dots,r_{N}\})},italic_A start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = divide start_ARG italic_r start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - roman_Mean ( { italic_r start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_r start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_r start_POSTSUBSCRIPT italic_N end_POSTSUBSCRIPT } ) end_ARG start_ARG roman_Std ( { italic_r start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_r start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_r start_POSTSUBSCRIPT italic_N end_POSTSUBSCRIPT } ) end_ARG ,(8)

where Mean Mean\mathrm{Mean}roman_Mean and Std Std\mathrm{Std}roman_Std represent the mean and standard deviation of the rewards, respectively. This normalization step ensures that responses are compared within the context of the group, allowing GRPO to better capture nuanced differences between candidates. Policy updates are further constrained by minimizing the KL divergence between the updated and reference models, ensuring stable RL learning.

Appendix B More training details.
---------------------------------

Due to resource constraints, during training on the 7B model, we limited the maximum input resolution to 2 million pixels. We believe that increasing this limit could lead to further performance improvements. Below, we present several representative prompts used during training, along with the complete reward curves, as shown in Fig.[4](https://arxiv.org/html/2505.12370v2#A2.F4 "Figure 4 ‣ Appendix B More training details. ‣ Enhancing Visual Grounding for GUI Agents via Self-Evolutionary Reinforcement Learning").

![Image 4: Refer to caption](https://arxiv.org/html/2505.12370v2/x4.png)

Figure 4: Training curve of our SE-GUI-7B model.

Appendix C Attention visulization.
----------------------------------

We present several visualizations of the model’s attention behavior. As shown in Fig.[5](https://arxiv.org/html/2505.12370v2#A3.F5 "Figure 5 ‣ Appendix C Attention visulization. ‣ Enhancing Visual Grounding for GUI Agents via Self-Evolutionary Reinforcement Learning"), given the user instruction “pin Jack’s conversation”, the model correctly attends to the upper-right region of the image. During the visualization process, we first extract the attention matrices from the forward pass. We then identify the positions of the visual tokens and compute the attention weights from the generated tokens to these visual tokens. Finally, the attention values corresponding to the visual tokens are mapped back to the original image resolution, yielding a spatial attention map that reflects how the model attends to different regions of the image.

![Image 5: Refer to caption](https://arxiv.org/html/2505.12370v2/x5.png)

Figure 5: The model attention on specific training sample.

![Image 6: Refer to caption](https://arxiv.org/html/2505.12370v2/x6.png)

Figure 6: The model attention on specific training sample.

Appendix D Hyper parameters
---------------------------

As shown in Tab.[6](https://arxiv.org/html/2505.12370v2#A4.T6 "Table 6 ‣ Appendix D Hyper parameters ‣ Enhancing Visual Grounding for GUI Agents via Self-Evolutionary Reinforcement Learning"), we observed that the relative weights of the format reward and point reward have a noticeable impact on the final performance. Specifically, setting the ratio of format reward to point reward to 1:1 leads to a slight performance drop. To address this, we adjusted the weight ratio to 1:2 during training. We attribute this to the relative simplicity of learning format consistency; reducing its weight encourages the model to focus more on prediction accuracy.

In addition, the KL divergence coefficient also influences model performance. Our experiments show that a smaller KL divergence leads to better results. We hypothesize that a large KL term forces the model to overly align with the reference policy, thereby limiting its capacity for exploratory learning. Consequently, we set the KL divergence coefficient to 0.004 in our final experimental setup.

We also conducted detailed ablation studies on the attention map filtering threshold. When the threshold is set too high, it may filter out samples with potential for reasoning. Conversely, when the threshold is too low, it fails to provide effective supervisory guidance. Therefore, we selected a threshold of 0.2.

Table 5: Ablations for SE-GUI-3B on ScreenSpot-Pro.

Reward Weight (α:β:𝛼 𝛽\alpha:\beta italic_α : italic_β)γ 𝛾\gamma italic_γ Average
1:1 1:2 0.04 0.004
✓✗✓✗24.85
✓✗✗✓27.07
✗✓✓✗31.31
✗✓✗✓34.28

Table 6: Ablations for SE-GUI-7B on ScreenSpot-Pro.

τ 𝜏\tau italic_τ Average
0.4 0.2 0.1
✓✗✗42.95
✗✓✗44.78
✗✗✓43.51

