---

# ACCELERATING NEURAL ARCHITECTURE EXPLORATION ACROSS MODALITIES USING GENETIC ALGORITHMS

---

**Daniel Cummings**

Intel Labs, Intel Corporation  
daniel.cummings@intel.com

**Sharath Nittur Sridhar**

Intel Labs, Intel Corporation  
sharath.nittur.sridhar@intel.com

**Anthony Sarah**

Intel Labs, Intel Corporation  
anthony.sarah@intel.com

**Maciej Szankin**

Intel Labs, Intel Corporation  
maciej.szankin@intel.com

## ABSTRACT

Neural architecture search (NAS), the study of automating the discovery of optimal deep neural network architectures for tasks in domains such as computer vision and natural language processing, has seen rapid growth in the machine learning research community. While there have been many recent advancements in NAS, there is still a significant focus on reducing the computational cost incurred when validating discovered architectures by making search more efficient. Evolutionary algorithms, specifically genetic algorithms, have a history of usage in NAS and continue to gain popularity versus other optimization approaches as a highly efficient way to explore the architecture objective space. Most NAS research efforts have centered around computer vision tasks and only recently have other modalities, such as the rapidly growing field of natural language processing, been investigated in depth. In this work, we show how genetic algorithms can be paired with lightly trained objective predictors in an iterative cycle to accelerate multi-objective architectural exploration in a way that works in the modalities of both machine translation and image classification.

## 1 Introduction

Automating the process of finding optimal deep neural network (DNN) architectures for a given task, known as neural architecture search (NAS), has seen significant progress in the research community particularly in computer vision. However, the computational overhead of evaluating the performance of the discovered DNN architectures can be very costly due to the training and validation cycle. To address the training overhead, novel weight sharing approaches known as one-shot or super-networks [1, 2, 3] have offered a way to mitigate the training overhead by reducing training times from thousands to a few GPU days [4]. These approaches train a task specific super-network architecture with a weight-sharing mechanism that allows the sub-networks to be treated as their own architectures. This enables sub-network model validation without a separate training cycle. However, the validation component still comes with a high overhead since there are many possible models to search across for large super-networks and the validation step itself comes with a computational cost, especially for larger datasets such as ImageNet [5]. One popular way to mitigate the validation cost is to train predictors for objectives such as inference time (a.k.a. latency) and accuracy from a training set with thousands of sampled architectures. Then the trained predictors are used for approximating model performance during the NAS process where only the best performing candidates are validated in the end. While reinforcement learning, sequential model-based optimization, and gradient optimization have been applied to the model search problem, evolutionary approaches, specifically genetic algorithms (GAs), have seen ongoing popularity in NAS work [6]. GAs have been broadly applied specifically for computer vision NAS problems in both single-objective implementations [7] and in multi-objective approaches such as NSGA-Net [8]. In the context of super-networks, they have been used to either inform a fine-tuned training of the super-network or to search sub-network architectures after training [9].Most NAS research efforts have centered around the computer vision task of image classification and only recently have other modalities, such as the rapidly growing field of language modeling or language translation, been investigated in detail [10, 11]. Moreover, understanding how NAS approaches generalize and perform across modalities and tasks has not been studied in depth. In this work we demonstrate how pairing GAs in an iterated fashion with lightly trained predictors can yield an accelerated and less costly exploration of the architecture search space in what we term as Lightweight Iterative NAS (LINAS). We show how this approach is extensible to machine translation NAS, given that most research has been focused on computer vision tasks and focus our NAS experiments on super-network frameworks given their large architectural design spaces. For the machine translation task, we use a transformer super-network architecture that has a search space size of  $10^{15}$  [10]. For the image classification task, we apply our approach to a MobileNetV3 super-network that has a search space size of  $10^{19}$  [9]. While many NAS approaches focus only on a single optimization objective, such as maximizing model accuracy for a particular latency or model complexity metric, we perform our experiments in the multi-objective context since searching for architectures that are optimized to a specific hardware platform (e.g., finding an optimal Pareto front for trade-offs in accuracy and latency) continues to be an important industry-wide application.

## 2 Proposed Algorithm

The goal for our algorithm is to reduce the number of validation measurements that are required to find optimal DNN architectures in a multi-objective search space in a way that works well across modalities, in our case machine translation (e.g., Transformer super-network) and image classification (e.g., MobileNetV3 super-network). While existing work shows that using trained predictors can speed up the DNN architecture search process, there remains a substantial cost to training predictors since the number of validated training samples can often range between 1000 and 16000 samples [12]. Interestingly, as shown in Figure 1, accuracy predictors can achieve acceptable mean absolute percentage error (MAPE) with far fewer training samples. Likewise, we have found that predictors perform well for latency and multiply-and-accumulates (MACs) objectives. We build on this insight that lightly trained predictors can offer a useful surrogate signal during search and combine this with the knowledge that GAs have been used for NAS applications with great success.

Algorithm 1 illustrates the LINAS flow where we first randomly sample the architecture search space to serve as the initial validation population. For each validation population, we measure each objective for the individuals and store the result. These results are combined with all previous validation population results and are used to train the objective predictors. For each iteration, we run a multi-objective genetic algorithm search (NSGA-II [13] in this work) using that iteration’s trained predictors for a high number of generations (e.g.,  $> 200$ ) to allow the algorithm to explore the predicted objective space sufficiently. This predictor-based GA search runs very quickly since no validation measurements occur. Finally, we select the most optimal population of diverse DNN architectures from the predictor-based GA search to add to the next validation population, which then informs the next round of predictor training. This cycle continues until the iteration count limit is met or an end-user decides a sufficient set of architectures has been discovered. We note that the LINAS approach can be applied with any single-, multi-, or many-objective evolutionary algorithm and generalizes to work with any fully-trained super-network framework. Additionally, it allows for the interchanging of GA tuning parameters (e.g. crossover, mutation, population), GA models, and predictor types for each iteration.

---

### Algorithm 1 Lightweight Iterative Neural Architecture Search

---

**Input:** Objectives  $f_m$ , super-network with weights  $\mathcal{W}$  and configurations  $\Omega$ , predictor model for each objective  $Y_m$ , LINAS population  $P$  size  $n$ , number of LINAS iterations  $I$ , genetic algorithm  $\mathcal{G}$  with the number of generations  $J$ .  
 $P_{i=0} \leftarrow \{\omega_n\} \in \Omega$  // sample  $n$  sub-networks for first population  
**while**  $i++ < I$  **do**  
     $D_{i,m} \leftarrow f_m(P_i \in \Omega; \mathcal{W})$  // measure  $f_m$ , store data  $D_{i,m}$   
     $D_{all,m} \leftarrow D_{all,m} \cup D_{i,m}$   
     $Y_{m,pred} \leftarrow Y_{m,train}(D_{all,m})$  // train predictors  
    **while**  $j++ < J$  **do**  
         $P_{\mathcal{G}_j} \leftarrow \mathcal{G}(Y_{m,pred}, j)$  // run  $\mathcal{G}$  for  $J$  generations  
    **end while**  
     $P_i \leftarrow P_{\mathcal{G},best} \in P_{\mathcal{G}_J}$  // retrieve optimal population  
**end while**  
**Output:** All LINAS validation populations  $P_I$ , GA predictor search results  $P_{\mathcal{G}_I, J}$ , and validation data  $D_{all,m}$ .

---Figure 1: MAPE versus training examples for Transformer super-network BLEU score (SVR predictor) and MobileNetV3 super-network top-1 accuracy (ridge predictor). Each point is the average over 10 trials with error bars showing one standard deviation.

### 3 Experiment

#### 3.1 Experimental Setup

We demonstrate the LINAS approach on the modalities of machine translation and image classification since they are representative of highly popular Transformer and convolutional DNN layer types respectively. Since our focus is on the DNN architecture search process, we do not re-train from scratch or fine-tune the optimal discovered architectures. For our experiments, we compare against validation-only measurements from a random search that uniformly samples the architecture space and NSGA-II for a well-known GA baseline. We leverage the pymoo [14] implementation of the NSGA-II algorithm and note that similar results can be achieved with AGE-MOEA [15].

Recent studies have looked at the performance for a wide range of predictor types for smaller DNN search spaces [16]. For this work, the LINAS algorithm uses ridge and support vector machine regression (SVR) predictors with a one-hot encoding approach. In our studies, we found that these simpler methods converge more quickly, require fewer training examples, and require much less hyper-parameter optimization than multi-layer perceptrons (MLPs). The LINAS internal predictor-based GA uses NSGA-II with the same population, mutation, and crossover settings as the NSGA-II baseline. Table 1 summarizes the DNN architectures, search space size, predictor types, and GA settings used in the experiments.

<table border="1">
<thead>
<tr>
<th>Architecture (Modality)</th>
<th>Transformer (Machine Translation)</th>
<th>MobileNetV3 (Image Classification)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Predictor</td>
<td>SVR w/ RBF kernel</td>
<td>Ridge</td>
</tr>
<tr>
<td>Search Space</td>
<td><math>10^{15}</math></td>
<td><math>10^{19}</math></td>
</tr>
<tr>
<td>Population</td>
<td>50</td>
<td>50</td>
</tr>
<tr>
<td>Crossover</td>
<td>0.9</td>
<td>0.9</td>
</tr>
<tr>
<td>Mutation</td>
<td>0.02</td>
<td>0.02</td>
</tr>
</tbody>
</table>

Table 1: Experiment settings for both neural network architecture types. The predictor types apply only to the LINAS setup.

Building a GA compatible encoding or representation of the architectural design variables is a critical step when applying GAs to NAS problems. We illustrate our encoding strategy in Figures 2 and 3 that allows the GA operators (e.g., mutation, crossover, etc.) to run correctly and gives each design variable several integer options. For the machine translation modality, we run our experiments using the Transformer super-network. Our work is based on a recent and popular approach for machine translation [10] that samples the architecture encoder and decoder space during training to achieve a super-network model. We use a search space with an embedding dimension chosen from  $\{512, 640\}$ , hidden dimension from  $\{1024, 2048, 3072\}$ , attention head number from  $\{4, 8\}$ , decoder layer number from  $\{1, 2, 3, 4\}$ ,EDim  $\rightarrow$  Embedding Dimension = {512, 640}  
 HDim  $\rightarrow$  Hidden Dimension = {1024, 2048, 3072}  
 SelfAttn  $\rightarrow$  Self-Attention Heads = {4, 8}  
 EnDeAttn  $\rightarrow$  Encoder-Decoder Attention Heads = {4, 8}  
 ArbEnDeAttn  $\rightarrow$  Arbitrary Encoder-Decoder Attn. = {1, 2, 3}

Encoder Layers = {6}  
 Decoder Layers = {1, 2, 3, 4, 5, 6}

<table border="1">
<thead>
<tr>
<th>EDim</th>
<th colspan="2">Encoder Layer 1</th>
<th colspan="2">Encoder Layer 2</th>
<th colspan="2">Encoder Layer 6</th>
<th>EDim</th>
<th colspan="2">Decoder Layer 1</th>
<th colspan="2">Decoder Layer j</th>
</tr>
</thead>
<tbody>
<tr>
<td>512</td>
<td>8</td>
<td>1024</td>
<td>4</td>
<td>2048</td>
<td>8</td>
<td>2048</td>
<td>640</td>
<td>4</td>
<td>4</td>
<td>2048</td>
<td>8</td>
<td>3072</td>
</tr>
</tbody>
</table>

Figure 2: Encoding diagram for the Transformer search space which has 40 design variables.

{5, 6} and a constant encoder layer number {6}. Additionally, each decoder layer can attend to the last {1, 2, 3} encoder layers (arbitrary encoder-decoder attention). The first objective for the Transformer super-network is to maximize the bilingual evaluation understudy (BLEU) [17] score evaluated on the WMT 2014 En-De data set. For the BLEU score evaluation, we use a beam size of 5 and a length penalty of 0.6.

For the computer vision modality, we perform an image classification task using a super-network based on the MobileNetV3 architecture [9]. In this framework, convolutional parameters such as block depth, channel width, and kernel size are used with a progressive shrinking approach during training. We allow for an layer depth chosen from {2, 3, 4}, an width expansion ratio chosen from {3, 4, 6}, and a kernel size chosen from {3, 5, 7} as shown in Figure 3. The first objective is to maximize the top-1 accuracy using the ImageNet validation data set. The second objective for both modalities is to minimize latency and is chosen instead of MACs or DNN parameter counts since it is a more relevant hardware performance metric for real-life applications.

224x224 image

Static Layers

Block 1: W4 K3, W3 K3, skip, skip

Block 2: W4 K3, W6 K5, W4 K7, W3 K7

Block 5: W3 K5, W6 K3, W6 K7, skip

D = {2, 3, 4}

K = {3, 5, 7}

W = {3, 4, 6}

<table border="1">
<thead>
<tr>
<th></th>
<th colspan="4">2</th>
<th colspan="4">4</th>
</tr>
</thead>
<tbody>
<tr>
<td>D</td>
<td>2</td>
<td>2</td>
<td>2</td>
<td>2</td>
<td>2</td>
<td>2</td>
<td>2</td>
<td>2</td>
</tr>
<tr>
<td>K</td>
<td>3</td>
<td>3</td>
<td>X</td>
<td>X</td>
<td>3</td>
<td>5</td>
<td>7</td>
<td>7</td>
</tr>
<tr>
<td>W</td>
<td>4</td>
<td>3</td>
<td>X</td>
<td>X</td>
<td>4</td>
<td>6</td>
<td>4</td>
<td>3</td>
</tr>
</tbody>
</table>

<table border="1">
<thead>
<tr>
<th></th>
<th colspan="4">3</th>
</tr>
</thead>
<tbody>
<tr>
<td>D</td>
<td>3</td>
<td>3</td>
<td>3</td>
<td>3</td>
</tr>
<tr>
<td>K</td>
<td>5</td>
<td>3</td>
<td>7</td>
<td>X</td>
</tr>
<tr>
<td>W</td>
<td>3</td>
<td>6</td>
<td>6</td>
<td>X</td>
</tr>
</tbody>
</table>

Figure 3: Encoding diagram of the MobileNetV3 search space which has 45 design variables.

### 3.2 Results

The main purpose of our proposed LINAS approach is to reduce the total number of validation measurements required to find optimal DNN architectures in the multi-objective space for any modality (e.g., computer vision, machine translation). Specifically, for this work we want to efficiently discover architectures with optimal trade-offs in high accuracy/BLUE and low latency. Figures 4 and 5 illustrate the differences in how the NSGA-II baseline search evolves versus the proposed LINAS approach. While NSGA-II reliably progresses towards an optimal trade-off region, the LINAS results show how that exploration can be accelerated. An important note for discussion is that the Transformer architectural distributions in the BLEU and latency objective space fall into more discrete clusters that are a function ofFigure 4: Comparison of LINAS and NSGA-II in the Transformer multi-objective search space for the same number of evaluations (validations) on a NVIDIA Titan-V GPU system.

Figure 5: Comparison of LINAS and NSGA-II in the MobileNetV3 multi-objective search space for the same number of evaluations (validations) on a NVIDIA Titan-V GPU system.

the decoder layers. Since the distribution of these models is both constrained in range and occurs closer to an optimal region, even if randomly sampled, the results look less differentiated than the MobileNetV3 results.

Another way of visualizing the benefits of our approach is by evaluating the hypervolume (HV) versus evaluation counts. The hypervolume indicator [18] offers a way to measure how well the Pareto front approximates the optimal solution. When measuring over two objectives, the hypervolume term represents the area of the Pareto front with respect to a reference point. Figures 6 and 7 show the results of 5 trials for each algorithm with respect to the number of architecture evaluations (a.k.a. validation measurements). Each evaluation takes approximately 3 minutes in our setup meaning that 1000 evaluations would take about 50 GPU hours.

A key observation is how quickly LINAS accelerates to a better hypervolume versus the baseline NSGA-II search since each LINAS validation population after the first, represents the best predicted objective space information from the GA-predictor pairing. Depending on which region of the Pareto front is important, an end-user would be more likely to identify optimal architectures in fewer evaluations with LINAS. Given the constrained characteristics of the Transformer objective space and the higher MAPE in the BLEU predictor (see Figure 1), the LINAS result is less differentiating for the Transformer results than in the MobileNetV3 case.Figure 6: Hypervolume versus evaluation count in the machine translation Transformer search space (HV reference point BLEU=20, latency=200 ms). Shaded regions show the standard error for 5 trials.

Figure 7: Hypervolume versus evaluation count for various search approaches in the image classification MobileNetV3 search space (HV reference point top-1=70%, latency=70 ms). Shaded regions show the standard error for 5 trials.

## 4 Conclusion

The goal of the work was to demonstrate how GAs can be uniquely leveraged to accelerate multi-objective neural architecture search on the modalities of machine translation and image classification. The LINAS algorithm offers a modular framework that can easily be modified to fit a variety of NAS application domains. As NAS research continues to gain momentum, we highlight the need to continue to investigate the generalizability of NAS approaches in modalities outside of computer vision. Future work includes evaluating the use of proxy functions [19] and advances in meta-learning [20] to extend this type of algorithmic framework.## References

- [1] Hanxiao Liu, Karen Simonyan, and Yiming Yang. DARTS: differentiable architecture search. *CoRR*, abs/1806.09055, 2018.
- [2] Gabriel Bender, Pieter-Jan Kindermans, Barret Zoph, Vijay Vasudevan, and Quoc Le. Understanding and simplifying one-shot architecture search. In Jennifer Dy and Andreas Krause, editors, *Proceedings of the 35th International Conference on Machine Learning*, volume 80 of *Proceedings of Machine Learning Research*, pages 550–559. PMLR, 10–15 Jul 2018.
- [3] J. Pablo Muñoz, Nikolay Lyalyushkin, Yash Akhauri, Anastasia Senina, Alexander Kozlov, and Nilesh Jain. Enabling NAS with automated super-network generation. *CoRR*, abs/2112.10878, 2021.
- [4] Thomas Elsken, Jan Hendrik Metzen, and Frank Hutter. Neural architecture search: A survey, 2019.
- [5] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *2009 IEEE Conference on Computer Vision and Pattern Recognition*, pages 248–255, 2009.
- [6] Pengzhen Ren, Yun Xiao, Xiaojun Chang, Po-Yao Huang, Zhihui Li, Xiaojia Chen, and Xin Wang. A comprehensive survey of neural architecture search: Challenges and solutions, 2021.
- [7] Zichao Guo, Xiangyu Zhang, Haoyuan Mu, Wen Heng, Zechun Liu, Yichen Wei, and Jian Sun. Single path one-shot neural architecture search with uniform sampling, 2020.
- [8] Zhichao Lu, Ian Whalen, Vishnu Boddeti, Yashesh Dhebar, Kalyanmoy Deb, Erik Goodman, and Wolfgang Banzhaf. Nsga-net: Neural architecture search using multi-objective genetic algorithm, 2019.
- [9] Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, and Song Han. Once-for-all: Train one network and specialize it for efficient deployment, 2020.
- [10] Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan, and Song Han. Hat: Hardware-aware transformers for efficient natural language processing. *arXiv preprint arXiv:2005.14187*, 2020.
- [11] Ben Feng, Dayiheng Liu, and Yanan Sun. *Evolving Transformer Architecture for Neural Machine Translation*, page 273–274. Association for Computing Machinery, New York, NY, USA, 2021.
- [12] Zhichao Lu, Kalyanmoy Deb, Erik Goodman, Wolfgang Banzhaf, and Vishnu Naresh Boddeti. Nsganetv2: Evolutionary multi-objective surrogate-assisted neural architecture search, 2020.
- [13] Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, and TAMT Meyarivan. A fast and elitist multiobjective genetic algorithm: Nsga-ii. *IEEE transactions on evolutionary computation*, 6(2):182–197, 2002.
- [14] J. Blank and K. Deb. pymoo: Multi-objective optimization in python. *IEEE Access*, 8:89497–89509, 2020.
- [15] Annibale Panichella. An adaptive evolutionary algorithm based on non-euclidean geometry for many-objective optimization. In *Proceedings of the Genetic and Evolutionary Computation Conference*, GECCO '19, page 595–603, New York, NY, USA, 2019. Association for Computing Machinery.
- [16] Colin White, Arber Zela, Binxin Ru, Yang Liu, and Frank Hutter. How powerful are performance predictors in neural architecture search?, 2021.
- [17] Kishore Papineni, Salim Roukos, Todd Ward, and Wei Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In *Proceedings of the 40th annual meeting of the Association for Computational Linguistics*, pages 311–318, 10 2002.
- [18] Eckart Zitzler and Lothar Thiele. Multiobjective evolutionary algorithms: a comparative case study and the strength pareto approach. *IEEE transactions on Evolutionary Computation*, 3(4):257–271, 1999.
- [19] Joseph Mellor, Jack Turner, Amos Storkey, and Elliot J. Crowley. Neural architecture search without training, 2021.
- [20] Hayeon Lee, Sewoong Lee, Song Chong, and Sung Ju Hwang. Help: Hardware-adaptive efficient latency prediction for nas via meta-learning, 2021.

