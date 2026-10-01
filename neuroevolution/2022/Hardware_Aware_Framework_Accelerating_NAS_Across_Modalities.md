---

# A HARDWARE-AWARE FRAMEWORK FOR ACCELERATING NEURAL ARCHITECTURE SEARCH ACROSS MODALITIES

---

**Daniel Cummings**

Intel Labs, Intel Corporation  
daniel.cummings@intel.com

**Anthony Sarah**

Intel Labs, Intel Corporation  
anthony.sarah@intel.com

**Sharath Nittur Sridhar**

Intel Labs, Intel Corporation  
sharath.nittur.sridhar@intel.com

**Maciej Szankin**

Intel Labs, Intel Corporation  
maciej.szankin@intel.com

**Juan Pablo Munoz**

Intel Labs, Intel Corporation  
pablo.munoz@intel.com

**Sairam Sundaresan**

Intel Labs, Intel Corporation  
sairam.sundaresan@intel.com

## ABSTRACT

Recent advances in Neural Architecture Search (NAS) such as one-shot NAS offer the ability to extract specialized hardware-aware sub-network configurations from a task-specific super-network. While considerable effort has been employed towards improving the first stage, namely, the training of the super-network, the search for derivative high-performing sub-networks is still under-explored. Popular methods decouple the super-network training from the sub-network search and use performance predictors to reduce the computational burden of searching on different hardware platforms. We propose a flexible search framework that automatically and efficiently finds optimal sub-networks that are optimized for different performance metrics and hardware configurations. Specifically, we show how evolutionary algorithms can be paired with lightly trained objective predictors in an iterative cycle to accelerate architecture search in a multi-objective setting for various modalities including machine translation and image classification.

## 1 Introduction

Artificial intelligence researchers are continually pushing the state-of-the-art in deep learning model performance across many application domains. Neural architecture search (NAS) has become an increasingly popular technique to achieve these performance gains with results that often outperform hand-designed architectures. In many cases, the deep neural network (DNN) design and evaluation process is tied to the hardware platform available to the researcher at the time (e.g., GPU). Furthermore, the researcher may have only been interested in a single performance objective such as accuracy when evaluating the network. Therefore, the network is inherently optimized for a specific hardware platform and specific objective. However, users wanting to solve the same problem for which the network was designed may have different hardware platforms available and may be interested in multiple performance metrics (e.g., accuracy and latency). The performance of the network provided by the researcher is then sub-optimal for these users.

Unfortunately, optimizing the network for the user’s hardware and performance objectives is a time-consuming effort requiring highly specialized knowledge. Network optimization is typically done manually with a great deal of in-depth understanding of the hardware platform since certain hardware characteristics (e.g., clock speed, number of logical processor cores, amount of RAM, architecture) will affect the optimization process. The optimization process is also affected by the characteristics of the input data to the DNN (e.g., batch size, image size). Finally, any change to the performance objectives (e.g., going from latency to power consumption), input data characteristics, hardware characteristics, or hardware platform (e.g., going from GPU to CPU) would require starting this expensive optimization process again. With this in mind, we adopt the weight-sharing super-network NAS approach that is well suited to the hardware-aware model optimization task. This NAS approach generates a highly-diverse set of architectural configurations (a.k.a. sub-networks) from a reference super-network architecture and allows us to explore an extremely large search space to find optimal models for a particular hardware and objective setting. Additionally, this approach offers insight into what constitutes an optimal model for a given hardware setting.The main contribution of this work is the creation of a generalizable NAS framework that offers a variety of search algorithms, performance predictor solutions in a variety of objective optimization settings across several modalities. Additionally, we demonstrate how pairing evolutionary algorithms in an iterative fashion with lightly trained performance predictors can yield an accelerated and less costly exploration of a DNN architectural design space across the modalities of machine translation, recommendation, and image classification.

## 2 Related Work

The computational overhead of evaluating DNN architectures during NAS can be very costly due to the training and validation cycles. To address the training overhead, novel weight-sharing approaches known as one-shot or super-networks [Liu et al., 2018, Bender et al., 2018] have offered a way to mitigate the training overhead by reducing training times from thousands to a few GPU days [Elsken et al., 2019]. These approaches train a task-specific super-network architecture with a weight-sharing mechanism that allows the sub-networks to be treated as unique individual architectures. This enables sub-network model extraction and validation without a separate training cycle. However, the validation component still comes with a high overhead since there are many possible sub-networks which may be found from large super-networks (e.g., search space size of  $\sim 10^{19}$ ) and the validation step itself comes with a computational cost, especially for larger datasets such as ImageNet [Deng et al., 2009]. One popular way to mitigate the validation cost in one-shot networks is to train predictors for objectives such as inference time (a.k.a. latency) and accuracy from a training set with thousands of sampled architectures [Cai et al., 2019].

Tangentially, some approaches iterate training and search in order to fine-tune the super-network during training [Lu et al., 2021]. However, the fine-tuning of these approaches is influenced by the hardware platform used during search which could require training to be redone if the resulting super-network were deployed on a different platform (e.g., trained on GPU, deployed onto Raspberry Pi). To further address validation costs, novel approaches in NAS without training [Mellor et al., 2021] and meta-learning [Lee et al., 2021] offer promising solutions but are designed for a single-objective settings whereas in this work we are interested in the multi-objective setting for obtaining hardware-performance trade-offs.

In this work, we focus on demonstrating evolutionary algorithm (EA) and sequential model-based optimization (SMBO) approaches that are known to pair well with weight-sharing-based search spaces. Genetic algorithms, a subset of EA, have been broadly applied for image classification NAS problems in both single-objective implementations [Guo et al., 2020] and in multi-objective approaches such as NSGA-Net [Lu et al., 2019]. We note that reinforcement learning (RL) and gradient optimization have found success in the NAS field as well [Ren et al., 2021].

**Limitations** From a framework perspective, we focus on accelerating the post-training sub-network search process, not the optional fine-tuning stage. After promising DNN architectures are discovered with NAS, a user may achieve state-of-the-art performance for a particular performance bound (e.g., top-1 accuracy for a specific latency or MACs range) by finding the right combination of fine-tuning tactics or by completely re-training the sub-network from scratch [Wu et al., 2021]. Additionally, we do not evaluate the search performance of RL-based NAS and note Liu et al. [2017] found comparable performance to EAs.

**Broader Impact** We do not anticipate that our work will have negative societal impacts. Our work leverages the one-shot weight sharing NAS paradigm which inherently provides massive savings in computation resources resulting in lower CO<sub>2</sub> emissions [Cai et al., 2019]. Moreover the computational cost (i.e., energy consumption) is further reduced by our approach to accelerate the architecture search process although it remains non-trivial. In terms of data, the datasets in this work such as are openly available and have been widely used in previous research (Table 1).

## 3 Search Methodology

### 3.1 Framework Description

Given the growing popularity of super-network DNN architectures across a plethora of machine learning problem domains, we describe a flexible hardware-aware super-network search framework in Figure 1. For an arbitrary super-network reference architecture, modality, and task, our system flow automates the architecture search process and discovers sub-networks that are optimal for a set of one or more performance objectives (e.g., accuracy, latency, MACs, etc.). The primary goal of this framework is to reduce the number of validation measurements (not predictions) that are required to find optimal DNN architectures given a set of performance objectives and hardware platform.

The simplest *validation only* search method performs a validation measurement for sub-networks identified by the search algorithm and hence comes with a high computational cost. Even in the weight-sharing super-network NASFigure 1: Generalizable framework for accelerating super-network type neural architecture search.

context, a validation measurement still requires a non-trivial amount of time and computational resources. The framework also offers a *one-shot* predictor approach to reduce the validation cost overhead as described in the related work. In addition to these methods, we propose a lightweight iterative NAS (LINAS) method described in Section 3.5 that builds on the idea that lightly trained predictors can yield useful information for ranking sub-network configurations. The LINAS method increases the probability that optimal architectures will be identified in early stages of the search and avoids the upfront validation cost of the one-shot predictor approach. Our framework is built on top of the pymoo<sup>1</sup> [Blank and Deb, 2020] and Optuna<sup>2</sup> [Akiba et al., 2019] optimization libraries for reproducibility and ease of future algorithmic enablement.

### 3.2 Super-Network Modalities

The majority of NAS research efforts have focused on the computer vision task of image classification and only recently have other modalities, such as the rapidly growing field of language modeling or language translation, been investigated in detail (Wang et al. [2020], Feng et al. [2021]). Subsequently, understanding how NAS approaches generalize across modalities has not been studied in depth. In the study of our framework, our experiments encompass the modalities of image classification, machine translation, and recommendation as shown in Table 1.

For the modality of image classification, we leverage two super-networks derived from MobileNetV3 [Howard et al., 2019] and ResNet50 [He et al., 2016] which are described in Once-for-all (OFA) [Cai et al., 2019]. OFA employs a progressive shrinking method during super-network training resulting in *elastic* design parameters that can represent the full architectural search space. For additional variety in this domain, we use recent work by Muñoz et al. [2021], called BootstrapNAS, and discuss the sub-network search process for the quantized INT8 space in the Appendix.

Super-network approaches have recently been applied in the domain of Natural Language Processing (NLP). Hardware-aware Transformers (HAT) [Wang et al., 2020] achieve this goal by extending the network elasticity type of weight sharing approach to this domain. In HAT, the authors introduce *arbitrary encoder-decoder attention*, to break the information bottleneck between the encoder and decoder layers in Transformers [Vaswani et al., 2017]. Additionally, they propose heterogenous transformer layers to allow for different layers to have different parameters.

Neural Collaborative Filtering (NCF) [He et al., 2017], a popular method for recommendation problems, combines the benefits of traditional matrix factorization and fully connected neural networks. We adapt this model architecture into an elastic training framework similar to HAT wherein each embedding layer and dense layer is fully elastic.

<sup>1</sup><https://pymoo.org>

<sup>2</sup><https://optuna.org>Table 1: Summary of super-networks and associated design search spaces used in this work. Additional details for each super-network are provided in Appendix D.

<table border="1">
<thead>
<tr>
<th>Super-Network</th>
<th>Task</th>
<th>Dataset</th>
<th>Number Format</th>
<th>Objectives</th>
<th>Search Space Size (unique DNNs)</th>
</tr>
</thead>
<tbody>
<tr>
<td>MobileNetV3</td>
<td>Image Classification</td>
<td>ImageNet</td>
<td>FP32</td>
<td>Top-1 Accuracy, Latency</td>
<td><math>\sim 10^{19}</math></td>
</tr>
<tr>
<td>ResNet50</td>
<td>Image Classification</td>
<td>ImageNet</td>
<td>FP32</td>
<td>Top-1 Accuracy, Latency</td>
<td><math>\sim 10^{13}</math></td>
</tr>
<tr>
<td>Transformer</td>
<td>Machine Translation</td>
<td>WMT 2014 En-De</td>
<td>FP32</td>
<td>BLEU Score, Latency</td>
<td><math>\sim 10^{15}</math></td>
</tr>
<tr>
<td>NCF</td>
<td>Recommendation</td>
<td>Pinterest-20</td>
<td>FP32</td>
<td>HR@10, Latency</td>
<td><math>\sim 10^7</math></td>
</tr>
</tbody>
</table>

### 3.3 Search Space Encoding

A key consideration of the super-network NAS process is encoding a representation of the architectural design variables in a way that is useful for the search algorithms. For illustration, we summarize our encoding strategy for the MobileNetV3 and Transformer super-networks in Figure 2. By mapping each super-network architecture design variable to several integer options, this search space encoding offers a compatible interface for the evolutionary operators (e.g., mutation, crossover, etc.).

(a) MobileNetV3 design variable encoding.

<table border="1">
<tr>
<td>D = {2,3,4}</td>
<td colspan="2">2</td>
<td colspan="2">4</td>
<td colspan="2">3</td>
</tr>
<tr>
<td>K = {3,5,7}</td>
<td>3</td>
<td>3</td>
<td>X</td>
<td>X</td>
<td>3</td>
<td>5</td>
</tr>
<tr>
<td>W = {3,4,6}</td>
<td>4</td>
<td>3</td>
<td>X</td>
<td>X</td>
<td>4</td>
<td>6</td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>4</td>
<td>3</td>
</tr>
</table>

(b) Transformer design variable encoding.

<table border="1">
<tr>
<td>EDim</td>
<td>Encoder Layer 1</td>
<td>Encoder Layer 2</td>
<td>...</td>
<td>Encoder Layer 6</td>
<td>EDim</td>
<td>Decoder Layer 1</td>
<td>...</td>
<td>Decoder Layer j</td>
</tr>
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
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>4</td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>2048</td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>8</td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>3072</td>
</tr>
</table>

Figure 2: Super-network encoding strategies for MobileNetV3 and Transformer each having 45 and 40 design variables respectively.

### 3.4 Predictors

Since validation evaluations of performance objectives, such as top-1 accuracy and latency, require a large amount of time, we follow the work in [Cai et al., 2019] and [Wang et al., 2020] and employ predictors. More specifically, we predict top-1 accuracy of sub-networks derived from MobileNetV3 and ResNet50 super-networks, hit ratio (HR@10) of sub-networks derived from NCF super-networks, bilingual evaluation understudy (BLEU) [Papineni et al., 2002] score of sub-networks derived from Transformer super-networks and latency of sub-networks derived from all super-networks.

However, unlike prior work which use multi-layer perceptrons (MLPs) to perform prediction, we employ much simpler methods such as ridge regression, support vector machine regression (SVR) and stacked regression predictors. The authors of [Lu et al., 2021] and [Laube et al., 2022] have found that MLPs are inferior to other methods of prediction for low training example counts. We have found that these simpler methods converge more quickly and require both fewer training examples and much less hyper-parameter optimization than MLPs. The combination of performance objective prediction via simple predictors allows us to significantly accelerate the selection of sub-networks with minimal prediction error. See Section 4.1 for an analysis of our predictor performance.

### 3.5 Search Algorithms

The foundational goal of NAS is to find DNN architectures that are optimal for one or more performance objectives. In the context of weight-sharing super-networks, consider a pre-trained super-network with weights  $W$ , a set of sub-network architectural configurations  $\Omega$  derived from the super-network and  $m$  competing objectives  $f_1(\omega; W), \dots, f_m(\omega; W)$  where  $\omega \in \Omega$ . Each of the sub-network configurations  $\omega$  is a valid set of parametersused during training of the super-network. For example, a given  $\omega$  will contain values for each design parameter (e.g., depth and kernel size) used during super-network training. Our system aims to minimize a subset of objectives  $S_i \subseteq \{f_1(\omega; W), \dots, f_m(\omega; W)\}$  to discover the near-optimal sub-network  $\omega_i^*$ . In other words,

$$\omega_i^* = \underset{\omega \in \Omega}{\operatorname{argmin}}(S_i) \quad (1)$$

An objective can be negated to transform a minimization objective into a maximization objective (e.g., accuracy is a maximization objective). During optimization, multiple architectures  $\omega_i^* \in \Omega$  will be scored in the objective space allowing for the identification of a *Pareto front*, as illustrated in Figure 4.

In this work we focus on examining random search, multi-objective sequential model-based optimization (SMBO), and multi-objective evolutionary algorithm (MOEA) approaches to the sub-network search problem. From a hardware-aware standpoint, we evaluate in the multi-objective (a.k.a. bi-objective) setting as we are focused on finding a highly diverse set of near-optimal architectures across the accuracy and latency trade-off (Pareto front) region. However, we note that our framework works with any number of objectives. To test a SMBO algorithm in our framework we employ the multi-objective tree-structured parzen estimator (MOTPE) as proposed by Ozaki et al. [2020]. From the Pareto-based MOEA category, the framework supports the popular NSGA-II [Deb et al., 2002] algorithm and a similar approach called AGE-MOEA [Panichella, 2019]. For indicator- and decomposition-based MOEAs we support U-NSGA-II [Deb and Sundar, 2006], MOEA/D [Zhang and Li, 2007], and CTAEA [Li et al., 2019].

One of the primary goals for our framework is to reduce the number of validation measurements that are required to find optimal DNN architectures in a multi-objective search space that works well across modalities. While related work shows that using trained predictors can speed up the DNN architecture search process, there remains a substantial cost to training the predictors since the number of validated training samples can range between 1000 and 16,000 [Lu et al., 2020]. Interestingly, as shown in Figure 3, simple accuracy predictors can achieve acceptable mean absolute percentage error (MAPE) with far fewer training samples. We build on this insight that lightly trained predictors can offer a useful surrogate signal during search. Algorithm 1 describes our generalizable Lightweight Iterative NAS (LINAS) method. We first randomly sample the architecture search space to serve as the initial validation population. For each sub-network in the validation population, we measure each objective and store the result. These results are combined with all previous validation results and are used to train the objective predictors. For each iteration, we run a multi-objective algorithm search (e.g., NSGA-II) using that iteration’s trained predictors for a high number of generations (e.g.,  $> 100$ ) to allow the algorithm to explore the predicted objective space sufficiently. This predictor-based search runs very quickly since no validation measurements occur. Finally, we select the most optimal population of diverse DNN architectures from the predictor-based search to add to the next validation population, which then informs the next round of predictor training. This cycle continues until the iteration count limit is met or an end-user decides a sufficient set of architectures has been discovered. We note that the LINAS approach can be applied with any single-, multi-, or many-objective EA and generalizes to work with any super-network framework. Additionally, it allows for the interchanging of tuning parameters (e.g., crossover, mutation, population), EAs, and predictor types for each iteration.

---

**Algorithm 1** Generalizable Lightweight Iterative Neural Architecture Search (LINAS)

---

**Input:** Objectives  $f_m$ , super-network with weights  $\mathcal{W}$  and configurations  $\Omega$ , predictor model for each objective  $Y_m$ , LINAS population  $P$  size  $n$ , number of LINAS iterations  $I$ , evolutionary algorithm  $\mathcal{E}$  with number of evaluations  $J$ .  
 $P_{i=0} \leftarrow \{\omega_n\} \in \Omega$  // sample  $n$  sub-networks for first population  
**while**  $i++ < I$  **do**  
     $D_{i,m} \leftarrow f_m(P_i \in \Omega; \mathcal{W})$  // measure objectives  $f_m$ , store data  $D_{i,m}$   
     $D_{all,m} \leftarrow D_{all,m} \cup D_{i,m}$   
     $Y_{m,pred} \leftarrow Y_{m,train}(D_{all,m})$  // train predictors for each objective  
    **while**  $j++ < J$  **do**  
         $P_{\mathcal{E}_j} \leftarrow \mathcal{E}(Y_{m,pred}, j)$  // run  $\mathcal{E}$  for  $J$  evaluations  
    **end while**  
     $P_i \leftarrow P_{\mathcal{E},best\_unique} \in P_{\mathcal{E}_j}$  // retrieve optimal and unique population of sub-networks  
**end while**  
**Output:** All validated sub-networks configurations  $P_I$ , predictor search results  $P_{\mathcal{E}_{I,J}}$ , and validation data  $D_{all,m}$ .

---## 4 Experiments & Results

In the hardware-aware NAS context, latency is a highly important metric since it directly relates to the real-time performance of a hardware system. Often MACs, FLOPs, or model parameter counts are used as an approximation of latency but do not guarantee correlation (see Appendix E). We use the term “hardware-aware” to emphasize the focus on using latency as one of our main objectives but do not use any hardware architectural information to inform the search process. In this work we experiment on CPU, GPU, and mobile device platforms for evaluating our framework and we describe the transferability of NAS results between CPU and GPU platforms in Appendix B. Since our experiments measure latency values from different manufacturers and there are possible proprietary issues in sharing what could be perceived as official benchmark data, we normalize latency results to be within  $[0, 1]$ . More specifically, the normalized latency  $\hat{l}$  is given by  $\hat{l} = \frac{l - l_{min}}{l_{max}} \in [0, 1]$  where  $l$  is the unnormalized latency,  $l_{min}$  is the minimum unnormalized latency and  $l_{max}$  is the maximum unnormalized latency. Using normalized latency does not change the underlying search results we are demonstrating. For comparative latency performance metrics related to our test platforms, we point the reader to the MLCommons<sup>3</sup> benchmark suite.

### 4.1 Predictors

As described in Section 3.4, our work makes extensive use of predictors to accelerate the selection of sub-networks, particularly when applying LINAS. Predictors are necessary since performing actual measurements of performance objectives such as accuracy or latency would be prohibitively slow. In light of their importance, a better understanding of their performance is needed.

The analysis of predictors is performed over a number of different trials to account for variance in the results. In each trial, the data set for each predictor is first split into train and test sets. Subsets of the train data set within the range of 100 to 1000 examples are used to train the predictor. For a given trial, the *same* test set with 500 examples is used to compute the prediction mean absolute percentage error (MAPE). This process is repeated for a total of 100 trials and the results averaged to compute the MAPE shown in Figure 3.

The top row of Figure 3 shows the MAPE of different predictors for each super-network type in Table 1. The *stacked* predictor is a combination of ridge and SVR (RBF) regressors which “stacks” the predictions from each of these two regressors and uses them as the input to a final ridge regressor. The bottom row shows the correlation between actual and predicted values after training the stacked predictor with 1000 examples. Note that the Kendall rank correlation coefficient  $\tau$  is also shown for each case. In all cases, these simple predictors provide small error (maximum MAPE of 0.91%) and high correlation (minimum  $\tau$  of 0.8348) with actual values. Results for latency prediction are shown in Appendix C.

### 4.2 Search Results

The main purpose of our framework and proposed LINAS approach is to reduce the total number of validation measurements required to find optimal DNN architectures in the multi-objective space for any modality or domain-specific task. In other words, the goal of LINAS is to maximize the hypervolume while minimizing the number of evaluations during NAS. Specifically, we want to efficiently discover architectures with optimal trade-offs in high top-1 accuracy/BLEU/HR@10 and low latency. Since we are interested in evaluating the performance of various search algorithms in the multi-objective setting, we use the hypervolume indicator [Zitzler and Thiele, 1999] as shown in Figure 4. When measuring two objectives, the hypervolume term represents the dominated *area* of the Pareto front. In our experiments we use the term evaluation to refer to an actual validation measurement, not a predicted measurement.

We start our experimental analysis using the MobileNetV3 super-network since it offers the largest search space size (e.g.,  $10^{19}$ ). Figures 5a and 5b illustrate the differences in how LINAS (with NSGA-II for the internal predictor loop), random search, and NSGA-II progress in the multi-objective search space. For the same evaluation count of 250, while NSGA-II begins progressing towards an optimal trade-off region, the LINAS results show how the exploration can be accelerated. Moreover, one can see how both approaches perform better than random search.

Since MobileNetV3 is an OFA super-network, we run the genetic algorithm (GA) search as used in the OFA paper and show the results in Figure 5c for comparison. This approach follows the one-shot predictor method as shown in Figure 1 where predictors for the objectives are trained with 1000 samples before the search starts in this setup. The search then runs a large amount of predictor-based evaluations in the latency range of interest. The intent of the OFA GA search algorithm is to maximize the accuracy for a particular latency constraint. In the multi-objective setting this has a few limitations such as needing prior knowledge of the latency space and requiring a user to manually define separate

---

<sup>3</sup><https://mlcommons.org>Figure 3: MAPE of predictors performing top-1 accuracy, BLEU score and HR@10 prediction versus the number of training examples for sub-networks derived from the super-networks shown in Table 1 (top row). Correlation and Kendall  $\tau$  coefficient between actual and predicted values after training the stacked predictor with 1000 examples (bottom row). The ideal correlation is shown by the green line.

Figure 4: Illustration of the hypervolume metric in a bi-objective space.

search groups across the known latency range that are unique to each hardware platform. In our Figure 5c example we define four search groups (each with unique latency constraints) and note that LINAS only requires 250 evaluations to find a more diverse Pareto front compared to the GA search from the OFA paper which uses 1000 evaluations to build predictors. A key takeaway is that LINAS can be used to extend the search capabilities of any super-network or weight-sharing NAS framework in the multi-objective setting.

LINAS offers a great deal of flexibility in terms predictor and algorithm options for the internal loop. Figure 6a shows a LINAS specific ablation study using the various EA algorithms for the internal predictor loop including the performance of various algorithms without LINAS. We find that Pareto based MOEAs such as NSGA-II and AGE-MOEA and the indicator based U-NSGA-III perform well for this task. MOTPE by itself finds good sub-networks in the very early stage of the search process but suffers from very high run-times for evaluation counts above 500. This limits the ability of MOTPE to efficiently be used in the LINAS internal predictor loop since it will not approach the near-optimal Pareto region in the predictor space.

Figure 7a highlights that the choice of the underlying predictor algorithm has little impact on the performance of LINAS. Next, in Figure 7b we compare various LINAS runs with different population sizes where a population of 50 gives the best performance for the MobileNetV3 super-network. Finally, we note that while the intent of LINAS is to run for the fewest number of evaluations as possible, an extended run shows that it would take NSGA-II a significant amount of evaluations to catch up with the LINAS hypervolume at 20,000 evaluations. For the subsequent experiments and consistency, we compare LINAS (with NSGA-II for the internal predictor loop) against validation-Figure 5: Search results in the MobileNetV3 search space (Titan-V GPU, batch size = 128) comparing a: LINAS, and b: NSGA-II approaches (algorithm settings in Table 5). Sub-figure c illustrates the GA approach used in the OFA paper that uses predictors trained from 1000 evaluations where we run four different latency constrained searches.

Figure 6: Comparison of search algorithms in the MobileNetV3 design space for hypervolume (top-1 accuracy and latency) versus evaluation count. Shaded regions show the standard error for 5 trials with different random seeds. Search parameter settings in Appendix G.

only measurements from a random search that uniformly samples the architecture space and NSGA-II itself using the algorithm and the predictor settings in Table 5.

Figure 7: Hypervolume ablations studies on MobileNetV3 (Titan-V GPU, batch=128). Shaded regions show the standard error for 5 trials with different random seeds.

In this work, we note that each hardware platform has very specific DNN inference time or latency characteristics and a benefit of LINAS is that it can be used to run an accelerated NAS process without any prior knowledge of the latency range. Figure 8 shows a consistent behavior for LINAS across GPU, CPU and mobile hardware settings. When considering the performance of LINAS across various modalities as shown in Figure 9, a key observation is how quickly LINAS accelerates to a better hypervolume versus the baseline NSGA-II and random search. Depending on which region of the Pareto front is most important, an end-user would be more likely to identify optimal architectures in fewer evaluations with LINAS. Evaluation (validation measurement) counts directly correlate to the search timesince the evolutionary algorithm runtime component is far smaller than evaluation runtimes (compute time breakdown given in Appendix A). Given the characteristics of the Transformer and NCF objective spaces, the LINAS result is less differentiated than in the image classification cases. We found that since the distribution of the sub-networks in these super-networks is both more constrained in range and occurs closer to an optimal region, one would be more likely to randomly find a good performing sub-network than in the MobileNetV3 or ResNet50 search spaces. Specifically, NCF is heavily biased towards matrix factorization as described by Rendle et al. [2020] and we discuss this more in Appendix D. Nevertheless, LINAS provides some benefit for all super-network types.

Figure 8: Search comparison on various hardware platforms for the MobileNetV3 super-network.

Figure 9: Hypervolume comparison for LINAS, NSGA-II, and random search across modalities (Titan-V GPU). Shaded regions = standard error for 5 trials. Search parameter settings in Table 5.

## 5 Conclusion

We have proposed and demonstrated a NAS framework and LINAS algorithm that efficiently finds a *diverse* multi-objective Pareto set of sub-networks from a pre-trained super-network for various modalities. As NAS research continues to gain momentum, we highlight the need to continue to investigate the generalizability of NAS approaches in modalities outside of computer vision. Future work will include extending to experiments to a larger variety of hardware platforms (e.g., TinyML) and application specific deep learning accelerators. Additionally, we plan to explore NAS without training and meta-learning approaches to further reduce the evaluation overhead.

## References

Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. Optuna: A next-generation hyperparameter optimization framework. In *Proceedings of the 25rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2019.

Gabriel Bender, Pieter-Jan Kindermans, Barret Zoph, Vijay Vasudevan, and Quoc Le. Understanding and simplifying one-shot architecture search. In Jennifer Dy and Andreas Krause, editors, *Proceedings of the 35th International Conference on Machine Learning*, volume 80 of *Proceedings of Machine Learning Research*, pages 550–559. PMLR, 07 2018. URL <https://proceedings.mlr.press/v80/bender18a.html>.

J. Blank and K. Deb. pymoo: Multi-objective optimization in python. *IEEE Access*, 8:89497–89509, 2020.

Julian Blank, Kalyanmoy Deb, Yashesh Dhebar, Sunith Bandaru, and Haitham Seada. Generating well-spaced points on a unit simplex for evolutionary many-objective optimization. *IEEE Transactions on Evolutionary Computation*, 25(1):48–60, 2021. doi: 10.1109/TEVC.2020.2992387.Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, and Song Han. Once-for-all: Train one network and specialize it for efficient deployment. *arXiv preprint arXiv:1908.09791*, 2019.

Kalyanmoy Deb and J. Sundar. Reference point based multi-objective optimization using evolutionary algorithms. In *Proceedings of the 8th Annual Conference on Genetic and Evolutionary Computation*, GECCO '06, page 635–642, New York, NY, USA, 2006. Association for Computing Machinery. ISBN 1595931864. doi: 10.1145/1143997.1144112. URL <https://doi.org/10.1145/1143997.1144112>.

Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, and TAMT Meyarivan. A fast and elitist multiobjective genetic algorithm: Nsga-ii. *IEEE transactions on evolutionary computation*, 6(2):182–197, 2002.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *2009 IEEE Conference on Computer Vision and Pattern Recognition*, pages 248–255, 2009. doi: 10.1109/CVPR.2009.5206848.

Thomas Elskens, Jan Hendrik Metzen, and Frank Hutter. Neural architecture search: A survey, 2019.

Ben Feng, Dayiheng Liu, and Yanan Sun. *Evolving Transformer Architecture for Neural Machine Translation*, page 273–274. Association for Computing Machinery, New York, NY, USA, 2021. ISBN 9781450383516. URL <https://doi.org/10.1145/3449726.3459441>.

Zichao Guo, Xiangyu Zhang, Haoyuan Mu, Wen Heng, Zechun Liu, Yichen Wei, and Jian Sun. Single path one-shot neural architecture search with uniform sampling, 2020.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 06 2016. doi: 10.1109/cvpr.2016.90. URL <http://dx.doi.org/10.1109/cvpr.2016.90>.

Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. Neural collaborative filtering. In *Proceedings of the 26th international conference on world wide web*, pages 173–182, 2017.

Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, et al. Searching for mobilenetv3. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 1314–1324, 2019.

Kevin Alexander Laube, Maximus Mutschler, and Andreas Zell. What to expect of hardware metric predictors in NAS, 2022. URL <https://openreview.net/forum?id=2DJn3E7lXu>.

Hayeon Lee, Sewoong Lee, Song Chong, and Sung Ju Hwang. Help: Hardware-adaptive efficient latency prediction for nas via meta-learning, 2021.

Ke Li, Renzhi Chen, Guangtao Fu, and Xin Yao. Two-archive evolutionary algorithm for constrained multiobjective optimization. *IEEE Transactions on Evolutionary Computation*, 23(2):303–315, 2019. doi: 10.1109/TEVC.2018.2855411.

Hanxiao Liu, Karen Simonyan, Oriol Vinyals, Chrisantha Fernando, and Koray Kavukcuoglu. Hierarchical representations for efficient architecture search. *CoRR*, abs/1711.00436, 2017. URL <http://arxiv.org/abs/1711.00436>.

Hanxiao Liu, Karen Simonyan, and Yiming Yang. DARTS: differentiable architecture search. *CoRR*, abs/1806.09055, 2018. URL <http://arxiv.org/abs/1806.09055>.

Zhichao Lu, Ian Whalen, Vishnu Boddeeti, Yashesh Dhebar, Kalyanmoy Deb, Erik Goodman, and Wolfgang Banzhaf. Nsga-net: Neural architecture search using multi-objective genetic algorithm, 2019.

Zhichao Lu, Kalyanmoy Deb, Erik Goodman, Wolfgang Banzhaf, and Vishnu Naresh Boddeeti. Nsganetv2: Evolutionary multi-objective surrogate-assisted neural architecture search, 2020.

Zhichao Lu, Gautam Sree Kumar, Erik Goodman, Wolfgang Banzhaf, Kalyanmoy Deb, and Vishnu Naresh Boddeeti. Neural architecture transfer. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 43(9):2971–2989, 09 2021. ISSN 1939-3539. doi: 10.1109/tpami.2021.3052758. URL <http://dx.doi.org/10.1109/TPAMI.2021.3052758>.

Joseph Mellor, Jack Turner, Amos Storkey, and Elliot J. Crowley. Neural architecture search without training, 2021.

J. Pablo Muñoz, Nikolay Lyalyushkin, Yash Akhauri, Anastasia Senina, Alexander Kozlov, and Nilesh Jain. Enabling NAS with automated super-network generation. *CoRR*, abs/2112.10878, 2021. URL <https://arxiv.org/abs/2112.10878>.

Yoshihiko Ozaki, Yuki Tanigaki, Shuhei Watanabe, and Masaki Onishi. Multiobjective tree-structured parzen estimator for computationally expensive optimization problems. In *Proceedings of the 2020 Genetic and Evolutionary Computation Conference*, GECCO '20, page 533–541, New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450371285. doi: 10.1145/3377930.3389817. URL <https://doi.org/10.1145/3377930.3389817>.Annibale Panichella. An adaptive evolutionary algorithm based on non-euclidean geometry for many-objective optimization. In *Proceedings of the Genetic and Evolutionary Computation Conference*, GECCO '19, page 595–603, New York, NY, USA, 2019. Association for Computing Machinery. ISBN 9781450361118. doi: 10.1145/3321707.3321839. URL <https://doi.org/10.1145/3321707.3321839>.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In *Proceedings of the 40th annual meeting of the Association for Computational Linguistics*, pages 311–318, 10 2002. doi: 10.3115/1073083.1073135.

Pengzhen Ren, Yun Xiao, Xiaojun Chang, Po-Yao Huang, Zhihui Li, Xiaojia Chen, and Xin Wang. A comprehensive survey of neural architecture search: Challenges and solutions, 2021.

Steffen Rendle, Walid Krichene, Li Zhang, and John Anderson. Neural collaborative filtering vs. matrix factorization revisited. In *Fourteenth ACM conference on recommender systems*, pages 240–248, 2020.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need, 2017.

Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan, and Song Han. Hat: Hardware-aware transformers for efficient natural language processing. *arXiv preprint arXiv:2005.14187*, 2020.

Junru Wu, Xiyang Dai, Dongdong Chen, Yinpeng Chen, Mengchen Liu, Ye Yu, Zhangyang Wang, Zicheng Liu, Mei Chen, and Lu Yuan. Stronger nas with weaker predictors, 2021.

Qingfu Zhang and Hui Li. Moea/d: A multiobjective evolutionary algorithm based on decomposition. *IEEE Transactions on Evolutionary Computation*, 11(6):712–731, 2007. doi: 10.1109/TEVC.2007.892759.

Eckart Zitzler and Lothar Thiele. Multiobjective evolutionary algorithms: a comparative case study and the strength pareto approach. *IEEE transactions on Evolutionary Computation*, 3(4):257–271, 1999.## A Test Platforms and Compute Time

In this work, we use both CPU and GPU platforms for running our experiments. The hardware platforms and their characteristics are shown in Table 2. For the Note10 mobile CPU experiment shown in Figure 8, we use a latency look-up table provided by Cai et al. [2019] since we did not have direct access to that platform.

Table 2: Hardware platforms used for NAS experimentation.

<table border="1">
<thead>
<tr>
<th>Name</th>
<th>Memory</th>
<th>Thread Count (Host CPU)</th>
<th>Microarchitecture (Host CPU)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Intel® Xeon® Platinum 8180</td>
<td>192 GB</td>
<td>56</td>
<td>Skylake (SKX)</td>
</tr>
<tr>
<td>Intel® Xeon® Platinum 8280</td>
<td>192 GB</td>
<td>56</td>
<td>Cascade Lake (CLX)</td>
</tr>
<tr>
<td>NVIDIA® Titan V®</td>
<td>32 GB</td>
<td>32</td>
<td>Cascade Lake (CLX)</td>
</tr>
<tr>
<td>NVIDIA® Tesla® V100</td>
<td>32 GB</td>
<td>32</td>
<td>Skylake (SKX)</td>
</tr>
<tr>
<td>NVIDIA® Tesla® A100</td>
<td>32 GB</td>
<td>32</td>
<td>Cascade Lake (CLX)</td>
</tr>
</tbody>
</table>

In terms of GPU wall clock time required to perform search, we note that there is a wide range of results that would be dependent on the supporting hardware platform configuration. For example, for MobileNetV3, a sub-network search with 2000 evaluations would take approximately 9.5 GPU hours with an evolutionary algorithm run time on the order of minutes. Because the evolutionary algorithm run times are extremely small when compared to validation measurement run times, we view the evaluation count (e.g., Figure 9) as a more universal metric of search time efficiency in this work.

To provide more insights into time complexity of presented algorithms, an extensive set of tests was performed to measure wall-clock time of each algorithm needed to achieve a certain hypervolume threshold on each of the search spaces presented in this work. For each super-network, the hypervolume thresholds were selected based on the maximum hypervolume achieved by random search and NSGA-II for a given search space, respectively. In the latter case, the results for random search are not shown as it never achieved the given hypervolume level within a set number of evaluations. Table 3 shows detailed information on how much time was spent on the model evaluation and the search process itself.

## B Hardware Platform Transferability

One of the key goals of our framework is to accelerate the sub-network search process to address the issue that every hardware platform and/or configuration has unique latency characteristics and therefore unique optimal sub-networks in their respective multi-objective search spaces. To illustrate this behavior, we use the MobileNetV3 super-network where Figure 10 shows that an optimal set of sub-networks found on a CPU platform may not transfer to the optimal objective region on a GPU platform and vice versa. Furthermore, within a hardware platform, Figure 11 shows that sub-network configurations found to be optimal to one CPU hardware configuration (e.g., batch size = 1, thread count = 1), do not transfer optimally to other hardware batch size and thread count configurations.

Figure 10: MobileNetV3 Pareto fronts specialized to GPU (V100) and CPU (CLX) showing that optimal sub-network configurations found on one hardware platform do not translate to the optimal sub-networks for another. Batch size was 128.Table 3: Comparison of algorithms and their average run time on all presented search spaces to a given normalized hypervolume (HV) threshold based on a platform with NVIDIA<sup>®</sup> Titan V<sup>®</sup> (evaluation) and Intel<sup>®</sup> Xeon<sup>®</sup> Platinum 8280 (search).

<table border="1">
<thead>
<tr>
<th>Super-Network</th>
<th>Search Algorithm</th>
<th>Evaluations</th>
<th>Evaluation Cost (GPU Hours)</th>
<th>Search Cost (CPU Hours)</th>
<th>Total Cost (Hours)</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="7">MobileNetV3</td>
<td colspan="5">Normalized HV = 0.810</td>
</tr>
<tr>
<td>LINAS + NSGA-II</td>
<td>100</td>
<td>0.479</td>
<td>0.0095</td>
<td>0.489</td>
</tr>
<tr>
<td>NSGA-II</td>
<td>260</td>
<td>1.247</td>
<td>0.0014</td>
<td>1.248</td>
</tr>
<tr>
<td>Random</td>
<td>2000</td>
<td>9.593</td>
<td>0.0017</td>
<td>9.594</td>
</tr>
<tr>
<td colspan="5">Normalized HV = 0.955</td>
</tr>
<tr>
<td>LINAS + NSGA-II</td>
<td>346</td>
<td>1.746</td>
<td>0.0331</td>
<td>1.779</td>
</tr>
<tr>
<td>NSGA-II</td>
<td>2000</td>
<td>9.593</td>
<td>0.0017</td>
<td>9.594</td>
</tr>
<tr>
<td></td>
<td>Random</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td rowspan="7">ResNet50</td>
<td colspan="5">Normalized HV = 0.800</td>
</tr>
<tr>
<td>LINAS + NSGA-II</td>
<td>57</td>
<td>0.545</td>
<td>0.0047</td>
<td>0.549</td>
</tr>
<tr>
<td>NSGA-II</td>
<td>255</td>
<td>2.438</td>
<td>0.0014</td>
<td>2.439</td>
</tr>
<tr>
<td>Random</td>
<td>1000</td>
<td>9.559</td>
<td>0.0016</td>
<td>9.560</td>
</tr>
<tr>
<td colspan="5">Normalized HV = 0.925</td>
</tr>
<tr>
<td>LINAS + NSGA-II</td>
<td>155</td>
<td>1.481</td>
<td>0.0142</td>
<td>1.496</td>
</tr>
<tr>
<td>NSGA-II</td>
<td>1000</td>
<td>9.559</td>
<td>0.0016</td>
<td>9.560</td>
</tr>
<tr>
<td></td>
<td>Random</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td rowspan="7">Transformer</td>
<td colspan="5">Normalized HV = 0.967</td>
</tr>
<tr>
<td>LINAS + NSGA-II</td>
<td>111</td>
<td>1.886</td>
<td>0.0035</td>
<td>1.890</td>
</tr>
<tr>
<td>NSGA-II</td>
<td>191</td>
<td>3.246</td>
<td>0.0014</td>
<td>3.248</td>
</tr>
<tr>
<td>Random</td>
<td>600</td>
<td>10.197</td>
<td>0.0015</td>
<td>10.199</td>
</tr>
<tr>
<td colspan="5">Normalized HV = 0.997</td>
</tr>
<tr>
<td>LINAS + NSGA-II</td>
<td>465</td>
<td>7.903</td>
<td>0.0156</td>
<td>7.918</td>
</tr>
<tr>
<td>NSGA-II</td>
<td>600</td>
<td>10.197</td>
<td>0.0015</td>
<td>10.199</td>
</tr>
<tr>
<td></td>
<td>Random</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td rowspan="7">NCF</td>
<td colspan="5">Normalized HV = 0.965</td>
</tr>
<tr>
<td>LINAS + NSGA-II</td>
<td>87</td>
<td>2.884</td>
<td>0.0138</td>
<td>2.898</td>
</tr>
<tr>
<td>NSGA-II</td>
<td>89</td>
<td>2.950</td>
<td>0.0014</td>
<td>2.952</td>
</tr>
<tr>
<td>Random</td>
<td>160</td>
<td>5.304</td>
<td>0.0014</td>
<td>5.305</td>
</tr>
<tr>
<td colspan="5">Normalized HV = 0.989</td>
</tr>
<tr>
<td>LINAS + NSGA-II</td>
<td>148</td>
<td>4.906</td>
<td>0.0241</td>
<td>4.930</td>
</tr>
<tr>
<td>NSGA-II</td>
<td>160</td>
<td>5.304</td>
<td>0.0014</td>
<td>5.305</td>
</tr>
<tr>
<td></td>
<td>Random</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
</tr>
</tbody>
</table>

## C Latency Prediction

The analysis of latency prediction is performed in the same way as described in Section 4.1 with the results shown in Figure 12. The top row shows the MAPE of different predictors for each super-network type. The bottom row shows the correlation between actual and predicted latencies after training the stacked predictor with 1000 examples. Note that the Kendall rank correlation coefficient  $\tau$  is also shown for each case.

## D Super-network Details

### D.1 MobileNetV3

For the image classification task with MobileNetV3, we experiment on the ImageNet validation dataset [Deng et al., 2009] and use the pre-trained super-network weights from *ofa\_mbv3\_d234\_e346\_k357\_w1.0*<sup>4</sup>, trained with progressive shrinking. For the architecture design variables, we allow for an elastic layer depth chosen from [2, 3, 4], an elastic width expansion ratio chosen from [3, 4, 6], an elastic kernel size chosen from [3, 5, 7], and use an input image resolution of 224x224. The layer depth can affect the mapping of the kernel size and expansion ratio design variables as shown in Figure 2(a). For more details on this super-network please refer to the work by Cai et al. [2019].

<sup>4</sup><https://github.com/mit-han-lab/once-for-all>Figure 11: MobileNetV3 Pareto fronts with CLX for specialized thread counts/batch sizes, and the non-specialized configurations for comparison.

Figure 12: MAPE of predictors performing latency prediction versus the number of training examples for sub-networks derived from the super-networks shown in Table 1 (top row). Correlation and Kendall  $\tau$  coefficient between actual and predicted latencies after training the stacked predictor with 1000 examples (bottom row). The ideal correlation is shown by the green line.

## D.2 ResNet50

For the image classification task with ResNet50, we use the ImageNet validation dataset [Deng et al., 2009] and use the pre-trained super-network weights from  $ofa\_resnet50\_d=0+1+2\_e=0.2+0.25+0.35\_w=0.65+0.8+1.0^4$ , trained with progressive shrinking for our experiments. For the architecture design variables, we allow for an elastic layer depth chosen from  $[0, 1, 2]$ , an elastic width expansion ratio chosen from  $[0.65, 0.8, 1.0]$ , an elastic expansion ratio chosen from  $[0.2, 0.25, 0.35]$ , and use an input image resolution of 224x224.

## D.3 Transformer

For the machine translation task, we mainly experiment on the WMT 2014 En-De data set. We follow a similar pre-processing technique proposed in [Wang et al., 2020] for the data. Similar to [Wang et al., 2020], we use the search space with an embedding dimension chosen from  $[512, 640]$ , hidden dimension from  $[1024, 2048, 3072]$ , attention head number from  $[4, 8]$ , decoder layer number from  $[1, 2, 3, 4, 5, 6]$  and a constant encoder layer number of  $[6]$ . In [Wang et al., 2020], although the authors use the inherited weights from the Transformer super-network, for the evolutionary search, they re-train the sub-networks from scratch in the final results. In our results, we do not re-trainthese networks from scratch. Additionally, we train the predictor directly on the *bilingual evaluation understudy (BLEU)* [Papineni et al., 2002] score. For the BLEU score evaluation, we use a beam size of 5 and a length penalty of 0.6.

#### D.4 NCF

For the recommendation task, we experiment on the Pinterest-20 dataset and follow a similar pre-processing technique used in [He et al., 2017]. We use the *Neural Matrix Factorization* (NeuMF) model from [He et al., 2017], which is a fusion of *Generalized Matrix Factorization* (GMF) and *Multi-Layer Perceptron* (MLP). We create an elastic NCF super-network model with the embedding dimension for MLP and GMF layers sampled from [8, 16, 32, 64, 128], MLP layer number from [1, 2, 3, 4, 5, 6], and MLP hidden sizes from [8, 16, 32, 64, 128, 256, 512, 1024]. We train the NCF super-network by uniformly sampling different sub-networks for each mini-batch of training.

In our experiments we see that when the matrix factorization module of the NCF sub-network was sufficiently large, the results of the subnetwork were dominated by it versus the MLP module. Rendle et al. [2020] substantiate this hypothesis in their paper by showing that a well-tuned matrix factorization approach can substantially outperform proposed learned similarities such as an MLP. We thus attribute the diminished improvement in the performance of LINAS on NCF to the degeneracies in the search space caused by a more powerful matrix factorization module which strongly dominates the HR@10.

#### E Multiply-Accumulates to Latency

In addition to evaluating search performance on the latency, accuracy (Top-1), and BLEU score objectives, we looked at the search trends in terms of multiply-accumulates (MACs) and accuracy as shown in Figure 13a using the fvcore<sup>5</sup> library. Often, multiply-accumulates (MACs) or floating point operations per second (FLOPs) are used to approximate latency. However, we note that the transferrability between these metrics has its limitations. For example, Figure 13b highlights that optimal sub-networks identified during a lengthy (e.g., run search until the Pareto front is saturated with sub-network options) multi-objective MACs and top-1 accuracy NSGA-II search do not translate to the most optimal sub-networks identified during a latency-based NSGA-II search. One benefit of a MACs search is that the best Pareto front population would be ideal for a warm-start population on subsequent searches for a given super-network. Another option in our framework would be to perform a many-objective search (e.g., U-NSGA-III) to find optimal sub-networks in the latency, accuracy, and MACs search space.

Figure 13: Comparison between latency and MACs-based NSGA-II searches using the MobileNetV3 super-network showing that the best Pareto front sub-networks from a MACs-based search do not always translate optimally to the latency objective space.

#### F LINAS Performance for Quantized Super-Networks

In addition to searching for optimal configurations using the OFA image classification super-networks (based in FP32 number format), we also experimented with finding optimal INT8 models from the novel BootstrapNAS (BNAS)

<sup>5</sup><https://github.com/facebookresearch/fvcore>ResNet50 super-network [Muñoz et al., 2021]. BNAS transforms a single reference pre-trained DNN architecture into a super-network and streamlines the sub-network search process in the quantized INT8 space. Specifically, we leverage a BNAS ResNet50 super-network (BNAS-ResNet50Q) that has a different design space, specified with different values for the search design variables, than the Once-for-all ResNet50 model discussed in Section D.2. In this setup, the elastic layer depth chosen from  $[0, 1]$ , an elastic width expansion ratio chosen from  $[0.65, 0.8, 1.0]$  and an elastic expansion ratio chosen from  $[0.2, 0.25]$ .

Our experiment follows the steps outlined in Figure 1 with an additional weight conversion from FP32 to INT8 and standardized fine-tuning of the INT8 model in the last step. As shown on the Figure 14, LINAS offers improvements in terms of hypervolume progression and in the time required to find diverse models in the Pareto front. NSGA-II under-performs in early stages of the search when compared to random search, which could be explained by the characteristics of the BootstrapNAS super-network, and its smaller selection of elastic parameters that have been limited to those promising better performance for the extracted sub-networks. As shown on the Figure 15, the overall distribution of the randomly sampled configurations is closer to the optimal region of the objective space, which may be the cause of random search yielding comparable results in the early stages of the search process. This result is likely due to the nature of BNAS’ process for selecting promising elastic design parameters and also that the search space derived by BNAS-ResNet50Q ( $\sim 10^7$ ) is much smaller than OFA’s ResNet-50 ( $\sim 10^{13}$ ).

Figure 14: Hypervolume comparison of LINAS, NSGA-II and Random Sampling search methods applied to the quantized model (INT8) of BootstrapNAS ResNet50 super-network. Shaded regions show the standard error for 5 trials with different random seeds.

Figure 15: Search results in the BootstrapNAS ResNet50 INT8 search space (CLX, batch size = 128) comparing a: LINAS, and b: NSGA-II approaches.Table 4: Evolutionary algorithm parameter settings for the comparison study in Figure 6. Settings generally follow those recommended by Blank and Deb [2020] for each algorithm.

<table border="1">
<thead>
<tr>
<th rowspan="2"></th>
<th colspan="5">Evolutionary Algorithm</th>
</tr>
<tr>
<th>NSGA-II</th>
<th>AGE-MOEA</th>
<th>U-NSGA-III</th>
<th>C-TAEA</th>
<th>MOEA/D</th>
</tr>
</thead>
<tbody>
<tr>
<td>Number of supported objectives</td>
<td>2</td>
<td>2</td>
<td><math>\geq 2</math></td>
<td><math>\geq 2</math></td>
<td><math>\geq 2</math></td>
</tr>
<tr>
<td>Population size</td>
<td>50</td>
<td>50</td>
<td>50</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>Mutation probability</td>
<td>0.02</td>
<td>0.02</td>
<td>0.02</td>
<td>0.05</td>
<td>-</td>
</tr>
<tr>
<td>Crossover probability</td>
<td>0.9</td>
<td>0.9</td>
<td>0.9</td>
<td>1.0</td>
<td>-</td>
</tr>
<tr>
<td>Reference direction method</td>
<td>-</td>
<td>-</td>
<td>Riesz s-Energy (20 partitions)</td>
<td>Riesz s-Energy (20 partitions)</td>
<td>Riesz s-Energy (20 partitions)</td>
</tr>
<tr>
<td>Number of neighbors</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>20</td>
</tr>
<tr>
<td>Neighbor mating probability</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>0.9</td>
</tr>
</tbody>
</table>

## G Search Algorithm Details

For the search algorithm comparison study in Section 4.2, we evaluate the performance of various evolutionary algorithms, a SMBO multi-objective tree-structured parzen estimator (MOTPE), and a random search using the MobileNetV3 super-network. The evolutionary algorithm settings used for the experiments are shown in Table 4. Evolutionary algorithms that support two or more objectives typically fall in the categories of indicator- or decomposition-based algorithms where the latter often use a predefined set of reference directions on a unit simplex to create objective space partitions. For generating a well-spaced set of reference points from the objective space origin we use the Riesz s-Energy approach [Blank et al., 2021]. For the MOTPE parameters, we use the recommended settings provided by the authors Akiba et al. [2019] including a prior weight of 1.0 and number of candidate samples used to calculate the expected hypervolume improvement equal to 24.

Table 5: Experiment settings for the LINAS (with NSGA-II internal loop) and NSGA-II comparison studies in Figures 5, 7, 8 and 9. The predictor types apply only to the LINAS setup.

<table border="1">
<thead>
<tr>
<th>Super-Network (Modality)</th>
<th>Transformer (Machine Translation)</th>
<th>MobileNetV3, ResNet50 (Image Classification)</th>
<th>NCF (Recommendation)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Accuracy Predictor</td>
<td>SVR w/ RBF kernel</td>
<td>Ridge</td>
<td>SVR w/ RBF kernel</td>
</tr>
<tr>
<td>Latency Predictor</td>
<td>Ridge</td>
<td>Ridge</td>
<td>SVR w/ Linear kernel</td>
</tr>
<tr>
<td>Search Space</td>
<td><math>10^{15}</math></td>
<td><math>10^{19}</math></td>
<td><math>10^7</math></td>
</tr>
<tr>
<td>Population</td>
<td>50</td>
<td>50</td>
<td>10</td>
</tr>
<tr>
<td>Crossover</td>
<td>0.9</td>
<td>0.9</td>
<td>0.1</td>
</tr>
<tr>
<td>Mutation</td>
<td>0.02</td>
<td>0.02</td>
<td>0.02</td>
</tr>
<tr>
<td>LINAS evaluations (Predictor)</td>
<td>20000</td>
<td>20000</td>
<td>2000</td>
</tr>
</tbody>
</table>

For the LINAS and NSGA-II experiments across different modalities (Figures 5, 7, 8 and 9) we show parameter settings in Table 5. In our ablation studies, cross-over rates between 0.9 and 1.0 performed nearly the same, smaller populations work well with smaller search space sizes, and a mutation rate equal to the inverse of the population size as recommended by [Blank and Deb, 2020] gives the best search performance for NSGA-II. The same settings were used for the LINAS inner-loop predictor search which also often uses NSGA-II in this work. An important note is that NSGA-II and AGE-MOEA are not compatible with three or more objectives and that the other many-objective EA approaches, such as U-NSGA-III would need to be considered in that setting.

