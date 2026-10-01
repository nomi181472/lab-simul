# Efficient Automation of Neural Network Design: A Survey on Differentiable Neural Architecture Search

ALEXANDRE HEUILLET, IBISC, Univ Evry, Université Paris-Saclay, France and Massachusetts Institute of Technology, USA

AHMAD NASSER, IBISC, Univ Evry, Université Paris-Saclay, France

HICHEM ARIOUI, IBISC, Univ Evry, Université Paris-Saclay, France

HEDI TABIA, IBISC, Univ Evry, Université Paris-Saclay, France

In the past few years, Differentiable Neural Architecture Search (DNAS) rapidly imposed itself as the trending approach to automate the discovery of deep neural network architectures. This rise is mainly due to the popularity of DARTS, one of the first major DNAS methods. In contrast with previous works based on Reinforcement Learning or Evolutionary Algorithms, DNAS is faster by several orders of magnitude and uses fewer computational resources. In this comprehensive survey, we focus specifically on DNAS and review recent approaches in this field. Furthermore, we propose a novel challenge-based taxonomy to classify DNAS methods. We also discuss the contributions brought to DNAS in the past few years and its impact on the global NAS field. Finally, we conclude by giving some insights into future research directions for the DNAS field.

CCS Concepts: • **General and reference** → **Surveys and overviews**; • **Computing methodologies** → **Search methodologies**; **Computer vision**.

Additional Key Words and Phrases: deep learning, neural architecture search, meta learning

## ACM Reference Format:

Alexandre Heuliet, Ahmad Nasser, Hichem Arioui, and Hedi Tabia. 2018. Efficient Automation of Neural Network Design: A Survey on Differentiable Neural Architecture Search. 1, 1 (May 2018), 36 pages. <https://doi.org/XXXXXXXX.XXXXXXX>

## 1 INTRODUCTION

Neural Architecture Search (NAS) has witnessed rapid development in recent years. The automation of this field supported the development of novel Deep Learning (DL) [62] architectures, especially Convolutional Neural Networks (CNN) [61], that competed with previous state-of-the-art handcrafted models. Since the introduction of CNNs with LeNet [61] and the beginning of Deep Learning with AlexNet [60], most improvements in the field (e.g., deepening the architecture or adding residual connections) were driven by empiricism. NAS aims to put an end to this trial-and-error practice and bring a formal way to smoothen the progress in deep learning architecture design. Moreover, automatically discovering more efficient architectures is particularly relevant in the ecological transition context (i.e., green Deep Learning [119]). The role of manual feature engineering and model development has gradually decreased ever since.

---

Authors' addresses: Alexandre Heuliet, alexandre.heuliet@univ-evry.fr, IBISC, Univ Evry, Université Paris-Saclay, France and Massachusetts Institute of Technology, USA; Ahmad Nasser, ahmad.nasser@universite-paris-saclay.fr, IBISC, Univ Evry, Université Paris-Saclay, France; Hichem Arioui, hichem.arioui@univ-evry.fr, IBISC, Univ Evry, Université Paris-Saclay, France; Hedi Tabia, hedi.tabia@univ-evry.fr, IBISC, Univ Evry, Université Paris-Saclay, France.

---

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from [permissions@acm.org](mailto:permissions@acm.org).

© 2018 Association for Computing Machinery.

Manuscript submitted to ACMConsequently, new challenges such as memory efficiency, transferability between datasets, or computational efficiency have been introduced in the field. This has encouraged researchers to integrate all possible approaches in the literature to improve NAS methodologies. Notably, parameter space differentiability, which uses state-of-the-art optimizers for deep learning model training, is considered one of the most promising leads.

A typical NAS method always consists of three components regardless of the methodology employed, as visually summarized in Fig 1. Firstly, a search space that comprises all the possible hyperparameter sets of the architecture. Generally, this is a discrete search space as it often only covers the categorical choice of operations that compose the architecture. However, it can be made continuous by relaxing this choice using a projection function (e.g., *softmax*) [73] or by including continuous hyperparameters (e.g., the learning rate). It is theoretically infinite, but in practice, it is limited to finite bounds to reduce computational cost and avoid too deep/dense/big architectures. Secondly, a search strategy also denoted as search algorithm or optimizer, that is responsible for exploring the search space and sampling candidate architectures by taking into account the performance feedback of candidates previously sampled. Finally, an evaluation strategy that evaluates each candidate architecture selected by the search strategy. This strategy aims to estimate the model’s performance, ideally without the need to fully train the model to reduce computational cost.

```

graph LR
    SS["Search Space S"] --> SA["Search Algorithm"]
    SA -- "a ∈ S" --> PES["Performance Evaluation Strategy"]
    PES -- "Performance feedback for a" --> SA
  
```

Fig. 1. Typical layout of a Neural Architecture Search framework.

In the literature, handcrafted approaches have first been used to find optimal deep learning architectures. However, in the past few years, ML-based methodologies emerged to accurately address the NAS problem within well-defined constraints. In the following subsections, we briefly present the main categories of NAS approaches as well as the organization of this article.

### 1.1 Tabular-based Approaches

The first methods that were implemented to automate neural network design were heuristic-based approaches that process tabular data. Such methods include random search [64], regular grid sampling of search space [37], tree-based sampling [55], and many other methods inspired by the field of optimization [48, 57, 71, 118]. Although quickly eclipsed by ML-based approaches (see the following subsections) and especially Differentiable NAS, these methods remain present in the NAS landscape as several ones have been proposed in the past few years [48, 64].

### 1.2 Reinforcement Learning

Zoph et al. [141] deployed Reinforcement Learning (RL) to drive the architecture search process. This was one of the first ML-based NAS methods. The RL controller (or agent) continuously chooses new sets of hyperparameters based on the previously evaluated ones. For each set of hyperparameters, the model is fully trained on a given dataset, and a final evaluation score is used as a reward for the RL controller. Since fully training each candidate model is time-consuming, the authors implemented parallelism and asynchronous parameter update to speed up the search process. Later on, Zoph et al. [142] introduced the concept of cell-based hyperparameter tuning, where only the internal architectureof a building block (denoted “cell”) is optimized instead of a complete neural network. Thus, the final model consists of a stacked series of cells. This structural approach, called NASNet, was employed to support model transferability and reduce the search space size. Although NASNet has offered new advantages to help with model optimization, it imposed new challenges concerning the gap between the developing cell structure and the actual multi-cell model derived for evaluation. This structural concept has later migrated to other NAS approaches, as discussed later.

### 1.3 Evolutionary Algorithms

Evolutionary Algorithms (EA) have also presented a valid approach for NAS. Applying EA to NAS is straightforward, as hyperparameter sets are “genotypes” that define an architecture. In EA, a Darwinist process gradually improves an initial population of randomly initialized architectures over several generations. A new generation is obtained by deriving “children” through the recombination (crossover) of genes from the best individuals in the current population. Recent works implemented evolutionary strategies such as guided evolution [74], reinforced evolution [16], and regularized EA [95]. These NAS approaches were among the first to be implemented. However, they are computationally expensive, especially when handling vast search spaces, as they blindly explore the search space during the first iterations. Thus, the overall computational time in GPU days is often ludicrous (e.g., more than 3000 GPU days for AmoebaNet [95]). Moreover, most works in the literature are case-specific. Hence, transferability was less practical than in later gradient-based approaches.

### 1.4 Gradient Descent and Differentiability

Gradient descent is a powerful technique that has been known since the early 19th century with the works of French mathematician Augustin-Louis Cauchy [63]. Its practical implementation is nearly as old as computer science itself, as it was first explored by Haskell Curry in the 1940s [24]. Since then, it became increasingly well-studied and eventually led to the tremendous success of ML with the gradient backpropagation algorithm [98]. More specifically, gradient-based methods for hyperparameter optimization have been in the literature since the early 1990s [3, 11]. Similar to many contributions to Machine Learning at the time, these works were responding to control challenges. They implemented gradient descent for automatically selecting and changing the shape of activation functions in artificial neural networks. In a practical sense, this process is similar to weighing the neurons of a neural network, which affects the model’s output shape. The past few years saw the rise of a novel approach for combining gradient descent and NAS: Differentiable NAS (DNAS).

DNAS implements gradient-based methods to tune the hyperparameters of deep learning models as first pointed out by Bender et al. [5]. The hyperparameter tuning problem is converted into a continuous optimization problem by considering the search space as a smooth manifold. Similar to model training, DNAS uses the gradient information of hyperparameter weights to find the optimal set. This makes it possible to use a supernet that instantiates in memory all candidate architectures, thus removing the need to evaluate each candidate independently to get performance feedback. As a result, fewer computational resources are used to reach the optimal set compared to other implementations, such as evolutionary strategies or reinforcement learning. As shown earlier, implementing such approaches requires a relaxation of the discrete search space. This has been first achieved by Differentiable ARchitectuRe Search (DARTS) [73] using a cell-based paradigm for the search space. From this point onward, DARTS became increasingly popular, helping to democratize NAS with its computational efficiency.Importantly, DNAS and the other ML-based NAS methods discussed above are generally considered black-box approaches. However, ML explainability (eXplainable Artificial Intelligence, XAI) is starting to emerge [44, 78]. This point will be more thoroughly discussed in Section 6.

### 1.5 Article Organization

In this survey, we focused on Differentiable Neural Architecture Search (DNAS) as a trending approach to perform NAS. We analyzed 26 papers that have appeared since 2019 in leading machine learning and computer vision conferences and journals. The goal is to help the reader navigate this emerging field, which gained significant momentum in the past few years. To the extent of our knowledge, we are the first to propose a survey specifically centered on DNAS. We also put the emphasis on computer vision tasks, although other fields of application are discussed in Section 5. Compared with the existing survey literature [96, 116], the main contributions of this article are as follows:

- • An attempt to create a taxonomy of DNAS (Section 2) where we identify the main challenges posed by DNAS (Section 3).
- • A comprehensive review of recent DNAS works according to the challenges they addressed (Section 4). We provide a summary of the reviewed methods in tabular form (Table 2).
- • A discussion on the impact of DNAS on NAS and Deep Learning, along with information on Explainable NAS and insights on future research directions (Section 6).

Finally, in Section 7, we bring a conclusion to this article.

## 2 PROBLEM STATEMENT AND TAXONOMY OF DIFFERENTIABLE NAS

This survey article aims to discuss and analyze Differentiable NAS works according to a novel taxonomy. DNAS methods are typically categorized into two classes: **(a)** DARTS [73] derivatives, to which the majority (62 %) of prior works we reviewed belong, and **(b)** all other DNAS studies. Works in **(a)** assumed that DARTS is on the right track and could be further improved to go past its limitations (detailed in Section 3). The fact that a large number of works belong to **(a)** can be explained by the high popularity that DARTS enjoyed in the past few years. In 2022, DARTS was already an old method (having been published in 2019) but continued to inspire new publications [121, 123], thus demonstrating it passed the test of time. Studies in **(b)** proposed novel DNAS algorithms and search spaces. In many cases [9, 106, 112], they would assume that DARTS' issues are inherent to its conception and that a *tabula rasa* approach was necessary.

However, this classification is trivial and does not help navigate the DNAS landscape. Thus, in this article, we propose a novel taxonomy where methods are classified according to the challenge they attempt to solve rather than belonging to **(a)** or **(b)**. We identified 4 different challenges: **(I)** Bridging the optimization gap between the proxy (i.e., used during search) and final models and resolving gradient approximation issues, **(II)** Countering the over-representation of non-parametric operations (e.g., *skip connections*), **(III)** Improving computational efficiency and reducing latency at inference time, and **(IV)** Bypassing the search space restrictions inherent to DNAS. These challenges are presented in detail in Section 3. Fig. 2 presents a taxonomic tree summarizing the different works we reviewed and our proposed DNAS classification.```

graph LR
    DNA[Differentiable Neural Architecture Search] --> GAI[Gradient Approximation Inconsistencies and Optimization Gap]
    DNA --> ORS[Over-representation of skip connections]
    DNA --> CEL[Computational Efficiency and Latency Reduction]
    DNA --> SSR[Search Space Restrictions]

    GAI --> GAI_L[ProxylessNAS[9] UNAS[104]  
EnTranNAS[122] E2NAS[133]  
OFA[8] BigNAS[126]]
    GAI --> GAI_R[FairDARTS[21] P-DARTS[15]  
S-DARTS[14] iDARTS[134]  
DARTS+PT[107] β-DARTS[123]  
DOTS[40] CDARTS[124]]

    ORS --> ORS_L[E2NAS[133]]
    ORS --> ORS_R[FairDARTS[21] P-DARTS[15]  
DARTS+[68] R-DARTS[129]  
DARTS-[19] PR-DARTS[139]  
DARTS+PT[107] NoisyDARTS[20]  
β-DARTS[123] CDARTS[124]]

    CEL --> CEL_L[ProxylessNAS[9] FBNet[112]  
FBNetV2 [106] VIM-NAS [108]  
HardCoRe-NAS[89] RADARS[121]]
    CEL --> CEL_R[PC-DARTS[120] DOTS[40]]

    SSR --> SSR_L[ProxylessNAS[9] FBNet[106]  
FBNetV2[113] FBNetV5[113]]
    SSR --> SSR_R[D-DARTS[46] DenseNAS[35]]
  
```

Fig. 2. Taxonomy of the reviewed Differentiable Neural Architecture Search literature. References in orange, and light blue correspond to DARTS-based, and non-DARTS-based approaches respectively.

### 3 DARTS AND THE CHALLENGES OF DIFFERENTIABLE NAS

In this section, we present the Differentiable ARchiTecture Search (DARTS) family, which is currently the most popular DNAS family of approaches, and we analyze its limitations along with the development it has inspired in the literature.

DARTS was introduced by Liu et al. [73] as a novel method to implement DNAS with a cell-based approach. In contrast to the majority of evolutionary algorithms [16, 74, 95, 105], and other DNAS methods such as FBNet [106, 112, 113], DARTS defines a modular search space composed of a few building blocks called “cells”, similar to the one used in some RL-based works [141, 142]. There are two types of cells: normal cells (i.e., that make up most of the architecture) and reduction cells (i.e., that perform dimension reduction). Each cell is a direct acyclic graph comprising  $N$  nodes (i.e., intermediary representations of data such as feature maps) linked by edges. Each edge  $e_{i,j}$  connecting node  $i$  to node  $j$  is the sum of the outputs of  $|O_{i,j}| = K$  operations, where  $O_{i,j} = \{o_{i,j}^1, \dots, o_{i,j}^K\}$  represents the set of all possible operations for  $e_{i,j}$ . As a result, each node receives a combination of operation outputs from all previous nodes. Whensearching for CNN cells, the search space of DARTS is composed of  $K = 7$  operations: *skip\_connect*, *max\_pool\_3x3*, *avg\_pool\_3x3*, *sep\_conv\_3x3*, *sep\_conv\_5x5*, *dil\_conv\_3x3* and *dil\_conv\_5x5*.

Fig. 3. Overview of the search strategy of DARTS. Figure from Liu et al. [73].

Hence, DARTS' search strategy is to progressively prune incoming edges from each cell node until a maximum of 2 remains (see Fig 3). To that end, each operation  $o \in O_{i,j}$  in the mixed output of each edge  $e_{i,j}$  is associated to a specific weight  $\alpha_{i,j}^k \in \alpha_{i,j} = \{\alpha_{i,j}^1, \dots, \alpha_{i,j}^K\}$  with  $\alpha_{i,j} \in \alpha$ . DARTS relaxes the categorical choice of operations for  $e_{i,j}$  into a continuous form (i.e., a probability distribution) by performing a *softmax* operation on  $\alpha_{i,j}$ . Thus, the mixed output  $\bar{o}_{i,j}$  of  $e_{i,j}$  is defined as

$$\bar{o}_{i,j}(x) = \sum_{k=1}^K \frac{\exp(\alpha_{i,j}^k)}{\sum_{k'=1}^K \exp(\alpha_{i,j}^{k'})} o_{i,j}^k(x) = \sum_{k=1}^K \text{softmax}(\alpha_{i,j}^k) o_{i,j}^k(x), \quad (1)$$

where  $x$  is the input feature and  $\alpha_{i,j}^k \in \alpha_{i,j}$  is the weight associated with operation  $o_{i,j}^k \in O_{i,j}$ .

The architectural parameters  $\alpha$  are learned through gradient descent to minimize the validation loss  $\mathcal{L}_{val}$  while a supernet (also denoted proxy network) is trained to minimize the training loss  $\mathcal{L}_{train}$ . This supernet comprises a small number (e.g., 8) of stacked search cells and is a representation of all candidate architectures. This way, DARTS solves a bi-level optimization problem formulated as

$$\begin{aligned} \min_{\alpha} \mathcal{L}_{val}(w^*(\alpha), \alpha), \\ \text{s.t. } w^*(\alpha) = \underset{w}{\text{argmin}} \mathcal{L}_{train}(w, \alpha), \end{aligned} \quad (2)$$

where  $w$  denotes the supernet weights. The gradient of the architectural parameters  $\alpha$  is thus computed as follows:

$$\Delta_{\alpha} \mathcal{L}_{val} = \frac{\partial \mathcal{L}_{val}}{\partial \alpha} + \frac{\partial \mathcal{L}_{val}}{\partial w} \frac{\partial w^*(\alpha)}{\partial \alpha}, \quad (3)$$

where  $w^*(\alpha) = \underset{w}{\text{argmin}} \mathcal{L}_{train}(w, \alpha)$  (i.e., the optimal value of  $w$  obtained by minimizing the training loss  $\mathcal{L}_{train}$ ). Finally, once the search phase is over, the internal structure of each type of cell is discretized into the final model by edge selection through a *softmax*. It is possible to derive a final architecture of any size by simply stacking repetitive sequences of the two types of cells.

This way DARTS' search process does not need to fully train each candidate architecture and instead use the supernet as an approximator to get performance feedback by considering the current weights  $w$  as equivalent to the optimal weights  $w^*$ . Hence, DARTS is significantly faster than RL-based or EA-based approaches.However, DARTS suffers from major limitations and poses new challenges. Firstly, as discussed by several articles [19, 21, 46, 123], gradient approximation methods inevitably cause **inconsistencies in the optimization process (I)** that affect the architectural parameters. Combined with the limited convergence ability of *softmax* (itself a smooth approximation of the *argmax* function), this leads to a final probability distribution mainly composed of values very close to one another (i.e., with a low standard deviation compared to the mean). Having only a small difference between the highest values of the probability distribution makes the discretization process more challenging. For instance, it is non-trivial to assert that operation  $o_1$  associated with probability  $p_{o_1} = 0.92$  should be selected rather than operation  $o_2$  associated with probability  $p_{o_2} = 0.91$ . Hence, it is noteworthy to point out that there is a significant gap between the proxy network (smaller, with mixed outputs on edges) used during the search process and the final discretized model (larger, with a maximum of 2 operations per edge). Chen et al. [15] highlighted that this gap also appears when transferring the model to a different dataset than the one used during the search phase.

Secondly, DARTS tends to **overly represent skip connections (II)** within the discovered architectures. This problem is related to the first one since, as argued by Chu et al. [21], the *softmax* operation increases exclusive competition between the different operations (i.e., if one operation is favored, it is at the cost of the others). Edges that include at least one *skip connection* resemble residual blocks [43] and quickly provide a performance boost (i.e., by accelerating forward and backward operations) over other operations that are hence suppressed by *softmax*. In addition, *skip connections* are unparameterized (weight-free) operations and thus have a limited ability to learn data representations. This leads to a global performance collapse as *skip connections* are selected even in edges where they are not the fittest operation. Fig 4 showcases an illustration of the *skip connections* over-representation phenomenon.

Fig. 4. Evolution of the number of *skip connections* w.r.t. the number of search epochs. Figure from Chu et al. [21]

Thirdly, another challenge posed by DARTS is the ability to **efficiently browse the search space (III)** (i.e., to limit the amount of computational resources used by the search algorithm). Eq. 1 shows that every possible path connecting every pair of nodes together must be instantiated in memory in the form of a supernet. Although limited when using toy/proxy datasets such as CIFAR-10 and CIFAR-100 [60], this is especially concerning when using large real-world-like datasets such as ImageNet [26] or MS-COCO [69]. With modern Nvidia GPUs (i.e., starting from Volta), it is possible to take advantage of Tensor Cores (built-in hardware specialized in matrix multiplication) by using Automatic Mixed Precision (AMP) [85]. This training process uses IEEE half-precision format (FP16) [1] when single-precision (FP32) is not needed, hence speeding up computations and resulting in a smaller memory footprint. However, using a lower precision can sometimes cause numerical instability (e.g., gradient overflow). Thus, researchers should primarily focus on designing efficient DNAS approaches instead of only relying on AMP.

Finally, **DARTS' search space is very restricted (IV)**. Searching only for two building blocks leads to a substantially simpler optimization problem. Still, it is insufficient as empirical evidence showed that most modern high-performingCNN architectures such as ResNets [43], ResNext [117], or EfficientNetV2 [102] are composed of more than two different blocks. Thus, increasing diversity in the discoverable architectures is one of the challenges that DARTS-based methods must overcome.

In Section 4, we reviewed some of the most impactful DNAS methods (DARTS derivatives and other methods) and analyzed how they address the four challenges we identified (**I, II, III, and IV**).

#### 4 LITERATURE REVIEW OF DIFFERENTIABLE NAS

In this section, we perform a comprehensive review of recent DNAS literature, with a focus on its most studied subcategory: DARTS [73] and its derivatives. Each of the **28 approaches** we reviewed is presented in the following subsections according to the challenge (see Section 3) it attempted to solve, according to our proposed taxonomy presented in Section 2. Some approaches are listed in several subsections as they address multiple DNAS challenges. We present a summary of this review in Table 1.

Table 1. Summary of the literature we reviewed in this survey. We included DARTS [73] for comparison purposes. The search cost is expressed with the GPU used by the authors of the original article. † denotes models searched on CIFAR-10 or CIFAR-100 [60].

<table border="1">
<thead>
<tr>
<th>Title</th>
<th>Type</th>
<th>Challenges tackled</th>
<th>Top 1 accuracy on ImageNet (%)</th>
<th>Search cost (GPU days)</th>
<th>Search space</th>
</tr>
</thead>
<tbody>
<tr>
<td>DARTS [73]</td>
<td>N.A.</td>
<td>N.A.</td>
<td>73.1</td>
<td>4</td>
<td><i>darts</i></td>
</tr>
<tr>
<td>ProxylessNAS [9]</td>
<td>other</td>
<td><b>(I, III, IV)</b></td>
<td>75.1</td>
<td>8.3</td>
<td><i>MobileNet-like</i></td>
</tr>
<tr>
<td>P-DARTS† [15]</td>
<td>DARTS-based</td>
<td><b>(I, II)</b></td>
<td>75.3</td>
<td>0.3</td>
<td><i>darts</i></td>
</tr>
<tr>
<td>FBNet [112]</td>
<td>other</td>
<td><b>(III, IV)</b></td>
<td>74.9</td>
<td>9</td>
<td><i>MobileNet-like</i></td>
</tr>
<tr>
<td>PC-DARTS [120]</td>
<td>DARTS-based</td>
<td><b>(III)</b></td>
<td>75.8</td>
<td>3.8</td>
<td><i>darts</i></td>
</tr>
<tr>
<td>DARTS+ [68]</td>
<td>DARTS-based</td>
<td><b>(II)</b></td>
<td>76.1</td>
<td>6.8</td>
<td><i>MobileNet-like</i></td>
</tr>
<tr>
<td>PR-DARTS† [139]</td>
<td>DARTS-based</td>
<td><b>(II)</b></td>
<td>75.9</td>
<td>0.17</td>
<td><i>darts</i></td>
</tr>
<tr>
<td>OFA [8]</td>
<td>other</td>
<td><b>(I, III)</b></td>
<td>80.0</td>
<td>1.7</td>
<td><i>MobileNet-like</i></td>
</tr>
<tr>
<td>BigNAS [126]</td>
<td>other</td>
<td><b>(I, III)</b></td>
<td>80.9</td>
<td>1.6</td>
<td><i>MobileNet-like</i></td>
</tr>
<tr>
<td>FBNetV2 [106]</td>
<td>other</td>
<td><b>(III, IV)</b></td>
<td>77.2</td>
<td>25</td>
<td><i>custom</i></td>
</tr>
<tr>
<td>R-DARTS† [129]</td>
<td>DARTS-based</td>
<td><b>(II)</b></td>
<td>-</td>
<td>1.6</td>
<td><i>custom</i></td>
</tr>
<tr>
<td>S-DARTS† [14]</td>
<td>DARTS-based</td>
<td><b>(I)</b></td>
<td>74.8</td>
<td>1.3</td>
<td><i>darts</i></td>
</tr>
<tr>
<td>FairDARTS [21]</td>
<td>DARTS-based</td>
<td><b>(I, II)</b></td>
<td>75.6</td>
<td>3</td>
<td><i>MobileNet-like</i></td>
</tr>
<tr>
<td>DenseNAS [35]</td>
<td>DARTS-based</td>
<td><b>(IV)</b></td>
<td>75.3</td>
<td>2.7</td>
<td><i>MobileNet-like</i></td>
</tr>
<tr>
<td>UNAS [104]</td>
<td>other</td>
<td><b>(I)</b></td>
<td>75.5</td>
<td>4.3</td>
<td><i>MobileNet-like</i></td>
</tr>
<tr>
<td>iDARTS† [131]</td>
<td>DARTS-based</td>
<td><b>(I)</b></td>
<td>75.7</td>
<td>-</td>
<td><i>darts</i></td>
</tr>
<tr>
<td>FBNetV5 [113]</td>
<td>other</td>
<td><b>(IV)</b></td>
<td>81.7</td>
<td>&gt;100</td>
<td><i>custom</i></td>
</tr>
<tr>
<td>NoisyDARTS [20]</td>
<td>DARTS-based</td>
<td><b>(II)</b></td>
<td>76.1</td>
<td>12</td>
<td><i>MobileNet-like</i></td>
</tr>
<tr>
<td>DARTS- [19]</td>
<td>DARTS-based</td>
<td><b>(II)</b></td>
<td>76.2</td>
<td>4.5</td>
<td><i>MobileNet-like</i></td>
</tr>
<tr>
<td>VIM-NAS [108]</td>
<td>other</td>
<td><b>(III)</b></td>
<td>76.2</td>
<td>0.26</td>
<td><i>darts</i></td>
</tr>
<tr>
<td>DOTS [40]</td>
<td>DARTS-based</td>
<td><b>(I, III)</b></td>
<td>76.0</td>
<td>1.3</td>
<td><i>darts</i></td>
</tr>
<tr>
<td>DARTS+PT† [107]</td>
<td>DARTS-based</td>
<td><b>(I, II)</b></td>
<td>74.5</td>
<td>0.8</td>
<td><i>darts</i></td>
</tr>
<tr>
<td>HardCoRe-NAS [89]</td>
<td>other</td>
<td><b>(III)</b></td>
<td>77.9</td>
<td>16.7</td>
<td><i>custom</i></td>
</tr>
<tr>
<td>D-DARTS [46]</td>
<td>DARTS-based</td>
<td><b>(IV)</b></td>
<td>77.0</td>
<td>0.3</td>
<td><i>custom</i></td>
</tr>
<tr>
<td>EnTranNAS [122]</td>
<td>other</td>
<td><b>(I)</b></td>
<td>75.7</td>
<td>1.9</td>
<td><i>darts</i></td>
</tr>
<tr>
<td>RADARS [121]</td>
<td>other</td>
<td><b>III</b></td>
<td>73.8</td>
<td>3.1</td>
<td><i>MobileNet-like</i></td>
</tr>
<tr>
<td><i>beta</i>-DARTS [123]</td>
<td>DARTS-based</td>
<td><b>(I, II)</b></td>
<td>76.1</td>
<td>0.4</td>
<td><i>darts</i></td>
</tr>
<tr>
<td>CDARTS [124]</td>
<td>DARTS-based</td>
<td><b>(I, II)</b></td>
<td>76.3</td>
<td>1.7</td>
<td><i>darts</i></td>
</tr>
</tbody>
</table>#### 4.1 Gradient Approximation Inconsistencies and Optimization Gap

**FairDARTS** [21] tackled DARTS' gradient-related issues by replacing *softmax* for the categorical choice of operations by the *sigmoid* operation  $\sigma$ . This change is motivated by the fact that, contrary to *softmax*,  $\sigma$  does not create exclusive competition between the different operations (i.e., the weights associated with operations can independently increase or decrease). This improves fairness in the operation selection and thus results in better gradient approximations (see Section 4.1). In addition, FairDARTS introduced a novel loss function dubbed zero-one loss and denoted  $\mathcal{L}_{01}$ . This loss function aims to push the sigmoid values of the architectural weights (i.e.,  $\sigma(\alpha)$ ) towards 0 or 1 to minimize the discretization gap. Its gradient magnitude is adequately designed to let the  $\alpha$  weights fluctuate but still pull them towards 0 or 1 if they stray away from 0.5.  $\mathcal{L}_{01}$  is designed as follows:

$$\mathcal{L}_{01}(\alpha) = -\frac{1}{N} \sum_i^N (\sigma(\alpha_i) - 0.5)^2, \quad (4)$$

where  $N$  is the number of nodes in a cell.  $\mathcal{L}_{01}$  is differentiable and thus can be backpropagated to help optimize the architectural weights  $\alpha$ . This loss is combined with  $\mathcal{L}_{val}$  to form the total loss

$$\mathcal{L}_T(w^*, \alpha) = \mathcal{L}_{val}(w^*(\alpha), \alpha) + w_{01} \mathcal{L}_{01} \quad (5)$$

where  $w_{01}$  is an hyperparameter weighting  $\mathcal{L}_{01}$ .

**ProxylessNAS** [9] also sought to close the optimization gap but strayed away from what had been set up by DARTS. The authors proposed to search directly on the target large-scale dataset (e.g., ImageNet [26]) instead of transferring from a small-scale proxy dataset (e.g., CIFAR-10) as done by DARTS and most of its derivatives. To achieve this, they designed a search space that is no longer composed of repetitive building blocks but instead comprises an entire architecture and includes additional candidate operations. However, this comes at the cost of a greatly increased memory consumption as, if we recall Eq. 1, every output feature vector associated with every path in the mixed output of every cell edge must be instantiated and stored in GPU memory. To alleviate this issue, ProxylessNAS replaced DARTS' real-value architectural weights  $\alpha$  with binary gates  $g$  that output one-hot vectors according to a probability distribution  $\{p_1, \dots, p_K\}$ :

$$g = \text{binarize}(p_1, \dots, p_K) = \begin{cases} [1, 0, \dots, 0] & \text{with probability } p_1 \\ \dots & \\ [0, 0, \dots, 1] & \text{with probability } p_K \end{cases} \quad (6)$$

Thus, Eq. 1 is modified as follows:

$$\frac{\partial \text{Binary}}{\partial_{i,j}}(x) = \sum_{k=1}^K g_{i,j}^k o_{i,j}^k(x) = \begin{cases} o_{i,j}^1(x) & \text{with probability } p_1 \\ \dots & \\ o_{i,j}^K(x) & \text{with probability } p_K \end{cases} \quad (7)$$

This mechanism allows entire paths to be binarized and instantiates only one path at a time in memory during the search phase. Thus, it reduces memory consumption to the same level as a regular model. ProxylessNAS successfully overperforms DARTS by 2 % on ImageNet. Nevertheless, these positive results come with a drastically increased search cost from 1.5 GPU days (DARTS) to 8.3 GPU days.

Another work, entitled Progressive DARTS (**P-DARTS**) [15], focused on reducing the optimization gap between the search and final architectures by improving the proxy model used during search. More precisely, the authors gradually increased the proxy network depth during search (e.g., from 5 cells to 20 cells) in contrast with the original DARTSthat uses a fixed 8-cell proxy network that is later derived into a 20-cell final model. Furthermore, the number of candidate operations is progressively reduced according to their performance score. This search space approximation method alleviates the computational efficiency issues encountered when increasing the depth of the proxy network. This process, resumed in Fig. 5, improved performance by around 0.5 % on CIFAR-10/CIFAR-100 [60] and reduced the search cost from 1.5 GPU days to 0.3 GPU day compared to DARTS.

Fig. 5. Layout of the P-DARTS search process. Figure from Chen et al. [15].

SmoothDARTS (**SDARTS**) was designed by Chen et al. [14] as a way to stabilize the bi-level optimization in DARTS (see Eq. 2). Similarly to the authors of R-DARTS [129] (discussed in Section 4.2), they argued that the optimization gap between the proxy model and the final discretized architecture is highly correlated (inversely proportional) to the spectral norm of the Hessian matrix of the validation loss  $\Delta_{\alpha}^2 \mathcal{L}_{val}$ . Hence, they proposed to smooth the validation landscape of DARTS by computing  $\min_{\alpha} \mathcal{L}_{val}(w^*(\alpha), \alpha)$  using  $w^*(\alpha)$  obtained either through random smoothing (SDARTS-RS) or through adversarial training (SDARTS-ADV). SDARTS-RS reformulates Eq. 2 as

$$w^*(\alpha) = \underset{w}{\operatorname{argmin}} \mathbb{E}_{\delta \sim U_{[-\epsilon, \epsilon]}} \mathcal{L}_{train}(w, \alpha + \delta), \quad (8)$$

where  $\delta$  is a perturbation sampled from the uniform distribution  $U_{[-\epsilon, \epsilon]}$  between  $-\epsilon$  and  $\epsilon$ . The idea behind this is to minimize  $\mathcal{L}_{val}(\alpha)$  under a small randomized perturbation  $\epsilon$ . Similarly, SDARTS-ADV is formulated as follows:

$$w^*(\alpha) = \underset{w}{\operatorname{argmin}} \max_{\|\delta\| < \epsilon} \mathcal{L}_{train}(w, \alpha + \delta). \quad (9)$$

Here, Chen et al. strove to increase adversarial robustness by minimizing the worst-case loss under a certain perturbation (computed using a multistep Projected Gradient Descent). They theoretically and empirically demonstrated that both SDARTS-RS and SDARTS-ADV improve the stability and generability of DARTS (e.g., SDARTS-ADV overperforms DARTS by +1.1 % top 1 accuracy on ImageNet). However, both methods have downsides: SDARTS-ADV increases computational cost sharply, but SDARTS-RS is less accurate. Fig. 6 provides an illustration of the smoothing at work in SDARTS.

**BigNAS** [126] is an original approach where the authors proposed to palliate the optimization gap by directly reusing the weights of the supernet/one-shot model to evaluate the performance of the final architecture. This contrasts withFig. 6. The landscape of validation accuracy w.r.t. the architectural parameters  $\alpha$  of DARTS [73], SDARTS-RS and SDARTS-ADV. Figure from Chen et al. [14].

previous works [8, 9, 73] that either retrained the network weights or preprocessed them in some way. To achieve this, BigNAS performs single-stage training using adapted versions of existing training methods such as in-place distillation [125], the sandwich rule [125], exponential learning rate decay scheduling, or dropout-based regularization [102]. These modifications aim to stabilize training and enable BigNAS to efficiently train both large and small candidate architecture within its supernet. Furthermore, the authors proposed a coarse-to-fine architectural selection scheme where a skeleton architecture is first selected according to specific sets of requirements (e.g., input resolution, network depth, or kernel size). Then, these sets are fine-tuned with random mutations to obtain an optimal architecture. BigNAS reaches up to 80.9 % top-1 accuracy on ImageNet [26] for its largest model (9.5 M parameters, 1 GFLOPS), thus overperforming previous approaches.

Vahdat et al. [104] proposed to combine DNAS and Reinforcement Learning-based NAS in a unified framework, denoted **UNAS**, that would bring out the strengths of both approaches. This way, UNAS can search for both differentiable and non-differentiable objectives. In particular, they combine a corrected variant of the classical REINFORCE RL algorithm [111] with a Gumbel-Softmax [52] sampled DNAS algorithm to jointly search for either a differentiable or a non-differentiable objective. Hence, the gradient of a differentiable loss  $\mathcal{L}_d$  can be computed as

$$\begin{aligned} \Delta_{\alpha} \mathcal{L}_d &= \text{REINFORCE}(\mathcal{L}_d, c_d(\alpha)) + C(c_d(\alpha)) + \text{gumbel\_softmax}(\alpha, c_d(\alpha)) \\ &= \mathbb{E}_{i_{\phi}(\alpha)} \left[ (\mathcal{L}_d(\alpha) - c_d(\alpha)) \frac{\partial \log p_{\phi}(\alpha)}{\partial \phi} \right] - \mathbb{E}_{i_{\phi}(\alpha)} \left[ \frac{\partial c_d(\alpha)}{\partial \phi} \right] + \frac{\partial \mathbb{E}_{i_{\phi}(\alpha)} [c_d(\alpha)]}{\partial \phi}, \end{aligned} \quad (10)$$

where  $c_d$  is a control variate defined as  $c_d(\alpha) = \mathbb{E}_{\phi(\zeta|\alpha)} [\mathcal{L}_d(\zeta)]$  used to lower the high variance of REINFORCE where  $\zeta$  is a smooth architecture sampled from a conditional Gumbel-Softmax distribution  $r_{\phi}(\zeta|\alpha)$ . In addition, UNAS can help bridge the optimization gap by introducing a novel objective function  $\mathcal{L}_{gen}$  to avoid architectural overfitting by taking into account the gap between  $\mathcal{L}_{train}$  and  $\mathcal{L}_{val}$  in the optimization process.  $\mathcal{L}_{gen}$  is defined as follows:

$$\mathcal{L}_{gen}(\alpha, w) = \mathcal{L}_{train}(\alpha, w) + \lambda |\mathcal{L}_{val}(\alpha, w) - \mathcal{L}_{train}| \quad (11)$$

where  $\alpha$  denotes the architectural parameters,  $w$  represents the network weights, and  $\lambda$  is a coefficient weighting the generalization gap. UNAS overperforms previous DNAS works on CIFAR-10/100 and ImageNet while maintaining a search cost comparable to DARTS (4 GPU days).In addition to the optimization gap issue, Zhang et al. [133] observed that a catastrophic forgetting problem (multi-model forgetting [6]) occurs in the supernet's weights training, leading to a deterioration of the optimization process for all the candidate architectures derived from the supernet. To palliate these issues, they introduced **E<sup>2</sup>NAS** (Exploration Enhancing Neural Architecture Search with Architecture Complementation), a novel DNAS approach that leverages a VGAE (Variational Graph AutoEncoder) to create an injection between the final discrete architectures and the continuous search space. More precisely, an asynchronous message-passing scheme encodes the architecture into an injective space by encoding the final output  $C$  of the network into a continuous representation  $z$  (i.e., a latent space). Hence, the hidden state  $h_v$  of node  $v$  is defined as

$$h_v = \mathcal{U}(w_v, h_v^{in}) \text{ with } h_v^{in} = \mathcal{G}(h_u : u \rightarrow v), \quad (12)$$

where  $\mathcal{U}$  is a function updating the hidden state  $h_v^{in}$  obtained by aggregating all its predecessors with function  $\mathcal{G}$ . Since both  $\mathcal{G}$  and  $\mathcal{U}$  are injective, the VGAE maps  $C$  to  $z$  injectively. From then, E<sup>2</sup>NAS performs differentiable architecture search on the latent continuous space. In addition, a new complementation loss  $\mathcal{L}_c$  is introduced to tackle the catastrophic forgetting problem. This loss works in conjunction with a replay buffer that contains the last architecture  $\alpha_{i-1}$  along with another complementary architecture  $\alpha_i^c$ .  $\mathcal{L}_{train}$  in Eq. 2 is replaced with  $L_c$  defined for weights  $w_i = w^*(\alpha_i)$  and  $w_i^c = w^*(\alpha_i^c)$  at step  $i$  as

$$\mathcal{L}_c(w_i) = (1 - \epsilon)\mathcal{L}_{CE} + \epsilon(\mathcal{L}_{CE}(w_i^c) + \mathcal{L}_{CE}(w_{i-1})) + \eta\mathcal{R}(w_i), \quad (13)$$

where  $\mathcal{L}_{CE}$  is the Cross-Entropy loss,  $\mathcal{R}$  is a  $l_2$  regularization term,  $\epsilon$  is a value that balances between optimizing the current architecture (exploitation) or preventing other alternatives from vanishing (exploration). E<sup>2</sup>NAS successfully overperformed previous works on the three datasets available in NAS-Bench-201 [30] (e.g., a +29.45 % top 1 accuracy improvement on ImageNet-16-120).

Zhang et al. [134] proposed **iDARTS**, a solution that reformulates the optimization process of DARTS with a Neumann-approximation of the Implicit Theorem Function (IFT) [76]. Concretely, the architectural parameter gradients  $\Delta_\alpha \mathcal{L}_{val}$  (see Eq. 3) are calculated as follows:

$$\Delta_\alpha \mathcal{L}_{val} = \frac{\partial \mathcal{L}_{val}}{\partial \alpha} - \frac{\partial \mathcal{L}_{val}}{\partial w} \left[ \frac{\partial^2 \mathcal{L}_{train}}{\partial w^2} \right]^{-1} \frac{\partial^2 \mathcal{L}_{train}}{\partial \alpha \partial w}, \quad (14)$$

where  $\mathcal{L}_{val}$  is the validation loss,  $\mathcal{L}_{train}$  is the training loss, and  $w$  are the network weights. However, it is computationally intensive to compute the inverse of the Hessian matrix  $\frac{\partial^2 \mathcal{L}_{train}}{\partial w \partial w}$  in Eq. 14. Hence, to alleviate this burden, the authors approximated this inverse matrix using a Neumann series [76]. This Neumann approximation is computed in a stochastic setting where minibatches are used instead of the whole dataset. Thus, the stochastic approximation of the gradients described in Eq. 14 is formulated as follows:

$$\Delta_\alpha \hat{\mathcal{L}}_{val}^i(w^j(\alpha), \alpha) = \frac{\partial \mathcal{L}_{val}^i}{\partial \alpha} - \gamma \frac{\partial \mathcal{L}_{val}^i}{\partial w} \sum_{k=0}^K \left[ I - \frac{\partial^2 \mathcal{L}_{train}^j}{\partial w^2} \right]^k \frac{\partial^2 \mathcal{L}_{train}^j}{\partial \alpha \partial w}, \quad (15)$$

where  $i$  and  $j$  are minibatches randomly sampled from the training and validation datasets respectively,  $\gamma$  is the learning rate,  $K$  is the number of terms of the Neumann series used for approximation, and  $I$  is the identity matrix. This reformulation of the architectural gradient computation performs multiple optimization steps before updating  $\alpha$ , hence making  $w(\alpha)$  closer to its optimal value  $w^*(\alpha)$ . The authors empirically showed that iDARTS improves performance over standard DARTS by 2.6 % on ImageNet [26].Wang et al. [107] argued that the optimization gap in DARTS is linked to the architecture selection process, as the  $\alpha$  weight values associated with an operation might not always reflect this operation’s strength. They defined the discretization accuracy at convergence of an operation as the supernet accuracy after discretizing to this operation and fine-tuning the remaining network until it converges again. Hence, Fig. 7 showcases empirical evidence that the discretization accuracy at convergence of an operation does not necessarily match its  $\alpha$  weight value. In fact, some operations with a small  $\alpha$  value can reach a high discretization accuracy, further reinforcing the critical aspect of the underlying architecture selection problem.

Fig. 7. Bar chart comparing the value of  $\alpha$  w.r.t. the discretization accuracy at convergence for each operation of 3 randomly selected edges from a pretrained DARTS model. Figure from Wang et al. [107].

To alleviate this issue, the authors proposed a perturbation-based architecture selection (PT) where each operation of each edge of the architecture is masked in turn. Then, the operation that leads to the highest drop in performance when masked is considered to be the most important on that edge. This process is not too invasive as it only masks one operation at a time, thus making the supernet accuracy close to the one of the unmodified supernet. Finally, the authors showed that training a DARTS supernet normally and then using PT to discretize the architecture (a process denoted **DARTS+PT**) significantly improves performance (e.g., +0.4 % top 1 accuracy on CIFAR-10 compared to DARTS).

In the continuation of DARTS- (discussed in Section 4.2),  $\beta$ -DARTS [123] introduced a novel and very simple regularization method called Beta-Decode inspired from  $\mathcal{L}_2$  regularization that involves imposition restrictions on the architectural parameters to reduce optimization discrepancies. This regularization occurs on the  $\alpha$  parameters after the *softmax* activation and consists of a straightforward loss function  $\mathcal{L}_{Beta}$ :

$$\mathcal{L}_{Beta}(\alpha) = \log\left(\sum_{k=1}^K \exp(\alpha^k)\right), \quad (16)$$

where  $K$  is the total number of candidate operations.  $\mathcal{L}_{Beta}$  is differentiable and added to the validation loss  $\mathcal{L}_{val}$  pondered by a parameter denoted  $\lambda$ . Thus, Eq. 2 is modified as follows:

$$\min_{\alpha} (\mathcal{L}_{val}(w^*(\alpha), \alpha) + \lambda \mathcal{L}_{Beta}(\alpha)). \quad (17)$$

According to the theoretical analysis provided by the authors,  $\mathcal{L}_{Beta}$  improves generalization and increases robustness. Ultimately,  $\beta$ -DARTS reached competitive scores on both small-scale (CIFAR-10/100) and large-scale (ImageNet) datasets while searching only on CIFAR-10 and CIFAR-100. The search process of  $\beta$ -DARTS is resumed in Fig. 8.

Gu et al. [40] argued that the ranking of operations in DARTS edges is not representative of the final model performance as it does not take correctly into account operations that are related to topology (e.g., *skip connections*),Fig. 8. Layout of the  $\beta$ -DARTS search process in comparison with DARTS [73] and DARTS- [19]. Figure from Ye et al. [123]

hence there is an optimization gap between the proxy model and the final model. To solve this issue, they proposed the novel concept of decoupling the operation and topology search that are performed simultaneously in the original DARTS. This solution, named **DOTS**, divides the search process into two stages. First, during the topology search stage, the topology search space  $\mathcal{E}$  is continuously relaxed into topology weights that are associated with pairwise combinations of edges. For instance, considering node  $x_j$ ,  $\mathcal{E}_{x_j}$  is defined as follows:

$$\mathcal{E}_{x_j} = \{\langle e_{i_1,j}, e_{i_2,j} \rangle | 0 < i_1 < i_2 < j\}. \quad (18)$$

Moreover, for each edge  $e_{i,j}$ , weights of combinations containing this edge are aggregated into  $\gamma_{i,j}$  weights to reduce the search cost.  $\gamma_{i,j}$  is defined by the following equation:

$$\gamma_{i,j} = \sum_{c \in \mathcal{E}_{i,j}, e_{i,j} \in c} \frac{\exp(\beta_{x_j}^c / T_\beta)}{N(c) \sum_{c' \in \mathcal{E}_{x_j}} \exp(\beta_{x_j}^{c'} / T_\beta)}, \quad (19)$$

where  $N(c)$  is the number of edges in edge combination  $c$ , and  $\beta_{x_j}^c$  represents the weight associated with  $c$ . Eq. 19 uses an architectural annealing scheme with temperature  $T_\beta$  as previous works [91, 118] found that this mechanism helps to bridge the optimization gap when searching. In the second phase, DOTS performs an operation search to select the single optimal operation for each edge according to architectural weights  $\alpha$  (similarly to DARTS). However, this strategy could drop some topology-oriented operations before the topology search, thus altering the optimization process. To prevent this, DOTS introduced a group strategy where the operation search space  $O$  is divided into  $p$  subspaces on which the search process is performed independently. Once the search process is over, the best operation from each subspace is selected and merged into a new operation search space. The authors showed that this group strategy effectively preserves topology-related and topology-agnostic operations. DOTS successfully overperformed DARTS top 1 accuracy scores by +0.63 % on CIFAR-10 and by +2.7 % on ImageNet. The global process of DOTS is summarized in Fig. 9.

Yang et al. [122] introduced **EnTranNAS** as a different solution to the optimization gap problem. EnTranNAS comprises Engine-cells (standard DARTS-like differentiable cells) and Transit-cells (transits the derived/discretized architecture). It only searches for a single cell, as the author argues it is sufficient to perform DNAS. Contrary to DARTS,Fig. 9. Layout of the DOTS search process featuring both the operational and topological subprocesses. Figure from Gu et al. [40]

the architecture discretization process in EnTranNAS is no longer part of post-processing but rather done at the end of each search iteration. Hence, the Transit-cells serve to host the currently derived architecture and transmit it to later cells. EnTranNAS includes the target (derived) architecture in the search process, resulting in higher confidence when selecting operations. In addition, the authors introduced a feature-sharing strategy to improve search efficiency, assuming that the same operation from node  $i$  to node  $j > i$  always shares the same features in a single cell. Thus, Eq. 1 is modified as follows

$$\bar{o}_{i,j}(x) = \begin{cases} \sum_{i < j} \sum_{k=1}^K \frac{\exp(\alpha_{i,j}^k/\tau)}{\sum_{k'=1}^K \exp(\alpha_{i,j}^{k'}/\tau)} o_{i,j}^k(x) = p_{i,j}^k o_{i,j}^k(x), & \text{in Engine-cells,} \\ \sum_{(i,k) \in S_j} o_{i,j}^k(x), & \text{in Transit-cells,} \end{cases} \quad (20)$$

where  $\tau$  is a temperature parameter that acts as a regularization factor for the differentiable process in the Engine-cells. This strategy helps to balance optimization between parametric and non-parametric operations. It reduces the computational cost by only computing feature maps of each operation only once per cell (from node  $i$  to ulterior nodes  $j > i$ ). However, EnTranNAS does not completely eliminate the optimization gap. Hence the authors also proposed a novel topology-search-oriented architecture derivation method dubbed EnTranNAS-DST. Concretely, they introduced an additional set of trainable parameters  $\{\beta_j\}_{j=2}^n$  for each intermediary node  $j$  and implemented thresholds  $t_j = \text{sigmoid}(\beta_j)$  to perform operation pruning on those nodes as

$$q_{i,j}^k = \text{ReLU}\left(\frac{p_{i,j}^k}{\max_{i < j, 1 \leq k \leq K} \{p_{i,j}^k\}} - t_j\right). \quad (21)$$

If there is  $k$  s.t.  $q_{i,j}^k \neq 0$ ,  $q_{i,j}^k$  is further normalized by

$$\hat{q}_{i,j}^k = \frac{q_{i,j}^k}{\sum_k q_{i,j}^k}. \quad (22)$$

EnTranNAS-DST node output is thus obtained simply by replacing  $p_{i,j}^k$  with  $\hat{q}_{i,j}^k$  in Eq. 20. The authors experimentally showed that EnTranNAS overperforms most prior works on both CIFAR-10 (+0.28 % top 1 accuracy vs. DARTS) and ImageNet (+2.9 % top 1 accuracy compared to DARTS).

**CDARTS** [124] proposed to address the optimization gap issue by implementing a cyclic feedback mechanism between the search and evaluation networks analogous to a teacher-student model. The search network (composed of 8 cells) provides an intermediate architecture to the evaluation network (composed of 20 cells) and, in return, gets performance feedback. Hence, the search strategy takes into account the performance of the final discretized (and larger) architecture. Furthermore, the two networks are jointly trained and unified into a single architecture. The jointoptimization problem is defined as:

$$\min_{\alpha} \mathcal{L}_{val}(w_E^*, w_S^*, \alpha) \quad \text{s.t.} \quad \begin{cases} w_E^* = \underset{w_E}{\operatorname{argmin}} \mathcal{L}_{val}(w_E, \alpha), \\ w_S^* = \underset{w_S}{\operatorname{argmin}} \mathcal{L}_{train}(w_S, \alpha), \end{cases} \quad (23)$$

where  $w_S$  and  $w_E$  are the weights of the search and evaluation networks respectively. CDARTS' search process comprises two stages. Firstly, the separate learning stage during which both networks are trained individually on the input dataset.  $\alpha$  weights are initialized with random values, whereas the cell architectures of the evaluated are initialized from the top- $k$  discretization  $\bar{\alpha}$  of the learned  $\alpha$ . Secondly, the joint optimization stage where the search algorithm leverages performance feedback from the evaluation network to update  $\alpha$  defined as follows:

$$\alpha^*, w_E^* = \underset{\alpha, w_E}{\operatorname{argmin}} \mathcal{L}_{val}^S(w_S^*, \alpha) + \mathcal{L}_{val}^E(w_E, \bar{\alpha}) + \lambda \mathcal{L}_{val}^{S,E}(w_S^*, \alpha, w_E, \bar{\alpha}), \quad (24)$$

where  $\mathcal{L}_{val}^{S,E}$  denotes the knowledge transfer procedure between the search and evaluation networks, dubbed *introspective distillation* and formulated as:

$$\mathcal{L}_{val}^{S,E}(w_S^*, \alpha, w_E, \bar{\alpha}) = \frac{T^2}{N} \sum_{i=1}^N p(w_E, \bar{\alpha}) \log\left(\frac{p(w_E, \bar{\alpha})}{q(w_S^*, \alpha)}\right) \quad (25)$$

where  $N$  is the number of training samples,  $T$  is a temperature coefficient, and  $p$  and  $q$  are the output feature logits of the evaluation and search networks respectively (computed using a *softmax*). CDARTS overperforms previous methods on DARTS search space (e.g., +3% top 1 accuracy improvement compared to DARTS) while keeping the computational cost reasonable (1.7 GPU days). The main concept behind CDARTS is showcased in Fig. 10.

Fig. 10. Comparison between the search processes of DARTS [73], P-DARTS [15], EnTranNAS [122], and CDARTS [124]. Figure from Yu et al. [124]

#### 4.2 Over-representation of skip connections in DARTS

As already discussed in Section 4.1, **FairDARTS** [21] replaced *softmax* with the *sigmoid* operation  $\sigma$  to ensure fair competition between the different operations (i.e., the weights associated with operations can independently increaseor decrease). This means that a high prominence of *skip connections* will not suppress the other operations that can thus overperform and replace them. Empirically, this results in a lessened presence of *skip connections* in the final architectures as shown in Fig. 11. Some other methods, such as D-DARTS [46] (detailed in Sec. 4.4), tackled the *skip connections* issue using the same approach as FairDARTS.

Fig. 11. Stacked area plot of the number of dominant operations of DARTS and FairDARTS when searching on ImageNet. Figure from Chu et al. [21]

In a different manner, the authors of **P-DARTS** [15] managed to restrict the number of *skip connections* by introducing an operation-level *dropout* [101] to regularize the search space. More accurately, the *dropout* mechanism is placed after every *skip connection* to block the path and entice the search algorithm to explore other operations. In addition, the *dropout* rate is gradually decayed to prevent the *skip connections* from being completely suppressed (i.e., *skip connections* are heavily penalized at the start of the search process and then treated equally with the other operations at the end). Additional details on P-DARTS can be found in Section 4.1.

Liang et al. [68] argued that the over-representation of *skip connections* results from an overfitting phenomenon in the optimization process of DARTS. To alleviate this issue, they proposed **DARTS+**, an early-stopping procedure that ends the search phase if the following criteria are met:

1. (1) *Two or more skip connections are present in the normal cell architecture.*
2. (2) *The ranking of architecture parameters  $\alpha$  for learnable operations becomes stable for a determined number of epochs (e.g., 10 epochs).*

The authors showed that using either of these criteria led to performance improvements over previous baselines (e.g., + 0.7 % top 1 accuracy compared to DARTS when using Criterion 1). Furthermore, they provided empirical evidence that Criterion 1 is easier to use and implement but yields less accurate results than Criterion 2. This simple early stopping procedure was dubbed DARTS+ and is illustrated by Fig. 12.

Zela et al. [129] also focused on robustifying DARTS as they found out that performance collapses in many cases with high dominance of unparameterized (i.e., *skip/pooling*) operations. Hence, they proposed DARTS-ES, a novel method that performs early stopping according to the eigenvalues of the Hessian matrix of the validation loss  $\Delta_{\alpha}^2 \mathcal{L}_{val}$  w.r.t. the  $\alpha$  weights. More precisely, they showed that large eigenvalues often lead to degenerate architectures and tracked these values to stop the search process before the performance collapses. Furthermore, they implementedFig. 12. Illustration of the early stopping process in DARTS+. Figure from Liang et al. [68]

two different regularization methods. The first one uses a combination of well-known data augmentations techniques (Cutout [27], and ScheduledDropPath [142]). The second one increases  $\mathcal{L}_2$  regularization by choosing among several factors (e.g., 1, 3, 9, 27, 81). Both techniques successfully increased robustness, especially when combined with DARTS-ES, an approach dubbed **R-DARTS**. The authors tested their approach on several computer vision datasets (CIFAR10/100 [60], SVHN [90]) and under different search spaces. R-DARTS improved top 1 accuracy on CIFAR-10 up to +3.64 % compared to DARTS.

As a non-DARTS approach, **E<sup>2</sup>NAS** [133] (first presented in Section 4.1) also addressed the over-representation of non-parametric operations in their own way. They tackled the *rich-get-richer problem*, in which the optimizer is biased towards architectures with high performance in their early stage. They added a measure of the novelty into the gradient to avoid being stuck in local minima, hence computing architectural weights  $\alpha_\theta$  update as

$$\alpha_\theta^{i+1} \leftarrow \alpha_\theta^i - (1 - \gamma) \nabla_{\alpha_\theta^i} \mathcal{L}_{val}(\alpha_\theta^i, w^*) - \gamma \nabla_{\alpha_\theta^i} N(\alpha_\theta^i, A), \quad (26)$$

where  $N(\alpha_\theta^i, A)$  is a measure of architecture  $\alpha_\theta^i$  from the history of architectures  $A$ . This enhancement led to a higher probability of sampling novel architectures rather than well-trained architectures from previous iterations.

**DARTS-** [19] tackled the global performance collapse induced by *skip connections* by adding an auxiliary *skip connection* to the classic mixed output of operations (see Eq. 1). The authors asserted that previous works based on analyzing the Hessian matrix eigenvalues (e.g., R-DARTS [129]) were imperfect as those methods tend to reject good architectures if they do not meet some arbitrary threshold. This auxiliary operation is pondered by  $\beta$ , a coefficient independent from architectural weights that is progressively decayed to 0 during the search phase. Moreover, the authors introduced  $\beta^{skip}$ , a parameter that denotes the importance of *skip connections* inside of the mixed output of operations. Thus Eq. 1 is modified as follows:

$$\bar{o}_{i,j}(x) = (\beta + \beta_{i,j}^{skip})x + \sum_{o \in O \setminus \{skip\}} \frac{\exp(\alpha_{i,j}^o)}{\sum_{o' \in O \setminus \{skip\}} \exp(\alpha_{i,j}^{o'})} o_{i,j}(x). \quad (27)$$Fig. 8 (b) illustrates this mechanism. The authors showed that DARTS- can significantly improve robustness and stabilization during the search process, with +0.5 % improvement on CIFAR-10 [60], and +4.5 % on ImageNet [26] compared to standard DARTS. DARTS- also uses fewer computational resources than previous approaches such as R-DARTS [129]. In addition to standard DARTS, this approach is able to improve the performance of other derivatives such as P-DARTS [15] or PC-DARTS [120].

Path-Regularized Differential Network Architecture Search (**PR-DARTS**) [139] was the first method to propose a theoretical in-depth analysis of why the over-representation of *skip connections* phenomena happens and how it is connected to performance collapse. This differs from prior works [15, 21, 68] that mainly observed this issue and empirically tested their own solutions. In particular, the authors introduced a convergence theorem demonstrating that the number of *skip connections* heavily influences the supernet’s convergence rate (i.e., the more *skip connections*, the faster the supernet converges). This is linked to *skip connections* faster decaying the validation loss  $\mathcal{L}_{val}$ , and thus leading DARTS search algorithm to increase  $\alpha$  weights associated with *skip connections* at the cost of decreasing all other weights. To palliate this issue, they replaced architectural weights with stochastic binary gates, denoted  $g_{i,j}^k$  for the  $k$ th operation between nodes  $i$  and  $j$ . At each iteration,  $g_{i,j}^k$  is sampled from a Bernoulli distribution to compute the output of each node. Thus, Eq. 1 is modified as follows:

$$\bar{o}_{i,j} = \sum_{k=1}^K g_{i,j}^k o_{i,j}^k(x). \quad (28)$$

However, leaving the gates unregularized could bias the operation selection in cells since DARTS will increase the weights of all operations to achieve faster convergence. Furthermore, increasing the value of any operation weight could reduce or maintain the loss  $\mathcal{L}_{val}$ . The authors resolve these issues by using a group-structured sparsity regularization on the gates via rescaling and imposing thresholds:  $g_{i,j}^k = \min(1, \max(0, a + (b - a)\bar{g}_{i,j}^k))$  with  $a < 0$  and  $b > 1$ , and  $\bar{g}_{i,j}^k$  is an approximation of  $g_{i,j}^k$  using the Gumbel reparametrization trick. This regularization is expressed by two loss functions targeting *skip* and *non-skip* connections respectively:

$$\mathcal{L}_{skip}(\alpha) = \zeta \sum_{l=1}^{h-1} \sum_{s=0}^{l-1} \sigma(\alpha_{s,l}^{skip} - \tau \log\left(\frac{-a}{b}\right)), \quad \mathcal{L}_{non-skip}(\alpha) = \frac{\zeta}{r-1} \sum_{l=1}^{h-1} \sum_{s=0}^{l-1} \sum_{k=1}^K \sigma(\alpha_{s,l}^k - \tau \log\left(\frac{-a}{b}\right)), \quad (29)$$

where  $\sigma$  denotes the sigmoid function,  $h$  is the path depth,  $\zeta = \frac{2}{h(h-1)}$  and  $\tau$  is a temperature hyperparameter. In addition, they introduced path regularization to reduce the unfair competition between deep and shallow cells (i.e., a cell containing a large amount of intermediary *skip connections*) as follows:

$$\mathcal{L}_{path}(\alpha) = \prod_{l=1}^{h-1} \sum_{k \in O_p} \sigma(\alpha_{l,l+1}^k - \tau \log\left(\frac{-a}{b}\right)), \quad (30)$$

where  $O_p$  denotes the parameterized operations. Hence, Eq. 2 is modified as follows:

$$\begin{aligned} \min_{\alpha} \mathcal{L}_{val}(w^*(\alpha), \alpha) + \lambda_1 \mathcal{L}_{skip}(\alpha) + \lambda_2 \mathcal{L}_{non-skip}(\alpha) - \lambda_3 \mathcal{L}_{path}(\alpha), \\ \text{s.t. } w^*(\alpha) = \underset{w}{\operatorname{argmin}} \mathcal{L}_{train}(w, \alpha), \end{aligned} \quad (31)$$

where  $\lambda_1$ ,  $\lambda_2$  and  $\lambda_3$  are constants. All of these improvements make PR-DARTS search for performance-oriented networks rather than fast-convergence-oriented ones. Empirical results show that PR-DARTS overperformed DARTS and earlier variants on image classification datasets (ImageNet, CIFAR-10) by a large margin.The authors **DARTS+PT** [107] (presented in Section 4.1) showed that their proposed perturbation-based architecture selection method prevents *skip connections* from becoming dominant. From a theoretical aspect, they refer to Greff et al. [39], who proved that ResNet [43] layers are robust to reordering as their outputs correspond to the same estimated optimal feature map values. As the presence of *skip connections* makes DARTS' supernet resembles ResNet, this may explain why DARTS layers are also robust to reordering. Thus, this fact indicates that edges in a cell all try to estimate the same optimal feature maps  $m^*$ . Wang et al. [107] used this finding to define the estimated optimal feature maps  $\bar{m}_e$  for input  $x_e$  of edge  $e$ :

$$\bar{m}_e(x_e) = \frac{\exp(\alpha_{conv})}{\exp(\alpha_{conv}) + \exp(\alpha_{skip})} o_e(x_e) + \frac{\exp(\alpha_{skip})}{\exp(\alpha_{conv}) + \exp(\alpha_{skip})} x_e, \quad (32)$$

$$\text{with } \alpha_{conv} \propto (x_e - m^*), \alpha_{skip} \propto (o_e(x_e) - m^*)$$

where  $\alpha_{conv}$  and  $\alpha_{skip}$  are architectural parameters, and  $o_e$  is the mixed output of operations associated with edge  $e$ . It can be deduced from Eq. 32 that the better the supernet is optimized, the closer  $x_e$  will get to  $m^*$  (since the goal of the training phase is to make edges estimate  $m^*$ ). Consequently, this will widen the  $(\alpha_{skip} - \alpha_{conv})$  gap and ultimately this will lead to  $\alpha_{skip} > \alpha_{conv}$ . However, Wang et al. showed that this only becomes problematic if the architecture selection process relies on  $\alpha$ . On the contrary, DARTS+PT does not suffer from this issue, although it retains the same search algorithm as DARTS.

Another work dubbed **NoisyDARTS** [20] addressed this problem in an original way. The authors proposed to inject unbiased random noise during training to prevent the optimizer from increasing architectural weights associated with *skip connections* ( $\alpha_{skip}$ ) too much. They argued that adding noise is an efficient way to improve generalization by smoothing the loss landscape, as pointed out by a prior study [114]. In practice, NoisyDARTS adds Gaussian noise to the input of *skip connections*. Thus, Eq. 1 can be rewritten as

$$\bar{o}_{i,j}(x) = \sum_{k=1}^{K-1} \text{softmax}(\alpha_{i,j}^k) o_{i,j}^k(x) + \text{softmax}(\alpha_{i,j}^{skip}) o_{i,j}^{skip}(x + \tilde{x}), \quad (33)$$

where  $\tilde{x} \sim \mathcal{N}(\mu, \sigma^2)$  is a random noise sampled from a Gaussian distribution parameterized by mean  $\mu$  and variance  $\sigma^2$  ( $\mu = 0$  and  $\sigma = 0.2$  when searching on ImageNet). Despite its simplicity, NoisyDARTS managed to suppress *skip connections* and consistently overperformed prior DARTS derivatives on CIFAR-10 [60], ImageNet [26], and NAS-Bench-201 [30] (e.g., +9.7 % top 1 accuracy on ImageNet compared to DARTS).

Ye et al. ( **$\beta$ -DARTS**) [123] continued the work initiated by DARTS- [19] with the introduction of the Beta-Decay regularization method (presented in detail in Section 4.1). In addition to reducing the optimization gap problem, they argued that the Beta-Decay mechanism also alleviates the over-representation of *skip connections* issue and ensures fair competition between the operations. More accurately, the authors provide a theoretical explanation with the following equation:

$$\phi \propto \sum_{i=0}^{h-2} [(\theta_{i,h-1}^{conv} \beta_{i,h-1}^{conv})^2 \prod_{t=0}^{i-1} (\theta_{i,h-1}^{skip} \beta_{i,h-1}^{skip})], \quad (34)$$

where  $h$  is the number of layers in the supernet,  $\phi$  represents the architectural weight gradients,  $\theta$  represents the influence of the Beta Decay regularization, and  $\beta_k = \frac{\exp(\alpha_k)}{\sum_{k' \in \mathcal{O}} \exp(\alpha_{k'})} = \text{softmax}(\alpha_k)$ . By taking into account that  $\theta$  varies antagonistically to  $\beta$  (as it is a regularization function), Eq. 34 shows us that the convergence of networks rely more on  $\beta_{conv}$  than on  $\beta_{skip}$ . Consequently, this means that Beta-Decay helps to reduce the prominence of *skip connections*.**CDARTS** [124] straightforwardly addressed the over-representation issue. They simply added a L1 regularization factor to the architectural weights of non-parametric operations  $O_{np} = \text{skip\_connect}, \text{max\_pool\_3x3}, \text{avg\_pool\_3x3}$  as:

$$\mathcal{L}_{reg} = \lambda \sum_{o \in O_{np}} |\alpha_o|, \quad (35)$$

where  $\lambda$  is a hyperparameter that balances the value of  $\mathcal{L}_{reg}$ . The authors showed that this method successfully prevented the operations in  $O_{np}$  from becoming dominant.

### 4.3 Computational Efficiency and Latency Reduction

Chen et al. [15] defined the problem of *NAS in the wild* as being able to search for an architecture on a proxy dataset (e.g., CIFAR-10 [60]) to limit computational cost and successfully transfer to another, more challenging dataset (e.g., ImageNet [26]). Most DARTS derivatives followed this paradigm, contrary to most non-DARTS approaches such as ProxylessNAS [9].

As discussed above (see Section 4.1 for additional details), **ProxylessNAS** searches directly on the target dataset (e.g., ImageNet [26]) rather than a proxy dataset (e.g., CIFAR-10). It also enforces latency constraints on specific hardware (e.g., mobile phones, GPU, or CPU). Hence, it is a multi-objective NAS method, but one of the objectives (latency) is not differentiable. Instead, latency is measured in real-time on GPU/CPU and is predicted from a lookup table on mobile settings. ProxylessNAS successfully constrained mobile latency to a similar level to MobileNetV2 [99] and improved top 1 accuracy by 2.6 % on ImageNet.

With **PC-DARTS** (Partially-Connected DARTS), Xu et al. [120] sought to improve computational efficiency without compromising performance. To this end, they perform architecture search in only a subset of randomly sampled channels while bypassing the rest. This concept is based on the assumption that computation on this subset is an adequate approximation of the effective computation on all the channels. Considering edge  $e_{i,j}$ , partial channel connection involves defining a channel sampling mask  $S_{i,j}$  which nullifies (i.e., assigns a weight value of 0) all channels except selected ones in the mixed output  $\bar{o}_{i,j}$ , thus modifying Eq. 1 as follows:

$$\bar{o}_{i,j}^{PC}(x) = \sum_{k=1}^K \frac{\exp(\alpha_{i,j}^k)}{\sum_{k'=1}^K \exp(\alpha_{i,j}^{k'})} o_{i,j}^k(S_{i,j}x) + (1 - S_{i,j})x. \quad (36)$$

This process has the advantage of reducing the memory overhead by  $K$  times, with  $\frac{1}{K}$  being the channel selection ratio. Subsequently, it helps reduce the search cost on CIFAR-10 from 1 GPU day (DARTS) to only 0.1 GPU day, and PC-DARTS achieves a better top 1 accuracy on ImageNet than ProxylessNAS (75.8 % vs. 75.1 %) with half the search cost. However, partial channel connection induces an inconsistency in the selection of operations across the different sampled channels. To palliate this issue, the authors introduced an additional set of learning parameters  $\beta_{i,j}$  that are shared throughout the search process to act as an *edge normalization* mechanism. The PC-DARTS approach is summarized in Fig 13.

Cai et al. proposed Once-for-All (**OFA**) [8] as a solution for decoupling the training and search phases with the objective of drastically reducing the computational cost of DNAS. In particular, they trained a single large supernet whose configuration (e.g., kernel size, depth, or width) can be altered and directly deployed without further training. OFA follows two stages: (1) a training phase where the different subnetworks that compose the supernet are optimized to improve their accuracy (2) a hardware-aware NAS phase (model specialization stage) where sub-networks are sampled to train accuracy and latency predictors. This enables OFA to target specific hardware and latencies. However, since simultaneously optimizing the parameters of the very large ( $10^{19}$ ) number of subnetworks is prohibitively expensive,Fig. 13. Layout of the PC-DARTS search process. Figure from Xu et al. [120]

the authors introduced a novel training process during which the OFA network is progressively fine-tuned to train subnetworks of increasingly smaller size. This is akin to a pruning process performed over different modalities (i.e., input resolution, width, depth, and kernel size). Overall, OFA only requires 4.2k GPU hours for end-to-end training (3 times lower than DARTS [73]) to overperform all previous approaches on ImageNet in the mobile setting (i.e., less than 600M FLOPS).

Although it was not their primary objective, the authors of **DOTS** [40] were able to reduce computational cost to only 0.26 GPU day when searching on CIFAR-10 and 1.3 GPU days when searching on ImageNet. This is due to the decoupling between the topology search and operation search that greatly reduces the number of candidate operations on each edge (and thus the search space size), making both processes converge fast.

Wu et al. [112] proposed **FBNet** as a DNAS framework aimed at improving latency and computational efficiency, especially targeted at low-power hardware such as mobile phones. Firstly, FBNet browses a search space  $\mathcal{A}$  different than DARTS' that is not organized around cell building blocks but rather around layers. The macro-architecture (i.e., the pre-processing/post-processing layers, the number of intermediate layers, and their input shapes) is fixed, whereas independent architectures (from a selection of "blocks") are searched for each layer. This leads to a greater diversity of candidate architectures than in DARTS' search space. In addition, FBNet combines the standard Cross-Entropy loss  $\mathcal{L}_{CE}$  with a hardware-aware latency loss  $\mathcal{L}_{LAT}$  defined as follows:

$$\mathcal{L}_{LAT} = \alpha \log \left( \sum_i LAT(b_l^a) \right)^\beta, \quad (37)$$

where  $LAT(b_l^a)$  denotes the latency of block  $a$  of the  $l$ -th layer, and  $\alpha$  and  $\beta$  are coefficients weighting  $\mathcal{L}_{LAT}$ . The latency values are retrieved from a latency lookup table (similarly to ProxylessNAS [9]), as measuring latency from mobile processors in real time is prohibitively expensive. In addition, using a lookup table makes  $\mathcal{L}_{LAT}$  differentiable. Finally, the authors devised a differentiable NAS algorithm where the search space is modeled by a stochastic supernet. Thus, only one candidate block is sampled at a time independently for each layer from a probability distribution obtained through a softmax instead of a weighted mixed output of operation as featured in DARTS (see Eq. 1). Consequently, theoutput  $x_l$  of layer  $l$  is a masked output defined as:

$$x_l = \sum_i m_{l,i} b_{l,i}(x_{l-1}), \quad (38)$$

where  $m_{l,i}$  is a mask that equals to 1 if block  $b_{l,i}$  is sampled or 0 otherwise. Therefore, the probability of sampling an architecture  $a \in \mathcal{A}$  is described by the following equation:

$$P_{\Theta}(a) = \prod_l P_{\Theta_l}(b_l = b_{l,i}^a), \quad (39)$$

where  $\Theta$  is composed of all the parameters that determine the sampling probabilities of blocks for each layer. Furthermore, the authors resorted to the Gumbel-Softmax [52] function to relax the discrete masks  $m_l$  into a continuous distribution and thus make the whole search process differentiable w.r.t. the sampling parameters  $\Theta$ . The authors empirically showed that FBNet reached a higher top 1 accuracy (e.g., +1.8 % for FBNet-C) on ImageNet than DARTS for a 33 % lower search cost. FBNet-A also reached a latency as low as 19.8 ms when targeting a Samsung Galaxy S8.

However, FBNet is not free from limitations, and hence Wan et al. [106] designed an updated method dubbed **FBNetV2**. Their primary concern was to palliate the small search space size issue present in FBNet and DARTS. Consequently, they introduced a greatly enlarged search space (see Section 4.4). To keep their method computationally efficient, the authors proposed DMaskingNAS. This NAS algorithm uses weight-sharing approximations to efficiently search over additional hyperparameters, such as the number of filters and the input dimensions. They kept the layer-wise DNAS paradigm described in FBNet but used a channel-masking mechanism parameterized through a Gumbel-Softmax function. Thus the output  $y$  of a block  $b$  can be computed as follows:

$$y = b(x) \circ \sum_{i=1}^k g_i \mathbb{1}_i, \quad (40)$$

where  $g_i$  denotes Gumbel weights and  $\mathbb{1}_i$  is a mask vector whose first  $i$  values are 1s with the rest being 0s. This way, each block's channel number can be searched without significant additional computational cost. Furthermore, FBNetV2 searches for different input resolutions by performing resolution subsampling from the original input (i.e., extracting smaller input feature maps using the nearest neighbors method). Once the output feature map has been computed, it is upsampled into a larger fixed-size one to preserve dimensional consistency. FBNetV2 maintains a computational cost similar to FBNet despite searching on a search space up to  $10^{14}$  times larger.

**VIM-NAS** (Variational Information Maximization Neural Architecture Search) [108] observed that each cell edge is considered independent in the global architecture of previous DNAS methods. In contrast, the authors introduced a novel way of formulating the NAS problem by assuming that the architectural distribution  $\mathcal{A}$  is a latent representation of specific data points from dataset  $\mathcal{D}$  such as there is a distribution  $p_{\phi}(\mathcal{D}, \mathcal{A}) = p(\mathcal{D})p_{\phi}(\mathcal{A}|\mathcal{D})$  parameterized by  $\phi$ . More specifically, VIM-NAS strives to maximize the mutual information  $I_{\phi}(\mathcal{D}, \mathcal{A})$  between  $\mathcal{A}$  and  $\mathcal{D}$  as

$$\max_{\phi} I_{\phi}(\mathcal{D}, \mathcal{A}) = \mathbb{E}_{p_{\phi}(\mathcal{D}, \mathcal{A})}[\log p_{\phi}(\mathcal{D}|\mathcal{A})]. \quad (41)$$

Thus, the objective  $\mathcal{L}(\phi, \theta, \mathcal{D})$  of the DNAS process can be formulated as

$$\max_{\theta, \phi} \mathcal{L}(\phi, \theta, \mathcal{D}) = \sum_{d \in \mathcal{D}} \mathbb{E}_{p_{\phi}(\mathcal{D}, \mathcal{A})}[\log q_{\theta}(\mathcal{D}, \mathcal{A})], \quad (42)$$

where  $\log q_{\theta}(\mathcal{D}, \mathcal{A})$  is a supernet approximation of  $\log p_{\phi}(\mathcal{D}|\mathcal{A})$ . In practice,  $p_{\phi}(\mathcal{D}|\mathcal{A})$  is reformulated to the Gaussian noise  $\mathcal{N}(\mu_{\theta}, 1)$  where  $\mu_{\theta}$  is parameterized by the convolutional network  $\phi$ . This makes VIM-NAS very fast as it canconverge in only a single epoch in DARTS' search space (i.e., a 0.007 GPU day search cost) while providing a top-1 accuracy improvement of +0.55 % on CIFAR-10 [60] and +2.04 % on ImageNet [26] compared to DARTS.

Methods such as FBNet [112] or ProxylessNAS [9] sought to impose hardware and latency constraints softly by formulating an objective function which is a trade-off between accuracy and computational resources. In contrast, **HardCoRe-NAS** (Hard Constrained diffeRentiable NAS) [89] searches for high-accuracy architectures that strictly respect a hard latency constraint. The authors reformulated the classic bilevel optimization problem of DNAS (see Eq. 2) as

$$\begin{aligned} \min_{\zeta \in \mathcal{S}} \mathbb{E}_{x,y \sim \mathcal{D}_{val}; \hat{\zeta}} \mathcal{P}_{\zeta}(\mathcal{S}) [\mathcal{L}_{CE}(x, y | w^*, \hat{\zeta})] \text{ s.t. } \text{LAT}(\zeta) \leq T \\ w^* = \underset{w}{\operatorname{argmin}} \mathbb{E}_{x,y \sim \mathcal{D}_{train}; \hat{\zeta}} \mathcal{P}_{\zeta}(\mathcal{S}) [\mathcal{L}_{CE}(x, y | w, \hat{\zeta})], \end{aligned} \quad (43)$$

where  $\mathcal{S}$  is a fully differentiable block-based search space parameterized by  $\zeta = (\alpha, \beta)$ ,  $\mathcal{D}_{train}$  and  $\mathcal{D}_{val}$  are the train and validation datasets' distributions,  $\mathcal{P}_{\zeta}(\mathcal{S})$  is a probability measure over  $\mathcal{S}$ , and  $\text{LAT}(\alpha, \beta)$  is the estimated latency of the model.  $\mathcal{S}$  is composed of a micro  $\mathcal{A}$  (i.e., block internal architecture  $c \in C$ ) and macro (i.e., connections between blocks at every stage  $s \in S$ )  $\mathcal{B}$  search spaces parameterized by  $\alpha \in \mathcal{A}$  and  $\beta \in \mathcal{B}$  respectively. Thus, the overall expected latency  $\text{LAT}(\alpha, \beta)$  is computed by summing over the latency  $t_{b,c}^s$  for every possible configuration  $c \in C$  of every block  $b$ , over all possible depths  $d$ , and over all the stages:

$$\text{LAT}(\alpha, \beta) = \sum_{s=1}^S \sum_{b'=1}^d \sum_{b=1}^{b'} \sum_{c \in C} \alpha_{b,c}^s \cdot t_{b,c}^s \cdot \beta_{b'}^s. \quad (44)$$

HardCoRe-NAS uses LAT to build a constrained search space  $\mathcal{S}_{LAT} = \{\zeta | \zeta \in \mathcal{P}_{\zeta}(\mathcal{S}), \text{LAT}(\zeta) \leq T\}$ . Similarly to DARTS,  $\mathcal{S}$  is relaxed to be continuous by searching for  $\zeta \in \mathcal{S}_{LAT}$ . The search process of HardCoRe-NAS is visually summarized in Fig 14. The authors experimentally showed that HardCoRe-NAS managed to constrain latency to the same level or lower than previous methods while overperforming them on ImageNet [26] (e.g., -9 ms latency reduction and +1.6 % top 1 accuracy improvement compared to FBNet on an Nvidia P100 GPU).

Similarly to UNAS [104], **RADARS** [121] (Reinforcement Learning Aided Differentiable Architecture Search) leverages Reinforcement Learning to help the differentiable search process. However, RADARS focuses on reducing computational and memory costs, whilst UNAS is performance-oriented. The RL algorithm prunes the search space through iterative exploration/exploitation phases. It identifies promising subsets of operations for each layer and prunes the search space of the other operations (exploration phase). Differentiable NAS is then performed on this reduced search space instead of the entire search space (exploitation phase). The authors bounded the GPU memory usage to a maximum of 12 Go (Nvidia RTX 2080ti). They showed that RADARS could reach competitive scores for restricted memory (11 Go) and time (3.08 GPU days) on ImageNet despite using a large MobileNet-like search space.

#### 4.4 Search Space Restrictions

Bypassing the search space restrictions of DNAS (especially concerning DARTS) is one of the main goals pursued by researchers in the field. For instance, **ProxylessNAS** [9] step out of the cell-based paradigm to instead search directly for an entire architecture. However, as discussed in Section 4.1, this led to an exponential increase in computational resources that the authors alleviated by using a binarization mechanism to only instantiate a single path in memory at a given time.Fig. 14. Layout of the HardCoRe-NAS search process. Figure from Nayman et al. [89]

**D-DARTS** [46] kept the cell-based paradigm of DARTS but extended it to cover an entire architecture *via* a distributed process. The authors still used two types of cells (*normal* and *reduction*), but they were individualized to each cover a specific portion of the architecture. Each individual cell can be considered a small independent neural network. This means that a search model of size  $N$  is now composed of  $N$  individual cells  $C$  (with *reduction* cells positioned at the 1/3 and 2/3 of the network). To help with the optimization of individualized cells, D-DARTS introduced a novel loss function dubbed *ablation loss* and denoted  $\mathcal{L}_{AB}$ . This loss function, derived from Game Theory (i.e., from the concept of Shapley Values [100]), considers the Neural Architecture Search process as a cooperative game with cells being assimilated to players. Hence,  $\mathcal{L}_{AB}$  computes the marginal contribution  $M_C$  of each cell to the common reward (the validation loss  $\mathcal{L}_{val}$ ) and quantifies the difference between the marginal contribution  $M_C^i$  of given cell  $C^i \in C$  and the mean marginal contribution:

$$\mathcal{L}_{AB}^i = \begin{cases} \frac{M_C^i - \text{mean}(M_C)}{\text{mean}(M_C)} & \text{if } \text{mean}(M_C) \neq 0 \\ 0 & \text{else} \end{cases} \quad (45)$$

with  $M_C^i$  defined as follows:

$$M_C^i = \mathcal{L}_{val}^{(C)} - \mathcal{L}_{val}^{(C \setminus \{C_i\})}, \quad (46)$$

Furthermore, this distributed search space makes it possible to directly use existing handcrafted architectures as starting points for the search process. A handcrafted architecture can be considered a local minimum in the search space. This way, architectures such as ResNet50 [43] or Xception [18] have been successfully encoded in D-DARTS' space and have been significantly improved.

This process led to a drastic increase in the diversity of discovered architectures that allowed D-DARTS to reach competitive scores in both small-scale (CIFAR-10 [60]) and large-scale (ImageNet [26]) image classification datasets. The search process of D-DARTS is summarized in Fig. 15.The diagram illustrates the D-DARTS search process. It is divided into two main sections: the Search Algorithm (Architect) and the Partial Network Training. The Search Algorithm takes Hyperparameters and Ground Truth as inputs. It consists of multiple cells (Cell 1 to Cell n) with their own criteria (Criterion 1 to Criterion n) and architecture optimizers (Adam). The search algorithm outputs a network (Supernet) which is directly connected to the dataset. The dataset is used for partial network training, which involves a model optimizer (SGD with momentum + Cosine Annealing scheduler) and a criterion (L\_CE).

Fig. 15. Layout of the D-DARTS search process. Figure from Heuillet et al. [46].

The authors of DenseNAS [35] introduced a densely-connected block-based search space that allows them to search for block widths and the number of blocks per layer. They designed routing blocks with gradually increasing widths, and each block output linked to multiple other blocks, covering a large spectrum of block widths and block numbers per layer. Each routing block comprises several shape-alignment layers and several basic layers, which are mixed outputs of the different candidate operations. The basic layers are relaxed into continuous operations using *softmax*, similarly to DARTS (see Eq. 1). In the same way, the routing block output is a mixed output of all the possible paths leading to the next routing block and relaxed using a *softmax*. Furthermore, DenseNAS features a chained cost estimation algorithm that aims to restrict the computational cost and latency of the final architecture. The cost of each basic layer operation is retrieved from a lookup table and is used to estimate the cost of the whole architecture as follows:

$$cost^l = \sum_{o \in O} w_o^l cost_o^l \quad (47)$$

$$\tilde{cost}^i = cost_b^i + \sum_{j=i+1}^{i+m} p_{i,j} (cost_{align}^{i,j} + cost_b^j), \quad (48)$$

where  $w_o^l$  is the *softmax* weight of operation  $o$  in basic layer  $l$ ,  $p_{i,j}$  is the *softmax* weight of the path from routine block  $B_i$  to routine block  $B_j$ ,  $m$  is the number of subsequent blocks, and  $cost_{align}^{i,j}$  is the cost of the shape-alignment layer in block  $B_j$  with input from block  $B_i$ . This cost is then integrated into the loss function

$$\mathcal{L}(w, \alpha, \beta) = \mathcal{L}_{CE} + \lambda \log_{\mathcal{T}}(\tilde{cost}^1). \quad (49)$$

DenseNAS overperformed previous methods on ImageNet (+2.8 % top 1 accuracy compared to DARTS) while being able to constrain the number of FLOPS and the latency.

The authors of **FBNetV2** [106] (first presented in Section 4.3) pointed out that the main issue in previous DNAS works (DARTS [73] and FBNet [112] primarily) is related to their severely restricted search space. Therefore, they crafted a novel search space encompassing two new hyperparameters (i.e., the number of channels and the input resolution). This novel search space comprises  $10^{35}$  candidate architectures and is  $10^{14}$  times larger than FBNet's. By leveragingthis expanded search space, FBNetV2 overperforms FBNet on ImageNet significantly (e.g., +1.9 % top 1 accuracy for FBNetV2-F4 compared to FBNet-B).

Building upon what has been laid by previous FBNet works [25, 106, 112], **FBNetV5** [113] is an interesting work as it did not simply seek to lift search space restrictions but more specifically focused on a less-explored subject: improving the transferability of DNAS architectures between different computer vision tasks. The authors addressed this challenge by combining a differentiable NAS process with FBNetV3, a NAS approach that stepped out of DNAS and instead proposed its own novel paradigm where architectures and training hyperparameters ("recipe") are matched to reach optimal performance. More precisely, they created a supernet trained on a multitask dataset (generated from ImageNet [26]) to disentangle the search process from the training pipeline of the target. FBNetV5 performs topology search to find the optimal backbone architecture for each task (i.e., semantic segmentation, object detection, and image classification). This is done simultaneously for all tasks. Furthermore, the authors designed a search algorithm that produces architectures for each task at a constant computational agnostic to the number of tasks. At the task level, they leverage a DNAS process following [112] where they browsed a block-based search space  $\mathcal{A} = \{0, 1\}^B$  comprising  $B$  blocks. An architecture  $a \in \mathcal{A}$  is hence a set of binary masks  $a_b$  sampled independently from a Bernoulli distribution. When searching on multiple tasks, the DNAS problem is relaxed as follows:

$$\min_{\pi^1, \dots, \pi^T, w} \sum_{t=1}^T \mathbb{E}_{a^t} p_{\pi^t} \{\ell^t(a^t, w)\}, \quad (50)$$

where  $a^t$  are architectures sampled from task-specific distributions  $p_{\pi^t}$ ,  $\ell^t$  are task-specific losses,  $T$  is the number of tasks, and  $w$  denotes the supernet weights. In addition, to restrict computational cost, the authors adopt the RL algorithm REINFORCE [111] and importance sampling [42] to reduce the number of forward and backward passes of the search algorithm from  $T$  to 1. Fig. 16 summarizes the search process of FBNetV5. FBNetV5 successfully overperforms all previous NAS methods on datasets (ImageNet [26], COCO [69], and ADE20k [138]) corresponding to the three tasks this method focused on.

The diagram illustrates the FBNetV5 search process, which is divided into two main components: **Search (Train Supernet)** and **Train Searched Arch.**.

**Search (Train Supernet):**

- An input image of size  $(H, W)$  is processed through a series of stages (Stage 0, Stage 1, Stage 2, ..., Stage  $s$ ).
- Each stage consists of a sequence of blocks. A legend indicates that solid green blocks are **Selected Blocks** and dashed green blocks are **Skipped Blocks**.
- Between stages, there are **Fusion** layers (Fusion 1, Fusion 2, ..., Fusion  $s-1$ ).
- Four paths are shown, representing different task-specific resolutions:
  - Path 0:  $(H/4, W/4)$
  - Path 1:  $(H/8, W/8)$
  - Path 2:  $(H/16, W/16)$
  - Path 3:  $(H/32, W/32)$
- Each path ends with a task-specific head: **SEG Head**, **DET Head**, and **CLS Head**.
- **Back propagation** is indicated by arrows pointing from the heads back to the stages.
- **Importance Sampling** is shown as a vertical arrow pointing from the heads to the stages.
- At the top, three probability distributions are shown: **SEG Arch. Prob.**, **DET Arch. Prob.**, and **CLS Arch. Prob.**.
- **REINFORCE Estimation** is shown as a circular arrow pointing from the heads to the probability distributions.

**Train Searched Arch.:**

- This part shows the training of the individual architectures for each task.
- Three existing training pipelines are shown: **SEG Arch.**, **DET Arch.**, and **CLS Arch.**.
- Each pipeline is associated with a task-specific loss:
  - For SEG:  $\gamma_{\text{SEG}} \times \text{Loss}_{\text{SEG}}$
  - For DET:  $\gamma_{\text{DET}} \times \text{Loss}_{\text{DET}}$
  - For CLS:  $\gamma_{\text{CLS}} \times \text{Loss}_{\text{CLS}}$
- A **Border collie** image is shown as an example of the output for the CLS task.

Fig. 16. Layout of the FBNetV5 search process. Figure from Wu et al [113].## 5 APPLICATIONS

One salient fact to note is that most of the major DNAS works (e.g., DARTS [73], ProxylessNAS [9], or FBNet [112]) focused on improving CNN architectures targeted at computer vision tasks. Hence, they mostly validate their approaches on image classification datasets (e.g., CIFAR-10/100 [60], or ImageNet [26]) and object detection or semantic segmentation datasets (e.g., MS-COCO [69], Pascal-VOC [33], or Cityscapes [23]). However, many works explored other fields of application. In addition to the major approaches discussed in the following paragraphs and Section 4, we included more minor works that focused on a specific application of DNAS. Table 2 summarizes those approaches.

**Remote Sensing** A few applications were developed to address image-oriented tasks such as radar image analysis [29, 135] and scene classification [92]. For instance, Zhang et al. [135] leveraged a DARTS-based method to search for AutoDL detector CNN backbones optimized for sonar image and maritime radar image analysis.

**Natural Language Processing.** DARTS [73], and some of its variants (such as  $\beta$ -DARTS [123]) searched for Recurrent Neural Networks (RNN) [98] architectures to perform Natural Language Processing (NLP) tasks such as language modeling on the Penn Tree Bank [80] and WikiText-2 [83] datasets. In the same area, AdaBERT [12] leveraged Differentiable NAS to compress BERT [58] large language models into smaller task-oriented models.

**Medical Applications.** Numerous works proposed DNAS-based approaches to analyze medical data, such as MRIs [65, 77] or volumetric images [51, 127, 140]. The application to medical imaging is straightforward as most developed DNAS approaches target CNNs and computer vision tasks, as shown in Section 4. For example, Guo et al. [41] developed a DNAS approach to modulate the composition of CNNs used to perform organ at risk segmentation in patients treated for head and neck cancer.

**Reinforcement Learning.** RL-DARTS [84] used DARTS to search for backbone architectures for Deep Reinforcement Learning on-policy and off-policy algorithms. They showed that performing Differentiable NAS with DARTS is relevant for both discrete action and continuous control environments. For instance, in the Procgen [22] environment, RL-DARTS improved performance by 250 % over the baseline IMPALA-CNN [32].

**Audio Processing.** A few studies targeted speech recognition with the aim of discovering novel convolutional neural network architectures specialized in audio pattern extraction [28, 49, 137]. For instance, Hu et al. [49] designed a DARTS-based search strategy to discover Time Delay Neural Network (TDNN) architectures optimized for automatic speech recognition.

**Hardware Optimization.** Matching architectures and hardware has been a major focus of DNAS. Multiple works (some, such as ProxylessNAS [9], are discussed in detail in Section 4.3) tried to optimize computational cost and latency on multiple platforms such as GPUs [9, 112, 121], CPUs [9], and embedded systems [9, 59, 75]. In our opinion, this is paramount to ensure the real-world deployment of Deep Learning, as consumer-grade devices are not as powerful as high-end/data-center GPUs/TPUs typically used by researchers.

## 6 DISCUSSION AND FUTURE DIRECTIONS

In Section 4, we reviewed 26 recent DNAS approaches that targeted 4 different challenges (presented in Section 3). More than half (62 %) of these approaches are based on DARTS [73] due to the high popularity this method enjoyed from the moment it was first published (2019) until now, with novel DARTS-based methods still being proposed in 2022 [123, 124]. Each reviewed DNAS method addressed at least one of the four challenges we identified in Section 3.

One noteworthy fact (clearly shown in Fig.2) is that there is a clear partition between DARTS-based and non-DARTS-based works when considering the challenges they tackled. The vast majority (81 %) of DARTS-based methods addressedTable 2. Summary of Differentiable NAS works according to their field of application. Some articles are referenced under multiple categories.

<table border="1">
<thead>
<tr>
<th>Field</th>
<th>Subfield</th>
<th>References</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="8">Computer Vision</td>
<td>Image Classification</td>
<td>[14, 15, 19, 21, 46, 68, 73, 106, 112, 113, 120, 134]</td>
</tr>
<tr>
<td>Object Detection</td>
<td>[67, 81]</td>
</tr>
<tr>
<td>Video Processing</td>
<td>[94]</td>
</tr>
<tr>
<td>Semantic Segmentation</td>
<td>[34, 70, 140]</td>
</tr>
<tr>
<td>Image Super-Resolution</td>
<td>[50, 109, 115]</td>
</tr>
<tr>
<td>Image Denoising</td>
<td>[38, 130]</td>
</tr>
<tr>
<td>Pose Estimation</td>
<td>[136]</td>
</tr>
<tr>
<td>Image Generation</td>
<td>[36]</td>
</tr>
<tr>
<td rowspan="2">Remote Sensing</td>
<td>Facial Recognition</td>
<td>[66]</td>
</tr>
<tr>
<td>Radar Image Analysis</td>
<td>[29, 135]</td>
</tr>
<tr>
<td rowspan="3">Natural Language Processing</td>
<td>Scene Classification</td>
<td>[92]</td>
</tr>
<tr>
<td>Language Modeling</td>
<td>[54, 73, 123]</td>
</tr>
<tr>
<td>Keyword Spotting</td>
<td>[87]</td>
</tr>
<tr>
<td>Medical Applications</td>
<td>Medical Image Analysis</td>
<td>[41, 51, 65, 77, 110, 127, 140]</td>
</tr>
<tr>
<td>Reinforcement Learning</td>
<td>Deep Reinforcement Learning Backbone</td>
<td>[84]</td>
</tr>
<tr>
<td>Audio Processing</td>
<td>Speech Recognition</td>
<td>[28, 49, 137]</td>
</tr>
<tr>
<td rowspan="2">Hardware Optimization</td>
<td>Embedded Systems and Latency Reduction</td>
<td>[8, 9, 53, 59, 75, 79, 89, 93, 112, 121, 126, 128, 132]</td>
</tr>
<tr>
<td>Predictive Maintenance</td>
<td>[131]</td>
</tr>
</tbody>
</table>

either challenge **I** (gradient approximation discrepancies) or challenge **II** (over-representations of *skip connections*) while the rest mostly targeted challenge **III** (computational efficiency and latency reduction) and challenge **IV** (search space restrictions). This can be explained as **I** and **II** are DARTS’ most prominent issues and thus constitute the main leads to pursue any follow-up work. On the other hand, non-DARTS-based DNAS saw those issues are inherent to DARTS and proposed a change of paradigm that allowed them to focus on other, more global, problems (i.e., **III** and **IV**). Let us dive into the main conclusions of each category.

**(I)** A large subset of methods [9, 21, 40, 107, 122, 124] agreed that the final discretized architecture is dissimilar to the proxy model used during the search process, thus resulting in the optimization gap. However, these works differ in their proposed solutions to that analysis. Some replaced the discretization process to yield models that better fit the proxy network, while others designed a novel search process when the proxy network is more closely tied to the final model (or even a proxyless search process [9]). These DNAS works yielded improved results that show the relevance of their respective contributions. However, these improvements are often marginal (i.e., less than 1 % top 1 accuracy improvement on ImageNet [26] compared to previous state-of-the-art), and there is no general consensus among all recent articles on how to reduce the optimization gap. Thus, this may indicate that, despite efforts to provide mathematical background, we still lack a formal model that would bring an optimal solution to this problem. The DNAS optimization gap is not closed yet.

**(II)** One interesting fact to note is that nearly every paper that addresses the *skip connection* issue provides an analysis of why DARTS fails and draws similar conclusions: the non-parametric operations have an unfair advantage as they accelerate gradient descent in the early stage by forming structures akin to residual blocks [43]. Eventually, this unfair competition suppresses parametric operations and leads to architectural “overfitting”. Furthermore, most of the reviewed works (e.g., [15, 21, 123]) proposed to add regularization on the search space to prevent this phenomenon.This regularization is generally applied either before ( $\alpha$  weights) or after ( $\beta$  weights) the *softmax* relaxation. This proved relevant as adding regularization prevented *skip connections* from becoming dominant and improved performance. This outcome is logical as a search space mixing parametric and non-parametric operations is inherently unbalanced, and regularization is a well-explored solution to overfitting and rebalancing ill-formed problems [7, 103]. Finally, as an alternative solution, other works [68, 129] devised early-stopping mechanisms to stop the search process before the architecture overfits. However, these approaches are based on arbitrary or empirical criteria that are less formal than regularization-based approaches, hence explaining the popularity of the latter.

**(III)** Some approaches [108, 120, 121] managed to reduce the search cost drastically (up to 43 times for VIM-NAS [108]) compared to DARTS. This made it possible to launch the search process on low-end, consumer-grade GPUs (and even CPUs in some cases). Hence, it contributed to making DNAS a very accessible process to automate neural network design. However, other methods [9, 89, 106] chose to trade computational cost for reduced latency at inference, hence helping Deep Learning to deploy on low-resource devices, such as mobile phones. They did so by adding the inference latency as a differentiable objective so that raw performance is no longer the only goal of the search process. To save computational resources, the latency values for a specific platform are often retrieved from a latency lookup table.

**(IV)** As previously stated, the fact that most DNAS methods that addressed search space restrictions are not DARTS-based highlights that it is an issue more closely associated with DARTS. Thus, those methods had to craft search algorithms and/or search spaces that deeply diverge from DARTS. Most notably, all approaches in that area abandoned the cell-based building block paradigm as it is one of the main elements that restrict the search space. For instance, the FBNet family [25, 106, 112, 113] relied on a novel MobileNet-like search space that is block-based rather than cell-based. ProxylessNAS [9], and DenseNAS [35] also leveraged a similar search space. Finally, the low number of studies tackling the search restrictions (i.e., 6) means that researchers mostly focused on other issues judged more urgent. In addition, using a larger search space leads to a drastic increase in computational cost, hence making it necessary to design mechanisms to save resources. This fact might explain why proposing a novel method addressing this issue is difficult.

Overall, over the past few years, all of these works contributed to making DNAS more and more viable, with the ability to discover architectures that can now far surpass the performance of handcrafted ones (e.g., CDARTS [124] reaches 78.2 % top 1 accuracy on ImageNet [26] vs. 74.7 % for MobileNetV2 [99]). In addition, the computational cost (i.e., the number of FLOPs) and the latency can be restrained to deploy models on embedded platforms such as mobile phones [9, 89, 112]. This makes DNAS (and Deep Learning by extension) easier to deploy to solve real-world tasks and accessible to a wider audience. Thus, one could argue that DNAS is now a maturing field, with some methods being included in major AutoML libraries such as Microsoft NNI [86] and NASLib [97].

Nevertheless, DNAS still suffers from limitations as none of the proposed could solve all of the four identified challenges at once. Hence, no DNAS method could impose itself as a novel standard. This may substantially explain how DARTS withstood the test of time and remains popular despite its age. In our opinion, this also indicates that DNAS has room for improvement and has not reached its full potential yet.

As discussed in Section 5, DNAS has already been applied to a wide range of applications (mainly related to computer vision). In the near future, with DNAS becoming more and more robust, we can expect it to be applied to an ever-increasing number of fields. For instance, works on transformers improvements [17] and self-supervised learning [45] have already started to emerge. Other already explored fields, such as Generative Adversarial Networks (GANs) design [36], could be further expanded to other applications, such as face generation [56].

Additionally, Explainable AI (XAI) is paramount to ensure the deployment of Deep Learning in everyday tasks. In the case of automated decision making (e.g., judicial case analysis or autonomous driving), people affected by those

