Title: Adaptive Spiking Neurons for Vision and Language Modeling

URL Source: https://arxiv.org/html/2604.12365

Markdown Content:
, Sihang Guo Harbin Institute of Technology Shenzhen China, Jiaqi Wang Harbin Institute of Technology Shenzhen China, Dongyang Ma School of Computer Science Peking University Beijing China, Jin Cheng School of Electronic and Computer Engineering, Peking University Beijing China, Qingyan Meng Pengcheng Laboratory Shenzhen China, Zhengyu Ma Pengcheng Laboratory Shenzhen China and Yonghong Tian School of Computer Science Peking University Beijing China

###### Abstract.

Regarded as the third generation of neural networks, Spiking Neural Networks (SNNs) have garnered significant traction due to their biological plausibility and energy efficiency. Recent advancements in large models necessitate spiking neurons capable of high performance, adaptability, and training efficiency. In this work, we first propose a novel functional perspective that provides general guidance for designing the new generation of spiking neurons. Following the insightful guidelines, we propose the Adaptive Spiking Neuron (ASN), which incorporates trainable parameters to learn membrane potential dynamics and enable adaptive firing. ASN adopts an integer training and spike inference paradigm, facilitating efficient SNN training. To further enhance robustness, we propose a specialized variant of ASN, the Normalized Adaptive Spiking Neuron (NASN), which integrates normalization to stabilize training. We evaluate our neuron model on 19 datasets spanning five distinct tasks in both vision and language modalities, demonstrating the effectiveness and versatility of the ASN family. Our ASN family is expected to become the new generation of general-purpose spiking neurons.

Spiking Neural Networks, Spiking Neuron, Neuromorphic Computing, Spiking Transformer, Brain-inspired Computing

## 1. Introduction

Spiking Neural Networks (SNNs), widely regarded as the third generation of neural networks (Maass, [1997](https://arxiv.org/html/2604.12365#bib.bib21)), have attracted increasing attention due to their high biological plausibility and superior energy efficiency. Unlike conventional Artificial Neural Networks (ANNs), which rely on continuous-valued activations, SNNs communicate through discrete spikes and process information via event-driven dynamics. This spike-based computation paradigm enables neurons to remain inactive in the absence of events, resulting in sparse computation. Accordingly, SNNs replace traditional high-power dense multiply–accumulate (MAC) operations with sparse low-power accumulate (AC) operations, thereby achieving substantial energy savings (Roy et al., [2019](https://arxiv.org/html/2604.12365#bib.bib25)). SNNs are increasingly being considered a promising paradigm for energy-efficient artificial intelligence (Fang et al., [2023a](https://arxiv.org/html/2604.12365#bib.bib8)).

The pivotal role of spiking neurons within SNNs means that, as artificial intelligence (AI) tasks grow in complexity, the design of these neural units must evolve to meet higher benchmarks of efficiency and functional adaptability. The comparison of the previous foundation spiking neurons in the SNN domain is shown in the Table [1](https://arxiv.org/html/2604.12365#S1.T1 "Table 1 ‣ 1. Introduction ‣ Adaptive Spiking Neurons for Vision and Language Modeling"). In addition to training efficiency, architectural compatibility, and spike-driven properties, adaptive firing with respect to the membrane potential constitutes another key property. As an essential characteristic of general-purpose spiking neurons, this adaptability regulates firing dynamics to ensure effective information transmission, preventing both excessive and insufficient neuronal activity. The Leaky Integrate-and-Fire (LIF) neuron is hindered by low training efficiency and a lack of membrane potential adaptivity. The Parametric Leaky Integrate-and-Fire (PLIF) neuron (Fang et al., [2021](https://arxiv.org/html/2604.12365#bib.bib10)) introduces learnable membrane time constants to capture dynamic neuronal behavior, enabling adaptive firing. However, PLIF still suffers from low training efficiency, which is further exacerbated by the sigmoid-constrained learnable parameters. Parallel Spiking Neuron (PSN) enhances adaptivity via parameter matrices and improves training efficiency; nevertheless, it retains T-dimensional training, leading to non-negligible computational time and memory overheads. Furthermore, the compatibility of PSN with mainstream spiking backbones, such as Spiking Transformers (Zhou et al., [2023b](https://arxiv.org/html/2604.12365#bib.bib44), [2026b](https://arxiv.org/html/2604.12365#bib.bib40); Yao et al., [2023](https://arxiv.org/html/2604.12365#bib.bib35)), remains an open challenge, likely attributable to optimization difficulties associated with its parameter matrices. On a different trajectory, the Integer LIF (ILIF) pioneers an efficient integer Training and Spike Inference paradigm. Building on this, NILIF (Lei et al., [2025](https://arxiv.org/html/2604.12365#bib.bib17)) incorporates normalization to enhance neuronal training stability. However, both ILIF and NILIF are constrained by insufficient membrane potential adaptivity, limiting their potential as a universal neuron. Although the recent SpikingBrain (Pan et al., [2025](https://arxiv.org/html/2604.12365#bib.bib23)) enhances ILIF by employing an adaptive threshold derived from pre-calculated average membrane potentials, this input-dependent strategy remains incongruent with the inherently spike-driven paradigm essential for efficient neuromorphic computation.

Table 1. Comparison with other foundation spiking neurons in the SNN domain, including LIF (Fang et al., [2022](https://arxiv.org/html/2604.12365#bib.bib9)), PLIF (Fang et al., [2021](https://arxiv.org/html/2604.12365#bib.bib10)), PSN (Fang et al., [2023b](https://arxiv.org/html/2604.12365#bib.bib11)), ILIF (Yao et al., [2023](https://arxiv.org/html/2604.12365#bib.bib35)), and NILIF(Lei et al., [2025](https://arxiv.org/html/2604.12365#bib.bib17)). ❍ indicates that PSN is substantially faster than LIF and PLIF, while remaining notably slower than ILIF and NILIF in training. Our ASN family (including ASN and NASN) satisfies the four characteristics: Efficient Training, Adaptive Firing, Architecture Compatibility, Spike-Driven Inference.

Spiking Neurons Efficient Training Adaptive Firing Architecture Compatibility Spike-Driven Inference
LIF(Gerstner et al., [2014](https://arxiv.org/html/2604.12365#bib.bib13))✗✗✓✓
PLIF(Fang et al., [2021](https://arxiv.org/html/2604.12365#bib.bib10))✗✓✓✓
PSN(Fang et al., [2023b](https://arxiv.org/html/2604.12365#bib.bib11))❍✓✗✓
ILIF(Luo et al., [2024](https://arxiv.org/html/2604.12365#bib.bib19))✓✗✓✓
NILIF(Lei et al., [2025](https://arxiv.org/html/2604.12365#bib.bib17))✓✗✓✓
ASN, NASN✓✓✓✓

Based on the above analysis, we adopt a novel functional perspective to systematically revisit the fundamental design principles of the new generation of spiking neurons in SNNs. We argue that the four basic characteristics of spiking neurons—efficient training, adaptive firing, architecture compatibility, and spike-driven inference—must be considered simultaneously. Guided by these principles, we propose the Adaptive Spiking Neuron (ASN) and its variant, the Normalized Adaptive Spiking Neuron (NASN). Both ASN and NASN are general spike-driven neurons that combine efficient training with adaptive membrane potential dynamics. The ASN family (ASN and NASN) introduces a layer-wise learnable parameter \alpha to dynamically shift the activation window in the ILIF family (ILIF and NILIF), aligning with the evolving distribution of membrane potentials across layers. This adaptive mechanism enhances both representational capacity and distributional flexibility, making ASN and NASN particularly suitable for vision and language modeling. Overall, this work advances the design of spiking neurons in SNNs. Our main contributions are summarized as follows.

*   •
We systematically analyze the limitations of existing foundation spiking neurons and provide a functional view for the new generation of spiking neuron design.

*   •
Based on our insightful guidelines, we propose the Adaptive Spiking Neuron (ASN) and its variants, the Normalized Adaptive Spiking Neuron (NASN). Both ASN and NASN are general spike-driven neurons, enabling both efficient training and adaptive membrane potential behavior.

*   •
We conduct extensive evaluations across 19 datasets spanning five distinct tasks in both vision and language modalities. Our method demonstrates consistently competitive performance, highlighting its effectiveness and strong generalizability.

## 2. Related works

Spiking neural networks are mainly obtained through two ways: ANN-to-SNN conversion (Cao et al., [2015](https://arxiv.org/html/2604.12365#bib.bib3); Rueckauer et al., [2017](https://arxiv.org/html/2604.12365#bib.bib26); Wang et al., [2022](https://arxiv.org/html/2604.12365#bib.bib30)) and Direct Training(Fang et al., [2022](https://arxiv.org/html/2604.12365#bib.bib9); Zhou et al., [2023b](https://arxiv.org/html/2604.12365#bib.bib44), [2026b](https://arxiv.org/html/2604.12365#bib.bib40), [2024a](https://arxiv.org/html/2604.12365#bib.bib41)). This work mainly discusses the latter way.

### 2.1. Spiking Neural Networks in Vision Tasks

Spiking Neural Networks (SNNs) have gained increasing traction in visual tasks (Zhou et al., [2023b](https://arxiv.org/html/2604.12365#bib.bib44), [2026b](https://arxiv.org/html/2604.12365#bib.bib40); Yao et al., [2023](https://arxiv.org/html/2604.12365#bib.bib35), [2024](https://arxiv.org/html/2604.12365#bib.bib34); Zhou et al., [2024b](https://arxiv.org/html/2604.12365#bib.bib42); Yao et al., [2025](https://arxiv.org/html/2604.12365#bib.bib36); Fang et al., [2025](https://arxiv.org/html/2604.12365#bib.bib12)). A notable advancement is Spikformer (Zhou et al., [2023b](https://arxiv.org/html/2604.12365#bib.bib44)), which introduces Spiking Self-Attention (SSA) by replacing softmax with sparse spike-form Query, Key, and Value, achieving 74.81\% accuracy on ImageNet-1k with only four time steps—highlighting the potential of transformer-based SNNs. Building on this, Spikingformer (Zhou et al., [2026b](https://arxiv.org/html/2604.12365#bib.bib40)) incorporates a pre-activation shortcut to eliminate floating-point multiplications and reduce firing rates, boosting accuracy to 77.64\%. As a backbone architecture, it effectively unifies spike-driven computation with global modeling capabilities. Spike-Driven Transformer (Yao et al., [2023](https://arxiv.org/html/2604.12365#bib.bib35)) proposed Spike-Driven Self-Attention (SDSA), which relies solely on masking and addition, achieving 77.07% on ImageNet-1k with substantially lower computational cost. More recently, hierarchical visual spiking transformers (Yao et al., [2024](https://arxiv.org/html/2604.12365#bib.bib34); Zhou et al., [2024b](https://arxiv.org/html/2604.12365#bib.bib42); Yao et al., [2025](https://arxiv.org/html/2604.12365#bib.bib36); Fang et al., [2025](https://arxiv.org/html/2604.12365#bib.bib12)) have surpassed 80% accuracy on ImageNet while maintaining high energy efficiency.

### 2.2. Spiking Neural Networks in Language Tasks

SpikeBert (Lv et al., [2023](https://arxiv.org/html/2604.12365#bib.bib20)) extends Spikformer (Zhou et al., [2023b](https://arxiv.org/html/2604.12365#bib.bib44)) to language modeling by introducing a softmax-free spiking transformer architecture, along with a two-stage knowledge distillation strategy. Specifically, it first performs pretraining by distilling knowledge from BERT on large-scale unlabeled corpora, followed by task-specific fine-tuning using a BERT model trained on the same downstream data. SpikeGPT (Zhu et al., [2023](https://arxiv.org/html/2604.12365#bib.bib45)) builds a spiking language model upon the RWKV architecture (Peng et al., [2023](https://arxiv.org/html/2604.12365#bib.bib24)). SpikeLM (Xing et al., [2024b](https://arxiv.org/html/2604.12365#bib.bib32)) introduces a spike-based formulation with bidirectional ternary firing, yielding competitive performance in language modeling. SpikeLLM (Xing et al., [2024a](https://arxiv.org/html/2604.12365#bib.bib31)) incorporates generalized integrate-and-fire (GIF) neurons within an optimal brain spiking framework for transformer architectures. However, both SpikeLM and SpikeLLM retaining several non-spiking and nonlinear components, such as softmax-based attention, non-spiking activations in MLP layers. Zhou et al. ([2026a](https://arxiv.org/html/2604.12365#bib.bib39)) incorporates the Winner-Take-All (WTA) biological mechanism into spiking transformers and proposes the WTA-based Encoder-only Spiking Transformer (WE-SpikingFormer) for masked language modeling, as well as the WTA-based Decoder-only Spiking Transformer (WD-SpikingFormer) for causal language modeling, systematically exploring softmax-free, spike-driven transformers trained directly for language tasks.

![Image 1: Refer to caption](https://arxiv.org/html/2604.12365v1/x1.png)

Figure 1. Overview of Adaptive Spiking Neuron(ASN). (a) schematics of a biological spiking neuron. (b) depicts a neuron lacking membrane potential adaptation, thereby suppressing the transmission of membrane potential information. (c) illustrates the proposed neuron, where a learnable shift parameter enables adaptive firing, thereby ensuring better membrane potential information transmission. From the perspective of spike dynamics at the neural level. (d) and (e) show the dynamics of LIF (spike training and spike inference) and ILIF (integer training and spike inference) neurons, respectively. Both employ fixed-interval clamping, resulting in significant membrane potential quantization bias. (f) shows the spike dynamics of our proposed ASN with adaptive firing, achieving more precise spike quantization of membrane potential. 

### 2.3. Foundation Spiking Neurons

The LIF (Gerstner et al., [2014](https://arxiv.org/html/2604.12365#bib.bib13)) neuron model is the most commonly used neuron in SNNs due to its simplicity. In recent years, several improved LIF-based neurons have been proposed to better adapt to modern artificial intelligence tasks. The PLIF neuron (Fang et al., [2021](https://arxiv.org/html/2604.12365#bib.bib10)) introduces learnable membrane time constants to capture dynamic neuronal behavior. PSN (Fang et al., [2023b](https://arxiv.org/html/2604.12365#bib.bib11)) further extends LIF by removing the reset mechanism and incorporating a learnable parameter matrix, generating hidden states that are independent across time steps and enabling parallelizable neuronal dynamics. LIF, PLIF, and PSN represent key approaches within the spike training and spike inference paradigm. An alternative training approach is the integer training and spike inference paradigm. ILIF (Luo et al., [2024](https://arxiv.org/html/2604.12365#bib.bib19)) implements this paradigm by employing integer-valued activations during training while maintaining spike-driven inference through virtual timesteps. NILIF (Lei et al., [2025](https://arxiv.org/html/2604.12365#bib.bib17)) introduces a normalized version of ILIF to improve training stability for SNNs with complex architectures. SpikingBrain (Pan et al., [2025](https://arxiv.org/html/2604.12365#bib.bib23)) further adopts an adaptive threshold in ILIF by precomputing the average membrane potential based on the input; however, this input-dependent adaptation is not well-suited to the spike-driven nature of neuromorphic computation. Previous works on neurons primarily validated them in a single modality. In this work, we will simultaneously validate the effectiveness and generalizability of our methods in both visual and language modalities.

## 3. Method

### 3.1. A Functional View of Spiking Neurons

We start with a review of the principles of the three most representative neurons: LIF, ILIF, and PSN.

LIF Model. The dynamics of LIF can be formulated as follows:

(1)\displaystyle U[t]=H[t-\mathbf{1}]+X[t],
(2)\displaystyle S[t]=\Theta\left(U[t]-V_{th}\right),
(3)\displaystyle H[t]=\beta(U[t]-S[t]),

where t denotes the timestep, and X[t] is the input current at time step t. When the membrane potential U[t] exceeds the firing threshold V_{th}, the spiking neuron will trigger a spike S[t]. \Theta(v) is the Heavisine step function, which equals to 1 when v\geq 0 and 0 otherwise. U(t) will subtract the spike and then decays to H(t) by a factor of \beta. Shown in Figure [1](https://arxiv.org/html/2604.12365#S2.F1 "Figure 1 ‣ 2.2. Spiking Neural Networks in Language Tasks ‣ 2. Related works ‣ Adaptive Spiking Neurons for Vision and Language Modeling") (d), LIF employs spike training and spike inference paradigm, in which neuronal activity is constrained within a fixed range of [0, 1]. Owing to its T-dimensional temporal expansion, LIF neuron (Gerstner et al., [2014](https://arxiv.org/html/2604.12365#bib.bib13)) suffers from low training efficiency during training, making it more suitable for small to medium-scale SNN models. Moreover, LIF lacks both the capability for efficient training and adaptive dynamics.

ILIF Model. The dynamics of ILIF are formulated as follows:

\displaystyle U[t]=H[t-1]+X[t],
(4)\displaystyle S[t]=\operatorname{clip}(\operatorname{round}(U[t]),0,D),
\displaystyle H[t]=\beta(U[t]-S[t]),

where \operatorname{clip}(U[t],min,max) denotes the operation of clipping U[t] to [min,max], D indicates the maximum quantized integer value and unfold into D time steps when inference on neuromorphic chips. That is, the total time step \operatorname{T} of the spike sequence is T*D, where T is the normalized integer time step. For example, ILIF(1\times 4) unfolds into a binary spike sequence of time step \operatorname{T} = 4. Shown in Figure [1](https://arxiv.org/html/2604.12365#S2.F1 "Figure 1 ‣ 2.2. Spiking Neural Networks in Language Tasks ‣ 2. Related works ‣ Adaptive Spiking Neurons for Vision and Language Modeling") (e), ILIF and its variant NILIF(Lei et al., [2025](https://arxiv.org/html/2604.12365#bib.bib17)) employ an integer-training and spike-inference way (Luo et al., [2024](https://arxiv.org/html/2604.12365#bib.bib19)), where the membrane potential is clamped to a fixed interval of [0, D] when training. Although both ILIF and its variants enable efficient training, the absence of adaptive neuronal dynamics limits their potential as universal spiking neuron models.

PSN Model(Fang et al., [2023b](https://arxiv.org/html/2604.12365#bib.bib11)) eliminates the reset process in LIF and achieves adaptive dynamics by applying a linear transformation to the T-dimensional input membrane potential using a T^{2} parameter matrix.

(5)\displaystyle\bm{H}=\bm{W}\bm{X},\displaystyle\bm{W}\in\mathbb{R}^{T\times T},\bm{X}\in\mathbb{R}^{T\times N},
(6)\displaystyle\bm{S}=\Theta(\bm{H}-\bm{B}),\displaystyle\bm{B}\in\mathbb{R}^{T},\bm{S}\in\{0,1\}^{T\times N},

where \bm{X} is the input sequence, \bm{w} is the learnable weight matrix, \bm{H} is the hidden state sequence, \bm{B} is the learnable threshold, and \bm{S} is the binary output spike sequence. While PSN retains the T-dimensional training simulation, resulting in training efficiency that lies between that of LIF and ILIF. More critically, the parameter matrix of PSN is difficult to optimize, making its integration with Spiking Transformers an unresolved challenge and thereby limiting its general applicability.

Based on the above analysis, we present a novel functional perspective on the new generation of spiking neuron design to better accommodate vision and language modeling. We argue that a next-generation foundational spiking neuron should simultaneously satisfy four essential characteristics:

1) Efficient Training: The neuron should support efficient SNN training, making it suitable for large-scale models and modern research settings.

2) Adaptive Firing: The neuron should incorporate adaptive Firing with respect to the membrane potential to avoid excessive or insufficient spike firing.

3) Architecture Compatibility: The neuron should demonstrate cross-domain generalization (e.g., vision and language) while maintaining compatibility with advanced architectures such as spiking transformers.

4) Spike-Driven Inference: The neuron should conform to a spike-driven computational paradigm when inference to ensure high energy efficiency in SNNs.

We present a comparison of mainstream spiking neurons in Table [1](https://arxiv.org/html/2604.12365#S1.T1 "Table 1 ‣ 1. Introduction ‣ Adaptive Spiking Neurons for Vision and Language Modeling"). Our ASN family (including ASN and NASN) satisfies the four characteristics.

### 3.2. Adaptive Spiking Neuron

We propose the Adaptive Spiking Neuron (ASN), a general spike-driven neuron model that simultaneously achieves efficient training and adaptive membrane potential dynamics. By adopting the integer training and spike inference paradigm in ILIF, our ASN emits integer values during training and unfolds them into 0/1 spikes during inference, leading to the extreme efficient training. On the other hand, ASN introduces a layer-wise trainable parameter \alpha to adaptively shift the activation window, enabling better alignment with the data distribution across different layers. Shown in Figure [1](https://arxiv.org/html/2604.12365#S2.F1 "Figure 1 ‣ 2.2. Spiking Neural Networks in Language Tasks ‣ 2. Related works ‣ Adaptive Spiking Neurons for Vision and Language Modeling") (c) and (f), the learnable offset facilitates adaptive spike firing regulated by the membrane potential. Accordingly, the proposed ASN can be formulated as follows:

\displaystyle U[t]=H[t-\mathbf{1}]+X[t],
(7)\displaystyle S[t]=\operatorname{clip}(\operatorname{round}(U[t]),\alpha,\alpha+D),
\displaystyle H[t]=\beta(U[t]-S[t]),

where \operatorname{round}(*) rounds to the nearest integer, \operatorname{clip}(X,min,max) denotes that clipping X to [min,max]. \alpha is the learnable shift parameter. Since the rounding and clipping operations in Eq.[7](https://arxiv.org/html/2604.12365#S3.E7 "In 3.2. Adaptive Spiking Neuron ‣ 3. Method ‣ Adaptive Spiking Neurons for Vision and Language Modeling") is discontinuous, the gradient optimization of the input x and \alpha is shown in Section [3.4](https://arxiv.org/html/2604.12365#S3.SS4 "3.4. Gradient Optimization ‣ 3. Method ‣ Adaptive Spiking Neurons for Vision and Language Modeling").

The synaptic computation within a layer is illustrated as

(8)\mathbf{X}^{l+1}=\mathbf{W}^{l}S^{l}[t]=\mathbf{W}^{l}\operatorname{clip}(\operatorname{round}(\mathbf{U}^{l}),\alpha,\alpha+D),

where \mathbf{X}^{l} means the membrane potential before quantization at l-th layer, \mathbf{W}^{l} denotes the synaptic weights in (l)-th layer. Similar to ILIF, we convert the integer values to binary spikes in the inference stage. Linear computation in inference can be divided into spike-based computation and the addition of a constant term, which is shown as follows:

(9)\mathbf{X}^{l+1}=\mathbf{W}^{l}{S_{1}^{l}[t]+\operatorname{C}}.

We extend the T time step to T\times D, and convert the integer value S_{1}^{l}[t] to a spike sequence \left\{S^{l}[t,d]\right\}_{d=1}^{D}, which satisfied:

(10)\sum_{d=1}^{D}S^{l}[t,d]=S_{1}^{l}[t].

The constant term can be formulated as

(11)\mathbf{W}^{l}\lceil\alpha\rceil=\operatorname{C},

where \lceil*\rceil means round up operation. Compared to the inference process of ILIF, ASN only needs to add a constant \operatorname{C}=\mathbf{W}^{l+1}\lceil\alpha\rceil to each layer.

### 3.3. Normalized Adaptive Spiking Neuron

Similar to NILIF (Lei et al., [2025](https://arxiv.org/html/2604.12365#bib.bib17)), ASN may also suffer from gradient instability due to integer values. We introduce a more versatile variant by approximately normalizing the virtual step size D, which stabilizes both numerical values and gradients. This variant is termed the Normalized Adaptive Spiking Neuron (NASN). Its dynamic behavior is described as follows:

\displaystyle U[t]=H[t-\mathbf{1}]+X[t],
(12)\displaystyle S[t]=\operatorname{clip}(\operatorname{round}(U[t]),\alpha,\alpha+D)/N,
(13)\displaystyle H[t]=\beta\left(U[t]-S[t]^{*}N\right).

For simplicity, we usually set N=D. The synaptic computation within a layer is illustrated as follows:

(14)\mathbf{X}^{l+1}=\mathbf{W}^{l}S^{l}[t]=\mathbf{W}^{l}(\frac{\mathbf{1}}{N}\operatorname{clip}(\operatorname{round}(\mathbf{U}^{l}),\alpha,\alpha+D)).\\

Similar to ASN, the linear computation in Normalized ASN can also be divided into spike-based computation and the addition of a constant term in the inference stage, which is shown as

(15)\mathbf{X}^{l+1}=\mathbf{W_{1}}^{l}{S_{1}^{l}[t]+\operatorname{C}},

where {S_{1}^{l}[t]} is same as Eq. [10](https://arxiv.org/html/2604.12365#S3.E10 "In 3.2. Adaptive Spiking Neuron ‣ 3. Method ‣ Adaptive Spiking Neurons for Vision and Language Modeling"). The weight needs to be scaled as follows:

(16)\mathbf{W_{1}}^{l}=\frac{\mathbf{1}}{N}\mathbf{W_{1}}^{l}.

The constant term can be formulated as follows:

(17)\operatorname{C}=\frac{\mathbf{1}}{N}\mathbf{W}^{l+1}\lceil\alpha\rceil.

Compared to the inference process of NILIF, NASN only needs to add a constant \operatorname{C} to each layer.

Unlike conventional spiking neurons that employ fixed-interval clamping, our ASN family (including ASN and NASN) incorporates a learnable offset parameter \alpha, enabling dynamic shifts of the quantization window to better align with the evolving distribution of membrane potentials across layers. This mechanism enhances both the representational capacity and distributional adaptability of SNNs.

Table 2. The results on ImageNet-1k classification. ”E (mJ)” means ”Energy Consumption (mJ)”. Spikingformer† is the variant of Spikingformer with the improved downsampling (Zhou et al., [2023a](https://arxiv.org/html/2604.12365#bib.bib43)).

Methods Architecture Param (M)Train Size Test Size Time Step E (mJ)Top-1 Acc (\%)
SEW ResNet(Fang et al., [2022](https://arxiv.org/html/2604.12365#bib.bib9))SEW-ResNet-34 21.79 224 2 224 2 4-67.04
MS-ResNet(Hu et al., [2021](https://arxiv.org/html/2604.12365#bib.bib15))MS-ResNet-34 21.80 224 2 224 2 6-69.42
Spikformer(Zhou et al., [2023b](https://arxiv.org/html/2604.12365#bib.bib44))Spikformer-8-384 16.81 224 2 224 2 4 12.43 70.24
SD-Transformer(Yao et al., [2023](https://arxiv.org/html/2604.12365#bib.bib35))SD-Transformer-8-384 16.81 224 2 224 2 4 3.90 72.28
LIF (Spikingformer)(Zhou et al., [2026b](https://arxiv.org/html/2604.12365#bib.bib40))Spikingformer-8-384 16.81 224 2 224 2 4 4.69 72.45
LIF (Spikingformer†)(Zhou et al., [2026b](https://arxiv.org/html/2604.12365#bib.bib40))Spikingformer-8-384 16.81 224 2 224 2 4 5.61 74.35
NILIF (Spikingformer†)Spikingformer-8-384 16.81 224 2 224 2 4 5.58 75.41
Ours Spikingformer-8-384 16.81 224 2 224 2 4 5.67 75.53

Table 3. Comparision on CIFAR-10 and CIFAR-100. ”Param” denotes ”Parameter (M)”, ”Acc” denotes ”Top-1 Accuracy (%)”, and ”T” denotes ”Time Step”. Our model adopts the architecture of Spikingformer†. 

Method CIFAR-10 CIFAR-100
Param T Acc Param T Acc
Spikformer(Zhou et al., [2023b](https://arxiv.org/html/2604.12365#bib.bib44))5.76 4 94.80 5.76 4 76.95
Spikformer(Zhou et al., [2023b](https://arxiv.org/html/2604.12365#bib.bib44))9.32 4 95.51 9.32 4 78.21
SD-Transformer(Yao et al., [2023](https://arxiv.org/html/2604.12365#bib.bib35))10.28 4 95.60 10.28 4 78.4
Spikingformer(Zhou et al., [2026b](https://arxiv.org/html/2604.12365#bib.bib40))9.32 4 95.81 9.32 4 79.21
Spikingformer†(Zhou et al., [2026b](https://arxiv.org/html/2604.12365#bib.bib40))9.32 4 95.95 9.32 4 80.37
Transformer(ANN)9.32 1 96.73 9.32 1 81.02
Ours 9.32 4 96.27 9.32 4 81.11

### 3.4. Gradient Optimization

#### Gradient Estimation via STE

Since the rounding and clipping operations are non-differentiable (possessing zero derivatives almost everywhere), we employ the Straight-Through Estimator (STE) to facilitate end-to-end backpropagation.

#### Gradient of the Input x

The gradient is permitted to flow only within the active quantization range. We denate the lower bound d_{\min}=\alpha and the upper bound d_{\max}=\alpha+D . Then, the surrogate gradient for x is defined as:

(18)\frac{\partial y}{\partial x}=\begin{cases}1,&\text{if }d_{\min}<=x<=d_{\max},\\
0,&\text{otherwise}.\end{cases}

#### Gradient of the Learnable Parameter \alpha

The optimization of \alpha is primarily driven by samples falling into the truncated region (outside the boundaries). When x resides at or beyond the boundaries, the output y becomes a direct function of the boundary values, yielding the following approximation:

(19)\frac{\partial y}{\partial\alpha}=\begin{cases}1,&\text{if }x<d_{\min}\text{ or }x>d_{\max},\\
0,&\text{otherwise}.\end{cases}

Consequently, the total gradient of the objective loss \mathcal{L} with respect to \alpha is accumulated over the mini-batch as:

(20)\nabla_{\alpha}\mathcal{L}=a*\sum_{i}\frac{\partial\mathcal{L}}{\partial y_{i}}\cdot\mathbb{I}(x_{i}\notin[d_{\min},d_{\max}]),

where \mathbb{I}(\cdot) is the indicator function, and a is a hyperparameter for adjusting gradient scaling. This gradient mechanism allows the network to adaptively shift its activation window to minimize task-specific loss, providing a flexible balance between quantization error and dynamic range preservation.

To facilitate experimental evaluation, NASN (1\times 4) is used as the default configuration throughout in our main experiments. Note that NASN with the setting of 1\times 4 is equal to 4 time steps.

Table 4. The results on the Natural Language Understanding task (GLUE datasets). ”Avg.” denotes ”Average Accuracy (%)”. The results of LIF-BERT and PSN-BERT are reported in Xing et al. ([2024b](https://arxiv.org/html/2604.12365#bib.bib32)). Our model adopts the architecture of WE-SpikingFormer. 

Model Time Step MNLI QQP QNLI SST-2 CoLA STS-B MRPC RTE Avg.
BERT{}_{\texttt{base}}(Devlin et al., [2019](https://arxiv.org/html/2604.12365#bib.bib7))–83.4 71.2 90.5 93.5 52.1 85.8 88.9 66.4 79.6
Q2BERT (Zhang et al., [2020](https://arxiv.org/html/2604.12365#bib.bib38))–47.3 67.0 61.3 80.6 0.0 4.7 81.2 52.7 49.1
BERT{}_{\texttt{3L}}(Devlin et al., [2019](https://arxiv.org/html/2604.12365#bib.bib7))–77.1 85.2 85.8 88.1 31.7 85.7 86.4 66.4 75.9
SpikeLM (Xing et al., [2024b](https://arxiv.org/html/2604.12365#bib.bib32))4 77.2 83.9 85.3 87.0 38.8 84.9 85.7 69.0 76.5
LIF-BERT (Xing et al., [2024b](https://arxiv.org/html/2604.12365#bib.bib32))4 35.2 0 50.5 50.9 0 0 81.2 52.7 34.6
PSN-BERT (Xing et al., [2024b](https://arxiv.org/html/2604.12365#bib.bib32))4 35.2 0 50.5 50.9 0 6.8 81.2 52.7 34.7
SpikeBERT (Lv et al., [2023](https://arxiv.org/html/2604.12365#bib.bib20))4 71.0 68.2 66.4 85.4 16.9 18.7 82.0 57.5 59.7
NILIF (WE-SpikingFormer) (Zhou et al., [2026a](https://arxiv.org/html/2604.12365#bib.bib39))4 70.1 85.1 77.5 89.0 27.9 42.8 81.6 55.9 66.3
Ours 4 75.5 86.1 82.9 87.0 33.5 45.0 79.1 50.9 67.5

## 4. Experiments

To validate the effectiveness and generalization of the proposed adaptive neuron, we conduct extensive experiments across a diverse suite of 19 datasets covering both vision and language modeling. Specifically, our evaluation encompasses large-scale image classification on ImageNet-1K (Deng et al., [2009](https://arxiv.org/html/2604.12365#bib.bib6)), as well as medium-scale analysis on CIFAR-10/100 (Krizhevsky, [2009](https://arxiv.org/html/2604.12365#bib.bib16)). To further demonstrate its versatility, we extend our evaluation to a wide array of linguistic tasks, including Natural Language Understanding, Question Answering, and Commonsense Reasoning, providing a holistic assessment of the neuron’s performance across different modalities and scales.

### 4.1. ImageNet-1k Classification

In this part, we conduct experiments on the ImageNet-1K dataset, which contains approximately 1.28 million training images and 50,000 validation images across 1,000 object classes. We benchmark our model against several state-of-the-art SNN foundantion architectures, including CNN-based foundations such as SEW ResNet (Fang et al., [2022](https://arxiv.org/html/2604.12365#bib.bib9)) and MS-ResNet (Hu et al., [2021](https://arxiv.org/html/2604.12365#bib.bib15)), as well as Transformer-based SNN models like Spikformer (Zhou et al., [2023b](https://arxiv.org/html/2604.12365#bib.bib44)), Spike-driven Transformer (SD-Transformer) (Yao et al., [2023](https://arxiv.org/html/2604.12365#bib.bib35)), and Spikingformer (Zhou et al., [2026b](https://arxiv.org/html/2604.12365#bib.bib40)).

Among these baselines, Spikingformer represents a robust backbone that integrates spike-driven operations with global modeling capabilities, currently achieving leading performance on ImageNet-1K. For a rigorous and fair comparison, we adopt the Spikingformer-8-384 architecture as our structural framework, replacing the standard spiking neurons with our proposed NASN. As summarized in Tab.[2](https://arxiv.org/html/2604.12365#S3.T2 "Table 2 ‣ 3.3. Normalized Adaptive Spiking Neuron ‣ 3. Method ‣ Adaptive Spiking Neurons for Vision and Language Modeling"), Our model achieves 75.53\% Top-1 accuracy, exceeding the baseline LIF-based Spikingformer† by 1.18\% and outperforming NILIF-based Spikingformer† by 0.12\% under the same architectural settings. These results underscore the efficacy of the learnable offset \alpha in optimizing the firing threshold to better capture the complex data distributions of large-scale visual tasks.

### 4.2. CIFAR Datasets

To further validate the effectiveness and generalization of our method, we conduct experiments on the CIFAR-10 and CIFAR-100 datasets (Krizhevsky, [2009](https://arxiv.org/html/2604.12365#bib.bib16)). Both datasets consist of 60,000 RGB images with a resolution of 32\times 32, partitioned into 50,000 training and 10,000 testing samples. While they share the same data scale, CIFAR-100 presents a higher degree of classification complexity with 100 object categories compared to the 10 categories in CIFAR-10. These benchmarks serve as standard metrics for evaluating model performance on medium-scale visual recognition tasks.

We compared our method with Spikformer (Zhou et al., [2023b](https://arxiv.org/html/2604.12365#bib.bib44)), SD-Transformer (Yao et al., [2023](https://arxiv.org/html/2604.12365#bib.bib35)) and Spikingformer(Zhou et al., [2026b](https://arxiv.org/html/2604.12365#bib.bib40)). The quantitative results are detailed in Tab.[3](https://arxiv.org/html/2604.12365#S3.T3 "Table 3 ‣ 3.3. Normalized Adaptive Spiking Neuron ‣ 3. Method ‣ Adaptive Spiking Neurons for Vision and Language Modeling"). For a consistent evaluation, we adopt the identical architecture and experimental configurations as the baseline Spikingformer†. On CIFAR-10, our model achieves an accuracy of 96.27%, surpassing the Spikingformer† baseline by 0.32%. On the more challenging CIFAR-100 dataset, our method achieves 81.11% accuracy, yielding a more pronounced improvement of 0.74% over Spikingformer†.

Table 5. The results on Question Answering Tasks (QAT). ”E (mJ)” means ”Energy Consumption (mJ)”. ”Avg.” denotes ”Average Accuracy (%)”. Our model adopts the architecture of WD-SpikingFormer-0.4B. 

Model E (mJ)Time Step ARC-e ARC-c BoolQ HeadQA OBQA Avg.
Qwen-1.5B (Yang et al., [2024](https://arxiv.org/html/2604.12365#bib.bib33))3398.3-26.2 26.9 60.3 25.7 27.1 33.2
NILIF (WD-SpikingFormer-0.4B) (Zhou et al., [2026a](https://arxiv.org/html/2604.12365#bib.bib39))238.4 4 30.0 22.3 37.8 26.0 26.1 28.4
Ours 245.1 4 35.3 20.6 37.9 25.9 27.2 29.4

Table 6. The results on Commonsense Reasoning Tasks (CRT). ”E (mJ)” means ”Energy Consumption (mJ)”. ”Avg.” denotes ”Average Accuracy (%)” . Our model adopts the architecture of WD-SpikingFormer-0.4B. 

Model E (mJ)Time Step HellaSwag PIQA WinoGrande Avg.
SpikeLLM-7B (Xing et al., [2024a](https://arxiv.org/html/2604.12365#bib.bib31))-4 33.9 53.4 51.5 46.3
Qwen-1.5B (Yang et al., [2024](https://arxiv.org/html/2604.12365#bib.bib33))3398.3-26.5 53.7 52.8 44.3
NILIF (WD-SpikingFormer-0.4B) (Zhou et al., [2026a](https://arxiv.org/html/2604.12365#bib.bib39))238.4 4 25.9 53.4 50.2 43.2
Ours 245.1 4 26.0 54.6 49.4 43.3

### 4.3. Natural language understanding

We evaluate our method on the GLUE benchmark (Wang et al., [2018](https://arxiv.org/html/2604.12365#bib.bib29)), a widely used suite of natural language understanding tasks designed to assess the general linguistic capabilities of machine learning models across diverse domains. GLUE consists of eight tasks, and these tasks cover several categories of language understanding, including single-sentence classification (CoLA, SST-2), sentence pair classification and paraphrase detection (MRPC, QQP, RTE), semantic textual similarity (STS-B), and natural language inference (MNLI, QNLI), providing a comprehensive evaluation of a model’s ability to capture syntactic, semantic, and relational information between sentences. We pretrain the model on the Wikipedia-English corpus (Devlin et al., [2019](https://arxiv.org/html/2604.12365#bib.bib7)) using the Masked Language Modeling (MLM). The pretraining stage is conducted on 8 GPUs, following the architectural configurations and experimental settings established by WE-Spikingformer (Zhou et al., [2026a](https://arxiv.org/html/2604.12365#bib.bib39)), which uses NILIF neuron. Subsequently, we fine-tune the pretrained model on the GLUE set. For a comprehensive comparison, we include Artificial Neural Network (ANN) baselines, specifically the original BERT (Devlin et al., [2019](https://arxiv.org/html/2604.12365#bib.bib7)) and the quantized Q2BERT (Zhang et al., [2020](https://arxiv.org/html/2604.12365#bib.bib38)), the latter of which utilizes 2-bit weights and 8-bit activations.

The experimental results are summarized in Table[4](https://arxiv.org/html/2604.12365#S3.T4 "Table 4 ‣ Gradient of the Learnable Parameter 𝛼 ‣ 3.4. Gradient Optimization ‣ 3. Method ‣ Adaptive Spiking Neurons for Vision and Language Modeling"). All evaluated models, with the exception of the 3-layer BERT{}_{\texttt{3L}}, consist of 12 encoder blocks and maintain a comparable parameter scale of approximately 0.1B. Our method achieves an average accuracy of 67.5%, outperforming the NILIF version of WE-Spikingformer by 1.2%. In our comparative analysis, we note that while SpikeLM achieves higher performance, it functions as a softmax-based spiking transformer and retains non-spiking GeLU (Hendrycks and Gimpel, [2016](https://arxiv.org/html/2604.12365#bib.bib14)) activations within its MLP blocks. In contrast, our method operates as a softmax-free, spike-driven transformer. Within the category of spike-driven architectures, our method significantly outperforms existing methods such as LIF-BERT (34.6%), PSN-BERT, and SpikeBERT (59.7%), achieving a substantial performance gain with an accuracy of 67.5%. These results demonstrate that the learnable offset \alpha effectively bridges the representational gap in complex language understanding tasks.

### 4.4. Question Answering Tasks

We evaluate our model on several Question Answering Tasks (QAT) covering scientific reasoning, Boolean inference, and domain-specific knowledge. ARC-e and ARC-c (Clark et al., [2018](https://arxiv.org/html/2604.12365#bib.bib5)) (AI2 Reasoning Challenge, easy and challenge subsets) evaluate scientific knowledge and reasoning abilities. ARC-e focuses on relatively straightforward multiple-choice science questions, while ARC-c contains more challenging problems that require advanced reasoning. BoolQ (Boolean Questions) (Clark et al., [2019](https://arxiv.org/html/2604.12365#bib.bib4)) is a yes/no question-answering dataset derived from naturally occurring queries, where models must determine whether a statement is true or false given a supporting passage. HeadQA (Vilares and Gómez-Rodríguez, [2019](https://arxiv.org/html/2604.12365#bib.bib28)) is a multilingual medical question-answering benchmark constructed from professional healthcare examinations, designed to assess domain-specific medical knowledge. OBQA (OpenBookQA) (Mihaylov et al., [2018](https://arxiv.org/html/2604.12365#bib.bib22)) evaluates a model’s ability to answer elementary science questions by combining a small set of provided core facts with external common knowledge.

We pretrain our model on the FineWeb-Edu corpus (Lozhkov et al., [2024](https://arxiv.org/html/2604.12365#bib.bib18)), a high-quality subset of the FineWeb dataset specifically curated for factual and educational content. The experimental configurations follow WD-Spikingformer (Zhou et al., [2026a](https://arxiv.org/html/2604.12365#bib.bib39)), and the corresponding results are detailed in Table[5](https://arxiv.org/html/2604.12365#S4.T5 "Table 5 ‣ 4.2. CIFAR Datasets ‣ 4. Experiments ‣ Adaptive Spiking Neurons for Vision and Language Modeling"). Overall, our model achieves an average accuracy of 29.4% across the evaluated Question-Answering benchmarks, 1.0\% higher than models with an equivalent architecture but utilizing NILIF neurons. Furthermore, we evaluate the energy efficiency of our approach. Compared to Qwen-1.5B, which utilizes the same pretraining setup as WE-Spikingformer, our model reduces energy consumption by an order of magnitude, requiring only 245.1 mJ compared to 3398.3 mJ (representing only 7% of the energy). While maintaining this high energy efficiency, our model preserves a narrow accuracy gap (29.4% vs. 33.2%) against the 1.5B dense ANN baseline. These results underscore the superior energy-accuracy trade-off of our proposed method.

### 4.5. Commonsense Reasoning Tasks

We evaluate our methods on several Commonsense Reasoning Tasks (CRT) that assess a model’s ability to perform grounded inference, physical reasoning, and context-aware understanding. HellaSwag (Zellers et al., [2019](https://arxiv.org/html/2604.12365#bib.bib37)) is a large-scale benchmark for grounded commonsense inference, where models must select the most plausible continuation of a given context. PIQA (Physical Interaction Question Answering) (Bisk et al., [2020](https://arxiv.org/html/2604.12365#bib.bib2))(Bisk et al., [2020](https://arxiv.org/html/2604.12365#bib.bib2)) evaluates physical commonsense reasoning by requiring models to choose the more plausible solution to everyday tasks. WinoGrande (Sakaguchi et al., [2021](https://arxiv.org/html/2604.12365#bib.bib27)) tests commonsense reasoning through coreference resolution, requiring models to identify the correct pronoun reference that cannot be resolved by syntax alone.

The experimental configurations for our model are aligned with the protocols established by WD-Spikingformer (Zhou et al., [2026a](https://arxiv.org/html/2604.12365#bib.bib39)). The comprehensive results for commonsense reasoning are presented in Table[6](https://arxiv.org/html/2604.12365#S4.T6 "Table 6 ‣ 4.2. CIFAR Datasets ‣ 4. Experiments ‣ Adaptive Spiking Neurons for Vision and Language Modeling"). Our model achieves an average accuracy of 43.3% across these tasks, representing a 0.1% improvement over the baseline WD-Spikingformer with NILIF neuron. When compared to significantly larger models such as Qwen-1.5B and SpikeLLM-7B, our 0.4B-parameter model demonstrates remarkable parameter efficiency. Despite the substantial disparity in model scale (0.4B vs. 1.5B and 7B parameters), our model maintains a highly competitive performance level, with an average accuracy of 43.2% compared to 44.3% for Qwen-1.5B and 46.3% for SpikeLLM-7B. These findings indicate that the proposed adaptive mechanism allows the model to retain strong reasoning capabilities while utilizing a fraction of the parameters required by dense ANN or larger spiking counterparts.

![Image 2: Refer to caption](https://arxiv.org/html/2604.12365v1/x2.png)

Figure 2. Our model (WD-SpikingFormer with NASN) on parameter scaling and pretraining data scaling. (a) and (b) show increasing model parameters for pretraining on Question-Answering Tasks (QAT) and Commonsense Reasoning Tasks (CRT). (c) and (d) show increasing tokens for model pretraining on the above two Tasks. 

Table 7. Ablation study for Spiking Neurons on CIFAR-100. Base model adopts the architecture of Spikingformer†(Zhou et al., [2026b](https://arxiv.org/html/2604.12365#bib.bib40)). 

Neuron model CIFAR-100 (\%)
LIF (Zhou et al., [2026b](https://arxiv.org/html/2604.12365#bib.bib40))80.22
PLIF (Fang et al., [2021](https://arxiv.org/html/2604.12365#bib.bib10))80.25
PSN (Fang et al., [2023b](https://arxiv.org/html/2604.12365#bib.bib11))77.84
ILIF (Luo et al., [2024](https://arxiv.org/html/2604.12365#bib.bib19))80.32
ASN (Ours)80.36
NILIF (Lei et al., [2025](https://arxiv.org/html/2604.12365#bib.bib17))80.98
NASN (Ours)81.11
![Image 3: Refer to caption](https://arxiv.org/html/2604.12365v1/x3.png)

Figure 3. Neuron comparison on scaling. The base model is WD-SpikingFormer. (a) and (b) show the neuron comparison with the setting of 0.4B param with 0.1B Tokens on Question-Answering Tasks (QAT) and Commonsense Reasoning Tasks (CRT). (c) and (d) show the neuron comparison with the setting of 1.0B param with 0.5B Tokens on the above two Tasks. 

## 5. Discussion and Ablation Study

### 5.1. Discussion

In this part, we discuss the scaling characteristics of our method in the spiking transformers from the model parameter and pretraining data perspectives. The base model is the WD-SpikingFormer architecture with our NASN.

Parameter scaling. The experimental results are shown in Figure [2](https://arxiv.org/html/2604.12365#S4.F2 "Figure 2 ‣ 4.5. Commonsense Reasoning Tasks ‣ 4. Experiments ‣ Adaptive Spiking Neurons for Vision and Language Modeling") (a) and (b). We increase the model parameters from 0.4B to 1.0B. The experimental results show that the performance of the model can be further improved on both QAT and CRT. NASN (0.4B) vs. NASN (1.0B). QAT: 29.4\% vs. 29.8\%; CRT: 43.3\% vs. 43.5\%.

Pretraining data scaling. The experimental results are presented in Figure [2](https://arxiv.org/html/2604.12365#S4.F2 "Figure 2 ‣ 4.5. Commonsense Reasoning Tasks ‣ 4. Experiments ‣ Adaptive Spiking Neurons for Vision and Language Modeling") (c) and (d). In this part, we increase the tokens for NASN (1.0B) pretraining and validate it on QAT and CRT. We found that NASN (1.0B) can achieve a further significant improvement on both QAT and CRT. NASN (1.0B) with 0.1B tokens vs. NASN (1.0B) with 0.5B tokens. QAT: 29.8\% vs. 35.5\%; CRT: 43.5\% vs. 45.1\%. In particular, by improving the pretraining data of NASN (1.0B) from 0.1B tokens to 0.5B tokens, the performance of NASN (1.0B) achieves a significant performance improvement of 5.7\% on QAT, demonstrating that our methods are capable of adapting to larger-scale pretraining.

### 5.2. Ablation Study

In this part, we systematically compared the performance of the ILIF family (including ILIF and NILIF) and the ASN family (including ASN and NASN).

Ablation study on CIFAR dataset. In this part, we carry out the ablation study for foundation spiking neurons on the CIFAR-100 dataset. The results are presented in Table [7](https://arxiv.org/html/2604.12365#S4.T7 "Table 7 ‣ 4.5. Commonsense Reasoning Tasks ‣ 4. Experiments ‣ Adaptive Spiking Neurons for Vision and Language Modeling"). Our NASN achieves the best performance. Compared to the original LIF-based model, ILIF, NILIF, ASN, and NASN can improve the performance. However, ASN achieves superior performance over ILIF, while NASN outperforms NILIF. Consistent improvements are observed, which further validate the effectiveness of our method.

Neuron comparison on scaling. In this part, we compare the performance differences between NILIF and our NASN during scaling. The experimental results show in Figure [3](https://arxiv.org/html/2604.12365#S4.F3 "Figure 3 ‣ 4.5. Commonsense Reasoning Tasks ‣ 4. Experiments ‣ Adaptive Spiking Neurons for Vision and Language Modeling"). Notably, as scaling, the advantages of our method over NILIF become more pronounced, which further demonstrates that our method exhibits superior scaling properties compared to NILIF.

## 6. Conclusion

In this work, we present a unified functional perspective for designing next-generation spiking neurons, emphasizing the joint importance of training efficiency, architectural compatibility, spike-driven inference, and adaptive firing dynamics. Motivated by the limitations of existing neuron models, we propose the Adaptive Spiking Neuron (ASN), which introduces learnable membrane potential dynamics to enable flexible and effective adaptive firing while maintaining computational efficiency through an integer training and spike inference paradigm. Furthermore, we develop the Normalized Adaptive Spiking Neuron (NASN) to enhance training stability via normalization. Extensive experiments across 19 datasets spanning multiple vision and language tasks demonstrate the effectiveness and universality of our approach. Overall, this work provides both a practical neuron design and a general guideline for future research, and we believe the ASN family represents a promising step toward general-purpose spiking neurons for scalable and energy-efficient artificial intelligence.

## Limitation

Our energy consumption estimates are based on theoretical calculations and do not include measurements on real neuromorphic hardware. In addition, although we have demonstrated the effectiveness and generality of our method across 19 vision and language datasets with consistent results, further validation on neuromorphic tasks—such as DVS-based applications—remains an important direction for future work.

## References

*   (1)
*   Bisk et al. (2020) Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. 2020. Piqa: Reasoning about physical commonsense in natural language. In _Proceedings of the AAAI conference on artificial intelligence_, Vol.34. 7432–7439. 
*   Cao et al. (2015) Yongqiang Cao, Yang Chen, and Deepak Khosla. 2015. Spiking deep convolutional neural networks for energy-efficient object recognition. _International Journal of Computer Vision_ 113, 1 (2015), 54–66. 
*   Clark et al. (2019) Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. Boolq: Exploring the surprising difficulty of natural yes/no questions. _arXiv preprint arXiv:1905.10044_ (2019). 
*   Clark et al. (2018) Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. _arXiv preprint arXiv:1803.05457_ (2018). 
*   Deng et al. (2009) Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009. Imagenet: A large-scale hierarchical image database. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_. 248–255. 
*   Devlin et al. (2019) Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers)_. 4171–4186. 
*   Fang et al. (2023a) Wei Fang, Yanqi Chen, Jianhao Ding, Zhaofei Yu, Timothée Masquelier, Ding Chen, Liwei Huang, Huihui Zhou, Guoqi Li, and Yonghong Tian. 2023a. SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence. _Science Advances_ 9, 40 (2023), eadi1480. [doi:10.1126/sciadv.adi1480](https://doi.org/10.1126/sciadv.adi1480) arXiv:https://www.science.org/doi/pdf/10.1126/sciadv.adi1480 
*   Fang et al. (2022) Wei Fang, Zhaofei Yu, Yanqi Chen, Tiejun Huang, Timoth Masquelier, and Yonghong Tian. 2022. Deep Residual Learning in Spiking Neural Networks. In _Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS)_, Vol.34. 21056–21069. 
*   Fang et al. (2021) Wei Fang, Zhaofei Yu, Yanqi Chen, Timothée Masquelier, Tiejun Huang, and Yonghong Tian. 2021. Incorporating learnable membrane time constant to enhance learning of spiking neural networks. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_. 2661–2671. 
*   Fang et al. (2023b) Wei Fang, Zhaofei Yu, Zhaokun Zhou, Ding Chen, Yanqi Chen, Zhengyu Ma, Timothée Masquelier, and Yonghong Tian. 2023b. Parallel spiking neurons with high efficiency and ability to learn long-term dependencies. _Advances in Neural Information Processing Systems_ 36 (2023), 53674–53687. 
*   Fang et al. (2025) Yuetong Fang, Deming Zhou, Ziqing Wang, Hongwei Ren, ZeCui Zeng, Lusong Li, Renjing Xu, et al. 2025. Spiking Neural Networks Need High-Frequency Information. In _The Thirty-ninth Annual Conference on Neural Information Processing Systems_. 
*   Gerstner et al. (2014) Wulfram Gerstner, Werner M Kistler, Richard Naud, and Liam Paninski. 2014. _Neuronal dynamics: From single neurons to networks and models of cognition_. Cambridge University Press. 
*   Hendrycks and Gimpel (2016) Dan Hendrycks and Kevin Gimpel. 2016. Gaussian error linear units (gelus). _arXiv preprint arXiv:1606.08415_ (2016). 
*   Hu et al. (2021) Yifan Hu, Yujie Wu, Lei Deng, and Guoqi Li. 2021. Advancing residual learning towards powerful deep spiking neural networks. _arXiv preprint arXiv:2112.08954_ (2021). 
*   Krizhevsky (2009) Alex Krizhevsky. 2009. Learning multiple layers of features from tiny images. (2009). 
*   Lei et al. (2025) Zhenxin Lei, Man Yao, Jiakui Hu, Xinhao Luo, Yanye Lu, Bo Xu, and Guoqi Li. 2025. Spike2former: Efficient spiking transformer for high-performance image segmentation. In _Proceedings of the AAAI Conference on Artificial Intelligence_, Vol.39. 1364–1372. 
*   Lozhkov et al. (2024) Anton Lozhkov, Loubna Ben Allal, Leandro von Werra, and Thomas Wolf. 2024. FineWeb-Edu: the Finest Collection of Educational Content. [doi:10.57967/hf/2497](https://doi.org/10.57967/hf/2497)
*   Luo et al. (2024) Xinhao Luo, Man Yao, Yuhong Chou, Bo Xu, and Guoqi Li. 2024. Integer-valued training and spike-driven inference spiking neural network for high-performance and energy-efficient object detection. In _European Conference on Computer Vision_. Springer, 253–272. 
*   Lv et al. (2023) Changze Lv, Tianlong Li, Jianhan Xu, Chenxi Gu, Zixuan Ling, Cenyuan Zhang, Xiaoqing Zheng, and Xuanjing Huang. 2023. SpikeBERT: A Language Spikformer Learned from BERT with Knowledge Distillation. _arXiv preprint arXiv:2308.15122_ (2023). 
*   Maass (1997) Wolfgang Maass. 1997. Networks of spiking neurons: the third generation of neural network models. _Neural networks_ 10, 9 (1997), 1659–1671. 
*   Mihaylov et al. (2018) Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct electricity? a new dataset for open book question answering. _arXiv preprint arXiv:1809.02789_ (2018). 
*   Pan et al. (2025) Yuqi Pan, Yupeng Feng, Jinghao Zhuang, Siyu Ding, Han Xu, Zehao Liu, Bohan Sun, Yuhong Chou, Xuerui Qiu, Anlin Deng, et al. 2025. SpikingBrain: Spiking Brain-inspired Large Models. _arXiv preprint arXiv:2509.05276_ (2025). 
*   Peng et al. (2023) Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, et al. 2023. Rwkv: Reinventing rnns for the transformer era. _arXiv preprint arXiv:2305.13048_ (2023). 
*   Roy et al. (2019) Kaushik Roy, Akhilesh Jaiswal, and Priyadarshini Panda. 2019. Towards Spike-based Machine Intelligence With Neuromorphic Computing. _Nature_ 575, 7784 (2019), 607–617. 
*   Rueckauer et al. (2017) Bodo Rueckauer, Iulia-Alexandra Lungu, Yuhuang Hu, Michael Pfeiffer, and Shih-Chii Liu. 2017. Conversion of continuous-valued deep networks to efficient event-driven networks for image classification. _Frontiers in neuroscience_ 11 (2017), 682. 
*   Sakaguchi et al. (2021) Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. Winogrande: An adversarial winograd schema challenge at scale. _Commun. ACM_ 64, 9 (2021), 99–106. 
*   Vilares and Gómez-Rodríguez (2019) David Vilares and Carlos Gómez-Rodríguez. 2019. HEAD-QA: A healthcare dataset for complex reasoning. _arXiv preprint arXiv:1906.04701_ (2019). 
*   Wang et al. (2018) Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. 2018. GLUE: A multi-task benchmark and analysis platform for natural language understanding. _arXiv preprint arXiv:1804.07461_ (2018). 
*   Wang et al. (2022) Yuchen Wang, Malu Zhang, Yi Chen, and Hong Qu. 2022. Signed Neuron with Memory: Towards Simple, Accurate and High-Efficient ANN-SNN Conversion. In _International Joint Conference on Artificial Intelligence_. 
*   Xing et al. (2024a) Xingrun Xing, Boyan Gao, Zheng Zhang, David A Clifton, Shitao Xiao, Li Du, Guoqi Li, and Jiajun Zhang. 2024a. SpikeLLM: Scaling up spiking neural network to large language models via saliency-based spiking. _arXiv preprint arXiv:2407.04752_ (2024). 
*   Xing et al. (2024b) Xingrun Xing, Zheng Zhang, Ziyi Ni, Shitao Xiao, Yiming Ju, Siqi Fan, Yequan Wang, Jiajun Zhang, and Guoqi Li. 2024b. SpikeLM: Towards General Spike-Driven Language Modeling via Elastic Bi-Spiking Mechanisms. In _International Conference on Machine Learning_. PMLR, 54698–54714. 
*   Yang et al. (2024) An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. 2024. Qwen2 Technical Report. _arXiv preprint arXiv:2407.10671_ (2024). 
*   Yao et al. (2024) Man Yao, JiaKui Hu, Tianxiang Hu, Yifan Xu, Zhaokun Zhou, Yonghong Tian, Bo Xu, and Guoqi Li. 2024. Spike-driven transformer v2: Meta spiking neural network architecture inspiring the design of next-generation neuromorphic chips. _arXiv preprint arXiv:2404.03663_ (2024). 
*   Yao et al. (2023) Man Yao, Jiakui Hu, Zhaokun Zhou, Li Yuan, Yonghong Tian, Bo Xu, and Guoqi Li. 2023. Spike-driven transformer. _Advances in neural information processing systems_ 36 (2023), 64043–64058. 
*   Yao et al. (2025) Man Yao, Xuerui Qiu, Tianxiang Hu, Jiakui Hu, Yuhong Chou, Keyu Tian, Jianxing Liao, Luziwei Leng, Bo Xu, and Guoqi Li. 2025. Scaling spike-driven transformer with efficient spike firing approximation training. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ (2025). 
*   Zellers et al. (2019) Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really finish your sentence? _arXiv preprint arXiv:1905.07830_ (2019). 
*   Zhang et al. (2020) Wei Zhang, Lu Hou, Yichun Yin, Lifeng Shang, Xiao Chen, Xin Jiang, and Qun Liu. 2020. Ternarybert: Distillation-aware ultra-low bit bert. _arXiv preprint arXiv:2009.12812_ (2020). 
*   Zhou et al. (2026a) Chenlin Zhou, Sihang Guo, Jiaqi Wang, Dongyang Ma, Kaiwei Che, Baiyu Chen, Qingyan Meng, Zhengyu Ma, and Yonghong Tian. 2026a. Winner-Take-All Spiking Transformer for Language Modeling. arXiv:2604.11321[cs.NE] [https://arxiv.org/abs/2604.11321](https://arxiv.org/abs/2604.11321)
*   Zhou et al. (2026b) Chenlin Zhou, Liutao Yu, Zhaokun Zhou, Han Zhang, Jiaqi Wang, Huihui Zhou, Zhengyu Ma, and Yonghong Tian. 2026b. Spikingformer: A key foundation model for spiking neural networks. In _Proceedings of the AAAI Conference on Artificial Intelligence_, Vol.40. 2236–2244. 
*   Zhou et al. (2024a) Chenlin Zhou, Han Zhang, Liutao Yu, Yumin Ye, Zhaokun Zhou, Liwei Huang, Zhengyu Ma, Xiaopeng Fan, Huihui Zhou, and Yonghong Tian. 2024a. Direct training high-performance deep spiking neural networks: a review of theories and methods. _Frontiers in Neuroscience_ 18 (2024), 1383844. 
*   Zhou et al. (2024b) Chenlin Zhou, Han Zhang, Zhaokun Zhou, Liutao Yu, Liwei Huang, Xiaopeng Fan, Li Yuan, Zhengyu Ma, Huihui Zhou, and Yonghong Tian. 2024b. Qkformer: Hierarchical spiking transformer using qk attention. _Advances in Neural Information Processing Systems_ 37 (2024), 13074–13098. 
*   Zhou et al. (2023a) Chenlin Zhou, Han Zhang, Zhaokun Zhou, Liutao Yu, Zhengyu Ma, Huihui Zhou, Xiaopeng Fan, and Yonghong Tian. 2023a. Enhancing the performance of transformer-based spiking neural networks by SNN-optimized downsampling with precise gradient backpropagation. _arXiv preprint arXiv:2305.05954_ (2023). 
*   Zhou et al. (2023b) Zhaokun Zhou, Yuesheng Zhu, Chao He, Yaowei Wang, Shuicheng YAN, Yonghong Tian, and Li Yuan. 2023b. Spikformer: When Spiking Neural Network Meets Transformer. In _The Eleventh International Conference on Learning Representations_. [https://openreview.net/forum?id=frE4fUwz_h](https://openreview.net/forum?id=frE4fUwz_h)
*   Zhu et al. (2023) Rui-Jie Zhu, Qihang Zhao, Guoqi Li, and Jason K Eshraghian. 2023. Spikegpt: Generative pre-trained language model with spiking neural networks. _arXiv preprint arXiv:2302.13939_ (2023).

