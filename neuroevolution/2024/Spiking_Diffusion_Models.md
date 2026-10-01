Title: Spiking Diffusion Models

URL Source: https://arxiv.org/html/2408.16467

Markdown Content:
Back to arXiv

This is experimental HTML to improve accessibility. We invite you to report rendering errors. 
Use Alt+Y to toggle on accessible reporting links and Alt+Shift+Y to toggle off.
Learn more about this project and help improve conversions.

Why HTML?
Report Issue
Back to Abstract
Download PDF
 Abstract
1Introduction
2Related Work
3Background
4Method
5Theoretical Energy Consumption Calculation
6Implement Spiking Diffusion Models via ANN-SNN conversion
7Experiment
8Discussion and Conclusion
 References
License: arXiv.org perpetual non-exclusive license
arXiv:2408.16467v1 [cs.NE] 29 Aug 2024
Spiking Diffusion Models
Jiahang Cao∗
∗ Equal contribution. 
‡
 Second author. 
†
 Corresponding author.
Hanzhong Guo∗
Ziqing Wang∗
Deming Zhou‡
Hao Cheng
Qiang Zhang
and Renjing Xu†
Jiahang Cao, Deming Zhou, Hao Cheng, and Qiang Zhang are with the Hong Kong University of Science and Technology (Guangzhou), Guangzhou, China. Email: jcao248@connect.hkust-gz.edu.cn.Hanzhong Guo is with the Renmin University of China, Beijing, China. Email: guohanzhong@ruc.edu.cn.Ziqing Wang is with the North Carolina State University, North Carolina, USA. Email: zwang247@ncsu.edu.Renjing Xu is with the Hong Kong University of Science and Technology (Guangzhou), Guangzhou, China. Email: renjingxu@hkust-gz.edu.cn.© 20XX IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.
Abstract

Recent years have witnessed Spiking Neural Networks (SNNs) gaining attention for their ultra-low energy consumption and high biological plausibility compared with traditional Artificial Neural Networks (ANNs). Despite their distinguished properties, the application of SNNs in the computationally intensive field of image generation is still under exploration. In this paper, we propose the Spiking Diffusion Models (SDMs), an innovative family of SNN-based generative models that excel in producing high-quality samples with significantly reduced energy consumption. In particular, we propose a Temporal-wise Spiking Mechanism (TSM) that allows SNNs to capture more temporal features from a bio-plasticity perspective. In addition, we propose a threshold-guided strategy that can further improve the performances by up to 16.7% without any additional training. We also make the first attempt to use the ANN-SNN approach for SNN-based generation tasks. Extensive experimental results reveal that our approach not only exhibits comparable performance to its ANN counterpart with few spiking time steps, but also outperforms previous SNN-based generative models by a large margin. Moreover, we also demonstrate the high-quality generation ability of SDM on large-scale datasets, e.g., LSUN bedroom. This development marks a pivotal advancement in the capabilities of SNN-based generation, paving the way for future research avenues to realize low-energy and low-latency generative applications. Our code is available at https://github.com/AndyCao1125/SDM.

{IEEEImpStatement}

Diffusion models have achieved great success in image synthesis through iterative noise estimation using deep neural networks. However, the slow inference, high memory consumption, and computation intensity of the noise estimation model hinder the efficient adoption of diffusion models. In this paper, we overcame these challenges by introducing spiking diffusion models. In particular, the SDM showcased substantial energy efficiency, consuming merely 
∼
30% of the energy required by the ANN model, while still delivering superior generative outcomes. This technology could offer an alternative way of achieving sustainable, low-energy, and efficient image generation tasks.

{IEEEkeywords}

Deep Generative Models, Spiking Neural Networks, Brain-inspired Learning

1Introduction
Figure 1:Comparisons of the state-of-the-art SNN models. The FID is at 
𝑙
⁢
𝑜
⁢
𝑔
2
 scale and the marker size corresponds to the IS metric. In comparison to other SNN generative models, our models demonstrate better FID while requiring lower time steps.
\IEEEPARstart

Human brain is recognized as an amazingly intelligent system that regulates the connections between multiple neuronal layers to cope with a variety of complex tasks. Inspired by this, Artificial Neural Networks (ANNs) mimic the structural patterns of the human brain, achieving performance that approaches human levels across various fields (e.g., GPT4 [1]). However, the great success of artificial neural networks is highly dependent on a large number of intensive computations [2], whereas the human brain only requires a small power budget (
≈
20 Watts) [3, 4] and performs operations in an asynchronous and low-latency manner.

In the quest for the next wave of advanced intelligent systems, Spiking Neural Networks (SNNs), being regarded as the third generation of neural networks, are gaining interest for their distinctive attributes: high biological plausibility, event-driven nature, and low power consumption. In SNNs, all information is represented in the form of spikes (binary vectors that can either be 0 or 1) diverging from the continuous representation in ANNs, which allows SNNs to adopt spike-based low-power accumulation (AC) instead of the traditional high-power multiply-accumulation (MAC), thus achieving considerable improvements in energy efficiency. Existing works reveal that employing SNNs on neuromorphic hardware, such as Loihi [5], TrueNorth [6] and Tianjic [7] can save energy by orders of magnitude compared with ANNs. Due to the specialties of SNNs, they have been widely adopted in various tasks, including recognition [8, 9], tracking [10], detection [11, 12], segmentation [13] and images restoration [14].

Recently, deep generative models (DGMs) and especially (score-based) diffusion models [15, 16, 17, 18, 19] have made remarkable progress in various domains, including text-to-image generation [20, 21, 22, 23], audio generation [24, 25], video generation [26], text-to-3D generation [27, 28]. However, the main challenge is that the computational cost of the diffusion model is high, due to the computational complexity of the ANN backbone and the large number of iterations of the denoising process.

Hence, leveraging SNN is a potential way to develop energy-efficient diffusion models. However, SNN-based diffusion models remain little studied. Liu et al. [29] introduce Spiking-Diffusion based on the vector quantized variational autoencoder (VQ-VAE), where image features are encoded using spike firing rate and diffusion process is performed in the discrete latent space to generate images. Kamata et al. [30] propose a fully spiking variational autoencoder (FSVAE) combined with discrete Bernoulli sampling. Despite these advancements, the image produced by these SNN-based diffusion models tends to lag behind that of existing generation models [16, 31]. Consequently, it is imperative to develop a generative algorithm capable of producing high-quality images while also reducing energy consumption.

In this work, we propose Spiking Diffusion Models (SDMs), a new family of SNN-based diffusion models with excellent image generation capability and low energy consumption. SDMs can be applied to any diffusion solvers, including but not limited to DDPM [16], DDIM [32], Analytic-DPM [33] and DPM-solver [34]. Our approach adopts the pre-spike spiking UNet structure, ensuring the spikes can be transmitted more accurately. In addition, we propose Temporal-wise Spiking Mechanism (TSM), a bio-plasticity mechanism that enables the membrane potential to be dynamically updated at each time step to capture more time-dependent features. We also provide training-free threshold guidances, which further enhance the quality of resulting images by adjusting the threshold value of the spiking neurons. Experimental results demonstrate that both inhibitory and excitatory guidance provide facilitation for SDMs. Moreover, we make the first attempt to use the ANN-SNN method for providing SDMs with comparable results. We evaluate our approach on both standard datasets (e.g., CIFAR-10, and CelebA) and large-scale datasets (e.g., LSUN bedroom). As depicted in Fig. 1, our proposed SDMs outperform all SNN-based generative models by a substantial margin with only a few spiking time steps.

In summary, our contributions are:

• 

We propose Spiking Diffusion Model, a high-quality image generator that achieves state-of-the-art performances among SNN-based generative models and can be applied to any diffusion solvers.

• 

From a biological perspective, we propose a Temporal-wise Spiking Mechanism (TSM), allowing spiking neurons to capture more dynamic information to enhance the quality of denoised images.

• 

Extensive results show SDMs outperform the SNN baselines by up to 12
×
 of the FID score on the CIFAR-10 while saving 
∼
60% energy consumption. Also, we put forward a threshold-guidance strategy to further improve the generative performance.

Figure 2:Overview of our Spiking Diffusion Models. The learning process of SDM consists of two stages: (1) the training stage and (2) the fine-tuning stage. During the training stage, our spiking UNet adopts the standard Pre-spike Resblock (bottom left, Sec. 4.1), and then converts the Pre-spike block into the TSM block (bottom right, Sec. 4.2) for the fine-tuning stage. Given a random Gaussian noise input 
𝑥
𝑡
, it is firstly converted into the spike representation by a spiking encoder and subsequently fed into the spiking UNet along with the time embeddings. The network transmits only spikes which are represented by 
0
/
1
 vector (
∈
ℤ
{
0
,
1
}
). Finally, the output spikes are passed through a decoder to obtain the predicted noise 
𝜖
, and the loss is computed to update the network. In the fine-tuning phase, we load the weights from the training phase and substitute the Pre-spike block with the TSM block, where the temporal parameter 
𝑝
 is initialized as 1.0. This stage continues to optimize the network’s parameters for better generative performance.
2Related Work
2.1Training Methods of Spiking Neural Networks

Generally, ANN-to-SNN conversion and direct training are two mainstream ways to obtain deep SNN models:

Direct Training. Direct training methods involve training the SNN directly from scratch. The inherent challenge with direct training stems from the non-differentiability of the spiking neuron’s output, which complicates the application of traditional backpropagation techniques reliant on continuous gradients. To address this challenge, direct training methods utilize surrogate gradients [35, 36, 37] for backpropagation. These surrogate gradients offer a smooth approximation of the spiking function, facilitating the use of gradient-based optimization methods for SNN training.

ANN-to-SNN. The basic idea of the ANN-to-SNN conversion methods [38, 39, 40, 41] is to replace the ReLU activation values in ANNs with the average firing rates of spiking neurons. The ANN-to-SNN conversion method is generally known to achieve higher accuracy compared to direct training methods. However, they typically require a longer training time compared to direct training methods, as the conversion process requires more time steps and additional optimization which may restrict the practical SNN application.

In our study, we delve into the application of direct training methods to integrate diffusion models within SNN frameworks, aiming to reduce power consumption and investigate the potential generative abilities of SNNs. In addition, we extend our investigation to include the ANN-to-SNN conversion method for implementing SNN-based diffusion models, allowing for a comprehensive comparison of outcomes between the two training paradigms.

2.2Spiking Neural Network in Generative Models

Most of the SNN-based generation algorithms mainly originate from the generation model in ANNs, such as VAEs and GANs. [42, 43, 44] propose hybrid architectures consisting an SNN-based encoder and an ANN-based decoder. However, these approaches utilize the structure of the ANN, leading to the entire model could not be fully deployed on neuromorphic hardware. Spiking GAN [45] adopts a fully SNN-based backbone and employs a time-to-first-spike coding scheme. It significantly increases the sparsity of the spike series, therefore giving large energy savings. Kamata et al. [30] then propose a fully spiking variational autoencoder (FSVAE) that is able to transmit only spikes throughout the whole generation process sampled by the Bernoulli distribution. Feng et al. [46] construct a spiking GAN with attention scoring decoding (SGAD) and identify the problems of out-of-domain inconsistency and temporal inconsistency inherent, which increases the performance compared to existing spiking GAN-based methods. Recently, Watanabe et al. [47] propose a fully spiking denoising diffusion implicit model to achieve high speed and low energy consumption features of SNNs via synaptic current learning.

However, irrespective of whether the proposed models are based on VAE/GAN, or entirely based on spiking neurons, the primary limitation of existing spiking generative models is their low performance and poor quality of generated images. These drawbacks hinder their competitiveness in the field of generative models, despite their low energy consumption. To tackle this issue, we introduce SDMs, which not only deliver substantial improvements over existing SNN-based generative models but also preserve the advantages of SNNs.

3Background
3.1Spiking Neural Network

The Spiking Neural Network is a bio-inspired algorithm that mimics the actual signaling process occurring in brains. Compared to the Artificial Neural Network (ANN), it transmits sparse spikes instead of continuous representations, offering benefits such as low energy consumption and robustness. In this paper, we adopt the widely used Leaky Integrate-and-Fire (LIF [48, 49]) model, which effectively characterizes the dynamic process of spike generation and can be defined as:

	
𝑈
⁢
[
𝑛
]
=
𝑒
1
𝜏
⁢
𝑉
⁢
[
𝑛
−
1
]
+
𝐼
⁢
[
𝑛
]
,
		
(1)

	
𝑆
⁢
[
𝑛
]
=
Θ
⁢
(
𝑈
⁢
[
𝑛
]
−
𝜗
th
)
,
		
(2)

	
𝑉
⁢
[
𝑛
]
=
𝑈
⁢
[
𝑛
]
⁢
(
1
−
𝑆
⁢
[
𝑛
]
)
+
𝑉
reset
⁢
𝑆
⁢
[
𝑛
]
,
		
(3)

where 
𝑛
 is the time step, 
𝑈
⁢
[
𝑛
]
 is the membrane potential before reset, 
𝑆
⁢
[
𝑛
]
 denotes the output spike which equals 1 when there is a spike and 0 otherwise, 
Θ
⁢
(
𝑥
)
 is the Heaviside step function, 
𝑉
⁢
[
𝑛
]
 represents the membrane potential after triggering a spike. In addition, we use the “hard reset” method [50] for resetting the membrane potential in Eq. (3), which means that the value of the membrane potential 
𝑉
⁢
[
𝑛
]
 after triggering a spike (
𝑆
⁢
[
𝑛
]
=
1
) will go back to 
𝑉
reset
=
0
.

3.2Diffusion Models and Classifier-Free Guidance

Diffusion models leverage a forward and reverse process for data generation. They gradually perturb data with a forward diffusion process, then learn to reverse this process to recover the data distribution. The process is detailed as follows:

Formally, let 
𝑥
0
∈
ℝ
𝑛
 be a random variable with unknown data distribution 
𝑞
⁢
(
𝑥
0
)
. The forward diffusion process 
{
𝑥
𝑡
}
𝑡
∈
[
0
,
𝑇
]
 indexed by time 
𝑡
, can be represented by the following forward Stochastic differential equations (SDE):

	
d
⁢
𝑥
𝑡
=
𝑓
⁢
(
𝑡
)
⁢
𝑥
𝑡
⁢
d
⁢
𝑡
+
𝑔
⁢
(
𝑡
)
⁢
d
⁢
𝜔
,
𝑥
0
∼
𝑞
⁢
(
𝑥
0
)
,
		
(4)

where 
𝜔
∈
ℝ
𝑛
 is a standard Wiener process. Let 
𝑞
⁢
(
𝑥
𝑡
)
 be the marginal distribution of the above SDE at time 
𝑡
. Its reversal process can be described by a corresponding continuous SDE which recovers the data distribution [17]:

	
d
⁢
𝑥
=
[
𝑓
⁢
(
𝑡
)
⁢
𝑥
𝑡
−
1
+
𝜆
2
2
⁢
𝑔
2
⁢
(
𝑡
)
⁢
∇
𝑥
𝑡
log
⁡
𝑞
⁢
(
𝑥
𝑡
)
]
⁢
d
⁢
𝑡
+
𝜆
⁢
𝑔
⁢
(
𝑡
)
⁢
d
⁢
𝜔
¯
,
		
(5)

where 
𝜔
¯
∈
ℝ
𝑛
 is a reverse-time standard Wiener process, 
𝜆
 controls the andomness added while keeping the marginal distribution 
𝑞
⁢
(
𝑥
𝑡
)
 the same, and this general reversed SDE starts from 
𝑥
𝑇
∼
𝑞
⁢
(
𝑥
𝑇
)
. In Eq. (5), the only unknown term is the score function 
∇
𝑥
𝑡
log
⁡
𝑞
⁢
(
𝑥
𝑡
)
. To estimate this term, prior works [16, 17, 18] employ a noise network 
𝜖
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
 to obtain a scaled score function 
𝜎
⁢
(
𝑡
)
⁢
∇
𝑥
𝑡
log
⁡
𝑞
⁢
(
𝑥
𝑡
)
 using denoising score matching [51], which ensures that the optimal solution equals to the ground truth and satisfies 
𝜖
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
=
−
𝜎
⁢
(
𝑡
)
⁢
∇
𝑥
𝑡
log
⁡
𝑞
⁢
(
𝑥
𝑡
)
, where 
𝛼
⁢
(
𝑡
)
,
𝜎
⁢
(
𝑡
)
 represents the noise schedule and 
𝑥
𝑡
 are sample from 
𝑞
⁢
(
𝑥
𝑡
|
𝑥
0
)
∼
𝒩
⁢
(
𝑥
𝑡
|
𝑎
⁢
(
𝑡
)
⁢
𝑥
0
,
𝜎
2
⁢
(
𝑡
)
⁢
𝐼
)
.

	
𝑓
⁢
(
𝑡
)
=
d
⁢
log
⁡
𝑎
⁢
(
𝑡
)
d
⁢
𝑡
,
𝑔
2
⁢
(
𝑡
)
=
d
⁢
𝜎
2
⁢
(
𝑡
)
d
⁢
𝑡
−
2
⁢
𝜎
2
⁢
(
𝑡
)
⁢
d
⁢
log
⁡
𝑎
⁢
(
𝑡
)
d
⁢
𝑡
.
		
(6)

Hence, sampling can be achieved by discretizing the reverse SDE Eq. (5) when replacing the 
∇
𝑥
𝑡
log
⁡
𝑞
⁢
(
𝑥
𝑡
)
 with noise network. What’s more, in order to achieve conditional sampling, we can refine the reverse SDE in Eq. (5) and let 
𝜆
=
1
 as:

	
d
⁢
𝑥
=
[
𝑓
⁢
(
𝑡
)
⁢
𝑥
𝑡
−
𝑔
2
⁢
(
𝑡
)
⁢
(
1
+
𝜔
)
⁢
𝜖
𝜃
⁢
(
𝑥
𝑡
)
−
𝜔
⁢
𝜖
𝜃
⁢
(
𝑥
𝑡
,
𝑐
)
𝜎
⁢
(
𝑡
)
]
⁢
d
⁢
𝑡
+
𝑔
⁢
(
𝑡
)
⁢
d
⁢
𝜔
¯
	
	
𝜖
𝜃
⁢
(
𝑥
𝑡
,
𝑐
)
=
𝜖
𝜃
⁢
(
𝑥
𝑡
)
−
𝜎
⁢
(
𝑡
)
⁢
∇
𝑥
𝑡
log
⁡
𝑝
⁢
(
𝑐
|
𝑥
𝑡
)
,
		
(7)

where 
log
⁡
𝑝
⁢
(
𝑐
|
𝑥
𝑡
)
 is the output probability of classifier and the second equation is derived from the Bayesian formula which implies that a conditional sample can be obtained without extra training. The term 
(
1
+
𝜔
)
⁢
𝜖
𝜃
⁢
(
𝑥
𝑡
)
−
𝜔
⁢
𝜖
𝜃
⁢
(
𝑥
𝑡
,
𝑐
)
 represents classifier-free guidance, which effectively enhances the diversity of generated samples. This approach has been widely adopted in text-to-image diffusion models [52], as evidenced by the works of Ho et al. [53].

Furthermore, it is important to note that the guidance mentioned above is not limited to a specific classifier. It can be applied to various forms of guidance function. For example, energy-based guidance [54, 55] is proposed to achieve image translation and molecular design under a guidance of energy-based function. Additionally, Kim et al. [56] introduce discriminator guidance to mitigate estimation bias in the noise network, resulting in state-of-the-art performance on the CIFAR-10 dataset.

4Method
4.1Pre-spike Residual Learning

We first analyze the limitations and conceptual inconsistencies present in the residual learning approaches of previous Spiking Neural Networks (SNNs), specifically SEW ResNet [57], which can be formulated as:

	
𝑂
𝑙
	
=
BN
𝑙
⁢
(
Conv
𝑙
⁢
(
𝑆
𝑙
−
1
)
)
+
𝑆
𝑙
−
1
,
		
(8)

	
𝑆
𝑙
	
=
SpikeNeuron
⁢
(
𝑂
𝑙
)
,
		
(9)

	
𝑂
𝑙
+
1
	
=
BN
𝑙
+
1
⁢
(
Conv
𝑙
+
1
⁢
(
𝑆
𝑙
)
)
+
𝑆
𝑙
,
		
(10)

	
𝑆
𝑙
+
1
	
=
SpikeNeuron
⁢
(
𝑂
𝑙
+
1
)
,
		
(11)

where 
𝑂
𝑙
 represents the output after the batch normalization (BN) and convolutional operations at the 
𝑙
𝑡
⁢
ℎ
 layer, and 
𝑆
𝑙
 denotes the spiking neuron activation function in Eq. (2). Such a residual structure [58, 59, 57] is inherited directly from traditional artificial neural network (ANN) ResNet architectures [60]. However, a fundamental issue arises with this approach concerning the output range of the residual blocks. The core of the problem lies in the nature of the spiking neuron outputs (
𝑆
𝑙
−
1
 and 
𝑆
𝑙
), which are binary spike sequences, taking values in the set 
{
0
,
1
}
. Consequently, when these sequences are summed in the residual structure (
𝑂
𝑙
), the resulting summation output domain expands to 
{
0
,
1
,
2
}
. The occurrence of the value 
{
2
}
 in this context is non-biological and represents a departure from plausible neural activation patterns. This overflow condition not only detracts from the biological realism of the model but also potentially disrupts the effective transmission of information during forward propagation.

Inspired by [61, 62] , we adopt pre-spike residual learning with the structure of 
𝐴
⁢
𝑐
⁢
𝑡
⁢
𝑖
⁢
𝑣
⁢
𝑎
⁢
𝑡
⁢
𝑖
⁢
𝑜
⁢
𝑛
-
𝐶
⁢
𝑜
⁢
𝑛
⁢
𝑣
-
𝐵
⁢
𝑎
⁢
𝑡
⁢
𝑐
⁢
ℎ
⁢
𝑁
⁢
𝑜
⁢
𝑟
⁢
𝑚
 in our Spiking UNet, addressing the dual challenges of gradient explosion/vanishing and performance degradation in convolution-based SNNs. Through the pre-spike blocks, the residuals and outputs are summed by floating point addition operation, ensuring that the representation is accurate before entering the next spiking neuron while avoiding the pathological condition mentioned above. The whole pre-spike residual learning process inside a resblock can be formulated as below:

	
𝑆
𝑙
	
=
SpikeNeuron
⁢
(
𝑂
𝑙
−
1
)
,
		
(12)

	
𝑂
𝑙
	
=
BN
𝑙
⁢
(
Conv
𝑙
⁢
(
𝑆
𝑙
)
)
+
𝑂
𝑙
−
1
,
		
(13)

	
𝑆
𝑙
+
1
	
=
SpikeNeuron
⁢
(
𝑂
𝑙
)
,
		
(14)

	
𝑂
𝑙
+
1
	
=
BN
𝑙
+
1
⁢
(
Conv
𝑙
+
1
⁢
(
𝑆
𝑙
+
1
)
)
+
𝑂
𝑙
.
		
(15)

Through the pre-spike residual mechanism, the output of the residual block can be summed by two floating points 
𝐵
⁢
𝑁
𝑙
⁢
(
𝐶
⁢
𝑜
⁢
𝑛
⁢
𝑣
𝑙
⁢
(
𝑆
𝑙
)
)
, 
𝑂
𝑙
−
1
 at the same scale and then enter the spiking neuron at the beginning of the next block, which guarantees that the energy consumption is still very low.

4.2Temporal-wise Spiking Mechanism

In this section, we first revisit the shortcomings of traditional SNNs from a biological standpoint and propose a novel TSM mechanism that fine-tunes the weights to capture temporal dynamics by incorporating temporal parameters. Considering the spike input at 
𝑙
-th layer as 
𝑆
𝑙
∈
ℝ
𝑁
×
𝑇
×
𝐶
𝑖
⁢
𝑛
×
𝐻
×
𝑊
, where 
𝑁
 denotes the minibatch size. For every time step 
𝑛
∈
{
1
,
2
,
…
,
𝑇
}
, the neuron will update its temporary membrane potential via Eq. (1), where 
𝐼
𝑙
−
1
⁢
[
𝑛
]
=
𝐖
𝑙
⁢
𝑆
𝑙
−
1
⁢
[
𝑛
]
 and 
𝐖
𝑙
∈
ℝ
𝑁
×
𝐶
𝑜
⁢
𝑢
⁢
𝑡
×
𝐻
×
𝑊
 denotes the weight matrix of the 
𝑙
-th convolutional layer. The traditional SNNs [58] will fuse the 
𝑇
 and 
𝐶
 dimension of the input into 
𝑆
~
𝑙
∈
ℝ
𝑁
⁢
𝑇
×
𝐶
𝑖
⁢
𝑛
×
𝐻
×
𝑊
 when performing the membrane potential update, and then compute it by 2D convolution operation. This leads to the input of each time step being operated by the same weight matrix. However, in the real nervous system, cortical pyramidal cells receive strong barrages of both excitatory and inhibitory postsynaptic potentials during regular network activity [63]. Moreover, different states of arousal can alter the membrane potential and impact synaptic integration [64]. These studies collectively demonstrate that the neuron input at each moment experiences considerable fluctuations due to the network states and other factors, rather than being predominantly controlled by fixed synaptic weights.

In order to provide more dynamic information through time, we propose the Temporally-wise Spiking Mechanism (TSM, see Fig. 3), which guarantees that the input information at each moment is computed with a temporal parameter 
𝑝
⁢
[
𝑛
]
∈
𝐏
 (
𝐏
∈
ℝ
𝑇
) related to time step 
𝑛
:

	
𝑈
𝑙
⁢
[
𝑛
]
=
𝑒
1
𝜏
⁢
𝑉
𝑙
⁢
[
𝑛
−
1
]
+
(
𝐖
𝑙
⁢
𝑆
𝑙
−
1
⁢
[
𝑛
]
)
×
𝑝
⁢
[
𝑛
]
		
(16)

We describe the whole learning process of SDMs precisely in Alg. 1. Specifically, the learning pipeline consists of (1) the training stage and (2) the fine-tuning stage, where we first train the SDMs with pre-spike block and fine-tune the model by utilizing TSM block. We only use few iterations 
𝐼
𝑓
⁢
𝑡
 (
<
1
10
 training iterations) to fine-tune our model. It is worth noting that, since 
𝑝
⁢
[
𝑡
]
 is only a scalar, the extra computational costs incurred by TSM are negligible. By calculating 
ℒ
𝑀
⁢
𝑆
⁢
𝐸
, we could further optimize the parameters of 
𝐖
 and 
𝐏
 to obtain satisfying networks. The detailed learning rules of SNN-UNet and 
𝑝
⁢
[
𝑡
]
 can be found in Appendix.

In summary, TSM allows the membrane potential to be dynamically updated across the time domain, improving the ability to capture potential time-dependent features. Subsequent experiments demonstrate that the TSM mechanism is superior to the traditional fixed update mechanism. We also visualized 
𝐏
 for detailed analysis in Sec. 7.6.

Figure 3:Overview of temporal-wise spiking mechanism. After a spike neuron triggers spikes, the spikes would be converted in the pre-synapse to obtain the input current 
𝐼
. To derive more dynamic information, the temporal parameter 
𝑃
 will act on the current to form the time-adaptive current 
𝐼
^
.
4.3Threshold Guidance in SDMs

As mentioned in Sec. 3.2, the sampling can be achieved by substituting the score 
∇
𝑥
𝑡
log
⁡
𝑞
⁢
(
𝑥
𝑡
)
 with either the score network 
𝑠
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
 or the scaled noise network -
𝜖
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
𝜎
⁢
(
𝑡
)
 while discretizing the reverse SDE as presented in Eq. (5). Because of the inaccuracy of the network estimates, we have the fact that 
𝑠
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
≈
−
𝜖
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
𝜎
⁢
(
𝑡
)
≠
∇
𝑥
𝑡
log
⁡
𝑞
⁢
(
𝑥
𝑡
)
 in most cases. Therefore, in order to sample better results, we can discretize the following rectified reverse SDE:

	
d
⁢
𝑥
=
[
𝑓
⁢
(
𝑡
)
⁢
𝑥
𝑡
−
𝑔
2
⁢
(
𝑡
)
⁢
[
𝑠
𝜃
+
𝑐
𝜃
]
⁢
(
𝑥
𝑡
,
𝑡
)
]
⁢
d
⁢
𝑡
+
𝑔
⁢
(
𝑡
)
⁢
d
⁢
𝜔
¯
,
		
(17)

Here, 
𝑠
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
 denotes the score network or scaled noise network, while 
𝑐
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
=
∇
𝑥
𝑡
log
⁡
𝑞
⁢
(
𝑥
𝑡
)
𝑝
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
 represents the rectified term for the original reverse SDE. The omission of the rectified term 
𝑐
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
 results in reduced discretization error and improved sampling performance. However, the practical calculation of 
𝑐
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
 poses challenges due to its intractability.

Given the presence of estimation error, Eq. (17) prompts us to investigate whether we can enhance sampling performance without additional training by calculating 
𝑐
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
. A crucial parameter in the SNN is the spike threshold, denoted as 
𝜗
th
, which directly influences the SNN’s output. For instance, a smaller threshold encourages more spike occurrences, while a larger threshold suppresses such occurrences. Following the training process, we can adjust the threshold within the SNN to estimate the rectified term 
𝑐
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
 as outlined below:

		
𝑠
𝜃
⁢
(
𝑥
𝑡
,
𝑡
,
𝜗
th
′
)
	
		
≈
𝑠
𝜃
⁢
(
𝑥
𝑡
,
𝑡
,
𝜗
th
0
)
+
d
⁢
𝑠
𝜃
⁢
(
𝑥
𝑡
,
𝑡
,
𝜗
th
)
d
⁢
𝜗
th
⁢
d
⁢
𝜗
th
+
𝑂
⁢
(
d
⁢
𝜗
th
)
	
		
≈
𝑠
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
+
𝑠
𝜃
′
|
𝜗
th
0
⁢
d
⁢
𝜗
th
+
𝑂
⁢
(
d
⁢
𝜗
th
)
	
		
≈
𝑠
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
+
𝑐
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
,
		
(18)

where 
𝜗
th
0
 is the threshold used in the training stage and 
𝜗
th
′
 is the adjusted threshold used in the inference stage. The first equation is obtained by Talyor expansion. Eq. (4.3) tells that when the derivative term is related to the rectified term, adjusting the threshold can boost the final sampling results. We call the threshold-decreasing case as the inhibitory guidance, and conversely as excitatory guidance.

Algorithm 1 The Learning Pipeline of the SDM.
0:  the number of finetune iterations 
𝐼
𝑓
⁢
𝑡
, the number of SNN layers 
𝐿
𝑠
⁢
𝑛
⁢
𝑛
, spiking time steps 
𝑇
𝑠
⁢
𝑛
⁢
𝑛
, encoding layer 
ℰ
, diffusion process 
𝒟
, decay factor 
𝛾
, threshold 
𝜗
th
, learning rate 
𝛼
0:  Optimized 
𝑊
, 
𝑝
  
1:  Training spiking diffusion models with the standard pre-spike residual block (Sec. 4.1).
▷
 Stage 1
2:  Prepare to finetune the spiking diffusion models with the TSM residual block (Sec. 4.2).
▷
 Stage 2
3:  Inherit pretrained weight 
𝑊
 and initialize temporal parameter 
𝑝
 with 1.0.
4:  for 
𝑖
=
1
 to 
𝐼
𝑓
⁢
𝑡
 do
5:     Initialize Gaussian noise 
𝜖
∼
𝒩
⁢
(
0
,
I
)
.
6:     Calculate the input spikes 
𝑆
0
 = 
ℰ
⁢
(
𝜖
)
.
7:     for 
𝑙
=
1
 to 
𝐿
𝑠
⁢
𝑛
⁢
𝑛
 do
8:        for every neuron 
𝑖
 in layer 
𝑙
 do
9:           for 
𝑡
=
1
 to 
𝑇
𝑠
⁢
𝑛
⁢
𝑛
 do
10:              
𝑈
𝑖
𝑙
⁢
[
𝑡
]
 = 
𝑒
1
𝛾
⋅
𝑈
𝑖
𝑙
⁢
[
𝑡
−
1
]
⁢
(
1
−
𝑆
𝑖
𝑙
⁢
[
𝑡
−
1
]
)
+
𝐼
^
𝑖
𝑙
−
1
⁢
[
𝑡
]
11:              
𝑆
𝑖
𝑙
⁢
[
𝑡
]
 = 
Θ
⁢
(
𝑈
𝑖
𝑙
⁢
[
𝑡
]
−
𝜗
th
)
12:              
𝐼
𝑖
𝑙
⁢
[
𝑡
]
 = 
𝐖
𝑖
𝑙
+
1
⁢
𝑆
𝑖
𝑙
⁢
[
𝑡
]
13:              
𝐼
^
𝑖
𝑙
⁢
[
𝑡
]
 = 
𝐼
𝑖
𝑙
⁢
[
𝑡
]
×
𝑝
⁢
[
𝑡
]
14:           end for
15:        end for
16:     end for
17:     
𝜖
𝑜
⁢
𝑢
⁢
𝑡
 = 
𝒟
(
𝑈
𝐿
[
1
:
𝑇
𝑠
⁢
𝑛
⁢
𝑛
]
)
18:     
{
𝑊
,
𝑝
}
←
{
𝑊
,
𝑝
}
−
𝛼
⁢
∇
{
𝑊
,
𝑝
}
⋅
ℒ
𝑀
⁢
𝑆
⁢
𝐸
⁢
(
𝜖
,
𝜖
𝑜
⁢
𝑢
⁢
𝑡
)
19:  end for


5Theoretical Energy Consumption Calculation

In this section, we describe the methodology for calculating the theoretical energy consumption of the Spiking UNet architecture. The calculation involves two main steps: determining the synaptic operations (SOPs) for each block within the architecture, and then estimating the overall energy consumption based on these operations. The synaptic operations for each block of the Spiking UNet can be quantified as follows [8]:

	
SOPs
⁡
(
𝑙
)
=
𝑓
⁢
𝑟
×
𝑇
×
FLOPs
⁡
(
𝑙
)
		
(19)

where 
𝑙
 denotes the block number in the Spiking UNet, 
𝑓
⁢
𝑟
 is the firing rate of the input spike train of the block and 
𝑇
 is the time step of the spike neuron. 
FLOPs
⁡
(
𝑙
)
 refers to floating point operations of 
𝑙
 block, which is the number of multiply-and-accumulate (MAC) operations. And SOPs are the number of spike-based accumulate 
(
AC
)
 operations.

To estimate the theoretical energy consumption of Spiking Diffusion, we assume that the MAC and AC operations are implemented on a 
45
⁢
𝑛
⁢
𝑚
 hardware, with energy costs of 
𝐸
𝑀
⁢
𝐴
⁢
𝐶
=
4.6
⁢
𝑝
⁢
𝐽
 and 
𝐸
𝐴
⁢
𝐶
=
0.9
⁢
𝑝
⁢
𝐽
, respectively. According to [65, 66], the calculation for the theoretical energy consumption of Spiking Diffusion is given by:

	
𝐸
Diffusion
	
=
𝐸
𝑀
⁢
𝐴
⁢
𝐶
×
FLOP
SNN
Conv
1
		
(20)

		
+
𝐸
𝐴
⁢
𝐶
×
(
∑
𝑛
=
2
𝑁
SOP
SNN
Conv
𝑛
+
∑
𝑚
=
1
𝑀
SOP
SNN
FC
𝑚
)
	

where 
𝑁
 and 
𝑀
 represent the total number of layers of convolutional (Conv) and fully connected (FC) layers, respectively. 
𝐸
𝑀
⁢
𝐴
⁢
𝐶
 and 
𝐸
𝐴
⁢
𝐶
 energy costs per operation for MAC and AC, respectively. 
FLOP
SNN
Conv
 refers to the FLOPs of the first Conv layer, and 
SOP
SNN
Conv
 and 
SOP
SNN
FC
 are the SOPs for the 
𝑛
𝑡
⁢
ℎ
 Conv and 
𝑚
𝑡
⁢
ℎ
 FC layers, respectively.

6Implement Spiking Diffusion Models via ANN-SNN conversion

In this paper, for the first time, we also utilize the ANN-SNN approach to successfully implement SNN diffusion. we adopt the Fast-SNN [67] approach to construct the conversion between quantized ANNs and SNNs. Since this implementation is not the main contribution of our paper, we briefly describe the ANN-SNN principle and more details can be found in [67].

The core idea of the conversion from ANNs to SNNs is to map the integer activation of quantized ANNs 
{
0
,
1
,
…
,
2
𝑏
−
1
}
 to spike count 
{
0
,
1
,
…
,
𝑇
}
, i.e., convert 
𝑇
 to 
2
𝑏
−
1
. Building quantized ANNs with integer activations is naturally equivalent to compressing activations with the quantization function that outputs uniformly distributed values. Such a function spatially discretizes a full-precision activation 
𝑥
𝑖
𝑙
 of neuron 
𝑖
 at layer 
𝑙
 in an ANN of ReLU activation into:

	
𝑄
𝑖
𝑙
=
𝑠
𝑙
2
𝑏
−
1
⁢
clip
⁢
(
round
⁢
(
(
2
𝑏
−
1
)
⁢
𝑥
𝑖
𝑙
𝑠
𝑙
)
,
0
,
2
𝑏
−
1
)
,
		
(21)

where 
𝑄
𝑖
𝑙
 denotes the spatially quantized value, 
𝑏
 denotes the number of bits (precision), the number of states is 
2
𝑏
−
1
, 
round
⁢
(
⋅
)
 denotes a rounding operator, 
𝑠
𝑙
 denotes the clipping threshold that determines the clipping range of input 
𝑥
𝑖
𝑙
, 
clip
⁢
(
𝑥
,
𝑚
⁢
𝑖
⁢
𝑛
,
𝑚
⁢
𝑎
⁢
𝑥
)
 is a clipping operator that saturates 
𝑥
 within range 
[
𝑚
⁢
𝑖
⁢
𝑛
,
𝑚
⁢
𝑎
⁢
𝑥
]
.

In SNNs, the spiking IF neuron inherently quantizes the membrane potential 
𝑈
𝑖
𝑙
 into a quantized value represented by the firing rate 
𝑟
𝑖
𝑙
:

	
𝑄
~
𝑖
𝑙
=
𝑟
𝑖
𝑙
=
1
𝑇
⁢
clip
⁢
(
floor
⁢
(
𝑈
𝑖
𝑙
𝜃
𝑙
)
,
0
,
𝑇
)
,
		
(22)

where 
𝑄
~
𝑖
𝑙
 denotes the spiking-based quantized value, 
floor
⁢
(
⋅
)
 denotes a flooring operator. Assume that the value of membrane potential is always satisfied to be 
𝑇
 times the value of the input current: 
𝑈
𝑖
𝑙
=
𝑇
⁢
𝐼
𝑖
𝑙
. Comparing Eq. 22 with Eq. 21, we let 
𝜇
𝑙
=
𝜃
𝑙
/
2
, 
𝑇
=
2
𝑏
−
1
, 
𝜃
𝑙
=
𝑠
𝑙
. Since a flooring operator can be converted to a rounding operator:

	
floor
⁢
(
𝑥
+
0.5
)
=
round
⁢
(
𝑥
)
,
		
(23)

By scaling the weights in the following layer to 
𝑠
𝑙
⁢
𝑊
𝑙
+
1
, we rewrite Eq. 22 into the following equation :

	
𝑄
~
𝑖
𝑙
=
𝑠
𝑙
𝑇
⁢
clip
⁢
(
floor
⁢
(
𝑇
⁢
𝐼
𝑖
𝑙
𝜃
𝑙
)
,
0
,
𝑇
)
=
𝑄
𝑖
𝑙
.
		
(24)

Hence, by establishing the equivalence of discrete ReLU value and spike firing rate (Eq. 24), we build a bridge between quantized ANNs and SNNs. It is important to note that the assumption of 
𝑈
𝑖
𝑙
=
𝑇
⁢
𝐼
𝑖
𝑙
 only holds in the first spiking layer which directly receives currents as inputs. However, as the network goes deeper, the interplay between membrane potential and input current grows increasingly complex, deviating from a simple linear relationship. This complexity is one of the fundamental reasons for the progressively larger error accumulation in the ANN-SNN conversion process.

7Experiment
7.1Experiment Settings

Datasets and Evaluation Metrics. To demonstrate the effectiveness and efficiency of the proposed algorithm, we conduct experiments on 32
×
32 MNIST [68], 32
×
32 Fashion-MNIST [69], 32
×
32 CIFAR-10 [70] and 64
×
64 CelebA [71]. The qualitative results are compared according to Fréchet Inception Distance (FID [72], lower is better) and Inception Score (IS [73], higher is better). FID score is computed by comparing 50,000 generated images against the corresponding reference statistics of the dataset.

Implementation Details. For the direct training method, our Spiking UNet inherits the standard UNet [74] architecture and no attention blocks are used. For the hyper-parameter settings, we set the decay rate 
𝑒
1
𝜏
 in Eq. (1) as 1.0 and the spiking threshold 
𝜗
th
 as 1.0. The SNN simulation time step is 4/8. The learning rate is set as 1 e-5 with batch size 128 and we train the model without exponential moving average (EMA [75]). ANN UNet also does not employ attention blocks, and its training process is consistent with SNN-UNet. For the ANN-SNN method, we use the same implementation of Fast-SNN [67], but we do not apply the signed-IF neuron since this neuron plays a negative role in the diffusion task. More details of the hyperparameter settings can be found in the Appendix.

Table 1:Results for different dataset. In all datasets, SDMs (Ours) outperform all SNN baselines and even some ANN models in terms of sample quality, which is mainly measured by FID and IS. Results of ▽ are taken from [30] and results of ♮ are taken from [46]. 
ema
 indicates the utilization of EMA [75] method. For fair comparisons, we re-evaluate the results of DDPM [16] and DDIM [32] using the same UNet architecture as SDMs. ∗ denotes that only FID is used for MNIST, Fashion-MNIST and CelebA since their data distributions are far from ImageNet, making Inception Score less meaningful. The top-1 and top-2 results are bold and underlined, respectively.
Dataset	Model	Method	#Param (M)	Time Steps	IS
↑
	FID
↓

MNIST∗	VAE▽ [76]	ANN	1.13	/	5.947	112.50
Hybrid GAN♮ [44] 	SNN&ANN	-	16	-	123.93
FSVAE [30] 	SNN	3.87	16	6.209	97.06
SGAD [46] 	SNN	-	16	-	69.64
Spiking-Diffusion [29] 	SNN	-	16	-	37.50
	SDDPM	SNN	63.61	4	-	29.48
Fashion
MNIST∗	VAE [76]	ANN	1.13	/	4.252	123.70
Hybrid GAN [44] 	SNN&ANN	-	16	-	198.94
FSVAE [30] 	SNN	3.87	16	4.551	90.12
SGAD [46] 	SNN	-	16	-	165.42
Spiking-Diffusion [29] 	SNN	-	16	-	91.98
	SDDPM	SNN	63.61	4	-	21.38
LSUN bedroom∗	DDPM [16]	ANN	64.47	/	-	29.48
SDDPM	SNN	63.61	4	-	47.64
CelebA∗	VAE [76]	ANN	3.76	/	3.231	92.53
Hybrid GAN [44] 	SNN&ANN	-	16	-	63.18
DDPM [16] 	ANN	64.47	/	-	20.34
FSVAE [30] 	SNN	6.37	16	3.697	101.60
SGAD [46] 	SNN	-	16	-	151.36
FSDDIM [47] 	SNN	-	4	-	36.08
SDDPM	SNN	63.61	4	-	25.09
CIFAR-10	VAE [76]	ANN	1.13	/	2.591	229.60
Hybrid GAN [44] 	SNN&ANN	-	16	-	72.64
DDIM [32] 	ANN	64.47	/	8.428	18.49
DDPM [16] 	ANN	64.47	/	8.380	19.04
DDIM
ema
 [32] 	ANN	64.47	/	8.902	12.13
DDPM
ema
 [16] 	ANN	64.47	/	8.846	13.38
FSVAE [30] 	SNN	3.87	16	2.945	175.50
SGAD [46] 	SNN	-	16	-	181.50
Spiking-Diffusion [29] 	SNN	-	16	-	120.50
FSDDIM [47] 	SNN	-	4	-	51.46
FSDDIM [47] 	SNN	-	8	-	46.14
SDDPM	SNN	63.61	4	7.440	19.73
SDDPM	SNN	63.61	8	7.584	17.27
SDDPM (TSM)	SNN	63.61 (+2
𝑒
−
4
)	4	7.654	18.57
SDDPM (TSM)	SNN	63.61 (+2
𝑒
−
4
)	8	7.814	15.45
SDDIM (TSM)	SNN	63.61 (+2
𝑒
−
4
)	8	7.834	16.02
SDPM-Solver (TSM)	SNN	63.61 (+2
𝑒
−
4
)	8	7.180	30.85
	Analytic-SDPM (TSM)	SNN	63.61 (+2
𝑒
−
4
)	8	7.844	12.95
7.2Comparisons with the state-of-the-art

In Tab. 1, we present a comparative analysis of our Spiking Diffusion Models (SDMs) with state-of-the-art generative models in unconditional generations. We also include ANN results for reference. Qualitative results are shown in Fig. 4. Our results demonstrate that SDMs outperforms SNN baselines across all datasets by a significant margin, even with smaller spiking simulation steps (4/8). In particular, SDDPM has 4
×
 and 6
×
 FID improvement in CelebA, while 11
×
 and 12
×
 enhancement in CIFAR-10 compared to FSVAE and SGAD (both are with 16 time steps). As expected, the sample quality becomes higher as we increase the time step. We additionally note that the incorporation of TSM leads to enhance performance with only a negligible increase in model parameters (2
𝑒
−
4
 M). SDMs can also handle fast sampling solvers [33, 34] and attain higher sampling quality in fewer steps (see Tab. 6). Importantly, SDMs obtain comparable quality to ANN baselines with the same UNet architecture and even surpasses ANN models(e.g., 15.45 vs. 19.04). This outcome highlights the superior expressive capability of SNNs employed in our model.

Table 2:Comparisons of diffusion models with direct training and ANN-SNN conversion. While ANN-SNN approaches always outperform direct training methods in classification tasks, their effectiveness falls short when applied to generative tasks, where direct training yields superior results.
Model	Method	Time Steps	IS
↑
	FID
↓

SDDPM	Direct Training	4	7.645	18.57
SDDPM	Direct Training	8	7.814	15.45
SDDPM (w/o ft)	ANN-SNN	3	5.484	51.18
SDDPM (w ft)	ANN-SNN	3	7.223	29.53
7.3Comparisons with the ANN-SNN method

To validate the generative ability of SDM under the ANN-SNN approach, we conduct experiments on the 32
×
32 CIFAR-10 and 64
×
64 FFHQ [77] datasets. As shown in Tab. 2, the ANN-SNN approach achieves great performance on CIFAR-10 (i.e., 51.18 FID) and improves the image quality substantially after the fine-tuning strategy (i.e., 29.53 FID). However, there is still a gap between the results of ANN-SNN and those of direct training. Although ANN-SNN methods are demonstrated to have comparable performances to ANN’s in classification-based tasks, there is still a lack of in-depth research on generative tasks. The qualitative results of the ANN-SNN method are illustrated in Fig. 7.









Figure 4:Unconditional image generation results on MNIST, Fashion-MNIST, CIFAR-10, CelebA, and LSUN-bed by using direct training-based SDMs.
Table 3:Results on CIFAR-10 by different threshold guidances. The top-1 and top-2 results are bold and underlined, respectively.
      Method	      Threshold	      FID
↓
	      IS
↑

      Baseline	      1.000	      19.73	      7.44
       
Inhibitory
Guidance
	      0.999	      19.25	      7.48
      0.998	      19.38	      7.55
      0.997	      19.20	      7.47
       
Excitatory
Guidance
	      1.001	      20.00	      7.47
      1.002	      19.98	      7.48
      1.003	      20.04	      7.46
7.4Effectiveness of the Temporal-wise Spiking Mechanism

To better visualize the performance improvement brought by the TSM module, we provide the generation results of CIFAR-10 using SDDIM with and without the TSM module. Here we use DDIM [32] instead of DDPM [16] for this comparison since DDIM operates based on Ordinary Differential Equations (ODEs), which ensure deterministic and consistent generation results. In contrast, DDPM relies on Stochastic Differential Equations (SDEs), which introduce randomness in the generation process, leading to variability in the output images and making direct comparisons challenging.

The results in Fig. 5 demonstrate a significant improvement in the quality of the generated images with the TSM module. The contours of the images are more pronounced, the backgrounds are clearer, and the texture details are richer compared to those without the TSM module, thereby proving the effectiveness of TSM.

Figure 5:Comparisons of the generation results with/without using the TSM method in CIFAR-10.
7.5Effectiveness of Threshold Guidance

In Sec. 4.3, we propose a training-free method: Threshold Guidance (TG), designed to enhance the quality of generated images by merely adjusting the threshold levels of spiking neurons slightly during the inference phase. As depicted in Tab. 3, applying inhibitory guidance through threshold adjustment significantly elevates image quality across two key metrics: the FID score decreases from 19.73 to 19.20 with a threshold reduction of 0.3%, and the IS score climbs from 7.44 to 7.55 following a 0.2% threshold decrease. Conversely, excitatory guidance similarly augments the sampling quality under certain conditions. These findings underscore the potency of threshold guidance as a means to substantially improve model efficacy post-training, without necessitating extra training resources. We provide more explanations about the threshold guidance in the Appendix.

Table 4:Comparisons of energy and FID of SNN and ANN models. In comparison to ANN, SNN models demonstrate lower energy consumption while achieving comparable FID results.
Model	DDPM-ANN	SDDPM-4T	SDDPM-8T
FID
↓
	19.04	18.57	15.45
Energy (mJ)
↓
	29.23	10.97	22.96
7.6Analysis of TSM method

To evaluate the efficacy of the Temporal-wise Spiking Mechanism (TSM) proposed in Sec. 4.2, we compute the mean of the temporal parameters over all layers. As depicted in Fig. 6, each instance presents distinct TSM values 
𝑝
⁢
[
𝑡
]
, which underscores the unique significance attributed to each time step. We notice that the evolutionary trend of 
𝑝
⁢
[
𝑡
]
 exhibits an increasing pattern as the number of time steps grows, suggesting that the information conveyed in later stages holds greater importance during the transmission process. As a result, the TSM values can serve as temporal adjusting factors, enabling the SNN to comprehend and incorporate temporal dynamics, thus improving the quality of the generated image.

Figure 6:Visualization of TSM Value. The averaged TSM values across all layers are depicted, with the red error bar.
Table 5:Ablation study on different proposed methods. The experiments are conducted on SDDIM (T=4) with 50 denoising steps. 
Δ
 represents the improvement of FID.
Solver	TSM	TG	FID
↓
	
Δ
 (%)

SDDIM
			20.68 (-0.00)	+0.00
✓		20.26 (-0.42)	+2.03
	✓	19.62 (-1.06)	+5.12
✓	✓	16.88 (-3.80)	+18.4
7.7Evaluation of the Computational Cost

To further emphasize the low-energy nature of our SDMs, we perform a comparative analysis of the FID and energy consumption between the proposed SDDPM and its corresponding ANN model. As shown in Tab. 4, when the time step is set at 4, the SDDPM presents significantly lower energy consumption, amounting to merely 37.5% of that exhibited by its ANN counterpart. Moreover, the FID of SDDPM also improved by 0.47, indicating that our model can effectively minimize energy consumption while maintaining competitive performance. As we extend the analysis to include varying time step increments, a discernible pattern emerges: the FID score improves as the time step increases, albeit at the expense of higher energy consumption. This observation points to a trade-off between FID improvement and the associated energy expenses as time steps increase.



Figure 7:Unconditional image generation results on CIFAR-10 and FFHQ64 by using ANN-SNN method.
Table 6:Ablation study on different diffusion solvers. 
𝑆
 denotes the diffusion timesteps. The best results for each SDM solver are shown in bold.
	CIFAR10 (
32
×
32
)

𝑆
	10	20	50	100	200	500
DDIM	48.51	30.60	22.46	20.31	18.49	19.02
SDDIM (T=8)	39.73	21.69	16.02	16.27	18.43	23.93
SDPM-Solver (T=8)	30.97	30.85	31.80	32.33	32.42	32.06
Analytic-SDPM (T=8)	58.38	28.31	17.35	15.38	13.41	12.95
7.8Ablation Study

Impact of different components of SDMs. We first conduct ablation studies on CIFAR-10 to investigate the effects of Temporal-wise Spiking Module (TSM) and Threshold Guidance (TG). As presented in Tab. 5, we observe that both TSM and TG contribute to improving image quality. The optimal FID results are obtained by using TSM and TG simultaneously, achieving an 18.4% increase compared to Vanilla SDDIM.

Effectiveness of SDM on different solvers. In Tab. 6, we verify the feasibility and validity of our SDM across various diffusion solvers. SDDIM exhibits a more stable performance depending on sampling steps, while Analytic-SDPM boasts exceptional capabilities, achieving new state-of-the-art performance and surpassing ANN-DDIM results. In conclusion, our SDM proves its proficiency in handling any diffusion solvers, and we believe there remains significant potential for further enhancement of the FID performance utilizing our SDM.

8Discussion and Conclusion

In this work, we propose a new family of SNN-based diffusion models names Spiking Diffusion Models (SDMs) that combine the energy efficiency of SNNs with superior generative performance. SDMs achieve state-of-the-art results with fewer spike time steps among the SNN baselines and also attain competitive results with lower energy consumption compared to ANNs. SDMs primarily benefit from two aspects: (1) the Temporal-wise Spiking Mechanism (TSM), which enables the synaptic currents of the denoised network SNN-UNet to gather more dynamic information at each time step, as opposed to being governed by fixed synaptic weights as traditional SNNs, and (2) the training-free Threshold Guidance (TG), which can further enhance the sampling quality by adjusting the spike threshold. Nevertheless, one limitation of our work is that the time step of our SNN-UNet is relatively small, leaving the full potential of SDMs unexplored. Additionally, testing on higher-resolution datasets (e.g., ImageNet) should be considered. In future research, we plan to explore further applications of SDMs in the generation domain, e.g., text-image generation, and attempt to combine it with advanced language models to achieve more interesting tasks.

References
[1]
↑
	J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.
[2]
↑
	V. Sze, Y.-H. Chen, T.-J. Yang, and J. S. Emer, “Efficient processing of deep neural networks: A tutorial and survey,” Proceedings of the IEEE, vol. 105, no. 12, pp. 2295–2329, 2017.
[3]
↑
	M. E. Raichle and D. A. Gusnard, “Appraising the brain’s energy budget,” PNAS, vol. 99, no. 16, pp. 10 237–10 239, 2002.
[4]
↑
	D. D. Cox and T. Dean, “Neural networks and neuroscience-inspired computer vision,” Current Biology, vol. 24, no. 18, pp. R921–R929, 2014.
[5]
↑
	M. Davies, N. Srinivasa, T.-H. Lin, G. Chinya, Y. Cao, S. H. Choday, G. Dimou, P. Joshi, N. Imam, S. Jain et al., “Loihi: A neuromorphic manycore processor with on-chip learning,” IEEE Micro, vol. 38, no. 1, pp. 82–99, 2018.
[6]
↑
	F. Akopyan, J. Sawada, A. Cassidy, R. Alvarez-Icaza, J. Arthur, P. Merolla, N. Imam, Y. Nakamura, P. Datta, G.-J. Nam et al., “Truenorth: Design and tool flow of a 65 mw 1 million neuron programmable neurosynaptic chip,” IEEE TCAD, vol. 34, no. 10, pp. 1537–1557, 2015.
[7]
↑
	J. Pei, L. Deng, S. Song, M. Zhao, Y. Zhang, S. Wu, G. Wang, Z. Zou, Z. Wu, W. He et al., “Towards artificial general intelligence with hybrid tianjic chip architecture,” Nature, vol. 572, no. 7767, pp. 106–111, 2019.
[8]
↑
	Z. Zhou, Y. Zhu, C. He, Y. Wang, S. Yan, Y. Tian, and L. Yuan, “Spikformer: When spiking neural network meets transformer,” arXiv preprint arXiv:2209.15425, 2022.
[9]
↑
	S. Deng, Y. Li, S. Zhang, and S. Gu, “Temporal efficient training of spiking neural network via gradient re-weighting,” arXiv preprint arXiv:2202.11946, 2022.
[10]
↑
	J. Zhang, B. Dong, H. Zhang, J. Ding, F. Heide, B. Yin, and X. Yang, “Spiking transformers for event-based single object tracking,” in CVPR, 2022, pp. 8801–8810.
[11]
↑
	S. Kim, S. Park, B. Na, and S. Yoon, “Spiking-yolo: spiking neural network for energy-efficient object detection,” in AAAI, vol. 34, no. 07, 2020, pp. 11 270–11 277.
[12]
↑
	M. Yuan, C. Zhang, Z. Wang, H. Liu, G. Pan, and H. Tang, “Trainable spiking-yolo for low-latency and high-performance object detection,” Neural Networks, vol. 172, p. 106092, 2024.
[13]
↑
	P. Kirkland, G. Di Caterina, J. Soraghan, and G. Matich, “Spikeseg: Spiking segmentation via stdp saliency mapping,” in IJCNN.   IEEE, 2020, pp. 1–8.
[14]
↑
	S. Li, Y. Feng, Y. Li, Y. Jiang, C. Zou, and Y. Gao, “Event stream super-resolution via spatiotemporal constraint learning,” in ICCV, 2021, pp. 4480–4489.
[15]
↑
	J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep unsupervised learning using nonequilibrium thermodynamics,” in International Conference on Machine Learning.   PMLR, 2015, pp. 2256–2265.
[16]
↑
	J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” NeurIPS, vol. 33, pp. 6840–6851, 2020.
[17]
↑
	Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” arXiv preprint arXiv:2011.13456, 2020.
[18]
↑
	T. Karras, M. Aittala, T. Aila, and S. Laine, “Elucidating the design space of diffusion-based generative models,” arXiv preprint arXiv:2206.00364, 2022.
[19]
↑
	J. Cao, Z. Wang, H. Guo, H. Cheng, Q. Zhang, and R. Xu, “Spiking denoising diffusion probabilistic models,” in WACV, 2024, pp. 4912–4921.
[20]
↑
	J. Ho, C. Saharia, W. Chan, D. J. Fleet, M. Norouzi, and T. Salimans, “Cascaded diffusion models for high fidelity image generation.” J. Mach. Learn. Res., vol. 23, no. 47, pp. 1–33, 2022.
[21]
↑
	P. Dhariwal and A. Nichol, “Diffusion models beat gans on image synthesis,” NeurIPS, vol. 34, pp. 8780–8794, 2021.
[22]
↑
	X. Xu, Z. Wang, E. Zhang, K. Wang, and H. Shi, “Versatile diffusion: Text, images and variations all in one diffusion model,” arXiv preprint arXiv:2211.08332, 2022.
[23]
↑
	F. Bao, S. Nie, K. Xue, C. Li, S. Pu, Y. Wang, G. Yue, Y. Cao, H. Su, and J. Zhu, “One transformer fits all distributions in multi-modal diffusion at scale,” arXiv preprint arXiv:2303.06555, 2023.
[24]
↑
	Z. Kong, W. Ping, J. Huang, K. Zhao, and B. Catanzaro, “Diffwave: A versatile diffusion model for audio synthesis,” arXiv preprint arXiv:2009.09761, 2020.
[25]
↑
	V. Popov, I. Vovk, V. Gogoryan, T. Sadekova, and M. Kudinov, “Grad-tts: A diffusion probabilistic model for text-to-speech,” in International Conference on Machine Learning.   PMLR, 2021, pp. 8599–8608.
[26]
↑
	J. Ho, T. Salimans, A. Gritsenko, W. Chan, M. Norouzi, and D. J. Fleet, “Video diffusion models,” arXiv preprint arXiv:2204.03458, 2022.
[27]
↑
	B. Poole, A. Jain, J. T. Barron, and B. Mildenhall, “Dreamfusion: Text-to-3d using 2d diffusion,” arXiv preprint arXiv:2209.14988, 2022.
[28]
↑
	Z. Wang, C. Lu, Y. Wang, F. Bao, C. Li, H. Su, and J. Zhu, “Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation,” Advances in Neural Information Processing Systems, vol. 36, 2024.
[29]
↑
	M. Liu, R. Wen, and H. Chen, “Spiking-diffusion: Vector quantized discrete diffusion model with spiking neural networks,” arXiv preprint arXiv:2308.10187, 2023.
[30]
↑
	H. Kamata, Y. Mukuta, and T. Harada, “Fully spiking variational autoencoder,” in AAAI, vol. 36, no. 6, 2022, pp. 7059–7067.
[31]
↑
	A. Q. Nichol and P. Dhariwal, “Improved denoising diffusion probabilistic models,” in ICML.   PMLR, 2021, pp. 8162–8171.
[32]
↑
	J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020.
[33]
↑
	F. Bao, C. Li, J. Zhu, and B. Zhang, “Analytic-dpm: an analytic estimate of the optimal reverse variance in diffusion probabilistic models,” arXiv preprint arXiv:2201.06503, 2022.
[34]
↑
	C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, and J. Zhu, “Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps,” arXiv preprint arXiv:2206.00927, 2022.
[35]
↑
	E. O. Neftci, H. Mostafa, and F. Zenke, “Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks,” IEEE Signal Processing Magazine, vol. 36, no. 6, pp. 51–63, 2019.
[36]
↑
	C. Lee, S. S. Sarwar, P. Panda, G. Srinivasan, and K. Roy, “Enabling spike-based backpropagation for training deep neural network architectures,” Frontiers in Neuroscience, p. 119, 2020.
[37]
↑
	M. Xiao, Q. Meng, Z. Zhang, Y. Wang, and Z. Lin, “Training feedback spiking neural networks by implicit differentiation on the equilibrium state,” NeurIPS, vol. 34, pp. 14 516–14 528, 2021.
[38]
↑
	T. Bu, W. Fang, J. Ding, P. Dai, Z. Yu, and T. Huang, “Optimal ann-snn conversion for high-accuracy and ultra-low-latency spiking neural networks,” arXiv preprint arXiv:2303.04347, 2023.
[39]
↑
	S. Deng and S. Gu, “Optimal conversion of conventional artificial neural networks to spiking neural networks,” arXiv preprint arXiv:2103.00476, 2021.
[40]
↑
	J. Ding, Z. Yu, Y. Tian, and T. Huang, “Optimal ann-snn conversion for fast and accurate inference in deep spiking neural networks,” arXiv preprint arXiv:2105.11654, 2021.
[41]
↑
	Y. Li, S. Deng, X. Dong, R. Gong, and S. Gu, “A free lunch from ann: Towards efficient, accurate spiking neural networks calibration,” in ICML.   PMLR, 2021, pp. 6316–6325.
[42]
↑
	N. Skatchkovsky, O. Simeone, and H. Jang, “Learning to time-decode in spiking neural networks through the information bottleneck,” arXiv preprint arXiv:2106.01177, 2021.
[43]
↑
	K. Stewart, A. Danielescu, T. Shea, and E. Neftci, “Encoding event-based data with a hybrid snn guided variational auto-encoder in neuromorphic hardware,” in NICE, 2022, pp. 88–97.
[44]
↑
	B. Rosenfeld, O. Simeone, and B. Rajendran, “Spiking generative adversarial networks with a neural network discriminator: Local training, bayesian models, and continual meta-learning,” IEEE Transactions on Computers, vol. 71, no. 11, pp. 2778–2791, 2022.
[45]
↑
	V. Kotariya and U. Ganguly, “Spiking-gan: A spiking generative adversarial network using time-to-first-spike coding,” in IJCNN.   IEEE, 2022, pp. 1–7.
[46]
↑
	L. Feng, D. Zhao, and Y. Zeng, “Sgad: Spiking generative adversarial network with attention scoring decoding,” arXiv preprint arXiv:2305.10246, 2023.
[47]
↑
	R. Watanabe, Y. Mukuta, and T. Harada, “Fully spiking denoising diffusion implicit models,” arXiv preprint arXiv:2312.01742, 2023.
[48]
↑
	E. Hunsberger and C. Eliasmith, “Spiking deep networks with lif neurons,” arXiv preprint arXiv:1510.08829, 2015.
[49]
↑
	A. N. Burkitt, “A review of the integrate-and-fire neuron model: I. homogeneous synaptic input,” Biological Cybernetics, vol. 95, pp. 1–19, 2006.
[50]
↑
	W. Fang, Z. Yu, Y. Chen, T. Masquelier, T. Huang, and Y. Tian, “Incorporating learnable membrane time constant to enhance learning of spiking neural networks,” in ICCV, 2021, pp. 2661–2671.
[51]
↑
	P. Vincent, “A connection between score matching and denoising autoencoders,” Neural Computation, vol. 23, no. 7, pp. 1661–1674, 2011.
[52]
↑
	A. Shonenkov, M. Konstantinov, D. Bakshandaeva, C. Schuhmann, K. Ivanova, and N. Klokova, 2023, https://github.com/deep-floyd/IF.
[53]
↑
	J. Ho and T. Salimans, “Classifier-free diffusion guidance,” arXiv preprint arXiv:2207.12598, 2022.
[54]
↑
	M. Zhao, F. Bao, C. Li, and J. Zhu, “Egsde: Unpaired image-to-image translation via energy-guided stochastic differential equations,” arXiv preprint arXiv:2207.06635, 2022.
[55]
↑
	F. Bao, M. Zhao, Z. Hao, P. Li, C. Li, and J. Zhu, “Equivariant energy-guided sde for inverse molecular design,” arXiv preprint arXiv:2209.15408, 2022.
[56]
↑
	D. Kim, Y. Kim, W. Kang, and I.-C. Moon, “Refining generative process with discriminator guidance in score-based diffusion models,” arXiv preprint arXiv:2211.17091, 2022.
[57]
↑
	W. Fang, Z. Yu, Y. Chen, T. Huang, T. Masquelier, and Y. Tian, “Deep residual learning in spiking neural networks,” NeurIPS, vol. 34, pp. 21 056–21 069, 2021.
[58]
↑
	H. Zheng, Y. Wu, L. Deng, Y. Hu, and G. Li, “Going deeper with directly-trained larger spiking neural networks,” in AAAI, vol. 35, no. 12, 2021, pp. 11 062–11 070.
[59]
↑
	Y. Hu, H. Tang, and G. Pan, “Spiking deep residual networks,” IEEE TNNLS, 2021.
[60]
↑
	K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in CVPR, 2016, pp. 770–778.
[61]
↑
	Z. Liu, B. Wu, W. Luo, X. Yang, W. Liu, and K.-T. Cheng, “Bi-real net: Enhancing the performance of 1-bit cnns with improved representational capability and advanced training algorithm,” in ECCV, 2018, pp. 722–737.
[62]
↑
	Y. Zhang, Z. Zhang, and L. Lew, “Pokebnn: A binary pursuit of lightweight accuracy,” in CVPR, 2022, pp. 12 475–12 485.
[63]
↑
	A. Hasenstaub, Y. Shu, B. Haider, U. Kraushaar, A. Duque, and D. A. McCormick, “Inhibitory postsynaptic potentials carry synchronized frequency information in active cortical networks,” Neuron, vol. 47, no. 3, pp. 423–435, 2005.
[64]
↑
	M. Steriade, I. Timofeev, and F. Grenier, “Natural waking and sleep states: a view from inside neocortical neurons,” Journal of neurophysiology, vol. 85, no. 5, pp. 1969–1985, 2001.
[65]
↑
	P. Panda, S. A. Aketi, and K. Roy, “Toward scalable, efficient, and accurate deep spiking neural networks with backward residual connections, stochastic softmax, and hybridization,” Frontiers in Neuroscience, vol. 14, p. 653, 2020.
[66]
↑
	M. Yao, G. Zhao, H. Zhang, Y. Hu, L. Deng, Y. Tian, B. Xu, and G. Li, “Attention spiking neural networks,” IEEE TPAMI, 2023.
[67]
↑
	Y. Hu, Q. Zheng, X. Jiang, and G. Pan, “Fast-snn: Fast spiking neural network by converting quantized ann,” arXiv preprint arXiv:2305.19868, 2023.
[68]
↑
	Y. LeCun, C. Cortes, and C. Burges, “Mnist handwritten digit database. att labs,” 2010.
[69]
↑
	H. Xiao, K. Rasul, and R. Vollgraf, “Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms,” arXiv preprint arXiv:1708.07747, 2017.
[70]
↑
	A. Krizhevsky, G. Hinton et al., “Learning multiple layers of features from tiny images,” 2009.
[71]
↑
	Z. Liu, P. Luo, X. Wang, and X. Tang, “Deep learning face attributes in the wild,” in ICCV, 2015, pp. 3730–3738.
[72]
↑
	M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” NeurIPS, vol. 30, 2017.
[73]
↑
	T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen, “Improved techniques for training gans,” NeurIPS, vol. 29, 2016.
[74]
↑
	O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in MICCAI.   Springer, 2015.
[75]
↑
	A. Tarvainen and H. Valpola, “Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results,” NeurIPS, vol. 30, 2017.
[76]
↑
	D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” arXiv preprint arXiv:1312.6114, 2013.
[77]
↑
	T. Karras, S. Laine, and T. Aila, “A style-based generator architecture for generative adversarial networks,” in CVPR, 2019, pp. 4401–4410.
[78]
↑
	Y. Wu, L. Deng, G. Li, J. Zhu, and L. Shi, “Spatio-temporal backpropagation for training high-performance spiking neural networks,” Frontiers in Neuroscience, vol. 12, p. 331, 2018.
[79]
↑
	D. Kingma, T. Salimans, B. Poole, and J. Ho, “Variational diffusion models,” NeurIPS, vol. 34, pp. 21 696–21 707, 2021.
{IEEEbiography}

[] Jiahang Cao receives the B.S. degree in mathematics from the Sun Yat-sen University. He is currently a Ph.D. student in the Humanoid Computing Lab, Microelectronics Thrust, The Hong Kong University of Science and Technology (Guangzhou). His research interests include neuromorphic computing and robot learning.

{IEEEbiography}

[]Hanzhong Guo receives the B.S. degree in economics from the Sun Yat-sen University. He is currently a master student in the Renmin University of China. His research interests include generative models, computer vision and trustworthy AI.

{IEEEbiography}

[]Ziqing Wang receives the B.S. degree in microelectronics from the Sun Yat-sen University. He is currently a CS Ph.D. student at the North Carolina State University. His research interests include brain-inspired computing, efficient AI, and generative AI.

{IEEEbiography}

[] Deming Zhou is currently an undergraduate in the Department of Information and Computing Sciences, School of Mathematics and Information Science, Guangzhou University. His research interest is brain-inspired computing.

{IEEEbiography}

[] Hao Cheng is a Ph.D. student in the Humanoid Computing Lab, Microelectronics Thrust, The Hong Kong University of Science and Technology (Guangzhou). His research interests include deep learning robustness, model efficiency, brain-inspired computing, and machine learning.

{IEEEbiography}

[]Qiang Zhang is currently a Ph.D. student in the Humanoid Computing Lab, Microelectronics Thrust, The Hong Kong University of Science and Technology (Guangzhou). His research interest is humanoid robots.

{IEEEbiography}

[] Renjing Xu is an Assistant Professor at the Microelectronics Thrust, Function Hub of the Hong Kong University of Science and Technology (Guangzhou). He received a B.Eng (First-class Honors) in photonic system from the Australian National University in 2015 and a Ph.D. in applied physics from Harvard University in 2021 (advised by Prof. Donhee Ham). From 2015 to 2016, he was a visiting scholar in the Photonics Lab at University of Wisconsin-Madison. His research interests include human-centered computing and hardware-efficient computing.

Appendix
8.1Learning Rules of SNNs

In this section, we revisit the overall training algorithm of the deep SNNs. Our goal is to optimize the weights 
𝐖
 in our SNN-UNet as well as the temporal parameter 
𝐏
=
{
𝑝
⁢
[
1
]
,
𝑝
⁢
[
2
]
⁢
…
,
𝑝
⁢
[
𝑇
]
}
, where 
𝑇
 is the SNN time steps. We follow one of the standard direct-training algorithms Spatio-Temporal Backpropagation (STBP [78]) to calculate the gradient process.

Consider the noise input 
ℰ
=
{
𝜖
1
,
𝜖
2
,
.
.
𝜖
𝑁
}
 with batch 
𝑁
 and the predicted noise output 
ℰ
^
=
{
𝜖
1
^
,
𝜖
2
^
,
.
.
𝜖
𝑁
^
}
. The loss function 
ℒ
𝑚
⁢
𝑠
⁢
𝑒
 is defined as:

	
ℒ
𝑚
⁢
𝑠
⁢
𝑒
=
1
𝑁
⁢
∑
𝑖
=
1
𝑁
(
𝜖
𝑖
^
−
𝜖
𝑖
)
2
.
		
(25)

Similar to the previous works [58], we compute the gradient by unfolding the network on both spatial and temporal domains. By applying the chain rule, we can get:

	
∂
ℒ
∂
𝑊
𝑖
⁢
𝑗
𝑙
	
=
∂
ℒ
∂
𝑢
𝑖
𝑙
⁢
[
𝑡
]
⁢
∂
𝑢
𝑖
𝑙
⁢
[
𝑡
]
∂
𝑊
𝑖
⁢
𝑗
𝑙
=
∂
ℒ
∂
𝑢
𝑖
𝑙
⁢
[
𝑡
]
⁢
𝑜
𝑗
𝑙
−
1
⁢
[
𝑡
]
,
		
(26)

	
∂
ℒ
∂
𝑝
𝑙
⁢
[
𝑡
]
	
=
∑
𝑖
∂
ℒ
∂
𝑢
𝑖
𝑙
⁢
[
𝑡
]
⁢
∂
𝑢
𝑖
𝑙
⁢
[
𝑡
]
∂
𝑝
𝑙
⁢
[
𝑡
]
,
		
(27)

where 
𝑊
𝑖
⁢
𝑗
𝑙
 denotes the synaptic weight of SNNs between the 
𝑖
-th neuron and 
𝑗
-th neuron. 
𝑢
𝑖
𝑙
⁢
[
𝑡
]
 and 
𝑜
𝑖
𝑙
⁢
[
𝑡
]
 denotes the membrane potential and spikes of the 
𝑖
-th neuron at layer 
𝑙
 of time 
𝑡
, respectively. By applying the chain rule, 
∂
ℒ
∂
𝑢
𝑖
𝑙
⁢
[
𝑡
]
 and 
∂
ℒ
∂
𝑜
𝑖
𝑙
⁢
[
𝑡
]
 can be computed by:

	
∂
ℒ
∂
𝑢
𝑖
𝑙
⁢
[
𝑡
]
	
=
∂
ℒ
∂
𝑜
𝑖
𝑙
⁢
[
𝑡
]
⁢
∂
𝑜
𝑖
𝑙
⁢
[
𝑡
]
∂
𝑢
𝑖
𝑙
⁢
[
𝑡
]
+
∂
ℒ
∂
𝑢
𝑖
𝑙
⁢
[
𝑡
+
1
]
⁢
∂
𝑢
𝑖
𝑙
⁢
[
𝑡
+
1
]
∂
𝑢
𝑖
𝑙
⁢
[
𝑡
]
,
		
(28)

	
∂
ℒ
∂
𝑜
𝑖
𝑙
⁢
[
𝑡
]
	
=
∑
𝑗
∂
ℒ
∂
𝑢
𝑗
𝑙
+
1
⁢
[
𝑡
]
⁢
∂
𝑢
𝑗
𝑙
+
1
⁢
[
𝑡
]
∂
𝑜
𝑖
𝑙
⁢
[
𝑡
]
+
∂
ℒ
∂
𝑢
𝑖
𝑙
⁢
[
𝑡
+
1
]
⁢
∂
𝑢
𝑖
𝑙
⁢
[
𝑡
+
1
]
∂
𝑜
𝑖
𝑙
⁢
[
𝑡
]
.
		
(29)

However, due to the non-differentiable spiking activities, 
∂
𝑜
⁢
[
𝑡
]
∂
𝑢
⁢
[
𝑡
]
 can not be computed directly. To address this problem, the surrogate gradient method is used to smooth out the step function by the following equation:

	
∂
𝑜
⁢
[
𝑡
]
∂
𝑢
⁢
[
𝑡
]
	
=
1
𝑎
⁢
𝑠
⁢
𝑖
⁢
𝑔
⁢
𝑛
⁢
(
|
𝑢
⁢
[
𝑡
]
−
𝜗
th
|
<
𝑎
2
)
.
		
(30)
8.2Different Solvers for Diffusion Models

The detail of integrating the proposed model with various inference solvers is that: The sampling process can be divided into multiple discretization steps and our proposed spiking UNet is trained to predict noise in each discretization step. Different solvers essentially optimize the discretization method, where each step of discretization requires inference based on the pre-trained noise prediction model. Therefore, for different diffusion solvers, the pre-trained noise prediction network is fixed. We introduce more explanations as follows.

Diffusion models incrementally perturb data through a forward diffusion process and subsequently learn to reverse this process to restore the original data distribution. Formally, let 
𝑥
0
∈
ℝ
𝑛
 represent a random variable with an unknown data distribution 
𝑞
⁢
(
𝑥
0
)
. The forward process, denoted as 
𝑥
𝑡
, where 
𝑡
∈
[
0
,
1
]
 and indexed by time 
𝑡
, perturbs the data by adding Gaussian noise to 
𝑥
0

	
𝑞
⁢
(
𝑥
𝑡
|
𝑥
0
)
=
𝒩
⁢
(
𝑥
𝑡
|
𝑎
⁢
(
𝑡
)
⁢
𝑥
0
,
𝜎
2
⁢
(
𝑡
)
⁢
𝐼
)
.
		
(31)

In general, the function 
𝑎
⁢
(
𝑡
)
 and 
𝜎
⁢
(
𝑡
)
 are selected so that the logarithmic signal-to-noise ratio 
log
⁡
𝑎
2
⁢
(
𝑡
)
𝜎
2
⁢
(
𝑡
)
 decreases monotonically with time 
𝑡
, causing the data to diffuse towards random Gaussian noise [79]. Furthermore, it has been demonstrated by [79] that the following SDE shares an identical transition distribution 
𝑞
𝑡
|
0
⁢
(
𝑥
𝑡
|
𝑥
0
)
 with Eq. (31):

	
𝑑
⁢
𝑥
𝑡
=
𝑓
⁢
(
𝑡
)
⁢
𝑥
𝑡
⁢
𝑑
⁢
𝑡
+
𝑔
⁢
(
𝑡
)
⁢
𝑑
⁢
𝜔
,
𝑥
0
∼
𝑞
⁢
(
𝑥
0
)
,
		
(32)

where 
𝜔
∈
ℝ
𝑛
 is a standard Wiener process and

	
𝑓
⁢
(
𝑡
)
=
𝑑
⁢
log
⁡
𝑎
⁢
(
𝑡
)
𝑑
⁢
𝑡
,
𝑔
2
⁢
(
𝑡
)
=
𝑑
⁢
𝜎
2
⁢
(
𝑡
)
𝑑
⁢
𝑡
−
2
⁢
𝜎
2
⁢
(
𝑡
)
⁢
𝑑
⁢
log
⁡
𝑎
⁢
(
𝑡
)
𝑑
⁢
𝑡
.
		
(33)

Let 
𝑞
⁢
(
𝑥
𝑡
)
 be the marginal distribution of the above SDE at time 
𝑡
. Its reversal process can be described by a corresponding continuous SDE which recovers the data distribution:

	
𝑑
⁢
𝑥
=
[
𝑓
⁢
(
𝑡
)
⁢
𝑥
𝑡
−
𝑔
2
⁢
(
𝑡
)
⁢
∇
𝑥
𝑡
log
⁡
𝑞
⁢
(
𝑥
𝑡
)
]
⁢
𝑑
⁢
𝑡
+
𝑔
⁢
(
𝑡
)
⁢
𝑑
⁢
𝜔
¯
,
		
(34)

where 
𝑥
1
∼
𝑞
⁢
(
𝑥
1
)
, 
𝜔
¯
∈
ℝ
𝑛
 is a reverse-time standard Wiener process. Therefore, to sample the data, it is need to replace the 
log
⁡
𝑞
⁢
(
𝑥
𝑡
)
 with a learned score network 
𝑠
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
 or noise network (
−
1
𝜎
2
⁢
(
𝑡
)
⁢
𝜖
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
) and discretize the reverse SDE or corresponding ODE in Eq. (35) to obtain the generated samples 
𝑥
^
0
.

	
𝑑
⁢
𝑥
𝑡
𝑑
⁢
𝑡
=
𝑓
⁢
(
𝑡
)
⁢
𝑥
𝑡
−
1
2
⁢
𝑔
2
⁢
(
𝑡
)
⁢
∇
𝑥
log
⁡
𝑞
𝑡
⁢
(
𝑥
𝑡
)
,
𝑥
1
∼
𝑞
⁢
(
𝑥
1
)
.
		
(35)

For the SDE-based solvers, their primary objective lie in decreasing discretization error and therefore minimizing function evaluations required for convergence during the process of discretizing Eq. (5). Discretizing the reverse SDE in Eq. (5) is equivalent to sample from a Markov chain 
𝑝
⁢
(
𝑥
0
:
1
)
=
𝑝
⁢
(
𝑥
1
)
⁢
∏
𝑡
𝑖
−
1
,
𝑡
𝑖
∈
𝑆
𝑡
𝑝
⁢
(
𝑥
𝑡
𝑖
−
1
|
𝑥
𝑡
𝑖
)
 with its trajectory 
𝑆
𝑡
=
[
0
,
𝑡
1
,
𝑡
2
,
…
,
𝑡
𝑖
,
.
.
,
1
]
. Song et al. [17] proves that the conventional ancestral sampling technique used in the DPMs [16] that models 
𝑝
⁢
(
𝑥
𝑡
𝑖
−
1
|
𝑥
𝑡
𝑖
)
 as a Gaussian distribution, can be perceived as a first-order solver for the reverse SDE in Eq. 
(
⁢
5
⁢
)
. Bao et al. [33] finds that the optimal variance of 
𝑝
⁢
(
𝑥
𝑡
𝑖
−
1
|
𝑥
𝑡
𝑖
)
∼
𝒩
⁢
(
𝑥
𝑡
𝑖
−
1
|
𝜇
𝑡
𝑖
−
1
|
𝑡
𝑖
,
Σ
𝑡
𝑖
−
1
|
𝑡
𝑖
⁢
(
𝑥
𝑡
𝑖
)
)
 is

		
Σ
𝑡
𝑖
−
1
|
𝑡
𝑖
∗
⁢
(
𝑥
𝑡
𝑖
)
=
𝜆
𝑡
𝑖
2
+
	
		
𝛾
𝑡
𝑖
2
⁢
𝜎
2
⁢
(
𝑡
𝑖
)
𝛼
⁢
(
𝑡
𝑖
)
⁢
(
1
−
𝔼
𝑞
⁢
(
𝑥
𝑡
𝑖
)
⁢
[
1
𝑑
⁢
‖
𝔼
𝑞
⁢
(
𝑥
0
|
𝑥
𝑡
𝑖
)
⁢
[
𝜖
𝜃
⁢
(
𝑥
𝑡
𝑖
,
𝑡
𝑖
)
]
‖
2
2
]
)
,
		
(36)

where, 
𝛾
𝑡
𝑖
=
𝛼
⁢
(
𝑡
𝑖
−
1
)
−
𝜎
2
⁢
(
𝑡
𝑖
−
1
)
−
𝜆
𝑡
𝑖
2
⁢
𝛼
⁢
(
𝑡
𝑖
)
𝜎
2
⁢
(
𝑡
𝑖
)
, 
𝜆
𝑡
𝑖
2
 is the variance of 
𝑞
⁢
(
𝑥
𝑡
𝑖
−
1
|
𝑥
𝑡
𝑖
,
𝑥
0
)
. AnalyticDPM [33] offers a significant reduction in discretization error during sampling and achieves faster convergence with fewer steps. Besides the learned noise network 
𝜖
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
, to use the the AnalyticDPM to sampling, it is required to obtain the statistic value 
ℎ
⁢
(
𝑡
𝑖
)
=
𝔼
𝑞
⁢
(
𝑥
𝑡
𝑖
)
⁢
[
1
𝑑
⁢
‖
𝔼
𝑞
⁢
(
𝑥
0
|
𝑥
𝑡
𝑖
)
⁢
[
𝜖
𝜃
⁢
(
𝑥
𝑡
𝑖
,
𝑡
𝑖
)
]
‖
2
2
]
. Therefore, we can pre-calculate the 
ℎ
⁢
(
𝑡
𝑖
)
 for each dataset such as CIFAR-10 via using the SOTA pre-trained diffusion models and use the 
ℎ
⁢
(
𝑡
𝑖
)
 to calculate the optimal variance of reverse transition kernel in sampling, which won’t affect the speed of sampling while significant improving the generation performance of spiking diffusion models.

In another setting, as for the ODE-based solvers, their primary objective lie in decreasing discretization error by using the higher-order solvers for the reverse ODE. Some researches figure out that the first of the two terms of the reverse ODE can be computed explicitly, while only the term with the noise network needs to be estimated, hence the following equation,

	
𝑥
𝑡
=
𝑒
∫
𝑠
𝑡
𝑓
⁢
(
𝜏
)
⁢
𝑑
𝜏
⁢
𝑥
𝑠
+
∫
𝑠
𝑡
(
𝑒
∫
𝜏
𝑡
𝑓
⁢
(
𝑟
)
⁢
𝑑
𝑟
⁢
𝑔
2
⁢
(
𝜏
)
2
⁢
𝜎
𝜏
⁢
𝜖
𝜃
⁢
(
𝑥
𝜏
,
𝜏
)
)
⁢
𝑑
𝜏
.
		
(37)

In order to decrease the discretization error, it can use higher-order estimates to approximate the terms followed by the noise network. With the first-order estimation, we obtain the following equation which is the same with the DDIM [32],

	
𝑥
~
𝑡
𝑖
=
𝛼
𝑡
𝑖
𝛼
𝑡
𝑖
−
1
⁢
𝑥
~
𝑡
𝑖
−
1
−
𝜎
𝑡
𝑖
⁢
(
𝑒
ℎ
𝑖
−
1
)
⁢
𝜖
𝜃
⁢
(
𝑥
~
𝑡
𝑖
−
1
,
𝑡
𝑖
−
1
)
,
		
(38)

where 
ℎ
𝑖
=
𝜆
𝑡
𝑖
−
𝜆
𝑡
𝑖
−
1
. While approximating the first k terms of the Taylor expansion needs additional intermediate points between t and s, more details in DPM-Solver [34].

However, since the estimation error in spiking diffusion models can’t be ignored, the performance of ODE-based solvers is worse than the performance of SDE-based solvers, which is shown in the experiments section. Meanwhile, the parameters in different solvers are set as default setting.

8.3Hyperparameter Settings

As illustrated in Tab. 7, we have presented all the hyperparameters of the Spiking Diffusion Model (SDM) to ensure the reproducibility of our results. We are committed to transparency in our research and will open source our code in the future.

Table 7:Hyperparameters of the spiking diffusion models.
Hyperparameter	Value
Spiking UNet hyperparameter	
Diffusion timesteps	1000
Base channel dimension	128
Time embedding dimension	512
Channel multipliers	[1, 2, 2, 4]
Dropout rate	0.1
Spiking timesteps	4 or 8
Diffusion sampler hyperparameter	
Beta_1	1e-4
Beta_T	0.02
Sampler type	DDPM, DDIM, DPM-Solver, Analytic-DPM
Training hyperparameter	
Batch size	128*GPUs
Training steps	500,000
Gradient clip	1.0
8.4Decided Threshold Voltage

In this section, we provide more details of deciding the threshold voltage in the spiking diffusion models. As shown in the paper, the change of score incurred by the change of the threshold voltage can be divided into two terms, one for the original estimated score, another is the rectified term 
𝑐
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
.

	
𝑠
𝜃
⁢
(
𝑥
𝑡
,
𝑡
,
𝜗
th
′
)
	
	
≈
𝑠
𝜃
⁢
(
𝑥
𝑡
,
𝑡
,
𝜗
th
0
)
+
d
⁢
𝑠
𝜃
⁢
(
𝑥
𝑡
,
𝑡
,
𝜗
th
)
d
⁢
𝜗
th
⁢
d
⁢
𝜗
th
+
𝑂
⁢
(
d
⁢
𝜗
th
)
	
	
≈
𝑠
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
+
𝑠
𝜃
′
|
𝜗
th
0
⁢
d
⁢
𝜗
th
+
𝑂
⁢
(
d
⁢
𝜗
th
)
	
	
≈
𝑠
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
+
𝑐
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
,
		
(39)

Therefore, due to the estimation error of the noise network, a discrepancy exists between the estimated score 
𝑠
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
 and the true score 
𝑠
⁢
(
𝑥
𝑡
,
𝑡
)
. Since the true score is unknown, it is impossible to calculate the actual discrepancy and thus 
𝑐
𝜃
⁢
(
𝑥
𝑡
,
𝑡
)
. Furthermore, even if the gap between the true score and the estimated score is calculated, solving the corresponding differential equation is challenging, and it is uncertain whether a solution exists However, our findings are consistent with those of Ho et al. [53], who demonstrated that generation quality can be improved by introducing classifier-free guidance. This improvement is essentially due to the gap between the estimated score and the true score, where an artificially introduced correction term enhances the results. In the future, we will explore methods to find the optimal threshold voltage.

8.5Effectiveness of the Temporal-wise Spiking Mechanism

To better visualize the performance improvement brought by the TSM module, we have provided the generation results of CIFAR-10 using SDDIM with and without the TSM module. Here we used DDIM instead of DDPM for this comparison since DDIM operates based on Ordinary Differential Equations (ODEs), which ensure deterministic and consistent generation results. In contrast, DDPM relies on Stochastic Differential Equations (SDEs), which introduce randomness in the generation process, leading to variability in the output images and making direct comparisons challenging.

The results in Fig. 8 demonstrate a significant improvement in the quality of the generated images with the TSM module. The contours of the images are more pronounced, the backgrounds are clearer, and the texture details are richer compared to those without the TSM module, thereby proving the effectiveness of TSM.

Figure 8:Comparisons of the generation results with/without using the TSM method in CIFAR-10.
Report Issue
Report Issue for Selection
Generated by L A T E xml 
Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

Click the "Report Issue" button.
Open a report feedback form via keyboard, use "Ctrl + ?".
Make a text selection and click the "Report Issue for Selection" button near your cursor.
You can use Alt+Y to toggle on and Alt+Shift+Y to toggle off accessible reporting links at each section.

Our team has already identified the following issues. We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a list of packages that need conversion, and welcome developer contributions.

