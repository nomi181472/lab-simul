Title: Fully Differentiable Spiking Neural Networks via Ultradiscretization and Max-Plus Algebra

URL Source: https://arxiv.org/html/2602.11206

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
3Preliminaries
4Method: Ultradiscretized Spiking Neurons
5Theoretical Analysis
6Experiments
7Discussion
8Conclusion
 References
License: CC BY 4.0
arXiv:2602.11206v1 [cs.LG] 10 Feb 2026
UltraLIF: Fully Differentiable Spiking Neural Networks via Ultradiscretization and Max-Plus Algebra
Jose Marie Antonio Miñoza
Abstract

Spiking Neural Networks (SNNs) offer energy-efficient, biologically plausible computation but suffer from non-differentiable spike generation, necessitating reliance on heuristic surrogate gradients. This paper introduces UltraLIF, a principled framework that replaces surrogate gradients with ultradiscretization, a mathematical formalism from tropical geometry providing continuous relaxations of discrete dynamics. The central insight is that the max-plus semiring underlying ultradiscretization naturally models neural threshold dynamics: the log-sum-exp function serves as a differentiable soft-maximum that converges to hard thresholding as a learnable temperature parameter 
𝜀
→
0
. Two neuron models are derived from distinct dynamical systems: UltraLIF from the LIF ordinary differential equation (temporal dynamics) and UltraDLIF from the diffusion equation modeling gap junction coupling across neuronal populations (spatial dynamics). Both yield fully differentiable SNNs trainable via standard backpropagation with no forward-backward mismatch. Theoretical analysis establishes pointwise convergence to classical LIF dynamics with quantitative error bounds and bounded non-vanishing gradients. Experiments on six benchmarks spanning static images, neuromorphic vision, and audio demonstrate improvements over surrogate gradient baselines, with gains most pronounced in single-timestep (
𝑇
=
1
) settings on neuromorphic and temporal datasets. An optional sparsity penalty enables significant energy reduction while maintaining competitive accuracy.

Spiking Neural Networks, Ultradiscretization, Max-Plus Algebra, Neuromorphic Computing
1Introduction

Spiking Neural Networks (SNNs) represent a promising paradigm for energy-efficient machine learning, with significant potential for neuromorphic hardware deployment (Maass, 1997; Roy et al., 2019). In contrast to artificial neural networks (ANNs) communicating via continuous activations, SNNs process information through discrete spike events, emulating biological neural computation. This event-driven nature enables substantial energy savings; Intel’s Loihi chip demonstrates up to 
1000
×
 energy reduction compared to GPUs on certain tasks (Davies et al., 2018).

However, SNN training remains challenging due to the non-differentiability of spike generation. The standard Leaky Integrate-and-Fire (LIF) neuron follows the dynamics:

	
𝑣
(
𝑡
+
1
)
	
=
𝜏
​
𝑣
(
𝑡
)
+
𝐼
(
𝑡
)
		
(1)

	
𝑠
(
𝑡
+
1
)
	
=
𝐻
​
(
𝑣
(
𝑡
+
1
)
−
𝜃
)
		
(2)

where the Heaviside step function 
𝐻
​
(
⋅
)
 has gradient zero almost everywhere. The dominant approach employs surrogate gradients, replacing the true gradient with a smooth approximation during backpropagation (Neftci et al., 2019; Zenke and Vogels, 2021). While empirically effective, surrogate gradients introduce a fundamental mismatch between forward (discrete) and backward (continuous) passes (Figure 1a), with limited theoretical understanding of convergence properties (Li et al., 2021; Gygax and Zenke, 2025).

Figure 1:(a) Spike activation functions: Heaviside (hard threshold), surrogate gradient (smooth approximation), and ultradiscretized (principled soft relaxation). (b) Gradients: Heaviside has zero gradient almost everywhere (delta function at threshold); surrogate and ultradiscretized provide smooth gradients, but only ultradiscretized maintains forward-backward consistency.

This paper proposes UltraLIF, a theoretically grounded alternative based on ultradiscretization, a limiting procedure from tropical geometry transforming continuous dynamical systems into discrete max-plus systems while preserving structural properties (Tokihiro et al., 1996; Grammaticos et al., 2004). The key contributions are:

1. 

Principled differentiability: The 
LSE
 function provides a natural soft relaxation of the max operation underlying spike generation, with explicit convergence bounds as 
𝜀
→
0
 (Lemma 3.2).

2. 

Forward-backward consistency: Unlike surrogate methods, UltraLIF employs identical dynamics in forward and backward passes, eliminating gradient mismatch (Remark 5.6).

3. 

Bounded gradients: For any 
𝜀
>
0
, gradients remain bounded and non-vanishing, enabling stable optimization (Proposition 5.4).

4. 

Consistent low-timestep improvements: On six benchmarks spanning static, neuromorphic, and audio modalities, ultradiscretized models improve over surrogate gradient baselines at 
𝑇
=
1
, with the largest gains on temporal and event-driven data (+11.22% SHD, +7.96% DVS-Gesture, +3.91% N-MNIST).

2Related Work

Surrogate Gradient Methods. The dominant paradigm for direct SNN training replaces non-differentiable spike gradients with smooth surrogates (Neftci et al., 2019). Common choices include piecewise linear (Bellec et al., 2018), sigmoid (Zenke and Ganguli, 2018), and arctangent (Fang et al., 2021) functions. Recent work introduces learnable surrogate parameters (Lian et al., 2023) and adaptive shapes (Li et al., 2021). Despite empirical success, the forward-backward mismatch remains theoretically problematic. Gygax and Zenke (2025) provide partial justification via stochastic neurons, showing surrogate gradients match escape noise derivatives in expectation.

Spike Timing Approaches. SpikeProp (Bohte et al., 2002) and variants (Mostafa, 2018; Kheradpisheh and Masquelier, 2020) compute exact gradients with respect to spike times. These methods require careful initialization and struggle with silent neurons. Recent work on exact smooth gradients through spike timing (Göltz et al., 2021) addresses some limitations but remains computationally intensive.

ANN-to-SNN Conversion. An alternative approach trains conventional ANNs then converts to SNNs (Cao et al., 2015; Rueckauer et al., 2017; Bu et al., 2022). While avoiding direct SNN training, conversion methods typically require many timesteps to achieve ANN accuracy and sacrifice temporal dynamics.

Tropical Geometry and Neural Networks. Tropical geometry studies algebraic structures where addition becomes max (or min) and multiplication becomes addition (Maclagan and Sturmfels, 2015). Zhang et al. (2018) establish that ReLU networks compute tropical rational functions. Recent work extends this to graph neural networks (Pham and Garg, 2024) and neural network compression (Fotopoulos et al., 2024). The connection to spiking networks via ultradiscretization appears novel.

Ultradiscretization. Originating in integrable systems (Tokihiro et al., 1996), ultradiscretization transforms difference equations into cellular automata preserving solution structure. Applications include soliton systems (Takahashi and Satsuma, 1990) and limit cycle analysis (Yamazaki and Ohmori, 2023, 2024). Application to neural networks has not been previously explored.

3Preliminaries
3.1The Max-Plus Semiring
Definition 3.1 (Max-Plus Semiring).

The max-plus semiring is the algebraic structure 
(
ℝ
∪
{
−
∞
}
,
⊕
,
⊙
)
 where:

• 

𝑎
⊕
𝑏
:=
max
⁡
(
𝑎
,
𝑏
)
 (tropical addition)

• 

𝑎
⊙
𝑏
:=
𝑎
+
𝑏
 (tropical multiplication)

with additive identity 
−
∞
 and multiplicative identity 
0
.

This semiring underlies tropical geometry and provides the limit structure for ultradiscretization.

3.2Log-Sum-Exp as Soft Maximum

The log-sum-exp function with temperature 
𝜀
>
0
 is defined as:

	
LSE
𝜀
​
(
𝐱
)
:=
𝜀
​
log
⁡
(
∑
𝑖
=
1
𝑛
𝑒
𝑥
𝑖
/
𝜀
)
		
(3)

The following lemma establishes its role as a smooth approximation to the maximum.

Lemma 3.2 (LSE Convergence).

Let 
𝐱
=
(
𝑥
1
,
…
,
𝑥
𝑛
)
∈
ℝ
𝑛
 and 
𝑀
:=
max
𝑖
⁡
𝑥
𝑖
. Then 
𝑀
≤
LSE
𝜀
​
(
𝐱
)
≤
𝑀
+
𝜀
​
log
⁡
𝑛
, hence 
lim
𝜀
→
0
+
LSE
𝜀
​
(
𝐱
)
=
𝑀
. Moreover, 
∇
LSE
𝜀
​
(
𝐱
)
=
softmax
​
(
𝐱
/
𝜀
)
∈
(
0
,
1
)
𝑛
.

Proof.

For the lower bound, the sum includes 
𝑒
𝑀
/
𝜀
, so 
LSE
𝜀
​
(
𝐱
)
≥
𝜀
​
log
⁡
(
𝑒
𝑀
/
𝜀
)
=
𝑀
. For the upper bound, since 
𝑥
𝑖
≤
𝑀
 for all 
𝑖
, 
LSE
𝜀
​
(
𝐱
)
≤
𝜀
​
log
⁡
(
𝑛
⋅
𝑒
𝑀
/
𝜀
)
=
𝑀
+
𝜀
​
log
⁡
𝑛
. The limit follows by the squeeze theorem. The gradient formula follows from direct differentiation: 
∂
LSE
𝜀
∂
𝑥
𝑗
=
𝑒
𝑥
𝑗
/
𝜀
∑
𝑖
𝑒
𝑥
𝑖
/
𝜀
=
softmax
​
(
𝐱
/
𝜀
)
𝑗
. ∎

Remark 3.3.

When the maximum is unique with gap 
𝛿
:=
𝑀
−
max
𝑖
:
𝑥
𝑖
≠
𝑀
⁡
𝑥
𝑖
>
0
, the error decays exponentially: 
LSE
𝜀
​
(
𝐱
)
−
𝑀
=
𝑂
​
(
𝜀
​
𝑒
−
𝛿
/
𝜀
)
.

4Method: Ultradiscretized Spiking Neurons

The key innovation of this work is applying ultradiscretization, a limiting procedure from tropical geometry, to derive differentiable spiking neurons. This section shows that ultradiscretization can be applied to different neural dynamics, yielding distinct models for temporal and spatial computations.

4.1Ultradiscretization Framework

Ultradiscretization transforms continuous dynamical systems into max-plus (tropical) systems while preserving structural properties (Tokihiro et al., 1996). The procedure operates via the substitution 
𝑥
=
𝑒
𝑋
/
𝜀
 followed by the limit 
𝜀
→
0
+
:

	
𝑥
+
𝑦
=
𝑒
𝑋
/
𝜀
+
𝑒
𝑌
/
𝜀
	
→
𝑒
max
⁡
(
𝑋
,
𝑌
)
/
𝜀
(addition 
→
 max)
		
(4)

	
𝑥
⋅
𝑦
=
𝑒
𝑋
/
𝜀
⋅
𝑒
𝑌
/
𝜀
	
=
𝑒
(
𝑋
+
𝑌
)
/
𝜀
(multiplication 
→
 addition)
		
(5)

For finite 
𝜀
>
0
, the log-sum-exp function 
LSE
𝜀
 (Eq. 3) provides a differentiable soft relaxation of the max operation. This is the foundation for all ultradiscretized neurons presented here.

4.1.1Temporal Dynamics: UltraLIF

UltraLIF is derived from the standard single-neuron LIF ordinary differential equation. The membrane potential evolves according to:

	
𝜏
𝑚
​
𝑑
​
𝑣
𝑑
​
𝑡
=
−
(
𝑣
−
𝑣
rest
)
+
𝑅
⋅
𝐼
​
(
𝑡
)
		
(6)

where 
𝜏
𝑚
 is the membrane time constant, 
𝑣
rest
 the resting potential, 
𝑅
 the membrane resistance, and 
𝐼
​
(
𝑡
)
 the input current.

Applying forward Euler discretization with timestep 
Δ
​
𝑡
 and setting 
𝑣
rest
=
0
 yields:

	
𝑣
(
𝑡
+
1
)
=
(
1
−
Δ
​
𝑡
𝜏
𝑚
)
⏟
=
⁣
:
𝜏
0
​
𝑣
(
𝑡
)
+
𝐼
(
𝑡
)
		
(7)

where the leak factor 
𝜏
0
∈
(
0
,
1
)
 controls temporal decay.

The ultradiscretization transform (Tokihiro et al., 1996) proceeds by substituting 
𝑣
=
𝑒
𝑉
/
𝜀
, 
𝐼
=
𝑒
𝐽
/
𝜀
, and parameterizing the leak as 
𝜏
=
𝑒
𝑇
/
𝜀
 where 
𝑇
=
log
⁡
𝜏
0
<
0
:

	
𝑒
𝑉
(
𝑡
+
1
)
/
𝜀
=
𝑒
𝑇
/
𝜀
⋅
𝑒
𝑉
(
𝑡
)
/
𝜀
+
𝑒
𝐽
(
𝑡
)
/
𝜀
=
𝑒
(
𝑉
(
𝑡
)
+
𝑇
)
/
𝜀
+
𝑒
𝐽
(
𝑡
)
/
𝜀
		
(8)

Taking 
𝜀
⋅
log
 of both sides and the limit 
𝜀
→
0
+
 recovers the max-plus dynamics:

	
𝑉
(
𝑡
+
1
)
=
max
⁡
(
𝑉
(
𝑡
)
+
𝑇
,
𝐽
(
𝑡
)
)
=
max
⁡
(
𝑉
(
𝑡
)
+
log
⁡
𝜏
0
,
𝐽
(
𝑡
)
)
		
(9)

For differentiable training, the hard maximum is relaxed to the log-sum-exp for finite 
𝜀
>
0
:

	
𝑉
𝜀
(
𝑡
+
1
)
=
LSE
𝜀
​
(
𝑉
(
𝑡
)
+
log
⁡
𝜏
0
,
𝐼
(
𝑡
)
)
		
(10)

This 2-term LSE captures temporal membrane integration with learnable leak. Note that the 
𝜀
-parameterized leak 
𝜏
=
𝑒
𝑇
/
𝜀
 becomes stronger as 
𝜀
→
0
, yielding sharper temporal dynamics in the tropical limit.

4.1.2Spatial Dynamics: UltraDLIF

An analogous derivation applies ultradiscretization to spatial dynamics, capturing lateral interactions across a neuronal population. Consider a simplified diffusive coupling where membrane potentials spread locally:

	
∂
𝑣
∂
𝑡
=
𝐷
​
∇
2
𝑣
		
(11)

where 
𝐷
>
0
 is the diffusion coefficient. This models gap junction (electrical synapse) coupling, where ionic currents flow directly between neurons enabling voltage spread (Connors and Long, 2004; Spek et al., 2020). The diffusion equation provides a first-order approximation of such lateral interactions.

Discretizing the Laplacian via finite differences 
∇
2
𝑣
≈
(
𝑣
𝑖
−
1
−
2
​
𝑣
𝑖
+
𝑣
𝑖
+
1
)
/
Δ
​
𝑥
2
 and applying forward Euler in time yields:

	
𝑣
𝑖
(
𝑡
+
1
)
=
𝑣
𝑖
(
𝑡
)
+
𝐷
​
Δ
​
𝑡
Δ
​
𝑥
2
​
(
𝑣
𝑖
−
1
(
𝑡
)
−
2
​
𝑣
𝑖
(
𝑡
)
+
𝑣
𝑖
+
1
(
𝑡
)
)
		
(12)

At the balanced diffusion regime where 
𝐷
​
Δ
​
𝑡
/
Δ
​
𝑥
2
=
1
/
3
, this simplifies to uniform spatial averaging where each neuron and its neighbors contribute equally:

	
𝑣
𝑖
(
𝑡
+
1
)
=
1
3
​
𝑣
𝑖
−
1
(
𝑡
)
+
1
3
​
𝑣
𝑖
(
𝑡
)
+
1
3
​
𝑣
𝑖
+
1
(
𝑡
)
		
(13)

This choice of 
1
/
3
 lies within the von Neumann stability bound for explicit finite difference schemes applied to the 1D diffusion equation, which requires 
𝐷
​
Δ
​
𝑡
/
Δ
​
𝑥
2
≤
1
/
2
 for numerical stability. The value 
1
/
3
 ensures stability while providing symmetric treatment of a neuron and its immediate neighbors, a natural balance for lateral coupling.

Remark on subtraction. Standard ultradiscretization cannot handle subtraction directly, as there is no tropical analog of 
𝑥
−
𝑦
 in the max-plus semiring (Ochiai and Nacher, 2005). This limitation is circumvented by selecting 
𝛼
=
1
/
3
: expanding Eq. (12) gives coefficients 
𝛼
, 
(
1
−
2
​
𝛼
)
, 
𝛼
 for the three terms, and at 
𝛼
=
1
/
3
 all become 
1
/
3
>
0
, eliminating subtraction entirely (Eq. (13)). For more general diffusion regimes, inversible max-plus algebras extend the framework to handle subtraction via 
𝑥
−
𝑦
→
max
⁡
(
𝑋
,
𝑌
+
𝜂
)
, where 
𝜂
 is an inverse element (Ochiai and Nacher, 2005).

Applying the ultradiscretization transform with 
𝑣
=
𝑒
𝑉
/
𝜀
 and taking 
𝜀
→
0
+
:

	
𝑉
𝑖
(
𝑡
+
1
)
=
max
⁡
(
𝑉
𝑖
−
1
(
𝑡
)
,
𝑉
𝑖
(
𝑡
)
,
𝑉
𝑖
+
1
(
𝑡
)
)
		
(14)

This limit corresponds to morphological dilation, a max-pooling operation over the spatial neighborhood.

Relaxing the hard maximum to the log-sum-exp for finite 
𝜀
>
0
 gives:

	
𝑉
𝑖
,
𝜀
(
𝑡
+
1
)
=
LSE
𝜀
​
(
𝑉
𝑖
−
1
(
𝑡
)
,
𝑉
𝑖
(
𝑡
)
,
𝑉
𝑖
+
1
(
𝑡
)
)
		
(15)

This 3-term LSE captures lateral spatial smoothing across neurons. External input 
𝐼
𝑖
(
𝑡
)
 is added separately (Eq. 19), following the standard neural field convention where diffusion handles lateral coupling and an additive term represents external drive.

4.1.3Comparison of Derivations
	UltraLIF	UltraDLIF
Source	LIF ODE	Diffusion PDE
Equation	
𝑑
​
𝑣
/
𝑑
​
𝑡
=
−
𝑣
/
𝜏
𝑚
+
𝐼
	
∂
𝑣
/
∂
𝑡
=
𝐷
​
∇
2
𝑣

LSE terms	2 (temporal)	3 (spatial)
Soft form	
LSE
​
(
𝑉
+
log
⁡
𝜏
0
,
𝐼
)
	
LSE
​
(
𝑉
−
1
,
𝑉
0
,
𝑉
+
1
)

Models	Membrane decay	Lateral diffusion
Table 1:Ultradiscretization applied to temporal and spatial dynamics.

Both models share the same theoretical foundation (ultradiscretization, LSE soft relaxation) but capture different biological phenomena. UltraLIF models single-neuron temporal dynamics; UltraDLIF models population-level spatial interactions.

4.2Neuron Models

Building on the ultradiscretization framework, complete neuron models are defined incorporating spike generation and reset mechanisms. Both variants share the same spike and reset logic, differing only in membrane dynamics.

Definition 4.1 (UltraLIF Neuron (Temporal)).

For temperature 
𝜀
>
0
, leak factor 
𝜏
0
∈
(
0
,
1
)
, and threshold 
𝜃
>
0
:

	
𝑉
~
𝜀
(
𝑡
+
1
)
	
=
LSE
𝜀
​
(
𝑉
𝜀
(
𝑡
)
+
log
⁡
𝜏
0
,
𝐼
(
𝑡
)
)
		
(16)

	
𝑠
𝜀
(
𝑡
+
1
)
	
=
𝜎
​
(
𝑉
~
𝜀
(
𝑡
+
1
)
−
𝜃
𝜀
)
		
(17)

	
𝑉
𝜀
(
𝑡
+
1
)
	
=
𝑉
~
𝜀
(
𝑡
+
1
)
⋅
(
1
−
𝑠
𝜀
(
𝑡
+
1
)
)
+
𝑉
reset
⋅
𝑠
𝜀
(
𝑡
+
1
)
		
(18)

where 
𝜎
​
(
𝑧
)
=
(
1
+
𝑒
−
𝑧
)
−
1
 is the logistic sigmoid and 
𝑉
reset
=
0
.

Definition 4.2 (UltraDLIF Neuron (Spatial)).

For temperature 
𝜀
>
0
, threshold 
𝜃
>
0
, and neuron index 
𝑖
:

	
𝑉
~
𝑖
,
𝜀
(
𝑡
+
1
)
	
=
LSE
𝜀
​
(
𝑉
𝑖
−
1
,
𝜀
(
𝑡
)
,
𝑉
𝑖
,
𝜀
(
𝑡
)
,
𝑉
𝑖
+
1
,
𝜀
(
𝑡
)
)
+
𝐼
𝑖
(
𝑡
)
		
(19)

	
𝑠
𝑖
,
𝜀
(
𝑡
+
1
)
	
=
𝜎
​
(
𝑉
~
𝑖
,
𝜀
(
𝑡
+
1
)
−
𝜃
𝜀
)
		
(20)

	
𝑉
𝑖
,
𝜀
(
𝑡
+
1
)
	
=
𝑉
~
𝑖
,
𝜀
(
𝑡
+
1
)
⋅
(
1
−
𝑠
𝑖
,
𝜀
(
𝑡
+
1
)
)
+
𝑉
reset
⋅
𝑠
𝑖
,
𝜀
(
𝑡
+
1
)
		
(21)

The LSE operates over the spatial neighborhood (circular boundary conditions).

Soft Spike Mechanism. The soft spike 
𝑠
𝜀
∈
(
0
,
1
)
 interpolates between no-spike (
𝑠
𝜀
≈
0
) and spike (
𝑠
𝜀
≈
1
). This differs fundamentally from surrogate gradient methods:

• 

Surrogate: Forward uses hard 
𝐻
​
(
𝑉
−
𝜃
)
; backward uses smooth 
𝑔
′
​
(
𝑉
)

• 

Ultradiscretized: Both forward and backward use same smooth 
𝜎
​
(
(
𝑉
−
𝜃
)
/
𝜀
)

On Soft vs. Binary Spikes. A natural concern is that 
𝑠
𝜀
∈
(
0
,
1
)
 does not represent “true” binary spikes. However, this deviation from binary behavior is not problematic in practice. The soft spike is used during training for gradient computation, while at inference one can employ small 
𝜀
 or hard thresholding (
𝑠
=
𝐻
​
(
𝑉
−
𝜃
)
) for neuromorphic deployment, following a standard training-inference separation analogous to dropout or batch normalization. Furthermore, the output layer uses mean spike rate 
𝑦
^
=
1
𝑇
​
∑
𝑡
𝑠
(
𝑡
)
, which is inherently robust to soft versus hard individual spikes since the classification decision depends on aggregate activity rather than precise spike values. The sparsity penalty 
𝜆
⋅
𝑠
¯
 encourages sparse soft activations that correspond to sparse hard spikes in the limit, ensuring energy-efficient inference. Most importantly, Proposition 5.2 guarantees that 
𝑠
𝜀
→
𝐻
​
(
𝑉
−
𝜃
)
 as 
𝜀
→
0
, providing a principled path from soft training dynamics to binary inference behavior.

Learnable Parameters. The temperature 
𝜀
 is made learnable via 
𝜀
=
exp
⁡
(
log
⁡
𝜀
param
)
, initialized to 
𝜀
0
=
1.0
. For UltraLIF, the leak factor 
𝜏
0
 can also be learned (UltraPLIF variant). For UltraDLIF, similarly UltraDPLIF learns 
𝜏
0
 for an optional temporal component. During training, the network discovers optimal soft-to-hard trade-offs, implementing automatic curriculum learning.

4.3Network Architecture

A feedforward SNN with 
𝐿
 layers of UltraLIF neurons is constructed as:

	
𝐼
𝑙
(
𝑡
)
	
=
𝑊
𝑙
⋅
𝑠
𝑙
−
1
(
𝑡
)
+
𝑏
𝑙
		
(22)

	
𝑉
𝑙
(
𝑡
+
1
)
	
=
LSE
𝜀
​
(
𝑉
𝑙
(
𝑡
)
+
log
⁡
𝜏
0
,
𝐼
𝑙
(
𝑡
)
)
​
(
1
−
𝑠
𝑙
(
𝑡
)
)
+
𝑉
reset
⋅
𝑠
𝑙
(
𝑡
)
		
(23)

	
𝑠
𝑙
(
𝑡
+
1
)
	
=
𝜎
​
(
(
𝑉
𝑙
(
𝑡
+
1
)
−
𝜃
)
/
𝜀
)
		
(24)

where 
𝑙
∈
{
1
,
…
,
𝐿
}
 indexes layers. The output layer employs spike rate coding:

	
𝑦
^
=
1
𝑇
​
∑
𝑡
=
1
𝑇
𝑠
𝐿
(
𝑡
)
		
(25)
5Theoretical Analysis
5.1Convergence to LIF Dynamics
Lemma 5.1 (Sigmoid Convergence).

Let 
𝜎
𝜀
​
(
𝑥
)
:=
𝜎
​
(
𝑥
/
𝜀
)
. For 
𝑥
≠
0
, 
lim
𝜀
→
0
+
𝜎
𝜀
​
(
𝑥
)
=
𝐻
​
(
𝑥
)
 with exponential convergence rate 
|
𝜎
𝜀
​
(
𝑥
)
−
𝐻
​
(
𝑥
)
|
≤
𝑒
−
|
𝑥
|
/
𝜀
.

Proof.

For 
𝑥
>
0
: 
𝜎
𝜀
​
(
𝑥
)
=
(
1
+
𝑒
−
𝑥
/
𝜀
)
−
1
→
1
 as 
𝜀
→
0
+
.

For 
𝑥
<
0
: 
𝜎
𝜀
​
(
𝑥
)
=
𝑒
𝑥
/
𝜀
/
(
𝑒
𝑥
/
𝜀
+
1
)
→
0
 as 
𝜀
→
0
+
.

The error bound follows from 
|
1
−
𝜎
𝜀
​
(
𝑥
)
|
=
𝑒
−
𝑥
/
𝜀
/
(
1
+
𝑒
−
𝑥
/
𝜀
)
≤
𝑒
−
𝑥
/
𝜀
=
𝑒
−
|
𝑥
|
/
𝜀
 for 
𝑥
>
0
. For 
𝑥
<
0
: 
|
𝜎
𝜀
​
(
𝑥
)
|
=
1
/
(
1
+
𝑒
−
𝑥
/
𝜀
)
≤
𝑒
𝑥
/
𝜀
=
𝑒
−
|
𝑥
|
/
𝜀
. ∎

Proposition 5.2 (Convergence to LIF).

Let 
{
𝑉
𝜀
​
(
𝑡
)
}
 denote the UltraLIF trajectory with temperature 
𝜀
>
0
, and 
{
𝑣
​
(
𝑡
)
,
𝑠
​
(
𝑡
)
}
 the standard LIF trajectory with 
𝑉
reset
=
0
. Assume (A1) bounded inputs 
|
𝐼
​
(
𝑡
)
|
≤
𝐼
max
 and (A2) threshold margin 
𝛿
𝑡
:=
|
𝑣
​
(
𝑡
)
−
𝜃
|
>
0
. Then 
lim
𝜀
→
0
+
𝑉
𝜀
​
(
𝑡
)
=
𝑣
​
(
𝑡
)
 and 
lim
𝜀
→
0
+
𝑠
𝜀
​
(
𝑡
)
=
𝑠
​
(
𝑡
)
 for each 
𝑡
, with 
|
𝑉
𝜀
​
(
𝑡
)
−
𝑣
​
(
𝑡
)
|
≤
𝑡
⋅
𝜀
​
log
⁡
2
 and 
|
𝑠
𝜀
​
(
𝑡
)
−
𝑠
​
(
𝑡
)
|
≤
𝑒
−
𝛿
𝑡
/
𝜀
. The linear error growth follows from the 1-Lipschitz property of 
LSE
𝜀
 (Lemma C.1) and the non-expansiveness of the reset interpolation. The set of inputs violating (A2) has Lebesgue measure zero.

Proof.

By strong induction on 
𝑡
. Base case: 
𝑉
𝜀
​
(
0
)
=
𝑣
​
(
0
)
; spike convergence by Lemma 5.1. For the inductive step, the update 
𝐹
𝜀
​
(
𝑉
)
=
𝑉
~
​
(
1
−
𝑠
𝜀
)
+
𝑉
reset
⋅
𝑠
𝜀
 with 
𝑉
~
=
LSE
𝜀
​
(
𝑉
+
log
⁡
𝜏
0
,
𝐼
)
 satisfies: (i) no spike (
𝑠
𝜀
→
0
): 
𝐹
𝜀
→
max
⁡
(
𝑣
​
(
𝑡
)
+
log
⁡
𝜏
0
,
𝐼
​
(
𝑡
)
)
=
𝑣
​
(
𝑡
+
1
)
; (ii) spike (
𝑠
𝜀
→
1
): 
𝐹
𝜀
→
𝑉
reset
=
0
, matching LIF reset. Each step adds at most 
𝜀
​
log
⁡
2
 error (Lemma 3.2); the convex reset interpolation does not amplify it. Full details in Appendix C. ∎

Corollary 5.3 (UltraDLIF Convergence).

The analogous result holds for UltraDLIF: as 
𝜀
→
0
+
, the 3-term 
LSE
𝜀
​
(
𝑉
𝑖
−
1
,
𝑉
𝑖
,
𝑉
𝑖
+
1
)
→
max
⁡
(
𝑉
𝑖
−
1
,
𝑉
𝑖
,
𝑉
𝑖
+
1
)
 by Lemma 3.2 with 
𝑛
=
3
, and the full UltraDLIF trajectory converges to the max-plus diffusion dynamics (Eq. 14) with error bound 
|
𝑉
𝑖
,
𝜀
​
(
𝑡
)
−
𝑉
𝑖
​
(
𝑡
)
|
≤
𝑡
⋅
𝜀
​
log
⁡
3
.

5.2Gradient Properties
Proposition 5.4 (Bounded Non-Vanishing Gradients).

For any 
𝜀
>
0
, consider a single-step spike output 
𝑠
𝜀
=
𝜎
​
(
(
𝑉
~
𝜀
−
𝜃
)
/
𝜀
)
 where 
𝑉
~
𝜀
 is the pre-reset voltage. The gradient satisfies 
0
<
∂
𝑠
𝜀
∂
𝑉
~
𝜀
≤
1
4
​
𝜀
 for all finite 
𝑉
~
𝜀
. For weights 
𝑊
 with input 
𝑥
 at a single layer: 
|
∂
𝑠
𝜀
∂
𝑊
𝑖
​
𝑗
|
≤
‖
𝑥
‖
∞
4
​
𝜀
.

Proof.

Let 
𝑧
=
(
𝑉
𝜀
−
𝜃
)
/
𝜀
. Then 
∂
𝑠
𝜀
∂
𝑉
𝜀
=
𝜎
​
(
𝑧
)
​
(
1
−
𝜎
​
(
𝑧
)
)
𝜀
. The function 
𝜎
​
(
1
−
𝜎
)
 achieves maximum 
1
/
4
 at 
𝜎
=
1
/
2
, establishing the upper bound. Positivity follows since 
𝜎
​
(
𝑧
)
∈
(
0
,
1
)
 for finite 
𝑧
. The weight gradient bound follows by chain rule: 
∂
𝑠
𝜀
∂
𝑊
𝑖
​
𝑗
=
∂
𝑠
𝜀
∂
𝑉
𝜀
⋅
∂
𝑉
𝜀
∂
𝐼
⋅
𝑥
𝑗
, with the 
LSE
 gradient being softmax with values in 
(
0
,
1
)
. ∎

Corollary 5.5 (Gradient Scaling).

The temperature 
𝜀
 controls the bias-variance trade-off in gradients:

• 

Larger 
𝜀
: smaller gradients, smoother optimization landscape

• 

Smaller 
𝜀
: larger gradients near threshold, better LIF approximation

5.3Forward-Backward Consistency
Remark 5.6 (Gradient Consistency).

For any 
𝜀
>
0
, UltraLIF is a composition of smooth operations (
LSE
𝜀
, sigmoid, affine maps), so the chain rule applies exactly: the backward pass differentiates the same function computed forward. Surrogate methods use 
𝐻
​
(
𝑉
−
𝜃
)
 forward but differentiate a surrogate 
𝑔
​
(
𝑉
−
𝜃
)
 backward; Gygax and Zenke (2025) show this can be interpreted as differentiating a stochastic forward pass with escape noise. UltraLIF avoids this mismatch: 
∇
𝑊
ℒ
​
(
𝑓
𝜀
​
(
𝐱
;
𝑊
)
)
 is an exact gradient of the actual forward computation.

5.4Connection to Tropical Geometry
Theorem 5.7 (Tropical Limit).

The map 
𝐷
𝜀
:
(
ℝ
>
0
,
+
,
⋅
)
→
(
ℝ
,
⊕
𝜀
,
+
)
 defined by 
𝐷
𝜀
​
(
𝑥
)
=
𝜀
​
log
⁡
𝑥
, where 
𝑎
⊕
𝜀
𝑏
=
LSE
𝜀
​
(
𝑎
,
𝑏
)
, is a semiring homomorphism. As 
𝜀
→
0
+
, 
⊕
𝜀
→
⊕
=
max
 (tropical addition), UltraLIF dynamics converge to a piecewise-linear map on 
ℝ
max
, and decision boundaries approach tropical hypersurfaces.

Proof.

𝐷
𝜀
​
(
𝑥
⋅
𝑦
)
=
𝜀
​
log
⁡
(
𝑥
​
𝑦
)
=
𝐷
𝜀
​
(
𝑥
)
+
𝐷
𝜀
​
(
𝑦
)
 (multiplication 
→
 addition) and 
𝐷
𝜀
​
(
𝑥
+
𝑦
)
=
𝜀
​
log
⁡
(
𝑥
+
𝑦
)
=
LSE
𝜀
​
(
𝐷
𝜀
​
(
𝑥
)
,
𝐷
𝜀
​
(
𝑦
)
)
 (addition 
→
 soft-max). Taking 
𝜀
→
0
 yields the tropical semiring by Lemma 3.2. The piecewise-linear limit and tropical hypersurface structure follow from Zhang et al. (2018). ∎

6Experiments

Setup. Evaluation spans six benchmarks: static images (MNIST, Fashion-MNIST, CIFAR-10), neuromorphic vision (N-MNIST, DVS-Gesture), and audio (SHD). A single hidden layer with 64 neurons is used across all experiments, with timesteps 
𝑇
∈
{
1
,
10
,
30
}
. Baselines include LIF, PLIF, AdaLIF, FullPLIF, and DSpike/DSpike+ with surrogate gradients. All four ultradiscretized variants (UltraLIF, UltraPLIF, UltraDLIF, UltraDPLIF) are evaluated. An optional sparsity penalty 
ℒ
=
ℒ
CE
+
𝜆
⋅
𝑠
¯
 enables explicit accuracy-efficiency trade-offs. Energy is estimated via the relative synaptic operation (SOP) count 
𝑇
⋅
𝑠
¯
 (Lemaire et al., 2023), which is proportional to computational energy since all models share the same architecture (Appendix D).

Table 2:Test accuracy (%) on CIFAR-10. UltraPLIF (temporal) achieves best at all timesteps.
Model	
𝑇
=
1
	
𝑇
=
10
	
𝑇
=
30

LIF	39.83	44.27	45.69
PLIF	39.83	45.06	46.15
AdaLIF	39.83	44.86	45.83
FullPLIF	39.60	45.43	46.28
DSpike	40.26	44.78	45.34
DSpike+	40.26	45.42	46.29
Temporal (LIF ODE)
UltraLIF	40.72	45.15	45.69
UltraPLIF	43.27	46.19	46.58
Spatial (Diffusion PDE)
UltraDLIF	43.11	45.65	45.00
UltraDPLIF	43.11	45.75	45.74
Table 3:Energy efficiency on CIFAR-10. Energy = relative SOP count (
𝑇
⋅
𝑠
¯
). Sparsity penalty 
𝜆
 reduces spike rate with minimal accuracy loss. At 
𝑇
=
30
, UltraPLIF with 
𝜆
=
0.1
 achieves best accuracy while reducing energy by 50%.
Model	
𝑇
	Acc (%)	Spike	Energy
LIF	1	39.83	0.404	0.40
DSpike+	1	40.26	0.386	0.39
UltraPLIF	1	43.27	0.458	0.46
UltraPLIF (
𝜆
=
0.1
) 	1	43.60	0.240	0.24
PLIF	10	45.06	0.356	3.56
UltraDPLIF	10	45.75	0.469	4.69
UltraDPLIF (
𝜆
=
0.1
) 	10	45.32	0.338	3.38
PLIF	30	46.15	0.377	11.30
UltraPLIF	30	46.58	0.500	15.01
UltraPLIF (
𝜆
=
0.1
) 	30	46.98	0.248	7.44
Table 4:Test accuracy (%) on MNIST. UltraDLIF (spatial) achieves best at 
𝑇
=
1
; UltraPLIF (temporal) at 
𝑇
=
30
.
Model	
𝑇
=
1
	
𝑇
=
10
	
𝑇
=
30

LIF	95.34	97.45	97.45
PLIF	95.34	97.40	97.33
AdaLIF	95.34	97.43	97.45
FullPLIF	95.27	97.33	97.31
DSpike	95.58	97.38	97.48
DSpike+	95.58	97.39	97.27
Temporal (LIF ODE)
UltraLIF	94.37	97.14	97.46
UltraPLIF	95.60	97.30	97.55
Spatial (Diffusion PDE)
UltraDLIF	95.67	97.35	97.38
UltraDPLIF	95.67	97.35	97.40
Table 5:Energy efficiency on MNIST. Energy = relative SOP count (
𝑇
⋅
𝑠
¯
). UltraDLIF with 
𝜆
=
0.1
 reduces spike rate by 40% (0.446 
→
 0.268). At 
𝑇
=
10
, UltraDPLIF with 
𝜆
=
0.1
 achieves 50% energy reduction.
Model	
𝑇
	Acc (%)	Spike	Energy
LIF	1	95.34	0.388	0.39
DSpike+	1	95.58	0.403	0.40
UltraDLIF	1	95.67	0.446	0.45
UltraDLIF (
𝜆
=
0.1
) 	1	95.71	0.268	0.27
DSpike+	10	97.39	0.415	4.15
UltraDLIF (
𝜆
=
0.01
) 	10	97.56	0.448	4.48
UltraDPLIF (
𝜆
=
0.1
) 	10	97.35	0.237	2.37
Table 6:Test accuracy (%) on N-MNIST (neuromorphic). UltraDLIF achieves +3.91% over baselines at 
𝑇
=
1
.
Model	
𝑇
=
1
	
𝑇
=
10
	
𝑇
=
30

LIF	88.54	97.48	97.29
PLIF	88.54	97.53	97.61
AdaLIF	88.54	97.38	97.30
FullPLIF	89.00	97.50	97.48
DSpike	90.23	97.39	97.59
DSpike+	90.23	97.55	97.65
Temporal (LIF ODE)
UltraLIF	90.41	96.10	95.87
UltraPLIF	93.11	96.33	95.77
Spatial (Diffusion PDE)
UltraDLIF	94.14	97.38	97.46
UltraDPLIF	94.14	97.38	97.68
Table 7:Test accuracy (%) on DVS-Gesture (neuromorphic). UltraPLIF achieves +7.96% at 
𝑇
=
1
.
Model	
𝑇
=
1
	
𝑇
=
10
	
𝑇
=
30

LIF	52.27	67.05	79.92
PLIF	52.27	68.94	78.79
AdaLIF	47.73	68.94	78.79
FullPLIF	47.73	68.56	77.27
DSpike	51.14	67.42	78.79
DSpike+	51.14	66.67	78.41
Temporal (LIF ODE)
UltraLIF	58.33	69.32	75.00
UltraPLIF	60.23	68.94	75.76
Spatial (Diffusion PDE)
UltraDLIF	58.33	69.32	78.41
UltraDPLIF	58.33	68.56	79.92
Table 8:Test accuracy (%) on SHD (audio). At 
𝑇
=
1
, UltraDLIF achieves +11.22% over the best baseline (FullPLIF). Baselines lead at 
𝑇
≥
10
.
Model	
𝑇
=
1
	
𝑇
=
10
	
𝑇
=
30

LIF	27.69	72.66	72.66
PLIF	27.69	71.69	73.19
AdaLIF	27.69	74.03	71.07
FullPLIF	40.02	71.69	72.48
DSpike	38.21	72.04	70.89
DSpike+	38.21	72.75	73.85
Temporal (LIF ODE)
UltraLIF	44.88	58.79	59.14
UltraPLIF	46.91	57.73	59.45
Spatial (Diffusion PDE)
UltraDLIF	51.24	67.62	71.60
UltraDPLIF	51.24	68.90	67.84
Table 9:Summary: 
𝑇
=
1
 accuracy (%) across datasets. Best baseline and best ultradiscretized model shown in parentheses. Full results in Tables 2–8 and Appendix E.
Dataset	Baseline	Best Ultra	
Δ

MNIST	95.58 (DSpike)	95.67 (DLIF)	+0.09
Fashion	82.67 (DSpike)	83.02 (PLIF)	+0.35
CIFAR-10	40.26 (DSpike)	43.27 (PLIF)	+3.01
N-MNIST	90.23 (DSpike)	94.14 (DLIF)	+3.91
DVS	52.27 (PLIF)	60.23 (PLIF)	+7.96
SHD	40.02 (FullPLIF)	51.24 (DLIF)	+11.22

Single-Timestep Advantage. The ultradiscretized models’ advantage is most pronounced at 
𝑇
=
1
 (Table 9), where the model must extract maximum information from a single forward pass. Crucially, gains are largest on neuromorphic and temporal datasets: SHD (+11.22%), DVS-Gesture (+7.96%), N-MNIST (+3.91%), and CIFAR-10 (+3.01%). On simpler static datasets, gains are smaller but consistent: Fashion-MNIST (+0.35%), MNIST (+0.09%). Among the two derivations, UltraDLIF (spatial) wins on N-MNIST, SHD, and MNIST, while UltraPLIF (temporal) wins on CIFAR-10, Fashion-MNIST, and DVS-Gesture (Table 15), suggesting both variants contribute complementary strengths.

Neuromorphic Dataset Performance. Tables 6, 7, and 8 show that ultradiscretized models dramatically outperform baselines on neuromorphic benchmarks at 
𝑇
=
1
. On N-MNIST, UltraDLIF achieves 94.14% versus 90.23% for DSpike (+3.91%). On DVS-Gesture, UltraPLIF (temporal) achieves the best result: 60.23% versus 52.27% for PLIF (+7.96%). These datasets capture asynchronous events from dynamic vision sensors, where the temporal structure is fundamental. The soft max-plus dynamics appear better suited to extract information from sparse, event-driven inputs than hard-thresholding surrogates.

Sparsity-Accuracy Trade-off. Without sparsity penalty, ultradiscretized models tend to have higher spike rates than baselines (e.g., CIFAR-10 
𝑇
=
1
: UltraPLIF 0.458 vs. LIF 0.404), as the soft spike 
𝑠
𝜀
∈
(
0
,
1
)
 contributes nonzero activity even below threshold. The sparsity penalty 
𝜆
 addresses this and enables flexible energy-accuracy trade-offs (Tables 3, 5). On CIFAR-10, 
𝜆
=
0.1
 reduces spike rates by 48% (0.458 
→
 0.240) while actually improving accuracy (43.27% 
→
 43.60%). On MNIST, 
𝜆
=
0.1
 reduces spike rates by 40% at 
𝑇
=
1
 (0.446 
→
 0.268) while maintaining or slightly improving accuracy. Remarkably, at 
𝑇
=
30
 on CIFAR-10, UltraPLIF with 
𝜆
=
0.1
 achieves both best accuracy (46.98%) and 50% energy reduction compared to the no-penalty baseline.

Timestep Scaling. Performance gaps narrow at higher 
𝑇
, and baselines often overtake: on CIFAR-10, the 
𝑇
=
1
 advantage (+3.01% for UltraPLIF) persists but shrinks at 
𝑇
=
10
 and 
𝑇
=
30
. On SHD, ultradiscretized models lead dramatically at 
𝑇
=
1
 (+11.22%) but baselines surpass them at 
𝑇
≥
10
. This pattern is consistent: on N-MNIST, UltraDLIF leads by +3.91% at 
𝑇
=
1
 but baselines lead at 
𝑇
=
10
. The explanation is that surrogate gradients suffer from forward-backward mismatch, but with sufficient 
𝑇
, the averaging effect of spike rate coding masks individual spike errors. At low 
𝑇
, each spike carries more information, making the consistency of ultradiscretization more valuable.

Learnable Temperature. The temperature 
𝜀
 implements automatic curriculum learning, starting soft (large 
𝜀
) for easy optimization then sharpening (small 
𝜀
) to approximate discrete spikes. Unlike DSpike’s heuristic sharpness parameter, 
𝜀
 has principled convergence guarantees (Proposition 5.2). Ablation studies (Appendix E.1) confirm that learned 
𝜀
 consistently outperforms fixed values, converging to the range 0.66–1.08.

Computational Cost. The 
LSE
 operation adds minor overhead (
∼
5% wall-clock time) compared to standard LIF, but is fully parallelizable on GPU/TPU.

7Discussion

Temporal and Spatial Instantiations. UltraLIF and UltraDLIF arise from applying ultradiscretization to different source equations (LIF ODE vs. diffusion PDE), yielding models with distinct computational properties. UltraLIF (2-term LSE) captures temporal membrane dynamics; UltraDLIF (3-term LSE) models lateral spatial diffusion. Both share the same theoretical guarantees from Lemmas and Theorems in Section 5.

Relation to Surrogate Gradients. The soft spike (17) superficially resembles sigmoid surrogates. The key distinction is consistency: ultradiscretized neurons employ identical soft dynamics in both forward and backward passes, whereas surrogate methods use hard spikes forward with soft gradients backward. This consistency eliminates the gradient mismatch analyzed in Gygax and Zenke (2025).

Biological Interpretation. The temperature 
𝜀
 admits interpretation as neural noise or stochasticity. Biological cortical neurons exhibit highly irregular firing patterns (Softky and Koch, 1993); the ultradiscretization framework provides a principled model where 
𝜀
 quantifies this variability.

Tropical Geometry Perspective. Theorem 5.7 connects SNN dynamics to tropical algebraic geometry. This opens avenues for applying tropical techniques such as Newton polytopes, Bezout bounds, and tropical Nullstellensatz to analyze SNN expressivity and decision boundaries.

Consequence: A New Activation Family. A direct consequence of the ultradiscretization framework is that UltraLIF subsumes classical activation functions: 
LSE
𝜀
​
(
0
,
𝑥
)
 recovers the softplus (and ReLU as 
𝜀
→
0
), while 
𝑠
𝜀
 is a scaled sigmoid. The learnable 
𝜀
 thus interpolates between hard spiking and smooth regimes. Details and connections to morphological neural networks are in Appendix B.

Limitations. UltraDLIF and UltraDPLIF produce identical results in our experiments, indicating that the learnable leak 
𝜏
 does not differentiate the spatial variant; the 3-term LSE dominates the dynamics. Tasks requiring precise spike timing may benefit differently from temporal (UltraLIF) versus spatial (UltraDLIF) variants. The soft spike 
𝑠
𝜀
∈
(
0
,
1
)
 during training deviates from binary spikes; however, as discussed in Section 4.2, this is addressed by: (1) using hard thresholding at inference for neuromorphic deployment, (2) the convergence guarantee 
𝑠
𝜀
→
𝐻
​
(
𝑉
−
𝜃
)
 as 
𝜀
→
0
, and (3) the learned 
𝜀
 converging to moderate values (0.66–1.08) that balance differentiability and spike sharpness (Figure 3).

8Conclusion

This paper introduced UltraLIF, a principled framework for differentiable spiking neural networks grounded in ultradiscretization from tropical geometry. Two neuron models, UltraLIF (temporal, from the LIF ODE capturing membrane decay) and UltraDLIF (spatial, from the diffusion PDE modeling gap junction coupling across neuronal populations), use log-sum-exp as a soft relaxation of max-plus dynamics, yielding fully differentiable SNNs without surrogate gradients. Theoretical analysis establishes convergence to classical LIF dynamics, bounded gradients, and forward-backward consistency. Experiments on six benchmarks demonstrate improvements over surrogate gradient baselines, with the largest gains at 
𝑇
=
1
 on neuromorphic and audio data (SHD +11.22%, DVS +7.96%, N-MNIST +3.91%). An optional sparsity penalty enables significant energy reduction while maintaining accuracy. The connection to tropical geometry opens new directions for principled analysis of spiking computation.

Impact Statement

This paper presents theoretical work whose goal is to advance the field of Machine Learning, specifically in the domain of energy-efficient spiking neural networks. The ultradiscretization framework provides mathematical foundations that could accelerate deployment of neuromorphic computing systems, potentially reducing the energy footprint of machine learning applications. All experiments are validated on existing public benchmark datasets. There are many potential societal consequences of this work, none of which must be specifically highlighted here.

References
G. Bellec, D. Salaj, A. Subramoney, R. Legenstein, and W. Maass (2018)
↑
	Long short-term memory and learning-to-learn in networks of spiking neurons.In Advances in Neural Information Processing Systems,Vol. 31.Cited by: §D.4, §2.
S. M. Bohte, J. N. Kok, and H. La Poutré (2002)
↑
	Error-backpropagation in temporally encoded networks of spiking neurons.Neurocomputing 48 (1-4), pp. 17–37.External Links: DocumentCited by: §2.
T. Bu, W. Fang, J. Ding, P. Dai, Z. Yu, and T. Huang (2022)
↑
	Optimal ANN-SNN conversion for high-accuracy and ultra-low-latency spiking neural networks.In International Conference on Learning Representations,Note: arXiv:2303.04347Cited by: §2.
Y. Cao, Y. Chen, and D. Khosla (2015)
↑
	Spiking deep convolutional neural networks for energy-efficient object recognition.International Journal of Computer Vision 113, pp. 54–66.External Links: DocumentCited by: §2.
B. W. Connors and M. A. Long (2004)
↑
	Electrical synapses in the mammalian brain.Annual Review of Neuroscience 27, pp. 393–418.External Links: DocumentCited by: §4.1.2.
M. Davies, N. Srinivasa, T. Lin, G. Chinya, Y. Cao, S. H. Choday, G. Dimou, P. Joshi, N. Imam, S. Jain, et al. (2018)
↑
	Loihi: a neuromorphic manycore processor with on-chip learning.IEEE Micro 38 (1), pp. 82–99.External Links: DocumentCited by: §1.
W. Fang, Z. Yu, Y. Chen, T. Masquelier, T. Huang, and Y. Tian (2021)
↑
	Incorporating learnable membrane time constant to enhance learning of spiking neural networks.In Proceedings of the IEEE/CVF International Conference on Computer Vision,pp. 2661–2671.External Links: DocumentCited by: §D.4, §2.
K. Fotopoulos, P. Maragos, and P. Misiakos (2024)
↑
	TropNNC: structured neural network compression using tropical geometry.arXiv preprint arXiv:2409.03945.Cited by: §2.
G. Franchi, A. Fehri, and A. Yao (2020)
↑
	Deep morphological networks.Pattern Recognition 102, pp. 107246.External Links: DocumentCited by: Appendix B.
J. Göltz, L. Kriener, A. Baumbach, S. Billaudelle, O. Breitwieser, B. Cramer, D. Dold, A. F. Kungl, W. Senn, J. Schemmel, et al. (2021)
↑
	Fast and energy-efficient neuromorphic deep learning with first-spike times.Nature Machine Intelligence 3 (9), pp. 823–835.External Links: DocumentCited by: §2.
B. Grammaticos, Y. Kosmann-Schwarzbach, and T. Tamizhmani (Eds.) (2004)
↑
	Discrete integrable systems.Lecture Notes in Physics, Vol. 644, Springer.External Links: DocumentCited by: §1.
J. Gygax and F. Zenke (2025)
↑
	Elucidating the theoretical underpinnings of surrogate gradient learning in spiking neural networks.Neural Computation 37 (5), pp. 886–925.Note: arXiv:2404.14964External Links: DocumentCited by: §1, §2, Remark 5.6, §7.
M. Horowitz (2014)
↑
	1.1 computing’s energy problem (and what we can do about it).IEEE International Solid-State Circuits Conference Digest of Technical Papers, pp. 10–14.External Links: DocumentCited by: §D.7.
S. R. Kheradpisheh and T. Masquelier (2020)
↑
	Temporal backpropagation for spiking neural networks with one spike per neuron.International Journal of Neural Systems 30 (06), pp. 2050027.External Links: DocumentCited by: §2.
E. Lemaire, L. Cordone, A. Castagnetti, P. Novac, J. Courtois, and B. Miramond (2023)
↑
	An analytical estimation of spiking neural networks energy efficiency.In International Conference on Neural Information Processing,pp. 574–587.External Links: DocumentCited by: §D.7, §6.
Y. Li, Y. Guo, S. Zhang, S. Deng, Y. Hai, and S. Gu (2021)
↑
	Differentiable spike: rethinking gradient-descent for training spiking neural networks.In Advances in Neural Information Processing Systems,Vol. 34, pp. 23426–23439.Cited by: §D.4, §1, §2.
S. Lian, J. Shen, Q. Liu, Z. Wang, R. Yan, and H. Tang (2023)
↑
	Learnable surrogate gradient for direct training spiking neural networks.In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence,pp. 3002–3010.External Links: DocumentCited by: §2.
W. Maass (1997)
↑
	Networks of spiking neurons: the third generation of neural network models.Neural Networks 10 (9), pp. 1659–1671.External Links: DocumentCited by: §1.
D. Maclagan and B. Sturmfels (2015)
↑
	Introduction to tropical geometry.Vol. 161, American Mathematical Society.External Links: DocumentCited by: §2.
H. Mostafa (2018)
↑
	Supervised learning based on temporal coding in spiking neural networks.IEEE Transactions on Neural Networks and Learning Systems 29 (7), pp. 3227–3235.External Links: DocumentCited by: §2.
E. O. Neftci, H. Mostafa, and F. Zenke (2019)
↑
	Surrogate gradient learning in spiking neural networks: bringing the power of gradient-based optimization to spiking neural networks.IEEE Signal Processing Magazine 36 (6), pp. 51–63.External Links: DocumentCited by: §1, §2.
T. Ochiai and J. C. Nacher (2005)
↑
	Inversible max-plus algebras and integrable systems.Journal of Mathematical Physics 46 (6), pp. 063507.External Links: DocumentCited by: §4.1.2.
T. A. Pham and V. Garg (2024)
↑
	What do graph neural networks learn? Insights from tropical geometry.In Advances in Neural Information Processing Systems,Vol. 37.External Links: DocumentCited by: §2.
K. Roy, A. Jaiswal, and P. Panda (2019)
↑
	Towards spike-based machine intelligence with neuromorphic computing.Nature 575 (7784), pp. 607–617.External Links: DocumentCited by: §1.
B. Rueckauer, I. Lungu, Y. Hu, M. Pfeiffer, and S. Liu (2017)
↑
	Conversion of continuous-valued deep networks to efficient event-driven networks for image classification.Frontiers in Neuroscience 11, pp. 682.External Links: DocumentCited by: §2.
W. R. Softky and C. Koch (1993)
↑
	The highly irregular firing of cortical cells is inconsistent with temporal integration of random EPSPs.Journal of Neuroscience 13 (1), pp. 334–350.External Links: DocumentCited by: §7.
L. Spek, Y. A. Kuznetsov, and S. A. van Gils (2020)
↑
	Neural field models with transmission delays and diffusion.Journal of Mathematical Neuroscience 10 (1), pp. 21.External Links: DocumentCited by: §4.1.2.
D. Takahashi and J. Satsuma (1990)
↑
	A soliton cellular automaton.Journal of the Physical Society of Japan 59 (10), pp. 3514–3519.External Links: DocumentCited by: §2.
T. Tokihiro, D. Takahashi, J. Matsukidaira, and J. Satsuma (1996)
↑
	From soliton equations to integrable cellular automata through a limiting procedure.Physical Review Letters 76 (18), pp. 3247–3250.External Links: DocumentCited by: §1, §2, §4.1.1, §4.1.
Y. Yamazaki and S. Ohmori (2023)
↑
	Emergence of ultradiscrete states due to phase lock caused by saddle-node bifurcation in discrete limit cycles.Progress of Theoretical and Experimental Physics 2023 (8), pp. 081A01.External Links: DocumentCited by: §2.
Y. Yamazaki and S. Ohmori (2024)
↑
	Ultradiscretization in discrete limit cycles of tropically discretized and max-plus Sel’kov models.JSIAM Letters 16, pp. 85–88.External Links: DocumentCited by: §2.
Z. Yan, Z. Bai, and W. Wong (2024)
↑
	Reconsidering the energy efficiency of spiking neural networks.arXiv preprint arXiv:2409.08290.Cited by: §D.7.
T. Zaslavsky (1975)
↑
	Facing up to arrangements: face-count formulas for partitions of space by hyperplanes.Memoirs of the American Mathematical Society 1 (154).External Links: DocumentCited by: §C.3.1, §C.3.3.
F. Zenke and S. Ganguli (2018)
↑
	SuperSpike: supervised learning in multilayer spiking neural networks.Neural Computation 30 (6), pp. 1514–1541.External Links: DocumentCited by: §2.
F. Zenke and T. P. Vogels (2021)
↑
	The remarkable robustness of surrogate gradient learning for instilling complex function in spiking neural networks.Neural Computation 33 (4), pp. 899–925.External Links: DocumentCited by: §1.
L. Zhang, G. Naitzat, and L. Lim (2018)
↑
	Tropical geometry of deep neural networks.In International Conference on Machine Learning,pp. 5824–5832.Cited by: §2, §5.4.
Appendix ASpike Mechanism Comparison
Figure 2:Comparison of spike mechanisms. (a) Traditional LIF uses Heaviside 
𝐻
​
(
𝑉
−
𝜃
)
 in the forward pass but a smooth surrogate 
𝜎
′
 for gradients, creating forward-backward mismatch (shaded region). (b) Ultradiscretized LIF (temporal, 2-term LSE from LIF ODE) and (c) Ultradiscretized DLIF (spatial, 3-term LSE from diffusion PDE) use identical smooth functions in both passes, ensuring gradient consistency. The membrane potential equations below each panel show the distinct derivations.
Appendix BAdditional Discussion

UltraLIF as a New Activation Family. The ultradiscretization framework yields a novel activation function family with deep connections to existing architectures. Observe that 
LSE
𝜀
​
(
0
,
𝑥
)
=
𝜀
​
log
⁡
(
1
+
exp
⁡
(
𝑥
/
𝜀
)
)
 is precisely the softplus function, which converges to 
ReLU
​
(
𝑥
)
=
max
⁡
(
0
,
𝑥
)
 as 
𝜀
→
0
. Thus, the membrane dynamics in UltraLIF generalize ReLU-family activations to the spiking domain. Meanwhile, the spike function 
𝑠
𝜀
=
𝜎
​
(
(
𝑉
−
𝜃
)
/
𝜀
)
 is a shifted, scaled sigmoid, i.e., a soft step function. Together, UltraLIF combines soft-max aggregation (ReLU family) with soft-step thresholding (sigmoid family), both controlled by the learnable temperature 
𝜀
. This provides a principled spectrum of activations:

• 

𝜀
→
0
: Hard max + hard step (classical LIF, non-differentiable)

• 

𝜀
→
∞
: Linear + constant (no nonlinearity)

• 

𝜀
 learned: Optimal sharpness (UltraLIF, fully differentiable)

Unlike heuristic surrogate gradients, this activation family has rigorous theoretical grounding in tropical geometry and maintains forward-backward consistency by design.

Connection to Morphological Neural Networks. The spatial max operation in UltraDLIF (Eq. 14) corresponds to morphological dilation, a fundamental operation in mathematical morphology. Deep morphological networks (Franchi et al., 2020) show that max pooling is equivalent to dilation with a flat structuring element. This connection suggests that UltraDLIF performs learned morphological operations on neural activation patterns, providing an alternative interpretation grounded in image processing theory.

Appendix CProofs
C.1Proof of Proposition 5.2

The proof proceeds by strong induction on 
𝑡
, tracking errors explicitly at each step.

At the base case 
𝑡
=
0
, the voltage error is zero by initialization (
𝑉
𝜀
​
(
0
)
=
𝑣
0
=
𝑣
​
(
0
)
), and the spike error 
|
𝑠
𝜀
​
(
0
)
−
𝑠
​
(
0
)
|
≤
𝑒
−
𝛿
0
/
𝜀
→
0
 follows from Lemma 5.1 and the threshold margin assumption (A2).

For the inductive step, assume 
|
𝑉
𝜀
​
(
𝑘
)
−
𝑣
​
(
𝑘
)
|
≤
𝑘
⋅
𝜀
​
log
⁡
2
 holds for all 
𝑘
≤
𝑡
, and consider the UltraLIF update 
𝐹
𝜀
​
(
𝑉
)
=
𝑉
~
​
(
1
−
𝑠
𝜀
)
+
𝑉
reset
⋅
𝑠
𝜀
 with 
𝑉
~
=
LSE
𝜀
​
(
𝑉
+
log
⁡
𝜏
0
,
𝐼
)
 at step 
𝑡
+
1
. When no spike occurs at 
𝑡
, the inductive hypothesis and Lemma 5.1 imply 
𝑠
𝜀
​
(
𝑡
)
→
0
, so the reset interpolation reduces to 
𝑉
𝜀
​
(
𝑡
+
1
)
≈
𝑉
~
𝜀
​
(
𝑡
+
1
)
=
LSE
𝜀
​
(
𝑉
𝜀
​
(
𝑡
)
+
log
⁡
𝜏
0
,
𝐼
​
(
𝑡
)
)
. Since 
LSE
𝜀
 is 1-Lipschitz (Lemma C.1), the accumulated error propagates with factor at most 1, and the intrinsic LSE approximation contributes at most 
𝜀
​
log
⁡
2
 (Lemma 3.2 with 
𝑛
=
2
), giving 
|
𝑉
𝜀
​
(
𝑡
+
1
)
−
𝑣
​
(
𝑡
+
1
)
|
≤
|
𝑉
𝜀
​
(
𝑡
)
−
𝑣
​
(
𝑡
)
|
+
𝜀
​
log
⁡
2
≤
(
𝑡
+
1
)
​
𝜀
​
log
⁡
2
. When a spike does occur, 
𝑠
𝜀
​
(
𝑡
)
→
1
 drives the reset interpolation toward 
𝑉
reset
=
0
, matching the standard LIF reset exactly; the error thus resets to 
𝑂
​
(
𝜀
)
, dominated by 
(
𝑡
+
1
)
​
𝜀
​
log
⁡
2
.

Crucially, the reset interpolation 
𝑉
~
​
(
1
−
𝑠
)
+
𝑉
reset
⋅
𝑠
 is a convex combination for 
𝑠
∈
(
0
,
1
)
, making it non-expansive: perturbations in 
𝑉
~
 are damped by the factor 
(
1
−
𝑠
)
≤
1
. Combined with the 1-Lipschitz property of 
LSE
𝜀
, the one-step map never amplifies errors, yielding the linear growth 
𝐶
𝑡
=
𝑡
. Spike convergence at 
𝑡
+
1
 then follows from Lemma 5.1: 
|
𝑠
𝜀
​
(
𝑡
+
1
)
−
𝑠
​
(
𝑡
+
1
)
|
≤
𝑒
−
𝛿
𝑡
+
1
/
𝜀
 under Assumption (A2). ∎

C.2Lipschitz Properties
Lemma C.1.

For fixed 
𝜀
>
0
, 
LSE
𝜀
:
ℝ
𝑛
→
ℝ
 is 1-Lipschitz in 
∥
⋅
∥
∞
, and 
𝜎
𝜀
:
ℝ
→
(
0
,
1
)
 is 
(
4
​
𝜀
)
−
1
-Lipschitz.

Proof.

By the mean value theorem, 
|
LSE
𝜀
​
(
𝐱
)
−
LSE
𝜀
​
(
𝐲
)
|
≤
sup
𝐳
|
⟨
∇
LSE
𝜀
​
(
𝐳
)
,
𝐱
−
𝐲
⟩
|
. Since 
∇
LSE
𝜀
=
softmax
(
⋅
/
𝜀
)
 has 
‖
∇
LSE
𝜀
‖
1
=
1
 (Lemma 3.2), Hölder’s inequality gives 
|
⟨
∇
LSE
𝜀
,
𝐱
−
𝐲
⟩
|
≤
‖
∇
LSE
𝜀
‖
1
⋅
‖
𝐱
−
𝐲
‖
∞
=
‖
𝐱
−
𝐲
‖
∞
. The sigmoid bound follows from 
|
𝜎
𝜀
′
​
(
𝑥
)
|
≤
1
/
(
4
​
𝜀
)
 (Proposition 5.4). ∎

C.3Tropical Geometry Analysis

Theorem 5.7 establishes that UltraLIF dynamics converge to piecewise-linear maps on the max-plus semiring, with decision boundaries approaching tropical hypersurfaces. Three concrete extensions exploiting this connection are developed below, providing explicit expressivity bounds, temporal dynamics analysis, and capacity results.

C.3.1Tropical Characterization of Decision Boundaries

An explicit geometric characterization of UltraLIF decision boundaries in the tropical limit is provided.

Setup.

Consider a single-hidden-layer UltraLIF network with 
ℎ
 neurons, 
𝑛
 inputs, and 
𝐶
 output classes at 
𝑇
=
1
 in the tropical limit (
𝜀
→
0
+
). Starting from 
𝑉
(
0
)
=
𝟎
, each hidden neuron 
𝑗
∈
[
ℎ
]
 computes:

	
𝑉
𝑗
(
1
)
=
max
⁡
(
log
⁡
𝜏
0
,
𝐰
𝑗
⊤
​
𝐱
)
		
(26)

where 
𝐰
𝑗
∈
ℝ
𝑛
 is the weight vector for neuron 
𝑗
.

Definition C.2 (Hyperplane Arrangement).

Each neuron 
𝑗
 defines a hyperplane 
ℋ
𝑗
:=
{
𝐱
∈
ℝ
𝑛
:
𝐰
𝑗
⊤
​
𝐱
=
log
⁡
𝜏
0
}
. The collection 
𝒜
=
{
ℋ
1
,
…
,
ℋ
ℎ
}
 partitions 
ℝ
𝑛
 into connected regions where the spike pattern 
𝐬
∈
{
0
,
1
}
ℎ
 is constant.

Proposition C.3 (Hyperplane Arrangement Bound).

A single-hidden-layer UltraLIF partitions 
ℝ
𝑛
 into at most 
𝑅
​
(
ℎ
,
𝑛
)
:=
∑
𝑘
=
0
min
⁡
(
𝑛
,
ℎ
)
(
ℎ
𝑘
)
 linear regions. Within each region, the network output is constant.

Proof.

Each neuron defines a half-space 
ℋ
𝑗
+
=
{
𝐱
:
𝐰
𝑗
⊤
​
𝐱
>
log
⁡
𝜏
0
}
 where 
𝑠
𝑗
=
1
. The spike pattern is determined by membership in intersections of these half-spaces. The number of regions created by 
ℎ
 hyperplanes in 
ℝ
𝑛
 in general position (Theorem C.6) is (Zaslavsky, 1975):

	
𝑟
𝑛
​
(
𝒜
)
=
∑
𝑘
=
0
𝑛
(
ℎ
𝑘
)
	

When 
ℎ
<
𝑛
, terms with 
𝑘
>
ℎ
 vanish since 
(
ℎ
𝑘
)
=
0
 for 
𝑘
>
ℎ
, yielding the equivalent formula 
∑
𝑘
=
0
min
⁡
(
𝑛
,
ℎ
)
(
ℎ
𝑘
)
. Within each region, 
𝐬
 is constant, so the output 
𝑦
^
𝑐
=
∑
𝑗
:
𝑠
𝑗
=
1
𝑊
𝑐
​
𝑗
out
 is constant. ∎

Tropical Hypersurface Structure.

The decision boundary 
ℬ
𝑖
​
𝑗
=
{
𝐱
:
𝑦
^
𝑖
​
(
𝐱
)
=
𝑦
^
𝑗
​
(
𝐱
)
}
 is a union of 
(
𝑛
−
1
)
-faces of the arrangement where class scores are equal. This realizes Theorem 5.7’s connection: 
ℬ
𝑖
​
𝑗
 is a tropical hypersurface in the sense that it is the locus where two piecewise-linear functions achieve equality.

C.3.2Temporal Expressivity Amplification
Proposition C.4 (Exponential Growth).

A single-hidden-layer UltraLIF unrolled for 
𝑇
 timesteps partitions 
ℝ
𝑛
 into at most 
𝑅
​
(
ℎ
,
𝑛
)
𝑇
 regions.

Proof.

Let 
𝜙
(
𝑡
)
:
ℝ
𝑛
→
ℝ
𝑛
 denote the map from input to hidden layer membrane potentials at timestep 
𝑡
. Each 
𝜙
(
𝑡
)
 is piecewise-linear with at most 
𝑅
​
(
ℎ
,
𝑛
)
 regions by Proposition C.3. The 
𝑇
-step computation is the composition 
𝜙
(
𝑇
)
∘
⋯
∘
𝜙
(
1
)
.

By induction on 
𝑇
, the composition has at most 
𝑅
​
(
ℎ
,
𝑛
)
𝑇
 linear regions. Base case (
𝑇
=
1
): immediate from Proposition C.3. Inductive step: assume the claim holds for 
𝑇
−
1
. Then 
𝜙
(
𝑇
−
1
)
∘
⋯
∘
𝜙
(
1
)
 has at most 
𝑅
​
(
ℎ
,
𝑛
)
𝑇
−
1
 regions. Composing with 
𝜙
(
𝑇
)
 (which has 
𝑅
​
(
ℎ
,
𝑛
)
 regions) yields at most 
𝑅
​
(
ℎ
,
𝑛
)
𝑇
−
1
⋅
𝑅
​
(
ℎ
,
𝑛
)
=
𝑅
​
(
ℎ
,
𝑛
)
𝑇
 regions, since each region of the 
(
𝑇
−
1
)
-step map can be subdivided by the 
𝑅
​
(
ℎ
,
𝑛
)
 hyperplanes of 
𝜙
(
𝑇
)
. ∎

T=1 vs T
≥
10 Analysis.

At 
𝑇
=
1
, expressivity is 
𝑅
​
(
ℎ
,
𝑛
)
≈
10
19
 for typical settings, far exceeding dataset size. The advantage comes from gradient quality: UltraLIF’s forward-backward consistency avoids spurious local minima induced by surrogate gradient mismatch.

At 
𝑇
≥
10
, expressivity grows to 
𝑅
​
(
ℎ
,
𝑛
)
10
. Output averaging 
𝑦
^
=
1
𝑇
​
∑
𝑡
𝑊
out
​
𝐬
(
𝑡
)
 smooths individual spike errors. Gradient mismatch becomes less critical as errors cancel across timesteps, explaining why baselines recover (SHD: UltraLIF +11.22% at 
𝑇
=
1
 but 
−
2.16
%
 at 
𝑇
=
30
).

C.3.3Zonotope Volume and Expressivity
Definition C.5 (Zonotope).

The zonotope generated by 
𝐰
1
,
…
,
𝐰
ℎ
∈
ℝ
𝑛
 is 
𝒵
​
(
𝐰
1
,
…
,
𝐰
ℎ
)
:=
{
∑
𝑖
=
1
ℎ
𝜆
𝑖
​
𝐰
𝑖
:
𝜆
𝑖
∈
[
0
,
1
]
}
.

The volume 
vol
𝑛
​
(
𝒵
)
 quantifies geometric diversity of weight directions. When 
vol
𝑛
​
(
𝒵
)
=
0
, weights are linearly dependent and the arrangement degenerates.

Theorem C.6 (Volume-Expressivity Connection).

Let 
𝑊
=
[
𝐰
1
,
…
,
𝐰
ℎ
]
⊤
∈
ℝ
ℎ
×
𝑛
 be the weight matrix. The hyperplane arrangement achieves the maximal region count 
𝑅
​
(
ℎ
,
𝑛
)
=
∑
𝑘
(
ℎ
𝑘
)
 if and only if the hyperplanes are in general position. General position holds if and only if:

1. 

For any 
𝑛
+
1
 weight vectors 
𝐰
𝑖
1
,
…
,
𝐰
𝑖
𝑛
+
1
, no point 
𝐱
 satisfies all 
𝑛
+
1
 hyperplane equations simultaneously.

2. 

Any subset of 
𝑛
 weight vectors 
{
𝐰
𝑖
1
,
…
,
𝐰
𝑖
𝑛
}
 with 
𝑛
≤
min
⁡
(
ℎ
,
dim
(
ℝ
𝑛
)
)
 has 
det
(
𝐰
𝑖
1
,
…
,
𝐰
𝑖
𝑛
)
≠
0
 when 
𝑛
=
dim
(
ℝ
𝑛
)
.

Moreover, if 
ℎ
≥
𝑛
 and the weight matrix 
𝑊
 has rank 
𝑛
, then 
vol
𝑛
​
(
𝒵
​
(
𝐰
1
,
…
,
𝐰
ℎ
)
)
>
0
 implies the arrangement is non-degenerate with at least 
2
𝑛
 regions.

Proof.

The first statement follows from Zaslavsky’s characterization of general position (Zaslavsky, 1975): the arrangement achieves the maximal count when no 
𝑛
+
1
 hyperplanes meet at a point and all intersections are transverse. Condition (1) ensures no common intersection of 
𝑛
+
1
 hyperplanes. Condition (2) ensures transversality: any 
𝑛
 hyperplanes intersect at a unique point (when 
𝑛
 weight vectors are linearly independent) rather than a higher-dimensional face.

For the volume statement, suppose 
ℎ
≥
𝑛
 and 
rank
​
(
𝑊
)
=
𝑛
. Then there exist 
𝑛
 linearly independent weight vectors, say 
𝐰
𝑖
1
,
…
,
𝐰
𝑖
𝑛
. The zonotope contains the parallelepiped spanned by these vectors:

	
𝒫
=
{
∑
𝑗
=
1
𝑛
𝜆
𝑗
​
𝐰
𝑖
𝑗
:
𝜆
𝑗
∈
[
0
,
1
]
}
	

The volume satisfies 
vol
𝑛
​
(
𝒵
)
≥
vol
𝑛
​
(
𝒫
)
=
|
det
(
𝐰
𝑖
1
,
…
,
𝐰
𝑖
𝑛
)
|
>
0
 by linear independence.

These 
𝑛
 linearly independent hyperplanes partition 
ℝ
𝑛
 into at least 
2
𝑛
 regions (the 
2
𝑛
 orthants when hyperplanes pass through the origin, or their translated analogues). Thus 
𝑅
≥
2
𝑛
 when 
vol
𝑛
​
(
𝒵
)
>
0
.

Conversely, if 
vol
𝑛
​
(
𝒵
)
=
0
, the weight vectors lie in a proper subspace of dimension 
<
𝑛
, yielding a degenerate arrangement with 
𝑅
<
2
𝑛
 (at most linear in 
ℎ
). ∎

Corollary C.7 (Capacity Lower Bound).

For a single-hidden-layer UltraLIF with 
ℎ
≥
𝑛
 neurons and 
rank
​
(
𝑊
)
=
𝑛
:

	
Expressivity
≥
2
𝑛
if 
​
vol
𝑛
​
(
𝒵
​
(
𝐰
1
,
…
,
𝐰
ℎ
)
)
>
0
	
Implications.

Initialization. Standard random initialization (Kaiming, Xavier) samples from isotropic Gaussians, yielding near-orthogonal weights with high probability in high dimensions. This ensures 
vol
𝑛
​
(
𝒵
)
>
0
 and non-degenerate arrangements. Explicit orthogonalization (QR decomposition) maximizes volume for fixed norms.

Regularization. A volume-regularized loss 
ℒ
=
ℒ
CE
−
𝛼
​
log
⁡
vol
𝑛
​
(
𝒵
)
 encourages diverse weight directions, preventing rank collapse.

Pruning. If 
𝐰
𝑗
≈
𝑐
⋅
𝐰
𝑘
, removing neuron 
𝑗
 reduces 
vol
𝑛
​
(
𝒵
)
 negligibly, providing a principled pruning criterion based on geometric redundancy.

Appendix DExperimental Setup
D.1Datasets and Preprocessing

Evaluation is conducted on six benchmarks spanning static images, neuromorphic vision, and audio:

Static Image Datasets.

• 

MNIST: 
28
×
28
 grayscale handwritten digits, 10 classes. 60,000 train / 10,000 test samples. Normalization: mean 0.1307, std 0.3081.

• 

Fashion-MNIST: 
28
×
28
 grayscale fashion items, 10 classes. 60,000 train / 10,000 test samples. Normalization: mean 0.2860, std 0.3530.

• 

CIFAR-10: 
32
×
32
 RGB natural images, 10 classes. 50,000 train / 10,000 test samples. Normalization per channel: mean (0.4914, 0.4822, 0.4465), std (0.247, 0.243, 0.262). Training augmentation: random crop (32
×
32 with 4-pixel padding), random horizontal flip.

Neuromorphic Datasets.

• 

N-MNIST: Neuromorphic MNIST captured with DVS camera. 60,000 train / 10,000 test samples. Input dimension: 
2
×
34
×
34
 (ON/OFF polarity channels).

• 

DVS-Gesture: 11 hand gesture classes recorded with DVS128 camera. 1,176 train / 288 test samples. Input dimension: 
2
×
128
×
128
.

Audio Dataset.

• 

SHD: Spiking Heidelberg Digits, 20 spoken digit classes. 8,156 train / 2,264 test samples. Input dimension: 700 frequency channels.

Temporal Encoding. For static datasets, rate coding converts pixel intensities to spike trains:

	
𝑃
​
(
spike at 
​
𝑡
)
=
gain
⋅
𝑥
pixel
,
gain
=
0.5
		
(27)

where 
𝑥
pixel
∈
[
0
,
1
]
 is the normalized pixel value. For neuromorphic datasets, events are binned into 
𝑇
 temporal frames using the Tonic library’s ToFrame transform with n_time_bins
=
𝑇
.

D.2Model Hyperparameters
Table 10:Hyperparameters for all neuron models.
Parameter	Value	Description
Common parameters

𝜃
 (threshold) 	0.5	Spike threshold

𝜏
0
 (leak) 	0.9	Membrane time constant

𝑉
reset
	0.0	Reset potential
Surrogate gradient (baselines)

𝛽
 (sharpness) 	10.0	Sigmoid surrogate steepness
AdaLIF specific

𝛽
adapt
	0.1	Threshold adaptation strength

𝜏
adapt
	0.9	Adaptation decay constant
DSpike specific

𝑏
0
 (init) 	4.0	Initial sharpness parameter
Ultradiscretized models

𝜀
0
 (init) 	1.0	Initial temperature

𝜀
 range 	[0.1, 20.0]	Clamped during training
D.3Architecture

Single hidden layer with 64 neurons. Input-to-hidden and hidden-to-output are fully connected layers. Timesteps 
𝑇
∈
{
1
,
10
,
30
}
 to evaluate performance across temporal regimes. Output is computed as mean spike rate over time:

	
𝑦
^
=
1
𝑇
​
∑
𝑡
=
1
𝑇
𝑊
out
⋅
𝑠
(
𝑡
)
		
(28)
D.4Baseline Model Definitions

All baselines use surrogate gradients: hard spike 
𝑠
=
𝐻
​
(
𝑣
−
𝜃
)
 in the forward pass, smooth gradient 
∂
𝑠
/
∂
𝑣
=
𝜎
′
​
(
(
𝑣
−
𝜃
)
​
𝛽
)
 in the backward pass.

LIF (Leaky Integrate-and-Fire):

	
𝑣
(
𝑡
+
1
)
	
=
𝜏
​
𝑣
(
𝑡
)
+
𝐼
(
𝑡
)
		
(29)

	
𝑠
(
𝑡
+
1
)
	
=
𝐻
​
(
𝑣
(
𝑡
+
1
)
−
𝜃
)
,
𝑣
←
𝑣
​
(
1
−
𝑠
)
		
(30)

PLIF (Parametric LIF, Fang et al., 2021):

	
𝑣
(
𝑡
+
1
)
	
=
𝜏
​
𝑣
(
𝑡
)
+
𝐼
(
𝑡
)
,
𝜏
=
𝜎
​
(
𝜏
param
)
​
 learnable
		
(31)

AdaLIF (Adaptive LIF, Bellec et al., 2018):

	
𝑣
(
𝑡
+
1
)
	
=
𝜏
​
𝑣
(
𝑡
)
+
𝐼
(
𝑡
)
		
(32)

	
𝜃
(
𝑡
+
1
)
	
=
𝜃
0
+
𝛽
adapt
⋅
𝑏
(
𝑡
)
		
(33)

	
𝑏
(
𝑡
+
1
)
	
=
𝜏
adapt
⋅
𝑏
(
𝑡
)
+
(
1
−
𝜏
adapt
)
⋅
𝑠
(
𝑡
)
		
(34)

where 
𝑏
 is the adaptation variable that increases after spikes.

FullPLIF (Fully Parametric LIF):

	
𝑣
(
𝑡
+
1
)
	
=
𝜏
​
𝑣
(
𝑡
)
+
𝐼
(
𝑡
)
		
(35)

	
𝜏
	
=
𝜎
​
(
𝜏
param
)
,
𝜃
=
𝜎
​
(
𝜃
param
)
		
(36)

Both 
𝜏
 and 
𝜃
 are learnable, constrained to 
(
0
,
1
)
 via sigmoid.

DSpike (Li et al., 2021):

	
𝑣
(
𝑡
+
1
)
	
=
𝜏
​
𝑣
(
𝑡
)
+
𝐼
(
𝑡
)
		
(37)

	
𝑠
(
𝑡
+
1
)
	
=
tanh
⁡
(
𝑘
​
(
𝑣
norm
−
0.5
)
)
+
tanh
⁡
(
𝑘
/
2
)
2
​
tanh
⁡
(
𝑘
/
2
)
		
(38)

where 
𝑣
norm
=
𝑣
/
(
2
​
𝜃
)
 normalizes membrane potential to 
[
0
,
1
]
, and 
𝑘
 is a learnable sharpness parameter (initialized to 4.0).

DSpike+: DSpike with learnable 
𝜏
=
𝜎
​
(
𝜏
param
)
.

D.5Proposed Methods

Ultradiscretized variants use soft spike in both forward and backward passes (no surrogate gradients):

UltraLIF (Temporal, 2-term LSE from LIF ODE):

	
𝑉
(
𝑡
+
1
)
	
=
LSE
𝜀
​
(
𝑉
(
𝑡
)
+
log
⁡
𝜏
0
,
𝐼
(
𝑡
)
)
		
(39)

	
𝑠
𝜀
(
𝑡
+
1
)
	
=
𝜎
​
(
(
𝑉
(
𝑡
+
1
)
−
𝜃
)
/
𝜀
)
		
(40)

UltraDLIF (Spatial, 3-term LSE from diffusion PDE):

	
𝑉
𝑖
(
𝑡
+
1
)
	
=
LSE
𝜀
​
(
𝑉
𝑖
−
1
(
𝑡
)
,
𝑉
𝑖
(
𝑡
)
,
𝑉
𝑖
+
1
(
𝑡
)
)
+
𝐼
𝑖
(
𝑡
)
		
(41)

	
𝑠
𝑖
,
𝜀
(
𝑡
+
1
)
	
=
𝜎
​
(
(
𝑉
𝑖
(
𝑡
+
1
)
−
𝜃
)
/
𝜀
)
		
(42)

UltraPLIF and UltraDPLIF add learnable 
𝜏
=
𝜎
​
(
𝜏
param
)
.

D.6Sparsity Penalty

To encourage energy efficiency, ultradiscretized variants support an optional sparsity penalty:

	
ℒ
=
ℒ
CE
+
𝜆
⋅
𝑠
¯
		
(43)

where 
𝑠
¯
 is the mean spike rate and 
𝜆
∈
{
0
,
0.01
,
0.1
}
. This enables explicit control over the accuracy-efficiency trade-off.

D.7Training Details
Table 11:Training configuration.
Parameter	Value
Optimizer	Adam
Learning rate	
10
−
3

Batch size	128
Epochs	100
LR scheduler	Cosine annealing
Weight init	PyTorch default (Kaiming)
Seed	42

Loss Function. Cross-entropy on mean spike rates:

	
ℒ
CE
=
−
∑
𝑐
𝑦
𝑐
​
log
⁡
(
softmax
​
(
𝑦
^
)
𝑐
)
		
(44)

Energy Estimation. SNN energy consumption is estimated using the standard synaptic operation (SOP) framework (Lemaire et al., 2023). In SNNs, spike-driven computation replaces multiply-accumulate (MAC) operations with accumulate-only (AC) operations, since pre-synaptic activations are binary:

	
𝐸
SNN
	
=
𝑇
⋅
𝑠
¯
⋅
𝑁
syn
⋅
𝐸
AC
		
(45)

	
𝐸
ANN
	
=
𝑁
syn
⋅
𝐸
MAC
		
(46)

where 
𝑇
 is the number of timesteps, 
𝑠
¯
 is the mean spike rate, 
𝑁
syn
 is the number of synaptic connections, 
𝐸
AC
≈
0.9
​
pJ
, and 
𝐸
MAC
≈
4.6
​
pJ
 at 45nm technology (Horowitz, 2014). Since all models in this work share the same architecture (and thus the same 
𝑁
syn
), the energy column in results tables reports the relative SOP count 
𝑇
⋅
𝑠
¯
, which is proportional to 
𝐸
SNN
 up to a constant factor. This enables direct energy comparison across models. Note that this estimate focuses on computational energy and does not account for memory access or data movement overhead, which can be significant in practice (Yan et al., 2024).

D.8Hardware and Compute

All experiments were conducted on NVIDIA T4 GPUs (16GB VRAM) using PyTorch 2.0+ with torch.compile for acceleration. Training time per model:

• 

MNIST/Fashion: 
∼
3 minutes (100 epochs)

• 

CIFAR-10: 
∼
8 minutes (100 epochs)

• 

N-MNIST: 
∼
15 minutes (100 epochs)

• 

DVS-Gesture: 
∼
20 minutes (100 epochs)

• 

SHD: 
∼
10 minutes (100 epochs)

Total compute for all experiments: approximately 50 GPU-hours across three T4 VMs.

Note on neuromorphic hardware. All experiments in this work were conducted on conventional GPUs. Deployment on dedicated neuromorphic hardware (e.g., Intel Loihi 2, IBM TrueNorth, SpiNNaker 2, BrainScaleS-2) has not yet been evaluated. Since the ultradiscretized spike function converges to hard thresholding as 
𝜀
→
0
 (Proposition 5.2), the trained models can in principle be mapped to neuromorphic substrates by quantizing the soft spike to binary at inference time. Benchmarking latency, energy consumption, and accuracy on neuromorphic platforms is an important direction for future work.

Appendix EAdditional Experiments
Figure 3:Epsilon ablation on MNIST (
𝑇
=
1
, 100 epochs). (a) Learned 
𝜀
 exhibits a characteristic U-shaped trajectory: initial drop from 1.0 to 
∼
0.42 (sharpening phase), followed by recovery to model-specific optima (0.66–1.08). This suggests the network first learns sharp discrimination, then softens for generalization. (b) Learned 
𝜀
 (dashed lines) consistently matches or exceeds all fixed values across models, validating the benefit of learnable temperature.
E.1Ablation Study: Learnable Temperature 
𝜀

A key design choice in ultradiscretized neurons is whether to fix the temperature parameter 
𝜀
 or learn it during training. This ablation study on MNIST at 
𝑇
=
1
 compares fixed values 
𝜀
∈
{
0.5
,
1.0
,
2.0
}
 against learned 
𝜀
 (initialized to 1.0) across all four ultradiscretized variants.

Table 12:Ablation: Effect of learnable 
𝜀
 on accuracy (%). Learned 
𝜀
 consistently achieves best or tied-best accuracy across all models.
Model	
𝜀
=
0.5
	
𝜀
=
1.0
	
𝜀
=
2.0
	Learned
Temporal (LIF ODE)
UltraLIF	96.56	96.26	95.71	96.79
UltraPLIF	96.61	96.48	96.26	96.63
Spatial (Diffusion PDE)
UltraDLIF	96.70	96.54	96.39	96.71
UltraDPLIF	96.70	96.54	96.39	96.71
Table 13:Ablation: Effect of learnable 
𝜀
 on spike rate. Learned 
𝜀
 achieves lowest spike rates for spatial models.
Model	
𝜀
=
0.5
	
𝜀
=
1.0
	
𝜀
=
2.0
	Learned
Temporal (LIF ODE)
UltraLIF	0.488	0.579	0.650	0.500
UltraPLIF	0.427	0.459	0.528	0.443
Spatial (Diffusion PDE)
UltraDLIF	0.417	0.412	0.423	0.393
UltraDPLIF	0.417	0.412	0.423	0.393

Key findings: (1) Learned 
𝜀
 provides consistent accuracy gains across all models, though margins are small on MNIST. (2) Smaller fixed 
𝜀
 (0.5) outperforms larger values (2.0), consistent with Lemma 3.2, as tighter approximation to the max yields better LIF emulation. (3) Learned 
𝜀
 converges to the range 0.66–1.08 (Table 14), automatically finding optimal soft-to-hard trade-offs. (4) Spatial models achieve lowest spike rates with learned 
𝜀
, suggesting the network learns to balance accuracy and efficiency.

Table 14:Final learned 
𝜀
 values after training (initialized at 1.0).
Model	Final 
𝜀

UltraLIF (temporal)	0.661
UltraPLIF (temporal)	0.900
UltraDLIF (spatial)	1.084
UltraDPLIF (spatial)	1.084

Table 15 reports full results on Fashion-MNIST. UltraPLIF (temporal) achieves the best accuracy at 
𝑇
=
1
, while baselines lead at higher timesteps.

Table 15:Test accuracy (%) on Fashion-MNIST. UltraPLIF (temporal) achieves best at 
𝑇
=
1
. Baselines lead at 
𝑇
≥
10
.
Model	
𝑇
=
1
	
𝑇
=
10
	
𝑇
=
30

LIF	82.45	86.06	86.70
PLIF	82.45	85.99	86.48
AdaLIF	82.45	86.12	86.59
FullPLIF	82.18	86.26	86.33
DSpike	82.67	86.24	86.42
DSpike+	82.67	86.03	86.76
Temporal (LIF ODE)
UltraLIF	81.79	85.76	86.65
UltraPLIF	83.02	86.03	86.59
Spatial (Diffusion PDE)
UltraDLIF	82.79	85.88	86.01
UltraDPLIF	82.79	85.69	85.92
E.2Full Sparsity Results

Tables 16–21 present full sparsity results for all ultradiscretized models across sparsity penalty values 
𝜆
∈
{
0
,
0.01
,
0.1
}
 and timesteps 
𝑇
∈
{
1
,
10
,
30
}
.

Table 16:Sparsity results on MNIST. Accuracy (%), spike rate, and relative SOP count (
𝑇
⋅
𝑠
¯
).
Model	
𝜆
	Acc	Spike	Energy

𝑇
=
1

UltraLIF	0	94.37	0.620	0.62
UltraLIF	0.01	94.55	0.608	0.61
UltraLIF	0.1	94.37	0.537	0.54
UltraPLIF	0	95.60	0.468	0.47
UltraPLIF	0.01	95.50	0.453	0.45
UltraPLIF	0.1	95.81	0.289	0.29
UltraDLIF	0	95.67	0.446	0.45
UltraDLIF	0.01	95.62	0.425	0.43
UltraDLIF	0.1	95.71	0.268	0.27
UltraDPLIF	0	95.67	0.446	0.45
UltraDPLIF	0.01	95.62	0.425	0.43
UltraDPLIF	0.1	95.71	0.268	0.27

𝑇
=
10

UltraLIF	0	97.14	0.519	5.19
UltraLIF	0.01	97.15	0.498	4.98
UltraLIF	0.1	97.23	0.328	3.28
UltraPLIF	0	97.30	0.492	4.92
UltraPLIF	0.01	97.28	0.466	4.66
UltraPLIF	0.1	97.37	0.242	2.42
UltraDLIF	0	97.35	0.479	4.79
UltraDLIF	0.01	97.56	0.448	4.48
UltraDLIF	0.1	97.19	0.254	2.54
UltraDPLIF	0	97.35	0.476	4.76
UltraDPLIF	0.01	97.37	0.444	4.44
UltraDPLIF	0.1	97.35	0.237	2.37

𝑇
=
30

UltraLIF	0	97.46	0.502	15.07
UltraLIF	0.01	97.41	0.475	14.26
UltraLIF	0.1	97.51	0.270	8.09
UltraPLIF	0	97.55	0.492	14.76
UltraPLIF	0.01	97.52	0.464	13.91
UltraPLIF	0.1	97.53	0.229	6.87
UltraDLIF	0	97.38	0.469	14.07
UltraDLIF	0.01	97.52	0.427	12.81
UltraDLIF	0.1	97.11	0.208	6.24
UltraDPLIF	0	97.40	0.481	14.43
UltraDPLIF	0.01	97.34	0.439	13.18
UltraDPLIF	0.1	96.99	0.209	6.27
Table 17:Sparsity results on Fashion-MNIST. Accuracy (%), spike rate, and relative SOP count (
𝑇
⋅
𝑠
¯
).
Model	
𝜆
	Acc	Spike	Energy

𝑇
=
1

UltraLIF	0	81.79	0.652	0.65
UltraLIF	0.01	82.06	0.629	0.63
UltraLIF	0.1	81.62	0.530	0.53
UltraPLIF	0	83.02	0.472	0.47
UltraPLIF	0.01	82.81	0.451	0.45
UltraPLIF	0.1	83.26	0.279	0.28
UltraDLIF	0	82.79	0.429	0.43
UltraDLIF	0.01	83.01	0.411	0.41
UltraDLIF	0.1	83.05	0.267	0.27
UltraDPLIF	0	82.79	0.429	0.43
UltraDPLIF	0.01	83.01	0.411	0.41
UltraDPLIF	0.1	83.05	0.267	0.27

𝑇
=
10

UltraLIF	0	85.76	0.522	5.22
UltraLIF	0.01	86.07	0.510	5.10
UltraLIF	0.1	85.98	0.334	3.34
UltraPLIF	0	86.03	0.493	4.93
UltraPLIF	0.01	85.93	0.457	4.57
UltraPLIF	0.1	86.11	0.273	2.73
UltraDLIF	0	85.88	0.456	4.56
UltraDLIF	0.01	85.63	0.442	4.42
UltraDLIF	0.1	85.74	0.292	2.92
UltraDPLIF	0	85.69	0.456	4.56
UltraDPLIF	0.01	85.86	0.427	4.27
UltraDPLIF	0.1	85.74	0.278	2.78

𝑇
=
30

UltraLIF	0	86.65	0.507	15.22
UltraLIF	0.01	86.46	0.485	14.56
UltraLIF	0.1	86.36	0.287	8.60
UltraPLIF	0	86.59	0.480	14.41
UltraPLIF	0.01	86.49	0.453	13.59
UltraPLIF	0.1	86.72	0.271	8.13
UltraDLIF	0	86.01	0.467	14.01
UltraDLIF	0.01	85.81	0.429	12.88
UltraDLIF	0.1	85.94	0.274	8.23
UltraDPLIF	0	85.92	0.463	13.88
UltraDPLIF	0.01	85.80	0.458	13.75
UltraDPLIF	0.1	85.84	0.276	8.27
Table 18:Sparsity results on CIFAR-10. Accuracy (%), spike rate, and relative SOP count (
𝑇
⋅
𝑠
¯
).
Model	
𝜆
	Acc	Spike	Energy

𝑇
=
1

UltraLIF	0	40.72	0.706	0.71
UltraLIF	0.01	40.58	0.662	0.66
UltraLIF	0.1	39.81	0.459	0.46
UltraPLIF	0	43.27	0.458	0.46
UltraPLIF	0.01	43.22	0.444	0.44
UltraPLIF	0.1	43.60	0.240	0.24
UltraDLIF	0	43.11	0.481	0.48
UltraDLIF	0.01	43.13	0.465	0.47
UltraDLIF	0.1	43.04	0.337	0.34
UltraDPLIF	0	43.11	0.481	0.48
UltraDPLIF	0.01	43.13	0.465	0.47
UltraDPLIF	0.1	43.04	0.337	0.34

𝑇
=
10

UltraLIF	0	45.15	0.504	5.04
UltraLIF	0.01	44.72	0.494	4.94
UltraLIF	0.1	45.18	0.311	3.11
UltraPLIF	0	46.19	0.494	4.94
UltraPLIF	0.01	46.06	0.457	4.57
UltraPLIF	0.1	46.13	0.241	2.41
UltraDLIF	0	45.65	0.471	4.71
UltraDLIF	0.01	45.58	0.452	4.52
UltraDLIF	0.1	45.39	0.334	3.34
UltraDPLIF	0	45.75	0.469	4.69
UltraDPLIF	0.01	46.26	0.452	4.52
UltraDPLIF	0.1	45.32	0.338	3.38

𝑇
=
30

UltraLIF	0	45.69	0.480	14.39
UltraLIF	0.01	45.84	0.466	13.99
UltraLIF	0.1	45.92	0.285	8.54
UltraPLIF	0	46.58	0.500	15.01
UltraPLIF	0.01	46.31	0.485	14.56
UltraPLIF	0.1	46.98	0.248	7.44
UltraDLIF	0	45.00	0.491	14.73
UltraDLIF	0.01	45.32	0.446	13.38
UltraDLIF	0.1	45.41	0.322	9.65
UltraDPLIF	0	45.74	0.496	14.89
UltraDPLIF	0.01	46.09	0.450	13.49
UltraDPLIF	0.1	45.79	0.331	9.94
Table 19:Sparsity results on N-MNIST. Accuracy (%), spike rate, and relative SOP count (
𝑇
⋅
𝑠
¯
).
Model	
𝜆
	Acc	Spike	Energy

𝑇
=
1

UltraLIF	0	90.41	0.579	0.58
UltraLIF	0.01	90.37	0.597	0.60
UltraLIF	0.1	90.74	0.546	0.55
UltraPLIF	0	93.11	0.396	0.40
UltraPLIF	0.01	92.73	0.409	0.41
UltraPLIF	0.1	93.28	0.318	0.32
UltraDLIF	0	94.14	0.506	0.51
UltraDLIF	0.01	93.97	0.492	0.49
UltraDLIF	0.1	93.52	0.291	0.29
UltraDPLIF	0	94.14	0.506	0.51
UltraDPLIF	0.01	93.97	0.492	0.49
UltraDPLIF	0.1	93.52	0.291	0.29

𝑇
=
10

UltraLIF	0	96.10	0.447	4.47
UltraLIF	0.01	96.22	0.415	4.15
UltraLIF	0.1	96.39	0.222	2.22
UltraPLIF	0	96.33	0.445	4.45
UltraPLIF	0.01	96.34	0.358	3.58
UltraPLIF	0.1	96.60	0.127	1.27
UltraDLIF	0	97.38	0.460	4.60
UltraDLIF	0.01	97.37	0.335	3.35
UltraDLIF	0.1	97.00	0.155	1.55
UltraDPLIF	0	97.38	0.463	4.63
UltraDPLIF	0.01	97.40	0.306	3.06
UltraDPLIF	0.1	96.93	0.144	1.44

𝑇
=
30

UltraLIF	0	95.87	0.404	12.13
UltraLIF	0.01	95.77	0.391	11.72
UltraLIF	0.1	96.30	0.267	8.02
UltraPLIF	0	95.77	0.463	13.90
UltraPLIF	0.01	95.91	0.412	12.36
UltraPLIF	0.1	96.48	0.240	7.21
UltraDLIF	0	97.46	0.429	12.86
UltraDLIF	0.01	97.34	0.301	9.04
UltraDLIF	0.1	97.28	0.140	4.20
UltraDPLIF	0	97.68	0.430	12.89
UltraDPLIF	0.01	97.52	0.275	8.26
UltraDPLIF	0.1	97.33	0.125	3.74
Table 20:Sparsity results on DVS-Gesture. Accuracy (%), spike rate, and relative SOP count (
𝑇
⋅
𝑠
¯
).
Model	
𝜆
	Acc	Spike	Energy

𝑇
=
1

UltraLIF	0	58.33	0.726	0.73
UltraLIF	0.01	57.58	0.746	0.75
UltraLIF	0.1	57.95	0.690	0.69
UltraPLIF	0	60.23	0.619	0.62
UltraPLIF	0.01	57.20	0.610	0.61
UltraPLIF	0.1	55.68	0.601	0.60
UltraDLIF	0	58.33	0.774	0.77
UltraDLIF	0.01	56.44	0.833	0.83
UltraDLIF	0.1	58.71	0.790	0.79
UltraDPLIF	0	58.33	0.774	0.77
UltraDPLIF	0.01	56.44	0.833	0.83
UltraDPLIF	0.1	58.71	0.790	0.79

𝑇
=
10

UltraLIF	0	69.32	0.707	7.07
UltraLIF	0.01	68.56	0.709	7.09
UltraLIF	0.1	67.80	0.702	7.02
UltraPLIF	0	68.94	0.559	5.59
UltraPLIF	0.01	69.70	0.567	5.67
UltraPLIF	0.1	70.83	0.543	5.43
UltraDLIF	0	69.32	0.611	6.11
UltraDLIF	0.01	71.97	0.601	6.01
UltraDLIF	0.1	70.45	0.543	5.43
UltraDPLIF	0	68.56	0.615	6.15
UltraDPLIF	0.01	71.97	0.614	6.14
UltraDPLIF	0.1	73.11	0.578	5.78

𝑇
=
30

UltraLIF	0	75.00	0.719	21.58
UltraLIF	0.01	75.38	0.719	21.56
UltraLIF	0.1	75.00	0.707	21.20
UltraPLIF	0	75.76	0.593	17.79
UltraPLIF	0.01	75.76	0.588	17.63
UltraPLIF	0.1	76.14	0.559	16.77
UltraDLIF	0	78.41	0.560	16.79
UltraDLIF	0.01	78.41	0.552	16.56
UltraDLIF	0.1	81.06	0.501	15.04
UltraDPLIF	0	79.92	0.570	17.09
UltraDPLIF	0.01	77.65	0.556	16.68
UltraDPLIF	0.1	79.55	0.521	15.62
Table 21:Sparsity results on SHD. Accuracy (%), spike rate, and relative SOP count (
𝑇
⋅
𝑠
¯
).
Model	
𝜆
	Acc	Spike	Energy

𝑇
=
1

UltraLIF	0	44.88	0.551	0.55
UltraLIF	0.01	45.76	0.542	0.54
UltraLIF	0.1	46.86	0.507	0.51
UltraPLIF	0	46.91	0.390	0.39
UltraPLIF	0.01	47.61	0.394	0.39
UltraPLIF	0.1	48.32	0.344	0.34
UltraDLIF	0	51.24	0.686	0.69
UltraDLIF	0.01	50.62	0.629	0.63
UltraDLIF	0.1	51.33	0.565	0.56
UltraDPLIF	0	51.24	0.686	0.69
UltraDPLIF	0.01	50.62	0.629	0.63
UltraDPLIF	0.1	51.33	0.565	0.56

𝑇
=
10

UltraLIF	0	58.79	0.472	4.72
UltraLIF	0.01	57.99	0.462	4.62
UltraLIF	0.1	60.51	0.397	3.97
UltraPLIF	0	57.73	0.420	4.20
UltraPLIF	0.01	58.48	0.411	4.11
UltraPLIF	0.1	58.79	0.348	3.48
UltraDLIF	0	67.62	0.416	4.16
UltraDLIF	0.01	69.52	0.442	4.42
UltraDLIF	0.1	69.92	0.367	3.67
UltraDPLIF	0	68.90	0.461	4.61
UltraDPLIF	0.01	65.77	0.496	4.96
UltraDPLIF	0.1	70.27	0.399	3.99

𝑇
=
30

UltraLIF	0	59.14	0.458	13.73
UltraLIF	0.01	59.41	0.451	13.52
UltraLIF	0.1	61.00	0.404	12.11
UltraPLIF	0	59.45	0.449	13.48
UltraPLIF	0.01	60.16	0.445	13.35
UltraPLIF	0.1	60.60	0.385	11.54
UltraDLIF	0	71.60	0.459	13.76
UltraDLIF	0.01	70.23	0.458	13.73
UltraDLIF	0.1	73.19	0.386	11.57
UltraDPLIF	0	67.84	0.454	13.62
UltraDPLIF	0.01	67.71	0.449	13.48
UltraDPLIF	0.1	69.88	0.404	12.13

Key observations across datasets: (1) Moderate sparsity (
𝜆
=
0.1
) consistently reduces spike rates by 40–50% with minimal accuracy loss, and in several cases (MNIST 
𝑇
=
1
, Fashion 
𝑇
=
1
, CIFAR-10 
𝑇
=
30
) actually improves accuracy, suggesting that sparsity acts as a regularizer. (2) The sparsity-accuracy trade-off is most favorable on temporal models (UltraPLIF), which achieve the lowest energy at competitive accuracy.

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

