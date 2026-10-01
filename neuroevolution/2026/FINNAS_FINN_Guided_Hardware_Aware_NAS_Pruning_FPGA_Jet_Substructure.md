# 2609.16367

Source: https://arxiv.org/html/2609.16367



FINNAS: FINN-Guided Hardware-Aware NAS and Pruning for FPGA Jet Substructure Classification

Report GitHub Issue
×

Title:

Content selection saved. Describe the issue below:
Description:

Submit without GitHub
Submit in GitHub

arXiv is now an independent nonprofit!
Learn more
×

Back to arXiv

Why HTML?

Report Issue

Back to Abstract

Download PDF

Abstract
I Introduction
II Methodology and Design

II-A Hardware-Aware Evolutionary Search
II-B Finalist Training and Pruning

III Experimental Results

III-A Experimental Setup
III-B Comparison with Prior Work

IV Conclusion
References

 License: arXiv.org perpetual non-exclusive license
 

arXiv:2609.16367v1 [cs.AR] 14 Sep 2026

FINNAS: FINN-Guided Hardware-Aware NAS and Pruning for FPGA Jet Substructure Classification
PubID: pubid: 979-8-3195-1905-4/26/$31.00 © 2026 IEEE

Eva Chauffour

  
Changhong Li

  
Georgios Floros

  
Shreejith Shanker

Affiliation: Reconfigurable Computing Systems Lab, Electronic & Electrical Engineering

Affiliation: Trinity College Dublin, Ireland

Affiliation: Email: {chauffoe, lic9, florosg, shreejith.shanker}@tcd.ie

Abstract
FPGAs are well suited to deploying quantised neural networks (QNNs)
under strict accuracy, latency, and resource constraints; however, identifying efficient model-accelerator combinations commonly requires extensive manual design-space exploration and repeated hardware synthesis. This paper presents FINNAS, a FINN-guided hardware-aware evolutionary neural architecture search framework. FINNAS jointly searches quantised MLP depth, width, and global precision settings, and ranks candidates using proxy validation accuracy together with FINN-estimated LUT usage and latency under a fully parallel mapping. Selected finalists are fully retrained, subjected to post-search unstructured pruning, and validated using RTL simulation and Vivado out-of-context synthesis. On the CERNBox jet substructure classification task, the searched implementations expose competitive accuracy-resource trade-offs. Compared with a manually optimised dense FINN accelerator, a compact FINNAS design improves accuracy from 73.78% to 74.36%, while reducing LUT usage by 8.5×8.5\times and RTL-simulation latency by 1.77×1.77\times. Unstructured pruning further provides consistent LUT and FF reductions across the fully parallel finalists.

Index Terms: Accelerator, Unstructured Sparsity, Field Programmable Gate Arrays, Quantised Neural Nets, Neural Architecture Search

I Introduction

The convergence of deep learning and edge computing has created a strong demand for low-latency neural network inference on specialised hardware.
In latency-critical scientific applications, inference must often be completed within submicroseconds while satisfying resource constraints.
High-energy physics experiments are among the most representative scenarios [1], where jet substructure classification (JSC) is a widely used benchmark for real-time FPGA inference.
Specifically, the HLF CERNBox JSC formulation represents each event using 16 high-level input features and is commonly evaluated using compact fully connected classifiers.

Several FPGA-oriented implementations have been developed for JSC, including LogicNets [2], PolyLUT [3], NeuraLUT [4], NeuraLUT-Assemble [5], and AmigoLUT [6], which map neurons directly to logical lookup-table operations to exploit FPGA primitives.
These approaches can achieve high classification accuracy with very limited resource utilisation while meeting the real-time requirements of the task.
However, these accelerators typically rely on specialised mappings and relatively complex implementation processes.

More general FPGA QNN deployment frameworks like FINN [7] provide an end-to-end solution for neural network acceleration on FPGAs, including QNN development, accelerator DSE, and automatic generation of bitstreams and drivers.
It abstracts away the complexity of low-level hardware deployment from deep learning developers and reduces the design effort.
However, FINN’s conventional DSE mainly supports heuristic folding exploration for predefined dense QNNs under a targeted throughput, limiting exploration of the model architecture itself.
Complementary efforts improve deployment-side feedback: an empirical quality-of-results (QoR) flow enables faster estimation of dataflow accelerators [8], while sparsity has been integrated into its accelerator generation [9].
These works further extend the FINN ecosystem towards more efficient accelerator exploration and deployment.

NAS is increasingly used for design-space exploration of FPGA-accelerated neural networks, where designers can exploit bespoke numeric representations and precisions, in addition to architecture-level optimisations.
NASB [10] explores NAS-based architectural optimisation for binary neural networks by searching hardware-suitable binary convolutional cells.
Building on this line of work, NASH [11] extends NAS toward FPGA-oriented QNN deployment and integrates the resulting models into the FINN flow for hardware generation and evaluation.
However, NASH mainly searches network architecture under predefined bit-width settings, while FINN-derived resource and latency costs are evaluated after architecture search rather than incorporated directly into the NAS objective. This leaves scope for search methods that jointly explore architecture and quantisation while using FPGA-oriented cost estimates to guide candidate selection.

To address these gaps, we propose FINNAS, which incorporates proxy validation accuracy and FINN-estimated LUT usage and latency directly into an evolutionary search over quantised MLP topology and precision settings, followed by post-search unstructured pruning (USP).
The main contributions of this paper are listed as follows:

•

A hardware-aware evolutionary NAS integrated with the end-to-end FINN flow, jointly considering predictive accuracy, estimated LUT usage, and latency during search.

•

An expanded search space over MLP topology and global weight and activation precisions, combined with post-search USP.

•

On the JSC task, FINNAS achieves higher accuracy and lower latency with up to an 8.5×\times LUT usage reduction relative to the hand-optimised dense FINN baseline.

Fig. 1: FINNAS workflow compared with conventional FPGA design-space exploration.

II Methodology and Design

Fig. 1 illustrates the overall workflow of the proposed FINNAS framework and compares it with conventional FPGA design-space exploration.
Designing QNN accelerators requires architectural and quantisation choices to be weighted against hardware objectives which, unlike software-only metrics, strongly depend on how a network is mapped to the target FPGA.
These hardware characteristics are most accurately evaluated through synthesis, but conventional design approaches make this expensive by requiring numerous model variations to be assessed before a suitable trade-off is found.

FINNAS reduces this exploration cost by adopting an evolutionary search algorithm to explore network architectures and quantisation bit-widths, while incorporating FINN-derived hardware estimates into candidate evaluation. This allows us to guide the search towards the most competitive design regions without synthesising every architecture. Selected finalists are subsequently fully retrained, pruned, fine-tuned, and evaluated through RTL simulation and OOC synthesis.

TABLE I: JSC model search space and evolutionary-search settings.

Parameter
Notation
Setting

Hidden depth
LL
{2,3,4,5,6}\{2,3,4,5,6\}

Hidden width
hih_{i}

{8,12,16,24,32,48,64,\{8,12,16,24,32,48,64,

96,128,160,192,256,320}96,128,160,192,256,320\}

Weight bits
BwB_{w}
{2,3,4,8}\{2,3,4,8\}

Input bits
Bi​aB_{ia}
{3,4,8}\{3,4,8\}

Hidden bits
Bh​aB_{ha}
{3,4}\{3,4\}

Output bits
Bo​aB_{oa}
{3,4,7,8}\{3,4,7,8\}

Population size
NpopN_{\mathrm{pop}}
30

Generations
NgenN_{\mathrm{gen}}
20

Elites per gen.
NeliteN_{\mathrm{elite}}
1

Tournament size
ktourk_{\mathrm{tour}}
2

Random candidates
NrandN_{\mathrm{rand}}
4

Crossover prob.
pcp_{c}
0.3

Mutation prob.
pmp_{m}
0.9

LUT budget
BlB_{\mathrm{l}}
run-dependent

Latency budget
BtB_{\mathrm{t}}
15 ns

Maximum LUT
LmaxL_{\mathrm{max}}
250,000

LUT exponent
βl\beta_{\mathrm{l}}
0.50

Latency exponent
βt\beta_{\mathrm{t}}
0.05

II-A Hardware-Aware Evolutionary Search

For the exploration of candidate architectures, we adopt an evolutionary algorithm, in contrast to the differentiable NAS strategy used in NASH [11].
This is particularly suitable for the discrete and constrained architectural and quantisation variables considered here, with candidate fitness obtained from proxy training and FINN hardware estimate reports.
The search parameters used in this work are listed in Table I.

Each individual is encoded as Equation 1,
i.e., an ordered list of hidden-layer widths and four global precision fields. At initialisation, LL, every hih_{i}, and each precision field are drawn from their corresponding finite sets in Table I.
Samples are repaired until they satisfy Bo​a≥Bh​aB_{oa}\geq B_{ha} and Bw<8B_{w}<8 when all activation precisions are below 8 bits.
Each candidate then undergoes 16 epochs of proxy training using cross-entropy loss and the AdamW optimiser to obtain validation accuracy A⁡(x)A(x).

x={[h1,…,hL],(Bw,Bi​a,Bh​a,Bo​a)}x=\{[h_{1},\ldots,h_{L}],(B_{w},B_{ia},B_{ha},B_{oa})\}

(1)

The candidates are then exported to QONNX and evaluated using the FINN performance estimator to obtain estimated LUT usage L⁡(x)L(x) and latency T⁡(x)T(x), without requiring out-of-context (OOC) synthesis or RTL simulation.
For every candidate, FINNAS applies the same fully parallel mapping by setting SIMD=Nin\mathrm{SIMD}=N_{\mathrm{in}} and PE=Nout\mathrm{PE}=N_{\mathrm{out}} for each fully connected layer; this mapping is retained during final hardware evaluation.

The estimated accuracy and hardware metrics are combined through the reward R⁡(x)R(x) defined in Equation 2.

R⁡(x)=A⁡(x)⋅max⁡(1,L⁡(x)Bl)−βl⋅max⁡(1,T⁡(x)Bt)−βtR(x)=A(x)\cdot\max\left(1,\frac{L(x)}{B_{\mathrm{l}}}\right)^{-\beta_{\mathrm{l}}}\cdot\max\left(1,\frac{T(x)}{B_{\mathrm{t}}}\right)^{-\beta_{\mathrm{t}}}

(2)

In this equation, BlB_{\mathrm{l}} and BtB_{\mathrm{t}} denote the LUT and latency budgets, respectively, while βl\beta_{\mathrm{l}} and βt\beta_{\mathrm{t}} control the penalty strength when an architecture exceeds either budget.
Candidates satisfying a budget receive no additional reward for further reducing that metric, allowing accuracy to drive selection within the feasible region.
The algorithm minimises F⁡(x)=−R⁡(x)+P⁡(x)F(x)=-R(x)+P(x), where P⁡(x)=0P(x)=0 for L⁡(x)≤LmaxL(x)\leq L_{\mathrm{max}} and P⁡(x)=1+10​(L⁡(x)/Lmax−1)2P(x)=1+10(L(x)/L_{\mathrm{max}}-1)^{2} otherwise, ensuring that oversized candidates remain rankable according to their degree of constraint violation.

Within each generation, candidates are ranked according to F⁡(x)F(x) after evaluation; NeliteN_{\mathrm{elite}} elites are retained, and the remaining parents are chosen by tournament selection of size ktourk_{\mathrm{tour}}.
Crossover exchanges hidden-layer suffixes and quantisation fields between parent pairs, while mutation can modify a hidden width, insert or remove a layer, or resample quantisation parameters.
After these operations, a repair step maps each offspring back to the search space by enforcing 2≤L≤62\leq L\leq 6, restricting every hih_{i} and bit-width to the sets in Table I, and re-enforcing the two cross-field quantisation constraints used at initialisation.
NrandN_{\mathrm{rand}} randomly generated candidates are then injected, duplicates are removed, and previously evaluated architectures reuse their cached fitness, overall
increasing exploration and reducing the search cost.
After NgenN_{\mathrm{gen}} generations, the five highest-ranked architectures in the archive are selected as finalists.

II-B Finalist Training and Pruning

The selected architectures are retrained with a maximum allowance of 500 epochs and early stopping with a patience of 50 epochs, and first evaluated as dense baselines. The large cap accommodates different convergence rates across the searched topologies, while early stopping limits unnecessary training once validation performance plateaus.
Post-hoc unstructured global magnitude pruning is then applied to their linear layers at different sparsity targets without changing the network tensor shapes.
For each target, a global threshold is determined from the absolute weight values of all eligible weight tensors, with weights below this threshold set to zero.
Subsequently, the pruned networks are fine-tuned for up to 100 epochs when their raw accuracy drops by more than a specified ϵ\epsilon, with binary masks enforcing the pruned weights throughout fine-tuning. Test accuracy is recorded on the held-out test set for each dense and pruned finalist.

III Experimental Results

III-A Experimental Setup

The model training and accelerator compilation are performed on a workstation fitted with an NVIDIA RTX 5070 Ti GPU and an AMD Ryzen 5 9600X CPU, using CUDA 13.0, Python 3.12 and FINN Docker.
All resource and latency reports are obtained from OOC synthesis and RTL simulation reports in Vivado 2022.2 targeting a Virtex UltraScale+ FPGA (xcvu9p-flgb2104-2-i) with the Flow_PerfOptimized_high synthesis strategy.
The VU9P target matches the device used by most prior JSC implementations considered here; HGQ [12] instead targets the larger VU13P.

Across the reported searches, the LUT budget was varied as Bl∈{70,100,150,160}B_{\mathrm{l}}\in\{70,100,150,160\}k LUTs to target different points along the accuracy-resource trade-off, while all other search settings were fixed.
A single NAS run evaluates 500 candidates in around six hours using proxy training and FINN estimates. For context, OOC synthesis of those same 500 sampled candidates would require at least 40 serial hours even at the shortest observed runtime (∼\sim5 min/design).

To validate the search-time hardware proxy, FINN estimates were compared against OOC synthesis and RTL-simulation results for all 20 dense finalists synthesised from the reported searches. Estimated and synthesised LUT usage exhibit a strong Spearman rank correlation of ρ=0.854\rho=0.854, while estimated and RTL cycle latency achieve ρ=0.781\rho=0.781.
The estimates show substantial absolute error, with LUT usage systematically overestimated and latency represented only crudely, but we found them to preserve sufficient relative ordering to be able to guide candidate ranking during search.

III-B Comparison with Prior Work

TABLE II: Comparison of FINNAS with prior HLF JSC FPGA implementations and FINN baselines

Implementation
 

Acc

(%)

LUT
DSP
FF
 

Fmax

(MHz)

 

Latency

(ns)

II

HGQ [12]
75.3
10,921
0
11,183
578.4
31.1
1

HGQ [12]
75.1
5,974
0
5,775
609.8
24.6
1

AmigoLUT [6]
74.4
42,742
0
4,717
520
9.6
1

AmigoLUT [6]
72.9
1,243
0
1,240
1008
5.0
1

ReducedLUT [13]
74.9
58,409
0
N/A
302.8
–
–

ReducedLUT [13]
72.5
2,786
0
N/A
408.5
–
–

NeuraLUT-Asm. [5]
75.0
8,539
0
1,332
352
5.7
1

NeuraLUT-Asm. [5]
75.0
8,535
0
2,717
994
7.0
1

QKeras [14]
74.8
39,782
124
8,128
∼\sim200
55.0
1

QKeras [14]
72.3
9,149
66
1,781
∼\sim200
55.0
1

PolyLUT [3]
75.1
246,071
0
12,384
203
24.6
1

PolyLUT-Add [15]
75.0
36,484
0
1,209
315
15.9
1

NeuraLUT [4]
75.0
92,357
0
4,885
368
13.6
1

LogicNets [2]
71.8
37,931
0
810
427
11.7
1

FINN [7] Dense
73.78
57,893
0
82,750
417.54
110.17
1

FINN [7] 30%
73.72
46,735
0
67,252
438.21
104.97
1

FINN [7] 50%
73.63
37,386
0
54,415
463.18
95.0
1

FINN [7] 70%
73.25
10,102
0
14,684
678.43
58.96
1

A [32,24] (30%)
75.03
21,409
0
10,621
451.4
71.1
1

B [16,24,12] (30%)
74.78
13,829
0
12,793
500.8
90.0
1

C [32,32] (50%)
74.53
16,233
0
11,033
466.8
51.1
1

D [16,16] (30%)
74.54
8,125
0
7,417
505.3
66.0
1

E [24,8] (30%)
74.36
6,806
0
4,796
462.5
62.2
1

Fig. 2: Accuracy-LUT trade-off for HLF JSC FPGA implementations. Selected FINNAS finalists are labelled A-E; additional NAS candidates and pruned FINN baseline variants are also shown.

To evaluate the effectiveness of FINNAS, we conduct a case study on the HLF JSC CERNBox task and compare the searched designs with three groups of baselines: (a) previously reported FPGA accelerators for the same task, (b) a dense FINN implementation using a representative architecture from [14], and (c) its sparse FINN variants. The prior-work baselines use non-FINN, method-specific accelerator flows, many employing specialised LUT-based mappings, whereas the FINN baseline and Designs A–E are generated through the general FINN dataflow toolchain. The results are summarised in Table II.

Designs A–E were selected from searches with Bl={160,150,100,70,100}B_{\mathrm{l}}=\{160,150,100,70,100\}k LUTs, respectively, using q=(Bw,Bi​a,Bh​a,Bo​a)q=(B_{w},B_{ia},B_{ha},B_{oa}), with qA=qB=(8,8,4,7)q_{A}=q_{B}=(8,8,4,7), qC=(4,8,4,7)q_{C}=(4,8,4,7), qD=(8,8,4,4)q_{D}=(8,8,4,4), and qE=(8,8,3,4)q_{E}=(8,8,3,4). Their hidden-layer topologies and selected USP levels are listed in Table II.
Among the proposed implementations, Design A achieves an accuracy of 75.03% using 21.4k LUTs, placing it in the same accuracy range as several specialised LUT-oriented implementations.
Designs D and E represent the more compact end of the searched design space, maintaining over 74% accuracy with only 8.1k and 6.8k LUTs, respectively.
Design C achieves the lowest latency among all synthesised designs exceeding 74% accuracy, while Design B is the highest-accuracy selected design that satisfies the 2 ns timing constraint.

TABLE III: Dense-to-selected-pruned results for Designs A–E.

Acc. (%)
LUT
FF
 

Fmax

(MHz)

A (30%)
75.05→\rightarrow75.03
26,667→\rightarrow21,409
14,549→\rightarrow10,621
395.7→\rightarrow451.4

B (30%)
74.84→\rightarrow74.78
16,640→\rightarrow13,829
15,850→\rightarrow12,793
496.5→\rightarrow500.8

C (50%)
74.42→\rightarrow74.53
24,637→\rightarrow16,233
13,009→\rightarrow11,033
461.8→\rightarrow466.8

D (30%)
74.52→\rightarrow74.54
10,360→\rightarrow8,125
9,997→\rightarrow7,417
497.5→\rightarrow505.3

E (30%)
74.48→\rightarrow74.36
8,235→\rightarrow6,806
6,224→\rightarrow4,796
363.9→\rightarrow462.5

Under the fully parallel mapping used here, USP produces hardware savings because embedded weights become compile-time constants, allowing zero-valued weights to be propagated and corresponding combinational logic removed during synthesis.
In folded mappings, weights instead share time-multiplexed datapaths, so individual zeros do not necessarily remove hardware resources [7].
Across all synthesised finalists evaluated in the pruning study, 30%, 50%, and 70% USP reduce LUT usage by an average of 14.7%, 28.6%, and 39.8%, respectively, and FF usage by 17.2%, 26.9%, and 41.4%.
This generally comes with a gradual accuracy trade-off for most larger MLPs, while some compact models maintain or slightly improve accuracy.
For the five representative designs A–E, the corresponding dense and selected-pruned results are shown in Table III.
Pruning also improves Fmax for all five reported designs, enabling timing closure for A, B, D, and E.

Fig. 2 places the reported implementations in the broader accuracy–LUT design space.
Additional NAS-discovered candidates are also plotted to illustrate the range of trade-offs exposed across the searches.
Relative to the FINN-based designs, FINNAS expands the attainable trade-off by jointly searching network topology and quantisation under hardware-aware resource and latency objectives.
In particular, compared with the dense FINN baseline, Design E reduces LUT usage by 8.5×8.5\times and latency by 1.7×1.7\times, while improving accuracy from 73.78% to 74.36%.

Compared with specialised FPGA implementations reported in prior work, FINNAS maintains competitive accuracy at moderate and low LUT usage despite relying on the general-purpose FINN dataflow architecture.
The latency of our designs is not as low as some LUT network implementations that heavily exploit FPGA primitives.
This is mainly because the automated FINN flow preserves the layer-level structure of deep learning models and inserts inter-layer FIFOs to support a more general end-to-end deployment process across models.
Nevertheless, the HLF JSC task is representative of real-time LHC trigger applications, which operate at the 40 MHz bunch-crossing rate and impose stringent microsecond-scale latency constraints [16].
As an end-to-end solution, the FINNAS-generated accelerators therefore provide sufficient latency and throughput margins while achieving a favourable accuracy-resource trade-off under hardware-aware constraints.

IV Conclusion

In this paper, we present FINNAS, which performs hardware-aware architecture and bit-width search for QNNs, enabling the co-optimisation of accuracy and hardware performance.
On the JSC task, compared with a hand-optimised dense accelerator implemented using FINN, the proposed design reduces LUT usage by up to 8.5×8.5\times and latency by 1.7×1.7\times, while achieving higher accuracy, thereby extending the performance frontier of this end-to-end design framework.
In future work, we will investigate the incorporation of sparse patterns into the search space and introduce inter-layer optimisation into the end-to-end automated design flow to improve accelerator performance.

References

[1]
J. Duarte et al. (2018)

Fast inference of deep neural networks in FPGAs for particle physics.

Journal of instrumentation 13 (07), pp. P07027.

Cited by: §I.

[2]
Y. Umuroglu et al. (2020)

LogicNets: Co-Designed Neural Networks and Circuits for Extreme-Throughput Applications.

In Proceedings of the International Conference on Field-Programmable Logic and Applications,

Los Alamitos, CA, USA, pp. 291–297.

Cited by: §I,
TABLE II.

[3]
M. Andronic et al. (2025)

PolyLUT: Ultra-Low Latency Polynomial Inference With Hardware-Aware Structured Pruning.

IEEE Transactions on Computers 74 (9), pp. 3181–3194.

External Links: Document,
ISSN 1557-9956

Cited by: §I,
TABLE II.

[4]
M. Andronic et al. (2024)

NeuraLUT: Hiding Neural Network Density in Boolean Synthesizable Functions.

In Proceedings of the International Conference on Field-Programmable Logic and Applications (FPL),

Vol. , pp. 140–148.

External Links: Document,
ISSN 1946-1488

Cited by: §I,
TABLE II.

[5]
M. Andronic and G. A. Constantinides (2025)

 NeuraLUT-Assemble: Hardware-Aware Assembling of Sub-Neural Networks for Efficient LUT Inference .

In Proceedings of the International Symposium on Field-Programmable Custom Computing Machines (FCCM),

Vol. , Los Alamitos, CA, USA, pp. 208–216.

External Links: ISSN ,
Document,
Link

Cited by: §I,
TABLE II,
TABLE II.

[6]
O. Weng et al. (2025)

Greater than the Sum of its LUTs: Scaling Up LUT-based Neural Networks with AmigoLUT.

In Proceedings of the International Symposium on Field Programmable Gate Arrays (FPGA),

FPGA ’25, New York, NY, USA, pp. 25–35.

External Links: ISBN 9798400713965,
Link,
Document

Cited by: §I,
TABLE II,
TABLE II.

[7]
M. Blott et al. (2018)

FINN-R: An end-to-end deep-learning framework for fast exploration of quantized neural networks.

ACM Transactions on Reconfigurable Technology and Systems (TRETS) 11 (3), pp. 1–23.

Cited by: §I,
§III-B,
TABLE II,
TABLE II,
TABLE II,
TABLE II.

[8]
F. Jentzsch and M. Platzner (2025)

Empirical QoR Estimation Flow for Fast Design Space Exploration of DNN Dataflow Accelerators.

In Proceedings of the International Conference on Field Programmable Technology (ICFPT),

pp. 231–232.

Cited by: §I.

[9]
C. Li, B. Basu, and S. Shanker (2025)

LogicSparse: Enabling Engine-Free Unstructured Sparsity for Quantised Deep-Learning Accelerators.

In Proceedings of the International Conference on Field Programmable Technology (ICFPT),

pp. 223–224.

Cited by: §I.

[10]
B. Zhu et al. (2020)

NASB: Neural Architecture Search for Binary Convolutional Neural Networks.

In Proceedings of the International Joint Conference on Neural Networks (IJCNN),

pp. 1–8.

Cited by: §I.

[11]
M. Ji et al. (2024)

 NASH: Neural Architecture Search for Hardware-Optimized Machine Learning Models.

arXiv preprint arXiv:2403.01845.

Cited by: §I,
§II-A.

[12]
C. Sun et al. (2026)

HGQ: High Granularity Quantization for Real-time Neural Networks on FPGAs.

In Proceedings of the 2026 ACM/SIGDA International Symposium on Field Programmable Gate Arrays,

FPGA ’26, New York, NY, USA, pp. 79–91.

External Links: ISBN 9798400720796,
Link,
Document

Cited by: §III-A,
TABLE II,
TABLE II.

[13]
O. Cassidy et al. (2025)

ReducedLUT: Table Decomposition with "Don’t Care" Conditions.

In Proceedings of the International Symposium on Field Programmable Gate Arrays (FPGA),

FPGA ’25, New York, NY, USA, pp. 36–42.

External Links: ISBN 9798400713965,
Link,
Document

Cited by: TABLE II,
TABLE II.

[14]
C. N. Coelho et al. (2021)

Automatic heterogeneous quantization of deep neural networks for low-latency inference on the edge for particle detectors.

Nature Machine Intelligence 3 (8), pp. 675–686.

External Links: ISSN 2522-5839,
Link,
Document

Cited by: §III-B,
TABLE II,
TABLE II.

[15]
B. Lou et al. (2024)

PolyLUT-Add: FPGA-based LUT Inference with Wide Inputs.

In Proceedings of the International Conference on Field-Programmable Logic and Applications (FPL),

pp. 149–155.

Cited by: TABLE II.

[16]
CMS Collaboration (2020)

The Phase-2 Upgrade of the CMS Level-1 Trigger.

Technical report

 CERN, Geneva.

External Links: Link

Cited by: §III-B.

 Experimental support, please
 view the build logs
 for errors. Generated by
 

 L
 A
 T
 E

xml

.
 

Instructions for reporting errors
We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile
 support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the
 methods listed below:

Click the "Report Issue" (

) button, located in the page header.

Tip: You can select the relevant text first, to include it in your report.
Our team has already identified the following issues. We appreciate your time reviewing and reporting rendering errors we
 may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability
 should not be a barrier to accessing research. Thank you for your continued support in championing open access for
 all.
Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a list of packages that need conversion, and welcome developer contributions.

 We gratefully acknowledge support from
 our major funders,
 member institutions, ,
 and all contributors.
 

About
·
Help
·
Contact
·
Subscribe
·
Copyright
·
Privacy
·
Accessibility
·
Operational Status (opens in new tab)

Major funding support from

