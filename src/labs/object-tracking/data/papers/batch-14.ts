import type { PaperRecord } from "../types";

/* batch-14 — T119..T128 */

export const BATCH_14: PaperRecord[] = [
{
id: "T119",
arxiv: "2507.23251",
title: "A Deep Dive into Generic Object Tracking: A Survey",
shortTitle: "GOT Survey",
year: 2025,
authors: ["Fereshteh Aghaee Meibodi", "Shadi Alijani", "Homayoun Najjaran"],
fileName: "2507.23251v1.pdf",
task: "survey",
tags: ["survey", "generic-object-tracking", "single-object", "siamese", "discriminative", "transformer", "taxonomy", "benchmark-catalog"],
difficulty: "intro",
summary:
"Comprehensive survey of generic (single) object tracking across three paradigms — discriminative (MOSSE to KeepTrack), Siamese (SiamFC to SiamAttn/SiamDMU) and fully/hybrid transformer trackers (STARK to OSTrack, MixFormer, SeqTrack, ODTrack) — with a unified taxonomy, reconstructed standardized architecture diagrams, multi-dimensional comparison tables (appearance model, backbone, design highlight, novelty, drawbacks, template update), a dataset/metric review (Tab. 7) and a LaSOT accuracy-vs-speed comparison (Fig. 29) concluding that fully transformer trackers lead accuracy while the speed/adaptability balance stays open.",
problem:
"Existing GOT surveys each cover only one family (DCF, Siamese, or transformer) or treat paradigms without per-paradigm architectural breakdown, so there is no unified taxonomy comparing appearance modeling, backbones, update strategies, novelties and drawbacks across all major GOT families plus their empirical accuracy/efficiency trade-offs.",
background: ["sot", "bounding-box", "siamese", "correlation-filter", "attention", "success-plot"],
previousWork: [
{
name: "Deep-learning tracking surveys (Marvasti-Zadeh [51], Javed DCF+Siamese outlook [55], Zhang et al. [57])",
limitation:
"Cover CNN/RNN/GAN families or DCF+Siamese jointly but give no fine-grained per-paradigm architectural and update-strategy breakdown including transformers.",
whyThisPaper:
"Adds a unified three-paradigm taxonomy with standardized reconstructed diagrams and multi-dimensional tables (Tabs. 2-6) spanning MOSSE to 2025 transformer trackers.",
},
{
name: "Siamese-only survey (Thangavel [56]) and transformer-only experimental survey ([58])",
limitation:
"Narrowed to one family, so no systematic comparison of Siamese/discriminative designs against transformer-based temporal modeling.",
whyThisPaper:
"Reviews all three families side by side and compares them on LaSOT AUC vs FPS (Fig. 29) plus functional contributions (Tab. 8).",
},
{
name: "Beyond-traditional survey (Abdelaziz et al. [59]: autoregressive, generative, self-supervised, RL, meta-learning)",
limitation:
"Highlights emerging learning paradigms but not the evolution of standard tracking architectures across Siamese/discriminative/transformer lines.",
whyThisPaper:
"Traces standard-architecture evolution (Fig. 1 timeline 2010-2025) with design-level novelty/drawback analysis per representative tracker.",
},
],
researchGap:
"No unified fine-grained survey jointly categorizes Siamese, discriminative and fully/hybrid transformer GOT trackers with consistent structural diagrams, multi-dimensional comparison and joint accuracy/efficiency analysis.",
contribution: [
"Unified taxonomy of GOT into Siamese-based, discriminative-based, and fully plus hybrid transformer-based paradigms with a 2010-2025 timeline (Fig. 1) and standardized reconstructed architecture diagrams (Figs. 2-5 and transformer figures).",
"Multi-dimensional comparison tables (Tabs. 2-6): appearance model, backbone, design highlight, focus, novelty, drawbacks, template update and architecture-level contribution for each representative tracker.",
"Dataset and metric review with structured Tab. 7 cards (OTB, VOT, TLP, UAV123, GOT-10k, LaSOT, TrackingNet and others) and Sec. 4.2 metric definitions (precision, normalized precision, CLE, success rate, AUC, EAO).",
"Empirical trade-off analysis on LaSOT (Fig. 29 AUC vs FPS) plus functional grouping (Tab. 8: distractor handling, robustness, relation modeling, box prediction) and application/future-direction discussion (Secs. 5-7).",
],
method: {
pipeline: ["collect-paradigms", "reconstruct-unified-diagrams", "tabulate-dimensions", "review-datasets-metrics", "accuracy-speed-comparison", "trend-and-future-analysis"],
architecture:
"Survey only: no tracker built. Reconstructs each method in one visual language — CF ridge-regression/FFT blocks with running-average update; Siamese template/search branches with cross/depthwise correlation plus RPN or anchor-free heads; discriminative two-stream classification plus IoU-regression with conjugate-gradient/steepest-descent online optimization; transformer one-stream self/cross-attention with token, memory and sequence-decoding variants.",
motionModel: "Not applicable — surveys appearance/relation models; motion handling reviewed via motion tokens (SwinTrack, ARTrack) and memory/sequence models.",
appearanceModel:
"Compares hand-crafted plus kernelized filters (MOSSE, KCF), regularized/background-aware filters (SRDCF, BACF), multi-domain CNNs (MDNet), meta-learned discriminative models (DiMP, PrDiMP), Siamese similarity maps with distractor-aware and attention upgrades, and transformer joint feature-relation attention with MAE pretraining.",
association: "Not applicable (single-object scope; long-term re-detection reviewed via Siam R-CNN TDPA and KeepTrack candidate association).",
detectionDependency: "None — first-frame box given throughout surveyed GOT methods.",
trackManagement:
"Reviews update rules: CF running-average with learning rate eta, MDNet online fine-tuning, DiMP steepest-descent adaptation, KeepTrack memory-confidence selection, Siamese mostly static templates with DaSiamRPN/DSiam dynamic exceptions, transformer score-gated template refresh.",
loss:
"Catalogues ridge-regression CF objectives, Siamese classification plus regression losses, ATOM/DiMP IoU-prediction losses, PrDiMP KL probabilistic regression and transformer token cross-entropy plus box losses.",
},
equations: [
{
id: "gotsurvey-formulation",
label: "GOT as classification plus state estimation",
formula: "classify: coarse location p(target | region); estimate: full state B = (x, y, w, h)",
variables: [
{ symbol: "p(target | region)", meaning: "classifier score that a region contains the target" },
{ symbol: "B = (x, y, w, h)", meaning: "refined bounding-box state to be estimated" },
],
intuition: "First decide roughly where the target is, then polish the exact box — the two-branch split behind ATOM-style designs.",
why: "Defines the common frame the survey uses to compare discriminative, Siamese and transformer heads.",
where: "Sec. 3, following [8]; classification branch plus state-estimation branch framing.",
paperIds: ["T119"],
},
{
id: "gotsurvey-success-auc",
label: "Success rate and AUC (SOT)",
formula: "SR(theta) = fraction of frames with IoU(pred, gt) > theta; AUC = integral of SR(theta) over theta",
variables: [
{ symbol: "theta", meaning: "IoU threshold swept from 0 to 1" },
{ symbol: "IoU(pred, gt)", meaning: "overlap of predicted and ground-truth boxes" },
],
intuition: "Count how many frames are good enough at each strictness level, then average over all strictness levels.",
why: "Threshold-robust single score used by OTB/LaSOT; the survey's Fig. 29 comparison axis.",
where: "Sec. 4.2; reported as standard on OTB and LaSOT.",
simulator: "metrics",
paperIds: ["T119"],
},
{
id: "gotsurvey-precision",
label: "Precision and normalized precision (SOT)",
formula: "P(delta) = fraction of frames with center distance < delta (20 px); PNorm normalizes distance by box size",
variables: [
{ symbol: "delta", meaning: "center-error threshold in pixels, commonly 20" },
{ symbol: "box size", meaning: "ground-truth width/height used to normalize PNorm" },
],
intuition: "Position-only accuracy: is the predicted center close enough, with PNorm fixing the small-vs-large target unfairness.",
why: "Pairs with AUC on LaSOT; EAO (VOT) additionally penalizes failures into one accuracy-plus-robustness number.",
where: "Sec. 4.2; OTB reports precision and CLE, LaSOT precision plus normalized precision, VOT EAO.",
paperIds: ["T119"],
},
],
datasets: ["otb", "vot", "got10k", "lasot", "trackingnet", "uav123", "others"],
metrics: ["success-auc", "precision", "norm-precision", "eao"],
baselines: ["MOSSE", "KCF", "MDNet", "SRDCF", "DeepDCF", "CFNet", "BACF", "ATOM", "DiMP", "PrDiMP", "KeepTrack", "SiamFC", "DSiam", "SA-Siam", "SiamRPN", "DaSiamRPN", "SiamRPN++", "SiamFC++", "SiamBAN", "Siam R-CNN", "SiamAttn", "STARK", "SwinTrack", "SimTrack", "OSTrack", "MixFormer", "MixFormerV2", "ARTrack", "SeqTrack", "ODTrack", "VideoTrack"],
results: [
"Fig. 29 (LaSOT AUC vs FPS, log scale): fully transformer-based trackers occupy the high-accuracy upper region at moderately fast speeds; discriminative trackers show moderate-to-low accuracy at slow speeds from online learning cost; Siamese trackers are fastest but lower AUC; hybrids sit mid-range on both axes.",
"Highest-ranking LaSOT AUC named as MixFormerV2, SeqTrack and VideoTrack, attributed to rich temporal context modeling and global attention, with runtime constrained relative to lightweight Siamese models.",
"Tab. 7 dataset cards: OTB-2015 (100 seqs), VOT2015/16/18 (60 seqs), TLP (50 seqs, 676k frames), UAV123 (123), GOT-10k (10k videos, 1.5M frames, 563 classes), LaSOT (1,400 seqs, 3.52M frames, 70 classes), TrackingNet (30k, 14M frames), plus ALOV300++, TC-128, OxUvA, LTB35, NUS-PRO with short/long track-type labels.",
"Tab. 8 functional grouping maps families to contributions: distractor handling (negative sampling, hard mining, suppression, MAE), robustness (online Siamese adaptation, meta-learning, long-term support, joint modeling), relation modeling (attention, memory, motion, sequential) and box prediction (anchor-free, IoU, corner, center regression).",
],
ablations: [
"No experiments of its own: accuracy/speed points in Fig. 29 are compiled from reported studies, not a controlled re-evaluation across years, backbones and hardware.",
"Within-text contrasts are qualitative: e.g. convolution-attention hybrids (STARK, CSWinTT, AiATrack, MixFormer) gain structure and distractor robustness but inherit integration dependence; pure-attention one-stream models gain expressiveness at the cost of token design, regularization and pretraining sensitivity.",
],
limitations: {
authorStated: [
"No single paradigm is optimal across scenarios: discriminative trackers pay online-tuning cost, Siamese trackers lack adaptability under occlusion/appearance change, hybrids inherit base-framework drawbacks depending on integration quality.",
"Pure transformer trackers depend on careful token design, attention regularization and specialized pretraining, limiting generalization in unseen or resource-constrained settings; extreme appearance variation and real-time adaptation remain challenging.",
"Short-term datasets lack compound-attribute annotations and segmentation masks; long-term sets neglect gradual temporal consistency and multi-target scenarios.",
],
evident: [
"Compiled cross-paper numbers inherit each source protocol's quirks (box conventions, hardware, FPS measurement) and must not be read as a controlled bake-off.",
"Coverage stops at 2025 GOT scope: MOT, multimodal and 3D tracking advances outside single-object tracking are thin.",
],
},
assumptions: [
"First-frame box given; GOT scope is class-agnostic single-target tracking with short-term visibility assumptions except long-term sections.",
"Compiled scores are quoted as reported under each dataset's own protocol (OTB AUC, LaSOT AUC/PNorm, VOT EAO).",
],
computation:
"No compute reported (survey; supported by NSERC Discovery Grant RGPIN-2023-05408).",
relations: [
{ to: "T009", type: "conceptual-successor", note: "Surveys SiamFC fully-convolutional base founding the Siamese paradigm tables." },
{ to: "T071", type: "conceptual-successor", note: "Surveys OSTrack one-stream joint feature-relation framework among pure-attention trackers." },
{ to: "T070", type: "conceptual-successor", note: "Surveys MixFormer iterative mixed-attention unification of extraction and relation modeling." },
{ to: "T025", type: "conceptual-successor", note: "Compiles GOT-10k scale/protocol card and its role in training and evaluation." },
{ to: "T024", type: "conceptual-successor", note: "Uses LaSOT as the main accuracy-vs-speed comparison benchmark (Fig. 29)." },
{ to: "T066", type: "conceptual-successor", note: "Joint three-paradigm taxonomy extending earlier single-family SOT surveys." },
],
concepts: ["sot", "siamese", "correlation-filter", "attention", "rpn-head", "anchor-free-head", "success-plot", "benchmark-design"],
impact:
"Reference taxonomy pushing later GOT work toward refined temporal-spatial attention, segmentation-assisted localization and memory/online-adaptation modules inside unified end-to-end transformer systems.",
},
{
id: "T120",
arxiv: "2508.11531",
title: "Multi-State Tracker: Enhancing Efficient Object Tracking via Multi-State Specialization and Interaction",
shortTitle: "MST",
year: 2025,
authors: ["Shilei Wang", "Gong Cheng", "Pujian Lai", "Dong Gao", "Junwei Han"],
fileName: "2508.11531v1.pdf",
task: "single-object",
tags: ["efficient-tracking", "lightweight", "one-stream", "multi-state", "state-space-model", "real-time", "vision-transformer"],
difficulty: "intermediate",
summary:
"MST boosts efficient SOT by replacing single-layer features with multi-state generation (MSG: last three backbone layers as diverse target states), state-specific enhancement (SSE) and cross-state interaction (CSI), both built on a lightweight hidden-state-adaptation state-space-duality module (HSA-SSD, +0.1 GFLOPs, +0.66M params). On GOT-10k it reaches 69.6 AO at 200/55 GPU/CPU FPS — 4.5 points over HCAT and 5x faster than OSTrack-256 — and leads all efficient trackers on TrackingNet (81.0 AUC), LaSOT (65.8), UAV123 (68.4), TNL2K (53.3), NFS (65.4) and LaSOText (45.1).",
problem:
"Efficient trackers cut FLOPs and parameters but collapse the target to a single-layer feature, weakening representation so occlusions, appearance change and motion blur break tracking; existing lightweight one-stream designs (MixFormerV2, HiT, FEAR, HCAT) all share this single-state bottleneck.",
background: ["sot", "bounding-box", "iou", "attention"],
previousWork: [
{
name: "Lightweight Siamese/efficient trackers (LightTrack, FEAR, HCAT, E.T.Track)",
limitation:
"Lightweight backbones plus single-layer heads cap feature expressiveness; HCAT, the prior best, stops at 65.1 AO on GOT-10k.",
whyThisPaper:
"MST keeps a ViT-Tiny backbone but adds MSG/SSE/CSI multi-state refinement, reaching 69.6 AO (+4.5) at similar cost (2.28 GFLOPs, 7.80M params).",
},
{
name: "One-stream efficient transformers (MixFormerV2-S, HiT-Base/Small/Tiny, SMAT)",
limitation:
"Distillation/pruning squeezing still decodes one single-layer representation, leaving complementary multi-level cues unused.",
whyThisPaper:
"Uses the last three MSG layers as interacting states: 3-layer input gains +2.1 average (Tab. 5) while 1-2 layers gain only +0.4/+0.9.",
},
{
name: "State-space vision models (Vim, VMamba, MambaVision, EfficientViM HSM-SSD, Mamba-2 SSD, NC-SSD)",
limitation:
"Built for classification with causal or costly channel-quadratic designs, underexplored for tracking state diversity.",
whyThisPaper:
"HSA-SSD adapts HSM-SSD with input-adaptive 1x1 conv plus 3x3 ADWConv: +2.0 AO vs +0.9 HSM-SSD at the same near-zero speed cost (Tab. 6).",
},
],
researchGap:
"No efficient tracker combined multi-state specialization plus cross-state interaction with near-zero overhead real-time SSM inference.",
contribution: [
"Multi-State Tracker framework: MSG generates three state representations from the final three backbone layers; SSE refines each state with HSA-SSD global enhancement; CSI exchanges information across concatenated states and sums them into one unified feature for a center head.",
"HSA-SSD module: adaptive 1x1 convolution generating input-dependent B, C, Delta plus 3x3 adaptive depthwise conv for local spatial dependencies, discretization and hidden-state projection, adding only 0.1 GFLOPs and 0.66M params (Tab. 7).",
"State-of-the-art efficient results on seven benchmarks with real-time speed: 69.6/79.8/64.1 (AO/SR0.5/SR0.75) GOT-10k, 81.0/86.1/78.7 TrackingNet, 65.8/75.2/70.1 LaSOT, 53.3 TNL2K, 68.4 UAV123, 65.4 NFS, 45.1 LaSOText at 200 GPU / 55 CPU FPS and 36 FPS on Jetson Xavier NX.",
"Component and design ablations isolating SSE (-1.2), CSI (-1.1), joint (-2.1), layer-count optimum at 3, and HSA-SSD vs Bi-SSM/HSM-SSD variants.",
],
method: {
pipeline: ["embed-template-search", "multi-state-generation", "state-specific-enhancement", "cross-state-interaction", "aggregate-states", "center-head-predict"],
architecture:
"ViT-Tiny backbone (MAE-distilled init) with MSG over concatenated template/search tokens; per-state SSE blocks and one CSI block, all HSA-SSD based; three-branch center head (classification map P, size map B, offset map O) with Conv-BN-ReLU stacks; Hanning-window penalty at inference.",
motionModel: "Not applicable — no explicit motion filter; temporal robustness comes from multi-state appearance interaction.",
appearanceModel:
"Last-three-layer MSG states refined by input-adaptive HSA-SSD enhancement emphasizing target-specific channels and spatial dependencies; CSI fuses complementary cues across states before the head.",
association: "Not applicable (single-object; argmax over penalized score map).",
detectionDependency: "None — first-frame template (128x128) plus search region (256x256) given.",
trackManagement:
"No online update; inference multiplies the response map by a Hanning window and takes the argmax as the target center (Eqs. 15-16).",
loss:
"Classification branch weighted focal loss plus regression branch L1 and generalized IoU loss; AdamW, 300 epochs of 60k pairs (100 epochs on GOT-10k-only protocol), 4x RTX 2080Ti.",
optimization:
"AdamW over LaSOT/TrackingNet/GOT-10k/COCO2017 pairs; GOT-10k protocol uses only its train split for 100 epochs.",
},
equations: [
{
id: "mst-patch",
label: "Patch projection (MSG input)",
formula: "p_i = W_p v_i + b_p, v_i in {x_i, z_i}",
variables: [
{ symbol: "v_i", meaning: "image patch from template Z or search X" },
{ symbol: "W_p", meaning: "projection matrix to latent dim d" },
{ symbol: "b_p", meaning: "projection bias" },
{ symbol: "p_i", meaning: "token fed to hierarchical attention blocks" },
],
intuition: "Chop both images into patches and embed them into one shared token language.",
why: "Lets template and search tokens mix inside the same self-attention stack.",
where: "Sec. 3.2, Eq. 1; tokens P[Z;X] processed by LN-MHSA-MLP blocks.",
paperIds: ["T120"],
},
{
id: "mst-attention",
label: "Multi-level self-attention (MSG)",
formula: "Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V",
variables: [
{ symbol: "Q, K, V", meaning: "query, key, value projections of concatenated template-search tokens" },
{ symbol: "d_k", meaning: "per-head attention dimension" },
],
intuition: "Every patch votes on every other patch so search features absorb template cues over long range.",
why: "Builds the spatial-context states S(L-2), S(L-1), S(L) that SSE/CSI later specialize.",
where: "Sec. 3.2, Eq. 2; last three layers extracted as multi-state features.",
paperIds: ["T120"],
},
{
id: "mst-aconv",
label: "Adaptive projection (HSA-SSD)",
formula: "B, C, Delta = AConv1x1(S_i); W~(S_i) = sum_k pi_k(S_i) W~_k",
variables: [
{ symbol: "S_i", meaning: "input state features of one MSG layer" },
{ symbol: "W~_k", meaning: "k-th candidate 1x1 kernel" },
{ symbol: "pi_k", meaning: "lightweight attention weight for kernel k from global pooling plus MLP" },
{ symbol: "B, C, Delta", meaning: "input-dependent SSM projection matrices and timescale" },
],
intuition: "Mix convolution kernels on the fly per state instead of using one fixed kernel for all states.",
why: "Core HSA-SSD adaptation letting one SSD design handle diverse state characteristics.",
where: "Sec. 3.3, Eqs. 7-9; followed by ADWConv 3x3 local modeling (Eq. 10).",
paperIds: ["T120"],
},
{
id: "mst-hidden",
label: "Hidden-state refinement (HSA-SSD)",
formula: "h_in = A (*) B^T S_in; Y = C h_out",
variables: [
{ symbol: "A, B", meaning: "discretized state matrices via Delta" },
{ symbol: "S_in", meaning: "current input sequence step" },
{ symbol: "h", meaning: "hidden state accumulating global context" },
{ symbol: "C", meaning: "output projection" },
{ symbol: "Y", meaning: "refined state features (Y_L per layer)" },
],
intuition: "Compress the whole state's story into a compact memory, then read out an enhanced version.",
why: "Linear-time global modeling inside SSE per state and CSI across concatenated states.",
where: "Sec. 3.3, Eqs. 11-13; CSI splits and sums interacted states (Eqs. 4-6, 14).",
paperIds: ["T120"],
},
{
id: "mst-head",
label: "Center-head decision",
formula: "(xd, yd) = argmax P; (x, y, w, h) = (xd + O, yd + O, B, B)",
variables: [
{ symbol: "P", meaning: "classification score map over grid" },
{ symbol: "O", meaning: "offset map correcting discretization error" },
{ symbol: "B", meaning: "size map with normalized width and height" },
{ symbol: "(xd, yd)", meaning: "peak location after Hanning-window penalty" },
],
intuition: "Pick the hottest spot on the heatmap, nudge it to sub-pixel accuracy, attach the predicted box size.",
why: "Anchor-free readout converting the fused multi-state feature into the final box.",
where: "Sec. 3.4, Eqs. 15-16; weighted focal plus L1/GIoU training losses.",
paperIds: ["T120"],
},
],
datasets: ["got10k", "trackingnet", "lasot", "uav123", "others"],
metrics: ["success-auc", "precision", "norm-precision", "fps"],
baselines: ["AsymTrack-B", "SMAT", "HiT-Base", "HiT-Small", "HiT-Tiny", "MixFormerV2-S", "E.T.Track", "FEAR", "HCAT", "LightTrack", "MCITrack-B224", "SAMURAI-L", "ODTrack", "ARTrack-256", "OSTrack-256", "AiATrack", "MixFormer-22k", "ToMP-101", "STARK-ST101", "AVTrack"],
results: [
"GOT-10k (Tab. 1, train-only protocol): MST 69.6 AO / 79.8 SR0.5 / 64.1 SR0.75 at 200 GPU / 55 CPU FPS; paper states +4.5 AO over prior best efficient tracker HCAT (65.1) and 5x faster than OSTrack-256 with equal 69.6 AO.",
"TrackingNet (Tab. 1): 81.0 AUC / 86.1 PNorm / 78.7 P, +1.0/+1.7/+1.4 over HiT-Base; LaSOT: 65.8 / 75.2 / 70.1, +1.2/+1.9/+2.0 over HiT-Base and +4.1/+4.1/+5.5 over SMAT; leads all 14 LaSOT attributes (Fig. 3).",
"TNL2K / UAV123 / NFS / LaSOText (Tab. 3): 53.3 / 68.4 / 65.4 / 45.1 AUC, best on all four with stated gains of +6.1/+1.6/+1.8/+1.0 over prior best.",
"Efficiency (Tab. 2, Tab. 7): 2.28 GFLOPs / 7.80M params (backbone 1.75/5.49, SSE 0.05/0.49, CSI 0.05/0.17, head 0.53/2.31); half the FLOPs of MixFormerV2-S and HiT-Base with +3.3/+2.8 UAV AUC gains; Jetson Xavier NX 36 FPS at 69.6 AO (Tab. 8).",
],
ablations: [
"Key components (Tab. 4, GOT-10k AO / UAV123 AUC): full MST 69.6/68.4; w/o SSE 68.4/67.3 (-1.2/-1.1); w/o CSI 68.6/67.2 (-1.0/-1.2); w/o both 67.5/66.4 (-2.1/-2.0), same as removing MSG/SSE/CSI entirely (67.6/66.2).",
"Layer count (Tab. 5): 1 layer +0.4, 2 layers +0.9, 3 layers +2.1 (69.6/68.4 peak), 4 layers +1.4, 5 layers +1.1 average gain.",
"SSD variants (Tab. 6): Bi-SSM +1.1/+1.5 but GPU/CPU FPS collapse 205/58 to 110/30; HSM-SSD +0.9/+1.4 at 201/56; HSA-SSD +2.0/+2.2 at 200/55 with +0.10 GFLOPs / +0.66M params.",
"Qualitative (Fig. 4): MST holds bicycle, blanket-distractor, goldfish and smoke-occlusion scenes where HiT-Small/Tiny and HCAT drift or misidentify.",
],
limitations: {
authorStated: [
"No dedicated limitations section; the closest stated caveat is that 4-5 MSG layers plateau or degrade because extra layers confuse feature information and shallow backbone noise interferes (Tab. 5 discussion).",
],
evident: [
"Single-dataset efficient protocol: headline GOT-10k numbers use GOT-10k-only training while other tables mix training sets, so cross-table gains are not strictly ablated.",
"CPU 55 FPS and Jetson 36 FPS are real-time but the 200 FPS GPU figure excludes detector-free SOT context; no long-term disappearance or multi-modal evaluation.",
],
},
assumptions: [
"Fixed 128x128 template and 256x256 search inputs; last-three-layer states suffice to represent target diversity.",
"A Hanning-window positional prior at inference correctly biases the peak toward smooth motion.",
],
computation:
"ViT-Tiny backbone with MAE-distilled init; AdamW, 300 epochs x 60k pairs (100 epochs GOT-10k-only); 4x RTX 2080Ti; inference 200 GPU / 55 CPU FPS, 36 FPS Jetson Xavier NX; SSE+CSI +0.1 GFLOPs / +0.66M params.",
relations: [
{ to: "T071", type: "uses-as-baseline", note: "Beats OSTrack-256 efficiency 5x at equal 69.6 GOT-10k AO (Fig. 1, Tab. 1)." },
{ to: "T087", type: "improves", note: "Half the FLOPs of MixFormerV2-S with +3.3 UAV AUC at similar params (Tab. 2)." },
{ to: "T070", type: "uses-as-baseline", note: "Compares against MixFormer-22k heavyweight point in Fig. 1 efficiency plot." },
],
concepts: ["sot", "attention", "bounding-box", "iou", "success-plot"],
impact:
"Shows multi-state specialization plus interaction as a near-free upgrade path for efficient trackers, shifting lightweight design from single-layer heads to SSM-based multi-level fusion.",
},
{
id: "T121",
arxiv: "2511.22896",
title: "DM3T: Harmonizing Modalities via Diffusion for Multi-Object Tracking",
shortTitle: "DM3T",
year: 2025,
authors: ["Weiran Li", "Yeqiang Liu", "Yijie Wei", "Mina Han", "Qiannan Guo", "Zhenbo Li"],
fileName: "2511.22896v1.pdf",
task: "multimodal-tracking",
tags: ["multimodal-tracking", "rgb-thermal", "diffusion", "cross-modal-fusion", "tracking-by-detection", "confidence-guided-association", "online-tracking"],
difficulty: "advanced",
summary:
"DM3T reframes RGB-thermal MOT fusion as iterative feature alignment: a Cross-Modal Diffusion Fusion module (C-MDF, S=3 steps of perturb plus cross-guided residual refinement per modality) projects RGB and thermal features onto a shared manifold, a plug-and-play Diffusion Refiner (DR) enhances the fused map with time-conditioned attention, and a Hierarchical Tracker associates six confidence stages with constant-velocity prediction, EMA velocity update and Hungarian matching on spatial plus size cost. On VT-MOT it reaches 41.7 HOTA / 48.0 IDF1 (1.54 percent relative over PFTrack) at 15.31 FPS.",
problem:
"RGB and thermal features live in different unaligned manifolds with a non-linear distribution gap, so single-step concatenation/addition/attention fusion causes modality conflicts and degrades tracking, especially under darkness, occlusion and low-confidence detections where neither modality alone suffices.",
background: ["mot", "tracking-by-detection", "multimodal", "data-association", "hungarian"],
previousWork: [
{
name: "TBD and JDE/Transformer MOT (SORT/ByteTrack family, FairMOT, CenterTrack, TransTrack, OC-SORT, Hybrid-SORT)",
limitation:
"TBD pipelines propagate detection errors and JDE/unified models under-learn discriminative appearance, while all assume single-modality RGB that collapses in darkness and bad weather.",
whyThisPaper:
"DM3T keeps online detection-plus-association but feeds it harmonized RGB-T features plus a confidence-hierarchical associator, topping VT-MOT at 41.70 HOTA.",
},
{
name: "Conventional RGB-T fusion (concatenation, element-wise ops, attention, progressive fusion PFTrack)",
limitation:
"Single-step integration cannot bridge the non-linear cross-modal distribution gap and leaves modality conflicts in the fused map (Fig. 4a clutter).",
whyThisPaper:
"C-MDF iterative cross-guidance with controlled perturbation forces shared-manifold learning: full model gains +2.00 HOTA / +2.17 IDF1 over concatenation (Tab. 1).",
},
{
name: "Diffusion-based trackers (DiffusionTrack, DiffMOT)",
limitation:
"Apply diffusion to detection heads or box/motion generation, not to feature-level multimodal fusion itself.",
whyThisPaper:
"Reformulates fusion as diffusion-style iterative alignment (C-MDF) plus a plug-and-play diffusion refiner (DR) on the harmonized manifold.",
},
],
researchGap:
"No RGB-T MOT framework treated multimodal fusion as iterative diffusion-style manifold alignment inside a full online detect-plus-associate tracker.",
contribution: [
"DM3T framework unifying DLA-34 detection backbone, C-MDF cross-modal fusion, DR refinement and Hierarchical Tracker into one online system without complex post-processing.",
"Cross-Modal Diffusion Fusion (C-MDF): S=3 stacked harmonization blocks perturbing each modality (Eqs. 2-3) and refining via lightweight residual conv nets on cross-concatenated features (Eqs. 4-5), summed into one unified map (Eq. 6).",
"Plug-and-play Diffusion Refiner (DR): time-embedded (Eq. 7), adaptive perturbation (Eq. 8), S-step attention-plus-skip blocks (Eq. 9) with controlled residual output (Eq. 10, alpha_R).",
"Hierarchical Tracker (Alg. 1): k=6 descending confidence stages, constant-velocity prediction (Eq. 11), EMA velocity update alpha=0.7 (Eq. 12), velocity decay beta=0.1 with amax termination (Eq. 13), spatial plus size cost with lambda_s=100, class constraint, size-gating and Hungarian assignment (Eqs. 14-16).",
],
method: {
pipeline: ["extract-rgb-thermal-features", "inject-history-heatmap", "cross-modal-diffusion-fusion", "diffusion-refine", "multitask-detect", "hierarchical-associate", "predict-update-tracks"],
architecture:
"DLA-34 backbone (Levels 0-5, HDA aggregation) with DLAUp/IDAUp deformable-conv (DCNv2) neck to stride-4 unified map; C-MDF dual residual refinement nets (3x3 conv, BN, ReLU, BN, 3x3 conv); DR time-conditioned attention blocks; multi-task head (position, size, motion displacement); Alg. 1 online tracker.",
motionModel:
"Adaptive constant-velocity: predict p = p_prev + v_prev (Eq. 11); EMA velocity update alpha=0.7 (Eq. 12); unmatched decay v *= max(0.5, 1 - beta*age), beta=0.1, terminate past amax (Eq. 13).",
appearanceModel:
"Harmonized RGB-T fused features with DR-enhanced discriminative detail; validated by entropy 3.25 to 2.59 and kurtosis 117.7 to 128.0 improvements (Tab. 6); no separate ReID embedding bank described.",
association:
"Six descending confidence thresholds; per-stage Hungarian assignment on C = D_spatial + 100*D_size with same-class constraint and size-proportional spatial gating; new tracks above tau_new, unmatched age/decay handling.",
detectionDependency: "Own DLA-34 plus deformable neck multi-task detection head; history frame features and position-prior heatmaps fused into current frame.",
trackManagement:
"Alg. 1 Predict-Associate-Update: matched tracks reset age and EMA-update velocity; unmatched increment age with velocity decay; age beyond amax terminated; high-confidence leftovers birth tracks.",
loss:
"Training uses Adam (lr 1.25e-4, batch 32, 10 epochs on VT-MOT, A100); detection/head loss decomposition not detailed in extracted text.",
optimization:
"Adam optimizer, learning rate 1.25e-4, batch size 32, 10 epochs, NVIDIA A100 GPU, standard VT-MOT protocol.",
},
equations: [
{
id: "dm3t-perturb",
label: "Controlled cross-modal perturbation",
formula: "r~_rgb = r_rgb + sigma * eps_rgb; r~_t = r_t + sigma * eps_t, eps ~ N(0, I)",
variables: [
{ symbol: "r_rgb, r_t", meaning: "current RGB / thermal features at iteration i-1" },
{ symbol: "sigma", meaning: "perturbation scaling hyperparameter" },
{ symbol: "eps", meaning: "standard normal random tensor" },
{ symbol: "r~", meaning: "perturbed features each modality must reconstruct with cross-modal help" },
],
intuition: "Deliberately corrupt each modality so it is forced to borrow information from the other to recover.",
why: "Prevents trivial feature stacking and forces shared-manifold alignment instead of modality conflicts.",
where: "Sec. 3.2.1, Eqs. 2-3; S=3 harmonization steps.",
paperIds: ["T121"],
},
{
id: "dm3t-refine",
label: "Cross-guided residual harmonization and fusion",
formula: "r_rgb = r~_rgb + D_rgb([r~_rgb, r_t]); r_t = r~_t + D_t([r~_t, r_rgb]); f = rS_rgb + rS_t",
variables: [
{ symbol: "D_rgb, D_t", meaning: "lightweight modality-specific residual conv refinement nets" },
{ symbol: "[., .]", meaning: "channel-wise concatenation of perturbed self plus other modality" },
{ symbol: "f", meaning: "final unified cross-modal feature for detection and tracking" },
{ symbol: "S", meaning: "iteration count, S=3" },
],
intuition: "Each modality cleans itself up using the other as a guide, then both summaries are added.",
why: "Progressively resolves cross-modal conflicts while enhancing complementary cues (darkness vs texture).",
where: "Sec. 3.2.1-3.2.2, Eqs. 4-6; residual plus batch-norm design for stable gradients.",
paperIds: ["T121"],
},
{
id: "dm3t-predict",
label: "Constant-velocity track prediction",
formula: "p_hat_t = p_{t-1} + v_{t-1}",
variables: [
{ symbol: "p_{t-1}", meaning: "track position in previous frame" },
{ symbol: "v_{t-1}", meaning: "smoothed velocity from previous frame" },
{ symbol: "p_hat_t", meaning: "predicted position used in the association cost" },
],
intuition: "Glide each track forward along its recent motion before matching new detections.",
why: "Provides the spatial prior the multi-stage association matches against.",
where: "Sec. 3.4.2, Eq. 11; Alg. 1 step 1.",
simulator: "motion",
paperIds: ["T121"],
},
{
id: "dm3t-velocity",
label: "EMA velocity update and decay",
formula: "v_t = (1 - alpha) v_{t-1} + alpha (p_t - p_{t-1}), alpha = 0.7; v <- v * max(0.5, 1 - beta * a_t), beta = 0.1",
variables: [
{ symbol: "alpha", meaning: "EMA smoothing factor prioritizing recent measurements (0.7)" },
{ symbol: "p_t - p_{t-1}", meaning: "observed displacement from the matched detection" },
{ symbol: "a_t", meaning: "frames since last successful match (age)" },
{ symbol: "beta", meaning: "velocity decay rate for unmatched tracks (0.1)" },
],
intuition: "Trust fresh motion more than old, and let drifting predictions coast to a stop when unseen.",
why: "Keeps trajectory continuity through occlusion without unbounded prediction drift; tracks die past amax.",
where: "Sec. 3.4.2, Eqs. 12-13; age reset to 0 on match.",
simulator: "motion",
paperIds: ["T121"],
},
{
id: "dm3t-cost",
label: "Multi-cue association cost",
formula: "D_spatial = ||p_d - p_hat||^2; D_size = |size(d) - size(t)| / max(size); C = D_spatial + 100 * D_size",
variables: [
{ symbol: "p_d, p_hat", meaning: "detection center and predicted track center" },
{ symbol: "size(.)", meaning: "bounding-box scale of detection or track" },
{ symbol: "C", meaning: "final cost matrix solved with the Hungarian algorithm" },
{ symbol: "100", meaning: "lambda_s weighting factor on the size penalty" },
],
intuition: "Match nearby boxes first, penalize wildly different sizes, then solve the global pairing puzzle.",
why: "Balances center proximity with scale consistency; same-class constraint plus size-gating reject invalid pairs.",
where: "Sec. 3.4.3, Eqs. 14-16; per-stage Hungarian assignment in Alg. 1.",
simulator: "assoc-cost",
paperIds: ["T121"],
},
],
datasets: ["others"],
metrics: ["hota", "idf1", "mota", "motp", "deta", "assa", "idsw", "fps"],
baselines: ["FairMOT", "CenterTrack", "TransTrack", "ByteTrack", "OC-SORT", "MixSort-OC", "MixSort-Byte", "PID-MOT", "Hybrid-SORT", "PFTrack"],
results: [
"VT-MOT test (Tab. 4): DM3T 41.703 HOTA / 48.000 IDF1 / 36.759 MOTA / 41.461 DetA / 73.150 MOTP, top HOTA and IDF1 ahead of PFTrack (41.068/47.254/43.088/41.631/73.949); paper states 1.54 percent relative HOTA gain over prior SOTA.",
"MOTA trade-off (Tabs. 2, 4 and text): lowest IDFN (460k) drives HOTA/IDF1 but higher IDFP (341k) and IDs (7195) leave MOTA below PFTrack (36.76 vs 43.09).",
"Fusion ablation (Tab. 1): Base concatenation 38.633/44.259; DR-only 38.871/43.948 (+0.61 HOTA, -0.70 IDF1); full Cross and Refining 40.635/46.426 (+5.18/+4.90 percent) with DetA +2.67 and IDFN -42k.",
"Association ablation (Tab. 2): Hierarchical 41.703/48.000 beats Base 40.635/46.426 and BYTE 40.108/46.791; BYTE cuts IDs/IDFP but raises IDFN to 507k with DetA 38.02.",
"Platforms (Tab. 5): Surveillance 51.50 HOTA vs Handheld 39.99 and UAV 40.21; feature quality (Tab. 6): entropy 3.25 to 2.59, kurtosis 117.7 to 128.0, noise mean 4.93 to 4.23 with C&R.",
"Cost (Tab. 3): +0.56M params (17.307M), +41.58 GFLOPs (107.86G), -2.77 FPS to 15.31 FPS real-time on VT-MOT test (3-run mean).",
],
ablations: [
"DR on unaligned features amplifies modality noise (IDF1 -0.311); C-MDF alignment is prerequisite for DR gains (Tab. 1 discussion).",
"BYTE associator comparison shows confidence-hierarchy better exploits low-confidence valid detections rescued by C-MDF (Tab. 2).",
"Feature-map visualization (Fig. 4) plus Tab. 6 statistics confirm background-clutter suppression; LashHer-020 and Photo-0310-42 sequences show identity stability vs PFTrack switches (Figs. 5-6).",
],
limitations: {
authorStated: [
"Diffusion refinement increases false positives and identity switches, sacrificing MOTA while winning HOTA/IDF1; future work will constrain DR with explicit detection-uncertainty modeling to separate low-confidence true positives from hard false positives.",
"Severe camera motion remains a significant challenge: Handheld (39.99) and UAV (40.21) HOTA lag Surveillance (51.50), motivating motion-robust features or camera-motion compensation.",
],
evident: [
"Iterative fusion costs +41.58 GFLOPs for +2.00 HOTA (15.31 FPS on A100-class hardware), a heavier price than single-step fusion baselines.",
"Evaluation is VT-MOT-only; cross-dataset generalization of the harmonized manifold is not demonstrated.",
],
},
assumptions: [
"Paired synchronized RGB-thermal frames with history features and heatmap priors available each step.",
"Six fixed descending confidence thresholds plus fixed lambda_s=100, alpha=0.7, beta=0.1 transfer across VT-MOT scenes.",
],
computation:
"Adam, lr 1.25e-4, batch 32, 10 epochs, NVIDIA A100; inference 15.31 FPS mean over 3 VT-MOT test runs; backbone-agnostic (DLA-34 default).",
relations: [
{ to: "T092", type: "builds-on", note: "Lifts diffusion iterative-refinement idea from box-level DiffusionTrack to feature-level cross-modal alignment." },
{ to: "T099", type: "builds-on", note: "Follows DiffMOT-style diffusion-for-MOT line but diffuses fusion, not motion prediction." },
{ to: "T061", type: "uses-as-baseline", note: "Compares Hierarchical associator against BYTE association on identical C-MDF+DR features (Tab. 2)." },
{ to: "T073", type: "uses-as-baseline", note: "Outranks OC-SORT 31.479 HOTA on VT-MOT benchmark (Tab. 4)." },
{ to: "T053", type: "uses-as-baseline", note: "Outranks TransTrack 38.000 HOTA on VT-MOT benchmark (Tab. 4)." },
],
concepts: ["mot", "tracking-by-detection", "multimodal", "diffusion-tracking", "data-association", "cost-matrix", "hungarian"],
impact:
"Establishes iterative diffusion-style manifold alignment (perturb plus cross-guide) as a fusion paradigm for RGB-T MOT, with the C-MDF plus DR plus confidence-hierarchy recipe reusable for other heterogeneous modality pairs.",
},
{
id: "T122",
arxiv: "2512.07385",
title: "How Far are Modern Trackers from UAV-Anti-UAV? A Million-Scale Benchmark and New Baseline",
shortTitle: "UAV-Anti-UAV Benchmark and MambaSTS",
year: 2025,
authors: ["Chunhui Zhang", "Li Liu", "Zhipeng Zhang", "Yong Wang", "Hao Wen", "Xi Zhou", "Shiming Ge", "Yanfeng Wang"],
fileName: "2512.07385v1.pdf",
task: "benchmark",
tags: ["benchmark", "uav", "anti-uav", "air-to-air", "vision-language-tracking", "mamba", "state-space-model", "million-scale"],
difficulty: "intermediate",
summary:
"Introduces the UAV-Anti-UAV task (pursuer drone tracking an adversarial drone under dual-dynamic motion) with the first million-scale benchmark: 1,810 videos, 1.05M annotated frames (9.85 h), five target UAV categories, per-sequence language prompts and 15 attributes. The MambaSTS baseline (HiViT visual plus Mamba language streams fused by unidirectional STS Mamba with temporal-token propagation, anchor-free head) tops all 50 evaluated trackers at 0.437 AUC / 0.602 Pre / 0.443 mACC while the 50-tracker average sits at 0.272 mACC, and also leads on UAV123 (0.699), UAV20L (0.711), UAVDT (0.623), VisDrone (0.646) and Anti-UAV sets.",
problem:
"Existing UAV benchmarks track ground targets from a moving camera and Anti-UAV benchmarks track drones from a static ground camera, so neither captures air-to-air pursuit where both observer and target maneuver fast — with dual-dynamic disturbance, tiny scales, motion blur, viewpoint switches and rapid background change — leaving modern trackers unevaluated and unprepared for low-altitude security.",
background: ["sot", "bounding-box", "iou", "benchmark-design", "attention"],
previousWork: [
{
name: "UAV tracking benchmarks and trackers (UAV123, UAV20L, DTB70, UAVDT, VisDrone, WebUAV-3M, TCTrack, HiFT, MambaNUT)",
limitation:
"UAV-to-ground paradigm with motion-stable targets; no dual-dynamic observer-plus-target motion and no language prompts in most.",
whyThisPaper:
"Collects 1,810 air-to-air pursuit videos (1.05M frames) with pursuer motion, language prompts and 15 attributes; MambaSTS still beats UAV specialists on their own sets.",
},
{
name: "Anti-UAV benchmarks and methods (Anti-UAV318/410/600, DUT Anti-UAV, MM-UAV, SiamDT, FocusTrack, UAUTrack, MMA-SORT)",
limitation:
"Static or quasi-static ground observer assumption; methods tuned for tiny thermal targets against sky, not for platform vibration, pursuit viewpoint switches and close-range scale bursts.",
whyThisPaper:
"Adds the moving-observer regime plus a baseline fusing video-level temporal memory with semantics; exposes FO/IV success rates below 0.15 even for top trackers.",
},
{
name: "Modern deep trackers across five families (SiamFC to OSTrack/MixFormerV2, MambaNUT/MambaLCT/MCITrack, VL trackers, MambaTrack/SUTrack/DUTrack)",
limitation:
"Evaluated on conventional sets (LaSOT AUC ~0.7); their global-context and motion-modeling limits under dual dynamics were unmeasured.",
whyThisPaper:
"Evaluates 50 trackers under one OPE protocol: average AUC ~0.30 and mACC 0.272 reveal a large modern-tracker vs pursuit-task gap.",
},
],
researchGap:
"No million-scale air-to-air tracking benchmark with language prompts and fine-grained attributes, and no baseline unifying spatial, temporal and semantic learning for dual-dynamic pursuit.",
contribution: [
"UAV-Anti-UAV task definition (Fig. 1 three-paradigm comparison) plus the first million-scale multi-modal benchmark: 1,810 sequences, 1.05M frames, 9.85 hours, five UAV categories (fixed-wing, FPV, multi-rotor, VTOL, unmanned helicopter), per-frame boxes, language prompts, absent labels and 15 attributes (CM, VC, PO, FO, OV, ROT, SD, IV, MB, PTI, SO, FM, SV, ARV, LEN).",
"MambaSTS baseline: hierarchical HiViT visual stream plus Mamba language stream fused in stacked STS Mamba modules with unidirectional scanning and temporal-token propagation compressing whole-history cues into a latent state; anchor-free classification/offset/size head with focal plus L1 plus GIoU loss.",
"50-tracker evaluation across CNN, CNN-Transformer, Transformer, Mamba and Mamba-Transformer families with OPE AUC/Pre/cAUC/nPre/mACC plus 15-attribute breakdowns (Figs. 8-10).",
"Ablations (Tab. 5: temporal/spatial/semantic stacking 27.8 to 43.7 AUC), complexity analysis (54 FPS on A6000), cross-task generalization (UAV tracking plus Anti-UAV sets) and stated limitations with future directions (Sec. 5.8).",
],
method: {
pipeline: ["crop-template-search", "extract-hivit-visual-tokens", "extract-mamba-language-tokens", "fuse-via-sts-mamba", "propagate-temporal-token", "anchor-free-predict"],
architecture:
"HiViT-base visual backbone (n=20 blocks, MAE init; 4x4 embedding plus two 2x2 merges, stride 16, D=512) for template Z and search X; GPT-NeoX plus Mamba-130M language branch (Nl=40 tokens); stacked STS Mamba modules (m=24 Vim-small layers, unidirectional scan) inserted stage-wise; fully-convolutional anchor-free head on reshaped search tokens.",
motionModel:
"Learned long-term memory instead of a filter: temporal token T_temp propagated frame to frame by selective SSM scanning of all historical search features (Eq. 10), injected into each stage as motion/appearance prior for occlusion and blur recovery.",
appearanceModel:
"Multi-scale HiViT spatial tokens modulated stage-wise by language semantics and video-level temporal context; selective scanning filters background noise (clouds, clutter) while retaining target cues.",
association: "Not applicable (single-object per sequence; recursive per-frame state estimation without template updates).",
detectionDependency: "None — first-frame template (128x128, 22x target area) and language prompt given; search 256x256 (42x area).",
trackManagement:
"Recursive inference: previous temporal token plus current search features enter STS Mamba each frame; language features extracted once per video so overhead is negligible.",
loss:
"L_total = L_cls (weighted focal) + 5 L1 + 2 L_GIoU; AdamW (weight decay 1e-4, batch 32), backbone lr 2e-4, rest 2e-3, 300 epochs with 10x decay after epoch 240; 2-frame clips; trained on GOT-10k plus LaSOT plus COCO plus TrackingNet plus UAV-Anti-UAV train.",
optimization:
"AdamW as above on 8x A6000 (Ubuntu 20.04, PyTorch 2.1.1); 54 FPS inference on a single A6000; default official hyperparameters for all 50 compared trackers.",
},
equations: [
{
id: "mambasts-ssm",
label: "Continuous state-space model",
formula: "h'(t) = A h(t) + B x(t); y(t) = C h(t)",
variables: [
{ symbol: "x(t)", meaning: "1-D input signal at time t" },
{ symbol: "h(t)", meaning: "latent state in R^N" },
{ symbol: "A, B, C", meaning: "state-evolution, input-projection and output-projection parameters" },
{ symbol: "y(t)", meaning: "output response" },
],
intuition: "Keep a running memory that evolves linearly with the input — the mathematical ancestor of Mamba and the Kalman filter.",
why: "Foundation the paper revisits before discretizing it for deep visual tracking (Mamba makes parameters data-dependent).",
where: "Sec. 4.2, Eqs. 1-2; SSMs noted as originating from the Kalman filter.",
paperIds: ["T122"],
},
{
id: "mambasts-discretize",
label: "Zero-order-hold discretization",
formula: "Abar = exp(Delta A); Bbar = (Delta A)^-1 (exp(Delta A) - I) Delta B",
variables: [
{ symbol: "Delta", meaning: "timescale parameter discretizing continuous dynamics" },
{ symbol: "Abar, Bbar", meaning: "discrete state matrices used in the scan" },
{ symbol: "I", meaning: "identity matrix" },
],
intuition: "Convert smooth physics into step-by-step updates a neural network can run per token.",
why: "Enables the selective scan to run as a discrete recursion over patch and temporal tokens.",
where: "Sec. 4.2, Eqs. 3-4; tokenization with position embeddings and class token follows (Eq. 9).",
paperIds: ["T122"],
},
{
id: "mambasts-temporal",
label: "Temporal-token propagation",
formula: "T_temp <- Mamba{Fx1, ..., Fxi}; h_t = Abar h_{t-1} + Bbar F_t",
variables: [
{ symbol: "Fx1..Fxi", meaning: "search-region visual features from frame 1 to current frame i" },
{ symbol: "T_temp", meaning: "compact temporal token carrying video-level target history" },
{ symbol: "h_t", meaning: "recurrent hidden state filtering background noise, keeping target cues" },
],
intuition: "Squeeze the entire pursuit history into one pocket token handed to the next frame as a memory of how the target moves and looks.",
why: "Linear-complexity long-term context that survives full occlusion and blur where 2-frame clips fail; injected into every HiViT stage.",
where: "Sec. 4.4, Eq. 10; initial token random, updated stage-wise and propagated.",
simulator: "motion",
paperIds: ["T122"],
},
{
id: "mambasts-fusion",
label: "STS Mamba spatial-temporal-semantic fusion",
formula: "Z_in = Concat(Ds(Fz), Ds(Fx), Fl, T_temp); H_k = Abar H_{k-1} + Bbar Z_in[k]; Y = C H",
variables: [
{ symbol: "Fz, Fx", meaning: "template and search visual tokens (Ds = downsample alignment)" },
{ symbol: "Fl", meaning: "language tokens (Nl=40) as semantic prior" },
{ symbol: "H_k", meaning: "unidirectional-scan hidden state at step k (forward/causal only)" },
{ symbol: "Y", meaning: "fused output split back into visual tokens plus next temporal token" },
],
intuition: "Lay appearance, words and memory side by side and read them strictly forward in time so the present can only learn from the past.",
why: "Critical change from bidirectional Vim: respects tracking causality while fusing all three cues coarse-to-fine across stages.",
where: "Sec. 4.5, Eqs. 11-12; hierarchical stacking with global-attention plus MLP refinement per stage.",
paperIds: ["T122"],
},
{
id: "mambasts-loss",
label: "Anchor-free multi-task loss",
formula: "L_total = L_cls + 5 L1 + 2 L_GIoU",
variables: [
{ symbol: "L_cls", meaning: "weighted focal loss on the center classification map P" },
{ symbol: "L1", meaning: "box regression L1 loss (weight 5)" },
{ symbol: "L_GIoU", meaning: "generalized IoU loss (weight 2)" },
{ symbol: "P, O, B", meaning: "predicted center map, offset map and size map heads" },
],
intuition: "Reward confident centers and tight boxes with standard tracking loss weights.",
why: "Trains the three head branches (center probability, discretization offset, width/height) jointly.",
where: "Sec. 4.6, Eq. 13; follows OSTrack-style anchor-free conventions.",
paperIds: ["T122"],
},
],
datasets: ["antitrack", "uav123", "trackingnet", "got10k", "lasot", "others"],
metrics: ["success-auc", "precision", "norm-precision", "fps"],
baselines: ["SiamFC", "ECO", "VITAL", "ATOM", "SiamMask", "SiamRPN++", "SiamFC++", "SiamBAN", "SiamCAR", "LightTrack", "SiamGAT", "TrDiMP", "TransT", "STARK-ST50", "KeepTrack", "HiFT", "AutoMatch", "TCTrack", "ToMP-101", "AiATrack", "SimTrack-B32", "OSTrack", "ZoomTrack", "MixFormerV2-B", "GRM", "DropTrack", "SeqTrack-B256", "ODTrack", "EVPTrack", "AQATrack", "HIPTrack", "ARTrackV2", "LORAT-B224", "MambaNUT", "SGLATrack", "ORTrack", "MambaLCT", "MCITrack-B224", "VLTTT", "JointNLT", "CiteTracker-256", "All-in-One", "UVLTrack", "MambaTrack", "SUTrack-B224", "DUTrack-256", "ATCTrack"],
results: [
"UAV-Anti-UAV OPE (Figs. 8-9): MambaSTS 0.437 AUC / 0.602 Pre / 0.433 cAUC / 0.480 nPre / 0.443 mACC, first on every metric; +6.6pp mACC over SUTrack-B224 (0.377), ~17pp over the 50-tracker mean (0.272); CNN classics (SiamFC 0.135, HiFT, TCTrack) trail with AUC below 0.27.",
"Attributes (Fig. 10): MambaSTS leads on FM (0.315), CM (0.213), MB (0.195), SO (0.142), SD (0.205 vs 0.125 MambaTrack); all trackers collapse on FO (MambaSTS 0.050) and IV (0.118), and CM/FM/ROT averages sit below 0.3.",
"Ablation (Tab. 5, from OSTrack-ViT-B 27.8 AUC): plus temporal 33.4, plus spatial (HiViT) 36.9, plus semantic 43.7 AUC / 60.2 Pre — total +15.9 AUC with semantics alone worth +6.8.",
"Generalization: UAV123 0.699, UAV20L 0.711, UAVDT 0.623, VisDrone 0.646 (Fig. 11, best on all four); DBT70 competitive (Fig. 12); Anti-UAV sets lead the SOTA table (Tab. 4).",
"Dataset stats: brightness mean 114.06 (central across sets, Fig. 4); relative speed 0.79 fastest (Fig. 5); 76.5 percent short / 17.8 medium / 5.7 long videos; SV in 1,324 and ARV in 1,040 of 1,810 sequences (Fig. 3).",
],
ablations: [
"Tab. 5 component stacking proves each cue indispensable with the largest jump from language semantics (+6.8 AUC).",
"Attribute slices (Fig. 10) isolate failure modes: illumination, full occlusion and motion attributes bound all 50 trackers.",
"Cross-task checks (Figs. 11-12, Tab. 4) confirm the baseline transfers to ground-target UAV sets and ground-based Anti-UAV sets.",
],
limitations: {
authorStated: [
"RGB-only videos: night and adverse-weather operations need infrared/LiDAR modalities not yet in the benchmark; adversarial maneuvers, collisions and electronic countermeasures cannot be captured at scale.",
"MambaSTS AUC 43.7 percent leaves considerable headroom; language-prompt dependence requires accurate descriptions unavailable in fully autonomous pursuit; offline training should become online adaptation (Sec. 5.8).",
],
evident: [
"Template-style venue header (IEEE TRANSACTIONS XXX) means no peer-reviewed venue can be recorded.",
"Even the best scores on FO/IV/PTI attributes are near-failure (<0.15), so pursuit under disappearance or bad light is unsolved.",
],
},
assumptions: [
"Axis-aligned boxes plus one language prompt per sequence adequately describe small maneuvering drones.",
"OPE without re-initialization plus default official hyperparameters fairly ranks 50 trackers built on diverse frameworks.",
],
computation:
"8x A6000 (48GB), Xeon 8375C, 512GB RAM, PyTorch 2.1.1; MambaSTS 54 FPS single A6000 (vs OSTrack 105, MixFormerV2-B 165); HiViT-base (n=20) plus 24 Vim-small STS layers; train batch 32.",
relations: [
{ to: "T065", type: "extends", note: "Million-scale air-to-air benchmark with prompts/attributes beyond WebUAV-3M ground-target scope." },
{ to: "T071", type: "uses-as-baseline", note: "OSTrack-ViT-B is the Tab. 5 ablation baseline (27.8 AUC) improved to 43.7." },
{ to: "T009", type: "uses-as-baseline", note: "SiamFC evaluated as the classic CNN baseline (0.135 mACC, lowest rank)." },
],
concepts: ["sot", "benchmark-design", "multimodal", "attention", "motion-model", "success-plot"],
impact:
"Foundational air-to-air testbed plus a spatial-temporal-semantic baseline recipe; attribute results redirect future work to illumination/occlusion adaptation, motion modeling and online learning for pursuit.",
},
{
id: "T123",
arxiv: "2604.00452",
title: "Out of Sight, Out of Track: Adversarial Attacks on Propagation-based Multi-Object Trackers via Query State Manipulation",
shortTitle: "FADE",
year: 2026,
authors: ["Halima Bouzidi", "Haoyu Liu", "Yonatan Achamyeleh", "Praneetsai Iddamsetty", "Mohammad Al Faruque"],
fileName: "2604.00452v1.pdf",
task: "multi-object",
tags: ["adversarial-attack", "query-propagation", "end-to-end-mot", "temporal-memory", "sensor-spoofing", "robustness", "pgd"],
difficulty: "advanced",
summary:
"FADE is the first adversarial framework for Tracking-by-Propagation (query-based end-to-end) MOT: Temporal Query Flooding (TQF) injects persistent spurious tracks via flood plus cost-mimicry plus identity-siphoning losses to starve the fixed query budget, while Temporal Memory Corruption (TMC) erases legitimate tracks via state de-correlation plus feature erasure. Optimized by PGD in pixel space or through differentiable acoustic (AAI) and electromagnetic (EAI) sensor-spoofing simulations, FADE drops MOTRv2/MOT20 HOTA ~30 points (59.56 to 29.64), MeMOTR/MOT20 by 31.5 points with 10x IDSW, and corrupts memory so one attacked frame erases all tracks for 15+ clean frames.",
problem:
"All prior MOT attacks target Tracking-by-Detection machinery (NMS thresholds, Kalman Mahalanobis gating, separate ReID banks) that does not exist in query-propagation trackers, leaving the novel TBP attack surface — fixed query budgets, recurrent hidden-state propagation and long-term memory/SSM banks — unstudied despite TBP's temporal smoothing resisting naive single-frame perturbations.",
background: ["mot", "tracking-by-detection", "end-to-end-mot", "attention", "data-association"],
previousWork: [
{
name: "TBD detection-evasion attacks (False-Negative attacks, Daedalus false-positive flooding)",
limitation:
"Assume hard NMS/IoU thresholds; TBP suppresses duplicates with learned attention, so naive floods are filtered (Daedalus barely moves MeMOTR/MOT20).",
whyThisPaper:
"TQF adds cost-mimicry and identity-siphoning so spurious queries persist as tracks: full LTQF beats naive flood on CO-MOT (37.26 vs 40.25 HOTA).",
},
{
name: "TBD association attacks (Hijack, F&F vs Kalman; BankTweak vs appearance banks)",
limitation:
"Coupled to Kalman predictions or separable ReID features with pipeline access; TBP learns dynamics in propagated queries with unified representations.",
whyThisPaper:
"TMC directly de-correlates recurrent query states and erases matched embeddings, collapsing AssA (MOTR/MOT20 AssA -36 via AAI) where Hijack/F&F stay near clean on MOTRv2.",
},
{
name: "TBP trackers (MOTR, MOTRv2, MeMOTR, Samba, CO-MOT) and physical sensor spoofing (Poltergeist/TPatch AAI, GlitchHiker EAI)",
limitation:
"TBP work hardens accuracy/memory without threat modeling; spoofing work only hit single-frame detection, never temporal tracking logic.",
whyThisPaper:
"Formalizes zero-sum budget plus memory-dependence vulnerabilities and a differentiable physical-parameter PGD pipeline (blur/stripe params bounded to calibrated sensor ranges) proving realizability.",
},
],
researchGap:
"No attack exploited TBP's query-budget and recurrent-memory constraints, and no differentiable bridge existed from digital query-state losses to physically plausible sensor-spoofing vectors.",
contribution: [
"Vulnerability analysis of TBP: fixed query budget as zero-sum allocation, recurrent hidden-state error chains, and memory/SSM amplification of corrupted states (vs per-frame TBD resets).",
"Temporal Query Flooding (TQF): L_Flood (Eq. 1) plus L_Cost bipartite-mimicry (Eq. 2) plus L_Siphon identity-mimicry (Eq. 3) summed in Eq. 4 to exhaust budgets and siphon legitimate identities.",
"Temporal Memory Corruption (TMC): L_Decorr frame-to-frame cosine minimization (Eq. 5) plus L_Erase matched-embedding norm collapse (Eq. 6) summed in Eq. 7 to sever temporal links and erase identities.",
"Differentiable digital-to-physical pipeline: PGD over pixels (eps 8/255) or over calibrated AAI (x, y, phase, D=10) / EAI (M stripes, N=20) parameters within plausible ranges; 1-frame digital / 3-frame physical attack windows.",
"Evaluation on MOTR, MOTRv2, MeMOTR, Samba, CO-MOT over MOT17/MOT20 with HOTA/DetA/AssA/IDF1/IDR/IDP/IDSW plus ablations, convergence and qualitative amnesia analysis.",
],
method: {
pipeline: ["encode-frame", "init-track-plus-detect-queries", "decode-jointly", "predict-boxes-scores", "update-queries-memory", "pgd-optimize-ltqf-or-ltmc", "apply-digital-or-physical-perturbation"],
architecture:
"Victim-agnostic TBP primer: vision encoder to Xt; input queries Qin = propagated track queries T_{t-1} (M <= B budget) plus N detect queries; L-layer deformable transformer decoder with self-attention duplicate suppression; box plus score heads; QueryUpdater (MLP/SSM/memory bank Mt) producing next T_t. Attack reads logits z, costs C, hidden states H.",
motionModel: "Victims learn motion in query propagation (no Kalman); FADE poisons exactly that learned temporal state.",
appearanceModel:
"Unified query embeddings carry identity; TMC erases matched embeddings (L_Erase) and de-correlates consecutive states (L_Decorr) so re-association fails.",
association:
"Victims use learned attention plus bipartite matching; TQF deceives it with low-cost adversarial queries (L_Cost) that look like re-emerging known objects (L_Siphon).",
detectionDependency: "Victim detectors (incl. MOTRv2 external detector) attacked indirectly through query objectness maximization rather than NMS thresholds.",
trackManagement:
"Attack exploits fixed budget B: spurious persistent tracks consume slots (query starvation); corrupted memories force track termination and 15-frame re-initialization gaps.",
loss:
"Adversarial only: L_TQF (Eq. 4) or L_TMC (Eq. 7) maximized by PGD (alpha 1/255 digital, 8/255 physical; T=50/100 iters; ~3s/frame RTX 3090). No victim retraining.",
optimization:
"PGD with sign gradients, Clip to Lp ball (digital) or calibrated sensor ranges (physical); same PGD settings across all five victim trackers.",
},
equations: [
{
id: "fade-flood",
label: "Query flooding loss",
formula: "L_Flood = E_{q in T_adv}[(1 - max_c softmax(z)_c)^2]",
variables: [
{ symbol: "T_adv", meaning: "K unmatched (available) track queries designated adversarial" },
{ symbol: "z", meaning: "output logits of an adversarial query" },
{ symbol: "max_c softmax(z)_c", meaning: "perceived objectness (best-class confidence)" },
],
intuition: "Convince the tracker every spare query slot holds a perfectly confident object.",
why: "First step of budget exhaustion; alone it makes only naive single-frame false positives.",
where: "Sec. 3.2.1, Eq. 1; persistence requires cost plus siphon terms.",
paperIds: ["T123"],
},
{
id: "fade-cost",
label: "Bipartite cost-mimicry loss",
formula: "L_Cost = -E_{g in G_t}[log sum_{q in T_adv} exp(-C(q, g))]",
variables: [
{ symbol: "G_t", meaning: "ground-truth objects (or tracker predictions) at frame t" },
{ symbol: "C(q, g)", meaning: "bipartite matching cost between adversarial query and object" },
{ symbol: "T_adv", meaning: "adversarial query set competing for the match" },
],
intuition: "Make a fake query look like the cheapest match for a real object so the tracker adopts it.",
why: "Deceives the assignment logic into misassociating real identities onto adversarial queries.",
where: "Sec. 3.2.1, Eq. 2; critical against memory trackers (removing it weakens Samba attack by 7.6 HOTA).",
paperIds: ["T123"],
},
{
id: "fade-siphon",
label: "Identity-siphoning loss",
formula: "L_Siphon = -E[cosine(h_adv^t, h_anchor^{t-1})]",
variables: [
{ symbol: "h_adv^t", meaning: "hidden state of a current adversarial query" },
{ symbol: "h_anchor^{t-1}", meaning: "hidden state of a previous legitimate track" },
{ symbol: "cosine", meaning: "cosine similarity maximized to force indistinguishability" },
],
intuition: "Disguise the fake as the comeback of a familiar face so its history gets inherited.",
why: "Grants temporal persistence: the adversarial query rides legitimate memory instead of being born fresh each frame.",
where: "Sec. 3.2.1, Eq. 3; full L_TQF weighted sum in Eq. 4.",
paperIds: ["T123"],
},
{
id: "fade-decorr",
label: "Temporal de-correlation loss",
formula: "L_Decorr = E[cosine(norm(H^t), norm(H^{t-1}))] (minimized)",
variables: [
{ symbol: "H^t, H^{t-1}", meaning: "current and previous query hidden-state sets" },
{ symbol: "norm", meaning: "normalization before similarity" },
],
intuition: "Scramble the resemblance between a track's today and yesterday so the updater cannot recognize it.",
why: "Ablations prove it the single most critical TMC term: removing it rebounds Samba HOTA by 13.6 and MeMOTR by 10.2 points.",
where: "Sec. 3.2.2, Eq. 5; complemented by erasure in Eq. 7.",
paperIds: ["T123"],
},
{
id: "fade-erase",
label: "Track-identity erasure loss",
formula: "L_Erase = E_{h in H_matched^t}[||h||_2^2] (minimized)",
variables: [
{ symbol: "H_matched^t", meaning: "embeddings of currently matched (legitimate) queries" },
{ symbol: "||.||_2^2", meaning: "squared L2 magnitude collapsed toward zero" },
],
intuition: "Drain the feature lifeblood out of every valid track until nothing trackable remains.",
why: "Renders legitimate tracks untrackable in later frames; removing it weakens the Samba attack by 4.3 HOTA.",
where: "Sec. 3.2.2, Eq. 6; weighted with de-correlation in Eq. 7.",
paperIds: ["T123"],
},
{
id: "fade-objective",
label: "Unified PGD attack objective",
formula: "max_{omega} L_adv(f_MOT(F_adv(omega)), G_t); digital: F+delta, ||delta||_inf <= 8/255; physical: P(F; theta_AAI || theta_EAI)",
variables: [
{ symbol: "omega", meaning: "optimized variables: pixels delta or physical params theta" },
{ symbol: "f_MOT", meaning: "victim TBP tracker mapping frames to outputs and hidden states" },
{ symbol: "theta_AAI", meaning: "acoustic params (x, y amplitudes, phase phi, D=10 blur steps)" },
{ symbol: "theta_EAI", meaning: "electromagnetic params (stripe mask M, N=20 stripe count)" },
],
intuition: "Search the worst allowed tweak — pixels or sensor physics — that maximizes the chosen query-state damage.",
why: "One PGD loop serves both threat models with ranges clipped to lab-calibrated real-sensor behavior.",
where: "Sec. 3.2.3-3.2.4, Eq. 8; 1-frame digital / 3-frame physical windows, T=50/100 iters.",
paperIds: ["T123"],
},
],
datasets: ["mot17", "mot20"],
metrics: ["hota", "deta", "assa", "idf1", "idsw"],
baselines: ["MOTR", "MOTRv2", "MeMOTR", "Samba", "CO-MOT", "Daedalus", "Hijacking", "F&F"],
results: [
"MOT20 digital (Tab. 2): MOTRv2 clean 59.56 HOTA falls to 29.64 under FADE-TQF (~30-point drop, IDSW 0.73 to 5.10); MeMOTR 69.61 to 37.70 under FADE-TMC (-31.5, IDSW 0.46 to 4.90, over 10x); CO-MOT 64.31 to 33.37 TQF; MOTR 55.06 to 40.38 TMC.",
"MOT17 digital (Tab. 1): CO-MOT 58.16 to 37.26 TQF (5.2x IDSW); Samba 62.49 to 48.04 TMC (-14.78); MOTR ~58.63 to ~45.9 both variants; Hijack/F&F stay near clean on MOTRv2 (58.27/58.85 vs 59.56) proving Kalman-targeting fails on TBP.",
"Physical (Tabs. 3-4): MOTR/MOT20 AAI 55.06 to 36.17 HOTA with AssA -36; MOTR/MOT17 TMC-EAI 58.63 to 42.81; robust-backbone MOTRv2/CO-MOT still lose 7-9 HOTA with 8-9 AssA leakage past the detector; denser MOT20 amplifies damage (-18.9 vs -13.5 AAI on MOTR).",
"Ablations (Tab. 5, MOT17): full LTQF beats LFlood-only (CO-MOT 37.26 vs 40.25); L_Cost removal +7.6 HOTA on Samba; L_Decorr removal +13.6 Samba / +10.2 MeMOTR; EAI converges by T=5 (43.42) while AAI needs T=100 (39.85, Tab. 6).",
"Persistence (Figs. 3-4): one attacked frame wipes all MeMOTR tracks with zero recovery for 15 clean frames (re-init at frame 25); physical AAI/EAI blur/stripe victims still unrecoverable at frame 25.",
],
ablations: [
"Loss-component study (Tab. 5) on CO-MOT/MeMOTR/Samba isolates flood vs mimicry vs siphon and decorrelation vs erasure, revealing tracker-dependent weak points (Samba filters siphon via SSM memory; CO-MOT falls to it).",
"PGD convergence study (Tab. 6, MOTR/MOT17) shows EAI's flat fast landscape vs AAI's complex gradual one.",
"Qualitative amnesia demos (Figs. 3-4) distinguish transient glitches from persistent memory corruption.",
],
limitations: {
authorStated: [
"No closed-loop validation on real hardware: simulations use physically calibrated sensor profiles but real-world dynamics remain unexplored (Sec. 5).",
"Robust cross-architecture black-box transferability is an open challenge; observed transfer is partial among models sharing architectural lineage.",
],
evident: [
"White-box PGD needs logits, costs and hidden states plus ~3s per frame on RTX 3090, far from real-time weaponization.",
"Physical vectors are simulated (differentiable AAI/EAI), not captured through real acoustic coils or EMI rigs.",
],
},
assumptions: [
"White-box access to victim outputs and recurrent hidden states during optimization.",
"Calibrated AAI/EAI parameter ranges from prior sensor studies faithfully bound real-world artifact magnitudes.",
],
computation:
"PGD T=50 digital / T=100 physical, eps 8/255, alpha 1/255 digital and 8/255 physical; ~3s per attacked frame on RTX 3090; 1-frame digital and 3-frame physical windows.",
relations: [
{ to: "T059", type: "uses-as-baseline", note: "MOTR is a victim tracker; FADE-TMC drops its MOT20 HOTA 55.06 to 40.38." },
{ to: "T080", type: "uses-as-baseline", note: "MOTRv2 is a victim tracker; FADE-TQF drops its MOT20 HOTA 59.56 to 29.64." },
{ to: "T054", type: "uses-as-baseline", note: "TrackFormer lineage frames the TBP query-propagation threat model FADE formalizes." },
],
concepts: ["mot", "end-to-end-mot", "memory-network", "data-association", "cost-matrix", "hungarian"],
impact:
"First robustness benchmark logic for query-propagation tracking: future TBP work must budget for query-slot defense, memory sanitization and sensor-spoofing-aware temporal links before safety-critical deployment.",
},
{
id: "T124",
arxiv: "2605.26933",
title: "Leveraging Text-to-Image Diffusion Models for Unsupervised Visual Object Tracking",
shortTitle: "Diff-Tracking",
year: 2026,
authors: ["Zhengbo Zhang", "Zhigang Tu", "Junsong Yuan", "De Wen Soh", "Bo Du"],
fileName: "2605.26933v1.pdf",
task: "single-object",
tags: ["unsupervised-tracking", "diffusion", "text-to-image", "cross-attention", "prompt-learning", "frequency-domain", "online-update"],
difficulty: "advanced",
summary:
"Diff-Tracking reinterprets a frozen Stable Diffusion v2.1 model as a bridge between text and image: it learns a target-representative prompt embedding whose cross-attention map activates the target region in each frame. An initial prompt learner (target-shared plus target-specific embeddings, attention harmonization, multi-layer fusion head) localizes the target, and an online prompt updater refines the prompt frame-by-frame from RGB plus frequency-domain motion cues. It reports 70.4 success on TrackingNet, 50.4 on LaSOT and 68.7 on OTB2015, the best among unsupervised trackers.",
problem:
"Unsupervised trackers must follow arbitrary targets with no annotated training video, and the hardest cases need fine-grained semantic and structural understanding inside frames plus contextual relations across frames, where correlation-filter and Siamese self-supervision fall short.",
background: ["sot", "bounding-box", "iou", "siamese", "correlation-filter", "attention", "diffusion-tracking", "success-plot"],
previousWork: [
{
name: "Supervised Siamese trackers (SiamFC, SiamRPN++, ATOM, DiMP, OSTrack)",
limitation:
"Depend on large-scale annotated video training to learn template-search matching, so they cannot operate in the annotation-free regime.",
whyThisPaper:
"Diff-Tracking needs no annotated video training: it reuses semantic knowledge already inside a frozen text-to-image diffusion model and learns only prompt embeddings from pseudo-labels.",
},
{
name: "Unsupervised CF/Siamese trackers (UDT, S2SiamFC, LUDT+, USOT two-stage cycle training, ULAST)",
limitation:
"Self-supervision from single frames or forward-backward consistency captures limited semantics and structure, and relies on online updating to handle variation.",
whyThisPaper:
"Diffusion cross-attention maps provide pretrained semantic-structure grounding, lifting TrackingNet success 59.9 (USOT) and 64.9 (ULAST) to 70.4 without test-time template updates.",
},
{
name: "Diff-Tracker conference version (ECCV 2024)",
limitation:
"Target embedding is purely target-specific, uses a single UNet cross-attention layer, and extracts motion only in RGB, failing when target and background share color.",
whyThisPaper:
"This extension adds a target-shared saliency embedding, a multi-layer attention fusion head, and frequency-domain motion, gaining +2.9 TrackingNet success (67.5 to 70.4) and +1.8 LaSOT success (48.6 to 50.4).",
},
],
researchGap:
"No unsupervised tracker exploited the semantic-structure knowledge inside pretrained text-to-image diffusion models via prompt-driven cross-attention activation.",
contribution: [
"Diff-Tracking framework: initial prompt learner generates a target prompt that activates the first-frame box in diffusion cross-attention; online prompt updater refines it with motion for continuous tracking.",
"Attention harmonization plus multi-layer fusion head: self-attention-enhanced cross-attention maps from 8 UNet decoder layers fused with learnable weights, encoding target-background relations.",
"Target-shared embedding Esh: class-agnostic saliency prior (nearest words object/foreground/thing) concatenated with per-video target-specific embedding Esp.",
"Frequency-domain motion branch: DCT-based RGB-to-frequency transform plus dual RGB/frequency short- and long-term motion encoders with cross-attention target conditioning and MLP fusion heads.",
"State of the art among unsupervised trackers on six benchmarks with real-time 35 FPS (49 FPS with distilled BK-SDM backbone).",
],
method: {
pipeline: ["pseudo-label-mining", "learn-shared-prompt", "test-time-target-prompt", "activate-cross-attention", "motion-conditioned-update", "box-from-attention"],
architecture:
"Frozen Stable Diffusion v2.1 UNet; 2-layer MLP embedding projector (1024-d Esp/Esh); attention harmonization; 2-layer MLP fusion head over 8 decoder layers (64x64 maps); ResNet-18 motion encoders plus 2 Conv3D layers; 2-layer MLP long-short, RGB-frequency and blend heads.",
motionModel:
"Target-conditioned short-term (current vs previous frame) and long-term (current plus T preceding frames) motion from ResNet-18 encoders, queried by first-frame template appearance via cross-attention; RGB and DCT-frequency branches fused by MLP.",
appearanceModel:
"No appearance template matching: target identity is a learned prompt (Esp+Esh) that activates semantically aligned regions in diffusion cross-attention maps; self-attention maps supply target-background relational cues.",
association:
"Not applicable (single target): per-frame target box is the minimum axis-aligned box enclosing the activated cross-attention region.",
detectionDependency: "No detector; first-frame box given; pseudo training boxes mined by ARFlow optical flow plus dynamic-programming selection on GOT-10k, ImageNet VID, LaSOT, YouTube-VOS.",
trackManagement:
"Initial prompt tracks frames 1-5; online updater refines prompt every frame from frame 6 onward with residual blend pk = (1-beta) Hb(pk-1,mk) + beta pk-1.",
loss:
"MSE between final attention map M and binarized pseudo-mask Mb plus LDM text-space regularizer (Eq. 7); Esp test-time adaptation 3 epochs.",
optimization:
"Adam: projector/fusion/Esh lr 5e-4 for 20 epochs; updater lr 1e-4 for 60 epochs; Esp test-time lr 5e-3 for 3 epochs (~1.7 s init on RTX 4090); 512x512 inputs.",
},
equations: [
{
id: "difftrack-ldm",
label: "Latent diffusion training objective",
formula: "LDM = E[||eps - eps_theta(zt, t, Etxt(p))||^2], eps ~ N(0,1)",
variables: [
{ symbol: "eps", meaning: "Gaussian noise added to the image latent at timestep t" },
{ symbol: "eps_theta", meaning: "UNet noise predictor conditioned on text prompt p" },
{ symbol: "zt", meaning: "noisy image latent at diffusion timestep t" },
{ symbol: "Etxt(p)", meaning: "text-encoder embedding of prompt p" },
],
intuition: "Learn to undo noise step by step, guided by text, which forces the model to bind words to image regions.",
why: "This pretraining is what makes cross-attention maps semantically grounded localization signals for tracking.",
where: "Sec. III-A, Eq. 1; reused as regularizer in Eq. 7 so learned prompts stay in the diffusion text space.",
paperIds: ["T124"],
},
{
id: "difftrack-crossattn",
label: "Cross-attention localization signal",
formula: "Mc = Softmax(Qc Kc^T / sqrt(d))",
variables: [
{ symbol: "Qc", meaning: "query projected from noisy image latent" },
{ symbol: "Kc", meaning: "key projected from text prompt embedding" },
{ symbol: "Mc", meaning: "cross-attention map highlighting prompt-aligned image regions" },
{ symbol: "d", meaning: "projection dimension" },
],
intuition: "Each text token scores every image patch; high scores mark where the described concept appears.",
why: "The entire tracker hangs on this: a learned target prompt activates the target region in Mc each frame.",
where: "Sec. III-B, Eq. 2; Mc extracted from 8 UNet decoder layers for the fusion head.",
paperIds: ["T124"],
},
{
id: "difftrack-enhanced",
label: "Self-attention enhancement",
formula: "Mcprime(:,:) = sum_i sum_j Mc(i,j) * Ms(i,j,:,:)",
variables: [
{ symbol: "Ms", meaning: "self-attention map encoding pixel-to-pixel semantic relations" },
{ symbol: "Mcprime", meaning: "cross-attention map reweighted by inter-pixel relationships" },
{ symbol: "i,j", meaning: "spatial positions in the attention map" },
],
intuition: "Spread each pixel vote to its semantically related pixels, so the target region fills in coherently and background leaks less.",
why: "Injects target-background relational cues that pure cross-attention misses under deformation and occlusion.",
where: "Sec. IV-A, Eq. 4; Mc resized to Mcprime size before fusion.",
paperIds: ["T124"],
},
{
id: "difftrack-harmonize",
label: "Attention harmonization",
formula: "M = (1 - alpha) * Mcprime + alpha * Mc",
variables: [
{ symbol: "alpha", meaning: "learnable balance between raw and relation-enhanced maps" },
{ symbol: "M", meaning: "output cross-attention map per UNet layer" },
],
intuition: "A learned knob mixing direct prompt response with context-smoothed response.",
why: "Ablation: removing harmonization drops VOT2018 EAO 38.4 to 37.1.",
where: "Sec. IV-A, Eq. 5; applied per layer before the fusion head.",
params: "alpha learned; endpoints recover either map alone.",
paperIds: ["T124"],
},
{
id: "difftrack-fusion",
label: "Multi-layer attention fusion head",
formula: "M = (1/N) * sum_{n=1..N} w(n) * Hf(M(n))",
variables: [
{ symbol: "M(n)", meaning: "harmonized map from n-th UNet decoder layer, resized to uniform size" },
{ symbol: "Hf", meaning: "shared fusion head (two FC layers plus ReLU)" },
{ symbol: "w(n)", meaning: "learnable weight of n-th layer output" },
{ symbol: "N", meaning: "number of layers fused (N = 8)" },
],
intuition: "Intermediate layers contribute semantic structure, deeper layers fine texture; learn how much to trust each.",
why: "Single-layer maps (conference version) discard complementary cues; fusion improves LaSOT success 49.5 to 50.4.",
where: "Sec. IV-A, Eq. 6; final map M supervised against pseudo-mask.",
paperIds: ["T124"],
},
{
id: "difftrack-loss",
label: "Prompt learning loss",
formula: "L = (1/HW) * sum_{i,j} (Mb(i,j) - M(i,j))^2 + LDM",
variables: [
{ symbol: "Mb", meaning: "binarized pseudo-mask (1 inside box, 0 outside)" },
{ symbol: "M", meaning: "final fused attention map" },
{ symbol: "LDM", meaning: "diffusion loss keeping embeddings in interpretable text space" },
],
intuition: "Push the attention spotlight to exactly cover the box and nowhere else.",
why: "Only supervision for Esp/Esh, projector and fusion head; diffusion UNet stays frozen.",
where: "Sec. IV-A, Eq. 7; Mb from ARFlow+DP pseudo-labels in training, GT box at test time.",
paperIds: ["T124"],
},
{
id: "difftrack-dct",
label: "RGB-to-frequency transform",
formula: "Y(u,v) = (2/sqrt(MN)) C(u)C(v) sum_m sum_n X(m,n) cos((2m+1)u pi/2M) cos((2n+1)v pi/2N)",
variables: [
{ symbol: "X", meaning: "input frame in RGB spatial domain" },
{ symbol: "Y", meaning: "DCT coefficients in frequency domain" },
{ symbol: "C(u), C(v)", meaning: "1/sqrt(2) at zero frequency else 1 (orthonormal scaling)" },
],
intuition: "Rewrite the image as texture frequencies; target and background with similar color but different texture separate here.",
why: "Camouflaged-soldier example indistinguishable in RGB separates in frequency Y-channel; removing this branch drops LaSOT precision 49.2 to 48.3.",
where: "Sec. IV-B, Eqs. 8-9; standard 8x8-block DCT before frequency motion encoders.",
paperIds: ["T124"],
},
{
id: "difftrack-promptupdate",
label: "Residual prompt update",
formula: "pk = (1 - beta) * Hb(pk-1, mk) + beta * pk-1",
variables: [
{ symbol: "pk", meaning: "prompt for k-th frame" },
{ symbol: "mk", meaning: "fused RGB-frequency long-short motion representation" },
{ symbol: "Hb", meaning: "blend head fusing previous prompt with motion" },
{ symbol: "beta", meaning: "learnable balance between update and history" },
],
intuition: "Nudge last frame prompt with fresh motion evidence but keep most of what already worked.",
why: "Removing the updater drops VOT2018 EAO 38.4 to 35.5; per-frame updates beat every-3/5/10-frame schedules.",
where: "Sec. IV-B, Eq. 10; active from frame 6 to sequence end.",
params: "Start frame 6 optimal (frames 2/4 add noise, 8/10 lag); update interval 1 best.",
paperIds: ["T124"],
},
],
datasets: ["otb", "vot", "trackingnet", "lasot", "got10k", "others"],
metrics: ["success-auc", "precision", "norm-precision", "eao", "fps"],
baselines: ["SiamFC", "DaSiamRPN", "SiamRPN", "SiamRPN++", "ATOM", "DiMP", "KCF", "DSST", "ECO", "S2SiamFC", "LUDT+", "USOT", "USOT-on", "AUDI-T", "ULAST", "ULAST-on", "Diff-Tracker"],
results: [
"TrackingNet test (Tab. I): 70.4 Suc / 64.1 Pre / 77.5 NPre, best unsupervised; beats ULAST-on 65.4/59.2/73.2 and Diff-Tracker 67.5/61.4/75.1, near supervised ATOM 70.3/64.8.",
"VOT2016 (Tab. I): 45.3 EAO / 61.1 Acc / 19.8 Rob vs prior best ULAST-on 41.7/60.3/21.4 and Diff-Tracker 43.0/60.5/20.6.",
"VOT2018 (Tab. I): 38.4 EAO / 59.4 Acc / 25.9 Rob vs ULAST-on 35.5/57.1/28.6 and Diff-Tracker 36.5/58.0/27.3.",
"OTB2015 (Tab. II): 68.7 Suc / 91.2 Pre vs Diff-Tracker 66.1/89.8 and ULAST-on 64.8/87.9.",
"LaSOT test (Tab. II): 50.4 Suc / 49.2 Pre vs Diff-Tracker 48.6/47.2 and ULAST-on 47.1/45.1.",
"VOT2020 (Tab. II): 47.6 Acc / 56.8 Rob / 25.4 EAO vs Diff-Tracker 46.1/58.7/23.9 and USOT 45.8/60.0/22.2.",
"Speed (Tab. VII, RTX 4090): 35 FPS on LaSOT; BK-SDM distilled backbone reaches 49 FPS at 50.7 Suc; RTX 3060 runs 25 FPS at 5.1 GB.",
],
ablations: [
"Attention harmonization (Tab. III, VOT2018): full 38.4/59.4/25.9 vs w/o 37.1/58.3/27.8 EAO/Acc/Rob.",
"Online updater (Tab. III): full 38.4 vs w/o updater 35.5 EAO; updater start frame 6 best (49.7/50.1/50.4/50.2/49.8 Suc for starts 2/4/6/8/10); interval 1 best (50.4 vs 49.5/49.2/48.6 at 3/5/10).",
"Target-shared embedding Esh (Tab. IV, LaSOT): full 50.4/49.2 vs w/o 49.1/48.9 Suc/Pre; attention maps collapse under occlusion/blur without Esh (Fig. 7).",
"Fusion head (Tab. IV): full 50.4/49.2 vs single-layer 49.5/48.3; frequency motion (Tab. IV): full 49.2 vs RGB-only 48.3 Pre; long-term motion matters more than short-term (47.1 vs 48.4 w/o, Tab. V).",
"Pretraining-isolation (Tab. XI, VOT2018 EAO): naive category-name SD baseline 17.1; LUDT+/USOT with SD features gain only +2.3/+2.2; full method 38.4, proving gains come from tracking designs.",
"Pseudo-label robustness (Tabs. XII-XIII): TV-L1/ARFlow/SEA-RAFT labels give 49.8/50.4/50.8 Suc; relaxing quality threshold Qv 0.4 to 0.2 costs only 0.6 Suc.",
],
limitations: {
authorStated: [
"Pseudo-label mining inherits ARFlow plus dynamic-programming failure modes under camera shake and occlusion, producing noisy boxes that the DP reward and quality filters (Qv below 0.4 discarded) are designed to suppress.",
"No explicit failure-mode section; robustness to noisy labels is argued empirically rather than guaranteed.",
],
evident: [
"Test-time Esp adaptation costs ~1.7 s per sequence and the diffusion UNet dominates per-frame compute, so throughput trails lightweight Siamese trackers despite 35 FPS on flagship GPUs.",
"First-frame box plus frames 1-5 rely purely on the static initial prompt, so early appearance change before frame 6 is unhandled by design.",
],
},
assumptions: [
"Frozen Stable Diffusion cross-attention maps are semantically grounded enough to separate any tracking target from its background.",
"ARFlow plus dynamic-programming pseudo-boxes are sufficiently clean after quality filtering to supervise prompt learning.",
],
computation:
"RTX 3090 training; RTX 4090 inference 35 FPS (SD v2.1) / 49 FPS (BK-SDM distilled, 24% less memory); 1.7 s per-sequence Esp init; 512x512 inputs; 64x64 attention maps.",
relations: [
{ to: "T108", type: "extends", note: "Journal extension of the ECCV 2024 Diff-Tracker: adds shared embedding, fusion head and frequency motion; 70.4 vs 67.5 TrackingNet success." },
{ to: "T009", type: "uses-as-baseline", note: "Supervised SiamFC tabulated on all six benchmarks (e.g. 57.1 TrackingNet success) as the annotation-dependent reference." },
],
concepts: ["diffusion-tracking", "sot", "attention", "multimodal"],
impact:
"First unsupervised tracker built on frozen text-to-image diffusion cross-attention, with contribution analysis showing tracking-specific prompt designs rather than pretraining scale drive the gains.",
},
{
id: "T125",
arxiv: "2606.02724",
title: "AVTrack: Audio-Visual Tracking in Human-centric Complex Scenes",
shortTitle: "AVTrack",
year: 2026,
authors: ["Yaoting Wang", "Yun Zhou", "Zipei Zhang", "Henghui Ding"],
venue: "ICML 2026",
fileName: "2606.02724v1.pdf",
task: "multimodal-tracking",
tags: ["audio-visual", "benchmark", "instance-segmentation", "human-centric", "multimodal-tracking", "vision-language-model", "test-only"],
difficulty: "intermediate",
summary:
"AVTrack is a test-only human-centric audio-visual instance segmentation (AVIS) benchmark: 871 clips averaging 54.0 s with 3,120 densely annotated sounding-person tracklets spanning 8 complex conditions (camera motion, occlusion, position change, multi-turn speech). VIS methods collapse below 12 HOTA and AVIS methods below 21.5; the modular AVTracker baseline (Whisper plus Qwen3-VL plus SAM3, local-to-global tracklet grouping) reaches 29.08 HOTA.",
problem:
"Existing audio-visual datasets are lab-controlled, single-source, short (5-10 s) or visually simple, so evaluation rewards static audio-visual co-occurrence instead of robust spatiotemporal modeling and cross-modal reasoning in dynamic human scenes.",
background: ["mot", "detection", "data-association", "multimodal", "mota", "hota", "idf1", "benchmark-design"],
previousWork: [
{
name: "Audio-visual speaker tracking corpora (AV16.3, CAV3D, AVRI, AVA-ActiveSpeaker)",
limitation:
"Controlled rooms, static cameras, few speakers, or box-only labels without cross-frame identity, missing real-world dynamics.",
whyThisPaper:
"AVTrack curates 871 in-the-wild clips over 6 genres with mask plus identity labels under 8 enforced complexity criteria.",
},
{
name: "Audio-visual segmentation benchmarks (AVSBench, AVSS, Ref-AVS, AVISeg)",
limitation:
"Short trimmed clips or visually simple scenes (AVISeg: only 7.1% camera motion, 8.6% occlusion) reduce the task to frame-level audio-conditioned segmentation.",
whyThisPaper:
"AVTrack enforces complexity: 90.5% camera motion, 80.9% occlusion, 70.7% position change, 56.8% multi-turn speech, 9.2% audio-visual inconsistency.",
},
{
name: "VIS/AVIS methods (VITA, CAVIS, AVISM, ACVIS)",
limitation:
"Trained on simple benchmarks, they fail at long-range identity association under fragmented asynchronous human-centric conditions.",
whyThisPaper:
"They score 9.7-21.5 HOTA on AVTrack vs their in-domain levels, and the AVTracker baseline doubles them via explicit local-to-global reasoning.",
},
],
researchGap:
"No rigorously curated, annotation-intensive, test-only benchmark measured robust human-centric audio-visual instance tracking in complex dynamic scenes.",
contribution: [
"AVTrack dataset: 871 clips, 100% test split, 54.0 s average, 3,120 instance tracklets with mask plus identity labels across TV, film, vlog, animation, reality, interview and stage sources.",
"Eight enforced complexity criteria with distribution analysis against AVISeg showing large difficulty increases on every axis.",
"Full benchmark of VIS and AVIS methods plus AVTrackFormer, an end-to-end AVISM variant with bidirectional audio-object interaction (21.47 HOTA).",
"AVTracker: modular three-stage baseline (speaker-chunk aggregation, local window, global window) reaching 29.08 HOTA with plug-and-play components.",
],
method: {
pipeline: ["asr-chunking", "speaker-embedding-merge", "local-av-grounding", "mask-pooling", "keyframe-selection", "global-identity-grouping", "trajectory-merge"],
architecture:
"Whisper (809M large-v3-turbo) ASR plus ECAPA-TDNN speaker embeddings (tau 0.35 merge) plus optional MossFormer2 separation; Qwen3-VL-8B local/global reasoners; SAM3 mask sampler; 1 FPS processing; AVTrackFormer appendix variant: 1-s snippets, frozen audio encoder, AV-PFM cross-attention, Nf frame queries, VITA-style object tokens with bidirectional AV-OFM.",
association:
"Local: argmax-IoU match of VLM-predicted speaker box to SAM3 boxes per frame forming chunk tracklets; global: VLM groups keyframes (max-area frame per chunk) into identities per Eq. 9-10.",
detectionDependency: "SAM3 person boxes/masks per frame; VLM local reasoner conditioned on chunk transcript; no trained detector.",
trackManagement:
"Dynamic ASR windows merged by speaker similarity; empty masks inserted for frames without observations to keep trajectories continuous over the full video.",
optimization:
"Training-free baseline (frozen foundation models); AVTrackFormer trained as AVISM with frozen audio encoder; ablations over Whisper 244M/809M and Qwen 4B/8B.",
},
equations: [
{
id: "avtrack-chunksim",
label: "Speaker chunk similarity and merge",
formula: "sim(ci,ci+1) = ei^T ei+1 / (||ei|| ||ei+1||); merge if sim > tau (tau = 0.35)",
variables: [
{ symbol: "ei", meaning: "ECAPA-TDNN speaker embedding of enhanced chunk audio" },
{ symbol: "ci", meaning: "ASR chunk (start, end, transcript)" },
{ symbol: "tau", meaning: "similarity threshold 0.35" },
],
intuition: "Adjacent utterances that sound like the same voice belong to one speaker chunk, cutting redundant windows.",
why: "Removing compression collapses HOTA 24.01 to 16.88 in ablations.",
where: "Sec. 4.2, Eqs. 1-3; temporal plus textual concatenation of merged chunks.",
paperIds: ["T125"],
},
{
id: "avtrack-localalign",
label: "Local box-to-mask alignment",
formula: "b(f) = argmax_{b in B_SAM3(f)} IoU(bR(f), b)",
variables: [
{ symbol: "bR(f)", meaning: "VLM local-reasoner speaker box for frame f given transcript" },
{ symbol: "B_SAM3(f)", meaning: "SAM3 candidate person boxes in frame f" },
{ symbol: "b(f)", meaning: "matched box whose SAM3 mask enters the local tracklet" },
],
intuition: "Let the VLM point at the speaker with words, then snap to the best segmentation mask.",
why: "Grounds language-guided reasoning into pixel masks for the tracklet (Eq. 7).",
where: "Sec. 4.3, Eqs. 5-6; keyframe is the max-area mask frame (Eq. 8).",
simulator: "iou-track",
paperIds: ["T125"],
},
{
id: "avtrack-hota",
label: "HOTA",
formula: "HOTA = (1/|A|) sum_{alpha in A} sqrt(DetA_alpha * AssA_alpha)",
variables: [
{ symbol: "alpha", meaning: "IoU matching threshold; averaged over set A" },
{ symbol: "DetA_alpha", meaning: "detection accuracy at threshold alpha" },
{ symbol: "AssA_alpha", meaning: "association accuracy at threshold alpha" },
],
intuition: "Geometric mean of finding boxes and keeping identities, so neither alone can win.",
why: "Primary AVTrack metric balancing the two failure modes of AVIS.",
where: "Appendix metric definitions, Eq. 11; Tab. 2 reports it for all methods.",
simulator: "metrics",
paperIds: ["T125"],
},
{
id: "avtrack-deta",
label: "Detection accuracy (DetA)",
formula: "DetA = TP / (TP + FN + FP)",
variables: [
{ symbol: "TP", meaning: "true-positive sounding instances at frame level" },
{ symbol: "FN, FP", meaning: "missed and spurious sounding instances" },
],
intuition: "Did we draw masks on the right speaking people in each frame.",
why: "Isolates localization from identity tracking in the HOTA decomposition.",
where: "Eq. 12; AVTracker reaches 31.18 vs AVISM 23.22.",
simulator: "metrics",
paperIds: ["T125"],
},
{
id: "avtrack-assa",
label: "Association accuracy (AssA)",
formula: "AssA = (1/|C|) sum_{c in C} |TPA(c)| / (|TPA(c)| + |FNA(c)| + |FPA(c)|)",
variables: [
{ symbol: "C", meaning: "set of true-positive detection matches" },
{ symbol: "TPA/FNA/FPA(c)", meaning: "true/false association links induced by match c" },
],
intuition: "For each found speaker, how consistently the same identity sticks across frames.",
why: "The metric where AVTracker gains most (+8 over AVISM), reflecting long-range grouping.",
where: "Eq. 13; ablations show model scale and separation act mainly here.",
simulator: "metrics",
paperIds: ["T125"],
},
{
id: "avtrack-idf1",
label: "IDF1",
formula: "IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN)",
variables: [
{ symbol: "IDTP", meaning: "identity true positives over the whole video" },
{ symbol: "IDFP/IDFN", meaning: "spurious and missed identity assignments" },
],
intuition: "Global identity preservation; fragmentation anywhere in a long video hurts.",
why: "Complements HOTA for the multi-turn long-video regime (54 s average).",
where: "Eq. 14; AVTracker 34.55 vs AVISM 26.57.",
simulator: "metrics",
paperIds: ["T125"],
},
{
id: "avtrack-mota",
label: "MOTA",
formula: "MOTA = 1 - (FN + FP + IDSW) / GT",
variables: [
{ symbol: "IDSW", meaning: "identity switches" },
{ symbol: "GT", meaning: "total ground-truth instances" },
],
intuition: "All errors per ground truth; dominated by detection misses in AVTrack.",
why: "Reported for comparability although HOTA is primary (VIS methods score ~2 MOTA).",
where: "Eq. 15; AVTracker 16.20 vs best prior 4.23.",
simulator: "metrics",
paperIds: ["T125"],
},
],
datasets: ["others"],
metrics: ["hota", "deta", "assa", "idf1", "mota"],
baselines: ["VITA", "LBVQ", "CAVIS", "AVISM", "ACVIS", "AVTrackFormer", "Gemini 2.5 Pro"],
results: [
"AVTrack test (Tab. 2, HOTA/DetA/AssA/IDF1/MOTA): VIS collapses - VITA 9.70/10.54/9.35/12.32/1.91, LBVQ 10.29, CAVIS 11.46; AVIS doubles but stays low - AVISM 20.84/23.22/19.53/26.57/3.95, ACVIS 20.60, AVTrackFormer 21.47/22.51/20.26/26.41/4.11.",
"AVTracker reaches 29.08/31.18/28.47/34.55/16.20, about +8 HOTA over the strongest AVIS baseline, with MOTA 4x higher.",
"Scale ablation (Tab. 3): Base 28.85/31.75/27.39; smaller ASR (M1) 25.19, smaller VLM (M2) 24.47, both small (M3) 24.01, face-feature global (M4) 23.62 HOTA.",
"Separation (Tab. 3): MossFormer2 (S2) 29.08/28.47 vs base 28.85/27.39; SepFormer (S1) degrades to 28.41, showing unreliable separation adds noise.",
"Chunk processing (Tab. 3): no compression (C1) collapses to 16.88 HOTA; fixed windows (C2) drop to 27.45.",
"Commercial AV LLM probe: Gemini 2.5 Pro reaches only 14.4 HOTA, below trained AVIS baselines.",
],
ablations: [
"Model scale: shrinking either modality backbone costs 3.7-4.4 HOTA; replacing the VLM global reasoner with face features costs 5.23 HOTA and 6.08 AssA.",
"Speech separation helps only when reliable: MossFormer2 +0.23 HOTA, SepFormer -0.44 HOTA.",
"Chunk compression is load-bearing: without it HOTA falls 24.01 to 16.88; dynamic windowing adds +1.4 over fixed windows (28.85 vs 27.45).",
"Qualitative: AVTracker holds identities under relative position change but still struggles under simultaneous multi-speaker plus occlusion (Fig. 6).",
],
limitations: {
authorStated: [
"AVTrack is test-only by design, so whether scaling training data closes the exposed gaps remains an open question for future data-collection pipelines.",
"AVTracker struggles when multiple speakers coincide with visual occlusion; alignment under multi-turn speech, inconsistency and multi-speaker conditions remains extreme.",
],
evident: [
"Cascaded foundation-model pipeline (Whisper plus 8B VLM plus SAM3 at 1 FPS) implies heavy inference cost with no reported FPS, limiting real-time use.",
"Privacy and surveillance misuse risks of speaker localization are flagged in the impact statement rather than mitigated technically.",
],
},
assumptions: [
"ASR transcripts plus speaker embeddings suffice to partition speakers before any visual grounding.",
"1 FPS sampling preserves the semantics needed for speaker association.",
],
computation:
"No training for AVTracker (frozen Whisper-809M, Qwen3-VL-8B, SAM3); tau 0.35; r = 1 FPS; annotation cost: ~1300 collected, ~1000 retained, 871 final via Grounded-SAM pre-annotation plus 15 annotators over ~3 months with LabelMe refinement.",
relations: [],
concepts: ["multimodal", "mot", "benchmark-design", "hota"],
impact:
"Establishes the first complex human-centric AVIS testbed with metric definitions (Eqs. 11-15) reused by the lab registry, plus a training-free VLM baseline paradigm.",
},
{
id: "T126",
arxiv: "2606.05587",
title: "HDST-GNN: Heterogeneous Dynamic Spatiotemporal Graph Neural Networks for Multi-Object Tracking in UAV Aerial Imagery",
shortTitle: "HDST-GNN",
year: 2026,
authors: ["Phillip Jiang"],
fileName: "2606.05587v1.pdf",
task: "multi-object",
tags: ["tracking-by-detection", "graph-neural-network", "uav", "heterogeneous-graph", "occlusion-handling", "sinkhorn", "re-identification"],
difficulty: "advanced",
summary:
"HDST-GNN tackles UAV MOT on VisDrone2019-MOT with a heterogeneous spatiotemporal graph: an altitude proxy from mean object area adapts the edge radius, detections plus active plus lost tracklets get distinct node types and five typed edges, and an occlusion gate scales messages from unreliable nodes. A ResNet-18 appearance backbone with Sinkhorn training and ByteTrack-style two-stage Hungarian inference reaches 94.51 MOTA and 97.24 IDF1 under oracle detections, cutting SORT identity switches 144 to 28.",
problem:
"Drone altitude varies within and across sequences so fixed-radius graphs over-connect at high altitude and under-connect near ground; tiny dense objects plus frequent occlusion demand re-identification, while homogeneous graphs conflate new detections, confirmed tracklets and lost targets with different matching semantics.",
background: ["mot", "tracking-by-detection", "data-association", "cost-matrix", "hungarian", "kalman", "reid", "appearance-features", "dual-threshold", "learned-association"],
previousWork: [
{
name: "SORT-line TBD trackers (SORT, DeepSORT, ByteTrack, OC-SORT, StrongSORT)",
limitation:
"Fixed IoU thresholds and Euclidean matching are poorly calibrated for altitude-varying aerial sequences with tiny objects.",
whyThisPaper:
"HDST-GNN replaces fixed matching with learned heterogeneous graph affinities, beating SORT by +5.0 MOTA and -81% ID switches under oracle detections.",
},
{
name: "Graph MOT methods (MPNTrack, GNNMatch, GSDT, TrackFormer, MOTR, NOWA-MOT)",
limitation:
"Homogeneous nodes and fixed spatial context ignore altitude effects and the distinct lifecycle states of detections vs tracklets vs lost targets.",
whyThisPaper:
"Altitude-adaptive radius plus three node types plus five typed edges plus occlusion gating; removing the adaptive radius alone costs 6.07 MOTA.",
},
{
name: "ReID plus differentiable matching (SuperGlue Sinkhorn, triplet loss)",
limitation:
"Appearance embeddings and soft assignment existed separately but not inside an altitude-aware heterogeneous aerial tracker.",
whyThisPaper:
"ResNet-18 128-d L2 embeddings trained with BCE on Sinkhorn assignments plus triplet margin 0.3 give 49% fewer ID switches under noisy detections.",
},
],
researchGap:
"No aerial graph tracker jointly adapted spatial context to altitude, separated lifecycle states into node types, and gated messages by occlusion confidence.",
contribution: [
"Altitude-adaptive edge construction: altitude proxy z from mean object area drives radius reff = r0 exp(-beta z) in [30, 500] px.",
"Heterogeneous graph: Type-D detection, Type-T confirmed and Type-L lost nodes with dedicated projections and five typed edge relations (match, match-rev, context, interact, reid).",
"Occlusion-gated temporal aggregation: source messages scaled by occlusion confidence phi, with 4-head attention plus type-specific FFN over L = 3 layers.",
"End-to-end training with Sinkhorn dustbin assignment plus BCE and triplet loss; ByteTrack-protocol two-stage Hungarian inference; detection augmentation simulating noise.",
],
method: {
pipeline: ["crop-embed", "altitude-proxy", "build-heterogeneous-graph", "gated-message-passing", "sinkhorn-affinity", "two-stage-hungarian", "track-update"],
architecture:
"ResNet-18 appearance extractor (128-d L2 embeddings); heterogeneous GNN with H = 256, L = 3 layers, Nh = 4 heads, type-specific projections and FFNs; Sinkhorn head (K = 20, tau 0.07); Hungarian inference.",
motionModel:
"Normalized box plus velocity in Type-T features with Kalman-free learned refinement; lost nodes carry frames-lost/30 with temporal-decay re-ID edges exp(-lambda_d frames-lost).",
appearanceModel:
"ResNet-18 ImageNet-pretrained, MLP head to 128-d L2-normalized crops (detections from frame t, tracklets from t-1, lost from last seen); cosine affinity over temperature 0.07.",
association:
"Training: Sinkhorn with learnable dustbin on affinity S; inference: ByteTrack two-stage Hungarian with high threshold 0.6; 3-frame confirmation; 30-frame deletion.",
detectionDependency: "Oracle GT boxes for association study; simulated noise (15% FN, 10% FP, 10% jitter) for robustness; YOLOv8n COCO-only for end-to-end.",
trackManagement:
"High-confidence births after 3-frame confirmation; unmatched tracks held as Type-L up to 30 frames for re-ID; then deleted.",
loss:
"L = 1.0 (BCE on Sinkhorn block vs GT assignment, both D-T and D-L) + 0.5 triplet (margin 0.3); AdamW lr 1e-4, cosine annealing, 50 epochs.",
optimization:
"PyTorch 2.11 plus PyG 2.7, single RTX 5070 12 GB; detection augmentation: N(0,5px) jitter, 5% FP injection, 10% FN suppression.",
},
equations: [
{
id: "hdstgnn-altitude",
label: "Altitude proxy from object size",
formula: "z = -log(abar / aref) + eps, aref = 400 px^2",
variables: [
{ symbol: "abar", meaning: "mean apparent object area in current frame" },
{ symbol: "aref", meaning: "reference area at nominal altitude (400 px^2)" },
{ symbol: "z", meaning: "altitude proxy; positive means higher altitude (smaller objects)" },
],
intuition: "Small on-screen objects mean the drone is high; invert the size ratio into an altitude score.",
why: "Drives the adaptive radius without needing GPS or IMU.",
where: "Sec. 3.3.1, Eq. 2; eps = 1e-6; example z = 1.2 at 120 px^2, z = -0.8 at 900 px^2.",
paperIds: ["T126"],
},
{
id: "hdstgnn-radius",
label: "Altitude-adaptive connection radius",
formula: "reff = r0 * exp(-beta z), reff in [30, 500] px; r0 = 150, beta = 0.3",
variables: [
{ symbol: "reff", meaning: "effective graph edge radius for the frame" },
{ symbol: "r0", meaning: "base radius 150 px" },
{ symbol: "beta", meaning: "decay factor 0.3" },
],
intuition: "High altitude squeezes the neighborhood (dense tiny objects); low altitude stretches it.",
why: "Dominant contribution: fixing the radius costs 6.07 MOTA and collapses the high-variance sequence to 49.44%.",
where: "Sec. 3.3.1, Eq. 3; re-ID edges use 2x reff.",
params: "beta = 0 recovers the fixed-radius ablation.",
paperIds: ["T126"],
},
{
id: "hdstgnn-message",
label: "Occlusion-gated message",
formula: "m_{j->i} = alpha_ij * phi_j * Wmsg hj",
variables: [
{ symbol: "alpha_ij", meaning: "multi-head attention weight with edge-type bias we^T aij" },
{ symbol: "phi_j", meaning: "source occlusion confidence in [0,1]" },
{ symbol: "hj", meaning: "source node hidden state" },
],
intuition: "Listen to neighbors in proportion to attention, then turn down the volume on occluded or lost ones.",
why: "Stops corrupted embeddings from polluting the affinity matrix.",
where: "Sec. 3.4.2, Eqs. 5-6; 4 heads; node update aggregates per type then residual type-FFN (Eqs. 7-8).",
paperIds: ["T126"],
},
{
id: "hdstgnn-occlusion",
label: "Occlusion confidence",
formula: "phi = max(0.1, 1 - 0.1 dt) for tracklets; max(0.05, 0.5 exp(-0.08 dt)) for lost; phi = c for detections",
variables: [
{ symbol: "dt", meaning: "frames since the node was last matched" },
{ symbol: "c", meaning: "detector confidence for detection nodes" },
],
intuition: "Fresh matches are trustworthy; stale or lost nodes decay toward a floor.",
why: "The gate signal that implements contribution C3.",
where: "Sec. 3.4.3, Eq. 9; ablated by setting phi = 1.",
paperIds: ["T126"],
},
{
id: "hdstgnn-affinity",
label: "Affinity with temperature",
formula: "Sij = (ziD^T zjT) / tau, tau = 0.07",
variables: [
{ symbol: "ziD, zjT", meaning: "L2-normalized output embeddings of detection i and tracklet j" },
{ symbol: "tau", meaning: "temperature sharpening the affinity" },
],
intuition: "Cosine similarity sharpened so confident matches dominate Sinkhorn/Hungarian.",
why: "Converts refined graph embeddings into assignment costs for both match and re-ID matrices.",
where: "Sec. 3.5.1, Eq. 11; separate Sprime for lost-node re-ID.",
simulator: "reid",
paperIds: ["T126"],
},
{
id: "hdstgnn-loss",
label: "Joint matching plus metric loss",
formula: "L = 1.0 (BCE_DT + BCE_DL) + 0.5 (tri_DT + tri_DL); BCE = -mean(G log P + (1-G) log(1-P))",
variables: [
{ symbol: "P", meaning: "Sinkhorn soft assignment (dustbin block excluded)" },
{ symbol: "G", meaning: "ground-truth binary assignment" },
{ symbol: "tri", meaning: "triplet loss pulling matched pairs closer than random negatives by margin 0.3" },
],
intuition: "Supervise who-matches-whom directly while shaping the embedding space for re-ID.",
why: "End-to-end signal for graph, appearance and assignment head together.",
where: "Sec. 3.6, Eqs. 12-13; K = 20 log-space Sinkhorn iterations.",
paperIds: ["T126"],
},
{
id: "hdstgnn-inference",
label: "Two-stage Hungarian inference",
formula: "stage1: Hungarian(high-conf dets, ci >= 0.6); stage2: Hungarian(low-conf dets, unmatched tracklets)",
variables: [
{ symbol: "ci", meaning: "detection confidence" },
{ symbol: "Hungarian", meaning: "optimal one-to-one assignment per stage" },
],
intuition: "Confident boxes claim tracks first; leftovers rescue the occluded.",
why: "ByteTrack protocol giving the reported ID-switch reductions at inference while training stays differentiable.",
where: "Sec. 3.5.3; new tracks from unmatched high-conf dets; delete after 30 misses.",
simulator: "hungarian",
paperIds: ["T126"],
},
],
datasets: ["others"],
metrics: ["mota", "idf1", "hota", "idsw", "mt-ml", "fp", "fn"],
baselines: ["SORT", "DeepSORT", "ByteTrack", "StrongSORT", "NOWA-MOT"],
results: [
"Oracle detections, VisDrone val (Tab. 3): HDST-GNN 94.51 MOTA / 97.24 IDF1 / 28 IDs / 53 FP / 848 FN vs SORT 89.53/94.93/144/0/1493; MT 103 vs 91, ML 6 vs 8; wins all 7 sequences, best +9.68 on uav0000305.",
"Noisy detections (Tab. 5): HDST-GNN 12.88/52.02/638 IDs vs SORT 30.29/62.89/1264 IDs (-49% switches); SORT leads MOTA since 3-frame confirmation adds FNs (9234 vs 7841).",
"YOLOv8n end-to-end (Tab. 6): HDST-GNN 11.58/24.93/60 IDs vs SORT 13.57/28.54/117 IDs (-49%); detector finds only ~17% of objects so absolute MOTA stays low for both.",
"Ablation (Tab. 7, oracle): w/o C1 fixed radius 88.44 (-6.07 MOTA, 290 IDs); w/o C2 homogeneous 94.70 (+0.19); w/o C3 no gate 94.76 (+0.25); C2/C3 gaps within training noise (20 vs 50 epochs).",
],
ablations: [
"C1 altitude adaptation dominates: removal collapses sequence uav0000117 from 91.38% to 49.44% MOTA; radius spans ~70 px at high altitude to ~150 px near ground (Fig. 4).",
"C2 heterogeneous types and C3 occlusion gate show <0.3 pp MOTA change with inconsistent ID-switch trends at 20-epoch ablation depth; authors note full 50-epoch schedule may be needed.",
"Qualitative (Fig. 3): re-ID after occlusion and ID stability across altitude change where the baseline switches.",
],
limitations: {
authorStated: [
"Training uses GT boxes with simulated noise as pseudo-detections; joint training with a VisDrone-finetuned detector (e.g. YOLOv8x) is left as future work.",
"Altitude proxy comes only from object size without GPS/IMU, so sequences with unusually large or small categories give inaccurate estimates.",
"Throughput is ~10 FPS bottlenecked by sequential frame loading; DataLoader-worker extraction could approach real time; multi-class and UAVDT evaluation left for future work.",
],
evident: [
"Headline 94.51 MOTA hinges on oracle detections; with realistic detectors the same model scores 11-13 MOTA, so association gains are conditional on detection recall.",
"Ablation table contradicts the claimed independent complementary benefit of C2/C3 (both removals slightly improve MOTA), weakening that conclusion.",
],
},
assumptions: [
"Mean object area is a reliable altitude proxy within a sequence.",
"Appearance crops at 128x64 carry identity signal even for sub-10px aerial objects.",
],
computation:
"Single RTX 5070 12 GB; AdamW lr 1e-4, 50 epochs (ablations 20); ~10 FPS inference; hyperparameters Tab. 2 (H 256, L 3, 4 heads, K 20, tau 0.07).",
relations: [
{ to: "T005", type: "uses-as-baseline", note: "Canonical IoU-only SORT beaten 94.51 vs 89.53 MOTA with 81% fewer ID switches under oracle detections." },
{ to: "T061", type: "builds-on", note: "Two-stage high/low-confidence Hungarian inference follows the ByteTrack protocol." },
{ to: "T013", type: "uses-as-baseline", note: "DeepSORT published row 32.1 MOTA cited as real-detector context." },
{ to: "T067", type: "uses-as-baseline", note: "StrongSORT published row 44.7 MOTA cited as real-detector context." },
{ to: "T073", type: "uses-as-baseline", note: "OC-SORT motion-consistency line cited among TBD methods the graph affinity replaces." },
],
concepts: ["learned-association", "reid", "occlusion", "data-association"],
impact:
"Shows altitude-aware graph context is the decisive inductive bias for UAV MOT, while questioning how much heterogeneous typing and gating add beyond it.",
},
{
id: "T127",
arxiv: "2606.14094",
title: "FEMOT: Multi-Object Tracking using Frame and Event Cameras",
shortTitle: "FEMOT",
year: 2026,
authors: ["Shiao Wang", "Xiao Wang", "Chao Wang", "Yitao Li", "Menghao Liu", "Bo Jiang", "Yaowei Wang", "Yonghong Tian", "Jin Tang"],
fileName: "2606.14094v1.pdf",
task: "multimodal-tracking",
tags: ["rgb-event", "benchmark", "multimodal-tracking", "frequency-fusion", "transformer", "query-propagation", "tracking-by-detection"],
difficulty: "advanced",
summary:
"FEMOT fills the RGB-event MOT gap with a dataset (100 sequences, 200K frames, 14.58K trajectories, 0.42M boxes, people plus vehicles, 14 attributes from a DVS346 camera) plus a benchmark of 15 retrained trackers, and a method FEMOTR that fuses RGB and event features in the frequency domain (amplitude/phase modulation across RGB, event and joint branches) inside a MeMOTR-style query-propagation Transformer, reaching 48.9 HOTA and 52.0 MOTA on FEMOT vs 44.0/45.4 for MeMOTR.",
problem:
"Frame cameras fail under low light, overexposure and motion blur while event cameras supply complementary high-temporal-resolution cues, but RGB-event MOT lacked any large-scale diverse identity-annotated benchmark, and naive spatial fusion underuses the modalities complementary frequency structure.",
background: ["mot", "tracking-by-detection", "data-association", "multimodal", "attention", "end-to-end-mot", "memory-network", "reid", "benchmark-design"],
previousWork: [
{
name: "TBD trackers (ByteTrack, OC-SORT, MixSort, Hybrid-SORT, FairMOT, TraDes)",
limitation:
"Two-stage detect-then-match or joint embeddings rely on RGB appearance and hand-crafted association, degrading under adverse illumination and fast motion.",
whyThisPaper:
"Retrained with RGB-event fusion as benchmark lines; best TBD ByteTrack reaches only 36.3 HOTA vs FEMOTR 48.9.",
},
{
name: "Query-propagation Transformers (MOTRv2, MeMOTR, MOTIP, Samba)",
limitation:
"Temporal query memory improves association but remains RGB-bound with no event-fusion design.",
whyThisPaper:
"FEMOTR builds on MeMOTR, adding frequency-aware RGB-event fusion and reaching +4.9 HOTA over MeMOTR on FEMOT.",
},
{
name: "RGB-event SOT/detection fusion (FE108, VisEvent, COESOT, multi-stage fusion)",
limitation:
"SOT needs only per-frame localization and existing sets lack MOT identity trajectories; generic frequency fusions (FECNet, FCFE, FMAP) even degrade MOT metrics.",
whyThisPaper:
"FEMOT provides 14.58K identity trajectories, and the dedicated FAF module beats Add/Concat and generic frequency fusions on nearly all metrics.",
},
],
researchGap:
"No comprehensive diverse well-annotated RGB-event MOT dataset existed, and no fusion method explicitly modeled RGB low-frequency appearance vs event high-frequency contour structure for joint localization plus identity association.",
contribution: [
"FEMOT dataset: 100 DVS346 sequences, 200K frames, 14.58K tracks, 0.42M boxes (people, vehicles), 14 attributes, MOT20-compatible format with per-sequence event (dvs) and frame (aps) folders.",
"Benchmark: 15 state-of-the-art trackers retrained on FEMOT with feature-level RGB-event fusion, plus DSEC-MOT cross-evaluation.",
"FEMOTR: shared ResNet-50 backbone, frequency-aware fusion (RGB/event/joint branches with FFT amplitude-phase attention), deformable encoder, detection plus joint decoders, dynamic temporal interaction memory.",
"Analyses showing lowest adjacent-frame IoU and highest position-switch rate among MOT17/MOT20/VTMOT/DanceTrack, with entangled t-SNE re-ID features.",
],
method: {
pipeline: ["event-accumulation", "shared-backbone", "frequency-fusion", "deformable-encode", "detect-decode", "joint-decode", "memory-update"],
architecture:
"Shared ResNet-50 for RGB and event frames; FAF module (DWConv, FFT, conv amplitude/phase, attention modulation, iFFT plus residual, 3-branch sum); 6-layer deformable encoder; 300 detection queries; 3 detection plus 3 joint decoder layers; DTIM memory (EMA rate 0.01, MLP channel gating, memory-attention).",
appearanceModel:
"RGB low-frequency appearance/semantics plus event high-frequency motion/contour cues fused per scale; fused memory decoded by detection and track queries rather than explicit ReID embeddings.",
association:
"Query propagation: track embeddings from memory concatenated with detection embeddings in joint decoder; Hungarian bipartite matching of detection queries to unassociated GT during training; direct track-query prediction at inference.",
detectionDependency: "End-to-end DAB-Deformable-DETR init; no external detector; 300 learnable detection queries.",
trackManagement:
"DTIM maintains current plus historical plus long-term memory; newborn memory initialized from current output; EMA update with lambda 0.01.",
loss:
"L = 2 focal + 5 L1 + 2 GIoU; matching cost C = 2 cls + 5 L1 + 2 GIoU; AdamW lr 2e-4 decay 0.1 at epoch 12, 13 epochs, progressive clip lengths 2/3/4.",
optimization:
"PyTorch on A800-80GB GPUs, batch 1, weight decay 1e-4, random resize/crop augmentation, sampling interval 4.",
},
equations: [
{
id: "femot-eventaccum",
label: "Event accumulation window",
formula: "Ep_t = {e_i = (x_i,y_i,t_i,p_i) | t_i in [ts, te]}; events color-encoded to 3-channel frame E_t",
variables: [
{ symbol: "ts, te", meaning: "exposure start/end timestamps of RGB frame t" },
{ symbol: "p_i", meaning: "event polarity (ON/OFF)" },
{ symbol: "E_t", meaning: "frame-like event image aligned to RGB frame" },
],
intuition: "Slice the asynchronous event rain into per-RGB-frame piles so standard CNNs can consume both.",
why: "Temporal alignment prerequisite for every fused representation in the paper.",
where: "Sec. 3.2, Eq. 1; DVS346 streams are spatially aligned by hardware.",
paperIds: ["T127"],
},
{
id: "femot-fft",
label: "Frequency decoupling per branch",
formula: "Ub = DWConv(Bb); Sb = FFT(Ub); Ab = Conv(A(Sb)), Pb = Conv(P(Sb))",
variables: [
{ symbol: "Bb", meaning: "branch input: X (RGB), Y (event), or Z = Concat(X,Y)" },
{ symbol: "Ab, Pb", meaning: "amplitude and phase components" },
],
intuition: "Split each modality into what (amplitude) and where (phase) before letting them talk to each other.",
why: "Lets RGB contribute semantic amplitude and events contour phase selectively.",
where: "Sec. 3.4, Eqs. 2-5; applied independently per feature scale.",
paperIds: ["T127"],
},
{
id: "femot-attnmod",
label: "Amplitude/phase modulation",
formula: "Attn(V,W) = V + V * Softmax(V * W)",
variables: [
{ symbol: "V", meaning: "frequency component being enhanced" },
{ symbol: "W", meaning: "guidance component (self or fused-branch Abar_Z/Pbar_Z)" },
{ symbol: "*", meaning: "element-wise multiplication" },
],
intuition: "Amplify frequencies where the signal agrees with its guide, leave the rest.",
why: "Core fusion operator: self-modulates the joint branch, then cross-guides RGB and event branches.",
where: "Sec. 3.4, Eqs. 6-8; followed by projection, iFFT reconstruction plus residual (Eqs. 9-11) and 3-branch sum (Eq. 12).",
paperIds: ["T127"],
},
{
id: "femot-memory",
label: "Long-term memory update",
formula: "Mf_{t+1} = (1 - lam) M_t + lam * O_t; lam = 0.01",
variables: [
{ symbol: "M_t", meaning: "long-term memory embedding" },
{ symbol: "O_t", meaning: "current-frame output track embedding" },
{ symbol: "lam", meaning: "update rate 0.01" },
],
intuition: "Barely nudge a stable identity memory with each new frame so history dominates transients.",
why: "Stability mechanism behind association gains; newborns initialize from current output.",
where: "Sec. 3.5, Eq. 13; channel gating (Eq. 14-15) plus memory-attention produce next track queries.",
paperIds: ["T127"],
},
{
id: "femot-loss",
label: "Multi-task loss and matching cost",
formula: "L = 2 Lfocal + 5 L1 + 2 LGIoU; Cij = 2 Ccls + 5 CL1 + 2 Cgiou",
variables: [
{ symbol: "Lfocal", meaning: "sigmoid focal classification loss" },
{ symbol: "L1/LGIoU", meaning: "coordinate and overlap-geometry box losses" },
{ symbol: "Cij", meaning: "Hungarian matching cost between prediction i and target j" },
],
intuition: "Weight box geometry heaviest, classify with hard-example focus, match with the same recipe.",
why: "Standard detection-plus-tracking supervision; track queries supervised by identity, detection queries by unassociated GT.",
where: "Sec. 3.6, Eqs. 16-17.",
simulator: "hungarian",
paperIds: ["T127"],
},
{
id: "femot-deta-assa",
label: "DetA and AssA decomposition",
formula: "DetA = TPdet/(TPdet+FNdet+FPdet); AssA = TPAss/(TPAss+FNAss+FPAss)",
variables: [
{ symbol: "TPdet/FNdet/FPdet", meaning: "detection-level match counts" },
{ symbol: "TPAss/FNAss/FPAss", meaning: "association-level link counts" },
],
intuition: "Separate finding objects from keeping their identities so HOTA can balance both.",
why: "Paper's fine-grained lens: FEMOTR improves both DetA and AssA over MeMOTR.",
where: "Sec. 5.1, Eqs. 18-19; evaluated with TrackEval plus class-aware script.",
simulator: "metrics",
paperIds: ["T127"],
},
],
datasets: ["others"],
metrics: ["hota", "deta", "assa", "mota", "motp", "idf1"],
baselines: ["FairMOT", "TraDes", "ByteTrack", "MixSort-Oc", "MOTRv2", "OC-SORT", "MixSort-Byte", "MeMOTR", "TLDMOT", "Hybrid-SORT", "SRTrack", "PFTrack", "TOPICTrack", "Samba", "MOTIP", "GTR", "SiamMOT", "TrackFormer", "SpikeMOT"],
results: [
"FEMOT test (Tab. 2, HOTA/DetA/AssA/MOTA/MOTP/IDF1): Ours 48.9/45.2/53.8/52.0/76.7/62.0 vs best baseline MeMOTR 44.0/40.2/49.0/45.4/76.9/54.9; TBD ByteTrack 36.3/35.4/38.1/38.0/44.3; Samba 38.1/30.0/49.1/33.7/48.1.",
"DSEC-MOT (Tab. 3): Ours 52.4/40.6/67.8/37.5/59.0 - best AssA of all; Samba leads HOTA 55.4 and IDF1 64.9; event-only SpikeMOT leads DetA 49.5 and MOTA 54.7, so detection caps overall scores.",
"Dataset scale (Tab. 1): 100 videos, 200K frames, 14.58K tracks, 0.42M boxes vs DSEC-MOT 12/23.08K/0.50K/0.037M.",
"Attributes (Fig. 5): FEE 96, SV 88, FM 67, DS 51, SO 50 most frequent of 14; long-tailed small-target size distribution.",
],
ablations: [
"Components (Tab. 4): w/o shared multi-scale weights 40.5 HOTA (-8.4, largest drop); w/o FAF 44.0 (= MeMOTR baseline); w/o modality branches 45.5; w/o amplitude/phase attention 48.3 HOTA but slightly higher DetA/MOTA (association-specific benefit).",
"Fusion strategies (Tab. 5): FAF 48.9/62.0 beats Add 47.1/59.1, Concat 45.5/57.3, FECNet 39.2/48.7, FCFE 41.0/51.0, FMAP 40.8/51.1 (generic frequency fusions degrade).",
"Modalities (Tab. 6): dual 48.9/62.0 vs RGB-only 42.5/53.9 vs event-only 14.3/11.8; event alone insufficient for detection (DetA 6.6).",
"Attributes (Fig. 8): best or competitive on most, clearest wins on HE, SO, SA, DB.",
],
limitations: {
authorStated: [
"No dynamic event-density adaptation: fast targets give dense streams and slow ones sparse, so adapting the accumulation window to motion speed is future work.",
"Language modality unused: descriptions of category, spatial relations and motion intent could further aid localization and association.",
],
evident: [
"On DSEC-MOT the method trails event-only SpikeMOT and Samba on HOTA/IDF1/MOTA, so the frequency-fusion advantage is dataset-conditional and detection-bound.",
"All 15 baselines were extended to RGB-event by the authors with feature-level fusion, so baseline numbers reflect re-implementations rather than original published configurations.",
],
},
assumptions: [
"Fixed per-RGB-frame accumulation windows fairly represent event streams across motion speeds.",
"DVS346 hardware alignment holds across day/night, rotation, shake and vehicle-mounted captures.",
],
computation:
"A800-80GB GPUs; 13 epochs batch 1; 6 encoder plus 6 decoder layers; 300 queries; clip lengths 2/3/4 at epochs 0/6/10; code plus dataset released (Event-AHU/FEMOT).",
relations: [
{ to: "T061", type: "uses-as-baseline", note: "ByteTrack retrained with RGB-event fusion as TBD baseline: 36.3 HOTA vs 48.9." },
{ to: "T043", type: "uses-as-baseline", note: "FairMOT retrained as joint detection-tracking baseline: 22.1 HOTA." },
{ to: "T073", type: "uses-as-baseline", note: "OC-SORT retrained as motion-based baseline: 23.1 HOTA FEMOT, 27.7 DSEC-MOT." },
],
concepts: ["multimodal", "mot", "end-to-end-mot", "memory-network", "benchmark-design"],
impact:
"First large-scale RGB-event MOT dataset plus benchmark, establishing frequency-domain fusion as the reference approach and a retraining protocol for RGB trackers on event data.",
},
{
id: "T128",
arxiv: "2606.23604",
title: "Polycepta: Object-Centric Appearance Estimation for Multi-Object Tracking",
shortTitle: "Polycepta",
year: 2026,
authors: ["Mohamed Nagy", "Naoufel Werghi", "Jorge Dias", "Majid Khonji"],
fileName: "2606.23604v4.pdf",
task: "multi-object",
tags: ["tracking-by-detection", "appearance-modeling", "state-space", "re-identification", "cross-category", "real-time", "fourier"],
difficulty: "advanced",
summary:
"Polycepta reformulates appearance from static frame-wise descriptors into a recursive per-object appearance state: MobileNetV3-small embeddings projected to d = 24 evolve through Fourier-domain relational reasoning (VRR), an adaptive U-gate update, and next-frame estimation, trained with contrastive plus orthogonality losses under per-epoch state erasure. Plugged into RobMOT/FastTracker without touching motion pipelines, it improves during inference (median similarity 0.950 to 0.969 while ReID degrades), reaching 92.27 MOTA on KITTI at 90.57 Hz with cross-category gaps under 0.8%.",
problem:
"TBD appearance cues are static frame-independent ReID vectors that assume frozen looks and cost heavy backbones, so real-time systems drop them entirely; no recursive estimator existed that accumulates visual identity the way the Kalman filter accumulates motion.",
background: ["mot", "tracking-by-detection", "kalman", "state-space", "motion-model", "appearance-features", "reid", "cost-matrix", "data-association"],
previousWork: [
{
name: "Static ReID appearance TBD (feature-extraction plus matching frameworks)",
limitation:
"Frame-by-frame descriptors assume static appearance, break under occlusion and viewpoint change, and need ResNet-50-class backbones.",
whyThisPaper:
"Recursive appearance states improve over time (margin 0.045 to 0.290, ID switches 9792 to 247) from a MobileNetV3-small backbone.",
},
{
name: "Motion-only real-time trackers (RobMOT, MCTrack, FastTracker, KF-based 2D/3D)",
limitation:
"Forgo visual cues for latency, losing association quality under ambiguity.",
whyThisPaper:
"Drop-in appearance costs complement unchanged motion pipelines: FastTracker AssA +3.34% and 185 fewer switches on MOT17.",
},
{
name: "Tracking-by-propagation (MeMOT, MeMOTR, Samba/Mamba)",
limitation:
"Propagate trajectories with memory/attention but remain heavy and outside the dominant real-time TBD paradigm.",
whyThisPaper:
"Stays inside TBD with 90.57 Hz full-pipeline throughput, targeting accuracy-efficiency tradeoff instead.",
},
],
researchGap:
"Appearance had no state-estimation counterpart to the Kalman filter: no lightweight recursive formulation that accumulates identity evidence and predicts future representations.",
contribution: [
"Object-centric appearance-state estimation: per-track state Ht in d = 24 updated recursively and used to estimate next-frame descriptors Xhat_{t+1}.",
"VRR module: FFT circular cross-correlation between observations and history with learnable complex filter, O(d log d), batch-parallel, plus gated cross-appearance fusion.",
"Adaptive U-gate update plus state-erasure training (H0 = 0 each epoch) with contrastive and orthogonality losses balanced by EMA-normalized dynamic weighting.",
"Plug-in gated fusion with trajectory costs handling asymmetric detector availability; proven cross-category transfer and 90.57 Hz operation.",
],
method: {
pipeline: ["crop-embed", "project-discretize", "relational-reasoning", "gated-update", "appearance-estimate", "cost-fusion", "associate"],
architecture:
"MobileNetV3-small (576-d) plus LayerNorm projections to ds = 24; VRR with FFT, learnable complex filter, latent dl = 6 fusion gate; U-gate updater; Ct selection plus Wproj estimation head with residual Dt; trained on KITTI/WOD/MOT17/MOT20 object sequences.",
appearanceModel:
"Recursive state Ht per object replaces static ReID: relational features St from VRR plus gated blend of incoming psi_t and transitioned Htilde_t; Xhat_{t+1} from Ct-selected state plus residual.",
association:
"Cosine similarity Mp = Xt Xhat^T clamped at beta_p, converted to cost 1 - Mp, normalized and selectively fused with trajectory cost via availability gating (alpha prior when bimodal).",
detectionDependency: "Host frameworks detectors unchanged (CasA/VirConv/PV-RCNN/PointRCNN/Second 3D, YOLOv8/vx 2D); 2D+3D fusion evaluated.",
trackManagement: "Host TBD management unchanged; Polycepta consumes matched pairs and predicts next-step appearance per active track.",
loss:
"Contrastive Lf (same-object attraction, cross-object repulsion, Eq. 14) plus orthogonality Lorth on HH^T/tau_h off-diagonals (Eq. 15), EMA-balanced; MobileNetV3 warmed 5 epochs.",
optimization:
"PyTorch from scratch, TorchScript to C++17 for RobMOT; trainable weights reset states each epoch; component studies at 5 epochs.",
},
equations: [
{
id: "polycepta-project",
label: "Projection and discretization inputs",
formula: "zt = LN(Xt); dt = softplus(Wd zt + bd); Bt = LN(WB zt + bB); Ct = WC zt + bC",
variables: [
{ symbol: "Xt", meaning: "576-d MobileNetV3 embeddings for n objects" },
{ symbol: "dt", meaning: "learned time step sizing the state update" },
{ symbol: "Bt", meaning: "projected input in 24-d state space" },
{ symbol: "Ct", meaning: "selection vector extracting state features for estimation" },
],
intuition: "Compress each crop into a small state-space dialect plus knobs controlling how fast the state may change.",
why: "Entry point mapping high-dim vision into the temporal state dynamics.",
where: "Sec. IV, Eq. 1; ds = 24 throughout.",
paperIds: ["T128"],
},
{
id: "polycepta-transition",
label: "HiPPO-plus-diagonal transition and discretization",
formula: "At = Ahippo - diag(exp(theta)); Abar = I + dt At; Bbar = dt Bt",
variables: [
{ symbol: "Ahippo", meaning: "fixed structured memory matrix" },
{ symbol: "theta", meaning: "learnable diagonal adaptation" },
{ symbol: "Abar, Bbar", meaning: "discrete-time transition and input matrices" },
],
intuition: "A principled fading memory plus a learned per-dimension tweak, stepped forward by the learned dt.",
why: "Governs how appearance information accumulates vs decays over time.",
where: "Sec. IV, Eqs. 2-3; follows standard SSM discretization.",
paperIds: ["T128"],
},
{
id: "polycepta-vrr",
label: "Fourier relational reasoning",
formula: "F(Bbar) = FFT(Bbar); F(H) = FFT(Ht-1); F(I) = F(H) * F(Bbar) * Phi; I = LN(iFFT(F(I)))",
variables: [
{ symbol: "Phi", meaning: "learnable complex-valued filter per object (Eq. 4)" },
{ symbol: "I", meaning: "interaction tensor between current observation and history" },
],
intuition: "Correlate now vs accumulated past in frequency space to find agreement cheaply and in parallel.",
why: "VRR cuts ID switches 247 to 222 and lifts late-horizon quality; O(ds log ds) per object, batch-vectorized.",
where: "Sec. V-A, Eq. 5; compressed to dl = 6 latent then gated (Eqs. 6-7).",
paperIds: ["T128"],
},
{
id: "polycepta-update",
label: "U-gate appearance-state update",
formula: "Htilde = Abar Ht-1; psi = tanh(LN(WK(Bbar + S))); U = sig(LN(Wu[Ht-1; psi])); H = LN(U*psi + (1-U)*Htilde)",
variables: [
{ symbol: "S", meaning: "VRR relational features" },
{ symbol: "U", meaning: "element-wise gate favoring fresh input early, memory when noisy" },
{ symbol: "H", meaning: "updated object-centric appearance state" },
],
intuition: "Trust new looks when the state is young, trust memory when the frame is occluded; blend per element.",
why: "U-gate lifts final quality (P50 0.969 to 0.975, P10 0.933 to 0.947) and cuts switches 247 to 204.",
where: "Sec. VI, Eqs. 8-11; gate evolution visualized over 10 frames (Fig. 4).",
paperIds: ["T128"],
},
{
id: "polycepta-estimate",
label: "Next-appearance estimation",
formula: "Y = LN(Ct * H); Xhat_{t+1} = (Dt * Xt) + (Wproj Y)",
variables: [
{ symbol: "Y", meaning: "Ct-selected state features" },
{ symbol: "Dt", meaning: "learnable residual on current observation" },
{ symbol: "Xhat", meaning: "predicted next-frame 576-d appearance descriptor" },
],
intuition: "Read out the future look from accumulated state, anchored by what is visible now.",
why: "This prediction is the association cue replacing static ReID vectors.",
where: "Sec. VII, Eq. 12; Xhat used in Eq. 16 similarity.",
paperIds: ["T128"],
},
{
id: "polycepta-sim",
label: "Appearance association similarity",
formula: "Mp = Xt Xhat^T; Mpprime = max(Mp, beta_p)",
variables: [
{ symbol: "Mp", meaning: "cosine similarity between current and predicted appearances" },
{ symbol: "beta_p", meaning: "minimum appearance-similarity threshold" },
{ symbol: "Xhat", meaning: "Polycepta predictions for m tracks" },
],
intuition: "Match what you see now against what each track said it would look like.",
why: "Turns improving estimates into fewer mismatches over time.",
where: "Sec. IX, Eqs. 16-17; converted to cost Cpprime = 1 - Mpprime (Eq. 19).",
simulator: "reid",
paperIds: ["T128"],
},
{
id: "polycepta-fuse",
label: "Selective trajectory-appearance fusion",
formula: "Ctilde_k = Ckprime/beta_k; Ctilde_p = Cpprime/(1-beta_p); Cfuse = Wk*Ctilde_k + Wp*Ctilde_p",
variables: [
{ symbol: "Ckprime", meaning: "trajectory cost clamped at beta_k (Eq. 18)" },
{ symbol: "Wk, Wp", meaning: "availability-gated weights: (alpha,1-alpha) bimodal, (1,0)/(0,1) single (Eq. 21)" },
{ symbol: "alpha", meaning: "tunable bimodal balance prior" },
],
intuition: "Put motion and appearance costs on one scale, then listen to whichever sensor actually fired.",
why: "Handles LiDAR/camera FoV and detector asymmetry without changing host assignment.",
where: "Sec. IX, Eqs. 18-22; Hadamard fusion.",
simulator: "assoc-cost",
paperIds: ["T128"],
},
{
id: "polycepta-erasure",
label: "State-erasure training",
formula: "H0 = 0_{(n x ds)} at every epoch start",
variables: [
{ symbol: "H0", meaning: "all appearance states reinitialized to zero" },
],
intuition: "Wipe memory each epoch so the model must re-learn how to build states, not memorize training objects.",
why: "Erasure cuts unseen-class switches 144 to 118 and stabilizes the margin (0.188 vs 0.169).",
where: "Sec. VIII, Eq. 13; orthogonality loss Eq. 15 keeps states distinct (collapse to 1.0 similarity without it).",
paperIds: ["T128"],
},
],
datasets: ["kitti-tracking", "mot17", "mot20", "others"],
metrics: ["hota", "deta", "assa", "mota", "motp", "idsw", "mt-ml", "fps"],
baselines: ["RobMOT", "MCTrack", "FastTracker", "TrackTrack", "PC3TMOT", "PolarMOT", "UG3DMOT", "JMODT", "DeepFusionMOT", "MMF-JDT", "MobileNetV3-ReID"],
results: [
"KITTI test (Tab. II, HOTA/DetA/AssA/MOTA/IDSw/FN): RobMOT+Polycepta 3D 81.28/77.75/85.61/90.84/5/1543 vs RobMOT 80.83/76.99/85.50/89.83/6/1544; 2D+3D fusion 82.05/80.16/84.56/92.27/73/823 - SOTA MOTA.",
"MOT17 (Tab. III): FastTracker+Polycepta 75.12/88.69/73.31 HOTA/MOTA/AssA with 382 switches vs 73.04/88.34/69.97 with 567 (-185).",
"KITTI-2D cars: RobMOT+Polycepta 84.54/92.18 (+1.48/+2.61) with IDSw 1 to 3 (+2 from unannotated YOLOv8 detections, as authors note).",
"WOD vehicles: 57.31/52.46 with 444 vs 501 switches (-57); DetA/AssA gains transfer to 3D point-cloud evaluation from image observations.",
"Detectors (Tab. IV): MOTA gains on all five 3D detectors, largest on weak Second 85.85 to 90.75 (+4.90); VirConv ML 14 to 3, MT 159 to 174.",
"Cross-category (Tab. V): vehicle-trained on pedestrians 92.33 vs 93.05 (-0.72 MOTA); pedestrian-trained on vehicles 88.63 vs 88.69 (-0.06).",
"Throughput (Tab. VIII): full pipeline 90.57 Hz (feature 124.78, Polycepta 377.16, association 2713.25 Hz) on RTX 3080 laptop.",
],
ablations: [
"Appearance evolution (Tab. I, 148 MOT20 objects, 1000 frames): ReID median 0.493 to 0.472 down; Polycepta 0.950 to 0.969 up, P10 0.832 to 0.933 up, margin 0.045 to 0.290, switches 9792 to 247.",
"VRR adds: switches 247 to 222, final P50 0.969 to 0.971; U-gate adds: P50 to 0.975, P10 to 0.947, switches to 204, but margin compresses 0.290 to 0.250; full both: 184 switches, margin 0.300.",
"Orthogonality (Tab. VI): without Lorth states collapse to 1.0 similarity; with it states span [-0.9962, 0.9720] and stay discriminative at frame 10 (-0.8051).",
"State erasure (Tab. VII): seen switches 224 to 184, unseen 144 to 118 with steadier margin, supporting construction-over-memorization.",
],
limitations: {
authorStated: [
"KITTI-2D IDSw rises 1 to 3 because the extra YOLOv8 detector fires on objects unannotated in ground truth, creating counted false positives.",
"No dedicated limitations section; ablations themselves show VRR slightly degrades early-track quality and U-gate alone compresses the discrimination margin.",
],
evident: [
"Per-object state memory and FFT reasoning scale with object count; crowded-scene throughput beyond MOT20-148-object test is unreported.",
"All gains are measured inside host TBD pipelines and detectors, so appearance benefits are conditional on host detection and motion quality.",
],
},
assumptions: [
"Temporally ordered object crops with GT identity are available for state-erasure training.",
"Host TBD association accepts an auxiliary appearance cost matrix without modification.",
],
computation:
"AMD Ryzen 9 plus RTX 3080 laptop 16 GB; PyTorch plus TorchScript C++17; MobileNetV3 warmed 5 epochs; studies at 5 epochs; full pipeline 90.57 Hz.",
relations: [],
concepts: ["reid", "appearance-features", "memory-network", "track-management"],
impact:
"Introduces appearance-state estimation as a reusable TBD plug-in with the demonstrated property that estimates improve during inference, plus a cross-category generalization protocol.",
}
];
