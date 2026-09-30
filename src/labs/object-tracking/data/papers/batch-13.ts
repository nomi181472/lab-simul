import type { PaperRecord } from "../types";

/* batch-13 — T109..T118 */

export const BATCH_13: PaperRecord[] = [
{
id: "T109",
arxiv: "2407.10485",
title: "MM-Tracker: Motion Mamba with Margin Loss for UAV-platform Multiple Object Tracking",
shortTitle: "MM-Tracker",
year: 2024,
authors: ["Mufeng Yao", "Jinlong Peng", "Qingdong He", "Bo Peng", "Hao Chen", "Mingmin Chi", "Chao Liu", "Jon Atli Benediktsson"],
venue: "AAAI 2025",
fileName: "2407.10485v3.pdf",
task: "multi-object",
tags: ["uav-tracking", "tracking-by-detection", "motion-prediction", "state-space-model", "mamba", "margin-loss", "motion-blur"],
difficulty: "intermediate",
summary:
"MM-Tracker tackles UAV multi-object tracking where global camera motion and motion blur break Kalman and local-only motion models. It reuses bi-temporal YOLOX detection features in a Motion Mamba module (local cross-correlation plus bidirectional horizontal/vertical selective scans) to predict a 2-channel motion map, and trains the detector classification branch with a Motion Margin loss that demands larger decision margins for fast-moving objects. It reports 44.7 MOTA / 58.3 IDF1 on VisDrone and 51.4 / 68.9 on UAVDT at 31.1 FPS.",
problem:
"UAV-MOT contains both local object motion and global camera motion, plus motion blur that makes fast-moving objects hard to detect. Kalman filters assume linear motion and learn nothing, while learned motion models use local convolution or cross-correlation and miss global displacement; blurred large-motion objects are also rare in the data so they are under-trained.",
background: ["mot", "tracking-by-detection", "motion-model", "kalman", "data-association", "detection"],
previousWork: [
{
name: "Kalman-filter motion models (SORT, OC-SORT and variants)",
limitation:
"Non-learnable with pre-assumed linear motion, so accuracy collapses under human-made global camera motion.",
whyThisPaper:
"Motion Mamba learns global displacement from bi-temporal detection features with linear-time selective scans, beating the EKF baseline 40.9 vs 39.6 MOTA at 6.9 vs 15.2 ms.",
},
{
name: "Local learned motion (SiamMOT, FOLT, optical-flow prediction)",
limitation:
"Local correlation/convolution plus repeated feature extraction from raw images: redundant compute and no global modelling; direct flow networks cost ~115 ms.",
whyThisPaper:
"Feature reuse plus V-SSM/H-SSM global scans match EMD-Flow accuracy (40.9 vs 40.7 MOTA) at 1/16 the motion-inference time (6.9 vs 115 ms).",
},
{
name: "Standard and long-tail classification losses (cross-entropy, focal, LDAM)",
limitation:
"Class-imbalance losses do not model the motion-blur long tail, so rarely-seen fast objects stay under-detected.",
whyThisPaper:
"MMLoss assigns motion-dependent decision boundaries and lifts detection of blurred objects: 42.1 vs 40.3 (LDAM) and 36.4 (focal) MOTA.",
},
],
researchGap:
"No UAV tracker combined linear-time global motion modelling with a loss that explicitly compensates the motion-blur long tail.",
contribution: [
"Motion Mamba module: local cross-correlation of bi-temporal detection features fused with bidirectional Mamba (V-SSM plus H-SSM) scans, multi-scale fusion upward to a 1/8-resolution 2-channel (horizontal, vertical) motion map.",
"Motion Margin loss (MMLoss): sigmoid-margin classification loss D(x) with s=10 that imposes a larger boundary for larger object offsets, saturating above ~30 pixels.",
"Feature-reuse design: motion estimated from existing detector features at 3 scales instead of re-extracting from raw images, cutting motion-inference time to 6.9 ms.",
"State of the art on two public UAV-MOT sets: 44.7 MOTA / 58.3 IDF1 VisDrone and 51.4 / 68.9 UAVDT at 31.1 FPS with YOLOX-S.",
],
method: {
pipeline: ["detect", "extract-bitemporal-features", "local-correlation", "global-mamba-scan", "predict-motion-map", "propagate-boxes", "spatial-match"],
architecture:
"YOLOX-S backbone plus head (DetBackbone/DetHead, 1/8, 1/16, 1/32 features); Motion Mamba blocks (V-SSM, H-SSM, add fusion, shortcut) per scale with stepwise upsampling fusion to a 1/8 motion map.",
motionModel:
"Learned global-plus-local: per-object motion read from the predicted motion map at its center and added to propagate the previous box; supervised by a ground-truth map built from EMD-Flow optical flow overlapped with annotated box offsets.",
appearanceModel:
"None: association is motion plus spatial matching; blurred similar vehicle appearances are declared unreliable for UAV scenes.",
association:
"Motion Estimation and Spatial Matching: propagated previous-frame positions matched to current detections (baseline described as extended Kalman Filter plus two-stage spatial matching).",
detectionDependency: "YOLOX-S, input 1088x608, full ten VisDrone categories trained, five evaluated (car, bus, truck, pedestrian, van); UAVDT car/truck/bus via VisDrone toolkit.",
trackManagement:
"Propagate-then-match per frame; no explicit birth/death ages stated beyond the two-stage spatial matching scheme.",
loss:
"L1 loss on the motion map and on the detector regression branch (plus IoU loss); proposed MMLoss on the classification branch.",
optimization:
"SGD, learning rate 0.0001, batch size 8, 10 epochs per dataset, single 2080TI GPU.",
},
equations: [
{
id: "mmt-ssm",
label: "State-space selective scan",
formula: "h_t = A_hat h_{t-1} + B_hat x_t;  y_t = C h_t + D x_t",
variables: [
{ symbol: "x_t", meaning: "input feature vector at scan position t (one pixel vector of size C)" },
{ symbol: "h_t", meaning: "hidden state carrying long-range context along the scan" },
{ symbol: "y_t", meaning: "output feature at position t" },
{ symbol: "C, D", meaning: "output projection matrices" },
{ symbol: "A_hat, B_hat", meaning: "discretized state matrices (input-dependent, selective)" },
],
intuition: "Sweep the feature map pixel by pixel while a memory vector accumulates what was seen far away, so each output knows about distant motion.",
why: "Gives Transformer-like global context with linear-time scans and parallel training instead of O(n^2) attention.",
where: "Sec. Motion Mamba Module, Eq. 1; applied as V-SSM (W column scans) and H-SSM (H row scans), then added.",
simulator: "motion",
paperIds: ["T109"],
},
{
id: "mmt-discrete",
label: "Input-dependent discretization",
formula: "A_hat = exp(A dt);  B_hat = B dt;  dt = MLP(x_t)",
variables: [
{ symbol: "A, B", meaning: "learnable continuous-time state matrices" },
{ symbol: "dt", meaning: "input-dependent step size from an MLP over x_t (the selective mechanism)" },
{ symbol: "MLP", meaning: "multi-layer perceptron predicting the step from the current input" },
],
intuition: "Let the current pixel decide how fast memory decays and how much new input enters, so the scan pays attention selectively.",
why: "Input dependence is what makes Mamba selective rather than a fixed linear filter.",
where: "Sec. Motion Mamba Module, Eq. 2.",
paperIds: ["T109"],
},
{
id: "mmt-margin",
label: "Motion margin function",
formula: "D(x) = s (1 / (1 + e^(-(x-5)/s))) - M;  M = s (1 / (1 + e^(5/s)))",
variables: [
{ symbol: "x", meaning: "object center offset between previous and next frames (pixels of motion)" },
{ symbol: "s", meaning: "slope parameter, set to 10" },
{ symbol: "D(x)", meaning: "motion margin subtracted from the classification logit" },
{ symbol: "M", meaning: "zero-motion offset so D(0) = 0" },
],
intuition: "Zero margin for still objects, growing margin for fast ones, flattening past ~30 pixels where blur is already severe.",
why: "Forces the detector to output higher scores for rarely-seen fast blurred objects to cross the decision boundary.",
where: "Sec. Motion Margin Loss, Eqs. 3-4 and Fig. 5; s=10 chosen because ~30-pixel motion causes heavy blur.",
params: "Larger s flattens the curve; with s=10 the margin saturates above motion 30.",
paperIds: ["T109"],
},
{
id: "mmt-loss",
label: "Motion Margin loss",
formula: "MMLoss(y_i, y_hat_i, D_i) = -y_i log(sigma(y_hat_i - D_i)) - (1 - y_i) log(1 - sigma(y_hat_i))",
variables: [
{ symbol: "y_hat_i", meaning: "predicted classification logit for box i" },
{ symbol: "y_i", meaning: "ground-truth label for box i" },
{ symbol: "D_i", meaning: "motion margin of box i from D(x)" },
{ symbol: "sigma", meaning: "sigmoid function" },
],
intuition: "Shift the positive-class boundary rightward by the object's own motion, so fast objects must be scored higher to count as detected.",
why: "Rebalances training toward the under-trained large-motion tail without resampling.",
where: "Sec. Motion Margin Loss, Eq. 5; supervises the detector classification branch only.",
paperIds: ["T109"],
},
],
datasets: ["uavdt", "others"],
metrics: ["mota", "idf1", "fps"],
baselines: ["SiamMOT", "FairMOT", "ByteTrack", "UAVMOT", "OC-SORT", "FOLT", "U2MOT", "TrackSSM", "FastFlowNet", "GMA", "EMD-Flow"],
results: [
"VisDrone test (Tab. 5): MM-Tracker 44.7 MOTA / 58.3 IDF1; UAVDT test: 51.4 / 68.9; 31.1 FPS, best in both accuracy and speed.",
"VisDrone vs SOTA (Tab. 5): SiamMOT 31.9/48.3, FairMOT 34.3/46.1, ByteTrack 35.7/37, UAVMOT 36.1/51, OC-SORT 39.6/50.4, FOLT 42.1/56.9, U2MOT 42.8/53.9, TrackSSM 41.9/55.3 (MOTA/IDF1); UAVDT: OC-SORT 47.5/64.9, FOLT 48.5/68.3, U2MOT 47.1/65.2, TrackSSM 48.1/65.4 vs ours 51.4/68.9.",
"Motion modelling (Tab. 2): OC-SORT 39.6/50.4/15.2 ms; FastFlowNet 40.4/53.1/11 ms; GMA 40.5/53.5/102 ms; EMD-Flow 40.7/54.0/115 ms; Motion Mamba 40.9/54.5/6.9 ms.",
"Combination (Tab. 1): baseline 39.6/50.4 (VisDrone MOTA/IDF1) and 47.5/63.1 (UAVDT); plus MMLoss 42.1/54.4 and 49.2/64.1; plus Mamba 40.9/54.5 and 49.7/66.1; both 44.7/58.3 and 51.4/68.9 at 6.9 ms motion time.",
],
ablations: [
"Scan directions (Tab. 3): baseline 39.6/50.4; Local 40.2/52.5; Local+V-SSM 40.5/53.0; Local+H-SSM 40.5/53.5; Local+both 40.9/54.5 (MOTA/IDF1).",
"Loss functions (Tab. 4): cross-entropy 39.6/50.4; focal 36.4/48.3; LDAM 40.3/51.0; MMLoss 42.1/54.4.",
"MMLoss score effect (Fig. 9): with MMLoss the average detection score of large-motion objects rises above 0.5, vs below without it.",
"Qualitative: Motion Mamba recovers three fast tricycles (IDs 20, 25, 47) missed without it (Fig. 6); MMLoss recovers a blurred bike-rider (ID 38) missed without it (Fig. 7).",
],
limitations: {
authorStated: [
"No explicit limitation statement is given; the conclusion reports only gains on VisDrone and UAVDT.",
],
evident: [
"Only MOTA/IDF1/FPS reported, so identity vs detection trade-offs (HOTA/AssA/IDSW) are invisible.",
"Margin design (s=10, ~30-pixel saturation) is tuned to UAV blur statistics and may not transfer to non-UAV scenes.",
],
},
assumptions: [
"Object motion between adjacent frames is readable from detector-feature correlation plus global scans.",
"Large inter-frame offset is a reliable proxy for motion-blur detection difficulty.",
"YOLOX-S detections plus propagated-position spatial matching suffice; no appearance cues used.",
],
computation:
"YOLOX-S, 1088x608 input, SGD lr 0.0001, batch 8, 10 epochs, single 2080TI; motion inference 6.9 ms vs 115 ms (EMD-Flow); full tracker 31.1 FPS.",
relations: [
{ to: "T073", type: "improves", note: "EKF-based OC-SORT-style baseline beaten 44.7 vs 39.6 MOTA on VisDrone." },
{ to: "T061", type: "uses-as-baseline", note: "ByteTrack two-stage matching lineage; beaten 44.7 vs 35.7 MOTA VisDrone." },
{ to: "T043", type: "uses-as-baseline", note: "FairMOT compared in Tab. 5; beaten 44.7 vs 34.3 MOTA VisDrone." },
{ to: "T060", type: "uses-as-baseline", note: "SiamMOT compared in Tab. 5; beaten 44.7 vs 31.9 MOTA VisDrone." },
],
concepts: ["motion-model", "tracking-by-detection", "data-association"],
impact:
"Pushed UAV-MOT toward global motion modelling with linear-time state-space scans and introduced motion-aware margin training for blurred fast objects.",
},
{
id: "T110",
arxiv: "2407.15707",
title: "Predicting the Best of N Visual Trackers",
shortTitle: "BofN",
year: 2024,
authors: ["Basit Alawode", "Sajid Javed", "Arif Mahmood", "Jiri Matas"],
fileName: "2407.15707v1.pdf",
task: "single-object",
tags: ["single-object-tracking", "meta-tracker", "tracker-selection", "performance-prediction", "self-supervised-learning", "vision-transformer", "benchmarking"],
difficulty: "intermediate",
summary:
"No single visual tracker wins everywhere: across LaSOT attributes and datasets the best tracker changes per sequence. The paper builds a Tracking Performance Prediction Network (TP2N) that classifies a video into its best of N=17 trackers from early frames only, then runs just that tracker (video-level) or re-predicts every 5 frames (frame-level). The frame-level BofN reaches 81.7 AUC on LaSOT, 92.1 on TrackingNet, 91.1 AO on GOT-10k and 70.98 EAO on VOT2022.",
problem:
"SOTA trackers vary surprisingly strongly across video attributes and datasets (Fig. 1: every one of 8 trackers is best on at least 4% of sequences; no tracker leads all 14 LaSOT attributes), yet fusion approaches must run all trackers at inference. The gap is a selector that predicts the winner without executing the pool.",
background: ["sot", "bounding-box", "iou", "success-plot", "benchmark-design", "siamese", "attention"],
previousWork: [
{
name: "Early fusion of complementary features",
limitation:
"A fixed fused feature set cannot be optimal per scenario, and finding the optimal per-scenario subset is computationally intractable.",
whyThisPaper:
"BofN skips fusion entirely and selects one whole tracker predicted best for the scene.",
},
{
name: "Late fusion and decision-level ensembles (MEEM, HCF, MCCT)",
limitation:
"Multiple tracker responses must be estimated in parallel per sequence, raising computational complexity.",
whyThisPaper:
"Video-level BofN runs TP2N once (0.84 s overhead) plus a single tracker, and still beats every individual tracker.",
},
{
name: "Tracker ensembling with late fusion (ACFN, PTAV, confidence/drift-based selection)",
limitation:
"All trackers must be executed at inference and there is no mechanism to predict a tracker's performance without running it.",
whyThisPaper:
"TP2N predicts the winner from initial frames only, so the 17 trackers are never executed for video-level selection.",
},
],
researchGap:
"No mechanism existed to predict which tracker will perform best on a given sequence without running all candidates.",
contribution: [
"Empirical finding: no paradigm dominates SOT; per-attribute best-tracker shares quantified on LaSOT, UAV123 and VOT2022 for 8 diverse trackers.",
"Ground-truth label generation: run N=17 SOTA trackers per training video, build AUC vector a_j, L2-normalize to p_j and one-hot encode the argmax as the classification label.",
"TP2N: fine-tuned SSL backbones (MoCoV2, SwAV, Barlow Twins on ResNet-50; DINO ViT-S p=8/16) with an MLP classifier; DINO ViT-S wins (up to 96.6 percent top-1 video-level).",
"BofN meta-tracker in video-level (predict once) and frame-level (re-predict every 5 frames) modes, plus VOT-specific augmentation (temporal/spatial subsampling, backward adaptation, scale variation, 10x data).",
"Large-scale evaluation on nine benchmarks with ablations over backbones, fine-tuning protocols and pool size N=3..17.",
],
method: {
pipeline: ["run-pool-on-train-videos", "label-best-tracker", "augment", "train-tp2n", "predict-best", "track-with-winner"],
architecture:
"TP2N: SSL-pretrained backbone (best DINO ViT-S) plus linear/MLP classifier (three linear layers); trained on the union of LaSOT, GOT-10k and TrackingNet training splits labelled by 17 trackers.",
motionModel:
"None: selection only; motion behaviour is whatever the predicted winner tracker implements.",
appearanceModel:
"Scene representation is the SSL backbone feature of early frames; no target template learning in TP2N itself.",
association:
"Not applicable (single-object tracking); frame-level mode switches whole trackers, it does not fuse boxes.",
detectionDependency: "None; SOT initialization from the first-frame box, prediction from the first five frames only.",
trackManagement:
"Video-level: one prediction per video, winner runs the whole sequence. Frame-level: re-predict after fixed 5-frame intervals and switch trackers on scene change.",
loss:
"Classification fine-tuning over one-hot best-tracker labels (loss L in Fig. 2C); exact loss form is not named in the paper.",
optimization:
"SSL protocols of the original authors (MoCoV2 SGD lr 0.3 batch 4096; SwAV batch 2048; BT LARS; DINO AdamW lr 0.0005 batch 1024); linear probing vs full fine-tuning compared.",
},
equations: [
{
id: "bofn-map",
label: "Best-tracker prediction mapping",
formula: "f(theta): S_j -> T_i",
variables: [
{ symbol: "f(theta)", meaning: "TP2N classifier with parameters theta" },
{ symbol: "S_j", meaning: "subsequence of two or more frames of video V_j plus its box" },
{ symbol: "T_i", meaning: "predicted best-performing tracker among the N in the pool" },
],
intuition: "Turn tracker selection into image/scene classification: show the network the start of a video, ask which tracker wins it.",
why: "Formalizes selection so a standard classifier can replace executing N trackers.",
where: "Sec. 3.1 problem formulation.",
paperIds: ["T110"],
},
{
id: "bofn-label",
label: "Performance vector and normalization",
formula: "a_j = [a_{Vj,T1}, ..., a_{Vj,TN}]^T;  p_j = a_j / ||a_j||_2",
variables: [
{ symbol: "a_{Vj,Ti}", meaning: "success-rate AUC of tracker T_i on training video V_j" },
{ symbol: "a_j", meaning: "N-vector of all trackers AUC scores on video V_j" },
{ symbol: "p_j", meaning: "L2-normalized probability-like score vector" },
],
intuition: "Score every tracker on the video, rescale the scores to unit length, and the tallest bar names the winner.",
why: "Produces comparable per-video labels across videos of different absolute difficulty.",
where: "Sec. 3.2 ground-truth label generation, Fig. 2A.",
paperIds: ["T110"],
},
{
id: "bofn-ohe",
label: "One-hot winner encoding",
formula: "OHE(p_j): max entry -> 1, all others -> 0",
variables: [
{ symbol: "p_j", meaning: "normalized tracker-score vector for video V_j" },
{ symbol: "OHE", meaning: "one-hot encoding used as the classifier ground truth" },
],
intuition: "Keep only the winner: the video becomes a single-label example of the tracker that beat the other sixteen.",
why: "Reduces selection to single-label classification solvable by fine-tuning SSL backbones.",
where: "Sec. 3.2 and Fig. 2B.",
paperIds: ["T110"],
},
],
datasets: ["lasot", "trackingnet", "got10k", "vot", "uav123", "otb", "webuav-3m"],
metrics: ["success-auc", "norm-precision", "precision", "eao"],
baselines: ["SiamFC", "SiamRPN++", "Ocean", "TransT", "KeepTrack", "STARK", "SimTrack", "MixFormer", "SwinTrack", "RTS", "ToMP", "OSTrack", "AiATrack", "DropTrack", "GRM", "CiteTracker", "ARTrack"],
results: [
"LaSOT (Tab. 4, AUC/PNorm/P): video-level 77.6/85.8/83.7, frame-level 81.7/87.6/86.2 vs best single ARTrack 73.1/82.2/80.3 (+8.6 AUC frame-level).",
"TrackingNet (Tab. 4): video 88.9/93.4/89.7, frame 92.1/96.6/92.5 vs ARTrack 85.6/89.6/86.0 (+6.5 AUC).",
"GOT-10k (Tab. 4, AO/SR0.75/SR0.5): video 88.7/85.20/95.31, frame 91.1/88.56/97.02 vs RTS 85.2/82.6/94.5 (+5.9 AO frame-level).",
"VOT2021/VOT2022 (Tab. 5, EAO/A/R): VOT2021 video 69.13/86.70/91.40, frame 72.33/88.20/93.70 vs ARTrack 65.90; VOT2022 video 67.88/87.60/90.87, frame 70.98/89.50/91.87 vs ARTrack 64.12 (+6.86 EAO).",
"More sets (Tab. 6, AUC): UAV123 video 74.6 / frame 76.8 (ARTrack 71.2); OTB100 73.8/77.1 (ToMP 70.1); VOT2019 77.2/79.6 (OSTrack 75.4); WebUAV-3M 72.5/76.3 (DropTrack 69.4).",
"TP2N accuracy (Tabs. 1-2, fine-tuned top-1): video-level LaSOT 94.50, TrackingNet 96.60, GOT-10k 96.20, VOT2022 95.50 percent (DINO ViT-S); frame-level best 83.3/89.2/93.6/95.3 percent.",
"Speed/overhead (Tab. 7, Sec. 4.9): video-level 0.84 s overhead per video; BofN fps 68.91 LaSOT / 78.56 VOT2022 / 65.54 TrackingNet, faster than ARTrack/GRM/DropTrack but slower than CiteTracker at lower accuracy.",
],
ablations: [
"Backbone selection (Tab. 3): ViT-S DINO frame-level wins on all four sets (LaSOT 81.77, TrackingNet 92.10, GOT-10k 91.12, VOT2022 70.98); ResNet MoCoV2 second on LaSOT/TrackingNet.",
"Full fine-tuning beats linear probing for every backbone and dataset in TP2N top-1 accuracy (Tabs. 1-2).",
"Pool size N (Tab. 8): LaSOT video 74.2 to 77.6 and frame 76.5 to 81.7 from Bof3 to Bof17; monotonic gains on all four datasets.",
"Frame-level beats video-level by up to 4.17 percent by switching trackers within long sequences (Sec. 4.5).",
],
limitations: {
authorStated: [
"Frame-level mode runs all N trackers (footnote 1), losing the minimal-overhead advantage of video-level selection.",
"No Free Lunch framing: gains assume the pool contains a suitable tracker per scene; larger pools help but cost more.",
],
evident: [
"Video-level prediction from only the first five frames cannot foresee late-sequence attribute shifts (the reason frame-level adds up to 4.17 percent).",
"TP2N is trained on only three training splits, so generalization to unseen domains rests on the four extra test sets rather than broader training.",
],
},
assumptions: [
"Early frames plus the initial box reveal enough scene character to name the best tracker for the whole video.",
"Per-video argmax AUC is a stable label: one winner per video, ties ignored.",
"Running 17 trackers offline for labelling is affordable; only inference cost matters.",
],
computation:
"TP2N pretrained 200 ImageNet epochs on 4x V100; inference overhead 0.84 s per video-level prediction, times n_e evaluations at frame level; BofN fps in Tab. 7.",
relations: [
{ to: "T009", type: "uses-as-baseline", note: "SiamFC in the 17-tracker pool and reference set; BofN frame-level beats it 81.7 vs 33.6 AUC on LaSOT." },
{ to: "T028", type: "uses-as-baseline", note: "SiamRPN++ in the pool; beaten 81.7 vs 49.6 AUC on LaSOT." },
{ to: "T071", type: "uses-as-baseline", note: "OSTrack (one-stream) in the pool; beaten 81.7 vs 71.1 AUC on LaSOT and 79.6 vs 77.2 on VOT2019." },
],
concepts: ["benchmark-design", "sot", "success-plot", "attention"],
impact:
"Reframed SOT progress as a selection problem over diverse paradigms and supplied the largest cross-benchmark SOTA comparison of the 17-trackers era.",
},
{
id: "T111",
arxiv: "2408.07344",
title: "RTAT: A Robust Two-stage Association Tracker for Multi-Object Tracking",
shortTitle: "RTAT",
year: 2024,
authors: ["Song Guo", "Rujie Liu", "Narishige Abe"],
fileName: "2408.07344v1.pdf",
task: "multi-object",
tags: ["tracking-by-detection", "two-stage-association", "graph-neural-network", "message-passing", "tracklet-merging", "hierarchical-graph"],
difficulty: "advanced",
summary:
"RTAT splits association into cheap high-purity tracklet generation and learned tracklet merging. Stage one runs a simple matcher (ByteTrack or BoT-SORT) with a low cost threshold (thc=0.2) so tracklets are short but 93.4 percent pure; stage two merges them with a 3-level message-passing GNN doing edge classification on sparse tracklet graphs. It ranks first on MOT17 (67.2 HOTA / 84.7 IDF1) and MOT20 (66.2 / 82.5), nearly erasing the ByteTrack vs BoT-SORT gap.",
problem:
"Handcrafted association is simple and fast but rule-based, hard to generalize across crowded, night or fast-camera scenes, and errors are unfixable; learned detection-level association (Transformers, GNNs) captures high-order context but builds huge graphs over detections with high complexity and memory. MOT17/MOT20 training sets are tiny (7 and 4 videos), so GNNs overfit.",
background: ["mot", "tracking-by-detection", "data-association", "cost-matrix", "hungarian", "reid", "kalman", "track-management"],
previousWork: [
{
name: "Handcrafted association (motion models, ReID features, cost-matrix design, cascade matching)",
limitation:
"Endless manual design of cost matrices and matching rules that still fail on some scenes, with no recovery from a wrong match.",
whyThisPaper:
"RTAT keeps only a deliberately simple first stage for pure fragments and moves all hard decisions to the learned merger, making first-stage choice nearly irrelevant (88.0-88.5 IDF1 for any of three trackers).",
},
{
name: "Detection-level GNN association (MPNTrack, SUSHI, SGT)",
limitation:
"Graphs over all detections in long or crowded videos explode in size, memory and label imbalance.",
whyThisPaper:
"Graphs are built over tracklets (hundreds, not thousands, of nodes) with K=10 nearest-neighbor edges, so whole-video tracklet association is tractable.",
},
{
name: "Tracklet split-merge pipelines (TAT, ReMOT, tracklet booster)",
limitation:
"Tracklets come from small sliding windows or aggressive splits, often too short, and message-passing GNN merging is underexploited.",
whyThisPaper:
"Low-threshold simple tracking yields high-purity pieces (HPR 93.4 percent) and a shared-weight hierarchical GNN recursively merges them over 3 levels.",
},
],
researchGap:
"No tracker combined trivially simple high-purity fragment generation with whole-video hierarchical message-passing tracklet merging.",
contribution: [
"Two-stage design: low-threshold (thc=0.2) simple matching for high-purity tracklets, then hierarchical GNN edge-classification merging into complete trajectories.",
"Sparse tracklet graph: edges only between temporally non-overlapping pairs, top-K=10 neighbors by appearance/motion/spatial similarity.",
"8-dim edge initialization (relative position/scale, time gap, Euclidean plus top-L cosine appearance, KF-based GIoU motion consistency) with node features from averaged tracklet ReID embeddings.",
"Shared-weight 3-level GNN with per-level adapter, L=12 message-passing steps, focal-loss training, plus video-level and tracklet-level augmentation for tiny MOT datasets.",
"First place on MOT17 and MOT20 test sets in HOTA, IDF1 and AssA under the private-detector protocol.",
],
method: {
pipeline: ["detect", "low-threshold-match", "build-tracklet-graph", "init-node-edge-features", "message-pass", "classify-edges", "merge-hierarchically", "interpolate"],
architecture:
"YOLOX detector plus ResNet50 ReID (node dim 32); light-weight MLP node/edge encoders, message-passing GNN and edge classifier shared across 3 hierarchical levels with a learned level adapter.",
motionModel:
"Kalman-filter prediction of each tracklet pair to a middle frame for the GIoU motion-consistency edge cue; first-stage motion follows the chosen simple tracker.",
appearanceModel:
"ResNet50 ReID per detection, averaged over the tracklet (robust to blur/occlusion/illumination); pairwise Euclidean distance plus averaged top-L cosine similarity.",
association:
"Stage one: bipartite Hungarian matching with normalized costs, rejecting matches above thc=0.2. Stage two: hierarchical edge classification with exact rounding, merging nodes linked by active edges.",
detectionDependency: "YOLOX trained following ByteTrack for both MOT17 and MOT20, private-detector protocol.",
trackManagement:
"Fragment-tolerant first stage (1693 pieces at thc=0.2 vs 546 GT) reduced to ~612 trajectories after 3 levels; linear interpolation fills missing detections.",
loss:
"Focal loss (gamma=1) on edge classification per level, summed over all levels.",
optimization:
"3 hierarchical levels jointly trained 500 epochs, Adam lr 3e-4, weight decay 1e-4; K=10 edges per node, L=12 message-passing steps.",
},
equations: [
{
id: "rtat-spatial",
label: "Relative spatial-scale edge cue",
formula: "[2(x_j - x_i)/(h_j + h_i), 2(y_j - y_i)/(h_j + h_i), log(h_j/h_i), log(w_j/w_i)];  dt_gap = (t_b1 - t_an)/fps",
variables: [
{ symbol: "T_a, T_b", meaning: "tracklet pair with T_a ending before T_b starts" },
{ symbol: "i = a_n, j = b_1", meaning: "closest boxes: last of T_a and first of T_b" },
{ symbol: "x, y, w, h", meaning: "box center coordinates, width, height" },
{ symbol: "dt_gap", meaning: "temporal gap in seconds via video fps" },
],
intuition: "How far and how differently sized the two fragments endpoints are, scaled by their heights so near and far objects compare fairly.",
why: "Short-term spatial continuity is the cheapest reliable merge signal.",
where: "Sec. 3.4 graph initialization, Eq. 1.",
simulator: "assoc-cost",
paperIds: ["T111"],
},
{
id: "rtat-appearance",
label: "Tracklet appearance edge cue",
formula: "[||app_Tb - app_Ta||, (1/(L L)) sum_i sum_j cos(app_Ta_i, app_Tb_j)]",
variables: [
{ symbol: "app_T", meaning: "average ReID embedding over all detections in tracklet T" },
{ symbol: "||.||", meaning: "Euclidean distance for global appearance discrepancy" },
{ symbol: "cos", meaning: "cosine similarity between detections; averaged over top-L closest pairs for local robustness" },
{ symbol: "L", meaning: "number of closest detections compared per pair" },
],
intuition: "Compare both the whole-fragment average look and the best-matching glimpse pairs, so one occluded frame cannot veto a merge.",
why: "Averaged features resist blur and partial occlusion better than single-detection embeddings.",
where: "Sec. 3.4 graph initialization, Eq. 2.",
simulator: "reid",
paperIds: ["T111"],
},
{
id: "rtat-motion",
label: "KF motion-consistency cue",
formula: "GIOU(pred_box_Ta(t_mid), pred_box_Tb(t_mid));  t_mid = t_an + (t_b1 - t_an)/2",
variables: [
{ symbol: "t_mid", meaning: "middle frame between the two tracklets" },
{ symbol: "pred_box", meaning: "KF-predicted box of each tracklet at t_mid" },
{ symbol: "GIOU", meaning: "Generalized IoU of the two predicted boxes" },
],
intuition: "Extrapolate both fragments to the gap midpoint: same identity means the two predictions should land on top of each other.",
why: "Enforces constant-velocity motion consistency across the gap the fragments do not cover.",
where: "Sec. 3.4 graph initialization; concatenated into the 8-dim edge vector.",
simulator: "kalman",
paperIds: ["T111"],
},
{
id: "rtat-mp",
label: "Message-passing update",
formula: "f_(i,j)^l = U_e([f_i^{l-1}, f_j^{l-1}, f_(i,j)^{l-1}]);  m_(i,j)^l = U_n([f_i^{l-1}, f_(i,j)^l]);  f_i^l = phi({m_(i,j)^l}_{j in N_i})",
variables: [
{ symbol: "f_i, f_(i,j)", meaning: "node and edge feature vectors" },
{ symbol: "U_e, U_n", meaning: "MLP aggregators for edges and nodes" },
{ symbol: "N_i", meaning: "neighbors of node i" },
{ symbol: "phi", meaning: "order-invariant pool (max/sum/average)" },
{ symbol: "l = 1..L", meaning: "message-passing iteration, L=12" },
],
intuition: "Let each candidate merge hear from neighboring fragments and competing merges for 12 rounds before deciding.",
why: "Propagates high-order context (crowding, parallel identities) that pairwise costs cannot see.",
where: "Sec. 3.4 graph update, Eq. 3.",
paperIds: ["T111"],
},
{
id: "rtat-edge",
label: "Edge classification and merging",
formula: "y_(i,j) = C_eclass(f_(i,j)^L) in (0,1); merge tracklets linked by active edges after exact rounding",
variables: [
{ symbol: "C_eclass", meaning: "MLP plus sigmoid edge classifier" },
{ symbol: "f_(i,j)^L", meaning: "final edge feature after L message-passing steps" },
{ symbol: "y_(i,j)", meaning: "probability the two tracklets share an identity" },
],
intuition: "Turn association into binary decisions on merge candidates, then glue the accepted ones into longer trajectories.",
why: "Classification over a sparse tracklet graph replaces global combinatorial optimization over detections.",
where: "Sec. 3.4 edge classification, Eq. 4; repeated hierarchically over 3 levels.",
paperIds: ["T111"],
},
],
datasets: ["mot17", "mot20"],
metrics: ["hota", "idf1", "mota", "assa", "idsw"],
baselines: ["ByteTrack", "StrongSORT", "Deep OC-SORT", "BoT-SORT", "MotionTrack", "ConfTrack", "C-BIoU", "PIA", "ImprAsso", "SUSHI", "FineTrack", "BoT-SORT-ReID"],
results: [
"MOT17 test private detector (Tab. 5, HOTA/IDF1/MOTA/AssA/IDs): RTAT-BoT-SORT 67.2/84.7/80.4/69.7/912; RTAT-ByteTrack 67.0/84.4/80.1/69.3/942, both first in HOTA, IDF1, AssA.",
"MOT17 vs SOTA: ByteTrack 63.1/77.3/80.3/62.0/2196; StrongSORT 64.4/79.5/79.6/64.4/1194; Deep OC-SORT 64.9/80.6/79.4/65.9/1023; BoT-SORT 65.0/80.2/80.5/65.5/1212; MotionTrack 65.1/80.1/81.1/65.1/1140; SUSHI 66.5/83.1/81.1/67.8/1149.",
"MOT20 test (Tab. 6): RTAT-BoT-SORT 66.2/82.5/78.4/68.2/787; RTAT-ByteTrack 65.9/82.1/78.1/67.7/817 vs second-place ConfTrack 64.8/80.2/77.2/66.2/702 (+1.4/+2.3/+1.9 HOTA/IDF1/AssA over runner-up).",
"First-stage invariance (Tab. 4, MOT17 val): ByteTrack 76.6/1571/91.9 to 88.0/623/276; BoT-SORT 83.6/1693/93.4 to 88.5/612/261; Deep OC-SORT 81.2/1264/89.7 to 88.2/617/264 (first-stage IDF1/pieces/HPR to second-stage IDF1/pieces/IDs).",
],
ablations: [
"Cost threshold (Tab. 1): thc 0.7 to 0.1 raises pieces 753 to 4964 and HPR 84.5 to 98.1 percent; post-merge IDF1 peaks 88.5 at thc=0.2 (83.6/1693/93.4 to 88.5/612/261); thc=0.1 gains most (+19.2 IDF1) but costs 4964 nodes.",
"Hierarchical levels (Tab. 2): HL 0 to 5 gives IDF1 83.6/86.2/87.6/88.5/88.6/88.6, IDs 1344/406/287/261/258/257, pieces 1693/1021/728/612/608/607; gains negligible past 3.",
"Augmentation (Tab. 3): none 85.5/293; video-level 86.7/278; tracklet-level 87.2/272; both 88.5/261 (IDF1/IDs).",
"HPR metric: tracklet is high-purity if over 80 percent of its detections share one identity.",
],
limitations: {
authorStated: [
"No explicit limitation section; the paper reports that gains saturate past 3 hierarchical levels while time and memory costs keep growing.",
],
evident: [
"MOTA barely moves (second-stage gains are association-only), so detection errors and linear interpolation artefacts remain.",
"Whole-video tracklet graphs plus 12 message-passing rounds are offline by construction, not a streaming design.",
],
},
assumptions: [
"A threshold of thc=0.2 yields fragments pure enough that merging never needs to split.",
"Top-10 neighbor pruning never discards a true match; non-overlapping pairs only.",
"One shared GNN plus a level adapter fits all merging stages.",
],
computation:
"YOLOX detections; ResNet50 ReID with node dim 32; 500 epochs Adam lr 3e-4 wd 1e-4; HL=3, K=10, L=12; video clips sampled every 50 frames (plus/minus 15) at 25-100 percent length.",
relations: [
{ to: "T061", type: "improves", note: "RTAT-ByteTrack beats ByteTrack 67.0 vs 63.1 HOTA MOT17 and 65.9 vs 61.3 MOT20." },
{ to: "T075", type: "improves", note: "RTAT-BoT-SORT beats BoT-SORT 67.2 vs 65.0 HOTA MOT17 and 66.2 vs 63.3 MOT20." },
{ to: "T067", type: "uses-as-baseline", note: "StrongSORT compared in Tabs. 5-6; beaten 67.2 vs 64.4 MOT17, 66.2 vs 62.6 MOT20." },
{ to: "T082", type: "uses-as-baseline", note: "MotionTrack compared in Tabs. 5-6; beaten 67.2 vs 65.1 MOT17, 66.2 vs 62.8 MOT20." },
],
concepts: ["data-association", "learned-association", "track-management", "cost-matrix", "hungarian", "reid"],
impact:
"Showed simple matchers suffice for fragment generation and moved the frontier to learned tracklet merging, relieving researchers of hand-tuning association rules.",
},
{
id: "T112",
arxiv: "2408.15548",
title: "ConsistencyTrack: A Robust Multi-Object Tracker with a Generation Strategy of Consistency Model",
shortTitle: "ConsistencyTrack",
year: 2024,
authors: ["Lifan Jiang", "Zhihui Wang", "Siqi Yin", "Guangxiao Ma", "Peng Zhang", "Boxi Wu"],
fileName: "2408.15548v1.pdf",
task: "multi-object",
tags: ["joint-detection-tracking", "consistency-model", "diffusion", "generative-tracking", "denoising", "occlusion-handling"],
difficulty: "advanced",
summary:
"ConsistencyTrack ports the consistency model into joint detection-and-tracking MOT: paired boxes across frames (k-dk, k) are noised from ground truth and the network learns to map any noisy point back to the trajectory origin in one step, with a spatial-temporal fusion module and an association-score head. It reaches 69.9 MOTA / 65.7 IDF1 / 54.4 HOTA on MOT17 and 87.8 MOTA on DanceTrack test, holding ~10.5 FPS for any 1-6 sampling steps where DiffusionTrack falls from 2.5 to 0.84.",
problem:
"Diffusion-based tracking (DiffusionTrack) is noise-robust but its gradual iterative denoising is inflexible and slow, blocking real-time use; TBD methods collapse when detection fails under occlusion, and JDT methods struggle to separate close objects and keep identities in dynamic scenes.",
background: ["mot", "tracking-by-detection", "diffusion-tracking", "data-association", "iou", "kalman", "occlusion"],
previousWork: [
{
name: "Tracking-by-detection (SORT, DeepSORT)",
limitation:
"Wholly dependent on per-frame detection accuracy; one missed detection under occlusion breaks the track.",
whyThisPaper:
"Generative paired-box denoising detects and associates jointly from noise, tolerating perturbed inputs (ConsistencyDet stable past 300 boxes where DETR collapses).",
},
{
name: "Joint detection-embedding (JDE)",
limitation:
"Joint optimization of detection and embedding can degrade both under complex scenes.",
whyThisPaper:
"Separate association-score head on STF-fused paired features keeps detection and association objectives disentangled in one decoder.",
},
{
name: "Query/offset/trajectory one-stage JDT (TrackFormer, TransTrack, CenterTrack, GTR, TubeTK) and DiffusionTrack",
limitation:
"Query methods tangle identities in crowds; DiffusionTrack needs many gradual denoise iterations, so inference is far from real-time.",
whyThisPaper:
"Self-consistent one-step mapping keeps DiffusionTrack accuracy while FPS stays ~10.5 regardless of sampling steps (10.51 vs 2.50 at nss=2).",
},
],
researchGap:
"No MOT tracker combined diffusion-grade noise robustness with single-step consistency sampling plus an occlusion-aware low-confidence association stage.",
contribution: [
"ConsistencyTrack JDT framework: paired noise boxes over frames (k-dk, k) denoised into paired detections plus association scores in minimal steps.",
"Paired training loss aggregating predictions at adjacent timesteps (t-1, t) over focal plus L1 plus 3D-GIoU terms, enforcing self-consistency back to the origin.",
"Spatial-Temporal Fusion module (linear-split, dual BMM, linear) plus association-score head for paired-box data association without extra matching features.",
"Four-stage IoU association strategy emphasizing low-confidence boxes and Kalman-based lost-object reassociation for occlusions.",
"Efficiency proof: stable ~10.5 FPS across nss=1..6 with best MOTA/IDF1 at nss=2, and ablations for prior-repeat (nrp=8), box-renewal (Bth=0.6) and log stretch.",
],
method: {
pipeline: ["extract-paired-features", "pad-gt", "add-noise", "denoise-to-origin", "predict-boxes-scores", "nms", "four-stage-associate"],
architecture:
"YOLOX-FPN backbone on frame pairs plus ConsistencyHead decoder blocks (self-attention, STF, box/class/association heads); Ntrain=500 padded boxes in training, np=2000 proposals with nrp=8 prior repeats in inference.",
motionModel:
"None learned explicitly; temporal link comes from paired-box regression across the dk=5-spaced frame pair plus a Kalman filter for lost-object reassociation.",
appearanceModel:
"RoI features from the backbone conditioned into the denoiser; association confidence from the fused paired-box head rather than a ReID embedding.",
association:
"JDT four-stage IoU matching over high/low-confidence splits (association-score thresholding, track/non-lost/lost/new partitions, unactivated handling) with Kalman reassociation; NMS plus box-renewal post-process.",
detectionDependency: "YOLOX backbone pretrained weights from ByteTrack; input 1440x800; Mosaic/Mixup augmentation.",
trackManagement:
"Activated/lost/unactivated/removed states with nlost frame limit; new-track birth from unmatched high-confidence pairs; lost tracks reassociated before removal.",
loss:
"Per-step focal plus L1 plus 3D-GIoU (frustum volume over the frame pair), summed over adjacent pair (t-1, t) per Eq. 15.",
optimization:
"AdamW lr 1e-4 cosine decay (final factor 0.1), warmup 2.5e-5 for 1 epoch; MOT17 60 detection epochs on MOT17+CrowdHuman+Cityperson+ETHZ plus 60 tracking epochs; DanceTrack 80 epochs; batch 3, single A100 FP32.",
},
equations: [
{
id: "ct-forward",
label: "Forward diffusion kernel",
formula: "q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) x_{t-1}, beta_t I)",
variables: [
{ symbol: "x_t", meaning: "box state at diffusion step t" },
{ symbol: "beta_t", meaning: "noise schedule at step t" },
{ symbol: "N", meaning: "Gaussian distribution" },
],
intuition: "Gradually snow under the true boxes with Gaussian noise until only static remains.",
why: "Defines the corruption process the consistency mapper must learn to invert.",
where: "Sec. 3.1 preliminaries, Eq. 1.",
paperIds: ["T112"],
},
{
id: "ct-consistency",
label: "Consistency parametrization",
formula: "f_theta(x, t) = c_skip(t) x + c_out(t) F_theta(x, t);  c_skip(tau) = 1, c_out(tau) = 0",
variables: [
{ symbol: "F_theta", meaning: "free-form denoising network" },
{ symbol: "f_theta", meaning: "consistency function mapping any trajectory point to its origin" },
{ symbol: "c_skip, c_out", meaning: "differentiable skip/output scales enforcing the boundary condition" },
{ symbol: "tau", meaning: "origin timestep of the probability-flow ODE trajectory" },
],
intuition: "Whatever noisy version you hand it along one trajectory, the network must point back to the same clean start.",
why: "Self-consistency is what permits one-step generation instead of hundred-step integration.",
where: "Sec. 3.1 preliminaries, Eqs. 3-4.",
paperIds: ["T112"],
},
{
id: "ct-noise",
label: "Box corruption schedule",
formula: "sigma_t = (sigma_max^{1/rho} + (t/(T-1))(sigma_min^{1/rho} - sigma_max^{1/rho}))^rho;  x_t = x_s + eps sigma_t;  x_t <- (c_in(t)/2) x_t",
variables: [
{ symbol: "sigma_t", meaning: "noise scale at step t between sigma_min and sigma_max" },
{ symbol: "rho, T", meaning: "schedule curvature and total timesteps" },
{ symbol: "x_s", meaning: "padded ground-truth paired boxes (origin)" },
{ symbol: "eps", meaning: "standard Gaussian noise" },
{ symbol: "c_in(t) = 1/sqrt(sigma_t^2 + sigma_data^2)", meaning: "input rescaling keeping noised boxes in range" },
],
intuition: "Shake each ground-truth box pair with a calibrated amount of noise that grows along the schedule, then rescale to stay valid.",
why: "Controlled corruption teaches the denoiser every noise level it will meet at inference.",
where: "Sec. 3.3 box corruption, Eqs. 8-11.",
paperIds: ["T112"],
},
{
id: "ct-loss",
label: "Paired self-consistent loss",
formula: "L = lambda_cls (L_cls^{t-1} + L_cls^t) + lambda_L1 (L_L1^{t-1} + L_L1^t) + lambda_GIoU3d (L_GIoU3d^{t-1} + L_GIoU3d^t)",
variables: [
{ symbol: "L_cls", meaning: "focal classification loss on predicted boxes" },
{ symbol: "L_L1", meaning: "L1 box regression loss" },
{ symbol: "L_GIoU3d", meaning: "1 minus 3D-GIoU over the paired frustum" },
{ symbol: "lambda_*", meaning: "positive weights per loss item" },
],
intuition: "Grade the denoiser at two neighboring noise levels at once, so both answers agree on the same clean origin.",
why: "Adjacent-step aggregation is the training embodiment of the self-consistency principle.",
where: "Sec. 3.3 loss, Eqs. 12-15.",
paperIds: ["T112"],
},
{
id: "ct-giou3d",
label: "3D GIoU over paired boxes",
formula: "L_GIoU3d = 1 - GIoU3d(T_d, T_gt);  GIoU3d from frustum intersection/union plus enclosing hull over frames k-1..k",
variables: [
{ symbol: "T_d, T_gt", meaning: "square frustums from estimated and ground-truth box pairs across the two frames" },
{ symbol: "B_d, B_gt", meaning: "per-frame estimated and ground-truth boxes" },
{ symbol: "D", meaning: "smallest convex hull enclosing B_d and B_gt per frame" },
],
intuition: "Treat a tracklet snippet as a 3D volume through time and score overlap of predicted vs true volumes, not just per-frame areas.",
why: "Couples detection accuracy across the pair so association errors hurt the loss directly.",
where: "Sec. 3.3 loss, Eqs. 13-14 and Fig. 4.",
simulator: "iou-track",
paperIds: ["T112"],
},
],
datasets: ["mot17", "dancetrack", "others"],
metrics: ["mota", "idf1", "hota", "assa", "deta", "mt-ml", "fp", "fn", "idsw", "fragments", "fps"],
baselines: ["Tracktor++v2", "TubeTK", "CTTrack17", "CJTracker", "TrajE", "SpCon", "PCL", "UTM", "IoU", "DeepSORT", "MOTDT", "CenterTrack", "FairMOT", "DiffusionTrack", "ConsistencyDet", "DiffusionDet", "DETR"],
results: [
"MOT17 test (Tab. 2): 69.9 MOTA / 65.7 IDF1 / 54.4 HOTA / MT 907 / ML 428 / FP 24186 / FN 142145 / AssA 51.2 / DetA 58.2 / IDs 3774 / Frag 5854, best in MOTA/IDF1/HOTA/DetA; vs UTM 63.5/65.1/52.5 (+6.4/+0.6 MOTA/IDF1) and PCL 58.8/61.2/49.0.",
"DanceTrack val (Tab. 3, HOTA/DetA/AssA/MOTA/IDF1): 45.5/77.7/26.9/88.1/43.4, best MOTA; HOTA second to DeepSORT 45.8.",
"DanceTrack test (Tab. 4): 42.3/76.4/25.4/87.8/41.2 vs CenterTrack 41.8/78.1/22.6/86.8/35.7 and FairMOT 39.7/66.7/23.8/82.2/40.8; DetA slightly below CenterTrack.",
"MOT17 val-half vs DiffusionTrack (Tab. 8): 75.7/76.5/83.3/MT 52.8/FN 19.4/IDs 298 vs 74.4/74.5/82.7/46.6/21.3/433 (MOTA/IDF1/IDP/MT/FN/IDs).",
"Speed (Tab. 6, np=2000, RTX 3090): ConsistencyTrack 10.53/10.51/10.39/10.27 FPS at nss=1/2/4/6 vs DiffusionTrack 2.50/1.25/0.84 at nss=2/4/6; best MOTA and IDF1 at nss=2 (Fig. 7).",
"Detector robustness (Fig. 6, COCO): ConsistencyDet stable past np=300 peaking at 500, while DETR falls from 38.8 to 26.4 percent AP at np=4000.",
],
ablations: [
"Prior repeats (Tab. 5 left): nrp 6/8/10 gives MOTA 75.5/75.8/75.2, IDF1 76.6/76.2/76.4; nrp=8 best.",
"Box-renewal threshold (Tab. 5 right): Bth 0.5/0.6/0.7 gives 75.8/75.8/75.4 MOTA; 0.6 best balanced.",
"Stretch function (Tab. 7): log (base 1.01) 75.8/76.2/82.9 beats x, exp, sqrt, tanh and z-norm on MOTA/IDF1/IDP.",
"Association strategy (text, same np/nss/nrp): plus 1.3 MOTA, 2.0 IDF1, 0.6 IDP over DiffusionTrack matching (Fig. 10).",
],
limitations: {
authorStated: [
"Single-step denoising uses excessive add/remove amplitude, costing accuracy: frequent loss of tracked targets and delayed identification of new targets (Fig. 11).",
"MOTA oscillates for nss above 2 with peaks below the nss=2 case (Fig. 7).",
"Future work must improve detection/tracking precision and integrate consistency principles into stronger trackers.",
],
evident: [
"MOT17 test IDs 3774 and Frag 5854 are the highest in Tab. 2 despite best MOTA, so identity continuity lags TBD association SOTA.",
"Training is heavyweight (paired frames, 500 boxes, A100) while inference still needs 1000-4000 proposal boxes.",
],
},
assumptions: [
"Paired boxes from frames dk=5 apart carry enough correlation for joint detect-and-associate.",
"If both frames are identical the task degenerates to pure detection (shared head).",
"Random Gaussian proposals at inference cover all true objects given enough boxes.",
],
computation:
"1440x800 input, Ntrain=500, np=2000, nrp=8, Bth=0.6, nss=2 best; AdamW 1e-4 cosine x0.1; single A100 FP32; ~10.5 FPS on RTX 3090 independent of nss.",
relations: [
{ to: "T092", type: "improves", note: "DiffusionTrack reframed with one-step consistency: 75.7 vs 74.4 MOTA and 10.51 vs 2.50 FPS at nss=2." },
{ to: "T061", type: "builds-on", note: "Adopts the pretrained YOLOX detector from ByteTrack for backbone and detection schedule." },
{ to: "T013", type: "uses-as-baseline", note: "DeepSORT compared on DanceTrack val; HOTA 45.5 vs 45.8, MOTA 88.1 vs 87.1." },
{ to: "T043", type: "uses-as-baseline", note: "FairMOT compared on DanceTrack test; beaten 42.3 vs 39.7 HOTA and 87.8 vs 82.2 MOTA." },
],
concepts: ["diffusion-tracking", "end-to-end-mot", "data-association", "occlusion"],
impact:
"First consistency-model MOT tracker, proving single-step generative denoising can keep diffusion robustness at an order of magnitude higher inference speed.",
},
{
id: "T113",
arxiv: "2409.07904",
title: "FACT: Feature Adaptive Continual-learning Tracker for Multiple Object Tracking",
shortTitle: "FACT",
year: 2024,
authors: ["Rongzihan Song", "Zhenyu Weng", "Huiping Zhuang", "Jinping Ren", "Yongming Chen", "Zhiping Lin"],
fileName: "2409.07904v1.pdf",
task: "multi-object",
tags: ["tracking-by-detection", "continual-learning", "online-learning", "analytic-learning", "re-identification", "two-stage-association", "occlusion-handling"],
difficulty: "advanced",
summary:
"FACT learns appearance online from all past tracking information instead of recent frames or a bounded memory bank. Its FAC module (random ET expansion to 3000 dims plus a linear FCN classifier) is trained recursively in closed form: the autocorrelation unit R(k) and weight update reproduce exact joint ridge regression over the full history with O(d_et^2 d_Tk) cost. A two-stage associator (FAC affinity first, cosine plus IoU instant matching for unconfident/new tracks) yields 66.8 HOTA / 82.9 IDF1 on MOT17 and 67.2 / 83.6 on MOT20 with FACT+.",
problem:
"Occlusions contaminate appearance embeddings. Online learners adapt on recent frames only (or assume linear separability), while memory-bank offline methods store bounded local history due to memory or recency bias, so neither exploits all past tracking information in real time; new tracks are also unusable before a learned module initializes.",
background: ["mot", "tracking-by-detection", "appearance-features", "reid", "data-association", "cost-matrix", "hungarian", "memory-network"],
previousWork: [
{
name: "ReID optimization trackers (hand-crafted features, DeepSORT, siamese/transformer ReID, quasi-dense contrastive)",
limitation:
"All effort goes into the offline ReID model; contaminated embeddings during tracking are never corrected online.",
whyThisPaper:
"FAC continually re-learns a per-track classifier over the full history during tracking, cutting occlusion distance from ~1.0 (cosine) to 0.37-0.14 (Fig. 3).",
},
{
name: "Temporal feature-update methods (weighted averaging, MeMOT, BLSTM-MTP, MeMOTR memory banks)",
limitation:
"Bounded memory or recency-biased attention restricts them to local past information, weakest on long videos.",
whyThisPaper:
"Exact recursive least-squares over all history helps most where sequences are longest: up to +2.0 IDF1 on MOT20 (2233 frames/video) vs +1.1 on MOT17.",
},
{
name: "Online learning trackers (ILDA classifiers, CNN transfer, per-target SOT learners)",
limitation:
"Linear-separability assumptions, fixed sample budgets, or one-model-per-target cost; none learns all history online at real-time speed.",
whyThisPaper:
"One unified FAC module with closed-form recursion adds only 0.3-0.4 FPS cost on large models while matching bigger-model gains (+1.1 IDF1 for free vs +18.5 FPS cost of larger ReID).",
},
],
researchGap:
"No online MOT learner exploited all past appearance information recursively in real time with exact equivalence to joint training.",
contribution: [
"FACT framework: detection plus ReID extraction, FAC affinity estimation, two-stage association, and online FAC update each frame, pluggable into FairMOT, Bot-SORT, JDE and CSTrack.",
"FAC module: ET random expansion (ReLU, det=3000) plus learnable FCN classifier with analytic continual-learning methodology and Theorem 3.1 recursion.",
"Two-stage association: FAC affinity-distance plus Kalman Hungarian matching first, cosine plus IoU instant association for low-confidence and new tracks.",
"Complexity analysis showing linear growth in stored tracks with the analytic recursion vs re-solving over history.",
"State-of-the-art online results: FACT+ 66.8/82.9 MOT17 and 67.2/83.6 MOT20 (HOTA/IDF1), at 12.3/3.2 FPS on 2080Ti and 21.2 FPS on RTX4090 (MOT17).",
],
method: {
pipeline: ["detect-and-embed", "fac-affinity", "stage1-match", "instant-match", "update-tracks", "recursive-fac-train"],
architecture:
"YOLOX-X detector plus FastReID SBS-50 embeddings; FAC module (ET layer Wet: d_ReID x 3000, ReLU; FCN classifier over d_Tk tracks); CMC motion compensation; Hungarian assignment.",
motionModel:
"Kalman filter fused with the FAC affinity-distance matrix for stage-one matching; CMC as in BoT-SORT.",
appearanceModel:
"FastReID SBS-50 one-dimensional embeddings transformed by fixed random ET; per-track discrimination from the online FCN weights rather than raw cosine distance.",
association:
"Stage 1: affinity matrix A_k from FAC output over M_k active tracks, converted to distance D_k, thresholded, Hungarian with Kalman. Stage 2: cosine plus IoU Hungarian for unconfident/new targets; unmatched become new tracks; track features updated by weighted summation.",
detectionDependency: "YOLOX-X detector on the ByteTrack training schedule; ablations also with FairMOT detector and FairMOT embeddings.",
trackManagement:
"Active/lost/new sets; new tracks appended as extra FCN columns; low-confidence FAC predictions deferred to instant association until initialized over a few frames; linear interpolation for final benchmarks.",
loss:
"Ridge-regularized least-squares on stacked one-hot association labels (Eq. 7), solved exactly via the R(k) recursion; no gradient steps during tracking.",
optimization:
"No gradient training online: closed-form base solution plus per-frame recursion; offline detector/ReID follow ByteTrack and FastReID schedules.",
},
equations: [
{
id: "fact-et",
label: "Embedding transformation",
formula: "X_k^{(et)} = fact(X_k^{(ReID)} W_et)",
variables: [
{ symbol: "X_k^{(ReID)}", meaning: "N_k x d_ReID ReID embeddings at frame k" },
{ symbol: "W_et", meaning: "fixed randomized d_ReID x d_et expansion matrix (d_et = 3000)" },
{ symbol: "fact", meaning: "ReLU activation" },
{ symbol: "X_k^{(et)}", meaning: "N_k x d_et transformed features fed to the FCN classifier" },
],
intuition: "Randomly project each appearance vector into a wide nonlinear space where tracks become linearly separable.",
why: "Makes the closed-form ridge classifier expressive without backpropagation.",
where: "Sec. III-C, Eq. 1.",
paperIds: ["T113"],
},
{
id: "fact-base",
label: "Joint ridge base solution",
formula: "W_hat^{(k-1)} = (gamma I + X_hist^T X_hist)^{-1} X_hist^T Y_hat_hist;  R^{(k-1)} = (gamma I + X_hist^T X_hist)^{-1}",
variables: [
{ symbol: "X_hist", meaning: "all transformed embeddings from frames 0..k-1" },
{ symbol: "Y_hat_hist", meaning: "all stacked one-hot association labels (block matrix over track birth groups)" },
{ symbol: "gamma", meaning: "ridge regularization weight" },
{ symbol: "R", meaning: "d_et x d_et feature autocorrelation unit storing all history" },
{ symbol: "W_hat", meaning: "d_et x d_T(k-1) FCN classifier weights" },
],
intuition: "The best linear map from appearances to track IDs over everything seen so far, with a penalty keeping weights small.",
why: "Defines the exact all-history target the recursion must reproduce.",
where: "Sec. III-C base learning, Eqs. 2-6.",
paperIds: ["T113"],
},
{
id: "fact-recurse",
label: "Recursive continual update",
formula: "W_hat^{(k)} = [V_k W_hat^{(k-1)} + R^{(k)} X_k^T Y_hat_k^{(0:k-1)}, R^{(k)} X_k^T Y_hat_k^{(k)}];  V_k = I - R^{(k)} X_k X_k^T",
variables: [
{ symbol: "Y_hat_k^{(0:k-1)}, Y_hat_k^{(k)}", meaning: "frame-k labels for old tracks and for tracks born at k" },
{ symbol: "V_k", meaning: "forgetting-correction factor preserving old knowledge" },
{ symbol: "R^{(k)}", meaning: "autocorrelation updated by Woodbury identity from R^{(k-1)} and X_k only" },
],
intuition: "Fold the new frame into the classifier with a correction term, exactly as if all history had been re-solved, while new tracks get fresh columns.",
why: "Constant-time online learning with absolute memorization and no history storage (Theorem 3.1).",
where: "Sec. III-C continual phase, Eqs. 7-10; R update is the matrix-inversion-lemma recursion.",
paperIds: ["T113"],
},
{
id: "fact-affinity",
label: "FAC affinity and distance",
formula: "O_hat_k = fact(X_k^{(ReID)} W_et) W_hat^{(k-1)};  A_k = active-track columns;  D_k = 1 1^T - A_k",
variables: [
{ symbol: "O_hat_k", meaning: "N_k x d_T(k-1) affinity of each target to every known track" },
{ symbol: "A_k", meaning: "N_k x M_k affinity submatrix over active tracks (no softmax, preserving confidence scale)" },
{ symbol: "D_k", meaning: "affinity-distance matrix fused with Kalman motion for Hungarian matching" },
],
intuition: "Ask the continually trained classifier how strongly each detection belongs to each track, then turn strength into a matching distance.",
why: "History-trained affinities separate occluded targets (distance 0.14-0.37) where cosine saturates at 1.0.",
where: "Sec. III-D affinity estimation, Eqs. 11-13.",
simulator: "reid",
paperIds: ["T113"],
},
{
id: "fact-cost",
label: "Per-frame complexity",
formula: "T = O(N_k d_Tk + N_k^3 + N_k d_et^2 + d_et^2 d_Tk)",
variables: [
{ symbol: "N_k", meaning: "targets in frame k" },
{ symbol: "d_Tk", meaning: "total track identities up to k" },
{ symbol: "d_et", meaning: "transformed dimension (3000)" },
{ symbol: "T", meaning: "dominated by d_et^2 d_Tk, linear in stored tracks" },
],
intuition: "Cost grows with tracks times the square of the expansion width, not with video length, so speed never degrades over time.",
why: "Justifies the real-time claim: no reprocessing of history regardless of sequence length.",
where: "Sec. III-E, Eq. 14.",
paperIds: ["T113"],
},
],
datasets: ["mot17", "mot20"],
metrics: ["hota", "idf1", "mota", "idsw", "fp", "fn", "fps"],
baselines: ["SORT", "DMAN", "DAN", "TubeTK", "CenterTrack", "SOTMOT", "PermaTrack", "TrackFormer", "CSTrack", "FairMOT", "MeMOT", "RelationTrack", "MOTR", "ByteTrack", "OC-SORT", "StrongSORT", "BoT-SORT-ReID", "MotionTrack", "BoostTrack+", "JDE", "SHUSHI"],
results: [
"MOT17 test private (Tab. VI, HOTA/IDF1/MOTA/AssA/DetA/IDSW): FACT 65.3/80.5/80.4/65.9/65.0/1367; FACT+ 66.8/82.9/80.4/68.7/65.3/1026 vs BoostTrack+ 66.4/81.8/80.6/67.7/65.4/1086 and BoT-SORT-ReID 65.0/80.2/80.5/65.5/64.9/1212.",
"MOT20 test (Tab. VII): FACT 65.0/79.8/77.9/66.1/64.2/1263; FACT+ 67.2/83.6/77.5/70.6/64.2/723 vs BoostTrack+ 66.2/81.5/77.2/68.6/64.1/966 and offline SHUSHI 64.3/79.8/74.3/67.5/61.5/706.",
"FAC module gain (Tab. II): MOT17 YOLO-X+FastReID 69.2/81.8/78.4/157 to 69.9/82.7/78.3/165 (HOTA/IDF1/MOTA/IDSW); MOT20 68.9/84.2/86.9/1140 to 70.1/86.2/86.9/1146; consistent +0.4-1.2 HOTA, +0.8-2.0 IDF1 across six detector/ReID combos.",
"Memory length (Tab. III): HOTA/IDF1 69.2/81.8 (3), 69.3/82.0 (5), 69.3/82.1 (10), 69.5/82.2 (20), 69.9/82.7 (all); full history best.",
"Plug-in generality (Tab. IV): FairMOT 57.3/72.9 to 58.1/74.0; Bot-SORT 69.2/81.8 to 69.9/82.7; JDE 35.5/51.2/43.6/23.7 to 40.0/52.3/43.6/20.5; CSTrack 55.7/68.8 to 56.3/69.7 (HOTA/IDF1, MOTA unchanged).",
"Components (Tab. V): IoU-only 55.8/60.2/71.0/836; plus cosine 68.3/79.5/77.9/307; plus CMC 69.2/81.8/78.4/157; plus FAC 69.9/82.7/78.3/165.",
"Speed: FACT+ 12.3 FPS MOT17 / 3.2 MOT20 on 2080Ti; 21.2 FPS MOT17 on RTX4090 (FP16, batch 1).",
],
ablations: [
"Occlusion case study (Fig. 3, ID 6): FAC distance falls 0.37 to 0.14 over frames 623-629 while cosine stays 1.0; target tracked through partial occlusion.",
"Distance histograms (Fig. 4): unassociated pairs pile near 1.0, associated pairs spread, confirming discriminative affinity; new tracks fall back to instant association until initialized.",
"FPS cost of FAC (Tab. II): minus 0.3-0.4 on large models (YOLO-X+FastReID), minus 3.1-3.6 on light FairMOT pairs, for near-large-model gains.",
],
limitations: {
authorStated: [
"FAC needs a few frames to initialize each new track; while confidence is low the instant association path handles matching instead.",
"Gains are association-only: MOTA is essentially unchanged across all plug-in experiments (Tabs. II, IV, V).",
],
evident: [
"ID switches sometimes rise slightly with FAC (e.g. 157 to 165 MOT17, 1140 to 1146 MOT20 in Tab. II) despite higher IDF1.",
"Closed-form recursion assumes association labels are correct, so a wrong match is baked into R(k) and W_hat permanently.",
],
},
assumptions: [
"Each track is one class and past association labels are trustworthy training targets.",
"Fixed random ET projection plus linear classifier separates tracks (no representation learning online).",
"Low FAC confidence reliably flags uninitialized tracks for instant-association fallback.",
],
computation:
"ET dim 3000; YOLOX-X plus FastReID SBS-50; CMC; per-frame cost linear in tracks (Eq. 14); 2080Ti FP32 for ablations, RTX4090 FP16 batch-1 for real-time FPS.",
relations: [
{ to: "T061", type: "builds-on", note: "YOLOX detector plus training schedule and instant association follow ByteTrack; FACT adds continual appearance learning." },
{ to: "T043", type: "improves", note: "FACT+ beats FairMOT 66.8 vs 59.3 HOTA MOT17; plug-in lifts FairMOT itself by 0.8/1.1." },
{ to: "T075", type: "improves", note: "FACT+ beats BoT-SORT-ReID 66.8 vs 65.0 HOTA MOT17 and 67.2 vs 63.3 MOT20 with same extractor." },
{ to: "T005", type: "uses-as-baseline", note: "SORT in Tabs. VI-VII; beaten 66.8 vs 34.0 HOTA MOT17." },
{ to: "T073", type: "uses-as-baseline", note: "OC-SORT in benchmark tables; beaten 66.8 vs 63.2 HOTA MOT17." },
{ to: "T082", type: "uses-as-baseline", note: "MotionTrack in benchmark tables; beaten 66.8 vs 65.1 HOTA MOT17, 67.2 vs 62.8 MOT20." },
],
concepts: ["reid", "memory-network", "learned-association", "data-association", "occlusion", "online-vs-offline"],
impact:
"Introduced exact recursive continual learning to MOT appearance modelling, showing full-history training can run in real time and stack with any instant association advance.",
},
{
id: "T114",
arxiv: "2411.17468",
title: "Adversarial Bounding Boxes Generation (ABBG) Attack against Visual Object Trackers",
shortTitle: "ABBG Attack",
year: 2024,
authors: ["Fatemeh Nourilenjan Nokabadi", "Jean-François Lalonde", "Christian Gagné"],
venue: "AdvML-Frontiers'24 Workshop @ NeurIPS'24",
fileName: "2411.17468v1.pdf",
task: "single-object",
tags: ["adversarial-attack", "white-box", "transformer-tracker", "robustness-evaluation", "single-object-tracking"],
difficulty: "intermediate",
summary:
"ABBG is a white-box adversarial attack that works on transformer trackers using only the single predicted bounding box as its attack proxy. From that box it samples k random shifted/scaled adversarial boxes, keeps the ones close to the prediction via an adaptive IoU threshold, and backpropagates a smoothed-L1 regression loss to perturb the search region. With 10 iterations in an epsilon-ball of 10 it drives TransT-M, ROMTrack and MixFormer scores to nearly zero on GOT-10k and is the first white-box attack applicable to all three.",
problem:
"Transformer trackers predict one box per frame instead of a candidate list with classification/regression labels or heatmaps, so existing white-box attacks (SPARK, RTAA need labels; CSA needs heatmaps; TrackPGD needs masks) have no attack proxy on trackers such as ROMTrack and MixFormer and cannot be applied to compare their robustness.",
background: ["sot", "siamese", "attention", "iou", "success-plot"],
previousWork: [
{
name: "Label-based white-box attacks (SPARK, RTAA)",
limitation:
"Need classification and regression labels inferred by the tracker, so they are inapplicable to transformer trackers that output only a single box.",
whyThisPaper:
"ABBG replaces labels with k generated adversarial boxes around the single prediction, applying to TransT-M, ROMTrack and MixFormer alike.",
},
{
name: "Heatmap/mask-proxy attacks (CSA, TrackPGD)",
limitation:
"CSA needs heatmaps and TrackPGD needs binary masks; neither proxy exists in all transformer pipelines, and CSA must transfer from SiamRPN++ as a black-box.",
whyThisPaper:
"ABBG uses only the predicted box, beating TrackPGD on TransT-M GOT-10k AO 0.027 vs 0.388 and applying where TrackPGD cannot.",
},
{
name: "Black-box IoU attack",
limitation:
"No gradients, so only mild degradation (TransT-M GOT-10k AO 0.587, 20.03% drop) and no diagnostic value about gradient robustness.",
whyThisPaper:
"ABBG exploits transformer gradients directly, reaching 96.32% AO drop on the same tracker and dataset.",
},
],
researchGap:
"No white-box attack was applicable across the new transformer tracker family with a single-box output.",
contribution: [
"ABBG white-box attack manipulating only the single predicted bounding box, applicable to TransT-M, ROMTrack and MixFormer.",
"Adversarial box generation: k=1024 random boxes from uniform translation (0.1-0.4) and scale (0.7-0.9) sets around the prediction, filtered by an adaptive IoU threshold keeping boxes above 80% of all IoUs.",
"Adversarial loss as smoothed-L1 regression between predicted box and retained adversarial boxes, backpropagated for 10 iterations inside epsilon-ball 10.",
"Evaluation on GOT-10k, UAV123 and VOT2022-ST showing first-place white-box damage in at least one metric per dataset and near-zero scores on ROMTrack/MixFormer.",
"Sparsity/imperceptibility analysis (L1 norm, SSIM) ranking ABBG second after SPARK among white-box attacks.",
],
method: {
pipeline: ["predict-box", "generate-adversarial-boxes", "iou-threshold-filter", "regression-loss", "backpropagate-perturbation", "evaluate"],
architecture:
"Attack wrapper around frozen transformer trackers (TransT-M, MixFormer, ROMTrack); no tracker retraining, perturbation applied to the search region per frame starting from the second frame.",
association: "Not a tracker: positive/negative sample split of the k boxes by adaptive IoU threshold per attack iteration.",
optimization:
"10 attack iterations per frame like RTAA/SPARK; perturbation clipped in epsilon-ball with epsilon = 10; translation sampled U(0.1,0.4), scale U(0.7,0.9), k = 1024.",
loss:
"Smoothed-L1 regression loss between the tracker-predicted box and each retained adversarial box, summed over retained boxes.",
},
equations: [
{
id: "abbg-generate",
label: "Adversarial box generation",
formula: "x'_i = x + Tx_i, y'_i = y + Ty_i, w'_i = w * s_i, h'_i = h * s_i, i in {1..k}",
variables: [
{ symbol: "x,y,w,h", meaning: "predicted box top-left corner and size" },
{ symbol: "Tx_i,Ty_i", meaning: "x/y translation offsets sampled U(0.1,0.4)" },
{ symbol: "s_i", meaning: "scale factor sampled U(0.7,0.9)" },
{ symbol: "k", meaning: "number of adversarial boxes, fixed 1024" },
],
intuition: "Jitter the predicted box randomly in position and scale to fabricate a cloud of plausible-but-wrong targets.",
why: "Creates the label-free attack proxy: boxes near the prediction act as positives the loss can push against.",
where: "Sec. 3, Eq. 1 and Fig. 2; applied to the predicted box each attack step.",
params: "Larger translation/scale ranges move boxes farther; k=1024 gives a stable adaptive threshold.",
simulator: "iou-track",
paperIds: ["T114"],
},
{
id: "abbg-threshold",
label: "Adaptive IoU positive selection",
formula: "keep box i iff IoU(box_i, box_pred) > zeta, zeta = 80th percentile of all k IoUs",
variables: [
{ symbol: "zeta", meaning: "adaptive threshold recomputed per iteration from the k IoUs" },
{ symbol: "IoU", meaning: "overlap between adversarial box and predicted box" },
],
intuition: "Keep only the boxes closest to the prediction so the loss always has positives to pull toward.",
why: "Fixed thresholds could retain nothing; the 80% rule guarantees positives every iteration.",
where: "Sec. 3 and Fig. 2; selection precedes loss computation.",
simulator: "iou-track",
paperIds: ["T114"],
},
{
id: "abbg-loss",
label: "ABBG adversarial regression loss",
formula: "L_ABBG = SUM_i L_r(b_pred, b*_i), L_r = smoothed-L1",
variables: [
{ symbol: "b_pred", meaning: "tracker-predicted bounding box" },
{ symbol: "b*_i", meaning: "retained adversarial box i" },
{ symbol: "L_r", meaning: "smoothed L1 norm between the two boxes" },
],
intuition: "Penalize disagreement between the prediction and the fabricated boxes, then perturb pixels to maximize tracker error.",
why: "A pure box regression objective needs no labels, heatmaps or masks, hence works on any single-box tracker.",
where: "Sec. 3, Eq. 2; backpropagated through the tracker to perturb the search region.",
paperIds: ["T114"],
},
],
datasets: ["got10k", "uav123", "vot"],
metrics: ["success-auc", "eao", "precision"],
baselines: ["SPARK", "RTAA", "TrackPGD", "IoU-attack", "CSA-via-SiamRPN++", "TransT-M", "ROMTrack", "MixFormer"],
results: [
"GOT-10k TransT-M (Tab. 1): ABBG AO 0.027 (drop 96.32%), SR0.5 0.019 (97.74%), SR0.75 0.001 (99.85%) vs SPARK 0.116/0.065/0.023, RTAA 0.173/0.180/0.112, TrackPGD 0.388/0.429/0.188.",
"GOT-10k ROMTrack (Tab. 1): ABBG AO 0.006 (99.18%), SR0.5 0.001 (99.88%), SR0.75 0.000 (100%); no other white-box attack applicable.",
"GOT-10k MixFormer (Tab. 1): ABBG AO 0.002 (99.71%), SR0.5 0.000 (100%), SR0.75 0.000 (100%); no other white-box attack applicable.",
"VOT2022-ST TransT-M baseline (Tab. 2): ABBG EAO 0.007 (98.71%), accuracy 0.370 (50.27%), robustness 0.018 (97.92%) vs SPARK 0.009/0.261/0.041, RTAA 0.052/0.601/0.088.",
"UAV123 TransT-M (Tab. 3): ABBG success 0.024 (96.55%), precision 0.079 (91.15%) vs SPARK 0.056/0.068, RTAA 0.083/0.120, TrackPGD 0.264/0.441.",
"Sparsity/imperceptibility on VOT2022 TransT-M (Tab. 4): ABBG L1-norm 95.77, SSIM 89.50% vs SPARK 69.98/94.43%, RTAA 113.48/60.14%, TrackPGD 122.52/64.04%; ABBG second in both.",
],
ablations: [
"No component ablation; sensitivity fixed by design: k=1024, translations U(0.1,0.4), scales U(0.7,0.9), 10 steps, epsilon-ball 10 shared with SPARK/RTAA.",
"Accuracy is the only metric where ABBG ranks second (0.370 vs SPARK 0.261 damage on VOT2022-ST TransT-M); it leads on EAO and robustness there.",
"On UAV123 precision SPARK damages slightly more (92.38% vs 91.15% drop) while ABBG leads on success rate (96.55% vs 91.96%).",
],
limitations: {
authorStated: [
"ABBG ranks second to SPARK in perturbation sparsity and imperceptibility (L1 95.77 vs 69.98, SSIM 89.50% vs 94.43%) at fixed 10 iterations.",
"Accuracy damage on VOT2022-ST trails SPARK (50.27% vs 64.92% drop), i.e. not first on every metric.",
],
evident: [
"Evaluated as attacker on only three transformer trackers; no defense or adversarial-training remedy is proposed.",
"White-box only: needs tracker gradients, so nothing is shown about transferable black-box damage.",
],
},
assumptions: [
"Full white-box access to tracker gradients each frame.",
"Single predicted box per frame is available as the attack proxy.",
],
computation:
"10 attack iterations per frame, k=1024 boxes, epsilon-ball 10; perturbation budget matched to SPARK/RTAA.",
relations: [
{ to: "T055", type: "uses-as-baseline", note: "TransT-M victim attacked; TransT cross/self-attention pipeline is the robustness target." },
{ to: "T070", type: "uses-as-baseline", note: "MixFormer victim driven to AO 0.002; first white-box attack shown to apply to it." },
{ to: "T071", type: "uses-as-baseline", note: "OSTrack one-stream joint extraction discussed as the transformer line ABBG generalizes across." },
{ to: "T028", type: "uses-as-baseline", note: "CSA transfer baseline runs through SiamRPN++; ABBG beats it without any surrogate." },
],
concepts: ["sot", "attention", "iou", "success-plot", "benchmark-design"],
impact: "Gives the first broadly applicable white-box robustness probe for single-box transformer trackers, resetting their reported robustness to near zero under attack.",
},
{
id: "T115",
arxiv: "2501.02467",
title: "DeTrack: In-model Latent Denoising Learning for Visual Object Tracking",
shortTitle: "DeTrack",
year: 2025,
authors: ["Xinyu Zhou", "Jinglun Li", "Lingyi Hong", "Kaixun Jiang", "Pinxue Guo", "Weifeng Ge", "Wenqiang Zhang"],
venue: "NeurIPS 2024",
fileName: "2501.02467v1.pdf",
task: "single-object",
tags: ["denoising-learning", "diffusion-inspired", "vision-transformer", "one-stream", "memory", "real-time"],
difficulty: "advanced",
summary:
"DeTrack reframes tracking as in-model latent denoising: Gaussian noise is added to ground-truth boxes in training, and a denoising ViT (ViT-B, 12 denoising blocks) removes that noise progressively in a single forward pass conditioned on template/search embeddings. A 6-layer box-refining module with trajectory memory and a collaborative visual-memory update then polish the box. DeTrack384 reaches 77.9 AO on GOT-10k, 72.9 AUC on LaSOT and 60.2 on AVisT at 30-42 FPS.",
problem:
"Image-feature regression trackers ignore positional priors and live or die by template-search matching, while coordinate-autoregressive trackers (SeqTrack, ARTrack) train only on boxes present in the training set and degrade on unseen positions/scales; naively running a diffusion model multi-pass would destroy real-time speed.",
background: ["sot", "siamese", "attention", "diffusion-tracking", "memory-network", "success-plot"],
previousWork: [
{
name: "Image-feature regression trackers (Siamese, TransT, OSTrack line)",
limitation:
"Heavily depend on matching quality and use no positional prior, so unseen box positions and scales break them.",
whyThisPaper:
"DeTrack trains on arbitrary noisy boxes around the truth, gaining +3.0 AO over SeqTrack256 on the disjoint GOT-10k test set.",
},
{
name: "Coordinate autoregression (SeqTrack, ARTrack)",
limitation:
"Train only on existing training boxes and need sequential token passes (SeqTrack four decoder passes per box).",
whyThisPaper:
"In-model denoising sees arbitrary positions/sizes in training and finishes in one forward pass at 42 FPS.",
},
{
name: "DiffusionDet-style multi-pass denoising for detection",
limitation:
"Requires multiple decoder passes (DDIM iterations), which tracking frame rates cannot afford.",
whyThisPaper:
"DeTrack decomposes denoising over 12 in-model blocks: single pass 77.1 AO at 42 FPS vs multi-pass-24 75.9 at 29 FPS and 2x FLOPs.",
},
],
researchGap:
"No tracking paradigm combined diffusion-grade robustness to unseen boxes with single-forward-pass real-time denoising.",
contribution: [
"In-model latent denoising paradigm: full Markov denoising chain decomposed over l denoising blocks inside one ViT forward pass, no sampling randomness.",
"Denoising ViT: image attention fusing template/search plus per-block denoising attention, FFN and NoisePred predicting noise subtracted from box embedding (Eqs. 5-10).",
"Box refining and mapping: 6x masked self-attention over trajectory plus cross-attention to image, Softmax over word embedding (800/1200 bins) like ARTrack.",
"Compound memory: FIFO trajectory memory of 7 boxes plus visual memory with collaborative IoU (0.75) and Softmax (0.9) update gating.",
"State of the art on GOT-10k (77.9), LaSOT (72.9), LaSOText (53.6), AVisT (60.2) at 42/30 FPS for 256/384 resolutions.",
],
method: {
pipeline: ["add-noise-to-box", "embed-templates-search-noisybox", "denoising-vit-single-pass", "trajectory-refine", "word-embedding-mapping", "memory-update"],
architecture:
"Denoising ViT-B with MAE init (12 denoising blocks); 6-layer box-refining transformer; IoUNet for template gating; word embedding with 800 (256) / 1200 (384) bins.",
appearanceModel:
"Visual memory of multiple templates (length 3 optimal); collaborative update only if IoU score above 0.75 and Softmax confidence above 0.9.",
motionModel:
"Trajectory memory of previous 7 boxes (FIFO, updated every frame) as positional prior via causally masked self-attention; inference box initialized from frame t-1 prediction.",
trackManagement:
"Visual memory update interval 5 for t<=100, doubled per 100 frames to 160 at t=500; on GOT-10k updated directly, elsewhere gated by IoU/Softmax scores.",
optimization:
"3-stage: (1) 240 epochs full data (COCO/GOT-10k/TrackingNet/LaSOT), lr 8e-5 ViT / 8e-6 refining, decay x0.1 at 192; (2) 60 epochs sequential with trajectory memory, lr 4e-6/4e-7; (3) IoUNet-only 40 epochs lr 1e-4. GOT-10k-only: 120+25 epochs, no IoUNet.",
loss:
"Cross-entropy plus SIoU, same as ARTrack; noise-prediction supervision inside denoising blocks.",
},
equations: [
{
id: "detrack-noise",
label: "Noisy-box training input",
formula: "x_I = sqrt(alpha_bar) * x_0 + eps * sqrt(1 - alpha_bar), eps ~ N(0,I)",
variables: [
{ symbol: "x_0", meaning: "ground-truth box (x1,y1,x2,y2)" },
{ symbol: "x_I", meaning: "noisy box at denoising state I" },
{ symbol: "alpha_bar", meaning: "cumulative variance schedule product over T steps" },
{ symbol: "eps", meaning: "standard Gaussian noise" },
],
intuition: "Corrupt the true box with Gaussian noise so training sees arbitrary sizes and positions.",
why: "Exposes the model to boxes absent from the training set, the source of GOT-10k generalization.",
where: "Sec. 3.1, Eq. 1; noisy box embedded via word embedding to x_I in R(4xC).",
paperIds: ["T115"],
},
{
id: "detrack-markov",
label: "Decomposed denoising chain",
formula: "p(x_0|x_I,c) = p(x_I) * PROD p(x_{i-I/l}|x_i,c); f = {d_1..d_l}",
variables: [
{ symbol: "c", meaning: "condition: visual memory plus search region" },
{ symbol: "d_j", meaning: "j-th denoising block predicting one chain link" },
{ symbol: "l", meaning: "number of blocks, 12" },
],
intuition: "Cut the long denoising chain into 12 links and hand one link to each transformer block.",
why: "Lets one forward pass do what diffusion does in many model runs.",
where: "Sec. 3.1, Eq. 4; each block predicts p(x_{i-I/l}|x_i,c).",
paperIds: ["T115"],
},
{
id: "detrack-imgattn",
label: "Template-search image attention",
formula: "Attn(z,s) = Softmax([qs,qz][ks,kz]^T / sqrt(d)) [vs,vz]",
variables: [
{ symbol: "qs,qz", meaning: "search and template queries" },
{ symbol: "ks,kz,vs,vz", meaning: "search and template keys and values" },
{ symbol: "d", meaning: "key dimensionality" },
],
intuition: "Let template and search patches attend to each other exactly like a one-stream tracker.",
why: "Builds the conditional features every denoising block uses as guidance.",
where: "Sec. 3.2, Eq. 5, inside each ViT block.",
paperIds: ["T115"],
},
{
id: "detrack-denoise",
label: "Denoising attention plus noise subtraction",
formula: "Attn(s,x_i) = Softmax(qx_i ks^T / sqrt(d)) vs; x' = Attn + x_i; x'' = x' + FFN(x'); eps = NoisePred(x''); x_{i-I/l} = x'' - eps",
variables: [
{ symbol: "qx_i", meaning: "noisy-box query" },
{ symbol: "ks,vs", meaning: "search-region key and value" },
{ symbol: "eps", meaning: "predicted noise from two linear layers with ReLU" },
{ symbol: "x_0", meaning: "final denoised embedding: x_I - SUM_j eps_j" },
],
intuition: "Ask the search region what the noise looks like, then subtract it from the box, block after block.",
why: "The ablated core: removing NoisePred costs 2.0 AO; removing denoising attention costs more.",
where: "Sec. 3.2, Eqs. 6-10 and Fig. 2b.",
params: "Box is 4 tokens so the block adds only ~1.8G FLOPs.",
paperIds: ["T115"],
},
],
datasets: ["got10k", "lasot", "others"],
metrics: ["success-auc", "precision", "norm-precision"],
baselines: ["SiamRPN++", "DiMP", "ATOM", "PrDiMP", "Ocean", "TransT", "STARK", "MixFormer", "OSTrack", "SwinTrack", "ROMTrack", "SeqTrack", "ARTrack", "GRM", "AiATrack", "ToMP", "KeepTrack", "TATrack", "CTTrack", "F-BDMTrack"],
results: [
"GOT-10k test (Tab. 2, GOT-only training): DeTrack256 AO 77.1 / SR0.5 86.1 / SR0.75 73.5; DeTrack384 77.9/86.5/74.9 vs ARTrack384 75.5/84.3/74.3 and SeqTrack384 74.8/81.9/72.2.",
"LaSOT (Tab. 2): DeTrack256 AUC 71.3 / PNorm 80.1 / P 76.8; DeTrack384 72.9/81.7/79.1, best among 256/384 peers tabulated.",
"LaSOText (Tab. 2): DeTrack256 47.9/56.6/52.1; DeTrack384 53.6/64.4/60.4 vs SeqTrack384 50.5/61.6/57.5 and ARTrack384 51.9/62.0/58.5.",
"AVisT (Tab. 2): DeTrack256 60.1/69.7/50.6; DeTrack384 60.2/69.1/50.2, +2.4 AUC over SeqTrack384 (57.8).",
"Speed/cost (Tab. 1/4): DeTrack256 53.0G FLOPs at 42 FPS, DeTrack384 117.1G at 30 FPS on RTX 3090.",
"Denoising steps (Tab. 3 GOT-10k): AO climbs 1.1/1.6/4.8/7.5/12.5/21.4/33.1/52.3/65.7/70.2/74.8/77.1 over steps 1-12.",
],
ablations: [
"Paradigm (Tab. 4): single pass 77.1/86.1/73.5 at 53.0G/42FPS vs multi-pass-96 75.7/84.8/72.9 at 424G/8FPS, multi-48 75.7 at 212G/12FPS, multi-24 75.9 at 106G/29FPS.",
"Block (Tab. 5): full 77.1/86.1/73.5; without NoisePred 75.1/84.1/72.0; without denoising attention further down 74.0/84.7/72.5; block adds only 1.8G.",
"Memory on LaSOT (Fig. 5): visual length peaks at 3 (length-1 only 70.2); trajectory improves monotonically 1-7 boxes; no memory 70.2 AUC.",
"Update gating: Softmax threshold 0.9 and IoU threshold 0.75 optimal; IoU 0.85 hurts by starving updates.",
],
limitations: {
authorStated: [
"Recovery after occlusion and out-of-view remains weak: trajectory memory helps reacquire the target in some cases but challenges like occlusion and disappearance need further work (Sec. 5).",
],
evident: [
"Three-stage training (240+60+40 epochs on 8x RTX 3090) plus resolution-specific word embeddings (800/1200 bins) is heavy to reproduce.",
"GOT-10k protocol disables IoUNet and updates memory directly, so the headline generalization number uses a simplified configuration.",
],
},
assumptions: [
"Gaussian-corrupted boxes cover the test-time distribution of unseen positions and scales.",
"Inference initialization from the previous frame prediction is close enough for 12 blocks to denoise.",
],
computation:
"DeTrack256 53.0G FLOPs 42 FPS; DeTrack384 117.1G 30 FPS (RTX 3090); training 8x RTX 3090, three stages as above.",
relations: [
{ to: "T071", type: "builds-on", note: "One-stream ViT joint extraction/fusion backbone with MAE init; beats OSTrack384 73.7 AO on GOT-10k." },
{ to: "T070", type: "uses-as-baseline", note: "MixFormer rows (70.7 GOT-10k AO, 69.2 LaSOT AUC) beaten in Tab. 2." },
{ to: "T055", type: "uses-as-baseline", note: "TransT rows (67.1 GOT-10k AO, 64.9 LaSOT AUC) beaten in Tab. 2." },
{ to: "T064", type: "uses-as-baseline", note: "SwinTrack rows (72.4 GOT-10k AO, 71.3 LaSOT AUC) beaten in Tab. 2." },
{ to: "T028", type: "uses-as-baseline", note: "SiamRPN++ rows (51.7 GOT-10k AO, 49.6 LaSOT AUC) beaten in Tab. 2." },
],
concepts: ["sot", "attention", "diffusion-tracking", "memory-network", "success-plot"],
impact: "Shows denoising learning can be made real-time for tracking via in-model decomposition, giving a reusable single-pass alternative to autoregressive box prediction.",
},
{
id: "T116",
arxiv: "2502.18143",
title: "LightFC-X: Lightweight Convolutional Tracker for RGB-X Tracking",
shortTitle: "LightFC-X",
year: 2025,
authors: ["Yunfeng Li", "Bo Wang", "Ye Li"],
fileName: "2502.18143v1.pdf",
task: "multimodal-tracking",
tags: ["multimodal-tracking", "rgb-t", "rgb-d", "rgb-e", "lightweight", "convolutional", "cross-attention", "temporal"],
difficulty: "intermediate",
summary:
"LightFC-X is a unified lightweight convolutional RGB-X family built on LightFC: an efficient cross-attention module (ECAM, 0.08M params) fuses template-search integrated features across modalities with joint spatial-channel refinement, and a spatiotemporal template aggregation module (STAM) upgrades fixed+dynamic templates via a freeze-then-finetune paradigm. LightFC-T-ST beats CMD by 4.3 SR / 5.7 PR on LasHeR with 2.6x fewer params at 81 fps, and runs 22 fps on CPU.",
problem:
"Multimodal trackers (TBSI 202M params, OSTrack 92M) are too heavy for UAVs and edge devices, while existing lightweight options (MDNet variants, CMD distillation, TBSI-Tiny) lag in performance; RGB-D/E/Sonar lightweight tracking is barely studied and no unified convolutional family exists.",
background: ["sot", "siamese", "multimodal", "attention", "success-plot"],
previousWork: [
{
name: "Heavy transformer RGB-X trackers (TBSI, CSTNet, ViPT, OSTrack-based)",
limitation:
"202M/144M params and 82.5G/74.7G FLOPs (TBSI/CSTNet); OSTrack baseline 92.1M params, undeployable on cheap edge devices lacking attention optimization.",
whyThisPaper:
"Convolutional LightFC base plus 0.08M ECAM reaches OSTrack-level scores with 12-14x fewer parameters.",
},
{
name: "Lightweight RGB-T (MDNet line, CMD distillation, TBSI-Tiny)",
limitation:
"Shallow MDNet features or distillation still leave a gap to high-performance trackers; TBSI-Tiny stays heavy at 14.9M.",
whyThisPaper:
"LightFC-T beats CMD by 3.7 SR/4.8 PR on LasHeR with 3x fewer params at 2.7x speed.",
},
{
name: "Template-update methods (UpdateNet, Stark-ST, ODTrack, FEAR linear interpolation)",
limitation:
"Built for large trackers or naive linear interpolation, unfit for lightweight models needing deep temporal appearance use.",
whyThisPaper:
"STAM models fixed-dynamic template association with cross-attention and refines via decoupled spatial-channel layers under module finetuning.",
},
],
researchGap:
"No unified lightweight convolutional architecture achieved joint cross-modal plus spatiotemporal refinement across RGB-T/D/E/Sonar with real-time CPU speed.",
contribution: [
"ECAM (0.08M): lightweight cross-attention on template-search integrated features plus joint feature encoding (1x1 downsample, multi-scale 3/5/7 DW convs, channel integration, residual).",
"STAM: cross-attention between fixed first-frame and dynamic templates, decoupled DW-spatial plus channel refinement, linear fusion; trained by freezing the tracker and finetuning STAM only.",
"LightFC-X family (T/D/E/S variants, incl. D-N3 with 3 ECAMs and sonar dual-head) with unified insert-after-TSAIM design validated by variant ablations.",
"SOTA lightweight results on LasHeR, RGBT234, GTOT, DepthTrack, VOT-RGBD2022, VisEvent, RGBS50 with 22 fps CPU / 51 fps Xavier real-time operation.",
],
method: {
pipeline: ["extract-rgb-x-features", "stam-temporal-aggregate", "tsaim-template-search-interaction", "ecam-cross-modal-fusion", "concat-search-features", "rep-center-head"],
architecture:
"LightFC convolutional backbone (C=160, stride 16, Ca=96 refined); TSAIM per modality; ECAM after interaction; rep-center-head (one head, two heads for RGB-Sonar); 6.68M (T), 7.61M (ST), 10.40M (S).",
appearanceModel:
"First-frame fixed template plus dynamic templates aggregated by STAM; update interval 400 frames, confidence threshold 0.7.",
association: "Single-object regression; no data association.",
optimization:
"Two-phase on 2x RTX A6000: phase 1 trains ECAM model 45 epochs (batch 64, 60k pairs/epoch, AdamW wd 1e-4, lr 2e-4, /10 after 10); phase 2 freezes tracker, finetunes STAM 15 epochs.",
loss:
"Weighted focal loss for classification plus GIoU and L1 for regression: L = Lcls + 2*L_iou + 5*L1.",
},
equations: [
{
id: "lightfc-tsaim",
label: "Template-search interaction",
formula: "Ar = Zrt^T Xr, Ax = Zxt^T Xx",
variables: [
{ symbol: "Zrt,Zxt", meaning: "STAM-aggregated RGB and X template features" },
{ symbol: "Xr,Xx", meaning: "RGB and X search-region features" },
{ symbol: "Ar,Ax", meaning: "per-modality template-search response maps" },
],
intuition: "Correlate each modality template with its search area to locate the target per modality.",
why: "Gives the single-modality integrated features ECAM will fuse across modalities.",
where: "Sec. 3.1, Eq. 1; refined by LightFC framework to Xrts, Xxts.",
simulator: "siamese",
paperIds: ["T116"],
},
{
id: "lightfc-fusion",
label: "Cross-modal plus semantic fusion",
formula: "Xrx^st = f(Xrts, Xxts); Xfusion = [Xrx^st; Xr; Xx]",
variables: [
{ symbol: "Xrts,Xxts", meaning: "refined template-search integrated features per modality" },
{ symbol: "Xrx^st", meaning: "ECAM cross-modal integrated feature" },
{ symbol: "Xfusion", meaning: "head input concatenating fused plus original search features" },
],
intuition: "Fuse across modalities, then add back raw search features so fusion cannot erase target semantics.",
why: "Ablation shows this concatenation is the biggest single gain (46.0 to 50.1 SR on LasHeR).",
where: "Sec. 3.1, Eq. 2; RGB-Sonar variant uses per-modality heads (Eq. 3).",
paperIds: ["T116"],
},
{
id: "lightfc-ecam",
label: "Lightweight cross-modal attention",
formula: "Xr_crs = Softmax(Xr' Xx'^T / sqrt(C)) Xr'; Xx_crs = Softmax(Xx' Xr'^T / sqrt(C)) Xx'",
variables: [
{ symbol: "Xr',Xx'", meaning: "tokenized integrated features (N x Ca, Ca=96)" },
{ symbol: "C", meaning: "channel scale (Ca)" },
{ symbol: "Xr_crs,Xx_crs", meaning: "cross-attended features with residual add-back" },
],
intuition: "Let RGB features query X-modality features and vice versa with a tiny attention.",
why: "Cross-modal interaction for 0.08M params; naive cross-attention alone hurts (29.7 SR), full ECAM reaches 50.1.",
where: "Sec. 3.2, Eqs. 4-6, followed by joint spatial-channel encoding (Eqs. 7-10).",
params: "One ECAM optimal for T/E; three for D (N3); more degrades.",
paperIds: ["T116"],
},
{
id: "lightfc-stam",
label: "Spatiotemporal template aggregation",
formula: "Zrt^1,Zrt^i = CrossAttn(Zr1,Zri); Zrt = LN(Linear(Zrt^1c + Zrt^ic) + Zrt^1c + Zrt^ic)",
variables: [
{ symbol: "Zr1", meaning: "fixed first-frame template" },
{ symbol: "Zri", meaning: "dynamic tracking template" },
{ symbol: "Zrt^1c,Zrt^ic", meaning: "spatially+channel refined template features" },
{ symbol: "Zrt", meaning: "spatiotemporal template fed to TSAIM" },
],
intuition: "Associate what the target looked like at the start with what it looks like now, then merge both views.",
why: "Adds temporal appearance modeling to a spatial-only lightweight tracker via finetuning alone.",
where: "Sec. 3.3, Eqs. 12-15; spatial DW-conv plus channel MLP refinement.",
params: "Freezing tracker and finetuning STAM beats joint training (53.8 vs 48.5 F on DepthTrack).",
paperIds: ["T116"],
},
{
id: "lightfc-loss",
label: "Training loss",
formula: "L = Lcls + 2 * Liou + 5 * L1",
variables: [
{ symbol: "Lcls", meaning: "weighted focal classification loss" },
{ symbol: "Liou", meaning: "GIoU box regression loss" },
{ symbol: "L1", meaning: "L1 box regression loss" },
],
intuition: "Classify target vs background while regressing tight boxes with overlap plus coordinate penalties.",
why: "Same recipe as LightFC, keeping the family comparable.",
where: "Sec. 3.4, Eq. 16.",
paperIds: ["T116"],
},
],
datasets: ["lasheR", "others"],
metrics: ["success-auc", "precision", "eao"],
baselines: ["LightFC", "OSTrack", "CMD", "TBSI-Tiny", "MANet++", "DAFNet", "MFNet", "LiteTrack", "HiT-Tiny", "HiT-Small", "SMAT", "DDiMP", "KeepTrack", "SiamRCNN", "ProTrack", "DeT", "SCANet"],
results: [
"LasHeR (Tab. 3): LightFC-T PR 63.8 / NPR 59.9 / SR 50.1; LightFC-T-ST 64.7/60.8/50.7 vs CMD 59.0/54.6/46.4 and TBSI-Tiny 61.7/57.8/48.9; T uses 6.68M vs CMD 19.90M at 81 vs 30 fps.",
"RGBT234 (Tab. 3): LightFC-T-ST MPR 83.4 / MSR 60.3 vs CMD 82.4/58.4; MSR near MFNet 60.1 at 10.6x fewer params.",
"GTOT (Tab. 3): LightFC-T MPR 88.2 / MSR 71.1; ST 88.7/71.4.",
"DepthTrack (Tab. 1): LightFC-D-N3-ST F 0.538 / Pr 0.557 / Re 0.520 vs OSTrack 0.529/0.536/0.522 at 7.76 vs 92.18M params (+9.8 F / +10.7 Pr over LightFC baseline).",
"VOT-RGBD2022 (Tab. 2): LightFC-D-N3-ST EAO 0.605 / A 0.760 / R 0.778, near KeepTrack/OSTrack level.",
"VisEvent (Tab. 5): LightFC-E SR 52.9 / PR 68.7; ST 53.2/69.1 vs OSTrack 53.4/69.5 at 13.8x fewer params.",
"RGBS50 (Tab. 4): LightFC-S RGB SR 58.6 / PR 59.6 / NPR 65.0; sonar 39.5/58.3/52.0, beating LightFC by 2.4-3.6 RGB and 17.3/30.2 sonar points.",
"Speed (Tab. 6): 22 fps on Intel I9 CPU, 34 fps on GTX1050Ti, 51 fps on Jetson Xavier (LightFC-X 6.68M).",
],
ablations: [
"ECAM buildup (Tab. 7): baseline 37.8 LasHeR SR; +raw cross-attention drops to 29.7; +Conv1x1 43.5; +spatial 46.0; +channel 46.5; +residual 46.0; +concat 50.1; STAM stack reaches 50.7.",
"ECAM count (Tab. 8): 1x best for T/E (50.1 LasHeR SR, 53.2 VisEvent SR); 3x best for D (N3); 2x/4x degrade.",
"Variants (Tab. 9): insert-after-TSAIM (LightFC-X) 50.1/52.9 beats in-backbone (45.4/51.3) and before-TSAIM (44.1-47.4/48.3-50.0).",
"STAM training (Tab. 10): freeze-then-finetune beats joint training on all three benchmarks (e.g. DepthTrack F 53.8 vs 48.5).",
],
limitations: {
authorStated: [
"Spatiotemporal feature use remains deficient; future work is a sequence-training paradigm to strengthen temporal modeling in lightweight multimodal trackers (Sec. 5).",
],
evident: [
"STAM adds a second training phase and fixed update schedule (interval 400, threshold 0.7) with no online adaptation analysis.",
"RGB-Sonar needs a separate dual-head fork due to spatial misalignment, breaking full unification.",
],
},
assumptions: [
"RGB and X modalities are spatially aligned except RGB-Sonar, which gets its own dual-head path.",
"Freezing the spatial model preserves discrimination when adding temporal capacity.",
],
computation:
"6.68M (T) / 6.91M (D-N3) / 10.40M (S) params; 169/156/150 fps on GTX3090Ti, 22 fps CPU, 51 fps Xavier; training 2x RTX A6000, 45+15 epochs.",
relations: [
{ to: "T095", type: "extends", note: "LightFC convolutional base, TSAIM and rep-center-head kept; +12.3 SR / +15.2 PR over LightFC on LasHeR." },
{ to: "T071", type: "uses-as-baseline", note: "OSTrack matched on DepthTrack/VisEvent at 12-14x fewer params (92.18M vs ~7M)." },
{ to: "T083", type: "uses-as-baseline", note: "ViPT prompt-learning line cited as heavy (43G FLOPs class); ECAM replaces prompts with 0.08M fusion." },
],
concepts: ["sot", "siamese", "multimodal", "attention", "benchmark-design"],
impact: "First unified convolutional family proving multimodal tracking can be both state-of-the-art-lightweight and CPU-real-time across four modality pairs.",
},
{
id: "T117",
arxiv: "2504.01457",
title: "Deep LG-Track: An Enhanced Localization-Confidence-Guided Multi-Object Tracker",
shortTitle: "Deep LG-Track",
year: 2025,
authors: ["Ting Meng", "Chunyun Fu", "Xiangyan Yan", "Zheng Liang", "Pan Ji", "Jianwen Wang", "Tao Huang"],
fileName: "2504.01457v1.pdf",
task: "multi-object",
tags: ["tracking-by-detection", "kalman-filter", "adaptive-association", "appearance-update", "localization-confidence"],
difficulty: "intermediate",
summary:
"Deep LG-Track upgrades the TBD tracker LG-Track with three quality-aware enhancements: an adaptive Kalman measurement-noise covariance scaled by detection confidence and disappearance length, an adaptive cost matrix weighting IoU by localization confidence and appearance by detection confidence, and strong dynamic appearance (SDA) blending history/current embeddings by classification and localization confidences. It leads MOT17 test with 65.9 HOTA / 66.4 AssA / 81.4 IDF1 / 999 IDSW.",
problem:
"TBD trackers use fixed Kalman noise covariances, fixed motion-appearance fusion weights, and fixed EMA appearance blending, so low-quality detections (bad localization but clear appearance, or vice versa) corrupt prediction, association and ReID features alike, causing ID switches on MOT17/MOT20.",
background: ["mot", "tracking-by-detection", "kalman", "motion-model", "data-association", "cost-matrix", "reid", "hota"],
previousWork: [
{
name: "Fixed-covariance Kalman TBD trackers (SORT, JDE, FairMOT, ByteTrack, OC-SORT)",
limitation:
"Uniform measurement-noise covariance assumes identical detection quality and trusts stale predictions of long-lost trajectories, so errors accumulate through occlusions.",
whyThisPaper:
"ACMN scales noise by detection confidence and disappearance frames, lifting LG-Track HOTA 77.72 to 78.06 on MOT17 train.",
},
{
name: "Fixed-fusion cost matrices (BoT-SORT min-pick, DeepSORT gating, Stadler weighted sum)",
limitation:
"Experimentally fixed motion-appearance weights cannot serve boxes with clear appearance but poor localization and vice versa (Fig. 4 MOT20 examples).",
whyThisPaper:
"Adaptive matrix weights IoU by localization confidence and appearance by detection confidence: largest single gain +0.54 HOTA.",
},
{
name: "EMA and detection-confidence DA appearance update (DeepSORT buffer, BoT-SORT/StrongSORT EMA, Deep OC-SORT DA)",
limitation:
"Fixed blending or blending by overall detection confidence conflates localization accuracy with appearance clarity.",
whyThisPaper:
"SDA conditions blending on classification and localization confidences separately, adding +0.21 HOTA / +0.49 AssA over DA.",
},
],
researchGap:
"No TBD tracker jointly adapted Kalman uncertainty, motion-appearance fusion and appearance blending to per-box localization accuracy and appearance clarity.",
contribution: [
"Adaptive covariance of measurement noise (ACMN): factor alpha from detection confidence and lost-frame count scaling preset R, trusting measurements more for long-lost re-matches.",
"Adaptive cost matrix (ACM): IoU weighted by localization confidence, cosine appearance cost weighted by detection confidence, used uniformly at all association levels.",
"Strong dynamic appearance (SDA): EMA weight from classification confidence and localization confidence with separate thresholds, replacing fixed and DA blending.",
"MOT17 test leadership (65.9 HOTA, 66.4 AssA, 81.4 IDF1, 999 IDSW) and MOT20 top HOTA (64.0); ACMN shown transferable to ByteTrack/BoT-SORT.",
],
method: {
pipeline: ["detect-yolox", "kalman-predict-acmn", "adaptive-cost-associate", "update-tracks", "sda-appearance-update"],
architecture:
"LG-Track TBD framework with YOLOX detector (ByteTrack-trained weights) and FastReID SBS-50 appearance (BoT-SORT); four trajectory states New/Tracked/Lost/Removed.",
motionModel:
"Per-trajectory Kalman filter with adaptive measurement-noise covariance R-tilde = alpha * R; alpha falls with detection confidence and, for re-matched Lost tracks, with disappearance length.",
appearanceModel:
"FastReID SBS-50 embeddings; cosine distance appearance cost; SDA blending of trajectory history and new detection embedding.",
association:
"Single adaptive cost matrix for all levels fusing IoU and appearance with per-detection localization/detection confidence weights, then matching.",
trackManagement:
"New/Tracked/Lost/Removed states; detection confidence affixed to matched trajectory; Lost tracks retained up to M frames with m counted disappearances.",
},
equations: [
{
id: "deeplg-acmn",
label: "Adaptive measurement covariance",
formula: "R_tilde = alpha * R",
variables: [
{ symbol: "R", meaning: "preset measurement-noise covariance" },
{ symbol: "R_tilde", meaning: "adaptive covariance actually used in update" },
{ symbol: "alpha", meaning: "quality factor from detection confidence and disappearance" },
],
intuition: "Scale the filter trust knob per measurement: good detections get small noise (trusted), bad ones big noise (ignored).",
why: "Fixed R pretends all boxes are equally reliable, which illumination, distance and occlusion falsify.",
where: "Sec. III-A, Eq. 1; applied in every trajectory Kalman update.",
simulator: "kalman",
paperIds: ["T117"],
},
{
id: "deeplg-alpha",
label: "Quality factor from confidence and disappearance",
formula: "alpha = g(d, tau, m, M): falls as detection confidence d rises above threshold tau; for Lost re-matches falls as disappeared frames m grows toward cap M",
variables: [
{ symbol: "d", meaning: "detection confidence affixed to the trajectory" },
{ symbol: "tau", meaning: "detection confidence threshold" },
{ symbol: "m", meaning: "consecutive disappeared frames (0 for New/Tracked)" },
{ symbol: "M", meaning: "maximum retained disappearance length" },
],
intuition: "Trust confident boxes, and distrust stale predictions of tracks that were gone a long time.",
why: "Prediction error accumulates without updates, so re-found lost tracks should lean on the new measurement.",
where: "Sec. III-A, Eqs. 2-3 and Fig. 3.",
params: "Higher tau demands more confident boxes before trusting; larger M keeps lost tracks longer.",
simulator: "kalman",
paperIds: ["T117"],
},
{
id: "deeplg-cost",
label: "Adaptive motion-appearance cost",
formula: "C(i,j) = w_loc(loc_i) * motion(1 - IoU_ij) + w_det(det_i) * appear_ij",
variables: [
{ symbol: "C(i,j)", meaning: "cost between detection i and trajectory j" },
{ symbol: "IoU_ij", meaning: "box overlap motion term" },
{ symbol: "appear_ij", meaning: "cosine-distance appearance term" },
{ symbol: "loc_i", meaning: "localization confidence weighting motion" },
{ symbol: "det_i", meaning: "detection confidence weighting appearance" },
],
intuition: "Let each box vote with its reliable side: sharp boxes steer by position, clear-looking boxes by appearance.",
why: "Fixed weights fail Fig. 4 cases (low-loc/high-cls vs high-loc/occluded); per-box weights fix them.",
where: "Sec. III-B, Eq. 4; uniform across all association levels unlike LG-Track.",
simulator: "assoc-cost",
paperIds: ["T117"],
},
{
id: "deeplg-ema-sda",
label: "EMA appearance update and SDA weight",
formula: "e_t = a * e_{t-1} + (1 - a) * f_t; SDA sets a from (cls conf vs thr_c) and (loc conf vs thr_l)",
variables: [
{ symbol: "e_t,e_{t-1}", meaning: "trajectory appearance feature now and last frame" },
{ symbol: "f_t", meaning: "newly matched detection feature" },
{ symbol: "a", meaning: "fixed (EMA/DA) or adaptive (SDA) history weight" },
{ symbol: "cls conf, loc conf", meaning: "appearance-clarity and localization-accuracy signals with thresholds thr_c, thr_l" },
],
intuition: "Remember what the person looked like, but listen to the new frame only as much as its clarity and alignment deserve.",
why: "Low-quality detections otherwise poison the identity embedding and cause switches.",
where: "Sec. III-C, Eqs. 5-7; DA (Eq. 6, detection-confidence only) vs SDA (Eq. 7, split signals).",
simulator: "reid",
paperIds: ["T117"],
},
],
datasets: ["mot17", "mot20"],
metrics: ["hota", "assa", "deta", "idf1", "mota", "idsw"],
baselines: ["SORT", "DeepSORT", "FairMOT", "JDE", "ByteTrack", "BoT-SORT", "StrongSORT++", "OC-SORT", "Deep-OC-SORT", "MotionTrack", "UTM", "BPMTrack", "ConfTrack", "UCMCTrack", "SparseTrack", "LG-Track", "Hybrid-SORT-ReID", "GHOST", "FineTrack", "TrackFormer"],
results: [
"MOT17 test private detector (Tab. I, YOLOX): Deep LG-Track HOTA 65.9 / AssA 66.4 / IDF1 81.4 / MOTA 81.3 / DetA 65.5 / IDSW 999, best in HOTA, AssA, IDF1, IDSW.",
"MOT20 test (Tab. II): HOTA 64.0 (best) / AssA 64.1 / IDF1 78.4 / MOTA 77.6 / DetA 64.0 (second) / IDSW 1081 vs LG-Track 63.4/62.9/77.4/77.8/64.1/1161.",
"Association gain: AssA +1.0 on MOT17 (65.4 to 66.4) and +1.2 on MOT20 (62.9 to 64.1) over LG-Track.",
"MOT17 train ablation (Tab. III): baseline 77.72/76.35/86.54/90.68 (HOTA/AssA/IDF1/MOTA); +ACMN 78.06/77.01/87.04/90.69; +ACM 78.60/77.63/87.89/90.62; +SDA 78.81/78.12/88.12/90.43.",
"ACMN vs NSAKF (Tab. IV): ACMN beats NSAKF on ByteTrack (75.73 vs 75.80 HOTA but +AssA/IDF1/MOTA), BoT-SORT (77.13 vs 76.44) and LG-Track (78.06 vs 77.39).",
"SDA vs DA (Tab. V): SDA 78.81/78.12/88.12 vs DA 78.60/77.54/87.72 (HOTA/AssA/IDF1); MOTA dips 90.62 to 90.43.",
],
ablations: [
"ACM is the largest single gain (+0.54 HOTA to 78.60); ACMN adds +0.34; SDA adds +0.21 HOTA / +0.49 AssA.",
"NSAKF hurts BoT-SORT (-0.31 HOTA) and LG-Track (-0.33) while ACMN helps all three frameworks tested.",
"DA alone gives no HOTA gain over ACMN+ACM (78.60 flat, AssA -0.09); splitting confidences in SDA recovers +0.21/+0.49.",
"Qualitative MOT17-05/10: LG-Track ID switches (56-to-66, ID 2-to-36; 59/48/64 flips) disappear under Deep LG-Track.",
],
limitations: {
authorStated: [
"No dedicated limitations section; the stated open challenge motivating ACMN is prediction-error accumulation for trajectories lost over many frames without update.",
],
evident: [
"SDA trades MOTA for association (90.62 to 90.43 on MOT17 train) and adds four thresholds (detection, classification, localization, disappearance cap) to tune.",
"Headline test numbers ride the ByteTrack-trained YOLOX plus FastReID SBS-50 stack under the private-detector protocol, bounding cross-detector comparability.",
],
},
assumptions: [
"YOLOX localization and detection confidences faithfully separate alignment error from appearance clarity.",
"Disappearance length is a sufficient proxy for accumulated prediction error.",
],
computation:
"Detector YOLOX plus FastReID SBS-50; no speed/FPS reported; thresholds tau, thr_c, thr_l and cap M user-defined.",
relations: [
{ to: "T061", type: "builds-on", note: "ByteTrack-trained YOLOX detector weights reused; ACMN transfer also demonstrated inside ByteTrack." },
{ to: "T073", type: "uses-as-baseline", note: "OC-SORT beaten 65.9 vs 63.2 HOTA MOT17 and 64.0 vs 62.1 MOT20." },
{ to: "T043", type: "uses-as-baseline", note: "FairMOT-style joint detection-ReID TBD line used as reference point." },
{ to: "T067", type: "uses-as-baseline", note: "StrongSORT++ beaten 65.9 vs 64.4 HOTA MOT17; EMA appearance lineage extended by SDA." },
{ to: "T005", type: "conceptual-successor", note: "SORT Kalman TBD lineage carried forward with quality-adaptive noise." },
],
concepts: ["mot", "tracking-by-detection", "kalman", "data-association", "cost-matrix", "reid", "hota", "mota", "idf1"],
impact: "Establishes per-box quality (split localization vs appearance confidence) as the control signal for filtering, fusion and feature update in TBD tracking.",
},
{
id: "T118",
arxiv: "2505.00752",
title: "DARTer: Dynamic Adaptive Representation Tracker for Nighttime UAV Tracking",
shortTitle: "DARTer",
year: 2025,
authors: ["Xuzhao Li", "Xuchen Li", "Shiyu Hu"],
fileName: "2505.00752v2.pdf",
task: "single-object",
tags: ["nighttime-tracking", "uav", "vision-transformer", "dynamic-computation", "feature-fusion", "end-to-end"],
difficulty: "intermediate",
summary:
"DARTer is an end-to-end nighttime UAV tracker: a Dynamic Feature Blender (DFB) fuses static and dynamic template features (plus overlapped patches) via bidirectional cross-attention, and a Dynamic Feature Activator (DFA) skips ViT blocks whose activation probability falls below 0.3. With plain CE plus SIoU loss it tops five nighttime benchmarks (85.2 precision on NAT2024-1, 64.9 on NAT2021-L) at 74 fps.",
problem:
"Nighttime UAV tracking collapses under extreme illumination, low contrast and viewpoint change; light-enhancer pipelines cost extra networks and break end-to-end training, while domain-adaptation/prompt methods (TDA-Track, DCPT) add data and compute redundancy and still ignore multi-perspective dynamic features.",
background: ["sot", "siamese", "attention", "memory-network", "success-plot"],
previousWork: [
{
name: "Light-enhancement plus daytime tracker (HighlightNet, ADTrack and enhancer+filter lines)",
limitation:
"Extra enhancement networks raise cost and prevent end-to-end training.",
whyThisPaper:
"DARTer needs no enhancer: DFB blends dark multi-view features directly, reaching 85.2 vs enhancer-line scores at 74 fps.",
},
{
name: "Domain-adaptation nighttime tracking (UDAT, TDA-Track prompt adaptation)",
limitation:
"Needs scarce high-quality nighttime training data and redundant adaptation machinery.",
whyThisPaper:
"DFB/DFA learn dynamic nighttime representations from mixed day+night training without adaptation modules.",
},
{
name: "Heavy prompt/curriculum nighttime trackers (DCPT dark prompts, MambaNUT curriculum)",
limitation:
"Dark-prompt redundancy (DCPT 92.9M) and optimization complexity for marginal robustness.",
whyThisPaper:
"DARTer beats DCPT on all five benchmarks (e.g. NAT2024-1 85.2 vs 80.9 P) with plain CE+SIoU loss and adaptive depth.",
},
],
researchGap:
"No end-to-end nighttime UAV tracker jointly exploited multi-perspective dynamic template features while adaptively skipping redundant transformer computation.",
contribution: [
"Dynamic Feature Blender: bidirectional cross-attention fusion of static/dynamic templates and their overlapped patches into robust nighttime representations.",
"Dynamic Feature Activator: per-layer activation probability from all tokens via linear+conv sigmoid gate, skipping blocks below threshold 0.3 (all but first block eligible).",
"Streamlined training: single loss L = 2*Lce + 2*LsIoU with no multi-task or hand-designed objectives.",
"Five-benchmark leadership: NAT2024-1, NAT2021, UAVDark135, NAT2021-L, DarkTrack2021, with ablations showing +3.7 P from DFB+DFA combined.",
],
method: {
pipeline: ["patch-embed-search-static-dynamic", "dfb-blend-templates", "dfa-gated-overlapped-vit", "corner-style-head", "dynamic-template-interval-update"],
architecture:
"Overlapped ViT backbone; DFB cross-attention on initial plus overlapped patches (search 16x16/15x15, template 8x8/7x7, patch 16x16); 4-layer Conv-BN-ReLU head predicting offset, size, classification score; 80.9M params.",
appearanceModel:
"Static template plus dynamic template updated at fixed intervals; overlapped patches strengthen inter-patch association across views.",
trackManagement:
"Dynamic template refreshed on fixed schedule; DFA re-evaluates activation at each layer from previous-block fused tokens.",
optimization:
"AdamW, batch 32, 150 epochs x 60k pairs, lr 1e-4 with x0.1 decay after 120 epochs; 4x A5000 train, RTX-3090 test; threshold beta 0.3.",
loss:
"L = 2 * cross-entropy + 2 * SIoU; no complex multi-task loss.",
},
equations: [
{
id: "darter-dfb",
label: "Dark feature blending",
formula: "fZs' = CA(fZs, fZd); fZd' = CA(fZd, fZs); fZ = Concat(fZs', fZd')",
variables: [
{ symbol: "fZs,fZd", meaning: "static and dynamic template features (initial or overlapped patches)" },
{ symbol: "CA(a,b)", meaning: "cross-attention with a as Q and b as K,V" },
{ symbol: "fZ", meaning: "fused nighttime representation (fZo for overlapped patches)" },
],
intuition: "Look at the target from the start-frame viewpoint and the recent viewpoint, and let each query the other.",
why: "Fuses state changes and essential characteristics across views for illumination-robust representation (+1.8 P alone).",
where: "Sec. 2.1, Eqs. 1-3; same operation repeated on overlapped-patch features.",
paperIds: ["T118"],
},
{
id: "darter-dfa",
label: "Dynamic layer activation",
formula: "r = v^T t_{1:k}; p = sigmoid(Linear(r) + Conv(r)); activate iff p > beta",
variables: [
{ symbol: "t_{1:k}", meaning: "all search+template tokens from previous ViT block" },
{ symbol: "v", meaning: "standard-normal feature extraction vector" },
{ symbol: "p", meaning: "activation probability in (0,1)" },
{ symbol: "beta", meaning: "threshold 0.3; first ViT block always active" },
],
intuition: "Glance at current features and decide whether the next transformer layer is worth running.",
why: "Skips redundant depth adaptively, keeping 74 fps without losing nighttime accuracy (+0.9-1.0 P in ablation).",
where: "Sec. 2.2, Eq. 4 and Fig. 1b; inactive block output bypassed to the next layer.",
params: "Higher beta skips more layers (faster, weaker); 0.3 is the reported operating point.",
paperIds: ["T118"],
},
{
id: "darter-loss",
label: "Streamlined training loss",
formula: "L = 2 * Lce + 2 * LsIoU",
variables: [
{ symbol: "Lce", meaning: "softmax cross-entropy classification loss" },
{ symbol: "LsIoU", meaning: "SIoU box regression loss" },
{ symbol: "2,2", meaning: "fixed loss weights" },
],
intuition: "Score the right location and regress a tight box with two plain terms.",
why: "Deliberate contrast to curriculum/multi-task nighttime losses: simplicity with SOTA results.",
where: "Sec. 2.3; head outputs offset o, size s and score p with argmax location.",
paperIds: ["T118"],
},
],
datasets: ["others"],
metrics: ["precision", "success-auc"],
baselines: ["TCTrack", "TCTrack++", "MAT", "HiT-Base", "Aba-ViTrack", "SGDViT", "TDA-Track", "AVTrack-DeiT", "DCPT", "MambaNUT", "SiamRPN++", "Ocean", "HiFT", "SiamAPN++", "UDAT-BAN", "UDAT-CAR", "DIMP", "SiamAPN-SCT", "DIMP-SCT"],
results: [
"NAT2024-1 (Tab. 1): DARTer P 85.2 / PNorm 80.1 / AUC 65.6 vs MambaNUT 83.3/76.9/63.6 and DCPT 80.9/75.4/62.1.",
"NAT2021 (Tab. 1): 70.2/63.7/53.2, best AUC; UAVDark135 (Tab. 1): 71.6/72.1/58.2, best on all three.",
"NAT2021-L (Tab. 2): 64.9/58.6/50.9 vs DCPT 58.6/54.6/47.4 (+6.3 P / +4.0 PNorm / +3.5 AUC).",
"DarkTrack2021 (Tab. 3): 67.6/64.8/54.5, best P/PNorm/AUC vs DCPT 66.7/64.6/54.0 and DIMP50-SCT 67.7/63.3/52.1.",
"Efficiency (Tab. 1): 74 fps on RTX-3090 at 80.9M params vs DCPT 35 fps/92.9M, AVTrack 212 fps/7.9M.",
"Module ablation on NAT2024-1 (Tab. 4): full 85.2/80.1/65.6; DFA-only 84.3/79.5/64.6; DFB-only 83.4/78.6/64.2; neither 81.5/76.9/62.3.",
],
ablations: [
"DFB contributes +1.8-1.9 P over baseline variants; DFA adds ~+0.9 P while enabling adaptive depth; combined +3.7 P / +3.2 PNorm / +3.3 AUC.",
"Qualitative Fig. 2 (NAT2021 N01001, DarkTrack car_4, UAVDark135 car_l1): DARTer holds small distant targets with similar-object interference where DCPT/AVTrack drift.",
"Template schedule and beta=0.3 fixed; no threshold sweep reported.",
],
limitations: {
authorStated: [
"No explicit limitations section is given in the paper.",
],
evident: [
"80.9M params is not lightweight: 10x AVTrack-DeiT (7.9M) and slower than several daytime UAV trackers (74 vs 122-212 fps).",
"Dynamic template on fixed-interval update with no quality gating risks ingesting occluded templates at night.",
"Nighttime training still depends on BDD100K_Night, SHIFT-night and ExDark mixes plus four day datasets; data-scarcity claim is partial.",
],
},
assumptions: [
"Static+dynamic template pair plus overlapped patches span the viewpoint/illumination variation of the night sequence.",
"Skipped ViT layers are redundant given current fused features; first-block features suffice to seed gating.",
],
computation:
"80.9M params, 74 fps (RTX-3090); train 4x A5000, 150 epochs AdamW batch 32; search/template 256x256/128x128 per text, patch 16x16, beta 0.3.",
relations: [
{ to: "T084", type: "addresses-limitation", note: "Removes the separate transformer enhancer of the Tracker-Meets-Night line with end-to-end dark blending." },
{ to: "T071", type: "builds-on", note: "One-stream ViT joint extraction/matching plus OSTrack-style corner prediction head." },
{ to: "T028", type: "uses-as-baseline", note: "SiamRPN++ beaten on NAT2021-L (64.9 vs 42.9 P) and DarkTrack2021 (67.6 vs 50.9-class rows)." },
],
concepts: ["sot", "attention", "memory-network", "success-plot", "benchmark-design"],
impact: "Validates that multi-view template blending plus adaptive transformer depth can top nighttime UAV leaderboards without enhancers or adaptation modules.",
}
];
