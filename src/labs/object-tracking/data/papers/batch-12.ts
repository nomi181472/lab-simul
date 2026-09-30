import type { PaperRecord } from "../types";

/* batch-12 — T099..T108 */

export const BATCH_12: PaperRecord[] = [
{
id: "T099",
arxiv: "2403.02075",
title: "DiffMOT: A Real-time Diffusion-based Multiple Object Tracker with Non-linear Prediction",
shortTitle: "DiffMOT",
year: 2024,
authors: ["Weiyi Lv", "Yuhang Huang", "Ning Zhang", "Ruei-Sung Lin", "Mei Han", "Dan Zeng"],
fileName: "2403.02075v2.pdf",
task: "multi-object",
tags: ["tracking-by-detection", "diffusion", "motion-prediction", "non-linear-motion", "real-time", "decoupled-diffusion"],
difficulty: "advanced",
summary:
"DiffMOT replaces the Kalman Filter with a Decoupled Diffusion-based Motion Predictor (D2MP) that learns the whole-dataset motion distribution and generates one object's future motion from pure noise conditioned on its own n=5-frame history. A one-branch reverse step gives one-step sampling at 22.7 FPS (YOLOX-X). It reports 62.3 HOTA on DanceTrack and 76.2 HOTA on SportsMOT (train+val detector), plus comparable MOT17/MOT20 numbers.",
problem:
"Tracking-by-detection with constant-velocity Kalman prediction works on linear pedestrian scenes (MOT17/MOT20) but breaks on dancers and athletes with acceleration, deceleration and irregular direction changes, where Fig. 1(c) shows linear-KF predicted-vs-GT IoU collapsing on non-linear datasets.",
background: ["mot", "tracking-by-detection", "motion-model", "kalman", "data-association", "hungarian", "cost-matrix"],
previousWork: [
{
name: "Kalman-filter TBD trackers (SORT, DeepSORT, FairMOT, ByteTrack, OC-SORT)",
limitation:
"Constant-velocity linear assumption plus per-trajectory filtering, so error magnifies on non-linear dance/sports motion.",
whyThisPaper:
"DiffMOT keeps the TBD association but swaps the predictor for a diffusion generator conditioned on 5-frame history, gaining +8.9 HOTA over KF on DanceTrack val (46.8 to 55.7).",
},
{
name: "Neural motion models (optical-flow, LSTM, Transformer predictors, e.g. MotionTrack)",
limitation:
"Rigid structures underfit motion diversity (MLP/LSTM) or cost far too much compute (flow/Transformer) for real-time use.",
whyThisPaper:
"D2MP fits the full motion distribution with one-step decoupled sampling: 55.7 HOTA at 22.7 FPS vs Transformer predictor 54.6 HOTA and two-branch 20-step diffusion 54.6 HOTA at 7.5 FPS.",
},
{
name: "DiffusionTrack (concurrent diffusion MOT)",
limitation:
"Uses DDPM to model relations between paired boxes without modelling non-linear motion itself.",
whyThisPaper:
"DiffMOT models the entire motion distribution and generates future motion from noise during prediction, beating DiffusionTrack 62.3 vs 52.4 HOTA on DanceTrack test.",
},
],
researchGap:
"No MOT motion predictor combined diffusion-grade non-linear fitting with one-step real-time sampling.",
contribution: [
"DiffMOT tracker: YOLOX detection plus D2MP motion prediction plus ByteTrack-style two-stage Hungarian association with ReID distance, IoU cost, dynamic appearance and adaptive weighting.",
"Decoupled Diffusion-based Motion Predictor (D2MP): data-to-zero plus zero-to-noise forward process with constant attenuation c = -Mf,0, and a one-branch reverse needing only c_theta for one-step sampling.",
"Historical Memory Information Network (HMINet): 6x MHSA condition embedding over Cf in R(nx8) fused into noisy motion via motion-fusion layer, then stacked MHSA+MFL plus MLP head, trained with smooth-L1 on c.",
"State of the art on non-linear benchmarks: 62.3 HOTA DanceTrack, 76.2 HOTA SportsMOT (train+val), with real-time 22.7 FPS (X) / 30.3 FPS (S) and cross-dataset generalization without retraining.",
],
method: {
pipeline: ["detect", "build-history-condition", "diffusion-sample-motion", "predict-boxes", "two-stage-associate", "update-tracks"],
architecture:
"YOLOX-X detector; D2MP with HMINet (6 MHSA layers, class token E in R(1x512) to Ece, motion-fusion layer, stacked MHSA+MFL, MLP head predicting c_theta); BYTE-style association.",
motionModel:
"Learned generative: condition Cf = [If-1..If-n], If = (x,y,w,h,dx,dy,dw,dh), n=5; one-step sample motion Mf from N(0,I) conditioned on Cf; Pf = Mf + last location; min diffusion time 0.001.",
appearanceModel:
"ReID feature distance fused with IoU for first-stage cost, plus dynamic appearance and adaptive weighting from Deep OC-SORT/BoT-SORT line [22][1]; no online embedding training described.",
association:
"Two groups by tau_high=0.6 / tau_low=0.4; stage 1: predictions vs high-score boxes via ReID+IoU Hungarian; stage 2: remainders vs low-score boxes via IoU Hungarian; unmatched tracks deleted, high-conf detections birth tracks.",
detectionDependency: "YOLOX-X by default (S/M/L/X compared); MOT17/MOT20 under private-detector protocol with shared YOLOX.",
trackManagement:
"Alg. 1: init empty; per frame split detections, predict each track, two matches, delete unmatched, birth dets above epsilon; no explicit max_age stated.",
loss:
"Smooth-L1 on attenuation c: 0.5(c_theta-c)^2 if |.|<1 else |.| - 0.5; Adam, lr 1e-4, 800 epochs, batch 2048 on 4x RTX 3090.",
optimization:
"Adam, lr 1e-4, 800 epochs, batch size 2048, 4x GeForce RTX 3090; ablations on DanceTrack val with fixed YOLOX-X.",
},
equations: [
{
id: "diffmot-motion",
label: "Frame motion vector",
formula: "Mf = Bf - Bf-1 = (dx_f, dy_f, dw_f, dh_f), Bf = (xf, yf, wf, hf)",
variables: [
{ symbol: "Bf", meaning: "box center-x, center-y, width, height at frame f" },
{ symbol: "Mf", meaning: "clean motion data diffused in D2MP" },
{ symbol: "dx,dy,dw,dh", meaning: "per-coordinate frame-to-frame displacements" },
],
intuition: "Motion is the box delta, so predicting deltas instead of absolutes factors out where in the image the object is.",
why: "Defines the clean data distribution the diffusion process learns over the whole dataset.",
where: "Sec. 3.2.1, Eq. 1; history Cf built from these deltas plus boxes.",
simulator: "motion",
paperIds: ["T099"],
},
{
id: "diffmot-forward",
label: "Decoupled forward diffusion",
formula: "Mf,t = Mf,0 + t*c + sqrt(t)*z, c = -Mf,0, z ~ N(0,I), t in [0,1]",
variables: [
{ symbol: "Mf,0", meaning: "clean motion" },
{ symbol: "Mf,t", meaning: "noisy motion at diffusion time t" },
{ symbol: "c", meaning: "constant attenuation velocity driving data to zero at t=1" },
{ symbol: "z", meaning: "standard normal noise (zero-to-noise branch)" },
],
intuition: "Slide the true motion to zero while independently turning up noise, so at t=1 only pure noise remains.",
why: "Analytic attenuation is what later permits one-step reversal instead of thousand-step DDPM integration.",
where: "Sec. 3.2.1, Eqs. 2-4; Df,t = Mf,0 + t*c and Wf,t = sqrt(t)*z summed.",
params: "Smallest time step floored at 0.001 in training.",
simulator: "motion",
paperIds: ["T099"],
},
{
id: "diffmot-reverse",
label: "One-branch one-step reversal",
formula: "mu_theta = (t-dt)/t * Mf,t - (dt/t) * c_theta(Mf,t, t, Cf); z_theta = (Mf,t - (t-1)*c_theta)/sqrt(t)",
variables: [
{ symbol: "mu_theta", meaning: "predicted mean of q(Mf,t-dt | Mf,t, Mf,0)" },
{ symbol: "c_theta", meaning: "HMINet prediction of attenuation c given noisy motion, time and history" },
{ symbol: "Cf", meaning: "n=5-frame history condition" },
{ symbol: "dt", meaning: "reverse interval; dt=t=1 gives one-step sampling" },
],
intuition: "If you can guess how fast the signal was attenuated, you can back out both the noise and the previous state in closed form.",
why: "Eliminates the second noise branch of DDM: one network, one step, +11.2 HOTA over two-branch one-step and 22.7 vs 7.5 FPS.",
where: "Sec. 3.2.2, Eqs. 8-9; sample Mf,t-dt ~ N(mu_theta, dt(t-dt)/t * I).",
params: "Two-branch needs 20 steps to reach 54.6 HOTA; one-branch hits 55.7 in 1 step.",
simulator: "motion",
paperIds: ["T099"],
},
{
id: "diffmot-condition",
label: "History condition and fusion",
formula: "If = (xf,yf,wf,hf,dxf,dyf,dwf,dhf); Cf = [If-1;...;If-n]; Mbar = Sigmoid(MLP(Ece))*MLP(Mf,t) + MLP(Ece)",
variables: [
{ symbol: "If", meaning: "box plus motion at frame f" },
{ symbol: "Ece", meaning: "class-token condition embedding from 6x MHSA over Cf" },
{ symbol: "Mbar", meaning: "history-guided noisy motion feature fed to fusion stack" },
],
intuition: "Summarize the last 5 frames into one token, then use it as scale-and-shift knobs on the noisy motion.",
why: "Box-only (51.0) or motion-only (50.4) conditions underperform the joint If-1 (51.7 HOTA); n=5 (55.7) beats n=1..3 and n=7/10.",
where: "Sec. 3.2.3, Eqs. 10-11; Ece concatenated with Mf,t into MHSA+MFL then MLP head.",
params: "n=5 optimal; longer history adds interference, shorter under-constrains.",
paperIds: ["T099"],
},
{
id: "diffmot-loss",
label: "Smooth-L1 supervision on c",
formula: "L = 0.5*(c_theta-c)^2 if |c_theta-c|<1 else |c_theta-c|-0.5",
variables: [
{ symbol: "c", meaning: "ground-truth attenuation (= -Mf,0)" },
{ symbol: "c_theta", meaning: "HMINet prediction" },
],
intuition: "Quadratic for small errors, linear for outliers: learn precise small motions without exploding on jumps.",
why: "Single regression target replaces joint c+z supervision of vanilla decoupled diffusion.",
where: "Sec. 3.2.4, Eq. 12.",
paperIds: ["T099"],
},
{
id: "diffmot-assoc",
label: "Two-stage BYTE association",
formula: "D_first = {det: conf > 0.6}; D_second = {det: 0.4 < conf < 0.6}; stage1: Hungarian(ReID + IoU); stage2: Hungarian(IoU)",
variables: [
{ symbol: "D_first/D_second", meaning: "high- / low-score detection groups" },
{ symbol: "ReID + IoU", meaning: "first-stage cost fusing appearance distance and box overlap" },
{ symbol: "Hungarian", meaning: "optimal one-to-one assignment per stage" },
],
intuition: "Match confident boxes with looks plus overlap, then rescue occluded tracks with leftover low-score boxes on overlap alone.",
why: "Lets the better motion predictions convert directly into fewer switches and fragments under occlusion.",
where: "Sec. 3.1 and Alg. 1; same grouping as ByteTrack with paper thresholds 0.6/0.4.",
simulator: "bytetrack",
paperIds: ["T099"],
},
],
datasets: ["dancetrack", "mot17", "mot20", "others"],
metrics: ["hota", "assa", "deta", "idf1", "mota", "fps"],
baselines: ["SORT", "DeepSORT", "FairMOT", "CenterTrack", "TraDes", "QDTrack", "TransTrack", "MOTR", "GTR", "ByteTrack", "OC-SORT", "StrongSORT", "MotionTrack", "SparseTrack", "C-BIoU", "Deep OC-SORT", "DiffusionTrack", "BoT-SORT", "MixSort-OC", "MixSort-Byte"],
results: [
"DanceTrack test (Tab. 1, YOLOX shared): DiffMOT 62.3 HOTA / 63.0 IDF1 / 47.2 AssA / 92.8 MOTA / 82.5 DetA; paper states +1.0 HOTA, +1.5 IDF1, +1.4 AssA over Deep OC-SORT (61.3/61.5/45.8/92.3/82.2); DiffusionTrack rows 52.4/47.5/33.5/89.5/82.2.",
"SportsMOT test (Tab. 2, train-only): DiffMOT 72.1 HOTA / 72.8 IDF1 / 60.5 AssA / 94.5 MOTA / 86.0 DetA; train+val (*): 76.2/76.1/65.1/97.1/89.3 vs MixSort-OC* 74.1/74.4/62.0/96.5/88.5 (paper states +2.1/+1.7/+3.1/+0.6/+0.8).",
"MOT17 test private detector (Tab. 3): DiffMOT 64.5 HOTA / 79.3 IDF1 / 64.6 AssA / 79.8 MOTA / 64.7 DetA, comparable to SOTA despite non-linear design.",
"MOT20 test (Tab. 9): DiffMOT 61.7 HOTA / 74.9 IDF1 / 60.5 AssA / 76.7 MOTA / 63.2 DetA.",
"Speed/accuracy (Fig. 1, Tab. 4): 22.7 FPS with YOLOX-X at 62.3 HOTA; YOLOX-S reaches 30.3 FPS at 53.3 HOTA / 56.6 IDF1 / 88.4 MOTA.",
"Generalization (Tab. 10): DanceTrack-trained model on SportsMOT 75.6/75.1/97.1 vs SportsMOT-trained 76.2/76.1/97.1; on MOT17 61.8/74.4/79.1 vs 64.5/79.3/79.8; on MOT20 61.2/73.8/76.2 vs 61.7/74.9/76.7.",
],
ablations: [
"Motion models val (Tab. 5): IoU-only 44.7 HOTA; KF 46.8; LSTM 51.2; Transformer 54.6/54.6/38.1/89.2/78.6; D2MP 55.7/55.2/39.5/89.3/78.9 best.",
"Branches/steps (Tab. 6): two-branch 1-step 44.5 HOTA at 20.9 FPS, 10-step 52.3 at 13.1, 20-step 54.6 at 7.5; one-branch 1-step 55.7 at 22.7 (+11.2 HOTA over two-branch one-step).",
"Conditions (Tab. 7): Bf-1 51.0/49.1/33.3/88.6/78.5; Mf-1 50.4/46.8/32.2/89.0/79.3; If-1 joint 51.7/48.5/33.8/89.1/79.4 best.",
"History length (Tab. 8): n=1: 51.7; n=2: 52.5; n=3: 53.0; n=5: 55.7 peak; n=7: 52.5; n=10: 51.1 HOTA.",
],
limitations: {
authorStated: [
"No velocity-direction constraint: crossing objects exchange IDs (IDs 1/2 and 4/5 swap in Fig. 10 row 1); constraining generation by direction could help.",
"Long-term lost objects hard to re-associate: an object gone for frames returns as a new ID (Fig. 10 row 2); future work is multi-frame trajectory generation for long-term matching.",
],
evident: [
"Association, ReID and detector (YOLOX-X, 800-epoch motion training, 4x3090) carry the headline numbers; motion gain isolated only on DanceTrack val.",
"Generalization gap remains off-domain (e.g. Dance-trained on MOT17 -2.7 HOTA / -4.9 IDF1-class gaps per Tab. 10 discussion) despite SOTA-on-SportsMOT claim.",
],
},
assumptions: [
"Objects move near enough that a 5-frame box+delta history determines the next delta.",
"YOLOX detections plus ReID/IoU Hungarian association are given; D2MP only replaces prediction.",
],
computation:
"Inference 22.7 FPS (YOLOX-X) / 30.3 FPS (YOLOX-S) on RTX 3090; training Adam lr 1e-4, 800 epochs, batch 2048, 4x RTX 3090; n=5 history; tau_high 0.6 / tau_low 0.4.",
relations: [
{ to: "T061", type: "builds-on", note: "Two-stage high/low Hungarian association and cost design taken from ByteTrack; beats it 62.3 vs 47.3 HOTA DanceTrack." },
{ to: "T092", type: "improves", note: "Concurrent diffusion tracker reframed around whole-distribution non-linear motion; 62.3 vs 52.4 HOTA per Tab. 1." },
{ to: "T005", type: "uses-as-baseline", note: "SORT/KF beaten in Tab. 1 (47.9 HOTA) and replaced as predictor; Fig. 6 shows KF ID-switch cases fixed." },
{ to: "T013", type: "uses-as-baseline", note: "DeepSORT rows 45.6 HOTA DanceTrack / buried in MOT17 table." },
{ to: "T043", type: "uses-as-baseline", note: "FairMOT rows 39.7 HOTA DanceTrack / 49.3 SportsMOT." },
{ to: "T042", type: "uses-as-baseline", note: "CenterTrack rows 41.8 DanceTrack / 62.7 SportsMOT." },
{ to: "T045", type: "uses-as-baseline", note: "QDTrack rows 45.7 DanceTrack / 60.4 SportsMOT." },
{ to: "T053", type: "uses-as-baseline", note: "TransTrack rows 45.5 DanceTrack / 68.9 SportsMOT." },
{ to: "T059", type: "uses-as-baseline", note: "MOTR rows 54.2 DanceTrack; cited as slow end-to-end alternative in Fig. 1(b)." },
{ to: "T067", type: "uses-as-baseline", note: "StrongSORT rows 55.6 DanceTrack / 64.4 SportsMOT." },
{ to: "T073", type: "uses-as-baseline", note: "OC-SORT rows 55.1 DanceTrack / 71.9 SportsMOT train-only." },
{ to: "T075", type: "uses-as-baseline", note: "BoT-SORT rows 68.7 HOTA SportsMOT; dynamic appearance/adaptive weighting lineage via [1][22]." },
{ to: "T082", type: "uses-as-baseline", note: "MotionTrack Transformer predictor rows 52.9 DanceTrack / 61.6 SportsMOT, beaten by D2MP." },
{ to: "T063", type: "uses-as-baseline", note: "Evaluated on DanceTrack (40 train/25 val/35 test) as primary non-linear benchmark." },
{ to: "T006", type: "uses-as-baseline", note: "MOT16/MOT17 lineage cited for pedestrian-linear protocol; MOT17 private-detector table reported." },
],
concepts: ["mot", "motion-model", "diffusion-tracking", "tracking-by-detection", "data-association", "cost-matrix", "hota", "occlusion"],
impact:
"First diffusion probabilistic model for MOT motion prediction per authors, with a one-step decoupled recipe showing diffusion can beat Kalman and Transformer predictors at real-time speed on non-linear data.",
},
{
id: "T100",
arxiv: "2403.05839",
title: "Long-Term Visual Object Tracking with Event Cameras: An Associative Memory Augmented Tracker and A Benchmark Dataset",
shortTitle: "AMTTrack / FELT",
year: 2024,
authors: ["Xiao Wang", "Xufeng Lou", "Shiao Wang", "Ju Huang", "Lan Chen", "Bo Jiang"],
fileName: "2403.05839v3.pdf",
task: "single-object",
tags: ["rgb-event", "multimodal-fusion", "long-term", "one-stream-transformer", "hopfield", "associative-memory", "benchmark"],
difficulty: "advanced",
summary:
"Proposes FELT, the first long-term large-scale frame-event SOT dataset (1,044 videos, 1,949,680 RGB+event pairs, 60 classes, 14 attributes, 730/314 split), retrains 21 baselines on it, and adds AMTTrack: a one-stream ViT where Hopfield retrieval layers cross-fuse multi-level RGB/event tokens plus short/long-term dynamic templates. Reports FELT 54.8 SR / 67.9 PR / 65.7 NPR, FE108 65.6/95.9, COESOT 68.8/82.9, VisEvent 58.8/76.6.",
problem:
"Event trackers were evaluated on short-term sets (FE108 avg 1,932 frames but indoor/limited; VisEvent avg 453; COESOT avg 354) while real surveillance, reconnaissance and wildlife tasks need minutes-long RGB-event tracking under occlusion, disappearance/reappearance, illumination change and appearance drift.",
background: ["sot", "multimodal", "attention", "appearance-features", "memory-network", "success-plot", "benchmark-design"],
previousWork: [
{
name: "Short-term RGB-event trackers (CMT, AFNet, CEUTrack, STNet, SDSTrack, ViPT, DANet, HDETrack)",
limitation:
"Frame-by-frame fusion without long-term memory; collapse under long occlusion, out-of-view and appearance change.",
whyThisPaper:
"AMTTrack adds Hopfield cross-modal retrieval over multi-level tokens plus ST/LT dynamic templates, topping FELT SR/NPR and COESOT SR/PR.",
},
{
name: "One-stream RGB trackers (OSTrack baseline)",
limitation:
"Single modality fails at low illumination/fast motion; static template cannot track drift over ~1867-frame videos.",
whyThisPaper:
"Same OSTrack head/training recipe but dual RGB+event streams with retrieval fusion and template memory: +2.5 SR / +2.0 PR / +2.4 NPR on FELT (52.3 to 54.8).",
},
{
name: "Existing event datasets (FE108, VisEvent, COESOT, EventVOT, CRSOT, VOT-DVS)",
limitation:
"Small, short-term, or single-modal/unaligned; none is a large-scale long-term aligned frame-event SOT train-and-eval benchmark.",
whyThisPaper:
"FELT gives 1,044 videos x1000+ frames, 1.9M pairs, 60 classes, 14 attributes with train/test split plus 21 retrained baselines.",
},
],
researchGap:
"No large-scale long-term aligned frame-event SOT benchmark and no tracker with associative multi-level fusion plus long-term template memory for it.",
contribution: [
"FELT dataset: 1,044 DVS346 (346x260) videos, 1,949,680 RGB+event pairs, 60 object classes, 14 attributes, 730 train / 314 test, largest frame-event set per Tab. 1, with 21-baseline benchmark.",
"Associative-memory Transformer tracker: 12-layer one-stream ViT on 128 templates / 256 search for RGB+event with Hopfield retrieval layers at [5,8,11] doing cross-modal multi-level fusion.",
"Associative template update: ST/LT memory banks with score-gated interval updates, appearance-diversity LT refresh plus Hopfield-refined dynamic templates (Algorithm 1).",
"OSTrack-style FCN head with L1 + GIoU (5.0/2.0) + weighted focal loss; SOTA/competitive on FELT, FE108, COESOT, VisEvent with full ablations over weights, layers, beta and memory sizes.",
],
method: {
pipeline: ["patch-embed-rgb-event", "one-stream-vit", "hopfield-cross-modal-retrieval", "fuse-search", "fcn-head", "st-lt-template-update"],
architecture:
"Unified 12-layer ViT; tokens Zv/Xv/Ze/Xe (P=16) from 128 templates and 256 search; Hopfield layers at L={5,8,11} with stored patterns P5={1,3}, P8={4,6}, P11={7,9}; OSTrack FCN head.",
motionModel: "None — per-frame detection-style localization inside search window; continuity only via dynamic templates.",
appearanceModel:
"Multi-level RGB/event token fusion via cross-modal Hopfield retrieval plus ST/LT dynamic template banks (default 5/10) refined by a Hopfield lookup; static template init.",
association: "Not applicable (single object); dynamic templates resampled from ST/LT and concatenated as z = [zLT_d; zST_d].",
detectionDependency: "None — first-frame box given; event frames are fixed-interval stacks of {x,y,t,p} events.",
trackManagement:
"Alg. 1: init ST/LT from static template; if score > tau every Isu frames update ST, every Ilu update LT with diversity bound theta; RI/UI 10/20 best; beta 4.0 for template Hopfield.",
loss:
"L = 5.0*L1 + 2.0*L_GIoU + L_focal; AdamW lr 4e-4, wd 1e-4, batch 12, 100-frame sampling window, 20k samples/epoch; OSTrack-pretrained init beats MAE on all four sets.",
},
equations: [
{
id: "amttrack-energy",
label: "Modern Hopfield energy",
formula: "E = -(1/beta) * log(sum_i exp(beta * yi^T r)) + 0.5*||r||^2 + C",
variables: [
{ symbol: "yi", meaning: "i-th stored pattern (N total)" },
{ symbol: "r", meaning: "input state pattern" },
{ symbol: "beta", meaning: "temperature sharpening retrieval" },
{ symbol: "C", meaning: "constant term" },
],
intuition: "An energy landscape whose valleys are memories: drop in a partial cue and roll to the nearest stored pattern.",
why: "Justifies attention-as-retrieval: minimizing E gives the update used for fusion and template refinement.",
where: "Sec. 4.3, Eq. 1; update r(t+1) = Y softmax(beta Y r(t)).",
paperIds: ["T100"],
},
{
id: "amttrack-retrieval",
label: "Projected Hopfield retrieval",
formula: "Z = Y Wv * softmax(beta * R Wq Wk^T Y^T)",
variables: [
{ symbol: "Z", meaning: "retrieved result patterns" },
{ symbol: "R", meaning: "state patterns (current-layer features)" },
{ symbol: "Y", meaning: "stored patterns (preceding-layer features)" },
{ symbol: "Wq/Wk/Wv", meaning: "query/key/value projections" },
],
intuition: "Attention with memories as keys: each current token is rebuilt as a weighted mix of earlier-layer tokens.",
why: "High beta retrieves one pattern crisply, low beta superposes several; backbone fusion uses beta 0.25 (superposition wins).",
where: "Sec. 4.3, Eq. 3; embedded at layers 5/8/11.",
params: "Beta sweep 0.1/0.25/1.0/4.0: 0.25 best for backbone; 4.0 best for template Hopfield.",
paperIds: ["T100"],
},
{
id: "amttrack-cross",
label: "Cross-modal Hopfield fusion",
formula: "Zv_i = Hopfield(Rv_i, Ye_i) + Rv_i; Ze_i = Hopfield(Re_i, Yv_i) + Re_i",
variables: [
{ symbol: "Rv_i/Re_i", meaning: "RGB/event search features at layer li" },
{ symbol: "Ye_i/Yv_i", meaning: "event/RGB preceding-layer features as memory bank" },
{ symbol: "Zv_i/Ze_i", meaning: "retrieval-augmented outputs" },
],
intuition: "Let RGB query the event past and events query the RGB past, then add the answer back residually.",
why: "Beats MLP (63.2), Hopfield-pooling (63.2) and non-cross Hopfield (63.5) with 63.9 SR on COESOT; integrates local detail plus global context.",
where: "Sec. 4.3, Eq. 4; [5,8,11]2 setting chosen over [3,7,11] and [5,8,11]1.",
paperIds: ["T100"],
},
{
id: "amttrack-template",
label: "Template Hopfield refinement",
formula: "Z = Wv * softmax(beta * R Wk^T); inputs zs, zd, score; if score > tau and n mod Isu==0 update ST, and if n mod Ilu==0 update LT",
variables: [
{ symbol: "zs/zd", meaning: "static / dynamic templates" },
{ symbol: "score", meaning: "response-map max confidence" },
{ symbol: "Isu/Ilu", meaning: "short/long update intervals (10/20 best)" },
{ symbol: "theta/tau", meaning: "diversity lower bound / response threshold" },
],
intuition: "Keep two photo albums (recent vs diverse old looks), only paste in good frames, then denoise the album with associative lookup.",
why: "Hopfield on (63.9 to 64.8 SR COESOT); ST+LT beats either alone; prevents one bad update polluting long-term memory.",
where: "Sec. 4.4, Eq. 5 and Algorithm 1; LT re-initialized periodically against error accumulation.",
params: "LT/ST 5/10, RI/UI 10/20, resample ST&LT best per Tab. 9.",
simulator: "track-mgmt",
paperIds: ["T100"],
},
{
id: "amttrack-loss",
label: "Tracking loss",
formula: "L = 5.0*L1 + 2.0*L_GIoU + L_focal",
variables: [
{ symbol: "L1", meaning: "box coordinate loss" },
{ symbol: "L_GIoU", meaning: "generalized IoU box loss" },
{ symbol: "L_focal", meaning: "weighted focal classification loss" },
],
intuition: "Penalize wrong boxes linearly/quadratically and wrong pixels with hard-example focus.",
why: "Same recipe as OSTrack head so gains isolate to Hopfield fusion and memory.",
where: "Sec. 4.5, Eq. 6.",
paperIds: ["T100"],
},
],
datasets: ["others"],
metrics: ["success-auc", "precision", "norm-precision", "fps"],
baselines: ["OSTrack", "STARK", "MixFormer", "AiATrack", "SimTrack", "GRM", "ROMTrack", "ViPT", "SeqTrack", "ARTrackv2", "HIPTrack", "ODTrack", "EVPTrack", "AQATrack", "SDSTrack", "UnTrack", "FERMT", "LMTrack", "AsymTrack", "ORTrack", "UNTrack", "SiamRPN", "SiamBAN", "SiamFC++", "KYS", "CLNet", "ATOM", "DiMP", "PrDiMP", "CEUTrack", "MamTrack", "TransT", "KeepTrack", "TrDiMP", "ToMP", "SiamCAR", "SiamR-CNN", "CMDTrack"],
results: [
"FELT (Tab. 4, 314 test videos): Ours 54.8 SR / 67.9 PR / 65.7 NPR at 23 FPS vs OSTrack 52.3/65.9/63.3; best SR and NPR, PR below AQATrack (69.1) and EVPTrack (68.7).",
"FE108 (Tab. 3): Ours 65.6 SR / 95.9 PR vs MamTrack 66.4/94.2 and ViPT 65.8/93.8; best PR, second SR.",
"COESOT (Tab. 5): Ours 68.8 SR / 82.9 PR vs ViPT 68.3/81.0, UnTrack 67.9/80.9, SDSTrack 66.7/79.7; best both.",
"VisEvent (Tab. 6): Ours 58.8 SR / 76.6 PR vs SDSTrack 59.7/76.7, UnTrack 59.7/76.3, ViPT 59.2/75.8; competitive, slightly lower SR attributed to shorter sequences.",
"Pretraining (Tab. 7): OSTrack init beats MAE everywhere, e.g. COESOT 68.8/82.9/81.8 vs 64.8/79.4/78.2; FELT 54.8/67.9/65.7 vs 53.5/66.9/64.6.",
"Attributes (Fig. 5): strong on VT, DEF, SV, OE; BI/OV/VT are the top-3 dataset challenges per Fig. 2(a).",
],
ablations: [
"Components (Tab. 8, MAE weights): base 63.2/77.1/76.1 (COESOT) and 60.4/91.6/65.0 (FE108); +ATU 64.1/78.7/77.5 and 61.0/92.3/65.3; +AMAH 63.9/78.4/77.1 and 62.0/93.2/67.6; both 64.8/79.4/78.2 and 63.0/94.4/68.1.",
"Template update (Tab. 9): Hopfield off/on 63.9 to 64.8 SR; beta 0.25/1.0/4.0/16.0 best 4.0; LT/ST 5/10 best; RI/UI 10/20 best; resample ST&LT best.",
"Hopfield fusion (Tab. 10): cross-modal Hopfield 63.9/78.4/77.1 beats MLP 63.2, pooling 63.2, plain Hopfield 63.5; beta 0.25 best; layers [5,8,11]2 beats [3,7,11] (63.4) and [5,8,11]1 (63.6).",
],
limitations: {
authorStated: [
"PR on FELT trails temporal-model trackers (AQATrack, EVPTrack); future work will use frame-to-frame temporal cues and dense event temporal resolution.",
],
evident: [
"Headline numbers use OSTrack pretraining (Tab. 7 shows MAE 1-4 points lower), so fusion/memory gains are entangled with strong RGB initialization.",
"Long-video claims rest on fixed-interval event stacking and score-gated updates; static or motionless targets (NMO) and total occlusion still depend on threshold tau and re-init heuristics.",
],
},
assumptions: [
"RGB and event streams from DVS346 are hardware-aligned spatially and temporally.",
"Response-map maximum is a reliable gate for writing ST/LT memory.",
],
computation:
"23 FPS reported in FELT table; training AdamW lr 4e-4 wd 1e-4 batch 12, 20k samples/epoch, 100-frame window; template 128, search 256; ST 5 / LT 10, RI 10 / UI 20.",
relations: [
{ to: "T071", type: "builds-on", note: "One-stream backbone, FCN head config and training recipe from OSTrack; OSTrack-pretrained init used throughout." },
{ to: "T083", type: "uses-as-baseline", note: "ViPT prompt-fusion beaten on FELT (52.8/65.3/63.1), COESOT (68.3/81.0) and VisEvent (59.2/75.8)." },
{ to: "T070", type: "uses-as-baseline", note: "MixFormer beaten on FELT (53.0/67.5/63.8)." },
{ to: "T055", type: "uses-as-baseline", note: "TransT beaten on COESOT (60.5/72.4) and VisEvent (47.4/65.0)." },
{ to: "T056", type: "uses-as-baseline", note: "STARK spatio-temporal baseline beaten on FELT (52.7/67.9/62.8) and COESOT (56.0/67.7)." },
{ to: "T037", type: "uses-as-baseline", note: "SiamR-CNN re-detection beaten on VisEvent (49.9/65.9)." },
{ to: "T009", type: "uses-as-baseline", note: "SiamFC lineage (PUL/UDT training discussions) as classical baseline class; FE108 Siamese rows 21.8-23.8 SR." },
],
concepts: ["sot", "multimodal", "attention", "memory-network", "appearance-features", "benchmark-design", "success-plot"],
impact:
"First long-term large-scale aligned RGB-event SOT benchmark (largest per Tab. 1) with 21 retrained baselines, plus a Hopfield retrieval recipe for multi-level cross-modal fusion and template memory reused by later RGB-event trackers.",
},
{
id: "T101",
arxiv: "2403.10826",
title: "MambaMOT: State-Space Model as Motion Predictor for Multi-Object Tracking",
shortTitle: "MambaMOT",
year: 2024,
authors: ["Hsiang-Wei Huang", "Cheng-Yen Yang", "Wenhao Chai", "Zhongyu Jiang", "Jeng-Neng Hwang"],
fileName: "2403.10826v2.pdf",
task: "multi-object",
tags: ["tracking-by-detection", "motion-prediction", "state-space", "mamba", "non-linear-motion", "real-time", "tracklet-merging"],
difficulty: "advanced",
summary:
"Drops Mamba selective state-space blocks in place of the Kalman Filter inside a ByteTrack (BYTE + Hungarian + IoU) pipeline: past boxes project up, propagate hidden states, and an MLP head predicts the next box (GIoU+MSE); MambaMOT+ adds a trajectory-embedding head with cosine loss plus hierarchical clustering to merge fragments. Reports DanceTrack 55.5/56.1 HOTA and SportsMOT 70.4/71.3 HOTA at 28.8 FPS.",
problem:
"Kalman prediction assumes linear constant-velocity motion, so on DanceTrack/SportsMOT it is noise-sensitive with magnifying temporal error: a few untracked frames permanently shift position estimates and Ids fragment under diverse irregular motion and occlusion.",
background: ["mot", "tracking-by-detection", "motion-model", "kalman", "state-space", "data-association", "hungarian"],
previousWork: [
{
name: "Kalman TBD trackers (SORT, DeepSORT, ByteTrack, OC-SORT, BoT-SORT, StrongSORT, FairMOT)",
limitation:
"Linear motion + IoU association collapses when appearances match and motions diverge (ByteTrack 47.3 HOTA DanceTrack despite same detector).",
whyThisPaper:
"Same BYTE association and YOLOX detections but Mamba-predicted boxes: +8.2 HOTA over ByteTrack on both DanceTrack and SportsMOT.",
},
{
name: "Learned/Transformer MOT (CenterTrack, TraDes, QDTrack, TransTrack, MOTR, GTR, MotionTrack)",
limitation:
"End-to-end query models associate well but detect poorly and run under ~10 FPS (MOTR 54.2 HOTA DanceTrack).",
whyThisPaper:
"Lightweight 2-block Mamba predictor keeps detector fixed and runs 28.8 FPS while reaching 55.5-56.1 HOTA DanceTrack.",
},
{
name: "Siamese tracklet mergers and MambaTrack",
limitation:
"Standalone Siamese merging costs O(N^2) pair evaluations; MambaTrack uses bidirectional SSM for motion but not MOT tracklet merging.",
whyThisPaper:
"MambaMOT+ reuses the prediction backbone's hidden features as trajectory embeddings (O(N) forward) for clustering-based merge, +0.6 HOTA / +1.0 IDF1 on DanceTrack.",
},
],
researchGap:
"No online real-time Kalman replacement that adapts per-dataset motion patterns and yields reusable trajectory features without a second network.",
contribution: [
"MambaMOT: linear-projection plus stacked Mamba blocks as drop-in motion predictor inside BYTE association, trained on MOT17+DanceTrack+SportsMOT tracklets of length 2..n with padding.",
"Selective SSM discretization (ZOH with input-dependent timescale) giving linear-time recurrence ht = Abar_t ht-1 + Bbar_t xt, yt = C ht for box sequences.",
"MambaMOT+: trajectory embedding head MLPemb on yt with cosine embedding loss, joint Ltotal = Lpred + Lcos, plus hierarchical clustering merge under 50-frame / 50-px gates.",
"Advanced non-linear results at speed: 55.5/56.1 HOTA DanceTrack, 70.4/71.3 SportsMOT*, 28.8 FPS single GPU.",
],
method: {
pipeline: ["collect-tracklet-boxes", "linear-project", "mamba-propagate", "predict-next-box", "byte-associate", "merge-tracklets-plus"],
architecture:
"2 Mamba blocks, hidden dim 64, expansion factor 2; linear projection of [x,y,w,h] up; MLPpred box head; (+ version) MLPemb trajectory head; BYTE association unchanged.",
motionModel:
"Autoregressive box-level: input {Bt-n..Bt-1} in R(nx[x,y,w,h]) predicts BT; hidden state carries motion pattern; training tracklets sampled length (2,n) with padding, next GT box as target.",
appearanceModel: "None — pure motion + IoU association; same YOLOX detections as ByteTrack for fair comparison.",
association:
"BYTE two-stage + Hungarian on IoU between Mamba predictions and detections; MambaMOT+ adds post-track cosine-similarity hierarchical clustering merge.",
detectionDependency: "DanceTrack: ByteTrack YOLOX detections reused; SportsMOT: YOLOX trained on SportsMOT train per ByteTrack recipe.",
trackManagement:
"Online tracking then offline merge (+): temporal gate 50 frames and spatial gate 50 px block unreasonable pairs before clustering.",
loss:
"Lpred = Lgiou + Lmse on boxes; Lcos(i,j) = 1-cos(fi,fj) if same else max(0,cos); Ltotal = Lpred + Lcos; Adam lr 1e-4, 500 epochs, batch 32.",
optimization:
"Adam, lr 0.0001, 500 epochs, batch 32, single NVIDIA RTX 4080; 2 Mamba blocks, dim 64, expansion 2.",
},
equations: [
{
id: "mambamot-ssm",
label: "Continuous state-space motion",
formula: "y(t) = C h(t); h(t) = A h(t-1) + B x(t)",
variables: [
{ symbol: "x(t)", meaning: "projected box input at time t" },
{ symbol: "h(t)", meaning: "hidden motion-pattern state" },
{ symbol: "A/B/C", meaning: "continuous system matrices" },
{ symbol: "y(t)", meaning: "discrete SSM output feeding heads" },
],
intuition: "A running memory vector updated by each past box, read out as the motion summary for the next frame.",
why: "Replaces fixed-velocity Kalman state with a learned long-context memory over the tracklet.",
where: "Sec. III-B, Eqs. 1-2.",
simulator: "motion",
paperIds: ["T101"],
},
{
id: "mambamot-selective",
label: "Selective (Mamba) discretization",
formula: "Abar_t = exp(dA) = 1 - sigmoid(Linear(xt)); Bbar_t = (Abar)^-1 (exp(dA)-I) dB = sigmoid(Linear(xt)); yt = C ht; ht = Abar_t ht-1 + Bbar_t xt",
variables: [
{ symbol: "d", meaning: "input-dependent timescale (ZOH step)" },
{ symbol: "Abar_t/Bbar_t", meaning: "discrete selective matrices at step t" },
{ symbol: "xt/yt/ht", meaning: "input / output / hidden state at t" },
],
intuition: "Let each box decide how much history to keep vs overwrite: steady motion keeps memory, abrupt moves reset it.",
why: "Input-dependence plus no-attention recurrence gives linear scaling and 28.8 FPS vs sub-10-FPS Transformers.",
where: "Sec. III-B, Eqs. 3-5.",
params: "Hidden 64, expansion 2, L blocks; longer n helps to dataset-dependent point (padding used).",
simulator: "motion",
paperIds: ["T101"],
},
{
id: "mambamot-pred",
label: "Prediction head and loss",
formula: "Yt = MLPpred(yt); Lpred = Lgiou + Lmse",
variables: [
{ symbol: "Yt", meaning: "predicted next box [x,y,w,h]" },
{ symbol: "Lgiou/Lmse", meaning: "generalized-IoU and mean-squared box losses vs GT" },
],
intuition: "Translate the motion summary into coordinates, then punish both overlap shape and raw distance.",
why: "Direct box supervision is what makes Mamba predictions beat Kalman IoU-vs-GT in Fig. 1.",
where: "Sec. III-C, Eq. 6; associated via BYTE Hungarian on IoU.",
simulator: "motion",
paperIds: ["T101"],
},
{
id: "mambamot-embed",
label: "Trajectory embedding and merge",
formula: "ft = MLPemb(yt); Lcos = 1-cos(fi,fj) if i==j else max(0,cos(fi,fj)); Ltotal = Lpred + Lcos",
variables: [
{ symbol: "ft", meaning: "trajectory feature for a tracklet" },
{ symbol: "fi/fj", meaning: "features of sampled tracklet pair i/j" },
{ symbol: "Lcos", meaning: "pull same identity together, push different apart" },
],
intuition: "Boxes that moved alike should sit near each other in feature space, so fragments of one dancer rejoin after occlusion.",
why: "O(N) per-tracklet forward replaces O(N^2) Siamese pair network; +0.6 HOTA DanceTrack, +0.9 SportsMOT.",
where: "Sec. III-D, Eqs. 7-8; cosine similarity + hierarchical clustering with 50/50 gates.",
params: "Gates 50 frames / 50 px prevent absurd merges.",
paperIds: ["T101"],
},
],
datasets: ["dancetrack", "others"],
metrics: ["hota", "assa", "deta", "idf1", "mota", "fps"],
baselines: ["SORT", "DeepSORT", "ByteTrack", "OC-SORT", "BoT-SORT", "StrongSORT", "FairMOT", "CenterTrack", "TraDes", "QDTrack", "TransTrack", "MOTR", "GTR", "MotionTrack"],
results: [
"DanceTrack test (Tab. I, shared YOLOX): MambaMOT 55.5 HOTA / 80.8 DetA / 38.3 AssA / 53.9 IDF1 / 90.1 MOTA; MambaMOT+ 56.1/80.8/39.0/54.9/90.3; paper states +8.2 HOTA over ByteTrack (47.3/71.6/31.4/52.5/89.5).",
"SportsMOT test (Tab. II, YOLOX *, train-only): MambaMOT* 70.4/86.7/57.2/69.5/94.7; MambaMOT+* 71.3/86.7/58.6/71.1/94.9 vs ByteTrack* 62.0/77.0/49.9/68.7/93.9 (paper states +8.2 HOTA, +7.3 AssA); OC-SORT* 70.2/85.9/57.4/70.4/93.9; TransTrack 68.9/82.7/57.5/71.5/92.6.",
"Merge gain: MambaMOT+ over MambaMOT +0.6 HOTA / +1.0 IDF1 DanceTrack (paper-stated); +0.9 HOTA / +1.6 IDF1 SportsMOT* per Tab. II.",
"Speed (Sec. IV-C): 28.8 FPS single GPU vs MOTR-class methods typically under 10 FPS.",
],
ablations: [
"Backbone reuse (Tab. I/II): same BYTE+YOLOX with Kalman (ByteTrack) vs Mamba gives +8.2 HOTA on both sets, isolating the predictor.",
"Embedding head: joint Lpred+Lcos training plus clustering adds +0.6/0.9 HOTA and +1.0/1.6 IDF1 with zero extra pair network.",
"No predictor/association ablations beyond that are printed; training mixes MOT17+DanceTrack+SportsMOT tracklets length (2,n) with padding.",
],
limitations: {
authorStated: [
"Conclusion claims only comparable-to-SOTA and complex-pattern gains; no failure/limitation section is given in this short paper.",
],
evident: [
"Pure motion + IoU with no appearance: uniform-appearance DanceTrack suits it, but long occlusion with similar motion has no ReID fallback.",
"Merge depends on fixed 50-frame/50-px gates and post-hoc clustering; sensitivity and MOT17/MOT20-linear performance are not reported.",
],
},
assumptions: [
"Past box coordinates alone (no image, no appearance) determine the next box.",
"Detector outputs (ByteTrack YOLOX) are fixed and reliable; association is IoU-Hungarian.",
],
computation:
"28.8 FPS single GPU; Adam lr 1e-4, 500 epochs, batch 32, RTX 4080; 2 Mamba blocks, hidden 64, expansion 2; merge gates 50 frames / 50 px.",
relations: [
{ to: "T061", type: "builds-on", note: "Reuses BYTE association, Hungarian-IoU matching and YOLOX detections; Mamba swaps only the Kalman predictor." },
{ to: "T005", type: "addresses-limitation", note: "Replaces SORT constant-velocity Kalman whose noise sensitivity and error magnification motivate Sec. II-A." },
{ to: "T013", type: "uses-as-baseline", note: "DeepSORT rows 45.6 DanceTrack." },
{ to: "T043", type: "uses-as-baseline", note: "FairMOT rows 39.7 DanceTrack / 49.3 SportsMOT." },
{ to: "T042", type: "uses-as-baseline", note: "CenterTrack rows 41.8 DanceTrack / 62.7 SportsMOT." },
{ to: "T045", type: "uses-as-baseline", note: "QDTrack rows 45.7 DanceTrack / 60.4 SportsMOT." },
{ to: "T053", type: "uses-as-baseline", note: "TransTrack rows 45.5 DanceTrack / 68.9 SportsMOT." },
{ to: "T059", type: "uses-as-baseline", note: "MOTR rows 54.2 DanceTrack; cited as slow end-to-end alternative." },
{ to: "T067", type: "uses-as-baseline", note: "StrongSORT in Kalman-based comparison set." },
{ to: "T073", type: "uses-as-baseline", note: "OC-SORT rows 54.6 DanceTrack / 70.2 SportsMOT*." },
{ to: "T075", type: "uses-as-baseline", note: "BoT-SORT in Kalman-based comparison set." },
{ to: "T082", type: "uses-as-baseline", note: "MotionTrack learned predictor rows 52.9 DanceTrack." },
{ to: "T063", type: "uses-as-baseline", note: "DanceTrack is the primary diverse-motion benchmark; SportsMOT the second." },
{ to: "T006", type: "uses-as-baseline", note: "MOT16/MOT17 lineage cited for linear-pedestrian regime and CLEAR/HOTA protocol." },
],
concepts: ["mot", "motion-model", "state-space", "tracking-by-detection", "data-association", "hungarian", "hota", "occlusion"],
impact:
"Shows a tiny (2-block, dim-64) selective SSM is a viable real-time Kalman drop-in for non-linear MOT and that its hidden states double as free trajectory embeddings for fragment merging.",
},
{
id: "T102",
arxiv: "2404.05518",
title: "DepthMOT: Depth Cues Lead to a Strong Multi-Object Tracker",
shortTitle: "DepthMOT",
year: 2024,
authors: ["Jiapeng Wu", "Yichen Liu"],
fileName: "2404.05518v1.pdf",
task: "multi-object",
tags: ["tracking-by-detection", "depth-cues", "self-supervised-depth", "pose-compensation", "uav", "cascade-matching", "joint-detection-depth"],
difficulty: "intermediate",
summary:
"Builds on FairMOT (DLA-34) with a U-Net-style depth decoder over 5 multi-scale features plus a ResNet-18 pose branch, trained self-supervised (reprojection + smoothness, uncertainty-weighted with detection loss), then at inference splits association into depth-interval cascades and warps Kalman predictions by the 6-DoF pose. Reports VisDrone 42.448 HOTA (best) and UAVDT 66.44 HOTA (best) with ablated +5.6 HOTA / +11.1 IDF1 over FairMOT.",
problem:
"In crowded UAV scenes overlapping 2D boxes share near-identical motion features so IDs confuse, and irregular drone rotation/acceleration breaks linear Kalman prediction; image-registration fixes (BoT-SORT) are slow and texture-sensitive.",
background: ["mot", "tracking-by-detection", "detection", "motion-model", "kalman", "data-association", "iou"],
previousWork: [
{
name: "2D TBD/JDT trackers (SORT, DeepSORT, ByteTrack, BoT-SORT, FairMOT, TrackFormer, UAVMOT)",
limitation:
"Represent objects as 2D boxes only, losing stereoscopic order; motion correction needs slow affine registration.",
whyThisPaper:
"Adds estimated depth per object for cascade matching plus pose-warped Kalman, topping VisDrone HOTA/IDs/MT and UAVDT HOTA/FP.",
},
{
name: "QuoVadis (depth+segmentation to BEV) and SparseTrack (pseudo-depth from box-bottom height)",
limitation:
"QuoVadis is heavy point-cloud BEV machinery; SparseTrack parallax heuristic (2000 - y1) misjudges depth without scene cues.",
whyThisPaper:
"Self-supervised monocular depth from the same DLA-34 encoder replaces the heuristic and feeds Eq. 13 object depths for reliable cascades.",
},
{
name: "Self-supervised monocular depth (Monodepth2, HRDepth, SC-Depth v3)",
limitation:
"Needs video with camera motion and pose estimation, previously disconnected from MOT association.",
whyThisPaper:
"Bridges the gap: one network learns detection plus 5-scale depth with a pose branch, and both outputs are consumed by association and motion compensation.",
},
],
researchGap:
"No end-to-end MOT that both perceives per-object depth for occlusion cascades and reuses the required pose estimate to fix linear motion under drone camera shake.",
contribution: [
"DepthMOT: FairMOT DLA-34 encoder feeding detection heads plus 5-scale depth decoder (Eq. 2) and ResNet-18 6-DoF pose branch, fused end-to-end.",
"Object depth from bottom-edge average (Eq. 13) with depth-interval cascade IoU matching in the style of SparseTrack but with estimated depths.",
"Pose-warped Kalman compensation: bottom corners reprojected by T(t-1 to t) = {R, tau} with real depth (Eq. 14), box height kept.",
"Best HOTA on VisDrone-MOT (42.448) and UAVDT (66.44) with ablations isolating depth vs motion-compensation gains.",
],
method: {
pipeline: ["encode-frame", "detect", "estimate-depthmap", "estimate-pose", "kalman-predict", "pose-warp", "depth-cascade-associate"],
architecture:
"DLA-34 encoder (ft in R(cxhxw), h=H/4); 3 detection heads (heatmap/size/offset); depth decoder on 5 tree scales fi in R(2^i c0 x H/2^i) predicting disparity Ds; ResNet-18 pose net on concatenated adjacent frames to 6-DoF.",
motionModel:
"Kalman Filter base, corrected by pose warp of bottom-left/right corners via Eq. 14; height unchanged; grounded on overhead-view bottom-edge-on-ground prior.",
appearanceModel: "None — IoU-only cascade matching within depth intervals; no ReID embedding.",
association:
"Depth-interval cascade: split detections and tracks into evenly spaced depth bins, IoU-match per bin, carry leftovers forward; separates high-overlap pairs by depth order.",
detectionDependency: "Joint detector (FairMOT-style center/size/offset); TBD baselines use YOLOX-m trained 35 (VisDrone) / 50 (UAVDT) epochs for fairness.",
trackManagement: "Cascade leftovers propagate across depth rounds; standard birth/death otherwise (not parameterized in text).",
loss:
"Ldet = Lheat (focal, Eq. 5) + Lbox (L1 centers + 0.1 shape, Eq. 6); Ldepth = sum Lip (min-reprojection, Eqs. 8-9) + 0.001 Lis (edge-aware smooth, Eq. 10) over S=5; L = 0.5(e^-w1 Ldet + e^-w2 50 Ldepth + w1 + w2).",
optimization:
"Adam, lr 1e-4, batch 4, 1088x608, single Tesla A100; 10 epochs VisDrone, 20 epochs UAVDT; YOLOX-m baselines retrained per best config.",
},
equations: [
{
id: "depthmot-reproject",
label: "Self-supervised view synthesis",
formula: "p' = K R K^-1 p + K^-1 tau, p = [x, y, d]^T",
variables: [
{ symbol: "p", meaning: "pixel plus predicted depth in frame t" },
{ symbol: "p'", meaning: "reprojected position in frame t-dt" },
{ symbol: "R/tau", meaning: "rotation and translation from 6-DoF pose" },
{ symbol: "K", meaning: "camera intrinsic matrix" },
],
intuition: "If depth and camera motion are right, warping frame t should repaint frame t-dt: pixel mismatch is the supervision.",
why: "No depth labels exist on MOT sets, so reprojection error (SSIM+L1) is the only depth teacher.",
where: "Sec. 3.1, Eq. 1; supervisory signal in Fig. 2.",
paperIds: ["T102"],
},
{
id: "depthmot-disparity",
label: "Multi-scale disparity decoder",
formula: "Ds = Sigmoid(Conv(fs, Upsample(fs+1))); dtilde = 1 / (1/dmax + (1/dmin - 1/dmax) * d)",
variables: [
{ symbol: "Ds/d", meaning: "predicted disparity (inverse-depth-like) at scale s" },
{ symbol: "dtilde", meaning: "real depth after min/max mapping" },
{ symbol: "dmin/dmax", meaning: "depth range constraints" },
],
intuition: "Predict bounded inverse depth at 5 resolutions, then stretch to meters with a fixed near/far range.",
why: "Multi-scale DLA-34 features out-represent ResNet-18 depth encoders per authors; upsampled maps give Ldepth at every scale.",
where: "Sec. 3.2, Eqs. 2-3.",
paperIds: ["T102"],
},
{
id: "depthmot-photometric",
label: "Photometric + smoothness depth loss",
formula: "pe = (a/2)(1-SSIM(It', It'_tilde)) + (1-a)||It' - It'_tilde||; Lp = min_{t'} pe; Ls = |dx Ds|e^-|dx It| + |dy Ds|e^-|dy It|; Ldepth = sum(Lp + 0.001 Ls)",
variables: [
{ symbol: "pe", meaning: "per-pixel appearance error" },
{ symbol: "Lp", meaning: "occlusion-robust min over t+-dt reprojections" },
{ symbol: "Ls", meaning: "edge-aware smoothness" },
{ symbol: "a", meaning: "SSIM/L1 balance (Monodepth2-style)" },
],
intuition: "Match looks, take the better of the two neighbor warps per pixel to forgive occlusions, and keep depth smooth except at image edges.",
why: "Handles static-camera and occluded pixels that naive warping would punish.",
where: "Sec. 3.3, Eqs. 8-11; upsampled Ds compared at full resolution.",
paperIds: ["T102"],
},
{
id: "depthmot-object-depth",
label: "Object depth from bottom edge",
formula: "di = (1/(x1-x0)) * sum_{x=x0}^{x1} D[x, y1], oi = [x0,y0,x1,y1]",
variables: [
{ symbol: "di", meaning: "object depth" },
{ symbol: "D", meaning: "full-scale depth map (max scale only)" },
{ symbol: "y1", meaning: "box bottom edge (assumed on ground)" },
],
intuition: "Read depth where the feet touch the ground, averaged across the box width for stability.",
why: "Overhead UAV geometry makes the bottom edge the ground contact; visualized depths stay ordinally stable over time (Fig. 6).",
where: "Sec. 3.4, Eq. 13; bins feed cascade matching.",
simulator: "iou-track",
paperIds: ["T102"],
},
{
id: "depthmot-compensate",
label: "Pose-warped Kalman correction",
formula: "[xi, yj, d~]^T_new = K R K^-1 [xi, yj, d~]^T + K^-1 tau, i=0,1; j=1 (bottom corners only)",
variables: [
{ symbol: "xi/yj", meaning: "bottom-left/right corner coordinates from Kalman box" },
{ symbol: "d~", meaning: "real object depth" },
{ symbol: "R/tau", meaning: "pose change t-1 to t" },
],
intuition: "Ask where the Kalman box bottom would appear after the drone itself moved, and shift the prediction there.",
why: "Corrects irregular-motion misalignment without slow RANSAC/affine registration; ablation adds +1.7 HOTA / +2.4 IDF1 over RANSAC variant.",
where: "Sec. 3.4, Eq. 14; height preserved.",
params: "Only bottom corners warped; top follows from fixed height.",
simulator: "kalman",
paperIds: ["T102"],
},
{
id: "depthmot-uncertainty",
label: "Uncertainty-weighted multi-task loss",
formula: "L = 0.5 * (e^-w1 Ldet + e^-w2 * 50 * Ldepth + w1 + w2)",
variables: [
{ symbol: "Ldet", meaning: "heatmap + box detection loss" },
{ symbol: "Ldepth", meaning: "5-scale depth loss" },
{ symbol: "w1/w2", meaning: "learned log-variances balancing tasks" },
],
intuition: "Let the network learn how noisy each task is and down-weight the noisier one automatically (gamma=50 aligns scales).",
why: "Detection and self-supervised depth live on wildly different loss scales; fixed weights would drown one task.",
where: "Sec. 3.3, Eq. 12.",
paperIds: ["T102"],
},
],
datasets: ["uavdt", "others"],
metrics: ["hota", "mota", "idf1", "idsw", "fp", "fn", "mt-ml"],
baselines: ["SORT", "DeepSORT", "ByteTrack", "ByteTrack+ReID", "BoT-SORT", "UAVMOT", "FairMOT", "TrackFormer", "BIoU_Tracker", "MOTDT"],
results: [
"VisDrone-MOT (Tab. 1): DepthMOT 42.448 HOTA / 37.041 MOTA / 54.023 IDF1 / FN 104054 / FP 41001 / IDs 1248 / MT 626 / ML 467; best HOTA, IDs, MT; MOTA/IDF1 below BoT-SORT (41.652/56.843).",
"UAVDT (Tab. 2): DepthMOT 66.44 HOTA / 62.279 MOTA / 78.13 IDF1 / FN 28951 / FP 3036 / IDs 82 / MT 134 / ML 40; best HOTA and FP; MOTA/IDF1 trail BIoU_Tracker (70.323/79.622), attributed by authors to detector gap.",
"Ablation VisDrone-test-dev (Tab. 3): FairMOT baseline 36.112/33.688/42.573/5438; +ByteTrack low-conf 40.089/33.422/49.843/2216; +SparseTrack pseudo-depth 40.102/33.249/49.124/2721; +our depth 40.088/33.944/49.639/2689; +RANSAC 40.013/30.295/51.216/2142; +our motion comp 41.704/32.622/53.635/1381 (paper: +5.592 HOTA / +11.062 IDF1 over baseline).",
"Depth-room check: predicted 1000x-disparity maps separate adjacent objects with stable ordering over t, t+5, t+10 (Fig. 6).",
],
ablations: [
"Depth source: estimated depth (40.088 HOTA / 2689 IDs) essentially ties SparseTrack heuristic (40.102 / 2721) on HOTA but cuts IDs and lifts MOTA/IDF1.",
"Motion fix: pose warp (41.704/53.635/1381) beats BoT-SORT RANSAC variant (40.013/51.216/2142) by +1.7 HOTA / +2.4 IDF1 / -761 IDs.",
"Low-conf handling alone (ByteTrack trick) gives the single biggest jump: 36.112 to 40.089 HOTA and 5438 to 2216 IDs.",
],
limitations: {
authorStated: [
"Extra depth branch costs compute.",
"Accurate self-supervised depth remains hard.",
"Bottom-edge averaging is only one heuristic; better object-from-map representations are open.",
],
evident: [
"UAVDT MOTA/FN lag (62.279 vs 70.323; FN 28951 vs 17405) suggests the FairMOT joint detector, not association, bounds recall.",
"Overhead ground-contact prior and known-intrinsics warping limit transfer to non-UAV / uncalibrated scenes; only camera-motion videos used by design.",
],
},
assumptions: [
"Camera looks down enough that box bottoms lie on the ground plane.",
"Adjacent UAV frames share enough rigid-scene overlap for pose/depth self-supervision; intrinsics K known.",
],
computation:
"Train 10 epochs VisDrone / 20 UAVDT, Adam lr 1e-4, batch 4, 1088x608, 1x Tesla A100; TBD baselines YOLOX-m 35/50 epochs; S=5 scales, lambda 0.001, gamma 50.",
relations: [
{ to: "T043", type: "builds-on", note: "FairMOT baseline: DLA-34 encoder, heatmap/size/offset heads and Ldet kept; 36.112-HOTA ablation starting point." },
{ to: "T061", type: "builds-on", note: "ByteTrack low-confidence handling adopted; +3.977 HOTA row in Tab. 3." },
{ to: "T075", type: "uses-as-baseline", note: "BoT-SORT beaten on VisDrone HOTA (42.448 vs 42.42) and its RANSAC compensation replaced by pose warp." },
{ to: "T005", type: "uses-as-baseline", note: "SORT rows 35.08 VisDrone / 60.4 UAVDT." },
{ to: "T013", type: "uses-as-baseline", note: "DeepSORT rows 36.921 VisDrone / 61.97 UAVDT." },
{ to: "T073", type: "uses-as-baseline", note: "OC-SORT lineage cited for irregular-motion problem framing." },
{ to: "T059", type: "uses-as-baseline", note: "MOTR/Transformer query line cited as explicit-association-free alternative." },
{ to: "T054", type: "uses-as-baseline", note: "TrackFormer rows 35.344 VisDrone / 43.165 UAVDT." },
],
concepts: ["mot", "3d-tracking", "tracking-by-detection", "motion-model", "kalman", "data-association", "iou", "occlusion", "benchmark-design"],
impact:
"First bridge the authors claim between MOT and self-supervised monocular depth: shows estimated depth plus free pose can replace pseudo-depth heuristics and affine registration in crowded UAV tracking.",
},
{
id: "T103",
arxiv: "2405.10439",
title: "Beyond Traditional Single Object Tracking: A Survey",
shortTitle: "Beyond Traditional SOT Survey",
year: 2024,
authors: ["Omar Abdelaziz", "Mohamed Shehata", "Mohamed Mohamed"],
fileName: "2405.10439v1.pdf",
task: "survey",
tags: ["survey", "single-object", "sequence-models", "generative-models", "self-supervised", "meta-learning", "continual-learning", "domain-adaptation"],
difficulty: "intro",
summary:
"Taxonomy survey of non-traditional SOT: sequence models (autoregressive SeqTrack/ARTrack, memory RFL/GCT/R-RPN/AD-LSTM/BA+TID+DDM), generative (MAE MAT/DropMAE, GAN TGGAN/VITAL/AFSL/GARAT, diffusion Diff-SiamRPN, VAE SINT++/VAE-MCMC), self-supervised, unsupervised, meta, continual and domain-adaptation methods, with compiled GOT-10k/LaSOT/TrackingNet/OTB/VOT tables, metric definitions, trend analysis (transformers overtaking CNNs 2020-2024) and future paths.",
problem:
"Prior SOT reviews catalogued correlation-filter/Siamese or Transformer families and their scores, but gave no principled map of the statistical paradigms (autoregression, generation, self-supervision, meta/continual/adaptation) now driving performance, leaving readers unable to choose or extend non-traditional techniques.",
background: ["sot", "siamese", "attention", "correlation-filter", "success-plot", "benchmark-design"],
previousWork: [
{
name: "Classical pipelines (Yilmaz feature/geometric/contour; Smeulders ALOV++ 19 trackers; Zhang deep+CF; Javed DCF+Siamese 90 trackers)",
limitation:
"Organized by engineering family or experiment snapshot, without covering generative/sequence/meta paradigms and their theories.",
whyThisPaper:
"Re-categorizes by learning paradigm with equations, diagrams and pros/cons guide instead of another family leaderboard.",
},
{
name: "Transformer surveys (Kugarajeevan ViT-in-tracking attribute comparisons)",
limitation:
"Compares transformer vs non-transformer per challenge but not autoregressive/memory/generative/self-supervised theory underneath.",
whyThisPaper:
"Adds paradigm-level analysis plus a 2020-2024 venue count showing transformer papers overtaking CNN/others (Fig. 8).",
},
{
name: "Handcrafted/heuristic traditional trackers",
limitation:
"Fixed features and assumptions fail scale/pose/illumination/occlusion dynamics of modern applications (surveillance, driving, analysis).",
whyThisPaper:
"Surveys adaptive alternatives (autoregressive likelihood, distractor-aware memory, GAN/VAE/diffusion augmentation, cycle-consistency, MAML-style adaptation).",
},
],
researchGap:
"No SOT survey organized around modern statistical learning paradigms with unified equations, benchmark compilation and a non-traditional technique selection guide.",
contribution: [
"Novel taxonomy tree (Fig. 2): autoregressive + memory sequence models; masked/GAN/diffusion/VAE generative; self-supervised; unsupervised; meta; continual; domain adaptation, each with representative trackers.",
"Method sheets with architectures, losses and diagrams: Eq. 1 autoregression, Eq. 2 GAN minimax, Eqs. 3-4 VAE bound, Eqs. 5-6 cycle/palindrome consistency, plus memory/GNN/LSTM designs.",
"Compiled performance: Tab. 1 (GOT-10k/LaSOT/LaSOText/TrackingNet/OTB50/100) and Tab. 2 (VOT16-20 EAO) with training sets, plus dataset cards (Tab. 3) and metric definitions (AO/SR/AUC/P/PNorm/EAO).",
"Discussion: transformer-dominance trend, language-tracking rise, temporal-correspondence and pretraining lessons, and concrete futures (joint autoregressive+reconstruction, MixAE/NMS pretext tasks, DiffusionDet-style box denoising for SOT).",
],
method: {
pipeline: ["collect-paradigms", "derive-unified-equations", "compile-benchmark-tables", "define-metrics-datasets", "trend-analysis", "future-paths"],
architecture:
"Survey only: no tracker built. Covers encoder-decoder autoregressive ViTs with discrete [x,y,w,h]/[xmin,ymin,xmax,ymax] tokens; ConvLSTM/GCN/RNN memories; MAE pretraining; GAN generator-discriminator maps; VAE manifolds; cycle/palindrome transformers; MAML-style adapters.",
motionModel: "Not applicable — surveys appearance/temporal models, not a motion filter.",
appearanceModel:
"Compares static-template correlation vs dynamic/progressive templates, distractor mining (DDM/TID), mask-reconstruction pretraining (MAT/DropMAE) and generated hard positives (TGGAN/SINT++).",
association: "Not applicable (single-object scope; long-term tracking noted as out of scope except continual-learning relevance).",
detectionDependency: "None — first-frame box given throughout surveyed methods.",
trackManagement:
"Reviews update rules: SeqTrack Tu-frame refresh by token probability; AD-LSTM updater nets; RTS/LWL memory; continual-learning distillation episodes; MDNet online FC layers.",
loss:
"Catalogues cross-entropy+L1 (ARTrack), focal+GIoU+L1 variants, gradient-harmonized (AD-LSTM), GAN+contrastive (TGGAN), VAE ELBO (Eqs. 3-4), cycle-orthogonality/concentration (Joint-Task).",
},
equations: [
{
id: "sotsurvey-autoreg",
label: "Autoregressive tracking likelihood",
formula: "l = max sum_{t=1}^{T} log P(Y^t | Y^t_{t-N:t-1}, X^t, Z)",
variables: [
{ symbol: "Y^t", meaning: "4-token box sequence at frame t" },
{ symbol: "N", meaning: "autoregression order (past frames used)" },
{ symbol: "X^t/Z", meaning: "search image / exemplar (first or past frames)" },
{ symbol: "T", meaning: "number of frames" },
],
intuition: "Spell the box one number at a time, each number allowed to peek at previous numbers and both images.",
why: "Turns tracking into language modeling so likelihood training plus Hanning penalty and Tu-frame refresh apply directly (SeqTrack/ARTrack).",
where: "Sec. 3.1.1, Eq. 1; coordinates discretized into [1, nbins].",
paperIds: ["T103"],
},
{
id: "sotsurvey-gan",
label: "Conditional GAN game for tracking",
formula: "l = E[log D(x,y)] + E[log(1 - D(x, G(x,z)))]",
variables: [
{ symbol: "G", meaning: "generator mapping data+noise to templates/features" },
{ symbol: "D", meaning: "discriminator scoring real vs generated" },
{ symbol: "x/y/z", meaning: "condition data / target maps / noise" },
],
intuition: "A forger (generator) improves by fooling an expert critic (discriminator) that learns what real target variation looks like.",
why: "Unifies TGGAN template generation, VITAL feature-dropout and GARAT distractor removal under one minimax objective.",
where: "Sec. 3.2.2, Eq. 2; Fig. 6 scheme.",
paperIds: ["T103"],
},
{
id: "sotsurvey-vae",
label: "VAE lower bound for tracking",
formula: "L = int q(z|x)(log p(x|z) + log p(x) - log q(z|x))dz; min ||x - f(z)||^2 + KL(q(z|x) || p(z))",
variables: [
{ symbol: "z", meaning: "compact latent object code (Gaussian mu, sigma)" },
{ symbol: "q", meaning: "encoder" },
{ symbol: "p", meaning: "decoder / prior" },
{ symbol: "KL", meaning: "Kullback-Leibler regularizer to Gaussian" },
],
intuition: "Squeeze objects into a smooth compact code space where sampling yields new plausible hard positives.",
why: "Underlies SINT++ PSGN diversity and VAE-MCMC supporter marginalization without enumerating samples.",
where: "Sec. 3.2.4, Eqs. 3-4; Fig. 7 CVAE diagram.",
paperIds: ["T103"],
},
{
id: "sotsurvey-cycle",
label: "Cycle-consistency self-supervision",
formula: "Pp_f = Pt (forward prediction equals start patch); palindrome {Fi..FT..Fi} with Fis ~= Fi .. FTs ~= FT",
variables: [
{ symbol: "Pt/Pp_f", meaning: "start patch / forward-tracked patch" },
{ symbol: "Fi/FT", meaning: "frames along palindrome" },
{ symbol: "Fis", meaning: "model prediction at mirrored position" },
],
intuition: "Track forward then backward: you must return to where you started, which teaches correspondence with zero labels.",
why: "Common backbone of Joint-Task affinity orthogonality, CycleSiam, UDT/PUL, ETC/QCT palindrome transformers.",
where: "Sec. 3.3.1, Eqs. 5-6.",
paperIds: ["T103"],
},
],
datasets: ["got10k", "lasot", "trackingnet", "otb", "vot", "others"],
metrics: ["success-auc", "precision", "norm-precision", "eao"],
baselines: ["SeqTrack", "ARTrack", "RFL", "GCT", "Siamese R-RPN", "AD-LSTM", "BA+TID+DDM", "MAT", "DropMAE", "TGGAN", "VITAL", "AFSL", "GARAT", "Diff-SiamRPN", "SINT++", "VAE-MCMC", "Joint-Task", "CycleSiam", "UniTrack", "Crop-Transform-Paste", "ETC", "QCT", "UDT", "PUL", "Meta-Tracker", "DIMP", "CLNet", "MGA-Net", "RTS", "Continual-Learning", "ConTACT", "MDNet", "CODA"],
results: [
"Tab. 1 compiles (train-on-GOT/LaSOT/TrackingNet/COCO row examples): ARTrack-L384 GOT-10k AO 78.5 / SR0.5 87.4 / SR0.75 77.8, LaSOT AUC 73.1; SeqTrack-L384 AO 74.8 / LaSOT 72.5 / TrackingNet 85.5; DropMAE AO 75.9; Diff-SiamRPN AO 73.5 / LaSOT 63.6; DIMP-50 AO 61.1 / TrackingNet 74.0; RTS LaSOT 69.7 / TrackingNet 81.6.",
"Tab. 2 compiles VOT EAO: MGA-SiamRPN++ 47.5 (VOT16) / 42.5 (VOT18); Siamese R-RPN 37.5/39.8; VITAL 32.3; GARAT 36.0/34.7/36.1; UDT 30.1; RTS 50.6 (VOT20); BA+TID+DDM 50.1 (VOT20).",
"Trend (Fig. 8, 10 top venues 2020-2024): CNN-led before 2023, transformer-led after; attention's receptive field, large SOT sets and weak inductive bias given as causes.",
"Metric/dataset reference: AO (class-weighted GOT-10k), SR0.5/0.75 (x3 averaged), AUC success-curve, P@20px, PNorm (size-normalized), EAO (VOT); Tab. 3 cards for GOT-10k/LaSOT/LaSOText/TrackingNet/OTB.",
],
ablations: [
"No experiments: survey explicitly warns Tabs. 1-2 are evolution records, not fair fights across years/backbones/compute (Sec. 4.3).",
"Within-text contrasts: ARTrack384 (75.5 AO) vs ARTrack256 (73.5) for resolution; MAE-pretrained vs CLS variants in ProContEXT-era rows; ULS vs FLS in ETC (44.7 vs 50.6 LaSOT AUC).",
],
limitations: {
authorStated: [
"Diffusion in SOT is only sample generation (Diff-SiamRPN on ImageNet-1K prompts), not end-to-end box denoising; GAN discrimination lessons not yet fused with diffusion.",
"Temporal cues and visual representations are learned separately; stronger pretext tasks (Mixed Autoencoder, No-More-Shortcuts skip/order) and language-conditioned tracking remain future work.",
"Performance tables mix distant years, backbones and compute and must not be read as controlled comparisons (Sec. 4.3).",
],
evident: [
"Coverage stops at May 2024 with sequence/generative focus; long-term, multimodal and MOT advances outside SOT scope are thin.",
"No re-implementation or unified re-evaluation: compiled numbers inherit each source paper's protocol quirks (e.g. VOT box convention vs axis-aligned sets).",
],
},
assumptions: [
"First-frame box given; surveyed SOT scope excludes long-term disappearance handling except via continual-learning notes.",
"Compiled scores are quoted as reported under each dataset's own protocol (GOT-10k class-weighted AO, VOT EAO).",
],
computation:
"No compute reported (survey; funding/code/materials listed as not applicable); trend count covers CVPR/ICCV/ECCV/AAAI/NeurIPS/WACV plus TPAMI/TIP/IJCV/TCSVT 2020-2024.",
relations: [
{ to: "T009", type: "conceptual-successor", note: "Surveys SiamFC fully-convolutional base reused by PUL/RFL-style Siamese analyses." },
{ to: "T071", type: "conceptual-successor", note: "Surveys OSTrack one-stream framework used as DropMAE integration base." },
{ to: "T028", type: "conceptual-successor", note: "Surveys SiamRPN++ evolution line via CLNet/MGA adaptation discussions." },
{ to: "T025", type: "conceptual-successor", note: "Compiles GOT-10k AO/SR tables and protocol (train/test class split, absence labels)." },
{ to: "T024", type: "conceptual-successor", note: "Compiles LaSOT/LaSOText AUC/P/PNorm tables and attribute taxonomy." },
],
concepts: ["sot", "siamese", "attention", "correlation-filter", "success-plot", "benchmark-design", "multimodal"],
impact:
"Reference map sending later SOT work toward joint autoregressive-plus-reconstruction training, harder self-supervised pretext tasks and DiffusionDet-style box-denoising trackers.",
},
{
id: "T104",
arxiv: "2405.14200",
title: "Awesome Multi-modal Object Tracking",
shortTitle: "Awesome MMOT",
year: 2024,
authors: ["Chunhui Zhang", "Li Liu", "Hao Wen", "Xi Zhou", "Yanfeng Wang"],
fileName: "2405.14200v2.pdf",
task: "survey",
tags: ["survey", "multimodal-tracking", "rgbt-tracking", "rgbd-tracking", "vision-language-tracking", "rgb-event-tracking", "benchmark-catalog"],
difficulty: "intro",
summary:
"A project report that gives the first comprehensive investigation of multi-modal object tracking (MMOT): it divides all MMOT work into five tasks (RGBL, RGBE, RGBD, RGBT, miscellaneous RGB+X), tabulates widely used datasets per task with project/code links, tabulates mainstream methods by technical paradigm (self-supervised learning, prompt learning, knowledge distillation, generative models, state-space models), and maintains a continuously updated paper list on GitHub.",
problem:
"RGB trackers fail under lighting change, fast motion, occlusion and appearance variation, so extra modalities (thermal, depth, event, language, audio) are added; but existing reviews cover only modality pairs (RGB+depth, RGB+thermal) and no investigation covers the popular RGBL/RGBE tasks, unified any-modality models, or the new 3+-modality benchmarks (WebUAV-3M vision-language-audio, UniMod1K vision-depth-language).",
background: ["sot", "bounding-box", "multimodal", "benchmark-design", "success-plot", "appearance-features"],
previousWork: [
{
name: "RGB+depth reviews (Tang et al., Yang et al., Ou et al.)",
limitation: "Cover only the RGBD pair, missing language/event/thirst-modality work.",
whyThisPaper: "Adds RGBL, RGBE and miscellaneous RGB+X alongside RGBD in one five-way taxonomy.",
},
{
name: "RGB+thermal reviews (Tang et al., Zhang et al.)",
limitation: "Fusion-perspective or benchmark-validity studies of RGBT only.",
whyThisPaper: "Places RGBT as one of five MMOT branches with its datasets (GTOT, RGBT210/234, LasHeR, VTUAV, MV-RGBT) and method list.",
},
{
name: "Depth+thermal review (Zhang, Wang, Lu) and early multi-modal survey (Li et al.)",
limitation: "Predate the RGBL/RGBE wave and unified any-modality trackers (ViPT, Un-Track, OneTracker).",
whyThisPaper: "Catalogs prompt-learning/unified trackers and new multi-modal benchmarks (WebUAV-3M, UniMod1K) plus a living GitHub list.",
},
],
researchGap:
"No investigation jointly covered RGBL, RGBE, RGBD, RGBT and 3+-modality tracking with datasets, methods and paradigms in one place.",
contribution: [
"Five-way MMOT taxonomy: RGBL (vision-language), RGBE (frame+event), RGBD (color+depth), RGBT (visible+thermal), miscellaneous RGB+X (any extra modalities).",
"Informal definitions of each task with sample figures and the complementary strength each modality adds (e.g. events for rapid motion/extreme lighting, depth for scene geometry, thermal for illumination).",
"Dataset catalog per task with publish venue, project page and code base: RGBL (OTB99-L, LaSOT, LaSOTExt, TNL2K, WebUAV-3M, MGIT, VastTrack, WebUOT-1M), RGBE (FE108, COESOT, VisEvent, EventVOT, CRSOT, FELT), RGBD (PTB, STC, CDTB, DepthTrack, RGBD1K, DTTD, ARKitTrack), RGBT (GTOT, RGBT210/234, LasHeR, VTUAV, MV-RGBT), misc (WebUAV-3M, UniMod1K).",
"Method catalog per task organized by technical paradigm: self-supervised learning, prompt learning, knowledge distillation, generative models and state-space models.",
"Living GitHub paper list (Awesome-Multimodal-Object-Tracking) for continuous updates toward large multi-modal foundation tracking models.",
],
method: {
pipeline: ["define-taxonomy", "catalog-datasets", "catalog-methods", "paradigm-analysis", "maintain-list"],
architecture:
"Literature report, no tracker: five-branch taxonomy (Fig. 1) with per-task dataset table (Table 1: publish/title/links/introduction) and per-task method table (Table 2: publish/title/paper-link/code).",
detectionDependency: "None — survey of single-object tracking work (all discussed tasks are single-object tracking).",
trackManagement: "Not applicable — no tracking algorithm proposed.",
optimization: "Not applicable — no training.",
loss: "Not applicable — no training.",
},
equations: [],
datasets: ["otb", "lasot", "lasheR", "webuav-3m", "others"],
metrics: ["success-auc", "precision"],
baselines: ["RGBT benchmarks (GTOT, RGBT210, RGBT234, LasHeR)", "RGBD benchmarks (PTB, CDTB, DepthTrack)", "RGBE benchmarks (FE108, VisEvent)", "RGBL benchmarks (OTB99-L, TNL2K)", "Unified trackers (ViPT, ProTrack, Un-Track, OneTracker, SDSTrack)"],
results: [
"Scope fixed as five MMOT tasks with informal definitions: RGBL tracks from natural-language specification; RGBE fuses frames with asynchronous event streams; RGBD adds depth; RGBT adds thermal; miscellaneous RGB+X combines RGB with any extra modalities.",
"Dataset table documents scale landmarks: LasHeR 1224 RGBT pairs over 730K frames; VTUAV 500 sequences with 1,664,549 visible-thermal pairs; WebUAV-3M 4500 videos / 3.3M frames with vision, language and audio; UniMod1K 1050 pairs / 2.5M frames with vision, depth and language.",
"Method table spans classical correlation-filter RGBD/RGBT work (PTB, DS-KCF, STC, MANet, DAFNet, mfDiMP) through Siamese and Transformer fusion (SiamCDA, DMCNet, TBSI, BAT, GMMT) to unified prompt-learning trackers (ViPT, ProTrack, Un-Track, OneTracker, SDSTrack, SeqTrackv2).",
"Paradigm analysis named for mainstream algorithms: self-supervised learning, prompt learning, knowledge distillation, generative models and state-space models.",
],
ablations: [
"No ablations — survey report with no experiments; Tables 1–2 are catalogs, not comparisons.",
],
limitations: {
authorStated: [
"Report is a starting snapshot of a rapidly evolving field and defers currency to the continuously updated GitHub paper list.",
],
evident: [
"No quantitative comparison or controlled evaluation between cataloged methods — tables give links, not scores.",
"All discussed tasks are single-object tracking; multi-object and segmentation tracking are out of scope.",
"Dataset/method entries inherit whatever the cited projects claim; no independent verification shown.",
],
},
assumptions: [
"Reader accepts the five-way modality taxonomy as the organizing principle.",
"Project/code links tabulated were live at report time (May 2024).",
],
computation: "No training or inference — literature investigation plus GitHub list maintenance.",
relations: [
{ to: "T057", type: "conceptual-successor", note: "Consolidates the LasHeR RGBT benchmark thread (1224 pairs / 730K frames) into the five-task survey." },
{ to: "T083", type: "conceptual-successor", note: "Catalogs ViPT prompt-learning as a miscellaneous RGB+X unified-tracking paradigm." },
{ to: "T097", type: "conceptual-successor", note: "Catalogs Un-Track single-model any-modality tracking in the miscellaneous branch." },
],
concepts: ["multimodal", "sot", "benchmark-design", "bounding-box", "success-plot", "appearance-features"],
impact:
"First five-task MMOT map plus a living paper list that later unified-tracker papers use to position RGB-X work.",
},
{
id: "T105",
arxiv: "2405.17773",
title: "XTrack: Multimodal Training Boosts RGB-X Video Object Trackers",
shortTitle: "XTrack",
year: 2024,
authors: ["Yuedong Tan", "Zongwei Wu", "Yuqian Fu", "Zhuyun Zhou", "Guolei Sun", "Eduard Zamfir", "Chao Ma", "Danda Pani Paudel", "Luc Van Gool", "Radu Timofte"],
fileName: "2405.17773v2.pdf",
task: "multimodal-tracking",
tags: ["multimodal-tracking", "mixture-of-experts", "soft-routing", "prompt-learning", "rgbt-tracking", "rgbd-tracking", "rgb-event-tracking", "transformer"],
difficulty: "advanced",
summary:
"XTrack trains one RGB-X tracker on paired RGB-E, RGB-D and RGB-T data but deploys with only one auxiliary modality at a time. A Mixture of Modal Experts (MeME) with a deliberately weak modality classifier routes confusingly similar cross-modal samples to each other's experts so transferable attributes (geometry, motion, low-light adaptation) are shared, while an edge-gated shared expert and modal prompting inject the shared knowledge into frozen RGB foundation features. It reports average +3% precision over SOTA across RGB-X scenarios with XTrack-B and XTrack-L variants.",
problem:
"Multimodal tracking is data-sparse: only one auxiliary modality is available at a time and no dataset contains all modalities, so prior unified trackers use rigid modality-specific branches activated by prior input-type knowledge. Strict isolation prevents cross-modal knowledge transfer even though samples across modalities share attributes (e.g. event and thermal both handle illumination change), and multi-modality training can even hurt (ViPT worse on events when trained on multiple modalities).",
background: ["sot", "bounding-box", "attention", "multimodal", "appearance-features", "success-plot", "iou"],
previousWork: [
{
name: "Modality-tailored trackers (BAT, TBSI, DeT, DepthTrack baselines)",
limitation: "Dual architectures handle one extra modality only; knowledge cannot transfer across modal domains.",
whyThisPaper: "One MeME model trains on RGB-E/D/T pairs jointly and improves every single-modality test (Table 6).",
},
{
name: "Rigid unified trackers (ViPT, UnTrack, OneTracker, SDSTrack, ProTrack)",
limitation: "Only the matching modality-specific parameters activate at inference, so no interaction or joint benefit across modalities during training.",
whyThisPaper: "Soft router lets other modalities' experts process a sample when features align, beating the rigid 100%-classification design (DepthTrack F-score 58.8 rigid vs 61.5 soft).",
},
{
name: "Generalist multimodal models (Meta-Transformer, ImageBind, PolyViT)",
limitation: "Joint-space pretraining outside tracking; no mechanism for sparse-data RGB-X tracking with missing modalities at inference.",
whyThisPaper: "First systematic cross-modal sharing inside a VOT tracker: weak-classifier confusion signals the minimal-domain-gap sharing opportunity.",
},
],
researchGap:
"No RGB-X tracker shared knowledge across modalities during training while deploying with a single modality and no prior modality information.",
contribution: [
"Weak-classifier insight: when a modality classifier fails on similar cross-modal samples, the confusion marks minimal domain gap and maximal sharing opportunity.",
"Mixture of Modal Experts (MeME) inserted after each attention and FFN block of a frozen RGB foundation tracker (OSTrack-based XTrack-B, DropMAE-based XTrack-L), with bidirectional RGB↔X refinement.",
"Soft router supervised by expert-balance loss plus multi-class modality-classification loss (Eqs. 3–8) with top-k=2 routing; 80% classification operating point beats rigid (100%) and random (33%) assignment.",
"Shared expert plus modality-specific low-dimensional experts (k << c, single 24GB GPU), edge-gated shared expert with Laplacian-initialized EdgeMix convolution, shrinkage fusion and modal prompting of RGB tokens.",
"SOTA on DepthTrack, VOT-RGBD22, LasHeR, RGBT234 and VisEvent with one jointly trained model and no per-modality parameters, average +3 precision points over prior SOTA.",
],
method: {
pipeline: ["patch-embed", "frozen-rgb-attention", "meme-refine", "ffn", "meme-refine", "edge-shared-expert", "fuse", "modal-prompt", "head"],
architecture:
"Frozen transformer RGB foundation (OSTrack weights for XTrack-B, DropMAE for XTrack-L); trainable MeME after every Attn and FFN: per-modality low-dim experts plus one edge-gated shared expert, batch-concat shrinkage fusion, prompt module gating RGB tokens.",
motionModel: "No filter — template/search matching with cross-modal prompted features.",
appearanceModel:
"Modality-specific experts (geometry from depth, motion from events, temperature/low-light from thermal) plus shared edge-biased expert; only the respective modality expert chosen at inference.",
detectionDependency: "None — class-agnostic single-object RGB-X tracker.",
trackManagement: "Standard template/search inference of the frozen foundation tracker; no update rule described.",
loss:
"Foundation tracking loss plus Lmoe = Lcls + lambda·Lbalance (Lbalance = LImp + LLoad); binary-cross-entropy modality classification against expert mapping h with balance weight 0.001.",
optimization:
"Only MeME learnable; batch 32, lr 4e-4 (÷10 after 78 epochs), 90 epochs; trains on DepthTrack (RGB-D) + LasHeR (RGB-T) + VisEvent (RGB-E) pairs only.",
},
equations: [
{
id: "xtrack-meme-attn",
label: "MeME-refined attention output",
formula: "T_rgb^attn,l = T_rgb^l + Attn(T_rgb^l) + MeME(T_rgb^l, T_x^l)",
variables: [
{ symbol: "T_rgb^l / T_x^l", meaning: "RGB and auxiliary-modality tokens at block l" },
{ symbol: "Attn", meaning: "frozen foundation transformer attention" },
{ symbol: "MeME", meaning: "mixture of modal experts refinement" },
],
intuition: "Add a cross-modal correction on top of the frozen RGB features at every block.",
why: "Lets X knowledge improve RGB features (and vice versa) without unfreezing the foundation.",
where: "Sec. 3.1, Eq. 1; repeated after the FFN in Eq. 2.",
paperIds: ["T105"],
},
{
id: "xtrack-routing",
label: "Top-k expert routing",
formula: "y = sum_{i in top-k} p_i(T_x) * eps_i(T_x)",
variables: [
{ symbol: "p_i", meaning: "router probability for expert i" },
{ symbol: "eps_i", meaning: "expert i network" },
{ symbol: "k", meaning: "number of routed experts (2)" },
],
intuition: "Send each token to its k most suitable experts and blend their outputs.",
why: "Sparse blending gives specialization plus cross-modal sharing when features align.",
where: "Sec. 3.2, Eq. 3; visualized top-2 choices in Fig. 5.",
params: "k=2; more experts per modality degrades (Table 5: 2 experts best).",
paperIds: ["T105"],
},
{
id: "xtrack-balance",
label: "Expert balance loss",
formula: "L_balance = L_Imp + L_Load; L_Load = Var(Load_i)/Mean(Load_i)^2, Load_i = sum_{Tx in B} Phi(p_i(Tx)); L_Imp same without Phi",
variables: [
{ symbol: "Load_i / Imp_i", meaning: "assignments / probability mass of expert i over batch B" },
{ symbol: "Phi", meaning: "CDF of N(0, sigma^2 I), sigma = gate-noise / |eps|" },
{ symbol: "B", meaning: "mini-batch" },
],
intuition: "Penalize lopsided expert usage so no expert collapses or hogs all tokens.",
why: "Keeps modality-specific experts alive for sharing instead of one degenerate router winner.",
where: "Sec. 3.2, Eqs. 4–6.",
paperIds: ["T105"],
},
{
id: "xtrack-cls",
label: "Modality classification (weak router) loss",
formula: "L_cls = -sum_{n=1}^{|M|} sum_{i=1}^{|eps|} 1{h(m_n)=i} log p_i(T_x)",
variables: [
{ symbol: "M / m_n", meaning: "modality set / n-th modality subset" },
{ symbol: "h", meaning: "mapping from modality to its dedicated experts" },
{ symbol: "p_i", meaning: "router probability of expert i" },
],
intuition: "Teach the router which experts belong to which modality — but only weakly.",
why: "Deliberately imperfect classification leaves confusion that routes similar samples to other modalities' experts (sharing).",
where: "Sec. 3.2, Eq. 7; L_moe = L_cls + lambda·L_balance, Eq. 8.",
params: "Balance weight 0.001 gives 80% classification, the best operating point (Table 7).",
paperIds: ["T105"],
},
{
id: "xtrack-edge",
label: "Edge-gated shared expert",
formula: "X = Norm(m_s^k); Out = (sigma(EdgeMix(X W1)) * (X W2)) W3 + m_s^k",
variables: [
{ symbol: "m_s^k", meaning: "shared low-dimensional feature (k << c)" },
{ symbol: "EdgeMix", meaning: "token-mixing convolution with Laplacian initialization" },
{ symbol: "W1/W2/W3", meaning: "learnable low-dim projections" },
],
intuition: "Bias the shared expert toward high-frequency edge structure common to all modalities.",
why: "Small downstream data cannot learn shared features implicitly; the edge prior guides commonality discovery.",
where: "Sec. 3.3, Eq. 10, Fig. 4(a).",
paperIds: ["T105"],
},
{
id: "xtrack-prompt",
label: "Modal prompting of RGB tokens",
formula: "Xi = Norm(I_k), Xm = Norm(M_k); Out = ((Xi W5 * sigma(Xm W6)) W7 + I_k) W8",
variables: [
{ symbol: "I_k", meaning: "RGB token projected to low-dim latent space" },
{ symbol: "M_k", meaning: "fused modality-specific + shared matrix (Eq. 11)" },
{ symbol: "W5..W8", meaning: "learnable gating and back-projection weights" },
],
intuition: "Gate RGB features with the distilled cross-modal knowledge before projecting back.",
why: "Makes frozen RGB features modality-aware for challenging (low-light/blur/occlusion) frames.",
where: "Sec. 3.4, Eq. 12.",
paperIds: ["T105"],
},
],
datasets: ["lasheR", "others"],
metrics: ["precision", "success-auc", "eao"],
baselines: ["SPT", "DeT", "DDiMP", "ATCAIS", "DAFNet", "OSTrack", "ProTrack", "ViPT", "UnTrack", "OneTracker", "SDSTrack", "SGT", "CAT", "FANet", "APFNet", "TransT", "SiamRCNN", "KeepTrack", "DRefine"],
results: [
"DepthTrack: XTrack-B F-score 61.5 / Re 62.0 / Pr 61.8 and XTrack-L 64.8 / 64.3 / 65.4 vs best prior SDSTrack 61.4 / 60.9 / 61.9.",
"VOT-RGBD22: XTrack-B EAO 74.0 / Acc 82.1 / Rob 88.8 and XTrack-L 74.0 / 82.8 / 88.9 vs SDSTrack 72.8 / 81.2 / 88.3, despite the train/test domain gap.",
"LasHeR: XTrack-B Pr 69.1 / Sr 55.7 and XTrack-L 73.1 / 58.7 vs OneTracker 67.2 / 53.8.",
"RGBT234: XTrack-B MSR 87.4 / MPR 64.9 and XTrack-L 87.8 / 65.4 vs OneTracker 85.7 / 64.2.",
"VisEvent: XTrack-B Pr 77.5 / Sr 60.9 and XTrack-L 80.5 / 63.3 vs OneTracker/SDSTrack 76.7 / 60.8/59.7.",
"Shared-expert ablation (Table 4): removing shared experts drops DepthTrack F-score 61.5 to 57.6 (-3.9); single-expert-per-all-modalities drops to 59.1.",
"Expert count (Table 5): 2 experts per modality best (61.5 F-score); 1 gives 60.0, 3 gives 60.9.",
"Joint training (Table 6): event-only training already reaches 76.3 Pr on VisEvent (+8 over OSTrack 69.5 baseline) with thermal zero-shot 58.1 Pr; adding depth then thermal lifts all three (77.5 / 61.5 / 69.1 Pr).",
"Router softness (Table 7): 80% classification (61.5 F-score) beats rigid 100% (58.8) and near-random 33% (61.0).",
],
ablations: [
"Shared vs specific experts (Table 4): both needed; shared-expert removal hurts depth most.",
"Expert number (Table 5): 1 limits representation, 3 causes internal conflicts.",
"Progressive joint training (Table 6): each added modality helps its own and mostly others; event→thermal zero-shot works, event→depth does not (larger domain gap).",
"Rigid vs soft classifier (Table 7): strict separation hinders; ambiguity enables sharing.",
"Routing visualization (Fig. 5): low-light activates thermal experts, blur activates event experts, geometry activates depth experts alongside the primary expert.",
],
limitations: {
authorStated: [
"Identical expert designs for every modality; salient attributes that deserve adaptive compute are not studied.",
"Adaptive computational cost across modalities left for future work.",
],
evident: [
"Trained only on three paired datasets (DepthTrack, LasHeR, VisEvent); no all-modalities or missing-modality training shown.",
"Inference still selects the respective modality's expert — deployment assumes the input modality is known.",
"Zero-shot transfer fails across large domain gaps (event-trained model poor on depth).",
],
},
assumptions: [
"Only one RGB-X pair available at a time during training; no complete multimodal sample exists.",
"Similar cross-modal samples share transferable attributes worth routing jointly.",
"Frozen RGB foundation features are sufficient and only need prompting.",
],
computation:
"Only MeME learnable; batch 32, lr 4e-4 (÷10 after epoch 78), 90 epochs on a single 24 GB GPU.",
relations: [
{ to: "T071", type: "builds-on", note: "XTrack-B freezes OSTrack as the RGB foundation and learns only MeME on top." },
{ to: "T083", type: "improves", note: "Beats ViPT on all three modalities (e.g. VisEvent 77.5 vs 75.8 Pr) where multi-modal ViPT training hurt." },
{ to: "T097", type: "uses-as-baseline", note: "UnTrack single-model any-modality baseline beaten on DepthTrack/LasHeR/VisEvent tables." },
{ to: "T055", type: "uses-as-baseline", note: "TransT event variant (65.0 Pr) beaten in VisEvent Table 3." },
],
concepts: ["sot", "attention", "multimodal", "appearance-features", "bounding-box", "success-plot"],
impact:
"First weak-classifier soft-routing recipe for RGB-X training with single-modality deployment; event+thermal low-light sharing analysis reused by later unified trackers.",
},
{
id: "T106",
arxiv: "2406.20024",
title: "eMoE-Tracker: Environmental MoE-based Transformer for Robust Event-guided Object Tracking",
shortTitle: "eMoE-Tracker",
year: 2024,
authors: ["Yucheng Chen", "Lin Wang"],
fileName: "2406.20024v3.pdf",
task: "multimodal-tracking",
tags: ["multimodal-tracking", "rgb-event-tracking", "mixture-of-experts", "contrastive-learning", "prompt-tuning", "one-stream-transformer", "attribute-disentanglement"],
difficulty: "advanced",
summary:
"eMoE-Tracker is a one-stream RGB-event tracker that disentangles the environment into four learnable attributes (illumination variance, motion blur, scale variance, occlusion). An environmental Mixture-of-Experts (eMoE) module learns one attribute-specific feature per expert and assembles them with learnable video-level attribute scores to prompt-tune a frozen ViT backbone, while a contrastive relation modeling (CRM) module pulls target template features toward in-box search tokens and pushes background away. It reaches 61.3 SR / 76.4 PR on VisEvent and 67.1 SR / 79.9 PR on COESOT.",
problem:
"Frame+event fusion methods fuse both modalities directly and ignore environmental attributes, while weak search–template interaction cannot separate target from background — so performance collapses under illumination variance, motion blur, scale change and occlusion, the exact conditions event cameras should help with.",
background: ["sot", "bounding-box", "attention", "multimodal", "appearance-features", "reid", "success-plot"],
previousWork: [
{
name: "Two-stream Siamese RGB-E trackers",
limitation: "Separate RGB/event branches need complex hand-designed fusion modules, raising model complexity.",
whyThisPaper: "One-stream ViT with eMoE prompt-tuning needs only 8.42 trainable params vs 25–42 for two-stream AFNet/FENet.",
},
{
name: "One-stream RGB-E trackers (CEUTrack, ViPT)",
limitation: "Concatenate or prompt tokens but ignore environmental attributes; inadequate search–template interaction.",
whyThisPaper: "Disentangles four attributes into experts plus CRM contrastive interaction; beats ViPT by 6.2 PR points on COESOT.",
},
{
name: "RGB backbones (OSTrack, MixFormer, ATOM, SiamRPN++)",
limitation: "Frame-only features fail in low light/fast motion where events complement RGB.",
whyThisPaper: "Stacked event frames concatenated as extra tokens; eMoE assembles attribute features onto the frozen backbone (+6.2 SR over backbone on VisEvent).",
},
],
researchGap:
"No one-stream RGB-E framework distinguished environmental attributes while strengthening target–background interaction.",
contribution: [
"Environmental MoE (eMoE): four CONV-MLP-CONV experts disentangle illumination variance, motion blur, scale variance and occlusion from video-level manual labels.",
"Attribute assembling network (CONV-BN-ReLU-CONV-Sigmoid, two loops) predicting K attribute scores W^{l,t} to weight and assemble attribute-specific features per ViT layer: T^l = T_{RGB-E}^l + P^{l+1}.",
"Contrastive relation modeling (CRM): fuses template/search tokens, builds positive pairs (template vs in-box search tokens) and negative pairs (vs background) with InfoNCE loss.",
"Prompt-tuning design: backbone frozen, only eMoE+CRM updated; eMoE insertable at arbitrary layers (interval 1 = every layer best).",
"SOTA on VisEvent (61.3/76.4/79.6 SR/PR/NPR) and COESOT (67.1/79.9/82.3), plus per-attribute PR wins and FELT-SOT generalization (65.1/72.9/73.6).",
],
method: {
pipeline: ["patch-embed", "concat-rgb-event", "frozen-vit", "emoe-disentangle", "assemble", "add", "crm-contrast", "decode", "regress"],
architecture:
"One-stream ViT (OSTrack/CEUTrack-style, 12 encoder layers, frozen) on concatenated [T_RGB^z, T_RGB^x, T_E^z, T_E^x]; eMoE inserted per layer; CRM head; vanilla box decoder.",
motionModel: "No filter — search window matching with environment-assembled features.",
appearanceModel:
"Four attribute-specific expert features assembled by learned scores; CRM contrastive pull/push sharpens target vs background.",
detectionDependency: "None — single-object RGB-E tracker from first-frame box.",
trackManagement: "Implicit — no birth/death rules; tracking by per-frame box regression.",
loss:
"L = Ltracking + alpha·LNCE + beta·Lattr; Ltracking = Lcls (focal) + lambda_iou·Liou + lambda_L1·L1; LNCE InfoNCE over template/search pairs; Lattr L1 between predicted scores W^{l,t} and video-level labels Gt.",
optimization:
"End-to-end on 1×A800, batch 64, 60 epochs (6×10^4 pairs/epoch), AdamW wd 1e-4, lr 2e-4 (÷10 after 32 epochs); only eMoE+CRM updated.",
},
equations: [
{
id: "emoe-fuse",
label: "eMoE prompt-tuning injection",
formula: "T^l = T_{RGB-E}^l + P^{l+1}, l = 1..L",
variables: [
{ symbol: "T_{RGB-E}^l", meaning: "backbone tokens (RGB+event template and search) at layer l" },
{ symbol: "P^{l+1}", meaning: "assembled attribute features from eMoE at layer l+1" },
{ symbol: "L", meaning: "number of ViT encoder layers (12)" },
],
intuition: "Add environment-aware correction features onto the frozen backbone at each layer.",
why: "Prompt-tunes the backbone efficiently instead of fine-tuning all weights.",
where: "Sec. III-B, Eq. 1.",
paperIds: ["T106"],
},
{
id: "emoe-assemble",
label: "Attribute assembling",
formula: "F_assemble^l = sum_{t=1}^{K} W^{l,t} H_t^l",
variables: [
{ symbol: "H_t^l", meaning: "attribute-specific feature of expert t at layer l" },
{ symbol: "W^{l,t}", meaning: "learnable attribute score (contribution ratio) from assembling net" },
{ symbol: "K", meaning: "number of experts (4)" },
],
intuition: "Weight each attribute's evidence by how present that challenge is in the video.",
why: "A motion-blurred video should listen to the blur expert and suppress occlusion noise.",
where: "Sec. III-B, assembling network over all RGB+event tokens.",
paperIds: ["T106"],
},
{
id: "emoe-trackloss",
label: "Tracking loss",
formula: "L_tracking = L_cls + lambda_iou L_iou + lambda_L1 L_1",
variables: [
{ symbol: "L_cls", meaning: "focal classification loss" },
{ symbol: "L_iou / L_1", meaning: "IoU and L1 box regression losses" },
{ symbol: "lambda_iou / lambda_L1", meaning: "regression regularization weights" },
],
intuition: "Classify foreground while regressing tight boxes, as in the OSTrack backbone.",
why: "Keeps the standard detection-quality objective alongside contrastive and attribute terms.",
where: "Sec. III-C, Eq. 3.",
simulator: "iou-track",
paperIds: ["T106"],
},
{
id: "emoe-infonce",
label: "CRM contrastive loss",
formula: "L_NCE = -log( sum_{k=1}^{Npos} e^{s_k^p} / (sum_{k=1}^{Npos} e^{s_k^p} + sum_{k=1}^{Nneg} e^{s_k^n}) ), s_i = sim(T_f^{z,used}, t_f^{x,i,used}) / tau",
variables: [
{ symbol: "T_f^{z,used}", meaning: "fused target template tokens" },
{ symbol: "s_k^p / s_k^n", meaning: "similarities of positive (in-box) / negative (background) search tokens" },
{ symbol: "tau", meaning: "temperature parameter" },
{ symbol: "Npos / Nneg", meaning: "counts of positive / negative search tokens" },
],
intuition: "Pull the template toward target pixels and push it away from background pixels.",
why: "Explicit interaction that plain concatenation lacks; makes target/background unambiguous.",
where: "Sec. III-B, Eq. 4, Fig. 4.",
simulator: "reid",
paperIds: ["T106"],
},
{
id: "emoe-attr",
label: "Attribute supervision loss",
formula: "L_attr = sum_{t,l} ||W^{l,t} - G_t||_{l1}; L = L_tracking + alpha L_NCE + beta L_attr",
variables: [
{ symbol: "G_t", meaning: "manual video-level attribute label for expert t" },
{ symbol: "W^{l,t}", meaning: "predicted attribute score" },
{ symbol: "alpha / beta", meaning: "contrastive / attribute loss weights" },
],
intuition: "Force the assembling weights to match human judgments of which challenges appear.",
why: "Grounds each expert in its named environmental attribute instead of arbitrary latents.",
where: "Sec. III-C, Eqs. 5–6.",
paperIds: ["T106"],
},
],
datasets: ["others"],
metrics: ["success-auc", "precision", "norm-precision"],
baselines: ["Ocean", "SiamCAR", "SiamRPN++", "ATOM", "PrDiMP", "LTMU", "SiamAPN++", "EFTrack", "FENet", "AFNet", "OSTrack", "CEUTrack", "ViPT", "MixFormer", "STARK", "AiATrack"],
results: [
"VisEvent: SR 61.3 / PR 76.4 / NPR 79.6 vs ViPT 59.2 / 75.8 / 73.2 and backbone OSTrack 53.4 / 69.5 / 72.6 (+6.2 SR / +6.9 PR over backbone).",
"COESOT: SR 67.1 / PR 79.9 / NPR 82.3 vs ViPT 65.3 / 73.7 / 65.9 (+1.8 SR / +6.2 PR) and CEUTrack 62.7 / 70.9 / 73.9.",
"COESOT attributes (PR): occlusion 63.6, illumination variance 77.9, motion blur 73.5, scale variance 80.7 — best on all four vs backbone and ViPT (Fig. 7).",
"Module ablation (Table III): backbone 53.4/69.5 → +eMoE 59.2/75.8 → +CRM 61.3/76.4 on VisEvent (SR/PR); unfreezing the header drops to 59.0/74.2.",
"Expert count (Table IV): 4 experts (61.3/76.4) beat 1 (54.2/70.8), 2 (58.6/71.6), 3 (59.0/73.5) on VisEvent.",
"Insert interval (Table V): every layer (interval 1: 61.3/76.4) beats intervals 2/4/6/12 on both datasets.",
"FELT-SOT generalization (appendix): SR 65.1 / PR 72.9 / NPR 73.6.",
"Complexity (Table VI): 8.42 trainable params vs AFNet 25.16 / FENet 41.87 / CEUTrack 93.7; inference 46 FPS vs OSTrack 93.1 / ViPT 49 / CEUTrack 75.",
],
ablations: [
"eMoE vs CRM (Table III): eMoE carries most of the gain (+5.8 SR); CRM adds +2.1 SR / +0.6 PR.",
"Frozen vs unfrozen header: unfreezing hurts (59.0 vs 61.3 SR) — prompt-tuning beats full fine-tuning here.",
"Expert number (Table IV): monotonic gains 1→4; more than 4 infeasible with manual annotations.",
"Insertion density (Table V): full insertion wins; sparse high-layer-only insertion worst (54.7 SR).",
"Attribute curves (Figs. 11–12): wins across 16/17 attributes on both datasets.",
],
limitations: {
authorStated: [
"Highly dependent on manual video-level attribute annotations, restricting generalization.",
"Video-level labels may be imprecise for individual sequences under some conditions.",
"Future work: learnable agent for environmental attributes instead of manual labels.",
],
evident: [
"Events used as stacked frames, not raw asynchronous streams — temporal resolution advantage partially discarded.",
"46 FPS is the slowest of compared one-stream trackers (ViPT 49, CEUTrack 75).",
"Four-attribute choice fixed by annotation budget, not by data-driven selection.",
],
},
assumptions: [
"First-frame box given; event streams stackable into frames aligned with RGB.",
"Video-level attribute labels adequately describe per-frame conditions.",
"Template tokens are mostly target, search tokens mix target and background (CRM pairing premise).",
],
computation:
"1×NVIDIA A800, batch 64, 60 epochs, AdamW (wd 1e-4), lr 2e-4 (÷10 after 32); 8.42 trainable params; 46 FPS inference.",
relations: [
{ to: "T083", type: "builds-on", note: "Adopts ViPT-style frozen-ViT prompt-tuning; appendix uses the same ViT structure as ViPT." },
{ to: "T071", type: "uses-as-baseline", note: "OSTrack backbone beaten by 6.2 SR / 6.9 PR on VisEvent; tracking loss follows OSTrack." },
{ to: "T070", type: "uses-as-baseline", note: "MixFormer one-stream baseline beaten on COESOT (56.0 SR vs 67.1)." },
{ to: "T028", type: "uses-as-baseline", note: "SiamRPN++ two-modal baseline beaten on both datasets (33.66/60.58 SR/PR VisEvent)." },
{ to: "T026", type: "uses-as-baseline", note: "ATOM baseline beaten on both datasets (31.34/60.45 VisEvent)." },
],
concepts: ["sot", "attention", "multimodal", "appearance-features", "reid", "bounding-box", "success-plot"],
impact:
"Showed environmental attribute disentanglement plus contrastive template–search interaction lifts RGB-E tracking; per-attribute PR breakdown became a template for later RGB-E ablations.",
},
{
id: "T107",
arxiv: "2407.00738",
title: "Engineering an Efficient Object Tracker for Non-Linear Motion",
shortTitle: "DeepMoveSORT",
year: 2024,
authors: ["Momir Adzemovic", "Predrag Tadic", "Andrija Petrovic", "Mladen Nikolic"],
fileName: "2407.00738v1.pdf",
task: "multi-object",
tags: ["tracking-by-detection", "motion-model", "transformer", "learnable-filter", "bytetrack", "camera-motion-compensation", "reid", "non-linear-motion"],
difficulty: "advanced",
summary:
"DeepMoveSORT engineers a tracker for non-linear motion inside the ByteTrack framework: a new transformer end-to-end filter (TransFilter) replaces the Kalman filter for motion prediction and noise filtering, a rewritten measurement buffer plus camera-motion alignment keeps motion history clean, and three heuristics (DT-IoU decay-threshold association, HPC height/vertical-position cues, ATCM adaptive confidence modeling) join FastReID appearance in one Hungarian cost. It leads DanceTrack (63.0 HOTA), SportsMOT (78.7 HOTA) and SoccerNet (62.8 HOTA) but trails on linear-motion MOT17/20.",
problem:
"SOTA on non-linear-motion benchmarks still uses the linear Kalman filter, patched with heuristics and appearance; deep motion models (MoveSORT RNN/NODE filters, MotionTrack) predict non-linear motion but lack strong heuristics/appearance and are slow, recursive and error-accumulating — so no tracker combines learned non-linear filtering with production-grade association.",
background: ["mot", "tracking-by-detection", "motion-model", "state-space", "kalman", "data-association", "cost-matrix", "hungarian", "reid", "track-management", "dual-threshold", "mota", "hota", "idf1"],
previousWork: [
{
name: "SORT with Kalman filter",
limitation: "Linear motion model cannot follow dancers/athletes; KF predicts degenerate aspect ratios after occlusion.",
whyThisPaper: "TransFilter learns non-linear motion from data: +4.8 HOTA over KF on DanceTrack val (57.4→62.2).",
},
{
name: "MoveSORT (RNNFilter/NODEFilter + old buffer + IoU+L1 hybrid)",
limitation: "Recursive encoding is slow, accumulates error; buffer drains during occlusion; no appearance or crowded-scene heuristics.",
whyThisPaper: "Parallel TransFilter + occlusion-preserving buffer + ByteTrack + ReID + DT-IoU/HPC/ATCM: +6.9 HOTA over MoveSORT on DanceTrack test.",
},
{
name: "ByteTrack two-stage association and Hybrid-SORT TCM",
limitation: "Fixed IoU gate ignores growing uncertainty under occlusion; TCM uses constant confidence noise; neither learns motion.",
whyThisPaper: "DT-IoU decays the gate with occlusion time; ATCM scales KF confidence noise by (1−Dconf); both plug into one weighted cost matrix.",
},
],
researchGap:
"No tracker combined end-to-end learnable non-linear filtering with ByteTrack-grade cascades, appearance and occlusion-aware heuristics.",
contribution: [
"TransFilter: transformer encoder (RPE positional encoding, avg-pool, single-shot multi-step head) for prediction plus decoder (no self-attention at inference) for noise filtering; Huber loss on trajectory features.",
"Rewritten measurement buffer (Algorithm 2): freeze buffer during occlusion, purge outdated points on re-detection; affine CMC alignment of buffered points (Eq. 3); encoder caching makes occlusion inference fast.",
"DT-IoU: IoUmin(t_occluded) = max(IoUupper − IoUdecay·t_occluded, IoUlower) with optional box expansion Erate.",
"HPC: C_HPC = lambda_h·|h_hat−h_D| + lambda_y·|y_hat−y_D| using height and bottom-y only (more stable than width/x).",
"ATCM: KF track-confidence with adaptive measurement noise (sigma_conf·(1−Dconf))^2; fused cost C = weighted DT-IoU + HPC + ATCM + appearance for Hungarian matching.",
"DeepMoveSORT system (ByteTrack + filters + CMC + ReID + heuristics) with SOTA on DanceTrack/SportsMOT/SoccerNet and full ablations.",
],
method: {
pipeline: ["detect", "predict-transfilter", "cmc-align", "associate-dt-iou", "associate-hpc", "associate-atcm", "associate-appearance", "hungarian", "buffer-update", "manage-tracks"],
architecture:
"ByteTrack two-cascade framework; YOLOX detections; TransFilter (6 encoder layers, 1 decoder layer, width 256, history 10, predict 30/60 steps) or improved RNNFilter (history 30); FastReID BoT-S50 EMA embeddings (first cascade only); OpenCV GMC CMC on MOT17.",
motionModel:
"TransFilter end-to-end: RPE token embeddings → encoder → avg-pool → one-shot multi-step prediction; decoder filters observed measurements; CMC affine maps buffered points across frames.",
appearanceModel:
"FastReID BoT-S50 cosine distance vs EMA track embedding, first (high-confidence) ByteTrack cascade only.",
association:
"Single weighted cost (Eq. 6) of DT-IoU + HPC + ATCM + appearance solved by Hungarian algorithm per cascade.",
reid:
"BoT-S50 from FastReID; DanceTrack model from Hybrid-SORT (trained DanceTrack+CUHKSYSU), SportsMOT from Market1501, MOT from BoT-SORT.",
detectionDependency:
"YOLOX public/fine-tuned detectors per dataset; tracker keeps detector-swappable design (filters trained on noisy GT, not detector outputs).",
trackManagement:
"ByteTrack cascades: high-confidence association, then low-confidence rescue of unmatched tracks, then high-confidence birth; DT-IoU gate replaces fixed IoU gate.",
loss:
"Filter: Le2e = Lpredict + Lupdate with Huber (delta 0.5); ATCM confidence KF separate; no end-to-end tracker loss.",
optimization:
"AdamW (TransFilter lr 5e-5, RNNFilter 1e-3, wd 1e-4, ×0.1 every 4 epochs, 4-epoch warmup for TransFilter); GT trajectories + Gaussian noise + point dropout augmentation.",
},
equations: [
{
id: "deepmove-e2e-nll",
label: "End-to-end filter NLL objective (background)",
formula: "L_e2e(mu_i, mu_hat_i, Sigma_hat_i, mu_tilde_i, Sigma_tilde_i) = L_nll(mu_i, mu_hat_i, Sigma_hat_i) + L_nll(mu_i, mu_tilde_i, Sigma_tilde_i)",
variables: [
{ symbol: "mu_i", meaning: "ground-truth position at frame i" },
{ symbol: "mu_hat_i / Sigma_hat_i", meaning: "predicted mean / covariance" },
{ symbol: "mu_tilde_i / Sigma_tilde_i", meaning: "filtered (updated) mean / covariance" },
{ symbol: "L_nll", meaning: "negative Gaussian log-likelihood" },
],
intuition: "Train the filter to both predict the future and clean up the present observation.",
why: "Joint objective of the MoveSORT filters this work replaces with a non-probabilistic variant.",
where: "Sec. 2.1, Eq. 1 (prior work formulation).",
paperIds: ["T107"],
},
{
id: "deepmove-huber",
label: "Non-probabilistic end-to-end loss",
formula: "L_e2e(mu_i, mu_hat_i, mu_tilde_i) = L_predict(mu_i, mu_hat_i) + L_update(mu_i, mu_tilde_i), Huber delta",
variables: [
{ symbol: "mu_hat_i", meaning: "model-predicted trajectory point" },
{ symbol: "mu_tilde_i", meaning: "decoder-corrected (denoised) point" },
{ symbol: "delta", meaning: "Huber threshold (0.5)" },
],
intuition: "Penalize prediction and denoising errors linearly for outliers instead of quadratically.",
why: "Robust to detection outliers; Huber beats MSE by 0.8 HOTA (Table 10).",
where: "Sec. 3.1, Eq. 2; trajectory features absolute/relative/differences (App. A).",
params: "delta=0.5 on all datasets.",
paperIds: ["T107"],
},
{
id: "deepmove-cmc",
label: "CMC buffer alignment",
formula: "x_k^(i) = A^(i) x_k^(i-1) = A^(i) A^(i-1) ... A^(k+1) x_k",
variables: [
{ symbol: "x_k", meaning: "measurement observed at frame k" },
{ symbol: "A^(i)", meaning: "affine CMC transform from frame i−1 to i" },
{ symbol: "x_k^(i)", meaning: "measurement k expressed in frame-i coordinates" },
],
intuition: "Re-project old motion points into the current camera view before predicting.",
why: "Unpredictable camera motion otherwise corrupts the learned motion history (+1.7 HOTA on MOT17 for TransFilter).",
where: "Sec. 3.3, Eq. 3.",
simulator: "motion",
paperIds: ["T107"],
},
{
id: "deepmove-dtiou",
label: "Decay-threshold IoU gate",
formula: "IoU_min(t_occluded) = max(IoU_upper − IoU_decay * t_occluded, IoU_lower)",
variables: [
{ symbol: "t_occluded", meaning: "frames since the track was last matched" },
{ symbol: "IoU_upper", meaning: "gate for active tracks" },
{ symbol: "IoU_decay", meaning: "per-frame decay rate" },
{ symbol: "IoU_lower", meaning: "floor gate for long occlusions" },
],
intuition: "The longer a target hides, the less certain its position — so accept smaller overlaps.",
why: "Fixed gates kill re-identification after occlusion; decay generalizes standard IoU (equal bounds = standard gate).",
where: "Sec. 3.4, Eq. 4; box expansion Erate optional for fast sports motion.",
params: "Bounds/decay tuned per dataset (App. B.4).",
simulator: "iou-track",
paperIds: ["T107"],
},
{
id: "deepmove-hpc",
label: "Horizontal perspective cues cost",
formula: "C_HPC(x_hat, D) = lambda_h * |x_hat_h − D_h| + lambda_y * |x_hat_y − D_y|",
variables: [
{ symbol: "x_hat_h / D_h", meaning: "predicted vs detected box heights" },
{ symbol: "x_hat_y / D_y", meaning: "predicted vs detected bottom-y positions" },
{ symbol: "lambda_h / lambda_y", meaning: "height / vertical-position weights" },
],
intuition: "Farther people look shorter and higher in the image — match on the stable cues (height, y), ignore jittery width/x.",
why: "Beats IoU+L1 hybrid by 2.6 HOTA on DanceTrack (Table 11).",
where: "Sec. 3.4, Eq. 5.",
simulator: "assoc-cost",
paperIds: ["T107"],
},
{
id: "deepmove-cost",
label: "Fused association cost",
formula: "C = lambda_DT-IoU C_DT-IoU + lambda_HPC C_HPC + lambda_ATCM C_ATCM + lambda_Appr C_Appr",
variables: [
{ symbol: "C_DT-IoU / C_HPC / C_ATCM", meaning: "motion gate, perspective and confidence costs" },
{ symbol: "C_Appr", meaning: "cosine distance of ReID embeddings" },
{ symbol: "lambda_*", meaning: "per-cue weights balancing motion vs appearance" },
],
intuition: "One number per track–detection pair blending where it should be, what shape it is, how confident it looks, and who it looks like.",
why: "Single Hungarian input unifying all heuristics with appearance.",
where: "Sec. 3.4, Eq. 6; ATCM noise = (sigma_conf·(1−Dconf))^2 beats TCM by 1.0 HOTA.",
simulator: "assoc-cost",
paperIds: ["T107"],
},
],
datasets: ["dancetrack", "mot17", "mot20", "others"],
metrics: ["hota", "deta", "assa", "idf1", "mota"],
baselines: ["SORT", "DeepSORT", "ByteTrack", "ByteTrack-ReID", "OC-SORT", "BoT-SORT", "BoT-SORT-ReID", "Deep OC-SORT", "SparseTrack", "MotionTrack", "MoveSORT", "MixSORT", "Hybrid-SORT", "StrongSORT++", "FairMOT", "CenterTrack", "QDTrack"],
results: [
"DanceTrack test: DeepMoveSORT-TransFilter HOTA 63.0 / DetA 82.0 / AssA 48.6 / MOTA 92.6 / IDF1 65.0 vs Deep OC-SORT 61.3 / 81.6 / 45.8 / 91.8 / 61.5 (+1.7 HOTA, +3.5 IDF1) and MoveSORT 56.1 HOTA (+6.9).",
"SportsMOT test: TransFilter 77.6 and RNNFilter 78.7 HOTA vs Deep-EIoU 77.2 (+0.4 / +1.5); RNNFilter IDF1 81.7 vs 79.8.",
"SoccerNet test: RNNFilter HOTA 62.8 / AssA 60.2 vs BoT-SORT-ReID 60.8 / 56.2 (+2.0 HOTA, +4.0 AssA).",
"MOT17 test: 63.2 HOTA / 78.7 MOTA / 77.3 IDF1; MOT20: 60.6 / 73.6 / 74.1 — behind SUSHI/PIA2 (expected: overfits non-linear motion).",
"Component ablation DanceTrack val (Table 6, base 62.2 HOTA): KF 57.4 (−4.8), RNNFilter 61.2, no HPC 60.5, no DT-IoU 59.7, no ATCM 58.6, no appearance 57.5 (−4.5).",
"SportsMOT val (Table 7, base 72.6): KF 70.1 (−2.5), no appearance 66.0 (−6.6), no DT-IoU 70.7.",
"Buffer (Table 8): new algorithm +0.8 HOTA TransFilter / +1.2 RNNFilter; RPE +2.9 HOTA over standard PE (Table 9); Huber +0.8 over MSE (Table 10).",
"HPC (Table 11): 56.2 HOTA vs hybrid IoU+L1 53.7; ATCM beats TCM 54.2 vs 53.2 (Table 12); CMC +1.0–1.7 HOTA on MOT17 (Table 13).",
"TransFilter beats KF under every tracker (SORT/ByteTrack/ReID variants) on both val sets (Table 14); speed: KF 4522 vs TransFilter 160 FPS 1-step, but TransFilter 4770 vs KF 451 FPS at 30-step occlusion (Table 16).",
],
ablations: [
"Filter swap: learnable filters dominate KF on non-linear data; TransFilter best on DanceTrack, RNNFilter on SportsMOT/SoccerNet (longer history 30 helps).",
"Heuristics: ATCM matters most on crowded DanceTrack; DT-IoU (with expansion) matters most on fast SportsMOT.",
"Buffer/RPE/loss: each change isolated positive; RPE largest single filter-training gain.",
"Noise filtering (Table 15): helps DetA under synthetic noise, negligible (<0.1) with accurate YOLOX.",
"Image-feature motion (App. C.2): no benefit; real-detector training: no benefit over noisy-GT augmentation.",
],
limitations: {
authorStated: [
"High flexibility for non-linear motion causes overfitting on simpler linear datasets (MOT17/20 underperformance).",
"KF up to 30× faster than TransFilter for 1-step prediction; learned filters win only during occlusions.",
"Concluded TransFilter as default but RNNFilter still better on sports domains — no single best filter.",
],
evident: [
"Dataset-specific configs throughout (detectors, ReID, history lengths, prediction horizons, cost weights).",
"CMC applied only on MOT17; no gain elsewhere despite the machinery.",
"SportsMOT test uses a detector trained on train+val, limiting comparability.",
],
},
assumptions: [
"Tracking-by-detection with reliable boxes; appearance helps only in the high-confidence cascade.",
"Height/bottom-y stable over time (frontal-ish views); perspective cue valid.",
"Detector confidence inversely relates to measurement noise (ATCM premise).",
],
computation:
"YOLOX + FastReID BoT-S50 per dataset; filter training AdamW with GT+noise augmentation; inference FPS measured on i7-12700K + RTX 3070, batch 256 (Table 16).",
relations: [
{ to: "T005", type: "extends", note: "Extends the SORT tracking-by-detection framework with learnable filters, ByteTrack cascades and new heuristics." },
{ to: "T061", type: "builds-on", note: "DeepMoveSORT built directly inside the ByteTrack two-cascade association framework." },
{ to: "T073", type: "uses-as-baseline", note: "OC-SORT beaten on DanceTrack (55.1→63.0 HOTA) and SportsMOT (73.7→78.7)." },
{ to: "T063", type: "uses-as-baseline", note: "DanceTrack benchmark: headline +1.7 HOTA SOTA with the public detector." },
{ to: "T067", type: "uses-as-baseline", note: "StrongSORT++ listed among MOT17/20 test competitors (64.4/62.6 HOTA)." },
],
concepts: ["mot", "tracking-by-detection", "motion-model", "kalman", "data-association", "cost-matrix", "hungarian", "reid", "track-management", "dual-threshold", "hota", "mota", "idf1"],
impact:
"Reference engineering study that learnable filters must be paired with cascades, appearance and occlusion-aware gates; DT-IoU/HPC/ATCM heuristics and the occlusion-frozen buffer reused by later non-linear trackers.",
},
{
id: "T108",
arxiv: "2407.08394",
title: "Diff-Tracker: Text-to-Image Diffusion Models are Unsupervised Trackers",
shortTitle: "Diff-Tracker",
year: 2024,
authors: ["Zhengbo Zhang", "Li Xu", "Duo Peng", "Hossein Rahmani", "Jun Liu"],
fileName: "2407.08394v2.pdf",
task: "single-object",
tags: ["unsupervised-tracking", "diffusion-model", "prompt-learning", "cross-attention", "stable-diffusion", "motion-aware", "single-object"],
difficulty: "advanced",
summary:
"Diff-Tracker reframes Stable Diffusion as an unsupervised tracker: an initial prompt learner optimizes a 1024-dim prompt embedding so the UNet cross-attention map (harmonized with self-attention to encode target–background relations) fires only inside the first-frame box, then an online prompt updater fuses target-conditioned long/short-term motion (ResNet18+Conv3D encoders, cross-attention to the template, MLP fusion, residual blend head) to refresh the prompt every frame from frame 6 on. The smallest box enclosing the activated area is the output. It leads all unsupervised trackers on TrackingNet, VOT2016/2018, OTB2015 and LaSOT.",
problem:
"Supervised trackers need heavy annotation while unsupervised trackers (forward-backward CF, Siamese frame-pair classifiers) exploit neither rich frame semantics/structure nor video contextual relations; meanwhile pretrained text-to-image diffusion models demonstrably understand semantics, layout and even video context but generate images from text and have no tracking interface.",
background: ["sot", "bounding-box", "siamese", "correlation-filter", "attention", "appearance-features", "success-plot", "iou"],
previousWork: [
{
name: "UDT (forward-backward correlation filters)",
limitation: "Consistency loss on filters cannot exploit semantic/structure knowledge of frames.",
whyThisPaper: "Frozen Stable Diffusion supplies semantics for free; only the prompt is learned.",
},
{
name: "Siamese unsupervised (S2SiamFC, LUDT+, USOT, ULAST)",
limitation: "Single-frame template–search pairs with adversarial masking miss long-range video context and target–background relations.",
whyThisPaper: "Self-attention harmonization encodes pixel relations into the prompt; long+short-term motion updater keeps spatio-temporal continuity — beats ULAST-on everywhere.",
},
{
name: "Diffusion models for other vision tasks (editing, counting, pose, segmentation)",
limitation: "None applied text-to-image cross-attention activation to unsupervised tracking.",
whyThisPaper: "First bridge: prompt↔image semantic link read out as a target activation map for tracking.",
},
],
researchGap:
"No method leveraged pretrained text-to-image diffusion knowledge (cross-attention activation + self-attention relations) for unsupervised visual tracking.",
contribution: [
"Perspective: text-to-image diffusion as a prompt↔image semantic bridge whose cross-attention maps highlight prompt-related regions across images.",
"Initial prompt learner: optimize prompt embedding p1 (diffusion frozen) with MSE between harmonized map M and box-only GT map plus L_DM to stay in text-embedding space.",
"Attention harmonization: M_c'( , ) = sum_{i,j} M_c(i,j)·M_s(i,j,:,:) then M = (1−α)M_c' + αM_c (α=0.5) to inject target–background relations.",
"Online prompt updater: ResNet18+Conv3D long/short motion encoders → target-conditioned cross-attention with template appearance → MLP fusion → residual blend p_k = (1−β)H_b(p_{k−1}+l_k) + βp_{k−1} (β=0.7).",
"Unsupervised SOTA on five benchmarks (TrackingNet, VOT2016, VOT2018, OTB2015, LaSOT) with ablations isolating harmonization, updater and motion scales.",
],
method: {
pipeline: ["learn-initial-prompt", "encode-frames", "extract-motion", "condition-on-template", "fuse-motion", "blend-prompt", "activate-map", "enclose-box"],
architecture:
"Frozen Stable Diffusion UNet (initial learner) + online updater: two ResNet18+2×Conv3D motion encoders (long: current+T previous frames; short: current+previous), ResNet18 appearance extractor on first-frame template, MLP fusion head, MLP blend head; maps 64×64, prompt dim 1024.",
motionModel:
"Target-conditioned long-term (T-frame stack) and short-term (frame pair) motion via cross-attention queries from template appearance; long-term gives spatio-temporal continuity against occlusion/illumination noise.",
appearanceModel:
"Prompt embedding itself is the appearance model: optimized to activate the target region; refreshed each frame from motion-conditioned blend.",
detectionDependency: "None — first-frame box given; output is smallest axis-aligned box enclosing the activated cross-attention area.",
trackManagement: "No explicit management; prompt updated every frame from the 6th frame onward.",
loss:
"Initial: L = ||M − F1||_2^2 + L_DM (Eq. 6); online: MSE between pk-generated map and k-th frame GT box map; pseudo-labels from off-the-shelf optical flow on GOT-10k/ImageNet VID/LaSOT/YouTube-VOS.",
optimization:
"Diffusion frozen; Adam lr 5e-3 × 3 epochs (initial prompt) and 5e-4 × 35 epochs (updater); per-video initial-prompt learning at test time with training-time setup.",
},
equations: [
{
id: "difftrack-dm",
label: "Stable Diffusion denoising objective",
formula: "L_DM = E_{Ei(I), eps~N(0,1), t}[ ||eps − eps_theta(z_t, t, Et(p))||_2^2 ]",
variables: [
{ symbol: "Ei / Et", meaning: "image / text encoders" },
{ symbol: "eps", meaning: "noise added to the latent" },
{ symbol: "eps_theta", meaning: "denoising UNet" },
{ symbol: "z_t / t", meaning: "noisy latent / diffusion timestep" },
],
intuition: "Learn to remove noise conditioned on the prompt — this is what bakes semantics into the model.",
why: "Regularizer keeping the learned prompt inside diffusion-understandable embedding space (Eq. 6).",
where: "Sec. 3, Eq. 1; reused as L_DM term in Eq. 6.",
paperIds: ["T108"],
},
{
id: "difftrack-cross",
label: "Cross-attention activation map",
formula: "M_c = Softmax(Q_c K_c^T / sqrt(d))",
variables: [
{ symbol: "Q_c", meaning: "queries from the noisy image latent" },
{ symbol: "K_c", meaning: "keys from the text/prompt embedding Et(p)" },
{ symbol: "d", meaning: "projection dimension" },
],
intuition: "Each image pixel votes how semantically related it is to the prompt — the tracker readout.",
why: "A target-representing prompt lights up the target region on this map across frames.",
where: "Sec. 3, Eq. 2.",
paperIds: ["T108"],
},
{
id: "difftrack-self",
label: "Self-attention relation map",
formula: "M_s = Softmax(Q_s K_s^T / sqrt(d))",
variables: [
{ symbol: "Q_s / K_s", meaning: "queries / keys both from the image latent" },
{ symbol: "M_s", meaning: "pixel-to-pixel semantic correlation map" },
],
intuition: "Captures which pixels belong together — the target–background relationship structure.",
why: "Raw material for harmonization: propagates activation along related pixels.",
where: "Sec. 3, Eq. 3.",
paperIds: ["T108"],
},
{
id: "difftrack-harmonize",
label: "Attention harmonization",
formula: "M_c'(:,:) = sum_{i,j} M_c(i,j) * M_s(i,j,:,:); M = (1−α) M_c' + α M_c",
variables: [
{ symbol: "M_c(i,j)", meaning: "cross-attention value at (i,j)" },
{ symbol: "M_s(i,j,:,:)", meaning: "self-attention field of pixel (i,j)" },
{ symbol: "α", meaning: "balance weight (0.5)" },
{ symbol: "M", meaning: "final output map supervised against the box map" },
],
intuition: "Smear each pixel's activation over its related pixels, then mix with the original map.",
why: "Encodes target–background relations into the learned prompt; removing it drops VOT2018 EAO 0.365→0.359.",
where: "Sec. 4.2, Eqs. 4–5.",
params: "α=0.5; maps resized to common size before summation.",
paperIds: ["T108"],
},
{
id: "difftrack-initloss",
label: "Initial prompt loss",
formula: "L = ||M − F_1||_2^2 + L_DM",
variables: [
{ symbol: "F_1", meaning: "GT map activating only inside the first-frame box" },
{ symbol: "M", meaning: "harmonized output map" },
{ symbol: "p_1", meaning: "optimized initial prompt embedding" },
],
intuition: "Sculpt the prompt until the diffusion model highlights exactly the target box.",
why: "MSE teaches where; L_DM keeps the prompt speaking the diffusion model's language.",
where: "Sec. 4.2, Eq. 6; diffusion weights frozen.",
paperIds: ["T108"],
},
{
id: "difftrack-update",
label: "Residual prompt update",
formula: "l_k^L = Cross-Attn(Q_k, m_k^L), l_k^S = Cross-Attn(Q_k, m_k^S); p_k = (1−β) H_b(p_{k−1} + l_k) + β p_{k−1}",
variables: [
{ symbol: "m_k^L / m_k^S", meaning: "long / short-term global motion encodings" },
{ symbol: "Q_k", meaning: "target appearance query from first-frame template" },
{ symbol: "l_k", meaning: "MLP-fused target-conditioned motion" },
{ symbol: "H_b / β", meaning: "blend head / stability weight (0.7)" },
],
intuition: "Nudge last frame's prompt with what the target just did, while anchoring to history.",
why: "Adapts to appearance change from motion; long-term arm resists occlusion/illumination glitches; removing updater drops EAO 0.365→0.349.",
where: "Sec. 4.3, Eqs. 7–8, Fig. 2.",
params: "β=0.7; update every frame from frame 6.",
paperIds: ["T108"],
},
],
datasets: ["otb", "vot", "trackingnet", "lasot", "got10k"],
metrics: ["success-auc", "precision", "norm-precision", "eao"],
baselines: ["SiamFC", "DaSiamRPN", "SiamRPN", "SiamRPN++", "ATOM", "DiMP", "KCF", "DSST", "ECO", "S2SiamFC", "LUDT+", "USOT", "ULAST-off", "ULAST-on"],
results: [
"TrackingNet: Suc 0.675 / Pre 0.614 / NPre 0.751 vs best unsupervised ULAST-on 0.654 / 0.592 / 0.732.",
"VOT2016: EAO 0.430 / Acc 0.605 / Rob 0.206 vs ULAST-on 0.417 / 0.603 / 0.214.",
"VOT2018: EAO 0.365 / Acc 0.580 / Rob 0.273 vs ULAST-on 0.355 / 0.571 / 0.286.",
"OTB2015: Suc 0.661 / Pre 0.898 vs ULAST-on 0.648 / 0.879.",
"LaSOT: Suc 0.486 / Pre 0.472 vs ULAST-on 0.471 / 0.451.",
"Ablation VOT2018 (Table 3): full 0.365 EAO; w/o harmonization 0.359; w/o updater 0.349.",
"Motion ablation (Table 4): w/o long-term 0.355, w/o short-term 0.360, both 0.365 EAO.",
],
ablations: [
"Harmonization: +0.006 EAO over plain cross-attention MSE — relations help under deformation/occlusion/distractors.",
"Updater: +0.016 EAO — motion-conditioned refresh beats frozen initial prompt.",
"Long+short motion: each alone worse than combined; long-term guards spatio-temporal continuity.",
],
limitations: {
authorStated: [],
evident: [
"No author-stated limitations section; conclusion claims SOTA without discussing failure modes.",
"Pseudo-labels come from an off-the-shelf optical-flow model, capping supervision quality at flow accuracy.",
"Test-time cost: per-video prompt optimization (3-epoch setup) plus per-frame updater inference — no FPS reported.",
"Output box is a heuristic (smallest box enclosing activation), decoupled from the learned objective.",
"α=0.5/β=0.7 fixed; sensitivity and long-term drift of repeated residual blending unstudied.",
],
},
assumptions: [
"First-frame box given; tracker trained only on unlabeled videos (unsupervised protocol).",
"Box-interior activation is the correct readout of prompt semantics.",
"Optical-flow pseudo boxes approximate true targets well enough for training.",
],
computation:
"RTX 3090; 512×512 inputs, 64×64 maps, prompt dim 1024; Adam 5e-3 ×3 epochs (prompt) / 5e-4 ×35 epochs (updater); diffusion frozen.",
relations: [
{ to: "T009", type: "uses-as-baseline", note: "SiamFC supervised (0.571 TrackingNet) and S2SiamFC unsupervised baselines beaten; siamese pipeline discussed as prior art." },
{ to: "T028", type: "uses-as-baseline", note: "SiamRPN++ supervised baseline (0.733 TrackingNet, 0.414 VOT2018 EAO) reported for context." },
{ to: "T011", type: "uses-as-baseline", note: "ECO unsupervised-context baseline (0.561 TrackingNet, 0.375/0.280 EAO) beaten." },
{ to: "T024", type: "uses-as-baseline", note: "LaSOT evaluation (0.486 Suc) and pseudo-label training source." },
{ to: "T025", type: "uses-as-baseline", note: "GOT-10k pseudo-label training source following USOT protocol." },
],
concepts: ["sot", "attention", "appearance-features", "bounding-box", "success-plot", "siamese", "correlation-filter"],
impact:
"First unsupervised tracker built on frozen text-to-image diffusion cross-attention; prompt-plus-motion-updater pattern picked up by later diffusion-tracking work.",
}
];
