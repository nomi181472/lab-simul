import type { PaperRecord } from "../types";

/* batch-16 — T139..T149 */

export const BATCH_16: PaperRecord[] = [
{
id: "T139",
arxiv: "2608.01807",
title: "Parameter-Dynamic Adaptive Fusion and Calibration Network for RGBT Tracking",
shortTitle: "PAFCNet",
year: 2026,
authors: ["Zhaoding Ding", "Chenglong Li", "Jiandong Jin", "Keiwei Ying", "Wentao Wu"],
fileName: "2608.01807v1.pdf",
task: "multimodal-tracking",
tags: ["rgbt-tracking", "multimodal-fusion", "hypernetwork", "target-conditioned", "spatio-temporal", "transformer", "one-stream"],
difficulty: "advanced",
summary:
"PAFCNet replaces fixed-parameter RGBT fusion with a Target-Adaptive Hypernetwork (TA-HyperNet) that generates target-conditioned modulation parameters from initial plus dynamic template tokens. A Target-aware Parameter-Dynamic Fusion Module (TPDFM, inserted in ViT layers 10-12) uses those parameters to modulate the fusion network and predict token-wise RGB/TIR weights, while a Dynamic Spatio-temporal Calibration Module (DSCM) recalibrates propagated temporal tokens before use. It ranks first on RGBT234 (94.1 percent MPR, 70.1 percent MSR) and LasHeR (80.3 percent PR, 76.4 percent NPR, 63.9 percent SR at 384 resolution).",
problem:
"RGBT fusion functions use fixed parameters shared across all targets and scenes, so they cannot adapt to evolving target appearance or fluctuating modality quality; dynamic-architecture alternatives only select among predefined operations, and propagated spatio-temporal information accumulates tracking noise that no existing method calibrates.",
background: ["sot", "attention", "multimodal", "motion-model"],
previousWork: [
{
name: "Fixed-parameter fusion: quality-aware QAT/TUMFNet and cross-modal TBSI, AINet, BAT",
limitation:
"One shared fusion function (or fixed reliability weights) is applied to every target and scenario, so fusion cannot follow target-state or illumination changes.",
whyThisPaper:
"TA-HyperNet generates per-target channel-wise scaling and shifting parameters from template cues, making the same shared backbone behave target-conditionally.",
},
{
name: "Dynamic-architecture fusion: AFTER attention-based fusion router",
limitation:
"Adaptability is confined to selecting or combining predefined candidate fusion units, so fusion parameters are never directly adjusted to the target state.",
whyThisPaper:
"TPDFM keeps one fusion network but modulates its hidden features with generated parameters, outperforming a router-style DAF variant by 1.4/1.2/0.9 points PR/NPR/SR on LasHeR.",
},
{
name: "Spatio-temporal RGBT trackers: QSTNet token propagation, STTrack",
limitation:
"Historical tokens are propagated verbatim, so tracking noise accumulates in temporal representations without any calibration step.",
whyThisPaper:
"DSCM generates modality-specific calibration parameters from templates and rescales historical tokens before propagation, gaining 1.6/1.7 MPR/MSR on RGBT234 over template-update alone.",
},
],
researchGap:
"No RGBT tracker directly generates fusion parameters conditioned on the evolving target state while also calibrating propagated spatio-temporal tokens against accumulated noise.",
contribution: [
"Target-Adaptive Hypernetwork (TA-HyperNet) with a Template Representation Aggregation Module (TRAM): global content-aware plus K equals 4 query-guided local branches producing bounded channel-wise modulation parameters (H equals 256).",
"Target-aware Parameter-Dynamic Fusion Module (TPDFM): target-conditioned residual modulation of fusion features plus independent-sigmoid token-wise RGB/TIR weights, inserted into the 10th-12th ViT layers.",
"Dynamic Spatio-temporal Calibration Module (DSCM): per-modality template-conditioned scaling/shifting of historical tokens followed by independent self-attention, preventing cross-modal temporal interference.",
"State-of-the-art RGBT results with ablations isolating template update, DSCM, TPDFM, insertion layers, token counts, and fusion paradigms (SF vs DAF vs TPDFM).",
],
method: {
pipeline: ["extract-rgb-tir-tokens", "aggregate-template-representation", "generate-target-conditioned-parameters", "modulate-fusion-features", "predict-token-wise-modality-weights", "fuse-search-features", "calibrate-spatio-temporal-tokens", "propagate-and-predict"],
architecture:
"OSTrack one-stream ViT baseline initialized from DropMAE weights, taking an initial template, a dynamically updated template and one search region; TPDFM blocks in Transformer layers 10, 11, 12; DSCM runs an independent self-attention sequence per modality; OSTrack-style prediction head.",
motionModel: "Not applicable as a filter — temporal continuity comes from Np equals 64 spatio-temporal tokens propagated and calibrated across frames.",
appearanceModel:
"RGB and TIR ViT token streams projected to a 64-d interaction space, concatenated with an 8-d replicated target embedding, then target-conditionally modulated before modality weighting.",
association: "Not applicable (single-object; token-wise weight maps fused by weighted sum, argmax-style box prediction).",
detectionDependency: "None — first-frame template plus dynamically updated template and search region given.",
trackManagement:
"Dynamic template updated during tracking; 64 spatio-temporal tokens per modality initialized from template representations (Eq. 15) and recalibrated each frame before propagation.",
optimization:
"AdamW, learning rate 1e-4, weight decay 1e-4, 30 epochs on the LasHeR training set, batch size 12 at 256 resolution and 4 at 384 resolution, two NVIDIA RTX 4090 GPUs.",
loss:
"Same training objective as OSTrack (Ye et al. 2022) — classification plus box regression objective, no new loss introduced.",
},
equations: [
{
id: "pafc-hypernet",
label: "TA-HyperNet target-conditioned parameter generation",
formula: "T = [Ti; Td], T = LN(T); zg = Softmax(T Wg)^T T; A = Softmax(Q T^T / sqrt(C)), Zl = A T; u = [zg; Zl], h = phi_s(phi_f(u)); gamma_j = tanh(phi_gamma_j(h)), beta_j = tanh(phi_beta_j(h)), j in {1,2}",
variables: [
{ symbol: "Ti, Td", meaning: "initial and dynamic template tokens preserving stable identity plus recent appearance" },
{ symbol: "zg", meaning: "global content-aware target summary weighted by learned token importance" },
{ symbol: "Zl", meaning: "K equals 4 query-guided local appearance summaries" },
{ symbol: "gamma_j, beta_j", meaning: "bounded channel-wise scaling and shifting parameters for fusion (j) and calibration heads" },
{ symbol: "H = 256", meaning: "hidden modulation dimension" },
],
intuition: "Summarize what the target looks like right now (global gist plus four local details), then translate that summary into small bounded knobs that retune fusion.",
why: "Fixed fusion parameters cannot follow appearance or modality-quality drift; generated knobs make one shared network behave target-specifically.",
where: "TRAM inside TA-HyperNet, feeding both TPDFM (Eqs. 11-12) and DSCM (Eq. 16); Fig. 2.",
params: "K equals 4 queries and H equals 256; tanh bounds prevent excessive target-conditioned perturbation and stabilize optimization.",
paperIds: ["T139"],
},
{
id: "pafc-fusion",
label: "Target-modulated fusion with token-wise modality weights",
formula: "H1 = phi1(X) * (1 + gamma1) + beta1; H2 = phi2(H1) * (1 + gamma2) + beta2; W = Sigmoid(phi3(H2)) = [w_rgb, w_tir]; Sf = w_rgb * S_rgb + w_tir * S_tir",
variables: [
{ symbol: "X", meaning: "concatenated projected RGB, TIR and replicated target-embedding tokens" },
{ symbol: "gamma, beta", meaning: "TA-HyperNet target-conditioned modulation, broadcast along tokens" },
{ symbol: "w_rgb, w_tir", meaning: "independent token-wise modality weights in (0,1)" },
{ symbol: "Sf", meaning: "fused search representation" },
],
intuition: "Retune the fusion features for this target first, then let each spatial location decide how much to trust RGB versus thermal independently.",
why: "Softmax weights would force modalities to compete; independent sigmoids let both be emphasized when complementary or both suppressed when unreliable.",
where: "TPDFM in ViT layers 10-12; projected dims Cp equals 64, Dt equals 8.",
params: "Residual form (1 + gamma) preserves the base transformation while adding bounded target-conditioned adjustments.",
paperIds: ["T139"],
},
{
id: "pafc-calibration",
label: "Dynamic spatio-temporal token calibration",
formula: "(gamma_m^t, beta_m^t) = Hs([Ti_m; Td_m]); Pbar_m^t = P_m^{t-1} * (1 + gamma_m^t) + beta_m^t; X_m^t = [Ti_m; Td_m; Pbar_m^t; S_m^t]; Y_m^t = X_m^t + phi_o(Softmax((X_m^t WQ)(X_m^t WK)^T / sqrt(C)) X_m^t WV)",
variables: [
{ symbol: "P_m^{t-1}", meaning: "propagated historical tokens of modality m" },
{ symbol: "gamma_m^t, beta_m^t", meaning: "template-conditioned calibration parameters for modality m" },
{ symbol: "X_m^t", meaning: "per-modality sequence of initial template, dynamic template, calibrated history and search tokens" },
{ symbol: "Y_m^t", meaning: "self-attention updated sequence; history and search parts are split out afterwards" },
],
intuition: "Clean the remembered past with what the target looks like now, separately per modality, before letting history talk to the current frame.",
why: "Recurrent propagation accumulates tracking noise; uncalibrated history degrades temporal representations.",
where: "DSCM, applied independently to RGB and TIR streams so one modality history cannot corrupt the other.",
paperIds: ["T139"],
},
],
datasets: ["lasheR", "others"],
metrics: ["precision", "norm-precision", "success-auc"],
baselines: ["OSTrack", "TBSI", "CKD", "BAT", "TATrack", "US-Track", "AINet", "STTrack", "SUTrack", "TUMFNet", "XTrack", "VCT", "UATrack", "RAGTrack", "SCDT", "CADTrack"],
results: [
"GTOT: PAFCNet-256 95.0 percent MPR and 81.1 percent MSR (best MSR, plus 0.6 over VCT 80.5); PAFCNet-384 95.3 percent MPR and 80.2 percent MSR.",
"RGBT210: PAFCNet-256 92.4 percent PR and 67.2 percent SR (best SR, plus 0.1 over RAGTrack); PAFCNet-384 92.6 percent PR and 66.8 percent SR.",
"RGBT234: PAFCNet-256 94.1 percent MPR and 70.1 percent MSR, first on both (plus 0.3/0.6 over RAGTrack, 0.8/0.6 over UATrack); PAFCNet-384 94.0/69.5.",
"LasHeR: PAFCNet-384 80.3 percent PR, 76.4 percent NPR, 63.9 percent SR, first on all three (plus 1.8/1.7/1.3 over UATrack); PAFCNet-256 79.8/75.7/63.4.",
"Attribute gains on LasHeR concentrate in low/high/abrupt illumination for TPDFM and in similar-appearance, partial/total occlusion and motion blur for DSCM (Fig. 3).",
],
ablations: [
"Component stack on RGBT234 MPR/MSR and LasHeR PR/NPR/SR: baseline 89.4/66.5 and 71.9/68.3/57.8; plus template update 90.5/67.5 and 75.4/71.5/59.9; plus DSCM 92.1/69.2 and 77.9/73.8/61.6; full with TPDFM 94.1/70.1 and 79.8/75.7/63.4.",
"Fusion paradigms on LasHeR: static fusion 77.6/73.4/61.7 (106.3M params, 70.9G FLOPs, 50 FPS); dynamic-architecture fusion 78.4/74.5/62.5 (108.7M, 71.4G, 45 FPS); TPDFM 79.8/75.7/63.4 (111.9M, 70.9G, 49 FPS).",
"Inserting TPDFM into layers 10, then 10-11, then 10-12 raises LasHeR PR/NPR/SR from 75.4/71.5/59.9 to 77.3/73.5/61.7 to 79.0/75.1/62.9 to 79.8/75.7/63.4.",
"Spatio-temporal token count on LasHeR: 16 tokens 78.9/74.8/62.8; 32 tokens 79.3/75.4/63.1; 64 tokens 79.8/75.7/63.4; 128 tokens drops to 78.6/74.7/62.6.",
"Attention visualizations (Fig. 4): baseline dispersed and background-distracted; TPDFM concentrates on target; DSCM further compacts activation.",
],
limitations: {
authorStated: [
"Reported sensitivity: 128 spatio-temporal tokens degrade all metrics versus 64, attributed to redundant temporal information increasing modeling difficulty.",
"Reported dependence: best results require TPDFM in all three final Transformer layers; single-layer insertion leaves 2.5 points PR on the table.",
],
evident: [
"Trained only on the LasHeR training set, so GTOT/RGBT210/RGBT234 generalization rests on one training distribution.",
"Target conditioning assumes template tokens stay trustworthy; corrupted initial or dynamic templates would propagate into every fusion and calibration parameter.",
"No runtime, parameter count or limitation discussion for the 384-resolution variant beyond accuracy tables.",
],
},
assumptions: [
"RGB and TIR streams are spatially aligned; template tokens carry stable identity with less background interference than search tokens.",
"Initial plus dynamic templates suffice as the full description of current target state for parameter generation.",
],
computation:
"Trained on LasHeR train split with two NVIDIA RTX 4090 GPUs, 30 epochs, AdamW (lr 1e-4, wd 1e-4); batch 12 at 256 and 4 at 384 resolution; DropMAE-pretrained OSTrack initialization.",
relations: [
{ to: "T071", type: "builds-on", note: "Adopts OSTrack one-stream ViT baseline, weights initialization and training objective, adding TPDFM/DSCM." },
{ to: "T105", type: "uses-as-baseline", note: "Compares against XTrack RGB-X tracker on RGBT234 and LasHeR tables." },
{ to: "T057", type: "uses-as-baseline", note: "Trains on and evaluates against the LasHeR RGBT benchmark protocol (PR/NPR/SR)." },
],
concepts: ["multimodal", "attention", "memory-network"],
impact:
"Shows parameter-dynamic (hypernetwork-generated) fusion beats operation-selection routing for RGBT tracking, plus template-conditioned calibration as a fix for noisy temporal propagation.",
},
{
id: "T140",
arxiv: "2608.06773",
title: "AnyTrack: Unifying Visual Object Tracking with Any Modalities",
shortTitle: "AnyTrack",
year: 2026,
authors: ["Hao Li", "Yunzhi Zhuge", "Wenning Hao", "Pingping Zhang", "Xiaoxiong Zhang", "Dong Wang", "Huchuan Lu"],
venue: "ACM MM 2026",
fileName: "2608.06773v1.pdf",
task: "multimodal-tracking",
tags: ["any-modality", "unified-tracking", "mixture-of-experts", "multimodal-fusion", "prompt-learning", "transformer", "temporal-aggregation"],
difficulty: "advanced",
summary:
"AnyTrack is a single unified model that tracks with any combination of RGB, grayscale, depth, thermal, event, language and audio inputs, without RGB being required. A Modality-aware Interaction Module (MIM) routes each modality through top-2 of 6 appearance experts plus a shared semantic expert with cross-modal temporal aggregation, and a Context Understanding Module (CUM) grounds global-local prompts (motion, language, audio) into vision via asymmetric bidirectional attention and dynamic Gaussian pseudo-masks. One model tops RGBDT500 (79.2 percent AUC), LasHeR (77.2 percent PR), DepthTrack (65.9 percent F-score) and VisEvent (81.3 percent PR) and leads all modality-missing benchmarks.",
problem:
"Multi-modal trackers are built for fixed modality combinations with separate models per input type, so they waste parameters, cannot operate when a modality is missing or degraded, cannot transfer knowledge across datasets, and always require RGB — no single model handles arbitrary modality combinations.",
background: ["sot", "attention", "multimodal", "siamese"],
previousWork: [
{
name: "Bi-modal specialists: TBSI, BAT, ViPT, SDSTrack for RGB+T/D/E",
limitation:
"Each fixed combination needs its own architecture and weights, and depth-only or thermal-only operation collapses (reported depth-only 14.6 AUC).",
whyThisPaper:
"MIM routes any modality subset through shared experts, so depth-only, thermal-only and grayscale-only tracking work inside one model.",
},
{
name: "Architecture-shared ProTrack and fixed-set unified trackers (Un-Track, OneTracker, SUTrack, XTrack)",
limitation:
"They share backbones but retrain per task or support only a predefined modality set with RGB mandatory, failing on missing-modality and RGB-absent inputs.",
whyThisPaper:
"AnyTrack removes the RGB requirement and is evaluated over dozens of modality subsets plus dedicated missing-modality benchmarks (LasHeRmiss, DepthTrackmiss, VisEventmiss, RGBDT500miss).",
},
{
name: "Shallow reference fusion for language/audio (joint token learning, MLLM referring trackers)",
limitation:
"Simple concatenation or shallow fusion of language/audio references gives ambiguous boundaries and no spatial correspondence with visual features.",
whyThisPaper:
"CUM builds global-local prompts with asymmetric bidirectional attention, mask memory and position-adaptive Gaussian pseudo-masks for target-aware context modeling.",
},
],
researchGap:
"No tracking framework combines arbitrary modality inputs (including non-visual language/audio prompts) in one fixed-parameter model that stays robust when modalities are missing or degraded.",
contribution: [
"AnyTrack framework: first unified tracker accepting any combination of RGB, G, D, T, E, L, A in a single model with a shared HiViT-B vision encoder.",
"Modality-aware Interaction Module (MIM): noisy-gated MoE routing (6 experts, top-2) separating explicit appearance experts from a shared implicit semantic expert, with cross-modal temporal aggregation for spatio-temporal consistency.",
"Context Understanding Module (CUM): multi-modal prompt encoder plus asymmetric bidirectional attention (memory-to-prompt, prompt-to-vision) with mask memory and dynamic variance Gaussian pseudo-masks.",
"Extended benchmarks: 2,744 language descriptions, 2,744 audio clips and 1604.1K grayscale frames added to RGBDT500, DepthTrack, LasHeR and VisEvent with a two-stage MLLM-plus-human annotation pipeline.",
],
method: {
pipeline: ["patch-embed-any-modality", "concat-temporal-tokens", "shared-vision-encoding", "modality-aware-routing", "expert-fusion", "cross-modal-temporal-aggregation", "prompt-encoding", "asymmetric-bidirectional-attention", "dynamic-mask-generation", "prediction-head"],
architecture:
"HiViT-B modality-shared vision encoder (Fast-ITPN init); frozen CLIP text and WavLM audio encoders; MIM with 6 MLP appearance experts (top-2) plus one shared semantic expert and MoE balancing loss; CUM with 2 asymmetric attention layers, mask memory Q and variance network V; head outputs classification, offsets and normalized sizes.",
motionModel: "Not applicable as a filter — temporal coherence comes from temporal tokens aggregated across frames inside MIM.",
appearanceModel:
"Heterogeneous appearance decoupled into fine-grained expert subspaces while a shared semantic expert extracts modality-invariant cues; semantic-guided gated fusion with residual flow preserves original features.",
association: "Not applicable (single-object; peak classification response selected as target state).",
detectionDependency: "None — template plus search region per active modality given; motion trajectory, language and audio act as prompts, not detectors.",
trackManagement:
"Temporal tokens H and mask memory Q propagate across frames (NH equals 1); no online weight update — adaptation happens through routing and prompt conditioning.",
optimization:
"AdamW lr 1e-4 wd 1e-4, batch 16, templates 128, search 256, sampling ratio 1:1:1:1 over RGBDT500/DepthTrack/LasHeR/VisEvent train sets, rotation/translation/color-jitter augmentation, 4 NVIDIA V100 GPUs.",
loss:
"Total L equals 1.0 Lcls plus 2.0 Liou plus 5.0 L1 plus 0.01 Lbalance, with an additional 1e-6 coefficient term; Lbalance penalizes imbalanced expert importance and load across modalities.",
},
equations: [
{
id: "anytrack-routing",
label: "Noisy modality-aware expert routing",
formula: "z_m = R(P(V_m)); ztilde_m = z_m + xi * (Softplus(N(P(V_m))) + eps); g_m^(i) = Softmax(TopK(ztilde_m, k))_i, i in K_m",
variables: [
{ symbol: "V_m", meaning: "shared-encoder features of modality m" },
{ symbol: "P", meaning: "channel-wise average pooling to a compact routing descriptor" },
{ symbol: "xi ~ N(0,I)", meaning: "standard Gaussian noise for diverse expert utilization in training" },
{ symbol: "K_m", meaning: "index set of top-k (k equals 2 of 6) experts selected for modality m" },
{ symbol: "g_m^(i)", meaning: "softmax gating weight of selected expert i" },
],
intuition: "Summarize each modality, add learnable noise so different experts get exercised in training, then sparsely pick the two best specialists for that modality.",
why: "One static architecture cannot fit heterogeneous modalities; sparse adaptive selection tailors capacity per modality without activating the full pool.",
where: "MIM router; all tokens of a modality go to its selected experts only.",
params: "N equals 6 experts, top-2 routing, eps equals 1e-2 numerical stability; C equals 512 feature dim.",
paperIds: ["T140"],
},
{
id: "anytrack-fusion",
label: "Semantic-guided expert fusion",
formula: "h_m^(i) = E_i(V_m); htilde_m = E_s(V_m); G_m = F(V_m, sum_{i in K_m} g_m^(i) h_m^(i) + htilde_m); E_s(x) = M(C(M(O(x))) * GELU(M(O(x))))",
variables: [
{ symbol: "E_i", meaning: "explicit appearance expert MLPs capturing modality-specific cues" },
{ symbol: "E_s", meaning: "shared implicit semantic expert aligning modality-invariant semantics with gating" },
{ symbol: "G_m", meaning: "modality-specific enhanced visual features" },
{ symbol: "F", meaning: "residual gated fusion of original flow with expert mixture" },
],
intuition: "Let specialists describe how each modality looks while one shared expert describes what the object means, then blend both without losing the original signal.",
why: "Appearances vary wildly across modalities but object semantics are invariant; the split gives fine-grained representation plus cross-modal alignment.",
where: "MIM after routing; feeds cross-modal temporal aggregation.",
paperIds: ["T140"],
},
{
id: "anytrack-temporal",
label: "Cross-modal temporal aggregation",
formula: "Htilde_m = M(Phi(H_m, G_m, G_m)); U = sum_m G_m * (G_m tensor Htilde_m^T) over all modalities",
variables: [
{ symbol: "H_m", meaning: "temporal tokens of modality m carried from previous frames" },
{ symbol: "Phi", meaning: "multi-head cross-attention propagating enhanced features into temporal tokens" },
{ symbol: "U", meaning: "unified features weighting spatial representations by affinity with multi-modal history" },
{ symbol: "Htilde_m", meaning: "refined temporal tokens reused next frame" },
],
intuition: "Refresh each modality memory with what is seen now, then let current features vote weighted by how well they agree with the remembered past.",
why: "Scale variation, blur and similar objects break single-frame fusion; temporal affinity restores spatio-temporal consistency.",
where: "MIM output into CUM; NH equals 1 temporal token.",
paperIds: ["T140"],
},
{
id: "anytrack-cum",
label: "Asymmetric bidirectional attention with dynamic masks",
formula: "P = [Btilde; Ltilde; Atilde]; Ptilde = M(Phi(Gamma(P,P,P), Q, Q)); Utilde = Phi(U, Ptilde, Ptilde); D(x,y) propto exp(-(x-xc)^2/2sx^2 - (y-yc)^2/2sy^2), s = stilde + V(P(Utilde)); Y = T(Utilde + I(D))",
variables: [
{ symbol: "P", meaning: "global-local prompts from motion, language and audio encoders" },
{ symbol: "Q", meaning: "mask memory storing encoded target representations of past frames" },
{ symbol: "Utilde", meaning: "prompt-conditioned unified visual features" },
{ symbol: "D", meaning: "position-adaptive Gaussian pseudo-mask with learned variance offsets" },
{ symbol: "Y", meaning: "refined mask features stored back to memory" },
],
intuition: "Let history inform the prompts and the prompts inform vision, then draw a flexible spotlight (not a rigid box) around the target that adapts to its size.",
why: "Shallow concatenation cannot build spatial correspondence; bidirectional flow plus adaptive masks sharpen foreground-background discrimination.",
where: "CUM, iterated over L equals 2 layers; dense spatial maps as positional encodings.",
params: "NB equals 2 motion tokens, NL equals 1, NA equals 1; binary masks lose 1.2 DP, static variance loses 1.0 AUC.",
paperIds: ["T140"],
},
],
datasets: ["lasheR", "others"],
metrics: ["precision", "norm-precision", "success-auc", "recall"],
baselines: ["MixFormer", "DeT_ATOM", "DeT_DiMP", "ViPT", "SDSTrack", "Un-Track", "TBSI", "BAT", "DCEvo+OSTrack", "RDTTrack", "ProTrack", "OneTracker", "SUTrack", "SMSTracker", "XTrack", "UM-ODTrack", "UniSOT", "MamTrack", "AlignTrack", "MCITrack", "SeqTrackv2", "IPT", "STTrack"],
results: [
"RGBDT500: RGB-only 83.7 percent DP (over MixFormer); grayscale 72.9 percent AUC; RGB+D 77.8 and RGB+T 78.0 AUC (over Un-Track); all-modality RGB+D+T+L+A 79.2 percent AUC and 86.1 percent DP (over RDTTrack); missing split 67.7/68.5.",
"LasHeR: RGB+T 75.5 percent PR, 71.1 NPR, 59.7 SR; with language plus audio 77.2/72.9/61.1 (over SMSTracker, XTrack, TBSI, BAT); RGB-only 69.4, grayscale-only 63.1, thermal-only 60.0 PR.",
"DepthTrack: RGB+D+L 65.9 percent F-score (plus 2.3 over SMSTracker); G+D 56.5 F-score versus 28.0 depth-only (plus 28.5); RGB+D plus 37.2 over depth-only.",
"VisEvent: RGB+E+L 63.9 percent SR and 81.3 percent PR (over MamTrack, XTrack); G+E+L 61.2/79.7 without color.",
"Missing-modality: LasHeRmiss 70.6 PR, DepthTrackmiss 54.9 F-score, VisEventmiss 54.0 SR — all best, e.g. plus 12.3 PR over SUTrack on LasHeRmiss.",
],
ablations: [
"Key modules (DP RGBDT500 / PR LasHeR, DepthTrack, VisEvent): full 86.1/77.2/66.0/81.3; without MIM 83.8/75.6/63.4/78.5 (minus 2.3); without CUM 82.2/74.3/62.9/77.9 (minus 3.3).",
"Interaction strategies on RGBDT500 AUC/DP: GAT 76.0/81.5 (101.6M, 62.4G, 23 FPS); MLP concat 75.3/80.5; Mamba 74.6/79.4; MIM 79.2/86.1 (90.6M, 61.9G, 26 FPS).",
"MIM parts: without noisy gating 78.1 AUC; without semantic expert 84.2 DP; without balancing loss 78.5/85.4.",
"CUM parts: without mask memory minus 1.7 AUC; binary hard masks instead of D minus 1.2 DP; static variance instead of V minus 1.0 AUC.",
"Attention maps (Fig. 4): single modalities dilute; each added modality progressively concentrates attention on the target.",
],
limitations: {
authorStated: [
"No explicit limitations section; reported behavior: missing-modality performance still falls far below full-modality (RGBDT500miss 68.5 vs 86.1 DP; depth-only 14.6 AUC), and blindly adding modalities can degrade accuracy without the MIM design.",
],
evident: [
"Depends on a two-stage MLLM-plus-human annotation pipeline for 2,744 language/audio descriptions — costly to replicate for new domains.",
"Text and audio encoders frozen, so prompt representations cannot co-adapt with tracking; audio is synthetic TTS at 16 kHz, not real-world sound.",
"Single fixed sampling ratio and V100 training setup; no latency/FLOPs breakdown per modality subset for deployment planning.",
],
},
assumptions: [
"Grayscale-by-luminance faithfully simulates monochrome/low-light conditions; TTS audio from templated sentences stands in for genuine audio cues.",
"Temporal token of size 1 and two-layer bidirectional attention suffice for cross-frame consistency across all modality subsets.",
],
computation:
"PyTorch on 4 NVIDIA V100 GPUs; HiViT-B backbone, C equals 512; batch 16 with 1:1:1:1 dataset sampling; AdamW lr 1e-4 wd 1e-4; loss weights 1/2/5/0.01.",
relations: [
{ to: "T083", type: "uses-as-baseline", note: "Evaluates against ViPT prompt multi-modal tracker on all four benchmarks." },
{ to: "T097", type: "improves", note: "Removes Un-Track fixed-modality-set plus RGB-mandatory restriction; any-subset tracking in one model." },
{ to: "T105", type: "uses-as-baseline", note: "Compares against XTrack RGB-X unified tracker on LasHeR, DepthTrack and VisEvent." },
{ to: "T071", type: "uses-as-baseline", note: "Compares against OSTrack-based DCEvo+OSTrack RGB+T baseline." },
{ to: "T070", type: "uses-as-baseline", note: "Compares against MixFormer RGB baseline on RGBDT500." },
{ to: "T057", type: "uses-as-baseline", note: "Evaluates on the LasHeR benchmark including new missing-modality splits." },
],
concepts: ["multimodal", "attention", "memory-network", "learned-association"],
impact:
"First any-modality tracking model plus extended grayscale/language/audio benchmarks, pushing later work toward RGB-optional unified trackers with learned routing instead of per-combination models.",
},
{
id: "T141",
arxiv: "2608.08016",
title: "EgoTrack3D: A Modular Framework for Egocentric 3D Object Tracking",
shortTitle: "EgoTrack3D",
year: 2026,
authors: ["Jan Kulik", "Bjarni Dagur Thor Karason", "Yung-Hsu Yang", "Boyang Sun", "Marc Pollefeys", "Xi Wang"],
fileName: "2608.08016v1.pdf",
task: "3d-tracking",
tags: ["egocentric", "3d-tracking", "scene-reconstruction", "multi-object", "modular-framework", "point-cloud", "motion-scoring"],
difficulty: "intermediate",
summary:
"EgoTrack3D builds persistent full-scene 3D object tracks from egocentric RGB video by lifting 2D masks into a shared world frame and associating them with a point-motion plus voxel-merge mechanism. A CoTracker3 grid (60 by 60 over 10 frames) classifies static versus moving objects so static point clouds aggregate while dynamic ones refresh; Hungarian matching over 3D IoU, Chamfer and MASA appearance cost plus voxel-overlap merging yields 63.01 percent average PCL on ADT (plus 11 percent relative over the strongest baseline). A sparse variant swaps dense lifting for BoxerNet 3D boxes with interaction-guided dynamic anchoring for depth-free deployment.",
problem:
"Egocentric video has rapid viewpoint change, partial occlusion and objects entering/leaving, while existing 3D trackers cover only predefined interacted objects, static scenes, or compact center representations — nothing maintains persistent full-3D tracks of all observed static and dynamic objects, and dense metric depth is unreliable in the wild.",
background: ["mot", "3d-tracking", "detection", "iou"],
previousWork: [
{
name: "Egocentric 3D trackers: IT3DEgo (predefined interacted objects), OSNOM Lift-Match-Keep, EgoSeg3D (2D-track refinement)",
limitation:
"Restricted to enrolled or task-relevant objects, or to compact locations rather than full object geometry, with fragile IDs under clutter and occlusion.",
whyThisPaper:
"Tracks every visible object with full point clouds plus oriented boxes, aggregating static geometry and refreshing dynamic objects via motion scoring.",
},
{
name: "Static 3D lifting: Boxer open-vocabulary 2D-to-3D boxes",
limitation:
"Designed for static object fusion, not persistent dynamic identities — it outputs a post-processed scene set with no per-frame tracks.",
whyThisPaper:
"Dense variant lifts per-frame masks into tracked instances; sparse variant reuses BoxerNet as an interchangeable lifting module inside a real tracker.",
},
{
name: "2D video segmentation: SAM family, DEVA, Track Anything, SAM-PT",
limitation:
"Purely 2D identity breaks under long occlusion, large viewpoint change and hand-object interaction without a 3D frame to anchor it.",
whyThisPaper:
"Lifts masks to world coordinates with poses and depth so out-of-sight objects persist in 3D and re-associate geometrically.",
},
],
researchGap:
"No modular framework maintains temporally consistent full-3D tracks of all static and dynamic objects from egocentric video across both dense-depth and sparse-geometry input regimes.",
contribution: [
"Modular dense/sparse pipeline: per-frame segmentation plus depth, CoTracker3 motion scoring, MASA appearance embeddings, 3D lifting, Hungarian association, voxel-based track merging.",
"Point-based 3D motion scoring (Eqs. 1-2) with a 60 by 60 grid over 10 frames; static tracks aggregate point clouds, dynamic tracks refresh (Eq. 5).",
"Voxel-overlap merging heuristic (Eq. 6) pruning duplicate tracks, accounting for most of the PCL gain (plus 83.38 percent average).",
"Sparse-input variant: BoxerNet 3D-box lifting from model masks plus sparse points, Hands23 interaction-guided dynamics and SAM2 short-term anchoring for noisy hand-held objects.",
],
method: {
pipeline: ["segment-masks", "estimate-depth", "score-motion-cotracker3", "extract-appearance-embeddings", "lift-masks-to-3d", "associate-hungarian", "merge-voxel-tracks"],
architecture:
"Dense: ground-truth or predicted masks plus metric depth lifted via intrinsics and Aria poses, PCA-oriented boxes; MASA-pooled appearance embeddings; sparse: CropFormer masks to 2D prompts, BoxerNet oriented boxes, Hands23 interaction masks, SAM2 K-frame dynamic anchors.",
motionModel: "No Kalman propagation — CoTracker3 3D point trajectories decide static aggregation versus dynamic refresh; unmatched tracks propagate to next frame.",
appearanceModel:
"MASA backbone features pooled in each 2D box into object embedding q; cosine similarity term keeps visual consistency across viewpoint change.",
association: "Hungarian assignment over combined 3D-IoU, Chamfer and appearance cost (Eq. 4); dynamic-mask hard association precedes geometric matching in the sparse variant.",
detectionDependency: "Full dependence on external masks (ground truth in dense eval; CropFormer in sparse) plus poses and depth or sparse points.",
trackManagement:
"Unmatched detections birth tracks; unmatched tracks propagate; voxel-overlap merging prunes duplicates; sparse static tracks use confidence-weighted interpolation, dynamic tracks take the current prediction.",
loss:
"Training-free tracking pipeline — no learned tracker weights and no tracking loss; relies on pretrained segmentation, depth, CoTracker3, MASA, BoxerNet and Hands23 modules.",
},
equations: [
{
id: "egotrack-motion",
label: "Point-based 3D motion and direction scoring",
formula: "m_n,t = sum_{i=2..Dt} ||Dp_n,t,i||_2; d_n,t = (1/(Dt-1)) sum_{i=2..Dt} Dp_n,t,i / ||Dp_n,t,i||_2; m^M_j = mean_{p in Mj} m_n,t; d^M_j = mean_{p in Mj} d_n,t; moving if m^M_j >= 0.1 and |d^M_j| >= 0.1",
variables: [
{ symbol: "Dp_n,t,i", meaning: "3D displacement of grid point n between consecutive frames" },
{ symbol: "m_n,t", meaning: "accumulated motion magnitude of point n over the window" },
{ symbol: "d_n,t", meaning: "mean motion direction (coherence) of point n" },
{ symbol: "m^M_j, d^M_j", meaning: "mask-averaged motion and direction scores of object j" },
{ symbol: "N=60, Dt=10", meaning: "60 by 60 point grid tracked over 10 frames, points within 1 m" },
],
intuition: "Watch a grid of 3D points for ten frames: far-travelled points that agree on direction mean real object motion; jittery disagreement means noise.",
why: "Static geometry should accumulate while moving objects must refresh; without the split, aggregation smears motion and refresh discards useful static history.",
where: "Per-frame motion detection (Sec. 3.1.1); lookahead W equals 10 selected on ADT trajectories.",
params: "Thresholds tm equals td equals 0.1; reported 48.96 percent precision and 94.69 percent recall for 0.10 m center motion.",
simulator: "motion",
paperIds: ["T141"],
},
{
id: "egotrack-lifting",
label: "Monocular mask lifting to world coordinates",
formula: "[X Y Z 1]^T = T_cw [d K^-1 u; 1]; o = BOX_PCA(P) after voxel downsampling",
variables: [
{ symbol: "u", meaning: "2D image point in homogeneous coordinates" },
{ symbol: "d", meaning: "metric depth at that pixel" },
{ symbol: "K, T_cw", meaning: "camera intrinsics and camera-to-world pose" },
{ symbol: "P", meaning: "object 3D point cloud gathered from its mask" },
{ symbol: "o", meaning: "PCA-fitted oriented 3D bounding box" },
],
intuition: "Unproject each masked pixel through depth and camera pose into the shared world, then shrink-wrap the resulting cloud with an oriented box.",
why: "A common 3D frame lets objects persist through occlusion and viewpoint change where 2D overlap is meaningless.",
where: "3D detection via lifting (Sec. 3.1.3); dense uses per-pixel depth, sparse uses BoxerNet boxes instead.",
paperIds: ["T141"],
},
{
id: "egotrack-association",
label: "Geometric-appearance association with voxel merging",
formula: "c_ij = l_iou (1 - IoU(i,j)) + l_chamfer Chamfer(i,j) + l_feat (1 - s(i,j)); Hungarian assignment; P_i^t = P_i^{t-1} union P_j^t if static else P_j^t; q = a q_prev + (1-a) q_new; merge if (|V_i| + |V_j| - |V_union|) / (|V_i| + |V_j|) > tau",
variables: [
{ symbol: "c_ij", meaning: "matching cost between existing track i and detection j" },
{ symbol: "IoU, Chamfer, s", meaning: "3D box overlap, shape-plus-distance similarity, MASA embedding similarity" },
{ symbol: "P_i^t", meaning: "accumulated (static) or refreshed (dynamic) point cloud" },
{ symbol: "V(P)", meaning: "occupied 3D voxels of a point cloud used for duplicate merging" },
],
intuition: "Match tracks to detections by overlap, shape and looks together; then fuse point clouds of doubles that occupy the same voxels.",
why: "Any single cue fails egocentrically — IoU vanishes under fast motion, geometry drifts, appearance blurs — so the sum plus merge keeps identities compact.",
where: "Tracking in 3D (Sec. 3.1.4) and track merging (Sec. 3.1.5); sparse variant merges on box plus appearance similarity.",
simulator: "assoc-cost",
paperIds: ["T141"],
},
],
datasets: ["ego4d", "others"],
metrics: ["3d-error"],
baselines: ["IT3DEgo", "OSNOM", "EgoSeg3D", "Boxer", "EgoSeg3D with ground-truth IDs"],
results: [
"Dense ADT mean PCL (0.1/0.2/0.3/0.6/0.9/1.2 m, avg): EgoTrack3D merge 56.28/60.14/62.56/65.88/66.48/66.72 avg 63.01 — best overall, about 11 percent relative over strongest baseline Boxer 56.53 avg.",
"Without merging: 30.42/32.91/34.25/35.96/36.21/36.38 avg 34.36; merging improves average PCL by 83.38 percent. Baselines: IT3DEgo 36.55, OSNOM 21.41, EgoSeg3D (alpha 0) 10.44, EgoSeg3D (alpha 1e4) 23.33.",
"Meal-132 scene: 211 visible ground-truth objects; no-merge predicts 267 with 96 duplicates; merge predicts 182 with 28 duplicates (rare overmerges of nearby distinct objects).",
"Motion module: 48.96 percent precision, 94.69 percent recall for 0.10 m center motion over 10 frames.",
"Sparse ADT (3D IoU 0.05-0.50): EgoTrack3D-Sparse AP 24.49 and F1 40.79 versus Boxer 14.12 and 29.88; removing dynamic anchoring 24.11/39.28, removing interaction guidance 23.13/40.43.",
"HD-EPIC qualitative: preserves notepad and pot tracks across large viewpoint changes where interaction guidance removal loses them.",
],
ablations: [
"Merging on/off over time (Fig. 2): PCL0.3 decays for all methods as errors accumulate; disabling merging causes sharper decay, confirming its temporal-consistency role.",
"Threshold sweep (Fig. 4): larger PCL radii raise scores with diminishing returns beyond 0.6 m.",
"Sparse ablations show modest aggregate effect because scene AP/F1 are dominated by static objects while the components target manipulated-object failures.",
"Baseline adaptations: IT3DEgo enrolled with max area-perimeter-ratio boxes and 2D DINOv2 only; EgoSeg3D alpha 1e4 isolates the object-ID cue effect.",
],
limitations: {
authorStated: [
"Inherits perception-module errors; dense setting needs accurate masks, poses and metric depth, and monocular depth replacement stays unstable under egocentric motion.",
"Sparse variant remains sensitive to BoxerNet errors on hand-held or moving objects with unstable 3D boxes.",
"Association relies strongly on geometry, so re-identification fails after large estimated-location changes with weak visual evidence; misclassified-static objects lose useful history.",
"Future work: confidence-aware track management, temporally aware or interaction-based re-identification, motion-aware 3D box estimation.",
],
evident: [
"Dense numbers use ground-truth masks, depth and poses, so they isolate tracking from perception and overstate end-to-end real-world accuracy.",
"PCL is a custom OSNOM metric penalizing ID switches and duplicates jointly; not comparable with MOT HOTA/MOTA leaderboards.",
"Quantitative evaluation covers eight ADT sequences plus qualitative HD-EPIC only; no runtime or efficiency analysis reported.",
],
},
assumptions: [
"Accurate camera poses and intrinsics available (Project Aria poses; fisheye-to-pinhole conversion done).",
"Only ground-truth objects visible up to time t count in PCL; predicted-to-truth correspondence for F1/AP assumes fixed confidence thresholding.",
],
computation:
"No training compute (training-free pipeline of pretrained modules); evaluated on eight ADT sequences with Aria data plus qualitative HD-EPIC runs.",
relations: [],
concepts: ["3d-tracking", "data-association", "hungarian", "appearance-features"],
impact:
"Establishes full-scene persistent 3D tracking (not just interacted objects) for egocentric video with a swappable dense/sparse lifting design that later scene-graph work can build on.",
},
{
id: "T142",
arxiv: "2608.09575",
title: "MSP-Net: Manifold-Guided Spectral Prompt Network for Hyperspectral Object Tracking",
shortTitle: "MSP-Net",
year: 2026,
authors: ["Juliu Li", "Hanlin Qin", "Shuowen Yang", "Jingjing Li", "Yuedong Tan", "Shuai Yuan", "Huixin Zhou"],
fileName: "2608.09575v1.pdf",
task: "multimodal-tracking",
tags: ["hyperspectral-tracking", "manifold-learning", "graph-routing", "prompt-learning", "mixture-of-experts", "temporal-memory", "transformer"],
difficulty: "advanced",
summary:
"MSP-Net treats hyperspectral bands as nodes on a learnable nonlinear spectral manifold instead of fixed ordered groups. Graph-Driven Manifold Spectral Routing (GMSR) builds a Gaussian-kernel adjacency with symmetric normalization, propagates with a GCN, and routes bands via Gumbel-Softmax training plus soft inference assignment; Dual-Stream Spectral-Conditioned Prompt Modulation (DSCPM) turns grouped spectral statistics plus template appearance into dynamic prompts injected with cross-attention and cosine gating; Decoupled Spectral Condition Evolution (DSCE) refreshes conditions at layers 3/6/9 without parameter updates; and TMMA post-processing stabilizes boxes. It reaches 0.8072 AUC and 0.9756 precision on HOT2020 at 34.71 FPS.",
problem:
"Hyperspectral trackers either keep full spectra (redundant, expensive) or compress bands with fixed, sequential or weighted grouping that destroys inter-band manifold topology and ties models to specific sensors; static prompts then go stale as illumination, deformation and occlusion evolve spectral responses in long sequences.",
background: ["sot", "siamese", "attention"],
previousWork: [
{
name: "Full-spectrum perception: CHP, HHTrack, E2E-MPT material unmixing",
limitation:
"Preserves spectra but pays high redundancy and compute cost, impractical for real-time inference.",
whyThisPaper:
"Keeps a dimensionality-reduction design but routes bands adaptively on the manifold, holding 73.01 GFLOPs at 34.71 FPS — fastest in Table I.",
},
{
name: "Fixed grouping reduction: SEE-Net, SiamBAG, band-select COALT, HyA-T, TBRNet",
limitation:
"Rigid wavelength-order partitions disrupt spectral continuity and topology and cannot adapt group contributions as target/scene change.",
whyThisPaper:
"GMSR groups semantically correlated (even non-adjacent) bands per scene via graph routing, and MoE weights score each group contribution online.",
},
{
name: "Static prompts and fixed template updates: SP-HST, PHTrack, ProFiT, ViPT-style modality prompts",
limitation:
"Prompts frozen or shallowly fused go stale under appearance drift; static scale penalties cannot balance deformation adaptation against drift suppression.",
whyThisPaper:
"DSCPM generates target-aware prompts per layer from live spectral-target conditions evolved by DSCE, with cosine gating blocking background amplification.",
},
],
researchGap:
"No hyperspectral tracker jointly models nonlinear inter-band manifold structure, per-layer target-conditioned prompting, parameter-free condition evolution, and morphology-aware temporal stabilization in one framework.",
contribution: [
"Graph-Driven Manifold Spectral Routing (GMSR): dynamic Gaussian adjacency, symmetric normalization, GCN propagation, Gumbel-hard training with soft inference routing into decoupled group features.",
"Dual-Stream Spectral-Conditioned Prompt Modulation (DSCPM): grouped spectral statistics plus template appearance plus MoE weights fused into a 97-d condition driving cross-attention prompts with cosine-similarity gating.",
"Decoupled Spectral Condition Evolution (DSCE): inference-time condition refresh from intermediate template tokens at backbone layers 3, 6 and 9 with zero new parameters.",
"Temporal Morphology Memory-Adaptive (TMMA) post-processing: history-constrained box calibration using spatial displacement and morphological variation cues.",
],
method: {
pipeline: ["pool-band-nodes", "build-manifold-graph", "propagate-gcn", "route-band-groups", "extract-group-statistics", "build-spectral-target-condition", "generate-dynamic-prompts", "modulate-tokens", "evolve-conditions", "stabilize-predictions"],
architecture:
"ViT backbone on routed group tokens; GMSR with 4 by 4 node sampling; DSCPM prompt injection at successive Transformer layers with 12 prompts; DSCE refresh at layers {3,6,9}; TMMA as inference-only post-processing; search 256, template 128.",
motionModel: "Not applicable as a filter — temporal adaptation comes from DSCE condition evolution plus TMMA history-constrained box calibration.",
appearanceModel:
"Joint spectral-manifold context (group means plus std) with explicit template geometry/texture (mean plus max responses); cross-attention preserves spatial structure while adding spectral discriminability.",
association: "Not applicable (single-object; head prediction refined by TMMA overlap trajectories).",
detectionDependency: "None — first-frame template plus search region given; search-region routing reused for the template branch to keep spectra aligned.",
trackManagement:
"No online fine-tuning or memory network; DSCE recalibrates conditions from deep template tokens and TMMA penalizes background drift (mu_r 0.04, mu_a 0.20, eta_base 0.99).",
optimization:
"PyTorch, AdamW lr 1e-4 wd 1e-4, batch 32, 30 epochs; trained on HOT2024 train only (IMEC25 trained separately); Intel i7-14700 with two RTX 3090 GPUs.",
},
equations: [
{
id: "msp-adjacency",
label: "Dynamic Gaussian manifold adjacency",
formula: "V = AdaptiveAvgPool2d(X); A_ij = exp(-||Vi - Vj||^2 / (2 zeta(sigma)^2 + eps)); Ahat = D^{-1/2} A D^{-1/2}, D_ii = sum_j A_ij; H_new = GELU(Ahat V W_gcn)",
variables: [
{ symbol: "V", meaning: "spectral-band node features after spatial pooling to d dims" },
{ symbol: "sigma", meaning: "learnable kernel bandwidth via Softplus zeta" },
{ symbol: "Ahat", meaning: "symmetrically normalized adjacency stabilizing high/low-degree message passing" },
{ symbol: "H_new", meaning: "manifold-propagated node features" },
],
intuition: "Treat each band as a node, connect bands that look alike with Gaussian weights, normalize so popular bands do not shout down rare ones, then let neighbors share information.",
why: "Fixed wavelength grouping breaks intrinsic band relations; graph propagation aggregates semantically related bands instead.",
where: "GMSR graph construction; W_gcn updated by gradient descent (Eq. 5).",
params: "4 by 4 node sampling optimal; 2 by 2 underrepresents topology (minus 1.25 AUC), 8 by 8 overcouples (minus 1.45 AUC).",
paperIds: ["T142"],
},
{
id: "msp-routing",
label: "Gumbel-hard training with soft inference routing",
formula: "N = Gumbel-Softmax(L, tau) if training else Softmax(L); F_group = (1/C) sum_c N_{i,c} H_new; W_moe = Softmax(F_group W_proj)",
variables: [
{ symbol: "L", meaning: "logits projecting manifold features into grouping space" },
{ symbol: "N", meaning: "band-to-group assignment matrix" },
{ symbol: "F_group", meaning: "representative feature per spectral group" },
{ symbol: "W_moe", meaning: "expert-mixture weights scoring each group contribution" },
],
intuition: "Commit to crisp band groups while learning, but blend softly when deployed so group boundaries do not jitter on new scenes.",
why: "Hard inference assignments are oversensitive; soft assignment stabilizes routing without extra parameters.",
where: "GMSR routing; same search-region routing applied to template branch.",
params: "Soft-instead-of-hard routing loses 1.14 AUC and 2.04 DP; removing manifold convolution loses 2.66/3.26.",
paperIds: ["T142"],
},
{
id: "msp-prompt",
label: "Dual-stream conditioned prompt modulation with cosine gating",
formula: "C_cond = Concat(C_group, C_template, W_moe); X_attn = Softmax((X_in Wq)(P Wk)^T / sqrt(dh)) (P Wv); M = sigmoid(gamma CosSim(X_in, P_center) + beta); X_out = X_in + alpha (M * X_attn)",
variables: [
{ symbol: "C_group", meaning: "group means plus std capturing low-frequency layout and spectral volatility" },
{ symbol: "C_template", meaning: "template mean plus max responses encoding target geometry/texture" },
{ symbol: "P", meaning: "dynamic prompt generated from C_cond as cross-attention key/value" },
{ symbol: "M", meaning: "spatial gate blocking prompt injection into background regions" },
],
intuition: "Write a 97-number memo describing the spectrum plus the target, turn it into a prompt, let image tokens ask it questions — but only where the memo actually matches the image.",
why: "Static prompts amplify background noise under drift; gating confines modulation to target-relevant tokens.",
where: "DSCPM at successive Transformer layers over joint template-search tokens with residual connection.",
params: "No prompts loses 1.59 AUC; no cosine gate loses 2.52 AUC and 2.31 DP.",
paperIds: ["T142"],
},
{
id: "msp-evolution",
label: "Decoupled spectral condition evolution",
formula: "Z^{(l)} = T[:, 0:Iz]; C_cond^{(l)} = X_ext(F_group, W_moe, Z^{(l)}) if l in {3,6,9} (inference) else C_cond^{(l-1)}; P^{(l)} = G_prompt(C_cond^{(l)})",
variables: [
{ symbol: "Z^{(l)}", meaning: "deep template representation split from the joint sequence at layer l" },
{ symbol: "Omega = {3,6,9}", meaning: "update layer indices balancing efficiency and refresh rate" },
{ symbol: "X_ext", meaning: "dual-stream condition extractor reusing DSCPM statistics" },
{ symbol: "P^{(l)}", meaning: "layer-specific visual prompt from the latest condition" },
],
intuition: "Mid-network, peek at how the target looks now and rewrite the memo — no retraining, just a refresh at three checkpoints.",
why: "First-frame spectral priors decay under illumination and deformation change; periodic refresh closes the loop without parameter updates.",
where: "DSCE; training uses the initial static condition, inference refreshes at layers 3, 6, 9.",
params: "Adding DSCE to GMSR+DSCPM raises AUC 0.7980 to 0.8028; full model reaches 0.8072.",
paperIds: ["T142"],
},
],
datasets: ["others"],
metrics: ["success-auc", "precision"],
baselines: ["BAE-Net", "SEE-Net", "SiamBAG", "SSTrack", "SENSE", "MMF-Net", "SPIRIT", "PHTrack", "DASSP", "HOT-MoE", "HyMamba", "ProFiT", "TIPSST", "Trans-DAT", "DRSST-Net", "UBSTrack", "VIPT", "CSSTrack", "SiamTU", "SPTrack"],
results: [
"HOT2020: 0.8072 AUC and 0.9756 DP at 20 px (over ProFit 0.7580/0.9710, plus 4.92/0.46); best in 9 of 11 attributes, second in low resolution, third in illumination variation.",
"HOT2023: VIS 0.7576/0.9138 (best AUC, DP 0.0012 below ProFit); NIR 0.8192/0.9666 (plus 6.52/0.46); RedNIR 0.6951/0.8339 (plus 8.21/7.89 over ProFit).",
"HOT2024: NIR 0.7954/0.9428 (strong); VIS 0.4819/0.6222 and RedNIR 0.5044/0.6461 trail UBSTrack/HOT-MoE — limited spectral dimensionality weakens the routing advantage.",
"IMEC25 cross-domain: 0.7381 AUC (below ProFit 0.7540) with best DP 0.9725 (over 0.9670); best in DEF/IV/MB/OCC/SV attributes.",
"Efficiency: 34.71 FPS fastest in Table I at 73.01 GFLOPs, trading slightly higher FLOPs than Trans-DAT/HOT-MoE for accuracy.",
],
ablations: [
"Component buildup on HOT2020 AUC/DP: baseline 0.7247/0.9348; plus GMSR 0.7982/0.9684 (plus 7.35/3.36); plus DSCPM alone 0.7911/0.9557; GMSR plus DSCPM without DSCE 0.7980/0.9606 (no complementarity); plus DSCE 0.8028/0.9728; full plus TMMA 0.8072/0.9752 (plus 8.25/4.04 total).",
"GMSR structure: no manifold convolution minus 2.66/3.26; soft routing minus 1.14/2.04; node grids 2 by 2 minus 1.25/1.18 and 8 by 8 minus 1.45/1.16 versus 4 by 4.",
"DSCPM structure: no dynamic prompts minus 1.59/0.98; no cosine gate minus 2.52/2.31; feature maps concentrate on target after modulation (Fig. 10).",
"TMMA: removing it costs full-model 0.70/0.71 and Model-5 0.36/0.76; it helps most on high-quality predictions and can slightly jitter weak baselines.",
],
limitations: {
authorStated: [
"Failure analysis: most HOT2024 VIS/RedNIR failures come from unseen backgrounds, materials and sensor spectral-response shifts that deform manifold topology and weaken routing.",
"Under such out-of-distribution conditions the tracker is vulnerable to deceptive clutter, abrupt appearance change and severe drift; zero-shot cross-domain generalization stays future work.",
],
evident: [
"Trained exclusively on HOT2024 (IMEC25 separate), so all cross-benchmark claims are zero-shot transfers with the failure modes above.",
"TMMA is inference-only post-processing tuned with fixed penalties (0.04/0.20/0.99); no learned temporal memory competes with it in ablations.",
"Comparisons span many venues and years without re-implementation, so baseline numbers are quoted as reported under each benchmark protocol.",
],
},
assumptions: [
"Hyperspectral bands lie on a learnable nonlinear manifold capturable by pairwise Gaussian similarity of pooled node features.",
"Search-region routing transfers to the template branch, i.e. both branches share the same spectral grouping.",
],
computation:
"Intel i7-14700 CPU plus two NVIDIA RTX 3090 GPUs; AdamW (1e-4, 1e-4), batch 32, 30 epochs; template 128, search 256; 34.71 FPS inference at 73.01 GFLOPs.",
relations: [
{ to: "T071", type: "conceptual-successor", note: "Continues the OSTrack one-stream transformer tracking line with a ViT backbone." },
{ to: "T083", type: "uses-as-baseline", note: "Compares against ViPT visual-prompt tracking on the HOT2023 modalities." },
{ to: "T009", type: "conceptual-successor", note: "Builds on the Siamese template-search tracking lineage cited in related work." },
],
concepts: ["attention", "appearance-features", "memory-network"],
impact:
"Moves hyperspectral tracking from fixed band-group compression to manifold-adaptive routing with evolving prompts, setting the HOT-series state of the art while exposing sensor-shift generalization as the next barrier.",
},
{
id: "T143",
arxiv: "2608.09581",
title: "GenTrack3: Hybrid Stochastic-Deterministic Online Multi-Object Tracking with Cluster-Aware Association",
shortTitle: "GenTrack3",
year: 2026,
authors: ["Toan Van Nguyen", "Rasmus G. K. Christiansen", "Dirk Kraft", "Leon Bodenhagen"],
fileName: "2608.09581v1.pdf",
task: "multi-object",
tags: ["multi-object", "tracking-by-detection", "particle-filter", "pso", "mcmc", "cluster-aware-association", "online"],
difficulty: "advanced",
summary:
"GenTrack3 fuses stochastic particle filtering with deterministic tracking-by-detection: each target carries an MCMC-sampled, PSO-refined particle set scored by motion, HoG appearance, interaction, confidence and penalty fitness, while track birth/death stays deterministic. Its cluster-aware association partitions the global cost matrix via a 5-pixel occupancy grid with connected components, masks implausible pairs, resolves shared detections across clusters, and assigns locally with Hungarian matching — cutting space/time complexity while supporting group tracking. On MOT17-04 it reaches 84.86 MOTA and on crowded MOT20-05 the best 78.72 MOTA, all CPU-only.",
problem:
"Deterministic trackers depend on linear-Gaussian motion models and break on noisy detections and nonlinear motion; particle-filter MOT handles uncertainty but scales poorly as target count grows and struggles with time-varying target numbers, scale and appearance — and GenTrack2 global cost matrices plus neighbor weak-track refinement stay expensive and ambiguous in crowds.",
background: ["mot", "tracking-by-detection", "kalman", "data-association"],
previousWork: [
{
name: "Deterministic tracking-by-detection: SORT, DeepSORT, ByteTrack, BoT-SORT, OC-SORT, ConfTrack, BoostTrack",
limitation:
"Kalman-based, linear-Gaussian motion with global association; sensitive to noisy detections, nonlinear motion and occlusion-là-custom fixes do not scale cost with target count.",
whyThisPaper:
"Keeps deterministic birth/death and Hungarian matching but replaces the motion core with per-target MCMC plus PSO particles and partitions association into clusters.",
},
{
name: "Particle-filter MOT: BraMBLe, MCMC add-delete/stay-leave, detection-driven confidence particle filters",
limitation:
"High-dimensional joint state spaces scale badly with target numbers, duplicate tracks, and weak handling of scale, appearance and track-count changes.",
whyThisPaper:
"Samples per-target particles only (no full posterior, no sampling-time birth/death) and refines them with a history/exploration/social PSO fitness before association.",
},
{
name: "GenTrack and GenTrack2 hybrid predecessors",
limitation:
"Global T-by-D cost matrix for all tracks and detections plus neighbor-based weak-track refinement that turns ambiguous in crowded scenes.",
whyThisPaper:
"Cluster-aware matching (grid clustering, valid-pair masking, shared-detection reallocation) plus reliability-gated weak updates using strong-neighbor medians.",
},
],
researchGap:
"No online MOT framework simultaneously handles nonlinear non-Gaussian dynamics with per-target stochastic optimization and scales association sub-quadratically via clustered matching with group-tracking support.",
contribution: [
"Hybrid stochastic-deterministic pipeline: MCMC sampling plus PSO refinement per target with unified fitness (motion, HoG appearance, interaction, confidence, penalty); deterministic matching history for initiation and removal.",
"Tracklet state carrying identifiers, box states, velocities, penalties and ages with strong/weak/new taxonomy, decoupled position-size updates, and trend-based velocity regression over H states.",
"Cluster-aware association in four stages: occupancy-grid grouping, valid-pair masking, shared-detection reallocation, per-cluster Hungarian assignment preserving global optimality.",
"Reliability-gated weak-track update using strong-neighbor medians with velocity-similarity switching, plus dense-scene fallback to image-region partitioning.",
],
method: {
pipeline: ["sample-particles-mcmc", "optimize-particles-pso", "cluster-tracks-detections", "mask-and-compute-costs", "resolve-shared-detections", "associate-per-cluster", "update-tracklets", "refine-weak-tracks", "regress-velocities"],
architecture:
"Per-target particle set (S particles) with random-motion proposal; PSO over history, exploration and social fitness; HoG cosine appearance; Hungarian assignment per cluster; reference implementation on CPU with re-evaluable SORT/DeepSORT/ByteTrack/BoT-SORT/OC-SORT/SMILEtrack/ConfTrack/BoostTrack baselines.",
motionModel: "Random motion proposal (Eq. 1) with bounded perturbations scaled by box size — explicitly no Kalman linear model; trend velocity regressed over past H states with window F.",
appearanceModel:
"Non-negative HoG cosine similarity between particle and reference boxes for fitness; neighbor социального divergence term steers particles away from nearby identities to cut ID switches.",
association: "Cluster-restricted cost matrices with distance masks; shared detections reallocated to the minimum-representative-cost cluster; local Hungarian per cluster keeps global one-to-one matching.",
detectionDependency: "Full tracking-by-detection dependence — identical detections and ground truths shared with all baselines; low-score detections exploited via confidence-weighted costs and refinement.",
trackManagement:
"Strong tracks take detection states; weak tracks update from PSO global best plus reliable neighbors with penalty/age accrual (penalty in [0,1], age up to max); expired tracks removed; entrance-area penalty optional; new tracks from unmatched detections.",
optimization:
"PSO guided by sigma-weighted history/prior/social fitness; MCMC proposal tuned by epsilon/lambda exploration parameters; association hyperparameters tuned with Optuna on the most crowded per-scene training sequences.",
},
equations: [
{
id: "gentrack-sampling",
label: "Random-motion particle proposal",
formula: "V_t = V_{t-1} + eV UV; X_t = X_{t-1} + lV V_t + lX eX UX, with lX + lV = 1",
variables: [
{ symbol: "X_t = (u,v,w,h)", meaning: "target box center plus size state" },
{ symbol: "V_t", meaning: "box velocity bounded by per-target Vmax" },
{ symbol: "UX, UV", meaning: "perturbations bounded by box-size-derived maxima" },
{ symbol: "eX, eV, lX, lV", meaning: "state/velocity exploration regulators" },
],
intuition: "Jiggle each particle velocity and position within limits set by how big the target is, without assuming straight-line motion.",
why: "Kalman linearity plus Gaussian noise fails on irregular real-world dynamics; bounded random proposals explore nonlinear moves for PSO to exploit.",
where: "Particle sampling before PSO refinement; velocities also regressed over H past states for the next generation step.",
simulator: "motion",
paperIds: ["T143"],
},
{
id: "gentrack-cost",
label: "PSO fitness and track-detection matching cost",
formula: "f = ls fs + lm fm; f_PSO = sh fh + sp fp + si fi_s; C_ij = lp (1/S) sum_s Cm^{Xs,detj} + ld (1 - conf_j) + lh pen_i; Cm = su C_IoU + sd C_d, su + sd = 1",
variables: [
{ symbol: "fs", meaning: "HoG cosine appearance similarity in [0,1]" },
{ symbol: "fm", meaning: "target-adaptive normalized motion distance" },
{ symbol: "fi_s", meaning: "social fitness pushing particles away from neighbor states" },
{ symbol: "C_IoU, C_d", meaning: "spatial-plus-height IoU cost and size-normalized box distance cost" },
{ symbol: "conf_j, pen_i", meaning: "detection confidence and track penalty weighting reliability" },
],
intuition: "Score particles by looks plus motion plus keeping away from neighbors, then price each track-detection pair by averaged particle motion cost discounted for confident detections and trusted tracks.",
why: "Unifies motion, appearance, interaction, confidence and history into one price tag the Hungarian algorithm can minimize.",
where: "PSO refinement (Eqs. 2-4) and Hungarian cost matrix (Eqs. 5-6); strong/weak split follows the assignment.",
simulator: "assoc-cost",
paperIds: ["T143"],
},
{
id: "gentrack-clustering",
label: "Cluster-aware matching with valid-pair masking",
formula: "g_yx = 1 if (y in By and x in Bx) else 0 on 5 px grid; clusters = 8-connected components of occupied cells; M_ij^L = 1 if |Xi^L - det_j^L| <= lD sqrt(w_j^2 + h_j^2)/2 else 0; shared det_s assigned to cluster minimizing representative cost C^{det_s}",
variables: [
{ symbol: "g_yx", meaning: "occupancy cell inflated by ratio eps around each track box" },
{ symbol: "M_ij^L", meaning: "valid-pair mask skipping full cost computation for distant pairs" },
{ symbol: "det_s", meaning: "detection falling in multiple expanded clusters, reallocated to one" },
{ symbol: "lD > 1", meaning: "valid spatial range factor around a detection" },
],
intuition: "Draw a coarse grid, blob nearby tracks into groups, ignore far-apart pairs cheaply, and give contested detections to the group that prices them lowest.",
why: "Full T-by-D cost matrices cost O(T D tc) plus cubic assignment; partitioning plus one-distance masking beats S-IoU-plus-distance full costing per entry.",
where: "Four-stage matching (Figs. 2-6); dense fallback partitions the image into fixed regions when one cluster holds ~186 targets.",
simulator: "track-mgmt",
paperIds: ["T143"],
},
],
datasets: ["mot17", "mot20"],
metrics: ["mota", "idf1", "hota"],
baselines: ["SORT", "DeepSORT", "ByteTrack", "BoT-SORT", "OC-SORT", "SMILEtrack", "ConfTrack", "BoostTrack", "GenTrack Super", "GenTrack2 Super", "GenTrack2 Super+"],
results: [
"MOT17-04 (1050 frames, 1920 by 1080, 30 FPS): GenTrack3 Super ATA 63.62, IDF1 93.16, HOTA 67.61, MOTA 84.86; GenTrack2 Super+ best ATA 68.94 and MOTA 87.31; GenTrack2 Super best IDF1 93.88 and HOTA 70.04.",
"MOT20-05 crowded (3315 frames, 1654 by 1080, 25 FPS): GenTrack3 Super best MOTA 78.72 with HOTA 44.23, IDF1 88.23, ATA 34.00; GenTrack2 Super best ATA 35.84 and IDF1 91.88; ConfTrack best HOTA 45.59.",
"GenTrack2 Super+ matches GenTrack2 Super closely on both sequences, showing the clustered matcher preserves global accuracy at lower complexity.",
"Complexity: partitioned matching costs sum over clusters, strictly below global O(T D) space and O(max(T,D)^3) assignment; masking needs one box distance versus S IoU plus S distances per cost entry.",
],
ablations: [
"No classical component ablation table; comparison across GenTrack Super, GenTrack2 Super, GenTrack2 Super+ and GenTrack3 Super isolates the matcher and weak-refinement effects per sequence type.",
"Sparse MOT17-04 favors GenTrack2 neighbor refinement (smooth trajectories); dense MOT20-05 exposes its state-shift ambiguity, where GenTrack3 reliability gating wins MOTA.",
"Dense edge case (Fig. 7: 186 of 188 targets in one cluster) defeats grid grouping; fixed image-region partitioning (Fig. 8: six regions) restores grouping at the price of more shared detections.",
"Framework runs CPU-only (Ryzen 7 PRO 8840U, 16 GB, no GPU) with Optuna-tuned association hyperparameters on the most crowded training sequences.",
],
limitations: {
authorStated: [
"Cluster matching suits sparse scenes (robot navigation, human-robot interaction with group tracking); in dense crowds grouping collapses and regional partitioning raises shared-detection load.",
"Global-mask cost matrix balances sparse and dense performance better overall; full cross-version analysis deferred to future work.",
],
evident: [
"Evaluated on two sequences only (MOT17-04 sparse, MOT20-05 crowded), so generality across viewpoints, weather and camera motion is untested.",
"Uses non-standard ATA metric and reports no FPS/latency numbers despite real-time motivation; CPU-only timing is not quantified.",
"All comparisons share identical detections, so detection-robustness (a claimed weakness of determinism) is not actually stressed.",
],
},
assumptions: [
"Bounding-box state (center plus size) with bounded velocities suffices; position and size updates decoupled to avoid occlusion scale drift.",
"Weak tracks move only if neighbor velocity magnitude exceeds tau_V; size held near previous optimal until a detection match.",
],
computation:
"AMD Ryzen 7 PRO 8840U, 16 GB RAM, no dedicated GPU; identical detections and ground truths for all trackers; reference implementation released for re-evaluation.",
relations: [
{ to: "T005", type: "extends", note: "Extends the SORT tracking-by-detection lineage with per-target stochastic particle sets." },
{ to: "T013", type: "uses-as-baseline", note: "Compares against DeepSORT appearance-augmented association on both sequences." },
{ to: "T061", type: "uses-as-baseline", note: "Compares against ByteTrack high/low-confidence association handling." },
{ to: "T073", type: "uses-as-baseline", note: "Compares against OC-SORT observation-centric occlusion handling." },
{ to: "T075", type: "uses-as-baseline", note: "Compares against BoT-SORT refined Kalman plus appearance association." },
],
concepts: ["motion-model", "kalman", "data-association", "hungarian", "cost-matrix", "track-management"],
impact:
"Shows per-target stochastic optimization can replace Kalman motion inside a deterministic TBD scaffold, with clustered matching as a scalability pattern for sparse multi-target scenes.",
},
{
id: "T144",
arxiv: "2608.10790",
title: "MVTrack: Ultrafast Appearance-Free Moving Object Tracking from Compressed Bitstreams",
shortTitle: "MVTrack",
year: 2026,
authors: ["Inaki Erregue", "Kamal Nasrollahi", "Sergio Escalera"],
fileName: "2608.10790v1.pdf",
task: "multi-object",
tags: ["compressed-domain", "motion-vectors", "appearance-free", "multi-object", "real-time", "tracking-by-detection", "privacy-preserving"],
difficulty: "intermediate",
summary:
"MVTrack tracks pedestrians and vehicles using only H.264 motion vectors and macroblock metadata — never decoding RGB. MVDet, a 41K-parameter CenterNet-style detector on a 16-px MV grid, adds a learnable MB-partition embedding and a single-gate temporal gate (3x cheaper than ConvGRU) plus dilated bottleneck and heatmap-weighted box decoding; MVLink extends ByteTrack with box-scale-normalized kinematics that parks decelerated tracks in a Stopped state instead of drifting them. On VIRAT it beats YOLO26n-ByteTrack on HOTA (53.88 vs 52.78), MOTA (63.68 vs 54.56) and IDF1 with 60x fewer parameters, 40x fewer FLOPs and 8.6x lower CPU latency (5.21 ms, 2.17 ms ONNX).",
problem:
"Planet-scale surveillance needs multi-stream CPU-only tracking, but tracking-by-detection depends on GPU-heavy RGB detectors on decoded frames; compressed-domain methods still invoke RGB periodically (slow-fast paradigm), and motion-only tracking fragments identities whenever objects stop moving and vanish from motion vectors.",
background: ["mot", "tracking-by-detection", "detection", "iou"],
previousWork: [
{
name: "Hand-crafted compressed-domain analysis: MB activity maps, coding-decision detectors, MRF smoothing",
limitation:
"No RGB reconstruction but heuristic thresholds that fail on complex scenes and generalize poorly.",
whyThisPaper:
"Replaces heuristics with MVDet, a learned CenterNet detector over MV plus partition embeddings with temporal gating.",
},
{
name: "Slow-fast deep propagation: MV/box-level aggregation and learned feature propagation between RGB keyframes",
limitation:
"Still needs expensive RGB inference on I-frames and is tied to the codec settings seen in training.",
whyThisPaper:
"Always-fast design: I-frames are zero tensors (structured temporal dropout) and no RGB is ever decoded; robustness verified across FPS, GoP, bitrate and B-frames.",
},
{
name: "CoVA blob tracking and ByteTrack-style association",
limitation:
"CoVA needs per-video training and connectivity blobs that merge overlaps; standard trackers propagate unmatched tracks by velocity, which drags stopped objects away and fragments IDs.",
whyThisPaper:
"MVLink classifies unmatched tracks by normalized kinematics and anchors Stopped tracks at their last location for re-matching on motion resume.",
},
],
researchGap:
"No always-fast tracker detects and associates moving objects from bitstream metadata alone while explicitly surviving stationary intervals without appearance features.",
contribution: [
"MVDet: lightweight anchor-free detector on H.264 MV fields with learnable partition-size embeddings, single-gate EMA temporal gate (0.5K params, 4.2 MFLOPs), dilated bottleneck, and heatmap-weighted 3 by 3 neighborhood box decoding.",
"H.264 bitstream tensor representation: per-16-px (dx, dy) normalized by the 95th percentile and clipped to [-2,2] plus partition channel; destination-coordinate remapping; causal P-only encoding at 1080p/4 Mbps with 8 by 8 minimum partitions.",
"MVLink: ByteTrack association augmented with perspective-invariant normalized speed/acceleration (EMA-smoothed) and a Stopped versus Lost split that anchors stopped boxes for a fixed window.",
"VIRAT evaluation with a distractor-aware dynamic protocol plus efficiency proof: 41K params, 139M FLOPs, 5.21 ms CPU (2.17 ms ONNX) against YOLO26n.",
],
method: {
pipeline: ["parse-h264-bitstream", "build-mv-tensor", "embed-partitions", "temporal-gate", "detect-centers", "decode-boxes", "associate-mvlink"],
architecture:
"MVDet encoder-decoder (Tab. 1: 16ch conv plus gate, maxpool, 32ch conv, three dilated convs d 2/4/8, upsample, skip concat, 16ch conv; three 1 by 1 heads for heatmap, size, offset); MVLink minimalist kinematic associator without ReID.",
motionModel: "Kalman filter on image-plane velocity plus EMA-smoothed normalized kinematics; Stopped tracks anchored (no propagation), Lost tracks handled as in ByteTrack.",
appearanceModel:
"None by design — appearance-free; motion coherence plus MB-partition structure are the only cues, which also gives privacy preservation (no faces/plates reconstructed).",
association: "ByteTrack pipeline with pre-propagation motion-state classification; stopped-track anchoring enables re-matching after stationary gaps within a fixed window.",
detectionDependency: "Tracker is its own detector on MV tensors; needs no RGB detector. I-frames are zero tensors forcing extrapolation from preceding MV fields.",
trackManagement:
"Stopped versus Lost split for unmatched tracks; stopped boxes held at last reliable location for a fixed temporal window; standard birth/death otherwise; parked vehicles beyond the horizon still cause residual AssA gap.",
optimization:
"MVDet trained 100 epochs, batch 64, T equals 10 BPTT window (training-only horizon, streaming inference), Adam lr 6e-3 halved on mAP stagnation, translation plus flip augmentation; association hyperparameters via Optuna on most crowded per-scene training sequences.",
loss:
"Multi-task L equals 1.0 heatmap focal plus 0.1 size L1 plus 1.0 offset L1 plus 2.0 CIoU at centers; heatmap focal evaluated outside 3 by 3 neighborhoods, regressions masked inside them.",
},
equations: [
{
id: "mvtrack-gate",
label: "Single-gate EMA temporal gate",
formula: "g_t = sigmoid(Conv1x1(Concat(f_t, h_{t-1}))); h_t = GroupNorm(g_t * h_{t-1} + (1 - g_t) * f_t)",
variables: [
{ symbol: "f_t", meaning: "current-frame motion features after the first conv block" },
{ symbol: "h_{t-1}", meaning: "recurrent motion state carried across frames" },
{ symbol: "g_t", meaning: "single learned gate interpolating memory and input" },
],
intuition: "Decide per location how much of the remembered motion to keep versus the fresh motion, with one cheap gate instead of three.",
why: "Single MV frames are noisy (jitter, quantization); selective accumulation beats fixed-window averaging while staying tiny.",
where: "MVDet encoder after the first ConvBlock (S1), before maxpool; skip-connected to the decoder.",
params: "0.5K parameters, 4.2 MFLOPs — about 3x cheaper than ConvGRU; BPTT horizon T equals 10, gains saturate past T equals 5.",
paperIds: ["T144"],
},
{
id: "mvtrack-loss",
label: "Multi-task detection objective with CIoU refinement",
formula: "L = 1.0 L_hm + 0.1 L_wh + 1.0 L_off + 2.0 L_ciou; positives = center peaks, focal negatives outside 3 by 3; size/offset L1 inside R; CIoU on decoded center boxes",
variables: [
{ symbol: "L_hm", meaning: "modified focal loss on anisotropic Gaussian class heatmaps (max-merged overlaps, peak forced to 1)" },
{ symbol: "L_wh, L_off", meaning: "masked L1 size and sub-grid offset regressions over 3 by 3 neighborhoods" },
{ symbol: "L_ciou", meaning: "Complete-IoU loss on boxes decoded from size plus offset maps at centers" },
{ symbol: "R", meaning: "3 by 3 regression mask around each quantized center; ties broken by closest center" },
],
intuition: "Find centers with heatmaps, fine-tune width/height/offsets nearby, then directly optimize box overlap quality at the centers.",
why: "Coarse 16-px MV grids need sub-grid localization plus overlap-aware refinement for usable boxes.",
where: "MVDet training head; inference decodes with heatmap-weighted local expectation over the same 3 by 3 neighborhood.",
paperIds: ["T144"],
},
{
id: "mvtrack-kinematics",
label: "Perspective-invariant stopped-track kinematics",
formula: "v_norm^(t) = sqrt((vx^2 + vy^2) / (w h)); a_norm^(t) = v_norm^(t) - v_norm^(t-1); Stopped if smoothed v_norm < threshold and decelerating, else Lost",
variables: [
{ symbol: "vx, vy", meaning: "Kalman image-plane velocity components" },
{ symbol: "w, h", meaning: "current box width and height normalizing away perspective scale" },
{ symbol: "v_norm, a_norm", meaning: "EMA-smoothed normalized speed and acceleration" },
],
intuition: "Measure speed in units of the object own size so far-away and close objects share one stopped threshold, then freeze stopped boxes instead of drifting them.",
why: "Stopped objects vanish from MVs; velocity propagation walks the box away and guarantees an ID switch on resume — anchoring preserves the match.",
where: "MVLink unmatched-track handling; decoded boxes anchored at last reliable location for a fixed window.",
simulator: "motion",
paperIds: ["T144"],
},
],
datasets: ["others"],
metrics: ["hota", "deta", "assa", "mota", "idf1", "fps"],
baselines: ["YOLO26n"],
results: [
"VIRAT dynamic protocol: MVDet plus MVLink HOTA 53.88 (pedestrian 45.81, vehicle 61.95), DetA 53.47, AssA 54.63, MOTA 63.68, IDF1 70.50 — beats YOLO26n plus ByteTrack HOTA 52.78, MOTA 54.56, IDF1 66.94 (AssA 57.24 remains higher for RGB).",
"MVDet plus ByteTrack alone: HOTA 51.92, DetA 53.34, MOTA 62.18 — detection already ahead of RGB; MVLink adds the association recovery (AssA 50.91 to 54.63).",
"Efficiency: YOLO26n 2.5M params, 5.4G FLOPs, 44.80 ms CPU versus MVDet 41K, 139M, 5.21 ms (2.17 ms ONNX) — 60x fewer params, 40x fewer FLOPs, 8.6x lower latency.",
"Detection: single-frame baseline 28.05 mAP; temporal gate T equals 10 reaches 34.79; plus MB partition 36.45 mAP and 73.77 mAP50; LOSO cross-scene avg 31.91 (within 5 of standard).",
"Codec robustness: stable across FPS 25/20/15, 2 s GoP, 2/1 Mbps bitrates and B-frames; worst drop at low FPS for vehicles with large inter-frame displacements.",
],
ablations: [
"Temporal aggregation (mAP): baseline T equals 1 28.05; mean pooling T5 31.85; channel stack T5 32.04; gate T3 32.40; gate T5 34.51; gate T10 34.79; plus MB partition 36.45 — ordering discarded by windowing, gate wins.",
"Leave-one-scene-out mAP: 25.37/31.08/38.84/27.55/36.72 across folds, avg 31.91 — transferable motion patterns, modest scene specificity.",
"MVLink versus ByteTrack on identical MVDet boxes: HOTA 53.88 vs 51.92, IDF1 70.50 vs 65.86, isolating the stopped-state gain.",
"Qualitative: stable through tree occlusion, separates overlapping pedestrians and shadows, ignores dogs and parked cars; failure modes in Fig. 3 (merged neighbors, motion-mimic false positives, cyclist/pedestrian/vehicle confusion).",
],
limitations: {
authorStated: [
"One cell regresses one box, so small pedestrians in close proximity collapse into a single motion blob with imprecise coarse-grid localization.",
"Ambiguous non-target motion (rigid movers, elongated blobs) causes false positives and pedestrian/vehicle confusion until motion structure changes.",
"Long stationary intervals, especially parked vehicles, exceed the stopped-track horizon and explain the residual AssA and vehicle-HOTA gaps.",
"Future work: infrared/weather/H.265/viewpoint transfer, RGB-initialized conventional MOT extension, velocity-aware affinities replacing Kalman filters.",
],
evident: [
"VIRAT-only evidence (five scenes, static overhead CCTV, pedestrians/vehicles); moving-only scope excludes objects stationary from sequence start.",
"Evaluation marks static segments as distractors, which flatters motion-only detection versus conventional trackers penalized on full MOT.",
"Detector latency reported without bitstream-parsing versus RGB-decode cost separation; association overhead called negligible without numbers.",
],
},
assumptions: [
"Causal P-only H.264 at 1080p/4 Mbps with 8 by 8 minimum partitions; B-frames disabled for low latency (evaluated separately by sign-flipped mapping).",
"Static ground-truth intervals as distractors is a fair bridge between moving-object and full-MOT evaluation.",
],
computation:
"MVDet 100 epochs batch 64 on VIRAT DIVA-V1 (119 videos, 4.3 h, 30 FPS, 55 validation); YOLO26n at 640 input trained on moving plus static targets; Optuna-tuned association; CPU latency on consumer hardware.",
relations: [
{ to: "T061", type: "builds-on", note: "MVLink minimally modifies the ByteTrack association pipeline with Stopped/Lost kinematic states." },
],
concepts: ["tracking-by-detection", "data-association", "cost-matrix", "dual-threshold"],
impact:
"Proof that bitstream motion metadata alone beats a modern RGB detector for surveillance MOT, licensing always-on CPU front ends that trigger pixel models only on activity.",
},
{
id: "T145",
arxiv: "2608.15688",
title: "Training-Free Long-Term Multi-Object Tracking for Sports Video Analytics",
shortTitle: "McByte++",
year: 2026,
authors: ["Tomasz Stanczyk", "Seongro Yoon", "Francois Bremond"],
fileName: "2608.15688v1.pdf",
task: "multi-object",
tags: ["tracking-by-detection", "training-free", "sports-tracking", "long-term-tracking", "mask-propagation", "re-identification", "camera-motion-compensation", "online-tracking"],
difficulty: "intermediate",
summary:
"McByte++ extends the training-free McByte tracking-by-detection framework toward long-term sports MOT by adding an online re-identification mechanism that reconnects newly initialized tracklets with terminated identities, plus lightweight EdgeTAM mask propagation and conditional camera-motion compensation. On SoccerNet-tracking 2022 it reaches 87.5 HOTA and 84.5 IDF1 online (+2.5/+4.6 over McByte), on SportsMOT 79.9 HOTA and 83.6 IDF1 (+3.0/+6.1), while running up to an order of magnitude faster, all without detector retraining or dataset-specific tuning.",
problem:
"Sports MOT combines fast abrupt motion, frequent occlusions, broadcast camera motion and players repeatedly leaving and re-entering the view, so short-term tracking-by-detection fragments trajectories and loses identities; learned long-term alternatives need costly annotated training data, suffer domain shift across sports and leagues, and are too heavy for live large-scale analytics.",
background: ["mot", "tracking-by-detection", "data-association", "kalman", "hungarian", "iou", "occlusion", "reid", "track-management", "online-vs-offline"],
previousWork: [
{
name: "ByteTrack two-stage IoU association",
limitation:
"Separates high- and low-confidence detections but terminates identities after short absences, so re-entering players start new IDs and trajectories fragment.",
whyThisPaper:
"McByte++ keeps ByteTrack as its baseline and adds mask-guided short-term costs plus online long-term re-ID reconnection on top of the same detections.",
},
{
name: "McByte short-term mask-guided association",
limitation:
"Temporally propagated masks improve occlusion robustness but there is no explicit long-term identity reconnection, and the heavy propagation model costs runtime (about 1 FPS on SoccerNet).",
whyThisPaper:
"Adds selective online re-ID invoked only at tracklet initialization and swaps the heavy propagator for lightweight EdgeTAM, gaining identity preservation and roughly 10x speed.",
},
{
name: "DeepSORT / StrongSORT / BoT-SORT / Deep OC-SORT appearance fusion",
limitation:
"Appearance embeddings are matched at every frame-level association step, which is unreliable under sports occlusion and blur and can amplify errors in crowded scenes.",
whyThisPaper:
"Decouples re-ID from short-term matching entirely: motion- and mask-based costs handle frame-to-frame association while appearance is reserved for long-term reconnection.",
},
],
researchGap:
"No training-free online sports tracker combined selective mask guidance, reliability-gated camera-motion compensation and long-term identity reconnection without retraining, tuning or offline post-processing.",
contribution: [
"Unified training-free long-term framework extending McByte: regulated mask-guided short-term association, conditional camera-motion compensation, and online re-ID invoked only when new tracklets initialize.",
"Online long-term identity association with pretrained OSNet features, temporal aggregation, mutual-best-match gating and a fixed 0.8 similarity threshold shared across all datasets.",
"System redesign for efficiency: EdgeTAM lightweight mask propagation plus conditional affine CMC with 4x spatial downscaling, giving more than 6x speedup on SoccerNet and about 5x on SportsMOT.",
"State-of-the-art training-free sports results: online 87.5/84.5 and offline 88.6/87.2 HOTA/IDF1 on SoccerNet-tracking 2022, 79.9/83.6 online and 81.5/86.0 offline on SportsMOT.",
"Fair-comparison protocol using identical YOLOX detections and fixed parameters across all sequences, plus a mask-based-systems study showing structured tracking-by-detection beats pure DEVA, Grounded SAM 2 and MASA pipelines.",
],
method: {
pipeline: ["detect", "kalman-predict", "conditional-cmc-compensate", "iou-cost", "mask-gated-refine", "hungarian-match", "track-manage", "reid-reconnect"],
architecture:
"ByteTrack tracking-by-detection backbone with three auxiliary cues: EdgeTAM masks propagated synchronously with tracklet state, ORB-based affine camera-motion compensation applied conditionally after Kalman prediction, and OSNet appearance memory consulted only at new-tracklet birth for long-term identity transfer.",
motionModel:
"Linear Kalman-filter motion model on bounding-box state, with predicted boxes corrected by a conditionally applied global affine camera transform estimated from downscaled ORB feature matching.",
appearanceModel:
"Pretrained OSNet re-ID backbone without fine-tuning; per-detection crops give l2-normalized descriptors temporally averaged into one tracklet-level representation for cosine-similarity reconnection.",
association:
"Two-stage ByteTrack matching on IoU costs refined by mask fill ratio only under ambiguity or isolation gating; Hungarian assignment with cost-threshold rejection; long-term reconnection by mutual-best-match cosine similarity above 0.8.",
detectionDependency:
"YOLOX detectors pretrained on each dataset, never retrained; identical detections shared with baselines; oracle detections on SoccerNet-tracking 2022; fixed 0.6 confidence split threshold everywhere.",
trackManagement:
"Unmatched detections birth provisional tracklets; re-ID attempts reconnection to terminated long-term identities; delayed finalization confirms genuinely new identities only if no match occurs in a short window; unmatched tracklets terminate after extended misses.",
},
equations: [
{
id: "mcbyte-mask-coverage",
label: "Bounding-box coverage of propagated mask (Eq. 1)",
formula: "mc(i,j) = |mask(tracklet_i) INTERSECT bbox_j| / |mask(tracklet_i)|, in [0,1]",
variables: [
{ symbol: "mc(i,j)", meaning: "fraction of tracklet i mask covered by detection j" },
{ symbol: "mask(tracklet_i)", meaning: "EdgeTAM mask propagated with tracklet i" },
{ symbol: "bbox_j", meaning: "detection j bounding box" },
],
intuition: "How much of the expected object shape lands inside the candidate box; near 1 means the box fully contains the propagated shape.",
why: "Gating criterion ensuring mask cues only influence pairs where the detection plausibly contains the whole tracked shape.",
where: "Short-term association gating, condition 3, threshold mc >= 0.90 fixed across datasets.",
simulator: "iou-track",
paperIds: ["T145"],
},
{
id: "mcbyte-mask-fill",
label: "Mask fill ratio of detection box (Eq. 2)",
formula: "mf(i,j) = |mask(tracklet_i) INTERSECT bbox_j| / |bbox_j|, in [0,1]",
variables: [
{ symbol: "mf(i,j)", meaning: "fraction of detection j area filled by tracklet i mask" },
{ symbol: "bbox_j", meaning: "detection j bounding box whose pixel count normalizes the overlap" },
],
intuition: "How well the detection spatially aligns with the propagated shape; discriminative because different masks inside one box give different fill values.",
why: "Provides the actual cost correction signal, unlike coverage which only gates.",
where: "Short-term association gating condition 2 with threshold mf >= 0.05, then cost modifier in Eq. 3.",
simulator: "iou-track",
paperIds: ["T145"],
},
{
id: "mcbyte-mask-cost",
label: "Mask-gated association cost update (Eq. 3)",
formula: "costs(i,j) = costsIoU(i,j) - mf(i,j) if ambiguity/isolation and all gates pass, else costsIoU(i,j)",
variables: [
{ symbol: "costs(i,j)", meaning: "final tracklet-detection assignment cost" },
{ symbol: "costsIoU(i,j)", meaning: "original IoU-based cost, 1 - IoU" },
{ symbol: "mf(i,j)", meaning: "mask fill ratio bonus subtracted when mask guidance is trusted" },
],
intuition: "Trusted masks discount the cost of the shape-consistent pair so the Hungarian solver prefers it in ambiguous rows or isolated high-cost cells.",
why: "Lets spatial extent break ties and rescue matches that pure box overlap would reject, without letting noisy masks corrupt ordinary matches.",
where: "Cost-matrix construction before Hungarian matching, applied only in ambiguity or isolation cases.",
params: "Lowering the mf threshold admits more mask influence but risks drifted-mask errors; raising mc strictness has the opposite effect.",
simulator: "assoc-cost",
paperIds: ["T145"],
},
{
id: "mcbyte-reid-aggregate",
label: "Re-ID normalization and tracklet aggregation (Eqs. 4-6)",
formula: "f_tilde_t = f_t / ||f_t||2; f_avg = (1/N) SUM f_tilde_i; f_tilde_avg = f_avg / ||f_avg||2; match by cosine similarity",
variables: [
{ symbol: "f_t", meaning: "OSNet appearance vector from one detection crop" },
{ symbol: "f_tilde_t", meaning: "l2-normalized single-frame descriptor" },
{ symbol: "N", meaning: "number of reliable observations aggregated for the tracklet" },
{ symbol: "f_tilde_avg", meaning: "renormalized tracklet-level appearance representation" },
],
intuition: "Average what a person looked like over many clean sightings so one blurred or half-occluded crop cannot decide identity.",
why: "Stabilizes long-term reconnection against the noisy per-frame embeddings typical of sports footage.",
where: "Appearance memory update during short-term tracking; comparison happens only when a new tracklet initializes.",
simulator: "reid",
paperIds: ["T145"],
},
{
id: "mcbyte-bytetrack-match",
label: "ByteTrack two-stage Hungarian association",
formula: "costsIoU(i,j) = 1 - IoU(pred_i, det_j); match high-confidence detections first, then low-confidence; reject pairs above cost threshold",
variables: [
{ symbol: "pred_i", meaning: "Kalman-predicted and CMC-compensated tracklet box" },
{ symbol: "det_j", meaning: "observed detection box in current frame" },
{ symbol: "IoU", meaning: "box overlap between prediction and observation" },
],
intuition: "First link the trustworthy sightings, then use leftover weak sightings to rescue occluded tracks instead of deleting them.",
why: "Baseline association skeleton that McByte++ enriches with mask and long-term cues while keeping its efficiency.",
where: "Every frame in both association stages before track management and re-ID reconnection.",
simulator: "bytetrack",
paperIds: ["T145"],
},
],
datasets: ["others"],
metrics: ["hota", "idf1", "mota", "fps"],
baselines: ["ByteTrack", "OC-SORT", "McByte", "MixSORT", "DiffMOT", "GTR", "CenterTrack", "MeMOTR", "MOTIP", "MotionTrack", "Deep-EIoU", "SportMamba", "DEVA", "Grounded SAM 2", "MASA"],
results: [
"SoccerNet-tracking 2022 test (oracle detections): online McByte++ 87.5 HOTA, 84.5 IDF1, 97.1 MOTA versus McByte 85.0, 79.9, 96.8; offline with GTA-link 88.6 HOTA, 87.2 IDF1, 97.1 MOTA.",
"SportsMOT test: online McByte++ 79.9 HOTA, 83.6 IDF1, 96.9 MOTA versus McByte 76.9, 77.5, 97.2 and ByteTrack 62.1, 69.1, 93.4; offline with GTA-link 81.5 HOTA, 86.0 IDF1, 96.8 MOTA.",
"SoccerNet-tracking Challenge 2023 (SportsMOT-trained YOLOX): online with re-ID 64.3 HOTA, 78.6 IDF1, 81.8 MOTA versus McByte 64.1, 76.5, 81.8; no-re-ID variant 62.4 HOTA, 74.1 IDF1 at 15.05 FPS versus McByte 1.46 FPS.",
"Ablation: EdgeTAM swap alone lifts SoccerNet runtime 1.04 to 6.49 FPS and SportsMOT 3.60 to 18.65 FPS at ds=4; conditional CMC with ds=4 gives 10.71 FPS at identical 84.1 HOTA; re-ID threshold 0.8 is best on both datasets (SoccerNet 87.5/84.5, SportsMOT 79.9/83.6), threshold 0.6 merges distinct players and 0.9 blocks true reconnections.",
"SportsMOT validation versus mask systems with YOLOX detections: McByte++ online 85.5 HOTA, 87.6 IDF1 and offline 87.2, 90.4, ahead of MASA (73.6, 71.2), Grounded SAM 2 (66.1, 70.2) and DEVA (42.4, 42.1).",
],
ablations: [
"Mask replacement with ds=2 keeps accuracy (SoccerNet 84.6 vs 85.0 HOTA) while multiplying speed 6x; ds=4 chosen as accuracy-efficiency trade-off, ds=6 degrades both datasets.",
"Conditional versus unconditional CMC holds HOTA/IDF1 and slightly improves FPS (10.71 vs 10.51 SoccerNet, 19.08 FPS SportsMOT at ds=4).",
"Similarity threshold sweep 0.6/0.7/0.8/0.9 peaks at 0.8 on both benchmarks with MOTA and runtime nearly flat, confirming gains come from identity recovery.",
"Offline GTA-link post-processing adds +1.1 HOTA/+2.7 IDF1 on SoccerNet and +1.6/+2.4 on SportsMOT over online re-ID at reduced FPS.",
],
limitations: {
authorStated: [
"Very similar teammates can still occasionally merge or resist reconnection under strong appearance change or limited visual evidence.",
"SAM3 could not be evaluated because its VRAM demand exceeds available resources on long dense sports sequences.",
"SoccerNet-tracking Challenge 2023 comparison is restricted to McByte variants because the official evaluation server was unavailable.",
],
evident: [
"Oracle detections on SoccerNet-tracking 2022 inflate MOTA near 97, so headline gains live entirely in HOTA/IDF1 association terms.",
"Single fixed re-ID threshold 0.8 and fixed mask gate values are asserted to generalize but are validated only on soccer, basketball and volleyball footage.",
"Best reported numbers rely on offline GTA-link, which needs the full tracker output and breaks the online deployment story.",
],
},
assumptions: [
"Same off-the-shelf detections for all compared trackers; no detector training or per-sequence tuning.",
"Fixed operating points everywhere: detection split 0.6, mask gates mf 0.05 and mc 0.90, re-ID similarity 0.8, CMC downscale 4.",
"Re-ID backbone pretrained elsewhere is trusted without adaptation to each league or broadcast style.",
],
computation:
"Online 8.69 FPS on SoccerNet and 14.57 FPS on SportsMOT with re-ID (10.71/19.08 without); 15.05 FPS on the 2023 split; HPC support from IDRIS/GENCI allocation 2025-AD011014370.",
relations: [
{ to: "T061", type: "builds-on", note: "ByteTrack two-stage IoU association is the baseline framework McByte++ augments." },
{ to: "T073", type: "uses-as-baseline", note: "Evaluated under identical detections on both benchmarks (SoccerNet 82.0 HOTA, SportsMOT 68.1)." },
{ to: "T075", type: "uses-as-baseline", note: "Compared on SportsMOT (68.7 HOTA) as appearance-plus-motion tracking-by-detection reference." },
{ to: "T067", type: "uses-as-baseline", note: "Cited as appearance-every-frame system contrasted with selective long-term-only re-ID." },
{ to: "T005", type: "uses-as-baseline", note: "SORT/Kalman-filter lineage underlying the motion model; compared on DanceTrack-style tables via SportsMOT results." },
{ to: "T013", type: "conceptual-successor", note: "Revisits DeepSORT-style deep appearance but restricts it to tracklet-birth reconnection." },
],
concepts: ["mot", "tracking-by-detection", "data-association", "kalman", "hungarian", "iou", "cost-matrix", "track-management", "occlusion", "reid", "hota", "idf1"],
impact:
"Shows long-term sports identity continuity is achievable training-free by relegating appearance to rare reconnection events, keeping structured tracking-by-detection competitive with heavy learned and mask-based systems.",
},
{
id: "T146",
arxiv: "2608.24365",
title: "MaST: Motion-aware Sparse Pipeline for Lightweight Object Tracking",
shortTitle: "MaST",
year: 2026,
authors: ["Qingmao Wei", "Fagui Liu", "Dengke Zhang", "Qingze He", "Quan Tang"],
fileName: "2608.24365v1.pdf",
task: "single-object",
tags: ["single-object", "lightweight", "vision-transformer", "token-sparsification", "motion-prior", "sparse-prediction-head", "real-time", "edge-deployment"],
difficulty: "advanced",
summary:
"MaST makes token sparsity effective end to end for lightweight SOT: a Gaussian motion window built from the previous prediction re-weights diffuse early-layer cross-attention scores so 70 percent of search tokens can be dropped at encoder layer 1, and a natively sparse MLP head decodes boxes directly from retained unstructured tokens with a score-first regress-once path. MaST-tiny scores 63.8 LaSOT AUC and 80.1 TrackingNet SUC, beating AsymTrack-S by 1.0 and 2.2 points while running at 152 FPS on Jetson Nano, nearly twice as fast.",
problem:
"Dense one-stream transformers are accurate but quadratically expensive; existing token-pruning methods prune late because early attention is diffuse and they still reshape sparse tokens back into dense convolutional heads, so worst-case edge latency barely improves and aggressive pruning can even delete the target-center token.",
background: ["sot", "bounding-box", "iou", "siamese", "attention", "motion-model", "success-plot"],
previousWork: [
{
name: "OSTrack progressive cross-attention pruning",
limitation:
"Prunes at layers 4/7/11 because early attention is noisy, leaving costly early layers dense; multi-stage schedule gives negligible speedup (9.1 to 10.5 RPi FPS).",
whyThisPaper:
"Injects a nearly free temporal motion prior so one-shot pruning at layer 1 matches dense accuracy (63.8 vs 64.0 AUC) at 22.6 RPi FPS.",
},
{
name: "Dense convolutional prediction heads",
limitation:
"Assume a full 2D token grid, forcing pad-and-reshape of sparse tokens and redundant regressions at every location, which caps throughput at 13.8 FPS.",
whyThisPaper:
"Replaces the head with sparse-anchor MLPs that score retained tokens and regress only the argmax token, reaching 23.2 FPS with zero accuracy loss.",
},
{
name: "Lightweight and compressed trackers (LightTrack, FEAR, HiT, MixFormerV2-S, LoReTrack, CompressTracker)",
limitation:
"Compact backbones, resolution cuts or layer pruning trade large accuracy drops (4.9 to 5.5 AUC) for speed, or distill without fixing the dense-head bottleneck.",
whyThisPaper:
"One-shot motion-guided sparsification keeps full resolution and depth, beating resolution-reduction and layer-pruning baselines by over 4 AUC at equal MACs.",
},
],
researchGap:
"No lightweight tracker combined reliable early-layer token reduction with a prediction head that natively consumes sparse unstructured tokens at fixed predictable latency.",
contribution: [
"Motion-aware token sparsification: previous-box Gaussian window softly re-weights center-template cross-attention scores, retaining top 30 percent of search tokens after an early encoder layer.",
"Natively sparse MLP prediction head with score-first regress-once decoding on retained tokens, no padding or dense reshape, scaling as O(NK) for scoring plus constant-time regression.",
"Three ViT-Tiny variants (nano 8-layer, tiny 12-layer, small high-resolution) all Pareto-optimal on Raspberry Pi 5, CPU and Jetson Orin Nano speed-accuracy plots.",
"State-of-the-art lightweight results: MaST-tiny 63.8/80.1/66.6 on LaSOT/TrackingNet/GOT-10k; MaST-small 65.8/82.3/70.0; MaST-nano 58.6 AUC at 30.1 RPi and 230 Nano FPS.",
"Full sparsification diagnostics: motion-plus-attention closes the dense gap, layer-1 pruning matches later layers, and one checkpoint supports smooth post-training retention-rate trade-offs.",
],
method: {
pipeline: ["embed-template-search", "cross-attention-score", "motion-window-reweight", "topk-sparsify", "sparse-encode", "sparse-score", "argmax-select", "regress-once-decode"],
architecture:
"One-stream ViT-Tiny encoder with a single General Sparsification Module inserted at layer 1; all later blocks process only the retained 30 percent of search tokens, decoded by twin lightweight MLP branches for scoring and box regression.",
motionModel:
"No learned motion network: a 2D Gaussian window centered on the previous predicted box with sigma proportional to box size (gamma 0.5) supplies the temporal locality prior for token selection.",
appearanceModel:
"Joint template-search self-attention features; token relevance measured by cross-attention of each search query against the center template token only.",
association: "Not applicable (single-object tracking; no detection association stage).",
detectionDependency: "None — first-frame box given; template/search crops embedded as patches.",
trackManagement:
"Fixed top-K retention guarantees predictable per-frame latency; previous-frame prediction alone carries temporal state, with no template update or memory bank.",
optimization:
"AdamW with backbone learning rate 4e-5 and 4e-4 elsewhere, batch 128 pairs, 300 epochs of 60k pairs, 10x decay after 240 epochs, gradual sparsification warm-up; loss weights lambda-L1 5 and lambda-GIoU 2.",
loss:
"Classification loss over sparse token scores plus L1 and GIoU regression losses supervised only on the retained token nearest the ground-truth center.",
},
equations: [
{
id: "mast-cross-attention",
label: "Template-conditioned cross-attention map (Eq. 1)",
formula: "A(x->z) = Softmax(Q_x K_z^T / sqrt(d))",
variables: [
{ symbol: "Q_x", meaning: "search-token query features, Px by d" },
{ symbol: "K_z", meaning: "template-token key features, Pz by d" },
{ symbol: "d", meaning: "feature dimension scaling the dot product" },
{ symbol: "A(x->z)[i,j]", meaning: "attention weight between search token i and template token j" },
],
intuition: "Each search patch asks how much it looks like each template patch; high weights mark candidate target regions.",
why: "Raw relevance signal that token pruning must refine before it can be trusted at early layers.",
where: "Inside the General Sparsification Module at the chosen early encoder layer.",
paperIds: ["T146"],
},
{
id: "mast-importance",
label: "Center-token importance score (Eq. 2)",
formula: "s_i = exp(q_i^T k_c / sqrt(d)) / SUM_k exp(q_k^T k_c / sqrt(d))",
variables: [
{ symbol: "q_i", meaning: "query feature of search token i" },
{ symbol: "k_c", meaning: "key feature of the template center patch" },
{ symbol: "s_i", meaning: "normalized importance of search token i" },
],
intuition: "Collapse the full attention map to one number per search token by comparing only against the most target-like template patch.",
why: "Lightweight scoring that avoids full pairwise aggregation while retaining OSTrack-level pruning signal.",
where: "Token scoring step feeding the motion-prior fusion.",
paperIds: ["T146"],
},
{
id: "mast-motion-window",
label: "Gaussian motion window from previous prediction (Eq. 3)",
formula: "G_t(u,v) = exp(-(u-x_{t-1})^2/(2 sig_x^2) - (v-y_{t-1})^2/(2 sig_y^2)); sig_x = gamma w_{t-1}, sig_y = gamma h_{t-1}",
variables: [
{ symbol: "G_t(u,v)", meaning: "spatial prior weight at search-region token position (u,v)" },
{ symbol: "(x_{t-1}, y_{t-1})", meaning: "previous predicted box center anchoring the window" },
{ symbol: "(w_{t-1}, h_{t-1})", meaning: "previous box size setting the window scale" },
{ symbol: "gamma", meaning: "window-size factor, default 0.5" },
{ symbol: "sig_x, sig_y", meaning: "Gaussian standard deviations in x and y" },
],
intuition: "Trust tokens near where the target just was, with tolerance growing for bigger targets that can move farther.",
why: "Supplies the missing temporal guidance that makes early-layer pruning stable instead of scattering retention over background.",
where: "Motion prior injection before top-K selection; gamma 0.5 balances inclusion versus exclusion.",
params: "Larger gamma keeps more context but dilutes savings; smaller gamma risks cutting fast-moving targets.",
simulator: "motion",
paperIds: ["T146"],
},
{
id: "mast-fused-selection",
label: "Motion-reweighted top-K selection (Eq. 4)",
formula: "w_i = G_t(u_i, v_i) * s_i; keep token set L_K of top-K w_i with original grid positions p_k = (u_k, v_k)",
variables: [
{ symbol: "w_i", meaning: "combined retention score of token i" },
{ symbol: "s_i", meaning: "appearance importance from cross-attention" },
{ symbol: "(u_i, v_i)", meaning: "spatial location of token i on the patch grid" },
{ symbol: "L_K", meaning: "retained sparse token set, NK much smaller than full grid" },
],
intuition: "Multiply what-it-looks-like by where-it-should-be, then keep only the winners while remembering where each winner came from.",
why: "Fusion is the key enabler: attention alone drops to 60.5 AUC and motion alone to 62.6, while fused reaches 63.8 near the 64.0 dense baseline.",
where: "Output of the sparsification block; all subsequent transformer blocks run on L_K only.",
paperIds: ["T146"],
},
{
id: "mast-sparse-decode",
label: "Score-first regress-once sparse decoding (Eqs. 5-8)",
formula: "k* = argmax_k s_k; b_hat = Decode(p_{k*}, Delta_{k*}) = (u_{k*}+dx, v_{k*}+dy, w, h); L = L_cls + 5 L_L1 + 2 L_GIoU on token nearest GT center",
variables: [
{ symbol: "s_k", meaning: "score-branch confidence of retained token k" },
{ symbol: "k*", meaning: "selected target token index" },
{ symbol: "p_{k*}", meaning: "pre-pruning grid coordinate of the selected token acting as sparse anchor" },
{ symbol: "Delta", meaning: "regression-branch box offsets and size" },
{ symbol: "b_hat", meaning: "final box after grid-to-pixel conversion" },
],
intuition: "Pick the best surviving token first, then refine only its box instead of regressing everywhere including empty padding.",
why: "Removes the dense-head bottleneck: same 63.8 AUC as dense MLP head but 23.2 versus 21.3 FPS, far above the 13.8 FPS convolutional head.",
where: "Prediction head on retained tokens; regression supervised only at the ground-truth-nearest token k_gt.",
paperIds: ["T146"],
},
],
datasets: ["lasot", "trackingnet", "got10k", "uav123", "others"],
metrics: ["success-auc", "precision", "norm-precision", "fps"],
baselines: ["AsymTrack-T", "AsymTrack-S", "AsymTrack-B", "HiT-Tiny", "HiT-Small", "HiT-Base", "FEAR-XS", "LightTrack", "E.T.Track", "FARTrack-pico", "FARTrack-nano", "FARTrack-tiny", "ORTrack-D", "FERMT", "MixFormerV2-S", "LiteTrack-B4", "CompressTracker-2", "OSTrack", "LoRAT", "KCF"],
results: [
"LaSOT: MaST-tiny 63.8 AUC and 72.2 PNorm (best in 1G tier, +1.0 over AsymTrack-S, +3.3 over HiT-Small); MaST-small 65.8 AUC; MaST-nano 58.6 AUC at 30.1 RPi FPS.",
"TrackingNet: MaST-tiny 80.1 SUC and 85.3 PNorm (+2.2 over AsymTrack-S); MaST-small 82.3 SUC; MaST-nano 77.2 SUC.",
"GOT-10k: MaST-tiny 66.6 AO, 76.0 SR0.5, 62.5 SR0.75 (+1.1 AO over AsymTrack-S); MaST-small 70.0 AO with best 67.0 SR0.75.",
"Edge speed: MaST-tiny 152 FPS on Jetson Nano versus 88 for AsymTrack-S; MaST-nano 230 Nano FPS and 101 CPU FPS; MaST-small 98 Nano FPS; Raspberry Pi 5 22.6/30.1/7.5 FPS for tiny/nano/small.",
"UAV123 66.6, NFS 66.2 and VastTrack 35.6 SUC with 43.2 PNorm for the small/tiny variants, leading or second in tier; sparsification at layer 1 matches later layers within 0.4 AUC while halving latency.",
],
ablations: [
"Attention-only pruning 60.5 AUC and predictor-only 59.4 versus 63.8 fused and 62.6 motion-only at equal 824M MACs, proving the prior is the enabler.",
"Head study: global-token and decoder heads reach only 57.9-60.1 sparse AUC; 3x3 conv head 64.1 AUC but 13.8 FPS; sparse MLP 63.8 AUC at 23.2 FPS with no loss over dense MLP.",
"Retention sweep on GOT-10k val: 384 resolution needs only 30-40 percent tokens for dense parity, 256 needs 50-60 percent; one checkpoint serves all operating points.",
"Sparsify-at-layer 1 through 6 varies AUC by at most 0.4 while FPS falls 22.6 to 10.1, so layer 1 is the default; one-shot beats progressive OSTrack pruning and resolution/layer compression at equal 1G budget.",
],
limitations: {
authorStated: [
"High-resolution inputs still incur substantial attention computation before sparsification takes effect, motivating input-adaptive pruning as future work.",
],
evident: [
"At 30 percent retention the target-center token can be pruned under fast motion or bad previous predictions, and the fixed top-K budget cannot recover it.",
"Motion prior inherits previous-frame errors with no uncertainty gating, so one mislocalization biases the next token selection.",
"All variants use ViT-Tiny with MAE-lite init; scaling beyond lightweight backbones is only a supplementary study.",
],
},
assumptions: [
"Smooth inter-frame motion: target stays within a gamma-0.5 Gaussian of its previous box.",
"Template center patch is a sufficient relevance representative for all search tokens.",
"ONNX Runtime edge benchmarks with fixed retention reflect real deployment latency.",
],
computation:
"Training on one RTX 3090, 300 epochs; inference via ONNX OpSet 17 on Raspberry Pi 5, Apple M4 and Jetson Orin Nano with re-evaluated baselines under the same protocol.",
relations: [
{ to: "T071", type: "builds-on", note: "Adapts OSTrack center-template cross-attention scoring into motion-guided one-shot pruning." },
{ to: "T009", type: "conceptual-successor", note: "Moves the SiamFC cosine-window locality prior from score post-processing into token selection." },
{ to: "T028", type: "conceptual-successor", note: "Inherits SiamRPN++ size and displacement penalty intuition for suppressing abrupt jumps." },
{ to: "T001", type: "conceptual-successor", note: "Continues the KCF/ECO cosine-window tradition of penalizing off-center predictions." },
{ to: "T058", type: "uses-as-baseline", note: "Compares against LightTrack lightweight NAS tracker in the sub-1G tier." },
{ to: "T087", type: "uses-as-baseline", note: "Compares against MixFormerV2-S efficient fully-transformer tracker and its token-decoding head style." },
],
concepts: ["sot", "siamese", "attention", "motion-model", "success-plot", "bounding-box", "benchmark-design"],
impact:
"Demonstrates end-to-end sparsity from tokens to boxes as a practical real-time path, shifting efficient-tracking design from smaller backbones to motion-guided computation plus sparse-native heads.",
},
{
id: "T147",
arxiv: "2608.29126",
title: "Efficient Language-to-Vision Feature Injection for Referring Single-Object Tracking",
shortTitle: "LVTrack",
year: 2026,
authors: ["Han Wang", "Yuxuan Liu", "Yuhan Sun", "Jian Yang", "Xiaotong Xu", "Yixuan Lv", "Zhuang Zhou", "Shengyang Li"],
fileName: "2608.29126v1.pdf",
task: "multimodal-tracking",
tags: ["referring-tracking", "vision-language", "frozen-backbone", "gated-fusion", "autoregressive-localization", "memory", "single-object", "multimodal"],
difficulty: "advanced",
summary:
"LVTrack tackles referring SOT, where language grounds the target but overemphasized text causes semantic drift during tracking. A mode-conditioned Gated Feature Injector (MGFI) injects frozen Perception Encoder language features into visual tokens with amplitude-regularized dual-mode gates (conservative in tracking, strong in grounding), while hybrid absolute-plus-2D-RoPE positions, a lightweight appearance memory and a Gaussian-smoothed KL loss supervise autoregressive box-token prediction. LVTrack-384 leads NL-only tracking on TNL2K (58.7 AUC) and OTB99 (63.6), tops RefCOCOg grounding (74.88), and trains in about 2 hours with 100M trainable parameters.",
problem:
"Referring trackers need language for first-frame grounding but suffer semantic drift toward similar distractors when text dominates later frames; joint vision-language alignment training is costly and overfits small tracking datasets, while frozen VLP backbones lack instance-level binding and standard cross-entropy ignores coordinate ordinality.",
background: ["sot", "multimodal", "attention", "memory-network", "success-plot", "benchmark-design"],
previousWork: [
{
name: "Joint concatenation trackers (JointNLT, UVLTrack, SUTrack, qTrackv2)",
limitation:
"Concatenate BERT language tokens with visual tokens in a learnable encoder, needing costly end-to-end alignment training (up to 300 epochs) and entangling grounding with tracking.",
whyThisPaper:
"Freezes the Perception Encoder VLP backbone and injects text via lightweight MGFI cross-attention, converging in 80 epochs with 100M trainable parameters.",
},
{
name: "QueryNLT historical-evidence suppression",
limitation:
"Uses past visual evidence to down-weight misaligned descriptions but keeps a heavy joint reasoning stack without explicit control of injection strength per mode.",
whyThisPaper:
"Adds explicit amplitude-regularized dual-mode gates so tracking injection is provably bounded to a fraction rho of grounding injection.",
},
{
name: "SeqTrack-style autoregressive coordinate prediction with cross-entropy",
limitation:
"Discretized bins trained with one-hot cross-entropy penalize near-miss and far-off errors equally, causing quantization ambiguity and coarse localization.",
whyThisPaper:
"Replaces it with Gaussian-smoothed KL soft targets that weight nearby bins higher, adding regression-like distance sensitivity to token classification.",
},
],
researchGap:
"No referring tracker combined frozen-VLP efficiency, explicit per-mode calibration of textual injection against semantic drift, and ordinal-aware autoregressive supervision in one unified grounding-plus-tracking model.",
contribution: [
"LVTrack pure-transformer RSOT framework on a fully frozen Perception Encoder backbone with MGFI modules at visual layers 5-10 fed by language layers 14-24.",
"Amplitude-regularized dual-mode Gated Feature Injector sharing cross-attention geometry across modes while bounding tracking-mode gate norm to rho times grounding-mode norm.",
"Gaussian-smoothed KL loss replacing one-hot coordinate targets with distance-aware Gaussian soft distributions over 4000 bins.",
"Hybrid positional modeling (shared learnable 2D absolute anchors plus 2D axial RoPE) with a 64-token RoI-Align appearance memory updated only above confidence tau-mem.",
"Strong cheap results: best NL-only TNL2K and OTB99 scores, best RefCOCOg accuracy, competitive NL+BBOX scores, 100M params, 80 epochs, 30K samples per epoch, about 2 hours on 8 RTX 4090.",
],
method: {
pipeline: ["vlp-encode", "mgfi-inject", "adapter-project", "memory-fuse", "hybrid-position-encode", "autoregressive-decode"],
architecture:
"Frozen PE-Base dual backbone (first 10 of 12 visual layers) with six MGFI cross-attention injectors, MLP adapters, an 8-layer tracking encoder over template plus memory-fused search tokens with mode prompts, and a 4-layer RoPE autoregressive decoder emitting x, y, w, h tokens.",
motionModel: "None learned — temporal continuity comes from the appearance memory and local search-region decoding with Hanning-window center bias.",
appearanceModel:
"VLP visual tokens enriched by language-conditioned injection; temporal priors from RoI-Align memory tokens of past predicted boxes fused by gated cross-attention.",
association: "Not applicable (single-object scope; template-only, grounding and joint-tracking modes selected by input masking).",
detectionDependency: "None — NL-only mode grounds from text alone with template tokens masked; NL+BBOX mode adds the visual template.",
trackManagement:
"First frame localized in grounding mode to build the template; memory refreshed only when aggregated coordinate-head confidence is high; spatial directive words stripped from expressions after the first frame.",
optimization:
"AdamW, learning rate 2e-4, batch 192, cosine annealing with 20 warmup epochs for 80 epochs; unified hybrid sampling of tracking, joint and grounding data at 0.45/0.40/0.15.",
loss:
"Localization loss as summed KL divergence to Gaussian-smoothed bin targets (sigma 4) plus amplitude-ratio loss weighted by lambda-ratio 0.01.",
},
equations: [
{
id: "lvtrack-vlp-encode",
label: "Frozen VLP feature extraction (Eq. 1)",
formula: "F_L = phi_l(L); F_t = phi_v(I_t); F_s = phi_v(I_s)",
variables: [
{ symbol: "L", meaning: "referring expression text" },
{ symbol: "I_t, I_s", meaning: "template and search images" },
{ symbol: "phi_l, phi_v", meaning: "frozen language and visual backbones" },
{ symbol: "F_L, F_t, F_s", meaning: "textual, template and search features" },
],
intuition: "Borrow a pretrained bilingual brain for words and pictures without rewiring it, then do all learning in small adapter modules.",
why: "Preserves general VLP alignment and cuts training to 100M parameters and 80 epochs.",
where: "Backbone stage; only the first 10 visual layers are tapped since late layers over-compress spatial detail.",
paperIds: ["T147"],
},
{
id: "lvtrack-mgfi",
label: "Mode-conditioned gated injection (Eqs. 2-4)",
formula: "f_hat = f + lambda_attn(m) * MHCA(f, text, text); f_tilde = f + Delta(m); lambda = tanh(gamma), init gamma = 0",
variables: [
{ symbol: "f", meaning: "intermediate frozen search feature at visual layer n" },
{ symbol: "MHCA", meaning: "multi-head cross-attention aggregating language semantics" },
{ symbol: "lambda_attn(m)", meaning: "mode-specific channel gate for mode m in {Grounding, Tracking}" },
{ symbol: "Delta(m)", meaning: "learnable language-induced perturbation including the MLP branch" },
{ symbol: "f_tilde", meaning: "language-enhanced visual feature" },
],
intuition: "Season the visual features with a pinch of language, with a bigger pinch when grounding from text alone and a tiny one when a template already shows the target.",
why: "Makes drift explicit as a bounded perturbation: gate norms cap how far language can push the visual representation.",
where: "Six MGFI modules at visual layers 5-10; shared attention, mode-specific gates and MLP branches.",
paperIds: ["T147"],
},
{
id: "lvtrack-ratio-loss",
label: "Amplitude ratio loss (Eq. 5)",
formula: "L_ratio = max(0, log||lambda_T|| - log||lambda_G|| - log rho), enforcing ||lambda_T|| <= rho ||lambda_G||",
variables: [
{ symbol: "lambda_T", meaning: "tracking-mode attention gate vector" },
{ symbol: "lambda_G", meaning: "grounding-mode attention gate vector" },
{ symbol: "rho", meaning: "decay factor in (0,1), default 0.6" },
{ symbol: "L_ratio", meaning: "regularizer penalizing overly strong tracking-mode injection" },
],
intuition: "Force the tracking seasoning to stay a fixed fraction of the grounding seasoning, so text can never take over when vision should lead.",
why: "A relative constraint stable across layers and samples; removing it drops OTB99 by 3.1 and grounding by 3.4 points.",
where: "Training objective with weight 0.01; rho 0.6 beats 0.5 (too strict) and 0.7 (drift-prone).",
params: "Smaller rho suppresses useful guidance and hurts NL initialization; larger rho admits language-dominated drift.",
paperIds: ["T147"],
},
{
id: "lvtrack-memory",
label: "Appearance memory fusion (Eqs. 6-7)",
formula: "F_hat = lambda_mem * MHCA(F_tilde_s, M, M); F_out = F_tilde_s + MLP(F_hat); update M by RoI-Align only if confidence > tau_mem",
variables: [
{ symbol: "F_tilde_s", meaning: "MGFI-enhanced search feature" },
{ symbol: "M", meaning: "K memory tokens from historical predicted boxes" },
{ symbol: "lambda_mem", meaning: "learnable channel gate from zero init" },
{ symbol: "tau_mem", meaning: "confidence threshold 0.03 gating memory refresh" },
],
intuition: "Remind the tracker what the target looked like a moment ago so deformation and occlusion do not erase it.",
why: "Carries object-specific appearance across frames; removal costs 1.4 LaSOT and 3.0 OTB99 AUC on long-term sequences.",
where: "Encoder input after MGFI; memory built by RoI-Align over past feature maps with 64 tokens.",
paperIds: ["T147"],
},
{
id: "lvtrack-gskl",
label: "Gaussian-smoothed KL localization loss (Eqs. 9-11)",
formula: "T_i(k) = exp(-(k-y_i)^2/2sig^2)/SUM_j exp(-(j-y_i)^2/2sig^2); L_loc = SUM_i KL(T_i||P_i); L = L_loc + lambda_ratio L_ratio",
variables: [
{ symbol: "y_i", meaning: "ground-truth bin of coordinate head i in {x,y,w,h}" },
{ symbol: "T_i(k)", meaning: "Gaussian soft target mass on bin k" },
{ symbol: "sig", meaning: "smoothing bandwidth, default 4" },
{ symbol: "P_i", meaning: "predicted bin distribution" },
{ symbol: "BINS", meaning: "discretization resolution, 4000" },
],
intuition: "Give partial credit to almost-right coordinates instead of treating a one-pixel miss like a total failure.",
why: "Injects spatial ordinality into token classification; beats cross-entropy by 2.3 LaSOT and 3.7 OTB99 points with faster convergence.",
where: "Decoder supervision; sigma 4 balances locality against sharpness, sigma 2 and 6 both degrade.",
params: "Small sigma approaches cross-entropy harshness; large sigma blurs fine-grained gradients.",
paperIds: ["T147"],
},
],
datasets: ["lasot", "otb", "got10k", "others"],
metrics: ["success-auc", "precision", "norm-precision"],
baselines: ["TNL2K-I", "TNL2K-II", "JointNLT", "QueryNLT", "UVLTrack-B", "DUTrack-256", "DUTrack-384", "MambaVLT", "SAVLT-B", "CTRNLT", "VLTVG"],
results: [
"NL-only: LVTrack-384 best TNL2K AUC 58.7 and Prec 59.9, best OTB99 63.6 AUC, 83.0 Prec, 76.3 NPrec, second LaSOT 61.3 AUC, 64.5 Prec, 71.2 NPrec; LVTrack-256 second on OTB99 at 63.3/82.9/75.9.",
"NL+BBOX: LVTrack-384 best TNL2K AUC 66.5 and OTB99 72.4 AUC with 88.9 NPrec and 95.3 Prec, LaSOT 72.7 AUC; LVTrack-256 LaSOT 69.6 and OTB99 71.7.",
"RefCOCOg grounding 74.88 accuracy with the same unified tracking model, ahead of UVLTrack 73.86 and VLTVG 72.98.",
"Efficiency: 100M trainable params versus 129-169M baselines, 80 versus 300 epochs, 30K samples per epoch, about 2 hours on 8 RTX 4090, 40 FPS at 256 and 22 FPS at 384 resolution.",
"Ablations (LVTrack-256 NL): removing GS-KL costs 2.3/3.7 LaSOT/OTB99 AUC, dual-mode fusion 1.0/4.6, memory 1.4/3.0, 2D RoPE 3.4/5.2; ViT layer 10 beats layers 9 and 11 on all three benchmarks.",
],
ablations: [
"GS-KL versus cross-entropy training curves show higher asymptote plus faster early convergence on both grounding and LaSOT tracking.",
"Amplitude ratio loss and dual-mode fusion ablations isolate drift suppression from grounding preservation; MGFI heatmaps sharpen from diffuse saliency to target across layers 5 to 10.",
"Sampling ratio 0.45/0.40/0.15 beats grounding-light (hurts NL init) and grounding-heavy 0.25 (biases toward language, drops LaSOT 58.7 to 57.6) variants.",
"Backbone study: naive PE feature-concatenation baselines degrade, confirming MGFI rather than the backbone alone drives the gains.",
],
limitations: {
authorStated: [
"Only fixed input resolutions (256 and 384 variants) are implemented; arbitrary-resolution tracking is unexplored.",
"Language guidance is studied only at initialization; dynamically updated user instructions or mid-tracking constraints are not handled.",
],
evident: [
"Frozen PE backbone plus directive-stripping and Hanning-window heuristics carry robustness load without learned alternatives.",
"NL+BBOX LaSOT still trails specialized DUTrack (72.7 versus 74.1 AUC), showing the unified-mode compromise.",
"Memory refresh and directive stripping depend on hand-set thresholds that may not transfer across expression styles.",
],
},
assumptions: [
"Referring expression describes the first frame reliably enough for grounding-mode initialization.",
"Template-only tracking samples suffice to counteract language dominance during unified training.",
"Intermediate ViT layer 10 balances semantics and spatial fidelity for all benchmarks.",
],
computation:
"About 2 hours on 8 NVIDIA RTX 4090 GPUs in bfloat16; inference about 40 FPS for LVTrack-256 and 22 FPS for LVTrack-384.",
relations: [
{ to: "T070", type: "conceptual-successor", note: "Extends MixFormer one-stream joint template-search modeling with language-conditioned injection." },
{ to: "T024", type: "conceptual-successor", note: "Evaluates NL and NL+BBOX protocols on LaSOT and trains on its splits." },
{ to: "T025", type: "conceptual-successor", note: "Trains on GOT-10k and reports AO/SR under its one-shot protocol." },
],
concepts: ["multimodal", "sot", "attention", "memory-network", "success-plot", "benchmark-design", "siamese"],
impact:
"Shows frozen VLP models transfer to referring tracking with tiny trainable cost once injection strength is explicitly mode-calibrated, reframing drift control as perturbation budgeting.",
},
{
id: "T148",
arxiv: "2609.00924",
title: "Beyond the Image Plane: World-Grounded Queries for Multi-Object Tracking",
shortTitle: "PLANET",
year: 2026,
authors: ["Orcun Cetintas", "Guillem Braso", "Tim Meinhardt", "Laura Leal-Taixe"],
fileName: "2609.00924v1.pdf",
task: "multi-object",
tags: ["multi-object", "end-to-end", "world-grounded-queries", "3d-geometry", "query-based", "temporal-memory", "transformer"],
difficulty: "advanced",
summary:
"PLANET lifts monocular 2D tracking datasets into 3D with Depth Anything 3 streaming reconstruction, then grounds a MOTIP-style end-to-end tracker in that geometry: geometric query lifting fuses dense point-map features and positions into detector queries, an anchored 3D head refines the lifted 2D reference point with l1 supervision on pseudo-3D centers, and dual-resolution memory stretches identity context from 30 to about 200 frames. It sets new state of the art on DanceTrack (72.1 HOTA), SportsMOT (73.7) and BFT (72.6) without external training data.",
problem:
"Monocular trackers reason from appearance and image-plane position where distinct objects overlap and physical motion is ambiguous, while prior 3D aids rely on category-specific models or handcrafted post-hoc association; meanwhile end-to-end query trackers limit identity context to tens of frames, discarding evidence needed to survive long occlusions.",
background: ["mot", "end-to-end-mot", "attention", "3d-tracking", "hungarian", "data-association", "hota", "memory-network"],
previousWork: [
{
name: "MOTIP shared-query ID prediction",
limitation:
"Detection and identity heads share detector queries but queries encode only image-plane evidence within a 30-frame context, so occlusion ambiguities persist.",
whyThisPaper:
"PLANET keeps the MOTIP formulation and Deformable DETR plus ResNet-50 setup while grounding the shared queries in reconstructed 3D and extending memory to 200 frames.",
},
{
name: "3D-cue trackers (visibility reasoning, world-space trajectories, 3D human models, depth-motion heuristics, 2D-3D filters)",
limitation:
"Geometry enters through specialized category models, separate forecasting modules or handcrafted association costs after representations are formed.",
whyThisPaper:
"Injects dense geometry into query formation itself and supervises object-level 3D centers, so one representation serves both detection and association end to end.",
},
{
name: "Tracking-by-detection association (SORT, ByteTrack, OC-SORT, StrongSORT, BoT-SORT, QDTrack)",
limitation:
"Links per-frame detections with image-plane motion, overlap and appearance that collapse under uniform appearance, dance crossovers and dense flocks.",
whyThisPaper:
"Beats all of them on DanceTrack, SportsMOT and BFT by reasoning in a world-grounded query space with long temporal support.",
},
],
researchGap:
"No end-to-end monocular tracker embedded reconstructed scene geometry directly into query formation with object-level 3D supervision while preserving that evidence across long temporal gaps.",
contribution: [
"Dataset lifting pipeline (DA3-Streaming with overlapping windows and loop closure) producing dense world-coordinate point maps, camera parameters and per-box pseudo-3D centers with invalid-marking for uncertain occluded estimates.",
"Geometric query lifting: point maps encoded by geometric feature and position heads, residually added to visual features and fused with 2D sine-cosine encodings for deformable attention.",
"Anchored 3D localization: lifts the pretrained 2D reference point into 3D and predicts a query-conditioned residual supervised by an auxiliary l1 loss on valid pseudo-centers.",
"Dual-resolution temporal memory: fixed context budget split into dense recent frames plus strided long-term samples with varied training sampling rates, reaching about 200 frames with no extra training time.",
"State of the art on three diverse test sets without external data: DanceTrack 72.1, SportsMOT 73.7 and BFT 72.6 HOTA, each ahead of MOTIP with ablations isolating every component.",
],
method: {
pipeline: ["reconstruct-3d", "assign-3d-labels", "geometric-lift", "detect-and-localize-3d", "dual-memory-associate", "link-trajectories"],
architecture:
"MOTIP-on-Deformable-DETR with ResNet-50 from COCO weights: backbone 2D features plus geometric feature embeddings feed deformable attention with fused 2D/3D positional encodings; each query predicts a 2D box, an anchored 3D center and a (K+1)-way identity label.",
motionModel: "None explicit — motion evidence is implicit in world-grounded query embeddings matched across the dual-resolution temporal memory.",
appearanceModel:
"ResNet-50 appearance features residually enriched with local 3D structure embeddings from dense point maps; object queries accumulate scene plus object-level 3D information.",
association:
"In-context (K+1)-way identity classification over a learnable ID dictionary, attending to dense short-term plus sparse long-term query history; trajectories link equal predicted identities.",
detectionDependency:
"End-to-end detector trained per benchmark from COCO initialization; no external detector or CrowdHuman-style extra data, unlike several compared methods.",
trackManagement: "Identity labels emitted per query per frame and linked across frames; no handcrafted birth/death rules described.",
optimization:
"AdamW with learning rate 1e-4 and weight decay 5e-4 following MOTIP, with random resizing, cropping, flipping, trajectory occlusion and identity-switch augmentation; each ablation trained five times and averaged.",
loss:
"Standard MOTIP detection plus identity objectives with an added l1 term on predicted 3D centers for valid pseudo-labels only.",
},
equations: [
{
id: "planet-geometric-features",
label: "Geometric feature residual fusion",
formula: "F_e = F_2D + F_3D where F_3D = L_F(P)",
variables: [
{ symbol: "P", meaning: "aligned dense world-coordinate point map of the frame" },
{ symbol: "L_F", meaning: "learnable geometric feature head" },
{ symbol: "F_3D", meaning: "geometric feature embeddings from scene structure" },
{ symbol: "F_2D", meaning: "backbone visual features" },
{ symbol: "F_e", meaning: "geometry-conditioned features used by deformable attention" },
],
intuition: "Add a layer of where-things-are-in-the-world on top of what-they-look-like without destroying pretrained appearance.",
why: "Lets local 3D structure complement appearance during query updating rather than as a post-hoc cost.",
where: "Query formation; lightweight added heads keep runtime near MOTIP at 12 FPS FP32 and 20 FPS FP16.",
paperIds: ["T148"],
},
{
id: "planet-geometric-positions",
label: "Geometric position embedding fusion",
formula: "E_e = M([E_2D || E_3D]) where E_3D = L_P(P)",
variables: [
{ symbol: "E_2D", meaning: "sine-cosine 2D positional encoding" },
{ symbol: "L_P", meaning: "learnable geometric position head" },
{ symbol: "E_3D", meaning: "geometric position embeddings from the point map" },
{ symbol: "M", meaning: "learned fusion head over concatenated encodings" },
{ symbol: "E_e", meaning: "geometry-augmented positional embeddings for attention" },
],
intuition: "Tell each pixel not only its image address but also its room address in the reconstructed world.",
why: "Augments image-plane position with scene geometry so overlapping projections become separable in attention.",
where: "Deformable attention positional encodings alongside the fused features.",
paperIds: ["T148"],
},
{
id: "planet-anchored-3d",
label: "Anchored 3D center refinement",
formula: "r_3D = lift(r_2D, P); c_hat = r_3D + head_3D(q); L_3D = |c_hat - c|_1 over valid labels",
variables: [
{ symbol: "r_2D", meaning: "detector pretrained 2D reference point of the query" },
{ symbol: "lift", meaning: "lookup of the 2D reference in the aligned scene geometry" },
{ symbol: "r_3D", meaning: "anchored 3D reference point" },
{ symbol: "q", meaning: "matched object query embedding" },
{ symbol: "c_hat, c", meaning: "predicted and pseudo-label 3D object centers (position only, no size or orientation)" },
],
intuition: "Start from the trusted 2D guess, look up where that ray hits the reconstructed world, then nudge to the object center.",
why: "Reuses large-scale 2D localization priors instead of learning absolute 3D from scratch; naive direct 3D drops HOTA 64.0 to 63.8 while anchored gains to 64.4.",
where: "Extra lightweight 3D head with iterative refinement; invalid estimates excluded from supervision.",
paperIds: ["T148"],
},
{
id: "planet-dual-memory",
label: "Dual-resolution temporal memory budget",
formula: "M slots = M/2 dense recent frames + M/2 earlier frames at stride S; horizon about 200 versus 30 consecutive frames",
variables: [
{ symbol: "M", meaning: "fixed identity-decoder context budget in query observations" },
{ symbol: "S", meaning: "sparse sampling stride for long-term history" },
{ symbol: "horizon", meaning: "temporal span reachable by identity reasoning" },
],
intuition: "Remember yesterday in headlines and today in detail so old identity evidence survives without paying for every frame.",
why: "Makes world-grounded cues available across long gaps; adds 1.2 HOTA and 2.5 IDF1 on the 3D-aware model with detection accuracy unchanged.",
where: "Identity decoder context at inference with temporally varied sampling during training; stored detector query embeddings unchanged.",
paperIds: ["T148"],
},
],
datasets: ["dancetrack", "others"],
metrics: ["hota", "deta", "assa", "idf1", "mota", "fps"],
baselines: ["MOTIP", "ByteTrack", "OC-SORT", "SORT", "StrongSORT", "BoT-SORT", "DeepSORT", "CenterTrack", "FairMOT", "QDTrack", "TransTrack", "TrackFormer", "MOTR", "MeMOTR", "SparseTrack", "Hybrid-SORT", "DiffMOT", "TrackTrack", "SambaMOTR", "CO-MOT", "LA-MOTR", "GTR"],
results: [
"DanceTrack test: 72.1 HOTA, 63.9 AssA, 81.4 DetA, 78.0 IDF1, 91.6 MOTA, +2.5 HOTA over MOTIP 69.6 with every metric improved.",
"SportsMOT test: 73.7 HOTA, 63.2 AssA, 86.0 DetA, 78.7 IDF1, 95.2 MOTA, +1.1 HOTA over MOTIP 72.6 with DetA up from 83.5.",
"BFT test: 72.6 HOTA, 73.9 AssA, 71.6 DetA, 84.6 IDF1, 80.4 MOTA, +2.1 HOTA over prior best MOTIP 70.5 across all five metrics.",
"Ablations: lifting alone 63.0 to 64.0 HOTA, anchored localization to 64.4 with association-led gains, dual memory to 65.6 HOTA and 70.3 IDF1; memory helps the 3D model (+1.2/+2.5) more than the 2D one (+0.8/+1.5); raw 3D foundation-feature fusion only reaches 63.1.",
"Runtime comparable to MOTIP at 12 FPS FP32 and 20 FPS FP16 with lightweight geometric heads; ablations averaged over five runs.",
],
ablations: [
"Cumulative component table isolates lifting, anchored localization and memory with detection accuracy flat at the memory step, attributing late gains to association.",
"Direct versus anchored 3D localization: direct supervision hurts (63.8), anchored helps (64.4), proving integration design matters more than the labels alone.",
"3D foundation-feature injection baseline stalls at 63.1 HOTA with unchanged association, showing generic fusion cannot substitute query grounding.",
],
limitations: {
authorStated: [
"3D supervision covers only estimated object centers without dimensions or orientation, and uncertain occluded estimates are marked invalid and excluded from training.",
"Released 3D-enriched datasets, code and models are promised upon acceptance rather than available with the preprint.",
],
evident: [
"Tracking quality inherits Depth Anything 3 reconstruction errors, especially under fast dance motion, flocks and moving broadcast cameras.",
"Dataset-specific per-benchmark training without external data aids fairness but each model is confined to its own domain.",
"Pseudo-center assignment heuristics (visibility gating, temporal depth, trajectory smoothing) are described procedurally without ablated sensitivity.",
],
},
assumptions: [
"Overlapping-window DA3-Streaming with loop closure yields a drift-bounded world frame per sequence.",
"Box-averaged local point clouds of visible boxes approximate true object centers well enough for auxiliary supervision.",
"Fixed half-dense half-strided memory split suits short and long gaps across all three domains.",
],
computation:
"Same detector family and MOTIP training recipe (AdamW 1e-4); inference 12 FPS in FP32 and 20 FPS in FP16, comparable to MOTIP.",
relations: [
{ to: "T059", type: "builds-on", note: "Extends the MOTR query-based end-to-end formulation via the MOTIP shared-query interface." },
{ to: "T053", type: "conceptual-successor", note: "Continues the TransTrack learned query lineage for joint detection and tracking." },
{ to: "T054", type: "conceptual-successor", note: "Continues the TrackFormer track-query paradigm with longer grounded memory." },
{ to: "T061", type: "uses-as-baseline", note: "Beats ByteTrack on all three benchmarks (DanceTrack 47.7, SportsMOT 62.8, BFT 62.5 HOTA)." },
{ to: "T073", type: "uses-as-baseline", note: "Beats OC-SORT on all three benchmarks (55.1, 68.1, 66.8 HOTA)." },
{ to: "T075", type: "uses-as-baseline", note: "Beats BoT-SORT on SportsMOT (68.7 HOTA) as appearance-based reference." },
{ to: "T063", type: "conceptual-successor", note: "Lifts DanceTrack into 3D and advances its test state of the art under uniform-appearance motion." },
],
concepts: ["mot", "end-to-end-mot", "attention", "3d-tracking", "data-association", "memory-network", "hota", "benchmark-design"],
impact:
"Establishes reconstructed 3D scene geometry as first-class query evidence for 2D tracking and releases 3D-enriched benchmarks, pointing future MOT beyond image-plane association.",
},
{
id: "T149",
arxiv: "2609.02318",
title: "YesTrack: Referring Multi-Object Tracking via MLLM-based Yes/No Verification",
shortTitle: "YesTrack",
year: 2026,
authors: ["Quansheng Hu", "Qin Sun", "Qiansen Dai", "Jin Ding", "Wan Zhang", "Xue Zhou", "Jianxiao Zou"],
fileName: "2609.02318v1.pdf",
task: "multimodal-tracking",
tags: ["referring-tracking", "multi-object", "mllm", "yes-no-verification", "discriminative", "two-stage", "temporal-consistency", "multimodal"],
difficulty: "intermediate",
summary:
"YesTrack recasts referring MOT from caption generation into discriminative verification: a Qwen3-VL-2B model answers Yes or No per candidate trajectory via softmax over the two decision-token logits in one forward pass, with a frame mode for cheap scoring, a video mode with memory bank for ambiguous cases, plus Temporal Confidence Prior stabilization and Temporal Reference Propagation that skips verification on non-key frames. It reaches 54.00 HOTA on Refer-KITTI and 43.75 on Refer-KITTI-V2, and the same paradigm instantiated as YesTrack-MOT pairwise identity verifier hits 45.10 HOTA on KITTI MOT.",
problem:
"MLLM-based referring trackers use models as autoregressive caption generators plus external text-similarity modules, adding decoding latency and underusing built-in vision-language alignment, while frame-wise verification is unstable under occlusion and redundant across temporally stable identities.",
background: ["mot", "multimodal", "attention", "reid", "data-association", "hungarian", "track-management", "hota"],
previousWork: [
{
name: "End-to-end RMOT (TransRMOT, TempRMOT, DKGTrack, HFF-Tracker)",
limitation:
"Task-specific multimodal fusion trained on small-vocabulary expression sets generalizes poorly to diverse real-world phrasing including noisy or misspelled input.",
whyThisPaper:
"Offloads language understanding to a pretrained MLLM verifier, showing robustness to misspellings and colloquial fillers where DKGTrack and ReferGPT lose the target.",
},
{
name: "ReferGPT generative two-stage referring",
limitation:
"Autoregressive captioning plus a separate similarity module costs sequential decoding latency and extra matching machinery per candidate.",
whyThisPaper:
"Replaces generation with single-forward Yes/No logit softmax, needing no decoder loop and no external decision module.",
},
{
name: "Embedding-based ReID association (FastReID in BoT-SORT and TrackTrack)",
limitation:
"Hand-trained appearance embeddings decide association independent of language-capable verification.",
whyThisPaper:
"YesTrack-MOT swaps embeddings for MLLM pairwise image-image verification with distance gating and Hungarian matching, topping KITTI MOT at 45.10 HOTA.",
},
],
researchGap:
"No RMOT framework exploited MLLMs directly as single-pass discriminative verifiers with temporal consistency handling, nor showed the same verifier paradigm transferring to generic MOT association.",
contribution: [
"Discriminative referring paradigm: binary image-text matching via Yes/No decision-token logits with BCE training, one forward pass per candidate and continuous confidence for routing.",
"Two-stage verifier with frame mode plus video mode over a 4-frame memory bank, routing by confidence interval [0.2, 0.8] and final threshold 0.4.",
"Temporal Confidence Prior (window 3, alpha 0.4, lambda 0.3) stabilizing decisions from identity history without architecture change.",
"Temporal Reference Propagation triggering verification at interval Delta (5/10), new identities or expression changes and inheriting scores otherwise, cutting inference from 94 to 25 minutes.",
"YesTrack-MOT: minimal MOT pipeline of distance gating (200 px), MLLM pairwise verification, Hungarian assignment and 10-frame lost-track re-association, plus tracker-agnostic referential metrics (accuracy, precision, recall on ground-truth tracklets).",
],
method: {
pipeline: ["track-candidates", "trp-key-select", "frame-verify", "tcp-stabilize", "route-uncertain", "video-refine", "propagate-scores"],
architecture:
"Off-the-shelf tracker (TempRMOT without language, YesTrack-MOT or ByteTrack) proposing crops, IDs and normalized boxes, followed by a Qwen3-VL-2B-Instruct verifier on 320px crops with coordinate-augmented Yes/No prompts in frame and video modes.",
motionModel: "None learned — YesTrack-MOT uses only center-displacement gating within 200 pixels before verification.",
appearanceModel:
"MLLM joint vision-language alignment itself is the appearance and semantic model; crops plus spatial coordinates embedded in the prompt supply location cues.",
association:
"RMOT: per-identity referring scores filter tracker trajectories. MOT: verification probabilities form a cost matrix solved by Hungarian matching with lost-track buffering.",
detectionDependency:
"Two-stage by design: inherits detections and tracks from the chosen backbone (RF-DETR detections standardized for KITTI MOT comparison); quality bounds final scores.",
trackManagement:
"TRP propagates key-frame scores to active identities until the next trigger; YesTrack-MOT keeps lost tracks 10 frames for re-association before deletion.",
},
equations: [
{
id: "yestrack-yesno",
label: "Yes/No matching probability (Eq. 1)",
formula: "p_i = exp(l_yes_i) / (exp(l_yes_i) + exp(l_no_i))",
variables: [
{ symbol: "l_yes_i, l_no_i", meaning: "MLLM projected logits of the Yes and No decision tokens for candidate i" },
{ symbol: "p_i", meaning: "continuous matching confidence in [0,1]" },
{ symbol: "I_i, E", meaning: "target crop and referring expression inputs" },
],
intuition: "Ask the model a closed question and read its hesitation directly from the two answer scores instead of waiting for an essay.",
why: "Avoids open-ended decoding instability and yields confidence for threshold routing and temporal refinement in one pass.",
where: "Frame-mode verifier per candidate; video mode reuses the same extraction over clips.",
paperIds: ["T149"],
},
{
id: "yestrack-bce",
label: "Binary verification training loss (Eq. 2)",
formula: "L_i = -[y_i log p_i + (1-y_i) log(1-p_i)] with softmax restricted to the two decision tokens",
variables: [
{ symbol: "y_i", meaning: "binary ground-truth match label, 1 for positive" },
{ symbol: "p_i", meaning: "Yes probability from the decision-token softmax" },
{ symbol: "L_i", meaning: "per-pair binary cross-entropy objective" },
],
intuition: "Reward confident Yes on true targets and confident No on distractors, with calibration preserved for later thresholding.",
why: "Trains the closed-set decision boundary directly instead of caption quality plus a proxy similarity.",
where: "Verifier fine-tuning on referring pairs and identity-verification pairs.",
paperIds: ["T149"],
},
{
id: "yestrack-tcp",
label: "Temporal Confidence Prior (Eq. 3)",
formula: "p_tilde_i(t) = min(1, p_i(t) + lambda * 1(min over k in [t-K,t-1] p_i(k) >= alpha))",
variables: [
{ symbol: "p_i(t)", meaning: "current-frame MLLM matching probability" },
{ symbol: "K", meaning: "temporal window size, default 3" },
{ symbol: "alpha", meaning: "high-confidence threshold, default 0.4" },
{ symbol: "lambda", meaning: "prior strength, default 0.3" },
{ symbol: "p_tilde_i(t)", meaning: "history-adjusted probability clipped to [0,1]" },
],
intuition: "If an identity confidently matched for the last several frames, give it the benefit of the doubt through a brief wobble.",
why: "Stabilizes trajectory selection under occlusion without touching the model; ablation top HOTA 54.00 and DetA 43.91.",
where: "Inference-time regularization after frame-mode scoring, before confidence routing.",
simulator: "track-mgmt",
paperIds: ["T149"],
},
{
id: "yestrack-gating",
label: "Distance-based candidate gating (Eq. 4)",
formula: "g(tau_j, d_k) = 1(||c(tau_j) - c(d_k)||_2 <= delta)",
variables: [
{ symbol: "tau_j", meaning: "track j from the previous frame" },
{ symbol: "d_k", meaning: "detection k in the current frame" },
{ symbol: "c(.)", meaning: "bounding-box center" },
{ symbol: "delta", meaning: "gating threshold, 200 pixels" },
],
intuition: "Do not ask the expensive language model about pairs that are physically implausible jumps.",
why: "Prunes verification workload so the MLLM only judges plausible continuations.",
where: "YesTrack-MOT association front end before pairwise verification.",
params: "Larger delta admits faster motion at quadratic verification cost; smaller delta risks dropping true fast matches.",
simulator: "motion",
paperIds: ["T149"],
},
{
id: "yestrack-mot-verify",
label: "Pairwise identity verification and Hungarian assignment (Eqs. 5-6)",
formula: "p_jk = exp(l_yes_jk)/(exp(l_yes_jk)+exp(l_no_jk)); c_jk = 1 - p_jk; pi* = argmin_pi SUM c_jk",
variables: [
{ symbol: "p_jk", meaning: "MLLM same-identity probability for track crop versus detection crop" },
{ symbol: "c_jk", meaning: "assignment cost derived from verification confidence" },
{ symbol: "pi*", meaning: "optimal one-to-one track-detection matching" },
],
intuition: "Show the model two mugshots, convert its Yes-confidence into a price tag, then buy the cheapest consistent matching.",
why: "Replaces embedding cosine distance with direct verification, reaching best KITTI MOT HOTA with a minimal pipeline.",
where: "YesTrack-MOT association core; matched pairs update tracks, unmatched become lost or new.",
simulator: "hungarian",
paperIds: ["T149"],
},
],
datasets: ["kitti-tracking", "others"],
metrics: ["hota", "deta", "assa", "idf1"],
baselines: ["TransRMOT", "TempRMOT", "DKGTrack", "iKUN", "ReferGPT", "EchoTrack", "DeepRMOT", "TenRMOT", "ByteTrack", "BoT-SORT", "OC-SORT", "TrackTrack", "MGLT-MOTR", "MLS-Track", "CDRMT"],
results: [
"Refer-KITTI: YesTrack with TempRMOT tracker 54.00 HOTA, 43.91 DetA, 66.57 AssA (best overall and association); with YesTrack-MOT 52.96 HOTA, 46.84 DetA (best detection) and 60.03 AssA; referential accuracy 91.15, precision 74.14, recall 85.71 versus iKUN 84.62, 59.88, 70.24.",
"Refer-KITTI-V2: YesTrack-MOT variant best 43.75 HOTA and 37.04 DetA with 48.78 DetRe; TempRMOT variant 41.78 HOTA and 32.69 DetA, far above ReferGPT 30.12 and iKUN 10.32.",
"KITTI MOT (RF-DETR detections, car and person): YesTrack-MOT 45.10 HOTA and 40.96 DetA, ahead of OC-SORT 41.34, ByteTrack 40.83 and TrackTrack 40.24.",
"Ablation on Refer-KITTI: frame-only 52.64 HOTA in 40 min, video-only 53.49 in 135 min, combined 53.47 in 92 min, plus TCP 54.00 in 94 min, plus TRP 53.54 in 22 min, full 53.66 HOTA with best 66.71 AssA in 25 min.",
"Consistent gains over iKUN across ByteTrack, TempRMOT, YesTrack-MOT and ground-truth trackers, e.g. 54.00 versus 42.95 HOTA on TempRMOT tracks.",
],
ablations: [
"Frame and video modes complement: video-only improves association at 3x latency, combined balances detection and association.",
"TCP gives the best HOTA/DetA while TRP trades a small association dip for a 4x speedup; together they give the best AssA near TRP-only runtime.",
"Referential metrics on ground-truth tracklets isolate language ability from tracker quality and favor YesTrack on all four backbones.",
],
limitations: {
authorStated: [
"Two-stage design inherits off-the-shelf tracker errors; wrong verification or tracking at a key frame propagates through TRP until re-evaluation, causing temporary loss or distractor tracking.",
"Future work should replace heuristic TCP with a learnable module and pursue tighter coupling for challenging dynamics.",
],
evident: [
"All referring results use the smallest Qwen3-VL-2B model, so headroom from larger MLLMs is untested within the reported numbers.",
"Fixed choices (interval 5/10, memory 4, thresholds 0.2/0.8 and 0.4) are set once and may not suit dense or long-occlusion scenes.",
"Inference remains minutes per sequence even with TRP, far from the real-time edge speeds of non-MLLM trackers.",
],
},
assumptions: [
"Referring relevance of an identity is stable within the propagation interval unless a new identity or expression change triggers re-verification.",
"All previous K frames confidently matched implies current relevance, formalized by the strict inner-minimum in TCP.",
"Cropped regions plus coordinate prompts suffice for the MLLM to judge identity and expression match.",
],
computation:
"Qwen3-VL-2B-Instruct backbone with 320px crops; all experiments on two NVIDIA RTX 4090 24GB GPUs; ablation runtimes 22 to 135 minutes on Refer-KITTI.",
relations: [
{ to: "T061", type: "uses-as-baseline", note: "Evaluates referring over ByteTrack tracks and standardizes on it for tracker-agnostic comparison." },
{ to: "T075", type: "uses-as-baseline", note: "Compares YesTrack-MOT against BoT-SORT with FastReID on KITTI MOT (38.92 HOTA)." },
{ to: "T073", type: "uses-as-baseline", note: "Compares YesTrack-MOT against OC-SORT on KITTI MOT (41.34 HOTA)." },
{ to: "T005", type: "conceptual-successor", note: "Minimal distance-gating plus Hungarian pipeline descends from the SORT tracking-by-detection skeleton." },
{ to: "T013", type: "conceptual-successor", note: "Replaces DeepSORT-style embedding ReID with direct MLLM pairwise verification." },
{ to: "T059", type: "conceptual-successor", note: "Contrasts MOTR end-to-end lineage with a decoupled track-then-verify referring design." },
],
concepts: ["mot", "multimodal", "attention", "reid", "data-association", "hungarian", "track-management", "hota"],
impact:
"Reframes MLLMs in tracking from caption generators to single-pass decision heads, with temporal propagation and prior tricks that generalize the verifier idea from referring to generic association.",
}
];
