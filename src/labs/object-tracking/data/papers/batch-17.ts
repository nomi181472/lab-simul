import type { PaperRecord } from "../types";

/* batch-17 — T150..T159 */

export const BATCH_17: PaperRecord[] = [
{
id: "T150",
arxiv: "2609.06302",
title: "CST-WM: A Causally Structured World Model for Embodied Visual Tracking",
shortTitle: "CST-WM",
year: 2026,
authors: ["Junyi Hu", "Shuaihang Yuan", "Yi Fang"],
fileName: "2609.06302v1.pdf",
task: "single-object",
tags: ["embodied-tracking", "world-model", "model-predictive-control", "causal-structure", "diffusion-dynamics", "target-reacquisition", "cross-entropy-method"],
difficulty: "advanced",
summary:
"CST-WM reframes embodied visual tracking as planning over future target observability and apparent scale, and proposes a causally structured diffusion world model with three branches: target-evidence, robot-motion, and observation. Its transition factorizes so the target-evidence branch is updated without direct action injection, blocking a task-specific causal hallucination where control leaks directly into target evidence, while action still reaches future observations through the robot branch. Combined with CEM rollout-based MPC, one framework handles stable following and temporary target re-acquisition, improving following quality, distance-range control, safety, and re-acquisition on EVT-Bench and Habitat 3.0 plus cross-dataset transfer.",
problem:
"Embodied tracking is a predictive decision problem: the robot must choose actions that preserve or recover future target visibility and following distance under ego-motion, occlusion, and distractors. Reactive trackers are myopic once the target vanishes, while generic action-conditioned world models let the current action write directly into the target-evidence latent, producing plausible futures with wrong planning semantics for re-acquisition.",
background: ["sot", "bounding-box", "motion-model", "state-space", "online-vs-offline", "occlusion", "benchmark-design"],
previousWork: [
{
name: "Reactive embodied trackers (RL active trackers, EVT offline-RL, TrackVLA, Uni-NaVid, vision-language-action variants)",
limitation:
"Map recent observations to actions rather than comparing how alternative actions affect future target observability and apparent scale over multiple steps, so they offer little basis for multi-step recovery once the target disappears.",
whyThisPaper:
"Adds rollout-based MPC over an imagined target-evidence future so following and re-acquisition are planned jointly before acting.",
},
{
name: "Generic action-conditioned world models (latent dynamics, Dreamer family, Navigation World Models diffusion planning)",
limitation:
"A non-factorized transition can inject the current action directly into the target-evidence branch, hallucinating a direct causal effect of action on target evidence instead of routing it through robot motion and observation change.",
whyThisPaper:
"Factorizes dynamics into target-evidence, robot, and observation branches with architectural masking that blocks direct action injection into target evidence.",
},
{
name: "Detection-plus-planning following systems",
limitation:
"Achieve basic following but degrade under occlusion and visual ambiguity in cluttered scenes because target evidence is not predicted forward.",
whyThisPaper:
"Maintains a compact planning-oriented evidence proxy (GroundingDINO confidence plus normalized box area) forecast over the horizon with safety-aware scoring.",
},
],
researchGap:
"No embodied tracking world model enforced a structural constraint aligning the action pathway in prediction with how target evidence actually arises under ego-motion.",
contribution: [
"Formulates embodied tracking as planning over future target evidence and proposes CST-WM with separated target-evidence, robot, and observation branches plus a factorized transition blocking direct action injection.",
"Introduces a compact target-evidence representation (detector confidence plus normalized area as observability and distance proxy) that needs no privileged geometric supervision at test time.",
"Builds rollout-based CEM MPC (T=10, N=128, K=16, M=4) with a planning value combining visibility, distance regulation, action-validity, and safety terms.",
"Provides three-perspective evaluation on EVT-Bench and Habitat 3.0: standard tracking, target-loss recovery, and offline rollout-fidelity, planning-value consistency, and leakage diagnostics.",
],
method: {
pipeline: ["extract-target-evidence", "encode-observation", "integrate-robot-state", "masked-diffusion-transition", "rollout-candidates", "cem-planning", "receding-horizon-execution"],
architecture:
"Frozen VAE visual latent Z plus 2-D target-evidence token H (GroundingDINO confidence, normalized area) plus robot state x form structured state S=[H,Z,x]; 6-block transformer transition (hidden 512, 8 heads) denoises branches in fixed order target-evidence, robot, observation under causal masking; DDPM training (1000 steps, 20 DDIM inference steps) with auxiliary distance-aware head.",
motionModel:
"Robot branch integrates action history under kinematic update x=F(a0:l-1); action a=(v,omega) with linear and angular velocity; observation branch fuses robot-carried action representation R=[x_next, W(a)] with updated target evidence.",
appearanceModel:
"No appearance embedding learned; target evidence is the observation-derived GroundingDINO confidence and box-area proxy, not a full external state estimate.",
association: "Not applicable (single-target embodied control; planning scores rolled-out evidence vectors directly without decoding full images).",
detectionDependency:
"GroundingDINO open-vocabulary detector supplies confidence and box at every step; planning signal inherits detector stability limits.",
trackManagement:
"No track birth/death logic; following vs re-acquisition emerges from one planning value trading visibility reward against distance, validity, and safety penalties.",
loss:
"L = Ldiff + lambda_dist * Ldist with lambda_dist=0.2; DDPM noise-prediction loss plus binary in-range distance cross-entropy on simulator-only relative-distance labels (offline supervision only).",
optimization:
"AdamW (lr 1e-4, wd 1e-4, batch 64, 80 epochs, grad-clip 1.0); 3 seeds; checkpoint selected on validation mean of Following Rate and DRS with CR tiebreak.",
},
equations: [
{
id: "cstwm-evidence",
label: "Target-evidence proxy",
formula: "H_l = [H_vis_l, H_area_l] = [H_GDINO(O_l), S_GDINO / S(O_l)], in [0,1]^2",
variables: [
{ symbol: "O_l", meaning: "egocentric RGB observation at time l" },
{ symbol: "H_GDINO", meaning: "GroundingDINO open-vocabulary detection confidence" },
{ symbol: "S_GDINO / S(O_l)", meaning: "detected box area divided by image area" },
{ symbol: "H_l", meaning: "2-D target-evidence token for observability and scale" },
],
intuition: "Two numbers summarize tracking state: is the target found, and how big does it look (a stand-in for distance).",
why: "Gives the planner a compact signal that works without privileged target state at test time.",
where: "Sec. 3.2.1, Eq. 1; built from every observation before forming the structured state.",
},
{
id: "cstwm-nonleakage",
label: "Non-leakage constraint and factorized transition",
formula: "p(H_{l+1} | H_l, Z_l, x_l, a_l) = p(H_{l+1} | H_l, Z_l, x_l); dH_{l+1}/da_l = 0",
variables: [
{ symbol: "H_l", meaning: "target-evidence branch state" },
{ symbol: "Z_l", meaning: "VAE visual latent" },
{ symbol: "x_l", meaning: "robot state" },
{ symbol: "a_l", meaning: "current action (linear plus angular velocity)" },
],
intuition: "The target-evidence forecast may not peek at the current joystick command; the command may only move the robot, which then changes what is seen.",
why: "Blocks causal hallucination where control masquerades as target evidence in imagined futures.",
where: "Sec. 3.2.1, Eq. 2; enforced by architectural masking with transition graph a->x->Z and H->Z and no edge a->H.",
},
{
id: "cstwm-factorization",
label: "Factorized one-step transition",
formula: "p(H+, x+, Z+ | H, Z, x, a) = p(H+ | H, Z, x) p(x+ | x, a) p(Z+ | Z, H+, x+)",
variables: [
{ symbol: "H+", meaning: "next target-evidence state" },
{ symbol: "x+", meaning: "next robot state" },
{ symbol: "Z+", meaning: "next observation latent" },
],
intuition: "Update evidence from the scene, move the robot from the action, then render the next view from both.",
why: "Lets action affect future evidence only indirectly through ego-motion and observation change.",
where: "Sec. 3.2.1, Eq. 3; single-layer order target-evidence, robot, observation with R=[x+, W(a)] as sole action carrier.",
simulator: "motion",
},
{
id: "cstwm-diffusion",
label: "Masked diffusion training objective",
formula: "L_diff = E[ lambda(u) * ||eps - eps_Theta(Stilde_u; S_l, a_l)||^2 ]; L = L_diff + 0.2 * L_dist",
variables: [
{ symbol: "S_l", meaning: "structured state [H_l, Z_l, x_l]" },
{ symbol: "Stilde_u", meaning: "noise-corrupted next structured state at diffusion step u" },
{ symbol: "eps", meaning: "sampled Gaussian noise" },
{ symbol: "L_dist", meaning: "binary in-range distance cross-entropy on g_dist(H_{l+1})" },
],
intuition: "Learn to denoise the next whole tracking state while the mask controls which branch may see the action; a small side loss teaches the evidence branch distance awareness.",
why: "Trains one shared denoiser over all branches so rollouts stay coherent over long horizons.",
where: "Sec. 3.2.3, Eqs. 4-5; simulator distance used only as offline supervision, never at test time.",
},
{
id: "cstwm-planning",
label: "MPC planning value and CEM update",
formula: "V = sum_t [beta_vis * Hvis_t - beta_dist * |Harea_t - alpha|] - penalties(valid, safe); mu, Sigma from top-K elites",
variables: [
{ symbol: "Hvis_t", meaning: "predicted visibility component" },
{ symbol: "Harea_t", meaning: "predicted scale component" },
{ symbol: "alpha", meaning: "reference scale 0.5" },
{ symbol: "V", meaning: "accumulated rollout value over horizon T=10" },
],
intuition: "Score each imagined driving sequence by how visible and well-distanced the target stays, minus penalties for illegal or unsafe moves; refit sampling toward the best sequences.",
why: "Unifies stable following and re-acquisition: visible targets earn following reward, lost targets earn recovery reward under the same objective.",
where: "Sec. 3.3, Eqs. 6-8; N=128 candidates, K=16 elites, M=4 iterations, receding-horizon execution.",
},
],
datasets: ["others"],
metrics: [],
baselines: ["Uni-NaVid", "TrackVLA", "TrackVLA++ (single view)", "Habitat 3.0 baseline", "SDA-S2", "adapted Navigation World Model (retrained, same planner)"],
results: [
"EVT-Bench in-domain: Ours SR 88.7, TR 83.4, CR 1.41 vs TrackVLA++ 86.0/81.0/2.10 and TrackVLA 85.1/78.6/1.65 (Tab. 4).",
"Habitat 3.0 standard tracking: Ours F 0.53, DRS 0.70, CR 0.27, ES 0.61 vs adapted NWM 0.41/0.61/0.33/0.49 (Tab. 4).",
"Cross-dataset transfer (train EVT-Bench, test Habitat 3.0): Ours F 0.48, DRS 0.65, CR 0.30, ES 0.54, best of transferable baselines (Tab. 4).",
"Target-loss recovery on EVT-Bench: Ours re-acquisition 0.84/0.69/0.73/0.65 with TTR 6.4/9.8/8.7/10.6 steps across short occlusion, long occlusion, out-of-FOV, distractor crossing; post-following 0.71/0.57/0.60/0.54 (Tab. 5).",
"Planning-value consistency: Ours Spearman rho 0.68, Kendall tau 0.51 vs TrackVLA-style 0.52/0.37 and adapted NWM 0.47/0.33 (Tab. 6).",
"Leakage diagnostics: Jacobian leakage 0.001 overall for Ours vs 0.112 adapted NWM; interventional variance 0.006 and Wasserstein 0.011 vs 0.084/0.127 NWM (Tabs. 7, 10); masking sweep 0.19 (unmasked) to 0.00 (full masking, Tab. 8).",
"Proxy validation: distance-aware loss raises proxy-distance monotonic Spearman from 0.71 to 0.83; full proxy (F 0.53, DRS 0.70) approaches direct-distance oracle (0.54/0.72, Tabs. 11-12).",
],
ablations: [
"W/o action masking: largest degradation in re-acquisition and leakage diagnostic (Tab. 13: F 0.47, Re-acq 0.61, plan-rho 0.55).",
"W/o branch factorization (single-branch): F 0.49, Re-acq 0.64, plan-rho 0.58 (Tab. 13).",
"W/o distance-aware loss: mainly harms DRS (0.70 to 0.65) while Re-acq stays 0.72 (Tab. 13).",
"Deterministic rollout: weakens long-horizon recovery (Re-acq 0.68 vs 0.76 full, Tab. 13).",
"Partial masking / shuffled update order underperform full masked order (Tab. 14); per-seed F/DRS/CR vary by at most 0.02 (Tab. 15).",
],
limitations: {
authorStated: [
"Detector dependence: GroundingDINO confidence/area proxy becomes unstable under heavy motion blur, low light, extreme viewpoint, or out-of-distribution appearance; factorization cannot recover from a fundamentally unreliable detector.",
"Proxy is not metric distance: area cue saturates at very long range (near zero) and extreme close range; regulation outside the calibrated 1-3 m range is not guaranteed.",
"Simulation-to-real gap: main quantitative claims rest on EVT-Bench and Habitat 3.0 with only qualitative real-world tracking; no large-scale physical-robot statistics.",
"Single-target identity ambiguity: evidence branch summarizes one observability signal and does not maintain persistent identity under prolonged multi-person distractor ambiguity.",
"Privileged training-only supervision: auxiliary distance loss uses simulator-only relative distance unavailable in real-data-only deployments.",
"Compute cost: about 19.5 h per main model on 4x RTX 4090 (about 320 GPU-h reported plus 190 GPU-h preliminary) and about 150 ms per planning step, heavier than reactive policies.",
],
evident: [
"Evaluation protocol fixes planner hyperparameters and interfaces across methods, so gains isolate transition structure but do not show robustness to planner retuning.",
"Offline rollout-fidelity and ranking-consistency diagnostics share starting states and candidate sets, which favors models aligned with the chosen reward rather than open-ended behavior.",
],
},
assumptions: [
"Robot receives only egocentric RGB observations and platform actions at test time; no privileged humanoid state, future trajectory, or oracle distance.",
"Following success defined as 1-3 m distance while facing the target; action bounds and safety clearance (dsafe 0.20 m) matched to simulator limits shared by all baselines.",
"Scene-disjoint train/val/test splits; three random seeds with mean reported.",
],
computation:
"4x NVIDIA RTX 4090, 80 epochs DDPM, batch 64; about 19.5 h per main model, about 320 GPU-h reported plus 190 GPU-h preliminary/ablation; inference about 150 ms per planning step (T=10, N=128, K=16, M=4, batched on 1x RTX 4090); fp16, PyTorch 2.2 / CUDA 12.1.",
relations: [],
concepts: ["sot", "motion-model", "state-space", "occlusion", "online-vs-offline", "benchmark-design"],
impact:
"Argues that for embodied tracking the predictive structure must match how evidence arises under ego-motion, pushing world-model work toward causally masked factorized transitions and leakage diagnostics rather than raw rollout fidelity alone.",
},
{
id: "T151",
arxiv: "2609.07070",
title: "Continuous Token-Level Spatio-Temporal Context Modeling for Visual Object Tracking",
shortTitle: "TLCTrack",
year: 2026,
authors: ["Ding Xia", "Meiqin Liu", "Jing Zhou", "Jian Lan"],
fileName: "2609.07070v1.pdf",
task: "single-object",
tags: ["single-object-tracking", "one-stream", "vision-transformer", "spatio-temporal", "token-update", "salient-token", "masked-attention"],
difficulty: "intermediate",
summary:
"TLCTrack replaces discrete sampling-based template/feature updates with continuous token-level spatio-temporal modeling via salient tokens. A Masked Unidirectional Attention (MUA) backbone fuses long/short-term temporal tokens into search features without letting temporal tokens interact; Spatial Salient Token Collection (SSTC) progressively keeps high-score search tokens to suppress background; the Temporal Salient Token Bank (TSTB) continuously refreshes long- and short-term tokens. TLCTrack384 reaches 77.5% AO on GOT-10k, 60.1% AUC on TNL2K, 70.7% on NFS at 67 fps with 92M params and 58G FLOPs.",
problem:
"One-stream ViT trackers learn from static template-search pairs offline and ignore relations across search frames; existing feature-update and template-update remedies rely on coarse discrete sampling that deviates from the continuity of spatio-temporal context, adds compute, and lets background noise contaminate learning.",
background: ["sot", "bounding-box", "iou", "siamese", "attention", "anchor-free-head", "success-plot"],
previousWork: [
{
name: "Offline one-stream trackers (OSTrack, MixFormer, SimTrack, SwinTrack)",
limitation:
"Formulate tracking as offline template-search matching with no temporal context across search frames, so distractors and appearance change cause drift.",
whyThisPaper:
"Adds continuously updated temporal salient tokens Et fused into search features through unidirectional masked attention.",
},
{
name: "Feature-update methods (ARTrack autoregressive queries, AQATrack spatio-temporal queries)",
limitation:
"Refine state/motion features over time but still sample discretely, missing fine-grained continuous context.",
whyThisPaper:
"MUA integrates temporal tokens token-by-token into every search token while freezing temporal tokens to preserve independence.",
},
{
name: "Template-update methods (STARK score-gated refresh, SeqTrack sequence decoding, DropTrack)",
limitation:
"Maintain sparse online templates that coarsely capture appearance change and incur update cost plus background contamination.",
whyThisPaper:
"Replaces templates with SSTC-collected spatial salient tokens and a TSTB long/short-term bank updated by importance scores every frame.",
},
],
researchGap:
"No tracker modeled continuously refined token-level spatio-temporal context without discrete template sampling while explicitly suppressing background at the token level.",
contribution: [
"TLCTrack framework with three modules: MUA for unidirectional temporal-to-search fusion, SSTC for background-suppressed spatial token collection, TSTB for continuous long/short-term temporal token refresh.",
"MUA masking rules letting template self-attend, search attend to all, and temporal tokens stay frozen, preserving temporal independence.",
"SSTC with learnable queries, adapter plus auxiliary head (stop-gradient decoupled) and keeping ratio rho=0.8 for discriminative spatial tokens.",
"Two scaled variants (256/384) with state-of-the-art results on five benchmarks plus efficiency analysis (28-58G FLOPs, 67-97 fps on RTX 3090).",
],
method: {
pipeline: ["patch-embed", "masked-unidirectional-attention", "spatial-salient-collection", "temporal-bank-update", "conv-head-predict"],
architecture:
"ViT-Base MAE-pretrained one-stream backbone with MUA inserted; SSTC at blocks 4/7/10 (rho 0.8); TSTB concatenates long-term Elt and short-term Est into Et; prediction and auxiliary heads each with three Conv branches (score map, offset, box size).",
motionModel: "Not applicable — no explicit filter; temporal continuity comes from continuously refreshed salient tokens.",
appearanceModel:
"Target-aware search features guided by temporal tokens; SSTC TopK token selection by learned importance suppresses background before feature learning.",
association: "Not applicable (single-object; argmax over classification score map with box regression).",
detectionDependency: "None — first-frame template given; template 128/192 px, search 256/384 px for the two variants.",
trackManagement:
"No template refresh heuristic; Et initialized from template tokens then updated every frame via importance TopK on prediction-score-weighted attention.",
loss:
"Per head L = Lcls (focal) + 5*L1 + 2*LGIoU; auxiliary head loss weighted by lambda_aux=0.5; AdamW (backbone lr 2e-5, rest 2e-4, wd 1e-4), 300 epochs x 60k pairs (100 for GOT-10k-only), batch 64 on 2x RTX 4090.",
},
equations: [
{
id: "tlctrack-mua",
label: "Masked Unidirectional Attention fusion",
formula: "E_z = self-attend(E_z); E_x = E_x + MHA(q=E_x, k=v=[E_t; E_z; E_x]); E_t = E_t (frozen)",
variables: [
{ symbol: "E_z", meaning: "template tokens" },
{ symbol: "E_x", meaning: "search-region tokens" },
{ symbol: "E_t", meaning: "temporal salient tokens (long plus short term)" },
{ symbol: "MHA", meaning: "multi-head attention operator" },
],
intuition: "Let the search area listen to memory and the template, but do not let memories chatter among themselves so each frame memory stays independent.",
why: "Injects spatio-temporal context into localization while protecting temporal feature integrity from cross-frame contamination.",
where: "Sec. II-B, Eqs. 1-5; applied inside each backbone block with unidirectional mask.",
},
{
id: "tlctrack-sstc",
label: "Spatial salient token collection",
formula: "a = sum_i (Q X^T)_{i,:}; I_s = arg-TopK(a, N_s); E_s = X[I_s]; rho = N_s / N",
variables: [
{ symbol: "Q", meaning: "learnable query matrix" },
{ symbol: "X", meaning: "candidate search tokens" },
{ symbol: "a", meaning: "per-token importance score vector" },
{ symbol: "rho", meaning: "token keeping ratio, set to 0.8" },
],
intuition: "Score every search patch for target relevance and keep only the best 80 percent, throwing away background patches.",
why: "Prevents background noise from polluting target spatial representation, unlike soft masked relation modeling.",
where: "Sec. II-C, Eqs. 6-9; SSTC at blocks 4/7/10 with adapter plus auxiliary head and stop-gradient decoupling.",
simulator: "siamese",
},
{
id: "tlctrack-tstb",
label: "Temporal salient token bank update",
formula: "m_t = sum_i Softmax(Q_s K_t^T / sqrt(D)) x S; E_lt = E_t[arg-TopK(m_t)]; E_st from F = Embedding(S_bin)+F; E_t = [E_lt; E_st]",
variables: [
{ symbol: "m_t", meaning: "temporal token importance weighted by prediction score map S" },
{ symbol: "E_lt", meaning: "long-term salient tokens preserving history" },
{ symbol: "E_st", meaning: "short-term salient tokens from latest feature map" },
{ symbol: "S", meaning: "head classification score map (S_bin its binarized form)" },
],
intuition: "Keep the historically most trustworthy memory tokens and append fresh tokens from the current confident prediction.",
why: "Maintains continuous high-quality spatio-temporal context that guides the next frame while filtering invalid temporal features.",
where: "Sec. II-D, Eqs. 10-16; long-term from current Et, short-term from score-aggregated feature map.",
simulator: "track-mgmt",
},
{
id: "tlctrack-loss",
label: "Head loss",
formula: "L = L_cls + 5 * L_1 + 2 * L_GIoU; total += 0.5 * L_aux",
variables: [
{ symbol: "L_cls", meaning: "focal classification loss on score map" },
{ symbol: "L_1", meaning: "L1 box regression loss" },
{ symbol: "L_GIoU", meaning: "generalized IoU regression loss" },
{ symbol: "L_aux", meaning: "same-form loss from SSTC auxiliary head" },
],
intuition: "Reward confident correct centers plus tight boxes; the auxiliary copy teaches the token selector with half weight.",
why: "Jointly optimizes localization accuracy and discriminative token collection.",
where: "Sec. II-E, Eq. 17; three-branch conv heads for score, offset, and size.",
simulator: "metrics",
},
],
datasets: ["got10k", "trackingnet", "lasot", "otb", "others"],
metrics: ["success-auc", "precision", "norm-precision", "fps"],
baselines: ["OSTrack384", "SeqTrack256/384", "ARTrack256/384", "MixFormer384", "MixFormer-L384", "DropTrack384", "HIPTrack384", "AQATrack256/384", "ADAT384", "AiATrack320", "SuperSBT256", "Stark-ST50"],
results: [
"GOT-10k test (GOT-10k-only protocol): TLCTrack384 AO 77.5, SR0.5 88.4, SR0.75 76.1 vs HIPTrack384 77.4/88.0/74.5 and ARTrack384 75.5/84.3/74.3; TLCTrack256 75.4/86.3/74.3 (Tab. I).",
"TrackingNet: TLCTrack384 AUC 84.8, PNorm 89.6, P 84.3 (Tab. I); TLCTrack256 83.7/88.6/82.7.",
"TNL2K: TLCTrack384 AUC 60.1, P 62.7, new state of the art over ARTrack384 59.8 (Tab. I).",
"NFS: TLCTrack384 AUC 70.7 vs HIPTrack384 68.1; TLCTrack256 67.3 (Tab. I).",
"OTB100: TLCTrack256 success 72.4 and TLCTrack384 precision 94.9 top the success/precision plots over HIPTrack384 71.0/93.0 (Fig. 4).",
"Efficiency on RTX 3090: TLCTrack256 92M params, 28G FLOPs, 97 fps, 951M memory; TLCTrack384 92M, 58G, 67 fps, 1122M vs ARTrack384 173M/83G/43 fps and SeqTrack384 89M/148G/22 fps (Tab. II).",
],
ablations: [
"Components on GOT-10k (256): baseline 72.0 AO; +TSTB 73.3 (+1.3); +MUA 74.6; +SSTC 75.4, with SR0.5 86.3 and SR0.75 74.3 (Tab. IIIa).",
"Keeping ratio rho: 0.7 gives 74.0, 0.8 peaks at 75.4, 0.9 gives 74.9, 1.0 (no SSTC) 74.6 (Tab. IIIb).",
"Auxiliary weight lambda_aux: 0.1 gives 74.3, 0.3 gives 74.9, 0.5 peaks at 75.4, 0.7 drops to 74.6 (Tab. IIIc).",
],
limitations: {
authorStated: [
"No explicit limitations section; design trade-offs stated: lower keeping ratio may drop important target features while higher ratio fails to suppress background; improper auxiliary weight degrades SSTC generalization.",
],
evident: [
"Training simulates temporal context with 3 templates plus 4 search frames per batch, so very long-term drift beyond the training window is untested.",
"Temporal tokens are initialized from the first-frame template, inheriting sensitivity to poor initialization despite continuous updating.",
"Evaluation emphasizes short-term benchmarks (GOT-10k, TrackingNet, OTB100, NFS); long-term disappearance and re-detection are not isolated.",
],
},
assumptions: [
"First-frame target state given; template plus search sizes fixed per variant.",
"GOT-10k test evaluation trains only on its train split (1000 videos removed per protocol) to satisfy one-shot rules.",
],
computation:
"Training on 2x RTX 4090, batch 64, 300 epochs (100 for GOT-10k protocol); inference 97 fps (256) / 67 fps (384) on RTX 3090 with 28G/58G FLOPs and 92M params.",
relations: [
{ to: "T009", type: "conceptual-successor", note: "Extends the SiamFC fully-convolutional template-search lineage into one-stream ViT tracking." },
{ to: "T071", type: "improves", note: "Adds continuous salient-token temporal context to the OSTrack one-stream baseline; attention maps show tighter target focus." },
{ to: "T070", type: "uses-as-baseline", note: "Compares against MixFormer iterative mixed attention on GOT-10k, TrackingNet, and OTB100." },
],
concepts: ["sot", "siamese", "attention", "anchor-free-head", "success-plot", "iou", "bounding-box"],
impact:
"Pushes SOT from discrete template refresh toward every-frame token-level memory refresh, showing continuous salient tokens beat sampling-based updates on accuracy and speed simultaneously.",
},
{
id: "T152",
arxiv: "2609.07547",
title: "Re-engineering SORT-based algorithms for low-cost small object tracking from omnidirectional footage",
shortTitle: "OmniSORT",
year: 2026,
authors: ["Xin Shu", "Meegan Gower", "Yvonne Buckley", "Anil Kokaram"],
fileName: "2609.07547v1.pdf",
task: "multi-object",
tags: ["tracking-by-detection", "sort", "oc-sort", "small-object", "omnidirectional", "seam-aware", "kalman-filter", "cpu-only"],
difficulty: "intermediate",
summary:
"OmniSORT re-engineers SORT and OC-SORT for battery-powered omnidirectional wildlife cameras where equirectangular seams break linearity and sub-32px flying targets make IoU and ReID embeddings useless. It adds a Seam-Aware Motion Model (SAMM) that wraps Kalman horizontal velocity across the left-right seam, an Omni-Euclidean wrapped center distance (OmniEuc), and a fused cost E_fuse = lambda*OmniEuc + (1-lambda)*GIoU-cost, plus the OmniSmall benchmark (2551 frames, 199 tracks, 87% small boxes). With ground-truth detections OmniOCSORT gains +8.51 HOTA, +9.41 MOTA, +10.17 IDF1 over OC-SORT; with YOLOX detections the gain narrows to +1.95 HOTA, staying CPU-only.",
problem:
"Standard-FoV MOT assumes locally linear image motion, overlapping boxes for IoU matching, and discriminative appearance; in 360-degree equirectangular wildlife footage targets wrap discontinuously across the seam, move fast relative to their few pixels, and their boxes are mostly background, while remote power budgets forbid ReID/transformer encoders.",
background: ["mot", "tracking-by-detection", "kalman", "state-space", "motion-model", "iou", "data-association", "cost-matrix", "hungarian", "mota", "hota", "idf1"],
previousWork: [
{
name: "SORT with Kalman prediction plus Hungarian IoU matching",
limitation:
"Constant-velocity Kalman filter plus IoU cost assumes smooth linear motion and overlapping predictions; seam jumps corrupt velocity and small fast targets leave zero overlap.",
whyThisPaper:
"Keeps SORT lightness but adds SAMM seam wrapping and a non-overlap association cost.",
},
{
name: "OC-SORT observation-centric recovery",
limitation:
"Recovers tracks disrupted by missed detections but still uses standard geometry that fails at the equirectangular seam for tiny targets.",
whyThisPaper:
"Ports the same seam-aware motion plus fused OmniEuc/GIoU cost into OC-SORT as OmniOCSORT.",
},
{
name: "Appearance-heavy MOT (DeepSORT, StrongSORT, BoT-SORT, HybridSORT, ByteTrack score-aware association, OmniTrack panorama features)",
limitation:
"ReID/transformer embeddings exceed remote CPU/GPU budgets and are undiscriminative when the box is dominated by background; two-stage score handling is absent from single-stage SORT variants.",
whyThisPaper:
"Stays appearance-free and CPU-only, showing motion-geometry fixes suffice on OmniSmall while documenting the remaining gap to score-aware trackers under noisy detections.",
},
],
researchGap:
"No lightweight SORT-style tracker jointly handled equirectangular seam discontinuity and small-object non-overlap association under a strict on-device compute envelope, with a dedicated wildlife benchmark.",
contribution: [
"Seam-Aware Motion Model (SAMM): wraps predicted x by mod 1 and corrects horizontal velocity onto signed principal interval [-0.5,0.5) when relative velocity jump exceeds tau=10.0.",
"OmniEuc wrapped center distance with E in [0,1] plus composite cost E_fuse combining OmniEuc and GIoU-derived cost with per-setting grid-searched lambda.",
"OmniSmall benchmark: 9 sequences, 2551 frames, 199 tracks, 20771 boxes, 87.12% small, with seam crossings and dense-flock stress clip; plus YOLOX tiled-crop detection protocol.",
"Evaluation isolating tracking (GT detections) vs full pipeline (YOLOX) on OmniSmall and JRDB panorama, with CPU tracking-stage FPS and per-sequence sign-flip significance.",
],
method: {
pipeline: ["tile-detect", "soft-nms", "seam-aware-predict", "fused-cost-associate", "velocity-correct", "track-update"],
architecture:
"SORT/OC-SORT skeleton unchanged (detect, Kalman predict, Hungarian associate, manage); only motion prediction and association cost replaced; no appearance encoder; tiled YOLOX single-class detector with linear Soft-NMS (IoU 0.45, conf 0.10) mapped to full frame.",
motionModel:
"SORT constant-velocity Kalman state [u,v,s,r,u_dot,v_dot,s_dot,r_dot] on normalized [0,1] centers; SAMM wraps predicted u by mod 1 and wraps u_dot across seam when |u_dot_new - u_dot_old|/max(|u_dot_old|,1e-6) > 10.",
appearanceModel: "None by design — appearance deemed unreliable for sub-32px boxes dominated by background and too costly for field deployment.",
association:
"Hungarian matching on E_fuse = lambda*E_OmniEuc + (1-lambda)*E_GIoU where E_GIoU=(1-GIoU)/2; lambda grid-searched per dataset/tracker/detection source (0.7 OmniSmall, 0.2-0.3 JRDB).",
detectionDependency:
"Single-class YOLOX trained per dataset on tiled crops (1024px/200 overlap OmniSmall, 480px/100 JRDB) without identity labels; chronological 2:8 frame split on OmniSmall to simulate field deployment.",
trackManagement:
"Inherits SORT/OC-SORT birth/death; OC-SORT observation-centric re-update retained, making OmniOCSORT the robust variant under missed detections.",
},
equations: [
{
id: "omnisort-giou",
label: "GIoU association basis",
formula: "IoU = |bm cap bn| / |bm cup bn|; GIoU = IoU - (|Cmn| - |bm cup bn|) / |Cmn|",
variables: [
{ symbol: "bm, bn", meaning: "predicted and detected boxes" },
{ symbol: "Cmn", meaning: "smallest enclosing region of both boxes" },
{ symbol: "GIoU", meaning: "overlap score in (-1,1] that orders non-overlapping boxes by separation" },
],
intuition: "When boxes do not touch, penalize by how much empty room their joint wrapper contains.",
why: "Gives IoU-family ordering power for near-miss small targets where plain IoU is uniformly zero.",
where: "Sec. II, Eqs. 1-2; adopted as the overlap half of the fused cost.",
simulator: "iou-track",
},
{
id: "omnisort-omnieuc",
label: "Omni-Euclidean wrapped distance",
formula: "d = sqrt(du^2+dv^2); d_tilde = sqrt(min(du,1-du)^2+dv^2); E = sqrt(2) * min(d, d_tilde), in [0,1]",
variables: [
{ symbol: "du, dv", meaning: "normalized horizontal and vertical center gaps" },
{ symbol: "d", meaning: "standard center Euclidean distance" },
{ symbol: "d_tilde", meaning: "seam-wrapped distance shifting by one image width" },
{ symbol: "E", meaning: "normalized seam-aware association cost" },
],
intuition: "Measure the shorter of the direct path and the path that exits one edge and re-enters the other, like a wrapped-around world map.",
why: "Keeps seam-crossing trajectories cheap while unrelated detections stay expensive, even with zero box overlap.",
where: "Sec. III-B, Algorithm 2; centers normalized to [0,1] before min-direct-vs-wrapped.",
simulator: "assoc-cost",
},
{
id: "omnisort-efuse",
label: "Composite association cost",
formula: "E_GIoU = (1 - GIoU)/2; E_fuse = lambda * E_OmniEuc + (1 - lambda) * E_GIoU",
variables: [
{ symbol: "E_GIoU", meaning: "GIoU-derived cost in [0,1]" },
{ symbol: "E_OmniEuc", meaning: "wrapped Euclidean cost in [0,1]" },
{ symbol: "lambda", meaning: "fusion weight in [0,1], grid-searched per setting" },
{ symbol: "E_fuse", meaning: "final Hungarian cost, lower means better match" },
],
intuition: "Blend a distance sense (works when boxes miss) with an overlap sense (precise when boxes align).",
why: "Exploits complementary signals: Euclidean dominates for small non-overlapping fast targets, GIoU refines well-aligned matches.",
where: "Sec. III-C, Eq. 3; lambda 0.7 best on OmniSmall, 0.2-0.3 on JRDB.",
params: "Endpoints are complementary: OmniEuc-only favors SORT (73.75 vs 60.53 HOTA), GIoU-only favors OCSORT (65.71 vs 52.23), both below fused 94.1 (Tab. IV).",
simulator: "assoc-cost",
},
{
id: "omnisort-samm",
label: "SAMM velocity wrap",
formula: "u_hat = u_hat mod 1; if |u_dot_new - u_dot_old|/max(|u_dot_old|, eps) > tau: u_dot_new = (u_dot_new + 0.5) mod 1 - 0.5",
variables: [
{ symbol: "u_hat", meaning: "predicted normalized horizontal center" },
{ symbol: "u_dot", meaning: "horizontal velocity component" },
{ symbol: "tau", meaning: "velocity-change ratio threshold, set to 10.0" },
{ symbol: "eps", meaning: "floor 1e-6 preventing division by zero" },
],
intuition: "If horizontal speed suddenly explodes, assume the target crossed the panorama seam and fold the speed back into range.",
why: "Undoes seam-induced velocity jumps that would otherwise fling Kalman predictions across the image.",
where: "Sec. III-A, Algorithm 1; SAMM alone without fused cost hurts (needs overlap-free association to exploit it).",
simulator: "kalman",
},
],
datasets: ["others"],
metrics: ["hota", "mota", "idf1", "idsw", "fps"],
baselines: ["SORT", "OCSORT", "ByteTrack", "HybridSORT"],
results: [
"OmniSmall GT detections: OmniOCSORT 94.12 HOTA, 99.16 MOTA, 94.16 IDF1, IDSw 62 vs OCSORT 85.61/89.75/83.99/382 (gain +8.51/+9.41/+10.17, Tab. II).",
"OmniSmall GT: OmniSORT 94.10/99.24/94.00/42 vs SORT 66.98/86.02/80.31/559 (gain +27.12 HOTA, +13.22 MOTA, Tab. II).",
"OmniSmall YOLOX: OmniOCSORT 41.76 HOTA, 26.97 MOTA, 50.72 IDF1, IDSw 108 (best) vs OCSORT 39.81/22.14/46.76/127 (gain +1.95 HOTA, Tab. III).",
"JRDB panorama GT: OmniSORT HOTA 67.69 vs SORT 62.43; OmniOCSORT 67.01 vs OCSORT 66.24 (Tab. II).",
"JRDB YOLOX: OmniOCSORT 27.70 vs SORT/OCSORT 26.45; OmniSORT 25.35 slightly below SORT 26.45, while ByteTrack leads 31.92 (Tab. III).",
"Tracking-stage CPU FPS (single core AMD Ryzen 9 3900): OmniSORT 320/1122 and OmniOCSORT 260/862 on JRDB/OmniSmall with YOLOX (Tab. III).",
"Significance at lambda 0.7: mean p-value 0.0039 across HOTA/MOTA/IDF1 paired sign-flip tests for both trackers (Tab. IV).",
],
ablations: [
"SAMM alone on OmniSmall GT collapses: SORT 66.98 to 20.05 HOTA, OCSORT 85.61 to 23.67, because corrected motion still needs overlap-free association (Tab. IV).",
"Single-term costs recover partially: SORT+OmniEuc 73.75 vs +GIoU 60.53; OCSORT+GIoU 65.71 vs +OmniEuc 52.23 (Tab. IV).",
"Lambda sweep peaks at 0.7 for both trackers (94.10/94.12 HOTA); endpoints and 0.9 fall well below fused (Tab. IV).",
],
limitations: {
authorStated: [
"Small OmniSmall scale (9 sequences) limits power of per-sequence significance tests.",
"Fusion weight lambda is a per-setting grid search, not adaptive; future work targets adaptive lambda.",
"Detector is the bottleneck: missed detections fragment trajectories; remaining gap to score-aware trackers (ByteTrack/HybridSORT) under noisy pedestrian-scale detections.",
"Planned extensions: stronger detector, detection-confidence integration into seam-aware association, while preserving appearance-free CPU-only design.",
],
evident: [
"SAMM relative-velocity test saturates at eps for slow/near-stationary tracks, applying spurious wraps beyond genuine seam events when used without the fused cost.",
"Center-distance cost can accept false positives that non-overlap would reject on noisy pedestrian-scale JRDB boxes, so E_fuse suits small objects better than dense scenes.",
"Tiled detection plus Soft-NMS cost dominates the full pipeline; reported CPU FPS covers tracking stage only.",
],
},
assumptions: [
"Boxes normalized to [0,1]; horizontal wrap only (vertical rotation applied once for bee sequence); small defined as area below 32x32 px.",
"JRDB invisible/fully-occluded objects ignored; only fully/partially visible treated as ground truth.",
"All baselines run at published defaults on identical detection sets to isolate tracking-stage contribution.",
],
computation:
"Tracking stage CPU-only on single core AMD Ryzen 9 3900; detector is QooCam 8K/Ricoh THETA tiled YOLOX (dominant cost); no appearance encoder and no GPU needed at tracking stage.",
relations: [
{ to: "T005", type: "extends", note: "Adds SAMM seam wrapping plus fused OmniEuc/GIoU cost to the SORT Kalman-Hungarian pipeline." },
{ to: "T073", type: "extends", note: "Ports the same seam-aware motion and cost into OC-SORT as OmniOCSORT." },
{ to: "T061", type: "uses-as-baseline", note: "Compares against ByteTrack single-stage vs two-stage score-aware association under GT and YOLOX detections." },
{ to: "T013", type: "conceptual-successor", note: "Explicitly rejects the DeepSORT ReID-embedding route as undiscriminative and over-budget for sub-32px wildlife targets." },
{ to: "T067", type: "uses-as-baseline", note: "Discusses StrongSORT-class appearance/motion extensions as exceeding remote power budgets." },
],
concepts: ["tracking-by-detection", "kalman", "motion-model", "state-space", "data-association", "cost-matrix", "hungarian", "track-management", "mota", "hota", "idf1", "benchmark-design"],
impact:
"Shows geometry-aware motion plus non-overlap association can beat appearance-heavy trackers for tiny seam-crossing targets at zero tracking-GPU cost, and releases OmniSmall as the first small-object omnidirectional wildlife MOT benchmark.",
},
{
id: "T153",
arxiv: "2609.07738",
title: "TFTrack: A Template-Free Framework for Efficient 3D Point Cloud Tracking",
shortTitle: "TFTrack",
year: 2026,
authors: ["Zhaofeng Hu", "Sifan Zhou", "Jiahao Nie", "Ziyu Zhao", "Weizi Li", "Ci-jyun Liang"],
fileName: "2609.07738v1.pdf",
task: "3d-tracking",
tags: ["3d-tracking", "point-cloud", "template-free", "siamese-free", "single-frame", "motion-prior", "lidar", "real-time"],
difficulty: "advanced",
summary:
"TFTrack argues the Siamese template branch is redundant for LiDAR 3D SOT: masking up to 80% of template points in M2Track costs under 1% success, since the previous box center already encodes motion context. It reformulates tracking as single-frame geometric transformation: recenter the current cloud by the prior center, extract features with a Siamese-free backbone, condition on box size (BCFE), and regress a 4-DoF offset trained with residual log-likelihood. Three variants (voxel/point/pillar) match template-based accuracy on KITTI and nuScenes while halving FLOPs (34.7G vs 67.1G) and running 92-121 FPS on RTX 3090.",
problem:
"Mainstream 3D SOT inherits dual-input Siamese appearance matching from 2D vision, doubling FLOPs and memory, then patches sparsity with complex multi-frame motion pipelines (segmentation, refinement) that mix foreground dynamics with static background — too heavy and too noisy for embedded robots, while the true marginal value of the full template cloud was never quantified.",
background: ["3d-tracking", "bounding-box", "motion-model", "state-space", "siamese", "attention", "benchmark-design"],
previousWork: [
{
name: "Siamese appearance-matching 3D trackers (SC3D, P2B, 3D-SiamRPN, BAT, PTT/PTTR, STNet, CXTrack, MBPTrack)",
limitation:
"Dual-branch template-search encoding plus correlation heads double compute and rely on appearance that collapses in sparse textureless occluded clouds.",
whyThisPaper:
"Deletes the template branch entirely, keeping one Siamese-free backbone on the recentered current frame.",
},
{
name: "Motion-centric dual-frame trackers (M2Track, M2Track++, FlowTrack)",
limitation:
"Reduce appearance reliance but keep two-frame inputs, foreground segmentation, and refinement stages with uniform-motion assumptions, leaving speed and accuracy suboptimal.",
whyThisPaper:
"Replaces explicit inter-frame matching with target-centric recentering plus box-size conditioning and a two-layer MLP offset head.",
},
{
name: "BEV/pillar unified trackers (PillarTrack, BEVTrack, STTracker, TrackAny3D)",
limitation:
"Strong accuracy (BEVTrack 71.8/89.2 KITTI mean) but retain dual-frame or heavy matching cost, e.g. BEVTrack needs full template-search machinery.",
whyThisPaper:
"Matches BEVTrack on nuScenes (59.01/71.41 vs 59.71/71.19) at a fraction of the FLOPs with three interchangeable representations.",
},
],
researchGap:
"No 3D SOT framework had removed both the template input and the Siamese/cross-frame matching head, reformulating tracking as single-frame transformation around the prior box prior.",
contribution: [
"Mask-out ablation proving Siamese redundancy: M2Track success drops under 1% until 80% template masking on nuScenes (57.21 to 56.74).",
"TFTrack template-free paradigm: P_hat = P - c_{t-1} recentering, Siamese-free backbone, Box-Conditioned Feature Encoding with box-size MLP, two-layer MLP 4-DoF head with rigid transform.",
"Three unified variants (TFTrack-Voxel/Pillar/Point) covering sparse and dense scenes with identical tiny head capacity.",
"Competitive KITTI/nuScenes accuracy at about 50% FLOPs and about 92-121 FPS with latency/memory analysis and robustness diagnostics.",
],
method: {
pipeline: ["target-centric-recenter", "single-frame-encode", "box-conditioned-encoding", "global-readout", "offset-regress", "rigid-transform"],
architecture:
"Recentering by prior center c_{t-1}; backbone Phi on current frame only (VoxelNet sparse, PointTransformerV3, or PillarNet); BCFE concatenates MLP(w,h,l) size embedding to each token then domain encoder Psi (pointwise mixing or BEV convs) plus readout R to global descriptor g; two-layer MLP head to (dx,dy,dz,dtheta).",
motionModel:
"Implicit geometric prior from recentering; explicit prediction is a 4-DoF rigid-body delta applied to previous box Bt-1; constant size and upright ground-supported assumption.",
appearanceModel:
"No appearance matching module; geometric features in the aligned local frame conditioned on expected target scale/shape.",
association: "Not applicable (single-object 3D; no detection-to-track assignment).",
detectionDependency: "None — initial 7-DoF box B1=(x,y,z,w,h,l,theta) given; size (w,h,l) held fixed thereafter.",
trackManagement: "No re-detection or multi-hypothesis logic; per-frame delta chained forward.",
loss:
"Residual log-likelihood (RLE) loss L=-log p_phi(m_gt - m_pred) with conditional normalizing flow on motion residuals; head capacity identical across variants (<0.1% of total).",
optimization:
"LiDAR 3D SOT protocol with class-specific search ranges (cars +-4.8m, pedestrians +-1.92m, z +-1.5m); backbone output downsampled to 16x16x128; OPE Success/Precision reporting.",
},
equations: [
{
id: "tftrack-appearance",
label: "Template-based appearance matching (prior art)",
formula: "(x_t,y_t,z_t,theta_t) = F_app(P_tmp, P_search)",
variables: [
{ symbol: "P_tmp", meaning: "template cloud from first/previous frame" },
{ symbol: "P_search", meaning: "search cloud cropped around previous prediction" },
{ symbol: "F_app", meaning: "Siamese backbone plus similarity module" },
],
intuition: "Crop both clouds, encode both, correlate features, regress the box — paying twice for feature extraction.",
why: "Baseline the paper attacks: dual encoding doubles FLOPs and memory for marginal information gain.",
where: "Sec. III-B, Eq. 1; covers P2B, BAT, PTT-style appearance trackers.",
},
{
id: "tftrack-motioncentric",
label: "Template-based motion-centric (prior art)",
formula: "(x_t,y_t,z_t,theta_t) = F_r(F_m(B_{t-1}, F_s(P_{t-1}, P_t)))",
variables: [
{ symbol: "F_s", meaning: "foreground segmentation over dual frames" },
{ symbol: "F_m", meaning: "motion inference module" },
{ symbol: "F_r", meaning: "motion refinement module" },
{ symbol: "B_{t-1}", meaning: "previous predicted box" },
],
intuition: "Segment the moving points across two frames, estimate their shift, polish it, then push the old box forward.",
why: "Stronger in sparse scenes than appearance matching but keeps multi-stage dual-frame complexity the paper removes.",
where: "Sec. III-B, Eq. 2; covers M2Track/M2Track++ pipelines.",
},
{
id: "tftrack-templatefree",
label: "Template-free motion modeling (proposed)",
formula: "P_hat_t = {p_i - c_{t-1}}; (dx,dy,dz,dtheta) = F_sf(P_hat_t cropped); B_t = Transform(B_{t-1}, delta)",
variables: [
{ symbol: "c_{t-1}", meaning: "previous bounding-box center acting as location prior" },
{ symbol: "P_hat_t", meaning: "current cloud recentered into target-centric frame" },
{ symbol: "F_sf", meaning: "Siamese-free backbone plus BCFE plus MLP head" },
{ symbol: "delta", meaning: "predicted 4-DoF relative offset" },
],
intuition: "Slide the whole current cloud so the old center sits at the origin; the network only has to predict a small nudge from there.",
why: "Embeds history into coordinates, eliminating template encoding and cross-frame matching while preserving geometric continuity.",
where: "Sec. III-C, Eqs. 3-4; centering plus rigid transform at test time.",
simulator: "motion",
},
{
id: "tftrack-rle",
label: "Residual log-likelihood training loss",
formula: "L_RLE = -log p_phi(m_gt - m_pred)",
variables: [
{ symbol: "m_gt", meaning: "ground-truth 4-DoF motion vector" },
{ symbol: "m_pred", meaning: "predicted motion vector from global descriptor" },
{ symbol: "p_phi", meaning: "conditional normalizing-flow density over residuals" },
],
intuition: "Instead of punishing raw distance, learn how wrong predictions usually distribute and maximize the likelihood of the actual error.",
why: "Models heteroscedastic motion uncertainty better than L1/L2 for sparse noisy clouds.",
where: "Sec. III-C, Eq. 5; trains the shared two-layer MLP head.",
},
],
datasets: ["kitti-tracking", "others"],
metrics: ["success-auc", "precision", "fps"],
baselines: ["SC3D", "P2B", "PTT", "LTTR", "BAT", "PTTR", "STNet", "GLT-T", "OSP2B", "CXTrack", "MBPTrack", "M2Track", "M2Track++", "MoCUT", "PillarTrack", "BEVTrack", "TrackAny3D"],
results: [
"KITTI mean Success/Precision: TFTrack-Voxel 64.0/81.6, TFTrack-Point 64.4/83.7, TFTrack-Pillar 60.8/79.0 at 92/121/95 FPS on RTX 3090 (Tab. II).",
"nuScenes mean: TFTrack-Voxel 59.01/71.41 (Car 64.59/72.41, Truck 66.53/68.48, Trailer 72.48/69.00) vs BEVTrack 59.71/71.19; Point 57.61/69.12; Pillar 58.01/71.10 (Tab. IV).",
"Template-free vs template-based voxel: 59.0/71.4 at 34.7G FLOPs vs 59.2/71.2 at 67.1G — halved compute for tied accuracy (Tab. V).",
"Template masking on M2Track (nuScenes): success 57.21 at 0% to 56.74 at 80% masking, collapsing only at 100% (47.86); precision stays 65.73 to 66.00 until 80% (Tab. I, Fig. 2).",
"Robustness on nuScenes Car: TFTrack best precision in all splits and best success on overall (67.79), late-25% (54.27), fast-start (15.19); sparse-init 61.76 vs BEVTrack 62.37 (Tab. IX).",
"Efficiency: voxel 10.87 ms end-to-end latency, 185/226 MB allocated/reserved GPU memory on RTX 3090.",
],
ablations: [
"Re-centering plus BCFE (Tab. VII): neither 50.2/62.5; BCFE only 50.7/62.5; re-center only 55.1/67.6; both 59.0/71.4 — recentering carries motion normalization, BCFE adds scale conditioning.",
"Backbones (Tab. VI): VoxelNet beats FocalsConv +3.7/+3.2; PTv3 beats Point-MAE slightly; PillarNet beats PointPillars +3.3/+3.2.",
"Template masking sweep (Tab. VIII): dual-frame voxel degrades gracefully 64.6/72.0 to 62.7/70.3 at 80% masking for Car.",
],
limitations: {
authorStated: [
"May degrade under extremely sparse LiDAR observations, severe occlusion, or adverse weather.",
"Predicts 4-DoF rigid motion with constant object size, limiting handling of strong deformation.",
"Future work: multimodal sensing, lightweight recovery mechanisms, richer motion representations.",
],
evident: [
"Single-frame chaining has no re-detection: a lost track has no recovery path beyond the next delta prediction.",
"KITTI numbers trail the best dual-frame trackers (BEVTrack 71.8/89.2), so minimalism trades peak small-benchmark accuracy for speed and generalization; authors attribute KITTI overfitting risk to its small size (Tab. III).",
"Robustness diagnostics are Car-only on nuScenes, leaving pedestrian/small-object sparsity claims narrower.",
],
},
assumptions: [
"Objects upright, ground-supported, rotating only about vertical axis; size fixed across frames so tracking reduces to 4-DoF center plus yaw.",
"Prior box center sufficiently accurate to keep the target inside the cropped current frame.",
],
computation:
"Single NVIDIA RTX 3090; TFTrack-Voxel/Pillar/Point at 92/95/121 FPS vs M2Track 57 and BEVTrack 108 FPS; FLOPs 34.7G template-free vs 67.1G template-based voxel; voxel latency 10.87 ms.",
relations: [
{ to: "T089", type: "improves", note: "Single-frame alternative to the STTracker spatio-temporal 3D tracking design at far lower cost." },
{ to: "T091", type: "improves", note: "Removes the BOTT box-only transformer dual-input matching while matching its accuracy regime." },
{ to: "T093", type: "conceptual-successor", note: "Continues the LEGO graph-optimized online 3D MOT thread toward minimalist single-frame estimation." },
{ to: "T009", type: "conceptual-successor", note: "Replaces the SiamFC-inherited Siamese template-search paradigm the paper traces to 2D vision." },
],
concepts: ["3d-tracking", "motion-model", "state-space", "bounding-box", "siamese", "attention", "benchmark-design"],
impact:
"Establishes template-free single-frame 3D SOT as a viable minimalist paradigm, redirecting efficiency efforts from lighter Siamese backbones to deleting the template branch and matching head altogether.",
},
{
id: "T154",
arxiv: "2609.08265",
title: "Tracking-by-detection in Multi-object Tracking: Survey and Experiments",
shortTitle: "TBD Survey and Experiments",
year: 2026,
authors: ["Yujin Yang", "Kyujin Shim", "Kangwook Ko", "Changick Kim"],
fileName: "2609.08265v1.pdf",
task: "survey",
tags: ["survey", "tracking-by-detection", "multi-object", "data-association", "kalman-filter", "camera-motion-compensation", "benchmark", "experimental-survey"],
difficulty: "intro",
summary:
"Experimental survey of tracking-by-detection MOT that fixes the field inconsistent-protocol problem: starting from a minimal SORT baseline, it incrementally integrates the best-per-category module only if it improves HOTA on all of MOT17, MOT20, and DanceTrack under sequence-level (not temporal-half) splits. It reviews Kalman state vectors, initialization, update strategies, CMC, spatial/appearance distances, summation, one/two/four-stage association, score fusion, and interpolation. The final generalized baseline (w,h state, OAI, CWKU, GMC, confidence boost, EMA appearance, weighted sum, two-stage association) lifts HOTA by +5.44/+2.18/+10.01 over SORT with post-processing reaching 69.82/69.62 on MOT17/MOT20.",
problem:
"TBD papers each add similarity, association, or motion modules but evaluate under inconsistent detectors, feature extractors, NMS thresholds, hyperparameters, and datasets (mostly linear-motion MOT17), with temporal-half splits leaking identities between train and validation — so no one knows which module genuinely contributes or generalizes to non-linear uniform-appearance regimes like DanceTrack.",
background: ["mot", "tracking-by-detection", "detection", "kalman", "state-space", "motion-model", "data-association", "cost-matrix", "hungarian", "reid", "mota", "idf1", "hota", "benchmark-design"],
previousWork: [
{
name: "SORT minimal TBD baseline (Kalman plus IoU plus Hungarian)",
limitation:
"Strong minimalist starting point but leaves initialization, update weighting, camera motion, and low-confidence detection handling unaddressed.",
whyThisPaper:
"Adopts SORT as the controlled starting baseline and measures every addition against it under one protocol.",
},
{
name: "Appearance-augmented TBD (DeepSORT feature bank, StrongSORT, BoT-SORT ReID, Deep OC-SORT dynamic appearance)",
limitation:
"Each evaluated with different detectors, NMS thresholds, CMC, and post-processing (Tab. 1), obscuring whether gains come from the claimed module.",
whyThisPaper:
"Re-evaluates appearance update (bank vs EMA vs dynamic), summation rules, and CMC under fixed YOLOX plus SBS-S50 with per-experiment tuning.",
},
{
name: "Association-centric TBD (ByteTrack two-stage, BoostTrack similarity/confidence boosts, HybridSORT weak cues, LGTrack four-stage, LGTrack-style localization/classification staging)",
limitation:
"Validated mainly on MOT17/MOT20 linear distinctive-appearance data with temporal-half splits overlapping scenes and identities.",
whyThisPaper:
"Adds DanceTrack non-linear similar-appearance evaluation with sequence-level splits (MOT17-02/10/11/13 train vs 04/05/09 val) for a generalization test.",
},
],
researchGap:
"No TBD survey combined a functional taxonomy (Kalman, init, CMC, spatial/additional distances, tricks, appearance update, summation, association, score fusion, post-processing) with controlled cumulative experiments building one generalized baseline across linear, crowded, and non-linear benchmarks.",
contribution: [
"Functional taxonomy of TBD pipeline stages with equations for IoU/GIoU/DIoU/CIoU/BIoU/HMIoU, EMA/dynamic appearance, weighted/geometric/adaptive/minimum summation, and one/two/four-stage association.",
"Standardized protocol: shared YOLOX detector (CrowdHuman plus WiderPerson training, NMS 0.8) and SBS-S50 ReID, sequence-level splits, HOTA-primary reporting with MOTA/IDF1/DetA/AssA.",
"Stepwise cumulative ablations (Tabs. 3-15) keeping only modules that improve all three datasets, yielding a final generalized baseline plus post-processing guidance.",
"Design guidelines: (w,h) state, OAI init, CWKU update, GMC, plain IoU, confidence similarity boost, EMA, weighted sum, two-stage association, score fusion, Gaussian interpolation/AFLink conditionally.",
],
method: {
pipeline: ["detect-yolox", "kalman-predict", "cmc-compensate", "distance-measure", "two-stage-associate", "ema-update", "track-manage", "interpolate-postprocess"],
architecture:
"Survey plus experimental harness, no new tracker: SORT-based baseline (Alg. 1) with confirmed/unconfirmed two-round matching, lost-track chasing for L_lost frames, and staged upgrades; YOLOX detector plus SBS-S50 FastReID (Market-1501 init, 60 epochs, CE plus triplet) feeding cosine appearance distances.",
motionModel:
"Kalman filter with constant-velocity prediction and compared state vectors (cx,cy,s,a) vs (cx,cy,a,h) vs (cx,cy,w,h); compared prediction/update strategies (fixed inactive height/width, CWKU thresholds 0.6/0.7, ORU virtual-trajectory re-update).",
appearanceModel:
"Compared Feature Bank (10/50/100) vs EMA (alpha 0.9) vs Dynamic Appearance confidence-weighted EMA (alpha_f 0.95); SBS-S50 embeddings L2-normalized on 128x348 crops.",
association:
"Compared one-stage (SORT, BoostTrack-filtered, combined matching) vs two-stage (ByteTrack/BoT-SORT high-then-low confidence) vs four-stage (LGTrack localization/classification quadrants); Hungarian bipartite matching throughout; score fusion C=1-IoU*s in stage two.",
detectionDependency:
"YOLOX-X per dataset trained with CrowdHuman plus WiderPerson, COCO init, mosaic/perspective/mixup, 80 epochs; NMS 0.8, confidence 0.1 at inference; OAI init with IoU gate plus activation 2 frames.",
trackManagement:
"Confirmed tracks matched first, then unconfirst; L_min promotion, L_lost chase window set to frame rate; OAI blocks init when max IoU with existing tracks exceeds threshold.",
loss: "Not applicable — survey harness; detector uses YOLOX defaults and ReID uses cross-entropy plus triplet (training details in Sec. 4.2).",
optimization:
"Per-experiment hyperparameter tuning (lambda 0.7 weighted sum via grid search; boost weights 0.50/0.25; BIoU scales 0.1-0.3); greedy cumulative rule: keep a module only if HOTA rises on all three datasets.",
},
equations: [
{
id: "tbdsurvey-iou",
label: "IoU spatial distance",
formula: "IoU = |A cap B| / |A cup B|",
variables: [
{ symbol: "A, B", meaning: "track-predicted and detected boxes" },
{ symbol: "|.|", meaning: "box or set area" },
],
intuition: "Overlap divided by total footprint: 1 means identical, 0 means disjoint.",
why: "Standard assignment cost; the survey finds it the most robust across all three datasets versus fancier variants.",
where: "Sec. 3.6, Eq. 1; baseline cost C1/C2 in Alg. 1 with Hungarian matching.",
simulator: "iou-track",
},
{
id: "tbdsurvey-giou",
label: "GIoU/DIoU/CIoU penalties",
formula: "GIoU = IoU - |C-(A cup B)|/|C|; DIoU = IoU - rho(a,b)^2/c^2; CIoU = DIoU - alpha*v",
variables: [
{ symbol: "C", meaning: "smallest enclosing rectangle of A and B" },
{ symbol: "a, b", meaning: "box centers" },
{ symbol: "rho", meaning: "Euclidean center distance" },
{ symbol: "c", meaning: "diagonal of enclosing box C" },
{ symbol: "v, alpha", meaning: "aspect-ratio consistency term and its IoU-dependent weight" },
],
intuition: "When boxes miss, penalize empty wrapper area, center gap, and shape mismatch so ranking still orders near-misses.",
why: "Tests whether non-overlap penalties beat plain IoU; finding: marginal on MOT17, worse on crowded/dynamic sets.",
where: "Sec. 3.6, Eqs. 2-6 with Fig. 2 geometry; BIoU/HMIoU variants Eqs. 7-10.",
simulator: "assoc-cost",
},
{
id: "tbdsurvey-ema",
label: "Appearance update (EMA and dynamic)",
formula: "e_k = alpha*e_{k-1} + (1-alpha)*f_k; alpha_t = alpha_f + (1-alpha_f)*(1-(s_det-sigma)/(1-sigma))",
variables: [
{ symbol: "e_k", meaning: "track appearance feature at frame k" },
{ symbol: "f_k", meaning: "current detection embedding" },
{ symbol: "alpha", meaning: "EMA weight, 0.9" },
{ symbol: "s_det", meaning: "detection confidence driving dynamic weight" },
],
intuition: "Blend a little of the new look into the running identity portrait each frame; dynamic version trusts confident sightings more.",
why: "Balances adaptation against drift; EMA wins as the most balanced online strategy versus static banks and noisy dynamic updates.",
where: "Sec. 3.9, Eqs. 11-12; alpha_f 0.95, sigma threshold.",
simulator: "reid",
},
{
id: "tbdsurvey-sum",
label: "Spatial-appearance summation",
formula: "D = lambda*D_a + (1-lambda)*D_s (kept); alternatives: sqrt(D_a*D_s), min(d_a,d_s), IoU-gated threshold form",
variables: [
{ symbol: "D_s", meaning: "spatial (IoU-based) distance matrix" },
{ symbol: "D_a", meaning: "appearance (cosine) distance matrix" },
{ symbol: "lambda", meaning: "fusion weight, 0.7 by grid search" },
{ symbol: "D", meaning: "final assignment cost matrix" },
],
intuition: "Weighted average of where-it-is versus what-it-looks-like; gating and min/max variants try to veto bad pairs.",
why: "Decides how much to trust geometry vs identity; plain weighted sum beats geometric-mean, adaptive, and minimum rules.",
where: "Sec. 3.10, Eqs. 13-20; threshold form Eq. 14 with IoU gate o_min.",
simulator: "assoc-cost",
},
{
id: "tbdsurvey-assoc",
label: "Two-stage association with score fusion",
formula: "Stage1: match high-conf detections; Stage2: match low-conf to unmatched; C = 1 - IoU(D,B)*s",
variables: [
{ symbol: "s", meaning: "detection confidence score fused into stage-two cost" },
{ symbol: "C", meaning: "score-fused cost matrix" },
{ symbol: "D,B", meaning: "detection and predicted-track box sets" },
],
intuition: "Seat the confident passengers first, then find leftover seats for uncertain ones while favoring trustworthy tickets.",
why: "Rescues occluded tracks from low-confidence boxes without letting them corrupt confident matches; best HOTA on all datasets.",
where: "Secs. 3.11-3.12, Fig. 4 and Eq. 21; four-stage LGTrack overcomplicates and underperforms.",
simulator: "bytetrack",
},
],
datasets: ["mot17", "mot20", "dancetrack"],
metrics: ["hota", "mota", "idf1", "deta", "assa"],
baselines: ["SORT", "DeepSORT", "ByteTrack", "BoT-SORT", "StrongSORT", "Deep OC-SORT", "HybridSORT", "BoostTrack", "MAATrack", "ConfTrack", "ImprAsso", "LGTrack"],
results: [
"Final baseline vs SORT (Tab. 16): MOT17 HOTA 63.91 to 69.35 (+5.44), MOTA 78.66 to 81.04, IDF1 77.85 to 87.16; MOT20 66.40 to 68.58 (+2.18); DanceTrack 53.26 to 63.27 (+10.01, largest gain).",
"With post-processing: MOT17 69.82 HOTA (GSI variant 70.34, GBI 70.79 in Tab. 15); MOT20 69.62 (GSI best 69.69); DanceTrack 63.68 with AFLink (no interpolation evaluated there).",
"Kalman state (Tab. 3): (cx,cy,w,h) best on all sets (65.07/67.33/55.71 HOTA) over (s,a) and (a,h) forms.",
"OAI init (Tab. 4): +1.93/+0.44/+1.85 HOTA (MOT17/MOT20/DanceTrack); CWKU (Tab. 5): 68.09/68.03/57.92; GMC CMC (Tab. 7): 68.61/68.26/58.23.",
"Distances (Tabs. 8-10): plain IoU most balanced (68.61/68.26/58.23); DIoU/CIoU tie on MOT17 only; confidence similarity boost lifts DanceTrack 58.23 to 60.17 and combined conf+shape to 60.37; DLO/DUO/CF/PIA tricks all degrade or hold flat.",
"Summation/association (Tabs. 12-13): weighted sum (lambda 0.7) best (68.94/68.42/62.85); two-stage best (69.35/68.58/63.27) over one-stage and four-stage (67.78/67.24/61.39).",
],
ablations: [
"Each table is an ablation keeping all else fixed: state vector, OAI on/off, Kalman strategies (fixed-h, fixed-wh, CWKU 0.6/0.7, plus ORU), NMS/activation/lost-max grid, CMC none/GMC/ECC, 8 spatial distances, OCM/ROCM/confidence/shape/Mahalanobis boosts, 4 distance tricks, 3 appearance updates, 6 summation rules, 6 association structures, score fusion on/off, AFLink x LI/GSI/GBI post-processing.",
"ORU plus CWKU is mixed: gains on MOT17/MOT20 but slight DanceTrack drop (Tab. 5); fixed inactive states show no consistent benefit.",
"Interpolation not evaluated on DanceTrack (no GT for fully occluded objects; bridging would create false positives); AFLink helps DanceTrack (+0.41 HOTA) but slightly hurts MOT17/MOT20 (Tab. 15).",
],
limitations: {
authorStated: [
"No specific funding; code released upon acceptance (not yet verifiable); no datasets generated — figures/tables derived from public literature.",
"Interpolation methods inapplicable to DanceTrack under its annotation policy; AFLink-plus-interpolation combos give no added gain over interpolation alone.",
"Evaluation is pedestrian/dance human tracking (MOT17/MOT20/DanceTrack); vehicle, animal, and 3D regimes are out of scope.",
],
evident: [
"Greedy keep-only-if-all-three-improve rule may discard modules that help one regime substantially while neutral elsewhere (e.g. HMIoU excels on DanceTrack 58.76 but is dropped).",
"Detector and ReID fixed to YOLOX plus SBS-S50, so conclusions are conditional on that front end; NMS 0.8 and activation 2 are baked into later comparisons.",
"Temporal-half-split criticism is addressed by sequence splits, but the chosen sequence partition itself becomes the new fixed evaluation quirk.",
],
},
assumptions: [
"HOTA primary (joint detection/association/localization) with MOTA/IDF1/DetA/AssA secondary; hyperparameters tuned per experiment for fairness.",
"Sequence-level splits: MOT17 train on 02/10/11/13, val on 04/05/09; MOT20 train on 02/05, val on 01/03 — no identity/scene overlap by construction.",
],
computation:
"YOLOX trained on 8x RTX 3090 (80 epochs, cosine schedule, mosaic/perspective/mixup, COCO init); ReID fine-tuned 60 epochs batch 64; Feature Bank with 50+ features noted as below frame-rate (unsuitable for online MOT).",
relations: [
{ to: "T005", type: "builds-on", note: "Uses SORT Kalman-IoU-Hungarian pipeline as the controlled starting baseline for all experiments." },
{ to: "T006", type: "uses-as-baseline", note: "Re-evaluates MOT16/MOT17-family sequences and metrics under new sequence-level splits." },
{ to: "T013", type: "uses-as-baseline", note: "Benchmarks DeepSORT detector/extractor settings and Feature Bank appearance update." },
{ to: "T061", type: "improves", note: "Confirms ByteTrack two-stage association as the best association strategy across all three datasets." },
{ to: "T067", type: "uses-as-baseline", note: "Compares StrongSORT GSI plus AFLink post-processing under the unified protocol." },
{ to: "T073", type: "uses-as-baseline", note: "Re-evaluates OC-SORT OCM/ORU update strategy with mixed DanceTrack findings." },
{ to: "T059", type: "conceptual-successor", note: "Positions TBD cumulative baseline against end-to-end transformer MOTR-style alternatives discussed in related work." },
],
concepts: ["tracking-by-detection", "kalman", "state-space", "motion-model", "data-association", "cost-matrix", "hungarian", "dual-threshold", "reid", "learned-association", "track-management", "mota", "idf1", "hota", "benchmark-design", "occlusion"],
impact:
"Gives the field a reusable generalized TBD baseline and a fair-evaluation recipe (sequence splits, fixed front end, keep-only-if-all-improve), shifting future claims from single-dataset wins to cross-regime robustness.",
},
{
id: "T155",
arxiv: "2609.12261",
title: "Revisiting Multi-Object Tracking Baselines: Hyperparameter Optimization with Multi-Fidelity Greedy Coordinate Search",
shortTitle: "MFGCS MOT tuning",
year: 2026,
authors: ["Momir Adzemovic"],
fileName: "2609.12261v1.pdf",
task: "multi-object",
tags: ["tracking-by-detection", "hyperparameter-optimization", "multi-fidelity", "coordinate-search", "baseline-tuning", "benchmark"],
difficulty: "advanced",
summary:
"Systematic hyperparameter optimization study of four tracking-by-detection MOT methods (SORT, ByteTrack, SparseTrack, MoveSORT-KF) on DanceTrack and SportsMOT, comparing eight optimizers from six families under shared search spaces. Proposes Multi-Fidelity Greedy Coordinate Search (MFGCS), which tunes one hyperparameter at a time on a shared 6-scene subset and accepts updates only on full-dataset HOTA gain. TPE and MFGCS beat hand-tuned and published configurations in all eight tracker-dataset combinations, and the tuned configs transfer to held-out test splits.",
problem:
"Tracking-by-detection MOT methods depend on hand-set thresholds and lifecycle parameters whose selection procedures are undocumented and costly, so reported comparisons may conflate architectural progress with unequal tuning effort, and configs tuned on one benchmark transfer poorly to motion-heavy benchmarks such as DanceTrack and SportsMOT.",
background: ["mot", "tracking-by-detection", "detection", "iou", "data-association", "cost-matrix", "hungarian", "track-management", "motion-model", "kalman", "hota", "mota"],
previousWork: [
{
name: "SORT Kalman plus IoU matching pipeline",
limitation: "Minimal association machinery with hand-set IoU threshold, lifecycle and noise scales that must be re-tuned per benchmark.",
whyThisPaper: "Treats all such settings as a black-box search space and optimizes HOTA automatically instead of by hand.",
},
{
name: "ByteTrack two-stage low-confidence association",
limitation: "Adds high-tier and low-tier match thresholds plus score-fusion switches, expanding the manual tuning burden.",
whyThisPaper: "Includes its 12-parameter space in the same controlled optimizer comparison with identical evaluation pipeline.",
},
{
name: "SparseTrack pseudo-depth stratified association and MoveSORT-KF motion-cost variant",
limitation: "Depth-level counts and motion weights add further interacting hyperparameters tuned only on their original benchmarks.",
whyThisPaper: "Retunes their 14- and 9-parameter spaces per dataset and releases the resulting configurations as stronger baselines.",
},
],
researchGap:
"No broad comparison of HPO strategies for modern deep-detector tracking-by-detection MOT using HOTA as objective, and no multi-fidelity optimizer matched to scene-subset evaluation cost.",
contribution: [
"Multi-Fidelity Greedy Coordinate Search: anchored coordinate-wise search with coarse-to-fine grid (g=5, r=3), shared 6-scene subset comparison, full-dataset acceptance gate, interval shrink and barren-sweep coordinate dropping.",
"Two-stage controlled study: eight optimizers from six families compared on SORT DanceTrack-val, then TPE and MFGCS evaluated across four trackers and two datasets with shared per-tracker search spaces and cumulative-scene budget.",
"Evidence that published and hand-tuned baselines are under-tuned: gains of up to 4.38 HOTA over manual configs and up to 16.05 over published scores, with test-split transfer verified on official servers.",
"Released code (Motrack library) and stored tuned configurations plus full search-space and optimizer-setting documentation for reproducible baselines.",
],
method: {
pipeline: ["cache-yolox-detections", "define-search-space", "subset-coordinate-search", "full-dataset-acceptance", "select-validation-best", "evaluate-on-test-server"],
architecture:
"HPO study only, no new tracker. Tunes SORT, ByteTrack, SparseTrack and MoveSORT-KF association, motion and lifecycle hyperparameters around frozen YOLOX-X detections with interpolation postprocessing; MFGCS sweeps coordinates with grid search on a shared scene subset and gates acceptance on full-split HOTA.",
motionModel: "Kalman filter with tunable position-noise scale; MoveSORT-KF adds tunable motion weight lambda and L1/L2 motion distance with BoT-SORT-style camera-motion compensation.",
appearanceModel: "No appearance tuning; study restricted to motion plus IoU association family, excluding DeepSORT and OC-SORT variants.",
association: "Hungarian matching on IoU or Move cost with tunable match thresholds, duplicate-IoU suppression, high-tier and low-tier two-stage matching and categorical score-fusion switches.",
detectionDependency: "Frozen public YOLOX-X checkpoints per dataset, cached once; search includes detection and new-track confidence thresholds so DetA is within reach.",
trackManagement: "Tunable initialization threshold, remember threshold (track retention), interpolation gap fill (10 DanceTrack, 20 SportsMOT frames) and minimum track length filtering.",
optimization: "MFGCS grid g=5 points, r=3 rounds, m=6 scenes, T=6 sweeps, epsilon 1e-4, rho 0.25 shrink, seed 42, cap 100 full-fidelity trials; baselines via Optuna TPE, GP-BO, random, Hyperband and BOHB.",
},
equations: [
{
id: "mfgcs-objective",
label: "Tracker tuning as black-box maximization",
formula: "theta* = argmax over theta in Theta of f(theta; D), where f is HOTA on scene split D",
variables: [
{ symbol: "theta", meaning: "vector of tracker hyperparameters from bounded search space Theta" },
{ symbol: "D", meaning: "dataset split consisting of |D| scenes" },
{ symbol: "f(theta; D)", meaning: "HOTA obtained by evaluating the tracker configured with theta on D" },
],
intuition: "Treat the whole discrete tracking plus evaluation pipeline as a black box that maps settings to one HOTA number, then search for the settings with the best number.",
why: "No gradient exists through discrete association and HOTA scoring, so tuning must be derivative-free search.",
where: "Section 3, Equation 1; HOTA is the optimization objective throughout.",
},
{
id: "mfgcs-budget",
label: "Cumulative scene budget as common cost",
formula: "B = sum over i=1..N of |Di|, where Di is the scene subset used in evaluation i",
variables: [
{ symbol: "B", meaning: "total number of scene evaluations accumulated over an optimization run" },
{ symbol: "Di", meaning: "set of scenes used in the i-th objective evaluation, including subset and full evaluations" },
{ symbol: "N", meaning: "total number of objective evaluations performed" },
],
intuition: "Count scenes actually scored rather than trials, because a subset trial is cheaper than a full-split trial.",
why: "Trial counts are incomparable across single-fidelity and multi-fidelity optimizers, so cost must be measured in scenes.",
where: "Section 3, Equation 2; cost axis of Figure 2 and Tables 2, 4 and 5.",
},
{
id: "mfgcs-hota",
label: "HOTA optimization objective with DetA and AssA",
formula: "HOTA = mean over alpha of sqrt(DetA_alpha * AssA_alpha); MOTA = 1 - sum(FN+FP+IDSW)/sum(GT)",
variables: [
{ symbol: "alpha", meaning: "IoU matching threshold; HOTA averages over the threshold set" },
{ symbol: "DetA", meaning: "detection accuracy component at threshold alpha" },
{ symbol: "AssA", meaning: "association accuracy component at threshold alpha" },
{ symbol: "FN, FP, IDSW, GT", meaning: "misses, false positives, identity switches and ground-truth counts in MOTA" },
],
intuition: "HOTA balances finding boxes against keeping identities, so tuning cannot win by detection thresholds alone.",
why: "Single scalar objective without hand weighting detection versus association; MOTA, IDF1 and IDSW reported alongside.",
where: "Section 5.1 evaluation metrics following TrackEval definitions; Tables 3 and B.8.",
simulator: "metrics",
},
],
datasets: ["dancetrack", "others"],
metrics: ["hota", "deta", "assa", "mota", "idf1", "idsw", "fps"],
baselines: ["SORT", "ByteTrack", "SparseTrack", "MoveSORT-KF", "Manual hand-tuned configs", "Random search", "GP-BO", "Hyperband", "BOHB", "GCS"],
results: [
"Optimizer selection on SORT DanceTrack-val (Tab. 2): MFGCS 53.69 HOTA at 1297 scenes and 17 trials; GCS 53.59 at 3950 scenes; TPE 53.27, TPE-plus-prior 53.35, GP-BO 52.50, random 52.28, BOHB 51.71, Random-plus-Hyperband 51.46, each at 2500 scenes except pruned runs near 1530.",
"DanceTrack test (Tab. 3, official servers): SORT-MFGCS 54.05 HOTA (DetA 79.41, AssA 36.89, MOTA 90.50, IDF1 55.14, IDSW 1728) vs published 47.90; ByteTrack-TPE 53.73 vs 47.70; SparseTrack-MFGCS 58.17 (DetA 81.94, AssA 41.44) vs 55.50; MoveSORT-KF-TPE 55.58 vs 53.30.",
"SportsMOT test (Tab. 3): SORT-TPE 72.38, ByteTrack-MFGCS 72.79 vs published 64.10, SparseTrack-TPE 69.52, MoveSORT-KF-MFGCS 72.89 (DetA 88.08, AssA 60.38, MOTA 96.54, IDF1 72.57) vs published 71.00.",
"Validation (Tab. 4): both optimizers beat paper and manual configs in all eight cells; MFGCS reaches the shared per-cell HOTA target faster than TPE in seven of eight (e.g. SORT DanceTrack 73 vs 125 min, SparseTrack 78 vs 207 min).",
"Association throughput with cached detections on i7-12700K plus RTX 3070 (Tab. B.8): SORT 930-1082, ByteTrack 1012-1121, SparseTrack 698-755, MoveSORT-KF 845-964 FPS.",
],
ablations: [
"Coordinate optimizer (Tab. 5): grid g5r3-n6-s6 53.69 beats ternary best 53.19 (tern5-n3-s6) and random best 52.90 (rand20-n6-s6); three contraction rounds beat two by 1.1 points (53.69 vs 52.55); coarsest grid g4r2-n3-s6 stalls at 52.18.",
"Subset size (Tab. 5): grid best at m=6 (53.69) vs m=3 52.95 (noisy discards) and m=9 53.33; ternary best at m=3 (53.19) vs m=6 52.40.",
"Multi-fidelity value: full-fidelity coordinate search (m=25) reaches 53.59 at 3950 scenes but only 53.08 at matched 1297-scene cost, so the subset stage is worth 0.61 HOTA at equal budget while cutting budget threefold.",
"TPE settings (Tab. C.11): gamma 0.30 loses 1.5 points (51.77); wider candidate pools and multivariate kernel land within noise of selected gamma 0.20 with 24 candidates (53.27).",
"Hyperband pruning (Tab. C.13): pruning cuts cost 2500 to about 1530 scenes but costs 0.82 HOTA (random) and 1.56 (TPE/BOHB) at matched budget, attributed to unpaired per-trial scene samples.",
],
limitations: {
authorStated: [
"Restricted to the tracking-by-detection family without appearance-based or observation-centric variants such as DeepSORT and OC-SORT.",
"Scene-count budget assumes scenes have approximately similar lengths; with strongly differing lengths, evaluated frames would be the more accurate cost measure.",
"Each cell is a single run at seed 42; no multi-seed variance is reported.",
],
evident: [
"SportsMOT test scores drop 6.06 to 8.83 HOTA below validation for all four trackers, largest on SparseTrack, so per-dataset tuning does not remove the validation-to-test gap.",
"Search spaces differ in size by a factor of two (7 SORT to 14 SparseTrack parameters), so optimizer comparisons across trackers are not size-controlled.",
],
},
assumptions: [
"Detector frozen and identical within each dataset so score differences originate in tracker configuration.",
"All reported scores computed on interpolation-postprocessed outputs with dataset-specific gap and length settings.",
],
computation:
"Single machine with 12th-gen Intel i7-12700K CPU and NVIDIA RTX 3070 GPU; per optimizer-times-tracker run about two to seven hours; hand-tuning took several hours to one day per tracker; MFGCS reached target in under two hours unattended wall-clock.",
relations: [
{ to: "T005", type: "uses-as-baseline", note: "Tunes SORT Kalman plus IoU pipeline and beats its published DanceTrack numbers." },
{ to: "T061", type: "uses-as-baseline", note: "Tunes ByteTrack two-stage association under the same shared search protocol." },
{ to: "T063", type: "uses-as-baseline", note: "Uses DanceTrack validation and test splits as one of two tuning benchmarks." },
{ to: "T075", type: "conceptual-successor", note: "MoveSORT-KF variant tuned here incorporates BoT-SORT camera-motion compensation." },
],
concepts: ["tracking-by-detection", "data-association", "cost-matrix", "hungarian", "track-management", "motion-model", "kalman", "hota", "mota", "benchmark-design"],
impact:
"Reframes MOT progress claims by showing reported baselines are under-tuned, and provides released tuned configurations as the comparison point for future tracking-by-detection work.",
},
{
id: "T156",
arxiv: "2609.15171",
title: "EECTracker: Swarm Motion Prior-Guided Feature Compensation for Airborne Optical UAV Swarm Tracking",
shortTitle: "EECTracker",
year: 2026,
authors: ["Zhaochen Chu", "Tao Song", "Ren Jin", "Mingdong Jia", "Defu Lin"],
fileName: "2609.15171v1.pdf",
task: "multi-object",
tags: ["uav-swarm-tracking", "joint-detection-and-tracking", "motion-prior", "feature-compensation", "aerial-tracking", "small-object"],
difficulty: "advanced",
summary:
"Joint detection-and-tracking framework for airborne optical UAV swarm tracking that builds a probabilistic swarm motion prior from reliable historical tracklets with an EqMotion predictor plus uncertainty head, then compensates weak current-frame features via Energy-Entropy Consistency Activation. EEC Activation projects historical features under the prior, scores motion-prior-conditioned consistency from residual energy plus local residual entropy, and fuses them through a channel-wise soft mask. Beats SCT-MOT and HOMATracker on AIRMOT and UAVSwarm while staying online.",
problem:
"Airborne UAV swarm targets are tiny, fast-moving against cluttered backgrounds, so detector responses weaken or drop for consecutive frames; existing TBD, JDT and transformer trackers depend on reliable per-target observations and history, and fragment trajectories with identity switches once those cues go missing.",
background: ["mot", "tracking-by-detection", "detection", "motion-model", "data-association", "reid", "appearance-features", "occlusion", "mota", "idf1", "hota"],
previousWork: [
{
name: "Detection-driven TBD and JDT trackers (ByteTrack, OC-SORT, FairMOT, DeepSORT, Hybrid-SORT)",
limitation: "Need reliable per-target detections and states to update tracks and associate; weak or missing responses break the update and association chain.",
whyThisPaper: "Adds swarm-level motion guidance plus selective historical-feature compensation before detection so weak responses are reinforced.",
},
{
name: "Transformer end-to-end trackers with persistent queries (MOTR, MOTRv3, TransTrack)",
limitation: "Track queries still propagate from target-specific history that degrades under intermittent observation.",
whyThisPaper: "Replaces per-target-only guidance with a shared swarm motion prior aggregated from whichever tracklets remain reliable.",
},
{
name: "Closed-loop and motion-guided fusion (SCT-MOT with SMTP plus TG-STFF, HOMATracker, FGFA-style propagation)",
limitation: "Spatial references come from detector locations or deterministic per-target predictions, which are unavailable or inaccurate exactly when needed.",
whyThisPaper: "Projects whole feature maps under a probabilistic swarm displacement distribution and gates fusion by learned feature consistency instead of detection locations.",
},
],
researchGap:
"No swarm-level probabilistic motion prior combined with a consistency-gated pixel-level compensation that works without requiring a reliable observation for every UAV.",
contribution: [
"Probabilistic swarm motion prior construction from reliable tracklets: swarm-velocity decomposition, EqMotion interaction-aware prediction, per-tracklet Gaussian offset heads, directional-consistency weighting and moment-matched compact Gaussian prior Mt.",
"Energy-Entropy Consistency Activation: prior-guided projection, channel-wise expected residual energy plus local residual-entropy, Local EEC score and learnable channel-wise soft consistency mask for selective fusion.",
"Closed-loop EECTracker on a HOMATracker base with sliding-window association, plus plug-in transfer to ByteTrack, OC-SORT and FairMOT and Jetson Orin NX embedded validation.",
],
method: {
pipeline: ["select-reliable-tracklets", "encode-swarm-motion", "predict-plus-uncertainty", "aggregate-swarm-prior", "project-historical-features", "eec-score-and-mask", "fuse-compensated-features", "detect-and-associate", "feedback-tracks"],
architecture:
"HOMATracker-based joint detection and tracking loop. Shared backbone extracts adjacent-frame multi-scale features; compensation runs on the stride-8 highest-resolution level; compensated set feeds detection, appearance-embedding and multi-frame homogeneous association branches; updated tracks refresh the motion-state buffer.",
motionModel: "EqMotion-based image-plane trajectory predictor over T=8 history frames with global-plus-residual velocity encoding and an appended two-layer MLP uncertainty head predicting per-tracklet Gaussian offset distribution.",
appearanceModel: "Appearance embeddings from compensated features for association cost alongside motion consistency; high-confidence detections matched by Hungarian on appearance plus motion, low-confidence by spatial IoU.",
association: "HOMATracker multi-frame homogeneous strategy in a sliding window: high-confidence detections via Hungarian on appearance-motion cost, unmatched tracks considered against low-confidence detections, new tracks from leftovers.",
detectionDependency: "YOLOX-X unified front-end for TBD comparisons; JDT and end-to-end baselines keep original designs; inference resized to 1088 by 1088 with 0.01 confidence and 0.7 NMS thresholds.",
trackManagement: "Reliable tracklet defined as valid observation every frame in window with stable identity; if none available, PSMP and EEC are bypassed and raw features retained.",
optimization: "PSMP pretrained on auxiliary STP segments (L=20, 8 history plus 12 future) with position MSE plus NLL loss; full framework SGD with detection, objectness, box and appearance losses; EEC gating parameters initialized alpha 1.0, tau 0.8.",
},
equations: [
{
id: "eec-prior",
label: "Probabilistic swarm motion prior",
formula: "Mt = {v_sw_hat_t, N(mu_sw_hat, Sigma_sw_hat)}; mu_sw_hat = sum_i alpha_i mu_i_hat; Sigma_sw_hat = sum_i alpha_i (Sigma_i_hat + (mu_i_hat - mu_sw_hat)(mu_i_hat - mu_sw_hat)^T)",
variables: [
{ symbol: "v_sw_hat_t", meaning: "predicted global swarm velocity displacement at frame t" },
{ symbol: "mu_i_hat, Sigma_i_hat", meaning: "predicted mean offset and spatial covariance of reliable tracklet i" },
{ symbol: "alpha_i", meaning: "directional-consistency weight of tracklet i against the swarm velocity" },
{ symbol: "Mt", meaning: "swarm motion prior guiding historical feature projection" },
],
intuition: "Average where the swarm as a whole is drifting, plus how uncertain each member is, weighted toward members moving with the swarm.",
why: "Gives dense pixel-level reachable regions for feature projection without needing a good observation of every UAV.",
where: "Section III-B, Equations 5 through 8; consumes EqMotion predictions over an 8-frame window.",
},
{
id: "eec-score",
label: "Local EEC consistency score and soft mask",
formula: "Gc(pt) = Ec(pt) + lambda_s Sc(pt); Mc(pt) = HardSigmoid(alpha_c (tau_c - Gc(pt)))",
variables: [
{ symbol: "Ec", meaning: "channel-wise expected squared cross-frame residual energy under the displacement distribution" },
{ symbol: "Sc", meaning: "channel-wise local residual entropy of the normalized consistency distribution" },
{ symbol: "Gc", meaning: "Local EEC score; lower means stronger motion-prior-conditioned consistency" },
{ symbol: "Mc", meaning: "channel-wise soft consistency mask weight" },
{ symbol: "alpha_c, tau_c", meaning: "learnable per-channel scale and consistency threshold" },
{ symbol: "lambda_s", meaning: "balance factor between energy and entropy terms, fixed 0.1" },
],
intuition: "Keep historical help where it agrees with the current frame in both strength and local concentration; mute it where it looks like background noise.",
why: "Projection alone also carries background interference, so fusion needs an explicit reliability gate.",
where: "Section III-C, Equations 14 through 21; 3 by 3 window on AIRMOT and 7 by 7 on UAVSwarm.",
},
{
id: "eec-fusion",
label: "Compensated feature fusion",
formula: "F1_hat_t = phi_fu([M * F1_tilde_{t-1->t}; F1_t]) + F1_t",
variables: [
{ symbol: "F1_tilde", meaning: "motion-prior-projected historical feature map" },
{ symbol: "M", meaning: "channel-wise soft consistency mask" },
{ symbol: "F1_t", meaning: "current-frame highest-resolution feature map" },
{ symbol: "phi_fu", meaning: "3 by 3 convolutional fusion layer" },
{ symbol: "F1_hat_t", meaning: "compensated feature replacing F1_t downstream" },
],
intuition: "Add back a cleaned version of where the swarm used to be, on top of the weak current view.",
why: "Reinforces potential target regions while attenuating motion-inconsistent background before detection.",
where: "Section III-C, Equation 22; fused set feeds detection and ReID branches.",
},
{
id: "eec-mota",
label: "MOTA tracking accuracy",
formula: "MOTA = 1 - sum_t (FN_t + FP_t + IDSW_t) / sum_t GT_t",
variables: [
{ symbol: "FN_t, FP_t", meaning: "missed and spurious objects in frame t" },
{ symbol: "IDSW_t", meaning: "identity switches in frame t" },
{ symbol: "GT_t", meaning: "ground-truth object count in frame t" },
],
intuition: "Fraction of everything that went wrong relative to how many targets existed.",
why: "Primary overall accuracy alongside IDF1, HOTA and AP50 on both swarm benchmarks.",
where: "Section IV-B, Equation 24; IDF1, HOTA and AP50 defined in Equations 23 through 27.",
simulator: "metrics",
},
],
datasets: ["others"],
metrics: ["mota", "idf1", "hota", "idsw", "precision", "recall", "fps"],
baselines: ["SCT-MOT", "HOMATracker", "ByteTrack", "OC-SORT", "DeepSORT", "FairMOT", "Hybrid-SORT", "BELGTracker", "UAVS-MOT", "MOTRv3"],
results: [
"AIRMOT test (Tab. I): EECTracker 36.19 MOTA, 31.94 IDF1, 26.15 HOTA, 400 IDSW at 22.4 FPS vs SCT-MOT 32.30, 30.15, 25.54, 476 and HOMATracker 30.08, 29.31, 25.55, 506; gains over SCT-MOT of 3.89, 1.79 and 0.61 points with 76 fewer switches.",
"UAVSwarm test (Tab. I): 84.71 MOTA, 90.19 IDF1, 69.16 HOTA, 54 IDSW at 22.9 FPS vs SCT-MOT 81.90, 88.45, 68.56, 56; gains of 2.81, 1.74 and 0.60 points.",
"Compensation ablation (Tab. II): swarm-prior guidance reaches 46.10 AP50 and 36.19 MOTA on AIRMOT vs 44.80 and 31.84 for detector-location guidance and 41.60 and 30.08 baseline; on UAVSwarm 89.30 and 84.71 vs 87.10 and 80.05.",
"PSMP plus EEC vs SCT-MOT components (Tab. III): under the same backbone, PSMP-plus-TG-STFF already beats SMTP-plus-TG-STFF, and swapping in EEC adds 1.27 MOTA and 0.58 HOTA on AIRMOT and 2.26 MOTA and 0.88 HOTA on UAVSwarm.",
"Plug-in (Tab. IV): ByteTrack gains 2.90 MOTA on AIRMOT and 3.18 on UAVSwarm; FairMOT gains 2.53 and 1.61 MOTA with IDSW falling on both sets; OC-SORT also improves with fewer switches.",
"Embedded (Tab. XII, Jetson Orin NX TensorRT): EECTracker-Light 31.68 MOTA at 21.0 FPS on AIRMOT vs ByteTrack-Light 23.46 at 24.6 FPS; 78.73 at 21.3 FPS on UAVSwarm vs 63.71 at 23.9 FPS.",
],
ablations: [
"EEC score (Tab. IX): full Local EEC beats energy-only by 0.59 MOTA on AIRMOT and 0.84 on UAVSwarm, and beats direct fusion and entropy-only on almost all metrics.",
"Window size (Tab. X): AIRMOT prefers 3 by 3 while UAVSwarm prefers 7 by 7; oversized windows on AIRMOT admit background interference.",
"Gating (Tab. XI): channel-adaptive beats shared gating by 0.50 MOTA on AIRMOT and 0.24 on UAVSwarm with 34 and 19 fewer switches.",
"Prior form (Tabs. V and VIII): swarm-aware learned prior beats individual learned and Kalman priors; compact moment-matched Gaussian beats explicit mixture by 0.11 and 0.43 MOTA on the two sets.",
"Robustness (Tabs. VI and VII): 50 percent reliable tracklets keep most gains; online tracklets match ground-truth-history priors within 0.15 MOTA; 0 percent reduces to baseline.",
],
limitations: {
authorStated: [
"Effectiveness depends on having some reliable historical tracklets; with none available the prior and compensation are bypassed entirely.",
"Optimal local window is dataset-dependent (3 by 3 vs 7 by 7), so new scenes need window retuning.",
"Future work targets more robust swarm motion modeling under severely limited historical observations.",
],
evident: [
"Training pipeline is two-stage with frozen PSMP during full-framework training, so compensation cannot correct systematic prior errors end to end.",
"All main results use one RTX 3090 setup and fixed 1088 input; embedded numbers come from a separate lightweight YOLOX-S variant rather than the full model.",
],
},
assumptions: [
"UAVs in a swarm share short-term image-plane motion tendency so aggregation across reliable members is informative.",
"Lower residual energy plus lower local entropy implies motion-consistent target rather than background.",
],
computation:
"Single NVIDIA RTX 3090 GPU; PSMP SGD 60 epochs at 5e-4; full framework SGD at 0.002 on 1088 by 1088 inputs; online inference about 22 FPS on RTX 3090 and above 20 FPS TensorRT on Jetson Orin NX lightweight variant.",
relations: [
{ to: "T061", type: "uses-as-baseline", note: "Compares against ByteTrack with unified YOLOX-X detector and plugs compensation into it." },
{ to: "T073", type: "uses-as-baseline", note: "Compares against OC-SORT and plugs the same compensation into its detector." },
{ to: "T043", type: "uses-as-baseline", note: "Compares against FairMOT joint detection-ReID framework and plugs compensation before its heads." },
],
concepts: ["tracking-by-detection", "motion-model", "data-association", "reid", "appearance-features", "occlusion", "mota", "idf1", "hota"],
impact:
"Shows swarm-level shared motion is a usable spatial prior for feature-level rescue of tiny intermittent UAV targets, with a plug-in design reusable across TBD and JDT trackers.",
},
{
id: "T157",
arxiv: "2609.16662",
title: "SAVTrack: Selective Vote Aggregation for Reliability-Aware Point Cloud Tracking",
shortTitle: "SAVTrack",
year: 2026,
authors: ["Sifan Zhou", "Linyue Tan", "Qiwei Wang", "Ziyu Zhao", "Xiaobo Lu"],
fileName: "2609.16662v1.pdf",
task: "3d-tracking",
tags: ["3d-single-object-tracking", "point-cloud", "vote-aggregation", "motion-centric", "reliability-gating", "real-time"],
difficulty: "advanced",
summary:
"Motion-centric 3D single-object tracker that makes point-to-center voting reliability explicit. On top of a P2P part-to-part motion representation, each seed predicts a posterior over 12 center-relative sub-regions plus region-specific center offsets; the posterior acts as vote confidence for hard pre-aggregation gating before VoteNet-style clustering. Retains about 250 of 1536 votes, improves P2P-point from 66.2 to 68.4 Success on KITTI and 55.92 to 58.44 on nuScenes, at 82 FPS.",
problem:
"In sparse incomplete LiDAR views, seeds on planar ambiguous surfaces cast far noisier center votes than seeds on corners and intersections, but dense voting pipelines aggregate every hypothesis equally so bad votes corrupt neighborhood construction and proposal formation.",
background: ["3d-tracking", "bounding-box", "sot", "motion-model", "iou", "attention", "benchmark-design", "success-plot"],
previousWork: [
{
name: "Siamese appearance-matching 3D trackers (SC3D, P2B, BAT, PTTR, STNet, CXTrack, MBPTrack)",
limitation: "Improve template-search correspondence but pass all seed hypotheses to clustering without modeling per-vote reliability, and degrade under sparse incomplete observations.",
whyThisPaper: "Keeps point-wise voting but inserts a supervised reliability gate before clustering.",
},
{
name: "Motion-centric trackers (M2Track, P2P part-to-part, VoxelTrack, DMT)",
limitation: "Reduce appearance dependence via inter-frame motion yet still aggregate motion-based hypotheses indiscriminately.",
whyThisPaper: "Adopts P2P-point as baseline and asks whether each seed-to-center hypothesis should enter clustering at all.",
},
{
name: "Selective voting in single-frame detection (SPOT)",
limitation: "Static local-geometry-only confidence, unsuited to tracking where vote quality depends on temporal motion context.",
whyThisPaper: "Estimates posterior from joint point-wise plus broadcast inter-frame motion features and compares static-only against motion-aware gating.",
},
],
researchGap:
"No lightweight target-conditioned temporal reliability estimate with hard pre-aggregation removal of unreliable center votes in 3D single-object tracking.",
contribution: [
"Selective Vote Aggregation module: per-seed sub-region posterior classifier plus region-specific vote regressors with hard threshold gating at tau 0.3 and empty-vote top-K fallback.",
"Motion-aware seed representation fusing PointNet++ point features with broadcast P2P part-to-part motion feature, trained with vote classification plus masked L1 regression plus proposal losses.",
"Systematic reliability evidence: confidence-error calibration, geometry correlation, aggregation-strategy controls and point-count-stratified sparsity analysis on KITTI and nuScenes.",
],
method: {
pipeline: ["crop-template-search", "extract-point-features", "fuse-part-to-part-motion", "sample-seeds", "predict-subregion-posterior", "regress-region-votes", "hard-gate-votes", "cluster-and-refine-proposals", "select-best-box"],
architecture:
"P2P-point motion-centric baseline with shared PointNet++ encoder, cascaded 1D-conv part-to-part fusion, SAV head (small MLP classifier plus region regressors), VoteNet-style ball-query clustering and MLP proposal head predicting center plus yaw with fixed inherited box size.",
motionModel: "Inter-frame part-to-part relative-offset representation broadcast to every seed; no explicit Kalman or box-history predictor.",
appearanceModel: "Point-wise local geometry plus motion context; template from previous frame, search cropped around previous box with margin.",
association: "Not applicable in single-object scope; proposal objectness selects the winning box per frame.",
detectionDependency: "None; first-frame 3D box given, size fixed to [h, w, l] throughout the sequence.",
trackManagement: "Frame-to-frame box handoff with search-region cropping; proposal seeds capped at 256 with 0.3 m grouping radius and 0.3/0.6 m positive/negative labeling.",
optimization: "Adam, learning rate 1e-4, batch 128, 30 epochs; loss weights lambda-reg, gamma-1 and gamma-2 all 1.0; default NR 12 sub-regions and tau 0.3.",
},
equations: [
{
id: "sav-seed",
label: "Motion-aware seed representation",
formula: "f_tilde_i = [f_i; F_m], for i = 1..Ns",
variables: [
{ symbol: "f_i", meaning: "PointNet++ point-wise feature of seed i" },
{ symbol: "F_m", meaning: "broadcast inter-frame part-to-part motion representation" },
{ symbol: "f_tilde_i", meaning: "motion-aware seed feature used for posterior and vote regression" },
{ symbol: "Ns", meaning: "number of FPS-sampled seed points" },
],
intuition: "Give every local point a summary of how the whole target moved, so confidence knows about time as well as shape.",
why: "Vote reliability in tracking depends on temporal context, not just static local geometry.",
where: "Section 3.2, Equation 1; seeds sampled from current-frame crop.",
},
{
id: "sav-posterior",
label: "Sub-region posterior and vote regression",
formula: "p_i = Softmax(g(f_tilde_i)); d_hat_ij = z_i + phi_j(f_tilde_i); L_vote = L_cls + lambda_reg L_reg",
variables: [
{ symbol: "p_i", meaning: "posterior distribution over NR center-relative sub-regions for seed i" },
{ symbol: "z_i", meaning: "3D coordinate of seed i" },
{ symbol: "d_hat_ij", meaning: "predicted center from seed i under sub-region j" },
{ symbol: "phi_j", meaning: "region-specific MLP offset regressor" },
{ symbol: "L_cls, L_reg", meaning: "cross-entropy sub-region loss and masked L1 offset loss" },
],
intuition: "Ask each point which direction the center lies in and how far, then trust only the answers the network is confident about.",
why: "Supervised sub-region prediction yields a calibrated confidence proxy without needing explicit reliability labels.",
where: "Section 3.3, Equations 2 through 9; NR 12 by default.",
},
{
id: "sav-gating",
label: "Hard pre-aggregation gating and clustering",
formula: "V_SAV = {v_ij | p_ij > tau}; C_k = {v_ij in V_SAV | ||d_hat_ij - q_k||_2 < r_p}; (b_k, s_k) = Theta(C_k)",
variables: [
{ symbol: "v_ij", meaning: "candidate vote tuple of predicted center and posterior confidence" },
{ symbol: "tau", meaning: "confidence threshold, default 0.3" },
{ symbol: "q_k", meaning: "k-th proposal seed center" },
{ symbol: "r_p", meaning: "ball-query grouping radius, 0.3 m" },
{ symbol: "b_k, s_k", meaning: "proposal box state and objectness score" },
],
intuition: "Delete weak guesses before they get to vote on the final box, so clusters form only from trustworthy evidence.",
why: "Unlike soft reweighting, hard gating changes the candidate set and protects neighborhood construction.",
where: "Section 3.3 through 3.4, Equations 8 through 11; empty-set fallback to top-K0 ranked votes.",
simulator: "track-mgmt",
},
],
datasets: ["kitti-tracking", "others"],
metrics: ["success-auc", "precision", "fps"],
baselines: ["P2P-point", "P2P-voxel", "M2Track", "M2Track++", "VoxelTrack", "FocusTrack", "CompTrack", "STNet", "CXTrack", "MBPTrack", "MoCUT", "OSP2B", "GLT-T"],
results: [
"KITTI mean (Tab. 1): SAVTrack 68.4 Success and 87.4 Precision at 82 FPS on RTX 3090 vs P2P-point 66.2 and 85.4; per-category Car 71.1/83.9, Pedestrian 65.1/91.8, Van 68.0/82.5, Cyclist 75.9/94.8 matching the table best on Cyclist precision.",
"nuScenes mean (Tab. 2, 117278 frames): 58.44/69.82 vs P2P-point 55.92/66.64 for gains of 2.52 and 3.18; per-category Car 64.27/71.72, Pedestrian 42.54/68.31, Truck 65.16/66.18, Trailer 72.60/70.48, Bus 64.02/61.67.",
"Aggregation control on KITTI (Tab. 4): dense 65.7/85.9 with about 1536 votes, soft weighting 66.0/86.3, top-1 66.4/86.8 with 128 votes, SAV 68.4/87.4 with about 250 votes.",
"Component split (Tab. 5): multi-hypothesis regression alone 67.2/87.0 with 1536 votes; adding confidence filtering reaches 68.4/87.4 with about 250 votes.",
"Sparsity stratification on nuScenes (Tab. 11): gain over baseline grows from 2.3/2.2 above 50 points to 5.2/6.4 in the 0-10 point bin; retained votes fall to about 62 in the sparsest bin.",
"Cost (Tab. 9): 7.76M vs 7.39M baseline parameters; 82 vs 98 FPS at batch 1 on RTX 3090.",
],
ablations: [
"Threshold (Tab. 6): tau 0.3 best at 68.4/87.4; tau 0.1 keeps about 820 votes at 67.5/87.7; tau 0.5 keeps about 80 votes but drops to 68.1/87.2.",
"Sub-regions (Tab. 7): NR 12 best on KITTI Car at 71.1/83.9; NR 4 gives 69.8/82.6 and NR 24 drops to 70.3/83.0.",
"Motion conditioning (Tab. 12): static-only posterior 69.1/82.0 Car and 62.8/89.4 Pedestrian vs motion-aware SAV 71.1/83.9 and 65.1/91.8; hard gating beats soft weighting by 1.9/1.7 on Car.",
"Calibration and geometry (Tabs. 3 and 8): mean error falls 0.72 m to 0.13 m from lowest to highest confidence quintile with success ratio 28.3 to 89.4 percent; planarity correlates minus 0.52 with confidence and plus 0.47 with error; curvature plus 0.49 and minus 0.43.",
],
limitations: {
authorStated: [
"Fixed uniformly defined center-relative partition may not capture anisotropic or object-dependent center distributions; adaptive or hierarchical partitions left to future work.",
"Reliability estimated per seed independently without explicit geometric consensus among retained hypotheses or temporal consistency verification.",
"Future extension to other point-cloud localization tasks with heterogeneous hypothesis quality.",
],
evident: [
"Voxel and BEV trackers (VoxelTrack, FocusTrack, CompTrack, P2P-voxel) still lead on absolute KITTI mean, so gains are within the point-based representation rather than across representations.",
"Category-level gains do not track point density monotonically (Bus and Trailer gain most despite dense points), so sparsity is isolated only via the stratified analysis.",
],
},
assumptions: [
"Target size fixed from initialization; tracking reduces to center plus heading estimation.",
"Posterior confidence is a valid proxy for geometric vote reliability.",
],
computation:
"PyTorch, Adam with batch 128 for 30 epochs; training on a single RTX 3090; runtime measured on the same hardware under identical inference protocol.",
relations: [
{ to: "T089", type: "uses-as-baseline", note: "Compares against STTracker spatio-temporal 3D tracker on nuScenes." },
{ to: "T153", type: "conceptual-successor", note: "Same-group template-free point-cloud tracking line (TFTrack) cited as efficient 3D tracking context." },
],
concepts: ["3d-tracking", "sot", "motion-model", "bounding-box", "iou", "success-plot", "attention", "benchmark-design"],
impact:
"Establishes pre-clustering reliability gating as a cheap complement to representation upgrades, with evidence that motion context plus hard removal matters most under sparsity.",
},
{
id: "T158",
arxiv: "2609.16695",
title: "MAETrack: Unleashing the Potential of Pretrained Geometric Priors for 3D Single Object Tracking",
shortTitle: "MAETrack",
year: 2026,
authors: ["Sifan Zhou", "Qiwei Wang", "Linyue Tan", "Ziyu Liu", "Ziyu Zhao", "Xiaobo Lu"],
fileName: "2609.16695v1.pdf",
task: "3d-tracking",
tags: ["3d-single-object-tracking", "masked-autoencoder", "transfer-learning", "bev", "geometric-prior", "lightweight-adaptation"],
difficulty: "advanced",
summary:
"Lightweight adaptation framework for moving BEV-MAE reconstruction pretraining into 3D single-object tracking. Layer-Selective Initialization loads only shallow backbone stages from Waymo-pretrained BEV-MAE and randomly initializes deeper stages, while Geometric Residual Gating applies a learnable residual spatial gate on search BEV features before template-search fusion. Improves the P2P baseline on KITTI (72.0/89.9 mean) and nuScenes (61.16/73.54) with negligible overhead at 84.3 FPS.",
problem:
"Naive full-network fine-tuning of MAE 3D encoders transfers reconstruction-specialized deep semantics along with geometry, causing negative transfer for tracking which needs instance-level spatial-temporal matching under sparse partial occluded views.",
background: ["3d-tracking", "sot", "bounding-box", "motion-model", "iou", "attention", "success-plot", "benchmark-design"],
previousWork: [
{
name: "Siamese 3D trackers (SC3D, 3D-SiamRPN, P2B, BAT, PTT) and memory-augmented MBPTrack",
limitation: "Rely on task-specific labeled supervision and appearance matching sensitive to texture-less sparse LiDAR.",
whyThisPaper: "Injects large-scale self-supervised geometric priors instead of stronger supervised matching.",
},
{
name: "Motion-centric trackers (M2Track, P2P part-to-part) used as the strong baseline",
limitation: "Strong but trained from scratch, so data-hungry and brittle under sparse geometry.",
whyThisPaper: "Keeps P2P motion modeling and head unchanged and isolates gains to backbone initialization plus search-side modulation.",
},
{
name: "3D MAE pretraining (Point-MAE, Voxel-MAE, GeoMAE, BEV-MAE) for detection and classification",
limitation: "Static intra-frame reconstruction success does not transfer automatically to discriminative cross-frame association.",
whyThisPaper: "Diagnoses layer-wise transfer mismatch with CKA and geometric probes and adapts selectively.",
},
],
researchGap:
"No tracking-oriented transfer principle that preserves shallow MAE geometry while freeing deep layers and reinforcing salient search regions for 3D SOT.",
contribution: [
"Layer-wise transfer diagnosis: CKA similarity plus frozen-stage linear probes for foreground occupancy and boundary prediction across backbone stages.",
"Layer-Selective Initialization: binary depth mask inheriting first three stages from BEV-MAE with all parameters left trainable for geometry-aware optimization trajectory.",
"Geometric Residual Gating: search-only 1 by 1 conv plus sigmoid gate with learnable scalar residual scaling before template-search fusion.",
],
method: {
pipeline: ["crop-template-search", "voxelize-bev", "lsi-backbone-encode", "gate-search-features", "fuse-template-search", "regress-relative-motion"],
architecture:
"Single unified sparse-conv tracking backbone (not dual-stream) with BEV features, GRG on the search branch, channel-concat plus 3-conv fusion head and VoxelHead-style regression of relative motion [dx, dy, dz, dr] applied to the previous box.",
motionModel: "Part-to-part motion modeling inherited from P2P baseline; prediction head and supervision unchanged so transfer effects are isolated.",
appearanceModel: "BEV geometric features with LSI shallow priors; template branch left ungated as a stable reference.",
association: "Not applicable in single-object scope; frame-to-frame relative-motion chaining.",
detectionDependency: "None; first-frame 3D box given, size inherited from template.",
trackManagement: "Template from frame t-1 and search from frame t around previous prediction with category-dependent crop ranges and reference-box perturbation augmentation.",
optimization: "AdamW 2e-4 with cosine decay to 1e-6, weight decay 1e-5, gradient clip 35, mixed precision, batch 256 on one RTX 4090; tracking loss lambda-weighted x-y, z and rotation terms.",
},
equations: [
{
id: "mae-lsi",
label: "Layer-Selective Initialization mask",
formula: "theta_i_init = m_i theta_i_MAE + (1 - m_i) theta_i_rand; m_i = 1 if i <= k else 0",
variables: [
{ symbol: "theta_i_MAE", meaning: "BEV-MAE pretrained parameters of backbone stage i" },
{ symbol: "theta_i_rand", meaning: "random initialization for stage i" },
{ symbol: "m_i", meaning: "binary transfer mask selecting inherited stages" },
{ symbol: "k", meaning: "number of inherited shallow stages, default input conv plus first two residual stages" },
],
intuition: "Keep the foundation layers that know shapes and edges; let upper layers relearn tracking-specific logic from scratch.",
why: "Shallow layers hold transferable occupancy and boundary cues while deep layers are reconstruction-specialized.",
where: "Section 3.4, Equations 5, 7 and 8; Stage 1 denotes input sparse conv, Stages 2-4 residual stages.",
},
{
id: "mae-grg",
label: "Geometric Residual Gating",
formula: "R_s = sigmoid(W_g * F_search + b_g); F_search_gated = F_search * (1 + alpha R_s)",
variables: [
{ symbol: "F_search", meaning: "search-branch BEV feature map" },
{ symbol: "R_s", meaning: "predicted spatial geometric reliability map in [0,1]" },
{ symbol: "W_g, b_g", meaning: "1 by 1 convolution weights and bias of the spatial gate" },
{ symbol: "alpha", meaning: "learnable scalar reinforcement strength, initialized 1e-3" },
],
intuition: "Turn up the volume on boundary and occupied pixels while leaving the pretrained map as the dominant signal.",
why: "Spatial reliability is non-uniform under sparsity, so structurally salient regions deserve reinforcement without replacing priors.",
where: "Section 3.5, Equations 7 through 9; applied search-only before fusion.",
},
{
id: "mae-fusion-loss",
label: "Fusion and tracking objective",
formula: "F_fusion = Conv_head(Concat(F_template, F_search_gated)); L_track = l1 L_xy + l2 L_z + l3 L_rot",
variables: [
{ symbol: "F_fusion", meaning: "fused template-search representation fed to the tracking head" },
{ symbol: "L_xy, L_z, L_rot", meaning: "center, height and yaw regression loss terms" },
{ symbol: "l1, l2, l3", meaning: "balancing weights of the position and orientation terms" },
],
intuition: "Compare the stable template against the cleaned search view, then nudge the previous box by the predicted motion.",
why: "Keeps supervision identical to P2P so measured gains come only from initialization plus gating.",
where: "Sections 3.5 and 3.6, Equations 10 and 11; inference chains relative motion frame by frame.",
},
],
datasets: ["kitti-tracking", "others"],
metrics: ["success-auc", "precision", "fps"],
baselines: ["P2P", "M2Track", "M2Track++", "VoxelTrack", "PTTR", "PTTR++", "GLT-T", "MBPTrack", "SyncTrack", "VPMCAN", "V2B", "BAT"],
results: [
"KITTI mean (Tab. 1): MAETrack 72.0/89.9 vs reimplemented P2P 71.1/88.9; Car 75.2/87.5 vs 73.2/85.4 for plus 2.0/2.1; Pedestrian 68.8/93.2, Van 70.5/84.6, Cyclist 75.5/94.7.",
"nuScenes mean (Tab. 2): 61.16/73.54 vs P2P 59.22/71.19; Car 66.05/73.57 vs 64.61/71.98; Truck 68.50/70.53 vs 64.42/65.37; Trailer 74.10/71.63 vs 70.23/66.08 for plus 3.87/5.55; Bus 62.11/60.86 vs 58.54/56.13 for plus 3.57/4.73.",
"Speed (Tab. 3, RTX 4090 Car): 84.3 FPS vs P2P 88.9, M2Track 51.2, VoxelTrack 36.0.",
"Sparsity (Tab. 4, Car): KITTI 0-10 point bin 67.1/77.7 vs P2P 64.8/75.0 and 10-20 bin 71.7/85.8 vs 70.7/83.6; nuScenes 0-10 bin 64.3/71.7 vs 63.3/71.4; dense bins nearly tied.",
"Module split (Tab. 5): on nuScenes Car, LSI alone 65.22/72.43 and GRG alone 65.09/72.78 over baseline 64.61/71.98, combined 66.05/73.57.",
],
ablations: [
"Initialization depth (Tab. 6): full-network init only reaches 65.09/72.58 on nuScenes Car vs LSI Stage 1+2+3 at 66.05/73.57; Only-Stage-1 65.39/72.90 beats Only-Stage-3 63.65/71.23 and Only-Stage-4 63.14/69.60.",
"Representation diagnosis (Tab. 7, nuScenes Car): CKA falls 0.84, 0.71, 0.52, 0.36 from Stage 1 to 4; foreground F1 72.8 to 62.5 and boundary IoU 44.6 to 35.1, matching the only-stage transfer ranking.",
"GRG vs attention (Tab. 8): SE, CBAM and spatial-only CBAM give limited gains over LSI-only 65.22/72.43; residual gate 65.86/73.31; GRG with learnable alpha best at 66.05/73.57.",
"GRG placement (Tab. 9): search-only 75.20/87.50 KITTI Car and 66.05/73.57 nuScenes Car beats template-only, both-branch and after-fusion variants.",
],
limitations: {
authorStated: [
"Single pretrained source only; multi-modal camera-LiDAR or heterogeneous pretraining not explored.",
"No explicit temporal modeling in GRG, which may limit fast motion and severe occlusion handling.",
"CNN-backbone focus only; transformer-backbone extension and multi-object tracking generalization left to future work.",
],
evident: [
"KITTI Pedestrian shows no gain (68.8/93.2 vs baseline 69.0/93.4) and the KITTI 40-50 point bin regresses on 3 sequences and 80 frames, so transfer benefit is category- and density-dependent.",
"KITTI has only 19 training and 2 validation sequences, so small-split numbers are less stable than the 700/150-sequence nuScenes evaluation.",
],
},
assumptions: [
"Shallow MAE layers encode transferable occupancy and boundary geometry while deep layers encode reconstruction semantics.",
"Search branch is noisier than the template reference, justifying search-only reinforcement.",
],
computation:
"One RTX 4090 GPU, global batch 256, mixed precision; nuScenes car/pedestrian 20 epochs, truck/bus 50, trailer 100, KITTI 50; validation each epoch with best-precision checkpoint.",
relations: [
{ to: "T153", type: "conceptual-successor", note: "Same-group efficient 3D tracking line; TFTrack cited as template-free point-cloud tracking context." },
],
concepts: ["3d-tracking", "sot", "motion-model", "bounding-box", "success-plot", "attention", "benchmark-design", "iou"],
impact:
"Suggests 3D foundation-model value lies in hierarchical shallow geometry with tracking-specific adaptation, motivating selective rather than full fine-tuning.",
},
{
id: "T159",
arxiv: "2609.17427",
title: "Tracking the Unseen: An Occlusion-Robust Framework for Target Tracking Under Full and Long-Term Occlusion",
shortTitle: "Occlusion-Robust YOLO-KF-OAMN",
year: 2026,
authors: ["Mais Mohammed", "Sharifa Mohammed", "Hanan Awadh", "Haneen Bamaas", "Raghad Bawazeer", "Elham Alghamdi"],
fileName: "2609.17427v1.pdf",
task: "multi-object",
tags: ["occlusion-handling", "tracking-by-detection", "re-identification", "kalman-filter", "yolo", "surveillance"],
difficulty: "intro",
summary:
"Lightweight real-time tracking-by-detection pipeline for full and long-term occlusion: fine-tuned YOLOv11n detection, 7D constant-velocity Kalman prediction with occlusion-duration process-noise inflation, and Occlusion-Aware Mask Network Re-ID with an IoU-gate bypass beyond 30 occluded frames up to 150. Six Re-ID architectures compared under identical conditions select OAMN. Beats OccluTrack on OVIS (0.477 vs 0.404 MOTA, 0.773 vs 0.618 IDF1) and on a 64-video custom soldier dataset (0.734 vs 0.643 MOTA).",
problem:
"Conventional detectors plus IoU association terminate trajectories once targets vanish behind structures or terrain, losing identity and situational awareness; defense surveillance needs continuity through full disappearance and long gaps plus correct identity recovery at reappearance.",
background: ["mot", "tracking-by-detection", "detection", "kalman", "state-space", "motion-model", "data-association", "hungarian", "iou", "reid", "appearance-features", "occlusion", "track-management", "mota", "idf1"],
previousWork: [
{
name: "SORT Kalman plus Hungarian IoU association",
limitation: "Geometric-only with no appearance memory, so overlapping or briefly hidden targets cause switches and premature termination.",
whyThisPaper: "Keeps its motion core but adds long-lived tracks plus appearance gallery for re-acquisition.",
},
{
name: "Deep hybrids (DeepSORT, StrongSORT, ByteTrack two-stage, BoT-SORT motion compensation, TrackFormer)",
limitation: "More resilient yet still assume near-continuous visibility and struggle to preserve identity across full long-term gaps.",
whyThisPaper: "Extends the hybrid recipe with occlusion-duration-aware prediction and a masking Re-ID explicitly built for hidden targets.",
},
{
name: "Occlusion systems led by OccluTrack with abnormal-motion suppression and pose-guided Re-ID",
limitation: "Strongest published long-gap baseline but unevaluated under heavy unpredictable defense occlusion with similar-looking targets.",
whyThisPaper: "Benchmarks head-to-head against OccluTrack on OVIS and a purpose-built soldier occlusion dataset.",
},
],
researchGap:
"No lightweight real-time pipeline jointly combining detection, gap prediction and systematically selected occlusion-aware Re-ID evaluated for full plus long-term occlusion in defense-like scenes.",
contribution: [
"Three-stage real-time pipeline: YOLOv11n detection (tau-conf 0.50, NMS 0.45), 7D Kalman filter with adaptive Q inflation over up to 150 frames, and OAMN Re-ID with 30-frame IoU-gate bypass and 0.40/0.60 IoU-appearance association weights.",
"Controlled six-way Re-ID comparison (OccludedReID, PGFA, HOReID, PAT, TransReID, OAMN) on shared MobileNetV2 backbone, hyperparameters and 80/20 OVIS split.",
"Custom 64-video 8945-frame soldier dataset with 111-183-frame full-hidden intervals, CVAT COCO annotations and hybrid synthetic plus real footage, with code released on GitHub.",
],
method: {
pipeline: ["detect-with-yolov11n", "kalman-predict-update", "inflate-noise-under-occlusion", "associate-iou-plus-appearance", "bypass-iou-when-long-occluded", "reidentify-with-oamn-gallery"],
architecture:
"Tracking-by-detection with backbone-neck-head YOLOv11n (C3k2 block, about 2.6M params, 640 input), per-track 7D Kalman filter, OAMN two-branch appearance gallery with cosine matching, and combined IoU-appearance cost matrix.",
motionModel: "Constant-velocity 7D Kalman state [x, y, s, r, x-dot, y-dot, s-dot]; predict-only during occlusion with process noise scaled 1 plus 2 min(t-occ,150)/150, reset on re-match; 3 hits to confirm.",
appearanceModel: "OAMN: convolutional feature map F with full-image average descriptor plus visibility-masked descriptor from a learned sigmoid mask, fused by a scalar visibility weight alpha through a projection head to a 512D L2 embedding.",
association: "Hungarian on 0.40 IoU plus 0.60 appearance cost; IoU gate bypassed for tracks occluded over 30 frames so appearance-only matching up to 150 frames (10 s at 15 fps) can resume identity.",
detectionDependency: "YOLOv11n fine-tuned on OVIS (and separately on soldier data to 71.1 percent mAP50); 3.12 detections per frame at 15.41 FPS selected over YOLOv8n, v9t and v10n.",
trackManagement: "Confirmed after 3 consecutive matches; unmatched occluded tracks predicted forward up to 150-frame lifetime instead of termination; gallery embeddings stored from visible phase.",
optimization: "Re-ID training: Adam, 3e-4 with cosine decay, batch P-by-K triplet sampling, margin 0.3, 128 by 256 crops, 15 epochs with early-stop patience 4, seed 42.",
},
equations: [
{
id: "unseen-detect",
label: "Detection set with confidence gate",
formula: "Dt = {(b_i, c_i, s_i)}, s_i >= tau_conf with tau_conf = 0.50 and NMS IoU 0.45",
variables: [
{ symbol: "b_i", meaning: "bounding box [x1,y1,x2,y2] of detection i" },
{ symbol: "c_i", meaning: "predicted class label" },
{ symbol: "s_i", meaning: "detection confidence score" },
{ symbol: "tau_conf", meaning: "confidence threshold, 0.50" },
],
intuition: "Keep only confident non-duplicate boxes as the visible evidence; everything downstream depends on them.",
why: "Missed or duplicate boxes fragment trajectories beyond what motion or Re-ID can repair.",
where: "Section 3.2.1, Equation 1; YOLOv11n single forward pass per frame.",
},
{
id: "unseen-kalman",
label: "Kalman predict-update with occlusion inflation",
formula: "x_{k|k-1} = F x_{k-1|k-1}; x_{k|k} = x_{k|k-1} + K(z_k - H x_{k|k-1}); Q_k = Q_base (1 + 2 min(t_occ,150)/150)",
variables: [
{ symbol: "x", meaning: "7D track state [x,y,s,r,x-dot,y-dot,s-dot]" },
{ symbol: "F, H, K", meaning: "state transition, measurement map and Kalman gain" },
{ symbol: "z_k", meaning: "matched detection observation, skipped during occlusion" },
{ symbol: "Q_k", meaning: "process-noise covariance inflated with occlusion duration t_occ" },
],
intuition: " coast the box forward on velocity when blind, and admit growing uncertainty the longer blindness lasts.",
why: "Bridges visibility gaps with a stable training-free predictor while quantifying drift for later appearance-only recovery.",
where: "Section 3.2.2, Equations 2 through 5; correction skipped when no detection matches.",
simulator: "kalman",
},
{
id: "unseen-reid",
label: "OAMN masked embedding and cosine recovery",
formula: "f_vis = AvgPool(F * 1[M >= tau]); f_fused = [(1-alpha) f_full, alpha f_vis]; sim = (f_q . f_g)/(||f_q|| ||f_g||)",
variables: [
{ symbol: "F", meaning: "backbone feature map of the detection crop" },
{ symbol: "M", meaning: "learned per-pixel visible-vs-occluder confidence map" },
{ symbol: "f_full, f_vis", meaning: "global-average and visible-only descriptors" },
{ symbol: "alpha", meaning: "learned scalar visibility weight favoring f_vis under heavy occlusion" },
{ symbol: "sim", meaning: "cosine similarity between query and gallery embeddings" },
],
intuition: "Describe only the pixels believed to be the person, then match the reappearing crop to the stored look.",
why: "Full-box pooling absorbs occluders and shifts embeddings across the gap; masking keeps gallery and query comparable.",
where: "Section 3.2.3, Equations 6 through 11; gallery built during visible phase.",
simulator: "reid",
},
{
id: "unseen-mota",
label: "MOTA and IDF1 evaluation",
formula: "MOTA = 1 - (FN + FP + ID-switches)/GT; IDF1 = 2 P_ID R_ID/(P_ID + R_ID) overThrottle IDTP/IDFP/IDFN matching",
variables: [
{ symbol: "FN, FP", meaning: "missed and spurious objects" },
{ symbol: "ID-switches", meaning: "identity handoff errors" },
{ symbol: "GT", meaning: "total ground-truth objects" },
{ symbol: "IDTP, IDFP, IDFN", meaning: "identity-level true/false positives and misses via Hungarian matching" },
],
intuition: "MOTA counts per-frame errors while IDF1 rewards keeping one ID on one person for the whole video.",
why: "Jointly measures coverage and long-term identity preservation under occlusion.",
where: "Section 4.1, Equations 12 and 13; ADE and normalized ADE also reported.",
simulator: "metrics",
},
],
datasets: ["others"],
metrics: ["mota", "idf1", "idsw", "fps"],
baselines: ["OccluTrack", "Kalman-plus-IoU no-ReID", "OccludedReID", "PGFA", "HOReID", "PAT", "TransReID", "YOLOv8n", "YOLOv9t", "YOLOv10n", "Particle filter", "LSTM predictor", "Transformer predictor"],
results: [
"OVIS Re-ID selection (Tab. 2): OAMN best among Re-ID methods at 0.477 MOTA and 0.773 IDF1 with 34 switches and 0.249 ADE; followed by TransReID 0.462, HOReID 0.459, PAT 0.456, OccludedReID 0.453, PGFA 0.439; Kalman-plus-IoU baseline 0.13 MOTA with 39 switches.",
"OVIS full pipeline vs OccluTrack (Tab. 3): 0.477 vs 0.404 MOTA for plus 18.07 percent relative, 0.773 vs 0.618 IDF1 for plus 25.08 percent, 34 vs 39 switches for minus 12.82 percent, ADE tied 0.249 vs 0.248.",
"Military 12-clip evaluation (Tab. 5): 0.7340 vs 0.6429 MOTA for plus 14.17 percent relative, 0.7286 vs 0.6887 IDF1 for plus 5.79 percent, 256 vs 263 switches for minus 2.66 percent, ADE 0.0877 vs 0.0872.",
"Detector screening (App. A, 50 videos 3234 frames): YOLOv11n 3.12 detections per frame at 15.41 FPS vs YOLOv8n 3.06 at 15.80, YOLOv9t 2.91 at 7.97, YOLOv10n 2.64 at 13.84.",
],
ablations: [
"Filter screening (App. B): standard Kalman best identity stability (0.83 IDF1, 0.24 ADE, 39 switches) despite lowest MOTA 0.13; Transformer highest MOTA 0.54 but 0.33 IDF1 and 0.74 ADE; Enhanced Kalman, Particle and LSTM sit between.",
"Re-ID ordering differs by metric: HOReID fewest switches (29) but lower MOTA than OAMN; baseline IDF1 0.83 exceeds all Re-ID IDF1s because it tracks only easy identities, a coverage-vs-stability trade-off discussed explicitly.",
"ADE neutrality: all six Re-ID integrations keep ADE 0.248-0.256 vs baseline 0.24, confirming Re-ID does not degrade Kalman positional accuracy.",
],
limitations: {
authorStated: [
"Crowded scenes with similar appearance lower Re-ID confidence.",
"Only static-camera evaluation; moving-camera compensation not tested.",
"Constant-velocity Kalman assumption breaks under sudden nonlinear motion.",
],
evident: [
"Experiments use only the OVIS train split (607 videos) because validation and test annotations are null, so no held-out OVIS split evaluation is possible.",
"Proposed pipeline uses YOLOv11n while OccluTrack uses YOLOX, so detector choice is confounded with tracking differences despite matched thresholds and fine-tuning.",
],
},
assumptions: [
"Single soldier class focus in the military case study; OVIS covers 25 categories with labels every 5 frames.",
"Maximum 150-frame (10 s at 15 fps) memory with 30-frame IoU-bypass threshold suffices for targeted long gaps.",
],
computation:
"YOLOv11n selection timing on 3234 frames (e.g. 276.86 s total at 15.41 FPS); Re-ID crops 256 by 128 on MobileNetV2 backbone; inference auto-fallback across CUDA GPU, Apple MPS and CPU.",
relations: [
{ to: "T005", type: "builds-on", note: "Extends SORT Kalman-plus-Hungarian recipe with noise inflation and appearance recovery." },
{ to: "T061", type: "uses-as-baseline", note: "Discusses ByteTrack two-stage low-confidence association as prior hybrid art." },
{ to: "T067", type: "uses-as-baseline", note: "Discusses StrongSORT embedding plus motion-compensation improvements." },
{ to: "T073", type: "uses-as-baseline", note: "Discusses OC-SORT observation-centric robustness over short occlusions." },
{ to: "T075", type: "uses-as-baseline", note: "Discusses BoT-SORT hybrid motion-compensation recipe." },
],
concepts: ["tracking-by-detection", "detection", "kalman", "state-space", "motion-model", "data-association", "hungarian", "reid", "appearance-features", "occlusion", "track-management", "mota", "idf1"],
impact:
"Provides a reproducible lightweight recipe showing masked appearance embeddings plus gap-aware motion bridging beat a stronger occlusion baseline in both public and defense-like long-occlusion settings.",
}
];
