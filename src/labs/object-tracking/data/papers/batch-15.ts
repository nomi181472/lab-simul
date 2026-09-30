import type { PaperRecord } from "../types";

/* batch-15 — T129..T138 */

export const BATCH_15: PaperRecord[] = [
{
id: "T129",
arxiv: "2606.26455",
title: "Active Adversarial Perturbation-driven Associative Memory Retrieval for RGB-Event Visual Object Tracking",
shortTitle: "APRTrack",
year: 2026,
authors: ["Xiao Wang", "Xufeng Lou", "Zikang Yan", "Lan Chen", "Sibao Chen", "Yaowei Wang", "Yonghong Tian", "Jin Tang"],
fileName: "2606.26455v1.pdf",
task: "multimodal-tracking",
tags: ["rgb-event-tracking", "multimodal-tracking", "associative-memory", "hopfield-network", "adversarial-perturbation", "missing-modality", "transformer", "single-object"],
difficulty: "advanced",
summary:
"APRTrack is a missing-robust RGB-Event single-object tracker that decomposes real-world structured degradation into modality-level failure and spatial-level target absence, simulates both with adversarial perturbation branches trained through gradient reversal, disentangles them with hierarchical routing plus a clean branch, and compensates locally corrupted observations with Footprint-guided Channel-calibrated Hopfield Retrieval (FCHR) over target-constrained historical memory. It reports the best precision rate on all four RGB-Event benchmarks tested (FE108, COESOT, VisEvent, FELT), reaching 97.0 PR on FE108 and 84.0 PR on COESOT.",
problem:
"RGB-Event fusion trackers assume intact inputs, but real scenes produce structured degradation: a whole modality can fail (RGB under low light or overexposure, event stream under low motion or heavy noise) and the target can be locally missing (occlusion, truncation, clutter), and these two degradation types demand different robustness mechanisms that prior fusion, representation and temporal-modeling work does not explicitly model.",
background: ["sot", "bounding-box", "multimodal", "attention", "occlusion", "memory-network"],
previousWork: [
{
name: "Cross-modal RGB-Event fusion trackers (frame-event alignment and fusion, unified Transformer modeling, high-rank interaction, prompt adaptation, distractor suppression)",
limitation:
"Focus on exploiting available multi-modal cues and assume intact inputs, leaving structured missing patterns such as full-modality failure and local target absence unexplored.",
whyThisPaper:
"Adds explicit hierarchical adversarial perturbation that realistically reproduces modality loss and spatial layout loss during training instead of assuming clean dual inputs.",
},
{
name: "Unified RGB-X frameworks (UnTrack, SUTrack, SDSTrack, XTrack) and long-term temporal models (AMTTrack, MamTrack, Mamba-FETrack V2)",
limitation:
"Generalize across modalities or model long-term history, but do not separate whole-modality degradation from local target corruption and provide no reliability-gated historical compensation.",
whyThisPaper:
"Separates the two degradation levels with decoupled routing and adds FCHR, which calibrates the retrieval metric space and gates historical injection instead of blindly fusing history.",
},
{
name: "Missing-modality learning (MMIN, SMIL, prompt-based recovery, shared-specific modeling) and memory trackers with dynamic templates or historical prompts",
limitation:
"Built for recognition tasks or unconstrained temporal aggregation, so retrieval under locally corrupted queries can inject background clutter, outdated appearance or false matches.",
whyThisPaper:
"Introduces target-constrained memory with ROI bias, footprint-based reliability estimation and gated residual fusion so history compensates rather than overwrites current observations.",
},
],
researchGap:
"No RGB-Event tracker explicitly models multi-level modality missing (spatial-level and modality-level) together with controlled, reliability-aware historical retrieval inside one framework.",
contribution: [
"APRTrack framework: dual adversarial perturbation branches (modality-level mutually exclusive gating, spatial-level continuous occlusion sampling) inserted between patch embedding and the Transformer backbone, with perturbations disabled at inference while memory retrieval stays active.",
"Adversarial hierarchical perturbation with gradient reversal and hard Gumbel-Softmax selection, including balance regularization against gate collapse and a target-overlap constraint so spatial masks match real occlusion instead of random token dropout.",
"Footprint-guided Channel-calibrated Hopfield Retrieval (FCHR): target-constrained cross-modal memory, query-memory association footprints (entropy plus max probability) for channel calibration, ROI-biased Hopfield association and gated residual fusion.",
"Hierarchical routing training strategy with clean, modality and spatial branches plus a progressive modality-perturbation schedule, avoiding feature collapse from stacking whole-modality failure and spatial occlusion on the same sample.",
"Best reported precision on four RGB-Event benchmarks (FE108 97.0 PR, COESOT 84.0 PR, VisEvent 79.4 PR, FELT 70.1 PR) with full component, routing and attribute ablations.",
],
method: {
pipeline: ["patch-embed-rgb-event", "adversarial-modality-perturbation", "adversarial-spatial-perturbation", "hierarchical-routing", "hopfield-historical-retrieval", "transformer-backbone-fusion", "ostrack-head-predict"],
architecture:
"Unified RGB-Event Transformer tracker built on the OSTrack prediction paradigm with a HiViT-B encoder (template 128x128, search 256x256, MAE-pretrained initialization); perturbation and FCHR modules sit between shared patch embedding and the backbone and operate mainly on search tokens; a dual-modality fusion module feeds an OSTrack-style classification plus box regression head.",
motionModel: "None explicit — robustness comes from perturbation training and cross-temporal retrieval rather than a motion predictor.",
appearanceModel:
"RGB appearance textures plus stacked event images (events aggregated over fixed intervals as position, timestamp and polarity tuples); cross-modal Hopfield retrieval lets the instantaneously more reliable modality fetch target-correlated historical features for the other branch.",
association: "Biased Hopfield association over concatenated multi-frame memory with ROI bias from soft target masks; cross-modal queries (Event queries RGB memory and vice versa) with footprint-calibrated channels.",
detectionDependency: "None — first-frame box given; single-object tracking throughout.",
trackManagement:
"Target-constrained historical memory banks of RGB tokens, Event tokens and soft target masks; gated residual fusion injects retrieval as reliability-conditioned compensation; memory update and retrieval remain active at inference.",
optimization:
"AdamW with learning rate 1e-4, weight decay 1e-4, batch size 16, 50 epochs with 0.1 decay at epoch 40; trained separately per benchmark split; progressive modality-perturbation intensity schedule stabilizes early joint training.",
loss:
"L = 5.0 L1 + 2.0 L-GIoU + 1.0 L-focal over box regression and response classification, following the OSTrack head.",
},
equations: [
{
id: "aprtrack-modality-perturbation",
label: "Mutually exclusive modality perturbation",
formula: "Pmod(Xr, Xe) = (ar Xr, ae Xe) with (ar, ae) in {(1,1), (0,1), (1,0)}; a = Psi_mod(GRL(rho(Xr, Xe))) via hard Gumbel-Softmax",
variables: [
{ symbol: "Xr, Xe", meaning: "patch-embedded RGB and Event search tokens" },
{ symbol: "(ar, ae)", meaning: "mutually exclusive gate: keep both, drop RGB, or drop Event; never both" },
{ symbol: "GRL", meaning: "gradient reversal layer driving the gate toward harder missing states" },
{ symbol: "rho, Psi_mod", meaning: "dual-modality context aggregation and the mutually exclusive modality gate" },
],
intuition: "Randomly but adversarially unplug one whole sensor stream so the tracker must localize from the surviving modality.",
why: "Simulates real full-modality failure (sensor failure or severe degradation) instead of assuming clean dual inputs.",
where: "Modality branch of the perturbation module, after patch embedding, training only.",
params: "Balance regularization keeps the gate from collapsing to one state; a progressive schedule ramps perturbation intensity up during joint training.",
},
{
id: "aprtrack-spatial-perturbation",
label: "Continuous spatial occlusion perturbation",
formula: "Pspa(X; s) = (1 - Ss) .* X with s = Psi_spa(GRL(eta(X)), G) under a target-overlap penalty",
variables: [
{ symbol: "Ss", meaning: "continuous rectangular occlusion mask induced by candidate window s" },
{ symbol: "eta(X)", meaning: "token-level spatial scoring with gradient reversal" },
{ symbol: "G", meaning: "soft target mask from the ground-truth box constraining window selection" },
],
intuition: "Paste a coherent occlusion rectangle over the target region, the way real occluders cover contiguous structure rather than random pixels.",
why: "Models local target absence (occlusion, truncation, contamination) with realistic spatial continuity instead of scattered token dropout.",
where: "Spatial branch of the perturbation module; windows sampled with hard Gumbel-Softmax and straight-through retention masks.",
params: "Occlusion area ratio [0.15, 0.30] of the search grid is optimal; smaller ranges under-simulate absence, larger ones destroy recoverable cues.",
},
{
id: "aprtrack-hopfield-retrieval",
label: "Footprint-calibrated ROI-biased Hopfield retrieval",
formula: "Af = Softmax((Q Wqf)(K Wkf)T / sqrt(d)); phi = [E(Af); M(Af)]; Gamma = g(phi); Qtilde = Gamma .* Q; Xhat = Softmax(beta Qtilde KT + Broi) V with Broi = gamma(G - 1)",
variables: [
{ symbol: "Af", meaning: "query-memory association footprint recording how the query distributes matches over history" },
{ symbol: "E, M", meaning: "normalized entropy (retrieval uncertainty) and max probability (strongest historical match)" },
{ symbol: "Gamma", meaning: "channel calibration weights reshaping the query metric space before association" },
{ symbol: "Broi", meaning: "ROI bias near zero on target regions and negative on background, from soft target masks" },
{ symbol: "beta", meaning: "inverse temperature sharpening the Hopfield association distribution" },
],
intuition: "Before trusting old memories, check how confused the current query is, retune its channels toward reliable patterns, and look up history while ignoring background regions.",
why: "Corrupted queries under occlusion otherwise retrieve clutter or false matches; calibration plus ROI bias makes compensation controllable and target-bounded.",
where: "FCHR module: cross-modal retrieval (Event query completes RGB and vice versa) followed by gated residual fusion Y = X + Omega(X, Xhat) .* (Xhat - X).",
},
{
id: "aprtrack-loss",
label: "Tracking head objective",
formula: "L = 5.0 L1 + 2.0 L_GIoU + 1.0 L_focal",
variables: [
{ symbol: "L1", meaning: "box coordinate regression loss" },
{ symbol: "L_GIoU", meaning: "generalized IoU geometric overlap loss" },
{ symbol: "L_focal", meaning: "target response classification loss" },
],
intuition: "Penalize wrong box coordinates, wrong box overlap geometry and wrong target-versus-background responses jointly.",
why: "Standard OSTrack-style supervision for the classification plus regression head on fused search tokens.",
where: "Tracking head during training; weights fixed at 5.0, 2.0 and 1.0.",
},
],
datasets: ["others"],
metrics: ["success-auc", "precision", "norm-precision"],
baselines: ["OSTrack", "MixFormer", "TransT", "STARK", "AiATrack", "ViPT", "SDSTrack", "UnTrack", "SUTrack", "XTrack", "CEUTrack", "AMTTrack", "MamTrack", "SEATrack", "DiMP", "ATOM"],
results: [
"FE108 (108 videos, 208672 frames): 65.1 SR and best 97.0 PR, up from AMTTrack 65.6/95.9 with the highest center-localization precision of all compared trackers.",
"COESOT (1354 sequences, 478721 frames): 68.3 SR and best 84.0 PR, ahead of AMTTrack 68.8/82.9 and ViPT 68.3/81.0.",
"VisEvent (820 sequences, 371127 frames): 60.0 SR and best 79.4 PR, ahead of AMTTrack 60.1/78.1, SDSTrack 59.7/76.7 and UnTrack 59.7/76.3, just under SEATrack 60.3 SR.",
"FELT long-term (1044 videos, 1949680 frames): best 55.3 SR, 70.1 PR and 66.6 NPR, improving AMTTrack by 0.5, 2.2 and 0.9 points and beating SUTrack and XTrack on PR and NPR at similar SR.",
"FELT attribute success rates: consistent gains on viewpoint transformation, deformation, partial occlusion, scale and aspect-ratio change; competitive on fast motion, full occlusion, low illumination and out-of-view; small target remains hardest for all trackers.",
"Efficiency: parameters 70.65M to 81.86M, inference FLOPs 56.87G to 78.50G, 31 FPS on FELT with FPS measured on an RTX 2080 Ti GPU.",
],
ablations: [
"Component analysis on COESOT: MPL alone +1.5 SR/+2.1 PR/+2.0 NPR; SPL alone +0.8 PR/+0.5 NPR; vanilla Hopfield on top +0.3/+0.5/+0.8; footprint calibration +1.0/+1.1/+1.0 more; full model +2.0/+2.8/+2.6 over baseline.",
"MPL design: adversarial gate beats fixed random dropout; adding balance regularization gains +0.1/+0.4/+0.2 over the unregularized adversarial gate.",
"Spatial severity: occlusion ratio [0.15, 0.30] beats [0.10, 0.20] and [0.20, 0.40]; training history of 3 search frames (current plus two historical) beats 2, 4 or 5.",
"Footprint descriptor: entropy plus max probability beats either statistic alone or no footprint; hierarchical routing with clean branch and progressive scheduling (68.3/84.0/82.0) beats stacked MPL+SPL on one sample (64.0/80.5/79.4).",
"Gate dynamics: the Event branch shows higher average compensation gates than RGB across epochs, consistent with sparser event observations needing more historical help.",
],
limitations: {
authorStated: [
"No explicit long-term state evolution modeling across frames, which may hurt sequences with drastic appearance changes or continuous target state transitions.",
"When RGB and Event observations are both severely degraded, or historical memory holds unreliable target states, complementary cues for recovery may be insufficient.",
],
evident: [
"All four evaluation datasets map to the undocumented others registry id, so dataset scale and protocol claims rest entirely on the paper text rather than lab dataset cards.",
"Training-time perturbation machinery (adversarial gates, Gumbel sampling, three routing branches) adds optimization complexity and hyperparameters beyond the reported inference cost.",
],
},
assumptions: [
"Event streams are stacked into image-like frames over fixed intervals, so native asynchronous resolution is already quantized before tracking.",
"Soft target masks from bounding boxes are adequate proxies for true target regions in memory biasing and spatial constraints.",
"Historical memory contains mostly reliable target states worth retrieving under local corruption.",
],
computation:
"Trained on an RTX 4090 server; FPS measured on an RTX 2080 Ti; full model 81.86M training parameters and 78.50G inference FLOPs with HiViT-B backbone.",
relations: [
{ to: "T071", type: "builds-on", note: "Adopts the OSTrack Transformer tracking pipeline, head paradigm and optimization settings as its base." },
{ to: "T100", type: "extends", note: "Extends AMTTrack-style associative memory for frame-event tracking with calibrated, gated, target-bounded retrieval." },
{ to: "T070", type: "uses-as-baseline", note: "Evaluates against MixFormer on the FELT long-term benchmark." },
{ to: "T083", type: "uses-as-baseline", note: "Evaluates against ViPT prompt-based multimodal tracking on COESOT, VisEvent and FELT." },
{ to: "T097", type: "uses-as-baseline", note: "Evaluates against UnTrack single-model any-modality tracking on COESOT and VisEvent." },
{ to: "T105", type: "uses-as-baseline", note: "Evaluates against XTrack multimodal training-boosted tracking on FELT." },
],
concepts: ["multimodal", "memory-network", "attention", "occlusion"],
impact:
"First RGB-Event framework to treat modality-level and spatial-level missing as separate trainable robustness objectives with reliability-gated memory compensation, setting new precision marks on four RGB-Event benchmarks.",
},
{
id: "T130",
arxiv: "2606.29783",
title: "FalconTrack: Photorealistic Auto-Labeled Perception and Physics-Aware Vision-Based Aerial Tracking",
shortTitle: "FalconTrack",
year: 2026,
authors: ["Yan Miao", "Karteek Gandiboyina", "Noah Giles", "Hideki Okamoto", "Bardh Hoxha", "Georgios Fainekos", "Sayan Mitra"],
fileName: "2606.29783v1.pdf",
task: "3d-tracking",
tags: ["aerial-tracking", "6dof-pose", "gaussian-splatting", "auto-labeling", "sim-to-real", "physics-aware-tracking", "ekf", "visual-servoing", "single-target"],
difficulty: "intermediate",
summary:
"FalconTrack is a unified perception-and-tracking stack for vision-based aerial pursuit of a single moving target with an onboard RGB camera: a Gaussian-Splatting auto-labeling pipeline turns a 2-minute object video into about 10k labeled RGB, mask, class and 6-DoF pose images in under 20 minutes with zero manual annotation, a compact 15M-parameter multi-head network trained with staged curriculum plus reprojection consistency transfers zero-shot to the real world at 96 to 100 percent class accuracy, and a class-conditioned EKF plus pose-based visual servoing controller closes the loop at about 25 Hz with 100 percent success on real F1-tenth and gate tracking.",
problem:
"Reliable aerial tracking needs large-scale labeled perception data, but manual annotation is costly and slow, non-photorealistic simulators leave a sim-to-real gap, diffusion generators give no 6-DoF labels, and existing Gaussian-Splatting labeling assumes analytic geometry for a single gate object, while downstream 2D-box controllers ignore target yaw and dynamics needed for stable pursuit.",
background: ["bounding-box", "state-space", "kalman", "3d-tracking", "mot"],
previousWork: [
{
name: "Manual perception datasets (ImageNet, BDD100K, Ego-Exo4D) and non-photorealistic simulator labels (CARLA, Gazebo, KITTI-CARLA)",
limitation:
"Human annotation is costly, error-prone and unscalable to irregular objects, while game-engine renders lack photorealism and cause sim-to-real gaps.",
whyThisPaper:
"Replaces both with photorealistic Gaussian-Splatting rendering plus exact geometric projection of masks and 6-DoF poses, fully automatic.",
},
{
name: "Photorealistic transfer (FalconGym 2.0 gate detector) and RGB-only or mask-only pose baselines (ViT pose regression, mask-centered IBVS)",
limitation:
"Semi-manual labeling tied to analytic ring geometry and a single object type; RGB-only models overfit backgrounds and mask-only control is yaw-ambiguous and loses fast out-of-view targets.",
whyThisPaper:
"Adds SAGA-based splat isolation for arbitrary irregular objects and a mask-gated RGB pose head fused with class-conditioned dynamics priors.",
},
{
name: "Racing trackers and unified 2D-box following with model-free control",
limitation:
"Overfit to one customized track or target type and ignore target-specific dynamics and full 6-DoF relative pose, failing on rapid turns that push targets out of view.",
whyThisPaper:
"Estimates full 6-DoF pose in the camera frame and smooths it with per-class motion models (Dubins car, rigid-body quadrotor, double integrator) inside an EKF.",
},
],
researchGap:
"No unified framework combined photorealistic auto-labeling of irregular objects, multi-head class plus mask plus 6-DoF perception with sim-to-real transfer, and physics-aware closed-loop aerial tracking in one deployable system.",
contribution: [
"Automated labeling pipeline: 2-minute handheld video to object GSplat at real-world scale via a calibrated marker, SAGA plus FalconGym 2.0 Edit API splat isolation, compositing with randomized backgrounds, and projection to pixel-accurate masks and 6-DoF poses — about 10k samples in 20 minutes (10 reconstruction plus 10 labeling) on a desktop RTX 4090.",
"Unified 15M-parameter multi-head perception (shared ResNet-18 encoder with classification, segmentation and gated-attention 6-DoF pose heads) trained with a three-stage curriculum and FalconGym 2.0 SSIM reprojection consistency, running about 30 Hz on a Jetson Orin.",
"Physics-aware tracking and control: per-class dynamics lookup, EKF fusion of raw poses with ego IMU for smoothing and dropout propagation, and pose-based visual servoing with a PD translation plus P yaw law.",
"Zero-shot sim-to-real validation on three geometrically diverse objects and two environments each, plus real closed-loop quadrotor tracking at about 25 Hz with 100 percent success on F1-tenth and gate trajectories.",
],
method: {
pipeline: ["capture-object-video", "gsplat-reconstruct", "saga-isolate-target", "composite-backgrounds", "render-autolabels", "staged-perception-training", "ekf-dynamics-fusion", "pbvs-tracking-control"],
architecture:
"Shared ResNet-18 encoder branching into classification and segmentation heads, with the predicted mask conditioning a gated-attention 6-DoF pose head that fuses RGB appearance and foreground-mask features while suppressing background clutter; empty-scene samples included for out-of-view robustness.",
motionModel:
"Class-conditioned process models selected by predicted class: Dubins car for F1-tenth, 12-state rigid-body dynamics for quadrotors, double integrator for gates; EKF maintains target position, velocity and yaw in the ego body frame and propagates through brief measurement dropouts.",
appearanceModel:
"RGB plus foreground-mask gated attention: mask features gate RGB features so orientation cues survive background suppression, with an SSIM reprojection loss re-rendering the target at the predicted pose to correct small yaw errors.",
association: "Not applicable — single target per run with no target switching or data association stage.",
detectionDependency: "Detector-free in the MOT sense: assumes one known target instance per run; perception classifies among trained instances (F1-tenth, quadrotor, gate) or empty scene.",
trackManagement:
"Desired per-class tracking offsets (2 m forward standoff, 1 m vertical for vehicles, centered 2 m for gates); geo-fence safety abort beyond 8 body lengths; reliable operating domain 2 to 6 body lengths from ODD analysis.",
optimization:
"Staged curriculum over 250 epochs with the full 10k dataset: stage 1 class head only for 20 epochs, stage 2 class plus mask for 30 epochs, stage 3 all heads jointly for 200 epochs with reprojection loss enabled only when a target is present; about 3 hours on a 4090 GPU.",
loss:
"Per-batch L = lambda1 L-class + lambda2 L-mask + lambda3 L-pose + lambda4 SSIM(Render(p-hat), I), where the reprojection term compares the FalconGym 2.0 re-render at the predicted pose against the input RGB.",
},
equations: [
{
id: "falcontrack-pbvs-error",
label: "Body-frame tracking errors",
formula: "ep = p-hat - pd; e-dot-p = v-hat; e-psi = normalize(psi-hat - psi-d)",
variables: [
{ symbol: "p-hat, v-hat, psi-hat", meaning: "EKF-estimated target position, velocity and yaw in the ego body frame" },
{ symbol: "pd, psi-d", meaning: "desired class-specific relative offset position and yaw from the lookup table" },
{ symbol: "ep, e-psi", meaning: "position and yaw errors driving the servoing controller" },
],
intuition: "Measure how far the target sits from where it should hover relative to the pursuer, in the pursuer own coordinates.",
why: "Converts the 6-DoF perception output into actionable servoing errors for closed-loop pursuit.",
where: "Pose-based visual servoing controller fed by EKF estimates; gains Kp = [0.5, 0.2, 0.05], Kd = [0.1, 0.05, 0.01], k-psi = 0.5.",
},
{
id: "falcontrack-pbvs-law",
label: "PBVS velocity and yaw-rate law",
formula: "v = -Kp .* ep - Kd .* e-dot-p; psi-dot = -k-psi e-psi",
variables: [
{ symbol: "v", meaning: "commanded body-frame velocity in R3" },
{ symbol: "psi-dot", meaning: "commanded body-frame yaw rate" },
{ symbol: "Kp, Kd, k-psi", meaning: "empirically set proportional, derivative and yaw gains" },
],
intuition: "Fly toward the desired standoff with damping, and turn to face the target heading — a PD controller on relative pose.",
why: "Translates smoothed pose estimates into stable quadrotor body commands that recover from brief out-of-view events.",
where: "Downstream of the EKF in the onboard closed-loop stack running at about 25 Hz with motors spinning, logging and inference.",
},
{
id: "falcontrack-reprojection-loss",
label: "Reprojection consistency loss",
formula: "L = L + lambda4 SSIM(Render(p-hat), I), applied only when a target is present",
variables: [
{ symbol: "Render(p-hat)", meaning: "FalconGym 2.0 re-render of the target at the predicted 6-DoF pose" },
{ symbol: "I", meaning: "input RGB image" },
{ symbol: "SSIM", meaning: "structural similarity between re-render and input" },
],
intuition: "Ask the simulator to redraw what the network thinks it saw and penalize visual mismatch — geometry must agree with pixels.",
why: "Enforces geometric consistency and fixes visually significant small pose errors, especially yaw, beyond direct regression supervision.",
where: "Stage 3 of Algorithm 1; ablations show it yields the largest pose-accuracy gains in MTE and MAE.",
params: "Skipped for no-object samples so empty scenes do not corrupt pose gradients.",
},
],
datasets: ["others"],
metrics: ["3d-error", "fps"],
baselines: ["PnP pose estimation", "NPE neural pose estimator", "mask-centered IBVS", "state-based oracle with motion-capture pose"],
results: [
"Perception (simulation, 200 unseen samples per object per background): F1-tenth 100 percent class, 0.83 IoU, 24 percent MTE, 0.27 rad MAE; quadrotor 100 percent, 0.79, 32 percent, 0.31 rad; gate 100 percent, 0.72, 41 percent, 0.41 rad — all beating PnP and per-object NPE.",
"Perception zero-shot real (about 200 motion-capture images each): F1-tenth 100 percent class, 43 percent MTE, 0.38 rad MAE; quadrotor 96 percent, 41 percent, 0.37 rad; gate 100 percent, 52 percent, 0.52 rad; staged training lifts class and IoU while reprojection loss drives the pose gains.",
"ODD analysis over 200 images per distance bin: reliable perception within 2 to 6 body lengths on all four metrics with degradation beyond; geo-fence set to 8 body lengths.",
"Closed-loop real F1-tenth (5 trajectories, 2 environments): physics-aware variant 100 percent success, 99 percent target-in-FOV, 0.71 m ATE versus mask-centered IBVS 60 percent, 68 percent, 1.23 m; without physics priors 100 percent, 95 percent, 0.92 m.",
"Closed-loop real gate: physics-aware 100 percent, 100 percent, 0.43 m versus IBVS 100 percent, 100 percent, 0.47 m; simulation trends match with lower ATE (e.g. F1-tenth 0.50 m), the real gap attributed to lighting, motion blur and aerodynamics.",
"Throughput: perception about 30 Hz offline on Jetson Orin, full closed-loop stack about 25 Hz in flight; labeling about 10k samples in 20 minutes.",
],
ablations: [
"Perception variants: joint training without staging versus staged without reprojection versus full; staging slightly improves class and IoU, reprojection gives the largest MTE and MAE gains across all three objects.",
"Tracking variants: state-based oracle is best as expected with 100 percent in-FOV confirming feasible trajectories; 6-DoF yaw estimates let the no-physics variant recover where IBVS loses fast-turning F1-tenth targets; dynamics-aware EKF further smooths and improves accuracy.",
"IBVS failure concentrated in the S-shape trajectory where rapid yaw changes push the target out of view and 2D-box centering cannot recover.",
],
limitations: {
authorStated: [
"Single target per run with no target switching or multi-object tracking.",
"Perception assumes known object instances and needs a short reconstruction video per object, limiting open-world deployment to unknown objects.",
"Hardware validation excludes fast targets above about 2 m/s with stronger motion blur and excludes real quadrotor-tracking-quadrotor runs pending stronger arena safety measures.",
],
evident: [
"No quantitative real-world mask IoU is reported because pixel-level ground-truth masks are unavailable without manual annotation.",
"Sim-to-real ATE consistently rises on hardware, so the claimed transfer rests on success and in-FOV rates plus centimeter-scale error growth rather than matched accuracy.",
],
},
assumptions: [
"Camera frame treated as coincident with the ego body frame throughout estimation and control.",
"Operating domain of 2 to 6 body lengths holds during pursuit, enforced by the 8-body-length geo-fence abort.",
"One target instance from the three trained classes is present or the scene is empty; unknown objects are out of scope.",
],
computation:
"Auto-labeling about 20 minutes on a desktop RTX 4090; perception training about 3 hours on a 4090; onboard Jetson Orin inference with a 15M-parameter model.",
relations: [],
concepts: ["3d-tracking", "kalman", "benchmark-design"],
impact:
"Shows that photorealistic Gaussian-Splatting auto-labeling plus class-conditioned physics priors is sufficient for zero-shot sim-to-real closed-loop aerial pursuit of irregular objects with no manual annotation.",
},
{
id: "T131",
arxiv: "2607.01395",
title: "Rethinking Generic Object Tracking Toward Human-Level Perceptual Intelligence",
shortTitle: "Rethinking GOT",
year: 2026,
authors: ["Shih-Fang Chen"],
fileName: "2607.01395v1.pdf",
task: "single-object",
tags: ["generic-object-tracking", "single-object", "visual-prompting", "test-time-adaptation", "occlusion-perception", "geometry-aware-tracking", "model-editing", "transformer", "survey-synthesis"],
difficulty: "advanced",
summary:
"Doctoral dissertation organizing three first-authored studies into one capability-unlocking progression for generic object tracking: PiVOT adds automatic CLIP-refined visual prompting for distractor suppression, GOT-JEPA extends joint-embedding predictive learning to tracking-model prediction with an OccuSolver point-visibility module for fine-grained occlusion handling, and GOT-Edit fuses VGGT geometric cues with DINOv2 semantics through online null-space model editing. Each stage reports state-of-the-art or best-in-class numbers, rising from PiVOT-L 73.4 LaSOT success to GOT-JEPA 75.4 and GOT-Edit-378 75.0 to 75.5 with VOT2022 robustness up to 0.898.",
problem:
"Generic object tracking gives only a first-frame box of a possibly unseen target in a streaming video, yet prevailing trackers rely on appearance matching optimized for training targets, adapt weakly online, reason about occlusion only coarsely at box level, and ignore 3D geometry — while naive geometry-semantics fusion degrades the semantic discrimination tracking depends on.",
background: ["sot", "bounding-box", "siamese", "attention", "occlusion"],
previousWork: [
{
name: "Tracking-by-detection trackers (DiMP, ToMP) with meta-learned model predictors and confidence-gated updates",
limitation:
"Predictors are optimized to identify familiar training targets so generalization collapses on unseen objects, and appearance-derived confidence cannot express which target parts stay visible under partial occlusion.",
whyThisPaper:
"PiVOT adds prompt-guided discrimination, GOT-JEPA retrains the predictor itself for corrupted-observation robustness, and OccuSolver supplies pixel-level visibility labels.",
},
{
name: "Matching-based Siamese and transformer trackers (SiamRPN++, OSTrack, MixFormer, SeqTrack, ARTrack) with offline optimization",
limitation:
"No explicit online model adaptation at inference, so novel targets, corrupted observations and shifted search distributions degrade performance.",
whyThisPaper:
"Reframes adaptation as online tracking-model prediction (GOT-JEPA) and online cross-modal model editing (GOT-Edit) inside the tracking-by-detection paradigm.",
},
{
name: "Prompting trackers (ViPT, OneTracker, OVTrack, CiteTracker) and geometry or point-tracking lines (VGGT, CoTracker, SAM-PT)",
limitation:
"Language or depth prompts need extra descriptors or 3D inputs unavailable in the wild, distillation concentrates on familiar classes, and point tracking lacks object semantics while naive 2D-3D fusion harms discrimination.",
whyThisPaper:
"Uses image-only CLIP prompts refined at test time, adapts CoTracker with GOT object priors, and constrains geometry fusion to the semantic null space.",
},
],
researchGap:
"No single program had progressed GOT from external contrastive discrimination to endogenous adaptive prediction with fine-grained occlusion perception and finally to geometry-aware yet semantic-preserving online adaptation.",
contribution: [
"Progressive capability-unlocking paradigm linking three studies: externally assisted discrimination, endogenous generalization with occlusion perception, geometry-aware semantic-preserving adaptation.",
"PiVOT: Prompt Generation Network plus Relation Modeling plus test-time CLIP refinement, with a frozen DINOv2 ViT-L variant needing under 1 percent trainable adapter parameters; PiVOT-L-27 reaches 73.4 LaSOT, 62.2 AVisT and 76.9 GOT-10k AO.",
"GOT-JEPA: teacher-student tracking-model prediction with corruption asymmetry, invariance plus covariance objectives and a ProjNet tail; reaches 63.7 AVisT, 70.8 NfS, 79.6 GOT-10k AO and 75.4 LaSOT.",
"OccuSolver: GOT-prior-conditioned CoTracker adaptation with ladder-side tuning, visibility head, Gaussian mapping and ensemble modulation, lifting AVisT occlusion attributes by several points.",
"GOT-Edit: dual semantic and geometric model predictors with SVD null-space projection of geometry perturbations (whitened, ridge-regularized, symmetrized); GOT-Edit-378 reaches 64.5 to 64.7 AVisT, 71.1 NfS, 75.0 OTB and VOT-STb2022 robustness 0.898.",
],
method: {
pipeline: ["extract-backbone-features", "generate-visual-prompt", "refine-prompt-test-time", "relation-modeling", "predict-tracking-model", "jepa-robust-pretraining", "solve-point-visibility", "edit-model-with-geometry", "localize-target"],
architecture:
"ToMP tracking-by-detection base throughout: model predictor generating filter weights plus classification and regression decoders; PiVOT prepends a ConvNet PGN and Conv-BN-GeLU relation network with a frozen ViT-L/14 CLIP refiner and optional frozen DINOv2 ViT-L backbone; GOT-JEPA adds frozen teacher and student predictors with ProjNet and Expander tails plus an OccuSolver branch (CoTracker image encoder, iterative transformer, light-Trans ladder adapters, VisHead, ensemble transformer); GOT-Edit adds parallel DINOv2 semantic and VGGT geometric branches with align-and-fuse gating and dual predictors.",
motionModel: "None explicit — adaptation is carried by online predicted filter weights, dynamic reference frames and point-track propagation rather than a kinematic prior.",
appearanceModel:
"Stage-specific: CLIP image-embedding contrast for prompt refinement; DINOv2 dense semantic features frozen during training; VGGT DPT-head geometric features aligned and gated into semantic features; copy-paste feature-space corruption (rho up to 0.2) for JEPA student training.",
association: "Not applicable in the MOT sense — single-target GOT with candidate score maps and point-track visibility, no cross-identity association.",
detectionDependency: "None — first-frame box given; reference templates are the fixed initial frame plus a confidence-gated dynamic frame.",
trackManagement:
"Reference labels updated online from predictions; OccuSolver FIFO window with first-frame visibility gating (skip initialization below 85 percent visible) and duplication of the last unoccluded frame; confidence-gated dynamic references at inference.",
optimization:
"PiVOT: two-stage (60 epochs tracker, 40 epochs prompting) with AdamW; GOT-JEPA: JEPA pretraining then head fine-tuning with alpha to beta 25 to 1 and ProjNet learning rate 10x; GOT-Edit: 200K subsequences per epoch for 25 epochs, AdamW 1e-4 with StepLR decay, DeepSpeed and mixed precision.",
loss:
"DiMP compound hinge classification plus GIoU regression throughout (PiVOT adds a candidate-map classification term with weights 100, 10, 1); JEPA adds invariance plus covariance model-prediction losses; OccuSolver adds dual point-level and GOT-level classification plus regression terms.",
},
equations: [
{
id: "got-prompted-feature",
label: "Prompt-conditioned current-frame feature",
formula: "hcan = phi([vtem1, vtem2, vcur]); vcur-p = g-phi([hcan, vcur])",
variables: [
{ symbol: "hcan", meaning: "initial visual prompt score map highlighting candidate target centers" },
{ symbol: "vtem1, vtem2", meaning: "exact-box template features, the second updated by CLIP template similarity" },
{ symbol: "vcur", meaning: "current-frame backbone feature map" },
{ symbol: "g-phi", meaning: "relation-network classifier turning prompt plus features into prompted features" },
],
intuition: "Draw a heat map of where the target might be, staple it to the image features, and let a small network dim everything the map does not highlight.",
why: "Makes the tracker promptable so foundation-model knowledge can steer feature responses toward the target and away from distractors.",
where: "Between backbone and Tracking Head in PiVOT; at inference hcan is replaced by the CLIP-refined prompt.",
},
{
id: "got-clip-importance",
label: "CLIP contrastive candidate importance",
formula: "Di = (1/2) sum-j exp(cos(Ecani, Etemj)) / sum-k exp(cos(Ecank, Etemj)); refine location to 1 if Di > gamma",
variables: [
{ symbol: "Ecani, Etemj", meaning: "CLIP image-encoder features of candidate RoIs and the two reference templates" },
{ symbol: "Di", meaning: "normalized pairwise cosine-similarity importance of candidate i" },
{ symbol: "gamma", meaning: "importance threshold, 0.25 in experiments" },
{ symbol: "tau", meaning: "candidate confidence threshold 0.05 with 3x3 local-maximum extraction" },
],
intuition: "Crop every suspicious region, ask CLIP which one looks most like the remembered target, and keep only the winners on the heat map.",
why: "Transfers CLIP zero-shot arbitrary-object discrimination into the tracker at test time without any prompt annotation or extra training cost.",
where: "Test-time Prompt Refinement module; ablations show refinement is what lifts out-of-distribution sets like NfS, OTB-100 and AVisT.",
simulator: "reid",
},
{
id: "got-jepa-objective",
label: "Tracking-model prediction objective",
formula: "Lmp = alpha Linv(w, w-hat) + beta Lcov(w-exp); Linv = mean ||wi - w-hat-i||2^2; Lcov penalizes off-diagonal covariance of Expander outputs",
variables: [
{ symbol: "w-hat", meaning: "pseudo tracking model from the frozen teacher on the clean current frame" },
{ symbol: "w", meaning: "student prediction from the corrupted current frame via ProjNet under identical history" },
{ symbol: "alpha, beta", meaning: "loss weights with best ratio 25 to 1" },
{ symbol: "w-exp", meaning: "Expander (1x1 conv) output whose channel redundancy is penalized" },
],
intuition: "Show the student a vandalized frame and demand the same tracking filter the teacher built from the clean frame — learn to see through corruption.",
why: "Trains adaptation itself as a transferable skill instead of overfitting filters to familiar training targets, giving robustness to occlusion and distractors.",
where: "Stage-1 JEPA pretraining of the model predictor only; heads are fine-tuned jointly afterward.",
},
{
id: "got-nullspace-edit",
label: "Semantic-preserving geometry edit",
formula: "p = (Wsem + Pnull Delta) * zcur with Pnull = (P-hat + P-hatT)/2 from low-energy eigenvectors of M = Z ZT + lambda I",
variables: [
{ symbol: "Wsem", meaning: "semantic model weights from the DINOv2-only predictor (knowledge to preserve)" },
{ symbol: "Delta", meaning: "geometry perturbation weights from the fused-feature predictor" },
{ symbol: "Pnull", meaning: "symmetrized null-space projector of whitened semantic features via SVD" },
{ symbol: "zcur", meaning: "fused semantic-geometric current-frame features" },
],
intuition: "Add 3D knowledge only in directions the 2D semantics do not use, so new spatial awareness cannot overwrite what distinguishes this target.",
why: "Naive fusion improves occlusion and clutter attributes but degrades distractor, fast-motion and illumination handling; the constraint keeps both.",
where: "Online model editing in GOT-Edit before the localization head; whitening plus ridge regularization stabilize the per-frame SVD.",
},
],
datasets: ["otb", "vot", "got10k", "lasot", "trackingnet", "uav123", "others"],
metrics: ["success-auc", "precision", "norm-precision", "eao"],
baselines: ["ToMP", "MixFormer", "OSTrack", "SeqTrack", "ARTrack", "ROMTrack", "GRM", "TransT", "STARK", "KeepTrack", "TrDiMP", "LoRAT", "UVLTrack", "SwinTrack", "CSWinTT", "ODTrack", "CiteTracker", "SwinTrack"],
results: [
"PiVOT-L-27: LaSOT 73.4 SUC, 84.7 NPr, 82.1 Pr; AVisT 62.2 SUC, 73.3 OP50, 55.5 OP75; NfS 69.0; OTB-100 71.3; UAV123 70.9; GOT-10k 76.9 AO; TrackingNet 85.3 SUC; VOT2022 EAO 0.560 with best robustness 0.873.",
"GOT-JEPA: AVisT 63.7, NfS 70.8, OTB-100 73.2 (all best reported); GOT-10k AO 79.6 best; LaSOT 75.4 SUC with best 85.3 NPr; TrackingNet best 86.4 SUC and 90.6 NPr; VOT-STb2022 best robustness 0.898 and AUC 0.728; OP50 best on NfS 89.60, AVisT 73.67, LaSOT 86.46.",
"GOT-Edit-378: AVisT 64.5 to 64.7, NfS 71.1, OTB-100 75.0, LaSOT 75.0 to 75.5, TrackingNet 86.4 to 86.7 (best or tied-best across variants); VOT-STb2022 robustness 89.8 and VOT-ST2020 90.3; gains about 2 to 3 percent over the ToMP-378 baseline on every dataset.",
"Attribute analyses: PiVOT leads LaSOT deformation and fast motion plus AVisT imaging, camouflage and obstruction effects; GOT-JEPA leads OTB, AVisT and LaSOT occlusion, deformation and visibility attributes; GOT-Edit leads background clutter, occlusion and rotation while trailing PiVOT only in low-light imaging effects.",
],
ablations: [
"PiVOT Tab 3.6: initial prompt alone without CLIP refinement drops out-of-distribution sets (NfS, OTB-100, AVisT) below baseline while CLIP-refined prompts beat ToMP on all five sets for both backbones; CLIP ViT-L/14 at 336px and gamma 0.25 are optimal.",
"GOT-JEPA Tab 4.5: invariance pretraining plus covariance plus OccuSolver lifts the ToMP-L baseline by 2.5 points AVisT, 2.7 LaSOT and 1.4 OTB; OccuSolver alone helps modestly but gains substantially more on JEPA-pretrained priors; copy-paste corruption beats masking.",
"GOT-Edit Tab 5.4: geometry-only features collapse below baseline (55.8 AVisT), naive fusion is mixed (helps occlusion and clutter, hurts distractors and fast motion), null-space constraint plus whitening and regularization recovers all rows for mean gains of 1.8 AVisT, 1.7 NfS and 2.5 LaSOT.",
"Cost: PiVOT inference 240 ms per frame dominated by frozen DINOv2 plus CLIP with only 29M trainable of about 300M frozen parameters; GOT-JEPA 41 ms with 128 query points; GOT-Edit editing modules only 9 to 17 ms with VGGT extraction (1000 to 2253 GFLOPs) the bottleneck.",
],
limitations: {
authorStated: [
"PiVOT still depends on external contrastive priors at inference, struggles with small low-resolution targets lacking semantics, and does not explicitly adapt under partial occlusion — the stated motive for GOT-JEPA.",
"GOT-JEPA adds system complexity through its dependence on an auxiliary point-tracking module, and geometric reasoning remains limited without integrated spatial structure.",
"GOT-Edit still needs work on moving objects, fast motion and large viewpoint changes where geometry weakens, and on out-of-distribution data such as AVisT target effects.",
],
evident: [
"As a dissertation compilation, the three methods are separate systems sharing a ToMP base rather than one unified tracker, so cross-stage interactions are narrative rather than jointly trained.",
"All stages lean on massive frozen foundations (CLIP, DINOv2 ViT-L, VGGT) with heavy inference footprints, leaving real-time deployment unaddressed beyond low-resolution variants.",
],
},
assumptions: [
"First-frame box is correct and reference-label propagation stays trustworthy enough for online predictor updates.",
"CLIP image-embedding similarity is a valid proxy for instance-level target identity across arbitrary categories.",
"VGGT geometry extracted from 2D RGB streams is sufficiently consistent to complement semantics under occlusion and clutter.",
],
computation:
"PyTorch with PyTracking; RTX 3090 GPUs for PiVOT, RTX 4090 GPUs (up to 8 with DeepSpeed and mixed precision) for GOT-JEPA and GOT-Edit; evaluation on single GPUs with 3 to 9 GB memory.",
relations: [
{ to: "T071", type: "uses-as-baseline", note: "Compares all three trackers against OSTrack one-stream joint feature-relation modeling." },
{ to: "T070", type: "uses-as-baseline", note: "Compares against MixFormer iterative mixed attention on LaSOT, AVisT and OTB attributes." },
{ to: "T083", type: "conceptual-successor", note: "Extends ViPT-style prompting to test-time CLIP visual prompts and VGGT geometry prompts without extra modalities." },
{ to: "T024", type: "conceptual-successor", note: "Uses LaSOT as the main long-term benchmark and training source across all three studies." },
{ to: "T025", type: "conceptual-successor", note: "Adopts the GOT-10k class-disjoint protocol as the unseen-category generalization test." },
],
concepts: ["sot", "attention", "occlusion", "memory-network"],
impact:
"Establishes a stepwise recipe from prompt-guided discrimination to predictive adaptation with visibility reasoning to semantic-preserving geometry fusion, each stage setting new marks on adverse-visibility GOT benchmarks.",
},
{
id: "T132",
arxiv: "2607.14726",
title: "AE-UAV: An Air-to-Air Event-Based UAV Tracking Benchmark and a Real-Time Frequency-Domain Tracker",
shortTitle: "AE-UAV / FSFT",
year: 2026,
authors: ["Zixin Jiang", "Bing He", "Chaoran Xiong", "Zhenzhen Wang", "Xin Zhao", "Ling Pei"],
fileName: "2607.14726v1.pdf",
task: "benchmark",
tags: ["benchmark", "event-camera", "uav-tracking", "air-to-air", "frequency-domain", "training-free", "correlation-filter", "real-time", "cpu-tracking"],
difficulty: "intermediate",
summary:
"First event-camera benchmark captured from an aerial platform for air-to-air UAV tracking: 178 sequences, about 2140 seconds and over 8.15 billion events with cubic B-spline continuous-time annotations yielding C2-continuous trajectories and event-level labels at microsecond resolution. Its companion Fast-Slow Frequency-domain Tracker (FSFT) fuses per-packet Fourier magnitude matching with Kalman prediction and normal-flow pruning (fast path) and periodic detection-based drift correction (slow path), reaching 56.43 percent SR-AUC at 420 FPS on CPU only — 93.97 percent of the best GPU tracker accuracy with 5.32x effective speedup and the top accuracy-throughput ATQ score of 100.22.",
problem:
"Air-to-air tracking breaks frame cameras (motion blur, limited dynamic range, fixed sampling) and event cameras promise a fix, but existing UAV event datasets are all ground-to-air with discrete constant-box labels, while deep event trackers need GPUs, discard microsecond timing through frame accumulation, and collapse when the accumulation rate changes — all incompatible with small onboard UAV platforms.",
background: ["bounding-box", "motion-model", "kalman", "correlation-filter", "benchmark-design"],
previousWork: [
{
name: "Event UAV datasets (VisEvent, EventVOT, F-UAV-D, NeRDD, FRED, EV-UAV) with ground-to-air capture and cumulative-frame or linear-extension labels",
limitation:
"Cannot model dense ego-motion events filling the whole field of view in air-to-air flight, and piecewise-constant labels introduce systematic errors during maneuvers with no fair cross-resolution comparison.",
whyThisPaper:
"Captures from a Matrice 300 RTK observer platform and replaces discrete labels with cubic B-spline continuous-time annotations supporting evaluation at arbitrary temporal resolution.",
},
{
name: "Deep event trackers (EventVOT transformers, MambaEVT, OSTrack, ARTrack, ROMTrack, AQATrack, ODTrack adapted via frame reconstruction)",
limitation:
"GPU-dependent, training-heavy, and tied to one accumulation rate — e.g. ROMTrack trained at 30 Hz collapses from 55.82 to 0.52 percent SR-AUC at 200 Hz — while rendering overhead cuts actual throughput far below inference FPS.",
whyThisPaper:
"Proposes a training-free CPU-only packet-native alternative whose accuracy varies only 6.04 points across 30 to 200 Hz.",
},
{
name: "Training-free event methods (time surfaces, optical flow, asynchronous corners, EKLT) and frequency-domain correlation (KCF, DCF, ECO, eFFT)",
limitation:
"No complete lightweight event-native tracking framework combining rapid localization with long-term drift control for air-to-air UAV pursuit.",
whyThisPaper:
"Fuses Fourier magnitude matching with Kalman-gated search, normal-flow direction pruning and periodic density or contour detection correction in one fast-slow architecture.",
},
],
researchGap:
"No air-to-air event-based UAV tracking benchmark with continuous-time labels existed, and no training-free real-time CPU tracker spanned the accuracy-throughput gap for onboard deployment.",
contribution: [
"AE-UAV benchmark: 178 flight sequences (125 train, 18 val, 35 test via hierarchical stratified sampling) with pursuit, evasion and head-on geometries, daylight, backlit and night illumination, Prophesee EVK4 HD 1280x720 events plus RGB, thermal and 200 Hz IMU, and dual frame-level plus event-level outputs.",
"Continuous-time annotation: sparse keyframe boxes interpolated with regularized cubic B-splines (lambda 0.001) into C2 position, velocity and acceleration profiles, with per-event inside-or-rectangle binary labels.",
"FSFT fast pathway: constant-velocity Kalman prediction with covariance-adaptive radius, spatiotemporal-plane normal-flow direction estimation, forward-sector candidate pruning, FFT magnitude-spectrum normalized cross-correlation and confidence-gated exponential template averaging.",
"FSFT slow pathway: linear-extrapolation ROI with automatic density mode (night lights-on) versus contour mode (day or lights-off) weighted feature fusion, adaptive thresholding and morphology, resetting Kalman state and scale on valid detections.",
"Experiments: 93.97 percent of the best deep accuracy at 5.32x effective speedup, best ATQ 100.22, best scale-variation and head-on scores, full ablations and cross-resolution generalization where every deep baseline degrades severely off its training rate.",
],
method: {
pipeline: ["capture-a2a-events", "keyframe-annotate", "bspline-interpolate", "event-level-label", "kalman-predict", "normal-flow-estimate", "frequency-match", "periodic-detect-correct"],
architecture:
"Two-path training-free framework on raw event packets with no dense frames and no learned weights: fast path localizes every packet via prediction, direction estimation, event filtering (repeated-event removal plus reactivated-event exclusion) and Fourier magnitude correlation; slow path corrects drift every N packets via event-intrinsic detection; both share one Kalman state with bidirectional interaction.",
motionModel:
"Constant-velocity Kalman filter over position and velocity with process noise covariance; search radius grows with position-covariance trace; normal-flow cone halves the candidate area toward the estimated motion direction.",
appearanceModel:
"No learned appearance: Fourier magnitude spectra encode edge structure at characteristic frequencies while attenuating spatially incoherent noise; time-surface gradients, event density, direction consistency, compactness, edge density and contour coherence form the detection features.",
association: "Not applicable — single-target frequency matching plus detection validation inside a predicted ROI, no multi-target association stage.",
detectionDependency: "Tracker-internal: slow-path detector with aspect-ratio and compactness validation gates corrections; no external detector required.",
trackManagement:
"Confidence-gated template update only above matching threshold; validated detections reset Kalman state and adapt scale; template frozen through occlusions and target absence.",
optimization: "None — training-free with hand-set thresholds, learning rate eta, cone half-angle and correction period N.",
loss: "None — similarity-thresholded matching (rho-star above tau-match) and percentile-thresholded score maps replace learned losses.",
},
equations: [
{
id: "aeuav-bspline-trajectory",
label: "Cubic B-spline continuous trajectory",
formula: "a(t) = sum-j cj Bj,3(t); {cj}* = argmin sum-i ||a(ti) - ai||2 + lambda integral ||a-double-dot(t)||2 dt with lambda = 0.001",
variables: [
{ symbol: "a in {cx, cy, w, h}", meaning: "each bounding-box parameter as a smooth time function" },
{ symbol: "Bj,3", meaning: "cubic B-spline basis guaranteeing C2 continuity" },
{ symbol: "ai at ti", meaning: "sparse human keyframe box observations" },
{ symbol: "lambda", meaning: "curvature penalty weight preventing overfitting, fixed at 0.001" },
],
intuition: "Thread one smooth flexible curve through a few hand-drawn boxes so position, speed and acceleration all evolve plausibly between keyframes.",
why: "Piecewise-constant labels assume zero intra-frame motion, which fails during maneuvers; smooth trajectories enable fair evaluation at any temporal resolution from one annotation effort.",
where: "Annotation pipeline stage two; stage three queries B(t) per event timestamp for binary inside-rectangle labels.",
},
{
id: "aeuav-kalman-predict",
label: "Kalman prediction with adaptive radius",
formula: "x-hat-t|t-1 = F x-t-1; P-t|t-1 = F P-t-1 FT + Q; rt = r0 + alpha tr(P-xy)",
variables: [
{ symbol: "x = [x, y, x-dot, y-dot]", meaning: "constant-velocity target state" },
{ symbol: "F, Q", meaning: "state transition matrix and process noise covariance" },
{ symbol: "P-xy", meaning: "position covariance submatrix whose trace drives the search radius" },
{ symbol: "rt", meaning: "adaptive search radius expanding under uncertainty, contracting when stable" },
],
intuition: "Guess where the drone glides next from its velocity, and look wider when the guess is uncertain, narrower when tracking is solid.",
why: "Constrains frequency matching to a small probable region, cutting cost and background interference under six-degree-of-freedom relative dynamics.",
where: "Fast-path candidate selection before direction-guided sector generation; ablation shows removal costs 11.30 SR-AUC points.",
simulator: "kalman",
},
{
id: "aeuav-magnitude-correlation",
label: "Fourier magnitude spectrum correlation",
formula: "rho-i = <|F(T)| - mu-T, |F(Ci)| - mu-i> / (sigma-T sigma-i); track rho-star = max-i rho-i if above tau-match",
variables: [
{ symbol: "|F(T)|, |F(Ci)|", meaning: "Fourier magnitude spectra of template and candidate event representations" },
{ symbol: "mu, sigma", meaning: "spectrum means and standard deviations for normalization" },
{ symbol: "rho-star, tau-match", meaning: "best similarity accepted only above the match threshold" },
],
intuition: "Compare the frequency fingerprints of target and candidates: edges live at characteristic frequencies while sensor noise and small shifts wash out.",
why: "Gives shift-invariant, O(n log n) CPU-only matching that concentrates on structure and attenuates high-frequency noise without any learned features.",
where: "Fast-path template matching after repeated and reactivated event filtering; template updated only on high-confidence matches.",
simulator: "corr-filter",
},
{
id: "aeuav-atq",
label: "A2A Tracking Quality trade-off metric",
formula: "ATQ = SR-AUC x log-30(Actual FPS), with 30 FPS as the neutral reference rate",
variables: [
{ symbol: "SR-AUC", meaning: "area under the success plot across IoU thresholds" },
{ symbol: "Actual FPS", meaning: "end-to-end throughput from raw events to output including rendering overhead" },
{ symbol: "log-30", meaning: "logarithmic scaling giving diminishing marginal credit for higher frame rates" },
],
intuition: "One number rewarding both accuracy and deployable speed, where being too slow for the drone counts against an accurate method.",
why: "Test FPS hides event-to-frame rendering cost (e.g. 126 vs 80 for ROMTrack); ATQ penalizes accurate-but-undeployable trackers per VOT-style practice.",
where: "Supplementary metric alongside SR-AUC, PR and NPR; standard metrics remain primary and FSFT leads ATQ by 23.08 points.",
simulator: "metrics",
},
],
datasets: ["others"],
metrics: ["success-auc", "precision", "norm-precision", "fps"],
baselines: ["OSTrack", "SimTrack", "ARTrack", "ROMTrack", "AQATrack", "ODTrack", "MambaEVT", "FocusTrack"],
results: [
"Overall (best accumulation rate per method): FSFT 56.43 SR-AUC, 88.45 PR, 79.49 NPR at 420 FPS CPU-only with no training, versus best deep FocusTrack 60.05, 88.37, 81.63 at 79 actual FPS — 93.97 percent retained accuracy at 5.32x effective speedup and best ATQ 100.22 ahead of FocusTrack 77.14.",
"Scenario SR-AUC: best on scale variation 55.63 and head-on approach 48.03; direction change 56.05; small target 54.41; pursuit 56.30; evasion 58.52; trails on complex background 51.11 versus 54.95 where clutter shares the target spectral band.",
"Ablations: naive FFT matching alone 21.34; removing detection correction drops to 31.62 (minus 24.81, the most critical component); removing Kalman gives 45.13; removing direction estimation gives 53.62; correction costs 1.1 ms, Kalman 0.1 ms, direction 0.06 ms per activation.",
"Temporal generalization: FSFT spans 50.39 to 56.43 (6.04 points) across 200, 120, 60 and 30 Hz with the best average 53.62; ROMTrack, AQATrack and FocusTrack trained at 30 Hz collapse to 0.52, 0.98 and 3.40 at 200 Hz.",
"Dataset: 178 sequences over about 2140 seconds with 8.15 billion events of which 31.2M (0.38 percent) are target events; 57.3 percent small and 40.1 percent medium targets; night sequences emphasized for counter-UAV realism.",
],
ablations: [
"Component study isolates drift correction as dominant, Kalman search constraining as essential under dynamic motion, and normal-flow pruning as a measurable precision plus efficiency gain.",
"Detection modes selected automatically from initial template traits: density mode (density, direction consistency, compactness, gradient) for lights-on night versus contour mode (edge density, coherence, gradient, density) for day or lights-off.",
"Resolution study shows higher training accumulation rates generalize better for deep baselines (FocusTrack at 60 Hz averages 53.28) yet none match the training-free stability of FSFT.",
],
limitations: {
authorStated: [
"Textured backgrounds share the target high-frequency spectral band, so Fourier magnitude matching cannot cleanly separate clutter that learned discriminative features suppress.",
"Constant-velocity Kalman model cannot capture rapid curvature changes of evasive UAV maneuvers; higher-order or learned short-horizon predictors would fit better.",
"Single-target assumption in the predictive ROI means the correction path may lock onto the wrong UAV when several targets coincide in view.",
],
evident: [
"ATQ is an author-defined metric with an arbitrary 30 FPS reference, so the headline 23-point lead depends on accepting throughput-accuracy log scaling.",
"Deep baselines were trained by the authors on the 125-sequence train split and evaluated at author-chosen accumulation rates, so baseline tuning effort is not independently verifiable.",
],
},
assumptions: [
"Target events form the minority signal (0.38 percent) separable by edge-frequency structure from dense ego-motion clutter.",
"Constant-velocity motion plus periodic correction is sufficient between slow-path activations.",
"One UAV target occupies the predictive region of interest during correction.",
],
computation:
"Workstation with Intel i5-12400F CPU and RTX 3090 GPU for baselines; FSFT runs 420 FPS CPU-only with no training stage and lower power draw.",
relations: [
{ to: "T001", type: "conceptual-successor", note: "Continues the KCF circulant Fourier correlation lineage as packet-native magnitude matching." },
{ to: "T011", type: "conceptual-successor", note: "Follows the ECO efficient-convolution-operator drive toward CPU-real-time frequency tracking." },
{ to: "T071", type: "uses-as-baseline", note: "Evaluates against OSTrack adapted to event frames among eight deep baselines." },
{ to: "T005", type: "conceptual-successor", note: "Inherits the SORT constant-velocity Kalman prediction structure for search constraining." },
],
concepts: ["benchmark-design", "correlation-filter", "kalman", "motion-model"],
impact:
"Supplies the missing air-to-air event benchmark with resolution-free labels and proves training-free frequency-domain tracking is deployable on small UAVs, redirecting future work toward hybrid efficient-plus-learned architectures.",
},
{
id: "T133",
arxiv: "2607.17120",
title: "The generator is the tracker: Multi-object tracking by painting persistent identity colours",
shortTitle: "Generator Tracker",
year: 2026,
authors: ["Haiyu Yang", "Miel Hostens"],
fileName: "2607.17120v1.pdf",
task: "multi-object",
tags: ["multi-object-tracking", "generative-tracking", "video-diffusion", "identity-as-colour", "detector-free", "re-identification", "dancetrack", "lora-finetuning"],
difficulty: "advanced",
summary:
"Replaces the entire MOT stack with video generation: a 22B text-to-video diffusion model (LTX-2.3) with a rank-32 in-context LoRA is fine-tuned to translate RGB clips into ID-map videos where each person is painted a flat persistent palette colour, chained across 49-frame windows via cleaned-prefix continuation training so identity lives purely in pixels. On the official DanceTrack test server it scores 40.3 HOTA with AssA 44.1 above every tracker of the original benchmark suite — the only entry with no detector, no motion model and no association step — while detection (DetA 37.6) is the declared sole deficit.",
problem:
"MOT assumes identity must be external state (buffers, motion models, appearance embeddings) stitched onto per-frame detections, yet this stack collapses exactly where association matters: DanceTrack dancers in uniform costume with chaotic motion defeat appearance and geometric linking, and no one had tested whether a video generator internal binding knowledge could carry identity directly in generated pixels.",
background: ["mot", "bounding-box", "data-association", "reid", "tracking-by-detection"],
previousWork: [
{
name: "Tracking-by-detection (SORT, DeepSORT, FairMOT, ByteTrack, QDTrack, OC-SORT) and end-to-end query trackers (TransTrack, MOTR, MOTRv2, MeMOTR)",
limitation:
"Keep identity in slots, buffers or embeddings outside the video; on DanceTrack strong detectors reach DetA 70 to 80 while association collapses to about 30.",
whyThisPaper:
"Removes the stack entirely: the palette index of generated pixels is the track id, so no detection, motion, re-id or Hungarian step exists anywhere.",
},
{
name: "Perception-as-generation (painter-style image models, image generators as generalist vision learners) and controllable video diffusion (LTX-Video, flow matching, LoRA)",
limitation:
"No prior generative formulation required temporally persistent instance colouring that turns segmentation into tracking, and none reported on MOT leaderboards.",
whyThisPaper:
"Instantiates the thesis on video with a frozen ID-map codec plus continuation fine-tuning, the first generative tracker evaluated on an official MOT server.",
},
{
name: "Image-model identity probes (zero-shot editing, VAE and DINO appearance banks, motion-primary matching)",
limitation:
"All score zero on re-binding identities after occlusion gaps on DanceTrack because appearance embeddings are uninformative by benchmark design.",
whyThisPaper:
"Shows the video generator re-acquires identities after gaps at 42 percent conditionally, including gaps beyond its temporal context, acting as its own re-identification function.",
},
],
researchGap:
"Whether temporally persistent identity can be emitted as generation itself — with no tracker, motion model or association — had never been formalized, trained (continuation) and measured on an official MOT benchmark.",
contribution: [
"Identity-as-colour video translation: fixed K=48 CIELAB-separated palette (minimum delta-E 26.4), lossless-only codec with round-trip mask IoU at least 0.99, SAM-derived silhouette supervision on 777 clips from 40 DanceTrack training videos.",
"Continuation training for unbounded length: 49-frame windows at stride 40 chained on decode-correct-reencode 9-frame prefixes, with teacher-forced prefix fine-tuning at probability 0.5 lifting full-video IDF1 2.7x from 0.172 to 0.466.",
"First generative tracker on the official DanceTrack server: single frozen submission at 40.3 HOTA, 44.1 AssA, 45.2 IDF1 — association above the whole original suite, detection the entire deficit.",
"Controlled A/B proof the mechanism matters: identical windows scored 34.8 HOTA via generator colours versus 18.2 via classical post-hoc association; per-frame IoU fragments 14 colour-consistent units into 44 tracklets.",
"Emergent re-identification measurement on 383 mined gap events: 42 percent conditional re-acquisition (38 percent at 25 to 48 frames, 27 to 31 percent beyond the 49-frame context) where all baselines score zero.",
],
method: {
pipeline: ["build-cielab-palette", "sam-silhouette-supervision", "lora-video-translation-finetune", "continuation-prefix-finetune", "chained-window-generation", "palette-quantize-decode", "merge-overlap-tracks"],
architecture:
"LTX-2.3 latent video diffusion transformer (22B total, 19B 48-layer video branch, causal VAE with 32x spatial and 8x temporal compression) with native in-context LoRA: VAE-encoded RGB reference concatenated to target tokens with aligned positions, kept clean at timestep 0 and excluded from loss; only video-branch attention QKV, output and FFN projections adapted (rank 32, 0.4 percent of parameters); fixed precomputed text prompt dropping the 24B text encoder from memory.",
motionModel: "None — no Kalman, no velocity prior, no motion matching; temporal coherence comes from video-attention pretraining.",
appearanceModel: "None classical — appearance enters only as the RGB reference conditioning the translation plus the learned appearance-to-colour mapping that doubles as re-identification.",
association: "None — palette index is the global track id; same-colour boxes in a frame merged by union, overlap ownership split at window midpoints.",
detectionDependency: "No detector: boxes are connected-component bounding boxes of quantised colours with a 150 px minimum-area filter and speckle removal.",
trackManagement: "No birth, death or buffer logic; palette caps simultaneous identities at K minus 1 (47); dense scenes saturate at 10 to 12 painted instances per frame at 512 resolution.",
optimization:
"Phase A translation 1500 steps at lr 2e-4 bf16 with flow matching and shifted-logit-normal timestep sampling, batch 1; Phase B continuation resumed to step 2500 with clean 2-latent-frame prefix at p 0.5; best checkpoint by in-loop validation foreground IoU on all 25 validation videos.",
loss:
"Flow-matching video-translation objective on target tokens only (reference and prefix tokens excluded); text fixed so no language loss participates.",
},
equations: [
{
id: "gentrack-idmap-codec",
label: "ID-map encode and decode",
formula: "encode: pixel of instance i -> c-a(i) on black with one colour per identity per video; decode: id = argmin-k distance-CIELAB(pixel, ck), then per-colour connected components with area >= 150 px",
variables: [
{ symbol: "c-a(i)", meaning: "palette colour assigned to identity i, stable for the whole video" },
{ symbol: "K = 48", meaning: "fixed palette size with minimum pairwise CIELAB separation delta-E 26.4" },
{ symbol: "distance-CIELAB", meaning: "nearest-palette quantisation metric in perceptual colour space" },
],
intuition: "Give every person their own crayon for the whole video, then read identities back by asking which crayon each pixel is closest to.",
why: "Freezes tracking into a lossless pixel code so generation output is directly readable as tracks with no association step.",
where: "Frozen infrastructure around the generator: SAM-box supervision at train time, quantisation plus union-merge at inference; PNG or FFV1 only since lossy compression corrupts flat colours.",
},
{
id: "gentrack-continuation-chain",
label: "Chained continuation generation",
formula: "window-k = Generate(RGB-k, prefix = reencode(decode-correct(tail-9 of window k-1))); training mixes clean target prefix with p = 0.5; inference uses 20 steps, CFG 4.0, STG 1.0, seed 0",
variables: [
{ symbol: "window-k", meaning: "49-frame generated ID-map at stride 40" },
{ symbol: "prefix", meaning: "first 9 frames anchored as clean tokens from the previous cleaned tail" },
{ symbol: "CFG, STG", meaning: "classifier-free guidance 4.0 and spatio-temporal guidance 1.0" },
],
intuition: "Paint the video in overlapping strips, each starting by tracing the last strip cleaned edge so colours continue instead of restarting.",
why: "49 trained frames cannot cover 700 to 2400 frame videos and longer clips degrade identity; naive inference chaining without prefix training reverts to invented colourings and compounds errors.",
where: "Full-video inference chain; prefix training preserves single-window quality (foreground IoU 0.589 vs 0.605) while lifting chain IDF1 2.7x.",
},
],
datasets: ["dancetrack"],
metrics: ["hota", "deta", "assa", "idf1", "mota"],
baselines: ["CenterTrack", "FairMOT", "TransTrack", "ByteTrack", "QDTrack", "OC-SORT", "MOTR", "Deep OC-SORT", "DiffMOT", "MeMOTR", "MOTRv2"],
results: [
"Official DanceTrack test server, single frozen submission: 40.3 HOTA, 37.6 DetA, 44.1 AssA, 45.2 IDF1, 16.2 MOTA — in the 2020 to 2021 detector band (FairMOT 39.7, CenterTrack 41.8) but the only method above the AssA equals DetA diagonal.",
"Association above the entire original suite including end-to-end MOTR (40.2) and OC-SORT (38.3) on the benchmark built to break association; modern specialists lead only via detector-boosted pipelines (62 to 70 HOTA) with AssA-to-DetA ratios still below one.",
"Pre-registered A/B on 25 full validation videos: generation-carried identity 34.8 HOTA, 38.3 AssA, 38.3 IDF1 versus post-hoc Hungarian overlap association of identical windows 18.2, 17.7, 17.3 with MOTA minus 13.3.",
"Re-acquisition on 383 mined gap events: 42.0 percent conditional overall (47.2 at 9 to 24 frames, 38.0 at 25 to 48, 26.9 at 49 to 150) against zero for all appearance-bank and editing baselines; independent windows plus association reach only 36.1.",
"Diagnostics: prefix training lifts development-video full IDF1 0.172 to 0.466; 121-frame direct generation drops IDF1 0.549 to 0.312 past frame 49 with 2.3x colour-switch rate; IoU linking fragments 14 colour units into 44 tracklets per window.",
"Failure axis: dense scenes with 14 to 25 people score IDF1 0.18 to 0.33 versus 0.40 to 0.86 for 4 to 9 people; throughput about 0.8 frames per second on one A100 with no speed claims.",
],
ablations: [
"Continuation training is necessary: inference-only chaining scores below independent windows until the p 0.5 prefix mix teaches colour propagation.",
"Longer windows are not the answer: identity persistence does not extrapolate beyond the trained 49-frame horizon even as mask quality holds.",
"Resolution probe: regenerating a 19-person scene at 768 halves merge rates and reveals more instances, but the 512-trained LoRA alignment warps so boxes degrade — resolution-matched training left as future work.",
"Oracle decomposition on the development video: per-window IDF1 0.661 with a perfect stitcher ceiling of 0.698, showing window identity is strong and chaining is the binding constraint.",
],
limitations: {
authorStated: [
"Detection in dense scenes (DetA 37.6) and throughput (about 0.8 fps, 22B model with 20 denoising steps per window) are the two honest deficits.",
"Palette bounds simultaneous identities at K minus 1 equals 47; evaluation is single-domain DanceTrack with MOT17-scale transfer untested; continuation teacher-forcing on ground-truth prefixes leaves a train-to-test gap with mild erosion deep in long chains.",
],
evident: [
"Absolute scores sit far below modern specialists (up to 69.9 published HOTA), so the contribution is the inverted error profile and paradigm evidence, not leaderboard position.",
"Compute totals about 85 A100-hours for one domain, and all re-identification evidence comes from choreographed-dance footage with near-identical costumes.",
],
},
assumptions: [
"Lossless ID-map storage throughout; any lossy step silently corrupts decoding.",
"Fixed 48-colour palette suffices for the scene population; crowds beyond token capacity are accepted as failure.",
"Test distribution matches the DanceTrack association-stress regime; appearance-diverse domains are out of scope.",
],
computation:
"Entire fine-tune fits one 80GB A100 via precomputed text embeddings; about 85 A100-hours total plus one CPU-parallel evaluation pass; inference 20 steps per 49-frame window.",
relations: [
{ to: "T063", type: "uses-as-baseline", note: "Evaluates on the DanceTrack benchmark and beats its full original association suite on AssA." },
{ to: "T061", type: "uses-as-baseline", note: "Compares against ByteTrack two-stage association (32.1 AssA) which the generator colours surpass." },
{ to: "T073", type: "uses-as-baseline", note: "Compares against observation-centric motion association OC-SORT (38.3 AssA)." },
{ to: "T043", type: "uses-as-baseline", note: "Compares against FairMOT joint detection plus embedding association (39.7 HOTA)." },
{ to: "T059", type: "uses-as-baseline", note: "Compares against end-to-end query association MOTR (40.2 AssA)." },
{ to: "T005", type: "conceptual-successor", note: "Rejects the SORT tracking-by-detection ontology of external motion plus association state." },
],
concepts: ["mot", "data-association", "reid", "diffusion-tracking"],
impact:
"First task-level evidence that a lightly fine-tuned video generator carries persistent identity in pixels with association quality above dedicated trackers, opening generation-as-interface as a measurable path for video perception.",
},
{
id: "T134",
arxiv: "2607.23209",
title: "Counterfactual Motion Reliability Learning for Robust UAV Tracking",
shortTitle: "CMRTrack",
year: 2026,
authors: ["Yuehai Chen", "Jian Lan", "Yuan Wei"],
fileName: "2607.23209v1.pdf",
task: "single-object",
tags: ["infrared-tracking", "uav-tracking", "anti-uav", "one-stream", "transformer", "motion-reliability", "counterfactual-learning", "score-fusion"],
difficulty: "advanced",
summary:
"CMRTrack adds counterfactual motion reliability learning to a one-stream OSTrack-style ViT-B tracker for infrared UAV tracking. A lightweight motion evidence encoder estimates a factual motion map from adjacent search regions, while a training-only target-erased history branch supplies a counterfactual reference that teaches the encoder to keep target-consistent motion and suppress background pseudo motion. Motion-guided token modulation plus reliability-aware score fusion then enhance search tokens and refine the response, lifting OSTrack from 53.7 to 67.3 AUC on Anti-UAV410 at 62 fps with only 0.1G extra MACs.",
problem:
"Infrared UAV targets are tiny, low-contrast and surrounded by thermal distractors, so appearance-only transformer responses drift to background structures. Naive motion cues do not fix this because camera jitter, dynamic backgrounds, sensor noise and target disappearance produce pseudo motion stronger than true target motion.",
background: ["sot", "bounding-box", "iou", "siamese", "attention", "motion-model", "success-plot"],
previousWork: [
{
name: "One-stream transformer trackers (OSTrack baseline)",
limitation:
"Appearance matching alone assigns high confidence to distractor regions when the tiny infrared target response is weak, causing localization drift.",
whyThisPaper:
"Adds factual motion evidence with token modulation and reliability-aware score fusion on top of the same one-stream backbone.",
},
{
name: "Temporal and aerial trackers (STARK spatio-temporal transformer, TCTrack temporal contexts, autoregressive trackers)",
limitation:
"Extract, enhance or aggregate motion cues but never verify whether observed motion is truly target-induced, so background dynamics leak into predictions.",
whyThisPaper:
"Introduces a counterfactual target-erased history and relative ranking constraints that explicitly separate target-consistent motion from pseudo motion.",
},
{
name: "Anti-UAV methods (GASiam graph attention, GlobalTrack and Siam R-CNN global re-detection, FocusTrack adaptive search)",
limitation:
"Improve features or enlarge the search region for recovery at higher cost, without modeling whether the motion response is reliable.",
whyThisPaper:
"Targets motion reliability itself with a training-only counterfactual branch that is discarded at inference, keeping real-time speed.",
},
],
researchGap:
"No infrared UAV tracker explicitly learned motion reliability by comparing factual motion against a target-erased counterfactual reference before fusing motion into features and scores.",
contribution: [
"CMRTrack framework: lightweight motion evidence encoder plus counterfactual target-erased history branch used only during training to regularize motion learning.",
"Counterfactual reliability losses: factual foreground versus counterfactual foreground ranking loss, factual foreground versus strongest background loss, and soft reliability target supervising the fusion reliability score.",
"Motion-guided token modulation and reliability-aware score fusion with peak, center-distance statistics and an MLP reliability gate controlling motion correction strength.",
"State-of-the-art infrared results on Anti-UAV and Anti-UAV410 with ablations, attribute analysis, erasing-strategy study, parameter study and real-time efficiency comparison.",
],
method: {
pipeline: ["encode-template-search", "estimate-factual-motion", "counterfactual-regularize-train-only", "modulate-search-tokens", "predict-box-and-appearance-score", "reliability-fused-score", "carry-history-forward"],
architecture:
"ViT-B one-stream backbone with center-based head (DropMAE init) plus parallel lightweight motion encoder of two 3x3 conv-BN-ReLU blocks and a 1x1 sigmoid head; token modulation and a 3x3 plus 1x1 residual fusion module with MLP reliability predictor.",
motionModel:
"Channel-averaged absolute frame difference between current and historical search regions encoded into a motion response map; no Kalman filter, causality preserved by storing the previous search region.",
appearanceModel:
"One-stream joint template-search self-attention producing target-aware search tokens, residually enhanced in motion-relevant locations before the box head.",
association: "Not applicable (single-object; argmax over fused score map with Hanning penalty inherited from OSTrack-style inference).",
detectionDependency: "None — first-frame template given; historical search region maintained causally frame to frame.",
trackManagement:
"No online weight update; current search region becomes history for the next frame; counterfactual branch disabled at inference so no extra forward pass.",
loss:
"OSTrack box losses Ltrack with lambda_giou 2 and lambda_l1 5, plus LMEE factual focal plus counterfactual ranking plus background ranking plus reliability BCE, plus fusion focal loss on the final score map; positive-target frames only.",
},
equations: [
{
id: "cmr-diff",
label: "Channel-averaged temporal difference",
formula: "Dt = (1/C) sum_c |Xt^c - Xt-1^c|; Mt = Phi(Xt, Xt-1)",
variables: [
{ symbol: "Xt, Xt-1", meaning: "current and historical search regions" },
{ symbol: "C", meaning: "number of image channels" },
{ symbol: "Dt", meaning: "pixel-level temporal variation map" },
{ symbol: "Mt", meaning: "factual motion response map in [0,1]" },
],
intuition: "Subtract consecutive crops and average over channels to highlight what changed, then let a tiny CNN turn raw change into a target-aware motion map.",
why: "Provides explicit temporal evidence complementary to weak infrared appearance before any fusion.",
where: "Motion evidence encoder input; Mt resized to score-map resolution for modulation and fusion.",
simulator: "motion",
paperIds: ["T134"],
},
{
id: "cmr-cfrank",
label: "Counterfactual foreground ranking loss",
formula: "Lcf = max(0, m - sf + s_tilde_f); Lcf-bg = max(0, m - sf + max(sb, s_tilde_b))",
variables: [
{ symbol: "sf, s_tilde_f", meaning: "average factual and counterfactual foreground responses under target heatmap Gt" },
{ symbol: "sb, s_tilde_b", meaning: "average factual and counterfactual background responses under 1-Gt" },
{ symbol: "m", meaning: "ranking margin, default 0.12" },
],
intuition: "Demand that motion with the true history respond more strongly on the target than motion with the target erased, and more than any background response.",
why: "Forces the encoder to preserve target-induced motion while suppressing camera and clutter pseudo motion without forcing counterfactual maps to zero.",
where: "Training-only motion reliability learning on valid positive target samples.",
paperIds: ["T134"],
},
{
id: "cmr-modulate",
label: "Motion-guided token modulation",
formula: "E_tilde_X = EX * (1 + alpha * M_hat_t)",
variables: [
{ symbol: "EX", meaning: "search tokens from one-stream backbone" },
{ symbol: "M_hat_t", meaning: "motion map resized and flattened to token layout" },
{ symbol: "alpha", meaning: "modulation strength, default 0.5" },
],
intuition: "Keep original appearance features but turn up the volume on tokens where motion evidence says something target-like moved.",
why: "Injects temporal evidence into representations before localization so the head sees target-aware features.",
where: "Applied to search tokens only; template tokens unchanged; then fed to localization head.",
params: "Alpha 0.3 to 1.0 tested; 0.5 best; too large over-amplifies uncertain regions.",
paperIds: ["T134"],
},
{
id: "cmr-fusion",
label: "Reliability-aware score fusion",
formula: "S_tm = sigmoid(logit(St) + Delta_t); S_final = St + beta * rt * (S_tm - St); rt = MLP([ps, pm, dsm])",
variables: [
{ symbol: "St", meaning: "appearance score map from box head" },
{ symbol: "Delta_t", meaning: "residual correction from fused appearance-motion conv module" },
{ symbol: "rt", meaning: "predicted reliability score in [0,1]" },
{ symbol: "ps, pm", meaning: "peak responses of appearance and motion maps" },
{ symbol: "dsm", meaning: "normalized soft-argmax center distance between St and Mt" },
{ symbol: "beta", meaning: "fusion coefficient, default 0.5" },
],
intuition: "Propose a motion-corrected score but let a small reliability network decide how much of the correction to trust based on peak agreement.",
why: "Prevents unreliable motion from corrupting good appearance responses under jitter and infrared noise.",
where: "Final response refinement; fused map supervised by focal loss against target heatmap Gt.",
simulator: "metrics",
paperIds: ["T134"],
},
],
datasets: ["others"],
metrics: ["success-auc", "precision", "norm-precision"],
baselines: ["OSTrack", "ROMTrack", "ZoomTrack", "DropTrack", "FocusTrack", "TransT", "ETTrack", "MixFormerV2-S", "MixFormerV2-B", "GRM", "ARTrack", "JointNLT", "SeqTrack", "PromptVT", "Stark-ST101", "TCTrack", "ToMP50", "ToMP101", "SwinTrack-Tiny", "SwinTrack-Base", "AiATrack", "SiamDT"],
results: [
"Anti-UAV test: 72.3 AUC, 92.5 precision, 91.7 normalized precision, 73.6 state accuracy; beats FocusTrack by 4.6 AUC, 3.3 normalized precision and 4.7 state accuracy points.",
"Anti-UAV410 test: 67.3 AUC, 89.9 precision, 86.2 normalized precision, 68.5 state accuracy; OSTrack baseline 53.7, 73.9, 70.9, 54.7 for gains of 13.6, 16.0, 15.3, 13.8 points; beats FocusTrack by 4.5, 3.7, 3.4, 4.6 points.",
"Attributes on Anti-UAV410: 57.7 FM, 59.2 SV, 65.5 TC, 48.8 DBC AUC, improving OSTrack by 14.6, 21.4, 15.3, 10.5 points; scale subsets 78.9 normal, 73.0 medium, 63.2 small, 51.4 tiny AUC, beating FocusTrack by 8.0, 6.2, 5.4, 4.1 points.",
"Efficiency on RTX 3090: 62 fps at 29.2G MACs versus OSTrack 137 fps at 29.1G, FocusTrack 44 fps at 30.1G, SiamDT 8 fps at 225.3G; best AUC and state accuracy among compared local and global trackers.",
],
ablations: [
"Removing counterfactual learning drops AUC 67.3 to 63.0; removing token modulation to 62.8; removing score fusion to 62.7; removing reliability supervision to 60.3, the largest drop.",
"Motion baselines: raw temporal difference 56.4, supervised motion map 60.8, motion fusion without reliability 62.7, full CMRTrack 67.3 AUC.",
"Erasing strategy: global mean fill 67.3 beats local mean 66.7, Gaussian blur 66.2, zero fill 65.8 and random noise 65.4 AUC.",
"Defaults gamma 1.0, alpha 0.5, beta 0.5, margin 0.12 are each best in sweeps; performance stable in a reasonable neighborhood.",
],
limitations: {
authorStated: [
"FocusTrack remains better in several disappearance-related scenarios such as occlusion and out-of-view; CMRTrack still beats the OSTrack baseline on all attributes.",
"Counterfactual branch and motion handling add inference overhead versus OSTrack (137 to 62 fps) though still real-time.",
],
evident: [
"All motion losses computed only on valid positive target samples, so target-absent frames contribute no reliability supervision and long disappearance relies on appearance recovery.",
"Global mean fill erasing is a synthetic counterfactual that may not match real background evolution under strong camera motion.",
],
},
assumptions: [
"Adjacent search regions are spatially comparable so channel-averaged difference is meaningful temporal evidence.",
"Target heatmap supervision and scaled erasing box approximately cover the true historical target.",
],
computation:
"Training on RTX 5090 GPUs; ViT-B DropMAE init; 30 epochs on Anti-UAV410 train with AdamW, base lr 4e-4 with 0.1 backbone multiplier, batch 64, 128 template and 256 search; speed measured on RTX 3090.",
relations: [
{ to: "T071", type: "builds-on", note: "Adopts OSTrack one-stream framework as baseline and improves it by 13.6 AUC points." },
{ to: "T009", type: "conceptual-successor", note: "Continues SiamFC similarity-matching lineage via Siamese and one-stream appearance matching." },
{ to: "T064", type: "uses-as-baseline", note: "Evaluates against SwinTrack-Tiny and Base on Anti-UAV410." },
{ to: "T068", type: "uses-as-baseline", note: "Evaluates against TCTrack temporal-context aerial tracker on Anti-UAV410." },
],
concepts: ["sot", "attention", "motion-model", "memory-network", "success-plot", "occlusion"],
impact:
"Shows reliability-gated motion fusion rather than raw temporal difference is the key gain for tiny infrared UAV tracking, with a reusable training-only counterfactual recipe.",
},
{
id: "T135",
arxiv: "2607.24701",
title: "Spatio-Temporal Conditional Denoising Transformer for Modality-Missing RGBT Tracking",
shortTitle: "SCDT",
year: 2026,
authors: ["Andong Lu", "Ziyi Zha", "Jiandong Jin", "Shihao Li", "Chenglong Li", "Jin Tang", "Bin Luo"],
fileName: "2607.24701v1.pdf",
task: "multimodal-tracking",
tags: ["rgbt-tracking", "modality-missing", "denoising-transformer", "temporal-conditioning", "unified-framework", "reconstruction", "enhancement"],
difficulty: "advanced",
summary:
"SCDT unifies modality-missing reconstruction and modality-complete enhancement for RGBT tracking as spatio-temporal conditional denoising. Short-term cues from recent history capture fine motion continuity while a long-term modality token encodes global evolution; together they condition a transformer denoiser that refines noisy available-modality features. A noise-modulated adaptation rule uses strong noise for missing cases and weak noise for complete cases with shared weights, reaching new state of the art on LasHeR-Miss, RGBT234-Miss and VTUAV-Miss as well as competitive complete-modality scores.",
problem:
"When RGB or thermal input is missing from misalignment, occlusion or hardware failure, multimodal features become incomplete and unstable. Existing copy, zero-fill, GAN, invertible-prompt and mixture-of-experts fixes rely on current-frame cues and need separate branches for missing versus complete cases, giving spatially biased and temporally inconsistent reconstructions.",
background: ["sot", "bounding-box", "attention", "multimodal", "memory-network", "success-plot"],
previousWork: [
{
name: "Complete-modality RGBT trackers (TBSI cross-modal interaction, AINet Mamba fusion, STTrack temporal modeling)",
limitation:
"Performance drops sharply when one modality is unavailable because fusion assumes both inputs present and models mostly intra-modal dependencies.",
whyThisPaper:
"Adds conditional denoising that reconstructs the absent modality from spatial plus short-term and long-term temporal conditions.",
},
{
name: "Modality-missing RGBT methods (IPL invertible prompt learning, FlexTrack mixture of experts)",
limitation:
"Generate missing features from current-frame available cues with scenario-dependent switching or separate branches, neglecting historical temporal correlations.",
whyThisPaper:
"Replaces switching with one shared denoiser handling both missing and complete inputs via noise-level adaptation.",
},
{
name: "Basic missing-modality fallbacks (copy, zero-fill, GAN generation)",
limitation:
"Crude imputation produces unsatisfactory features in challenging scenes with no temporal coherence.",
whyThisPaper:
"Reformulates fusion as progressive spatio-temporal conditional refinement supervised by reconstruction plus statistical alignment losses.",
},
],
researchGap:
"No single RGBT model jointly exploited short-term and long-term temporal contexts inside a conditional denoising process that serves both missing-modality reconstruction and complete-modality enhancement without changing architecture or parameters.",
contribution: [
"SCDT unified framework: shared ViT-B encoder plus spatio-temporal conditioned denoiser that outputs reconstructed missing features or enhanced complete features for the tracking head.",
"Dual temporal conditioning: short-term cross-attention over recent non-missing frames plus long-term FiLM scale-shift modulation from a global modality token.",
"Noise-modulated adaptation: strong noise for reconstruction under missing input, weak noise for enhancement under complete input, with identical denoiser weights.",
"State-of-the-art results on three complete and three missing RGBT benchmarks with ablations on conditions, noise, supervision and depth.",
],
method: {
pipeline: ["encode-rgb-thermal-templates-search", "perturb-available-features-with-adaptive-noise", "short-term-cross-attention-conditioning", "long-term-film-modulation", "denoise-to-reconstruct-or-enhance", "concat-fuse-and-predict-box"],
architecture:
"Shared ViT-B encoder initialized from ODTrack plus 4-layer SCDT denoiser blocks each with self-attention, short-term cross-attention, FiLM modulation and FFN; fully convolutional prediction head; template 128 and search 256.",
motionModel: "Not applicable — temporal coherence comes from short-term history tokens and long-term modality token conditioning, not an explicit motion filter.",
appearanceModel:
"Available-modality spatial cues plus complementary-modality history guide denoising toward semantically coherent and temporally stable multimodal features.",
association: "Not applicable (single-object RGBT; argmax over predicted response).",
detectionDependency: "None — first-frame box given for both modalities.",
trackManagement:
"No online update described; multi-template and search frames encoded spatio-temporally with shared weights across missing and complete cases.",
loss:
"Ltotal with lambda3 1.0 for ODTrack-style tracking loss; missing cases lambda1 1.0 reconstruction MSE and lambda2 0.0; complete cases lambda1 0.0 and lambda2 1.0 mean-variance alignment; AdamW with backbone lr 1e-5 and rest 1e-4.",
},
equations: [
{
id: "scdt-noise",
label: "Adaptive noisy input for denoiser",
formula: "f_tilde_m = sqrt(alpha_bar) f_m + sqrt(1 - alpha_bar) epsilon, epsilon ~ N(0, sigma^2 I)",
variables: [
{ symbol: "f_m", meaning: "encoded feature of the available modality" },
{ symbol: "sigma^2", meaning: "task-dependent noise variance, higher for reconstruction and lower for enhancement" },
{ symbol: "f_tilde_m", meaning: "noisy input fed to denoiser" },
],
intuition: "Deliberately corrupt good features a little or a lot so the network must learn to recover clean multimodal content from context.",
why: "Encodes the fusion objective in the noise level, letting one model serve both missing and complete scenarios.",
where: "SCDT module input before conditional denoising.",
paperIds: ["T135"],
},
{
id: "scdt-recon",
label: "Missing-modality reconstruction objective",
formula: "f_hat_mprime = D_theta(f_tilde_m; cs, ct); Lrecon = ||f_hat_mprime - f_mprime||_2^2",
variables: [
{ symbol: "cs", meaning: "spatial condition from current available frame" },
{ symbol: "ct", meaning: "temporal condition from short-term history plus long-term token" },
{ symbol: "f_mprime", meaning: "ground-truth feature of the missing modality" },
],
intuition: "Ask the denoiser to hallucinate the absent modality features guided by what is visible now and what that modality looked like recently.",
why: "Direct feature-level supervision for accurate spatial and semantic recovery of missing input.",
where: "Missing-modality training path; reconstructed feature concatenated with available feature for tracking.",
paperIds: ["T135"],
},
{
id: "scdt-align",
label: "Complete-modality statistical alignment",
formula: "Lalign = ||mu(f_hat_m) - mu(f_m)||_2^2 + ||Var(f_hat_m) - Var(f_m)||_2^2",
variables: [
{ symbol: "mu", meaning: "mean computed across spatial tokens" },
{ symbol: "Var", meaning: "variance computed across spatial tokens" },
{ symbol: "f_hat_m", meaning: "enhanced feature under weak noise" },
],
intuition: "Keep the overall modality distribution while encouraging more discriminative directions instead of forcing exact pixel fidelity.",
why: "Enhances weak but present modalities and improves cross-modal alignment without deterministic overfitting.",
where: "Complete-modality training path with weak noise regime.",
paperIds: ["T135"],
},
{
id: "scdt-block",
label: "Spatio-temporal conditioned denoiser block",
formula: "fm_prime = f_tilde_SA + CrossAttn(f_tilde_SA, sc, sc); fm_double = fm_prime * (1 + tanh(Ws lc)) + tanh(Wr lc); f_hat = fm_double + FFN(LN(fm_double))",
variables: [
{ symbol: "sc", meaning: "short-term temporal tokens from recent non-missing complementary frames" },
{ symbol: "lc", meaning: "long-term global modality token" },
{ symbol: "Ws, Wr", meaning: "learnable FiLM projection matrices" },
],
intuition: "First align locally with recent frames, then stabilize globally with a sequence-level scale and shift.",
why: "Balances fine motion continuity against drift suppression for temporally coherent generation.",
where: "Each of the 4 SCDT blocks; moderate depth avoids over-smoothing seen at 6 layers.",
paperIds: ["T135"],
},
],
datasets: ["lasheR", "others"],
metrics: ["success-auc", "precision"],
baselines: ["FlexTrack", "IPL", "STTrack", "SUTrack", "AINet", "CAFormer", "CKD", "OneTracker", "SDSTrack", "UnTrack", "ViPT", "TBSI", "ProTrack", "HMFT", "OSTrack", "APFNet", "ADRNet", "CAT", "mfDiMP"],
results: [
"Complete: LasHeR 77.4 PR and 61.0 SR (best PR); RGBT234 93.1 MPR new state of the art and 69.6 MSR second best; VTUAV 93.6 PR and 78.9 SR highest.",
"Missing: LasHeR-Miss 69.3 PR and 54.4 SR beating FlexTrack by 4.2 and 2.1 points; RGBT234-Miss 88.1 MPR and 64.3 MSR new state of the art; VTUAV-Miss 84.1 PR and 69.6 SR best.",
"Attribute robustness on LasHeR-Miss leads on all conventional attributes with largest gains in low resolution, similar appearance and thermal crossover, and leads every missing pattern and ratio including long-time missing and 90 percent missing.",
],
ablations: [
"Temporal conditions: spatial-only already helps; adding short-term gains LasHeR-Miss by 1.4 PR and 1.4 SR; adding long-term helps complete sets; full dual conditioning best with 77.4 and 61.0 on LasHeR and 69.3 and 54.4 on LasHeR-Miss.",
"Noise: strong-strong 73.2 and 65.4, weak-weak 75.8 and 68.9, weak-complete plus strong-missing 77.4 and 69.3 PR on LasHeR and LasHeR-Miss.",
"Supervision: alignment-only 76.0 and 67.2, reconstruction-only 75.6 and 68.0, combined 77.4 and 69.3 PR on LasHeR and LasHeR-Miss.",
"Depth: 2 layers 75.9 and 68.8, 4 layers 77.4 and 69.3, 6 layers 76.4 and 69.2 PR on LasHeR and LasHeR-Miss.",
],
limitations: {
authorStated: [
"No explicit limitation section; reliance on recent non-missing historical frames means prolonged total absence of a modality leaves weaker temporal evidence.",
],
evident: [
"Missing benchmarks are constructed variants (LasHeR-Miss, RGBT234-Miss, VTUAV-Miss) so simulated missing patterns may not cover all real sensor failures.",
"Long-term token and multi-template history increase memory and compute versus single-frame fusion baselines.",
],
},
assumptions: [
"Recent historical frames of the missing modality exist and are locally aligned enough for short-term cross-attention.",
"Strong versus weak Gaussian noise is a sufficient proxy for missing versus complete fusion objectives.",
],
computation:
"Initialized from ODTrack; 6 RTX 4090 GPUs with batch 24; LasHeR and RGBT234 trained 30 epochs of 40000 samples, VTUAV 5 epochs of 60000 samples; weight decay 1e-4 late in training.",
relations: [
{ to: "T057", type: "uses-as-baseline", note: "Evaluates and advances LasHeR complete and missing protocols." },
{ to: "T083", type: "uses-as-baseline", note: "Compares against ViPT visual-prompt multimodal tracker on all RGBT sets." },
{ to: "T097", type: "improves", note: "Outperforms UnTrack single-model any-modality tracker, especially under missing inputs." },
{ to: "T071", type: "uses-as-baseline", note: "Compares against OSTrack ViT-B one-stream baseline." },
],
concepts: ["multimodal", "attention", "memory-network", "diffusion-tracking", "success-plot", "benchmark-design"],
impact:
"Establishes noise-level adaptation plus dual temporal conditioning as a strong unified baseline for modality-missing RGBT tracking without architecture switching.",
},
{
id: "T136",
arxiv: "2607.25239",
title: "CD-RMOT-Bench: Benchmarking the Cross-Domain Referring Multi-Object Tracking",
shortTitle: "CD-RMOT-Bench and QCA",
year: 2026,
authors: ["Xiangqun Zhang", "Likai Wang", "Zekun Qian", "Ruize Han", "Wei Feng"],
fileName: "2607.25239v1.pdf",
task: "benchmark",
tags: ["benchmark", "referring-tracking", "multi-object", "cross-domain", "domain-adaptation", "vision-language", "query-centric", "digital-twin"],
difficulty: "intermediate",
summary:
"CD-RMOT-Bench defines cross-domain referring multi-object tracking: train on a labeled source domain and follow language expressions in an unlabeled target domain with only videos and expressions available for adaptation. It combines Refer-KITTI-V2 real clear data, Refer-vKITTI controlled digital-twin core with 10 weather and viewpoint domains built by identity mapping and expression transfer, and Refer-BDD real adverse data, totaling 121 videos, 38573 frames and 42118 expressions. The QCA reference method stabilizes queries with a temporal mean teacher, aligns task-relevant queries adversarially and anchors them to language, beating TempRMOT on every transfer while remaining below in-domain performance.",
problem:
"RMOT is evaluated in-domain where train and test share visual distribution, so it is unknown whether language-conditioned trackers stay reliable under weather, viewpoint and synthetic-real shifts. Real videos entangle domain shift with scene and semantic changes while synthetic sets lack realism, leaving no controlled plus realistic evaluation suite.",
background: ["mot", "tracking-by-detection", "data-association", "hungarian", "hota", "benchmark-design"],
previousWork: [
{
name: "RMOT methods and benchmarks (TransRMOT, TempRMOT and Refer-KITTI-V2, EchoTrack, MGLT, reasoning and zero-shot variants)",
limitation:
"Evaluate closed or nearly in-domain behavior and improve language fusion without testing expression-conditioned trajectories under visual domain shift.",
whyThisPaper:
"Creates controlled weather and viewpoint transfer plus bidirectional synthetic-real transfer under one RMOT protocol.",
},
{
name: "Online tracking-by-detection and transformer trackers (SORT, DeepSORT, FairMOT, ByteTrack, OC-SORT, TransTrack, TrackFormer, MOTR)",
limitation:
"Provide strong detection, association and temporal reasoning but do not determine trajectories from open-ended referring expressions.",
whyThisPaper:
"Uses them as tracking foundations while adding language-conditioned query adaptation as the transfer object.",
},
{
name: "Cross-domain adaptation and video adaptation (DANN, ADDA, detection teachers, TA3N, Tent, CoTTA, synthetic-to-real video studies)",
limitation:
"Align features or instances for recognition and detection without a protocol for expression-conditioned multi-object trajectories where detection, association and selection must be judged jointly.",
whyThisPaper:
"Defines language-conditioned unsupervised adaptation for RMOT and provides query-level stabilization, alignment and language anchoring.",
},
],
researchGap:
"No benchmark isolated controlled visual domain shift with fixed referring semantics while also supporting realistic synthetic-real referring tracking transfer, and no query-centric adaptation baseline targeted the resulting association and selection failures.",
contribution: [
"New CD-RMOT problem formulation with language-conditioned unsupervised adaptation: target videos and expressions available, target boxes, identities, masks and expression-object links reserved for evaluation.",
"CD-RMOT-Bench with Refer-vKITTI controlled core via per-scene identity mapping, expression carry-over, visibility filtering and residual quality control, plus Refer-KITTI-V2 and Refer-BDD anchors.",
"QCA framework with CDT-MT temporal teacher-student consistency, ACDA curricular query-level adversarial alignment and LGCA language-anchored prototype refinement.",
"Experiments showing domain shift breaks expression-consistent association and selection more than detection alone, with ablations, extended decompositions and qualitative cases.",
],
method: {
pipeline: ["map-identities-across-domains", "transfer-expressions", "quality-control", "supervised-source-rmot-train", "stabilize-target-queries", "align-task-relevant-queries", "language-anchor-correction", "trajectories-evaluation"],
architecture:
"Shared RMOT backbone emitting temporal object queries with objectness, box and referring logits; QCA adds EMA teacher consistency, domain classifier with gradient reversal on retained queries, and momentum prototype residual refinement.",
motionModel: "Not applicable — temporal behavior modeled through track queries and teacher-student box and referring consistency rather than an explicit motion filter.",
appearanceModel:
"Visual encoder plus text encoder fused into language-conditioned temporal queries; task relevance scored as sigmoid objectness times sigmoid referring logit.",
association:
"Query-based track decoding with expression-conditioned selection; TrackEval-based RMOT protocol reports trajectory-level detection and association.",
detectionDependency: "Detector-free query predictions evaluated under RMOT protocol; detection quality diagnosed via DetA decomposition.",
trackManagement:
"Temporal queries propagate identities; LGCA refines queries with prototype residual blending gated by quality while keeping language correspondence.",
optimization:
"AdamW for 60 epochs on 4 RTX 3090 GPUs; total loss is source RMOT plus weighted teacher, adversarial and language-guided terms with warmup and adversarial scheduling.",
},
equations: [
{
id: "qca-query",
label: "Language-conditioned temporal queries",
formula: "Q = {q1..qN}, qi in R^d with objectness oi, box bi and referring logit ri; ci = sigmoid(oi) * sigmoid(ri)",
variables: [
{ symbol: "qi", meaning: "temporal object query embedding" },
{ symbol: "oi, ri", meaning: "objectness and referring logits" },
{ symbol: "ci", meaning: "task-relevance score for alignment" },
],
intuition: "Each query proposes an object and says how well it matches the sentence; only queries that look like objects and match the text matter for transfer.",
why: "Focuses adaptation on queries that decide referring trajectories instead of global scene features.",
where: "Shared RMOT backbone output; ci weights ACDA adversarial alignment.",
paperIds: ["T136"],
},
{
id: "qca-teacher",
label: "Cross-domain temporal mean teacher",
formula: "teacher = alpha * teacher + (1 - alpha) * student; Lmt over reliable subset with feature, smooth-L1 box and referring consistency",
variables: [
{ symbol: "alpha", meaning: "EMA decay, 0.999" },
{ symbol: "Omega_t", meaning: "retained reliable target query subset with soft weights" },
{ symbol: "psi", meaning: "selection score combining teacher reliability and student-teacher agreement" },
],
intuition: "Keep a slow-moving teacher copy and force the student to agree with it only on trustworthy target queries.",
why: "Stabilizes target-domain queries that fluctuate in features, boxes or referring confidence under shift.",
where: "Target adaptation path with warmup factor eta(e); no target boxes or identities used.",
paperIds: ["T136"],
},
{
id: "qca-language",
label: "Language-guided query correction",
formula: "kappa_i = (1 + cos(qi, l))/2 * nu_i; p_hat = weighted mean of confident queries; p = m*p_prev + (1-m)*p_hat; q_out = (1-a)*q + a*q_star",
variables: [
{ symbol: "l", meaning: "pooled referring expression anchor" },
{ symbol: "nu_i", meaning: "tracking confidence" },
{ symbol: "p", meaning: "momentum language prototype with momentum m 0.9" },
{ symbol: "alpha_i", meaning: "quality-modulated refinement weight" },
],
intuition: "Pull drifting queries back toward the sentence meaning by blending them with appearance and motion anchors built around a slowly updated prototype.",
why: "Preserves expression-consistent target selection when visually plausible distractors compete.",
where: "LGCA module applied during training and inference; language used as anchor not as extra labels.",
paperIds: ["T136"],
},
],
datasets: ["others"],
metrics: ["hota", "deta", "assa"],
baselines: ["TempRMOT", "TransRMOT", "MGLT", "MUTR", "ReferDINO"],
results: [
"In-domain vKITTI clear to clear TempRMOT reference is 28.69 HOTA, 17.38 DetA, 47.39 AssA; changing only target domain already causes large drops.",
"Controlled transfer with QCA versus TempRMOT: clear to fog 21.37 versus 16.00 HOTA and 41.38 versus 29.20 AssA; clear to rain 24.20 versus 19.50 HOTA; clear to 30L 25.25 versus 20.43 HOTA.",
"Cross-real transfer with QCA versus TempRMOT: clear to Refer-KITTI-V2 18.92 versus 17.17 HOTA; clear to Refer-BDD 19.09 versus 14.32 HOTA; Refer-KITTI-V2 to vKITTI fog 14.86 versus 11.47 HOTA with AssA 43.98 versus 24.57.",
"Adapted RVOS baselines MUTR and ReferDINO stay near 3 to 7 HOTA across transfers, far behind native RMOT methods, showing mask prediction does not solve identity-aware referring trajectories.",
],
ablations: [
"Two-direction ablations: single modules give small gains over TempRMOT, two-module combinations stronger, full CDT-MT plus ACDA plus LGCA best at 21.37 HOTA clear to fog and 19.09 HOTA clear to Refer-BDD.",
"Extended decompositions show controlled shifts hurt both DetA and especially AssA in fog and rain, supporting association and selection failure diagnosis.",
"Construction audit: 285160 initial expression-track pairs expanded over 10 domains, 272043 retained for 95.40 percent retention; 6887 overlapping candidates to 374 IoU-supported to 251 matched identities; 806110 frame mentions removed for target absence.",
],
limitations: {
authorStated: [
"Results remain below the in-domain reference, leaving substantial room for language-conditioned cross-domain tracking.",
"Refer-vKITTI is a controlled digital-twin shift protocol rather than a scene-disjoint or identity-disjoint generalization benchmark; realism axis relies on Refer-KITTI-V2 and Refer-BDD complements.",
"RVOS baselines are adapted references via pseudo-mask rasterization and mask-to-box conversion, not native RMOT competitors.",
],
evident: [
"Only five aligned sequences underlie the controlled core with limited source scenes, so weather and viewpoint conclusions rest on a narrow scene sample.",
"Target expressions are available during adaptation, so the setting is not fully unsupervised in the language channel despite no target tracking labels.",
],
},
assumptions: [
"Identity mapping with permissive IoU evidence plus one-to-one assignment preserves referring semantics across domains after visibility filtering.",
"Source and target split identity provides sufficient domain labels for query-level adversarial alignment.",
],
computation:
"AdamW 60 epochs with decay after epoch 40, batch 1 per GPU on 4 RTX 3090 GPUs; EMA decay 0.999, top-M 16 queries in ACDA, prototype momentum 0.9; TrackEval-based RMOT evaluation.",
relations: [
{ to: "T006", type: "extends", note: "Extends MOT16-style detection plus association evaluation to cross-domain language-conditioned trajectories." },
{ to: "T005", type: "conceptual-successor", note: "Builds on SORT online tracking-by-detection lineage as a tracking foundation." },
{ to: "T061", type: "conceptual-successor", note: "Continues ByteTrack-era association foundations while adding referring selection." },
{ to: "T053", type: "conceptual-successor", note: "Continues TransTrack transformer query tracking foundations for temporal queries." },
],
concepts: ["mot", "tracking-by-detection", "data-association", "hota", "benchmark-design", "cost-matrix"],
impact:
"Opens robust language-guided tracking across visual domains and diagnoses that transfer fails at temporal association and expression-consistent selection, not only detection.",
},
{
id: "T137",
arxiv: "2607.26511",
title: "Semantic-Aware Temporal Adaptation for UAV Anti-UAV Tracking",
shortTitle: "SATATrack",
year: 2026,
authors: ["Xiaozhen Qiao", "Da Zhang", "Yubin Guo", "Junyu Gao", "Zhiyuan Zhao", "Xuelong Li"],
fileName: "2607.26511v1.pdf",
task: "multimodal-tracking",
tags: ["uav-tracking", "anti-uav", "vision-language", "state-space-model", "temporal-adaptation", "test-time-adaptation", "contrastive-learning"],
difficulty: "advanced",
summary:
"SATATrack tackles air-to-air UAV Anti-UAV tracking where both observer and target move by letting stable language descriptions control temporal memory and by aligning feature distributions at test time without gradient updates. Semantic-Aware Context Propagation injects a CLIP semantic embedding into selective state-space write and retention dynamics across Fast-iTPN stages, while Temporal-Aware Distribution Alignment blends recent-frame top-k statistics with frozen training statistics for center and size branches. It reaches 46.40 AUC on UAV-Anti-UAV, beating MambaSTS by 2.37 points and leading all attributes, with best average scores across Anti-UAV318, DUT Anti-UAV, UAV123 and UAVDT at real-time speed.",
problem:
"Dual-dynamic air-to-air tracking induces rapid viewpoint change, motion blur, scale variation and UAV-like distractors while test distributions shift per video. Fixed visual representations drift because propagated states absorb distractor evidence, and online parameter updates can amplify errors from unreliable predictions.",
background: ["sot", "bounding-box", "attention", "multimodal", "motion-model", "state-space", "success-plot"],
previousWork: [
{
name: "Pure visual trackers (Siamese, discriminative, OSTrack, ODTrack, TCTrack, HiFT, Mamba variants)",
limitation:
"Temporal memory driven by visual similarity and confidence absorbs look-alike background when tiny blurred targets become ambiguous, causing drift and lag.",
whyThisPaper:
"Conditions state-space write gate and retention timescale on target language so memory keeps described-target evidence across backbone stages.",
},
{
name: "Vision-language trackers (JointNLT, CiteTracker, All-in-One, UVLTrack, SUTrack, DUTrack, ATCTrack)",
limitation:
"Consume language once at matching, query or fusion stage without steering how temporal states evolve, so propagated states stay vulnerable to contamination.",
whyThisPaper:
"SACP makes semantics the controller of temporal memory rather than a one-shot matching cue, with multilevel injection and contrastive hardening.",
},
{
name: "Temporal and state-space models plus MambaSTS spatial-temporal-semantic baseline",
limitation:
"Fuse semantics as an extra feature while transitions stay visually driven; test-time statistic mismatch from generic training data remains unaddressed.",
whyThisPaper:
"Adds parameter-free TADA that recalibrates normalization statistics from recent observations while freezing learned weights.",
},
],
researchGap:
"No UAV Anti-UAV tracker used target language to directly govern selective state-space memory dynamics together with gradient-free temporal distribution alignment for per-video shifts.",
contribution: [
"SATATrack framework combining language-guided temporal modeling with inference-time feature-distribution alignment.",
"SACP module with token-level semantic write gating and sequence-level retention modulation across four backbone interaction points.",
"Semantic-discriminative contrastive regularizer mining top-M background patches most similar to the description as hard negatives.",
"TADA online alignment with temporal queue, top-k spatial statistics, confidence gating and center-plus-size adaptation without parameter updates.",
],
method: {
pipeline: ["encode-template-search-and-description", "propagate-semantic-aware-context-across-stages", "inject-context-into-visual-stream", "write-back-visual-evidence", "contrastive-harden-during-training", "align-distributions-at-test-time", "predict-box"],
architecture:
"Fast-iTPN-Base visual backbone with CLIP ViT-B32 frozen text encoder producing 512-dim semantics; SACP inserted at four stages with 512-dim neck; localization head with center and size branches.",
motionModel:
"Selective state-space temporal propagation across frames and stages with data-dependent step size and transition matrix; no explicit Kalman filter.",
appearanceModel:
"Multilevel visual tokens refined by propagated target-aware context through input and output cross-attention plus residual fusion.",
association: "Not applicable (single-object; decoded box from temporally stabilized feature map).",
detectionDependency: "None — template plus search plus description given; description encoded once per sequence.",
trackManagement:
"No long-term re-detection branch; TADA smooths statistics over a short temporal queue and rejects drifted or low-confidence frames.",
loss:
"Localization loss with focal weight 1.0 plus GIoU 2.0 plus L1 5.0, with semantic-discriminative contrastive loss weight 0.2 enabled in stage-two finetuning; AdamW with gradient clipping 0.1.",
},
equations: [
{
id: "sata-sacp",
label: "Semantic-aware context propagation stage",
formula: "C_bar, H = SACP(C_prev, H_prev, s); F_tilde = F + Attn_in(F, C_bar, C_bar); F_out = Block(F_tilde); C_new = Attn_out(C_bar, F_out, F_out) + C_bar",
variables: [
{ symbol: "C, H", meaning: "propagated context tokens and recurrent hidden state" },
{ symbol: "s", meaning: "global CLIP semantic embedding of target description" },
{ symbol: "F", meaning: "visual tokens at the current backbone stage" },
],
intuition: "Update a target-aware memory with language control, lend it to vision, then write fresh visual evidence back for the next stage and frame.",
why: "Provides persistent semantic guidance through the visual hierarchy instead of one-shot language fusion.",
where: "Repeated across backbone stages and video frames; TADA applied later at test time.",
paperIds: ["T137"],
},
{
id: "sata-gate",
label: "Semantic write and retention control",
formula: "Delta = Softplus(W u + b); g = sigmoid(MLP([u; s_prime])); Delta_tilde = Delta * (1 + eta * g); A modulated by s_prime for retention timescale",
variables: [
{ symbol: "Delta", meaning: "selective state-space step size acting as write and forget gate" },
{ symbol: "A", meaning: "state-transition matrix setting memory retention timescale" },
{ symbol: "g", meaning: "token-level semantic gate amplifying target-consistent tokens" },
{ symbol: "s_prime", meaning: "projected semantic embedding in SSM hidden space" },
],
intuition: "Let the description decide which observations may write into memory and how long consistent context is kept.",
why: "Down-weights distractor tokens at update time so contamination does not accumulate, especially under similar distractors.",
where: "Inside each SACP selective scan; full design beats text-fusion-only by large margins.",
params: "Removing Delta modulation or A modulation each costs 0.56 to 1.17 AUC points; A removal hurts more.",
paperIds: ["T137"],
},
{
id: "sata-sdc",
label: "Semantic-discriminative contrastive loss",
formula: "z_v = psi_v(v_plus), z_s = psi_s(s); negatives = TopK sim(background, z_s); L = -log exp(s_plus) / (exp(s_plus) + sum exp(s_minus))",
variables: [
{ symbol: "v_plus", meaning: "masked average pooled target feature from ground-truth box" },
{ symbol: "s_plus", meaning: "target-visual to semantic cosine similarity over temperature" },
{ symbol: "negatives", meaning: "top-M background patches most similar to description" },
],
intuition: "Force the target to look like its description while pushing away the most confusing background patches.",
why: "Sharpens separation from semantically similar sky, building and vegetation distractors during training.",
where: "Training-only auxiliary loss; hard negatives mined per frame from propagated feature map.",
paperIds: ["T137"],
},
{
id: "sata-tada",
label: "Temporal-aware distribution alignment",
formula: "lambda0 = n / (n + N); test stats from top-k spatial responses smoothed over temporal queue then blended with frozen train stats; applied to center and size branches",
variables: [
{ symbol: "N, n", meaning: "candidate count 8 and current-frame weight 6 controlling blending" },
{ symbol: "rho", meaning: "top-k ratio 0.20 selecting informative spatial locations" },
{ symbol: "alpha", meaning: "output blending coefficient 1.0 favoring adapted prediction" },
],
intuition: "Recalibrate normalization to the current video using recent reliable evidence, without touching learned weights.",
why: "Reduces video-specific shifts in sensor, illumination, altitude and background while avoiding error amplification from bad predictions.",
where: "Pure test-time module with confidence gate, drift rejection, clipping and regularization.",
params: "N 8, n 6, alpha 1.0 and rho 0.20 best; larger rho admits noisy background statistics.",
paperIds: ["T137"],
},
],
datasets: ["uav123", "uavdt", "others"],
metrics: ["success-auc", "precision", "norm-precision"],
baselines: ["OSTrack", "ODTrack", "MambaLCT", "MCITrack-B224", "MambaTrack", "MambaSTS", "DUTrack-256", "ATCTrack", "UVLTrack", "CiteTracker-256", "SiamRPN++", "SiamFC++", "SiamBAN", "SiamCAR", "TCTrack", "HiFT", "Aba-ViTrack", "SGLATrack", "ORTrack", "AutoMatch", "LightTrack", "SiamGAT"],
results: [
"UAV-Anti-UAV: 46.40 AUC, 54.94 OP50, 38.63 OP75, 62.39 precision, 68.64 normalized precision; beats MambaSTS by 2.37, 0.58, 5.60, 1.04, 0.34 points with notably stronger high-overlap localization.",
"Anti-UAV318 67.27 AUC and DUT Anti-UAV 66.36 AUC are best; average over three anti-UAV sets 60.01 AUC and 77.88 precision is best despite slightly lower DUT precision than ODTrack.",
"Attributes: best on all six with 40.19 fast motion, 39.88 motion blur, 31.67 small object, 35.06 out-of-view, 41.32 similar distractor and 45.10 aspect-ratio variation, beating MambaSTS by 1.64, 2.80, 1.01, 0.75, 3.04, 2.28 points.",
"Generic UAV transfer: UAV123 70.28 AUC and 92.31 precision best; UAVDT 65.80 AUC best with 86.93 precision close to DUTrack 87.73; 41.91 fps with 189.87M total params including frozen CLIP text encoder and 126.44M trainable.",
],
ablations: [
"Components from 35.84 baseline: SACP alone 42.37 AUC, plus SDC 44.07, plus TADA full 46.40 with precision 47.90 to 59.86 to 61.60 to 62.39.",
"SACP designs without TADA: direct text fusion 40.32, without Delta modulation 43.51, without A modulation 42.90, full SACP 44.07 AUC.",
"TADA: without top-k statistics 43.34, center-only 43.84, without temporal queue 45.51, size-only 45.72, full 46.40 AUC; top-k selection most critical.",
],
limitations: {
authorStated: [
"Smallest gain on small objects where few pixels limit both semantic and visual evidence; suggests need for tiny-target localization or re-detection.",
"Out-of-view gain small because missing visual evidence cannot be fully recovered without an explicit long-term re-detection branch.",
"Failures remain under long absence with outdated priors, multiple UAV-like instances, severe blur destroying boundaries, and extreme few-pixel scale sensitivity.",
],
evident: [
"Response and stability claims rest on selected qualitative clips and three temporal curves rather than full-set drift statistics.",
"Two-stage pretraining on GOT-10k, LaSOT, TrackingNet and COCO plus finetuning means generic data strongly shapes the reported transfer.",
],
},
assumptions: [
"Natural-language target description is stable across frames and correctly describes the intended UAV instance.",
"Recent-frame statistics with confidence and drift gating are more trustworthy than frozen training statistics for the current video.",
],
computation:
"Two-stage training on 4 H100 GPUs with batch 64: stage one 300 epochs base lr 4e-4 with 0.1 backbone multiplier decayed at 240, stage two 100 epochs at 1e-4 dropped at 80 with frozen BN; template 112 and search 224 with patch stride 16.",
relations: [
{ to: "T009", type: "conceptual-successor", note: "Continues Siamese appearance-matching lineage among compared pure-vision trackers." },
{ to: "T071", type: "uses-as-baseline", note: "Compares against OSTrack one-stream transformer on all anti-UAV sets." },
{ to: "T068", type: "uses-as-baseline", note: "Compares against TCTrack temporal-context aerial tracker." },
{ to: "T122", type: "extends", note: "Evaluates air-to-air UAV Anti-UAV protocol related to the million-scale AntiUAV benchmark direction." },
],
concepts: ["sot", "multimodal", "attention", "motion-model", "memory-network", "success-plot"],
impact:
"Demonstrates that semantics should govern temporal memory selection dynamics and that frozen-weight statistic alignment is a safe test-time complement for dual-dynamic tracking.",
},
{
id: "T138",
arxiv: "2608.00847",
title: "Models as Tools: An Agentic Coordination Framework for Unified Multimodal Visual Tracking",
shortTitle: "ACTrack",
year: 2026,
authors: ["Wenrui Cai", "Yuzhe Li", "Qingjie Liu", "Yunhong Wang"],
fileName: "2608.00847v1.pdf",
task: "multimodal-tracking",
tags: ["agentic-tracking", "unified-tracking", "multimodal", "sam3", "vision-language-model", "tool-coordination", "parameter-efficient", "lora"],
difficulty: "advanced",
summary:
"ACTrack reformulates tracking as agentic tool coordination instead of scaling one prediction stream. A Tracker-based Instance Matching Tool gives default boxes, a SAM3 Motion Tool supplies mask-derived motion priors and search re-anchoring with Kalman re-ranking, a SAM3 Perception Tool detects similar instances and flags conflicts, and a VLM Reprompt Tool arbitrates only after five consecutive conflicting frames. ACTrack-L384 reaches 81.4 LaSOT AUC and 86.8 GOT-10k AO, while the parameter-efficient ACTrack-EM shares a frozen SAM3 backbone with stream-specific LoRA to unify RGB, RGB-D, RGB-T, RGB-Event and RGB-Language tracking with only 173.3M trainable parameters.",
problem:
"Matching-based one-stream transformers compress correspondence, foreground perception, disappearance recovery, distractor rejection and multimodal handling into one stream, so gains depend on context length and scale and hit a bottleneck. Monolithic foundation-model conversions ignore that matching trackers, SAM-style segmenters and VLMs fail in different ways and lack mutual error correction.",
background: ["sot", "bounding-box", "iou", "siamese", "attention", "multimodal", "success-plot"],
previousWork: [
{
name: "Matching-based one-stream trackers (OSTrack joint encoding, MixFormer, SeqTrack and ARTrack autoregression, ODTrack and HIPTrack temporal prompts, MCITrack context)",
limitation:
"Single prediction stream produces temporally smooth but semantically wrong boxes under occlusion, distractors or polluted history, with Hanning priors biasing toward local candidates.",
whyThisPaper:
"Treats any compatible matching tracker as a replaceable Instance Matching Tool checked by SAM3 instance evidence.",
},
{
name: "Foundation-model trackers (LoRAT low-rank adaptation, SPMTrack spatio-temporal tuning, SAMURAI motion-aware memory, DAM4SAM distractor-aware memory)",
limitation:
"Convert a backbone or segmenter into the whole tracker instead of invoking each model for the subproblem it does best.",
whyThisPaper:
"Invokes SAM3 as Motion and Perception tools for mask priors, anchors and conflict evidence while keeping dense localization in the tracker.",
},
{
name: "Unified multimodal trackers (SeqTrackV2, UnTrack any-modality, SUTrack simple unified, Uni-MDTrack, plus ProTrack, ViPT, SDSTrack, OneTracker)",
limitation:
"Fuse every modality inside every component or train separate models, coupling modality adaptation to the whole pipeline.",
whyThisPaper:
"Isolates modality fusion inside the Instance Matching Tool via ACTrack-EM while motion, perception and reprompting stay unchanged on RGB.",
},
],
researchGap:
"No framework coordinated heterogeneous matching, segmentation and VLM models through search re-anchoring, instance-conflict detection and sparse reprompt renewal while sharing parameters across RGB and RGB-X tracking.",
contribution: [
"Models-as-tools formulation with shared target state, tool-invocation triggers and inter-tool coordination policy rather than fixed fusion.",
"ACTrack closed loop: Kalman re-ranked SAM3 motion boxes, tracker prediction under motion prior, SAM3 instance assignment and conflict flag, persistent-conflict VLM arbitration with SAM3 reprompting.",
"ACTrack-EM unified multimodal matching tool: modality-aware patch embedding with residual fusion, dynamic auxiliary templates, stream-specific LoRA and chunked causal temporal attention with language prefix.",
"Large evaluation over twelve RGB and multimodal datasets showing top-three sweeps and strong parameter and speed efficiency.",
],
method: {
pipeline: ["propagate-sam3-mask-and-motion-prior", "re-anchor-search-region", "match-instance-under-prior", "detect-instances-and-assign-identities", "flag-conflict", "arbitrate-persistent-conflict-with-vlm", "commit-shared-state"],
architecture:
"RGB ACTrack-B224 on MCITrack-B224 and ACTrack-L384 on MCITrack-L384 with SAM3 Motion plus Perception and Seed 2.0 Pro VLM; ACTrack-EM uses frozen 31-layer SAM3 encoder with hidden 1024, LoRA rank 64, modality experts, dynamic templates and fully convolutional style box plus center heads.",
motionModel:
"Constant-velocity Kalman filter over center, aspect, height and velocities; candidate masks scored by predicted-IoU plus geometric IoU with motion weight 0.15 after 15-frame warmup; objectness below 0.5 clears the filter.",
appearanceModel:
"Tracker discriminative matching under mask-guided crop plus SAM3 foreground masks and similar-instance detections for identity checking.",
association:
"Perception assignment maps tracker and motion box centers into mask or box instances with confidence-first and distance-tiebreak selection; same-instance means no conflict.",
detectionDependency: "SAM3 detection branch with 200 queries and 0.5 confidence and mask thresholds supplies instance evidence inside the search crop.",
trackManagement:
"Shared state of accepted box, tracker state, SAM3 video state and K-frame conflict buffer committed each frame; dynamic auxiliary templates refreshed FIFO only when center confidence exceeds 0.9 with frame-gap spacing.",
optimization:
"RGB variants use off-the-shelf checkpoints with no extra training; ACTrack-EM two-stage training on 8 H800 GPUs with AdamW, warmup to 1e-4 and cosine decay, GIoU plus center BCE plus modality cross-entropy all weighted 1.0.",
},
equations: [
{
id: "act-state",
label: "Shared agentic target state",
formula: "Omega_t = {b_bar_{t-1}, H_{t-1}, S_{t-1}, Q_{t-1}, (I0, b0)}",
variables: [
{ symbol: "b_bar", meaning: "last accepted target box" },
{ symbol: "H, S", meaning: "instance-matching tracker state and SAM3 video memory state" },
{ symbol: "Q", meaning: "sliding buffer of recent conflict flags" },
{ symbol: "I0, b0", meaning: "immutable first-frame reference" },
],
intuition: "Keep one agreed target story that every tool reads and writes, instead of letting each model drift independently.",
why: "Makes each frame decision condition the next frame across heterogeneous tools.",
where: "Maintained every frame; accepted box committed back to both tracker and SAM3 states.",
paperIds: ["T138"],
},
{
id: "act-motion",
label: "Motion-aware mask selection",
formula: "i_star = argmax_i (1 - Wm) * u_i + Wm * g_i; Rt = Expand(b_t^m, rho)",
variables: [
{ symbol: "u_i", meaning: "mask quality score from SAM3 decoder" },
{ symbol: "g_i", meaning: "geometric IoU of candidate mask box with Kalman prediction" },
{ symbol: "Wm", meaning: "motion prior weight 0.15 after warmup" },
{ symbol: "Rt", meaning: "re-anchored search region with expansion rho 2.5 RGB or 1.5 RGB-X" },
],
intuition: "Prefer masks that look good and agree with where physics says the target should be, then search around that motion box.",
why: "Supplies a clean mask-derived prior so the tracker crop is not misled by background pixels.",
where: "SAM3 Motion Tool; adds no learnable parameters and leaves SAM3 frozen.",
simulator: "motion",
paperIds: ["T138"],
},
{
id: "act-match",
label: "Instance matching under motion prior",
formula: "(b_t^r, H_t^r) = D(It, Rt, H_{t-1}); accept b_t^r if Ct = 0 else b_t^m unless VLM says otherwise",
variables: [
{ symbol: "D", meaning: "pluggable matching tracker interface" },
{ symbol: "b_t^r, b_t^m", meaning: "tracker-side and motion-tool boxes" },
{ symbol: "Ct", meaning: "perception conflict flag from instance assignment" },
],
intuition: "Let the tracker do what it does best inside a motion-guided window, but distrust it when SAM3 sees two tools on different physical objects.",
why: "Avoids contaminating the tracker with distractor evidence while preserving annotated-instance fidelity by default.",
where: "Default policy on ordinary frames; VLM path only after persistent conflict.",
simulator: "siamese",
paperIds: ["T138"],
},
{
id: "act-vlm",
label: "Sparse VLM reprompt arbitration",
formula: "yt = V((I0,b0), It, b_t^r, b_t^m) in {Matching, Motion, Uncertain}; trigger only after K = 5 same-conflict frames",
variables: [
{ symbol: "yt", meaning: "three-way identity verdict" },
{ symbol: "K", meaning: "persistence threshold of 5 consecutive frames" },
{ symbol: "V", meaning: "VLM judge over reference plus current frame with both hypotheses overlaid and per-box crops" },
],
intuition: "Call expensive semantic judgment only when tools disagree persistently, then refresh the losing tool with the winner.",
why: "Keeps dense localization fast and stable while allowing rare high-level correction of polluted SAM3 memory or drifted tracker.",
where: "Reprompt Tool; Matching verdict accepts tracker box and injects it as a new SAM3 prompt from the next frame.",
paperIds: ["T138"],
},
],
datasets: ["lasot", "got10k", "trackingnet", "lasheR", "uav123", "otb", "others"],
metrics: ["success-auc", "precision", "norm-precision"],
baselines: ["MCITrack-L384", "MCITrack-B224", "SPMTrack-L", "SPMTrack-G", "LoRAT-G378", "LoRAT-L378", "SAMITE-B", "SAMURAI-L", "SAM2.1-L", "SeqTrack-L384", "ARTrack-L384", "ARTrackV2-L384", "ODTrack-L384", "MambaLCT384", "RELO-L256", "SUTrack-L384", "UVLTrack-L", "ViPT"],
results: [
"LaSOT: ACTrack-B224 80.1 AUC and ACTrack-L384 81.4 AUC, first to pass 80, with best precision and normalized precision and leads on every attribute subset; TrackingNet ACTrack-L384 88.0 AUC best; LaSOText ACTrack-L384 66.5 and ACTrack-EM 65.8 AUC top two.",
"GOT-10k: ACTrack-L384 86.8 AO, 95.9 SR0.5, 87.2 SR0.75 best; ACTrack-B224 83.8 AO; reported separately because foundation pretraining breaks strict one-shot protocol.",
"TNL2K RGB-Language: ACTrack-EM 77.2 AUC, 87.1 normalized precision, 84.7 precision, beating SUTrack-L384 67.9 by a clear margin.",
"Classical RGB: ACTrack-B224 best on OTB2015 73.9, UAV123 74.3 and NfS 73.7 AUC; VastTrack all three ACTrack variants beat all competitors by a large margin.",
"Multimodal with ACTrack-EM: VisEvent improves previous best by over 10 AUC points; also substantially leads LasHeR, DepthTrack and other RGB-D, RGB-T and RGB-Event tables.",
],
ablations: [
"Individual-tool analysis section started in the paper shows matching and SAM3 tools fail differently across multi-distractor, disappearance, expansion and conflict cases motivating the decomposition.",
"Parameter and speed table: ACTrack-B224 0 trainable with 613.5M total at 15.1 fps, ACTrack-L384 0 trainable with 865.0M at 12.7 fps, ACTrack-EM 173.3 trainable of 559.0M at 14.2 fps versus SPMTrack-G 204.0 of 1339.5M at 8.6 fps.",
"Coordination settings: expansion rho 2.5 RGB and 1.5 RGB-X, motion weight 0.15, objectness gate 0.5, Kalman warmup 15 frames, conflict persistence 5 frames, template refresh only above 0.9 center confidence.",
],
limitations: {
authorStated: [
"GOT-10k comparison reported for reference because large-scale pretrained foundation models may no longer satisfy its one-shot protocol.",
"Public baseline numbers taken from published comparison tables rather than rerun under one harness.",
],
evident: [
"System complexity spans SAM3 motion plus perception plus tracker plus sparse VLM calls, making failure attribution and deployment harder than a single tracker.",
"Multimodal evidence handled only inside the matching tool while other tools stay on RGB, so non-RGB failures outside the matching crop have limited correction paths.",
],
},
assumptions: [
"Any compatible tracker exposes init, prior-conditioned matching and state-commit operations so it can be swapped as the matching tool.",
"SAM3 instance detections inside the effective search crop are sufficient evidence for tracker versus motion identity conflict.",
],
computation:
"RGB ACTrack variants assembled from off-the-shelf checkpoints with no training; ACTrack-EM trained on LaSOT, GOT-10k, COCO, TrackingNet, VastTrack, TNL2K, DepthTrack, VisEvent and LasHeR sampled equally with 4-frame clips on 8 H800 GPUs for two 50-epoch stages; speed averaged over LaSOT test on RTX 4090.",
relations: [
{ to: "T071", type: "builds-on", note: "Uses OSTrack-style one-stream matching lineage as the Instance Matching Tool basis." },
{ to: "T009", type: "conceptual-successor", note: "Continues Siamese instance-correspondence lineage coordinated with segmentation tools." },
{ to: "T083", type: "improves", note: "Outperforms ViPT visual-prompt multimodal adaptation on RGB-T, RGB-D and RGB-Event sets." },
{ to: "T097", type: "improves", note: "Outperforms UnTrack single-model any-modality unified tracker across modalities." },
],
concepts: ["sot", "multimodal", "attention", "siamese", "track-management", "success-plot"],
impact:
"Shows agentic coordination with sparse semantic arbitration beats isolated model scaling and offers a reusable parameter-efficient path to unified RGB-X tracking.",
}
];
