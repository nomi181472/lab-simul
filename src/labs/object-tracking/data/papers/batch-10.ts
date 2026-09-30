import type { PaperRecord } from "../types";

/* batch-10 — T079..T088 */

export const BATCH_10: PaperRecord[] = [
{
    id: "T079",
    arxiv: "2210.15511",
    title: "ProContEXT: Exploring Progressive Context Transformer for Tracking",
    shortTitle: "ProContEXT",
    year: 2022,
    authors: ["Jin-Peng Lan", "Zhi-Qi Cheng", "Jun-Yan He", "Chenyang Li", "Bin Luo", "Xu Bao", "Wangmeng Xiang", "Yifeng Geng", "Xuansong Xie"],
    fileName: "2210.15511v4.pdf",
    task: "single-object",
    tags: ["transformer", "context-aware", "progressive-encoding", "static-dynamic-template", "token-pruning", "one-stream"],
    difficulty: "advanced",
    summary:
      "ProContEXT revamps Siamese tracking with progressive context encoding: multi-scale static templates carry spatial context and progressively updated multi-scale dynamic templates carry temporal context. A 12-layer context-aware self-attention jointly encodes template–search interactions, and a revised token-pruning keeps foreground tokens using both template groups. It reports 74.6 AO on GOT-10k and 84.6 AUC on TrackingNet at 54.3 FPS with no extra update branch.",
    problem:
      "Single-template context-free trackers (Siamese and early Transformer) fail in fast-changing and crowded scenes because the first-frame template cannot account for appearance change and cannot disambiguate similar instances. Spatial context (extended background) and temporal context (recent appearance) were studied separately but never coherently encoded in a real-time Transformer tracker.",
    background: ["sot", "bounding-box", "siamese", "attention", "appearance-features", "success-plot"],
    previousWork: [
      {
        name: "Siamese trackers (SiamFC, SiamRPN, SiamRPN++)",
        limitation:
          "Match a single first-frame template by correlation, so deformation, fast change and similar distractors break the fixed similarity.",
        whyThisPaper:
          "ProContEXT replaces single-template matching with a static-plus-dynamic template group and joint self-attention, gaining +1.6 AO on GOT-10k val from multi-scale static templates alone.",
      },
      {
        name: "Transformer trackers without context (TransT, OSTrack)",
        limitation:
          "TransT fuses one static template and OSTrack prunes tokens against the first frame only, so changed appearances lose similarity and get pruned as background.",
        whyThisPaper:
          "Context-aware attention over static and dynamic tokens plus pruning against both groups improves OSTrack-MAE by 0.9 AO / 1.5 SR0.5 / 2.1 SR0.75 on GOT-10k.",
      },
      {
        name: "Dynamic-template trackers with extra branch (STARK, MixFormer)",
        limitation:
          "They need a separate score/update branch and extra compute to refresh the dynamic template.",
        whyThisPaper:
          "ProContEXT reuses its own score-head maximum as confidence (threshold 0.7) to progressively refresh multi-scale dynamic templates with no extra branch.",
      },
    ],
    researchGap:
      "No Transformer tracker progressively encoded multi-temporal and multi-spatial context together while staying real-time.",
    contribution: [
      "Template group of multi-scale static templates S={S1..Sm} (scales K between 2 and 4) plus multi-scale dynamic templates D={D1..Dm} for complementary spatial and temporal context.",
      "Context-aware self-attention over concatenated [Zs;Zdx;Zx] tokens (12 ViT-base layers, MAE pretrained) whose 3x3 block attention matrix separates intra-region, template–search and inter-template interactions.",
      "Revised token pruning scoring search tokens by summed center-template attention from both static and dynamic groups, keeping top-k (ratio 0.7) before blocks 4/7/10; zero-padding pruned tokens for the head.",
      "Score/offset/size tracking head with Gaussian supervision, focal loss plus IoU and L1 box losses; inference reuses the score maximum as the update confidence.",
      "State of the art on TrackingNet (84.6 AUC) and GOT-10k (74.6 AO) at 54.3 FPS, beating extra-branch updaters without extra update compute.",
    ],
    method: {
      pipeline: ["crop-templates", "embed", "context-attention", "prune", "score", "regress", "update-dynamic"],
      architecture:
        "ViT-base backbone (MAE pretrained) on 192x192 templates and 384x384 search, patch 16, 12 context-aware self-attention blocks; tokens reshaped to Wx x Hx and fed to score, offset and size heads (CenterNet-style).",
      motionModel:
        "No filter: temporal continuity via dynamic templates refreshed from bpred whenever score above tau=0.7 (Algorithm 1); search area cropped around previous prediction.",
      appearanceModel:
        "Joint static-template (first frame, multi-scale) and dynamic-template (recent predictions, multi-scale) features fused by self-attention; no online weight update.",
      detectionDependency: "None — class-agnostic single-object tracker.",
      trackManagement:
        "Frame 1: S=crop(I1,binit,K), D=S. Later: X=crop(Ii,bpred); if max score map > 0.7 then D=crop(Ii,bpred,K). Lost/invalid handling is implicit via low score (no update).",
      loss:
        "Loss = Ls + lambda_iou*Liou + lambda_l1*L1 with lambda_iou=2, lambda_l1=5; Ls is focal loss (alpha=2, beta=4) against Gaussian map; Liou is GIoU and L1 is box L1.",
      optimization:
        "PyTorch, AdamW, batch 128, lr 1e-4, 300 epochs on 4xA100; horizontal flip and template scale/size jitter; ablations rescale to 128/256 and train 100 epochs.",
    },
    equations: [
      {
        id: "procontext-attn",
        label: "Context-aware self-attention",
        formula: "A(Q,K,V) = Softmax(Q K^T / sqrt(dk)) V over [Qs;Qd;Qx][Ks;Kd;Kx]^T acting on [Vs;Vd;Vx], 12 layers",
        variables: [
          { symbol: "Qs,Ks,Vs", meaning: "query/key/value tokens of static templates" },
          { symbol: "Qd,Kd,Vd", meaning: "query/key/value tokens of dynamic templates" },
          { symbol: "Qx,Kx,Vx", meaning: "query/key/value tokens of search area" },
          { symbol: "dk", meaning: "attention scaling dimension" },
        ],
        intuition:
          "Let every template piece and every search piece attend to each other in one operation, so background context and recent appearance mix into the search features.",
        why:
          "Diagonal blocks learn intra-region appearance while off-diagonals (QsKx, QdKx, QxKs, QxKd, QsKd) inject spatial and temporal context without a separate fusion module.",
        where: "Section 2.1, Eq. 1; every frame over the concatenated token sequence.",
        params: "12 stacked layers progressively refine context; more scales add tokens and cost linearly.",
        paperIds: ["T079"],
      },
      {
        id: "procontext-pruning",
        label: "Context-aware token pruning score",
        formula: "w = phi(softmax(Qs Kx^T / sqrt(dk)) + softmax(Qd Kx^T / sqrt(dk))) in R^{1xNx}; keep top-k of w",
        variables: [
          { symbol: "w", meaning: "per-search-token foreground score" },
          { symbol: "Nx", meaning: "number of search tokens" },
          { symbol: "phi", meaning: "sum over template-center token rows" },
          { symbol: "k / rho", meaning: "kept tokens / keeping ratio (0.7 before blocks 4,7,10)" },
        ],
        intuition:
          "Rank each search patch by how much the template centers (old and recent) attend to it; drop low-rank background patches.",
        why:
          "Unlike OSTrack which consults only the first frame, consulting dynamic templates keeps changed foreground alive while still cutting GFLOPs by 16.5% at rho=0.7.",
        where: "Section 2.1; pruning before 4th, 7th, 10th blocks, pruned tokens zero-padded for the head.",
        params: "Smaller rho saves compute but risks dropping the target; rho=0.7 gives +1.1 AO with fewer GFLOPs than rho=1.0.",
        paperIds: ["T079"],
      },
      {
        id: "procontext-gaussian",
        label: "Score-map Gaussian supervision",
        formula: "Gxy = exp(-((x-px)^2 + (y-py)^2) / (2 sigma_p^2))",
        variables: [
          { symbol: "(px,py)", meaning: "ground-truth center coordinates" },
          { symbol: "sigma_p", meaning: "object-size-adaptive standard deviation" },
          { symbol: "Gxy", meaning: "target heatmap value at (x,y)" },
        ],
        intuition: "Paint a soft hill centered on the true target so nearby positions get partial credit.",
        why: "Dense smooth supervision for the score head instead of a single hard positive.",
        where: "Section 2.2, Eq. 2; supervises the score head.",
        paperIds: ["T079"],
      },
      {
        id: "procontext-focal",
        label: "Focal score loss",
        formula: "Ls = -sum[(1-Ghat)^alpha log(Ghat) if G=1 else (1-G)^beta (Ghat)^alpha log(1-Ghat)], alpha=2, beta=4",
        variables: [
          { symbol: "Ghat", meaning: "predicted score map in [0,1]^{Wx x Hx}" },
          { symbol: "G", meaning: "Gaussian target map" },
          { symbol: "alpha/beta", meaning: "focusing weights (2 and 4)" },
        ],
        intuition: "Down-weight easy background so the rare target peak drives learning.",
        why: "Handles extreme foreground–background imbalance in dense score maps.",
        where: "Section 2.2, Eq. 3.",
        paperIds: ["T079"],
      },
      {
        id: "procontext-box",
        label: "Box formation and total loss",
        formula: "bhat = (xhat+dhat_x, yhat+dhat_y, what, hhat); Loss = Ls + 2*Liou(bhat,bgt) + 5*L1(bhat,bgt); (xhat,yhat)=argmax(Ghat)",
        variables: [
          { symbol: "bhat / bgt", meaning: "predicted / ground-truth box" },
          { symbol: "(dhat_x,dhat_y)", meaning: "offset-head correction at the peak" },
          { symbol: "(what,hhat)", meaning: "size-head prediction at the peak" },
          { symbol: "Liou / L1", meaning: "GIoU loss and L1 box loss" },
        ],
        intuition: "Pick the hottest score location, nudge it with the offset head, size it with the size head, and penalize bad boxes.",
        why: "Couples classification with geometric accuracy; argmax localization plus Hanning-free direct peak is the whole tracker output.",
        where: "Section 2.2, Eqs. 4-5.",
        params: "Weights 2 and 5 follow STARK/MixFormer convention.",
        simulator: "iou-track",
        paperIds: ["T079"],
      },
      {
        id: "procontext-update",
        label: "Score-reuse dynamic update",
        formula: "score = max(Ghat); if score > tau (0.7): D = crop(Ii, bpred, K)",
        variables: [
          { symbol: "score", meaning: "maximum of the score map, used as confidence" },
          { symbol: "tau", meaning: "update threshold (0.7 at inference)" },
          { symbol: "D", meaning: "multi-scale dynamic template set refreshed from current prediction" },
        ],
        intuition: "Only trust good-looking frames to become the new memory of what the target looks like.",
        why: "Replaces STARK/MixFormer extra update branch with zero extra parameters while still progressing temporal context.",
        where: "Section 2.2, Eq. 6 and Algorithm 1, lines 10-11.",
        params: "Higher tau is safer but updates rarely; 0.7 is the reported operating point.",
        simulator: "track-mgmt",
        paperIds: ["T079"],
      },
    ],
    datasets: ["trackingnet", "got10k"],
    metrics: ["success-auc", "precision", "norm-precision", "fps"],
    baselines: ["SiamFC", "SiamRPN++", "SimTrack", "Stmtrack", "TransT", "STARK", "MixFormer", "OSTrack"],
    results: [
      "GOT-10k test (train-only protocol, MAE-1k): ProContEXT AO 74.6 / SR0.5 84.7 / SR0.75 72.9 vs OSTrack 73.7 / 83.2 / 70.8 (+0.9 / +1.5 / +2.1).",
      "TrackingNet (MAE-1k): ProContEXT AUC 84.6 / N-PRE 89.2 / PRE 83.8 vs OSTrack 83.9 / 88.5 / 83.2 (+0.7 / +0.7 / +0.6).",
      "TrackingNet/GOT-10k (CLS-1k): ProContEXT 83.4 / 88.3 / 82.2 and 72.0 / 82.7 / 68.0 vs MixFormer 82.6 / 87.7 / 81.2 and 71.2 / 79.9 / 65.8; STARK 82.0 and 68.8 AO.",
      "Speed: 54.3 FPS context encoding plus tracking despite expanded temporal/multi-spatial templates.",
      "GOT-10k val ablations (128/256, 100 epochs): two static scales [2.0,4.0] reach 86.7 AO vs 85.1 single scale; adding dynamic templates 85.1 to 86.3 AO (+1.2) and 94.6 to 96.0 SR0.5; both give 86.8 AO / 96.8 SR0.5.",
      "Token pruning val: rho=0.7 gives 86.8 AO / 96.8 SR0.5 at 38.0 GFLOPs (down 16.5% from 45.5), beating rho=1.0 (85.7 / 95.5).",
    ],
    ablations: [
      "Static scales (Table 2): [2.0,4.0] best (+1.6 AO, +1.9 SR); 3-4 scales add noise and score 85.4 AO.",
      "Dynamic templates (Table 3): +1.2 AO / +1.4 SR alone; multi-scale static plus dynamic +2.2 SR, showing complementarity.",
      "Keeping ratio (Table 4): rho 0.6/0.7/0.8/0.9 trade GFLOPs 36.1/38.0/40.1/42.7 for AO 85.8/86.8/86.3/86.3; 0.7 is the accuracy–cost sweet spot.",
      "Pretraining (Table 1): same ranking holds under CLS-1k, CLS-22k (83.2 AUC / 72.5 AO) and MAE-1k; MAE gives the headline numbers.",
    ],
    limitations: {
      authorStated: [
        "Future work is needed on more effective context-learning strategies and token-pruning schemes to reduce the impact of complex contexts.",
      ],
      evident: [
        "Evaluated only on TrackingNet and GOT-10k; no MOT/multimodal or long-term disappearance study shown.",
        "Update hinges on a fixed score threshold (0.7) and previous-frame search window, so a confident drift can pollute dynamic templates.",
      ],
    },
    assumptions: [
      "First-frame box given; target appears near its previous location (adjacent-region search).",
      "Score maximum is a reliable proxy for tracklet reliability justifying template refresh.",
    ],
    computation:
      "Inference 54.3 FPS; training batch 128, lr 1e-4, AdamW, 300 epochs on 4xA100; templates 192, search 384 (128/256 in ablations).",
    relations: [
      { to: "T009", type: "uses-as-baseline", note: "SiamFC cited as context-free Siamese baseline beaten in Table 1 (57.1 TrackingNet vs 83.4+)." },
      { to: "T028", type: "uses-as-baseline", note: "SiamRPN++ cited as deep Siamese baseline beaten in Table 1 (73.3 TrackingNet vs 83.4+)." },
      { to: "T055", type: "uses-as-baseline", note: "TransT transformer baseline beaten on TrackingNet (81.4) and GOT-10k (67.1 AO)." },
      { to: "T056", type: "uses-as-baseline", note: "STARK spatio-temporal baseline with extra update branch beaten (82.0 TrackingNet, 68.8 AO) without needing its branch." },
      { to: "T070", type: "uses-as-baseline", note: "MixFormer extra-branch updater beaten under CLS-1k/22k on both benchmarks." },
      { to: "T071", type: "improves", note: "Revises OSTrack one-stream backbone with template group, context attention and dual-group pruning (+0.9 AO)." },
      { to: "T025", type: "uses-as-baseline", note: "Evaluated under GOT-10k protocol (train split only for GOT numbers); ablations on GOT-10k val." },
    ],
    concepts: ["sot", "attention", "siamese", "appearance-features", "memory-network", "success-plot", "bounding-box"],
    impact:
      "Bridged classical spatial/temporal context tracking and context-free Transformers with a template-group plus pruning recipe reused by later multi-context trackers.",
  },
  {
    id: "T080",
    arxiv: "2211.09791",
    title: "MOTRv2: Bootstrapping End-to-End Multi-Object Tracking by Pretrained Object Detectors",
    shortTitle: "MOTRv2",
    year: 2022,
    authors: ["Yuang Zhang", "Tiancai Wang", "Xiangyu Zhang"],
    fileName: "2211.09791v2.pdf",
    task: "multi-object",
    tags: ["end-to-end-mot", "query-propagation", "anchor-query", "yolox-proposal", "deformable-detr", "tracking-by-detection"],
    difficulty: "advanced",
    summary:
      "MOTRv2 bootstraps end-to-end MOTR with a frozen YOLOX detector: YOLOX boxes become anchors whose sine-cosine encodings initialize proposal queries, replacing MOTR detect queries, while track queries propagate with previous-frame boxes as anchors. The decoder predicts only relative offsets, decoupling detection from association. It reaches 69.9 HOTA on DanceTrack (73.4 with extras), 43.6 mMOTA on BDD100K and 62.0 HOTA on MOT17.",
    problem:
      "End-to-end query trackers (MOTR, TrackFormer) associate well but detect poorly because one shared decoder must jointly learn detection and association, losing to tracking-by-detection on detection-heavy datasets and metrics. Directly matching track queries to detector boxes gives only marginal gains and breaks end-to-endness.",
    background: ["mot", "tracking-by-detection", "end-to-end-mot", "attention", "data-association", "hungarian", "cost-matrix"],
    previousWork: [
      {
        name: "MOTR (Deformable-DETR query propagation)",
        limitation:
          "Learnable detect queries plus joint detection–association training cause poor detection (e.g. 73.5 DetA DanceTrack, 32.3 mMOTA BDD100K) and require pdrop/pinsert tricks.",
        whyThisPaper:
          "Anchor-based proposal queries from YOLOX supply detection priors so MOTR learns mainly association: +9.3 AssA with joint CrowdHuman training, 54.2 to 69.9 HOTA on DanceTrack.",
      },
      {
        name: "Tracking-by-detection (ByteTrack, OC-SORT, FairMOT, CenterTrack)",
        limitation:
          "Strong detection plus IoU/ReID matching wins MOTA but association collapses on uniform-appearance diverse-motion data (OC-SORT 38.3 AssA DanceTrack).",
        whyThisPaper:
          "Query propagation with detector anchors keeps end-to-end association (59.0 AssA) while matching detector DetA (83.0 vs 80.3).",
      },
      {
        name: "Query MOT (TrackFormer, TransTrack, MeMOT)",
        limitation:
          "Propagate queries or match locations frame-to-frame but still learn detection from scratch, staying behind detector-based methods on DanceTrack/MOT17.",
        whyThisPaper:
          "Ten learnable anchors plus YOLOX anchors recall detector misses while 4D box (not center-point) propagation adds the width/height association cue.",
      },
    ],
    researchGap:
      "No method fed pretrained-detector proposals as anchor-queries into an end-to-end query tracker to remove the detection–association conflict.",
    contribution: [
      "Anchor formulation of MOTR queries: sine-cosine positional encoding of 4D anchors replaces learnable positional embeddings for both proposal and track queries.",
      "Proposal-query generation: broadcast shared learnable query plus score embedding of YOLOX boxes (score > 0.05) plus 10 learnable anchors to recall misses; boxes serve as decoder anchors.",
      "Proposal propagation: previous-frame track boxes concatenated with current YOLOX proposals as anchors; decoder predicts confidence plus relative offsets; self-attention exchanges proposal–track information for the same instance.",
      "Training/inference recipe: propagate queries scoring > 0.5 (creating natural FP/FN), CrowdHuman pseudo-video joint training, query denoising, and MOT17-specific track-query alignment (anchor/prediction/removal at IoU > 0.5).",
      "First-place DanceTrack challenge entry (73.4 HOTA with extra association, val training and 4-model ensemble) and BDD100K/MOT17 state of the art among query methods.",
    ],
    method: {
      pipeline: ["detect-yolox", "generate-proposal-queries", "concat-track-queries", "decode-offsets", "propagate", "align"],
      architecture:
        "YOLOX (ByteTrack/DanceTrack weights, or BDD100K-trained) proposes boxes; modified MOTR (ResNet-50 + Deformable-DETR decoder) consumes concatenated proposal and track queries with anchor sine-cosine PE, self-attention then deformable cross-attention to image features, outputting offsets and scores per frame.",
      motionModel:
        "Implicit: track queries carry identity and previous box anchor forward; no Kalman. MOT17 alignment optionally replaces anchors/predictions with matched YOLOX boxes.",
      appearanceModel:
        "Implicit in query embeddings and decoder image interaction with temporal aggregation; no explicit ReID head.",
      association:
        "End-to-end query identity propagation plus decoder self-attention dedup; tracklet-aware label assignment and collective average loss; optional post extra association linking single-exit/re-enter tracks within 20-100 frames (DanceTrack) and Hungarian IoU>0.5 alignment (MOT17/20).",
      reid:
        "None separate; association is query-embedding based rather than cosine ReID.",
      detectionDependency:
        "Fully detector-bound for proposals: all YOLOX boxes over 0.05 confidence; 10 learnable anchors cover detector misses; per-class shared queries for multi-class BDD100K.",
      trackManagement:
        "Frame 0: proposals only. Later: track queries (score>0.5 propagated) plus fresh proposals; unmatched MOTR predictions removable as likely FP; max gap handling via alignment/removal ablations.",
      loss:
        "MOTR collective average loss over clip frames plus query-denoising auxiliary (noise lambda1=0.1, lambda2=0.05 best); temporal aggregation network retained.",
      optimization:
        "8 GPUs, 1 clip/GPU; DanceTrack 5 epochs clip-5 stride 1-10 lr 2e-4 decay at epoch 4; MOT17 50 epochs decay at 40; BDD100K 2.5 epochs clip-4; HSV augmentation; CrowdHuman joint pseudo-clips length 5.",
    },
    equations: [
      {
        id: "motrv2-query",
        label: "Proposal query generation",
        formula: "qp = broadcast(qs, Nt) + PE_score(s); Q = [qtr ; qp]; anchors = [Yhat_{t-1} ; Pt]",
        variables: [
          { symbol: "qs", meaning: "shared learnable query embedding (1xD, per-class for BDD100K)" },
          { symbol: "s / Nt", meaning: "YOLOX confidence scores and proposal count (score>0.05)" },
          { symbol: "qtr / qp", meaning: "propagated track queries and fresh proposal queries" },
          { symbol: "Pt / Yhat", meaning: "YOLOX boxes plus 10 learnable anchors, and previous predictions" },
        ],
        intuition:
          "Stamp each detector box with how confident it is, then hand the pile plus existing tracks to the Transformer as its working set.",
        why:
          "Dynamic query count follows the detector so MOTR need not learn to detect from fixed queries; score embedding is shown necessary (63.0 to 63.7 HOTA).",
        where: "Section 3.4 and Figure 3; each frame, first frame uses proposals only.",
        paperIds: ["T080"],
      },
      {
        id: "motrv2-offset",
        label: "Anchor-relative prediction",
        formula: "PE = sine-cosine(x, y, w, h); Yhat_t = anchors + (dx, dy, dw, dh); offsets from deformable decoder",
        variables: [
          { symbol: "PE", meaning: "sine-cosine positional encoding of 4D anchor" },
          { symbol: "(dx,dy,dw,dh)", meaning: "decoder-predicted relative offsets" },
          { symbol: "Yhat_t", meaning: "final boxes (confidence plus offsets)" },
        ],
        intuition:
          "Start from the detector box and learn only the small correction instead of the absolute position.",
        why:
          "Anchor regression eases detection optimization and lets 4D size (not just center) aid association; box propagation beats point propagation (63.7 vs 60.5 HOTA).",
        where: "Sections 3.4-3.5; decoder output each frame.",
        params: "Learnable PE vs sine-cosine is a wash (63.8 vs 63.7); 4D vs point is decisive for AssA.",
        simulator: "motion",
        paperIds: ["T080"],
      },
      {
        id: "motrv2-interact",
        label: "Proposal–track self-attention",
        formula: "SelfAttn([qtr;qp]) couples same-instance proposal and track queries; deformable cross-attn grounds them in image features",
        variables: [
          { symbol: "SelfAttn", meaning: "decoder self-attention exchanging duplicate information" },
          { symbol: "cross-attn", meaning: "deformable attention to frame features with anchor reference points" },
        ],
        intuition:
          "Let the new detection candidate and the old track of the same person talk so they do not both claim it.",
        why:
          "Attention map (Figure 4) shows high same-instance similarity, giving dedup and letting tracks borrow detector localization.",
        where: "Section 3.5 analysis; N stacked decoders per frame.",
        paperIds: ["T080"],
      },
      {
        id: "motrv2-align",
        label: "Track-query alignment (MOT17/20)",
        formula: "IoU matrix MOTR x YOLOX -> Hungarian; keep pairs IoU>0.5; replace prediction and/or next anchor; drop unmatched MOTR boxes",
        variables: [
          { symbol: "IoU", meaning: "box overlap between MOTR prediction and YOLOX proposal" },
          { symbol: "0.5", meaning: "match-keeping threshold" },
          { symbol: "replace/drop", meaning: "three independent options: align prediction, align anchor, remove unmatched" },
        ],
        intuition:
          "Snap shaky end-to-end boxes back to the strong detector and throw away hallucinations with no detector support.",
        why:
          "Anchor alignment alone adds +8.4 MOTA / +3.9 IDF1 on MOT17 valhalf; all three reach 86.8 MOTA / 79.7 IDF1, curbing localization-error accumulation.",
        where: "Section 4.6 and Figure 5; applied in training and inference for MOT17/20 only.",
        simulator: "hungarian",
        paperIds: ["T080"],
      },
      {
        id: "motrv2-train",
        label: "Confidence-propagated training",
        formula: "propagate q with score>0.5 (not GT-matched); clip 5, stride 1-10; CrowdHuman pseudo-clips joint; query denoise (0.1, 0.05)",
        variables: [
          { symbol: "0.5", meaning: "propagation threshold creating natural FP/FN queries" },
          { symbol: "stride/clip", meaning: "temporal sampling (DanceTrack clip 5, BDD clip 4)" },
          { symbol: "denoise", meaning: "DN-DETR auxiliary with small noise range" },
        ],
        intuition:
          "Train on the messy tracks the model itself produces, plus still-image pseudo-videos for detection breadth.",
        why:
          "Removes MOTR pdrop/pinsert hand-tuning; YOLOX proposals turn CrowdHuman joint training from AssA-harmful (-5.6) into helpful (up to 63.7 HOTA).",
        where: "Section 4.2; DanceTrack/MOT17/BDD100K schedules differ.",
        simulator: "track-mgmt",
        paperIds: ["T080"],
      },
    ],
    datasets: ["dancetrack", "mot17", "mot20", "others"],
    metrics: ["hota", "deta", "assa", "mota", "idf1", "fps"],
    baselines: ["MOTR", "TrackFormer", "TransTrack", "ByteTrack", "OC-SORT", "FairMOT", "CenterTrack", "QDTrack", "GTR", "Unicorn"],
    results: [
      "DanceTrack test: MOTRv2 69.9 HOTA / 83.0 DetA / 59.0 AssA / 91.9 MOTA / 71.7 IDF1 vs OC-SORT 55.1 / 80.3 / 38.3 / 92.0 / 54.6 and MOTR 54.2 / 73.5 / 40.2 / 79.7 / 51.5 (+14.8 HOTA over prior best).",
      "DanceTrack challenge entry MOTRv2* (extra association + val training + 4-model ensemble): 73.4 HOTA / 83.7 DetA / 64.4 AssA / 92.1 MOTA / 76.0 IDF1, 1st place.",
      "BDD100K val: MOTRv2 43.6 mMOTA / 56.5 mIDF1 / 65.6 MOTA / 72.7 IDF1 vs Unicorn 41.2 / 54.0 / 66.6 / 71.3 and MOTR 32.3 / 44.8 / 56.2 / 65.8 (+8.1/+8.3 over MOTR*).",
      "MOT17 test: MOTRv2 62.0 HOTA / 60.6 AssA / 63.8 DetA / 75.0 IDF1 / 78.6 MOTA vs MOTR 57.8 / 55.7 / 60.3 / 68.6 / 73.4; still below ByteTrack 63.1 / 80.3 and BoT-SORT 64.6 HOTA.",
      "MOT20 test: MOTRv2 60.3 HOTA / 58.1 AssA / 62.9 DetA / 72.2 IDF1 / 76.2 MOTA; +MOT17 joint 61.0 / 59.3 / 63.0 / 73.1 / 76.2.",
      "Ablations (DanceTrack val): implementation 51.7 to 54.8, +box propagation 57.1, +YOLOX&CrowdHuman 63.7, +query denoise 64.5 HOTA; score embedding sine-cosine 63.7 vs none 63.0.",
    ],
    ablations: [
      "YOLOX vs CrowdHuman (Table 6): YOLOX alone 60.7 HOTA beats CrowdHuman alone 56.7; CrowdHuman alone drops AssA 49.5 to 43.9; together 63.7.",
      "Box vs point propagation (Table 7): 4D boxes 63.7-63.8 HOTA / 53.1-53.2 AssA vs points 60.5-61.2 / 48.0-49.2; PE type barely matters.",
      "Query denoise (Table 9): default noise (0.4,0.4) hurts AssA (51.5); small noise (0.1,0.05) gives 64.5 HOTA / 78.7 DetA.",
      "Alignment (Table 10 MOT17 valhalf): anchor align most beneficial; prediction+anchor+removal reach 86.8 MOTA / 79.7 IDF1.",
    ],
    limitations: {
      authorStated: [
        "Still data-hungry and underperforms on small datasets such as MOT17, attributed to insufficient real video scale for query training.",
        "Duplicate track queries occur when individuals cross: one query can follow the wrong subject leaving two queries on one person.",
        "Inefficient: YOLOX 25 FPS plus MOTR 9.5 FPS yields 6.9 FPS on 2080Ti, MOTR being the bottleneck.",
      ],
      evident: [
        "Best DanceTrack number needs extra association, val-set training and ensemble beyond the clean 69.9 model.",
        "MOT17/20 alignment leans on detector boxes, partially departing from pure end-to-end propagation at inference.",
      ],
    },
    assumptions: [
      "A strong pretrained detector (YOLOX) exists for the domain and its boxes over 0.05 are high-recall proposals.",
      "Relative-offset regression from anchors is easier to optimize than absolute detection plus association jointly.",
    ],
    computation:
      "ResNet-50 MOTR on 8 GPUs; YOLOX 25 FPS, MOTR 9.5 FPS, combined 6.9 FPS on 2080Ti; 4-model ensemble for challenge entry.",
    relations: [
      { to: "T059", type: "builds-on", note: "Directly modifies MOTR: anchor queries replace detect queries, propagation kept (+15.7 DanceTrack HOTA)." },
      { to: "T054", type: "uses-as-baseline", note: "TrackFormer query-propagation competitor beaten on MOT17 (65.0 vs 78.6 MOTA)." },
      { to: "T053", type: "uses-as-baseline", note: "TransTrack beaten on DanceTrack (45.5 HOTA) and MOT17 (54.1 HOTA)." },
      { to: "T061", type: "uses-as-baseline", note: "ByteTrack YOLOX weights reused for proposals and beaten on DanceTrack AssA (32.1 vs 59.0)." },
      { to: "T063", type: "uses-as-baseline", note: "Evaluated on DanceTrack benchmark; 1st-place challenge entry (73.4 HOTA)." },
      { to: "T006", type: "uses-as-baseline", note: "Evaluated on MOT17/MOT20 MOTChallenge protocol against the leaderboard." },
    ],
    concepts: ["mot", "end-to-end-mot", "tracking-by-detection", "data-association", "hungarian", "cost-matrix", "hota", "mota"],
    impact:
      "Showed end-to-end query tracking can inherit detector strength via anchor-queries, resetting the DanceTrack/BDD100K state of the art and framing detection–association decoupling for later MOTRv3-style work.",
  },
  {
    id: "T081",
    arxiv: "2301.01482",
    title: "Underwater Object Tracker: UOSTrack for Marine Organism Grasping of Underwater Vehicles",
    shortTitle: "UOSTrack",
    year: 2023,
    authors: ["YunFeng Li", "Bo Wang", "Ye Li", "Wei Huo", "ZhuoYan Liu", "Yueming Li", "Jian Cao"],
    fileName: "2301.01482v5.pdf",
    task: "single-object",
    tags: ["underwater-tracking", "sample-imbalance", "hybrid-training", "detection-to-tracking", "kalman-filter", "post-processing", "marine-robotics"],
    difficulty: "intermediate",
    summary:
      "UOSTrack adapts OSTrack to underwater grasping with two training-free-at-test ideas: UOHT hybrid training converts RUOD/FishExtend detection images into tracking pairs (with gray/flip/noise/blur/rotation) and rebalances open-air:underwater from 99:1 to 2:1; MBPP adds a SORT-style Kalman estimator plus candidate-box location scores f*IoU to re-catch drifted targets. It leads UOT100 (68.24 AUC) and UTB180 (66.71 AUC) and runs 110 FPS on Jetson AGX Xavier in Dalian field grasps.",
    problem:
      "Underwater trackers inherit open-air training (LaSOT 16/1400 and GOT-10k 284/9335 underwater sequences) so underwater losses are drowned, and schooling fish/dolphins create similar-object interference that Hanning windows and stronger backbones do not fix. Grasping UVs need stable accuracy under degradation, imbalance and aggregation.",
    background: ["sot", "bounding-box", "iou", "attention", "appearance-features", "motion-model", "kalman", "success-plot"],
    previousWork: [
      {
        name: "OSTrack one-stream Transformer",
        limitation:
          "Trained 99:1 open-air biased, it attends to boundaries/any dolphin-like region and drifts to similar objects (UOT similar-subset 58.81 AUC).",
        whyThisPaper:
          "UOHT teaches underwater feature expression and MBPP re-locates the true target, lifting similar-subset AUC by 3.47 (UOT100) and 6.27 (UTB180).",
      },
      {
        name: "Underwater enhancers (CRN-UIE, UStark, IEFF-ECO)",
        limitation:
          "Improve image quality before tracking rather than the tracker data bias or drift logic, leaving aggregation behavior unaddressed.",
        whyThisPaper:
          "UOHT makes the tracker itself domain-adapted and MBPP adds motion matching, beating UStark 67.80 to 68.24 on UOT100.",
      },
      {
        name: "Similar-object handlers (Hanning window, KeepTrack, Stark/MixFormer discrimination)",
        limitation:
          "Hanning fails at search-center interference; KeepTrack-style association is costly and hard to port, needing retraining and tuning.",
        whyThisPaper:
          "MBPP needs no training and two parameters (n, conf=0.6), porting to OSTrack/Stark/TransT for +1.0 to +6.0 similar-object AUC.",
      },
    ],
    researchGap:
      "No work fixed underwater sample imbalance at the pair-construction level and similar-object drift at the motion-matching level together.",
    contribution: [
      "UOHT paradigm: average-sample LaSOT/GOT-10k/TrackingNet/COCO plus RUOD and FishExtend detection images turned into template–search pairs by per-op random augmentation (gray 0.1, flip 0.15, noise/blur/rotation 0.05, rotation 0-10 deg).",
      "Sample-ratio rebalance from ~99:1 to ~2:1 open-air:underwater with Grad-CAM evidence of whole-object vs boundary attention.",
      "MBPP paradigm: top-n response candidates with NMS plus SORT-state Kalman estimation box and location score f(z,ci)*IoU; IoU(bmax,bE)<0.6 triggers motion re-match.",
      "New UOT100 similar-subset split (28 sequences) and full 14+15 tracker comparisons plus 30-scenario Dalian field and grasping validation.",
      "SOTA on UOT100/UTB180 with +8 ms MBPP cost and 110 FPS deployment for IBVS manipulator grasping (overlap >= 0.8 in demo).",
    ],
    method: {
      pipeline: ["build-pairs", "hybrid-train", "embed-relate", "detect-max", "predict-kalman", "score-candidates", "rematch"],
      architecture:
        "OSTrack backbone unchanged: ViT joint feature extraction and relation modeling plus Center-head (Stark ported to Center-head for MBPP tests); detection-based post-processing replaced/augmented by MBPP at test.",
      motionModel:
        "SORT-state constant-velocity Kalman: X=[u,v,s,r,u',v',s'] with state/observation Eqs. 4-5, time update 7-8, measurement update 10-12, box decode Eq. 9; initialized frame 1, updated per frame.",
      appearanceModel:
        "Learned underwater expression via UOHT: joint relational matching phi(z,x) plus discriminative underwater features; no online template update.",
      association:
        "Single-target motion re-match, not MOT association: if IoU(max-response box, estimation box) < conf (0.6), score n candidates by f(z,ci)*IoU(bi,bE) and output argmax.",
      detectionDependency:
        "Detection datasets (RUOD, FishExtend) used only as pair sources for training, not as test detectors.",
      trackManagement:
        "Extra Kalman track paralleling the SOT output; candidate set Ct top-n (n=40 UOT100, 30 UTB180) filtered by NMS; drift correction only on low-agreement frames.",
      loss:
        "Ltotal = Lcls + 2*Liou + 5*L1; weight focal classification plus L1 and generalized-IoU box regression, same weights as OSTrack/CenterNet family.",
      optimization:
        "300 epochs on 2xRTX A6000, batch 32, lr 0.004 decaying to 0.0004 at epoch 240, AdamW; Stark port fine-tuned 50 epochs.",
    },
    equations: [
      {
        id: "uostrack-loss",
        label: "Hybrid training total loss",
        formula: "Ltotal = Lcls + 2*Liou + 5*L1",
        variables: [
          { symbol: "Lcls", meaning: "weight focal classification loss" },
          { symbol: "Liou / L1", meaning: "generalized-IoU and L1 box regression losses" },
        ],
        intuition: "Classify foreground while forcing tight, well-overlapping boxes.",
        why: "Joint objective keeps UOHT pairs (including synthetic detection pairs) geometrically honest.",
        where: "Section 4.1.3, Eq. 1.",
        paperIds: ["T081"],
      },
      {
        id: "uostrack-sim",
        label: "Template–search similarity",
        formula: "f(z, xi) = phi(z, x); search split into 16x16 (256) patches, one target xt plus candidates",
        variables: [
          { symbol: "z / x", meaning: "template and search area" },
          { symbol: "phi", meaning: "ViT joint extraction and relation modeling (OSTrack)" },
          { symbol: "f", meaning: "per-patch similarity score" },
        ],
        intuition: "Everything looking like the template lights up; the brightest is the naive answer.",
        why: "Defines the candidate set that both causes drift and enables correction.",
        where: "Section 4.2.1, Eq. 2.",
        simulator: "siamese",
        paperIds: ["T081"],
      },
      {
        id: "uostrack-nms",
        label: "Candidate set with NMS",
        formula: "Ct = {ci: f(z,ci) > hn (Nth score)}; (c', b) = NMS(Ct, B); final boxes (b1..bn), n<l",
        variables: [
          { symbol: "hn", meaning: "Nth-largest similarity threshold (n=40/30)" },
          { symbol: "B", meaning: "raw candidate boxes from response map" },
          { symbol: "NMS", meaning: "non-maximum suppression dedup" },
        ],
        intuition: "Keep the high-scoring lookalikes, merge overlapping duplicates, and hold them as rescue options.",
        why: "True target persists in the map even when not maximal; NMS yields clean per-instance boxes (Figure 9).",
        where: "Section 4.2.1, Eq. 3.",
        simulator: "track-mgmt",
        paperIds: ["T081"],
      },
      {
        id: "uostrack-kalman",
        label: "Kalman predict–update",
        formula: "Xk = A X_{k-1}+B mu+W; Zk = H Xk+V; Xhat- = A Xhat+B mu; P- = A P A^T+Q; K = P- H^T(H P- H^T+R)^{-1}; Xhat = Xhat-+K(Z-H Xhat-); P = (I-KH)P-",
        variables: [
          { symbol: "X = [u,v,s,r,u',v',s']", meaning: "center, area, aspect plus velocities (SORT state)" },
          { symbol: "Z / H", meaning: "observation vector and observation matrix" },
          { symbol: "Q/R/W/V", meaning: "Gaussian process/observation noises and covariances" },
          { symbol: "K", meaning: "Kalman gain" },
        ],
        intuition: "Extrapolate the trajectory with constant velocity, then correct it with the observed box weighted by uncertainty.",
        why: "Provides a motion prior independent of appearance to judge whether the max-response box is plausible.",
        where: "Section 4.2.2, Eqs. 4-8 and 10-12.",
        simulator: "kalman",
        paperIds: ["T081"],
      },
      {
        id: "uostrack-decode",
        label: "Estimation box decode",
        formula: "bE = [uhat-, vhat-, sqrt(shat-/rhat-), sqrt(shat-*rhat-)] from Xhat-",
        variables: [
          { symbol: "bE", meaning: "Kalman-predicted estimation box" },
          { symbol: "uhat-/vhat-/shat-/rhat-", meaning: "priori center, area, aspect estimates" },
        ],
        intuition: "Convert center-plus-area state back into width/height box corners the tracker can compare.",
        why: "Bridges filter state space and image box space for IoU gating.",
        where: "Section 4.2.2, Eq. 9.",
        simulator: "motion",
        paperIds: ["T081"],
      },
      {
        id: "uostrack-score",
        label: "Motion-appearance location score",
        formula: "score_i = f(z, ci) x IoU(bi, bE); output argmax; trigger if IoU(bmax, bE) < 0.6",
        variables: [
          { symbol: "f(z,ci)", meaning: "appearance similarity of candidate i" },
          { symbol: "IoU(bi,bE)", meaning: "motion agreement of candidate box with estimation" },
          { symbol: "0.6", meaning: "conf gating threshold for motion re-match" },
        ],
        intuition: "Pick the lookalike that is also where physics says the target should be.",
        why: "Product fuses appearance and motion so center-region distractors scoring higher than the target get overruled (Tables 4-5, Figure 13).",
        where: "Sections 4.2.3-4.2.4, Eq. 13 and Figure 10.",
        simulator: "iou-track",
        paperIds: ["T081"],
      },
    ],
    datasets: ["lasot", "got10k", "trackingnet", "others"],
    metrics: ["success-auc", "precision", "norm-precision", "fps"],
    baselines: ["OSTrack", "MixFormer", "AiATrack", "ToMP", "KeepTrack", "Stark", "TransT", "TrDiMP", "SiamBAN", "DiMP", "ATOM", "SiamRPN++", "NeighborTrack", "UStark"],
    results: [
      "UOT100: UOSTrack 68.24 AUC / 63.15 P / 85.62 P-Norm vs OSTrack 66.88 / 62.11 / 84.55 (+1.36/+1.04/+1.07) and NeighborTrack 67.32 / 62.56 / 85.23; UStark 67.80 / 59.80.",
      "UOT100 similar subset (28 seqs): 62.28 / 58.22 / 76.38 vs OSTrack 58.81 / 54.33 / 72.08 (+3.47/+3.89/+4.30).",
      "UTB180: 66.71 / 60.25 / 77.02 vs OSTrack 63.03 / 56.52 / 72.61 (+3.08/+3.73/+4.41) and NeighborTrack 64.53 / 58.10 / 74.16.",
      "UTB180 similar subset: 61.05 / 56.14 / 70.04 vs OSTrack 54.78 / 50.29 / 62.50 (+6.27/+5.82/+7.54).",
      "MBPP alone on OSTrack: UOT100 66.88 to 68.40 AUC (+1.52), similar 58.81 to 63.27 (+4.46); UTB180 63.03 to 64.13, similar 54.78 to 57.70; +8.04 ms vs NeighborTrack +21.14 ms.",
      "Field 30 sequences (10291 frames, Jetson AGX Xavier, 110 FPS): UOSTrack 50.21 AUC / 55.40 P / 58.70 P-Norm vs OSTrack 42.26 / 47.42 / 47.69 (+7.95/+7.98/+11.01); grasp demo minimum overlap 0.8.",
    ],
    ablations: [
      "UOHT data (Table 7): adding FishExtend or RUOD each lifts both benchmarks; full UOHT without MBPP reaches 67.80/65.84 AUC; MBPP alone reaches 68.40/64.13; both give 68.24/66.71.",
      "MBPP generalization (Fig. 12): similar-object AUC gains 4.5 OSTrack, 6.0 Starks, 4.3 Starkst, 1.0 TransT on UOT100 (2.9/9.7/8.1/2.0 on UTB180); also helps occlusion, low-resolution, fast motion.",
      "MBPP vs NeighborTrack/Alpha-Refine (Tables 4-5): MBPP halves extra time and stacks with both (OSTrack+N+MBPP 68.52 UOT100, 65.33 UTB180).",
      "Augmentation for pairs: gray 0.1, flip 0.15, noise/blur/rotation 0.05, rotation 0-10 deg; sampling rebalanced 99:1 to 2:1 (Fig. 4).",
    ],
    limitations: {
      authorStated: [
        "Cannot discriminate highly similar closed targets at instance level: neighboring boxes merge into one large box (frames 113/139), contaminating the Kalman filter and MBPP.",
        "Real-sea success 50.2% and normalized precision 58.7% show further research is still needed for marine deployment; deformation and low-resolution remain unsatisfactory though more robust.",
      ],
      evident: [
        "Field/ablation gains rest on the authors' 30-sequence annotations and fixed n=40/30, conf=0.6 without sensitivity bands shown.",
        "Synthetic detection pairs cannot model target motion/deformation alone, which is why open-air sequences must be retained.",
      ],
    },
    assumptions: [
      "Single underwater target specified first frame; detection images can stand in for underwater appearance if motion comes from open-air sequences.",
      "Constant-velocity SORT motion plus 0.6 IoU gate approximates marine motion between frames.",
    ],
    computation:
      "Train 300 epochs on 2xRTX A6000 batch 32 AdamW; test on i9/3090Ti; MBPP +8.04 ms; deployment 110 FPS at 128 template / 256 search on Jetson AGX Xavier.",
    relations: [
      { to: "T071", type: "builds-on", note: "Directly extends OSTrack one-stream backbone and training paradigm with UOHT/MBPP." },
      { to: "T009", type: "uses-as-baseline", note: "SiamFC lineage cited for Hanning-window drift analysis." },
      { to: "T028", type: "uses-as-baseline", note: "SiamRPN++ compared (54.00 UOT100) and its deeper-feature remedy discussed." },
      { to: "T031", type: "uses-as-baseline", note: "DiMP discriminative-prediction baseline compared (59.82/50.52 AUC)." },
      { to: "T055", type: "uses-as-baseline", note: "TransT compared (63.75/57.52) and MBPP-ported (+1.0/+2.0 similar AUC)." },
      { to: "T056", type: "uses-as-baseline", note: "Stark compared (66.33/55.86) and MBPP-ported in both S/T variants." },
      { to: "T070", type: "uses-as-baseline", note: "MixFormer compared (66.20/57.44) as hybrid-attention competitor." },
      { to: "T024", type: "builds-on", note: "LaSOT retained in UOHT for relation modeling (only 16 underwater of 1400 sequences noted)." },
      { to: "T025", type: "builds-on", note: "GOT-10k retained in UOHT (284 underwater of 9335 sequences noted)." },
    ],
    concepts: ["sot", "appearance-features", "motion-model", "kalman", "data-association", "iou", "success-plot", "bounding-box"],
    impact:
      "Gave underwater tracking a data-centric fix (detection-to-tracking pairs) plus a portable training-free motion rescue now testable on any response-map tracker, validated through to UV grasping.",
  },
  {
    id: "T082",
    arxiv: "2303.10404",
    title: "MotionTrack: Learning Robust Short-term and Long-term Motions for Multi-Object Tracking",
    shortTitle: "MotionTrack",
    year: 2023,
    authors: ["Zheng Qin", "Sanping Zhou", "Le Wang", "Jinghai Duan", "Gang Hua", "Wei Tang"],
    fileName: "2303.10404v2.pdf",
    task: "multi-object",
    tags: ["tracking-by-detection", "motion-model", "interaction-module", "graph-convolution", "refind-module", "long-term-occlusion", "yolox"],
    difficulty: "advanced",
    summary:
      "MotionTrack is a YOLOX-based tracking-by-detection MOT that learns motion only: an Interaction Module builds a directed asymmetric attention/convolution adjacency between tracklets and GCN-predicts next offsets for short-range IoU association, and a Refind Module correlates 30-frame history trajectories with unmatched detections then linearly compensates the occluded gap. It tops MOT17 (80.1 IDF1, 81.1 MOTA, 65.1 HOTA) and MOT20 (76.5 IDF1, 78.0 MOTA) with the fewest ID switches.",
    problem:
      "Dense crowds make boxes too small for appearance and force collision-avoiding complex motion, breaking short-range matching; long occlusions break appearance (pose/resolution/illumination change) while memory banks are heavy and slow. Prior motion models assume independent targets and miss explicit inter-target interaction.",
    background: ["mot", "tracking-by-detection", "motion-model", "kalman", "data-association", "hungarian", "cost-matrix", "iou"],
    previousWork: [
      {
        name: "SORT/DeepSORT/ByteTrack association chain",
        limitation:
          "Independent constant-velocity Kalman plus appearance/IoU matching cannot predict avoidance maneuvers and degrades sharply under 30-to-120-frame gaps.",
        whyThisPaper:
          "Interaction-aware GCN prediction plus history-trajectory correlation keeps improving to 120 frames while IoU drops 0.8 and ReID drops 6.8 IDF1 (Table 4).",
      },
      {
        name: "Appearance ReID and memory banks (DeepSORT, MeMOT, MTrack)",
        limitation:
          "Appearance collapses under occlusion/pose change and multi-query memory costs time and storage unfriendly to real-time tracking.",
        whyThisPaper:
          "Pure-motion Refind with correlation plus error compensation needs no ReID yet cuts IDs 40% vs P3AFormer on MOT17 (1140 vs 1893).",
      },
      {
        name: "Model-based regressors (Tracktor, CenterTrack, ArTIST, FFT)",
        limitation:
          "Regress displacement or pool interactions to one feature without explicit directed target–target influence, failing dense-crowd avoidance.",
        whyThisPaper:
          "Asymmetric attention plus 1xkappa/kappax1 cascade with 0.6 gating explicitly models who influences whom, raising prediction IoU over Kalman (Figure 5).",
      },
    ],
    researchGap:
      "No online TBD tracker learned directed inter-target short motion and history-based long motion jointly without appearance.",
    contribution: [
      "Interaction Module: self-attention attention matrix plus L-layer asymmetric-convolution cascade, sigmoid masking at xi=0.6, normalization, then GCN+MLP offset prediction for IoU association.",
      "Refind Module: 30-frame history encoder plus detection-difference encoder into correlation matrix Ccorr in [0,1]^{SxU} with greedy matching (keep >= 0.9) and linear error compensation across (t1,t2).",
      "Unified TBD loop with ByteTrack double matching (high 0.6 / low 0.1, init 0.7, IoU reject < 0.2) plus global motion compensation; alive/lost/dead states with 60/120-frame memory.",
      "IoU-squared interaction loss plus binary-cross-entropy correlation loss trained on 3-frame clips and sampled trajectory pairs.",
      "SOTA on MOT17/MOT20 with strongest association metrics and new crowdMOTA (visibility<0.25) diagnosis showing +0.8 to +1.6 gains.",
    ],
    method: {
      pipeline: ["detect", "extract-interaction", "predict-offsets", "associate-short", "correlate-long", "compensate", "manage"],
      architecture:
        "YOLOX-public (ByteTrack-trained) detections Dt=(x,y,w,h); Interaction Module input It in R^{Mx8} (absolute plus t-2 to t-1 offsets); Refind inputs Tlost in R^{Sx30x5} and Drest in R^{Ux5} with (t,x,y,w,h).",
      motionModel:
        "Learned interaction-aware short prediction replacing Kalman baseline, plus linear occluded-gap compensation dtp = ptp + (dt-pt)(tp-t1)/(t2-t1); lost tracks iteratively predicted during occlusion.",
      appearanceModel: "None — deliberately appearance-free; association uses geometry and trajectory history only.",
      association:
        "Short: IoU between GCN predictions and Dt with ByteTrack two-stage matching. Long: greedy on Ccorr>=0.9 linking S lost to U unmatched; remainder birth new tracklets.",
      detectionDependency:
        "Entirely YOLOX-bound (public ByteTrack weights, no retraining); high/low thresholds 0.6/0.1, init 0.7.",
      trackManagement:
        "Alive/lost/dead; initialize from D1; update alive on match, predict lost, terminate after 60 (MOT17) / 120 (MOT20) frames; GMC applied.",
      loss:
        "L_INTR = 1 - IoU(Pcoor,Pgt)^2 for interaction; L_CORR = binary cross-entropy over sampled positive/negative trajectory-detection pairs.",
      optimization:
        "PyTorch on single RTX 3090 Ti; Interaction and Refind trained on half MOT17/MOT20 train; 3-frame clip samples for interaction.",
    },
    equations: [
      {
        id: "motiontrack-attn",
        label: "Interaction attention matrix",
        formula: "E = phi(I,W^E); Q = phi(E,W^Q); K = phi(E,W^K); A^atte = Softmax(Q K^T / sqrt(d))",
        variables: [
          { symbol: "I in R^{Mx8}", meaning: "concatenated absolute coords and t-2 to t-1 offsets for M tracklets" },
          { symbol: "Q/K/E", meaning: "query/key/higher-dim embedding (D-dim)" },
          { symbol: "A^atte", meaning: "MxM asymmetric influence (i on j)" },
          { symbol: "d", meaning: "sqrt(D) scaling factor" },
        ],
        intuition: "Ask every pedestrian how much every other pedestrian affects its next step, with direction (i affects j need not equal j affects i).",
        why: "Replaces symmetric distance with directed social influence, the prerequisite for avoidance prediction.",
        where: "Section 3.3, Eq. 1; interaction extraction per frame.",
        paperIds: ["T082"],
      },
      {
        id: "motiontrack-adj",
        label: "Asymmetric cascade and mask",
        formula: "A_l = PReLU(conv(A_{l-1},K_{1xK}) + conv(A_{l-1},K_{Kx1})); A^mask = sgn(sigmoid(phi(A_L)) - 0.6); A^adjc = A^mask * A^atte, normalized",
        variables: [
          { symbol: "K_{1xK}/K_{Kx1}", meaning: "asymmetric convolution kernels modeling group behavior" },
          { symbol: "0.6", meaning: "sign-function retention threshold" },
          { symbol: "A^adjc", meaning: "sparse directed adjacency in [0,1]" },
        ],
        intuition: "Diffuse influence through the crowd with cross-shaped filters, then keep only strong ties.",
        why: "Higher-order group interaction plus denoising; retains the attention magnitude only where it matters.",
        where: "Section 3.3, Eqs. 2-3; L cascade layers.",
        params: "Threshold 0.6 controls sparsity; larger kappa widens social receptive field.",
        paperIds: ["T082"],
      },
      {
        id: "motiontrack-gcn",
        label: "GCN motion prediction",
        formula: "P^offs = MLP(PReLU(phi(A^adjc O^t, W^G))); associate P^offs-shifted boxes by IoU",
        variables: [
          { symbol: "O^t in R^{Mx4}", meaning: "iterative offsets (dx,dy,dw,dh) from t-2 to t-1" },
          { symbol: "W^G", meaning: "graph-convolution linear weights" },
          { symbol: "P^offs", meaning: "predicted offsets to frame t" },
        ],
        intuition: "Blend each target's last move with its neighbors' weighted moves, then read out where it goes next.",
        why: "Fused prediction beats Kalman average-IoU and lifts IDF1/AssA (Table 3, Figure 5), rescuing short-occluded dense targets.",
        where: "Section 3.3, Eq. 4; short-range association.",
        simulator: "motion",
        paperIds: ["T082"],
      },
      {
        id: "motiontrack-corr",
        label: "History–detection correlation",
        formula: "T_l = PReLU(conv(T_{l-1},K_{Kx1})); F^traj = pool(PReLU(conv(T_L,K_{1xK}))); F^dete = phi(Dhat^rest,W^D); Ccorr = sigmoid(FC([F^traj;F^dete]))",
        variables: [
          { symbol: "Tlost in R^{Sx30x5}", meaning: "last 30 alive (t,x,y,w,h) per lost tracklet, normalized" },
          { symbol: "Dhat^rest in R^{(SxU)x10}", meaning: "unmatched detection plus last-alive difference, concatenated" },
          { symbol: "Ccorr in [0,1]^{SxU}", meaning: "lost-detection association probabilities" },
        ],
        intuition: "Compare the shape and timing of an old trail with each orphan detection; high score means same person reappearing.",
        why: "Spatial-distribution plus velocity-time pattern survives appearance change where ReID collapses at 120 frames.",
        where: "Section 3.4, Eqs. 5-6 and Figure 4; greedy keep >= 0.9.",
        params: "30-frame history; reject pairs below 0.9 correlation.",
        simulator: "assoc-cost",
        paperIds: ["T082"],
      },
      {
        id: "motiontrack-comp",
        label: "Occlusion error compensation",
        formula: "d^{tp} = p^{tp} + (d^t - p^t)(tp-t1)/(t2-t1), t1 < tp < t2",
        variables: [
          { symbol: "t1/t2", meaning: "lost moment and refound moment" },
          { symbol: "p / d", meaning: "iteratively predicted vs matched-detection boxes" },
          { symbol: "tp", meaning: "intermediate occluded frame to fill" },
        ],
        intuition: "Spread the endpoint error linearly back across the gap instead of inventing a new path.",
        why: "Corrects rather than interpolates the interacted prediction, preserving identity continuity (Figure 7).",
        where: "Section 3.4, Eq. 7; after greedy long match.",
        simulator: "motion",
        paperIds: ["T082"],
      },
      {
        id: "motiontrack-loss",
        label: "Interaction and correlation losses",
        formula: "L^INTR = 1 - IoU(P^coor, P^gt)^2; L^CORR = -1/n sum[y log c + (1-y) log(1-c)]",
        variables: [
          { symbol: "P^coor/P^gt", meaning: "predicted and supervisory third-frame locations (x,y,w,h jointly)" },
          { symbol: "c / y", meaning: "predicted correlation score and same-trajectory label (1/0)" },
        ],
        intuition: "Reward overlapping predicted boxes as whole shapes and reward correct same-or-not verdicts.",
        why: "IoU-squared couples correlated box variables like detector training; BCE trains the refind verdict on sampled pairs.",
        where: "Section 3.5, Eqs. 8-9; 3-frame clips and random trajectory couples.",
        simulator: "iou-track",
        paperIds: ["T082"],
      },
    ],
    datasets: ["mot17", "mot20"],
    metrics: ["mota", "idf1", "hota", "deta", "assa", "fp", "fn", "idsw", "fragments"],
    baselines: ["SORT", "DeepSORT", "ByteTrack", "FairMOT", "CenterTrack", "TrackFormer", "TransTrack", "MOTR", "MeMOT", "P3AFormer", "RelationTrack", "CorrTracker"],
    results: [
      "MOT17 test (private detector): MotionTrack 80.1 IDF1 / 81.1 MOTA / 65.1 HOTA / 65.1 AssA / 65.4 DetA / 1140 IDs / 1605 Frag vs ByteTrack 77.3 / 80.3 / 63.1 / 62.0 / 64.5 / 2196 / 2277 and P3AFormer 78.1 / 81.2 / 1893 IDs.",
      "MOT20 test: 76.5 IDF1 / 78.0 MOTA / 62.8 HOTA / 61.8 AssA / 64.0 DetA / 1165 IDs / 1321 Frag vs ByteTrack 75.2 / 77.8 / 61.3 / 59.6 / 63.4 / 1223 and P3AFormer 76.4 / 78.1 / 1332 (+1.3/+0.2/+1.5 core).",
      "Component val (MOT17): baseline 82.6/80.4/70.2/72.4/68.7/402 IDs; +Interaction 83.0/80.5/70.5/72.9; +Refind 83.7/80.7/70.8/73.5/378.",
      "Long occlusion (30 to 120 frames): ours 82.6 to 83.3 IDF1 (+0.7) while IoU-based 80.9 to 80.1 (-0.8) and ReID 77.2 to 70.4 (-6.8); MOTA -2.2 and -9.5 for competitors vs +0.3 ours.",
      "CrowdMOTA (visibility<0.25, 20-100 frames): +1.1/+1.5/+1.6/+0.8/+0.9 over baseline across difficulty levels.",
    ],
    ablations: [
      "Interaction vs Kalman (Fig. 5): higher mean prediction IoU plus steady IDF1/AssA gains confirm prediction quality drives association.",
      "Refind advantage (Table 4): only motion-history method improves at 120-frame gaps; ReID MOTA collapses 77.0 to 67.5.",
      "Crowd/occlusion split (Table 5): gains persist at all minimum-length cutoffs, isolating dense-crowd and extreme-occlusion effects from full-dataset averaging.",
      "Thresholds: adjacency gate 0.6, correlation keep 0.9, IoU reject 0.2, high/low 0.6/0.1, init 0.7; lost memory 60 MOT17 / 120 MOT20.",
    ],
    limitations: {
      authorStated: [
        "Only pedestrian motion patterns and inter-pedestrian relations are modeled; drivable prior information for motion prediction is ignored, which may weaken prediction.",
        "Interaction and Refind modules are currently separate and could be further combined to support each other.",
      ],
      evident: [
        "Appearance-free by design, so uniform-motion crowds with ambiguous geometry have no fallback cue shown.",
        "Uses public YOLOX weights and half-split training only, so detector and data scale cap the reported numbers.",
      ],
    },
    assumptions: [
      "YOLOX boxes plus 8-dim (position+offset) inputs suffice; targets interact in image space without camera-motion-corrected world coordinates beyond GMC.",
      "Thirty-frame history plus linear gap compensation approximates long occluded motion.",
    ],
    computation:
      "Single RTX 3090 Ti implementation; training on half MOT17/MOT20; thresholds as above; GMC on; no ReID/memory-bank cost.",
    relations: [
      { to: "T061", type: "builds-on", note: "Follows ByteTrack TBD, double matching and public YOLOX weights; beats it 80.1 vs 77.3 IDF1 MOT17." },
      { to: "T059", type: "uses-as-baseline", note: "MOTR end-to-end competitor beaten (68.6 vs 80.1 IDF1 MOT17)." },
      { to: "T054", type: "uses-as-baseline", note: "TrackFormer beaten on MOT17 (68.0 vs 80.1 IDF1)." },
      { to: "T053", type: "uses-as-baseline", note: "TransTrack cited as query-propagation alternative; association still location-Hungarian." },
      { to: "T006", type: "uses-as-baseline", note: "Evaluated under MOT17/MOT20 MOTChallenge private-detector protocol with CLEAR/IDF1/HOTA." },
    ],
    concepts: ["mot", "tracking-by-detection", "motion-model", "data-association", "cost-matrix", "hungarian", "mota", "hota", "idf1"],
    impact:
      "Established that explicit directed social interaction plus history-only refind can beat appearance-heavy MOT without ReID, with crowdMOTA as a reusable dense-crowd diagnostic.",
  },
  {
    id: "T083",
    arxiv: "2303.10826",
    title: "Visual Prompt Multi-Modal Tracking",
    shortTitle: "ViPT",
    year: 2023,
    authors: ["Jiawen Zhu", "Simiao Lai", "Xin Chen", "Dong Wang", "Huchuan Lu"],
    fileName: "2303.10826v2.pdf",
    task: "multimodal-tracking",
    tags: ["multimodal-tracking", "visual-prompt", "prompt-tuning", "parameter-efficient", "rgb-d", "rgb-t", "rgb-e", "modality-complementary"],
    difficulty: "advanced",
    summary:
      "ViPT freezes an OSTrack RGB foundation (ViT, 93.36M params) and tunes only 0.84M parameters: an auxiliary patch embed plus stage-wise Modality-Complementary Prompters that compress RGB and auxiliary tokens, apply spatial fovea, sum and re-expand into residual prompts added to the RGB stream. One paradigm covers RGB-D, RGB-T and RGB-E, beating full fine-tuning (178.6M) on DepthTrack (59.4 F-score), VOT-RGBD2022 (0.721 EAO), RGBT234 (61.7 MSR), LasHeR and VisEvent.",
    problem:
      "Multi-modal tracking has 10x less paired data than RGB tracking (DepthTrack 150, LasHeR 979, VisEvent 500 train sequences) so extending RGB trackers with a full auxiliary branch and full fine-tuning is slow, storage-heavy, poor at transfer and overfits, discarding large-scale RGB pretraining.",
    background: ["multimodal", "sot", "attention", "appearance-features", "success-plot", "benchmark-design"],
    previousWork: [
      {
        name: "Full fine-tuning dual-branch (DeT/ATOM-DiMP, mfDiMP, RGB-E dual trackers)",
        limitation:
          "Adds a whole auxiliary feature branch and retrains everything (178.6M params), corrupting pretrained knowledge on scarce pairs and storing one giant copy per task.",
        whyThisPaper:
          "Frozen foundation plus 0.84M prompt params beats the FFT variant on DepthTrack 59.4 vs 55.6, LasHeR 52.5 vs 51.7 SR and VisEvent 59.2 vs 58.7 SR.",
      },
      {
        name: "RGB foundation trackers (OSTrack, STARK, SiamRPN++)",
        limitation:
          "Pure RGB fails under illumination, clutter and blur where depth/thermal/event complementarity is needed, yet holds the only large-scale representation.",
        whyThisPaper:
          "Keeps all 93.36M RGB params frozen and injects auxiliary knowledge as residual prompts, gaining +6.5 F-score DepthTrack and +11.3 SR LasHeR over the foundation.",
      },
      {
        name: "ProTrack prompting",
        limitation:
          "Linearly sums modalities with frozen RGB weights and no downstream tuning, so inter-modal complementarity at multiple semantic levels is unexplored.",
        whyThisPaper:
          "Learned stage-wise MCP with fovea fusion beats ProTrack on DepthTrack (59.4 vs 57.8), RGBT234 MSR (61.7 vs 59.9) and VOT-RGBD2022 (0.721 vs 0.651).",
      },
    ],
    researchGap:
      "No parameter-efficient prompt-tuning framework adapted a frozen RGB tracker to RGB-D/T/E while jointly learning cross-modal complementarity.",
    contribution: [
      "ViPT framework FM_M:{X_RGB,X_A,B0}->B initializing from frozen OSTrack foundation f* and tuning only theta={tau_A,{P^l}} (0.84M, <1%).",
      "Modality-Complementary Prompter block: 1x1 reduce by 8x, RGB spatial-fovea with learnable lambda, elementwise sum with auxiliary embedding, 1x1 re-expand to prompt tokens.",
      "Stage-wise insertion (default every layer; ablated per 1/2/4/6/12) with residual addition H_l = H^l_RGB + P^{l+1}, unifying RGB-D/T/E without task-specific modulation.",
      "Same loss as OSTrack (focal plus L1/IoU) and lean training (DepthTrack/LasHeR/VisEvent train sets only, no synthetic expansion) with 60-epoch convergence.",
      "SOTA on five benchmarks with ablations over modalities, prompt depth, data volume and frozen-vs-unfrozen tuning.",
    ],
    method: {
      pipeline: ["embed-rgb", "embed-aux", "prompt-mcp", "add-residual", "encode-frozen", "predict-head"],
      architecture:
        "Frozen OSTrack ViT encoder (L layers, C=768) plus head; trainable auxiliary patch embed tau_A and L MCP blocks (two 1x1 reducers g1/g2, fovea, 1x1 expander g3, reduce factor beta=96 i.e. 768/8); exemplar+search tokens Hz/Hx with positional embedding.",
      motionModel: "None — per-frame prompt-conditioned localization without temporal filter.",
      appearanceModel:
        "Frozen RGB representation plus learned auxiliary-conditioned prompts at multiple semantic levels; head reused frozen.",
      association: "Not applicable — single-object prompt tracking; evaluation is OPE-style per benchmark.",
      detectionDependency: "None — box given first frame; auxiliary flow is depth/thermal/event image aligned to RGB.",
      trackManagement: "None beyond foundation head; short-term protocol per benchmark (DepthTrack long-term treated short-term).",
      loss:
        "L = Lcls + lambda_iou*Liou + lambda_L1*L1; weighted focal classification plus L1 and generalized-IoU regression, identical settings to OSTrack.",
      optimization:
        "2xA100, batch 64, 60 epochs x 6e4 pairs, AdamW wd 1e-4, lr 4e-5 decay x0.1 after 48 epochs; prompt params Xavier init; foundation frozen including embed, encoder and head.",
    },
    equations: [
      {
        id: "vipt-foundation",
        label: "Frozen foundation forward",
        formula: "H^l_RGB = E^l(H^{l-1}_RGB), l=1..L; B = phi(H^L_RGB); H^0 = [Hz;Hx]",
        variables: [
          { symbol: "E^l", meaning: "l-th frozen ViT encoder layer (MSA+LN+FFN+residual)" },
          { symbol: "H_RGB", meaning: "RGB exemplar+search tokens (D=768)" },
          { symbol: "phi", meaning: "frozen box head" },
        ],
        intuition: "Run the big RGB tracker exactly as pretrained; it already knows how to find objects in RGB.",
        why: "Preserves large-scale RGB knowledge that scarce multimodal pairs would otherwise corrupt.",
        where: "Section 3.1, Eqs. 1-2; OSTrack details referenced.",
        paperIds: ["T083"],
      },
      {
        id: "vipt-add",
        label: "Residual prompt injection",
        formula: "H_l = H^l_RGB + P^{l+1}, l=0..L-1",
        variables: [
          { symbol: "P^{l+1}", meaning: "prompt tokens from (l+1)-th MCP block" },
          { symbol: "H_l", meaning: "prompted tokens fed to next frozen layer" },
        ],
        intuition: "Nudge each layer's RGB features with a small learned multimodal correction.",
        why: "Residual form plugs into any pretrained tracker without changing its architecture; summation aligns modalities better than concatenation in ablations.",
        where: "Section 3.2, Eq. 3; stage-wise placement.",
        paperIds: ["T083"],
      },
      {
        id: "vipt-mcp",
        label: "MCP block I/O",
        formula: "P^l = P^l(H^{l-1}, P^{l-1}); H^0=H^0_RGB, P^0=H^0_A",
        variables: [
          { symbol: "H^{l-1}", meaning: "foundation-stream tokens into block l" },
          { symbol: "P^{l-1}", meaning: "auxiliary/prompt-stream tokens into block l" },
        ],
        intuition: "Each block looks at what RGB understands and what the auxiliary modality says, then writes a note for the next layer.",
        why: "Multi-level prompting learns complementarity at diverse semantics; more blocks monotonically help (Figure 8).",
        where: "Section 3.2, Eq. 4 and Figure 3.",
        paperIds: ["T083"],
      },
      {
        id: "vipt-fovea",
        label: "MCP compress–fovea–fuse–expand",
        formula: "M_RGB=g1(H); M_A=g2(P); M_fovea=softmax_lambda(M_RGB); Me=M_RGB * M_fovea; P^l=g3(Me+M_A); C=768, beta=8",
        variables: [
          { symbol: "g1/g2/g3", meaning: "1x1 convolutions (reduce, reduce, expand)" },
          { symbol: "lambda", meaning: "learnable per-block smoothing factor" },
          { symbol: "M_fovea", meaning: "channel-wise spatial attention mask" },
        ],
        intuition: "Squeeze both modalities, spotlight the informative RGB spots, mix in auxiliary cues, and blow back up.",
        why: "Bottleneck keeps prompts tiny (0.84M total) while fovea emphasizes discriminative regions verified by response/t-SNE maps.",
        where: "Section 3.2, Eqs. 5-8 and Figure 3.",
        params: "Reduce factor 8 (768 to 96 channels); unshared tiny params per block.",
        paperIds: ["T083"],
      },
      {
        id: "vipt-optim",
        label: "Prompt-only optimization",
        formula: "theta_tuned = argmin_theta 1/|D| sum L(phi(H^L), Bgt); theta = {tau_A, {P^l}}",
        variables: [
          { symbol: "theta", meaning: "only trainable params: aux embed plus MCP blocks" },
          { symbol: "D", meaning: "downstream paired set (DepthTrack/LasHeR/VisEvent train)" },
        ],
        intuition: "Freeze the library, train only the translator notes for the new modality.",
        why: "0.84M vs 178.6M FFT gives faster convergence, less storage per task and less overfitting on 0.2-0.5M-frame downstream sets.",
        where: "Section 3.2, Eq. 9.",
        paperIds: ["T083"],
      },
      {
        id: "vipt-loss",
        label: "Tracking loss (inherited)",
        formula: "L = Lcls + lambda_iou Liou + lambda_L1 L1",
        variables: [
          { symbol: "Lcls", meaning: "weighted focal classification loss" },
          { symbol: "Liou / L1", meaning: "generalized-IoU and L1 box regression" },
        ],
        intuition: "Score the right spot and draw a tight box, exactly as the RGB teacher did.",
        why: "No loss retuning needed, isolating the prompting gain from objective tricks.",
        where: "Section 3.2, Eq. 10.",
        paperIds: ["T083"],
      },
    ],
    datasets: ["lasheR", "others"],
    metrics: ["success-auc", "precision", "recall", "eao"],
    baselines: ["OSTrack", "DeT", "ProTrack", "SPT", "mfDiMP", "ATOM", "DiMP", "SiamRPN++", "APFNet", "JMMAC", "STARK"],
    results: [
      "DepthTrack test: ViPT 0.594 F-score / 0.596 Re / 0.592 Pr vs OSTrack 0.529 / 0.522 / 0.536 (+6.5 F), ProTrack 0.578 / 0.573 / 0.583 and SPT 0.538.",
      "VOT-RGBD2022: ViPT 0.721 EAO / 0.815 Accuracy / 0.871 Robustness vs OSTrack 0.676 / 0.803 / 0.833 (+4.5 EAO) and ProTrack 0.651.",
      "RGBT234: ViPT 0.835 MPR / 0.617 MSR vs ProTrack 0.795 / 0.599 (+1.8 MSR) and APFNet 0.827 / 0.579.",
      "LasHeR test: ViPT 52.5 SR / 65.1 PR vs foundation 41.2 / 51.5 and FFT 51.7 / 64.8; paper reports +10.5 success / +11.3 precision over second place.",
      "VisEvent test (320 videos): ViPT 59.2 SR / 75.8 PR vs foundation 53.4 / 69.5 and FFT 58.7 / 75.2 (+5.8/+6.3 over OSTrack runner-up).",
      "Cost: 0.84M trainable (0.9% of 93.36M) vs FFT 178.6M (100%); prompt-shallow 0.59M already 52.0 DepthTrack F; ViPT-deep best on all three ablated benchmarks.",
    ],
    ablations: [
      "Modalities (Table 4): RGB-only foundation 52.9/41.2/53.4; +FFT 55.6/51.7/58.7; +ViPT 59.4/52.5/59.2 across DepthTrack/LasHeR/VisEvent.",
      "Variants: VPT-style prompt-shallow/deep 52.0/54.9 DepthTrack F vs ViPT-shallow 56.4 vs ViPT-deep 59.4; similar gaps on LasHeR/VisEvent.",
      "Blocks (Fig. 8): inserting per 12/6/4/2/1 layers monotonically improves all benchmarks with modest param growth to 0.9M.",
      "Data/params (Table 5): synthetic expansion alone 59.3 or unfreezing alone 55.4 DepthTrack F do not beat frozen ViPT 59.4; joint 58.3, costlier.",
    ],
    limitations: {
      authorStated: [
        "Focuses on visual prompting only; extension to vision-language tracking is left as potential future work.",
        "Still trains separately per multi-modal task; joint multi-modality general model is future interest.",
      ],
      evident: [
        "Event inputs are event images converted from raw events, not raw event flow, limiting temporal-resolution claims.",
        "Long-term DepthTrack treated with a short-term algorithm; re-detection after disappearance is not isolated.",
      ],
    },
    assumptions: [
      "RGB and auxiliary streams are spatially aligned and temporally synchronized paired inputs.",
      "RGB foundation (OSTrack) representation transfers to depth/thermal/event with only residual prompting.",
    ],
    computation:
      "2xA100, batch 64, 60 epochs x 6e4 pairs, AdamW wd 1e-4, lr 4e-5 decay at 48; train sets are DepthTrack/LasHeR/VisEvent only.",
    relations: [
      { to: "T071", type: "builds-on", note: "Freezes OSTrack one-stream foundation (feature, interaction and head) and reuses its loss." },
      { to: "T028", type: "uses-as-baseline", note: "SiamRPN++ dual-modal extension cited as full-tuning precedent for RGB-T." },
      { to: "T031", type: "uses-as-baseline", note: "DiMP cited as RGB foundation extended by DeT/mfDiMP full-tuning competitors." },
      { to: "T057", type: "uses-as-baseline", note: "Evaluated and prompt-tuned on LasHeR RGB-T benchmark (979 train / 245 test).", },
      { to: "T055", type: "uses-as-baseline", note: "TransT-class RGB Transformer lineage referenced among foundation works." },
    ],
    concepts: ["multimodal", "sot", "attention", "appearance-features", "success-plot", "benchmark-design"],
    impact:
      "First proof that <1% prompt-tuning can beat full dual-branch fine-tuning across RGB-D/T/E, shifting multimodal tracking toward frozen-foundation adaptation.",
  },
{
    id: "T084",
    arxiv: "2303.10951",
    title: "Tracker Meets Night: A Transformer Enhancer for UAV Tracking",
    shortTitle: "SCT (Tracker Meets Night)",
    year: 2023,
    authors: ["Junjie Ye", "Changhong Fu", "Ziang Cao", "Shan An", "Guangze Zheng", "Bowen Li"],
    fileName: "2303.10951v1.pdf",
    task: "single-object",
    tags: ["nighttime-tracking", "uav-tracking", "low-light-enhancement", "transformer", "plug-and-play", "benchmark"],
    difficulty: "intermediate",
    summary:
      "SCT is a spatial-channel Transformer low-light enhancer plugged in front of any UAV tracker: it estimates illumination and noise curve maps and enlightens template/search patches through a robust non-linear curve projection with a denoising term. It is trained with a task-inspired perceptual loss that aligns enhanced-image features with normal-light features under a tracking backbone (modified AlexNet), needing only 485 paired LOL images and no nighttime tracking labels. Full SCT lifts a SiamRPN++-AlexNet baseline from 0.372/0.474 to 0.421/0.547 success/precision on UAVDark135, boosts all six tested SOTA trackers on both UAVDark135 and the newly built DarkTrack2021 (110 sequences, 100,377 frames), and runs at ~31.25 FPS on a Jetson AGX Xavier.",
    problem:
      "SOTA trackers trained on daytime images lose feature-extraction effectiveness at night (dim brightness, low contrast, sensor noise), so daytime superiority does not carry over to nighttime UAV operation. Existing low-light enhancers optimize human-perception metrics (PSNR/SSIM), not tracking features, and are too heavy for UAV onboard compute; annotated nighttime tracking data is too scarce to retrain trackers.",
    background: ["sot", "bounding-box", "attention", "appearance-features", "success-plot"],
    previousWork: [
      {
        name: "Generic low-light enhancers (DCE++, EnlightenGAN, LIME, LLVE, RUAS)",
        limitation:
          "They optimize signal fidelity for human perception and retouch illumination while neglecting real-world noise; their gains transfer poorly to tracking (second-best DCE++ gains only ~8% vs SCT's >13%).",
        whyThisPaper:
          "SCT is trained with a tracking-backbone perceptual loss and a noise-aware curve model so enhancement directly serves feature extraction.",
      },
      {
        name: "DarkLighter (Retinex-inspired enhancer for UAV tracking)",
        limitation:
          "Designed intuitively with weak collaboration with the tracking task, so its nighttime tracking benefit is inferior despite the UAV motivation.",
        whyThisPaper:
          "SCT's task-inspired training and spatial-channel global modeling beat DarkLighter by a large margin in Fig. 6.",
      },
      {
        name: "Daytime Siamese/Discriminative trackers (SiamRPN++, DiMP, PrDiMP, HiFT, SiamAPN++)",
        limitation:
          "Backbones trained on daytime images produce non-discriminative confidence maps in darkness (Fig. 4), and no nighttime labels exist to adapt them.",
        whyThisPaper:
          "A plug-and-play preprocessor brings them to night (+19.69% precision for HiFT) with zero nighttime tracking sequences.",
      },
    ],
    researchGap:
      "Before this paper, no tracking-tailored, lightweight low-light enhancer existed that models global context with local detail, denoises jointly, and is trained against a tracker's feature needs — nor a large high-altitude urban nighttime UAV benchmark.",
    contribution: [
      "Task-inspired SCT enhancer: U-shaped CNN-Transformer hybrid with a spatial-channel attention module (window W-MSA + ResFFN) estimating illumination map I and noise map N, trained by a tracking-backbone perceptual loss on layers 3–5 of a modified AlexNet.",
      "Robust non-linear curve projection with an explicit noise subtraction term, performing simultaneous denoising and illumination retouching over T iterations.",
      "DarkTrack2021 benchmark: 110 manually annotated HD nighttime urban UAV sequences, 100,377 frames (92–6579 per sequence, mean 913), covering person/bus/car/truck/motor/dog/building with viewpoint change, fast motion, occlusion, low resolution/brightness, out-of-view.",
      "Plug-and-play gains on six SOTA trackers across UAVDark135 and DarkTrack2021 plus real-world Jetson AGX Xavier tests at ~31.25 FPS.",
    ],
    method: {
      pipeline: ["crop-patches", "estimate-curves", "denoise", "project-curves", "extract-features", "localize", "update"],
      architecture:
        "U-shaped CNN-Transformer hybrid (K=4 encoders/decoders, 32-dim first stage): CNN encoders halve resolution and double channels; one spatial Transformer layer (flattened H·W tokens) plus one channel Transformer layer (flattened C tokens) with W-MSA (4×4 windows) and ResFFN; mirrored CNN decoders with Tanh outputting I and N maps at 128×128 estimation resolution.",
      appearanceModel:
        "No tracker retraining: frozen tracking backbones receive enlightened template/search patches; enhancement trained so Fm(Xo) ≈ Fm(Y) for backbone layers m=3,4,5.",
      detectionDependency: "None — preprocessing for class-agnostic single-object trackers.",
      trackManagement:
        "Template cropped from frame one, search region from last prediction; at test time only template/search patches pass through SCT (full-frame enhancement is visualization-only).",
      loss:
        "Task-inspired perceptual loss only (Eq. 7); no PSNR/SSIM or detection loss.",
      optimization:
        "AdamW, lr 8e-4, weight decay 0.02, 5-epoch warmup, 100 epochs, batch 32, 256×256 random crops from 485 LOL pairs; inference on i9-9920X + TITAN RTX, deployment on Jetson AGX Xavier.",
    },
    equations: [
      {
        id: "sct-partition",
        label: "Window patch partition",
        formula: "F = {f1, f2, ..., fj, ..., f_{L²/M²}}, fj ∈ R^{M²×N}",
        variables: [
          { symbol: "F", meaning: "flattened feature map reshaped to L×L 2D layout" },
          { symbol: "M", meaning: "non-overlapping window size (4×4)" },
          { symbol: "fj / N", meaning: "j-th window patch / token channel dimension" },
        ],
        intuition: "Cut the feature map into small tiles so attention runs cheaply inside each tile.",
        why: "Vanilla global attention is unaffordable on a UAV; windowing keeps the Transformer light.",
        where: "Section III-A-1, Eq. 1; applied to spatial and channel-flattened features.",
        paperIds: ["T084"],
      },
      {
        id: "sct-layer",
        label: "ResFFN Transformer layer",
        formula: "F̂ = W-MSA(LN(F)) + F; F_out = ResFFN(LN(F̂)) + F̂",
        variables: [
          { symbol: "W-MSA", meaning: "window-based multi-head self-attention with relative bias B" },
          { symbol: "ResFFN", meaning: "linear → reshape → Conv + DWConv residual → flatten → linear" },
          { symbol: "LN", meaning: "layer normalization" },
        ],
        intuition: "Attention gathers global context; the convolutional feed-forward restores local detail the attention misses.",
        why: "Standard Transformers are weak at local context critical for enhancement; ResFFN beats MLP by ~5 points (Table I).",
        where: "Section III-A-1, Eq. 4, Fig. 3(b); one spatial and one channel layer.",
        params: "Removing ResFFN (MLP instead) drops gains from +13.3/+15.4 to +8.0/+8.5.",
        paperIds: ["T084"],
      },
      {
        id: "sct-attention",
        label: "Biased window attention",
        formula: "Attention(Q,K,V) = SoftMax(QKᵀ/√d + B)V",
        variables: [
          { symbol: "Q/K/V", meaning: "query, key, value matrices per window (M²×d)" },
          { symbol: "B", meaning: "relative position bias (M²×M²)" },
          { symbol: "d", meaning: "query dimension" },
        ],
        intuition: "Each tile location votes on every other location in its tile, with a learned preference for relative offsets.",
        why: "Models long-range illumination dependencies convolutions cannot capture.",
        where: "Section III-A-1, Eq. 3; inside every W-MSA.",
        paperIds: ["T084"],
      },
      {
        id: "sct-curve",
        label: "Non-linear curve projection",
        formula: "X^t = X^{t−1} + I ⊙ X^{t−1} ⊙ (1 − X^{t−1}), t = 1..T; Xo = X^T",
        variables: [
          { symbol: "I", meaning: "estimated illumination curve parameter map (3×Hi×Wi)" },
          { symbol: "X^t", meaning: "intermediate enhanced image at iteration t (X^0 = input)" },
          { symbol: "T", meaning: "number of projection iterations" },
        ],
        intuition: "Repeatedly bend each pixel's brightness upward with a curve whose shape the network predicts per pixel.",
        why: "Reformulates enhancement as pixel-wise curve estimation (following DCE++) instead of costly image-to-image mapping.",
        where: "Section III-A-2, Eq. 5.",
        paperIds: ["T084"],
      },
      {
        id: "sct-robust-curve",
        label: "Robust curve projection with noise term",
        formula: "X̂^{t−1} = X^{t−1} − N; X^t = X̂^{t−1} + I ⊙ X̂^{t−1} ⊙ (1 − X̂^{t−1}), t = 1..T",
        variables: [
          { symbol: "N", meaning: "estimated noise curve map (3×Hi×Wi)" },
          { symbol: "X̂^{t−1}", meaning: "denoised intermediate before illumination retouching" },
        ],
        intuition: "Subtract the predicted grime first, then brighten — so brightening amplifies the scene, not the noise.",
        why: "Ablation shows the noise term doubles the benefit (removing it halves gains to +5.6/+6.6); real night UAV frames are noisy.",
        where: "Section III-A-2, Eq. 6; core enhancement step.",
        params: "Without the noise term the curve model wastes capacity fitting noise as signal.",
        paperIds: ["T084"],
      },
      {
        id: "sct-task-loss",
        label: "Task-inspired perceptual loss",
        formula: "L = Σ_{m∈{3,4,5}} (1/(cm·hm·wm)) ‖Fm(Xo) − Fm(Y)‖²",
        variables: [
          { symbol: "Fm", meaning: "m-th layer feature map of the frozen tracking backbone (modified AlexNet)" },
          { symbol: "Xo / Y", meaning: "enhanced low-light image / ground-truth normal-light image" },
          { symbol: "cm, hm, wm", meaning: "channel/height/width dims of layer m" },
        ],
        intuition: "Train the enhancer until a tracker's eyes see the same features in the enhanced night image as in daylight.",
        why: "Aligns optimization with tracking instead of human fidelity; the sole training signal, needing no night tracking labels.",
        where: "Section III-B, Eq. 7; only loss used.",
        paperIds: ["T084"],
      },
    ],
    datasets: ["others"],
    metrics: ["success-auc", "precision", "fps"],
    baselines: ["SiamRPN++", "SiamAPN++", "HiFT", "DiMP18", "DiMP50", "PrDiMP50", "DCE++", "EnlightenGAN", "LIME", "LLVE", "RUAS", "DarkLighter"],
    results: [
      "DarkTrack2021: 110 sequences, 100,377 frames; lengths 92–6579, mean 913 frames.",
      "UAVDark135 ablation (SiamRPN++-AlexNet baseline 0.372 success / 0.474 precision): full SCT reaches 0.421/0.547, i.e. +13.3%/+15.4%; second-best DCE++ gains only ~8%.",
      "Module ablations: w/o channel attention 0.415/0.536 (+11.7/+13.0); w/o spatial attention 0.409/0.523 (+10.1/+10.3); pure-CNN UNet 0.412/0.533 (+10.9/+12.4); MLP FFN 0.401/0.514 (+8.0/+8.5); w/o denoise 0.393/0.505 (+5.6/+6.6).",
      "UAVDark135 (Fig. 7a): HiFT gains +19.69% precision / +16.59% success with SCT; all six trackers improve.",
      "DarkTrack2021 (Fig. 7b): DiMP50 gains +5.84%/+4.80% to 0.681 precision / 0.518 success; shallow AlexNet backbones gain >13%, ResNet-50 backbones ~5%.",
      "Real-world: ~31.25 FPS on NVIDIA Jetson AGX Xavier without TensorRT; CLE curves stay within the 20-pixel success threshold.",
    ],
    ablations: [
      "Attention: each of spatial/channel attention contributes ~1.6–3 points; Transformer UNet beats CNN UNet by ~2.5 points.",
      "ResFFN vs MLP feed-forward: +13.3/+15.4 vs +8.0/+8.5 — local-context preservation is crucial.",
      "Noise term: removing it halves the gain (+5.6/+6.6), confirming joint denoise-illuminate design.",
      "Enhancer comparison (Fig. 6): all six SOTA enhancers cluster at a suboptimal level; SCT exceeds them by ~5+ points.",
      "Backbone depth effect: SCT helps shallow backbones more, consistent with shallow features collapsing harder at night.",
    ],
    limitations: {
      authorStated: [
        "In the practical pipeline only template and search patches are enhanced (full-frame enhancement is visualization-only), so global scene context is never enlightened.",
        "Training relies on 485 paired LOL low/normal-light images; generalization to true UAV night scenes rests on this small proxy set.",
        "Real-time speed (~31.25 FPS) is reported without TensorRT acceleration.",
      ],
      evident: [
        "Perceptual loss uses a shallow modified-AlexNet backbone, so enhancement is tuned to one feature family while deeper backbones gain less.",
        "Enhancement runs per patch per frame with no temporal consistency constraint across frames.",
      ],
    },
    assumptions: [
      "Nighttime degradation is factorizable into per-pixel illumination plus additive noise correctable by curve projection.",
      "Daylight-feature similarity under one backbone predicts tracking success for arbitrary tracker heads.",
    ],
    computation:
      "Training on 485 LOL pairs (100 epochs, batch 32); inference PC i9-9920X + TITAN RTX; Jetson AGX Xavier ~31.25 FPS without TensorRT.",
    relations: [
      { to: "T028", type: "uses-as-baseline", note: "SiamRPN++ (AlexNet) is the ablation baseline (0.372/0.474) and a Fig. 7 beneficiary." },
    ],
    concepts: ["sot", "bounding-box", "attention", "appearance-features", "success-plot", "benchmark-design"],
    impact:
      "Introduced tracking-tailored enhancement (perceptual loss from a tracking backbone) plus the DarkTrack2021 night-UAV benchmark, establishing the plug-and-play preprocessor pattern for nighttime tracking.",
  },
  {
    id: "T085",
    arxiv: "2304.08408",
    title: "OVTrack: Open-Vocabulary Multiple Object Tracking",
    shortTitle: "OVTrack",
    year: 2023,
    authors: ["Siyuan Li", "Tobias Fischer", "Lei Ke", "Henghui Ding", "Martin Danelljan", "Fisher Yu"],
    fileName: "2304.08408v1.pdf",
    task: "multi-object",
    tags: ["open-vocabulary", "multi-object-tracking", "vision-language", "clip-distillation", "diffusion-hallucination", "tracking-by-detection"],
    difficulty: "advanced",
    summary:
      "OVTrack defines open-vocabulary MOT (train on base classes C_base, evaluate on base plus held-out novel classes C_novel = LVIS rare classes on TAO, scored with closed-set TETA/Track-mAP so precision counts) and builds the first tracker for it. Faster R-CNN is made class-agnostic for localization; classification becomes CLIP text/image embedding distillation (open prompt set at test time); association is appearance-only contrastive learning that also benefits from CLIP distillation. A stable-diffusion data-hallucination strategy synthesizes reference views (perturbed background, preserved positives, caption-guided distractors) so tracking learns from static LVIS images only. OVTrack sets SOTA on TAO (val TETA 35.5 base / 27.8 novel; closed-set Track mAP 15.9) and zero-shot transfers to BDD100K (42.5 TETA).",
    problem:
      "Closed-set MOT is bounded by small benchmark taxonomies and cannot track unseen categories; open-world tracking alternatives resort to recall-only, class-agnostic evaluation that cannot penalize false positives or score classification. Video annotation at vocabulary scale is prohibitively expensive, and motion cues are brittle across arbitrary scenery, so open-vocabulary tracking must learn association from static images.",
    background: ["mot", "tracking-by-detection", "data-association", "appearance-features", "reid"],
    previousWork: [
      {
        name: "Closed-set trackers (QDTrack, TETer, GTR, AOA)",
        limitation:
          "Bound to pre-defined taxonomies (trained even on the held-out rare classes) yet score ~0 ClsA on novel classes (TETer novel ClsA 0.2 val / 0.0 test).",
        whyThisPaper:
          "OVTrack replaces the classifier with CLIP embedding heads, reaching 11.4/6.1 novel ClsA with RegionCLIP while training only on base classes and static images.",
      },
      {
        name: "Open-world tracking (TAO-OW) and class-agnostic pipelines",
        limitation:
          "Dismisses classification as ill-posed and evaluates recall only — precision and semantic accuracy are unmeasurable.",
        whyThisPaper:
          "Open-vocabulary MOT assumes the novel class list is given at test time, enabling standard closed-set TETA/Track-mAP scoring of precision and recall.",
      },
      {
        name: "Learning tracking from static images (CenterTrack offsets, FairMOT pseudo-identities, QDTrack augmentation)",
        limitation:
          "Classic translation/scale/rotation augmentation cannot simulate viewpoint, lighting, or context change, capping appearance-robustness learning.",
        whyThisPaper:
          "DDPM hallucination synthesizes realistic background perturbations, preserved positives and distractor negatives (+1.8 AssocA).",
      },
    ],
    researchGap:
      "Before this paper, no task definition, benchmark protocol, or tracker existed for evaluating multi-object tracking beyond training categories with precision-aware closed-set metrics.",
    contribution: [
      "Open-vocabulary MOT task and TAO benchmark protocol: LVIS rare classes as held-out C_novel (35 val / 33 test classes), TETA (LocA/ClsA/AssocA) as primary metric.",
      "OVTrack model: class-agnostic Faster R-CNN localization + CLIP text/image distillation heads + contrastive appearance-only association with bi-softmax matching and 10-frame track memory.",
      "DDPM data-hallucination strategy: branched foreground-preserving denoising (δ0=0.75 → η) with caption guidance for background perturbation, positive preservation, and distractor generation from a single static image.",
      "State-of-the-art TAO results training on static LVIS images only, plus zero-shot generalization to BDD100K and arbitrary internet videos (heron/hippo/drone, pikachu).",
    ],
    method: {
      pipeline: ["localize-agnostic", "classify-via-text", "embed-appearance", "hallucinate-reference", "contrast", "associate"],
      architecture:
        "ResNet50-FPN Faster R-CNN trained class-agnostic (RPN + regression only); RoI features feed a text head (t̂r) and image head (îr) distilled from CLIP, plus a tracking head (qr, 4-conv-1-fc each; text/regression share a head with parallel linear layers). Inference: NMS proposals (train 256 @0.7 IoU; test |P|=50 class-agnostic NMS for TETA, 300 class-specific for Track mAP), text-head labeling, bi-softmax appearance association (β=0.5, γ=0.0001), temporal voting for video-level class.",
      appearanceModel:
        "Contrastive instance embeddings (Eq. 6) with auxiliary magnitude loss (Eq. 9); CLIP image/text distillation improves both ClsA (15.6→18.1) and AssocA.",
      association:
        "Appearance-only bi-softmax similarity (Eq. 10) with cosine term; motion cues deliberately unused as brittle in open-vocabulary scenery.",
      reid:
        "Track memory of appearance embeddings over 10 frames for re-identification after occlusion; new track if pr > γ.",
      detectionDependency:
        "Entirely dependent on class-agnostic RPN/RCNN proposals; shown to generalize to unseen classes.",
      trackManagement:
        "Match to max-similarity track if s > β else birth; duplicate removal; confidence-gated creation.",
      loss:
        "Two-stage: detection losses + Ltext + Limage first; then tracking fine-tune with 0.25·Ltrack + 1.0·Laux.",
      optimization:
        "SGD lr 0.02, momentum 0.9, wd 1e-4, batch 16, decay ×0.1 at epochs [8,16] (20-epoch detector stage); 6-epoch tracking stage (decay [3,5]); ResNet50 backbone, self-supervised ImageNet init.",
    },
    equations: [
      {
        id: "ovtrack-text",
        label: "Open-vocabulary text affinity and loss",
        formula: "z(r) = [cos(t̂r,tbg), cos(t̂r,t1), ..., cos(t̂r,tC)]; Ltext = (1/|P|) Σ_r LCE(softmax(z(r)/λ), cr), λ = 0.07",
        variables: [
          { symbol: "t̂r / tc", meaning: "predicted RoI text embedding / CLIP text embedding of prompted class c" },
          { symbol: "tbg", meaning: "learned background prompt embedding" },
          { symbol: "P(c)", meaning: "L context vectors + class-name embedding fed to CLIP text encoder" },
          { symbol: "cr", meaning: "ground-truth base-class label of proposal r" },
        ],
        intuition: "Classify by asking CLIP which prompted name the box looks most like, instead of a fixed softmax over training classes.",
        why: "A fixed classifier cannot emit unseen classes; embedding similarity makes the vocabulary swappable at test time.",
        where: "Section 4.1, Eqs. 1–2; classification branch.",
        params: "Temperature λ=0.07 sharpens affinities; learned context vectors adapt CLIP prompts to crowded MOT scenes.",
        paperIds: ["T085"],
      },
      {
        id: "ovtrack-image",
        label: "CLIP image distillation loss",
        formula: "Limage = (1/|P|) Σ_r ‖îr − ir‖1, ir = CLIP-Image(resize(crop(I, br)))",
        variables: [
          { symbol: "îr", meaning: "predicted RoI image embedding" },
          { symbol: "ir", meaning: "CLIP image encoder embedding of the cropped-resized proposal box" },
        ],
        intuition: "Force each box's visual embedding to speak CLIP's visual language, not just its textual one.",
        why: "Adding Limage lifts ClsA 15.6→18.1 at constant LocA/AssocA (Table 4); it also strengthens appearance features for association.",
        where: "Section 4.1, Eq. 3; trained on RPN proposals for diversity.",
        paperIds: ["T085"],
      },
      {
        id: "ovtrack-track",
        label: "Contrastive instance similarity loss",
        formula: "PosD(q) = (1/|Q+|) Σ exp(q·q+/τ); Sim(q) = exp(q·q+/τ)/(PosD(q) + Σ_{Q−} exp(q·q−/τ)); Ltrack = −Σ_q (1/|Q+|) Σ log Sim(q+)",
        variables: [
          { symbol: "q", meaning: "appearance embedding of a matched RoI in Ikey" },
          { symbol: "Q+ / Q−", meaning: "same-identity / different-identity RoIs in Iref (IoU-matched)" },
          { symbol: "τ", meaning: "contrastive temperature" },
        ],
        intuition: "Pull views of the same object together and push different objects apart across the hallucinated image pair.",
        why: "Appearance is the only reliable open-vocabulary cue; the loss learns identity-discriminative embeddings without video.",
        where: "Section 4.1, Eqs. 4–6; tracking head supervision.",
        paperIds: ["T085"],
      },
      {
        id: "ovtrack-diffusion",
        label: "DDPM forward and reverse steps",
        formula: "n(xk|x_{k−1}) = N(xk; √(1−δk)·x_{k−1}, δk·I); mθ(x_{k−1}|xk) = N(x_{k−1}; μθ(xk,k), Σθ(xk,k))",
        variables: [
          { symbol: "δk", meaning: "variance schedule at step k (initialized δ0 = 0.75)" },
          { symbol: "x", meaning: "latent representation of Iref via the stable-diffusion encoder" },
          { symbol: "μθ/Σθ", meaning: "denoising network's predicted Gaussian parameters" },
        ],
        intuition: "Noise the image toward static, then learn to walk it back — the walk back is steered to hallucinate new content.",
        why: "Provides the generative engine whose fidelity classic augmentation lacks (viewpoint/lighting/context change).",
        where: "Section 4.2, Eqs. 7–8; backbone of data hallucination.",
        paperIds: ["T085"],
      },
      {
        id: "ovtrack-hallucinate",
        label: "Foreground-preserving branched composition",
        formula: "xk = A+ ⊙ x̂k + (1 − A+) ⊙ xk, iterate δ0 = 0.75 → η = 0.02 with caption guidance; homogenize with mθ below η",
        variables: [
          { symbol: "A+", meaning: "union mask of positive-instance (LVIS mask) regions" },
          { symbol: "x̂k", meaning: "forward-noised original foreground at level k" },
          { symbol: "xk", meaning: "caption-conditioned denoised sample at level k" },
          { symbol: "η", meaning: "homogenization threshold (0.02)" },
        ],
        intuition: "Repaint the background and invent distractors while holding the target still, like reshooting the same actor on a new set.",
        why: "Achieves all three goals at once: perturbed backgrounds, identity-preserved positives, caption-guided negatives; +1.8 AssocA.",
        where: "Section 4.2, Fig. 5; generates Iref per LVIS image (boxes >64² in A+).",
        params: "Ablation: hallucination without geometric transform/language prompt fails; adding both gives the full +1.8.",
        paperIds: ["T085"],
      },
      {
        id: "ovtrack-aux",
        label: "Auxiliary embedding magnitude loss",
        formula: "Laux = ((q·q′)/(‖q‖‖q′‖) − e)², e = 1 if same identity else 0",
        variables: [
          { symbol: "q/q′", meaning: "two appearance embeddings being compared" },
          { symbol: "e", meaning: "binary same-identity indicator" },
        ],
        intuition: "Pin the cosine similarity itself to 1 for matches and 0 for non-matches, not just their ranking.",
        why: "Constrains logit magnitude so bi-softmax scores are calibrated for the β threshold at inference.",
        where: "Appendix Eq. 9; weight 1.0 in tracking stage.",
        paperIds: ["T085"],
      },
      {
        id: "ovtrack-assoc",
        label: "Bi-softmax association score",
        formula: "s(τ,r) = ½[exp(qr·qτ)/Σ_{r′∈P} exp(qr′·qτ) + exp(qr·qτ)/Σ_{τ′∈T} exp(qr·qτ′)]",
        variables: [
          { symbol: "qτ / qr", meaning: "track memory embedding / candidate embedding" },
          { symbol: "β / γ", meaning: "match threshold 0.5 / birth confidence 0.0001" },
        ],
        intuition: "A match counts only if the candidate is the track's favorite AND the track is the candidate's favorite.",
        why: "Two-way normalization suppresses ambiguous assignments in dense, diverse open-vocabulary scenes.",
        where: "Appendix Eq. 10, Algorithm 1; per-frame association.",
        params: "Higher β reduces ID switches at the cost of fragmentation; 10-frame memory bridges occlusions.",
        simulator: "reid",
        paperIds: ["T085"],
      },
    ],
    datasets: ["others"],
    metrics: [],
    baselines: ["QDTrack", "TETer", "DeepSORT+ViLD", "Tracktor+++ViLD", "RegionCLIP+DeepSORT", "RegionCLIP+Tracktor++", "GTR", "AOA", "SORT-TAO"],
    results: [
      "TAO val open-vocabulary: base TETA 35.5 (LocA 49.3 / AssocA 36.9 / ClsA 20.2), novel 27.8 (48.8/33.6/1.5) vs TETer novel 25.7 and QDTrack 22.5 — trained on static images only, baselines use video.",
      "TAO test: base 32.6 (45.6/35.4/16.9), novel 24.1 (41.8/28.7/1.8) vs TETer novel 21.7.",
      "With RegionCLIP detector: novel ClsA 11.4 (val) / 6.1 (test), novel TETA 32.0/25.7 — best by a wide margin.",
      "Closed-set TAO val: Track mAP 15.9 (mAP50 21.2, mAP75 10.6) and TETA 34.7 (49.3/36.7/18.1), beating TETer (33.3) with a weaker ResNet50 backbone.",
      "Open-vocabulary Track mAP val: base 15.6 / novel 18.8 (mAP50 21.0/23.0); test: base 12.9 / novel 8.2 vs QDTrack novel 1.0.",
      "Zero-shot BDD100K: TETA 42.5 (41.0/36.7/49.7) vs masked TETer 36.1; approaches the BDD-trained upper bound trend.",
    ],
    ablations: [
      "CLIP distillation (Table 4): Ltext alone ClsA 15.6 → with Limage 18.1, LocA/AssocA unchanged.",
      "Hallucination (Table 5): DDPM adds +1.8 AssocA (OVTrack) / +2.4 (TETer-SwinT); +heavy augmentation adds another +2.8/+2.5; complementary to classic augmentation.",
      "Hallucination hyperparams (Table 7): without language prompt or geometric transform no gain; both required for the full +1.8.",
      "Protocol note: novel ClsA stays absolutely low (fine-grained TAO classes, top-1 ClsA metric) — classification remains the hardest axis.",
    ],
    limitations: {
      authorStated: [
        "Classification on TAO remains very challenging with low absolute novel ClsA; fine-grained errors persist (puffin misclassified as sea gull).",
        "Detection is imperfect (false negatives shown) and ID switches occur (6th qualitative row, t+3→t+4).",
        "RegionCLIP-boosted numbers rely on additional CC3M training data outside the LVIS-only protocol.",
      ],
      evident: [
        "Appearance-only association discards motion, so identically appearing nearby objects have no secondary cue.",
        "Per-image diffusion hallucination cannot synthesize true temporal phenomena (motion blur, deformation trajectories).",
      ],
    },
    assumptions: [
      "Novel class names/prompts are known at test time.",
      "LVIS rare classes are a valid proxy for arbitrary real-world novel objects.",
      "Class-agnostic RPN proposals generalize to unseen categories.",
    ],
    computation:
      "Two-stage training on LVIS (20 + 6 epochs, batch 16); one hallucinated counterpart per image; ablations on a 10k-image subset for resource reasons.",
    relations: [
      { to: "T013", type: "uses-as-baseline", note: "DeepSORT combined with ViLD is an open-vocabulary baseline in Table 1." },
      { to: "T030", type: "uses-as-baseline", note: "Tracktor++ combined with ViLD is an open-vocabulary baseline in Table 1." },
    ],
    concepts: ["mot", "tracking-by-detection", "data-association", "appearance-features", "reid", "learned-association", "cost-matrix", "benchmark-design"],
    impact:
      "Created the open-vocabulary MOT task/protocol on TAO and showed CLIP distillation plus diffusion hallucination lets a static-image-trained tracker beat video-trained closed-set trackers on both base and novel classes.",
  },
  {
    id: "T086",
    arxiv: "2305.14298",
    title: "MOTRv3: Release-Fetch Supervision for End-to-End Multi-Object Tracking",
    shortTitle: "MOTRv3",
    year: 2023,
    authors: ["En Yu", "Tiancai Wang", "Zhuoling Li", "Yuang Zhang", "Xiangyu Zhang", "Wenbing Tao"],
    fileName: "2305.14298v1.pdf",
    task: "multi-object",
    tags: ["end-to-end-tracking", "transformer", "detr", "label-assignment", "release-fetch-supervision", "pseudo-label-distillation"],
    difficulty: "advanced",
    summary:
      "MOTRv3 diagnoses MOTR's detection/association conflict as unfair label assignment: only first-frame (free) GTs supervise detect queries while all later (locked) GTs train track queries, starving detection (~40% of labels, falling over epochs). Release-Fetch Supervision matches ALL GTs against ALL queries in the first 5 decoders so labels are released to detection early and fetched back to association automatically as track queries mature. Pseudo-Label Distillation (confidence-reweighted auxiliary matching to YOLOX/Sparse R-CNN boxes, training-only) and Track Group Denoising (4 query groups with reference-point noise and attention mask) add detection/association supervision. Fully end-to-end, MOTRv3 reaches 70.4 HOTA on DanceTrack test (vs MOTR 54.2, MOTRv2 69.9 with external detector) and 60.2 on MOT17.",
    problem:
      "MOTR unifies detection and association elegantly but its detection precision is poor; prior work blamed a vague detection/association conflict and MOTRv2 patched it with an external pretrained detector, sacrificing end-to-end purity without revealing the conflict's essence.",
    background: ["mot", "tracking-by-detection", "attention", "end-to-end-mot", "hungarian", "data-association"],
    previousWork: [
      {
        name: "MOTR (detect queries + generated track queries)",
        limitation:
          "Locked-GT assignment starves detect queries (few activations, ~40%→shrinking label share), giving poor detection (DanceTrack MOTA 79.7).",
        whyThisPaper:
          "RFS rebalances assignment without architecture change: +10.2 MOTA / +7.4 DetA on DanceTrack val.",
      },
      {
        name: "MOTRv2 (external YOLOX proposals + post-processing)",
        limitation:
          "Needs a separately trained detector at inference plus MOT17 post-processing (without it: 57.6 HOTA / 70.1 MOTA); not end-to-end and scales poorly with backbone size.",
        whyThisPaper:
          "MOTRv3 beats it fully end-to-end (70.4 vs 69.9 DanceTrack; 60.2 vs 57.6* MOT17) and keeps improving when scaling ResNet-50→ConvNeXt-Base.",
      },
      {
        name: "Tracking-by-detection MOT (ByteTrack, OC-SORT, FairMOT, TransTrack)",
        limitation:
          "Strong via hand-crafted association/NMS post-processing but cannot propagate queries end-to-end across frames.",
        whyThisPaper:
          "Shows the end-to-end paradigm can match/exceed them (92.9 MOTA DanceTrack) once supervision is fixed.",
      },
    ],
    researchGap:
      "Before this paper, nobody had identified the assignment mechanism (free vs locked GTs) behind MOTR's conflict, so all fixes required external detectors or post-processing.",
    contribution: [
      "Diagnosis: locked-GT analysis (Fig. 2) showing detect-query activation scarcity and the 60/40→worsening label split, with RFS dynamics proving automatic release-then-fetch.",
      "Release-Fetch Supervision: all-GT↔all-query Hungarian matching in decoders 1–5, MOTR matching only in decoder 6.",
      "Pseudo-Label Distillation: training-only auxiliary matching to YOLOX/Sparse R-CNN boxes (conf ≥0.05) with confidence reweighting (0.5 for background).",
      "Track Group Denoising: 4 track-query groups sharing assignment, reference-point noise, inter-group attention mask (Eq. 4).",
      "MOTRv3: end-to-end SOTA on DanceTrack (70.4 HOTA) and competitive MOT17 (60.2) with no external detector or post-processing.",
    ],
    method: {
      pipeline: ["encode", "match-all", "detect", "propagate-tracks", "distill-pseudo", "denoise-groups", "decode"],
      architecture:
        "MOTR on Deformable-DETR (ConvNeXt-Base, COCO-pretrained): backbone + 6 encoders + 6 decoders; detect queries spawn track queries above 0.5 confidence; training-only changes (RFS/PLD/TGD), inference identical to MOTR.",
      motionModel: "Implicit — track queries propagate boxes across frames; no Kalman or explicit motion module.",
      appearanceModel: "Implicit query embeddings; no ReID network.",
      association:
        "Query-identity propagation (Eq. 2) plus Hungarian matching per decoder; TGD stabilizes association (+2.7 AssA).",
      detectionDependency:
        "None at inference (end-to-end); pretrained YOLOX/Sparse R-CNN used ONLY offline for PLD pseudo-labels.",
      trackManagement:
        "Track queries generated from confident detect queries; queries of instances gone for consecutive frames removed (Ψ process).",
      loss:
        "Per-frame L = 2·Lfocal + 5·Ll1 + 2·Lgiou; clip loss averages RFS/PLD/TGD matching losses over T=5-frame clips normalized by object count (Eq. 5).",
      optimization:
        "Adam, lr 2e-4; batch 8 clips × 5 frames (sampling interval 1–10); DanceTrack 5 epochs (÷10 at 4th), MOT17 50 epochs (÷10 at 40th).",
    },
    equations: [
      {
        id: "motrv3-frame-loss",
        label: "Per-frame detection loss",
        formula: "L = λcls·Lcls + λl1·Ll1 + λgiou·Lgiou, λcls = 2, λl1 = 5, λgiou = 2",
        variables: [
          { symbol: "Lcls", meaning: "focal loss for query classification" },
          { symbol: "Ll1 / Lgiou", meaning: "L1 and GIoU box regression losses" },
        ],
        intuition: "Each query is punished for wrong labels and rewarded for tight boxes, with box accuracy weighted heaviest.",
        why: "Common currency under all three matchings (RFS/PLD/TGD) so supervision strategies compose additively.",
        where: "Section 3.1; applied per frame inside the clip loss.",
        paperIds: ["T086"],
      },
      {
        id: "motrv3-detect-match",
        label: "RFS detection matching",
        formula: "σ̂d = argmin_{σ∈S} Σ_j L(dj, ŷ_{σ(j)}); decoders 1–5: S = Sa (ALL GTs ↔ ALL queries); decoder 6: S = Sd (free GTs ↔ detect queries)",
        variables: [
          { symbol: "ŷd / ŷt", meaning: "free GTs (new targets) / locked GTs (previously seen targets)" },
          { symbol: "Sa / Sd", meaning: "expanded all-to-all / original detect-only matching spaces" },
          { symbol: "dj", meaning: "box decoded from query j" },
        ],
        intuition: "Let every label audition for every query in early decoders; the best-fitting query — detect or track — wins the supervision.",
        why: "Early in training track queries mislocalize so labels flow to detect queries; as tracks mature labels return automatically (Fig. 2d).",
        where: "Section 3.2, Eq. 1 modified; association matching (Eq. 2) unchanged.",
        params: "Keeping decoder 6 MOTR-style preserves the end-to-end detect/track division of labor.",
        simulator: "hungarian",
        paperIds: ["T086"],
      },
      {
        id: "motrv3-track-propagate",
        label: "Track-query label propagation",
        formula: "σt^i = Ψ(σt^{i−1}, σ̂d^{i−1,L})",
        variables: [
          { symbol: "σt^i", meaning: "locked-GT↔track-query assignment in frame i" },
          { symbol: "σ̂d^{i−1,L}", meaning: "final-decoder detect matching of the previous frame" },
          { symbol: "Ψ", meaning: "spawning/removal operator (new tracks from detections, drop long-gone instances)" },
        ],
        intuition: "Once a detect query claims a target, its descendant track query inherits the claim in all future frames.",
        why: "This inheritance is exactly what locks labels away from detection — the mechanism RFS counterbalances.",
        where: "Section 3.2, Eq. 2; association branch, unchanged by RFS.",
        paperIds: ["T086"],
      },
      {
        id: "motrv3-pld",
        label: "Confidence-reweighted pseudo-label loss",
        formula: "Lσp = Σ_j ωj·L(dj, ỹ_{σp(j)}), ωj = ce if matched else 0.5",
        variables: [
          { symbol: "ỹ", meaning: "pseudo boxes from frozen YOLOX/Sparse R-CNN (conf ≥ 0.05), diverse vs GT" },
          { symbol: "ce", meaning: "detector confidence of the matched pseudo box" },
          { symbol: "σp", meaning: "one-to-one matching between all queries and pseudo labels, all 6 decoders" },
        ],
        intuition: "Borrow extra homework from a pretrained detector, trusting confident examples fully and background half.",
        why: "Pseudo boxes are diverse and carry confidences GTs lack; beats training on copied GTs (61.6/61.7 vs 60.3 HOTA).",
        where: "Section 3.3, Eq. 3; training-only, detector discarded at inference.",
        params: "Threshold 0.05 filters noise; parallel YOLOX+SparseRCNN union helps DetA but confuses association.",
        paperIds: ["T086"],
      },
      {
        id: "motrv3-mask",
        label: "TGD attention mask",
        formula: "aij = 1 if (i < M+N, j ≥ M+N) or (i,j ≥ M+N, ⌊(i−M−N)/N⌋ ≠ ⌊(j−M−N)/N⌋); else 0; S = G·N + M",
        variables: [
          { symbol: "M / N / G", meaning: "detect queries / track queries / groups (G = 4)" },
          { symbol: "aij = 1", meaning: "blocked attention between the query pair (information leak prevention)" },
        ],
        intuition: "Give each track query 4 noisy twins that practice independently without copying each other's answers.",
        why: "One-to-many supervision plus reference-point noise desensitizes association to initialization quality (+2.7 AssA).",
        where: "Section 3.4, Eq. 4; decoder self-attention during training.",
        params: "4 groups optimal (63.7); 3/5 worse; added noise gives the final 63.9.",
        paperIds: ["T086"],
      },
      {
        id: "motrv3-clip",
        label: "Multi-frame clip loss",
        formula: "Lclip = Σ_{i=1..T} (Lσr + Lσp + Lσg)/Oi",
        variables: [
          { symbol: "Lσr / Lσp / Lσg", meaning: "RFS / PLD / TGD matching losses in frame i" },
          { symbol: "T / Oi", meaning: "clip length (5) / object count in frame i" },
        ],
        intuition: "Grade the whole short movie, not single frames, normalizing each frame by its crowd size.",
        why: "Multi-frame training teaches query propagation; per-frame normalization prevents crowded frames dominating.",
        where: "Section 3.4, Eq. 5.",
        paperIds: ["T086"],
      },
    ],
    datasets: ["dancetrack", "mot17"],
    metrics: ["hota", "assa", "deta", "mota", "idf1", "idsw"],
    baselines: ["MOTR", "MOTRv2", "TransTrack", "ByteTrack", "QDTrack", "FairMOT", "CenterTrack", "OC-SORT"],
    results: [
      "DanceTrack test: HOTA 70.4 / AssA 59.3 / DetA 83.8 / MOTA 92.9 / IDF1 72.3 vs MOTR 54.2/40.2/73.5/79.7/51.5 and MOTRv2 69.9/59.0/83.0/91.9/71.7 — end-to-end (✓) vs MOTRv2 ✗.",
      "MOT17 test: HOTA 60.2 / AssA 58.7 / DetA 62.1 / MOTA 75.9 / IDF1 72.4 / IDS 2403 vs MOTR 57.8/73.4/68.6/2439; MOTRv2 without post-processing collapses to 57.6/70.1.",
      "Val ablation (Table 3): base 56.6 HOTA → full MOTRv3 63.9 (+7.3), MOTA 75.3→86.8 (+11.5); RFS alone +10.2 MOTA/+7.4 DetA; PLD alone +9.6 MOTA/+7.1 DetA; TGD +2.7 AssA/+2.1 IDF1.",
      "PLD variants (Table 4): YOLOX 61.6 / Sparse R-CNN 61.7 HOTA beat copied-GT 60.3; parallel union raises DetA (77.1) but lowers AssA.",
      "TGD groups (Table 5): 4 groups peak (63.7); +reference noise reaches 63.9/53.5/76.7/86.8/67.2 with IDS 1151.",
      "Scaling (Fig. 4): MOTRv3 improves ResNet-50→ConvNeXt-T/S/B monotonically; MOTRv2 degrades since only its association half scales.",
    ],
    ablations: [
      "RFS dynamics (Fig. 2): detect-label share boosted early, converging to original split late — evidence of automatic fetch-back.",
      "RFS+PLD combine super-additively (61.7) since RFS fixes the ratio and PLD adds label quantity/diversity.",
      "Group count sweep 3/4/5 and noise on/off isolates TGD's association gain.",
      "Detector-choice study (YOLOX vs Sparse R-CNN vs parallel vs GT) for PLD.",
    ],
    limitations: {
      authorStated: [
        "While RFS mitigates the conflict in terms of supervision, the trade-off between detection and association remains unresolved; how to disentangle the two sub-tasks deserves further study (Sec. 6).",
      ],
      evident: [
        "PLD still depends on external pretrained detectors and extra detection data at training time, so training is not detector-free.",
        "All SOTA claims use large ConvNeXt-Base backbones; the RFS mechanism's benefit at small-backbone / real-time budgets is not shown.",
      ],
    },
    assumptions: [
      "Early-decode similarity is a reliable router of labels between detect and track queries.",
      "Frozen-detector pseudo boxes (conf ≥0.05) are sufficiently precise to supervise detection.",
    ],
    computation:
      "ConvNeXt-Base Deformable-DETR backbone; batch 8 × 5-frame clips; DanceTrack 5 epochs / MOT17 50 epochs.",
    relations: [
      { to: "T059", type: "builds-on", note: "Keeps MOTR architecture/training; fixes its locked-GT assignment." },
      { to: "T080", type: "improves", note: "Beats MOTRv2 (69.9→70.4 DanceTrack) with no inference-time detector." },
      { to: "T061", type: "uses-as-baseline", note: "ByteTrack tabulated on both benchmarks (47.7/63.1 HOTA). " },
      { to: "T073", type: "uses-as-baseline", note: "OC-SORT tabulated (55.1 DanceTrack)." },
      { to: "T053", type: "uses-as-baseline", note: "TransTrack tabulated (45.5/54.1 HOTA)." },
      { to: "T043", type: "uses-as-baseline", note: "FairMOT tabulated (59.3 HOTA both benchmarks)." },
      { to: "T063", type: "uses-as-baseline", note: "Evaluated on the DanceTrack benchmark under its protocol." },
      { to: "T006", type: "uses-as-baseline", note: "MOT17 evaluation follows the MOT16/MOT17 benchmark protocol family." },
    ],
    concepts: ["end-to-end-mot", "attention", "hungarian", "data-association", "mot", "mota", "hota", "cost-matrix"],
    impact:
      "Reframed MOTR's conflict as a label-assignment problem with the free/locked-GT analysis, giving the field a detector-free fix (RFS) plus reusable PLD/TGD supervision tricks for query-based trackers.",
  },
  {
    id: "T087",
    arxiv: "2305.15896",
    title: "MixFormerV2: Efficient Fully Transformer Tracking",
    shortTitle: "MixFormerV2",
    year: 2023,
    authors: ["Yutao Cui", "Tianhui Song", "Gangshan Wu", "Limin Wang"],
    venue: "NeurIPS 2023",
    fileName: "2305.15896v2.pdf",
    task: "single-object",
    tags: ["single-object-tracking", "transformer", "one-stream", "efficient-tracking", "knowledge-distillation", "model-pruning"],
    difficulty: "advanced",
    summary:
      "MixFormerV2 removes all dense convolutions and the complex SPM score module from one-stream tracking: four learnable prediction tokens join template/search tokens in plain ViT mixed attention (P-MAM) and directly regress box-coordinate distributions plus a quality score through tiny MLP heads (token head: 166 vs 120 FPS over pyramidal corner at higher accuracy). A distillation reduction paradigm — dense-to-sparse logits distillation from MixViT corner marginals, then deep-to-shallow progressive depth pruning with feature mimicking, intermediate teacher, and MLP-ratio reduction — yields MixFormerV2-B (8 layers, 58.8M, LaSOT 70.6 @165 FPS) and MixFormerV2-S (4 layers, 16.2M, LaSOT 60.6 @325 FPS GPU / 30 FPS CPU, +2.7 over FEAR-L).",
    problem:
      "One-stream Transformer trackers (MixFormer/MixViT) are accurate but undeployable: the pyramidal corner head runs ten conv layers on high-resolution maps and the online-update score module needs RoI pooling plus attention blocks; designing a fresh lightweight one-stream backbone is blocked by the cost of large-scale pretraining.",
    background: ["sot", "siamese", "attention", "bounding-box", "success-plot"],
    previousWork: [
      {
        name: "MixFormer/MixViT (mixed attention + pyramidal corner head + SPM)",
        limitation:
          "Dense corner head (~33% of latency) and SPM score module (−13% speed) bottleneck inference; accuracy depends on them.",
        whyThisPaper:
          "Four prediction tokens + MLP heads replace both at higher speed (Table 1: 90→166 FPS) and distilled accuracy (70.6 LaSOT).",
      },
      {
        name: "Efficient trackers (LightTrack, FEAR, HCAT, E.T.Track, STARK-Lightning)",
        limitation:
          "Siamese/dual-branch efficient designs top out at ~59 AUC on LaSOT and none brings one-stream accuracy to CPU real-time.",
        whyThisPaper:
          "MixFormerV2-S is the first CPU real-time one-stream tracker (30 FPS) at 60.6 AUC, +2.7 over FEAR-L.",
      },
      {
        name: "ViT compression (token pruning, NAS, MiniViT, ViTKD)",
        limitation:
          "Generic ViT shrinkers ignore tracking's pretraining dependence and the dense-to-sparse head mismatch.",
        whyThisPaper:
          "Progressive depth pruning keeps student/teacher distributions aligned (64.8 vs 62.9 MAE-init) and marginal-KL bridges dense teachers to token students.",
      },
    ],
    researchGap:
      "Before this paper, no fully Transformer tracker (zero convolution, zero complex score module) existed, and no distillation recipe transferred dense corner-head knowledge into token regression while pruning depth without fresh pretraining.",
    contribution: [
      "Prediction-token-involved mixed attention (P-MAM): 4 learnable tokens concatenated with template/search tokens capture target–search correlation as compact representations for regression and scoring.",
      "Distribution-based token regression (Eq. 2–3) with weight-shared MLP plus mean-pooled MLP score head, removing all convolutions and SPM.",
      "Dense-to-sparse distillation: MixViT 2D corner joints → 1D coordinate marginals (Eq. 4) as KL soft labels (Eq. 5).",
      "Deep-to-shallow distillation: progressive depth pruning with cosine decay γ (Eqs. 8–9), feature mimicking (Eq. 6), intermediate teacher, and MLP-ratio reduction for CPU.",
      "Two instantiations: B (70.6 LaSOT @165 FPS) and S (60.6 @325/30 FPS), SOTA on 6 benchmarks incl. VOT2022 EAO 0.556.",
    ],
    method: {
      pipeline: ["tokenize", "mix-attend", "regress-tokens", "score-tokens", "distill", "prune", "update-online"],
      architecture:
        "Plain ViT backbone of N P-MAM layers (asymmetric attention as MixFormer) over multi-template + search + 4 prediction tokens; weight-shared MLP head per coordinate token; 2-layer MLP score head on mean token. B: 8 layers, 288×288 search / 128×128 template, 58.8M. S: 4 layers, MLP ratio 1.0, 224/112, 16.2M. Online: templates updated every 200 frames by top score.",
      appearanceModel:
        "First + dynamic templates fused with search via mixed attention; no online filter; target information compressed into prediction tokens.",
      detectionDependency: "None — single-object tracker, no detector.",
      trackManagement:
        "Search cropped around last prediction; Hanning-style post-processing inherited from MixFormer pipeline; score-gated dynamic template selection.",
      loss:
        "L = 5·L1 + 2·LCIoU + 1·Llog(KL) + 0.2·Lfeat(L2) (Eq. 10); score MLP trained 50 extra epochs.",
      optimization:
        "AdamW wd 1e-4, lr 1e-4→1e-5 after 400 epochs; 500-epoch distillation stages (m=40 progressive) on TrackingNet/LaSOT/GOT-10k/COCO, batch 256 on 8× RTX 8000.",
    },
    equations: [
      {
        id: "mixv2-pmam",
        label: "Prediction-token-involved mixed attention",
        formula: "ktse = Concat(kt,ks,ke), vtse = Concat(vt,vs,ve); Atten_t = SoftMax(qt·ktseᵀ/√d)·vtse (same for s, e)",
        variables: [
          { symbol: "t/s/e", meaning: "template / search / prediction-token queries, keys, values" },
          { symbol: "Atten_e", meaning: "prediction-token output aggregating target–search correlation" },
          { symbol: "d", meaning: "per-head dimension" },
        ],
        intuition: "Four blank delegate tokens sit in the meeting of template and search tokens and take notes on where the target is.",
        why: "Compresses dense correlation into 4 tokens so MLP heads — not conv maps — can regress boxes (90→166 FPS).",
        where: "Section 3.1, Eq. 1; every P-MAM layer, asymmetric scheme for online efficiency.",
        paperIds: ["T087"],
      },
      {
        id: "mixv2-head",
        label: "Token-based distribution and score heads",
        formula: "P̂X(x) = MLP(tokenX), X ∈ {T,L,B,R}; s = MLP(mean(tokenX))",
        variables: [
          { symbol: "P̂X", meaning: "predicted probability density of coordinate X (top/left/bottom/right)" },
          { symbol: "s", meaning: "target quality score for online sample selection" },
          { symbol: "MLP", meaning: "weight-shared head across the four coordinate tokens" },
        ],
        intuition: "Instead of one exact number per box side, predict a full hunch distribution per side plus a confidence gut feeling.",
        why: "Distributions enable KL distillation from dense teachers and beat direct regression (T4 67.5 vs T1 63.1); score head adds +1.7 AUC at ~zero cost.",
        where: "Section 3.1, Eq. 2; heads on final-layer prediction tokens.",
        paperIds: ["T087"],
      },
      {
        id: "mixv2-expect",
        label: "Box from distribution expectation",
        formula: "BX = E_{P̂X}[X] = ∫ x·P̂X(x) dx",
        variables: [{ symbol: "BX", meaning: "final predicted box coordinate" }],
        intuition: "The box side is the average of the hunch, weighted by how strongly each position is believed.",
        why: "Differentiable read-out (following LD-style detection) compatible with L1/CIoU supervision.",
        where: "Section 3.2.1, Eq. 3.",
        paperIds: ["T087"],
      },
      {
        id: "mixv2-marginal",
        label: "Dense corner joints to 1D marginals",
        formula: "PT(x) = ∫ PTL(x,y)dy; PL(y) = ∫ PTL(x,y)dx; PB, PR likewise from PBR",
        variables: [
          { symbol: "PTL / PBR", meaning: "teacher's 2D top-left / bottom-right corner distributions" },
          { symbol: "PT, PL, PB, PR", meaning: "1D student-compatible coordinate distributions" },
        ],
        intuition: "Flatten the teacher's 2D corner heatmaps onto each axis to get answer keys the token student can read.",
        why: "Bridges the dense↔sparse representation gap so MixViT logits become valid soft labels.",
        where: "Section 3.2.1, Eq. 4.",
        paperIds: ["T087"],
      },
      {
        id: "mixv2-kl",
        label: "Dense-to-sparse logits distillation",
        formula: "Llog = Σ_{X∈{T,L,B,R}} LKL(P̂X, PX)",
        variables: [
          { symbol: "PX", meaning: "teacher marginal distribution for coordinate X" },
          { symbol: "LKL", meaning: "KL-divergence between student and teacher distributions" },
        ],
        intuition: "Grade the student not just on the right answer but on matching the teacher's full pattern of uncertainty.",
        why: "Lifts the 12-layer token model 67.5→68.9 (MixViT-B) / →69.6+ (MixViT-L) without inference cost.",
        where: "Section 3.2.1, Eq. 5; λ3 = 1.",
        params: "Larger teacher (MixViT-L 71.5) distills better (+2.2) than base (69.0, +1.4).",
        paperIds: ["T087"],
      },
      {
        id: "mixv2-feat",
        label: "Feature mimicking loss",
        formula: "Lfeat = Σ_{(i,j)∈M} L2(Fi^S, Fj^T)",
        variables: [
          { symbol: "Fi^S / Fj^T", meaning: "student layer-i / teacher layer-j feature maps" },
          { symbol: "M", meaning: "matched supervised layer pairs" },
        ],
        intuition: "Make the student's intermediate thoughts resemble the teacher's at matched depths.",
        why: "Adds +0.4 AUC over logits-only (62.4→62.9) in the 4-layer compression study.",
        where: "Section 3.2.2, Eq. 6; λ4 = 0.2.",
        paperIds: ["T087"],
      },
      {
        id: "mixv2-prune",
        label: "Progressive depth pruning",
        formula: "xi = γ(FFN(ATTN(x)+x) + ATTN(x)) + x, i ∈ E; γ(t) = 0.5(1 + cos(πt/m)) for t ≤ m, else 0",
        variables: [
          { symbol: "E", meaning: "set of layers to eliminate" },
          { symbol: "γ", meaning: "cosine decay rate from 1 → 0 over m = 40 epochs" },
          { symbol: "xi", meaning: "output of layer i (LN omitted)" },
        ],
        intuition: "Fade condemned layers to ghosts that pass input through unchanged, instead of amputating them outright.",
        why: "Keeps student/teacher representations aligned from identical initialization: 64.8 vs 62.9 (MAE-init) / 64.4 (skip-init); eliminated layers become identity and are deleted.",
        where: "Section 3.2.2, Eqs. 7–9, Fig. 4.",
        params: "m ≥ 40 saturates (Table 3g); MLP weights pruned by slicing w′ = w[:d1′,:d2′] then distilled.",
        paperIds: ["T087"],
      },
      {
        id: "mixv2-total",
        label: "Joint distillation loss",
        formula: "L = 5·L1(B^S,B^gt) + 2·LCIoU(B^S,B^gt) + 1·Llog(S,T) + 0.2·Lfeat(S,T)",
        variables: [
          { symbol: "B^S / B^gt", meaning: "student box / ground-truth box" },
          { symbol: "S / T", meaning: "student / teacher (optionally via 8-layer intermediate teacher)" },
        ],
        intuition: "Learn the truth from labels and the craft from the teacher, with boxes mattering most.",
        why: "Unifies GT supervision with both distillation channels; intermediate teacher bridges 12→4-layer gap (+0.7).",
        where: "Section 3.3, Eq. 10.",
        paperIds: ["T087"],
      },
    ],
    datasets: ["lasot", "got10k", "trackingnet", "uav123", "others"],
    metrics: ["success-auc", "norm-precision", "precision", "eao", "fps"],
    baselines: ["MixFormer", "MixViT", "OSTrack", "SimTrack", "TransT", "STARK", "SwinTrack", "ToMP", "FEAR", "HCAT", "E.T.Track", "LightTrack"],
    results: [
      "LaSOT: B 70.6 AUC / 80.8 PNorm / 76.2 P @165 FPS; B* (MixViT-B teacher) 69.5; S 60.6/69.9/60.4 @325 FPS GPU / 30 FPS CPU vs FEAR-L 57.9 (+2.7).",
      "TrackingNet: B 83.4/88.1/81.6; TNL2k: B 57.4/58.4; LaSOText: B 50.6/56.9; UAV123: B 69.9/92.1.",
      "VOT2022: B EAO 0.556 / Acc 0.795 / Robustness 0.851 (best EAO/robustness); S 0.431/0.715/0.757.",
      "Head study (Table 1/3a): token head 166 FPS vs pyramidal corner 90 FPS; T4-distribution 67.5 vs T1-direct 63.1 vs corner 69.0 AUC.",
      "Distillation (Table 3c–f): dense-to-sparse +1.4 (base) / +2.2 (large teacher); PMDP 64.8 vs 62.9 MAE-init; intermediate teacher +0.7; score head +1.7 AUC.",
      "Pruning routes (3h–i): 12→8 layers minimal loss (68.9→68.5/66.6); 4-layer + MLP-ratio-1.0 reaches 59.4→60.6 final S.",
    ],
    ablations: [
      "Regression method (3a), score prediction (3b: token MLP +1.7 free vs SPM −13% speed), teacher size (3c).",
      "Logits vs +feature mimicking (3d), init methods incl. skipped-layers (3e), intermediate teacher (3f), eliminating epochs m (3g).",
      "Full pruning routes to B (3h) and S (3i) with head/MLP-ratio variants.",
    ],
    limitations: {
      authorStated: [
        "The token head still trails the pyramidal corner head slightly at equal depth (12-layer: 68.9/67.7 vs 69.0/68.2 AUC) — speed is bought with a small accuracy gap closed only by distilling a larger teacher.",
        "Extreme 4-layer compression requires an 8-layer intermediate teacher bridge plus MLP-ratio reduction, i.e. a multi-stage pipeline rather than one-step pruning.",
      ],
      evident: [
        "The whole recipe depends on a large pretrained MixViT teacher and 500-epoch distillation stages — efficiency is inference-only, not training.",
        "Online template update still relies on the learned score head with a fixed 200-frame interval, unexamined under fast appearance change.",
      ],
    },
    assumptions: [
      "Teacher corner-joint marginals are faithful soft targets for token distributions.",
      "Fading layers to identity preserves representational alignment well enough for feature mimicking.",
    ],
    computation:
      "Distillation on 8× RTX 8000, 500 epochs/stage (m=40), batch 256; inference GPU RTX 8000 + Xeon Gold 6230R CPU. Params 58.8M (B) / 16.2M (S).",
    relations: [
      { to: "T070", type: "builds-on", note: "Extends MixFormer/MixViT mixed attention with prediction tokens; distills from MixViT teachers." },
      { to: "T071", type: "uses-as-baseline", note: "OSTrack-256 beaten 69.1→70.6 LaSOT at 105→165 FPS." },
      { to: "T055", type: "uses-as-baseline", note: "TransT compared in Table 5 (64.9 vs 70.6 LaSOT)." },
      { to: "T024", type: "uses-as-baseline", note: "Trained and evaluated on LaSOT; headline numbers are LaSOT AUC." },
      { to: "T025", type: "uses-as-baseline", note: "Trained on GOT-10k alongside LaSOT/TrackingNet/COCO." },
    ],
    concepts: ["sot", "attention", "siamese", "appearance-features", "success-plot", "bounding-box"],
    impact:
      "First fully-Transformer (convolution-free) tracker with token distribution regression, plus a reusable dense-to-sparse / progressive-pruning distillation recipe that made one-stream tracking CPU real-time.",
  },
  {
    id: "T088",
    arxiv: "2306.05238",
    title: "SparseTrack: Multi-Object Tracking by Performing Scene Decomposition based on Pseudo-Depth",
    shortTitle: "SparseTrack",
    year: 2023,
    authors: ["Zelin Liu", "Xinggang Wang", "Cheng Wang", "Wenyu Liu", "Xiang Bai"],
    fileName: "2306.05238v2.pdf",
    task: "multi-object",
    tags: ["multi-object-tracking", "tracking-by-detection", "iou-matching", "pseudo-depth", "scene-decomposition", "depth-cascade-matching"],
    difficulty: "intermediate",
    summary:
      "SparseTrack observes occlusion order follows depth order, so it decomposes dense crowds into sparse depth subsets before association. Pseudo-depth Lp = H − yp (bottom-of-box to image-bottom distance) estimates relative depth from two scene priors (camera above ground, flat ground). Depth Cascade Matching (DCM) splits detections and Kalman-predicted tracks into k pseudo-depth intervals and runs IoU-Hungarian association near-to-far, carrying unmatched objects to the next level; it wraps ByteTrack's high-then-low-score stages. IoU-only, SparseTrack reaches 65.1/81.0/80.1 (HOTA/MOTA/IDF1) on MOT17, 63.4/78.2/77.3 on MOT20, and 55.5/91.3/58.3 on DanceTrack (+7.8 HOTA over ByteTrack), with DCM plug-and-play into other trackers.",
    problem:
      "Dense crowds and frequent occlusions break association: ByteTrack's low-score batch matching collides trajectories with similar x-y positions, while attention/graph/memory alternatives pay heavy compute; appearance cues are unreliable under occlusion and motion blur.",
    background: ["mot", "tracking-by-detection", "data-association", "iou", "cost-matrix", "hungarian"],
    previousWork: [
      {
        name: "ByteTrack (high/low-score split + IoU association)",
        limitation:
          "Low-score detections are still crowded among themselves, so simultaneous IoU matching misassigns occluded targets (Fig. 1).",
        whyThisPaper:
          "DCM further decomposes each score band by pseudo-depth: +2.0 HOTA MOT17, +2.1 MOT20, +7.8 DanceTrack with the same detector.",
      },
      {
        name: "Appearance/ReID trackers (DeepSORT, FairMOT, BoT-SORT-ReID) and graph/attention trackers (MOTR, TransTrack, GSDT)",
        limitation:
          "Occlusion and blur corrupt appearance; attention/graph temporal modeling costs scale steeply with crowd size.",
        whyThisPaper:
          "Pure geometry (pseudo-depth + IoU) matches their SOTA using no appearance features at all.",
      },
      {
        name: "Set-decomposition precedents (DeepSORT cascade, FairMOT cue split, LMGP 3D pre-clustering, DP-MOT SODE depth)",
        limitation:
          "Decompositions use confidence/appearance/stage heuristics or unsupervised depth sorting — none splits dense sets by explicit relative depth order.",
        whyThisPaper:
          "Pseudo-depth gives a cheap, explicit near-to-far ordering that makes each association step sparse.",
      },
    ],
    researchGap:
      "Before this paper, no method used depth information to decompose the association target set; occlusion handling meant better features or heavier temporal models, not sparser matching problems.",
    contribution: [
      "Pseudo-depth method: relative depth from 2D box geometry under two scene priors, Lp = H − yp, sufficient to order objects near-to-far.",
      "Depth Cascade Matching (Algorithm 1): uniform k-interval split of detection/track pseudo-depth ranges, per-level IoU-Hungarian matching with carryover of unmatched objects to deeper levels.",
      "SparseTrack (Algorithm 2): ByteTrack-style high-score DCM → Kalman update → low-score DCM with finer granularity (1–2 levels high, 4–8+ low; defaults 3/8/12 per dataset) plus fast online GMC.",
      "IoU-only SOTA-comparable results on MOT17/MOT20/DanceTrack and plug-and-play gains in any tracker.",
    ],
    method: {
      pipeline: ["detect", "pseudo-depth", "predict", "dcm-high", "dcm-low", "birth", "kill"],
      architecture:
        "Tracking-by-detection with frozen YOLOX detector (ByteTrack weights/NMS): per-frame boxes + scores → pseudo-depth per box → KF-predicted tracks → DCM on high-score subset (τ=0.6) → DCM on low-score subset (≥0.1) for unmatched active tracks → new births from unmatched high-score, deaths after 30/60/60 lost frames.",
      motionModel:
        "Standard Kalman filter (constant velocity) for track prediction; fast online global motion compensation (GMC) aligns frames before depth-level matching.",
      association:
        "IoU-distance cost + Hungarian matching per depth level (threshold τ), near-to-far with unmatched carryover; no appearance, no learned affinity.",
      detectionDependency:
        "Fully detector-bound: same YOLOX as ByteTrack; association quality hinges on detecting occluded low-confidence targets.",
      trackManagement:
        "Lost-track buffers 30 (MOT17) / 60 (MOT20, DanceTrack) frames; no track confirmation changes vs baseline.",
      optimization: "No training — inference-only association over frozen detections.",
    },
    equations: [
      {
        id: "sparsetrack-pseudo-depth",
        label: "Pseudo-depth projection",
        formula: "Lp = H − yp",
        variables: [
          { symbol: "yp", meaning: "y-pixel of projection point P1 (ground-contact point P0 projected to image plane)" },
          { symbol: "H", meaning: "image height in pixels; bottom edge Lb is the reference line" },
          { symbol: "Lp", meaning: "pseudo-depth: distance from box bottom to image bottom, larger = farther" },
          { symbol: "D", meaning: "foot of perpendicular from P1 to Lb" },
        ],
        intuition: "Things whose feet touch the image lower down are closer; feet higher up are farther — read depth off the ground contact.",
        why: "Orders occluders before occludees (Fig. 2) with zero learned parameters, replacing true 3D depth for association purposes.",
        where: "Section 3.2, Eq. 1, Fig. 5; computed per detection and per predicted track.",
        params: "Valid only under the two priors: camera above ground, all objects on one flat plane.",
        paperIds: ["T088"],
      },
      {
        id: "sparsetrack-partition",
        label: "Depth-interval scene decomposition",
        formula: "{I0..I_{k−1}} = split([Dmin, Dmax], k); Di = {d : d.depth ∈ Ii}; Ti likewise",
        variables: [
          { symbol: "Dmin / Dmax", meaning: "min/max pseudo-depth of the detection (or track) set" },
          { symbol: "k", meaning: "number of depth levels (high-score 1–2; low-score 4–8; defaults 3/8/12)" },
          { symbol: "Di / Ti", meaning: "detection / track subset at depth level i" },
        ],
        intuition: "Slice the crowd into near-to-far shelves so each shelf holds few, well-separated people.",
        why: "Sparsity per matching step removes x-y collisions between depths; gains saturate past ~7 levels (Table 4).",
        where: "Section 3.3, Algorithm 1 lines 2–27; uniform split of each set's own range.",
        params: "Finer k helps low-score dense occlusion; excessive k over-sparsifies with no further gain.",
        paperIds: ["T088"],
      },
      {
        id: "sparsetrack-cascade",
        label: "Depth-cascade IoU association",
        formula: "Ci = IoU-dist(Ti ∪ T0, Di ∪ D0); (Tmatched, T0, D0) = Hungarian(Ci, τ), levels 0 → k−1",
        variables: [
          { symbol: "Ci", meaning: "IoU-distance cost matrix at depth level i" },
          { symbol: "T0 / D0", meaning: "unmatched tracks/detections carried into the next deeper level" },
          { symbol: "τ", meaning: "matching threshold (high/low score split at 0.6 / 0.1)" },
        ],
        intuition: "Seat the front row first, then let the still-standing find seats further back — nobody competes across rows.",
        why: "Near-first ordering matches occlusion order; carryover keeps recipients flexible across levels (Fig. 3).",
        where: "Section 3.3–3.4, Algorithm 1 lines 28–35 inside Algorithm 2; high-score DCM then low-score DCM.",
        params: "Low-score threshold lowered to 0.01 in ablations to stress-test dense occlusion handling.",
        simulator: "iou-track",
        paperIds: ["T088"],
      },
    ],
    datasets: ["mot17", "mot20", "dancetrack"],
    metrics: ["hota", "mota", "idf1", "assa", "deta", "idsw", "fp", "fn"],
    baselines: ["ByteTrack", "SORT", "BoT-SORT", "OC-SORT", "MOTR", "TransTrack", "FairMOT", "CenterTrack", "QDTrack", "StrongSORT"],
    results: [
      "MOT17 test: HOTA 65.1 / MOTA 81.0 / IDF1 80.1, FP 23904, FN 81927, IDS 1170 @19.9 FPS vs ByteTrack 63.1/80.3/77.3/2196 (+2.0/+0.7/+2.8, IDS nearly halved).",
      "MOT20 test: 63.4 / 78.2 / 77.3, IDS 1116 vs ByteTrack 61.3/77.8/75.2/1223 (+2.1/+0.4/+2.1).",
      "DanceTrack test: HOTA 55.5 / MOTA 91.3 / IDF1 58.3 / AssA 39.1 / DetA 78.9 vs ByteTrack 47.7/89.6/53.9 (+7.8/+1.7/+4.4/+7.0/+7.9).",
      "Val (Table 6, uniform settings): MOT17 69.2/72.3/81.4/76.8; MOT20 69.9/67.8/84.5/86.1; DanceTrack 53.8/38.5/56.0/87.1 — best of SORT/ByteTrack/BoT-SORT/OC-SORT on MOT17+MOT20.",
      "Depth levels (Table 4): MOT20 HOTA 68.6→68.9, DanceTrack 52.6→53.9 as k goes 1→7/5; saturates beyond.",
      "GMC (Table 5): MOT17 HOTA 67.9→69.2 with GMC; MOT20 unchanged (little camera motion).",
    ],
    ablations: [
      "Pseudo-depth level sweep 1/2/5/7/9 on MOT20 + DanceTrack val (Table 4) with low-score threshold 0.01.",
      "Input resolution sweep (Fig. 6): detector mAP saturates but both trackers' association keeps improving; SparseTrack improves more than ByteTrack.",
      "GMC on/off on MOT17/MOT20 val (Table 5).",
      "Location-only associator comparison (Table 6) under identical YOLOX weights/NMS, no per-video tuning.",
    ],
    limitations: {
      authorStated: [
        "Poor performance on the public-detection MOT benchmark; association depends heavily on the detector finding partially obscured, low-confidence targets (Sec. 4.6).",
        "Rapid motion/deformation and low frame rates break relative-depth capture and hurt tracking (Sec. 4.6).",
      ],
      evident: [
        "Flat-ground / elevated-camera priors fail on stairs, slopes, drones with strong pitch, or non-grounded objects.",
        "Uniform interval splitting ignores actual depth distribution, risking empty or still-crowded levels.",
      ],
    },
    assumptions: [
      "Camera above ground; all objects on a single flat plane without undulation.",
      "Occlusion order equals depth order (nearer occludes farther).",
    ],
    computation:
      "Inference on RTX 3090; 800×1440 (MOT17/DanceTrack), 896×1600 (MOT20); 19.9/12.5 FPS on MOT17/MOT20 test; no training.",
    relations: [
      { to: "T061", type: "builds-on", note: "Adopts ByteTrack's detector, weights, thresholds and high/low-score staging; DCM upgrades its association." },
      { to: "T005", type: "uses-as-baseline", note: "SORT compared on all three val sets (e.g. MOT17 62.9 HOTA vs 69.2)." },
      { to: "T073", type: "uses-as-baseline", note: "OC-SORT compared in Tables 1 and 6 (63.2 MOT17 test)." },
      { to: "T059", type: "uses-as-baseline", note: "MOTR compared in Tables 1–3 (54.2 DanceTrack vs 55.5)." },
      { to: "T053", type: "uses-as-baseline", note: "TransTrack compared in Tables 1–3." },
      { to: "T043", type: "uses-as-baseline", note: "FairMOT compared in Tables 1–3." },
      { to: "T063", type: "uses-as-baseline", note: "Evaluated on the DanceTrack benchmark with its detector/protocol." },
    ],
    concepts: ["mot", "tracking-by-detection", "data-association", "iou", "cost-matrix", "track-management", "occlusion", "dual-threshold", "kalman", "mota", "hota"],
    impact:
      "Proved scene decomposition by pseudo-depth is a first-class occlusion strategy — IoU-only matching reaching appearance/graph/attention SOTA — and released DCM as a plug-and-play association upgrade.",
  },
];
