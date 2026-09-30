import type { PaperRecord } from "../types";

/* batch-07 — T059..T063 */

export const BATCH_07: PaperRecord[] = [
  {
    id: "T059",
    arxiv: "2105.03247",
    title: "MOTR: End-to-End Multiple-Object Tracking with Transformer",
    shortTitle: "MOTR",
    year: 2021,
    authors: ["Fangao Zeng", "Bin Dong", "Yuang Zhang", "Tiancai Wang", "Xiangyu Zhang", "Yichen Wei"],
    fileName: "2105.03247v4.pdf",
    task: "multi-object",
    tags: ["transformer", "end-to-end", "track-query", "set-prediction", "deformable-detr", "online-tracking"],
    difficulty: "advanced",
    summary:
      "MOTR extends DETR with a dynamically-sized set of track queries that are updated frame-by-frame to iteratively predict whole object trajectories, trained with tracklet-aware label assignment (TALA) over multi-frame clips via a collective average loss (CAL) plus a temporal aggregation network (TAN). With no IoU matching, track NMS or Re-ID post-processing, it beats ByteTrack by 6.5% HOTA on DanceTrack and leads its concurrent transformer trackers on association metrics.",
    problem:
      "MOT was dominated by tracking-by-detection with hand-crafted association (motion/appearance similarity heuristics plus Hungarian matching), whose post-processing nature blocks end-to-end temporal learning across frames. DETR had shown one-shot end-to-end set prediction for detection, but how to do sequence prediction over time inside that framework was an open question.",
    background: ["detection", "mot", "tracking-by-detection", "attention", "hungarian", "data-association"],
    previousWork: [
      {
        name: "Tracking-by-detection MOT (SORT, DeepSORT, JDE, FairMOT)",
        limitation:
          "Association is a hand-crafted post-process over appearance/motion similarity (Re-ID cosine distance, IoU, Kalman filter), so temporal information cannot flow end-to-end through the network.",
        whyThisPaper:
          "MOTR replaces the whole associate-after-detect pipeline with track queries that carry identity through the Transformer decoder, learning appearance and position variance jointly.",
      },
      {
        name: "TransTrack (concurrent transformer tracker)",
        limitation:
          "Models a full track as independent short tracklets (object pairs over two adjacent frames) and still needs IoU matching to stitch them into full tracks — tracking remains decoupled detection plus matching.",
        whyThisPaper:
          "MOTR models the full track end-to-end through iterative track-query updates with no IoU matching at inference (Table 1 comparison).",
      },
      {
        name: "TrackFormer (concurrent track-query tracker)",
        limitation:
          "Learns only within two adjacent frames, so temporal learning is weak and duplicate tracks must be removed with heuristics (track NMS, Re-ID features).",
        whyThisPaper:
          "Multi-frame CAL plus TAN give stronger temporal modeling, removing the need for those heuristics; MOTR leads TrackFormer by 4.5% IDF1 on MOT17.",
      },
    ],
    researchGap:
      "Before this paper, no DETR-style framework performed iterative set-of-sequence prediction for MOT, so end-to-end tracking without any association post-process did not exist.",
    contribution: [
      "MOTR framework: track queries as hidden states of tracks, transferred and updated frame-by-frame through a shared Deformable DETR decoder for iterative trajectory prediction.",
      "Tracklet-aware label assignment (TALA): newborn-only bipartite matching for detect queries (Eq. 1) plus target-consistent assignment inheritance for track queries (Eq. 2), with an entrance/exit mechanism on classification scores.",
      "Collective average loss (CAL): training over whole video clips with loss normalized by total object count, generating long-range motion samples instead of two-frame training.",
      "Temporal aggregation network (TAN): a modified decoder layer in QIM that aggregates each track query with its previous state via multi-head attention as a historical shortcut.",
      "State-of-the-art association on DanceTrack (HOTA 54.2, +6.5% over ByteTrack) and best association scores among transformer methods on MOT17 and BDD100k.",
    ],
    method: {
      pipeline: ["extract", "decode-detect", "decode-track", "predict", "entrance-exit", "aggregate-temporal"],
      architecture:
        "ResNet-50 CNN plus Deformable DETR encoder per frame; concatenation of fixed learnable detect queries and variable-length track queries fed into the Deformable DETR decoder; query interaction module (QIM: entrance/exit filters plus TAN) converts decoder hidden states into next-frame track queries; boxes from prediction heads on hidden states.",
      motionModel:
        "None hand-crafted: motion is learned implicitly through iterative track-query updates, TAN history aggregation and multi-frame CAL training — explicitly positioned against Kalman-filter heuristics.",
      appearanceModel:
        "None separate: appearance and position variance are learned jointly inside the query embeddings; no Re-ID branch or cosine-distance matching anywhere.",
      association:
        "No explicit association at inference: identity is carried by the track query itself (TALA guarantees one query per identity during training); duplicates are suppressed by decoder self-attention as in DETR.",
      detectionDependency:
        "Built-in Deformable DETR detector (COCO-pretrained initialization); detect queries handle newborn objects only and are suppressed on already-tracked objects.",
      trackManagement:
        "Entrance/exit mechanism: during training, hidden states of disappeared objects (or IoU < 0.5 with target) are dropped and newborn assignments from Eq. 1 are kept; at inference, detect predictions above τ_en enter, track predictions below τ_ex for M consecutive frames exit; augmented with query erase (p_drop) and false-positive insert (p_insert) during training.",
      optimization:
        "AdamW, lr 2e-4 with decay ×10 at fixed epochs; clip length progressively increased (2→5) during training; batch of one 5-frame clip; 200 epochs MOT17, 20 epochs DanceTrack/BDD100k; short side resized to 800 (max 1536); random flip/crop/shift augmentation.",
      loss:
        "CAL (Eq. 3) averaging per-frame DETR-style losses (Eq. 4: focal classification + L1 + generalized IoU) over the whole clip, normalized by total ground-truth count.",
    },
    equations: [
      {
        id: "motr-tala-detect",
        label: "TALA newborn-only assignment for detect queries",
        formula: "ω_det^i = argmin_{ω ∈ Ω_i} L(Ŷ_det^i|_ω, Y_new^i)",
        variables: [
          { symbol: "Ŷ_det^i", meaning: "predictions of the detect queries at frame i" },
          { symbol: "Y_new^i", meaning: "ground-truth boxes of newborn objects at frame i only" },
          { symbol: "Ω_i", meaning: "space of all bipartite matches between detect queries and newborn objects" },
          { symbol: "L", meaning: "pairwise DETR matching cost (classification + box costs)" },
        ],
        intuition:
          "Detect queries may only claim objects appearing for the first time — anything already tracked is someone else's (a track query's) job.",
        why:
          "Standard DETR matching would let any query match any object; restricting detect queries to newborns is what makes one-query-per-identity possible.",
        where: "Section 3.3, Eq. 1; applied per frame during training.",
        params: "Self-attention in the decoder suppresses detect queries that fire on tracked objects, mirroring duplicate removal in DETR.",
        simulator: "hungarian",
      },
      {
        id: "motr-tala-track",
        label: "Target-consistent assignment inheritance for track queries",
        formula: "ω_tr^i = ω_tr^{i-1} ∪ ω_det^{i-1} (for i > 1); ω_tr^1 = ∅",
        variables: [
          { symbol: "ω_tr^i", meaning: "label assignment of track queries at frame i" },
          { symbol: "ω_det^{i-1}", meaning: "newborn assignment from the previous frame, promoted into tracks" },
          { symbol: "∅", meaning: "empty set — no tracked objects exist at the first frame" },
        ],
        intuition:
          "Once a query adopts an identity it keeps it forever: each frame's track assignments are last frame's tracks plus last frame's newborns.",
        why:
          "Inheritance (no re-matching) is the mechanism that forces a single query to predict a whole trajectory, eliminating association post-processing.",
        where: "Section 3.3, Eq. 2; applied for every frame after the first during training.",
        simulator: "track-mgmt",
      },
      {
        id: "motr-cal",
        label: "Collective average loss over a video clip",
        formula: "L_o(Ŷ|_ω, Y) = Σ_{n=1}^{N} [L(Ŷ_tr^i|_ω_tr^i, Y_tr^i) + L(Ŷ_det^i|_ω_det^i, Y_det^i)] / Σ_{n=1}^{N} V_i",
        variables: [
          { symbol: "N", meaning: "clip length in frames (progressively 2→5 during training)" },
          { symbol: "V_i = V_tr^i + V_det^i", meaning: "total number of ground-truth objects (tracked + newborn) at frame i" },
          { symbol: "L", meaning: "single-frame DETR-style loss of Eq. 4" },
        ],
        intuition:
          "Grade the whole clip at once, normalized by how many objects actually appeared, instead of grading frame by frame.",
        why:
          "Two-frame training cannot generate long-range motion samples; clip-level loss lets the network learn occlusion, duplication and ID-switch failures that only emerge over time.",
        where: "Section 3.6, Eq. 8; the training objective over each sampled clip.",
        params: "Longer clips help monotonically in ablations (MOTA 44.9→53.2 from length 2→5); sampling interval 10 is best, beyond 12 hurts.",
      },
      {
        id: "motr-frame-loss",
        label: "Single-frame detection-style loss",
        formula: "L(Ŷ_i|_ω_i, Y_i) = λ_cls · L_cls + λ_l1 · L_l1 + λ_giou · L_giou",
        variables: [
          { symbol: "L_cls", meaning: "focal loss on query classification scores" },
          { symbol: "L_l1", meaning: "L1 loss on predicted box coordinates" },
          { symbol: "L_giou", meaning: "generalized IoU loss on predicted boxes" },
          { symbol: "λ_cls, λ_l1, λ_giou", meaning: "weight coefficients of the three terms" },
        ],
        intuition:
          "Each frame is still supervised like a DETR detection problem: right class, tight box, good overlap.",
        why:
          "Keeps per-frame localization precise while CAL handles the temporal credit assignment across frames.",
        where: "Section 3.6, Eq. 9; summed inside CAL for detect and track predictions separately.",
      },
    ],
    datasets: ["mot17", "dancetrack", "others"],
    metrics: ["hota", "assa", "deta", "mota", "idf1", "idsw"],
    baselines: ["Tracktor++", "CenterTrack", "TraDes", "QDTrack", "FairMOT", "ByteTrack", "TransTrack", "TrackFormer", "CorrTracker"],
    results: [
      "MOT17 test, transformer comparison (Table 3): MOTR HOTA 57.8, AssA 55.7, DetA 60.3, IDF1 68.6, MOTA 73.4, IDS 2439 — vs TransTrack HOTA 54.1 / IDF1 63.9 / MOTA 74.5 and TrackFormer IDF1 63.9 / MOTA 65.0.",
      "DanceTrack (Table 4): MOTR HOTA 54.2, AssA 40.2, DetA 73.5, MOTA 79.7, IDF1 51.5 — surpassing ByteTrack (HOTA 47.7, AssA 32.1) by 6.5% HOTA and 8.1% AssA.",
      "BDD100k validation (Table 5): MOTR mMOTA 32.0, mIDF1 43.5, IDSw 3493 — best mMOTA with fewest switches among compared entries.",
      "MOT17 vs ByteTrack: MOTR is inferior on MOT17 (ByteTrack MOTA 80.3), which the authors attribute to shared-decoder detect queries being suppressed on tracked objects, limiting newborn detection.",
    ],
    ablations: [
      "Components (Table 6a): detect-only baseline IDF1 1.2 → +track query 49.8 → +TAN (MOTA 44.9, IDF1 63.4) → +CAL (MOTA 53.2, IDF1 70.5, IDS 155); TAN alone adds 7.8% MOTA / 13.6% IDF1.",
      "Clip length in CAL (Table 6b): length 2→5 improves MOTA 44.9→53.2 and IDF1 63.4→70.5 with IDS 257→155.",
      "Query erase/insert (Tables 6c–6d): best p_drop 0.1 (MOTA 53.2) and p_insert 0.3; larger values degrade both metrics.",
      "Entrance/exit thresholds (Table 6e): performance within 0.5% MOTA across τ_en 0.7–0.9; τ_ex 0.5 slightly better than 0.6.",
      "Sampling interval (Table 6f): interval 2→10 cuts IDS 209→155; beyond 10–12 performance degrades as dynamics get too long-range.",
    ],
    limitations: {
      authorStated: [
        "Newborn-object detection is unsatisfactory: detect queries suppressed on tracked objects goes against the nature of object queries and limits detection of new objects (visible in the weaker MOTA/DetA).",
        "Frame-by-frame query passing limits training efficiency; parallel decoding as in VisTR fails on complex MOT scenes — both are named as future research topics.",
      ],
      evident: [
        "Inference runs at ~7.5 FPS on a V100 at 800px short side, far from the real-time SORT/ByteTrack regime.",
        "Identity relies entirely on query-embedding continuity with no re-identification fallback, so a lost query has no recovery mechanism.",
      ],
    },
    assumptions: [
      "Video processed online in temporal order; track query set starts empty at frame one.",
      "COCO-pretrained Deformable DETR weights available for initialization.",
    ],
    computation:
      "Inference ~7.5 FPS on Tesla V100 (800px short side); training on 8× V100 (memory-optimized variant fits 2080 Ti); ResNet-50 backbone, one 5-frame clip per batch.",
    relations: [
      { to: "T053", type: "uses-as-baseline", note: "TransTrack compared on MOT17 (HOTA 54.1 vs 57.8) and DanceTrack; MOTR removes its IoU-matching stitching step." },
      { to: "T054", type: "uses-as-baseline", note: "TrackFormer compared on MOT17 (IDF1 63.9 vs 68.6); MOTR replaces two-frame learning plus track-NMS/Re-ID heuristics with CAL+TAN." },
      { to: "T061", type: "uses-as-baseline", note: "ByteTrack beaten by 6.5% HOTA / 8.1% AssA on DanceTrack; inferior to it on MOT17 detection metrics." },
      { to: "T043", type: "uses-as-baseline", note: "FairMOT in MOT17 Table 3 (HOTA 59.3 vs MOTR 57.8) as the Re-ID-based reference." },
      { to: "T030", type: "uses-as-baseline", note: "Tracktor++ in MOT17 Table 3 (HOTA 44.8 vs 57.8)." },
      { to: "T042", type: "uses-as-baseline", note: "CenterTrack in Tables 3–4 as the point-based joint detection-tracking reference." },
      { to: "T063", type: "uses-as-baseline", note: "Evaluated on the DanceTrack benchmark, where MOTR posts its headline association gains." },
    ],
    concepts: ["end-to-end-mot", "attention", "mot", "data-association", "track-management", "hota"],
    impact:
      "MOTR established the track-query plus clip-level training template for end-to-end MOT: its TALA/CAL/TAN design became the baseline that later query-propagation trackers (MOTRv2/v3) directly build on, and its DanceTrack results made the dataset the standard association stress test.",
  },
  {
    id: "T060",
    arxiv: "2105.11595",
    title: "SiamMOT: Siamese Multi-Object Tracking",
    shortTitle: "SiamMOT",
    year: 2021,
    authors: ["Bing Shuai", "Andrew Berneshawi", "Xinyu Li", "Davide Modolo", "Joseph Tighe"],
    fileName: "2105.11595v1.pdf",
    task: "multi-object",
    tags: ["siamese", "motion-model", "tracking-by-detection", "region-based", "explicit-matching", "online-tracking"],
    difficulty: "intermediate",
    summary:
      "SiamMOT grafts a region-based Siamese tracker onto Faster-RCNN so detection and instance-motion estimation train jointly, comparing an implicit MLP motion model (IMM) against an explicit channel-wise cross-correlation model with pixel-level supervision (EMM). EMM wins everywhere — 65.9 MOTA on MOT17, 41.1 TrackAP on TAO-person, and victory over the ACM MM'20 HiEve challenge winners — proving instance-level motion modeling is the key SORT ingredient.",
    problem:
      "Online SORT-style trackers lived or died by their motion model: Kalman filters on geometry fail under fast camera motion and deforming poses, while implicit displacement regression (Tracktor, CenterTrack) lacked the fine-grained matching supervision that made Siamese single-object trackers robust. No unified network jointly trained detection with an explicit Siamese motion model for MOT.",
    background: ["detection", "mot", "tracking-by-detection", "siamese", "data-association", "motion-model"],
    previousWork: [
      {
        name: "SORT with Kalman filter on geometric features",
        limitation:
          "Motion modeled from box geometry alone; collapses when people move fast (CRP: MOTA 15.9) because the linear prior cannot follow large displacements.",
        whyThisPaper:
          "A learned Siamese motion model on region features raises CRP MOTA to ~76 with the same solver, isolating motion modeling as the bottleneck.",
      },
      {
        name: "Tracktor (detection-network regression as tracker)",
        limitation:
          "Regresses the target location from previous-frame features with the detector itself — implicit motion with no template matching and no dense supervision, weak on fast motion and deformation.",
        whyThisPaper:
          "Both IMM and EMM beat the authors' own Tracktor implementation on all three datasets (e.g. MOT17 58.6→63.3 MOTA), with EMM's explicit correlation best.",
      },
      {
        name: "CenterTrack (point-based implicit displacement)",
        limitation:
          "Infers motion implicitly from point-based center features, which the paper argues is weaker than region-based explicit template matching for localization and tracking.",
        whyThisPaper:
          "Region-based EMM outperforms CenterTrack by 4.4 MOTA on MOT17 public detections (61.5→65.9).",
      },
    ],
    researchGap:
      "Before this paper, no end-to-end MOT network combined a region-based detector with a Siamese tracker offering both implicit and explicit motion parameterizations, so the value of explicit template matching for MOT was unmeasured.",
    contribution: [
      "SiamMOT architecture: Faster-RCNN (RPN + detection head) plus a region-based Siamese tracker sharing one backbone forward pass, with parallel per-instance tracking and joint loss L = L_rpn + L_detect + L_motion.",
      "Implicit motion model (IMM): MLP on concatenated template/search ROIAlign features predicting visibility plus relative box motion vector (Eq. 2), trained with focal + smooth-L1 loss.",
      "Explicit motion model (EMM): channel-wise cross-correlation response maps with an FCOS-style dense head predicting per-pixel visibility and box offsets, decoded by argmax over a motion-penalty-modulated score map (Eqs. 4–5) with IoU + focal loss (Eq. 6).",
      "Training/inference recipe: triplet types (positive/hard/negative), kept-alive memory τ = 30 frames for short occlusions, linking thresholds α = 0.4 / β = 0.6.",
      "State of the art on MOT17 public (65.9 MOTA), TAO-person (41.1 TrackAP@0.5) and HiEve over the challenge winners, at 17 FPS on 720p video.",
    ],
    method: {
      pipeline: ["detect", "propose-search", "correlate", "estimate-motion", "score-visibility", "match-spatial", "manage-tracks"],
      architecture:
        "DLA-34 + FPN Faster-RCNN backbone (features computed once); ROIAlign template features (15×15) and search-region features (search = 2× target size; EMM 30×30) per instance; IMM is a 2-layer 512-hidden MLP; EMM is a 2-layer fully-convolutional head on channel-wise correlation maps predicting dense visibility and offset maps.",
      motionModel:
        "Learned instance-level Siamese motion: IMM regresses the normalized motion vector of Eq. 2; EMM does dense pixel-level template matching with a cosine-window plus scale-change penalty map (Eq. 5, λ = 0.4, 0.1 on CRP).",
      appearanceModel:
        "Template ROIAlign features from frame t matched against the search region at t+δ; no separate Re-ID embedding — identity comes from motion continuity plus the spatial matching solver.",
      association:
        "SORT-style spatial matching: NMS on detections and tracker outputs independently, suppress detections with IoU ≥ 0.5 to a tracked instance, then the standard online solver (continue if visibility > α, birth if detection > β, kill after τ invisible frames).",
      detectionDependency:
        "Faster-RCNN with RPN; triplets sampled from RPN outputs during training; under public protocol the provided detections are re-scored with the SiamMOT detector.",
      trackManagement:
        "Short-occlusion memory: tracks with visibility below α are kept and re-searched from their last location for up to τ = 30 frames before termination; inference thresholds α = 0.4, β = 0.6.",
      optimization:
        "SGD with momentum, batch 16 image pairs, 25K–50K iterations; lr 0.02 decayed ×10 at 60% and 80%; weight decay 1e-4; image-pair training with crop/rescale/blur when video labels are unavailable, video pairs up to 1s apart otherwise.",
      loss:
        "Joint L = L_rpn + L_detect + L_motion; IMM loss is focal visibility plus masked smooth-L1 on the motion vector (Eq. 3); EMM loss is dense focal visibility plus centerness-weighted IoU regression (Eq. 6).",
    },
    equations: [
      {
        id: "siammot-track",
        label: "Siamese motion estimation per instance",
        formula: "(v_i^{t+δ}, R̃_i^{t+δ}) = T(f_{R_i^t}, f_{S_i^{t+δ}}; Θ)",
        variables: [
          { symbol: "f_{R_i^t}", meaning: "ROIAlign features of detected instance i at frame t (the template)" },
          { symbol: "f_{S_i^{t+δ}}", meaning: "ROIAlign features of the search region (template box expanded ×2) at frame t+δ" },
          { symbol: "T(; Θ)", meaning: "learnable Siamese tracker (IMM or EMM parameterization)" },
          { symbol: "v_i^{t+δ}, R̃_i^{t+δ}", meaning: "predicted visibility confidence and propagated box at t+δ" },
        ],
        intuition:
          "Cut out the person now, cut out a bigger window later, and ask the network: where did this exact person go, and is it still visible?",
        why:
          "One shared-backbone operation, run in parallel per instance, turns MOT association into per-instance motion estimation instead of graph optimization.",
        where: "Section 3.1, Eq. 1; executed for every detected instance each frame pair.",
        params: "Search expansion factor r = 2 around the same geometric center.",
        simulator: "siamese",
      },
      {
        id: "siammot-motionvec",
        label: "Normalized relative motion vector (IMM target)",
        formula: "m_i = [(x_i^{t+δ}−x_i^t)/w_i^t, (y_i^{t+δ}−y_i^t)/h_i^t, log(w_i^{t+δ}/w_i^t), log(h_i^{t+δ}/h_i^t)]",
        variables: [
          { symbol: "(x, y, w, h)", meaning: "box center coordinates, width and height at the superscripted frame" },
          { symbol: "m_i", meaning: "translation normalized by template size plus log scale change — the IMM regression target" },
        ],
        intuition:
          "Express movement in units of the person's own size, so a tall nearby pedestrian and a small distant one are learned on the same scale.",
        why:
          "Normalization makes one MLP handle all object scales; inversion of this transform decodes the predicted box.",
        where: "Section 3.2, Eq. 2; regressed by the IMM MLP head.",
        simulator: "motion",
      },
      {
        id: "siammot-emm-decode",
        label: "EMM dense decode with motion penalty",
        formula: "R̃_i^{t+δ} = R(p_i(x*,y*)); v_i^{t+δ} = v_i(x*,y*); (x*,y*) = argmax_{x,y}(v_i ⊙ η_i); η_i(x,y) = λC + (1−λ)S(R(p(x,y)), R_i^t)",
        variables: [
          { symbol: "v_i, p_i", meaning: "dense visibility map and dense offset map (offsets to top-left/bottom-right corners) from head ψ" },
          { symbol: "η_i", meaning: "penalty map discouraging dramatic movement" },
          { symbol: "C", meaning: "cosine window w.r.t. the previous target center" },
          { symbol: "S", meaning: "Gaussian on relative scale change vs the previous box" },
          { symbol: "λ", meaning: "weighting scalar (0.4 default, 0.1 on fast-motion CRP)" },
        ],
        intuition:
          "Score every pixel as the person's new center, down-weight jumps that are far away or suddenly a different size, and pick the best surviving pixel.",
        why:
          "Pixel-level supervision plus the penalty is exactly what makes EMM robust to fast motion and distractors where IMM's global regression fails.",
        where: "Section 3.3, Eqs. 4–5; EMM inference decode.",
        params: "Lower λ on CRP tolerates the dataset's very large displacements.",
        simulator: "siamese",
      },
      {
        id: "siammot-emm-loss",
        label: "EMM dense training loss",
        formula: "L = Σ_{x,y} L_focal(v_i, v_i*) + Σ_{x,y} 1[v_i* = 1] · (w(x,y) · L_reg(p_i, p_i*))",
        variables: [
          { symbol: "v_i*, p_i*", meaning: "pixel-wise ground truth: 1 inside the target box, corner offsets per pixel" },
          { symbol: "L_focal", meaning: "focal loss on dense visibility classification" },
          { symbol: "L_reg", meaning: "IoU loss on box-offset regression" },
          { symbol: "w(x,y)", meaning: "centerness weight of pixel (x,y) w.r.t. the target instance" },
        ],
        intuition:
          "Teach every pixel its job: background pixels say 'not here', inside pixels draw the box — with center pixels graded most strictly.",
        why:
          "Dense supervision is the paper's core claim over implicit models: fine-grained spatial targets learn robust matching functions.",
        where: "Section 3.3, Eq. 6; over all valid search-region locations of positive triplets.",
      },
    ],
    datasets: ["mot17", "others"],
    metrics: ["mota", "idf1", "mt-ml", "fp", "fn", "idsw", "fps"],
    baselines: ["Tracktor", "Tracktor++", "CenterTrack", "DeepSORT", "optical-flow (PWC-Net) + Tracktor", "STRN", "FCS-Track", "Selective-JDE", "LinkBox"],
    results: [
      "MOT17 test, public detection (Table 4): SiamMOT 65.9 MOTA, 63.3 IDF1, MT 34.6%, ML 23.9%, FP 18098, FN 170955, IDsw 3040 — above CenterTrack (61.5/59.6) and Tracktor++ v2 (56.5/55.1).",
      "TAO-person validation (Table 5): 41.1 TrackAP@0.5 (ResNet-101) vs Tracktor++ 36.7; DLA-169 reaches 42.1, SiamMOT+ with off-the-shelf Re-ID linking 44.3.",
      "HiEve test, public detection (Table 6): DLA-34 51.5 MOTA / 47.9 IDF1 matches challenge winners; DLA-169 53.2 / 51.7 beats all tuned winners.",
      "Motion-model ladder (Table 1, MOT17-train): Tracktor 58.6 → +Flow 60.3 → IMM 61.5 → EMM 63.3 MOTA; on CRP, Tracktor 15.9 → EMM 76.4 MOTA (+35 over Tracktor+Flow).",
      "Speed (Table 1): EMM 17.6 FPS and IMM 19.5 FPS on 720p vs 12.5 FPS for the Flow variant — the Siamese models share the backbone forward pass.",
    ],
    ablations: [
      "Triplet sampling (Table 2): P+H alone gives low MOTA (59.7) from unkilled false positives; P+N better; P+H+N best (63.3 MOTA) — negatives teach the tracker to kill, hard examples cut ID switches.",
      "Joint training (Sec 5.3): EMM with joint detector+tracker training 63.3 vs 61.5 MOTA without; detector AP unchanged (73.3% vs 73.4%), so joint training costs nothing on detection.",
      "Keep-alive τ (Table 3): IDF1/TrackAP rise with τ and saturate around 30 frames (~1s); beyond that targets leave the ×2 search region.",
      "Linking thresholds α/β (Table 11, appendix): best at α = 0.4 with β = 0.4–0.6 (MOTA 63.8/63.0); high β = 0.8 drops MOTA to ~59.",
    ],
    limitations: {
      authorStated: [
        "Re-ID linking still helps on TAO-person videos where people leave and re-enter view — beyond instance-level motion modeling, so SiamMOT+ adds an off-the-shelf Re-ID merge.",
        "The tracker cannot follow targets through occlusions longer than ~30 frames since people move outside the ×2 search region; improving long-occlusion motion modeling is named as future work.",
      ],
      evident: [
        "Public-detection results rely on re-scoring the provided boxes with the SiamMOT detector, so the headline numbers mix detection and tracking gains.",
        "Framework demonstrated only on person tracking; multi-class extension is stated as a plan, not an experiment.",
      ],
    },
    assumptions: [
      "Targets move within a 2×-expanded search region between processed frames (pairs sampled ≤1s apart in training).",
      "Evaluation on CRP uses only annotated frames and ignores detections overlapping background boxes at IoU > 0.2.",
    ],
    computation:
      "17 FPS on 720p video with a single modern GPU (EMM, DLA-34); training 25K–50K iterations SGD on CrowdHuman/COCO with batch 16 image pairs.",
    relations: [
      { to: "T005", type: "builds-on", note: "SORT's online propagate-then-match solver is the skeleton SiamMOT upgrades with a learned Siamese motion model." },
      { to: "T030", type: "uses-as-baseline", note: "Tracktor/Tracktor++ beaten on MOT17 (+9.4 MOTA) and TAO-person (36.7→41.1 TrackAP); used as the no-Siamese ablation (-tracker) implementation." },
      { to: "T042", type: "uses-as-baseline", note: "CenterTrack beaten 61.5→65.9 MOTA; its point-based implicit motion is the foil for region-based explicit matching." },
      { to: "T013", type: "uses-as-baseline", note: "DeepSORT appears on the HiEve leaderboard SiamMOT tops (27.1 vs 53.2 MOTA)." },
    ],
    concepts: ["siamese", "motion-model", "tracking-by-detection", "data-association", "reid"],
    impact:
      "SiamMOT settled the implicit-vs-explicit motion debate for online MOT in favor of explicit dense template matching, and its region-based Siamese-plus-detector blueprint with joint training became a reference design for motion-centric trackers.",
  },
  {
    id: "T061",
    arxiv: "2110.06864",
    title: "ByteTrack: Multi-Object Tracking by Associating Every Detection Box",
    shortTitle: "ByteTrack",
    year: 2021,
    authors: ["Yifu Zhang", "Peize Sun", "Yi Jiang", "Dongdong Yu", "Fucheng Weng", "Zehuan Yuan", "Ping Luo", "Wenyu Liu", "Xinggang Wang"],
    fileName: "2110.06864v3.pdf",
    task: "multi-object",
    tags: ["tracking-by-detection", "data-association", "low-score-association", "kalman-filter", "yolox", "real-time"],
    difficulty: "intro",
    summary:
      "ByteTrack observes that discarding low-score detections deletes occluded-but-real objects, so its BYTE association matches high-score boxes first with Kalman-predicted tracks, then matches the leftover tracks against low-score boxes (IoU only) to rescue true objects while background stays unmatched. Applied to 9 existing trackers for +1–10 IDF1 points and packaged with a YOLOX detector, it tops MOT17 (80.3 MOTA), MOT20, HiEve and BDD100K at 30 FPS.",
    problem:
      "Tracking-by-detection pipelines threshold detections (e.g. score > 0.5) and associate only survivors: when occlusion or motion blur drops a true object's score, the track fragments irreversibly, while naively keeping every box floods association with background false positives. Almost no MOT method handled this threshold dilemma.",
    background: ["detection", "mot", "tracking-by-detection", "data-association", "cost-matrix", "motion-model"],
    previousWork: [
      {
        name: "Threshold-then-associate trackers (FairMOT, QDTrack, CenterTrack and 6 more)",
        limitation:
          "All boxes below the score threshold are deleted before association, so occluded objects with scores like 0.4→0.1 are unrecoverable and trajectories fragment.",
        whyThisPaper:
          "BYTE's second association recovers those objects from low-score boxes via track similarity, lifting CenterTrack IDF1 64.2→74.0 and cutting its IDs 528→144.",
      },
      {
        name: "SORT (single-stage Kalman + Hungarian association)",
        limitation:
          "One matching round over high-score boxes only: unmatched tracks die even when a low-score detection of the same object exists.",
        whyThisPaper:
          "BYTE keeps SORT's machinery and adds the low-score rescue round, moving MOTA 74.6→76.6 and IDs 291→159 with the same detector.",
      },
      {
        name: "MOTDT (motion-propagated boxes fused into association)",
        limitation:
          "Uses propagated boxes as tracklet boxes, which drift; shares BYTE's motivation of handling unreliable detections but trails it by a large margin.",
        whyThisPaper:
          "BYTE re-associates unmatched tracks to real low-score detections instead, keeping tracklet boxes accurate (Table 2).",
      },
    ],
    researchGap:
      "Before this paper, the junction between detection and association was a fixed score threshold, so no generic association method exploited the track-similarity signal inside low-score boxes.",
    contribution: [
      "BYTE two-stage association (Algorithm 1): first match tracks to high-score boxes with motion or Re-ID similarity, then match leftover tracks to low-score boxes with IoU only, deleting unmatched low-score boxes as background.",
      "Evidence that low-score boxes are worth mining: rescued boxes contain notably more TPs than FPs (Figure 4), and BYTE is robust to the threshold τ_high from 0.2 to 0.8 (Figure 3).",
      "Plug-in gains on 9 state-of-the-art trackers (JDE, CSTrack, FairMOT, TraDes, QDTrack, CenterTrack, Chained-Tracker, TransTrack, MOTR), +1–10 IDF1 points in two application modes.",
      "ByteTrack system: YOLOX-X detector plus BYTE with Kalman motion only — 80.3 MOTA / 77.3 IDF1 / 63.1 HOTA on MOT17 at 30 FPS, rank 1 on MOT17, MOT20, HiEve and BDD100K.",
      "Practical recipe details: IoU < 0.2 match rejection, 30-frame lost-track buffer with rebirth, tracklet interpolation for fully-occluded (visibility-0) pedestrians, boundary-aware YOLOX label handling for MOT17.",
    ],
    method: {
      pipeline: ["detect", "split-by-score", "predict-kalman", "associate-high", "associate-low", "birth-kill"],
      architecture:
        "Detector (YOLOX-X, COCO-pretrained, Mosaic+Mixup, multi-scale training) plus a detector-agnostic association module; no appearance network on MOT17/MOT20/HiEve — Kalman filter plus Hungarian matching only; optional Re-ID (UniTrack/FastReID) only in the first stage on BDD100K and ablations.",
      motionModel:
        "Kalman filter predicting each track's box in the current frame (as in SORT); on BDD100K (large camera motion, low frame rate) the Kalman predictor is dropped in favor of Re-ID similarity.",
      appearanceModel:
        "None by default — deliberately IoU-only in the second stage because low-score boxes under occlusion/blur have unreliable Re-ID features (Table 1: IoU second stage +1.0 MOTA over Re-ID).",
      association:
        "Two Hungarian rounds: round one matches all tracks (incl. lost) to high-score boxes (IoU or Re-ID); round two matches leftover tracks to low-score boxes with IoU only; matches with IoU < 0.2 rejected; unmatched high-score boxes birth new tracks; unmatched low-score boxes deleted.",
      detectionDependency:
        "Entirely detector-fed; MOT17 numbers use a custom YOLOX trained on MOT17+CrowdHuman+Cityperson+ETHZ with boundary-unclipped boxes; public-detection protocol also reported (re-scored public boxes).",
      trackManagement:
        "Lost tracks kept 30 frames for rebirth before deletion; new tracks birth from unmatched high-score boxes only; tracklet interpolation (appendix Eq. 1) fills fully-occluded gaps up to σ frames.",
      optimization:
        "Detector training: SGD, batch 48 on 8× V100, lr 1e-3 with warmup + cosine annealing, ~12 hours, 80 epochs (MOT17) / 50 (BDD100K); association has no learned parameters.",
      loss:
        "No tracking loss — association is non-parametric; detector trained with the standard YOLOX detection losses (incl. SimOTA assignment).",
    },
    equations: [
      {
        id: "bytetrack-split",
        label: "High/low detection split by score threshold",
        formula: "D_high = {d ∈ D : score(d) > τ}; D_low = {d ∈ D : score(d) ≤ τ} (default τ = 0.6)",
        variables: [
          { symbol: "D", meaning: "all detection boxes with scores from the detector in the current frame" },
          { symbol: "τ", meaning: "score threshold (0.6 default); high boxes associate first, low boxes rescue leftovers" },
        ],
        intuition:
          "Sort the detector's output into 'trustworthy' and 'suspicious' piles instead of throwing the suspicious pile away.",
        why:
          "The split is the whole junction innovation: the first round gets clean boxes, the second round gets a second chance to explain the leftovers.",
        where: "Algorithm 1, lines 3–13; first operation per frame.",
        params: "BYTE stays above SORT for τ_high anywhere in 0.2–0.8 (Figure 3) because round two recovers whatever round one misses.",
        simulator: "bytetrack",
      },
      {
        id: "bytetrack-match",
        label: "Two-round Hungarian matching with IoU gate",
        formula: "match(T, D_high; sim₁) → (T_remain, D_remain); match(T_remain, D_low; IoU) → Tre-remain; reject pairs with IoU < 0.2",
        variables: [
          { symbol: "sim₁", meaning: "first-round similarity: IoU with Kalman-predicted boxes, optionally plus Re-ID distance" },
          { symbol: "T_remain", meaning: "tracks unmatched in round one — usually occluded objects" },
          { symbol: "D_remain", meaning: "unmatched high-score boxes — birth new tracks" },
        ],
        intuition:
          "Match the easy cases first, then let the struggling tracks pick from the bargain bin while the actual garbage (unmatched low boxes) goes in the trash.",
        why:
          "Ordering plus IoU-only second round is what separates true occluded objects (similar to a predicted track) from background (similar to nothing).",
        where: "Algorithm 1, lines 17–27; lost tracks over 30 frames old are deleted, rest kept for rebirth.",
        params: "IoU gate 0.2; lost buffer 30 frames; Re-ID excluded from round two by design.",
        simulator: "hungarian",
      },
      {
        id: "bytetrack-interp",
        label: "Tracklet interpolation over fully-occluded gaps",
        formula: "B_t = B_t1 + (B_t2 − B_t1) · (t − t1)/(t2 − t1), applied when t2 − t1 ≤ σ",
        variables: [
          { symbol: "B_t1, B_t2", meaning: "tracklet boxes at the last visible frame t1 and reappearance frame t2" },
          { symbol: "σ", meaning: "maximum gap length eligible for interpolation" },
        ],
        intuition:
          "If someone vanishes behind a pillar and reappears, draw a straight line between the two sightings.",
        why:
          "MOT17 annotates fully-occluded (visibility 0) pedestrians that no detector can see; linear fill-in recovers them for the metric.",
        where: "Appendix D; ablations show MOTA 76.6→78.3 as σ goes 0→20–30 (Table 11).",
        params: "Gains saturate at σ = 20–30 with slightly more FPs.",
        simulator: "track-mgmt",
      },
    ],
    datasets: ["mot17", "mot20", "others"],
    metrics: ["mota", "idf1", "hota", "idsw", "fp", "fn", "fps"],
    baselines: ["SORT", "DeepSORT", "MOTDT", "JDE", "CSTrack", "FairMOT", "TraDes", "QDTrack", "CenterTrack", "Chained-Tracker", "TransTrack", "MOTR", "YOLOX"],
    results: [
      "MOT17 test private (Table 4): ByteTrack 80.3 MOTA, 77.3 IDF1, 63.1 HOTA at 29.6 FPS — rank 1, +3.3 MOTA / +5.3 IDF1 over runner-up ReMOT.",
      "MOT20 test (Table 5): 77.8 MOTA, 75.2 IDF1, 61.3 HOTA at 17.5 FPS — rank 1, IDs cut 4209→1223 (−71%) vs runner-up SOTMOT.",
      "HiEve test (Table 6): 61.7 MOTA, 63.1 IDF1 — rank 1 over CenterTrack (40.9/45.1).",
      "BDD100K (Table 7): val mMOTA 45.5 (vs QDTrack 36.6), test mMOTA 40.1 / mIDF1 55.8 — rank 1.",
      "Public detection protocol (Tables 12–13): MOT17 67.4 MOTA / 70.0 IDF1 (+1.5/+6.7 over SiamMOT); MOT20 67.0 / 70.2, IDs cut to one quarter of TMOH.",
      "BYTE vs SORT/DeepSORT/MOTDT (Table 2, MOT17 val): MOTA 74.6→76.6, IDF1 76.9→79.3, IDs 291→159 with identical Kalman motion.",
      "BYTE on 9 trackers (Table 3): consistent gains, e.g. CenterTrack +1.3 MOTA / +9.8 IDF1, Chained-Tracker +1.9/+5.8, TransTrack +1.2/+4.1 with Kalman motion.",
    ],
    ablations: [
      "Similarity choice (Table 1): IoU second stage beats Re-ID second stage by ~1.0 MOTA on both MOT17 and BDD100K; Re-ID first stage helps on BDD100K (large camera motion) but not MOT17.",
      "Threshold robustness (Figure 3): BYTE's MOTA/IDF1 nearly flat as τ_high varies 0.2–0.8 while SORT collapses — the rescue round absorbs threshold choice.",
      "Low-score mining audit (Figure 4): rescued low-score boxes contain notably more TPs than FPs on every MOT17 sequence despite raw low-score pools being FP-heavy.",
      "Light detectors (Table 8): BYTE beats DeepSORT at every YOLOX scale, +3.0 MOTA with YOLOX-Nano — the cheap-detector regime benefits most.",
      "Speed/accuracy (Table 9): 75.0 MOTA at 45.7 FPS (512×928) up to 76.6 at 29.6 FPS (800×1440); association costs ~4ms.",
      "Training data (Table 10): MOT17-half alone already 75.8 MOTA thanks to Mosaic/Mixup; CrowdHuman adds mostly IDF1 (occlusion recognition).",
    ],
    limitations: {
      authorStated: [
        "Tracklet interpolation assumes linear motion across full occlusions — a heuristic fill-in for visibility-0 annotations, not a motion model.",
        "On BDD100K the Kalman predictor fails (low frame rate, large ego-motion) and must be replaced with an off-the-shelf Re-ID model, so the 'motion-only' simplicity does not transfer there.",
      ],
      evident: [
        "All headline private-protocol numbers couple BYTE with a strong custom YOLOX detector and extra training data, so association gains cannot be read off the leaderboard alone (val ablations with fixed detections are the clean comparison).",
        "Low-score rescue is purely spatial (IoU), so crowded scenes with overlapping low-score boxes have no appearance fallback.",
      ],
    },
    assumptions: [
      "A per-frame detector emits scored boxes separable at threshold τ (0.6 default).",
      "Occluded objects reappear within the 30-frame lost buffer and near their Kalman-predicted position (IoU ≥ 0.2).",
    ],
    computation:
      "30 FPS on MOT17 (800×1440, FP16, single V100: ~26ms detection + ~4ms association); detector trains ~12h on 8× V100; YOLOX-Nano variant runs the same association for edge use.",
    relations: [
      { to: "T005", type: "uses-as-baseline", note: "SORT is the Table 2 baseline BYTE improves (74.6→76.6 MOTA) using identical Kalman motion." },
      { to: "T013", type: "uses-as-baseline", note: "DeepSORT compared in Tables 2 and 8; BYTE matches or beats it with no Re-ID model." },
      { to: "T034", type: "improves", note: "BYTE applied to JDE detections gains +0.6 MOTA / +2.4 IDF1 with Kalman motion (Table 3)." },
      { to: "T043", type: "improves", note: "BYTE applied to FairMOT gains +1.2–1.3 MOTA / +1.4 IDF1 (Table 3)." },
      { to: "T053", type: "improves", note: "BYTE applied to TransTrack gains +1.2 MOTA / +4.1 IDF1 with Kalman motion (Table 3)." },
      { to: "T059", type: "uses-as-baseline", note: "MOTR compared in Tables 3–4 and BYTE applied to it (+1.0 MOTA with Kalman motion)." },
      { to: "T060", type: "uses-as-baseline", note: "SiamMOT beaten under public detections on MOT17 (+1.5 MOTA) and MOT20 (Tables 12–13)." },
      { to: "T042", type: "uses-as-baseline", note: "CenterTrack is the headline BYTE application (+9.8 IDF1) and a beaten SOTA line in Tables 4–6." },
    ],
    concepts: ["dual-threshold", "tracking-by-detection", "data-association", "track-management", "mota"],
    impact:
      "BYTE's 'associate every box' principle became the default second-chance mechanism in tracking-by-detection — absorbed into OC-SORT, StrongSORT-era baselines and the DanceTrack association studies — while ByteTrack reset all four major MOT leaderboards with a detector-plus-matching recipe.",
  },
  {
    id: "T062",
    arxiv: "2110.13027",
    title: "TAPL: Dynamic Part-based Visual Tracking via Attention-guided Part Localization",
    shortTitle: "TAPL",
    year: 2021,
    authors: ["Han Wei", "Huang Hantao", "Yu Xiaoxi"],
    fileName: "2110.13027v1.pdf",
    task: "single-object",
    tags: ["part-based", "attention", "transformer", "template-update", "siamese", "gumbel-softmax"],
    difficulty: "intermediate",
    summary:
      "TAPL reformulates single-object tracking as predicting the location of every target part (each template feature cell) with a transformer encoder, deriving the box from the parts' mean and spread, and supervises the otherwise label-free part predictions with a Gumbel-softmax attention loss that ties each part to its most-attended search location. A multi-head-attention updater fuses the most recent pseudo-template into the part representation, reaching VOT2018 EAO 0.489 and the best OTB100 success at 33 FPS.",
    problem:
      "Holistic Siamese trackers with fixed first-frame templates collapse under deformation and occlusion, while existing local-pattern methods learn parts as anonymous response-map peaks with no physical meaning, and updating templates online needs either large history banks or separate reliability classifiers.",
    background: ["sot", "siamese", "bounding-box", "attention", "appearance-features"],
    previousWork: [
      {
        name: "Holistic Siamese trackers (SiamFC, SiamRPN)",
        limitation:
          "Fixed template matched as one global kernel; susceptible to large appearance change since the representation never adapts and has no notion of object parts.",
        whyThisPaper:
          "Each template cell becomes an independently localized part with a dynamically updated representation, so deformation moves parts rather than breaking one global match.",
      },
      {
        name: "Local-pattern Siamese variants (structured Siamese nets, local semantic branches)",
        limitation:
          "Local patterns are response-map peaks learned without part-level supervision — no clear physical meaning and all maps are fused blindly into one box.",
        whyThisPaper:
          "Parts get explicit predicted coordinates supervised by the attention loss, giving each part a verifiable location on the object silhouette (Figure 3).",
      },
      {
        name: "History-heavy updaters (STARK-style dynamic templates with scoring nets)",
        limitation:
          "STARK-ST50 needs a separately trained template scoring network to reject bad dynamic templates and a 320×320 search region; the update path is complex.",
        whyThisPaper:
          "TAPL updates from just the ground-truth template plus the previous frame's pseudo-template through one attention module — simpler, 6× faster than SiamR-CNN, though behind STARK on GOT-10k (64.2 vs 68.0 AO).",
      },
    ],
    researchGap:
      "Before this paper, no end-to-end tracker directly predicted per-part locations from dynamic part representations with attention-guided part-level supervision, so part-based tracking lacked both physical interpretability and a lightweight update rule.",
    contribution: [
      "Two-stage formulation: per-part location prediction from template part vectors followed by box estimation from the part distribution (mean for center, scaled std for size, σ = 3).",
      "Attention-based part representation updater: multi-head attention fusing pseudo-template parts into template parts with template-as-query (misalignment-recoverable) plus learnable positional encodings.",
      "Attention-guided part localization: Gumbel-softmax hard-attention loss (Eqs. 5–6) forcing each predicted part toward its most-attended search location, jointly trained with L1 + GIoU box loss (Eq. 7, λ = 0.1).",
      "Results: VOT2018 EAO 0.489 (best, +4% over SiamRN 0.470) with top accuracy 0.617; best success on OTB100; GOT-10k AO 64.2 at 33 FPS on a 2080 Ti.",
    ],
    method: {
      pipeline: ["extract-triplet", "mask-background", "update-parts", "encode-global", "localize-parts", "estimate-box"],
      architecture:
        "ImageNet-pretrained ResNet-50 backbone; last three layers concatenated and downsampled to 512 channels for template z, search x and pseudo-template y; 4-layer transformer encoder (8 heads, 512 hidden) over masked target parts plus sinusoidally-encoded search parts; 2-layer MLP per part predicting 2D coordinates.",
      motionModel:
        "None: localization comes from part-to-search attention each frame; the pseudo-template is always cropped from the previous frame's tracking result.",
      appearanceModel:
        "Dynamic part vectors f̂_z(i) (Eq. 1): template cell features plus learned positional encoding plus attention-fused pseudo-template features; background cells zeroed by the box-derived mask M_z.",
      association:
        "Not applicable (single object); temporal continuity is carried by the pseudo-template updater rather than any matching step.",
      detectionDependency: "None — class-agnostic SOT initialized from the first-frame box.",
      trackManagement:
        "No explicit lost/found logic; robustness to bad pseudo-templates comes from the template-as-query fusion direction and training-time random shift/scale augmentation of the pseudo-template.",
      optimization:
        "SGD, 40 epochs on 4× RTX-2080Ti; warmup 0.001→0.005 over 5 epochs then exponential decay to 0.0005; backbone last three layers unfrozen after epoch 10; trained on COCO + ImageNet VID + LaSOT + GOT-10k (GOT-10k-only training for that benchmark).",
      loss:
        "L = L_bbox + 0.1·L_atten: L_bbox is L1 + generalized-IoU between the distribution-derived box and ground truth; L_atten (Eq. 6) is the L1 distance between predicted part locations and their Gumbel-attended search coordinates.",
    },
    equations: [
      {
        id: "tapl-updater",
        label: "Attention-based dynamic part representation",
        formula: "f̂_z(i) = f_z(i) + Pos_z(i) + Atten(Q, K)V; K = V = [f_y(1)…f_y(Hy·Wy)], Q = f_z(i)",
        variables: [
          { symbol: "f_z(i)", meaning: "template feature cell i (one target part candidate)" },
          { symbol: "f_y(·)", meaning: "pseudo-template cells from the previous frame's tracking result" },
          { symbol: "Pos_z(i)", meaning: "learnable positional encoding at template position i (DETR-style)" },
          { symbol: "Atten(Q,K)V", meaning: "multi-head attention aligning pseudo-template parts onto the template part" },
        ],
        intuition:
          "Ask every piece of the original target: which piece of last frame's result looks like you? Then blend that fresh appearance into yourself — while keeping your original identity as the anchor.",
        why:
          "Template-as-query (rather than the reverse) lets a misaligned pseudo-template be recovered instead of corrupting the representation, using only one history frame.",
        where: "Section 3.1, Eq. 1; applied to all Hz·Wz template cells before masking.",
        params: "Pseudo-template realism comes from training augmentation: random shift/scale of the ground-truth region.",
      },
      {
        id: "tapl-partloc",
        label: "Direct per-part location prediction",
        formula: "l_i = MLP(h_z(i)), i = 1…Hz·Wz",
        variables: [
          { symbol: "h_z(i)", meaning: "transformer-encoded representation of template part i (global context fused)" },
          { symbol: "MLP", meaning: "two-linear-layer perceptron shared across parts" },
          { symbol: "l_i", meaning: "predicted 2D coordinates of part i in the search region" },
        ],
        intuition:
          "Each puzzle piece of the target points to where it thinks it moved in the new frame — after having seen the whole puzzle and the whole search area.",
        why:
          "Direct coordinate prediction per part (instead of a fused response map) is what makes parts interpretable and the box estimate deformation-flexible.",
        where: "Section 3.2, Eq. 3; one forward pass per frame over masked target parts.",
      },
      {
        id: "tapl-box",
        label: "Bounding box from part distribution",
        formula: "o = (1/Nt)Σ_i l_i·M_z(i); s = σ·√((1/Nt)Σ_i (l_i − o)²·M_z(i)); σ = 3",
        variables: [
          { symbol: "o", meaning: "box center: mean of predicted target-part locations" },
          { symbol: "s", meaning: "box scale: scaled standard deviation of part locations" },
          { symbol: "M_z(i) ∈ {0,1}", meaning: "target mask (1 inside the template ground-truth box)" },
          { symbol: "Nt", meaning: "number of target parts, Σ_i M_z(i)" },
          { symbol: "σ = 3", meaning: "fixed scaling from uniform-in-box assumption" },
        ],
        intuition:
          "The middle of the scattered parts is the object center; how widely they scatter tells you how big it is.",
        why:
          "Statistics over many parts smooth out individual part errors, which is why accuracy (0.617) leads VOT2018 even under deformation.",
        where: "Section 3.2, Eq. 4; supervised by L1 + GIoU against the ground-truth box.",
        params: "σ fixed at 3; rectangular-box masking can label background cells as parts on irregular shapes (noted in qualitative analysis).",
      },
      {
        id: "tapl-attenloss",
        label: "Gumbel attention loss and total objective",
        formula: "a_i = Gumbel(h_z(i)·[h_x(1)…h_x(Hx·Wx)]ᵀ); L_atten = Σ_i ‖a_i·P − l_i‖₁; L = L_bbox + λ·L_atten (λ = 0.1)",
        variables: [
          { symbol: "a_i", meaning: "differentiable hard-attention (one-hot-like) vector from part i to search cells" },
          { symbol: "P", meaning: "2×(Hx·Wx) matrix of all search-cell coordinates" },
          { symbol: "l_i", meaning: "the MLP-predicted location of part i" },
          { symbol: "L_bbox", meaning: "L1 + generalized-IoU loss on the distribution-derived box" },
        ],
        intuition:
          "Each part must agree with its own spotlight: 'you said you moved here — your attention peak should be here too.'",
        why:
          "Box loss alone constrains only the parts' overall distribution and leaves individuals free to drift; the attention term forces semantically meaningful per-part locations and is worth +0.167 EAO.",
        where: "Section 3.2, Eqs. 5–7; Gumbel-softmax keeps the hard attention differentiable end-to-end.",
        params: "λ = 0.1; removing the term drops VOT2018 EAO 0.489→0.322 and OTB success 0.705→0.674.",
      },
    ],
    datasets: ["vot", "otb", "got10k", "others"],
    metrics: ["eao", "success-auc", "precision", "fps"],
    baselines: ["SiamRPN++", "DiMP", "Ocean", "SiamBAN", "KYS", "TrDiMP", "SiamRN", "ATOM", "DaSiamRPN", "SiamCAR", "TransT", "SiamFC++", "PrDiMP", "SiamR-CNN", "STARK-ST50"],
    results: [
      "VOT2018 (Table 1): accuracy 0.617, robustness 0.140, EAO 0.489 — best EAO, +4% over SiamRN (0.470), with the highest accuracy of all compared methods.",
      "OTB100 (Figure 2): highest success score among ATOM, DaSiamRPN, DiMP, KYS, SiamRPN++, SiamBAN, TrDiMP, SiamCAR and TransT; precision 0.908 (Table 3).",
      "GOT-10k (Table 2, train-on-GOT-10k-only protocol): AO 64.2, SR0.50 74.7, SR0.75 54.6 at 33 FPS on a 2080 Ti — near SiamR-CNN (64.9) at 6× its speed, behind STARK-ST50 (68.0).",
    ],
    ablations: [
      "Attention loss (Table 3): removing it drops VOT2018 EAO 0.489→0.322 (accuracy 0.617→0.563, robustness 0.140→0.302) and OTB success 0.705→0.674 — the largest single-component effect.",
      "Updater (Table 3): removing it drops EAO to 0.269 with robustness collapsing to 0.421 — adapting to appearance change matters more than precise localization.",
      "Qualitative (Figure 3, ant1/basketball/fish1): part centers track silhouettes under deformation, rotation and distractors; stray background points come from rectangular-box masking of irregular shapes.",
    ],
    limitations: {
      authorStated: [
        "End-to-end update from the previous frame is more susceptible to occlusion and out-of-view than STARK-ST50's separately scored dynamic templates; adopting a reliability scorer is named as future work.",
        "Smaller 255×255 search region (vs 320×320 in STARK-ST50) weakens robustness to fast motion and out-of-view targets.",
      ],
      evident: [
        "Target mask from an axis-aligned box labels background cells as parts for non-rectangular objects, injecting noise the attention loss must absorb.",
        "No re-detection or lost-state handling: a fully lost target has no recovery path beyond the next pseudo-template crop.",
      ],
    },
    assumptions: [
      "First-frame ground-truth box given; single target per sequence.",
      "Target lies within the 255×255 search region cropped around the previous prediction.",
    ],
    computation:
      "33 FPS on a single RTX 2080 Ti (ResNet-50, 4-layer transformer encoder); training 40 epochs SGD on 4× 2080 Ti.",
    relations: [
      { to: "T028", type: "uses-as-baseline", note: "SiamRPN++ beaten on VOT2018 (EAO 0.417 vs 0.489) and OTB100 success." },
      { to: "T031", type: "uses-as-baseline", note: "DiMP-50 compared on all three benchmarks (VOT2018 EAO 0.441; GOT-10k AO 61.1 vs 64.2)." },
      { to: "T038", type: "uses-as-baseline", note: "SiamBAN compared on VOT2018 (0.452) and OTB100." },
      { to: "T055", type: "uses-as-baseline", note: "TransT (Transformer Tracking) compared on OTB100; TAPL differs by predicting per-part locations instead of fusing a global template." },
      { to: "T056", type: "uses-as-baseline", note: "STARK-ST50 leads TAPL on GOT-10k (68.0 vs 64.2 AO); its scoring-network update is the named future improvement." },
      { to: "T037", type: "uses-as-baseline", note: "SiamR-CNN nearly matched on GOT-10k AO (64.9 vs 64.2) at 6× lower speed (5 vs 33 FPS)." },
      { to: "T035", type: "uses-as-baseline", note: "SiamFC++ compared on GOT-10k (AO 59.5 vs 64.2)." },
      { to: "T040", type: "uses-as-baseline", note: "PrDiMP compared on GOT-10k (AO 63.4 vs 64.2)." },
    ],
    concepts: ["sot", "siamese", "attention", "appearance-features", "bounding-box"],
    impact:
      "TAPL showed part coordinates with attention supervision can replace response-map fusion in Siamese tracking, giving the field a lightweight template-update alternative to history banks and scorer networks alongside top VOT2018 accuracy.",
  },
  {
    id: "T063",
    arxiv: "2111.14690",
    title: "DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion",
    shortTitle: "DanceTrack",
    year: 2021,
    authors: ["Peize Sun", "Jinkun Cao", "Yi Jiang", "Zehuan Yuan", "Song Bai", "Kris Kitani", "Ping Luo"],
    fileName: "2111.14690v3.pdf",
    task: "benchmark",
    tags: ["benchmark", "dataset", "motion-modeling", "uniform-appearance", "nonlinear-motion", "association"],
    difficulty: "intro",
    summary:
      "DanceTrack is a 100-video, 105K-frame MOT benchmark of group dancing where performers wear near-identical clothes and move with complex non-linear motion, frequent crossovers and occlusions — deliberately breaking the appearance-matching shortcut of MOT17/MOT20. Oracle analysis shows association, not detection, is the bottleneck (IoU-only matching already near-perfect on MOT17 but collapses on DanceTrack), and the paper's benchmark plus mask/pose/depth studies redirect the field toward motion modeling.",
    problem:
      "MOT datasets (MOT17/20, KITTI, BDD100K) feature distinguishable appearances and near-linear motion, so trackers that match Re-ID embeddings plus linear predictors top the leaderboards while failing catastrophically on uniform-appearance, dynamic-motion scenes common in real life — a dataset bias the community had no platform to measure or fix.",
    background: ["mot", "tracking-by-detection", "benchmark-design", "mota", "reid"],
    previousWork: [
      {
        name: "MOT17 / MOT20 pedestrian benchmarks",
        limitation:
          "Handful of videos with regular motion and distinguishable clothes: oracle IoU-only matching on ground-truth boxes already scores HOTA 98.1, so detection quality decides everything and association research is unrewarded.",
        whyThisPaper:
          "DanceTrack provides 10× more videos/images where the same oracle scores only 72.8 HOTA, making association the measurable bottleneck.",
      },
      {
        name: "Appearance-matching trackers (JDE, FairMOT, QDTrack)",
        limitation:
          "Built around Re-ID embeddings and contrastive appearance learning; the paper's oracle shows adding appearance similarity on DanceTrack hurts every metric (HOTA 72.8→59.7).",
        whyThisPaper:
          "A benchmark where appearance is provably uninformative forces investment in motion and temporal-dynamics modeling instead.",
      },
      {
        name: "Linear motion models (SORT Kalman prediction)",
        limitation:
          "Constant-velocity priors handle MOT17's regular walking but cannot follow dance crossover, extreme articulation and frequent relative-position switches.",
        whyThisPaper:
          "DanceTrack quantifies motion diversity (switch-frequency metric, Eq. 3) and shows temporal models (Kalman, LSTM) beating static IoU matching by large margins.",
      },
    ],
    researchGap:
      "Before this paper, no large-scale MOT benchmark combined uniform appearance with diverse non-linear motion, so trackers were never evaluated on association quality independent of detection and Re-ID strength.",
    contribution: [
      "DanceTrack dataset: 100 group-dance/sport videos, 105,855 frames at 20 FPS, 990 tracks, full-body boxes (no annotation for fully-occluded frames, identity kept on reappearance), 40/25/35 train/val/test split with private test annotations.",
      "Bias diagnostics: Re-ID cosine-distance distributions (DanceTrack far more similar than MOT17), adjacent-frame IoU (comparable speed) and relative-position-switch frequency (far higher crossovers) with formal definitions (Eqs. 1–3).",
      "Oracle analysis on ground-truth boxes proving the bottleneck flip: MOT17 solved by IoU matching alone (98.1 HOTA) while DanceTrack peaks at 72.8 and appearance matching hurts.",
      "Full private-setting benchmark of 8–9 modern trackers showing universal association collapse (detection metrics higher on DanceTrack than MOT17, association metrics far lower) plus YOLOX-fixed association and motion-model studies.",
      "Modality roadmap: joint training with COCO mask (+1.2 HOTA) and pose (+3.7, +pose-association further) helps; KITTI depth does not (domain shift) but depth-assisted association shows a small positive signal.",
    ],
    method: {
      pipeline: ["collect", "annotate", "verify", "split", "benchmark", "analyze"],
      architecture:
        "Dataset construction, not a tracker: videos scraped with dance-genre keywords (street/pop/classical, gymnastics, kung fu, cheerleading, large groups up to 40 people); commercial annotation tool with propagated boxes refined per frame; independent re-check pass; HOTA primary metric with AssA/IDF1 (association) and DetA/MOTA (detection) plus FP/FN/IDs.",
      association:
        "Benchmarked association strategies on fixed YOLOX detections: IoU, SORT (Kalman), DeepSORT (+appearance), MOTDT, BYTE, OC-SORT — OC-SORT best (52.1 HOTA val); motion-model study: none 44.7 → Kalman 47.8 → LSTM 51.6 HOTA.",
      detectionDependency:
        "Private-setting benchmark (each method detects + associates); association-only studies fix YOLOX detections trained on DanceTrack train to isolate matching quality.",
      trackManagement:
        "Annotation policy as track management ground truth: full-body box for partial occlusion, no box when fully occluded, identity preserved across gaps.",
    },
    equations: [
      {
        id: "dancetrack-appearance",
        label: "Cross-object appearance similarity diagnostic",
        formula: "V = (1/T)Σ_t (1/Nt²)Σ_{i}Σ_{j≠i} (1 − cos<F(B_i^t), F(B_j^t)>)",
        variables: [
          { symbol: "F(B_i^t)", meaning: "pretrained Re-ID embedding of object i on frame t" },
          { symbol: "T, Nt", meaning: "frame count of the video and object count on frame t" },
          { symbol: "V", meaning: "mean pairwise appearance distance — low means uniforms, not individuals" },
        ],
        intuition:
          "Average how different everybody looks from everybody else: street clothes score high, matching dance costumes score low.",
        why:
          "Quantifies the dataset's core claim — Figure 3a shows DanceTrack's distribution clearly below MOT17's, invalidating appearance matching.",
        where: "Section 3.2, Eq. 1; computed over full videos with a pretrained Re-ID model.",
        simulator: "reid",
      },
      {
        id: "dancetrack-adjacent-iou",
        label: "Adjacent-frame IoU (speed check)",
        formula: "U = (1/(N(T−1)))Σ_iΣ_{t=1}^{T−1} IoU(B_i^t, B_i^{t+1})",
        variables: [
          { symbol: "B_i^t", meaning: "box of object i on frame t" },
          { symbol: "U", meaning: "mean box overlap between consecutive frames — low means too fast or too slow a frame rate" },
        ],
        intuition:
          "Check the dataset isn't just hard because everyone sprints: overlapping boxes frame-to-frame means speeds are sane.",
        why:
          "Figure 3b shows DanceTrack matches MOT datasets here, so the difficulty must come from motion complexity, not speed.",
        where: "Section 3.2, Eq. 2.",
        simulator: "iou-track",
      },
      {
        id: "dancetrack-switch",
        label: "Relative-position switch frequency (crossover metric)",
        formula: "S = Σ_iΣ_{j≠i}Σ_{t=1}^{T−1} sw(B_i^t, B_j^t, B_i^{t+1}, B_j^{t+1}) / (2N(T−1)(N−1))",
        variables: [
          { symbol: "sw(·)", meaning: "1 if the pair swaps left-right or top-down order between adjacent frames (counted only when overlapping), else 0" },
          { symbol: "S", meaning: "average crossover rate — the dataset's motion-diversity signature" },
        ],
        intuition:
          "Count how often dancers swap places while overlapping: walkers almost never do, dancers constantly do.",
        why:
          "Figure 3c shows DanceTrack far above MOT17/MOT20/KITTI — the quantitative reason linear predictors fail and occlusion is dynamic.",
        where: "Section 3.2, Eq. 3; positions compared via box centers.",
        params: "Restricted to overlapping pairs so only occlusion-relevant crossovers count.",
        simulator: "motion",
      },
    ],
    datasets: ["dancetrack", "mot17", "mot20", "others"],
    metrics: ["hota", "assa", "deta", "mota", "idf1", "idsw", "fp", "fn"],
    baselines: ["CenterTrack", "FairMOT", "QDTrack", "TransTrack", "TraDes", "MOTR", "GTR", "ByteTrack", "OC-SORT", "SORT", "DeepSORT", "MOTDT", "YOLOX"],
    results: [
      "Scale (Table 1): 100 videos, 105,855 images, 990 tracks, avg 52.9s at 20 FPS — ~10× the videos/images of MOT17 (14 videos, 11,235 images).",
      "Oracle on GT boxes (Table 2): MOT17 IoU-only HOTA 98.1 vs DanceTrack 72.8; adding motion+appearance on DanceTrack drops to 59.7 — appearance matching is actively harmful.",
      "Test benchmark v3 (Table 3): every method's association collapses on DanceTrack vs MOT17 (e.g. ByteTrack AssA 62.0→32.1, FairMOT 58.0→23.8) while detection is higher (ByteTrack DetA 64.5→71.0); OC-SORT leads (55.1 HOTA), then MOTR (54.2) and QDTrack (54.2).",
      "Fixed-detector association val (Table 4, YOLOX): OC-SORT 52.1 > SORT 47.8 > BYTE 47.1 > DeepSORT 45.8 > IoU 44.7 > MOTDT 39.2 HOTA — appearance (DeepSORT) loses to pure motion (SORT).",
      "Motion models (Table 5): LSTM 51.6 > Kalman 47.8 > none 44.7 HOTA, IDF1 50.8/48.3/36.8.",
      "Modalities (Table 6, CenterNet+BYTE baseline 36.9 HOTA): +COCO mask 38.1, +mask association 39.2; +COCO pose 40.6, +pose association 41.0; +KITTI depth hurts jointly (34.4) but depth association adds +0.7 over that baseline.",
    ],
    ablations: [
      "Association decomposition (Table 2): IoU-only is the best DanceTrack oracle (72.8); adding Kalman keeps 69.4; adding appearance drops to 59.7–68.0 depending on combination.",
      "t-SNE of Re-ID features (Figure 4): identities separable on MOT17, entangled on DanceTrack — visual proof of uniform appearance.",
      "Detection-vs-association split (Table 3): DetA/MOTA higher on DanceTrack than MOT17 for all methods while AssA/IDF1 far lower, isolating association as the bottleneck.",
      "Mask vs pose vs depth (Table 6): pose helps most (+3.7 HOTA box-only), mask moderately, depth only within its own joint baseline — fine-grained human representations beat scene-depth transfer.",
    ],
    limitations: {
      authorStated: [
        "No algorithm is provided that solves the dataset — outperforming prior trackers is explicitly left as an open question for future study.",
        "Only bounding-box + identity annotations in this version; pose and segmentation-mask labels judged important for fine-grained study are missing (limited time/resources).",
      ],
      evident: [
        "Single domain (dance/sport performance video): uniform-appearance stress comes at the cost of scene-type diversity vs MOT17/KITTI/BDD100K.",
        "Benchmark numbers are version-sensitive (Table 3 updated across v1–v3 with new methods and retrained QDTrack), so cross-paper comparisons must pin the version.",
      ],
    },
    assumptions: [
      "Full-body amodal boxes for partial occlusion; fully-occluded objects unannotated with identity resumed on reappearance.",
      "HOTA as primary metric (supplemented by AssA/IDF1 for association, DetA/MOTA for detection).",
    ],
    relations: [
      { to: "T061", type: "uses-as-baseline", note: "ByteTrack benchmarked on DanceTrack test (47.7 HOTA) and its BYTE strategy ablated on val (47.1); MOTR beats it by 6.5% HOTA." },
      { to: "T059", type: "uses-as-baseline", note: "MOTR benchmarked (54.2 HOTA / 40.2 AssA test) as the strongest transformer baseline at v1." },
      { to: "T053", type: "uses-as-baseline", note: "TransTrack benchmarked (45.5 HOTA test) as the transformer short-tracklet reference." },
      { to: "T043", type: "uses-as-baseline", note: "FairMOT benchmarked (39.7 HOTA test) as the Re-ID-based reference that collapses without appearance cues." },
      { to: "T045", type: "uses-as-baseline", note: "QDTrack benchmarked (54.2 HOTA test v3, YOLOX-based) as the appearance-learning reference." },
      { to: "T042", type: "uses-as-baseline", note: "CenterTrack benchmarked (41.8 HOTA test) and its train/val split protocol reused for MOT17-side evaluation." },
      { to: "T005", type: "uses-as-baseline", note: "SORT/Kalman used in oracle, association (47.8 val) and motion-model studies as the linear-motion reference." },
    ],
    concepts: ["benchmark-design", "mot", "hota", "motion-model", "occlusion"],
    impact:
      "DanceTrack reoriented MOT evaluation from detection-dominated leaderboards to association-first stress testing: it became the required benchmark for motion-modeling claims (OC-SORT, MOTR follow-ups) and its mask/pose findings pushed fine-grained representations into tracker design.",
  },
];
