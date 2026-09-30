import type { PaperRecord } from "../types";

/* batch-01 — exemplar record. Later batches follow this shape exactly. */

export const BATCH_01: PaperRecord[] = [
  {
    id: "T005",
    arxiv: "1602.00763",
    title: "Simple Online and Realtime Tracking",
    shortTitle: "SORT",
    year: 2016,
    authors: ["Alex Bewley", "Zongyuan Ge", "Lionel Ott", "Fabio Ramos", "Ben Upcroft"],
    fileName: "1602.00763v2.pdf",
    task: "multi-object",
    tags: [
      "tracking-by-detection",
      "online",
      "kalman-filter",
      "hungarian",
      "iou-association",
      "real-time",
      "baseline",
    ],
    difficulty: "intro",
    summary:
      "A deliberately minimal online MOT framework: a Faster R-CNN detector per frame, a constant-velocity Kalman filter to predict box positions, and the Hungarian algorithm on IoU distance for association. It runs at 260 Hz and matches far more complex trackers on MOTA.",
    problem:
      "MOT had grown into complex batch systems (MHT, JPDA, tracklet stitching) that delay decisions, are combinatorially expensive, and were often held back by weak detectors rather than by their association logic. There was no simple, real-time, online baseline that isolated how much of tracking is really just 'predict, then match IoU'.",
    background: [
      "bounding-box",
      "iou",
      "detection",
      "mot",
      "tracking-by-detection",
      "online-vs-offline",
      "motion-model",
      "state-space",
      "kalman",
      "data-association",
      "cost-matrix",
      "hungarian",
      "track-management",
      "mota",
    ],
    previousWork: [
      {
        name: "Multiple Hypothesis Tracking (MHT) and JPDA",
        limitation:
          "Combinatorial complexity is exponential in the number of tracked objects and decisions are delayed while assignments are uncertain — impractical for real-time, highly dynamic scenes.",
        whyThisPaper:
          "Shows that a single-stage, greedy-free but simple assignment with motion prediction reaches comparable accuracy at 260 Hz.",
      },
      {
        name: "Tracklet / two-stage association methods (e.g. Geiger et al.'s TBD, ALExTRAC, DP-NMS)",
        limitation:
          "They combine geometry and appearance in two association stages, require batch computation, and add appearance machinery that costs latency.",
        whyThisPaper:
          "SORT keeps a single association stage over adjacent frames only, using box geometry alone, and is explicitly designed as an open-source baseline.",
      },
      {
        name: "ACF-based trackers on the MOT benchmark",
        limitation:
          "The only top tracker not using the ACF detector was far behind — detection quality, not tracking logic, was suspected to be the bottleneck.",
        whyThisPaper:
          "SORT swaps in a CNN detector (Faster R-CNN) and measures the tracking gain from detection alone: up to 18.9% MOTA improvement.",
      },
    ],
    researchGap:
      "No one had measured how far a transparent, real-time, appearance-free Kalman+Hungarian pipeline could go once given a strong CNN detector, and what that implied about where MOT research effort actually pays off.",
    contribution: [
      "Leverages CNN detection inside MOT and demonstrates detection quality dominates tracking accuracy (ACF → FrRCNN(VGG16) raises the proposed tracker from 15.1 to 34.0 MOTA on validation sequences).",
      "Pragmatic tracker: Kalman filter for state propagation, Hungarian algorithm on IoU distance for data association, no appearance model beyond the detector.",
      "Explicit track management: birth from unmatched detections (velocity 0, large velocity covariance), probation to suppress false positives, death after T_Lost missed frames.",
      "Runs the tracking component at 260 Hz on a single core — over 20× faster than other state-of-the-art trackers at the time.",
      "Open-source baseline that later MOT work (DeepSORT onward) builds on and compares against.",
    ],
    method: {
      pipeline: ["detect", "predict", "cost", "associate", "update", "birth/death"],
      architecture:
        "Two components: Faster R-CNN (FrRCNN(ZF) or FrRCNN(VGG16), PASCAL VOC defaults, person class only, score > 0.5) as the detector, plus a lean tracking module that owns state estimation, association and track lifecycle.",
      motionModel:
        "Linear constant-velocity model, independent of camera motion and other objects; inter-frame displacements of centre, scale and (fixed) aspect ratio.",
      association:
        "Cost matrix = IoU distance between each predicted box and each detection; solved optimally with the Hungarian algorithm; assignments below IOU_min are rejected.",
      detectionDependency:
        "Explicitly high — the paper's central measurement. Detector is pluggable; tracking logic is detector-agnostic.",
      trackManagement:
        "New track from any unmatched detection whose overlap with all tracks is < IOU_min; terminated after T_Lost = 1 frame without a match (chosen because the constant-velocity model is a poor long-horizon predictor and re-identification is out of scope).",
      optimization:
        "No learned parameters in the tracker; Kalman covariances, IOU_min and T_Lost are tuned on the MOT train/val split.",
    },
    equations: [
      {
        id: "sort-state",
        label: "Track state vector",
        formula: "x = [u, v, s, r, u̇, v̇, ṡ]ᵀ",
        variables: [
          { symbol: "u, v", meaning: "pixel coordinates of the box centre" },
          { symbol: "s", meaning: "box scale (area)" },
          { symbol: "r", meaning: "aspect ratio, held constant" },
          { symbol: "u̇, v̇, ṡ", meaning: "velocities of centre and scale" },
        ],
        intuition:
          "A track is a small physical state, not just a box: by carrying velocity the filter can predict a box into the next frame before any detection arrives.",
        why:
          "Prediction shrinks the search: association only has to consider boxes near where each object is expected, which is what makes frame-to-frame matching work.",
        where: "Section 3.2 (Estimation Model); initialized on track birth, propagated every frame.",
        params:
          "Velocity is unobserved at birth, so its covariance starts large — the filter then trusts early measurements strongly and slowly starts trusting its own prediction.",
        paperIds: ["T005"],
      },
      {
        id: "sort-kalman",
        label: "Kalman predict / update",
        formula:
          "predict: x̂⁻ = F x̂ , P⁻ = F P Fᵀ + Q\nupdate: K = P⁻ Hᵀ (H P⁻ Hᵀ + R)⁻¹ , x̂ = x̂⁻ + K (z − H x̂⁻) , P = (I − K H) P⁻",
        variables: [
          { symbol: "F, H", meaning: "constant-velocity motion model and measurement (box) projection" },
          { symbol: "Q, R", meaning: "process noise (how erratic motion is) and measurement noise (how noisy detections are)" },
          { symbol: "P", meaning: "state covariance — the filter's uncertainty about the track" },
          { symbol: "K", meaning: "Kalman gain — how much of the new detection is absorbed" },
          { symbol: "z", meaning: "the associated detection box" },
        ],
        intuition:
          "Predict where the box should be, then blend that guess with the actual detection in proportion to which is more certain. Unmatched tracks are simply predicted with no correction.",
        why:
          "It is the cheapest principled way to fuse motion and measurement, and its uncertainty naturally implements 'trust the detector more when motion is uncertain'.",
        where: "Section 3.2; prediction runs every frame, correction only runs for associated detections.",
        params:
          "Large Q → the filter follows detections quickly (agile but jittery). Large R → the filter smooths detections heavily (stable but laggy). SORT's T_Lost = 1 is justified partly because the model is a poor long-horizon predictor.",
        paperIds: ["T005"],
      },
      {
        id: "sort-iou-cost",
        label: "IoU association cost",
        formula: "C(i, j) = 1 − IoU(â_i , z_j)",
        variables: [
          { symbol: "â_i", meaning: "predicted box of track i" },
          { symbol: "z_j", meaning: "detection box j" },
        ],
        intuition:
          "Cheapest pairs are the ones whose boxes overlap most; the matrix is then solved for the globally cheapest one-to-one matching.",
        why:
          "IoU distance needs no training, is scale-invariant, and — as the paper observes — implicitly handles short-term occlusion: when a target is covered by a passing occluder, the IoU cost favours the occluder's detection and the covered track is simply left unmatched (and predicted), instead of being stolen.",
        where: "Section 3.3 (Data Association), before the Hungarian step; IOU_min additionally gates the result.",
        params:
          "Lower IOU_min admits more (riskier) matches; T_Lost = 1 + strict gating makes the tracker fast and fragmenting rather than sticky and identity-swapping.",
        paperIds: ["T005"],
      },
      {
        id: "sort-mota",
        label: "MOTA (evaluation)",
        formula: "MOTA = 1 − (Σ_t (FN_t + FP_t + IDSW_t)) / Σ_t GT_t",
        variables: [
          { symbol: "FN, FP", meaning: "missed and false detections" },
          { symbol: "IDSW", meaning: "identity switches" },
          { symbol: "GT", meaning: "ground-truth objects in the frame" },
        ],
        intuition:
          "Count every tracking error relative to the number of real objects; one number in [-∞, 1] for the whole sequence.",
        why: "It is the leaderboard metric the paper optimizes and reports (MOTA 33.4 on the MOT benchmark test set).",
        where: "Section 4.1 (Metrics); reported in Table 2.",
        params:
          "MOTA is dominated by FP/FN counts, which is exactly why the paper's detector-swap experiment moves it so much — the metric mostly measures detection quality.",
        simulator: "metrics",
        paperIds: ["T005"],
      },
    ],
    datasets: ["mot15", "mot16"],
    metrics: ["mota", "motp", "idsw", "fp", "fn", "fragments", "mt-ml", "fps"],
    baselines: ["NOMT", "MDP", "TDAM", "DP-NMS", "SMOT", "TBD", "ALExTRAC"],
    results: [
      "MOT benchmark test set: MOTA 33.4, MOTP 72.1, FAF 1.3%, MT 11.7%, ML 30.9%, FP 7318, FN 32615, ID switches 1001, fragmentations 1764 — highest MOTA among the online methods compared, close to the far more complex near-online NOMT (33.7).",
      "Detection swap (validation sequences, proposed tracker): ACF 15.1 → FrRCNN(ZF) 24.0 → FrRCNN(VGG16) 34.0 MOTA; same effect measured on MDP (24.0 → 22.6 → 33.5).",
      "Tracking component runs at 260 Hz on a single core (Intel i7 2.5 GHz).",
      "Abstract claim: changing the detector can improve tracking by up to 18.9%.",
    ],
    ablations: [
      "Detector identity is the ablation: ACF vs FrRCNN(ZF) vs FrRCNN(VGG16), holding the tracker fixed — detection recall/precision and ID switches are reported alongside MOTA for both SORT and MDP.",
      "FrRCNN architecture depth (ZF vs VGG16) affects tracking accuracy through detection quality only.",
    ],
    limitations: {
      authorStated: [
        "T_Lost = 1: the constant-velocity model is a poor predictor of true dynamics, and object re-identification is explicitly beyond scope — a reappearing object resumes under a new identity.",
        "Appearance features beyond the detector are deliberately ignored, so long-term occlusion and identity preservation are not handled.",
        "Applies only to pedestrians in the experiments (generalization to other classes argued but not tested).",
        "Short-term/long-term occlusion issues are said to occur rarely in the benchmark and are not explicitly treated.",
      ],
      evident: [
        "IoU-only association cannot re-link a track after a gap or after a large appearance/motion change; fragmentation and ID switches under occlusion are the visible cost.",
        "Results are tied to detector output; the pipeline inherits every detector failure mode (missed detections create track deletions).",
      ],
    },
    assumptions: [
      "Detections are reasonably accurate and per-frame; objects move with roughly constant velocity between frames.",
      "One-to-one, bipartite assignment per frame is sufficient (no explicit data-association ambiguity handling).",
    ],
    computation:
      "Linear Kalman predict/update and Hungarian assignment over a small matrix; no GPU needed for tracking. 260 Hz single-core tracking (excluding detection).",
    relations: [
      {
        to: "T006",
        type: "uses-as-baseline",
        note: "Evaluates on the MOT benchmark (MOT16 evaluation code and test server) introduced for this community.",
      },
    ],
    concepts: [
      "tracking-by-detection",
      "kalman",
      "motion-model",
      "data-association",
      "hungarian",
      "track-management",
      "iou",
      "deep-detection",
      "online-vs-offline",
      "mota",
    ],
    impact:
      "SORT became the canonical MOT baseline: DeepSORT (T013) adds appearance Re-ID on top of exactly this skeleton, StrongSORT/OC-SORT/ByteTrack-era papers all report 'vs SORT', and its Kalman+Hungarian+IoU pipeline is still the reference architecture taught for tracking-by-detection.",
  },
];
