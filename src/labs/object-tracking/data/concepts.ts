import type { Concept } from "./types";

/* Concept registry — the prerequisite DAG that drives the Learning Path
 * and the Foundations view. `simulator` keys point at SIMULATORS in sim/.
 */

export const CONCEPTS: Concept[] = [
  {
    id: "bounding-box",
    name: "Bounding box",
    category: "basics",
    intuition:
      "A rectangle (x, y, w, h) around an object. Tracking carries one box per object per frame, plus an identity that must not change.",
    prereqs: [],
  },
  {
    id: "iou",
    name: "IoU (intersection over union)",
    category: "basics",
    intuition:
      "Overlap of two boxes divided by their union: 1 = identical, 0 = disjoint. In MOT it is both a distance for association and a match quality for evaluation.",
    formula: "IoU(A,B) = |A ∩ B| / |A ∪ B|",
    simulator: "iou-track",
    prereqs: ["bounding-box"],
  },
  {
    id: "detection",
    name: "Object detection as input",
    category: "basics",
    intuition:
      "A detector localizes objects in each frame independently. Tracking adds the temporal question: which detection in frame t+1 belongs to which track?",
    prereqs: ["bounding-box"],
  },
  {
    id: "mot",
    name: "Multi-object tracking (MOT)",
    category: "paradigm",
    intuition:
      "Estimate bounding boxes and persistent identities of many objects in video, frame by frame.",
    prereqs: ["detection"],
  },
  {
    id: "sot",
    name: "Single-object tracking (SOT)",
    category: "paradigm",
    intuition:
      "Given a target box in the first frame, find the same target in every following frame — no detector, one target, appearance-driven.",
    prereqs: ["bounding-box"],
  },
  {
    id: "online-vs-offline",
    name: "Online vs offline tracking",
    category: "paradigm",
    intuition:
      "Online trackers decide associations using only past and current frames (real-time, causal). Offline/batch trackers may re-visit the whole sequence to fix ids later.",
    prereqs: ["mot"],
  },
  {
    id: "tracking-by-detection",
    name: "Tracking-by-detection",
    category: "paradigm",
    intuition:
      "Run a detector per frame, then solve the association problem between detections and existing tracks. Most modern MOT systems are built this way.",
    prereqs: ["detection", "mot"],
  },
  {
    id: "motion-model",
    name: "Motion model",
    category: "motion",
    intuition:
      "A rule that predicts where a box goes next (constant velocity, constant acceleration, learned motion). Prediction turns tracking into a local search.",
    formula: "x̂_{t+1|t} = F x_t",
    simulator: "motion",
    prereqs: ["bounding-box"],
  },
  {
    id: "state-space",
    name: "State representation",
    category: "motion",
    intuition:
      "What numbers describe a track: position, size, aspect ratio, velocity. The choice of state decides what the filter can predict.",
    prereqs: ["bounding-box"],
  },
  {
    id: "kalman",
    name: "Kalman filter",
    category: "motion",
    intuition:
      "A recursive predict→update loop that fuses a motion prediction with a noisy measurement and keeps an uncertainty (covariance) that decides how much to trust each.",
    formula: "K = P⁻ Hᵀ (H P⁻ Hᵀ + R)⁻¹ ; x = x⁻ + K (z − H x⁻)",
    simulator: "kalman",
    prereqs: ["state-space", "motion-model"],
  },
  {
    id: "data-association",
    name: "Data association",
    category: "association",
    intuition:
      "The matching problem: assign each detection to at most one existing track, so identities stay consistent across frames.",
    simulator: "association",
    prereqs: ["mot", "iou"],
  },
  {
    id: "cost-matrix",
    name: "Cost / affinity matrix",
    category: "association",
    intuition:
      "A matrix C[i,j] = cost of assigning detection j to track i, built from IoU, motion distance, and/or appearance distance.",
    formula: "C_ij = λ₁·motion + λ₂·appearance + λ₃·(1−IoU)",
    simulator: "assoc-cost",
    prereqs: ["data-association"],
  },
  {
    id: "hungarian",
    name: "Hungarian algorithm",
    category: "association",
    intuition:
      "Solves the linear assignment problem: the globally cheapest one-to-one matching between tracks and detections, in cubic time.",
    simulator: "hungarian",
    prereqs: ["cost-matrix"],
  },
  {
    id: "greedy-matching",
    name: "Greedy / threshold matching",
    category: "association",
    intuition:
      "Repeatedly take the cheapest pair that is under a threshold. Faster than Hungarian and, with a good gate, nearly as good — used by ByteTrack-style two-stage association.",
    simulator: "hungarian",
    prereqs: ["cost-matrix"],
  },
  {
    id: "track-management",
    name: "Track management",
    category: "association",
    intuition:
      "Rules that birth a track from an unmatched detection, keep it alive while predicted, and kill it after max_age misses — the source of both robustness and fragmentation.",
    simulator: "track-mgmt",
    prereqs: ["data-association"],
  },
  {
    id: "occlusion",
    name: "Occlusion handling",
    category: "association",
    intuition:
      "Objects disappear behind each other. A tracker must survive the gap without swapping identities or deleting the track.",
    prereqs: ["track-management"],
  },
  {
    id: "appearance-features",
    name: "Appearance features",
    category: "appearance",
    intuition:
      "A descriptor of how an object looks (hand-crafted HOG/histograms, or a CNN embedding). Appearance is what survives long occlusions where motion fails.",
    prereqs: ["detection"],
  },
  {
    id: "reid",
    name: "Re-identification (Re-ID)",
    category: "appearance",
    intuition:
      "Deciding that a detection is the same physical object as a track after a gap, by comparing appearance embeddings — the identity backbone of long-term tracking.",
    formula: "cost = 1 − cosine(f_det, f_track)",
    simulator: "reid",
    prereqs: ["appearance-features"],
  },
  {
    id: "deep-detection",
    name: "Deep detectors feed tracking",
    category: "basics",
    intuition:
      "MOT quality is bounded by detector quality: swapping ACF for Faster R-CNN changed SORT's MOTA far more than any tracking trick in the paper.",
    prereqs: ["detection"],
  },
  {
    id: "correlation-filter",
    name: "Correlation filters",
    category: "architecture",
    intuition:
      "Learn a filter that produces a sharp response at the target's location, trained on cyclic shifts of the target patch — closed-form, extremely fast SOT.",
    formula: "α = (K^x K^x + λI)⁻¹ K^y",
    simulator: "corr-filter",
    prereqs: ["sot", "appearance-features"],
  },
  {
    id: "siamese",
    name: "Siamese trackers",
    category: "architecture",
    intuition:
      "Score the similarity between a template patch and a search patch with a weight-shared network; the argmax of the similarity map is the target.",
    formula: "score(u) = ⟨φ(template), φ(search patch at u)⟩",
    simulator: "siamese",
    prereqs: ["sot", "correlation-filter"],
  },
  {
    id: "rpn-head",
    name: "RPN / anchor-based head",
    category: "architecture",
    intuition:
      "Turn a similarity map into box regression + classification with region proposals — the SOT line from SiamRPN onward, giving classification-grade accuracy.",
    prereqs: ["siamese"],
  },
  {
    id: "anchor-free-head",
    name: "Anchor-free heads",
    category: "architecture",
    intuition:
      "Predict classification centerness + regression distances directly from the feature map, removing anchor hyper-parameters and IoU-triggered sample selection.",
    prereqs: ["siamese"],
  },
  {
    id: "attention",
    name: "Attention / transformers",
    category: "architecture",
    intuition:
      "Self-attention relates every position to every other position, letting a tracker model target–background and temporal context jointly instead of with fixed filters.",
    prereqs: ["siamese"],
  },
  {
    id: "end-to-end-mot",
    name: "End-to-end MOT (set prediction)",
    category: "architecture",
    intuition:
      "Predict the whole set of tracks for a frame with Hungarian-matched queries, removing hand-tuned association and NMS-free track propagation.",
    prereqs: ["attention", "hungarian"],
  },
  {
    id: "dual-threshold",
    name: "Low-score association (ByteTrack idea)",
    category: "association",
    intuition:
      "Don't throw away low-confidence detections: first associate high-score boxes, then use the leftovers to rescue occluded tracks before pruning background.",
    simulator: "bytetrack",
    prereqs: ["data-association", "track-management"],
  },
  {
    id: "memory-network",
    name: "Track memory / queues",
    category: "architecture",
    intuition:
      "Store a history of appearance and motion per track so association consults more than the last frame — RobustMT, TrackFormer-style memory, MeMOT queues.",
    prereqs: ["reid"],
  },
  {
    id: "mota",
    name: "MOTA / CLEAR metrics",
    category: "evaluation",
    intuition:
      "One number from counting FP, FN and ID switches against ground truth — detection-dominated, the classic MOT leaderboard metric.",
    formula: "MOTA = 1 − (Σ(FN+FP+IDSW)) / ΣGT",
    simulator: "metrics",
    prereqs: ["mot"],
  },
  {
    id: "idf1",
    name: "IDF1 / identity metrics",
    category: "evaluation",
    intuition:
      "Measures how consistently identities are maintained over the whole sequence, not just per-frame errors.",
    prereqs: ["mota"],
  },
  {
    id: "hota",
    name: "HOTA",
    category: "evaluation",
    intuition:
      "Geometric mean of detection and association accuracy averaged over matching thresholds — designed so a tracker cannot win by ignoring identity.",
    formula: "HOTA = (1/|A|) Σ_α √(DetA_α · AssA_α)",
    simulator: "metrics",
    prereqs: ["mota", "idf1"],
  },
  {
    id: "success-plot",
    name: "Success plot / AUC (SOT)",
    category: "evaluation",
    intuition:
      "Track quality = IoU (or center distance) with ground truth over time; plot the fraction of frames above each threshold and integrate.",
    prereqs: ["iou", "sot"],
  },
  {
    id: "benchmark-design",
    name: "Benchmark design",
    category: "evaluation",
    intuition:
      "Datasets define the problems: scale, diversity, occlusion, modality. New benchmarks (MOT16, LaSOT, GOT-10k, DanceTrack) have repeatedly redirected research.",
    prereqs: ["mot", "sot"],
  },
  {
    id: "multimodal",
    name: "Multimodal / X-modal tracking",
    category: "paradigm",
    intuition:
      "Track with thermal, event, depth or audio alongside RGB — robustness when the RGB signal fails (night, heat, flicker), at the cost of modality fusion.",
    prereqs: ["sot"],
  },
  {
    id: "3d-tracking",
    name: "3D / point-cloud tracking",
    category: "paradigm",
    intuition:
      "Track objects in 3D (boxes in world space) from LiDAR/RGB-D — the state, motion and association math moves from the image plane to metric space.",
    prereqs: ["mot", "motion-model"],
  },
  {
    id: "diffusion-tracking",
    name: "Diffusion / generative tracking",
    category: "architecture",
    intuition:
      "Model the conditional distribution of the next box (or denoise a noisy track state) instead of regressing one point estimate — the 2023–2026 wave.",
    prereqs: ["attention"],
  },
  {
    id: "learned-association",
    name: "Learned affinity",
    category: "association",
    intuition:
      "Replace IoU distance with a learned similarity (appearance, relation, graph) — DeepSORT onward; the main axis of MOT improvement after SORT.",
    prereqs: ["reid", "cost-matrix"],
  },
];

export const CONCEPT_BY_ID: Record<string, Concept> = Object.fromEntries(
  CONCEPTS.map((c) => [c.id, c]),
);
