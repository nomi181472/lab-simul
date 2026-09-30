import type { MetricRecord } from "./types";

/* Metric registry. Formulas are quoted from corpus papers:
 *  - MOTA / MOTP / MT-ML: MOT16 benchmark paper (T006, Sec 4.1)
 *  - HOTA / DetA / AssA / IDF1: AVTrack (T125, Eqs 11–14) and EECTracker (T156, Eqs 24–27)
 * Missing formula = not documented in the corpus yet; never invent one.
 */

export const METRIC_IDS = [
  "mota",
  "motp",
  "idf1",
  "hota",
  "deta",
  "assa",
  "idsw",
  "fp",
  "fn",
  "fragments",
  "mt-ml",
  "faf",
  "precision",
  "recall",
  "success-auc",
  "norm-precision",
  "eao",
  "fps",
  "3d-error",
] as const;

export type MetricId = (typeof METRIC_IDS)[number];

export const METRICS: MetricRecord[] = [
  {
    id: "mota",
    name: "MOTA (Multiple Object Tracking Accuracy)",
    family: "association",
    formula: "MOTA = 1 − Σ_t (FN_t + FP_t + IDSW_t) / Σ_t GT_t",
    variables: [
      { symbol: "FN_t", meaning: "ground-truth objects missed in frame t" },
      { symbol: "FP_t", meaning: "hypotheses with no ground-truth match in frame t" },
      { symbol: "IDSW_t", meaning: "identity switches in frame t" },
      { symbol: "GT_t", meaning: "number of ground-truth objects in frame t" },
    ],
    meaning:
      "Everything that goes wrong (misses, false alarms, identity switches) as a fraction of how many objects there were.",
    intuition:
      "One number per sequence in (−∞, 100]. Because FP and FN usually dwarf IDSW counts, MOTA mostly measures how good your detector is.",
    example:
      "SORT on MOT test: 7318 FP + 32615 FN + 1001 IDSW → MOTA 33.4.",
    limitations: [
      "Corpus MOT16 paper itself: 'highly debatable whether this number alone can serve as a single performance measure'.",
      "Identity errors are numerically negligible next to FP/FN, so a tracker can switch identities constantly and still score well.",
      "Sensitive to detector quality rather than tracking logic (SORT's detector-swap table: same tracker moves 15.1 → 34.0 MOTA).",
    ],
    simulator: "metrics",
    paperIds: ["T006", "T005"],
  },
  {
    id: "motp",
    name: "MOTP (Multiple Object Tracking Precision)",
    family: "localization",
    formula: "MOTP = Σ_{t,i} d_{t,i} / Σ_t c_t",
    variables: [
      { symbol: "d_{t,i}", meaning: "overlap of matched hypothesis i with its ground truth in frame t" },
      { symbol: "c_t", meaning: "number of matches in frame t" },
    ],
    meaning: "Average overlap between correctly matched boxes — pure localization quality.",
    intuition:
      "How tight your boxes are once matched. Ranges from the matching threshold (e.g. 50%) to 100%.",
    example: "SORT reports MOTP 72.1 on the MOT test set.",
    limitations: [
      "The corpus MOT16 paper states it 'mostly quantifies the localization accuracy of the detector, and therefore provides little information about the actual performance of the tracker'.",
    ],
    paperIds: ["T006", "T005"],
  },
  {
    id: "idf1",
    name: "IDF1",
    family: "association",
    formula: "IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN)",
    variables: [
      { symbol: "IDTP", meaning: "ground-truth identities correctly and consistently tracked" },
      { symbol: "IDFP / IDFN", meaning: "hypothesis identities that should not exist / missed identities" },
    ],
    meaning: "Global identity score over the whole sequence instead of per-frame counting.",
    intuition:
      "Rewards keeping one ID on one person for the entire video: a single identity cut in half counts as one IDTP plus IDFN/IDFP, not as a tiny per-frame error.",
    example: "DeepSORT's headline improvements are usually reported in IDF1 next to MOTA.",
    limitations: [
      "Still depends on the detector's recall — missed frames break identity matches by construction.",
    ],
    paperIds: ["T125", "T156"],
  },
  {
    id: "hota",
    name: "HOTA (Higher-Order Tracking Accuracy)",
    family: "association",
    formula: "HOTA = (1/|A|) Σ_{α∈A} √(DetA_α · AssA_α)",
    variables: [
      { symbol: "α", meaning: "the IoU matching threshold; results are averaged over the set A of thresholds" },
      { symbol: "DetA_α", meaning: "detection accuracy TP/(TP+FN+FP) at threshold α" },
      { symbol: "AssA_α", meaning: "association accuracy averaged over matched detections at threshold α" },
    ],
    meaning:
      "Geometric mean of localization/detection quality and association quality, averaged over matching thresholds.",
    intuition:
      "Designed so neither a perfect detector with bad IDs nor a perfect ID tracker with bad boxes can win — association is weighted equally with detection at every threshold.",
    example:
      "Corpus papers report the decomposition side by side, e.g. HOTA 85.8 / AssA 85.8 at K=3 (LEGO, T093).",
    limitations: [
      "Threshold-averaged: it hides behaviour at any particular operating point.",
      "Still moves with detector quality (DetA term), so cross-paper comparisons must fix the detections.",
    ],
    simulator: "metrics",
    paperIds: ["T125", "T156"],
  },
  {
    id: "deta",
    name: "DetA (detection accuracy)",
    family: "detection",
    formula: "DetA = TP / (TP + FN + FP)",
    variables: [
      { symbol: "TP", meaning: "matched detections" },
      { symbol: "FN, FP", meaning: "missed and spurious detections" },
    ],
    meaning: "Standard detection F-score component of HOTA, computed per matching threshold.",
    intuition: "Same counting as detection's precision/recall pairing, expressed as one ratio.",
    example: "Reported alongside HOTA in modern MOT papers (e.g. DepthMOT, MambaMOT-era results tables).",
    limitations: ["Purely detection-driven: does not know anything about identity."],
    paperIds: ["T125"],
  },
  {
    id: "assa",
    name: "AssA (association accuracy)",
    family: "association",
    formula: "AssA = (1/|C|) Σ_{c∈C} |TPA(c)| / (|TPA(c)| + |FNA(c)| + |FPA(c)|)",
    variables: [
      { symbol: "C", meaning: "set of true-positive matches at the chosen threshold" },
      { symbol: "TPA(c), FNA(c), FPA(c)", meaning: "true-positive, false-negative and false-positive associations induced by match c" },
    ],
    meaning: "For each matched detection, how consistently its identity is linked across frames.",
    intuition:
      "Measures whether the same ID sticks to the same object over time — the part of HOTA that IoU-only trackers like SORT score poorly on.",
    example: "ByteTrack-style improvements typically raise AssA while DetA stays fixed (same detections).",
    limitations: ["Only defined over matched detections — a tracker that loses targets entirely can hide fragmentation here."],
    paperIds: ["T125"],
  },
  {
    id: "idsw",
    name: "ID switches (IDSW)",
    family: "association",
    formula: "IDSW = Σ_t IDSW_t",
    variables: [{ symbol: "IDSW_t", meaning: "times a tracked ID jumps from one ground-truth object to another in frame t" }],
    meaning: "Raw count of identity errors.",
    intuition: "Every switch is a moment where the system said 'this is a new person' about the wrong object.",
    example: "SORT reports 1001 ID switches on MOT test; DeepSORT's appearance model exists to drive this toward zero.",
    limitations: ["Absolute counts are not comparable across datasets of different length/density."],
    paperIds: ["T005", "T013"],
  },
  {
    id: "fp",
    name: "False positives (FP)",
    family: "detection",
    formula: "FP = Σ_t FP_t",
    variables: [{ symbol: "FP_t", meaning: "hypotheses with no ground-truth counterpart in frame t" }],
    meaning: "Tracker output that does not correspond to any real object.",
    intuition: "Ghost tracks — usually caused by detector noise or by a track kept alive too long.",
    example: "SORT: 7318 FP on MOT test; ByteTrack explicitly uses low-score boxes, which is a FP/Recall trade.",
    limitations: ["Counts detector errors as tracker errors — inherent to tracking-by-detection."],
    paperIds: ["T005", "T061"],
  },
  {
    id: "fn",
    name: "False negatives (FN)",
    family: "detection",
    formula: "FN = Σ_t FN_t",
    variables: [{ symbol: "FN_t", meaning: "ground-truth objects with no hypothesis in frame t" }],
    meaning: "Real objects the system failed to output.",
    intuition: "Misses — the dominant error term in crowded scenes and the reason ByteTrack rescues low-score detections.",
    example: "SORT: 32,615 FN on MOT test, far outweighing FP and IDSW.",
    limitations: ["Detector-limited: no association logic can recover an object that was never detected."],
    paperIds: ["T005", "T061"],
  },
  {
    id: "fragments",
    name: "Fragmentations (Frag)",
    family: "association",
    formula: "Frag = number of times a ground-truth track is interrupted by misses",
    variables: [],
    meaning: "How often an identity is broken and restarted.",
    intuition: "A fragment is a continuity failure without necessarily an identity switch — the track died and was reborn.",
    example: "SORT reports 1764 fragmentations on MOT test.",
    limitations: ["Sensitive to track-management parameters (max_age / T_Lost)."],
    paperIds: ["T005"],
  },
  {
    id: "mt-ml",
    name: "Mostly Tracked / Mostly Lost (MT/ML)",
    family: "robustness",
    formula: "MT: recovered ≥ 80% of a trajectory · ML: recovered < 20%",
    variables: [],
    meaning: "Share of trajectories tracked for most / almost none of their length.",
    intuition: "Distribution-level view: does the tracker survive entire trajectories, not just single frames?",
    example: "SORT: MT 11.7%, ML 30.9% on MOT test.",
    limitations: ["Ignores identity continuity by design (the corpus MOT16 paper notes ID may change and it still counts)."],
    paperIds: ["T006", "T005"],
  },
  {
    id: "faf",
    name: "False alarms per frame (FAF)",
    family: "detection",
    formula: "FAF = FP / number of frames",
    variables: [],
    meaning: "Rate of spurious output per frame.",
    intuition: "Normalizes false positives so sequences of different length can be compared.",
    example: "SORT: 1.3 FAF on MOT test.",
    limitations: ["Still detector-driven."],
    paperIds: ["T005"],
  },
  {
    id: "success-auc",
    name: "Success plot AUC (SOT)",
    family: "localization",
    formula: "AUC = ∫₀¹ fraction of frames with IoU(pred, gt) > θ dθ",
    variables: [
      { symbol: "θ", meaning: "overlap threshold swept from 0 to 1" },
    ],
    meaning: "Area under the curve of 'how many frames are tracked above threshold θ'.",
    intuition:
      "A single score that is robust to the choice of threshold: a tracker that is almost-right scores better than one that alternates between perfect and lost.",
    example: "Standard protocol on OTB / LaSOT / GOT-10k; the corpus SOT surveys tabulate AUC for every method.",
    limitations: [
      "Short-term and box-based: says nothing about identity switches or long-term re-detection on its own.",
      "Sensitive to the ground-truth box convention (eccentricity padding etc.).",
    ],
    paperIds: ["T066", "T103"],
  },
  {
    id: "precision",
    name: "Precision plot (SOT)",
    family: "localization",
    formula: "precision(θ) = fraction of frames with center distance < θ px",
    variables: [{ symbol: "θ", meaning: "center-error threshold in pixels (commonly reported at 20 px)" }],
    meaning: "How often the predicted center is within θ pixels of ground truth.",
    intuition: "Position-only accuracy; ignores box size entirely.",
    example: "Reported next to success AUC in OTB-style evaluations.",
    limitations: ["Fixed pixel threshold is not scale-aware — a 20 px error means something different for a 30 px target and a 300 px target."],
    paperIds: ["T066"],
  },
  {
    id: "norm-precision",
    name: "Normalized precision (SOT)",
    family: "localization",
    formula: "center error normalized by ground-truth box size",
    variables: [],
    meaning: "Scale-aware version of the precision plot.",
    intuition: "Makes errors comparable between small and large targets — important on LaSOT where scales vary hugely.",
    example: "Reported on LaSOT alongside success AUC.",
    limitations: ["Still position-only."],
    paperIds: ["T049"],
  },
  {
    id: "eao",
    name: "EAO (Expected Average Overlap)",
    family: "robustness",
    formula: "EAO = average overlap over a window of typical short-sequence lengths",
    variables: [],
    meaning: "VOT's estimate of expected tracking quality on a short sequence without re-initialization.",
    intuition:
      "Simulates the realistic 'run until failure' regime: unlike OTB's per-frame success, a single failure drags the whole score down.",
    example: "Primary VOT challenge number; used by CF and Siamese trackers evaluated on VOT.",
    limitations: ["Tied to VOT's restart protocol — not comparable with OTB-style AUC."],
    paperIds: ["T011"],
  },
  {
    id: "fps",
    name: "Speed (FPS / Hz)",
    family: "efficiency",
    formula: "FPS = frames processed per second",
    variables: [],
    meaning: "Tracking throughput, usually excluding (or including, if stated) detection and feature extraction.",
    intuition:
      "The constraint that decides deployability: SORT's 260 Hz and SiamFC's 58 FPS were headline claims.",
    example: "SORT reports 260 Hz for the tracking component on a single core.",
    limitations: [
      "Reporting is inconsistent across papers (GPU vs CPU, with/without detector, batch size) — compare only like with like.",
    ],
    paperIds: ["T005", "T009"],
  },
  {
    id: "3d-error",
    name: "3D localization error",
    family: "localization",
    formula: "mean 3D distance / translation error between predicted and GT box centre",
    variables: [],
    meaning: "Metric-space accuracy for 3D trackers (BCOT and point-cloud trackers).",
    intuition: "Image-plane IoU does not exist in 3D — errors are measured in centimetres/millimetres instead.",
    example: "BCOT's headline evaluation protocol for markerless 3D tracking.",
    limitations: ["Requires 3D ground truth; not comparable with 2D IoU metrics."],
    paperIds: ["T072"],
  },
];

export const METRIC_BY_ID: Record<string, MetricRecord> = Object.fromEntries(
  METRICS.map((m) => [m.id, m]),
);
