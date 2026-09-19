import type { ResearchDirection } from "./types";

/* NEW RESEARCH DIRECTIONS detected in the corpus.
 * "Growth" fields are inferred from the years of representative papers.
 */

export const RESEARCH_DIRECTIONS: ResearchDirection[] = [
  {
    id: "open-vocabulary",
    name: "Closed-set -> Open-vocabulary detection",
    firstAppearance: 2016,
    sourceIn: ["Fixed label-set detection (P001/P002)", "Hierarchy-based joint training (P003)"],
    representativePapers: ["P003", "P028", "P039", "P035", "P046", "P048"],
    driver: "Long-tail and unseen categories cannot be covered by fixed label lists with box annotations.",
    previousLimitation: "2000-class VGG/NMS pipelines and fixed training vocabularies stop generalizing to new names.",
    unlockedCapability: "Detect arbitrary categories described by text; 3D open-vocabulary by 2026.",
    newProblems: [
      "Vocabulary/domain coverage gaps ('edge' categories).",
      "Huge pretraining data requirements (20M-image scale).",
      "Evaluation methodology for open sets.",
    ],
    status: "HIGH",
  },
  {
    id: "nms-free",
    name: "NMS-dependent -> NMS-free / end-to-end detection",
    firstAppearance: 2021,
    sourceIn: ["Post-processing pipelines (NMS) in P001/P002/P016/P017", "One-to-one set prediction (external DETR, via P029/P030)"],
    representativePapers: ["P022", "P029", "P033", "P037", "P046"],
    driver: "NMS adds latency, hyper-parameters, and duplicate-suppression failure modes; end-to-end set prediction is permutation-invariant.",
    previousLimitation: "Latency + heuristics of post-processing in real-time and edge deployment.",
    unlockedCapability: "NMS-free latency-critical deployment; single-head one-to-one inference.",
    newProblems: [
      "Sparse supervision -> convergence cost.",
      "Dual/one-to-many matching design complexity.",
      "Head-consistency between training and inference.",
    ],
    status: "HIGH",
  },
  {
    id: "anchor-free",
    name: "Anchor-based -> Anchor-free detection",
    firstAppearance: 2021,
    sourceIn: ["Anchor/default-box priors (P001)", "Data-driven priors (P003)"],
    representativePapers: ["P022", "P024", "P026", "P033"],
    driver: "Anchor sets are sensitive hyper-parameters; static matching mis-assigns positives across scales.",
    previousLimitation: "Prior design burden, imbalance, dataset-specific re-clustering.",
    unlockedCapability: "Simpler heads, no prior design; assignment moved to learned/dynamic rules.",
    newProblems: ["Center-point assumptions", "Dynamic assignment (SimOTA/TAL) tuning burden"],
    status: "HIGH",
  },
  {
    id: "transformer-detr",
    name: "CNN grid -> Transformer set-prediction (real-time DETR)",
    firstAppearance: 2023,
    sourceIn: ["Dense grid/one-stage CNN heads (P002/P013)", "Attention import (P015/P041)"],
    representativePapers: ["P029", "P030", "P036", "P037", "P034", "P042", "P044", "P045"],
    driver: "Enable NMS-free end-to-end detection and content-dependent global interactions in real time.",
    previousLimitation: "NMS dependence + local receptive fields of CNN heads.",
    unlockedCapability: "Pareto-strong real-time NMS-free detectors; query-based set prediction.",
    newProblems: [
      "Decoder latency domination.",
      "Convergence under sparse matching (addressed by dense/dual supervision).",
      "Small-object inferiority in early DETRs.",
    ],
    status: "HIGH",
  },
  {
    id: "efficiency-edge",
    name: "Accuracy-first -> Real-time / edge efficiency",
    firstAppearance: 2015,
    sourceIn: ["Two-stage accuracy pipelines (referenced)", "Real-time one-stage (P002)"],
    representativePapers: ["P002", "P011", "P016", "P017", "P026", "P035", "P046"],
    driver: "Deployment reality: cameras, robots, embedded devices with bounded compute.",
    previousLimitation: "Proposal-stage and NMS latency; heavy backbones.",
    unlockedCapability: "fps-class detection at all scales; quantization-native NMS-free models.",
    newProblems: ["Capacity/accuracy loss at high efficiency", "Hardware-specific kernels/quantization"],
    status: "HIGH",
  },
  {
    id: "unified-multitask",
    name: "Single-task -> Unified multi-task vision",
    firstAppearance: 2021,
    sourceIn: ["Single-task detectors (P001/P002/P013)"],
    representativePapers: ["P021", "P046"],
    driver: "Deploying one model for detection + segmentation/pose/tracking/classification reduces system cost.",
    previousLimitation: "Per-task heads and duplicated pipelines.",
    unlockedCapability: "One shared representation/one pass for many tasks.",
    newProblems: ["Task head contention", "Multi-task loss balancing"],
    status: "MEDIUM",
  },
  {
    id: "nas-foundation",
    name: "Hand-designed -> NAS / foundation-model driven design",
    firstAppearance: 2022,
    sourceIn: ["Hand-designed real-time architectures (P016/P025)", "NAS discourse (P025/P018)"],
    representativePapers: ["P027", "P045", "P044", "P042"],
    driver: "Explosion of hand-tuned freebies; need Pareto-optimal latency-accuracy without manual search.",
    previousLimitation: "Manual module engineering cost and reproducibility.",
    unlockedCapability: "Automated Pareto discovery; VLM-pretraining adaptation to new domains.",
    newProblems: ["Search cost/transferability", "Dependence on foundation-model weights"],
    status: "MEDIUM",
  },
  {
    id: "matching-convergence",
    name: "Static matching -> Matching/label-assignment engineering",
    firstAppearance: 2021,
    sourceIn: ["IoU-threshold assignment (P001)", "Dynamic assignment (P022)"],
    representativePapers: ["P022", "P024", "P026", "P040", "P037", "P046"],
    driver: "Detection quality tracks who gets called a positive for each ground truth.",
    previousLimitation: "Fixed assignment cannot adapt to scale/difficulty; sparse one-to-one slows DETRs.",
    unlockedCapability: "Assignment as a design axis (SimOTA/TAL/dual/group/STAL).",
    newProblems: ["New hyper-parameters (top-k, costs)", "Training/inference matching alignment"],
    status: "HIGH",
  },
  {
    id: "distribution-regression",
    name: "Point/coordinate regression -> Distribution-based regression",
    firstAppearance: 2022,
    sourceIn: ["Smooth-L1 coordinate regression (P001)", "IoU-family losses (P016)"],
    representativePapers: ["P024", "P027", "P038"],
    driver: "Single-scalar box regression is coarse; distributions express boundary uncertainty.",
    previousLimitation: "Bin/scale resolution and loss metric mismatch.",
    unlockedCapability: "Fine-grained iterative refinement (D-FINE), quality-aware losses.",
    newProblems: ["Bin resolution limits for small objects", "Dense supervision needs"],
    status: "MEDIUM",
  },
  {
    id: "beyond-2d",
    name: "2D image detection -> 3D / temporal detection+tracking",
    firstAppearance: 2025,
    sourceIn: ["Image-only detection (P001..P025)"],
    representativePapers: ["P048", "P047", "P042"],
    driver: "Real deployments (driving, robotics) need geometry and time; open-vocabulary reaches 3D.",
    previousLimitation: "2D boxes lack depth; tracking broken by occlusion; 3D annotation scarcity.",
    unlockedCapability: "Open-vocabulary 3D detection via lifting; occlusion-robust tracking.",
    newProblems: ["3D annotation scarcity → lifting reliance", "Occluder leakage", "Full/long-term occlusion"],
    status: "MEDIUM",
  },
];

export const DIRECTION_BY_ID: Record<string, ResearchDirection> = Object.fromEntries(
  RESEARCH_DIRECTIONS.map((d) => [d.id, d]),
);

/** Research oscillations detected in the corpus (lab synthesis, evidence-cited). */
export const RESEARCH_OSCILLATIONS = [
  {
    pattern: "hand-designed priors -> learned methods -> new architectural priors",
    cycle: [
      { year: 2015, state: "hand-designed anchors + fixed matching", representativePapers: "P001" },
      { year: 2016, state: "learned prior shapes (k-means)", representativePapers: "P003" },
      { year: 2021, state: "learned dynamic assignment (SimOTA), anchor-free", representativePapers: "P022" },
      { year: 2024, state: "new structural priors (dual assignment, STAL coverage)", representativePapers: "P033, P046" },
    ],
    evidence:
      "The field oscillates between explicit priors (anchors/matching rules) and learned alternatives; 2024-2026 re-introduces structural priors (dual heads, small-target-aware assignment) as design choices.",
    status: "labInterpretation" as const,
  },
  {
    pattern: "complexity -> efficiency -> complexity -> optimization",
    cycle: [
      { year: 2018, state: "efficiency: mobile backbones", representativePapers: "P011" },
      { year: 2020, state: "complexity: bag of freebies/specials", representativePapers: "P016" },
      { year: 2022, state: "efficiency: re-param + NAS + real-time families", representativePapers: "P025, P027" },
      { year: 2024, state: "complexity: dual heads, distribution regression, matching design", representativePapers: "P033, P038" },
    ],
    evidence:
      "Corpus shows repeated swings between training/architectural sophistication and efficiency-focused simplifications.",
    status: "crossPaper" as const,
  },
  {
    pattern: "accuracy -> speed -> accuracy (one-stage vs two-stage / CNN vs DETR)",
    cycle: [
      { year: 2015, state: "speed-first one-stage vs accuracy two-stage", representativePapers: "P002" },
      { year: 2020, state: "accuracy via tricks within speed budget", representativePapers: "P016" },
      { year: 2023, state: "accuracy frontier via end-to-end DETR, then re-optimized for speed", representativePapers: "P029" },
      { year: 2024, state: "convergence + NMS-free of all families", representativePapers: "P033, P037" },
    ],
    evidence:
      "Every generation renegotiates the accuracy-latency trade rather than resolving it; latency is a persistent design pressure (see Inference speed lifecycle).",
    status: "crossPaper" as const,
  },
];