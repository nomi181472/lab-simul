import type { DetectionScenario } from "./types";

/* EDGE CASES / DETECTION SCENARIOS discovered in the corpus.
 * Only scenarios with paper-backed evidence are listed.
 */

export const DETECTION_SCENARIOS: DetectionScenario[] = [
  {
    name: "Small objects",
    firstObservedYear: 2015,
    papers: ["P001", "P002", "P004", "P005", "P008", "P013", "P015", "P023", "P029", "P038", "P040"],
    difficulty: [
      "Little signal survives stride-heavy deep conv features.",
      "Fine boundary resolution needed for mAP contribution.",
      "Dense positive supervision needed in transformer lines.",
    ],
    attemptedSolutions: [
      "Multi-scale feature maps (SSD)",
      "Context injection (deconv/RRC/fusion: DSSD, FSSD, ASSD)",
      "Multi-level pyramids (M2Det) and multi-scale heads (YOLOv3)",
      "Distribution-guided fine regression (D-FINE)",
      "Dense supervision and group matching (RT-DETRv3, DEIM)",
      "Small-target-aware assignment (STAL, YOLO26)",
    ],
    remainingProblems: [
      "Every generation still reports small-object accuracy below large-object accuracy.",
      "Distribution-bin resolution limits (D-FINE).",
    ],
  },
  {
    name: "Tiny objects",
    firstObservedYear: 2026,
    papers: ["P046"],
    difficulty: ["Positive-coverage failure (tiny objects get no positive samples in assignment)."],
    attemptedSolutions: ["Small-Target-Aware Label Assignment guaranteeing coverage (YOLO26)."],
    remainingProblems: ["Still the hardest perceptual class; coverage improved, accuracy gap open."],
  },
  {
    name: "Dense / overlapping / grouped objects",
    firstObservedYear: 2015,
    papers: ["P002", "P005", "P016", "P017", "P047"],
    difficulty: [
      "Grouped instances overwhelm grid cells (YOLO).",
      "Localization failure for overlapping objects (RRC).",
      "NMS destroys adjacent true positives.",
    ],
    attemptedSolutions: [
      "Context fusion for overlap (RRC)",
      "DIoU-NMS and Matrix NMS (YOLOv4, PP-YOLO)",
      "One-to-one set prediction removes duplication (NMS-free line)",
    ],
    remainingProblems: ["Crowded/mutual occlusion still degrades identity association in tracking."],
  },
  {
    name: "Full / long-term occlusion",
    firstObservedYear: 2026,
    papers: ["P047"],
    difficulty: ["Targets disappear entirely for long frames; identity association breaks."],
    attemptedSolutions: [
      "Prediction/compensation + Kalman + Hungarian association with detection front-end (Tracking-the-Unseen).",
    ],
    remainingProblems: ["MOTA/IDF1 improved but full-occlusion re-identification remains hard."],
  },
  {
    name: "Rotation / oriented objects",
    firstObservedYear: 2017,
    papers: ["P009", "P046"],
    difficulty: ["Axis-aligned boxes poorly fit rotated, dense aerial objects."],
    attemptedSolutions: ["Rotatable bounding boxes (RotBox, 2017); oriented detection in unified YOLO26."],
    remainingProblems: ["Oriented regression + evaluation complexity; smaller ecosystem."],
  },
  {
    name: "Thermal / low-feature imagery",
    firstObservedYear: 2021,
    papers: ["P023", "P039"],
    difficulty: ["Low visual detail; different appearance distribution from natural images."],
    attemptedSolutions: ["Domain-adapted single-shot detector (Thermal-SSD); open-set coverage of thermal edge cases (Grounding DINO 1.5)."],
    remainingProblems: ["Domain gap persists; data scarcity for IR objects."],
  },
  {
    name: "Aerial / long-range scenes",
    firstObservedYear: 2017,
    papers: ["P009", "P028", "P043"],
    difficulty: ["Distant, rotated, tiny objects; huge image extents."],
    attemptedSolutions: ["Oriented detection; open-set/generalist models evaluated over aerial-drone benchmarks (Grounding DINO, YOLO survey)."],
    remainingProblems: ["Coverage and annotation cost for aerial domains."],
  },
  {
    name: "Open-vocabulary / unknown objects",
    firstObservedYear: 2016,
    papers: ["P003", "P028", "P039", "P048", "P046"],
    difficulty: [
      "Categories not in the training set.",
      "Requires language-conditioned or hierarchy-based outputs.",
      "Long-tail names and unseen domains.",
    ],
    attemptedSolutions: [
      "WordTree hierarchy (YOLO9000)",
      "Grounded pretraining + text prompts (Grounding DINO)",
      "Edge-domain data scaling (Grounding DINO 1.5)",
      "3D open-vocab via promptable segmentation (OV-3D)",
    ],
    remainingProblems: ["'Edge' categories and true open-world generalization still open."],
  },
  {
    name: "Rare / long-tail classes",
    firstObservedYear: 2016,
    papers: ["P003", "P039"],
    difficulty: ["Few positive examples; imbalance against common classes."],
    attemptedSolutions: ["Hierarchy sharing (YOLO9000); 20M-image grounded pretraining with edge-domain coverage (GDINO 1.5)."],
    remainingProblems: ["Long-tail recall still trails common classes."],
  },
  {
    name: "Domain shift (natural -> specialized)",
    firstObservedYear: 2021,
    papers: ["P023", "P028", "P039", "P045", "P043"],
    difficulty: ["Same objects, different imaging: thermal, aerial, maritime, underwater, weather."],
    attemptedSolutions: [
      "Domain-adapted detectors (Thermal-SSD)",
      "Generalist open-set evaluation (Grounding DINO/1.5)",
      "NAS fine-tuning with VLM pretraining (RF-DETR)",
      "Domain generalization as survey frontier (YOLO survey)",
    ],
    remainingProblems: ["Listed explicitly as an open frontier in 2025-2026 works."],
  },
  {
    name: "Adverse / edge-condition objects (weather, potholes)",
    firstObservedYear: 2024,
    papers: ["P039"],
    difficulty: ["Crowded, degraded, infra-style objects far from benchmark classes."],
    attemptedSolutions: ["Explicit 'edge' evaluation suite (Grounding DINO 1.5)."],
    remainingProblems: ["Recognition robustness under adversarial conditions still limited."],
  },
  {
    name: "3D range / truncation (occluder leakage)",
    firstObservedYear: 2026,
    papers: ["P048"],
    difficulty: ["Lifting masks to 3D leaks occluders into small objects; truncates large objects."],
    attemptedSolutions: ["Promptable-seg 2D->3D lifting with range handling (OV-3D)."],
    remainingProblems: ["Boundary/range fidelity of lifted 3D boxes."],
  },
];

export const SCENARIO_BY_NAME: Record<string, DetectionScenario> = Object.fromEntries(
  DETECTION_SCENARIOS.map((s) => [s.name, s]),
);

export const EDGE_SCENE_TYPES = [
  "small",
  "tiny",
  "dense",
  "occluded",
  "rotated",
  "thermal",
  "aerial",
  "unknown",
  "rare",
  "shift",
] as const;
export type EdgeSceneType = (typeof EDGE_SCENE_TYPES)[number];