import type { ApplicationLink, Confidence } from "./types";

/* INDUSTRY APPLICATIONS extracted from the corpus.
 * Every application links the research capability that enabled it and the
 * real-world constraint it feeds back into research (research <-> industry loop).
 * Any app that could not be sourced from the corpus is excluded (not listed).
 */

type App = ApplicationLink & { available: Confidence };

export const APPLICATIONS: App[] = [
  {
    id: "real-time-embedded",
    name: "Real-time & embedded detection",
    researchCapability: "Single-pass regression (YOLO), one-stage multi-box (SSD), mobile backbones.",
    whyUseful: "Frame-rate detection on GPUs and mobile CPUs/embedded devices without proposal-stage latency.",
    constraints: "Latency budgets, low compute, small memory.",
    detectorEvolution: "YOLO (2015) -> MobileNetV2/SSDLite (2018) -> YOLOv4-family real-time (2020) -> DAMO-YOLO/NAS re-param (2022) -> NMS-free YOLO/DETR (2023-2026).",
    edgeCases: "Small objects under latency caps; scale variance in one pass.",
    paperIds: ["P002", "P001", "P011", "P016", "P027", "P033", "P046"],
    firstAppearance: 2015,
    status: "HIGH",
    available: "HIGH",
  },
  {
    id: "industrial",
    name: "Industrial deployment at scale",
    researchCapability: "Curated training trick-stacking and deployment-friendly architectures.",
    whyUseful: "Reproducible, deployable accuracy without research novelty - PP-YOLO/PP-YOLOv2 and Meituan YOLOv6 target company-scale serving.",
    constraints: "Fixed hardware, throughput, stability, reproducibility.",
    detectorEvolution: "PP-YOLO (2020) -> PP-YOLOv2 (2021) -> YOLOv6 (2022) -> PP-YOLOE (2022) aligned heads.",
    edgeCases: "Imbalanced real-world data, quantization drift, crowded production scenes.",
    paperIds: ["P017", "P020", "P024", "P026"],
    firstAppearance: 2020,
    status: "HIGH",
    available: "HIGH",
  },
  {
    id: "autonomous-vehicles",
    name: "Autonomous driving / vehicles",
    researchCapability: "Real-time robust detection and (by 2026) 3D open-vocabulary boxes.",
    whyUseful: "Vehicles need dense, low-latency detection and, in 3D, geometry for planning.",
    constraints: "Latency, safety-critical robustness, occlusion, range.",
    detectorEvolution: "YOLOv7 cites autonomous-driving contexts; DEIM cites driving 3D detection; OV-3D tackles 3D open-vocabulary in driving-like scenes.",
    edgeCases: "Occlusion, tiny far objects, range truncation, open-vocabulary road agents.",
    paperIds: ["P025", "P040", "P048"],
    firstAppearance: 2022,
    status: "MEDIUM",
    available: "MEDIUM",
  },
  {
    id: "aerial-drone",
    name: "Aerial / satellite / drone imagery",
    researchCapability: "Oriented boxes and robust small-object handling.",
    whyUseful: "Rotated, dense, small objects and distance-viewing are common in satellite/drone imagery.",
    constraints: "Rotation invariance, huge image sizes, class imbalance.",
    detectorEvolution: "RotBox (2017) introduces rotatable bboxes for aerial/satellite; Grounding DINO/1.5 evaluate drone/aerial benchmarks; YOLO survey uses aerial imagery.",
    edgeCases: "Rotated dense objects, tiny objects from altitude, aerial domain shift.",
    paperIds: ["P009", "P028", "P039", "P043"],
    firstAppearance: 2017,
    status: "HIGH",
    available: "HIGH",
  },
  {
    id: "thermal-ir",
    name: "Thermal / infrared inspection",
    researchCapability: "Single-shot detection adapted to low-feature thermal imagery.",
    whyUseful: "IR cameras operate in darkness/heat; small warm objects need detection.",
    constraints: "Low visual detail, domain gap from RGB.",
    detectorEvolution: "Thermal-SSD (2021) adapts SSD+FPN to thermal; Grounding DINO 1.5 evaluates thermal edge cases.",
    edgeCases: "Low-contrast IR objects, small/dim targets, thermal domain shift.",
    paperIds: ["P023", "P039"],
    firstAppearance: 2021,
    status: "HIGH",
    available: "HIGH",
  },
  {
    id: "tracking-surveillance",
    name: "Tracking & surveillance",
    researchCapability: "Fast, reliable per-frame detection feeding data-association pipelines.",
    whyUseful: "Multi-object tracking depends on detection quality; occlusion breaks naive association.",
    constraints: "Long occlusions, crowded scenes, real-time.",
    detectorEvolution: "YOLOv7/RT-DETR cite tracking/surveillance contexts; Tracking-the-Unseen (2026) builds occlusion-robust tracking on YOLOv8n + Kalman/Hungarian.",
    edgeCases: "Full and long-term occlusion, identity switches.",
    paperIds: ["P025", "P029", "P047"],
    firstAppearance: 2022,
    status: "HIGH",
    available: "HIGH",
  },
  {
    id: "open-vocab-retrieval",
    name: "Open-vocabulary / prompt-based search",
    researchCapability: "Grounding arbitrary text to boxes (Grounding DINO/SAM-style pipelines).",
    whyUseful: "Search and caption-style tasks need detecting objects never seen in training.",
    constraints: "Vocabulary coverage, long-tail names, domain specificity.",
    detectorEvolution: "Grounding DINO (2023) -> Grounding DINO 1.5 scaling (2024) -> YOLOE26 open-vocab (2026).",
    edgeCases: "Rare class names, adversarial weather objects, unseen domains.",
    paperIds: ["P028", "P039", "P046"],
    firstAppearance: 2023,
    status: "HIGH",
    available: "HIGH",
  },
  {
    id: "maritime",
    name: "Maritime / shore monitoring",
    researchCapability: "Robust detection over large distances and low-contrast water scenes.",
    whyUseful: "Coastal/maritime surveillance uses detection over open water.",
    constraints: "Long-range small targets, weather, domain shift.",
    detectorEvolution: "Grounding DINO evaluates AerialMaritimeDrone benchmarks.",
    edgeCases: "Long-range small vessels, glare/weather degradation.",
    paperIds: ["P028"],
    firstAppearance: 2023,
    status: "MEDIUM",
    available: "MEDIUM",
  },
  {
    id: "robotics-mobile",
    name: "Robotics & mobile autonomy",
    researchCapability: "Single-pass real-time detection usable on robot compute.",
    whyUseful: "Robots need latency-critical, edge-runnable perception.",
    constraints: "Compute ceilings, NMS-free latency, multitask.",
    detectorEvolution: "YOLOv10 cites robot navigation; YOLO26 unified models; OV-3D 3D capability.",
    edgeCases: "Occlusion, motion, unified-task heads on shared compute.",
    paperIds: ["P033", "P046", "P048"],
    firstAppearance: 2024,
    status: "MEDIUM",
    available: "MEDIUM",
  },
  {
    id: "agriculture-medical",
    name: "Agriculture & medical analysis",
    researchCapability: "Affordable real-time detection across specialized imagery.",
    whyUseful: "Survey-authors document aerial agriculture and medical-analysis use of YOLO models; LW-DETR benchmarks on microscopic domains.",
    constraints: "Domain-specific imagery, low-resource settings.",
    detectorEvolution: "YOLO survey (2025) catalogs agriculture/medical; LW-DETR class-agnostic domain evaluation (aerial, videogames, microscopic, underwater).",
    edgeCases: "High domain shift, rare findings, microscopic scale variance.",
    paperIds: ["P043", "P034"],
    firstAppearance: 2024,
    status: "MEDIUM",
    available: "MEDIUM",
  },
  {
    id: "underwater",
    name: "Underwater / non-natural domains",
    researchCapability: "Class-agnostic and open-set detection robustness.",
    whyUseful: "Underwater imagery breaks RGB priors; general-purpose detectors are evaluated there.",
    constraints: "Color distortion, low contrast, domain shift.",
    detectorEvolution: "LW-DETR evaluates underwater domains for real-time detection.",
    edgeCases: "Domain shift, low-contrast objects.",
    paperIds: ["P034"],
    firstAppearance: 2024,
    status: "MEDIUM",
    available: "MEDIUM",
  },
];

/** The research <-> industry feedback loop, converted to a chain of nodes. */
export const INDUSTRY_LOOP: {
  label: string;
  detail: string;
}[] = [
  {
    label: "Better real-time detection",
    detail: "Single-pass regression + multi-scale pyramids (2015-2020)",
  },
  {
    label: "New capability",
    detail: "NMS-free, GPU/edge-class latency detection (2021-2024)",
  },
  {
    label: "New application",
    detail: "Cameras/robotics/drones/AV/open-vocab search (2017-2026)",
  },
  {
    label: "New real-world constraint",
    detail: "Limited compute, quantization, occlusion, domain shift, latency",
  },
  {
    label: "New research problem",
    detail: "Efficient detectors, NMS removal, assignment design, domain adaptation",
  },
  {
    label: "New architecture research",
    detail: "CSP backbones, anchor-free heads, re-param, NAS, dual-assignment, attention",
  },
];

export function applicationById(id: string) {
  return APPLICATIONS.find((a) => a.id === id);
}