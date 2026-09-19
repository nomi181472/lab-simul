import type { ProblemLifecycle } from "./types";

/* RESEARCH PROBLEM LIFECYCLES
 * Every problem below was verified against corpus text (see per-occurrence papers).
 * Status is deliberately conservative: "resolved" is only used where the corpus
 * states a problem no longer required active research (none qualify here).
 */

export const PROBLEM_LIFECYCLES: ProblemLifecycle[] = [
  {
    id: "small-objects",
    problem:
      "Small objects contribute little signal in deep, low-resolution feature maps and are disproportionately missed.",
    shortLabel: "Small objects",
    firstObservedYear: 2015,
    occurrences: [
      {
        year: 2015,
        papers: ["P001", "P002"],
        attemptedSolutions: [
          "Multi-scale feature maps (SSD) so small objects are handled in shallow, high-res layers.",
          "Specialized shallow-resolution layers retained for detection.",
        ],
        evidence: [
          "SSD: 'those small objects may not even have any information' in the deepest feature maps.",
          "YOLO: early variant struggled with small objects; grid-cell prediction weak on small/grouped cases.",
        ],
      },
      {
        year: 2017,
        papers: ["P004", "P005", "P008", "P010", "P012"],
        attemptedSolutions: [
          "Deconvolution to inject context into shallow layers (DSSD).",
          "Recurrent rolling convolution for context aggregation (RRC).",
          "Feature fusion before pyramid construction (FSSD).",
          "FPN-style context features for small-object layers (RF-UNet).",
          "Extending shallow layers (ExtSSD).",
        ],
        evidence: [
          "DSSD: improves 'accuracy, especially for small objects'.",
          "RRC: 'failure localization of either small objects or overlapping objects'.",
          "FSSD: SSD has a 'limitation to detect too large or too small objects'.",
        ],
      },
      {
        year: 2018,
        papers: ["P013", "P014"],
        attemptedSolutions: [
          "Three-scale prediction + per-class logistics in YOLOv3 (a claimed 'reversal' of YOLO's small-object weakness).",
          "Multi-level feature pyramid network (M2Det).",
        ],
        evidence: [
          "YOLOv3: 'In the past YOLO struggled with small objects. However, now we see a reversal'.",
        ],
      },
      {
        year: 2019,
        papers: ["P015"],
        attemptedSolutions: [
          "Attention-based fusion of features across levels (ASSD).",
        ],
        evidence: [
          "ASSD: shallow features 'insufficient for detecting small objects'; attention fusion addresses it.",
        ],
      },
      {
        year: 2021,
        papers: ["P023"],
        attemptedSolutions: [
          "Domain-adapted single-shot detector with feature pyramid for thermal small objects.",
        ],
        evidence: [
          "Thermal-SSD: 'Detecting small objects in thermal images using single-shot detector'.",
        ],
      },
      {
        year: 2022,
        papers: ["P031"],
        attemptedSolutions: [
          "Gather-and-Distribute feature exchange; cited real-time small-object detection on embedded devices.",
        ],
        evidence: [
          "Gold-YOLO: 'feature fusion pyramid network for real-time small object detection on embedded devices' (surveyed).",
        ],
      },
      {
        year: 2023,
        papers: ["P029"],
        attemptedSolutions: [
          "Hybrid encoder + uncertainty-minimal query selection advances DETR line but small objects still lag.",
        ],
        evidence: [
          "RT-DETR: 'the performance on small objects is still inferior than the strong CNN detectors'.",
        ],
      },
      {
        year: 2024,
        papers: ["P037", "P038", "P040", "P033"],
        attemptedSolutions: [
          "Distribution-guided fine-grained regression to fix coarse bins for small objects (D-FINE).",
          "Hierarchical dense positive supervision for small objects (RT-DETRv3).",
          "Group matching (DEIM: 'dense supervision is crucial, particularly for small objects').",
          "Small-object-friendly feature maps in NMS-free YOLO (YOLOv10).",
        ],
        evidence: [
          "D-FINE: coarse distribution -> 'coarse localization, especially for small objects'.",
          "DEIM: dense supervision crucial for small objects.",
        ],
      },
      {
        year: 2025,
        papers: ["P042", "P045"],
        attemptedSolutions: [
          "Higher input resolution discovered by NAS for smaller-object gains (RF-DETR).",
          "SOTA real-time lines report small-object gains lag large objects (DINOv3-DEIM).",
        ],
        evidence: [
          "DINOv3-DEIM: 'performance on small objects remains largely unchanged' vs large objects.",
        ],
      },
      {
        year: 2026,
        papers: ["P046", "P048"],
        attemptedSolutions: [
          "Small-Target-Aware Label Assignment (STAL) guaranteeing positive coverage for tiny objects (YOLO26).",
          "Occluder-leakage control for small objects in 3D lifting (OV-3D).",
        ],
        evidence: [
          "YOLO26: STAL 'guarantees positive candidate coverage for tiny objects'.",
        ],
      },
    ],
    currentStatus: "persistent",
    statusEvidence:
      "Present in the corpus from 2015 (SSD/YOLO) through 2026 (YOLO26/OV-3D). Never resolved; partially reduced by multi-scale features, FPNs, better fusion, attention, dense supervision and distribution regression, but every generation re-reports small/tiny objects as the weak category.",
  },
  {
    id: "localization",
    problem:
      "Precise box localization: predicted boxes are precise enough for mAP but the optimized target (L1/IoU) does not always match the metric.",
    shortLabel: "Box localization",
    firstObservedYear: 2015,
    occurrences: [
      {
        year: 2015,
        papers: ["P002"],
        attemptedSolutions: ["Grid-cell regression trained end-to-end."],
        evidence: ["YOLO: localization error relative to two-stage systems is a stated weakness."],
      },
      {
        year: 2017,
        papers: ["P005", "P006"],
        attemptedSolutions: [
          "Context recurrent rolling conv (RRC) for localization under clutter.",
          "Anchor refinement cascade (RefineDet).",
        ],
        evidence: ["RRC: 'failure localization of either small objects or overlapping objects'."],
      },
      {
        year: 2020,
        papers: ["P016", "P017"],
        attemptedSolutions: [
          "IoU-family losses: GIoU -> DIoU -> CIoU (YOLOv4); IoU-aware classification (PP-YOLO).",
          "DIoU-based NMS to pick better boxes (YOLOv4).",
        ],
        evidence: [
          "YOLOv4: 'IoU loss ... DIoU loss ... CIoU loss' lineage; adopts CIoU + DIoU-NMS.",
        ],
      },
      {
        year: 2024,
        papers: ["P038"],
        attemptedSolutions: [
          "Fine-grained distribution refinement over position bins (D-FINE).",
        ],
        evidence: ["D-FINE: distribution bins directly encode box-boundary hypotheses."],
      },
    ],
    currentStatus: "reduced",
    statusEvidence:
      "Localization improved substantially via IoU-family losses and distribution regression, but small-object and strict-metric precision still reported as weak across all years; not 'resolved'.",
  },
  {
    id: "class-imbalance",
    problem:
      "Dense one-stage detectors evaluate enormous numbers of background/anchor positions vs a handful of foreground objects - easy negatives dominate the loss.",
    shortLabel: "Class imbalance",
    firstObservedYear: 2017,
    occurrences: [
      {
        year: 2017,
        papers: ["P006", "P010"],
        attemptedSolutions: [
          "Two-step cascade to filter most anchors before classification (RefineDet).",
          "Focal-loss-style reweighting referenced in the one-stage literature (RF-UNet).",
        ],
        evidence: [
          "RefineDet: ARM filters negatives before heavy classification; notes focal loss is complementary.",
          "RF-UNet: references focal loss for single-stage rebalancing.",
        ],
      },
      {
        year: 2018,
        papers: ["P014"],
        attemptedSolutions: ["Focal loss adopted in training objective (M2Det)."],
        evidence: ["M2Det: 'focal loss ... thus gets AP of ...' (training objective)"],
      },
      {
        year: 2020,
        papers: ["P016", "P017"],
        attemptedSolutions: [
          "Focal loss formalized as the solution for data imbalance in one-stage (cited).",
          "IoU-aware weighting between branches (PP-YOLO).",
        ],
        evidence: ["YOLOv4: 'Lin et al. proposed focal loss to deal with the problem of data imbalance'."],
      },
      {
        year: 2022,
        papers: ["P024", "P026", "P027"],
        attemptedSolutions: [
          "Quality/VariFocal losses replacing plain focal for alignment (YOLOv6, DAMO-QFL).",
          "Task-aligned assignment reducing easy-negative dominance (PP-YOLOE).",
        ],
        evidence: [
          "YOLOv6: 'select VariFocal Loss as our classification loss'.",
          "DAMO-YOLO: 'Quality Focal Loss (QFL) for classification supervision'.",
        ],
      },
    ],
    currentStatus: "transformed",
    statusEvidence:
      "Plain cross-entropy imbalance largely addressed by focal-family and quality-aligned losses; the imbalance problem transformed into alignment/quality weighting (VFL/QFL/DFL) rather than disappearing.",
  },
  {
    id: "speed-latency",
    problem:
      "Inference speed/latency: detection must run in real time on constrained hardware; accuracy gains must not destroy usability.",
    shortLabel: "Inference speed",
    firstObservedYear: 2015,
    occurrences: [
      {
        year: 2015,
        papers: ["P002", "P001"],
        attemptedSolutions: [
          "Unified regression in one pass (YOLO).",
          "Single-shot multi-box scoring (SSD).",
        ],
        evidence: ["YOLO: 'Unified, Real-Time Object Detection' is the core pitch."],
      },
      {
        year: 2018,
        papers: ["P011"],
        attemptedSolutions: ["Efficient inverted-residual backbone + SSDLite head."],
        evidence: ["MobileNetV2: SSDLite targets mobile/latency-constrained detection."],
      },
      {
        year: 2020,
        papers: ["P016", "P017", "P018"],
        attemptedSolutions: [
          "Real-time GPU detectors with CSP backbones + neck; trick-stacking without new modules (PP-YOLO).",
          "Scaling across latency regimes (Scaled-YOLOv4).",
        ],
        evidence: ["PP-YOLO/YDOLOv4: real-time (fps) is the reported axis alongside mAP."],
      },
      {
        year: 2022,
        papers: ["P025", "P026", "P027"],
        attemptedSolutions: [
          "Re-parameterization to collapse multi-branch training modules into fast inference convs.",
          "NAS-designed backbones for latency (DAMO-YOLO).",
        ],
        evidence: ["YOLOv7/DAMO: speed-oriented 'bag of freebies' and T4-GPU latency reporting."],
      },
      {
        year: 2024,
        papers: ["P033", "P034"],
        attemptedSolutions: [
          "Removing NMS to cut latency (YOLOv10).",
          "Lightweight DETRs to replace YOLO (LW-DETR): decoder/encoder latency analysis.",
        ],
        evidence: ["YOLOv10: 'NMS hampers end-to-end deployment and adversely impacts inference latency'."],
      },
    ],
    currentStatus: "reduced",
    statusEvidence:
      "Massive speedups from 2015 (real-time) to NMS-free multi-scale real-time detectors; latency remains a live design axis (Pareto accuracy-latency), so the tension persists rather than resolving.",
  },
  {
    id: "nms",
    problem:
      "Non-maximum suppression: post-processing heuristics remove duplicate detections but add latency, hyper-parameters, and failure modes on crowded scenes.",
    shortLabel: "NMS dependence",
    firstObservedYear: 2015,
    occurrences: [
      {
        year: 2020,
        papers: ["P016", "P017"],
        attemptedSolutions: [
          "DIoU-NMS (distance-aware suppression) (YOLOv4).",
          "Matrix NMS for parallel suppression (PP-YOLO).",
        ],
        evidence: ["PP-YOLO: 'Matrix NMS' used; YOLOv4: 'DIoU-NMS'."],
      },
      {
        year: 2021,
        papers: ["P022"],
        attemptedSolutions: ["End-to-end YOLO variant: one-to-one assignment + stop gradient removes NMS."],
        evidence: ["YOLOX: 'end-to-end version ... NMS-free' via one-to-one labels."],
      },
      {
        year: 2023,
        papers: ["P029"],
        attemptedSolutions: ["Transformer set prediction eliminates NMS entirely in the DETR line."],
        evidence: ["RT-DETR: end-to-end detectors 'do not require NMS'."],
      },
      {
        year: 2024,
        papers: ["P033"],
        attemptedSolutions: ["Consistent dual assignment: NMS removed from YOLO with one-to-one head."],
        evidence: ["YOLOv10: NMS-free design as the headline."],
      },
      {
        year: 2026,
        papers: ["P046"],
        attemptedSolutions: ["One-to-many/one-to-one heads with only one-to-one at inference; no NMS."],
        evidence: ["YOLO26: 'NMS-free inference' as standard."],
      },
    ],
    currentStatus: "transformed",
    statusEvidence:
      "NMS was replaced (not just improved) along the end-to-end line: DETR set prediction and consistent dual assignment made NMS optional/absent. In earlier years it was only improved (DIoU-NMS, Matrix NMS).",
  },
  {
    id: "anchor-design",
    problem:
      "Anchor design: default-box priors (shape, scale, ratio, count) must be chosen/tuned; suboptimal priors hurt accuracy and complicate deployment.",
    shortLabel: "Anchor design",
    firstObservedYear: 2015,
    occurrences: [
      {
        year: 2015,
        papers: ["P001"],
        attemptedSolutions: ["Hand-specified default boxes with a few aspect ratios/scales."],
        evidence: ["SSD: default boxes defined by hand for each feature map."],
      },
      {
        year: 2016,
        papers: ["P003"],
        attemptedSolutions: ["k-means dimension clustering over training boxes for data-driven priors."],
        evidence: ["YOLO9000: 'dimension clusters' use k-means with IoU distance for anchors."],
      },
      {
        year: 2021,
        papers: ["P022", "P024"],
        attemptedSolutions: [
          "Anchor-free heads remove priors entirely (YOLOX, PP-YOLOE).",
          "Samples assigned by learned/dynamic rules (SimOTA/TAL).",
        ],
        evidence: [
          "YOLOX: 'We switch the YOLO detector to an anchor-free manner'.",
          "PP-YOLOE: 'anchor-free paradigm'.",
        ],
      },
    ],
    currentStatus: "transformed",
    statusEvidence:
      "Anchor priors were introduced (2015), made data-driven (2016), then removed by anchor-free heads and learned assignment (2021+). The problem transformed into label-assignment design.",
  },
  {
    id: "label-assignment",
    problem:
      "Label assignment: which anchor/position/query is a positive for each ground truth sharply changes training quality.",
    shortLabel: "Label assignment",
    firstObservedYear: 2021,
    occurrences: [
      {
        year: 2021,
        papers: ["P022"],
        attemptedSolutions: ["SimOTA top-k dynamic assignment."],
        evidence: ["YOLOX: 'leading label assignment strategy SimOTA'."],
      },
      {
        year: 2022,
        papers: ["P024", "P026", "P027"],
        attemptedSolutions: [
          "Task Alignment Learning (PP-YOLOE).",
          "OTA + anchor-based auxiliary branch (YOLOv6).",
          "Aligned-OTA with alignment metric (DAMO-YOLO).",
        ],
        evidence: ["PP-YOLOE: 'Task alignment learning'; YOLOv6: 'SimOTA ... direct'."],
      },
      {
        year: 2023,
        papers: ["P029", "P030"],
        attemptedSolutions: [
          "One-to-one Hungarian matching defined as DETR assignment (the bottleneck).",
          "Self-supervised pretraining to relieve sparse assignment training (DETR-PT).",
        ],
        evidence: [
          "DETR-PT: one-to-one matching is a 'training-efficiency bottleneck'.",
        ],
      },
      {
        year: 2024,
        papers: ["P040", "P033", "P037"],
        attemptedSolutions: [
          "MatchOR one-to-many group matching (DEIM).",
          "Consistent dual assignment one-to-many + one-to-one (YOLOv10).",
          "Hierarchical dense positive supervision (RT-DETRv3).",
        ],
        evidence: ["DEIM: 'DETR with improved matching for fast convergence'."],
      },
      {
        year: 2026,
        papers: ["P046"],
        attemptedSolutions: ["Small-Target-Aware Label Assignment for tiny-object coverage."],
        evidence: ["YOLO26: STAL 'guarantees positive candidate coverage for tiny objects'."],
      },
    ],
    currentStatus: "reduced",
    statusEvidence:
      "Assignment evolved from IoU-threshold rules to learned/dynamic cost-based matching (OTA/SimOTA/TAL) and dual/one-to-many frameworks; assignment choice still significantly affects results, so the problem is reduced, not closed.",
  },
  {
    id: "convergence",
    problem:
      "Convergence speed: DETR-style one-to-one matching provides sparse supervision -> slow training and hard optimization.",
    shortLabel: "DETR convergence",
    firstObservedYear: 2023,
    occurrences: [
      {
        year: 2023,
        papers: ["P029", "P030"],
        attemptedSolutions: [
          "Pretraining + denoising-style objectives (DETR-PT).",
          "Investigate the sparse one-to-one supervision cost (RT-DETR).",
        ],
        evidence: [
          "DETR-PT: 'one-to-one matching ... training-efficiency bottleneck'.",
          "RT-DETR: real-time DETR framework with efficiency from design + Hungarian matching.",
        ],
      },
      {
        year: 2024,
        papers: ["P040", "P037"],
        attemptedSolutions: [
          "MatchOR group matching to densify supervision (DEIM, 'training time by 50%').",
          "Dense positive supervision + self-attention perturbation (RT-DETRv3).",
        ],
        evidence: [
          "DEIM: improved matching 'fast convergence'.",
          "RT-DETRv3: Hungarian provides 'much sparser supervision' vs dense YOLO supervision.",
        ],
      },
    ],
    currentStatus: "reduced",
    statusEvidence:
      "Convergence is a recognized, actively-mitigated bottleneck of the DETR line; large training-time reductions reported, yet matching design remains a core design knob - reduced, not resolved.",
  },
  {
    id: "occlusion",
    problem:
      "Occlusion: partially or fully occluded targets are missed, merged, or lost across frames.",
    shortLabel: "Occlusion",
    firstObservedYear: 2017,
    occurrences: [
      {
        year: 2017,
        papers: ["P005"],
        attemptedSolutions: ["Context fusion to localize overlapping objects (RRC)."],
        evidence: ["RRC: 'failure localization of either small objects or overlapping objects'."],
      },
      {
        year: 2020,
        papers: ["P016", "P017"],
        attemptedSolutions: ["Better suppression (DIoU-NMS, Matrix NMS) for overlapping detections."],
        evidence: ["PP-YOLO/YOLOv4: NMS improvements for crowded overlaps."],
      },
      {
        year: 2026,
        papers: ["P047"],
        attemptedSolutions: [
          "Prediction/compensation under full and long-term occlusion in tracking (Kalman + Hungarian + YOLOv8n front-end).",
        ],
        evidence: [
          "Tracking-the-Unseen: 'robust under full and long-term occlusion'; MOTA and IDF1 gains.",
        ],
      },
    ],
    currentStatus: "persistent",
    statusEvidence:
      "Reported from 2017 (overlapping objects) to 2026 (full/long-term occlusion tracking). Partial solutions exist per-clue; full occlusion remains a stated failure regime.",
  },
  {
    id: "open-vocabulary",
    problem:
      "Open-vocabulary/open-set: detecting categories beyond the training label list - rare, unseen, or language-specified objects.",
    shortLabel: "Open vocabulary",
    firstObservedYear: 2016,
    occurrences: [
      {
        year: 2016,
        papers: ["P003"],
        attemptedSolutions: [
          "WordTree hierarchy + joint classification/detection to cover thousands of classes.",
        ],
        evidence: ["YOLO9000: detection of '9000 classes' via hierarchy."],
      },
      {
        year: 2023,
        papers: ["P028"],
        attemptedSolutions: [
          "Grounded pretraining with text prompts; language-aware queries; open-set detector (Grounding DINO).",
        ],
        evidence: ["Grounding DINO: 'open-set object detection' by marrying DINO with grounded pretraining."],
      },
      {
        year: 2024,
        papers: ["P039", "P035"],
        attemptedSolutions: [
          "Scaling pretraining data (20M images) + ViT backbones to push open-set 'edge' (GDINO 1.5).",
          "Lightweight open-vocabulary DETR distilling VLMs for edge (OVLW-DETR).",
        ],
        evidence: ["Grounding DINO 1.5: advancing the 'edge' of open-set detection."],
      },
      {
        year: 2026,
        papers: ["P048", "P046"],
        attemptedSolutions: [
          "Open-vocabulary 3D detection via promptable segmentation + 2D-to-3D lifting (OV-3D).",
          "YOLOE26 open-vocab extension in unified models (YOLO26).",
        ],
        evidence: ["OV-3D: 'open-vocabulary 3D object detection with promptable segmentation'."],
      },
    ],
    currentStatus: "transformed",
    statusEvidence:
      "The line starts in 2016 (hierarchy-based), becomes a real direction in 2023 (Grounding DINO), and expands to 3D + unified models by 2026. The problem transformed from 'few labels' to 'arbitrary text-conditioned categories with robust coverage'.",
  },
  {
    id: "domain-shift",
    problem:
      "Domain shift: detectors trained on natural images degrade on thermal, aerial, drone, maritime, weather-affected or deployment-specific domains.",
    shortLabel: "Domain shift",
    firstObservedYear: 2021,
    occurrences: [
      {
        year: 2021,
        papers: ["P023"],
        attemptedSolutions: ["Domain-adapted single-shot detector for thermal imagery."],
        evidence: ["Thermal-SSD: detection in thermal images with IR-specific small-object layout."],
      },
      {
        year: 2023,
        papers: ["P028"],
        attemptedSolutions: ["Grounding an open-set model; evaluation over diverse domains (aerial/drone/maritime...)."],
        evidence: ["Grounding DINO: ODinW-style domain benchmark sets."],
      },
      {
        year: 2024,
        papers: ["P039"],
        attemptedSolutions: ["Scaling data to cover 'edge' domains (potholes, thermal, aerial drones...)."],
        evidence: ["Grounding DINO 1.5: 'edge' domain suite."],
      },
      {
        year: 2025,
        papers: ["P045"],
        attemptedSolutions: ["NAS fine-tuning of a VLM-pretrained base for target domains."],
        evidence: ["RF-DETR: 'adaptable' via weight-sharing NAS to diverse target domains."],
      },
      {
        year: 2026,
        papers: ["P046"],
        attemptedSolutions: ["Domain generalization listed as frontier in the YOLO survey."],
        evidence: ["YOLO Survey: 'domain generalization' among open frontiers."],
      },
    ],
    currentStatus: "persistent",
    statusEvidence:
      "Domain shift is attacked from 2021 (thermal) onward, but remains an explicitly listed frontier problem in 2025-2026 works.",
  },
  {
    id: "deployability",
    problem:
      "Deployment: compute, memory, quantization, and latency constraints shape real deployable detectors.",
    shortLabel: "Deployment constraints",
    firstObservedYear: 2018,
    occurrences: [
      {
        year: 2018,
        papers: ["P011"],
        attemptedSolutions: ["MobileNetV2 + SSDLite for mobile."],
        evidence: ["MobileNetV2: detector variant designed for mobile devices."],
      },
      {
        year: 2020,
        papers: ["P016", "P017"],
        attemptedSolutions: ["Real-time GPU designs; edge-GPU metrics."],
        evidence: ["YOLOv4/PP-YOLO: fps on edge-class GPUs reported."],
      },
      {
        year: 2024,
        papers: ["P035"],
        attemptedSolutions: ["Lightweight open-vocabulary transformer for edge/VLM distillation."],
        evidence: ["OVLW-DETR: 'deployment friendly' open-vocab detector."],
      },
      {
        year: 2026,
        papers: ["P046"],
        attemptedSolutions: ["Native quantization support, NMS-free, unified multi-task."],
        evidence: ["YOLO26: quantization and deployment-native design; survey discusses compiler/thermal constraints."],
      },
    ],
    currentStatus: "reduced",
    statusEvidence:
      "Deployability improved steadily (mobile backbones, quantization, NMS-free, lightweight open-vocab); remains an active constraint but substantially addressed.",
  },
];

export const PROBLEM_BY_ID: Record<string, ProblemLifecycle> = Object.fromEntries(
  PROBLEM_LIFECYCLES.map((p) => [p.id, p]),
);

/** Matrix of problems present per year; derived from occurrences. */
export function buildProblemMatrix() {
  const years = Array.from({ length: 2026 - 2015 + 1 }, (_, i) => 2015 + i);
  return PROBLEM_LIFECYCLES.map((p) => {
    const present: Record<number, string[]> = {};
    for (const o of p.occurrences) {
      present[o.year] = (present[o.year] ?? []).concat(o.papers);
    }
    return {
      id: p.id,
      label: p.shortLabel,
      firstObservedYear: p.firstObservedYear,
      status: p.currentStatus,
      years,
      present,
    };
  });
}