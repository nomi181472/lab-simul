"use client";

import type { ComponentType } from "react";
import { IouSim } from "./iou";
import { NmsSim } from "./nms";
import { FocalSim } from "./focal";
import { AnchorsSim } from "./anchors";
import { PyramidSim } from "./pyramid";
import { RfSim } from "./rf";
import { AttentionSim, SelfAttnVsConvSim, TransformerQuerySim } from "./attention";
import { HungarianSim } from "./hungarian";
import { BoxRegressionSim, LossLabSim } from "./regression";
import { EdgeScenarioSim } from "./edge-scenarios";
import { DynamicAssignmentSim } from "./dynamic-assignment";
import { AnchorsVsFreeSim } from "./anchors-vs-free";
import { CascadeSim } from "./cascade";
import { DenoisingSim } from "./denoising";
import { DeformableAttnSim } from "./deformable";
import { MosaicSim } from "./mosaic";

export type SimId =
  | "iou"
  | "focal"
  | "nms"
  | "anchors"
  | "anchors-vs-free"
  | "pyramid"
  | "rf"
  | "attention"
  | "selfattn-vs-conv"
  | "transformer-query"
  | "deformable-attention"
  | "hungarian"
  | "dynamic-assignment"
  | "query-denoising"
  | "cascade"
  | "box-regression"
  | "loss-lab"
  | "mosaic"
  | "edge-scenarios";

export type SimEntry = {
  id: SimId;
  label: string;
  comp: ComponentType;
  blurb: string;
};

/** All interactive simulators. ids are the canonical sim ids referenced by
 *  Solution.simulator and ProblemCluster.simulator. */
export const SIM_REGISTRY: SimEntry[] = [
  { id: "iou", label: "IoU family", comp: IouSim,
    blurb: "IoU / GIoU / DIoU / CIoU overlap math and why the family exists." },
  { id: "focal", label: "Focal loss", comp: FocalSim,
    blurb: "Foreground/background imbalance and the focal down-weighting." },
  { id: "nms", label: "NMS", comp: NmsSim,
    blurb: "Greedy suppression of overlapping detections, threshold by threshold." },
  { id: "anchors", label: "Anchors", comp: AnchorsSim,
    blurb: "Anchor boxes, aspect ratios and matching to ground truth." },
  { id: "anchors-vs-free", label: "Anchor vs anchor-free", comp: AnchorsVsFreeSim,
    blurb: "Priors versus direct regression — the design split of the field." },
  { id: "pyramid", label: "Feature pyramid", comp: PyramidSim,
    blurb: "Multi-scale features: stride, level choice, small vs large objects." },
  { id: "rf", label: "Receptive field", comp: RfSim,
    blurb: "How depth and kernels grow the receptive field in pixels." },
  { id: "attention", label: "Attention", comp: AttentionSim,
    blurb: "Softmax attention over queries, keys and values." },
  { id: "selfattn-vs-conv", label: "Self-attn vs conv", comp: SelfAttnVsConvSim,
    blurb: "Global mixing vs local kernels — cost and reach." },
  { id: "transformer-query", label: "Transformer query", comp: TransformerQuerySim,
    blurb: "Learned queries reading the image through decoder cross-attention." },
  { id: "deformable-attention", label: "Deformable attention", comp: DeformableAttnSim,
    blurb: "Sparse, sampled attention around learned reference points." },
  { id: "hungarian", label: "Hungarian matching", comp: HungarianSim,
    blurb: "One-to-one bipartite matching and its many-to-one tension." },
  { id: "dynamic-assignment", label: "Dynamic assignment", comp: DynamicAssignmentSim,
    blurb: "Top-k and cost-based positive selection (SimOTA family)." },
  { id: "query-denoising", label: "Query denoising", comp: DenoisingSim,
    blurb: "Perturbed ground-truth queries that stabilize early training." },
  { id: "cascade", label: "Cascade refinement", comp: CascadeSim,
    blurb: "Stage-wise IoU thresholds refining boxes over heads." },
  { id: "box-regression", label: "Box regression", comp: BoxRegressionSim,
    blurb: "Encoding, offsets and gradient behaviour of box losses." },
  { id: "loss-lab", label: "Loss lab", comp: LossLabSim,
    blurb: "Compose detection losses and watch the surface change." },
  { id: "mosaic", label: "Mosaic augmentation", comp: MosaicSim,
    blurb: "Four-image mosaics: scale variety versus distribution shift." },
  { id: "edge-scenarios", label: "Edge scenarios", comp: EdgeScenarioSim,
    blurb: "Occlusion, crowds, tiny objects — where detectors break." },
];

export const SIM_BY_ID: Record<string, SimEntry> = Object.fromEntries(
  SIM_REGISTRY.map((s) => [s.id, s]),
);

export const SIM_IDS: string[] = SIM_REGISTRY.map((s) => s.id);
