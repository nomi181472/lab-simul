"use client";

import { useLab } from "@/labs/object-detection/context";
import { SimErrorBoundary } from "@/labs/object-detection/sims/error-boundary";
import { IouSim } from "@/labs/object-detection/sims/iou";
import { NmsSim } from "@/labs/object-detection/sims/nms";
import { FocalSim } from "@/labs/object-detection/sims/focal";
import { AnchorsSim } from "@/labs/object-detection/sims/anchors";
import { PyramidSim } from "@/labs/object-detection/sims/pyramid";
import { RfSim } from "@/labs/object-detection/sims/rf";
import { AttentionSim, SelfAttnVsConvSim, TransformerQuerySim } from "@/labs/object-detection/sims/attention";
import { HungarianSim } from "@/labs/object-detection/sims/hungarian";
import { BoxRegressionSim, LossLabSim } from "@/labs/object-detection/sims/regression";
import { EdgeScenarioSim } from "@/labs/object-detection/sims/edge-scenarios";
import { DynamicAssignmentSim } from "@/labs/object-detection/sims/dynamic-assignment";
import { AnchorsVsFreeSim } from "@/labs/object-detection/sims/anchors-vs-free";
import { CascadeSim } from "@/labs/object-detection/sims/cascade";
import { DenoisingSim } from "@/labs/object-detection/sims/denoising";
import { DeformableAttnSim } from "@/labs/object-detection/sims/deformable";
import { MosaicSim } from "@/labs/object-detection/sims/mosaic";

const TABS = [
  { id: "iou", label: "IoU family", comp: IouSim },
  { id: "focal-loss", label: "Focal loss", comp: FocalSim },
  { id: "nms", label: "NMS", comp: NmsSim },
  { id: "anchors", label: "Anchors", comp: AnchorsSim },
  { id: "anchor-vs-free", label: "Anchor vs anchor-free", comp: AnchorsVsFreeSim },
  { id: "feature-pyramid", label: "Feature pyramid", comp: PyramidSim },
  { id: "receptive-field", label: "Receptive field", comp: RfSim },
  { id: "attention", label: "Attention", comp: AttentionSim },
  { id: "selfattention-vs-conv", label: "Self-attn vs conv", comp: SelfAttnVsConvSim },
  { id: "transformer-query", label: "Transformer query", comp: TransformerQuerySim },
  { id: "deformable-attention", label: "Deformable attention", comp: DeformableAttnSim },
  { id: "hungarian", label: "Hungarian", comp: HungarianSim },
  { id: "dynamic-assignment", label: "Dynamic assignment", comp: DynamicAssignmentSim },
  { id: "query-denoising", label: "Query denoising", comp: DenoisingSim },
  { id: "cascade", label: "Cascade refinement", comp: CascadeSim },
  { id: "box-regression", label: "Box regression", comp: BoxRegressionSim },
  { id: "loss-lab", label: "Loss lab", comp: LossLabSim },
  { id: "mosaic", label: "Mosaic augmentation", comp: MosaicSim },
  { id: "edge-scenarios", label: "Edge scenarios", comp: EdgeScenarioSim },
];

export function MathView() {
  const { mathSim, setMathSim } = useLab();
  const active = TABS.some((t) => t.id === mathSim) ? mathSim : "iou";
  const ActiveComp = TABS.find((t) => t.id === active)!.comp;

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Mathematics lab</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          Every concept in the lab runs as a simulator: textbook math, deterministic seeds,
          and a live link to the corpus claims. Change a parameter and watch the math change.
        </p>
      </div>

      <div className="flex flex-wrap gap-1.5">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setMathSim(t.id)}
            className={`rounded-md border px-2.5 py-1 text-[11px] transition-colors ${
              active === t.id
                ? "border-emerald-600 bg-emerald-600/20 text-emerald-200"
                : "border-zinc-800 text-zinc-500 hover:border-zinc-700 hover:text-zinc-300"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <SimErrorBoundary key={active} label={TABS.find((t) => t.id === active)?.label}>
        <ActiveComp />
      </SimErrorBoundary>
    </div>
  );
}