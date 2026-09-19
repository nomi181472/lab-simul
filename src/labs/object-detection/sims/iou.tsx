"use client";

import { useState } from "react";
import { iouFamily, type Box } from "@/labs/object-detection/sim/math";
import { Slider, Segmented, Sim } from "./shared";
import { BoxCanvas } from "./canvas";

const GT: Box = { x1: 70, y1: 45, x2: 150, y2: 100 };

export function IouSim() {
  const [x, setX] = useState(60);
  const [y, setY] = useState(40);
  const [w, setW] = useState(70);
  const [h, setH] = useState(45);
  const [metric, setMetric] = useState<"iou" | "giou" | "diou" | "ciou">("iou");

  const pred: Box = { x1: x, y1: y, x2: x + w, y2: y + h };
  const f = iouFamily(pred, GT);
  const activeVal = f[metric];
  const LOSS_KEY: Record<typeof metric, keyof ReturnType<typeof iouFamily>> = {
    iou: "lossIou",
    giou: "lossGioU",
    diou: "lossDioU",
    ciou: "lossCioU",
  };
  const activeLoss = f[LOSS_KEY[metric]];

  return (
    <Sim
      name="IoU family"
      identifies={`Similarity of prediction box vs ground truth (GT). Live ${metric.toUpperCase()} = ${activeVal.toFixed(3)}, loss = ${activeLoss.toFixed(3)}.`}
      math={[
        "IoU = |A ∩ B| / |A ∪ B|",
        "GIoU = IoU − (|C \\ (A∪B)| / |C|)    (C = smallest enclosing box)",
        "DIoU = IoU − ρ²(b,bᵍᵗ)/c²    (center distance normalized)",
        "CIoU = DIoU − α·v    (aspect-ratio penalty)",
      ]}
      need="IoU saturates and gives no gradient when boxes don't overlap. GIoU restores gradient via the enclosing box, DIoU adds center-distance, CIoU adds aspect matching. Families re-align the training objective with the evaluation metric (AP)."
      real="Annotation-matching thresholds (SSD), and the 2020 wave of IoU-based losses across YOLOv4 / PP-YOLO / PP-YOLOE / YOLOv7, and per-bin distribution refinement in D-FINE."
      papers={["P001", "P016", "P017", "P024", "P038"]}
      cites={[
        {
          text: "IoU-family regression losses are the box/loc branch throughout the corpus’s one-stage line.",
          papers: ["P001", "P016", "P017"],
        },
        {
          text: "CIoU-style aspect penalty and distribution refinement co-evolve with the YOLO/PP-YOLO real-time branch.",
          papers: ["P024", "P038"],
        },
      ]}
      params={
        <>
          <Slider label="pred x" value={x} min={10} max={180} step={1} onChange={setX} />
          <Slider label="pred y" value={y} min={10} max={110} step={1} onChange={setY} />
          <Slider label="pred w" value={w} min={10} max={150} step={1} onChange={setW} />
          <Slider label="pred h" value={h} min={10} max={90} step={1} onChange={setH} />
          <Segmented
            label="metric"
            value={metric}
            onChange={setMetric}
            options={[
              { value: "iou", label: "IoU" },
              { value: "giou", label: "GIoU" },
              { value: "diou", label: "DIoU" },
              { value: "ciou", label: "CIoU" },
            ]}
          />
        </>
      }
    >
      <BoxCanvas
        boxes={[
          { box: GT, color: "#fbbf24", label: "GT" },
          { box: pred, color: "#38bdf8", label: "pred" },
        ]}
      />
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        {(
          [
            ["iou", f.iou],
            ["giou", f.giou],
            ["diou", f.diou],
            ["ciou", f.ciou],
          ] as const
        ).map(([k, v]) => (
          <div
            key={k}
            className={`rounded-lg border p-2 text-center ${
              metric === k ? "border-emerald-700/60 bg-emerald-950/20" : "border-zinc-800"
            }`}
          >
            <div className="font-mono text-[9px] uppercase text-zinc-500">{k}</div>
            <div className="mt-1 font-mono text-sm text-zinc-200">{v.toFixed(3)}</div>
            <div className="font-mono text-[9px] text-zinc-600">loss {(1 - v).toFixed(3)}</div>
          </div>
        ))}
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Move the prediction far from the GT: IoU drops to 0 and cannot tell you how far off you
        are, while GIoU becomes negative — the gradient path never vanishes.
      </p>
    </Sim>
  );
}