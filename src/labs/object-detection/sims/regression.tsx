"use client";

import { useState } from "react";
import { boxRegressionLoss, iou, type Box } from "@/labs/object-detection/sim/math";
import { Slider, Segmented, Sim } from "./shared";
import { BoxCanvas } from "./canvas";

const GT: Box = { x1: 70, y1: 45, x2: 150, y2: 100 };

export function BoxRegressionSim() {
  const [x, setX] = useState(70);
  const [y, setY] = useState(45);
  const [w, setW] = useState(80);
  const [h, setH] = useState(55);
  const [kind, setKind] = useState<"l1" | "smooth-l1" | "l2">("smooth-l1");

  const pred: Box = { x1: x, y1: y, x2: x + w, y2: y + h };
  const { loss, perCoord } = boxRegressionLoss(pred, GT, kind);
  const iouV = iou(pred, GT);

  const rel = (v: number, i: number) => {
    const g = [75, 72.5, 80, 55][i]; // GT center cx, cy, w, h
    return v / (Math.abs(g) || 1);
  };

  return (
    <Sim
      name="Box regression losses"
      identifies={`Prediction vs GT: loss ${loss.toFixed(3)} (${kind}), IoU ${iouV.toFixed(3)}. Coordinate deltas shown per-head.`}
      math={[
        "deltas: (Δcx, Δcy, Δw, Δh) between pred and GT box parameters",
        "L1: Σ|Δ|          L2: ΣΔ²          Smooth-L1: ½Δ² if |Δ|<β else |Δ|−β/2",
        "detector heads read these deltas off anchors/prior boxes and decode final boxes",
      ]}
      need="Predicting box parameters directly: vision-language (4 coordinates) vs label-space losses. L2 over-penalizes large outliers, pure L1 under-weights tiny fixes; Smooth-L1 blends both — the standard in SSD/Faster/DETR heads."
      real="Smooth-L1 heavy use in the two-stage and DETR lineage; D-FINE replaces the single-offset prediction with a full distribution over offsets (next sim: loss lab ties to distribution decode)."
      papers={["P001", "P037", "P038", "P024"]}
      params={
        <>
          <Slider label="pred x" value={x} min={20} max={180} step={1} onChange={setX} />
          <Slider label="pred y" value={y} min={20} max={110} step={1} onChange={setY} />
          <Slider label="pred w" value={w} min={10} max={150} step={1} onChange={setW} />
          <Slider label="pred h" value={h} min={10} max={90} step={1} onChange={setH} />
          <Segmented
            label="loss"
            value={kind}
            onChange={setKind}
            options={[
              { value: "l1", label: "L1" },
              { value: "smooth-l1", label: "Smooth-L1" },
              { value: "l2", label: "L2" },
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
      <div className="mt-3 grid grid-cols-4 gap-2">
        {["Δcx", "Δcy", "Δw", "Δh"].map((k, i) => (
          <div key={k} className="rounded-lg border border-zinc-800 p-2 text-center">
            <div className="font-mono text-[9px] uppercase text-zinc-500">{k}</div>
            <div className="mt-1 font-mono text-sm text-zinc-200">
              {(perCoord[i] / (i < 2 ? 1 : 1)).toFixed(2)}
            </div>
            <div className="font-mono text-[8px] text-zinc-600">
              rel {rel(perCoord[i], i).toFixed(2)}
            </div>
          </div>
        ))}
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Shove the pred far outside the GT: L2 blows up super-linearly while L1 stays calm;
        Smooth-L1 keeps the outlier gradient capped. The 4 boxes are exactly what the box head
        outputs — the loss identifies which term the network is minimizing.
      </p>
    </Sim>
  );
}

export function LossLabSim() {
  const [g, setG] = useState(2);
  const [doBox, setDoBox] = useState(true);
  const [doCls, setDoCls] = useState(true);

  const clsTotal = focalShare(g);
  const boxShare = 1 - clsTotal;

  return (
    <Sim
      name="Loss laboratory (combined objective)"
      identifies={`The supervision signal a YOLO/DETR head maximizes: classification ${Math.round(clsTotal * 100)}% + box regression ${Math.round(boxShare * 100)}% (γ=${g}). Toggle components to see the total change shape.`}
      math={[
        "L_total = λ_cls · L_cls(γ)  +  λ_box · L_box(Smooth-L1 / IoU-family)",
        "imbalance rebalancing: focal/DFL for class+distribution (see Focal sim)",
        "rescaled, the two terms must be tuned together or one silences the other",
      ]}
      need="A detector is never trained on one objective. YOLOv4 configures entire “bags of specials/freebies” trading λ’s; PP-YOLOE unifies loss design; D-FINE iterates the distribution head. This sim shows which component dominates at your γ."
      real="YOLOv4 v3’s config of loss weights, PP-YOLOE’s aligned classification+localization, D-FINE’s distribution refinement on top."
      papers={["P016", "P017", "P024", "P038"]}
      params={
        <>
          <Slider label="γ for class term" value={g} min={0} max={5} step={0.1} onChange={setG} />
          <Segmented
            label="class term"
            value={doCls ? "on" : "off"}
            onChange={(v) => setDoCls(v === "on")}
            options={[
              { value: "on", label: "classifier on" },
              { value: "off", label: "classifier off" },
            ]}
          />
          <Segmented
            label="box term"
            value={doBox ? "on" : "off"}
            onChange={(v) => setDoBox(v === "on")}
            options={[
              { value: "on", label: "box on" },
              { value: "off", label: "box off" },
            ]}
          />
        </>
      }
    >
      <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-3">
        <div className="mb-2 flex justify-between font-mono text-[9px] text-zinc-600">
          <span>classifier allowance ({doCls ? "active" : "muted"})</span>
          <span>box allowance ({doBox ? "active" : "muted"})</span>
        </div>
        <div className="flex h-8 w-full overflow-hidden rounded-md border border-zinc-800">
          <div
            className="bg-emerald-500/70 transition-all"
            style={{ width: `${(doCls ? clsTotal : 0) * 100}%` }}
          />
          <div
            className="bg-sky-500/70 transition-all"
            style={{ width: `${(doCls && doBox ? boxShare : doBox ? 1 : 0) * 100}%` }}
          />
          <div className="flex-1 bg-zinc-800" />
        </div>
        <div className="mt-1 flex justify-between font-mono text-[9px] text-zinc-600">
          <span>cls {Math.round((doCls ? clsTotal : 0) * 100)}%</span>
          <span>box {Math.round((doCls && doBox ? boxShare : doBox ? 100 : 0) * 100)}%</span>
          <span>idle</span>
        </div>
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Raise γ and the classifier produces almost no signal, leaving the box branch to pull
        alone — the exact configuration error that makes detectors “regress boxes everywhere”.
      </p>
    </Sim>
  );
}

function focalShare(gamma: number) {
  // monotone heuristic: classifier share decays with γ (educational, not a corpus claim)
  return Math.max(0.05, Math.min(0.95, 0.8 * Math.pow(0.82, gamma)));
}