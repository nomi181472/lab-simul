"use client";

import { useMemo, useState } from "react";
import { generateAnchors, bestAnchorsForBox, type Box } from "@/labs/object-detection/sim/math";
import { Slider, Sim } from "./shared";
import { BoxCanvas, VIEW_W, VIEW_H } from "./canvas";

const GT: Box = { x1: 88, y1: 48, x2: 168, y2: 92 };

export function AnchorsSim() {
  const [scale, setScale] = useState(1);
  const [cell, setCell] = useState(48);
  const [ratioStr, setRatioStr] = useState<"1" | "2" | "0.5">("1");
  const [scaleMul, setScaleMul] = useState(1);

  const gridW = Math.ceil(VIEW_W / cell);
  const gridH = Math.ceil(VIEW_H / cell);

  const anchors = useMemo(
    () => generateAnchors(gridW, gridH, cell, [scale * scaleMul], [Number(ratioStr)]),
    [gridW, gridH, cell, scale, scaleMul, ratioStr],
  );

  const top = useMemo(() => bestAnchorsForBox(anchors, GT, 1), [anchors]);

  return (
    <Sim
      name="Anchors over a grid"
      identifies={`${anchors.length} anchors on a ${gridW}×${gridH} grid (cell ${cell}px). Best anchor IoU with the GT object: ${top[0] ? top[0].iou.toFixed(3) : "0"}.`}
      math={[
        "anchor = pre-defined box (scale s, ratio r) centered at each grid cell",
        "backbone head predict deltas (tx, ty, tw, th) to refine the nearest anchor into a box",
        "best anchor = argmax IoU(anchor, GT) over all anchors",
      ]}
      need="Hand-designing anchors (scale/ratio/tiling) was the pivot between one-stage SSD and two-stage proposals: the detector regresses from an anchor rather than predicting boxes from nothing. Choosing wrong shapes leaves poor coverage and loses small or elongated objects."
      real="SSD’s fixed scale/ratio priors gave way to k-means anchor shapes (YOLOv9000), then to entirely anchor-free heads (PP-YOLOE, DAMO-YOLO, RT-DETR) once per-location dense prediction matured."
      papers={["P001", "P003", "P017", "P024", "P027"]}
      params={
        <>
          <Slider label="anchor scale" value={scale} min={0.3} max={2} step={0.05} onChange={setScale} />
          <Slider label="base cell (px)" value={cell} min={24} max={70} step={1} onChange={setCell} />
          <Slider label="scale multiplier" value={scaleMul} min={0.5} max={2.5} step={0.1} onChange={setScaleMul} />
          <div className="flex flex-col gap-1">
            <span className="font-mono text-[10px] uppercase text-zinc-500">aspect ratio</span>
            <div className="flex gap-1">
              {(["1", "2", "0.5"] as const).map((r) => (
                <button
                  key={r}
                  onClick={() => setRatioStr(r)}
                  className={`rounded-md border px-2 py-1 text-[11px] ${
                    ratioStr === r
                      ? "border-emerald-600 bg-emerald-600/20 text-emerald-200"
                      : "border-zinc-800 text-zinc-500"
                  }`}
                >
                  {r}:1
                </button>
              ))}
            </div>
          </div>
        </>
      }
    >
      <BoxCanvas
        boxes={[
          ...anchors.map((a, i) => ({
            box: a,
            color: top[0]?.i === i ? "#fbbf24" : "#3f3f46",
            dashed: true,
          })),
          { box: GT, color: "#34d399", label: "GT" },
        ]}
      />
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        The anchor whose shape hits GT best (highlighted amber) becomes the regression
        reference. Sink the scale multiplier low or pick the wrong ratio and no anchor
        overlaps the GT — a detector that needs a proposal can&apos;t even start.
      </p>
    </Sim>
  );
}