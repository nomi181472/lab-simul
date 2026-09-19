"use client";

import { useState } from "react";
import { nms, nmsDiou } from "@/labs/object-detection/sim/math";
import { generateDetectionCandidates } from "@/labs/object-detection/sim/mock";
import { Slider, Segmented, Seeded, Sim } from "./shared";
import { BoxCanvas } from "./canvas";

export function NmsSim() {
  const [seed, setSeed] = useState(11);
  const [conf, setConf] = useState(0.3);
  const [iouThr, setIouThr] = useState(0.5);
  const [mode, setMode] = useState<"standard" | "diou">("standard");

  const candList = generateDetectionCandidates(seed);
  const runner = mode === "standard" ? nms : nmsDiou;
  const { kept, steps } = runner(candList, conf, iouThr);
  const keptIds = new Set(kept);

  return (
    <Sim
      name="Non-Maximum Suppression"
      identifies={`${candList.length} raw proposals; ${kept.length} kept after conf≥${conf.toFixed(2)} and IoU>${iouThr.toFixed(2)} suppression.`}
      math={[
        "Sort proposals by score, descending.",
        "Pick highest; suppress any remaining proposal of the SAME class whose IoU with it exceeds the threshold.",
        "Repeat until empty. Output the kept set.",
      ]}
      need="Classical one-stage and two-stage detectors emit many overlapping boxes per object. NMS collapses duplicates to one detection; too aggressive and dense crowds are wiped out, too loose and one object emits many boxes."
      real="NMS was assumed necessary from 2015–2020, improved by DIoU-NMS / Matrix NMS (PP-YOLO / Scaled-YOLOv4, 2020), then replaced by one-to-one set prediction (2023 DETR line) and dual-assignment heads (2024–2026 yolo)."
      papers={["P001", "P016", "P017", "P022", "P029", "P033", "P046"]}
      cites={[
        {
          text: "Matrix NMS runs suppression in parallel, cutting latency on dense detections.",
          papers: ["P022", "P017"],
        },
        {
          text: "NMS-free set prediction (DETR line) and dual one-to-one/many-to-one heads remove the knob entirely.",
          papers: ["P029", "P033", "P046"],
        },
      ]}
      params={
        <>
          <Seeded seed={seed} onReseed={() => setSeed(Math.floor(Math.random() * 1000))} />
          <Slider label="confidence threshold" value={conf} min={0} max={0.9} step={0.05} onChange={setConf} />
          <Slider label="IoU threshold" value={iouThr} min={0.2} max={0.99} step={0.01} onChange={setIouThr} />
          <Segmented
            label="mode"
            value={mode}
            onChange={setMode}
            options={[
              { value: "standard", label: "standard NMS" },
              { value: "diou", label: "DIoU-NMS" },
            ]}
          />
        </>
      }
    >
      <BoxCanvas
        boxes={candList.map((c) => ({
          box: c.box,
          color: keptIds.has(c.id) ? "#34d399" : "#71717a",
          label: `${c.cls.slice(0, 1)}${c.score.toFixed(2)}`,
          dashed: !keptIds.has(c.id),
        }))}
      />
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <div>
          <div className="font-mono text-[9px] uppercase text-emerald-400">kept</div>
          <div className="mt-1 space-y-1">
            {kept.length === 0 && <div className="text-[11px] text-zinc-600">none — lower IoU threshold for fewer suppressions</div>}
            {kept.map((id) => {
              const c = candList.find((k) => k.id === id)!;
              return (
                <div key={id} className="flex items-center gap-2 font-mono text-[11px]">
                  <span className="text-emerald-300">{c.cls}</span>
                  <span className="text-zinc-500">score {c.score.toFixed(2)}</span>
                </div>
              );
            })}
          </div>
        </div>
        <div>
          <div className="font-mono text-[9px] uppercase text-rose-400">suppression trace</div>
          <div className="mt-1 space-y-1">
            {steps.length === 0 && <div className="text-[11px] text-zinc-600">no proposals above threshold</div>}
            {steps.map((s, i) => {
              const pick = candList.find((c) => c.id === s.pick)!;
              return (
                <div key={i} className="text-[11px] leading-5 text-zinc-500">
                  pick <span className="text-emerald-300">{pick.cls}</span>
                  {s.suppressed.length > 0
                    ? ` → suppresses ids ${s.suppressed.join(", ")}`
                    : " → nothing to suppress"}
                </div>
              );
            })}
          </div>
        </div>
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Lower the IoU threshold to zero: all overlapping cars collapse to one. Raise it to 0.99:
        duplicates survive. This is the delicate speed-quality knob every NMS-based detector tuned.
      </p>
    </Sim>
  );
}