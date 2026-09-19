"use client";

import { useMemo, useState } from "react";
import { cascadeRefine, type Box } from "@/labs/object-detection/sim/math";
import { BoxCanvas } from "./canvas";
import { Seeded, Sim, Slider } from "./shared";

const INIT: Box = { x1: 56, y1: 30, x2: 172, y2: 118 };
const GT: Box = { x1: 92, y1: 52, x2: 138, y2: 92 };

export function CascadeSim() {
  const [seed, setSeed] = useState(21);
  const [jitter, setJitter] = useState(0.5);

  const stages = useMemo(() => cascadeRefine(INIT, GT, seed), [seed]);

  const shown = useMemo(() => {
    if (jitter <= 0.01) return stages.map((s) => ({ ...s, box: s.box }));
    const ratio = jitter;
    return stages.map((s) => ({
      ...s,
      box: {
        x1: INIT.x1 + (s.box.x1 - INIT.x1) * ratio,
        y1: INIT.y1 + (s.box.y1 - INIT.y1) * ratio,
        x2: INIT.x2 + (s.box.x2 - INIT.x2) * ratio,
        y2: INIT.y2 + (s.box.y2 - INIT.y2) * ratio,
      },
    }));
  }, [stages, jitter]);

  const finalIoU = stages[stages.length - 1]?.iouVal ?? 0;

  return (
    <Sim
      name="Cascade refinement (Cascade R-CNN)"
      identifies={`Three heads at IoU thresholds 0.5 → 0.6 → 0.7 lift IoU from ${stages[0]?.iouVal.toFixed(3) ?? 0} to ${finalIoU.toFixed(3)} on the same box.`}
      math={[
        "head r is trained on positives with IoU ≥ t_r,  t = [0.5, 0.6, 0.7]",
        "stage r: box_{r+1} = refine(box_r)  ← regression trained at occupancy t_r",
        "each re-sampled positive gives the next head a tighter training set",
      ]}
      need="A single head trained at IoU 0.5 must handle candidates of very different quality. Cascade splits the job: each subsequent head assumes its input already overlaps the GT at a higher threshold, so its regression only has to finish the job — and it does, stage by stage, with fewer false positives introduced."
      real="The corpus documents the two-stage cascade heritage that the DETR line replaces end-to-end; the tradeoff is extra stages and latency versus cleaner localization on hard boxes."
      papers={["P016", "P017"]}
      params={
        <>
          <Seeded seed={seed} onReseed={() => setSeed(Math.floor(Math.random() * 1000))} />
          <Slider label="regression jitter" value={jitter} min={0} max={1} step={0.05} onChange={setJitter} />
        </>
      }
    >
      <BoxCanvas
        boxes={[
          { box: GT, color: "#fbbf24", label: "gt" },
          { box: INIT, color: "#f87171", label: "0" },
          ...shown.map((s, i) => ({
            box: s.box,
            color: ["#f59e0b", "#34d399", "#22d3ee"][i],
            label: `${s.stage} (t=${s.threshold})`,
            dashed: false,
          })),
        ]}
      />
      <div className="mt-3 grid grid-cols-3 gap-3">
        {shown.map((s, i) => (
          <div key={s.stage} className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-2 text-center">
            <div className="font-mono text-[9px] uppercase text-zinc-500">stage {s.stage}</div>
            <div className="font-mono text-sm" style={{ color: ["#f59e0b", "#34d399", "#22d3ee"][i] }}>
              IoU {s.iouVal.toFixed(3)}
            </div>
            <div className="font-mono text-[10px] text-zinc-500">t={s.threshold}</div>
            <div className="mt-1 h-2 w-full overflow-hidden rounded bg-zinc-800">
              <div className="h-full bg-emerald-500" style={{ width: `${s.iouVal * 100}%` }} />
            </div>
          </div>
        ))}
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Each stage’s head is therefore “tuned up”; the delta-evidence box gap closes. Raising
        the threshold per stage also means later heads never see easy, overlapping
        negatives — exactly the ↔ two-stage repricing the end-to-end line revisits.
      </p>
    </Sim>
  );
}