"use client";

import { useMemo, useState } from "react";
import { mosaicAugment, type Box } from "@/labs/object-detection/sim/math";
import { generateObjects } from "@/labs/object-detection/sim/mock";
import { BoxCanvas } from "./canvas";
import { Seeded, Sim, Segmented } from "./shared";

const W = 220;
const H = 130;

export function MosaicSim() {
  const [seed, setSeed] = useState(77);
  const [view, setView] = useState<"before" | "after">("after");

  const quadrants = useMemo(() => {
    const qs: Box[][] = [];
    for (let qi = 0; qi < 4; qi++) {
      const { objects } = generateObjects({ seed: seed + qi * 13, numObjects: 4 + (qi % 3), objectSize: 0.55 });
      qs.push(objects.map((o) => o.box));
    }
    return qs;
  }, [seed]);

  const result = useMemo(() => mosaicAugment(quadrants, W, H), [quadrants]);

  const avgPerQuadrant = quadrants.slice(0, 4).reduce((s, q) => s + q.length, 0) / quadrants.length;
  const smallBefore = quadrants.flat()
    .filter((b) => (b.x2 - b.x1) < 8)
    .length;

  return (
    <Sim
      name="Mosaic augmentation"
      identifies={`4 quadrants → ${result.total} objects on one canvas; small boxes go from ${smallBefore} to ${result.smallCount}; instance density ×${result.densityRatio.toFixed(1)}.`}
      math={[
        "canvas = stitch(4 crops): object_j → quadrant_q, box/2 + offset_q",
        "each sample now mixes 4 scenes → N× decoupled object density",
        "small objects (w < 8px on the joint canvas) grow in share",
      ]}
      need="Dense detectors can under-train on small objects and sparse scenes. Mosaic stitches four images (with their labels) so one training sample carries the box-density of four — cheaply raising small-object coverage and regularization without new data."
      real="The corpus documents data recipes — bag of freebies — built on richer augmentation (mosaic-style combination, mixup, multi-scale training) to push real-time accuracy at zero inference cost."
      papers={["P016", "P042", "P047"]}
      params={
        <>
          <Seeded seed={seed} onReseed={() => setSeed(Math.floor(Math.random() * 1000))} />
          <Segmented
            label="view"
            value={view}
            options={[
              { value: "after", label: "mosaicked" },
              { value: "before", label: "4 source crops" },
            ]}
            onChange={setView}
          />
        </>
      }
    >
      {view === "after" ? (
        <BoxCanvas
          boxes={result.boxes.map((b) => ({
            box: b.box,
            color: ["#34d399", "#60a5fa", "#c084fc", "#fbbf24"][b.q],
          }))}
        />
      ) : (
        <div className="grid grid-cols-2 gap-1">
          {quadrants.slice(0, 4).map((qs, q) => (
            <BoxCanvas
              key={q}
              boxes={qs.map((b) => ({ box: b, color: ["#34d399", "#60a5fa", "#c084fc", "#fbbf24"][q] }))}
            />
          ))}
        </div>
      )}
      <div className="mt-3 grid grid-cols-3 gap-3">
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-2 text-center">
          <div className="font-mono text-[9px] uppercase text-zinc-500">instances on canvas</div>
          <div className="font-mono text-lg text-zinc-200">{result.total}</div>
          <div className="font-mono text-[10px] text-zinc-600">avg/quadrant {avgPerQuadrant.toFixed(0)}</div>
        </div>
        <div className="rounded-lg border border-sky-800/50 bg-sky-950/20 p-2 text-center">
          <div className="font-mono text-[9px] uppercase text-sky-400">{"small boxes (w<8px)"}</div>
          <div className="font-mono text-lg text-sky-300">{smallBefore} → {result.smallCount}</div>
        </div>
        <div className="rounded-lg border border-violet-800/50 bg-violet-950/20 p-2 text-center">
          <div className="font-mono text-[9px] uppercase text-violet-400">instance density</div>
          <div className="font-mono text-lg text-violet-300">×{result.densityRatio.toFixed(1)}</div>
        </div>
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Halving each quadrant halves its boxes — that is exactly where the small-object
        coverage comes from. Mosaic is a training-time label-preserving transform: the loss
        still uses the (re-mapped) original boxes, so accuracy is gained without any
        inference-time cost.
      </p>
    </Sim>
  );
}