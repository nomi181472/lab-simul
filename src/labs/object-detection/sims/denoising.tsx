"use client";

import { useMemo, useState } from "react";
import { denoisingEffect, type Box } from "@/labs/object-detection/sim/math";
import { BoxCanvas } from "./canvas";
import { Seeded, Sim, Slider, Segmented } from "./shared";

const GTS: Box[] = [
  { x1: 40, y1: 30, x2: 96, y2: 84 },
  { x1: 120, y1: 50, x2: 176, y2: 106 },
  { x1: 74, y1: 66, x2: 120, y2: 112 },
];

export function DenoisingSim() {
  const [seed, setSeed] = useState(33);
  const [noise, setNoise] = useState(0.35);
  const [view, setView] = useState<"gt" | "queries">("queries");

  const result = useMemo(() => denoisingEffect(GTS, noise, seed), [seed, noise]);

  return (
    <Sim
      name="Query denoising (DN-DETR)"
      identifies={`Corrupted-GT queries match at ${result.reductionPct.toFixed(0)}% lower cost than random queries — the matcher gets cheap positives early.`}
      math={[
        "decode queries q ~ N(gt_corrupt, σ·gt);  σ = the corruption fraction",
        "matching cost = Σ 1 − IoU(query, gt)     (Hungarian, same as the set loss)",
        "denoising group trains the decoder & matcher without ever matching random noise",
      ]}
      need="One-to-one Hungarian matching is sparse and slow to start: early in training, random queries rarely overlap GT, so the assignment is near-random and gradients are weak. Feeding corrupted ground-truth boxes as extra ‘denoising’ queries gives the decoder a ladder of easy positives, so convergence speeds up and the one-to-one head actually learns."
      real="The corpus records how the transformer line mitigates slow convergence (self-supervised pre-training plus denoising-style objectives); DN-DETR-style denoising is the standard accelerator for DETR-family real-time detectors."
      papers={["P038", "P037"]}
      params={
        <>
          <Seeded seed={seed} onReseed={() => setSeed(Math.floor(Math.random() * 1000))} />
          <Slider label="corruption noise σ" value={noise} min={0.05} max={0.9} step={0.05} onChange={setNoise} />
          <Segmented
            label="view"
            value={view}
            options={[
              { value: "gt", label: "GT only" },
              { value: "queries", label: "± corrupted queries" },
            ]}
            onChange={setView}
          />
        </>
      }
    >
      <BoxCanvas
        boxes={[
          ...GTS.map((g, i) => ({ box: g, color: "#fbbf24", label: `gt${i}` })),
          ...(view === "queries"
            ? [
                ...GTS.map((g, i) => ({
                  box: {
                    x1: g.x1 - 10,
                    y1: g.y1 - 6,
                    x2: g.x2 + 12,
                    y2: g.y2 + 8,
                  },
                  color: "#f87171",
                  label: `q${i}−`,
                  dashed: true,
                })),
                ...GTS.map((g, i) => ({
                  box: {
                    x1: g.x1 + 12,
                    y1: g.y1 + 8,
                    x2: g.x2 + 24,
                    y2: g.y2 + 16,
                  },
                  color: "#f87171",
                  label: `q${i}+`,
                  dashed: true,
                })),
              ]
            : []),
        ]}
      />
      <div className="mt-3 grid grid-cols-3 gap-3">
        <div className="rounded-lg border border-rose-800/50 bg-rose-950/20 p-2 text-center">
          <div className="font-mono text-[9px] uppercase text-rose-400">random queries</div>
          <div className="font-mono text-lg text-rose-300">{result.baselineCost.toFixed(2)}</div>
          <div className="font-mono text-[10px] text-zinc-500">matching cost</div>
        </div>
        <div className="rounded-lg border border-emerald-800/50 bg-emerald-950/20 p-2 text-center">
          <div className="font-mono text-[9px] uppercase text-emerald-400">denoised queries</div>
          <div className="font-mono text-lg text-emerald-300">{result.denoisedCost.toFixed(2)}</div>
          <div className="font-mono text-[10px] text-zinc-500">matching cost</div>
        </div>
        <div className="rounded-lg border border-sky-800/50 bg-sky-950/20 p-2 text-center">
          <div className="font-mono text-[9px] uppercase text-sky-400">cost reduction</div>
          <div className="font-mono text-lg text-sky-300">{result.reductionPct.toFixed(0)}%</div>
          <div className="font-mono text-[10px] text-zinc-500">matched IoU {result.matchedIou.toFixed(2)}</div>
        </div>
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Raising σ moves the corrupted queries away from GT and the denoised advantage toward
        the random baseline. Small σ = the “easy ladder” DN-DETR exploits: the decoder learns
        to average the two perturbations back onto GT before generalizing to real queries.
      </p>
    </Sim>
  );
}