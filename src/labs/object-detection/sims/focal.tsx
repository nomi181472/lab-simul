"use client";

import { useMemo, useState } from "react";
import { focalLossStats } from "@/labs/object-detection/sim/math";
import { generateClassificationSamples } from "@/labs/object-detection/sim/mock";
import { Slider, Segmented, Seeded, Sim } from "./shared";

export function FocalSim() {
  const [seed, setSeed] = useState(7);
  const [fg, setFg] = useState(10);
  const [bg, setBg] = useState(900);
  const [easy, setEasy] = useState(0.9);
  const [gamma, setGamma] = useState(2);
  const [loss, setLoss] = useState<"ce" | "focal">("focal");

  const samples = useMemo(() => generateClassificationSamples(seed, fg, bg, easy), [seed, fg, bg, easy]);
  const stats = useMemo(() => focalLossStats(samples, loss === "focal" ? gamma : 0, 0.25), [samples, gamma, loss]);

  const total = loss === "focal" ? stats.fl : stats.ce;
  const fgShare = ((loss === "focal" ? stats.fgFL : stats.fgCE) / Math.max(1e-9, total)) * 100;
  const bgShare = ((loss === "focal" ? stats.bgFL : stats.bgCE) / Math.max(1e-9, total)) * 100;

  return (
    <Sim
      name="Focal loss and class imbalance"
      identifies={`${fg} foreground vs ${bg} background candidates (1:${(bg / Math.max(1, fg)).toFixed(0)} imbalance). With ${loss}${loss === "focal" ? `, γ=${gamma}` : ""}, background owns ${bgShare.toFixed(1)}% of the loss.`}
      math={[
        "CE = −log(pᵗ)",
        "FL = −α (1−pᵗ)^γ log(pᵗ)",
        "easy negatives → pᵗ ≈ 1 → (1−pᵗ)^γ ≈ 0 → their loss collapses",
      ]}
      need="In dense detection, ~1000 background anchors dominate every foreground sample: plain CE lets confident backgrounds drown out rare hard objects. γ down-weights well-classified samples; α balances rare foreground."
      real="Introduced for one-stage detectors to rival two-stage accuracy; the same imbalance logic recurs in ASE-focal (ASSD), quality focal loss (PP-YOLOE), and label-assignment weighting in the YOLO line."
      papers={["P015", "P017", "P024", "P021"]}
      cites={[
        {
          text: "Focal loss is the class-imbalance response of the dense one-stage strand of the corpus.",
          papers: ["P015", "P017"],
        },
        {
          text: "Focal-style re-weighting recurs in later heads of the real-time line.",
          papers: ["P024", "P021"],
        },
      ]}
      params={
        <>
          <Seeded seed={seed} onReseed={() => setSeed(Math.floor(Math.random() * 1000))} />
          <Slider label="foreground samples" value={fg} min={2} max={80} step={1} onChange={setFg} />
          <Slider label="background samples" value={bg} min={50} max={2000} step={50} onChange={setBg} />
          <Slider label="easy fraction" value={easy} min={0.6} max={0.95} step={0.01} onChange={setEasy} />
          <Slider label="γ (gamma)" value={gamma} min={0} max={5} step={0.1} onChange={setGamma} />
          <Segmented
            label="loss"
            value={loss}
            onChange={setLoss}
            options={[
              { value: "ce", label: "CE" },
              { value: "focal", label: "Focal" },
            ]}
          />
        </>
      }
    >
      <div className="grid gap-3 sm:grid-cols-3">
        <div className="rounded-lg border border-zinc-800 p-3 text-center">
          <div className="font-mono text-[9px] uppercase text-zinc-500">total loss</div>
          <div className="mt-1 font-mono text-lg text-amber-200">{total.toFixed(2)}</div>
        </div>
        <div className="rounded-lg border border-zinc-800 p-3 text-center">
          <div className="font-mono text-[9px] uppercase text-emerald-400">foreground share</div>
          <div className="mt-1 font-mono text-lg text-emerald-300">{fgShare.toFixed(1)}%</div>
        </div>
        <div className="rounded-lg border border-zinc-800 p-3 text-center">
          <div className="font-mono text-[9px] uppercase text-rose-400">background share</div>
          <div className="mt-1 font-mono text-lg text-rose-300">{bgShare.toFixed(1)}%</div>
        </div>
      </div>

      <div className="mt-4">
        <div className="mb-2 font-mono text-[9px] uppercase text-zinc-500">
          contribution to gradient signal (loss share)
        </div>
        <div className="flex h-6 w-full overflow-hidden rounded-md border border-zinc-800">
          <div className="bg-emerald-500/80" style={{ width: `${fgShare}%` }} />
          <div className="bg-rose-500/60" style={{ width: `${bgShare}%` }} />
        </div>
        <div className="mt-1 flex justify-between font-mono text-[9px] text-zinc-600">
          <span>fg {fgShare.toFixed(1)}%</span>
          <span>bg {bgShare.toFixed(1)}%</span>
        </div>
      </div>

      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Flip between CE and Focal at γ=2 at a 90:1 imbalance: the background&apos;s slice of the
        loss collapses while hard foreground stays audible. Raise γ and the easy-negative noise
        vanishes — but push γ too high and even informative hard backgrounds fade.
      </p>
    </Sim>
  );
}