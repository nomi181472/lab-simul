"use client";

import { useState } from "react";
import {
  generateCrowdedScene,
  generateOcclusionScene,
  generateSmallObjectScene,
} from "@/labs/object-detection/sim/mock";
import { Seeded, Segmented, Sim } from "./shared";
import { BoxCanvas, VIEW_W, VIEW_H } from "./canvas";

export function EdgeScenarioSim() {
  const [seed, setSeed] = useState(17);
  const [kind, setKind] = useState<"small" | "occluded" | "crowded">("small");
  const [n, setN] = useState(6);

  const scene =
    kind === "small"
      ? generateSmallObjectScene({ seed, numObjects: n })
      : kind === "occluded"
        ? generateOcclusionScene({ seed, numObjects: Math.min(5, n) })
        : generateCrowdedScene({ seed, numObjects: n, objectSize: 0.7 });

  return (
    <Sim
      name="Edge scenario scenes"
      identifies={`${scene.objects.length} objects in a ${kind} scene. Difficulty layer: ${scene.difficulty[0]}.`}
      math={[
        "annotation -> visible footprint `visible` ∈ [0,1] (occlusion)",
        "detector score is capped by how many features remain after occlusion/scale/strided pooling",
        "scene generators are seeded (same seed -> same scene)",
      ]}
      need="Benchmarks ship regular grids; real failure happens on tiny, buried and overlapping objects. These scenes compress the three hardest categories into a canvas you can regenerate deterministically and reason about."
      real="Small-object loss (SSD→D-FINE STAL), occlusion robustness (OC-SORT / Tracking-the-Unseen), crowded-scene NMS collapse (DIoU/Matrix NMS) — each label above maps to a corpus thread."
      papers={["P001", "P016", "P042", "P047", "P048"]}
      params={
        <>
          <Seeded seed={seed} onReseed={() => setSeed(Math.floor(Math.random() * 1000))} />
          <Segmented
            label="scene"
            value={kind}
            onChange={setKind}
            options={[
              { value: "small", label: "small objects" },
              { value: "occluded", label: "occlusion" },
              { value: "crowded", label: "crowded" },
            ]}
          />
          <label className="font-mono text-[10px] text-zinc-500">
            objects
            <input
              type="number"
              min={2}
              max={12}
              value={n}
              onChange={(e) => setN(Math.max(2, Math.min(12, Number(e.target.value) || 6)))}
              className="ml-2 w-14 rounded border border-zinc-800 bg-zinc-900 px-1 py-0.5 text-zinc-200"
            />
          </label>
        </>
      }
    >
      <BoxCanvas
        boxes={scene.objects.map((o) => ({
          box: o.box,
          color: o.visible < 0.4 ? "#fbbf24" : o.confidence > 0.5 ? "#34d399" : "#38bdf8",
          label: o.visible < 1 ? `${Math.round(o.visible * 100)}%` : undefined,
          dashed: o.visible < 0.6,
        }))}
      />
      <div className="mt-3 space-y-1.5">
        {scene.difficulty.map((d, i) => (
          <p key={i} className="flex gap-2 text-[11px] leading-5 text-zinc-500">
            <span className="text-emerald-600">↳</span>
            {d}
          </p>
        ))}
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Sizes are relative to a 220×130 canvas; the real task is to pick the detector response
        to “this object has {kind === "small" ? "≤ a handful of pixels" : kind === "occluded" ? "only part of its footprint visible" : "neighbours on top of it"}”.
      </p>
      <div className="mt-2 font-mono text-[9px] text-zinc-700">
        canvas {VIEW_W}×{VIEW_H} · seeded · regenerating is reproducible
      </div>
    </Sim>
  );
}