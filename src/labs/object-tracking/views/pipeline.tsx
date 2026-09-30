"use client";

import { PAPERS } from "@/labs/object-tracking/data/papers";
import { Card, Badge, Formula } from "@/labs/object-tracking/ui";

/* How a tracker actually works: MOT loop vs SOT loop, with live corpus examples. */

function Flow({ steps, accent }: { steps: string[]; accent: string }) {
  return (
    <div className="flex flex-col items-stretch gap-1.5">
      {steps.map((s, i) => (
        <div key={s} className="flex flex-col items-center gap-1.5">
          <div className={`w-full rounded-lg border px-3 py-2 text-center font-mono text-[11px] ${accent}`}>{s}</div>
          {i < steps.length - 1 && <span className="font-mono text-xs text-zinc-600">↓</span>}
        </div>
      ))}
    </div>
  );
}

export function PipelineView() {
  const mot = PAPERS.filter((p) => p.task === "multi-object");
  const sot = PAPERS.filter((p) => p.task === "single-object");
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Tracking pipelines</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          Two loops cover nearly the whole corpus. Multi-object tracking is{" "}
          <span className="text-sky-300">detect → predict → associate → manage</span>; single-object
          tracking is <span className="text-emerald-300">template → search → score → update</span>.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card tone="accent">
          <div className="font-mono text-[11px] uppercase tracking-widest text-sky-300">MOT · tracking-by-detection ({mot.length} papers)</div>
          <div className="mt-3">
            <Flow accent="border-sky-800/60 bg-sky-950/30 text-sky-200" steps={["Input: frame t + detections", "Predict: Kalman / learned motion → expected boxes", "Cost: C(i,j) = λ·motion + λ·appearance + λ·(1−IoU)", "Associate: Hungarian / greedy under gate", "Update: correct matched tracks; predict unmatched", "Manage: birth unmatched · kill after max_age", "Output: boxes + persistent IDs"]} />
          </div>
          <Formula>C(i,j) = λ₁·motion(i,j) + λ₂·appearance(i,j) + λ₃·(1 − IoU(âᵢ, zⱼ))</Formula>
          <p className="mt-2 text-[12px] text-zinc-400">
            Canonical instance: <span className="font-mono text-sky-300">T005 SORT</span> (Kalman + IoU +
            Hungarian, 260 Hz). DeepSORT (T013) adds the appearance term; ByteTrack (T061) splits
            association into high/low-score stages.
          </p>
        </Card>
        <Card tone="ok">
          <div className="font-mono text-[11px] uppercase tracking-widest text-emerald-300">SOT · template matching ({sot.length} papers)</div>
          <div className="mt-3">
            <Flow accent="border-emerald-800/60 bg-emerald-950/30 text-emerald-200" steps={["Input: first-frame template box", "Crop: search region around last position", "Embed: φ(template), φ(search) — shared network", "Score: cross-correlation similarity map", "Locate: argmax (+ box regression head)", "Update: template / memory update rule", "Output: one box per frame"]} />
          </div>
          <Formula>score(u) = ⟨ φ(template), φ(search patch at u) ⟩</Formula>
          <p className="mt-2 text-[12px] text-zinc-400">
            Canonical instance: <span className="font-mono text-emerald-300">T009 SiamFC</span>
            (fully-convolutional Siamese, 58 FPS). SiamRPN++ (T028) adds the RPN head; ATOM/DiMP
            (T026/T031) replace correlation with learned model prediction.
          </p>
        </Card>
      </div>

      <Card>
        <div className="font-mono text-[11px] uppercase tracking-widest text-zinc-500">Where detection plugs in</div>
        <p className="mt-2 text-[12px] leading-5 text-zinc-400">
          The MOT loop&apos;s first box is a detector output. SORT&apos;s headline result is a detector swap
          (ACF → Faster R-CNN: 15.1 → 34.0 MOTA) — study detectors in{" "}
          <a href="#object-detection" className="text-emerald-300 underline underline-offset-2">Object Detection</a>,
          study everything after them here. End-to-end transformers (TransTrack T053, MOTR T059)
          later dissolve the boundary by predicting tracks directly from pixels.
        </p>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {["T005", "T013", "T034", "T043", "T053", "T059", "T061", "T067", "T073"].map((id) => (
            <Badge key={id} tone="sky">{id}</Badge>
          ))}
          <span className="font-mono text-[10px] text-zinc-600">key MOT pipeline papers (structured → more as batches land)</span>
        </div>
      </Card>
    </div>
  );
}
