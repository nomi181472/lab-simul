"use client";

import { useMemo, useState } from "react";
import { deformableAttention } from "@/labs/object-detection/sim/math";
import { Sim, Slider } from "./shared";

const GRID = 6;

export function DeformableAttnSim() {
  const [K, setK] = useState(4);
  const [focusX, setFocusX] = useState(0.6);
  const [focusY, setFocusY] = useState(0.5);

  const { points, q } = useMemo(() => {
    const pts = [];
    for (let y = 0; y < GRID; y++) {
      for (let x = 0; x < GRID; x++) {
        pts.push({ p: [x / (GRID - 1), y / (GRID - 1)], label: `${x},${y}` });
      }
    }
    const q = [focusX, focusY];
    return { points: pts, q };
  }, [focusX, focusY]);

  const { sampled, weights, costDense, costSparse, focus } = useMemo(
    () => deformableAttention(q, points, K),
    [q, points, K],
  );

  const span = costDense / Math.max(1, costSparse);
  const weightOf = (i: number) => {
    const idx = sampled.indexOf(i);
    return idx >= 0 ? weights[idx] : 0;
  };

  return (
    <Sim
      name="Multi-scale deformable attention"
      identifies={`k=${K} of ${costDense} cells suffice: attention FLOP cost ${span.toFixed(0)}× lower than dense, concentration ${(focus * 100).toFixed(0)}%.`}
      math={[
        "dense:  y[q] = Σ_{all cells} softmax(q·k)·v      (cost = N)",
        "deform: choose k ≤ N offset positions around a reference point",
        "y[q] = Σ_{s∈sample} softmax(q·k_s)·v_s            (cost = k ≪ N)",
      ]}
      need="Dense self-attention is quadratic in the feature map, which is unaffordable for high-res detection. Deformable attention samples a small learned set of points per query across pyramid levels, keeping long-range context at (near-)constant cost — the mechanism RT-DETR/D-FINE use to fuse multi-scale features in real time."
      real="The corpus traces transformer detectors using sampled/area attention to cut the quadratic cost, and the real-time DETR line makes that the backbone of its hybrid encoder."
      papers={["P029", "P037", "P038", "P041"]}
      params={
        <>
          <Slider label="sampled points k" value={K} min={1} max={16} step={1} onChange={setK} fmt={(v) => String(v)} />
          <Slider label="query x (focus column)" value={focusX} min={0} max={1} step={0.05} onChange={setFocusX} />
          <Slider label="query y (focus row)" value={focusY} min={0} max={1} step={0.05} onChange={setFocusY} />
        </>
      }
    >
      <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-3">
        <div className="mb-2 font-mono text-[9px] uppercase tracking-wide text-zinc-500">
          {GRID}×{GRID} feature cells — query focus · ({focusX.toFixed(2)}:{focusY.toFixed(2)})
        </div>
        <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${GRID}, 1fr)` }}>
          {points.map((pt, i) => {
            const w = weightOf(i);
            const on = sampled.includes(i);
            return (
              <div
                key={pt.label}
                className="relative flex h-10 items-center justify-center rounded border border-zinc-800 font-mono text-[9px]"
                style={{
                  background: on ? `rgba(34,211,238,${0.18 + w * 0.6})` : "rgba(24,24,27,0.6)",
                  borderColor: on ? "#22d3ee" : "#3f3f46",
                }}
                title={`cell ${pt.label} ${on ? `weight ${(w * 100).toFixed(0)}%` : "unsampled"}`}
              >
                {on ? (w * 100).toFixed(0) : ""}
              </div>
            );
          })}
        </div>
        <div className="mt-2 flex flex-wrap justify-between font-mono text-[10px] text-zinc-500">
          <span>dense cost (FLOPs) {costDense}</span>
          <span>deformable cost {costSparse}</span>
          <span>speedup {span.toFixed(1)}×</span>
        </div>
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Cyan cells are the k sampled positions the query actually attends to; brighter cyan
        is more weight. Only k cells are weighted, so FLOPs stay ~k regardless of image size —
        “sparse but global” is the deformable-attention contract.
      </p>
    </Sim>
  );
}