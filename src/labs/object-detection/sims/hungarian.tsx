"use client";

import { useMemo, useState } from "react";
import { hungarianMinCost } from "@/labs/object-detection/sim/math";
import { generateCostMatrix } from "@/labs/object-detection/sim/mock";
import { Seeded, Sim } from "./shared";

export function HungarianSim() {
  const [seed, setSeed] = useState(5);
  const [nPred, setNPred] = useState(4);
  const [nGt, setNGt] = useState(3);

  const cost = useMemo(() => generateCostMatrix(seed, nPred, nGt), [seed, nPred, nGt]);
  const { assignment } = useMemo(() => hungarianMinCost(cost), [cost]);

  // greedy baseline (always attach a prediction to its nearest GT)
  const greedy = useMemo(() => {
    const used = new Set<number>();
    const a: number[] = cost.map(() => -1);
    for (let c = 0; c < nGt; c++) {
      let bestR = -1;
      let bestV = Infinity;
      for (let r = 0; r < nPred; r++) {
        if (used.has(r)) continue;
        if (cost[r][c] < bestV) {
          bestV = cost[r][c];
          bestR = r;
        }
      }
      if (bestR >= 0) {
        used.add(bestR);
        a[bestR] = c;
      }
    }
    return a;
  }, [cost, nPred, nGt]);

  const costOf = (a: number[]) =>
    a.reduce((s, c, r) => (c >= 0 ? s + cost[r][c] : s), 0);

  return (
    <Sim
      name="Hungarian (optimal) label assignment"
      identifies={`${nPred} predictions vs ${nGt} ground-truth boxes. Optimal matching cost ${costOf(assignment).toFixed(2)} vs greedy ${costOf(greedy).toFixed(2)}.`}
      math={[
        "cost(i,j) = λ_cls · Lcls(predᵢ, gtⱼ) + λ_box · Lbox(predᵢ, gtⱼ)",
        "assignment = argmin_π Σᵢ cost(i, π(i))   (bipartite, O(n³))",
        "then supervision is applied only to matched pairs (one-to-one)",
      ]}
      need="Before DETR, every anchor was supervised against the closest GT (many-to-one, ambiguous). Hungarian matches each GT to exactly one prediction — the minimal total cost split — producing a clean one-to-one signal that makes NMS unnecessary."
      real="DETR-style set loss uses Hungarian matching with a per-query loss; RT-DETR and D-FINE inherit it; YOLOv10/YOLO26 keep one-to-one heads but pair them with many-to-one heads for denser supervision."
      papers={["P029", "P037", "P033", "P046", "P038"]}
      params={
        <>
          <Seeded seed={seed} onReseed={() => setSeed(Math.floor(Math.random() * 1000))} />
          <div className="flex gap-2">
            <label className="font-mono text-[10px] text-zinc-500">
              preds
              <input
                type="number"
                min={2}
                max={8}
                value={nPred}
                onChange={(e) => setNPred(Math.max(2, Math.min(8, Number(e.target.value) || 4)))}
                className="ml-2 w-14 rounded border border-zinc-800 bg-zinc-900 px-1 py-0.5 text-zinc-200"
              />
            </label>
            <label className="font-mono text-[10px] text-zinc-500">
              gts
              <input
                type="number"
                min={2}
                max={8}
                value={nGt}
                onChange={(e) => setNGt(Math.max(2, Math.min(8, Number(e.target.value) || 3)))}
                className="ml-2 w-14 rounded border border-zinc-800 bg-zinc-900 px-1 py-0.5 text-zinc-200"
              />
            </label>
          </div>
        </>
      }
    >
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-center">
          <thead>
            <tr>
              <th className="p-1 font-mono text-[9px] text-zinc-600">pred \ gt</th>
              {Array.from({ length: nGt }, (_, c) => (
                <th key={c} className="p-1 font-mono text-[10px] text-amber-300">
                  gt{c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {cost.map((row, r) => (
              <tr key={r}>
                <td className="p-1 font-mono text-[10px] text-sky-300">pred{r}</td>
                {row.map((v, c) => {
                  const matched = assignment[r] === c;
                  const greed = greedy[r] === c;
                  return (
                    <td
                      key={c}
                      className={`p-1 font-mono text-[11px] ${
                        matched && greed
                          ? "rounded bg-emerald-600/25 text-emerald-200 ring-1 ring-emerald-500"
                          : matched
                            ? "rounded bg-emerald-600/25 text-emerald-200 ring-1 ring-emerald-500"
                            : greed
                              ? "bg-amber-600/15 text-amber-300"
                              : "text-zinc-500"
                      }`}
                    >
                      {v.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3">
        <div className="rounded-lg border border-emerald-800/50 bg-emerald-950/20 p-2 text-center">
          <div className="font-mono text-[9px] uppercase text-emerald-400">hungarian total</div>
          <div className="font-mono text-lg text-emerald-300">{costOf(assignment).toFixed(2)}</div>
        </div>
        <div className="rounded-lg border border-amber-800/50 bg-amber-950/20 p-2 text-center">
          <div className="font-mono text-[9px] uppercase text-amber-400">greedy total</div>
          <div className="font-mono text-lg text-amber-300">{costOf(greedy).toFixed(2)}</div>
        </div>
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Greedy takes each GT’s nearest prediction one at a time and can collide; Hungarian
        resolves the whole board at once. Cells: amber = greedy pick, green = optimal pick.
        Whenever the totals differ, the suboptimal one-to-one supervision has begun.
      </p>
    </Sim>
  );
}