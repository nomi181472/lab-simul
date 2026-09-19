"use client";

import { useMemo, useState } from "react";
import { topKAssignment } from "@/labs/object-detection/sim/math";
import { generateCostMatrix } from "@/labs/object-detection/sim/mock";
import { Seeded, Sim, Slider } from "./shared";

export function DynamicAssignmentSim() {
  const [seed, setSeed] = useState(11);
  const [k, setK] = useState(2);
  const nPred = 8;
  const nGt = 5;

  const cost = useMemo(() => generateCostMatrix(seed, nPred, nGt), [seed]);
  const { positives, assignment, nPairs, collisions } = useMemo(
    () => topKAssignment(cost, k),
    [cost, k],
  );

  const supervisionDensity = (nPairs / nGt).toFixed(2);

  return (
    <Sim
      name="Dynamic label assignment (SimOTA-style topK)"
      identifies={`k=${k}: each GT pulls its top-${k} predictions into positives → ${nPairs} supervised pairs, ${collisions} GT(s) with a shared prediction.`}
      math={[
        "for each GT j: positives ≡ argmin_k cost(i, j)          (the k cheapest preds)",
        "cost(i,j) = λ_cls · Lcls + λ_box · 1 − IoU(pred_i, gt_j)   (+ center prior)",
        "then backprop flows only through predictions that became positive",
      ]}
      need="Hungarian (previous tab) is one-to-one and clean but sparse — one signal per object. SimOTA-style assignment lets k candidates per GT share the gradient; at k=1 it collapses to one-to-one, at k>1 it gives the dense supervision a one-stage head needs, with k adapted to the scores."
      real="The lab’s corpus records the static-vs-dynamic labeling debate; dynamic topK assignment (SimOTA-style, a cited external reference) is how the modern YOLO line keeps one-to-one inference while training many-to-one."
      papers={["P033", "P038", "P046"]}
      params={
        <>
          <Seeded seed={seed} onReseed={() => setSeed(Math.floor(Math.random() * 1000))} />
          <Slider label="top-k per GT" value={k} min={1} max={6} step={1} onChange={setK} fmt={(v) => String(v)} />
        </>
      }
    >
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-center">
          <thead>
            <tr>
              <th className="p-1 font-mono text-[9px] text-zinc-600">pred \ gt</th>
              {Array.from({ length: nGt }, (_, c) => (
                <th key={c} className="p-1 font-mono text-[10px] text-amber-300">gt{c}</th>
              ))}
              <th className="p-1 font-mono text-[9px] text-zinc-600">assigned</th>
            </tr>
          </thead>
          <tbody>
            {cost.map((row, r) => (
              <tr key={r}>
                <td className="p-1 font-mono text-[10px] text-sky-300">pred{r}</td>
                {row.map((v, c) => (
                  <td
                    key={c}
                    className={`p-1 font-mono text-[11px] ${
                      positives[r][c]
                        ? "rounded bg-emerald-600/25 text-emerald-200 ring-1 ring-emerald-500"
                        : "text-zinc-500"
                    }`}
                  >
                    {v.toFixed(2)}
                  </td>
                ))}
                <td className="p-1 font-mono text-[11px] text-emerald-300">
                  {assignment[r] >= 0 ? `gt${assignment[r]}` : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-3 grid grid-cols-3 gap-3">
        <div className="rounded-lg border border-emerald-800/50 bg-emerald-950/20 p-2 text-center">
          <div className="font-mono text-[9px] uppercase text-emerald-400">supervised pairs</div>
          <div className="font-mono text-lg text-emerald-300">{nPairs}</div>
        </div>
        <div className="rounded-lg border border-sky-800/50 bg-sky-950/20 p-2 text-center">
          <div className="font-mono text-[9px] uppercase text-sky-400">supervision density</div>
          <div className="font-mono text-lg text-sky-300">{supervisionDensity}↑</div>
        </div>
        <div className="rounded-lg border border-amber-800/50 bg-amber-950/20 p-2 text-center">
          <div className="font-mono text-[9px] uppercase text-amber-400">shared-pred GTs</div>
          <div className="font-mono text-lg text-amber-300">{collisions}</div>
        </div>
      </div>

      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Green cells are the positions that learn. Total possible pairs: {nPred}×{nGt} ={" "}
        {nPred * nGt}; the {k}-top rule keeps a {`${((nPairs / (nPred * nGt)) * 100).toFixed(0)}%`} slice
        of the dense grid. Move k to {" "}
        {k + 1 <= 6 ? "see density rise" : "the max"} — notice how few predictions
        the one-to-one limit (k=1) actually supervises.
      </p>
    </Sim>
  );
}