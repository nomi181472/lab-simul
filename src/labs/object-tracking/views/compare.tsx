"use client";

import { useMemo, useState } from "react";
import { PAPERS } from "@/labs/object-tracking/data/papers";
import { Card } from "@/labs/object-tracking/ui";

/* Technical differences, not scores: motion × appearance × association × data. */

const COLS = ["paper", "year", "task", "motion", "appearance", "association", "re-id", "datasets", "contribution"] as const;

export function CompareView() {
  const [task, setTask] = useState<string | null>(null);
  const rows = useMemo(() => PAPERS.filter((p) => !task || p.task === task).sort((a, b) => a.year - b.year), [task]);
  const tasks = [...new Set(PAPERS.map((p) => p.task))];
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Paper comparison</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          Compare <span className="text-zinc-200">technical choices</span>, not winners: which motion model,
          which appearance model, which association, which data. Scroll horizontally on small screens.
        </p>
        <div className="mt-3 flex flex-wrap gap-1.5">
          <button onClick={() => setTask(null)} className={`rounded-full border px-2.5 py-1 text-[11px] ${task === null ? "border-sky-600 bg-sky-600/20 text-sky-200" : "border-zinc-800 text-zinc-500"}`}>all ({PAPERS.length})</button>
          {tasks.map((t) => (
            <button key={t} onClick={() => setTask(task === t ? null : t)} className={`rounded-full border px-2.5 py-1 text-[11px] ${task === t ? "border-sky-600 bg-sky-600/20 text-sky-200" : "border-zinc-800 text-zinc-500"}`}>{t}</button>
          ))}
        </div>
      </div>
      <Card className="!p-0 overflow-x-auto">
        <table className="w-full min-w-[1100px] border-collapse text-left text-[11px] leading-4">
          <thead>
            <tr className="border-b border-zinc-800 font-mono text-[10px] uppercase text-zinc-500">
              {COLS.map((c) => <th key={c} className="px-3 py-2">{c}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((p) => (
              <tr key={p.id} className="border-b border-zinc-900 align-top hover:bg-sky-950/10">
                <td className="px-3 py-2"><span className="font-mono text-sky-300">{p.id}</span><div className="font-semibold text-zinc-200">{p.shortTitle}</div></td>
                <td className="px-3 py-2 font-mono text-zinc-400">{p.year}</td>
                <td className="px-3 py-2 font-mono text-violet-300">{p.task}</td>
                <td className="max-w-[180px] px-3 py-2 text-zinc-400">{p.method.motionModel ?? "—"}</td>
                <td className="max-w-[180px] px-3 py-2 text-zinc-400">{p.method.appearanceModel ?? "—"}</td>
                <td className="max-w-[180px] px-3 py-2 text-zinc-400">{p.method.association ?? "—"}</td>
                <td className="max-w-[140px] px-3 py-2 text-zinc-400">{p.method.reid ?? "—"}</td>
                <td className="px-3 py-2 font-mono text-zinc-500">{p.datasets.join(", ") || "—"}</td>
                <td className="max-w-[240px] px-3 py-2 text-zinc-400">{p.contribution[0]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}
