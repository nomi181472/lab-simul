"use client";

import { useMemo, useState } from "react";
import { PROBLEM_LIFECYCLES, PROBLEM_BY_ID, buildProblemMatrix } from "@/labs/object-detection/data/problems";
import { PAPER_BY_ID } from "@/labs/object-detection/data/papers";
import { Card, Badge, SectionTitle, StatusBadge } from "@/labs/object-detection/ui";

const STATUS_COLOR: Record<string, string> = {
  resolved: "bg-emerald-500",
  reduced: "bg-sky-500",
  persistent: "bg-amber-500",
  transformed: "bg-violet-500",
  uncertain: "bg-rose-500",
};

export function ProblemsView() {
  const matrix = useMemo(() => buildProblemMatrix(), []);
  const [selId, setSelId] = useState<string>(matrix[0]?.id ?? "");

  const years = Array.from({ length: 2026 - 2015 + 1 }, (_, i) => 2015 + i);
  const selected = PROBLEM_BY_ID[selId];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Problem matrix</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          12 problem lifecycles traced across 2015–2026. Each cell is a year where a
          corpus paper explicitly evidences the problem. Click a row to open the lifecycle.
        </p>
      </div>

      {/* matrix */}
      <Card>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-[11px]">
            <thead>
              <tr>
                <th className="sticky left-0 bg-zinc-900 p-1.5 text-left font-mono text-[10px] uppercase text-zinc-500">
                  problem
                </th>
                {years.map((y) => (
                  <th key={y} className="p-1 text-center font-mono text-[9px] text-zinc-600">
                    {String(y).slice(2)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row) => (
                <tr
                  key={row.id}
                  onClick={() => setSelId(row.id)}
                  className={`cursor-pointer border-t border-zinc-900 ${selId === row.id ? "bg-emerald-950/30" : "hover:bg-zinc-900/50"}`}
                >
                  <td className="sticky left-0 bg-zinc-900 p-1.5">
                    <div className="flex items-center gap-1.5">
                      <span className={`h-2 w-2 rounded-full ${STATUS_COLOR[row.status]}`} />
                      <span className="font-mono text-zinc-400">{row.label}</span>
                    </div>
                  </td>
                  {years.map((y) => (
                    <td key={y} className="p-0 text-center">
                      {row.present[y] ? (
                        <span className="inline-flex h-3.5 w-3.5 items-center justify-center">
                          <span className="h-2 w-2 rounded-sm bg-emerald-500/70" />
                        </span>
                      ) : (
                        <span className="text-zinc-800">·</span>
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* lifecycle explorer */}
      {selected && (
        <Card tone="accent">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <SectionTitle>{selected.problem}</SectionTitle>
            <div className="flex items-center gap-2">
              <StatusBadge status={selected.currentStatus} />
              <Badge tone="zinc">first observed {selected.firstObservedYear}</Badge>
            </div>
          </div>

          <div className="mt-4 space-y-4">
            {selected.occurrences
              .slice()
              .sort((a, b) => a.year - b.year)
              .map((o) => (
                <div key={o.year} className="border-l-2 border-zinc-800 pl-4">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-mono text-sm text-emerald-300">{o.year}</span>
                    {o.papers.map((pid) => {
                      const p = PAPER_BY_ID[pid];
                      return p ? (
                        <span key={pid} className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-300">
                          {p.id} {p.shortTitle}
                        </span>
                      ) : (
                        <span key={pid} className="rounded bg-rose-900/50 px-1.5 py-0.5 text-[10px] text-rose-300">
                          ?{pid}
                        </span>
                      );
                    })}
                  </div>
                  {o.attemptedSolutions.length > 0 && (
                    <ul className="mt-2 space-y-1">
                      {o.attemptedSolutions.map((s, i) => (
                        <li key={i} className="flex gap-2 text-[12px] leading-5 text-zinc-400">
                          <span className="text-emerald-600">↳</span>
                          {s}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              ))}
          </div>

          <div className="mt-4 flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/50 p-3">
            <Badge tone="emerald">status evidence</Badge>
            <p className="text-[11px] leading-5 text-zinc-400">{selected.statusEvidence}</p>
          </div>
        </Card>
      )}

      {/* problem selector chips */}
      <div className="flex flex-wrap gap-1.5">
        {PROBLEM_LIFECYCLES.map((p) => (
          <button
            key={p.id}
            onClick={() => setSelId(p.id)}
            className={`rounded-md border px-2 py-1 font-mono text-[10px] ${
              selId === p.id
                ? "border-emerald-600 bg-emerald-600/20 text-emerald-200"
                : "border-zinc-800 text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {p.shortLabel}
          </button>
        ))}
      </div>
    </div>
  );
}