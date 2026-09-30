"use client";

import { PAPERS } from "@/labs/object-tracking/data/papers";
import { MANIFEST } from "@/labs/object-tracking/data/manifest";
import { Card, Badge } from "@/labs/object-tracking/ui";

/* Corpus coverage & provenance: what is structured, what is missing, policy. */

export function AuditView() {
  const done = new Set(PAPERS.map((p) => p.id));
  const missing = MANIFEST.filter((m) => !done.has(m.id));
  const byYear: Record<number, { done: number; total: number }> = {};
  for (const m of MANIFEST) {
    byYear[m.year] ??= { done: 0, total: 0 };
    byYear[m.year].total += 1;
    if (done.has(m.id)) byYear[m.year].done += 1;
  }
  const noRelations = PAPERS.filter((p) => p.relations.length === 0);
  const noEquations = PAPERS.filter((p) => p.equations.length === 0);

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Audit</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          Provenance policy (same as Object Detection): every field grounded in the named PDF;
          results only if printed in the paper; relations only to corpus papers the work itself
          builds on or compares against. Nothing invented.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <Card><div className="font-mono text-3xl text-sky-300">{PAPERS.length}/{MANIFEST.length}</div><div className="text-xs text-zinc-500">records structured</div></Card>
        <Card><div className="font-mono text-3xl text-amber-300">{missing.length}</div><div className="text-xs text-zinc-500">papers awaiting structuring</div></Card>
        <Card><div className="font-mono text-3xl text-emerald-300">{PAPERS.reduce((a, p) => a + p.equations.length, 0)}</div><div className="text-xs text-zinc-500">equations with variable-level explanations</div></Card>
      </div>

      <Card>
        <div className="font-mono text-[11px] uppercase tracking-widest text-zinc-500">coverage by year</div>
        <div className="mt-2 space-y-1.5">
          {Object.entries(byYear).sort((a, b) => Number(a[0]) - Number(b[0])).map(([y, v]) => (
            <div key={y} className="flex items-center gap-2">
              <span className="w-10 font-mono text-[11px] text-zinc-500">{y}</span>
              <div className="h-3 flex-1 overflow-hidden rounded bg-zinc-800">
                <div className="h-full bg-sky-600" style={{ width: `${(v.done / v.total) * 100}%` }} />
              </div>
              <span className="w-20 text-right font-mono text-[10px] text-zinc-400">{v.done}/{v.total}</span>
            </div>
          ))}
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card tone="warn">
          <div className="font-mono text-[11px] uppercase tracking-widest text-amber-300">missing records ({missing.length})</div>
          <div className="mt-2 flex max-h-64 flex-wrap gap-1 overflow-y-auto">
            {missing.map((m) => (
              <span key={m.id} className="rounded border border-zinc-800 px-1.5 py-0.5 font-mono text-[10px] text-zinc-400" title={m.title}>{m.id} · {m.year}</span>
            ))}
          </div>
          <p className="mt-2 text-[11px] text-zinc-500">Next agent: structure these in manifest order (see PROGRESS.md batching).</p>
        </Card>
        <Card>
          <div className="font-mono text-[11px] uppercase tracking-widest text-zinc-500">records needing enrichment</div>
          <p className="mt-2 text-[12px] text-zinc-400">No corpus relations yet (thread roots or unlinked): {noRelations.length ? noRelations.map((p) => p.id).join(", ") : "none"}</p>
          <p className="mt-1 text-[12px] text-zinc-400">No equations (benchmarks/surveys expected): {noEquations.length ? noEquations.map((p) => p.id).join(", ") : "none"}</p>
          <div className="mt-2 flex flex-wrap gap-1">
            <Badge tone="sky">relations must cite corpus ids only</Badge>
            <Badge tone="sky">results need paper-printed numbers</Badge>
          </div>
        </Card>
      </div>
    </div>
  );
}
