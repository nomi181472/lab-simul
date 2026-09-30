"use client";

import { useMemo, useState } from "react";
import { PAPERS, successorsOf } from "@/labs/object-tracking/data/papers";
import { Card, Badge } from "@/labs/object-tracking/ui";

/* Research-gap chain: before → contribution → after → who picked it up. */

export function GapsView() {
  const [q, setQ] = useState("");
  const rows = useMemo(() => {
    const needle = q.toLowerCase();
    return PAPERS.filter((p) =>
      !needle ||
      `${p.id} ${p.title} ${p.researchGap} ${p.contribution.join(" ")} ${p.limitations.authorStated.join(" ")}`.toLowerCase().includes(needle),
    ).sort((a, b) => a.year - b.year);
  }, [q]);

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Research gaps</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          The central educational object: for every paper —{" "}
          <span className="text-rose-300">what was missing before</span> →{" "}
          <span className="text-sky-300">what it contributed</span> →{" "}
          <span className="text-amber-300">what remained unsolved</span> →{" "}
          <span className="text-emerald-300">who addressed it next</span>.
        </p>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="filter gaps, e.g. occlusion, long-term, appearance…"
          className="mt-3 w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-sky-600 focus:outline-none"
        />
        <div className="mt-2 font-mono text-[10px] text-zinc-500">{rows.length} gap chains</div>
      </div>

      <div className="space-y-3">
        {rows.map((p) => {
          const succs = successorsOf(p.id);
          return (
            <Card key={p.id}>
              <div className="flex flex-wrap items-center gap-2">
                <span className="rounded bg-sky-950 px-1.5 py-0.5 font-mono text-[10px] text-sky-300">{p.id}</span>
                <Badge tone="zinc">{p.year}</Badge>
                <span className="text-sm font-semibold text-zinc-100">{p.shortTitle}</span>
              </div>
              <div className="mt-3 grid gap-2 md:grid-cols-[1fr_1fr_1fr]">
                <div className="rounded-lg border border-rose-900/50 bg-rose-950/15 p-3 min-w-0">
                  <div className="font-mono text-[10px] uppercase tracking-wide text-rose-300">before · the gap</div>
                  <p className="mt-1 text-[12px] leading-5 text-zinc-300">{p.researchGap}</p>
                  {p.previousWork.slice(0, 2).map((w) => (
                    <p key={w.name} className="mt-1.5 text-[11px] leading-4 text-zinc-500">
                      <span className="text-zinc-400">{w.name}:</span> {w.limitation}
                    </p>
                  ))}
                </div>
                <div className="rounded-lg border border-sky-800/50 bg-sky-950/15 p-3 min-w-0">
                  <div className="font-mono text-[10px] uppercase tracking-wide text-sky-300">paper · what it solved</div>
                  <ul className="mt-1 list-disc space-y-1 pl-4 text-[12px] leading-5 text-zinc-300">
                    {p.contribution.slice(0, 4).map((c, i) => <li key={i}>{c}</li>)}
                  </ul>
                </div>
                <div className="rounded-lg border border-amber-800/50 bg-amber-950/15 p-3 min-w-0">
                  <div className="font-mono text-[10px] uppercase tracking-wide text-amber-300">after · what remained</div>
                  <ul className="mt-1 list-disc space-y-1 pl-4 text-[12px] leading-5 text-zinc-300">
                    {p.limitations.authorStated.slice(0, 3).map((l, i) => <li key={i}>{l}</li>)}
                  </ul>
                  {p.limitations.evident.slice(0, 2).map((l, i) => (
                    <p key={`e${i}`} className="mt-1 text-[11px] italic leading-4 text-zinc-500">(lab synthesis) {l}</p>
                  ))}
                </div>
              </div>
              <div className="mt-2 font-mono text-[11px] text-zinc-500">
                {succs.length > 0 ? (
                  <span>→ picked up by <span className="text-emerald-300">{succs.map((s) => `${s.id} ${s.shortTitle}`).join(" · ")}</span></span>
                ) : (
                  <span className="text-zinc-600">→ open thread: no structured successor yet</span>
                )}
              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
