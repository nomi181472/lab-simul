"use client";

import { useMemo, useState } from "react";
import { PAPERS } from "@/labs/neuroevolution/data/papers";
import { METRICS } from "@/labs/neuroevolution/data/metrics";
import { Card, SectionTitle, Badge, NeuroPaperLink, Formula, Unstated } from "@/labs/neuroevolution/ui";

/* Metric definitions with their reporting traps. */

export function MetricsView() {
  const [fam, setFam] = useState("all");
  const fams = useMemo<string[]>(() => {
    const set = new Set<string>();
    for (const m of METRICS) set.add(String(m.family));
    return ["all", ...[...set].sort()];
  }, []);
  const list = useMemo(
    () => (fam === "all" ? METRICS : METRICS.filter((m) => m.family === fam)),
    [fam],
  );
  const used = useMemo(() => {
    const m = new Map<string, number>();
    for (const p of PAPERS)
      for (const x of p.metrics) m.set(x, (m.get(x) ?? 0) + 1);
    return m;
  }, []);

  return (
    <div className="flex flex-col gap-5">
      <SectionTitle>Metrics</SectionTitle>
      <div className="flex flex-wrap gap-2">
        {fams.map((f) => (
          <button
            key={f}
            onClick={() => setFam(f)}
            className={`rounded-md border px-2.5 py-1 font-mono text-[10px] ${
              fam === f ? "border-rose-700/60 bg-rose-600/20 text-rose-300" : "border-zinc-800 text-zinc-500"
            }`}
          >
            {f}
          </button>
        ))}
      </div>
      <div className="grid gap-3 lg:grid-cols-2">
        {list.map((m) => (
          <Card key={m.id} className="flex flex-col gap-2">
            <div className="flex items-center justify-between gap-2">
              <span className="text-[13px] font-semibold text-zinc-100">{m.name}</span>
              <Badge tone="violet">{m.family}</Badge>
            </div>
            <Formula>{m.formula}</Formula>
            {m.variables?.length ? (
              <div className="flex flex-col gap-0.5">
                {m.variables.map((v) => (
                  <div key={v.symbol} className="font-mono text-[10px] text-zinc-500">
                    <span className="text-amber-300">{v.symbol}</span> — {v.meaning}
                  </div>
                ))}
              </div>
            ) : null}
            <div className="text-[11px] leading-5 text-zinc-300">{m.meaning}</div>
            <div className="text-[11px] leading-5 text-zinc-400">{m.intuition}</div>
            <div className="font-mono text-[10px] text-zinc-600">e.g. {m.example}</div>
            <ul className="flex flex-col gap-0.5 border-t border-zinc-800 pt-2">
              {m.limitations.map((l, i) => (
                <li key={i} className="text-[10px] leading-4 text-amber-300/70">
                  · {l}
                </li>
              ))}
            </ul>
            {m.paperIds?.length ? (
              <div className="flex flex-wrap gap-1">
                {m.paperIds.slice(0, 8).map((id) => (
                  <NeuroPaperLink key={id} id={id} short link />
                ))}
              </div>
            ) : (
              <Unstated what="corpus coverage" />
            )}
          </Card>
        ))}
      </div>
    </div>
  );
}
