"use client";

import { useMemo, useState } from "react";
import { PAPERS } from "@/labs/object-tracking/data/papers";
import { SIMULATOR_META, SimShell } from "@/labs/object-tracking/sims";
import { Card, Badge, Formula } from "@/labs/object-tracking/ui";
import { useTrackingLab } from "@/labs/object-tracking/context";

/* Equations you can operate: every equation + its simulator + its papers. */

export function MathView() {
  const { mathSim, setMathSim } = useTrackingLab();
  const [q, setQ] = useState("");
  const equations = useMemo(() => {
    const out: { paperId: string; short: string; eq: (typeof PAPERS)[number]["equations"][number] }[] = [];
    for (const p of PAPERS) for (const eq of p.equations) out.push({ paperId: p.id, short: p.shortTitle, eq });
    const needle = q.toLowerCase();
    return needle ? out.filter((e) => `${e.eq.label} ${e.eq.formula} ${e.eq.intuition}`.toLowerCase().includes(needle)) : out;
  }, [q, PAPERS.length]);

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Math lab</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          Every equation carries <span className="text-zinc-200">variables → intuition → why it exists →
          where it runs → what its parameters do</span>. {equations.length} equations structured so far.
        </p>
      </div>

      <div>
        <div className="mb-2 font-mono text-[11px] uppercase tracking-widest text-zinc-500">Simulators</div>
        <div className="flex flex-wrap gap-1.5">
          {SIMULATOR_META.map((s) => (
            <button
              key={s.id}
              onClick={() => setMathSim(s.id)}
              title={s.blurb}
              className={`rounded-lg border px-2.5 py-1.5 text-left text-[11px] ${mathSim === s.id ? "border-sky-500 bg-sky-600/20 text-sky-100" : "border-zinc-800 text-zinc-400 hover:border-zinc-600"}`}
            >
              <div className="font-semibold">{s.label}</div>
              <div className="font-mono text-[9px] opacity-70">{s.blurb}</div>
            </button>
          ))}
        </div>
        <div className="mt-3"><SimShell id={mathSim} /></div>
      </div>

      <div>
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="filter equations…" className="w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-sky-600 focus:outline-none" />
        <div className="mt-2 font-mono text-[10px] text-zinc-500">{equations.length} equations</div>
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        {equations.map(({ paperId, short, eq }) => (
          <Card key={`${paperId}-${eq.id}`}>
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded bg-sky-950 px-1.5 py-0.5 font-mono text-[10px] text-sky-300">{paperId}</span>
              <span className="font-mono text-[10px] text-zinc-500">{short}</span>
              {eq.simulator && <Badge tone="sky">sim: {eq.simulator}</Badge>}
            </div>
            <div className="mt-1.5 text-sm font-semibold text-zinc-100">{eq.label}</div>
            <div className="mt-2"><Formula>{eq.formula}</Formula></div>
            <div className="mt-2 space-y-1.5 text-[12px] leading-5">
              <p className="text-zinc-400"><span className="font-mono text-[10px] uppercase text-zinc-600">intuition · </span>{eq.intuition}</p>
              <p className="text-zinc-400"><span className="font-mono text-[10px] uppercase text-zinc-600">why · </span>{eq.why}</p>
              <p className="text-zinc-500"><span className="font-mono text-[10px] uppercase text-zinc-600">where · </span>{eq.where}</p>
              {eq.params && <p className="text-zinc-500"><span className="font-mono text-[10px] uppercase text-zinc-600">params · </span>{eq.params}</p>}
              <div className="rounded bg-black/30 p-2 font-mono text-[10px] leading-4 text-zinc-500">
                {eq.variables.map((v) => <div key={v.symbol}><span className="text-amber-300">{v.symbol}</span> — {v.meaning}</div>)}
              </div>
              {eq.simulator && (
                <button onClick={() => setMathSim(eq.simulator!)} className="rounded border border-sky-800 px-2 py-1 font-mono text-[10px] text-sky-300 hover:bg-sky-950/40">
                  operate this equation →
                </button>
              )}
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
