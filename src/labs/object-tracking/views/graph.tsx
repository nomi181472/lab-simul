"use client";

import { useMemo, useState } from "react";
import { PAPERS, PAPER_BY_ID, successorsOf } from "@/labs/object-tracking/data/papers";
import { Card, Badge } from "@/labs/object-tracking/ui";

/* Interactive paper-to-paper graph: click a node → predecessors + successors light up. */

export function GraphView() {
  const [focus, setFocus] = useState<string | null>("T005");
  const [yearMin, setYearMin] = useState<number>(2014);
  const [concept, setConcept] = useState<string | null>(null);

  const concepts = useMemo(() => {
    const s = new Set<string>();
    for (const p of PAPERS) for (const c of p.concepts) s.add(c);
    return [...s].sort();
  }, []);

  const visible = useMemo(
    () => PAPERS.filter((p) => p.year >= yearMin && (!concept || p.concepts.includes(concept))),
    [yearMin, concept],
  );

  const focusPaper = focus ? PAPER_BY_ID[focus] : null;
  const predIds = useMemo(
    () => new Set((focusPaper?.relations ?? []).map((r) => r.to)),
    [focusPaper],
  );
  const succIds = useMemo(
    () => new Set(focus ? successorsOf(focus).map((p) => p.id) : []),
    [focus],
  );

  const years = [...new Set(visible.map((p) => p.year))].sort();

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Research graph</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          Nodes are papers, edges are <span className="font-mono text-[11px] text-zinc-300">builds-on / improves / extends / uses-as-baseline / addresses-limitation</span>.
          Click a paper to highlight its lineage; relations exist only when the paper itself cites or compares against the predecessor — never inferred.
        </p>
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1 font-mono text-[10px] text-zinc-500">
          YEAR ≥ {yearMin}
          <input type="range" min={2014} max={2026} step={1} value={yearMin} onChange={(e) => setYearMin(Number(e.target.value))} className="w-40 accent-sky-500" />
        </label>
        <label className="flex flex-col gap-1 font-mono text-[10px] text-zinc-500">
          CONCEPT
          <select value={concept ?? ""} onChange={(e) => setConcept(e.target.value || null)} className="rounded-lg border border-zinc-800 bg-zinc-900 px-2 py-1.5 text-xs text-zinc-200">
            <option value="">all concepts</option>
            {concepts.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>
        <div className="font-mono text-[10px] text-zinc-500">{visible.length} nodes</div>
        {focus && (
          <button onClick={() => setFocus(null)} className="rounded-md border border-zinc-700 px-2 py-1 font-mono text-[10px] text-zinc-300 hover:bg-zinc-900">clear focus ✕</button>
        )}
      </div>

      {focusPaper && (
        <Card tone="accent">
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded bg-sky-950 px-1.5 py-0.5 font-mono text-[11px] text-sky-300">{focusPaper.id}</span>
            <span className="text-sm font-semibold text-zinc-100">{focusPaper.title}</span>
          </div>
          <p className="mt-1 text-[12px] italic text-zinc-400">gap: {focusPaper.researchGap}</p>
          <div className="mt-2 grid gap-2 font-mono text-[11px] sm:grid-cols-2">
            <div className="rounded-lg border border-sky-800/50 p-2">
              <div className="text-sky-400">← predecessors ({focusPaper.relations.length})</div>
              {focusPaper.relations.map((r) => (
                <button key={r.to} onClick={() => setFocus(r.to)} className="mr-1 mt-1 rounded bg-zinc-800 px-1.5 py-0.5 text-sky-200 hover:bg-zinc-700">
                  {r.to} · {r.type}
                </button>
              ))}
              {focusPaper.relations.length === 0 && <div className="mt-1 text-zinc-600">root of its thread</div>}
            </div>
            <div className="rounded-lg border border-emerald-800/50 p-2">
              <div className="text-emerald-400">successors → ({succIds.size})</div>
              {[...succIds].map((s) => (
                <button key={s} onClick={() => setFocus(s)} className="mr-1 mt-1 rounded bg-zinc-800 px-1.5 py-0.5 text-emerald-200 hover:bg-zinc-700">{s}</button>
              ))}
              {succIds.size === 0 && <div className="mt-1 text-zinc-600">no structured successor yet (see audit)</div>}
            </div>
          </div>
          {focusPaper.relations[0]?.note && (
            <p className="mt-2 text-[11px] text-zinc-500">“{focusPaper.relations[0].note}”</p>
          )}
        </Card>
      )}

      <div className="space-y-4 overflow-x-auto">
        {years.map((y) => (
          <div key={y} className="min-w-0">
            <div className="mb-1.5 font-mono text-[11px] font-bold text-zinc-500 whitespace-nowrap">{y}</div>
            <div className="flex flex-wrap gap-1.5 min-w-max">
              {visible.filter((p) => p.year === y).map((p) => {
                const isFocus = p.id === focus;
                const isPred = predIds.has(p.id);
                const isSucc = succIds.has(p.id);
                const dim = focus && !isFocus && !isPred && !isSucc;
                return (
                  <button
                    key={p.id}
                    onClick={() => setFocus(p.id)}
                    title={`${p.id} · ${p.title}`}
                    className={`rounded-lg border px-2 py-1.5 font-mono text-[10px] whitespace-nowrap transition-all ${
                      isFocus ? "border-sky-400 bg-sky-600/25 text-sky-100"
                      : isPred ? "border-sky-700 bg-sky-950/60 text-sky-200"
                      : isSucc ? "border-emerald-700 bg-emerald-950/50 text-emerald-200"
                      : "border-zinc-800 bg-zinc-900/60 text-zinc-400 hover:border-zinc-600"
                    } ${dim ? "opacity-35" : ""}`}
                  >
                    {p.id} {p.shortTitle.slice(0, 14)}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
