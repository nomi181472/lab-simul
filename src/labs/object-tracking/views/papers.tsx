"use client";

import { useMemo, useState } from "react";
import { PAPERS, PAPER_BY_ID, successorsOf, predecessorsOf } from "@/labs/object-tracking/data/papers";
import { CONCEPT_BY_ID } from "@/labs/object-tracking/data/concepts";
import { Card, Badge, DifficultyBadge, TaskBadge, Formula } from "@/labs/object-tracking/ui";
import { useTrackingLab } from "@/labs/object-tracking/context";

/* Deep paper understanding: search/filter + full structured record. */

export function PapersView() {
  const { selectedPaper, setSelectedPaper, setMathSim } = useTrackingLab();
  const [q, setQ] = useState("");
  const [year, setYear] = useState<number | null>(null);
  const [tag, setTag] = useState<string | null>(null);

  const tags = useMemo(() => {
    const s = new Set<string>();
    for (const p of PAPERS) for (const t of p.tags) s.add(t);
    return [...s].sort();
  }, [PAPERS.length]);

  const filtered = useMemo(() => {
    return PAPERS.filter((p) => {
      if (year !== null && p.year !== year) return false;
      if (tag && !p.tags.includes(tag)) return false;
      if (q) {
        const hay = `${p.id} ${p.title} ${p.shortTitle} ${p.summary} ${p.authors.join(" ")} ${p.tags.join(" ")}`.toLowerCase();
        if (!hay.includes(q.toLowerCase())) return false;
      }
      return true;
    });
  }, [q, year, tag]);

  const active = selectedPaper ? PAPER_BY_ID[selectedPaper] : null;

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Paper explorer</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          {PAPERS.length} structured records. Click any paper for its full dossier: problem →
          gap → method → equations → experiments → limitations → impact.
        </p>
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1">
          <label className="font-mono text-[9px] uppercase text-zinc-600">search</label>
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="SORT, Siamese, Kalman…" className="w-60 rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-sky-600 focus:outline-none" />
        </div>
        <div className="flex flex-col gap-1">
          <label className="font-mono text-[9px] uppercase text-zinc-600">year</label>
          <select value={year ?? ""} onChange={(e) => setYear(e.target.value ? Number(e.target.value) : null)} className="rounded-lg border border-zinc-800 bg-zinc-900 px-2 py-2 text-sm text-zinc-200">
            <option value="">all</option>
            {[...new Set(PAPERS.map((p) => p.year))].sort((a, b) => b - a).map((y) => <option key={y} value={y}>{y}</option>)}
          </select>
        </div>
        <div className="font-mono text-[10px] text-zinc-500">{filtered.length} / {PAPERS.length}</div>
      </div>

      <div className="flex flex-wrap gap-1.5">
        {tags.slice(0, 24).map((t) => (
          <button key={t} onClick={() => setTag(tag === t ? null : t)} className={`rounded-full border px-2 py-0.5 text-[10px] ${tag === t ? "border-sky-600 bg-sky-600/20 text-sky-200" : "border-zinc-800 text-zinc-500 hover:text-zinc-300"}`}>{t}</button>
        ))}
      </div>

      {active && (
        <Card tone="accent" className="space-y-4 overflow-hidden">
          <div className="flex flex-wrap items-start justify-between gap-2">
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <span className="rounded bg-sky-950 px-1.5 py-0.5 font-mono text-[11px] text-sky-300">{active.id}</span>
                <Badge tone="zinc">{active.year}</Badge>
                <DifficultyBadge d={active.difficulty} />
                <TaskBadge t={active.task} />
                {active.venue && <Badge tone="emerald">{active.venue}</Badge>}
              </div>
              <h2 className="mt-2 text-lg font-semibold text-zinc-50 truncate">{active.title}</h2>
              <p className="font-mono text-[11px] text-zinc-500 truncate">{active.authors.join(", ")} · arXiv:{active.arxiv} · {active.fileName}</p>
            </div>
            <button onClick={() => setSelectedPaper(null)} className="rounded-md border border-zinc-700 px-2 py-1 font-mono text-[11px] text-zinc-300 hover:bg-zinc-900 flex-shrink-0">close ✕</button>
          </div>

          <p className="text-[13px] leading-6 text-zinc-300">{active.summary}</p>

          <div className="grid gap-3 lg:grid-cols-2">
            <div className="rounded-lg border border-zinc-800 p-3 min-w-0">
              <div className="font-mono text-[10px] uppercase tracking-wide text-rose-300">problem</div>
              <p className="mt-1 text-[12px] leading-5 text-zinc-400">{active.problem}</p>
              <div className="mt-2 font-mono text-[10px] uppercase tracking-wide text-amber-300">research gap</div>
              <p className="mt-1 text-[12px] italic leading-5 text-zinc-300">{active.researchGap}</p>
            </div>
            <div className="rounded-lg border border-zinc-800 p-3 min-w-0">
              <div className="font-mono text-[10px] uppercase tracking-wide text-sky-300">contributions</div>
              <ul className="mt-1 list-disc space-y-1 pl-4 text-[12px] leading-5 text-zinc-300">
                {active.contribution.map((c, i) => <li key={i}>{c}</li>)}
              </ul>
            </div>
          </div>

          <div className="rounded-lg border border-zinc-800 p-3 min-w-0">
            <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">pipeline</div>
            <div className="mt-2 flex flex-wrap items-center gap-1 font-mono text-[11px]">
              {active.method.pipeline.map((s, i) => (
                <span key={s} className="flex items-center gap-1">
                  <span className="rounded border border-sky-800/60 bg-sky-950/30 px-2 py-1 text-sky-200 whitespace-nowrap">{s}</span>
                  {i < active.method.pipeline.length - 1 && <span className="text-zinc-600">→</span>}
                </span>
              ))}
            </div>
            <p className="mt-2 text-[12px] leading-5 text-zinc-400">{active.method.architecture}</p>
            <div className="mt-2 grid gap-1.5 text-[12px] sm:grid-cols-2">
              {Object.entries({ motionModel: active.method.motionModel, appearanceModel: active.method.appearanceModel, association: active.method.association, reid: active.method.reid, detectionDependency: active.method.detectionDependency, trackManagement: active.method.trackManagement, optimization: active.method.optimization, loss: active.method.loss }).filter(([, v]) => v).map(([k, v]) => (
                <p key={k} className="text-zinc-500 min-w-0"><span className="font-mono text-[10px] uppercase text-zinc-600">{k} · </span><span className="text-zinc-400 truncate">{v}</span></p>
              ))}
            </div>
          </div>

          {active.equations.length > 0 && (
            <div className="space-y-2">
              <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">equations ({active.equations.length})</div>
              {active.equations.map((eq) => (
                <div key={eq.id} className="rounded-lg border border-zinc-800 bg-black/20 p-3 min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-[13px] font-semibold text-zinc-100 truncate">{eq.label}</span>
                    {eq.simulator && <button onClick={() => setMathSim(eq.simulator!)} className="rounded border border-sky-800 px-1.5 py-0.5 font-mono text-[10px] text-sky-300 hover:bg-sky-950/40 flex-shrink-0">simulate →</button>}
                  </div>
                  <div className="mt-2"><Formula>{eq.formula}</Formula></div>
                  <p className="mt-1.5 text-[12px] text-zinc-400">{eq.intuition}</p>
                  <p className="text-[11px] text-zinc-500">why: {eq.why} · where: {eq.where}</p>
                  {eq.params && <p className="text-[11px] text-zinc-500">params: {eq.params}</p>}
                </div>
              ))}
            </div>
          )}

          <div className="grid gap-3 lg:grid-cols-3">
            <div className="rounded-lg border border-zinc-800 p-3 min-w-0">
              <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">experiments</div>
              <p className="mt-1 font-mono text-[11px] text-zinc-400">datasets: {active.datasets.join(" · ") || "—"}</p>
              <p className="font-mono text-[11px] text-zinc-400">metrics: {active.metrics.join(" · ") || "—"}</p>
              <p className="font-mono text-[11px] text-zinc-400">baselines: {active.baselines.join(" · ") || "—"}</p>
              <ul className="mt-1.5 list-disc space-y-1 pl-4 text-[12px] text-zinc-300">{active.results.map((r, i) => <li key={i}>{r}</li>)}</ul>
              {active.ablations.length > 0 && <div className="mt-1.5 font-mono text-[10px] uppercase text-zinc-600">ablations</div>}
              <ul className="list-disc space-y-1 pl-4 text-[11px] text-zinc-500">{active.ablations.map((a, i) => <li key={i}>{a}</li>)}</ul>
            </div>
            <div className="rounded-lg border border-zinc-800 p-3 min-w-0">
              <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">limitations</div>
              <div className="mt-1 font-mono text-[10px] uppercase text-amber-300">author-stated</div>
              <ul className="list-disc space-y-1 pl-4 text-[12px] text-zinc-300">{active.limitations.authorStated.map((l, i) => <li key={i}>{l}</li>)}</ul>
              <div className="mt-1.5 font-mono text-[10px] uppercase text-violet-300">lab synthesis (not author claims)</div>
              <ul className="list-disc space-y-1 pl-4 text-[11px] italic text-zinc-500">{active.limitations.evident.map((l, i) => <li key={i}>{l}</li>)}</ul>
              {active.assumptions && <p className="mt-1.5 text-[11px] text-zinc-500">assumes: {active.assumptions.join(" ")}</p>}
              {active.computation && <p className="mt-1 text-[11px] text-zinc-500">compute: {active.computation}</p>}
            </div>
            <div className="rounded-lg border border-zinc-800 p-3 min-w-0">
              <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">lineage & impact</div>
              <div className="mt-1 font-mono text-[11px] text-sky-300 truncate">← {active.relations.length ? active.relations.map((r) => `${r.to} (${r.type})`).join(" · ") : "root"}</div>
              <div className="mt-1 font-mono text-[11px] text-emerald-300 truncate">→ {successorsOf(active.id).map((s) => s.id).join(" · ") || "open thread"}</div>
              <p className="mt-1.5 text-[12px] leading-5 text-zinc-400">{active.impact}</p>
              <div className="mt-1.5 flex flex-wrap gap-1">
                {active.concepts.map((c) => <Badge key={c} tone="violet">{CONCEPT_BY_ID[c]?.name ?? c}</Badge>)}
              </div>
            </div>
          </div>
        </Card>
      )}

      <div className="space-y-2">
        {filtered.map((p) => (
          <button key={p.id} onClick={() => setSelectedPaper(p.id)} className={`w-full rounded-xl border p-3 text-left transition-colors ${selectedPaper === p.id ? "border-sky-600 bg-sky-950/20" : "border-zinc-800 bg-zinc-900/40 hover:border-sky-700/60"}`}>
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="rounded bg-sky-950 px-1.5 py-0.5 font-mono text-[10px] text-sky-300 flex-shrink-0">{p.id}</span>
              <Badge tone="zinc">{p.year}</Badge>
              <DifficultyBadge d={p.difficulty} />
              <TaskBadge t={p.task} />
              {p.tags.slice(0, 3).map((t) => <Badge key={t} tone="zinc">{t}</Badge>)}
            </div>
            <div className="mt-1 text-sm font-semibold text-zinc-100 truncate">{p.shortTitle} <span className="font-normal text-zinc-500">· {p.title}</span></div>
            <p className="mt-0.5 line-clamp-2 text-[12px] text-zinc-400">{p.summary}</p>
          </button>
        ))}
        {filtered.length === 0 && <Card tone="warn"><p className="text-sm text-zinc-400">No papers match.</p></Card>}
      </div>
    </div>
  );
}

// re-export for graph lineage use
export { predecessorsOf };
