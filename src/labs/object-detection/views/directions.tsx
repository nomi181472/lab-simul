"use client";

import { RESEARCH_DIRECTIONS, RESEARCH_OSCILLATIONS } from "@/labs/object-detection/data/directions";
import { Card, ConfidenceBadge, PaperLink, SectionTitle } from "@/labs/object-detection/ui";

export function DirectionsView() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Research directions</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          10 paradigm shifts the corpus documents, each with the limitation it escaped
          and the problems it created. Syntheses are flagged by evidence status.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {RESEARCH_DIRECTIONS.map((d) => (
          <Card key={d.id} tone="default">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <SectionTitle>{d.name}</SectionTitle>
              <ConfidenceBadge confidence={d.status} />
            </div>
            <div className="mt-1 font-mono text-[10px] text-zinc-500">
              first appearance {d.firstAppearance} · from {d.sourceIn.join(", ") || "—"}
            </div>

            <div className="mt-3 space-y-2">
              <div className="rounded-lg border border-rose-900/40 bg-rose-950/20 p-2">
                <div className="font-mono text-[9px] uppercase text-rose-400">escapes</div>
                <p className="mt-0.5 text-[11px] leading-4 text-zinc-400">{d.previousLimitation}</p>
              </div>
              <div className="rounded-lg border border-emerald-900/40 bg-emerald-950/20 p-2">
                <div className="font-mono text-[9px] uppercase text-emerald-400">unlocks</div>
                <p className="mt-0.5 text-[11px] leading-4 text-zinc-300">{d.unlockedCapability}</p>
              </div>
              <div className="rounded-lg border border-amber-900/40 bg-amber-950/20 p-2">
                <div className="font-mono text-[9px] uppercase text-amber-400">creates problems</div>
                <ul className="mt-0.5 list-inside space-y-0.5">
                  {d.newProblems.map((p, i) => (
                    <li key={i} className="text-[11px] leading-4 text-zinc-400">
                      {p}
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="mt-3 flex flex-wrap items-center gap-1.5">
              <span className="font-mono text-[9px] uppercase text-zinc-600">rep papers</span>
              {d.representativePapers.map((pid) => (
                <PaperLink key={pid} id={pid} />
              ))}
            </div>
            <p className="mt-2 font-mono text-[10px] text-zinc-600">driver: {d.driver}</p>
          </Card>
        ))}
      </div>

      {/* oscillations */}
      <div>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-zinc-500">
          Detected oscillations
        </h2>
        <div className="grid gap-4 lg:grid-cols-3">
          {RESEARCH_OSCILLATIONS.map((o) => (
            <Card key={o.pattern} tone="warn">
              <div className="font-mono text-[10px] uppercase tracking-wide text-amber-300">
                {o.status}
              </div>
              <p className="mt-2 text-[12px] font-medium leading-5 text-zinc-200">{o.pattern}</p>
              <div className="mt-3 space-y-2">
                {o.cycle.map((c, i) => (
                  <div key={i} className="flex gap-2">
                    <span className="font-mono text-[10px] text-amber-400">{c.year}</span>
                    <div>
                      <p className="text-[11px] leading-4 text-zinc-400">{c.state}</p>
                      <p className="font-mono text-[9px] text-zinc-600">{c.representativePapers}</p>
                    </div>
                  </div>
                ))}
              </div>
              <p className="mt-3 border-t border-zinc-800 pt-2 text-[10px] leading-4 text-zinc-500">
                {o.evidence}
              </p>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}