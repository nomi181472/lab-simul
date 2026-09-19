"use client";

import { DETECTION_SCENARIOS } from "@/labs/object-detection/data/edgecases";
import { Card, Badge, PaperLink, SectionTitle } from "@/labs/object-detection/ui";

export function EdgeView() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Edge case lab</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          Detection scenarios the corpus calls out as difficult, who attempted them, and
          what remains unsolved. Companion interactive scenes live in the Mathematics Lab
          under “Edge scenarios”.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {DETECTION_SCENARIOS.map((s) => (
          <Card key={s.name} tone="default">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <SectionTitle>{s.name}</SectionTitle>
              <Badge tone="zinc">first {s.firstObservedYear}</Badge>
            </div>

            <div className="mt-3 flex flex-wrap items-center gap-1.5">
              {s.papers.map((pid) => (
                <PaperLink key={pid} id={pid} />
              ))}
            </div>

            <div className="mt-3 space-y-2">
              <div>
                <div className="font-mono text-[9px] uppercase text-amber-400">why hard</div>
                <ul className="mt-1 list-inside space-y-0.5">
                  {s.difficulty.map((d, i) => (
                    <li key={i} className="text-[11px] leading-4 text-zinc-400">
                      {d}
                    </li>
                  ))}
                </ul>
              </div>
              {s.attemptedSolutions.length > 0 && (
                <div>
                  <div className="font-mono text-[9px] uppercase text-emerald-400">attempted</div>
                  <ul className="mt-1 list-inside space-y-0.5">
                    {s.attemptedSolutions.map((d, i) => (
                      <li key={i} className="text-[11px] leading-4 text-zinc-400">
                        {d}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              <div className="rounded-lg border border-rose-900/40 bg-rose-950/20 p-2">
                <div className="font-mono text-[9px] uppercase text-rose-400">remaining</div>
                <ul className="mt-1 list-inside space-y-0.5">
                  {s.remainingProblems.map((d, i) => (
                    <li key={i} className="text-[11px] leading-4 text-rose-200/70">
                      {d}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}