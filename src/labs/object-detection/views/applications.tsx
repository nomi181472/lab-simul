"use client";

import { APPLICATIONS, INDUSTRY_LOOP } from "@/labs/object-detection/data/applications";
import { Card, Badge, ConfidenceBadge, PaperLink, SectionTitle } from "@/labs/object-detection/ui";

export function ApplicationsView() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Applications</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          Every application is tied to the research capability that produced it and the
          real-world constraint it feeds back into research. Claims are corpus-grounded.
        </p>
      </div>

      {/* closed loop */}
      <Card tone="accent">
        <SectionTitle>Research ↔ industry feedback loop</SectionTitle>
        <div className="mt-3 flex flex-col flex-wrap items-stretch gap-2 md:flex-row md:items-center">
          {INDUSTRY_LOOP.map((s, i) => (
            <div key={s.label} className="flex flex-1 flex-col items-center gap-1.5">
              <div className="w-full rounded-lg border border-emerald-800/50 bg-emerald-950/30 px-3 py-2 text-center">
                <div className="font-mono text-[10px] uppercase tracking-wide text-emerald-400">
                  {s.label}
                </div>
                <p className="mt-1 text-[10px] leading-4 text-zinc-400">{s.detail}</p>
              </div>
              {i < INDUSTRY_LOOP.length - 1 && (
                <span className="font-mono text-xs text-emerald-600">→</span>
              )}
            </div>
          ))}
        </div>
      </Card>

      {/* application cards */}
      <div className="grid gap-4 lg:grid-cols-2">
        {APPLICATIONS.map((a) => (
          <Card key={a.id} tone="default">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <SectionTitle>{a.name}</SectionTitle>
              <div className="flex items-center gap-1.5">
                <ConfidenceBadge confidence={a.status} />
                <Badge tone="zinc">since {a.firstAppearance}</Badge>
              </div>
            </div>

            <div className="mt-3 space-y-2 text-[12px] leading-5">
              <p className="text-zinc-300">
                <span className="font-mono text-[9px] uppercase text-emerald-400">capability </span>
                {a.researchCapability}
              </p>
              <p className="text-zinc-400">
                <span className="font-mono text-[9px] uppercase text-sky-400">why useful </span>
                {a.whyUseful}
              </p>
              <p className="text-zinc-400">
                <span className="font-mono text-[9px] uppercase text-amber-400">constraints </span>
                {a.constraints}
              </p>
              <p className="text-zinc-400">
                <span className="font-mono text-[9px] uppercase text-violet-400">evolution </span>
                {a.detectorEvolution}
              </p>
              <p className="text-zinc-500">
                <span className="font-mono text-[9px] uppercase text-rose-400">edge cases </span>
                {a.edgeCases}
              </p>
            </div>

            <div className="mt-3 flex flex-wrap items-center gap-1.5">
              {a.paperIds.map((pid) => (
                <PaperLink key={pid} id={pid} />
              ))}
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}