"use client";
import { useState } from "react";
import { Card, SectionTitle, Badge, Quote, PaperLink, truncate } from "@/labs/networking/ui";
import { ANALYSIS } from "@/labs/networking/data/analysis";
import { KB } from "@/labs/networking/data/kb";

const FAMILY_LABEL: Record<string, string> = Object.fromEntries(
  KB.solutions.map((f) => [f.familyId, f.label]),
);

export function EvolutionView() {
  const [tab, setTab] = useState<"problem" | "solution">("problem");

  return (
    <div className="space-y-6">
      <SectionTitle>Evolution explorer</SectionTitle>

      <Card tone="accent">
        <p className="text-[13px] leading-6 text-zinc-400">
          {tab === "problem"
            ? "Problem evolution: how each domain's stated problems travel across the four eras, with the earliest quoted statements in order."
            : "Solution evolution: how solution families rise and fall across eras, with a trend verdict computed from period counts."}
        </p>
        <div className="mt-3 flex gap-1">
          <button
            onClick={() => setTab("problem")}
            className={`rounded-md border px-2 py-0.5 font-mono text-[10px] ${
              tab === "problem"
                ? "border-sky-700/50 bg-sky-600/20 text-sky-300"
                : "border-zinc-700 bg-zinc-900 text-zinc-400"
            }`}
          >
            problem evolution
          </button>
          <button
            onClick={() => setTab("solution")}
            className={`rounded-md border px-2 py-0.5 font-mono text-[10px] ${
              tab === "solution"
                ? "border-sky-700/50 bg-sky-600/20 text-sky-300"
                : "border-zinc-700 bg-zinc-900 text-zinc-400"
            }`}
          >
            solution evolution
          </button>
        </div>
      </Card>

      {tab === "problem" ? (
        <div className="space-y-3">
          {ANALYSIS.problemEvolution.map((d) => {
            const max = Math.max(1, ...d.periodCounts.map((p) => p.count));
            return (
              <Card key={d.domainId}>
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <span className="text-[13px] font-medium text-zinc-100">{d.label}</span>
                  <span className="font-mono text-[10px] text-zinc-500">
                    {d.totalPapers} papers · {d.firstYear}–{d.lastYear}
                  </span>
                </div>
                <div className="mt-2 grid grid-cols-4 gap-2">
                  {d.periodCounts.map((p) => (
                    <div key={p.period}>
                      <div className="h-1.5 rounded bg-zinc-800">
                        <div
                          className="h-1.5 rounded bg-sky-600"
                          style={{ width: `${(p.count / max) * 100}%` }}
                        />
                      </div>
                      <p className="mt-1 font-mono text-[9px] text-zinc-500">
                        {p.period}: {p.count}
                      </p>
                    </div>
                  ))}
                </div>
                {d.problemTrail.length > 0 && (
                  <div className="mt-3 space-y-2 border-t border-zinc-800 pt-3">
                    <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
                      problem statements over time
                    </p>
                    {d.problemTrail.map((s, i) => (
                      <div key={i} className="flex gap-2">
                        <span className="shrink-0 font-mono text-[10px] text-sky-400">
                          {s.year ?? "?"}
                        </span>
                        <div className="min-w-0">
                          <PaperLink id={s.paperId} short />
                          <Quote>{truncate(s.text, 260)}</Quote>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      ) : (
        <div className="space-y-3">
          {ANALYSIS.solutionEvolution.map((s) => {
            const max = Math.max(1, ...s.periodCounts.map((p) => p.count));
            const tone =
              s.trend === "rising" ? "emerald" : s.trend === "declining" ? "rose" : "amber";
            return (
              <Card key={s.familyId}>
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <span className="text-[13px] font-medium text-zinc-100">
                    {FAMILY_LABEL[s.familyId] ?? s.familyId}
                  </span>
                  <div className="flex items-center gap-2">
                    <Badge tone={tone}>{s.trend}</Badge>
                    <span className="font-mono text-[10px] text-zinc-500">
                      {s.totalPapers} papers
                    </span>
                  </div>
                </div>
                <div className="mt-2 grid grid-cols-4 gap-2">
                  {s.periodCounts.map((p) => (
                    <div key={p.period}>
                      <div className="h-1.5 rounded bg-zinc-800">
                        <div
                          className="h-1.5 rounded bg-violet-600"
                          style={{ width: `${(p.count / max) * 100}%` }}
                        />
                      </div>
                      <p className="mt-1 font-mono text-[9px] text-zinc-500">
                        {p.period}: {p.count}
                      </p>
                    </div>
                  ))}
                </div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {s.paperIds.slice(0, 8).map((id) => (
                    <PaperLink key={id} id={id} short />
                  ))}
                </div>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
