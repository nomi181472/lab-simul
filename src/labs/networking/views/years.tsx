"use client";

import { Card, SectionTitle, Badge, PaperLink } from "@/labs/networking/ui";
import { ANALYSIS } from "@/labs/networking/data/analysis";
import { KB } from "@/labs/networking/data/kb";

const DOMAIN_LABEL: Record<string, string> = Object.fromEntries(
  KB.domains.map((d) => [d.id, d.label]),
);

export function YearsView() {
  const rows = ANALYSIS.years;
  const maxCount = Math.max(...rows.map((r) => r.count));
  const maxCites = Math.max(...rows.map((r) => r.citations));

  return (
    <div className="space-y-6">
      <SectionTitle>Year-by-year comparison</SectionTitle>

      <Card tone="accent">
        <p className="text-[13px] leading-6 text-zinc-400">
          What changed each year: paper volume, citation weight, dominant problem domains
          and the most-cited work of the year. Bars are scaled within the corpus.
        </p>
      </Card>

      <div className="space-y-3">
        {rows.map((r) => (
          <Card key={r.year}>
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <div className="flex items-baseline gap-3">
                <span className="font-mono text-lg font-semibold text-sky-300">{r.year}</span>
                <span className="font-mono text-[11px] text-zinc-400">
                  {r.count} papers · {r.citations.toLocaleString()} citations · median{" "}
                  {r.medianCitations}
                </span>
              </div>
              <div className="flex gap-1">
                {Object.entries(r.sources).map(([s, n]) => (
                  <Badge key={s} tone={s === "full-text" ? "emerald" : "amber"}>
                    {s}: {n as number}
                  </Badge>
                ))}
              </div>
            </div>

            <div className="mt-2 grid gap-3 lg:grid-cols-2">
              <div>
                <div className="h-1.5 rounded bg-zinc-800">
                  <div
                    className="h-1.5 rounded bg-sky-600"
                    style={{ width: `${(r.count / maxCount) * 100}%` }}
                  />
                </div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {r.topDomains.map(([d, n]) => (
                    <Badge key={d}>
                      {DOMAIN_LABEL[d] ?? d} ({n})
                    </Badge>
                  ))}
                </div>
              </div>
              <div className="space-y-1">
                {r.topPapers.map((p) => (
                  <div key={p.id} className="flex items-baseline justify-between gap-2">
                    <span className="truncate text-[11px] text-zinc-300">
                      <PaperLink id={p.id} />
                    </span>
                    <span className="shrink-0 font-mono text-[10px] text-zinc-500">
                      {p.citations.toLocaleString()} cites
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="mt-2 h-1 rounded bg-zinc-800">
              <div
                className="h-1 rounded bg-emerald-700/70"
                style={{ width: `${(r.citations / maxCites) * 100}%` }}
              />
            </div>
            <p className="mt-1 font-mono text-[9px] text-zinc-600">
              citation weight relative to peak year
            </p>
          </Card>
        ))}
      </div>
    </div>
  );
}
