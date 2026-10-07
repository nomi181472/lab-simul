"use client";

import { Card, SectionTitle, Badge } from "@/labs/llms/ui";
import { PAPERS } from "@/labs/llms/data/papers";
import { DATASETS } from "@/labs/llms/data/datasets";

export function DatasetsView() {
  const byPaper: Record<string, string[]> = {};
  for (const p of PAPERS)
    for (const r of p.results) {
      byPaper[r.dataset] = byPaper[r.dataset] ?? [];
      byPaper[r.dataset].push(p.id);
    }

  const rows = DATASETS.map((d) => ({
    key: d.key,
    label: d.label,
    n: (byPaper[d.key] ?? []).length,
    papers: byPaper[d.key] ?? [],
  }))
    .filter((r) => r.n > 0)
    .sort((a, b) => b.n - a.n);

  const total = rows.reduce((n, r) => n + r.n, 0);

  return (
    <div className="space-y-6">
      <SectionTitle>Benchmarks</SectionTitle>

      <Card tone="accent">
        <p className="text-[13px] leading-6 text-zinc-400">
          A benchmark counts as present when a paper names it in its own text. That is a
          weaker claim than &quot;reports a number for it&quot; — extraction of tabulated
          values is deliberately out of scope for this lab, because reading a results table
          reliably from `pdftotext` output needs layout-mode parsing and this corpus mixes
          two-column and single-column bodies. The quotes in the paper view show exactly
          what sentence each hit came from.
        </p>
      </Card>

      <Card>
        <SectionTitle>Coverage</SectionTitle>
        <p className="mt-2 font-mono text-[10px] text-zinc-500">
          {rows.length} benchmarks named · {total} paper-benchmark mentions across{" "}
          {PAPERS.length} papers
        </p>
        <div className="mt-4 space-y-2">
          {rows.map((r) => {
            const pct = Math.round((r.n / PAPERS.length) * 100);
            return (
              <div key={r.key} className="flex items-center gap-2">
                <span className="w-32 shrink-0 truncate font-mono text-[10px] text-zinc-300">
                  {r.label}
                </span>
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-zinc-800">
                  <div
                    className="h-full rounded-full bg-cyan-600"
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className="w-8 shrink-0 text-right font-mono text-[10px] text-zinc-500">
                  {r.n}
                </span>
              </div>
            );
          })}
        </div>
      </Card>

      <SectionTitle>Which papers use which benchmark</SectionTitle>
      <div className="grid gap-3 md:grid-cols-2">
        {rows.map((r) => (
          <Card key={r.key}>
            <div className="flex items-center justify-between gap-2">
              <span className="text-[13px] font-medium text-zinc-200">{r.label}</span>
              <Badge tone="cyan">{r.n}</Badge>
            </div>
            <div className="mt-2 flex flex-wrap gap-1">
              {r.papers.slice(0, 40).map((id) => (
                <span
                  key={id}
                  className="rounded bg-cyan-950 px-1 py-0.5 font-mono text-[10px] text-cyan-300"
                >
                  {id}
                </span>
              ))}
              {r.papers.length > 40 && (
                <span className="font-mono text-[10px] text-zinc-600">
                  +{r.papers.length - 40}
                </span>
              )}
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}