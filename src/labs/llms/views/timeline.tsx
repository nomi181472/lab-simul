"use client";

import { useTfLab } from "@/labs/llms/context";
import { Card, SectionTitle, Badge, CiteBadge } from "@/labs/llms/ui";
import { PAPERS } from "@/labs/llms/data/papers";

export function TimelineView() {
  const { setSelectedPaper } = useTfLab();

  const byYear = new Map<number, typeof PAPERS>();
  for (const p of PAPERS) {
    const y = p.year ?? 0;
    if (!byYear.has(y)) byYear.set(y, []);
    byYear.get(y)!.push(p);
  }
  const years = [...byYear.keys()].sort((a, b) => a - b);
  const peak = Math.max(...[...byYear.values()].map((v) => v.length), 1);
  const totalCitations = PAPERS.reduce((n, p) => n + p.citations, 0);

  return (
    <div className="space-y-6">
      <SectionTitle>Timeline · {years[0]} to {years[years.length - 1]}</SectionTitle>

      <Card tone="accent">
        <p className="text-[13px] leading-6 text-zinc-400">
          The corpus begins in 2017 so the originals are in it: Vaswani et al. propose the
          transformer, GPT-1/2 and BERT establish the scale, and everything after is
          measured against those. The compression literature then grows up around them,
          which is the arc this lab follows from attention to micro-LLM.
        </p>
      </Card>

      <Card>
        <div className="space-y-1">
          {years.map((y) => {
            const ps = byYear.get(y)!;
            const cites = ps.reduce((n, p) => n + p.citations, 0);
            const share = Math.round((cites / totalCitations) * 100);
            return (
              <div key={y} className="flex items-center gap-3 py-1">
                <span className="w-11 shrink-0 font-mono text-[11px] text-zinc-500">
                  {y || "?"}
                </span>
                <div className="h-5 w-40 shrink-0 overflow-hidden rounded bg-zinc-800/60">
                  <div
                    className="h-full rounded bg-cyan-700/60"
                    style={{ width: `${(ps.length / peak) * 100}%` }}
                  />
                </div>
                <span className="w-12 shrink-0 text-right font-mono text-[11px] text-zinc-400">
                  {ps.length} papers
                </span>
                <span className="w-20 shrink-0 text-right font-mono text-[11px] text-zinc-500">
                  {cites.toLocaleString()}
                </span>
                <span className="w-9 shrink-0 text-right font-mono text-[10px] text-zinc-600">
                  {share}%
                </span>
              </div>
            );
          })}
        </div>
        <p className="mt-3 font-mono text-[10px] text-zinc-600">
          bar = paper count · right columns = citations held and share of corpus citations
        </p>
      </Card>

      <SectionTitle>Paper by paper</SectionTitle>
      {years.map((y) => (
        <Card key={y}>
          <div className="mb-3 flex items-baseline gap-2">
            <span className="font-mono text-sm text-zinc-200">{y || "?"}</span>
            <span className="font-mono text-[10px] text-zinc-600">
              {byYear.get(y)!.length} papers
            </span>
          </div>
          <div className="space-y-2">
            {byYear
              .get(y)!
              .sort((a, b) => b.citations - a.citations)
              .map((p) => (
                <button
                  key={p.id}
                  onClick={() => setSelectedPaper(p.id)}
                  className="flex w-full items-start gap-3 rounded-lg border border-transparent px-2 py-1.5 text-left hover:border-zinc-700 hover:bg-zinc-900/60"
                >
                  <span className="mt-0.5 shrink-0">
                    <Badge tone="cyan">{p.id}</Badge>
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-[12px] leading-5 text-zinc-300">
                      {p.title}
                    </span>
                  </span>
                  <CiteBadge n={p.citations} />
                </button>
              ))}
          </div>
        </Card>
      ))}
    </div>
  );
}