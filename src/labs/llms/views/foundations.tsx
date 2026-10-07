"use client";

import { Card, SectionTitle, Badge, CiteBadge } from "@/labs/llms/ui";
import { PAPERS } from "@/labs/llms/data/papers";
import { MANIFEST } from "@/labs/llms/data/manifest";

/** How concentrated the corpus is in its most-cited head. */
function headShare(n: number): number {
  const sorted = [...PAPERS].sort((a, b) => b.citations - a.citations);
  const head = sorted.slice(0, n).reduce((s, p) => s + p.citations, 0);
  return MANIFEST.totalCitations ? (head / MANIFEST.totalCitations) * 100 : 0;
}

export function FoundationsView() {
  const sorted = [...PAPERS].sort((a, b) => b.citations - a.citations);
  const top = sorted.slice(0, 60);

  return (
    <div className="space-y-6">
      <SectionTitle>Citation order</SectionTitle>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Card tone="accent">
          <p className="font-mono text-[10px] uppercase tracking-wide text-cyan-400">
            top 10 share
          </p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">
            {headShare(10).toFixed(1)}%
          </p>
          <p className="mt-1 font-mono text-[10px] text-zinc-500">
            of all corpus citations sit in ten papers
          </p>
        </Card>
        <Card>
          <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            top 50 share
          </p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">
            {headShare(50).toFixed(1)}%
          </p>
          <p className="mt-1 font-mono text-[10px] text-zinc-500">top 50 papers</p>
        </Card>
        <Card>
          <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            median cites
          </p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">
            {sorted[Math.floor(sorted.length / 2)]?.citations.toLocaleString() ?? 0}
          </p>
          <p className="mt-1 font-mono text-[10px] text-zinc-500">
            long tail is real, not an artefact
          </p>
        </Card>
        <Card>
          <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            uncited
          </p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">
            {sorted.filter((p) => p.citations === 0).length}
          </p>
          <p className="mt-1 font-mono text-[10px] text-zinc-500">
            recent preprints with no citing work yet
          </p>
        </Card>
      </div>

      <Card>
        <SectionTitle>Most cited {top.length}</SectionTitle>
        <p className="mt-2 text-[13px] leading-6 text-zinc-400">
          Ordered by OpenAlex <span className="font-mono text-zinc-300">cited_by_count</span>,
          which is a real citation count rather than a fuzzy title match. Ties break on
          year, then title, so the order is stable across regenerations.
        </p>
        <ol className="mt-4 space-y-1">
          {top.map((p, i) => (
            <li
              key={p.id}
              className="flex items-start gap-3 rounded-lg px-2 py-1.5 hover:bg-zinc-900/60"
            >
              <span className="mt-0.5 w-6 shrink-0 text-right font-mono text-[11px] text-zinc-600">
                {i + 1}
              </span>
              <span className="min-w-0 flex-1">
                <span className="block text-[12px] leading-5 text-zinc-300">{p.title}</span>
                <span className="mt-1 flex flex-wrap items-center gap-1.5">
                  <Badge tone="cyan">{p.id}</Badge>
                  {p.year && <Badge>{p.year}</Badge>}
                  {p.venue && <Badge>{p.venue.slice(0, 30)}</Badge>}
                  {p.arxiv && <Badge tone="violet">arXiv:{p.arxiv}</Badge>}
                </span>
              </span>
              <CiteBadge n={p.citations} />
            </li>
          ))}
        </ol>
      </Card>
    </div>
  );
}