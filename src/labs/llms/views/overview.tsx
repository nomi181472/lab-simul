"use client";

import { useTfLab } from "@/labs/llms/context";
import { Card, SectionTitle, Badge, CiteBadge, Quote } from "@/labs/llms/ui";
import { PAPERS } from "@/labs/llms/data/papers";
import { MANIFEST } from "@/labs/llms/data/manifest";
import { ALL_CHANGE_OPERATORS } from "@/labs/llms/data/vocab";

function count(map: Record<string, number>) {
  return Object.entries(map).sort((a, b) => b[1] - a[1]);
}

export function OverviewView() {
  const { setSection } = useTfLab();

  const years = PAPERS.map((p) => p.year ?? 0).filter(Boolean);
  const minY = years.length ? Math.min(...years) : 0;
  const maxY = years.length ? Math.max(...years) : 0;

  const withArch = PAPERS.filter((p) => p.architecture.layers.length > 0).length;
  const withOps = PAPERS.filter((p) => p.changes.length > 0).length;
  const totalOps = PAPERS.reduce((n, p) => n + p.changes.length, 0);

  const opCounts: Record<string, number> = {};
  for (const p of PAPERS)
    for (const c of p.changes) opCounts[c.operator] = (opCounts[c.operator] ?? 0) + 1;

  const objectives: Record<string, number> = {};
  for (const p of PAPERS)
    for (const o of p.objectives) objectives[o.kind] = (objectives[o.kind] ?? 0) + 1;

  const top = [...PAPERS].sort((a, b) => b.citations - a.citations).slice(0, 5);

  return (
    <div className="space-y-6">
      <SectionTitle>
        From attention to the micro-LLM, in {MANIFEST.total} papers
      </SectionTitle>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Card tone="accent">
          <p className="font-mono text-[10px] uppercase tracking-wide text-cyan-400">
            corpus
          </p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">{MANIFEST.total}</p>
          <p className="mt-1 font-mono text-[10px] text-zinc-500">
            {MANIFEST.window} · most cited first
          </p>
        </Card>
        <Card>
          <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            citations
          </p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">
            {MANIFEST.totalCitations.toLocaleString()}
          </p>
          <p className="mt-1 font-mono text-[10px] text-zinc-500">OpenAlex cited_by_count</p>
        </Card>
        <Card>
          <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            architecture read
          </p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">
            {withArch}
            <span className="text-sm text-zinc-500">/{PAPERS.length}</span>
          </p>
          <p className="mt-1 font-mono text-[10px] text-zinc-500">
            papers with a quoted structural stack
          </p>
        </Card>
        <Card>
          <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            transformations
          </p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">{totalOps}</p>
          <p className="mt-1 font-mono text-[10px] text-zinc-500">
            across {withOps} papers · {ALL_CHANGE_OPERATORS.length} operator kinds
          </p>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <SectionTitle>Two axes, one tension</SectionTitle>
          <p className="mt-2 text-[13px] leading-6 text-zinc-400">
            Every paper in this corpus is pulling on the same trade-off: a model gets
            cheaper to run by being smaller, and worse at its job by being smaller. The
            left column is how often a paper names a quality objective; the right column
            is how often it names a cost objective.
          </p>
          <div className="mt-4 space-y-2">
            {Object.entries(objectives)
              .sort((a, b) => b[1] - a[1])
              .map(([kind, n]) => {
                const pct = Math.round((n / PAPERS.length) * 100);
                return (
                  <div key={kind} className="flex items-center gap-2">
                    <span className="w-24 shrink-0 font-mono text-[10px] text-zinc-400">
                      {kind}
                    </span>
                    <div className="h-2 flex-1 overflow-hidden rounded-full bg-zinc-800">
                      <div
                        className="h-full rounded-full bg-cyan-600"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <span className="w-10 shrink-0 text-right font-mono text-[10px] text-zinc-500">
                      {n}
                    </span>
                  </div>
                );
              })}
          </div>
        </Card>

        <Card>
          <SectionTitle>What the field applied</SectionTitle>
          <p className="mt-2 text-[13px] leading-6 text-zinc-400">
            Modification operators, counted by how many papers name them in their own
            text. This is the vocabulary that took a 70B-parameter model and put it on a
            phone.
          </p>
          <div className="mt-4 flex flex-wrap gap-1.5">
            {ALL_CHANGE_OPERATORS.map((op) => {
              const n = opCounts[op] ?? 0;
              return (
                <span
                  key={op}
                  className={`inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 font-mono text-[10px] ${
                    n
                      ? "border-zinc-700 bg-zinc-900 text-zinc-300"
                      : "border-zinc-800/60 bg-zinc-900/40 text-zinc-600"
                  }`}
                  title={n ? `${n} papers` : "no paper in the corpus names this"}
                >
                  {op}
                  <span className="text-zinc-500">{n}</span>
                </span>
              );
            })}
          </div>
          <button
            onClick={() => setSection("operators")}
            className="mt-4 rounded-md border border-cyan-700/50 bg-cyan-600/20 px-3 py-1.5 text-[11px] font-medium text-cyan-300"
          >
            Open the operator reference
          </button>
        </Card>
      </div>

      <Card>
        <SectionTitle>What the field actually leaned on</SectionTitle>
        <p className="mt-2 text-[13px] leading-6 text-zinc-400">
          The corpus is ordered by descending real citation count, so the head of the list
          is the field&apos;s actual centre of gravity.
        </p>
        <div className="mt-4 space-y-2">
          {top.map((p, i) => (
            <div key={p.id} className="flex items-start gap-3">
              <span className="mt-0.5 w-5 shrink-0 font-mono text-[11px] text-zinc-600">
                {i + 1}
              </span>
              <div className="min-w-0 flex-1">
                <p className="text-[13px] leading-5 text-zinc-200">{p.title}</p>
                <div className="mt-1 flex flex-wrap items-center gap-1.5">
                  <Badge tone="cyan">{p.id}</Badge>
                  <CiteBadge n={p.citations} />
                  {p.year && <Badge>{p.year}</Badge>}
                  {p.venue && <Badge>{p.venue.slice(0, 28)}</Badge>}
                </div>
              </div>
            </div>
          ))}
        </div>
        <button
          onClick={() => setSection("foundations")}
          className="mt-4 rounded-md border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-[11px] font-medium text-zinc-300"
        >
          See the full citation order
        </button>
      </Card>

      <Card>
        <SectionTitle>How to read this lab</SectionTitle>
        <ul className="mt-3 space-y-2 text-[12px] leading-6 text-zinc-400">
          <li>
            <span className="text-zinc-200">Architecture</span> shows the structural stack
            a paper names — tokenizer through attention to the output head — with the
            sentence quoted for each element.
          </li>
          <li>
            <span className="text-zinc-200">Modification</span> shows the transformation a
            paper applies: quantize, prune, distill, approximate.
          </li>
          <li>
            <span className="text-zinc-200">Audit</span> checks the whole corpus for the
            failure modes that would make the lab quietly wrong: missing PDFs, empty
            quotes, duplicate records.
          </li>
        </ul>
      </Card>
    </div>
  );
}