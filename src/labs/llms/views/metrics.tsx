"use client";

import { Card, SectionTitle, Badge } from "@/labs/llms/ui";
import { PAPERS } from "@/labs/llms/data/papers";

const QUALITY = ["accuracy", "perplexity"] as const;
const COST = ["latency", "memory", "throughput", "energy", "compression", "size"] as const;

export function MetricsView() {
  const counts: Record<string, number> = {};
  for (const p of PAPERS)
    for (const o of p.objectives) counts[o.kind] = (counts[o.kind] ?? 0) + 1;

  const both = PAPERS.filter(
    (p) =>
      p.objectives.some((o) => QUALITY.includes(o.kind as never)) &&
      p.objectives.some((o) => COST.includes(o.kind as never)),
  ).length;
  const qualityOnly = PAPERS.filter(
    (p) =>
      p.objectives.some((o) => QUALITY.includes(o.kind as never)) &&
      !p.objectives.some((o) => COST.includes(o.kind as never)),
  ).length;
  const costOnly = PAPERS.filter(
    (p) =>
      !p.objectives.some((o) => QUALITY.includes(o.kind as never)) &&
      p.objectives.some((o) => COST.includes(o.kind as never)),
  ).length;
  const neither = PAPERS.length - both - qualityOnly - costOnly;

  return (
    <div className="space-y-6">
      <SectionTitle>Metrics</SectionTitle>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Card tone="accent">
          <p className="font-mono text-[10px] uppercase tracking-wide text-cyan-400">
            both axes
          </p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">{both}</p>
          <p className="mt-1 font-mono text-[10px] text-zinc-500">
            names a quality and a cost objective
          </p>
        </Card>
        <Card>
          <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            quality only
          </p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">{qualityOnly}</p>
          <p className="mt-1 font-mono text-[10px] text-zinc-500">
            accuracy or perplexity, no cost axis
          </p>
        </Card>
        <Card>
          <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            cost only
          </p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">{costOnly}</p>
          <p className="mt-1 font-mono text-[10px] text-zinc-500">
            latency, memory or energy, no quality axis
          </p>
        </Card>
        <Card>
          <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            neither
          </p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">{neither}</p>
          <p className="mt-1 font-mono text-[10px] text-zinc-500">no objective stated</p>
        </Card>
      </div>

      <Card>
        <SectionTitle>Objective counts</SectionTitle>
        <p className="mt-2 text-[13px] leading-6 text-zinc-400">
          Counted by how many papers name the objective in their own text. A paper can
          name several, which is why these sum to more than the corpus.
        </p>
        <div className="mt-4 space-y-2">
          {Object.entries(counts)
            .sort((a, b) => b[1] - a[1])
            .map(([kind, n]) => {
              const pct = Math.round((n / PAPERS.length) * 100);
              const isCost = (COST as readonly string[]).includes(kind);
              return (
                <div key={kind} className="flex items-center gap-2">
                  <span className="w-24 shrink-0 font-mono text-[10px] text-zinc-400">
                    {kind}
                  </span>
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-zinc-800">
                    <div
                      className={`h-full rounded-full ${isCost ? "bg-amber-600" : "bg-cyan-600"}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="w-10 shrink-0 text-right font-mono text-[10px] text-zinc-500">
                    {n}
                  </span>
                  <span className="w-12 shrink-0 text-right">
                    <Badge tone={isCost ? "amber" : "cyan"}>{isCost ? "cost" : "quality"}</Badge>
                  </span>
                </div>
              );
            })}
        </div>
      </Card>
    </div>
  );
}