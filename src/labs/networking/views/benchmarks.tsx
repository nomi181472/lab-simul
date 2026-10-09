"use client";
import { Card, SectionTitle, Badge, BarRow } from "@/labs/networking/ui";
import { ANALYSIS } from "@/labs/networking/data/analysis";

export function BenchmarksView() {
  const b = ANALYSIS.benchmarks;
  const maxDs = Math.max(1, ...b.datasets.map((d) => d.useCount));
  const maxAl = Math.max(1, ...b.algorithms.map((a) => a.useCount));
  const maxMe = Math.max(1, ...b.metrics.map((m) => m.useCount));
  const periods = ["11-14", "15-18", "19-22", "23-26"];

  return (
    <div className="space-y-6">
      <SectionTitle>Benchmark &amp; metric maps</SectionTitle>

      <Card tone="accent">
        <p className="text-[13px] leading-6 text-zinc-400">
          What the corpus actually measures against: datasets, metrics and algorithms
          counted by the papers that name them. Counts come from keyword detection over
          each paper&apos;s retrieved text — a paper that never names CIFAR does not
          contribute to it.
        </p>
      </Card>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card>
          <p className="mb-3 font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            datasets / testbeds
          </p>
          <div className="space-y-1">
            {b.datasets.map((d) => (
              <BarRow
                key={d.name}
                label={d.name}
                value={d.useCount}
                max={maxDs}
                tone="cyan"
              />
            ))}
          </div>
          {b.datasets.length === 0 && (
            <p className="text-[11px] text-zinc-600">no dataset named in retrieved text</p>
          )}
        </Card>

        <Card>
          <p className="mb-3 font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            metrics reported
          </p>
          <div className="space-y-1">
            {b.metrics.map((m) => (
              <BarRow key={m.name} label={m.name} value={m.useCount} max={maxMe} tone="amber" />
            ))}
          </div>
        </Card>

        <Card>
          <p className="mb-3 font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            algorithms / models
          </p>
          <div className="space-y-1">
            {b.algorithms.map((a) => (
              <BarRow key={a.name} label={a.name} value={a.useCount} max={maxAl} tone="violet" />
            ))}
          </div>
        </Card>
      </div>

      <Card>
        <p className="mb-3 font-mono text-[10px] uppercase tracking-wide text-zinc-500">
          metric evolution (papers per era)
        </p>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-[11px]">
            <thead>
              <tr>
                <th className="border-b border-zinc-800 px-2 py-1.5 text-left font-mono text-[10px] font-normal text-zinc-500">
                  metric
                </th>
                {periods.map((p) => (
                  <th
                    key={p}
                    className="border-b border-zinc-800 px-2 py-1.5 text-right font-mono text-[10px] font-normal text-zinc-500"
                  >
                    {p}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {b.metricEvolution.map((m) => {
                const max = Math.max(1, ...m.cells);
                return (
                  <tr key={m.name}>
                    <td className="border-b border-zinc-900 px-2 py-1.5 text-zinc-300">
                      {m.name}
                    </td>
                    {m.cells.map((n, i) => (
                      <td
                        key={i}
                        className="border-b border-zinc-900 px-2 py-1.5 text-right font-mono"
                        style={{
                          background: n
                            ? `rgba(180, 120, 20, ${(0.12 + (n / max) * 0.7).toFixed(2)})`
                            : "transparent",
                          color: n ? (n / max > 0.55 ? "#fef3c7" : "#a1a1aa") : "#3f3f46",
                        }}
                      >
                        {n || "·"}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card tone="ok">
          <p className="mb-2 font-mono text-[10px] uppercase tracking-wide text-emerald-400">
            best-known approaches (mean citations, ≥3 papers)
          </p>
          <div className="space-y-1">
            {b.bestKnown.map((a) => (
              <div key={a.name} className="flex items-baseline justify-between gap-2">
                <span className="truncate text-[11px] text-zinc-300">{a.name}</span>
                <span className="shrink-0 font-mono text-[10px] text-zinc-500">
                  {a.meanCitations} mean · {a.papers} papers
                </span>
              </div>
            ))}
          </div>
          <p className="mt-2 font-mono text-[9px] text-zinc-600">
            citation mean is a field-level proxy, not a head-to-head benchmark result
          </p>
        </Card>

        <Card tone="warn">
          <p className="mb-2 font-mono text-[10px] uppercase tracking-wide text-amber-400">
            benchmark gaps
          </p>
          <div className="space-y-1">
            {b.gaps.map((g) => (
              <div key={g.name}>
                <Badge tone="amber">{g.name}</Badge>
                <p className="mt-0.5 text-[11px] text-zinc-500">{g.note}</p>
              </div>
            ))}
            {b.gaps.length === 0 && (
              <p className="text-[11px] text-zinc-600">no fading metrics detected</p>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
