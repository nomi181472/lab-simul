"use client";

import { PAPERS, YEAR_RANGE } from "@/labs/object-detection/data/papers";
import { countPapersByYear } from "@/labs/object-detection/data/audit";
import { SECTIONS } from "@/labs/object-detection/data/types";
import { INDUSTRY_LOOP } from "@/labs/object-detection/data/applications";
import { useLab } from "@/labs/object-detection/context";
import { Card, Badge } from "@/labs/object-detection/ui";

export function OverviewView() {
  const { setSection } = useLab();
  const byYear = countPapersByYear();

  const loopSteps = INDUSTRY_LOOP;

  return (
    <div className="space-y-8">
      {/* hero */}
      <section className="rounded-2xl border border-emerald-800/40 bg-gradient-to-b from-emerald-950/30 to-zinc-950 p-8">
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone="emerald">evidence-grounded</Badge>
          <Badge tone="zinc">{PAPERS.length} papers</Badge>
          <Badge tone="zinc">{YEAR_RANGE.min}–{YEAR_RANGE.max}</Badge>
          <Badge tone="zinc">no hallucination policy</Badge>
        </div>
        <h1 className="mt-4 max-w-2xl text-3xl font-semibold leading-tight tracking-tight text-zinc-50">
          Object Detection Research Lab
        </h1>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-zinc-400">
          A single-page interactive lab over the 48-paper corpus. Every claim is
          traced to a paper (<span className="font-mono text-emerald-300">P001…P048</span>);
          every math object is executable in the Mathematics Lab; every section is
          one year of field state. Claims that are not in the corpus are flagged{" "}
          <span className="font-mono text-rose-300">INSUFFICIENT EVIDENCE</span>.
        </p>
        <div className="mt-6 flex flex-wrap gap-2">
          <button
            onClick={() => setSection("math")}
            className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-zinc-950 hover:bg-emerald-500"
          >
            Open Mathematics Lab
          </button>
          <button
            onClick={() => setSection("papers")}
            className="rounded-lg border border-zinc-700 px-4 py-2 text-sm font-medium text-zinc-200 hover:bg-zinc-900"
          >
            Browse 48 papers
          </button>
        </div>
      </section>

      {/* corpus stats */}
      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card>
          <div className="font-mono text-3xl text-emerald-300">{PAPERS.length}</div>
          <div className="mt-1 text-xs text-zinc-500">corpus papers, 49 PDFs</div>
        </Card>
        <Card>
          <div className="font-mono text-3xl text-emerald-300">{YEAR_RANGE.max - YEAR_RANGE.min + 1}</div>
          <div className="mt-1 text-xs text-zinc-500">years of evolution (2015–2026)</div>
        </Card>
        <Card>
          <div className="font-mono text-3xl text-emerald-300">
            {Object.values(byYear).reduce((a, b) => a + b, 0) > 0 ? "12" : "0"}
          </div>
          <div className="mt-1 text-xs text-zinc-500">problem lifecycles traced</div>
        </Card>
        <Card>
          <div className="font-mono text-3xl text-emerald-300">13</div>
          <div className="mt-1 text-xs text-zinc-500">executable simulators</div>
        </Card>
      </section>

      {/* year profile strip */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-zinc-500">
          Corpus by year
        </h2>
        <div className="flex h-28 items-end gap-1 rounded-xl border border-zinc-900 bg-zinc-900/40 p-3">
          {Object.entries(byYear)
            .sort((a, b) => Number(a[0]) - Number(b[0]))
            .map(([y, n]) => (
              <div key={y} className="flex flex-1 flex-col items-center justify-end gap-1">
                <span className="font-mono text-[9px] text-zinc-600">{n}</span>
                <div
                  className="w-full rounded-t bg-emerald-800/70"
                  style={{ height: `${(n / Math.max(...Object.values(byYear))) * 80}px` }}
                />
                <span className="font-mono text-[9px] text-zinc-600">{y}</span>
              </div>
            ))}
        </div>
      </section>

      {/* research <-> application feedback loop */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-zinc-500">
          The research ↔ application closed loop
        </h2>
        <Card tone="accent">
          <div className="flex flex-col items-stretch gap-2 md:flex-row md:items-center">
            {loopSteps.map((s, i) => (
              <div key={s.label} className="flex flex-1 flex-col items-center gap-2">
                <div className="w-full rounded-lg border border-emerald-800/50 bg-emerald-950/30 px-3 py-2 text-center">
                  <div className="font-mono text-[10px] uppercase tracking-wide text-emerald-400">
                    {s.label}
                  </div>
                  <p className="mt-1 text-[11px] leading-4 text-zinc-400">{s.detail}</p>
                </div>
                {i < loopSteps.length - 1 && (
                  <span className="font-mono text-xs text-emerald-600">→</span>
                )}
              </div>
            ))}
          </div>
          <p className="mt-3 text-[11px] text-zinc-500">
            Every deployed detector founds the next research problem: latency sources → efficiency,
            NMS failure → end-to-end, occlusion → robust matching, unseen classes → open-vocabulary.
          </p>
        </Card>
      </section>

      {/* section index */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-zinc-500">
          Inside the lab
        </h2>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {SECTIONS.filter((s) => s.id !== "overview").map((s) => (
            <button
              key={s.id}
              onClick={() => setSection(s.id)}
              className="group rounded-xl border border-zinc-800 bg-zinc-900/40 p-4 text-left transition-colors hover:border-emerald-700/60 hover:bg-emerald-950/20"
            >
              <div className="font-mono text-[10px] uppercase tracking-wide text-emerald-400">
                {s.id}
              </div>
              <div className="mt-1 text-sm font-semibold text-zinc-100 group-hover:text-emerald-200">
                {s.label}
              </div>
              <div className="mt-1 text-xs text-zinc-500">{s.blurb}</div>
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}