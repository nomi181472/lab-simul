"use client";

import { useMemo } from "react";
import {
  PAPERS,
  PAPER_COUNT,
  YEAR_MIN,
  YEAR_MAX,
  TOTAL_CITATIONS,
  PAPERS_WITH_DOI,
} from "@/labs/object-detection/data/papers";
import { PROBLEM_LIFECYCLES } from "@/labs/object-detection/data/problems";
import { SOLUTIONS } from "@/labs/object-detection/data/solutions";
import { CLUSTERS } from "@/labs/object-detection/data/clusters";
import { YEARS } from "@/labs/object-detection/data/years";
import { BENCHMARK_BEST } from "@/labs/object-detection/data/benchmarks";
import { SECTIONS, type ProblemStatus } from "@/labs/object-detection/data/types";
import { useLab } from "@/labs/object-detection/context";
import { Card, Badge, StatusBadge, PaperLink, SectionTitle } from "@/labs/object-detection/ui";

const STATUSES: ProblemStatus[] = ["resolved", "reduced", "persistent", "transformed", "uncertain"];

function Stat({ value, label, sub }: { value: string | number; label: string; sub?: string }) {
  return (
    <Card>
      <div className="font-mono text-3xl text-emerald-300">{value}</div>
      <div className="mt-1 text-xs text-zinc-500">{label}</div>
      {sub && <div className="mt-0.5 font-mono text-[10px] text-zinc-600">{sub}</div>}
    </Card>
  );
}

export function OverviewView() {
  const { setYear, setSection } = useLab();

  const perYear = useMemo(
    () => YEARS.map((y) => ({ year: y, count: PAPERS.filter((p) => p.year === y).length })),
    [],
  );
  const maxYearCount = Math.max(1, ...perYear.map((d) => d.count));

  const statusCounts = useMemo(
    () =>
      STATUSES.map((s) => ({
        status: s,
        count: PROBLEM_LIFECYCLES.filter((p) => p.currentStatus === s).length,
      })),
    [],
  );

  const topClusters = useMemo(
    () =>
      [...CLUSTERS]
        .sort((a, b) => b.papers.length - a.papers.length)
        .slice(0, 8),
    [],
  );

  const cocoMap = BENCHMARK_BEST.find((b) => b.dataset === "COCO" && b.metric === "mAP");

  return (
    <div className="space-y-8">
      {/* hero */}
      <section className="rounded-2xl border border-emerald-800/40 bg-gradient-to-b from-emerald-950/30 to-zinc-950 p-8">
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone="emerald">evidence-grounded</Badge>
          <Badge tone="zinc">{PAPER_COUNT} papers</Badge>
          <Badge tone="zinc">
            {YEAR_MIN}–{YEAR_MAX}
          </Badge>
          <Badge tone="zinc">no hallucination policy</Badge>
        </div>
        <h1 className="mt-4 max-w-2xl text-3xl font-semibold leading-tight tracking-tight text-zinc-50">
          Object Detection Research Lab
        </h1>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-zinc-400">
          An interactive lab over {PAPER_COUNT} corpus papers from {YEAR_MIN}–{YEAR_MAX}. Every
          claim is traced to a paper (<span className="font-mono text-emerald-300">OD001…OD121</span>);
          problem lifecycles, clusters and year states are built from the corpus itself. Claims
          the corpus does not support are withheld as{" "}
          <span className="font-mono text-rose-300">INSUFFICIENT EVIDENCE</span>.
        </p>
      </section>

      {/* hero stats */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-zinc-500">
          Corpus at a glance
        </h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Stat value={PAPER_COUNT} label="corpus papers" sub="quote-verified PDFs" />
          <Stat
            value={`${YEAR_MIN}–${YEAR_MAX}`}
            label="years of evolution"
            sub={`${YEAR_MAX - YEAR_MIN + 1} year states`}
          />
          <Stat
            value={TOTAL_CITATIONS.toLocaleString()}
            label="total citations"
            sub="OpenAlex, resolved"
          />
          <Stat
            value={PAPERS_WITH_DOI}
            label="papers with DOI"
            sub={`${PAPER_COUNT - PAPERS_WITH_DOI} without`}
          />
          <Stat value={PROBLEM_LIFECYCLES.length} label="problem lifecycles traced" />
          <Stat value={SOLUTIONS.length} label="solution tags catalogued" />
          <Stat value={CLUSTERS.length} label="identical-problem clusters" />
          <Stat value={BENCHMARK_BEST.length} label="benchmark bests tracked" />
        </div>
      </section>

      {/* problem status distribution */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-zinc-500">
          Problem statuses
        </h2>
        <Card>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
            {statusCounts.map(({ status, count }) => (
              <div
                key={status}
                className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-3 text-center"
              >
                <StatusBadge status={status} />
                <div className="mt-2 font-mono text-2xl text-zinc-100">{count}</div>
                <div className="mt-0.5 font-mono text-[10px] text-zinc-600">
                  of {PROBLEM_LIFECYCLES.length} lifecycles
                </div>
              </div>
            ))}
          </div>
          <p className="mt-3 text-[11px] leading-5 text-zinc-500">
            Judgments come from the corpus review pass ({PROBLEM_LIFECYCLES.length} problems).
            Open the Problem Matrix for per-year evidence of each status.
          </p>
        </Card>
      </section>

      {/* top clusters */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-zinc-500">
          Top problem clusters
        </h2>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {topClusters.map((c) => (
            <button
              key={c.id}
              onClick={() => setSection("clusters")}
              className="group rounded-xl border border-zinc-800 bg-zinc-900/40 p-4 text-left transition-colors hover:border-emerald-700/60 hover:bg-emerald-950/20"
            >
              <div className="flex items-start justify-between gap-2">
                <span className="text-sm font-semibold text-zinc-100 group-hover:text-emerald-200">
                  {c.name}
                </span>
                <span className="font-mono text-lg text-emerald-300">{c.papers.length}</span>
              </div>
              <div className="mt-1 font-mono text-[10px] uppercase tracking-wide text-zinc-500">
                papers · {c.firstYear}–{c.lastYear}
              </div>
            </button>
          ))}
        </div>
      </section>

      {/* corpus over time */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-zinc-500">
          Corpus over time
        </h2>
        <Card>
          <div className="flex items-end gap-1.5">
            {perYear.map((d) => (
              <button
                key={d.year}
                onClick={() => {
                  setYear(d.year);
                  setSection("year");
                }}
                className="group flex flex-1 flex-col items-center justify-end gap-1"
              >
                <span className="font-mono text-[10px] text-zinc-500 group-hover:text-emerald-300">
                  {d.count}
                </span>
                <div
                  className="w-full rounded-t bg-emerald-800/70 transition-colors group-hover:bg-emerald-500"
                  style={{ height: `${Math.max(4, (d.count / maxYearCount) * 88)}px` }}
                />
                <span className="font-mono text-[9px] text-zinc-600 group-hover:text-zinc-300">
                  {d.year}
                </span>
              </button>
            ))}
          </div>
          <p className="mt-3 font-mono text-[10px] text-zinc-600">
            click a bar to open that year&apos;s field state
          </p>
        </Card>
      </section>

      {/* COCO mAP highlight */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-zinc-500">
          Benchmark highlight
        </h2>
        {cocoMap ? (
          <Card tone="accent">
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone="emerald">best COCO mAP</Badge>
              <Badge tone="zinc">{cocoMap.dataset}</Badge>
            </div>
            <div className="mt-3 flex flex-wrap items-end gap-6">
              <div>
                <div className="font-mono text-4xl text-emerald-300">{cocoMap.value}</div>
                <div className="mt-1 font-mono text-[10px] uppercase tracking-wide text-zinc-500">
                  mAP · corpus best
                </div>
              </div>
              <div>
                <div className="font-mono text-2xl text-zinc-100">{cocoMap.year}</div>
                <div className="mt-1 font-mono text-[10px] uppercase tracking-wide text-zinc-500">
                  reported in
                </div>
              </div>
              <div className="flex flex-col gap-1">
                <span className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
                  source paper
                </span>
                <PaperLink id={cocoMap.paperId} />
              </div>
            </div>
          </Card>
        ) : (
          <Card tone="warn">
            <Badge tone="amber">withheld</Badge>
            <p className="mt-2 text-[12px] leading-5 text-zinc-400">
              No corpus paper states a best COCO mAP figure; the number is not shown.
            </p>
          </Card>
        )}
      </section>

      {/* section index */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-zinc-500">
          Inside the lab
        </h2>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {SECTIONS.map((s) => (
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

      {/* evidence policy */}
      <section>
        <SectionTitle>Evidence policy</SectionTitle>
        <Card>
          <div className="flex flex-wrap items-start gap-2">
            <Badge tone="emerald">policy</Badge>
            <p className="text-[11px] leading-5 text-zinc-400">
              Every assertion in this lab is either traced to a corpus paper, marked cross-paper
              or lab synthesis, or withheld as{" "}
              <span className="font-mono text-rose-300">INSUFFICIENT EVIDENCE</span>. Years,
              figures and judgments are never invented to fill gaps; unsupported fields render a
              muted “—”.
            </p>
          </div>
        </Card>
      </section>
    </div>
  );
}
