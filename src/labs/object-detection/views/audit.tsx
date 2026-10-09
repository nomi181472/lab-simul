"use client";

import { useMemo } from "react";
import { runAudit, countPapersByYear } from "@/labs/object-detection/data/audit";
import { PAPER_COUNT, PAPERS_WITH_DOI } from "@/labs/object-detection/data/papers";
import { BENCHMARKS } from "@/labs/object-detection/data/benchmarks";
import { RELATIONS } from "@/labs/object-detection/data/relations";
import { FUTURE_DIRECTIONS } from "@/labs/object-detection/data/directions";
import { PROBLEM_LIFECYCLES } from "@/labs/object-detection/data/problems";
import { Card, Badge } from "@/labs/object-detection/ui";

const CHAIN = [
  {
    step: "01",
    title: "pdftotext extraction",
    detail:
      "Every harvest quote is verified verbatim against the extracted PDF text by scripts/od_harvest_check.py.",
    tag: "121/121 pass",
  },
  {
    step: "02",
    title: "harvest JSON records",
    detail:
      "The per-paper harvest JSON is the source of truth: schema-checked records, each pointing back at its verified PDF text.",
    tag: "source of truth",
  },
  {
    step: "03",
    title: "generated TS data",
    detail:
      "Every quote in the data files must be a substring of its paper's harvest record — enforced by __tests__/audit.test.ts.",
    tag: "quote substring check",
  },
] as const;


function ChainArrow() {
  return (
    <div className="flex items-center justify-center font-mono text-lg text-emerald-500 rotate-90 md:rotate-0">
      →
    </div>
  );
}

export function AuditView() {
  const checks = useMemo(() => runAudit(), []);
  const byYear = useMemo(() => countPapersByYear(), []);
  const passed = checks.filter((c) => c.pass).length;
  const problemsJudged = PROBLEM_LIFECYCLES.filter(
    (p) => p.currentStatus !== "uncertain",
  ).length;

  const counts: { value: number; label: string; sub: string }[] = [
    { value: BENCHMARKS.length, label: "benchmark observations", sub: "dataset × metric × year" },
    { value: RELATIONS.length, label: "paper relations", sub: "builds on / contrasts" },
    { value: FUTURE_DIRECTIONS.length, label: "future directions", sub: "stated vs realized" },
    {
      value: problemsJudged,
      label: "problems judged",
      sub: `of ${PROBLEM_LIFECYCLES.length} · non-uncertain`,
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Research audit</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          Machine checks over the 121-paper corpus: every paper processed, no duplicates,
          contiguous years, every reference resolves, and chronology is consistent — plus the
          chain of trust from PDF text to the TypeScript you are reading.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card tone={passed === checks.length ? "ok" : "warn"}>
          <div className="font-mono text-3xl text-emerald-300">
            {passed}/{checks.length}
          </div>
          <div className="mt-1 text-xs text-zinc-500">audit checks passing</div>
        </Card>
        <Card>
          <div className="font-mono text-3xl text-emerald-300">{PAPER_COUNT}</div>
          <div className="mt-1 text-xs text-zinc-500">papers in corpus</div>
          <div className="mt-0.5 font-mono text-[10px] text-zinc-600">quote-verified PDFs</div>
        </Card>
        <Card>
          <div className="font-mono text-3xl text-emerald-300">{PAPERS_WITH_DOI}</div>
          <div className="mt-1 text-xs text-zinc-500">papers with DOI</div>
          <div className="mt-0.5 font-mono text-[10px] text-zinc-600">
            {PAPER_COUNT - PAPERS_WITH_DOI} without
          </div>
        </Card>
        <Card>
          <div className="font-mono text-3xl text-emerald-300">
            {Object.keys(byYear).length}
          </div>
          <div className="mt-1 text-xs text-zinc-500">distinct years represented</div>
        </Card>
      </div>

      <section>
        <div className="mb-2 flex flex-wrap items-baseline gap-2">
          <h2 className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            chain of trust
          </h2>
          <span className="font-mono text-[10px] text-zinc-600">
            PDF text → harvest JSON → generated TS data
          </span>
        </div>
        <div className="flex flex-col gap-2 md:flex-row md:items-stretch">
          {CHAIN.map((c, i) => (
            <div key={c.step} className="flex flex-1 flex-col md:flex-row md:items-stretch">
              <div className="flex flex-1 flex-col rounded-xl border border-emerald-800/50 bg-emerald-950/10 p-4">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-mono text-[10px] text-emerald-500">{c.step}</span>
                  <span className="text-sm font-semibold text-zinc-100">{c.title}</span>
                </div>
                <p className="mt-1.5 text-[12px] leading-5 text-zinc-400">{c.detail}</p>
                <div className="mt-auto pt-2">
                  <Badge tone="emerald">{c.tag}</Badge>
                </div>
              </div>
              {i < CHAIN.length - 1 && <ChainArrow />}
            </div>
          ))}
        </div>
        <p className="mt-2 font-mono text-[10px] leading-4 text-zinc-600">
          break any link and the downstream claim is withheld — no quote reaches a data file
          without passing every step for its paper.
        </p>
      </section>

      <section>
        <h2 className="mb-2 font-mono text-[10px] uppercase tracking-wide text-zinc-500">
          corpus records generated
        </h2>
        <Card>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {counts.map((c) => (
              <div key={c.label}>
                <div className="font-mono text-2xl text-emerald-300">{c.value}</div>
                <div className="mt-0.5 text-xs text-zinc-500">{c.label}</div>
                <div className="mt-0.5 font-mono text-[10px] text-zinc-600">{c.sub}</div>
              </div>
            ))}
          </div>
        </Card>
      </section>

      <section>
        <h2 className="mb-2 font-mono text-[10px] uppercase tracking-wide text-zinc-500">
          integrity checks
        </h2>
        <div className="grid gap-3 lg:grid-cols-2">
          {checks.map((c) => (
            <div
              key={c.id}
              className={`rounded-xl border p-4 ${
                c.pass
                  ? "border-emerald-800/50 bg-emerald-950/10"
                  : "border-rose-800/60 bg-rose-950/15"
              }`}
            >
              <div className="flex items-center gap-2">
                <span
                  className={`font-mono text-sm ${c.pass ? "text-emerald-400" : "text-rose-400"}`}
                >
                  {c.pass ? "✓" : "✗"}
                </span>
                <span className="text-sm font-medium text-zinc-200">{c.label}</span>
              </div>
              <p className="mt-1.5 font-mono text-[10px] leading-4 text-zinc-500">{c.detail}</p>
            </div>
          ))}
        </div>
      </section>

      <Card>
        <div className="flex items-center gap-2">
          <Badge tone="emerald">policy</Badge>
          <p className="text-[11px] leading-5 text-zinc-400">
            Every assertion in this lab is either traced to a corpus paper, flagged as
            cross-paper or lab synthesis, or withheld as{" "}
            <span className="font-mono text-rose-300">INSUFFICIENT EVIDENCE</span>. Novel
            equations, years, and claims are never invented to fill gaps.
          </p>
        </div>
      </Card>
    </div>
  );
}
