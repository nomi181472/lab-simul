"use client";

import { useMemo } from "react";
import { PAPERS } from "@/labs/neuroevolution/data/papers";
import { Card, SectionTitle } from "@/labs/neuroevolution/ui";

/* The lab's own honesty check: what the extractor actually found. */

export function AuditView() {
  const stats = useMemo(() => {
    const n = PAPERS.length;
    const withSummary = PAPERS.filter((p) => p.summary && !p.summary.startsWith("not stated")).length;
    const withLayers = PAPERS.filter((p) => p.architecture.layers.length > 0).length;
    const withChanges = PAPERS.filter((p) => p.method.evolution.changes.length > 0).length;
    const withFitness = PAPERS.filter((p) => !!p.method.evolution.fitness).length;
    const withMutation = PAPERS.filter((p) => p.method.evolution.mutation.length > 0).length;
    const withResults = PAPERS.filter((p) => p.results.length > 0).length;
    const withRelations = PAPERS.filter((p) => p.relations.length > 0).length;
    const totalRelations = PAPERS.reduce((a, p) => a + p.relations.length, 0);
    return { n, withSummary, withLayers, withChanges, withFitness, withMutation, withResults, withRelations, totalRelations };
  }, []);

  const pct = (a: number) => `${Math.round((100 * a) / Math.max(1, stats.n))}%`;

  const rows: [string, number][] = [
    ["summary", stats.withSummary],
    ["drawn architecture", stats.withLayers],
    ["locus-level modification", stats.withChanges],
    ["mutation operator named", stats.withMutation],
    ["fitness described", stats.withFitness],
    ["quoted results", stats.withResults],
    ["corpus relations", stats.withRelations],
  ];

  return (
    <div className="flex flex-col gap-5">
      <SectionTitle>Audit</SectionTitle>
      <p className="max-w-3xl text-[13px] leading-6 text-zinc-400">
        What the extractor actually found. These are the lab&rsquo;s own counts over
        the generated records, not claims made by any paper. A field that is low
        here means the corpus genuinely does not state that thing in its text.
      </p>
      <Card>
        <div className="flex flex-col gap-2">
          {rows.map(([label, n]) => (
            <div key={label} className="flex items-center gap-3">
              <span className="w-40 shrink-0 font-mono text-[10px] text-zinc-400">{label}</span>
              <div className="h-2 flex-1 overflow-hidden rounded-full bg-zinc-800">
                <div
                  className="h-full rounded-full bg-rose-500/70"
                  style={{ width: `${Math.max(1, (100 * n) / stats.n)}%` }}
                />
              </div>
              <span className="w-16 shrink-0 text-right font-mono text-[10px] text-zinc-500">
                {n}/{stats.n} · {pct(n)}
              </span>
            </div>
          ))}
        </div>
      </Card>
      <Card tone="warn">
        <div className="flex flex-col gap-2 text-[11px] leading-5 text-amber-300/80">
          <div className="font-mono text-[10px] uppercase tracking-widest">
            known limits of this extraction
          </div>
          <div>· Architecture layers are only drawn when the paper enumerates them in prose; many papers only state weights or a genome size.</div>
          <div>· Operator detection is keyword-based over the paper text, so a paper that uses an operator without naming it will not be counted.</div>
          <div>· Results are only quoted when the paper prints a number; no number is ever inferred.</div>
          <div>· Relation edges come from reference-block matching, so a citation without a matching title or author-year is not counted.</div>
          <div>· Locus attribution is inferred from the sentence that describes the locus; where the operator is unnamed, the record says so instead of guessing.</div>
        </div>
      </Card>
    </div>
  );
}
