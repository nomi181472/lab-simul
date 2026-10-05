"use client";

import { useMemo, useState } from "react";
import { PAPERS } from "@/labs/neuroevolution/data/papers";
import { CONCEPTS, CONCEPT_BY_ID } from "@/labs/neuroevolution/data/concepts";
import { Card, SectionTitle, Badge, Formula, NeuroPaperLink } from "@/labs/neuroevolution/ui";
import { SIMULATORS } from "@/labs/neuroevolution/sims";

/* Publication years from the corpus, with the evolved phenotype beside each year. */

/* ------------------------------ timeline ------------------------------ */

export function TimelineView() {
  const byYear = useMemo(() => {
    const m = new Map<number, typeof PAPERS>();
    for (const p of PAPERS) {
      if (!p.year) continue;
      if (!m.has(p.year)) m.set(p.year, []);
      m.get(p.year)!.push(p);
    }
    return [...m.entries()].sort((a, b) => a[0] - b[0]);
  }, []);

  const maxInYear = Math.max(...byYear.map(([, ps]) => ps.length), 1);

  return (
    <div className="flex flex-col gap-6">
      <SectionTitle>Timeline</SectionTitle>
      <p className="max-w-3xl text-[13px] leading-6 text-zinc-400">
        Publication years from the corpus, with each paper&rsquo;s evolved
        phenotype beside it — so the shift in what gets searched for over time is
        visible: weight-only early, then topology, then behaviour.
      </p>

      <div className="flex flex-col gap-2">
        {byYear.map(([year, ps]) => (
          <div key={year} className="flex items-start gap-3">
            <div className="w-12 shrink-0 pt-1 text-right font-mono text-[11px] text-zinc-500">
              {year}
            </div>
            <div
              className="mt-1 h-2 shrink-0 rounded-full bg-rose-500/60"
              style={{ width: `${Math.max(8, (ps.length / maxInYear) * 160)}px` }}
              title={`${ps.length} papers`}
            />
            <div className="flex flex-1 flex-wrap gap-1.5">
              {ps.slice(0, 8).map((p) => (
                <span
                  key={p.id}
                  className="rounded border border-zinc-800 bg-zinc-900/60 px-1.5 py-0.5 font-mono text-[10px] text-zinc-400"
                  title={`${p.title} · ${p.architecture.phenotype}`}
                >
                  <NeuroPaperLink id={p.id} short link />
                </span>
              ))}
              {ps.length > 8 ? (
                <span className="font-mono text-[10px] text-zinc-600">
                  +{ps.length - 8} more
                </span>
              ) : null}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
