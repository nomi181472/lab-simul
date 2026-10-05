"use client";

import { useMemo } from "react";
import { PAPERS } from "@/labs/neuroevolution/data/papers";
import { DATASETS } from "@/labs/neuroevolution/data/datasets";
import { Card, SectionTitle, Badge, NeuroPaperLink, Formula, Quote, Unstated } from "@/labs/neuroevolution/ui";

/* Benchmarks named by the corpus, with the papers that use each. */

/* Datasets, metrics and audit.
 *
 * Datasets and metrics come from the curated KB, with corpus linkage computed
 * from the records: which papers actually name each benchmark, and which report
 * each metric. The audit view is the lab's own honesty check — it states what is
 * missing rather than hiding it.
 */

export function DatasetsView() {
  const used = useMemo(() => {
    const m = new Map<string, number>();
    for (const p of PAPERS)
      for (const d of p.datasets) m.set(d, (m.get(d) ?? 0) + 1);
    return m;
  }, []);

  return (
    <div className="flex flex-col gap-5">
      <SectionTitle>Datasets & benchmarks</SectionTitle>
      <div className="grid gap-3 lg:grid-cols-2">
        {DATASETS.map((d) => (
          <Card key={d.id} className="flex flex-col gap-2">
            <div className="flex items-center justify-between gap-2">
              <span className="text-[13px] font-semibold text-zinc-100">{d.name}</span>
              <Badge tone="zinc">{d.year}</Badge>
            </div>
            <div className="text-[11px] leading-5 text-zinc-300">{d.purpose}</div>
            <div className="flex flex-wrap gap-1.5">
              <Badge tone="violet">{d.domain}</Badge>
              <Badge tone="cyan">{d.task}</Badge>
              <Badge tone={used.get(d.id) ? "emerald" : "zinc"}>
                {used.get(d.id) ?? 0} corpus papers
              </Badge>
            </div>
            {d.characteristics.length > 0 ? (
              <div className="flex flex-col gap-1">
                {d.characteristics.map((c, i) => (
                  <Quote key={i}>{c}</Quote>
                ))}
              </div>
            ) : (
              <Unstated what="quoted characteristics" />
            )}
            {d.paperIds.length > 0 ? (
              <div className="flex flex-wrap gap-1 border-t border-zinc-800 pt-2">
                {d.paperIds.slice(0, 10).map((id) => (
                  <NeuroPaperLink key={id} id={id} short link />
                ))}
                {d.paperIds.length > 10 ? (
                  <span className="font-mono text-[10px] text-zinc-600">
                    +{d.paperIds.length - 10}
                  </span>
                ) : null}
              </div>
            ) : null}
          </Card>
        ))}
      </div>
    </div>
  );
}
