"use client";

import { DATASETS } from "@/labs/object-tracking/data/datasets";
import { PAPER_BY_ID } from "@/labs/object-tracking/data/papers";
import { Card, Badge } from "@/labs/object-tracking/ui";

/* Dataset → papers → metrics → results. */

export function DatasetsView() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Datasets</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          Datasets define the problems: scale, density, modality. New benchmarks repeatedly
          redirect research (MOT16 → crowded MOT20 → motion-first DanceTrack → RGB-T LasHeR).
          Figures below are stated only where a corpus paper documents them.
        </p>
      </div>
      <div className="grid gap-3 lg:grid-cols-2">
        {DATASETS.map((d) => (
          <Card key={d.id} className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-semibold text-zinc-100 truncate">{d.name}</span>
              {d.year > 0 && <Badge tone="zinc">{d.year}</Badge>}
              <Badge tone="violet">{d.trackingType}</Badge>
            </div>
            <p className="mt-1 text-[12px] text-zinc-400">{d.purpose}</p>
            <p className="font-mono text-[10px] text-zinc-600 truncate">{d.domain}</p>
            <ul className="mt-2 list-disc space-y-0.5 pl-4 text-[12px] text-zinc-400">
              {d.characteristics.map((c, i) => <li key={i}>{c}</li>)}
            </ul>
            <div className="mt-2 flex flex-wrap gap-1">
              {d.metrics.map((m) => <Badge key={m} tone="sky">{m}</Badge>)}
            </div>
            <div className="mt-2 font-mono text-[11px] text-zinc-500 flex flex-wrap gap-1">
              papers: {d.paperIds.length ? d.paperIds.map((id) => (
                <span key={id} className="mr-1 rounded bg-sky-950 px-1 py-0.5 text-sky-300 whitespace-nowrap" title={PAPER_BY_ID[id]?.title ?? id}>{id}</span>
              )) : <span className="text-zinc-600">no structured paper yet</span>}
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
