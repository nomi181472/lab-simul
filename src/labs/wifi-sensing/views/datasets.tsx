"use client";

import { useMemo } from "react";
import { DATASETS } from "@/labs/wifi-sensing/data/datasets";
import { PAPERS } from "@/labs/wifi-sensing/data/papers";
import { MODALITY_LABEL } from "@/labs/wifi-sensing/data/types";
import { Badge, Card, SectionTitle, WifiPaperLink } from "@/labs/wifi-sensing/ui";

/* Dataset records are never authored: each one is emitted only when a paper
 * announces a dataset in its own text, and the description is that paper's own
 * sentence, quoted. The hard part of a WiFi dataset is ground truth, so the
 * ground-truth column is the thing to read first. */

export function DatasetsView() {
  const gtCounts = useMemo(() => {
    const m = new Map<string, number>();
    for (const d of DATASETS) m.set(d.groundTruth, (m.get(d.groundTruth) ?? 0) + 1);
    return [...m.entries()].sort((a, b) => b[1] - a[1]);
  }, []);

  const papersWithDataset = PAPERS.filter((p) => p.datasets.length > 0).length;

  return (
    <div className="space-y-5">
      <SectionTitle>Datasets</SectionTitle>
      <p className="max-w-3xl text-[12px] leading-6 text-zinc-500">
        {DATASETS.length} papers in the corpus announce a dataset in their own text;{" "}
        {papersWithDataset} records are linked to one. Descriptions are quoted, not written — the
        sentence below each title is that paper&apos;s own announcement.
      </p>

      <Card tone="warn">
        <div className="font-mono text-[10px] uppercase tracking-wide text-amber-400">
          ground truth is the hard part
        </div>
        <p className="mt-1 text-[12px] leading-6 text-zinc-300">
          In vision, a labelled image disturbs nothing. In WiFi sensing, the instrument used to
          label the data — a camera, a depth sensor, a wearable strap — changes the room&apos;s
          radio environment and the person&apos;s behaviour while the measurement is being taken. A
          dataset is therefore partly a statement about how it was labelled. Counts across the
          corpus:
        </p>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {gtCounts.map(([k, n]) => (
            <Badge key={k} tone={k === "manual-annotation" ? "amber" : "violet"}>
              {k} {n}
            </Badge>
          ))}
        </div>
      </Card>

      <div className="space-y-2">
        {DATASETS.map((d) => {
          const src = PAPERS.filter((p) => p.id === d.paperIds[0])[0];
          return (
            <Card key={d.id}>
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone="cyan">{d.id}</Badge>
                {d.year ? <span className="font-mono text-[10px] text-zinc-500">{d.year}</span> : null}
                <Badge tone="violet">
                  {MODALITY_LABEL[d.domain as keyof typeof MODALITY_LABEL] ?? d.domain}
                </Badge>
                <Badge tone="zinc">GT: {d.groundTruth}</Badge>
                {src ? <WifiPaperLink id={src.id} link /> : null}
              </div>
              <h3 className="mt-1.5 text-[13px] font-semibold leading-5 text-zinc-100">{d.name}</h3>
              <div className="mt-1.5 rounded-lg border border-zinc-800 bg-black/20 p-2.5">
                <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-500">
                  quoted from the paper
                </div>
                <p className="mt-0.5 text-[11px] leading-5 text-zinc-300">{d.purpose}</p>
              </div>
              {src ? (
                <div className="mt-2 flex flex-wrap items-center gap-2 text-[10px] text-zinc-600">
                  <span className="font-mono">{src.chain.hardware}</span>
                  <span>·</span>
                  <span>{src.chain.model}</span>
                  {src.pages ? (
                    <>
                      <span>·</span>
                      <span className="font-mono">{src.pages}p</span>
                    </>
                  ) : null}
                </div>
              ) : null}
            </Card>
          );
        })}
        {DATASETS.length === 0 ? (
          <Card>
            <p className="text-[12px] text-zinc-500">
              No paper in the corpus announced a dataset in a form this extractor could detect.
            </p>
          </Card>
        ) : null}
      </div>
    </div>
  );
}