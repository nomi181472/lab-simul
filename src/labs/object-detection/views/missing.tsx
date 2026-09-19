"use client";

import { EXTERNAL_REFERENCES } from "@/labs/object-detection/data/papers";
import { Card, Badge, SectionTitle } from "@/labs/object-detection/ui";

export function MissingView() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">What&apos;s missing</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          Honest boundaries of the lab: ideas that appear in this analysis but are NOT part
          of the 48-paper corpus are labelled here so they are never mistaken for corpus fact.
        </p>
      </div>

      <Card tone="warn">
        <SectionTitle>External references cited by the corpus, not in the corpus</SectionTitle>
        <div className="mt-3 space-y-3">
          {Object.entries(EXTERNAL_REFERENCES).map(([name, note]) => (
            <div key={name} className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-3">
              <div className="flex items-center gap-2">
                <span className="font-mono text-sm text-amber-300">{name}</span>
                <Badge tone="amber">external</Badge>
              </div>
              <p className="mt-1 text-[12px] leading-5 text-zinc-400">{note}</p>
            </div>
          ))}
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card tone="default">
          <SectionTitle>What the corpus itself does not cover</SectionTitle>
          <ul className="mt-3 list-inside space-y-1.5">
            <li className="text-[12px] leading-5 text-zinc-400">
              The original DETR, RetinaNet, and two-stage lineage papers (Faster/Mask R-CNN) are absent as PDFs — referenced only.
            </li>
            <li className="text-[12px] leading-5 text-zinc-400">
              No segmentation, pose, or video-tube tasks are present as papers; tracking appears only via “Tracking-the-Unseen” (P047).
            </li>
            <li className="text-[12px] leading-5 text-zinc-400">
              Modern large-scale synthesis and scaled-annotation pipelines (SA-1B/SAM-style data) are not in the corpus.
            </li>
            <li className="text-[12px] leading-5 text-zinc-400">
              Deployment wilderness (TPU/quantization graphs, compiler passes, latency reports) is only indirectly evidenced.
            </li>
          </ul>
        </Card>
        <Card tone="default">
          <SectionTitle>Lab claims that must be read as synthesis</SectionTitle>
          <ul className="mt-3 list-inside space-y-1.5">
            <li className="text-[12px] leading-5 text-zinc-400">
              Oscillation patterns (“complexity ↔ efficiency”) are cross-paper syntheses, not single-paper facts.
            </li>
            <li className="text-[12px] leading-5 text-zinc-400">
              Year field-state labels (e.g. “declining ideas”) are aggregates over the year’s papers.
            </li>
            <li className="text-[12px] leading-5 text-zinc-400">
              Simulators execute textbook math (IoU, focal, attention, Hungarian) to illustrate claims the papers make.
            </li>
            <li className="text-[12px] leading-5 text-zinc-400">
              Any claim that cannot be traced to a corpus PDF is rendered as{" "}
              <span className="font-mono text-rose-300">INSUFFICIENT EVIDENCE</span> rather than asserted.
            </li>
          </ul>
        </Card>
      </div>
    </div>
  );
}