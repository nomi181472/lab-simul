"use client";

import { useMemo } from "react";
import {
  Card,
  SectionTitle,
  Badge,
  Quote,
  CiteBadge,
  TfPaperLink,
} from "@/labs/llms/ui";
import { PAPERS } from "@/labs/llms/data/papers";

/**
 * The scaling path: how a pretrained base model becomes something deployable.
 *
 * Each stage names the papers in the corpus that work at that stage. A stage with
 * no papers is shown as empty rather than skipped, because that is itself the
 * finding: the corpus is dense at pretraining and thinning out at the edges.
 */
const STAGES: { id: string; label: string; blurb: string; test: RegExp }[] = [
  {
    id: "pretrain",
    label: "Pretrain",
    blurb: "The base model exists, and the scale is what it is.",
    test: /pretrain|pre-train|scaling law|continued? pretrain/i,
  },
  {
    id: "adapt",
    label: "Adapt",
    blurb: "Teach it a task or a persona without retraining from scratch.",
    test: /fine-?tun|instruction tuning|prompt tuning|LoRA|adapter|RLHF|human feedback|PEFT|prefix/i,
  },
  {
    id: "extend",
    label: "Extend",
    blurb: "Give it more context, more modalities, or more capability at inference time.",
    test: /retrieval|RAG|augmented|long-?context|context window|multimodal|tool|agent|reasoning|chain-of-thought/i,
  },
  {
    id: "compress",
    label: "Compress",
    blurb: "Make it smaller: quantize, prune, distill, search for a better small model.",
    test: /quantiz|quantis|prun|sparsif|distill|compress|low-?rank|architecture search|edge|on-device/i,
  },
  {
    id: "serve",
    label: "Serve",
    blurb: "Run it fast enough that the latency is acceptable.",
    test: /serving|inference|latency|throughput|KV-?cache|speculative|flash|batching|parallelism|deployment/i,
  },
];

export function PathView() {
  const stages = useMemo(
    () =>
      STAGES.map((s) => {
        const papers = PAPERS.filter(
          (p) => s.test.test(p.title) || s.test.test(p.abstractNote ?? ""),
        ).sort((a, b) => b.citations - a.citations);
        return { ...s, papers };
      }),
    [],
  );

  const max = Math.max(...stages.map((s) => s.papers.length), 1);

  return (
    <div className="space-y-6">
      <SectionTitle>The scaling path</SectionTitle>

      <Card tone="accent">
        <p className="text-[13px] leading-6 text-zinc-400">
          This is the arc the corpus actually covers, in the order the work happens. The
          bar is paper count, and the drop-off is the interesting part: the literature
          thins as you move right, because making a model small enough to ship is harder
          than making a model large.
        </p>
      </Card>

      <Card>
        <div className="space-y-3">
          {stages.map((s, i) => (
            <div key={s.id} className="flex items-center gap-3">
              <span className="w-20 shrink-0 text-right font-mono text-[10px] text-zinc-600">
                {i + 1}. {s.label}
              </span>
              <div className="h-4 flex-1 overflow-hidden rounded bg-zinc-800/60">
                <div
                  className="h-full rounded bg-cyan-700/60"
                  style={{ width: `${(s.papers.length / max) * 100}%` }}
                />
              </div>
              <span className="w-16 shrink-0 text-right font-mono text-[11px] text-zinc-400">
                {s.papers.length}
              </span>
            </div>
          ))}
        </div>
      </Card>

      {stages.map((s, i) => (
        <Card key={s.id}>
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-[10px] text-zinc-600">stage {i + 1}</span>
            <SectionTitle>{s.label}</SectionTitle>
            <span className="ml-auto font-mono text-[10px] text-zinc-600">
              {s.papers.length} papers
            </span>
          </div>
          <p className="mt-1.5 text-[12px] leading-5 text-zinc-500">{s.blurb}</p>
          <div className="mt-3 space-y-2">
            {s.papers.slice(0, 8).map((p) => (
              <div
                key={p.id}
                className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-2.5"
              >
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-1.5">
                      <Badge tone="cyan">{p.id}</Badge>
                      <CiteBadge n={p.citations} />
                      {p.year && <Badge>{p.year}</Badge>}
                    </div>
                    <p className="mt-1.5 text-[12px] leading-5 text-zinc-200">{p.title}</p>
                  </div>
                  <TfPaperLink id={p.id} link short />
                </div>
                {p.changes[0] && <Quote>{p.changes[0].quote}</Quote>}
              </div>
            ))}
            {s.papers.length === 0 && (
              <p className="font-mono text-[10px] text-zinc-600">
                no paper in the corpus matches this stage
              </p>
            )}
          </div>
          {s.papers.length > 8 && (
            <p className="mt-2 font-mono text-[10px] text-zinc-600">
              showing the 8 most cited of {s.papers.length}
            </p>
          )}
        </Card>
      ))}
    </div>
  );
}