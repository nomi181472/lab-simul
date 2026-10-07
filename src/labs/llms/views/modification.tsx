"use client";

import { useMemo, useState } from "react";
import {
  Card,
  SectionTitle,
  Badge,
  Quote,
  Unstated,
  ChangeChip,
  CiteBadge,
  TfPaperLink,
} from "@/labs/llms/ui";
import { PAPERS } from "@/labs/llms/data/papers";
import { SimRunner } from "@/labs/llms/sims";
import { useTfLab } from "@/labs/llms/context";
import {
  ALL_CHANGE_OPERATORS,
  CHANGE_COLOR,
  CHANGE_LABEL,
  type ChangeOperator,
} from "@/labs/llms/data/vocab";

/**
 * Modification, rendered as a pipeline.
 *
 * The order here is the order the operations are actually applied to a model:
 * first something is removed (prune / reduce), then whatever survived is
 * approximated (quantize / low-rank), and only then is a replacement trained
 * (distill / finetune). Caching and speculative decoding sit after training
 * because they change no weights at all.
 */
const PIPELINE: { stage: string; ops: ChangeOperator[]; note: string }[] = [
  {
    stage: "Reduce",
    ops: ["pruning", "layer-reduction", "token-reduction", "weight-sharing"],
    note: "Take structure away before approximating anything.",
  },
  {
    stage: "Approximate",
    ops: [
      "quantization",
      "low-rank-adaptation",
      "attention-approximation",
      "mixed-precision",
      "calibration",
    ],
    note: "Spend less per parameter on what remains.",
  },
  {
    stage: "Replace",
    ops: ["distillation", "architecture-search"],
    note: "Train a smaller or better model to take the original's job.",
  },
  {
    stage: "Serve",
    ops: ["caching", "speculative-decoding"],
    note: "Weights unchanged; only the inference loop is rewritten.",
  },
];

export function ModificationView() {
  const [selected, setSelected] = useState<ChangeOperator | null>(null);
  const { setSection } = useTfLab();

  const counts = useMemo(() => {
    const m: Record<string, number> = {};
    for (const p of PAPERS)
      for (const c of p.changes) m[c.operator] = (m[c.operator] ?? 0) + 1;
    return m;
  }, []);

  const stageOf = useMemo(() => {
    const m = new Map<ChangeOperator, { stage: string; note: string }>();
    for (const s of PIPELINE) for (const op of s.ops) m.set(op, { stage: s.stage, note: s.note });
    return m;
  }, []);

  const papersFor = (op: ChangeOperator) =>
    PAPERS.filter((p) => p.changes.some((c) => c.operator === op)).sort(
      (a, b) => b.citations - a.citations,
    );

  const active = selected ? papersFor(selected) : [];
  const maxStage = Math.max(...PIPELINE.map((s) => Math.max(...s.ops.map((o) => counts[o] ?? 0))), 1);

  return (
    <div className="space-y-6">
      <SectionTitle>Modification</SectionTitle>

      <Card tone="accent">
        <p className="text-[13px] leading-6 text-zinc-400">
          A pretrained model is not redesigned, it is <em>edited</em>. These are the edits
          the corpus names, grouped by when they apply. The count is how many papers name
          the operation in their own text, not how many times the word appears.
        </p>
      </Card>

      <div className="space-y-3">
        {PIPELINE.map((stage, i) => (
          <Card key={stage.stage}>
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <div className="flex items-baseline gap-2">
                <span className="font-mono text-[10px] text-zinc-600">step {i + 1}</span>
                <span className="text-sm font-semibold text-zinc-100">{stage.stage}</span>
              </div>
              <span className="font-mono text-[10px] text-zinc-600">{stage.note}</span>
            </div>
            <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {stage.ops.map((op) => {
                const n = counts[op] ?? 0;
                const on = selected === op;
                return (
                  <button
                    key={op}
                    onClick={() => setSelected(on ? null : op)}
                    className={`rounded-lg border px-2.5 py-2 text-left transition-colors ${
                      on
                        ? "border-cyan-600 bg-cyan-950/40"
                        : "border-zinc-800 bg-zinc-900/50 hover:border-zinc-700"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="flex items-center gap-1.5">
                        <span
                          aria-hidden
                          className="inline-block h-2.5 w-2.5 rounded-sm"
                          style={{ background: CHANGE_COLOR[op] }}
                        />
                        <span className="font-mono text-[11px] text-zinc-200">
                          {CHANGE_LABEL[op]}
                        </span>
                      </span>
                      <span className="font-mono text-[11px] text-zinc-400">{n}</span>
                    </div>
                    <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-zinc-800">
                      <div
                        className="h-full rounded-full"
                        style={{
                          background: CHANGE_COLOR[op],
                          width: `${Math.round((n / maxStage) * 100)}%`,
                        }}
                      />
                    </div>
                  </button>
                );
              })}
            </div>
          </Card>
        ))}
      </div>

      {ALL_CHANGE_OPERATORS.filter((o) => !stageOf.has(o)).length > 0 && (
        <Card tone="warn">
          <p className="font-mono text-[10px] text-amber-400">
            declared operators outside the pipeline:{" "}
            {ALL_CHANGE_OPERATORS.filter((o) => !stageOf.has(o)).join(", ")}
          </p>
        </Card>
      )}

      {selected && (
        <Card tone="accent">
          <div className="flex items-center justify-between gap-2">
            <SectionTitle>{CHANGE_LABEL[selected]}</SectionTitle>
            <span className="font-mono text-[10px] text-zinc-500">
              {active.length} papers
            </span>
          </div>
          <div className="mt-3 space-y-2">
            {active.slice(0, 25).map((p) => {
              const hit = p.changes.find((c) => c.operator === selected);
              return (
                <div key={p.id} className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-2.5">
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-1.5">
                        <Badge tone="cyan">{p.id}</Badge>
                        <CiteBadge n={p.citations} />
                        {p.year && <Badge>{p.year}</Badge>}
                        <ChangeChip operator={selected} />
                      </div>
                      <p className="mt-1.5 text-[12px] leading-5 text-zinc-200">{p.title}</p>
                    </div>
                    <TfPaperLink id={p.id} link short />
                  </div>
                  {hit && <Quote>{hit.quote}</Quote>}
                </div>
              );
            })}
            {active.length === 0 && <Unstated what={CHANGE_LABEL[selected]} />}
          </div>
          {active.length > 25 && (
            <p className="mt-2 font-mono text-[10px] text-zinc-600">
              showing 25 of {active.length}
            </p>
          )}
          <button
            onClick={() => setSection("papers")}
            className="mt-3 rounded-md border border-zinc-700 bg-zinc-900 px-2.5 py-1 text-[11px] text-zinc-400"
          >
            open in papers
          </button>
        </Card>
      )}

      <SimRunner />
    </div>
  );
}