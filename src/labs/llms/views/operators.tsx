"use client";

import { useMemo, useState } from "react";
import {
  Card,
  SectionTitle,
  Badge,
  Quote,
  Formula,
  CiteBadge,
  TfPaperLink,
} from "@/labs/llms/ui";
import { PAPERS } from "@/labs/llms/data/papers";
import {
  ALL_CHANGE_OPERATORS,
  CHANGE_COLOR,
  CHANGE_LABEL,
  type ChangeOperator,
} from "@/labs/llms/data/vocab";

/**
 * Operator reference: what each transformation does to a weight, plus the papers
 * that use it.
 *
 * The formulas are the standard definitions, written in plain unicode because
 * this project has no KaTeX. They are here to explain the mechanism, not to
 * claim anything about what an individual paper does -- the quote under each
 * formula is the paper's own statement.
 */
const REFERENCE: Record<
  ChangeOperator,
  { what: string; effect: string; formula: string }
> = {
  quantization: {
    what: "Replace each weight with a value from a low-precision grid.",
    effect:
      "Memory and bandwidth fall roughly in proportion to bit width. Accuracy loss depends on whether per-channel scales are calibrated.",
    formula: "ŵ = round(w / s) · s      w ∈ ℝ → ŵ ∈ {0, 1, …, 2ᵇ−1}",
  },
  pruning: {
    what: "Set a fraction of weights to exactly zero.",
    effect:
      "Zeros are compressible and skippable, so unstructured sparsity buys latency only when the hardware actually skips them.",
    formula: "w̃ᵢ = wᵢ · 1[|wᵢ| ≤ τ]",
  },
  distillation: {
    what: "Train a smaller student to match a larger teacher's output distribution.",
    effect:
      "Transfers capability rather than compressing it, so the student can in principle exceed the teacher on the distillation task.",
    formula: "L = Σₜ KL( p_teacher(·|xₜ) ‖ p_student(·|xₜ) ) · T²",
  },
  "low-rank-adaptation": {
    what: "Freeze the base weights and learn a low-rank correction.",
    effect:
      "Trainable parameters drop by two orders of magnitude, which is what makes per-task adaptation cheap enough to serve many variants.",
    formula: "ΔW = B·A,   A ∈ ℝʳˣᵈ,  B ∈ ℝᵈˣʳ,   r ≪ min(d, d)",
  },
  "mixed-precision": {
    what: "Keep most weights in half precision and a few in full.",
    effect:
      "Halves weight memory and doubles tensor-core throughput; the few full-precision weights are the ones whose small values would otherwise underflow.",
    formula: "x_fp32 = x_fp16 for sensitive layers, x_fp16 otherwise",
  },
  "layer-reduction": {
    what: "Remove whole transformer blocks from the depth of the stack.",
    effect:
      "Reduces latency linearly in depth and needs no retraining to measure, which is why it survives as a baseline against distillation.",
    formula: "depth D → D′ < D",
  },
  "token-reduction": {
    what: "Merge or drop tokens before they reach the expensive layers.",
    effect:
      "Attention cost is quadratic in sequence length, so shortening the sequence beats every linear-attention approximation on short-context work.",
    formula: "T → T′ < T before layer 1",
  },
  "attention-approximation": {
    what: "Replace exact attention with a linear-cost approximation.",
    effect:
      "Buys O(n) instead of O(n²), at the cost of an approximation that the paper has to justify against a kernel-level baseline.",
    formula: "softmax(QKᵀ)V  ≈  φ(Q)(φ(K)ᵀV)",
  },
  "weight-sharing": {
    what: "Tie the input embedding to the output projection.",
    effect:
      "Removes the largest single matrix in the model for models with a small vocabulary relative to hidden size.",
    formula: "E_out = E_inᵀ",
  },
  "architecture-search": {
    what: "Search over model configurations rather than training one by hand.",
    effect:
      "Finds better size/accuracy points than scaling does, at the cost of many training runs to evaluate each candidate.",
    formula: "A* = argmax_A  Quality(A) / Cost(A)",
  },
  "speculative-decoding": {
    what: "Propose several tokens with a small draft model, verify them in one batch.",
    effect:
      "Output distribution is unchanged, so this is a pure latency win with no accuracy cost — the reason it is attractive for serving.",
    formula: "E_p[accepted tokens / γ · (1 − ρ)]",
  },
  caching: {
    what: "Keep attention keys and values from earlier steps instead of recomputing them.",
    effect:
      "Avoids recomputation proportional to sequence length, at the cost of memory that grows with batch size and context length.",
    formula: "KV_t = KV_{t−1} ‖ (k_t, v_t)",
  },
  calibration: {
    what: "Rescale activations so that quantization error does not accumulate.",
    effect:
      "Often a larger accuracy win than a finer grid, because it removes the outliers that dominate the scale.",
    formula: "x̂ = x / (max|x| / Q_max)",
  },
};

export function OperatorsView() {
  const [open, setOpen] = useState<ChangeOperator | null>(null);

  const counts = useMemo(() => {
    const m: Record<string, number> = {};
    for (const p of PAPERS)
      for (const c of p.changes) m[c.operator] = (m[c.operator] ?? 0) + 1;
    return m;
  }, []);

  const papersFor = (op: ChangeOperator) =>
    PAPERS.filter((p) => p.changes.some((c) => c.operator === op)).sort(
      (a, b) => b.citations - a.citations,
    );

  const total = PAPERS.reduce((n, p) => n + p.changes.length, 0);

  return (
    <div className="space-y-6">
      <SectionTitle>Operators</SectionTitle>

      <Card tone="accent">
        <div className="grid gap-3 sm:grid-cols-3">
          <div>
            <p className="font-mono text-[10px] uppercase tracking-wide text-cyan-400">
              operator kinds
            </p>
            <p className="mt-1 text-2xl font-semibold text-zinc-100">
              {ALL_CHANGE_OPERATORS.length}
            </p>
          </div>
          <div>
            <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
              applied
            </p>
            <p className="mt-1 text-2xl font-semibold text-zinc-100">{total}</p>
          </div>
          <div>
            <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
              unused in corpus
            </p>
            <p className="mt-1 text-2xl font-semibold text-zinc-100">
              {ALL_CHANGE_OPERATORS.filter((o) => !(counts[o] > 0)).length}
            </p>
          </div>
        </div>
      </Card>

      <div className="grid gap-3 md:grid-cols-2">
        {ALL_CHANGE_OPERATORS.map((op) => {
          const n = counts[op] ?? 0;
          const ref = REFERENCE[op];
          const on = open === op;
          return (
            <Card key={op} className={on ? "border-cyan-700/60" : ""}>
              <button
                onClick={() => setOpen(on ? null : op)}
                className="flex w-full items-start justify-between gap-2 text-left"
              >
                <span className="flex items-center gap-2">
                  <span
                    aria-hidden
                    className="mt-0.5 inline-block h-3 w-3 rounded-sm"
                    style={{ background: CHANGE_COLOR[op] }}
                  />
                  <span className="text-[13px] font-medium text-zinc-100">
                    {CHANGE_LABEL[op]}
                  </span>
                </span>
                <Badge tone={n ? "cyan" : "zinc"}>
                  {n} {n === 1 ? "paper" : "papers"}
                </Badge>
              </button>

              <p className="mt-2 text-[12px] leading-5 text-zinc-400">{ref.what}</p>

              {on && (
                <div className="mt-3 space-y-2">
                  <Formula>{ref.formula}</Formula>
                  <p className="text-[12px] leading-5 text-zinc-400">{ref.effect}</p>
                  <div className="space-y-1.5 pt-1">
                    {papersFor(op)
                      .slice(0, 5)
                      .map((p) => {
                        const hit = p.changes.find((c) => c.operator === op);
                        return (
                          <div
                            key={p.id}
                            className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-2"
                          >
                            <div className="flex flex-wrap items-start justify-between gap-2">
                              <div className="min-w-0">
                                <div className="flex flex-wrap items-center gap-1.5">
                                  <Badge tone="cyan">{p.id}</Badge>
                                  <CiteBadge n={p.citations} />
                                  {p.year && <Badge>{p.year}</Badge>}
                                </div>
                                <p className="mt-1 text-[11px] leading-5 text-zinc-300">
                                  {p.shortTitle}
                                </p>
                              </div>
                              <TfPaperLink id={p.id} link short />
                            </div>
                            {hit && <Quote>{hit.quote}</Quote>}
                          </div>
                        );
                      })}
                    {n === 0 && (
                      <p className="font-mono text-[10px] text-zinc-600">
                        no paper in the corpus names this operator
                      </p>
                    )}
                  </div>
                </div>
              )}
            </Card>
          );
        })}
      </div>
    </div>
  );
}