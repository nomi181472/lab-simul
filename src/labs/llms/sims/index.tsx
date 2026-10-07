"use client";

import { useMemo, useState } from "react";
import { Card, SectionTitle, Badge, TfPaperLink } from "@/labs/llms/ui";
import { useTfLab } from "@/labs/llms/context";
import { PAPER_BY_ID } from "@/labs/llms/data/papers";
import { SIMULATOR_PAPERS } from "@/labs/llms/sims/papers";
import * as K from "@/labs/llms/sims/kernels";
import type { Kernel } from "@/labs/llms/sims/kernels";

interface Meta {
  id: string;
  label: string;
  blurb: string;
  kernel: Kernel;
}

export const SIMULATORS: Meta[] = [
  {
    id: "quantization",
    label: "Quantization",
    blurb: "Spend fewer bits per weight and watch perplexity find the knee.",
    kernel: K.quantization,
  },
  {
    id: "pruning",
    label: "Pruning",
    blurb: "Zero weights out and see how much of the model was load-bearing.",
    kernel: K.pruning,
  },
  {
    id: "layer-reduction",
    label: "Layer reduction",
    blurb: "Remove whole blocks: latency falls linearly, quality does not.",
    kernel: K.layerReduction,
  },
  {
    id: "distillation",
    label: "Distillation",
    blurb: "A small student against a large teacher's quality and cost.",
    kernel: K.distillation,
  },
  {
    id: "peft",
    label: "Low-rank adaptation",
    blurb: "Trainable parameters collapse; quality barely notices.",
    kernel: K.peft,
  },
  {
    id: "moe",
    label: "Mixture of experts",
    blurb: "Total parameters grow, active parameters do not.",
    kernel: K.moe,
  },
  {
    id: "kv-cache",
    label: "KV cache",
    blurb: "Why long context is bounded by memory, not by attention.",
    kernel: K.kvCache,
  },
  {
    id: "speculative-decoding",
    label: "Speculative decoding",
    blurb: "Latency bought without changing the output distribution.",
    kernel: K.speculative,
  },
  {
    id: "retrieval",
    label: "Retrieval",
    blurb: "Capability added at inference time with no new parameters.",
    kernel: K.retrieval,
  },
  {
    id: "reasoning",
    label: "Reasoning strategies",
    blurb: "Accuracy bought with sampling passes.",
    kernel: K.reasoning,
  },
];

/** Minimal inline sparkline: no chart library is used anywhere in this project. */
function Spark({ points }: { points: { value: number }[] }) {
  if (points.length < 2) return null;
  const max = Math.max(...points.map((p) => p.value));
  const min = Math.min(...points.map((p) => p.value));
  const span = max - min || 1;
  const w = 240;
  const h = 40;
  const path = points
    .map((p, i) => {
      const x = (i / (points.length - 1)) * w;
      const y = h - ((p.value - min) / span) * h;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="h-10 w-full"
      role="img"
      aria-label="series trend"
      preserveAspectRatio="none"
    >
      <path d={path} fill="none" stroke="#22d3ee" strokeWidth="1.5" />
    </svg>
  );
}

export function SimRunner() {
  const { sim, setSim } = useTfLab();
  const meta = SIMULATORS.find((s) => s.id === sim) ?? SIMULATORS[0];
  const [config, setConfig] = useState<Record<string, number>>({});

  const out = useMemo(() => meta.kernel.run(config), [meta, config]);

  const evidence = SIMULATOR_PAPERS.find((s) => s.id === meta.id)?.papers ?? [];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <SectionTitle>Simulators</SectionTitle>
        <span className="font-mono text-[10px] text-zinc-600">
          illustrative models, not measurements
        </span>
      </div>

      <Card tone="accent">
        <p className="text-[13px] leading-6 text-zinc-400">
          Ten kernels, one per mechanism the corpus applies. Each is a small honest model
          of the mechanism — a real trade-off with the real shape — not a reimplementation
          of any paper. Every kernel lists the papers it is grounded in.
        </p>
      </Card>

      <div className="flex flex-wrap gap-1">
        {SIMULATORS.map((s) => (
          <button
            key={s.id}
            onClick={() => setSim(s.id)}
            className={`rounded-md border px-2.5 py-1 text-[11px] ${
              s.id === meta.id
                ? "border-cyan-700/50 bg-cyan-600/20 text-cyan-300"
                : "border-zinc-800 bg-zinc-900/60 text-zinc-400 hover:text-zinc-200"
            }`}
          >
            {s.label}
          </button>
        ))}
      </div>

      <Card>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="text-sm font-semibold text-zinc-100">{meta.label}</span>
          <span className="font-mono text-[10px] text-zinc-600">{meta.id}</span>
        </div>
        <p className="mt-1.5 text-[12px] leading-5 text-zinc-400">{meta.blurb}</p>

        {Object.keys(out.series[0]?.points[0] ?? {}).length > 0 && null}

        <div className="mt-4 space-y-4">
          {out.series.map((s) => (
            <div key={s.key}>
              <div className="flex items-baseline justify-between gap-2">
                <span className="font-mono text-[10px] text-zinc-400">{s.label}</span>
                <span className="font-mono text-[10px] text-zinc-600">
                  {s.points[0]?.value} → {s.points[s.points.length - 1]?.value}
                </span>
              </div>
              <Spark points={s.points} />
              <div className="mt-1 flex flex-wrap gap-1">
                {s.points.map((p) => (
                  <span
                    key={p.label}
                    className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[10px] text-zinc-400"
                  >
                    {p.label}
                    <span className="ml-1 text-zinc-200">{p.value}</span>
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="mt-4 space-y-1.5 border-t border-zinc-800 pt-3">
          <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            how the run went
          </p>
          {out.steps.map((st) => (
            <div key={st.t} className="flex gap-2">
              <span className="mt-0.5 w-5 shrink-0 font-mono text-[10px] text-zinc-600">
                {st.t}
              </span>
              <div className="min-w-0">
                <p className="font-mono text-[11px] text-zinc-200">{st.action}</p>
                <p className="text-[11px] leading-5 text-zinc-500">{st.detail}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-4 border-t border-zinc-800 pt-3">
          <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            grounded in
          </p>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {evidence.length === 0 ? (
              <span className="font-mono text-[10px] text-zinc-600">
                no binding generated yet
              </span>
            ) : (
              evidence.map((id) => {
                const p = PAPER_BY_ID[id];
                return (
                  <span
                    key={id}
                    className="inline-flex items-center gap-1 rounded bg-cyan-950 px-1.5 py-0.5 font-mono text-[10px] text-cyan-300"
                    title={p?.title}
                  >
                    {id}
                    <span className="text-cyan-500/80">
                      {p ? p.citations.toLocaleString() : ""}
                    </span>
                  </span>
                );
              })
            )}
          </div>
          {evidence.length > 0 && (
            <div className="mt-2 space-y-1">
              {evidence.map((id) => {
                const p = PAPER_BY_ID[id];
                if (!p) return null;
                return (
                  <p key={id} className="truncate text-[11px] text-zinc-500">
                    <TfPaperLink id={id} short /> {p.shortTitle}
                  </p>
                );
              })}
            </div>
          )}
        </div>
      </Card>

      <Card>
        <SectionTitle>What these do not show</SectionTitle>
        <ul className="mt-2 space-y-1.5 text-[12px] leading-5 text-zinc-400">
          <li>
            Hardware. Latency curves assume a fixed amount of parallelism; real kernels
            are bound by memory bandwidth and kernel occupancy.
          </li>
          <li>
            Calibration. Every quantization and pruning curve here is uncalibrated, which
            is the pessimistic case.
          </li>
          <li>
            Data. Accuracy numbers are placeholders on a fixed scale so the curves are
            comparable to each other, not to any published benchmark.
          </li>
        </ul>
      </Card>
    </div>
  );
}