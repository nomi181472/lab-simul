"use client";

import type { ReactNode } from "react";
import type { Confidence, Evidence, EvidenceKind, Paper, ProblemStatus } from "@/labs/object-detection/data/types";
import { PAPER_BY_ID } from "@/labs/object-detection/data/papers";
import { SIMULATOR_INDEX } from "@/labs/object-detection/data/concepts";

/* ---------- atoms ---------- */

export function Card({
  children,
  className = "",
  tone = "default",
}: {
  children: ReactNode;
  className?: string;
  tone?: "default" | "accent" | "warn" | "ok";
}) {
  const tones: Record<string, string> = {
    default: "border-zinc-800 bg-zinc-900/60",
    accent: "border-emerald-700/50 bg-emerald-950/20",
    warn: "border-amber-700/50 bg-amber-950/20",
    ok: "border-sky-700/50 bg-sky-950/20",
  };
  return (
    <div className={`rounded-xl border p-4 ${tones[tone]} ${className}`}>{children}</div>
  );
}

export function SectionTitle({ children, id }: { children: ReactNode; id?: string }) {
  return (
    <h2 id={id} className="text-lg font-semibold tracking-tight text-zinc-100">
      {children}
    </h2>
  );
}

export type BadgeTone = "zinc" | "emerald" | "amber" | "sky" | "rose" | "violet";

export function Badge({
  children,
  tone = "zinc",
}: {
  children: ReactNode;
  tone?: BadgeTone;
}) {
  const tones: Record<BadgeTone, string> = {
    zinc: "bg-zinc-800 text-zinc-300 border-zinc-700",
    emerald: "bg-emerald-950 text-emerald-300 border-emerald-800",
    amber: "bg-amber-950 text-amber-300 border-amber-800",
    sky: "bg-sky-950 text-sky-300 border-sky-800",
    rose: "bg-rose-950 text-rose-300 border-rose-800",
    violet: "bg-violet-950 text-violet-300 border-violet-800",
  };
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

export function ConfidenceBadge({ confidence }: { confidence: Confidence }) {
  const tone = confidence === "HIGH" ? "emerald" : confidence === "MEDIUM" ? "amber" : "rose";
  return <Badge tone={tone}>{confidence}</Badge>;
}

export function StatusBadge({ status }: { status: ProblemStatus }) {
  const tone: Record<ProblemStatus, BadgeTone> = {
    resolved: "emerald",
    reduced: "sky",
    persistent: "amber",
    transformed: "violet",
    uncertain: "rose",
  };
  return <Badge tone={tone[status]}>{status}</Badge>;
}

export function PaperLink({ id, short = false }: { id: string; short?: boolean }) {
  const p: Paper | undefined = PAPER_BY_ID[id];
  if (!p) return <span className="text-rose-400">?{id}</span>;
  return (
    <span className="inline-flex items-baseline gap-1">
      <span className="rounded bg-zinc-800 px-1 py-0.5 font-mono text-[10px] text-emerald-300">
        {p.id}
      </span>
      {!short && <span className="text-zinc-400">{p.shortTitle}</span>}
    </span>
  );
}

function quoteIfLong(q?: string) {
  if (!q) return null;
  return q.length > 200 ? q.slice(0, 197) + "…" : q;
}

export function EvidenceLine({
  evidence,
  compact = false,
}: {
  evidence: Evidence;
  compact?: boolean;
}) {
  const kindTone: Record<EvidenceKind, BadgeTone> = {
    paperSays: "emerald",
    crossPaper: "sky",
    labInterpretation: "violet",
  };
  const kindLabel: Record<EvidenceKind, string> = {
    paperSays: "paper says",
    crossPaper: "cross-paper",
    labInterpretation: "lab synthesis",
  };
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={kindTone[evidence.kind]}>{kindLabel[evidence.kind]}</Badge>
        <ConfidenceBadge confidence={evidence.confidence} />
        {evidence.location && (
          <span className="font-mono text-[10px] text-zinc-500">{evidence.location}</span>
        )}
      </div>
      {!compact && evidence.paperIds.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {evidence.paperIds.map((pid) => (
            <PaperLink key={pid} id={pid} />
          ))}
        </div>
      )}
      {evidence.quote && (
        <p className="mt-2 text-[12px] leading-5 text-zinc-400">
          <span className="text-zinc-600">“</span>
          {quoteIfLong(evidence.quote)}
          <span className="text-zinc-600">”</span>
        </p>
      )}
    </div>
  );
}

export function InsufficientEvidence({ what }: { what: string }) {
  return (
    <div className="rounded-lg border border-rose-900/60 bg-rose-950/20 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone="rose">insufficient evidence</Badge>
      </div>
      <p className="mt-2 text-[12px] leading-5 text-zinc-400">
        No corpus PDF supports the claim “{what}”. This assertion is withheld.
      </p>
    </div>
  );
}

export function Formula({ children }: { children: ReactNode }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-zinc-800 bg-black/40 px-3 py-2 font-mono text-[12px] leading-6 text-amber-200">
      {children}
    </div>
  );
}

export function MiniBar({
  values,
  labels,
  max,
  color = "bg-emerald-500",
}: {
  values: number[];
  labels: string[];
  max?: number;
  color?: string;
}) {
  const m = max ?? Math.max(1, ...values);
  return (
    <div className="flex flex-col gap-1.5">
      {values.map((v, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="w-24 shrink-0 truncate text-right font-mono text-[10px] text-zinc-500">
            {labels[i]}
          </span>
          <div className="h-3 flex-1 overflow-hidden rounded-sm bg-zinc-800">
            <div
              className={`h-full ${color}`}
              style={{ width: `${(v / m) * 100}%` }}
            />
          </div>
          <span className="w-16 shrink-0 font-mono text-[10px] text-zinc-400">
            {v.toFixed(3)}
          </span>
        </div>
      ))}
    </div>
  );
}

export function checkSim(sim?: string) {
  if (!sim) return null;
  return SIMULATOR_INDEX[sim] ? sim : null;
}

export function KV({ k, v }: { k: string; v: ReactNode }) {
  return (
    <div className="flex flex-wrap items-baseline justify-between gap-2">
      <span className="font-mono text-[11px] text-zinc-500">{k}</span>
      <span className="font-mono text-[12px] text-zinc-200">{v}</span>
    </div>
  );
}