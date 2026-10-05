"use client";

import type { ReactNode } from "react";
import Link from "next/link";
import { PAPER_BY_ID } from "@/labs/neuroevolution/data/papers";
import {
  LAYER_COLOR,
  LAYER_LABEL,
  LOCUS_COLOR,
  LOCUS_LABEL,
  type LayerKind,
  type Locus,
} from "@/labs/neuroevolution/data/types";

/* ---------- atoms (rose accent = neuroevolution lab identity) ---------- */

export function Card({
  children,
  className = "",
  tone = "default",
}: {
  children: ReactNode;
  className?: string;
  tone?: "default" | "accent" | "warn" | "ok" | "violet";
}) {
  const tones: Record<string, string> = {
    default: "border-zinc-800 bg-zinc-900/60",
    accent: "border-rose-700/50 bg-rose-950/20",
    warn: "border-amber-700/50 bg-amber-950/20",
    ok: "border-emerald-700/50 bg-emerald-950/20",
    violet: "border-violet-700/50 bg-violet-950/20",
  };
  return (
    <div className={`rounded-xl border p-4 ${tones[tone]} ${className}`}>{children}</div>
  );
}

export function SectionTitle({ children }: { children: ReactNode }) {
  return (
    <h2 className="text-lg font-semibold tracking-tight text-zinc-100">{children}</h2>
  );
}

export type BadgeTone = "zinc" | "rose" | "amber" | "emerald" | "cyan" | "violet";

export function Badge({
  children,
  tone = "zinc",
}: {
  children: ReactNode;
  tone?: BadgeTone;
}) {
  const tones: Record<BadgeTone, string> = {
    zinc: "bg-zinc-800 text-zinc-300 border-zinc-700",
    rose: "bg-rose-950 text-rose-300 border-rose-800",
    amber: "bg-amber-950 text-amber-300 border-amber-800",
    emerald: "bg-emerald-950 text-emerald-300 border-emerald-800",
    cyan: "bg-cyan-950 text-cyan-300 border-cyan-800",
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

/** Paper chip. `link` additionally exposes the PDF route when fileName is known. */
export function NeuroPaperLink({
  id,
  short = false,
  link = false,
}: {
  id: string;
  short?: boolean;
  link?: boolean;
}) {
  const p = PAPER_BY_ID[id];
  if (!p)
    return <span className="font-mono text-[10px] text-rose-400">?{id}</span>;
  const chip = (
    <span className="inline-flex items-baseline gap-1">
      <span className="rounded bg-rose-950 px-1 py-0.5 font-mono text-[10px] text-rose-300">
        {p.id}
      </span>
      {!short && <span className="text-[11px] text-zinc-400">{p.shortTitle}</span>}
    </span>
  );
  if (!link) return chip;
  return (
    <Link
      href={`/papers/${encodeURIComponent(p.fileName)}`}
      target="_blank"
      title={`Open ${p.fileName}`}
      className="hover:text-rose-200"
    >
      {chip}
    </Link>
  );
}

/** Formulas are plain unicode in a monospace box — no KaTeX in this project. */
export function Formula({ children }: { children: ReactNode }) {
  return (
    <div className="max-w-full overflow-x-auto whitespace-nowrap rounded-lg border border-zinc-800 bg-black/40 px-3 py-2 font-mono text-[12px] leading-6 text-amber-200">
      {children}
    </div>
  );
}

export function DifficultyBadge({ d }: { d: string }) {
  const tone: BadgeTone =
    d === "intro" ? "emerald" : d === "intermediate" ? "amber" : "rose";
  return <Badge tone={tone}>{d}</Badge>;
}

export function CiteBadge({ n }: { n: number }) {
  const tone: BadgeTone = n >= 100 ? "emerald" : n >= 20 ? "rose" : "zinc";
  return <Badge tone={tone}>{n} cites</Badge>;
}

/* ---------- domain atoms: the colourful bits ---------- */

/** Colour swatch + name for a layer kind. */
export function LayerChip({ kind }: { kind: LayerKind }) {
  const c = LAYER_COLOR[kind];
  return (
    <span className="inline-flex items-center gap-1 rounded-md border border-zinc-800 bg-zinc-900/70 px-1.5 py-0.5 text-[10px] text-zinc-300">
      <span
        aria-hidden
        className="inline-block h-2.5 w-2.5 rounded-sm"
        style={{ background: c }}
      />
      <span className="font-mono">{LAYER_LABEL[kind]}</span>
    </span>
  );
}

/** Colour swatch + name for what the operators modify. */
export function LocusChip({ locus }: { locus: Locus }) {
  const c = LOCUS_COLOR[locus];
  return (
    <span className="inline-flex items-center gap-1 rounded-md border border-zinc-800 bg-zinc-900/70 px-1.5 py-0.5 text-[10px] text-zinc-300">
      <span
        aria-hidden
        className="inline-block h-2.5 w-2.5 rounded-sm"
        style={{ background: c }}
      />
      <span className="font-mono">{LOCUS_LABEL[locus]}</span>
    </span>
  );
}

/**
 * A compact "no evidence" marker. The lab's provenance policy says an unstated
 * field must be visibly empty rather than quietly filled, so every place that
 * could have guessed uses this instead.
 */
export function Unstated({ what }: { what: string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-md border border-zinc-800 bg-zinc-900/60 px-1.5 py-0.5 font-mono text-[10px] text-zinc-500">
      {what} not stated
    </span>
  );
}

/** Quoted evidence line, marked as such so it is never read as lab prose. */
export function Quote({ children }: { children: ReactNode }) {
  return (
    <blockquote className="border-l-2 border-rose-800/60 pl-2 text-[11px] italic leading-5 text-zinc-400">
      {children}
    </blockquote>
  );
}