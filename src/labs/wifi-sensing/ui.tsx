"use client";

import type { ReactNode } from "react";
import Link from "next/link";
import { PAPER_BY_ID } from "@/labs/wifi-sensing/data/papers";

/* ---------- atoms (cyan accent = wifi-sensing lab identity) ---------- */

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
    accent: "border-cyan-700/50 bg-cyan-950/20",
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

export type BadgeTone = "zinc" | "cyan" | "amber" | "emerald" | "rose" | "violet";

export function Badge({
  children,
  tone = "zinc",
}: {
  children: ReactNode;
  tone?: BadgeTone;
}) {
  const tones: Record<BadgeTone, string> = {
    zinc: "bg-zinc-800 text-zinc-300 border-zinc-700",
    cyan: "bg-cyan-950 text-cyan-300 border-cyan-800",
    amber: "bg-amber-950 text-amber-300 border-amber-800",
    emerald: "bg-emerald-950 text-emerald-300 border-emerald-800",
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

/** Paper chip. `link` additionally exposes the PDF route when fileName is known. */
export function WifiPaperLink({
  id,
  short = false,
  link = false,
}: {
  id: string;
  short?: boolean;
  link?: boolean;
}) {
  const p = PAPER_BY_ID[id];
  if (!p) return <span className="font-mono text-[10px] text-rose-400">?{id}</span>;
  const chip = (
    <span className="inline-flex items-baseline gap-1">
      <span className="rounded bg-cyan-950 px-1 py-0.5 font-mono text-[10px] text-cyan-300">
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
      className="hover:text-cyan-200"
    >
      {chip}
    </Link>
  );
}

/** Equations are plain unicode in a monospace box — no KaTeX in this project. */
export function Formula({ children }: { children: ReactNode }) {
  return (
    <div className="max-w-full overflow-x-auto whitespace-nowrap rounded-lg border border-zinc-800 bg-black/40 px-3 py-2 font-mono text-[12px] leading-6 text-amber-200">
      {children}
    </div>
  );
}

export function DifficultyBadge({ d }: { d: string }) {
  const tone: BadgeTone = d === "intro" ? "emerald" : d === "intermediate" ? "amber" : "rose";
  return <Badge tone={tone}>{d}</Badge>;
}

export function TaskBadge({ t }: { t: string }) {
  return <Badge tone="violet">{t}</Badge>;
}

/** Citation count, rendered as a sortable-looking chip. */
export function CiteBadge({ n }: { n: number }) {
  const tone: BadgeTone = n >= 100 ? "emerald" : n >= 20 ? "cyan" : "zinc";
  return <Badge tone={tone}>{n} cites</Badge>;
}