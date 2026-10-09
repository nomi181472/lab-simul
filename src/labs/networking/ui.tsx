"use client";

import type { ReactNode } from "react";
import Link from "next/link";
import { PAPERS } from "@/labs/networking/data/records";

export type BadgeTone = "zinc" | "cyan" | "emerald" | "amber" | "rose" | "violet";

export function Badge({
  children,
  tone = "zinc",
}: {
  children: ReactNode;
  tone?: BadgeTone;
}) {
  const tones: Record<BadgeTone, string> = {
    zinc: "border-zinc-700 bg-zinc-900 text-zinc-400",
    cyan: "border-cyan-700/60 bg-cyan-950/40 text-cyan-300",
    emerald: "border-emerald-700/60 bg-emerald-950/40 text-emerald-300",
    amber: "border-amber-700/60 bg-amber-950/40 text-amber-300",
    rose: "border-rose-700/60 bg-rose-950/40 text-rose-300",
    violet: "border-violet-700/60 bg-violet-950/40 text-violet-300",
  };
  return (
    <span
      className={`inline-flex items-center rounded-md border px-1.5 py-0.5 font-mono text-[10px] ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

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
    accent: "border-sky-700/50 bg-sky-950/20",
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

export function CiteBadge({ n }: { n: number }) {
  const tone: BadgeTone = n >= 5000 ? "emerald" : n >= 500 ? "cyan" : "zinc";
  return <Badge tone={tone}>{n.toLocaleString()} cites</Badge>;
}

export function Quote({ children }: { children: ReactNode }) {
  return (
    <blockquote className="border-l-2 border-sky-800/60 pl-2 text-[11px] italic leading-5 text-zinc-400">
      {children}
    </blockquote>
  );
}

export function Unstated({ what }: { what: string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-md border border-zinc-800 bg-zinc-900/60 px-1.5 py-0.5 font-mono text-[10px] text-zinc-500">
      {what} not stated in the retrieved text
    </span>
  );
}

const PAPER_BY_ID: Record<string, (typeof PAPERS)[number]> = Object.fromEntries(
  PAPERS.map((p) => [p.id, p]),
);
export { PAPER_BY_ID };

export function PaperLink({
  id,
  short = false,
  link = false,
}: {
  id: string;
  short?: boolean;
  link?: boolean;
}) {
  const p = PAPER_BY_ID[id];
  if (!p) return <span className="font-mono text-[10px] text-sky-400">?{id}</span>;
  const chip = (
    <span className="inline-flex items-baseline gap-1">
      <span className="rounded bg-sky-950 px-1 py-0.5 font-mono text-[10px] text-sky-300">
        {p.id}
      </span>
      {!short && <span className="text-[11px] text-zinc-400">{truncate(p.title, 58)}</span>}
    </span>
  );
  if (!link || !p.fileName) return chip;
  return (
    <Link
      href={`/papers/${encodeURIComponent(p.fileName)}`}
      target="_blank"
      title={`Open ${p.fileName}`}
      className="hover:text-sky-200"
    >
      {chip}
    </Link>
  );
}

export function truncate(s: string, n: number): string {
  if (s.length <= n) return s;
  return s.slice(0, n - 1).replace(/\s+\S*$/, "") + "…";
}

/** Horizontal bar used by every count-ranked view. */
export function BarRow({
  label,
  value,
  max,
  tone = "cyan",
  onClick,
  active = false,
}: {
  label: ReactNode;
  value: number;
  max: number;
  tone?: string;
  onClick?: () => void;
  active?: boolean;
}) {
  const pct = max > 0 ? Math.max(2, Math.round((value / max) * 100)) : 0;
  const colors: Record<string, string> = {
    cyan: "bg-sky-600/70",
    amber: "bg-amber-600/70",
    emerald: "bg-emerald-600/70",
    violet: "bg-violet-600/70",
    rose: "bg-rose-600/70",
  };
  return (
    <button
      onClick={onClick}
      disabled={!onClick}
      className={`block w-full rounded-md px-2 py-1.5 text-left transition-colors ${
        onClick ? "hover:bg-zinc-800/70 cursor-pointer" : "cursor-default"
      } ${active ? "ring-1 ring-sky-600/60" : ""}`}
    >
      <div className="flex items-baseline justify-between gap-2">
        <span className="truncate text-[11px] text-zinc-300">{label}</span>
        <span className="font-mono text-[10px] text-zinc-500">{value}</span>
      </div>
      <div className="mt-1 h-1 rounded bg-zinc-800">
        <div className={`h-1 rounded ${colors[tone] ?? colors.cyan}`} style={{ width: `${pct}%` }} />
      </div>
    </button>
  );
}
