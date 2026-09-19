"use client";

import type { ReactNode } from "react";
import { Card, Formula, PaperLink } from "@/labs/object-detection/ui";

/* ---------- controls ---------- */

export function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  fmt,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  fmt?: (v: number) => string;
}) {
  return (
    <label className="flex min-w-0 flex-col gap-1">
      <span className="flex items-baseline justify-between gap-2 font-mono text-[10px] text-zinc-500">
        {label}
        <span className="text-emerald-300">{(fmt ?? ((v: number) => v.toFixed(2)))(value)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-emerald-500"
      />
    </label>
  );
}

export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">{label}</span>
      <div className="inline-flex flex-wrap gap-1">
        {options.map((o) => (
          <button
            key={o.value}
            onClick={() => onChange(o.value)}
            className={`rounded-md border px-2 py-1 text-[11px] transition-colors ${
              value === o.value
                ? "border-emerald-600 bg-emerald-600/20 text-emerald-200"
                : "border-zinc-800 text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function Seeded({
  seed,
  onReseed,
}: {
  seed: number;
  onReseed: () => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="font-mono text-[10px] text-zinc-500">seed {seed}</span>
      <button
        onClick={onReseed}
        className="rounded-md border border-zinc-800 px-2 py-1 text-[11px] text-zinc-300 hover:bg-zinc-900"
      >
        regenerate (deterministic)
      </button>
    </div>
  );
}

/* ---------- sim frame ---------- */

export type SimCite = { text: string; papers: string[] };

export function Sim({
  name,
  identifies,
  math,
  need,
  real,
  papers,
  cites,
  params,
  children,
}: {
  name: string;
  identifies: string;
  math: (string | ReactNode)[];
  need: string;
  real: string;
  papers: string[];
  cites?: SimCite[];
  params: ReactNode;
  children: ReactNode;
}) {
  return (
    <Card tone="default">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-zinc-100">{name}</h3>
          <p className="font-mono text-[10px] text-emerald-400">{identifies}</p>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {papers.map((p) => (
            <PaperLink key={p} id={p} />
          ))}
        </div>
      </div>

      <div className="mt-3">
        <div className="font-mono text-[9px] uppercase tracking-wide text-sky-400">what am I looking at</div>
        <p className="mt-0.5 text-[12px] leading-5 text-zinc-400">{identifies}</p>
      </div>

      <div className="mt-3">
        <div className="font-mono text-[9px] uppercase tracking-wide text-amber-400">the math</div>
        <div className="mt-1 space-y-1.5">
          {math.map((m, i) => (
            <Formula key={i}>{m}</Formula>
          ))}
        </div>
        <p className="mt-1.5 text-[11px] leading-5 text-zinc-500">{need}</p>
      </div>

      <div className="mt-3 rounded-lg border border-emerald-900/40 bg-emerald-950/20 p-2">
        <div className="font-mono text-[9px] uppercase tracking-wide text-emerald-400">
          real research problem it represents
        </div>
        <p
          className="mt-0.5 text-[11px] leading-4 text-zinc-300"
          title={`evidence: ${papers.join(", ")}`}
        >
          {real}
        </p>
        {(cites ?? []).length > 0 && (
          <div className="mt-2 space-y-1.5">
            <div className="font-mono text-[9px] uppercase tracking-wide text-amber-400">
              evidence chips
            </div>
            {(cites ?? []).map((c, i) => (
              <div
                key={i}
                className="flex flex-wrap items-center gap-2 rounded-lg border border-amber-900/40 bg-amber-950/20 px-2 py-1.5"
                title={`evidence: ${c.papers.join(", ")}`}
              >
                <span className="text-[11px] leading-4 text-zinc-300">{c.text}</span>
                <span className="ml-auto flex gap-1">
                  {c.papers.map((p) => (
                    <PaperLink key={p} id={p} short />
                  ))}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="mt-4">
        <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-500">controls</div>
        <div className="mt-2 flex flex-wrap items-end gap-4">{params}</div>
      </div>

      <div className="mt-4">{children}</div>
    </Card>
  );
}