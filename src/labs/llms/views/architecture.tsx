"use client";

import { useMemo, useState } from "react";
import { Card, SectionTitle, Badge, Quote, Unstated, ArchChip, CiteBadge, TfPaperLink } from "@/labs/llms/ui";
import { useTfLab } from "@/labs/llms/context";
import { PAPERS } from "@/labs/llms/data/papers";
import {
  ALL_ARCH_KINDS,
  ARCH_COLOR,
  ARCH_LABEL,
  type ArchKind,
} from "@/labs/llms/data/vocab";

/**
 * The stack diagram.
 *
 * Drawn from the paper's own quoted layers, ordered by position in the pipeline,
 * so a reader can see which structural element a paper actually commits to. A gap
 * in the column is real: the paper did not name that element in its extracted
 * text, and the diagram leaves it empty rather than interpolating a standard
 * transformer.
 */
function StackDiagram({
  layers,
  title,
}: {
  layers: { kind: ArchKind; label: string; quote: string }[];
  title: string;
}) {
  const byKind = useMemo(() => {
    const m = new Map<ArchKind, string>();
    for (const l of layers) if (!m.has(l.kind)) m.set(l.kind, l.quote);
    return m;
  }, [layers]);

  return (
    <div>
      <p className="mb-2 text-[12px] font-medium text-zinc-200">{title}</p>
      <div className="flex flex-col gap-1">
        {ALL_ARCH_KINDS.map((k) => {
          const present = byKind.has(k);
          return (
            <div
              key={k}
              className={`flex items-center gap-2 rounded-md border px-2 py-1.5 ${
                present
                  ? "border-zinc-700 bg-zinc-900/70"
                  : "border-dashed border-zinc-800/70 bg-zinc-900/20"
              }`}
            >
              <span
                aria-hidden
                className="h-3.5 w-1.5 shrink-0 rounded-full"
                style={{ background: present ? ARCH_COLOR[k] : "#3f3f46" }}
              />
              <span
                className={`w-28 shrink-0 font-mono text-[10px] ${
                  present ? "text-zinc-200" : "text-zinc-600"
                }`}
              >
                {ARCH_LABEL[k]}
              </span>
              {present ? (
                <span className="min-w-0 flex-1 truncate text-[11px] italic text-zinc-500">
                  {byKind.get(k)}
                </span>
              ) : (
                <span className="flex-1">
                  <Unstated what={ARCH_LABEL[k]} />
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function ArchitectureView() {
  const { kindFilter, setKindFilter, setSection } = useTfLab();
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    let list = [...PAPERS].sort((a, b) => b.citations - a.citations);
    if (kindFilter) list = list.filter((p) => p.architecture.layers.some((l) => l.kind === kindFilter));
    if (query.trim()) {
      const q = query.trim().toLowerCase();
      list = list.filter(
        (p) =>
          p.title.toLowerCase().includes(q) ||
          p.architecture.layers.some((l) => l.quote.toLowerCase().includes(q)),
      );
    }
    return list;
  }, [kindFilter, query]);

  const withArch = PAPERS.filter((p) => p.architecture.layers.length > 0);
  const totalLayers = PAPERS.reduce((n, p) => n + p.architecture.layers.length, 0);

  return (
    <div className="space-y-6">
      <SectionTitle>Architecture</SectionTitle>

      <div className="grid gap-3 sm:grid-cols-3">
        <Card tone="accent">
          <p className="font-mono text-[10px] uppercase tracking-wide text-cyan-400">
            papers with a stack
          </p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">
            {withArch.length}
            <span className="text-sm text-zinc-500">/{PAPERS.length}</span>
          </p>
        </Card>
        <Card>
          <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            structural elements
          </p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">{totalLayers}</p>
        </Card>
        <Card>
          <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            element types
          </p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">{ALL_ARCH_KINDS.length}</p>
        </Card>
      </div>

      <Card>
        <div className="flex flex-wrap items-center gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="filter by title or quoted text"
            className="min-w-[220px] flex-1 rounded-md border border-zinc-800 bg-black/40 px-3 py-1.5 font-mono text-[11px] text-zinc-200 placeholder:text-zinc-600"
          />
          <button
            onClick={() => setKindFilter(null)}
            className={`rounded-md border px-2.5 py-1.5 text-[11px] ${
              kindFilter
                ? "border-zinc-700 bg-zinc-900 text-zinc-300"
                : "border-cyan-700/50 bg-cyan-600/20 text-cyan-300"
            }`}
          >
            all elements
          </button>
        </div>
        <div className="mt-3 flex flex-wrap gap-1.5">
          {ALL_ARCH_KINDS.map((k) => {
            const n = PAPERS.filter((p) => p.architecture.layers.some((l) => l.kind === k)).length;
            const on = kindFilter === k;
            return (
              <button
                key={k}
                onClick={() => setKindFilter(on ? null : k)}
                className={`inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[10px] ${
                  on
                    ? "border-cyan-600 bg-cyan-950 text-cyan-200"
                    : "border-zinc-800 bg-zinc-900/60 text-zinc-400 hover:text-zinc-200"
                }`}
              >
                <span
                  aria-hidden
                  className="inline-block h-2 w-2 rounded-sm"
                  style={{ background: ARCH_COLOR[k] }}
                />
                <span className="font-mono">{ARCH_LABEL[k]}</span>
                <span className="text-zinc-500">{n}</span>
              </button>
            );
          })}
        </div>
        <p className="mt-3 font-mono text-[10px] text-zinc-600">{filtered.length} papers</p>
      </Card>

      <div className="space-y-4">
        {filtered.slice(0, 40).map((p) => (
          <Card key={p.id}>
            <div className="flex flex-wrap items-start justify-between gap-2">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-1.5">
                  <Badge tone="cyan">{p.id}</Badge>
                  <CiteBadge n={p.citations} />
                  {p.year && <Badge>{p.year}</Badge>}
                </div>
                <p className="mt-1.5 text-[13px] leading-5 text-zinc-100">{p.title}</p>
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  {p.architecture.layers.map((l) => (
                    <ArchChip key={l.kind} kind={l.kind} />
                  ))}
                </div>
              </div>
              <TfPaperLink id={p.id} link short />
            </div>
            <div className="mt-3 grid gap-4 lg:grid-cols-2">
              <StackDiagram layers={p.architecture.layers} title="Structural stack" />
              <div className="space-y-2">
                <p className="text-[12px] font-medium text-zinc-200">Quoted evidence</p>
                {p.architecture.layers.length === 0 ? (
                  <Unstated what="architecture" />
                ) : (
                  p.architecture.layers.slice(0, 4).map((l) => (
                    <div key={l.kind}>
                      <p className="font-mono text-[10px] text-zinc-500">{l.label}</p>
                      <Quote>{l.quote}</Quote>
                    </div>
                  ))
                )}
              </div>
            </div>
            <button
              onClick={() => setSection("papers")}
              className="mt-3 rounded-md border border-zinc-700 bg-zinc-900 px-2.5 py-1 text-[11px] text-zinc-400"
            >
              open in papers
            </button>
          </Card>
        ))}
      </div>

      {filtered.length > 40 && (
        <p className="font-mono text-[10px] text-zinc-600">
          showing the top 40 of {filtered.length} by citation count
        </p>
      )}
    </div>
  );
}