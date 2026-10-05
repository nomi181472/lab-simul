"use client";

import { useMemo, useState } from "react";
import { PAPERS, PAPER_BY_ID } from "@/labs/neuroevolution/data/papers";
import { MANIFEST, MANIFEST_TOTAL_CITATIONS } from "@/labs/neuroevolution/data/manifest";
import {
  Card,
  SectionTitle,
  Badge,
  CiteBadge,
  NeuroPaperLink,
  Unstated,
  LayerChip,
} from "@/labs/neuroevolution/ui";
import {
  ArchitectureDiagram,
  LayerLegend,
  ModificationDiagram,
} from "@/labs/neuroevolution/architecture";
import {
  LAYER_LABEL,
  LOCUS_COLOR,
  LOCUS_LABEL,
  type LayerKind,
  type Locus,
} from "@/labs/neuroevolution/data/types";
import { useNeuroLab } from "@/labs/neuroevolution/context";

/* The Architecture view.
 *
 * The brief for this lab was that architecture information must be on every
 * paper, otherwise the corpus is unreadable. So this view is deliberately the
 * richest one: every paper gets a diagram, and a paper whose text never
 * enumerates its layers says so in the diagram itself rather than being skipped.
 */

export function ArchitectureView() {
  const { locusFilter, setLocusFilter } = useNeuroLab();
  const [q, setQ] = useState("");
  const [onlyDrawn, setOnlyDrawn] = useState(false);

  const drawn = useMemo(() => PAPERS.filter((p) => p.architecture.layers.length > 0), []);

  const locusCounts = useMemo(() => {
    const m = new Map<Locus, number>();
    for (const p of PAPERS) {
      for (const c of p.method.evolution.changes) {
        m.set(c.locus, (m.get(c.locus) ?? 0) + 1);
      }
    }
    return [...m.entries()].sort((a, b) => b[1] - a[1]);
  }, []);

  const list = useMemo(() => {
    let out = PAPERS;
    if (onlyDrawn) out = out.filter((p) => p.architecture.layers.length > 0);
    if (locusFilter) {
      const l = locusFilter as Locus;
      out = out.filter((p) => p.method.evolution.changes.some((c) => c.locus === l));
    }
    const needle = q.trim().toLowerCase();
    if (needle) {
      out = out.filter(
        (p) =>
          p.title.toLowerCase().includes(needle) ||
          p.architecture.summary.toLowerCase().includes(needle) ||
          p.tags.some((t) => t.includes(needle)),
      );
    }
    return out;
  }, [q, onlyDrawn, locusFilter]);

  const kinds = useMemo(() => {
    const s = new Set<LayerKind>(drawn.flatMap((p) => p.architecture.layers.map((l) => l.kind)));
    return [...s];
  }, [drawn]);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-3">
        <SectionTitle>Architecture</SectionTitle>
        <p className="max-w-3xl text-[13px] leading-6 text-zinc-400">
          What each paper actually evolved, drawn from its own description. Every
          paper in the corpus is here; a paper that never enumerates its network
          in the retrieved text is shown with an explicit placeholder rather than
          dropped, so a gap in the record is visible instead of silent.
        </p>
        <div className="flex flex-wrap gap-2 font-mono text-[10px] text-zinc-500">
          <Badge tone="rose">{drawn.length} with drawn topology</Badge>
          <Badge tone="zinc">{PAPERS.length - drawn.length} prose-only</Badge>
          <Badge tone="zinc">{MANIFEST_TOTAL_CITATIONS.toLocaleString()} citations</Badge>
        </div>
      </div>

      <Card>
        <div className="flex flex-col gap-2">
          <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">
            layer legend · colours are used consistently across every diagram
          </div>
          <LayerLegend kinds={kinds} />
        </div>
      </Card>

      <Card tone="violet">
        <div className="flex flex-col gap-3">
          <div>
            <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">
              which locus does the search change?
            </div>
            <p className="mt-1 text-[11px] leading-5 text-zinc-400">
              Filter the corpus by the part of the artefact a paper&rsquo;s operators
              are allowed to move. Weights-only and weight-plus-topology searches
              are different algorithms, not variants of one.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setLocusFilter(null)}
              className={`rounded-md border px-2.5 py-1 font-mono text-[10px] ${
                !locusFilter
                  ? "border-rose-700/60 bg-rose-600/20 text-rose-300"
                  : "border-zinc-800 text-zinc-500"
              }`}
            >
              all loci
            </button>
            {locusCounts.map(([locus, n]) => (
              <button
                key={locus}
                onClick={() => setLocusFilter(locus === locusFilter ? null : locus)}
                title={`${n} papers`}
                className="rounded-md border px-2.5 py-1 font-mono text-[10px] transition-colors"
                style={{
                  borderColor:
                    locusFilter === locus ? LOCUS_COLOR[locus] : "#27272a",
                  color: locusFilter === locus ? "#e4e4e7" : "#71717a",
                  background:
                    locusFilter === locus ? `${LOCUS_COLOR[locus]}22` : "transparent",
                }}
              >
                {LOCUS_LABEL[locus]} · {n}
              </button>
            ))}
          </div>
        </div>
      </Card>

      <div className="flex flex-wrap items-center gap-3">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="search papers, phenotypes, tags…"
          className="min-w-[220px] flex-1 rounded-md border border-zinc-800 bg-zinc-900/70 px-3 py-1.5 text-[12px] text-zinc-200 placeholder-zinc-600 focus:border-rose-800 focus:outline-none"
        />
        <label className="flex items-center gap-2 font-mono text-[10px] text-zinc-500">
          <input
            type="checkbox"
            checked={onlyDrawn}
            onChange={(e) => setOnlyDrawn(e.target.checked)}
            className="accent-rose-500"
          />
          only papers with a drawn topology
        </label>
        <span className="font-mono text-[10px] text-zinc-600">{list.length} shown</span>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {list.map((p) => {
          const changes = p.method.evolution.changes;
          return (
            <Card key={p.id} className="flex flex-col gap-3">
              <div className="flex flex-col gap-1.5">
                <div className="flex flex-wrap items-center gap-2">
                  <NeuroPaperLink id={p.id} link />
                  <CiteBadge n={p.citations} />
                  <Badge tone="zinc">{p.year}</Badge>
                  <Badge tone="cyan">{p.architecture.phenotype}</Badge>
                </div>
                <div className="text-[13px] font-medium leading-5 text-zinc-100">
                  {p.title}
                </div>
                {p.architecture.layers.length > 0 ? (
                  <div className="flex flex-wrap gap-1.5">
                    {p.architecture.layers.map((l, i) => (
                      <LayerChip key={i} kind={l.kind} />
                    ))}
                  </div>
                ) : null}
              </div>

              <ArchitectureDiagram architecture={p.architecture} compact />

              {p.architecture.genotypeToPhenotype ? (
                <div className="flex flex-col gap-1">
                  <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-600">
                    genotype → phenotype
                  </div>
                  <div className="text-[11px] leading-5 text-zinc-400">
                    {p.architecture.genotypeToPhenotype}
                  </div>
                </div>
              ) : (
                <Unstated what="genotype-to-phenotype mapping" />
              )}

              {changes.length > 0 ? (
                <ModificationDiagram changes={changes.slice(0, 4)} />
              ) : null}

              {p.architecture.quotes.length > 0 ? (
                <div className="flex flex-col gap-1">
                  <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-600">
                    quoted from the paper
                  </div>
                  {p.architecture.quotes.slice(0, 2).map((qt, i) => (
                    <blockquote
                      key={i}
                      className="border-l-2 border-rose-800/60 pl-2 text-[11px] italic leading-5 text-zinc-400"
                    >
                      &ldquo;{qt}&rdquo;
                    </blockquote>
                  ))}
                </div>
              ) : null}
            </Card>
          );
        })}
      </div>

      {list.length === 0 ? (
        <Card tone="warn">
          <div className="text-[12px] text-amber-300/80">
            No paper matches this filter. The corpus does not claim every paper
            describes its network enumerably.
          </div>
        </Card>
      ) : null}
    </div>
  );
}

/* Compact per-paper card reused by the Papers view. */
export function PaperArchitectureCard({ id }: { id: string }) {
  const p = PAPER_BY_ID[id];
  if (!p) return null;
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        {p.architecture.layers.map((l, i) => (
          <span
            key={i}
            className="rounded px-1.5 py-0.5 font-mono text-[10px] text-zinc-900"
            style={{ backgroundColor: LAYER_LABEL[l.kind] ? "#a1a1aa" : "#a1a1aa" }}
            title={LAYER_LABEL[l.kind]}
          >
            {l.label}
          </span>
        ))}
      </div>
      <ArchitectureDiagram architecture={p.architecture} compact />
    </div>
  );
}