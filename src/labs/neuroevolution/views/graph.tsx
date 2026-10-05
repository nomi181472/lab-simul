"use client";

import { useMemo, useState } from "react";
import { PAPERS, PAPER_BY_ID } from "@/labs/neuroevolution/data/papers";
import {
  Card,
  SectionTitle,
  Badge,
  CiteBadge,
  NeuroPaperLink,
  Quote,
} from "@/labs/neuroevolution/ui";
import { ArchitectureDiagram } from "@/labs/neuroevolution/architecture";
import type { RelationType } from "@/labs/neuroevolution/data/types";

/* Relation graph.
 *
 * Edges are corpus-derived: a paper's reference block was checked against the
 * titles and author-years of the rest of the corpus, and the citing sentence was
 * read to label the edge. A cited paper is not automatically a comparison, so
 * the type is never asserted without that sentence.
 */

const REL_TONE: Record<RelationType, "rose" | "emerald" | "violet" | "cyan" | "amber" | "zinc"> = {
  "uses-as-baseline": "rose",
  "builds-on": "violet",
  "improves": "emerald",
  "replaces": "amber",
  "extends": "violet",
  "combines": "cyan",
  "addresses-limitation": "amber",
  "conceptual-successor": "zinc",
  related: "zinc",
};

const REL_LABEL: Record<RelationType, string> = {
  "uses-as-baseline": "uses as baseline",
  "builds-on": "builds on",
  "improves": "improves",
  "replaces": "replaces",
  "extends": "extends",
  "combines": "combines with",
  "addresses-limitation": "addresses limitation of",
  "conceptual-successor": "conceptual successor of",
  related: "cites (relation unstated)",
};

export function GraphView() {
  const [filter, setFilter] = useState<RelationType | "all">("all");

  const edges = useMemo(
    () =>
      PAPERS.flatMap((p) =>
        p.relations.map((r) => ({ from: p.id, to: r.to, type: r.type, note: r.note })),
      ),
    [],
  );

  const byType = useMemo(() => {
    const m = new Map<RelationType, number>();
    for (const e of edges) m.set(e.type, (m.get(e.type) ?? 0) + 1);
    return [...m.entries()].sort((a, b) => b[1] - a[1]);
  }, [edges]);

  const shown = useMemo(
    () => (filter === "all" ? edges : edges.filter((e) => e.type === filter)),
    [edges, filter],
  );

  const degrees = useMemo(() => {
    const m = new Map<string, number>();
    for (const e of edges) {
      m.set(e.from, (m.get(e.from) ?? 0) + 1);
      m.set(e.to, (m.get(e.to) ?? 0) + 1);
    }
    return [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, 20);
  }, [edges]);

  const maxDeg = degrees[0]?.[1] ?? 1;

  return (
    <div className="flex flex-col gap-6">
      <SectionTitle>Relation graph</SectionTitle>
      <p className="max-w-3xl text-[13px] leading-6 text-zinc-400">
        {edges.length} edges found by matching each paper&rsquo;s reference block
        against the rest of the corpus, then reading the citing sentence to decide
        what kind of relation it actually is. Most cited pairs here are genuine
        baselines rather than passing citations.
      </p>

      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => setFilter("all")}
          className={`rounded-md border px-2.5 py-1 font-mono text-[10px] ${
            filter === "all"
              ? "border-rose-700/60 bg-rose-600/20 text-rose-300"
              : "border-zinc-800 text-zinc-500"
          }`}
        >
          all ({edges.length})
        </button>
        {byType.map(([t, n]) => (
          <button
            key={t}
            onClick={() => setFilter(filter === t ? "all" : t)}
            className={`rounded-md border px-2.5 py-1 font-mono text-[10px] ${
              filter === t
                ? "border-rose-700/60 bg-rose-600/20 text-rose-300"
                : "border-zinc-800 text-zinc-500"
            }`}
          >
            {REL_LABEL[t]} ({n})
          </button>
        ))}
      </div>

      <Card>
        <div className="flex flex-col gap-2">
          <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">
            most connected papers
          </div>
          {degrees.map(([id, deg]) => {
            const p = PAPER_BY_ID[id];
            if (!p) return null;
            return (
              <div key={id} className="flex items-center gap-3">
                <div
                  className="h-1.5 shrink-0 rounded-full bg-rose-500/70"
                  style={{ width: `${Math.max(6, (deg / maxDeg) * 220)}px` }}
                />
                <span className="font-mono text-[10px] text-zinc-500">{deg}</span>
                <span className="truncate text-[11px] text-zinc-400">{p.title}</span>
              </div>
            );
          })}
        </div>
      </Card>

      <div className="grid gap-3 lg:grid-cols-2">
        {shown.slice(0, 80).map((e, i) => {
          const a = PAPER_BY_ID[e.from];
          const b = PAPER_BY_ID[e.to];
          if (!a || !b) return null;
          return (
            <Card key={i} className="flex flex-col gap-2">
              <div className="flex flex-wrap items-center gap-2">
                <NeuroPaperLink id={a.id} short link />
                <Badge tone={REL_TONE[e.type]}>{REL_LABEL[e.type]}</Badge>
                <NeuroPaperLink id={b.id} short link />
              </div>
              <div className="text-[12px] leading-5 text-zinc-300">{a.title}</div>
              <div className="text-[11px] leading-5 text-zinc-500">{b.title}</div>
              {e.note ? (
                <div className="font-mono text-[10px] text-zinc-600">{e.note}</div>
              ) : null}
            </Card>
          );
        })}
      </div>

      {shown.length > 80 ? (
        <div className="font-mono text-[10px] text-zinc-600">
          showing 80 of {shown.length} edges
        </div>
      ) : null}

      <Card>
        <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">
          two most-cited papers side by side
        </div>
        <div className="mt-2 grid gap-3 sm:grid-cols-2">
          {PAPERS.slice(0, 2).map((p) => (
            <div key={p.id} className="flex flex-col gap-2">
              <div className="flex items-center gap-2">
                <CiteBadge n={p.citations} />
                <span className="text-[11px] text-zinc-300">{p.title}</span>
              </div>
              <ArchitectureDiagram architecture={p.architecture} compact />
              {p.summary ? <Quote>{p.summary}</Quote> : null}
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}