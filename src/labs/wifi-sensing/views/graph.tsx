"use client";

import { useMemo, useState } from "react";
import { PAPERS } from "@/labs/wifi-sensing/data/papers";
import { Badge, Card, SectionTitle, WifiPaperLink } from "@/labs/wifi-sensing/ui";
import type { RelationType } from "@/labs/wifi-sensing/data/types";

const REL_TONE: Record<string, "cyan" | "amber" | "emerald" | "rose" | "violet" | "zinc"> = {
  "builds-on": "cyan",
  improves: "emerald",
  replaces: "rose",
  extends: "cyan",
  combines: "violet",
  "uses-as-baseline": "amber",
  "addresses-limitation": "rose",
  "conceptual-successor": "zinc",
};

export function GraphView() {
  const [relFilter, setRelFilter] = useState<RelationType | "all">("all");

  const edges = useMemo(
    () =>
      PAPERS.flatMap((p) =>
        p.relations.map((r) => ({ from: p.id, to: r.to, type: r.type as RelationType, note: r.note })),
      ),
    [],
  );

  const filtered = relFilter === "all" ? edges : edges.filter((e) => e.type === relFilter);

  const degrees = useMemo(() => {
    const d = new Map<string, { cited: number; citing: number }>();
    for (const p of PAPERS) d.set(p.id, { cited: 0, citing: 0 });
    for (const e of edges) {
      const a = d.get(e.from);
      const b = d.get(e.to);
      if (a) a.cited += 1;
      if (b) b.citing += 1;
    }
    return d;
  }, [edges]);

  const top = useMemo(
    () =>
      [...degrees.entries()]
        .map(([id, v]) => ({ id, ...v, total: v.cited + v.citing, paper: PAPERS.find((p) => p.id === id)! }))
        .filter((x) => x.paper && x.total > 0)
        .sort((a, b) => b.total - a.total),
    [degrees],
  );

  const incoming = useMemo(() => {
    const m = new Map<string, string[]>();
    for (const e of edges) {
      const arr = m.get(e.to) ?? [];
      arr.push(e.from);
      m.set(e.to, arr);
    }
    return m;
  }, [edges]);

  const types = [...new Set(edges.map((e) => e.type))] as RelationType[];

  return (
    <div className="space-y-5">
      <SectionTitle>Research graph</SectionTitle>
      <p className="max-w-3xl text-[12px] leading-6 text-zinc-500">
        {edges.length} citation edges detected inside the corpus: for each paper, another corpus
        paper was matched inside its own reference block. The relation kind is read from the
        sentence containing the citation — a paper compared against as a baseline is marked as
        one, not guessed. This is a sparse graph, not a citation-complete one.
      </p>

      <div className="flex flex-wrap items-center gap-1.5">
        <button
          onClick={() => setRelFilter("all")}
          className={`rounded-md border px-2 py-1 font-mono text-[10px] ${
            relFilter === "all"
              ? "border-cyan-600 bg-cyan-600/20 text-cyan-200"
              : "border-zinc-800 text-zinc-500"
          }`}
        >
          all ({edges.length})
        </button>
        {types.map((t) => (
          <button
            key={t}
            onClick={() => setRelFilter(t)}
            className={`rounded-md border px-2 py-1 font-mono text-[10px] ${
              relFilter === t
                ? "border-cyan-600 bg-cyan-600/20 text-cyan-200"
                : "border-zinc-800 text-zinc-500"
            }`}
          >
            {t} ({edges.filter((e) => e.type === t).length})
          </button>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            most connected papers
          </div>
          <p className="mt-1 text-[10px] leading-4 text-zinc-600">
            cited by other corpus papers (left) and citing other corpus papers (right).
          </p>
          <div className="mt-2 space-y-1">
            {top.slice(0, 18).map((x) => (
              <div key={x.id} className="flex items-center gap-2">
                <WifiPaperLink id={x.id} short link />
                <span className="min-w-0 flex-1 truncate text-[11px] text-zinc-500">
                  {x.paper.title}
                </span>
                <span className="flex shrink-0 gap-1.5 font-mono text-[10px]">
                  <span className="text-cyan-400" title="cited by corpus papers">
                    ↓{x.cited}
                  </span>
                  <span className="text-zinc-500" title="cites corpus papers">
                    ↑{x.citing}
                  </span>
                </span>
              </div>
            ))}
          </div>
        </Card>

        <Card>
          <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            edges ({filtered.length})
          </div>
          <div className="mt-2 max-h-[60vh] space-y-1 overflow-y-auto pr-1">
            {filtered.map((e, i) => (
              <div key={i} className="flex items-center gap-1.5">
                <WifiPaperLink id={e.from} short link />
                <span className="text-zinc-600">→</span>
                <WifiPaperLink id={e.to} short link />
                <Badge tone={REL_TONE[e.type] ?? "zinc"}>{e.type}</Badge>
              </div>
            ))}
            {filtered.length === 0 ? (
              <p className="py-6 text-center text-[11px] text-zinc-600">no edges of this kind</p>
            ) : null}
          </div>
        </Card>
      </div>

      {top.length ? (
        <Card tone="accent">
          <div className="font-mono text-[10px] uppercase tracking-wide text-cyan-400">
            lineage of the most-connected paper
          </div>
          <Lineage id={top[0].id} incoming={incoming} depth={0} seen={new Set()} />
        </Card>
      ) : null}
    </div>
  );
}

function Lineage({
  id,
  incoming,
  depth,
  seen,
}: {
  id: string;
  incoming: Map<string, string[]>;
  depth: number;
  seen: Set<string>;
}) {
  if (depth > 3 || seen.has(id)) return null;
  const next = new Set(seen);
  next.add(id);
  const parents = incoming.get(id) ?? [];
  const p = PAPERS.find((x) => x.id === id);
  if (!p) return null;
  return (
    <div style={{ marginLeft: depth ? 14 : 0 }} className="mt-1.5">
      <div className="flex items-center gap-2">
        <WifiPaperLink id={id} link />
        <span className="min-w-0 flex-1 truncate text-[11px] text-zinc-400">{p.title}</span>
        <span className="shrink-0 font-mono text-[10px] text-zinc-600">{p.year}</span>
      </div>
      {parents.length === 0 ? (
        <div className="ml-1 mt-0.5 font-mono text-[10px] text-zinc-600">
          no earlier corpus paper detected in its references
        </div>
      ) : (
        parents
          .slice(0, 4)
          .map((par) => (
            <div key={par} className="ml-3 border-l border-zinc-800 pl-2">
              <div className="font-mono text-[9px] text-zinc-600">cited by ↓</div>
              <Lineage id={par} incoming={incoming} depth={depth + 1} seen={next} />
            </div>
          ))
      )}
    </div>
  );
}