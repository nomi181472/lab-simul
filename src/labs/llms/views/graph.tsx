"use client";

import { useMemo, useState } from "react";
import { Card, SectionTitle, Badge, TfPaperLink } from "@/labs/llms/ui";
import { PAPERS } from "@/labs/llms/data/papers";
import { CONCEPTS } from "@/labs/llms/data/concepts";
import { ALL_CHANGE_OPERATORS } from "@/labs/llms/data/vocab";

type Node = { id: string; label: string; kind: "concept" | "operator"; n: number };

/**
 * Concept and operator co-occurrence.
 *
 * This is not a citation graph. It is a bipartite projection of which ideas
 * appear together in the corpus, which answers a different question the citation
 * graph cannot: whether the efficient-LLM work and the scaling work are actually
 * in the same papers, or merely in the same field.
 */
export function GraphView() {
  const [focus, setFocus] = useState<string | null>(null);

  const { nodes, edges, byId } = useMemo(() => {
    const nodes: Node[] = [];
    const byId = new Map<string, Node>();
    for (const c of CONCEPTS) {
      const n: Node = { id: `c:${c.key}`, label: c.label, kind: "concept", n: c.papers.length };
      nodes.push(n);
      byId.set(n.id, n);
    }
    for (const op of ALL_CHANGE_OPERATORS) {
      const n = PAPERS.filter((p) => p.changes.some((c) => c.operator === op)).length;
      if (!n) continue;
      const node: Node = { id: `o:${op}`, label: op, kind: "operator", n };
      nodes.push(node);
      byId.set(node.id, node);
    }

    // concept <-> operator edges, weighted by shared papers
    const pairPapers = new Map<string, Set<string>>();
    for (const c of CONCEPTS) {
      const cs = new Set(c.papers);
      for (const op of ALL_CHANGE_OPERATORS) {
        const shared = PAPERS.filter(
          (p) => cs.has(p.id) && p.changes.some((ch) => ch.operator === op),
        ).map((p) => p.id);
        if (shared.length >= 2) {
          pairPapers.set(`${c.key}|${op}`, new Set(shared));
        }
      }
    }
    const edges = [...pairPapers.entries()]
      .map(([k, set]) => {
        const [ck, op] = k.split("|");
        return { a: `c:${ck}`, b: `o:${op}`, w: set.size, papers: [...set] };
      })
      .sort((x, y) => y.w - x.w);

    return { nodes, edges, byId };
  }, []);

  const maxW = edges[0]?.w ?? 1;
  const visible = focus ? edges.filter((e) => e.a === focus || e.b === focus) : edges;
  const top = [...visible].sort((a, b) => b.w - a.w).slice(0, 30);

  return (
    <div className="space-y-6">
      <SectionTitle>Graph</SectionTitle>

      <Card tone="accent">
        <p className="text-[13px] leading-6 text-zinc-400">
          Concepts on one side, modification operators on the other; an edge means at
          least two papers mention both. The heaviest edges are where the field actually
          couples ideas — for example whether <em>edge deployment</em> in practice also
          means <em>quantization</em>, or something else.
        </p>
      </Card>

      <Card>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="font-mono text-[10px] text-zinc-500">
            {nodes.length} nodes · {edges.length} edges (weight ≥ 2 papers)
          </span>
          {focus && (
            <button
              onClick={() => setFocus(null)}
              className="rounded-md border border-cyan-700/50 bg-cyan-600/20 px-2.5 py-1 text-[11px] text-cyan-300"
            >
              clear focus: {byId.get(focus)?.label ?? focus}
            </button>
          )}
        </div>
      </Card>

      <Card>
        <SectionTitle>Heaviest couplings</SectionTitle>
        <div className="mt-3 space-y-1.5">
          {top.map((e) => {
            const a = byId.get(e.a);
            const b = byId.get(e.b);
            if (!a || !b) return null;
            const dim = Math.round((e.w / maxW) * 100);
            return (
              <button
                key={`${e.a}-${e.b}`}
                onClick={() => setFocus(focus === e.a ? null : e.a)}
                className="flex w-full items-center gap-3 rounded-lg px-2 py-1.5 text-left hover:bg-zinc-900/60"
              >
                <span
                  className={`shrink-0 rounded px-1.5 py-0.5 font-mono text-[10px] ${
                    a.kind === "concept"
                      ? "bg-cyan-950 text-cyan-300"
                      : "bg-amber-950 text-amber-300"
                  }`}
                >
                  {a.label}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block h-1.5 overflow-hidden rounded-full bg-zinc-800">
                    <span
                      className="block h-full rounded-full bg-zinc-500"
                      style={{ width: `${dim}%` }}
                    />
                  </span>
                </span>
                <span
                  className={`shrink-0 rounded px-1.5 py-0.5 font-mono text-[10px] ${
                    b.kind === "concept"
                      ? "bg-cyan-950 text-cyan-300"
                      : "bg-amber-950 text-amber-300"
                  }`}
                >
                  {b.label}
                </span>
                <span className="w-8 shrink-0 text-right font-mono text-[10px] text-zinc-500">
                  {e.w}
                </span>
              </button>
            );
          })}
        </div>
      </Card>

      {focus && (
        <Card tone="accent">
          <SectionTitle>Papers behind the strongest edge</SectionTitle>
          <div className="mt-3 flex flex-wrap gap-1.5">
            {(top[0]?.papers ?? []).map((id) => (
              <span
                key={id}
                className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[10px] text-zinc-300"
              >
                {id}
              </span>
            ))}
          </div>
          <p className="mt-2 font-mono text-[10px] text-zinc-600">
            strongest edge for {byId.get(focus)?.label}
          </p>
        </Card>
      )}

      <Card>
        <SectionTitle>All nodes</SectionTitle>
        <div className="mt-3 grid gap-4 sm:grid-cols-2">
          <div>
            <p className="font-mono text-[10px] uppercase tracking-wide text-cyan-400">
              concepts
            </p>
            <div className="mt-2 flex flex-wrap gap-1">
              {CONCEPTS.map((c) => (
                <span
                  key={c.key}
                  className="rounded bg-cyan-950 px-1.5 py-0.5 font-mono text-[10px] text-cyan-300"
                >
                  {c.label} <span className="text-cyan-500">{c.papers.length}</span>
                </span>
              ))}
            </div>
          </div>
          <div>
            <p className="font-mono text-[10px] uppercase tracking-wide text-amber-400">
              operators
            </p>
            <div className="mt-2 flex flex-wrap gap-1">
              {nodes
                .filter((n) => n.kind === "operator")
                .map((n) => (
                  <span
                    key={n.id}
                    className="rounded bg-amber-950 px-1.5 py-0.5 font-mono text-[10px] text-amber-300"
                  >
                    {n.label} <span className="text-amber-500">{n.n}</span>
                  </span>
                ))}
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}