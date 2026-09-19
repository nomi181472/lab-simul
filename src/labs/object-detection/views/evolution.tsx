"use client";

import { useMemo, useState } from "react";
import { RESEARCH_EPISODES, EPISODE_BY_ID, LONG_TERM_THREADS } from "@/labs/object-detection/data/loops";
import { PAPER_BY_ID } from "@/labs/object-detection/data/papers";
import type { Evidence } from "@/labs/object-detection/data/types";
import { Card, Badge, EvidenceLine, SectionTitle } from "@/labs/object-detection/ui";

const PALETTE = [
  "#34d399", // emerald
  "#38bdf8", // sky
  "#a78bfa", // violet
  "#fbbf24", // amber
  "#fb7185", // rose
  "#22d3ee", // cyan
];

function cidY(s: string) {
  const e = EPISODE_BY_ID[s];
  return e ? e.year : 0;
}

export function EvolutionView() {
  const [selected, setSelected] = useState<string | null>(null);

  const byYear = useMemo(() => {
    const m: Record<number, string[]> = {};
    for (const e of RESEARCH_EPISODES) {
      (m[e.year] ??= []).push(e.id);
    }
    return m;
  }, []);

  const years = Object.keys(byYear).map(Number).sort((a, b) => a - b);

  const selEp = selected ? EPISODE_BY_ID[selected] : undefined;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Evolution map</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          Directed graph of research episodes. Nodes are episodes (a problem →
          an idea); edges are the forks and convergences the corpus actually
          documents. Click a node to inspect its evidence chain.
        </p>
      </div>

      {/* graph */}
      <Card>
        <div className="overflow-x-auto">
          <div className="min-w-[640px]">
            <div className="flex gap-6">
              {years.map((y) => (
                <div key={y} className="flex flex-1 flex-col gap-4">
                  <div className="text-center font-mono text-[10px] text-emerald-400">{y}</div>
                  {(byYear[y] ?? []).map((id) => {
                    const e = EPISODE_BY_ID[id];
                    return (
                      <button
                        key={id}
                        onClick={() => setSelected(id)}
                        style={{ borderColor: PALETTE[cidY(id) % PALETTE.length] }}
                        className={`rounded-lg border bg-zinc-900/80 px-2 py-1.5 text-left text-[10px] leading-4 transition-colors ${
                          selected === id ? "ring-2 ring-emerald-400" : "hover:bg-zinc-800"
                        }`}
                      >
                        <div className="font-mono text-[9px] text-zinc-600">{id}</div>
                        <div className="text-zinc-200">{e.title}</div>
                      </button>
                    );
                  })}
                  {(byYear[y] ?? []).length === 0 && (
                    <div className="text-center text-[10px] text-zinc-700">—</div>
                  )}
                </div>
              ))}
            </div>
            <div className="mt-4 font-mono text-[10px] text-zinc-600">
              edges: {RESEARCH_EPISODES.reduce((a, e) => a + e.leadsTo.length, 0)} · nodes:{" "}
              {RESEARCH_EPISODES.length}
            </div>
          </div>
        </div>
      </Card>

      {/* edges list (legible fallback) */}
      <Card>
        <SectionTitle>Documented forks & convergence</SectionTitle>
        <div className="mt-3 space-y-1.5">
          {RESEARCH_EPISODES.filter((e) => e.leadsTo.length > 0).map((e) => (
            <div key={e.id} className="flex flex-wrap items-center gap-1.5 font-mono text-[11px]">
              <span className="text-zinc-300">{e.id}</span>
              <span className="text-zinc-600">→</span>
              {e.leadsTo.map((t) => (
                <span key={t} className="rounded bg-zinc-800 px-1.5 py-0.5 text-emerald-300">
                  {t}
                </span>
              ))}
              <span className="text-[10px] text-zinc-500">{e.title}</span>
            </div>
          ))}
        </div>
      </Card>

      {/* selected episode detail */}
      {selEp && (
        <Card tone="accent">
          <SectionTitle>
            {selEp.id} · {selEp.year} — {selEp.title}
          </SectionTitle>
          <div className="mt-3 grid gap-4 lg:grid-cols-2">
            <div className="space-y-3">
              <div>
                <Badge tone="amber">problem</Badge>
                <p className="mt-1 text-xs leading-5 text-zinc-300">{selEp.problem}</p>
              </div>
              <div>
                <Badge tone="rose">pressure</Badge>
                <p className="mt-1 text-xs leading-5 text-zinc-300">{selEp.pressure}</p>
              </div>
              <div>
                <Badge tone="emerald">idea</Badge>
                <p className="mt-1 text-xs leading-5 text-zinc-200">{selEp.idea}</p>
              </div>
              <div>
                <Badge tone="violet">mathematics</Badge>
                <ul className="mt-1 space-y-1">
                  {selEp.mathematics.map((m, i) => (
                    <li key={i} className="font-mono text-[10px] text-amber-200">
                      {m}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
            <div className="space-y-3">
              <div>
                <Badge tone="sky">architecture</Badge>
                <p className="mt-1 text-xs leading-5 text-zinc-300">{selEp.architecture}</p>
              </div>
              <div>
                <Badge tone="emerald">capability unlocked</Badge>
                <p className="mt-1 text-xs leading-5 text-zinc-200">{selEp.capability}</p>
              </div>
              <div>
                <Badge tone="amber">application</Badge>
                <p className="mt-1 text-xs leading-5 text-zinc-300">{selEp.application}</p>
              </div>
              <div>
                <Badge tone="rose">tradeoff</Badge>
                <p className="mt-1 text-xs leading-5 text-zinc-300">{selEp.tradeoff}</p>
              </div>
              <div>
                <Badge tone="sky">new problem</Badge>
                <p className="mt-1 text-xs leading-5 text-emerald-200">{selEp.newProblem}</p>
              </div>
            </div>
          </div>

          <div className="mt-3 flex flex-wrap gap-1.5">
            {selEp.paperIds.map((id) => {
              const p = PAPER_BY_ID[id];
              return p ? (
                <span key={id} className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-300">
                  {p.id} {p.shortTitle}
                </span>
              ) : (
                <span key={id} className="rounded bg-rose-900/50 px-1.5 py-0.5 text-[10px] text-rose-300">
                  ?{id}
                </span>
              );
            })}
          </div>

          <div className="mt-4">
            <EvidenceLine evidence={selEp.evidence as Evidence} />
          </div>
        </Card>
      )}

      {/* long-term threads */}
      <div>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-zinc-500">
          Long-term threads across 2015–2026
        </h2>
        <div className="grid gap-3 lg:grid-cols-2">
          {LONG_TERM_THREADS.map((t) => {
            const tone = t.status === "TRANSFORMED" ? "violet" : t.status === "PERSISTENT" ? "amber" : "sky";
            return (
              <Card key={t.id}>
                <div className="flex items-center justify-between gap-2">
                  <span className="text-sm font-semibold text-zinc-100">{t.name}</span>
                  <Badge tone={tone as "violet"}>{t.status}</Badge>
                </div>
                <ol className="mt-3 space-y-1.5">
                  {t.chain.map((c, i) => (
                    <li key={i} className="flex gap-2 text-[11px] leading-5 text-zinc-400">
                      <span className="font-mono text-zinc-600">{String(i + 1).padStart(2, "0")}</span>
                      {c}
                    </li>
                  ))}
                </ol>
                <p className="mt-3 border-t border-zinc-800 pt-2 text-[10px] text-zinc-500">
                  {t.evidenceNote}
                </p>
              </Card>
            );
          })}
        </div>
      </div>
    </div>
  );
}