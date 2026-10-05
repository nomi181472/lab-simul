"use client";

import { useMemo, useState } from "react";
import { PAPERS } from "@/labs/wifi-sensing/data/papers";
import { MANIFEST } from "@/labs/wifi-sensing/data/manifest";
import { TASK_LABEL } from "@/labs/wifi-sensing/data/types";
import { Card, SectionTitle, WifiPaperLink } from "@/labs/wifi-sensing/ui";

/* A "gap" is a limitation stated in the paper's own words, matched to the corpus
 * papers that later cite that paper. We never invent a gap: each row's text is a
 * sentence extracted from the source paper, and the successors column is built
 * from detected citation edges. */

export function GapsView() {
  const [task, setTask] = useState("all");

  const rows = useMemo(() => {
    const citing = new Map<string, string[]>();
    for (const p of PAPERS) {
      for (const r of p.relations) {
        const arr = citing.get(r.to) ?? [];
        arr.push(p.id);
        citing.set(r.to, arr);
      }
    }
    return PAPERS.filter((p) => p.limitations.authorStated.length > 0)
      .map((p) => ({
        p,
        gap: p.limitations.authorStated[0],
        others: p.limitations.authorStated.slice(1, 3),
        successors: (citing.get(p.id) ?? []).slice(0, 6),
      }))
      .sort((a, b) => b.successors.length - a.successors.length || b.p.citations - a.p.citations);
  }, []);

  const filtered = task === "all" ? rows : rows.filter((r) => r.p.task === task);
  const tasks = [...new Set(MANIFEST.map((m) => m.task))].sort();
  const unaddressed = rows.filter((r) => r.successors.length === 0).length;

  return (
    <div className="space-y-5">
      <SectionTitle>Research gaps</SectionTitle>
      <p className="max-w-3xl text-[12px] leading-6 text-zinc-500">
        {rows.length} papers state a limitation in their own text. Each entry quotes that
        sentence, then lists the corpus papers that go on to cite it — an open question that
        nothing in the corpus answers is itself the interesting result. {unaddressed} stated gaps
        have no detected successor in this corpus.
      </p>

      <div className="flex flex-wrap items-center gap-1.5">
        <button
          onClick={() => setTask("all")}
          className={`rounded-md border px-2.5 py-1 text-[11px] ${
            task === "all"
              ? "border-cyan-600 bg-cyan-600/20 text-cyan-200"
              : "border-zinc-800 text-zinc-500"
          }`}
        >
          all tasks ({rows.length})
        </button>
        {tasks.map((t) => {
          const n = rows.filter((r) => r.p.task === t).length;
          if (!n) return null;
          return (
            <button
              key={t}
              onClick={() => setTask(t)}
              className={`rounded-md border px-2.5 py-1 text-[11px] ${
                task === t
                  ? "border-cyan-600 bg-cyan-600/20 text-cyan-200"
                  : "border-zinc-800 text-zinc-500"
              }`}
            >
              {TASK_LABEL[t as keyof typeof TASK_LABEL] ?? t} ({n})
            </button>
          );
        })}
      </div>

      <div className="space-y-2">
        {filtered.map(({ p, gap, others, successors }) => (
          <Card key={p.id}>
            <div className="flex flex-wrap items-center gap-2">
              <WifiPaperLink id={p.id} link />
              <span className="font-mono text-[10px] text-zinc-600">
                {p.year} · {p.citations}c
              </span>
              <span className="rounded bg-zinc-800 px-1 py-0.5 font-mono text-[9px] text-zinc-400">
                {TASK_LABEL[p.task] ?? p.task}
              </span>
              {successors.length ? (
                <span className="ml-auto font-mono text-[10px] text-emerald-400">
                  {successors.length} successor{successors.length === 1 ? "" : "s"} in corpus
                </span>
              ) : (
                <span className="ml-auto font-mono text-[10px] text-rose-400">unaddressed here</span>
              )}
            </div>
            <p className="mt-1.5 text-[11px] leading-5 text-zinc-500">{p.title}</p>

            <div className="mt-2 rounded-lg border border-amber-900/40 bg-amber-950/10 p-2">
              <div className="font-mono text-[9px] uppercase tracking-wide text-amber-400">
                stated gap (quoted)
              </div>
              <p className="mt-0.5 text-[11px] leading-5 text-zinc-300">{gap}</p>
              {others.map((o, i) => (
                <p key={i} className="mt-1 text-[11px] leading-5 text-zinc-500">
                  {o}
                </p>
              ))}
            </div>

            {successors.length ? (
              <div className="mt-2">
                <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-500">
                  later corpus papers citing it
                </div>
                <div className="mt-1 flex flex-wrap gap-1.5">
                  {successors.map((s) => (
                    <WifiPaperLink key={s} id={s} short link />
                  ))}
                </div>
              </div>
            ) : null}
          </Card>
        ))}
        {filtered.length === 0 ? (
          <p className="py-10 text-center text-[12px] text-zinc-600">
            no paper in this task states a limitation in its retrieved text
          </p>
        ) : null}
      </div>
    </div>
  );
}