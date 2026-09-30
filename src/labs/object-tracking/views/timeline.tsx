"use client";

import { useMemo, useState } from "react";
import { PAPERS, successorsOf } from "@/labs/object-tracking/data/papers";
import { Card, Badge, DifficultyBadge, TaskBadge } from "@/labs/object-tracking/ui";

/* Evolution of ideas: papers grouped by year, each with its lineage edges. */

export function TimelineView() {
  const [task, setTask] = useState<string | null>(null);
  const byYear = useMemo(() => {
    const m: Record<number, typeof PAPERS> = {};
    for (const p of PAPERS.filter((p) => !task || p.task === task)) {
      (m[p.year] ??= []).push(p);
    }
    return Object.entries(m).sort((a, b) => Number(a[0]) - Number(b[0]));
  }, [task]);
  const tasks = [...new Set(PAPERS.map((p) => p.task))];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Research timeline</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          Not a list of cards — a lineage. Each paper names the corpus papers it builds on
          (<span className="font-mono text-sky-300">← predecessors</span>) and the ones that later
          built on it (<span className="font-mono text-emerald-300">successors →</span>), so you read
          history as <span className="text-zinc-200">“this paper exists because of these ideas.”</span>
        </p>
        <div className="mt-3 flex flex-wrap gap-1.5">
          <button onClick={() => setTask(null)} className={`rounded-full border px-2.5 py-1 text-[11px] ${task === null ? "border-sky-600 bg-sky-600/20 text-sky-200" : "border-zinc-800 text-zinc-500"}`}>all</button>
          {tasks.map((t) => (
            <button key={t} onClick={() => setTask(task === t ? null : t)} className={`rounded-full border px-2.5 py-1 text-[11px] ${task === t ? "border-sky-600 bg-sky-600/20 text-sky-200" : "border-zinc-800 text-zinc-500"}`}>{t}</button>
          ))}
        </div>
      </div>

      <div className="relative space-y-8 before:absolute before:bottom-0 before:left-[7px] before:top-0 before:w-px before:bg-sky-900/60 overflow-x-auto">
        {byYear.map(([year, papers]) => (
          <div key={year} className="relative pl-8 min-w-0">
            <div className="absolute left-0 top-1 h-4 w-4 rounded-full border-2 border-sky-600 bg-zinc-950 flex-shrink-0" />
            <div className="mb-2 font-mono text-sm font-bold text-sky-300 whitespace-nowrap">{year} <span className="font-normal text-zinc-600">· {papers.length} papers</span></div>
            <div className="grid gap-2 lg:grid-cols-2 min-w-0">
              {papers.sort((a, b) => a.id.localeCompare(b.id)).map((p) => {
                const succs = successorsOf(p.id);
                return (
                  <Card key={p.id} className="!py-3 min-w-0">
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="rounded bg-sky-950 px-1.5 py-0.5 font-mono text-[10px] text-sky-300 flex-shrink-0">{p.id}</span>
                      <DifficultyBadge d={p.difficulty} />
                      <TaskBadge t={p.task} />
                    </div>
                    <div className="mt-1.5 text-sm font-semibold text-zinc-100 truncate">{p.shortTitle}</div>
                    <div className="font-mono text-[10px] text-zinc-500 truncate">{p.title}</div>
                    <p className="mt-1.5 border-l-2 border-amber-700/60 pl-2 text-[12px] italic leading-5 text-zinc-400">
                      gap: {p.researchGap}
                    </p>
                    <div className="mt-2 space-y-1 font-mono text-[10px] leading-4">
                      {p.relations.length > 0 ? (
                        <div className="text-zinc-500">
                          <span className="text-sky-400">← builds on </span>
                          {p.relations.map((r) => `${r.to} (${r.type})`).join(" · ")}
                        </div>
                      ) : (
                        <div className="text-zinc-600">← no corpus predecessors (root of its thread)</div>
                      )}
                      {succs.length > 0 && (
                        <div className="text-zinc-500">
                          <span className="text-emerald-400">→ continued by </span>
                          {succs.map((s) => s.id).join(" · ")}
                        </div>
                      )}
                    </div>
                    <div className="mt-2 flex flex-wrap gap-1">
                      {p.tags.slice(0, 5).map((t) => <Badge key={t} tone="zinc">{t}</Badge>)}
                    </div>
                  </Card>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
