"use client";

import { CONCEPTS } from "@/labs/object-tracking/data/concepts";
import { PAPERS } from "@/labs/object-tracking/data/papers";
import { Card, Badge } from "@/labs/object-tracking/ui";
import { useTrackingLab } from "@/labs/object-tracking/context";

const CATEGORY_META: Record<string, { color: string; theme: string }> = {
  basics: { color: "emerald", theme: "What everything assumes" },
  paradigm: { color: "violet", theme: "Which game are we playing" },
  motion: { color: "sky", theme: "Where will the box be" },
  association: { color: "amber", theme: "Which box belongs to whom" },
  appearance: { color: "rose", theme: "What does it look like" },
  architecture: { color: "cyan", theme: "How the network is shaped" },
  evaluation: { color: "zinc", theme: "How we know it worked" },
};

export function FoundationsView() {
  const { setMathSim } = useTrackingLab();
  const cats = [...new Set(CONCEPTS.map((c) => c.category))];
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Foundations</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          Prerequisite concepts grouped by the question they answer. Counts show how many
          structured papers rely on each concept.
        </p>
      </div>
      {cats.map((cat) => (
        <section key={cat}>
          <h2 className="mb-2 font-mono text-[11px] uppercase tracking-widest text-zinc-500">
            {cat} — {(CATEGORY_META[cat] ?? { theme: "" }).theme}
          </h2>
          <div className="grid gap-2 md:grid-cols-2 lg:grid-cols-3">
            {CONCEPTS.filter((c) => c.category === cat).map((c) => {
              const users = PAPERS.filter((p) => p.background.includes(c.id) || p.concepts.includes(c.id));
              return (
                <Card key={c.id} className="!py-3 min-w-0">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="text-[13px] font-semibold text-zinc-100 truncate">{c.name}</span>
                  </div>
                  <p className="mt-1 text-[12px] leading-5 text-zinc-400">{c.intuition}</p>
                  {c.formula && (
                    <div className="mt-2 overflow-x-auto rounded bg-black/40 px-2 py-1 font-mono text-[11px] text-amber-200 whitespace-nowrap">{c.formula}</div>
                  )}
                  <div className="mt-2 flex flex-wrap items-center gap-1.5">
                    <Badge tone="zinc">{users.length} papers</Badge>
                    {c.simulator && (
                      <button onClick={() => setMathSim(c.simulator!)} className="rounded border border-sky-800 px-1.5 py-0.5 font-mono text-[10px] text-sky-300 hover:bg-sky-950/40">simulate →</button>
                    )}
                  </div>
                  {c.prereqs.length > 0 && (
                    <p className="mt-1 font-mono text-[10px] text-zinc-600 truncate">needs: {c.prereqs.join(" · ")}</p>
                  )}
                </Card>
              );
            })}
          </div>
        </section>
      ))}
    </div>
  );
}
