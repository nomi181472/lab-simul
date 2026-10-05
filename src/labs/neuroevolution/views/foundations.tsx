"use client";

import { useMemo, useState } from "react";
import { PAPERS } from "@/labs/neuroevolution/data/papers";
import { CONCEPTS, CONCEPT_BY_ID } from "@/labs/neuroevolution/data/concepts";
import { Card, SectionTitle, Badge, Formula, NeuroPaperLink } from "@/labs/neuroevolution/ui";
import { SIMULATORS } from "@/labs/neuroevolution/sims";

/* Curated concept reference. Background material written for this lab, kept
 * separate from paper extracts — the corpus links each concept to its papers. */

/* ------------------------------ foundations ------------------------------ */

export function FoundationsView() {
  const cats = useMemo(() => {
    const set = new Set<string>(CONCEPTS.map((c) => c.category));
    return ["all", ...[...set].sort()];
  }, []);
  const [active, setActive] = useState<string>("all");

  const list = useMemo(
    () => (active === "all" ? CONCEPTS : CONCEPTS.filter((c) => c.category === active)),
    [active],
  );

  return (
    <div className="flex flex-col gap-6">
      <SectionTitle>Foundations</SectionTitle>
      <div className="flex flex-wrap gap-2">
        {cats.map((c) => (
          <button
            key={c}
            onClick={() => setActive(c)}
            className={`rounded-md border px-2.5 py-1 font-mono text-[10px] ${
              active === c
                ? "border-rose-700/60 bg-rose-600/20 text-rose-300"
                : "border-zinc-800 text-zinc-500"
            }`}
          >
            {c}
          </button>
        ))}
      </div>
      <div className="grid gap-3 lg:grid-cols-2">
        {list.map((c) => (
          <Card key={c.id} className="flex flex-col gap-2.5">
            <div className="flex items-center justify-between gap-2">
              <span className="text-[13px] font-semibold text-zinc-100">{c.name}</span>
              <Badge tone="violet">{c.category}</Badge>
            </div>
            <div className="text-[12px] leading-5 text-zinc-300">{c.intuition}</div>
            {c.formula ? <Formula>{c.formula}</Formula> : null}
            {c.variables && c.variables.length > 0 ? (
              <div className="flex flex-col gap-1">
                {c.variables.map((v) => (
                  <div key={v.symbol} className="font-mono text-[10px] text-zinc-500">
                    <span className="text-amber-300">{v.symbol}</span> — {v.meaning}
                  </div>
                ))}
              </div>
            ) : null}
            {c.prereqs.length > 0 ? (
              <div className="font-mono text-[10px] text-zinc-600">
                needs: {c.prereqs.map((p) => CONCEPT_BY_ID[p]?.name ?? p).join(", ")}
              </div>
            ) : null}
            {c.simulator ? (
              <details className="rounded-lg border border-zinc-800 bg-zinc-950/50 px-3 py-2">
                <summary className="cursor-pointer font-mono text-[10px] text-zinc-500">
                  run the simulator
                </summary>
                <div className="mt-2">
                  <Simulators />
                </div>
              </details>
            ) : null}
            {c.paperIds && c.paperIds.length > 0 ? (
              <div className="flex flex-wrap gap-1 border-t border-zinc-800 pt-2">
                {c.paperIds.slice(0, 10).map((id) => (
                  <NeuroPaperLink key={id} id={id} short link />
                ))}
              </div>
            ) : null}
          </Card>
        ))}
      </div>
    </div>
  );
}

function Simulators() {
  const [sim, setSim] = useState<string>(CONCEPTS.find((c) => c.simulator)?.simulator ?? "ga-demo");
  const Sim = SIMULATORS[sim as keyof typeof SIMULATORS];
  if (!Sim) return null;
  return (
    <div className="flex flex-col gap-2">
      <select
        value={sim}
        onChange={(e) => setSim(e.target.value)}
        className="rounded-md border border-zinc-800 bg-zinc-900 px-2 py-1 font-mono text-[10px] text-zinc-300"
      >
        {Object.keys(SIMULATORS).map((k) => (
          <option key={k} value={k}>
            {k}
          </option>
        ))}
      </select>
      <Sim />
    </div>
  );
}
