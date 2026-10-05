"use client";

import { useState } from "react";
import { CONCEPTS } from "@/labs/wifi-sensing/data/concepts";
import { useWifiLab } from "@/labs/wifi-sensing/context";
import { Badge, Card, Formula, SectionTitle, WifiPaperLink } from "@/labs/wifi-sensing/ui";
import type { ConceptCategory } from "@/labs/wifi-sensing/data/types";

const CATEGORY_LABEL: Record<ConceptCategory, string> = {
  propagation: "Propagation",
  signal: "Signal representation",
  features: "Features",
  estimation: "Estimation",
  learning: "Learning",
  tasks: "Tasks",
  evaluation: "Evaluation",
};

const CATEGORY_ORDER: ConceptCategory[] = [
  "propagation",
  "signal",
  "features",
  "estimation",
  "learning",
  "tasks",
  "evaluation",
];

export function FoundationsView() {
  const { setMathSim, setSection } = useWifiLab();
  const [open, setOpen] = useState<string | null>(CONCEPTS[0]?.id ?? null);

  return (
    <div className="space-y-5">
      <SectionTitle>Foundations</SectionTitle>
      <p className="max-w-3xl text-[12px] leading-6 text-zinc-500">
        {CONCEPTS.length} concepts a reader needs before the papers make sense, ordered by category
        rather than difficulty. Each is defined plainly, carries its standard formula where one
        exists, links to its prerequisites, and lists the corpus papers that actually use it. The
        definitions are standard terminology — the paper links are the evidence that the term is
        load-bearing here, not that the definition came from any one of them.
      </p>

      {CATEGORY_ORDER.map((cat) => {
        const items = CONCEPTS.filter((c) => c.category === cat);
        if (!items.length) return null;
        return (
          <section key={cat} className="space-y-2">
            <div className="font-mono text-[10px] uppercase tracking-wide text-cyan-500">
              {CATEGORY_LABEL[cat]} ({items.length})
            </div>
            <div className="grid gap-2 lg:grid-cols-2">
              {items.map((c) => {
                const isOpen = open === c.id;
                return (
                  <Card key={c.id} tone={isOpen ? "accent" : "default"}>
                    <button
                      onClick={() => setOpen(isOpen ? null : c.id)}
                      className="block w-full text-left"
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-[12px] font-semibold text-zinc-100">{c.name}</span>
                        {c.simulator ? (
                          <span className="rounded bg-zinc-800 px-1 py-0.5 font-mono text-[9px] text-cyan-400">
                            interactive
                          </span>
                        ) : null}
                        <span className="ml-auto font-mono text-[10px] text-zinc-600">
                          {c.paperIds.length} papers
                        </span>
                      </div>
                    </button>

                    {c.prereqs.length ? (
                      <div className="mt-1 flex flex-wrap items-center gap-1">
                        <span className="font-mono text-[9px] text-zinc-600">needs</span>
                        {c.prereqs.map((p) => {
                          const pr = CONCEPTS.find((x) => x.id === p);
                          return (
                            <button
                              key={p}
                              onClick={() => setOpen(p)}
                              className="rounded bg-zinc-800 px-1 py-0.5 font-mono text-[9px] text-zinc-400 hover:bg-zinc-700"
                            >
                              {pr?.name ?? p}
                            </button>
                          );
                        })}
                      </div>
                    ) : null}

                    {isOpen ? (
                      <div className="mt-2 space-y-2">
                        {c.formula ? <Formula>{c.formula}</Formula> : null}
                        <p className="text-[11px] leading-5 text-zinc-400">{c.intuition}</p>
                        {c.simulator ? (
                          <button
                            onClick={() => {
                              setMathSim(c.simulator!);
                              setSection("math");
                            }}
                            className="rounded-md border border-cyan-800 px-2 py-1 font-mono text-[10px] text-cyan-300 hover:bg-cyan-950"
                          >
                            open the simulator →
                          </button>
                        ) : null}
                        {c.paperIds.length ? (
                          <div>
                            <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-500">
                              used by
                            </div>
                            <div className="mt-1 flex flex-wrap gap-1">
                              {c.paperIds.map((pid) => (
                                <WifiPaperLink key={pid} id={pid} short link />
                              ))}
                            </div>
                          </div>
                        ) : (
                          <Badge tone="zinc">no corpus paper matched this term</Badge>
                        )}
                      </div>
                    ) : null}
                  </Card>
                );
              })}
            </div>
          </section>
        );
      })}
    </div>
  );
}