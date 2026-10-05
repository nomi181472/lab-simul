"use client";

import { useMemo, useState } from "react";
import { METRICS } from "@/labs/wifi-sensing/data/metrics";
import { PAPERS } from "@/labs/wifi-sensing/data/papers";
import { useWifiLab } from "@/labs/wifi-sensing/context";
import { Badge, Card, Formula, SectionTitle, WifiPaperLink } from "@/labs/wifi-sensing/ui";

export function MetricsView() {
  const { setMathSim, setSection } = useWifiLab();
  const [open, setOpen] = useState<string | null>(METRICS[0]?.id ?? null);

  const usage = useMemo(() => {
    const m = new Map<string, number>();
    for (const p of PAPERS) {
      const set = new Set(p.metrics);
      for (const k of set) m.set(k, (m.get(k) ?? 0) + 1);
    }
    return m;
  }, []);

  const families = useMemo(() => {
    const m = new Map<string, typeof METRICS>();
    for (const x of METRICS) {
      const arr = m.get(x.family) ?? [];
      arr.push(x);
      m.set(x.family, arr);
    }
    return [...m.entries()];
  }, []);

  return (
    <div className="space-y-5">
      <SectionTitle>Metrics</SectionTitle>
      <p className="max-w-3xl text-[12px] leading-6 text-zinc-500">
        {METRICS.length} metrics with their definitions, why each one is easy to misreport, and the
        corpus papers whose text actually contains them. The failure mode in this field is not
        usually a wrong number — it is the right number of the wrong thing (accuracy on imbalanced
        classes, mean instead of median when outliers dominate).
      </p>

      {families.map(([family, items]) => (
        <section key={family} className="space-y-2">
          <div className="font-mono text-[10px] uppercase tracking-wide text-cyan-500">
            {family} metrics ({items.length})
          </div>
          <div className="grid gap-2 lg:grid-cols-2">
            {items.map((x) => {
              const isOpen = open === x.id;
              const n = usage.get(x.id) ?? 0;
              return (
                <Card key={x.id} tone={isOpen ? "accent" : "default"}>
                  <button
                    onClick={() => setOpen(isOpen ? null : x.id)}
                    className="block w-full text-left"
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-[12px] font-semibold text-zinc-100">{x.name}</span>
                      <Badge tone={n > 30 ? "emerald" : n > 5 ? "cyan" : "zinc"}>
                        {n} papers
                      </Badge>
                      {x.simulator ? (
                        <span className="ml-auto font-mono text-[9px] text-cyan-500">
                          interactive →
                        </span>
                      ) : null}
                    </div>
                  </button>

                  {isOpen ? (
                    <div className="mt-2 space-y-2">
                      <Formula>{x.formula}</Formula>
                      <div>
                        <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-500">
                          meaning
                        </div>
                        <p className="text-[11px] leading-5 text-zinc-400">{x.meaning}</p>
                      </div>
                      <div>
                        <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-500">
                          intuition
                        </div>
                        <p className="text-[11px] leading-5 text-zinc-400">{x.intuition}</p>
                      </div>
                      <div>
                        <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-500">
                          example
                        </div>
                        <p className="text-[11px] leading-5 text-zinc-400">{x.example}</p>
                      </div>
                      <div className="rounded-lg border border-amber-900/40 bg-amber-950/10 p-2">
                        <div className="font-mono text-[9px] uppercase tracking-wide text-amber-400">
                          how it gets misreported
                        </div>
                        <ul className="mt-1 space-y-0.5">
                          {x.limitations.map((l, i) => (
                            <li key={i} className="text-[11px] leading-5 text-zinc-400">
                              · {l}
                            </li>
                          ))}
                        </ul>
                      </div>
                      {x.simulator ? (
                        <button
                          onClick={() => {
                            setMathSim(x.simulator!);
                            setSection("math");
                          }}
                          className="rounded-md border border-cyan-800 px-2 py-1 font-mono text-[10px] text-cyan-300 hover:bg-cyan-950"
                        >
                          see it move →
                        </button>
                      ) : null}
                      {x.paperIds.length ? (
                        <div>
                          <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-500">
                            reported by
                          </div>
                          <div className="mt-1 flex flex-wrap gap-1">
                            {x.paperIds.map((pid) => (
                              <WifiPaperLink key={pid} id={pid} short link />
                            ))}
                          </div>
                        </div>
                      ) : (
                        <Badge tone="zinc">no corpus paper mentions this term</Badge>
                      )}
                    </div>
                  ) : null}
                </Card>
              );
            })}
          </div>
        </section>
      ))}
    </div>
  );
}