"use client";
import { useState } from "react";
import { Card, SectionTitle, Badge, Quote, PaperLink } from "@/labs/networking/ui";
import { ANALYSIS } from "@/labs/networking/data/analysis";
import { SIMS, SIM_META } from "@/labs/networking/sims/registry";
import { useNetworking } from "@/labs/networking/context";

export function ClustersView() {
  const { setSection } = useNetworking();
  const [openId, setOpenId] = useState<string | null>(null);

  return (
    <div className="space-y-6">
      <SectionTitle>Identical-problem clusters</SectionTitle>

      <Card tone="accent">
        <p className="text-[13px] leading-6 text-zinc-400">
          Domains where ≥3 papers state the same problem: the earliest statement becomes
          the cluster core, every later statement is a repeated attempt against it. Each
          cluster with an implementable model links to its simulator.
        </p>
      </Card>

      <div className="space-y-3">
        {ANALYSIS.clusters.map((c) => {
          const open = openId === c.clusterId;
          const sim = SIMS[c.simulatorKey ?? ""];
          return (
            <Card key={c.clusterId} className={open ? "border-sky-800/60" : ""}>
              <button
                className="flex w-full flex-wrap items-baseline justify-between gap-2 text-left"
                onClick={() => setOpenId(open ? null : c.clusterId)}
              >
                <span className="text-[13px] font-medium text-zinc-100">{c.label}</span>
                <div className="flex items-center gap-2">
                  <Badge tone={c.attemptCount >= 6 ? "rose" : "amber"}>
                    {c.attemptCount} attempts
                  </Badge>
                  <span className="font-mono text-[10px] text-zinc-500">
                    {c.spanYears[0]}–{c.spanYears[1]}
                  </span>
                </div>
              </button>

              {open && (
                <div className="mt-3 space-y-3 border-t border-zinc-800 pt-3">
                  <div>
                    <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
                      core problem (earliest statement)
                    </p>
                    <div className="mt-1">
                      <PaperLink id={c.corePaperId} />
                    </div>
                    <Quote>{c.coreStatement}</Quote>
                  </div>

                  <div>
                    <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
                      repeated attempts
                    </p>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {c.repeatPaperIds.map((id) => (
                        <PaperLink key={id} id={id} short />
                      ))}
                    </div>
                  </div>

                  {sim ? (
                    <button
                      onClick={() => setSection("simulators")}
                      className="rounded-md border border-sky-700/50 bg-sky-600/20 px-2 py-1 font-mono text-[10px] text-sky-300 hover:bg-sky-600/30"
                    >
                      → open simulator:{" "}
                      {c.simulatorKey
                        ? (SIM_META[c.simulatorKey as keyof typeof SIM_META]?.label ??
                          c.simulatorKey)
                        : ""}
                    </button>
                  ) : (
                    <p className="font-mono text-[10px] text-zinc-600">
                      no implementable simulator bound to this cluster yet
                    </p>
                  )}
                </div>
              )}
            </Card>
          );
        })}
      </div>
    </div>
  );
}
