"use client";

import { SIM_REGISTRY, SIM_BY_ID } from "@/labs/object-detection/sims/registry";
import { SimErrorBoundary } from "@/labs/object-detection/sims/error-boundary";
import { useLab } from "@/labs/object-detection/context";
import { SOLUTIONS } from "@/labs/object-detection/data/solutions";
import { CLUSTERS } from "@/labs/object-detection/data/clusters";
import { PROBLEM_LIFECYCLES } from "@/labs/object-detection/data/problems";
import type { SolutionCategory } from "@/labs/object-detection/data/types";
import {
  Card,
  Badge,
  PaperLink,
  SectionTitle,
  checkSim,
  type BadgeTone,
} from "@/labs/object-detection/ui";

const CATEGORY_TONE: Record<SolutionCategory, BadgeTone> = {
  architecture: "sky",
  mathematics: "amber",
  training: "emerald",
  postprocessing: "violet",
  data: "rose",
};

const PROBLEM_LABEL: Record<string, string> = Object.fromEntries(
  PROBLEM_LIFECYCLES.map((p) => [p.id, p.shortLabel]),
);

export function SimulatorsView() {
  const { sim, setSim, setSection } = useLab();
  const active = checkSim(sim) ?? SIM_REGISTRY[0].id;
  const entry = SIM_BY_ID[active];
  const ActiveComp = entry.comp;

  const solutions = SOLUTIONS.filter((s) => s.simulator === active);
  const clusters = CLUSTERS.filter((c) => c.simulator === active);
  const noLinks = solutions.length === 0 && clusters.length === 0;

  return (
    <div className="space-y-6">
      <div>
        <SectionTitle>Simulators — {SIM_REGISTRY.length} interactive labs</SectionTitle>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          Deterministic, slider-driven kernels for the mechanisms the corpus keeps re-debating.
          Each sim states the research problem it stands for; the corpus links below it show which
          solutions and problem clusters in the 121-paper corpus wire into the same mechanism.
        </p>
      </div>

      <div className="space-y-2">
        <div className="flex gap-1.5 overflow-x-auto pb-1">
          {SIM_REGISTRY.map((s) => (
            <button
              key={s.id}
              onClick={() => setSim(s.id)}
              title={s.blurb}
              className={`shrink-0 rounded-md border px-2.5 py-1.5 font-mono text-[10px] uppercase tracking-wide transition-colors ${
                active === s.id
                  ? "border-emerald-700/60 bg-emerald-600/20 text-emerald-300"
                  : "border-zinc-800 bg-zinc-900/40 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200"
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap items-baseline gap-2">
          <Badge tone="emerald">{entry.label}</Badge>
          <span className="text-[12px] leading-5 text-zinc-500">{entry.blurb}</span>
        </div>
      </div>

      <SimErrorBoundary key={active} label={entry.label}>
        <ActiveComp />
      </SimErrorBoundary>

      <section>
        <div className="mb-2 flex flex-wrap items-baseline gap-2">
          <h3 className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            corpus links
          </h3>
          <span className="font-mono text-[10px] text-zinc-600">
            solutions &amp; clusters wired to “{entry.label}”
          </span>
        </div>

        {noLinks ? (
          <p className="font-mono text-[11px] text-zinc-600">
            No corpus entry links here yet.
          </p>
        ) : (
          <div className="grid gap-3 lg:grid-cols-2">
            <div className="space-y-3">
              <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-600">
                solutions ({solutions.length})
              </div>
              {solutions.length === 0 && (
                <p className="font-mono text-[11px] text-zinc-600">
                  No corpus entry links here yet.
                </p>
              )}
              {solutions.map((s) => (
                <Card key={s.id}>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-sm font-semibold text-zinc-100">{s.name}</span>
                    <Badge tone={CATEGORY_TONE[s.category]}>{s.category}</Badge>
                  </div>
                  <p className="mt-1.5 text-[12px] leading-5 text-zinc-500">{s.summary}</p>
                  <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[10px] text-zinc-500">
                    <span>{s.paperCount} papers</span>
                    <span>first year {s.firstYear}</span>
                  </div>
                  {s.addresses.length > 0 && (
                    <div className="mt-2 flex flex-wrap items-center gap-1.5">
                      <span className="font-mono text-[9px] uppercase tracking-wide text-zinc-600">
                        addresses
                      </span>
                      {s.addresses.map((a) => (
                        <Badge key={a}>{PROBLEM_LABEL[a] ?? a}</Badge>
                      ))}
                    </div>
                  )}
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {s.paperIds.slice(0, 8).map((pid) => (
                      <PaperLink key={pid} id={pid} />
                    ))}
                    {s.paperIds.length > 8 && (
                      <span className="font-mono text-[10px] text-zinc-500">
                        +{s.paperIds.length - 8}
                      </span>
                    )}
                  </div>
                </Card>
              ))}
            </div>

            <div className="space-y-3">
              <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-600">
                problem clusters ({clusters.length})
              </div>
              {clusters.length === 0 && (
                <p className="font-mono text-[11px] text-zinc-600">
                  No corpus entry links here yet.
                </p>
              )}
              {clusters.map((c) => (
                <button
                  key={c.id}
                  onClick={() => setSection("clusters")}
                  className="group w-full rounded-xl border border-zinc-800 bg-zinc-900/60 p-4 text-left transition-colors hover:border-emerald-700/60 hover:bg-emerald-950/20"
                >
                  <div className="flex items-start justify-between gap-2">
                    <span className="text-sm font-semibold text-zinc-100 group-hover:text-emerald-200">
                      {c.name}
                    </span>
                    <span className="font-mono text-lg text-emerald-300">{c.papers.length}</span>
                  </div>
                  <div className="mt-1 font-mono text-[10px] uppercase tracking-wide text-zinc-500">
                    papers · open problem clusters →
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
