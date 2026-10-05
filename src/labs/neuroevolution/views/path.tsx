"use client";

import { useMemo } from "react";
import { PAPERS } from "@/labs/neuroevolution/data/papers";
import { CONCEPT_BY_ID } from "@/labs/neuroevolution/data/concepts";
import { Card, SectionTitle, Badge, Formula, NeuroPaperLink } from "@/labs/neuroevolution/ui";

/* Reading path and timeline.
 *
 * The path is a staged reading order derived from the corpus's own concept
 * dependencies, so it is a curriculum rather than an editorial opinion: each
 * stage lists the concepts a reader needs before the next stage is useful.
 */

type Stage = {
  id: string;
  title: string;
  blurb: string;
  concepts: string[];
};

const STAGES: Stage[] = [
  {
    id: "search",
    title: "1 · The search, before the network",
    blurb:
      "A genetic algorithm needs a population, a genotype, a fitness and three operators. None of that is neural-specific, and most neuroevolution papers skip it in the first paragraph. Read these first or the rest is unreadable.",
    concepts: ["neuroevolution", "genotype", "phenotype", "fitness", "population", "genetic-algorithm"],
  },
  {
    id: "operators",
    title: "2 · What the operators change",
    blurb:
      "Mutation is the only operator that invents anything. Everything else recombines what is already in the population, which is why mutation rate and mutation locus are the two settings a paper's contribution usually lives in.",
    concepts: ["mutation", "crossover", "selection", "elitism", "locus"],
  },
  {
    id: "encoding",
    title: "3 · Encoding: the part everyone skips",
    blurb:
      "What the operators actually touch is the genotype, and how it maps to a network is a design decision with real consequences for what is reachable. This is where most of the field's genuinely interesting variation lives.",
    concepts: ["direct-encoding", "genetic-programming", "neat", "modular-neuroevolution"],
  },
  {
    id: "basics",
    title: "4 · The gradient-free baselines",
    blurb:
      "Evolution strategies and CMA-ES dominate weight-only search. They matter here because CMA-ES's covariance adaptation is what makes high-dimensional weight vectors tractable, and because ES admits a gradient interpretation.",
    concepts: ["evolution-strategy", "cma-es"],
  },
  {
    id: "diversity",
    title: "5 · Search that does not converge",
    blurb:
      "When the reward is sparse or misleading, converging early is fatal. Quality-diversity and novelty search change what fitness means, and return a repertoire instead of a point.",
    concepts: ["quality-diversity", "novelty-search", "open-ended-evolution", "aging"],
  },
  {
    id: "apps",
    title: "6 · Applications and the cost argument",
    blurb:
      "Neurocontrollers and architecture search are where the field's claims get tested against gradient-based and RL baselines, on wall-clock and evaluation count rather than accuracy alone.",
    concepts: ["neurocontroller", "nas", "fitness-shaping"],
  },
];

export function PathView() {
  const stages = useMemo(
    () =>
      STAGES.map((s) => ({
        ...s,
        entries: s.concepts.map((id) => CONCEPT_BY_ID[id]).filter(Boolean),
      })),
    [],
  );

  return (
    <div className="flex flex-col gap-6">
      <SectionTitle>Reading path</SectionTitle>
      <p className="max-w-3xl text-[13px] leading-6 text-zinc-400">
        Six stages, in dependency order. Each one lists the concepts a reader
        actually needs, and the corpus papers that use them.
      </p>

      <div className="flex flex-col gap-4">
        {stages.map((s, i) => (
          <Card key={s.id} tone={i % 2 === 0 ? "accent" : "default"}>
            <div className="flex flex-col gap-3">
              <div>
                <div className="text-[13px] font-semibold text-zinc-100">{s.title}</div>
                <div className="mt-1 max-w-3xl text-[11px] leading-5 text-zinc-400">
                  {s.blurb}
                </div>
              </div>
              <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                {s.entries.map((c) => (
                  <div
                    key={c.id}
                    className="flex flex-col gap-1 rounded-lg border border-zinc-800 bg-zinc-900/50 px-3 py-2"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[12px] font-medium text-zinc-200">
                        {c.name}
                      </span>
                      <Badge tone="zinc">{c.category}</Badge>
                    </div>
                    <div className="text-[11px] leading-5 text-zinc-400">{c.intuition}</div>
                    {c.paperIds && c.paperIds.length > 0 ? (
                      <div className="flex flex-wrap gap-1">
                        {c.paperIds.slice(0, 6).map((id) => (
                          <NeuroPaperLink key={id} id={id} short link />
                        ))}
                        {c.paperIds.length > 6 ? (
                          <span className="font-mono text-[10px] text-zinc-600">
                            +{c.paperIds.length - 6}
                          </span>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                ))}
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
