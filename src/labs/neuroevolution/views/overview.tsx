"use client";

import { useMemo } from "react";
import { PAPERS, PAPER_COUNT } from "@/labs/neuroevolution/data/papers";
import { MANIFEST, MANIFEST_TOTAL_CITATIONS, YEAR_MIN, YEAR_MAX } from "@/labs/neuroevolution/data/manifest";
import { CONCEPTS } from "@/labs/neuroevolution/data/concepts";
import {
  Card,
  SectionTitle,
  Badge,
  CiteBadge,
  NeuroPaperLink,
  DifficultyBadge,
} from "@/labs/neuroevolution/ui";
import { ArchitectureDiagram } from "@/labs/neuroevolution/architecture";
import { LOCUS_LABEL, LAYER_LABEL, type LayerKind, type Locus } from "@/labs/neuroevolution/data/types";
import { useNeuroLab } from "@/labs/neuroevolution/context";

/* ------------------------------ overview ------------------------------ */

export function OverviewView() {
  const { setSection } = useNeuroLab();

  const stats = useMemo(() => {
    const withLayers = PAPERS.filter((p) => p.architecture.layers.length > 0).length;
    const withChanges = PAPERS.filter(
      (p) => p.method.evolution.changes.length > 0,
    ).length;
    const withFitness = PAPERS.filter((p) => p.method.evolution.fitness).length;
    const loci = new Set<Locus>(
      PAPERS.flatMap((p) => p.method.evolution.changes.map((c) => c.locus)),
    );
    const families = new Set<string>(PAPERS.map((p) => String(p.method.evolution.family)));
    const kinds = new Set<LayerKind>(
      PAPERS.flatMap((p) => p.architecture.layers.map((l) => l.kind)),
    );
    return { withLayers, withChanges, withFitness, loci, families, kinds };
  }, []);

  const top = useMemo(() => PAPERS.slice(0, 8), []);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-3">
        <SectionTitle>Neuroevolution Research Lab</SectionTitle>
        <p className="max-w-3xl text-[13px] leading-6 text-zinc-400">
          {MANIFEST.length} papers on evolving neural networks, ordered by
          citation count. The thing that makes this field hard to read is that
          &ldquo;the architecture&rdquo; means something different in every paper:
          a fixed MLP whose weights are evolved, a topology that grows a node at a
          time, a program tree, a collection of modules. So this lab puts the
          architecture first — on every paper, with the locus its operators change
          made explicit — and quotes the paper for every claim it makes.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {[
          { k: "papers", v: String(MANIFEST.length), tone: "rose" as const },
          { k: "structured", v: String(PAPER_COUNT), tone: "rose" as const },
          { k: "citations", v: MANIFEST_TOTAL_CITATIONS.toLocaleString(), tone: "zinc" as const },
          { k: "drawn topologies", v: String(stats.withLayers), tone: "emerald" as const },
          { k: "with locus changes", v: String(stats.withChanges), tone: "violet" as const },
          { k: "with fitness stated", v: String(stats.withFitness), tone: "cyan" as const },
        ].map((s) => (
          <Card key={s.k}>
            <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">
              {s.k}
            </div>
            <div className="mt-1 font-mono text-xl text-zinc-100">{s.v}</div>
          </Card>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card tone="accent" className="flex flex-col gap-3">
          <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-400">
            start here
          </div>
          <div className="flex flex-col gap-2">
            {[
              { id: "architecture", label: "Architecture", blurb: "every paper's evolved network, drawn" },
              { id: "modification", label: "Modification", blurb: "which locus each operator changes, with simulators" },
              { id: "path", label: "Reading path", blurb: "an ordered route through the field" },
              { id: "foundations", label: "Foundations", blurb: "concepts, from genotype to aging" },
            ].map((s) => (
              <button
                key={s.id}
                onClick={() => setSection(s.id as never)}
                className="flex items-center justify-between gap-3 rounded-lg border border-zinc-800 bg-zinc-900/50 px-3 py-2 text-left transition-colors hover:border-rose-800"
              >
                <span className="text-[12px] font-medium text-zinc-100">{s.label}</span>
                <span className="text-[11px] text-zinc-500">{s.blurb}</span>
              </button>
            ))}
          </div>
        </Card>

        <Card className="flex flex-col gap-3">
          <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">
            what the corpus is made of
          </div>
          <div className="flex flex-col gap-2 font-mono text-[10px] text-zinc-500">
            <div>
              algorithm families:{" "}
              <span className="text-zinc-300">{[...stats.families].join(", ")}</span>
            </div>
            <div>
              mutation loci found:{" "}
              <span className="text-zinc-300">
                {[...stats.loci].map((l) => LOCUS_LABEL[l]).join(", ")}
              </span>
            </div>
            <div>
              layer kinds found:{" "}
              <span className="text-zinc-300">
                {[...stats.kinds].map((k) => LAYER_LABEL[k]).join(", ")}
              </span>
            </div>
            <div>
              years:{" "}
              <span className="text-zinc-300">
                {YEAR_MIN}–{YEAR_MAX}
              </span>
            </div>
          </div>
          <div className="border-t border-zinc-800 pt-3 text-[11px] leading-5 text-zinc-400">
            Most papers in this corpus are not &ldquo;neuroevolution&rdquo; by
            reputation: many evolve a fixed MLP&rsquo;s weights and little else.
            That is the point of the locus axis — it separates weight-only search
            from search that genuinely restructures the network.
          </div>
        </Card>
      </div>

      <div className="flex flex-col gap-3">
        <SectionTitle>Most-cited papers</SectionTitle>
        <div className="grid gap-3 lg:grid-cols-2">
          {top.map((p) => (
            <Card key={p.id} className="flex flex-col gap-2.5">
              <div className="flex flex-wrap items-center gap-2">
                <NeuroPaperLink id={p.id} link />
                <CiteBadge n={p.citations} />
                <Badge tone="zinc">{p.year}</Badge>
                <DifficultyBadge d={p.difficulty} />
              </div>
              <div className="text-[13px] font-medium leading-5 text-zinc-100">
                {p.title}
              </div>
              <ArchitectureDiagram architecture={p.architecture} compact />
              <div className="text-[11px] leading-5 text-zinc-400">{p.summary}</div>
            </Card>
          ))}
        </div>
      </div>

      <Card tone="warn">
        <div className="flex flex-col gap-2 text-[11px] leading-5 text-amber-300/80">
          <div className="font-mono text-[10px] uppercase tracking-widest">
            evidence policy
          </div>
          <div>
            Every architecture, operator, locus and result in this lab is quoted or
            paraphrased from the corpus PDF named on the card. Where a paper does
            not describe something, the lab says &ldquo;not stated&rdquo; and draws
            no diagram — it never infers a network the paper did not print. Lab
            commentary is labelled as such and is never presented as an author
            claim.
          </div>
          {CONCEPTS.length > 0 ? (
            <div>
              Curated concept material ({CONCEPTS.length} entries) is background
              written for this lab, and is kept separate from paper extracts.
            </div>
          ) : null}
        </div>
      </Card>
    </div>
  );
}