"use client";

import { useMemo } from "react";
import { PAPERS } from "@/labs/neuroevolution/data/papers";
import { Card, SectionTitle, Badge, NeuroPaperLink, Quote, Unstated } from "@/labs/neuroevolution/ui";
import { ModificationDiagram, GenomeStrip } from "@/labs/neuroevolution/architecture";
import { LOCUS_COLOR, LOCUS_LABEL, type Locus } from "@/labs/neuroevolution/data/types";

/* Operators: what mutation, crossover and selection actually do.
 *
 * Operator frequency is counted from the corpus, and every example is quoted from
 * the paper that uses it. The counts are a description of this corpus, not a
 * claim about the field's consensus.
 */

const OP_DEFS: { kind: "mutation" | "crossover" | "selection"; label: string; does: string }[] = [
  { kind: "mutation", label: "point mutation", does: "Replace or perturb one gene at a time. The only operator that introduces information the population did not already contain." },
  { kind: "mutation", label: "reset / reinitialise", does: "Assign a fresh random value to a gene. Escapes local optima that small perturbations cannot." },
  { kind: "mutation", label: "Cauchy noise", does: "Perturb with a heavy-tailed distribution instead of Gaussian, so rare large jumps are not suppressed." },
  { kind: "mutation", label: "structural add", does: "Append a node or connection gene, so the decoded network grows." },
  { kind: "crossover", label: "uniform crossover", does: "Take each gene from either parent at random, independently per locus." },
  { kind: "crossover", label: "one-point crossover", does: "Take a prefix from one parent and a suffix from the other at a single cut point." },
  { kind: "crossover", label: "subtree / module crossover", does: "Swap whole subtrees or modules. Makes structure, not values, the recombined unit." },
  { kind: "selection", label: "tournament selection", does: "Sample k candidates and keep the fittest. Selection pressure is the tournament size k." },
  { kind: "selection", label: "truncation selection", does: "Keep the top fraction of the population, discarding the rest outright." },
  { kind: "selection", label: "rank selection", does: "Select by rank rather than raw fitness, which flattens the pressure when fitness scales are extreme." },
  { kind: "selection", label: "elitism", does: "Carry the best individual forward untouched, guaranteeing the search never regresses." },
];

export function OperatorsView() {
  const counts = useMemo(() => {
    const m = new Map<string, { mutation: number; crossover: number; selection: number }>();
    const bump = (k: "mutation" | "crossover" | "selection", label: string) => {
      if (!m.has(label)) m.set(label, { mutation: 0, crossover: 0, selection: 0 });
      m.get(label)![k]++;
    };
    for (const p of PAPERS) {
      const e = p.method.evolution;
      for (const x of e.mutation) bump("mutation", x);
      for (const x of e.crossover) bump("crossover", x);
      for (const x of e.selection) bump("selection", x);
    }
    return m;
  }, []);

  const examples = useMemo(() => {
    const m = new Map<string, string[]>();
    for (const p of PAPERS) {
      const e = p.method.evolution;
      const all = [
        ...e.mutation.map((x) => ["mutation", x] as const),
        ...e.crossover.map((x) => ["crossover", x] as const),
        ...e.selection.map((x) => ["selection", x] as const),
      ];
      for (const [, label] of all) {
        if (!m.has(label)) m.set(label, []);
        const arr = m.get(label)!;
        if (arr.length < 8) arr.push(p.id);
      }
    }
    return m;
  }, []);

  const locusTotals = useMemo(() => {
    const m = new Map<Locus, number>();
    for (const p of PAPERS)
      for (const c of p.method.evolution.changes)
        m.set(c.locus, (m.get(c.locus) ?? 0) + 1);
    return [...m.entries()].sort((a, b) => b[1] - a[1]);
  }, []);

  const maxLocus = locusTotals[0]?.[1] ?? 1;

  return (
    <div className="flex flex-col gap-6">
      <SectionTitle>Operators</SectionTitle>
      <p className="max-w-3xl text-[13px] leading-6 text-zinc-400">
        The three operators, and what this corpus actually uses. Counts are
        keyword-detected over each paper&rsquo;s own text, so a paper that never
        names its selection scheme contributes nothing here rather than a guess.
      </p>

      <Card>
        <div className="flex flex-col gap-2">
          <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">
            where papers put their mutations
          </div>
          {locusTotals.map(([l, n]) => (
            <div key={l} className="flex items-center gap-3">
              <span className="w-28 shrink-0 font-mono text-[10px] text-zinc-400">
                {LOCUS_LABEL[l]}
              </span>
              <div
                className="h-2 rounded-full"
                style={{
                  width: `${Math.max(6, (n / maxLocus) * 260)}px`,
                  backgroundColor: LOCUS_COLOR[l],
                }}
              />
              <span className="font-mono text-[10px] text-zinc-500">{n}</span>
            </div>
          ))}
        </div>
      </Card>

      <div className="grid gap-3 lg:grid-cols-2">
        {OP_DEFS.map((op) => {
          const c = counts.get(op.label) ?? {
            mutation: 0,
            crossover: 0,
            selection: 0,
          };
          const n = c[op.kind];
          const papers = examples.get(op.label) ?? [];
          return (
            <Card key={op.label} className="flex flex-col gap-2">
              <div className="flex items-center justify-between gap-2">
                <span className="text-[12px] font-medium text-zinc-100">{op.label}</span>
                <Badge
                  tone={
                    op.kind === "mutation"
                      ? "rose"
                      : op.kind === "crossover"
                        ? "violet"
                        : "emerald"
                  }
                >
                  {op.kind} · {n} papers
                </Badge>
              </div>
              <div className="text-[11px] leading-5 text-zinc-400">{op.does}</div>
              {n === 0 ? (
                <Unstated what="explicit use in this corpus" />
              ) : (
                <div className="flex flex-wrap gap-1 border-t border-zinc-800 pt-2">
                  {papers.map((id) => (
                    <NeuroPaperLink key={id} id={id} short link />
                  ))}
                </div>
              )}
            </Card>
          );
        })}
      </div>

      <div className="flex flex-col gap-3">
        <SectionTitle>Worked examples</SectionTitle>
        <p className="max-w-3xl text-[12px] leading-6 text-zinc-400">
          Full papers whose extracted mechanism is shown here: the loci their
          operators touch, and the sentence that says so.
        </p>
        <div className="grid gap-3 lg:grid-cols-2">
          {PAPERS.filter((p) => p.method.evolution.changes.length >= 2)
            .slice(0, 10)
            .map((p) => (
              <Card key={p.id} className="flex flex-col gap-2">
                <div className="flex items-center gap-2">
                  <NeuroPaperLink id={p.id} link />
                  <Badge tone="zinc">{p.method.evolution.family}</Badge>
                </div>
                <div className="text-[12px] leading-5 text-zinc-100">{p.title}</div>
                <GenomeStrip
                  loci={p.method.evolution.changes.map((c) => c.locus)}
                  changed={p.method.evolution.changes.map((c) => c.locus)}
                />
                <ModificationDiagram changes={p.method.evolution.changes} />
                {p.method.evolution.representation ? (
                  <Quote>{p.method.evolution.representation}</Quote>
                ) : null}
              </Card>
            ))}
        </div>
      </div>
    </div>
  );
}