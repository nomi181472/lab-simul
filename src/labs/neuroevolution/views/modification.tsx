"use client";

import { useMemo, useState } from "react";
import { PAPERS } from "@/labs/neuroevolution/data/papers";
import {
  Card,
  SectionTitle,
  Badge,
  CiteBadge,
  NeuroPaperLink,
  Unstated,
} from "@/labs/neuroevolution/ui";
import {
  ModificationDiagram,
  GenomeStrip,
  LocusLegend,
} from "@/labs/neuroevolution/architecture";
import {
  LOCUS_COLOR,
  LOCUS_LABEL,
  type Locus,
} from "@/labs/neuroevolution/data/types";
import { SIMULATORS, SIMULATOR_IDS, SIMULATOR_META, SIMULATOR_GROUP } from "@/labs/neuroevolution/sims";

/* The Modification view.
 *
 * Two halves, because "how is the modification being done" has two answers:
 *
 *   1. the interactive simulators, which show the mechanism itself;
 *   2. the corpus, which shows what real papers actually change — grouped by
 *      locus, with the paper's own sentence quoted for each claim.
 */

const ALL_LOCI = Object.keys(LOCUS_LABEL) as Locus[];

export function ModificationView() {
  const [sim, setSim] = useState<string>(SIMULATOR_IDS[0]);
  const Sim = SIMULATORS[sim as keyof typeof SIMULATORS] ?? SIMULATORS[SIMULATOR_IDS[0]];

  const [locus, setLocus] = useState<Locus | "all">("all");
  const [family, setFamily] = useState<string>("all");

  const groups = useMemo(() => {
    const m = new Map<string, string[]>();
    for (const s of SIMULATOR_IDS) {
      const g = SIMULATOR_GROUP[s as keyof typeof SIMULATOR_GROUP] || "Other";
      if (!m.has(g)) m.set(g, []);
      m.get(g)!.push(s);
    }
    return [...m.entries()];
  }, []);

  const locusPapers = useMemo(() => {
    const m = new Map<Locus, typeof PAPERS>();
    for (const p of PAPERS) {
      for (const c of p.method.evolution.changes) {
        if (!m.has(c.locus)) m.set(c.locus, []);
        m.get(c.locus)!.push(p);
      }
    }
    return m;
  }, []);

  const families = useMemo(() => {
    const s = new Set<string>(PAPERS.map((p) => String(p.method.evolution.family)));
    return ["all", ...[...s].sort()];
  }, []);

  const shown = useMemo(
    () => (locus === "all" ? PAPERS : (locusPapers.get(locus) ?? [])),
    [locus, locusPapers],
  );

  const meta = SIMULATOR_META.find((m) => m.id === sim);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-3">
        <SectionTitle>Modification</SectionTitle>
        <p className="max-w-3xl text-[13px] leading-6 text-zinc-400">
          Every algorithm here does one thing: it changes part of a neural
          network and keeps the change if a number went up. The simulators below
          run that loop deterministically; the corpus half of this page then shows
          which locus the real papers let their operators touch.
        </p>
      </div>

      <Card>
        <LocusLegend />
      </Card>

      {/* ---------------- simulators ---------------- */}
      <div className="flex flex-col gap-3">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">
            mechanism · interactive
          </div>
          <div className="font-mono text-[10px] text-zinc-600">
            deterministic · same controls, same run
          </div>
        </div>

        <div className="flex flex-col gap-3">
          {groups.map(([g, ids]) => (
            <div key={g} className="flex flex-wrap items-center gap-2">
              <span className="w-24 shrink-0 font-mono text-[10px] text-zinc-600">
                {g}
              </span>
              {ids.map((id) => {
                const m = SIMULATOR_META.find((x) => x.id === id)!;
                return (
                  <button
                    key={id}
                    onClick={() => setSim(id)}
                    title={m.blurb}
                    className={`rounded-md border px-2.5 py-1 font-mono text-[10px] transition-colors ${
                      sim === id
                        ? "border-rose-700/60 bg-rose-600/20 text-rose-300"
                        : "border-zinc-800 text-zinc-500 hover:border-zinc-700 hover:text-zinc-300"
                    }`}
                  >
                    {m.label}
                  </button>
                );
              })}
            </div>
          ))}
        </div>

        {meta ? (
          <div className="text-[11px] leading-5 text-zinc-400">{meta.blurb}</div>
        ) : null}

        <Card tone="accent">
          <Sim />
        </Card>
      </div>

      {/* ---------------- corpus ---------------- */}
      <div className="flex flex-col gap-3">
        <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">
          mechanism · what the corpus actually changes
        </div>

        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setLocus("all")}
            className={`rounded-md border px-2.5 py-1 font-mono text-[10px] ${
              locus === "all"
                ? "border-rose-700/60 bg-rose-600/20 text-rose-300"
                : "border-zinc-800 text-zinc-500"
            }`}
          >
            all papers ({PAPERS.length})
          </button>
          {ALL_LOCI.map((l) => {
            const n = locusPapers.get(l)?.length ?? 0;
            return (
              <button
                key={l}
                onClick={() => setLocus(locus === l ? "all" : l)}
                className="rounded-md border px-2.5 py-1 font-mono text-[10px] transition-colors"
                style={{
                  borderColor: locus === l ? LOCUS_COLOR[l] : "#27272a",
                  color: locus === l ? "#e4e4e7" : n ? "#71717a" : "#3f3f46",
                  background: locus === l ? `${LOCUS_COLOR[l]}22` : "transparent",
                }}
              >
                {LOCUS_LABEL[l]} · {n}
              </button>
            );
          })}
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <span className="font-mono text-[10px] text-zinc-600">family</span>
          {families.map((f) => (
            <button
              key={f}
              onClick={() => setFamily(f)}
              className={`rounded-md border px-2 py-0.5 font-mono text-[10px] ${
                family === f
                  ? "border-zinc-600 bg-zinc-800 text-zinc-200"
                  : "border-zinc-800 text-zinc-600"
              }`}
            >
              {f}
            </button>
          ))}
        </div>

        <div className="flex flex-col gap-4">
          {shown
            .filter((p) => family === "all" || p.method.evolution.family === family)
            .slice(0, 60)
            .map((p) => (
              <Card key={p.id} className="flex flex-col gap-3">
                <div className="flex flex-wrap items-center gap-2">
                  <NeuroPaperLink id={p.id} link />
                  <CiteBadge n={p.citations} />
                  <Badge tone="violet">{p.method.evolution.family}</Badge>
                  <Badge tone="cyan">{p.method.evolution.encoding}</Badge>
                </div>
                <div className="text-[13px] font-medium leading-5 text-zinc-100">
                  {p.title}
                </div>

                <div className="flex flex-wrap gap-2">
                  {p.method.evolution.mutation.length > 0 ? (
                    p.method.evolution.mutation.map((m) => (
                      <Badge key={m} tone="rose">
                        mutation: {m}
                      </Badge>
                    ))
                  ) : (
                    <Unstated what="mutation operator" />
                  )}
                  {p.method.evolution.crossover.map((c) => (
                    <Badge key={c} tone="violet">
                      crossover: {c}
                    </Badge>
                  ))}
                  {p.method.evolution.selection.map((s) => (
                    <Badge key={s} tone="emerald">
                      selection: {s}
                    </Badge>
                  ))}
                </div>

                <GenomeStrip
                  loci={p.method.evolution.changes.map((c) => c.locus)}
                  changed={p.method.evolution.changes.map((c) => c.locus)}
                />

                <ModificationDiagram changes={p.method.evolution.changes} />

                {p.method.evolution.fitness ? (
                  <div className="flex flex-col gap-1">
                    <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-600">
                      fitness
                    </div>
                    <div className="text-[11px] leading-5 text-zinc-400">
                      {p.method.evolution.fitness}
                    </div>
                  </div>
                ) : (
                  <Unstated what="fitness function" />
                )}
              </Card>
            ))}
        </div>

        {shown.length > 60 ? (
          <div className="font-mono text-[10px] text-zinc-600">
            showing 60 of {shown.length} papers for this locus
          </div>
        ) : null}
      </div>
    </div>
  );
}