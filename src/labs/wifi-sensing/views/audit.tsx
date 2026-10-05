"use client";

import { useMemo } from "react";
import { PAPERS, PAPER_COUNT } from "@/labs/wifi-sensing/data/papers";
import { MANIFEST } from "@/labs/wifi-sensing/data/manifest";
import { CONCEPTS } from "@/labs/wifi-sensing/data/concepts";
import { METRICS } from "@/labs/wifi-sensing/data/metrics";
import { DATASETS } from "@/labs/wifi-sensing/data/datasets";
import { SIMULATOR_IDS, SIMULATOR_META } from "@/labs/wifi-sensing/sims";
import { SIM_PAPERS } from "@/labs/wifi-sensing/sims/papers";
import { Badge, Card, SectionTitle } from "@/labs/wifi-sensing/ui";

function pct(n: number, d: number) {
  return d ? ((100 * n) / d).toFixed(1) : "0.0";
}

export function AuditView() {
  const stats = useMemo(() => {
    const total = PAPER_COUNT;
    const has = (f: (p: (typeof PAPERS)[number]) => boolean) => PAPERS.filter(f).length;
    return {
      total,
      summary: has((p) => p.summary.length > 0),
      contributions: has((p) => p.contribution.length > 0),
      results: has((p) => p.results.length > 0),
      problems: has((p) => p.problem.length > 0),
      gaps: has((p) => p.researchGap.length > 0),
      ablations: has((p) => p.ablations.length > 0),
      assumptions: has((p) => (p.assumptions?.length ?? 0) > 0),
      limitsAuthor: has((p) => p.limitations.authorStated.length > 0),
      limitsEvident: has((p) => p.limitations.evident.length > 0),
      equations: has((p) => p.equationIds.length > 0),
      relations: has((p) => p.relations.length > 0),
      baselines: has((p) => p.baselines.length > 0),
      model: has((p) => p.chain.model !== "not described in the retrieved text"),
      preproc: has((p) => p.chain.preprocessing[0] !== "not described in the retrieved text"),
      groundTruth: has((p) => !!p.groundTruth),
      simBound: SIMULATOR_IDS.filter((id) => (SIM_PAPERS[id]?.papers.length ?? 0) > 0).length,
      conceptBound: CONCEPTS.filter((c) => c.paperIds.length > 0).length,
      metricBound: METRICS.filter((m) => m.paperIds.length > 0).length,
      totalRelations: PAPERS.reduce((s, p) => s + p.relations.length, 0),
      totalEquations: PAPERS.reduce((s, p) => s + p.equationIds.length, 0),
      danglingConceptRefs: new Set(
        PAPERS.flatMap((p) => [...p.equationIds, ...p.concepts]).filter((c) => !CONCEPTS.some((x) => x.id === c)),
      ).size,
      danglingRelations: new Set(
        PAPERS.flatMap((p) => p.relations.map((r) => r.to)).filter((id) => !PAPERS.some((x) => x.id === id)),
      ).size,
    };
  }, []);

  const rows: [string, number, number][] = [
    ["summary extracted", stats.summary, stats.total],
    ["problem extracted", stats.problems, stats.total],
    ["research gap extracted", stats.gaps, stats.total],
    ["contributions extracted", stats.contributions, stats.total],
    ["reported numbers extracted", stats.results, stats.total],
    ["ablation sentences extracted", stats.ablations, stats.total],
    ["assumptions extracted", stats.assumptions, stats.total],
    ["author-stated limitations", stats.limitsAuthor, stats.total],
    ["methodology limitations (synthesis)", stats.limitsEvident, stats.total],
    ["standard equations linked", stats.equations, stats.total],
    ["model detected", stats.model, stats.total],
    ["preprocessing detected", stats.preproc, stats.total],
    ["ground truth detected", stats.groundTruth, stats.total],
    ["citation relations detected", stats.relations, stats.total],
    ["baselines identified", stats.baselines, stats.total],
  ];

  const unboundSims = SIMULATOR_IDS.filter((id) => (SIM_PAPERS[id]?.papers.length ?? 0) === 0);
  const unboundConcepts = CONCEPTS.filter((c) => c.paperIds.length === 0);

  return (
    <div className="space-y-5">
      <SectionTitle>Corpus audit</SectionTitle>
      <p className="max-w-3xl text-[12px] leading-6 text-zinc-500">
        What fraction of the corpus actually yielded evidence for each field. Low coverage on a
        field is not a bug to hide — it means the paper did not state that thing, and an empty
        field is the honest record. Integrity checks at the bottom must all pass.
      </p>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {[
          { k: "manifest papers", v: String(MANIFEST.length) },
          { k: "structured records", v: String(stats.total) },
          { k: "total citations", v: MANIFEST.reduce((s, p) => s + p.citations, 0).toLocaleString() },
          { k: "pages extracted", v: MANIFEST.reduce((s, p) => s + p.pages, 0).toLocaleString() },
        ].map((s) => (
          <Card key={s.k}>
            <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">{s.k}</div>
            <div className="mt-1 font-mono text-xl text-cyan-300">{s.v}</div>
          </Card>
        ))}
      </div>

      <Card>
        <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
          field coverage
        </div>
        <div className="mt-3 space-y-1.5">
          {rows.map(([label, n, d]) => {
            const pctv = Number(pct(n, d));
            return (
              <div key={label} className="flex items-center gap-2">
                <span className="w-52 shrink-0 truncate text-[11px] text-zinc-400">{label}</span>
                <div className="h-2 flex-1 overflow-hidden rounded bg-zinc-800">
                  <div
                    className={`h-full ${
                      pctv >= 80 ? "bg-emerald-600/70" : pctv >= 40 ? "bg-amber-600/70" : "bg-rose-600/60"
                    }`}
                    style={{ width: `${pctv}%` }}
                  />
                </div>
                <span className="w-16 text-right font-mono text-[10px] text-zinc-500">
                  {n}/{d} ({pctv}%)
                </span>
              </div>
            );
          })}
        </div>
      </Card>

      <div className="grid gap-3 lg:grid-cols-2">
        <Card tone={stats.danglingConceptRefs === 0 && stats.danglingRelations === 0 ? "ok" : "warn"}>
          <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            referential integrity
          </div>
          <ul className="mt-2 space-y-1.5 text-[11px]">
            {[
              ["dangling concept refs", stats.danglingConceptRefs],
              ["dangling relation targets", stats.danglingRelations],
              ["manifest ids without a record", MANIFEST.filter((m) => !PAPERS.some((p) => p.id === m.id)).length],
              ["records not in manifest", PAPERS.filter((p) => !MANIFEST.some((m) => m.id === p.id)).length],
            ].map(([k, v]) => (
              <li key={String(k)} className="flex items-center justify-between">
                <span className="text-zinc-400">{k}</span>
                <span className={`font-mono ${Number(v) === 0 ? "text-emerald-400" : "text-rose-400"}`}>
                  {String(v)}
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-2 text-[10px] leading-4 text-zinc-600">
            Every id referenced by a concept, a relation or a simulator must resolve to a real
            corpus record. Zero on all four is the requirement.
          </p>
        </Card>

        <Card tone={unboundSims.length === 0 ? "ok" : "warn"}>
          <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            simulator / concept grounding
          </div>
          <ul className="mt-2 space-y-1.5 text-[11px]">
            {[
              ["simulators bound to papers", `${stats.simBound}/${SIMULATOR_IDS.length}`],
              ["concepts bound to papers", `${stats.conceptBound}/${CONCEPTS.length}`],
              ["metrics bound to papers", `${stats.metricBound}/${METRICS.length}`],
              ["datasets quoted from papers", `${DATASETS.length}`],
              ["citation edges total", String(stats.totalRelations)],
              ["equation links total", String(stats.totalEquations)],
            ].map(([k, v]) => (
              <li key={k} className="flex items-center justify-between">
                <span className="text-zinc-400">{k}</span>
                <span className="font-mono text-cyan-300">{v}</span>
              </li>
            ))}
          </ul>
          {unboundSims.length ? (
            <p className="mt-2 text-[10px] leading-4 text-amber-500">
              unbound simulators: {unboundSims.join(", ")}
            </p>
          ) : null}
          {unboundConcepts.length ? (
            <p className="mt-1 text-[10px] leading-4 text-zinc-600">
              concepts with no matching paper: {unboundConcepts.map((c) => c.id).join(", ")}
            </p>
          ) : null}
        </Card>
      </div>

      <Card>
        <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
          per-simulator record
        </div>
        <div className="mt-2 overflow-x-auto">
          <table className="w-full text-left text-[11px]">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-500">
                <th className="py-1 pr-3 font-normal">simulator</th>
                <th className="py-1 pr-3 font-normal">papers</th>
                <th className="py-1 font-normal">grounds</th>
              </tr>
            </thead>
            <tbody>
              {SIMULATOR_META.map((m) => {
                const b = SIM_PAPERS[m.id];
                return (
                  <tr key={m.id} className="border-b border-zinc-800/50">
                    <td className="py-1 pr-3 text-zinc-300">{m.label}</td>
                    <td className="py-1 pr-3">
                      {b?.papers.length ? (
                        <span className="font-mono text-emerald-400">{b.papers.join(" ")}</span>
                      ) : (
                        <span className="font-mono text-rose-400">none</span>
                      )}
                    </td>
                    <td className="py-1 text-zinc-500">{b?.note}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <Card>
        <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
          corpus provenance
        </div>
        <ul className="mt-2 space-y-1 text-[11px] leading-5 text-zinc-500">
          <li>
            · Harvest: arXiv candidate queries, title-gated, ranked by Crossref{" "}
            <span className="font-mono">is-referenced-by-count</span>, then filtered to
            human-sensing from each PDF&apos;s own text (scripts/wifi_refill.py).
          </li>
          <li>· Text extraction: pdftotext -layout, cached under .cache/wifi_text/.</li>
          <li>
            · Classification: keyword rules over title + abstract + opening text
            (scripts/wifi_index.py). Deterministic and re-runnable.
          </li>
          <li>
            · Records: cue-phrase sentence extraction over the same text
            (scripts/wifi_records.py). No field is authored.
          </li>
          <li>
            · Relations: title / surname-year matching inside each paper&apos;s own reference block.
          </li>
        </ul>
        <div className="mt-3 flex flex-wrap gap-1.5">
          <Badge tone="cyan">no invented results</Badge>
          <Badge tone="cyan">no invented equations</Badge>
          <Badge tone="cyan">no invented relations</Badge>
          <Badge tone="amber">lab synthesis labelled separately</Badge>
        </div>
      </Card>
    </div>
  );
}