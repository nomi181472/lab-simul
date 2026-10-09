"use client";

import { useMemo } from "react";
import { Card, SectionTitle, Badge, PaperLink } from "@/labs/networking/ui";
import { PAPERS } from "@/labs/networking/data/records";
import { MANIFEST } from "@/labs/networking/data/manifest";
import { AUDIT, CORPUS_STATS, KB } from "@/labs/networking/data/kb";
import { ANALYSIS } from "@/labs/networking/data/analysis";
import { SIM_META } from "@/labs/networking/sims/registry";

/**
 * Corpus-wide provenance checks. Every row is a count of records that would make
 * the lab quietly wrong — not a quality score. A record with no quoted evidence is
 * not a smaller paper, it is a claim the lab cannot source, and it is shown as
 * "not stated" everywhere rather than hidden.
 */
function useChecks() {
  return useMemo(() => {
    const ids = new Set<string>();
    const dupIds: string[] = [];
    for (const p of PAPERS) {
      if (ids.has(p.id)) dupIds.push(p.id);
      ids.add(p.id);
    }

    const noEvidence = PAPERS.filter((p) =>
      Object.values(p.evidence).every((arr) => arr.length === 0),
    ).map((p) => p.id);

    const fullTextNoQuote = PAPERS.filter(
      (p) =>
        p.hasFullText &&
        p.problem === "not stated in the retrieved text" &&
        p.solution === "not stated in the retrieved text",
    ).map((p) => p.id);

    const missingManifest = MANIFEST.papers.filter(
      (m) => !PAPERS.some((p) => p.id === m.id),
    ).map((m) => m.id);

    const danglingIds = new Set<string>();
    const seen = (arr: (string | null | undefined)[] | undefined) => {
      for (const id of arr ?? []) if (id && !ids.has(id)) danglingIds.add(id);
    };
    seen(ANALYSIS.future.all.map((f) => f.paperId));
    seen(ANALYSIS.crossDomain.flatMap((a) => a.paperIds));
    seen(ANALYSIS.clusters.flatMap((c) => [c.corePaperId, ...c.repeatPaperIds]));

    const simClusterKeys = ANALYSIS.clusters
      .map((c) => c.simulatorKey)
      .filter((k): k is string => Boolean(k));
    const simsWithoutCluster = Object.keys(SIM_META).filter(
      (k) => !simClusterKeys.includes(k),
    );

    return {
      total: PAPERS.length,
      manifestTotal: MANIFEST.total,
      dupIds,
      noEvidence,
      fullTextNoQuote,
      missingManifest,
      danglingIds: [...danglingIds],
      validationProblems: AUDIT.validationProblems,
      deduped: AUDIT.deduped,
      simsWithoutCluster,
      domainCoverage: KB.domains.map((d) => ({
        id: d.id,
        label: d.label,
        n: PAPERS.filter((p) => p.domains.includes(d.id)).length,
      })),
      sourceDepth: [
        ["full-text", CORPUS_STATS.fullText],
        ["abstract", CORPUS_STATS.withAbstract],
        ["metadata-only", CORPUS_STATS.metadataOnly],
      ] as [string, number][],
      linkStatus: Object.entries(CORPUS_STATS.byLinkStatus) as [string, number][],
    };
  }, []);
}

export function AuditView() {
  const c = useChecks();

  const rows: [string, number, number][] = [
    ["duplicate record ids", c.dupIds.length, c.total],
    ["records with zero quoted evidence", c.noEvidence.length, c.total],
    [
      "full-text records with neither problem nor solution quote",
      c.fullTextNoQuote.length,
      CORPUS_STATS.fullText,
    ],
    ["manifest entries without a record", c.missingManifest.length, c.manifestTotal],
    ["ids referenced by analysis but absent from records", c.danglingIds.length, 1],
    ["validation problems reported by the KB pass", c.validationProblems.length, c.total],
  ];

  const allClean = rows.every(([, bad]) => bad === 0);
  const maxDomain = Math.max(1, ...c.domainCoverage.map((d) => d.n));

  return (
    <div className="space-y-6">
      <SectionTitle>Audit</SectionTitle>

      <Card tone={allClean ? "ok" : "warn"}>
        <div className="flex items-center gap-2">
          <Badge tone={allClean ? "emerald" : "amber"}>
            {allClean ? "clean" : "issues found"}
          </Badge>
          <span className="text-[13px] text-zinc-300">
            {c.total} records over {c.manifestTotal} manifest entries
          </span>
        </div>
        <p className="mt-2 text-[13px] leading-6 text-zinc-400">
          A record with no quoted evidence is not hidden from the lab. It is counted here
          and rendered as <em>not stated in the retrieved text</em> in every view, so an
          absent extraction stays visible rather than silently rounded away.
        </p>
      </Card>

      <Card>
        <SectionTitle>Corpus invariants</SectionTitle>
        <table className="mt-3 w-full text-left font-mono text-[11px]">
          <thead className="text-zinc-600">
            <tr>
              <th className="py-1 font-normal">check</th>
              <th className="py-1 text-right font-normal">failing</th>
              <th className="py-1 text-right font-normal">of</th>
              <th className="py-1 text-right font-normal">status</th>
            </tr>
          </thead>
          <tbody className="text-zinc-300">
            {rows.map(([label, bad, of]) => (
              <tr key={label} className="border-t border-zinc-800/70">
                <td className="py-1.5">{label}</td>
                <td className="py-1.5 text-right text-zinc-200">{bad}</td>
                <td className="py-1.5 text-right text-zinc-600">{of}</td>
                <td className="py-1.5 text-right">
                  <Badge tone={bad === 0 ? "emerald" : "amber"}>
                    {bad === 0 ? "pass" : "failing"}
                  </Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            evidence depth (source of every quote)
          </p>
          <div className="mt-3 space-y-2">
            {c.sourceDepth.map(([label, n]) => (
              <div key={label} className="flex items-center gap-2">
                <span className="w-28 shrink-0 font-mono text-[10px] text-zinc-400">
                  {label}
                </span>
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-zinc-800">
                  <div
                    className="h-full bg-cyan-600"
                    style={{ width: `${(n / c.total) * 100}%` }}
                  />
                </div>
                <span className="w-16 text-right font-mono text-[10px] text-zinc-500">
                  {n} ({Math.round((n / c.total) * 100)}%)
                </span>
              </div>
            ))}
          </div>
          <p className="mt-3 text-[11px] leading-5 text-zinc-500">
            Abstract-only and metadata-only records can still fill fields, but every
            quote they contribute comes from the abstract or title — never invented.
          </p>
        </Card>

        <Card>
          <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            link status
          </p>
          <div className="mt-3 space-y-2">
            {c.linkStatus.map(([label, n]) => (
              <div key={label} className="flex items-center gap-2">
                <span className="w-28 shrink-0 font-mono text-[10px] text-zinc-400">
                  {label}
                </span>
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-zinc-800">
                  <div
                    className="h-full bg-emerald-600"
                    style={{ width: `${(n / c.total) * 100}%` }}
                  />
                </div>
                <span className="w-16 text-right font-mono text-[10px] text-zinc-500">
                  {n}
                </span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            validation problems (KB pass)
          </p>
          <div className="mt-2 space-y-1.5">
            {c.validationProblems.map((v) => (
              <div key={v.paperId} className="flex items-baseline gap-2">
                <PaperLink id={v.paperId} short />
                <span className="text-[11px] text-zinc-500">{v.issue}</span>
              </div>
            ))}
            {c.validationProblems.length === 0 && (
              <p className="text-[11px] text-emerald-500">none</p>
            )}
          </div>
          <p className="mt-3 font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            deduplicated during validation
          </p>
          <div className="mt-1 space-y-1">
            {c.deduped.map((d) => (
              <p key={d.dropped} className="font-mono text-[10px] text-zinc-500">
                kept <span className="text-zinc-300">{d.kept}</span> · dropped{" "}
                <span className="text-zinc-300">{d.dropped}</span>
              </p>
            ))}
            {c.deduped.length === 0 && (
              <p className="text-[11px] text-zinc-600">none</p>
            )}
          </div>
        </Card>

        <Card>
          <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            domain coverage (papers per domain)
          </p>
          <div className="mt-2 space-y-1">
            {c.domainCoverage.map((d) => (
              <div key={d.id} className="flex items-center gap-2">
                <span className="w-36 shrink-0 truncate text-[10px] text-zinc-400">
                  {d.label}
                </span>
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-zinc-800">
                  <div
                    className="h-full bg-sky-600"
                    style={{ width: `${(d.n / maxDomain) * 100}%` }}
                  />
                </div>
                <span className="w-8 text-right font-mono text-[10px] text-zinc-500">
                  {d.n}
                </span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <Card>
        <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
          simulator grounding
        </p>
        <p className="mt-1 font-mono text-[10px] text-zinc-600">
          {Object.keys(SIM_META).length} simulators · {ANALYSIS.clusters.filter((x) => x.simulatorKey).length}{" "}
          clusters bound to one
        </p>
        <div className="mt-2 space-y-1">
          {ANALYSIS.clusters
            .filter((x) => x.simulatorKey)
            .map((x) => (
              <div key={x.clusterId} className="flex flex-wrap items-baseline gap-2">
                <span className="text-[11px] text-zinc-300">{x.label}</span>
                <span className="font-mono text-[10px] text-sky-500">
                  → {SIM_META[x.simulatorKey as keyof typeof SIM_META]?.label}
                </span>
                <PaperLink id={x.corePaperId} short />
              </div>
            ))}
        </div>
        {c.simsWithoutCluster.length > 0 && (
          <p className="mt-3 font-mono text-[10px] text-amber-500">
            simulators with no cluster binding: {c.simsWithoutCluster.join(", ")} (queueing
            and traffic-classification models are bound to related clusters)
          </p>
        )}
      </Card>
    </div>
  );
}
