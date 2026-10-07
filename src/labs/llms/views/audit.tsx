"use client";

import { useMemo } from "react";
import { Card, SectionTitle, Badge } from "@/labs/llms/ui";
import { PAPERS } from "@/labs/llms/data/papers";
import { MANIFEST } from "@/labs/llms/data/manifest";
import { SIMULATOR_PAPERS } from "@/labs/llms/sims/papers";
import {
  ALL_ARCH_KINDS,
  ALL_CHANGE_OPERATORS,
} from "@/labs/llms/data/vocab";

/**
 * Corpus-wide provenance checks.
 *
 * Each check is a count of records that would make the lab quietly wrong, not a
 * quality score: a paper with an empty architecture quote is not a smaller paper,
 * it is a claim the lab cannot source.
 */
function useChecks() {
  return useMemo(() => {
    const ids = new Set<string>();
    const dupIds: string[] = [];
    for (const p of PAPERS) {
      if (ids.has(p.id)) dupIds.push(p.id);
      ids.add(p.id);
    }

    const emptyQuotes = PAPERS.filter((p) =>
      [...p.architecture.layers, ...p.changes, ...p.objectives, ...p.results].some(
        (e) => !e.quote || e.quote.trim().length < 20,
      ),
    ).map((p) => p.id);

    const thinText = PAPERS.filter((p) => p.textChars < 12_000).map((p) => p.id);

    // A paper with no architecture layers AND no changes has no extractable
    // mechanism at all, which usually means a bad PDF rather than a bad paper.
    const noMechanism = PAPERS.filter(
      (p) => p.architecture.layers.length === 0 && p.changes.length === 0,
    ).map((p) => p.id);

    const missingManifest = MANIFEST.papers.filter(
      (m) => !PAPERS.some((p) => p.id === m.id && p.fileName === m.fileName),
    ).length;

    const simIds = new Set(SIMULATOR_PAPERS.flatMap((s) => s.papers));
    const danglingSimPapers = [...simIds].filter((id) => !ids.has(id));

    return {
      total: PAPERS.length,
      manifestTotal: MANIFEST.total,
      dupIds,
      emptyQuotes,
      thinText,
      noMechanism,
      missingManifest,
      danglingSimPapers,
      simulators: SIMULATOR_PAPERS.length,
      simBindings: simIds.size,
    };
  }, []);
}

export function AuditView() {
  const c = useChecks();
  const ids = useMemo(() => new Set(PAPERS.map((p) => p.id)), []);

  const archCoverage = ALL_ARCH_KINDS.map((k) => ({
    kind: k,
    n: PAPERS.filter((p) => p.architecture.layers.some((l) => l.kind === k)).length,
  }));
  const opCoverage = ALL_CHANGE_OPERATORS.map((o) => ({
    op: o,
    n: PAPERS.filter((p) => p.changes.some((ch) => ch.operator === o)).length,
  }));

  const rows: [string, number, number][] = [
    ["duplicate ids", c.dupIds.length, c.total],
    ["records with an empty quote", c.emptyQuotes.length, c.total],
    ["records with thin extracted text", c.thinText.length, c.total],
    ["records with no mechanism at all", c.noMechanism.length, c.total],
    ["manifest entries without a record", c.missingManifest, c.manifestTotal],
    ["simulator bindings to unknown papers", c.danglingSimPapers.length, c.simBindings],
  ];

  const allClean = rows.every(([, bad]) => bad === 0);

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
          A record with a missing mechanism is not hidden from the lab. It is counted
          here and shown in the paper view as <em>not stated</em>, so an absent
          extraction is visible rather than silently rounded away.
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
          <SectionTitle>Architecture element coverage</SectionTitle>
          <div className="mt-3 space-y-1.5">
            {archCoverage.map((a) => (
              <div key={a.kind} className="flex items-center gap-2">
                <span className="w-28 shrink-0 font-mono text-[10px] text-zinc-400">
                  {a.kind}
                </span>
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-zinc-800">
                  <div
                    className="h-full rounded-full bg-cyan-600"
                    style={{ width: `${Math.round((a.n / c.total) * 100)}%` }}
                  />
                </div>
                <span className="w-10 shrink-0 text-right font-mono text-[10px] text-zinc-500">
                  {a.n}
                </span>
              </div>
            ))}
          </div>
        </Card>

        <Card>
          <SectionTitle>Modification operator coverage</SectionTitle>
          <div className="mt-3 space-y-1.5">
            {opCoverage.map((o) => (
              <div key={o.op} className="flex items-center gap-2">
                <span className="w-40 shrink-0 truncate font-mono text-[10px] text-zinc-400">
                  {o.op}
                </span>
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-zinc-800">
                  <div
                    className="h-full rounded-full bg-amber-600"
                    style={{ width: `${Math.round((o.n / c.total) * 100)}%` }}
                  />
                </div>
                <span className="w-10 shrink-0 text-right font-mono text-[10px] text-zinc-500">
                  {o.n}
                </span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {c.simulators > 0 && (
        <Card>
          <SectionTitle>Simulator grounding</SectionTitle>
          <p className="mt-2 font-mono text-[10px] text-zinc-500">
            {c.simulators} simulators · {c.simBindings} paper bindings
          </p>
          <div className="mt-3 space-y-1.5">
            {SIMULATOR_PAPERS.map((s) => (
              <div key={s.id} className="flex items-center gap-2">
                <span className="w-28 shrink-0 font-mono text-[10px] text-zinc-400">
                  {s.id}
                </span>
                <div className="flex flex-wrap gap-1">
                  {s.papers.map((id) => (
                    <span
                      key={id}
                      className={`rounded px-1 py-0.5 font-mono text-[10px] ${
                        ids.has(id)
                          ? "bg-zinc-800 text-zinc-300"
                          : "bg-rose-950 text-rose-300"
                      }`}
                    >
                      {id}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}