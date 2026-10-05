"use client";

import { useMemo, useState } from "react";
import { PAPERS } from "@/labs/wifi-sensing/data/papers";
import { MANIFEST } from "@/labs/wifi-sensing/data/manifest";
import { MODALITY_LABEL, TASK_LABEL } from "@/labs/wifi-sensing/data/types";
import { Badge, Card, SectionTitle, WifiPaperLink } from "@/labs/wifi-sensing/ui";

/* The comparison axis that actually separates papers in this field is not the
 * architecture — it is what was measured (RSSI vs CSI vs FMCW) and what was
 * inferred from it. This view builds that matrix from the generated records. */

type Axis = "modality" | "task";

export function CompareView() {
  const [axis, setAxis] = useState<Axis>("modality");
  const [left, setLeft] = useState<string>("csi-80211n");
  const [right, setRight] = useState<string>("rssi");

  const values = useMemo(() => {
    const src = axis === "modality" ? MANIFEST : PAPERS;
    return [...new Set(src.map((p) => (axis === "modality" ? p.modality : p.task)))].sort();
  }, [axis]);

  const axisLabel = (v: string) =>
    axis === "modality" ? (MODALITY_LABEL[v as keyof typeof MODALITY_LABEL] ?? v) : (TASK_LABEL[v as keyof typeof TASK_LABEL] ?? v);

  const stats = (v: string) => {
    const ps = PAPERS.filter((p) => (axis === "modality" ? p.modality === v : p.task === v));
    const cites = ps.map((p) => p.citations);
    const mean = cites.length ? cites.reduce((s, c) => s + c, 0) / cites.length : 0;
    return {
      n: ps.length,
      meanCites: mean,
      medianYear: median(ps.map((p) => p.year).filter(Boolean)),
      withResults: ps.filter((p) => p.results.length > 0).length,
      withRelations: ps.filter((p) => p.relations.length > 0).length,
      models: topCount(ps.flatMap((p) => p.chain.model.split(/,\s*/)), 6),
      preproc: topCount(ps.flatMap((p) => p.chain.preprocessing), 6),
      bandwidths: countBy(ps.map((p) => p.bandwidth)),
      groundTruth: countBy(ps.map((p) => p.groundTruth ?? "unstated")),
    };
  };

  const L = stats(left);
  const R = stats(right);

  return (
    <div className="space-y-5">
      <SectionTitle>Compare</SectionTitle>
      <p className="max-w-3xl text-[12px] leading-6 text-zinc-500">
        The field&apos;s most useful comparison axis is what was measured, not what model was used.
        Choose two values on either axis to see how their pipelines, eras and reporting habits
        differ. Every count here comes from the generated records.
      </p>

      <div className="space-y-2">
        <div className="inline-flex gap-1">
          {(
            [
              ["modality", "by measurement modality"],
              ["task", "by task"],
            ] as [Axis, string][]
          ).map(([a, l]) => (
            <button
              key={a}
              onClick={() => {
                setAxis(a);
                setLeft(a === "modality" ? "csi-80211n" : "activity-recognition");
                setRight(a === "modality" ? "rssi" : "localization");
              }}
              className={`rounded-md border px-2.5 py-1 text-[11px] ${
                axis === a
                  ? "border-cyan-600 bg-cyan-600/20 text-cyan-200"
                  : "border-zinc-800 text-zinc-500"
              }`}
            >
              {l}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <select
            value={left}
            onChange={(e) => setLeft(e.target.value)}
            className="rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-[11px] text-cyan-300"
          >
            {values.map((v) => (
              <option key={v} value={v}>
                {axisLabel(v)}
              </option>
            ))}
          </select>
          <span className="font-mono text-[11px] text-zinc-600">vs</span>
          <select
            value={right}
            onChange={(e) => setRight(e.target.value)}
            className="rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-[11px] text-cyan-300"
          >
            {values.map((v) => (
              <option key={v} value={v}>
                {axisLabel(v)}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        {[L, R].map((s, i) => {
          const v = i === 0 ? left : right;
          const ps = PAPERS.filter((p) =>
            axis === "modality" ? p.modality === v : p.task === v,
          );
          return (
            <Card key={v + i} tone={i === 0 ? "accent" : "violet"}>
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-semibold text-zinc-100">{axisLabel(v)}</h3>
                <Badge tone="zinc">{s.n} papers</Badge>
              </div>
              <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
                {[
                  ["mean citations", s.meanCites.toFixed(0)],
                  ["median year", String(s.medianYear || "—")],
                  ["report numbers", `${s.withResults}/${s.n}`],
                  ["have relations", `${s.withRelations}/${s.n}`],
                ].map(([k, val]) => (
                  <div key={k} className="rounded-lg border border-zinc-800 bg-black/20 p-2">
                    <div className="font-mono text-[9px] uppercase text-zinc-500">{k}</div>
                    <div className="mt-0.5 font-mono text-sm text-zinc-200">{val}</div>
                  </div>
                ))}
              </div>

              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <div>
                  <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-500">
                    models
                  </div>
                  <div className="mt-1 flex flex-wrap gap-1">
                    {s.models.map(([k, n]) => (
                      <Badge key={k} tone="zinc">
                        {k} {n}
                      </Badge>
                    ))}
                  </div>
                </div>
                <div>
                  <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-500">
                    preprocessing
                  </div>
                  <div className="mt-1 flex flex-wrap gap-1">
                    {s.preproc.map(([k, n]) => (
                      <Badge key={k} tone="zinc">
                        {k} {n}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>

              {s.bandwidths.length ? (
                <div className="mt-3">
                  <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-500">
                    bandwidth stated
                  </div>
                  <div className="mt-1 flex flex-wrap gap-1">
                    {s.bandwidths.map(([k, n]) => (
                      <Badge key={k} tone={k === "unknown" ? "zinc" : "cyan"}>
                        {k} {n}
                      </Badge>
                    ))}
                  </div>
                </div>
              ) : null}

              <div className="mt-3">
                <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-500">
                  top papers
                </div>
                <div className="mt-1 space-y-0.5">
                  {[...ps]
                    .sort((a, b) => b.citations - a.citations)
                    .slice(0, 5)
                    .map((p) => (
                      <div key={p.id} className="flex items-center gap-2">
                        <WifiPaperLink id={p.id} short link />
                        <span className="min-w-0 flex-1 truncate text-[10px] text-zinc-500">
                          {p.title}
                        </span>
                        <span className="shrink-0 font-mono text-[10px] text-zinc-600">
                          {p.citations}c
                        </span>
                      </div>
                    ))}
                </div>
              </div>
            </Card>
          );
        })}
      </div>

      <Card>
        <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
          cross-tab · modality × task
        </div>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-left text-[10px]">
            <thead>
              <tr className="border-b border-zinc-800">
                <th className="py-1 pr-2 font-normal text-zinc-500">modality \ task</th>
                {[...new Set(MANIFEST.map((m) => m.task))]
                  .sort()
                  .map((t) => (
                    <th key={t} className="px-1 py-1 font-normal text-zinc-500">
                      {TASK_LABEL[t as keyof typeof TASK_LABEL]?.split(" ")[0] ?? t}
                    </th>
                  ))}
              </tr>
            </thead>
            <tbody>
              {[...new Set(MANIFEST.map((m) => m.modality))].sort().map((mo) => (
                <tr key={mo} className="border-b border-zinc-800/50">
                  <td className="py-1 pr-2 text-zinc-400">
                    {MODALITY_LABEL[mo as keyof typeof MODALITY_LABEL] ?? mo}
                  </td>
                  {[...new Set(MANIFEST.map((m) => m.task))].sort().map((t) => {
                    const n = MANIFEST.filter((m) => m.modality === mo && m.task === t).length;
                    const max = 40;
                    return (
                      <td key={t} className="px-1 py-1">
                        {n > 0 ? (
                          <span
                            className="inline-block rounded px-1 font-mono"
                            style={{
                              background: `rgba(34,211,238,${0.08 + (n / max) * 0.5})`,
                            }}
                          >
                            {n}
                          </span>
                        ) : (
                          <span className="text-zinc-700">·</span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function countBy(xs: string[]): [string, number][] {
  const m = new Map<string, number>();
  for (const x of xs) m.set(x, (m.get(x) ?? 0) + 1);
  return [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5);
}

function topCount(xs: string[], n: number): [string, number][] {
  return countBy(xs.filter((x) => x && x !== "not described in the retrieved text")).slice(0, n);
}

function median(xs: number[]): number {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}