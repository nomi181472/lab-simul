"use client";

import { useMemo, useState, type Dispatch, type SetStateAction } from "react";
import { PAPERS, PAPER_COUNT } from "@/labs/object-detection/data/papers";
import { YEARS } from "@/labs/object-detection/data/years";
import { Card, Badge } from "@/labs/object-detection/ui";
import type { Paper } from "@/labs/object-detection/data/types";

/* Facet counts are tallied from PAPERS itself: every number here is a paper
 * count, never an inferred claim. */

type Facet = { key: string; count: number };

const PAGE_SIZE = 12;

function tally(pick: (p: Paper) => string[]): Map<string, number> {
  const m = new Map<string, number>();
  for (const p of PAPERS) for (const v of pick(p)) m.set(v, (m.get(v) ?? 0) + 1);
  return m;
}

function top(m: Map<string, number>, n: number): Facet[] {
  return [...m.entries()]
    .map(([key, count]) => ({ key, count }))
    .sort((a, b) => b.count - a.count || a.key.localeCompare(b.key))
    .slice(0, n);
}

function toggle(set: Dispatch<SetStateAction<Set<string>>>, key: string) {
  set((prev) => {
    const next = new Set(prev);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    return next;
  });
}

function Chip({
  label,
  count,
  active,
  onClick,
}: {
  label: string;
  count: number;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded-md border px-1.5 py-0.5 font-mono text-[10px] tracking-wide transition-colors ${
        active
          ? "border-emerald-600 bg-emerald-600/20 text-emerald-200"
          : "border-zinc-800 text-zinc-500 hover:border-zinc-600 hover:text-zinc-300"
      }`}
    >
      {label}
      <span className={active ? "text-emerald-400/80" : "text-zinc-600"}> {count}</span>
    </button>
  );
}

function FacetGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="border-b border-zinc-800/70 pb-4 last:border-0 last:pb-0">
      <div className="mb-2 font-mono text-[10px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className="flex flex-wrap gap-1.5">{children}</div>
    </div>
  );
}

function PaperDetail({ p }: { p: Paper }) {
  return (
    <div className="mt-3 space-y-3 border-t border-zinc-800 pt-3">
      <div>
        <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">summary</div>
        <p className="mt-1 text-[12px] leading-5 text-zinc-300">{p.summary}</p>
      </div>
      <div>
        <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
          problem summary
        </div>
        <p className="mt-1 text-[12px] leading-5 text-zinc-400">{p.problemSummary}</p>
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-wide text-emerald-400">
            contributions ({p.contributions.length})
          </div>
          <ul className="mt-1 list-disc space-y-1 pl-4 text-[12px] leading-5 text-zinc-400">
            {p.contributions.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        </div>
        <div>
          <div className="font-mono text-[10px] uppercase tracking-wide text-amber-400">
            limitations ({p.limitations.length})
          </div>
          <ul className="mt-1 list-disc space-y-1 pl-4 text-[12px] leading-5 text-amber-200/80">
            {p.limitations.map((l, i) => (
              <li key={i}>{l}</li>
            ))}
          </ul>
        </div>
      </div>
      {p.benchmarks.length > 0 && (
        <div>
          <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            benchmarks ({p.benchmarks.length})
          </div>
          <div className="mt-1 overflow-x-auto rounded-lg border border-zinc-800">
            <table className="w-full border-collapse text-[11px]">
              <thead>
                <tr className="bg-zinc-900/80 font-mono text-[10px] uppercase text-zinc-500">
                  <th className="p-1.5 text-left">dataset</th>
                  <th className="p-1.5 text-left">metric</th>
                  <th className="p-1.5 text-right">value</th>
                  <th className="p-1.5 text-left">split</th>
                  <th className="p-1.5 text-left">location</th>
                </tr>
              </thead>
              <tbody>
                {p.benchmarks.map((b, i) => (
                  <tr key={i} className="border-t border-zinc-800/70">
                    <td className="p-1.5 text-zinc-300">{b.dataset}</td>
                    <td className="p-1.5 font-mono text-zinc-400">{b.metric}</td>
                    <td className="p-1.5 text-right font-mono text-emerald-300">{b.value}</td>
                    <td className="p-1.5 text-zinc-500">{b.split ?? "—"}</td>
                    <td className="p-1.5 font-mono text-zinc-500">{b.location ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      <div className="font-mono text-[10px] text-zinc-600">
        <span className="text-zinc-500">harvest:</span> {p.harvestFile}
      </div>
    </div>
  );
}

export function ExplorerView() {
  const [years, setYears] = useState<Set<string>>(() => new Set());
  const [problems, setProblems] = useState<Set<string>>(() => new Set());
  const [solutions, setSolutions] = useState<Set<string>>(() => new Set());
  const [datasets, setDatasets] = useState<Set<string>>(() => new Set());
  const [q, setQ] = useState("");
  const [expanded, setExpanded] = useState<string | null>(null);
  const [paging, setPaging] = useState<{ key: string; page: number }>({ key: "", page: 0 });

  const { yearFacets, problemFacets, solutionFacets, datasetFacets } = useMemo(() => {
    const yearCount = tally((p) => [String(p.year)]);
    return {
      yearFacets: YEARS.map((y) => ({ key: String(y), count: yearCount.get(String(y)) ?? 0 })),
      problemFacets: top(tally((p) => p.problemTags), 20),
      solutionFacets: top(tally((p) => p.solutionTags), 20),
      datasetFacets: top(tally((p) => p.datasets), 12),
    };
  }, []);

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return PAPERS.filter((p) => {
      if (years.size > 0 && !years.has(String(p.year))) return false;
      if (problems.size > 0 && !p.problemTags.some((t) => problems.has(t))) return false;
      if (solutions.size > 0 && !p.solutionTags.some((t) => solutions.has(t))) return false;
      if (datasets.size > 0 && !p.datasets.some((d) => datasets.has(d))) return false;
      if (needle) {
        const hay = `${p.title} ${p.summary}`.toLowerCase();
        if (!hay.includes(needle)) return false;
      }
      return true;
    });
  }, [years, problems, solutions, datasets, q]);

  const filterKey = [
    q,
    [...years].sort().join(","),
    [...problems].sort().join(","),
    [...solutions].sort().join(","),
    [...datasets].sort().join(","),
  ].join("|");

  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const pageSafe = paging.key === filterKey ? Math.min(paging.page, pageCount - 1) : 0;
  const setPage = (n: number) => setPaging({ key: filterKey, page: n });
  const rows = filtered.slice(pageSafe * PAGE_SIZE, pageSafe * PAGE_SIZE + PAGE_SIZE);
  const activeCount = years.size + problems.size + solutions.size + datasets.size + (q ? 1 : 0);

  const clearAll = () => {
    setYears(new Set());
    setProblems(new Set());
    setSolutions(new Set());
    setDatasets(new Set());
    setQ("");
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Corpus explorer</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          Cross-filter the {PAPER_COUNT}-paper corpus by year, problem tag, solution tag and
          dataset — then narrow with free text over titles and summaries. Facet counts are
          paper counts computed from the corpus itself.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-[280px_1fr]">
        <Card className="h-fit space-y-4">
          <FacetGroup label="year">
            {yearFacets.map((f) => (
              <Chip
                key={f.key}
                label={f.key}
                count={f.count}
                active={years.has(f.key)}
                onClick={() => toggle(setYears, f.key)}
              />
            ))}
          </FacetGroup>
          <FacetGroup label={`problem tags · top ${problemFacets.length}`}>
            {problemFacets.map((f) => (
              <Chip
                key={f.key}
                label={f.key}
                count={f.count}
                active={problems.has(f.key)}
                onClick={() => toggle(setProblems, f.key)}
              />
            ))}
          </FacetGroup>
          <FacetGroup label={`solution tags · top ${solutionFacets.length}`}>
            {solutionFacets.map((f) => (
              <Chip
                key={f.key}
                label={f.key}
                count={f.count}
                active={solutions.has(f.key)}
                onClick={() => toggle(setSolutions, f.key)}
              />
            ))}
          </FacetGroup>
          <FacetGroup label={`datasets · top ${datasetFacets.length}`}>
            {datasetFacets.map((f) => (
              <Chip
                key={f.key}
                label={f.key}
                count={f.count}
                active={datasets.has(f.key)}
                onClick={() => toggle(setDatasets, f.key)}
              />
            ))}
          </FacetGroup>
        </Card>

        <div className="space-y-3">
          <Card>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex flex-wrap items-center gap-3">
                <span className="font-mono text-[12px] text-emerald-300">
                  {filtered.length} / {PAPER_COUNT} papers
                </span>
                {activeCount > 0 && (
                  <span className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
                    {activeCount} filter{activeCount === 1 ? "" : "s"} active
                  </span>
                )}
              </div>
              <button
                onClick={clearAll}
                disabled={activeCount === 0}
                className="rounded-md border border-zinc-700 px-2 py-1 font-mono text-[10px] uppercase tracking-wide text-zinc-300 hover:border-zinc-500 hover:text-zinc-100 disabled:cursor-not-allowed disabled:opacity-40"
              >
                clear all
              </button>
            </div>
            <label className="mt-3 block">
              <span className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
                full-text search (title / summary)
              </span>
              <input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="e.g. focal loss, NMS-free, feature pyramid…"
                className="mt-1 w-full rounded-md border border-zinc-800 bg-zinc-900 px-3 py-2 text-[13px] text-zinc-200 placeholder:text-zinc-600 focus:border-emerald-600 focus:outline-none"
              />
            </label>
          </Card>

          <div className="space-y-2">
            {rows.map((p) => (
              <Card key={p.id} className={expanded === p.id ? "border-emerald-700/50" : ""}>
                <button className="w-full text-left" onClick={() => setExpanded(expanded === p.id ? null : p.id)}>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="rounded bg-emerald-950 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300">
                      {p.id}
                    </span>
                    <Badge tone="zinc">{p.year}</Badge>
                    <span className="text-[13px] font-semibold text-zinc-100">{p.shortTitle}</span>
                    <span className="ml-auto shrink-0 font-mono text-[10px] text-zinc-600">
                      {expanded === p.id ? "− collapse" : "+ expand"}
                    </span>
                  </div>
                  <p className="mt-1 text-[11px] leading-4 text-zinc-500">{p.title}</p>
                  <p className="mt-1.5 line-clamp-2 text-[12px] leading-5 text-zinc-400">
                    {p.summary}
                  </p>
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {p.problemTags.map((t) => (
                      <Badge key={t} tone="amber">{t}</Badge>
                    ))}
                    {p.solutionTags.map((t) => (
                      <Badge key={t} tone="sky">{t}</Badge>
                    ))}
                  </div>
                  {p.datasets.length > 0 && (
                    <div className="mt-1.5 flex flex-wrap gap-1">
                      {p.datasets.map((d) => (
                        <span
                          key={d}
                          className="rounded border border-zinc-800 px-1.5 py-0.5 font-mono text-[10px] text-zinc-500"
                        >
                          {d}
                        </span>
                      ))}
                    </div>
                  )}
                </button>
                {expanded === p.id && <PaperDetail p={p} />}
              </Card>
            ))}
            {filtered.length === 0 && (
              <Card tone="warn">
                <p className="text-sm text-zinc-400">
                  No papers match this combination of filters.
                </p>
              </Card>
            )}
          </div>

          {filtered.length > PAGE_SIZE && (
            <div className="flex items-center justify-between rounded-xl border border-zinc-800 bg-zinc-900/60 px-3 py-2">
              <button
                onClick={() => setPage(Math.max(0, pageSafe - 1))}
                disabled={pageSafe === 0}
                className="rounded-md border border-zinc-700 px-2 py-1 font-mono text-[10px] uppercase text-zinc-300 hover:border-zinc-500 disabled:cursor-not-allowed disabled:opacity-40"
              >
                ← prev
              </button>
              <span className="font-mono text-[10px] text-zinc-500">
                page {pageSafe + 1} / {pageCount} · {pageSafe * PAGE_SIZE + 1}–
                {Math.min(filtered.length, (pageSafe + 1) * PAGE_SIZE)} of {filtered.length}
              </span>
              <button
                onClick={() => setPage(Math.min(pageCount - 1, pageSafe + 1))}
                disabled={pageSafe >= pageCount - 1}
                className="rounded-md border border-zinc-700 px-2 py-1 font-mono text-[10px] uppercase text-zinc-300 hover:border-zinc-500 disabled:cursor-not-allowed disabled:opacity-40"
              >
                next →
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
