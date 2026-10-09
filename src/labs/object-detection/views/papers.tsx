"use client";

import { Suspense, lazy, useMemo, useState } from "react";
import { PAPERS } from "@/labs/object-detection/data/papers";
import { Card, Badge, SectionTitle } from "@/labs/object-detection/ui";
import type { Paper } from "@/labs/object-detection/data/types";

/* Corpus browser: 121 records, paginated 15 per page. Expanding a row lazily
 * loads the full harvest record (views/paper-detail.tsx + data/harvest-records.ts). */

const PaperDetail = lazy(() => import("./paper-detail"));

const PAGE_SIZE = 15;

const SORTS = [
  { value: "year-desc", label: "year ↓" },
  { value: "year-asc", label: "year ↑" },
  { value: "title", label: "title A–Z" },
  { value: "citations", label: "citations ↓" },
] as const;

type SortKey = (typeof SORTS)[number]["value"];

function sortPapers(list: Paper[], key: SortKey): Paper[] {
  const out = [...list];
  if (key === "year-desc") {
    out.sort((a, b) => b.year - a.year || a.shortTitle.localeCompare(b.shortTitle));
  } else if (key === "year-asc") {
    out.sort((a, b) => a.year - b.year || a.shortTitle.localeCompare(b.shortTitle));
  } else if (key === "title") {
    out.sort((a, b) => a.shortTitle.localeCompare(b.shortTitle));
  } else {
    out.sort((a, b) => (b.citations ?? 0) - (a.citations ?? 0));
  }
  return out;
}

export function PapersView() {
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<SortKey>("year-desc");
  const [expanded, setExpanded] = useState<string | null>(null);
  const [paging, setPaging] = useState<{ key: string; page: number }>({ key: "", page: 0 });

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const hits = needle
      ? PAPERS.filter((p) =>
          `${p.id} ${p.title} ${p.summary}`.toLowerCase().includes(needle),
        )
      : PAPERS;
    return sortPapers(hits, sort);
  }, [q, sort]);

  const filterKey = `${q}|${sort}`;
  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const pageSafe = paging.key === filterKey ? Math.min(paging.page, pageCount - 1) : 0;
  const setPage = (n: number) => setPaging({ key: filterKey, page: n });
  const rows = filtered.slice(pageSafe * PAGE_SIZE, pageSafe * PAGE_SIZE + PAGE_SIZE);

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Paper explorer</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          All {PAPERS.length} corpus papers. Search id / title / summary, sort the corpus, and
          expand any row for the full record — problem statement, contributions, limitations,
          reported benchmarks, future directions, provenance.
        </p>
      </div>

      <Card>
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex flex-col gap-1">
            <label className="font-mono text-[9px] uppercase text-zinc-600">search</label>
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="e.g. OD001, DETR, focal loss…"
              className="w-64 rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-emerald-600 focus:outline-none"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="font-mono text-[9px] uppercase text-zinc-600">sort</label>
            <select
              value={sort}
              onChange={(e) => setSort(e.target.value as SortKey)}
              className="rounded-lg border border-zinc-800 bg-zinc-900 px-2 py-2 text-sm text-zinc-200"
            >
              {SORTS.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label}
                </option>
              ))}
            </select>
          </div>
          <div className="ml-auto flex items-center gap-3">
            <span className="font-mono text-[10px] text-zinc-500">
              {filtered.length} / {PAPERS.length} papers
            </span>
            <div className="flex items-center gap-1.5">
              <button
                onClick={() => setPage(Math.max(0, pageSafe - 1))}
                disabled={pageSafe === 0}
                className="rounded-md border border-zinc-700 px-2 py-1 font-mono text-[10px] uppercase text-zinc-300 hover:border-zinc-500 disabled:cursor-not-allowed disabled:opacity-40"
              >
                ← prev
              </button>
              <span className="font-mono text-[10px] text-zinc-500">
                {pageSafe + 1} / {pageCount}
              </span>
              <button
                onClick={() => setPage(Math.min(pageCount - 1, pageSafe + 1))}
                disabled={pageSafe >= pageCount - 1}
                className="rounded-md border border-zinc-700 px-2 py-1 font-mono text-[10px] uppercase text-zinc-300 hover:border-zinc-500 disabled:cursor-not-allowed disabled:opacity-40"
              >
                next →
              </button>
            </div>
          </div>
        </div>
      </Card>

      <div className="space-y-2">
        {rows.map((p) => (
          <Card key={p.id} className={expanded === p.id ? "border-emerald-700/50" : ""}>
            <button
              className="w-full text-left"
              onClick={() => setExpanded(expanded === p.id ? null : p.id)}
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="rounded bg-emerald-950 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300">
                  {p.id}
                </span>
                <Badge tone="zinc">{p.year}</Badge>
                {p.citations != null && (
                  <Badge tone="violet">{p.citations.toLocaleString()} cites</Badge>
                )}
                {p.venue && (
                  <Badge tone="zinc">
                    <span className="max-w-[160px] truncate">{p.venue}</span>
                  </Badge>
                )}
                <span className="ml-auto shrink-0 font-mono text-[10px] text-zinc-600">
                  {expanded === p.id ? "− collapse" : "+ expand"}
                </span>
              </div>
              <h2 className="mt-1.5 text-sm font-semibold leading-6 text-zinc-100">
                {p.shortTitle}
              </h2>
              <p className="mt-0.5 truncate text-[12px] leading-5 text-zinc-500">{p.summary}</p>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {p.tags.slice(0, 6).map((t) => (
                  <Badge key={t} tone="zinc">{t}</Badge>
                ))}
                {p.tags.length > 6 && (
                  <span className="self-center font-mono text-[10px] text-zinc-600">
                    +{p.tags.length - 6}
                  </span>
                )}
              </div>
            </button>
            {expanded === p.id && (
              <Suspense
                fallback={
                  <p className="mt-3 font-mono text-[10px] text-zinc-600">
                    loading harvest record…
                  </p>
                }
              >
                <PaperDetail p={p} />
              </Suspense>
            )}
          </Card>
        ))}
        {filtered.length === 0 && (
          <Card tone="warn">
            <SectionTitle>No papers match</SectionTitle>
            <p className="mt-1 text-sm text-zinc-400">
              Try a paper id (OD001), a method name, or a shorter query.
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
  );
}
