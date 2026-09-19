"use client";

import { useMemo, useState } from "react";
import { PAPERS } from "@/labs/object-detection/data/papers";
import { Card, Badge } from "@/labs/object-detection/ui";

export function PapersView() {
  const [q, setQ] = useState("");
  const [yearFilter, setYearFilter] = useState<number | null>(null);
  const [tagFilter, setTagFilter] = useState<string | null>(null);

  const allTags = useMemo(() => {
    const s = new Set<string>();
    for (const p of PAPERS) for (const t of p.tags) s.add(t);
    return [...s].sort();
  }, []);

  const filtered = useMemo(() => {
    return PAPERS.filter((p) => {
      if (yearFilter !== null && p.year !== yearFilter) return false;
      if (tagFilter && !p.tags.includes(tagFilter)) return false;
      if (q) {
        const needle = q.toLowerCase();
        const hay =
          `${p.id} ${p.title} ${p.shortTitle} ${p.summary} ${p.contributions.join(" ")} ${p.tags.join(" ")} ${p.authors.join(" ")}`.toLowerCase();
        if (!hay.includes(needle)) return false;
      }
      return true;
    });
  }, [q, yearFilter, tagFilter]);

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Paper explorer</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          All {PAPERS.length} corpus papers. Search by anything; each paper lists the
          evidence keys it supports and the concepts it evidences.
        </p>
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1">
          <label className="font-mono text-[9px] uppercase text-zinc-600">search</label>
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="e.g. focal, anchor-free, DETR, PP-YOLO…"
            className="w-64 rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-emerald-600 focus:outline-none"
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="font-mono text-[9px] uppercase text-zinc-600">year</label>
          <select
            value={yearFilter ?? ""}
            onChange={(e) => setYearFilter(e.target.value ? Number(e.target.value) : null)}
            className="rounded-lg border border-zinc-800 bg-zinc-900 px-2 py-2 text-sm text-zinc-200"
          >
            <option value="">all years</option>
            {Array.from(new Set(PAPERS.map((p) => p.year).sort((a, b) => b - a))).map((y) => (
              <option key={y} value={y}>
                {y}
              </option>
            ))}
          </select>
        </div>
        <div className="flex-1" />
        <div className="font-mono text-[10px] text-zinc-500">
          {filtered.length} / {PAPERS.length} papers
        </div>
      </div>

      {allTags.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {allTags.slice(0, 20).map((t) => (
            <button
              key={t}
              onClick={() => setTagFilter(tagFilter === t ? null : t)}
              className={`rounded-full border px-2 py-0.5 text-[10px] ${
                tagFilter === t
                  ? "border-emerald-600 bg-emerald-600/20 text-emerald-200"
                  : "border-zinc-800 text-zinc-500 hover:text-zinc-300"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      )}

      <div className="space-y-2">
        {filtered.map((p) => {
          const evCount = (p.evidences ?? []).length;
          return (
            <Card key={p.id} className="!py-3">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="rounded bg-emerald-950 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300">
                      {p.id}
                    </span>
                    <Badge tone="zinc">{p.year}</Badge>
                    {p.tags.slice(0, 4).map((t) => (
                      <Badge key={t} tone="sky">
                        {t}
                      </Badge>
                    ))}
                  </div>
                  <h2 className="mt-1.5 text-sm font-semibold leading-6 text-zinc-100">
                    {p.shortTitle}
                  </h2>
                  <p className="font-mono text-[10px] text-zinc-500">
                    {p.authors.slice(0, 4).join(", ")}
                    {p.authors.length > 4 ? " et al." : ""}
                    {p.arxiv ? ` · arXiv:${p.arxiv}` : ""}
                  </p>
                  <p className="mt-1.5 text-[12px] leading-5 text-zinc-400">{p.summary}</p>
                </div>
                <div className="shrink-0 text-right font-mono text-[10px] text-zinc-600">
                  {evCount} evidence key{evCount === 1 ? "" : "s"}
                </div>
              </div>
              {(p.contributions?.length ?? 0) > 0 && (
                <ul className="mt-2 flex flex-wrap gap-1.5">
                  {p.contributions.map((c, i) => (
                    <li
                      key={i}
                      className="rounded-md border border-zinc-800 bg-zinc-900/60 px-1.5 py-0.5 text-[10px] text-zinc-400"
                    >
                      {c}
                    </li>
                  ))}
                </ul>
              )}
              {evCount > 0 && (
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {p.evidences.map((ev) => (
                    <span
                      key={ev}
                      className="inline-flex items-center gap-1 rounded border border-zinc-800 px-1.5 py-0.5 font-mono text-[9px] text-zinc-500"
                    >
                      {ev}
                    </span>
                  ))}
                </div>
              )}
            </Card>
          );
        })}
        {filtered.length === 0 && (
          <Card tone="warn">
            <p className="text-sm text-zinc-400">No papers match the current filters.</p>
          </Card>
        )}
      </div>
    </div>
  );
}