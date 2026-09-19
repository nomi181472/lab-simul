"use client";

import { useMemo, useState } from "react";
import { PAPERS } from "@/labs/object-detection/data/papers";
import { Card, PaperLink, Badge, MiniBar } from "@/labs/object-detection/ui";
import { Segmented } from "@/labs/object-detection/sims/shared";
import { countPapersByYear } from "@/labs/object-detection/data/audit";

const FAMILIES = [
  { value: "all", label: "all" },
  { value: "transformer", label: "transformer" },
  { value: "two-stage", label: "two-stage" },
  { value: "one-stage", label: "one-stage" },
  { value: "real-time", label: "real-time" },
  { value: "NMS-free", label: "NMS-free" },
  { value: "anchor-free", label: "anchor-free" },
] as const;

type Family = (typeof FAMILIES)[number]["value"];

export function ExplorerView() {
  const [family, setFamily] = useState<Family>("all");
  const [yearMin, setYearMin] = useState(2015);
  const [yearMax, setYearMax] = useState(2026);
  const [q, setQ] = useState("");

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return PAPERS.filter((p) => {
      if (p.year < yearMin || p.year > yearMax) return false;
      if (family !== "all") {
        const matches = p.tags.some((t) => t.toLowerCase().includes(family.toLowerCase()));
        if (family === "NMS-free" && !p.tags.some((t) => t.toLowerCase() === "nms-free")) return false;
        if (!matches) return false;
      }
      if (needle) {
        const hay = `${p.title} ${p.shortTitle} ${p.summary} ${p.contributions.join(" ")}`.toLowerCase();
        const terms = needle.split(/\s+/).filter(Boolean);
        if (!terms.every((t) => hay.includes(t))) return false;
      }
      return true;
    });
  }, [family, yearMin, yearMax, q]);

  const counts = countPapersByYear();
  const years = useMemo(() => {
    const ys: number[] = [];
    for (let y = 2015; y <= 2026; y++) ys.push(counts[y] ?? 0);
    return ys;
  }, [counts]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Corpus explorer</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          Cross-filter all 48 papers by design family, year range, and free text.
          Every result is corpus-evidenced — the filter never adds claims.
        </p>
      </div>

      <Card>
        <div className="flex flex-wrap items-end gap-6">
          <Segmented label="family" value={family} options={[...FAMILIES]} onChange={setFamily} />
          <label className="flex flex-col gap-1">
            <span className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">range 2015–2026</span>
            <div className="font-mono text-[11px] text-zinc-300">
              {yearMin} – {yearMax}
            </div>
            <input type="range" min={2015} max={2026} step={1} value={yearMin}
              onChange={(e) => { const v = Number(e.target.value); setYearMin(Math.min(v, yearMax)); }}
              className="w-40 accent-emerald-500" />
            <input type="range" min={2015} max={2026} step={1} value={yearMax}
              onChange={(e) => { const v = Number(e.target.value); setYearMax(Math.max(v, yearMin)); }}
              className="w-40 accent-emerald-500" />
          </label>
          <label className="flex min-w-44 flex-col gap-1">
            <span className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">full-text search</span>
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="e.g. focal loss, NMS-free, small objects…"
              className="rounded-md border border-zinc-800 bg-zinc-900 px-2 py-1 text-[12px] text-zinc-200 placeholder:text-zinc-600"
            />
          </label>
        </div>

        <div className="mt-5 grid gap-6 md:grid-cols-3">
          <div className="md:col-span-1">
            <div className="mb-1 font-mono text-[9px] uppercase tracking-wide text-zinc-500">corpus by year</div>
            <MiniBar values={years} labels={years.map((_, i) => String(2015 + i))} max={Math.max(...years)} />
          </div>
          <div className="md:col-span-2">
            <div className="mb-2 flex items-center justify-between">
              <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
                {filtered.length} of {PAPERS.length} papers match
              </div>
            </div>
            <div className="max-h-[520px] space-y-2 overflow-y-auto pr-1">
              {filtered.map((p) => (
                <div key={p.id} className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <PaperLink id={p.id} />
                    <span className="text-[13px] text-zinc-200">{p.title}</span>
                    <span className="font-mono text-[10px] text-zinc-500">{p.year}</span>
                  </div>
                  <div className="mt-1.5 flex flex-wrap gap-1">
                    {p.tags.slice(0, 5).map((t) => (
                      <Badge key={t} tone="zinc">{t}</Badge>
                    ))}
                  </div>
                  <p className="mt-2 text-[12px] leading-5 text-zinc-400">{p.summary}</p>
                </div>
              ))}
              {filtered.length === 0 && (
                <div className="rounded-lg border border-rose-900/60 bg-rose-950/20 p-3 text-[12px] text-zinc-400">
                  No papers match this filter.
                </div>
              )}
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}