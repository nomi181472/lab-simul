"use client";

import { useMemo, useState } from "react";
import { SATURATION } from "@/labs/object-detection/data/saturation";
import { PROBLEM_BY_ID } from "@/labs/object-detection/data/problems";
import type { SaturationEntry } from "@/labs/object-detection/data/types";
import {
  Card,
  SectionTitle,
  Badge,
  StatusBadge,
  EvidenceLine,
  KV,
  type BadgeTone,
} from "@/labs/object-detection/ui";

type SatClass = SaturationEntry["saturation"];
type SortId = "intensity" | "papers" | "year";

const SAT_TONE: Record<SatClass, BadgeTone> = {
  saturated: "rose",
  active: "emerald",
  dormant: "zinc",
  emerging: "violet",
};

const SAT_BAR: Record<SatClass, string> = {
  saturated: "bg-rose-500",
  active: "bg-emerald-500",
  dormant: "bg-zinc-500",
  emerging: "bg-violet-500",
};

const CLASSES: SatClass[] = ["saturated", "active", "dormant", "emerging"];
const SORTS: { id: SortId; label: string }[] = [
  { id: "intensity", label: "intensity ↓" },
  { id: "papers", label: "papers ↓" },
  { id: "year", label: "first year ↑" },
];

const PAGE_SIZE = 12;

function chip(active: boolean) {
  return `rounded-md border px-2 py-1 font-mono text-[10px] transition-colors ${
    active
      ? "border-emerald-600 bg-emerald-600/20 text-emerald-200"
      : "border-zinc-800 text-zinc-500 hover:text-zinc-300"
  }`;
}

function EntryCard({ e, maxIntensity }: { e: SaturationEntry; maxIntensity: number }) {
  const problem = PROBLEM_BY_ID[e.problemId];
  const label = problem?.shortLabel ?? e.problemId;
  return (
    <Card>
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-[13px] font-medium text-zinc-100">{label}</h3>
        <span className="font-mono text-[10px] text-zinc-600">{e.problemId}</span>
        <StatusBadge status={e.status} />
        <Badge tone={SAT_TONE[e.saturation]}>{e.saturation}</Badge>
      </div>

      <div className="mt-3 flex items-center gap-2">
        <span className="w-20 shrink-0 font-mono text-[10px] uppercase tracking-wide text-zinc-500">
          intensity
        </span>
        <div className="h-2.5 flex-1 overflow-hidden rounded-sm bg-zinc-800">
          <div
            className={`h-full ${SAT_BAR[e.saturation]}`}
            style={{ width: `${(e.intensity / maxIntensity) * 100}%` }}
          />
        </div>
        <span className="w-12 shrink-0 text-right font-mono text-[11px] text-zinc-300">
          {e.intensity.toFixed(2)}
        </span>
      </div>

      <div className="mt-3 grid gap-1.5 sm:grid-cols-3">
        <KV k="papers" v={e.paperCount} />
        <KV k="attempts" v={e.attemptCount} />
        <KV k="first year" v={e.firstYear} />
      </div>

      <div className="mt-3 rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
        <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
          remaining gap
        </div>
        <p className="mt-1 text-[12px] leading-5 text-zinc-400">{e.remainingGap}</p>
      </div>

      {e.evidence.length > 0 && (
        <div className="mt-3 space-y-2">
          <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            evidence
          </div>
          {e.evidence.map((ev, i) => (
            <EvidenceLine key={i} evidence={ev} />
          ))}
        </div>
      )}
    </Card>
  );
}

export function SaturationView() {
  const [filter, setFilter] = useState<SatClass | "all">("all");
  const [sort, setSort] = useState<SortId>("intensity");
  const [page, setPage] = useState(0);

  const maxIntensity = useMemo(
    () => Math.max(1, ...SATURATION.map((e) => e.intensity)),
    [],
  );

  const counts = useMemo(() => {
    const c: Record<SatClass, number> = { saturated: 0, active: 0, dormant: 0, emerging: 0 };
    for (const e of SATURATION) c[e.saturation]++;
    return c;
  }, []);

  const list = useMemo(() => {
    const out = filter === "all" ? [...SATURATION] : SATURATION.filter((e) => e.saturation === filter);
    out.sort((a, b) => {
      if (sort === "intensity") return b.intensity - a.intensity || b.paperCount - a.paperCount;
      if (sort === "papers") return b.paperCount - a.paperCount || b.intensity - a.intensity;
      return a.firstYear - b.firstYear || b.intensity - a.intensity;
    });
    return out;
  }, [filter, sort]);

  const pageCount = Math.max(1, Math.ceil(list.length / PAGE_SIZE));
  const cur = Math.min(page, pageCount - 1);
  const slice = list.slice(cur * PAGE_SIZE, cur * PAGE_SIZE + PAGE_SIZE);

  const pick = (f: SatClass | "all") => {
    setFilter(f);
    setPage(0);
  };
  const pickSort = (s: SortId) => {
    setSort(s);
    setPage(0);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">
          Saturation — attention vs progress
        </h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          {SATURATION.length} problems scored by how densely papers chase them. A saturated
          problem attracts attempts every year without closing; a dormant one was hit hard
          then left alone.
        </p>
      </div>

      {/* header stats + intensity callout */}
      <div className="grid gap-3 md:grid-cols-2">
        <Card>
          <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            entries per saturation class
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {CLASSES.map((c) => (
              <div
                key={c}
                className="rounded-lg border border-zinc-800 bg-zinc-900/40 px-3 py-2 text-center"
              >
                <div className="font-mono text-lg text-zinc-100">{counts[c]}</div>
                <Badge tone={SAT_TONE[c]}>{c}</Badge>
              </div>
            ))}
          </div>
        </Card>
        <Card tone="accent">
          <div className="font-mono text-[10px] uppercase tracking-wide text-emerald-400">
            how to read intensity
          </div>
          <p className="mt-2 text-[12px] leading-5 text-zinc-300">
            intensity = attempts (papers) per active year — problem papers ÷ years since it
            first appeared. It measures attention, not progress: a high value with a
            &quot;persistent&quot; status means many attempts and an unclosed gap.
          </p>
          <p className="mt-2 font-mono text-[10px] text-zinc-500">
            scale: 0 – {maxIntensity.toFixed(2)} across all {SATURATION.length} entries
          </p>
        </Card>
      </div>

      {/* controls */}
      <Card>
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="mr-1 font-mono text-[10px] uppercase tracking-wide text-zinc-500">
              class
            </span>
            <button className={chip(filter === "all")} onClick={() => pick("all")}>
              all · {SATURATION.length}
            </button>
            {CLASSES.map((c) => (
              <button key={c} className={chip(filter === c)} onClick={() => pick(c)}>
                {c} · {counts[c]}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="mr-1 font-mono text-[10px] uppercase tracking-wide text-zinc-500">
              sort by
            </span>
            {SORTS.map((s) => (
              <button key={s.id} className={chip(sort === s.id)} onClick={() => pickSort(s.id)}>
                {s.label}
              </button>
            ))}
          </div>
        </div>
      </Card>

      {/* entries */}
      <SectionTitle>Problems</SectionTitle>
      <div className="space-y-3">
        {slice.map((e) => (
          <EntryCard key={e.problemId} e={e} maxIntensity={maxIntensity} />
        ))}
        {slice.length === 0 && (
          <div className="rounded-lg border border-rose-900/60 bg-rose-950/20 p-3 text-[12px] text-zinc-400">
            No entries in this saturation class.
          </div>
        )}
      </div>

      {pageCount > 1 && (
        <div className="flex items-center justify-center gap-3">
          <button
            onClick={() => setPage(cur - 1)}
            disabled={cur === 0}
            className="rounded-md border border-zinc-800 px-3 py-1 font-mono text-[10px] text-zinc-400 disabled:opacity-40 hover:text-zinc-200"
          >
            ← prev
          </button>
          <span className="font-mono text-[10px] text-zinc-500">
            page {cur + 1} / {pageCount} · {list.length} entries
          </span>
          <button
            onClick={() => setPage(cur + 1)}
            disabled={cur >= pageCount - 1}
            className="rounded-md border border-zinc-800 px-3 py-1 font-mono text-[10px] text-zinc-400 disabled:opacity-40 hover:text-zinc-200"
          >
            next →
          </button>
        </div>
      )}
    </div>
  );
}
