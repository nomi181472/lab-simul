"use client";

import { useMemo, useState } from "react";
import type { DirectionStatus } from "@/labs/object-detection/data/types";
import { FUTURE_DIRECTIONS } from "@/labs/object-detection/data/directions";
import { Badge, Card, PaperLink, type BadgeTone } from "@/labs/object-detection/ui";

type Filter = "all" | DirectionStatus;

const STATUS_TONE: Record<DirectionStatus, BadgeTone> = {
  realized: "emerald",
  partial: "amber",
  unresolved: "rose",
};

const FILTERS: { value: Filter; label: string }[] = [
  { value: "all", label: "all" },
  { value: "realized", label: "realized" },
  { value: "partial", label: "partial" },
  { value: "unresolved", label: "unresolved" },
];

const PAGE_SIZE = 20;

function clipQuote(q: string) {
  return q.length > 220 ? q.slice(0, 217) + "…" : q;
}

export function DirectionsView() {
  const [filter, setFilter] = useState<Filter>("all");
  const [q, setQ] = useState("");
  const [page, setPage] = useState(0);

  const counts = useMemo(() => {
    const c: Record<DirectionStatus, number> = { realized: 0, partial: 0, unresolved: 0 };
    for (const d of FUTURE_DIRECTIONS) c[d.status]++;
    return c;
  }, []);

  const list = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const filtered = FUTURE_DIRECTIONS.filter((d) => {
      if (filter !== "all" && d.status !== filter) return false;
      if (!needle) return true;
      const hay = `${d.direction} ${d.realizationNote} ${d.quote}`.toLowerCase();
      return needle.split(/\s+/).filter(Boolean).every((t) => hay.includes(t));
    });
    return filtered.sort(
      (a, b) => b.sourceYear - a.sourceYear || a.id.localeCompare(b.id),
    );
  }, [filter, q]);

  const totalPages = Math.max(1, Math.ceil(list.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages - 1);
  const slice = list.slice(safePage * PAGE_SIZE, safePage * PAGE_SIZE + PAGE_SIZE);

  const setFilterReset = (f: Filter) => {
    setFilter(f);
    setPage(0);
  };
  const setQuery = (v: string) => {
    setQ(v);
    setPage(0);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Future tracker</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          {FUTURE_DIRECTIONS.length} directions stated by corpus papers, each checked against
          later work: did the field realize it, partly address it, or leave it open?
        </p>
      </div>

      {/* ---------- stats + filters ---------- */}

      <Card>
        <div className="flex flex-wrap items-center gap-2">
          {FILTERS.map((f) => {
            const n = f.value === "all" ? FUTURE_DIRECTIONS.length : counts[f.value];
            const active = filter === f.value;
            return (
              <button
                key={f.value}
                onClick={() => setFilterReset(f.value)}
                className={`rounded-md border px-2.5 py-1 font-mono text-[10px] uppercase tracking-wide transition-colors ${
                  active
                    ? "border-emerald-600 bg-emerald-600/20 text-emerald-200"
                    : "border-zinc-800 text-zinc-500 hover:text-zinc-300"
                }`}
              >
                {f.label} · {n}
              </button>
            );
          })}
          <label className="ml-auto flex min-w-56 flex-col gap-1">
            <span className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
              search directions
            </span>
            <input
              value={q}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. end-to-end, labels, real-time…"
              className="rounded-md border border-zinc-800 bg-zinc-900 px-2 py-1 text-[12px] text-zinc-200 placeholder:text-zinc-600 focus:border-emerald-600 focus:outline-none"
            />
          </label>
        </div>
        <div className="mt-2 font-mono text-[10px] text-zinc-500">
          {list.length} direction{list.length === 1 ? "" : "s"} match · page {safePage + 1} of{" "}
          {totalPages}
        </div>
      </Card>

      {/* ---------- list ---------- */}

      <div className="space-y-3">
        {slice.map((d) => (
          <Card key={d.id}>
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone={STATUS_TONE[d.status]}>{d.status}</Badge>
                <span className="font-mono text-[10px] text-zinc-500">{d.id}</span>
              </div>
              <div className="flex flex-wrap items-center gap-2 font-mono text-[10px] text-zinc-500">
                source
                <PaperLink id={d.sourcePaperId} />
                <Badge tone="zinc">{d.sourceYear}</Badge>
              </div>
            </div>

            <p className="mt-2 text-[13px] leading-5 text-zinc-200">{d.direction}</p>

            <div className="mt-3 rounded-lg border border-zinc-800 bg-black/30 p-3">
              <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-600">
                stated in the paper{d.location ? ` · ${d.location}` : ""}
              </div>
              <blockquote className="mt-1 border-l-2 border-emerald-800/70 pl-2.5 text-[12px] italic leading-5 text-zinc-400">
                “{clipQuote(d.quote)}”
              </blockquote>
            </div>

            <div className="mt-3 grid gap-3 md:grid-cols-2">
              <div>
                <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-600">
                  what happened next
                </div>
                <p className="mt-1 text-[12px] leading-5 text-zinc-400">{d.realizationNote}</p>
              </div>
              <div className="space-y-2">
                {d.followUps.length > 0 && (
                  <div>
                    <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-600">
                      later work
                    </div>
                    <div className="mt-1 flex flex-wrap gap-1.5">
                      {d.followUps.map((pid) => (
                        <PaperLink key={pid} id={pid} />
                      ))}
                    </div>
                  </div>
                )}
                {d.realizingPaperId && (
                  <div>
                    <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-600">
                      best realization
                    </div>
                    <div className="mt-1 flex flex-wrap items-center gap-1.5">
                      <PaperLink id={d.realizingPaperId} />
                      <Badge tone="emerald">realized</Badge>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </Card>
        ))}
        {slice.length === 0 && (
          <Card tone="warn">
            <p className="text-sm text-zinc-400">No directions match the current filters.</p>
          </Card>
        )}
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-3">
          <button
            onClick={() => setPage(Math.max(0, safePage - 1))}
            disabled={safePage === 0}
            className="rounded-md border border-zinc-800 px-3 py-1 font-mono text-[10px] uppercase tracking-wide text-zinc-400 hover:text-zinc-200 disabled:cursor-not-allowed disabled:opacity-40"
          >
            prev
          </button>
          <span className="font-mono text-[10px] text-zinc-500">
            {safePage + 1} / {totalPages}
          </span>
          <button
            onClick={() => setPage(Math.min(totalPages - 1, safePage + 1))}
            disabled={safePage >= totalPages - 1}
            className="rounded-md border border-zinc-800 px-3 py-1 font-mono text-[10px] uppercase tracking-wide text-zinc-400 hover:text-zinc-200 disabled:cursor-not-allowed disabled:opacity-40"
          >
            next
          </button>
        </div>
      )}
    </div>
  );
}
